"""Dispatch and combine same-surface F12 energy and analytic-force increments."""

from __future__ import annotations

import json
import math
import typing
from dataclasses import dataclass, replace
from functools import cached_property
from os import PathLike
from typing import Generic, Literal, TypeVar

import numpy as np

from .. import _native
from .._api_types import Atom
from .._cbs_plan import native_cbs_plan
from .._model_resolution import snapshot_basis
from ..basis import BasisSet
from ..elements import checked_integer, electron_state
from ..profiles import canonical_hash

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

BasisInput = BasisSet | str | PathLike[str]
Role = Literal[
    "hf_cabs", "mp2_f12_correlation", "post_mp2", "ccsd_minus_mp2", "triples"
]
Backend = Literal["cuda", "cpu_reference"]


@dataclass(frozen=True)
class FocalTerm:
    """One center from a fixed protocol, energy in Eh and force in Eh/bohr.

    surface_id identifies ordered nuclei, coordinates, Q/M, electronic root,
    pseudopotentials and the correlated-electron convention. series_id identifies
    a basis family and auxiliary policy across cardinal numbers. basis_id
    identifies the actual orbital/auxiliary files used for this term.
    """

    surface_id: str
    series_id: str
    basis_id: str
    cardinal: int
    role: Role
    energy: float
    forces: np.ndarray


@dataclass(frozen=True, init=False)
class FocalBasis:
    """Owned canonical basis records for the separate fitted operator roles.

    Canonical files are loaded once. Bundled names use real spherical shells.
    Declare the correlation and F12 fitting records separately, even when
    they are the same record. For F12 pair correlation, declare beta in
    inverse Bohr at each cardinal.
    """

    cardinal: int
    series_id: str
    orbital: BasisSet
    jk_fit: BasisSet
    mp2_fit: BasisSet | None
    f12_fit: BasisSet | None
    cabs: BasisSet | None
    beta: float | None

    def __init__(
        self,
        cardinal: int,
        series_id: str,
        orbital: BasisInput,
        *,
        jk_fit: BasisInput,
        mp2_fit: BasisInput | None = None,
        f12_fit: BasisInput | None = None,
        cabs: BasisInput | None = None,
        beta: float | None = None,
    ) -> None:
        object.__setattr__(
            self, "cardinal", checked_integer(cardinal, "cardinal", low=2, high=100)
        )
        if not isinstance(series_id, str) or not series_id.strip():
            raise ValueError("a focal basis requires a series identity")
        object.__setattr__(self, "series_id", series_id)
        if beta is not None and (
            isinstance(beta, bool) or not math.isfinite(beta) or beta <= 0
        ):
            raise ValueError("F12 beta must be finite and positive")
        object.__setattr__(self, "beta", float(beta) if beta is not None else None)
        for name, value in (
            ("orbital", orbital),
            ("jk_fit", jk_fit),
            ("mp2_fit", mp2_fit),
            ("f12_fit", f12_fit),
            ("cabs", cabs),
        ):
            if value is None and name not in ("orbital", "jk_fit"):
                object.__setattr__(self, name, None)
                continue
            if not isinstance(value, (BasisSet, str, PathLike)):
                raise TypeError(
                    f"{name} requires a bundled name or a canonical basis record"
                )
            representation = (
                "spherical"
                if isinstance(value, str) and not value.endswith(".json")
                else None
            )
            record = snapshot_basis(value, representation)
            if not isinstance(record, BasisSet):
                raise TypeError(f"{name} requires a canonical basis record")
            object.__setattr__(self, name, record)

    @cached_property
    def basis_id(self) -> str:
        """Identify the resolved basis roles and the associated F12 exponent."""
        roles = {
            role: record.identity if record is not None else None
            for role, record in (
                ("orbital", self.orbital),
                ("jk_fit", self.jk_fit),
                ("mp2_fit", self.mp2_fit),
                ("f12_fit", self.f12_fit),
                ("cabs", self.cabs),
            )
        }
        return canonical_hash({**roles, "f12_beta": self.beta})


_Endpoint = TypeVar("_Endpoint", FocalTerm, FocalBasis)


@dataclass(frozen=True)
class FocalSeries(Generic[_Endpoint]):
    """A finite endpoint or a pair with a fixed exponent and decay law."""

    upper: _Endpoint
    lower: _Endpoint | None = None
    power: float | None = None
    extrapolation: Literal["inverse_power", "exponential"] = "inverse_power"


@dataclass(frozen=True)
class FocalResult:
    """The composed energy and forces and their actual per-role input weights."""

    energy: float
    forces: np.ndarray
    weights: tuple[tuple[Role, str, int, float], ...]


@dataclass(frozen=True)
class FocalRequest:
    """One role/cardinal requested from a component provider."""

    atoms: tuple[Atom, ...]
    charge: int
    multiplicity: int
    root_id: str
    correlation_space_id: str
    surface_id: str
    basis: FocalBasis
    role: Role
    backend: Backend


@dataclass(frozen=True)
class FocalEvaluation:
    """Component output with the convergence and backend reported by its provider."""

    term: FocalTerm
    executed_backend: Backend
    converged: bool


@dataclass(frozen=True)
class FocalRunResult(FocalResult):
    """Composed result with the completed component outputs in weight order."""

    components: tuple[FocalEvaluation, ...]


def conventional_df_post_mp2(request: FocalRequest) -> FocalEvaluation:
    """Evaluate combined DF-CCSD(T) minus DF-MP2 on one native reference.

    Supply canonical cold-start root ``rhf`` or ``uhf`` and correlation space
    ``all-active``. The current grouped native interface shares one auxiliary
    record across the SCF and correlation operators. Its return contains
    the full CCSD(T) force, so this provider supplies the combined increment.
    The other focal components must use the matching electronic root.
    """
    if request.role != "post_mp2":
        raise ValueError("the native grouped correction only returns the post_mp2 role")
    if (
        request.root_id not in ("rhf", "uhf")
        or request.correlation_space_id != "all-active"
    ):
        raise ValueError(
            "native post_mp2 requires an rhf or uhf root and all-active correlation"
        )
    basis = request.basis
    if basis.mp2_fit is None or basis.jk_fit.identity != basis.mp2_fit.identity:
        raise ValueError("native post_mp2 requires the same JKFit and MP2Fit record")
    if request.backend not in ("cuda", "cpu_reference"):
        raise ValueError("focal backend must be cuda or cpu_reference")

    from ..calculator import Calculator

    unrestricted = request.root_id == "uhf"
    method, companion = ("uccsd(t)", "ump2") if unrestricted else ("df-rccsd(t)", "mp2")
    calculator = Calculator(
        method=method,
        device="cuda" if request.backend == "cuda" else "cpu",
        basis=basis.orbital,
        auxiliary_basis=basis.mp2_fit,
        density_fitting="full_cuda",
        initial_guess=None,
    )
    try:
        results = calculator.singlepoint_components(
            request.atoms,
            companions=(companion,),
            charge=request.charge,
            multiplicity=request.multiplicity,
            properties=("energy", "forces"),
        )
    finally:
        calculator.clear_cache()
    if set(results) != {method, companion}:
        raise RuntimeError("native post_mp2 did not return both grouped components")
    high, mp2 = results[method], results[companion]
    shape = (len(request.atoms), 3)
    for result in (high, mp2):
        if not result.converged or result.executed_backend != request.backend:
            raise RuntimeError(
                "native post_mp2 did not converge on the requested backend"
            )
        if (
            result.forces is None
            or result.forces.shape != shape
            or not np.isfinite(result.forces).all()
            or not math.isfinite(result.energy)
        ):
            raise RuntimeError(
                "native post_mp2 requires complete finite energy and forces"
            )
    if (
        high.correlation is None
        or mp2.correlation is None
        or high.correlation.reference_energy != mp2.correlation.reference_energy
    ):
        raise RuntimeError("native post_mp2 components must share one reference energy")
    for name, expected in (
        ("orbital", basis.orbital.identity),
        ("auxiliary", basis.mp2_fit.identity),
    ):
        if high.basis_metadata.get(name, {}).get("basis_identity") != expected:
            raise RuntimeError(f"native post_mp2 used a different {name} basis")
    assert high.forces is not None and mp2.forces is not None
    term = FocalTerm(
        request.surface_id,
        basis.series_id,
        basis.basis_id,
        basis.cardinal,
        request.role,
        high.energy - mp2.energy,
        high.forces - mp2.forces,
    )
    return FocalEvaluation(
        term, typing.cast("Backend", high.executed_backend), converged=True
    )


def _weighted_terms(
    hf_cabs: FocalSeries[_Endpoint],
    correlation: FocalSeries[_Endpoint],
    post_mp2: _Endpoint | FocalSeries[_Endpoint],
    triples: _Endpoint | FocalSeries[_Endpoint] | None,
) -> list[tuple[Role, _Endpoint, float]]:
    high = post_mp2 if isinstance(post_mp2, FocalSeries) else FocalSeries(post_mp2)
    roles: list[tuple[FocalSeries[_Endpoint], Role]] = [
        (hf_cabs, "hf_cabs"),
        (correlation, "mp2_f12_correlation"),
        (high, "post_mp2" if triples is None else "ccsd_minus_mp2"),
    ]
    if triples is not None:
        triple_series = (
            triples if isinstance(triples, FocalSeries) else FocalSeries(triples)
        )
        roles.append((triple_series, "triples"))
    weighted = []
    for series, role in roles:
        upper, lower = series.upper, series.lower
        if series.extrapolation not in ("inverse_power", "exponential"):
            raise ValueError("unknown extrapolation law")
        if series.extrapolation == "exponential" and (
            role != "hf_cabs" or lower is None
        ):
            raise ValueError(
                "exponential extrapolation requires an HF plus CABS cardinal pair"
            )
        if lower is None:
            if series.power is not None:
                raise ValueError("a finite endpoint has no extrapolation power")
            weighted.append((role, upper, 1.0))
            continue
        if (
            isinstance(series.power, bool)
            or not isinstance(series.power, (int, float))
            or not math.isfinite(series.power)
            or series.power <= 0
        ):
            raise ValueError("a cardinal pair requires a finite positive power")
        if lower.series_id != upper.series_id or lower.basis_id == upper.basis_id:
            raise ValueError(
                "a cardinal pair requires distinct bases from the same series"
            )
        if (
            isinstance(lower, FocalBasis)
            and isinstance(upper, FocalBasis)
            and lower.orbital.identity == upper.orbital.identity
        ):
            raise ValueError("a cardinal pair requires distinct orbital bases")
        lo = checked_integer(lower.cardinal, "lower cardinal", low=2, high=100)
        hi = checked_integer(upper.cardinal, "upper cardinal", low=2, high=100)
        if lo >= hi:
            raise ValueError("cardinal numbers must increase")
        if series.extrapolation == "exponential":
            weights = native_cbs_plan(
                _native.load_library(), lo, hi, 1.0, series.power, "exponential"
            ).hf_weights
        else:
            weights = native_cbs_plan(
                _native.load_library(), lo, hi, series.power, 1.0, "highest"
            ).mp2_weights
        weighted.extend(((role, lower, weights[0]), (role, upper, weights[1])))
    return weighted


def _values(term: FocalTerm, shape: tuple[int, ...]) -> tuple[float, np.ndarray]:
    value = float(term.energy)
    arr = np.asarray(term.forces)
    if (
        arr.shape != shape
        or not np.issubdtype(arr.dtype, np.floating)
        or not np.isfinite(arr).all()
        or not math.isfinite(value)
    ):
        raise ValueError(
            "energy and Cartesian forces must be finite, with matching shape"
        )
    return value, np.asarray(arr, dtype=np.float64)


def _assemble(weighted: list[tuple[Role, FocalTerm, float]]) -> FocalResult:
    first = weighted[0][1]
    shape = np.shape(first.forces)
    if len(shape) != 2 or shape[0] == 0 or shape[1] != 3:
        raise ValueError("Cartesian forces require shape (atoms, 3)")
    energies = []
    force_terms = []
    inputs = []
    for role, term, coefficient in weighted:
        if term.role != role or term.surface_id != first.surface_id:
            raise ValueError(
                "focal terms must have the requested role and identical surface identity"
            )
        if not all(
            isinstance(v, str) and v.strip()
            for v in (term.surface_id, term.basis_id, term.series_id)
        ):
            raise ValueError("each term requires a surface, basis and series identity")
        checked_integer(term.cardinal, "cardinal", low=2, high=100)
        value, arr = _values(term, shape)
        energies.append(value * coefficient)
        force_terms.append(arr * coefficient)
        inputs.append((term.role, term.basis_id, term.cardinal, coefficient))
    f = np.sum(force_terms, axis=0)
    e = math.fsum(energies)
    if not math.isfinite(e) or not np.isfinite(f).all():
        raise ValueError("combined energy or force is not finite")
    return FocalResult(e, f, tuple(inputs))


def combine(
    hf_cabs: FocalSeries[FocalTerm],
    correlation: FocalSeries[FocalTerm],
    post_mp2: FocalTerm | FocalSeries[FocalTerm],
    *,
    triples: FocalTerm | FocalSeries[FocalTerm] | None = None,
) -> FocalResult:
    """Apply fixed coefficients to both E and -dE/dR after the electronic solves.

    Correlation includes ordinary MP2 correlation plus the explicit F12 pair term.
    Supply a combined post_mp2 increment, or supply a ccsd_minus_mp2
    increment together with separate conventional triples. Each increment can
    have its own basis series. CCSD and MP2 must be differenced at matching
    basis and correlation space for each cardinal in their increment.
    Exponential extrapolation is available for the HF plus CABS series.
    All energy and force terms must describe the same ordered geometry.
    """
    return _assemble(_weighted_terms(hf_cabs, correlation, post_mp2, triples))


def _hamiltonian(
    weighted: list[tuple[Role, FocalBasis, float]],
    geometry: tuple[Atom, ...],
    charge: int,
    multiplicity: int,
) -> tuple[tuple[int, int, str | None], ...]:
    nuclei = {atom.atomic_number for atom in geometry}
    reference = None
    for role, basis, _ in weighted:
        if not isinstance(basis, FocalBasis):
            raise TypeError("singlepoint requires FocalBasis inputs")
        if role == "mp2_f12_correlation" and basis.beta is None:
            raise ValueError("F12 pair correlation requires an explicit beta")
        for name, record, required in (
            ("orbital", basis.orbital, True),
            ("JKFit", basis.jk_fit, True),
            ("MP2Fit", basis.mp2_fit, role != "hf_cabs"),
            ("F12 fit", basis.f12_fit, role == "mp2_f12_correlation"),
            ("CABS", basis.cabs, role in ("hf_cabs", "mp2_f12_correlation")),
        ):
            if required and record is None:
                raise ValueError(f"{role} requires an explicit {name} basis")
            if record is not None and nuclei - record.by_element.keys():
                raise ValueError(
                    f"{name} basis {record.name!r} lacks a nucleus in the geometry"
                )
        orbital = basis.orbital.by_element
        state = electron_state(
            geometry, charge=charge, multiplicity=multiplicity, element_metadata=orbital
        )
        hamiltonian = tuple(
            (z, core, canonical_hash(json.loads(data)) if data is not None else None)
            for z, core, data in zip(
                state.nuclear_charges,
                state.ecp_core_electrons,
                (orbital[atom.atomic_number].ecp_data for atom in geometry),
                strict=True,
            )
        )
        if reference is not None and hamiltonian != reference:
            raise ValueError(
                "all focal orbital bases must use the same ECP Hamiltonian"
            )
        reference = hamiltonian
    assert reference is not None
    return reference


def singlepoint(
    atoms: Iterable[Atom | tuple[str | int, Sequence[float]]],
    hf_cabs: FocalSeries[FocalBasis],
    correlation: FocalSeries[FocalBasis],
    post_mp2: FocalBasis | FocalSeries[FocalBasis],
    *,
    provider: Callable[[FocalRequest], FocalEvaluation],
    root_id: str,
    correlation_space_id: str,
    charge: int = 0,
    multiplicity: int = 1,
    backend: Backend = "cuda",
    triples: FocalBasis | FocalSeries[FocalBasis] | None = None,
) -> FocalRunResult:
    """Evaluate a geometry with an explicit electronic component provider.

    Positions are in Bohr. The provider evaluates each requested role and
    returns matched, converged energy (Eh) and analytic forces (Eh/Bohr).
    It must echo the request identities and report the executed backend.
    The caller declares the root and correlated space shared by all roles.
    Providers own electronic execution and method support. The dispatcher
    raises on a mismatched result or backend and never retries on another backend.
    """
    geometry = tuple(Atom.from_value(atom) for atom in atoms)
    if not geometry:
        raise ValueError("at least one atom is required")
    charge = checked_integer(charge, "ionic charge", low=-(2**31))
    multiplicity = checked_integer(multiplicity, "multiplicity", low=1)
    if not all(
        isinstance(x, str) and x.strip() for x in (root_id, correlation_space_id)
    ):
        raise ValueError(
            "a focal calculation requires root and correlation-space identities"
        )
    if backend not in ("cuda", "cpu_reference"):
        raise ValueError("focal backend must be cuda or cpu_reference")
    if not callable(provider):
        raise TypeError("a focal component provider is required")

    weighted = _weighted_terms(hf_cabs, correlation, post_mp2, triples)
    hamiltonian = _hamiltonian(weighted, geometry, charge, multiplicity)
    surface_id = canonical_hash(
        {
            "geometry_bohr": [
                (
                    atom.atomic_number,
                    tuple((v if v else 0.0).hex() for v in atom.position),
                )
                for atom in geometry
            ],
            "charge": charge,
            "multiplicity": multiplicity,
            "hamiltonian": hamiltonian,
            "electronic_root": root_id,
            "correlation_space": correlation_space_id,
        }
    )
    components = []
    computed = []
    for role, basis, weight in weighted:
        request = FocalRequest(
            geometry,
            charge,
            multiplicity,
            root_id,
            correlation_space_id,
            surface_id,
            basis,
            role,
            backend,
        )
        evaluation = provider(request)
        if not isinstance(evaluation, FocalEvaluation) or not isinstance(
            evaluation.term, FocalTerm
        ):
            raise TypeError("provider must return a FocalEvaluation with a FocalTerm")
        if evaluation.executed_backend != backend:
            raise RuntimeError(f"{role} returned a different execution backend")
        if evaluation.converged is not True:
            raise RuntimeError(f"{role} did not converge")
        term = evaluation.term
        if (
            term.surface_id != surface_id
            or term.basis_id != basis.basis_id
            or term.series_id != basis.series_id
            or term.role != role
            or term.cardinal != basis.cardinal
        ):
            raise ValueError(
                "component result does not match its requested surface, role and basis"
            )
        value, forces = _values(term, (len(geometry), 3))
        # Snapshot provider storage before the next call can reuse its workspace.
        term = replace(term, energy=value, forces=forces.copy())
        components.append(replace(evaluation, term=term))
        computed.append((role, term, weight))
    result = _assemble(computed)
    return FocalRunResult(
        result.energy, result.forces, result.weights, tuple(components)
    )
