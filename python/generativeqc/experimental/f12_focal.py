"""Dispatch and combine same-surface F12 energy and analytic-force increments."""

from __future__ import annotations

import json
import math
import typing
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Generic, Literal, TypeVar

import numpy as np

from .. import (
    _f12_focal_increment,
    _f12_focal_molecular,
    _f12_focal_one_electron,
    _f12_focal_projector,
    _f12_focal_reference,
    _f12_focal_singles,
    _f12_focal_ucc_triples,
    _f12_focal_ump2,
    _native,
    _uccsd_rhf_virtual_pairs,
    _uccsd_uhf_virtual_ladder,
)
from .._api_types import Atom
from .._cbs_plan import native_cbs_plan
from .._model_resolution import snapshot_basis
from ..basis import BasisProvenance, BasisSet
from ..elements import SYMBOLS, atomic_number, checked_integer, electron_state
from ..profiles import canonical_hash

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from .._api_types import Result

BasisRecordInput = BasisSet | str | PathLike[str]
BasisInput = BasisRecordInput | Mapping[str | int, BasisRecordInput]
Role = Literal[
    "hf_cabs",
    "mp2_f12_correlation",
    "post_mp2",
    "ccsd_minus_mp2",
    "ccsd_f12b_minus_mp2_f12",
    "triples",
]
Backend = Literal["cuda", "cpu_reference"]
MolecularFocalProvider = _f12_focal_molecular.MolecularFocalProvider
RHFDFLadderOptions = _uccsd_rhf_virtual_pairs.RHFDFLadderOptions
UHFDFLadderOptions = _uccsd_uhf_virtual_ladder.UHFDFLadderOptions
FocalLowOrderForces = _f12_focal_molecular.FocalLowOrderForces
FocalLowOrderEnergy = _f12_focal_molecular.FocalLowOrderEnergy
FocalUCCEquationSources = _f12_focal_molecular.FocalUCCEquationSources
FocalUCCFixedSources = _f12_focal_molecular.FocalUCCFixedSources
FocalUCCStationary = _f12_focal_molecular.FocalUCCStationary
FocalUCCSDResult = _f12_focal_molecular.FocalUCCSDResult
conventional_df_ucc_triples = _f12_focal_ucc_triples.conventional_df_ucc_triples
FocalOneElectron = _f12_focal_one_electron.FocalOneElectron
one_electron_sources = _f12_focal_one_electron.one_electron_sources
FocalReference = _f12_focal_reference.FocalReference
FocalReferenceSeed = _f12_focal_reference.FocalReferenceSeed
df_reference = _f12_focal_reference.df_reference
FocalF12Increment = _f12_focal_increment.FocalF12Increment
FocalF12Sources = _f12_focal_increment.FocalF12Sources
reference_f12_increment = _f12_focal_increment.reference_f12_increment
focal_cabs_projector = _f12_focal_projector.focal_cabs_projector
FocalCabsSingles = _f12_focal_singles.FocalCabsSingles
reference_cabs_singles = _f12_focal_singles.reference_cabs_singles
FocalCoulombSource = _f12_focal_ump2.FocalCoulombSource
FocalUMP2Weights = _f12_focal_ump2.FocalUMP2Weights
reference_df_ump2 = _f12_focal_ump2.reference_df_ump2


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


def _role_basis(role: str, value: BasisInput, files: dict[Path, BasisSet]) -> BasisSet:
    def record(source: BasisRecordInput) -> BasisSet:
        if isinstance(source, BasisSet):
            return source
        if isinstance(source, str) and not source.endswith(".json"):
            return snapshot_basis(source, "spherical")
        if isinstance(source, (str, PathLike)):
            path = Path(source).resolve()
            if path not in files:
                files[path] = snapshot_basis(path)
            return files[path]
        raise TypeError(f"{role} requires a bundled name or a canonical basis record")

    if not isinstance(value, Mapping):
        return record(value)
    if not value:
        raise ValueError(f"{role} requires at least one element assignment")
    sources: dict[int, BasisSet] = {}
    for key, source in value.items():
        z = atomic_number(key)
        if z in sources:
            raise ValueError(f"{role} has duplicate assignments for {SYMBOLS[z - 1]}")
        basis = record(source)
        if z not in basis.by_element:
            raise ValueError(f"{role} basis {basis.name!r} lacks {SYMBOLS[z - 1]}")
        sources[z] = basis
    representations = {basis.representation for basis in sources.values()}
    if len(representations) != 1:
        raise ValueError(f"{role} element assignments must have one AO representation")
    selections = sorted(sources.items())
    manifest = [
        {"element": z, "name": b.name, "basis_id": b.identity, **asdict(b.provenance)}
        for z, b in selections
    ]
    return BasisSet(
        name=", ".join(f"{SYMBOLS[z - 1]}={b.name}" for z, b in selections),
        elements=tuple(b.by_element[z] for z, b in selections),
        provenance=BasisProvenance(
            source=json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            version="element-selection-v1",
            license=", ".join(
                f"{SYMBOLS[z - 1]}={b.provenance.license}" for z, b in selections
            ),
            checksum=canonical_hash(manifest),
        ),
        representation=representations.pop(),
    )


@dataclass(frozen=True, init=False)
class FocalBasis:
    """Owned canonical basis records for the separate fitted operator roles.

    Give each role a bundled name, canonical record/file, or a mapping from
    element symbol or atomic number to those inputs. Each selection keeps the
    source's shells and ECP. All selections for one role share an AO
    representation. Canonical files are loaded once per FocalBasis and
    bundled names use real spherical shells.
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
        files: dict[Path, BasisSet] = {}
        for name, value in (
            ("orbital", orbital),
            ("jk_fit", jk_fit),
            ("mp2_fit", mp2_fit),
            ("f12_fit", f12_fit),
            ("cabs", cabs),
        ):
            if value is None:
                if name in ("orbital", "jk_fit"):
                    raise TypeError(
                        f"{name} requires a bundled name or a canonical basis record"
                    )
                object.__setattr__(self, name, None)
            else:
                object.__setattr__(self, name, _role_basis(name, value, files))

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


@dataclass(frozen=True)
class FocalMolecularResult(FocalRunResult):
    """Analytic center with basis-specific determinants and complementary-space ranks."""

    references: dict[str, FocalReferenceSeed]
    cabs_ranks: dict[str, int]


@dataclass(frozen=True)
class FocalEnergyComponent:
    """A converged forward-only molecular energy and its recipe coefficient."""

    request: FocalRequest
    energy: float
    weight: float


@dataclass(frozen=True)
class FocalEnergyResult:
    """Composed energy in Hartree and its actual forward-only component values."""

    energy: float
    components: tuple[FocalEnergyComponent, ...]
    references: dict[str, FocalReferenceSeed]
    cabs_ranks: dict[str, int]


def _native_post_mp2_components(
    request: FocalRequest, *, with_forces: bool
) -> tuple[Result, Result]:
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
            properties=("energy", "forces") if with_forces else ("energy",),
        )
    finally:
        calculator.clear_cache()
    if set(results) != {method, companion}:
        raise RuntimeError("native post_mp2 did not return both grouped components")
    high, mp2 = results[method], results[companion]
    for result in (high, mp2):
        if not result.converged or result.executed_backend != request.backend:
            raise RuntimeError(
                "native post_mp2 did not converge on the requested backend"
            )
        if not math.isfinite(result.energy):
            raise RuntimeError("native post_mp2 requires finite component energies")
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
    return high, mp2


def conventional_df_post_mp2_energy(request: FocalRequest) -> float:
    """Compute the conventional increment in Eh for one displaced geometry."""
    high, mp2 = _native_post_mp2_components(request, with_forces=False)
    return high.energy - mp2.energy


def conventional_df_post_mp2(request: FocalRequest) -> FocalEvaluation:
    """Evaluate combined DF-CCSD(T) minus DF-MP2 on one native reference.

    Supply canonical cold-start root ``rhf`` or ``uhf`` and correlation space
    ``all-active``. The grouped native interface shares one auxiliary record
    across the SCF and correlation operators. It returns the full CCSD(T)
    force. The other focal components must use the matching electronic root.
    """
    high, mp2 = _native_post_mp2_components(request, with_forces=True)
    for result in (high, mp2):
        if (
            result.forces is None
            or result.forces.shape != (len(request.atoms), 3)
            or not np.isfinite(result.forces).all()
        ):
            raise RuntimeError("native post_mp2 requires complete finite forces")
    assert high.forces is not None and mp2.forces is not None
    term = FocalTerm(
        request.surface_id,
        request.basis.series_id,
        request.basis.basis_id,
        request.basis.cardinal,
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
    post_mp2: _Endpoint | FocalSeries[_Endpoint] | None,
    triples: _Endpoint | FocalSeries[_Endpoint] | None,
    *,
    f12_doubles: _Endpoint | FocalSeries[_Endpoint] | None = None,
    nuclei: set[int] | None = None,
) -> list[tuple[Role, _Endpoint, float]]:
    roles: list[tuple[FocalSeries[_Endpoint], Role]] = [
        (hf_cabs, "hf_cabs"),
        (correlation, "mp2_f12_correlation"),
    ]
    if f12_doubles is not None:
        if post_mp2 is not None:
            raise ValueError("choose either a conventional or an F12 doubles increment")
        series = (
            f12_doubles
            if isinstance(f12_doubles, FocalSeries)
            else FocalSeries(f12_doubles)
        )
        roles.append((series, "ccsd_f12b_minus_mp2_f12"))
    if post_mp2 is not None:
        high = post_mp2 if isinstance(post_mp2, FocalSeries) else FocalSeries(post_mp2)
        roles.append((high, "post_mp2" if triples is None else "ccsd_minus_mp2"))
    if triples is not None:
        if post_mp2 is None and f12_doubles is None:
            raise ValueError(
                "separate triples require a ccsd_minus_mp2 increment or F12 doubles"
            )
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
        if isinstance(lower, FocalBasis) and isinstance(upper, FocalBasis):
            lower_elements = lower.orbital.by_element
            upper_elements = upper.orbital.by_element
            selected = (
                nuclei
                if nuclei is not None
                else lower_elements.keys() | upper_elements.keys()
            )
            if lower.orbital.representation == upper.orbital.representation and all(
                z in lower_elements and lower_elements[z] == upper_elements.get(z)
                for z in selected
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
    post_mp2: FocalTerm | FocalSeries[FocalTerm] | None,
    *,
    triples: FocalTerm | FocalSeries[FocalTerm] | None = None,
    f12_doubles: FocalTerm | FocalSeries[FocalTerm] | None = None,
) -> FocalResult:
    """Apply fixed coefficients to both E and -dE/dR after the electronic solves.

    Correlation includes ordinary MP2 correlation plus the explicit F12 pair term.
    Set post_mp2=None for HF+CABS plus MP2-F12. Supply f12_doubles to add
    CCSD-F12b minus MP2-F12, optionally followed by conventional triples.
    Supply a combined post_mp2 increment, or supply a ccsd_minus_mp2
    increment together with separate conventional triples. Each increment can
    have its own basis series. CCSD and MP2 must be differenced at matching
    basis and correlation space for each cardinal in their increment.
    Exponential extrapolation is available for the HF plus CABS series.
    All energy and force terms must describe the same ordered geometry.
    """
    return _assemble(
        _weighted_terms(
            hf_cabs, correlation, post_mp2, triples, f12_doubles=f12_doubles
        )
    )


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
        f12_pair = role in ("mp2_f12_correlation", "ccsd_f12b_minus_mp2_f12")
        if f12_pair and basis.beta is None:
            raise ValueError("F12 pair correlation requires an explicit beta")
        for name, record, required in (
            ("orbital", basis.orbital, True),
            ("JKFit", basis.jk_fit, True),
            ("MP2Fit", basis.mp2_fit, role != "hf_cabs"),
            ("F12 fit", basis.f12_fit, f12_pair),
            ("CABS", basis.cabs, role == "hf_cabs" or f12_pair),
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


def plan_singlepoint(
    atoms: Iterable[Atom | tuple[str | int, Sequence[float]]],
    hf_cabs: FocalSeries[FocalBasis],
    correlation: FocalSeries[FocalBasis],
    post_mp2: FocalBasis | FocalSeries[FocalBasis] | None,
    *,
    root_id: str,
    correlation_space_id: str,
    charge: int = 0,
    multiplicity: int = 1,
    backend: Backend = "cuda",
    triples: FocalBasis | FocalSeries[FocalBasis] | None = None,
    f12_doubles: FocalBasis | FocalSeries[FocalBasis] | None = None,
    protocol_id: str | None = None,
) -> FocalPlan:
    """Resolve one geometry and basis recipe into independently evaluable requests.

    Positions are in Bohr. Each request specifies its basis records, electronic
    surface, component role and backend. For independently archived returns,
    pass the campaign's immutable method and implementation ID as protocol_id.
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
    if protocol_id is not None and (
        not isinstance(protocol_id, str) or not protocol_id.strip()
    ):
        raise ValueError("protocol_id must be a nonempty string when supplied")
    weighted = _weighted_terms(
        hf_cabs,
        correlation,
        post_mp2,
        triples,
        f12_doubles=f12_doubles,
        nuclei={atom.atomic_number for atom in geometry},
    )
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
            **({"protocol_id": protocol_id} if protocol_id is not None else {}),
        }
    )
    steps = []
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
        steps.append((request, weight))
    return FocalPlan(tuple(steps))


def _accepted(request: FocalRequest, evaluation: FocalEvaluation) -> FocalEvaluation:
    if not isinstance(evaluation, FocalEvaluation) or not isinstance(
        evaluation.term, FocalTerm
    ):
        raise TypeError("provider must return a FocalEvaluation with a FocalTerm")
    if evaluation.executed_backend != request.backend:
        raise RuntimeError(f"{request.role} returned a different execution backend")
    if evaluation.converged is not True:
        raise RuntimeError(f"{request.role} did not converge")
    term = evaluation.term
    if (
        term.surface_id != request.surface_id
        or term.basis_id != request.basis.basis_id
        or term.series_id != request.basis.series_id
        or term.role != request.role
        or term.cardinal != request.basis.cardinal
    ):
        raise ValueError(
            "component result does not match its requested surface, role and basis"
        )
    value, forces = _values(term, (len(request.atoms), 3))
    return replace(evaluation, term=replace(term, energy=value, forces=forces.copy()))


@dataclass(frozen=True)
class FocalPlan:
    """Ordered (request, weight) pairs produced by plan_singlepoint()."""

    steps: tuple[tuple[FocalRequest, float], ...]

    def _finish(self, evaluations: tuple[FocalEvaluation, ...]) -> FocalRunResult:
        weighted = [
            (request.role, evaluation.term, weight)
            for (request, weight), evaluation in zip(
                self.steps, evaluations, strict=True
            )
        ]
        result = _assemble(weighted)
        return FocalRunResult(result.energy, result.forces, result.weights, evaluations)

    def collect(self, evaluations: Iterable[FocalEvaluation]) -> FocalRunResult:
        """Validate independently returned components and combine them in recipe order.

        Supply one converged result per role and cardinal from the same method
        and implementation used to plan the campaign.
        """
        requests = {(r.role, r.basis.cardinal): r for r, _ in self.steps}
        received = {}
        for evaluation in evaluations:
            if not isinstance(evaluation, FocalEvaluation) or not isinstance(
                evaluation.term, FocalTerm
            ):
                raise TypeError("expected a FocalEvaluation with a FocalTerm")
            key = evaluation.term.role, evaluation.term.cardinal
            if key not in requests:
                raise ValueError(f"unexpected focal component {key}")
            if key in received:
                raise ValueError(f"duplicate focal component {key}")
            received[key] = _accepted(requests[key], evaluation)
        missing = requests.keys() - received.keys()
        if missing:
            raise ValueError(f"missing focal components: {sorted(missing)}")
        return self._finish(tuple(received[key] for key in requests))


def singlepoint(
    atoms: Iterable[Atom | tuple[str | int, Sequence[float]]],
    hf_cabs: FocalSeries[FocalBasis],
    correlation: FocalSeries[FocalBasis],
    post_mp2: FocalBasis | FocalSeries[FocalBasis] | None,
    *,
    provider: Callable[[FocalRequest], FocalEvaluation],
    root_id: str,
    correlation_space_id: str,
    charge: int = 0,
    multiplicity: int = 1,
    backend: Backend = "cuda",
    triples: FocalBasis | FocalSeries[FocalBasis] | None = None,
    f12_doubles: FocalBasis | FocalSeries[FocalBasis] | None = None,
    protocol_id: str | None = None,
) -> FocalRunResult:
    """Evaluate a geometry (Bohr) and compose energies (Eh) and forces (Eh/Bohr).

    Pass post_mp2=None to evaluate only HF+CABS and MP2-F12 correlation.
    Each provider result is checked and copied before the next provider call.
    """
    if not callable(provider):
        raise TypeError("a focal component provider is required")
    plan = plan_singlepoint(
        atoms,
        hf_cabs,
        correlation,
        post_mp2,
        root_id=root_id,
        correlation_space_id=correlation_space_id,
        charge=charge,
        multiplicity=multiplicity,
        backend=backend,
        triples=triples,
        f12_doubles=f12_doubles,
        protocol_id=protocol_id,
    )
    if (
        isinstance(provider, MolecularFocalProvider)
        and provider.high_order is None
        and post_mp2 is None
    ):
        molecular = _molecular_f12_plan(plan, provider, forces=True)
        assert isinstance(molecular, FocalMolecularResult)
        return molecular
    evaluations = tuple(
        _accepted(request, provider(request)) for request, _ in plan.steps
    )
    return plan._finish(evaluations)


def _molecular_f12_plan(
    plan: FocalPlan,
    provider: MolecularFocalProvider,
    *,
    forces: bool,
    center: FocalMolecularResult | None = None,
) -> FocalMolecularResult | FocalEnergyResult:
    """Share a determinant within each physical basis for this geometry only."""
    by_basis: dict[str, list[FocalRequest]] = {}
    for request, _ in plan.steps:
        by_basis.setdefault(request.basis.basis_id, []).append(request)
    if center is not None:
        provider = replace(
            provider, reference_seed=lambda r: center.references[r.basis.basis_id]
        )
    roots = {}
    ranks = {}
    energies = {}
    gradients = {}
    for identity, requests in by_basis.items():
        roles = {r.role for r in requests}
        pair = replace(requests[0], role="mp2_f12_correlation")
        reference = None
        expected_rank = center.cabs_ranks.get(identity) if center is not None else None
        if roles & {"mp2_f12_correlation", "ccsd_f12b_minus_mp2_f12"}:
            include_hf_cabs = "hf_cabs" in roles
            if forces:
                low_forces = provider.low_order_energy_and_forces(
                    pair, include_hf_cabs=include_hf_cabs
                )
                low = low_forces.energies
                if include_hf_cabs:
                    gradients[identity, "hf_cabs"] = low_forces.hf_cabs_forces
                gradients[identity, "mp2_f12_correlation"] = (
                    low_forces.mp2_f12_correlation_forces
                )
            else:
                low = provider.low_order_energy(
                    pair,
                    expected_cabs_rank=expected_rank,
                    include_hf_cabs=include_hf_cabs,
                )
                if expected_rank is not None and low.cabs_rank != expected_rank:
                    raise ValueError(
                        "displaced CABS rank differs from the analytic center"
                    )
            reference = low.reference
            ranks[identity] = low.cabs_rank
            if include_hf_cabs:
                energies[identity, "hf_cabs"] = low.hf_cabs_energy
            energies[identity, "mp2_f12_correlation"] = low.mp2_f12_correlation_energy
            if "ccsd_f12b_minus_mp2_f12" in roles:
                key = identity, "ccsd_f12b_minus_mp2_f12"
                if forces:
                    energies[key], gradients[key] = (
                        provider.f12_doubles_energy_and_forces(
                            pair, low_order=low_forces
                        )
                    )
                else:
                    energies[key] = provider.f12_doubles_energy(
                        pair, low_order=low, expected_cabs_rank=expected_rank
                    )
        elif "hf_cabs" in roles:
            hf = replace(pair, role="hf_cabs")
            reference = df_reference(
                hf,
                forces=forces,
                budget_mib=provider.budget_mib,
                **provider._seed_options(hf),
            )
            one = one_electron_sources(
                hf,
                library=provider.one_electron_library or provider.shell_library,
                derivatives=forces,
                budget_mib=provider.budget_mib,
            )
            jk = provider._panel(hf, "jk_fit")
            proj = focal_cabs_projector(
                hf,
                one,
                expected_rank=expected_rank,
                relative_cutoff=provider.relative_cutoff,
                library=provider.response_library,
            )
            ranks[identity] = proj.rank
            if forces:
                energies[identity, "hf_cabs"], gradients[identity, "hf_cabs"] = (
                    provider._singles(
                        hf, reference, one, jk, expected_cabs_rank=proj.rank
                    )
                )
            else:
                singles = reference_cabs_singles(
                    hf,
                    reference,
                    one,
                    complement=proj.frame,
                    three_center=jk.data["ao_coulomb"],
                    metric=jk.data["metric"],
                    whitener=jk.whitener,
                    pullback=False,
                    relax=False,
                    run=provider.run,
                    library=provider.response_library,
                    relative_cutoff=provider.relative_cutoff,
                    budget_mib=provider.budget_mib,
                )
                energies[identity, "hf_cabs"] = singles.hf_cabs_energy
        if reference is None:
            reference = df_reference(
                pair,
                forces=False,
                budget_mib=provider.budget_mib,
                **provider._seed_options(pair),
            )
        if reference.executed_backend != pair.backend:
            raise RuntimeError(
                "molecular focal reference executed on a different backend"
            )
        roots[identity] = FocalReferenceSeed(pair, reference)
        if "triples" in roles:
            triple = replace(pair, role="triples")
            key = identity, "triples"
            # Fewer than three occupied spin orbitals leaves no triples determinants at any geometry.
            if (
                triple.root_id == "rhf"
                and len(reference.occupied) == 1
                and reference.occupied[0] == 1
            ):
                energies[key] = 0.0
                if forces:
                    gradients[key] = np.zeros((len(triple.atoms), 3))
            elif forces:
                energies[key], gradients[key] = provider.triples_energy_and_forces(
                    triple, reference=reference
                )
            else:
                energies[key] = provider.triples_energy(triple, reference=reference)
    parts = []
    for request, weight in plan.steps:
        value = float(energies[request.basis.basis_id, request.role])
        if not math.isfinite(value):
            raise RuntimeError(f"nonfinite {request.role} molecular energy")
        parts.append(FocalEnergyComponent(request, value, weight))
    if not forces:
        total = math.fsum(c.energy * c.weight for c in parts)
        if not math.isfinite(total):
            raise RuntimeError("combined molecular energy is not finite")
        return FocalEnergyResult(total, tuple(parts), roots, ranks)
    evaluated = []
    for part in parts:
        r = part.request
        term = FocalTerm(
            r.surface_id,
            r.basis.series_id,
            r.basis.basis_id,
            r.basis.cardinal,
            r.role,
            part.energy,
            gradients[r.basis.basis_id, r.role],
        )
        evaluated.append(_accepted(r, FocalEvaluation(term, r.backend, True)))
    result = plan._finish(tuple(evaluated))
    return FocalMolecularResult(
        result.energy, result.forces, result.weights, result.components, roots, ranks
    )


def singlepoint_energy(
    atoms: Iterable[Atom | tuple[str | int, Sequence[float]]],
    hf_cabs: FocalSeries[FocalBasis],
    correlation: FocalSeries[FocalBasis],
    post_mp2: None = None,
    *,
    provider: MolecularFocalProvider,
    root_id: str,
    correlation_space_id: str,
    charge: int = 0,
    multiplicity: int = 1,
    backend: Backend = "cuda",
    triples: FocalBasis | FocalSeries[FocalBasis] | None = None,
    f12_doubles: FocalBasis | FocalSeries[FocalBasis] | None = None,
    protocol_id: str | None = None,
    center: FocalMolecularResult | None = None,
) -> FocalEnergyResult:
    """Evaluate one molecular HF+CABS/MP2-F12 recipe with optional F12 doubles and (T).

    Positions are in Bohr and energies in Hartree. Each basis uses one forward-only
    determinant. Supply the analytic center to track its occupied root and CABS rank.
    """
    if post_mp2 is not None:
        raise ValueError("use f12_doubles for a molecular CCSD-F12b recipe")
    if (
        not isinstance(provider, MolecularFocalProvider)
        or provider.high_order is not None
    ):
        raise ValueError(
            "singlepoint_energy requires the molecular provider's native energy implementations"
        )
    plan = plan_singlepoint(
        atoms,
        hf_cabs,
        correlation,
        None,
        root_id=root_id,
        correlation_space_id=correlation_space_id,
        charge=charge,
        multiplicity=multiplicity,
        backend=backend,
        triples=triples,
        f12_doubles=f12_doubles,
        protocol_id=protocol_id,
    )
    if center is not None:
        if not isinstance(center, FocalMolecularResult):
            raise TypeError("center must be a molecular focal result")
        cabs_requests = {
            q.basis.basis_id: q
            for q, _ in plan.steps
            if q.role in ("hf_cabs", "mp2_f12_correlation", "ccsd_f12b_minus_mp2_f12")
        }
        if set(center.cabs_ranks) != set(cabs_requests):
            raise ValueError(
                "center CABS ranks must identify every requested complementary basis"
            )
        for identity, q in cabs_requests.items():
            assert q.basis.cabs is not None
            if q.basis.cabs.representation != "spherical":
                raise ValueError("molecular CABS requires a real-spherical basis")
            elements = q.basis.cabs.by_element
            dimension = sum(
                (2 * shell.angular_momentum + 1) * len(shell.coefficients)
                for atom in q.atoms
                for shell in elements[atom.atomic_number].shells
            )
            checked_integer(
                center.cabs_ranks[identity], "center CABS rank", low=1, high=dimension
            )
        weights = tuple(
            (q.role, q.basis.basis_id, q.basis.cardinal, w) for q, w in plan.steps
        )
        if center.weights != weights:
            raise ValueError(
                "center and displacement must use the same weighted focal recipe"
            )
        bases = {q.basis.basis_id for q, _ in plan.steps}
        if set(center.references) != bases:
            raise ValueError(
                "center and displacement must use the same molecular bases"
            )
        if any(
            not isinstance(seed, FocalReferenceSeed)
            or seed.request.basis.basis_id != identity
            or seed.reference.executed_backend != seed.request.backend
            for identity, seed in center.references.items()
        ):
            raise ValueError("center has an incompatible basis-specific determinant")
        source_atoms = next(iter(center.references.values())).request.atoms
        if tuple(a.atomic_number for a in source_atoms) != tuple(
            a.atomic_number for a in plan.steps[0][0].atoms
        ):
            raise ValueError(
                "center and displacement must have the same ordered nuclei"
            )
        source_plan = plan_singlepoint(
            source_atoms,
            hf_cabs,
            correlation,
            None,
            root_id=root_id,
            correlation_space_id=correlation_space_id,
            charge=charge,
            multiplicity=multiplicity,
            backend=backend,
            triples=triples,
            f12_doubles=f12_doubles,
            protocol_id=protocol_id,
        )
        if any(
            seed.request.atoms != source_atoms
            or seed.request.surface_id != source_plan.steps[0][0].surface_id
            for seed in center.references.values()
        ):
            raise ValueError(
                "center and displacement must use the same electronic surface and protocol"
            )
    result = _molecular_f12_plan(plan, provider, forces=False, center=center)
    assert isinstance(result, FocalEnergyResult)
    return result
