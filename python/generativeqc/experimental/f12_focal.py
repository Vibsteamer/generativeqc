"""Dispatch and combine same-surface F12 energy and analytic-force increments."""

from __future__ import annotations

import json
import math
import typing
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Generic, Literal, TypeVar

import numpy as np

from .. import (
    _f12_focal_cv,
    _f12_focal_integral_kernel,
    _f12_focal_molecular,
    _f12_focal_one_electron,
    _f12_focal_projector,
    _f12_focal_reference,
    _f12_focal_singles,
    _f12_frozen_core,
    _native,
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
    "triples_plus",
    "triples_f12",
]
TriplesMethod = Literal["conventional", "tplus", "f12"]
_TRIPLES_ROLES: dict[TriplesMethod, Role] = {
    "conventional": "triples",
    "tplus": "triples_plus",
    "f12": "triples_f12",
}
_METHOD_BY_ROLE = {role: method for method, role in _TRIPLES_ROLES.items()}
_CABS_ROLES = {
    "hf_cabs",
    "mp2_f12_correlation",
    "ccsd_f12b_minus_mp2_f12",
    "triples_plus",
    "triples_f12",
}
Backend = Literal["cuda", "cpu_reference"]
SlaterNormalization = Literal[
    "negative_exponential", "negative_exponential_over_beta"
]
DEFAULT_SLATER_FIT_NORMALIZATION = (
    _f12_focal_integral_kernel.DEFAULT_SLATER_FIT_NORMALIZATION
)
MolecularFocalProvider = _f12_focal_molecular.MolecularFocalProvider
conventional_mp2_request = _f12_focal_cv.conventional_mp2_request
DEFAULT_PRODUCTION_CORRELATION_SPACE = _f12_frozen_core.DEFAULT_PRODUCTION_CORRELATION_SPACE
FROZEN_CORE_DEFINITION_VERSION = _f12_frozen_core.FROZEN_CORE_DEFINITION_VERSION
frozen_core_orbitals_for_atom = _f12_frozen_core.frozen_core_orbitals_for_atom
frozen_core_spd_orbitals_for_atom = _f12_frozen_core.frozen_core_spd_orbitals_for_atom
cv_active_frozen_orbitals_for_atom = _f12_frozen_core.cv_active_frozen_orbitals_for_atom
resolve_frozen_occupied = _f12_frozen_core.resolve_frozen_occupied
FocalOneElectron = _f12_focal_one_electron.FocalOneElectron
one_electron_sources = _f12_focal_one_electron.one_electron_sources
FocalReference = _f12_focal_reference.FocalReference
FocalReferenceSeed = _f12_focal_reference.FocalReferenceSeed
df_reference = _f12_focal_reference.df_reference
focal_cabs_projector = _f12_focal_projector.focal_cabs_projector
CompensatedProductWorkLimit = _f12_focal_projector.CompensatedProductWorkLimit
FocalCabsSingles = _f12_focal_singles.FocalCabsSingles
reference_cabs_singles = _f12_focal_singles.reference_cabs_singles


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


_TriplesInput = (
    _Endpoint
    | FocalSeries[_Endpoint]
    | Mapping[TriplesMethod, _Endpoint | FocalSeries[_Endpoint]]
)
_Result = TypeVar("_Result")


@dataclass(frozen=True)
class FocalTriplesAlternatives(Generic[_Result]):
    """One total per requested triples method, in the requested method order."""

    variants: Mapping[TriplesMethod, _Result]


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
    slater_normalization: SlaterNormalization = DEFAULT_SLATER_FIT_NORMALIZATION


@dataclass(frozen=True)
class FocalEvaluation:
    """Component output with the convergence and backend reported by its provider."""

    term: FocalTerm
    executed_backend: Backend
    converged: bool
    triples_model_id: str | None = None
    telemetry: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FocalRunResult(FocalResult):
    """Composed result with the completed component outputs in weight order."""

    components: tuple[FocalEvaluation, ...]
    valence_correlation_space: str | None = field(default=None, kw_only=True)
    frozen_core_definition_version: int | None = field(default=None, kw_only=True)


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
    executed_backend: Backend | None = None
    converged: bool | None = None
    triples_model_id: str | None = None
    telemetry: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FocalEnergyResult:
    """Composed energy in Hartree and its actual forward-only component values."""

    energy: float
    components: tuple[FocalEnergyComponent, ...]
    references: dict[str, FocalReferenceSeed]
    cabs_ranks: dict[str, int]
    valence_correlation_space: str | None = field(default=None, kw_only=True)
    frozen_core_definition_version: int | None = field(default=None, kw_only=True)


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
    triples: _TriplesInput[_Endpoint] | None,
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
        alternatives = (
            triples if isinstance(triples, Mapping) else {"conventional": triples}
        )
        if not alternatives:
            raise ValueError("select at least one triples method")
        for method, source in alternatives.items():
            if method not in _TRIPLES_ROLES:
                raise ValueError(f"unknown focal triples method {method!r}")
            series = source if isinstance(source, FocalSeries) else FocalSeries(source)
            roles.append((series, _TRIPLES_ROLES[method]))
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


def _alternative_indices(roles: Iterable[Role]) -> dict[TriplesMethod, tuple[int, ...]]:
    roles = tuple(roles)
    return {
        method: tuple(
            i
            for i, role in enumerate(roles)
            if role not in _METHOD_BY_ROLE or role == selected
        )
        for method, selected in dict.fromkeys(
            (_METHOD_BY_ROLE[role], role) for role in roles if role in _METHOD_BY_ROLE
        )
    }


def combine(
    hf_cabs: FocalSeries[FocalTerm],
    correlation: FocalSeries[FocalTerm],
    post_mp2: FocalTerm | FocalSeries[FocalTerm] | None,
    *,
    triples: _TriplesInput[FocalTerm] | None = None,
    f12_doubles: FocalTerm | FocalSeries[FocalTerm] | None = None,
) -> FocalResult | FocalTriplesAlternatives[FocalResult]:
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
    weighted = _weighted_terms(
        hf_cabs, correlation, post_mp2, triples, f12_doubles=f12_doubles
    )
    alternatives = _alternative_indices(role for role, _, _ in weighted)
    if len(alternatives) > 1:
        return FocalTriplesAlternatives(
            {
                method: _assemble([weighted[i] for i in selected])
                for method, selected in alternatives.items()
            }
        )
    return _assemble(weighted)


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
        f12_pair = role in (
            "mp2_f12_correlation",
            "ccsd_f12b_minus_mp2_f12",
            "triples_plus",
            "triples_f12",
        )
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
    triples: _TriplesInput[FocalBasis] | None = None,
    f12_doubles: FocalBasis | FocalSeries[FocalBasis] | None = None,
    protocol_id: str | None = None,
    slater_normalization: SlaterNormalization = DEFAULT_SLATER_FIT_NORMALIZATION,
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
    _f12_focal_integral_kernel.slater_normalization_code(slater_normalization)
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
            "frozen_core_definition_version": _f12_frozen_core.FROZEN_CORE_DEFINITION_VERSION,
            "slater_normalization": slater_normalization,
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
            slater_normalization=slater_normalization,
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
    return replace(
        evaluation, term=replace(term, energy=value, forces=forces.copy()),
        telemetry=_correlation_telemetry(request, evaluation.telemetry),
    )


def _correlation_telemetry(
    request: FocalRequest, telemetry: Mapping[str, object], reference: FocalReference | None = None,
) -> dict[str, object]:
    version = _f12_frozen_core.FROZEN_CORE_DEFINITION_VERSION
    if ("frozen_core_definition_version" in telemetry and
            telemetry["frozen_core_definition_version"] != version):
        raise ValueError("component returned a different frozen-core definition version")
    if "correlation_space" in telemetry and telemetry["correlation_space"] != request.correlation_space_id:
        raise ValueError("component returned a different correlation space")
    result = {
        **telemetry,
        "correlation_space": request.correlation_space_id,
        "frozen_core_definition_version": version,
    }
    frame = getattr(reference, "frozen_frame", None)
    if frame is not None:
        result["frozen_occupied_indices"] = tuple(
            channel.indices for channel in frame.channels(request.correlation_space_id)
        )
    return result


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
        spaces = {request.correlation_space_id for request, _ in self.steps}
        if len(spaces) != 1:
            raise ValueError("Focal valence components must share one correlation space")
        return FocalRunResult(
            result.energy, result.forces, result.weights, evaluations,
            valence_correlation_space=spaces.pop(),
            frozen_core_definition_version=_f12_frozen_core.FROZEN_CORE_DEFINITION_VERSION,
        )

    def _selections(self) -> dict[TriplesMethod, tuple[int, ...]]:
        return _alternative_indices(request.role for request, _ in self.steps)

    def _finish_variants(
        self, evaluations: tuple[FocalEvaluation, ...]
    ) -> FocalRunResult | FocalTriplesAlternatives[FocalRunResult]:
        variants = self._selections()
        if len(variants) <= 1:
            return self._finish(evaluations)
        return FocalTriplesAlternatives(
            {
                method: FocalPlan(tuple(self.steps[i] for i in selected))._finish(
                    tuple(evaluations[i] for i in selected)
                )
                for method, selected in variants.items()
            }
        )

    def collect(
        self, evaluations: Iterable[FocalEvaluation]
    ) -> FocalRunResult | FocalTriplesAlternatives[FocalRunResult]:
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
        return self._finish_variants(tuple(received[key] for key in requests))


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
    triples: _TriplesInput[FocalBasis] | None = None,
    f12_doubles: FocalBasis | FocalSeries[FocalBasis] | None = None,
    protocol_id: str | None = None,
    slater_normalization: SlaterNormalization | None = None,
) -> FocalRunResult | FocalTriplesAlternatives[FocalRunResult]:
    """Evaluate a geometry (Bohr) and compose energies (Eh) and forces (Eh/Bohr).

    Pass post_mp2=None to evaluate only HF+CABS and MP2-F12 correlation.
    Each provider result is checked and copied before the next provider call.
    A triples mapping selects conventional, tplus or f12 contributions,
    each with its own total.
    """
    if not callable(provider):
        raise TypeError("a focal component provider is required")
    resolved_norm = (
        DEFAULT_SLATER_FIT_NORMALIZATION
        if slater_normalization is None
        else slater_normalization
    )
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
        slater_normalization=resolved_norm,
    )
    evaluations = tuple(
        _accepted(request, provider(request)) for request, _ in plan.steps
    )
    return plan._finish_variants(evaluations)
