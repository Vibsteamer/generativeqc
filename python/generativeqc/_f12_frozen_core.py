"""Standard chemical valence frozen-core and explicit occupied-space resolution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import numpy as np

from ._f12_triples_w_policy import reject_unqualified_single_pass_w_forces

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ._api_types import Atom
    from ._f12_focal_cv import FocalDeltaCVResult
    from ._x2c1e import DeltaSrResult
    from .basis import BasisSet, ElementBasis
    from .experimental.f12_focal import FocalEnergyResult, FocalRequest, FocalRunResult


DEFAULT_PRODUCTION_CORRELATION_SPACE = "frozen-core-spd"
DEFAULT_PRODUCTION_F12_BETA_BY_CARDINAL: dict[int, float] = {2: 0.9, 3: 1.0, 4: 1.0}
DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH = 0.05
FROZEN_CORE_DEFINITION_VERSION = 2
_MIN_CORE_OVERLAP = 0.9


def reject_diagnostic_components(components: Iterable[object]) -> None:
    """Refuse diagnostic results and unqualified single-pass FP32 triples W forces.

    Each component exposes a `telemetry` mapping, as focal result components and
    triples evaluations do. The component identifies analytic force scope
    independently of W telemetry, which may omit a direction or report no W work.
    """
    from .experimental.f12_focal import FocalEvaluation

    for component in components:
        telemetry = getattr(component, "telemetry", None)
        if not isinstance(telemetry, Mapping):
            raise TypeError(
                "production focal components must expose a telemetry mapping"
            )
        if telemetry.get("production_eligible") is False:
            raise ValueError(
                f"diagnostic focal result {telemetry.get('diagnostic_mode')!r} cannot enter "
                "production focal validation or assembly"
            )
        if (
            isinstance(component, FocalEvaluation)
            or getattr(component, "forces", None) is not None
        ):
            reject_unqualified_single_pass_w_forces(
                telemetry, executed_backend=getattr(component, "executed_backend", None)
            )


def _nearest_admissible_frozen_counts(
    eps: np.ndarray, n_frozen: int, upper: int, min_gap_eh: float
) -> tuple[int, int | None]:
    below = next(
        (k for k in range(n_frozen - 1, 0, -1) if eps[k] - eps[k - 1] >= min_gap_eh),
        0,
    )
    above = next(
        (k for k in range(n_frozen + 1, upper) if eps[k] - eps[k - 1] >= min_gap_eh),
        None,
    )
    return below, above


def check_frozen_core_shell_gap(
    orbital_energies: np.ndarray | Sequence[float],
    n_frozen: int,
    min_gap_eh: float = DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH,
    *,
    n_occupied: int | None = None,
    correlation_space_id: str | None = None,
    spin_label: str | None = None,
) -> float | None:
    """Detect when freezing `n_frozen` orbitals cuts through a near-degenerate occupied shell.

    Checks `eps[n_frozen] - eps[n_frozen - 1] >= min_gap_eh` (default 0.05 Eh) when
    `0 < n_frozen < upper` (where `upper` is `n_occupied` when provided, else
    `len(orbital_energies)`). A split shell raises `ValueError` naming the nearest
    explicit frozen counts whose boundary gap passes in this channel.
    """
    if isinstance(n_frozen, bool) or not isinstance(n_frozen, int) or n_frozen < 0:
        raise ValueError(f"n_frozen must be a nonnegative integer, got {n_frozen!r}")
    if isinstance(min_gap_eh, bool) or not math.isfinite(min_gap_eh) or min_gap_eh < 0.0:
        raise ValueError(f"min_gap_eh must be finite and nonnegative, got {min_gap_eh!r}")
    if n_frozen == 0 or min_gap_eh == 0.0:
        return None
    eps = np.asarray(orbital_energies, dtype=np.float64)
    if eps.ndim != 1:
        raise ValueError(
            f"orbital_energies must be 1D for shell-gap validation, got shape {eps.shape}"
        )
    upper = eps.size if n_occupied is None else n_occupied
    if upper > eps.size:
        raise ValueError(
            f"n_occupied={upper} exceeds orbital_energies length {eps.size}"
        )
    if n_frozen >= upper:
        return None
    e_hi_frozen = float(eps[n_frozen - 1])
    e_lo_active = float(eps[n_frozen])
    if not (math.isfinite(e_hi_frozen) and math.isfinite(e_lo_active)):
        raise ValueError("Nonfinite orbital energies across frozen-core boundary")
    gap = e_lo_active - e_hi_frozen
    if gap < min_gap_eh:
        spin_desc = f" ({spin_label} spin)" if spin_label else ""
        space_desc = (
            f" for correlation_space_id={correlation_space_id!r}"
            if correlation_space_id
            else ""
        )
        below, above = _nearest_admissible_frozen_counts(eps, n_frozen, upper, min_gap_eh)
        nearest = (
            f"counts with a boundary gap of at least {min_gap_eh:.4f} Eh are {below} and {above}"
            if above is not None
            else f"count with a boundary gap of at least {min_gap_eh:.4f} Eh is {below}"
        )
        message = (
            f"Frozen-core boundary n_frozen={n_frozen}{spin_desc}{space_desc} splits a "
            f"near-degenerate shell: eps[{n_frozen}] - eps[{n_frozen - 1}] = "
            f"{e_lo_active:.6f} - ({e_hi_frozen:.6f}) = {gap:.6f} Eh < {min_gap_eh:.4f} Eh. "
            f"In this channel, the nearest frozen {nearest}."
        )
        raise ValueError(message)
    return gap


def _split_spin_orbital_energies(
    orbital_energies: np.ndarray | Sequence[float] | Sequence[np.ndarray | Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(orbital_energies, np.ndarray):
        if orbital_energies.ndim == 1:
            return (orbital_energies, orbital_energies)
        if orbital_energies.ndim == 2 and orbital_energies.shape[0] == 1:
            return (orbital_energies[0], orbital_energies[0])
        if orbital_energies.ndim == 2 and orbital_energies.shape[0] == 2:
            return (orbital_energies[0], orbital_energies[1])
        raise ValueError(
            f"Expected 1D or (1|2, nmo) orbital_energies array, got shape {orbital_energies.shape}"
        )
    seq = tuple(orbital_energies)
    if not seq:
        raise ValueError("orbital_energies cannot be empty")
    if isinstance(seq[0], (int, float, np.floating, np.integer)):
        arr = np.asarray(seq, dtype=np.float64)
        return (arr, arr)
    if len(seq) == 1:
        arr = np.asarray(seq[0], dtype=np.float64)
        return (arr, arr)
    if len(seq) == 2:
        return (
            np.asarray(seq[0], dtype=np.float64),
            np.asarray(seq[1], dtype=np.float64),
        )
    raise ValueError(
        f"Expected 1 or 2 spin channels of orbital_energies, got {len(seq)}"
    )


def frozen_core_orbitals_for_atom(
    atomic_number: int, ecp_core_electrons: int = 0
) -> int:
    """Return the standard valence-correlation frozen spatial orbitals for one atom.

    Follows Peterson/Hill cc-pVXZ-F12 and cc-pVXZ-PP-F12 / W1-P34 conventions:
    - H, He (Z <= 2): 0
    - Li..Ne (3 <= Z <= 10): 1s (1 orbital) minus ECP core pairs
    - Na: 1s (1 orbital) minus ECP core pairs
    - Mg..Ar (12 <= Z <= 18): 1s2s2p (5 orbitals) minus ECP core pairs
    - K..Kr (19 <= Z <= 36): 1s2s2p3s3p (9 orbitals) minus ECP core pairs
      (for Br ECP10MDF with 10 ECP electrons = 5 pairs -> 4 frozen 3s3p orbitals,
       correlating 3d4s4p = 17 electrons)
    - Rb..Xe (37 <= Z <= 54): 1s..3d4s4p (18 orbitals) minus ECP core pairs
      (for I ECP28MDF with 28 ECP electrons = 14 pairs -> 4 frozen 4s4p orbitals,
       correlating 4d5s5p = 17 electrons, matching W1-P34)
    """
    if atomic_number < 1:
        raise ValueError(f"Atomic number must be positive: {atomic_number}")
    if ecp_core_electrons < 0 or ecp_core_electrons % 2 != 0:
        raise ValueError(f"ECP core electron count must be even and nonnegative: {ecp_core_electrons}")
    ecp_pairs = ecp_core_electrons // 2
    if atomic_number <= 2:
        total_core = 0
    elif atomic_number <= 11:
        total_core = 1
    elif atomic_number <= 18:
        total_core = 5
    elif atomic_number <= 36:
        total_core = 9
    elif atomic_number <= 54:
        total_core = 18
    else:
        raise ValueError(
            f"Unsupported atomic number {atomic_number} for standard frozen-core count; "
            "specify correlation_space_id='frozen:<n>' explicitly"
        )
    return max(0, total_core - ecp_pairs)


def frozen_core_spd_orbitals_for_atom(
    atomic_number: int, ecp_core_electrons: int = 0
) -> int:
    """Return Hill-Peterson / TA13 valence-only frozen spatial orbitals (freezing (n-1)d on post-d Br/I).

    Identical to frozen_core_orbitals_for_atom for Z <= 30 and 37 <= Z <= 48,
    and freezes the complete outer-core (n-1)s(n-1)p(n-1)d shell on post-d p-block
    elements (Ga..Kr: 14 total -> 9 frozen for Br ECP10MDF, correlating 7 valence e-;
    In..Xe: 23 total -> 9 frozen for I ECP28MDF, correlating 7 valence e-).
    Na freezes 1s and correlates 2s2p in both standard valence spaces.
    The complete halogen (n-1)spd contribution enters additive Delta-CV.
    """
    if atomic_number < 1:
        raise ValueError(f"Atomic number must be positive: {atomic_number}")
    if ecp_core_electrons < 0 or ecp_core_electrons % 2 != 0:
        raise ValueError(
            f"ECP core electron count must be even and nonnegative: {ecp_core_electrons}"
        )
    ecp_pairs = ecp_core_electrons // 2
    if atomic_number <= 2:
        total_core = 0
    elif atomic_number <= 11:
        total_core = 1
    elif atomic_number <= 18:
        total_core = 5
    elif atomic_number <= 30:
        total_core = 9
    elif atomic_number <= 36:
        total_core = 14
    elif atomic_number <= 48:
        total_core = 18
    elif atomic_number <= 54:
        total_core = 23
    else:
        raise ValueError(
            f"Unsupported atomic number {atomic_number} for frozen-core-spd count; "
            "specify correlation_space_id='frozen:<n>' explicitly"
        )
    return max(0, total_core - ecp_pairs)


def cv_active_frozen_orbitals_for_atom(
    atomic_number: int, ecp_core_electrons: int = 0
) -> int:
    """Return deep-core frozen spatial orbitals for core-valence ('cv-active') correlation.

    Correlates the outer-core shell together with the valence shell while keeping
    deep inner cores frozen where appropriate (Hill-Peterson cc-pCVXZ-F12 /
    cc-pwCVXZ-PP convention):
    - H..Ne (Z <= 10): 0 (correlates all electrons including 1s)
    - Na..Ar (11 <= Z <= 18): 1 minus ECP core pairs (keeps deep 1s frozen,
      correlates 2s2p + valence)
    - K..Kr (19 <= Z <= 36): 5 minus ECP core pairs (1s2s2p frozen for all-electron;
      for Br ECP10MDF with 10 ECP electrons = 5 pairs -> 0 frozen, correlating
      all 25 explicit electrons 3s3p3d4s4p and releasing 3s3p relative to FC)
    - Rb..Xe (37 <= Z <= 54): 14 minus ECP core pairs (1s..3d frozen for all-electron;
      for I ECP28MDF with 28 ECP electrons = 14 pairs -> 0 frozen, correlating
      all 25 explicit electrons 4s4p4d5s5p and releasing 4s4p relative to FC)
    """
    if atomic_number < 1:
        raise ValueError(f"Atomic number must be positive: {atomic_number}")
    if ecp_core_electrons < 0 or ecp_core_electrons % 2 != 0:
        raise ValueError(
            f"ECP core electron count must be even and nonnegative: {ecp_core_electrons}"
        )
    ecp_pairs = ecp_core_electrons // 2
    if atomic_number <= 10:
        total_deep_core = 0
    elif atomic_number <= 18:
        total_deep_core = 1
    elif atomic_number <= 36:
        total_deep_core = 5
    elif atomic_number <= 54:
        total_deep_core = 14
    else:
        raise ValueError(
            f"Unsupported atomic number {atomic_number} for cv-active frozen-core count; "
            "specify correlation_space_id='frozen:<n>' explicitly"
        )
    return max(0, total_deep_core - ecp_pairs)


_STANDARD_CORE_ORBITALS = {
    "frozen-core": frozen_core_orbitals_for_atom,
    DEFAULT_PRODUCTION_CORRELATION_SPACE: frozen_core_spd_orbitals_for_atom,
    "cv-active": cv_active_frozen_orbitals_for_atom,
}


def _standard_frozen_count(atoms: Sequence[Atom], orbital_basis: BasisSet, space: str) -> int:
    per_atom = _STANDARD_CORE_ORBITALS[space]
    elements = orbital_basis.by_element
    return sum(
        per_atom(
            atom.atomic_number,
            elements[atom.atomic_number].ecp_core_electrons,
        )
        for atom in atoms
    )


@dataclass(frozen=True)
class FrozenOrbitalChannel:
    """A frozen selection in the original canonical occupied order."""

    indices: tuple[int, ...]
    minimum_gap_eh: float | None
    shell_margin: float | None = None
    minimum_core_overlap: float | None = None


@dataclass(frozen=True)
class FrozenOccupiedFrame:
    """Named selections embedded as prefixes of a common canonical frame."""

    partitions: tuple[tuple[str, tuple[FrozenOrbitalChannel, ...]], ...]
    occupied_order: tuple[tuple[int, ...], ...]

    def channels(self, space: str) -> tuple[FrozenOrbitalChannel, ...]:
        return dict(self.partitions)[space]

    def counts(self, space: str) -> tuple[int, int]:
        channels = self.channels(space)
        return (len(channels[0].indices), len(channels[-1].indices))


@lru_cache(maxsize=128)
def _atomic_core_orbitals(
    basis: BasisSet, element: ElementBasis,
) -> tuple[np.ndarray, np.ndarray]:
    """Cache a neutral native spherical HF result by basis and element/ECP identity."""
    import ctypes as ct
    from contextlib import ExitStack

    from . import _native
    from ._api_types import Atom
    from .calculator import Calculator

    z, ecp = element.atomic_number, element.ecp_core_electrons
    count = frozen_core_spd_orbitals_for_atom(z, ecp)
    # The occupied angular sectors through Xe are s/p/d. A spherical atom
    # has no Fock coupling to polarization sectors of higher angular momentum.
    retained = tuple(shell for shell in element.shells if shell.angular_momentum <= 2)
    reduced = replace(basis, elements=(replace(element, shells=retained),))
    atom = (Atom(z, (0.0, 0.0, 0.0)),)
    multiplicity = 1 + (z - ecp) % 2
    calc = Calculator(
        basis=reduced, method="rhf" if multiplicity == 1 else "uhf",
        device="cpu", density_fitting="none", initial_guess=None, precision="fp64",
    )
    rows = np.asarray([
        shell.angular_momentum <= 2
        for shell in element.shells
        for _ in range((2 * shell.angular_momentum + 1) * len(shell.coefficients))
    ])
    coeff = np.empty((int(rows.sum()), count))
    energies = np.empty(count + 1)
    with ExitStack() as owners:
        owners.callback(calc.clear_cache)
        lib = calc._library
        native = getattr(lib, "generativeqc_focal_atomic_core_v1", None)
        if native is None:
            raise RuntimeError("the native library lacks spherical atomic HF for frozen-core selection")
        array = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        native.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_size_t,
                           array, ct.c_size_t, array, ct.c_size_t]
        native.restype = ct.c_int
        ctx = ct.c_void_p()
        _native.check(lib, lib.generativeqc_context_create(
            ct.byref(calc._context_descriptor()), ct.byref(ctx)
        ))
        owners.callback(lib.generativeqc_context_destroy, ctx)
        system = calc._create_native_system(ctx, atom, 0, multiplicity)
        owners.callback(lib.generativeqc_system_destroy, system)
        _native.check(lib, native(ctx, system, count, 256 << 20,
                                  coeff, coeff.size, energies, energies.size), context=ctx)
    full = np.zeros((rows.size, count))
    full[rows] = coeff
    full.setflags(write=False)
    energies.setflags(write=False)
    return full, energies


def _atomic_core_projector(
    atoms: Sequence[Atom], basis: BasisSet, overlap: np.ndarray, space: str,
) -> np.ndarray:
    """Return S times the Löwdin-orthonormalized embedded atomic core frame."""
    if basis.representation != "spherical":
        raise ValueError("frozen-core identity requires a real-spherical orbital basis")
    elements = basis.by_element
    row = column = 0
    per_atom = _STANDARD_CORE_ORBITALS[space]
    n_core = _standard_frozen_count(atoms, basis, space)
    embedded = np.zeros((len(overlap), n_core))
    for atom in atoms:
        element = elements[atom.atomic_number]
        k = per_atom(atom.atomic_number, element.ecp_core_electrons)
        rows = sum(
            (2 * shell.angular_momentum + 1) * len(shell.coefficients)
            for shell in element.shells
        )
        if k:
            orbitals, energies = _atomic_core_orbitals(basis, element)
            if energies[k] - energies[k - 1] < 1e-3:
                raise ValueError(f"{space!r} cuts an atomic core shell on Z={atom.atomic_number}")
            embedded[row:row + rows, column:column + k] = orbitals[:, :k]
        row += rows
        column += k
    if row != len(overlap):
        raise ValueError("frozen-core overlap and orbital basis AO counts differ")
    metric_core = overlap @ embedded
    e, u = np.linalg.eigh(embedded.T @ metric_core)
    if e[0] <= 1e-10 or not np.isfinite(e).all():
        raise ValueError("atomic frozen-core shells are linearly dependent")
    return metric_core @ ((u * e**-0.5) @ u.T)


def _identity_frozen_channel(
    eps: np.ndarray, projections: np.ndarray, n_frozen: int,
    *, space: str, spin: str, spd_projections: np.ndarray | None,
) -> FrozenOrbitalChannel:
    weights = np.sum(projections**2, axis=1)
    frozen = np.sort(np.argsort(-weights, kind="stable")[:n_frozen])
    active = np.flatnonzero(~np.isin(np.arange(len(eps)), frozen))
    description = f"Frozen-core {space!r} ({spin} spin)"
    margin = min(float(weights[frozen].min() - 0.5), float(0.5 - weights[active].max()))
    if margin <= 0.0 or not np.isfinite(margin):
        raise ValueError(
            f"{description} has an ambiguous shell identity: every frozen orbital must have "
            "majority atomic-core character and each active orbital must have less than half."
        )
    minimum_overlap = float(np.linalg.svd(projections[frozen], compute_uv=False)[-1])
    if minimum_overlap < _MIN_CORE_OVERLAP:
        raise ValueError(
            f"{description} does not span its atomic core shells: minimum overlap "
            f"{minimum_overlap:.6f} < {_MIN_CORE_OVERLAP:g}."
        )
    if spd_projections is not None:
        interleaved = active[eps[active] < np.max(eps[frozen])]
        if np.any(np.sum(spd_projections[interleaved]**2, axis=1) >= 0.5):
            raise ValueError(
                f"{description} leaves an interleaved (n-1)d core shell active. "
                "Use 'frozen-core-spd' to freeze the complete shell."
            )
    gaps = np.abs(eps[frozen, None] - eps[active])
    i, j = np.unravel_index(np.argmin(gaps), gaps.shape)
    gap = float(gaps[i, j])
    if gap < DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH:
        raise ValueError(
            f"{description} has a near-degenerate frozen-active pair "
            f"({frozen[i]}, {active[j]}): |Δε| = {gap:.6f} Eh < "
            f"{DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH:.4f} Eh."
        )
    return FrozenOrbitalChannel(tuple(map(int, frozen)), gap, margin, minimum_overlap)


def select_frozen_occupied_frame(
    atoms: Sequence[Atom], orbital_basis: BasisSet, overlap: np.ndarray,
    coefficients: Sequence[np.ndarray], orbital_energies: Sequence[np.ndarray],
    occupied: tuple[int, ...], *spaces: str,
) -> FrozenOccupiedFrame:
    """Select chemical cores by shell identity and form nested occupied prefixes.

    Consecutive explicit counts retain their lowest-energy interpretation. The
    common frame can represent several spaces only when their selections nest.
    """
    nspin = len(occupied)
    if nspin not in (1, 2):
        raise ValueError("frozen-core selection requires one or two physical spin channels")
    occ_pair = (occupied[0], occupied[-1])
    partitions = []
    order_by_spin = [[] for _ in occupied]
    for space in dict.fromkeys(spaces):
        counts = resolve_frozen_occupied_for_atoms(
            atoms, orbital_basis, occ_pair, space, root_id="rhf" if nspin == 1 else "uhf",
        )
        named = space in _STANDARD_CORE_ORBITALS
        if named and any(n and n == o for n, o in zip(counts, occ_pair, strict=True)):
            raise ValueError(
                f"Frozen-core {space!r} leaves no active occupied orbital in a spin channel "
                f"(frozen {counts}, occupied {occ_pair})."
            )
        projector = _atomic_core_projector(atoms, orbital_basis, overlap, space) if named and counts[0] else None
        spd_projector = (
            _atomic_core_projector(atoms, orbital_basis, overlap, DEFAULT_PRODUCTION_CORRELATION_SPACE)
            if projector is not None and space == "frozen-core"
            and _standard_frozen_count(atoms, orbital_basis, DEFAULT_PRODUCTION_CORRELATION_SPACE) > counts[0]
            else None
        )
        frozen_channels = []
        for k, (coeff, eps, occ) in enumerate(zip(coefficients, orbital_energies, occupied, strict=True)):
            spin = ("alpha", "beta")[k]
            nfrozen = counts[k]
            eps = np.asarray(eps[:occ])
            if not np.isfinite(eps).all():
                raise ValueError("Nonfinite orbital energies in the occupied frozen-core frame")
            if projector is None:
                gap = check_frozen_core_shell_gap(eps, nfrozen, correlation_space_id=space, spin_label=spin)
                channel = FrozenOrbitalChannel(tuple(range(nfrozen)), gap)
            else:
                c = coeff[:, :occ]
                channel = _identity_frozen_channel(
                    eps, c.T @ projector, nfrozen, space=space, spin=spin,
                    spd_projections=c.T @ spd_projector if spd_projector is not None else None,
                )
            frozen_channels.append(channel)
            order_by_spin[k].append(channel.indices)
        partitions.append((space, tuple(frozen_channels)))
    permutation = []
    for occ, sets in zip(occupied, order_by_spin, strict=True):
        indices = set()
        order = []
        for chosen in (*sets, tuple(range(occ))):
            if not indices.issubset(chosen):
                raise ValueError("Delta-CV requires the cv-active frozen selection to be a subset of the valence core")
            order.extend(i for i in chosen if i not in indices)
            indices.update(chosen)
        permutation.append(tuple(order))
    return FrozenOccupiedFrame(tuple(partitions), tuple(permutation))


def resolve_frozen_occupied_for_atoms(
    atoms: Sequence[Atom],
    orbital_basis: BasisSet,
    occupied: tuple[int, int],
    correlation_space_id: str,
    *,
    root_id: str | None = None,
    orbital_energies: np.ndarray
    | Sequence[float]
    | Sequence[np.ndarray | Sequence[float]]
    | None = None,
    min_gap_eh: float = DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH,
) -> tuple[int, int]:
    """Resolve (frozen_alpha, frozen_beta) for an atom list and orbital basis.

    Supported identifiers:
    - 'all-active': (0, 0)
    - 'frozen-core': standard chemical valence core per atom (minus ECP core pairs;
      4 frozen 3s3p/4s4p on Br/I ECP, keeping 3d/4d active)
    - 'frozen-core-spd': Hill-Peterson / TA13 valence-only core (9 frozen 3s3p3d/4s4p4d
      on Br/I ECP, correlating 7 valence electrons)
    - 'cv-active': outer-core + valence active (deep inner core frozen per atom)
    - 'frozen:<n>': freeze n lowest orbitals in both spin channels
    - 'frozen:<na>,<nb>': explicit per-spin frozen counts

    With `orbital_energies`, this count-only resolver validates the energy boundary
    of an explicit count. Chemical shell selection requires coefficients and overlap
    and uses `select_frozen_occupied_frame`.
    """
    spec = correlation_space_id
    oa, ob = occupied
    if spec == "all-active":
        return (0, 0)
    if spec in _STANDARD_CORE_ORBITALS:
        n_core = _standard_frozen_count(atoms, orbital_basis, spec)
        frozen = (n_core, n_core)
    elif spec.startswith("frozen:"):
        body = spec.removeprefix("frozen:")
        parts = body.split(",")
        if len(parts) == 1:
            n = int(parts[0])
            frozen = (n, n)
        elif len(parts) == 2:
            frozen = (int(parts[0]), int(parts[1]))
        else:
            raise ValueError(f"Invalid frozen correlation_space_id: {spec!r}")
    else:
        raise ValueError(
            f"Unsupported correlation_space_id {spec!r}; expected 'all-active', "
            "'frozen-core', 'frozen-core-spd', 'cv-active', or 'frozen:<n>'"
        )
    fa, fb = frozen
    if fa < 0 or fb < 0 or fa > oa or fb > ob:
        raise ValueError(
            f"Frozen orbitals {frozen} exceed occupied orbitals {occupied} for {spec!r}"
        )
    if root_id == "rhf" and fa != fb:
        raise ValueError(f"RHF requires equal alpha/beta frozen counts, got {frozen}")
    if orbital_energies is not None and (fa > 0 or fb > 0):
        if spec in _STANDARD_CORE_ORBITALS:
            raise ValueError("Chemical frozen-core selection requires orbital coefficients and overlap")
        eps_a, eps_b = _split_spin_orbital_energies(orbital_energies)
        channels = [("alpha", np.asarray(eps_a, dtype=np.float64), fa, oa)]
        if eps_b is not eps_a or fb != fa or ob != oa:
            channels.append(("beta", np.asarray(eps_b, dtype=np.float64), fb, ob))
        for spin_label, eps, n_frozen, n_occupied in channels:
            check_frozen_core_shell_gap(
                eps,
                n_frozen,
                min_gap_eh=min_gap_eh,
                n_occupied=n_occupied,
                correlation_space_id=spec,
                spin_label=spin_label,
            )
    return frozen


def resolve_frozen_occupied(
    request: FocalRequest,
    occupied: tuple[int, int],
    *,
    orbital_energies: np.ndarray
    | Sequence[float]
    | Sequence[np.ndarray | Sequence[float]]
    | None = None,
    min_gap_eh: float = DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH,
) -> tuple[int, int]:
    """Resolve (frozen_alpha, frozen_beta) from request.correlation_space_id.

    Supported identifiers:
    - 'all-active': (0, 0)
    - 'frozen-core': standard chemical valence core per atom (minus ECP core pairs;
      4 frozen 3s3p/4s4p on Br/I ECP, keeping 3d/4d active)
    - 'frozen-core-spd': Hill-Peterson / TA13 valence-only core (9 frozen 3s3p3d/4s4p4d
      on Br/I ECP, correlating 7 valence electrons)
    - 'cv-active': outer-core + valence active (deep inner core frozen per atom)
    - 'frozen:<n>': freeze n lowest orbitals in both spin channels
    - 'frozen:<na>,<nb>': explicit per-spin frozen counts
    """
    return resolve_frozen_occupied_for_atoms(
        request.atoms,
        request.basis.orbital,
        occupied,
        request.correlation_space_id,
        root_id=request.root_id,
        orbital_energies=orbital_energies,
        min_gap_eh=min_gap_eh,
    )


def active_occupied_counts(
    occupied: tuple[int, int], frozen: tuple[int, int]
) -> tuple[int, int]:
    """Return active (alpha, beta) occupied orbital counts after freezing."""
    return (occupied[0] - frozen[0], occupied[1] - frozen[1])


def default_production_f12_beta(cardinal: int) -> float:
    """Return the locked size-consistent Plan-A F12 geminal exponent beta (bohr^-1) for a cardinal."""
    if isinstance(cardinal, bool) or not isinstance(cardinal, int):
        raise ValueError(f"F12 cardinal must be an integer, got {cardinal!r}")
    if cardinal not in DEFAULT_PRODUCTION_F12_BETA_BY_CARDINAL:
        raise ValueError(
            f"Unsupported F12 cardinal {cardinal!r}; expected one of "
            f"{sorted(DEFAULT_PRODUCTION_F12_BETA_BY_CARDINAL)}"
        )
    return DEFAULT_PRODUCTION_F12_BETA_BY_CARDINAL[cardinal]


def _is_valid_valence_space_id(correlation_space_id: str) -> bool:
    if correlation_space_id in (
        "all-active",
        "frozen-core",
        DEFAULT_PRODUCTION_CORRELATION_SPACE,
    ):
        return True
    if correlation_space_id.startswith("frozen:"):
        parts = correlation_space_id.removeprefix("frozen:").split(",")
        if len(parts) not in (1, 2):
            return False
        try:
            return all(int(p) >= 0 for p in parts)
        except ValueError:
            return False
    return False


@dataclass(frozen=True)
class ProductionFocalRecipe:
    """Validated Plan-A focal recipe contract (matched core space and size-consistent F12 beta)."""

    valence_correlation_space: str
    delta_cv_valence_space: str | None
    delta_sr_correlation_space: str | None
    f12_beta_by_cardinal: dict[int, float]
    frozen_core_definition_version: int = FROZEN_CORE_DEFINITION_VERSION


@dataclass(frozen=True)
class ProductionFocalAssembly:
    """Validated additive focal total energy (Eh) and optional nuclear forces (Eh/bohr)."""

    recipe: ProductionFocalRecipe
    valence_energy_Eh: float
    delta_cv_energy_Eh: float
    delta_sr_energy_Eh: float
    total_energy_Eh: float
    valence_forces_Eh_per_bohr: np.ndarray | None = None
    delta_cv_forces_Eh_per_bohr: np.ndarray | None = None
    delta_sr_forces_Eh_per_bohr: np.ndarray | None = None
    total_forces_Eh_per_bohr: np.ndarray | None = None


def validate_production_focal_recipe(
    *,
    valence_correlation_space: str = DEFAULT_PRODUCTION_CORRELATION_SPACE,
    delta_cv_valence_space: str | None = None,
    delta_sr_correlation_space: str | None = None,
    f12_beta_by_cardinal: Mapping[int, float] | None = None,
    subsystem_f12_betas: Sequence[Mapping[int, float]] = (),
    require_production_defaults: bool = True,
    components: Sequence[object] = (),
) -> ProductionFocalRecipe:
    """Validate locked Plan-A focal correlation-space pairing and size-consistent F12 betas.

    Enforces:
    1. Valence, Delta-CV valence, and Delta-SR correlation spaces match identically
       (e.g., FC9 + Delta-CV9 + Delta-SR(FC9); mixing FC4 and FC9 raises ValueError).
    2. When require_production_defaults=True, valence correlation space must equal
       DEFAULT_PRODUCTION_CORRELATION_SPACE ('frozen-core-spd') and every declared
       F12 cardinal beta must match DEFAULT_PRODUCTION_F12_BETA_BY_CARDINAL
       ({2: 0.9, 3: 1.0, 4: 1.0}).
    3. Across all subsystems, every shared F12 cardinal must use an identical finite
       positive beta (rejecting mixed TZ beta=0.9 vs 1.0 across subsystems).
    4. No supplied result component is a diagnostic result or carries single-pass
       FP32 triples W forces executed elsewhere than on an NVIDIA L20.
    """
    reject_diagnostic_components(components)
    if not isinstance(valence_correlation_space, str) or not _is_valid_valence_space_id(
        valence_correlation_space
    ):
        raise ValueError(
            f"Unsupported valence correlation space {valence_correlation_space!r}; "
            f"expected {DEFAULT_PRODUCTION_CORRELATION_SPACE!r}, 'frozen-core', "
            "'all-active', or 'frozen:<n>'"
        )
    if require_production_defaults and (
        valence_correlation_space != DEFAULT_PRODUCTION_CORRELATION_SPACE
    ):
        raise ValueError(
            f"Production focal Plan A requires valence correlation space "
            f"{DEFAULT_PRODUCTION_CORRELATION_SPACE!r} (FC9), got {valence_correlation_space!r}; "
            "pass require_production_defaults=False only for explicit diagnostic comparisons"
        )
    if delta_cv_valence_space is not None:
        if delta_cv_valence_space != valence_correlation_space:
            raise ValueError(
                f"Mismatched valence ({valence_correlation_space!r}) and Delta-CV valence "
                f"({delta_cv_valence_space!r}) correlation spaces would double-count or omit "
                "core-valence correlation"
            )
    if delta_sr_correlation_space is not None:
        if delta_sr_correlation_space != valence_correlation_space:
            raise ValueError(
                f"Mismatched valence ({valence_correlation_space!r}) and Delta-SR "
                f"({delta_sr_correlation_space!r}) correlation spaces"
            )

    beta_maps: list[Mapping[int, float]] = []
    if f12_beta_by_cardinal is not None:
        beta_maps.append(f12_beta_by_cardinal)
    beta_maps.extend(subsystem_f12_betas)
    if not beta_maps:
        resolved_betas = dict(DEFAULT_PRODUCTION_F12_BETA_BY_CARDINAL)
    else:
        resolved_betas = {}
        for beta_map in beta_maps:
            if not beta_map:
                raise ValueError("F12 beta schedule mapping cannot be empty")
            for cardinal, raw_beta in beta_map.items():
                if isinstance(cardinal, bool) or not isinstance(cardinal, int) or cardinal < 2:
                    raise ValueError(f"Invalid F12 cardinal {cardinal!r}; expected integer >= 2")
                if isinstance(raw_beta, bool) or not isinstance(raw_beta, (int, float)):
                    raise ValueError(
                        f"Invalid F12 beta {raw_beta!r} for cardinal {cardinal}; expected finite float"
                    )
                beta = float(raw_beta)
                if not math.isfinite(beta) or beta <= 0.0:
                    raise ValueError(
                        f"F12 beta must be finite and positive for cardinal {cardinal}, got {raw_beta!r}"
                    )
                if cardinal in resolved_betas and not math.isclose(
                    resolved_betas[cardinal], beta, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"Inconsistent F12 beta at cardinal {cardinal} across subsystems: "
                        f"{resolved_betas[cardinal]!r} vs {beta!r} breaks size consistency"
                    )
                if require_production_defaults:
                    expected_beta = default_production_f12_beta(cardinal)
                    if not math.isclose(beta, expected_beta, rel_tol=0.0, abs_tol=1e-12):
                        raise ValueError(
                            f"Production focal Plan A requires F12 beta={expected_beta!r} at "
                            f"cardinal {cardinal} (DEFAULT_PRODUCTION_F12_BETA_BY_CARDINAL="
                            f"{DEFAULT_PRODUCTION_F12_BETA_BY_CARDINAL}), got {beta!r}"
                        )
                resolved_betas[cardinal] = beta

    return ProductionFocalRecipe(
        valence_correlation_space=valence_correlation_space,
        delta_cv_valence_space=delta_cv_valence_space,
        delta_sr_correlation_space=delta_sr_correlation_space,
        f12_beta_by_cardinal=dict(sorted(resolved_betas.items())),
    )


def _recomposed_valence(
    valence: FocalRunResult | FocalEnergyResult,
) -> tuple[float, np.ndarray | None, tuple[object, ...]]:
    """Return the valence energy, forces and components recomposed from the result itself.

    The stored totals must equal the weighted component sum exactly, as focal
    composition computes it, so totals cannot be paired with unrelated components.
    """
    from .experimental.f12_focal import (
        FocalEnergyComponent,
        FocalEnergyResult,
        FocalEvaluation,
        FocalRunResult,
        FocalTerm,
    )

    if not isinstance(valence, (FocalRunResult, FocalEnergyResult)):
        raise TypeError(
            "production focal assembly takes the composed valence FocalRunResult "
            "(energy and forces) or FocalEnergyResult (energy only)"
        )
    components = tuple(valence.components)
    if not components:
        raise ValueError(
            "production focal assembly requires the valence result components"
        )
    reject_diagnostic_components(components)
    if isinstance(valence, FocalEnergyResult):
        if not all(isinstance(c, FocalEnergyComponent) for c in components):
            raise TypeError(
                "valence FocalEnergyResult components must be FocalEnergyComponent"
            )
        if not all(c.converged is True for c in components):
            raise ValueError("valence components must be converged")
        energy = math.fsum(c.energy * c.weight for c in components)
        if energy != float(valence.energy):
            raise ValueError(
                "valence energy does not equal the weighted sum of its components"
            )
        return energy, None, components
    if not all(
        isinstance(c, FocalEvaluation) and isinstance(c.term, FocalTerm)
        for c in components
    ):
        raise TypeError(
            "valence FocalRunResult components must be FocalEvaluation with a FocalTerm"
        )
    if not all(c.converged is True for c in components):
        raise ValueError("valence components must be converged")
    weights = tuple(valence.weights)
    if len(weights) != len(components):
        raise ValueError("valence weights and components differ in length")
    first = components[0].term
    shape = np.shape(first.forces)
    energies = []
    force_terms = []
    for (role, basis_id, cardinal, coefficient), component in zip(
        weights, components, strict=True
    ):
        term = component.term
        if (term.role, term.basis_id, term.cardinal) != (role, basis_id, cardinal):
            raise ValueError("valence weights and components describe different terms")
        if term.surface_id != first.surface_id:
            raise ValueError("valence components must share one surface identity")
        value = float(term.energy)
        arr = np.asarray(term.forces, dtype=np.float64)
        if (
            len(shape) != 2
            or shape[1] != 3
            or arr.shape != shape
            or not math.isfinite(value)
            or not np.isfinite(arr).all()
        ):
            raise ValueError(
                "valence component energies and forces must be finite with one (natoms, 3) shape"
            )
        energies.append(value * coefficient)
        force_terms.append(arr * coefficient)
    energy = math.fsum(energies)
    forces = np.sum(force_terms, axis=0)
    stored = np.asarray(valence.forces, dtype=np.float64)
    if energy != float(valence.energy) or not np.array_equal(stored, forces):
        raise ValueError(
            "valence energy and forces do not equal the weighted sum of its components"
        )
    return energy, forces, components


def assemble_production_focal_energy_and_forces(
    *,
    valence: FocalRunResult | FocalEnergyResult,
    valence_correlation_space: str | None = None,
    f12_beta_by_cardinal: Mapping[int, float] | None = None,
    delta_cv: FocalDeltaCVResult | None = None,
    delta_sr: DeltaSrResult | None = None,
    delta_cv_energy: float | None = None,
    delta_cv_forces: np.ndarray | None = None,
    delta_cv_valence_space: str | None = None,
    delta_cv_frozen_core_definition_version: int | None = None,
    delta_sr_energy: float | None = None,
    delta_sr_forces: np.ndarray | None = None,
    delta_sr_correlation_space: str | None = None,
    delta_sr_frozen_core_definition_version: int | None = None,
    delta_sr_level: Literal["mp2", "hf"] = "mp2",
    subsystem_f12_betas: Sequence[Mapping[int, float]] = (),
    require_production_defaults: bool = True,
) -> ProductionFocalAssembly:
    """Validate the Plan-A recipe and assemble additive valence + Delta-CV + Delta-SR E and F.

    `valence` is the composed focal result: a `FocalRunResult` (or
    `FocalMolecularResult`) assembles energy and forces, a `FocalEnergyResult`
    assembles the energy only. The valence energy and forces are recomposed
    from its components and weights and must equal its stored totals. A result
    without components, a diagnostic component or single-pass FP32 triples W
    force telemetry without the device record of an NVIDIA L20 is refused.
    """
    val_e, val_f_arr, components = _recomposed_valence(valence)
    version = FROZEN_CORE_DEFINITION_VERSION
    if valence.frozen_core_definition_version != version or valence.valence_correlation_space is None:
        raise ValueError(f"valence result requires frozen-core definition version {version} and a correlation space")
    valence_space = valence.valence_correlation_space
    if valence_correlation_space is not None and valence_correlation_space != valence_space:
        raise ValueError("Explicit valence correlation space disagrees with the typed valence result")
    if delta_sr_level not in ("mp2", "hf"):
        raise ValueError(f"delta_sr_level must be 'mp2' or 'hf', got {delta_sr_level!r}")

    if delta_cv is not None:
        if any(x is not None for x in (delta_cv_energy, delta_cv_forces, delta_cv_frozen_core_definition_version)):
            raise ValueError("Supply either delta_cv result object or explicit delta_cv_energy/forces")
        if delta_cv.frozen_core_definition_version != version:
            raise ValueError(f"Delta-CV requires frozen-core definition version {version}")
        if delta_cv_valence_space is not None and delta_cv_valence_space != delta_cv.fc_mode:
            raise ValueError(
                f"Explicit delta_cv_valence_space {delta_cv_valence_space!r} disagrees with "
                f"delta_cv.fc_mode {delta_cv.fc_mode!r}"
            )
        cv_e = float(delta_cv.delta_mp2_f12_Eh)
        cv_f = delta_cv.delta_forces
        cv_space: str | None = delta_cv.fc_mode
    else:
        cv_e = 0.0 if delta_cv_energy is None else float(delta_cv_energy)
        cv_f = delta_cv_forces
        cv_space = delta_cv_valence_space
        if (delta_cv_energy is not None or delta_cv_forces is not None) and cv_space is None:
            raise ValueError(
                "Explicit delta_cv_energy/forces requires delta_cv_valence_space to verify core matching"
            )
        if (delta_cv_energy is not None or delta_cv_forces is not None) and delta_cv_frozen_core_definition_version != version:
            raise ValueError(f"Explicit Delta-CV requires frozen-core definition version {version}")

    if delta_sr is not None:
        if any(x is not None for x in (delta_sr_energy, delta_sr_forces, delta_sr_frozen_core_definition_version)):
            raise ValueError("Supply either delta_sr result object or explicit delta_sr_energy/forces")
        if delta_sr.frozen_core_definition_version != version:
            raise ValueError(f"Delta-SR requires frozen-core definition version {version}")
        if (
            delta_sr_correlation_space is not None
            and delta_sr_correlation_space != delta_sr.correlation_space
        ):
            raise ValueError(
                f"Explicit delta_sr_correlation_space {delta_sr_correlation_space!r} disagrees "
                f"with delta_sr.correlation_space {delta_sr.correlation_space!r}"
            )
        sr_e = float(delta_sr.delta_sr_mp2 if delta_sr_level == "mp2" else delta_sr.delta_sr_hf)
        sr_f = (
            delta_sr.delta_forces_mp2
            if delta_sr_level == "mp2"
            else delta_sr.delta_forces_hf
        )
        sr_space: str | None = delta_sr.correlation_space
    else:
        sr_e = 0.0 if delta_sr_energy is None else float(delta_sr_energy)
        sr_f = delta_sr_forces
        sr_space = delta_sr_correlation_space
        if (delta_sr_energy is not None or delta_sr_forces is not None) and sr_space is None:
            raise ValueError(
                "Explicit delta_sr_energy/forces requires delta_sr_correlation_space to verify core matching"
            )
        if (delta_sr_energy is not None or delta_sr_forces is not None) and delta_sr_frozen_core_definition_version != version:
            raise ValueError(f"Explicit Delta-SR requires frozen-core definition version {version}")

    recipe = validate_production_focal_recipe(
        valence_correlation_space=valence_space,
        delta_cv_valence_space=cv_space,
        delta_sr_correlation_space=sr_space,
        f12_beta_by_cardinal=f12_beta_by_cardinal,
        subsystem_f12_betas=subsystem_f12_betas,
        require_production_defaults=require_production_defaults,
        components=components,
    )

    if not (math.isfinite(val_e) and math.isfinite(cv_e) and math.isfinite(sr_e)):
        raise ValueError("Focal energy components must be finite")
    total_e = math.fsum((val_e, cv_e, sr_e))

    has_cv = delta_cv is not None or delta_cv_energy is not None or delta_cv_forces is not None
    has_sr = delta_sr is not None or delta_sr_energy is not None or delta_sr_forces is not None
    if val_f_arr is None:
        if cv_f is not None or sr_f is not None:
            raise ValueError(
                "Delta-CV or Delta-SR forces require a valence FocalRunResult with forces"
            )
        return ProductionFocalAssembly(
            recipe=recipe,
            valence_energy_Eh=val_e,
            delta_cv_energy_Eh=cv_e,
            delta_sr_energy_Eh=sr_e,
            total_energy_Eh=total_e,
        )

    if has_cv and cv_f is None:
        raise ValueError("Delta-CV energy was supplied without matching Delta-CV forces")
    if has_sr and sr_f is None:
        raise ValueError("Delta-SR energy was supplied without matching Delta-SR forces")

    cv_f_arr = (
        np.zeros_like(val_f_arr) if cv_f is None else np.asarray(cv_f, dtype=np.float64)
    )
    sr_f_arr = (
        np.zeros_like(val_f_arr) if sr_f is None else np.asarray(sr_f, dtype=np.float64)
    )
    for label, arr in (("delta_cv_forces", cv_f_arr), ("delta_sr_forces", sr_f_arr)):
        if arr.shape != val_f_arr.shape or not np.isfinite(arr).all():
            raise ValueError(
                f"{label} must be finite with shape {val_f_arr.shape}, got {arr.shape}"
            )
    total_f = val_f_arr + cv_f_arr + sr_f_arr
    return ProductionFocalAssembly(
        recipe=recipe,
        valence_energy_Eh=val_e,
        delta_cv_energy_Eh=cv_e,
        delta_sr_energy_Eh=sr_e,
        total_energy_Eh=total_e,
        valence_forces_Eh_per_bohr=val_f_arr.copy(),
        delta_cv_forces_Eh_per_bohr=cv_f_arr.copy(),
        delta_sr_forces_Eh_per_bohr=sr_f_arr.copy(),
        total_forces_Eh_per_bohr=total_f,
    )
