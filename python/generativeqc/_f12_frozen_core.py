"""Standard chemical valence frozen-core and explicit occupied-space resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ._api_types import Atom
    from .basis import BasisSet, ElementBasis
    from .experimental.f12_focal import FocalRequest


DEFAULT_PRODUCTION_CORRELATION_SPACE = "frozen-core-spd"
DEFAULT_FROZEN_CORE_MIN_SHELL_GAP_EH = 0.05
FROZEN_CORE_DEFINITION_VERSION = 2
_MIN_CORE_OVERLAP = 0.9


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
