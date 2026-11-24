"""Standard chemical valence frozen-core and explicit occupied-space resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .experimental.f12_focal import FocalRequest


def frozen_core_orbitals_for_atom(
    atomic_number: int, ecp_core_electrons: int = 0
) -> int:
    """Return the standard valence-correlation frozen spatial orbitals for one atom.

    Follows Peterson/Hill cc-pVXZ-F12 and cc-pVXZ-PP-F12 / W1-P34 conventions:
    - H, He (Z <= 2): 0
    - Li..Ne (3 <= Z <= 10): 1s (1 orbital) minus ECP core pairs
    - Na..Ar (11 <= Z <= 18): 1s2s2p (5 orbitals) minus ECP core pairs
    - K..Kr (19 <= Z <= 36): 1s2s2p3s3p (9 orbitals) minus ECP core pairs
      (for Br ECP10MDF with 10 ECP electrons = 5 pairs -> 4 frozen 3s3p orbitals,
       correlating 3d4s4p = 17 electrons)
    - Rb..Xe (37 <= Z <= 54): 1s..3d4s4p (18 orbitals) minus ECP core pairs
      (for I ECP28MDF with 28 ECP electrons = 14 pairs -> 4 frozen 4s4p orbitals,
       correlating 4d5s5p = 17 electrons, matching W1-P34)
    """
    if ecp_core_electrons < 0 or ecp_core_electrons % 2 != 0:
        raise ValueError(f"ECP core electron count must be even and nonnegative: {ecp_core_electrons}")
    ecp_pairs = ecp_core_electrons // 2
    if atomic_number <= 2:
        total_core = 0
    elif atomic_number <= 10:
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


def resolve_frozen_occupied(
    request: FocalRequest, occupied: tuple[int, int]
) -> tuple[int, int]:
    """Resolve (frozen_alpha, frozen_beta) from request.correlation_space_id.

    Supported identifiers:
    - 'all-active': (0, 0)
    - 'frozen-core': standard chemical valence core per atom (minus ECP core pairs)
    - 'frozen:<n>': freeze n lowest orbitals in both spin channels
    - 'frozen:<na>,<nb>': explicit per-spin frozen counts
    """
    spec = request.correlation_space_id
    oa, ob = occupied
    if spec == "all-active":
        return (0, 0)
    if spec == "frozen-core":
        n_core = sum(
            frozen_core_orbitals_for_atom(
                atom.atomic_number,
                request.basis.orbital.by_element[atom.atomic_number].ecp_core_electrons,
            )
            for atom in request.atoms
        )
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
            "'frozen-core', or 'frozen:<n>'"
        )
    fa, fb = frozen
    if fa < 0 or fb < 0 or fa > oa or fb > ob:
        raise ValueError(
            f"Frozen orbitals {frozen} exceed occupied orbitals {occupied} for {spec!r}"
        )
    if request.root_id == "rhf" and fa != fb:
        raise ValueError(f"RHF requires equal alpha/beta frozen counts, got {frozen}")
    return frozen


def active_occupied_counts(
    occupied: tuple[int, int], frozen: tuple[int, int]
) -> tuple[int, int]:
    """Return active (alpha, beta) occupied orbital counts after freezing."""
    return (occupied[0] - frozen[0], occupied[1] - frozen[1])
