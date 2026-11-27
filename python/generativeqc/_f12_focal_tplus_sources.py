"""Occupied energy partitions and pair cotangents for focal MP2 and F12."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from ._f12_focal_reference import FocalReference

SpinOccupied: TypeAlias = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class OccupiedMP2F12:
    """Canonical spin-orbital MP2 and F12 increments, omitting HF+CABS singles.

    Eq. (9) contains OBS singles for a general reference. Their value and first
    differential vanish at the canonical stationary RHF or UHF used here.
    The F12 increment follows the caller's pair model. The focal implementation
    supplies 3C(FIX), while the article evaluated a 2B fixed-amplitude variant.
    """

    mp2: SpinOccupied
    f12_increment: SpinOccupied

    def restricted_spatial(self) -> tuple[np.ndarray, np.ndarray]:
        """Group paired spin contributions in the same RHF orbital order."""
        if self.mp2[0].shape != self.mp2[1].shape:
            raise ValueError("a restricted partition requires paired occupied spaces")
        return sum(self.mp2), sum(self.f12_increment)


def checked_occupied_weights(
    weights: SpinOccupied, occupied: tuple[int, int]
) -> SpinOccupied:
    if len(weights) != 2 or any(
        not isinstance(block, np.ndarray)
        or block.shape != (count,)
        or block.dtype != np.float64
        or not np.isfinite(block).all()
        for block, count in zip(weights, occupied, strict=True)
    ):
        raise ValueError("occupied sources require a finite FP64 vector for each spin")
    return weights


def add_distinct_pair(
    partition: SpinOccupied,
    spins: tuple[str, str],
    indices: tuple[int, int],
    energy: float,
) -> None:
    """Assign an unordered spin pair's value equally to the two occupied legs."""
    for spin, i in zip(spins, indices, strict=True):
        partition[spin == "beta"][i] += 0.5 * energy


def add_occupied_pair_matrix(
    left: np.ndarray, right: np.ndarray, pair_energies: np.ndarray
) -> None:
    """Distribute the matrix entries of the already weighted ordered MP2 sum."""
    left += 0.5 * pair_energies.sum(axis=1)
    right += 0.5 * pair_energies.sum(axis=0)


def occupied_pair_energy_seed(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Transpose of the occupied MP2 marginals, prior to the pair graph VJP."""
    return (left[:, None] + right[None, :]) * 0.5
