"""Occupied energy partitions and pair cotangents for focal MP2 and F12."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from ._f12_focal_reference import FocalReference

SpinOccupied: TypeAlias = tuple[np.ndarray, np.ndarray]


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


def occupied_pair_energy_seed(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Transpose of the occupied MP2 marginals, prior to the pair graph VJP."""
    return (left[:, None] + right[None, :]) * 0.5
