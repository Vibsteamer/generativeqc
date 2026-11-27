"""Selected retained 3C(FIX) F12 increment on a canonical all-active RHF or UHF focal root."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_cabs import CABSProjector
    from ._f12_focal_one_electron import FocalOneElectron
    from ._f12_focal_reference import FocalReference
    from ._f12_focal_tplus_sources import SpinOccupied
    from ._f12_unrestricted import UnrestrictedF12Energy
    from .experimental.f12_focal import FocalRequest


@dataclass(frozen=True)
class FocalF12Sources:
    """Joint OBS/OptRI spherical sources for a physical fit and F12 exponent."""

    surface_id: str
    basis_id: str
    fit_id: str
    executed_backend: str
    beta: float | None
    integrals: Mapping[str, np.ndarray]
    whitener: np.ndarray
    ao_loader: Callable[[str], np.ndarray] | None = None
    obs_coulomb: np.ndarray | None = None


def _canonical_occupied_virtual_blocks(
    m: np.ndarray, eps: np.ndarray, occ: int
) -> np.ndarray:
    """Reference same-space multiplier of the canonical Fock constraint."""
    gap = eps[:, None] - eps
    groups = np.arange(len(eps)) < occ
    same = groups[:, None] == groups
    torque = m - m.T
    unresolved = same & (np.abs(gap) <= 1e-8)
    if np.any(np.abs(torque[unresolved]) > 1e-9):
        raise ValueError("F12 increment depends on unresolved canonical gauge")
    return np.divide(-0.5 * torque, gap, out=np.zeros_like(m), where=same & ~unresolved)
