"""Native real-spherical order and radial coefficients for focal molecular sources."""

from __future__ import annotations

import ctypes as ct
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from ._api_types import Atom
    from .basis import BasisSet


@dataclass(frozen=True)
class FocalShells:
    angular: np.ndarray
    atom: np.ndarray
    prim_offsets: np.ndarray
    exponents: np.ndarray
    radial: np.ndarray
    shell_counts: tuple[int, ...]
    ao_counts: tuple[int, ...]


def pack_shells(
    atoms: tuple[Atom, ...],
    roles: tuple[tuple[str, BasisSet | None], ...],
    budget_bytes: int,
) -> FocalShells:
    """Expand contractions in the declared role, atom, shell and contraction order."""
    angular, atom_ids, exponents, radial = [], [], [], []
    prim_offsets = [0]
    shell_counts, ao_counts = [], []
    for name, record in roles:
        if record is None or record.representation != "spherical":
            raise ValueError(f"{name} requires an explicit real-spherical basis")
        elements = record.by_element
        before = len(angular)
        for index, atom in enumerate(atoms):
            if atom.atomic_number not in elements:
                raise ValueError(f"{name} lacks element {atom.atomic_number}")
            for shell in elements[atom.atomic_number].shells:
                l = shell.angular_momentum
                if l > 6:
                    raise ValueError(
                        "the integral sink supports angular momentum through i"
                    )
                e = np.asarray(shell.exponents, dtype=np.float64)
                if 8 * e.size**2 > budget_bytes:
                    raise MemoryError(
                        "primitive contraction overlap exceeds the integral budget"
                    )
                overlap = (2 * np.sqrt(e[:, None]) * np.sqrt(e) / (e[:, None] + e)) ** (
                    l + 1.5
                )
                scale = (2 * e / np.pi) ** 0.75 * (4 * e) ** (0.5 * l)
                axial = math.prod(range(1, 2 * l + 1, 2))
                for coeffs in shell.coefficients:
                    c = np.asarray(coeffs, dtype=np.float64)
                    norm = float(c @ overlap @ c)
                    if not np.isfinite(norm) or norm <= 0:
                        raise ValueError("invalid contraction norm in focal shell")
                    values = c * scale / math.sqrt(norm * axial)
                    if not np.isfinite(values).all():
                        raise ValueError("nonfinite normalized focal shell")
                    radial.extend(values)
                    exponents.extend(e)
                    angular.append(l)
                    atom_ids.append(index)
                    prim_offsets.append(len(exponents))
        shell_counts.append(len(angular) - before)
        ao_counts.append(sum(2 * l + 1 for l in angular[before:]))

    if len(exponents) >= 2**31:
        raise MemoryError("focal primitive offsets exceed native capacity")
    return FocalShells(
        np.asarray(angular, dtype=np.int32),
        np.asarray(atom_ids, dtype=np.int32),
        np.asarray(prim_offsets, dtype=np.int32),
        np.asarray(exponents),
        np.asarray(radial),
        tuple(shell_counts),
        tuple(ao_counts),
    )


def spherical_projections(library: Path, angular: set[int]) -> dict[int, np.ndarray]:
    """Read the through-i real solid harmonics in the native sink normalization."""
    native = ct.CDLL(str(library.resolve())).gqc_f12_raw_spherical_projection
    native.argtypes = [
        ct.c_uint,
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        ct.c_size_t,
    ]
    native.restype = ct.c_int
    result = {}
    for l in sorted(angular):
        transform = np.empty((2 * l + 1, (l + 1) * (l + 2) // 2))
        if native(l, transform, transform.size):
            raise RuntimeError(f"native spherical projection failed for l={l}")
        result[l] = transform
    return result
