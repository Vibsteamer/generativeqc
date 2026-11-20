"""Native real-spherical order and radial coefficients for focal molecular sources."""

from __future__ import annotations

import ctypes as ct
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .elements import SYMBOLS

if TYPE_CHECKING:
    from pathlib import Path

    from ._api_types import Atom
    from .basis import BasisSet
    from .experimental.f12_focal import FocalRequest


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


def focal_shell_layout(
    request: FocalRequest,
    library: Path,
    *,
    host_reference: bool,
    budget_mib: int,
    ordinary_fit: str | None = None,
    joint_jk: bool = False,
) -> dict:
    """Use OBS then OptRI rows in native real-spherical order; fitting rows start at zero.

    HF+CABS uses its JKFit Coulomb channels. Pair-correlation requests yield
    Gaussian integral sources in their declared F12 fit. Computing ordinary
    MP2 or forming a physical Fock requires its own declared fitting basis.
    """
    if request.backend != ("cpu_reference" if host_reference else "cuda"):
        raise ValueError("focal request and integral backend disagree")
    if not 0 < budget_mib <= (2**64 - 1) >> 20:
        raise ValueError("invalid integral memory budget")
    basis = request.basis
    if not request.atoms:
        raise ValueError("focal shell integrals require at least one atom")
    if joint_jk and (ordinary_fit is not None or request.role != "mp2_f12_correlation"):
        raise ValueError(
            "joint JKFit needs a molecular pair request with a combined orbital frame"
        )
    if ordinary_fit is not None:
        if (
            ordinary_fit not in ("jk_fit", "mp2_fit")
            or request.role != "mp2_f12_correlation"
        ):
            raise ValueError(
                "ordinary pair sources require an explicit JKFit or MP2Fit on the pair request"
            )
        fit = getattr(basis, ordinary_fit)
    elif request.role == "hf_cabs" or joint_jk:
        fit = basis.jk_fit
    elif request.role == "mp2_f12_correlation":
        fit = basis.f12_fit
        if basis.beta is None:
            raise ValueError("the geminal sources require an explicit F12 beta")
    else:
        raise ValueError("focal shell integrals require an HF+CABS or MP2-F12 role")
    xyz = np.asarray([atom.position for atom in request.atoms], dtype=np.float64)
    groups = [("orbital", basis.orbital)]
    if ordinary_fit is None:
        groups.append(("CABS", basis.cabs))
    groups.append(("fitting", fit))
    packed = pack_shells(request.atoms, tuple(groups), budget_mib << 20)
    axes = packed.angular
    angular = axes.tolist()
    ao_counts = packed.ao_counts
    ncombined = sum(packed.shell_counts[:-1])
    carts = [(l + 1) * (l + 2) // 2 for l in angular]
    combined, fitting = carts[:ncombined], carts[ncombined:]

    def triangular_size(sizes: list[int]) -> int:
        return (sum(sizes) ** 2 + sum(x * x for x in sizes)) // 2

    largest = max(triangular_size(fitting), triangular_size(combined) * sum(fitting))
    if largest >= 2**31 or largest * 8 > budget_mib << 20:
        raise MemoryError(
            "focal shell offsets or Cartesian integral tensor exceed capacity"
        )
    xyz_ids = packed.atom
    rows = np.full((len(axes), 2 * max(angular) + 1), -1, dtype=np.int32)
    for start, end in ((0, ncombined), (ncombined, len(axes))):
        offset = 0
        for i in range(start, end):
            n = 2 * angular[i] + 1
            rows[i, :n] = np.arange(offset, offset + n)
            offset += n
    layout = {
        "angular": axes,
        "atom": xyz_ids,
        "centers": np.ascontiguousarray(xyz[xyz_ids]),
        "prim_offsets": packed.prim_offsets,
        "exponents": packed.exponents,
        "radial": packed.radial,
        "spherical_rows": rows,
        "orbital_shell_count": np.int32(ncombined),
        "ao_sizes": np.array([sum(ao_counts[:-1]), ao_counts[-1]], dtype=np.int32),
        "obs_ao_count": ao_counts[0],
        "manifest": json.dumps(
            {
                "case": request.surface_id,
                "basis_id": basis.basis_id,
                "elements": [SYMBOLS[a.atomic_number - 1] for a in request.atoms],
                "geometry_bohr": xyz.tolist(),
                "options": {"f12_beta": basis.beta},
                "surface_id": request.surface_id,
                "fitting_basis_id": fit.identity,
            }
        ),
    }
    # Assign each fitting shell the same triangular set of combined-basis pairs.
    orbital_pairs = np.stack(np.tril_indices(ncombined), axis=1).astype(np.int32)
    nfit = len(fitting)
    metric_blocks = np.stack(np.tril_indices(nfit), axis=1).astype(np.int32) + ncombined
    ao_blocks = np.empty((nfit, len(orbital_pairs), 3), dtype=np.int32)
    ao_blocks[:, :, :2] = orbital_pairs
    ao_blocks[:, :, 2] = np.arange(ncombined, len(axes), dtype=np.int32)[:, None]
    cart_sizes = np.asarray(carts, dtype=np.int64)
    for group, blocks in (("metric", metric_blocks), ("ao", ao_blocks.reshape(-1, 3))):
        counts = np.ones(len(blocks), dtype=np.int64)
        for column in blocks.T:
            counts *= cart_sizes[column]
        layout[group + "_blocks"] = blocks
        layout[group + "_offsets"] = np.r_[0, np.cumsum(counts)].astype(np.int32)
    for l, transform in spherical_projections(library, set(angular)).items():
        layout[f"transform_{l}"] = transform
    return layout
