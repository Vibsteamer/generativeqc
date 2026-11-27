"""Geometry input and checked CUDA ABI for the focal scalar ECP projectors."""

from __future__ import annotations

import ctypes as ct
from typing import TYPE_CHECKING

import numpy as np
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.integral.ecp_grid import harmonic_roots, radial_map_roots

from ._f12_focal_ecp import _nodes

if TYPE_CHECKING:
    from pathlib import Path

    from ._f12_focal_shells import FocalShells


def cuda_projectors(
    library: Path,
    shells: FocalShells,
    transforms: dict,
    xyz: np.ndarray,
    terms: dict,
    grid: tuple[int, int],
    derivatives: bool,
    budget: int,
) -> tuple[np.ndarray, int]:
    """Prepare only fixed quadrature constants and owned scientific input on the host."""
    lib = ct.CDLL(str(library.resolve()))
    try:
        native = lib.gqc_f12_focal_ecp_projectors
    except AttributeError as e:
        raise RuntimeError(
            "F12 library lacks its CUDA scalar ECP projector kernel"
        ) from e
    d = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    i = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
    native.argtypes = [ct.c_int] * 7 + [
        i,
        d,
        d,
        d,
        d,
        i,
        d,
        d,
        d,
        d,
        d,
        d,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.c_size_t),
    ]
    native.restype = ct.c_int
    n = sum(shells.ao_counts)
    dims = (1 + 3 * len(xyz) * int(derivatives), n, n)
    nr, npolar = grid
    nq = 2 * npolar**2
    # Host rule construction, contiguous immutable ABI inputs and two host
    # result matrices are accounted separately from the bounded device arena.
    host_bytes = 8 * (2 * int(np.prod(dims)) + nq * 64 + n * 32 + 15 * nr)
    if budget <= host_bytes:
        raise MemoryError("focal CUDA ECP inputs exceed the numeric budget")
    entries = [
        (atom, l, p, a, c) for atom, items in terms.items() for l, p, a, c in items
    ]
    if not entries:
        return np.zeros(dims), 0
    term = np.ascontiguousarray([x[:3] for x in entries], dtype=np.int32)
    param = np.ascontiguousarray([x[3:] for x in entries], dtype=np.float64)
    row = np.empty((n, 4), dtype=np.int32)
    angular = np.zeros((n, 28), dtype=np.float64)
    offset = 0
    for j, (l, at) in enumerate(zip(shells.angular, shells.atom, strict=True)):
        width = 2 * l + 1
        rows = slice(offset, offset + width)
        row[rows] = [l, at, *shells.prim_offsets[j : j + 2]]
        angular[rows, : transforms[l].shape[1]] = transforms[l]
        offset += width
    # The Gauss angular grid is independent of all molecular coordinates.
    # Device kernels evaluate the shifted AOs, their jets and every term's
    # radial potential, including all products and nuclear scatter.
    z, wz = _nodes(npolar)
    phi = np.arange(2 * npolar) * np.pi / npolar
    h = np.sqrt(1 - z[:, None] ** 2)
    direction = np.stack(
        np.broadcast_arrays(h * np.cos(phi), h * np.sin(phi), z[:, None]), axis=-1
    ).reshape(nq, 3)
    graph, roots = harmonic_roots()
    fixed_harmonics = np.stack(
        [
            np.broadcast_to(x, (nq,))
            for x in evaluate_array_graph(
                graph, roots, dict(zip("xyz", direction.T, strict=True))
            )
        ],
        axis=1,
    )
    fixed_harmonics *= np.repeat(wz, 2 * npolar)[:, None] * (np.pi / npolar)
    graph, roots = radial_map_roots()
    t, wt = _nodes(nr)
    radii, dr = evaluate_array_graph(graph, roots, {"z": t, "weight": wt})
    result = np.empty(dims)
    used = ct.c_size_t()
    rc = native(
        int(derivatives),
        len(xyz),
        n,
        len(shells.radial),
        len(term),
        nr,
        nq,
        row,
        xyz,
        shells.exponents,
        shells.radial,
        angular,
        term,
        param,
        radii,
        dr,
        direction,
        fixed_harmonics,
        result,
        result.size,
        budget - host_bytes,
        ct.byref(used),
    )
    if rc == 2:
        raise MemoryError("focal CUDA ECP radial tile exceeds the numeric budget")
    if rc or not 0 < used.value <= budget - host_bytes:
        raise RuntimeError(f"focal CUDA ECP projector kernel failed (status {rc})")
    return result, used.value


def add_cuda_projector_residual(
    hcore: np.ndarray,
    library: Path,
    shells: FocalShells,
    transforms: dict,
    xyz: np.ndarray,
    terms: dict,
    derivatives: bool,
    budget: int,
) -> int:
    from generativeqc_compiler.integral.ecp_policy import (
        EXTENDED_POLAR_POINTS,
        EXTENDED_RADIAL_POINTS,
        MAX_CUDA_POLAR_POINTS,
        MAX_CUDA_RADIAL_POINTS,
        ORBITAL_G_POLAR_POINTS,
        ORBITAL_G_RADIAL_POINTS,
    )

    from ._f12_focal_ecp import check_projector_grids

    coarse, w1 = cuda_projectors(
        library,
        shells,
        transforms,
        xyz,
        terms,
        (EXTENDED_RADIAL_POINTS, EXTENDED_POLAR_POINTS),
        derivatives,
        budget,
    )
    fine, w2 = cuda_projectors(
        library,
        shells,
        transforms,
        xyz,
        terms,
        (MAX_CUDA_RADIAL_POINTS, MAX_CUDA_POLAR_POINTS),
        derivatives,
        budget,
    )
    max_w = max(w1, w2)
    try:
        check_projector_grids(coarse, fine)
    except (ValueError, ArithmeticError):
        refined, w3 = cuda_projectors(
            library,
            shells,
            transforms,
            xyz,
            terms,
            (ORBITAL_G_RADIAL_POINTS, ORBITAL_G_POLAR_POINTS),
            derivatives,
            budget,
        )
        max_w = max(max_w, w3)
        check_projector_grids(fine, refined)
        fine = refined
    hcore += fine
    return max_w
