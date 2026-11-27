"""Tiled AO-to-MO orbital projection for CABS relaxation."""

from collections.abc import Callable
import ctypes as ct
from pathlib import Path

import numpy as np
from generativeqc_compiler.mp2.f12_density_fitting import molecular_orbital_sources
from generativeqc_compiler.tensor import Program

from ._f12_ao_mo_native import AoMoNativeOps, resolve_ao_mo_ops, run_ao_mo_project_fwd
from ._f12_spin_fock import _tile_groups, default_auxiliary_batch_size
from ._f12_spin_pairs import batch_execute


def project_ao_operator_tiled(
    nc: int,
    ext: int,
    occ: int,
    aux: int,
    field: str,
    ao_name: str,
    coefficients: np.ndarray,
    ao_mat: np.ndarray,
    run: Callable,
    *,
    batch_size: int | None = None,
    shell_library: Path | str | ct.CDLL | None = None,
    host_reference: bool | None = None,
    _ops: AoMoNativeOps | None = None,
) -> np.ndarray:
    """Transform one 3-center AO operator [aux, nc, nc] into the MO frame in bounded auxiliary tiles."""
    left_dim = ext if field == "raw_coulomb" else occ
    ops = (
        _ops
        if _ops is not None
        else resolve_ao_mo_ops(
            run,
            None,
            shell_library=shell_library,
            host_reference=host_reference,
        )
    )
    if ops is not None:
        return run_ao_mo_project_fwd(
            ops,
            nc,
            ext,
            left_dim,
            aux,
            coefficients,
            ao_mat,
            q_chunk_hint=0 if batch_size is None else batch_size,
        )
    width = (
        default_auxiliary_batch_size(aux, max(nc, ext))
        if batch_size is None
        else batch_size
    )
    coeffs = np.ascontiguousarray(coefficients)
    if width >= aux:
        frame = molecular_orbital_sources(nc, ext, occ, aux)
        graph = Program({field: frame.outputs[field]})
        return run(graph, {"coefficients": coeffs, ao_name: np.ascontiguousarray(ao_mat)}).outputs[field]
    out = np.empty((aux, left_dim, ext), dtype=np.float64)
    for tile_len, slices in _tile_groups(aux, width):
        frame = molecular_orbital_sources(nc, ext, occ, tile_len)
        graph = Program({field: frame.outputs[field]})
        with batch_execute(run, graph) as exec_one:
            for rows in slices:
                out[rows] = exec_one(
                    {"coefficients": coeffs, ao_name: np.ascontiguousarray(ao_mat[rows])}
                ).outputs[field]
    return out
