"""Native CABS singles component on an orthonormal occupied/extended-virtual frame."""

from __future__ import annotations

import ctypes
import math
from typing import NamedTuple

import numpy as np

from . import _native
from .elements import checked_integer


class CABSProjector(NamedTuple):
    kernel: np.ndarray
    frame: np.ndarray
    overlap_bar: np.ndarray | None
    rank: int
    cutoff_margin: float


def cabs_projector(
    overlap: np.ndarray,
    obs: int,
    *,
    cotangent: np.ndarray | None = None,
    relative_cutoff: float = 1e-9,
    branch_guard: float = 1e-12,
    expected_rank: int | None = None,
    device_id: int = 0,
    library: ctypes.CDLL | None = None,
) -> CABSProjector:
    """Form C_CABS, C_CABS C_CABS.T and the kernel overlap adjoint on CUDA.

    Input AO order is all orbitals followed by the atom-centred auxiliary
    functions. The complement is orthogonal to the entire OBS in this metric.
    The relative cutoff applies to the Gram matrix of the projected auxiliaries.
    Set expected_rank on displaced geometries to check branch continuity.
    A seed U requests the symmetric full-Frobenius derivative of <U,K>.
    The returned frame has shape (AO dimension, retained rank). Its columns
    are metric-orthonormal but do not define a smooth eigenvector gauge;
    propagate invariant frame weights through K, not individual eigenvectors.
    Physical probe/integral/SCF response is supplied by the molecular caller.
    """
    s = np.asarray(overlap)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.dtype != np.float64:
        raise ValueError("CABS projector requires a square real FP64 overlap")
    n = checked_integer(len(s), "overlap dimension", low=2)
    obs = checked_integer(obs, "OBS dimension", low=1, high=n - 1)
    device_id = checked_integer(device_id, "CUDA device")
    expected = (
        -1
        if expected_rank is None
        else checked_integer(expected_rank, "CABS rank", high=n - obs)
    )
    if (
        isinstance(relative_cutoff, (bool, np.bool_))
        or isinstance(branch_guard, (bool, np.bool_))
        or not 0 < branch_guard < relative_cutoff < 1
    ):
        raise ValueError("CABS requires 0 < branch_guard < relative_cutoff < 1")
    seed = None
    if cotangent is not None:
        seed = np.asarray(cotangent)
        if seed.shape != s.shape or seed.dtype != np.float64:
            raise ValueError("CABS cotangent must be FP64 with the overlap shape")
        seed = np.ascontiguousarray(seed)
    if library is None:
        library = _native.load_library(device="cuda", device_id=device_id)
    routine = library.generativeqc_f12_cabs_projector_cuda_v2
    ptr = ctypes.POINTER(ctypes.c_double)
    routine.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ptr,
        ptr,
        ptr,
        ptr,
        ctypes.c_size_t,
        ptr,
        ctypes.POINTER(ctypes.c_int),
        ptr,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    routine.restype = ctypes.c_int
    s = np.ascontiguousarray(s)
    k = np.empty_like(s)
    frame = np.empty(n * (n - obs), dtype=np.float64)
    bar = None if seed is None else np.empty_like(s)
    rank, margin = ctypes.c_int(), ctypes.c_double()
    error = ctypes.create_string_buffer(512)
    status = routine(
        device_id,
        n,
        obs,
        relative_cutoff,
        branch_guard,
        expected,
        s.ctypes.data_as(ptr),
        None if seed is None else seed.ctypes.data_as(ptr),
        k.ctypes.data_as(ptr),
        frame.ctypes.data_as(ptr),
        frame.size,
        None if bar is None else bar.ctypes.data_as(ptr),
        ctypes.byref(rank),
        ctypes.byref(margin),
        error,
        len(error),
    )
    if status:
        raise RuntimeError(error.value.decode(errors="replace"))
    frame = frame[: n * rank.value].reshape(n, rank.value)
    return CABSProjector(k, frame, bar, rank.value, margin.value)


def cabs_singles(
    fock: np.ndarray,
    occupied: int,
    *,
    spin_weight: int,
    device_id: int = 0,
    min_gap: float = 1e-12,
    pullback: bool = True,
    library: ctypes.CDLL | None = None,
) -> tuple[float, np.ndarray | None]:
    """Return the singles energy and its full Fock cotangent on CUDA.

    The occupied block comes first. The other block includes both orbital-basis
    virtuals and the complementary auxiliary space. Use weight 2 for RHF or
    sum two weight-1 calls for UHF, each on its own occupied/virtual frame.
    The cotangent holds this orthonormal frame fixed. Nuclear forces also
    require the geometry dependence of that frame and the stationary Fock.
    """
    matrix = np.asarray(fock)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.dtype != np.float64
    ):
        raise ValueError("CABS singles requires a square real FP64 Fock matrix")
    size = checked_integer(matrix.shape[0], "Fock dimension", low=1)
    occupied = checked_integer(occupied, "occupied orbitals", high=size)
    device_id = checked_integer(device_id, "CUDA device")
    spin_weight = checked_integer(spin_weight, "spin weight", low=1, high=2)
    if (
        not isinstance(pullback, bool)
        or isinstance(min_gap, bool)
        or not 0 < min_gap < math.inf
    ):
        raise ValueError(
            "CABS singles requires a positive finite gap tolerance and boolean pullback"
        )
    if library is None:
        library = _native.load_library(device="cuda", device_id=device_id)
    routine = library.generativeqc_f12_cabs_singles_cuda_v1
    pointer = ctypes.POINTER(ctypes.c_double)
    routine.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        pointer,
        pointer,
        pointer,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    routine.restype = ctypes.c_int
    matrix = np.ascontiguousarray(matrix)
    energy = ctypes.c_double()
    bar = np.empty_like(matrix) if pullback else None
    error = ctypes.create_string_buffer(512)
    status = routine(
        device_id,
        size,
        occupied,
        spin_weight,
        min_gap,
        matrix.ctypes.data_as(pointer),
        ctypes.byref(energy),
        None if bar is None else bar.ctypes.data_as(pointer),
        error,
        len(error),
    )
    if status:
        raise RuntimeError(error.value.decode(errors="replace"))
    return energy.value, bar
