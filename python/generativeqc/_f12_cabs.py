"""Native CABS singles component on an orthonormal occupied/extended-virtual frame."""

from __future__ import annotations

import ctypes
import math

import numpy as np

from . import _native
from .elements import checked_integer


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
