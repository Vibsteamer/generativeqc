"""Private native canonical constraint multipliers, before physical SCF response."""

from __future__ import annotations

import ctypes
import math

import numpy as np

from . import _native
from .elements import checked_integer


def canonical_frame_seed(
    rotation_gradient: np.ndarray,
    orbital_energies: np.ndarray,
    occupied: int,
    *,
    gap_threshold: float = 1e-8,
    stationarity_threshold: float = 1e-9,
    maximum_bytes: int = 128 << 20,
    device_id: int = 0,
    library: ctypes.CDLL | None = None,
) -> tuple[np.ndarray, int]:
    """Return symmetric full-Frobenius Fock multipliers and device bytes.

    M=C.T@barC must include every source weight at the same canonical reference.
    This consumes same-space rotations only; the physical ov Z solve and Pulay
    response still follow. A nonstationary small-gap block is refused, never
    regularized silently. All pair algebra and branch checks run on CUDA.
    """
    seed, _, _, size = _frame_seeds(
        rotation_gradient,
        orbital_energies,
        occupied,
        gap_threshold,
        stationarity_threshold,
        maximum_bytes,
        device_id,
        library,
        full=False,
    )
    return seed, size


def canonical_full_frame_seeds(
    rotation_gradient: np.ndarray,
    orbital_energies: np.ndarray,
    occupied: int,
    *,
    gap_threshold: float = 1e-8,
    stationarity_threshold: float = 1e-9,
    maximum_bytes: int = 128 << 20,
    device_id: int = 0,
    library: ctypes.CDLL | None = None,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Resolve every OBS pair and return Fock/overlap seeds, gauge torque and device bytes.

    The two seeds are zero on the diagonal and inside a stationary small-gap
    block. The norm of the unresolved antisymmetric torque counts each unique
    pair once. A small occupied/virtual gap requires a resolved occupation root.
    """
    seed, overlap, norm, size = _frame_seeds(
        rotation_gradient,
        orbital_energies,
        occupied,
        gap_threshold,
        stationarity_threshold,
        maximum_bytes,
        device_id,
        library,
        full=True,
    )
    return seed, overlap, norm, size


def _frame_seeds(
    rotation_gradient: np.ndarray,
    orbital_energies: np.ndarray,
    occupied: int,
    gap_threshold: float,
    stationarity_threshold: float,
    maximum_bytes: int,
    device_id: int,
    library: ctypes.CDLL | None,
    *,
    full: bool,
) -> tuple[np.ndarray, np.ndarray | None, float, int]:
    m, eps = np.asarray(rotation_gradient), np.asarray(orbital_energies)
    if (
        m.ndim != 2
        or m.shape[0] != m.shape[1]
        or m.dtype != np.float64
        or eps.shape != (len(m),)
        or eps.dtype != np.float64
    ):
        raise ValueError(
            "Canonical frame requires square FP64 M and matching FP64 energies"
        )
    n = checked_integer(len(m), "canonical dimension", low=2, high=2**31 - 1)
    occupied = checked_integer(
        occupied, "occupied orbitals", low=0 if full else 1, high=n - 1
    )
    device_id = checked_integer(device_id, "CUDA device")
    budget = checked_integer(
        maximum_bytes, "numeric byte budget", low=1, high=2**63 - 1
    )
    if any(
        isinstance(x, (bool, np.bool_)) or not math.isfinite(x) or x <= 0
        for x in (gap_threshold, stationarity_threshold)
    ):
        raise ValueError("Canonical response thresholds must be positive and finite")
    if library is None:
        library = _native.load_library(device="cuda", device_id=device_id)
    routine = (
        library.generativeqc_canonical_full_frame_seeds_cuda_v1
        if full
        else library.generativeqc_canonical_frame_seed_cuda_v1
    )
    ptr = ctypes.POINTER(ctypes.c_double)
    routine.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_size_t,
        ptr,
        ptr,
        ptr,
        *([ptr, ptr] if full else []),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_size_t,
    ]
    routine.restype = ctypes.c_int
    m, eps = np.ascontiguousarray(m), np.ascontiguousarray(eps)
    result = np.empty_like(m)
    overlap = np.empty_like(m) if full else None
    norm = ctypes.c_double()
    extra = (
        [overlap.ctypes.data_as(ptr), ctypes.byref(norm)] if overlap is not None else []
    )
    size = ctypes.c_size_t()
    error = ctypes.create_string_buffer(512)
    status = routine(
        device_id,
        n,
        occupied,
        gap_threshold,
        stationarity_threshold,
        budget,
        eps.ctypes.data_as(ptr),
        m.ctypes.data_as(ptr),
        result.ctypes.data_as(ptr),
        *extra,
        ctypes.byref(size),
        error,
        len(error),
    )
    if status:
        raise RuntimeError(error.value.decode(errors="replace"))
    return result, overlap, norm.value, size.value
