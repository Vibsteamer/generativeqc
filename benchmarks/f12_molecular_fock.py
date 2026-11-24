"""Preserve FP64 precision in the physical extended-frame DF Fock projection."""

import ctypes as ct
from pathlib import Path
import time

import numpy as np


def compensated_fock(
    hcore, coulomb, coefficients, fit, exchange, *, library: Path, budget_bytes: int
):
    """Compute C.T (h + 2 J) C - K with twofold AO contractions on CUDA."""
    arrays = [np.asarray(v) for v in (hcore, coulomb, coefficients, fit, exchange)]
    if any(a.dtype != np.float64 for a in arrays):
        raise ValueError("extended Fock requires FP64 tensors")
    if coefficients.ndim != 2 or fit.ndim != 3:
        raise ValueError(
            "extended Fock requires orbital coefficients and a three-index occupied fit"
        )
    ao, mo = coefficients.shape
    aux, occ, orbitals = fit.shape
    if (
        any(n <= 0 or n >= 2**31 for n in (ao, mo, occ, aux))
        or mo > ao
        or occ >= mo
        or orbitals != mo
        or hcore.shape != (ao, ao)
        or coulomb.shape != (aux, ao, ao)
        or exchange.shape != (mo, mo)
        or not 0 < budget_bytes < 2**64
    ):
        raise ValueError("inconsistent extended Fock dimensions or budget")
    lib = ct.CDLL(str(library.resolve()))
    routine = lib.gqc_f12_extended_fock
    tensor = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    routine.argtypes = (
        [ct.c_int] * 5
        + [tensor] * 5
        + [
            ct.c_uint64,
            tensor,
            ct.POINTER(ct.c_uint64),
            ct.POINTER(ct.c_char),
            ct.c_size_t,
        ]
    )
    routine.restype = ct.c_int
    result = np.empty((mo, mo))
    count = ct.c_uint64()
    error = ct.create_string_buffer(512)
    started = time.monotonic()
    status = routine(
        0,
        ao,
        mo,
        occ,
        aux,
        *[np.ascontiguousarray(v) for v in arrays],
        budget_bytes,
        result,
        ct.byref(count),
        error,
        len(error),
    )
    if status:
        raise RuntimeError(error.value.decode(errors="replace"))
    return result, {
        "backend": "cuda_fp64_compensated",
        "device_bytes": count.value,
        "seconds": time.monotonic() - started,
    }
