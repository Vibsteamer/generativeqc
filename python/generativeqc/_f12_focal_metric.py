"""Evaluate full-rank positive-definite RI Coulomb metric factors on CUDA."""

import ctypes as ct
import time
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


class _MetricInfo(ct.Structure):
    _fields_ = [
        (k, ct.c_uint64)
        for k in (
            "device_bytes",
            "solver_device_workspace_bytes",
            "solver_host_workspace_bytes",
            "host_to_device_bytes",
            "device_to_host_bytes",
            "rank",
        )
    ] + [
        (k, ct.c_double)
        for k in ("smallest", "largest", "relative_margin", "roundoff_guard")
    ]


def metric_whitener(
    metric: ArrayLike,
    library: str | Path,
    budget_bytes: int = 512 << 20,
    *,
    device: int = 0,
    cutoff: float = 1e-10,
    factorization: Literal["symmetric", "inverse_cholesky"] = "symmetric",
) -> tuple[NDArray[np.float64], dict[str, int | float | str]]:
    """Return W and diagnostics with W M W.T = I on the fixed full-rank branch."""
    a = np.ascontiguousarray(metric, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not 0 < len(a) < 2**31:
        raise ValueError("expected a nonempty square RI Coulomb metric")
    if not 0 <= device < 2**31 or not 0 < budget_bytes < 2**64:
        raise ValueError("invalid CUDA device or RI metric byte budget")
    if factorization == "symmetric":
        symbol = "gqc_f12_molecular_metric_whitener"
    elif factorization == "inverse_cholesky":
        symbol = "gqc_f12_molecular_metric_inverse_cholesky_v1"
    else:
        raise ValueError(f"unknown RI Coulomb metric factorization {factorization!r}")
    lib = ct.CDLL(str(Path(library).resolve()))
    try:
        run = getattr(lib, symbol)
    except AttributeError as exc:
        raise RuntimeError(f"CUDA RI metric library {library} requires symbol {symbol}") from exc
    matrix = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
    run.argtypes = [
        ct.c_int,
        ct.c_int,
        matrix,
        ct.c_double,
        ct.c_uint64,
        matrix,
        ct.POINTER(_MetricInfo),
        ct.POINTER(ct.c_char),
        ct.c_size_t,
    ]
    run.restype = ct.c_int
    output = np.empty_like(a)
    err = ct.create_string_buffer(512)
    measurement = _MetricInfo()
    cutoff = float(cutoff)
    start = time.monotonic()
    status = run(
        device,
        len(a),
        a,
        cutoff,
        budget_bytes,
        output,
        ct.byref(measurement),
        err,
        len(err),
    )
    fields = {k: getattr(measurement, k) for k, _ in measurement._fields_}
    fields.update(
        seconds=time.monotonic() - start,
        relative_threshold=cutoff,
        backend="cuda_fp64",
        native_symbol=symbol,
        factorization=factorization,
        device_byte_scope="owned arrays and explicit cuSOLVER/cuBLAS device workspace",
    )
    if status:
        raise RuntimeError(
            f"{err.value.decode()}: rank={measurement.rank}/{len(a)}, "
            f"relative_margin={measurement.relative_margin:g}, roundoff_guard={measurement.roundoff_guard:g}"
        )
    return output, fields
