"""Evaluate the symmetric positive-definite RI Coulomb metric whitener on CUDA."""

import ctypes as ct
import time
from pathlib import Path

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
) -> tuple[NDArray[np.float64], dict[str, int | float | str]]:
    """Return symmetric M^(-1/2) and diagnostics for the fixed full-rank branch."""
    a = np.ascontiguousarray(metric, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not 0 < len(a) < 2**31:
        raise ValueError("expected a nonempty square RI Coulomb metric")
    if not 0 <= device < 2**31 or not 0 < budget_bytes < 2**64:
        raise ValueError("invalid CUDA device or RI metric byte budget")
    lib = ct.CDLL(str(Path(library).resolve()))
    run = lib.gqc_f12_molecular_metric_whitener
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
    cutoff = 1e-10
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
        device_byte_scope="owned arrays and explicit cuSOLVER device workspace",
    )
    if status:
        raise RuntimeError(
            f"{err.value.decode()}: rank={measurement.rank}/{len(a)}, "
            f"relative_margin={measurement.relative_margin:g}, roundoff_guard={measurement.roundoff_guard:g}"
        )
    return output, fields
