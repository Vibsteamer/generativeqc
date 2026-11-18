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


def qualify_metric_whitener(library: str | Path) -> dict[str, object]:
    """Check a rotated, positive-definite matrix and both sides of the rank boundary."""
    rotation = np.array([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
    eigen = np.array([0.25, 1.0, 4.0])
    metric = (rotation * eigen) @ rotation.T
    ref = (rotation / np.sqrt(eigen)) @ rotation.T
    observed, info = metric_whitener(metric, library)
    error = float(np.max(np.abs(observed - ref)))
    residual = float(np.max(np.abs(observed @ metric @ observed - np.eye(3))))
    if (
        error > 2e-13
        or residual > 2e-13
        or info["rank"] != 3
        or not info["device_bytes"]
    ):
        raise ArithmeticError("analytic rotated positive-definite metric failed")
    near = np.diag([1e-10 + 1e-12, 1.0])
    above, rank_info = metric_whitener(near, library)
    if (
        rank_info["rank"] != 2
        or np.max(np.abs(above @ near @ above - np.eye(2))) > 2e-13
    ):
        raise ArithmeticError("well-resolved full-rank metric was rejected")
    failures = {}
    for label, value, budget, diagnostic in [
        ("indefinite", np.diag([-0.5, 1.0]), 512 << 20, "full-rank"),
        ("truncated", np.diag([0.5e-10, 1.0]), 512 << 20, "full-rank"),
        ("at_cutoff", np.diag([1e-10, 1.0]), 512 << 20, "full-rank"),
        (
            "roundoff_band",
            np.diag([1e-10 + np.finfo(float).eps, 1.0]),
            512 << 20,
            "full-rank",
        ),
        ("budget", metric, 1, "budget"),
        ("asymmetric", np.array([[1.0, 0.25], [0.0, 1.0]]), 512 << 20, "symmetric"),
    ]:
        try:
            metric_whitener(value, library, budget_bytes=budget)
        except RuntimeError as exc:
            if diagnostic not in str(exc):
                raise
            failures[label] = str(exc)
        else:
            raise AssertionError(f"unexpected acceptance of {label} metric")
    return {
        "passed": True,
        "analytic_whitener_error": error,
        "identity_residual": residual,
        "resources": info,
        "above_cutoff": rank_info,
        "rejected": failures,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(qualify_metric_whitener(args.library), indent=2, allow_nan=False)
        + "\n"
    )
