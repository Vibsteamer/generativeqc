"""Qualify the two full-rank CUDA Coulomb metric whitener contracts."""

import ctypes as ct
from functools import partial
from pathlib import Path

import numpy as np
from generativeqc._f12_focal_metric import _MetricInfo, metric_whitener
from numpy.typing import ArrayLike


def _inverse_cholesky_rejection(
    library: str | Path,
    metric: ArrayLike,
    budget: int,
    diagnostic: str,
    cutoff: float = 1e-10,
) -> dict[str, int | str]:
    """Call the raw ABI to verify that rejected requests preserve every output byte."""
    run = ct.CDLL(
        str(Path(library).resolve())
    ).gqc_f12_molecular_metric_inverse_cholesky_v1
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
    metric = np.ascontiguousarray(metric, dtype=np.float64)
    output = np.arange(metric.size, dtype=np.float64).reshape(metric.shape) + 17.25
    original = output.tobytes()
    report = _MetricInfo()
    err = ct.create_string_buffer(512)
    status = run(
        0, len(metric), metric, cutoff, budget, output, ct.byref(report), err, len(err)
    )
    error = err.value.decode()
    if not status or diagnostic not in error or original != output.tobytes():
        raise AssertionError(
            f"expected {diagnostic!r} rejection with preserved output, got {status}, {error!r}"
        )
    return {
        "error": error,
        "rank": report.rank,
        "required_device_bytes": report.device_bytes,
    }


def qualify_inverse_cholesky_preflight(
    library: str | Path,
) -> dict[str, dict[str, int | str]]:
    """These input and base-capacity checks run before any CUDA device selection."""
    return {
        "asymmetric": _inverse_cholesky_rejection(
            library, [[1, 0.25], [0, 1]], 512 << 20, "symmetric"
        ),
        "nonfinite": _inverse_cholesky_rejection(
            library, [[np.inf, 0], [0, 1]], 512 << 20, "finite"
        ),
        "budget": _inverse_cholesky_rejection(library, np.eye(2), 1, "budget"),
        "cutoff": _inverse_cholesky_rejection(
            library, np.eye(2), 512 << 20, "threshold", 0
        ),
    }


def qualify_inverse_cholesky(library: str | Path) -> dict[str, object]:
    """Check the row gauge and full-rank Coulomb solve response against NumPy."""
    # This rotated metric has a repeated eigenvalue and a nondiagonal Cholesky factor.
    axis = np.array([1.0, -2.0, 2.0]) / 3
    metric = np.eye(3) + 3 * np.outer(axis, axis)
    whiten = partial(metric_whitener, library=library, factorization="inverse_cholesky")
    w, info = whiten(metric)
    lower = np.linalg.cholesky(metric)
    err = float(np.max(np.abs(w - np.linalg.solve(lower, np.eye(3)))))
    residual = float(np.max(np.abs(w @ metric @ w.T - np.eye(3))))
    if not (err < 2e-13 and residual < 2e-13 and np.count_nonzero(np.triu(w, 1)) == 0):
        raise ArithmeticError("incorrect row-major inverse lower-Cholesky factor")
    if (
        info["rank"] != 3
        or not info["device_bytes"]
        or info["native_symbol"] != "gqc_f12_molecular_metric_inverse_cholesky_v1"
    ):
        raise ArithmeticError("missing inverse-Cholesky full-rank execution evidence")
    w_exact, _ = whiten(metric, budget_bytes=info["device_bytes"])
    np.testing.assert_allclose(w_exact, w, rtol=0, atol=2e-13)
    rejections = qualify_inverse_cholesky_preflight(library)
    rejections["workspace_budget"] = _inverse_cholesky_rejection(
        library, metric, info["device_bytes"] - 1, "budget"
    )
    for name, eigenvalue in (
        ("indefinite", -0.5),
        ("truncated", 0.5e-10),
        ("at_cutoff", 1e-10),
        ("roundoff_band", 1e-10 + np.finfo(float).eps),
    ):
        rejection = _inverse_cholesky_rejection(
            library, np.diag([eigenvalue, 1.0]), 512 << 20, "full-rank"
        )
        if name in ("indefinite", "truncated") and rejection["rank"] != 1:
            raise ArithmeticError(f"incorrect rank returned on {name} rejection")
        rejections[name] = rejection
    near = np.diag([1e-10 + 1e-12, 1.0])
    above, rank_info = whiten(near)
    np.testing.assert_allclose(above @ near @ above.T, np.eye(2), rtol=0, atol=2e-13)
    if rank_info["rank"] != 2:
        raise ArithmeticError("well-resolved positive metric must retain both rows")

    # Differentiate sum(left * solve(M, right)), with all three inputs displaced.
    # The oracle uses the implicit solve derivative and never differentiates a factor.
    rng = np.random.default_rng(713)
    left, right, dl, dr = rng.normal(size=(4, 3, 2))
    dm = rng.normal(scale=0.2, size=(3, 3))
    dm = (dm + dm.T) / 2
    lfit, rfit = np.linalg.solve(metric, left), np.linalg.solve(metric, right)
    analytic = float(np.sum(dl * rfit + lfit * (dr - dm @ rfit)))

    def energy(step: float) -> float:
        factor, _ = whiten(metric + step * dm)
        return float(
            np.sum((factor @ (left + step * dl)) * (factor @ (right + step * dr)))
        )

    step = 2e-4
    fd = [(energy(h) - energy(-h)) / (2 * h) for h in (step, step / 2)]
    richardson = (4 * fd[1] - fd[0]) / 3
    if not abs(analytic - richardson) < 2e-10:
        raise ArithmeticError("inverse-Cholesky fitted-pair derivative failed")
    return {
        "row_factor_error": err,
        "identity_residual": residual,
        "resources": info,
        "above_cutoff": rank_info,
        "rejected": rejections,
        "contraction_analytic": analytic,
        "contraction_central_differences": fd,
        "contraction_richardson": richardson,
    }


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
        "inverse_cholesky": qualify_inverse_cholesky(library),
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
