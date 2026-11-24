"""Qualify the native CUDA Coulomb metric whitener."""

from pathlib import Path

import numpy as np
from generativeqc._f12_focal_metric import metric_whitener


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
