"""Time fitted-density metric response and compare with displaced libcint/Cholesky solves."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc_compiler.mp2.f12_density_fitting import coulomb_fit_response

from benchmarks._f12_cuda import cuda_runner


def check(data: Any, run: Callable) -> list[dict[str, Any]]:
    rows = []
    for name in ("hydrogen", "water"):
        case = {
            k.split("/")[1]: data[k] for k in data.files if k.startswith(name + "/")
        }
        j, g = case["metric"], case["three"]
        eigen = case["eigenvalues"]
        if eigen[0] <= 1e-12 * eigen[-1]:
            raise ValueError("Full-rank response requires a well-resolved metric")
        out = run(
            coulomb_fit_response(*g.shape),
            {
                "metric_whitener": np.ascontiguousarray(case["whitener"]),
                "three_center": np.ascontiguousarray(g),
                "fit_bar": np.ascontiguousarray(case["seed"]),
            },
        ).outputs
        np.testing.assert_allclose(out["fit"], case["expected"], atol=5e-10, rtol=1e-9)
        np.testing.assert_allclose(j @ out["fit"], g, atol=2e-9, rtol=1e-10)
        np.testing.assert_allclose(
            out["metric_bar"], out["metric_bar"].T, atol=1e-11, rtol=0.0
        )
        dg = np.vdot(out["three_center_bar"], case["dthree"]).item()
        dj = np.vdot(out["metric_bar"], case["dmetric"]).item()
        diff = dg + dj - case["slopes"]
        np.testing.assert_allclose(diff, 0, atol=2e-7, rtol=0)
        rows.append(
            {
                "case": name,
                "aux": len(j),
                "pairs": g.shape[1],
                "condition_number": float(eigen[-1] / eigen[0]),
                "three_center_term": dg,
                "metric_term": dj,
                "steps_bohr": case["steps"].tolist(),
                "errors": diff.tolist(),
                "max_solve_error": float(abs(out["fit"] - case["expected"]).max()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    args = parser.parse_args()
    run, programs = cuda_runner(args.arch, args.cache)
    with np.load(args.fixture, allow_pickle=False) as data:
        rows = check(data, run)
    args.output.write_text(
        json.dumps({"rows": rows, "programs": programs}, indent=2, allow_nan=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
