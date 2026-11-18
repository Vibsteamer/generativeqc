"""Run physical H and NH2 unrestricted F12 component fixtures on an FP64 GPU."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc_compiler.mp2.f12_unrestricted import (
    unrestricted_extended_df_fock,
    unrestricted_pair_vx,
)
from generativeqc_compiler.tensor import Program, transpose_program

from benchmarks._f12_cuda import add_molecular_graph_budget, cuda_runner


def check(
    graph: Program,
    fixture: dict[str, np.ndarray],
    run: Callable,
    tolerance: float,
) -> dict[str, Any]:
    """Compare physical matrices/scalars and both finite-difference VJP slopes."""
    inputs = {
        k: v
        for k, v in fixture.items()
        if k != "space" and not k.startswith("expected_")
    }
    reference = {
        k.removeprefix("expected_"): v
        for k, v in fixture.items()
        if k.startswith("expected_")
    }
    answer = run(graph, inputs).outputs
    if set(answer) != set(reference):
        raise ValueError("Incomplete physical component or reference")
    errors = {k: float(np.max(np.abs(v - reference[k]))) for k, v in answer.items()}
    if not all(np.isfinite(v) and v <= tolerance for v in errors.values()):
        raise ValueError(f"Independent physical component errors {errors}")

    rng = np.random.default_rng(80541)

    def unit(value: np.ndarray) -> np.ndarray:
        a = np.asarray(rng.normal(size=value.shape), dtype=np.float64)
        return a / max(1.0, float(np.linalg.norm(a)))

    seeds = {k: unit(v) for k, v in reference.items()}
    directions = {k: unit(v) for k, v in inputs.items()}
    rev = transpose_program(graph, tuple(seeds)).program
    bars = run(rev, {**inputs, **{"bar_" + k: v for k, v in seeds.items()}}).outputs
    slope = sum(
        float(np.vdot(bars["bar_" + k], direction))
        for k, direction in directions.items()
    )

    def displaced(t: float) -> float:
        values = run(
            graph, {k: v + t * directions[k] for k, v in inputs.items()}
        ).outputs
        return sum(float(np.vdot(seeds[k], v)) for k, v in values.items())

    fd = {str(h): (displaced(h) - displaced(-h)) / (2 * h) for h in (1e-4, 3e-5)}
    if not np.isfinite(slope) or not all(
        np.isfinite(v) and abs(v - slope) < 2e-8 for v in fd.values()
    ):
        raise ValueError(f"Intermediate derivative {slope} versus {fd}")
    return {
        "max_abs_errors": errors,
        "intermediate_directional_vjp": slope,
        "finite_difference": fd,
        "derivative_threshold": 2e-8,
        "primal_threshold": tolerance,
    }


def run_cases(directory: Path, run: Callable) -> list[dict[str, Any]]:
    rows = []
    for name in ("H_doublet", "NH2_bent_distorted_doublet"):
        with np.load(directory / (name + ".npz"), allow_pickle=False) as data:
            fields = dict(data)
        graph = unrestricted_extended_df_fock(*map(int, fields["space"]))
        rows.append(
            {
                "case": name,
                "component": "extended Fock and exchange",
                "dimensions": fields["space"].tolist(),
                **check(graph, fields, run, 2e-8),
            }
        )
    with np.load(directory / "nh2_vx.npz", allow_pickle=False) as data:
        for i in range(5):
            prefix = str(i) + "/"
            fields = {
                k.removeprefix(prefix): data[k]
                for k in data.files
                if k.startswith(prefix)
            }
            a, b, c, d, e, f, sa, sb = map(int, fields["space"])
            spins = ("alpha", "beta")
            channel = spins[sa] + "_" + spins[sb]
            graph = unrestricted_pair_vx((a, b, c), (d, e, f), channel=channel)
            rows.append(
                {
                    "case": "NH2_bent_distorted_doublet",
                    "component": "V/X",
                    "channel": channel,
                    "reference_case": i,
                    **check(graph, fields, run, 1e-11),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_molecular_graph_budget(parser)
    args = parser.parse_args()
    run, metrics = cuda_runner(args.arch, args.cache, max_bytes=args.budget_mib << 20)
    rows = run_cases(args.fixtures, run)
    args.output.write_text(
        json.dumps({"rows": rows, "programs": metrics}, indent=2, allow_nan=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
