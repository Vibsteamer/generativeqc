"""Verify a coupled DF-UHF orbital response against a molecular reference."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from generativeqc.response_solver import DiagonalPreconditioner, GMRESOptions, solve
from generativeqc_compiler.mp2.f12_unrestricted_reference import (
    unrestricted_df_orbital_action,
)

from benchmarks._f12_cuda import add_molecular_graph_budget, cuda_runner


def evaluate(
    directory: Path,
    run: Callable,
    *,
    fixture_name: str = "nh2_D_uhf_cphf.npz",
    case_name: str = "NH2_D_f12",
    coordinate: str = "H1 z",
    solution_output: Path | None = None,
) -> dict:
    with np.load(directory / fixture_name, allow_pickle=False) as z:
        oa, ob, n, aux = map(int, z["space"])
        rhs, solution = z["rhs"], z["solution"]
        fixed = {k: z[k] for k in z if k not in {"space", "rhs", "solution"}}
    g = unrestricted_df_orbital_action(oa, ob, n, aux)
    spins = ("alpha", "beta")
    gaps = [fixed["gap_" + s] for s in spins]

    def apply(x: np.ndarray) -> np.ndarray:
        trial = np.split(x, [gaps[0].size])
        fields = {
            "trial_" + s: chunk.reshape(gap.shape)
            for s, chunk, gap in zip(spins, trial, gaps, strict=True)
        }
        result = run(g, {**fixed, **fields}).outputs
        return np.concatenate([result[s].ravel() for s in spins])

    # Common GMRES controls; the two-electron operator is applied by generated
    # code. Its forward fit blocks occupy device memory on the CUDA runner.
    owner = SimpleNamespace(
        dimension=len(rhs),
        apply=apply,
        device_workspace_bytes=sum(a.nbytes for a in fixed.values()),
    )
    result = solve(
        owner,
        rhs,
        options=GMRESOptions(rtol=1e-11, atol=1e-13, true_residual_every=4),
        preconditioner=DiagonalPreconditioner(
            np.concatenate([a.ravel() for a in gaps])
        ),
        collect_basis=False,
        raise_on_failure=True,
    )
    residual = float(np.max(abs(apply(result.solution) - rhs)))
    error = float(np.max(abs(result.solution - solution)))
    if not np.isfinite([error, residual]).all() or residual > 2e-10 or error > 2e-9:
        raise ValueError(
            f"Coupled UHF orbital response differs from the reference: residual={residual}, error={error}"
        )
    if solution_output is not None:
        np.save(solution_output, result.solution, allow_pickle=False)
    return {
        "case": case_name,
        "iterations": result.iterations,
        "alpha_occupied": oa,
        "beta_occupied": ob,
        "rotations": len(rhs),
        "nuclear_coordinate": coordinate,
        "residual": residual,
        "maximum_amplitude_error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-name", default="nh2_D_uhf_cphf.npz")
    parser.add_argument("--case-name", default="NH2_D_f12")
    parser.add_argument("--coordinate", default="H1 z")
    parser.add_argument(
        "--solution-output",
        type=Path,
        help="Save the verified alpha/beta rotation vector.",
    )
    add_molecular_graph_budget(parser)
    a = parser.parse_args()
    run, metrics = cuda_runner(a.arch, a.cache, max_bytes=a.budget_mib << 20)
    row = evaluate(
        a.fixtures,
        run,
        fixture_name=a.fixture_name,
        case_name=a.case_name,
        coordinate=a.coordinate,
        solution_output=a.solution_output,
    )
    a.output.write_text(
        json.dumps({"rows": [row], "programs": metrics}, indent=2, allow_nan=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
