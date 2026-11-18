"""Solve the coupled DF-UHF orbital response of the bent NH2 molecule on physical fit sources."""

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


def evaluate(directory: Path, run: Callable) -> dict:
    with np.load(directory / "nh2_D_uhf_cphf.npz", allow_pickle=False) as z:
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
            f"Coupled UHF orbital response differs from fully assembled Hessian: residual={residual}, error={error}"
        )
    return {
        "case": "NH2_D_f12",
        "iterations": result.iterations,
        "alpha_occupied": oa,
        "beta_occupied": ob,
        "rotations": len(rhs),
        "nuclear_coordinate": "H1 z",
        "residual": residual,
        "maximum_amplitude_error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_molecular_graph_budget(parser)
    a = parser.parse_args()
    run, metrics = cuda_runner(a.arch, a.cache, max_bytes=a.budget_mib << 20)
    row = evaluate(a.fixtures, run)
    a.output.write_text(
        json.dumps({"rows": [row], "programs": metrics}, indent=2, allow_nan=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
