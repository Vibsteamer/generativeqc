"""Run two CC-F12 component TensorIR programs and their source transposes on bent H2O."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from generativeqc_compiler.cc.f12_residual import (
    f12b_geminal_back_coupling,
    fixed_pair_cc_residual,
    standard_v_tile,
)
from generativeqc_compiler.tensor.ad_program import transpose_program

from benchmarks._f12_cuda import cuda_runner


def _b_reference(data: dict) -> np.ndarray:
    t1 = data["t1"]
    tau = data["t2"] + np.einsum("ia,jb->ijab", t1, t1)
    return (
        np.einsum("ab,ijab->ij", data["v_vv"], tau)
        + data["v_ov"] @ t1.T
        + t1 @ data["v_vo"]
    )


def compare(data: dict, run: Callable) -> dict:
    auxiliary, obs, _ = data["fit_left"].shape
    occupied, virtual = data["t1"].shape
    v_graph = standard_v_tile(obs, auxiliary, obs, obs)
    b_graph = f12b_geminal_back_coupling(occupied, virtual)
    v_feeds = {
        k: data[k]
        for k in ("geminal_coulomb", "geminal_obs", "fit_left", "raw_coulomb_right")
    }
    b_feeds = {k: data[k] for k in ("v_vv", "v_ov", "v_vo", "t1", "t2")}
    v = run(v_graph, v_feeds).outputs["v"]
    geminal = run(b_graph, b_feeds).outputs["back_coupling"]
    # Downstream use of the V matrix is also evaluated with the actual generated forward output.
    coupled = run(
        b_graph,
        {
            **b_feeds,
            "v_vv": v[occupied:, occupied:].copy(),
            "v_ov": v[:occupied, occupied:].copy(),
            "v_vo": v[occupied:, :occupied].copy(),
        },
    ).outputs["back_coupling"]
    rng = np.random.default_rng(8871)
    seed_v = rng.normal(size=v.shape) / np.sqrt(v.size)
    v_bars = run(
        transpose_program(v_graph, ("v",)).program, {**v_feeds, "bar_v": seed_v}
    ).outputs
    left, right, f = data["fit_left"], data["raw_coulomb_right"], data["geminal_obs"]
    v_reference = {
        "geminal_coulomb": seed_v,
        "geminal_obs": -np.einsum("uv,pur,pvs->rs", seed_v, left, right, optimize=True),
        "fit_left": -np.einsum("uv,rs,pvs->pur", seed_v, f, right, optimize=True),
        "raw_coulomb_right": -np.einsum(
            "uv,rs,pur->pvs", seed_v, f, left, optimize=True
        ),
    }
    b_bars = run(
        transpose_program(b_graph, ("back_coupling",)).program,
        {**b_feeds, "bar_back_coupling": data["seed"]},
    ).outputs
    tangents = {
        k: rng.normal(size=a.shape) / np.sqrt(a.size) for k, a in b_feeds.items()
    }
    slope = sum(
        np.vdot(b_bars["bar_" + k], direction) for k, direction in tangents.items()
    )
    fd_errors = []
    for h in (4e-4, 2e-4):
        ref = [
            _b_reference({k: x + sign * h * tangents[k] for k, x in b_feeds.items()})
            for sign in (-1, 1)
        ]
        fd_errors.append(
            float(abs(np.vdot(ref[1] - ref[0], data["seed"]) / (2 * h) - slope))
        )
    errors = {
        "v": float(np.max(abs(v - data["expected_v"]))),
        "geminal_residual": float(
            np.max(abs(geminal - data["expected_back_coupling"]))
        ),
        "coupled_residual": float(
            np.max(abs(coupled - data["expected_back_coupling"]))
        ),
        "v_source_adjoint": max(
            float(np.max(abs(v_bars["bar_" + k] - ref)))
            for k, ref in v_reference.items()
        ),
        "geminal_central_differences": fd_errors,
    }
    passed = (
        max(
            errors[k]
            for k in ("v", "geminal_residual", "coupled_residual", "v_source_adjoint")
        )
        < 2e-10
        and max(fd_errors) < 5e-8
        and fd_errors[1] < 1.5e-8
    )
    return {
        "passed": passed,
        "errors": errors,
        "v_norm": float(np.linalg.norm(v)),
        "geminal_residual_norm": float(np.linalg.norm(geminal)),
    }


def compare_fixed_pair(data: dict, run: Callable) -> dict:
    """Compare the physical R1/R2 source and its adjoints to an independent pair sum."""
    occupied, virtual = data["t1"].shape
    graph = fixed_pair_cc_residual(occupied, virtual, diagonal="select_n" not in data)
    names = ["t1", "v_vv", "v_ov", "v_vo", "fock_coupling_vv"]
    feeds = {k: data[k] for k in [*names, "select_m"]}
    if "select_n" in data:
        feeds["select_n"] = data["select_n"]
    result = run(graph, feeds).outputs
    errors = {
        key: float(np.max(abs(val - data["expected_" + key])))
        for key, val in result.items()
    }
    seeds = {"bar_" + key: data["seed_" + key] for key in result}
    bars = run(
        transpose_program(graph, tuple(result)).program, {**feeds, **seeds}
    ).outputs
    slope = sum(
        np.vdot(bars["bar_" + name], data["direction_" + name]) for name in names
    )
    slope_errors = abs(slope - data["expected_slopes"])
    return {
        "passed": bool(max(errors.values()) < 2e-11 and max(slope_errors) < 1e-9),
        "residual_errors": errors,
        "source_vjp_errors_two_step": slope_errors.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--fixed-pair-fixture",
        type=Path,
        help="Run the fixed-geminal ordinary R1/R2 correction with its matched independent CABS and finite-difference references.",
    )
    args = p.parse_args()
    run, metrics = cuda_runner(args.arch, args.cache)
    rows = {}
    with np.load(args.fixture, allow_pickle=False) as arrays:
        for name in ("diagonal", "ordered"):
            prefix = name + "_"
            rows[name] = compare(
                {
                    k.removeprefix(prefix): arrays[k]
                    for k in arrays
                    if k.startswith(prefix)
                },
                run,
            )
            args.output.write_text(
                json.dumps(
                    {"rows": rows, "programs": metrics}, indent=2, allow_nan=False
                )
                + "\n"
            )
    if args.fixed_pair_fixture:
        with np.load(args.fixed_pair_fixture, allow_pickle=False) as arrays:
            for name in ("diagonal", "ordered"):
                prefix = name + "_"
                rows["ordinary_" + name] = compare_fixed_pair(
                    {
                        key.removeprefix(prefix): arrays[key]
                        for key in arrays
                        if key.startswith(prefix)
                    },
                    run,
                )
        args.output.write_text(
            json.dumps({"rows": rows, "programs": metrics}, indent=2, allow_nan=False)
            + "\n"
        )
    if not all(r["passed"] for r in rows.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
