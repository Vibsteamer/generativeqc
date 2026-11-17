"""Qualify the closed-shell 3C(FIX) increment from density-fitted molecular sources."""

import argparse
import json
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
from generativeqc_compiler.mp2.f12_molecular_pair import (
    extended_df_fock,
    projected_f12_pair,
)
from generativeqc_compiler.tensor.ad_program import transpose_program
from generativeqc_compiler.tensor.program import Program

from benchmarks._f12_cuda import cuda_runner

fock_plan = lru_cache(maxsize=8)(extended_df_fock)
pair_plan = lru_cache(maxsize=16)(projected_f12_pair)


def molecular_fock(data: dict, run: Callable) -> tuple:
    occ, _, ri = map(int, data["space"])
    plan = fock_plan(occ, ri, len(data["fit"]))
    source = {k: data[k] for k in ("fit_all", "raw_occupied", "hcore")}
    return plan, source, run(plan, source).outputs


def factor_inputs(data: dict, i: int, j: int, fock: dict) -> dict:
    """Select pure input views. All reductions and projections belong to the native graph."""
    occ = int(data["space"][0])
    inputs = {"select_i": np.eye(occ)[i], **fock}
    if i != j:
        inputs["select_j"] = np.eye(occ)[j]
    for key in (
        "fit",
        "raw_f12",
        "raw_f12_squared",
        "raw_f12_over_coulomb",
        "raw_double_commutator",
        "raw_coulomb",
    ):
        if key != "raw_coulomb" or i == j:
            inputs[key + "_i"] = np.ascontiguousarray(data[key][:, i, :])
        if i != j:
            inputs[key + "_j"] = np.ascontiguousarray(data[key][:, j, :])
    for op in ("f12", "f12_squared", "f12_over_coulomb", "double_commutator"):
        inputs["metric_" + op] = data["metric_" + op]
    return inputs


def matrix_directions(data: dict) -> dict:
    rng = np.random.default_rng(923647)
    result = {}
    for key, value in sorted(data.items()):
        if key not in {"fit", "fit_all", "hcore"} and not key.startswith(
            ("raw_", "metric_")
        ):
            continue
        tangent = rng.normal(size=value.shape)
        if key == "hcore" or key.startswith("metric_"):
            tangent = (tangent + tangent.T) / 2
        elif key == "fit_all":
            tangent = (tangent + tangent.swapaxes(1, 2)) / 2
        result[key] = tangent / max(1.0, np.linalg.norm(tangent))
    return result


def response(
    data: dict,
    i: int,
    j: int,
    run: Callable,
    graph: Program,
    feeds: dict,
    fock_build: tuple,
) -> dict:
    directions = matrix_directions(data)
    reverse = transpose_program(graph, ("pair_energy",)).program
    bars = run(reverse, {**feeds, "bar_pair_energy": np.array(1.0)}).outputs
    expected = {"bar_" + k for k in feeds if not k.startswith("select_")}
    if set(bars) != expected:
        raise ValueError("Incomplete molecular integral/Fock adjoint")
    fock, source, _ = fock_build
    fock_response = transpose_program(fock, ("fock", "exchange")).program
    upstream = run(
        fock_response,
        {**source, "bar_fock": bars["bar_fock"], "bar_exchange": bars["bar_exchange"]},
    ).outputs
    slope = sum(float(np.vdot(upstream["bar_" + k], directions[k])) for k in source)
    pair_dirs = factor_inputs({**directions, "space": data["space"]}, i, j, {})
    slope += sum(
        float(np.vdot(bars["bar_" + k], v))
        for k, v in pair_dirs.items()
        if not k.startswith("select_")
    )

    def displaced(t: float) -> float:
        moved = {**data, **{k: data[k] + t * d for k, d in directions.items()}}
        new_fock = molecular_fock(moved, run)[2]
        changed = factor_inputs(moved, i, j, new_fock)
        return float(run(graph, changed).outputs["pair_energy"])

    errors = {
        str(h): abs(slope - (displaced(h) - displaced(-h)) / (2 * h))
        for h in (3e-4, 1e-4)
    }
    if max(errors.values()) > 2e-8:
        raise ValueError(f"Raw molecular integral/Fock adjoint failed: {errors}")
    return {
        "slope_Eh": slope,
        "two_step_errors_Eh": errors,
        "differentiated_sources": sorted(directions),
        "directions": "one combined perturbation of one-electron, three-center, fitted-density and operator metric sources, in a fixed orbital frame",
    }


def evaluate(
    data: dict, name: str, run: Callable, *, test_response: bool = True
) -> dict:
    occ, obs, ri = map(int, data["space"])
    maxima = {}
    pair_error = total = 0.0
    derivative = None
    start = time.monotonic()
    fock_build = molecular_fock(data, run)
    for key, value in fock_build[2].items():
        maxima[key] = float(np.max(abs(value - data[key])))
        if not np.isfinite(maxima[key]) or maxima[key] > 5e-10:
            raise ValueError(f"{name} {key} differs from Psi4: {maxima[key]}")
    for i, j in zip(*np.triu_indices(occ), strict=True):
        feeds = factor_inputs(data, int(i), int(j), fock_build[2])
        graph = pair_plan(
            occ,
            obs - occ,
            ri - obs,
            diagonal=bool(i == j),
            auxiliaries=len(data["fit"]),
        )
        outputs = run(graph, feeds).outputs
        expected = {
            "c_direct": data["c"][i, j],
            "inverse_gap": data["denominators"][i, j],
            "g": data["g"][i, j],
            "occupied_fock_sum": data["fock"][i, i] + data["fock"][j, j],
        }
        sides = [("direct", i, j)]
        if i != j:
            sides.append(("exchange", j, i))
            expected["c_exchange"] = data["c"][j, i]
        for suffix, k, l in sides:
            for key in ("v", "x", "b"):
                expected[key + "_" + suffix] = data[key][i, j, k, l]
        for key, ref in expected.items():
            err = float(np.max(np.abs(outputs[key] - ref)))
            maxima[key] = max(maxima.get(key, 0.0), err)
            if not np.isfinite(err) or err > 5e-10:
                raise ValueError(f"{name} ({i},{j}) {key}: {err}")
        energy = float(outputs["pair_energy"])
        total += energy
        pair_error = max(pair_error, abs(energy - float(data["pair_reference"][i, j])))
        if test_response and ((occ == 1 and i == j) or (occ > 1 and (i, j) == (0, 1))):
            derivative = response(data, int(i), int(j), run, graph, feeds, fock_build)
    error = total - float(data["energy"])
    if not np.isfinite(total) or abs(error) > 5e-11 or pair_error > 5e-11:
        raise ValueError(
            f"{name} pair energies disagree with Psi4: {error} {pair_error}"
        )
    return {
        "case": name,
        "occupied_pairs": occ * (occ + 1) // 2,
        "F12_increment_Eh": total,
        "psi4_F12_increment_Eh": float(data["energy"]),
        "error_Eh": error,
        "max_pair_error_Eh": pair_error,
        "max_intermediate_errors": maxima,
        "response": derivative,
        "wall_including_compilation_s": time.monotonic() - start,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run, metrics = cuda_runner(args.arch, args.cache)
    rows = []
    with np.load(args.fixture, allow_pickle=False) as data:
        for case in ("h2", "water"):
            prefix = case + "/"
            mol = {
                k.removeprefix(prefix): data[k]
                for k in data.files
                if k.startswith(prefix)
            }
            row = evaluate(mol, case, run)
            rows.append(row)
            args.output.write_text(
                json.dumps(
                    {"rows": rows, "programs": metrics}, indent=2, allow_nan=False
                )
                + "\n"
            )
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
