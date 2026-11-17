"""Qualify restricted 3C(FIX) pair energies and their intermediate VJP."""

import argparse
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc_compiler.mp2.f12_pairs import (
    restricted_f12_pair_response,
    restricted_f12_pairs,
)

CASES = ("he", "h2", "h2_df", "water_df")
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests/reference_data/f12/psi4_restricted_pairs.npz"
)


def displacement_directions(data: Any, name: str) -> dict[str, dict[str, np.ndarray]]:
    """Reconstruct the archived Psi4 displacement without storing random arrays."""
    occ, obs, ri, frozen = map(int, data[name + "/space"])
    active, vir = occ - frozen, obs - occ
    rng = np.random.default_rng(int(data[name + "/seed"]))
    v, x, b = (rng.standard_normal((active,) * 4) for _ in range(3))
    c, g, d = (rng.standard_normal((active, active, vir, vir)) for _ in range(3))
    f = rng.standard_normal((ri, ri))
    f = 0.5 * (f + f.T)
    for array in (v, x, b, c, g, d, f):
        array[:] /= max(1.0, np.linalg.norm(array))
    out = {}
    for group in ("diagonal", "offdiag"):
        i, j = (
            (np.arange(active), np.arange(active))
            if group == "diagonal"
            else np.triu_indices(active, 1)
        )
        out[group] = {
            "c_direct": c[i, j],
            "inverse_gap": d[i, j],
            "g": g[i, j],
            "occupied_fock_sum": f[i + frozen, i + frozen] + f[j + frozen, j + frozen],
            "v_direct": v[i, j, i, j],
            "x_direct": x[i, j, i, j],
            "b_direct": b[i, j, i, j],
        }
        if group == "offdiag":
            out[group].update(
                c_exchange=c[j, i],
                v_exchange=v[i, j, j, i],
                x_exchange=x[i, j, j, i],
                b_exchange=b[i, j, j, i],
            )
    return out


def evaluate_case(data: Any, name: str, run: Callable) -> dict[str, Any]:
    energy = response = pair_error = 0.0
    nvir = int(data[name + "/virtuals"])
    directions = displacement_directions(data, name)
    timings = []
    for diagonal, group in ((True, "diagonal"), (False, "offdiag")):
        prefix = f"{name}/{group}/"
        if prefix + "psi4_pairs" not in data:
            continue
        expected = data[prefix + "psi4_pairs"]
        feeds = {
            key.removeprefix(prefix + "in/"): data[key]
            for key in data.files
            if key.startswith(prefix + "in/")
        }
        start = time.perf_counter()
        pairs = restricted_f12_pairs(len(expected), nvir, diagonal=diagonal)
        primal = run(pairs, feeds).outputs["pair_energy"]
        forward = time.perf_counter() - start
        energy += float(np.sum(primal))
        pair_error = max(pair_error, float(np.max(np.abs(primal - expected))))
        start = time.perf_counter()
        vjp = restricted_f12_pair_response(len(expected), nvir, diagonal=diagonal)
        bars = run(vjp, {**feeds, "bar_pair_energy": np.ones_like(expected)}).outputs
        for key, value in bars.items():
            direction = directions[group][key.removeprefix("bar_")]
            response += float(np.sum(value * direction))
        timings.append(
            {
                "tile": group,
                "pairs": len(expected),
                "virtuals": nvir,
                "primal_s": forward,
                "vjp_s": time.perf_counter() - start,
            }
        )
    reference = float(data[name + "/energy"])
    slopes = data[name + "/slopes"]
    return {
        "case": name,
        "energy_Eh": energy,
        "psi4_Eh": reference,
        "error_Eh": energy - reference,
        "max_pair_error_Eh": pair_error,
        "intermediate_derivative": response,
        "psi4_displacements": [3e-4, 1e-4],
        "psi4_derivatives": slopes.tolist(),
        "derivative_errors": (response - slopes).tolist(),
        "timings": timings,
    }


def verify(row: Mapping[str, Any]) -> None:
    # Psi4 prints the individual pair table to 12 decimals. Its total and
    # independently displaced intermediate energies retain full precision.
    if (
        abs(row["error_Eh"]) > 2e-12
        or row["max_pair_error_Eh"] > 8e-13
        or max(map(abs, row["derivative_errors"])) > 2e-8
    ):
        raise ValueError(f"3C(FIX) pair qualification failed for {row['case']} ({row})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from benchmarks._f12_cuda import cuda_runner

    run, metrics = cuda_runner(args.arch, args.cache)

    rows = []
    with np.load(FIXTURE, allow_pickle=False) as data:
        for case in CASES:
            row = evaluate_case(data, case, run)
            rows.append(row)
            args.output.write_text(
                json.dumps(
                    {"rows": rows, "programs": metrics}, indent=2, allow_nan=False
                )
                + "\n"
            )
            verify(row)
            print(case, row["error_Eh"], row["derivative_errors"], flush=True)


if __name__ == "__main__":
    main()
