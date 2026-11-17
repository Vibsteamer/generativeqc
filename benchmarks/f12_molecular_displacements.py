"""Reconstruct F12 increments on displaced molecular inputs and compare Cartesian slopes."""

import argparse
import json
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
from generativeqc_compiler.tensor.program import Program

from benchmarks._f12_cuda import cuda_runner
from benchmarks.f12_molecular_pairs import factor_inputs, pair_plan


@lru_cache(maxsize=16)
def energy_plan(occ: int, obs: int, ri: int, aux: int, diagonal: bool) -> Program:
    pair = pair_plan(occ, obs - occ, ri - obs, diagonal=diagonal, auxiliaries=aux)
    return Program(
        {"pair_energy": pair.outputs["pair_energy"]}, provenance=pair.provenance
    )


def increment(data: dict, run: Callable) -> float:
    occ, obs, ri = map(int, data["space"])
    fock = {k: data[k] for k in ("fock", "exchange")}
    factor_count = len(data["fit"])
    total = 0.0
    for i, j in zip(*np.triu_indices(occ), strict=True):
        graph = energy_plan(occ, obs, ri, factor_count, bool(i == j))
        total += float(
            run(graph, factor_inputs(data, int(i), int(j), fock)).outputs["pair_energy"]
        )
    return total


def compare(fixture: Path, run: Callable) -> dict:
    protocol = json.loads(fixture.read_bytes())
    case = protocol["case"]
    steps = protocol["steps_bohr"]
    assert len(steps) == 2 and steps[0] > steps[1] > 0
    points = protocol["points"]
    center = case + "_center"
    xyz = np.asarray(points[center]["geometry_bohr"])
    raw, expected, times = {}, {}, {}
    for tag, row in points.items():
        with np.load(fixture.parent / (tag + ".npz"), allow_pickle=False) as f:
            data = {k: f[k] for k in f}
        start = time.monotonic()
        raw[tag] = increment(data, run)
        times[tag] = time.monotonic() - start
        expected[tag] = float(row["reference_increment_Eh"])
    values, reference = [], []
    expected_tags = {center}
    for level, h in enumerate(steps):
        force, stock = np.zeros_like(xyz), np.zeros_like(xyz)
        for a, axis in np.ndindex(xyz.shape):
            ids = [f"{case}_{a}_{axis}_{level}_{sign}" for sign in ("minus", "plus")]
            expected_tags.update(ids)
            for sign, tag in zip((-1, 1), ids, strict=True):
                shifted = np.array(xyz, copy=True)
                shifted[a, axis] += sign * h
                np.testing.assert_allclose(
                    points[tag]["geometry_bohr"], shifted, atol=1e-13, rtol=0
                )
            minus, plus = ids
            force[a, axis] = (raw[minus] - raw[plus]) / (2 * h)
            stock[a, axis] = (expected[minus] - expected[plus]) / (2 * h)
        values.append(force)
        reference.append(stock)
    if expected_tags != set(points):
        raise ValueError(
            "Cartesian stencil membership differs from its two declared point sets"
        )
    errors = np.abs(np.asarray(values) - np.asarray(reference)).max(axis=(1, 2))
    delta = max(abs(raw[k] - expected[k]) for k in expected)
    return {
        "case": case,
        "point_count": len(raw),
        "steps_bohr": steps,
        "increment_Eh": raw[center],
        "reference_increment_Eh": expected[center],
        "max_point_error_Eh": delta,
        "forces_from_FD_Eh_per_bohr": np.asarray(values).tolist(),
        "independent_Psi4_FD_Eh_per_bohr": np.asarray(reference).tolist(),
        "max_errors_by_step_Eh_per_bohr": errors.tolist(),
        "reference_step_difference_Eh_per_bohr": float(
            np.max(abs(reference[0] - reference[1]))
        ),
        "points_Eh": raw,
        "cold_and_warm_point_wall_s": times,
        "scope": "Numerical molecular force of the 3C(FIX) increment on supplied integral, Fock and orbital sources.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--fixture", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    run, metrics = cuda_runner(args.arch, args.cache)
    row = compare(args.fixture, run)
    args.output.write_text(
        json.dumps({"rows": [row], "programs": metrics}, allow_nan=False) + "\n"
    )
    # Independent points are reconverged separately. Keep force comparison and
    # its noise visible even if the molecular energy or a Cartesian slope fails.
    gates = json.loads(args.fixture.read_bytes())["gates"]
    if (
        not np.isfinite(row["max_point_error_Eh"])
        or row["max_point_error_Eh"] > gates["point_Eh"]
        or max(row["max_errors_by_step_Eh_per_bohr"]) > gates["slope_Eh_per_bohr"]
    ):
        raise ValueError(
            f"Molecular energy/finite-difference comparison requires review: {row['max_point_error_Eh']} {row['max_errors_by_step_Eh_per_bohr']}"
        )
    print(
        row["case"],
        row["point_count"],
        row["max_point_error_Eh"],
        row["max_errors_by_step_Eh_per_bohr"],
        flush=True,
    )


if __name__ == "__main__":
    main()
