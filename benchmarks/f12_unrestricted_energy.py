"""Qualify the polarized H2 F12 energy and spin-resolved H/NH2 CABS singles."""

import argparse
import ctypes
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from generativeqc._f12_cabs import cabs_singles
from generativeqc._f12_unrestricted import (
    UnrestrictedCabsSingles,
    unrestricted_cabs_energy,
)
from generativeqc_compiler.mp2.f12_unrestricted import projected_same_spin_f12_pair
from generativeqc_compiler.tensor import transpose_program

from benchmarks._f12_cuda import add_molecular_graph_budget, cuda_runner


def _require(measured: float, reference: float, tolerance: float) -> float:
    error = abs(float(measured) - float(reference))
    if not np.isfinite(error) or error > tolerance:
        raise ValueError(
            f"Measured {measured} versus {reference}; tolerance {tolerance}"
        )
    return error


def _slopes(direction: dict, weights: dict, energy_at: Callable, steps: tuple) -> dict:
    slope = sum(float(np.vdot(weights[k], tangent)) for k, tangent in direction.items())
    errors = {}
    for h in steps:
        estimate = (energy_at(h) - energy_at(-h)) / (2 * h)
        errors[str(h)] = _require(estimate, slope, 2e-8)
    return {"fixed_frame_slope": slope, "two_step_abs_errors": errors}


def _doublet(directory: Path, ref: dict, run: Callable, singles: Callable) -> dict:
    with np.load(directory / (ref["molecule"] + ".npz"), allow_pickle=False) as src:
        dims = tuple(map(int, src["space"]))
        inputs = {
            k: src[k] for k in src if k != "space" and not k.startswith("expected_")
        }

    def solve_singles(values: dict, reverse: bool) -> UnrestrictedCabsSingles:
        return unrestricted_cabs_energy(
            dims, values, run, singles=singles, pullback=reverse
        )

    result = solve_singles(inputs, True)
    errors = [
        _require(value, ref[key], 1e-10)
        for value, key in zip(
            result.spin_energies, ("alpha_CABS_Eh", "beta_CABS_Eh"), strict=True
        )
    ]
    rng = np.random.default_rng(98284)
    direction = {}
    for k, v in inputs.items():
        d = rng.normal(size=v.shape)
        direction[k] = d / max(1.0, np.linalg.norm(d))

    def energy_at(t: float) -> float:
        return solve_singles(
            {k: v + t * direction[k] for k, v in inputs.items()}, False
        ).energy

    response = _slopes(direction, result.source_cotangents, energy_at, (1e-4, 3e-5))
    _require(
        response["fixed_frame_slope"],
        ref["fixed_frame_projected_source_derivative"],
        2e-8,
    )
    return {
        "case": ref["molecule"],
        "scope": "DF source to weight-one singles in each spin frame",
        "alpha_beta_CABS_Eh": result.spin_energies,
        "spin_errors_Eh": errors,
        **response,
    }


def evaluate(directory: Path, run: Callable, singles: Callable) -> list[dict]:
    controls = json.loads(
        (directory / "unrestricted_energy_reference.json").read_text()
    )
    rows = [_doublet(directory, ref, run, singles) for ref in controls["doublets"]]
    ref = controls["triplet"]
    with np.load(directory / "psi4_h2_triplet_projected.npz", allow_pickle=False) as ds:
        graph = projected_same_spin_f12_pair(*map(int, ds["space"]))
        reference_f12 = float(ds["reference_pair_energy"])
        inputs = {
            k: ds[k]
            for k in ds
            if k not in {"space", "beta", "reference_pair_energy", "psi4_ump2_total"}
        }
    terms = run(graph, inputs).outputs
    correction = float(terms["pair_energy"])
    mp2 = float(terms["ordinary_pair_energy"])
    singles_e = float(singles(inputs["fock"], 2, spin_weight=1, pullback=False)[0])
    errors = {
        "F12_Eh": _require(correction, reference_f12, 2e-11),
        "UMP2_Eh": _require(ref["UHF_Eh"] + mp2, ref["Psi4_UMP2_total_Eh"], 3e-10),
        "singles_Eh": _require(singles_e, ref["CABS_singles_Eh"], 1e-10),
    }
    total = ref["UHF_Eh"] + mp2 + singles_e + correction
    errors["sum_Eh"] = _require(total, ref["assembled_energy_Eh"], 3e-10)
    rev = transpose_program(graph, ("pair_energy",)).program
    bars = run(rev, {**inputs, "bar_pair_energy": np.array(1.0)}).outputs
    rng = np.random.default_rng(41713)
    direction = {}
    for k, v in inputs.items():
        if k.startswith("select"):
            continue
        d = rng.normal(size=v.shape)
        if k in ("fock", "exchange"):
            d = (d + d.T) / 2
        direction[k] = d / max(1.0, np.linalg.norm(d))

    def increment_at(t: float) -> float:
        changed = {**inputs, **{k: inputs[k] + t * v for k, v in direction.items()}}
        return float(run(graph, changed).outputs["pair_energy"])

    response = _slopes(
        direction, {k[4:]: v for k, v in bars.items()}, increment_at, (3e-4, 1e-4)
    )
    rows.append(
        {
            "case": "H2_triplet",
            "scope": "all-parallel, frozen molecular four-center integral inputs, external UHF reference energy, fixed-frame source VJP",
            "reference_UHF_Eh": ref["UHF_Eh"],
            "UMP2_corr_Eh": mp2,
            "F12_increment_Eh": correction,
            "CABS_singles_Eh": singles_e,
            "assembled_Eh": total,
            "errors": errors,
            **response,
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--component-library", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_molecular_graph_budget(parser)
    args = parser.parse_args()
    run, metrics = cuda_runner(args.arch, args.cache, max_bytes=args.budget_mib << 20)
    singles = partial(cabs_singles, library=ctypes.CDLL(args.component_library))
    rows = evaluate(args.fixtures, run, singles)
    args.output.write_text(
        json.dumps({"rows": rows, "programs": metrics}, indent=2, allow_nan=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
