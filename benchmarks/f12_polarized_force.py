"""Recompute a real H2 triplet Cartesian force on native GPU from stored analytic tangents."""

import argparse
import ctypes
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from generativeqc._f12_cabs import cabs_singles
from generativeqc_compiler.mp2.f12_unrestricted import projected_same_spin_f12_pair
from generativeqc_compiler.mp2.f12_unrestricted_force import polarized_pair_forces
from generativeqc_compiler.tensor import transpose_program

from benchmarks._f12_cuda import add_molecular_graph_budget, cuda_runner


def evaluate(directory: Path, run: Callable, singles: Callable) -> dict:
    reference = json.loads((directory / "H2_triplet_analytic_force.json").read_text())
    energy_control = json.loads((directory / "H2_triplet_same_spin.json").read_text())
    with np.load(
        directory / "psi4_h2_triplet_projected.npz", allow_pickle=False
    ) as data:
        space = tuple(map(int, data["space"]))
        inputs = {
            k: data[k]
            for k in data
            if k not in {"space", "reference_pair_energy", "psi4_ump2_total", "beta"}
        }
    pair = projected_same_spin_f12_pair(*space)
    values = run(pair, inputs).outputs
    singles_energy, singles_bar = singles(
        inputs["fock"], space[0], spin_weight=1, pullback=True
    )
    energy = (
        energy_control["UHF_Eh"]
        + singles_energy
        + sum(float(z) for z in values.values())
    )
    if abs(energy - reference["reference_energy_Eh"]) > 3e-10 or not np.isfinite(
        energy
    ):
        raise ValueError(
            "Physical frozen-source polarized energy differs from the independent oracle"
        )
    reverse = transpose_program(pair, ("pair_energy", "ordinary_pair_energy")).program
    bars = run(
        reverse,
        {
            **inputs,
            "bar_pair_energy": np.array(1.0),
            "bar_ordinary_pair_energy": np.array(1.0),
        },
    ).outputs
    atoms = len(reference["geometry_bohr"])
    force_graph = polarized_pair_forces(*space, atoms)
    with np.load(
        directory / "h2_triplet_analytic_sources.npz", allow_pickle=False
    ) as data:

        def collect(name: str) -> np.ndarray:
            return np.asarray(
                [[data[f"{i}_{j}_{name}"] for j in range(3)] for i in range(atoms)]
            )

        sources = {"tangent_" + k: collect(k) for k in inputs if "bar_" + k in bars}
        sources["reference_gradient"] = collect("hf_gradient")
    forces = run(
        force_graph, {**sources, **bars, "singles_fock_bar": singles_bar}
    ).outputs["force"]
    center_error = float(
        np.max(np.abs(forces - np.asarray(reference["force_Eh_per_bohr"])))
    )
    fd_error = {
        h: float(np.max(np.abs(forces + np.asarray(rec["gradient_Eh_per_bohr"]))))
        for h, rec in reference["two_step_independent"].items()
    }
    if (
        not np.isfinite(forces).all()
        or center_error > 2e-10
        or max(fd_error.values()) > 5e-7
        or np.max(np.abs(forces.sum(axis=0))) > 2e-10
        or len(fd_error) != 2
    ):
        raise ValueError(
            f"Polarized physical force error {center_error}, FD {fd_error}"
        )
    return {
        "case": "H2_triplet",
        "scope": "GPU ordinary and F12 electronic VJPs, weight-one CABS and Cartesian force contraction; external nuclear integral/orbital tangents and reference UHF energy/gradient",
        "energy_Eh": energy,
        "forces_Eh_per_bohr": forces.tolist(),
        "max_center_error_Eh_per_bohr": center_error,
        "max_independent_fd_errors_Eh_per_bohr": fd_error,
    }


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
    row = evaluate(
        args.fixtures,
        run,
        partial(cabs_singles, library=ctypes.CDLL(args.component_library)),
    )
    args.output.write_text(
        json.dumps({"rows": [row], "programs": metrics}, indent=2, allow_nan=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
