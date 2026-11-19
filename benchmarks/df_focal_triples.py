"""Qualify conventional (T) forces from one missing native DF-CCSD endpoint.

Retained CCSD(T) forces and independent two-step energies are immutable inputs.
The resulting vector contains every coordinate, but this test qualifies only
the explicitly supplied directions; it is not full Cartesian F12 qualification.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np


def save(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


def compare(row: dict, native: dict, gates: dict) -> dict:
    retained = row["retained_ccsdt"]
    expected = row["independent_central"]
    force = np.asarray(native["forces_Eh_per_bohr"], dtype=float)
    total_force = np.asarray(retained["forces_Eh_per_bohr"], dtype=float)
    direction = np.asarray(row["direction"], dtype=float)
    if force.shape != total_force.shape or force.shape != direction.shape:
        raise ValueError("Component force/direction shape mismatch")
    if not np.isfinite(force).all() or not np.isfinite(total_force).all():
        raise ValueError("Non-finite component force")
    if native["backend"] != "cuda" or native["reference_model"] != "ri-jk":
        raise ValueError("Component did not execute the requested native DF reference")
    if native["correlation_model"] != "df-ccsd" or native["triples_energy_Eh"] != 0:
        raise ValueError("Expected the independent CCSD response without triples")
    triples_force = total_force - force
    triples_energy = retained["energy_Eh"] - native["energy_Eh"]
    energy_errors = {
        "ccsd": abs(native["energy_Eh"] - expected["reference_ccsd"]),
        "hf": abs(native["reference_energy_Eh"] - expected["reference_hf"]),
        "triples_independent": abs(triples_energy - expected["triples_Eh"]),
        "triples_retained": abs(triples_energy - retained["triples_energy_Eh"]),
    }
    residuals = {
        key: native[key]
        for key in (
            "lambda_residual",
            "lambda_independent_residual",
            "orbital_residual",
            "maximum_stationarity",
        )
    }
    displacements = row["independent_displacements"]
    if len(displacements) != 2 or {p["step_bohr"] for p in displacements} != {
        0.0004,
        0.0002,
    }:
        raise ValueError(
            "two distinct symmetric displacements of the same coordinate are required"
        )
    directions = []
    for pair in displacements:
        step = pair["step_bohr"]
        ccsd_fd = -(
            pair["plus"]["reference_ccsd"] - pair["minus"]["reference_ccsd"]
        ) / (2 * step)
        triples_fd = -(pair["plus"]["triples_Eh"] - pair["minus"]["triples_Eh"]) / (
            2 * step
        )
        directions.append(
            {
                "step_bohr": step,
                "ccsd_error_Eh_per_bohr": abs(
                    ccsd_fd - float(np.sum(force * direction))
                ),
                "triples_error_Eh_per_bohr": abs(
                    triples_fd - float(np.sum(triples_force * direction))
                ),
                "independent_triples_directional_force_Eh_per_bohr": triples_fd,
            }
        )
    if {p["step_bohr"] for p in directions} != {0.0004, 0.0002}:
        raise ValueError("Missing independent two-step controls")
    translation = float(np.max(np.abs(np.sum(triples_force, axis=0))))
    passed = (
        max(energy_errors.values()) <= gates["energy_Eh"]
        and all(
            np.isfinite(v) and abs(v) <= gates["response_residual"]
            for v in residuals.values()
        )
        and all(
            d[k] <= gates["directional_force_Eh_per_bohr"]
            for d in directions
            for k in ("ccsd_error_Eh_per_bohr", "triples_error_Eh_per_bohr")
        )
        and translation <= gates["translation_Eh_per_bohr"]
    )
    return {
        "case": row["case"],
        "geometry_sha256": row["geometry_sha256"],
        "variant": row["variant"],
        "passed": bool(passed),
        "native_ccsd": native,
        "triples_energy_Eh": triples_energy,
        "triples_forces_Eh_per_bohr": triples_force.tolist(),
        "energy_errors_Eh": energy_errors,
        "response_residuals": residuals,
        "directions": directions,
        "translation_error_Eh_per_bohr": translation,
        "qualified_independent_directions": 1,
        "full_cartesian_independent_qualification": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--component-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-components", type=int, default=6)
    args = parser.parse_args()
    fixture = json.loads((args.input_dir / "comparison_inputs.json").read_text())
    if (
        args.expected_components < 1
        or len(fixture["rows"]) != args.expected_components
        or len({r["case"] for r in fixture["rows"]}) != args.expected_components
    ):
        raise ValueError(
            "each expected retained method and geometry must have a unique case"
        )
    result = {
        "scope": "Native DF-CCSD central E/F plus retained CCSD(T); two-step independent CCSD and (T) directional force checks",
        "rows": [],
        "passed": False,
        "new_ccsdt_calculations": 0,
        "independent_cpu_oracle_recomputed": False,
    }
    for row in fixture["rows"]:
        stem = args.output.parent / row["case"]
        start = time.perf_counter()
        try:
            with stem.with_suffix(".log").open("w") as log:
                subprocess.run(
                    [
                        str(args.component_executable),
                        str(args.input_dir / row["native_input"]),
                        str(stem.with_suffix(".json")),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1800,
                    env={
                        **os.environ,
                        "GENERATIVEQC_DF_PROGRESS_TRACE": str(
                            stem.with_suffix(".jsonl")
                        ),
                    },
                )
            native = json.loads(stem.with_suffix(".json").read_text())
            trace = [
                json.loads(line)
                for line in stem.with_suffix(".jsonl").read_text().splitlines()
            ]
            if not any(
                e.get("key") == "reference_integral_model" and e.get("value") == "ri-jk"
                for e in trace
            ):
                raise ValueError("Missing physical RI-JK reference trace")
            checked = compare(row, native, fixture["gates"])
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            ArithmeticError,
            subprocess.SubprocessError,
        ):
            checked = {
                "case": row["case"],
                "passed": False,
                "error": traceback.format_exc(),
            }
        checked["complete_process_seconds"] = time.perf_counter() - start
        result["rows"].append(checked)
        save(args.output, result)
    result["passed"] = all(r["passed"] for r in result["rows"])
    result["new_ccsd_endpoints"] = len(result["rows"])
    save(args.output, result)
    if not result["passed"]:
        raise RuntimeError(
            "Conventional triples component qualification requires review"
        )


if __name__ == "__main__":
    main()
