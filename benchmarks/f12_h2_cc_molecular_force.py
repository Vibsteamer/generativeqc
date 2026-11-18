"""Qualify a complete two-electron CC-F12 force with robust Coulomb fitting of the external V."""

import json
from pathlib import Path

import numpy as np

from benchmarks.f12_h2_cc_molecular_correlation import h2_cc_correlation
from benchmarks.f12_molecular_force_composition import (
    calculate_panel,
    common_checks,
    endpoint_arguments,
    write_measurement,
)


def validate(report: dict, target: dict) -> dict:
    oracle = target["cc_reference"]
    forces = np.asarray(report["total_force_Eh_per_bohr"])
    steps = ("0.0004", "0.0002")
    energy_error = abs(report["total_energy_Eh"] - oracle["energy_Eh"])
    errors = []
    for step in steps:
        ref = np.array(oracle["forces_Eh_per_bohr"][step])
        if forces.shape != (2, 3) or ref.shape != forces.shape:
            raise ValueError("H2 requires six Cartesian derivatives per step")
        errors.append(float(np.max(abs(forces - ref))))
    arms = report["native_directional_checks"]
    directions = {(a["direction"], a["step_bohr"]) for a in arms}
    checks = {
        **common_checks(report, target),
        "independent_energy": energy_error <= 5e-11,
        "independent_cartesian_two_steps": max(errors) <= 2e-6,
        "native_directions": directions
        == {(direction, float(h)) for direction in ("mixed", "bond") for h in steps}
        and max(a["max_error_Eh_per_bohr"] for a in arms) <= 2e-6,
        "relaxed_cc_residual": report["cc_residual"] <= 1e-10
        and report["lambda_residual"] <= 1e-10,
        "ten_integral_channels": len(report["integral_gradients"]["rows"]) == 10
        and report["integral_gradients"]["passes_contraction_checks"],
    }
    report.update(
        phase="complete",
        checks=checks,
        passed=all(checks.values()),
        scope="Two-electron fixed-geminal restricted CC-F12a/b equations, robust external V, fully relaxed molecular force. Independent same-equation determinant solver and integral sources.",
        total_energy_error_Eh=energy_error,
        independent_force_max_errors_Eh_per_bohr=dict(zip(steps, errors, strict=True)),
    )
    return report


def main() -> None:
    args = endpoint_arguments()
    target = json.loads(Path(args.panel).read_bytes())
    if target["geometry"]["elements"] != ["H", "H"] or target["molecule"] != "h2":
        raise ValueError(
            "this two-electron force qualification requires molecular hydrogen"
        )
    report = calculate_panel(
        args,
        target,
        full_rank_constraint=args.full_rank_cabs_response,
        molecular_correlation=h2_cc_correlation,
    )
    validate(report, target)
    write_measurement(args.output, report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
