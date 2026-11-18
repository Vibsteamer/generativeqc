"""Validate a VIB5 molecular energy and the available directional derivatives."""

import json

import numpy as np

from benchmarks.f12_molecular_force_composition import (
    calculate_panel,
    common_checks,
    endpoint_arguments,
    write_measurement,
)


def assess(report: dict, target: dict) -> dict:
    """Retain distinct central scalar, analytic-force consistency and reference statuses."""
    oracle = target["stock"]
    forces = np.asarray(report["total_force_Eh_per_bohr"])
    vector = np.asarray(oracle["direction"])
    derivative = float(np.vdot(forces, vector))
    if forces.shape != vector.shape or abs(np.linalg.norm(vector) - 1) > 1e-12:
        raise ValueError("independent VIB5 reference direction differs from geometry")
    stock_errors = {
        key: abs(derivative - value)
        for key, value in oracle["directional_force_Eh_per_bohr"].items()
    }
    native = report["native_directional_checks"]
    steps = set(map(float, stock_errors))
    if len(native) != 2 or len(steps) != 2 or {r["step_bohr"] for r in native} != steps:
        raise ValueError("missing native or reference two-step calculation")
    energy_errors = {
        "total_energy_Eh": abs(report["total_energy_Eh"] - oracle["energy_Eh"]),
        "increment_energy_Eh": abs(
            report["increment_Eh"] - oracle["increment_energy_Eh"]
        ),
    }
    energy_checks = {name: err <= 5e-11 for name, err in energy_errors.items()}
    directional_checks = {
        "native_two_step_direction": max(row["max_error_Eh_per_bohr"] for row in native)
        <= 2e-6,
        "independent_two_step_direction": max(stock_errors.values()) <= 2e-6,
    }
    refs = list(oracle["directional_force_Eh_per_bohr"].values())
    spread = abs(refs[0] - refs[1])
    sink = report["integral_gradients"]
    checks = {
        **common_checks(report, target),
        "ten_physical_integral_channels": len(sink["rows"]) == 10
        and sink["passes_contraction_checks"] is True
        and sink["reference_channel_count"] == 0
        and sink["passes_central_channel_references"] is None,
        **energy_checks,
        **directional_checks,
    }
    report.update(
        phase="complete",
        passed=all(checks.values()),
        checks=checks,
        geometry_sha256=target["geometry"]["geometry_sha256"],
        energy_checks=energy_checks,
        directional_checks=directional_checks,
        total_energy_error_Eh=energy_errors["total_energy_Eh"],
        increment_error_Eh=energy_errors["increment_energy_Eh"],
        projected_native_force_Eh_per_bohr=derivative,
        independent_directional_error_Eh_per_bohr=stock_errors,
        independent_two_step_spread_Eh_per_bohr=spread,
        minimum_possible_max_reference_error_Eh_per_bohr=0.5 * spread,
        independent_full_cartesian_reference_available=False,
    )
    return report


def main() -> None:
    args = endpoint_arguments()
    if args.budget_mib != 4096:
        raise ValueError(
            "VIB5 molecular graph budget must match the 4096 MiB preflight"
        )
    target = json.loads(args.panel.read_bytes())
    report = assess(calculate_panel(args, target), target)
    write_measurement(args.output, report)
    if not report["passed"]:
        raise SystemExit(
            "VIB5 numerical gate requires review; completed quantities retained"
        )


if __name__ == "__main__":
    main()
