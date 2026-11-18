"""Validate a VIB5 molecular energy and the available directional derivatives."""

import hashlib
import json
from pathlib import Path

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


def check_graph_budget(
    target: dict, panel: Path, layout: Path, budget_mib: int
) -> None:
    """Bind the graph limit to the shape and estimates used for this geometry."""
    declaration = target.get("native_graph_plan")
    if declaration is None:
        if budget_mib == 4096 and target["layout"]["space"] == [5, 66, 220]:
            return  # The archived D panel predates the per-cardinal plan file.
        raise ValueError(
            "missing VIB5 graph preflight for the requested space or budget"
        )
    plan_bytes = (panel.parent / "graph_plan.json").read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != declaration["graph_plan_sha256"]:
        raise ValueError("VIB5 graph preflight hash differs")
    plan = json.loads(plan_bytes)
    with np.load(layout, allow_pickle=False) as arrays:
        sizes = arrays["ao_sizes"].tolist()
    space = target["layout"]["space"]
    if (
        plan["space"] != declaration["space"]
        or plan["space"] != space
        or sizes != [space[-1], plan["fitting_aos"]]
        or plan["fitting_aos"] != declaration["fitting_aos"]
    ):
        raise ValueError(
            "VIB5 graph preflight dimensions differ from the molecular inputs"
        )
    largest = max(row["combined_bytes"] for row in plan["per_graph"].values())
    if (
        not 0 < largest <= (budget_mib << 20)
        or declaration["largest_combined_bytes"] != largest
        or declaration["budget_mib"] != budget_mib
    ):
        raise ValueError(
            "the requested graph budget differs or cannot hold the largest graph"
        )


def main() -> None:
    args = endpoint_arguments()
    target = json.loads(args.panel.read_bytes())
    check_graph_budget(target, args.panel, args.layout, args.budget_mib)
    result = calculate_panel(
        args, target, full_rank_constraint=args.full_rank_cabs_response
    )
    report = assess(result, target)
    write_measurement(args.output, report)
    if not report["passed"]:
        raise SystemExit(
            "VIB5 numerical gate requires review; completed quantities retained"
        )


if __name__ == "__main__":
    main()
