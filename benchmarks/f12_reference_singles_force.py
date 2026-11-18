"""Compute just the HF+CABS term and its physical nuclear derivative for component extrapolation."""

import json

import numpy as np

from benchmarks.f12_molecular_force_composition import (
    calculate_panel,
    common_checks,
    endpoint_arguments,
    write_measurement,
)


def check_singles(result: dict, panel: dict) -> None:
    reference = panel["reference_plus_singles"]
    energy_error = abs(result["total_energy_Eh"] - reference["total_Eh"])
    singles_error = abs(result["cabs_singles_energy_Eh"] - reference["singles_Eh"])
    arms = result["native_directional_checks"]
    expected = {(row["name"], row["step"]) for row in panel["native_displacements"]}
    observed = {(row["direction"], row["step_bohr"]) for row in arms}
    steps = {0.0004, 0.0002}
    if {s for _, s in expected} != steps or len(expected) != 2:
        raise ValueError("one original direction at two step lengths is required")
    force = np.array(result["total_force_Eh_per_bohr"])
    direction = np.array(panel["native_displacements"][0]["direction"])
    if (
        force.shape != direction.shape
        or not np.array_equal(direction, panel["native_displacements"][1]["direction"])
        or abs(np.linalg.norm(direction) - 1) > 1e-12
    ):
        raise ValueError(
            "force and the shared unit direction must have the same molecular shape"
        )
    errors = {}
    for step in steps:
        plus, minus = reference["signed_HF_CABS_Eh"][str(step)]
        errors[str(step)] = abs(
            float(np.vdot(force, direction)) + (plus - minus) / (2 * step)
        )
    channels = {"metric", "ao_coulomb"}
    checks = {
        **common_checks(result, panel),
        "independent_HF_CABS_energy": energy_error <= 5e-11,
        "independent_CABS_energy": singles_error <= 5e-11,
        "forward_and_reverse_coulomb_channels": set(
            result["forward_resources"]["integrals"]
        )
        == channels
        and set(result["integral_gradients"]["rows"]) == channels
        and result["integral_gradients"]["passes_contraction_checks"],
        "independent_two_step_direction": all(v <= 2e-6 for v in errors.values()),
        "native_two_step_direction": len(arms) == 2
        and observed == expected
        and all(row["max_error_Eh_per_bohr"] <= 2e-6 for row in arms),
    }
    result.update(
        phase="central_and_direction_complete",
        passed=all(checks.values()),
        checks=checks,
        total_energy_error_Eh=energy_error,
        singles_error_Eh=singles_error,
        independent_force_direction_errors_Eh_per_bohr=errors,
        scope="Native field-free HF+CABS singles E/F, independent two-step directional control. For matched correlation-basis extrapolation use F_corr = F_total – F_HF+CABS in the same reference, geometry and basis.",
    )


def main() -> None:
    args = endpoint_arguments()
    panel = json.loads(args.panel.read_bytes())
    result = calculate_panel(
        args,
        panel,
        full_rank_constraint=args.full_rank_cabs_response,
        reference_singles_only=True,
    )
    check_singles(result, panel)
    write_measurement(args.output, result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
