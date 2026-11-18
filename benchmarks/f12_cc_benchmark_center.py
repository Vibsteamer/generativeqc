"""Score a CC-F12 benchmark center and its supplied native displacement checks."""

import numpy as np

from benchmarks.f12_molecular_force_composition import common_checks


def check_center(report: dict, target: dict) -> None:
    """Keep center, directional and prospective full Cartesian independent claims distinct."""
    reference = target["cc_reference"]
    energies = {
        key: abs(report[key] - expected)
        for key, expected in reference["components_Eh"].items()
    }
    total = abs(report["total_energy_Eh"] - reference["energy_Eh"])
    directions = report["native_directional_checks"]
    inputs = target["native_displacements"]
    steps = {0.0004, 0.0002}
    planned = {(row["name"], row["step"]) for row in inputs}
    observed = {(row["direction"], row["step_bohr"]) for row in directions}
    norms = [np.linalg.norm(row["direction"]) for row in inputs]
    if (
        len({row["name"] for row in inputs}) != 1
        or {row["step"] for row in inputs} != steps
        or len(inputs) != 2
        or not np.array_equal(inputs[0]["direction"], inputs[1]["direction"])
        or not np.allclose(norms, 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError(
            "one unit benchmark direction at both signed steps is required"
        )
    force = np.asarray(report["total_force_Eh_per_bohr"])
    keys = {
        "reference_energy_Eh",
        "cabs_singles_energy_Eh",
        "ordinary_cc_correlation_Eh",
        "common_geminal_Eh",
        "extra_f12b_geminal_Eh",
    }
    if set(energies) != keys or len(target["geometry"]["elements"]) < 2:
        raise ValueError(
            "all five reference energy terms and a polyatomic geometry are required"
        )
    checks = {
        **common_checks(report, target),
        "independent_energy": total <= 5e-11,
        "independent_energy_terms": all(v <= 5e-11 for v in energies.values()),
        "finite_cartesian_force": force.shape
        == (len(target["geometry"]["elements"]), 3)
        and bool(np.isfinite(force).all()),
        "native_benchmark_direction_two_steps": len(directions) == len(planned)
        and planned == observed
        and all(a["max_error_Eh_per_bohr"] <= 2e-6 for a in directions),
        "relaxed_cc_residual": report["cc_residual"] <= 1e-10
        and report["lambda_residual"] <= 1e-10,
        "ten_integral_channels": len(report["integral_gradients"]["rows"]) == 10
        and report["integral_gradients"]["passes_contraction_checks"],
    }
    report.update(
        checks=checks,
        phase="center_and_native_direction_complete",
        passed=all(checks.values()),
        total_energy_error_Eh=total,
        reference_energy_term_errors_Eh=energies,
        independent_cartesian_status="awaiting independently reconverged Cartesian displacements",
    )
