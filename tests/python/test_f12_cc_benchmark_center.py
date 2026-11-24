"""Preserve the separation of independent energy, native direction and full Cartesian gates."""

import pytest

from benchmarks import f12_cc_benchmark_center as center


def test_benchmark_center_requires_both_native_steps_and_all_energy_terms(monkeypatch):
    keys = (
        "reference_energy_Eh",
        "cabs_singles_energy_Eh",
        "ordinary_cc_correlation_Eh",
        "common_geminal_Eh",
        "extra_f12b_geminal_Eh",
    )
    monkeypatch.setattr(
        center, "common_checks", lambda report, target: {"physical_cuda": True}
    )
    rows = [
        {"name": "move F", "step": h, "direction": [[0, 0, 1], [0, 0, 0]]}
        for h in [0.0004, 0.0002]
    ]
    observed = [
        {"direction": r["name"], "step_bohr": r["step"], "max_error_Eh_per_bohr": 0.0}
        for r in rows
    ]
    target = {
        "geometry": {"elements": ["F", "H"]},
        "native_displacements": rows,
        "cc_reference": {"components_Eh": dict.fromkeys(keys, 0.0), "energy_Eh": 0.0},
    }
    report = {
        **dict.fromkeys(keys, 0.0),
        "total_force_Eh_per_bohr": [[0, 0, 0], [0, 0, 0]],
        "total_energy_Eh": 0.0,
        "cc_residual": 0.0,
        "lambda_residual": 0.0,
        "native_directional_checks": observed,
        "integral_gradients": {
            "rows": list(range(10)),
            "passes_contraction_checks": True,
        },
    }
    center.check_center(report, target)
    assert report["passed"] and report["independent_cartesian_status"].startswith(
        "awaiting"
    )
    report["ordinary_cc_correlation_Eh"] = 1e-8
    center.check_center(report, target)
    assert not report["passed"] and report["checks"]["independent_energy"]
    report["native_directional_checks"] = observed[:1]
    center.check_center(report, target)
    assert not report["checks"]["native_benchmark_direction_two_steps"]
    target["native_displacements"][1]["direction"][0][2] = 0
    with pytest.raises(ValueError, match="unit benchmark direction"):
        center.check_center(report, target)
