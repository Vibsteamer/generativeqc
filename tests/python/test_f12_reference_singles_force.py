"""Keep an isolated HF+CABS derivative independent of geminal and ordinary correlation terms."""

import pytest

from benchmarks import f12_reference_singles_force as singles


def test_two_identical_stepped_directions_and_full_reference_energy(monkeypatch):
    monkeypatch.setattr(
        singles, "common_checks", lambda result, panel: {"physical_cuda": True}
    )
    steps = (0.0004, 0.0002)
    unit = [[0.0, 1.0, 0.0]]
    panel = {
        "native_displacements": [
            {"name": "y", "direction": unit, "step": h} for h in steps
        ],
        "reference_plus_singles": {
            "total_Eh": -3.0,
            "singles_Eh": -0.01,
            "signed_HF_CABS_Eh": {
                str(h): [-3.0 - 0.04 * h, -3.0 + 0.04 * h] for h in steps
            },
        },
    }
    channels = {k: {} for k in ("ao_coulomb", "metric")}
    measured = {
        "native_directional_checks": [
            {"direction": "y", "step_bohr": h, "max_error_Eh_per_bohr": 0.0}
            for h in steps
        ],
        "total_energy_Eh": -3.0,
        "cabs_singles_energy_Eh": -0.01,
        "total_force_Eh_per_bohr": [[0.0, 0.04, 0.0]],
        "forward_resources": {"integrals": channels},
        "integral_gradients": {"rows": channels, "passes_contraction_checks": True},
    }
    singles.check_singles(measured, panel)
    assert measured["passed"]
    measured["total_force_Eh_per_bohr"] = [[0.0, -0.04, 0.0]]
    singles.check_singles(measured, panel)
    assert not measured["checks"]["independent_two_step_direction"]
    measured["forward_resources"]["integrals"] = {"metric": {}}
    singles.check_singles(measured, panel)
    assert not measured["checks"]["forward_and_reverse_coulomb_channels"]
    del panel["native_displacements"][0]
    with pytest.raises(ValueError, match="two step lengths"):
        singles.check_singles(measured, panel)
