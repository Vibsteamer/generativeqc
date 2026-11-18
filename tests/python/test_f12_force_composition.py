"""Exercise the sign and single-reference ownership of the full molecular force."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.f12_molecular_force_composition import endpoint_arguments, full_result


def test_only_correlation_is_added_to_the_owned_reference() -> None:
    reference = SimpleNamespace(
        reference_energy=-75.0,
        increment_energy=-0.1,
        reference_forces=np.array([[3.0, -1.0, 2.0]]),
    )
    ordinary = {
        "passed": True,
        "backend": "cuda",
        "reference_energy_Eh": -75.0,
        "correlation_energy_Eh": -0.2,
        "correlation_forces_Eh_per_bohr": [[0.5, 1.0, 1.0]],
        "mp2_energy_Eh": -75.2,
    }
    energy, forces = full_result(
        reference, [[0.1, 1.0, 0.5]], [[1.0, 2.0, 0.1]], ordinary
    )
    assert energy == pytest.approx(-75.3)
    np.testing.assert_allclose(forces, [[2.4, -3.0, 2.4]])
    with pytest.raises(ValueError, match="different RHF"):
        full_result(
            reference, forces, forces, {**ordinary, "reference_energy_Eh": -74.0}
        )
    with pytest.raises(ValueError, match="Cartesian"):
        full_result(reference, np.zeros(3), forces, ordinary)


def test_numpy_scalar_checkpoint_survives_later_invalid_report(tmp_path: Path) -> None:
    import json

    from benchmarks.f12_molecular_force_composition import write_measurement

    path = tmp_path / "receipt.json"
    partial = {
        "passed": np.bool_(True),
        "residual": np.float64(1e-12),
        "rows": np.int64(10),
    }
    write_measurement(path, partial)
    assert json.loads(path.read_text()) == {
        "passed": True,
        "residual": 1e-12,
        "rows": 10,
    }
    raw = path.read_bytes()
    for bad, exception in ((np.nan, ValueError), (np.array([1.0]), TypeError)):
        with pytest.raises(exception):
            write_measurement(path, {"value": bad})
        assert path.read_bytes() == raw


@pytest.mark.parametrize("selected", [False, True])
def test_full_rank_cabs_response_is_opt_in(
    monkeypatch: pytest.MonkeyPatch, selected: bool
) -> None:
    paths = [
        "scf-inputs",
        "component-library",
        "session-library",
        "cache",
        "output",
        "layout",
        "sink-library",
        "panel",
    ]
    argv = ["molecular-ef", "--arch", "sm_89", "--budget-mib", "16384"]
    for name in paths:
        argv.extend(["--" + name, name])
    if selected:
        argv.append("--full-rank-cabs-response")
    monkeypatch.setattr(sys, "argv", argv)
    assert endpoint_arguments().full_rank_cabs_response is selected
