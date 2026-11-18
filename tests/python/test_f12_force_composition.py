"""Exercise the sign and single-reference ownership of the full molecular force."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.f12_molecular_force_composition import full_result


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
