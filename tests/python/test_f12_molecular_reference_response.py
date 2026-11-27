"""Test the physical orbital map and the interface to the reference that owns it."""

from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc._f12_molecular_response import (
    molecular_reference_response,
    orbital_source_rows,
    with_reference_orbitals,
)


def test_nontrivial_orbital_layout_and_embedding() -> None:
    public = np.array([3, 1, 4, 0, 2])
    obs_first = np.array([2, 4, 1, 0, 3])
    rows = orbital_source_rows(public, obs_first, 3)
    np.testing.assert_array_equal(rows, [1, 4, 2])
    frame = np.arange(25.0).reshape(5, 5)
    coefficients = np.eye(3)
    actual = with_reference_orbitals(frame, rows, coefficients)
    np.testing.assert_array_equal(actual[rows, :3], coefficients)
    np.testing.assert_array_equal(actual[[0, 3], :3], 0)
    np.testing.assert_array_equal(actual[:, 3:], frame[:, 3:])
    np.testing.assert_array_equal(frame, np.arange(25.0).reshape(5, 5))


@pytest.mark.parametrize("layout", [[0, 0], [1, 2], [0, -1], [False, True], [0.0, 1.0]])
def test_refuse_ambiguous_orbital_indices(layout: list[int | float | bool]) -> None:
    with pytest.raises(ValueError):
        orbital_source_rows(np.array(layout), [0, 1], 2)
    with pytest.raises(ValueError):
        with_reference_orbitals(np.eye(2), np.array(layout), np.eye(2))


def test_live_orbitals_return_only_to_their_reference() -> None:
    c = np.array([[0.6, 0.8], [-0.8, 0.6]])
    rows = np.array([2, 0])
    data = {
        "space": [1, 2, 3],
        "ao_permutation": [0, 2, 1],
        "coefficients": np.eye(3),
        "overlap": np.diag([2.0, 5.0, 7.0]),
        "hcore_ao": np.eye(3),
    }
    native = np.array([2, 1, 0])
    bar_c = np.arange(6.0).reshape(3, 2)
    overlap_bar = np.eye(3)
    energies = np.array([-0.5, 0.9])
    mapped = data["overlap"][np.ix_(rows, rows)]
    seed = np.diag([1.0, 2.0])
    tested_forces = np.array([[1.0, 2.0, 3.0]])

    def snapshot() -> dict:
        return {
            "overlap": mapped,
            "hcore": np.eye(2),
            "coefficients": c,
            "energies": energies,
            "energy": -3.0,
        }

    def evaluate(inputs: dict, *, pullback: bool) -> tuple[float, dict, dict]:
        assert pullback
        np.testing.assert_array_equal(inputs["coefficients"][rows, :2], c)
        return -0.3, {"coefficients": bar_c, "overlap": overlap_bar}, {}

    def run(_prog: object, inputs: dict) -> SimpleNamespace:
        return SimpleNamespace(
            outputs={
                "rotation_gradient": inputs["coefficients"].T @ inputs["frame_bar"]
            }
        )

    def canonical(
        torque: np.ndarray, eps: np.ndarray, occupied: int, *, maximum_bytes: int
    ) -> tuple[np.ndarray, int]:
        np.testing.assert_array_equal(torque, c.T @ bar_c[rows])
        np.testing.assert_array_equal(eps, energies)
        assert occupied == 1 and maximum_bytes == 256
        return seed, 80

    def response(
        fock: np.ndarray, frame: np.ndarray, *, maximum_bytes: int
    ) -> tuple[np.ndarray, dict]:
        assert maximum_bytes == 256
        np.testing.assert_array_equal(fock, seed)
        np.testing.assert_array_equal(frame, bar_c[rows])
        return tested_forces, {"orbital_actions": 1}

    session = SimpleNamespace(occupied=1, n=2, snapshot=snapshot, response=response)
    result = molecular_reference_response(
        session, data, native, evaluate, run, canonical=canonical, maximum_bytes=256
    )
    np.testing.assert_array_equal(result.reference_forces, tested_forces)
    assert (result.increment_energy, result.reference_energy) == (-0.3, -3.0)
    assert result.diagnostics["canonical_bytes"] == 80
    assert result.explicit_weights.keys() == {"overlap"}
    np.testing.assert_array_equal(result.explicit_weights["overlap"], overlap_bar)
    for bad in (data["hcore_ao"] + 0.1, np.full((3, 3), np.nan)):
        with pytest.raises(ValueError, match="hcore"):
            molecular_reference_response(
                session, {**data, "hcore_ao": bad}, native, evaluate, run
            )
