"""Preserve the molecular electronic state when combining physical F12 sources."""

import numpy as np
import pytest

from benchmarks import f12_molecular_one_electron as integration
from benchmarks.f12_molecular_force_composition import combined_basis_for_geometry


def panel(elements: list[str], state: dict[str, int]) -> dict:
    xyz = [[0.0, 0.0, float(i)] for i in range(len(elements))]
    return {
        "geometry": {"elements": elements, "geometry_bohr": xyz, **state},
        "combined_basis": {
            "elements": elements,
            "coordinates_bohr": xyz,
            "shells": [
                {"atom_index": i, "angular_momentum": 0, "primitives": [[1.0, 1.0]]}
                for i in range(len(elements))
            ],
            **state,
        },
    }


@pytest.mark.parametrize(
    "elements, state, electrons",
    [
        (["H", "H"], {}, 2),
        (["O", "H", "H"], {"charge": 0, "multiplicity": 1}, 10),
        (["F"], {"charge": -1, "multiplicity": 1}, 10),
        (["O", "H", "H", "F"], {"charge": -1, "multiplicity": 1}, 20),
    ],
)
def test_physical_integral_contraction_keeps_state(
    monkeypatch: pytest.MonkeyPatch,
    elements: list[str],
    state: dict[str, int],
    electrons: int,
) -> None:
    case = combined_basis_for_geometry(panel(elements, state))
    n = len(elements)
    weights = np.eye(n)
    ordering = np.arange(n)

    def capture_context(
        calc: object,
        atoms: list,
        *,
        hcore: np.ndarray,
        overlap: np.ndarray,
        native_to_source: np.ndarray,
        charge: int,
        multiplicity: int,
    ) -> integration.IntegralGradient:
        assert [a for a, _ in atoms] == elements
        np.testing.assert_array_equal([r for _, r in atoms], case["coordinates_bohr"])
        assert (
            sum({"H": 1, "F": 9, "O": 8}[a] for a, _ in atoms) - charge,
            multiplicity,
        ) == (electrons, 1)
        assert hcore is weights and overlap is weights and native_to_source is ordering
        return integration.IntegralGradient(np.zeros((n, 3)), {})

    monkeypatch.setattr(integration, "Calculator", lambda *a, **k: object())
    monkeypatch.setattr(integration, "cuda_one_electron_gradient", capture_context)
    assert integration.native_contraction(
        case, weights, weights, ordering
    ).gradient.shape == (n, 3)


@pytest.mark.parametrize("key,value", [("charge", 0), ("multiplicity", 3)])
def test_reject_mixed_states(key: str, value: int) -> None:
    target = panel(["F"], {"charge": -1, "multiplicity": 1})
    target["combined_basis"][key] = value
    with pytest.raises(ValueError, match="geometry or state disagree"):
        combined_basis_for_geometry(target)
