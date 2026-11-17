"""Canonical cotangents against independently re-diagonalized Fock matrices."""

import numpy as np
import pytest
from generativeqc._canonical_frame import canonical_frame_seed
from generativeqc_compiler.method.canonical_frame_response import (
    canonical_pair_terms,
    orbital_rotation_cotangent,
)
from generativeqc_compiler.tensor import execute

from tools.generate_canonical_frame_reference import all_cases


@pytest.mark.parametrize("name,case", all_cases().items())
def test_canonical_constraints_against_displaced_eigenspaces(
    name: str, case: dict
) -> None:
    c, bar, eps = (case[k] for k in ("coefficients", "frame_bar", "energies"))
    m = execute(
        orbital_rotation_cotangent(*c.shape), {"coefficients": c, "frame_bar": bar}
    ).outputs["rotation_gradient"]
    np.testing.assert_allclose(m, case["rotation_gradient"], atol=3e-15, rtol=3e-15)
    diag, response = canonical_pair_terms()
    o, n = int(case["occupied"]), len(eps)
    seed = np.zeros_like(m)
    for p in range(n):
        for q in range(p + 1, n):
            if (p < o) != (q < o):
                continue
            fields = {
                "mpq": np.array(m[p, q]),
                "mqp": np.array(m[q, p]),
                "ep": np.array(eps[p]),
                "eq": np.array(eps[q]),
            }
            diagnostic = execute(diag, fields).outputs
            if abs(diagnostic["gap"]) <= 1e-8:
                assert abs(diagnostic["torque"]) < 1e-12
            else:
                seed[p, q] = seed[q, p] = execute(response, fields).outputs["seed"]
    predicted = np.vdot(seed, case["fock_direction_mo"])
    np.testing.assert_allclose(
        case["slopes"], predicted, atol=3e-8, rtol=0, err_msg=name
    )
    stationarity = m - m.T + 2 * (eps[:, None] - eps[None, :]) * seed
    np.testing.assert_allclose(stationarity[:o, :o], 0, atol=3e-15)
    np.testing.assert_allclose(stationarity[o:, o:], 0, atol=3e-15)


@pytest.mark.parametrize("dims", [(0, 3), (3, 0), (2, 3), (True, 1)])
def test_invalid_orbital_frame_dimensions(dims: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        orbital_rotation_cotangent(*dims)


@pytest.mark.parametrize(
    "case", ["shape", "dtype", "energy", "occupied", "threshold", "budget"]
)
def test_native_bridge_rejects_invalid_inputs_before_library_load(case: str) -> None:
    m, eps, o, options = np.eye(3), np.arange(3, dtype=float), 1, {}
    if case == "shape":
        m = np.ones((3, 2))
    elif case == "dtype":
        m = m.astype(np.float32)
    elif case == "energy":
        eps = eps[:2]
    elif case == "occupied":
        o = 3
    elif case == "threshold":
        options["gap_threshold"] = float("nan")
    else:
        options["maximum_bytes"] = 0
    with pytest.raises(ValueError):
        canonical_frame_seed(m, eps, o, **options)
