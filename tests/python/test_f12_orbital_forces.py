import os

import numpy as np
import pytest
from generativeqc.f12_orbital_response import unrestricted_orbital_forces


def inputs(oa: int = 2, ob: int = 1) -> tuple[tuple, np.ndarray]:
    n, q = 4, 3
    rng = np.random.default_rng(32)
    factors = {}
    gaps = []
    for spin, o in (("alpha", oa), ("beta", ob)):
        if o:
            v = n - o
            for name, shape in (
                ("vo", (q, v, o)),
                ("oo", (q, o, o)),
                ("vv", (q, v, v)),
            ):
                factors[f"{name}_{spin}"] = np.zeros(shape)
            gap = rng.uniform(1, 2, (v, o))
            factors[f"gap_{spin}"] = gap
            gaps.extend(gap.ravel())
    driver = rng.normal(size=len(gaps))
    rhs = rng.normal(size=(2, 3, len(gaps)))
    reference = rng.normal(size=(2, 3))
    fixed = rng.normal(size=(2, 2, 3))
    return ((oa, ob, n, q), factors, driver, rhs, reference, fixed), np.array(gaps)


def test_molecular_adjoint_validates_budget_and_inputs_before_device_load() -> None:
    data, _ = inputs()
    with pytest.raises(ValueError, match="budget"):
        unrestricted_orbital_forces(*data, maximum_bytes=1)
    data[2][0] = np.nan
    with pytest.raises(ValueError, match="electronic driver"):
        unrestricted_orbital_forces(*data)


@pytest.mark.skipif(
    os.getenv("GENERATIVEQC_TEST_CUDA") != "1",
    reason="requires native CUDA library and GPU",
)
@pytest.mark.parametrize("oa,ob", [(2, 1), (2, 0), (0, 1)])
def test_native_transposed_solve_force_sign_and_single_spin(oa: int, ob: int) -> None:
    data, gaps = inputs(oa, ob)
    expected = data[2] / gaps
    gradient = data[3] @ expected
    result = unrestricted_orbital_forces(*data)
    np.testing.assert_allclose(result.rotations, expected, atol=1e-12)
    np.testing.assert_allclose(result.orbital_gradient, gradient, atol=1e-12)
    np.testing.assert_allclose(
        result.forces, data[4] - data[5].sum(axis=0) - gradient, atol=1e-12
    )
