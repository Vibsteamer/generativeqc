"""The controlled source must exercise orbital response without gauge artifacts."""

import numpy as np
import pytest
from scipy.linalg import expm

from benchmarks.df_reference_response import controlled_source


@pytest.mark.parametrize("occupied,virtual", [(1, 4), (3, 5)])
def test_controlled_scalar_derivatives_and_gauge(occupied: int, virtual: int) -> None:
    n = occupied + virtual
    rng = np.random.default_rng(1903)
    c = rng.normal(size=(n, n))
    eps = np.r_[np.linspace(-2, -0.4, occupied), np.linspace(0.1, 1.2, virtual)]
    value, f, bar_c = controlled_source(c, eps, occupied)
    dc, de = rng.normal(size=c.shape), rng.normal(size=n)
    analytic = np.vdot(bar_c, dc) + np.vdot(np.diag(f), de)
    for h in (1e-4, 2e-5):
        positive = controlled_source(c + h * dc, eps + h * de, occupied)[0]
        negative = controlled_source(c - h * dc, eps - h * de, occupied)[0]
        np.testing.assert_allclose(
            (positive - negative) / (2 * h), analytic, atol=1e-11, rtol=0
        )

    # The test uses occupied and virtual traces: arbitrary within-space unitary
    # changes (including degenerate canonical orbitals) do not create a torque.
    u = np.zeros((n, n))
    for lo, hi in ((0, occupied), (occupied, n)):
        raw = rng.normal(size=(hi - lo, hi - lo))
        u[lo:hi, lo:hi] = expm(raw - raw.T)
    rotated, _, _ = controlled_source(c @ u, eps, occupied)
    np.testing.assert_allclose(rotated, value, atol=1e-15, rtol=0)
    torque = c.T @ bar_c - bar_c.T @ c
    np.testing.assert_allclose(torque[:occupied, :occupied], 0, atol=1e-15, rtol=0)
    np.testing.assert_allclose(torque[occupied:, occupied:], 0, atol=1e-15, rtol=0)
    assert np.max(abs(torque[:occupied, occupied:])) > 1e-4
