"""Regression for high/low and distinct h/h ordered AO adjoints."""

import ctypes as ct

import numpy as np
import pytest

from benchmarks.f12_cabs_h_batch import CabsHShells


def test_high_weight_order_and_symmetric_assembly() -> None:
    # Interleave two h-shell AO slices and a lower shell in source order.
    rng = np.random.default_rng(4719)
    n = 24
    indices = rng.permutation(n)
    parts = [indices[:11], indices[11:13], indices[13:]]
    high = CabsHShells.__new__(CabsHShells)
    high.low_source = np.sort(parts[1])
    high.n = n
    high.blocks = [
        (i, j, a, b)
        for i, a in enumerate(parts)
        for j, b in enumerate(parts[: i + 1])
        if i != 1 or j != 1
    ]
    high.size = sum(len(a) * len(b) for _, _, a, b in high.blocks)
    high.charges = np.array([8, 1])
    high.xyz = np.zeros((2, 3))
    matrices = rng.normal(size=(7, 2, n, n))
    matrices = matrices + matrices.swapaxes(-1, -2)
    high.data = np.ascontiguousarray(
        np.concatenate(
            [
                matrices[:, :, a[:, None], b[None, :]].reshape(7, 2, -1)
                for _, _, a, b in high.blocks
            ],
            axis=2,
        )
    )
    low = high.low_source
    S, H = high.assemble(*matrices[0][:, low[:, None], low[None, :]])
    np.testing.assert_array_equal(np.stack([S, H]), matrices[0])

    def inner(
        nd: int,
        cols: int,
        jac: np.ndarray,
        w: np.ndarray,
        out: np.ndarray,
        budget: int,
        used: object,
    ) -> int:
        # Independent explicit dot for the captured compact bridge inputs.
        assert nd == 6 and cols == w.size and jac.nbytes < budget
        out[:] = (jac.reshape(nd, cols) @ w.ravel()).reshape(out.shape)
        ct.cast(used, ct.POINTER(ct.c_size_t))[0] = jac.nbytes
        return 0

    high.contract = inner
    W = rng.normal(size=(2, n, n))
    masked = W.copy()
    masked[:, low[:, None], low[None, :]] = 0
    expected = np.einsum("qij,aqij->a", masked, matrices[1:]).reshape(2, 3)
    actual, _ = high.high_gradient(*W, 10 << 20)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    high.data = high.data[:1]
    with pytest.raises(ValueError, match="derivatives"):
        high.high_gradient(*W, 10 << 20)
