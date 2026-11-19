"""Check the Sylvester singles energy and stationary response with independent orbital denominators."""

import numpy as np
import pytest
from benchmarks.f12_cabs_singles_nuclear import (
    restricted_singles,
    restricted_singles_ao_weights,
)
from generativeqc_compiler.tensor import execute


def orbital_reference(f, o):
    eo, u = np.linalg.eigh(f[:o, :o])
    ev, v = np.linalg.eigh(f[o:, o:])
    vmo = v.T @ f[o:, :o] @ u
    return -2 * np.sum(vmo**2 / (ev[:, None] - eo))


def test_singles_sources_reach_ao_and_constituent_orbital_subspaces():
    rng = np.random.default_rng(906)
    o, n, r, a = 2, 4, 6, 5

    def sym(x):
        return 0.5 * (x + x.swapaxes(-2, -1))

    c = np.linalg.qr(rng.normal(size=(r, r)))[0]
    h = c @ np.diag([-3.0, -2.0, 1.0, 2.0, 2.0, 3.0]) @ c.T + sym(
        rng.normal(scale=0.05, size=(r, r))
    )
    ao = sym(rng.normal(scale=0.15, size=(a, r, r)))
    m = sym(rng.normal(scale=0.12, size=(a, a))) + 4 * np.eye(a)
    data = {
        "space": [o, n, r],
        "coefficients": c,
        "hcore_ao": h,
        "ao_coulomb": ao,
        "metric": m,
    }

    def e(d):
        c = d["coefficients"]
        g = c.T @ d["ao_coulomb"] @ c
        eri = (
            g.reshape(a, -1).T @ np.linalg.solve(d["metric"], g.reshape(a, -1))
        ).reshape((r,) * 4)
        f = (
            c.T @ d["hcore_ao"] @ c
            + 2 * np.einsum("pqii->pq", eri[:, :, :o, :o])
            - np.einsum("piiq->pq", eri[:, :o, :o, :])
        )
        return orbital_reference(f, o)

    energy, bars = restricted_singles_ao_weights(data, execute)
    np.testing.assert_allclose(energy, e(data), atol=2e-14)
    for field, bar in (
        ("coefficients", bars["coefficients"]),
        ("hcore_ao", bars["hcore"]),
        ("ao_coulomb", bars["ao"]["coulomb"]),
        ("metric", bars["metric"]["coulomb"]),
    ):
        t = rng.normal(scale=0.08, size=data[field].shape)
        if field != "coefficients":
            t = sym(t)
        step = 3e-5
        fd = (
            e({**data, field: data[field] + step * t})
            - e({**data, field: data[field] - step * t})
        ) / (2 * step)
        np.testing.assert_allclose(np.vdot(bar, t), fd, atol=3e-9, rtol=2e-8)
    with pytest.raises(ValueError, match="positive"):
        restricted_singles(np.diag([1.0, 0.0, -2.0]), 1, pullback=False)
