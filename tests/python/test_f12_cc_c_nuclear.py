"""Check all fixed-cusp C sources with a direct extended-Fock and geminal contraction."""

import numpy as np
import pytest

from benchmarks.f12_cc_c_nuclear import c_ao_weights
from generativeqc_compiler.tensor import execute


def explicit_c(c, h, ao, metric, o, v):
    raw = c.T @ ao["coulomb"] @ c
    g = c[:, :o].T @ ao["f12"] @ c
    a = len(g)
    fit = np.linalg.solve(metric["coulomb"], raw[:, :o].reshape(a, -1)).reshape(g.shape)
    f = (
        c.T @ h @ c
        + 2 * np.einsum("Ppq,Pii->pq", raw, fit[:, :, :o])
        - np.einsum("Pip,Piq->pq", fit, raw[:, :o])
    )
    n = o + v
    fvc = f[o:n, n:]
    result = np.empty((o, o, v, v))
    for i, j in np.ndindex(o, o):
        gem = (
            fit[:, i].T @ g[:, j]
            + g[:, i].T @ fit[:, j]
            - fit[:, i].T @ metric["f12"] @ fit[:, j]
        )
        result[i, j] = gem[o:n, n:] @ fvc.T + fvc @ gem[n:, o:n]
    return result


@pytest.mark.parametrize("o", [1, 3])
def test_c_follows_all_nuclear_inputs(o):
    rng = np.random.default_rng(1179)
    dim, r, v, a = 8, 7, 2, 4

    def sym(t):
        return (t + t.swapaxes(-2, -1)) * 0.5

    c = np.linalg.qr(rng.normal(size=(dim, r)))[0]
    h = sym(rng.normal(scale=0.4, size=(dim, dim)))
    ao = {
        op: sym(rng.normal(scale=0.3, size=(a, dim, dim))) for op in ("coulomb", "f12")
    }
    metric = {op: sym(rng.normal(scale=0.2, size=(a, a))) for op in ao}
    metric["coulomb"] += 3 * np.eye(a)
    weight = rng.normal(size=(o, o, v, v))
    table, bars = c_ao_weights(c, h, ao, metric, weight, execute)
    np.testing.assert_allclose(table, explicit_c(c, h, ao, metric, o, v), atol=2e-14)

    def energy(cc, hh, aa, mm):
        return np.vdot(weight, explicit_c(cc, hh, aa, mm, o, v))

    step = 2e-5
    for field, t in (("coefficients", c), ("hcore", h)):
        d = rng.normal(scale=0.07, size=t.shape)
        if field == "hcore":
            d = sym(d)

        def scalar(s):
            return (
                energy(c + s * d, h, ao, metric)
                if field == "coefficients"
                else energy(c, h + s * d, ao, metric)
            )

        np.testing.assert_allclose(
            np.vdot(bars[field], d),
            (scalar(step) - scalar(-step)) / (2 * step),
            rtol=2e-8,
            atol=2e-10,
        )
    for field, raw in (("ao", ao), ("metric", metric)):
        for op, t in raw.items():
            d = sym(rng.normal(scale=0.07, size=t.shape))

            def scalar(s):
                changed = {**raw, op: t + s * d}
                return (
                    energy(c, h, changed, metric)
                    if field == "ao"
                    else energy(c, h, ao, changed)
                )

            np.testing.assert_allclose(
                np.vdot(bars[field][op], d),
                (scalar(step) - scalar(-step)) / (2 * step),
                rtol=2e-8,
                atol=2e-10,
            )
    with pytest.raises(ValueError, match="Incompatible"):
        c_ao_weights(c, h[:-1], ao, metric, weight, execute)
