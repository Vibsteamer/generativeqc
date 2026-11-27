"""Independent matrix contractions for all sampled direct/exchange B entries and AO inputs."""

import numpy as np
import pytest
from benchmarks.f12_cc_b_nuclear import fixed_cusp_b_weights
from generativeqc_compiler.tensor import execute


def b_reference(data):
    o, n, r = data["space"]
    c = data["coefficients"]
    raw = c.T @ data["ao_coulomb"] @ c
    a = len(raw)
    fit = np.linalg.solve(data["metric"], raw[:, :o].reshape(a, -1)).reshape(a, o, r)
    exchange = np.einsum("Pip,Piq->pq", fit, raw[:, :o])
    fock = (
        c.T @ data["hcore_ao"] @ c
        + 2 * np.einsum("Ppq,Pii->pq", raw, fit[:, :, :o])
        - exchange
    )
    ops = ("f12", "f12_squared", "double_commutator")
    three = {k: c[:, :o].T @ data["ao_" + k] @ c for k in ops}
    occ, orb, vir, comp = slice(o), slice(n), slice(o, n), slice(n, None)

    def overlap(x, y):
        return (
            -np.vdot(x @ exchange, y)
            - np.vdot(x[occ] @ fock, y[occ])
            + np.vdot(x[comp, occ] @ fock[occ, occ], y[comp, occ])
            - np.vdot(x[vir, orb] @ fock[orb, orb], y[vir, orb])
            - 2 * np.vdot(x[comp] @ fock[occ].T, y[comp, occ])
            - 2 * np.vdot(x[vir, orb] @ fock[orb, comp], y[vir, comp])
        )

    values = {}
    for i, j in np.ndindex(o, o):
        mats = {
            k: fit[:, i].T @ three[k][:, j]
            + three[k][:, i].T @ fit[:, j]
            - fit[:, i].T @ data["metric_" + k] @ fit[:, j]
            for k in ops
        }
        g, sq, comm = (mats[k] for k in ops)
        for m, k, b, t in ((i, j, g, g.T), (j, i, g.T, g)):
            values[i, j, m, k] = (
                comm[m, k]
                + sq[m] @ (fock + exchange)[k]
                + sq[:, k] @ (fock + exchange)[m]
                + overlap(g, b)
                + overlap(g.T, t)
            )
    return values


def test_sampled_b_all_radial_fock_and_frame_derivatives():
    rng = np.random.default_rng(9191)
    o, n, r, a = 2, 4, 6, 4

    def sym(t):
        return (t + t.swapaxes(-2, -1)) * 0.5

    data = {
        "space": [o, n, r],
        "coefficients": np.linalg.qr(rng.normal(size=(r, r)))[0],
        "hcore_ao": sym(rng.normal(scale=0.3, size=(r, r))),
    }
    for op in (
        "coulomb",
        "f12",
        "f12_squared",
        "f12_over_coulomb",
        "double_commutator",
    ):
        data["ao_" + op] = sym(rng.normal(scale=0.25, size=(a, r, r)))
        key = "metric" if op == "coulomb" else "metric_" + op
        data[key] = sym(rng.normal(scale=0.2, size=(a, a))) + (
            3 * np.eye(a) if op == "coulomb" else 0
        )
    reference = b_reference(data)
    w = np.zeros((o,) * 4)
    for idx in reference:
        w[idx] = rng.normal()
    values, bars = fixed_cusp_b_weights(data, w, execute)
    assert set(values) == set(reference)
    for idx, value in values.items():
        np.testing.assert_allclose(value, reference[idx], atol=2e-14)
    gradients = {"coefficients": bars["coefficients"], "hcore_ao": bars["hcore"]}
    for kind in ("ao", "metric"):
        for op, t in bars[kind].items():
            key = "metric" if (kind, op) == ("metric", "coulomb") else kind + "_" + op
            gradients[key] = t
    for key, t in gradients.items():
        direction = rng.normal(scale=0.04, size=t.shape)
        if key != "coefficients":
            direction = sym(direction)

        def scalar(step):
            moved = b_reference({**data, key: data[key] + step * direction})
            return sum(w[idx] * v for idx, v in moved.items())

        step = 2e-5
        np.testing.assert_allclose(
            np.vdot(t, direction),
            (scalar(step) - scalar(-step)) / (2 * step),
            atol=2e-9,
            rtol=2e-8,
        )
    w[0, 0, 1, 0] = 1.0
    with pytest.raises(ValueError, match="interpair"):
        fixed_cusp_b_weights(data, w, execute)
