"""Independent full-four-index oracle for the external CC-F12 projected V response."""

import numpy as np
import pytest

from benchmarks.f12_cc_external_nuclear import EXTERNAL_OPERATORS, external_v_ao_weights
from generativeqc_compiler.tensor import execute


def explicit_external(c, aos, metrics, o):
    raw = {op: c.T @ x @ c for op, x in aos.items()}
    a, n, _ = raw["coulomb"].shape
    fit = np.linalg.solve(metrics["coulomb"], raw["coulomb"].reshape(a, -1)).reshape(
        a, n, n
    )
    g = (fit.reshape(a, -1).T @ raw["coulomb"].reshape(a, -1)).reshape((n,) * 4)
    mats = {}
    occupied = fit[:, :, :o].transpose(0, 2, 1)
    for op in ("f12", "f12_over_coulomb"):
        three = raw[op][:, :o]
        mats[op] = (
            np.einsum("Pkr,Pls->klrs", occupied, three)
            + np.einsum("Pkr,Pls->klrs", three, occupied)
            - np.einsum("Pkr,PQ,Qls->klrs", occupied, metrics[op], occupied)
        )
    return mats["f12_over_coulomb"] - np.einsum("prqs,klrs->klpq", g, mats["f12"])


@pytest.mark.parametrize("occupied", [1, 3])
def test_external_ao_metric_and_frame_response(occupied):
    rng = np.random.default_rng(5542)
    dim, n, a = 7, 5, 4
    c = np.linalg.qr(rng.normal(size=(dim, n)))[0]

    def sym(x):
        return (x + x.swapaxes(-2, -1)) / 2

    ao = {k: sym(rng.normal(scale=0.3, size=(a, dim, dim))) for k in EXTERNAL_OPERATORS}
    metrics = {k: sym(rng.normal(scale=0.2, size=(a, a))) for k in EXTERNAL_OPERATORS}
    metrics["coulomb"] += 4 * np.eye(a)
    w = rng.normal(size=(occupied, occupied, n, n))
    value, source = external_v_ao_weights(c, ao, metrics, w, execute)
    np.testing.assert_allclose(
        value, explicit_external(c, ao, metrics, occupied), atol=2e-14
    )

    def energy(c, ao, metric):
        return np.vdot(w, explicit_external(c, ao, metric, occupied))

    step = 3e-5
    for kind, tensors in (("ao", ao), ("metric", metrics)):
        for op in EXTERNAL_OPERATORS:
            direction = sym(rng.normal(scale=0.08, size=tensors[op].shape))

            def shifted(sign):
                changed = {**tensors, op: tensors[op] + sign * step * direction}
                return (
                    energy(c, changed, metrics)
                    if kind == "ao"
                    else energy(c, ao, changed)
                )

            numerical = (shifted(1) - shifted(-1)) / (2 * step)
            np.testing.assert_allclose(
                np.vdot(source[kind][op], direction), numerical, rtol=2e-8, atol=2e-9
            )
    dc = rng.normal(scale=0.05, size=c.shape)
    numerical = (
        energy(c + step * dc, ao, metrics) - energy(c - step * dc, ao, metrics)
    ) / (2 * step)
    np.testing.assert_allclose(
        np.vdot(source["coefficients"], dc), numerical, rtol=2e-8, atol=2e-9
    )
    with pytest.raises(ValueError, match="dimensions"):
        external_v_ao_weights(c, ao, metrics, w[:, :, :-1], execute)
