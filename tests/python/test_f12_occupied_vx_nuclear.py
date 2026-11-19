"""Contract physical AO/frame variations with independent finite differences of both V/X tables."""

import numpy as np
import pytest

from benchmarks.f12_occupied_vx_nuclear import (
    OPERATORS,
    complementary_frame_tangent,
    occupied_vx_ao_weights,
)
from generativeqc_compiler.tensor import execute


def test_occupied_vx_pullback_all_operators_and_fit() -> None:
    rng = np.random.default_rng(118057)
    o, obs, ri, aux = 2, 4, 6, 5
    c = np.linalg.qr(rng.normal(size=(ri, ri)))[0]
    dc = rng.normal(scale=0.02, size=c.shape)

    def symmetric(x: np.ndarray) -> np.ndarray:
        return 0.5 * (x + x.swapaxes(-2, -1))

    ao = {k: symmetric(rng.normal(scale=0.4, size=(aux, ri, ri))) for k in OPERATORS}
    metrics = {k: symmetric(rng.normal(scale=0.3, size=(aux, aux))) for k in OPERATORS}
    metrics["coulomb"] += 4 * np.eye(aux)
    dao = {
        k: symmetric(rng.normal(scale=0.02, size=val.shape)) for k, val in ao.items()
    }
    dm = {
        k: symmetric(rng.normal(scale=0.02, size=val.shape))
        for k, val in metrics.items()
    }
    weights = {k: rng.normal(size=(o,) * 4) for k in ("v", "x")}
    _, bars = occupied_vx_ao_weights(c, ao, metrics, o, obs, weights, execute)
    assert set(bars["v"]["ao"]) == {"coulomb", "f12", "f12_over_coulomb"}
    assert set(bars["x"]["ao"]) == {"coulomb", "f12", "f12_squared"}
    mask = np.zeros((ri, ri), dtype=bool)
    mask[:obs, :obs] = mask[obs:, :o] = mask[:o, obs:] = True

    def objective(t: float) -> dict[str, float]:
        cc = c + t * dc
        raw = {k: cc[:, :o].T @ (ao[k] + t * dao[k]) @ cc for k in OPERATORS}
        metric = {k: metrics[k] + t * dm[k] for k in OPERATORS}
        fit = np.linalg.solve(
            metric["coulomb"], raw["coulomb"].reshape(aux, -1)
        ).reshape(aux, o, ri)

        def matrix(k: str, i: int, j: int) -> np.ndarray:
            if k == "coulomb":
                return fit[:, i].T @ raw[k][:, j]
            return (
                fit[:, i].T @ raw[k][:, j]
                + raw[k][:, i].T @ fit[:, j]
                - fit[:, i].T @ metric[k] @ fit[:, j]
            )

        values = {"v": 0.0, "x": 0.0}
        for i, j, k, l in np.ndindex((o,) * 4):
            right = matrix("f12", k, l)
            for field, op, left in (
                ("v", "f12_over_coulomb", "coulomb"),
                ("x", "f12_squared", "f12"),
            ):
                values[field] += weights[field][i, j, k, l] * (
                    matrix(op, i, j)[k, l]
                    - np.vdot(matrix(left, i, j)[mask], right[mask])
                )
        return values

    h = 2e-5
    minus, plus = objective(-h), objective(h)
    for key, b in bars.items():
        response = np.vdot(b["coefficients"], dc)
        response += sum(np.vdot(w, dao[k]) for k, w in b["ao"].items())
        response += sum(np.vdot(w, dm[k]) for k, w in b["metric"].items())
        assert response == pytest.approx((plus[key] - minus[key]) / (2 * h), abs=2e-9)


def test_complementary_frame_motion_is_orthonormal_and_gauge_covariant() -> None:
    rng = np.random.default_rng(3382)
    c = np.linalg.qr(rng.normal(size=(7, 7)))[0]
    o = 3
    h = rng.normal(scale=0.02, size=c.shape)
    s = h + h.T
    k = -(c.T @ h @ c)[:o, :o]
    dc = complementary_frame_tangent(c, o, k, s, h)
    np.testing.assert_allclose(dc.T @ c + c.T @ dc + c.T @ s @ c, 0, atol=1e-15)
    np.testing.assert_allclose(dc[:, :o], c[:, :o] @ k, atol=1e-15)
    rotation = np.eye(7)
    rotation[o:, o:] = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    np.testing.assert_allclose(
        complementary_frame_tangent(c @ rotation, o, k, s, h), dc @ rotation, atol=2e-15
    )
    with pytest.raises(ValueError, match="orthonormality"):
        complementary_frame_tangent(c, o, k + 0.2, s, h)
