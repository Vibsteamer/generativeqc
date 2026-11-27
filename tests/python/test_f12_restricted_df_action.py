"""Check the F12 driver's ordinary DF staging and both adjoints against four-index RCCSD."""

import numpy as np
from generativeqc_compiler.cc.doubles import build_ccsd_program
from generativeqc_compiler.tensor import execute

from benchmarks.f12_restricted_df_action import RestrictedDFAction
from benchmarks.f12_restricted_multipair_cc import conventional_inputs


def test_df_actions_use_every_q_and_return_relaxed_factor_and_amplitude_weights() -> (
    None
):
    o, v, q = 2, 3, 7
    rng = np.random.default_rng(618)
    f = rng.normal(size=(o + v,) * 2)
    f = (f + f.T) * 0.5

    def symmetric(shape: tuple[int, ...]) -> np.ndarray:
        z = rng.normal(size=shape)
        return 0.5 * (z + z.swapaxes(1, 2))

    B = symmetric((q, o + v, o + v))
    t1 = rng.normal(size=(o, v))
    t2 = rng.normal(size=(o, o, v, v))
    t2 = 0.5 * (t2 + t2.transpose(1, 0, 3, 2))
    l1 = rng.normal(size=t1.shape)
    l2 = rng.normal(size=t2.shape)
    cc = build_ccsd_program(o, v, form="optimized", diagnostics=False)

    def dense(fock: np.ndarray, b: np.ndarray, t: np.ndarray, tt: np.ndarray) -> dict:
        g = np.einsum("Qij,Qkl->ijkl", b, b)
        h = conventional_inputs({"eri": g, "f": fock}, o)
        return execute(cc, {**h, "t1": t, "t2": tt}).outputs

    act = RestrictedDFAction(f, B, o, execute, batch_size=2)
    truth = dense(f, B, t1, t2)
    res = act.evaluate(t1, t2)
    for k in truth:
        np.testing.assert_allclose(res[k], truth[k], rtol=1e-12, atol=1e-12)
    ht = act.adjoint(t1, t2, l1, l2, amplitudes=True)
    hf = act.adjoint(t1, t2, l1, l2, amplitudes=False)
    df = symmetric((1, o + v, o + v))[0]
    db = symmetric(B.shape)
    dt = rng.normal(size=t1.shape)
    dtt = rng.normal(size=t2.shape)
    dtt = 0.5 * (dtt + dtt.transpose(1, 0, 3, 2))

    def energy(h: float) -> float:
        r = dense(f + h * df, B + h * db, t1 + h * dt, t2 + h * dtt)
        return float(
            r["correlation_energy"]
            + np.vdot(l1, r["singles_residual"])
            + np.vdot(l2, r["doubles_residual"])
        )

    slope = (
        np.vdot(hf["f"], df)
        + np.vdot(hf["bmo"], db)
        + np.vdot(ht["t1"], dt)
        + np.vdot(ht["t2"], dtt)
    )
    for h in (1e-5, 5e-6):
        np.testing.assert_allclose(
            slope, (energy(h) - energy(-h)) / (2 * h), rtol=2e-8, atol=1e-8
        )


def test_full_rank_metric_response_is_gauge_invariant_for_a_gram_functional() -> None:
    from generativeqc_compiler.cc.f12_factor_response import (
        coulomb_gram_factor_response,
    )

    r = np.random.default_rng(619)
    q, p = 4, 7
    z = r.normal(size=(q, q))
    j = z @ z.T + 0.1 * np.eye(q)
    g = r.normal(size=(q, p))
    u = r.normal(size=(p, p))
    u = (u + u.T) * 0.5
    l = np.linalg.cholesky(j)
    ww = np.linalg.solve(l, np.eye(q))
    d = np.linalg.solve(j, g)
    graph = coulomb_gram_factor_response(q, p)
    delta = r.normal(size=g.shape)
    dj = r.normal(size=j.shape)
    dj = (dj + dj.T) * 0.5

    def observable(h: float) -> float:
        raw = g + h * delta
        return float(np.sum(u * (raw.T @ np.linalg.solve(j + h * dj, raw))))

    for w in (ww, np.linalg.qr(r.normal(size=(q, q)))[0] @ ww):
        a = w @ g
        res = execute(
            graph, {"whitener": w, "raw": g, "whitened_bar": 2 * a @ u, "fitted": d}
        ).outputs
        np.testing.assert_allclose(res["whitened"], a, atol=1e-12, rtol=0)
        np.testing.assert_allclose(res["raw_bar"], 2 * d @ u, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(
            res["metric_bar"], -d @ u @ d.T, rtol=2e-13, atol=2e-13
        )
        slope = np.vdot(delta, res["raw_bar"]) + np.vdot(dj, res["metric_bar"])
        for h in (1e-5, 5e-6):
            two_step = (
                8 * (observable(h) - observable(-h))
                - (observable(2 * h) - observable(-2 * h))
            ) / (12 * h)
            np.testing.assert_allclose(slope, two_step, rtol=2e-9, atol=1e-8)
