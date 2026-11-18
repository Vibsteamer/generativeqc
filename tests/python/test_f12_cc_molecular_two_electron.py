"""Differentiate the same fitted ordinary Coulomb tensor the bounded CC driver consumes."""

import numpy as np
import pytest
from generativeqc_compiler.cc.f12_residual import (
    all_pair_cc_residual,
    fixed_pair_cc_residual,
)
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.tensor.ad_program import transpose_program

from benchmarks.f12_h2_cc_molecular_correlation import (
    _small_df_coulomb,
    h2_cc_correlation,
)


def test_bounded_cc_coulomb_adjoint_includes_both_electron_lines() -> None:
    rng = np.random.default_rng(5711)
    fit, raw, dfit, draw = rng.normal(size=(4, 5, 3, 3))
    bar = rng.normal(size=(3, 3, 3, 3))
    graph = _small_df_coulomb(3, 5)
    output = execute(graph, {"fit": fit, "raw": raw}).outputs["eri"]
    expected = np.einsum("pij,pkl->ijkl", fit, raw)
    np.testing.assert_allclose(output, expected, atol=2e-14, rtol=0)
    g = execute(
        transpose_program(graph, ("eri",)).program,
        {"fit": fit, "raw": raw, "bar_eri": bar},
    ).outputs
    actual = np.vdot(dfit, g["bar_fit"]) + np.vdot(draw, g["bar_raw"])
    ref = np.einsum("ijkl,pij,pkl", bar, dfit, raw) + np.einsum(
        "ijkl,pij,pkl", bar, fit, draw
    )
    np.testing.assert_allclose(actual, ref, atol=2e-13, rtol=0)


def test_simultaneous_fixed_pair_rhs_matches_internal_single_pair_schedule() -> None:
    rng = np.random.default_rng(11271)
    o, v = 3, 4
    z = rng.normal(size=(o, o, o + v, o + v))
    z = 0.5 * (z + z.transpose(1, 0, 3, 2))
    c = rng.normal(size=(o, o, v, v))
    c = 0.5 * (c + c.transpose(1, 0, 3, 2))
    t = rng.normal(size=(o, v))
    inp = {
        "v_vv": z[:, :, o:, o:],
        "v_vo": z[:, :, o:, :o],
        "v_ov": z[:, :, :o, o:],
        "c": c,
        "t1": t,
    }
    result = execute(all_pair_cc_residual(o, v), inp).outputs
    ref1 = np.zeros_like(t)
    ref2 = np.zeros_like(c)
    eye = np.eye(o)
    for m, n in zip(*np.triu_indices(o), strict=True):
        feed = {x: inp[x][m, n] for x in ("v_vv", "v_vo", "v_ov")}
        feed.update(select_m=eye[m], fock_coupling_vv=c[m, n], t1=t)
        if m != n:
            feed["select_n"] = eye[n]
        r = execute(fixed_pair_cc_residual(o, v, diagonal=bool(m == n)), feed).outputs
        ref1[m] += r["singles_m"]
        ref2[m, n] = r["doubles_mn"]
        if m != n:
            ref1[n] += r["singles_n"]
            ref2[n, m] = r["doubles_mn"].T
    np.testing.assert_allclose(result["singles"], ref1, atol=2e-14)
    np.testing.assert_allclose(result["doubles"], ref2, atol=2e-14)


@pytest.mark.parametrize("spaces", ((2, 3, 6), (1, 25, 30)))
def test_cc_two_electron_driver_enforces_occupied_and_diagnostic_limits(
    spaces: tuple[int, int, int],
) -> None:
    with pytest.raises(ValueError, match="one occupied pair"):
        h2_cc_correlation({"space": spaces}, {}, np.eye(3), execute, pullback=True)


def test_multipair_source_lagrangian_agrees_with_symmetric_variations() -> None:
    from benchmarks.f12_restricted_multipair_cc import fixed_source_cc_equations
    from benchmarks.f12_restricted_multipair_lambda import (
        prepare_lagrangian_gradient,
        relaxed_source_weights,
    )

    rng = np.random.default_rng(675913)
    o, v = 2, 3
    n = o + v

    def symmetric_pairs(a: np.ndarray) -> np.ndarray:
        return (a + a.transpose(1, 0, 3, 2)) * 0.5

    eri = rng.normal(scale=0.03, size=(n, n, n, n))
    eri = (eri + eri.swapaxes(0, 1)) * 0.5
    eri = (eri + eri.swapaxes(2, 3)) * 0.5
    eri = (eri + eri.transpose(2, 3, 0, 1)) * 0.5
    f = rng.normal(scale=0.03, size=(n, n))
    f = (f + f.T) * 0.5
    source = {
        "f": f,
        "eri": eri,
        "external": symmetric_pairs(rng.normal(size=(o, o, n, n))),
        "c": symmetric_pairs(rng.normal(size=(o, o, v, v))),
    }
    for k in ("v", "b", "x"):
        source[k] = symmetric_pairs(rng.normal(size=(o, o, o, o)))
    t = rng.normal(scale=0.1, size=(o, v))
    tt = symmetric_pairs(rng.normal(scale=0.1, size=(o, o, v, v)))
    l = rng.normal(scale=0.1, size=t.shape)
    ll = symmetric_pairs(rng.normal(scale=0.1, size=tt.shape))
    gradient = prepare_lagrangian_gradient(source, t, tt, execute)
    g1, g2 = gradient(l, ll)
    weights = relaxed_source_weights(source, t, tt, l, ll, execute)
    d = rng.normal(size=t.shape)
    dd = symmetric_pairs(rng.normal(size=tt.shape))
    exact = (
        np.vdot(g1, d)
        + np.vdot(g2, dd)
        + sum(np.vdot(weights[k], x) for k, x in source.items())
    )
    obs = []
    h = 1e-5
    for sign in (-1.0, 1.0):
        out, one, two = fixed_source_cc_equations(
            {k: x * (1 + sign * h) for k, x in source.items()},
            t + sign * h * d,
            tt + sign * h * dd,
            execute,
        )
        obs.append(out.sum() + np.vdot(one, l) + np.vdot(two, ll))
    np.testing.assert_allclose((obs[1] - obs[0]) / (2 * h), exact, atol=2e-8, rtol=0)
