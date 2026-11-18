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
