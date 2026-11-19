"""Check precontracted restricted spatial CC-F12 energies against the explicit cusp tensor."""

import numpy as np
from generativeqc_compiler.cc.f12_geminal_energy import all_pair_geminal_energy
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.tensor.ad_program import transpose_program


def independent(s: dict[str, np.ndarray]) -> np.ndarray:
    t1, t2 = s["t1"], s["t2"]
    o = len(t1)
    same = np.eye(o)
    T = 0.375 * np.einsum("mi,nj->mnij", same, same) + 0.125 * np.einsum(
        "mj,ni->mnij", same, same
    )
    W = 2 * T - T.swapaxes(2, 3)
    ein = np.einsum
    tx = ein("mnkl,klij->mnij", s["x"], T)
    common = (
        2 * s["v"]
        + ein("mnkl,klij->mnij", s["b"], T)
        - ein("ik,mnkj->mnij", s["foo"], tx)
        - ein("mnik,jk->mnij", tx, s["foo"])
        + ein("mnab,ijab->mnij", s["c"], t2)
    )
    extra = (
        ein("mnab,ijab->mnij", s["v_vv"], t2 + ein("ia,jb->ijab", t1, t1))
        + ein("mnib,jb->mnij", s["v_ov"], t1)
        + ein("ia,mnaj->mnij", t1, s["v_vo"])
    )
    return np.array([ein("mnij,mnij->", W, z) for z in (common, extra)])


def test_all_pair_geminal_energy_and_both_independent_pullbacks() -> None:
    o, v = 3, 4
    r = np.random.default_rng(70)
    shapes = {
        **dict.fromkeys(("x", "v", "b"), (o,) * 4),
        **dict.fromkeys(("t2", "c", "v_vv"), (o, o, v, v)),
        "v_ov": (o, o, o, v),
        "v_vo": (o, o, v, o),
        "t1": (o, v),
        "foo": (o, o),
    }
    inputs = {k: r.normal(size=shape) for k, shape in shapes.items()}
    graph = all_pair_geminal_energy(o, v)
    e = execute(graph, inputs).outputs
    np.testing.assert_allclose(
        list(e.values()), independent(inputs), rtol=1e-13, atol=2e-13
    )
    delta = {k: r.normal(size=shape) for k, shape in shapes.items()}
    diff = lambda h: (
        (
            independent({k: x + h * delta[k] for k, x in inputs.items()})
            - independent({k: x - h * delta[k] for k, x in inputs.items()})
        )
        / (2 * h)
    )
    adjoint = transpose_program(graph, tuple(e)).program
    for i in (0, 1):
        seeds = {f"bar_{k}": np.array(float(j == i)) for j, k in enumerate(e)}
        bars = execute(adjoint, {**inputs, **seeds}).outputs
        slope = sum(np.vdot(delta[k], bars["bar_" + k]) for k in delta)
        for h in (1e-5, 5e-6):
            np.testing.assert_allclose(slope, diff(h)[i], atol=2e-8, rtol=2e-9)
