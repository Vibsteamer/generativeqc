"""Check the CC two-electron source reverse with a dense spin-orbital DF oracle."""

import numpy as np
from generativeqc_compiler.cc.uccsd_equations import build_program
from generativeqc_compiler.tensor import execute

from tools.generativeqc_cc.uccsd_eri_nuclear import unrestricted_cc_eri_pullback


def test_frozen_unrestricted_eri_reverse() -> None:
    rng = np.random.default_rng(629)
    n, obs, q = 6, 5, 4
    occ = (3, 2)
    frozen = (1, 0)
    c = {s: rng.normal(size=(n, obs)) for s in ("alpha", "beta")}
    g = rng.normal(size=(q, n, n))
    g += g.swapaxes(1, 2)
    m = rng.normal(size=(q, q))
    m = m @ m.T + 2 * np.eye(q)
    shape = build_program(2, 2, 2, 3)
    weights = {
        node.attrs["name"]: rng.normal(size=node.spec.shape)
        for node in shape.live_nodes
        if node.op == "input" and node.attrs["name"].startswith("g")
    }
    bars = unrestricted_cc_eri_pullback(
        occ,
        weights,
        c,
        g,
        m,
        np.linalg.inv(np.linalg.cholesky(m)),
        execute,
        frozen_occupied=frozen,
    )
    variations = {k: rng.normal(size=v.shape) for k, v in bars.items()}
    for k in ("ao_coulomb", "metric/coulomb"):
        variations[k] += variations[k].swapaxes(-1, -2)

    def energy(h: float, varied: str) -> float:
        def delta(k: str) -> np.ndarray | float:
            return h * variations[k] if varied == k else 0.0

        raw = g + delta("ao_coulomb")
        metric = m + delta("metric/coulomb")
        frame = [v + delta("coefficients_" + s) for s, v in c.items()]
        inverse = np.linalg.inv(metric)
        # Independent pair order (p,r) and (q,s). Spin deltas enforce ordinary
        # electron exchange and both occupied partitions are sliced explicitly.
        transformed = [f.T @ raw @ f for f in frame]
        value = 0.0
        for name, w in weights.items():
            pattern, spin = name[1:].split("_")
            sp = [int(s == "b") for s in spin]
            ix = [
                slice(frozen[s], occ[s]) if k == "o" else slice(occ[s], obs)
                for s, k in zip(sp, pattern, strict=True)
            ]
            if sp[0] == sp[2] and sp[1] == sp[3]:
                direct = np.einsum(
                    "Pik,PQ,Qjl->ijkl",
                    transformed[sp[0]][:, ix[0], ix[2]],
                    inverse,
                    transformed[sp[1]][:, ix[1], ix[3]],
                    optimize=True,
                )
                value += np.vdot(w, direct)
            if sp[0] == sp[3] and sp[1] == sp[2]:
                exch = np.einsum(
                    "Pil,PQ,Qjk->ijkl",
                    transformed[sp[0]][:, ix[0], ix[3]],
                    inverse,
                    transformed[sp[1]][:, ix[1], ix[2]],
                    optimize=True,
                )
                value -= np.vdot(w, exch)
        return float(value)

    for k, dx in variations.items():
        eps = 2e-5
        finite = (energy(eps, k) - energy(-eps, k)) / (2 * eps)
        np.testing.assert_allclose(np.vdot(bars[k], dx), finite, rtol=2e-8, atol=3e-6)
