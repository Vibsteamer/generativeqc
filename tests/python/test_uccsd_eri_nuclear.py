"""Check the CC two-electron source reverse with a dense spin-orbital DF oracle."""

import numpy as np
from generativeqc._uccsd_df_hamiltonian import unrestricted_cc_df_hamiltonian
from generativeqc_compiler.cc.uccsd_df_virtual_ladder import (
    FACTOR_INPUTS,
    build_uccsd_df_virtual_ladder_amplitude_programs,
)
from generativeqc_compiler.cc.uccsd_equations import build_program
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.tensor import execute, transpose_program

from tools.generativeqc_cc.uccsd_eri_nuclear import unrestricted_cc_eri_pullback


def test_bounded_pair_projection_precedes_ill_conditioned_fit() -> None:
    rng = np.random.default_rng(637)
    obs, aux = 4, 3
    raw = rng.normal(size=(aux, obs, obs))
    raw = (raw + raw.swapaxes(1, 2)) / 2
    raw[0, 2, 3] += 2e-12
    whitener = np.diag([1e4, 1.0, 1.0])
    projected = (raw + raw.swapaxes(1, 2)) / 2
    expected_fit = np.linalg.solve(
        np.diag([1e-8, 1.0, 1.0]), projected.reshape(aux, -1)
    ).reshape(aux, obs, obs)
    assert abs(raw[0, 2, 3] - raw[0, 3, 2]) / 1e-8 > 1e-6
    fock = np.diag(np.linspace(-1, 1, obs))
    factors = {}
    unrestricted_cc_df_hamiltonian(
        (2, 2, obs, obs, aux),
        {"source/raw_coulomb_alpha": raw, "source/raw_coulomb_beta": raw},
        {"fock_alpha": fock, "fock_beta": fock},
        whitener,
        execute,
        retain_rhf_virtual_factors=factors,
    )
    np.testing.assert_array_equal(factors["raw_right"], projected[:, 2:, 2:])
    np.testing.assert_array_equal(
        factors["fit_left"], factors["fit_left"].swapaxes(1, 2)
    )
    np.testing.assert_allclose(
        factors["fit_left"], expected_fit[:, 2:, 2:], rtol=2e-15
    )


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


def test_virtual_factor_bars_share_the_ordinary_ao_and_metric_pullback() -> None:
    rng = np.random.default_rng(633)
    nao, obs, aux = 6, 5, 5
    occupied, frozen = (3, 2), (1, 0)
    layout = UCCSDLayout(2, 2, 2, 3)
    spins = ("alpha", "beta")
    coefficients = {spin: rng.normal(scale=0.2, size=(nao, obs)) for spin in spins}
    ao = rng.normal(scale=0.3, size=(aux, nao, nao))
    ao += ao.swapaxes(1, 2)
    metric = rng.normal(scale=0.2, size=(aux, aux))
    metric = metric @ metric.T + np.eye(aux)
    whitener = np.linalg.inv(np.linalg.cholesky(metric))
    amplitudes = layout.unpack(rng.normal(scale=0.1, size=layout.packed_size))
    seeds = {
        "bar_" + name.replace("t", "r", 1): value
        for name, value in layout.unpack_dual(
            rng.normal(size=layout.packed_size)
        ).items()
    }

    full = build_program(layout.oa, layout.ob, layout.va, layout.vb)
    virtual_names = tuple(
        node.attrs["name"]
        for node in full.live_nodes
        if node.op == "input" and node.attrs["name"].startswith("gvvvv_")
    )
    full_action = transpose_program(
        full, ("r2aa", "r2ab", "r2bb"), inputs=virtual_names
    ).program
    dense_bars = {
        name.removeprefix("bar_"): value
        for name, value in execute(full_action, {**amplitudes, **seeds}).outputs.items()
    }
    other_bars = {
        node.attrs["name"]: rng.normal(scale=0.2, size=node.spec.shape)
        for node in full.live_nodes
        if node.op == "input"
        and node.attrs["name"].startswith("g")
        and node.attrs["name"] not in virtual_names
    }

    factors = {}
    data = {
        "source/raw_coulomb_" + spin: frame.T @ ao @ frame
        for spin, frame in coefficients.items()
    }
    fock = {"fock_" + spin: np.diag(np.linspace(-1, 1, obs)) for spin in spins}
    bounded = unrestricted_cc_df_hamiltonian(
        (*occupied, obs, obs, aux),
        data,
        fock,
        whitener,
        execute,
        frozen_occupied=frozen,
        retain_uhf_virtual_factors=factors,
    )
    dense = unrestricted_cc_df_hamiltonian(
        (*occupied, obs, obs, aux),
        data,
        fock,
        whitener,
        execute,
        frozen_occupied=frozen,
    )
    assert dense.keys() - bounded.keys() == set(virtual_names)
    for name, value in bounded.items():
        np.testing.assert_allclose(value, dense[name], rtol=0, atol=2e-15)
    assert all(not factors[name].flags.writeable for name in FACTOR_INPUTS)
    factor_bars = {name: np.empty_like(value) for name, value in factors.items()}
    for start, stop in ((0, 2), (2, 4), (4, 5)):
        graph = build_uccsd_df_virtual_ladder_amplitude_programs(
            layout, stop - start
        ).factor_vjp.program
        slab = {name: factors[name][start:stop] for name in FACTOR_INPUTS}
        bars = execute(graph, {**amplitudes, **slab, **seeds}).outputs
        for name in FACTOR_INPUTS:
            factor_bars[name][start:stop] = bars["bar_" + name]

    expected = unrestricted_cc_eri_pullback(
        occupied,
        {**other_bars, **dense_bars},
        coefficients,
        ao,
        metric,
        whitener,
        execute,
        frozen_occupied=frozen,
    )
    calculated = unrestricted_cc_eri_pullback(
        occupied,
        other_bars,
        coefficients,
        ao,
        metric,
        whitener,
        execute,
        frozen_occupied=frozen,
        virtual_factor_weights=factor_bars,
    )
    dense_ao = expected["ao_coulomb"]
    expected["ao_coulomb"] = (dense_ao + dense_ao.swapaxes(1, 2)) / 2
    for name, value in expected.items():
        np.testing.assert_allclose(calculated[name], value, rtol=0, atol=2e-13)
