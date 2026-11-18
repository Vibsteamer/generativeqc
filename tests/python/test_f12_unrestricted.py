"""Spin-dependent extended Coulomb/exchange and F12 projector controls."""

from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.mp2.f12_molecular_pair import extended_df_fock
from generativeqc_compiler.mp2.f12_unrestricted import (
    projected_same_spin_f12_pair,
    unrestricted_extended_df_fock,
    unrestricted_pair_coupling,
    unrestricted_pair_vx,
)
from generativeqc_compiler.tensor import Program, execute, transpose_program


def _check_vjp(
    graph: Program, feeds: dict[str, np.ndarray], rng: np.random.Generator
) -> None:
    outputs = execute(graph, feeds).outputs
    seeds = {k: rng.normal(size=v.shape) for k, v in outputs.items()}
    reverse = transpose_program(graph, tuple(seeds)).program
    bars = execute(
        reverse, {**feeds, **{"bar_" + k: v for k, v in seeds.items()}}
    ).outputs
    direction = {k: rng.normal(size=v.shape) for k, v in feeds.items()}
    predicted = sum(np.vdot(bars["bar_" + k], v) for k, v in direction.items())

    def scalar(displacement: float) -> float:
        at = {k: v + displacement * direction[k] for k, v in feeds.items()}
        return sum(np.vdot(seeds[k], v) for k, v in execute(graph, at).outputs.items())

    for step in (1e-4, 3e-5):
        np.testing.assert_allclose(
            predicted, (scalar(step) - scalar(-step)) / (2 * step), atol=2e-8, rtol=0
        )


@pytest.mark.parametrize("occs", [(3, 2), (1, 0), (0, 1), (2, 2)])
def test_extended_jk_against_full_ao_density(occs: tuple[int, int]) -> None:
    rng = np.random.default_rng(901)
    n, a = 6, 7
    h = rng.normal(size=(n, n))
    h = (h + h.T) / 2
    factors = rng.normal(size=(a, n, n))
    factors = (factors + factors.swapaxes(1, 2)) / 2
    # Nonorthogonal fit metric. The oracle contracts the full AO ERI tensor.
    metric = rng.normal(size=(a, a))
    metric = metric @ metric.T + np.eye(a)
    fit = np.linalg.solve(metric, factors.reshape(a, -1)).reshape(factors.shape)
    eri = np.einsum("apq,ars->pqrs", factors, fit)
    ca, cb = (np.linalg.qr(rng.normal(size=(n, n)))[0] for _ in range(2))
    if occs == (2, 2):
        cb = ca
    da, db = (c[:, :o] @ c[:, :o].T for c, o in zip((ca, cb), occs, strict=True))
    j = np.einsum("pqrs,sr->pq", eri, da + db)
    graph = unrestricted_extended_df_fock(*occs, n, a)
    feeds, reference = {}, {}
    for spin, c, o, dm in zip(("alpha", "beta"), (ca, cb), occs, (da, db), strict=True):
        feeds["hcore_" + spin] = c.T @ h @ c
        feeds["raw_coulomb_" + spin] = c.T @ factors @ c
        if o:
            feeds["fit_" + spin] = c[:, :o].T @ fit @ c
        k = np.einsum("prsq,rs->pq", eri, dm)
        reference["fock_" + spin] = c.T @ (h + j - k) @ c
        reference["exchange_" + spin] = c.T @ k @ c
    answer = execute(graph, feeds).outputs
    for key, value in reference.items():
        np.testing.assert_allclose(answer[key], value, atol=2e-13, rtol=0)
    if occs == (2, 2):
        closed = execute(
            extended_df_fock(2, n, a),
            {
                "hcore": feeds["hcore_alpha"],
                "raw_coulomb_all": feeds["raw_coulomb_alpha"],
                "fit": feeds["fit_alpha"],
            },
        ).outputs
        np.testing.assert_allclose(
            answer["fock_beta"], closed["fock"], atol=2e-13, rtol=0
        )
    _check_vjp(graph, feeds, rng)


@pytest.mark.parametrize("channel", ["alpha_alpha", "alpha_beta", "beta_beta"])
def test_strong_orthogonality_in_separate_orbital_spaces(channel: str) -> None:
    rng = np.random.default_rng(613)
    left = (2, 2, 1)
    right = (1, 2, 2) if channel == "alpha_beta" else left
    nl, nr = sum(left), sum(right)
    bra_g, bra_f, ket_f = rng.normal(size=(3, nl, nr))

    # Form Q explicitly on the two-particle space for an independent oracle.
    def projectors(space: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        o, v, c = space
        return np.diag([1] * o + [0] * (v + c)), np.diag([0] * o + [1] * v + [0] * c)

    ol, vl = projectors(left)
    ort, vr = projectors(right)
    q = np.kron(np.eye(nl) - ol, np.eye(nr) - ort) - np.kron(vl, vr)
    projected_ket = q @ ket_f.ravel()
    feeds = {
        "geminal_coulomb_element": np.vdot(bra_g, ket_f),
        "squared_geminal_element": np.vdot(bra_f, ket_f),
    }
    lo, lb, ro, rb = left[0], sum(left[:2]), right[0], sum(right[:2])
    for name, source in (
        ("ket_geminal", ket_f),
        ("bra_geminal", bra_f),
        ("bra_coulomb", bra_g),
    ):
        feeds[name + "_obs_obs"] = source[:lb, :rb]
        feeds[name + "_occ_cabs"] = source[:lo, rb:]
        feeds[name + "_cabs_occ"] = source[lb:, :ro]
    graph = unrestricted_pair_vx(left, right, channel=channel)
    answer = execute(graph, feeds).outputs
    np.testing.assert_allclose(
        answer["v"], np.vdot(bra_g.ravel(), projected_ket), atol=2e-13
    )
    np.testing.assert_allclose(
        answer["x"], np.vdot(bra_f.ravel(), projected_ket), atol=2e-13
    )
    _check_vjp(graph, feeds, rng)


def test_invalid_orbital_spaces() -> None:
    with pytest.raises(ValueError):
        unrestricted_extended_df_fock(0, 0, 3, 4)
    with pytest.raises(ValueError):
        unrestricted_pair_vx((2, 2, 1), (1, 2, 2), channel="alpha_alpha")


@pytest.mark.parametrize("channel", ["alpha_beta", "alpha_alpha", "beta_beta"])
def test_coupling_and_gap_on_full_two_electron_space(channel: str) -> None:
    rng = np.random.default_rng(31763)
    left = (2, 2, 3)
    right = (1, 3, 2) if channel == "alpha_beta" else left
    n, m = sum(left), sum(right)
    sl, sr = channel.split("_")
    fl, fr = (rng.normal(size=(size, size)) for size in (n, m))
    fl = (fl + fl.T) / 2
    fr = fl if sl == sr else (fr + fr.T) / 2
    gem = rng.normal(size=(n, m))
    ol, vl, _ = left
    ore, vr, _ = right
    vsl = slice(ol, ol + vl)
    vsr = slice(ore, ore + vr)
    gel = np.diag(fl)[vsl]
    ger = np.diag(fr)[vsr]
    ei, ej = np.asarray(fl[ol - 1, ol - 1]), np.asarray(fr[0, 0])
    inputs = {
        "occupied_energy_i": ei,
        "occupied_energy_j": ej,
        "virtual_energies_" + sl: gel,
        "virtual_energies_" + sr: ger,
        "virtual_cabs_fock_" + sl: fl[vsl, ol + vl :],
        "virtual_cabs_fock_" + sr: fr[vsr, ore + vr :],
        "geminal_virtual_cabs": gem[vsl, ore + vr :],
        "geminal_cabs_virtual": gem[ol + vl :, vsr],
    }
    va, vb = np.eye(n)[:, vsl], np.eye(m)[:, vsr]
    olp, orp = np.eye(n)[:, :ol], np.eye(m)[:, :ore]
    p = np.kron(np.eye(n) - olp @ olp.T, np.eye(m) - orp @ orp.T) - np.kron(
        va @ va.T, vb @ vb.T
    )
    f = np.kron(fl, np.eye(m)) + np.kron(np.eye(n), fr)
    ref = (f @ p @ gem.ravel()).reshape(n, m)[vsl, vsr]
    plan = unrestricted_pair_coupling(left, right, channel=channel)
    actual = execute(plan, inputs).outputs
    np.testing.assert_allclose(actual["c"], ref, atol=1e-13, rtol=0)
    np.testing.assert_allclose(
        actual["gap"], gel[:, None] + ger[None, :] - ei - ej, atol=1e-13, rtol=0
    )
    _check_vjp(plan, inputs, rng)


def test_molecular_high_spin_pair() -> None:
    fixture = (
        Path(__file__).parents[1] / "reference_data/f12/psi4_h2_triplet_projected.npz"
    )
    with np.load(fixture, allow_pickle=False) as ds:
        graph = projected_same_spin_f12_pair(*map(int, ds["space"]))
        ref = float(ds["reference_pair_energy"])
        feeds = {
            k: ds[k]
            for k in ds
            if k not in {"reference_pair_energy", "space", "psi4_ump2_total", "beta"}
        }
    terms = execute(graph, feeds).outputs
    value = float(terms["pair_energy"])
    # One virtual determinant in this (2 occupied, 2 orbital virtual) triplet.
    v = feeds["coulomb_obs"][2:, 2:]
    energies = np.diag(feeds["fock"])
    gap = energies[2:4].sum() - energies[:2].sum()
    np.testing.assert_allclose(
        terms["ordinary_pair_energy"], -((v[0, 1] - v[1, 0]) ** 2) / gap, atol=1e-14
    )
    assert abs(ref) > 1e-4
    np.testing.assert_allclose(value, ref, atol=2e-11, rtol=0)
    reverse = transpose_program(graph, ("pair_energy",)).program
    weights = execute(reverse, {**feeds, "bar_pair_energy": np.array(1.0)}).outputs
    rng = np.random.default_rng(41713)
    dsources = {}
    for key, source in feeds.items():
        if key.startswith("select"):
            continue
        tangent = rng.normal(size=source.shape)
        if key in ("fock", "exchange"):
            tangent = (tangent + tangent.T) / 2
        dsources[key] = tangent / max(1.0, np.linalg.norm(tangent))
    slope = sum(float(np.vdot(weights["bar_" + k], d)) for k, d in dsources.items())

    def at(t: float) -> float:
        changed = {**feeds, **{k: feeds[k] + t * d for k, d in dsources.items()}}
        return float(execute(graph, changed).outputs["pair_energy"])

    for h in (3e-4, 1e-4):
        np.testing.assert_allclose(slope, (at(h) - at(-h)) / (2 * h), atol=2e-8, rtol=0)


def test_opposite_spin_source_and_orbital_overlap_pullback() -> None:
    # Form the physical ket in a common spatial basis, using the explicit
    # singlet/triplet spatial projectors before transforming the two legs.
    from generativeqc_compiler.mp2.f12_unrestricted import opposite_spin_cusp_source

    rng = np.random.default_rng(53412)
    n = 6
    left, right = (np.linalg.qr(rng.normal(size=(n, n)))[0] for _ in range(2))
    i, j = 1, 3
    pair = np.outer(left[:, i], right[:, j])
    sym, antisym = (pair + pair.T) / 2, (pair - pair.T) / 2
    answers = left.T @ (0.5 * sym + 0.25 * antisym) @ right
    graph = opposite_spin_cusp_source(n)
    inputs = {
        "cross_overlap": left.T @ right,
        "select_alpha": np.eye(n)[i],
        "select_beta": np.eye(n)[j],
    }
    actual = execute(graph, inputs).outputs
    np.testing.assert_allclose(actual["cusp_source"], answers, atol=1e-14)
    seed = rng.normal(size=(n, n))
    reverse = transpose_program(graph, ("cusp_source",)).program
    bar = execute(reverse, {**inputs, "bar_cusp_source": seed}).outputs[
        "bar_cross_overlap"
    ]
    direction = rng.normal(size=(n, n))

    def at(t: float) -> float:
        result = execute(
            graph, {**inputs, "cross_overlap": inputs["cross_overlap"] + t * direction}
        ).outputs
        return float(np.vdot(result["cusp_source"], seed))

    for dt in (3e-4, 1e-4):
        np.testing.assert_allclose(
            (at(dt) - at(-dt)) / (2 * dt), np.vdot(bar, direction), atol=2e-10
        )


def test_physical_mixed_pair_against_dense_determinant_space() -> None:
    from generativeqc_compiler.mp2.f12_opposite_spin import opposite_spin_projected_pair

    path = Path(__file__).parents[1] / "reference_data/f12/psi4_nh2_uhf_mixed_pair.npz"
    with np.load(path, allow_pickle=False) as data:
        graph = opposite_spin_projected_pair(*map(int, data["space"]))
        fields = {
            k: data[k]
            for k in data
            if k not in {"space", "expected_pair", "expected_ump2"}
        }
        result = execute(graph, fields).outputs
        np.testing.assert_allclose(
            result["ordinary_pair_energy"], data["expected_ump2"], atol=2e-11, rtol=0
        )
        np.testing.assert_allclose(
            result["pair_energy"], data["expected_pair"], atol=3e-10, rtol=0
        )
    reverse = transpose_program(graph, ("pair_energy",)).program
    bar = execute(reverse, {**fields, "bar_pair_energy": np.array(1.0)}).outputs
    rng = np.random.default_rng(23)
    dirs = {}
    for k, v in fields.items():
        if k.startswith("select"):
            continue
        d = rng.normal(size=v.shape)
        if k.startswith(("fock", "exchange")):
            d = (d + d.T) / 2
        dirs[k] = d / max(1.0, np.linalg.norm(d))
    pred = sum(float(np.vdot(bar["bar_" + k], y)) for k, y in dirs.items())

    def value(t: float) -> float:
        return float(
            execute(
                graph, {**fields, **{k: fields[k] + t * v for k, v in dirs.items()}}
            ).outputs["pair_energy"]
        )

    for h in (1e-4, 3e-5):
        np.testing.assert_allclose(
            pred, (value(h) - value(-h)) / (2 * h), atol=2e-8, rtol=0
        )


def test_spin_resolved_robust_factors_propagate_both_spin_sources() -> None:
    from generativeqc_compiler.mp2.f12_opposite_spin import opposite_spin_projected_pair

    rng = np.random.default_rng(668013)
    na, nb, obs, n, aux = 2, 1, 4, 5, 6
    overlap = np.linalg.qr(rng.normal(size=(n, n)))[0]
    base = {
        "cross_overlap": overlap,
        "select_i": np.eye(n)[1],
        "select_j": np.eye(n)[0],
    }
    for spin in ("alpha", "beta"):
        d = rng.normal(size=(n, n)) * 0.025
        base["fock_" + spin] = np.diag(np.linspace(-1.6, 1.3, n)) + (d + d.T) / 2
        k = rng.normal(size=(n, n)) * 0.02
        base["exchange_" + spin] = (k + k.T) / 2
    dl, dr = rng.normal(size=(2, aux, n)) * 0.1
    fit = {**base, "fit_i_alpha": dl, "fit_j_beta": dr}
    dense = dict(base)
    fit["raw_coulomb_j_beta"] = rng.normal(size=(aux, n)) * 0.1
    dense["coulomb"] = dl.T @ fit["raw_coulomb_j_beta"]
    for name in ("geminal", "geminal_coulomb", "squared_geminal", "double_commutator"):
        a, b = rng.normal(size=(2, aux, n)) * 0.1
        m = rng.normal(size=(aux, aux)) * 0.1
        m = (m + m.T) / 2
        fit["raw_" + name + "_i_alpha"] = a
        fit["raw_" + name + "_j_beta"] = b
        fit["metric_" + name] = m
        dense[name] = dl.T @ b + a.T @ dr - dl.T @ m @ dr
    graph = opposite_spin_projected_pair(na, nb, obs, n, auxiliaries=aux)
    ref = execute(opposite_spin_projected_pair(na, nb, obs, n), dense).outputs
    answer = execute(graph, fit).outputs
    for k, v in ref.items():
        np.testing.assert_allclose(answer[k], v, rtol=0, atol=2e-13)
    bar = execute(
        transpose_program(graph, ("pair_energy",)).program,
        {**fit, "bar_pair_energy": np.array(1.0)},
    ).outputs
    dirs = {
        k: rng.normal(size=v.shape)
        for k, v in fit.items()
        if k.startswith(("fit_", "metric_", "raw_"))
    }
    for k, d in dirs.items():
        dirs[k] = d / max(1.0, np.linalg.norm(d))
    predicted = sum(float(np.vdot(bar["bar_" + k], d)) for k, d in dirs.items())

    def perturb(t: float) -> float:
        return float(
            execute(
                graph, {**fit, **{k: fit[k] + t * v for k, v in dirs.items()}}
            ).outputs["pair_energy"]
        )

    for h in (1.0e-4, 3e-5):
        np.testing.assert_allclose(
            (perturb(h) - perturb(-h)) / (2 * h), predicted, atol=2e-8, rtol=0
        )


@pytest.mark.parametrize("occs", [(2, 1), (2, 0), (0, 1)])
def test_unrestricted_df_orbital_action_against_spin_density(
    occs: tuple[int, int],
) -> None:
    from generativeqc_compiler.mp2.f12_unrestricted_reference import (
        unrestricted_df_orbital_action,
    )

    rng = np.random.default_rng(734)
    n, aux = 7, 11
    sources = rng.normal(size=(aux, n, n))
    sources = (sources + sources.swapaxes(1, 2)) / 2
    coeff = [np.linalg.qr(rng.normal(size=(n, n)))[0] for _ in occs]
    dms = []
    inputs = {}
    for spin, o, c in zip(("alpha", "beta"), occs, coeff, strict=True):
        ov = c[:, o:].T @ sources @ c[:, :o]
        x = rng.normal(size=(n - o, o))
        rot = c[:, o:] @ x @ c[:, :o].T
        dms.append(rot + rot.T)
        if o:
            inputs["vo_" + spin] = ov
            inputs["oo_" + spin] = c[:, :o].T @ sources @ c[:, :o]
            inputs["vv_" + spin] = c[:, o:].T @ sources @ c[:, o:]
            inputs["trial_" + spin] = x
            inputs["gap_" + spin] = rng.uniform(0.3, 5, x.shape)
    out = execute(unrestricted_df_orbital_action(*occs, n, aux), inputs).outputs
    eri = np.einsum("Apq,Ars->pqrs", sources, sources)
    j = np.einsum("pqrs,rs->pq", eri, sum(dms))
    for spin, o, c, dm in zip(("alpha", "beta"), occs, coeff, dms, strict=True):
        if o:
            k = np.einsum("prsq,rs->pq", eri, dm)
            expected = (
                c[:, o:].T @ (j - k) @ c[:, :o]
                + inputs["gap_" + spin] * inputs["trial_" + spin]
            )
            np.testing.assert_allclose(out[spin], expected, atol=2e-13, rtol=0)
