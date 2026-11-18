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
    value = float(execute(graph, feeds).outputs["pair_energy"])
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
