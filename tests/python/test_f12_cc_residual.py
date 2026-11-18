"""Compare ordered CC-F12 tensor programs and all cotangents to dense independent contractions."""

import numpy as np
import pytest
from generativeqc_compiler.cc.f12_residual import (
    f12b_geminal_back_coupling,
    fixed_pair_cc_residual,
    standard_v_tile,
)
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.tensor.ad_program import transpose_program


def test_projected_v_tile_agrees_with_dense_pair_space_and_reverse() -> None:
    graph = standard_v_tile(5, 8, 2, 3)
    rng = np.random.default_rng(5511)
    fields = {
        n.attrs["name"]: rng.normal(size=n.spec.shape)
        for n in graph.live_nodes
        if n.op == "input"
    }
    f = fields["geminal_obs"]
    # Assemble the explicit (uv,rs) Coulomb matrix solely in this oracle.
    dense = np.einsum(
        "pur,pvs->uvrs", fields["fit_left"], fields["raw_coulomb_right"]
    ).reshape(6, 25)
    expected = fields["geminal_coulomb"] - (dense @ f.ravel()).reshape(2, 3)
    np.testing.assert_allclose(
        execute(graph, fields).outputs["v"], expected, atol=3e-13, rtol=0
    )
    seed = rng.normal(size=(2, 3))
    reverse = transpose_program(graph, ("v",)).program
    bars = execute(reverse, {**fields, "bar_v": seed}).outputs
    numerical = {
        "geminal_coulomb": seed,
        "geminal_obs": -(dense.T @ seed.ravel()).reshape(5, 5),
        "fit_left": -np.einsum("uv,rs,pvs->pur", seed, f, fields["raw_coulomb_right"]),
        "raw_coulomb_right": -np.einsum("uv,rs,pur->pvs", seed, f, fields["fit_left"]),
    }
    for k, v in numerical.items():
        np.testing.assert_allclose(bars["bar_" + k], v, atol=1e-13, rtol=0)


def test_f12b_extra_residual_and_adjoint_for_asymmetric_occupied_pair() -> None:
    no, nv = 2, 3
    graph = f12b_geminal_back_coupling(no, nv)
    rng = np.random.default_rng(150)
    fields = {
        n.attrs["name"]: rng.normal(size=n.spec.shape)
        for n in graph.live_nodes
        if n.op == "input"
    }

    def oracle(f: dict) -> np.ndarray:
        answer = np.zeros((no, no))
        for i, j in np.ndindex(no, no):
            for a in range(nv):
                answer[i, j] += (
                    f["v_ov"][i, a] * f["t1"][j, a] + f["t1"][i, a] * f["v_vo"][a, j]
                )
                for b in range(nv):
                    answer[i, j] += f["v_vv"][a, b] * (
                        f["t2"][i, j, a, b] + f["t1"][i, a] * f["t1"][j, b]
                    )
        return answer

    expected = oracle(fields)
    np.testing.assert_allclose(
        execute(graph, fields).outputs["back_coupling"], expected, atol=5e-14, rtol=0
    )
    seed = rng.normal(size=(no, no))
    reverse = transpose_program(graph, ("back_coupling",)).program
    bars = execute(reverse, {**fields, "bar_back_coupling": seed}).outputs
    for key, x in fields.items():
        direction = rng.normal(size=x.shape)
        predicted = np.vdot(bars["bar_" + key], direction)
        for h in (4e-4, 2e-4):
            pair = [
                oracle({**fields, key: x + sign * h * direction}) for sign in (-1, 1)
            ]
            assert np.vdot(pair[1] - pair[0], seed) / (2 * h) == pytest.approx(
                predicted, abs=2e-10
            )


@pytest.mark.parametrize("m,n", [(1, 1), (0, 2)])
def test_fixed_geminal_spatial_pair_scatter_and_transpose(m: int, n: int) -> None:
    no, nv = 3, 4
    rng = np.random.default_rng(1834)
    graph = fixed_pair_cc_residual(no, nv, diagonal=m == n)
    v = rng.normal(size=(no + nv, no + nv))
    c = rng.normal(size=(nv, nv))
    if m == n:
        v = (v + v.T) / 2
        c = (c + c.T) / 2
    feeds = {
        "t1": rng.normal(size=(no, nv)),
        "v_vv": v[no:, no:],
        "v_ov": v[:no, no:],
        "v_vo": v[no:, :no],
        "fock_coupling_vv": c,
        "select_m": np.eye(no)[m],
    }
    if m != n:
        feeds["select_n"] = np.eye(no)[n]

    def reference(values: dict) -> dict:
        # Expand the singlet/triplet projectors directly into every i,j ordinary residual.
        r1 = np.zeros((no, nv))
        r2 = np.zeros((no, no, nv, nv))
        t = values["t1"]
        orders = [
            (
                m,
                n,
                values["v_vv"],
                values["v_ov"],
                values["v_vo"],
                values["fock_coupling_vv"],
            )
        ]
        if m != n:
            orders.append(
                (
                    n,
                    m,
                    values["v_vv"].T,
                    values["v_vo"].T,
                    values["v_ov"].T,
                    values["fock_coupling_vv"].T,
                )
            )
        for x, y, vv, ov, vo, cblock in orders:
            tfix = np.zeros((no, no))
            for i, j in np.ndindex(no, no):
                direct = i == x and j == y
                exchange = i == y and j == x
                tfix[i, j] = 0.5 * (
                    0.5 * (direct + exchange) + 0.25 * (direct - exchange)
                )
            d = vv + cblock - vo @ t - t.T @ ov
            for i, j in np.ndindex(no, no):
                r2[i, j] += tfix[i, j] * d
                r1[i] += (2 * tfix[i, j] - tfix[j, i]) * vo[:, j]
        out = {"doubles_mn": r2[m, n], "singles_m": r1[m]}
        if m != n:
            np.testing.assert_allclose(r2[n, m], r2[m, n].T, atol=1e-13, rtol=0)
            out["singles_n"] = r1[n]
        return out

    out = execute(graph, feeds).outputs
    refs = reference(feeds)
    for k, a in refs.items():
        np.testing.assert_allclose(out[k], a, atol=2e-14, rtol=0)
    seeds = {k: rng.normal(size=a.shape) for k, a in refs.items()}
    reverse = transpose_program(graph, tuple(seeds)).program
    bars = execute(
        reverse, {**feeds, **{"bar_" + k: s for k, s in seeds.items()}}
    ).outputs
    for target in ("v_vv", "v_ov", "v_vo", "fock_coupling_vv", "t1"):
        direction = rng.normal(size=feeds[target].shape)
        for h in (4e-4, 2e-4):
            vals = [
                reference({**feeds, target: feeds[target] + sign * h * direction})
                for sign in (-1, 1)
            ]
            fd = sum(np.vdot(seeds[k], vals[1][k] - vals[0][k]) for k in seeds) / (
                2 * h
            )
            assert np.vdot(bars["bar_" + target], direction) == pytest.approx(
                fd, abs=1e-10
            )
    # At zero geminal coupling ordinary CC residuals must be recovered unchanged.
    zero = {
        **feeds,
        **{
            name: np.zeros_like(x)
            for name, x in feeds.items()
            if name.startswith("v_") or name == "fock_coupling_vv"
        },
    }
    for value in execute(graph, zero).outputs.values():
        np.testing.assert_array_equal(value, np.zeros_like(value))
