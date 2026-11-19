"""Compare ordered CC-F12 tensor programs and all cotangents to dense independent contractions."""

import numpy as np
import pytest
from generativeqc_compiler.cc.f12_residual import (
    f12b_geminal_back_coupling,
    fixed_pair_cc_residual,
    fixed_pair_geminal_energy,
    standard_v_tile,
)
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.tensor.ad_program import transpose_program

from benchmarks.f12_pair_component import FIXTURE


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


@pytest.mark.parametrize("m,n", [(0, 1), (2, 2)])
def test_ordered_cc_geminal_energies_and_all_source_gradients(m: int, n: int) -> None:

    no, nv = 3, 4
    rng = np.random.default_rng(73)
    graph = fixed_pair_geminal_energy(no, nv)
    fields = {
        q.attrs["name"]: rng.normal(size=q.spec.shape)
        for q in graph.live_nodes
        if q.op == "input"
    }
    fields.update(select_m=np.eye(no)[m], select_n=np.eye(no)[n])

    def ref(s: dict) -> dict:
        fixed = np.zeros((no, no, no, no))
        for i, j, k, l in np.ndindex(fixed.shape):
            p, ex = int((i, j) == (k, l)), int((i, j) == (l, k))
            fixed[i, j, k, l] = 0.5 * (p + ex) / 2 + 0.25 * (p - ex) / 2
        va = 0.0
        vb = 0.0
        for i, j in np.ndindex(no, no):
            weight = 2 * fixed[i, j, m, n] - fixed[i, j, n, m]
            rhs = 2 * s["v_occupied"][i, j] + np.vdot(
                s["fock_coupling_vv"], s["t2"][i, j]
            )
            for p, q in np.ndindex(no, no):
                rhs += s["b_row"][p, q] * fixed[i, j, p, q]
                for k in range(no):
                    rhs -= s["x_row"][p, q] * (
                        s["fock_occupied"][i, k] * fixed[k, j, p, q]
                        + s["fock_occupied"][j, k] * fixed[i, k, p, q]
                    )
            va += weight * rhs
            vb += weight * s["back_coupling"][i, j]
        return {"f12a_geminal_energy": va, "f12b_extra_energy": vb}

    expected = ref(fields)
    result = execute(graph, fields).outputs
    for k, y in expected.items():
        assert result[k] == pytest.approx(y, abs=2e-13)
    seeds = {k: np.array(rng.normal()) for k in result}
    reverse = transpose_program(graph, tuple(result)).program
    bars = execute(
        reverse, {**fields, **{"bar_" + key: v for key, v in seeds.items()}}
    ).outputs
    for name, arr in fields.items():
        if name.startswith("select_"):
            continue
        tangent = rng.normal(size=arr.shape)
        expected_slope = np.vdot(bars["bar_" + name], tangent)
        for h in (4e-4, 2e-4):
            values = [
                ref({**fields, name: arr + sign * h * tangent}) for sign in (-1, 1)
            ]
            slope = sum(
                seeds[key] * (values[1][key] - values[0][key]) for key in result
            ) / (2 * h)
            assert slope == pytest.approx(expected_slope, abs=3e-11)


@pytest.mark.parametrize("case", ["he", "h2", "h2_df", "water_df"])
def test_cc_common_geminal_energy_recovers_independent_stationary_mp2_pairs(
    case: str,
) -> None:
    """At first-order doubles the ordinary plus F12a functional recovers Psi4 3C(FIX)."""

    total = 0.0
    with np.load(FIXTURE, allow_pickle=False) as f:
        for group in ("diagonal", "offdiag"):
            name = case + "/" + group + "/"
            if name + "psi4_pairs" not in f:
                continue
            diagonal = group == "diagonal"
            expected = f[name + "psi4_pairs"]
            src = {
                k.removeprefix(name + "in/"): f[k]
                for k in f
                if k.startswith(name + "in/")
            }
            for p, psi_value in enumerate(expected):
                row = {k: v[p] for k, v in src.items()}
                corr = row["c_direct"]
                coulomb = row["g"]
                denom = row["inverse_gap"]
                exchange = row.get("c_exchange", corr)
                fixed_amplitude_term = (
                    corr / 2 if diagonal else (3 * corr + exchange) / 8
                )
                mp2 = -denom * coulomb
                t2 = -denom * (coulomb + fixed_amplitude_term)
                change = t2 - mp2
                # Ordinary CC correlation at the shifted amplitudes, with the bare MP2 removed.
                conventional_change = np.vdot(2 * change - change.T, coulomb)
                n = 0 if diagonal else 1
                no = n + 1
                nv = corr.shape[0]
                graph = fixed_pair_geminal_energy(no, nv)
                ans = (1 if diagonal else 2) * conventional_change
                t2_full = np.zeros((no, no, nv, nv))
                t2_full[0, n] = t2
                if not diagonal:
                    t2_full[n, 0] = t2.T
                # At a canonical reference only these two generator-pair matrix elements enter.
                for i, j, c in (
                    [(0, n, corr)] if diagonal else [(0, n, corr), (n, 0, exchange)]
                ):
                    feeds = {
                        "select_m": np.eye(no)[i],
                        "select_n": np.eye(no)[j],
                        "t2": t2_full,
                        "fock_coupling_vv": c,
                        "back_coupling": np.zeros((no, no)),
                        "fock_occupied": np.eye(no) * row["occupied_fock_sum"] / 2,
                    }
                    for label, target in [
                        ("v", "v_occupied"),
                        ("b", "b_row"),
                        ("x", "x_row"),
                    ]:
                        mat = np.zeros((no, no))
                        mat[i, j] = row[label + "_direct"]
                        if not diagonal:
                            mat[j, i] = row[label + "_exchange"]
                        feeds[target] = mat
                    ans += float(execute(graph, feeds).outputs["f12a_geminal_energy"])
                assert ans == pytest.approx(psi_value, abs=1e-12)
                total += ans
        assert total == pytest.approx(float(f[case + "/energy"]), abs=2e-12)


@pytest.mark.parametrize("m,n", [(1, 1), (0, 1)])
def test_robust_df_external_geminal_source_and_metric_response(m: int, n: int) -> None:
    from generativeqc_compiler.cc.f12_residual import df_standard_v

    rng = np.random.default_rng(104)
    obs = 4
    aux = 6
    graph = df_standard_v(obs, aux, diagonal=m == n)
    fields = {
        q.attrs["name"]: rng.normal(size=q.spec.shape)
        for q in graph.live_nodes
        if q.op == "input"
    }
    fields["select_m"] = np.eye(obs)[m]
    if m != n:
        fields["select_n"] = np.eye(obs)[n]

    def reference(s: dict) -> np.ndarray:
        fit = s["fit_all"]
        direct = {}
        for name in ("f12", "f12_over_coulomb"):
            a = s["raw_" + name + "_m"]
            b = s["raw_" + name + ("_m" if m == n else "_n")]
            x, y = fit[:, :, m], fit[:, :, n]
            direct[name] = x.T @ b + a.T @ y - x.T @ s["metric_" + name] @ y
        coulomb = np.einsum("pur,pvs->urvs", fit, s["raw_coulomb_all"])
        return direct["f12_over_coulomb"] - np.einsum(
            "urvs,rs->uv", coulomb, direct["f12"]
        )

    np.testing.assert_allclose(
        execute(graph, fields).outputs["v"], reference(fields), rtol=1e-12, atol=1e-12
    )
    seed = rng.normal(size=(obs, obs))
    bars = execute(
        transpose_program(graph, ("v",)).program, {**fields, "bar_v": seed}
    ).outputs
    for key, value in fields.items():
        if key.startswith("select_"):
            continue
        delta = rng.normal(size=value.shape) / np.sqrt(value.size)
        for h in (4e-4, 2e-4):
            slopes = [
                np.vdot(seed, reference({**fields, key: value + sign * h * delta}))
                for sign in [-1, 1]
            ]
            # The fitted density also occurs in the OBS subtraction. Its finite difference has an O(h^2) term.
            assert (
                abs(
                    np.vdot(bars["bar_" + key], delta)
                    - (slopes[1] - slopes[0]) / (2 * h)
                )
                < 2e-6
            )
