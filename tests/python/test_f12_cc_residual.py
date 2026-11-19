"""Compare ordered CC-F12 tensor programs and all cotangents to dense independent contractions."""

import numpy as np
import pytest
from generativeqc_compiler.cc.f12_residual import (
    f12b_geminal_back_coupling,
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
