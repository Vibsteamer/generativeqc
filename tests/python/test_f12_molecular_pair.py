"""A physical geminal increment, an extended-space Fock and their differentiable sources."""

from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.mp2.f12_molecular_pair import (
    extended_df_fock,
    projected_f12_pair,
)
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.tensor.ad_program import transpose_program

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "reference_data/f12/psi4_h2_molecular_pair.npz"
)


def test_h2_from_fitted_operator_sources() -> None:
    with np.load(FIXTURE, allow_pickle=False) as ref:
        occ, obs, ri = map(int, ref["space"])
        graph = projected_f12_pair(
            occ, obs - occ, ri - obs, diagonal=True, auxiliaries=len(ref["fit_i"])
        )
        feeds = {
            n.attrs["name"]: ref[n.attrs["name"]]
            for n in graph.live_nodes
            if n.op == "input"
        }
        answer = execute(graph, feeds).outputs
        for name in ("v", "x", "b", "c", "g", "pair_energy"):
            key = name if name in ("g", "pair_energy") else name + "_direct"
            np.testing.assert_allclose(answer[key], ref[name], atol=5e-12, rtol=0)
        # One geminal source feeds several asymmetric blocks, with both external
        # densities referring to the same occupied orbital for this pair.
        derivative = transpose_program(
            graph, ("pair_energy",), inputs=("raw_f12_i",)
        ).program
        bars = execute(derivative, {**feeds, "bar_pair_energy": np.array(1.0)}).outputs
        direction = np.random.default_rng(198).normal(size=feeds["raw_f12_i"].shape)
        direction /= np.linalg.norm(direction)
        predicted = np.vdot(bars["bar_raw_f12_i"], direction)
        for step in (1e-4, 3e-5):
            values = [
                float(
                    execute(
                        graph,
                        {**feeds, "raw_f12_i": feeds["raw_f12_i"] + s * direction},
                    ).outputs["pair_energy"]
                )
                for s in (step, -step)
            ]
            assert predicted == pytest.approx(
                (values[0] - values[1]) / (2 * step), abs=1e-8
            )


def test_extended_df_fock_and_reverse_on_all_sources() -> None:
    rng = np.random.default_rng(893)
    occ, ri, aux = 2, 6, 7
    p = extended_df_fock(occ, ri, aux)
    fields = {
        "fit_all": rng.normal(size=(aux, ri, ri)),
        "raw_occupied": rng.normal(size=(aux, occ, ri)),
        "hcore": rng.normal(size=(ri, ri)),
    }
    fit, raw, h = (fields[k] for k in ("fit_all", "raw_occupied", "hcore"))
    positive_k = sum(fit[a, :, :occ] @ raw[a] for a in range(aux))
    direct = (
        h + 2 * sum(fit[a] * np.trace(raw[a, :, :occ]) for a in range(aux)) - positive_k
    )
    answer = execute(p, fields).outputs
    np.testing.assert_allclose(answer["fock"], direct, atol=1e-13)
    np.testing.assert_allclose(answer["exchange"], positive_k, atol=1e-13)
    seeds = {k: rng.normal(size=v.shape) for k, v in answer.items()}
    bars = execute(
        transpose_program(p, tuple(seeds)).program,
        {**fields, **{"bar_" + k: v for k, v in seeds.items()}},
    ).outputs
    delta = {k: rng.normal(size=v.shape) for k, v in fields.items()}
    predicted = sum(np.vdot(bars["bar_" + k], v) for k, v in delta.items())

    def objective(t: float) -> float:
        shifted = execute(p, {k: v + t * delta[k] for k, v in fields.items()}).outputs
        return sum(np.vdot(seeds[k], v) for k, v in shifted.items())

    assert predicted == pytest.approx(
        (objective(1e-4) - objective(-1e-4)) / 2e-4, abs=1e-8
    )


@pytest.mark.parametrize("dimensions", [(0, 3, 5), (2, 0, 5), (2, 3, False)])
def test_invalid_orbital_spaces(dimensions: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="Positive"):
        projected_f12_pair(*dimensions, diagonal=False, auxiliaries=4)
