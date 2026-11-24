"""Off-diagonal rows of the strong-orthogonality projector on a finite orbital frame."""

import numpy as np
from generativeqc_compiler.mp2.f12_occupied_pair_overlap import (
    all_occupied_vx,
    occupied_pair_vx,
)
from generativeqc_compiler.tensor import execute


def test_distinct_bra_and_ket_use_the_full_strong_orthogonality_projector() -> None:
    o, obs, ri, aux = 3, 5, 8, 6
    rng = np.random.default_rng(581)
    sources = {}
    for key in (
        "fit",
        "raw_coulomb",
        "raw_f12",
        "raw_f12_squared",
        "raw_f12_over_coulomb",
    ):
        sources[key] = rng.normal(size=(aux, o, ri))
    for key in ("f12", "f12_squared", "f12_over_coulomb"):
        a = rng.normal(size=(aux, aux))
        sources["metric_" + key] = a + a.T
    m, n, k, l = 0, 1, 2, 0
    graph = occupied_pair_vx(
        o, obs - o, ri - obs, aux, bra_diagonal=False, ket_diagonal=False
    )
    feed = {}
    for node in graph.live_nodes:
        if node.op != "input":
            continue
        name = node.attrs["name"]
        if name.startswith("select_"):
            feed[name] = np.eye(o)[k if name == "select_k" else l]
        elif name.startswith("metric_"):
            feed[name] = sources[name]
        else:
            side, field = name.split("_", 1)
            field, leg = field.rsplit("_", 1)
            pair = (m, n) if side == "bra" else (k, l)
            feed[name] = sources[field][:, pair[leg == "j"], :]

    def fitted(name: str, i: int, j: int) -> np.ndarray:
        d = sources["fit"]
        raw = sources["raw_" + name]
        if name == "coulomb":
            return d[:, i].T @ raw[:, j]
        return (
            d[:, i].T @ raw[:, j]
            + raw[:, i].T @ d[:, j]
            - d[:, i].T @ sources["metric_" + name] @ d[:, j]
        )

    mask = np.zeros((ri, ri), bool)
    mask[:obs, :obs] = True
    mask[:o, obs:] = True
    mask[obs:, :o] = True
    right = fitted("f12", k, l)
    out = execute(graph, feed).outputs
    whole = all_occupied_vx(o, obs - o, ri - obs, aux)
    tables = execute(
        whole,
        {
            q.attrs["name"]: sources[q.attrs["name"]]
            for q in whole.live_nodes
            if q.op == "input"
        },
    ).outputs
    for name, left, direct in (
        ("v", "coulomb", "f12_over_coulomb"),
        ("x", "f12", "f12_squared"),
    ):
        expected = fitted(direct, m, n)[k, l] - np.vdot(
            fitted(left, m, n)[mask], right[mask]
        )
        np.testing.assert_allclose(out[name], expected, rtol=1e-12)
        np.testing.assert_allclose(tables[name][m, n, k, l], expected, rtol=1e-12)
