"""Stationary UCCSD-F12b W_vo and W_vv pullback to shared molecular Gaussian sources."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from generativeqc._f12_spin_pairs import occupied_pair_blocks
from generativeqc_compiler.cc.uccsd_f12_sources import unrestricted_f12_pair_sources
from generativeqc_compiler.mp2.f12_density_fitting import (
    coulomb_fit_response,
    molecular_orbital_sources,
)
from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_cross_frame_overlap
from generativeqc_compiler.tensor import Program, transpose_program

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


_OPERATOR_FIELDS = {
    "geminal": "f12",
    "geminal_coulomb": "f12_over_coulomb",
}


class _Frame(NamedTuple):
    projection: Program
    inputs: dict[str, np.ndarray]
    sources: Mapping[str, np.ndarray]
    seeds: dict[str, np.ndarray]
    fit_inputs: dict[str, np.ndarray]
    fit: np.ndarray
    fit_bar: np.ndarray


def _pair_seed(
    weights: Mapping[str, np.ndarray],
    nocc: Mapping[str, int],
    frozen: Mapping[str, int],
    obs: int,
    left: str,
    right: str,
    i: int,
    j: int,
) -> np.ndarray:
    """Transpose the occupied exchange and external spin permutations of the W feeds."""
    seed = np.zeros((obs, obs))
    l, r = left[0], right[0]
    ii, jj = i - frozen[l], j - frozen[r]
    for p, q in ((l, r),) if l == r else ((l, r), (r, l)):
        tile = np.zeros_like(seed)
        for label, columns in (
            ("vo", slice(frozen[q], nocc[q])),
            ("vv", slice(nocc[q], obs)),
        ):
            dest = tile[nocc[p] :, columns]
            for a, b, x, y, sign in ((l, r, ii, jj, 1), (r, l, jj, ii, -1)):
                name = f"w_{label}_{a}{b}{p}{q}"
                if name in weights:
                    dest += sign * weights[name][x, y]
        seed += tile if (p, q) == (l, r) else -tile.T
    return seed


def unrestricted_cc_w_ao_weights(
    space: tuple[int, int, int, int, int],
    ao: Mapping[str, np.ndarray],
    weights: Mapping[str, np.ndarray],
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
) -> dict[str, np.ndarray]:
    """Map external W energy cotangents to the three operators, cross overlap and both spin frames.

    Coefficient columns span the full occupied/virtual/CABS frame. W_vo and W_vv
    sample the same cusp-dressed OBS projection. Only supplied spin-block seeds
    contribute. Metric weights use the full symmetric Frobenius convention.
    """
    oa, ob, obs, ext, aux = space
    occ = {"a": oa, "b": ob}
    frozen = dict(zip(("a", "b"), frozen_occupied, strict=True))
    if not (
        0 <= frozen["a"] <= oa < obs < ext and 0 <= frozen["b"] <= ob < obs and aux > 0
    ):
        raise ValueError("Invalid W occupied, orbital or complementary space")
    allowed = {}
    for l, r, p, q in product("ab", repeat=4):
        if sorted((l, r)) != sorted((p, q)):
            continue
        prefix = (occ[l] - frozen[l], occ[r] - frozen[r], obs - occ[p])
        allowed["w_vo_" + l + r + p + q] = (*prefix, occ[q] - frozen[q])
        allowed["w_vv_" + l + r + p + q] = (*prefix, obs - occ[q])
    if not weights or any(
        name not in allowed or value.shape != allowed[name] or value.dtype != np.float64
        for name, value in weights.items()
    ):
        raise ValueError("Invalid stationary W spin-block cotangents")
    n = ao["overlap_ao"].shape[0]
    whitener = ao["metric_whitener"]
    if not np.allclose(
        whitener @ ao["metric/coulomb"] @ whitener.T, np.eye(aux), atol=2e-7, rtol=0
    ):
        raise ValueError(
            "A full-rank whitener for the same Coulomb fitting metric is required"
        )
    ao_names = ("ao_coulomb", "ao_f12", "ao_f12_over_coulomb")
    metrics = ("metric/coulomb", "metric/geminal", "metric/geminal_coulomb")
    frame_names = ("coefficients_alpha", "coefficients_beta", "overlap_ao")
    bars = {
        name: np.zeros_like(ao[name]) for name in (*frame_names, *ao_names, *metrics)
    }

    cross = unrestricted_cross_frame_overlap(n, ext)
    cross_inputs = {name: ao[name] for name in frame_names}
    overlap = run(cross, cross_inputs).outputs["cross_overlap"]
    cross_bar = np.zeros_like(overlap)
    solve = coulomb_fit_response(aux, obs * ext)
    fit_forward = Program({"fit": solve.outputs["fit"]})
    fit_reverse = Program(
        {k: solve.outputs[k] for k in ("three_center_bar", "metric_bar")}
    )
    frames = {}
    for spin in ("alpha", "beta"):
        occupied = occ[spin[0]]
        if occupied <= frozen[spin[0]]:
            continue
        frame = molecular_orbital_sources(n, ext, occupied, aux)
        graph = Program(
            {
                key: frame.outputs[key]
                for key in ("raw_coulomb", "raw_f12", "raw_f12_over_coulomb")
            }
        )
        inputs = {name: ao[name] for name in ao_names}
        inputs["coefficients"] = ao["coefficients_" + spin]
        sources = run(graph, inputs).outputs
        rhs = np.ascontiguousarray(sources["raw_coulomb"][:, :obs]).reshape(aux, -1)
        fit_fields = {"metric_whitener": whitener, "three_center": rhs}
        fit = run(fit_forward, fit_fields).outputs["fit"].reshape(aux, obs, ext)
        seeds = {k: np.zeros_like(v) for k, v in sources.items()}
        frames[spin] = _Frame(
            graph, inputs, sources, seeds, fit_fields, fit, np.zeros_like(fit)
        )

    for left, right, nl, nr, _, pairs in occupied_pair_blocks(
        space, frozen_occupied=frozen_occupied
    ):
        graph = unrestricted_f12_pair_sources(
            (nl, obs - nl, ext - obs),
            (nr, obs - nr, ext - obs),
            aux,
            channel=f"{left}_{right}",
        )
        reverse = transpose_program(
            Program({"w_obs": graph.outputs["w_obs"]}), ("w_obs",)
        ).program
        sl, sr = frames[left], frames[right]
        for i, j in pairs:
            fields = {
                "fit_obs_left": sl.fit[:, :, :obs],
                "raw_coulomb_obs_right": sr.sources["raw_coulomb"][:, :obs, :obs],
                "fit_left": sl.fit[:, i],
                "fit_right": sr.fit[:, j],
            }
            if left != right:
                fields["cross_overlap"] = overlap
            for op, raw in _OPERATOR_FIELDS.items():
                fields[f"raw_{op}_left"] = sl.sources["raw_" + raw][:, i]
                fields[f"raw_{op}_right"] = sr.sources["raw_" + raw][:, j]
                fields["metric_" + op] = ao["metric/" + op]
            fields["bar_w_obs"] = _pair_seed(
                weights, occ, frozen, obs, left, right, i, j
            )
            result = run(
                reverse, {k: np.ascontiguousarray(v) for k, v in fields.items()}
            ).outputs
            sl.fit_bar[:, :, :obs] += result["bar_fit_obs_left"]
            sr.seeds["raw_coulomb"][:, :obs, :obs] += result[
                "bar_raw_coulomb_obs_right"
            ]
            sl.fit_bar[:, i] += result["bar_fit_left"]
            sr.fit_bar[:, j] += result["bar_fit_right"]
            if left != right:
                cross_bar += result["bar_cross_overlap"]
            for op, raw in _OPERATOR_FIELDS.items():
                sl.seeds["raw_" + raw][:, i] += result[f"bar_raw_{op}_left"]
                sr.seeds["raw_" + raw][:, j] += result[f"bar_raw_{op}_right"]
                bars["metric/" + op] += result["bar_metric_" + op]

    for spin, (graph, inputs, _, seeds, fit_fields, _, fit_bar) in frames.items():
        fit_response = run(
            fit_reverse, {**fit_fields, "fit_bar": fit_bar.reshape(aux, -1)}
        ).outputs
        bars["metric/coulomb"] += fit_response["metric_bar"]
        seeds["raw_coulomb"][:, :obs] += fit_response["three_center_bar"].reshape(
            aux, obs, ext
        )
        reverse = transpose_program(graph, tuple(graph.outputs)).program
        response = run(
            reverse, {**inputs, **{"bar_" + k: x for k, x in seeds.items()}}
        ).outputs
        bars["coefficients_" + spin] += response["bar_coefficients"]
        for key in ao_names:
            bars[key] += response["bar_" + key]

    reverse = transpose_program(cross, ("cross_overlap",)).program
    response = run(reverse, {**cross_inputs, "bar_cross_overlap": cross_bar}).outputs
    for key in frame_names:
        bars[key] += response["bar_" + key]
    return bars
