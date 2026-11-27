"""Stationary UCCSD-F12b W_vo and W_vv pullback to shared molecular Gaussian sources."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from generativeqc_compiler.cc.uccsd_f12_sources import (
    batched_obs_w_projection,
    unrestricted_f12_pair_geminals,
)
from generativeqc_compiler.mp2.f12_density_fitting import coulomb_fit_response
from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_cross_frame_overlap
from generativeqc_compiler.tensor import Program, transpose_program

from generativeqc._f12_spin_ao import (
    project_ao_operator_pullback_tiled,
    project_ao_operator_tiled,
)
from generativeqc._f12_spin_fock import _tile_groups
from generativeqc._f12_spin_pairs import batch_execute, occupied_pair_blocks

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


_OPERATOR_FIELDS = {
    "geminal": "f12",
    "geminal_coulomb": "f12_over_coulomb",
}


class _Frame(NamedTuple):
    occupied: int
    coefficients: np.ndarray
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


def _validate_w_weights(
    space: tuple[int, int, int, int, int],
    weights: Mapping[str, np.ndarray],
    frozen_occupied: tuple[int, int],
) -> None:
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


def _same_spin_w_weights(weights: Mapping[str, np.ndarray]) -> bool:
    for label in ("w_vo_", "w_vv_"):
        ka, kb = label + "aaaa", label + "bbbb"
        if (ka in weights) != (kb in weights):
            return False
        if ka in weights and not np.array_equal(weights[ka], weights[kb]):
            return False
    return True


def _w_batch_dims(n_pairs: int, aux: int, obs: int) -> tuple[int, int]:
    """Choose (pair_batch, aux_tile) keeping intermediate [k, p, obs, obs] <= 192 MiB."""
    target_bytes = 192 << 20
    pair_batch = min(n_pairs, 32)
    slice_bytes = max(1, pair_batch * obs * obs * 8)
    aux_tile = max(1, min(aux, target_bytes // slice_bytes))
    if aux_tile == 1 and pair_batch > 1:
        pair_batch = max(1, min(n_pairs, target_bytes // max(1, obs * obs * 8)))
    return pair_batch, aux_tile


def _accumulate_w_mo_pullback(
    space: tuple[int, int, int, int, int],
    data: Mapping[str, np.ndarray],
    fit_obs: Mapping[str, np.ndarray],
    weights: Mapping[str, np.ndarray],
    run: Callable,
    dbar: dict[str, np.ndarray],
    fit_obs_bar: dict[str, np.ndarray],
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    native_library=None,
    host_reference: bool = True,
    budget_bytes: int = 0,
) -> None:
    """Accumulate MO-space W_vo/W_vv cotangents via GPU-resident cuBLAS or TensorIR fallback.

    A bound CUDA kernel requires the endpoint device budget in `budget_bytes`.
    """
    from ._f12_cc_fixed_sources_native import (
        REQUIRE_AOT_CCFIXED_ENV,
        require_aot_ccfixed,
    )
    from ._f12_cc_pair_pullback_native import (
        bind_pair_pullback,
        checked_device_budget,
        run_w_pullback_native,
    )

    _validate_w_weights(space, weights, frozen_occupied)
    oa, ob, obs, ext, aux = space
    occ = {"a": oa, "b": ob}
    frozen = dict(zip(("a", "b"), frozen_occupied, strict=True))
    native_fn = bind_pair_pullback(native_library, "w", host_reference)
    if native_fn is not None:
        budget_bytes = checked_device_budget(
            budget_bytes, host_reference=host_reference
        )
    spin_sym_aa_bb = (
        oa == ob
        and frozen["a"] == frozen["b"]
        and oa > frozen["a"]
        and _same_spin_w_weights(weights)
        and np.array_equal(fit_obs["alpha"], fit_obs["beta"])
        and np.array_equal(
            data["source/raw_coulomb_alpha"][:, :obs, :obs],
            data["source/raw_coulomb_beta"][:, :obs, :obs],
        )
        and np.array_equal(data["raw/geminal/alpha"], data["raw/geminal/beta"])
        and np.array_equal(
            data["raw/geminal_coulomb/alpha"], data["raw/geminal_coulomb/beta"]
        )
    )

    for left, right, nl, nr, _, pairs in occupied_pair_blocks(
        space, frozen_occupied=frozen_occupied
    ):
        if spin_sym_aa_bb and left == "beta" and right == "beta":
            continue
        pair_list = list(pairs)
        if not pair_list:
            continue
        mirror_bb = spin_sym_aa_bb and left == "alpha" and right == "alpha"
        n_pairs = len(pair_list)

        if native_fn is not None:
            bar_w_all = np.empty((n_pairs, obs, obs), dtype=np.float64)
            for idx, (i, j) in enumerate(pair_list):
                bar_w_all[idx] = _pair_seed(weights, occ, frozen, obs, left, right, i, j)
            res = run_w_pullback_native(
                native_fn,
                same_spin=(left == right),
                nl=nl,
                nr=nr,
                obs=obs,
                ext=ext,
                aux=aux,
                pairs=pair_list,
                fit_obs_l=fit_obs[left][:, :, :obs],
                raw_obs_r=data["source/raw_coulomb_" + right][:, :obs, :obs],
                fit_l=fit_obs[left][:, :nl, :],
                fit_r=fit_obs[right][:, :nr, :],
                raw_g_l=data[f"raw/geminal/{left}"],
                raw_g_r=data[f"raw/geminal/{right}"],
                raw_gc_l=data[f"raw/geminal_coulomb/{left}"],
                raw_gc_r=data[f"raw/geminal_coulomb/{right}"],
                metric_g=data["metric/geminal"],
                metric_gc=data["metric/geminal_coulomb"],
                cross_overlap=None if left == right else data["cross_overlap"],
                bar_w_all=bar_w_all,
                budget_bytes=budget_bytes,
            )
            fit_obs_bar[left][:, :, :obs] += res["bar_fit_obs_l"]
            dbar["source/raw_coulomb_" + right][:, :obs, :obs] += res["bar_raw_obs_r"]
            fit_obs_bar[left][:, :nl, :] += res["bar_fit_l"]
            dbar[f"raw/geminal/{left}"] += res["bar_raw_g_l"]
            dbar[f"raw/geminal_coulomb/{left}"] += res["bar_raw_gc_l"]
            dbar["metric/geminal"] += res["bar_metric_g"]
            dbar["metric/geminal_coulomb"] += res["bar_metric_gc"]
            if left != right:
                fit_obs_bar[right][:, :nr, :] += res["bar_fit_r"]
                dbar[f"raw/geminal/{right}"] += res["bar_raw_g_r"]
                dbar[f"raw/geminal_coulomb/{right}"] += res["bar_raw_gc_r"]
                dbar["cross_overlap"] += res["bar_cross_overlap"]
            elif mirror_bb:
                fit_obs_bar["beta"][:, :, :obs] += res["bar_fit_obs_l"]
                dbar["source/raw_coulomb_beta"][:, :obs, :obs] += res["bar_raw_obs_r"]
                fit_obs_bar["beta"][:, :nl, :] += res["bar_fit_l"]
                dbar["raw/geminal/beta"] += res["bar_raw_g_l"]
                dbar["raw/geminal_coulomb/beta"] += res["bar_raw_gc_l"]
                dbar["metric/geminal"] += res["bar_metric_g"]
                dbar["metric/geminal_coulomb"] += res["bar_metric_gc"]
            continue
        if require_aot_ccfixed():
            raise RuntimeError(
                f"{REQUIRE_AOT_CCFIXED_ENV}=1 forbids TensorIR fallback in "
                "_accumulate_w_mo_pullback (native_library missing or symbol unavailable)"
            )

        # Stage 1: tiny per-pair geminal_obs forward (zero [aux, obs, obs] transfers!).
        base_graph = unrestricted_f12_pair_geminals(
            (nl, obs - nl, ext - obs),
            (nr, obs - nr, ext - obs),
            aux,
            channel=f"{left}_{right}",
        )
        gem_fwd = Program({"geminal_obs": base_graph.outputs["geminal_obs"]})
        gem_rev = transpose_program(
            Program(
                {
                    "geminal_obs": base_graph.outputs["geminal_obs"],
                    "geminal_coulomb_obs": base_graph.outputs["geminal_coulomb_obs"],
                }
            ),
            ("geminal_obs", "geminal_coulomb_obs"),
            max_elements=max(1_000_000, ext * ext),
        ).program

        shared_gem = {
            "metric_geminal": np.ascontiguousarray(data["metric/geminal"]),
        }
        shared_rev = {
            "metric_geminal": shared_gem["metric_geminal"],
            "metric_geminal_coulomb": np.ascontiguousarray(
                data["metric/geminal_coulomb"]
            ),
        }
        if left != right:
            co = np.ascontiguousarray(data["cross_overlap"])
            shared_gem["cross_overlap"] = co
            shared_rev["cross_overlap"] = co

        f_obs_all = np.empty((n_pairs, obs, obs), dtype=np.float64)
        bar_w_all = np.empty((n_pairs, obs, obs), dtype=np.float64)
        with batch_execute(run, gem_fwd) as exec_fwd:
            for idx, (i, j) in enumerate(pair_list):
                fields = dict(shared_gem)
                fields["fit_left"] = np.ascontiguousarray(fit_obs[left][:, i])
                fields["fit_right"] = np.ascontiguousarray(fit_obs[right][:, j])
                fields["raw_geminal_left"] = np.ascontiguousarray(
                    data[f"raw/geminal/{left}"][:, i]
                )
                fields["raw_geminal_right"] = np.ascontiguousarray(
                    data[f"raw/geminal/{right}"][:, j]
                )
                f_obs_all[idx] = exec_fwd(fields).outputs["geminal_obs"]
                bar_w_all[idx] = _pair_seed(weights, occ, frozen, obs, left, right, i, j)

        # Stage 2: auxiliary-tiled batched OBS Coulomb projection reverse across ALL pairs.
        bar_f_obs_all = np.zeros((n_pairs, obs, obs), dtype=np.float64)
        fit_obs_l = fit_obs[left][:, :, :obs]
        raw_obs_r = data["source/raw_coulomb_" + right][:, :obs, :obs]
        pair_batch, aux_tile = _w_batch_dims(n_pairs, aux, obs)
        k_groups = _tile_groups(n_pairs, pair_batch)
        p_groups = _tile_groups(aux, aux_tile)
        neg_bar_w_all = -bar_w_all
        for k_len, k_slices in k_groups:
            for p_len, p_slices in p_groups:
                proj_prog = batched_obs_w_projection(obs, p_len, k_len)
                proj_rev = transpose_program(
                    proj_prog,
                    ("w_proj_batch",),
                    max_elements=max(1_000_000, ext * ext),
                ).program
                with batch_execute(run, proj_rev) as exec_proj:
                    for ks in k_slices:
                        f_b = np.ascontiguousarray(f_obs_all[ks])
                        bw_b = np.ascontiguousarray(neg_bar_w_all[ks])
                        for ps in p_slices:
                            out = exec_proj(
                                {
                                    "geminal_obs_batch": f_b,
                                    "fit_obs_left_tile": np.ascontiguousarray(
                                        fit_obs_l[ps]
                                    ),
                                    "raw_coulomb_obs_right_tile": np.ascontiguousarray(
                                        raw_obs_r[ps]
                                    ),
                                    "bar_w_proj_batch": bw_b,
                                }
                            ).outputs
                            bar_f_obs_all[ks] += out["bar_geminal_obs_batch"]
                            fit_obs_bar[left][ps, :, :obs] += out[
                                "bar_fit_obs_left_tile"
                            ]
                            dbar["source/raw_coulomb_" + right][
                                ps, :obs, :obs
                            ] += out["bar_raw_coulomb_obs_right_tile"]
                            if mirror_bb:
                                fit_obs_bar["beta"][ps, :, :obs] += out[
                                    "bar_fit_obs_left_tile"
                                ]
                                dbar["source/raw_coulomb_beta"][
                                    ps, :obs, :obs
                                ] += out["bar_raw_coulomb_obs_right_tile"]
        del f_obs_all, neg_bar_w_all

        # Stage 3: tiny per-pair reverse back to fit_{i,j} and raw_{geminal,geminal_coulomb}_{i,j}.
        with batch_execute(run, gem_rev) as exec_rev:
            for idx, (i, j) in enumerate(pair_list):
                fields = dict(shared_rev)
                fields["fit_left"] = np.ascontiguousarray(fit_obs[left][:, i])
                fields["fit_right"] = np.ascontiguousarray(fit_obs[right][:, j])
                for op in _OPERATOR_FIELDS:
                    fields[f"raw_{op}_left"] = np.ascontiguousarray(
                        data[f"raw/{op}/{left}"][:, i]
                    )
                    fields[f"raw_{op}_right"] = np.ascontiguousarray(
                        data[f"raw/{op}/{right}"][:, j]
                    )
                fields["bar_geminal_obs"] = np.ascontiguousarray(bar_f_obs_all[idx])
                fields["bar_geminal_coulomb_obs"] = np.ascontiguousarray(bar_w_all[idx])
                result = exec_rev(fields).outputs
                fit_obs_bar[left][:, i] += result["bar_fit_left"]
                fit_obs_bar[right][:, j] += result["bar_fit_right"]
                if left != right:
                    dbar["cross_overlap"] += result["bar_cross_overlap"]
                for op in _OPERATOR_FIELDS:
                    dbar[f"raw/{op}/{left}"][:, i] += result[f"bar_raw_{op}_left"]
                    dbar[f"raw/{op}/{right}"][:, j] += result[f"bar_raw_{op}_right"]
                    dbar["metric/" + op] += result["bar_metric_" + op]
                if mirror_bb:
                    fit_obs_bar["beta"][:, i] += result["bar_fit_left"]
                    fit_obs_bar["beta"][:, j] += result["bar_fit_right"]
                    for op in _OPERATOR_FIELDS:
                        dbar[f"raw/{op}/beta"][:, i] += result[f"bar_raw_{op}_left"]
                        dbar[f"raw/{op}/beta"][:, j] += result[f"bar_raw_{op}_right"]
                        dbar["metric/" + op] += result["bar_metric_" + op]


def unrestricted_cc_w_ao_weights(
    space: tuple[int, int, int, int, int],
    ao: Mapping[str, np.ndarray],
    weights: Mapping[str, np.ndarray],
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
) -> dict[str, np.ndarray]:
    """Map external W energy cotangents to the three operators, cross overlap and both spin frames."""
    _validate_w_weights(space, weights, frozen_occupied)
    oa, ob, obs, ext, aux = space
    occ = {"a": oa, "b": ob}
    frozen = dict(zip(("a", "b"), frozen_occupied, strict=True))
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
    solve = coulomb_fit_response(aux, obs * ext)
    fit_forward = Program({"fit": solve.outputs["fit"]})
    fit_reverse = Program(
        {k: solve.outputs[k] for k in ("three_center_bar", "metric_bar")}
    )
    same_spin = oa == ob and np.array_equal(
        ao["coefficients_alpha"], ao["coefficients_beta"]
    )

    data: dict[str, np.ndarray] = {
        "cross_overlap": overlap,
        "metric/geminal": ao["metric/geminal"],
        "metric/geminal_coulomb": ao["metric/geminal_coulomb"],
    }
    dbar: dict[str, np.ndarray] = {
        "cross_overlap": np.zeros_like(overlap),
        "metric/geminal": np.zeros_like(ao["metric/geminal"]),
        "metric/geminal_coulomb": np.zeros_like(ao["metric/geminal_coulomb"]),
    }
    fit_obs: dict[str, np.ndarray] = {}
    fit_obs_bar: dict[str, np.ndarray] = {}
    fit_inputs: dict[str, dict[str, np.ndarray]] = {}
    active_spins = [s for s in ("alpha", "beta") if occ[s[0]] > frozen[s[0]]]
    for spin in active_spins:
        occupied = occ[spin[0]]
        coeffs = ao["coefficients_" + spin]
        if same_spin and spin == "beta" and "alpha" in fit_obs:
            data["source/raw_coulomb_beta"] = data["source/raw_coulomb_alpha"]
            data["raw/geminal/beta"] = data["raw/geminal/alpha"]
            data["raw/geminal_coulomb/beta"] = data["raw/geminal_coulomb/alpha"]
            fit_obs["beta"] = fit_obs["alpha"]
            fit_inputs["beta"] = fit_inputs["alpha"]
        else:
            raw_c = project_ao_operator_tiled(
                n, ext, obs, aux, "raw_f12", "ao_f12", coeffs, ao["ao_coulomb"], run
            )
            data["source/raw_coulomb_" + spin] = raw_c
            data["raw/geminal/" + spin] = project_ao_operator_tiled(
                n, ext, occupied, aux, "raw_f12", "ao_f12", coeffs, ao["ao_f12"], run
            )
            data["raw/geminal_coulomb/" + spin] = project_ao_operator_tiled(
                n,
                ext,
                occupied,
                aux,
                "raw_f12_over_coulomb",
                "ao_f12_over_coulomb",
                coeffs,
                ao["ao_f12_over_coulomb"],
                run,
            )
            rhs = np.ascontiguousarray(raw_c).reshape(aux, -1)
            f_in = {"metric_whitener": whitener, "three_center": rhs}
            fit_inputs[spin] = f_in
            fit_obs[spin] = run(fit_forward, f_in).outputs["fit"].reshape(aux, obs, ext)
        dbar["source/raw_coulomb_" + spin] = np.zeros_like(
            data["source/raw_coulomb_" + spin]
        )
        dbar["raw/geminal/" + spin] = np.zeros_like(data["raw/geminal/" + spin])
        dbar["raw/geminal_coulomb/" + spin] = np.zeros_like(
            data["raw/geminal_coulomb/" + spin]
        )
        fit_obs_bar[spin] = np.zeros_like(fit_obs[spin])

    _accumulate_w_mo_pullback(
        space,
        data,
        fit_obs,
        weights,
        run,
        dbar,
        fit_obs_bar,
        frozen_occupied=frozen_occupied,
    )

    bars["metric/geminal"] += dbar["metric/geminal"]
    bars["metric/geminal_coulomb"] += dbar["metric/geminal_coulomb"]
    for spin in active_spins:
        occupied = occ[spin[0]]
        coeffs = ao["coefficients_" + spin]
        fit_response = run(
            fit_reverse,
            {**fit_inputs[spin], "fit_bar": fit_obs_bar[spin].reshape(aux, -1)},
        ).outputs
        bars["metric/coulomb"] += fit_response["metric_bar"]
        coul_seed = dbar["source/raw_coulomb_" + spin] + fit_response[
            "three_center_bar"
        ].reshape(aux, obs, ext)
        project_ao_operator_pullback_tiled(
            n,
            ext,
            obs,
            aux,
            "raw_f12",
            "ao_f12",
            coeffs,
            ao["ao_coulomb"],
            coul_seed,
            run,
            bar_coefficients=bars["coefficients_" + spin],
            bar_ao=bars["ao_coulomb"],
        )
        for op_key, field, ao_name in (
            ("geminal", "raw_f12", "ao_f12"),
            ("geminal_coulomb", "raw_f12_over_coulomb", "ao_f12_over_coulomb"),
        ):
            project_ao_operator_pullback_tiled(
                n,
                ext,
                occupied,
                aux,
                field,
                ao_name,
                coeffs,
                ao[ao_name],
                dbar[f"raw/{op_key}/{spin}"],
                run,
                bar_coefficients=bars["coefficients_" + spin],
                bar_ao=bars[ao_name],
            )

    reverse = transpose_program(cross, ("cross_overlap",)).program
    response = run(
        reverse, {**cross_inputs, "bar_cross_overlap": dbar["cross_overlap"]}
    ).outputs
    for key in frame_names:
        bars[key] += response["bar_" + key]
    return bars
