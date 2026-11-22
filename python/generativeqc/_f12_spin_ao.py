"""Unrestricted 3C(FIX) orbital sources and their reverse map to a common AO basis."""

from collections.abc import Callable
from typing import Literal

import numpy as np
from generativeqc_compiler.mp2.f12_density_fitting import (
    coulomb_fit_response,
    molecular_orbital_sources,
)
from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_cross_frame_overlap
from generativeqc_compiler.tensor import Program, transpose_program

from ._f12_spin_fock import _tile_groups, default_auxiliary_batch_size
from ._f12_spin_pairs import batch_execute

_OPERATORS = {
    "hcore": "hcore",
    "coulomb": "raw_coulomb",
    "geminal": "raw_f12",
    "squared_geminal": "raw_f12_squared",
    "geminal_coulomb": "raw_f12_over_coulomb",
    "double_commutator": "raw_double_commutator",
}


def project_ao_operator_tiled(
    nc: int,
    ext: int,
    occ: int,
    aux: int,
    field: str,
    ao_name: str,
    coefficients: np.ndarray,
    ao_mat: np.ndarray,
    run: Callable,
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    """Transform one 3-center AO operator [aux, nc, nc] into the MO frame in bounded auxiliary tiles."""
    width = (
        default_auxiliary_batch_size(aux, max(nc, ext))
        if batch_size is None
        else batch_size
    )
    left_dim = ext if field == "raw_coulomb" else occ
    coeffs = np.ascontiguousarray(coefficients)
    if width >= aux:
        frame = molecular_orbital_sources(nc, ext, occ, aux)
        graph = Program({field: frame.outputs[field]})
        return run(graph, {"coefficients": coeffs, ao_name: np.ascontiguousarray(ao_mat)}).outputs[field]
    out = np.empty((aux, left_dim, ext), dtype=np.float64)
    for tile_len, slices in _tile_groups(aux, width):
        frame = molecular_orbital_sources(nc, ext, occ, tile_len)
        graph = Program({field: frame.outputs[field]})
        with batch_execute(run, graph) as exec_one:
            for rows in slices:
                out[rows] = exec_one(
                    {"coefficients": coeffs, ao_name: np.ascontiguousarray(ao_mat[rows])}
                ).outputs[field]
    return out


def project_ao_operator_pullback_tiled(
    nc: int,
    ext: int,
    occ: int,
    aux: int,
    field: str,
    ao_name: str,
    coefficients: np.ndarray,
    ao_mat: np.ndarray,
    seed: np.ndarray,
    run: Callable,
    *,
    bar_coefficients: np.ndarray,
    bar_ao: np.ndarray | None = None,
    batch_size: int | None = None,
    seed_scale: float = 1.0,
) -> np.ndarray:
    """Pull MO operator cotangents [aux, *, ext] back to AO [aux, nc, nc] and coefficients in bounded tiles."""
    width = (
        default_auxiliary_batch_size(aux, max(nc, ext))
        if batch_size is None
        else batch_size
    )
    coeffs = np.ascontiguousarray(coefficients)
    if bar_ao is None:
        bar_ao = np.zeros_like(ao_mat)
    for tile_len, slices in _tile_groups(aux, width):
        frame = molecular_orbital_sources(nc, ext, occ, tile_len)
        graph = Program({field: frame.outputs[field]})
        rev = transpose_program(graph, (field,)).program
        with batch_execute(run, rev) as exec_one:
            for rows in slices:
                seed_tile = (
                    np.ascontiguousarray(seed[rows])
                    if seed_scale == 1.0
                    else np.ascontiguousarray(seed[rows] * seed_scale)
                )
                weights = exec_one(
                    {
                        "coefficients": coeffs,
                        ao_name: np.ascontiguousarray(ao_mat[rows]),
                        "bar_" + field: seed_tile,
                    }
                ).outputs
                bar_coefficients += weights["bar_coefficients"]
                bar_ao[rows] += weights["bar_" + ao_name]
    return bar_ao


def unrestricted_orbital_sources(
    space: tuple[int, int, int, int, int],
    ao: dict[str, np.ndarray],
    run: Callable,
    *,
    orbital_weights: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Project shared AO hcore, three-center operators, and metrics to both spin frames.

    The second return value transfers an optional electronic source adjoint to
    the independent spin coefficients and the common AO Hamiltonian, Coulomb
    and geminal metrics, overlap, and all integral factors. A single full-rank
    whitener must satisfy W.T @ W = inverse(Coulomb metric). Its physical
    response is taken through the defining metric equation. Only one operator
    tile per spin is transformed at a time; this keeps workspace independent
    of the number of operators.
    """
    return _orbital_sources(space, ao, run, orbital_weights, projected=None)


def _orbital_sources(
    space: tuple[int, int, int, int, int],
    ao: dict[str, np.ndarray],
    run: Callable,
    orbital_weights: dict[str, np.ndarray] | None,
    *,
    projected: dict[str, np.ndarray] | None,
    role: Literal["shared", "pairs", "fock", "ordinary_jk", "ordinary_fock"] = "shared",
    ao_bar_sink: Callable[[str, np.ndarray], None] | None = None,
    occupied_coulomb: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    # A combined energy/response evaluation owns the forward buffers through
    # the reverse pass. Standalone evaluations build their own projections.
    oa, ob, obs, ext, aux = space
    valid_frame = obs == ext if role in ("ordinary_jk", "ordinary_fock") else obs < ext
    if not (0 <= oa < obs and 0 <= ob < obs and valid_frame and oa + ob > 0):
        raise ValueError("Invalid UHF occupied, OBS, or extended dimensions")
    nc = len(ao["coefficients_alpha"])
    occupied = {"alpha": oa, "beta": ob}
    same_spin = oa == ob and np.array_equal(
        ao["coefficients_alpha"], ao["coefficients_beta"]
    )
    ao_loader = ao.get("__ao_loader__")
    needs_cross = role in ("shared", "pairs")
    if needs_cross:
        cross = unrestricted_cross_frame_overlap(nc, ext)
        csrc = {
            k: ao[k] for k in ("overlap_ao", "coefficients_alpha", "coefficients_beta")
        }
    if projected is None:
        data = {"space": np.array(space)}
        if needs_cross:
            data["cross_overlap"] = run(cross, csrc).outputs["cross_overlap"]
    else:
        data = projected
    bars = None
    if orbital_weights is not None:
        skip_keys = {"metric_whitener", "__ao_loader__", "__obs_coulomb__"}
        bars = {
            k: np.zeros_like(v)
            for k, v in ao.items()
            if k not in skip_keys and not (ao_bar_sink is not None and k.startswith("ao_"))
        }
        if needs_cross:
            cross_rev = transpose_program(cross, ("cross_overlap",)).program
            result = run(
                cross_rev,
                {**csrc, "bar_cross_overlap": orbital_weights["cross_overlap"]},
            ).outputs
            for name, value in result.items():
                bars[name.removeprefix("bar_")] += value
    operators = tuple(_OPERATORS)
    if role == "ordinary_jk":
        operators = ("coulomb",)
    elif role in ("fock", "ordinary_fock"):
        operators = ("hcore", "coulomb")
    elif role == "pairs":
        operators = tuple(op for op in operators if op != "hcore")
    for op in operators:
        if op not in ("hcore", "coulomb"):
            name = "metric/" + op
            data[name] = ao[name]
            if bars is not None:
                bars[name] += orbital_weights[name]
    hcore_frames = {
        spin: molecular_orbital_sources(nc, ext, occ, 1)
        for spin, occ in occupied.items()
    }
    # Iterate operators outermost so streaming mode holds at most ONE [Q, ext, ext] AO panel
    # (and at most ONE [Q, ext, ext] cotangent) in host RAM across both spin channels.
    for op in operators:
        field = _OPERATORS[op]
        ao_name = "hcore_ao" if op == "hcore" else "ao_" + field.removeprefix("raw_")
        if not any(field in hcore_frames[spin].outputs for spin in occupied):
            continue
        stream_this = op != "hcore" and ao_loader is not None and ao_name not in ao
        ao_mat = ao_loader(ao_name) if stream_this else ao[ao_name]
        sink_this = bars is not None and op != "hcore" and ao_bar_sink is not None
        bar_ao_mat = np.zeros_like(ao_mat) if sink_this else None
        if bars is not None and op != "hcore" and not sink_this and ao_name not in bars:
            bars[ao_name] = np.zeros_like(ao_mat)
        if op == "hcore":
            for spin, occ in occupied.items():
                if field not in hcore_frames[spin].outputs:
                    continue
                co = "coefficients_" + spin
                key = "source/hcore_" + spin
                graph = Program({field: hcore_frames[spin].outputs[field]})
                item = {"coefficients": ao[co], ao_name: ao_mat}
                if projected is None:
                    data[key] = run(graph, item).outputs[field]
                if bars is not None:
                    rev = transpose_program(graph, (field,)).program
                    weights = run(rev, {**item, "bar_" + field: orbital_weights[key]}).outputs
                    bars[co] += weights["bar_coefficients"]
                    bars[ao_name] += weights["bar_" + ao_name]
                    del weights
            continue

        use_occ_coul = occupied_coulomb and op == "coulomb"
        graph_field = "raw_f12" if use_occ_coul else field
        graph_ao_name = "ao_f12" if use_occ_coul else ao_name

        def _op_key(spin_name: str) -> str:
            return (
                "source/" + field + "_" + spin_name
                if op == "coulomb"
                else "raw/" + op + "/" + spin_name
            )

        # Forward pass (and metric solve for Coulomb)
        if projected is None:
            for spin, occ in occupied.items():
                if field not in hcore_frames[spin].outputs:
                    continue
                key = _op_key(spin)
                if same_spin and spin == "beta":
                    data[key] = data[_op_key("alpha")]
                    if op == "coulomb" and occ:
                        data["source/fit_beta"] = data["source/fit_alpha"]
                    continue
                if use_occ_coul and occ == 0:
                    data[key] = np.empty((aux, 0, ext), dtype=np.float64)
                else:
                    data[key] = project_ao_operator_tiled(
                        nc, ext, occ, aux, graph_field, graph_ao_name, ao["coefficients_" + spin], ao_mat, run
                    )
                if op == "coulomb" and occ:
                    solve = coulomb_fit_response(aux, occ * ext)
                    raw = np.ascontiguousarray(data[key][:, :occ]).reshape(aux, occ * ext)
                    solved = run(
                        Program({"fit": solve.outputs["fit"]}),
                        {"metric_whitener": ao["metric_whitener"], "three_center": raw},
                    ).outputs
                    data["source/fit_" + spin] = solved["fit"].reshape(aux, occ, ext)

        # Reverse pass
        if bars is not None:
            assert orbital_weights is not None
            target_bar_ao = bar_ao_mat if sink_this else bars[ao_name]
            # Fast spin-symmetric reverse when alpha and beta share frames and cotangents
            can_share_rev = (
                same_spin
                and all(field in hcore_frames[s].outputs for s in ("alpha", "beta"))
                and np.array_equal(orbital_weights[_op_key("alpha")], orbital_weights[_op_key("beta")])
                and (
                    op != "coulomb"
                    or oa == 0
                    or np.array_equal(orbital_weights["source/fit_alpha"], orbital_weights["source/fit_beta"])
                )
            )
            if can_share_rev:
                occ = oa
                key = _op_key("alpha")
                if op == "coulomb" and occ:
                    seed = orbital_weights[key].copy()
                    solve = coulomb_fit_response(aux, occ * ext)
                    raw = np.ascontiguousarray(data[key][:, :occ]).reshape(aux, occ * ext)
                    fit_arg = {
                        "metric_whitener": ao["metric_whitener"],
                        "three_center": raw,
                        "fit_bar": orbital_weights["source/fit_alpha"].reshape(aux, -1),
                    }
                    if projected is not None:
                        solve = Program(
                            {k: solve.outputs[k] for k in ("three_center_bar", "metric_bar")}
                        )
                    solved = run(solve, fit_arg).outputs
                    seed[:, :occ] += solved["three_center_bar"].reshape(aux, occ, ext)
                    bars["metric/coulomb"] += 2.0 * solved["metric_bar"]
                else:
                    seed = orbital_weights[key]
                if not (use_occ_coul and occ == 0):
                    bar_c_both = np.zeros_like(bars["coefficients_alpha"])
                    project_ao_operator_pullback_tiled(
                        nc,
                        ext,
                        occ,
                        aux,
                        graph_field,
                        graph_ao_name,
                        ao["coefficients_alpha"],
                        ao_mat,
                        seed,
                        run,
                        bar_coefficients=bar_c_both,
                        bar_ao=target_bar_ao,
                        seed_scale=2.0,
                    )
                    bars["coefficients_alpha"] += 0.5 * bar_c_both
                    bars["coefficients_beta"] += 0.5 * bar_c_both
                    del bar_c_both
                del seed
            else:
                for spin, occ in occupied.items():
                    if field not in hcore_frames[spin].outputs:
                        continue
                    co = "coefficients_" + spin
                    key = _op_key(spin)
                    if op == "coulomb" and occ:
                        seed = orbital_weights[key].copy()
                        solve = coulomb_fit_response(aux, occ * ext)
                        raw = np.ascontiguousarray(data[key][:, :occ]).reshape(aux, occ * ext)
                        fit_key = "source/fit_" + spin
                        fit_arg = {
                            "metric_whitener": ao["metric_whitener"],
                            "three_center": raw,
                            "fit_bar": orbital_weights[fit_key].reshape(aux, -1),
                        }
                        if projected is not None:
                            solve = Program(
                                {k: solve.outputs[k] for k in ("three_center_bar", "metric_bar")}
                            )
                        solved = run(solve, fit_arg).outputs
                        seed[:, :occ] += solved["three_center_bar"].reshape(aux, occ, ext)
                        bars["metric/coulomb"] += solved["metric_bar"]
                    else:
                        seed = orbital_weights[key]
                    if not (use_occ_coul and occ == 0):
                        project_ao_operator_pullback_tiled(
                            nc,
                            ext,
                            occ,
                            aux,
                            graph_field,
                            graph_ao_name,
                            ao[co],
                            ao_mat,
                            seed,
                            run,
                            bar_coefficients=bars[co],
                            bar_ao=target_bar_ao,
                        )
                    del seed
        del ao_mat
        if sink_this:
            assert bar_ao_mat is not None
            ao_bar_sink(ao_name, bar_ao_mat)
            del bar_ao_mat
    return data, bars
