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

_OPERATORS = {
    "hcore": "hcore",
    "coulomb": "raw_coulomb",
    "geminal": "raw_f12",
    "squared_geminal": "raw_f12_squared",
    "geminal_coulomb": "raw_f12_over_coulomb",
    "double_commutator": "raw_double_commutator",
}


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
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    # A combined energy/response evaluation owns the forward buffers through
    # the reverse pass. Standalone evaluations build their own projections.
    oa, ob, obs, ext, aux = space
    valid_frame = obs == ext if role in ("ordinary_jk", "ordinary_fock") else obs < ext
    if not (0 <= oa < obs and 0 <= ob < obs and valid_frame and oa + ob > 0):
        raise ValueError("Invalid UHF occupied, OBS, or extended dimensions")
    nc = len(ao["coefficients_alpha"])
    occupied = {"alpha": oa, "beta": ob}
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
    frames = {
        spin: molecular_orbital_sources(nc, ext, occ, aux)
        for spin, occ in occupied.items()
    }
    # Iterate operators outermost so streaming mode holds at most ONE [Q, ext, ext] AO panel
    # (and at most ONE [Q, ext, ext] cotangent) in host RAM across both spin channels.
    for op in operators:
        field = _OPERATORS[op]
        ao_name = "hcore_ao" if op == "hcore" else "ao_" + field.removeprefix("raw_")
        if not any(field in frames[spin].outputs for spin in occupied):
            continue
        stream_this = op != "hcore" and ao_loader is not None and ao_name not in ao
        ao_mat = ao_loader(ao_name) if stream_this else ao[ao_name]
        sink_this = bars is not None and op != "hcore" and ao_bar_sink is not None
        bar_ao_mat = None
        if bars is not None and op != "hcore" and not sink_this and ao_name not in bars:
            bars[ao_name] = np.zeros_like(ao_mat)
        for spin, occ in occupied.items():
            frame = frames[spin]
            if field not in frame.outputs:
                continue
            co = "coefficients_" + spin
            key = (
                "source/" + field + "_" + spin
                if op in ("coulomb", "hcore")
                else "raw/" + op + "/" + spin
            )
            graph = Program({field: frame.outputs[field]})
            item = {"coefficients": ao[co], ao_name: ao_mat}
            if projected is None:
                data[key] = run(graph, item).outputs[field]
            seed = None if bars is None else orbital_weights[key].copy()
            if op == "coulomb" and occ:
                solve = coulomb_fit_response(aux, occ * ext)
                raw = np.ascontiguousarray(data[key][:, :occ]).reshape(aux, occ * ext)
                fit_key = "source/fit_" + spin
                fit_arg = {
                    "metric_whitener": ao["metric_whitener"],
                    "three_center": raw,
                }
                if bars is None:
                    solved = run(
                        Program({"fit": solve.outputs["fit"]}), fit_arg
                    ).outputs
                else:
                    fit_arg["fit_bar"] = orbital_weights[fit_key].reshape(aux, -1)
                    if projected is not None:
                        solve = Program(
                            {
                                k: solve.outputs[k]
                                for k in ("three_center_bar", "metric_bar")
                            }
                        )
                    solved = run(solve, fit_arg).outputs
                    seed[:, :occ] += solved["three_center_bar"].reshape(aux, occ, ext)
                    bars["metric/coulomb"] += solved["metric_bar"]
                if projected is None:
                    data[fit_key] = solved["fit"].reshape(aux, occ, ext)
            if bars is not None:
                rev = transpose_program(graph, (field,)).program
                weights = run(rev, {**item, "bar_" + field: seed}).outputs
                bars[co] += weights["bar_coefficients"]
                if sink_this:
                    contrib = weights["bar_" + ao_name]
                    if bar_ao_mat is None:
                        bar_ao_mat = np.array(contrib, copy=True) if not contrib.flags.writeable else contrib
                    else:
                        bar_ao_mat += contrib
                    del contrib
                else:
                    bars[ao_name] += weights["bar_" + ao_name]
                del weights
        del ao_mat
        if sink_this:
            assert bar_ao_mat is not None
            ao_bar_sink(ao_name, bar_ao_mat)
            del bar_ao_mat
    return data, bars
