"""Unrestricted 3C(FIX) orbital sources and their reverse map to a common AO basis."""

from collections.abc import Callable

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
    oa, ob, obs, ext, aux = space
    if not (0 <= oa < obs and 0 <= ob < obs < ext and oa + ob > 0):
        raise ValueError("Invalid UHF occupied, OBS, or extended dimensions")
    nc = len(ao["overlap_ao"])
    occupied = {"alpha": oa, "beta": ob}
    cross = unrestricted_cross_frame_overlap(nc, ext)
    csrc = {k: ao[k] for k in ("overlap_ao", "coefficients_alpha", "coefficients_beta")}
    data = {
        "space": np.array(space),
        "cross_overlap": run(cross, csrc).outputs["cross_overlap"],
    }
    bars = None
    if orbital_weights is not None:
        bars = {
            k: np.zeros_like(v) for k, v in ao.items() if k not in ("metric_whitener",)
        }
        cross_rev = transpose_program(cross, ("cross_overlap",)).program
        result = run(
            cross_rev, {**csrc, "bar_cross_overlap": orbital_weights["cross_overlap"]}
        ).outputs
        for name, value in result.items():
            bars[name.removeprefix("bar_")] += value
    for op in _OPERATORS:
        if op not in ("hcore", "coulomb"):
            name = "metric/" + op
            data[name] = ao[name]
            if bars is not None:
                bars[name] += orbital_weights[name]
    for spin, occ in occupied.items():
        co = "coefficients_" + spin
        frame = molecular_orbital_sources(nc, ext, occ, aux)
        for op, field in _OPERATORS.items():
            if field not in frame.outputs:
                continue
            ao_name = (
                "hcore_ao" if op == "hcore" else "ao_" + field.removeprefix("raw_")
            )
            key = (
                "source/" + field + "_" + spin
                if op in ("coulomb", "hcore")
                else "raw/" + op + "/" + spin
            )
            graph = Program({field: frame.outputs[field]})
            item = {"coefficients": ao[co], ao_name: ao[ao_name]}
            value = run(graph, item).outputs[field]
            data[key] = value
            seed = None if bars is None else orbital_weights[key].copy()
            if op == "coulomb" and occ:
                solve = coulomb_fit_response(aux, occ * ext)
                raw = np.ascontiguousarray(value[:, :occ]).reshape(aux, occ * ext)
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
                    solved = run(solve, fit_arg).outputs
                    seed[:, :occ] += solved["three_center_bar"].reshape(aux, occ, ext)
                    bars["metric/coulomb"] += solved["metric_bar"]
                data[fit_key] = solved["fit"].reshape(aux, occ, ext)
            if bars is not None:
                rev = transpose_program(graph, (field,)).program
                weights = run(rev, {**item, "bar_" + field: seed}).outputs
                bars[co] += weights["bar_coefficients"]
                bars[ao_name] += weights["bar_" + ao_name]
    return data, bars
