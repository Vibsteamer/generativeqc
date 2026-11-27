"""Analytic C nuclear weights for the stationary restricted fixed-cusp CC-F12 equations."""

from collections.abc import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from generativeqc_compiler.mp2.f12_molecular_pair import (
    extended_df_fock,
    projected_f12_pair,
)
from generativeqc_compiler.tensor import Program
from generativeqc_compiler.tensor.ad_program import transpose_program


def c_ao_weights(
    coefficients: np.ndarray,
    hcore: np.ndarray,
    ao: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    weight: np.ndarray,
    run: Callable,
) -> tuple[np.ndarray, dict]:
    """Propagate independent ordered C weights to extended-frame AO and orbital inputs.

    The column order of coefficients is occupied, virtual, complementary. C
    samples both virtual–complementary geminal blocks and the self-consistent
    extended Fock. All one- and two-electron inputs follow nuclear motion.
    """
    if weight.ndim != 4:
        raise ValueError("Expected an occupied-pair C table")
    o, _, v, _ = weight.shape
    obs = o + v
    combined, r = coefficients.shape
    if (
        min(o, v, r - obs) <= 0
        or weight.shape != (o, o, v, v)
        or hcore.shape != (combined, combined)
    ):
        raise ValueError("Incompatible C, hcore or extended orbital dimensions")
    aux = len(metrics["coulomb"])
    raw = coefficients.T @ ao["coulomb"] @ coefficients
    raw_g = coefficients[:, :o].T @ ao["f12"] @ coefficients
    factor = cho_factor(metrics["coulomb"])
    fit = cho_solve(factor, raw[:, :o].reshape(aux, -1)).reshape(aux, o, r)
    feed = {
        "hcore": coefficients.T @ hcore @ coefficients,
        "raw_coulomb_all": raw,
        "fit": fit,
    }
    extended = extended_df_fock(o, r, aux)
    fock = Program({"fock": extended.outputs["fock"]}, provenance=extended.provenance)
    fmo = run(fock, feed).outputs["fock"]
    graphs = {}
    reverses = {}
    for diagonal in (True, False):
        full = projected_f12_pair(o, v, r - obs, diagonal=diagonal, auxiliaries=aux)
        sub = Program(
            {"c_direct": full.outputs["c_direct"]}, provenance=full.provenance
        )
        graphs[diagonal] = sub
        reverses[diagonal] = transpose_program(sub, ("c_direct",)).program
    c_table = np.empty_like(weight, dtype=np.float64)
    fit_bar = np.zeros_like(fit)
    raw_g_bar = np.zeros_like(raw_g)
    metric_g_bar = np.zeros_like(metrics["f12"])
    fock_bar = np.zeros_like(fmo)
    for i, j in zip(*np.triu_indices(o), strict=True):
        diagonal = i == j
        pair = {
            "fit_i": fit[:, i],
            "raw_f12_i": raw_g[:, i],
            "metric_f12": metrics["f12"],
            "fock": fmo,
        }
        seed = weight[i, j].copy()
        if not diagonal:
            pair.update(fit_j=fit[:, j], raw_f12_j=raw_g[:, j])
            seed += weight[j, i].T
        value = run(graphs[diagonal], pair).outputs["c_direct"]
        c_table[i, j] = value
        if not diagonal:
            c_table[j, i] = value.T
        response = run(reverses[diagonal], {**pair, "bar_c_direct": seed}).outputs
        fit_bar[:, i] += response["bar_fit_i"]
        raw_g_bar[:, i] += response["bar_raw_f12_i"]
        if not diagonal:
            fit_bar[:, j] += response["bar_fit_j"]
            raw_g_bar[:, j] += response["bar_raw_f12_j"]
        metric_g_bar += response["bar_metric_f12"]
        fock_bar += response["bar_fock"]
    fock_response = run(
        transpose_program(fock, ("fock",)).program, {**feed, "bar_fock": fock_bar}
    ).outputs
    hbar = fock_response["bar_hcore"]
    fit_bar += fock_response["bar_fit"]
    solved_bar = cho_solve(factor, fit_bar.reshape(aux, -1))
    raw_bar = fock_response["bar_raw_coulomb_all"].copy()
    raw_bar[:, :o] += solved_bar.reshape(fit.shape)
    ao_bar = {
        "coulomb": coefficients @ raw_bar @ coefficients.T,
        "f12": coefficients[:, :o] @ raw_g_bar @ coefficients.T,
    }
    hcore_bar = coefficients @ hbar @ coefficients.T
    coeff_bar = hcore @ coefficients @ hbar.T + hcore.T @ coefficients @ hbar
    for op, n_left, w in (("coulomb", r, raw_bar), ("f12", o, raw_g_bar)):
        coeff_bar[:, :n_left] += np.einsum(
            "Pui,Pji->uj", ao[op] @ coefficients, w, optimize=True
        )
        coeff_bar += np.einsum(
            "Pui,Pij->uj",
            ao[op].swapaxes(1, 2) @ coefficients[:, :n_left],
            w,
            optimize=True,
        )
    return c_table, {
        "hcore": hcore_bar,
        "coefficients": coeff_bar,
        "ao": ao_bar,
        "metric": {
            "coulomb": -solved_bar @ fit.reshape(aux, -1).T,
            "f12": metric_g_bar,
        },
    }
