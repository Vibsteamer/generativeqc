"""Analytic nuclear input weights for the standard external CC-F12 V projection."""

from collections.abc import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from generativeqc_compiler.cc.f12_residual import df_standard_v
from generativeqc_compiler.tensor.ad_program import transpose_program


EXTERNAL_OPERATORS = ("coulomb", "f12", "f12_over_coulomb")


def external_v_ao_weights(
    coefficients: np.ndarray,
    ao: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    weight: np.ndarray,
    run: Callable,
) -> tuple[np.ndarray, dict]:
    """Return the occupied-pair × OBS² external V table and its weighted AO transpose.

    Real restricted orbitals span the OBS columns of coefficients. AO factors
    and metrics are ordered (P,u,v) and (P,Q). The independent input weight
    can lack pair symmetry. Contract all returned weights in full AO order.
    """
    if coefficients.ndim != 2 or weight.ndim != 4:
        raise ValueError(
            "Expected an OBS coefficient matrix and an occupied-pair external table"
        )
    n = coefficients.shape[1]
    o = weight.shape[0]
    if not 0 < o <= n or weight.shape != (o, o, n, n):
        raise ValueError(
            "External weight must have dimensions (occupied,occupied,OBS,OBS)"
        )
    aux = len(metrics["coulomb"])
    raw = {op: coefficients.T @ ao[op] @ coefficients for op in EXTERNAL_OPERATORS}
    factor = cho_factor(metrics["coulomb"])
    fit = cho_solve(factor, raw["coulomb"].reshape(aux, -1)).reshape(aux, n, n)
    graphs = {d: df_standard_v(n, aux, diagonal=d) for d in (True, False)}
    reverses = {d: transpose_program(g, ("v",)).program for d, g in graphs.items()}
    output = np.empty_like(weight, dtype=np.float64)
    fit_bar = np.zeros_like(fit)
    raw_bar = {op: np.zeros_like(t) for op, t in raw.items()}
    metric_bar = {op: np.zeros_like(metrics[op]) for op in EXTERNAL_OPERATORS}
    geminals = EXTERNAL_OPERATORS[1:]
    select = np.eye(n)
    for m, k in zip(*np.triu_indices(o), strict=True):
        diagonal = m == k
        feed = {
            "raw_coulomb_all": raw["coulomb"],
            "fit_all": fit,
            "select_m": select[m],
        }
        seed = weight[m, k].copy()
        for op in geminals:
            feed["metric_" + op] = metrics[op]
            feed["raw_" + op + "_m"] = raw[op][:, m, :]
        if not diagonal:
            feed["select_n"] = select[k]
            seed += weight[k, m].T
            for op in geminals:
                feed["raw_" + op + "_n"] = raw[op][:, k, :]
        value = run(graphs[diagonal], feed).outputs["v"]
        output[m, k] = value
        if not diagonal:
            output[k, m] = value.T
        response = run(reverses[diagonal], {**feed, "bar_v": seed}).outputs
        fit_bar += response["bar_fit_all"]
        raw_bar["coulomb"] += response["bar_raw_coulomb_all"]
        for op in geminals:
            metric_bar[op] += response["bar_metric_" + op]
            raw_bar[op][:, m, :] += response["bar_raw_" + op + "_m"]
            if not diagonal:
                raw_bar[op][:, k, :] += response["bar_raw_" + op + "_n"]
    rhs_bar = cho_solve(factor, fit_bar.reshape(aux, -1))
    raw_bar["coulomb"] += rhs_bar.reshape(fit.shape)
    metric_bar["coulomb"] = -rhs_bar @ fit.reshape(aux, -1).T
    coeff_bar = np.zeros_like(coefficients)
    ao_bar = {}
    for op, w in raw_bar.items():
        ao_bar[op] = coefficients @ w @ coefficients.T
        # Both AO legs follow the orbital rotation even for asymmetric cotangents.
        coeff_bar += np.einsum("Pui,Pji->uj", ao[op] @ coefficients, w, optimize=True)
        coeff_bar += np.einsum(
            "Pui,Pij->uj", ao[op].swapaxes(1, 2) @ coefficients, w, optimize=True
        )
    return output, {"ao": ao_bar, "metric": metric_bar, "coefficients": coeff_bar}
