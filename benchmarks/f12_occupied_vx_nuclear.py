"""AO integral and orbital-frame response of the occupied CC-F12 V and X tables."""

from collections.abc import Callable
from typing import TypedDict

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from generativeqc_compiler.mp2.f12_occupied_pair_overlap import all_occupied_vx
from generativeqc_compiler.tensor import Program
from generativeqc_compiler.tensor.ad_program import transpose_program


OPERATORS = ("coulomb", "f12", "f12_over_coulomb", "f12_squared")


class OccupiedVXWeights(TypedDict):
    ao: dict[str, np.ndarray]
    metric: dict[str, np.ndarray]
    coefficients: np.ndarray


def occupied_vx_ao_weights(
    coefficients: np.ndarray,
    ao: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    occupied: int,
    obs: int,
    weights: dict[str, np.ndarray],
    run: Callable,
) -> tuple[dict[str, np.ndarray], dict[str, OccupiedVXWeights]]:
    """Differentiate each seeded V or X scalar down to AO two/three-centers and the common frame.

    Each key of weights has shape (occupied,)*4. The returned metric and AO
    weights need unsymmetrized Frobenius contractions in the supplied AO order.
    """
    n_aux = len(metrics["coulomb"])
    combined, ri = coefficients.shape
    if set(weights) - {"v", "x"} or not weights or not 0 < occupied < obs < ri:
        raise ValueError("Expected an occupied V/X source with OBS and CABS")
    if any(w.shape != (occupied,) * 4 for w in weights.values()):
        raise ValueError("Incorrect occupied-pair cotangent shape")
    occ_coeff = coefficients[:, :occupied]
    raw = {k: occ_coeff.T @ ao[k] @ coefficients for k in OPERATORS}
    metric_factor = cho_factor(metrics["coulomb"])
    fit = cho_solve(metric_factor, raw["coulomb"].reshape(n_aux, -1)).reshape(
        raw["coulomb"].shape
    )
    graph = all_occupied_vx(occupied, obs - occupied, ri - obs, n_aux)
    feed = {
        "fit": fit,
        **{"raw_" + k: raw[k] for k in OPERATORS},
        **{"metric_" + k: metrics[k] for k in OPERATORS if k != "coulomb"},
    }
    values = run(graph, feed).outputs
    seeds = {}
    for label, weight in weights.items():
        selected = Program({label: graph.outputs[label]}, provenance=graph.provenance)
        reverse = transpose_program(selected, (label,)).program
        live = {node.attrs["name"] for node in reverse.live_nodes if node.op == "input"}
        full_feed = {**feed, "bar_" + label: weight}
        response = run(reverse, {name: full_feed[name] for name in live}).outputs
        fit_weight = response["bar_fit"].reshape(n_aux, -1)
        rhs_weight = cho_solve(metric_factor, fit_weight)
        metric_bars = {"coulomb": -(rhs_weight @ fit.reshape(n_aux, -1).T)}
        raw_bars = {"coulomb": rhs_weight.reshape(raw["coulomb"].shape)}
        for op in OPERATORS:
            name = "bar_raw_" + op
            if name in response:
                raw_bars[op] = raw_bars.get(op, 0.0) + response[name]
            name = "bar_metric_" + op
            if name in response:
                metric_bars[op] = response[name]
        coef_bar = np.zeros((combined, ri))
        ao_bars = {}
        for op, w in raw_bars.items():
            ao_bars[op] = occ_coeff @ w @ coefficients.T
            # Contributions from the occupied and unrestricted extended orbital legs.
            coef_bar[:, :occupied] += np.einsum(
                "Pui,Pji->uj", ao[op] @ coefficients, w, optimize=True
            )
            coef_bar += np.einsum(
                "Pui,Pij->uj", ao[op].swapaxes(1, 2) @ occ_coeff, w, optimize=True
            )
        seeds[label] = {"ao": ao_bars, "metric": metric_bars, "coefficients": coef_bar}
    return values, seeds


def complementary_frame_tangent(
    coefficients: np.ndarray,
    obs: int,
    orbital_motion: np.ndarray,
    overlap_derivative: np.ndarray,
    half_overlap_derivative: np.ndarray,
) -> np.ndarray:
    """Move the complementary basis consistently with OBS CPHF and parallel transport within CABS.

    The columns form a complete S-orthonormal OBS+CABS frame in the combined AO
    basis. The OBS shells are fixed members of that combined basis. The
    supplied occupied/virtual orbital_motion is the coefficient derivative
    within the OBS block, including the finite AO overlap connection.
    """
    c = coefficients
    if (
        c.shape[0] != c.shape[1]
        or orbital_motion.shape != (obs, obs)
        or not 0 < obs < c.shape[1]
    ):
        raise ValueError(
            "A complete OBS+CABS frame and compatible orbital response are required"
        )
    overlap = c.T @ overlap_derivative @ c
    right = c.T @ half_overlap_derivative @ c
    connection = np.zeros_like(overlap)
    connection[:obs, :obs] = orbital_motion
    connection[:obs, obs:] = -overlap[:obs, obs:]
    connection[obs:, obs:] = -right[obs:, obs:]
    if not np.allclose(connection + connection.T, -overlap, rtol=0, atol=5e-8):
        raise ValueError("Orbital coefficients break extended-frame orthonormality")
    return c @ connection
