"""Reverse ordinary unrestricted CC two-electron blocks to Gaussian DF and orbital sources."""

import ctypes
from collections.abc import Callable, Mapping
from typing import NamedTuple

import numpy as np
from generativeqc._uccsd_df_eri_response import unrestricted_cc_eri_pullback
from generativeqc_compiler.method.canonical_frame_response import (
    orbital_metric_cotangent,
)

from .uccsd_fock_orbital import unrestricted_cc_fock_orbital_response


def unrestricted_cc_eri_frame_overlap(
    coefficients: Mapping[str, np.ndarray],
    eri_weights: Mapping[str, np.ndarray],
    run: Callable,
) -> np.ndarray:
    """Transfer the symmetric normalization response of both ordinary MO frames to S."""
    nao, nmo = coefficients["alpha"].shape
    graph = orbital_metric_cotangent(nao, nmo)
    mask = np.full((nmo, nmo), -0.5)
    overlap = np.zeros((nao, nao))
    for spin, c in coefficients.items():
        overlap += run(
            graph,
            {
                "coefficients": c,
                "frame_bar": eri_weights["coefficients_" + spin],
                "metric_blocks": mask,
            },
        ).outputs["overlap_bar"]
    return overlap


class UnrestrictedCCERIResponse(NamedTuple):
    direct: dict[str, np.ndarray]
    orbital: dict[str, np.ndarray]
    driver: np.ndarray
    rotations: np.ndarray


def unrestricted_cc_eri_nuclear_response(
    occupied: tuple[int, int],
    weights: Mapping[str, np.ndarray],
    central: Mapping[str, np.ndarray],
    energies: Mapping[str, np.ndarray],
    three_center: np.ndarray,
    metric: np.ndarray,
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    canonical_library: ctypes.CDLL | None = None,
    orbital_library: ctypes.CDLL | None = None,
    maximum_bytes: int = 1 << 30,
) -> UnrestrictedCCERIResponse:
    """Map the ordinary two-electron Lagrangian to direct and orbital nuclear weights.

    The canonical reference fields and orbital energies use the signed UHF
    frame used by the CC integral blocks. Sum the direct and orbital energy
    weights for each physical AO operator before contracting its nuclear
    derivative. The direct spin coefficient weights are inputs to the coupled
    solve and may also be used for an independent Cartesian check.
    """
    c = {s: central["coefficients_" + s] for s in ("alpha", "beta")}
    direct = unrestricted_cc_eri_pullback(
        occupied,
        weights,
        c,
        three_center,
        metric,
        central["whitener"],
        run,
        frozen_occupied=frozen_occupied,
    )
    frame_overlap = unrestricted_cc_eri_frame_overlap(c, direct, run)
    response = unrestricted_cc_fock_orbital_response(
        (*occupied, c["alpha"].shape[1], len(metric)),
        central,
        energies,
        direct,
        frame_overlap,
        run,
        canonical_library=canonical_library,
        orbital_library=orbital_library,
        maximum_bytes=maximum_bytes,
    )
    return UnrestrictedCCERIResponse(
        direct, response.weights, response.driver, response.rotations
    )
