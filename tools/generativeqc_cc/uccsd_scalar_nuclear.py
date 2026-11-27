"""Orbital-response adapters for the stationary scalar source on retained CC fixtures."""

import ctypes
from collections.abc import Callable, Mapping

import numpy as np
from generativeqc._f12_cc_scalar_nuclear import (
    UnrestrictedCCScalarResponse,
    unrestricted_cc_scalar_ao_weights,
)
from generativeqc_compiler.method.canonical_frame_response import (
    orbital_metric_cotangent,
)

from .uccsd_fock_orbital import (
    UnrestrictedCCFockOrbitalResponse,
    unrestricted_cc_fock_orbital_response,
)

__all__ = [
    "UnrestrictedCCScalarResponse",
    "unrestricted_cc_scalar_ao_weights",
    "unrestricted_cc_scalar_frame_overlap",
    "unrestricted_cc_scalar_orbital_response",
]


def unrestricted_cc_scalar_frame_overlap(
    obs: int,
    ao: Mapping[str, np.ndarray],
    weights: Mapping[str, np.ndarray],
    run: Callable,
) -> np.ndarray:
    """Transfer normalization and OBS/CABS orthogonality in each native spin gauge to S."""
    n, ext = ao["coefficients_alpha"].shape
    if not 0 < obs < ext:
        raise ValueError("A complete OBS and a nonempty complement are required")
    mask = np.full((ext, ext), -0.5)
    mask[:obs, obs:] = -1.0
    mask[obs:, :obs] = 0.0
    graph = orbital_metric_cotangent(n, ext)
    total = np.zeros((n, n))
    for spin in ("alpha", "beta"):
        c = "coefficients_" + spin
        total += run(
            graph,
            {"coefficients": ao[c], "frame_bar": weights[c], "metric_blocks": mask},
        ).outputs["overlap_bar"]
    return total


def unrestricted_cc_scalar_orbital_response(
    space: tuple[int, int, int, int, int],
    ao: Mapping[str, np.ndarray],
    central: Mapping[str, np.ndarray],
    energies: Mapping[str, np.ndarray],
    direct: Mapping[str, np.ndarray],
    run: Callable,
    *,
    canonical_library: ctypes.CDLL | None = None,
    orbital_library: ctypes.CDLL | None = None,
    maximum_bytes: int = 1 << 30,
) -> UnrestrictedCCFockOrbitalResponse:
    """Relax the scalar's two extended-frame coefficient weights through the same UHF root.

    Both spin gauges can use different rotations of their CABS columns. Their
    metric terms combine with the OBS canonical response and its common coupled
    spin solve to deliver weights on S, Hcore, the Coulomb fit and its metric.
    """
    oa, ob, obs, _, aux = space
    for spin in ("alpha", "beta"):
        k = "coefficients_" + spin
        if not np.array_equal(central[k], ao[k][:, :obs]):
            raise ValueError("Reference and scalar OBS frames differ")
    frame = unrestricted_cc_scalar_frame_overlap(obs, ao, direct, run)
    return unrestricted_cc_fock_orbital_response(
        (oa, ob, obs, aux),
        central,
        energies,
        direct,
        frame,
        run,
        canonical_library=canonical_library,
        orbital_library=orbital_library,
        maximum_bytes=maximum_bytes,
    )
