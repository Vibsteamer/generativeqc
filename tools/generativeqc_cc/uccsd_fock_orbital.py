"""Canonical DF-UHF orbital response of the six ordinary unrestricted CC Fock blocks."""

import ctypes
from collections.abc import Callable, Mapping
from typing import NamedTuple

import numpy as np
from generativeqc.f12_orbital_response import (
    unrestricted_canonical_source_seeds,
    unrestricted_nuclear_source_weights,
    unrestricted_source_orbital_adjoint,
)
from generativeqc_compiler.method.canonical_frame_response import (
    orbital_metric_cotangent,
)
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    input_tensor,
)

from .uccsd_fock_nuclear import _spin_fock_weight


class UnrestrictedCCFockOrbitalResponse(NamedTuple):
    weights: dict[str, np.ndarray]
    driver: np.ndarray
    rotations: np.ndarray


def unrestricted_cc_fock_frame_weights(
    occupied: tuple[int, int],
    weights: Mapping[str, np.ndarray],
    coefficients: Mapping[str, np.ndarray],
    hcore: np.ndarray,
    jk_orbital_weights: Mapping[str, np.ndarray],
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Combine Hcore and J−K coefficient derivatives and return their Pulay overlap weight.

    Coefficients and J−K weights use the same AO rows and complete ordinary
    MO columns. Frozen orbitals remain in the mean field and metric transport.
    The Fock block weights exclude frozen rows/columns in the CC occupied space.
    """
    nao, nmo = coefficients["coefficients_alpha"].shape
    if hcore.shape != (nao, nao) or hcore.dtype != np.float64:
        raise ValueError("Inconsistent core Hamiltonian or AO coefficients")
    aos, mos = IndexSpace("ao", "ao", nao), IndexSpace("mo", "orbital", nmo)
    u, v = (Index(x, aos) for x in ("u", "v"))
    i, j = (Index(x, mos) for x in ("i", "j"))
    h = input_tensor("hcore", TensorSpec((u, v), role="input"))
    c = input_tensor("coefficients", TensorSpec((v, i), role="input"))
    bar = input_tensor("fock_weight", TensorSpec((i, j), role="input"))
    jk = input_tensor("jk_weight", TensorSpec((u, j), role="input"))
    cb = einsum("vi,ij->vj", c, bar)
    graph = Program({"total": add(einsum("uv,vj->uj", h, cb, coefficient=2), jk)})
    metric_graph = orbital_metric_cotangent(nao, nmo)
    coeff_bars = {}
    overlap = np.zeros_like(hcore)
    for spin, no, nf in zip(("alpha", "beta"), occupied, frozen_occupied, strict=True):
        name = "coefficients_" + spin
        frame = coefficients[name]
        if frame.shape != (nao, nmo) or jk_orbital_weights[name].shape != frame.shape:
            raise ValueError(
                "Orbital coefficients and J−K cotangents must share an AO/MO frame"
            )
        fields = {
            "hcore": hcore,
            "coefficients": frame,
            "fock_weight": _spin_fock_weight(weights, spin, nmo, no, nf),
            "jk_weight": jk_orbital_weights[name],
        }
        coeff_bars[name] = run(graph, fields).outputs["total"]
        overlap += run(
            metric_graph,
            {
                "coefficients": frame,
                "frame_bar": coeff_bars[name],
                "metric_blocks": np.full((nmo, nmo), -0.5),
            },
        ).outputs["overlap_bar"]
    return coeff_bars, overlap


def unrestricted_cc_fock_orbital_response(
    space: tuple[int, int, int, int],
    central: Mapping[str, np.ndarray],
    energies: Mapping[str, np.ndarray],
    coefficient_weights: Mapping[str, np.ndarray],
    frame_overlap: np.ndarray,
    run: Callable,
    *,
    canonical_library: ctypes.CDLL | None = None,
    orbital_library: ctypes.CDLL | None = None,
    maximum_bytes: int = 1 << 30,
) -> UnrestrictedCCFockOrbitalResponse:
    """Transfer the complete two-spin orbital response to four nuclear integral weights.

    Central DF factors, their full-rank whitener, energies and MO columns must
    describe the same canonical UHF solution. The native coupled-spin Z solve
    uses one electronic driver for all nuclear coordinates. Add the
    returned energy weights to explicit Hcore and J−K weights or contract
    their negative with the matching AO derivatives for the orbital force.
    """
    nao = len(central["coefficients_alpha"])
    if (
        frame_overlap.shape != (nao, nao)
        or frame_overlap.dtype != np.float64
        or not np.isfinite(frame_overlap).all()
    ):
        raise ValueError("Pulay overlap weight and nuclear AO basis disagree")
    seeds, _ = unrestricted_canonical_source_seeds(
        space[:2],
        central,
        energies,
        coefficient_weights,
        run,
        library=canonical_library,
        maximum_bytes=maximum_bytes,
    )
    fields = {**central, **seeds}
    driver, z = unrestricted_source_orbital_adjoint(
        space,
        fields,
        energies,
        run,
        library=orbital_library,
        maximum_bytes=maximum_bytes,
    )
    physical = unrestricted_nuclear_source_weights(space, fields, z, run)
    physical["overlap_ao"] += frame_overlap
    return UnrestrictedCCFockOrbitalResponse(physical, driver, z)
