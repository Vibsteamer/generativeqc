"""Geometry-driven DF-HF plus CABS singles provider for the source checkout."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from generativeqc.experimental.f12_focal import (
    FocalEvaluation,
    FocalRequest,
    FocalTerm,
    df_reference,
    focal_cabs_projector,
    one_electron_sources,
    reference_cabs_singles,
)

from benchmarks.f12_focal_shell_layout import focal_shell_layout
from benchmarks.f12_molecular_integral_forward import forward_integrals
from benchmarks.f12_molecular_metric import metric_whitener
from benchmarks.f12_molecular_sink_live import contract_live_weights

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable
    from pathlib import Path


def molecular_hf_cabs(
    request: FocalRequest,
    *,
    shell_library: Path,
    one_electron_library: Path,
    metric_library: Path | None = None,
    response_library: ctypes.CDLL | None = None,
    run: Callable | None = None,
    expected_cabs_rank: int | None = None,
    relative_cutoff: float = 1e-9,
    budget_mib: int = 512,
) -> FocalEvaluation:
    """Return a complete HF+CABS component, including the relaxed analytical forces.

    All basis records and the determinant come from the request. Supply the
    compiled F12 spherical shell forward/sink and one-electron libraries.
    CUDA additionally uses the molecular metric CUDA library, an FP64 CUDA
    TensorIR runner, and the main CUDA response library. CPU reference uses
    a full-rank Cholesky whitening and a coupled spin MINRES orbital solve.
    """
    if request.role != "hf_cabs" or request.backend not in ("cpu_reference", "cuda"):
        raise ValueError(
            "the molecular HF+CABS provider requires the hf_cabs role and an explicit backend"
        )
    host = request.backend == "cpu_reference"
    if not host and (metric_library is None or run is None):
        raise ValueError(
            "molecular HF+CABS on CUDA requires its metric library and FP64 CUDA runner"
        )
    one = one_electron_sources(
        request, library=one_electron_library, budget_mib=budget_mib
    )
    reference = df_reference(request, budget_mib=budget_mib)
    projector = focal_cabs_projector(
        request,
        one,
        expected_rank=expected_cabs_rank,
        relative_cutoff=relative_cutoff,
        library=response_library,
    )
    layout = focal_shell_layout(
        request, shell_library, host_reference=host, budget_mib=budget_mib
    )
    raw, _ = forward_integrals(
        layout,
        shell_library,
        host_reference=host,
        coulomb_only=True,
        budget_mib=budget_mib,
    )
    metric = raw["metric"]
    if host:
        from scipy.linalg import solve_triangular

        whitener = solve_triangular(
            np.linalg.cholesky(metric), np.eye(len(metric)), lower=True
        )
    else:
        assert metric_library is not None
        whitener, _ = metric_whitener(
            metric, metric_library, budget_bytes=budget_mib << 20
        )
    result = reference_cabs_singles(
        request,
        reference,
        one,
        complement=projector.frame,
        three_center=raw["ao_coulomb"],
        metric=metric,
        whitener=whitener,
        relax=True,
        run=run,
        library=response_library,
        relative_cutoff=relative_cutoff,
        budget_mib=budget_mib,
    )
    weights = result.weights
    assert weights is not None
    sources = contract_live_weights(
        layout,
        {"metric": weights["metric"], "ao_coulomb": weights["three_center"]},
        shell_library,
        host_reference=host,
        singles_only=True,
        budget_mib=budget_mib,
    )
    if not sources["passes_contraction_checks"]:
        raise RuntimeError("CABS JKFit shell-derivative contraction failed")
    gradient = np.asarray(sources["gradient_Eh_per_bohr"])
    assert (
        one.d_overlap is not None
        and one.d_hcore is not None
        and reference.forces is not None
    )
    gradient += np.einsum("axij,ij->ax", one.d_overlap, weights["overlap"])
    gradient += np.einsum("axij,ij->ax", one.d_hcore, weights["hcore"])
    return FocalEvaluation(
        FocalTerm(
            request.surface_id,
            request.basis.series_id,
            request.basis.basis_id,
            request.basis.cardinal,
            "hf_cabs",
            result.hf_cabs_energy,
            reference.forces - gradient,
        ),
        executed_backend=request.backend,
        converged=True,
    )
