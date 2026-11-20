"""Molecular retained 3C(FIX) pair increment with separately fitted canonical UHF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from generativeqc.experimental.f12_focal import (
    FocalF12Sources,
    df_reference,
    one_electron_sources,
    reference_f12_increment,
)

from benchmarks.f12_focal_shell_layout import focal_shell_layout
from benchmarks.f12_molecular_integral_forward import forward_integrals
from benchmarks.f12_molecular_metric import metric_whitener
from benchmarks.f12_molecular_sink_live import contract_live_weights

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable
    from pathlib import Path

    from generativeqc.experimental.f12_focal import FocalRequest


@dataclass(frozen=True)
class MolecularF12Increment:
    """The explicit F12 energy in Eh and its relaxed analytic force in Eh/Bohr."""

    energy: float
    forces: np.ndarray
    reference_energy: float
    cabs_rank: int
    active_pairs: int
    executed_backend: str
    surface_id: str
    basis_id: str


def molecular_f12_increment(
    request: FocalRequest,
    *,
    shell_library: Path,
    one_electron_library: Path,
    metric_library: Path | None = None,
    response_library: ctypes.CDLL | None = None,
    canonical_library: ctypes.CDLL | None = None,
    run: Callable | None = None,
    expected_cabs_rank: int | None = None,
    relative_cutoff: float = 1e-9,
    budget_mib: int = 1536,
) -> MolecularF12Increment:
    """Solve the physical UHF reference and contract its explicit F12 pair response.

    The two integral layouts use the same full joint OBS/OptRI basis. The
    extended Fock uses JKFit and every Gaussian pair operator uses F12Fit.
    """
    host = request.backend == "cpu_reference"
    if (
        request.role != "mp2_f12_correlation"
        or request.correlation_space_id != "all-active"
        or request.root_id != "uhf"
        or request.backend not in ("cuda", "cpu_reference")
        or (not host
        and (metric_library is None or run is None))
    ):
        raise ValueError(
            "selected F12 increment needs its all-active UHF pair request and explicit backend"
        )
    reference = df_reference(request, forces=False, budget_mib=budget_mib)
    one = one_electron_sources(
        request, library=one_electron_library, budget_mib=budget_mib
    )
    panels, layouts = {}, {}
    for name, only in (("jk_fit", True), ("f12_fit", False)):
        layout = focal_shell_layout(
            request,
            shell_library,
            host_reference=host,
            budget_mib=budget_mib,
            joint_jk=only,
        )
        sources, _ = forward_integrals(
            layout,
            shell_library,
            host_reference=host,
            coulomb_only=only,
            budget_mib=budget_mib,
        )
        metric = sources["metric"]
        if host:
            from scipy.linalg import solve_triangular

            whitener = solve_triangular(
                np.linalg.cholesky(metric), np.eye(len(metric)), lower=True
            )
        else:
            whitener, _ = metric_whitener(
                metric, metric_library, budget_bytes=budget_mib << 20
            )
        panels[name] = FocalF12Sources(
            request.surface_id,
            request.basis.basis_id,
            getattr(request.basis, name).identity,
            request.backend,
            None if only else request.basis.beta,
            sources,
            whitener,
        )
        layouts[name] = layout
    value = reference_f12_increment(
        request,
        reference,
        one,
        panels["jk_fit"],
        panels["f12_fit"],
        expected_cabs_rank=expected_cabs_rank,
        relative_cutoff=relative_cutoff,
        budget_mib=budget_mib,
        run=run,
        library=response_library,
        canonical_library=canonical_library,
    )
    assert one.d_overlap is not None and one.d_hcore is not None
    gradient = np.einsum(
        "axij,ij->ax", one.d_overlap, value.one_electron_weights["overlap"]
    )
    gradient += np.einsum(
        "axij,ij->ax", one.d_hcore, value.one_electron_weights["hcore"]
    )
    for name, weights in (
        ("jk_fit", value.jkfit_weights),
        ("f12_fit", value.f12fit_weights),
    ):
        out = contract_live_weights(
            layouts[name],
            weights,
            shell_library,
            host_reference=host,
            singles_only=name == "jk_fit",
            budget_mib=budget_mib,
        )
        if not out["passes_contraction_checks"]:
            raise RuntimeError(f"F12 {name} nuclear shell contraction failed")
        gradient += np.asarray(out["gradient_Eh_per_bohr"])
    return MolecularF12Increment(
        value.energy,
        -gradient,
        reference.energy,
        value.cabs_rank,
        value.active_pairs,
        request.backend,
        request.surface_id,
        request.basis.basis_id,
    )
