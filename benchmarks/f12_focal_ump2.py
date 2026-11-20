"""Geometry-driven ordinary MP2 part of a canonical focal pair request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from generativeqc.experimental.f12_focal import (
    FocalCoulombSource,
    FocalRequest,
    df_reference,
    one_electron_sources,
    reference_df_ump2,
)

from benchmarks.f12_focal_shell_layout import focal_shell_layout
from benchmarks.f12_molecular_integral_forward import forward_integrals
from benchmarks.f12_molecular_metric import metric_whitener
from benchmarks.f12_molecular_sink_live import contract_live_weights

if TYPE_CHECKING:
    import ctypes
    from collections.abc import Callable
    from pathlib import Path


@dataclass(frozen=True)
class MolecularUMP2:
    """HF and ordinary MP2 correlation contributions in Eh and Eh/Bohr."""

    reference_energy: float
    reference_forces: np.ndarray
    spin_energies: tuple[float, float, float]
    correlation_forces: np.ndarray
    executed_backend: str
    surface_id: str
    basis_id: str
    jk_fit_id: str
    mp2_fit_id: str


def molecular_df_ump2(
    request: FocalRequest,
    *,
    shell_library: Path,
    one_electron_library: Path,
    metric_library: Path | None = None,
    response_library: ctypes.CDLL | None = None,
    run: Callable | None = None,
    budget_mib: int = 512,
) -> MolecularUMP2:
    """Evaluate the all-active RHF or UHF ordinary correlation energy and its full relaxed force.

    The geometry defines both independently fitted Coulomb operators. SCF, each
    shell derivative contraction and the electronic response use the requested
    backend. The measured sum of the three spin classes is the ordinary DF-MP2 energy.
    """
    host = request.backend == "cpu_reference"
    if (
        request.role != "mp2_f12_correlation"
        or request.root_id not in ("rhf", "uhf")
        or request.correlation_space_id != "all-active"
        or request.backend not in ("cuda", "cpu_reference")
    ):
        raise ValueError(
            "ordinary molecular DF-UMP2 requires an all-active RHF or UHF pair request and explicit backend"
        )
    if not host and (metric_library is None or run is None):
        raise ValueError(
            "CUDA ordinary MP2 requires a CUDA metric library and FP64 graph runner"
        )
    reference = df_reference(request, budget_mib=budget_mib)
    one = one_electron_sources(
        request, library=one_electron_library, orbital_only=True, budget_mib=budget_mib
    )
    if not np.allclose(
        one.overlap, reference.overlap, rtol=0, atol=2e-8
    ) or not np.allclose(one.hcore, reference.hcore, rtol=0, atol=2e-8):
        raise ValueError("molecular one-electron and canonical OBS frame disagree")
    layouts, sources = {}, {}
    for role in ("jk_fit", "mp2_fit"):
        layout = focal_shell_layout(
            request,
            shell_library,
            host_reference=host,
            budget_mib=budget_mib,
            ordinary_fit=role,
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
            whitener, _ = metric_whitener(
                metric, metric_library, budget_bytes=budget_mib << 20
            )
        sources[role] = FocalCoulombSource(
            request.surface_id,
            request.basis.orbital.identity,
            getattr(request.basis, role).identity,
            request.backend,
            raw["ao_coulomb"],
            metric,
            whitener,
        )
        layouts[role] = layout
    response = reference_df_ump2(
        request,
        reference,
        sources["jk_fit"],
        sources["mp2_fit"],
        run=run,
        library=response_library,
        budget_mib=budget_mib,
    )
    w = response.weights
    assert (
        one.d_overlap is not None
        and one.d_hcore is not None
        and reference.forces is not None
    )
    gradient = np.einsum("axij,ij->ax", one.d_overlap, w["overlap"])
    gradient += np.einsum("axij,ij->ax", one.d_hcore, w["hcore"])
    for role, prefix in (("jk_fit", ""), ("mp2_fit", "mp2_")):
        out = contract_live_weights(
            layouts[role],
            {"metric": w[prefix + "metric"], "ao_coulomb": w[prefix + "three_center"]},
            shell_library,
            host_reference=host,
            singles_only=True,
            budget_mib=budget_mib,
        )
        if not out["passes_contraction_checks"]:
            raise RuntimeError(f"{role} nuclear derivative contraction failed")
        gradient += np.asarray(out["gradient_Eh_per_bohr"])
    return MolecularUMP2(
        reference.energy,
        reference.forces,
        response.spin_energies,
        -gradient,
        request.backend,
        request.surface_id,
        request.basis.basis_id,
        reference.jk_fit_id,
        sources["mp2_fit"].fit_id,
    )
