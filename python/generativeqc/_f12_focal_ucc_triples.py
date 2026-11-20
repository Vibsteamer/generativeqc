"""Molecular conventional (T) increment for an unrestricted focal request."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .experimental.f12_focal import FocalEvaluation, FocalRequest


def conventional_df_ucc_triples(
    request: FocalRequest, *, correlation_memory_budget_bytes: int = 0
) -> FocalEvaluation:
    """Compute conventional UCCSD(T) minus UCCSD energy and analytic forces.

    Both native full-DF calculations cold-start on the declared all-active UHF
    surface. They use one auxiliary record for JK and correlation. Each solves
    its own relaxed force equations. The native (T) diagnostic gives the energy
    directly, including when the triples space is empty.
    """
    from .calculator import Calculator
    from .experimental.f12_focal import FocalEvaluation, FocalTerm

    if request.role != "triples" or request.root_id != "uhf":
        raise ValueError(
            "conventional UCC triples requires the triples role and UHF root"
        )
    if request.correlation_space_id != "all-active":
        raise ValueError("conventional UCC triples requires all-active correlation")
    if request.backend not in ("cpu_reference", "cuda"):
        raise ValueError("conventional UCC triples requires an explicit focal backend")
    basis = request.basis
    if basis.mp2_fit is None or basis.mp2_fit.identity != basis.jk_fit.identity:
        raise ValueError(
            "native full-DF UCC triples requires the same JKFit and MP2Fit"
        )

    components = {}
    for method in ("uccsd(t)", "uccsd"):
        calculator = Calculator(
            method=method,
            device="cuda" if request.backend == "cuda" else "cpu",
            basis=basis.orbital,
            auxiliary_basis=basis.mp2_fit,
            density_fitting="full_cuda",
            initial_guess=None,
            correlation_memory_budget_bytes=correlation_memory_budget_bytes,
        )
        try:
            result = calculator.singlepoint(
                request.atoms,
                charge=request.charge,
                multiplicity=request.multiplicity,
                properties=("energy", "forces"),
            )
        finally:
            calculator.clear_cache()
        if not result.converged or result.executed_backend != request.backend:
            raise RuntimeError(f"{method} did not converge on the requested backend")
        if result.correlation is None or result.forces is None:
            raise RuntimeError(f"{method} returned incomplete triples inputs")
        if (
            result.forces.shape != (len(request.atoms), 3)
            or not np.isfinite(result.forces).all()
        ):
            raise RuntimeError(f"{method} must supply all finite Cartesian forces")
        for role, expected in (
            ("orbital", basis.orbital),
            ("auxiliary", basis.mp2_fit),
        ):
            if (
                result.basis_metadata.get(role, {}).get("basis_identity")
                != expected.identity
            ):
                raise RuntimeError(f"{method} used a different {role} basis")
        components[method] = result.energy, result.correlation, result.forces

    corrected_energy, corrected, corrected_forces = components["uccsd(t)"]
    parent_energy, parent, parent_forces = components["uccsd"]
    triples = corrected.ccsd_t_triples_energy
    for left, right in (
        (corrected.reference_energy, parent.reference_energy),
        (corrected.ccsd_correlation_energy, parent.ccsd_correlation_energy),
        (corrected_energy - parent_energy, triples),
    ):
        if (
            not math.isfinite(left)
            or not math.isfinite(right)
            or abs(left - right) > 2e-10
        ):
            raise RuntimeError("native UCCSD(T) and UCCSD parent energies disagree")

    term = FocalTerm(
        request.surface_id,
        basis.series_id,
        basis.basis_id,
        basis.cardinal,
        request.role,
        triples,
        corrected_forces - parent_forces,
    )
    return FocalEvaluation(term, request.backend, converged=True)
