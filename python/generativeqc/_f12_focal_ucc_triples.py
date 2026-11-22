"""Molecular conventional (T) increments on focal references."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._f12_focal_reference import FocalReference
    from .experimental.f12_focal import FocalEvaluation, FocalRequest


def conventional_df_ucc_triples(
    request: FocalRequest, *, correlation_memory_budget_bytes: int = 0
) -> FocalEvaluation:
    """Compute conventional UCCSD(T) minus UCCSD energy and analytic forces.

    Both native full-DF calculations cold-start on the declared all-active UHF
    surface. Each uses the declared JK and correlation fits and solves
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
    if basis.mp2_fit is None:
        raise ValueError(
            "native full-DF UCC triples requires a correlation fitting basis"
        )
    split = basis.mp2_fit.identity != basis.jk_fit.identity

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
                reference_auxiliary_basis=basis.jk_fit if split else None,
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
        if (
            split
            and result.basis_metadata.get("reference_auxiliary", {}).get(
                "basis_identity"
            )
            != basis.jk_fit.identity
        ):
            raise RuntimeError(f"{method} did not use the declared JK fitting basis")
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


def molecular_triples_on_reference(
    request: FocalRequest,
    reference: FocalReference,
    *,
    budget_mib: int,
    forces: bool,
) -> tuple[float, np.ndarray | None]:
    """Use the native conventional CCSD and triples owner on a retained determinant."""
    import ctypes as ct
    from contextlib import ExitStack

    from . import _native
    from .calculator import Calculator

    basis = request.basis
    if (
        request.role != "triples"
        or request.correlation_space_id != "all-active"
        or reference.surface_id != request.surface_id
        or reference.executed_backend != request.backend
        or reference.orbital_id != basis.orbital.identity
        or reference.jk_fit_id != basis.jk_fit.identity
        or basis.mp2_fit is None
    ):
        raise ValueError(
            "Triples requires the matching molecular reference, JKFit and correlation fit"
        )
    spins = reference._pair_spin_indices(request)
    oa, ob = (reference.occupied[k] for k in spins)
    n = len(reference.overlap)
    # A spin-conserving triple needs enough holes and particles in each spin.
    if not any(
        oa >= a and ob >= 3 - a and n - oa >= a and n - ob >= 3 - a for a in range(4)
    ):
        return 0.0, np.zeros((len(request.atoms), 3)) if forces else None
    owner = reference._native_owner
    if owner is None:
        raise RuntimeError(
            "Nonempty focal triples requires the native retained DF reference ABI"
        )
    split_fit = basis.jk_fit.identity != basis.mp2_fit.identity
    spatial = None
    if (
        not forces
        and request.backend == "cuda"
        and request.root_id == "rhf"
        and not split_fit
        and not any(
            basis.orbital.by_element[a.atomic_number].ecp_core_electrons
            for a in request.atoms
        )
    ):
        spatial = getattr(
            owner.library, "generativeqc_focal_rcc_triples_energy_v1", None
        )

    def evaluate(
        spatial_energy: ct._CFuncPtr | None,
    ) -> tuple[float, np.ndarray | None] | None:
        calc = Calculator(
            method="df-rccsd(t)" if spatial_energy is not None else "uccsd(t)",
            device="cuda" if request.backend == "cuda" else "cpu",
            basis=basis.orbital,
            auxiliary_basis=basis.mp2_fit,
            density_fitting="full_cuda",
            initial_guess=None,
            correlation_memory_budget_bytes=budget_mib << 20,
        )
        with ExitStack() as stack:
            stack.callback(calc.clear_cache)
            lib = calc._library
            if lib._handle != owner.library._handle:
                raise RuntimeError("The triples library must own the supplied reference")
            ctx, plan = ct.c_void_p(), ct.c_void_p()
            _native.check(
                lib,
                lib.generativeqc_context_create(
                    ct.byref(calc._context_descriptor()), ct.byref(ctx)
                ),
            )
            stack.callback(lib.generativeqc_context_destroy, ctx)
            geometry = (ctx, request.atoms, request.charge, request.multiplicity)
            system = calc._create_native_system(*geometry)
            stack.callback(lib.generativeqc_system_destroy, system)
            correlation_fit = calc._create_native_system(*geometry, basis.mp2_fit)
            stack.callback(lib.generativeqc_system_destroy, correlation_fit)
            descriptor = calc._method_descriptor(
                correlation_fit, systems=(request.atoms,)
            )
            if split_fit:
                jk = calc._create_native_system(*geometry, basis.jk_fit)
                stack.callback(lib.generativeqc_system_destroy, jk)
                status = lib.generativeqc_calculation_prepare_ucc_df_fits_v1(
                    ctx, system, ct.byref(descriptor), jk, ct.byref(plan)
                )
            else:
                status = lib.generativeqc_calculation_prepare(
                    ctx, system, ct.byref(descriptor), ct.byref(plan)
                )
            fallback = (
                _native.STATUS_NOT_IMPLEMENTED,
                _native.STATUS_OUT_OF_MEMORY,
                _native.STATUS_RESOURCE_LIMIT,
            )
            if spatial_energy is not None and status in fallback:
                return None
            _native.check(lib, status, context=ctx)
            stack.callback(lib.generativeqc_calculation_destroy, plan)
            if spatial_energy is not None:
                spatial_energy.argtypes = [
                    ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_double),
                ]
                spatial_energy.restype = ct.c_int
                energy = ct.c_double()
                status = spatial_energy(plan, owner.handle, ct.byref(energy))
                if status in fallback:
                    return None
                _native.check(lib, status, context=ctx)
                return energy.value, None

            native = lib.generativeqc_focal_ucc_triples_v1
            native.argtypes = [
                ct.c_void_p,
                ct.c_void_p,
                ct.c_int,
                np.ctypeslib.ndpointer(np.float64, ndim=1, flags="C_CONTIGUOUS"),
                ct.c_size_t,
            ]
            native.restype = ct.c_int
            output = np.empty(1 + (3 * len(request.atoms) if forces else 0))
            _native.check(
                lib, native(plan, owner.handle, forces, output, output.size), context=ctx
            )
        return float(output[0]), output[1:].reshape(-1, 3) if forces else None

    result = evaluate(spatial)
    if result is None:
        result = evaluate(None)
        assert result is not None
    return result
