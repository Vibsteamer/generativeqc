"""Molecular conventional (T) increments on focal references."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._f12_focal_molecular import MolecularFocalProvider
    from ._f12_focal_reference import FocalReference
    from ._f12_focal_tplus_sources import FocalOccupiedTriples
    from .experimental.f12_focal import FocalEvaluation, FocalRequest


PRECONVERGENCE_CODES: dict[str, int] = {
    "legacy": 0,
    "fp64": 1,
    "mixed": 2,
    "auto": 3,
}


@dataclass(frozen=True)
class SpatialTriplesResult:
    """Result and execution telemetry from native spatial DF-RCCSD(T) / (T+)."""

    energy: float
    occupied: np.ndarray | None
    forces: np.ndarray | None
    telemetry: dict[str, float | bool | int]


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


def relax_spatial_rhf_force_sources(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    reference: FocalReference,
    bar_f: np.ndarray,
    bar_c: np.ndarray,
    corr_gradient: np.ndarray,
) -> np.ndarray:
    """Pull spatial RHF triples (bar_f, bar_c, corr_gradient) through split-fit/ECP CPHF.

    Returns Cartesian nuclear forces (shape (natoms, 3), minus energy derivative in Eh/bohr).
    """
    from functools import partial

    from generativeqc_compiler.method.canonical_frame_response import (
        orbital_rotation_cotangent,
    )
    from generativeqc_compiler.tensor import execute

    from ._canonical_frame import canonical_frame_seed
    from ._f12_focal_increment import _canonical_occupied_virtual_blocks
    from ._f12_focal_one_electron import one_electron_sources
    from ._f12_focal_singles_response import relaxed_reference_weights
    from ._f12_spin_ao import _orbital_sources
    from ._f12_spin_fock import spin_fock, spin_fock_pullback

    cuda = request.backend == "cuda"
    runner = (
        provider.run if cuda else partial(execute, max_bytes=provider.budget_mib << 20)
    )
    assert runner is not None

    def run(program, inputs):
        out = runner(program, inputs)
        if out.backend != (
            "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
        ):
            raise RuntimeError("Spatial triples CPHF program ran on another backend")
        return out

    pair_request = replace(request, role="mp2_f12_correlation")
    one = one_electron_sources(
        pair_request,
        library=provider.one_electron_library or provider.shell_library,
        orbital_only=True,
        derivatives=True,
        budget_mib=provider.budget_mib,
    )
    if not (
        np.allclose(one.overlap, reference.overlap, atol=2e-8, rtol=0)
        and np.allclose(one.hcore, reference.hcore, atol=2e-8, rtol=0)
    ):
        raise ValueError("Spatial triples one-electron integrals differ from root")
    jk = provider._panel(pair_request, "jk_fit", orbital_only=True)
    n = one.obs_ao_count
    o_full = reference.occupied[0]
    eps = reference.orbital_energies[0]
    c_mat = reference.coefficients[0]
    spins = ("alpha", "beta")
    ao = {
        "coefficients_alpha": c_mat,
        "coefficients_beta": c_mat,
        "hcore_ao": reference.hcore,
        "ao_coulomb": jk.data["ao_coulomb"],
        "metric/coulomb": jk.data["metric"],
        "metric_whitener": jk.whitener,
    }
    space = (o_full, o_full, n, n, len(jk.whitener))
    fspace = (o_full, o_full, n, len(jk.whitener))
    projected, _ = _orbital_sources(
        space, ao, run, None, projected=None, role="ordinary_fock"
    )
    values = {
        k.removeprefix("source/"): v
        for k, v in projected.items()
        if k.startswith("source/")
    }
    focks = spin_fock(fspace, values, run)
    for s in spins:
        if not np.allclose(focks["fock_" + s], np.diag(eps), atol=2e-8, rtol=0):
            raise ValueError("JKFit Fock differs from stationary canonical RHF frame")

    def fock_pullback(seeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        seed = spin_fock_pullback(fspace, values, seeds, run)
        _, bars = _orbital_sources(
            space,
            ao,
            run,
            {"source/" + k: v for k, v in seed.items()},
            projected=projected,
            role="ordinary_fock",
        )
        assert bars is not None
        return bars

    half_f = 0.5 * bar_f
    half_c = 0.5 * bar_c
    direct = fock_pullback({"fock_alpha": half_f, "fock_beta": half_f})
    rotation = orbital_rotation_cotangent(n, n)
    seeds = {}
    for spin in spins:
        key = "coefficients_" + spin
        direct[key] += half_c
        rot = run(
            rotation, {"coefficients": c_mat, "frame_bar": direct[key]}
        ).outputs["rotation_gradient"]
        seeds["fock_" + spin] = (
            canonical_frame_seed(
                rot,
                eps,
                o_full,
                library=provider.canonical_library,
                maximum_bytes=provider.budget_mib << 20,
            )[0]
            if cuda
            else _canonical_occupied_virtual_blocks(rot, eps, o_full)
        )
    canonical = fock_pullback(seeds)
    for key in direct:
        direct[key] += canonical[key]
    coefficient_bar = np.zeros_like(reference.coefficients)
    coefficient_bar[0] = direct["coefficients_alpha"] + direct["coefficients_beta"]
    relaxed = relaxed_reference_weights(
        reference,
        values,
        jk.whitener,
        {
            "hcore": direct["hcore_ao"],
            "three_center": direct["ao_coulomb"],
            "metric": direct["metric/coulomb"],
            "obs_coefficients": coefficient_bar,
        },
        np.zeros_like(reference.overlap),
        run,
        provider.response_library,
        provider.budget_mib,
    )
    grad = one.nuclear_gradient(relaxed["overlap"], relaxed["hcore"], run=run)
    jk_bars = {"ao_coulomb": relaxed["three_center"], "metric": relaxed["metric"]}
    grad += provider._gradient(pair_request, jk, jk_bars, coulomb_only=True)
    grad += corr_gradient.reshape(-1, 3)
    return -grad


def spatial_rhf_triples_v2(
    request: FocalRequest,
    reference: FocalReference,
    *,
    budget_mib: int,
    forces: bool = False,
    occupied_partition: bool = False,
    occupied_weights: np.ndarray | None = None,
    preconvergence: str = "auto",
    triples_fp32_w: bool = False,
    provider: MolecularFocalProvider | None = None,
) -> SpatialTriplesResult:
    """Invoke generativeqc_focal_rcc_triples_v2 for fast spatial CUDA RHF (T) / (T+)."""
    import ctypes as ct
    from contextlib import ExitStack

    from . import _native
    from ._f12_frozen_core import resolve_frozen_occupied
    from .calculator import Calculator

    if preconvergence not in PRECONVERGENCE_CODES:
        raise ValueError(
            f"Unsupported preconvergence {preconvergence!r}; expected one of {tuple(PRECONVERGENCE_CODES)}"
        )
    basis = request.basis
    if (
        request.root_id != "rhf"
        or request.backend != "cuda"
        or basis.mp2_fit is None
        or reference.surface_id != request.surface_id
        or reference.executed_backend != request.backend
        or reference.orbital_id != basis.orbital.identity
        or reference.jk_fit_id != basis.jk_fit.identity
    ):
        raise ValueError(
            "Spatial RHF triples v2 requires matching CUDA RHF reference and correlation fit"
        )
    if forces and provider is None:
        raise ValueError("Spatial RHF triples forces require the owning MolecularFocalProvider")
    o_full = reference.occupied[0]
    n = len(reference.overlap)
    frozen = resolve_frozen_occupied(request, (o_full, o_full))[0]
    o_act = o_full - frozen
    n_vir = n - o_full
    if o_act < 2 or n_vir < 2:
        occ_out = np.zeros(o_act, dtype=np.float64) if occupied_partition else None
        frc_out = np.zeros((len(request.atoms), 3), dtype=np.float64) if forces else None
        return SpatialTriplesResult(0.0, occ_out, frc_out, {})

    owner = reference._native_owner
    if owner is None:
        raise RuntimeError("Spatial RHF triples v2 requires the native retained DF reference ABI")
    fn = getattr(owner.library, "generativeqc_focal_rcc_triples_v2", None)
    if fn is None:
        raise NotImplementedError("Native generativeqc_focal_rcc_triples_v2 ABI is unavailable")

    weights_arr = None
    if occupied_weights is not None:
        weights_arr = np.ascontiguousarray(occupied_weights, dtype=np.float64)
        if weights_arr.shape != (o_act,) or not np.isfinite(weights_arr).all():
            raise ValueError(f"occupied_weights must be finite float64 of length {o_act}")

    calc = Calculator(
        method="df-rccsd(t)",
        device="cuda",
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
        descriptor = calc._method_descriptor(correlation_fit, systems=(request.atoms,))
        status = lib.generativeqc_calculation_prepare(
            ctx, system, ct.byref(descriptor), ct.byref(plan)
        )
        _native.check(lib, status, context=ctx)
        stack.callback(lib.generativeqc_calculation_destroy, plan)

        fn.restype = ct.c_int
        fn.argtypes = [
            ct.c_void_p, ct.c_void_p,
            ct.c_int, ct.c_int, ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.c_uint32, ct.c_int,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
        ]
        energy_out = ct.c_double()
        occ_buf = np.empty(o_act, dtype=np.float64) if occupied_partition else None
        grad_buf = np.empty(3 * len(request.atoms), dtype=np.float64) if forces else None
        bar_f_buf = np.empty(n * n, dtype=np.float64) if forces else None
        bar_c_buf = np.empty(n * n, dtype=np.float64) if forces else None
        tel_buf = np.zeros(8, dtype=np.float64)

        def ptr(arr: np.ndarray | None):
            return arr.ctypes.data_as(ct.POINTER(ct.c_double)) if arr is not None else None

        status = fn(
            plan,
            owner.handle,
            1 if forces else 0,
            1 if occupied_partition else 0,
            frozen,
            ptr(weights_arr),
            0 if weights_arr is None else weights_arr.size,
            PRECONVERGENCE_CODES[preconvergence],
            1 if triples_fp32_w else 0,
            ct.byref(energy_out),
            ptr(occ_buf),
            0 if occ_buf is None else occ_buf.size,
            ptr(grad_buf),
            0 if grad_buf is None else grad_buf.size,
            ptr(bar_f_buf),
            0 if bar_f_buf is None else bar_f_buf.size,
            ptr(bar_c_buf),
            0 if bar_c_buf is None else bar_c_buf.size,
            ptr(tel_buf),
            tel_buf.size,
        )
        _native.check(lib, status, context=ctx)

    telemetry: dict[str, float | bool | int] = {
        "ccsd_fp32_executed": bool(tel_buf[0] != 0.0),
        "lambda_fp32_executed": bool(tel_buf[1] != 0.0),
        "triples_fp32_gemms": int(tel_buf[2]),
        "triples_fp64_gemms": int(tel_buf[3]),
        "ccsd_seconds": float(tel_buf[4]),
        "triples_seconds": float(tel_buf[5]),
        "lambda_seconds": float(tel_buf[6]),
        "source_response_seconds": float(tel_buf[7]),
        "frozen_occupied": frozen,
        "preconvergence": preconvergence,
        "triples_fp32_w": triples_fp32_w,
    }
    forces_out = None
    if forces:
        assert provider is not None and grad_buf is not None and bar_f_buf is not None and bar_c_buf is not None
        forces_out = relax_spatial_rhf_force_sources(
            provider,
            request,
            reference,
            bar_f_buf.reshape(n, n),
            bar_c_buf.reshape(n, n),
            grad_buf,
        )
    return SpatialTriplesResult(float(energy_out.value), occ_buf, forces_out, telemetry)


def molecular_triples_on_reference(
    request: FocalRequest,
    reference: FocalReference,
    *,
    budget_mib: int,
    forces: bool,
    provider: MolecularFocalProvider | None = None,
    preconvergence: str = "auto",
    triples_fp32_w: bool = False,
) -> tuple[float, np.ndarray | None]:
    """Use the native conventional CCSD and triples owner on a retained determinant."""
    if (
        request.backend == "cuda"
        and request.root_id == "rhf"
        and reference._native_owner is not None
        and getattr(reference._native_owner.library, "generativeqc_focal_rcc_triples_v2", None) is not None
        and (not forces or provider is not None)
    ):
        res = spatial_rhf_triples_v2(
            request,
            reference,
            budget_mib=budget_mib,
            forces=forces,
            occupied_partition=False,
            preconvergence=preconvergence,
            triples_fp32_w=triples_fp32_w,
            provider=provider,
        )
        return res.energy, res.forces
    return _retained_conventional_triples(
        request, reference, budget_mib=budget_mib, forces=forces, occupied_partition=False
    )


def molecular_occupied_triples_on_reference(
    request: FocalRequest,
    reference: FocalReference,
    *,
    budget_mib: int,
    preconvergence: str = "auto",
    triples_fp32_w: bool = False,
) -> tuple[float, FocalOccupiedTriples]:
    """Return the conventional scalar and its spatial occupied partition on the (T+) frame."""
    from ._f12_focal_tplus_sources import FocalOccupiedTriples

    if (
        request.role != "triples_plus"
        or request.basis.basis_id != reference.basis_id
        or request.basis.mp2_fit is None
    ):
        raise ValueError("Occupied triples requires the matching (T+) basis and reference")
    if (
        request.backend == "cuda"
        and request.root_id == "rhf"
        and reference._native_owner is not None
        and getattr(reference._native_owner.library, "generativeqc_focal_rcc_triples_v2", None) is not None
    ):
        res = spatial_rhf_triples_v2(
            replace(request, role="triples"),
            reference,
            budget_mib=budget_mib,
            forces=False,
            occupied_partition=True,
            preconvergence=preconvergence,
            triples_fp32_w=triples_fp32_w,
        )
        assert res.occupied is not None
        return res.energy, FocalOccupiedTriples(
            reference, request.correlation_space_id, request.basis.mp2_fit.identity, res.occupied
        )
    scalar, values = _retained_conventional_triples(
        replace(request, role="triples"), reference,
        budget_mib=budget_mib, forces=False, occupied_partition=True,
    )
    assert values is not None
    return scalar, FocalOccupiedTriples(
        reference, request.correlation_space_id, request.basis.mp2_fit.identity, values
    )


def _retained_conventional_triples(
    request: FocalRequest,
    reference: FocalReference,
    *,
    budget_mib: int,
    forces: bool,
    occupied_partition: bool,
) -> tuple[float, np.ndarray | None]:
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
    split_fit = basis.jk_fit.identity != basis.mp2_fit.identity
    spatial_eligible = (
        not forces
        and request.backend == "cuda"
        and request.root_id == "rhf"
        and not split_fit
        and not any(
            basis.orbital.by_element[a.atomic_number].ecp_core_electrons
            for a in request.atoms
        )
    )
    if occupied_partition and not spatial_eligible:
        raise NotImplementedError(
            "Occupied triples requires all-electron CUDA RHF with one common fit"
        )
    # A spin-conserving triple needs enough holes and particles in each spin.
    if not any(
        oa >= a and ob >= 3 - a and n - oa >= a and n - ob >= 3 - a for a in range(4)
    ):
        if occupied_partition:
            return 0.0, np.zeros(oa)
        return 0.0, np.zeros((len(request.atoms), 3)) if forces else None
    owner = reference._native_owner
    if owner is None:
        raise RuntimeError(
            "Nonempty focal triples requires the native retained DF reference ABI"
        )
    spatial = None
    if spatial_eligible:
        symbol = (
            "generativeqc_focal_rcc_triples_occupied_v1"
            if occupied_partition else "generativeqc_focal_rcc_triples_energy_v1"
        )
        spatial = getattr(owner.library, symbol, None)
        if occupied_partition and spatial is None:
            raise NotImplementedError("Native occupied triples ABI is unavailable")

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
            if spatial_energy is not None and not occupied_partition and status in fallback:
                return None
            _native.check(lib, status, context=ctx)
            stack.callback(lib.generativeqc_calculation_destroy, plan)
            if spatial_energy is not None:
                spatial_energy.restype = ct.c_int
                if occupied_partition:
                    spatial_energy.argtypes = [
                        ct.c_void_p, ct.c_void_p,
                        np.ctypeslib.ndpointer(np.float64, ndim=1, flags="C_CONTIGUOUS"),
                        ct.c_size_t,
                    ]
                    output = np.empty(oa + 1)
                    status = spatial_energy(plan, owner.handle, output, output.size)
                    _native.check(lib, status, context=ctx)
                    return float(output[0]), output[1:]
                spatial_energy.argtypes = [
                    ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_double),
                ]
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
