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

TriplesTelemetry = dict[str, float | bool | int | str | tuple[tuple[int, ...], ...]]


def _record_frozen_core(
    telemetry: TriplesTelemetry, request: FocalRequest, reference: FocalReference,
) -> None:
    from ._f12_frozen_core import FROZEN_CORE_DEFINITION_VERSION

    telemetry["correlation_space"] = request.correlation_space_id
    telemetry["frozen_core_definition_version"] = FROZEN_CORE_DEFINITION_VERSION
    frame = getattr(reference, "frozen_frame", None)
    if frame is not None:
        telemetry["frozen_occupied_indices"] = tuple(
            channel.indices for channel in frame.channels(request.correlation_space_id)
        )


@dataclass(frozen=True)
class SpatialTriplesResult:
    """Result and execution telemetry from native spatial DF-RCCSD(T) / (T+)."""

    energy: float
    occupied: np.ndarray | None
    forces: np.ndarray | None
    telemetry: TriplesTelemetry


@dataclass(frozen=True)
class SpinBlockedTriplesResult:
    """Result and execution telemetry from native spin-blocked DF-UCCSD(T) / (T+) v2."""

    ccsd_energy: float
    energy: float
    occupied_alpha: np.ndarray | None
    occupied_beta: np.ndarray | None
    forces: np.ndarray | None
    telemetry: TriplesTelemetry


def _w_precision_code(raw_fp32: bool, schedule: str | None) -> int:
    if not isinstance(raw_fp32, (bool, np.bool_)):
        raise ValueError("triples_fp32_w must be a boolean")
    if schedule is None:
        return int(raw_fp32)
    if schedule == "fp32_split3" and not raw_fp32:
        return 2
    raise ValueError("Use triples_w_schedule='fp32_split3' alone, or triples_fp32_w for single-pass W")


def _counts(values: np.ndarray) -> tuple[int, ...]:
    if not np.isfinite(values).all() or any(n < 0 or n != int(n) for n in values):
        raise RuntimeError("Invalid native triples v2 work counts")
    return tuple(int(n) for n in values)


def _flags(values: np.ndarray) -> tuple[bool, ...]:
    if not all(x in (0.0, 1.0) for x in values):
        raise RuntimeError("Invalid native triples v2 execution flags")
    return tuple(bool(x) for x in values)


def _triples_execution_telemetry(
    slots: np.ndarray, *, forces: bool, fp32_requested: bool, spatial_cuda: bool, cpu: bool,
    schedule: str | None = None,
) -> TriplesTelemetry:
    """Decode measured W work. Every nonempty split3 endpoint must prove split execution."""
    required = 17 if spatial_cuda else 10
    if slots.ndim != 1 or slots.size < required or not np.isfinite(slots[:required]).all():
        raise RuntimeError("Native triples v2 returned incomplete execution telemetry")
    ccsd32, lambda32 = _flags(slots[:2])
    n32, n64 = _counts(slots[2:4])
    has_work = n32 > 0 or n64 > 0
    workspace = None
    if spatial_cuda:
        mixed, fallback, forward32, reverse32, has_reverse, split = _flags(slots[8:14])
        nforward, nreverse, workspace = _counts(slots[14:17])
        if (
            mixed != (n32 > 0)
            or forward32 != (nforward > 0)
            or reverse32 != (nreverse > 0)
            or (reverse32 and not has_reverse)
            or (has_reverse and not reverse32 and n64 == 0)
        ):
            raise RuntimeError("Inconsistent native directional W precision telemetry")
    else:
        # The original ten-slot spin-blocked engine counted only forward FP32.
        # Its host forward GEMMs round both inputs and output around a double GEMM.
        if slots.size == 12 and not np.isnan(slots[10:]).all():
            (nreverse,) = _counts(slots[10:11])
            (split,) = _flags(slots[11:12])
        elif slots.size in (10, 12) and schedule is None:
            nreverse, split = 0, False
        else:
            raise RuntimeError("Native spin-blocked triples v2 lacks split3 execution telemetry")
        nforward = n32 - nreverse
        has_reverse = forces and has_work
        fallback = False
    if (
        n32 != nforward + nreverse or nforward < 0
        or (has_reverse and not (forces and has_work))
        or (nreverse and not has_reverse)
        or (n32 and not (fp32_requested or schedule is not None))
        or split != (schedule == "fp32_split3" and has_work)
        or (split and (
            not nforward or nforward % 9 or nreverse % 9 or (has_reverse and not nreverse) or fallback
        ))
    ):
        raise RuntimeError("Inconsistent native directional W precision telemetry")

    forward = reverse = panel = observed_schedule = "n/a"
    if has_work:
        observed_schedule = "fp64"
        forward = "fp64"
        if nforward:
            observed_schedule = "fp32_split3" if split else "fp32_single"
            forward = "fp32_emulated" if cpu and not split else "fp32"
        panel = "fp64" if spatial_cuda and fallback else forward
        if has_reverse:
            reverse = "fp32" if nreverse else "fp64"
    result: TriplesTelemetry = {
        "ccsd_fp32_executed": ccsd32,
        "lambda_fp32_executed": lambda32,
        "triples_fp32_gemms": n32,
        "triples_fp64_gemms": n64,
        "triples_fp32_forward_gemms": nforward,
        "triples_fp32_reverse_gemms": nreverse,
        "ccsd_seconds": float(slots[4]),
        "triples_seconds": float(slots[5]),
        "lambda_seconds": float(slots[6]),
        "source_response_seconds": float(slots[7]),
        "triples_fp32_w": fp32_requested,
        "triples_fp32_w_executed": n32 > 0,
        "triples_fp32_w_resource_fallback": fallback,
        "triples_fp32_w_forward_executed": nforward > 0,
        "triples_fp32_w_reverse_executed": nreverse > 0,
        "forward_w_precision": forward,
        "df_panel_precision": panel,
        "df_panel_storage_precision": ("fp64" if split or cpu else panel) if has_work else "n/a",
        "requested_w_precision_schedule": (
            schedule if schedule is not None else "fp32_single" if fp32_requested else "fp64"
        ),
        "reverse_w_precision": reverse,
        "w_precision_schedule": observed_schedule,
    }
    if workspace is not None:
        result["triples_workspace_bytes"] = workspace
    return result


def _empty_triples_telemetry(fp32_requested: bool, schedule: str | None = None) -> TriplesTelemetry:
    """A dimension check returned the zero increment before either W direction ran."""
    return {
        "triples_fp32_w": fp32_requested,
        "requested_w_precision_schedule": (
            schedule if schedule is not None else "fp32_single" if fp32_requested else "fp64"
        ),
        "triples_fp32_gemms": 0,
        "triples_fp64_gemms": 0,
        "triples_fp32_w_executed": False,
        "triples_fp32_w_forward_executed": False,
        "triples_fp32_w_reverse_executed": False,
        "triples_fp32_w_resource_fallback": False,
        "w_precision_schedule": "n/a",
        "triples_fp32_forward_gemms": 0,
        "triples_fp32_reverse_gemms": 0,
        "forward_w_precision": "n/a",
        "df_panel_precision": "n/a",
        "df_panel_storage_precision": "n/a",
        "reverse_w_precision": "n/a",
    }


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


def _frame_rotation_action(provider, run, *, cuda: bool):
    """Bind the canonical C^T bar_C contraction for both spin channels."""
    from generativeqc_compiler.method.canonical_frame_response import orbital_rotation_cotangent

    from ._f12_cphf_relax_native import resolve_cphf_relax, run_orbital_rot_metric_native

    fn = resolve_cphf_relax(
        provider.shell_library, "orbital_rot_metric",
        fallback_library=provider.response_library, host_reference=not cuda,
    )
    budget = min(provider.budget_mib << 20, 512 << 20)

    def apply(coefficients, frame_bar):
        if fn is None:
            return run(orbital_rotation_cotangent(*coefficients.shape),
                       {"coefficients": coefficients, "frame_bar": frame_bar}).outputs["rotation_gradient"]
        rotation, _ = run_orbital_rot_metric_native(
            fn, coefficients, frame_bar, compute_rotation=True, compute_overlap=False,
            budget_bytes=budget,
        )
        assert rotation is not None
        return rotation

    return apply


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
    rotation = _frame_rotation_action(provider, run, cuda=cuda)
    seeds = {}
    for spin in spins:
        key = "coefficients_" + spin
        direct[key] += half_c
        rot = rotation(c_mat, direct[key])
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
    grad = one.nuclear_gradient(
        relaxed["overlap"], relaxed["hcore"], run=run, shell_library=provider.shell_library
    )
    jk_bars = {"ao_coulomb": relaxed["three_center"], "metric": relaxed["metric"]}
    grad += provider._gradient(pair_request, jk, jk_bars, coulomb_only=True)
    grad += corr_gradient.reshape(-1, 3)
    return -grad


def relax_spin_blocked_uhf_force_sources(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    reference: FocalReference,
    bar_f_alpha: np.ndarray,
    bar_f_beta: np.ndarray,
    bar_c_alpha: np.ndarray,
    bar_c_beta: np.ndarray,
    corr_gradient: np.ndarray,
) -> np.ndarray:
    """Pull spin-blocked UHF/RHF triples force sources through split-fit/ECP CPHF.

    Accepts independent alpha/beta active MO Fock cotangents (bar_f_alpha, bar_f_beta),
    MO coefficient cotangents (bar_c_alpha, bar_c_beta), and explicit DF correlation
    nuclear gradient (corr_gradient). Returns Cartesian nuclear forces of shape
    (natoms, 3) (minus energy derivative in Eh/bohr).
    """
    from functools import partial

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
            raise RuntimeError("Spin-blocked triples CPHF program ran on another backend")
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
        raise ValueError("Spin-blocked triples one-electron integrals differ from root")
    jk = provider._panel(pair_request, "jk_fit", orbital_only=True)
    n = one.obs_ao_count
    spin_indices = reference._pair_spin_indices(request)
    occs = tuple(reference.occupied[k] for k in spin_indices)
    spins = ("alpha", "beta")
    ao = {
        "coefficients_alpha": reference.coefficients[spin_indices[0]],
        "coefficients_beta": reference.coefficients[spin_indices[1]],
        "hcore_ao": reference.hcore,
        "ao_coulomb": jk.data["ao_coulomb"],
        "metric/coulomb": jk.data["metric"],
        "metric_whitener": jk.whitener,
    }
    space = (occs[0], occs[1], n, n, len(jk.whitener))
    fspace = (occs[0], occs[1], n, len(jk.whitener))
    projected, _ = _orbital_sources(
        space, ao, run, None, projected=None, role="ordinary_fock"
    )
    values = {
        k.removeprefix("source/"): v
        for k, v in projected.items()
        if k.startswith("source/")
    }
    focks = spin_fock(fspace, values, run)
    for k, s in enumerate(spins):
        eps = reference.orbital_energies[spin_indices[k]]
        if not np.allclose(focks["fock_" + s], np.diag(eps), atol=2e-8, rtol=0):
            raise ValueError(f"JKFit {s} Fock differs from stationary canonical reference frame")

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

    direct = fock_pullback({"fock_alpha": bar_f_alpha, "fock_beta": bar_f_beta})
    bar_c_by_spin = (bar_c_alpha, bar_c_beta)
    rotation = _frame_rotation_action(provider, run, cuda=cuda)
    seeds: dict[str, np.ndarray] = {}
    for k, spin in enumerate(spins):
        idx = spin_indices[k]
        key = "coefficients_" + spin
        direct[key] += bar_c_by_spin[k]
        rot = rotation(reference.coefficients[idx], direct[key])
        eps = reference.orbital_energies[idx]
        seeds["fock_" + spin] = (
            canonical_frame_seed(
                rot,
                eps,
                occs[k],
                library=provider.canonical_library,
                maximum_bytes=provider.budget_mib << 20,
            )[0]
            if cuda
            else _canonical_occupied_virtual_blocks(rot, eps, occs[k])
        )
    canonical = fock_pullback(seeds)
    for key in direct:
        direct[key] += canonical[key]

    coefficient_bar = np.zeros_like(reference.coefficients)
    for k, spin in enumerate(spins):
        coefficient_bar[spin_indices[k]] += direct["coefficients_" + spin]
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
    grad = one.nuclear_gradient(
        relaxed["overlap"], relaxed["hcore"], run=run, shell_library=provider.shell_library
    )
    jk_bars = {"ao_coulomb": relaxed["three_center"], "metric": relaxed["metric"]}
    grad += provider._gradient(pair_request, jk, jk_bars, coulomb_only=True)
    grad += corr_gradient.reshape(-1, 3)
    return -grad
