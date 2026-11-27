"""Spatial GPU-resident DF-CUDA RHF CCSD-F12b primal, Lambda, and exact 3N forces."""

from __future__ import annotations

import ctypes as ct
import math
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from ._f12_focal_ucc_triples import PRECONVERGENCE_CODES, relax_spatial_rhf_force_sources

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from generativeqc_compiler.tensor import Execution, Program

    from ._f12_focal_molecular import MolecularFocalProvider
    from ._f12_focal_reference import FocalReference
    from .experimental.f12_focal import FocalRequest


@dataclass(frozen=True)
class SpatialCcsdF12bResult:
    """Result and execution telemetry from native spatial DF-CUDA RHF CCSD-F12b."""

    correlation_energy: float
    cabs_rank: int
    t1: np.ndarray | None
    t2: np.ndarray | None
    lambda1: np.ndarray | None
    lambda2: np.ndarray | None
    ordinary_forces: np.ndarray | None
    fixed_forces: np.ndarray | None
    telemetry: dict[str, float | bool | int | str | None]


def _make_checked_runner(provider: MolecularFocalProvider, cuda: bool) -> Callable:
    from generativeqc_compiler.tensor import execute

    runner = (
        provider.run if cuda else partial(execute, max_bytes=provider.budget_mib << 20)
    )
    assert runner is not None
    expected_backend = (
        "cuda-fp64-ordinary-stream" if cuda else "numpy-cpu-interpreter"
    )

    def checked(graph: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        out = runner(graph, inputs)
        if out.backend != expected_backend:
            raise RuntimeError("Spatial CCSD-F12b tensor program executed on another backend")
        return out

    if hasattr(runner, "batch"):

        @contextmanager
        def checked_batch(graph: Program):
            with runner.batch(graph) as exec_one:

                def wrapped(inputs: Mapping[str, np.ndarray]) -> Execution:
                    out = exec_one(inputs)
                    if out.backend != expected_backend:
                        raise RuntimeError(
                            "Spatial CCSD-F12b tensor program executed on another backend"
                        )
                    return out

                yield wrapped

        checked.batch = checked_batch  # type: ignore[attr-defined]
    return checked


def spatial_f12_lambda_to_uccsd_blocks(
    lambda1: np.ndarray, lambda2: np.ndarray
) -> dict[str, np.ndarray]:
    """Map spatial closed-shell (lambda1, lambda2) to physical spin-resolved UCCSD Lambda blocks."""
    l1 = 0.5 * lambda1
    l2_swap = lambda2.swapaxes(2, 3)
    l2ab = (2.0 * lambda2 + l2_swap) / 3.0
    l2aa = (lambda2 - l2_swap) / 3.0
    return {
        "l1a": l1.copy(),
        "l1b": l1.copy(),
        "l2aa": l2aa,
        "l2ab": l2ab,
        "l2bb": l2aa.copy(),
    }


def spatial_f12_fixed_source_cotangents(
    t1: np.ndarray,
    t2: np.ndarray,
    lambda1: np.ndarray,
    lambda2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the 17 exact spin-resolved fixed-source cotangents from spatial (t1, t2, l1, l2)."""
    o, _v = t1.shape
    l1 = 0.5 * lambda1
    t2_swap = t2.swapaxes(2, 3)
    l2_swap = lambda2.swapaxes(2, 3)
    l2_ab = (2.0 * lambda2 + l2_swap) / 3.0
    l2_aa = (lambda2 - l2_swap) / 3.0
    t2_aa = t2 - t2_swap
    t1t1 = t1[:, None, :, None] * t1[None, :, None, :]

    c_aa = 0.25 * (t2_aa + l2_aa)
    c_ab = 0.25 * t2 + l2_ab
    c_ba = 0.25 * t2
    c_abba = -0.25 * t2_swap

    wvv_aa = c_aa + 0.5 * t1t1
    wvv_ab = c_ab + 0.5 * t1t1
    wvv_ba = c_ba + 0.5 * t1t1
    wvv_abba = c_abba

    v1 = t1 + l1
    eye_o = np.eye(o, dtype=np.float64)
    v1_diag = np.einsum("ia,jk->ijak", v1, eye_o)
    lt_aa = np.einsum("ijab,kb->ijak", l2_aa, t1)
    lt_ab = np.einsum("ijab,kb->ijak", l2_ab, t1)
    lt_abba = np.einsum("ijba,kb->ijak", l2_ab, t1)

    wvo_aa = v1_diag - 0.5 * lt_aa
    wvo_ab = v1_diag - lt_ab
    wvo_abba = lt_abba
    wvo_baba = v1_diag.copy()

    return {
        "fixed_geminal_energy": np.asarray(1.0, dtype=np.float64),
        "c_f12_aaaa": c_aa,
        "c_f12_bbbb": c_aa.copy(),
        "c_f12_abab": c_ab,
        "c_f12_baba": c_ba,
        "c_f12_abba": c_abba,
        "c_f12_baab": c_abba.copy(),
        "w_vv_aaaa": wvv_aa,
        "w_vv_bbbb": wvv_aa.copy(),
        "w_vv_abab": wvv_ab,
        "w_vv_baba": wvv_ba,
        "w_vv_abba": wvv_abba,
        "w_vv_baab": wvv_abba.copy(),
        "w_vo_aaaa": wvo_aa,
        "w_vo_bbbb": wvo_aa.copy(),
        "w_vo_abab": wvo_ab,
        "w_vo_abba": wvo_abba,
        "w_vo_baba": wvo_baba,
    }


def _build_spatial_rhf_fixed_sources(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    reference: FocalReference,
    *,
    expected_cabs_rank: int | None,
    derivatives: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    """Compute compact spatial (w_vo, w_vv, c_f12, fixed_geminal_energy, cabs_rank, frozen)."""
    from ._f12_cc_fock_pullback import extended_fock_response
    from ._f12_cc_spin_sources import spatial_rhf_cc_f12_fixed_sources
    from ._f12_focal_frame import canonical_pair_frame
    from ._f12_frozen_core import resolve_frozen_occupied
    from ._f12_spin_ao import _orbital_sources

    cuda = request.backend == "cuda"
    checked = _make_checked_runner(provider, cuda)
    one, (jk, _mp2, corr) = provider._cached_pair_sources(
        request, derivatives=derivatives, budget=provider.budget_mib
    )
    space, ao, panel, proj = canonical_pair_frame(
        request,
        reference,
        one,
        jk.joint(request, explicit=False),
        corr.joint(request, explicit=True),
        expected_cabs_rank=expected_cabs_rank,
        relative_cutoff=provider.relative_cutoff,
        library=provider.response_library,
    )
    oa, ob, obs, _ext, _aux = space
    if oa != ob:
        raise ValueError("Spatial RHF CCSD-F12b requires equal alpha and beta occupations")
    # Compute and release JKFit Fock before materializing the F12 pair projection.
    fock = extended_fock_response(
        space, ao, panel, checked, cuda=cuda, library=provider.response_library,
        native_library=provider.shell_library, host_reference=not cuda,
    )[0]
    if not np.allclose(
        fock["fock_alpha"][:obs, :obs],
        np.diag(reference.orbital_energies[0]),
        atol=2e-8,
        rtol=0,
    ):
        raise ValueError("Extended RHF Fock disagrees with the canonical DF reference")
    data, _ = _orbital_sources(
        space, ao, checked, None, projected=None, role="pairs", obs_coulomb=True,
        shell_library=provider.shell_library, host_reference=not cuda,
    )
    frozen = resolve_frozen_occupied(request, (oa, ob))[0]
    w_vo, w_vv, c_f12, fixed_energy = spatial_rhf_cc_f12_fixed_sources(
        space,
        data,
        fock,
        corr.whitener,
        checked,
        frozen_occupied=frozen,
        native_library=provider.shell_library,
        host_reference=(request.backend == "cpu_reference"),
    )
    return w_vo, w_vv, c_f12, fixed_energy, proj.rank, frozen


def spatial_rhf_fixed_cc_forces(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    reference: FocalReference,
    *,
    cabs_rank: int,
    fixed_geminal_energy: float,
    t1: np.ndarray,
    t2: np.ndarray,
    lambda1: np.ndarray,
    lambda2: np.ndarray,
    frozen_occupied: int,
) -> np.ndarray:
    """Pull stationary spatial RHF fixed-cusp F12b cotangents (V+B-X, W, C) to nuclear forces.

    Uses Track 6's GPU-resident cuBLAS pair pullbacks (with automatic TensorIR fallback),
    exact closed-shell aa==bb spin symmetry, a single OBS-only Coulomb projection, and
    streaming GPU spherical F12 AO panel contraction when corr.streaming is active.
    """
    from ._f12_cc_c_nuclear import _accumulate_c_mo_pullback
    from ._f12_cc_fixed_sources_native import coulomb_fit_forward, coulomb_fit_reverse
    from ._f12_cc_fock_pullback import extended_fock_response
    from ._f12_cc_scalar_nuclear import _accumulate_scalar_mo_pullback
    from ._f12_cc_w_nuclear import _accumulate_w_mo_pullback
    from ._f12_focal_frame import canonical_pair_frame
    from ._f12_focal_increment import relaxed_f12_source_weights
    from ._f12_focal_integrals import contract_single_channel
    from ._f12_focal_molecular import _AO_NAME_TO_OP, _METRIC_KEY_TO_OP
    from ._f12_ram_hygiene import disclose_host_assisted_stage
    from ._f12_spin_ao import _orbital_sources

    cuda = request.backend == "cuda"
    checked = _make_checked_runner(provider, cuda)
    one, (jk, _mp2, corr) = provider._cached_pair_sources(
        request, derivatives=True, budget=provider.budget_mib
    )
    jk_sources = jk.joint(request, explicit=False)
    space, ao, panel, proj = canonical_pair_frame(
        request,
        reference,
        one,
        jk_sources,
        corr.joint(request, explicit=True),
        expected_cabs_rank=cabs_rank,
        relative_cutoff=provider.relative_cutoff,
        library=provider.response_library,
    )
    oa, ob, obs, ext, aux = space
    if oa != ob:
        raise ValueError("Spatial RHF fixed CC forces require closed-shell RHF")
    o_act = oa - frozen_occupied
    n_vir = obs - oa
    if t1.shape != (o_act, n_vir) or t2.shape != (o_act, o_act, n_vir, n_vir):
        raise ValueError("Spatial RHF fixed CC forces amplitude shape mismatch")
    if lambda1.shape != t1.shape or lambda2.shape != t2.shape:
        raise ValueError("Spatial RHF fixed CC forces Lambda shape mismatch")

    fock, fock_pullback = extended_fock_response(
        space, ao, panel, checked, cuda=cuda, library=provider.response_library,
        native_library=provider.shell_library, host_reference=not cuda,
    )
    data, _ = _orbital_sources(
        space,
        ao,
        checked,
        None,
        projected=None,
        role="pairs",
        obs_coulomb=True,
        skip_coulomb_fit_solve=True,
        shell_library=provider.shell_library, host_reference=not cuda,
    )

    host_ref = request.backend == "cpu_reference"
    whitener = ao["metric_whitener"]
    fit_obs_alpha = coulomb_fit_forward(
        whitener,
        data["source/raw_coulomb_alpha"],
        native_library=provider.shell_library,
        host_reference=host_ref,
        run=checked,
    )
    fit_obs = {"alpha": fit_obs_alpha, "beta": fit_obs_alpha}
    if oa:
        data["source/fit_alpha"] = fit_obs_alpha[:, :oa, :]
        data["source/fit_beta"] = data["source/fit_alpha"]

    dbar = {k: np.zeros_like(v) for k, v in data.items() if k != "space"}
    fit_obs_bar = {
        "alpha": np.zeros_like(fit_obs_alpha),
        "beta": np.zeros_like(fit_obs_alpha),
    }
    fbar = {k: np.zeros_like(v) for k, v in fock.items()}

    # Build stationary closed-shell spin-orbital W and C cotangent blocks directly from
    # spatial (t1, t2, lambda1, lambda2) so Track 6 GPU cuBLAS pair pullbacks execute natively.
    disclose_host_assisted_stage(
        "rhf_fixed_cc_spin_cotangents",
        requested_backend=request.backend,
        detail="Host FP64 contractions of spatial T and Lambda construct the W and C response weights.",
    )
    v1 = t1 + 0.5 * lambda1
    t1_outer = np.einsum("ia,jb->ijab", t1, t1)
    l2_ab = (2.0 * lambda2 + lambda2.swapaxes(2, 3)) / 3.0
    c_ab = np.ascontiguousarray(t2 + l2_ab)
    w_vv_ab = np.ascontiguousarray(c_ab + t1_outer)
    w_vo_ab = -np.einsum("ijab,kb->ijak", l2_ab, t1)
    w_vo_ji = -np.einsum("ijab,ka->ijbk", l2_ab, t1)
    for k in range(o_act):
        w_vo_ab[:, k, :, k] += v1
        w_vo_ji[k, :, :, k] += v1
    w_weights: dict[str, np.ndarray] = {
        "w_vo_abab": np.ascontiguousarray(w_vo_ab),
        "w_vo_abba": np.ascontiguousarray(-w_vo_ji),
        "w_vv_abab": w_vv_ab,
    }
    c_weights: dict[str, np.ndarray] = {"c_f12_abab": c_ab}
    if o_act >= 2:
        t2_aa = t2 - t2.swapaxes(2, 3)
        l2_aa = (lambda2 - lambda2.swapaxes(2, 3)) / 3.0
        c_seed_aa = 0.5 * (t2_aa + l2_aa)
        c_half_aa = np.ascontiguousarray(0.5 * c_seed_aa)
        w_vv_half_aa = np.ascontiguousarray(
            0.5 * (c_seed_aa + 0.5 * (t1_outer - t1_outer.swapaxes(0, 1)))
        )
        w_vo_seed_aa = -np.einsum("ijab,kb->ijak", l2_aa, t1)
        for k in range(o_act):
            w_vo_seed_aa[:, k, :, k] += v1
            w_vo_seed_aa[k, :, :, k] -= v1
        w_vo_half_aa = np.ascontiguousarray(0.5 * w_vo_seed_aa)
        w_weights["w_vo_aaaa"] = w_vo_half_aa
        w_weights["w_vo_bbbb"] = w_vo_half_aa
        w_weights["w_vv_aaaa"] = w_vv_half_aa
        w_weights["w_vv_bbbb"] = w_vv_half_aa
        c_weights["c_f12_aaaa"] = c_half_aa
        c_weights["c_f12_bbbb"] = c_half_aa

    frozen_pair = (frozen_occupied, frozen_occupied)
    e_v, e_b, e_x = _accumulate_scalar_mo_pullback(
        space,
        data,
        fock,
        np.asarray(1.0, dtype=np.float64),
        checked,
        dbar,
        fbar,
        frozen_occupied=frozen_pair,
        native_library=provider.shell_library,
        host_reference=host_ref,
        budget_bytes=provider.budget_mib << 20,
    )
    energy = e_v + e_b - e_x
    if abs(energy - float(fixed_geminal_energy)) > 2e-9:
        raise ValueError(
            f"Spatial RHF V+B-X energy {energy} differs from fixed source {fixed_geminal_energy}"
        )

    _accumulate_w_mo_pullback(
        space,
        data,
        fit_obs,
        w_weights,
        checked,
        dbar,
        fit_obs_bar,
        frozen_occupied=frozen_pair,
        native_library=provider.shell_library,
        host_reference=host_ref,
        budget_bytes=provider.budget_mib << 20,
    )
    _accumulate_c_mo_pullback(
        space,
        data,
        fock,
        c_weights,
        checked,
        dbar,
        fbar,
        frozen_occupied=frozen_pair,
        native_library=provider.shell_library,
        host_reference=host_ref,
        budget_bytes=provider.budget_mib << 20,
    )
    del w_weights, c_weights

    # Fold occupied fit cotangents into full obs fit cotangents and apply Coulomb metric adjoint once.
    if oa:
        fit_obs_bar["alpha"][:, :oa, :] += dbar["source/fit_alpha"]
        fit_obs_bar["beta"][:, :oa, :] += dbar["source/fit_beta"]
    extra_metric_coulomb = np.zeros_like(ao["metric/coulomb"])
    if np.array_equal(fit_obs_bar["alpha"], fit_obs_bar["beta"]):
        tc_bar, m_bar = coulomb_fit_reverse(
            whitener,
            data["source/raw_coulomb_alpha"],
            fit_obs_bar["alpha"],
            fit_precomputed=fit_obs_alpha,
            native_library=provider.shell_library,
            host_reference=host_ref,
            run=checked,
        )
        dbar["source/raw_coulomb_alpha"] += tc_bar
        dbar["source/raw_coulomb_beta"] += tc_bar
        extra_metric_coulomb += 2.0 * m_bar
        del tc_bar, m_bar
    else:
        for spin in ("alpha", "beta"):
            tc_bar, m_bar = coulomb_fit_reverse(
                whitener,
                data["source/raw_coulomb_" + spin],
                fit_obs_bar[spin],
                fit_precomputed=fit_obs_alpha,
                native_library=provider.shell_library,
                host_reference=host_ref,
                run=checked,
            )
            dbar["source/raw_coulomb_" + spin] += tc_bar
            extra_metric_coulomb += m_bar
            del tc_bar, m_bar
    del fit_obs, fit_obs_bar, fit_obs_alpha

    streamed_grad = np.zeros((len(request.atoms), 3), dtype=np.float64)

    def _f12_ao_sink(ao_name: str, bar_ao_mat: np.ndarray) -> None:
        streamed_grad[:] += contract_single_channel(
            corr.layout,
            "ao",
            _AO_NAME_TO_OP[ao_name],
            bar_ao_mat,
            provider.shell_library,
            host_reference=host_ref,
            budget_mib=provider.budget_mib,
            slater_normalization=provider.slater_normalization,
        )

    _, bars = _orbital_sources(
        space,
        ao,
        checked,
        dbar,
        projected=data,
        role="pairs",
        ao_bar_sink=_f12_ao_sink if corr.streaming else None,
        obs_coulomb=True,
        skip_coulomb_fit_solve=True,
        shell_library=provider.shell_library, host_reference=not cuda,
    )
    del dbar, data
    assert bars is not None
    bars["metric/coulomb"] += extra_metric_coulomb
    fock_pullback(fbar, bars)
    del fbar

    wo, wj, wf = relaxed_f12_source_weights(
        request,
        reference,
        one,
        jk_sources,
        panel,
        proj,
        bars,
        checked,
        budget=provider.budget_mib,
        relative_cutoff=provider.relative_cutoff,
        library=provider.response_library,
        canonical_library=provider.canonical_library,
    )
    del bars
    if corr.streaming:
        grad = streamed_grad
        for mkey, op in _METRIC_KEY_TO_OP.items():
            grad += contract_single_channel(
                corr.layout,
                "metric",
                op,
                wf[mkey],
                provider.shell_library,
                host_reference=host_ref,
                budget_mib=provider.budget_mib,
                slater_normalization=provider.slater_normalization,
            )
        grad += provider._gradient(request, jk, wj, coulomb_only=True)
    else:
        if jk is corr:
            wf["metric"] += wj["metric"]
            wf["ao_coulomb"] += wj["ao_coulomb"]
        grad = provider._gradient(request, corr, wf, coulomb_only=False)
        if jk is not corr:
            grad += provider._gradient(request, jk, wj, coulomb_only=True)
    grad += one.nuclear_gradient(
        wo["overlap"], wo["hcore"], run=provider.run, shell_library=provider.shell_library
    )
    return -grad


def _reset_last_lambda_telemetry(lib: object, ctx: ct.c_void_p | None = None) -> None:
    if ctx is not None:
        ctx_reset = getattr(lib, "generativeqc_cc_reset_context_lambda_telemetry_v1", None)
        if ctx_reset is not None:
            ctx_reset.argtypes = [ct.c_void_p]
            ctx_reset.restype = None
            ctx_reset(ctx)
        return
    reset_fn = getattr(lib, "generativeqc_cc_reset_last_lambda_telemetry_v1", None)
    if reset_fn is not None:
        reset_fn.argtypes = []
        reset_fn.restype = None
        reset_fn()


def _read_last_lambda_telemetry(
    lib: object, ctx: ct.c_void_p | None = None
) -> dict[str, float | bool | int | str]:
    if ctx is None:
        getter = getattr(lib, "generativeqc_cc_last_lambda_telemetry_v1", None)
        handle_types, handles = [], ()
    else:
        getter = getattr(lib, "generativeqc_cc_context_lambda_telemetry_v1", None)
        handle_types, handles = [ct.c_void_p], (ctx,)
    if getter is None:
        return {}
    getter.argtypes = [
        *handle_types,
        ct.POINTER(ct.c_double),
        ct.c_size_t,
        ct.POINTER(ct.c_uint64),
        ct.c_size_t,
        ct.c_char_p,
        ct.c_size_t,
    ]
    getter.restype = ct.c_int
    scalars = np.zeros(10, dtype=np.float64)
    counts = np.zeros(16, dtype=np.uint64)
    reason_buf = ct.create_string_buffer(128)
    valid = getter(
        *handles,
        scalars.ctypes.data_as(ct.POINTER(ct.c_double)),
        scalars.size,
        counts.ctypes.data_as(ct.POINTER(ct.c_uint64)),
        counts.size,
        reason_buf,
        len(reason_buf),
    )
    if not valid:
        return {}
    reason_str = reason_buf.value.decode("utf-8", errors="replace") or "unknown"
    return {
        "fp32_iterations": int(counts[2]),
        "fp64_iterations": int(counts[3]),
        "fp32_actions": int(counts[4]),
        "fp64_actions": int(counts[5]),
        "fp32_seconds": float(scalars[0]),
        "fp64_seconds": float(scalars[1]),
        "precompute_seconds": float(scalars[2]),
        "audit_seconds": float(scalars[3]),
        "publication_seconds": float(scalars[4]),
        "final_residual": float(scalars[5]),
        "independent_residual_norm": float(scalars[6]),
        "independent_residual_max": float(scalars[7]),
        "handoff_reason": reason_str,
        "refinement_passes": int(counts[9]),
        "lambda_auxiliary_batch_size": int(counts[10]),
        "lambda_primal_batch_size": int(counts[11]),
        "lambda_core_reuse": bool(counts[12] != 0),
        "lambda_fp32_refused": bool(counts[7] != 0),
        "lambda_precision_switches": int(counts[8]),
        "lambda_fp32_gemm_calls": int(counts[13]),
        "lambda_fp64_gemm_calls": int(counts[14]),
        "lambda_owned_device_bytes": int(counts[15]),
    }


def _cuda_fixed_cusp_stage_counts(slots, *, forces, advertised):
    """Decode executed native passes and preserve historical host-stage attribution."""
    from ._f12_ram_hygiene import disclose_host_assisted_stage, strict_gpu_only_enabled

    stages = (
        ("fixed_cusp_constant_preparation", "f12_cuda_preparations", slots[8], True),
        ("fixed_cusp_lambda_rhs", "f12_cuda_rhs_evaluations", slots[9], forces),
    )
    result = {}
    for name, key, value, required in stages:
        if np.isnan(value):
            if advertised:
                raise RuntimeError(f"Native library advertised {key} but did not write it")
            result[key] = None
            if required:
                disclose_host_assisted_stage(
                    name, requested_backend="cuda",
                    detail="Historical spatial RCCSD-F12b runtime executes this stage on the host",
                )
                result[name + "_backend"] = "legacy_host_assisted"
            else:
                result[name + "_backend"] = "not_requested"
            continue
        if not np.isfinite(value) or value < 0 or value > 2**53 or int(value) != value:
            raise RuntimeError(f"Invalid native CUDA stage counter {key}={value!r}")
        count = int(value)
        if not required and count:
            raise RuntimeError(f"Energy-only fixed-cusp solve reported {count} Lambda RHS evaluations")
        if required and not count and strict_gpu_only_enabled():
            raise RuntimeError(f"Strict GPU-only mode requires positive native {key}")
        result[key] = count
        result[name + "_backend"] = ("cuda" if count else "unverified") if required else "not_requested"
    return result


def spatial_rhf_ccsdf12b_v1(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    reference: FocalReference,
    *,
    forces: bool = False,
    expected_cabs_rank: int | None = None,
    preconvergence: str | None = None,
    return_amplitudes: bool = False,
) -> SpatialCcsdF12bResult:
    """Solve closed-shell spatial DF-CUDA RHF CCSD-F12b energy and optional 3N analytic forces."""
    from . import _native
    from .calculator import Calculator

    started = time.monotonic()
    preconv = provider.preconvergence if preconvergence is None else preconvergence
    if preconv not in PRECONVERGENCE_CODES:
        raise ValueError(
            f"Unsupported preconvergence {preconv!r}; expected one of {tuple(PRECONVERGENCE_CODES)}"
        )
    basis = request.basis
    if (
        request.role != "mp2_f12_correlation"
        or request.root_id != "rhf"
        or request.multiplicity != 1
        or request.backend != "cuda"
        or basis.mp2_fit is None
        or basis.f12_fit is None
        or basis.cabs is None
        or basis.beta is None
    ):
        raise ValueError(
            "Spatial RHF CCSD-F12b CUDA requires a singlet RHF CUDA pair request with MP2Fit/F12Fit/OptRI"
        )
    if (
        reference.surface_id != request.surface_id
        or reference.basis_id != basis.basis_id
        or reference.executed_backend != request.backend
        or reference.orbital_id != basis.orbital.identity
        or reference.jk_fit_id != basis.jk_fit.identity
    ):
        raise ValueError("Spatial RHF CCSD-F12b requires a matching canonical CUDA RHF reference")

    owner = reference._native_owner
    if owner is None:
        raise RuntimeError("Spatial RHF CCSD-F12b requires the native retained DF reference ABI")
    fn = getattr(owner.library, "generativeqc_focal_rcc_f12b_v1", None)
    if fn is None:
        raise NotImplementedError("Native generativeqc_focal_rcc_f12b_v1 ABI is unavailable")

    if forces and provider.lambda_options is not None:
        from ._f12_ucc_solver import UCCSDLambdaOptions

        if provider.lambda_options != UCCSDLambdaOptions():
            raise NotImplementedError(
                "Native spatial RHF CCSD-F12b C ABI does not support custom "
                "provider.lambda_options; pass use_native=False or omit non-default lambda_options"
            )

    from ._f12_frozen_core import resolve_frozen_occupied
    from .elements import checked_integer

    cc_budget_bytes = (
        checked_integer(
            provider.budget_mib,
            "molecular focal budget in MiB",
            low=1,
            high=(2**63 - 1) >> 20,
        )
        << 20
    )

    reference._pair_spin_indices(request)
    o_full = reference.occupied[0]
    n = len(reference.overlap)
    if not 0 <= o_full <= n or reference.coefficients.shape != (1, n, n):
        raise ValueError("Incompatible spatial RHF orbital dimensions")
    frozen = resolve_frozen_occupied(request, (o_full, o_full))[0]
    o_act = o_full - frozen
    n_vir = n - o_full
    if o_act == 0:
        from ._f12_focal_projector import checked_reference_cabs_rank

        cabs_rank = checked_reference_cabs_rank(provider, request, reference, expected_cabs_rank)
        zero_f = np.zeros((len(request.atoms), 3), dtype=np.float64) if forces else None
        return SpatialCcsdF12bResult(
            0.0,
            cabs_rank,
            None,
            None,
            None,
            None,
            zero_f,
            zero_f,
            {
                "frozen_occupied": frozen,
                "preconvergence": preconv,
                "fixed_sources_seconds": 0.0,
                "f12_cuda_preparations": 0,
                "f12_cuda_rhs_evaluations": 0,
                "fixed_cusp_constant_preparation_backend": "empty_active_space",
                "fixed_cusp_lambda_rhs_backend": "empty_active_space" if forces else "not_requested",
                "final_residual": None,
                "independent_residual_norm": None,
                "independent_residual_max": None,
            },
        )

    if n_vir == 0:
        raise NotImplementedError(
            "RHF CCSD-F12b with occupied pairs and no OBS virtuals requires "
            "the fixed-geminal energy and its nuclear response"
        )

    from ._f12_ram_hygiene import STRICT_GPU_ONLY_ENV, strict_gpu_only_enabled

    probe = getattr(owner.library, "generativeqc_focal_rcc_f12b_cuda_stage_counters_v1", None)
    advertised = False
    if probe is not None:
        probe.restype = ct.c_int
        probe.argtypes = []
        advertised = probe() == 1
    if strict_gpu_only_enabled() and not advertised:
        raise NotImplementedError(
            f"{STRICT_GPU_ONLY_ENV}=1 requires native CUDA fixed-cusp stage counters before execution"
        )
    snap_fn = getattr(provider.run, "profile_snapshot", None)
    jit_before = snap_fn() if callable(snap_fn) else None
    t_fixed_start = time.monotonic()
    w_vo, w_vv, c_f12, fixed_energy, cabs_rank, frozen = _build_spatial_rhf_fixed_sources(
        provider,
        request,
        reference,
        expected_cabs_rank=expected_cabs_rank,
        derivatives=forces,
    )
    fixed_sources_s = time.monotonic() - t_fixed_start
    jit_after_fixed_sources = snap_fn() if callable(snap_fn) else None

    w_vo_c = np.ascontiguousarray(w_vo, dtype=np.float64)
    w_vv_c = np.ascontiguousarray(w_vv, dtype=np.float64)
    c_f12_c = np.ascontiguousarray(c_f12, dtype=np.float64)
    del w_vo, w_vv, c_f12

    need_amps = forces or return_amplitudes
    n1 = o_act * n_vir
    n2 = o_act * o_act * n_vir * n_vir
    required_core_bytes = w_vo_c.nbytes + w_vv_c.nbytes + c_f12_c.nbytes
    if need_amps:
        required_core_bytes += 8 * (n1 + n2)
    if forces:
        required_core_bytes += 8 * (n1 + n2 + 2 * n * n + 3 * len(request.atoms))
    if required_core_bytes > cc_budget_bytes:
        raise MemoryError(
            f"Spatial RHF CCSD-F12b requires at least {required_core_bytes} bytes for "
            f"fixed-source and amplitude buffers, exceeding exact budget_mib={provider.budget_mib} "
            f"({cc_budget_bytes} bytes)"
        )

    calc = Calculator(
        method="df-rccsd(t)",
        device="cuda",
        basis=basis.orbital,
        auxiliary_basis=basis.mp2_fit,
        density_fitting="full_cuda",
        initial_guess=None,
        correlation_memory_budget_bytes=cc_budget_bytes,
    )
    with ExitStack() as stack:
        stack.callback(calc.clear_cache)
        lib = calc._library
        if lib._handle != owner.library._handle:
            raise RuntimeError("The CCSD-F12b library must own the supplied reference")
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
            ct.c_int, ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.c_double, ct.c_uint32,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
            ct.POINTER(ct.c_double), ct.c_size_t,
        ]
        n1 = o_act * n_vir
        n2 = o_act * o_act * n_vir * n_vir
        energy_out = ct.c_double()
        t1_buf = np.empty(n1, dtype=np.float64) if need_amps else None
        t2_buf = np.empty(n2, dtype=np.float64) if need_amps else None
        l1_buf = np.empty(n1, dtype=np.float64) if forces else None
        l2_buf = np.empty(n2, dtype=np.float64) if forces else None
        grad_buf = np.empty(3 * len(request.atoms), dtype=np.float64) if forces else None
        bar_f_buf = np.empty(n * n, dtype=np.float64) if forces else None
        bar_c_buf = np.empty(n * n, dtype=np.float64) if forces else None
        tel_buf = np.zeros(10, dtype=np.float64)
        tel_buf[8:] = np.nan

        def ptr(arr: np.ndarray | None):
            return arr.ctypes.data_as(ct.POINTER(ct.c_double)) if arr is not None else None

        if forces:
            _reset_last_lambda_telemetry(lib, ctx)
        status = fn(
            plan,
            owner.handle,
            1 if forces else 0,
            frozen,
            ptr(w_vo_c), w_vo_c.size,
            ptr(w_vv_c), w_vv_c.size,
            ptr(c_f12_c), c_f12_c.size,
            float(fixed_energy),
            PRECONVERGENCE_CODES[preconv],
            ct.byref(energy_out),
            ptr(t1_buf), 0 if t1_buf is None else t1_buf.size,
            ptr(t2_buf), 0 if t2_buf is None else t2_buf.size,
            ptr(l1_buf), 0 if l1_buf is None else l1_buf.size,
            ptr(l2_buf), 0 if l2_buf is None else l2_buf.size,
            ptr(grad_buf), 0 if grad_buf is None else grad_buf.size,
            ptr(bar_f_buf), 0 if bar_f_buf is None else bar_f_buf.size,
            ptr(bar_c_buf), 0 if bar_c_buf is None else bar_c_buf.size,
            ptr(tel_buf), tel_buf.size,
        )
        _native.check(lib, status, context=ctx)
        fixed_cusp_counts = _cuda_fixed_cusp_stage_counts(
            tel_buf, forces=forces, advertised=advertised
        )
        lambda_detail_tel = _read_last_lambda_telemetry(lib, ctx) if forces else {}
    del w_vo_c, w_vv_c, c_f12_c

    t1_arr = t1_buf.reshape(o_act, n_vir) if t1_buf is not None else None
    t2_arr = t2_buf.reshape(o_act, o_act, n_vir, n_vir) if t2_buf is not None else None
    l1_arr = l1_buf.reshape(o_act, n_vir) if l1_buf is not None else None
    l2_arr = l2_buf.reshape(o_act, o_act, n_vir, n_vir) if l2_buf is not None else None

    ordinary_forces = None
    fixed_forces = None
    cphf_s = 0.0
    fixed_forces_s = 0.0
    if forces:
        assert grad_buf is not None and bar_f_buf is not None and bar_c_buf is not None
        assert t1_arr is not None and t2_arr is not None and l1_arr is not None and l2_arr is not None
        t_cphf = time.monotonic()
        ordinary_forces = relax_spatial_rhf_force_sources(
            provider,
            request,
            reference,
            bar_f_buf.reshape(n, n),
            bar_c_buf.reshape(n, n),
            grad_buf,
        )
        cphf_s = time.monotonic() - t_cphf
        t_ff = time.monotonic()
        fixed_forces = spatial_rhf_fixed_cc_forces(
            provider,
            request,
            reference,
            cabs_rank=cabs_rank,
            fixed_geminal_energy=fixed_energy,
            t1=t1_arr,
            t2=t2_arr,
            lambda1=l1_arr,
            lambda2=l2_arr,
            frozen_occupied=frozen,
        )
        fixed_forces_s = time.monotonic() - t_ff

    corr_energy = float(energy_out.value)
    if not math.isfinite(corr_energy):
        raise RuntimeError("Nonfinite spatial RHF CCSD-F12b correlation energy")
    jit_after = snap_fn() if callable(snap_fn) else None
    jit_cold_s = (
        float(jit_after["jit_cold_s"] - jit_before["jit_cold_s"])
        if jit_before is not None and jit_after is not None
        else 0.0
    )
    warm_reuse_s = (
        float(jit_after["warm_reuse_s"] - jit_before["warm_reuse_s"])
        if jit_before is not None and jit_after is not None
        else 0.0
    )
    jit_compile_count = (
        int(jit_after["jit_compile_count"] - jit_before["jit_compile_count"])
        if jit_before is not None and jit_after is not None
        else 0
    )
    warm_execute_count = (
        int(jit_after["warm_execute_count"] - jit_before["warm_execute_count"])
        if jit_before is not None and jit_after is not None
        else 0
    )
    def _tel_float(key: str) -> float | None:
        return float(lambda_detail_tel[key]) if key in lambda_detail_tel else None

    def _tel_int(key: str) -> int | None:
        return int(lambda_detail_tel[key]) if key in lambda_detail_tel else None

    def _tel_bool(key: str) -> bool | None:
        return bool(lambda_detail_tel[key]) if key in lambda_detail_tel else None

    telemetry: dict[str, float | bool | int | str | None] = {
        **fixed_cusp_counts,
        "ccsd_fp32_executed": bool(tel_buf[0] != 0.0),
        "lambda_fp32_executed": bool(tel_buf[1] != 0.0),
        "ccsd_iterations": int(tel_buf[2]),
        "lambda_iterations": int(tel_buf[3]),
        "ccsd_seconds": float(tel_buf[4]),
        "cc_f12b_cuda_s": float(tel_buf[4]),
        "lambda_seconds": float(tel_buf[5]),
        "lambda_cuda_s": float(tel_buf[5]),
        "source_response_seconds": float(tel_buf[6]),
        "native_total_seconds": float(tel_buf[7]),
        "fixed_sources_seconds": float(fixed_sources_s),
        "fixed_sources_jit_cold_s": (
            float(jit_after_fixed_sources["jit_cold_s"] - jit_before["jit_cold_s"])
            if jit_before is not None and jit_after_fixed_sources is not None
            else 0.0
        ),
        "fixed_sources_warm_reuse_s": (
            float(jit_after_fixed_sources["warm_reuse_s"] - jit_before["warm_reuse_s"])
            if jit_before is not None and jit_after_fixed_sources is not None
            else 0.0
        ),
        "cphf_seconds": float(cphf_s),
        "fixed_forces_seconds": float(fixed_forces_s),
        "cphf_pullback_s": float(cphf_s + float(tel_buf[6]) + fixed_forces_s),
        "jit_cold_s": jit_cold_s,
        "warm_reuse_s": warm_reuse_s,
        "jit_compile_count": jit_compile_count,
        "warm_execute_count": warm_execute_count,
        "wall_seconds": float(time.monotonic() - started),
        "frozen_occupied": frozen,
        "preconvergence": preconv,
        "lambda_telemetry_available": bool(lambda_detail_tel),
        "fp32_iterations": _tel_int("fp32_iterations"),
        "fp64_iterations": (
            int(lambda_detail_tel["fp64_iterations"])
            if "fp64_iterations" in lambda_detail_tel
            else (int(tel_buf[3]) if not forces else None)
        ),
        "fp32_actions": _tel_int("fp32_actions"),
        "fp64_actions": _tel_int("fp64_actions"),
        "fp32_seconds": _tel_float("fp32_seconds"),
        "fp64_seconds": _tel_float("fp64_seconds"),
        "precompute_seconds": _tel_float("precompute_seconds"),
        "audit_seconds": _tel_float("audit_seconds"),
        "publication_seconds": _tel_float("publication_seconds"),
        "final_residual": _tel_float("final_residual"),
        "independent_residual_norm": _tel_float("independent_residual_norm"),
        "independent_residual_max": _tel_float("independent_residual_max"),
        "handoff_reason": (
            str(lambda_detail_tel["handoff_reason"])
            if "handoff_reason" in lambda_detail_tel
            else ("not_requested" if not forces else None)
        ),
        "refinement_passes": _tel_int("refinement_passes"),
        "lambda_auxiliary_batch_size": _tel_int("lambda_auxiliary_batch_size"),
        "lambda_primal_batch_size": _tel_int("lambda_primal_batch_size"),
        "lambda_core_reuse": _tel_bool("lambda_core_reuse"),
        "lambda_fp32_refused": _tel_bool("lambda_fp32_refused"),
        "lambda_precision_switches": _tel_int("lambda_precision_switches"),
        "lambda_fp32_gemm_calls": _tel_int("lambda_fp32_gemm_calls"),
        "lambda_fp64_gemm_calls": _tel_int("lambda_fp64_gemm_calls"),
        "lambda_owned_device_bytes": _tel_int("lambda_owned_device_bytes"),
    }
    return SpatialCcsdF12bResult(
        corr_energy,
        cabs_rank,
        t1_arr if return_amplitudes else None,
        t2_arr if return_amplitudes else None,
        l1_arr if return_amplitudes else None,
        l2_arr if return_amplitudes else None,
        ordinary_forces,
        fixed_forces,
        telemetry,
    )


def spatial_rhf_ccsdf12b_cpu_solve(
    *,
    foo: np.ndarray,
    fov: np.ndarray,
    fvv: np.ndarray,
    ovov: np.ndarray,
    ovvo: np.ndarray,
    oovv: np.ndarray,
    ovvv: np.ndarray,
    ovoo: np.ndarray,
    oooo: np.ndarray,
    vvvv: np.ndarray,
    eps_o: np.ndarray,
    eps_v: np.ndarray,
    w_vo: np.ndarray,
    w_vv: np.ndarray,
    c_f12: np.ndarray,
    fixed_geminal_energy: float,
    solve_lambda: bool = True,
    max_bytes: int = 2 << 30,
) -> dict[str, object]:
    """Invoke generativeqc_rcc_f12b_cpu_solve_v1 for Gate 1 CPU verification."""
    from . import _native

    lib = _native.load_library(device="cpu")
    fn = lib.generativeqc_rcc_f12b_cpu_solve_v1
    fn.restype = ct.c_int
    fn.argtypes = [
        ct.c_size_t, ct.c_size_t,
        *(ct.POINTER(ct.c_double),) * 15,
        ct.c_double, ct.c_int, ct.c_size_t,
        *(ct.POINTER(ct.c_double),) * 8,
    ]
    o, v = int(fov.shape[0]), int(fov.shape[1])

    def c64(x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(x, dtype=np.float64)

    def ptr(x: np.ndarray | None):
        return x.ctypes.data_as(ct.POINTER(ct.c_double)) if x is not None else None

    a_foo, a_fov, a_fvv = c64(foo), c64(fov), c64(fvv)
    a_ovov, a_ovvo, a_oovv = c64(ovov), c64(ovvo), c64(oovv)
    a_ovvv, a_ovoo, a_oooo, a_vvvv = c64(ovvv), c64(ovoo), c64(oooo), c64(vvvv)
    a_eo, a_ev = c64(eps_o), c64(eps_v)
    a_wvo, a_wvv, a_c = c64(w_vo), c64(w_vv), c64(c_f12)

    energy = ct.c_double()
    t1 = np.empty((o, v), dtype=np.float64)
    t2 = np.empty((o, o, v, v), dtype=np.float64)
    l1 = np.empty((o, v), dtype=np.float64) if solve_lambda else None
    l2 = np.empty((o, o, v, v), dtype=np.float64) if solve_lambda else None
    bfoo = np.empty((o, o), dtype=np.float64) if solve_lambda else None
    bfov = np.empty((o, v), dtype=np.float64) if solve_lambda else None
    bfvv = np.empty((v, v), dtype=np.float64) if solve_lambda else None

    if solve_lambda:
        _reset_last_lambda_telemetry(lib)
    status = fn(
        o, v,
        ptr(a_foo), ptr(a_fov), ptr(a_fvv),
        ptr(a_ovov), ptr(a_ovvo), ptr(a_oovv),
        ptr(a_ovvv), ptr(a_ovoo), ptr(a_oooo), ptr(a_vvvv),
        ptr(a_eo), ptr(a_ev),
        ptr(a_wvo), ptr(a_wvv), ptr(a_c),
        float(fixed_geminal_energy),
        1 if solve_lambda else 0,
        int(max_bytes),
        ct.byref(energy),
        ptr(t1), ptr(t2),
        ptr(l1), ptr(l2),
        ptr(bfoo), ptr(bfov), ptr(bfvv),
    )
    _native.check(lib, status)
    out: dict[str, object] = {
        "correlation_energy": float(energy.value),
        "t1": t1,
        "t2": t2,
    }
    if solve_lambda:
        assert l1 is not None and l2 is not None
        out.update(
            lambda1=l1,
            lambda2=l2,
            bar_foo=bfoo,
            bar_fov=bfov,
            bar_fvv=bfvv,
            fixed_weights=spatial_f12_fixed_source_cotangents(t1, t2, l1, l2),
            uccsd_lambda_blocks=spatial_f12_lambda_to_uccsd_blocks(l1, l2),
            lambda_telemetry=_read_last_lambda_telemetry(lib),
        )
    return out
