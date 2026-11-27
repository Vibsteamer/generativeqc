"""Native spin-blocked DF-UCCSD-F12b v1 focal wrapper (CPU & CUDA)."""

from __future__ import annotations

import ctypes as ct
import time
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._f12_focal_ucc_triples import (
    PRECONVERGENCE_CODES,
    relax_spin_blocked_uhf_force_sources,
)

if TYPE_CHECKING:
    from ._f12_focal_cc_sources import FocalUCCFixedSources
    from ._f12_focal_molecular import MolecularFocalProvider
    from ._f12_focal_reference import FocalReference
    from .experimental.f12_focal import FocalRequest


_F12_COUPLING_KEYS = (
    "w_vo_aaaa",
    "w_vo_bbbb",
    "w_vo_abab",
    "w_vo_abba",
    "w_vv_aaaa",
    "w_vv_abab",
    "w_vv_bbbb",
    "c_f12_aaaa",
    "c_f12_abab",
    "c_f12_bbbb",
)


class _F12CouplingV1(ct.Structure):
    _fields_ = [
        ("fixed_geminal_energy", ct.c_double),
        ("w_vo_aaaa", ct.POINTER(ct.c_double)),
        ("w_vo_aaaa_count", ct.c_size_t),
        ("w_vo_bbbb", ct.POINTER(ct.c_double)),
        ("w_vo_bbbb_count", ct.c_size_t),
        ("w_vo_abab", ct.POINTER(ct.c_double)),
        ("w_vo_abab_count", ct.c_size_t),
        ("w_vo_abba", ct.POINTER(ct.c_double)),
        ("w_vo_abba_count", ct.c_size_t),
        ("w_vv_aaaa", ct.POINTER(ct.c_double)),
        ("w_vv_aaaa_count", ct.c_size_t),
        ("w_vv_abab", ct.POINTER(ct.c_double)),
        ("w_vv_abab_count", ct.c_size_t),
        ("w_vv_bbbb", ct.POINTER(ct.c_double)),
        ("w_vv_bbbb_count", ct.c_size_t),
        ("c_f12_aaaa", ct.POINTER(ct.c_double)),
        ("c_f12_aaaa_count", ct.c_size_t),
        ("c_f12_abab", ct.POINTER(ct.c_double)),
        ("c_f12_abab_count", ct.c_size_t),
        ("c_f12_bbbb", ct.POINTER(ct.c_double)),
        ("c_f12_bbbb_count", ct.c_size_t),
    ]


class _F12CotangentsV1(ct.Structure):
    _fields_ = [
        ("bar_fixed_geminal_energy", ct.POINTER(ct.c_double)),
        ("bar_w_vo_aaaa", ct.POINTER(ct.c_double)),
        ("bar_w_vo_aaaa_count", ct.c_size_t),
        ("bar_w_vo_bbbb", ct.POINTER(ct.c_double)),
        ("bar_w_vo_bbbb_count", ct.c_size_t),
        ("bar_w_vo_abab", ct.POINTER(ct.c_double)),
        ("bar_w_vo_abab_count", ct.c_size_t),
        ("bar_w_vo_abba", ct.POINTER(ct.c_double)),
        ("bar_w_vo_abba_count", ct.c_size_t),
        ("bar_w_vv_aaaa", ct.POINTER(ct.c_double)),
        ("bar_w_vv_aaaa_count", ct.c_size_t),
        ("bar_w_vv_abab", ct.POINTER(ct.c_double)),
        ("bar_w_vv_abab_count", ct.c_size_t),
        ("bar_w_vv_bbbb", ct.POINTER(ct.c_double)),
        ("bar_w_vv_bbbb_count", ct.c_size_t),
        ("bar_c_f12_aaaa", ct.POINTER(ct.c_double)),
        ("bar_c_f12_aaaa_count", ct.c_size_t),
        ("bar_c_f12_abab", ct.POINTER(ct.c_double)),
        ("bar_c_f12_abab_count", ct.c_size_t),
        ("bar_c_f12_bbbb", ct.POINTER(ct.c_double)),
        ("bar_c_f12_bbbb_count", ct.c_size_t),
    ]


@dataclass(frozen=True)
class SpinBlockedUccF12bResult:
    """Result, force components, cotangents and telemetry from native DF-UCCSD-F12b v1."""

    ccsd_f12b_energy: float
    raw_f12_cotangents: dict[str, np.ndarray] | None
    fixed_source_weights: dict[str, np.ndarray] | None
    ordinary_forces: np.ndarray | None
    fixed_forces: np.ndarray | None
    forces: np.ndarray | None
    telemetry: dict[str, float | bool | int | str]


def _required_fixed_source_weights(
    oa_act: int,
    ob_act: int,
    va: int,
    vb: int,
    feeds: dict[str, np.ndarray],
    bar_fixed_geminal_energy: float,
    independent_bars: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map the 11 independent fixed-cusp cotangents onto the exact active feed keys.

    In df_ucc_v2, dependent mixed-spin permutations (w_vo_baba, w_vv_{baab,abba,baba},
    c_f12_{baab,abba,baba}) are already folded into w_vo_abba, w_vv_abab and c_f12_abab
    via their exact spin-exchange signs. Filling redundant live keys with zeros preserves
    the exact linear pullback through fixed_ucc_source_forces.
    """
    from generativeqc_compiler.cc.uccsd_equations import build_f12_program
    from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout

    cc = build_f12_program(UCCSDLayout(oa_act, ob_act, va, vb))
    required = {
        node.attrs["name"]
        for node in cc.live_nodes
        if node.op == "input" and node.attrs["name"] in feeds
    }
    weights: dict[str, np.ndarray] = {}
    for key in required:
        if key == "fixed_geminal_energy":
            weights[key] = np.asarray(bar_fixed_geminal_energy, dtype=np.float64)
        elif key in independent_bars:
            weights[key] = np.ascontiguousarray(
                independent_bars[key].reshape(feeds[key].shape), dtype=np.float64
            )
        elif key == "w_vo_baba" and "w_vo_abba" not in required:
            weights[key] = np.ascontiguousarray(
                -np.swapaxes(independent_bars["w_vo_abba"], 0, 1), dtype=np.float64
            )
        else:
            weights[key] = np.zeros_like(feeds[key], dtype=np.float64)
    return weights


def spin_blocked_uhf_f12b_v1(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    fixed: FocalUCCFixedSources,
    *,
    forces: bool = False,
    preconvergence: str = "auto",
) -> SpinBlockedUccF12bResult:
    """Invoke generativeqc_focal_ucc_f12b_v1 for native GPU/CPU spin-blocked UCCSD-F12b."""
    from . import _native
    from ._f12_focal_cc_fixed import fixed_ucc_source_forces
    from ._f12_focal_cc_sources import FocalUCCEquationSources
    from ._f12_frozen_core import resolve_frozen_occupied
    from .calculator import Calculator

    if preconvergence not in PRECONVERGENCE_CODES:
        raise ValueError(
            f"Unsupported preconvergence {preconvergence!r}; expected one of {tuple(PRECONVERGENCE_CODES)}"
        )
    reference: FocalReference = fixed.reference
    basis = request.basis
    if (
        request.role != "mp2_f12_correlation"
        or request.root_id not in ("uhf", "rhf")
        or request.backend not in ("cpu_reference", "cuda")
        or basis.mp2_fit is None
        or basis.f12_fit is None
        or reference.surface_id != request.surface_id
        or reference.executed_backend != request.backend
        or reference.orbital_id != basis.orbital.identity
        or reference.jk_fit_id != basis.jk_fit.identity
    ):
        raise ValueError(
            "Spin-blocked UHF UCCSD-F12b v1 requires matching focal reference and fitting bases"
        )

    spins = reference._pair_spin_indices(request)
    oa_full, ob_full = (reference.occupied[k] for k in spins)
    n = len(reference.overlap)
    frozen_a, frozen_b = resolve_frozen_occupied(request, (oa_full, ob_full))
    if (frozen_a, frozen_b) != fixed.frozen_occupied:
        raise ValueError("Frozen-core counts disagree with FocalUCCFixedSources")
    oa_act = oa_full - frozen_a
    ob_act = ob_full - frozen_b
    va = n - oa_full
    vb = n - ob_full

    feeds = fixed.feeds
    fixed_geminal_energy = float(feeds["fixed_geminal_energy"])
    if oa_act == 0 and ob_act == 0:
        zero_f = np.zeros((len(request.atoms), 3), dtype=np.float64) if forces else None
        return SpinBlockedUccF12bResult(
            fixed_geminal_energy, None, None, zero_f, zero_f, zero_f, {}
        )

    owner = reference._native_owner
    if owner is None:
        raise RuntimeError(
            "Spin-blocked UHF UCCSD-F12b v1 requires the native retained DF reference ABI"
        )
    fn = getattr(owner.library, "generativeqc_focal_ucc_f12b_v1", None)
    if fn is None:
        raise NotImplementedError(
            "Native generativeqc_focal_ucc_f12b_v1 ABI is unavailable"
        )

    expected_shapes = {
        "w_vo_aaaa": (oa_act, oa_act, va, oa_act),
        "w_vo_bbbb": (ob_act, ob_act, vb, ob_act),
        "w_vo_abab": (oa_act, ob_act, va, ob_act),
        "w_vo_abba": (oa_act, ob_act, vb, oa_act),
        "w_vv_aaaa": (oa_act, oa_act, va, va),
        "w_vv_abab": (oa_act, ob_act, va, vb),
        "w_vv_bbbb": (ob_act, ob_act, vb, vb),
        "c_f12_aaaa": (oa_act, oa_act, va, va),
        "c_f12_abab": (oa_act, ob_act, va, vb),
        "c_f12_bbbb": (ob_act, ob_act, vb, vb),
    }
    in_bufs: dict[str, np.ndarray] = {}
    for key in _F12_COUPLING_KEYS:
        arr = np.ascontiguousarray(feeds[key], dtype=np.float64)
        if arr.shape != expected_shapes[key]:
            raise ValueError(
                f"Unexpected shape for {key}: {arr.shape} vs {expected_shapes[key]}"
            )
        in_bufs[key] = arr

    p_dbl = ct.POINTER(ct.c_double)

    def ptr(arr: np.ndarray | None):
        if arr is None or arr.size == 0:
            return None
        return arr.ctypes.data_as(p_dbl)

    coupling = _F12CouplingV1(
        fixed_geminal_energy=fixed_geminal_energy,
        w_vo_aaaa=ptr(in_bufs["w_vo_aaaa"]),
        w_vo_aaaa_count=in_bufs["w_vo_aaaa"].size,
        w_vo_bbbb=ptr(in_bufs["w_vo_bbbb"]),
        w_vo_bbbb_count=in_bufs["w_vo_bbbb"].size,
        w_vo_abab=ptr(in_bufs["w_vo_abab"]),
        w_vo_abab_count=in_bufs["w_vo_abab"].size,
        w_vo_abba=ptr(in_bufs["w_vo_abba"]),
        w_vo_abba_count=in_bufs["w_vo_abba"].size,
        w_vv_aaaa=ptr(in_bufs["w_vv_aaaa"]),
        w_vv_aaaa_count=in_bufs["w_vv_aaaa"].size,
        w_vv_abab=ptr(in_bufs["w_vv_abab"]),
        w_vv_abab_count=in_bufs["w_vv_abab"].size,
        w_vv_bbbb=ptr(in_bufs["w_vv_bbbb"]),
        w_vv_bbbb_count=in_bufs["w_vv_bbbb"].size,
        c_f12_aaaa=ptr(in_bufs["c_f12_aaaa"]),
        c_f12_aaaa_count=in_bufs["c_f12_aaaa"].size,
        c_f12_abab=ptr(in_bufs["c_f12_abab"]),
        c_f12_abab_count=in_bufs["c_f12_abab"].size,
        c_f12_bbbb=ptr(in_bufs["c_f12_bbbb"]),
        c_f12_bbbb_count=in_bufs["c_f12_bbbb"].size,
    )

    energy_out = ct.c_double()
    bar_fixed_e_out = ct.c_double()
    out_f12_bufs = {
        key: (np.empty(expected_shapes[key], dtype=np.float64) if forces else None)
        for key in _F12_COUPLING_KEYS
    }
    cotangents = (
        _F12CotangentsV1(
            bar_fixed_geminal_energy=ct.pointer(bar_fixed_e_out),
            bar_w_vo_aaaa=ptr(out_f12_bufs["w_vo_aaaa"]),
            bar_w_vo_aaaa_count=out_f12_bufs["w_vo_aaaa"].size,
            bar_w_vo_bbbb=ptr(out_f12_bufs["w_vo_bbbb"]),
            bar_w_vo_bbbb_count=out_f12_bufs["w_vo_bbbb"].size,
            bar_w_vo_abab=ptr(out_f12_bufs["w_vo_abab"]),
            bar_w_vo_abab_count=out_f12_bufs["w_vo_abab"].size,
            bar_w_vo_abba=ptr(out_f12_bufs["w_vo_abba"]),
            bar_w_vo_abba_count=out_f12_bufs["w_vo_abba"].size,
            bar_w_vv_aaaa=ptr(out_f12_bufs["w_vv_aaaa"]),
            bar_w_vv_aaaa_count=out_f12_bufs["w_vv_aaaa"].size,
            bar_w_vv_abab=ptr(out_f12_bufs["w_vv_abab"]),
            bar_w_vv_abab_count=out_f12_bufs["w_vv_abab"].size,
            bar_w_vv_bbbb=ptr(out_f12_bufs["w_vv_bbbb"]),
            bar_w_vv_bbbb_count=out_f12_bufs["w_vv_bbbb"].size,
            bar_c_f12_aaaa=ptr(out_f12_bufs["c_f12_aaaa"]),
            bar_c_f12_aaaa_count=out_f12_bufs["c_f12_aaaa"].size,
            bar_c_f12_abab=ptr(out_f12_bufs["c_f12_abab"]),
            bar_c_f12_abab_count=out_f12_bufs["c_f12_abab"].size,
            bar_c_f12_bbbb=ptr(out_f12_bufs["c_f12_bbbb"]),
            bar_c_f12_bbbb_count=out_f12_bufs["c_f12_bbbb"].size,
        )
        if forces
        else None
    )

    grad_buf = np.empty(3 * len(request.atoms), dtype=np.float64) if forces else None
    bar_fa_buf = np.empty(n * n, dtype=np.float64) if forces else None
    bar_fb_buf = np.empty(n * n, dtype=np.float64) if forces else None
    bar_ca_buf = np.empty(n * n, dtype=np.float64) if forces else None
    bar_cb_buf = np.empty(n * n, dtype=np.float64) if forces else None
    tel_buf = np.zeros(10, dtype=np.float64)

    cuda = request.backend == "cuda"
    budget_bytes = provider.budget_mib << 20
    calc = Calculator(
        method="uccsd",
        device="cuda" if cuda else "cpu",
        basis=basis.orbital,
        auxiliary_basis=basis.mp2_fit,
        density_fitting="full_cuda",
        initial_guess=None,
        correlation_memory_budget_bytes=budget_bytes,
    )
    with ExitStack() as stack:
        stack.callback(calc.clear_cache)
        lib = calc._library
        if lib._handle != owner.library._handle:
            raise RuntimeError("The UCC-F12b library must own the supplied reference")
        ctx = ct.c_void_p()
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

        fn.restype = ct.c_int
        fn.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_size_t,
            ct.c_int,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_uint32,
            ct.POINTER(_F12CouplingV1),
            p_dbl,
            ct.POINTER(_F12CotangentsV1),
            p_dbl,
            ct.c_size_t,
            p_dbl,
            ct.c_size_t,
            p_dbl,
            ct.c_size_t,
            p_dbl,
            ct.c_size_t,
            p_dbl,
            ct.c_size_t,
            p_dbl,
            ct.c_size_t,
        ]
        status = fn(
            ctx,
            system,
            correlation_fit,
            owner.handle,
            budget_bytes,
            1 if forces else 0,
            frozen_a,
            frozen_b,
            PRECONVERGENCE_CODES[preconvergence],
            ct.byref(coupling),
            ct.byref(energy_out),
            ct.byref(cotangents) if cotangents is not None else None,
            ptr(grad_buf),
            0 if grad_buf is None else grad_buf.size,
            ptr(bar_fa_buf),
            0 if bar_fa_buf is None else bar_fa_buf.size,
            ptr(bar_fb_buf),
            0 if bar_fb_buf is None else bar_fb_buf.size,
            ptr(bar_ca_buf),
            0 if bar_ca_buf is None else bar_ca_buf.size,
            ptr(bar_cb_buf),
            0 if bar_cb_buf is None else bar_cb_buf.size,
            ptr(tel_buf),
            tel_buf.size,
        )
        _native.check(lib, status, context=ctx)

    telemetry: dict[str, float | bool | int | str] = {
        "ccsd_fp32_executed": bool(tel_buf[0] != 0.0),
        "lambda_fp32_executed": bool(tel_buf[1] != 0.0),
        "ccsd_seconds": float(tel_buf[4]),
        "lambda_seconds": float(tel_buf[6]),
        "source_response_seconds": float(tel_buf[7]),
        "ccsd_iterations": int(tel_buf[8]),
        "lambda_iterations": int(tel_buf[9]),
        "frozen_alpha": frozen_a,
        "frozen_beta": frozen_b,
        "preconvergence": preconvergence,
    }

    raw_f12_cotangents: dict[str, np.ndarray] | None = None
    fixed_source_weights: dict[str, np.ndarray] | None = None
    ordinary_forces: np.ndarray | None = None
    fixed_forces: np.ndarray | None = None
    total_forces: np.ndarray | None = None
    if forces:
        assert (
            grad_buf is not None
            and bar_fa_buf is not None
            and bar_fb_buf is not None
            and bar_ca_buf is not None
            and bar_cb_buf is not None
        )
        t0 = time.perf_counter()
        ordinary_forces = relax_spin_blocked_uhf_force_sources(
            provider,
            request,
            reference,
            bar_fa_buf.reshape(n, n),
            bar_fb_buf.reshape(n, n),
            bar_ca_buf.reshape(n, n),
            bar_cb_buf.reshape(n, n),
            grad_buf,
        )
        telemetry["cphf_relax_seconds"] = float(time.perf_counter() - t0)

        indep_bars: dict[str, np.ndarray] = {}
        for key in _F12_COUPLING_KEYS:
            buf = out_f12_bufs[key]
            assert buf is not None
            indep_bars[key] = buf
        raw_f12_cotangents = {
            "fixed_geminal_energy": np.asarray(
                float(bar_fixed_e_out.value), dtype=np.float64
            ),
            **indep_bars,
        }
        fixed_source_weights = _required_fixed_source_weights(
            oa_act,
            ob_act,
            va,
            vb,
            feeds,
            float(bar_fixed_e_out.value),
            indep_bars,
        )
        t1 = time.perf_counter()
        eq_sources = FocalUCCEquationSources(fixed, basis.mp2_fit.identity, {})
        fixed_forces = fixed_ucc_source_forces(
            provider, request, eq_sources, fixed_source_weights
        )
        telemetry["fixed_forces_seconds"] = float(time.perf_counter() - t1)
        total_forces = ordinary_forces + fixed_forces

    return SpinBlockedUccF12bResult(
        float(energy_out.value),
        raw_f12_cotangents,
        fixed_source_weights,
        ordinary_forces,
        fixed_forces,
        total_forces,
        telemetry,
    )
