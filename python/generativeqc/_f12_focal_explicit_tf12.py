"""Native spatial-orbital DF-CPU and DF-CUDA RHF Kohn extended-SP (T)-F12 energy engine."""

from __future__ import annotations

import ctypes as ct
import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from ._f12_focal_rcc_f12b import (
    SpatialCcsdF12bResult,
    _make_checked_runner,
    spatial_rhf_ccsdf12b_cpu_solve,
    spatial_rhf_ccsdf12b_v1,
)

if TYPE_CHECKING:
    from ._f12_focal_molecular import MolecularFocalProvider
    from ._f12_focal_reference import FocalReference
    from .experimental.f12_focal import FocalRequest


Z2_MODE_CODES: dict[str, int] = {
    "direct": 0,
    "ffg": 1,
    "zero": 2,
}

BREAKDOWN_KEYS: tuple[str, ...] = (
    "e_pt_with_c2",
    "e_pt_conv",
    "t1_h3_x2",
    "t2_h3_x6",
    "t3_xh_h7",
    "t14_xh_h9",
    "t4_x2_h7",
    "t15_x3_h9",
    "t5_z2",
    "t6_t1_v9",
    "t7_t1_v7",
    "t8_v8",
    "t9_v10",
    "t16_bh4",
    "t17_fo_x7",
    "t18_x7_fo",
    "t19_fo_x4",
    "t20_fo_x7_ex",
    "t21_b4",
    "t22_b7",
    "t23_fv_x4",
    "t24_fv_x7",
    "a_sum",
    "total_energy",
    "seconds_pt",
    "seconds_a_vxbbh",
    "seconds_z2",
    "peak_workspace_bytes",
    "cuda_w_fp32_gemms",
    "cuda_w_fp64_gemms",
)


@dataclass(frozen=True)
class ExplicitTf12Result:
    """Result, 33-term scalar breakdown, and execution telemetry from spatial RHF (T)-F12."""

    energy: float
    breakdown: dict[str, float]
    ccsd_f12b_energy: float | None
    cabs_rank: int
    telemetry: dict[str, float | int | bool | str]


def _unpack_breakdown(buf: np.ndarray, *, conventional_evaluated: bool = True) -> dict[str, float]:
    out = dict(zip(BREAKDOWN_KEYS, map(float, buf), strict=True))
    if conventional_evaluated:
        out["delta_c2"] = out["e_pt_with_c2"] - out["e_pt_conv"]
    else:
        # A coupled CUDA calculation skips the separate conventional diagnostic for o > 10.
        out.pop("e_pt_conv")
    if not all(math.isfinite(value) for value in out.values()):
        raise RuntimeError("Incomplete or nonfinite native (T)-F12 result/precision telemetry")
    for key in ("cuda_w_fp32_gemms", "cuda_w_fp64_gemms"):
        if out[key] < 0 or not out[key].is_integer():
            raise RuntimeError("Invalid native (T)-F12 W GEMM counter")
    return out


def _as_f64_ptr(arr: np.ndarray | None):
    if arr is None:
        return None
    return arr.ctypes.data_as(ct.POINTER(ct.c_double))


def _native_array(value, shape, name, *, optional=False):
    if value is None:
        if optional:
            return None
        raise ValueError(f"{name} is required")
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.float64:
        raise ValueError(f"{name} requires an FP64 array of shape {shape}")
    array = np.ascontiguousarray(array)
    flat = array.reshape(-1)
    if any(not np.isfinite(flat[i:i + 1048576]).all() for i in range(0, flat.size, 1048576)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _validated_inputs(params, *, exact=False):
    from .elements import checked_integer

    def dim(name, minimum=0):
        return checked_integer(params[name], name, low=minimum, high=np.iinfo(np.intp).max)

    o_full, n, ext = dim("o_full"), dim("obs"), dim("ext")
    o, v = o_full - dim("frozen"), n - o_full
    if min(o, v) <= 0 or ext < n or not math.isfinite(params["geminal_scale"]):
        raise ValueError("invalid explicit triples orbital dimensions or geminal scale")

    # Preserve the argument order of the C API for both DF and exact-integral input.
    parent = {
        "ovoo": (o, v, o, o), "ovov": (o, v, o, v), "fov": (o, v),
        "t1": (o, v), "t2": (o, o, v, v), "eps_o": (o,), "eps_v": (v,),
    }
    optional = set()
    if exact:
        shapes = {
            "ovvv": (o, v, v, v), **parent, "c2_ab": (v, v, o, v),
            "g_obs_obs_ext_ext": (n, n, ext, ext),
            "g_chem_ext_ext_o_v": (ext, ext, o, v),
            "g_chem_ext_o_ext_v": (ext, o, ext, v), "f12": (ext, ext, n, n),
            **{k: (n, n, n, n) for k in ("fg", "ff", "ftf", "ffbar")},
            "f_ext": (ext, ext), "k_ext": (ext, ext),
        }
        optional.add("c2_ab")
        if params["z2_mode"] == "zero":
            optional.update(("g_chem_ext_ext_o_v", "g_chem_ext_o_ext_v"))
    else:
        q, f = dim("q_mp2", 1), dim("q_f12")
        corr = {
            "b_g_ext_obs": (f, ext, n), "b_g_ext_ext": (f, ext, ext),
            "fit_ext_obs": (f, ext, n), "r_f_ext_obs": (f, ext, n),
            "r_fg_obs_obs": (f, n, n), "r_ff_ext_obs": (f, ext, n),
            "r_u_obs_obs": (f, n, n), "f_ext": (ext, ext), "k_ext": (ext, ext),
        }
        shapes = {"bov": (q, o, v), "bvv": (q, v, v), **parent, **corr}
        if f == 0:
            optional.update(corr)
        elif ext == n:
            raise ValueError("coupled explicit triples requires a complementary space")
        if params["z2_mode"] == "zero":
            optional.add("b_g_ext_ext")
    return [
        _native_array(params[name], shape, name, optional=name in optional)
        for name, shape in shapes.items()
    ]



def explicit_tf12_df_cpu(
    *,
    o_full: int,
    frozen: int,
    obs: int,
    ext: int,
    q_mp2: int,
    q_f12: int,
    bov: np.ndarray,
    bvv: np.ndarray,
    ovoo: np.ndarray,
    ovov: np.ndarray,
    fov: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    eps_o: np.ndarray,
    eps_v: np.ndarray,
    b_g_ext_obs: np.ndarray | None = None,
    b_g_ext_ext: np.ndarray | None = None,
    fit_ext_obs: np.ndarray | None = None,
    r_f_ext_obs: np.ndarray | None = None,
    r_fg_obs_obs: np.ndarray | None = None,
    r_ff_ext_obs: np.ndarray | None = None,
    r_u_obs_obs: np.ndarray | None = None,
    f_ext: np.ndarray | None = None,
    k_ext: np.ndarray | None = None,
    z2_mode: str = "ffg",
    geminal_scale: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Invoke native CPU spatial DF Kohn extended-SP (T)-F12 solver."""
    from . import _native

    if z2_mode not in Z2_MODE_CODES:
        raise ValueError(f"unsupported z2_mode {z2_mode!r}; expected one of {tuple(Z2_MODE_CODES)}")
    lib = _native.load_library(device="cpu")
    fn = getattr(lib, "generativeqc_rcc_explicit_tf12_df_cpu_v1", None)
    if fn is None:
        raise NotImplementedError("Native generativeqc_rcc_explicit_tf12_df_cpu_v1 ABI is unavailable")

    c_ptr = ct.POINTER(ct.c_double)
    fn.restype = ct.c_int
    fn.argtypes = [
        ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_size_t,
        c_ptr, c_ptr, c_ptr, c_ptr, c_ptr, c_ptr, c_ptr, c_ptr, c_ptr,
        c_ptr, c_ptr, c_ptr, c_ptr, c_ptr, c_ptr, c_ptr, c_ptr, c_ptr,
        ct.c_uint32, ct.c_double, c_ptr, c_ptr, ct.c_size_t,
    ]
    arrs = _validated_inputs(locals())
    energy_out = ct.c_double()
    bk_buf = np.full(len(BREAKDOWN_KEYS), np.nan, dtype=np.float64)
    status = fn(
        int(o_full), int(frozen), int(obs), int(ext), int(q_mp2), int(q_f12),
        *(_as_f64_ptr(a) for a in arrs),
        Z2_MODE_CODES[z2_mode],
        float(geminal_scale),
        ct.byref(energy_out),
        _as_f64_ptr(bk_buf),
        bk_buf.size,
    )
    _native.check(lib, status)
    return float(energy_out.value), _unpack_breakdown(bk_buf)


def explicit_tf12_exact4_cpu(
    *,
    o_full: int,
    frozen: int,
    obs: int,
    ext: int,
    ovvv: np.ndarray,
    ovoo: np.ndarray,
    ovov: np.ndarray,
    fov: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    eps_o: np.ndarray,
    eps_v: np.ndarray,
    c2_ab: np.ndarray | None,
    g_obs_obs_ext_ext: np.ndarray,
    g_chem_ext_ext_o_v: np.ndarray | None,
    g_chem_ext_o_ext_v: np.ndarray | None,
    f12: np.ndarray,
    fg: np.ndarray,
    ff: np.ndarray,
    ftf: np.ndarray,
    ffbar: np.ndarray,
    f_ext: np.ndarray,
    k_ext: np.ndarray,
    z2_mode: str = "ffg",
    geminal_scale: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Invoke native CPU exact-4-index Kohn extended-SP (T)-F12 solver (GeCCo BH parity)."""
    from . import _native

    if z2_mode not in Z2_MODE_CODES:
        raise ValueError(f"unsupported z2_mode {z2_mode!r}")
    lib = _native.load_library(device="cpu")
    fn = getattr(lib, "generativeqc_rcc_explicit_tf12_exact4_cpu_v1", None)
    if fn is None:
        raise NotImplementedError("Native generativeqc_rcc_explicit_tf12_exact4_cpu_v1 ABI is unavailable")

    c_ptr = ct.POINTER(ct.c_double)
    fn.restype = ct.c_int
    fn.argtypes = [
        ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_size_t,
        *([c_ptr] * 19),
        ct.c_uint32, ct.c_double, c_ptr, c_ptr, ct.c_size_t,
    ]
    arrs = _validated_inputs(locals(), exact=True)
    energy_out = ct.c_double()
    bk_buf = np.full(len(BREAKDOWN_KEYS), np.nan, dtype=np.float64)
    status = fn(
        int(o_full), int(frozen), int(obs), int(ext),
        *(_as_f64_ptr(a) for a in arrs),
        Z2_MODE_CODES[z2_mode],
        float(geminal_scale),
        ct.byref(energy_out),
        _as_f64_ptr(bk_buf),
        bk_buf.size,
    )
    _native.check(lib, status)
    return float(energy_out.value), _unpack_breakdown(bk_buf)


def explicit_tf12_df_cuda(
    *,
    device: int,
    max_bytes: int,
    triples_fp32_w: bool,
    o_full: int,
    frozen: int,
    obs: int,
    ext: int,
    q_mp2: int,
    q_f12: int,
    bov: np.ndarray,
    bvv: np.ndarray,
    ovoo: np.ndarray,
    ovov: np.ndarray,
    fov: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    eps_o: np.ndarray,
    eps_v: np.ndarray,
    b_g_ext_obs: np.ndarray | None = None,
    b_g_ext_ext: np.ndarray | None = None,
    fit_ext_obs: np.ndarray | None = None,
    r_f_ext_obs: np.ndarray | None = None,
    r_fg_obs_obs: np.ndarray | None = None,
    r_ff_ext_obs: np.ndarray | None = None,
    r_u_obs_obs: np.ndarray | None = None,
    f_ext: np.ndarray | None = None,
    k_ext: np.ndarray | None = None,
    z2_mode: str = "ffg",
    geminal_scale: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Invoke native GPU-resident DF-CUDA spatial Kohn extended-SP (T)-F12 solver."""
    from . import _native

    if z2_mode not in Z2_MODE_CODES:
        raise ValueError(f"unsupported z2_mode {z2_mode!r}")
    lib = _native.load_library(device="cuda", device_id=device)
    fn = getattr(lib, "generativeqc_rcc_explicit_tf12_df_cuda_v1", None)
    if fn is None:
        raise NotImplementedError("Native generativeqc_rcc_explicit_tf12_df_cuda_v1 ABI is unavailable")

    c_ptr = ct.POINTER(ct.c_double)
    fn.restype = ct.c_int
    fn.argtypes = [
        ct.c_int, ct.c_size_t, ct.c_int,
        ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_size_t,
        *([c_ptr] * 18),
        ct.c_uint32, ct.c_double, c_ptr, c_ptr, ct.c_size_t,
    ]
    arrs = _validated_inputs(locals())
    energy_out = ct.c_double()
    bk_buf = np.full(len(BREAKDOWN_KEYS), np.nan, dtype=np.float64)
    status = fn(
        int(device), int(max_bytes), 1 if triples_fp32_w else 0,
        int(o_full), int(frozen), int(obs), int(ext), int(q_mp2), int(q_f12),
        *(_as_f64_ptr(a) for a in arrs),
        Z2_MODE_CODES[z2_mode],
        float(geminal_scale),
        ct.byref(energy_out),
        _as_f64_ptr(bk_buf),
        bk_buf.size,
    )
    _native.check(lib, status)
    return float(energy_out.value), _unpack_breakdown(
        bk_buf, conventional_evaluated=(q_f12 == 0 or o_full - frozen <= 10)
    )


def _project_ao_panel_tiled(
    ao_mat: np.ndarray,
    c_left: np.ndarray,
    c_right: np.ndarray,
    *,
    q_tile: int = 64,
) -> np.ndarray:
    """Q-tiled AO-to-MO projection: out[q, p, r] = sum_{m,n} ao_mat[q, m, n] * c_left[m, p] * c_right[n, r]."""
    q_tot, nc, _ = ao_mat.shape
    n_left = c_left.shape[1]
    n_right = c_right.shape[1]
    out = np.empty((q_tot, n_left, n_right), dtype=np.float64)
    c_l_t = np.ascontiguousarray(c_left.T, dtype=np.float64)
    c_r = np.ascontiguousarray(c_right, dtype=np.float64)
    for q0 in range(0, q_tot, q_tile):
        q1 = min(q_tot, q0 + q_tile)
        block = np.ascontiguousarray(ao_mat[q0:q1], dtype=np.float64)
        # half[q, m, r] = block[q, m, n] @ c_r[n, r]
        half = (block.reshape(-1, nc) @ c_r).reshape(q1 - q0, nc, n_right)
        # out[q, p, r] = c_l_t[p, m] @ half[q, m, r]
        for qi in range(q1 - q0):
            out[q0 + qi] = c_l_t @ half[qi]
    return out


def _apply_metric_left(mat_qq: np.ndarray, tensor_qpr: np.ndarray, *, alpha: float = 1.0, out_add: np.ndarray | None = None) -> np.ndarray:
    """Compute alpha * (mat_qq @ tensor_qpr.reshape(Q, -1)) reshaped to tensor_qpr.shape."""
    q = tensor_qpr.shape[0]
    res = (mat_qq @ tensor_qpr.reshape(q, -1)).reshape(tensor_qpr.shape)
    if alpha != 1.0:
        res *= alpha
    if out_add is not None:
        return out_add + res
    return res


def _apply_metric_left_inplace(
    mat_qq: np.ndarray, tensor_qpr: np.ndarray, *, col_tile: int = 65536
) -> np.ndarray:
    """Apply mat_qq @ tensor_qpr in-place in column tiles to bound peak host RAM."""
    q = tensor_qpr.shape[0]
    flat = tensor_qpr.reshape(q, -1)
    n = flat.shape[1]
    for c0 in range(0, n, col_tile):
        c1 = min(n, c0 + col_tile)
        flat[:, c0:c1] = mat_qq @ flat[:, c0:c1]
    return tensor_qpr


def _apply_metric_sub_inplace(
    mat_qq: np.ndarray,
    fit_qpr: np.ndarray,
    target_qpr: np.ndarray,
    *,
    alpha: float = -0.5,
    col_tile: int = 65536,
) -> np.ndarray:
    """Compute target_qpr += alpha * (mat_qq @ fit_qpr) in-place in column tiles."""
    q = fit_qpr.shape[0]
    fit_flat = fit_qpr.reshape(q, -1)
    tgt_flat = target_qpr.reshape(q, -1)
    n = fit_flat.shape[1]
    for c0 in range(0, n, col_tile):
        c1 = min(n, c0 + col_tile)
        tgt_flat[:, c0:c1] += alpha * (mat_qq @ fit_flat[:, c0:c1])
    return target_qpr


@dataclass
class PreparedExplicitTf12Inputs:
    """Pre-projected 3-index MO tensors ready for one or more (T)-F12 kernel evaluations."""

    backend: str
    o_full: int
    frozen: int
    obs: int
    ext: int
    q_mp2: int
    q_f12: int
    cabs_rank: int
    ccsd_f12b_energy: float | None
    zero_coupling: bool
    bov: np.ndarray
    bvv: np.ndarray
    ovoo: np.ndarray
    ovov: np.ndarray
    fov: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    eps_o: np.ndarray
    eps_v: np.ndarray
    b_g_ext_obs: np.ndarray | None
    b_g_ext_ext: np.ndarray | None
    fit_ext_obs: np.ndarray | None
    r_f_ext_obs: np.ndarray | None
    r_fg_obs_obs: np.ndarray | None
    r_ff_ext_obs: np.ndarray | None
    r_u_obs_obs: np.ndarray | None
    f_ext: np.ndarray | None
    k_ext: np.ndarray | None
    mp2_proj_seconds: float
    parent_seconds: float
    f12_proj_seconds: float


def prepare_spatial_rhf_explicit_tf12_inputs(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    reference: FocalReference,
    *,
    ccsd_f12b_result: SpatialCcsdF12bResult | None = None,
    t1: np.ndarray | None = None,
    t2: np.ndarray | None = None,
    expected_cabs_rank: int | None = None,
    need_z2: bool = True,
    preconvergence: str | None = None,
    zero_coupling: bool = False,
) -> PreparedExplicitTf12Inputs | None:
    """Build parent (T1, T2) and Q-tiled 3-index MO tensors once for spatial RHF (T)-F12."""
    import gc
    from ._f12_cc_fock_pullback import extended_fock_response
    from ._f12_focal_frame import canonical_pair_frame
    from ._f12_focal_reference import partition_focal_reference
    from ._f12_frozen_core import resolve_frozen_occupied

    if (t1 is None) != (t2 is None):
        raise ValueError("Precomputed explicit triples parent requires both T1 and T2")
    basis = request.basis
    if (
        request.root_id != "rhf"
        or request.multiplicity != 1
        or basis.mp2_fit is None
    ):
        raise ValueError("Spatial RHF (T)-F12 requires a singlet RHF request with MP2Fit")

    cuda = request.backend == "cuda"
    if not cuda and request.backend != "cpu_reference":
        raise ValueError(f"Unsupported backend for spatial RHF (T)-F12: {request.backend!r}")

    if (reference.executed_backend != request.backend or reference.surface_id != request.surface_id
            or reference.basis_id != basis.basis_id or reference.orbital_id != basis.orbital.identity
            or reference.jk_fit_id != basis.jk_fit.identity or len(reference.occupied) != 1
            or request.slater_normalization != provider.slater_normalization):
        raise ValueError("Explicit triples requires the same molecular determinant and fitting recipe")
    reference = partition_focal_reference(request, reference, provider.budget_mib)
    o_full = reference.occupied[0]
    obs = len(reference.overlap)
    frozen = resolve_frozen_occupied(request, (o_full, o_full))[0]
    o_act = o_full - frozen
    n_vir = obs - o_full
    if o_act == 0:
        return None
    if n_vir <= 0:
        raise NotImplementedError("Explicit triples with occupied pairs requires OBS virtual orbitals")

    eff_preconv = provider.preconvergence if preconvergence is None else preconvergence

    # 1. Build ordinary active MP2Fit DF blocks: bov[Q, o, v], bvv[Q, v, v], ovoo, ovov, fov.
    from ._f12_ram_hygiene import disclose_host_assisted_stage
    disclose_host_assisted_stage(
        "explicit_tf12_mo_projection", requested_backend=request.backend,
        detail="FP64 NumPy projects the MP2 and F12 fitting factors in the retained RHF/CABS frame",
    )
    if not cuda and ccsd_f12b_result is None and t1 is None and t2 is None:
        minimum = 8 * (n_vir**4 + o_act * n_vir**3 + 3 * (o_act*n_vir)**2)
        if minimum > provider.budget_mib << 20:
            raise MemoryError("Explicit triples CPU parent integral blocks exceed the caller budget")
    t_mp2_start = time.monotonic()
    pair_req = replace(request, role="mp2_f12_correlation")
    mp2_panel = provider._panel(pair_req, "mp2_fit", orbital_only=True)
    ao_mp2 = mp2_panel.obs_coulomb(obs)  # [Q_mp2, obs, obs]
    w_mp2 = np.ascontiguousarray(mp2_panel.whitener, dtype=np.float64)
    q_mp2 = len(w_mp2)
    c_obs = np.ascontiguousarray(reference.coefficients[0], dtype=np.float64)
    c_occ = c_obs[:, frozen:o_full]
    c_vir = c_obs[:, o_full:obs]

    raw_oo = _project_ao_panel_tiled(ao_mp2, c_occ, c_occ)
    raw_ov = _project_ao_panel_tiled(ao_mp2, c_occ, c_vir)
    raw_vv = _project_ao_panel_tiled(ao_mp2, c_vir, c_vir)
    boo = _apply_metric_left(w_mp2, raw_oo)  # [Q_mp2, o_act, o_act]
    bov = _apply_metric_left(w_mp2, raw_ov)  # [Q_mp2, o_act, n_vir]
    bvv = _apply_metric_left(w_mp2, raw_vv)  # [Q_mp2, n_vir, n_vir]
    bvv = 0.5 * (bvv + bvv.swapaxes(1, 2))
    del raw_oo, raw_ov, raw_vv, ao_mp2

    ovoo = np.einsum("qia,qjk->iajk", bov, boo, optimize=True)
    ovov = np.einsum("qia,qjb->iajb", bov, bov, optimize=True)
    eps_all = np.asarray(reference.orbital_energies[0], dtype=np.float64)
    eps_o = np.ascontiguousarray(eps_all[frozen:o_full], dtype=np.float64)
    eps_v = np.ascontiguousarray(eps_all[o_full:obs], dtype=np.float64)
    fock_mo = c_obs.T @ reference.fock[0] @ c_obs
    fov = np.ascontiguousarray(fock_mo[frozen:o_full, o_full:obs], dtype=np.float64)
    mp2_proj_s = time.monotonic() - t_mp2_start

    # 2. Obtain parent amplitudes (t1, t2) if not supplied by caller.
    t_parent_start = time.monotonic()
    ccsd_f12b_energy: float | None = None
    cabs_rank = 0
    if t1 is not None and t2 is not None:
        t1_use = np.ascontiguousarray(t1, dtype=np.float64)
        t2_use = np.ascontiguousarray(t2, dtype=np.float64)
        if ccsd_f12b_result is not None:
            ccsd_f12b_energy = float(ccsd_f12b_result.correlation_energy)
            cabs_rank = int(ccsd_f12b_result.cabs_rank)
    elif ccsd_f12b_result is not None and ccsd_f12b_result.t1 is not None and ccsd_f12b_result.t2 is not None:
        t1_use = np.ascontiguousarray(ccsd_f12b_result.t1, dtype=np.float64)
        t2_use = np.ascontiguousarray(ccsd_f12b_result.t2, dtype=np.float64)
        ccsd_f12b_energy = float(ccsd_f12b_result.correlation_energy)
        cabs_rank = int(ccsd_f12b_result.cabs_rank)
    elif zero_coupling:
        oooo = np.einsum("qij,qkl->ijkl", boo, boo, optimize=True)
        oovv = np.einsum("qij,qab->ijab", boo, bvv, optimize=True)
        ovvo = ovov.transpose(0, 1, 3, 2).copy()
        ovvv = np.einsum("qia,qbc->iabc", bov, bvv, optimize=True)
        vvvv = np.einsum("qab,qcd->abcd", bvv, bvv, optimize=True)
        foo = np.ascontiguousarray(fock_mo[frozen:o_full, frozen:o_full], dtype=np.float64)
        fvv = np.ascontiguousarray(fock_mo[o_full:obs, o_full:obs], dtype=np.float64)
        sol = spatial_rhf_ccsdf12b_cpu_solve(
            foo=foo, fov=fov, fvv=fvv, ovov=ovov, ovvo=ovvo, oovv=oovv,
            ovvv=ovvv, ovoo=ovoo, oooo=oooo, vvvv=vvvv, eps_o=eps_o, eps_v=eps_v,
            w_vo=np.zeros((o_act, o_act, n_vir, o_act), dtype=np.float64),
            w_vv=np.zeros((o_act, o_act, n_vir, n_vir), dtype=np.float64),
            c_f12=np.zeros((o_act, o_act, n_vir, n_vir), dtype=np.float64),
            fixed_geminal_energy=0.0,
            solve_lambda=False,
            max_bytes=(provider.budget_mib << 20),
        )
        t1_use = np.ascontiguousarray(sol["t1"], dtype=np.float64)
        t2_use = np.ascontiguousarray(sol["t2"], dtype=np.float64)
        ccsd_f12b_energy = float(sol["correlation_energy"])
    elif cuda:
        cc_res = spatial_rhf_ccsdf12b_v1(
            provider, pair_req, reference,
            forces=False,
            expected_cabs_rank=expected_cabs_rank,
            preconvergence=eff_preconv,
            return_amplitudes=True,
        )
        assert cc_res.t1 is not None and cc_res.t2 is not None
        t1_use = np.ascontiguousarray(cc_res.t1, dtype=np.float64)
        t2_use = np.ascontiguousarray(cc_res.t2, dtype=np.float64)
        ccsd_f12b_energy = float(cc_res.correlation_energy)
        cabs_rank = int(cc_res.cabs_rank)
    else:
        from ._f12_focal_rcc_f12b import _build_spatial_rhf_fixed_sources

        w_vo, w_vv, c_f12, fixed_e, cabs_rank, _ = _build_spatial_rhf_fixed_sources(
            provider, pair_req, reference,
            expected_cabs_rank=expected_cabs_rank,
            derivatives=False,
        )
        oooo = np.einsum("qij,qkl->ijkl", boo, boo, optimize=True)
        oovv = np.einsum("qij,qab->ijab", boo, bvv, optimize=True)
        ovvo = ovov.transpose(0, 1, 3, 2).copy()
        ovvv = np.einsum("qia,qbc->iabc", bov, bvv, optimize=True)
        vvvv = np.einsum("qab,qcd->abcd", bvv, bvv, optimize=True)
        foo = np.ascontiguousarray(fock_mo[frozen:o_full, frozen:o_full], dtype=np.float64)
        fvv = np.ascontiguousarray(fock_mo[o_full:obs, o_full:obs], dtype=np.float64)
        sol = spatial_rhf_ccsdf12b_cpu_solve(
            foo=foo, fov=fov, fvv=fvv, ovov=ovov, ovvo=ovvo, oovv=oovv,
            ovvv=ovvv, ovoo=ovoo, oooo=oooo, vvvv=vvvv, eps_o=eps_o, eps_v=eps_v,
            w_vo=w_vo, w_vv=w_vv, c_f12=c_f12, fixed_geminal_energy=fixed_e,
            solve_lambda=False,
            max_bytes=(provider.budget_mib << 20),
        )
        t1_use = np.ascontiguousarray(sol["t1"], dtype=np.float64)
        t2_use = np.ascontiguousarray(sol["t2"], dtype=np.float64)
        ccsd_f12b_energy = float(sol["correlation_energy"])
    parent_s = time.monotonic() - t_parent_start
    del boo

    # 3. Project F12Fit 3-index MO tensors (or skip when zero_coupling=True).
    t_f12_proj_start = time.monotonic()
    ext = obs
    q_f12 = 0
    b_g_ext_obs = b_g_ext_ext = fit_ext_obs = None
    r_f_ext_obs = r_fg_obs_obs = r_ff_ext_obs = r_u_obs_obs = None
    f_ext = k_ext = None

    if not zero_coupling:
        if basis.f12_fit is None or basis.cabs is None or basis.beta is None:
            raise ValueError("Explicit (T)-F12 requires F12Fit, CABS (OptRI), and beta")
        checked = _make_checked_runner(provider, cuda)
        one, (jk, _mp2, corr) = provider._cached_pair_sources(
            pair_req, derivatives=False, budget=provider.budget_mib
        )
        space, ao, panel, proj = canonical_pair_frame(
            pair_req,
            reference,
            one,
            jk.joint(pair_req, explicit=False),
            corr.joint(pair_req, explicit=True),
            expected_cabs_rank=expected_cabs_rank,
            relative_cutoff=provider.relative_cutoff,
            library=provider.response_library,
        )
        _, _, _, ext, q_f12 = space
        cabs_rank = int(proj.rank)
        fock = extended_fock_response(
            space, ao, panel, checked, cuda=cuda, library=provider.response_library,
            native_library=provider.shell_library, host_reference=not cuda,
        )[0]
        f_ext = np.ascontiguousarray(fock["fock_alpha"], dtype=np.float64)
        k_ext = np.ascontiguousarray(fock["exchange_alpha"], dtype=np.float64)
        del fock

        c_ext = np.ascontiguousarray(ao["coefficients_alpha"], dtype=np.float64)  # [nc, ext]
        c_obs_ext = np.ascontiguousarray(c_ext[:, :obs], dtype=np.float64)        # [nc, obs]
        w_f12 = np.ascontiguousarray(corr.whitener, dtype=np.float64)
        ao_loader = ao.get("__ao_loader__")

        def load_ao_op(ao_key: str) -> np.ndarray:
            if ao_key in ao:
                return ao.pop(ao_key)
            assert ao_loader is not None
            return ao_loader(ao_key)

        # Operator 1: Coulomb -> b_g_ext_obs, fit_ext_obs, and (if Z2 active) b_g_ext_ext
        # Apply whiteners in-place in column tiles and free AO tensors immediately to keep host RAM bounded.
        ao_coul = load_ao_op("ao_coulomb")
        b_g_ext_obs = _project_ao_panel_tiled(ao_coul, c_ext, c_obs_ext)
        b_g_ext_ext = _project_ao_panel_tiled(ao_coul, c_ext, c_ext) if need_z2 else None
        del ao_coul
        _apply_metric_left_inplace(w_f12, b_g_ext_obs)
        fit_ext_obs = _apply_metric_left(w_f12.T, b_g_ext_obs)
        if b_g_ext_ext is not None:
            _apply_metric_left_inplace(w_f12, b_g_ext_ext)
        fit_obs_obs = np.ascontiguousarray(fit_ext_obs[:, :obs, :], dtype=np.float64)

        # Operator 2: f12 -> r_f_ext_obs = raw_f12 - 0.5 * M_f12 @ fit_ext_obs
        ao_f12 = load_ao_op("ao_f12")
        r_f_ext_obs = _project_ao_panel_tiled(ao_f12, c_ext, c_obs_ext)
        del ao_f12
        _apply_metric_sub_inplace(ao["metric/geminal"], fit_ext_obs, r_f_ext_obs, alpha=-0.5)

        # Operator 3: f12_squared -> r_ff_ext_obs
        ao_ff = load_ao_op("ao_f12_squared")
        r_ff_ext_obs = _project_ao_panel_tiled(ao_ff, c_ext, c_obs_ext)
        del ao_ff
        _apply_metric_sub_inplace(ao["metric/squared_geminal"], fit_ext_obs, r_ff_ext_obs, alpha=-0.5)

        # Operator 4: f12_over_coulomb -> r_fg_obs_obs
        ao_fg = load_ao_op("ao_f12_over_coulomb")
        r_fg_obs_obs = _project_ao_panel_tiled(ao_fg, c_obs_ext, c_obs_ext)
        del ao_fg
        _apply_metric_sub_inplace(ao["metric/geminal_coulomb"], fit_obs_obs, r_fg_obs_obs, alpha=-0.5)

        # Operator 5: double_commutator -> r_u_obs_obs
        ao_u = load_ao_op("ao_double_commutator")
        r_u_obs_obs = _project_ao_panel_tiled(ao_u, c_obs_ext, c_obs_ext)
        del ao_u
        _apply_metric_sub_inplace(ao["metric/double_commutator"], fit_obs_obs, r_u_obs_obs, alpha=-0.5)
        del fit_obs_obs, ao, panel, one, jk, corr
        gc.collect()

    f12_proj_s = time.monotonic() - t_f12_proj_start
    return PreparedExplicitTf12Inputs(
        backend=request.backend,
        o_full=o_full,
        frozen=frozen,
        obs=obs,
        ext=ext,
        q_mp2=q_mp2,
        q_f12=q_f12,
        cabs_rank=cabs_rank,
        ccsd_f12b_energy=ccsd_f12b_energy,
        zero_coupling=zero_coupling,
        bov=bov,
        bvv=bvv,
        ovoo=ovoo,
        ovov=ovov,
        fov=fov,
        t1=t1_use,
        t2=t2_use,
        eps_o=eps_o,
        eps_v=eps_v,
        b_g_ext_obs=b_g_ext_obs,
        b_g_ext_ext=b_g_ext_ext,
        fit_ext_obs=fit_ext_obs,
        r_f_ext_obs=r_f_ext_obs,
        r_fg_obs_obs=r_fg_obs_obs,
        r_ff_ext_obs=r_ff_ext_obs,
        r_u_obs_obs=r_u_obs_obs,
        f_ext=f_ext,
        k_ext=k_ext,
        mp2_proj_seconds=mp2_proj_s,
        parent_seconds=parent_s,
        f12_proj_seconds=f12_proj_s,
    )


def evaluate_prepared_explicit_tf12(
    prep: PreparedExplicitTf12Inputs,
    *,
    z2_mode: str = "ffg",
    geminal_scale: float = 1.0,
    triples_fp32_w: bool = False,
    backend: str | None = None,
    device: int = 0,
    max_bytes: int = 24 << 30,
) -> ExplicitTf12Result:
    """Execute the native CPU or DF-CUDA (T)-F12 kernel on pre-projected 3-index MO tensors."""
    use_backend = prep.backend if backend is None else backend
    if use_backend not in ("cuda", "cpu_reference"):
        raise ValueError(f"Unknown explicit triples execution backend {use_backend!r}")
    cuda = use_backend == "cuda"
    eff_z2 = "zero" if prep.zero_coupling else z2_mode
    t_exec_start = time.monotonic()
    if cuda:
        energy, breakdown = explicit_tf12_df_cuda(
            device=device,
            max_bytes=int(max_bytes),
            triples_fp32_w=bool(triples_fp32_w),
            o_full=prep.o_full,
            frozen=prep.frozen,
            obs=prep.obs,
            ext=prep.ext,
            q_mp2=prep.q_mp2,
            q_f12=prep.q_f12,
            bov=prep.bov,
            bvv=prep.bvv,
            ovoo=prep.ovoo,
            ovov=prep.ovov,
            fov=prep.fov,
            t1=prep.t1,
            t2=prep.t2,
            eps_o=prep.eps_o,
            eps_v=prep.eps_v,
            b_g_ext_obs=prep.b_g_ext_obs,
            b_g_ext_ext=prep.b_g_ext_ext,
            fit_ext_obs=prep.fit_ext_obs,
            r_f_ext_obs=prep.r_f_ext_obs,
            r_fg_obs_obs=prep.r_fg_obs_obs,
            r_ff_ext_obs=prep.r_ff_ext_obs,
            r_u_obs_obs=prep.r_u_obs_obs,
            f_ext=prep.f_ext,
            k_ext=prep.k_ext,
            z2_mode=eff_z2,
            geminal_scale=float(geminal_scale),
        )
    else:
        energy, breakdown = explicit_tf12_df_cpu(
            o_full=prep.o_full,
            frozen=prep.frozen,
            obs=prep.obs,
            ext=prep.ext,
            q_mp2=prep.q_mp2,
            q_f12=prep.q_f12,
            bov=prep.bov,
            bvv=prep.bvv,
            ovoo=prep.ovoo,
            ovov=prep.ovov,
            fov=prep.fov,
            t1=prep.t1,
            t2=prep.t2,
            eps_o=prep.eps_o,
            eps_v=prep.eps_v,
            b_g_ext_obs=prep.b_g_ext_obs,
            b_g_ext_ext=prep.b_g_ext_ext,
            fit_ext_obs=prep.fit_ext_obs,
            r_f_ext_obs=prep.r_f_ext_obs,
            r_fg_obs_obs=prep.r_fg_obs_obs,
            r_ff_ext_obs=prep.r_ff_ext_obs,
            r_u_obs_obs=prep.r_u_obs_obs,
            f_ext=prep.f_ext,
            k_ext=prep.k_ext,
            z2_mode=eff_z2,
            geminal_scale=float(geminal_scale),
        )
    exec_s = time.monotonic() - t_exec_start
    if not math.isfinite(energy):
        raise RuntimeError("Nonfinite spatial RHF (T)-F12 energy")
    o_act = prep.o_full - prep.frozen
    n_vir = prep.obs - prep.o_full
    telemetry: dict[str, float | int | bool | str] = {
        "backend": use_backend,
        "input_backend": prep.backend,
        "execution_mode": "hybrid_host_assisted" if cuda else "host",
        "forward_w_precision": ("fp32" if breakdown["cuda_w_fp32_gemms"] else "fp64"),
        "reverse_w_precision": "n/a",
        "cuda_w_fp32_gemms": int(breakdown["cuda_w_fp32_gemms"]),
        "cuda_w_fp64_gemms": int(breakdown["cuda_w_fp64_gemms"]),
        "o_full": int(prep.o_full),
        "frozen_occupied": int(prep.frozen),
        "o_act": int(o_act),
        "n_vir": int(n_vir),
        "obs": int(prep.obs),
        "ext": int(prep.ext),
        "cabs_rank": int(prep.cabs_rank),
        "q_mp2": int(prep.q_mp2),
        "q_f12": int(prep.q_f12),
        "z2_mode": eff_z2,
        "geminal_scale": float(geminal_scale),
        "zero_coupling": bool(prep.zero_coupling),
        "triples_fp32_w_requested": bool(triples_fp32_w),
        "triples_fp32_w_executed": bool(breakdown["cuda_w_fp32_gemms"]),
        "mp2_proj_seconds": float(prep.mp2_proj_seconds),
        "parent_seconds": float(prep.parent_seconds),
        "f12_proj_seconds": float(prep.f12_proj_seconds),
        "kernel_seconds": float(exec_s),
        "seconds_pt": float(breakdown["seconds_pt"]),
        "seconds_a_vxbbh": float(breakdown["seconds_a_vxbbh"]),
        "seconds_z2": float(breakdown["seconds_z2"]),
        "peak_workspace_bytes": int(breakdown["peak_workspace_bytes"]),
        "wall_seconds": float(prep.mp2_proj_seconds + prep.parent_seconds + prep.f12_proj_seconds + exec_s),
    }
    return ExplicitTf12Result(
        energy=energy,
        breakdown=breakdown,
        ccsd_f12b_energy=prep.ccsd_f12b_energy,
        cabs_rank=prep.cabs_rank,
        telemetry=telemetry,
    )


def spatial_rhf_explicit_tf12_energy(
    provider: MolecularFocalProvider,
    request: FocalRequest,
    reference: FocalReference,
    *,
    ccsd_f12b_result: SpatialCcsdF12bResult | None = None,
    t1: np.ndarray | None = None,
    t2: np.ndarray | None = None,
    expected_cabs_rank: int | None = None,
    z2_mode: str = "ffg",
    geminal_scale: float | None = None,
    preconvergence: str | None = None,
    triples_fp32_w: bool | None = None,
    zero_coupling: bool = False,
    device: int = 0,
    max_bytes: int | None = None,
) -> ExplicitTf12Result:
    """Evaluate the physical RHF model at unit scale on the provider's Slater convention.

    The optional geminal_scale multiplies only the triples operators for a
    fixed-parent sensitivity probe. A different physical normalization requires
    a matching molecular request and provider, including the CCSD-F12b parent.
    """
    from ._f12_frozen_core import resolve_frozen_occupied

    started = time.monotonic()
    eff_scale = 1.0 if geminal_scale is None else float(geminal_scale)
    eff_fp32_w = provider.triples_fp32_w if triples_fp32_w is None else bool(triples_fp32_w)
    prep = prepare_spatial_rhf_explicit_tf12_inputs(
        provider,
        request,
        reference,
        ccsd_f12b_result=ccsd_f12b_result,
        t1=t1,
        t2=t2,
        expected_cabs_rank=expected_cabs_rank,
        need_z2=(not zero_coupling and z2_mode != "zero"),
        preconvergence=preconvergence,
        zero_coupling=zero_coupling,
    )
    if prep is None:
        o_full = reference.occupied[0]
        frozen = resolve_frozen_occupied(request, (o_full, o_full))[0]
        from ._f12_focal_projector import checked_reference_cabs_rank

        rank = checked_reference_cabs_rank(provider, request, reference, expected_cabs_rank)
        return ExplicitTf12Result(
            0.0,
            {k: 0.0 for k in BREAKDOWN_KEYS} | {"delta_c2": 0.0},
            0.0,
            rank,
            {"frozen_occupied": frozen, "wall_seconds": time.monotonic() - started,
             "backend": reference.executed_backend, "execution_mode": "empty_active_space",
             "forward_w_precision": "n/a", "reverse_w_precision": "n/a"},
        )
    budget_bytes = (
        int(max_bytes)
        if max_bytes is not None
        else provider.budget_mib << 20
    )
    return evaluate_prepared_explicit_tf12(
        prep,
        z2_mode=z2_mode,
        geminal_scale=eff_scale,
        triples_fp32_w=eff_fp32_w,
        device=device,
        max_bytes=budget_bytes,
    )
