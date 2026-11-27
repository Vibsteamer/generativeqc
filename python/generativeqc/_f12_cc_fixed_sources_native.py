"""Native GPU-resident cuBLAS (and host-reference) bindings for fixed-cusp CC-F12 helpers."""

from __future__ import annotations

import ctypes as ct
import functools
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._f12_shell_binding import _loaded_sink_paths, resolve_shell_binding
from ._f12_workspace import workspace_bytes

if TYPE_CHECKING:
    from collections.abc import Callable

_D_PTR = ct.POINTER(ct.c_double)
_I_PTR = ct.POINTER(ct.c_int32)

REQUIRE_AOT_CCFIXED_ENV = "GENERATIVEQC_F12_REQUIRE_AOT_CCFIXED"
DISABLE_AOT_CCFIXED_ENV = "GENERATIVEQC_F12_DISABLE_AOT_CCFIXED"
SHELL_LIBRARY_ENV = "GENERATIVEQC_F12_SHELL_LIBRARY"
HOST_SINK_ENV = "GENERATIVEQC_F12_HOST_SINK"
AUTO_NATIVE_LIBRARY = object()

def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def disable_aot_ccfixed() -> bool:
    return _env_truthy(DISABLE_AOT_CCFIXED_ENV)


def require_aot_ccfixed() -> bool:
    """Return True when strict AOT-only execution is required for CC fixed-source paths."""
    req = _env_truthy(REQUIRE_AOT_CCFIXED_ENV)
    dis = disable_aot_ccfixed()
    if req and dis:
        raise ValueError(
            f"Conflicting environment flags: {REQUIRE_AOT_CCFIXED_ENV} and "
            f"{DISABLE_AOT_CCFIXED_ENV} cannot both be enabled"
        )
    return req


def _dptr(arr: np.ndarray | None):
    if arr is None:
        return ct.cast(0, _D_PTR)
    return arr.ctypes.data_as(_D_PTR)


def _iptr(arr: np.ndarray):
    return arr.ctypes.data_as(_I_PTR)


@functools.lru_cache(maxsize=16)
def _load_cdll(resolved_path: str) -> ct.CDLL:
    return ct.CDLL(resolved_path)


def bind_fixed_pair_sources(library: Path | None, host_reference: bool):
    """Return bound ctypes callable for gqc_f12_cc_fixed_pair_sources_{host_reference,cuda}."""
    if library is None:
        return None
    suffix = "_host_reference" if host_reference else "_cuda"
    cdll = _load_cdll(str(Path(library).resolve()))
    func = getattr(cdll, f"gqc_f12_cc_fixed_pair_sources{suffix}", None)
    if func is None:
        return None
    func.argtypes = [ct.c_int] * 11 + [_I_PTR] * 2 + [_D_PTR] * 26 + [ct.c_int64]
    func.restype = ct.c_int
    return func


def discover_ccfixed_library(
    *,
    host_reference: bool | None = None,
    prefer_host: bool = False,
) -> tuple[Path | None, bool]:
    """Auto-discover a loaded or env-configured library for ONE single backend only."""
    require_aot_ccfixed()
    target_host = prefer_host if host_reference is None else bool(host_reference)
    if disable_aot_ccfixed():
        return None, target_host

    suffix = "_host_reference" if target_host else "_cuda"

    def fixed_sources_library(lib: ct.CDLL) -> Path | None:
        if (hasattr(lib, "gqc_f12_cc_fixed_pair_sources" + suffix)
                and hasattr(lib, "gqc_f12_cc_coulomb_fit_fwd" + suffix)):
            return Path(lib._name)
        return None

    path = resolve_shell_binding(
        None, fixed_sources_library, host_reference=target_host,
        env_keys=(SHELL_LIBRARY_ENV, HOST_SINK_ENV) if target_host else (SHELL_LIBRARY_ENV,),
        mapped_paths=_loaded_sink_paths,
    )
    return path, target_host


def resolve_ccfixed_library(
    native_library: object,
    host_reference: bool | None,
    run: Callable,
) -> tuple[Path | None, bool]:
    """Resolve (library_path, host_reference) strictly for one backend without cross-fallback."""
    require_aot_ccfixed()
    from ._f12_ao_mo_native import detect_host_reference

    target_host = (
        bool(host_reference)
        if host_reference is not None
        else detect_host_reference(run, None, None)
    )
    if disable_aot_ccfixed():
        return None, target_host
    if native_library is None:
        return None, target_host
    if native_library is AUTO_NATIVE_LIBRARY:
        run_lib = getattr(run, "shell_library", None)
        if run_lib is not None:
            p = Path(run_lib)
            return (p if p.is_file() else None), target_host
        return discover_ccfixed_library(
            host_reference=target_host, prefer_host=target_host
        )
    lib_path = Path(native_library)  # type: ignore[arg-type]
    return lib_path, target_host


def run_fixed_pair_sources_native(
    func,
    *,
    same_spin: bool,
    compute_wc: bool,
    compute_quad: bool,
    nl: int,
    nr: int,
    fl: int,
    fr: int,
    obs: int,
    ext: int,
    aux: int,
    pairs: list[tuple[int, int]],
    pair_mult: np.ndarray | None = None,
    fit_l: np.ndarray,
    fit_r: np.ndarray | None,
    raw_coul_r: np.ndarray,
    raw_g_l: np.ndarray,
    raw_g_r: np.ndarray | None,
    raw_gc_l: np.ndarray,
    raw_gc_r: np.ndarray | None,
    metric_g: np.ndarray,
    metric_gc: np.ndarray,
    raw_sq_l: np.ndarray | None = None,
    raw_sq_r: np.ndarray | None = None,
    raw_dc_l: np.ndarray | None = None,
    raw_dc_r: np.ndarray | None = None,
    metric_sq: np.ndarray | None = None,
    metric_dc: np.ndarray | None = None,
    fock_l: np.ndarray | None = None,
    fock_r: np.ndarray | None = None,
    exch_l: np.ndarray | None = None,
    exch_r: np.ndarray | None = None,
    cross_overlap: np.ndarray | None = None,
    fit_obs_l: np.ndarray | None = None,
    raw_obs_r: np.ndarray | None = None,
    max_workspace_bytes: int = 0,
) -> tuple[np.ndarray | None, np.ndarray | None, float, float, float]:
    """Execute batched fixed-cusp CC-F12 pair sources on GPU (or host reference)."""
    max_workspace_bytes = workspace_bytes(max_workspace_bytes, signed=True)
    n_pairs = len(pairs)
    nv_l, nv_r = obs - nl, obs - nr
    if n_pairs == 0:
        w_empty = np.empty((0, obs, obs), dtype=np.float64) if compute_wc else None
        c_empty = np.empty((0, nv_l, nv_r), dtype=np.float64) if compute_wc else None
        return w_empty, c_empty, 0.0, 0.0, 0.0

    def _act_t3(arr: np.ndarray | None, start: int, stop: int) -> np.ndarray | None:
        if arr is None:
            return None
        return np.ascontiguousarray(arr[:, start:stop, :].transpose(1, 0, 2), dtype=np.float64)

    def _c64(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        return np.ascontiguousarray(arr, dtype=np.float64)

    pi = np.ascontiguousarray([p[0] for p in pairs], dtype=np.int32)
    pj = np.ascontiguousarray([p[1] for p in pairs], dtype=np.int32)
    pm = _c64(pair_mult)
    if pm is not None and pm.shape != (n_pairs,):
        raise ValueError("pair_mult must be a 1D array matching len(pairs)")

    fl_t = _act_t3(fit_l, fl, nl)
    fr_t = fl_t if same_spin else _act_t3(fit_r, fr, nr)
    rcr_t = _act_t3(raw_coul_r, fr, nr)
    gl_t = _act_t3(raw_g_l, fl, nl)
    gr_t = gl_t if same_spin else _act_t3(raw_g_r, fr, nr)
    gcl_t = _act_t3(raw_gc_l, fl, nl)
    gcr_t = gcl_t if same_spin else _act_t3(raw_gc_r, fr, nr)
    sql_t = _act_t3(raw_sq_l, fl, nl) if compute_quad else None
    sqr_t = (sql_t if same_spin else _act_t3(raw_sq_r, fr, nr)) if compute_quad else None
    dcl_t = _act_t3(raw_dc_l, fl, nl) if compute_quad else None
    dcr_t = (dcl_t if same_spin else _act_t3(raw_dc_r, fr, nr)) if compute_quad else None

    mg = _c64(metric_g)
    mgc = _c64(metric_gc)
    msq = _c64(metric_sq) if compute_quad else None
    mdc = _c64(metric_dc) if compute_quad else None
    f_l = _c64(fock_l) if (compute_wc or compute_quad) else None
    f_r = (f_l if same_spin else _c64(fock_r)) if (compute_wc or compute_quad) else None
    k_l = _c64(exch_l) if compute_quad else None
    k_r = (k_l if same_spin else _c64(exch_r)) if compute_quad else None
    co = None if same_spin else _c64(cross_overlap)
    fo_l = _c64(fit_obs_l) if compute_wc else None
    ro_r = _c64(raw_obs_r) if compute_wc else None

    w_out = np.empty((n_pairs, obs, obs), dtype=np.float64) if compute_wc else None
    c_out = np.empty((n_pairs, nv_l, nv_r), dtype=np.float64) if compute_wc else None
    energies = np.zeros(3, dtype=np.float64)

    rc = func(
        int(same_spin),
        int(compute_wc),
        int(compute_quad),
        int(nl),
        int(nr),
        int(fl),
        int(fr),
        int(obs),
        int(ext),
        int(aux),
        int(n_pairs),
        _iptr(pi),
        _iptr(pj),
        _dptr(pm),
        _dptr(fl_t),
        _dptr(fr_t),
        _dptr(rcr_t),
        _dptr(gl_t),
        _dptr(gr_t),
        _dptr(gcl_t),
        _dptr(gcr_t),
        _dptr(sql_t),
        _dptr(sqr_t),
        _dptr(dcl_t),
        _dptr(dcr_t),
        _dptr(mg),
        _dptr(mgc),
        _dptr(msq),
        _dptr(mdc),
        _dptr(f_l),
        _dptr(f_r),
        _dptr(k_l),
        _dptr(k_r),
        _dptr(co),
        _dptr(fo_l),
        _dptr(ro_r),
        _dptr(w_out),
        _dptr(c_out),
        _dptr(energies),
        max_workspace_bytes,
    )
    if rc != 0:
        raise RuntimeError(f"native CC fixed pair sources failed with status {rc}")
    return w_out, c_out, float(energies[0]), float(energies[1]), float(energies[2])


def bind_coulomb_fit_fwd(library: Path | None, host_reference: bool):
    """Return bound ctypes callable for gqc_f12_cc_coulomb_fit_fwd_{host_reference,cuda}."""
    if library is None:
        return None
    suffix = "_host_reference" if host_reference else "_cuda"
    cdll = _load_cdll(str(Path(library).resolve()))
    func = getattr(cdll, f"gqc_f12_cc_coulomb_fit_fwd{suffix}", None)
    if func is None:
        return None
    func.argtypes = [
        ct.c_int,
        ct.c_int64,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        ct.c_int64,
    ]
    func.restype = ct.c_int
    return func


def bind_coulomb_fit_rev(library: Path | None, host_reference: bool):
    """Return bound ctypes callable for gqc_f12_cc_coulomb_fit_rev_{host_reference,cuda}."""
    if library is None:
        return None
    suffix = "_host_reference" if host_reference else "_cuda"
    cdll = _load_cdll(str(Path(library).resolve()))
    func = getattr(cdll, f"gqc_f12_cc_coulomb_fit_rev{suffix}", None)
    if func is None:
        return None
    func.argtypes = [
        ct.c_int,
        ct.c_int64,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        ct.c_int64,
    ]
    func.restype = ct.c_int
    return func


def run_coulomb_fit_fwd_native(
    func,
    whitener: np.ndarray,
    three_center: np.ndarray,
    *,
    max_workspace_bytes: int = 0,
) -> np.ndarray:
    """Compute fit = W^T @ (W @ three_center) via column-chunked native cuBLAS/host kernel."""
    max_workspace_bytes = workspace_bytes(max_workspace_bytes, signed=True)
    w = np.ascontiguousarray(whitener, dtype=np.float64)
    tc = np.ascontiguousarray(three_center, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("whitener must be a square 2D FP64 matrix")
    aux = int(w.shape[0])
    if tc.ndim < 2 or tc.shape[0] != aux:
        raise ValueError("three_center leading axis must match whitener dimension")
    ncols = int(tc.size // aux)
    if ncols <= 0:
        raise ValueError("three_center must have at least one RHS column")
    fit_out = np.empty_like(tc)
    rc = func(
        aux,
        ncols,
        _dptr(w),
        _dptr(tc),
        _dptr(fit_out),
        max_workspace_bytes,
    )
    if rc != 0:
        raise RuntimeError(f"native CC Coulomb fit forward failed with status {rc}")
    return fit_out


def run_coulomb_fit_rev_native(
    func,
    whitener: np.ndarray,
    fit_bar: np.ndarray,
    *,
    three_center: np.ndarray | None = None,
    fit_precomputed: np.ndarray | None = None,
    max_workspace_bytes: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (three_center_bar, metric_bar) via column-chunked native cuBLAS/host kernel."""
    max_workspace_bytes = workspace_bytes(max_workspace_bytes, signed=True)
    if three_center is None and fit_precomputed is None:
        raise ValueError("coulomb fit reverse requires three_center or fit_precomputed")
    w = np.ascontiguousarray(whitener, dtype=np.float64)
    fb = np.ascontiguousarray(fit_bar, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("whitener must be a square 2D FP64 matrix")
    aux = int(w.shape[0])
    if fb.ndim < 2 or fb.shape[0] != aux:
        raise ValueError("fit_bar leading axis must match whitener dimension")
    ncols = int(fb.size // aux)
    if ncols <= 0:
        raise ValueError("fit_bar must have at least one RHS column")
    tc = (
        None
        if three_center is None
        else np.ascontiguousarray(three_center, dtype=np.float64)
    )
    fp = (
        None
        if fit_precomputed is None
        else np.ascontiguousarray(fit_precomputed, dtype=np.float64)
    )
    if tc is not None and tc.shape != fb.shape:
        raise ValueError("three_center and fit_bar must share identical shape")
    if fp is not None and fp.shape != fb.shape:
        raise ValueError("fit_precomputed and fit_bar must share identical shape")
    tc_bar = np.empty_like(fb)
    metric_bar = np.empty((aux, aux), dtype=np.float64)
    rc = func(
        aux,
        ncols,
        _dptr(w),
        _dptr(tc),
        _dptr(fp),
        _dptr(fb),
        _dptr(tc_bar),
        _dptr(metric_bar),
        max_workspace_bytes,
    )
    if rc != 0:
        raise RuntimeError(f"native CC Coulomb fit reverse failed with status {rc}")
    return tc_bar, metric_bar


def coulomb_fit_forward(
    whitener: np.ndarray,
    three_center: np.ndarray,
    *,
    native_library: Path | None,
    host_reference: bool,
    run: Callable,
    max_workspace_bytes: int = 0,
) -> np.ndarray:
    """Compute fit = W^T @ (W @ three_center) via AOT native kernel or TensorIR fallback."""
    native_fn = bind_coulomb_fit_fwd(native_library, host_reference)
    if native_fn is not None:
        return run_coulomb_fit_fwd_native(
            native_fn,
            whitener,
            three_center,
            max_workspace_bytes=max_workspace_bytes,
        )
    if require_aot_ccfixed():
        raise RuntimeError(
            f"{REQUIRE_AOT_CCFIXED_ENV}=1 forbids TensorIR fallback in coulomb_fit_forward "
            "(native_library missing or gqc_f12_cc_coulomb_fit_fwd symbol unavailable)"
        )
    from generativeqc_compiler.mp2.f12_density_fitting import coulomb_fit_response
    from generativeqc_compiler.tensor import Program

    aux = int(whitener.shape[0])
    rhs = np.ascontiguousarray(three_center, dtype=np.float64).reshape(aux, -1)
    solve = coulomb_fit_response(aux, int(rhs.shape[1]))
    fit_forward = Program({"fit": solve.outputs["fit"]})
    out = run(
        fit_forward,
        {"metric_whitener": whitener, "three_center": rhs},
    ).outputs["fit"]
    return out.reshape(three_center.shape)


def coulomb_fit_reverse(
    whitener: np.ndarray,
    three_center: np.ndarray,
    fit_bar: np.ndarray,
    *,
    fit_precomputed: np.ndarray | None = None,
    native_library: Path | None,
    host_reference: bool,
    run: Callable,
    max_workspace_bytes: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (three_center_bar, metric_bar) via AOT native kernel or TensorIR fallback."""
    native_fn = bind_coulomb_fit_rev(native_library, host_reference)
    if native_fn is not None:
        return run_coulomb_fit_rev_native(
            native_fn,
            whitener,
            fit_bar,
            three_center=three_center,
            fit_precomputed=fit_precomputed,
            max_workspace_bytes=max_workspace_bytes,
        )
    if require_aot_ccfixed():
        raise RuntimeError(
            f"{REQUIRE_AOT_CCFIXED_ENV}=1 forbids TensorIR fallback in coulomb_fit_reverse "
            "(native_library missing or gqc_f12_cc_coulomb_fit_rev symbol unavailable)"
        )
    from generativeqc_compiler.mp2.f12_density_fitting import coulomb_fit_response
    from generativeqc_compiler.tensor import Program

    aux = int(whitener.shape[0])
    rhs = np.ascontiguousarray(three_center, dtype=np.float64).reshape(aux, -1)
    fb = np.ascontiguousarray(fit_bar, dtype=np.float64).reshape(aux, -1)
    solve = coulomb_fit_response(aux, int(rhs.shape[1]))
    fit_reverse = Program(
        {k: solve.outputs[k] for k in ("three_center_bar", "metric_bar")}
    )
    out = run(
        fit_reverse,
        {"metric_whitener": whitener, "three_center": rhs, "fit_bar": fb},
    ).outputs
    return out["three_center_bar"].reshape(fit_bar.shape), out["metric_bar"]
