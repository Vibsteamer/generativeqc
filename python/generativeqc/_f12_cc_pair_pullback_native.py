"""Native GPU-resident cuBLAS (and host-reference) bindings for MO-space CC-F12 pair pullbacks."""

from __future__ import annotations

import ctypes as ct
from pathlib import Path

import numpy as np

from ._f12_workspace import workspace_bytes

_D_PTR = ct.POINTER(ct.c_double)
_I_PTR = ct.POINTER(ct.c_int32)
_C_INT_MAX = (1 << 31) - 1


def _dptr(arr: np.ndarray | None):
    if arr is None:
        return ct.cast(0, _D_PTR)
    return arr.ctypes.data_as(_D_PTR)


def _iptr(arr: np.ndarray):
    return arr.ctypes.data_as(_I_PTR)


def _check_c_int(value: int, name: str, *, low: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    ival = int(value)
    if ival < low or ival > _C_INT_MAX:
        raise ValueError(f"{name}={ival} exceeds C int range [{low}, {_C_INT_MAX}]")
    return ival


def _check_budget(budget_bytes: int) -> int:
    if isinstance(budget_bytes, bool):
        raise ValueError("budget_bytes must be an integer, not bool")
    return workspace_bytes(budget_bytes, signed=False)


def checked_device_budget(budget_bytes: int, *, host_reference: bool) -> int:
    """Validate the endpoint budget for a bound pair pullback.

    The CUDA ABI maps 0 to a fixed 4 GiB device cap that ignores the endpoint
    budget, so CUDA callers must pass a positive value. Host-reference kernels
    ignore it.
    """
    value = _check_budget(budget_bytes)
    if value == 0 and not host_reference:
        raise ValueError(
            "CUDA CC-F12 pair pullbacks require an explicit positive budget_bytes; "
            "0 would select the fixed 4 GiB native default instead of the endpoint budget"
        )
    return value


def _validate_f64_array(
    arr: np.ndarray | None,
    expected_shape: tuple[int, ...],
    name: str,
    *,
    allow_min_axis1: int | None = None,
) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}")
    if arr.dtype == np.bool_ or arr.dtype.kind not in ("f", "i", "u"):
        raise ValueError(f"{name} must have a real numeric dtype, got {arr.dtype}")
    if allow_min_axis1 is not None:
        if (
            arr.ndim != 3
            or arr.shape[0] != expected_shape[0]
            or arr.shape[1] < allow_min_axis1
            or arr.shape[2] != expected_shape[2]
        ):
            raise ValueError(
                f"{name} must have shape ({expected_shape[0]}, >={allow_min_axis1}, "
                f"{expected_shape[2]}), got {arr.shape}"
            )
    elif arr.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if not out.flags.c_contiguous or out.dtype != np.float64:
        raise ValueError(f"{name} failed contiguous float64 normalization")
    if not np.isfinite(out).all():
        raise ValueError(f"Nonfinite input in {name} passed to native pair pullback")
    return out


def _validate_writable_output(arr: np.ndarray, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Output {name} must be a numpy.ndarray")
    if arr.dtype != np.float64 or not arr.flags.c_contiguous or not arr.flags.writeable:
        raise ValueError(f"Output {name} must be a writable C-contiguous float64 array")
    if arr.shape != expected_shape:
        raise ValueError(f"Output {name} must have shape {expected_shape}, got {arr.shape}")
    return arr


def _validate_pairs(
    pairs: list[tuple[int, int]],
    *,
    i_low: int,
    i_high: int,
    j_low: int,
    j_high: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        raise ValueError("pairs must be non-empty for native pair pullback")
    _check_c_int(len(pairs), "n_pairs", low=1)
    pi_list: list[int] = []
    pj_list: list[int] = []
    for idx, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError(f"pairs[{idx}] must be a 2-tuple, got length {len(pair)}")
        i = _check_c_int(pair[0], f"pairs[{idx}][0]", low=i_low)
        j = _check_c_int(pair[1], f"pairs[{idx}][1]", low=j_low)
        if i >= i_high or j >= j_high:
            raise ValueError(
                f"pairs[{idx}]=({i}, {j}) out of valid range [{i_low}, {i_high}) x [{j_low}, {j_high})"
            )
        pi_list.append(i)
        pj_list.append(j)
    pi = np.ascontiguousarray(pi_list, dtype=np.int32)
    pj = np.ascontiguousarray(pj_list, dtype=np.int32)
    return pi, pj


def _transpose_occ_first(arr: np.ndarray, aux: int, n_occ: int, ext: int, name: str) -> np.ndarray:
    checked = _validate_f64_array(arr, (aux, n_occ, ext), name, allow_min_axis1=n_occ)
    out = np.ascontiguousarray(checked[:, :n_occ, :].transpose(1, 0, 2), dtype=np.float64)
    return out


def bind_pair_pullback(library: Path | None, kind: str, host_reference: bool):
    """Return bound ctypes callable for kind in ('w', 'c', 'scalar'), or None if absent."""
    if library is None:
        return None
    suffix = "_host_reference" if host_reference else "_cuda"
    cdll = ct.CDLL(str(Path(library).resolve()))
    func = getattr(cdll, f"gqc_f12_cc_{kind}_mo_pullback{suffix}", None)
    if func is None:
        return None
    if kind == "w":
        func.argtypes = (
            [ct.c_int] * 7
            + [ct.c_uint64, _I_PTR, _I_PTR]
            + [_D_PTR] * 23
        )
    elif kind == "c":
        func.argtypes = (
            [ct.c_int] * 7
            + [ct.c_uint64, _I_PTR, _I_PTR]
            + [_D_PTR] * 17
        )
    elif kind == "scalar":
        func.argtypes = (
            [ct.c_int] * 9
            + [ct.c_uint64, ct.c_double, _I_PTR, _I_PTR]
            + [_D_PTR] * 50
        )
    else:
        raise ValueError(f"Unknown pair pullback kind {kind!r}")
    func.restype = ct.c_int
    return func


def run_w_pullback_native(
    func,
    *,
    same_spin: bool,
    nl: int,
    nr: int,
    obs: int,
    ext: int,
    aux: int,
    pairs: list[tuple[int, int]],
    fit_obs_l: np.ndarray,
    raw_obs_r: np.ndarray,
    fit_l: np.ndarray,
    fit_r: np.ndarray,
    raw_g_l: np.ndarray,
    raw_g_r: np.ndarray,
    raw_gc_l: np.ndarray,
    raw_gc_r: np.ndarray,
    metric_g: np.ndarray,
    metric_gc: np.ndarray,
    cross_overlap: np.ndarray | None,
    bar_w_all: np.ndarray,
    budget_bytes: int = 0,
) -> dict[str, np.ndarray]:
    """Execute W MO-space pullback across all pairs on GPU (or host reference)."""
    nl = _check_c_int(nl, "nl", low=1)
    nr = _check_c_int(nr, "nr", low=1)
    obs = _check_c_int(obs, "obs", low=max(nl, nr))
    ext = _check_c_int(ext, "ext", low=obs + 1)
    aux = _check_c_int(aux, "aux", low=1)
    eff_budget = _check_budget(budget_bytes)
    if same_spin and nl != nr:
        raise ValueError(f"same_spin W pullback requires nl == nr, got nl={nl}, nr={nr}")
    pi, pj = _validate_pairs(pairs, i_low=0, i_high=nl, j_low=0, j_high=nr)

    fo_l = _validate_f64_array(fit_obs_l, (aux, obs, obs), "fit_obs_l")
    ro_r = _validate_f64_array(raw_obs_r, (aux, obs, obs), "raw_obs_r")
    fl_t = _transpose_occ_first(fit_l, aux, nl, ext, "fit_l")
    fr_t = fl_t if same_spin else _transpose_occ_first(fit_r, aux, nr, ext, "fit_r")
    gl_t = _transpose_occ_first(raw_g_l, aux, nl, ext, "raw_g_l")
    gr_t = gl_t if same_spin else _transpose_occ_first(raw_g_r, aux, nr, ext, "raw_g_r")
    gcl_t = _transpose_occ_first(raw_gc_l, aux, nl, ext, "raw_gc_l")
    gcr_t = gcl_t if same_spin else _transpose_occ_first(raw_gc_r, aux, nr, ext, "raw_gc_r")
    mg = _validate_f64_array(metric_g, (aux, aux), "metric_g")
    mgc = _validate_f64_array(metric_gc, (aux, aux), "metric_gc")
    co = None if same_spin else _validate_f64_array(cross_overlap, (ext, ext), "cross_overlap")
    bw = _validate_f64_array(bar_w_all, (len(pairs), obs, obs), "bar_w_all")

    b_fo_l = _validate_writable_output(np.zeros_like(fo_l), fo_l.shape, "bar_fit_obs_l")
    b_ro_r = _validate_writable_output(np.zeros_like(ro_r), ro_r.shape, "bar_raw_obs_r")
    b_fl_t = _validate_writable_output(np.zeros_like(fl_t), fl_t.shape, "bar_fit_l")
    b_fr_t = b_fl_t if same_spin else _validate_writable_output(np.zeros_like(fr_t), fr_t.shape, "bar_fit_r")
    b_gl_t = _validate_writable_output(np.zeros_like(gl_t), gl_t.shape, "bar_raw_g_l")
    b_gr_t = b_gl_t if same_spin else _validate_writable_output(np.zeros_like(gr_t), gr_t.shape, "bar_raw_g_r")
    b_gcl_t = _validate_writable_output(np.zeros_like(gcl_t), gcl_t.shape, "bar_raw_gc_l")
    b_gcr_t = b_gcl_t if same_spin else _validate_writable_output(np.zeros_like(gcr_t), gcr_t.shape, "bar_raw_gc_r")
    b_mg = _validate_writable_output(np.zeros_like(mg), mg.shape, "bar_metric_g")
    b_mgc = _validate_writable_output(np.zeros_like(mgc), mgc.shape, "bar_metric_gc")
    b_co = None if same_spin else _validate_writable_output(np.zeros_like(co), co.shape, "bar_cross_overlap")

    rc = func(
        int(same_spin),
        nl,
        nr,
        obs,
        ext,
        aux,
        len(pairs),
        eff_budget,
        _iptr(pi),
        _iptr(pj),
        _dptr(fo_l),
        _dptr(ro_r),
        _dptr(fl_t),
        _dptr(fr_t),
        _dptr(gl_t),
        _dptr(gr_t),
        _dptr(gcl_t),
        _dptr(gcr_t),
        _dptr(mg),
        _dptr(mgc),
        _dptr(co),
        _dptr(bw),
        _dptr(b_fo_l),
        _dptr(b_ro_r),
        _dptr(b_fl_t),
        _dptr(b_fr_t),
        _dptr(b_gl_t),
        _dptr(b_gr_t),
        _dptr(b_gcl_t),
        _dptr(b_gcr_t),
        _dptr(b_mg),
        _dptr(b_mgc),
        _dptr(b_co),
    )
    if rc != 0:
        raise RuntimeError(f"native CC W MO pullback failed with status {rc}")
    out = {
        "bar_fit_obs_l": b_fo_l,
        "bar_raw_obs_r": b_ro_r,
        "bar_fit_l": b_fl_t.transpose(1, 0, 2),
        "bar_raw_g_l": b_gl_t.transpose(1, 0, 2),
        "bar_raw_gc_l": b_gcl_t.transpose(1, 0, 2),
        "bar_metric_g": b_mg,
        "bar_metric_gc": b_mgc,
    }
    if not same_spin:
        out["bar_fit_r"] = b_fr_t.transpose(1, 0, 2)
        out["bar_raw_g_r"] = b_gr_t.transpose(1, 0, 2)
        out["bar_raw_gc_r"] = b_gcr_t.transpose(1, 0, 2)
        assert b_co is not None
        out["bar_cross_overlap"] = b_co
    for key, arr in out.items():
        if not np.isfinite(arr).all():
            raise RuntimeError(f"native CC W MO pullback produced nonfinite output in {key}")
    return out


def run_c_pullback_native(
    func,
    *,
    same_spin: bool,
    nl: int,
    nr: int,
    obs: int,
    ext: int,
    aux: int,
    pairs: list[tuple[int, int]],
    fit_l: np.ndarray,
    fit_r: np.ndarray,
    raw_g_l: np.ndarray,
    raw_g_r: np.ndarray,
    metric_g: np.ndarray,
    fock_l: np.ndarray,
    fock_r: np.ndarray,
    cross_overlap: np.ndarray | None,
    bar_c_all: np.ndarray,
    budget_bytes: int = 0,
) -> dict[str, np.ndarray]:
    """Execute C_vv MO-space pullback across all pairs on GPU (or host reference)."""
    nl = _check_c_int(nl, "nl", low=1)
    nr = _check_c_int(nr, "nr", low=1)
    obs = _check_c_int(obs, "obs", low=max(nl, nr) + 1)
    ext = _check_c_int(ext, "ext", low=obs + 1)
    aux = _check_c_int(aux, "aux", low=1)
    eff_budget = _check_budget(budget_bytes)
    if same_spin and nl != nr:
        raise ValueError(f"same_spin C pullback requires nl == nr, got nl={nl}, nr={nr}")
    pi, pj = _validate_pairs(pairs, i_low=0, i_high=nl, j_low=0, j_high=nr)

    fl_t = _transpose_occ_first(fit_l, aux, nl, ext, "fit_l")
    fr_t = fl_t if same_spin else _transpose_occ_first(fit_r, aux, nr, ext, "fit_r")
    gl_t = _transpose_occ_first(raw_g_l, aux, nl, ext, "raw_g_l")
    gr_t = gl_t if same_spin else _transpose_occ_first(raw_g_r, aux, nr, ext, "raw_g_r")
    mg = _validate_f64_array(metric_g, (aux, aux), "metric_g")
    f_l = _validate_f64_array(fock_l, (ext, ext), "fock_l")
    f_r = f_l if same_spin else _validate_f64_array(fock_r, (ext, ext), "fock_r")
    co = None if same_spin else _validate_f64_array(cross_overlap, (ext, ext), "cross_overlap")
    bc = _validate_f64_array(bar_c_all, (len(pairs), obs - nl, obs - nr), "bar_c_all")

    b_fl_t = _validate_writable_output(np.zeros_like(fl_t), fl_t.shape, "bar_fit_l")
    b_fr_t = b_fl_t if same_spin else _validate_writable_output(np.zeros_like(fr_t), fr_t.shape, "bar_fit_r")
    b_gl_t = _validate_writable_output(np.zeros_like(gl_t), gl_t.shape, "bar_raw_g_l")
    b_gr_t = b_gl_t if same_spin else _validate_writable_output(np.zeros_like(gr_t), gr_t.shape, "bar_raw_g_r")
    b_mg = _validate_writable_output(np.zeros_like(mg), mg.shape, "bar_metric_g")
    b_f_l = _validate_writable_output(np.zeros_like(f_l), f_l.shape, "bar_fock_l")
    b_f_r = b_f_l if same_spin else _validate_writable_output(np.zeros_like(f_r), f_r.shape, "bar_fock_r")
    b_co = None if same_spin else _validate_writable_output(np.zeros_like(co), co.shape, "bar_cross_overlap")

    rc = func(
        int(same_spin),
        nl,
        nr,
        obs,
        ext,
        aux,
        len(pairs),
        eff_budget,
        _iptr(pi),
        _iptr(pj),
        _dptr(fl_t),
        _dptr(fr_t),
        _dptr(gl_t),
        _dptr(gr_t),
        _dptr(mg),
        _dptr(f_l),
        _dptr(f_r),
        _dptr(co),
        _dptr(bc),
        _dptr(b_fl_t),
        _dptr(b_fr_t),
        _dptr(b_gl_t),
        _dptr(b_gr_t),
        _dptr(b_mg),
        _dptr(b_f_l),
        _dptr(b_f_r),
        _dptr(b_co),
    )
    if rc != 0:
        raise RuntimeError(f"native CC C MO pullback failed with status {rc}")
    out = {
        "bar_fit_l": b_fl_t.transpose(1, 0, 2),
        "bar_raw_g_l": b_gl_t.transpose(1, 0, 2),
        "bar_metric_g": b_mg,
        "bar_fock_l": b_f_l,
    }
    if not same_spin:
        out["bar_fit_r"] = b_fr_t.transpose(1, 0, 2)
        out["bar_raw_g_r"] = b_gr_t.transpose(1, 0, 2)
        out["bar_fock_r"] = b_f_r
        assert b_co is not None
        out["bar_cross_overlap"] = b_co
    for key, arr in out.items():
        if not np.isfinite(arr).all():
            raise RuntimeError(f"native CC C MO pullback produced nonfinite output in {key}")
    return out


def run_scalar_pullback_native(
    func,
    *,
    same_spin: bool,
    nl: int,
    nr: int,
    fl: int,
    fr: int,
    obs: int,
    ext: int,
    aux: int,
    scalar_weight: float,
    pairs: list[tuple[int, int]],
    fit_l: np.ndarray,
    fit_r: np.ndarray,
    raw_coul_r: np.ndarray,
    raw_g_l: np.ndarray,
    raw_g_r: np.ndarray,
    raw_gc_l: np.ndarray,
    raw_gc_r: np.ndarray,
    raw_sq_l: np.ndarray,
    raw_sq_r: np.ndarray,
    raw_dc_l: np.ndarray,
    raw_dc_r: np.ndarray,
    fmix_fit_l: np.ndarray,
    fmix_fit_r: np.ndarray,
    fmix_g_l: np.ndarray,
    fmix_g_r: np.ndarray,
    fmix_sq_l: np.ndarray,
    fmix_sq_r: np.ndarray,
    metric_g: np.ndarray,
    metric_gc: np.ndarray,
    metric_sq: np.ndarray,
    metric_dc: np.ndarray,
    fock_l: np.ndarray,
    fock_r: np.ndarray,
    exch_l: np.ndarray,
    exch_r: np.ndarray,
    cross_overlap: np.ndarray | None,
    budget_bytes: int = 0,
) -> tuple[float, float, float, dict[str, np.ndarray]]:
    """Execute scalar V + B - X MO-space pullback across all pairs on GPU (or host reference)."""
    fl = _check_c_int(fl, "fl", low=0)
    fr = _check_c_int(fr, "fr", low=0)
    nl = _check_c_int(nl, "nl", low=fl + 1)
    nr = _check_c_int(nr, "nr", low=fr + 1)
    obs = _check_c_int(obs, "obs", low=max(nl, nr) + 1)
    ext = _check_c_int(ext, "ext", low=obs + 1)
    aux = _check_c_int(aux, "aux", low=1)
    eff_budget = _check_budget(budget_bytes)
    if same_spin and (nl != nr or fl != fr):
        raise ValueError(
            f"same_spin scalar pullback requires nl == nr and fl == fr, "
            f"got nl={nl}, nr={nr}, fl={fl}, fr={fr}"
        )
    if isinstance(scalar_weight, bool) or not isinstance(scalar_weight, (int, float, np.number)):
        raise ValueError(f"scalar_weight must be a real scalar, got {type(scalar_weight).__name__}")
    w_scalar = float(scalar_weight)
    if not np.isfinite(w_scalar):
        raise ValueError("scalar_weight must be finite")

    pi, pj = _validate_pairs(pairs, i_low=fl, i_high=nl, j_low=fr, j_high=nr)
    nact_l = nl - fl
    nact_r = nr - fr

    fl_t = _transpose_occ_first(fit_l, aux, nl, ext, "fit_l")
    fr_t = fl_t if same_spin else _transpose_occ_first(fit_r, aux, nr, ext, "fit_r")
    rcr_t = _transpose_occ_first(raw_coul_r, aux, nr, ext, "raw_coul_r")
    gl_t = _transpose_occ_first(raw_g_l, aux, nl, ext, "raw_g_l")
    gr_t = gl_t if same_spin else _transpose_occ_first(raw_g_r, aux, nr, ext, "raw_g_r")
    gcl_t = _transpose_occ_first(raw_gc_l, aux, nl, ext, "raw_gc_l")
    gcr_t = gcl_t if same_spin else _transpose_occ_first(raw_gc_r, aux, nr, ext, "raw_gc_r")
    sql_t = _transpose_occ_first(raw_sq_l, aux, nl, ext, "raw_sq_l")
    sqr_t = sql_t if same_spin else _transpose_occ_first(raw_sq_r, aux, nr, ext, "raw_sq_r")
    dcl_t = _transpose_occ_first(raw_dc_l, aux, nl, ext, "raw_dc_l")
    dcr_t = dcl_t if same_spin else _transpose_occ_first(raw_dc_r, aux, nr, ext, "raw_dc_r")

    def _fmix_t(arr: np.ndarray, nact: int, name: str) -> np.ndarray:
        checked = _validate_f64_array(arr, (aux, nact, ext), name)
        return np.ascontiguousarray(checked.transpose(1, 0, 2), dtype=np.float64)

    xfl_t = _fmix_t(fmix_fit_l, nact_l, "fmix_fit_l")
    xfr_t = xfl_t if same_spin else _fmix_t(fmix_fit_r, nact_r, "fmix_fit_r")
    xgl_t = _fmix_t(fmix_g_l, nact_l, "fmix_g_l")
    xgr_t = xgl_t if same_spin else _fmix_t(fmix_g_r, nact_r, "fmix_g_r")
    xsql_t = _fmix_t(fmix_sq_l, nact_l, "fmix_sq_l")
    xsqr_t = xsql_t if same_spin else _fmix_t(fmix_sq_r, nact_r, "fmix_sq_r")

    mg = _validate_f64_array(metric_g, (aux, aux), "metric_g")
    mgc = _validate_f64_array(metric_gc, (aux, aux), "metric_gc")
    msq = _validate_f64_array(metric_sq, (aux, aux), "metric_sq")
    mdc = _validate_f64_array(metric_dc, (aux, aux), "metric_dc")
    f_l = _validate_f64_array(fock_l, (ext, ext), "fock_l")
    f_r = f_l if same_spin else _validate_f64_array(fock_r, (ext, ext), "fock_r")
    k_l = _validate_f64_array(exch_l, (ext, ext), "exch_l")
    k_r = k_l if same_spin else _validate_f64_array(exch_r, (ext, ext), "exch_r")
    co = None if same_spin else _validate_f64_array(cross_overlap, (ext, ext), "cross_overlap")

    energies = _validate_writable_output(np.zeros(3, dtype=np.float64), (3,), "energies")
    b_fl_t = _validate_writable_output(np.zeros_like(fl_t), fl_t.shape, "bar_fit_l")
    b_fr_t = b_fl_t if same_spin else _validate_writable_output(np.zeros_like(fr_t), fr_t.shape, "bar_fit_r")
    b_rcr_t = _validate_writable_output(np.zeros_like(rcr_t), rcr_t.shape, "bar_raw_coul_r")
    b_gl_t = _validate_writable_output(np.zeros_like(gl_t), gl_t.shape, "bar_raw_g_l")
    b_gr_t = b_gl_t if same_spin else _validate_writable_output(np.zeros_like(gr_t), gr_t.shape, "bar_raw_g_r")
    b_gcl_t = _validate_writable_output(np.zeros_like(gcl_t), gcl_t.shape, "bar_raw_gc_l")
    b_gcr_t = b_gcl_t if same_spin else _validate_writable_output(np.zeros_like(gcr_t), gcr_t.shape, "bar_raw_gc_r")
    b_sql_t = _validate_writable_output(np.zeros_like(sql_t), sql_t.shape, "bar_raw_sq_l")
    b_sqr_t = b_sql_t if same_spin else _validate_writable_output(np.zeros_like(sqr_t), sqr_t.shape, "bar_raw_sq_r")
    b_dcl_t = _validate_writable_output(np.zeros_like(dcl_t), dcl_t.shape, "bar_raw_dc_l")
    b_dcr_t = b_dcl_t if same_spin else _validate_writable_output(np.zeros_like(dcr_t), dcr_t.shape, "bar_raw_dc_r")
    b_xfl_t = _validate_writable_output(np.zeros_like(xfl_t), xfl_t.shape, "bar_fmix_fit_l")
    b_xfr_t = b_xfl_t if same_spin else _validate_writable_output(np.zeros_like(xfr_t), xfr_t.shape, "bar_fmix_fit_r")
    b_xgl_t = _validate_writable_output(np.zeros_like(xgl_t), xgl_t.shape, "bar_fmix_g_l")
    b_xgr_t = b_xgl_t if same_spin else _validate_writable_output(np.zeros_like(xgr_t), xgr_t.shape, "bar_fmix_g_r")
    b_xsql_t = _validate_writable_output(np.zeros_like(xsql_t), xsql_t.shape, "bar_fmix_sq_l")
    b_xsqr_t = b_xsql_t if same_spin else _validate_writable_output(np.zeros_like(xsqr_t), xsqr_t.shape, "bar_fmix_sq_r")
    b_mg = _validate_writable_output(np.zeros_like(mg), mg.shape, "bar_metric_g")
    b_mgc = _validate_writable_output(np.zeros_like(mgc), mgc.shape, "bar_metric_gc")
    b_msq = _validate_writable_output(np.zeros_like(msq), msq.shape, "bar_metric_sq")
    b_mdc = _validate_writable_output(np.zeros_like(mdc), mdc.shape, "bar_metric_dc")
    b_f_l = _validate_writable_output(np.zeros_like(f_l), f_l.shape, "bar_fock_l")
    b_f_r = b_f_l if same_spin else _validate_writable_output(np.zeros_like(f_r), f_r.shape, "bar_fock_r")
    b_k_l = _validate_writable_output(np.zeros_like(k_l), k_l.shape, "bar_exch_l")
    b_k_r = b_k_l if same_spin else _validate_writable_output(np.zeros_like(k_r), k_r.shape, "bar_exch_r")
    b_co = None if same_spin else _validate_writable_output(np.zeros_like(co), co.shape, "bar_cross_overlap")

    rc = func(
        int(same_spin),
        nl,
        nr,
        fl,
        fr,
        obs,
        ext,
        aux,
        len(pairs),
        eff_budget,
        w_scalar,
        _iptr(pi),
        _iptr(pj),
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
        _dptr(xfl_t),
        _dptr(xfr_t),
        _dptr(xgl_t),
        _dptr(xgr_t),
        _dptr(xsql_t),
        _dptr(xsqr_t),
        _dptr(mg),
        _dptr(mgc),
        _dptr(msq),
        _dptr(mdc),
        _dptr(f_l),
        _dptr(f_r),
        _dptr(k_l),
        _dptr(k_r),
        _dptr(co),
        _dptr(energies),
        _dptr(b_fl_t),
        _dptr(b_fr_t),
        _dptr(b_rcr_t),
        _dptr(b_gl_t),
        _dptr(b_gr_t),
        _dptr(b_gcl_t),
        _dptr(b_gcr_t),
        _dptr(b_sql_t),
        _dptr(b_sqr_t),
        _dptr(b_dcl_t),
        _dptr(b_dcr_t),
        _dptr(b_xfl_t),
        _dptr(b_xfr_t),
        _dptr(b_xgl_t),
        _dptr(b_xgr_t),
        _dptr(b_xsql_t),
        _dptr(b_xsqr_t),
        _dptr(b_mg),
        _dptr(b_mgc),
        _dptr(b_msq),
        _dptr(b_mdc),
        _dptr(b_f_l),
        _dptr(b_f_r),
        _dptr(b_k_l),
        _dptr(b_k_r),
        _dptr(b_co),
    )
    if rc != 0:
        raise RuntimeError(f"native CC scalar MO pullback failed with status {rc}")

    out = {
        "bar_fit_l": b_fl_t.transpose(1, 0, 2),
        "bar_raw_coul_r": b_rcr_t.transpose(1, 0, 2),
        "bar_raw_g_l": b_gl_t.transpose(1, 0, 2),
        "bar_raw_gc_l": b_gcl_t.transpose(1, 0, 2),
        "bar_raw_sq_l": b_sql_t.transpose(1, 0, 2),
        "bar_raw_dc_l": b_dcl_t.transpose(1, 0, 2),
        "bar_fmix_fit_l": b_xfl_t.transpose(1, 0, 2),
        "bar_fmix_g_l": b_xgl_t.transpose(1, 0, 2),
        "bar_fmix_sq_l": b_xsql_t.transpose(1, 0, 2),
        "bar_metric_g": b_mg,
        "bar_metric_gc": b_mgc,
        "bar_metric_sq": b_msq,
        "bar_metric_dc": b_mdc,
        "bar_fock_l": b_f_l,
        "bar_exch_l": b_k_l,
    }
    if not same_spin:
        out["bar_fit_r"] = b_fr_t.transpose(1, 0, 2)
        out["bar_raw_g_r"] = b_gr_t.transpose(1, 0, 2)
        out["bar_raw_gc_r"] = b_gcr_t.transpose(1, 0, 2)
        out["bar_raw_sq_r"] = b_sqr_t.transpose(1, 0, 2)
        out["bar_raw_dc_r"] = b_dcr_t.transpose(1, 0, 2)
        out["bar_fmix_fit_r"] = b_xfr_t.transpose(1, 0, 2)
        out["bar_fmix_g_r"] = b_xgr_t.transpose(1, 0, 2)
        out["bar_fmix_sq_r"] = b_xsqr_t.transpose(1, 0, 2)
        out["bar_fock_r"] = b_f_r
        out["bar_exch_r"] = b_k_r
        assert b_co is not None
        out["bar_cross_overlap"] = b_co
    if not np.isfinite(energies).all():
        raise RuntimeError("native CC scalar MO pullback produced nonfinite energy")
    for key, arr in out.items():
        if not np.isfinite(arr).all():
            raise RuntimeError(f"native CC scalar MO pullback produced nonfinite output in {key}")
    return float(energies[0]), float(energies[1]), float(energies[2]), out
