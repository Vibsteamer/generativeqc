"""Native GPU-resident cuBLAS (and host-reference) bindings for F12 CPHF AOT primitives."""

from __future__ import annotations

import ctypes as ct
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ._f12_workspace import workspace_bytes
from ._native import installed_f12_shell_library

ENV_REQUIRE_CPHF = "GENERATIVEQC_F12_REQUIRE_AOT_CPHF"
ENV_REQUIRE_STEP1 = "GENERATIVEQC_F12_REQUIRE_AOT_CPHF_STEP1"
ENV_DISABLE_CPHF = "GENERATIVEQC_F12_DISABLE_AOT_CPHF"
ENV_CPHF_LIBRARY = "GENERATIVEQC_F12_CPHF_LIBRARY"
ENV_SHELL_LIBRARY = "GENERATIVEQC_F12_SHELL_LIBRARY"
ENV_SINK_LIBRARY = "GENERATIVEQC_F12_SINK_LIBRARY"
ENV_HOST_SINK = "GENERATIVEQC_F12_HOST_SINK"
DEFAULT_BUDGET_BYTES = 512 << 20
_C_INT_MAX = (1 << 31) - 1

_D_PTR = ct.POINTER(ct.c_double)
_CDLL_CACHE: dict[str, ct.CDLL] = {}
_RTLD_NOLOAD = getattr(os, "RTLD_NOLOAD", getattr(ct, "RTLD_NOLOAD", 0))
_AUTODISCOVER_TAGS = ("focal_shell", "molecular_integral_sink")

_SYMBOL_BASE = {
    "whiten_3c": "gqc_f12_cphf_whiten_3c",
    "orbital_rot_metric": "gqc_f12_orbital_rot_metric",
    "cphf_action": "gqc_f12_cphf_action",
    "df_canonical_driver": "gqc_f12_df_canonical_driver",
    "df_nuclear_weights": "gqc_f12_df_nuclear_weights",
}

_ARGTYPES: dict[str, list] = {
    "whiten_3c": [
        ct.c_int,
        ct.c_int,
        ct.c_int64,
        ct.c_uint64,
        _D_PTR,
        _D_PTR,
        _D_PTR,
    ],
    "orbital_rot_metric": [
        ct.c_int,
        ct.c_int,
        ct.c_uint64,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
    ],
    "cphf_action": (
        [ct.c_int] * 5
        + [ct.c_uint64]
        + [_D_PTR] * 12
    ),
    "df_canonical_driver": (
        [ct.c_int] * 4
        + [ct.c_uint64]
        + [_D_PTR] * 6
    ),
    "df_nuclear_weights": (
        [ct.c_int] * 6
        + [ct.c_uint64]
        + [_D_PTR] * 23
    ),
}


def _dptr(arr: np.ndarray | None):
    if arr is None:
        return ct.cast(0, _D_PTR)
    return arr.ctypes.data_as(_D_PTR)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _active_require_env() -> str | None:
    if _env_truthy(ENV_REQUIRE_CPHF):
        return ENV_REQUIRE_CPHF
    if _env_truthy(ENV_REQUIRE_STEP1):
        return ENV_REQUIRE_STEP1
    return None


def _cphf_disabled() -> bool:
    return _env_truthy(ENV_DISABLE_CPHF)


def require_aot_cphf() -> bool:
    """Return True when AOT CPHF kernels are strictly required."""
    return _active_require_env() is not None


def require_aot_cphf_step1() -> bool:
    """Return True when AOT CPHF kernels are strictly required."""
    return require_aot_cphf()


def _load_cdll(library: Path | str | ct.CDLL | None) -> ct.CDLL | None:
    if library is None:
        return None
    if isinstance(library, ct.CDLL):
        return library
    resolved = str(Path(library).resolve())
    cached = _CDLL_CACHE.get(resolved)
    if cached is not None:
        return cached
    if not Path(resolved).is_file():
        return None
    cdll = ct.CDLL(resolved)
    _CDLL_CACHE[resolved] = cdll
    return cdll


def _discover_loaded_shell_libraries() -> list[ct.CDLL]:
    """Return already-loaded *focal_shell* / *molecular_integral_sink* libraries via /proc/self/maps."""
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    try:
        text = maps.read_text()
    except OSError:
        return []
    seen: set[str] = set()
    handles: list[ct.CDLL] = []
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        raw_path = fields[5].strip()
        if not raw_path.startswith("/") or raw_path.endswith(" (deleted)"):
            continue
        if raw_path in seen:
            continue
        name = Path(raw_path).name
        if ".so" not in name or not any(tag in name for tag in _AUTODISCOVER_TAGS):
            continue
        seen.add(raw_path)
        cached = _CDLL_CACHE.get(raw_path)
        if cached is not None:
            handles.append(cached)
            continue
        try:
            mode = _RTLD_NOLOAD if _RTLD_NOLOAD else ct.DEFAULT_MODE
            cdll = ct.CDLL(raw_path, mode=mode)
        except OSError:
            continue
        _CDLL_CACHE[raw_path] = cdll
        handles.append(cdll)
    return handles


def _bind_from_cdll(cdll: ct.CDLL | None, kind: str, host_reference: bool):
    if cdll is None:
        return None
    base = _SYMBOL_BASE.get(kind)
    if base is None:
        raise ValueError(f"Unknown CPHF relax primitive kind {kind!r}")
    suffix = "_host_reference" if host_reference else "_cuda"
    func = getattr(cdll, base + suffix, None)
    if func is None:
        return None
    func.argtypes = _ARGTYPES[kind]
    func.restype = ct.c_int
    return func


def bind_cphf_relax(
    library: Path | str | ct.CDLL | None,
    kind: str,
    host_reference: bool,
):
    """Return bound ctypes callable for one CPHF AOT primitive, or None if absent."""
    if kind not in _SYMBOL_BASE:
        raise ValueError(f"Unknown CPHF relax primitive kind {kind!r}")
    return _bind_from_cdll(_load_cdll(library), kind, host_reference)


def resolve_cphf_relax(
    native_library: Path | str | ct.CDLL | None,
    kind: str,
    *,
    fallback_library: Path | str | ct.CDLL | None = None,
    host_reference: bool | None = None,
):
    """Resolve one CPHF AOT native symbol for a single explicit backend.

    Requires an explicit boolean ``host_reference`` (no dual-backend fallback).
    Honors GENERATIVEQC_F12_REQUIRE_AOT_CPHF=1 and GENERATIVEQC_F12_REQUIRE_AOT_CPHF_STEP1=1,
    and rejects simultaneous require + disable environment flags.
    """
    if kind not in _SYMBOL_BASE:
        raise ValueError(f"Unknown CPHF relax primitive kind {kind!r}")
    req_env = _active_require_env()
    if req_env is not None and _cphf_disabled():
        raise ValueError(
            f"Conflicting environment flags: {req_env} and {ENV_DISABLE_CPHF} "
            "cannot both be enabled"
        )
    if _cphf_disabled():
        return None
    if not isinstance(host_reference, bool):
        raise ValueError(
            "resolve_cphf_relax requires an explicit host_reference boolean "
            "(single-backend binding)"
        )
    is_host = host_reference

    # Strict precedence for ONE single backend:
    # 1. Explicit native_library, 2. fallback_library, 3. explicit env overrides,
    # 4. installed target, 5. already-loaded shell libraries (RTLD_NOLOAD).
    candidates: list[Path | str | ct.CDLL] = []
    for lib in (native_library, fallback_library):
        if lib is not None:
            candidates.append(lib)
    for env_key in (ENV_CPHF_LIBRARY, ENV_SHELL_LIBRARY, ENV_SINK_LIBRARY):
        env_path = os.environ.get(env_key)
        if env_path:
            candidates.append(env_path)

    if installed := installed_f12_shell_library(host_reference=is_host):
        candidates.append(installed)

    for lib in candidates:
        cdll = _load_cdll(lib)
        if cdll is None:
            continue
        func = _bind_from_cdll(cdll, kind, is_host)
        if func is not None:
            return func

    for cdll in _discover_loaded_shell_libraries():
        func = _bind_from_cdll(cdll, kind, is_host)
        if func is not None:
            return func

    if req_env is not None:
        mode = "host_reference" if is_host else "cuda"
        raise RuntimeError(
            f"{req_env}=1 requires native AOT CPHF symbol "
            f"{_SYMBOL_BASE[kind]!r} ({mode}), but no matching library/symbol was found"
        )
    return None


def _clamp_budget(budget_bytes: int | None) -> int:
    """None and zero select the default; positive integer byte counts are capped."""
    if budget_bytes is None:
        return DEFAULT_BUDGET_BYTES
    if isinstance(budget_bytes, (bool, np.bool_)):
        raise TypeError("CPHF workspace budget must be an integer byte count")
    size = workspace_bytes(budget_bytes, signed=False)
    return min(size, DEFAULT_BUDGET_BYTES) if size else DEFAULT_BUDGET_BYTES


def run_whiten_3c_native(
    func,
    whitener: np.ndarray,
    raw: np.ndarray,
    *,
    budget_bytes: int | None = DEFAULT_BUDGET_BYTES,
) -> np.ndarray:
    """Compute df[A, ...] = sum_B W[A, B] * raw[B, ...] via native BLAS/cuBLAS."""
    budget = _clamp_budget(budget_bytes)
    w = np.ascontiguousarray(whitener, dtype=np.float64)
    r = np.ascontiguousarray(raw, dtype=np.float64)
    if w.ndim != 2 or r.ndim < 2 or w.shape[1] != r.shape[0]:
        raise ValueError(
            f"Incompatible whiten_3c shapes: whitener={w.shape}, raw={r.shape}"
        )
    aux_out, aux_in = w.shape
    ncols = int(np.prod(r.shape[1:], dtype=np.int64))
    out = np.empty((aux_out, *r.shape[1:]), dtype=np.float64)
    rc = func(
        aux_out,
        aux_in,
        ncols,
        budget,
        _dptr(w),
        _dptr(r),
        _dptr(out),
    )
    if rc != 0:
        raise RuntimeError(f"native CPHF whiten_3c failed with status {rc}")
    return out


def run_orbital_rot_metric_native(
    func,
    coefficients: np.ndarray,
    frame_bar: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    compute_rotation: bool = True,
    compute_overlap: bool = False,
    budget_bytes: int | None = DEFAULT_BUDGET_BYTES,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute C^T @ frame_bar and/or symmetric metric overlap pullback."""
    budget = _clamp_budget(budget_bytes)
    c = np.ascontiguousarray(coefficients, dtype=np.float64)
    bar = np.ascontiguousarray(frame_bar, dtype=np.float64)
    if c.ndim != 2 or bar.shape != c.shape:
        raise ValueError(
            f"Expected matching 2D coefficients and frame_bar, got {c.shape} and {bar.shape}"
        )
    nao, nmo = c.shape
    if nmo < 1 or nmo > nao:
        raise ValueError(f"Invalid orbital frame shape ({nao}, {nmo})")
    if not compute_rotation and not compute_overlap:
        raise ValueError("At least one of compute_rotation or compute_overlap must be True")

    mask_arr: np.ndarray | None = None
    if compute_overlap:
        if mask is None:
            raise ValueError("mask is required when compute_overlap=True")
        mask_arr = np.ascontiguousarray(mask, dtype=np.float64)
        if mask_arr.shape != (nmo, nmo):
            raise ValueError(f"Expected mask of shape ({nmo}, {nmo}), got {mask_arr.shape}")

    rot_out = np.empty((nmo, nmo), dtype=np.float64) if compute_rotation else None
    ov_out = np.empty((nao, nao), dtype=np.float64) if compute_overlap else None

    rc = func(
        nao,
        nmo,
        budget,
        _dptr(c),
        _dptr(bar),
        _dptr(mask_arr),
        _dptr(rot_out),
        _dptr(ov_out),
    )
    if rc != 0:
        raise RuntimeError(f"native orbital_rot_metric failed with status {rc}")
    return rot_out, ov_out


def run_cphf_action_native(
    func,
    *,
    is_rhf: bool,
    oa: int,
    ob: int,
    obs: int,
    aux: int,
    vo_a: np.ndarray | None = None,
    oo_a: np.ndarray | None = None,
    vv_a: np.ndarray | None = None,
    gap_a: np.ndarray | None = None,
    trial_a: np.ndarray | None = None,
    vo_b: np.ndarray | None = None,
    oo_b: np.ndarray | None = None,
    vv_b: np.ndarray | None = None,
    gap_b: np.ndarray | None = None,
    trial_b: np.ndarray | None = None,
    budget_bytes: int | None = DEFAULT_BUDGET_BYTES,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Apply the DF-HF coupled orbital Hessian (RHF fast path or UHF general path)."""
    budget = _clamp_budget(budget_bytes)
    eff_ob = 0 if is_rhf else ob

    def _prep_spin(occ, vo, oo, vv, gap, trial, label):
        if occ <= 0:
            return None, None, None, None, None, None
        virt = obs - occ
        vo_c = np.ascontiguousarray(vo, dtype=np.float64)
        oo_c = np.ascontiguousarray(oo, dtype=np.float64)
        vv_c = np.ascontiguousarray(vv, dtype=np.float64)
        gap_c = np.ascontiguousarray(gap, dtype=np.float64)
        tr_c = np.ascontiguousarray(trial, dtype=np.float64)
        if (
            vo_c.shape != (aux, virt, occ)
            or oo_c.shape != (aux, occ, occ)
            or vv_c.shape != (aux, virt, virt)
            or gap_c.shape != (virt, occ)
            or tr_c.shape != (virt, occ)
        ):
            raise ValueError(f"Incompatible CPHF action shapes for {label}")
        out_c = np.empty((virt, occ), dtype=np.float64)
        return vo_c, oo_c, vv_c, gap_c, tr_c, out_c

    vo_ac, oo_ac, vv_ac, gap_ac, tr_ac, out_a = _prep_spin(
        oa, vo_a, oo_a, vv_a, gap_a, trial_a, "alpha"
    )
    vo_bc, oo_bc, vv_bc, gap_bc, tr_bc, out_b = _prep_spin(
        eff_ob, vo_b, oo_b, vv_b, gap_b, trial_b, "beta"
    )

    rc = func(
        int(is_rhf),
        oa,
        eff_ob,
        obs,
        aux,
        budget,
        _dptr(vo_ac),
        _dptr(oo_ac),
        _dptr(vv_ac),
        _dptr(gap_ac),
        _dptr(tr_ac),
        _dptr(out_a),
        _dptr(vo_bc),
        _dptr(oo_bc),
        _dptr(vv_bc),
        _dptr(gap_bc),
        _dptr(tr_bc),
        _dptr(out_b),
    )
    if rc != 0:
        raise RuntimeError(f"native CPHF action failed with status {rc}")
    return out_a, out_b


def run_df_canonical_driver_native(
    func,
    *,
    oa: int,
    ob: int,
    obs: int,
    aux: int,
    df_a: np.ndarray | None = None,
    fock_seed_a: np.ndarray | None = None,
    df_b: np.ndarray | None = None,
    fock_seed_b: np.ndarray | None = None,
    budget_bytes: int | None = DEFAULT_BUDGET_BYTES,
) -> dict[str, np.ndarray]:
    """Compute unrestricted DF canonical orbital driver via native BLAS/cuBLAS."""
    budget = _clamp_budget(budget_bytes)

    def _prep_spin(occ, df, seed, label):
        if occ <= 0:
            if df is None and seed is None:
                return None, None, None
            if df is None or seed is None:
                raise ValueError(
                    f"Empty spin {label} requires both df and fock_seed when supplied"
                )
            df_c = np.ascontiguousarray(df, dtype=np.float64)
            seed_c = np.ascontiguousarray(seed, dtype=np.float64)
            if df_c.shape != (aux, obs, obs) or seed_c.shape != (obs, obs):
                raise ValueError(f"Incompatible df_canonical_driver shapes for {label}")
            return df_c, seed_c, None
        df_c = np.ascontiguousarray(df, dtype=np.float64)
        seed_c = np.ascontiguousarray(seed, dtype=np.float64)
        if df_c.shape != (aux, obs, obs) or seed_c.shape != (obs, obs):
            raise ValueError(f"Incompatible df_canonical_driver shapes for {label}")
        out_c = np.empty((obs - occ, occ), dtype=np.float64)
        return df_c, seed_c, out_c

    df_ac, seed_ac, out_a = _prep_spin(oa, df_a, fock_seed_a, "alpha")
    df_bc, seed_bc, out_b = _prep_spin(ob, df_b, fock_seed_b, "beta")

    rc = func(
        oa,
        ob,
        obs,
        aux,
        budget,
        _dptr(df_ac),
        _dptr(seed_ac),
        _dptr(out_a),
        _dptr(df_bc),
        _dptr(seed_bc),
        _dptr(out_b),
    )
    if rc != 0:
        raise RuntimeError(f"native df_canonical_driver failed with status {rc}")

    channels: dict[str, np.ndarray] = {}
    if out_a is not None:
        channels["alpha"] = out_a
    if out_b is not None:
        channels["beta"] = out_b
    return channels


def run_df_nuclear_weights_native(
    func,
    *,
    oa: int,
    ob: int,
    obs: int,
    aux: int,
    nao: int,
    coefficients_alpha: np.ndarray,
    coefficients_beta: np.ndarray,
    occupied_energies_alpha: np.ndarray | None,
    occupied_energies_beta: np.ndarray | None,
    df_alpha: np.ndarray,
    df_beta: np.ndarray,
    fock_seed_alpha: np.ndarray,
    fock_seed_beta: np.ndarray,
    overlap_seed_alpha: np.ndarray,
    overlap_seed_beta: np.ndarray,
    whitener: np.ndarray,
    z_alpha: np.ndarray | None,
    z_beta: np.ndarray | None,
    direct_weights: Mapping[str, np.ndarray] | None = None,
    frame_overlap: Mapping[str, np.ndarray] | None = None,
    budget_bytes: int | None = DEFAULT_BUDGET_BYTES,
) -> dict[str, np.ndarray]:
    """Compute DF-UHF nuclear source weights via native BLAS/cuBLAS with bounded Q-tiling."""
    budget = _clamp_budget(budget_bytes)
    include_direct = direct_weights is not None
    if include_direct != (frame_overlap is not None):
        raise ValueError(
            "direct_weights and frame_overlap must be provided together"
        )

    coeff_a = np.ascontiguousarray(coefficients_alpha, dtype=np.float64)
    coeff_b = (
        coeff_a
        if coefficients_beta is coefficients_alpha
        else np.ascontiguousarray(coefficients_beta, dtype=np.float64)
    )
    df_a = np.ascontiguousarray(df_alpha, dtype=np.float64)
    df_b = (
        df_a
        if df_beta is df_alpha
        else np.ascontiguousarray(df_beta, dtype=np.float64)
    )
    fs_a = np.ascontiguousarray(fock_seed_alpha, dtype=np.float64)
    fs_b = (
        fs_a
        if fock_seed_beta is fock_seed_alpha
        else np.ascontiguousarray(fock_seed_beta, dtype=np.float64)
    )
    os_a = np.ascontiguousarray(overlap_seed_alpha, dtype=np.float64)
    os_b = (
        os_a
        if overlap_seed_beta is overlap_seed_alpha
        else np.ascontiguousarray(overlap_seed_beta, dtype=np.float64)
    )
    w_c = np.ascontiguousarray(whitener, dtype=np.float64)

    if (
        coeff_a.shape != (nao, obs)
        or coeff_b.shape != (nao, obs)
        or df_a.shape != (aux, obs, obs)
        or df_b.shape != (aux, obs, obs)
        or fs_a.shape != (obs, obs)
        or fs_b.shape != (obs, obs)
        or os_a.shape != (obs, obs)
        or os_b.shape != (obs, obs)
        or w_c.shape != (aux, aux)
    ):
        raise ValueError("Incompatible df_nuclear_weights central tensor shapes")

    occ_e_a = (
        np.ascontiguousarray(occupied_energies_alpha, dtype=np.float64)
        if oa > 0
        else None
    )
    occ_e_b = (
        (
            occ_e_a
            if occupied_energies_beta is occupied_energies_alpha and oa == ob
            else np.ascontiguousarray(occupied_energies_beta, dtype=np.float64)
        )
        if ob > 0
        else None
    )
    z_ac = np.ascontiguousarray(z_alpha, dtype=np.float64) if oa > 0 else None
    z_bc = (
        (
            z_ac
            if z_beta is z_alpha and oa == ob
            else np.ascontiguousarray(z_beta, dtype=np.float64)
        )
        if ob > 0
        else None
    )
    if oa > 0 and (
        occ_e_a is None
        or occ_e_a.shape != (oa,)
        or z_ac is None
        or z_ac.shape != (obs - oa, oa)
    ):
        raise ValueError("Incompatible alpha occupied_energies or z shape")
    if ob > 0 and (
        occ_e_b is None
        or occ_e_b.shape != (ob,)
        or z_bc is None
        or z_bc.shape != (obs - ob, ob)
    ):
        raise ValueError("Incompatible beta occupied_energies or z shape")

    dir_ov = dir_h = dir_ao = dir_m = fr_a = fr_b = None
    if include_direct:
        assert direct_weights is not None and frame_overlap is not None
        dir_ov = np.ascontiguousarray(direct_weights["overlap_ao"], dtype=np.float64)
        dir_h = np.ascontiguousarray(direct_weights["hcore_ao"], dtype=np.float64)
        dir_ao = np.ascontiguousarray(direct_weights["ao_coulomb"], dtype=np.float64)
        dir_m = np.ascontiguousarray(
            direct_weights["metric/coulomb"], dtype=np.float64
        )
        fr_a = np.ascontiguousarray(frame_overlap["alpha"], dtype=np.float64)
        fr_b = np.ascontiguousarray(frame_overlap["beta"], dtype=np.float64)
        if (
            dir_ov.shape != (nao, nao)
            or dir_h.shape != (nao, nao)
            or dir_ao.shape != (aux, nao, nao)
            or dir_m.shape != (aux, aux)
            or fr_a.shape != (nao, nao)
            or fr_b.shape != (nao, nao)
        ):
            raise ValueError("Incompatible direct_weights or frame_overlap shapes")

    overlap_ao_out = np.empty((nao, nao), dtype=np.float64)
    hcore_ao_out = np.empty((nao, nao), dtype=np.float64)
    ao_coulomb_out = np.empty((aux, nao, nao), dtype=np.float64)
    metric_coulomb_out = np.empty((aux, aux), dtype=np.float64)

    rc = func(
        oa,
        ob,
        obs,
        aux,
        nao,
        int(include_direct),
        budget,
        _dptr(coeff_a),
        _dptr(coeff_b),
        _dptr(occ_e_a),
        _dptr(occ_e_b),
        _dptr(df_a),
        _dptr(df_b),
        _dptr(fs_a),
        _dptr(fs_b),
        _dptr(os_a),
        _dptr(os_b),
        _dptr(w_c),
        _dptr(z_ac),
        _dptr(z_bc),
        _dptr(dir_ov),
        _dptr(dir_h),
        _dptr(dir_ao),
        _dptr(dir_m),
        _dptr(fr_a),
        _dptr(fr_b),
        _dptr(overlap_ao_out),
        _dptr(hcore_ao_out),
        _dptr(ao_coulomb_out),
        _dptr(metric_coulomb_out),
    )
    if rc != 0:
        raise RuntimeError(f"native df_nuclear_weights failed with status {rc}")

    return {
        "overlap_ao": overlap_ao_out,
        "hcore_ao": hcore_ao_out,
        "ao_coulomb": ao_coulomb_out,
        "metric/coulomb": metric_coulomb_out,
    }
