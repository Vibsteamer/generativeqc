"""Native AOT cuBLAS (and FP64 host-reference) bindings for ordinary DF-MP2."""

from __future__ import annotations

import ctypes as ct
import os
from pathlib import Path
from typing import Any

import numpy as np

from ._f12_shell_binding import _loaded_sink_paths, resolve_shell_binding
from ._f12_workspace import workspace_bytes

_D_PTR = ct.POINTER(ct.c_double)
_DISABLE_ENV = "GENERATIVEQC_F12_DISABLE_AOT_MP2F12"
_REQUIRE_ENV = "GENERATIVEQC_F12_REQUIRE_AOT_MP2F12"
_CUDA_LIB_ENVS = ("GENERATIVEQC_F12_SHELL_LIBRARY",)
_HOST_LIB_ENVS = (
    "GENERATIVEQC_F12_SHELL_LIBRARY",
    "GENERATIVEQC_F12_HOST_SINK",
    "GQC_F12_HOST_SHELL_LIBRARY",
)

def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def aot_mp2f12_disabled() -> bool:
    """Return True only when the explicit Gate 1 oracle opt-out env var is set."""
    return _env_truthy(_DISABLE_ENV)


def aot_mp2f12_required() -> bool:
    return _env_truthy(_REQUIRE_ENV)


def _check_mp2f12_env_flags() -> tuple[bool, bool]:
    req = aot_mp2f12_required()
    dis = aot_mp2f12_disabled()
    if req and dis:
        raise ValueError(
            f"Conflicting environment flags: {_REQUIRE_ENV} and {_DISABLE_ENV} "
            "cannot both be enabled"
        )
    return req, dis


def _dptr(arr: np.ndarray | None):
    if arr is None or arr.size == 0:
        return ct.cast(0, _D_PTR)
    return arr.ctypes.data_as(_D_PTR)


def _resolve_symbol(
    symbol: str,
    shell_library: Path | str | ct.CDLL | None = None,
    *,
    host_reference: bool = False,
) -> Any | None:
    return resolve_shell_binding(
        shell_library, lambda lib: getattr(lib, symbol, None),
        host_reference=host_reference,
        env_keys=_HOST_LIB_ENVS if host_reference else _CUDA_LIB_ENVS,
        mapped_paths=_loaded_sink_paths,
    )


def bind_ordinary_df_mp2(
    *,
    host_reference: bool,
    shell_library: Path | str | ct.CDLL | None = None,
) -> Any | None:
    """Bind gqc_f12_ordinary_df_mp2_{host_reference,cuda} if exported by shell_library."""
    req, dis = _check_mp2f12_env_flags()
    if dis:
        return None
    suffix = "_host_reference" if host_reference else "_cuda"
    symbol = f"gqc_f12_ordinary_df_mp2{suffix}"
    func = _resolve_symbol(symbol, shell_library, host_reference=host_reference)
    if func is None:
        if req:
            raise RuntimeError(
                f"{_REQUIRE_ENV}=1 requires native symbol {symbol!r}, but no matching library was found"
            )
        return None
    func.argtypes = (
        [ct.c_int] * 9
        + [ct.c_uint64]
        + [_D_PTR] * 15
    )
    func.restype = ct.c_int
    return func


def run_ordinary_df_mp2_native(
    func: Any,
    *,
    same_spin: bool,
    compute_response: bool,
    oa: int,
    fa: int,
    ob: int,
    fb: int,
    budget_bytes: int,
    coeff_a: np.ndarray,
    coeff_b: np.ndarray,
    eps_a: np.ndarray,
    eps_b: np.ndarray,
    ao_coul: np.ndarray,
    whitener: np.ndarray,
    occ_w_a: np.ndarray | None = None,
    occ_w_b: np.ndarray | None = None,
) -> tuple[
    tuple[float, float, float],
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Execute AOT ordinary DF-MP2 energy + optional unrelaxed response pullback."""
    budget_bytes = workspace_bytes(budget_bytes, signed=False)
    ca = np.ascontiguousarray(coeff_a, dtype=np.float64)
    cb = ca if same_spin else np.ascontiguousarray(coeff_b, dtype=np.float64)
    ea = np.ascontiguousarray(eps_a, dtype=np.float64)
    eb = ea if same_spin else np.ascontiguousarray(eps_b, dtype=np.float64)
    ao = np.ascontiguousarray(ao_coul, dtype=np.float64)
    wh = np.ascontiguousarray(whitener, dtype=np.float64)

    n_orb = int(ca.shape[0])
    n_aux = int(wh.shape[0])
    if ca.shape != (n_orb, n_orb) or cb.shape != (n_orb, n_orb):
        raise ValueError("ordinary DF-MP2 coefficients must have shape (n_orb, n_orb)")
    if ea.shape != (n_orb,) or eb.shape != (n_orb,):
        raise ValueError("ordinary DF-MP2 orbital energies must have shape (n_orb,)")
    if ao.shape != (n_aux, n_orb, n_orb) or wh.shape != (n_aux, n_aux):
        raise ValueError("ordinary DF-MP2 AO/whitener shapes do not match")

    has_occ_weights = occ_w_a is not None or occ_w_b is not None
    wa = (
        np.ascontiguousarray(occ_w_a, dtype=np.float64)
        if has_occ_weights and occ_w_a is not None
        else None
    )
    wb = (
        wa
        if (has_occ_weights and same_spin)
        else (
            np.ascontiguousarray(occ_w_b, dtype=np.float64)
            if has_occ_weights and occ_w_b is not None
            else None
        )
    )
    if has_occ_weights:
        if wa is None or wb is None or wa.shape != (oa - fa,) or wb.shape != (ob - fb,):
            raise ValueError("ordinary DF-MP2 occupied weights have invalid shapes")

    spin_energies = np.zeros(3, dtype=np.float64)
    if compute_response:
        fock_a = np.zeros((n_orb, n_orb), dtype=np.float64)
        fock_b = np.zeros((n_orb, n_orb), dtype=np.float64)
        bar_coeff_a = np.zeros((n_orb, n_orb), dtype=np.float64)
        bar_coeff_b = np.zeros((n_orb, n_orb), dtype=np.float64)
        bar_ao_coul = np.zeros((n_aux, n_orb, n_orb), dtype=np.float64)
        bar_metric = np.zeros((n_aux, n_aux), dtype=np.float64)
    else:
        fock_a = fock_b = bar_coeff_a = bar_coeff_b = bar_ao_coul = bar_metric = None

    rc = int(
        func(
            int(same_spin),
            int(compute_response),
            int(has_occ_weights),
            n_orb,
            n_aux,
            int(oa),
            int(fa),
            int(ob),
            int(fb),
            budget_bytes,
            _dptr(wa),
            _dptr(wb),
            _dptr(ca),
            _dptr(cb),
            _dptr(ea),
            _dptr(eb),
            _dptr(ao),
            _dptr(wh),
            _dptr(spin_energies),
            _dptr(fock_a),
            _dptr(fock_b),
            _dptr(bar_coeff_a),
            _dptr(bar_coeff_b),
            _dptr(bar_ao_coul),
            _dptr(bar_metric),
        )
    )
    if rc != 0:
        raise RuntimeError(f"native ordinary DF-MP2 kernel failed with status {rc}")

    return (
        (float(spin_energies[0]), float(spin_energies[1]), float(spin_energies[2])),
        fock_a,
        fock_b,
        bar_coeff_a,
        bar_coeff_b,
        bar_ao_coul,
        bar_metric,
    )
