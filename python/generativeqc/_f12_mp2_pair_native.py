"""Native AOT cuBLAS (and FP64 host-reference) bindings for ordinary DF-MP2 and MP2-F12 pairs."""

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


def bind_mp2_f12_pair_energies(
    *,
    host_reference: bool,
    shell_library: Path | str | ct.CDLL | None = None,
) -> Any | None:
    """Bind gqc_f12_mp2_pair_energies_{host_reference,cuda} if exported by shell_library."""
    req, dis = _check_mp2f12_env_flags()
    if dis:
        return None
    suffix = "_host_reference" if host_reference else "_cuda"
    symbol = f"gqc_f12_mp2_pair_energies{suffix}"
    func = _resolve_symbol(symbol, shell_library, host_reference=host_reference)
    if func is None:
        if req:
            raise RuntimeError(
                f"{_REQUIRE_ENV}=1 requires native symbol {symbol!r}, but no matching library was found"
            )
        return None
    func.argtypes = (
        [ct.c_int] * 12
        + [ct.c_uint64]
        + [_D_PTR] * 26
    )
    func.restype = ct.c_int
    return func


def _prep_raw_coul(
    arr: np.ndarray | None,
    *,
    aux: int,
    occ: int,
    ext: int,
    label: str,
) -> tuple[np.ndarray | None, int]:
    if arr is None or occ == 0:
        return None, 0
    if arr.ndim != 3 or arr.shape[0] != aux or arr.shape[1] < occ or arr.shape[2] != ext:
        raise ValueError(
            f"{label} must have shape ({aux}, >={occ}, {ext}), got {arr.shape}"
        )
    if arr.flags.c_contiguous and arr.dtype == np.float64:
        return arr, int(arr.shape[1])
    sliced = np.ascontiguousarray(arr[:, :occ, :], dtype=np.float64)
    return sliced, occ


def _prep_occ_3c(
    arr: np.ndarray | None,
    *,
    aux: int,
    occ: int,
    ext: int,
    label: str,
) -> np.ndarray | None:
    if arr is None or occ == 0:
        return None
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if out.shape != (aux, occ, ext):
        raise ValueError(f"{label} must have shape ({aux}, {occ}, {ext}), got {out.shape}")
    return out


def run_mp2_f12_pair_energies_native(
    func: Any,
    *,
    space: tuple[int, int, int, int, int],
    frozen_occupied: tuple[int, int],
    rhf_spatial: bool,
    include_ordinary: bool,
    budget_bytes: int,
    data: dict[str, np.ndarray],
    sources: dict[str, np.ndarray],
    fock: dict[str, np.ndarray],
    occ_w_a: np.ndarray | None = None,
    occ_w_b: np.ndarray | None = None,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    np.ndarray,
    np.ndarray | None,
]:
    """Execute AOT forward MP2-F12 3C(FIX) SP pair energies (RHF spatial fast path + UHF)."""
    budget_bytes = workspace_bytes(budget_bytes, signed=False)
    oa, ob, obs, ext, aux = space
    fa, fb = frozen_occupied
    if not (0 <= fa <= oa < obs and 0 <= fb <= ob < obs < ext and aux > 0):
        raise ValueError(
            "Invalid occupied/frozen spin spaces or orbital/fitting dimensions"
        )
    if rhf_spatial and (oa != ob or fa != fb):
        raise ValueError("rhf_spatial requires equal alpha/beta occupied and frozen counts")

    oa_act = oa - fa
    ob_act = ob - fb
    has_occ_weights = occ_w_a is not None or occ_w_b is not None
    wa = (
        np.ascontiguousarray(occ_w_a, dtype=np.float64)
        if has_occ_weights and occ_w_a is not None
        else None
    )
    wb = (
        wa
        if (has_occ_weights and rhf_spatial)
        else (
            np.ascontiguousarray(occ_w_b, dtype=np.float64)
            if has_occ_weights and occ_w_b is not None
            else None
        )
    )
    if has_occ_weights:
        if (oa_act > 0 and (wa is None or wa.shape != (oa_act,))) or (
            not rhf_spatial and ob_act > 0 and (wb is None or wb.shape != (ob_act,))
        ):
            raise ValueError("MP2-F12 pair energies occupied weights have invalid shapes")

    mg = np.ascontiguousarray(data["metric/geminal"], dtype=np.float64)
    msq = np.ascontiguousarray(data["metric/squared_geminal"], dtype=np.float64)
    mgc = np.ascontiguousarray(data["metric/geminal_coulomb"], dtype=np.float64)
    mdc = np.ascontiguousarray(data["metric/double_commutator"], dtype=np.float64)
    for name, mat in (
        ("metric/geminal", mg),
        ("metric/squared_geminal", msq),
        ("metric/geminal_coulomb", mgc),
        ("metric/double_commutator", mdc),
    ):
        if mat.shape != (aux, aux):
            raise ValueError(f"{name} must have shape ({aux}, {aux}), got {mat.shape}")

    fa_mat = np.ascontiguousarray(fock["fock_alpha"], dtype=np.float64)
    ka_mat = np.ascontiguousarray(fock["exchange_alpha"], dtype=np.float64)
    if fa_mat.shape != (ext, ext) or ka_mat.shape != (ext, ext):
        raise ValueError("alpha Fock/exchange matrices must have shape (ext, ext)")

    if rhf_spatial:
        fb_mat = fa_mat
        kb_mat = ka_mat
        cross = None
    else:
        fb_mat = np.ascontiguousarray(fock["fock_beta"], dtype=np.float64)
        kb_mat = np.ascontiguousarray(fock["exchange_beta"], dtype=np.float64)
        if fb_mat.shape != (ext, ext) or kb_mat.shape != (ext, ext):
            raise ValueError("beta Fock/exchange matrices must have shape (ext, ext)")
        if oa_act > 0 and ob_act > 0:
            cross = np.ascontiguousarray(data["cross_overlap"], dtype=np.float64)
            if cross.shape != (ext, ext):
                raise ValueError(
                    f"cross_overlap must have shape ({ext}, {ext}), got {cross.shape}"
                )
        else:
            cross = None

    fit_a = _prep_occ_3c(sources.get("fit_alpha") if oa_act > 0 else None, aux=aux, occ=oa, ext=ext, label="fit_alpha")
    rca, rca_n1 = _prep_raw_coul(sources.get("raw_coulomb_alpha") if oa_act > 0 else None, aux=aux, occ=oa, ext=ext, label="raw_coulomb_alpha")
    rga = _prep_occ_3c(data.get("raw/geminal/alpha") if oa_act > 0 else None, aux=aux, occ=oa, ext=ext, label="raw/geminal/alpha")
    rsqa = _prep_occ_3c(data.get("raw/squared_geminal/alpha") if oa_act > 0 else None, aux=aux, occ=oa, ext=ext, label="raw/squared_geminal/alpha")
    rgca = _prep_occ_3c(data.get("raw/geminal_coulomb/alpha") if oa_act > 0 else None, aux=aux, occ=oa, ext=ext, label="raw/geminal_coulomb/alpha")
    rdca = _prep_occ_3c(data.get("raw/double_commutator/alpha") if oa_act > 0 else None, aux=aux, occ=oa, ext=ext, label="raw/double_commutator/alpha")

    if rhf_spatial or ob_act == 0:
        fit_b = rgb = rsqb = rgcb = rdcb = None
        rcb, rcb_n1 = None, 0
    else:
        fit_b = _prep_occ_3c(sources.get("fit_beta"), aux=aux, occ=ob, ext=ext, label="fit_beta")
        rcb, rcb_n1 = _prep_raw_coul(sources.get("raw_coulomb_beta"), aux=aux, occ=ob, ext=ext, label="raw_coulomb_beta")
        rgb = _prep_occ_3c(data.get("raw/geminal/beta"), aux=aux, occ=ob, ext=ext, label="raw/geminal/beta")
        rsqb = _prep_occ_3c(data.get("raw/squared_geminal/beta"), aux=aux, occ=ob, ext=ext, label="raw/squared_geminal/beta")
        rgcb = _prep_occ_3c(data.get("raw/geminal_coulomb/beta"), aux=aux, occ=ob, ext=ext, label="raw/geminal_coulomb/beta")
        rdcb = _prep_occ_3c(data.get("raw/double_commutator/beta"), aux=aux, occ=ob, ext=ext, label="raw/double_commutator/beta")

    n_aa = (oa_act * (oa_act - 1)) // 2 if oa_act >= 2 else 0
    n_ab = oa_act * ob_act
    n_bb = (ob_act * (ob_act - 1)) // 2 if ob_act >= 2 else 0
    n_total = n_aa + n_ab + n_bb

    out_scalars = np.zeros(6, dtype=np.float64)
    pair_f12 = np.zeros(n_total, dtype=np.float64)
    pair_ord = np.zeros(n_total, dtype=np.float64) if include_ordinary else None

    rc = int(
        func(
            int(rhf_spatial),
            int(include_ordinary),
            int(has_occ_weights),
            int(oa),
            int(ob),
            int(fa),
            int(fb),
            int(obs),
            int(ext),
            int(aux),
            int(rca_n1),
            int(rcb_n1),
            budget_bytes,
            _dptr(wa),
            _dptr(wb),
            _dptr(fit_a),
            _dptr(fit_b),
            _dptr(rca),
            _dptr(rcb),
            _dptr(rga),
            _dptr(rgb),
            _dptr(rsqa),
            _dptr(rsqb),
            _dptr(rgca),
            _dptr(rgcb),
            _dptr(rdca),
            _dptr(rdcb),
            _dptr(mg),
            _dptr(msq),
            _dptr(mgc),
            _dptr(mdc),
            _dptr(fa_mat),
            _dptr(fb_mat),
            _dptr(ka_mat),
            _dptr(kb_mat),
            _dptr(cross),
            _dptr(out_scalars),
            _dptr(pair_f12),
            _dptr(pair_ord),
        )
    )
    if rc != 0:
        raise RuntimeError(f"native MP2-F12 pair energies kernel failed with status {rc}")

    return (
        (float(out_scalars[0]), float(out_scalars[1]), float(out_scalars[2])),
        (float(out_scalars[3]), float(out_scalars[4]), float(out_scalars[5])),
        pair_f12,
        pair_ord,
    )
