"""Native dynamic-dimension AOT bindings for conventional DF-MP2 Delta-CV."""

from __future__ import annotations

import ctypes as ct
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._f12_workspace import workspace_bytes

_D_PTR = ct.POINTER(ct.c_double)
_CDLL_CACHE: dict[Path, ct.CDLL] = {}
_C_INT_MAX = (1 << 31) - 1


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _check_c_int(value: int, name: str, *, low: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    ival = int(value)
    if ival < low or ival > _C_INT_MAX:
        raise ValueError(f"{name}={ival} exceeds C int range [{low}, {_C_INT_MAX}]")
    return ival

_SYMBOL_SPECS: dict[str, list] = {
    "df_jk_fock": [ct.c_int] * 6 + [ct.c_uint64] + [_D_PTR] * 7,
    "mp2_factors_fwd": [ct.c_int] * 5 + [ct.c_uint64] + [_D_PTR] * 5,
    "mp2_factors_rev": [ct.c_int] * 5 + [ct.c_uint64] + [_D_PTR] * 10,
    "mp2_pairs": [ct.c_int] * 10 + [ct.c_uint64] + [_D_PTR] * 14,
    "jk_mo_fock_fwd": [ct.c_int] * 6 + [ct.c_uint64] + [_D_PTR] * 9,
    "jk_fock_pullback": [ct.c_int] * 6 + [ct.c_uint64] + [_D_PTR] * 14,
    "cphf_matvec": [ct.c_int] * 6 + [ct.c_uint64] + [_D_PTR] * 8,
    "nuclear_source_weights": [ct.c_int] * 6 + [ct.c_uint64] + [_D_PTR] * 13,
}


def _dptr(arr: np.ndarray | None):
    if arr is None or arr.size == 0:
        return ct.cast(0, _D_PTR)
    return arr.ctypes.data_as(_D_PTR)


def _as_f64(arr: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite FP64 input passed to AOT DF-MP2 Delta-CV kernel")
    return out


def _checked_budget(budget_bytes: int, *, host_reference: bool) -> int:
    """Validate a device workspace budget; CUDA callers must state it explicitly.

    The C ABI maps 0 to fixed per-kernel defaults (1 GiB chunks, 2.5 GiB pairs,
    2.8 GiB J/K) that ignore the endpoint budget, so CUDA bindings reject it.
    Host-reference kernels do not allocate device memory and ignore the value.
    """
    if isinstance(budget_bytes, bool):
        raise ValueError("budget_bytes must be an integer, not bool")
    value = workspace_bytes(budget_bytes, signed=False)
    if value == 0 and not host_reference:
        raise ValueError(
            "CUDA AOT DF-MP2 Delta-CV kernels require an explicit positive budget_bytes; "
            "0 would select fixed native defaults instead of the endpoint budget"
        )
    return value


def _check_rc(rc: int, name: str) -> None:
    if rc == 0:
        return
    if rc == -2:
        raise NotImplementedError(
            f"AOT DF-MP2 Delta-CV CUDA kernel {name} is not implemented in this build"
        )
    raise RuntimeError(f"AOT DF-MP2 Delta-CV kernel {name} failed with status {rc}")


@dataclass(frozen=True)
class DeltaCVNativeBindings:
    """Bound ctypes entry points for all AOT DF-MP2 Delta-CV kernels."""

    library: Path
    host_reference: bool
    df_jk_fock: object
    mp2_factors_fwd: object
    mp2_factors_rev: object
    mp2_pairs: object
    jk_mo_fock_fwd: object
    jk_fock_pullback: object
    cphf_matvec: object
    nuclear_source_weights: object
    budget_bytes: int = 0

    def _effective_budget(self, budget_bytes: int | None) -> int:
        raw = self.budget_bytes if budget_bytes is None else budget_bytes
        return _checked_budget(raw, host_reference=self.host_reference)

    def run_df_jk_fock(
        self,
        B_jk: np.ndarray,
        D_tot: np.ndarray,
        Co_a: np.ndarray,
        Co_b: np.ndarray | None,
        *,
        restricted: bool,
        q_chunk: int = 0,
        budget_bytes: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        b_arr = _as_f64(B_jk)
        d_arr = _as_f64(D_tot)
        coa = _as_f64(Co_a)
        if b_arr.ndim != 3:
            raise ValueError("Inconsistent dimensions for AOT DF-HF J/K Fock build")
        q, n, n2 = b_arr.shape
        if n != n2 or d_arr.shape != (n, n) or coa.ndim != 2 or coa.shape[0] != n:
            raise ValueError("Inconsistent dimensions for AOT DF-HF J/K Fock build")
        q = _check_c_int(q, "q", low=1)
        n = _check_c_int(n, "n", low=1)
        oa = _check_c_int(coa.shape[1], "oa", low=0)
        cob = None
        ob = 0
        if not restricted:
            if Co_b is None:
                raise ValueError("UHF DF-HF J/K Fock build requires Co_b")
            cob = _as_f64(Co_b)
            if cob.ndim != 2 or cob.shape[0] != n:
                raise ValueError("Inconsistent beta occupied dimensions for DF-HF J/K")
            ob = _check_c_int(cob.shape[1], "ob", low=0)
        q_chunk = _check_c_int(q_chunk, "q_chunk", low=0)
        eff_budget = self._effective_budget(budget_bytes)

        j_out = np.empty((n, n), dtype=np.float64)
        ka_out = np.empty((n, n), dtype=np.float64)
        kb_out = None if restricted else np.empty((n, n), dtype=np.float64)

        rc = self.df_jk_fock(  # type: ignore[operator]
            q,
            n,
            oa,
            ob,
            int(restricted),
            q_chunk,
            eff_budget,
            _dptr(b_arr),
            _dptr(d_arr),
            _dptr(coa),
            _dptr(cob),
            _dptr(j_out),
            _dptr(ka_out),
            _dptr(kb_out),
        )
        _check_rc(rc, "df_jk_fock")
        return j_out, ka_out, kb_out

    def run_mp2_factors_fwd(
        self,
        ao_coulomb: np.ndarray,
        whitener: np.ndarray,
        co: np.ndarray,
        cv: np.ndarray,
        *,
        q_chunk: int = 0,
        budget_bytes: int | None = None,
    ) -> np.ndarray:
        ao = _as_f64(ao_coulomb)
        w = _as_f64(whitener)
        co_arr = _as_f64(co)
        cv_arr = _as_f64(cv)
        if ao.ndim != 3:
            raise ValueError("Inconsistent dimensions for AOT MP2Fit factor forward")
        q, n, n2 = ao.shape
        if (
            n != n2
            or w.shape != (q, q)
            or co_arr.ndim != 2
            or cv_arr.ndim != 2
            or co_arr.shape[0] != n
            or cv_arr.shape[0] != n
        ):
            raise ValueError("Inconsistent dimensions for AOT MP2Fit factor forward")
        q = _check_c_int(q, "q", low=1)
        n = _check_c_int(n, "n", low=1)
        o_act = _check_c_int(co_arr.shape[1], "o_act", low=0)
        n_vir = _check_c_int(cv_arr.shape[1], "n_vir", low=0)
        q_chunk = _check_c_int(q_chunk, "q_chunk", low=0)
        eff_budget = self._effective_budget(budget_bytes)
        b_out = np.empty((q, o_act, n_vir), dtype=np.float64)
        rc = self.mp2_factors_fwd(  # type: ignore[operator]
            q,
            n,
            o_act,
            n_vir,
            q_chunk,
            eff_budget,
            _dptr(ao),
            _dptr(w),
            _dptr(co_arr),
            _dptr(cv_arr),
            _dptr(b_out),
        )
        _check_rc(rc, "mp2_factors_fwd")
        return b_out

    def run_mp2_factors_rev(
        self,
        ao_coulomb: np.ndarray,
        whitener: np.ndarray,
        co: np.ndarray,
        cv: np.ndarray,
        B: np.ndarray,
        bar_B: np.ndarray,
        *,
        q_chunk: int = 0,
        budget_bytes: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ao = _as_f64(ao_coulomb)
        w = _as_f64(whitener)
        co_arr = _as_f64(co)
        cv_arr = _as_f64(cv)
        b_arr = _as_f64(B)
        bb_arr = _as_f64(bar_B)
        if ao.ndim != 3 or co_arr.ndim != 2 or cv_arr.ndim != 2:
            raise ValueError("Inconsistent dimensions for AOT MP2Fit factor reverse")
        q, n, n2 = ao.shape
        o_act = co_arr.shape[1]
        n_vir = cv_arr.shape[1]
        if (
            n != n2
            or w.shape != (q, q)
            or co_arr.shape != (n, o_act)
            or cv_arr.shape != (n, n_vir)
            or b_arr.shape != (q, o_act, n_vir)
            or bb_arr.shape != (q, o_act, n_vir)
        ):
            raise ValueError("Inconsistent dimensions for AOT MP2Fit factor reverse")
        q = _check_c_int(q, "q", low=1)
        n = _check_c_int(n, "n", low=1)
        o_act = _check_c_int(o_act, "o_act", low=0)
        n_vir = _check_c_int(n_vir, "n_vir", low=0)
        q_chunk = _check_c_int(q_chunk, "q_chunk", low=0)
        eff_budget = self._effective_budget(budget_bytes)
        bar_co = np.empty((n, o_act), dtype=np.float64)
        bar_cv = np.empty((n, n_vir), dtype=np.float64)
        bar_ao = np.empty((q, n, n), dtype=np.float64)
        metric_bar = np.empty((q, q), dtype=np.float64)
        rc = self.mp2_factors_rev(  # type: ignore[operator]
            q,
            n,
            o_act,
            n_vir,
            q_chunk,
            eff_budget,
            _dptr(ao),
            _dptr(w),
            _dptr(co_arr),
            _dptr(cv_arr),
            _dptr(b_arr),
            _dptr(bb_arr),
            _dptr(bar_co),
            _dptr(bar_cv),
            _dptr(bar_ao),
            _dptr(metric_bar),
        )
        _check_rc(rc, "mp2_factors_rev")
        return bar_co, bar_cv, bar_ao, metric_bar

    def run_mp2_pairs(
        self,
        *,
        restricted: bool,
        need_cotangents: bool,
        B_a: np.ndarray | None,
        B_b: np.ndarray | None,
        eo_a: np.ndarray,
        eo_b: np.ndarray | None,
        ev_a: np.ndarray,
        ev_b: np.ndarray | None,
        sh_a: int,
        sh_b: int,
        tile_limit: int = 0,
        budget_bytes: int | None = None,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        eoa = _as_f64(eo_a)
        eva = _as_f64(ev_a)
        oa = eoa.size
        va = eva.size
        ba = _as_f64(B_a) if (B_a is not None and oa > 0 and va > 0) else None
        if restricted:
            ob, vb = oa, va
            bb = None
            eob = None
            evb = None
            q = ba.shape[0] if ba is not None else 1
        else:
            assert eo_b is not None and ev_b is not None
            eob = _as_f64(eo_b)
            evb = _as_f64(ev_b)
            ob = eob.size
            vb = evb.size
            bb = _as_f64(B_b) if (B_b is not None and ob > 0 and vb > 0) else None
            if ba is not None:
                q = ba.shape[0]
                if bb is not None and bb.shape[0] != q:
                    raise ValueError("Alpha and beta MP2Fit B factors must share auxiliary Q")
            elif bb is not None:
                q = bb.shape[0]
            else:
                q = 1

        if ba is not None and ba.shape != (q, oa, va):
            raise ValueError(f"B_a must have shape {(q, oa, va)}, got {ba.shape}")
        if bb is not None and bb.shape != (q, ob, vb):
            raise ValueError(f"B_b must have shape {(q, ob, vb)}, got {bb.shape}")
        q = _check_c_int(q, "q", low=1)
        oa = _check_c_int(oa, "oa", low=0)
        ob = _check_c_int(ob, "ob", low=0)
        va = _check_c_int(va, "va", low=0)
        vb = _check_c_int(vb, "vb", low=0)
        _check_c_int(sh_a, "sh_a", low=0)
        _check_c_int(sh_b, "sh_b", low=0)
        _check_c_int(tile_limit, "tile_limit", low=0)
        eff_budget = self._effective_budget(budget_bytes)

        cv_e = np.zeros(3, dtype=np.float64)
        fc_e = np.zeros(3, dtype=np.float64)
        bar_Ba = np.empty((q, oa, va), dtype=np.float64) if (need_cotangents and oa > 0 and va > 0) else None
        bar_Bb = (
            np.empty((q, ob, vb), dtype=np.float64)
            if (need_cotangents and not restricted and ob > 0 and vb > 0)
            else None
        )
        bar_Fo_a = np.empty((oa, oa), dtype=np.float64) if (need_cotangents and oa > 0 and va > 0) else None
        bar_Fo_b = (
            np.empty((ob, ob), dtype=np.float64)
            if (need_cotangents and not restricted and ob > 0 and vb > 0)
            else None
        )
        bar_Fv_a = np.empty((va, va), dtype=np.float64) if (need_cotangents and oa > 0 and va > 0) else None
        bar_Fv_b = (
            np.empty((vb, vb), dtype=np.float64)
            if (need_cotangents and not restricted and ob > 0 and vb > 0)
            else None
        )

        rc = self.mp2_pairs(  # type: ignore[operator]
            int(restricted),
            int(need_cotangents),
            q,
            oa,
            ob,
            va,
            vb,
            int(sh_a),
            int(sh_b),
            int(tile_limit),
            eff_budget,
            _dptr(ba),
            _dptr(bb),
            _dptr(eoa),
            _dptr(eob),
            _dptr(eva),
            _dptr(evb),
            _dptr(cv_e),
            _dptr(fc_e),
            _dptr(bar_Ba),
            _dptr(bar_Bb),
            _dptr(bar_Fo_a),
            _dptr(bar_Fo_b),
            _dptr(bar_Fv_a),
            _dptr(bar_Fv_b),
        )
        _check_rc(rc, "mp2_pairs")
        return (
            (float(cv_e[0]), float(cv_e[1]), float(cv_e[2])),
            (float(fc_e[0]), float(fc_e[1]), float(fc_e[2])),
            bar_Ba,
            bar_Bb,
            bar_Fo_a,
            bar_Fo_b,
            bar_Fv_a,
            bar_Fv_b,
        )

    def run_jk_mo_fock_fwd(
        self,
        ao_coulomb: np.ndarray,
        hcore_ao: np.ndarray,
        whitener: np.ndarray,
        C_a: np.ndarray,
        C_b: np.ndarray | None,
        oa: int,
        ob: int,
        *,
        restricted: bool,
        q_chunk: int = 0,
        budget_bytes: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
        ao = _as_f64(ao_coulomb)
        h = _as_f64(hcore_ao)
        w = _as_f64(whitener)
        ca = _as_f64(C_a)
        cb = None if restricted else _as_f64(C_b)  # type: ignore[arg-type]
        if ao.ndim != 3:
            raise ValueError("Inconsistent dimensions for AOT JK MO Fock forward")
        q, n, n2 = ao.shape
        if n != n2 or h.shape != (n, n) or w.shape != (q, q) or ca.shape != (n, n):
            raise ValueError("Inconsistent dimensions for AOT JK MO Fock forward")
        if not restricted and (cb is None or cb.shape != (n, n)):
            raise ValueError("Inconsistent beta coefficients for AOT JK MO Fock forward")
        q = _check_c_int(q, "q", low=1)
        n = _check_c_int(n, "n", low=1)
        oa = _check_c_int(oa, "oa", low=0)
        ob = _check_c_int(ob, "ob", low=0)
        q_chunk = _check_c_int(q_chunk, "q_chunk", low=0)
        eff_budget = self._effective_budget(budget_bytes)

        df_a = np.empty((q, n, n), dtype=np.float64)
        df_b = None if restricted else np.empty((q, n, n), dtype=np.float64)
        fock_a = np.empty((n, n), dtype=np.float64)
        fock_b = None if restricted else np.empty((n, n), dtype=np.float64)

        rc = self.jk_mo_fock_fwd(  # type: ignore[operator]
            q,
            n,
            oa,
            ob,
            int(restricted),
            q_chunk,
            eff_budget,
            _dptr(ao),
            _dptr(h),
            _dptr(w),
            _dptr(ca),
            _dptr(cb),
            _dptr(df_a),
            _dptr(df_b),
            _dptr(fock_a),
            _dptr(fock_b),
        )
        _check_rc(rc, "jk_mo_fock_fwd")
        return df_a, df_b, fock_a, fock_b

    def run_jk_fock_pullback(
        self,
        ao_coulomb: np.ndarray,
        hcore_ao: np.ndarray,
        whitener: np.ndarray,
        C_a: np.ndarray,
        C_b: np.ndarray | None,
        df_a: np.ndarray,
        df_b: np.ndarray | None,
        S_a: np.ndarray,
        S_b: np.ndarray | None,
        oa: int,
        ob: int,
        *,
        restricted: bool,
        q_chunk: int = 0,
        budget_bytes: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ao = _as_f64(ao_coulomb)
        h = _as_f64(hcore_ao)
        w = _as_f64(whitener)
        ca = _as_f64(C_a)
        dfa = _as_f64(df_a)
        sa = _as_f64(S_a)
        if ao.ndim != 3:
            raise ValueError("Inconsistent ao_coulomb ndim for AOT JK Fock pullback")
        q, n, n2 = ao.shape
        if (
            n != n2
            or h.shape != (n, n)
            or w.shape != (q, q)
            or ca.shape != (n, n)
            or dfa.shape != (q, n, n)
            or sa.shape != (n, n)
        ):
            raise ValueError("Inconsistent dimensions for AOT JK Fock pullback")
        q = _check_c_int(q, "q", low=1)
        n = _check_c_int(n, "n", low=1)
        _check_c_int(oa, "oa", low=0)
        _check_c_int(ob, "ob", low=0)
        _check_c_int(q_chunk, "q_chunk", low=0)
        eff_budget = self._effective_budget(budget_bytes)

        share_spin = restricted and (S_b is None or np.array_equal(sa, S_b))
        cb = None if share_spin else _as_f64(C_a if restricted else C_b)  # type: ignore[arg-type]
        dfb = None if share_spin else _as_f64(df_a if restricted else df_b)  # type: ignore[arg-type]
        sb = None if share_spin else _as_f64(S_b)  # type: ignore[arg-type]
        if not share_spin and (
            cb is None
            or dfb is None
            or sb is None
            or cb.shape != (n, n)
            or dfb.shape != (q, n, n)
            or sb.shape != (n, n)
        ):
            raise ValueError("Inconsistent beta dimensions for AOT JK Fock pullback")

        bar_ca = np.empty((n, n), dtype=np.float64)
        bar_cb = None if share_spin else np.empty((n, n), dtype=np.float64)
        bar_h = np.empty((n, n), dtype=np.float64)
        bar_ao = np.empty((q, n, n), dtype=np.float64)
        bar_met = np.empty((q, q), dtype=np.float64)

        rc = self.jk_fock_pullback(  # type: ignore[operator]
            q,
            n,
            int(oa),
            int(ob),
            int(share_spin),
            int(q_chunk),
            eff_budget,
            _dptr(ao),
            _dptr(h),
            _dptr(w),
            _dptr(ca),
            _dptr(cb),
            _dptr(dfa),
            _dptr(dfb),
            _dptr(sa),
            _dptr(sb),
            _dptr(bar_ca),
            _dptr(bar_cb),
            _dptr(bar_h),
            _dptr(bar_ao),
            _dptr(bar_met),
        )
        _check_rc(rc, "jk_fock_pullback")
        return bar_ca, (bar_ca.copy() if share_spin else bar_cb), bar_h, bar_ao, bar_met  # type: ignore[return-value]

    def run_cphf_matvec(
        self,
        df_a: np.ndarray,
        df_b: np.ndarray | None,
        gap_a: np.ndarray,
        gap_b: np.ndarray | None,
        trial_a: np.ndarray,
        trial_b: np.ndarray | None,
        oa: int,
        ob: int,
        *,
        restricted: bool,
        q_chunk: int = 0,
        budget_bytes: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        dfa = _as_f64(df_a)
        if dfa.ndim != 3 or dfa.shape[1] != dfa.shape[2]:
            raise ValueError("Inconsistent df_a shape for AOT CPHF matvec")
        q, n, _ = dfa.shape
        q = _check_c_int(q, "q", low=1)
        n = _check_c_int(n, "n", low=1)
        oa = _check_c_int(oa, "oa", low=0)
        ob = _check_c_int(ob, "ob", low=0)
        q_chunk = _check_c_int(q_chunk, "q_chunk", low=0)
        eff_budget = self._effective_budget(budget_bytes)
        if oa > n or ob > n:
            raise ValueError("Occupied count exceeds orbital count in AOT CPHF matvec")
        va = n - oa
        vb = n - ob
        ga = _as_f64(gap_a) if oa > 0 else None
        ta = _as_f64(trial_a) if oa > 0 else None
        if oa > 0 and (ga is None or ta is None or ga.shape != (va, oa) or ta.shape != (va, oa)):
            raise ValueError("Inconsistent alpha gap/trial shapes for AOT CPHF matvec")
        dfb = None if restricted else (_as_f64(df_b) if df_b is not None else None)
        if not restricted and dfb is not None and dfb.shape != (q, n, n):
            raise ValueError("Inconsistent df_b shape for AOT CPHF matvec")
        gb = None if (restricted or ob == 0) else _as_f64(gap_b)  # type: ignore[arg-type]
        tb = None if (restricted or ob == 0) else _as_f64(trial_b)  # type: ignore[arg-type]
        if not restricted and ob > 0 and (
            dfb is None or gb is None or tb is None or gb.shape != (vb, ob) or tb.shape != (vb, ob)
        ):
            raise ValueError("Inconsistent beta shapes for AOT CPHF matvec")

        out_a = np.empty((va, oa), dtype=np.float64) if oa > 0 else np.empty((va, 0), dtype=np.float64)
        out_b = (
            None
            if restricted
            else (
                np.empty((vb, ob), dtype=np.float64)
                if ob > 0
                else np.empty((vb, 0), dtype=np.float64)
            )
        )

        rc = self.cphf_matvec(  # type: ignore[operator]
            q,
            n,
            int(oa),
            int(ob),
            int(restricted),
            int(q_chunk),
            eff_budget,
            _dptr(dfa),
            _dptr(dfb),
            _dptr(ga),
            _dptr(gb),
            _dptr(ta),
            _dptr(tb),
            _dptr(out_a),
            _dptr(out_b),
        )
        _check_rc(rc, "cphf_matvec")
        return out_a, out_b

    def run_nuclear_source_weights(
        self,
        whitener: np.ndarray,
        C_a: np.ndarray,
        C_b: np.ndarray | None,
        df_a: np.ndarray,
        df_b: np.ndarray | None,
        p_a: np.ndarray,
        p_b: np.ndarray | None,
        sm_a: np.ndarray,
        sm_b: np.ndarray | None,
        oa: int,
        ob: int,
        *,
        restricted: bool,
        q_chunk: int = 0,
        budget_bytes: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        w = _as_f64(whitener)
        ca = _as_f64(C_a)
        dfa = _as_f64(df_a)
        pa = _as_f64(p_a)
        sma = _as_f64(sm_a)
        if dfa.ndim != 3 or dfa.shape[1] != dfa.shape[2]:
            raise ValueError("Inconsistent df_a shape for AOT nuclear source weights")
        q, n, _ = dfa.shape
        if w.shape != (q, q) or ca.shape != (n, n) or pa.shape != (n, n) or sma.shape != (n, n):
            raise ValueError("Inconsistent alpha shapes for AOT nuclear source weights")
        q = _check_c_int(q, "q", low=1)
        n = _check_c_int(n, "n", low=1)
        _check_c_int(oa, "oa", low=0)
        _check_c_int(ob, "ob", low=0)
        _check_c_int(q_chunk, "q_chunk", low=0)
        eff_budget = self._effective_budget(budget_bytes)

        cb = None if restricted else _as_f64(C_b)  # type: ignore[arg-type]
        dfb = None if restricted else _as_f64(df_b)  # type: ignore[arg-type]
        pb = None if restricted else _as_f64(p_b)  # type: ignore[arg-type]
        smb = None if restricted else _as_f64(sm_b)  # type: ignore[arg-type]
        if not restricted and (
            cb is None
            or dfb is None
            or pb is None
            or smb is None
            or cb.shape != (n, n)
            or dfb.shape != (q, n, n)
            or pb.shape != (n, n)
            or smb.shape != (n, n)
        ):
            raise ValueError("Inconsistent beta shapes for AOT nuclear source weights")

        bar_ds = np.empty((n, n), dtype=np.float64)
        bar_dh = np.empty((n, n), dtype=np.float64)
        bar_dr = np.empty((q, n, n), dtype=np.float64)
        bar_dj = np.empty((q, q), dtype=np.float64)

        rc = self.nuclear_source_weights(  # type: ignore[operator]
            q,
            n,
            int(oa),
            int(ob),
            int(restricted),
            int(q_chunk),
            eff_budget,
            _dptr(w),
            _dptr(ca),
            _dptr(cb),
            _dptr(dfa),
            _dptr(dfb),
            _dptr(pa),
            _dptr(pb),
            _dptr(sma),
            _dptr(smb),
            _dptr(bar_ds),
            _dptr(bar_dh),
            _dptr(bar_dr),
            _dptr(bar_dj),
        )
        _check_rc(rc, "nuclear_source_weights")
        return bar_ds, bar_dh, bar_dr, bar_dj


def bind_delta_cv_native(
    library: Path | None,
    *,
    host_reference: bool,
    budget_bytes: int = 0,
) -> DeltaCVNativeBindings | None:
    """Bind all AOT DF-MP2 Delta-CV kernels from `library`.

    In CUDA mode (`host_reference=False`), missing libraries, missing symbols,
    or disabling AOT raises RuntimeError instead of silently falling back to
    TensorIR, and `budget_bytes` must be the caller's positive device budget.
    Explicit CPU/reference mode (`host_reference=True`) may opt out for Gate-1
    oracle comparisons when REQUIRE env flags are not set.
    """
    eff_budget = _checked_budget(budget_bytes, host_reference=host_reference)
    require = _env_truthy("GENERATIVEQC_F12_REQUIRE_AOT_CV") or _env_truthy(
        "GENERATIVEQC_F12_REQUIRE_AOT_DELTACV"
    )
    disable = _env_truthy("GENERATIVEQC_F12_DISABLE_AOT_CV") or _env_truthy(
        "GENERATIVEQC_F12_DISABLE_AOT_DELTACV"
    )
    if require and disable:
        raise ValueError(
            "Conflicting environment flags: GENERATIVEQC_F12_REQUIRE_AOT_CV/"
            "GENERATIVEQC_F12_REQUIRE_AOT_DELTACV and GENERATIVEQC_F12_DISABLE_AOT_CV/"
            "GENERATIVEQC_F12_DISABLE_AOT_DELTACV cannot both be enabled"
        )
    if disable:
        if not host_reference:
            raise RuntimeError(
                "Disabling native CUDA AOT DF-MP2 Delta-CV kernels is not permitted; "
                "use explicit host_reference=True for CPU/TensorIR oracle comparisons"
            )
        return None
    if library is None:
        if not host_reference or require:
            raise RuntimeError(
                "Required AOT DF-MP2 Delta-CV library is missing (library=None); "
                "silent fallback to TensorIR is disabled"
            )
        return None
    resolved = Path(library).resolve()
    cdll = _CDLL_CACHE.get(resolved)
    if cdll is None:
        cdll = ct.CDLL(str(resolved))
        _CDLL_CACHE[resolved] = cdll

    suffix = "_host_reference" if host_reference else "_cuda"
    funcs: dict[str, object] = {}
    for key, argtypes in _SYMBOL_SPECS.items():
        sym = f"gqc_f12_cv_{key}{suffix}"
        fn = getattr(cdll, sym, None)
        if fn is None:
            if not host_reference or require:
                raise RuntimeError(
                    f"Required AOT DF-MP2 Delta-CV symbol {sym!r} missing from {resolved}; "
                    "silent fallback to TensorIR is disabled"
                )
            return None
        fn.argtypes = argtypes
        fn.restype = ct.c_int
        funcs[key] = fn

    return DeltaCVNativeBindings(
        library=resolved,
        host_reference=host_reference,
        budget_bytes=eff_budget,
        **funcs,  # type: ignore[arg-type]
    )
