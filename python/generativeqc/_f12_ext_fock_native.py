"""Native FP64 cuBLAS (and matching host-reference) bindings for extended AO DF Fock."""

from __future__ import annotations

import ctypes as ct
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._f12_shell_binding import resolve_shell_binding

_D_PTR = ct.POINTER(ct.c_double)
_NULL_D = ct.cast(0, _D_PTR)
_CDLL_CACHE: dict[str, ct.CDLL] = {}
_C_INT_MAX = (1 << 31) - 1


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _dptr(arr: np.ndarray | None):
    if arr is None or arr.size == 0:
        return _NULL_D
    return arr.ctypes.data_as(_D_PTR)


def _check_c_int(value: int, name: str, *, low: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    ival = int(value)
    if ival < low or ival > _C_INT_MAX:
        raise ValueError(f"{name}={ival} is outside valid C int range [{low}, {_C_INT_MAX}]")
    return ival


def _check_finite_tiled(arr: np.ndarray, name: str) -> None:
    if arr.size == 0:
        return
    if arr.ndim == 3 and arr.nbytes > (16 << 20):
        step = max(1, (16 << 20) // max(1, arr.shape[1] * arr.shape[2] * arr.itemsize))
        for i in range(0, arr.shape[0], step):
            if not np.isfinite(arr[i : i + step]).all():
                raise ValueError(f"{name} contains non-finite values")
        return
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")


def _as_checked_f64_in(
    arr: np.ndarray,
    expected_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name} must be a float64 numpy ndarray")
    if arr.dtype != np.float64:
        raise ValueError(f"{name} must have dtype float64, got {arr.dtype}")
    if arr.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")
    out = np.ascontiguousarray(arr, dtype=np.float64)
    _check_finite_tiled(out, name)
    return out


def _check_f64_out(
    arr: np.ndarray,
    expected_shape: tuple[int, ...],
    name: str,
    *,
    check_finite: bool = True,
) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name} must be a float64 numpy ndarray")
    if arr.dtype != np.float64:
        raise ValueError(f"{name} must have dtype float64, got {arr.dtype}")
    if not arr.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous float64 array")
    if not arr.flags.writeable:
        raise ValueError(f"{name} must be a writeable float64 array")
    if arr.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")
    if check_finite:
        _check_finite_tiled(arr, name)
    return arr


def _load_cdll(library: Path | str | ct.CDLL) -> tuple[ct.CDLL, str | None]:
    if isinstance(library, ct.CDLL):
        return library, getattr(library, "_name", None)
    resolved = str(Path(library).resolve())
    cdll = _CDLL_CACHE.get(resolved)
    if cdll is None:
        cdll = ct.CDLL(resolved)
        _CDLL_CACHE[resolved] = cdll
    return cdll, resolved


@dataclass(frozen=True)
class ExtFockNative:
    """Bound native symbol bundle for extended spin Fock forward and pullback."""

    half_fwd_fn: ct._CFuncPtr
    fwd_fn: ct._CFuncPtr
    j_fwd_fn: ct._CFuncPtr
    congruence_rev_fn: ct._CFuncPtr
    j_rev_fn: ct._CFuncPtr
    rev_fn: ct._CFuncPtr
    half_rev_fn: ct._CFuncPtr
    host_reference: bool
    library_path: str | None = None

    def half_fwd(self, raw_tile: np.ndarray, occ: np.ndarray) -> np.ndarray:
        if not isinstance(raw_tile, np.ndarray) or raw_tile.ndim != 3:
            raise ValueError("raw_tile must be a 3D float64 ndarray of shape (qt, n, n)")
        qt, n, n2 = raw_tile.shape
        if n != n2:
            raise ValueError(f"raw_tile must be square on AO axes, got {raw_tile.shape}")
        qt = _check_c_int(qt, "qt", low=1)
        n = _check_c_int(n, "n", low=1)
        if not isinstance(occ, np.ndarray) or occ.ndim != 2:
            raise ValueError("occ must be a 2D float64 ndarray of shape (n, o)")
        o = _check_c_int(occ.shape[1], "o", low=1)
        raw_c = _as_checked_f64_in(raw_tile, (qt, n, n), "raw_tile")
        occ_c = _as_checked_f64_in(occ, (n, o), "occ")
        out = np.empty((qt, o, n), dtype=np.float64)
        rc = self.half_fwd_fn(qt, n, o, _dptr(raw_c), _dptr(occ_c), _dptr(out))
        if rc != 0:
            raise RuntimeError(f"gqc_f12_ext_fock_half_fwd failed with status {rc}")
        return out

    def fwd(
        self,
        whitener: np.ndarray,
        occ_a: np.ndarray | None,
        occ_b: np.ndarray | None,
        R_a: np.ndarray | None,
        R_b: np.ndarray | None,
        *,
        same_occ_frame: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(whitener, np.ndarray) or whitener.ndim != 2 or whitener.shape[0] != whitener.shape[1]:
            raise ValueError("whitener must be a square 2D float64 ndarray")
        q = _check_c_int(whitener.shape[0], "q", low=1)
        w_c = _as_checked_f64_in(whitener, (q, q), "whitener")
        if occ_a is not None:
            if not isinstance(occ_a, np.ndarray) or occ_a.ndim != 2:
                raise ValueError("occ_a must be a 2D float64 ndarray")
            n = _check_c_int(occ_a.shape[0], "n", low=1)
            oa = _check_c_int(occ_a.shape[1], "oa", low=0)
        elif occ_b is not None:
            if not isinstance(occ_b, np.ndarray) or occ_b.ndim != 2:
                raise ValueError("occ_b must be a 2D float64 ndarray")
            n = _check_c_int(occ_b.shape[0], "n", low=1)
            oa = 0
        else:
            raise ValueError("fwd requires at least one spin frame to infer AO dimension")
        if occ_b is not None:
            if not isinstance(occ_b, np.ndarray) or occ_b.ndim != 2:
                raise ValueError("occ_b must be a 2D float64 ndarray")
            if occ_b.shape[0] != n:
                raise ValueError(f"occ_b row count {occ_b.shape[0]} disagrees with n={n}")
            ob = _check_c_int(occ_b.shape[1], "ob", low=0)
        else:
            ob = 0
        if same_occ_frame and oa != ob:
            raise ValueError(f"same_occ_frame=True requires oa == ob, got {oa} vs {ob}")

        oa_c = _as_checked_f64_in(occ_a, (n, oa), "occ_a") if oa > 0 else None
        ob_c = None if (ob == 0 or same_occ_frame) else _as_checked_f64_in(occ_b, (n, ob), "occ_b")
        if oa > 0:
            if R_a is None:
                raise ValueError("R_a is required when oa > 0")
            ra_c = _as_checked_f64_in(R_a, (q, oa, n), "R_a")
        else:
            ra_c = None
        if ob > 0 and not same_occ_frame:
            if R_b is None:
                raise ValueError("R_b is required when ob > 0 and not same_occ_frame")
            rb_c = _as_checked_f64_in(R_b, (q, ob, n), "R_b")
        else:
            rb_c = None

        D_a = np.empty((q, oa, n), dtype=np.float64) if oa > 0 else None
        if ob == 0:
            D_b = None
        elif same_occ_frame:
            D_b = D_a
        else:
            D_b = np.empty((q, ob, n), dtype=np.float64)

        K_a = np.empty((n, n), dtype=np.float64)
        K_b = np.empty((n, n), dtype=np.float64)
        rho = np.empty(q, dtype=np.float64)
        rc = self.fwd_fn(
            q,
            n,
            oa,
            ob,
            int(bool(same_occ_frame)),
            _dptr(w_c),
            _dptr(oa_c),
            _dptr(ob_c),
            _dptr(ra_c),
            _dptr(rb_c),
            _dptr(D_a),
            _dptr(None if same_occ_frame else D_b),
            _dptr(K_a),
            _dptr(K_b),
            _dptr(rho),
        )
        if rc != 0:
            raise RuntimeError(f"gqc_f12_ext_fock_fwd failed with status {rc}")
        return D_a, D_b, K_a, K_b, rho

    def j_fwd(
        self, raw_tile: np.ndarray, rho_tile: np.ndarray, j_acc: np.ndarray
    ) -> None:
        if not isinstance(raw_tile, np.ndarray) or raw_tile.ndim != 3:
            raise ValueError("raw_tile must be a 3D float64 ndarray of shape (qt, n, n)")
        qt, n, n2 = raw_tile.shape
        if n != n2:
            raise ValueError(f"raw_tile must be square on AO axes, got {raw_tile.shape}")
        qt = _check_c_int(qt, "qt", low=1)
        n = _check_c_int(n, "n", low=1)
        raw_c = _as_checked_f64_in(raw_tile, (qt, n, n), "raw_tile")
        rho_c = _as_checked_f64_in(rho_tile, (qt,), "rho_tile")
        _check_f64_out(j_acc, (n, n), "j_acc", check_finite=True)
        rc = self.j_fwd_fn(qt, n, _dptr(raw_c), _dptr(rho_c), _dptr(j_acc))
        if rc != 0:
            raise RuntimeError(f"gqc_f12_ext_fock_j_fwd failed with status {rc}")

    def congruence_rev(
        self,
        coeff: np.ndarray,
        f_primal: np.ndarray,
        bar_m: np.ndarray,
        bar_coeff_acc: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(coeff, np.ndarray) or coeff.ndim != 2:
            raise ValueError("coeff must be a 2D float64 ndarray of shape (n, ext)")
        n = _check_c_int(coeff.shape[0], "n", low=1)
        ext = _check_c_int(coeff.shape[1], "ext", low=1)
        c_c = _as_checked_f64_in(coeff, (n, ext), "coeff")
        f_c = _as_checked_f64_in(f_primal, (n, n), "f_primal")
        bm_c = _as_checked_f64_in(bar_m, (ext, ext), "bar_m")
        _check_f64_out(bar_coeff_acc, (n, ext), "bar_coeff_acc", check_finite=True)
        bar_f = np.empty((n, n), dtype=np.float64)
        rc = self.congruence_rev_fn(
            n, ext, _dptr(c_c), _dptr(f_c), _dptr(bm_c), _dptr(bar_f), _dptr(bar_coeff_acc)
        )
        if rc != 0:
            raise RuntimeError(f"gqc_f12_ext_fock_congruence_rev failed with status {rc}")
        return bar_f

    def j_rev(
        self,
        raw_tile: np.ndarray,
        bar_g: np.ndarray,
        *,
        rho_tile: np.ndarray | None = None,
        bar_raw_tile: np.ndarray | None = None,
    ) -> np.ndarray:
        if not isinstance(raw_tile, np.ndarray) or raw_tile.ndim != 3:
            raise ValueError("raw_tile must be a 3D float64 ndarray of shape (qt, n, n)")
        qt, n, n2 = raw_tile.shape
        if n != n2:
            raise ValueError(f"raw_tile must be square on AO axes, got {raw_tile.shape}")
        qt = _check_c_int(qt, "qt", low=1)
        n = _check_c_int(n, "n", low=1)
        raw_c = _as_checked_f64_in(raw_tile, (qt, n, n), "raw_tile")
        bg_c = _as_checked_f64_in(bar_g, (n, n), "bar_g")
        rho_c = _as_checked_f64_in(rho_tile, (qt,), "rho_tile") if rho_tile is not None else None
        if bar_raw_tile is not None:
            _check_f64_out(bar_raw_tile, (qt, n, n), "bar_raw_tile", check_finite=False)
        bar_rho = np.empty(qt, dtype=np.float64)
        rc = self.j_rev_fn(
            qt,
            n,
            _dptr(raw_c),
            _dptr(rho_c),
            _dptr(bg_c),
            _dptr(bar_rho),
            _dptr(bar_raw_tile),
        )
        if rc != 0:
            raise RuntimeError(f"gqc_f12_ext_fock_j_rev failed with status {rc}")
        return bar_rho

    def rev(
        self,
        whitener: np.ndarray,
        n: int,
        occ_a: np.ndarray | None,
        occ_b: np.ndarray | None,
        R_a: np.ndarray | None,
        R_b: np.ndarray | None,
        D_a: np.ndarray | None,
        D_b: np.ndarray | None,
        bar_rho: np.ndarray,
        k_seed_a: np.ndarray | None,
        k_seed_b: np.ndarray | None,
        bar_occ_a_acc: np.ndarray | None,
        bar_occ_b_acc: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
        if not isinstance(whitener, np.ndarray) or whitener.ndim != 2 or whitener.shape[0] != whitener.shape[1]:
            raise ValueError("whitener must be a square 2D float64 ndarray")
        q = _check_c_int(whitener.shape[0], "q", low=1)
        n = _check_c_int(n, "n", low=1)
        w_c = _as_checked_f64_in(whitener, (q, q), "whitener")
        br_c = _as_checked_f64_in(bar_rho, (q,), "bar_rho")

        if occ_a is not None:
            if not isinstance(occ_a, np.ndarray) or occ_a.ndim != 2:
                raise ValueError("occ_a must be a 2D float64 ndarray")
            oa = _check_c_int(occ_a.shape[1], "oa", low=0)
        else:
            oa = 0
        if occ_b is not None:
            if not isinstance(occ_b, np.ndarray) or occ_b.ndim != 2:
                raise ValueError("occ_b must be a 2D float64 ndarray")
            ob = _check_c_int(occ_b.shape[1], "ob", low=0)
        else:
            ob = 0

        oa_c = _as_checked_f64_in(occ_a, (n, oa), "occ_a") if oa > 0 else None
        ob_c = _as_checked_f64_in(occ_b, (n, ob), "occ_b") if ob > 0 else None
        if oa > 0:
            if R_a is None or D_a is None:
                raise ValueError("R_a and D_a are required when oa > 0")
            ra_c = _as_checked_f64_in(R_a, (q, oa, n), "R_a")
            da_c = _as_checked_f64_in(D_a, (q, oa, n), "D_a")
            ka_c = _as_checked_f64_in(k_seed_a, (n, n), "k_seed_a") if k_seed_a is not None else None
            if bar_occ_a_acc is not None:
                _check_f64_out(bar_occ_a_acc, (n, oa), "bar_occ_a_acc", check_finite=True)
        else:
            ra_c = da_c = ka_c = None
        if ob > 0:
            if R_b is None or D_b is None:
                raise ValueError("R_b and D_b are required when ob > 0")
            rb_c = _as_checked_f64_in(R_b, (q, ob, n), "R_b")
            db_c = _as_checked_f64_in(D_b, (q, ob, n), "D_b")
            kb_c = _as_checked_f64_in(k_seed_b, (n, n), "k_seed_b") if k_seed_b is not None else None
            if bar_occ_b_acc is not None:
                _check_f64_out(bar_occ_b_acc, (n, ob), "bar_occ_b_acc", check_finite=True)
        else:
            rb_c = db_c = kb_c = None

        bar_R_a = np.empty((q, oa, n), dtype=np.float64) if oa > 0 else None
        bar_R_b = np.empty((q, ob, n), dtype=np.float64) if ob > 0 else None
        metric_bar = np.empty((q, q), dtype=np.float64)
        rc = self.rev_fn(
            q,
            n,
            oa,
            ob,
            _dptr(w_c),
            _dptr(oa_c),
            _dptr(ob_c),
            _dptr(ra_c),
            _dptr(rb_c),
            _dptr(da_c),
            _dptr(db_c),
            _dptr(br_c),
            _dptr(ka_c),
            _dptr(kb_c),
            _dptr(bar_R_a),
            _dptr(bar_R_b),
            _dptr(bar_occ_a_acc if oa > 0 else None),
            _dptr(bar_occ_b_acc if ob > 0 else None),
            _dptr(metric_bar),
        )
        if rc != 0:
            raise RuntimeError(f"gqc_f12_ext_fock_rev failed with status {rc}")
        return bar_R_a, bar_R_b, metric_bar

    def half_rev(
        self,
        raw_tile: np.ndarray,
        occ_a: np.ndarray | None,
        occ_b: np.ndarray | None,
        bar_R_a: np.ndarray | None,
        bar_R_b: np.ndarray | None,
        bar_raw_tile: np.ndarray,
        bar_occ_a_acc: np.ndarray | None,
        bar_occ_b_acc: np.ndarray | None,
        *,
        rho_tile: np.ndarray | None = None,
        bar_g: np.ndarray | None = None,
        add_to_bar_raw: bool = False,
    ) -> None:
        if not isinstance(raw_tile, np.ndarray) or raw_tile.ndim != 3:
            raise ValueError("raw_tile must be a 3D float64 ndarray of shape (qt, n, n)")
        qt, n, n2 = raw_tile.shape
        if n != n2:
            raise ValueError(f"raw_tile must be square on AO axes, got {raw_tile.shape}")
        qt = _check_c_int(qt, "qt", low=1)
        n = _check_c_int(n, "n", low=1)
        raw_c = _as_checked_f64_in(raw_tile, (qt, n, n), "raw_tile")
        _check_f64_out(
            bar_raw_tile, (qt, n, n), "bar_raw_tile", check_finite=bool(add_to_bar_raw)
        )
        if occ_a is not None:
            if not isinstance(occ_a, np.ndarray) or occ_a.ndim != 2:
                raise ValueError("occ_a must be a 2D float64 ndarray")
            oa = _check_c_int(occ_a.shape[1], "oa", low=0)
        else:
            oa = 0
        if occ_b is not None:
            if not isinstance(occ_b, np.ndarray) or occ_b.ndim != 2:
                raise ValueError("occ_b must be a 2D float64 ndarray")
            ob = _check_c_int(occ_b.shape[1], "ob", low=0)
        else:
            ob = 0

        oa_c = _as_checked_f64_in(occ_a, (n, oa), "occ_a") if oa > 0 else None
        ob_c = _as_checked_f64_in(occ_b, (n, ob), "occ_b") if ob > 0 else None
        if oa > 0:
            if bar_R_a is None:
                raise ValueError("bar_R_a is required when oa > 0")
            bra_c = _as_checked_f64_in(bar_R_a, (qt, oa, n), "bar_R_a")
            if bar_occ_a_acc is not None:
                _check_f64_out(bar_occ_a_acc, (n, oa), "bar_occ_a_acc", check_finite=True)
        else:
            bra_c = None
        if ob > 0:
            if bar_R_b is None:
                raise ValueError("bar_R_b is required when ob > 0")
            brb_c = _as_checked_f64_in(bar_R_b, (qt, ob, n), "bar_R_b")
            if bar_occ_b_acc is not None:
                _check_f64_out(bar_occ_b_acc, (n, ob), "bar_occ_b_acc", check_finite=True)
        else:
            brb_c = None
        rho_c = _as_checked_f64_in(rho_tile, (qt,), "rho_tile") if rho_tile is not None else None
        bg_c = _as_checked_f64_in(bar_g, (n, n), "bar_g") if bar_g is not None else None
        rc = self.half_rev_fn(
            qt,
            n,
            oa,
            ob,
            _dptr(raw_c),
            _dptr(rho_c),
            _dptr(bg_c),
            _dptr(oa_c),
            _dptr(ob_c),
            _dptr(bra_c),
            _dptr(brb_c),
            _dptr(bar_raw_tile),
            int(bool(add_to_bar_raw)),
            _dptr(bar_occ_a_acc if oa > 0 else None),
            _dptr(bar_occ_b_acc if ob > 0 else None),
        )
        if rc != 0:
            raise RuntimeError(f"gqc_f12_ext_fock_half_rev failed with status {rc}")


def bind_ext_fock_native(
    library: Path | str | ct.CDLL | None,
    *,
    host_reference: bool,
) -> ExtFockNative | None:
    """Bind paired native ext-fock symbols from a shared library, or return None if absent."""
    if library is None:
        return None
    cdll, path_str = _load_cdll(library)
    suffix = "_host_reference" if host_reference else "_cuda"
    names = (
        "gqc_f12_ext_fock_half_fwd",
        "gqc_f12_ext_fock_fwd",
        "gqc_f12_ext_fock_j_fwd",
        "gqc_f12_ext_fock_congruence_rev",
        "gqc_f12_ext_fock_j_rev",
        "gqc_f12_ext_fock_rev",
        "gqc_f12_ext_fock_half_rev",
    )
    funcs = [getattr(cdll, name + suffix, None) for name in names]
    if any(fn is None for fn in funcs):
        return None

    half_fwd_fn, fwd_fn, j_fwd_fn, cong_rev_fn, j_rev_fn, rev_fn, half_rev_fn = funcs
    half_fwd_fn.argtypes = [ct.c_int] * 3 + [_D_PTR] * 3
    fwd_fn.argtypes = [ct.c_int] * 5 + [_D_PTR] * 10
    j_fwd_fn.argtypes = [ct.c_int] * 2 + [_D_PTR] * 3
    cong_rev_fn.argtypes = [ct.c_int] * 2 + [_D_PTR] * 5
    j_rev_fn.argtypes = [ct.c_int] * 2 + [_D_PTR] * 5
    rev_fn.argtypes = [ct.c_int] * 4 + [_D_PTR] * 15
    half_rev_fn.argtypes = [ct.c_int] * 4 + [_D_PTR] * 8 + [ct.c_int] + [_D_PTR] * 2
    for fn in funcs:
        fn.restype = ct.c_int

    return ExtFockNative(
        half_fwd_fn=half_fwd_fn,
        fwd_fn=fwd_fn,
        j_fwd_fn=j_fwd_fn,
        congruence_rev_fn=cong_rev_fn,
        j_rev_fn=j_rev_fn,
        rev_fn=rev_fn,
        half_rev_fn=half_rev_fn,
        host_reference=host_reference,
        library_path=path_str,
    )


def resolve_ext_fock_native(
    *,
    cuda: bool,
    library: Path | str | ct.CDLL | None = None,
    native_library: Path | str | ct.CDLL | None = None,
    host_reference: bool | None = None,
) -> ExtFockNative | None:
    """Resolve the native extended-Fock symbol bundle, honoring strict AOT env enforcement."""
    require = _env_truthy("GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK")
    disable = _env_truthy("GENERATIVEQC_F12_DISABLE_AOT_EXTFOCK")
    if require and disable:
        raise ValueError(
            "Conflicting environment flags: GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK and "
            "GENERATIVEQC_F12_DISABLE_AOT_EXTFOCK cannot both be enabled"
        )
    if disable:
        return None

    use_host = (host_reference is True) or (host_reference is None and not cuda)
    if native_library is not None:
        bound = bind_ext_fock_native(native_library, host_reference=use_host)
    else:
        # The separate CABS response DSO may also contain the fused Fock ABI.
        if library is not None:
            try:
                bound = bind_ext_fock_native(library, host_reference=use_host)
            except OSError:
                bound = None
            if bound is not None:
                return bound
        env_keys = (
            ("GENERATIVEQC_F12_EXT_FOCK_LIBRARY", "GENERATIVEQC_F12_HOST_SINK")
            if use_host
            else (
                "GENERATIVEQC_F12_EXT_FOCK_LIBRARY",
                "GENERATIVEQC_F12_FOCAL_CUDA_LIBRARY",
                "GENERATIVEQC_F12_SHELL_LIBRARY",
            )
        )
        bound = resolve_shell_binding(
            None, lambda lib: bind_ext_fock_native(lib, host_reference=use_host),
            host_reference=use_host, env_keys=env_keys, mapped_paths=tuple,
        )

    if bound is None and require:
        mode = "host_reference" if use_host else "cuda"
        raise RuntimeError(
            f"GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK=1 requires native ext-fock {mode} "
            "symbols, but no compatible shared library was found"
        )
    return bound
