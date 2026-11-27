"""Dynamic-dimension AOT C++/cuBLAS bindings for F12 one-electron gradient and spin Fock."""

from __future__ import annotations

import ctypes as ct
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._f12_native_buffers import check_disjoint_outputs
from ._f12_shell_binding import _loaded_sink_paths, open_shell_library, resolve_shell_binding

_D_PTR = ct.POINTER(ct.c_double)
_C_INT_MAX = (1 << 31) - 1


def _check_c_int(value: int, name: str, *, low: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    ival = int(value)
    if ival < low or ival > _C_INT_MAX:
        raise ValueError(f"{name}={ival} exceeds C int range [{low}, {_C_INT_MAX}]")
    return ival

_STATUS_MESSAGES = {
    -1: "invalid dimensions or null required pointer",
    -2: "non-finite FP64 value detected",
    -3: "device workspace budget exceeded",
    -4: "CUDA runtime error",
    -5: "cuBLAS error",
}

_REQUIRED_EXPORTS = (
    "gqc_f12_one_electron_grad",
    "gqc_f12_spin_fock_fwd",
    "gqc_f12_spin_fock_rev",
)

_VALID_SEEDS = frozenset({"fock_alpha", "fock_beta", "exchange_alpha", "exchange_beta"})

_OPS_CACHE: dict[tuple[int, bool], SpinFock1eNativeOps | None] = {}
_REGISTERED_LIBRARIES: list[str] = []
_INPLACE_ALIAS_SAFE_CACHE: dict[int, bool] = {}


def _dptr(arr: np.ndarray | None):
    if arr is None:
        return ct.cast(0, _D_PTR)
    return arr.ctypes.data_as(_D_PTR)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def aot_spinfock_1e_disabled() -> bool:
    return _env_truthy("GENERATIVEQC_F12_DISABLE_AOT_SPINFOCK_1E")


def aot_spinfock_1e_required() -> bool:
    return _env_truthy("GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E")


def register_shell_library(path: Path | str | None) -> None:
    """Remember a loaded F12 shell/integral library path for subsequent AOT lookups."""
    if path is None:
        return
    resolved = str(Path(path).resolve())
    if resolved in _REGISTERED_LIBRARIES:
        _REGISTERED_LIBRARIES.remove(resolved)
    _REGISTERED_LIBRARIES.insert(0, resolved)


def _check_rc(rc: int, label: str) -> None:
    if rc != 0:
        detail = _STATUS_MESSAGES.get(rc, f"status {rc}")
        msg = f"native AOT F12 {label} failed: {detail} (rc={rc})"
        if rc in (-1, -2):
            raise ValueError(msg)
        raise RuntimeError(msg)


@dataclass(frozen=True)
class SpinFock1eNativeOps:
    """Bound AOT native entry points for one backend (host_reference or cuda)."""

    host_reference: bool
    library_name: str
    one_electron_grad: Callable
    spin_fock_fwd: Callable
    spin_fock_rev: Callable


def bind_spin_fock_1e_ops(
    library: Path | str | ct.CDLL | None,
    *,
    host_reference: bool,
) -> SpinFock1eNativeOps | None:
    """Bind all 3 AOT spin-Fock / 1e-grad symbols from ``library`` for one backend."""
    if library is None:
        return None
    if isinstance(library, ct.CDLL):
        cdll = library
        lib_name = getattr(library, "_name", "<cdll>")
    else:
        cdll = open_shell_library(library)
        if cdll is None:
            return None
        lib_name = str(Path(library).resolve())

    key = (int(cdll._handle), bool(host_reference))
    if key in _OPS_CACHE:
        return _OPS_CACHE[key]

    suffix = "_host_reference" if host_reference else "_cuda"
    funcs = []
    for base in _REQUIRED_EXPORTS:
        fn = getattr(cdll, base + suffix, None)
        if fn is None:
            _OPS_CACHE[key] = None
            return None
        funcs.append(fn)

    grad_fn, fwd_fn, rev_fn = funcs

    grad_fn.argtypes = [
        ct.c_int,
        ct.c_int,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
    ]
    grad_fn.restype = ct.c_int

    fwd_fn.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
    ]
    fwd_fn.restype = ct.c_int

    rev_fn.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
    ]
    rev_fn.restype = ct.c_int

    ops = SpinFock1eNativeOps(
        host_reference=bool(host_reference),
        library_name=lib_name,
        one_electron_grad=grad_fn,
        spin_fock_fwd=fwd_fn,
        spin_fock_rev=rev_fn,
    )
    _OPS_CACHE[key] = ops
    return ops


def _detect_cuda_from_callable(
    fn: object, visited: set[int], depth: int = 0
) -> bool | None:
    if fn is None or depth > 6 or id(fn) in visited:
        return None
    visited.add(id(fn))

    explicit_hr = getattr(fn, "host_reference", None)
    if isinstance(explicit_hr, bool):
        return not explicit_hr

    # focal_cuda_runner returns a closure carrying both .batch and .profile_snapshot.
    if hasattr(fn, "profile_snapshot") and hasattr(fn, "batch"):
        return True

    inner = getattr(fn, "func", None)
    if inner is not None:
        res = _detect_cuda_from_callable(inner, visited, depth + 1)
        if res is not None:
            return res

    code = getattr(fn, "__code__", None)
    closure = getattr(fn, "__closure__", None)
    if code is not None and closure is not None:
        callables: list[object] = []
        for name, cell in zip(code.co_freevars, closure):
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if name == "cuda" and isinstance(val, bool):
                return val
            if name == "expected_backend" and isinstance(val, str):
                if "cuda" in val:
                    return True
                if "numpy" in val or "cpu" in val:
                    return False
            if callable(val):
                if name in ("runner", "executor", "run"):
                    callables.insert(0, val)
                else:
                    callables.append(val)
        for sub in callables:
            res = _detect_cuda_from_callable(sub, visited, depth + 1)
            if res is not None:
                return res
    return None


def detect_host_reference(
    run: Callable | None,
    explicit_host_reference: bool | None = None,
) -> bool:
    """Infer whether ``run`` targets host CPU reference (True) or CUDA (False)."""
    if explicit_host_reference is not None:
        return bool(explicit_host_reference)
    cuda_flag = _detect_cuda_from_callable(run, set())
    if cuda_flag is not None:
        return not cuda_flag
    return True


def resolve_spin_fock_1e_ops(
    run: Callable | None = None,
    *,
    library: Path | str | ct.CDLL | None = None,
    host_reference: bool | None = None,
) -> SpinFock1eNativeOps | None:
    """Resolve matching AOT spin-Fock / 1e-grad native operations without cross-backend fallback."""
    required = aot_spinfock_1e_required()
    disabled = aot_spinfock_1e_disabled()
    if required and disabled:
        raise ValueError(
            "Conflicting environment flags: GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E and "
            "GENERATIVEQC_F12_DISABLE_AOT_SPINFOCK_1E cannot both be enabled"
        )
    if disabled:
        return None
    is_host = detect_host_reference(run, host_reference)

    selected = library if library is not None else getattr(run, "shell_library", None)
    env_keys = ("GENERATIVEQC_F12_SHELL_LIBRARY",)
    if is_host:
        env_keys += ("GENERATIVEQC_F12_HOST_SINK",)
    ops = resolve_shell_binding(
        selected, lambda lib: bind_spin_fock_1e_ops(lib, host_reference=is_host),
        host_reference=is_host, env_keys=env_keys, mapped_paths=_loaded_sink_paths,
        registered=_REGISTERED_LIBRARIES,
    )
    if ops is None and required:
        backend = "host_reference" if is_host else "cuda"
        raise RuntimeError(
            "GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E=1 but no library exporting "
            f"AOT spin-Fock / 1e-grad {backend} symbols was found"
        )
    return ops


def _as_f64_contiguous(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float64)


def _arrays_equal_fast(a: np.ndarray, b: np.ndarray) -> bool:
    if a is b:
        return True
    if a.shape != b.shape:
        return False
    if a.size == 0:
        return True
    if a.flat[0] != b.flat[0]:
        return False
    return bool(np.array_equal(a, b))


def _validate_space(space: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    oa, ob, orbitals, auxiliaries = space
    if (
        any(type(v) is not int for v in (oa, ob, orbitals, auxiliaries))
        or min(oa, ob) < 0
        or oa + ob < 1
        or max(oa, ob) >= orbitals
        or auxiliaries < 1
    ):
        raise ValueError("Need occupied electrons, nonempty virtual and fitting spaces")
    _check_c_int(oa, "oa", low=0)
    _check_c_int(ob, "ob", low=0)
    _check_c_int(orbitals, "orbitals", low=1)
    _check_c_int(auxiliaries, "auxiliaries", low=1)
    if orbitals * orbitals > _C_INT_MAX:
        raise ValueError(f"orbitals^2={orbitals * orbitals} exceeds C int limit {_C_INT_MAX}")
    return oa, ob, orbitals, auxiliaries


def run_one_electron_grad(
    ops: SpinFock1eNativeOps,
    d_overlap: np.ndarray,
    d_hcore: np.ndarray,
    overlap_weight: np.ndarray,
    hcore_weight: np.ndarray,
) -> np.ndarray:
    """Compute grad[a, x] = sum_{i,j} (d_overlap[a,x,i,j]*ov[i,j] + d_hcore[a,x,i,j]*hc[i,j])."""
    if d_overlap.ndim != 4 or d_hcore.shape != d_overlap.shape or d_overlap.shape[1] != 3:
        raise ValueError("One-electron Jacobians must have shape (natoms, 3, n, n)")
    natoms, _, n, n2 = d_overlap.shape
    if n != n2 or overlap_weight.shape != (n, n) or hcore_weight.shape != (n, n):
        raise ValueError("One-electron weights must match Jacobian matrix extent")
    natoms = _check_c_int(natoms, "natoms", low=1)
    n = _check_c_int(n, "n", low=1)
    if 3 * natoms > _C_INT_MAX or n * n > _C_INT_MAX:
        raise ValueError("One-electron gradient dimensions exceed C int bounds")

    d_ov_c = _as_f64_contiguous(d_overlap)
    d_hc_c = _as_f64_contiguous(d_hcore)
    ov_w_c = _as_f64_contiguous(overlap_weight)
    hc_w_c = _as_f64_contiguous(hcore_weight)
    grad_out = np.empty((natoms, 3), dtype=np.float64)

    rc = ops.one_electron_grad(
        int(natoms),
        int(n),
        _dptr(d_ov_c),
        _dptr(d_hc_c),
        _dptr(ov_w_c),
        _dptr(hc_w_c),
        _dptr(grad_out),
    )
    _check_rc(rc, "one_electron_grad")
    return grad_out


def run_spin_fock_fwd(
    ops: SpinFock1eNativeOps,
    space: tuple[int, int, int, int],
    sources: Mapping[str, np.ndarray],
    *,
    batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Execute AOT extended UHF/RHF DF spin Fock forward contraction."""
    oa, ob, n, q = _validate_space(space)
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("The auxiliary count and batch size must be positive")
    q_tile_hint = 0 if batch_size is None else _check_c_int(batch_size, "batch_size", low=1)

    raw_a_in = sources.get("raw_coulomb_alpha")
    raw_b_in = sources.get("raw_coulomb_beta")
    if raw_a_in is None or raw_b_in is None:
        raise ValueError("spin_fock requires raw_coulomb_alpha and raw_coulomb_beta")
    if raw_a_in.shape != (q, n, n) or raw_b_in.shape != (q, n, n):
        raise ValueError("Inconsistent raw_coulomb shape for spin_fock")

    fit_a_in = sources.get("fit_alpha") if oa > 0 else None
    fit_b_in = sources.get("fit_beta") if ob > 0 else None
    if oa > 0:
        if fit_a_in is None or fit_a_in.shape != (q, oa, n):
            raise ValueError("Inconsistent fit_alpha shape for spin_fock")
    if ob > 0:
        if fit_b_in is None or fit_b_in.shape != (q, ob, n):
            raise ValueError("Inconsistent fit_beta shape for spin_fock")

    hcore_a_in = sources.get("hcore_alpha")
    hcore_b_in = sources.get("hcore_beta")
    if hcore_a_in is not None and hcore_a_in.shape != (n, n):
        raise ValueError("Inconsistent hcore_alpha shape for spin_fock")
    if hcore_b_in is not None and hcore_b_in.shape != (n, n):
        raise ValueError("Inconsistent hcore_beta shape for spin_fock")

    share_beta = (
        oa == ob
        and _arrays_equal_fast(raw_a_in, raw_b_in)
        and (
            oa == 0
            or (
                fit_a_in is not None
                and fit_b_in is not None
                and _arrays_equal_fast(fit_a_in, fit_b_in)
            )
        )
    )

    hcore_a_c = _as_f64_contiguous(hcore_a_in) if hcore_a_in is not None else None
    hcore_b_c = _as_f64_contiguous(hcore_b_in) if hcore_b_in is not None else None
    raw_a_c = _as_f64_contiguous(raw_a_in)
    raw_b_c = None if share_beta else _as_f64_contiguous(raw_b_in)
    fit_a_c = _as_f64_contiguous(fit_a_in) if fit_a_in is not None else None
    fit_b_c = (
        None
        if share_beta or fit_b_in is None
        else _as_f64_contiguous(fit_b_in)
    )

    fock_a = np.empty((n, n), dtype=np.float64)
    fock_b = np.empty((n, n), dtype=np.float64)
    exch_a = np.empty((n, n), dtype=np.float64)
    exch_b = np.empty((n, n), dtype=np.float64)

    rc = ops.spin_fock_fwd(
        int(oa),
        int(ob),
        int(n),
        int(q),
        1 if hcore_a_c is not None else 0,
        1 if hcore_b_c is not None else 0,
        1 if share_beta else 0,
        q_tile_hint,
        _dptr(hcore_a_c),
        _dptr(hcore_b_c),
        _dptr(raw_a_c),
        _dptr(raw_b_c),
        _dptr(fit_a_c),
        _dptr(fit_b_c),
        _dptr(fock_a),
        _dptr(fock_b),
        _dptr(exch_a),
        _dptr(exch_b),
    )
    _check_rc(rc, "spin_fock_fwd")
    return {
        "fock_alpha": fock_a,
        "exchange_alpha": exch_a,
        "fock_beta": fock_b,
        "exchange_beta": exch_b,
    }


def _spin_fock_rev_alias_safe(ops: SpinFock1eNativeOps) -> bool:
    """Verify once per bound spin_fock_rev symbol that bar_raw_a == raw_a is alias-safe."""
    fn = ops.spin_fock_rev
    if isinstance(fn, ct._CFuncPtr):
        key = int(ct.cast(fn, ct.c_void_p).value or 0)
    else:
        key = id(fn)
    cached = _INPLACE_ALIAS_SAFE_CACHE.get(key)
    if cached is not None:
        return cached
    raw_ref = np.array(
        [[[1.2, -0.3], [-0.3, 0.7]], [[-0.5, 0.4], [0.4, 0.9]]], dtype=np.float64
    )
    fit_ref = np.array([[[0.6, -0.2]], [[0.3, 0.8]]], dtype=np.float64)
    bar_f = np.array([[0.5, -0.1], [0.2, 0.4]], dtype=np.float64)
    bar_k = np.array([[0.3, 0.7], [-0.4, 0.2]], dtype=np.float64)
    out_raw_ref = np.zeros_like(raw_ref)
    out_fit_ref = np.zeros_like(fit_ref)
    out_raw_in = raw_ref.copy()
    out_fit_in = np.zeros_like(fit_ref)
    try:
        rc1 = fn(
            1, 1, 2, 2, 1, 0, 1,
            _dptr(raw_ref), _dptr(None), _dptr(fit_ref), _dptr(None),
            _dptr(bar_f), _dptr(None), _dptr(bar_k), _dptr(None),
            _dptr(None), _dptr(None), _dptr(out_raw_ref), _dptr(None),
            _dptr(out_fit_ref), _dptr(None),
        )
        rc2 = fn(
            1, 1, 2, 2, 1, 0, 1,
            _dptr(out_raw_in), _dptr(None), _dptr(fit_ref), _dptr(None),
            _dptr(bar_f), _dptr(None), _dptr(bar_k), _dptr(None),
            _dptr(None), _dptr(None), _dptr(out_raw_in), _dptr(None),
            _dptr(out_fit_in), _dptr(None),
        )
        safe = bool(
            rc1 == 0
            and rc2 == 0
            and np.allclose(out_raw_in, out_raw_ref, atol=1e-13, rtol=0.0)
            and np.allclose(out_fit_in, out_fit_ref, atol=1e-13, rtol=0.0)
        )
    except Exception:
        safe = False
    _INPLACE_ALIAS_SAFE_CACHE[key] = safe
    return safe


def _prep_output_buffer(
    bars: dict[str, np.ndarray],
    key: str,
    expected_shape: tuple[int, ...],
    *,
    accumulate_out: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (c_buffer, writeback_target_if_copied) for one output array."""
    arr = bars.get(key)
    if arr is None:
        return None, None
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Output buffer {key} must be a numpy ndarray")
    if arr.shape != expected_shape:
        raise ValueError(f"Accumulate buffer {key} has wrong shape {arr.shape}")
    if not arr.flags.writeable:
        raise ValueError(f"Output buffer {key} is read-only and cannot be written")
    if arr.dtype == np.float64 and arr.flags["C_CONTIGUOUS"]:
        return arr, None
    buf = (
        np.ascontiguousarray(arr, dtype=np.float64)
        if accumulate_out
        else np.empty(expected_shape, dtype=np.float64)
    )
    return buf, arr


def run_spin_fock_rev(
    ops: SpinFock1eNativeOps,
    space: tuple[int, int, int, int],
    sources: Mapping[str, np.ndarray],
    seeds: Mapping[str, np.ndarray],
    *,
    batch_size: int | None = None,
    accumulate: dict[str, np.ndarray] | None = None,
    inplace_raw_coulomb: bool = False,
) -> dict[str, np.ndarray]:
    """Execute AOT extended UHF/RHF DF spin Fock transpose pullback."""
    oa, ob, n, q = _validate_space(space)
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("The auxiliary count and batch size must be positive")
    q_tile_hint = 0 if batch_size is None else _check_c_int(batch_size, "batch_size", low=1)

    unknown = set(seeds) - _VALID_SEEDS
    if unknown:
        raise ValueError(f"Unsupported spin_fock pullback seed keys: {sorted(unknown)}")

    raw_a_in = sources.get("raw_coulomb_alpha")
    raw_b_in = sources.get("raw_coulomb_beta")
    if raw_a_in is None or raw_b_in is None:
        raise ValueError("spin_fock_pullback requires raw_coulomb_alpha and raw_coulomb_beta")
    if raw_a_in.shape != (q, n, n) or raw_b_in.shape != (q, n, n):
        raise ValueError("Inconsistent raw_coulomb shape for spin_fock_pullback")

    fit_a_in = sources.get("fit_alpha") if oa > 0 else None
    fit_b_in = sources.get("fit_beta") if ob > 0 else None
    if oa > 0 and (fit_a_in is None or fit_a_in.shape != (q, oa, n)):
        raise ValueError("Inconsistent fit_alpha shape for spin_fock_pullback")
    if ob > 0 and (fit_b_in is None or fit_b_in.shape != (q, ob, n)):
        raise ValueError("Inconsistent fit_beta shape for spin_fock_pullback")

    for k, v in seeds.items():
        if v.shape != (n, n):
            raise ValueError(f"Seed {k} must have shape {(n, n)}")

    share_beta_raw = (
        accumulate is None
        and oa == ob
        and "raw_coulomb_alpha" in sources
        and "raw_coulomb_beta" in sources
        and _arrays_equal_fast(raw_a_in, raw_b_in)
        and (
            "fit_alpha" not in sources
            or (
                "fit_beta" in sources
                and _arrays_equal_fast(sources["fit_alpha"], sources["fit_beta"])
            )
        )
        and (
            "hcore_alpha" not in sources
            or (
                "hcore_beta" in sources
                and _arrays_equal_fast(sources["hcore_alpha"], sources["hcore_beta"])
            )
        )
        and all(
            (
                k.replace("_alpha", "_beta") in seeds
                and _arrays_equal_fast(seeds[k], seeds[k.replace("_alpha", "_beta")])
            )
            for k in seeds
            if k.endswith("_alpha")
        )
        and sum(1 for k in seeds if k.endswith("_alpha"))
        == sum(1 for k in seeds if k.endswith("_beta"))
    )

    accumulate_out = accumulate is not None
    inplace_raw = (
        inplace_raw_coulomb
        and share_beta_raw
        and accumulate is None
        and raw_a_in.flags.writeable
        and raw_a_in.flags["C_CONTIGUOUS"]
        and raw_a_in.dtype == np.float64
        and not any(
            arr is not None and np.shares_memory(raw_a_in, arr)
            for arr in (fit_a_in, fit_b_in, *seeds.values())
        )
        and _spin_fock_rev_alias_safe(ops)
    )
    if accumulate is None:
        bars = {
            key: (
                raw_a_in
                if (inplace_raw and key == "raw_coulomb_alpha")
                else np.zeros(value.shape, dtype=np.float64)
            )
            for key, value in sources.items()
            if not (share_beta_raw and key == "raw_coulomb_beta")
        }
    else:
        bars = accumulate
        check_disjoint_outputs(
            {"bar_" + key: bars.get(key) for key in (
                "hcore_alpha", "hcore_beta", "raw_coulomb_alpha", "raw_coulomb_beta",
                "fit_alpha", "fit_beta",
            )},
            {"raw_alpha": raw_a_in, "raw_beta": raw_b_in,
             "fit_alpha": fit_a_in, "fit_beta": fit_b_in, **seeds},
        )

    raw_a_c = _as_f64_contiguous(raw_a_in)
    raw_b_c = None if share_beta_raw else _as_f64_contiguous(raw_b_in)
    fit_a_c = _as_f64_contiguous(fit_a_in) if fit_a_in is not None else None
    fit_b_c = (
        None
        if share_beta_raw or fit_b_in is None
        else _as_f64_contiguous(fit_b_in)
    )

    bar_fa = _as_f64_contiguous(seeds["fock_alpha"]) if "fock_alpha" in seeds else None
    bar_fb = (
        None
        if share_beta_raw
        else (
            _as_f64_contiguous(seeds["fock_beta"]) if "fock_beta" in seeds else None
        )
    )
    bar_ka = (
        _as_f64_contiguous(seeds["exchange_alpha"])
        if "exchange_alpha" in seeds
        else None
    )
    bar_kb = (
        None
        if share_beta_raw
        else (
            _as_f64_contiguous(seeds["exchange_beta"])
            if "exchange_beta" in seeds
            else None
        )
    )

    writebacks: list[tuple[np.ndarray, np.ndarray]] = []
    out_hc_a, wb = _prep_output_buffer(
        bars, "hcore_alpha", (n, n), accumulate_out=accumulate_out
    )
    if wb is not None and out_hc_a is not None:
        writebacks.append((out_hc_a, wb))
    out_hc_b, wb = _prep_output_buffer(
        bars, "hcore_beta", (n, n), accumulate_out=accumulate_out
    )
    if wb is not None and out_hc_b is not None:
        writebacks.append((out_hc_b, wb))
    out_raw_a, wb = _prep_output_buffer(
        bars, "raw_coulomb_alpha", (q, n, n), accumulate_out=accumulate_out
    )
    if wb is not None and out_raw_a is not None:
        writebacks.append((out_raw_a, wb))
    if share_beta_raw:
        out_raw_b = None
    else:
        out_raw_b, wb = _prep_output_buffer(
            bars, "raw_coulomb_beta", (q, n, n), accumulate_out=accumulate_out
        )
        if wb is not None and out_raw_b is not None:
            writebacks.append((out_raw_b, wb))
    if oa > 0:
        out_fit_a, wb = _prep_output_buffer(
            bars, "fit_alpha", (q, oa, n), accumulate_out=accumulate_out
        )
        if wb is not None and out_fit_a is not None:
            writebacks.append((out_fit_a, wb))
    else:
        out_fit_a = None
    if ob > 0:
        out_fit_b, wb = _prep_output_buffer(
            bars, "fit_beta", (q, ob, n), accumulate_out=accumulate_out
        )
        if wb is not None and out_fit_b is not None:
            writebacks.append((out_fit_b, wb))
    else:
        out_fit_b = None

    rc = ops.spin_fock_rev(
        int(oa),
        int(ob),
        int(n),
        int(q),
        1 if share_beta_raw else 0,
        1 if accumulate_out else 0,
        q_tile_hint,
        _dptr(raw_a_c),
        _dptr(raw_b_c),
        _dptr(fit_a_c),
        _dptr(fit_b_c),
        _dptr(bar_fa),
        _dptr(bar_fb),
        _dptr(bar_ka),
        _dptr(bar_kb),
        _dptr(out_hc_a),
        _dptr(out_hc_b),
        _dptr(out_raw_a),
        _dptr(out_raw_b),
        _dptr(out_fit_a),
        _dptr(out_fit_b),
    )
    _check_rc(rc, "spin_fock_rev")

    for buf, target in writebacks:
        target[...] = buf

    if share_beta_raw:
        bars["raw_coulomb_beta"] = bars["raw_coulomb_alpha"]
    return bars
