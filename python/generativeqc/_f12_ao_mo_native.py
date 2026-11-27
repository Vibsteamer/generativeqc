"""Dynamic-dimension AOT C++/cuBLAS bindings for F12 AO->MO forward/reverse projections."""

from __future__ import annotations

import ctypes as ct
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    "gqc_f12_ao_mo_project_fwd",
    "gqc_f12_ao_mo_project_rev",
    "gqc_f12_ao_mo_bilin_fwd",
    "gqc_f12_ao_mo_bilin_rev",
    "gqc_f12_coulomb_fit_fwd",
    "gqc_f12_coulomb_fit_rev",
)

_CABS_SINGLES_EXPORTS = (
    "gqc_f12_cabs_fock_fwd",
    "gqc_f12_cabs_fock_rev",
    "gqc_f12_cabs_frame_kernel_cotangent",
    "gqc_f12_cabs_whiten_3c",
    "gqc_f12_cabs_cphf_action",
    "gqc_f12_cabs_nuclear_weights",
)

_OPS_CACHE: dict[tuple[int, bool], AoMoNativeOps | None] = {}
_CABS_OPS_CACHE: dict[tuple[int, bool], CabsSinglesNativeOps | None] = {}


def _dptr(arr: np.ndarray | None):
    if arr is None:
        return ct.cast(0, _D_PTR)
    return arr.ctypes.data_as(_D_PTR)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def aot_aomo_disabled() -> bool:
    return _env_truthy("GENERATIVEQC_F12_DISABLE_AOT_AOMO")


def aot_aomo_required() -> bool:
    return _env_truthy("GENERATIVEQC_F12_REQUIRE_AOT_AOMO")


def _check_aomo_env_flags() -> tuple[bool, bool]:
    require = aot_aomo_required()
    disable = aot_aomo_disabled()
    if require and disable:
        raise ValueError(
            "Conflicting environment flags: GENERATIVEQC_F12_REQUIRE_AOT_AOMO and "
            "GENERATIVEQC_F12_DISABLE_AOT_AOMO cannot both be enabled"
        )
    return require, disable


def _registered_shell_libraries() -> list[str]:
    try:
        from ._f12_spin_fock_1e_native import _REGISTERED_LIBRARIES

        return list(_REGISTERED_LIBRARIES)
    except Exception:
        return []


def _check_rc(rc: int, label: str) -> None:
    if rc != 0:
        detail = _STATUS_MESSAGES.get(rc, f"status {rc}")
        msg = f"native AOT F12 {label} failed: {detail} (rc={rc})"
        if rc in (-1, -2):
            raise ValueError(msg)
        raise RuntimeError(msg)


@dataclass(frozen=True)
class AoMoNativeOps:
    """Bound AOT native entry points for one backend (host_reference or cuda)."""

    host_reference: bool
    library_name: str
    project_fwd: Callable
    project_rev: Callable
    bilin_fwd: Callable
    bilin_rev: Callable
    coulomb_fit_fwd: Callable
    coulomb_fit_rev: Callable


def bind_ao_mo_ops(
    library: Path | str | ct.CDLL | None,
    *,
    host_reference: bool,
) -> AoMoNativeOps | None:
    """Bind all 6 AOT AO->MO symbols from ``library`` for the selected backend."""
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

    proj_fwd, proj_rev, bilin_fwd, bilin_rev, fit_fwd, fit_rev = funcs

    proj_fwd.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        _D_PTR,
        _D_PTR,
        _D_PTR,
    ]
    proj_fwd.restype = ct.c_int

    proj_rev.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.c_double,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
    ]
    proj_rev.restype = ct.c_int

    bilin_fwd.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_int,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
    ]
    bilin_fwd.restype = ct.c_int

    bilin_rev.argtypes = [
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
    ]
    bilin_rev.restype = ct.c_int

    fit_fwd.argtypes = [
        ct.c_int,
        ct.c_int,
        _D_PTR,
        _D_PTR,
        _D_PTR,
    ]
    fit_fwd.restype = ct.c_int

    fit_rev.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_double,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
        _D_PTR,
    ]
    fit_rev.restype = ct.c_int

    ops = AoMoNativeOps(
        host_reference=bool(host_reference),
        library_name=lib_name,
        project_fwd=proj_fwd,
        project_rev=proj_rev,
        bilin_fwd=bilin_fwd,
        bilin_rev=bilin_rev,
        coulomb_fit_fwd=fit_fwd,
        coulomb_fit_rev=fit_rev,
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
    ao: Mapping[str, object] | None = None,
    explicit_host_reference: bool | None = None,
) -> bool:
    """Infer whether ``run`` targets host CPU reference (True) or CUDA (False)."""
    if explicit_host_reference is not None:
        return bool(explicit_host_reference)
    if isinstance(ao, Mapping):
        loader = ao.get("__ao_loader__")
        owner = getattr(loader, "__self__", None)
        owner_hr = getattr(owner, "host_reference", None)
        if isinstance(owner_hr, bool):
            return owner_hr
    cuda_flag = _detect_cuda_from_callable(run, set())
    if cuda_flag is not None:
        return not cuda_flag
    return True


def resolve_ao_mo_ops(
    run: Callable | None,
    ao: Mapping[str, object] | None = None,
    *,
    shell_library: Path | str | ct.CDLL | None = None,
    host_reference: bool | None = None,
) -> AoMoNativeOps | None:
    """Resolve matching AOT AO->MO native operations without cross-backend fallback."""
    require, disable = _check_aomo_env_flags()
    if disable:
        return None
    is_host = detect_host_reference(run, ao, host_reference)
    selected = shell_library if shell_library is not None else getattr(run, "shell_library", None)
    if selected is None and ao is not None:
        selected = ao.get("__shell_library__")
        if selected is None:
            owner = getattr(ao.get("__ao_loader__"), "__self__", None)
            selected = getattr(owner, "library", None)
    env_keys = ("GENERATIVEQC_F12_AO_MO_LIBRARY", "GENERATIVEQC_F12_SHELL_LIBRARY")
    if is_host:
        env_keys += ("GENERATIVEQC_F12_HOST_SINK",)
    ops = resolve_shell_binding(
        selected, lambda lib: bind_ao_mo_ops(lib, host_reference=is_host),
        host_reference=is_host, env_keys=env_keys, mapped_paths=_loaded_sink_paths,
        registered=_registered_shell_libraries(),
    )
    if ops is None and require:
        backend = "host_reference" if is_host else "cuda"
        raise RuntimeError(
            f"GENERATIVEQC_F12_REQUIRE_AOT_AOMO=1 but no library exporting AOT AO->MO {backend} symbols was found"
        )
    return ops


def _as_f64_contiguous(arr: np.ndarray, name: str) -> np.ndarray:
    del name
    # Avoid allocating a full-size boolean array for multi-GiB 3-center panels;
    # the native C++/CUDA entry points validate finiteness in-place with O(1) memory.
    return np.ascontiguousarray(arr, dtype=np.float64)


def run_ao_mo_project_fwd(
    ops: AoMoNativeOps,
    nc: int,
    ext: int,
    left_dim: int,
    aux: int,
    coefficients: np.ndarray,
    ao_mat: np.ndarray,
    *,
    q_chunk_hint: int = 0,
) -> np.ndarray:
    """Execute AOT forward 3-center projection M[q, p, r] = C_left^T @ AO[q] @ C."""
    if nc <= 0 or ext <= 0 or ext > nc or left_dim < 0 or left_dim > ext or aux <= 0:
        raise ValueError(
            f"Invalid AO->MO forward dimensions: nc={nc}, ext={ext}, left_dim={left_dim}, aux={aux}"
        )
    _check_c_int(nc, "nc", low=1)
    _check_c_int(ext, "ext", low=1)
    _check_c_int(left_dim, "left_dim", low=0)
    _check_c_int(aux, "aux", low=1)
    _check_c_int(q_chunk_hint, "q_chunk_hint", low=0)
    if left_dim == 0:
        return np.empty((aux, 0, ext), dtype=np.float64)
    c_arr = _as_f64_contiguous(coefficients, "coefficients")
    ao_arr = _as_f64_contiguous(ao_mat, "ao_mat")
    if c_arr.shape != (nc, ext):
        raise ValueError(f"Expected coefficients shape {(nc, ext)}, got {c_arr.shape}")
    if ao_arr.shape != (aux, nc, nc):
        raise ValueError(f"Expected ao_mat shape {(aux, nc, nc)}, got {ao_arr.shape}")
    out = np.empty((aux, left_dim, ext), dtype=np.float64)
    rc = ops.project_fwd(
        int(nc),
        int(ext),
        int(left_dim),
        int(aux),
        int(q_chunk_hint),
        _dptr(c_arr),
        _dptr(ao_arr),
        _dptr(out),
    )
    _check_rc(rc, "project_fwd")
    return out


def run_ao_mo_project_rev(
    ops: AoMoNativeOps,
    nc: int,
    ext: int,
    left_dim: int,
    aux: int,
    coefficients: np.ndarray,
    ao_mat: np.ndarray | None,
    seed: np.ndarray,
    *,
    bar_coefficients: np.ndarray | None = None,
    bar_ao: np.ndarray | None = None,
    seed_scale: float = 1.0,
    three_center_bar: np.ndarray | None = None,
    inputs_mode: Literal["both", "coefficients", "ao"] = "both",
    q_chunk_hint: int = 0,
) -> np.ndarray | None:
    """Execute AOT reverse 3-center pullback matching project_ao_operator_pullback_tiled."""
    mode_map = {"both": 0, "coefficients": 1, "ao": 2}
    if inputs_mode not in mode_map:
        raise ValueError(f"Invalid inputs_mode {inputs_mode!r}")
    mode = mode_map[inputs_mode]
    if nc <= 0 or ext <= 0 or ext > nc or left_dim < 0 or left_dim > ext or aux <= 0:
        raise ValueError(
            f"Invalid AO->MO reverse dimensions: nc={nc}, ext={ext}, left_dim={left_dim}, aux={aux}"
        )
    if not np.isfinite(seed_scale):
        raise ValueError("seed_scale must be finite")

    _check_c_int(nc, "nc", low=1)
    _check_c_int(ext, "ext", low=1)
    _check_c_int(left_dim, "left_dim", low=0)
    _check_c_int(aux, "aux", low=1)
    _check_c_int(q_chunk_hint, "q_chunk_hint", low=0)

    c_arr = _as_f64_contiguous(coefficients, "coefficients")
    if c_arr.shape != (nc, ext):
        raise ValueError(f"Expected coefficients shape {(nc, ext)}, got {c_arr.shape}")

    seed_arr = _as_f64_contiguous(seed, "seed")
    if seed_arr.ndim != 3 or seed_arr.shape[0] != aux or seed_arr.shape[2] != ext:
        raise ValueError(
            f"Expected seed shape ({aux}, <= {left_dim}, {ext}), got {seed_arr.shape}"
        )
    seed_rows = int(seed_arr.shape[1])
    if seed_rows > left_dim:
        raise ValueError(f"seed_rows {seed_rows} exceeds left_dim {left_dim}")

    tc_arr: np.ndarray | None = None
    tc_rows = 0
    if three_center_bar is not None:
        tc_arr = _as_f64_contiguous(three_center_bar, "three_center_bar")
        if tc_arr.ndim != 3 or tc_arr.shape[0] != aux or tc_arr.shape[2] != ext:
            raise ValueError(
                f"Expected three_center_bar shape ({aux}, <= {left_dim}, {ext}), got {tc_arr.shape}"
            )
        tc_rows = int(tc_arr.shape[1])
        if tc_rows > left_dim:
            raise ValueError(f"tc_rows {tc_rows} exceeds left_dim {left_dim}")

    want_coeffs = mode in (0, 1) and bar_coefficients is not None
    want_ao = mode in (0, 2)
    fresh_bar_ao = want_ao and bar_ao is None

    ao_arr: np.ndarray | None = None
    if want_coeffs:
        if ao_mat is None:
            raise ValueError("ao_mat is required when computing bar_coefficients")
        ao_arr = _as_f64_contiguous(ao_mat, "ao_mat")
        if ao_arr.shape != (aux, nc, nc):
            raise ValueError(f"Expected ao_mat shape {(aux, nc, nc)}, got {ao_arr.shape}")

    check_disjoint_outputs(
        {"bar_coefficients": bar_coefficients if want_coeffs else None,
         "bar_ao": bar_ao if want_ao else None},
        {"coefficients": coefficients, "ao_mat": ao_mat if want_coeffs else None,
         "seed": seed, "three_center_bar": three_center_bar},
    )

    bar_c_target: np.ndarray | None = None
    bar_c_needs_copyback = False
    if want_coeffs:
        assert bar_coefficients is not None
        if bar_coefficients.shape != (nc, ext):
            raise ValueError(
                f"Expected bar_coefficients shape {(nc, ext)}, got {bar_coefficients.shape}"
            )
        if not bar_coefficients.flags.writeable:
            raise ValueError("bar_coefficients output buffer must be writeable")
        if bar_coefficients.dtype == np.float64 and bar_coefficients.flags.c_contiguous:
            bar_c_target = bar_coefficients
        else:
            bar_c_target = np.ascontiguousarray(bar_coefficients, dtype=np.float64)
            bar_c_needs_copyback = True

    bar_ao_target: np.ndarray | None = None
    bar_ao_needs_copyback = False
    if want_ao:
        if fresh_bar_ao:
            bar_ao = np.zeros((aux, nc, nc), dtype=np.float64)
        assert bar_ao is not None
        if bar_ao.shape != (aux, nc, nc):
            raise ValueError(f"Expected bar_ao shape {(aux, nc, nc)}, got {bar_ao.shape}")
        if not bar_ao.flags.writeable:
            raise ValueError("bar_ao output buffer must be writeable")
        if bar_ao.dtype == np.float64 and bar_ao.flags.c_contiguous:
            bar_ao_target = bar_ao
        else:
            bar_ao_target = np.ascontiguousarray(bar_ao, dtype=np.float64)
            bar_ao_needs_copyback = True

    if max(seed_rows, tc_rows) == 0 or (not want_coeffs and not want_ao):
        return bar_ao

    rc = ops.project_rev(
        int(nc),
        int(ext),
        int(left_dim),
        int(aux),
        seed_rows,
        tc_rows,
        mode,
        0 if fresh_bar_ao else 1,
        int(q_chunk_hint),
        float(seed_scale),
        _dptr(c_arr),
        _dptr(ao_arr),
        _dptr(seed_arr if seed_rows > 0 else None),
        _dptr(tc_arr if tc_rows > 0 else None),
        _dptr(bar_c_target),
        _dptr(bar_ao_target),
    )
    _check_rc(rc, "project_rev")

    if bar_c_needs_copyback:
        assert bar_coefficients is not None and bar_c_target is not None
        bar_coefficients[...] = bar_c_target
    if bar_ao_needs_copyback:
        assert bar_ao is not None and bar_ao_target is not None
        bar_ao[...] = bar_ao_target
    return bar_ao


def run_ao_mo_bilin_fwd(
    ops: AoMoNativeOps,
    a_mat: np.ndarray,
    c_left: np.ndarray,
    c_right: np.ndarray,
) -> np.ndarray:
    """Compute C_left^T @ A @ C_right via AOT kernel."""
    a_arr = _as_f64_contiguous(a_mat, "A")
    cl_arr = _as_f64_contiguous(c_left, "C_left")
    cr_arr = _as_f64_contiguous(c_right, "C_right")
    if a_arr.ndim != 2 or a_arr.shape[0] != a_arr.shape[1] or a_arr.shape[0] <= 0:
        raise ValueError(f"Expected nonempty square A, got {a_arr.shape}")
    nc = _check_c_int(a_arr.shape[0], "nc", low=1)
    if cl_arr.ndim != 2 or cl_arr.shape[0] != nc:
        raise ValueError(f"Expected C_left shape ({nc}, *), got {cl_arr.shape}")
    if cr_arr.ndim != 2 or cr_arr.shape[0] != nc:
        raise ValueError(f"Expected C_right shape ({nc}, *), got {cr_arr.shape}")
    n_left = _check_c_int(cl_arr.shape[1], "n_left", low=0)
    n_right = _check_c_int(cr_arr.shape[1], "n_right", low=0)
    out = np.empty((n_left, n_right), dtype=np.float64)
    rc = ops.bilin_fwd(
        nc, n_left, n_right, _dptr(a_arr), _dptr(cl_arr), _dptr(cr_arr), _dptr(out)
    )
    _check_rc(rc, "bilin_fwd")
    return out


def run_ao_mo_bilin_rev(
    ops: AoMoNativeOps,
    a_mat: np.ndarray,
    c_left: np.ndarray,
    c_right: np.ndarray,
    bar_out: np.ndarray,
    *,
    bar_a: np.ndarray | None = None,
    bar_c_left: np.ndarray | None = None,
    bar_c_right: np.ndarray | None = None,
) -> None:
    """Accumulate adjoints of C_left^T @ A @ C_right into bar_a, bar_c_left, bar_c_right."""
    a_arr = _as_f64_contiguous(a_mat, "A")
    cl_arr = _as_f64_contiguous(c_left, "C_left")
    cr_arr = _as_f64_contiguous(c_right, "C_right")
    bo_arr = _as_f64_contiguous(bar_out, "bar_out")
    if a_arr.ndim != 2 or a_arr.shape[0] != a_arr.shape[1] or a_arr.shape[0] <= 0:
        raise ValueError(f"Expected nonempty square A, got {a_arr.shape}")
    nc = _check_c_int(a_arr.shape[0], "nc", low=1)
    if cl_arr.ndim != 2 or cl_arr.shape[0] != nc:
        raise ValueError(f"Expected C_left shape ({nc}, *), got {cl_arr.shape}")
    if cr_arr.ndim != 2 or cr_arr.shape[0] != nc:
        raise ValueError(f"Expected C_right shape ({nc}, *), got {cr_arr.shape}")
    n_left = _check_c_int(cl_arr.shape[1], "n_left", low=0)
    n_right = _check_c_int(cr_arr.shape[1], "n_right", low=0)
    if bo_arr.shape != (n_left, n_right):
        raise ValueError(
            f"Expected bar_out shape {(n_left, n_right)}, got {bo_arr.shape}"
        )

    def _validate_target(
        buf: np.ndarray | None, expected: tuple[int, int], label: str
    ) -> None:
        if buf is None:
            return
        if not isinstance(buf, np.ndarray):
            raise ValueError(f"Expected {label} to be a numpy ndarray")
        if buf.shape != expected:
            raise ValueError(f"Expected {label} shape {expected}, got {buf.shape}")
        if not buf.flags.writeable:
            raise ValueError(f"Output target {label} must be writeable")

    def _prep_target(
        buf: np.ndarray | None, expected: tuple[int, int], label: str
    ) -> tuple[np.ndarray | None, bool]:
        _validate_target(buf, expected, label)
        if buf is None:
            return None, False
        if buf.dtype == np.float64 and buf.flags.c_contiguous:
            return buf, False
        return np.ascontiguousarray(buf, dtype=np.float64), True

    # Validate ALL targets unconditionally — including BOTH left and right shapes
    # and writeability even when bar_c_right is the same object as bar_c_left.
    _validate_target(bar_a, (nc, nc), "bar_a")
    _validate_target(bar_c_left, (nc, n_left), "bar_c_left")
    _validate_target(bar_c_right, (nc, n_right), "bar_c_right")

    check_disjoint_outputs(
        {"bar_a": bar_a, "bar_c_left": bar_c_left,
         "bar_c_right": bar_c_right if bar_c_right is not bar_c_left else None},
        {"a_mat": a_mat, "c_left": c_left, "c_right": c_right, "bar_out": bar_out},
    )

    ba_t, ba_cb = _prep_target(bar_a, (nc, nc), "bar_a")
    bcl_t, bcl_cb = _prep_target(bar_c_left, (nc, n_left), "bar_c_left")
    if bar_c_right is bar_c_left and bar_c_right is not None:
        bcr_t, bcr_cb = bcl_t, False
    else:
        bcr_t, bcr_cb = _prep_target(bar_c_right, (nc, n_right), "bar_c_right")

    rc = ops.bilin_rev(
        nc,
        n_left,
        n_right,
        _dptr(a_arr),
        _dptr(cl_arr),
        _dptr(cr_arr),
        _dptr(bo_arr),
        _dptr(ba_t),
        _dptr(bcl_t),
        _dptr(bcr_t),
    )
    _check_rc(rc, "bilin_rev")
    if ba_cb:
        assert bar_a is not None and ba_t is not None
        bar_a[...] = ba_t
    if bcl_cb:
        assert bar_c_left is not None and bcl_t is not None
        bar_c_left[...] = bcl_t
    if bcr_cb:
        assert bar_c_right is not None and bcr_t is not None
        bar_c_right[...] = bcr_t


def run_coulomb_fit_fwd(
    ops: AoMoNativeOps,
    whitener: np.ndarray,
    raw: np.ndarray,
) -> np.ndarray:
    """Compute fit = W^T @ (W @ raw) for full-rank Coulomb metric solve."""
    w_arr = _as_f64_contiguous(whitener, "metric_whitener")
    raw_arr = _as_f64_contiguous(raw, "three_center")
    if w_arr.ndim != 2 or w_arr.shape[0] != w_arr.shape[1] or w_arr.shape[0] <= 0:
        raise ValueError(f"Expected nonempty square whitener, got {w_arr.shape}")
    aux = _check_c_int(w_arr.shape[0], "aux", low=1)
    if raw_arr.ndim != 2 or raw_arr.shape[0] != aux:
        raise ValueError(f"Expected raw shape ({aux}, pairs), got {raw_arr.shape}")
    pairs = _check_c_int(raw_arr.shape[1], "pairs", low=0)
    out = np.empty((aux, pairs), dtype=np.float64)
    if pairs == 0:
        return out
    rc = ops.coulomb_fit_fwd(aux, pairs, _dptr(w_arr), _dptr(raw_arr), _dptr(out))
    _check_rc(rc, "coulomb_fit_fwd")
    return out


def run_coulomb_fit_rev(
    ops: AoMoNativeOps,
    whitener: np.ndarray,
    raw: np.ndarray,
    fit_bar: np.ndarray,
    metric_bar: np.ndarray,
    *,
    metric_scale: float = 1.0,
) -> np.ndarray:
    """Compute three_center_bar and accumulate metric_scale * metric_bar into metric_bar."""
    w_arr = _as_f64_contiguous(whitener, "metric_whitener")
    raw_arr = _as_f64_contiguous(raw, "three_center")
    fb_arr = _as_f64_contiguous(fit_bar, "fit_bar")
    aux = int(w_arr.shape[0])
    if w_arr.shape != (aux, aux):
        raise ValueError(f"Expected square whitener, got {w_arr.shape}")
    if raw_arr.ndim != 2 or raw_arr.shape[0] != aux or fb_arr.shape != raw_arr.shape:
        raise ValueError(
            f"Expected raw and fit_bar shape ({aux}, pairs), got {raw_arr.shape} and {fb_arr.shape}"
        )
    if metric_bar.shape != (aux, aux):
        raise ValueError(f"Expected metric_bar shape {(aux, aux)}, got {metric_bar.shape}")
    if not metric_bar.flags.writeable:
        raise ValueError("metric_bar output buffer must be writeable")
    check_disjoint_outputs(
        {"metric_bar": metric_bar},
        {"whitener": whitener, "raw": raw, "fit_bar": fit_bar},
    )
    if not np.isfinite(metric_scale):
        raise ValueError("metric_scale must be finite")
    _check_c_int(aux, "aux", low=1)
    pairs = _check_c_int(raw_arr.shape[1], "pairs", low=0)
    tc_bar = np.empty((aux, pairs), dtype=np.float64)
    if pairs == 0:
        return tc_bar

    if metric_bar.dtype == np.float64 and metric_bar.flags.c_contiguous:
        mb_t = metric_bar
        mb_cb = False
    else:
        mb_t = np.ascontiguousarray(metric_bar, dtype=np.float64)
        mb_cb = True

    rc = ops.coulomb_fit_rev(
        aux,
        pairs,
        float(metric_scale),
        _dptr(w_arr),
        _dptr(raw_arr),
        _dptr(fb_arr),
        _dptr(tc_bar),
        _dptr(mb_t),
    )
    _check_rc(rc, "coulomb_fit_rev")
    if mb_cb:
        metric_bar[...] = mb_t
    return tc_bar


# ============================================================================
# CABS Singles E+F dynamic-dimension AOT native bindings
# ============================================================================


@dataclass(frozen=True)
class CabsSinglesNativeOps:
    """Bound AOT native entry points for CABS Singles E+F (host_reference or cuda)."""

    host_reference: bool
    library_name: str
    ao_mo: AoMoNativeOps
    cabs_fock_fwd: Callable
    cabs_fock_rev: Callable
    frame_kernel_cotangent: Callable
    whiten_3c: Callable
    cphf_action: Callable
    nuclear_weights: Callable


def bind_cabs_singles_ops(
    library: Path | str | ct.CDLL | None,
    *,
    host_reference: bool,
) -> CabsSinglesNativeOps | None:
    """Bind all AOT CABS Singles + AO->MO symbols from ``library`` for one backend."""
    if library is None:
        return None
    ao_mo = bind_ao_mo_ops(library, host_reference=host_reference)
    if ao_mo is None:
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
    if key in _CABS_OPS_CACHE:
        return _CABS_OPS_CACHE[key]

    suffix = "_host_reference" if host_reference else "_cuda"
    funcs = []
    for base in _CABS_SINGLES_EXPORTS:
        fn = getattr(cdll, base + suffix, None)
        if fn is None:
            _CABS_OPS_CACHE[key] = None
            return None
        funcs.append(fn)

    fock_fwd, fock_rev, fk_cot, whiten_3c, cphf_act, nuc_w = funcs

    fock_fwd.argtypes = [ct.c_int] * 7 + [_D_PTR] * 12
    fock_fwd.restype = ct.c_int

    fock_rev.argtypes = [ct.c_int] * 8 + [_D_PTR] * 20
    fock_rev.restype = ct.c_int

    fk_cot.argtypes = [ct.c_int] * 2 + [_D_PTR] * 7
    fk_cot.restype = ct.c_int

    whiten_3c.argtypes = [ct.c_int, ct.c_int, _D_PTR, _D_PTR, _D_PTR]
    whiten_3c.restype = ct.c_int

    cphf_act.argtypes = [ct.c_int] * 6 + [_D_PTR] * 8
    cphf_act.restype = ct.c_int

    nuc_w.argtypes = [ct.c_int] * 7 + [_D_PTR] * 13
    nuc_w.restype = ct.c_int

    ops = CabsSinglesNativeOps(
        host_reference=bool(host_reference),
        library_name=lib_name,
        ao_mo=ao_mo,
        cabs_fock_fwd=fock_fwd,
        cabs_fock_rev=fock_rev,
        frame_kernel_cotangent=fk_cot,
        whiten_3c=whiten_3c,
        cphf_action=cphf_act,
        nuclear_weights=nuc_w,
    )
    _CABS_OPS_CACHE[key] = ops
    return ops


def resolve_cabs_singles_ops(
    run: Callable | None,
    *,
    shell_library: Path | str | ct.CDLL | None = None,
    host_reference: bool | None = None,
) -> CabsSinglesNativeOps | None:
    """Resolve matching AOT CABS Singles native ops without cross-backend fallback."""
    require, disable = _check_aomo_env_flags()
    if disable:
        return None
    is_host = detect_host_reference(run, None, host_reference)
    selected = shell_library if shell_library is not None else getattr(run, "shell_library", None)
    env_keys = ("GENERATIVEQC_F12_AO_MO_LIBRARY", "GENERATIVEQC_F12_SHELL_LIBRARY")
    if is_host:
        env_keys += ("GENERATIVEQC_F12_HOST_SINK",)
    ops = resolve_shell_binding(
        selected, lambda lib: bind_cabs_singles_ops(lib, host_reference=is_host),
        host_reference=is_host, env_keys=env_keys, mapped_paths=_loaded_sink_paths,
        registered=_registered_shell_libraries(),
    )
    if ops is None and require:
        backend = "host_reference" if is_host else "cuda"
        raise RuntimeError(
            f"GENERATIVEQC_F12_REQUIRE_AOT_AOMO=1 but no library exporting AOT CABS Singles {backend} symbols was found"
        )
    return ops


def _cabs_matrix_shape(arr: np.ndarray, name: str) -> tuple[int, int]:
    shape = np.shape(arr)
    if len(shape) != 2:
        raise ValueError(f"{name} must be a matrix, got shape {shape}")
    return (
        _check_c_int(shape[0], name + " rows", low=1),
        _check_c_int(shape[1], name + " columns"),
    )


def _cabs_input(arr: np.ndarray | None, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if arr is None or np.shape(arr) != shape:
        raise ValueError(f"Expected {name} shape {shape}, got {np.shape(arr)}")
    return _as_f64_contiguous(arr, name)


def run_cabs_fock_fwd(
    ops: CabsSinglesNativeOps,
    hcore: np.ndarray,
    raw: np.ndarray,
    whitener: np.ndarray,
    occ_a: np.ndarray | None,
    occ_b: np.ndarray | None,
    *,
    occ_rows: int,
    same_occ: bool,
    save_intermediates: bool,
    q_chunk_hint: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Build physical AO DF Fock matrices (alpha, beta) and optional pullback caches."""
    n, _ = _cabs_matrix_shape(hcore, "hcore")
    q, _ = _cabs_matrix_shape(whitener, "whitener")
    occ_rows = _check_c_int(occ_rows, "occ_rows", low=1)
    q_chunk_hint = _check_c_int(q_chunk_hint, "q_chunk_hint")
    if occ_rows > n:
        raise ValueError(f"Invalid occ_rows={occ_rows} for n={n}")
    h_arr = _cabs_input(hcore, "hcore", (n, n))
    raw_arr = _cabs_input(raw, "raw", (q, n, n))
    w_arr = _cabs_input(whitener, "whitener", (q, q))

    def occupied_frame(arr: np.ndarray | None, name: str) -> tuple[int, np.ndarray | None]:
        if arr is None:
            return 0, None
        _, occ = _cabs_matrix_shape(arr, name)
        if occ > n:
            raise ValueError(f"{name} has more occupied columns than AO rows")
        return occ, _cabs_input(arr, name, (occ_rows, occ))

    oa, occ_a_arr = occupied_frame(occ_a, "occ_a")
    ob, occ_b_arr = (oa, None) if same_occ else occupied_frame(occ_b, "occ_b")

    fock_a = np.empty((n, n), dtype=np.float64)
    fock_b = np.empty((n, n), dtype=np.float64)
    rho_out = np.empty((q,), dtype=np.float64) if save_intermediates else None
    raw_a_out = (
        np.empty((q, oa, n), dtype=np.float64)
        if save_intermediates and oa > 0
        else None
    )
    fit_a_out = (
        np.empty((q, oa, n), dtype=np.float64)
        if save_intermediates and oa > 0
        else None
    )
    raw_b_out = (
        np.empty((q, ob, n), dtype=np.float64)
        if save_intermediates and (not same_occ) and ob > 0
        else None
    )
    fit_b_out = (
        np.empty((q, ob, n), dtype=np.float64)
        if save_intermediates and (not same_occ) and ob > 0
        else None
    )

    rc = ops.cabs_fock_fwd(
        n,
        q,
        oa,
        ob,
        int(occ_rows),
        1 if same_occ else 0,
        int(q_chunk_hint),
        _dptr(h_arr),
        _dptr(raw_arr),
        _dptr(w_arr),
        _dptr(occ_a_arr),
        _dptr(occ_b_arr),
        _dptr(fock_a),
        _dptr(fock_b),
        _dptr(rho_out),
        _dptr(raw_a_out),
        _dptr(fit_a_out),
        _dptr(raw_b_out),
        _dptr(fit_b_out),
    )
    _check_rc(rc, "cabs_fock_fwd")
    if same_occ and save_intermediates:
        raw_b_out = raw_a_out
        fit_b_out = fit_a_out
    return fock_a, fock_b, rho_out, raw_a_out, fit_a_out, raw_b_out, fit_b_out


def run_cabs_fock_rev(
    ops: CabsSinglesNativeOps,
    fock_ao_a: np.ndarray,
    fock_ao_b: np.ndarray,
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    seed_a: np.ndarray,
    seed_b: np.ndarray,
    raw: np.ndarray,
    whitener: np.ndarray,
    rho: np.ndarray,
    raw_a: np.ndarray | None,
    fit_a: np.ndarray | None,
    raw_b: np.ndarray | None,
    fit_b: np.ndarray | None,
    occ_a: np.ndarray | None,
    occ_b: np.ndarray | None,
    *,
    oa: int,
    ob: int,
    occ_rows: int,
    same_spin: bool,
    q_chunk_hint: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull extended Fock cotangents back to frames, hcore, three_center, and metric."""
    n, ext = _cabs_matrix_shape(frame_a, "frame_a")
    q, _ = _cabs_matrix_shape(whitener, "whitener")
    oa = _check_c_int(oa, "oa")
    ob = _check_c_int(ob, "ob")
    occ_rows = _check_c_int(occ_rows, "occ_rows", low=1)
    q_chunk_hint = _check_c_int(q_chunk_hint, "q_chunk_hint")
    if not 0 < ext <= n or max(oa, ob) > ext or occ_rows > n or (same_spin and oa != ob):
        raise ValueError("Inconsistent CABS Fock reverse orbital dimensions")
    fa = _cabs_input(fock_ao_a, "fock_ao_a", (n, n))
    fb = fa if same_spin else _cabs_input(fock_ao_b, "fock_ao_b", (n, n))
    ca = _as_f64_contiguous(frame_a, "frame_a")
    cb = ca if same_spin else _cabs_input(frame_b, "frame_b", (n, ext))
    sa = _cabs_input(seed_a, "seed_a", (ext, ext))
    sb = sa if same_spin else _cabs_input(seed_b, "seed_b", (ext, ext))
    raw_arr = _cabs_input(raw, "raw", (q, n, n))
    w_arr = _cabs_input(whitener, "whitener", (q, q))
    rho_arr = _cabs_input(rho, "rho", (q,))

    ra_arr = da_arr = oa_arr = None
    if oa:
        ra_arr = _cabs_input(raw_a, "raw_a", (q, oa, n))
        da_arr = _cabs_input(fit_a, "fit_a", (q, oa, n))
        oa_arr = _cabs_input(occ_a, "occ_a", (occ_rows, oa))
    rb_arr = db_arr = ob_arr = None
    if ob and not same_spin:
        rb_arr = _cabs_input(raw_b, "raw_b", (q, ob, n))
        db_arr = _cabs_input(fit_b, "fit_b", (q, ob, n))
        ob_arr = _cabs_input(occ_b, "occ_b", (occ_rows, ob))

    ga_out = np.empty((n, ext), dtype=np.float64)
    gb_out = np.empty((n, ext), dtype=np.float64)
    core_bar = np.empty((n, n), dtype=np.float64)
    raw_bar = np.empty((q, n, n), dtype=np.float64)
    metric_bar = np.empty((q, q), dtype=np.float64)

    rc = ops.cabs_fock_rev(
        n,
        ext,
        q,
        int(oa),
        int(ob),
        int(occ_rows),
        1 if same_spin else 0,
        int(q_chunk_hint),
        _dptr(fa),
        _dptr(fb),
        _dptr(ca),
        _dptr(cb),
        _dptr(sa),
        _dptr(sb),
        _dptr(raw_arr),
        _dptr(w_arr),
        _dptr(rho_arr),
        _dptr(ra_arr),
        _dptr(da_arr),
        _dptr(rb_arr),
        _dptr(db_arr),
        _dptr(oa_arr),
        _dptr(ob_arr),
        _dptr(ga_out),
        _dptr(gb_out),
        _dptr(core_bar),
        _dptr(raw_bar),
        _dptr(metric_bar),
    )
    _check_rc(rc, "cabs_fock_rev")
    return ga_out, gb_out, core_bar, raw_bar, metric_bar


def run_cabs_frame_kernel_cotangent(
    ops: CabsSinglesNativeOps,
    overlap: np.ndarray,
    frame: np.ndarray,
    frame_bar: np.ndarray,
) -> dict[str, np.ndarray]:
    """Transfer an invariant CABS frame cotangent to kernel_bar + diagnostics."""
    atomic, rank = _cabs_matrix_shape(frame, "frame")
    if not 0 < rank <= atomic:
        raise ValueError("Inconsistent CABS frame rank")
    s_arr = _cabs_input(overlap, "overlap", (atomic, atomic))
    c_arr = _as_f64_contiguous(frame, "frame")
    g_arr = _cabs_input(frame_bar, "frame_bar", (atomic, rank))
    kernel_bar = np.empty((atomic, atomic), dtype=np.float64)
    dual_gram = np.empty((rank, rank), dtype=np.float64)
    gauge_skew = np.empty((rank, rank), dtype=np.float64)
    residual = np.empty((atomic, rank), dtype=np.float64)
    rc = ops.frame_kernel_cotangent(
        atomic,
        rank,
        _dptr(s_arr),
        _dptr(c_arr),
        _dptr(g_arr),
        _dptr(kernel_bar),
        _dptr(dual_gram),
        _dptr(gauge_skew),
        _dptr(residual),
    )
    _check_rc(rc, "frame_kernel_cotangent")
    return {
        "kernel_bar": kernel_bar,
        "dual_gram": dual_gram,
        "gauge_skew": gauge_skew,
        "reconstruction_residual": residual,
    }


def run_cabs_whiten_3c(
    ops: CabsSinglesNativeOps,
    whitener: np.ndarray,
    raw: np.ndarray,
) -> np.ndarray:
    """Compute df = whitener @ raw along axis 0 via AOT kernel."""
    aux, _ = _cabs_matrix_shape(whitener, "whitener")
    raw = np.asarray(raw)
    if raw.ndim < 2 or raw.shape[0] != aux:
        raise ValueError("Inconsistent whiten_3c shapes")
    pairs = _check_c_int(raw.size // aux, "pairs")
    w_arr = _cabs_input(whitener, "whitener", (aux, aux))
    raw_arr = _as_f64_contiguous(raw, "raw")
    out = np.empty_like(raw_arr)
    if pairs == 0:
        return out
    rc = ops.whiten_3c(aux, pairs, _dptr(w_arr), _dptr(raw_arr), _dptr(out))
    _check_rc(rc, "whiten_3c")
    return out


def run_cabs_cphf_action(
    ops: CabsSinglesNativeOps,
    oa: int,
    ob: int,
    obs: int,
    aux: int,
    *,
    is_rhf: bool,
    df_a: np.ndarray | None,
    df_b: np.ndarray | None,
    gap_a: np.ndarray | None,
    gap_b: np.ndarray | None,
    trial_a: np.ndarray | None,
    trial_b: np.ndarray | None,
    q_chunk_hint: int = 0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Apply the DF-RHF or DF-UHF occupied-virtual CPHF Hessian to trial vector(s)."""
    oa = _check_c_int(oa, "oa")
    ob = _check_c_int(ob, "ob")
    obs = _check_c_int(obs, "obs", low=1)
    aux = _check_c_int(aux, "aux", low=1)
    q_chunk_hint = _check_c_int(q_chunk_hint, "q_chunk_hint")
    if max(oa, ob) >= obs or oa + ob == 0 or (is_rhf and oa != ob):
        raise ValueError("Inconsistent CABS CPHF orbital dimensions")

    va = obs - oa
    vb = obs - ob
    dfa = ga = ta = out_a = None
    if oa:
        dfa = _cabs_input(df_a, "df_a", (aux, obs, obs))
        ga = _cabs_input(gap_a, "gap_a", (va, oa))
        ta = _cabs_input(trial_a, "trial_a", (va, oa))
        out_a = np.empty((va, oa), dtype=np.float64)
    dfb = gb = tb = out_b = None
    if ob and not is_rhf:
        dfb = _cabs_input(df_b, "df_b", (aux, obs, obs))
        gb = _cabs_input(gap_b, "gap_b", (vb, ob))
        tb = _cabs_input(trial_b, "trial_b", (vb, ob))
        out_b = np.empty((vb, ob), dtype=np.float64)

    rc = ops.cphf_action(
        int(oa),
        int(ob),
        int(obs),
        int(aux),
        1 if is_rhf else 0,
        int(q_chunk_hint),
        _dptr(dfa),
        _dptr(dfb),
        _dptr(ga),
        _dptr(gb),
        _dptr(ta),
        _dptr(tb),
        _dptr(out_a),
        _dptr(out_b),
    )
    _check_rc(rc, "cphf_action")
    return out_a, out_b


def run_cabs_nuclear_weights(
    ops: CabsSinglesNativeOps,
    oa: int,
    ob: int,
    obs: int,
    aux: int,
    nao: int,
    *,
    same_spin: bool,
    whitener: np.ndarray,
    c_a: np.ndarray,
    c_b: np.ndarray,
    df_a: np.ndarray,
    df_b: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    sm_a: np.ndarray,
    sm_b: np.ndarray,
    q_chunk_hint: int = 0,
) -> dict[str, np.ndarray]:
    """Compute symmetric DF-HF orbital-response AO/metric nuclear source weights."""
    oa = _check_c_int(oa, "oa")
    ob = _check_c_int(ob, "ob")
    obs = _check_c_int(obs, "obs", low=1)
    aux = _check_c_int(aux, "aux", low=1)
    nao = _check_c_int(nao, "nao", low=1)
    q_chunk_hint = _check_c_int(q_chunk_hint, "q_chunk_hint")
    if nao < obs or max(oa, ob) > obs or (same_spin and oa != ob):
        raise ValueError("Inconsistent CABS nuclear weights orbital dimensions")
    w_arr = _cabs_input(whitener, "whitener", (aux, aux))
    ca_arr = _cabs_input(c_a, "c_a", (nao, obs))
    cb_arr = ca_arr if same_spin else _cabs_input(c_b, "c_b", (nao, obs))
    dfa_arr = _cabs_input(df_a, "df_a", (aux, obs, obs))
    dfb_arr = dfa_arr if same_spin else _cabs_input(df_b, "df_b", (aux, obs, obs))
    pa_arr = _cabs_input(p_a, "p_a", (obs, obs))
    pb_arr = pa_arr if same_spin else _cabs_input(p_b, "p_b", (obs, obs))
    sma_arr = _cabs_input(sm_a, "sm_a", (obs, obs))
    smb_arr = sma_arr if same_spin else _cabs_input(sm_b, "sm_b", (obs, obs))

    bar_ds = np.empty((nao, nao), dtype=np.float64)
    bar_dh = np.empty((nao, nao), dtype=np.float64)
    bar_dr = np.empty((aux, nao, nao), dtype=np.float64)
    bar_dj = np.empty((aux, aux), dtype=np.float64)

    rc = ops.nuclear_weights(
        int(oa),
        int(ob),
        int(obs),
        int(aux),
        int(nao),
        1 if same_spin else 0,
        int(q_chunk_hint),
        _dptr(w_arr),
        _dptr(ca_arr),
        _dptr(cb_arr),
        _dptr(dfa_arr),
        _dptr(dfb_arr),
        _dptr(pa_arr),
        _dptr(pb_arr),
        _dptr(sma_arr),
        _dptr(smb_arr),
        _dptr(bar_ds),
        _dptr(bar_dh),
        _dptr(bar_dr),
        _dptr(bar_dj),
    )
    _check_rc(rc, "nuclear_weights")
    return {
        "overlap_ao": bar_ds,
        "hcore_ao": bar_dh,
        "ao_coulomb": bar_dr,
        "metric/coulomb": bar_dj,
    }
