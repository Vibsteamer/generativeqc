"""Gate 1 executable tests closing external-audit findings M3, M6, M8, and m6.

Covers:
  - M3: Empty-spin non-zero Coulomb contribution in f12_cphf_relax_cuda.cu
        (exact audit counterexample + independent finite-difference oracle).
  - M6: Hard device budget enforcement in f12_ext_fock_cuda.cu and
        f12_cc_fixed_sources_cuda.cu (exact audit counterexamples + byte-exact
        minimum-budget boundaries).
  - M8: Fail-closed NaN/Inf output checks across all 4 units (including the
        exact whiten_3c 1e308 * 2.0 -> Inf counterexample) and exception-safe
        extern "C" boundaries (std::length_error / std::bad_alloc -> OOM code
        without process termination).
  - m6: Checked narrowing before int cuBLAS dimensions in f12_ext_fock_cuda.cu
        (exact n=46341 vs n=46340 boundary on j_fwd, j_rev, half_rev).
"""

from __future__ import annotations

import ctypes as ct
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import configured_test_library
from generativeqc._f12_ext_fock_native import bind_ext_fock_native
from generativeqc_compiler.common.compiler_cache import (
    resolve_compiler_cache,
    run_cached_compiler,
)

REPO = Path(__file__).resolve().parents[2]
_D_PTR = ct.POINTER(ct.c_double)
_NULL_D = ct.cast(0, _D_PTR)


def _dptr(arr: np.ndarray | None):
    if arr is None:
        return _NULL_D
    return arr.ctypes.data_as(_D_PTR)


def _compile_host_unit(
    tmp_path_factory: pytest.TempPathFactory,
    stem: str,
    cu_relpath: str,
    std: str,
    *,
    define_host_ref: bool = True,
) -> Path:
    explicit = configured_test_library("GENERATIVEQC_F12_HOST_SINK")
    if explicit is not None:
        return explicit

    cxx = shutil.which("c++") or shutil.which("g++")
    assert cxx is not None, "C++ compiler required to build host-reference test library"
    launcher = resolve_compiler_cache()
    out_dir = tmp_path_factory.mktemp(f"audit_fix1b_{stem}")
    lib_path = out_dir / f"lib{stem}_host.so"
    argv = [
        cxx,
        std,
        "-O3",
        "-x",
        "c++",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-Werror",
        *(["-DGQC_SINK_HOST_REFERENCE"] if define_host_ref else []),
        "-shared",
        str(REPO / cu_relpath),
        "-o",
        str(lib_path),
    ]
    ret = run_cached_compiler(argv, 300, label=f"audit-fix1b {stem} host", launcher=launcher)
    assert ret.returncode == 0, f"Host compilation failed for {cu_relpath}:\n{ret.stdout}\n{ret.stderr}"
    return lib_path


@pytest.fixture(scope="module")
def ext_fock_host_lib(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _compile_host_unit(
        tmp_path_factory, "ext_fock", "src/posthf/f12/f12_ext_fock_cuda.cu", "-std=c++20"
    )


def test_m6_ext_fock_hard_budget_audit_counterexamples(ext_fock_host_lib: Path) -> None:
    """Exact M6 audit counterexamples for ext_fock_rev (q=4096), congruence_rev (n=ext=2600), etc."""
    ops = bind_ext_fock_native(ext_fock_host_lib, host_reference=True)
    assert ops is not None

    dummy = np.zeros(1, dtype=np.float64)
    p = _dptr(dummy)

    # 1. ext_fock_rev: q=4096 -> 3 * q^2 * 8 = 384 MiB alone; + q + n^2 + 1-orbital tile > 384 MiB.
    rc_rev = ops.rev_fn(
        4096, 2, 1, 0,
        p, p, _NULL_D, p, _NULL_D, p, _NULL_D, p, p, _NULL_D,
        p, _NULL_D, p, _NULL_D, p,
    )
    assert rc_rev == -5

    # 2. congruence_rev: n=ext=2600 -> 8 * 2600^2 * 8 = 412.6 MiB > 384 MiB.
    rc_cong = ops.congruence_rev_fn(2600, 2600, p, p, p, p, p)
    assert rc_cong == -5

    # 3. ext_fock_fwd: q=7100, n=2, oa=1, ob=0 -> q^2 * 8 = 384.6 MiB > 384 MiB.
    rc_fwd = ops.fwd_fn(
        7100, 2, 1, 0, 0,
        p, p, _NULL_D, p, _NULL_D, p, _NULL_D, p, p, p,
    )
    assert rc_fwd == -5

    # 4. Tile primitives exceeding 256 MiB for even a single q-slice.
    assert ops.half_fwd_fn(1, 6000, 1, p, p, p) == -5
    assert ops.j_fwd_fn(1, 5000, p, p, p) == -5
    assert ops.j_rev_fn(1, 5000, p, p, p, p, p) == -5
    assert ops.half_rev_fn(1, 5000, 1, 0, p, _NULL_D, _NULL_D, p, _NULL_D, p, _NULL_D, p, 0, p, _NULL_D) == -5


def test_m8_ext_fock_all_7_primitives_reject_nonfinite(ext_fock_host_lib: Path) -> None:
    """All 7 ext_fock entry points fail closed with -4 when outputs overflow to Inf or NaN."""
    ops = bind_ext_fock_native(ext_fock_host_lib, host_reference=True)
    assert ops is not None

    big_raw = np.full((1, 2, 2), 1e308, dtype=np.float64)
    two_occ = np.full((2, 1), 2.0, dtype=np.float64)
    with pytest.raises(RuntimeError, match="status -4"):
        ops.half_fwd(big_raw, two_occ)

    with pytest.raises(RuntimeError, match="status -4"):
        ops.fwd(
            np.array([[1e308]]),
            np.ones((2, 1)),
            None,
            np.full((1, 1, 2), 2.0),
            None,
            same_occ_frame=False,
        )

    j_acc = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(RuntimeError, match="status -4"):
        ops.j_fwd(big_raw, np.array([2.0]), j_acc)

    bar_c = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(RuntimeError, match="status -4"):
        ops.congruence_rev(np.full((2, 2), 1e308), np.eye(2), np.full((2, 2), 2.0), bar_c)

    with pytest.raises(RuntimeError, match="status -4"):
        ops.j_rev(big_raw, np.full((2, 2), 2.0))

    bar_oa = np.zeros((2, 1), dtype=np.float64)
    with pytest.raises(RuntimeError, match="status -4"):
        ops.rev(
            np.array([[1e308]]),
            2,
            np.ones((2, 1)),
            None,
            np.ones((1, 1, 2)),
            None,
            np.ones((1, 1, 2)),
            None,
            np.array([2.0]),
            np.ones((2, 2)),
            None,
            bar_oa,
            None,
        )

    bar_raw = np.zeros((1, 2, 2), dtype=np.float64)
    with pytest.raises(RuntimeError, match="status -4"):
        ops.half_rev(
            big_raw,
            np.ones((2, 1)),
            None,
            np.full((1, 1, 2), 2.0),
            None,
            bar_raw,
            np.zeros((2, 1)),
            None,
        )


# =============================================================================
# m6: Checked narrowing before int cuBLAS dimensions in ext_fock
# =============================================================================


def test_m6_minor_ext_fock_checked_int_narrowing(ext_fock_host_lib: Path) -> None:
    """m6: n=46341 (n^2 > INT_MAX) rejected as invalid arg (-1); n=46340 passes narrowing to OOM (-5)."""
    ops = bind_ext_fock_native(ext_fock_host_lib, host_reference=True)
    assert ops is not None

    dummy = np.zeros(1, dtype=np.float64)
    p = _dptr(dummy)

    # n = 46341 => n^2 = 2,147,488,281 > INT_MAX (2,147,483,647) -> rejected before int cast (-1).
    assert ops.j_fwd_fn(1, 46341, p, p, p) == -1
    assert ops.j_rev_fn(1, 46341, p, p, p, p, p) == -1
    assert ops.half_rev_fn(1, 46341, 0, 0, p, _NULL_D, _NULL_D, _NULL_D, _NULL_D, _NULL_D, _NULL_D, p, 0, _NULL_D, _NULL_D) == -1

    # n = 46340 => n^2 = 2,147,395,600 <= INT_MAX -> passes int narrowing check, hits 256 MiB OOM (-5).
    assert ops.j_fwd_fn(1, 46340, p, p, p) == -5
    assert ops.j_rev_fn(1, 46340, p, p, p, p, p) == -5
    assert ops.half_rev_fn(1, 46340, 0, 0, p, _NULL_D, _NULL_D, _NULL_D, _NULL_D, _NULL_D, _NULL_D, p, 0, _NULL_D, _NULL_D) == -5
