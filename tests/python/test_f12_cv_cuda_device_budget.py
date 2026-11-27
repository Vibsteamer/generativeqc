"""Exercise the exported CUDA ΔCV memory budgets and fail-closed binding contract."""

from __future__ import annotations

import ctypes as ct
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_MIB = 1 << 20
_GIB = 1 << 30

# Tuple format: (symbol_key, int_dims, budget_bytes, ptr_count, expected_rc)
# With no visible CUDA device, rc == 3 means the budget was admitted and stream
# creation failed. rc == 4 means kErrOom was returned BEFORE CUDA init.
CASES = [
    # 1. Default budget (budget_bytes == 0 -> 1 / 2.5 / 2.8 GiB defaults)
    ("mp2_factors_fwd", (11584, 2, 1, 1, 1), 0, 5, 3),
    ("mp2_factors_fwd", (11585, 2, 1, 1, 1), 0, 5, 4),
    ("mp2_factors_rev", (8191, 2, 1, 1, 1), 0, 10, 3),
    ("mp2_factors_rev", (8192, 2, 1, 1, 1), 0, 10, 4),
    # The streamed back-projection can exceed its budget independently of the metric stage.
    ("mp2_factors_rev", (1, 8192, 1, 1, 1), 0, 10, 4),
    ("cphf_matvec", (1, 6986, 3493, 3493, 1, 1), 0, 8, 3),
    ("cphf_matvec", (1, 6987, 3493, 3493, 1, 1), 0, 8, 4),
    ("cphf_matvec", (1, 5315, 2657, 2657, 0, 1), 0, 8, 3),
    ("cphf_matvec", (1, 5316, 2658, 2658, 0, 1), 0, 8, 4),
    ("mp2_pairs", (1, 1, 2**24, 2, 2, 8, 8, 1, 1, 1), 0, 14, 4),
    ("mp2_pairs", (0, 1, 2**24, 2, 2, 8, 8, 1, 1, 1), 0, 14, 4),
    ("mp2_pairs", (1, 1, 1, 128, 128, 1024, 1024, 1, 1, 1), 0, 14, 4),
    ("df_jk_fock", (1, 6000, 1, 1, 1, 1), 0, 7, 4),
    ("jk_mo_fock_fwd", (20000, 2, 1, 1, 1, 1), 0, 9, 4),
    ("jk_fock_pullback", (11000, 2, 1, 1, 1, 1), 0, 14, 3),
    ("jk_fock_pullback", (12000, 2, 1, 1, 1, 1), 0, 14, 4),
    ("nuclear_source_weights", (11060, 2, 1, 1, 1, 1), 0, 13, 3),
    ("nuclear_source_weights", (11061, 2, 1, 1, 1, 1), 0, 13, 4),
    ("nuclear_source_weights", (1, 5200, 1, 1, 1, 1), 0, 13, 3),
    ("nuclear_source_weights", (1, 5200, 1, 1, 0, 1), 0, 13, 4),
    ("nuclear_source_weights", (1, 5100, 1, 1, 0, 1), 0, 13, 3),
    # 2. Explicit tight caller budgets reject workloads that default 1-2.8 GiB admits
    ("df_jk_fock", (1, 2000, 1, 1, 1, 1), 32 * _MIB, 7, 4),
    ("mp2_factors_fwd", (4000, 2, 1, 1, 1), 32 * _MIB, 5, 4),
    ("mp2_factors_rev", (4000, 2, 1, 1, 1), 32 * _MIB, 10, 4),
    ("mp2_pairs", (1, 1, 1, 16, 16, 256, 256, 1, 1, 1), 16 * _MIB, 14, 4),
    ("jk_mo_fock_fwd", (4000, 2, 1, 1, 1, 1), 32 * _MIB, 9, 4),
    ("jk_fock_pullback", (4000, 2, 1, 1, 1, 1), 32 * _MIB, 14, 4),
    ("cphf_matvec", (1, 2000, 1000, 1000, 1, 1), 16 * _MIB, 8, 4),
    ("nuclear_source_weights", (4000, 2, 1, 1, 1, 1), 32 * _MIB, 13, 4),
    # 3. Explicit large caller budgets (> 3 GiB) admit workloads that exceeded old 1 / 2.5 / 2.8 GiB caps
    ("df_jk_fock", (1, 6000, 1, 1, 1, 1), 6 * _GIB, 7, 3),
    ("mp2_factors_fwd", (12000, 2, 1, 1, 1), 6 * _GIB, 5, 3),
    ("mp2_factors_rev", (9000, 2, 1, 1, 1), 6 * _GIB, 10, 3),
    ("mp2_pairs", (1, 1, 1, 128, 128, 1024, 1024, 1, 1, 1), 6 * _GIB, 14, 3),
    ("jk_mo_fock_fwd", (20000, 2, 1, 1, 1, 1), 6 * _GIB, 9, 3),
    ("jk_fock_pullback", (12000, 2, 1, 1, 1, 1), 6 * _GIB, 14, 3),
    ("cphf_matvec", (1, 7500, 3750, 3750, 1, 1), 6 * _GIB, 8, 3),
    ("nuclear_source_weights", (12000, 2, 1, 1, 1, 1), 6 * _GIB, 13, 3),
]


def test_cv_workspace_is_checked_before_cuda_initialization() -> None:
    from conftest import configured_test_library
    from generativeqc import _native

    library = configured_test_library("GENERATIVEQC_F12_SHELL_LIBRARY")
    if library is None:
        library = _native.installed_f12_shell_library(host_reference=False)
    if library is None:
        pytest.skip("install or configure a CUDA through-i shell for device budget tests")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(library)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    codes = json.loads(result.stdout)
    for (name, dimensions, budget_bytes, _, expected), actual in zip(CASES, codes, strict=True):
        assert actual == expected, (name, dimensions, budget_bytes, actual, expected)


def test_delta_cv_cuda_binding_fails_closed_without_tensorir_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M1: CUDA mode must raise RuntimeError rather than silently falling back to TensorIR."""
    from conftest import configured_test_library
    from generativeqc._f12_cv_dfmp2_native import bind_delta_cv_native

    for key in (
        "GENERATIVEQC_F12_REQUIRE_AOT_CV",
        "GENERATIVEQC_F12_REQUIRE_AOT_DELTACV",
        "GENERATIVEQC_F12_DISABLE_AOT_CV",
        "GENERATIVEQC_F12_DISABLE_AOT_DELTACV",
    ):
        monkeypatch.delenv(key, raising=False)

    # Missing library in CUDA mode fails closed.
    with pytest.raises(RuntimeError, match="silent fallback to TensorIR is disabled"):
        bind_delta_cv_native(None, host_reference=False, budget_bytes=_GIB)

    # Disabling AOT in CUDA mode fails closed.
    with monkeypatch.context() as mp:
        mp.setenv("GENERATIVEQC_F12_DISABLE_AOT_CV", "1")
        with pytest.raises(RuntimeError, match="not permitted"):
            bind_delta_cv_native(None, host_reference=False, budget_bytes=_GIB)

    # Missing CUDA symbols in a host-only library fails closed in CUDA mode.
    host_lib = configured_test_library("GENERATIVEQC_F12_HOST_SINK")
    if host_lib is not None and host_lib.exists():
        with pytest.raises(RuntimeError, match="missing from"):
            bind_delta_cv_native(host_lib, host_reference=False, budget_bytes=_GIB)

    # CUDA bindings never fall back to the fixed native defaults behind budget_bytes=0.
    with pytest.raises(ValueError, match="explicit positive budget_bytes"):
        bind_delta_cv_native(None, host_reference=False)

    # Explicit CPU/host_reference mode still permits oracle opt-out when REQUIRE is unset.
    assert bind_delta_cv_native(None, host_reference=True) is None
    with monkeypatch.context() as mp:
        mp.setenv("GENERATIVEQC_F12_DISABLE_AOT_CV", "1")
        assert bind_delta_cv_native(None, host_reference=True) is None

    # Negative or boolean budget_bytes are rejected cleanly.
    with pytest.raises(ValueError):
        bind_delta_cv_native(None, host_reference=True, budget_bytes=-1)
    with pytest.raises(ValueError):
        bind_delta_cv_native(None, host_reference=True, budget_bytes=True)


if __name__ == "__main__":
    lib = ct.CDLL(sys.argv[1])
    results = []
    for name, dimensions, budget_bytes, count, _ in CASES:
        function = getattr(lib, f"gqc_f12_cv_{name}_cuda")
        function.argtypes = (
            [ct.c_int] * len(dimensions) + [ct.c_uint64] + [ct.POINTER(ct.c_double)] * count
        )
        function.restype = ct.c_int
        # The pair ABI zeros its two three-spin outputs even when admission fails.
        buffers = [(ct.c_double * 3)() for _ in range(count)]
        results.append(function(*dimensions, budget_bytes, *buffers))
    print(json.dumps(results))
