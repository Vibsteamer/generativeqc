"""Executable Gate 1 counterexample tests for Audit Fix-3 (Python FFI, cache, ownership, binding, projector)."""

from __future__ import annotations

import ctypes as ct
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from conftest import f12_host_shell

from generativeqc._f12_ao_mo_native import (
    bind_ao_mo_ops,
    run_ao_mo_bilin_rev,
    run_ao_mo_project_rev,
    run_coulomb_fit_rev,
)
from generativeqc._f12_cabs_fock import physical_spin_operators
from generativeqc._f12_ext_fock_native import bind_ext_fock_native, resolve_ext_fock_native
from generativeqc._f12_focal_integral_kernel import (
    LEGACY_SLATER_FIT_NORMALIZATION,
    LEGACY_SLATER_NORM_CODE,
    contract_geometry,
    slater_normalization_code,
)
from generativeqc._f12_focal_integral_sink import _bind_fallback_weighted
from generativeqc._f12_focal_projector import _ozaki_mm, _ozaki_mm_left_dd


@pytest.fixture(scope="module")
def host_sink(tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest) -> Path:
    if os.environ.get("GENERATIVEQC_F12_HOST_SINK") is not None:
        src = f12_host_shell()
        dst = tmp_path_factory.mktemp("audit_fix3_host") / "libaudit_fix3_host_fixture.so"
        shutil.copy2(src, dst)
        return dst
    root = Path(__file__).resolve().parents[2]
    units = (
        "src/posthf/f12/f12_ao_mo_project_cuda.cu",
        "src/posthf/f12/f12_ext_fock_cuda.cu",
    )
    output = tmp_path_factory.mktemp("audit_host") / "libaudit_host_fixture.so"
    compiler = request.getfixturevalue("required_native_cxx")
    return compiler.build_shared(
        [root / unit for unit in units],
        output,
        compile_args=("-x", "c++", "-std=c++20", "-O3", "-DGQC_SINK_HOST_REFERENCE"),
    )


@pytest.fixture(autouse=True)
def _isolate_host_sink_env(host_sink: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("GENERATIVEQC_F12_HOST_SINK"):
        monkeypatch.setenv("GENERATIVEQC_F12_HOST_SINK", str(host_sink))


def _cpu_runner():
    from generativeqc_compiler.tensor import execute

    def run(program, inputs):
        return execute(program, inputs, max_bytes=256 << 20)

    return run


def _make_symmetric_fock_problem(seed: int, *, n: int = 4, ext: int = 3, q: int = 5):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)
    raw = rng.normal(size=(q, n, n))
    raw = 0.5 * (raw + raw.swapaxes(1, 2))
    a = rng.normal(size=(q, q))
    metric = a.T @ a + 2.0 * np.eye(q)
    whitener = np.linalg.inv(np.linalg.cholesky(metric))
    c0, _ = np.linalg.qr(rng.normal(size=(n, n)))
    c1, _ = np.linalg.qr(rng.normal(size=(n, n)))
    frames = [np.ascontiguousarray(c0[:, :ext]), np.ascontiguousarray(c1[:, :ext])]
    return h, raw, whitener, frames


# ============================================================================
# B2: Streamed Fock loader and ExtFockNative ctypes validation
# ============================================================================


def test_b2_streamed_fock_short_panel_rejected_forward_and_reload(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit B2: loader returning (1, 2, 2) when whitener has q=2 must fail before native read."""
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_EXTFOCK", raising=False)
    h, raw, whitener, frames = _make_symmetric_fock_problem(1001, n=2, ext=2, q=2)
    short_raw = raw[:1].copy()

    with pytest.raises(ValueError, match="shape"):
        physical_spin_operators(
            h,
            None,
            whitener,
            frames,
            (1, 0),
            _cpu_runner(),
            False,
            None,
            raw_loader=lambda: short_raw,
            native_library=host_sink,
            host_reference=True,
        )

    # Also reject malformed/non-finite reload on pullback (second loader call).
    calls = 0

    def bad_reload_loader():
        nonlocal calls
        calls += 1
        return raw.copy() if calls == 1 else short_raw

    _, pb = physical_spin_operators(
        h,
        None,
        whitener,
        frames,
        (1, 0),
        _cpu_runner(),
        False,
        None,
        raw_loader=bad_reload_loader,
        native_library=host_sink,
        host_reference=True,
    )
    seed = {"fock_alpha": np.eye(2), "fock_beta": np.eye(2)}
    with pytest.raises(ValueError, match="shape"):
        pb(seed)

    # Reject NaN on reload.
    calls = 0

    def nan_reload_loader():
        nonlocal calls
        calls += 1
        buf = raw.copy()
        if calls > 1:
            buf[0, 0, 0] = np.nan
        return buf

    _, pb_nan = physical_spin_operators(
        h,
        None,
        whitener,
        frames,
        (1, 0),
        _cpu_runner(),
        False,
        None,
        raw_loader=nan_reload_loader,
        native_library=host_sink,
        host_reference=True,
    )
    with pytest.raises(ValueError, match="non-finite"):
        pb_nan(seed)

    # Reject non-finite hcore/whitener/frames.
    h_bad = h.copy()
    h_bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="Hamiltonian"):
        physical_spin_operators(
            h_bad, raw, whitener, frames, (1, 0), _cpu_runner(), False, None
        )
    w_bad = whitener.copy()
    w_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="whitener"):
        physical_spin_operators(
            h, raw, w_bad, frames, (1, 0), _cpu_runner(), False, None
        )


@pytest.mark.parametrize("use_aot", [True])
def test_b2_streamed_fock_reloads_same_primal(host_sink, monkeypatch, use_aot):
    if not use_aot:
        monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_EXTFOCK", "1")
    h, raw, whitener, frames = _make_symmetric_fock_problem(1273)
    current_panel = raw
    _, pullback = physical_spin_operators(
        h, None, whitener, frames, (1, 1), _cpu_runner(), False, None,
        raw_loader=lambda: current_panel, native_library=host_sink, host_reference=True,
    )
    seed = {"fock_alpha": np.eye(frames[0].shape[1])}
    original = pullback(seed)
    # A different array and a different storage order represent the same integrals.
    current_panel = np.asfortranarray(raw)
    reloaded = pullback(seed)
    np.testing.assert_allclose(reloaded[2], original[2], rtol=1e-13, atol=1e-13)
    current_panel[-1, 0, 0] += 1.0
    with pytest.raises(ValueError, match="changed since the forward"):
        pullback(seed)


def test_b2_ext_fock_native_ctypes_entry_validation(host_sink: Path) -> None:
    """Audit B2: ExtFockNative validates shape, dtype, writeability, finiteness at every entry."""
    native = bind_ext_fock_native(host_sink, host_reference=True)
    assert native is not None

    q, n, ext, oa = 2, 2, 2, 1
    w = np.eye(q, dtype=np.float64)
    occ = np.ones((n, oa), dtype=np.float64)
    short_raw = np.eye(n, dtype=np.float64)[None, :, :]  # qt=1 < q=2
    R_short = native.half_fwd(short_raw, occ)
    assert R_short.shape == (1, oa, n)

    # Exact B2 counterexample: R_a has 1 row while whitener has q=2.
    with pytest.raises(ValueError, match="R_a must have shape"):
        native.fwd(w, occ, None, R_short, None, same_occ_frame=False)

    # Wrong dtype / non-finite / read-only outputs rejected across all methods.
    with pytest.raises(ValueError, match="float64"):
        native.half_fwd(short_raw.astype(np.float32), occ)

    nan_raw = short_raw.copy()
    nan_raw[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        native.half_fwd(nan_raw, occ)

    ro_j = np.zeros((n, n), dtype=np.float64)
    ro_j.setflags(write=False)
    with pytest.raises(ValueError, match="writeable"):
        native.j_fwd(short_raw, np.ones(1, dtype=np.float64), ro_j)

    with pytest.raises(ValueError, match="rho_tile must have shape"):
        native.j_fwd(short_raw, np.ones(2, dtype=np.float64), np.zeros((n, n)))

    ro_bc = np.zeros((n, ext), dtype=np.float64)
    ro_bc.setflags(write=False)
    with pytest.raises(ValueError, match="writeable"):
        native.congruence_rev(np.eye(n), np.eye(n), np.eye(ext), ro_bc)

    with pytest.raises(ValueError, match="bar_g must have shape"):
        native.j_rev(short_raw, np.eye(n + 1))

    R_full = np.ones((q, oa, n), dtype=np.float64)
    with pytest.raises(ValueError, match="D_a must have shape"):
        native.rev(
            w, n, occ, None, R_full, None, R_short, None,
            np.ones(q), np.eye(n), None, np.zeros((n, oa)), None,
        )

    ro_bar_raw = np.zeros((1, n, n), dtype=np.float64)
    ro_bar_raw.setflags(write=False)
    with pytest.raises(ValueError, match="writeable"):
        native.half_rev(
            short_raw, occ, None, R_short, None, ro_bar_raw, np.zeros((n, oa)), None
        )


@pytest.mark.parametrize("use_native", [True])
def test_m1_physical_spin_operators_cached_loader_repeated_vjp_safe(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_native: bool,
) -> None:
    """Audit M1(2): repeated VJP calls with a cached-buffer raw_loader never corrupt input by default."""
    if use_native:
        monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_EXTFOCK", raising=False)
        monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK", "1")
    else:
        monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK", raising=False)
        monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_EXTFOCK", "1")

    h, raw, whitener, frames = _make_symmetric_fock_problem(1212, n=5, ext=4, q=6)
    cached_panel = raw.copy()
    rng = np.random.default_rng(1213)
    seeds = {
        "fock_alpha": rng.normal(size=(4, 4)),
        "fock_beta": rng.normal(size=(4, 4)),
        "exchange_alpha": rng.normal(size=(4, 4)),
        "exchange_beta": rng.normal(size=(4, 4)),
    }

    _, pb = physical_spin_operators(
        h,
        None,
        whitener,
        frames,
        (2, 1),
        _cpu_runner(),
        False,
        None,
        raw_loader=lambda: cached_panel,
        native_library=host_sink if use_native else None,
        host_reference=True,
    )
    bars1 = pb(seeds)
    np.testing.assert_array_equal(cached_panel, raw)
    assert bars1[2] is not cached_panel

    bars2 = pb(seeds)
    np.testing.assert_array_equal(cached_panel, raw)
    for b1, b2 in zip(bars1[0], bars2[0], strict=True):
        np.testing.assert_allclose(b1, b2, atol=1e-14, rtol=0.0)
    for idx in (1, 2, 3):
        np.testing.assert_allclose(bars1[idx], bars2[idx], atol=1e-14, rtol=0.0)

    # Explicit opt-in raw_loader_exclusive=True still permits in-place reuse.
    loaded: list[np.ndarray] = []

    def fresh_loader():
        buf = raw.copy()
        loaded.append(buf)
        return buf

    _, pb_excl = physical_spin_operators(
        h,
        None,
        whitener,
        frames,
        (2, 1),
        _cpu_runner(),
        False,
        None,
        raw_loader=fresh_loader,
        raw_loader_exclusive=True,
        native_library=host_sink if use_native else None,
        host_reference=True,
    )
    bars_excl = pb_excl(seeds)
    assert bars_excl[2] is loaded[1]
    np.testing.assert_allclose(bars_excl[2], bars1[2], atol=1e-13, rtol=0.0)


# ============================================================================
# M7: Per-buffer FFI validation across ao_mo, cv_dfmp2, spin_fock_1e
# ============================================================================


def test_m7_ao_mo_bilin_rev_same_object_shape_and_readonly(host_sink: Path) -> None:
    """Audit M7(1): run_ao_mo_bilin_rev validates both left and right shapes even when aliased."""
    ops = bind_ao_mo_ops(host_sink, host_reference=True)
    assert ops is not None

    nc, n_left, n_right = 2, 1, 2
    a_mat = np.eye(nc, dtype=np.float64)
    c_left = np.ones((nc, n_left), dtype=np.float64)
    c_right = np.ones((nc, n_right), dtype=np.float64)
    bar_out = np.ones((n_left, n_right), dtype=np.float64)
    shared = np.zeros((nc, n_left), dtype=np.float64)  # Too small for (2, 2)!

    with pytest.raises(ValueError, match="bar_c_right shape"):
        run_ao_mo_bilin_rev(
            ops, a_mat, c_left, c_right, bar_out,
            bar_c_left=shared, bar_c_right=shared,
        )

    # Read-only target rejected across bilin_rev, project_rev, coulomb_fit_rev.
    ro_bar = np.zeros((nc, n_left), dtype=np.float64)
    ro_bar.setflags(write=False)
    with pytest.raises(ValueError, match="writeable"):
        run_ao_mo_bilin_rev(
            ops, a_mat, c_left, c_left, np.ones((n_left, n_left)),
            bar_c_left=ro_bar,
        )

    ro_ao = np.zeros((1, nc, nc), dtype=np.float64)
    ro_ao.setflags(write=False)
    with pytest.raises(ValueError, match="writeable"):
        run_ao_mo_project_rev(
            ops, nc, nc, nc, 1, np.eye(nc), np.zeros((1, nc, nc)),
            np.zeros((1, nc, nc)), bar_ao=ro_ao,
        )

    ro_m = np.zeros((2, 2), dtype=np.float64)
    ro_m.setflags(write=False)
    with pytest.raises(ValueError, match="writeable"):
        run_coulomb_fit_rev(ops, np.eye(2), np.ones((2, 2)), np.ones((2, 2)), ro_m)


def test_m9_ext_fock_native_library_owns_the_requested_abi(host_sink, monkeypatch):
    response_only = next(Path(np.__file__).parent.rglob("_multiarray_umath*.so"))
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_EXTFOCK", raising=False)
    monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_EXT_FOCK_LIBRARY", str(host_sink))
    args = {"cuda": False, "host_reference": True}
    assert resolve_ext_fock_native(**args, library=host_sink, native_library=response_only) is None
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK", "1")
    with pytest.raises(RuntimeError, match="REQUIRE_AOT_EXTFOCK"):
        resolve_ext_fock_native(**args, native_library=response_only)

    # A separate response library can be paired with a configured fused Fock DSO.
    bound = resolve_ext_fock_native(**args, library=response_only)
    assert bound is not None and Path(bound.library_path).resolve() == host_sink.resolve()
    # A stale operation-specific override cannot be replaced by a lower priority library.
    monkeypatch.setenv("GENERATIVEQC_F12_EXT_FOCK_LIBRARY", str(response_only))
    monkeypatch.setenv("GENERATIVEQC_F12_HOST_SINK", str(host_sink))
    with pytest.raises(RuntimeError, match="REQUIRE_AOT_EXTFOCK"):
        resolve_ext_fock_native(**args)


def test_m2_ozaki_mm_deep_scale_counterexample_returns_exact_one() -> None:
    """Audit m2: A=(1, 2^-30, 2^-60, 2^-90, 2^-120), B=(0,0,0,0,2^120)^T must return 1.0, not 0.0."""
    A = np.array([[1.0, 2.0**-30, 2.0**-60, 2.0**-90, 2.0**-120]], dtype=np.float64)
    B = np.array([[0.0], [0.0], [0.0], [0.0], [2.0**120]], dtype=np.float64)

    hi, lo = _ozaki_mm(A, B, slices=4)
    np.testing.assert_allclose(hi + lo, np.array([[1.0]]), atol=1e-15, rtol=0.0)

    hi_dd, lo_dd = _ozaki_mm_left_dd(A, B, np.zeros_like(B), slices=4)
    np.testing.assert_allclose(hi_dd + lo_dd, np.array([[1.0]]), atol=1e-15, rtol=0.0)

    # Even deeper 20-layer scale separation (forces exact arbitrary-precision escalation!).
    exps = np.array([2.0 ** (-25 * k) for k in range(20)], dtype=np.float64)[None, :]
    b_deep = np.zeros((20, 1), dtype=np.float64)
    b_deep[-1, 0] = 2.0 ** (25 * 19)
    hi_deep, lo_deep = _ozaki_mm(exps, b_deep, slices=4)
    np.testing.assert_allclose(hi_deep + lo_deep, np.array([[1.0]]), atol=1e-15, rtol=0.0)

    # Non-finite input must raise ValueError.
    with pytest.raises(ValueError, match="Nonfinite"):
        _ozaki_mm(np.array([[np.nan]]), np.array([[1.0]]))


def test_m9_legacy_integral_bind_and_execute_consistent_norm_code(
    tmp_path: Path,
) -> None:
    """Audit m9: legacy fallback bind and contract_geometry agree identically on norm_code."""
    fake_lib = tmp_path / "lib_legacy_only.so"
    fake_lib.write_bytes(b"")

    def fake_legacy_kernel(
        ncenter, op, beta, nshell, nprim, nblocks, nweights,
        angular, prim_offsets, exponents, radial, centers,
        blocks, offsets, weights, output, budget, size_ptr,
    ):
        output[:] = 0.0
        output[:, 0] = 1.25
        ct.cast(size_ptr, ct.POINTER(ct.c_uint64))[0] = 64
        return 0

    def fake_bind(library, kernel, host_reference, *, extended=False):
        if extended:
            raise AttributeError("legacy library has no *_ex symbol")
        return fake_legacy_kernel

    norm_default = slater_normalization_code(LEGACY_SLATER_FIT_NORMALIZATION)
    norm_unscaled = slater_normalization_code("negative_exponential")
    assert norm_default == LEGACY_SLATER_NORM_CODE == 1
    assert norm_unscaled == 0

    # 1. Non-_ex bind: accepts compiled default norm_code=1 (1/beta), rejects norm_code=0.
    with patch("generativeqc._f12_focal_integral_sink.bind", side_effect=fake_bind):
        fn, is_ext = _bind_fallback_weighted(fake_lib, "geminal", True, norm_default)
        assert fn is fake_legacy_kernel and is_ext is False
        with pytest.raises(RuntimeError, match="SlaterFitNormalization"):
            _bind_fallback_weighted(fake_lib, "geminal", True, norm_unscaled)

    # 2. Non-_ex execute (contract_geometry with extended_abi=False):
    #    MUST agree identically — accepts norm_code=1 (1/beta), rejects norm_code=0!
    fixture = {
        "metric_blocks": np.array([[0, 0]], dtype=np.int32),
        "metric_offsets": np.array([0, 1], dtype=np.int32),
        "weights_metric_f12": np.array([1.0], dtype=np.float64),
        "angular": np.array([0], dtype=np.int32),
        "atom": np.array([0], dtype=np.int32),
        "prim_offsets": np.array([0, 1], dtype=np.int32),
        "exponents": np.array([1.0], dtype=np.float64),
        "radial": np.array([1.0], dtype=np.float64),
        "centers": np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
    }
    manifest = {
        "elements": ["H"],
        "geometry_bohr": [[0.0, 0.0, 0.0]],
        "options": {"f12_beta": 0.9},
    }

    label, row = contract_geometry(
        fixture, manifest, {"geminal": fake_legacy_kernel},
        "metric", "f12", 1 << 20, True,
        slater_normalization="negative_exponential_over_beta",
        extended_abi=False,
    )
    assert label == "metric_f12" and row["weighted_value_Eh"] == pytest.approx(1.25)

    with pytest.raises(RuntimeError, match="SlaterFitNormalization"):
        contract_geometry(
            fixture, manifest, {"geminal": fake_legacy_kernel},
            "metric", "f12", 1 << 20, True,
            slater_normalization="negative_exponential",
            extended_abi=False,
        )
