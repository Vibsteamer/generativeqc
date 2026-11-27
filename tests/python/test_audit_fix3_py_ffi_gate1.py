"""Executable Gate 1 counterexample tests for Audit Fix-3 (Python FFI, cache, ownership, binding, projector)."""

from __future__ import annotations

import ctypes as ct
import gc
import os
import shutil
import subprocess
import sys
import threading
import types
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from conftest import f12_host_shell

from generativeqc._f12_ao_mo_native import (
    bind_ao_mo_ops,
    resolve_ao_mo_ops,
    run_ao_mo_bilin_rev,
    run_ao_mo_project_rev,
    run_coulomb_fit_rev,
)
from generativeqc._f12_cabs_fock import physical_spin_operators
from generativeqc._f12_cc_fixed_sources_native import (
    AUTO_NATIVE_LIBRARY,
    bind_fixed_pair_sources,
    discover_ccfixed_library,
    resolve_ccfixed_library,
)
from generativeqc._f12_cphf_relax_native import (
    resolve_cphf_relax,
    run_df_canonical_driver_native,
)
from generativeqc._f12_ext_fock_native import bind_ext_fock_native, resolve_ext_fock_native
from generativeqc._f12_focal_integral_kernel import (
    LEGACY_SLATER_FIT_NORMALIZATION,
    LEGACY_SLATER_NORM_CODE,
    contract_geometry,
    slater_normalization_code,
)
from generativeqc._f12_focal_integral_sink import _bind_fallback_weighted
from generativeqc._f12_focal_projector import _ozaki_mm, _ozaki_mm_left_dd
from generativeqc._f12_spin_ao import (
    _orbital_sources,
    clear_projected_source_cache,
    projected_source_cache_info,
)
from generativeqc._f12_spin_fock_1e_native import (
    SpinFock1eNativeOps,
    _prep_output_buffer,
    bind_spin_fock_1e_ops,
    resolve_spin_fock_1e_ops,
    run_spin_fock_rev,
)
from generativeqc._f12_unrestricted import unrestricted_mp2_f12_ao_response
from generativeqc.f12_orbital_response import unrestricted_source_orbital_adjoint


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
        "src/posthf/f12/f12_spin_fock_1e_cuda.cu",
        "src/posthf/f12/f12_cphf_relax_cuda.cu",
        "src/posthf/f12/f12_cc_fixed_sources_cuda.cu",
        "src/posthf/f12/f12_mp2_pair_response_cuda.cu",
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


@pytest.mark.parametrize("use_aot", [True, False])
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


# ============================================================================
# B3 + m8: Large writable AO panel stale cache & concurrent LRU locking
# ============================================================================


def test_b3_large_writable_ao_panel_mutation_never_returns_stale_cache(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit B3: 36 MiB FP64 (128, 192, 192) writable AO mutation on slice 1 must never hit stale cache."""
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_AOMO", raising=False)
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_HAD_REUSE", raising=False)
    clear_projected_source_cache(reset_stats=True)

    aux, nc, obs, ext = 128, 192, 2, 3
    panel = np.zeros((aux, nc, nc), dtype=np.float64)
    assert panel.nbytes == 36 * 1024 * 1024  # 36 MiB > 32 MiB threshold
    panel[0, 0, 0] = 1.0
    panel[1, 0, 0] = 2.0

    rng = np.random.default_rng(303)
    c_mat = np.zeros((nc, ext), dtype=np.float64)
    c_mat[:4, :] = rng.normal(size=(4, ext))
    c_mat[0, :] = np.array([1.0, 0.5, -0.25])

    ov = np.eye(nc, dtype=np.float64)
    metric = np.eye(aux, dtype=np.float64)
    ao = {
        "overlap_ao": ov,
        "hcore_ao": np.zeros((nc, nc), dtype=np.float64),
        "coefficients_alpha": c_mat,
        "coefficients_beta": c_mat,
        "metric_whitener": metric,
        "metric/coulomb": metric,
        "metric/geminal": metric,
        "metric/squared_geminal": metric,
        "metric/geminal_coulomb": metric,
        "metric/double_commutator": metric,
        "ao_coulomb": panel,
        "ao_f12": panel,
        "ao_f12_squared": panel,
        "ao_f12_over_coulomb": panel,
        "ao_double_commutator": panel,
    }
    space = (1, 1, obs, ext, aux)
    run = _cpu_runner()

    d1, _ = _orbital_sources(
        space, ao, run, None, projected=None, role="pairs",
        shell_library=host_sink, host_reference=True,
    )
    assert projected_source_cache_info()["size"] == 0  # Writable >32 MiB never cached

    # Mutate ONLY unsampled slice 1 (old sample indices were {0, 32, 64, 96, 127}).
    delta = 7.5
    panel[1, 0, 0] += delta
    d2, _ = _orbital_sources(
        space, ao, run, None, projected=None, role="pairs",
        shell_library=host_sink, host_reference=True,
    )
    expected_delta = delta * np.outer(c_mat[0], c_mat[0])
    np.testing.assert_allclose(
        d2["source/raw_coulomb_alpha"][1] - d1["source/raw_coulomb_alpha"][1],
        expected_delta,
        atol=1e-12,
        rtol=0.0,
    )

    # A read-only VIEW backed by a writable base MUST ALSO NOT be cached.
    ro_view = panel.view()
    ro_view.setflags(write=False)
    ao_view = {**ao, "ao_coulomb": ro_view, "ao_f12": ro_view, "ao_f12_squared": ro_view,
               "ao_f12_over_coulomb": ro_view, "ao_double_commutator": ro_view}
    d3, _ = _orbital_sources(
        space, ao_view, run, None, projected=None, role="pairs",
        shell_library=host_sink, host_reference=True,
    )
    assert projected_source_cache_info()["size"] == 0
    panel[1, 0, 0] += 3.0
    d4, _ = _orbital_sources(
        space, ao_view, run, None, projected=None, role="pairs",
        shell_library=host_sink, host_reference=True,
    )
    np.testing.assert_allclose(
        d4["source/raw_coulomb_alpha"][1] - d3["source/raw_coulomb_alpha"][1],
        3.0 * np.outer(c_mat[0], c_mat[0]),
        atol=1e-12,
        rtol=0.0,
    )

    # A truly immutable (writeable=False, base is None) >32 MiB panel DOES cache safely.
    panel.setflags(write=False)
    clear_projected_source_cache(reset_stats=True)
    d5, _ = _orbital_sources(
        space, ao, run, None, projected=None, role="pairs",
        shell_library=host_sink, host_reference=True,
    )
    d6, _ = _orbital_sources(
        space, ao, run, None, projected=None, role="pairs",
        shell_library=host_sink, host_reference=True,
    )
    stats = projected_source_cache_info()
    assert stats["size"] == 1 and stats["hits"] == 1
    np.testing.assert_array_equal(
        d5["source/raw_coulomb_alpha"], d6["source/raw_coulomb_alpha"]
    )
    clear_projected_source_cache(reset_stats=True)


def _project_streamed_sources(loader, nc, aux, host_sink):
    frame = np.eye(nc, 3)
    metric = np.eye(aux)
    ao = {
        "overlap_ao": np.eye(nc), "hcore_ao": np.eye(nc),
        "coefficients_alpha": frame, "coefficients_beta": frame,
        "metric_whitener": metric, "__ao_loader__": loader,
        **{"metric/" + name: metric for name in (
            "coulomb", "geminal", "squared_geminal", "geminal_coulomb", "double_commutator"
        )},
    }

    def evaluate():
        values, _ = _orbital_sources(
            (1, 1, 2, 3, aux), ao, _cpu_runner(), None, projected=None,
            role="pairs", shell_library=host_sink, host_reference=True,
        )
        return values["source/raw_coulomb_alpha"]

    return evaluate


@pytest.mark.parametrize("use_panel_owner", [False, True])
def test_b3_mutable_streamed_panel_reloads_with_identical_metadata(
    host_sink, monkeypatch, use_panel_owner
):
    from generativeqc._f12_focal_molecular import _Panel

    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_HAD_REUSE", raising=False)
    clear_projected_source_cache(reset_stats=True)
    raw = np.zeros((2, 3, 3))
    data = {name: raw for name in (
        "ao_coulomb", "ao_f12", "ao_f12_squared", "ao_f12_over_coulomb", "ao_double_commutator"
    )}

    class CustomPanel:
        layout = {"manifest": "unchanged"}
        fit_id = "fit"
        slater_normalization = "negative_exponential"
        host_reference = True
        coulomb_only = False

        def load_ao(self, name):
            return data[name]

    owner = (
        _Panel({"manifest": "unchanged"}, data, np.eye(2), "fit")
        if use_panel_owner else CustomPanel()
    )
    evaluate = _project_streamed_sources(owner.load_ao, 3, 2, host_sink)
    np.testing.assert_array_equal(evaluate(), raw)
    raw[1, 0, 0] = 2.5
    np.testing.assert_array_equal(evaluate(), raw)


def test_m8_projected_lru_concurrent_access_and_clear_no_keyerror(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit m8: concurrent _orbital_sources + clear_projected_source_cache never raises KeyError."""
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_AOMO", raising=False)
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_HAD_REUSE", raising=False)
    clear_projected_source_cache(reset_stats=True)

    rng = np.random.default_rng(808)
    nc, obs, ext, aux = 5, 3, 4, 4
    c_mat, _ = np.linalg.qr(rng.normal(size=(nc, nc)))
    c_mat = np.ascontiguousarray(c_mat[:, :ext])
    ov = np.eye(nc, dtype=np.float64)
    metric = np.eye(aux, dtype=np.float64)

    def make_ao(scale: float):
        p = rng.normal(size=(aux, nc, nc)) * scale
        p = 0.5 * (p + p.swapaxes(1, 2))
        return {
            "overlap_ao": ov,
            "hcore_ao": np.eye(nc, dtype=np.float64) * scale,
            "coefficients_alpha": c_mat,
            "coefficients_beta": c_mat,
            "metric_whitener": metric,
            "metric/coulomb": metric,
            "metric/geminal": metric,
            "metric/squared_geminal": metric,
            "metric/geminal_coulomb": metric,
            "metric/double_commutator": metric,
            "ao_coulomb": p,
            "ao_f12": p,
            "ao_f12_squared": p,
            "ao_f12_over_coulomb": p,
            "ao_double_commutator": p,
        }

    ao_variants = [make_ao(1.0 + 0.1 * k) for k in range(3)]
    space = (1, 1, obs, ext, aux)
    run = _cpu_runner()
    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def worker(tid: int):
        try:
            barrier.wait()
            for it in range(30):
                if tid == 0 and it % 3 == 0:
                    clear_projected_source_cache()
                else:
                    _orbital_sources(
                        space, ao_variants[(tid + it) % 3], run, None,
                        projected=None, role="pairs",
                        shell_library=host_sink, host_reference=True,
                    )
                    projected_source_cache_info()
        except BaseException as exc:  # pragma: no cover - test expects empty errors
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    clear_projected_source_cache(reset_stats=True)


# ============================================================================
# M1: Two in-place overwrite ownership bugs
# ============================================================================


def test_m1_unrestricted_jkfit_q1_snapshot_not_corrupted(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit M1(1): q_jk=1 independent JKFit must copy occupied slice before inplace spin_fock_pullback."""
    monkeypatch.setenv("GENERATIVEQC_F12_SHELL_LIBRARY", str(host_sink))
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_SPINFOCK_1E", raising=False)
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_AOMO", raising=False)
    clear_projected_source_cache(reset_stats=True)

    rng = np.random.default_rng(1111)
    oa, ob, obs, ext, aux = 1, 1, 2, 3, 2
    nc = ext
    c_mat, _ = np.linalg.qr(rng.normal(size=(nc, nc)))
    c_mat = np.ascontiguousarray(c_mat)

    def sym3(q_dim):
        x = rng.normal(size=(q_dim, nc, nc)) * 0.25
        return np.ascontiguousarray(0.5 * (x + x.swapaxes(1, 2)))

    def spd(q_dim):
        a = rng.normal(size=(q_dim, q_dim)) * 0.2
        m = a.T @ a + 2.0 * np.eye(q_dim)
        w = np.linalg.inv(np.linalg.cholesky(m))
        return m, w

    m_corr, w_corr = spd(aux)
    m_jk, w_jk = spd(1)  # EXACT M1 COUNTEREXAMPLE: q_jk == 1!
    h_ao = np.diag([-1.5, 0.8, 1.9]) + 0.05 * (rng.normal(size=(nc, nc)))
    h_ao = 0.5 * (h_ao + h_ao.T)

    ao = {
        "overlap_ao": np.eye(nc, dtype=np.float64),
        "hcore_ao": h_ao,
        "coefficients_alpha": c_mat,
        "coefficients_beta": c_mat,
        "metric_whitener": w_corr,
        "metric/coulomb": m_corr,
        "metric/geminal": m_corr * 0.7,
        "metric/squared_geminal": m_corr * 0.5,
        "metric/geminal_coulomb": m_corr * 0.6,
        "metric/double_commutator": m_corr * 0.4,
        "ao_coulomb": sym3(aux),
        "ao_f12": sym3(aux),
        "ao_f12_squared": sym3(aux),
        "ao_f12_over_coulomb": sym3(aux),
        "ao_double_commutator": sym3(aux),
    }
    jk_ao = sym3(1)
    jkfit = {
        "ao_coulomb": jk_ao,
        "metric/coulomb": m_jk,
        "metric_whitener": w_jk,
    }
    space = (oa, ob, obs, ext, aux)
    run = _cpu_runner()
    from generativeqc._f12_focal_singles import _reference_singles

    def cpu_singles(matrix, occupied, *, spin_weight=1, pullback=True):
        e, bar = _reference_singles(matrix, occupied, pullback)
        return spin_weight * e, (None if bar is None else spin_weight * bar)

    terms, bars = unrestricted_mp2_f12_ao_response(
        space, ao, run, jkfit=jkfit, component="singles", singles=cpu_singles
    )
    assert abs(sum(terms.cabs_singles)) > 1e-6

    # Independent finite-difference check on jkfit metric and jkfit ao_coulomb!
    eps = 1e-6

    def eval_singles(m_val: float, ao_mat: np.ndarray) -> float:
        m_mat = np.array([[m_val]], dtype=np.float64)
        w_mat = np.array([[1.0 / np.sqrt(m_val)]], dtype=np.float64)
        e, _ = unrestricted_mp2_f12_ao_response(
            space,
            ao,
            run,
            jkfit={
                "ao_coulomb": ao_mat,
                "metric/coulomb": m_mat,
                "metric_whitener": w_mat,
            },
            component="singles",
            singles=cpu_singles,
        )
        return float(sum(e.cabs_singles))

    fd_metric = (eval_singles(m_jk[0, 0] + eps, jk_ao) - eval_singles(m_jk[0, 0] - eps, jk_ao)) / (2.0 * eps)
    np.testing.assert_allclose(bars["jkfit/metric/coulomb"][0, 0], fd_metric, atol=1e-7, rtol=1e-6)

    d_ao = rng.normal(size=jk_ao.shape)
    d_ao = 0.5 * (d_ao + d_ao.swapaxes(1, 2))
    fd_ao = (eval_singles(m_jk[0, 0], jk_ao + eps * d_ao) - eval_singles(m_jk[0, 0], jk_ao - eps * d_ao)) / (2.0 * eps)
    analytic_ao = float(np.sum(bars["jkfit/ao_coulomb"] * d_ao))
    np.testing.assert_allclose(analytic_ao, fd_ao, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("use_native", [True, False])
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
# M2: Verify obs_three_center source identity in reference_cabs_singles
# ============================================================================


@pytest.mark.parametrize("use_aot", [True, False])
@pytest.mark.parametrize("use_loader", [False, True])
def test_m2_reference_cabs_singles_rejects_mismatched_obs_three_center(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_aot: bool,
    use_loader: bool,
) -> None:
    """Audit M2: supplying zeroed/mismatched obs_three_center raises ValueError before freeing tc_local."""
    from generativeqc._f12_focal_singles import reference_cabs_singles

    if use_aot:
        monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_AOMO", raising=False)
        monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_AOMO", "1")
    else:
        monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_AOMO", raising=False)
        monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_AOMO", "1")

    obs, rank, q = 3, 2, 4
    atomic = obs + rank
    rng = np.random.default_rng(2026)

    s = np.eye(atomic, dtype=np.float64)
    c_obs = np.eye(obs, dtype=np.float64)
    complement = np.zeros((atomic, rank), dtype=np.float64)
    complement[obs:, :] = np.eye(rank, dtype=np.float64)

    a = rng.normal(size=(q, q)) * 0.2
    metric = a.T @ a + 2.0 * np.eye(q)
    whitener = np.linalg.inv(np.linalg.cholesky(metric))

    tc = rng.normal(size=(q, atomic, atomic)) * 0.1
    tc = np.ascontiguousarray(0.5 * (tc + tc.swapaxes(1, 2)))

    # Construct h so that the canonical OBS Fock matches diag(eps).
    occ = 1
    b_scaled = np.einsum("ab,bpq->apq", whitener, tc)
    d_occ = np.zeros((atomic, atomic), dtype=np.float64)
    d_occ[0, 0] = 1.0
    j_mat = np.einsum("apq,a->pq", b_scaled, 2.0 * np.einsum("aij,ij->a", b_scaled, d_occ))
    k_mat = np.einsum("api,ij,aqj->pq", b_scaled, d_occ, b_scaled)
    eps = np.array([-1.2, 0.7, 1.6], dtype=np.float64)
    h = np.zeros((atomic, atomic), dtype=np.float64)
    h[:obs, :obs] = np.diag(eps) - (j_mat - k_mat)[:obs, :obs]
    h[obs:, obs:] = np.diag([2.1, 2.9])
    h = 0.5 * (h + h.T)

    basis = types.SimpleNamespace(
        basis_id="b-m2",
        orbital=types.SimpleNamespace(identity="orb-m2"),
        jk_fit=types.SimpleNamespace(identity="jk-m2"),
        cabs=types.SimpleNamespace(identity="cabs-m2"),
    )
    request = types.SimpleNamespace(
        role="hf_cabs",
        root_id="rhf",
        backend="cpu_reference",
        surface_id="s-m2",
        basis=basis,
    )
    reference = types.SimpleNamespace(
        surface_id="s-m2",
        basis_id="b-m2",
        executed_backend="cpu_reference",
        jk_fit_id="jk-m2",
        orbital_id="orb-m2",
        overlap=s[:obs, :obs].copy(),
        hcore=h[:obs, :obs].copy(),
        occupied=(occ,),
        coefficients=c_obs[None, :, :],
        orbital_energies=eps[None, :],
        energy=-1.0,
    )
    one = types.SimpleNamespace(
        surface_id="s-m2",
        basis_id="b-m2",
        executed_backend="cpu_reference",
        overlap=s,
        hcore=h,
        obs_ao_count=obs,
    )

    mismatched_obs = np.zeros((q, obs, obs), dtype=np.float64)
    with pytest.raises(ValueError, match="obs_three_center does not match"):
        reference_cabs_singles(
            request,
            reference,
            one,
            complement=complement,
            three_center=None if use_loader else tc,
            three_center_loader=(lambda: tc.copy()) if use_loader else None,
            obs_three_center=mismatched_obs,
            metric=metric,
            whitener=whitener,
            pullback=True,
            relax=False,
            shell_library=host_sink if use_aot else None,
        )

    # Matching obs_three_center succeeds and agrees with obs_three_center=None.
    matching_obs = tc[:, :obs, :obs].copy()
    res_match = reference_cabs_singles(
        request,
        reference,
        one,
        complement=complement,
        three_center=None if use_loader else tc,
        three_center_loader=(lambda: tc.copy()) if use_loader else None,
        obs_three_center=matching_obs,
        metric=metric,
        whitener=whitener,
        pullback=True,
        relax=False,
        shell_library=host_sink if use_aot else None,
    )
    res_none = reference_cabs_singles(
        request,
        reference,
        one,
        complement=complement,
        three_center=None if use_loader else tc,
        three_center_loader=(lambda: tc.copy()) if use_loader else None,
        obs_three_center=None,
        metric=metric,
        whitener=whitener,
        pullback=True,
        relax=False,
        shell_library=host_sink if use_aot else None,
    )
    np.testing.assert_allclose(res_match.hf_cabs_energy, res_none.hf_cabs_energy, atol=1e-13)

    if use_loader and not use_aot:
        # Only the forward and Fock reverse see the original source. An unrelated
        # load for orbital relaxation would use the changed OBS block.
        from generativeqc import _f12_focal_singles_response

        panels = iter((tc, tc))
        observed = {}

        def capture_response(req, ref, one, frame, raw, whitener, weights, *args, **kwargs):
            observed.update(raw)
            return weights

        monkeypatch.setattr(_f12_focal_singles_response, "relaxed_weights", capture_response)
        reference_cabs_singles(
            request, reference, one, complement=complement, three_center=None,
            three_center_loader=lambda: next(panels, np.zeros_like(tc)),
            metric=metric, whitener=whitener, pullback=True, relax=True,
        )
        for spin in ("alpha", "beta"):
            np.testing.assert_allclose(observed["raw_coulomb_" + spin], matching_obs)


# ============================================================================
# M3 (Python): Empty-spin Coulomb contribution into occupied spin channel
# ============================================================================


def test_m3_empty_spin_coulomb_driver_exact_counterexample(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit M3: q=1, n=2, oa=1, ob=0, fock_seed_beta=diag(1,0) must yield alpha driver == 1.0."""
    df_mat = np.array([[[1.0, 0.5], [0.5, 1.0]]], dtype=np.float64)
    seed_zero = np.zeros((2, 2), dtype=np.float64)
    seed_unit = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    central = {
        "df_alpha": df_mat,
        "df_beta": df_mat,
        "fock_seed_alpha": seed_zero,
        "fock_seed_beta": seed_unit,
    }
    energies = {
        "alpha": np.array([-1.0, 1.0], dtype=np.float64),
        "beta": np.array([-1.0, 1.0], dtype=np.float64),
    }

    # Direct AOT evaluation includes the empty-spin Coulomb seed exactly once.
    fn = resolve_cphf_relax(host_sink, "df_canonical_driver", host_reference=True)
    assert fn is not None
    ch_aot = run_df_canonical_driver_native(
        fn, oa=1, ob=0, obs=2, aux=1,
        df_a=df_mat, fock_seed_a=seed_zero,
        df_b=df_mat, fock_seed_b=seed_unit,
    )
    np.testing.assert_allclose(ch_aot["alpha"], np.array([[1.0]]), atol=1e-13, rtol=0.0)

    # A resource-constrained ABI still supplies the complete driver at the caller's budget.
    c_driver = ct.CFUNCTYPE(fn.restype, *fn.argtypes)

    @c_driver
    def constrained_driver(oa, ob, n, aux, budget, *buffers):
        return fn(oa, ob, n, aux, budget, *buffers) if budget >= 1 << 20 else -3

    constrained = run_df_canonical_driver_native(
        constrained_driver, oa=1, ob=0, obs=2, aux=1, budget_bytes=1 << 20,
        df_a=df_mat, fock_seed_a=seed_zero, df_b=df_mat, fock_seed_b=seed_unit,
    )
    np.testing.assert_allclose(constrained["alpha"], [[1.0]], atol=1e-13, rtol=0.0)

    # Symmetric case: oa=0, ob=1, non-zero fock_seed_alpha.
    ch_aot_sym = run_df_canonical_driver_native(
        fn, oa=0, ob=1, obs=2, aux=1,
        df_a=df_mat, fock_seed_a=seed_unit,
        df_b=df_mat, fock_seed_b=seed_zero,
    )
    np.testing.assert_allclose(ch_aot_sym["beta"], np.array([[1.0]]), atol=1e-13, rtol=0.0)

    # 2. End-to-end unrestricted_source_orbital_adjoint on BOTH AOT and TensorIR paths.
    dummy_solve = types.SimpleNamespace(rotations=np.zeros(1, dtype=np.float64))
    with patch("generativeqc.f12_orbital_response._solve_orbital_response", return_value=dummy_solve):
        drv_aot, _ = unrestricted_source_orbital_adjoint(
            (1, 0, 2, 1),
            central,
            energies,
            _cpu_runner(),
            native_library=host_sink,
            host_reference=True,
        )
        np.testing.assert_allclose(drv_aot, np.array([1.0]), atol=1e-13, rtol=0.0)

        monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_CPHF", "1")
        monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_CPHF", raising=False)
        monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_CPHF_STEP1", raising=False)
        drv_ir, _ = unrestricted_source_orbital_adjoint(
            (1, 0, 2, 1),
            central,
            energies,
            _cpu_runner(),
            host_reference=True,
        )
        np.testing.assert_allclose(drv_ir, np.array([1.0]), atol=1e-13, rtol=0.0)


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


def test_m7_reverse_rejects_overlapping_operands_before_writing(host_sink):
    ops = bind_ao_mo_ops(host_sink, host_reference=True)
    spin_ops = bind_spin_fock_1e_ops(host_sink, host_reference=True)
    assert ops is not None and spin_ops is not None
    storage = np.arange(6, dtype=np.float64)
    matrix, target = storage[:4].reshape(2, 2), storage[2:].reshape(2, 2)
    original = storage.copy()
    with pytest.raises(ValueError, match="overlap"):
        run_ao_mo_bilin_rev(ops, matrix, np.eye(2), np.eye(2), np.eye(2), bar_a=target)
    with pytest.raises(ValueError, match="overlap"):
        run_ao_mo_project_rev(
            ops, 2, 2, 2, 1, np.eye(2), matrix[None], np.ones((1, 2, 2)),
            bar_ao=target[None], bar_coefficients=np.zeros((2, 2)),
        )
    with pytest.raises(ValueError, match="overlap"):
        run_ao_mo_project_rev(
            ops, 2, 2, 2, 1, np.eye(2), np.ones((1, 2, 2)), np.ones((1, 2, 2)),
            bar_ao=target[None], bar_coefficients=matrix,
        )
    sources = {
        "raw_coulomb_alpha": matrix[None], "raw_coulomb_beta": matrix[None],
        "fit_alpha": np.ones((1, 1, 2)), "fit_beta": np.ones((1, 1, 2)),
    }
    with pytest.raises(ValueError, match="overlap"):
        run_spin_fock_rev(
            spin_ops, (1, 1, 2, 1), sources, {"fock_alpha": np.eye(2)},
            accumulate={"raw_coulomb_alpha": target[None]},
        )
    np.testing.assert_array_equal(storage, original)


def test_m7_spin_fock_1e_rejects_readonly_accumulate_and_c_int_overflow(
    host_sink: Path,
) -> None:
    """Audit M7(3): _prep_output_buffer rejects read-only arrays and validates C int bounds."""
    ro_buf = np.zeros((2, 3, 3), dtype=np.float64)
    ro_buf.setflags(write=False)
    with pytest.raises(ValueError, match="read-only"):
        _prep_output_buffer({"raw_coulomb_alpha": ro_buf}, "raw_coulomb_alpha", (2, 3, 3), accumulate_out=True)

    ops = bind_spin_fock_1e_ops(host_sink, host_reference=True)
    assert ops is not None
    n, q = 3, 2
    raw = np.ones((q, n, n), dtype=np.float64)
    fit = np.ones((q, 1, n), dtype=np.float64)
    sources = {
        "raw_coulomb_alpha": raw,
        "raw_coulomb_beta": raw.copy(),
        "fit_alpha": fit,
        "fit_beta": fit.copy(),
    }
    acc = {k: v.copy() for k, v in sources.items()}
    acc["raw_coulomb_alpha"].setflags(write=False)
    with pytest.raises(ValueError, match="read-only"):
        run_spin_fock_rev(ops, (1, 1, n, q), sources, {"fock_alpha": np.eye(n)}, accumulate=acc)

    # C int overflow check: n=50000 => n*n = 2.5e9 > INT_MAX.
    with pytest.raises(ValueError, match="C int"):
        run_spin_fock_rev(ops, (1, 1, 50000, 1), sources, {"fock_alpha": np.eye(n)})


# ============================================================================
# M9: Strict single-backend library binding, precedence & unsafe inplace guard
# ============================================================================


def test_m9_strict_single_backend_no_cross_match(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit M9: requesting CUDA backend never matches host_reference library (and vice versa)."""
    # Ensure host_sink is mapped into /proc/self/maps!
    _loaded = ct.CDLL(str(host_sink))
    assert _loaded is not None

    monkeypatch.setenv("GENERATIVEQC_F12_SHELL_LIBRARY", str(host_sink))
    monkeypatch.setenv("GENERATIVEQC_F12_HOST_SINK", str(host_sink))
    monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_CCFIXED", raising=False)
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_CCFIXED", raising=False)

    # CUDA runner stub (has .batch and .profile_snapshot like focal_cuda_runner).
    def cuda_run(p, i):
        raise AssertionError("not called")

    cuda_run.batch = lambda g: None  # type: ignore[attr-defined]
    cuda_run.profile_snapshot = lambda: {}  # type: ignore[attr-defined]

    # 1. cc_fixed_sources must NEVER cross-match host_sink when CUDA is requested.
    path, is_host = discover_ccfixed_library(host_reference=False)
    assert path is None and is_host is False

    path2, is_host2 = resolve_ccfixed_library(AUTO_NATIVE_LIBRARY, None, cuda_run)
    assert path2 is None and is_host2 is False

    path3, is_host3 = resolve_ccfixed_library(host_sink, None, cuda_run)
    assert is_host3 is False
    assert bind_fixed_pair_sources(path3, is_host3) is None

    # 2. cphf_relax requires explicit single-backend boolean and never cross-matches.
    with pytest.raises(ValueError, match="explicit host_reference"):
        resolve_cphf_relax(host_sink, "whiten_3c", host_reference=None)
    assert resolve_cphf_relax(host_sink, "whiten_3c", host_reference=False) is None

    # 3. ao_mo and spin_fock_1e never cross-match CUDA against host_sink.
    assert resolve_ao_mo_ops(cuda_run, None, shell_library=host_sink) is None
    assert resolve_spin_fock_1e_ops(cuda_run, library=host_sink) is None


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


def test_m9_mp2_pair_and_spinfock_precedence_explicit_gt_env_gt_maps(
    host_sink: Path,
    tmp_path: Path,
) -> None:
    """Audit M9: explicit library > explicit env > /proc/self/maps precedence (isolated subprocess)."""
    maps_copy = tmp_path / "libfocal_shell_maps_loaded.so"
    env_copy = tmp_path / "libfocal_shell_env_override.so"
    explicit_copy = tmp_path / "libfocal_shell_explicit_override.so"
    shutil.copy2(host_sink, maps_copy)
    shutil.copy2(host_sink, env_copy)
    shutil.copy2(host_sink, explicit_copy)

    script = """
import ctypes as ct
import os
import sys
from pathlib import Path
from generativeqc._f12_ao_mo_native import resolve_ao_mo_ops
from generativeqc._f12_mp2_pair_native import _resolve_symbol as mp2_resolve_symbol
from generativeqc._f12_spin_fock_1e_native import resolve_spin_fock_1e_ops

maps_copy, env_copy, explicit_copy = map(Path, sys.argv[1:4])
_loaded_maps = ct.CDLL(str(maps_copy))
assert _loaded_maps is not None

os.environ["GENERATIVEQC_F12_SHELL_LIBRARY"] = str(env_copy)
os.environ.pop("GENERATIVEQC_F12_AO_MO_LIBRARY", None)

def addr(fn):
    return int(ct.cast(fn, ct.c_void_p).value or 0)

# 1. mp2_pair_native: explicit > env > maps.
fn_env = mp2_resolve_symbol(
    "gqc_f12_mp2_pair_energies_host_reference", None, host_reference=True
)
fn_explicit = mp2_resolve_symbol(
    "gqc_f12_mp2_pair_energies_host_reference", explicit_copy, host_reference=True
)
fn_maps = mp2_resolve_symbol(
    "gqc_f12_mp2_pair_energies_host_reference", maps_copy, host_reference=True
)
assert fn_env is not None and fn_explicit is not None and fn_maps is not None
assert addr(fn_env) != addr(fn_maps), "env must take precedence over /proc/self/maps"
assert addr(fn_explicit) != addr(fn_env), "explicit arg must take precedence over env"

# 2. ao_mo_native: explicit > env > maps.
ops_env = resolve_ao_mo_ops(None, None, host_reference=True)
assert ops_env is not None and Path(ops_env.library_name).resolve() == env_copy.resolve()
ops_exp = resolve_ao_mo_ops(None, None, shell_library=explicit_copy, host_reference=True)
assert ops_exp is not None and Path(ops_exp.library_name).resolve() == explicit_copy.resolve()

# 3. spin_fock_1e_native: explicit > env > registered > maps.
sf_env = resolve_spin_fock_1e_ops(None, host_reference=True)
assert sf_env is not None and Path(sf_env.library_name).resolve() == env_copy.resolve()
sf_exp = resolve_spin_fock_1e_ops(None, library=explicit_copy, host_reference=True)
assert sf_exp is not None and Path(sf_exp.library_name).resolve() == explicit_copy.resolve()
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "python")
    proc = subprocess.run(
        [sys.executable, "-c", script, str(maps_copy), str(env_copy), str(explicit_copy)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_m9_implicit_discovery_requires_one_artifact(host_sink: Path, tmp_path: Path):
    """An explicit selection is authoritative; implicit matching requires one binary identity."""
    first = tmp_path / "libfocal_shell_first.so"
    identical = tmp_path / "libfocal_shell_equal.so"
    other = tmp_path / "libfocal_shell_other.so"
    for dest in (first, identical, other):
        shutil.copyfile(host_sink, dest)
    # The dynamic loader accepts an ELF trailer. Both binaries export the same
    # backend ABI while their artifact identities differ.
    with other.open("ab") as binary:
        binary.write(b"independent-shell-build")
    script = """
import ctypes
import os
import sys
from pathlib import Path

from generativeqc import _native
from generativeqc._f12_ao_mo_native import resolve_ao_mo_ops, resolve_cabs_singles_ops
from generativeqc._f12_cc_fixed_sources_native import discover_ccfixed_library
from generativeqc._f12_mp2_pair_native import bind_ordinary_df_mp2
from generativeqc._f12_spin_fock_1e_native import register_shell_library, resolve_spin_fock_1e_ops

first, identical, other = map(Path, sys.argv[1:])
_native.PACKAGE_DIR = first.parent / "empty-package"
for key in tuple(os.environ):
    if key.startswith(("GENERATIVEQC_F12_", "GQC_F12_")):
        os.environ.pop(key)
bindings = (
    lambda: resolve_ao_mo_ops(None, host_reference=True),
    lambda: resolve_cabs_singles_ops(None, host_reference=True),
    lambda: resolve_spin_fock_1e_ops(None, host_reference=True),
    lambda: bind_ordinary_df_mp2(host_reference=True),
    lambda: discover_ccfixed_library(host_reference=True)[0],
)
handles = [ctypes.CDLL(str(path)) for path in (first, identical)]
assert all(bind() is not None for bind in bindings)
register_shell_library(first)
register_shell_library(other)
handles.append(ctypes.CDLL(str(other)))
for bind in bindings:
    try:
        bind()
    except RuntimeError as error:
        assert "ambiguous" in str(error).lower(), error
    else:
        raise AssertionError(f"selected one of two different implicit artifacts: {bind}")
os.environ["GENERATIVEQC_F12_SHELL_LIBRARY"] = str(first)
assert all(bind() is not None for bind in bindings)
# A stale configured library is a capability miss. A valid lower-level choice
# and two compatible mapped libraries remain present.
os.environ["GENERATIVEQC_F12_HOST_SINK"] = str(first)
import numpy as np
numpy_dso = next(Path(np.__file__).parent.rglob("_multiarray_umath*.so"))
for bad in (first.parent / "missing.so", numpy_dso):
    os.environ["GENERATIVEQC_F12_SHELL_LIBRARY"] = str(bad)
    assert all(bind() is None for bind in bindings)
"""
    subprocess.run([sys.executable, "-c", script, str(first), str(identical), str(other)],
                   check=True)


def test_m9_spin_fock_1e_disables_inplace_on_unsafe_older_library(
    host_sink: Path,
) -> None:
    """Audit M9: spin_fock_rev falls back to separate buffer when library corrupts aliased raw_a."""
    real_ops = bind_spin_fock_1e_ops(host_sink, host_reference=True)
    assert real_ops is not None

    # Wrap real_ops.spin_fock_rev to simulate an older library that corrupts bar_fit_a
    # when out_raw_a aliases raw_a.
    def legacy_corrupting_rev(
        oa, ob, n, q, share_beta_raw, accumulate_out, q_tile_hint,
        raw_a, raw_b, fit_a, fit_b, bar_fa, bar_fb, bar_ka, bar_kb,
        out_hc_a, out_hc_b, out_raw_a, out_raw_b, out_fit_a, out_fit_b,
    ):
        rc = real_ops.spin_fock_rev(
            oa, ob, n, q, share_beta_raw, accumulate_out, q_tile_hint,
            raw_a, raw_b, fit_a, fit_b, bar_fa, bar_fb, bar_ka, bar_kb,
            out_hc_a, out_hc_b, out_raw_a, out_raw_b, out_fit_a, out_fit_b,
        )
        if ct.cast(raw_a, ct.c_void_p).value == ct.cast(out_raw_a, ct.c_void_p).value:
            # Simulate pre-alias-safe library clobbering out_fit_a[0]!
            out_fit_a[0] = 999999.0
        return rc

    legacy_ops = SpinFock1eNativeOps(
        host_reference=True,
        library_name="<simulated-legacy-so>",
        one_electron_grad=real_ops.one_electron_grad,
        spin_fock_fwd=real_ops.spin_fock_fwd,
        spin_fock_rev=legacy_corrupting_rev,
    )

    rng = np.random.default_rng(909)
    n, q = 4, 3
    raw = rng.normal(size=(q, n, n))
    raw = np.ascontiguousarray(0.5 * (raw + raw.swapaxes(1, 2)))
    fit = np.ascontiguousarray(rng.normal(size=(q, 1, n)))
    seed = rng.normal(size=(n, n))
    sources = {
        "raw_coulomb_alpha": raw.copy(),
        "raw_coulomb_beta": raw.copy(),
        "fit_alpha": fit.copy(),
        "fit_beta": fit.copy(),
    }
    ref_bars = run_spin_fock_rev(
        real_ops, (1, 1, n, q), sources,
        {"fock_alpha": seed, "fock_beta": seed.copy()},
        inplace_raw_coulomb=False,
    )
    raw_in = sources["raw_coulomb_alpha"]
    got_bars = run_spin_fock_rev(
        legacy_ops, (1, 1, n, q), sources,
        {"fock_alpha": seed, "fock_beta": seed.copy()},
        inplace_raw_coulomb=True,
    )
    # Alias-safety probe detected legacy corruption -> allocated separate buffer!
    assert got_bars["raw_coulomb_alpha"] is not raw_in
    np.testing.assert_allclose(got_bars["fit_alpha"], ref_bars["fit_alpha"], atol=1e-13)
    np.testing.assert_allclose(got_bars["raw_coulomb_alpha"], ref_bars["raw_coulomb_alpha"], atol=1e-13)


def test_m9_focal_cc_sources_passes_provider_shell_library_and_backend(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit M9: molecular_ucc_sources forwards provider.shell_library and host_reference=(not cuda)."""
    from generativeqc._f12_focal_cc_sources import (
        clear_had_cc_source_cache,
        molecular_ucc_sources,
    )

    clear_had_cc_source_cache(reset_stats=True)
    obs, ext, aux = 2, 3, 2
    eps = np.array([-1.0, 1.0], dtype=np.float64)
    fock_mat = np.diag([-1.0, 1.0, 2.0])

    captured: dict = {}

    def spy_feeds(*args, **kwargs):
        captured.update(kwargs)
        return {"fixed_geminal_energy": np.array(0.0)}

    dummy_panel = types.SimpleNamespace(
        fit_id="f12-fit",
        whitener=np.eye(aux, dtype=np.float64),
        joint=lambda req, explicit=False: {},
    )
    provider = types.SimpleNamespace(
        run=_cpu_runner(),
        metric_library=None,
        response_library=None,
        shell_library=host_sink,
        budget_mib=128,
        relative_cutoff=1e-9,
        slater_normalization="negative_exponential_over_beta",
        _seed_options=lambda req: {},
        _cached_pair_sources=lambda req, derivatives=False, budget=128: (
            types.SimpleNamespace(),
            (dummy_panel, dummy_panel, dummy_panel),
        ),
    )
    basis = types.SimpleNamespace(
        basis_id="b-cc",
        orbital=types.SimpleNamespace(identity="orb"),
        jk_fit=types.SimpleNamespace(identity="jk"),
        f12_fit=types.SimpleNamespace(identity="f12-fit"),
        mp2_fit=types.SimpleNamespace(identity="f12-fit"),
        cabs=types.SimpleNamespace(identity="cabs"),
        beta=1.0,
    )
    request = types.SimpleNamespace(
        role="mp2_f12_correlation",
        root_id="rhf",
        multiplicity=1,
        charge=0,
        backend="cpu_reference",
        surface_id="surf",
        correlation_space_id="all-active",
        atoms=(types.SimpleNamespace(atomic_number=1, position=(0.0, 0.0, 0.0)),),
        basis=basis,
    )
    ref = types.SimpleNamespace(
        surface_id="surf",
        basis_id="b-cc",
        executed_backend="cpu_reference",
        orbital_id="orb",
        jk_fit_id="jk",
        forces=None,
        occupied=(1,),
        coefficients=np.eye(obs, dtype=np.float64)[None, :, :],
        orbital_energies=eps[None, :],
        energy=-1.0,
        _pair_spin_indices=lambda req: (0, 0),
    )
    proj = types.SimpleNamespace(rank=1)

    with (
        patch(
            "generativeqc._f12_focal_cc_sources.canonical_pair_frame",
            return_value=((1, 1, obs, ext, aux), {}, None, proj),
        ),
        patch(
            "generativeqc._f12_focal_cc_sources._orbital_sources",
            return_value=({}, None),
        ),
        patch(
            "generativeqc._f12_focal_cc_sources.extended_fock_response",
            return_value=({"fock_alpha": fock_mat, "fock_beta": fock_mat}, None),
        ),
        patch(
            "generativeqc._f12_focal_cc_sources.unrestricted_cc_f12_fixed_source_feeds",
            side_effect=spy_feeds,
        ),
    ):
        molecular_ucc_sources(
            provider,
            request,
            include_ordinary=False,
            reference=ref,
        )

    assert captured.get("native_library") == host_sink
    assert captured.get("host_reference") is True
    clear_had_cc_source_cache(reset_stats=True)


# ============================================================================
# M12: Streaming reverse releases bar_ao_mat AND target_bar_ao before next load
# ============================================================================


def test_m12_streaming_reverse_zero_retained_bar_ao_before_next_load(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit M12: previous operator's bar_ao panel has zero retained references before next ao_loader call."""
    import generativeqc._f12_spin_ao as spin_ao_mod

    class _TrackableArray(np.ndarray):
        pass

    orig_zeros = np.zeros

    def tracked_zeros(shape, dtype=float, *args, **kwargs):
        arr = orig_zeros(shape, dtype, *args, **kwargs)
        if isinstance(shape, tuple) and len(shape) == 3:
            return arr.view(_TrackableArray)
        return arr

    monkeypatch.setattr(spin_ao_mod.np, "zeros", tracked_zeros)
    clear_projected_source_cache(reset_stats=True)

    rng = np.random.default_rng(12120)
    nc, obs, ext, aux = 5, 3, 4, 4
    c_mat, _ = np.linalg.qr(rng.normal(size=(nc, nc)))
    c_mat = np.ascontiguousarray(c_mat[:, :ext])
    ov = np.eye(nc, dtype=np.float64)
    metric = np.eye(aux, dtype=np.float64)
    panels = {}
    for name in ("ao_coulomb", "ao_f12", "ao_f12_squared", "ao_f12_over_coulomb", "ao_double_commutator"):
        p = rng.normal(size=(aux, nc, nc)) * 0.1
        panels[name] = np.ascontiguousarray(0.5 * (p + p.swapaxes(1, 2)))

    prev_bar_refs: list[tuple[str, weakref.ReferenceType]] = []
    liveness_checks: list[tuple[str, bool]] = []

    def ao_loader(name: str) -> np.ndarray:
        # Before loading the next operator's panel, ALL previously sunk bar_ao panels must be dead!
        for prev_name, ref in prev_bar_refs:
            liveness_checks.append((f"before_load_{name}_prev_{prev_name}_dead", ref() is None))
        return panels[name].copy()

    def ao_bar_sink(name: str, bar_ao: np.ndarray) -> None:
        assert isinstance(bar_ao, _TrackableArray)
        prev_bar_refs.append((name, weakref.ref(bar_ao)))

    ao = {
        "overlap_ao": ov,
        "hcore_ao": np.eye(nc, dtype=np.float64),
        "coefficients_alpha": c_mat,
        "coefficients_beta": c_mat,
        "metric_whitener": metric,
        "metric/coulomb": metric,
        "metric/geminal": metric,
        "metric/squared_geminal": metric,
        "metric/geminal_coulomb": metric,
        "metric/double_commutator": metric,
        "__ao_loader__": ao_loader,
    }
    space = (1, 1, obs, ext, aux)
    run = _cpu_runner()

    proj, _ = _orbital_sources(
        space, ao, run, None, projected=None, role="shared",
        shell_library=host_sink, host_reference=True,
    )
    prev_bar_refs.clear()
    liveness_checks.clear()

    weights = {k: np.ones_like(v) * 0.1 for k, v in proj.items() if k != "space"}
    _orbital_sources(
        space, ao, run, weights, projected=proj, role="shared",
        ao_bar_sink=ao_bar_sink, shell_library=host_sink, host_reference=True,
    )
    gc.collect()
    assert len(prev_bar_refs) == 5
    assert len(liveness_checks) > 0
    assert all(dead for _, dead in liveness_checks), f"Retained bar_ao alias detected: {liveness_checks}"
    assert all(ref() is None for _, ref in prev_bar_refs)


# ============================================================================
# m1, m2, m9: Singles MINRES fallback, Ozaki remainder escalation, legacy norm_code
# ============================================================================


def test_m1_singles_response_tensorir_fallback_uses_use_matrix_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit m1: auto-switched CUDA UHF matrix-free fallback passes gap preconditioner & 1e-12 tol."""
    import scipy.sparse.linalg as spla
    from generativeqc._f12_focal_singles_response import relaxed_reference_weights

    # Disable AOT CPHF so we hit the TensorIR MINRES fallback branch (line 620).
    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_CPHF", "1")
    monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_CPHF", raising=False)
    monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_CPHF_STEP1", raising=False)

    captured_kwargs: dict = {}
    orig_minres = spla.minres
    import functools

    @functools.wraps(orig_minres)
    def spy_minres(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return orig_minres(*args, **kwargs)

    # For budget_mib=1 (1 MiB), obs=28, oa=14, ob=13 => r = 196 + 195 = 391 => 8*r^2 = 1.22 MiB > 1 MiB.
    obs2, aux2, oa2, ob2 = 28, 2, 14, 13
    q_a2 = np.eye(obs2, dtype=np.float64)
    q_b2 = np.eye(obs2, dtype=np.float64)
    eps_a2 = np.concatenate([np.linspace(-3.0, -1.0, oa2), np.linspace(1.0, 3.0, obs2 - oa2)])
    eps_b2 = np.concatenate([np.linspace(-3.0, -1.0, ob2), np.linspace(1.0, 3.0, obs2 - ob2)])

    def gauge_bar2(occ):
        m = np.zeros((obs2, obs2), dtype=np.float64)
        m[:occ, occ:] = 0.01
        return m

    raw2 = np.zeros((aux2, obs2, obs2), dtype=np.float64)
    raw2[0] = 0.01 * np.eye(obs2, dtype=np.float64)
    reference2 = types.SimpleNamespace(
        occupied=(oa2, ob2),
        executed_backend="cuda",
        overlap=np.eye(obs2, dtype=np.float64),
        coefficients=np.stack([q_a2, q_b2]),
        orbital_energies=np.stack([eps_a2, eps_b2]),
    )
    weights2 = {
        "obs_coefficients": np.stack([gauge_bar2(oa2), gauge_bar2(ob2)]),
        "hcore": np.zeros((obs2, obs2), dtype=np.float64),
        "three_center": np.zeros((aux2, obs2, obs2), dtype=np.float64),
        "metric": np.zeros((aux2, aux2), dtype=np.float64),
    }

    dummy_bars = {
        "overlap_ao": np.zeros((obs2, obs2), dtype=np.float64),
        "hcore_ao": np.zeros((obs2, obs2), dtype=np.float64),
        "ao_coulomb": np.zeros((aux2, obs2, obs2), dtype=np.float64),
        "metric/coulomb": np.zeros((aux2, aux2), dtype=np.float64),
    }
    with (
        patch("scipy.sparse.linalg.minres", new=spy_minres),
        patch(
            "generativeqc.f12_orbital_response.unrestricted_nuclear_source_weights",
            return_value=dummy_bars,
        ),
    ):
        relaxed_reference_weights(
            reference2,
            {"raw_coulomb_alpha": raw2, "raw_coulomb_beta": raw2},
            np.eye(aux2, dtype=np.float64),
            weights2,
            np.zeros((obs2, obs2), dtype=np.float64),
            _cpu_runner(),
            None,
            budget_mib=1,  # 8 * 391^2 = 1.22 MiB > 1 MiB -> use_matrix_free=True while matrix_free=False!
            matrix_free=False,
            host_reference=True,
        )

    assert captured_kwargs.get("M") is not None, "Expected gap preconditioner when use_matrix_free=True"
    tol_val = captured_kwargs.get("rtol", captured_kwargs.get("tol"))
    assert tol_val == pytest.approx(1e-12), f"Expected 1e-12 tolerance, got {tol_val}"


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
