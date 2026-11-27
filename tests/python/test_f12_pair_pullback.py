"""Hardening unit tests for native CC-F12 MO-space pair pullbacks (W, C, Scalar)."""

from __future__ import annotations

import ctypes as ct
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from generativeqc._f12_cc_pair_pullback_native import (
    _validate_writable_output,
    bind_pair_pullback,
    run_c_pullback_native,
    run_scalar_pullback_native,
    run_w_pullback_native,
)


def _host_lib() -> Path:
    from conftest import configured_test_library
    from generativeqc import _native

    lib = configured_test_library("GENERATIVEQC_F12_HOST_SINK")
    if lib is None:
        lib = _native.installed_f12_shell_library(host_reference=True)
    if lib is None:
        pytest.skip("host focal_shell_i_host library not available")
    return lib


def _make_inputs(seed: int = 17, *, nl: int = 2, nr: int = 2, obs: int = 4, ext: int = 6, aux: int = 3):
    rng = np.random.default_rng(seed)
    return {
        "nl": nl,
        "nr": nr,
        "obs": obs,
        "ext": ext,
        "aux": aux,
        "fit_obs_l": rng.normal(scale=0.2, size=(aux, obs, obs)),
        "raw_obs_r": rng.normal(scale=0.2, size=(aux, obs, obs)),
        "fit_l": rng.normal(scale=0.2, size=(aux, nl, ext)),
        "fit_r": rng.normal(scale=0.2, size=(aux, nr, ext)),
        "raw_coul_r": rng.normal(scale=0.2, size=(aux, nr, ext)),
        "raw_g_l": rng.normal(scale=0.2, size=(aux, nl, ext)),
        "raw_g_r": rng.normal(scale=0.2, size=(aux, nr, ext)),
        "raw_gc_l": rng.normal(scale=0.2, size=(aux, nl, ext)),
        "raw_gc_r": rng.normal(scale=0.2, size=(aux, nr, ext)),
        "raw_sq_l": rng.normal(scale=0.2, size=(aux, nl, ext)),
        "raw_sq_r": rng.normal(scale=0.2, size=(aux, nr, ext)),
        "raw_dc_l": rng.normal(scale=0.2, size=(aux, nl, ext)),
        "raw_dc_r": rng.normal(scale=0.2, size=(aux, nr, ext)),
        "metric_g": rng.normal(scale=0.2, size=(aux, aux)),
        "metric_gc": rng.normal(scale=0.2, size=(aux, aux)),
        "metric_sq": rng.normal(scale=0.2, size=(aux, aux)),
        "metric_dc": rng.normal(scale=0.2, size=(aux, aux)),
        "fock_l": rng.normal(scale=0.2, size=(ext, ext)),
        "fock_r": rng.normal(scale=0.2, size=(ext, ext)),
        "exch_l": rng.normal(scale=0.2, size=(ext, ext)),
        "exch_r": rng.normal(scale=0.2, size=(ext, ext)),
        "cross_overlap": rng.normal(scale=0.2, size=(ext, ext)),
    }


def test_python_wrapper_validates_shapes_dtypes_writability_and_budgets() -> None:
    lib = _host_lib()
    w_fn = bind_pair_pullback(lib, "w", True)
    c_fn = bind_pair_pullback(lib, "c", True)
    s_fn = bind_pair_pullback(lib, "scalar", True)
    assert w_fn is not None and c_fn is not None and s_fn is not None

    d = _make_inputs()
    pairs = [(0, 0), (0, 1), (1, 1)]
    bar_w = np.ones((len(pairs), d["obs"], d["obs"]), dtype=np.float64)
    bar_c = np.ones((len(pairs), d["obs"] - d["nl"], d["obs"] - d["nr"]), dtype=np.float64)

    # Read-only output buffer rejected cleanly.
    ro = np.zeros((2, 2), dtype=np.float64)
    ro.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        _validate_writable_output(ro, (2, 2), "test_out")
    # Non-contiguous output buffer rejected cleanly.
    with pytest.raises(ValueError, match="C-contiguous"):
        _validate_writable_output(np.zeros((2, 2), dtype=np.float64, order="F"), (2, 2), "test_out")

    # Wrong shape rejected before ctypes.
    with pytest.raises(ValueError, match="bar_w_all"):
        run_w_pullback_native(
            w_fn,
            same_spin=True,
            nl=d["nl"],
            nr=d["nr"],
            obs=d["obs"],
            ext=d["ext"],
            aux=d["aux"],
            pairs=pairs,
            fit_obs_l=d["fit_obs_l"],
            raw_obs_r=d["raw_obs_r"],
            fit_l=d["fit_l"],
            fit_r=d["fit_r"],
            raw_g_l=d["raw_g_l"],
            raw_g_r=d["raw_g_r"],
            raw_gc_l=d["raw_gc_l"],
            raw_gc_r=d["raw_gc_r"],
            metric_g=d["metric_g"],
            metric_gc=d["metric_gc"],
            cross_overlap=None,
            bar_w_all=bar_w[:, :-1, :],
        )

    # Complex / bool dtype rejected before ctypes.
    with pytest.raises(ValueError, match="real numeric dtype"):
        run_c_pullback_native(
            c_fn,
            same_spin=True,
            nl=d["nl"],
            nr=d["nr"],
            obs=d["obs"],
            ext=d["ext"],
            aux=d["aux"],
            pairs=pairs,
            fit_l=d["fit_l"].astype(np.complex128),
            fit_r=d["fit_r"],
            raw_g_l=d["raw_g_l"],
            raw_g_r=d["raw_g_r"],
            metric_g=d["metric_g"],
            fock_l=d["fock_l"],
            fock_r=d["fock_r"],
            cross_overlap=None,
            bar_c_all=bar_c,
        )

    # Nonfinite input rejected before ctypes.
    bad_metric = d["metric_g"].copy()
    bad_metric[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        run_c_pullback_native(
            c_fn,
            same_spin=True,
            nl=d["nl"],
            nr=d["nr"],
            obs=d["obs"],
            ext=d["ext"],
            aux=d["aux"],
            pairs=pairs,
            fit_l=d["fit_l"],
            fit_r=d["fit_r"],
            raw_g_l=d["raw_g_l"],
            raw_g_r=d["raw_g_r"],
            metric_g=bad_metric,
            fock_l=d["fock_l"],
            fock_r=d["fock_r"],
            cross_overlap=None,
            bar_c_all=bar_c,
        )

    # Negative / bool budget_bytes and empty pairs rejected.
    for bad_budget in (-1, True):
        with pytest.raises(ValueError):
            run_c_pullback_native(
                c_fn,
                same_spin=True,
                nl=d["nl"],
                nr=d["nr"],
                obs=d["obs"],
                ext=d["ext"],
                aux=d["aux"],
                pairs=pairs,
                fit_l=d["fit_l"],
                fit_r=d["fit_r"],
                raw_g_l=d["raw_g_l"],
                raw_g_r=d["raw_g_r"],
                metric_g=d["metric_g"],
                fock_l=d["fock_l"],
                fock_r=d["fock_r"],
                cross_overlap=None,
                bar_c_all=bar_c,
                budget_bytes=bad_budget,
            )

    with pytest.raises(ValueError, match="non-empty"):
        run_c_pullback_native(
            c_fn,
            same_spin=True,
            nl=d["nl"],
            nr=d["nr"],
            obs=d["obs"],
            ext=d["ext"],
            aux=d["aux"],
            pairs=[],
            fit_l=d["fit_l"],
            fit_r=d["fit_r"],
            raw_g_l=d["raw_g_l"],
            raw_g_r=d["raw_g_r"],
            metric_g=d["metric_g"],
            fock_l=d["fock_l"],
            fock_r=d["fock_r"],
            cross_overlap=None,
            bar_c_all=np.zeros((0, d["obs"] - d["nl"], d["obs"] - d["nr"]), dtype=np.float64),
        )

    # Opposite-spin missing cross_overlap and nonfinite scalar_weight rejected.
    with pytest.raises(ValueError, match="cross_overlap"):
        run_c_pullback_native(
            c_fn,
            same_spin=False,
            nl=d["nl"],
            nr=d["nr"],
            obs=d["obs"],
            ext=d["ext"],
            aux=d["aux"],
            pairs=pairs,
            fit_l=d["fit_l"],
            fit_r=d["fit_r"],
            raw_g_l=d["raw_g_l"],
            raw_g_r=d["raw_g_r"],
            metric_g=d["metric_g"],
            fock_l=d["fock_l"],
            fock_r=d["fock_r"],
            cross_overlap=None,
            bar_c_all=bar_c,
        )

    with pytest.raises(ValueError, match="scalar_weight"):
        run_scalar_pullback_native(
            s_fn,
            same_spin=True,
            nl=d["nl"],
            nr=d["nr"],
            fl=0,
            fr=0,
            obs=d["obs"],
            ext=d["ext"],
            aux=d["aux"],
            scalar_weight=float("nan"),
            pairs=pairs,
            fit_l=d["fit_l"],
            fit_r=d["fit_r"],
            raw_coul_r=d["raw_coul_r"],
            raw_g_l=d["raw_g_l"],
            raw_g_r=d["raw_g_r"],
            raw_gc_l=d["raw_gc_l"],
            raw_gc_r=d["raw_gc_r"],
            raw_sq_l=d["raw_sq_l"],
            raw_sq_r=d["raw_sq_r"],
            raw_dc_l=d["raw_dc_l"],
            raw_dc_r=d["raw_dc_r"],
            fmix_fit_l=d["fit_l"],
            fmix_fit_r=d["fit_r"],
            fmix_g_l=d["raw_g_l"],
            fmix_g_r=d["raw_g_r"],
            fmix_sq_l=d["raw_sq_l"],
            fmix_sq_r=d["raw_sq_r"],
            metric_g=d["metric_g"],
            metric_gc=d["metric_gc"],
            metric_sq=d["metric_sq"],
            metric_dc=d["metric_dc"],
            fock_l=d["fock_l"],
            fock_r=d["fock_r"],
            exch_l=d["exch_l"],
            exch_r=d["exch_r"],
            cross_overlap=None,
        )


def test_native_c_abi_rejects_overflow_and_nan_without_mutating_outputs() -> None:
    lib = _host_lib()
    w_fn = bind_pair_pullback(lib, "w", True)
    c_fn = bind_pair_pullback(lib, "c", True)
    s_fn = bind_pair_pullback(lib, "scalar", True)
    assert w_fn is not None and c_fn is not None and s_fn is not None

    d = _make_inputs()
    pairs = [(0, 1)]
    # 1. Valid calls succeed for both same-spin and opposite-spin.
    w_res = run_w_pullback_native(
        w_fn,
        same_spin=False,
        nl=d["nl"],
        nr=d["nr"],
        obs=d["obs"],
        ext=d["ext"],
        aux=d["aux"],
        pairs=pairs,
        fit_obs_l=d["fit_obs_l"],
        raw_obs_r=d["raw_obs_r"],
        fit_l=d["fit_l"],
        fit_r=d["fit_r"],
        raw_g_l=d["raw_g_l"],
        raw_g_r=d["raw_g_r"],
        raw_gc_l=d["raw_gc_l"],
        raw_gc_r=d["raw_gc_r"],
        metric_g=d["metric_g"],
        metric_gc=d["metric_gc"],
        cross_overlap=d["cross_overlap"],
        bar_w_all=np.ones((1, d["obs"], d["obs"]), dtype=np.float64),
    )
    assert all(np.isfinite(v).all() for v in w_res.values())

    # 2. Finite inputs that overflow to Inf inside C++ must fail closed (status -2)
    # without mutating caller output arrays!
    huge = 1e160
    with pytest.raises(RuntimeError, match="status -2"):
        run_w_pullback_native(
            w_fn,
            same_spin=True,
            nl=d["nl"],
            nr=d["nr"],
            obs=d["obs"],
            ext=d["ext"],
            aux=d["aux"],
            pairs=pairs,
            fit_obs_l=np.full_like(d["fit_obs_l"], huge),
            raw_obs_r=np.full_like(d["raw_obs_r"], huge),
            fit_l=np.full_like(d["fit_l"], huge),
            fit_r=np.full_like(d["fit_r"], huge),
            raw_g_l=np.full_like(d["raw_g_l"], huge),
            raw_g_r=np.full_like(d["raw_g_r"], huge),
            raw_gc_l=np.full_like(d["raw_gc_l"], huge),
            raw_gc_r=np.full_like(d["raw_gc_r"], huge),
            metric_g=np.full_like(d["metric_g"], huge),
            metric_gc=np.full_like(d["metric_gc"], huge),
            cross_overlap=None,
            bar_w_all=np.full((1, d["obs"], d["obs"]), huge, dtype=np.float64),
        )

    with pytest.raises(RuntimeError, match="status -2"):
        run_c_pullback_native(
            c_fn,
            same_spin=True,
            nl=d["nl"],
            nr=d["nr"],
            obs=d["obs"],
            ext=d["ext"],
            aux=d["aux"],
            pairs=pairs,
            fit_l=np.full_like(d["fit_l"], huge),
            fit_r=np.full_like(d["fit_r"], huge),
            raw_g_l=np.full_like(d["raw_g_l"], huge),
            raw_g_r=np.full_like(d["raw_g_r"], huge),
            metric_g=np.full_like(d["metric_g"], huge),
            fock_l=np.full_like(d["fock_l"], huge),
            fock_r=np.full_like(d["fock_r"], huge),
            cross_overlap=None,
            bar_c_all=np.full((1, d["obs"] - d["nl"], d["obs"] - d["nr"]), huge, dtype=np.float64),
        )

    # 3. Direct C ABI check: sentinel outputs stay untouched when overflow fails closed.
    pi = np.array([0], dtype=np.int32)
    pj = np.array([1], dtype=np.int32)
    nl, nr, obs, ext, aux = 2, 2, 4, 6, 3
    ae_arr = np.full((nl, aux, ext), huge, dtype=np.float64)
    mg = np.full((aux, aux), huge, dtype=np.float64)
    fk = np.full((ext, ext), huge, dtype=np.float64)
    bc = np.full((1, obs - nl, obs - nr), huge, dtype=np.float64)
    sentinel = 42.5
    out_fl = np.full_like(ae_arr, sentinel)
    out_gl = np.full_like(ae_arr, sentinel)
    out_mg = np.full_like(mg, sentinel)
    out_fk = np.full_like(fk, sentinel)
    dptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_double))
    iptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_int32))
    rc = c_fn(
        1, nl, nr, obs, ext, aux, 1, 0, iptr(pi), iptr(pj),
        dptr(ae_arr), dptr(ae_arr), dptr(ae_arr), dptr(ae_arr),
        dptr(mg), dptr(fk), dptr(fk), ct.cast(0, ct.POINTER(ct.c_double)), dptr(bc),
        dptr(out_fl), dptr(out_fl), dptr(out_gl), dptr(out_gl),
        dptr(out_mg), dptr(out_fk), dptr(out_fk), ct.cast(0, ct.POINTER(ct.c_double)),
    )
    assert rc == -2
    assert np.all(out_fl == sentinel)
    assert np.all(out_gl == sentinel)
    assert np.all(out_mg == sentinel)
    assert np.all(out_fk == sentinel)

    # 4. Null pointer rejected with -1 at C ABI.
    null_d = ct.cast(0, ct.POINTER(ct.c_double))
    assert c_fn(
        1, nl, nr, obs, ext, aux, 1, 0, iptr(pi), iptr(pj),
        *([null_d] * 17),
    ) == -1


def test_cuda_pair_pullback_budget_enforced_before_cuda_init() -> None:
    from conftest import configured_test_library
    from generativeqc import _native

    library = configured_test_library("GENERATIVEQC_F12_PAIR_PULLBACK_CUDA_LIBRARY")
    if library is None:
        library = configured_test_library("GENERATIVEQC_F12_SHELL_LIBRARY")
    if library is None:
        library = _native.installed_f12_shell_library(host_reference=False)
    if library is None:
        pytest.skip("CUDA pair pullback library not configured")

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(library)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    codes = json.loads(result.stdout)
    # Tight 1 KiB budget must return kErrOom (-3) BEFORE cublasCreate;
    # ample 64 MiB budget must pass budget check and fail at cublasCreate (status 1) with no GPU.
    assert codes == {
        "w_tight": -3,
        "w_ample": 1,
        "c_tight": -3,
        "c_ample": 1,
        "scalar_tight": -3,
        "scalar_ample": 1,
    }, codes


if __name__ == "__main__":
    lib_path = Path(sys.argv[1])
    w_fn = bind_pair_pullback(lib_path, "w", False)
    c_fn = bind_pair_pullback(lib_path, "c", False)
    s_fn = bind_pair_pullback(lib_path, "scalar", False)
    assert w_fn is not None and c_fn is not None and s_fn is not None

    nl, nr, obs, ext, aux = 2, 2, 4, 6, 3
    pi = np.array([0], dtype=np.int32)
    pj = np.array([1], dtype=np.int32)
    dptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_double))
    iptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_int32))
    null_d = ct.cast(0, ct.POINTER(ct.c_double))

    aoo = np.zeros((aux, obs, obs), dtype=np.float64)
    ae = np.zeros((nl, aux, ext), dtype=np.float64)
    aa = np.zeros((aux, aux), dtype=np.float64)
    ee = np.zeros((ext, ext), dtype=np.float64)
    bw = np.zeros((1, obs, obs), dtype=np.float64)
    bc = np.zeros((1, obs - nl, obs - nr), dtype=np.float64)
    en = np.zeros(3, dtype=np.float64)

    def _call_w(budget: int) -> int:
        return w_fn(
            1, nl, nr, obs, ext, aux, 1, budget, iptr(pi), iptr(pj),
            dptr(aoo), dptr(aoo), dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae),
            dptr(aa), dptr(aa), null_d, dptr(bw),
            dptr(aoo), dptr(aoo), dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae),
            dptr(aa), dptr(aa), null_d,
        )

    def _call_c(budget: int) -> int:
        return c_fn(
            1, nl, nr, obs, ext, aux, 1, budget, iptr(pi), iptr(pj),
            dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(aa), dptr(ee), dptr(ee), null_d, dptr(bc),
            dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(aa), dptr(ee), dptr(ee), null_d,
        )

    def _call_s(budget: int) -> int:
        return s_fn(
            1, nl, nr, 0, 0, obs, ext, aux, 1, budget, 1.0, iptr(pi), iptr(pj),
            dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae),
            dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae),
            dptr(ae), dptr(ae), dptr(ae), dptr(aa), dptr(aa), dptr(aa), dptr(aa),
            dptr(ee), dptr(ee), dptr(ee), dptr(ee), null_d, dptr(en),
            dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae),
            dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae), dptr(ae),
            dptr(ae), dptr(ae), dptr(ae), dptr(aa), dptr(aa), dptr(aa), dptr(aa),
            dptr(ee), dptr(ee), dptr(ee), dptr(ee), null_d,
        )

    out = {
        "w_tight": _call_w(1024),
        "w_ample": _call_w(64 << 20),
        "c_tight": _call_c(1024),
        "c_ample": _call_c(64 << 20),
        "scalar_tight": _call_s(1024),
        "scalar_ample": _call_s(64 << 20),
    }
    print(json.dumps(out))
