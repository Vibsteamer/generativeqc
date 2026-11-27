"""Gate 1 verification for native AOT extended physical AO DF Fock (Patch A-2a)."""

from __future__ import annotations

import ctypes
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import configured_test_library

from generativeqc._f12_cabs_fock import physical_fock, physical_spin_operators
from generativeqc._f12_cc_fock_pullback import extended_fock_response
from generativeqc._f12_ext_fock_native import bind_ext_fock_native
from generativeqc_compiler.common.compiler_cache import (
    resolve_compiler_cache,
    run_cached_compiler,
)
from generativeqc_compiler.tensor import execute

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def host_ext_fock_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    explicit = configured_test_library("GENERATIVEQC_F12_EXT_FOCK_LIBRARY")
    if explicit is not None:
        if bind_ext_fock_native(explicit, host_reference=True) is None:
            pytest.fail(f"{explicit} does not export the host ext-Fock ABI")
        return explicit

    cxx = shutil.which("c++")
    assert cxx is not None, "C++ compiler required to build host reference ext-fock library"
    launcher = resolve_compiler_cache()
    build_dir = tmp_path_factory.mktemp("f12_ext_fock_host")
    out_lib = build_dir / "libf12_ext_fock_host.so"
    cmd = [
        cxx,
        "-std=c++20",
        "-O3",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-DGQC_SINK_HOST_REFERENCE",
        "-shared",
        "-x",
        "c++",
        str(REPO / "src/posthf/f12/f12_ext_fock_cuda.cu"),
        "-o",
        str(out_lib),
    ]
    res = run_cached_compiler(cmd, 300, label="f12_ext_fock host reference", launcher=launcher)
    assert res.returncode == 0, f"host reference compile failed:\n{res.stdout}\n{res.stderr}"
    return out_lib


def _make_problem(
    seed: int,
    *,
    n: int,
    ext: int,
    q: int,
    oa: int,
    ob: int,
    same_occ_frame: bool,
    non_symmetric_whitener: bool = True,
):
    rng = np.random.default_rng(seed)
    h_raw = rng.normal(scale=0.5, size=(n, n))
    h = 0.5 * (h_raw + h_raw.T)

    a_raw = rng.normal(scale=0.25, size=(q, n, n))
    raw = 0.5 * (a_raw + a_raw.swapaxes(1, 2))

    # Build a well-conditioned full-rank whitener W (not symmetric when requested)
    # such that M = (W^T W)^{-1} is SPD.
    w_base = rng.normal(scale=0.1, size=(q, q)) + 1.0 * np.eye(q)
    if not non_symmetric_whitener:
        w_base = 0.5 * (w_base + w_base.T)
    whitener = np.ascontiguousarray(w_base, dtype=np.float64)

    c_a = rng.normal(scale=0.4, size=(n, ext))
    if same_occ_frame:
        c_b = rng.normal(scale=0.4, size=(n, ext))
        c_b[:, :oa] = c_a[:, :oa]
    else:
        c_b = rng.normal(scale=0.4, size=(n, ext))
    return h, raw, whitener, [c_a, c_b], (oa, ob)


class CountingRunner:
    """Wrap TensorIR execute() and count every invocation."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, program, inputs):
        self.calls += 1
        return execute(program, inputs)


def _assert_parity(
    mats_ref: dict[str, np.ndarray],
    bars_ref,
    mats_nat: dict[str, np.ndarray],
    bars_nat,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> None:
    assert mats_nat.keys() == mats_ref.keys()
    for key in mats_ref:
        np.testing.assert_allclose(mats_nat[key], mats_ref[key], atol=atol, rtol=rtol)

    c_ref, h_ref, raw_ref, m_ref = bars_ref
    c_nat, h_nat, raw_nat, m_nat = bars_nat
    assert len(c_nat) == len(c_ref) == 2
    for s in range(2):
        np.testing.assert_allclose(c_nat[s], c_ref[s], atol=atol, rtol=rtol)
    np.testing.assert_allclose(h_nat, h_ref, atol=atol, rtol=rtol)
    np.testing.assert_allclose(raw_nat, raw_ref, atol=atol, rtol=rtol)
    np.testing.assert_allclose(m_nat, m_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("case_id", "case_name", "n", "ext", "q", "oa", "ob", "same_occ_frame"),
    [
        (1, "rhf_same_occ", 9, 7, 15, 3, 3, True),
        (2, "uhf_general", 10, 8, 17, 4, 2, False),
        (3, "uhf_high_spin_ob0", 8, 6, 13, 3, 0, False),
        (4, "uhf_high_spin_oa0", 7, 5, 11, 0, 2, False),
    ],
)
@pytest.mark.parametrize("include_exchange", [True, False])
@pytest.mark.parametrize("streaming_inplace", [False, True])
def test_physical_spin_operators_parity_vs_tensorir(
    host_ext_fock_library: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: int,
    case_name: str,
    n: int,
    ext: int,
    q: int,
    oa: int,
    ob: int,
    same_occ_frame: bool,
    include_exchange: bool,
    streaming_inplace: bool,
) -> None:
    del case_name
    seed_base = 1000 * case_id + 100 * int(include_exchange) + 10 * int(streaming_inplace)
    h, raw, whitener, frames, occupied = _make_problem(
        seed_base,
        n=n,
        ext=ext,
        q=q,
        oa=oa,
        ob=ob,
        same_occ_frame=same_occ_frame,
    )
    rng = np.random.default_rng(seed_base + 7)
    # Intentionally non-symmetric cotangent seeds (and distinct alpha vs beta even in RHF).
    seeds = {
        "fock_alpha": rng.normal(size=(ext, ext)) + 2.0 * np.triu(rng.normal(size=(ext, ext)), 1),
        "fock_beta": rng.normal(size=(ext, ext)) - 1.5 * np.tril(rng.normal(size=(ext, ext)), -1),
    }
    if include_exchange:
        seeds["exchange_alpha"] = rng.normal(size=(ext, ext))
        seeds["exchange_beta"] = rng.normal(size=(ext, ext))

    # 1. TensorIR reference run (explicitly disable AOT ext-fock).
    monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_EXTFOCK", "1")
    ref_runner = CountingRunner()
    if streaming_inplace:
        ref_loads = 0

        def ref_loader():
            nonlocal ref_loads
            ref_loads += 1
            return raw.copy()

        mats_ref, pb_ref = physical_spin_operators(
            h,
            None,
            whitener,
            frames,
            occupied,
            ref_runner,
            False,
            None,
            include_exchange=include_exchange,
            raw_loader=ref_loader,
        )
        bars_ref = pb_ref(seeds)
        assert ref_loads == 2
    else:
        raw_ref_in = raw.copy()
        mats_ref, pb_ref = physical_spin_operators(
            h,
            raw_ref_in,
            whitener,
            frames,
            occupied,
            ref_runner,
            False,
            None,
            include_exchange=include_exchange,
        )
        bars_ref = pb_ref(seeds)
    assert ref_runner.calls > 0

    # 2. Native AOT run with strict enforcement and tensorir_calls == 0.
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_EXTFOCK", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK", "1")
    nat_runner = CountingRunner()
    if streaming_inplace:
        loaded_buffers: list[np.ndarray] = []

        def nat_loader():
            buf = raw.copy()
            loaded_buffers.append(buf)
            return buf

        mats_nat, pb_nat = physical_spin_operators(
            h,
            None,
            whitener,
            frames,
            occupied,
            nat_runner,
            False,
            None,
            include_exchange=include_exchange,
            raw_loader=nat_loader,
            raw_loader_exclusive=True,
            native_library=host_ext_fock_library,
            host_reference=True,
        )
        bars_nat = pb_nat(seeds)
        assert len(loaded_buffers) == 2
        # Verify true in-place overwrite of the reloaded streaming buffer.
        assert bars_nat[2] is loaded_buffers[1]
    else:
        raw_nat_in = raw.copy()
        mats_nat, pb_nat = physical_spin_operators(
            h,
            raw_nat_in,
            whitener,
            frames,
            occupied,
            nat_runner,
            False,
            None,
            include_exchange=include_exchange,
            native_library=host_ext_fock_library,
            host_reference=True,
        )
        bars_nat = pb_nat(seeds)
        # Non-streaming raw array must remain untouched.
        np.testing.assert_array_equal(raw_nat_in, raw)
        assert bars_nat[2] is not raw_nat_in

    assert nat_runner.calls == 0, f"Expected tensorir_calls == 0, got {nat_runner.calls}"
    _assert_parity(mats_ref, bars_ref, mats_nat, bars_nat, atol=1e-12)


def test_partial_seeds_and_noncontiguous_raw(
    host_ext_fock_library: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h, raw, whitener, frames, occupied = _make_problem(
        4242, n=8, ext=6, q=14, oa=3, ob=2, same_occ_frame=False
    )
    rng = np.random.default_rng(4243)
    partial_seeds = {
        "fock_alpha": rng.normal(size=(6, 6)),
        "exchange_beta": rng.normal(size=(6, 6)),
    }

    # Build a strided (non-C-contiguous) view over raw.
    padded = np.zeros((raw.shape[0] * 2, raw.shape[1], raw.shape[2]), dtype=np.float64)
    padded[::2] = raw
    raw_strided = padded[::2]
    assert not raw_strided.flags.c_contiguous

    monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_EXTFOCK", "1")
    mats_ref, pb_ref = physical_spin_operators(
        h, raw, whitener, frames, occupied, CountingRunner(), False, None, include_exchange=True
    )
    bars_ref = pb_ref(partial_seeds)

    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_EXTFOCK", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK", "1")
    nat_runner = CountingRunner()
    mats_nat, pb_nat = physical_spin_operators(
        h,
        raw_strided,
        whitener,
        frames,
        occupied,
        nat_runner,
        False,
        None,
        include_exchange=True,
        native_library=host_ext_fock_library,
        host_reference=True,
    )
    bars_nat = pb_nat(partial_seeds)
    assert nat_runner.calls == 0
    _assert_parity(mats_ref, bars_ref, mats_nat, bars_nat, atol=1e-12)

    # Also test non-contiguous raw_loader (inplace_raw=True).
    def strided_loader():
        buf = np.zeros((raw.shape[0] * 2, raw.shape[1], raw.shape[2]), dtype=np.float64)
        buf[::2] = raw
        return buf[::2]

    nat_runner_stream = CountingRunner()
    mats_s, pb_s = physical_spin_operators(
        h,
        None,
        whitener,
        frames,
        occupied,
        nat_runner_stream,
        False,
        None,
        include_exchange=True,
        raw_loader=strided_loader,
        native_library=host_ext_fock_library,
        host_reference=True,
    )
    bars_s = pb_s(partial_seeds)
    assert nat_runner_stream.calls == 0
    _assert_parity(mats_ref, bars_ref, mats_s, bars_s, atol=1e-12)


@pytest.mark.parametrize("separate_jkfit", [False, True])
def test_extended_fock_response_and_strict_env(
    host_ext_fock_library: Path,
    monkeypatch: pytest.MonkeyPatch,
    separate_jkfit: bool,
) -> None:
    n, ext, q, oa, ob = 9, 7, 15, 3, 2
    h, raw, whitener, frames, _ = _make_problem(
        9001 + int(separate_jkfit), n=n, ext=ext, q=q, oa=oa, ob=ob, same_occ_frame=False
    )
    space = (oa, ob, ext - oa, ext - ob, ext)
    rng = np.random.default_rng(9010)
    seeds = {
        "fock_alpha": rng.normal(size=(ext, ext)),
        "fock_beta": rng.normal(size=(ext, ext)),
        "exchange_alpha": rng.normal(size=(ext, ext)),
        "exchange_beta": rng.normal(size=(ext, ext)),
    }

    ao = {
        "hcore_ao": h,
        "coefficients_alpha": frames[0],
        "coefficients_beta": frames[1],
    }
    if separate_jkfit:
        jkfit = {"ao_coulomb": raw, "metric_whitener": whitener}
    else:
        ao["ao_coulomb"] = raw
        ao["metric_whitener"] = whitener
        jkfit = None

    def make_bars():
        d = {
            "coefficients_alpha": np.zeros_like(frames[0]),
            "coefficients_beta": np.zeros_like(frames[1]),
            "hcore_ao": np.zeros_like(h),
            "metric/coulomb": np.zeros_like(whitener),
        }
        return d

    # TensorIR reference
    monkeypatch.delenv("GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_DISABLE_AOT_EXTFOCK", "1")
    fock_ref, pb_ref = extended_fock_response(space, ao, jkfit, CountingRunner())
    bars_ref = make_bars()
    pb_ref(seeds, bars_ref)

    # Native AOT discovered via env var + strict enforcement
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_EXTFOCK", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK", "1")
    monkeypatch.setenv("GENERATIVEQC_F12_EXT_FOCK_LIBRARY", str(host_ext_fock_library))
    nat_runner = CountingRunner()
    fock_nat, pb_nat = extended_fock_response(space, ao, jkfit, nat_runner)
    bars_nat = make_bars()
    pb_nat(seeds, bars_nat)
    assert nat_runner.calls == 0

    for k in fock_ref:
        np.testing.assert_allclose(fock_nat[k], fock_ref[k], atol=1e-12, rtol=1e-12)
    assert bars_nat.keys() == bars_ref.keys()
    for k in bars_ref:
        np.testing.assert_allclose(bars_nat[k], bars_ref[k], atol=1e-12, rtol=1e-12)

    # physical_fock wrapper with CDLL passed directly via library parameter
    cdll = ctypes.CDLL(str(host_ext_fock_library))
    nat_runner2 = CountingRunner()
    mats_pf, pb_pf = physical_fock(
        h, raw, whitener, frames, (oa, ob), nat_runner2, False, cdll
    )
    pf_bars = pb_pf([seeds["fock_alpha"], seeds["fock_beta"]])
    assert nat_runner2.calls == 0
    np.testing.assert_allclose(mats_pf[0], fock_ref["fock_alpha"], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(mats_pf[1], fock_ref["fock_beta"], atol=1e-12, rtol=0.0)
    assert pf_bars[0][0].shape == frames[0].shape


def test_require_aot_extfock_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h, raw, whitener, frames, occupied = _make_problem(
        101, n=5, ext=4, q=7, oa=2, ob=2, same_occ_frame=True
    )
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK", "1")
    for key in (
        "GENERATIVEQC_F12_EXT_FOCK_LIBRARY",
        "GENERATIVEQC_F12_HOST_SINK",
        "GENERATIVEQC_F12_FOCAL_CUDA_LIBRARY",
        "GENERATIVEQC_F12_SHELL_LIBRARY",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError, match="GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK=1"):
        physical_spin_operators(
            h, raw, whitener, frames, occupied, CountingRunner(), False, None
        )


def test_finite_difference_oracle_with_nonsymmetric_seeds(
    host_ext_fock_library: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent central-difference check against analytical pullback for all inputs."""
    n, ext, q, oa, ob = 6, 5, 9, 2, 1
    h, raw, whitener, frames, occupied = _make_problem(
        777, n=n, ext=ext, q=q, oa=oa, ob=ob, same_occ_frame=False
    )
    # Build SPD Coulomb metric J = (W^T W)^{-1} and derive exact Cholesky whitener.
    metric = np.linalg.inv(whitener.T @ whitener)
    metric = 0.5 * (metric + metric.T)
    w_chol = np.linalg.inv(np.linalg.cholesky(metric))  # w_chol.T @ w_chol == metric^{-1}

    rng = np.random.default_rng(778)
    seeds = {
        "fock_alpha": rng.normal(size=(ext, ext)),
        "fock_beta": rng.normal(size=(ext, ext)),
        "exchange_alpha": rng.normal(size=(ext, ext)),
        "exchange_beta": rng.normal(size=(ext, ext)),
    }

    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_EXTFOCK", "1")
    runner = CountingRunner()

    def eval_loss(h_in, raw_in, metric_in, c0_in, c1_in):
        w_in = np.linalg.inv(np.linalg.cholesky(0.5 * (metric_in + metric_in.T)))
        mats, _ = physical_spin_operators(
            h_in,
            raw_in,
            w_in,
            [c0_in, c1_in],
            occupied,
            runner,
            False,
            None,
            include_exchange=True,
            native_library=host_ext_fock_library,
            host_reference=True,
        )
        return sum(float(np.sum(seeds[k] * mats[k])) for k in seeds)

    _, pb = physical_spin_operators(
        h,
        raw,
        w_chol,
        frames,
        occupied,
        runner,
        False,
        None,
        include_exchange=True,
        native_library=host_ext_fock_library,
        host_reference=True,
    )
    coeff_bars, bar_h, bar_raw, bar_metric = pb(seeds)
    assert runner.calls == 0

    eps = 1e-6
    dh = rng.normal(size=h.shape)
    dh = 0.5 * (dh + dh.T)
    draw = rng.normal(size=raw.shape)
    draw = 0.5 * (draw + draw.swapaxes(1, 2))
    dm = rng.normal(size=metric.shape)
    dm = 0.5 * (dm + dm.T)
    dc0 = rng.normal(size=frames[0].shape)
    dc1 = rng.normal(size=frames[1].shape)

    fd = (
        eval_loss(h + eps * dh, raw + eps * draw, metric + eps * dm, frames[0] + eps * dc0, frames[1] + eps * dc1)
        - eval_loss(h - eps * dh, raw - eps * draw, metric - eps * dm, frames[0] - eps * dc0, frames[1] - eps * dc1)
    ) / (2.0 * eps)
    analytic = float(
        np.sum(bar_h * dh)
        + np.sum(bar_raw * draw)
        + np.sum(bar_metric * dm)
        + np.sum(coeff_bars[0] * dc0)
        + np.sum(coeff_bars[1] * dc1)
    )
    assert abs(analytic - fd) < 1e-8 * max(1.0, abs(fd))


def test_j_rev_with_bar_raw_accumulation(host_ext_fock_library: Path) -> None:
    native = bind_ext_fock_native(host_ext_fock_library, host_reference=True)
    assert native is not None
    rng = np.random.default_rng(314)
    qt, n = 7, 5
    raw_tile = rng.normal(size=(qt, n, n))
    rho_tile = rng.normal(size=qt)
    bar_g = rng.normal(size=(n, n))
    bar_raw = rng.normal(size=(qt, n, n))
    expected_bar_raw = bar_raw + np.einsum("a,mn->amn", rho_tile, bar_g)
    expected_bar_rho = np.einsum("amn,mn->a", raw_tile, bar_g)

    bar_rho = native.j_rev(raw_tile, bar_g, rho_tile=rho_tile, bar_raw_tile=bar_raw)
    np.testing.assert_allclose(bar_rho, expected_bar_rho, atol=1e-13, rtol=0.0)
    np.testing.assert_allclose(bar_raw, expected_bar_raw, atol=1e-13, rtol=0.0)
