"""Gate 1 qualification tests for Patch A-1a (AOT one-electron gradient & spin Fock)."""

from __future__ import annotations

import ctypes as ct
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest
from generativeqc._f12_focal_one_electron import FocalOneElectron
from generativeqc._f12_spin_fock import spin_fock, spin_fock_pullback
from generativeqc._f12_spin_fock_1e_native import (
    bind_spin_fock_1e_ops,
    run_one_electron_grad,
    run_spin_fock_fwd,
)
from generativeqc_compiler.method.nuclear_integral_response import (
    overlap_hcore_gradient,
)
from generativeqc_compiler.tensor import execute

REPO = Path(__file__).resolve().parents[2]
CU_SOURCE = REPO / "src/posthf/f12/f12_spin_fock_1e_cuda.cu"


def _has_cuda_device() -> bool:
    try:
        rt = ct.CDLL("libcudart.so")
        cnt = ct.c_int(0)
        return rt.cudaGetDeviceCount(ct.byref(cnt)) == 0 and cnt.value > 0
    except OSError:
        return False


@pytest.fixture(scope="module")
def host_lib_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cxx = shutil.which("g++") or shutil.which("c++")
    assert cxx is not None, "C++ compiler required to build host reference test library"
    out = tmp_path_factory.mktemp("patch_a1a_host") / "libtest_spinfock1e_host.so"
    subprocess.run(
        [
            cxx,
            "-std=c++20",
            "-O3",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-shared",
            "-x",
            "c++",
            str(CU_SOURCE),
            "-o",
            str(out),
        ],
        check=True,
    )
    return out


@pytest.fixture(scope="module")
def cuda_lib_path(tmp_path_factory: pytest.TempPathFactory) -> Path | None:
    nvcc = shutil.which("nvcc")
    if nvcc is None or not _has_cuda_device():
        return None
    out = tmp_path_factory.mktemp("patch_a1a_cuda") / "libtest_spinfock1e_cuda.so"
    cmd = [
        nvcc,
        "-std=c++20",
        "-O3",
        "-arch=sm_89",
        "-Xcompiler",
        "-fPIC,-Wall,-Wextra",
        "-shared",
        str(CU_SOURCE),
        "-lcublas",
        "-o",
        str(out),
    ]
    sccache = shutil.which("sccache")
    if sccache is not None:
        # Compile object via sccache, then link shared library.
        obj = out.with_suffix(".o")
        subprocess.run(
            [
                sccache,
                nvcc,
                "-std=c++20",
                "-O3",
                "-arch=sm_89",
                "-Xcompiler",
                "-fPIC,-Wall,-Wextra",
                "-c",
                str(CU_SOURCE),
                "-o",
                str(obj),
            ],
            check=True,
        )
        subprocess.run(
            [nvcc, "-arch=sm_89", "-shared", str(obj), "-lcublas", "-o", str(out)],
            check=True,
        )
    else:
        subprocess.run(cmd, check=True)
    return out


def _counting_runner(*, cuda: bool = False) -> tuple[Callable, list[int]]:
    calls = [0]

    def run(program, inputs):
        calls[0] += 1
        return execute(program, inputs)

    run.host_reference = not cuda  # type: ignore[attr-defined]
    return run, calls


def _make_sources(
    rng: np.random.Generator,
    oa: int,
    ob: int,
    n: int,
    q: int,
    *,
    closed_shell_alias: bool = False,
    closed_shell_copy: bool = False,
    distinct_hcore: bool = False,
) -> dict[str, np.ndarray]:
    hcore_a = rng.standard_normal((n, n))
    raw_a = rng.standard_normal((q, n, n))
    sources: dict[str, np.ndarray] = {
        "hcore_alpha": hcore_a,
        "raw_coulomb_alpha": raw_a,
    }
    if oa > 0:
        sources["fit_alpha"] = rng.standard_normal((q, oa, n))

    if closed_shell_alias:
        assert oa == ob
        sources["hcore_beta"] = rng.standard_normal((n, n)) if distinct_hcore else hcore_a
        sources["raw_coulomb_beta"] = raw_a
        if ob > 0:
            sources["fit_beta"] = sources["fit_alpha"]
    elif closed_shell_copy:
        assert oa == ob
        sources["hcore_beta"] = (
            rng.standard_normal((n, n)) if distinct_hcore else hcore_a.copy()
        )
        sources["raw_coulomb_beta"] = raw_a.copy()
        if ob > 0:
            sources["fit_beta"] = sources["fit_alpha"].copy()
    else:
        sources["hcore_beta"] = rng.standard_normal((n, n))
        sources["raw_coulomb_beta"] = rng.standard_normal((q, n, n))
        if ob > 0:
            sources["fit_beta"] = rng.standard_normal((q, ob, n))
    return sources


def _tensorir_spin_fock(
    monkeypatch: pytest.MonkeyPatch,
    space: tuple[int, int, int, int],
    sources: dict[str, np.ndarray],
    batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    with monkeypatch.context() as mp:
        mp.setenv("GENERATIVEQC_F12_DISABLE_AOT_SPINFOCK_1E", "1")
        mp.delenv("GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E", raising=False)
        return spin_fock(space, sources, execute, batch_size=batch_size)


def _tensorir_spin_fock_pullback(
    monkeypatch: pytest.MonkeyPatch,
    space: tuple[int, int, int, int],
    sources: dict[str, np.ndarray],
    seeds: dict[str, np.ndarray],
    batch_size: int | None = None,
    accumulate: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    with monkeypatch.context() as mp:
        mp.setenv("GENERATIVEQC_F12_DISABLE_AOT_SPINFOCK_1E", "1")
        mp.delenv("GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E", raising=False)
        return spin_fock_pullback(
            space, sources, seeds, execute, batch_size=batch_size, accumulate=accumulate
        )


@pytest.mark.parametrize("natoms,n", [(1, 5), (4, 13), (9, 24)])
def test_one_electron_grad_host_parity_vs_tensorir(
    host_lib_path: Path, natoms: int, n: int
) -> None:
    rng = np.random.default_rng(1000 + natoms * 37 + n)
    d_overlap = rng.standard_normal((natoms, 3, n, n))
    d_hcore = rng.standard_normal((natoms, 3, n, n))
    ov_w = rng.standard_normal((n, n))
    hc_w = rng.standard_normal((n, n))

    graph = overlap_hcore_gradient(natoms, n)
    ref = execute(
        graph,
        {
            "d_overlap": d_overlap,
            "d_hcore": d_hcore,
            "overlap": ov_w,
            "hcore": hc_w,
        },
    ).outputs["gradient"]

    ops = bind_spin_fock_1e_ops(host_lib_path, host_reference=True)
    assert ops is not None
    got = run_one_electron_grad(ops, d_overlap, d_hcore, ov_w, hc_w)
    assert got.shape == (natoms, 3)
    assert got.dtype == np.float64
    max_err = float(np.max(np.abs(got - ref)))
    assert max_err < 1e-13, f"one_electron_grad host max_err={max_err:.3e}"


def test_focal_one_electron_field_compatibility_and_require_aot(
    host_lib_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(42)
    natoms, n = 3, 10
    ov = rng.standard_normal((n, n))
    hc = rng.standard_normal((n, n))
    d_ov = rng.standard_normal((natoms, 3, n, n))
    d_hc = rng.standard_normal((natoms, 3, n, n))
    ov_w = rng.standard_normal((n, n))
    hc_w = rng.standard_normal((n, n))

    # Positional construction without native_library_path remains 100% compatible.
    one_e = FocalOneElectron(ov, hc, d_ov, d_hc, 4, "s", "b", "cpu_reference", 0)
    assert one_e.native_library_path is None
    ref = one_e.nuclear_gradient(ov_w, hc_w, run=None)

    # run=None on cuda still raises ValueError.
    cuda_one_e = replace(one_e, executed_backend="cuda")
    with pytest.raises(ValueError, match="graph runner"):
        cuda_one_e.nuclear_gradient(ov_w, hc_w, run=None)

    # End-to-end FocalOneElectron.nuclear_gradient AOT dispatch on executed_backend="cuda"
    # achieves < 1e-13 parity with 0 TensorIR calls.
    host_ops = bind_spin_fock_1e_ops(host_lib_path, host_reference=True)
    assert host_ops is not None
    with monkeypatch.context() as mp:
        mp.setattr(
            "generativeqc._f12_spin_fock_1e_native.resolve_spin_fock_1e_ops",
            lambda run, *, library=None, host_reference=None: host_ops,
        )
        run, calls = _counting_runner(cuda=True)
        got = replace(
            cuda_one_e, native_library_path=str(host_lib_path)
        ).nuclear_gradient(ov_w, hc_w, run=run)
        assert calls[0] == 0
        assert float(np.max(np.abs(got - ref))) < 1e-13

    # REQUIRE_AOT forbids TensorIR fallback when no CUDA AOT library is available.
    monkeypatch.delenv("GENERATIVEQC_F12_SHELL_LIBRARY", raising=False)
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_SPINFOCK_1E", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E", "1")
    monkeypatch.setattr(
        "generativeqc._f12_spin_fock_1e_native._REGISTERED_LIBRARIES", []
    )
    monkeypatch.setattr(
        "generativeqc._f12_spin_fock_1e_native._loaded_sink_paths", lambda: []
    )
    run, calls = _counting_runner(cuda=True)
    with pytest.raises(RuntimeError, match="GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E=1"):
        cuda_one_e.nuclear_gradient(ov_w, hc_w, run=run)
    assert calls[0] == 0


def test_pair_force_contracts_one_electron_weights_with_provider_shell(
    host_lib_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc import _f12_focal_molecular as molecular
    from generativeqc import _f12_spin_fock_1e_native as native
    from numpy.linalg import _umath_linalg

    # Model the sealed source DSO with any loadable binary lacking the response ABI.
    source_path = Path(_umath_linalg.__file__)
    assert bind_spin_fock_1e_ops(source_path, host_reference=True) is None
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E", "1")
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_SPINFOCK_1E", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_SHELL_LIBRARY", str(host_lib_path))

    # Exercise the CUDA routing decision with the compiled host-reference kernels.
    resolve = native.resolve_spin_fock_1e_ops

    def resolve_on_host(run, *, library=None, host_reference=None):
        assert host_reference is False
        return resolve(run, library=library, host_reference=True)

    monkeypatch.setattr(native, "resolve_spin_fock_1e_ops", resolve_on_host)
    run, calls = _counting_runner(cuda=True)
    provider = molecular.MolecularFocalProvider(
        shell_library=host_lib_path, one_electron_library=source_path, run=run
    )
    rng = np.random.default_rng(621)
    d_overlap, d_hcore = rng.normal(size=(2, 1, 3, 2, 2))
    ow, hw = rng.normal(size=(2, 2, 2))
    one = FocalOneElectron(
        np.eye(2), np.eye(2), d_overlap, d_hcore, 1, "s", "b", "cuda", 0,
        native_library_path=str(source_path),
    )
    reference = np.einsum("axij,ij->ax", d_overlap, ow)
    reference += np.einsum("axij,ij->ax", d_hcore, hw)

    # Supply precomputed energy responses and isolate their one-electron force.
    ordinary = SimpleNamespace(
        spin_energies=(0., 0., 0.), fock_seeds={}, obs_coefficients=None,
        mp2_metric=None, mp2_three_center=None,
    )
    increment = SimpleNamespace(
        energy=0., cabs_rank=0, one_electron_weights={"overlap": ow, "hcore": hw},
        jkfit_weights={}, f12fit_weights={},
    )
    monkeypatch.setattr(molecular, "ordinary_df_ump2_unrelaxed", lambda *a, **kw: ordinary)
    monkeypatch.setattr(molecular, "reference_f12_increment", lambda *a, **kw: increment)
    monkeypatch.setattr(
        molecular.MolecularFocalProvider, "_gradient", lambda *a, **kw: np.zeros((1, 3))
    )
    panels = tuple(
        SimpleNamespace(streaming=False, obs=lambda *a: None, joint=lambda *a, **kw: None)
        for _ in range(3)
    )
    *_, forces = provider._pair_components(
        SimpleNamespace(backend="cuda", correlation_space_id="all-active"),
        None,
        one,
        panels,
        expected_cabs_rank=0,
    )
    np.testing.assert_allclose(forces, -reference, atol=1e-13, rtol=0)

    # Direct callers can keep a unified source/response DSO. A bad explicit
    # response must fail even when both the source and the environment are valid.
    unified = replace(one, native_library_path=str(host_lib_path))
    np.testing.assert_allclose(
        unified.nuclear_gradient(ow, hw, run=run), reference, atol=1e-13, rtol=0
    )
    with pytest.raises(RuntimeError, match="GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E"):
        unified.nuclear_gradient(ow, hw, run=run, shell_library=source_path)
    assert calls[0] == 0


@pytest.mark.parametrize(
    "case_name,oa,ob,n,q,batch_size,kwargs",
    [
        ("rhf_alias", 3, 3, 10, 11, None, {"closed_shell_alias": True}),
        ("rhf_copy_multitile", 4, 4, 12, 13, 4, {"closed_shell_copy": True}),
        (
            "rhf_distinct_hcore",
            3,
            3,
            9,
            10,
            3,
            {"closed_shell_alias": True, "distinct_hcore": True},
        ),
        ("uhf_general", 4, 2, 11, 13, None, {}),
        ("uhf_multitile", 3, 5, 12, 17, 5, {}),
        ("oa_zero_ob_pos", 0, 3, 8, 9, 4, {}),
        ("oa_pos_ob_zero", 3, 0, 8, 9, 4, {}),
        ("q_tile_one", 2, 1, 7, 5, 1, {}),
    ],
)
def test_spin_fock_fwd_host_parity_and_zero_tensorir(
    host_lib_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    oa: int,
    ob: int,
    n: int,
    q: int,
    batch_size: int | None,
    kwargs: dict,
) -> None:
    rng = np.random.default_rng(abs(hash(case_name)) % (2**31))
    space = (oa, ob, n, q)
    sources = _make_sources(rng, oa, ob, n, q, **kwargs)
    ref = _tensorir_spin_fock(monkeypatch, space, sources, batch_size=batch_size)

    monkeypatch.setenv("GENERATIVEQC_F12_SHELL_LIBRARY", str(host_lib_path))
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E", "1")
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_SPINFOCK_1E", raising=False)

    run, calls = _counting_runner(cuda=False)
    got = spin_fock(space, sources, run, batch_size=batch_size)

    assert calls[0] == 0, f"Expected 0 TensorIR calls for {case_name}, got {calls[0]}"
    assert set(got.keys()) == set(ref.keys())
    for key in ref:
        max_err = float(np.max(np.abs(got[key] - ref[key])))
        assert max_err < 1e-13, f"{case_name} {key} max_err={max_err:.3e}"


def test_spin_fock_fwd_optional_hcore(
    host_lib_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(777)
    space = (3, 2, 9, 10)
    sources = _make_sources(rng, *space)
    sources_zero_hc = dict(sources)
    sources_zero_hc["hcore_alpha"] = np.zeros((9, 9), dtype=np.float64)
    sources_zero_hc["hcore_beta"] = np.zeros((9, 9), dtype=np.float64)
    ref = _tensorir_spin_fock(monkeypatch, space, sources_zero_hc)

    ops = bind_spin_fock_1e_ops(host_lib_path, host_reference=True)
    assert ops is not None
    sources_no_hc = {
        k: v for k, v in sources.items() if not k.startswith("hcore_")
    }
    got = run_spin_fock_fwd(ops, space, sources_no_hc, batch_size=3)
    for key in ref:
        assert float(np.max(np.abs(got[key] - ref[key]))) < 1e-13


@pytest.mark.parametrize(
    "case_name,oa,ob,n,q,batch_size,src_kwargs,seed_keys,sym_seeds,use_accumulate",
    [
        (
            "rhf_share_beta_all_seeds",
            3,
            3,
            10,
            11,
            4,
            {"closed_shell_alias": True},
            ("fock_alpha", "fock_beta", "exchange_alpha", "exchange_beta"),
            True,
            False,
        ),
        (
            "rhf_share_beta_fock_only",
            4,
            4,
            9,
            10,
            3,
            {"closed_shell_copy": True},
            ("fock_alpha", "fock_beta"),
            True,
            False,
        ),
        (
            "rhf_asym_seeds_no_share",
            3,
            3,
            10,
            9,
            4,
            {"closed_shell_alias": True},
            ("fock_alpha", "fock_beta", "exchange_alpha", "exchange_beta"),
            False,
            False,
        ),
        (
            "rhf_with_accumulate",
            3,
            3,
            9,
            11,
            3,
            {"closed_shell_alias": True},
            ("fock_alpha", "fock_beta", "exchange_alpha", "exchange_beta"),
            True,
            True,
        ),
        (
            "uhf_all_seeds_multitile",
            4,
            2,
            11,
            13,
            4,
            {},
            ("fock_alpha", "fock_beta", "exchange_alpha", "exchange_beta"),
            False,
            False,
        ),
        (
            "uhf_accumulate_all_seeds",
            3,
            5,
            10,
            11,
            3,
            {},
            ("fock_alpha", "fock_beta", "exchange_alpha", "exchange_beta"),
            False,
            True,
        ),
        (
            "uhf_only_fock_alpha",
            3,
            2,
            9,
            8,
            3,
            {},
            ("fock_alpha",),
            False,
            False,
        ),
        (
            "uhf_only_fock_beta",
            3,
            2,
            9,
            8,
            3,
            {},
            ("fock_beta",),
            False,
            False,
        ),
        (
            "uhf_only_exchange_alpha",
            3,
            2,
            9,
            8,
            3,
            {},
            ("exchange_alpha",),
            False,
            False,
        ),
        (
            "uhf_mixed_fock_a_exch_b",
            3,
            2,
            9,
            10,
            4,
            {},
            ("fock_alpha", "exchange_beta"),
            False,
            True,
        ),
        (
            "oa_zero_ob_pos_all",
            0,
            3,
            8,
            9,
            4,
            {},
            ("fock_alpha", "fock_beta", "exchange_alpha", "exchange_beta"),
            False,
            False,
        ),
        (
            "oa_pos_ob_zero_all",
            3,
            0,
            8,
            9,
            4,
            {},
            ("fock_alpha", "fock_beta", "exchange_alpha", "exchange_beta"),
            False,
            True,
        ),
    ],
)
def test_spin_fock_rev_host_parity_and_zero_tensorir(
    host_lib_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    oa: int,
    ob: int,
    n: int,
    q: int,
    batch_size: int | None,
    src_kwargs: dict,
    seed_keys: tuple[str, ...],
    sym_seeds: bool,
    use_accumulate: bool,
) -> None:
    rng = np.random.default_rng(abs(hash(case_name)) % (2**31))
    space = (oa, ob, n, q)
    sources = _make_sources(rng, oa, ob, n, q, **src_kwargs)

    # Deliberately asymmetric seeds to catch transpose mistakes on bar_K_eff.
    seeds: dict[str, np.ndarray] = {}
    for key in seed_keys:
        if sym_seeds and key.endswith("_beta"):
            seeds[key] = seeds[key.replace("_beta", "_alpha")].copy()
        else:
            seeds[key] = rng.standard_normal((n, n))

    if use_accumulate:
        acc_ref = {k: rng.standard_normal(v.shape) for k, v in sources.items()}
        acc_got = {k: v.copy() for k, v in acc_ref.items()}
    else:
        acc_ref = None
        acc_got = None

    ref = _tensorir_spin_fock_pullback(
        monkeypatch, space, sources, seeds, batch_size=batch_size, accumulate=acc_ref
    )

    monkeypatch.setenv("GENERATIVEQC_F12_SHELL_LIBRARY", str(host_lib_path))
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E", "1")
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_SPINFOCK_1E", raising=False)

    run, calls = _counting_runner(cuda=False)
    got = spin_fock_pullback(
        space, sources, seeds, run, batch_size=batch_size, accumulate=acc_got
    )

    assert calls[0] == 0, f"Expected 0 TensorIR calls for {case_name}, got {calls[0]}"
    assert set(got.keys()) == set(ref.keys())
    if sym_seeds and not use_accumulate and oa == ob:
        assert got["raw_coulomb_beta"] is got["raw_coulomb_alpha"]
    if use_accumulate:
        assert got is acc_got
    for key in ref:
        max_err = float(np.max(np.abs(got[key] - ref[key])))
        assert max_err < 1e-13, f"{case_name} {key} max_err={max_err:.3e}"


def test_spin_fock_require_aot_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(123)
    space = (2, 2, 6, 4)
    sources = _make_sources(rng, *space)
    seeds = {"fock_alpha": rng.standard_normal((6, 6))}

    monkeypatch.delenv("GENERATIVEQC_F12_SHELL_LIBRARY", raising=False)
    monkeypatch.delenv("GENERATIVEQC_F12_HOST_SINK", raising=False)
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_SPINFOCK_1E", raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E", "1")
    monkeypatch.setattr(
        "generativeqc._f12_spin_fock_1e_native._REGISTERED_LIBRARIES", []
    )
    monkeypatch.setattr(
        "generativeqc._f12_spin_fock_1e_native._loaded_sink_paths", lambda: []
    )

    run, calls = _counting_runner(cuda=False)
    with pytest.raises(RuntimeError, match="GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E=1"):
        spin_fock(space, sources, run)
    with pytest.raises(RuntimeError, match="GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E=1"):
        spin_fock_pullback(space, sources, seeds, run)
    assert calls[0] == 0


def test_cuda_real_device_parity_and_zero_tensorir(
    cuda_lib_path: Path | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if cuda_lib_path is None:
        pytest.skip("CUDA device or nvcc not available in this environment")

    rng = np.random.default_rng(20261124)

    # 1. FocalOneElectron.nuclear_gradient on CUDA with 0 TensorIR calls.
    natoms, n = 5, 16
    ov = rng.standard_normal((n, n))
    hc = rng.standard_normal((n, n))
    d_ov = rng.standard_normal((natoms, 3, n, n))
    d_hc = rng.standard_normal((natoms, 3, n, n))
    ov_w = rng.standard_normal((n, n))
    hc_w = rng.standard_normal((n, n))

    ref_1e = execute(
        overlap_hcore_gradient(natoms, n),
        {"d_overlap": d_ov, "d_hcore": d_hc, "overlap": ov_w, "hcore": hc_w},
    ).outputs["gradient"]

    one_e_cuda = FocalOneElectron(
        ov,
        hc,
        d_ov,
        d_hc,
        10,
        "s",
        "b",
        "cuda",
        0,
        native_library_path=str(cuda_lib_path),
    )
    monkeypatch.setenv("GENERATIVEQC_F12_REQUIRE_AOT_SPINFOCK_1E", "1")
    monkeypatch.delenv("GENERATIVEQC_F12_DISABLE_AOT_SPINFOCK_1E", raising=False)
    run_cuda, calls = _counting_runner(cuda=True)
    got_1e = one_e_cuda.nuclear_gradient(ov_w, hc_w, run=run_cuda)
    assert calls[0] == 0
    assert float(np.max(np.abs(got_1e - ref_1e))) < 1e-13

    # 2. spin_fock + spin_fock_pullback on CUDA across RHF, UHF, oa=0, multi-tile, accumulate.
    monkeypatch.setenv("GENERATIVEQC_F12_SHELL_LIBRARY", str(cuda_lib_path))
    for oa, ob, m, q, bs, src_kw, sym, acc in [
        (4, 4, 14, 15, 5, {"closed_shell_alias": True}, True, False),
        (4, 2, 13, 17, 6, {}, False, False),
        (4, 2, 13, 17, 6, {}, False, True),
        (0, 3, 10, 11, 4, {}, False, False),
    ]:
        space = (oa, ob, m, q)
        sources = _make_sources(rng, oa, ob, m, q, **src_kw)
        ref_fwd = _tensorir_spin_fock(monkeypatch, space, sources, batch_size=bs)
        got_fwd = spin_fock(space, sources, run_cuda, batch_size=bs)
        assert calls[0] == 0
        for key in ref_fwd:
            assert float(np.max(np.abs(got_fwd[key] - ref_fwd[key]))) < 1e-13

        seeds = {
            "fock_alpha": rng.standard_normal((m, m)),
            "exchange_alpha": rng.standard_normal((m, m)),
        }
        if sym:
            seeds["fock_beta"] = seeds["fock_alpha"].copy()
            seeds["exchange_beta"] = seeds["exchange_alpha"].copy()
        else:
            seeds["fock_beta"] = rng.standard_normal((m, m))
            seeds["exchange_beta"] = rng.standard_normal((m, m))

        acc_ref = {k: rng.standard_normal(v.shape) for k, v in sources.items()} if acc else None
        acc_got = {k: v.copy() for k, v in acc_ref.items()} if acc_ref is not None else None

        ref_rev = _tensorir_spin_fock_pullback(
            monkeypatch, space, sources, seeds, batch_size=bs, accumulate=acc_ref
        )
        got_rev = spin_fock_pullback(
            space, sources, seeds, run_cuda, batch_size=bs, accumulate=acc_got
        )
        assert calls[0] == 0
        if sym and not acc:
            assert got_rev["raw_coulomb_beta"] is got_rev["raw_coulomb_alpha"]
        for key in ref_rev:
            assert float(np.max(np.abs(got_rev[key] - ref_rev[key]))) < 1e-13
