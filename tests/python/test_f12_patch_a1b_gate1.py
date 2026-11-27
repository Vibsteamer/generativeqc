"""Gate-1 verification for Patch A-1b Steps 1 & 2 (all 5 AOT CPHF relax primitives)."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.common.compiler_cache import (
    resolve_compiler_cache,
    run_cached_compiler,
)
from generativeqc_compiler.method.canonical_frame_response import (
    orbital_metric_cotangent,
    orbital_rotation_cotangent,
)
from generativeqc_compiler.mp2.f12_unrestricted_reference import (
    unrestricted_df_canonical_driver,
    unrestricted_df_orbital_action,
)
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    execute,
    input_tensor,
)

from generativeqc import _f12_cphf_relax_native as cphf_native
from generativeqc._f12_cphf_relax_native import (
    ENV_CPHF_LIBRARY,
    ENV_DISABLE_CPHF,
    ENV_REQUIRE_CPHF,
    ENV_REQUIRE_STEP1,
    ENV_SHELL_LIBRARY,
    ENV_SINK_LIBRARY,
    bind_cphf_relax,
    run_cphf_action_native,
    run_df_canonical_driver_native,
    run_df_nuclear_weights_native,
    run_orbital_rot_metric_native,
    run_whiten_3c_native,
)
from generativeqc._f12_focal_singles_response import (
    _rhf_df_orbital_action,
    relaxed_reference_weights,
)
from generativeqc.f12_orbital_response import (
    UnrestrictedOrbitalForces,
    unrestricted_canonical_source_seeds,
    unrestricted_frame_overlap_weights,
    unrestricted_nuclear_source_weights,
    unrestricted_source_orbital_adjoint,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO = Path(__file__).resolve().parents[2]
CU_SOURCE = REPO / "src/posthf/f12/f12_cphf_relax_cuda.cu"


@pytest.fixture(scope="session")
def host_cphf_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cxx = shutil.which("c++")
    if cxx is None:
        pytest.skip("C++ compiler is required for host-reference AOT test")
    launcher = resolve_compiler_cache()
    out_dir = tmp_path_factory.mktemp("patch_a1b_host")
    lib_path = out_dir / "libf12_cphf_relax_host.so"
    argv = [
        cxx,
        "-std=c++20",
        "-O3",
        "-x",
        "c++",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-DGQC_SINK_HOST_REFERENCE",
        "-shared",
        str(CU_SOURCE),
        "-o",
        str(lib_path),
    ]
    ret = run_cached_compiler(argv, 300, label="patch-a1b host ref", launcher=launcher)
    assert ret.returncode == 0, f"Host compilation failed:\n{ret.stdout}\n{ret.stderr}"
    return lib_path


def _counting_runner():
    calls = {"count": 0}

    def run(program: Program, inputs: dict[str, np.ndarray]):
        calls["count"] += 1
        return execute(program, inputs)

    return run, calls


def _forbidding_runner():
    calls = {"count": 0}

    def run(program: Program, inputs: dict[str, np.ndarray]):
        calls["count"] += 1
        raise AssertionError(
            f"Unexpected TensorIR execution (call #{calls['count']}): {program.provenance}"
        )

    return run, calls


@contextlib.contextmanager
def _tensorir_oracle(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Without an explicit library the binder also accepts focal shell libraries
    # that earlier tests loaded into this process, which would replace the oracle
    # with the AOT kernels under test.
    with monkeypatch.context() as patch:
        patch.setenv(ENV_DISABLE_CPHF, "1")
        yield


def _random_sym(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.standard_normal((n, n))
    return 0.5 * (a + a.T)


def _make_gauge_invariant_bar(
    rng: np.random.Generator, coeff: np.ndarray, occ: int
) -> np.ndarray:
    """Build frame_bar = (C^-T) M where M_oo and M_vv are symmetric (zero gauge torque)."""
    obs = coeff.shape[1]
    m = np.zeros((obs, obs), dtype=np.float64)
    m[:occ, :occ] = _random_sym(rng, occ)
    m[occ:, occ:] = _random_sym(rng, obs - occ)
    m[:occ, occ:] = rng.standard_normal((occ, obs - occ))
    m[occ:, :occ] = rng.standard_normal((obs - occ, occ))
    # C^T bar = M => bar = (C^T)^-1 M
    return np.linalg.solve(coeff.T, m)


def _build_synthetic_cphf_inputs(
    rng: np.random.Generator,
    occs: tuple[int, ...],
    obs: int,
    aux: int,
    *,
    scale_df: float = 0.08,
):
    """Create a well-conditioned synthetic DF-HF problem for relaxed_reference_weights."""
    is_rhf = len(occs) == 1
    n_spin_arrays = 1 if is_rhf else 2
    coeffs = []
    energies = []
    bars = []
    for k in range(n_spin_arrays):
        o = occs[k]
        q_mat, _ = np.linalg.qr(rng.standard_normal((obs, obs)))
        coeffs.append(q_mat)
        # Positive, well-separated virtual - occupied gaps for guaranteed SPD Hessian.
        occ_e = np.sort(rng.uniform(-2.0, -0.8, size=o))
        virt_e = np.sort(rng.uniform(0.8, 2.5, size=obs - o))
        energies.append(np.concatenate([occ_e, virt_e]))
        bars.append(_make_gauge_invariant_bar(rng, q_mat, o))

    whitener = rng.standard_normal((aux, aux)) * 0.2
    raw = {}
    if is_rhf:
        raw_sym = rng.standard_normal((aux, obs, obs)) * scale_df
        raw_sym = 0.5 * (raw_sym + np.swapaxes(raw_sym, 1, 2))
        raw["raw_coulomb_alpha"] = raw_sym
        raw["raw_coulomb_beta"] = raw_sym
    else:
        for spin in ("alpha", "beta"):
            raw_s = rng.standard_normal((aux, obs, obs)) * scale_df
            raw_s = 0.5 * (raw_s + np.swapaxes(raw_s, 1, 2))
            raw["raw_coulomb_" + spin] = raw_s

    reference = types.SimpleNamespace(
        occupied=occs,
        executed_backend="cpu",
        overlap=np.eye(obs, dtype=np.float64),
        coefficients=np.stack(coeffs, axis=0),
        orbital_energies=np.stack(energies, axis=0),
    )
    weights = {
        "obs_coefficients": np.stack(bars, axis=0),
        "hcore": np.zeros((obs, obs), dtype=np.float64),
        "three_center": np.zeros((aux, obs, obs), dtype=np.float64),
        "metric": np.zeros((aux, aux), dtype=np.float64),
    }
    overlap_seed = _random_sym(rng, obs) * 0.1
    return reference, raw, whitener, weights, overlap_seed


@pytest.mark.parametrize(
    ("aux_out", "aux_in", "obs", "budget_bytes"),
    [
        (11, 11, 7, 512 << 20),
        (17, 17, 9, 4096),  # forced multi-tile (a, b)
        (13, 19, 8, 4096),  # rectangular whitener + forced multi-tile
        (9, 11, 15, 4096),  # forced 3D multi-tile (a, b, AND c columns)
    ],
)
def test_whiten_3c_parity_vs_tensorir(
    host_cphf_library: Path, aux_out: int, aux_in: int, obs: int, budget_bytes: int
):
    rng = np.random.default_rng(101 + aux_out * 7 + aux_in)
    whitener = rng.standard_normal((aux_out, aux_in)) * 0.35
    raw = rng.standard_normal((aux_in, obs, obs)) * 0.35

    a_idx = Index("A", IndexSpace("aux_out", "auxiliary", aux_out))
    b_idx = Index("B", IndexSpace("aux_in", "auxiliary", aux_in))
    mo = IndexSpace("obs", "orbital", obs)
    p_idx, q_idx = Index("p", mo), Index("q", mo)
    prog = Program(
        {
            "df": einsum(
                "AB,Bpq->Apq",
                input_tensor("W", TensorSpec((a_idx, b_idx), role="input")),
                input_tensor("raw", TensorSpec((b_idx, p_idx, q_idx), role="input")),
            )
        }
    )
    ref = execute(prog, {"W": whitener, "raw": raw}).outputs["df"]

    fn = bind_cphf_relax(host_cphf_library, "whiten_3c", host_reference=True)
    assert fn is not None
    got = run_whiten_3c_native(fn, whitener, raw, budget_bytes=budget_bytes)
    assert got.shape == ref.shape
    assert np.max(np.abs(got - ref)) < 1e-13


@pytest.mark.parametrize(("nao", "nmo"), [(8, 8), (15, 9), (12, 5)])
def test_orbital_rot_metric_parity_vs_tensorir(
    host_cphf_library: Path, nao: int, nmo: int
):
    rng = np.random.default_rng(202 + nao * 13 + nmo)
    coeff = rng.standard_normal((nao, nmo))
    bar = rng.standard_normal((nao, nmo))
    obs_split = max(1, nmo // 2)
    mask = np.full((nmo, nmo), -0.5)
    mask[:obs_split, obs_split:] = -1.0
    mask[obs_split:, :obs_split] = 0.0

    ref_rot = execute(
        orbital_rotation_cotangent(nao, nmo), {"coefficients": coeff, "frame_bar": bar}
    ).outputs["rotation_gradient"]
    ref_ov = execute(
        orbital_metric_cotangent(nao, nmo),
        {"coefficients": coeff, "frame_bar": bar, "metric_blocks": mask},
    ).outputs["overlap_bar"]

    fn = bind_cphf_relax(host_cphf_library, "orbital_rot_metric", host_reference=True)
    assert fn is not None

    rot_only, ov_none = run_orbital_rot_metric_native(
        fn, coeff, bar, compute_rotation=True, compute_overlap=False
    )
    assert ov_none is None and rot_only is not None
    assert np.max(np.abs(rot_only - ref_rot)) < 1e-13

    rot_none, ov_only = run_orbital_rot_metric_native(
        fn, coeff, bar, mask=mask, compute_rotation=False, compute_overlap=True
    )
    assert rot_none is None and ov_only is not None
    assert np.max(np.abs(ov_only - ref_ov)) < 1e-13
    assert np.array_equal(ov_only, ov_only.T)

    rot_both, ov_both = run_orbital_rot_metric_native(
        fn, coeff, bar, mask=mask, compute_rotation=True, compute_overlap=True
    )
    assert rot_both is not None and ov_both is not None
    assert np.max(np.abs(rot_both - ref_rot)) < 1e-13
    assert np.max(np.abs(ov_both - ref_ov)) < 1e-13
    assert np.array_equal(ov_both, ov_both.T)


def test_cuda_tiled_rotation_and_frame_against_numpy():
    path = os.environ.get("GENERATIVEQC_F12_CUDA_AUDIT_SINK")
    if not path:
        pytest.skip("requires the installed CUDA molecular shell on a GPU")
    primitive = bind_cphf_relax(Path(path), "orbital_rot_metric", host_reference=False)
    assert primitive is not None
    nao, nmo = 39, 32
    rng = np.random.default_rng(1973)
    coeff, cotangent = rng.normal(scale=0.25, size=(2, nao, nmo))
    mask = rng.normal(size=(nmo, nmo))
    reference_rotation = coeff.T @ cotangent
    weighted = mask * reference_rotation
    reference_overlap = coeff @ ((weighted + weighted.T) * 0.5) @ coeff.T
    # Only two MO-square matrices fit. One AO row per rotation tile and an
    # incomplete final AO block exercise both diagonals and rectangular tiles.
    budget = 8 * (2 * nmo**2 + 2 * nmo)
    rotation, overlap = run_orbital_rot_metric_native(
        primitive, coeff, cotangent, mask=mask, compute_rotation=True,
        compute_overlap=True, budget_bytes=budget,
    )
    np.testing.assert_allclose(rotation, reference_rotation, rtol=0, atol=1e-12)
    np.testing.assert_allclose(overlap, reference_overlap, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(overlap, overlap.T)


@pytest.mark.parametrize(
    ("oa", "obs", "aux", "budget_bytes"),
    [
        (3, 8, 12, 512 << 20),
        (4, 10, 23, 8192),  # forced multi-tile Q
    ],
)
def test_cphf_action_rhf_parity_vs_tensorir(
    host_cphf_library: Path, oa: int, obs: int, aux: int, budget_bytes: int
):
    rng = np.random.default_rng(303 + oa * 11 + aux)
    va = obs - oa
    vo = rng.standard_normal((aux, va, oa))
    oo = rng.standard_normal((aux, oa, oa))
    vv = rng.standard_normal((aux, va, va))
    gap = rng.uniform(0.5, 3.0, size=(va, oa))
    trial = rng.standard_normal((va, oa))

    ref = execute(
        _rhf_df_orbital_action(oa, obs, aux),
        {"vo": vo, "oo": oo, "vv": vv, "gap": gap, "trial": trial},
    ).outputs["rhf"]

    fn = bind_cphf_relax(host_cphf_library, "cphf_action", host_reference=True)
    assert fn is not None
    out_rhf, out_b = run_cphf_action_native(
        fn,
        is_rhf=True,
        oa=oa,
        ob=0,
        obs=obs,
        aux=aux,
        vo_a=vo,
        oo_a=oo,
        vv_a=vv,
        gap_a=gap,
        trial_a=trial,
        budget_bytes=budget_bytes,
    )
    assert out_b is None and out_rhf is not None
    assert np.max(np.abs(out_rhf - ref)) < 1e-13

    # Verify RHF fast path matches UHF path with identical alpha/beta tensors.
    out_ua, out_ub = run_cphf_action_native(
        fn,
        is_rhf=False,
        oa=oa,
        ob=oa,
        obs=obs,
        aux=aux,
        vo_a=vo,
        oo_a=oo,
        vv_a=vv,
        gap_a=gap,
        trial_a=trial,
        vo_b=vo,
        oo_b=oo,
        vv_b=vv,
        gap_b=gap,
        trial_b=trial,
        budget_bytes=budget_bytes,
    )
    assert out_ua is not None and out_ub is not None
    assert np.max(np.abs(out_ua - ref)) < 1e-13
    assert np.max(np.abs(out_ub - ref)) < 1e-13


@pytest.mark.parametrize(
    ("oa", "ob", "obs", "aux", "budget_bytes"),
    [
        (4, 2, 10, 19, 512 << 20),
        (4, 2, 10, 25, 8192),  # forced multi-tile Q
        (3, 0, 8, 17, 4096),  # alpha-only spin edge case + multi-tile Q
        (0, 2, 7, 15, 4096),  # beta-only spin edge case + multi-tile Q
    ],
)
def test_cphf_action_uhf_parity_vs_tensorir(
    host_cphf_library: Path, oa: int, ob: int, obs: int, aux: int, budget_bytes: int
):
    rng = np.random.default_rng(404 + oa * 17 + ob * 5 + aux)
    inputs: dict[str, np.ndarray] = {}
    kw: dict[str, np.ndarray | None] = {}
    for spin, suffix, occ in (("alpha", "a", oa), ("beta", "b", ob)):
        if not occ:
            continue
        virt = obs - occ
        vo = rng.standard_normal((aux, virt, occ))
        oo = rng.standard_normal((aux, occ, occ))
        vv = rng.standard_normal((aux, virt, virt))
        gap = rng.uniform(0.5, 3.0, size=(virt, occ))
        trial = rng.standard_normal((virt, occ))
        inputs.update(
            {
                f"vo_{spin}": vo,
                f"oo_{spin}": oo,
                f"vv_{spin}": vv,
                f"gap_{spin}": gap,
                f"trial_{spin}": trial,
            }
        )
        kw.update(
            {
                f"vo_{suffix}": vo,
                f"oo_{suffix}": oo,
                f"vv_{suffix}": vv,
                f"gap_{suffix}": gap,
                f"trial_{suffix}": trial,
            }
        )

    ref = execute(unrestricted_df_orbital_action(oa, ob, obs, aux), inputs).outputs
    fn = bind_cphf_relax(host_cphf_library, "cphf_action", host_reference=True)
    assert fn is not None
    out_a, out_b = run_cphf_action_native(
        fn,
        is_rhf=False,
        oa=oa,
        ob=ob,
        obs=obs,
        aux=aux,
        budget_bytes=budget_bytes,
        **kw,
    )
    if oa > 0:
        assert out_a is not None
        assert np.max(np.abs(out_a - ref["alpha"])) < 1e-13
    else:
        assert out_a is None
    if ob > 0:
        assert out_b is not None
        assert np.max(np.abs(out_b - ref["beta"])) < 1e-13
    else:
        assert out_b is None


@pytest.mark.parametrize(
    ("oa", "ob", "obs", "aux", "budget_bytes"),
    [
        (4, 2, 9, 17, 512 << 20),
        (4, 2, 9, 23, 8192),  # forced multi-tile Q
        (3, 3, 8, 19, 8192),  # symmetric occupations + multi-tile Q
        (3, 0, 8, 15, 4096),  # alpha-only + multi-tile Q
        (0, 2, 7, 13, 4096),  # beta-only + multi-tile Q
    ],
)
def test_df_canonical_driver_parity_vs_tensorir(
    host_cphf_library: Path, oa: int, ob: int, obs: int, aux: int, budget_bytes: int
):
    rng = np.random.default_rng(505 + oa * 19 + ob * 7 + aux)
    inputs: dict[str, np.ndarray] = {}
    kw: dict[str, np.ndarray | None] = {}
    for spin, suffix, occ in (("alpha", "a", oa), ("beta", "b", ob)):
        if not occ:
            continue
        df = rng.standard_normal((aux, obs, obs)) * 0.3
        seed = rng.standard_normal((obs, obs)) * 0.3
        inputs[f"df_{spin}"] = df
        inputs[f"fock_seed_{spin}"] = seed
        kw[f"df_{suffix}"] = df
        kw[f"fock_seed_{suffix}"] = seed

    ref = execute(unrestricted_df_canonical_driver(oa, ob, obs, aux), inputs).outputs
    fn = bind_cphf_relax(host_cphf_library, "df_canonical_driver", host_reference=True)
    assert fn is not None
    got = run_df_canonical_driver_native(
        fn, oa=oa, ob=ob, obs=obs, aux=aux, budget_bytes=budget_bytes, **kw
    )
    assert set(got.keys()) == set(ref.keys())
    for spin in ref:
        assert np.max(np.abs(got[spin] - ref[spin])) < 1e-13
        np.testing.assert_allclose(got[spin], ref[spin], rtol=1e-13, atol=1e-13)


def test_wired_orbital_response_helpers_zero_tensorir(
    host_cphf_library: Path, monkeypatch: pytest.MonkeyPatch
):
    rng = np.random.default_rng(606)
    nao, obs, cabs_rank, oa, ob, aux = 14, 7, 4, 3, 2, 16
    retained = obs + cabs_rank

    # 1) unrestricted_frame_overlap_weights (non-square nao x retained)
    central_frame = {
        "coefficients_alpha": rng.standard_normal((nao, obs)),
        "coefficients_beta": rng.standard_normal((nao, obs)),
    }
    complement = rng.standard_normal((nao, cabs_rank))
    direct = {
        "coefficients_alpha": rng.standard_normal((nao, retained)),
        "coefficients_beta": rng.standard_normal((nao, retained)),
    }
    ref_run, ref_calls = _counting_runner()
    with _tensorir_oracle(monkeypatch):
        ref_ov = unrestricted_frame_overlap_weights(
            central_frame, complement, direct, ref_run
        )
    assert ref_calls["count"] == 2

    aot_run, aot_calls = _forbidding_runner()
    aot_ov = unrestricted_frame_overlap_weights(
        central_frame,
        complement,
        direct,
        aot_run,
        native_library=host_cphf_library,
        host_reference=True,
    )
    assert aot_calls["count"] == 0
    for spin in ("alpha", "beta"):
        assert np.max(np.abs(aot_ov[spin] - ref_ov[spin])) < 1e-13

    # 2) unrestricted_canonical_source_seeds (non-square nao x obs, extra CABS columns in bar)
    import generativeqc.f12_orbital_response as orb_mod

    energies = {
        "alpha": np.linspace(-1.5, 2.0, obs),
        "beta": np.linspace(-1.3, 2.2, obs),
    }

    def cpu_canonical_seeds(m, e, occ, **_kw):
        # Pure-CPU stand-in for canonical_full_frame_seeds so we verify the exact
        # orbital_rotation_cotangent matrix `m` fed into it without requiring GPU libs.
        f = 0.5 * (m + m.T)
        s = -0.5 * (m + m.T)
        torque = float(np.max(np.abs(m[:occ, :occ] - m[:occ, :occ].T)))
        return f, s, torque, m.size

    monkeypatch.setattr(orb_mod, "canonical_full_frame_seeds", cpu_canonical_seeds)

    ref_run2, ref_calls2 = _counting_runner()
    with _tensorir_oracle(monkeypatch):
        ref_seeds, ref_torques = unrestricted_canonical_source_seeds(
            (oa, ob), central_frame, energies, direct, ref_run2
        )
    assert ref_calls2["count"] == 2

    aot_run2, aot_calls2 = _forbidding_runner()
    aot_seeds, aot_torques = unrestricted_canonical_source_seeds(
        (oa, ob),
        central_frame,
        energies,
        direct,
        aot_run2,
        native_library=host_cphf_library,
        host_reference=True,
    )
    assert aot_calls2["count"] == 0
    for key in ref_seeds:
        assert np.max(np.abs(aot_seeds[key] - ref_seeds[key])) < 1e-13
    for spin in ("alpha", "beta"):
        assert abs(aot_torques[spin] - ref_torques[spin]) < 1e-13

    # 3) unrestricted_source_orbital_adjoint (driver path, mocking GPU Hessian solve)
    captured_drivers: list[np.ndarray] = []

    def fake_solve(space, factors, electronic_driver, *_args):
        captured_drivers.append(electronic_driver.copy())
        return UnrestrictedOrbitalForces(
            rotations=np.zeros_like(electronic_driver),
            orbital_gradient=electronic_driver.copy(),
            forces=np.zeros((0, 3)),
        )

    monkeypatch.setattr(orb_mod, "_solve_orbital_response", fake_solve)
    central_drv = {
        "df_alpha": rng.standard_normal((aux, obs, obs)) * 0.3,
        "df_beta": rng.standard_normal((aux, obs, obs)) * 0.3,
        "fock_seed_alpha": rng.standard_normal((obs, obs)) * 0.3,
        "fock_seed_beta": rng.standard_normal((obs, obs)) * 0.3,
    }
    ref_run3, ref_calls3 = _counting_runner()
    with _tensorir_oracle(monkeypatch):
        ref_drv, _ = unrestricted_source_orbital_adjoint(
            (oa, ob, obs, aux), central_drv, energies, ref_run3
        )
    assert ref_calls3["count"] >= 1

    aot_run3, aot_calls3 = _forbidding_runner()
    aot_drv, _ = unrestricted_source_orbital_adjoint(
        (oa, ob, obs, aux),
        central_drv,
        energies,
        aot_run3,
        native_library=host_cphf_library,
        host_reference=True,
    )
    assert aot_calls3["count"] == 0
    assert np.max(np.abs(aot_drv - ref_drv)) < 1e-13


def _build_synthetic_nuclear_weights_inputs(
    rng: np.random.Generator,
    oa: int,
    ob: int,
    obs: int,
    aux: int,
    nao: int,
    *,
    rhf_shared: bool = False,
    obs_only_ao: bool = False,
    include_direct: bool = False,
):
    """Build synthetic central, adjoint, and optional direct inputs for nuclear weights."""
    whitener = rng.standard_normal((aux, aux)) * 0.25

    def _make_coeff():
        c = rng.standard_normal((nao, obs)) * 0.4
        if obs_only_ao and nao > obs:
            c[obs:, :] = 0.0
        return c

    if rhf_shared:
        assert oa == ob
        coeff = _make_coeff()
        df = rng.standard_normal((aux, obs, obs)) * 0.25
        fseed = rng.standard_normal((obs, obs)) * 0.3
        oseed = rng.standard_normal((obs, obs)) * 0.3
        eps = np.linspace(-1.8, -0.5, oa)
        z_spin = rng.standard_normal((obs - oa, oa)) * 0.3
        central = {
            "whitener": whitener,
            "coefficients_alpha": coeff,
            "coefficients_beta": coeff,
            "df_alpha": df,
            "df_beta": df,
            "fock_seed_alpha": fseed,
            "fock_seed_beta": fseed,
            "overlap_seed_alpha": oseed,
            "overlap_seed_beta": oseed,
            "occupied_energies_alpha": eps,
            "occupied_energies_beta": eps,
        }
        z = np.concatenate([z_spin.ravel(), z_spin.ravel()])
    else:
        central = {"whitener": whitener}
        z_parts = []
        for spin, occ in (("alpha", oa), ("beta", ob)):
            central[f"coefficients_{spin}"] = _make_coeff()
            central[f"df_{spin}"] = rng.standard_normal((aux, obs, obs)) * 0.25
            central[f"fock_seed_{spin}"] = rng.standard_normal((obs, obs)) * 0.3
            central[f"overlap_seed_{spin}"] = rng.standard_normal((obs, obs)) * 0.3
            if occ:
                central[f"occupied_energies_{spin}"] = rng.uniform(-2.0, -0.4, size=occ)
                z_parts.append(rng.standard_normal((obs - occ) * occ) * 0.3)
        z = np.concatenate(z_parts)

    direct_weights = None
    frame_overlap = None
    if include_direct:
        ao_coul = rng.standard_normal((aux, nao, nao)) * 0.2
        ao_coul = 0.5 * (ao_coul + np.swapaxes(ao_coul, 1, 2))
        direct_weights = {
            "overlap_ao": _random_sym(rng, nao) * 0.2,
            "hcore_ao": _random_sym(rng, nao) * 0.2,
            "ao_coulomb": ao_coul,
            "metric/coulomb": _random_sym(rng, aux) * 0.2,
        }
        frame_overlap = {
            "alpha": _random_sym(rng, nao) * 0.2,
            "beta": _random_sym(rng, nao) * 0.2,
        }
    return central, z, direct_weights, frame_overlap


@pytest.mark.parametrize(
    ("oa", "ob", "obs", "aux", "nao", "rhf_shared", "obs_only_ao", "include_direct", "budget_bytes"),
    [
        # RHF shared (df_alpha is df_beta), single-tile and multi-tile
        (3, 3, 7, 13, 7, True, False, False, 512 << 20),
        (3, 3, 7, 17, 7, True, False, True, 8192),  # forced multi-tile Q
        # General UHF (oa != ob), with & without direct_weights
        (4, 2, 8, 15, 8, False, False, False, 512 << 20),
        (4, 2, 8, 19, 8, False, False, True, 12288),  # forced multi-tile Q
        # Single-spin edge cases (ob == 0 and oa == 0)
        (3, 0, 7, 14, 7, False, False, False, 8192),
        (0, 2, 7, 14, 7, False, False, True, 8192),
        # Non-square nao > n: OBS+CABS zero-padded tail AND dense non-zero tail
        (3, 2, 6, 15, 11, False, True, True, 12288),
        (3, 2, 6, 15, 11, False, False, False, 16384),
        (2, 2, 6, 17, 10, True, False, True, 12288),
    ],
)
def test_df_nuclear_weights_parity_vs_tensorir(
    host_cphf_library: Path,
    monkeypatch: pytest.MonkeyPatch,
    oa: int,
    ob: int,
    obs: int,
    aux: int,
    nao: int,
    rhf_shared: bool,
    obs_only_ao: bool,
    include_direct: bool,
    budget_bytes: int,
):
    seed = (
        1100
        + oa * 31
        + ob * 17
        + obs * 7
        + aux * 3
        + nao * 5
        + int(rhf_shared) * 13
        + int(obs_only_ao) * 23
        + int(include_direct) * 29
    )
    rng = np.random.default_rng(seed)
    central, z, direct_weights, frame_overlap = _build_synthetic_nuclear_weights_inputs(
        rng,
        oa,
        ob,
        obs,
        aux,
        nao,
        rhf_shared=rhf_shared,
        obs_only_ao=obs_only_ao,
        include_direct=include_direct,
    )
    direct_ao_before = (
        direct_weights["ao_coulomb"].copy() if direct_weights is not None else None
    )

    ref_run, ref_calls = _counting_runner()
    with _tensorir_oracle(monkeypatch):
        ref = unrestricted_nuclear_source_weights(
            (oa, ob, obs, aux),
            central,
            z,
            ref_run,
            direct_weights=direct_weights,
            frame_overlap=frame_overlap,
        )
    assert ref_calls["count"] >= 1

    aot_run, aot_calls = _forbidding_runner()
    got = unrestricted_nuclear_source_weights(
        (oa, ob, obs, aux),
        central,
        z,
        aot_run,
        direct_weights=direct_weights,
        frame_overlap=frame_overlap,
        native_library=host_cphf_library,
        host_reference=True,
        budget_bytes=budget_bytes,
    )
    assert aot_calls["count"] == 0
    assert set(got.keys()) == set(ref.keys())
    for key in ref:
        assert got[key].shape == ref[key].shape
        max_err = float(np.max(np.abs(got[key] - ref[key])))
        assert max_err < 1e-12, f"{key} max_err={max_err:.3e}"
    if direct_ao_before is not None:
        assert direct_weights is not None
        assert np.array_equal(direct_weights["ao_coulomb"], direct_ao_before)


@pytest.mark.parametrize(
    ("occs", "matrix_free"),
    [
        ((3,), True),  # RHF matrix_free -> rhf_matvec + nuclear weights
        ((3,), False),  # RHF CPU -> batched_matvec + nuclear weights
        ((3, 2), True),  # UHF matrix_free -> batched_matvec + nuclear weights
    ],
)
def test_wired_relaxed_reference_weights_zero_tensorir(
    host_cphf_library: Path,
    monkeypatch: pytest.MonkeyPatch,
    occs: tuple[int, ...],
    matrix_free: bool,
):
    rng = np.random.default_rng(707 + len(occs) * 10 + int(matrix_free))
    obs, aux = 7, 13
    reference, raw, whitener, weights, overlap_seed = _build_synthetic_cphf_inputs(
        rng, occs, obs, aux
    )

    # Full end-to-end TensorIR reference (NO stubs!)
    ref_run, ref_calls = _counting_runner()
    with _tensorir_oracle(monkeypatch):
        ref_out = relaxed_reference_weights(
            reference,
            raw,
            whitener,
            weights,
            overlap_seed,
            ref_run,
            None,
            64,
            matrix_free=matrix_free,
        )
    assert ref_calls["count"] > 0

    # Full end-to-end AOT under strict GENERATIVEQC_F12_REQUIRE_AOT_CPHF=1 -> 0 TensorIR calls!
    monkeypatch.setenv(ENV_REQUIRE_CPHF, "1")
    aot_run, aot_calls = _forbidding_runner()
    aot_out = relaxed_reference_weights(
        reference,
        raw,
        whitener,
        weights,
        overlap_seed,
        aot_run,
        None,
        64,
        matrix_free=matrix_free,
        native_library=host_cphf_library,
        host_reference=True,
    )
    assert aot_calls["count"] == 0
    assert set(aot_out.keys()) == set(ref_out.keys())
    for key in ref_out:
        assert aot_out[key].shape == ref_out[key].shape
        max_err = float(np.max(np.abs(aot_out[key] - ref_out[key])))
        assert max_err < 1e-11, f"{key} max_err={max_err:.3e}"


@pytest.mark.parametrize("require_env", [ENV_REQUIRE_STEP1, ENV_REQUIRE_CPHF])
def test_require_aot_cphf_env_gate(
    host_cphf_library: Path,
    monkeypatch: pytest.MonkeyPatch,
    require_env: str,
):
    rng = np.random.default_rng(808)
    obs, aux = 6, 9
    reference, raw, whitener, weights, overlap_seed = _build_synthetic_cphf_inputs(
        rng, (2,), obs, aux
    )

    # Strict env var without any native library MUST fail cleanly for all 5 primitives.
    monkeypatch.delenv(ENV_REQUIRE_CPHF, raising=False)
    monkeypatch.delenv(ENV_REQUIRE_STEP1, raising=False)
    monkeypatch.setenv(require_env, "1")
    monkeypatch.delenv(ENV_CPHF_LIBRARY, raising=False)
    monkeypatch.delenv(ENV_SHELL_LIBRARY, raising=False)
    monkeypatch.delenv(ENV_SINK_LIBRARY, raising=False)
    # Libraries installed with the package or loaded by earlier tests in this
    # process would otherwise satisfy the strict gate.
    monkeypatch.setattr(cphf_native, "installed_f12_shell_library", lambda **_: None)
    monkeypatch.setattr(cphf_native, "_discover_loaded_shell_libraries", list)

    dummy_run, _ = _forbidding_runner()
    with pytest.raises(RuntimeError, match=require_env):
        relaxed_reference_weights(
            reference, raw, whitener, weights, overlap_seed, dummy_run, None, 64
        )

    c_mat = rng.standard_normal((8, 5))
    cabs = rng.standard_normal((8, 2))
    bar = rng.standard_normal((8, 7))
    with pytest.raises(RuntimeError, match=require_env):
        unrestricted_frame_overlap_weights(
            {"coefficients_alpha": c_mat, "coefficients_beta": c_mat},
            cabs,
            {"coefficients_alpha": bar, "coefficients_beta": bar},
            dummy_run,
        )

    with pytest.raises(RuntimeError, match=require_env):
        unrestricted_canonical_source_seeds(
            (2, 2),
            {"coefficients_alpha": c_mat, "coefficients_beta": c_mat},
            {"alpha": np.linspace(-1.0, 1.0, 5), "beta": np.linspace(-1.0, 1.0, 5)},
            {"coefficients_alpha": bar, "coefficients_beta": bar},
            dummy_run,
        )

    with pytest.raises(RuntimeError, match=require_env):
        unrestricted_source_orbital_adjoint(
            (2, 2, obs, aux),
            {
                "df_alpha": rng.standard_normal((aux, obs, obs)),
                "df_beta": rng.standard_normal((aux, obs, obs)),
                "fock_seed_alpha": rng.standard_normal((obs, obs)),
                "fock_seed_beta": rng.standard_normal((obs, obs)),
            },
            {
                "alpha": np.linspace(-1.0, 1.0, obs),
                "beta": np.linspace(-1.0, 1.0, obs),
            },
            dummy_run,
        )

    central_nw, z_nw, _, _ = _build_synthetic_nuclear_weights_inputs(
        rng, 2, 2, obs, aux, obs, rhf_shared=True
    )
    with pytest.raises(RuntimeError, match=require_env):
        unrestricted_nuclear_source_weights(
            (2, 2, obs, aux), central_nw, z_nw, dummy_run
        )

    # Setting GENERATIVEQC_F12_SHELL_LIBRARY (or CPHF_LIBRARY) makes strict end-to-end
    # relaxed_reference_weights pass with 0 TensorIR calls!
    monkeypatch.setenv(ENV_SHELL_LIBRARY, str(host_cphf_library))
    aot_run, aot_calls = _forbidding_runner()
    out = relaxed_reference_weights(
        reference,
        raw,
        whitener,
        weights,
        overlap_seed,
        aot_run,
        None,
        64,
        matrix_free=True,
    )
    assert aot_calls["count"] == 0
    assert all(np.isfinite(val).all() for val in out.values())


def test_cphf_autodiscover_loaded_focal_shell_maps(
    host_cphf_library: Path, tmp_path: Path
):
    """Verify /proc/self/maps RTLD_NOLOAD auto-discovers loaded libfocal_shell_*.so."""
    if not Path("/proc/self/maps").exists():
        pytest.skip("/proc/self/maps is required for loaded-library auto-discovery")

    shell_so = tmp_path / "libfocal_shell_autodiscover_test.so"
    shutil.copy2(host_cphf_library, shell_so)

    script = """
import ctypes as ct
import sys
import types
import numpy as np

from generativeqc._f12_cphf_relax_native import resolve_cphf_relax
from generativeqc._f12_focal_singles_response import relaxed_reference_weights

# Pre-load libfocal_shell_*.so without setting any GENERATIVEQC_F12_*_LIBRARY env vars.
_loaded = ct.CDLL(sys.argv[1])
# Pass an unrelated CDLL (libc) as fallback_library, mimicking provider.response_library.
unrelated = ct.CDLL(None)

for kind in (
    "whiten_3c",
    "orbital_rot_metric",
    "cphf_action",
    "df_canonical_driver",
    "df_nuclear_weights",
):
    fn = resolve_cphf_relax(None, kind, fallback_library=unrelated, host_reference=True)
    assert fn is not None, f"Failed to auto-discover {kind}"

rng = np.random.default_rng(1234)
obs, aux, occ = 6, 9, 2
q_mat, _ = np.linalg.qr(rng.standard_normal((obs, obs)))
occ_e = np.sort(rng.uniform(-2.0, -0.8, size=occ))
virt_e = np.sort(rng.uniform(0.8, 2.5, size=obs - occ))
m = np.zeros((obs, obs), dtype=np.float64)
a_oo = rng.standard_normal((occ, occ))
m[:occ, :occ] = 0.5 * (a_oo + a_oo.T)
a_vv = rng.standard_normal((obs - occ, obs - occ))
m[occ:, occ:] = 0.5 * (a_vv + a_vv.T)
m[:occ, occ:] = rng.standard_normal((occ, obs - occ))
m[occ:, :occ] = rng.standard_normal((obs - occ, occ))
bar = np.linalg.solve(q_mat.T, m)
raw_sym = rng.standard_normal((aux, obs, obs)) * 0.08
raw_sym = 0.5 * (raw_sym + np.swapaxes(raw_sym, 1, 2))
reference = types.SimpleNamespace(
    occupied=(occ,),
    executed_backend="cpu",
    overlap=np.eye(obs, dtype=np.float64),
    coefficients=q_mat[None, :, :],
    orbital_energies=np.concatenate([occ_e, virt_e])[None, :],
)
weights = {
    "obs_coefficients": bar[None, :, :],
    "hcore": np.zeros((obs, obs), dtype=np.float64),
    "three_center": np.zeros((aux, obs, obs), dtype=np.float64),
    "metric": np.zeros((aux, aux), dtype=np.float64),
}
raw = {"raw_coulomb_alpha": raw_sym, "raw_coulomb_beta": raw_sym}
whitener = rng.standard_normal((aux, aux)) * 0.2
ov_seed = np.zeros((obs, obs), dtype=np.float64)

calls = 0
def forbid_run(program, inputs):
    global calls
    calls += 1
    raise AssertionError(f"Unexpected TensorIR call: {program.provenance}")

out = relaxed_reference_weights(
    reference,
    raw,
    whitener,
    weights,
    ov_seed,
    forbid_run,
    unrelated,
    64,
    matrix_free=True,
)
assert calls == 0
assert all(np.isfinite(v).all() for v in out.values())
print("AUTODISCOVER_OK")
"""
    env = dict(os.environ)
    for k in (ENV_CPHF_LIBRARY, ENV_SHELL_LIBRARY, ENV_SINK_LIBRARY, ENV_REQUIRE_STEP1):
        env.pop(k, None)
    env[ENV_REQUIRE_CPHF] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script, str(shell_so)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "AUTODISCOVER_OK" in proc.stdout


def test_cphf_error_handling_and_gauge_check(host_cphf_library: Path):
    rng = np.random.default_rng(909)
    # 1) Too-small budget for a large single Q-slice returns status -5 -> RuntimeError.
    driver_fn = bind_cphf_relax(
        host_cphf_library, "df_canonical_driver", host_reference=True
    )
    assert driver_fn is not None
    obs = 16  # 3*256 + 2*2*256 > 512 doubles (4096 bytes)
    with pytest.raises(RuntimeError, match="-5"):
        run_df_canonical_driver_native(
            driver_fn,
            oa=4,
            ob=4,
            obs=obs,
            aux=3,
            df_a=np.zeros((3, obs, obs)),
            fock_seed_a=np.zeros((obs, obs)),
            df_b=np.zeros((3, obs, obs)),
            fock_seed_b=np.zeros((obs, obs)),
            budget_bytes=4096,
        )

    # 2) Too-small budget for df_nuclear_weights returns status -5 -> RuntimeError.
    nw_fn = bind_cphf_relax(
        host_cphf_library, "df_nuclear_weights", host_reference=True
    )
    assert nw_fn is not None
    with pytest.raises(RuntimeError, match="-5"):
        run_df_nuclear_weights_native(
            nw_fn,
            oa=4,
            ob=4,
            obs=obs,
            aux=3,
            nao=obs,
            coefficients_alpha=np.zeros((obs, obs)),
            coefficients_beta=np.zeros((obs, obs)),
            occupied_energies_alpha=np.zeros((4,)),
            occupied_energies_beta=np.zeros((4,)),
            df_alpha=np.zeros((3, obs, obs)),
            df_beta=np.zeros((3, obs, obs)),
            fock_seed_alpha=np.zeros((obs, obs)),
            fock_seed_beta=np.zeros((obs, obs)),
            overlap_seed_alpha=np.zeros((obs, obs)),
            overlap_seed_beta=np.zeros((obs, obs)),
            whitener=np.zeros((3, 3)),
            z_alpha=np.zeros((obs - 4, 4)),
            z_beta=np.zeros((obs - 4, 4)),
            budget_bytes=4096,
        )

    # 3) Non-gauge-invariant obs_coefficients still raises ValueError on AOT path.
    obs_s, aux_s = 6, 8
    reference, raw, whitener, weights, overlap_seed = _build_synthetic_cphf_inputs(
        rng, (2,), obs_s, aux_s
    )
    weights["obs_coefficients"] = rng.standard_normal((1, obs_s, obs_s))
    aot_run, aot_calls = _forbidding_runner()
    with pytest.raises(ValueError, match="gauge invariance"):
        relaxed_reference_weights(
            reference,
            raw,
            whitener,
            weights,
            overlap_seed,
            aot_run,
            None,
            64,
            matrix_free=True,
            native_library=host_cphf_library,
            host_reference=True,
        )
    assert aot_calls["count"] == 0
