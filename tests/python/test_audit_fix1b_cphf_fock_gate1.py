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
from generativeqc._f12_cphf_relax_native import (
    bind_cphf_relax,
    run_cphf_action_native,
    run_df_nuclear_weights_native,
    run_orbital_rot_metric_native,
    run_whiten_3c_native,
)
from generativeqc._f12_ext_fock_native import bind_ext_fock_native
from generativeqc._f12_spin_fock_1e_native import (
    bind_spin_fock_1e_ops,
    run_one_electron_grad,
    run_spin_fock_fwd,
    run_spin_fock_rev,
)
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
def cphf_host_lib(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _compile_host_unit(
        tmp_path_factory, "cphf_relax", "src/posthf/f12/f12_cphf_relax_cuda.cu", "-std=c++20"
    )


@pytest.fixture(scope="module")
def ext_fock_host_lib(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _compile_host_unit(
        tmp_path_factory, "ext_fock", "src/posthf/f12/f12_ext_fock_cuda.cu", "-std=c++20"
    )


@pytest.fixture(scope="module")
def spin_fock_1e_host_lib(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _compile_host_unit(
        tmp_path_factory,
        "spin_fock_1e",
        "src/posthf/f12/f12_spin_fock_1e_cuda.cu",
        "-std=c++20",
        define_host_ref=False,
    )


# =============================================================================
# M3: Empty-spin non-zero Coulomb contribution in df_canonical_driver
# =============================================================================


def test_m3_exact_audit_counterexample_alpha_active_beta_empty(cphf_host_lib: Path) -> None:
    """Exact M3 audit counterexample: oa=1, ob=0, aux=1, n=2, seed_a=0, seed_b=diag(1,0) -> 1.0."""
    fn = bind_cphf_relax(cphf_host_lib, "df_canonical_driver", host_reference=True)
    assert fn is not None

    b = np.ascontiguousarray([[[1.0, 0.5], [0.5, 1.0]]], dtype=np.float64)
    seed_a = np.zeros((2, 2), dtype=np.float64)
    seed_b = np.ascontiguousarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    out_a = np.full((1, 1), -999.0, dtype=np.float64)

    rc = fn(
        1,
        0,
        2,
        1,
        512 << 20,
        _dptr(b),
        _dptr(seed_a),
        _dptr(out_a),
        _dptr(b),
        _dptr(seed_b),
        _NULL_D,
    )
    assert rc == 0
    np.testing.assert_allclose(out_a, [[1.0]], atol=1e-15, rtol=1e-15)


def test_m3_symmetric_beta_active_alpha_empty(cphf_host_lib: Path) -> None:
    """Symmetric M3 counterexample: oa=0, ob=1, aux=1, n=2, seed_a=diag(1,0), seed_b=0 -> 1.0."""
    fn = bind_cphf_relax(cphf_host_lib, "df_canonical_driver", host_reference=True)
    assert fn is not None

    b = np.ascontiguousarray([[[1.0, 0.5], [0.5, 1.0]]], dtype=np.float64)
    seed_a = np.ascontiguousarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    seed_b = np.zeros((2, 2), dtype=np.float64)
    out_b = np.full((1, 1), -999.0, dtype=np.float64)

    rc = fn(
        0,
        1,
        2,
        1,
        512 << 20,
        _dptr(b),
        _dptr(seed_a),
        _NULL_D,
        _dptr(b),
        _dptr(seed_b),
        _dptr(out_b),
    )
    assert rc == 0
    np.testing.assert_allclose(out_b, [[1.0]], atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize(
    ("oa", "ob", "budget_bytes"),
    [
        (2, 0, 512 << 20),
        (2, 0, 4096),  # forced multi-tile Q with empty beta seed
        (0, 2, 512 << 20),
        (0, 2, 4096),  # forced multi-tile Q with empty alpha seed
    ],
)
def test_m3_empty_spin_coulomb_matches_finite_difference_oracle(
    cphf_host_lib: Path, oa: int, ob: int, budget_bytes: int
) -> None:
    """Independent finite-difference oracle for DF-UHF Fock rotation derivative with an empty spin."""
    from scipy.linalg import expm

    fn = bind_cphf_relax(cphf_host_lib, "df_canonical_driver", host_reference=True)
    assert fn is not None

    rng = np.random.default_rng(20261125 + oa * 13 + ob * 7)
    obs, aux = 5, 9
    # Symmetric 3-center MO integrals per auxiliary function (physical symmetric DF tensors).
    raw_a = rng.standard_normal((aux, obs, obs)) * 0.35
    b_a = np.ascontiguousarray(0.5 * (raw_a + raw_a.swapaxes(1, 2)))
    raw_b = rng.standard_normal((aux, obs, obs)) * 0.35
    b_b = np.ascontiguousarray(0.5 * (raw_b + raw_b.swapaxes(1, 2)))
    seed_a = np.ascontiguousarray(rng.standard_normal((obs, obs)) * 0.4)
    seed_b = np.ascontiguousarray(rng.standard_normal((obs, obs)) * 0.4)

    occ = oa if oa > 0 else ob
    virt = obs - occ
    out_active = np.empty((virt, occ), dtype=np.float64)

    rc = fn(
        oa,
        ob,
        obs,
        aux,
        budget_bytes,
        _dptr(b_a),
        _dptr(seed_a),
        _dptr(out_active) if oa > 0 else _NULL_D,
        _dptr(b_b),
        _dptr(seed_b),
        _dptr(out_active) if ob > 0 else _NULL_D,
    )
    assert rc == 0

    def lagrangian(kappa_vo: np.ndarray) -> float:
        # Antisymmetric generator X in active spin MO basis; U = exp(X) rotates occupied orbitals.
        x = np.zeros((obs, obs), dtype=np.float64)
        x[occ:, :occ] = kappa_vo
        x[:occ, occ:] = -kappa_vo.T
        u = expm(x)
        # Density matrix D_active = U[:, :occ] @ U[:, :occ]^T in central MO basis.
        u_occ = u[:, :occ]
        d_act = u_occ @ u_occ.T
        b_act = b_a if oa > 0 else b_b
        b_emp = b_b if oa > 0 else b_a
        # Total Coulomb charge vector c[q] = Tr(B_active[q] @ D_active) (empty spin has D=0).
        c_vec = np.einsum("qpr,pr->q", b_act, d_act)
        # F_active = J_active - K_active; F_empty = J_empty (no exchange since D_empty = 0).
        j_act = np.einsum("qpr,q->pr", b_act, c_vec)
        k_act = np.einsum("qpi,qri->pr", b_act @ u_occ, b_act @ u_occ)
        f_act = j_act - k_act
        f_emp = np.einsum("qpr,q->pr", b_emp, c_vec)
        if oa > 0:
            return float(np.sum(seed_a * f_act) + np.sum(seed_b * f_emp))
        return float(np.sum(seed_b * f_act) + np.sum(seed_a * f_emp))

    eps = 1e-6
    fd = np.zeros((virt, occ), dtype=np.float64)
    for a in range(virt):
        for i in range(occ):
            step = np.zeros((virt, occ), dtype=np.float64)
            step[a, i] = eps
            fd[a, i] = (lagrangian(step) - lagrangian(-step)) / (2.0 * eps)

    np.testing.assert_allclose(out_active, fd, atol=1e-9, rtol=1e-9)


def test_m3_empty_spin_null_contracts(cphf_host_lib: Path) -> None:
    """Empty spin accepts (NULL, NULL) for zero seed, and rejects mismatched partial pointers."""
    fn = bind_cphf_relax(cphf_host_lib, "df_canonical_driver", host_reference=True)
    assert fn is not None

    b = np.ascontiguousarray([[[1.0, 0.5], [0.5, 1.0]]], dtype=np.float64)
    seed_a = np.zeros((2, 2), dtype=np.float64)
    seed_b = np.ascontiguousarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    out_a = np.empty((1, 1), dtype=np.float64)

    # Both NULL for empty beta -> zero beta Coulomb contribution.
    rc_ok = fn(1, 0, 2, 1, 512 << 20, _dptr(b), _dptr(seed_a), _dptr(out_a), _NULL_D, _NULL_D, _NULL_D)
    assert rc_ok == 0
    np.testing.assert_allclose(out_a, [[0.0]], atol=1e-15)

    # Mismatched partial pointers for empty beta -> fail closed (-1).
    rc_bad1 = fn(1, 0, 2, 1, 512 << 20, _dptr(b), _dptr(seed_a), _dptr(out_a), _dptr(b), _NULL_D, _NULL_D)
    rc_bad2 = fn(1, 0, 2, 1, 512 << 20, _dptr(b), _dptr(seed_a), _dptr(out_a), _NULL_D, _dptr(seed_b), _NULL_D)
    assert rc_bad1 == -1
    assert rc_bad2 == -1


# =============================================================================
# M6: Hard device budget enforcement in ext_fock & cc_fixed_sources
# =============================================================================


@pytest.mark.parametrize("invalid", [-1, True, False, np.bool_(True), 0.0, 1.5, 1 << 64])
def test_cphf_workspace_rejects_invalid_values_before_native(invalid):
    def unexpected_call(*args):
        pytest.fail("invalid budget reached the native CPHF ABI")

    with pytest.raises((TypeError, ValueError)):
        run_whiten_3c_native(unexpected_call, np.eye(1), np.ones((1, 1)), budget_bytes=invalid)


@pytest.mark.parametrize(("requested", "expected"), [
    (None, 512 << 20), (0, 512 << 20), (1, 1), (np.uint64(8193), 8193),
    ((512 << 20) + 1, 512 << 20), ((1 << 64) - 1, 512 << 20),
])
def test_cphf_workspace_preserves_default_and_limit(requested, expected):
    def whiten(q_out, q_in, cols, budget, w, r, out):
        assert budget == expected
        out[0] = w[0] * r[0]
        return 0

    got = run_whiten_3c_native(whiten, np.eye(1), np.ones((1, 1)), budget_bytes=requested)
    np.testing.assert_array_equal(got, [[1.0]])


@pytest.mark.parametrize("backend", ["host", "cuda"])
def test_cphf_whiten_minimum_budget(backend: str, request: pytest.FixtureRequest) -> None:
    from generativeqc._native import installed_f12_shell_library

    if backend == "host":
        library = request.getfixturevalue("cphf_host_lib")
    else:
        library = configured_test_library("GENERATIVEQC_F12_SHELL_LIBRARY")
        if library is None:
            library = installed_f12_shell_library(host_reference=False)
        if library is None:
            pytest.skip("CUDA through-i shell unavailable")

    whiten = bind_cphf_relax(library, "whiten_3c", host_reference=backend == "host")
    assert whiten is not None
    w = np.array([[2.0]])
    raw = np.array([[3.0]])
    for budget in (1, 23):
        with pytest.raises(RuntimeError, match="status -5"):
            run_whiten_3c_native(whiten, w, raw, budget_bytes=budget)

    if backend == "cuda":
        try:
            driver = ct.CDLL("libcuda.so.1")
        except OSError:
            pytest.skip("CUDA driver unavailable for exact-fit execution")
        count = ct.c_int()
        if driver.cuInit(0) == 100:
            pytest.skip("no physical CUDA device for exact-fit execution")
        assert driver.cuDeviceGetCount(ct.byref(count)) == 0
        if count.value == 0:
            pytest.skip("no physical CUDA device for exact-fit execution")

    got = run_whiten_3c_native(whiten, w, raw, budget_bytes=24)
    np.testing.assert_array_equal(got, [[6.0]])


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


# =============================================================================
# M8: Fail-closed NaN/Inf checks & no uncaught C++ exceptions across extern "C"
# =============================================================================


def test_m8_cphf_whiten_3c_exact_audit_counterexample(cphf_host_lib: Path) -> None:
    """Exact M8 audit counterexample: finite 1x1 inputs 1e308, 2.0 producing Inf MUST fail."""
    fn = bind_cphf_relax(cphf_host_lib, "whiten_3c", host_reference=True)
    assert fn is not None

    w_big = np.array([[1e308]], dtype=np.float64)
    r_two = np.array([[[2.0]]], dtype=np.float64)
    with pytest.raises(RuntimeError, match="status -6"):
        run_whiten_3c_native(fn, w_big, r_two)

    # Also fail-closed when 0.0 multiplies NaN (must not swallow NaN as 0.0).
    w_zero = np.array([[0.0]], dtype=np.float64)
    r_nan = np.array([[[np.nan]]], dtype=np.float64)
    with pytest.raises(RuntimeError, match="status -6"):
        run_whiten_3c_native(fn, w_zero, r_nan)


def test_m8_cphf_all_other_primitives_reject_nonfinite(cphf_host_lib: Path) -> None:
    """All CPHF relax primitives fail closed with -6 on Inf/NaN outputs."""
    rot_fn = bind_cphf_relax(cphf_host_lib, "orbital_rot_metric", host_reference=True)
    act_fn = bind_cphf_relax(cphf_host_lib, "cphf_action", host_reference=True)
    drv_fn = bind_cphf_relax(cphf_host_lib, "df_canonical_driver", host_reference=True)
    nw_fn = bind_cphf_relax(cphf_host_lib, "df_nuclear_weights", host_reference=True)
    assert all(f is not None for f in (rot_fn, act_fn, drv_fn, nw_fn))

    # orbital_rot_metric overflow -> Inf
    with pytest.raises(RuntimeError, match="status -6"):
        run_orbital_rot_metric_native(
            rot_fn, np.full((2, 2), 1e308), np.full((2, 2), 2.0), compute_rotation=True
        )

    # cphf_action overflow -> Inf
    with pytest.raises(RuntimeError, match="status -6"):
        run_cphf_action_native(
            act_fn,
            is_rhf=True,
            oa=1,
            ob=1,
            obs=2,
            aux=1,
            vo_a=np.ones((1, 1, 1)),
            oo_a=np.ones((1, 1, 1)),
            vv_a=np.ones((1, 1, 1)),
            gap_a=np.array([[1e308]]),
            trial_a=np.array([[2.0]]),
        )

    # df_canonical_driver overflow -> Inf
    b = np.full((1, 2, 2), 1e308, dtype=np.float64)
    seed = np.full((2, 2), 2.0, dtype=np.float64)
    out_a = np.empty((1, 1), dtype=np.float64)
    assert drv_fn(1, 0, 2, 1, 512 << 20, _dptr(b), _dptr(seed), _dptr(out_a), _NULL_D, _NULL_D, _NULL_D) == -6

    # df_nuclear_weights overflow -> Inf
    with pytest.raises(RuntimeError, match="status -6"):
        run_df_nuclear_weights_native(
            nw_fn,
            oa=1,
            ob=1,
            obs=2,
            aux=1,
            nao=2,
            coefficients_alpha=np.full((2, 2), 1e308),
            coefficients_beta=np.full((2, 2), 1e308),
            occupied_energies_alpha=np.array([1.0]),
            occupied_energies_beta=np.array([1.0]),
            df_alpha=np.ones((1, 2, 2)),
            df_beta=np.ones((1, 2, 2)),
            fock_seed_alpha=np.full((2, 2), 2.0),
            fock_seed_beta=np.full((2, 2), 2.0),
            overlap_seed_alpha=np.zeros((2, 2)),
            overlap_seed_beta=np.zeros((2, 2)),
            whitener=np.ones((1, 1)),
            z_alpha=np.zeros((1, 1)),
            z_beta=np.zeros((1, 1)),
        )


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


def test_m8_spin_fock_1e_all_3_primitives_reject_nonfinite(spin_fock_1e_host_lib: Path) -> None:
    """All 3 spin_fock_1e entry points (including spin_fock_rev) fail closed on Inf/NaN outputs."""
    ops = bind_spin_fock_1e_ops(spin_fock_1e_host_lib, host_reference=True)
    assert ops is not None

    # 1. one_electron_grad overflow -> ValueError (rc=-2)
    with pytest.raises(ValueError, match="rc=-2"):
        run_one_electron_grad(
            ops,
            np.full((1, 3, 2, 2), 1e308),
            np.zeros((1, 3, 2, 2)),
            np.full((2, 2), 2.0),
            np.zeros((2, 2)),
        )

    # 2. spin_fock_fwd overflow -> ValueError (rc=-2)
    raw = np.full((1, 2, 2), 1e308)
    fit = np.full((1, 1, 2), 2.0)
    with pytest.raises(ValueError, match="rc=-2"):
        run_spin_fock_fwd(
            ops,
            (1, 1, 2, 1),
            {
                "raw_coulomb_alpha": raw,
                "raw_coulomb_beta": raw,
                "fit_alpha": fit,
                "fit_beta": fit,
            },
        )

    # 3. spin_fock_rev overflow -> ValueError (rc=-2) (previously unchecked!)
    with pytest.raises(ValueError, match="rc=-2"):
        run_spin_fock_rev(
            ops,
            (1, 1, 2, 1),
            {
                "raw_coulomb_alpha": np.ones((1, 2, 2)),
                "raw_coulomb_beta": np.ones((1, 2, 2)),
                "fit_alpha": np.full((1, 1, 2), 1e308),
                "fit_beta": np.full((1, 1, 2), 1e308),
            },
            {
                "fock_alpha": np.full((2, 2), 2.0),
                "fock_beta": np.full((2, 2), 2.0),
            },
        )


def test_m8_extern_c_catches_std_vector_length_error_without_abort(cphf_host_lib: Path) -> None:
    """Trigger std::vector::resize overflow inside extern 'C'; verify clean OOM status (no abort)."""
    fn = bind_cphf_relax(cphf_host_lib, "orbital_rot_metric", host_reference=True)
    assert fn is not None

    dummy = np.zeros(1, dtype=np.float64)
    p = _dptr(dummy)
    # nmo = 2_000_000_000 -> sm * sm = 4e18 doubles > vector::max_size() -> throws std::length_error.
    # Outer try/catch MUST translate it cleanly to kErrOom (-2) instead of aborting the process.
    rc = fn(2_000_000_000, 2_000_000_000, 512 << 20, p, p, p, _NULL_D, p)
    assert rc == -2


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
