"""Gate 1 verification for Patch A-2b Step 1 (zero residual TensorIR in cc_spatial_fixed_forces).

Verifies:
1. Native ``gqc_f12_cc_coulomb_fit_{fwd,rev}_host_reference`` matches TensorIR
   ``coulomb_fit_response`` to < 1e-12 across rectangular/3D shapes, non-symmetric
   whiteners, precomputed vs on-the-fly ``fit``, and forced multi-chunk workspace
   budgets (including 1-column chunks and non-divisible remainders).
2. Independent finite-difference directional derivative oracle validates ``metric_bar``
   and ``three_center_bar`` of ``gqc_f12_cc_coulomb_fit_rev_host_reference`` to < 1e-9.
3. Strict ``GENERATIVEQC_F12_REQUIRE_AOT_CCFIXED=1`` enforcement raises on every
   missing-library TensorIR fallback site (``coulomb_fit_forward``, ``coulomb_fit_reverse``,
   ``_accumulate_scalar_mo_pullback``, ``_accumulate_w_mo_pullback``,
   ``_accumulate_c_mo_pullback``) and succeeds with 0 TensorIR calls when the native
   library is provided.
4. End-to-end instrumented molecular verification proving 0 TensorIR calls inside
   ``spatial_rhf_fixed_cc_forces`` and ``fixed_ucc_source_forces`` (excluding external
   ``extended_fock_response`` / ``_orbital_sources`` / ``relaxed_f12_source_weights`` /
   ``one.nuclear_gradient``) AND < 1e-12 force parity vs full TensorIR fallback and
   across dense vs streaming F12 AO panels for RHF (all-active & frozen-core) and UHF.
"""

from __future__ import annotations

import inspect
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from conftest import f12_host_shell

from generativeqc import Atom, BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc._f12_cc_c_nuclear import _accumulate_c_mo_pullback
from generativeqc._f12_cc_fixed_sources_native import (
    REQUIRE_AOT_CCFIXED_ENV,
    bind_coulomb_fit_fwd,
    bind_coulomb_fit_rev,
    bind_fixed_pair_sources,
    coulomb_fit_forward,
    coulomb_fit_reverse,
    run_coulomb_fit_fwd_native,
    run_coulomb_fit_rev_native,
)
from generativeqc._f12_cc_scalar_nuclear import _accumulate_scalar_mo_pullback
from generativeqc._f12_cc_spin_sources import (
    spatial_rhf_cc_f12_fixed_sources,
    unrestricted_cc_f12_fixed_source_feeds,
    unrestricted_cc_f12_residual_sources,
)
from generativeqc._f12_cc_w_nuclear import _accumulate_w_mo_pullback
from generativeqc._f12_focal_cc_fixed import fixed_ucc_source_forces
from generativeqc._f12_focal_cc_sources import FocalUCCEquationSources
from generativeqc._f12_focal_rcc_f12b import (
    _build_spatial_rhf_fixed_sources,
    spatial_f12_fixed_source_cotangents,
    spatial_rhf_fixed_cc_forces,
)
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalSeries,
    MolecularFocalProvider,
    plan_singlepoint,
)
from generativeqc_compiler.mp2.f12_density_fitting import coulomb_fit_response
from generativeqc_compiler.tensor import Program, execute


@pytest.fixture(scope="module")
def host_sink() -> Path:
    return f12_host_shell()


def _make_nonsymmetric_whitener(rng: np.random.Generator, aux: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (J, W) where J is SPD and W is non-symmetric with W.T @ W = J^{-1}."""
    a = rng.normal(scale=0.35, size=(aux, aux))
    j_mat = a.T @ a + 0.5 * np.eye(aux)
    # Cholesky factor L of J^{-1}: L^T L = J^{-1} via upper-triangular inv(chol(J))^T
    # Multiply on the left by a random orthogonal matrix Q so W is genuinely non-symmetric.
    chol_j = np.linalg.cholesky(j_mat)
    inv_chol = np.linalg.solve(chol_j, np.eye(aux))
    q_orth, _ = np.linalg.qr(rng.normal(size=(aux, aux)))
    w_mat = np.ascontiguousarray(q_orth @ inv_chol, dtype=np.float64)
    np.testing.assert_allclose(w_mat.T @ w_mat, np.linalg.inv(j_mat), atol=1e-12, rtol=1e-12)
    return j_mat, w_mat


def _forbid_tensorir(*_args, **_kwargs):
    raise AssertionError("TensorIR runner must not be invoked on the AOT native path")


@pytest.mark.parametrize(
    "aux,shape_tail",
    [
        (5, (17,)),
        (8, (6, 11)),
        (13, (7, 15)),
    ],
)
def test_coulomb_fit_fwd_rev_parity_and_chunking(
    host_sink: Path, aux: int, shape_tail: tuple[int, ...]
) -> None:
    """Native Coulomb fit fwd & rev match TensorIR < 1e-12 across all chunk sizes."""
    fwd_fn = bind_coulomb_fit_fwd(host_sink, host_reference=True)
    rev_fn = bind_coulomb_fit_rev(host_sink, host_reference=True)
    assert fwd_fn is not None
    assert rev_fn is not None

    rng = np.random.default_rng(700 + aux * 31 + len(shape_tail))
    _j_mat, whitener = _make_nonsymmetric_whitener(rng, aux)
    raw = rng.normal(scale=0.4, size=(aux, *shape_tail))
    fit_bar = rng.normal(scale=0.3, size=(aux, *shape_tail))
    ncols = int(raw.size // aux)

    cpu_run = partial(execute, max_bytes=256 << 20)
    solve = coulomb_fit_response(aux, ncols)
    ref_fwd = cpu_run(
        Program({"fit": solve.outputs["fit"]}),
        {"metric_whitener": whitener, "three_center": raw.reshape(aux, ncols)},
    ).outputs["fit"].reshape(raw.shape)
    ref_rev = cpu_run(
        Program({k: solve.outputs[k] for k in ("three_center_bar", "metric_bar")}),
        {
            "metric_whitener": whitener,
            "three_center": raw.reshape(aux, ncols),
            "fit_bar": fit_bar.reshape(aux, ncols),
        },
    ).outputs
    ref_tc_bar = ref_rev["three_center_bar"].reshape(raw.shape)
    ref_m_bar = ref_rev["metric_bar"]

    # Hard budget check (M6): 1-byte budget must fail cleanly with status -5 (no silent over-allocation).
    with pytest.raises(RuntimeError, match="status -5"):
        run_coulomb_fit_fwd_native(fwd_fn, whitener, raw, max_workspace_bytes=1)
    with pytest.raises(RuntimeError, match="status -5"):
        run_coulomb_fit_rev_native(
            rev_fn, whitener, fit_bar, three_center=raw, fit_precomputed=None, max_workspace_bytes=1
        )

    # Test default budget (single chunk), exact 1-col minimal budget, and mid budget (remainder chunk).
    min_col_budget = 2 * aux * aux * 8 + 3 * aux * 8
    mid_budget = 2 * aux * aux * 8 + 3 * aux * 5 * 8
    for budget in (0, min_col_budget, mid_budget):
        fit_nat = run_coulomb_fit_fwd_native(
            fwd_fn, whitener, raw, max_workspace_bytes=budget
        )
        np.testing.assert_allclose(fit_nat, ref_fwd, atol=1e-13, rtol=1e-13)

        # Reverse with on-the-fly fit recomputation from three_center
        tc_bar_raw, m_bar_raw = run_coulomb_fit_rev_native(
            rev_fn,
            whitener,
            fit_bar,
            three_center=raw,
            fit_precomputed=None,
            max_workspace_bytes=budget,
        )
        np.testing.assert_allclose(tc_bar_raw, ref_tc_bar, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(m_bar_raw, ref_m_bar, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(m_bar_raw, m_bar_raw.T)

        # Reverse with precomputed fit
        tc_bar_pre, m_bar_pre = run_coulomb_fit_rev_native(
            rev_fn,
            whitener,
            fit_bar,
            three_center=None,
            fit_precomputed=fit_nat,
            max_workspace_bytes=budget,
        )
        np.testing.assert_allclose(tc_bar_pre, ref_tc_bar, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(m_bar_pre, ref_m_bar, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(m_bar_pre, m_bar_pre.T)


def test_coulomb_fit_rev_finite_difference_oracle(host_sink: Path) -> None:
    """Independent FD oracle verifies three_center_bar and symmetric metric_bar to < 1e-9."""
    fwd_fn = bind_coulomb_fit_fwd(host_sink, host_reference=True)
    rev_fn = bind_coulomb_fit_rev(host_sink, host_reference=True)
    assert fwd_fn is not None and rev_fn is not None

    rng = np.random.default_rng(12345)
    aux, ncols = 7, 19
    j_mat, whitener = _make_nonsymmetric_whitener(rng, aux)
    raw = rng.normal(scale=0.3, size=(aux, ncols))
    fit_bar = rng.normal(scale=0.3, size=(aux, ncols))

    fit = run_coulomb_fit_fwd_native(fwd_fn, whitener, raw, max_workspace_bytes=2048)
    tc_bar, m_bar = run_coulomb_fit_rev_native(
        rev_fn, whitener, fit_bar, fit_precomputed=fit, max_workspace_bytes=2048
    )

    def objective(j: np.ndarray, g: np.ndarray) -> float:
        return float(np.sum(fit_bar * np.linalg.solve(j, g)))

    eps = 1e-6
    dj = rng.normal(size=(aux, aux))
    dj = 0.5 * (dj + dj.T)
    dg = rng.normal(size=(aux, ncols))

    fd_j = (objective(j_mat + eps * dj, raw) - objective(j_mat - eps * dj, raw)) / (2 * eps)
    an_j = float(np.sum(m_bar * dj))
    assert abs(an_j - fd_j) < 1e-9

    fd_g = (objective(j_mat, raw + eps * dg) - objective(j_mat, raw - eps * dg)) / (2 * eps)
    an_g = float(np.sum(tc_bar * dg))
    assert abs(an_g - fd_g) < 1e-9


def test_coulomb_fit_native_error_handling(host_sink: Path) -> None:
    """Native Coulomb fit wrappers reject invalid shapes and missing inputs cleanly."""
    fwd_fn = bind_coulomb_fit_fwd(host_sink, host_reference=True)
    rev_fn = bind_coulomb_fit_rev(host_sink, host_reference=True)
    assert fwd_fn is not None and rev_fn is not None

    w = np.eye(4, dtype=np.float64)
    tc = np.ones((4, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="square"):
        run_coulomb_fit_fwd_native(fwd_fn, np.ones((4, 3)), tc)
    with pytest.raises(ValueError, match="leading axis"):
        run_coulomb_fit_fwd_native(fwd_fn, w, np.ones((5, 3)))
    with pytest.raises(ValueError, match="requires three_center or fit_precomputed"):
        run_coulomb_fit_rev_native(rev_fn, w, tc)
    with pytest.raises(ValueError, match="identical shape"):
        run_coulomb_fit_rev_native(rev_fn, w, tc, three_center=np.ones((4, 2)))


def test_require_aot_ccfixed_env_enforcement_and_zero_tir_pullbacks(
    host_sink: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GENERATIVEQC_F12_REQUIRE_AOT_CCFIXED=1 forbids TensorIR fallback at every site."""
    rng = np.random.default_rng(4242)
    oa, ob, obs, ext, aux = 3, 2, 6, 9, 7
    frozen = (1, 0)
    space = (oa, ob, obs, ext, aux)

    def sym(x: np.ndarray) -> np.ndarray:
        return 0.5 * (x + x.swapaxes(-1, -2))

    _j, whitener = _make_nonsymmetric_whitener(rng, aux)
    raw_coul = rng.normal(scale=0.2, size=(aux, obs, ext))
    fit_bar = rng.normal(scale=0.2, size=(aux, obs, ext))

    fock = {
        "fock_alpha": sym(rng.normal(scale=0.2, size=(ext, ext))),
        "fock_beta": sym(rng.normal(scale=0.2, size=(ext, ext))),
        "exchange_alpha": sym(rng.normal(scale=0.2, size=(ext, ext))),
        "exchange_beta": sym(rng.normal(scale=0.2, size=(ext, ext))),
    }
    fit_obs = {
        "alpha": rng.normal(scale=0.2, size=(aux, obs, ext)),
        "beta": rng.normal(scale=0.2, size=(aux, obs, ext)),
    }
    data = {
        "cross_overlap": rng.normal(scale=0.2, size=(ext, ext)),
        "source/fit_alpha": fit_obs["alpha"][:, :oa, :],
        "source/fit_beta": fit_obs["beta"][:, :ob, :],
        "source/raw_coulomb_alpha": raw_coul,
        "source/raw_coulomb_beta": rng.normal(scale=0.2, size=(aux, obs, ext)),
        "metric/geminal": sym(rng.normal(scale=0.2, size=(aux, aux))),
        "metric/geminal_coulomb": sym(rng.normal(scale=0.2, size=(aux, aux))),
        "metric/squared_geminal": sym(rng.normal(scale=0.2, size=(aux, aux))),
        "metric/double_commutator": sym(rng.normal(scale=0.2, size=(aux, aux))),
    }
    for op in ("geminal", "geminal_coulomb", "squared_geminal", "double_commutator"):
        data[f"raw/{op}/alpha"] = rng.normal(scale=0.2, size=(aux, oa, ext))
        data[f"raw/{op}/beta"] = rng.normal(scale=0.2, size=(aux, ob, ext))

    w_weights = {
        "w_vo_aaaa": rng.normal(scale=0.1, size=(oa - frozen[0], oa - frozen[0], obs - oa, oa - frozen[0])),
        "w_vv_abab": rng.normal(scale=0.1, size=(oa - frozen[0], ob - frozen[1], obs - oa, obs - ob)),
    }
    c_weights = {
        "c_f12_aaaa": rng.normal(scale=0.1, size=(oa - frozen[0], oa - frozen[0], obs - oa, obs - oa)),
        "c_f12_abab": rng.normal(scale=0.1, size=(oa - frozen[0], ob - frozen[1], obs - oa, obs - ob)),
    }

    monkeypatch.setenv(REQUIRE_AOT_CCFIXED_ENV, "1")

    # 1. Missing native_library must fail loudly at ALL 5 sites!
    with pytest.raises(RuntimeError, match=REQUIRE_AOT_CCFIXED_ENV):
        coulomb_fit_forward(
            whitener, raw_coul, native_library=None, host_reference=True, run=_forbid_tensorir
        )
    with pytest.raises(RuntimeError, match=REQUIRE_AOT_CCFIXED_ENV):
        coulomb_fit_reverse(
            whitener, raw_coul, fit_bar, native_library=None, host_reference=True, run=_forbid_tensorir
        )
    dbar = {k: np.zeros_like(v) for k, v in data.items()}
    fbar = {k: np.zeros_like(v) for k, v in fock.items()}
    fobs_bar = {s: np.zeros_like(v) for s, v in fit_obs.items()}
    with pytest.raises(RuntimeError, match=REQUIRE_AOT_CCFIXED_ENV):
        _accumulate_scalar_mo_pullback(
            space, data, fock, 1.0, _forbid_tensorir, dbar, fbar, frozen_occupied=frozen, native_library=None
        )
    with pytest.raises(RuntimeError, match=REQUIRE_AOT_CCFIXED_ENV):
        _accumulate_w_mo_pullback(
            space, data, fit_obs, w_weights, _forbid_tensorir, dbar, fobs_bar, frozen_occupied=frozen, native_library=None
        )
    with pytest.raises(RuntimeError, match=REQUIRE_AOT_CCFIXED_ENV):
        _accumulate_c_mo_pullback(
            space, data, fock, c_weights, _forbid_tensorir, dbar, fbar, frozen_occupied=frozen, native_library=None
        )

    # 2. With host_sink provided, ALL 5 sites execute natively with _forbid_tensorir (0 TensorIR calls!).
    fit_out = coulomb_fit_forward(
        whitener, raw_coul, native_library=host_sink, host_reference=True, run=_forbid_tensorir
    )
    tc_bar, m_bar = coulomb_fit_reverse(
        whitener,
        raw_coul,
        fit_bar,
        fit_precomputed=fit_out,
        native_library=host_sink,
        host_reference=True,
        run=_forbid_tensorir,
    )
    assert fit_out.shape == raw_coul.shape
    assert tc_bar.shape == raw_coul.shape and m_bar.shape == (aux, aux)

    e_v, e_b, e_x = _accumulate_scalar_mo_pullback(
        space, data, fock, 1.0, _forbid_tensorir, dbar, fbar,
        frozen_occupied=frozen, native_library=host_sink, host_reference=True,
    )
    assert np.isfinite(e_v + e_b - e_x)
    _accumulate_w_mo_pullback(
        space, data, fit_obs, w_weights, _forbid_tensorir, dbar, fobs_bar,
        frozen_occupied=frozen, native_library=host_sink, host_reference=True,
    )
    _accumulate_c_mo_pullback(
        space, data, fock, c_weights, _forbid_tensorir, dbar, fbar,
        frozen_occupied=frozen, native_library=host_sink, host_reference=True,
    )


def _compact_water_basis(beta: float = 1.0) -> FocalBasis:
    def make_basis(name: str, elems: tuple[ElementBasis, ...]) -> BasisSet:
        prov = BasisProvenance(source=name, version="1", license="MIT", checksum="0" * 64)
        return BasisSet(name, elems, prov, representation="spherical")

    def sh(l: int, exps: tuple[float, ...], coeffs: tuple[float, ...]) -> BasisShell:
        return BasisShell(l, tuple(str(x) for x in exps), (tuple(str(c) for c in coeffs),))

    c_orb = ElementBasis(
        6,
        (
            sh(0, (71.616837, 13.045096, 3.5305122), (0.15432897, 0.53532814, 0.44463454)),
            sh(0, (2.9412494, 0.6834831), (-0.09996723, 0.39951283)),
            sh(0, (0.2222899,), (1.0,)),
            sh(1, (2.9412494, 0.6834831), (0.15591627, 0.60768372)),
            sh(1, (0.2222899,), (1.0,)),
        ),
    )
    o_orb = ElementBasis(
        8,
        (
            sh(0, (130.70932, 23.808861, 6.4436083), (0.15432897, 0.53532814, 0.44463454)),
            sh(0, (5.0331513, 1.1695961), (-0.09996723, 0.39951283)),
            sh(0, (0.3803890,), (1.0,)),
            sh(1, (5.0331513, 1.1695961), (0.15591627, 0.60768372)),
            sh(1, (0.3803890,), (1.0,)),
        ),
    )
    h_orb = ElementBasis(
        1,
        (
            sh(0, (3.42525091, 0.62391373), (0.15432897, 0.53532814)),
            sh(0, (0.16885540,), (1.0,)),
        ),
    )
    c_cabs = ElementBasis(6, (sh(1, (1.1,), (1.0,)), sh(2, (0.9,), (1.0,))))
    o_cabs = ElementBasis(8, (sh(1, (1.4,), (1.0,)), sh(2, (1.2,), (1.0,))))
    h_cabs = ElementBasis(1, (sh(1, (0.9,), (1.0,)),))
    c_jk = ElementBasis(
        6,
        (
            sh(0, (12.0,), (1.0,)),
            sh(0, (2.2,), (1.0,)),
            sh(0, (0.45,), (1.0,)),
            sh(1, (1.7,), (1.0,)),
            sh(2, (1.1,), (1.0,)),
        ),
    )
    o_jk = ElementBasis(
        8,
        (
            sh(0, (18.0,), (1.0,)),
            sh(0, (3.0,), (1.0,)),
            sh(0, (0.6,), (1.0,)),
            sh(1, (2.2,), (1.0,)),
            sh(2, (1.5,), (1.0,)),
        ),
    )
    h_jk = ElementBasis(1, (sh(0, (4.0,), (1.0,)), sh(0, (0.7,), (1.0,)), sh(1, (1.1,), (1.0,))))
    c_f12 = ElementBasis(
        6,
        (
            sh(0, (10.0,), (1.0,)),
            sh(0, (1.8,), (1.0,)),
            sh(0, (0.4,), (1.0,)),
            sh(1, (1.5,), (1.0,)),
            sh(2, (1.0,), (1.0,)),
        ),
    )
    o_f12 = ElementBasis(
        8,
        (
            sh(0, (15.0,), (1.0,)),
            sh(0, (2.4,), (1.0,)),
            sh(0, (0.5,), (1.0,)),
            sh(1, (1.9,), (1.0,)),
            sh(2, (1.3,), (1.0,)),
        ),
    )
    h_f12 = ElementBasis(1, (sh(0, (3.5,), (1.0,)), sh(0, (0.6,), (1.0,)), sh(1, (1.0,), (1.0,))))
    return FocalBasis(
        cardinal=2,
        series_id=f"test-a2b-beta{beta}",
        orbital=make_basis("orb", (c_orb, o_orb, h_orb)),
        jk_fit=make_basis("jk", (c_jk, o_jk, h_jk)),
        mp2_fit=make_basis("mp2", (c_jk, o_jk, h_jk)),
        f12_fit=make_basis("f12", (c_f12, o_f12, h_f12)),
        cabs=make_basis("cabs", (c_cabs, o_cabs, h_cabs)),
        beta=beta,
    )


_IN_SCOPE_OUTPUT_KEYS = frozenset(
    {
        "fit",
        "three_center_bar",
        "metric_bar",
        "w_vo",
        "w_vv",
        "c_f12",
        "fixed_geminal_energy",
        "pair_energy",
        "scalar",
    }
)
_IN_SCOPE_OUTPUT_PREFIXES = (
    "bar_fit",
    "bar_raw_",
    "bar_res_",
    "bar_fock",
    "bar_exchange",
    "bar_metric_",
    "bar_cross_overlap",
    "w_vo",
    "w_vv",
    "c_f12",
)
_EXTERNAL_OUTPUT_KEYS = frozenset(
    {
        "rotation_gradient",
        "overlap_bar",
        "hcore_bar",
        "gradient",
        "fock_alpha",
        "fock_beta",
        "exchange_alpha",
        "exchange_beta",
        "cross_overlap",
    }
)


def _classify_tensorir_program(program: object) -> tuple[str, str]:
    """Classify a TensorIR Program semantically by its outputs/provenance rather than filenames."""
    outputs = tuple(sorted(getattr(program, "outputs", {}).keys()))
    prov = dict(getattr(program, "provenance", {}) or {})
    method = str(prov.get("method", "")).lower()
    desc = f"outputs={outputs}, method={method or 'unspecified'}"
    if any(
        key in _IN_SCOPE_OUTPUT_KEYS
        or key.startswith(_IN_SCOPE_OUTPUT_PREFIXES)
        for key in outputs
    ) or any(tag in method for tag in ("coulomb fit", "fixed", "ccsd-f12", "f12")):
        return "in_scope", desc
    if any(key in _EXTERNAL_OUTPUT_KEYS for key in outputs):
        return "external", desc
    return "other", desc


@pytest.mark.parametrize(
    "label,root_id,atoms,charge,mult,corr_space",
    [
        (
            "h2o_rhf_all",
            "rhf",
            (
                Atom("O", (0.0, 0.0, 0.0)),
                Atom("H", (1.43, 0.0, 1.11)),
                Atom("H", (-1.43, 0.0, 1.11)),
            ),
            0,
            1,
            "all-active",
        ),
        (
            "h2o_rhf_fc",
            "rhf",
            (
                Atom("O", (0.0, 0.0, 0.0)),
                Atom("H", (1.43, 0.0, 1.11)),
                Atom("H", (-1.43, 0.0, 1.11)),
            ),
            0,
            1,
            "frozen-core",
        ),
        (
            "oh_uhf_fc",
            "uhf",
            (
                Atom("O", (0.0, 0.0, 0.0)),
                Atom("H", (0.0, 0.0, 1.83)),
            ),
            0,
            2,
            "frozen-core",
        ),
    ],
)
def test_cc_spatial_fixed_forces_zero_in_scope_tensorir_and_parity(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    root_id: str,
    atoms: tuple[Atom, ...],
    charge: int,
    mult: int,
    corr_space: str,
) -> None:
    """Prove 0 in-scope TensorIR calls and < 1e-12 parity vs TensorIR in fixed CC forces."""
    import generativeqc_compiler.tensor as tensor_mod

    basis = _compact_water_basis(beta=1.0)
    plan = plan_singlepoint(
        atoms,
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        charge=charge,
        multiplicity=mult,
        root_id=root_id,
        correlation_space_id=corr_space,
        backend="cpu_reference",
    )
    req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")

    prov = MolecularFocalProvider(shell_library=host_sink, budget_mib=1024)
    fixed = prov.ucc_fixed_sources(req)
    eq_sources = FocalUCCEquationSources(fixed, basis.mp2_fit.identity, {})
    ref = fixed.reference
    oa, ob = (ref.occupied[i] for i in ref._pair_spin_indices(req))
    fa, fb = fixed.frozen_occupied
    n = ref.coefficients.shape[-1]

    rng = np.random.default_rng(9001 + len(label))
    if root_id == "rhf":
        o_act, n_vir = oa - fa, n - oa
        t1 = rng.normal(0.0, 0.04, size=(o_act, n_vir))
        l1 = rng.normal(0.0, 0.04, size=(o_act, n_vir))
        t2_raw = rng.normal(0.0, 0.03, size=(o_act, o_act, n_vir, n_vir))
        l2_raw = rng.normal(0.0, 0.03, size=(o_act, o_act, n_vir, n_vir))
        t2 = 0.5 * (t2_raw + t2_raw.transpose(1, 0, 3, 2))
        l2 = 0.5 * (l2_raw + l2_raw.transpose(1, 0, 3, 2))
        fixed_weights = spatial_f12_fixed_source_cotangents(t1, t2, l1, l2)
        # Filter to exact active keys required by UCCSDLayout (for small o_act if needed).
        from generativeqc_compiler.cc.uccsd_equations import build_f12_program
        from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout

        cc_prog = build_f12_program(UCCSDLayout(oa - fa, ob - fb, n - oa, n - ob))
        req_keys = {
            nd.attrs["name"]
            for nd in cc_prog.live_nodes
            if nd.op == "input" and nd.attrs["name"] in fixed.feeds
        }
        fixed_weights = {k: v for k, v in fixed_weights.items() if k in req_keys}
    else:
        from generativeqc_compiler.cc.uccsd_equations import build_f12_program
        from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout

        cc_prog = build_f12_program(UCCSDLayout(oa - fa, ob - fb, n - oa, n - ob))
        req_keys = sorted(
            nd.attrs["name"]
            for nd in cc_prog.live_nodes
            if nd.op == "input" and nd.attrs["name"] in fixed.feeds
        )
        fixed_weights = {
            k: (
                np.asarray(1.0, dtype=np.float64)
                if k == "fixed_geminal_energy"
                else rng.normal(0.0, 0.04, size=fixed.feeds[k].shape)
            )
            for k in req_keys
        }

    # Baseline reference: force ALL 5 CC-fixed native bindings off -> full TensorIR path.
    # Also verify our stack-based instrumentation positively detects those in-scope calls.
    monkeypatch.delenv(REQUIRE_AOT_CCFIXED_ENV, raising=False)
    monkeypatch.setenv("GENERATIVEQC_F12_STREAM_PANELS", "0")
    real_execute = tensor_mod.execute
    in_scope_calls: list[tuple[str, dict]] = []
    external_calls: list[str] = []
    other_calls: list[str] = []

    def instrumented_execute(program, inputs, *args, **kwargs):
        bucket, site = _classify_tensorir_program(program)
        if bucket == "in_scope":
            in_scope_calls.append((site, dict(program.provenance)))
        elif bucket == "external":
            external_calls.append(site)
        else:
            other_calls.append(site)
        return real_execute(program, inputs, *args, **kwargs)

    monkeypatch.setattr(tensor_mod, "execute", instrumented_execute)

    with monkeypatch.context() as mp:
        mp.setattr(
            "generativeqc._f12_cc_fixed_sources_native.bind_coulomb_fit_fwd",
            lambda *_a, **_kw: None,
        )
        mp.setattr(
            "generativeqc._f12_cc_fixed_sources_native.bind_coulomb_fit_rev",
            lambda *_a, **_kw: None,
        )
        mp.setattr(
            "generativeqc._f12_cc_pair_pullback_native.bind_pair_pullback",
            lambda *_a, **_kw: None,
        )
        prov._pair_source_cache.clear()
        in_scope_calls.clear()
        ref_ucc_forces = fixed_ucc_source_forces(prov, req, eq_sources, fixed_weights)
        assert len(in_scope_calls) > 0, "Detector sanity check: fallback must record in-scope calls"

    # Now enforce GENERATIVEQC_F12_REQUIRE_AOT_CCFIXED=1 with native host_sink active!
    monkeypatch.setenv(REQUIRE_AOT_CCFIXED_ENV, "1")

    for stream_flag in ("0", "1"):
        monkeypatch.setenv("GENERATIVEQC_F12_STREAM_PANELS", stream_flag)
        prov._pair_source_cache.clear()
        in_scope_calls.clear()
        external_calls.clear()
        other_calls.clear()
        aot_ucc_forces = fixed_ucc_source_forces(prov, req, eq_sources, fixed_weights)
        assert in_scope_calls == [], (
            f"fixed_ucc_source_forces ({label}, stream={stream_flag}) invoked in-scope TensorIR: "
            f"{in_scope_calls}"
        )
        assert other_calls == [], f"Unclassified TensorIR calls: {other_calls}"
        assert external_calls == [], f"Unexpected external TensorIR calls: {external_calls}"
        np.testing.assert_allclose(aot_ucc_forces, ref_ucc_forces, atol=1e-12, rtol=1e-12)

        if root_id == "rhf":
            prov._pair_source_cache.clear()
            in_scope_calls.clear()
            external_calls.clear()
            other_calls.clear()
            aot_spatial_forces = spatial_rhf_fixed_cc_forces(
                prov,
                req,
                ref,
                cabs_rank=fixed.cabs_rank,
                fixed_geminal_energy=float(fixed.feeds["fixed_geminal_energy"]),
                t1=t1,
                t2=t2,
                lambda1=l1,
                lambda2=l2,
                frozen_occupied=fa,
            )
            assert in_scope_calls == [], (
                f"spatial_rhf_fixed_cc_forces ({label}, stream={stream_flag}) invoked in-scope "
                f"TensorIR: {in_scope_calls}"
            )
            assert other_calls == [], f"Unclassified TensorIR calls: {other_calls}"
            assert external_calls == [], f"Unexpected external TensorIR calls: {external_calls}"
            np.testing.assert_allclose(aot_spatial_forces, ref_ucc_forces, atol=1e-12, rtol=1e-12)


def _make_synthetic_spin_frame(
    rng: np.random.Generator,
    oa: int,
    ob: int,
    obs: int,
    ext: int,
    aux: int,
    *,
    spin_symmetric: bool,
) -> tuple[tuple[int, int, int, int, int], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Build a synthetic DF frame with non-diagonal active F_oo to stress-test B/X/W/C."""

    def sym(x: np.ndarray) -> np.ndarray:
        return 0.5 * (x + x.swapaxes(-1, -2))

    _j, whitener = _make_nonsymmetric_whitener(rng, aux)
    fock_a = sym(rng.normal(scale=0.25, size=(ext, ext)))
    exch_a = sym(rng.normal(scale=0.25, size=(ext, ext)))
    raw_coul_a = rng.normal(scale=0.2, size=(aux, obs, ext))
    fit_a = (whitener.T @ (whitener @ raw_coul_a[:, :oa, :].reshape(aux, -1))).reshape(aux, oa, ext)
    if spin_symmetric:
        assert oa == ob
        fock_b = fock_a.copy()
        exch_b = exch_a.copy()
        raw_coul_b = raw_coul_a.copy()
        fit_b = fit_a.copy()
        cross = sym(rng.normal(scale=0.2, size=(ext, ext)))
    else:
        fock_b = sym(rng.normal(scale=0.25, size=(ext, ext)))
        exch_b = sym(rng.normal(scale=0.25, size=(ext, ext)))
        raw_coul_b = rng.normal(scale=0.2, size=(aux, obs, ext))
        fit_b = (
            (whitener.T @ (whitener @ raw_coul_b[:, :ob, :].reshape(aux, -1))).reshape(aux, ob, ext)
            if ob > 0
            else np.zeros((aux, 0, ext), dtype=np.float64)
        )
        cross = rng.normal(scale=0.2, size=(ext, ext))

    fock = {
        "fock_alpha": fock_a,
        "fock_beta": fock_b,
        "exchange_alpha": exch_a,
        "exchange_beta": exch_b,
    }
    data: dict[str, np.ndarray] = {
        "cross_overlap": cross,
        "source/fit_alpha": fit_a,
        "source/fit_beta": fit_b,
        "source/raw_coulomb_alpha": raw_coul_a,
        "source/raw_coulomb_beta": raw_coul_b,
    }
    for op in ("geminal", "geminal_coulomb", "squared_geminal", "double_commutator"):
        data[f"metric/{op}"] = sym(rng.normal(scale=0.2, size=(aux, aux)))
        raw_a = rng.normal(scale=0.2, size=(aux, oa, ext))
        data[f"raw/{op}/alpha"] = raw_a
        data[f"raw/{op}/beta"] = (
            raw_a.copy()
            if spin_symmetric
            else rng.normal(scale=0.2, size=(aux, ob, ext))
        )
    return (oa, ob, obs, ext, aux), data, fock, whitener


@pytest.mark.parametrize(
    "label,oa,ob,obs,ext,aux,frozen,spin_sym",
    [
        ("rhf_all", 3, 3, 6, 9, 8, (0, 0), True),
        ("rhf_fc", 4, 4, 7, 10, 9, (1, 1), True),
        ("uhf_general_fc", 4, 3, 7, 10, 9, (1, 0), False),
        ("uhf_o_act_b_eq_1", 4, 2, 7, 10, 8, (1, 1), False),
        ("uhf_ob_eq_0", 3, 0, 6, 9, 7, (0, 0), False),
        ("uhf_1e_empty_pairs", 1, 0, 5, 8, 6, (0, 0), False),
    ],
)
def test_fixed_pair_sources_synthetic_edge_cases_and_zero_jit(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    oa: int,
    ob: int,
    obs: int,
    ext: int,
    aux: int,
    frozen: tuple[int, int],
    spin_sym: bool,
) -> None:
    """Verify <1e-12 parity vs TensorIR and 0 TensorIR calls across UHF/RHF edge cases."""
    assert bind_fixed_pair_sources(host_sink, host_reference=True) is not None
    rng = np.random.default_rng(2026 + oa * 17 + ob * 31 + frozen[0] * 7 + int(spin_sym))
    space, data, fock, whitener = _make_synthetic_spin_frame(
        rng, oa, ob, obs, ext, aux, spin_symmetric=spin_sym
    )
    cpu_run = partial(execute, max_bytes=256 << 20)

    # 1. Reference TensorIR baseline (explicit native_library=None).
    monkeypatch.delenv(REQUIRE_AOT_CCFIXED_ENV, raising=False)
    ref_feeds = unrestricted_cc_f12_fixed_source_feeds(
        space, data, fock, whitener, cpu_run, frozen_occupied=frozen, native_library=None
    )
    ref_res_blocks, ref_res_v = unrestricted_cc_f12_residual_sources(
        space, data, fock, whitener, cpu_run, frozen_occupied=frozen, native_library=None
    )
    if spin_sym:
        ref_rhf = spatial_rhf_cc_f12_fixed_sources(
            space,
            data,
            fock,
            whitener,
            cpu_run,
            frozen_occupied=frozen[0],
            native_library=None,
        )

    # 2. Enforce GENERATIVEQC_F12_REQUIRE_AOT_CCFIXED=1 + _forbid_tensorir sentinel.
    monkeypatch.setenv(REQUIRE_AOT_CCFIXED_ENV, "1")
    has_pairs = (oa - frozen[0] >= 2) or (ob - frozen[1] >= 2) or (
        oa > frozen[0] and ob > frozen[1]
    )

    if has_pairs:
        with pytest.raises(RuntimeError, match=REQUIRE_AOT_CCFIXED_ENV):
            unrestricted_cc_f12_fixed_source_feeds(
                space,
                data,
                fock,
                whitener,
                _forbid_tensorir,
                frozen_occupied=frozen,
                native_library=None,
            )
        if spin_sym:
            with pytest.raises(RuntimeError, match=REQUIRE_AOT_CCFIXED_ENV):
                spatial_rhf_cc_f12_fixed_sources(
                    space,
                    data,
                    fock,
                    whitener,
                    _forbid_tensorir,
                    frozen_occupied=frozen[0],
                    native_library=None,
                )

    # 3. Explicit host_sink + hard-budget rejection (M6) + exact default budget + 0 TensorIR!
    if has_pairs:
        with pytest.raises(RuntimeError, match="status -5"):
            unrestricted_cc_f12_fixed_source_feeds(
                space,
                data,
                fock,
                whitener,
                _forbid_tensorir,
                frozen_occupied=frozen,
                native_library=host_sink,
                host_reference=True,
                max_workspace_bytes=1,
            )
    for budget in (0,):
        aot_feeds = unrestricted_cc_f12_fixed_source_feeds(
            space,
            data,
            fock,
            whitener,
            _forbid_tensorir,
            frozen_occupied=frozen,
            native_library=host_sink,
            host_reference=True,
            max_workspace_bytes=budget,
        )
        assert set(aot_feeds.keys()) == set(ref_feeds.keys())
        for key, ref_val in ref_feeds.items():
            np.testing.assert_allclose(
                aot_feeds[key], ref_val, atol=1e-12, rtol=1e-12, err_msg=f"{label}:{key}"
            )

        aot_res_blocks, aot_res_v = unrestricted_cc_f12_residual_sources(
            space,
            data,
            fock,
            whitener,
            _forbid_tensorir,
            frozen_occupied=frozen,
            native_library=host_sink,
            host_reference=True,
            max_workspace_bytes=budget,
        )
        assert abs(aot_res_v - ref_res_v) < 1e-12
        for key, ref_val in ref_res_blocks.items():
            np.testing.assert_allclose(
                aot_res_blocks[key], ref_val, atol=1e-12, rtol=1e-12, err_msg=f"{label}:res:{key}"
            )

        if spin_sym:
            aot_rhf = spatial_rhf_cc_f12_fixed_sources(
                space,
                data,
                fock,
                whitener,
                _forbid_tensorir,
                frozen_occupied=frozen[0],
                native_library=host_sink,
                host_reference=True,
                max_workspace_bytes=budget,
            )
            for idx_item, name in enumerate(("w_vo", "w_vv", "c_f12", "fixed_geminal_energy")):
                np.testing.assert_allclose(
                    aot_rhf[idx_item],
                    ref_rhf[idx_item],
                    atol=1e-12,
                    rtol=1e-12,
                    err_msg=f"{label}:rhf:{name}",
                )

    # 4. Auto-discovery (omitting native_library) finds loaded host_sink with 0 TensorIR calls!
    auto_feeds = unrestricted_cc_f12_fixed_source_feeds(
        space,
        data,
        fock,
        whitener,
        _forbid_tensorir,
        frozen_occupied=frozen,
    )
    for key, ref_val in ref_feeds.items():
        np.testing.assert_allclose(auto_feeds[key], ref_val, atol=1e-12, rtol=1e-12)
    if spin_sym:
        auto_rhf = spatial_rhf_cc_f12_fixed_sources(
            space,
            data,
            fock,
            whitener,
            _forbid_tensorir,
            frozen_occupied=frozen[0],
        )
        for idx_item in range(4):
            np.testing.assert_allclose(auto_rhf[idx_item], ref_rhf[idx_item], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "label,root_id,atoms,charge,mult,corr_space",
    [
        (
            "h2o_rhf_all",
            "rhf",
            (
                Atom("O", (0.0, 0.0, 0.0)),
                Atom("H", (1.43, 0.0, 1.11)),
                Atom("H", (-1.43, 0.0, 1.11)),
            ),
            0,
            1,
            "all-active",
        ),
        (
            "h2o_rhf_fc",
            "rhf",
            (
                Atom("O", (0.0, 0.0, 0.0)),
                Atom("H", (1.43, 0.0, 1.11)),
                Atom("H", (-1.43, 0.0, 1.11)),
            ),
            0,
            1,
            "frozen-core",
        ),
        (
            "ch4_asym_rhf_all",
            "rhf",
            (
                Atom("C", (0.0, 0.0, 0.0)),
                Atom("H", (1.18, 1.15, 1.12)),
                Atom("H", (-1.20, -1.14, 1.16)),
                Atom("H", (-1.15, 1.22, -1.10)),
                Atom("H", (1.24, -1.18, -1.19)),
            ),
            0,
            1,
            "all-active",
        ),
        (
            "ch4_asym_rhf_fc",
            "rhf",
            (
                Atom("C", (0.0, 0.0, 0.0)),
                Atom("H", (1.18, 1.15, 1.12)),
                Atom("H", (-1.20, -1.14, 1.16)),
                Atom("H", (-1.15, 1.22, -1.10)),
                Atom("H", (1.24, -1.18, -1.19)),
            ),
            0,
            1,
            "frozen-core",
        ),
        (
            "oh_uhf_all",
            "uhf",
            (
                Atom("O", (0.0, 0.0, 0.0)),
                Atom("H", (0.0, 0.0, 1.83)),
            ),
            0,
            2,
            "all-active",
        ),
        (
            "oh_uhf_fc",
            "uhf",
            (
                Atom("O", (0.0, 0.0, 0.0)),
                Atom("H", (0.0, 0.0, 1.83)),
            ),
            0,
            2,
            "frozen-core",
        ),
    ],
)
def test_molecular_fixed_sources_zero_in_scope_tensorir_and_parity(
    host_sink: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    root_id: str,
    atoms: tuple[Atom, ...],
    charge: int,
    mult: int,
    corr_space: str,
) -> None:
    """End-to-end molecular verification: 0 TensorIR in _f12_cc_spin_sources.py and <1e-12 parity."""
    import generativeqc_compiler.tensor as tensor_mod

    basis = _compact_water_basis(beta=1.0)
    plan = plan_singlepoint(
        atoms,
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        charge=charge,
        multiplicity=mult,
        root_id=root_id,
        correlation_space_id=corr_space,
        backend="cpu_reference",
    )
    req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
    prov = MolecularFocalProvider(shell_library=host_sink, budget_mib=1024)

    real_execute = tensor_mod.execute
    in_scope_calls: list[tuple[str, dict]] = []
    external_calls: list[str] = []
    other_calls: list[str] = []

    def instrumented_execute(program, inputs, *args, **kwargs):
        bucket, site = _classify_tensorir_program(program)
        if bucket == "in_scope":
            in_scope_calls.append((site, dict(program.provenance)))
        elif bucket == "external":
            external_calls.append(site)
        else:
            other_calls.append(site)
        return real_execute(program, inputs, *args, **kwargs)

    monkeypatch.setattr(tensor_mod, "execute", instrumented_execute)

    # 1. Baseline TensorIR reference (force native CC-fixed bindings off).
    monkeypatch.delenv(REQUIRE_AOT_CCFIXED_ENV, raising=False)
    with monkeypatch.context() as mp:
        mp.setattr(
            "generativeqc._f12_cc_fixed_sources_native.bind_coulomb_fit_fwd",
            lambda *_a, **_kw: None,
        )
        mp.setattr(
            "generativeqc._f12_cc_fixed_sources_native.bind_fixed_pair_sources",
            lambda *_a, **_kw: None,
        )
        mp.setattr(
            "generativeqc._f12_cc_spin_sources.bind_fixed_pair_sources",
            lambda *_a, **_kw: None,
        )
        in_scope_calls.clear()
        ref_ucc = prov.ucc_fixed_sources(req)
        assert len(in_scope_calls) > 0, f"{label}: fallback must record in-scope TensorIR calls"
        ref_rhf = None
        if root_id == "rhf":
            in_scope_calls.clear()
            ref_rhf = _build_spatial_rhf_fixed_sources(
                prov, req, ref_ucc.reference, expected_cabs_rank=ref_ucc.cabs_rank, derivatives=False
            )
            assert len(in_scope_calls) > 0, f"{label}: RHF fallback must record in-scope calls"

    # 2. Strict AOT path under GENERATIVEQC_F12_REQUIRE_AOT_CCFIXED=1:
    #    ZERO in-scope TensorIR calls AND < 1e-12 parity on every feed/block!
    monkeypatch.setenv(REQUIRE_AOT_CCFIXED_ENV, "1")
    prov._pair_source_cache.clear()
    in_scope_calls.clear()
    external_calls.clear()
    other_calls.clear()
    aot_ucc = prov.ucc_fixed_sources(req)
    assert in_scope_calls == [], (
        f"prov.ucc_fixed_sources ({label}) invoked in-scope TensorIR: {in_scope_calls}"
    )
    assert other_calls == [], f"Unclassified TensorIR calls: {other_calls}"
    assert set(aot_ucc.feeds.keys()) == set(ref_ucc.feeds.keys())
    for key, ref_val in ref_ucc.feeds.items():
        np.testing.assert_allclose(
            aot_ucc.feeds[key], ref_val, atol=1e-12, rtol=1e-12, err_msg=f"{label}:ucc:{key}"
        )

    if root_id == "rhf":
        assert ref_rhf is not None
        prov._pair_source_cache.clear()
        in_scope_calls.clear()
        external_calls.clear()
        other_calls.clear()
        aot_rhf = _build_spatial_rhf_fixed_sources(
            prov, req, aot_ucc.reference, expected_cabs_rank=aot_ucc.cabs_rank, derivatives=False
        )
        assert in_scope_calls == [], (
            f"_build_spatial_rhf_fixed_sources ({label}) invoked in-scope TensorIR: {in_scope_calls}"
        )
        assert other_calls == [], f"Unclassified TensorIR calls: {other_calls}"
        for idx_item, name in enumerate(("w_vo", "w_vv", "c_f12", "fixed_geminal_energy")):
            np.testing.assert_allclose(
                aot_rhf[idx_item],
                ref_rhf[idx_item],
                atol=1e-12,
                rtol=1e-12,
                err_msg=f"{label}:rhf:{name}",
            )
        assert aot_rhf[4] == ref_rhf[4] and aot_rhf[5] == ref_rhf[5]
