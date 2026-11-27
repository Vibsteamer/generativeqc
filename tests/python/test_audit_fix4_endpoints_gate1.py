"""Gate 1 regression & counterexample tests for Audit Fix-4 (Python endpoints, budget, AOT, Delta-CV, hygiene)."""

from __future__ import annotations

import ctypes as ct
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generativeqc import Atom, Primitive, Shell
from generativeqc._f12_focal_cv import (
    _paired_cv_fc_mp2_requests,
    _paired_cv_fc_requests,
    _solve_df_reference_from_focal_sources,
    conventional_mp2_request,
    default_valence_correlation_space_id,
)
from generativeqc._f12_focal_molecular import (
    MolecularFocalProvider,
)
from generativeqc._f12_focal_rcc_f12b import spatial_rhf_ccsdf12b_v1
from generativeqc._f12_focal_reference import (
    FocalReference,
    FocalReferenceSeed,
)
from generativeqc._f12_focal_tplus_sources import FocalOccupiedTriples, OccupiedMP2F12
from generativeqc._f12_ram_hygiene import (
    STRICT_GPU_ONLY_ENV,
    capture_hybrid_phases,
    capture_ram_trace,
    disclose_host_assisted_stage,
    record_ram_phase,
)
from generativeqc._f12_ucc_solver import UCCSDLambdaOptions
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.ecp import fitted_hf_ecp_domain
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalLowOrderEnergy,
    FocalLowOrderForces,
    FocalRequest,
    FocalSeries,
    plan_singlepoint,
)
from generativeqc.profiles import canonical_hash
from generativeqc.resources_hf import hf_resource_request


def _shell(ell: int, exps: tuple[float, ...], coeffs: tuple[float, ...]) -> BasisShell:
    return BasisShell(ell, tuple(str(x) for x in exps), (tuple(str(c) for c in coeffs),))


def _make_basis(name: str, elems: tuple[ElementBasis, ...]) -> BasisSet:
    prov = BasisProvenance(
        source="audit-fix4-test",
        version="1",
        license="MIT",
        checksum=canonical_hash((name, tuple(e.atomic_number for e in elems))),
    )
    return BasisSet(name, elems, prov, representation="spherical")


def _compact_h2o_split_basis(*, split_mp2: bool = True) -> FocalBasis:
    h_orb = ElementBasis(1, (_shell(0, (1.5,), (1.0,)), _shell(0, (0.32,), (1.0,))))
    o_orb = ElementBasis(
        8,
        (
            _shell(0, (18.0,), (1.0,)),
            _shell(0, (2.8,), (1.0,)),
            _shell(0, (0.55,), (1.0,)),
            _shell(1, (1.1,), (1.0,)),
        ),
    )
    h_cabs = ElementBasis(1, (_shell(0, (0.11,), (1.0,)),))
    o_cabs = ElementBasis(8, (_shell(0, (0.15,), (1.0,)),))
    h_jk = ElementBasis(
        1,
        (_shell(0, (4.0,), (1.0,)), _shell(0, (0.9,), (1.0,)), _shell(0, (0.2,), (1.0,))),
    )
    o_jk = ElementBasis(
        8,
        (
            _shell(0, (24.0,), (1.0,)),
            _shell(0, (5.0,), (1.0,)),
            _shell(0, (1.2,), (1.0,)),
            _shell(0, (0.3,), (1.0,)),
            _shell(1, (1.8,), (1.0,)),
        ),
    )
    h_f12 = ElementBasis(
        1,
        (_shell(0, (5.0,), (1.0,)), _shell(0, (1.1,), (1.0,)), _shell(0, (0.25,), (1.0,))),
    )
    o_f12 = ElementBasis(
        8,
        (
            _shell(0, (30.0,), (1.0,)),
            _shell(0, (6.0,), (1.0,)),
            _shell(0, (1.5,), (1.0,)),
            _shell(0, (0.35,), (1.0,)),
            _shell(1, (2.1,), (1.0,)),
        ),
    )
    h_mp2 = ElementBasis(
        1,
        (_shell(0, (4.5,), (1.0,)), _shell(0, (1.0,), (1.0,)), _shell(0, (0.22,), (1.0,))),
    )
    o_mp2 = ElementBasis(
        8,
        (
            _shell(0, (27.0,), (1.0,)),
            _shell(0, (5.5,), (1.0,)),
            _shell(0, (1.3,), (1.0,)),
            _shell(0, (0.32,), (1.0,)),
            _shell(1, (1.95,), (1.0,)),
        ),
    )
    orb = _make_basis("fix4-orb", (h_orb, o_orb))
    cabs = _make_basis("fix4-cabs", (h_cabs, o_cabs))
    jk = _make_basis("fix4-jk", (h_jk, o_jk))
    f12 = _make_basis("fix4-f12", (h_f12, o_f12))
    mp2 = _make_basis("fix4-mp2", (h_mp2, o_mp2)) if split_mp2 else f12
    return FocalBasis(2, "fix4-split", orb, jk_fit=jk, mp2_fit=mp2, f12_fit=f12, cabs=cabs, beta=0.9)


def _dummy_reference(
    request: FocalRequest,
    *,
    n: int = 7,
    occupied: tuple[int, ...] = (5,),
    native_owner: object | None = None,
) -> FocalReference:
    spin = len(occupied)
    eye = np.broadcast_to(np.eye(n, dtype=np.float64), (spin, n, n)).copy()
    eps = np.broadcast_to(np.arange(1, n + 1, dtype=np.float64), (spin, n)).copy()
    return FocalReference(
        -75.0,
        None,
        np.eye(n, dtype=np.float64),
        np.eye(n, dtype=np.float64),
        eye,
        eps,
        eye,
        eye,
        occupied,
        3,
        request.backend,
        request.surface_id,
        request.basis.basis_id,
        request.basis.orbital.identity,
        request.basis.jk_fit.identity,
        _native_owner=native_owner,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# M5 + M16 + M20: Exact budget enforcement, None telemetry, lambda_options, H2 frozen=1
# ---------------------------------------------------------------------------


def test_m5_m16_m20_spatial_rhf_ccsdf12b_contracts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify M5 exact budget (no 1.4x or +14 GiB inflation), M16 None telemetry & lambda_options, M20 o_act==0."""
    from generativeqc import _f12_focal_rcc_f12b as rcc_mod
    from generativeqc import _native
    from generativeqc import calculator as calc_mod

    basis = _compact_h2o_split_basis(split_mp2=False)
    h2_atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))]
    plan_frozen = plan_singlepoint(
        h2_atoms,
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        root_id="rhf",
        correlation_space_id="frozen:1",
        backend="cuda",
    )
    req_frozen = next(r for r, _ in plan_frozen.steps if r.role == "mp2_f12_correlation")

    fake_lib = SimpleNamespace(
        _handle=999,
        generativeqc_focal_rcc_f12b_v1=lambda *a: 0,
        generativeqc_context_create=lambda *a: 0,
        generativeqc_context_destroy=lambda *a: None,
        generativeqc_system_destroy=lambda *a: None,
        generativeqc_calculation_prepare=lambda *a: 0,
        generativeqc_calculation_destroy=lambda *a: None,
    )
    fake_owner = SimpleNamespace(library=fake_lib, handle=ct.c_void_p(1))
    ref_h2 = _dummy_reference(req_frozen, n=4, occupied=(1,), native_owner=fake_owner)
    prov = MolecularFocalProvider(
        shell_library=tmp_path / "unused.so",
        budget_mib=1024,
        run=lambda *a: None,
    )

    # M20: o_act == 0 (H2 with frozen:1) MUST short-circuit BEFORE _build_spatial_rhf_fixed_sources!
    def boom_fixed(*_a, **_kw):
        raise AssertionError("_build_spatial_rhf_fixed_sources must not be called when o_act == 0")

    monkeypatch.setattr(rcc_mod, "_build_spatial_rhf_fixed_sources", boom_fixed)
    from generativeqc import _f12_focal_one_electron as one_mod
    from generativeqc import _f12_focal_projector as projector_mod

    joint = SimpleNamespace(
        surface_id=req_frozen.surface_id, basis_id=req_frozen.basis.basis_id,
        executed_backend="cuda", overlap=np.eye(6), hcore=np.eye(6), obs_ao_count=4,
    )
    monkeypatch.setattr(one_mod, "one_electron_sources", lambda *args, **kwargs: joint)
    monkeypatch.setattr(
        projector_mod, "cabs_projector",
        lambda s, obs, **kw: projector_mod._reference(
            s, obs, kw["relative_cutoff"], kw["branch_guard"], kw["expected_rank"], None
        ),
    )
    with pytest.raises(ValueError, match="rank"):
        spatial_rhf_ccsdf12b_v1(prov, req_frozen, ref_h2, expected_cabs_rank=1)
    with pytest.raises(ValueError, match="spin channels"):
        spatial_rhf_ccsdf12b_v1(prov, req_frozen, replace(ref_h2, occupied=(1, 1)))
    res_zero = spatial_rhf_ccsdf12b_v1(
        prov, req_frozen, ref_h2, forces=True, expected_cabs_rank=2
    )
    assert res_zero.correlation_energy == 0.0
    assert res_zero.cabs_rank == 2
    assert res_zero.ordinary_forces is not None
    assert res_zero.fixed_forces is not None
    np.testing.assert_array_equal(res_zero.ordinary_forces, np.zeros((2, 3)))
    np.testing.assert_array_equal(res_zero.fixed_forces, np.zeros((2, 3)))
    assert res_zero.telemetry["final_residual"] is None

    # M16(b): non-default provider.lambda_options MUST raise NotImplementedError on native RHF E+F!
    prov_custom_lam = replace(prov, lambda_options=UCCSDLambdaOptions(max_operator_actions=32))
    with pytest.raises(NotImplementedError, match="provider.lambda_options"):
        spatial_rhf_ccsdf12b_v1(
            prov_custom_lam, req_frozen, ref_h2, forces=True, expected_cabs_rank=2
        )
    # Also verify MolecularFocalProvider.f12_doubles_energy_and_forces rejects it before calling native ABI!
    low_ef = FocalLowOrderForces(
        FocalLowOrderEnergy(ref_h2, 2, (0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
        np.zeros((2, 3)),
        np.zeros((2, 3)),
    )
    with pytest.raises(NotImplementedError, match="provider.lambda_options"):
        prov_custom_lam.f12_doubles_energy_and_forces(req_frozen, low_order=low_ef)

    # M5 & M16(a): verify exact budget_mib << 20 (zero 7/5 or +14 GiB inflation), MemoryError on tiny budget,
    # and None (NOT 0.0!) when native lambda telemetry is unavailable!
    plan_active = plan_singlepoint(
        h2_atoms,
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        root_id="rhf",
        correlation_space_id="all-active",
        backend="cuda",
    )
    req_active = next(r for r, _ in plan_active.steps if r.role == "mp2_f12_correlation")
    minimal = FocalSeries(replace(basis, orbital="sto-3g"))
    filled_plan = plan_singlepoint(
        h2_atoms, minimal, minimal, None, charge=-2,
        root_id="rhf", correlation_space_id="all-active", backend="cuda",
    )
    filled_req = next(r for r, _ in filled_plan.steps if r.role == "mp2_f12_correlation")
    no_virtual = _dummy_reference(filled_req, n=2, occupied=(2,), native_owner=fake_owner)
    for with_forces in (False, True):
        with pytest.raises(NotImplementedError, match="no OBS virtuals"):
            spatial_rhf_ccsdf12b_v1(prov, filled_req, no_virtual, forces=with_forces)

    ref_active = _dummy_reference(req_active, n=4, occupied=(1,), native_owner=fake_owner)
    o_act, n_vir = 1, 3

    monkeypatch.setattr(
        rcc_mod,
        "_build_spatial_rhf_fixed_sources",
        lambda *_a, **_kw: (
            np.zeros((o_act, n_vir)),
            np.zeros((n_vir, n_vir)),
            np.zeros((o_act, o_act, n_vir, n_vir)),
            -0.01,
            2,
            0,
        ),
    )

    seen_budgets: list[int] = []

    class DummyCalc:
        def __init__(self, **kwargs):
            seen_budgets.append(kwargs["correlation_memory_budget_bytes"])
            self._library = fake_lib

        def clear_cache(self):
            pass

        def _context_descriptor(self):
            return ct.c_int(0)

        def _create_native_system(self, *args):
            return ct.c_void_p(1)

        def _method_descriptor(self, *args, **kwargs):
            return ct.c_int(0)

    def fake_rcc_fn(
        _plan, _owner, _forces, _frozen, _wvo, _nwvo, _wvv, _nwvv, _cf, _ncf,
        _fe, _pre, e_out_ptr, *_rest
    ):
        e_out_ptr._obj.value = -0.025
        return 0

    fake_lib.generativeqc_focal_rcc_f12b_v1 = fake_rcc_fn
    monkeypatch.setattr(calc_mod, "Calculator", DummyCalc)
    monkeypatch.setattr(_native, "check", lambda *_a, **_kw: None)
    # An older library can return a valid diagnostic for a different calculation.
    def thread_diagnostic(scalars, *_):
        scalars[5] = 0.125
        return 1

    fake_lib.generativeqc_cc_last_lambda_telemetry_v1 = thread_diagnostic
    fake_lib.generativeqc_cc_reset_last_lambda_telemetry_v1 = lambda: pytest.fail(
        "a context-bound force must not reset the calling thread's diagnostic"
    )
    assert rcc_mod._read_last_lambda_telemetry(fake_lib)["final_residual"] == 0.125
    monkeypatch.setattr(
        rcc_mod, "relax_spatial_rhf_force_sources", lambda *_a, **_kw: np.zeros((2, 3))
    )
    monkeypatch.setattr(
        rcc_mod, "spatial_rhf_fixed_cc_forces", lambda *_a, **_kw: np.zeros((2, 3))
    )

    with capture_hybrid_phases() as historical:
        res_e = spatial_rhf_ccsdf12b_v1(prov, req_active, ref_active, forces=False, expected_cabs_rank=2)
        res_ef = spatial_rhf_ccsdf12b_v1(prov, req_active, ref_active, forces=True, expected_cabs_rank=2)
    assert [event["stage"] for event in historical] == [
        "fixed_cusp_constant_preparation", "fixed_cusp_constant_preparation", "fixed_cusp_lambda_rhs"
    ]
    assert res_e.telemetry["f12_cuda_preparations"] is None
    assert res_e.telemetry["fixed_cusp_lambda_rhs_backend"] == "not_requested"
    assert res_ef.telemetry["fixed_cusp_lambda_rhs_backend"] == "legacy_host_assisted"
    # Exact 1024 MiB passed for both energy-only AND forces (NEVER 1433 MiB or +14 GiB)!
    assert seen_budgets == [1024 << 20, 1024 << 20]
    assert res_e.telemetry["final_residual"] is None
    assert res_ef.telemetry["lambda_telemetry_available"] is False
    assert res_ef.telemetry["final_residual"] is None
    assert res_ef.telemetry["independent_residual_norm"] is None
    assert res_ef.telemetry["independent_residual_max"] is None

    with monkeypatch.context() as strict:
        strict.setenv(STRICT_GPU_ONLY_ENV, "1")
        with pytest.raises(NotImplementedError, match="before execution"):
            spatial_rhf_ccsdf12b_v1(prov, req_active, ref_active, forces=True)
        assert len(seen_budgets) == 2

        strict.setattr(fake_lib, "generativeqc_focal_rcc_f12b_cuda_stage_counters_v1",
                       lambda: 1, raising=False)

        def current_rcc_stub(*args):
            slots, length = args[-2:]
            assert length == 10 and np.isnan(slots[8]) and np.isnan(slots[9])
            slots[8] = 2.0
            slots[9] = 3.0 if args[2] else 0.0
            return fake_rcc_fn(*args)

        strict.setattr(fake_lib, "generativeqc_focal_rcc_f12b_v1", current_rcc_stub)
        for forces in (False, True):
            result = spatial_rhf_ccsdf12b_v1(prov, req_active, ref_active, forces=forces)
            assert result.telemetry["f12_cuda_preparations"] == 2
            assert result.telemetry["f12_cuda_rhs_evaluations"] == (3 if forces else 0)
            assert result.telemetry["fixed_cusp_constant_preparation_backend"] == "cuda"
            assert result.telemetry["fixed_cusp_lambda_rhs_backend"] == ("cuda" if forces else "not_requested")
        strict.setattr(fake_lib, "generativeqc_focal_rcc_f12b_v1", fake_rcc_fn)
        with pytest.raises(RuntimeError, match="did not write"):
            spatial_rhf_ccsdf12b_v1(prov, req_active, ref_active, forces=False)
        with pytest.raises(RuntimeError, match="requires positive"):
            rcc_mod._cuda_fixed_cusp_stage_counts(np.zeros(10), forces=True, advertised=True)
        with pytest.raises(RuntimeError, match="Energy-only"):
            rcc_mod._cuda_fixed_cusp_stage_counts(np.ones(10), forces=False, advertised=True)

    # Oversized problem vs tight budget_mib=1 MUST raise MemoryError!
    prov_tiny = replace(prov, budget_mib=1)
    big_o, big_v = 8, 32  # n2 = 8*8*32*32 = 65536 -> 3 * 512 KiB = 1.5 MiB > 1 MiB exact budget
    ref_big = _dummy_reference(req_active, n=big_o + big_v, occupied=(big_o,), native_owner=fake_owner)
    monkeypatch.setattr(
        rcc_mod,
        "_build_spatial_rhf_fixed_sources",
        lambda *_a, **_kw: (
            np.zeros((big_o, big_v)),
            np.zeros((big_v, big_v)),
            np.zeros((big_o, big_o, big_v, big_v)),
            -0.01,
            2,
            0,
        ),
    )
    with pytest.raises(MemoryError, match="exceeding exact budget_mib=1"):
        spatial_rhf_ccsdf12b_v1(prov_tiny, req_active, ref_big, forces=True, expected_cabs_rank=2)


def test_m10_tplus_from_occupied_zero_cuda_runner_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify tplus_from_occupied runs directly in exact FP64 NumPy (0 calls to provider.run on CUDA)."""
    from generativeqc_compiler.tensor import execute

    card = _compact_h2o_split_basis(split_mp2=False)
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8)), ("H", (1.75, 0.0, -0.4))]
    req = next(
        r
        for r, _ in plan_singlepoint(
            atoms,
            FocalSeries(card),
            FocalSeries(card),
            None,
            root_id="rhf",
            correlation_space_id="all-active",
            backend="cuda",
        ).steps
        if r.role == "mp2_f12_correlation"
    )
    ref = _dummy_reference(req, n=7, occupied=(5,))
    mp2_occ = np.array([-0.02, -0.03, -0.025, -0.015, -0.01], dtype=np.float64)
    f12_occ = np.array([-0.005, -0.006, -0.004, -0.003, -0.002], dtype=np.float64)
    triples_occ = np.array([-0.001, -0.002, -0.0015, -0.0008, -0.0005], dtype=np.float64)
    low = FocalLowOrderEnergy(
        ref,
        2,
        (0.0, 0.0),
        (0.0, float(2.0 * mp2_occ.sum()), 0.0),
        float(2.0 * f12_occ.sum()),
        OccupiedMP2F12((mp2_occ, mp2_occ), (f12_occ, f12_occ)),
    )
    assert card.mp2_fit is not None
    triples = FocalOccupiedTriples(ref, "all-active", card.mp2_fit.identity, triples_occ)

    cuda_run_calls = 0

    def counting_cuda_runner(*_a, **_kw):
        nonlocal cuda_run_calls
        cuda_run_calls += 1
        raise AssertionError("tplus_from_occupied must not invoke CUDA TensorIR runner")

    prov = MolecularFocalProvider(
        shell_library=Path("unused.so"),
        metric_library=Path("unused.so"),
        response_library=object(),  # type: ignore[arg-type]
        run=counting_cuda_runner,
    )
    monkeypatch.delenv("GENERATIVEQC_F12_STRICT_GPU_ONLY", raising=False)
    with capture_hybrid_phases() as phases:
        out = prov.tplus_from_occupied(req, low, triples)
    assert [(p["stage"], p["execution_mode"]) for p in phases] == [
        ("tplus_occupied_scaling", "hybrid_host_assisted"),
    ]
    assert cuda_run_calls == 0
    expected_q = (2.0 * mp2_occ + 2.0 * f12_occ) / (2.0 * mp2_occ)
    assert out.energy == pytest.approx(float(np.dot(expected_q, triples_occ)), abs=1e-14)
    np.testing.assert_allclose(out.triples, expected_q, atol=1e-14, rtol=0)

    # CPU tensor execute masquerading as CUDA runner is still rejected!
    prov_bad = replace(prov, run=execute)
    with pytest.raises(RuntimeError, match="FP64 CUDA executor"):
        prov_bad.tplus_from_occupied(req, low, triples)
    monkeypatch.setenv("GENERATIVEQC_F12_STRICT_GPU_ONLY", "1")
    with pytest.raises(NotImplementedError, match="tplus_occupied_scaling"):
        prov.tplus_from_occupied(req, low, triples)


# ---------------------------------------------------------------------------
# M11: Structured Hybrid-Phase Telemetry Disclosure & STRICT_GPU_ONLY Guard
# ---------------------------------------------------------------------------


def test_m11_hybrid_phase_disclosure_and_strict_gpu_only_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify capture_hybrid_phases records host-assisted stages and STRICT_GPU_ONLY=1 rejects them on CUDA."""
    monkeypatch.delenv(STRICT_GPU_ONLY_ENV, raising=False)
    with capture_hybrid_phases() as events:
        disclose_host_assisted_stage(
            "full_rank_cabs_overlap_bar",
            requested_backend="cuda",
            detail="Ozaki-4 host full-rank CABS overlap bar",
        )
        disclose_host_assisted_stage(
            "streamed_f12_pair_residuals",
            requested_backend="cpu_reference",
            detail="host residual math",
        )
    assert len(events) == 2
    assert events[0]["stage"] == "full_rank_cabs_overlap_bar"
    assert events[0]["execution_mode"] == "hybrid_host_assisted"
    assert events[1]["execution_mode"] == "host"

    monkeypatch.setenv(STRICT_GPU_ONLY_ENV, "1")
    # Host backend stays permitted under STRICT_GPU_ONLY=1
    disclose_host_assisted_stage(
        "full_rank_cabs_overlap_bar",
        requested_backend="cpu_reference",
        detail="host mode allowed",
    )
    # CUDA backend with host-contracted scientific stage MUST raise NotImplementedError!
    with pytest.raises(NotImplementedError, match=STRICT_GPU_ONLY_ENV):
        disclose_host_assisted_stage(
            "full_rank_cabs_overlap_bar",
            requested_backend="cuda",
            detail="Ozaki-4 host full-rank CABS overlap bar",
        )


# ---------------------------------------------------------------------------
# M13 + M14 + M21: Delta-CV Br/I FC9 Default, Narrow ECP Fallback, H2 None Abs
# ---------------------------------------------------------------------------


def test_m13_delta_cv_default_valence_space_bri_fc9_vs_light() -> None:
    """Verify omitted request_fc resolves frozen-core-spd (FC9) for Br/I ECP10/28 and frozen-core otherwise."""
    dummy_ecp = json.dumps(
        [
            {
                "ecp_type": "scalar_ecp",
                "angular_momentum": [0],
                "r_exponents": [0],
                "gaussian_exponents": ["1.0"],
                "coefficients": [["1.0"]],
            }
        ]
    )
    h_eb = ElementBasis(1, (_shell(0, (1.0,), (1.0,)),))
    o_eb = ElementBasis(8, (_shell(0, (1.0,), (1.0,)),))
    br_eb = ElementBasis(35, (_shell(0, (1.0,), (1.0,)),), ecp_core_electrons=10, ecp_data=dummy_ecp)
    i_eb = ElementBasis(53, (_shell(0, (1.0,), (1.0,)),), ecp_core_electrons=28, ecp_data=dummy_ecp)
    orb = _make_basis("m13-orb", (h_eb, o_eb, br_eb, i_eb))
    fb = FocalBasis(2, "m13", orb, jk_fit=orb, mp2_fit=orb, f12_fit=orb, cabs=orb, beta=1.0)

    for sym in ("Br", "I"):
        atoms = [(sym, (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.7))]
        r_cv = next(
            r
            for r, _ in plan_singlepoint(
                atoms,
                FocalSeries(fb),
                FocalSeries(fb),
                None,
                root_id="rhf",
                correlation_space_id="cv-active",
                backend="cpu_reference",
            ).steps
            if r.role == "mp2_f12_correlation"
        )
        assert default_valence_correlation_space_id(r_cv) == "frozen-core-spd"
        _, r_fc_f12 = _paired_cv_fc_requests(r_cv, None)
        assert r_fc_f12.correlation_space_id == "frozen-core-spd"

        r_cv_mp2 = conventional_mp2_request(
            atoms, fb, root_id="rhf", correlation_space_id="cv-active", backend="cpu_reference"
        )
        _, r_fc_mp2 = _paired_cv_fc_mp2_requests(r_cv_mp2, None)
        assert r_fc_mp2.correlation_space_id == "frozen-core-spd"

    r_h2o = conventional_mp2_request(
        [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 1.4, 1.1)), ("H", (0.0, -1.4, 1.1))],
        fb,
        root_id="rhf",
        correlation_space_id="cv-active",
        backend="cpu_reference",
    )
    assert default_valence_correlation_space_id(r_h2o) == "frozen-core"
    _, r_fc_h2o = _paired_cv_fc_mp2_requests(r_h2o, None)
    assert r_fc_h2o.correlation_space_id == "frozen-core"


def test_m14_narrow_ecp_fallback_and_strict_seed_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify unsupported ECP/spin states NEVER fall back and fallback SCF enforces exact seed checks."""
    from generativeqc import _f12_focal_cv as cv_mod

    dummy_ecp = json.dumps(
        [
            {
                "ecp_type": "scalar_ecp",
                "angular_momentum": [0],
                "r_exponents": [0],
                "gaussian_exponents": ["1.0"],
                "coefficients": [["1.0"]],
            }
        ]
    )
    h_eb = ElementBasis(1, (_shell(0, (1.0,), (1.0,)),))
    br_sf_eb = ElementBasis(
        35,
        (_shell(0, (1.0,), (1.0,)), _shell(3, (0.5,), (1.0,))),
        ecp_core_electrons=10,
        ecp_data=dummy_ecp,
    )
    orb_sf = _make_basis("m14-hbr-sf", (h_eb, br_sf_eb))
    br_fit_eb = ElementBasis(35, (_shell(0, (1.0,), (1.0,)), _shell(3, (0.5,), (1.0,))))
    jk_sf = _make_basis("m14-hbr-jk", (h_eb, br_fit_eb))
    fb_sf = FocalBasis(2, "m14-sf", orb_sf, jk_fit=jk_sf, mp2_fit=jk_sf)
    # Audit counterexample: neutral HBr, ECP10MDF, UHF triplet (mult=3), s-f orbitals!
    req_triplet = conventional_mp2_request(
        [("Br", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 2.7))],
        fb_sf,
        root_id="uhf",
        correlation_space_id="cv-active",
        charge=0,
        multiplicity=3,
        backend="cuda",
    )

    def boom_fallback(*_a, **_kw):
        raise AssertionError("Unsupported ECP triplet must NEVER reach focal-source DF-HF fallback")

    from generativeqc import _f12_cv_dfmp2_native as cv_nat_mod

    monkeypatch.setattr(cv_nat_mod, "bind_delta_cv_native", lambda *_a, **_kw: None)
    monkeypatch.setattr(cv_mod, "_solve_df_reference_from_focal_sources", boom_fallback)
    prov_cuda = MolecularFocalProvider(
        shell_library=Path("unused.so"),
        metric_library=Path("unused.so"),
        response_library=object(),  # type: ignore[arg-type]
        run=lambda *a: None,
    )
    with pytest.raises(NotImplementedError, match="fitted HF ECP domain"):
        prov_cuda.delta_cv_mp2_energy(req_triplet)

    with pytest.raises(NotImplementedError, match="fitted HF ECP domain"):
        prov_cuda.delta_cv_mp2_energy_and_forces(req_triplet)

    # An in-domain high-angular basis does not authorize a different algorithm
    # when the unavailable operation is, for example, spin-root transport.
    g_orb = _make_basis("m14-high-l", (ElementBasis(1, (_shell(4, (1.0,), (1.0,)),)),))
    high_l_req = conventional_mp2_request(
        [("H", (0., 0., 0.)), ("H", (0., 0., 1.4))],
        FocalBasis(2, "m14-high-l", g_orb, jk_fit=g_orb, mp2_fit=g_orb),
        root_id="rhf", correlation_space_id="cv-active", backend="cuda",
    )
    orbital_reason = (
        "orbital atom 0 Z=1 shell 0 contraction 0 l=4: "
        "cuda/df_three_center derivative 0 supports only l<=3"
    )
    assert cv_mod._is_eligible_for_focal_source_df_hf_fallback(
        high_l_req, NotImplementedError(orbital_reason)
    )
    for reason in ("spin-root transport unavailable", orbital_reason + "; seed unavailable"):
        assert not cv_mod._is_eligible_for_focal_source_df_hf_fallback(
            high_l_req, NotImplementedError(reason)
        )

    # Verify _solve_df_reference_from_focal_sources rejects wrong-basis/wrong-state seed of same shape!
    orb_h2 = _make_basis("m14-h2", (ElementBasis(1, (_shell(0, (1.5,), (1.0,)), _shell(0, (0.3,), (1.0,)))),))
    fb_a = FocalBasis(2, "m14-a", orb_h2, jk_fit=orb_h2, mp2_fit=orb_h2)
    orb_h2_b = _make_basis("m14-h2-b", (ElementBasis(1, (_shell(0, (1.6,), (1.0,)), _shell(0, (0.3,), (1.0,)))),))
    fb_b = FocalBasis(2, "m14-b", orb_h2_b, jk_fit=orb_h2_b, mp2_fit=orb_h2_b)
    atoms_h2 = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))]
    req_a = conventional_mp2_request(atoms_h2, fb_a, root_id="rhf", correlation_space_id="cv-active", backend="cpu_reference")
    req_b = conventional_mp2_request(atoms_h2, fb_b, root_id="rhf", correlation_space_id="cv-active", backend="cpu_reference")
    ref_b = _dummy_reference(req_b, n=4, occupied=(1,))
    seed_b = FocalReferenceSeed(req_b, ref_b)
    prov_seeded = MolecularFocalProvider(shell_library=Path("unused.so"), reference_seed=seed_b)
    one_dummy = SimpleNamespace(overlap=np.eye(4), hcore=-np.eye(4))
    jk_dummy = SimpleNamespace(data={"ao_coulomb": np.zeros((2, 4, 4))}, whitener=np.eye(2))
    with pytest.raises(ValueError, match="identical OBS and JKFit"):
        _solve_df_reference_from_focal_sources(prov_seeded, req_a, one_dummy, jk_dummy, run=lambda *a: None, seed=seed_b)


def test_m14_displaced_seed_uses_overlap_normalized_spin_density(monkeypatch) -> None:
    from generativeqc import _f12_cv_dfmp2_native as native
    from generativeqc import overlap

    basis = _compact_h2o_split_basis(split_mp2=False)

    def request(distance):
        return conventional_mp2_request(
            [("H", (0., 0., 0.)), ("H", (0., 0., distance))], basis,
            root_id="rhf", correlation_space_id="cv-active", backend="cpu_reference",
        )

    src, target = request(1.4), request(1.4002)
    metric = np.diag([4., 2., 1., 1.])
    coeff = np.diag([.5, 2.**-.5, 1., 1.])
    expected_density = 2. * np.outer(coeff[:, 0], coeff[:, 0])
    seed = FocalReferenceSeed(src, replace(
        _dummy_reference(src, n=4, occupied=(1,)),
        overlap=metric, coefficients=coeff[None, :, :],
    ))
    monkeypatch.setattr(overlap, "cross_overlap", lambda *a, **kw: metric)

    def checked_fock(_b, density, ca, _cb, **kwargs):
        np.testing.assert_allclose(density, expected_density, atol=1e-13)
        np.testing.assert_allclose(ca.T @ metric @ ca, [[1.]], atol=1e-13)
        return np.zeros_like(metric), np.zeros_like(metric), None

    monkeypatch.setattr(native, "bind_delta_cv_native", lambda *a, **kw: SimpleNamespace(
        run_df_jk_fock=checked_fock, host_reference=True
    ))
    ref = _solve_df_reference_from_focal_sources(
        MolecularFocalProvider(shell_library=Path("unused.so")), target,
        SimpleNamespace(overlap=metric, hcore=np.diag([-8., 0., 1., 2.])),
        SimpleNamespace(data={"ao_coulomb": np.zeros((1, 4, 4))}, whitener=np.eye(1)),
        run=None, seed=seed,
    )
    assert ref.occupied_overlaps == pytest.approx((1.,))
    assert ref.execution_mode == "cpu_reference"

    invalid_center = FocalReferenceSeed(target, replace(
        seed.reference, surface_id=target.surface_id, coefficients=np.eye(4)[None, :, :]
    ))
    with pytest.raises(ValueError, match="orthonormal"):
        _solve_df_reference_from_focal_sources(
            MolecularFocalProvider(shell_library=Path("unused.so")), target,
            SimpleNamespace(overlap=metric, hcore=np.diag([-8., 0., 1., 2.])),
            SimpleNamespace(data={"ao_coulomb": np.zeros((1, 4, 4))}, whitener=np.eye(1)),
            run=None, seed=invalid_center,
        )


# ---------------------------------------------------------------------------
# m3, m4, m8, D5, D1, D4
# ---------------------------------------------------------------------------


def test_m3_brh4_doublet_orbital_g_rejected_matching_cpp_halide_topology() -> None:
    """Verify BrH4 neutral doublet orbital-g is rejected in Python just like C++ small_core_halide.hpp."""
    dummy_ecp = json.dumps(
        [
            {
                "ecp_type": "scalar_ecp",
                "angular_momentum": [0],
                "r_exponents": [0],
                "gaussian_exponents": ["1.0"],
                "coefficients": [["1.0"]],
            }
        ]
    )
    h_eb = ElementBasis(1, (_shell(0, (1.0,), (1.0,)),))
    br_f = ElementBasis(
        35,
        (_shell(0, (1.0,), (1.0,)), _shell(3, (0.5,), (1.0,))),
        ecp_core_electrons=10,
        ecp_data=dummy_ecp,
    )
    br_g = ElementBasis(
        35,
        (_shell(0, (1.0,), (1.0,)), _shell(4, (0.5,), (1.0,))),
        ecp_core_electrons=10,
        ecp_data=dummy_ecp,
    )
    basis_f = _make_basis("br-f", (h_eb, br_f))
    basis_g = _make_basis("br-g", (h_eb, br_g))

    brh4 = [
        Atom("Br", (0.0, 0.0, 0.0)),
        Atom("H", (1.5, 0.0, 0.0)),
        Atom("H", (-1.5, 0.0, 0.0)),
        Atom("H", (0.0, 1.5, 0.0)),
        Atom("H", (0.0, -1.5, 0.0)),
    ]
    brh2 = brh4[:3]
    hbr = brh4[:2]
    br_atom = brh4[:1]

    # BrH4 neutral doublet: s..f admitted, orbital-g REJECTED (matches C++ limit == 3)!
    assert fitted_hf_ecp_domain("uhf", basis_f, brh4, 0, 2) is True
    assert fitted_hf_ecp_domain("uhf", basis_g, brh4, 0, 2) is False
    # BrH2 doublet, bare Br doublet, and HBr singlet with orbital-g remain admitted (limit == 4)!
    assert fitted_hf_ecp_domain("uhf", basis_g, brh2, 0, 2) is True
    assert fitted_hf_ecp_domain("uhf", basis_g, br_atom, 0, 2) is True
    assert fitted_hf_ecp_domain("rhf", basis_g, hbr, 0, 1) is True


def test_m4_hf_resource_request_does_not_exhaust_single_use_shell_iterator() -> None:
    """Verify hf_resource_request snapshots basis once so single-use iterators retain full shell counts."""
    atoms = (Atom("H", (0.0, 0.0, 0.0)), Atom("H", (0.0, 0.0, 1.4)))
    shells = (
        Shell(0, 0, (Primitive(3.42525091, 0.15432897), Primitive(0.62391373, 0.53532814))),
        Shell(1, 0, (Primitive(3.42525091, 0.15432897), Primitive(0.62391373, 0.53532814))),
    )
    req_tuple = hf_resource_request(
        method="rhf",
        backend="cpu",
        precision="fp64",
        basis=shells,
        auxiliary_basis=None,
        basis_representation="cartesian",
        systems=(atoms,),
        charges=(0,),
        multiplicities=(1,),
        density_fitting="none",
        diis_history=8,
        max_iterations=50,
        energy_tolerance=1e-9,
        density_tolerance=1e-8,
        screening_tolerance=1e-12,
        density_fitting_relative_threshold=1e-8,
        density_fitting_memory_budget_bytes=0,
        device_id=0,
    )
    req_iter = hf_resource_request(
        method="rhf",
        backend="cpu",
        precision="fp64",
        basis=(s for s in shells),
        auxiliary_basis=None,
        basis_representation="cartesian",
        systems=(atoms,),
        charges=(0,),
        multiplicities=(1,),
        density_fitting="none",
        diis_history=8,
        max_iterations=50,
        energy_tolerance=1e-9,
        density_tolerance=1e-8,
        screening_tolerance=1e-12,
        density_fitting_relative_threshold=1e-8,
        density_fitting_memory_budget_bytes=0,
        device_id=0,
    )
    assert req_iter.identity == req_tuple.identity
    payload = json.loads(req_iter.identity.topology)
    assert payload["items"][0]["orbital"]["nbf"] == 2

    shared_iterator = iter(shells)
    fitted_from_iter = hf_resource_request(
        (atoms,), basis=shared_iterator, auxiliary_basis=shared_iterator, density_fitting="cpu",
    )
    fitted_from_tuple = hf_resource_request(
        (atoms,), basis=shells, auxiliary_basis=shells, density_fitting="cpu",
    )
    assert fitted_from_iter.identity == fitted_from_tuple.identity


@pytest.mark.parametrize("capture", [capture_ram_trace, capture_hybrid_phases])
def test_overlapping_empty_traces_remove_only_the_closed_collector(capture) -> None:
    first, second, third = (capture() for _ in range(3))
    records_a = first.__enter__()
    records_b = second.__enter__()
    first.__exit__(None, None, None)
    with third as records_c:
        second.__exit__(None, None, None)
        record_ram_phase("active")
        disclose_host_assisted_stage("active", requested_backend="cpu_reference", detail="test")
    assert records_a == records_b == []
    assert len(records_c) == 1


def test_m8_capture_ram_trace_contextvar_interleaved_and_multithreaded() -> None:
    """Verify capture_ram_trace uses ContextVar and survives interleaved exits and concurrent threads."""
    # 1. Interleaved out-of-order manual __enter__ / __exit__ in the same thread
    cm_a = capture_ram_trace()
    cm_b = capture_ram_trace()
    rec_a = cm_a.__enter__()
    record_ram_phase("only_a")
    rec_b = cm_b.__enter__()
    record_ram_phase("both_ab")
    cm_a.__exit__(None, None, None)
    record_ram_phase("only_b_after_a_exit")
    cm_b.__exit__(None, None, None)
    record_ram_phase("none_active")

    assert [e["phase"] for e in rec_a] == ["only_a", "both_ab"]
    assert [e["phase"] for e in rec_b] == ["both_ab", "only_b_after_a_exit"]

    # 2. Concurrent threads never cross-contaminate each other's traces
    barrier = threading.Barrier(2)
    results: dict[str, list[str]] = {}

    def worker(tag: str) -> None:
        with capture_ram_trace() as trace:
            barrier.wait()
            record_ram_phase(f"phase_{tag}_1")
            barrier.wait()
            record_ram_phase(f"phase_{tag}_2")
            results[tag] = [str(e["phase"]) for e in trace]

    t1 = threading.Thread(target=worker, args=("A",))
    t2 = threading.Thread(target=worker, args=("B",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert results["A"] == ["phase_A_1", "phase_A_2"]
    assert results["B"] == ["phase_B_1", "phase_B_2"]


def test_d5_unified_uhf_cc_backend_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify GENERATIVEQC_F12_UHF_CC_BACKEND=native|legacy replaces the duplicate legacy env vars."""
    card = _compact_h2o_split_basis(split_mp2=False)
    req = next(
        r
        for r, _ in plan_singlepoint(
            [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8))],
            FocalSeries(card),
            FocalSeries(card),
            None,
            root_id="uhf",
            multiplicity=2,
            correlation_space_id="all-active",
            backend="cpu_reference",
        ).steps
        if r.role == "mp2_f12_correlation"
    )
    fake_owner = SimpleNamespace(
        library=SimpleNamespace(generativeqc_focal_ucc_f12b_v1=object())
    )
    ref = _dummy_reference(req, n=7, occupied=(5, 4), native_owner=fake_owner)
    prov = MolecularFocalProvider(shell_library=Path("unused.so"))

    monkeypatch.delenv("GENERATIVEQC_F12_UHF_CC_BACKEND", raising=False)
    assert prov._use_native_uhf_f12b(req, ref) is True

    # Old duplicate aliases MUST NOT override selection anymore
    monkeypatch.setenv("GENERATIVEQC_F12_LEGACY_UHF_CC", "1")
    monkeypatch.setenv("GENERATIVEQC_UHF_F12B_V1", "0")
    assert prov._use_native_uhf_f12b(req, ref) is True

    # Unified selector controls native vs legacy cleanly
    monkeypatch.setenv("GENERATIVEQC_F12_UHF_CC_BACKEND", "legacy")
    assert prov._use_native_uhf_f12b(req, ref) is False

    prov_legacy = replace(prov, use_native_uhf_f12b=False)
    monkeypatch.setenv("GENERATIVEQC_F12_UHF_CC_BACKEND", "native")
    assert prov_legacy._use_native_uhf_f12b(req, ref) is True

    monkeypatch.setenv("GENERATIVEQC_F12_UHF_CC_BACKEND", "invalid_mode")
    with pytest.raises(ValueError, match="GENERATIVEQC_F12_UHF_CC_BACKEND"):
        prov._use_native_uhf_f12b(req, ref)


def test_d4_host_libraries_reject_invalid_explicit_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An invalid explicit test library must fail before any alternative is selected."""
    from conftest import f12_host_libraries

    missing = tmp_path / "does_not_exist.so"
    monkeypatch.setenv("GENERATIVEQC_F12_HOST_SINK", str(missing))
    with pytest.raises(FileNotFoundError, match=missing.name):
        f12_host_libraries()

    shell = tmp_path / "shell.so"
    shell.touch()
    monkeypatch.setenv("GENERATIVEQC_F12_HOST_SINK", str(shell))
    monkeypatch.setenv("GENERATIVEQC_F12_ONE_ECP", str(missing))
    with pytest.raises(FileNotFoundError, match=missing.name):
        f12_host_libraries()
    monkeypatch.delenv("GENERATIVEQC_F12_ONE_ECP")
    assert f12_host_libraries() == (shell, shell)
