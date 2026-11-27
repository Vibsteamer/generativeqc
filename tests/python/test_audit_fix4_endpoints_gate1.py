"""Gate 1 regression & counterexample tests for Audit Fix-4 (Python endpoints, budget, AOT, Delta-CV, hygiene)."""

from __future__ import annotations

import ctypes as ct
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc._f12_focal_molecular import (
    MolecularFocalProvider,
)
from generativeqc._f12_focal_rcc_f12b import spatial_rhf_ccsdf12b_v1
from generativeqc._f12_focal_reference import (
    FocalReference,
)
from generativeqc._f12_ram_hygiene import (
    STRICT_GPU_ONLY_ENV,
    capture_hybrid_phases,
    capture_ram_trace,
    disclose_host_assisted_stage,
    record_ram_phase,
)
from generativeqc._f12_ucc_solver import UCCSDLambdaOptions
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.experimental.f12_focal import (
    FocalBasis,
    FocalLowOrderEnergy,
    FocalLowOrderForces,
    FocalRequest,
    FocalSeries,
    plan_singlepoint,
)
from generativeqc.profiles import canonical_hash


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
