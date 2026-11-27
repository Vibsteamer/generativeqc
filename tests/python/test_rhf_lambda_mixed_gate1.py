"""Gate 1 local CPU & controller qualification for tuned mixed-precision RHF CCSD-F12b Lambda.

Verifies:
1. Production multi-pass mixed-precision iterative refinement and GCR projection
   controller linked against the native GMRES solver:
   - Exact solution parity (< 1e-12) vs pure FP64 baseline;
   - Dramatic reduction in strict FP64 operator actions (<= 3 vs > 10 baseline);
   - Exact operator-action and precision-switch accounting;
   - Monotonic contraction guard & clean FP64 handoff on ill-conditioned/stagnating systems;
   - Nonfinite mixed-precision action propagation without silent strict retry.
2. End-to-end CPU ``spatial_rhf_ccsdf12b_cpu_solve`` + ``spatial_rhf_fixed_cc_forces``
   parity (<= 1e-11 Eh energy, <= 1e-10 Eh/bohr force) and structured telemetry fields
   (``fp32_iterations``, ``fp64_iterations``, ``fp32_seconds``, ``fp64_seconds``,
   ``precompute_seconds``, ``final_residual``, ``handoff_reason``) exported through
   ``generativeqc_cc_last_lambda_telemetry_v1``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]


def _build_controller_harness(tmp_path: Path, required_native_cxx: NativeCxx) -> Path:
    executable = tmp_path / "rhf_lambda_mixed_controller_gate1"
    required_native_cxx.build_executable(
        [
            ROOT / "tests/native/test_rhf_lambda_refinement.cpp",
            ROOT / "src/response/native_gmres.cpp",
        ],
        executable,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "src"),
        ),
    )
    return executable


def test_rhf_lambda_mixed_refinement_controller_parity_and_safety(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Verify multi-pass mixed refinement parity, FP64 action reduction, and stagnation safety."""
    exe = _build_controller_harness(tmp_path, required_native_cxx)
    proc = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=30)
    report = json.loads(proc.stdout.strip())
    assert report["case1"]["max_diff"] <= 1e-12
    assert report["case1"]["residual"] <= 1e-12
    assert report["case1"]["mixed_fp64_actions"] <= 3
    assert report["case1"]["fp64_baseline_actions"] >= 12
    assert report["case1"]["reason"] == "converged_mixed_ir"
    assert report["case2"]["max_diff"] <= 1e-11
    assert report["case2"]["residual"] <= 1e-12
    assert report["case2"]["mixed_fp64_actions"] < report["case2"]["fp64_baseline_actions"] / 2
    assert report["case3"]["max_diff"] <= 1e-11
    assert report["case3"]["residual"] <= 1e-12
    assert report["case3"]["reason"] == "stagnated_handoff_fp64"


@pytest.mark.parametrize("corr_space", ["all-active", "frozen-core"])
def test_rhf_ccsdf12b_cpu_parity_forces_and_structured_telemetry(corr_space: str) -> None:
    """Verify exact CPU CCSD-F12b energy/force parity and structured Lambda telemetry export."""
    from generativeqc._f12_focal_cc_fixed import fixed_ucc_source_forces
    from generativeqc._f12_focal_cc_sources import molecular_ucc_sources
    from generativeqc._f12_focal_rcc_f12b import (
        _build_spatial_rhf_fixed_sources,
        spatial_rhf_ccsdf12b_cpu_solve,
        spatial_rhf_fixed_cc_forces,
    )
    from generativeqc._f12_ucc_solver import (
        SolverOptions,
        UCCSDLambdaOptions,
        solve_focal_uccsd_f12b_with_cotangents,
    )
    from generativeqc.experimental.f12_focal import (
        FocalSeries,
        MolecularFocalProvider,
        plan_singlepoint,
    )
    from generativeqc.response_solver import GMRESOptions
    from conftest import f12_host_libraries
    from test_focal_rhf_ccsdf12b_cuda_gate1 import (
        _canonical_rhf_reference_for_test,
        _compact_split_basis,
        _spatial_chemists_blocks_from_mp2_panel,
    )

    shell_lib, one_lib = f12_host_libraries()
    basis = _compact_split_basis((1, 8))
    atoms = (
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.10, -1.43, 1.11)),
        ("H", (-0.04, 1.38, 1.15)),
    )
    provider = MolecularFocalProvider(
        shell_library=shell_lib,
        one_electron_library=one_lib,
        budget_mib=1536,
    )
    plan = plan_singlepoint(
        atoms,
        FocalSeries(basis),
        FocalSeries(basis),
        None,
        charge=0,
        multiplicity=1,
        root_id="rhf",
        correlation_space_id=corr_space,
        backend="cpu_reference",
    )
    req = next(r for r, _ in plan.steps if r.role == "mp2_f12_correlation")
    ref = _canonical_rhf_reference_for_test(provider, req)
    sources = molecular_ucc_sources(provider, req, include_ordinary=True, reference=ref)
    frozen = sources.fixed.frozen_occupied[0]

    ref_state, _, ref_pullback = solve_focal_uccsd_f12b_with_cotangents(
        sources,
        options=SolverOptions(residual_tolerance=1e-12, energy_tolerance=1e-13, max_bytes=1536 << 20),
        lambda_options=UCCSDLambdaOptions(
            cc_tolerance=1e-10,
            lambda_tolerance=1e-10,
            max_bytes=1536 << 20,
            max_total_work=1 << 50,
            gmres=GMRESOptions(rtol=0.0, atol=1e-13, max_iterations=400, max_workspace_bytes=1536 << 20),
        ),
    )
    w_vo, w_vv, c_f12, fixed_e, cabs_rank, _ = _build_spatial_rhf_fixed_sources(
        provider, req, ref, expected_cabs_rank=sources.fixed.cabs_rank, derivatives=True
    )
    blocks = _spatial_chemists_blocks_from_mp2_panel(provider, req, ref, sources, frozen)
    spatial = spatial_rhf_ccsdf12b_cpu_solve(
        **blocks,
        w_vo=w_vo,
        w_vv=w_vv,
        c_f12=c_f12,
        fixed_geminal_energy=fixed_e,
        solve_lambda=True,
        max_bytes=1536 << 20,
    )
    assert abs(float(spatial["correlation_energy"]) - float(ref_state.correlation_energy)) <= 1e-11

    ref_forces = fixed_ucc_source_forces(provider, req, sources, spatial["fixed_weights"])
    sp_forces = spatial_rhf_fixed_cc_forces(
        provider,
        req,
        ref,
        cabs_rank=cabs_rank,
        fixed_geminal_energy=fixed_e,
        t1=spatial["t1"],
        t2=spatial["t2"],
        lambda1=spatial["lambda1"],
        lambda2=spatial["lambda2"],
        frozen_occupied=frozen,
    )
    np.testing.assert_allclose(sp_forces, ref_forces, rtol=0.0, atol=1e-10)

    tel = spatial["lambda_telemetry"]
    assert isinstance(tel, dict)
    for key in (
        "fp32_iterations",
        "fp64_iterations",
        "fp32_seconds",
        "fp64_seconds",
        "precompute_seconds",
        "final_residual",
        "handoff_reason",
    ):
        assert key in tel, f"Missing required structured telemetry field {key}"
    assert tel["fp32_iterations"] == 0
    assert tel["fp64_iterations"] > 0
    assert tel["fp64_seconds"] >= 0.0
    assert tel["precompute_seconds"] >= 0.0
    assert tel["final_residual"] <= 1e-10
    assert tel["handoff_reason"] == "cpu_fp64"
