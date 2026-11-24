"""Qualify one complete candidate nuclear-gradient assembly against immutable displaced energies."""

import json

import numpy as np
from generativeqc_compiler.mp2.f12_frame_response import full_rank_cabs_cotangent

from benchmarks.f12_molecular_force_composition import (
    calculate_panel,
    common_checks,
    endpoint_arguments,
    write_measurement,
)


def verify(result: dict, target: dict, original: dict) -> dict:
    if original["geometry_sha256"] != target["geometry"]["geometry_sha256"]:
        raise ValueError("Original and candidate molecular coordinates differ")
    native_energies = original["native_displacement_energies_Eh"]
    independent = original["independent_full_RI_direction_Eh_per_bohr"]
    arm_steps = {str(arm["step"]) for arm in target["native_displacements"]}
    if (
        arm_steps != {"0.0004", "0.0002"}
        or set(native_energies) != arm_steps
        or set(independent) != arm_steps
    ):
        raise ValueError("Incomplete original two-step energy evidence")
    force = np.array(result["total_force_Eh_per_bohr"])
    direction = np.array(target["stock"]["direction"])
    projection = float(np.vdot(force, direction))
    observed = {
        step: -(native_energies[step][0] - native_energies[step][1]) / (2 * float(step))
        for step in arm_steps
    }
    native_errors = {step: abs(projection - value) for step, value in observed.items()}
    independent_errors = {
        step: abs(projection - value) for step, value in independent.items()
    }
    scalar_error = abs(result["total_energy_Eh"] - original["independent_total_Eh"])
    increment_error = abs(result["increment_Eh"] - original["independent_increment_Eh"])
    reconstructed = np.array(original["component_replacement_forces_Eh_per_bohr"])
    if (
        not np.isfinite(force).all()
        or force.shape != direction.shape
        or force.shape != reconstructed.shape
    ):
        raise ValueError("Invalid native or reconstructed molecular gradient")
    recovery_error = float(np.max(abs(force - reconstructed)))
    _, obs, n = target["layout"]["space"]
    graph_id = full_rank_cabs_cotangent(n, obs).logical_hash
    checks = {
        **common_checks(result, target),
        "ten_physical_integral_channels": len(result["integral_gradients"]["rows"])
        == 10
        and result["integral_gradients"]["passes_contraction_checks"],
        "generated_constraint_graph": any(
            p["program"] == graph_id for p in result["graph_gpu_programs"]
        ),
        "no_displaced_replays": not result["native_directional_checks"],
        "center_total_Eh": scalar_error <= 5e-11,
        "center_increment_Eh": increment_error <= 5e-11,
        "two_native_steps": max(native_errors.values()) <= 2e-6,
        "two_independent_steps": max(independent_errors.values()) <= 2e-6,
        "all_cartesian_native_assembly": recovery_error <= 3e-8,
    }
    return {
        **result,
        "passed": all(checks.values()),
        "checks": checks,
        "geometry_sha256": original["geometry_sha256"],
        "total_energy_error_Eh": scalar_error,
        "increment_error_Eh": increment_error,
        "max_cartesian_reconstruction_error_Eh_per_bohr": recovery_error,
        "native_directional_errors_Eh_per_bohr": native_errors,
        "independent_directional_errors_Eh_per_bohr": independent_errors,
        "original_endpoint_job_id": original["original_job_id"],
        "frame_component_job_id": original["frame_component_job_id"],
        "original_and_stable_references_sha256": original["hashes"],
        "scope": "Complete native central molecular F12 energy, analytic nuclear force and orbital/integral responses, with the original CUDA ordinary MP2 energy/gradient. Previously calculated displaced energies are reused.",
    }


if __name__ == "__main__":
    args = endpoint_arguments()
    target = json.loads(args.panel.read_bytes())
    evidence = json.loads((args.panel.parent / "original_comparison.json").read_bytes())
    # The original geometry, orbital basis, auxiliary bases, and reference definitions are retained.
    result = calculate_panel(
        args, {**target, "native_displacements": []}, full_rank_constraint=True
    )
    finished = verify(result, target, evidence)
    write_measurement(args.output, finished)
    if not finished["passed"]:
        raise SystemExit(1)
