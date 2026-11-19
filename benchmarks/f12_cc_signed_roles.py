"""Join independently returned CC-F12b E/F and signed energy pairs under one numerical protocol."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from benchmarks.f12_cc_benchmark_center import check_center
from benchmarks.f12_molecular_force_composition import (
    project_signed_pair,
    write_measurement,
)
from benchmarks.f12_molecular_sink_test import sha256 as file_sha


def identity(args: argparse.Namespace, panel: dict) -> dict:
    """Bind both job roles to the geometry, shell layout, sources, runtime and solver options."""
    inputs = args.scf_inputs
    case = json.loads((inputs / "cases.json").read_bytes())
    files = case["files"]
    hashes = {}
    for name in ("source.tar.gz", "runtime_source.tar.gz", *panel["scf_input_sha256"]):
        hashes[name] = file_sha(inputs / name)
        if hashes[name] != files[name] or hashes[name] != panel["scf_input_sha256"].get(
            name, files[name]
        ):
            raise ValueError(f"mismatched CC input digest: {name}")
    for name, path in (("panel.json", args.panel), ("layout.npz", args.layout)):
        hashes[name] = file_sha(path)
        if hashes[name] != files[name]:
            raise ValueError(f"mismatched CC input digest: {name}")
    return {
        "geometry_sha256": panel["geometry"]["geometry_sha256"],
        "files": hashes,
        "source_commit": case["source_commit"],
        "runtime_sha256": case["runtime_sha256"],
        "arch": args.arch,
        "budget_mib": args.budget_mib,
        "full_rank": args.full_rank_cabs_response,
        "relative_cutoff": args.cabs_relative_cutoff,
    }


def ordered_pairs(
    panel: dict, pairs: list[dict], *, verify_device: bool, step: float | None = None
) -> list[dict]:
    """Require both ordered signs and verify the five terms, independent HF, root and geometry key."""
    expected = {
        (arm["name"], arm["step"]): arm
        for arm in panel["native_displacements"]
        if step is None or arm["step"] == step
    }
    keys = [(p["direction"], p["step_bohr"]) for p in pairs]
    if (
        len(keys) != (2 if step is None else 1)
        or set(keys) != set(expected)
        or {h for _, h in keys} != ({0.0004, 0.0002} if step is None else {step})
        or (step is not None and step not in (0.0002, 0.0004))
    ):
        raise ValueError("missing, duplicate or unexpected signed pair")
    by_key = dict(zip(keys, pairs, strict=True))
    for key, arm in expected.items():
        observation = by_key[key]
        for sign, reference, component, energy in zip(
            (1, -1),
            arm["hf_energies_Eh"],
            observation["energy_components"],
            observation["energies_Eh"],
            strict=True,
        ):
            total = math.fsum(
                component[k] for k in panel["cc_reference"]["components_Eh"]
            )
            if (
                component["sign"] != sign
                or not 0 <= component["cc_residual"] <= 1e-10
                or not math.isclose(total, energy, rel_tol=0, abs_tol=2e-12)
                or not math.isclose(
                    component["reference_energy_Eh"],
                    reference,
                    rel_tol=0,
                    abs_tol=5e-11,
                )
            ):
                raise ValueError("signed HF, CC root, sign or total disagrees")
            if verify_device:
                device = component.get("integral_device_bytes", {})
                if len(device) != 10 or not all(v > 0 for v in device.values()):
                    raise ValueError("missing signed GPU integral evidence")
    return [by_key[key] for key in expected]


def join_measurements(
    panel: dict, center: dict, pairs: list[dict], *, verify_device: bool
) -> dict:
    """Project the returned central force onto the original axis and run the unchanged full gates."""
    force = np.asarray(center["total_force_Eh_per_bohr"])
    report = dict(center)
    report.pop("displacements", None)
    report["native_directional_checks"] = [
        project_signed_pair(pair, arm["direction"], force, {})
        for pair, arm in zip(
            ordered_pairs(panel, pairs, verify_device=verify_device),
            panel["native_displacements"],
            strict=True,
        )
    ]
    check_center(report, panel)
    return report


def qualify_role(report: dict, role: str, panel: dict, shared_identity: dict) -> dict:
    """Mark only the work measured in this role; the complete endpoint requires the join."""
    if role == "center":
        report["native_directional_checks"] = []
        check_center(report, panel)
        checks = dict(report["checks"])
        del checks["native_benchmark_direction_two_steps"]
    elif role == "signed":
        ordered_pairs(
            panel,
            report["signed_pairs"],
            verify_device=True,
            step=report.get("selected_step_bohr"),
        )
        programs = report["graph_gpu_programs"]
        checks = {
            "requested_signed_energies": True,
            "physical_cuda": bool(programs)
            and all(p["backend"] == "cuda-fp64-ordinary-stream" for p in programs),
        }
    else:
        raise ValueError("unknown CC role")
    report.update(
        role=role,
        phase=role + "_component",
        role_passed=all(checks.values()),
        role_checks=checks,
        passed=False,
        role_identity=shared_identity,
    )
    return report


def join(panel_path: Path, center_path: Path, *signed_paths: Path) -> dict:
    panel = json.loads(panel_path.read_bytes())
    center = json.loads(center_path.read_bytes())
    ident = center["role_identity"]
    if (
        ident["files"]["panel.json"] != file_sha(panel_path)
        or ident["geometry_sha256"] != panel["geometry"]["geometry_sha256"]
    ):
        raise ValueError("incompatible scientific identities")
    signed_parts = [json.loads(path.read_bytes()) for path in signed_paths]
    if not signed_parts or len(signed_parts) > 2:
        raise ValueError("one or two signed components required")
    for role, result in [
        ("center", center),
        *(("signed", part) for part in signed_parts),
    ]:
        if result.get("role_identity") != ident:
            raise ValueError("incompatible scientific identities")
        if (
            result.get("case") != panel["molecule"]
            or result.get("role") != role
            or result.get("phase") != role + "_component"
            or not result.get("role_passed")
        ):
            raise ValueError(f"missing qualified {role} component")
        if (
            role == "signed"
            and not qualify_role(result, role, panel, ident)["role_passed"]
        ):
            raise ValueError("signed-pair physical CUDA check failed")
    pairs = [row for part in signed_parts for row in part["signed_pairs"]]
    center_channels = set(center["forward_resources"]["integrals"])
    if any(
        set(point["integral_device_bytes"]) != center_channels
        for row in pairs
        for point in row["energy_components"]
    ):
        raise ValueError("signed and center integral channels differ")
    report = join_measurements(panel, center, pairs, verify_device=True)
    for key in ("role", "role_checks", "role_passed"):
        report.pop(key, None)
    report["component_receipt_sha256"] = {
        "center": file_sha(center_path),
        "signed": [file_sha(path) for path in signed_paths],
    }
    report["center_plus_signed_compute_seconds"] = center["central_seconds"] + sum(
        part["composition_seconds"] for part in signed_parts
    )
    report["scope"] = (
        "Restricted CC-F12b central energy, analytic force and both signed step sizes. "
        "Independent full Cartesian validation is recorded separately."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "center", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--signed", required=True, type=Path, nargs="+")
    args = parser.parse_args()
    result = join(args.panel, args.center, *args.signed)
    write_measurement(args.output, result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
