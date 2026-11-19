"""Join independently returned CC-F12b E/F and signed energy pairs under one numerical protocol."""

import argparse
import json
import math
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np

from benchmarks.f12_cc_benchmark_center import check_center
from benchmarks.f12_molecular_force_composition import (
    project_signed_pair,
    write_measurement,
)
from benchmarks.f12_molecular_sink_test import sha256 as file_sha


def physical_gpu(arch: str) -> dict:
    """Identify the single device actually assigned to this native measurement."""
    cmd = ["nvidia-smi", "--query-gpu=name,uuid,compute_cap", "--format=csv,noheader"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible != "all":
        if "," in visible:
            raise ValueError("each CC role requires a single visible GPU")
        cmd.extend(["-i", visible])
    rows = subprocess.check_output(cmd, text=True).strip().splitlines()
    if len(rows) != 1:
        raise ValueError("expected one physical GPU for a CC role")
    name, uuid, cap = [field.strip() for field in rows[0].split(",")]
    if not uuid.startswith("GPU-") or "sm_" + cap.replace(".", "") != arch:
        raise ValueError("physical GPU and configured kernel architecture disagree")
    return {"uuid": uuid, "name": name, "arch": arch}


def identity(args: argparse.Namespace, panel: dict) -> dict:
    """Bind both job roles to the geometry, shell layout, sources, runtime and solver options."""
    inputs = args.scf_inputs
    case = json.loads((inputs / "cases.json").read_bytes())
    files = case["files"]
    hashes = dict(panel["scf_input_sha256"])
    sign = {"plus": 1, "minus": -1, None: None}[args.cc_sign]
    if args.cc_role == "center":
        required_scf = [panel["molecule"] + "_central.in"]
    else:
        required_scf = [
            filename
            for arm in panel["native_displacements"]
            if args.cc_step is None or arm["step"] == args.cc_step
            for value, filename in zip((1, -1), arm["files"], strict=True)
            if sign is None or sign == value
        ]
    dynamic_libraries = sorted(name for name in files if name.endswith(".so"))
    for name in (
        "source.tar.gz",
        "runtime_source.tar.gz",
        *dynamic_libraries,
        *required_scf,
    ):
        actual = file_sha(inputs / name)
        if actual != files[name] or (name in hashes and actual != hashes[name]):
            raise ValueError(f"mismatched CC input digest: {name}")
        hashes[name] = actual
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
    panel: dict,
    rows: list[dict],
    *,
    verify_device: bool,
    step: float | None = None,
    sign: int | None = None,
) -> list[dict]:
    """Collect arbitrary arrivals by direction, step and sign and reconstruct each ordered pair."""
    if (
        step not in (None, 0.0004, 0.0002)
        or sign not in (None, 1, -1)
        or (sign is not None and step is None)
    ):
        raise ValueError("unexpected CC displacement selection")
    arms = {(a["name"], a["step"]): a for a in panel["native_displacements"]}
    if len(arms) != 2 or {h for _, h in arms} != {0.0004, 0.0002}:
        raise ValueError("the original CC panel requires both distinct step sizes")
    expected = {
        (*key, s)
        for key in arms
        if step is None or key[1] == step
        for s in (1, -1)
        if sign is None or s == sign
    }
    points = {}
    seconds = dict.fromkeys(arms, 0.0)
    for row in rows:
        key = (row["direction"], row["step_bohr"])
        observed_signs = [r["sign"] for r in row["energy_components"]]
        if key not in arms or observed_signs not in ([1], [-1], [1, -1]):
            raise ValueError("unexpected signed energy or unordered pair")
        seconds[key] += row.get("seconds", 0.0)
        arm = arms[key]
        for component, energy in zip(
            row["energy_components"], row["energies_Eh"], strict=True
        ):
            s = component["sign"]
            point = (*key, s)
            if point in points or point not in expected:
                raise ValueError("missing, duplicate or unexpected signed point")
            idx = 0 if s == 1 else 1
            total = math.fsum(
                component[k] for k in panel["cc_reference"]["components_Eh"]
            )
            if (
                not 0 <= component["cc_residual"] <= 1e-10
                or not math.isclose(total, energy, rel_tol=0, abs_tol=2e-12)
                or not math.isclose(
                    component["reference_energy_Eh"],
                    arm["hf_energies_Eh"][idx],
                    rel_tol=0,
                    abs_tol=5e-11,
                )
            ):
                raise ValueError("signed HF, CC root, sign or total disagrees")
            references = panel.get("cc_signed_reference")
            if references is not None:
                oracle = references[arm["files"][idx]]
                if (
                    not math.isclose(
                        energy, oracle["energy_Eh"], rel_tol=0, abs_tol=5e-11
                    )
                    or set(oracle["components_Eh"])
                    != set(panel["cc_reference"]["components_Eh"])
                    or any(
                        not math.isclose(component[k], value, rel_tol=0, abs_tol=5e-11)
                        for k, value in oracle["components_Eh"].items()
                    )
                ):
                    raise ValueError(
                        "signed CC energy or terms disagree with the matched independent root"
                    )
            if verify_device:
                device = component.get("integral_device_bytes", {})
                if len(device) != 10 or not all(v > 0 for v in device.values()):
                    raise ValueError("missing signed GPU integral evidence")
                if len(observed_signs) == 1 or "scf_filename" in component:
                    filename = arm["files"][idx]
                    if (
                        component.get("scf_filename") != filename
                        or component.get("scf_input_sha256")
                        != panel["scf_input_sha256"][filename]
                    ):
                        raise ValueError(
                            "single point has the wrong displaced SCF input identity"
                        )
            points[point] = (energy, component)
    if set(points) != expected:
        raise ValueError("missing, duplicate or unexpected signed point")
    joined = []
    for name, h in arms:
        if step is not None and h != step:
            continue
        values = [points[name, h, s] for s in (1, -1) if sign is None or s == sign]
        joined.append(
            {
                "direction": name,
                "step_bohr": h,
                "energies_Eh": [p[0] for p in values],
                "energy_components": [p[1] for p in values],
                "seconds": seconds[name, h],
            }
        )
    return joined


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
    independent = panel["cc_reference"].get("independent_cartesian_forces_Eh_per_bohr")
    if independent is not None:
        errors = {}
        if set(independent) != {"0.0004", "0.0002"}:
            raise ValueError("both independent Cartesian force steps required")
        for step, values in independent.items():
            reference = np.asarray(values)
            if reference.shape != force.shape or not np.isfinite(reference).all():
                raise ValueError("invalid independent Cartesian reference")
            errors[step] = float(np.max(np.abs(force - reference)))
        report["independent_cartesian_errors_Eh_per_bohr"] = errors
        report["checks"]["independent_cartesian_two_steps"] = (
            max(errors.values()) <= 2e-6
        )
        report["independent_cartesian_status"] = (
            "qualified"
            if report["checks"]["independent_cartesian_two_steps"]
            else "review"
        )
        report["passed"] = all(report["checks"].values())
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
            sign=report.get("selected_sign"),
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
    if not signed_parts or len(signed_parts) > 4:
        raise ValueError("one to four non-overlapping signed components required")
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
    # Earlier intact/pair workers did not expose a role-local UUID.
    if any(part.get("selected_sign") is not None for part in signed_parts):
        for part in (center, *signed_parts):
            gpu = part.get("physical_gpu", {})
            if (
                not gpu.get("uuid", "").startswith("GPU-")
                or not gpu.get("name")
                or gpu.get("arch") != ident.get("arch")
            ):
                raise ValueError(
                    "missing or incompatible physical GPU identity for point-wise qualification"
                )
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
    report["physical_gpus_by_role"] = {
        "center": center.get("physical_gpu"),
        "signed": [
            {
                "step_bohr": part.get("selected_step_bohr"),
                "sign": part.get("selected_sign"),
                "device": part.get("physical_gpu"),
            }
            for part in signed_parts
        ],
    }
    report["center_plus_signed_compute_seconds"] = center["central_seconds"] + sum(
        part["composition_seconds"] for part in signed_parts
    )
    report["scope"] = (
        "Restricted CC-F12b central energy, analytic force and both signed step sizes. "
        "See independent_cartesian_status for coverage of all Cartesian coordinates."
    )
    return report


def replay_complete(panel_path: Path, stage: Path) -> dict:
    """Use a full legacy archive only with its immutable input and physical L20 receipt."""
    case_path = stage / "inputs/cases.json"
    case = json.loads(case_path.read_bytes())
    manifest = json.loads((stage / "download/output_manifest.json").read_bytes())
    arc = stage / "download/results.tar.gz"
    if (
        file_sha(panel_path) != case["files"]["panel.json"]
        or file_sha(case_path) != manifest["input_sha256"]
        or file_sha(arc) != manifest["archive_sha256"]
        or case["source_commit"] != manifest["source_commit"]
        or not manifest["scientific_success"]
    ):
        raise ValueError("legacy scientific archive or input identity mismatch")
    with tarfile.open(arc) as archive:
        summary = json.load(archive.extractfile("summary.json"))
        full = json.load(archive.extractfile("measurements.json"))
        archived_case = json.load(archive.extractfile("input.json"))
    panel = json.loads(panel_path.read_bytes())
    if (
        archived_case != case
        or summary["gpu"]["name"] != "NVIDIA L20"
        or not summary["passed"]
        or not full["passed"]
        or full["case"] != panel["molecule"]
    ):
        raise ValueError("legacy archive lacks the matched physical full endpoint")
    report = join_measurements(
        panel, full, full["native_directional_checks"], verify_device=False
    )
    if not report["passed"] or report["checks"] != full["checks"]:
        raise ValueError("legacy full endpoint no longer satisfies its archived gates")
    report["source_full_archive_sha256"] = manifest["archive_sha256"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("panel", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--center", type=Path)
    source.add_argument("--complete-stage", type=Path)
    parser.add_argument("--signed", type=Path, nargs="+")
    args = parser.parse_args()
    if args.complete_stage is not None:
        if args.signed:
            parser.error("--complete-stage cannot consume signed shards")
        result = replay_complete(args.panel, args.complete_stage)
    else:
        if not args.signed:
            parser.error("--center requires --signed")
        result = join(args.panel, args.center, *args.signed)
    write_measurement(args.output, result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
