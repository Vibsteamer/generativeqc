"""Measure the native CABS projection and check motion of atom-centred functions."""

import argparse
import ctypes
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc._f12_cabs import cabs_projector


def evaluate(fixture: Path, library: ctypes.CDLL) -> dict[str, Any]:
    report = {
        "cases": {},
        "scope": "native FP64 CABS component; independent supplied AO overlaps",
    }
    with np.load(fixture, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata"]))
        report["oracle"] = meta
        for name, record in meta["cases"].items():

            def field(key: str, prefix: str = name) -> np.ndarray:
                return f[prefix + "_" + key]

            s, b, ref = field("s"), field("b"), field("k")
            obs, rank = record["obs"], record["aux"]
            seed = np.outer(b, b)
            elapsed = []
            for _ in range(4):
                start = time.perf_counter()
                central = cabs_projector(
                    s, obs, expected_rank=rank, cotangent=seed, library=library
                )
                elapsed.append((time.perf_counter() - start) * 1e3)
            k = central.kernel
            # K S E_OBS = 0, so K b = K b_perp. Subtracting the OBS component
            # reduces roundoff in the physical probe expectation.
            slope = np.vdot(central.overlap_bar, field("ds")) + 2.0 * field("db") @ (
                k @ field("b_perp")
            )
            norm = np.linalg.norm(ref)
            measurements = {
                "obs": obs,
                "rank": central.rank,
                "cutoff_margin": central.cutoff_margin,
                "kernel_relative_frobenius": float(np.linalg.norm(k - ref) / norm),
                "frame_kernel_relative": float(
                    np.linalg.norm(central.frame @ central.frame.T - k) / norm
                ),
                "frame_metric_orthonormality": float(
                    np.max(np.abs(central.frame.T @ s @ central.frame - np.eye(rank)))
                ),
                "frame_obs_orthogonality": float(
                    np.max(np.abs((s @ central.frame)[:obs]))
                ),
                "metric_idempotence_relative": float(
                    np.linalg.norm(k @ s @ k - k) / norm
                ),
                "max_obs_overlap": float(np.max(np.abs((s @ k)[:obs]))),
                "slope_per_bohr": float(slope),
                "reference_slope_per_bohr": float(field("slope")),
                "native_central_vjp_wall_ms": elapsed,
                "steps": [],
            }
            report["cases"][name] = measurements
            for i, h in enumerate(f["steps_bohr"]):
                values = []
                refs = []
                for sign in ("minus", "plus"):
                    head = f"{i}_{sign}_"
                    moved = cabs_projector(
                        field(head + "s"), obs, expected_rank=rank, library=library
                    )
                    probe = field(head + "b_perp")
                    values.append(float(probe @ moved.kernel @ probe))
                    refs.append(float(field(head + "observable")))
                fd = (values[1] - values[0]) / (2 * h)
                ref_fd = (refs[1] - refs[0]) / (2 * h)
                measurements["steps"].append(
                    {
                        "h_bohr": float(h),
                        "native_values": values,
                        "reference_values": refs,
                        "native_fd_per_bohr": float(fd),
                        "reference_fd_per_bohr": float(ref_fd),
                        "native_fd_error": float(abs(fd - slope)),
                        "reference_fd_error": float(abs(ref_fd - slope)),
                        "max_value_error": max(
                            abs(u - v) for u, v in zip(values, refs, strict=True)
                        ),
                    }
                )
            measurements["passed"] = bool(
                measurements["kernel_relative_frobenius"] < 1e-8
                and measurements["frame_kernel_relative"] < 1e-8
                and measurements["frame_metric_orthonormality"] < 2e-8
                and measurements["frame_obs_orthogonality"] < 2e-8
                and measurements["metric_idempotence_relative"] < 1e-8
                and measurements["max_obs_overlap"] < 2e-8
                and abs(slope - field("slope")) < 3e-8
                and all(
                    step["native_fd_error"] < 3e-8
                    and step["reference_fd_error"] < 3e-8
                    and step["max_value_error"] < 4e-10
                    for step in measurements["steps"]
                )
            )
    report["passed"] = all(c["passed"] for c in report["cases"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--component-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    results = evaluate(a.fixture, ctypes.CDLL(str(a.component_library)))
    a.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    if not results["passed"]:
        raise SystemExit("CABS projector or overlap derivative failed validation")


if __name__ == "__main__":
    main()
