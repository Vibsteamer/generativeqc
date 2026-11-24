"""Replay only the central molecular F12 frame derivative with its true energy cotangent."""

import argparse
import json
import time
from collections.abc import Callable

import numpy as np

from benchmarks.f12_molecular_force_composition import (
    calculate_panel,
    endpoint_arguments,
    write_measurement,
)


def measure(args: argparse.Namespace, target: dict) -> dict:
    started = time.monotonic()
    raw_path = args.output.parent / "physical_frame.npz"

    def component(
        data: dict, reverse: Callable, programs: list, forward_cost: dict
    ) -> dict:
        def record(
            overlap: np.ndarray,
            reference: np.ndarray,
            frame: np.ndarray,
            cotangent: np.ndarray,
            prior: np.ndarray | None,
            candidate: np.ndarray,
            checks: dict,
        ) -> None:
            if (
                prior is None
                or not np.isfinite(candidate).all()
                or not np.isfinite(prior).all()
            ):
                raise ValueError("Both physical comparison weights are required")
            np.savez_compressed(
                raw_path,
                overlap=overlap,
                orbital_frame=reference,
                frame=frame,
                frame_bar=cotangent,
                prior_overlap_bar=prior,
                constraint_overlap_bar=candidate,
            )

        t0 = time.monotonic()
        increment, _, checks = reverse(
            data, pullback=True, full_rank_constraint=True, frame_observer=record
        )
        passed = bool(
            raw_path.is_file()
            and np.isfinite(increment)
            and programs
            and all(p["backend"] == "cuda-fp64-ordinary-stream" for p in programs)
        )
        if not passed:
            raise ValueError("Physical CUDA frame replay incomplete")
        return {
            "case": target["molecule"],
            "geometry_sha256": target["geometry"]["geometry_sha256"],
            "increment_Eh": increment,
            "passed": True,
            "checks": {"physical_cuda_frame_replay": True},
            "frame_constraints": checks,
            "central_s": time.monotonic() - started,
            "reverse_and_comparison_s": time.monotonic() - t0,
            "graph_gpu_programs": programs,
            "forward_resources": forward_cost,
            "scope": "Single shared central E/F source and its real molecular frame energy weight; all original full endpoint and displaced calculations are reused.",
        }

    return calculate_panel(args, target, central_component=component)


if __name__ == "__main__":
    args = endpoint_arguments()
    report = measure(args, json.loads(args.panel.read_bytes()))
    write_measurement(args.output, report)
