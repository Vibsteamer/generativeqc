"""Replace supplied CABS frames with native CUDA frames in the molecular chain."""

import argparse
import ctypes
import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc._f12_cabs import cabs_projector, cabs_singles

from benchmarks._f12_cuda import cuda_runner
from benchmarks.f12_ao_response import evaluate
from benchmarks.f12_frame_response import bridge


def with_frame(data: dict, projector: Callable) -> tuple[dict, Any]:
    """Use explicit OBS-first AO indices, preserving the original source order."""
    permutation = np.asarray(data["ao_permutation"])
    coefficients = data["coefficients"]
    atomic, orbitals = coefficients.shape
    obs = int(data["space"][1])
    if (
        permutation.dtype.kind not in "iu"
        or permutation.shape != (atomic,)
        or not np.array_equal(np.sort(permutation), np.arange(atomic))
    ):
        raise ValueError("CABS requires an explicit bijective AO permutation")
    if np.any(coefficients[permutation[obs:], :obs] != 0):
        raise ValueError("OBS columns disagree with the declared AO embedding")
    overlap = data["overlap"][np.ix_(permutation, permutation)]
    current = projector(overlap, obs, expected_rank=orbitals - obs)
    assembled = coefficients.copy()
    assembled[permutation, obs:] = current.frame
    return {**data, "coefficients": assembled}, current


def chain(
    data: dict, run: Callable, projector: Callable, singles: Callable, *, pullback: bool
) -> tuple[float, dict, dict]:
    inputs, frame = with_frame(data, projector)
    energy, bars, _ = evaluate(inputs, run, pullback=pullback, singles=singles)
    if not pullback:
        return energy, {}, {}
    permutation = data["ao_permutation"]
    obs = int(data["space"][1])
    overlap = data["overlap"][np.ix_(permutation, permutation)]
    transferred, checks = bridge(
        {
            "overlap": overlap,
            "frame": frame.frame,
            "frame_bar": bars["coefficients"][permutation, obs:],
        },
        run,
    )
    reverse = projector(
        overlap, obs, expected_rank=frame.rank, cotangent=transferred["kernel_bar"]
    )
    overlap_bar = np.zeros_like(overlap)
    overlap_bar[np.ix_(permutation, permutation)] = reverse.overlap_bar
    bars["overlap"] = overlap_bar
    # The OBS cotangent remains for SCF response; the CABS cotangent has now
    # been consumed by the projector. Do not pass it to the SCF owner again.
    bars["coefficients"] = bars["coefficients"][:, :obs]
    return energy, bars, checks


def check(
    name: str,
    data: dict,
    reference: dict,
    run: Callable,
    projector: Callable,
    singles: Callable,
) -> dict:
    started = time.monotonic()
    supplied_energy, _, _ = evaluate(data, run, pullback=False, singles=singles)
    energy, bars, invariants = chain(data, run, projector, singles, pullback=True)
    frame_change = abs(energy - supplied_energy)
    if not np.isfinite(energy) or not np.isfinite(frame_change) or frame_change > 5e-11:
        raise ValueError(
            f"{name}: native CABS frame changes the fixed-source scalar by {frame_change}"
        )
    permutation = data["ao_permutation"]
    direction = reference["overlap_direction"]
    slope = float(np.vdot(bars["overlap"], direction))
    errors = []
    for index, h in enumerate(reference["steps"]):
        moved_values = []
        for sign in (1, -1):
            moved = {**data, "overlap": data["overlap"] + sign * h * direction}
            moved_values.append(
                chain(moved, run, projector, singles, pullback=False)[0]
            )
        native_fd = (moved_values[0] - moved_values[1]) / (2 * h)
        independent_fd = reference["slopes"][index]
        errors.append(
            {
                "step": float(h),
                "native_fd_error": abs(native_fd - slope),
                "independent_fd_error": float(abs(independent_fd - slope)),
                "fixed_source_value_error": float(
                    np.max(abs(np.array(moved_values) - reference["values"][index]))
                ),
            }
        )
    if any(
        not all(np.isfinite(v) for v in r.values())
        or r["native_fd_error"] > 2e-8
        or r["independent_fd_error"] > 2e-8
        or r["fixed_source_value_error"] > 5e-11
        for r in errors
    ):
        raise ValueError(f"{name}: composed molecular frame response failed: {errors}")
    return {
        "case": name,
        "scope": "native forward CABS frame and F12+singles source/overlap adjoints; fixed OBS and AO integrals, not nuclear forces",
        "energy_Eh": energy,
        "supplied_frame_parity_Eh": frame_change,
        "comparison_to_stock_full_scalar_Eh": energy - float(reference["stock_energy"]),
        "ao_permutation_nontrivial": bool(
            np.any(permutation != np.arange(len(permutation)))
        ),
        "invariants": invariants,
        "overlap_direction_slope": slope,
        "steps": errors,
        "retained_scf_cotangent_shape": list(bars["coefficients"].shape),
        "test_and_generated_compile_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--component-library", required=True, type=Path)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.component_library))
    projector = partial(cabs_projector, library=lib)
    singles = partial(cabs_singles, library=lib, spin_weight=2)
    run, programs = cuda_runner(args.arch, args.cache)
    rows = []
    with (
        np.load(args.fixture, allow_pickle=False) as source,
        np.load(args.reference, allow_pickle=False) as refs,
    ):
        for name in ("h2", "water"):
            data = {
                key.removeprefix(name + "/"): source[key]
                for key in source.files
                if key.startswith(name + "/")
            }
            reference = {
                key.removeprefix(name + "/"): refs[key]
                for key in refs.files
                if key.startswith(name + "/")
            }
            data["ao_permutation"] = reference["ao_permutation"]
            rows.append(check(name, data, reference, run, projector, singles))
    args.output.write_text(
        json.dumps(
            {"passed": True, "rows": rows, "programs": programs},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
