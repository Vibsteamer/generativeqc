"""A resolved physical UHF MP2-F12 nuclear force for a supplied Cartesian direction."""

import argparse
import ctypes
import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from generativeqc._f12_cabs import cabs_singles
from generativeqc._f12_spin_force import spin_source_force
from generativeqc._f12_unrestricted import unrestricted_mp2_f12_source_pullback

from benchmarks._f12_cuda import add_molecular_graph_budget, cuda_runner


def evaluate(directory: Path, run: Callable, singles: Callable) -> dict:
    """Use common orbital/metric and Gaussian tangents from the same reoptimized UHF root."""
    ref = json.loads((directory / "reference.json").read_text())
    t0 = time.monotonic()
    with np.load(directory / "central.npz", allow_pickle=False) as source:
        values = dict(source)
    space = tuple(map(int, values["space"]))
    if space[:2] != (5, 4):
        raise ValueError(
            "Expected five occupied alpha and four occupied beta orbitals of NH2"
        )
    bars = unrestricted_mp2_f12_source_pullback(
        space, values, run, singles=singles, auxiliary_batch_size=64
    )
    elapsed = time.monotonic() - t0
    with np.load(directory / "tangent.npz", allow_pickle=False) as tangent:
        force = spin_source_force(
            bars,
            tangent,
            float(tangent["reference_gradient"]),
            run,
            block_elements=1 << 20,
        )
    points = ref["displacements"]
    if len(points) != 2 or len({row["step_bohr"] for row in points}) != 2:
        raise ValueError(
            "Expected independently reoptimized energies at two distinct steps"
        )
    errors = []
    for row in points:
        h = row["step_bohr"]
        if h <= 0 or not np.isfinite(h):
            raise ValueError("Displacement step must be finite and positive")
        fd = -(row["plus_Eh"] - row["minus_Eh"]) / (2 * h)
        errors.append(abs(force - fd))
    if not np.isfinite(errors).all() or max(errors) > 3e-7:
        raise ValueError(
            f"Nuclear force {force} failed its independent displacement checks {errors}"
        )
    return {
        "case": ref["case"],
        "atom": ref["atom"],
        "axis": ref["axis"],
        "full_spin_pairs": 36,
        "force_Eh_per_bohr": force,
        "two_step_errors_Eh_per_bohr": errors,
        "adjoint_seconds": elapsed,
        "elapsed_seconds": time.monotonic() - t0,
        "active_source_elements": sum(v.size for v in bars.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--component-library", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--output", required=True, type=Path)
    add_molecular_graph_budget(parser)
    args = parser.parse_args()
    run, metrics = cuda_runner(args.arch, args.cache, max_bytes=args.budget_mib << 20)
    row = evaluate(
        args.fixtures,
        run,
        partial(cabs_singles, library=ctypes.CDLL(args.component_library)),
    )
    args.output.write_text(
        json.dumps({"rows": [row], "programs": metrics}, indent=2, allow_nan=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
