"""Join molecular F12/CABS cotangents to same-space canonical multipliers.

The prescribed orbital rotations are independent eigensolver test inputs.
All F12/CABS values, pullbacks and canonical multipliers execute on CUDA.
This fixed-source qualification does not include physical SCF Z or nuclear forces.
"""

import argparse
import ctypes
import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from generativeqc._canonical_frame import canonical_frame_seed
from generativeqc._f12_cabs import cabs_projector, cabs_singles
from generativeqc_compiler.method.canonical_frame_response import (
    orbital_rotation_cotangent,
)

from benchmarks._f12_cuda import add_molecular_graph_budget, cuda_runner
from benchmarks.f12_molecular_frame import chain


def check(
    name: str,
    data: dict,
    reference: dict,
    run: Callable,
    projector: Callable,
    singles: Callable,
    canonical: Callable,
) -> dict:
    started = time.monotonic()
    energy, bars, _ = chain(data, run, projector, singles, pullback=True)
    occupied, obs, _ = map(int, data["space"])
    coefficients = data["coefficients"][:, :obs]
    rotation = run(
        orbital_rotation_cotangent(*coefficients.shape),
        {"coefficients": coefficients, "frame_bar": bars["coefficients"]},
    ).outputs["rotation_gradient"]
    eps = reference["energies"]
    seed, owned_bytes = canonical(rotation, eps, occupied, maximum_bytes=43 << 30)
    same_space = np.equal.outer(np.arange(obs) < occupied, np.arange(obs) < occupied)
    gap = eps[:, None] - eps[None, :]
    torque = rotation - rotation.T
    residual = torque + 2 * gap * seed
    maximum_residual = float(abs(residual[same_space]).max())
    if not np.isfinite(seed).all() or maximum_residual > 1e-9:
        raise ValueError(f"{name}: molecular same-space stationarity failed")
    np.testing.assert_allclose(seed, seed.T, atol=3e-14, rtol=0)
    np.testing.assert_array_equal(seed[~same_space], 0)
    np.testing.assert_array_equal(np.diag(seed), 0)
    scalar_error = float(abs(energy - reference["energy"]))
    if not np.isfinite(energy) or scalar_error > 5e-11:
        raise ValueError(f"{name}: fixed-source molecular scalar mismatch")
    rows = []
    for index, direction in enumerate(reference["directions"]):
        slope = float(np.vdot(seed, direction))
        checks = []
        for step_index, step in enumerate(reference["steps"]):
            values = []
            for sign_index in range(2):
                moved = data["coefficients"].copy()
                moved[:, :obs] = (
                    coefficients @ reference["rotations"][index, step_index, sign_index]
                )
                values.append(
                    chain(
                        {**data, "coefficients": moved},
                        run,
                        projector,
                        singles,
                        pullback=False,
                    )[0]
                )
            native_fd = (values[0] - values[1]) / (2 * step)
            row = {
                "step": float(step),
                "native_fd_error": abs(native_fd - slope),
                "independent_eigensolver_fd_error": float(
                    abs(reference["slopes"][index, step_index] - slope)
                ),
                "fixed_source_scalar_error": float(
                    abs(
                        np.asarray(values) - reference["values"][index, step_index]
                    ).max()
                ),
            }
            if (
                not all(np.isfinite(v) for v in row.values())
                or row["native_fd_error"] > 2e-8
                or row["independent_eigensolver_fd_error"] > 2e-8
                or row["fixed_source_scalar_error"] > 5e-11
            ):
                raise ValueError(f"{name}: canonical composition failed: {row}")
            checks.append(row)
        rows.append(
            {
                "space": str(reference["spaces"][index]),
                "slope": slope,
                "steps": checks,
            }
        )
    return {
        "case": name,
        "scope": "fixed AO integrals and prescribed same-space eigenspaces; molecular F12+singles canonical source, not SCF Z or nuclear force",
        "energy_Eh": energy,
        "fixed_source_scalar_error_Eh": scalar_error,
        "maximum_same_space_residual": maximum_residual,
        "maximum_uncorrected_same_space_torque": float(abs(torque[same_space]).max()),
        "degenerate_block_torque": float(
            abs(torque[same_space & (abs(gap) <= 1e-8)]).max()
        ),
        "canonical_owned_bytes": owned_bytes,
        "directions": rows,
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
    add_molecular_graph_budget(parser)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.component_library))
    run, programs = cuda_runner(args.arch, args.cache, max_bytes=args.budget_mib << 20)
    rows = []
    with (
        np.load(args.fixture, allow_pickle=False) as source,
        np.load(args.reference, allow_pickle=False) as refs,
    ):
        for name in ("h2", "water"):
            data = {
                key.split("/", 1)[1]: source[key]
                for key in source.files
                if key.startswith(name + "/")
            }
            reference = {
                key.split("/", 1)[1]: refs[key]
                for key in refs.files
                if key.startswith(name + "/")
            }
            data["ao_permutation"] = reference["ao_permutation"]
            rows.append(
                check(
                    name,
                    data,
                    reference,
                    run,
                    partial(cabs_projector, library=lib),
                    partial(cabs_singles, library=lib, spin_weight=2),
                    partial(canonical_frame_seed, library=lib),
                )
            )
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
