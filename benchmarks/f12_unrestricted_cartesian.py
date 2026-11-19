"""All nine analytic Cartesian forces of bent NH2 using a shared UHF F12 pullback."""

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
from generativeqc.f12_orbital_response import unrestricted_orbital_forces

from benchmarks._f12_cuda import add_molecular_graph_budget, cuda_runner


def evaluate(
    directory: Path,
    run: Callable | None,
    singles: Callable | None,
    checkpoint: Callable,
    *,
    orbital_response: Path | None = None,
    library: ctypes.CDLL | None = None,
    maximum_bytes: int = 1536 << 20,
    response_output: Path | None = None,
) -> list[dict]:
    start = time.monotonic()
    if orbital_response is None:
        with np.load(directory / "central.npz", allow_pickle=False) as archive:
            data = dict(archive)
        spec = tuple(map(int, data["space"]))
        bars = unrestricted_mp2_f12_source_pullback(spec, data, run, singles=singles)
        metrics = {"source_gradient_elements": sum(v.size for v in bars.values())}
        orbital = None
    else:
        with np.load(orbital_response, allow_pickle=False) as archive:
            spec = tuple(map(int, archive["space"]))
            orbital = unrestricted_orbital_forces(
                spec,
                archive,
                archive["electronic_driver"],
                archive["nuclear_rhs"],
                archive["reference_forces"],
                archive["fixed_gradients"],
                library=library,
                maximum_bytes=maximum_bytes,
            )
        metrics = {
            "orbital_rotations": len(orbital.rotations),
            "orbital_gradient_Eh_per_bohr": orbital.orbital_gradient.tolist(),
        }
        if response_output is not None:
            np.savez(
                response_output,
                rotations=orbital.rotations,
                orbital_gradient=orbital.orbital_gradient,
                forces=orbital.forces,
            )
    if spec[:2] != (5, 4):
        raise ValueError(
            "The distorted NH2 doublet requires distinct alpha/beta occupied spaces"
        )
    ref = json.loads((directory / "reference.json").read_bytes())
    coords = np.asarray(ref["geometry_bohr"])
    expected = np.asarray(ref["force_Eh_per_bohr"])
    forces = np.empty_like(expected)
    rows = []
    for atom, axis in np.ndindex(3, 3):
        tag = f"{atom}{axis}"
        if orbital is None:
            with np.load(
                directory / ("tangent_" + tag + ".npz"), allow_pickle=False
            ) as tan:
                force = spin_source_force(
                    bars,
                    tan,
                    float(tan["reference_gradient"]),
                    run,
                    block_elements=1 << 20,
                )
        else:
            force = float(orbital.forces[atom, axis])
        errors = {
            h: abs(force + value) for h, value in ref["finite_gradients"][tag].items()
        }
        direct_error = abs(force - expected[atom, axis])
        if not np.isfinite(force) or max(direct_error, *errors.values()) > 3e-7:
            raise ValueError(
                f"Analytical force {tag} differs from independent response or displacement: {direct_error}, {errors}"
            )
        forces[atom, axis] = force
        rows.append(
            {
                "case": "NH2_D_f12",
                "atom": atom,
                "axis": axis,
                "force_Eh_per_bohr": force,
                "direct_response_difference": direct_error,
                "max_two_step_difference": max(errors.values()),
            }
        )
        checkpoint(rows)
    net = np.max(abs(forces.sum(axis=0)))
    torque = np.max(abs(np.cross(coords, forces).sum(axis=0)))
    if max(net, torque, np.max(abs(forces[:, 1]))) > 3e-7:
        raise ValueError(
            "Nuclear forces violate translation, rotation, or mirror symmetry"
        )
    rows.append(
        {
            "case": "NH2_D_f12",
            "force_Eh_per_bohr": forces.tolist(),
            "pairs": 36,
            "cartesian_components": 9,
            "net_force_max": float(net),
            "torque_max": float(torque),
            **metrics,
            "elapsed_seconds_including_graph_compiles": time.monotonic() - start,
        }
    )
    checkpoint(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--component-library", required=True)
    parser.add_argument("--arch")
    parser.add_argument("--cache", type=Path)
    parser.add_argument(
        "--orbital-response",
        type=Path,
        help="Solve the coupled orbital adjoint and assemble the Cartesian forces on the device",
    )
    parser.add_argument(
        "--response-output",
        type=Path,
        help="Save the adjoint, orbital gradient and final forces",
    )
    parser.add_argument("--output", type=Path, required=True)
    add_molecular_graph_budget(parser)
    args = parser.parse_args()
    if args.orbital_response is None and (args.arch is None or args.cache is None):
        parser.error(
            "--arch and --cache are required for the source-tangent evaluation"
        )
    if args.response_output is not None and args.orbital_response is None:
        parser.error("--response-output requires --orbital-response")
    run, metrics = (
        cuda_runner(args.arch, args.cache, max_bytes=args.budget_mib << 20)
        if args.orbital_response is None
        else (None, [])
    )
    library = ctypes.CDLL(args.component_library)

    def checkpoint(rows: list[dict]) -> None:
        tmp = args.output.with_suffix(".tmp.json")
        tmp.write_text(
            json.dumps({"rows": rows, "programs": metrics}, indent=2, allow_nan=False)
            + "\n"
        )
        tmp.replace(args.output)

    evaluate(
        args.fixtures,
        run,
        partial(cabs_singles, library=library),
        checkpoint,
        orbital_response=args.orbital_response,
        library=library,
        maximum_bytes=args.budget_mib << 20,
        response_output=args.response_output,
    )


if __name__ == "__main__":
    main()
