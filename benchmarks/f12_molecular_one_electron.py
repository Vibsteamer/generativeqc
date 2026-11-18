"""Contract actual molecular F12 overlap and hcore adjoints on physical atoms."""

import argparse
import ctypes
import json
import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc import Calculator, Primitive, Shell
from generativeqc._f12_cabs import cabs_projector, cabs_singles
from generativeqc._one_electron_gradient import (
    IntegralGradient,
    cuda_one_electron_gradient,
)

from benchmarks._f12_cuda import cuda_runner
from benchmarks.f12_molecular_frame import chain


def check(
    name: str,
    data: dict,
    case: dict,
    ref: dict,
    run: Callable,
    projector: Callable,
    singles: Callable,
    contract: Callable,
) -> dict:
    start = time.monotonic()
    energy, bars, _ = chain(data, run, projector, singles, pullback=True)
    p = ref["native_to_source"]
    size = len(data["overlap"])
    if (
        p.dtype.kind not in "iu"
        or p.shape != (size,)
        or not np.array_equal(np.sort(p), np.arange(size))
    ):
        raise ValueError("nuclear derivative contraction requires a bijective AO map")
    ix = np.ix_(p, p)
    ws, wh = (bars[k][ix] for k in ("overlap", "hcore_ao"))
    weights = np.stack([ws, wh])
    analytic = np.einsum("axkij,kij->ax", ref["derivatives"], weights)
    before = time.monotonic()
    measured, resources = contract(case, bars["overlap"], bars["hcore_ao"], p)
    contraction_time = time.monotonic() - before
    if measured.shape != analytic.shape or not np.isfinite(measured).all():
        raise ValueError("invalid nuclear gradient dimensions or values")
    error = float(np.max(abs(measured - analytic)))
    translation = float(np.max(abs(measured.sum(axis=0))))
    # All other AO sources, including the OBS orbitals, stay at their central
    # values. Rebuild the complementary frame for each displaced S, hcore pair.
    inv = np.argsort(p)
    back = np.ix_(inv, inv)
    atom, axis = case["nonlinear_coordinate"]
    fd_rows = []
    for i, h in enumerate(ref["steps"]):
        linear = np.einsum("axkij,kij->ax", ref["fd"][i], weights)
        five = np.einsum("axkij,kij->ax", ref["fd_five"][i], weights)
        fd_rows.append(
            {
                "step_bohr": float(h),
                "three_point_error": float(np.max(abs(linear - measured))),
                "five_point_error": float(np.max(abs(five - measured))),
            }
        )
    nonlinear_rows = []
    previous = None
    for i, h in enumerate(ref["nonlinear_steps"]):
        values = []
        for sm, hm in ref["moved"][i]:
            moved = {**data, "overlap": sm[back], "hcore_ao": hm[back]}
            values.append(chain(moved, run, projector, singles, pullback=False)[0])
        nonlinear = (values[0] - values[1]) / (2 * h)
        five_error = None
        if previous is not None:
            five_error = float(
                abs((4 * nonlinear - previous) / 3 - measured[atom, axis])
            )
        nonlinear_rows.append(
            {
                "step_bohr": float(h),
                "three_point_error": float(abs(nonlinear - measured[atom, axis])),
                "five_point_error": five_error,
                "fixed_source_energies_Eh": list(map(float, values)),
            }
        )
        previous = nonlinear
    scalar_error = abs(float(energy) - float(ref["source_energy"]))
    passed = (
        scalar_error <= 5e-11
        and error <= 3e-8
        and translation <= 3e-10
        and len(fd_rows) == 2
        and len(nonlinear_rows) == 3
        and all(q["five_point_error"] <= 3e-8 for q in fd_rows + nonlinear_rows[1:])
    )
    return {
        "case": name,
        "passed": passed,
        "atoms": len(measured),
        "spherical_AOs": size,
        "source_F12_plus_singles_Eh": float(energy),
        "fixed_source_scalar_parity_Eh": scalar_error,
        "gradient_Eh_per_bohr": measured.tolist(),
        "max_error_vs_libcint_Eh_per_bohr": error,
        "max_translation_residual_Eh_per_bohr": translation,
        "gradient_norms_by_channel": [
            float(
                np.linalg.norm(
                    np.einsum("axij,ij->ax", ref["derivatives"][:, :, k], weights[k])
                )
            )
            for k in range(2)
        ],
        "nonlinear_coordinate": [atom, axis],
        "all_coordinate_differences": fd_rows,
        "nonlinear_differences": nonlinear_rows,
        "native_resources": resources,
        "contraction_seconds": contraction_time,
        "total_test_seconds": time.monotonic() - start,
    }


def native_contraction(
    case: dict, ws: np.ndarray, wh: np.ndarray, native_to_source: np.ndarray
) -> IntegralGradient:
    shells = tuple(
        Shell(
            s["atom_index"],
            s["angular_momentum"],
            tuple(Primitive(*prim) for prim in s["primitives"]),
        )
        for s in case["shells"]
    )
    calc = Calculator(
        "rhf",
        device="cuda",
        precision="fp64",
        basis=shells,
        basis_representation="spherical",
    )
    atoms = list(zip(case["elements"], case["coordinates_bohr"], strict=True))
    return cuda_one_electron_gradient(
        calc, atoms, overlap=ws, hcore=wh, native_to_source=native_to_source
    )


def cases(
    source: Any, frames: Any, refs: Any
) -> Iterator[tuple[str, dict, dict, dict]]:
    metadata = json.loads(str(refs["metadata"]))
    for name, case in metadata["cases"].items():
        prefix = name + "/"
        data = {
            k[len(prefix) :]: source[k] for k in source.files if k.startswith(prefix)
        }
        data["ao_permutation"] = frames[prefix + "ao_permutation"]
        per_case = {
            k[len(prefix) :]: refs[k] for k in refs.files if k.startswith(prefix)
        }
        per_case["source_energy"] = frames[prefix + "energy"]
        yield name, data, case, per_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "source",
        "frame",
        "reference",
        "component-library",
        "cache",
        "output",
    ):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--arch", required=True)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.component_library))
    projector = partial(cabs_projector, library=lib)
    singles = partial(cabs_singles, library=lib, spin_weight=2)
    run, programs = cuda_runner(args.arch, args.cache)
    with (
        np.load(args.source, allow_pickle=False) as source,
        np.load(args.frame, allow_pickle=False) as frames,
        np.load(args.reference, allow_pickle=False) as refs,
    ):
        rows = [
            check(*entry, run, projector, singles, native_contraction)
            for entry in cases(source, frames, refs)
        ]
    passed = (
        {row["case"] for row in rows} == {"h2", "water"}
        and bool(programs)
        and all(
            row["passed"]
            and row["native_resources"]["device_bytes"] > 0
            and row["native_resources"]["device_to_host_bytes"] > 0
            for row in rows
        )
    )
    args.output.write_text(
        json.dumps(
            {
                "passed": passed,
                "rows": rows,
                "programs": programs,
                "scope": "CUDA explicit S and T+V_nuc term of the molecular F12+singles gradient",
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    if not passed:
        raise SystemExit("molecular F12 one-electron gradient failed")


if __name__ == "__main__":
    main()
