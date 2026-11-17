"""Compose native CABS overlap adjoints with real CUDA nuclear derivatives."""

import argparse
import ctypes
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc import Calculator, Primitive, Shell
from generativeqc._f12_cabs import cabs_projector
from generativeqc._one_electron_gradient import cuda_one_electron_gradient


def evaluate(fixture: Path, library: ctypes.CDLL) -> dict[str, Any]:
    report = {
        "scope": "native CABS projector and physical-atom overlap gradient; supplied primal overlaps; not full F12 molecular E/F",
        "rows": [],
    }
    with np.load(fixture, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        for name, case in metadata["cases"].items():
            start = time.monotonic()

            def get(key: str, prefix: str = name) -> np.ndarray:
                return data[prefix + "/" + key]

            s, b, obs = get("s"), get("b"), case["obs"]
            n = len(s)
            rank = case["aux"]
            primal = cabs_projector(
                s, obs, cotangent=np.outer(b, b), expected_rank=rank, library=library
            )
            full = np.zeros((n + case["probe"], n + case["probe"]))
            full[:n, :n] = primal.overlap_bar
            # Only one cross block is seeded. The native full-Frobenius
            # contract combines W_ij+W_ji exactly once, with no factor of two.
            full[:n, n:] = (2 * primal.kernel @ get("b_perp"))[:, None] * get(
                "weights"
            )[None, :]
            shells = tuple(
                Shell(
                    s["atom_index"],
                    s["angular_momentum"],
                    tuple(Primitive(*p) for p in s["primitives"]),
                )
                for s in case["shells"]
            )
            calc = Calculator(
                "rhf",
                basis=shells,
                basis_representation="spherical",
                device="cuda",
                precision="fp64",
            )
            atoms = list(zip(case["elements"], case["coordinates_bohr"], strict=True))
            timings = []
            for _ in range(3):
                tick = time.monotonic()
                got = cuda_one_electron_gradient(calc, atoms, overlap=full)
                timings.append(time.monotonic() - tick)
            gradient = got.gradient
            error = float(np.max(abs(gradient - get("gradient"))))
            translation = float(np.max(abs(gradient.sum(axis=0))))
            fd_rows = []
            for k, step in enumerate(data["steps"]):
                refs = get("displaced_values")[k]
                fd = (refs[:, :, 0] - refs[:, :, 1]) / (2 * step)
                values = []
                for moved_s, moved_b in zip(
                    get("displaced_s")[k], get("displaced_b")[k], strict=True
                ):
                    p = cabs_projector(
                        moved_s, obs, expected_rank=rank, library=library
                    )
                    values.append(float(moved_b @ p.kernel @ moved_b))
                values = np.array(values).reshape(refs.shape)
                native_fd = (values[:, :, 0] - values[:, :, 1]) / (2 * step)
                fd_rows.append(
                    {
                        "step_bohr": float(step),
                        "independent_fd_error": float(np.max(abs(fd - gradient))),
                        "native_fd_error": float(np.max(abs(native_fd - gradient))),
                        "independent_value_error": float(np.max(abs(values - refs))),
                    }
                )
            passed = (
                error <= 3e-8
                and translation <= 3e-10
                and all(
                    r["independent_fd_error"] <= 3e-8
                    and r["native_fd_error"] <= 3e-8
                    and r["independent_value_error"] <= 4e-10
                    for r in fd_rows
                )
            )
            report["rows"].append(
                {
                    "case": name,
                    "passed": passed,
                    "obs": obs,
                    "aux": rank,
                    "physical_atoms": len(atoms),
                    "maximum_angular_momentum": case["max_angular_momentum"],
                    "reference_gradient_error": error,
                    "translation_error": translation,
                    "gradient": gradient.tolist(),
                    "fd": fd_rows,
                    "nuclear_contraction_seconds": timings,
                    "native_resources": got.resources,
                    "component_and_fd_seconds": time.monotonic() - start,
                }
            )
    report["passed"] = all(r["passed"] for r in report["rows"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--component-library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(args.fixture, ctypes.CDLL(str(args.component_library)))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passed"]:
        raise SystemExit("CABS nuclear overlap component failed")


if __name__ == "__main__":
    main()
