"""Native CUDA same-space canonical multipliers before the physical RHF Z solve."""

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np
from generativeqc._canonical_frame import canonical_frame_seed
from generativeqc_compiler.method.canonical_frame_response import (
    orbital_rotation_cotangent,
)

from benchmarks._f12_cuda import cuda_runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--component-library", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.component_library))
    run, programs = cuda_runner(args.arch, args.cache)
    rows = []
    with np.load(args.fixture, allow_pickle=False) as archive:
        for name in sorted({k.split("/")[0] for k in archive.files}):
            case = {
                k.split("/", 1)[1]: archive[k]
                for k in archive.files
                if k.startswith(name + "/")
            }
            c, bar, eps = (case[k] for k in ("coefficients", "frame_bar", "energies"))
            o = int(case["occupied"])
            start = time.monotonic()
            m = run(
                orbital_rotation_cotangent(*c.shape),
                {"coefficients": c, "frame_bar": bar},
            ).outputs["rotation_gradient"]
            np.testing.assert_allclose(m, case["rotation_gradient"], atol=3e-13, rtol=0)
            seed, owned = canonical_frame_seed(
                m, eps, o, library=lib, maximum_bytes=43 << 30
            )
            assert owned > 0 and np.isfinite(seed).all()
            np.testing.assert_allclose(seed, seed.T, atol=3e-14, rtol=0)
            np.testing.assert_array_equal(seed[:o, o:], 0)
            np.testing.assert_array_equal(seed[o:, :o], 0)
            np.testing.assert_array_equal(np.diag(seed), 0)
            residual = m - m.T + 2 * (eps[:, None] - eps[None, :]) * seed
            error = max(
                float(abs(residual[:o, :o]).max()), float(abs(residual[o:, o:]).max())
            )
            assert error < 1e-9
            slope = float(np.vdot(seed, case["fock_direction_mo"]))
            fd = abs(case["slopes"] - slope)
            assert np.isfinite(fd).all() and fd.max() < 3e-8
            rows.append(
                {
                    "case": name,
                    "scope": "externally prescribed canonical eigenspaces, not SCF or nuclear force",
                    "maximum_same_space_residual": error,
                    "fd_errors": fd.tolist(),
                    "device_bytes": owned,
                    "wall_including_compile_s": time.monotonic() - start,
                }
            )
    # Native refusal must not write either user output, even after a CUDA launch.
    routine = lib.generativeqc_canonical_frame_seed_cuda_v1
    ptr = ctypes.POINTER(ctypes.c_double)
    failures = []
    for reason in ("budget", "degenerate", "nonfinite"):
        m = np.zeros((4, 4))
        eps = np.array([-1.0, 0.5, 0.5, 0.5])
        budget = 128 << 20
        if reason == "budget":
            budget = 1
        elif reason == "degenerate":
            m[1, 2] = 0.1
        else:
            m[2, 3] = np.nan
        out = np.full((4, 4), 12345.0)
        size = ctypes.c_size_t(6789)
        error = ctypes.create_string_buffer(512)
        status = routine(
            0,
            4,
            1,
            1e-8,
            1e-9,
            budget,
            eps.ctypes.data_as(ptr),
            m.ctypes.data_as(ptr),
            out.ctypes.data_as(ptr),
            ctypes.byref(size),
            error,
            len(error),
        )
        assert status and size.value == 6789 and np.all(out == 12345.0)
        failures.append({"case": reason, "error": error.value.decode()})
    args.output.write_text(
        json.dumps(
            {"passed": True, "rows": rows, "refusals": failures, "programs": programs},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
