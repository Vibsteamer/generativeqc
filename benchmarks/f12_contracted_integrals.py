"""Measure bounded contracted F12 sources against independent Libint2 shells."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc_compiler.common.compiler_cache import (
    resolve_compiler_cache,
    run_cached_compiler,
)
from generativeqc_compiler.integral.f12_contracted import emit_f12_df_contracted

ROOT = Path(__file__).resolve().parents[1]


def adapter_source(groups: list[dict], primitives: int, projection: int) -> str:
    stride = 16 + 2 * primitives + projection
    functions, cases = [], []
    for i, group in enumerate(groups):
        components = tuple(tuple(s) for s in group["components"])
        functions.append(emit_f12_df_contracted(components, f"group_{i}"))
        cases.append(
            f"    case {i}: out[0] = group_{i}(counts, in+16, in+{16 + primitives},"
            f" in+7, in+{16 + 2 * primitives}, std::size_t(in[3]), op, fit, fit, out+1);"
            + (
                " if(out[0]) for(int k=8;k<11;++k) out[k]=0.;"
                if len(components[0]) == 2
                else ""
            )
            + " break;"
        )
    return "\n".join(
        [
            "#include <cmath>",
            "#include <cstddef>",
            "#include <cstdint>",
            '#include "integrals/f12_geminal.hpp"',
            "#ifdef __CUDACC__",
            "#define GQC_F12_COMPONENT_HD __host__ __device__",
            "#else",
            "#define GQC_F12_COMPONENT_HD",
            "#endif",
            *functions,
            "GQC_F12_COMPONENT_HD void evaluate(const double* in, double* out) {",
            "  namespace gi = generativeqc::integrals;",
            "  out[0]=0.; for(int i=1;i<11;++i) out[i]=123.5;",
            "  gi::FittedSlater fit;",
            "  if(!gi::make_fitted_slater(in[2],gi::SlaterFitNormalization::negative_exponential,fit)) return;",
            "  const int counts[3]={int(in[4]),int(in[5]),int(in[6])};",
            "  const auto op=static_cast<gi::GeminalProduct>(int(in[1]));",
            "  switch(int(in[0])) {",
            *cases,
            "  }",
            "}",
            "#ifdef __CUDACC__",
            "__global__ void rows(int count,const double* in,double* out) {",
            "  const int i=blockIdx.x*blockDim.x+threadIdx.x;",
            f"  if(i<count) evaluate(in+{stride}*i,out+11*i);",
            "}",
            "#endif",
            'extern "C" int f12_contracted_rows(int count,const double* in,double* out) {',
            "#ifdef __CUDACC__",
            "  double *a=nullptr,*b=nullptr;",
            f"  const auto ni=std::size_t(count)*{stride}*sizeof(double), no=std::size_t(count)*11*sizeof(double);",
            "  auto status=cudaMalloc(&a,ni);",
            "  if(status==cudaSuccess) status=cudaMalloc(&b,no);",
            "  if(status==cudaSuccess) status=cudaMemcpy(a,in,ni,cudaMemcpyHostToDevice);",
            "  if(status==cudaSuccess) { rows<<<(count+31)/32,32>>>(count,a,b);status=cudaGetLastError(); }",
            "  if(status==cudaSuccess) status=cudaMemcpy(out,b,no,cudaMemcpyDeviceToHost);",
            "  if(b) cudaFree(b); if(a) cudaFree(a); return int(status);",
            "#else",
            f"  for(int i=0;i<count;++i) evaluate(in+{stride}*i,out+11*i);return 0;",
            "#endif",
            "}",
            "",
        ]
    )


def check(fixture: dict[str, Any], build: Path, arch: str | None) -> dict[str, Any]:
    compiler = shutil.which("nvcc" if arch else "c++")
    if compiler is None:
        raise RuntimeError("A compiler for the requested backend is required")
    launcher = resolve_compiler_cache()
    before = subprocess.check_output([str(launcher.path), "--show-stats"], text=True)
    rows = fixture["rows"]
    maximum_primitives = max(sum(row["counts"]) for row in rows)
    maximum_projection = max(len(row["projection"]) for row in rows)
    stride = 16 + 2 * maximum_primitives + maximum_projection
    inputs = np.zeros((len(rows), stride))
    work = []
    for i, row in enumerate(rows):
        primitives = sum(row["counts"])
        components = math.prod(row["counts"]) * len(row["projection"])
        work.append(components)
        inputs[i, :4] = row["group"], row["operator"], row["beta"], components
        inputs[i, 4 : 4 + len(row["counts"])] = row["counts"]
        inputs[i, 7 : 7 + 3 * len(row["counts"])] = np.asarray(row["centers"]).ravel()
        inputs[i, 16 : 16 + primitives] = row["exponents"]
        inputs[i, 16 + maximum_primitives : 16 + maximum_primitives + primitives] = row[
            "coefficients"
        ]
        inputs[
            i,
            16 + 2 * maximum_primitives : 16
            + 2 * maximum_primitives
            + len(row["projection"]),
        ] = row["projection"]
    build.mkdir(parents=True, exist_ok=True)
    source = build / ("contracted.cu" if arch else "contracted.cpp")
    source.write_text(
        adapter_source(fixture["groups"], maximum_primitives, maximum_projection)
    )
    obj, lib = build / "contracted.o", build / "contracted.so"
    flags = ["-arch=" + arch, "-Xcompiler=-fPIC"] if arch else ["-fPIC"]
    start = time.perf_counter()
    commands = [
        [
            compiler,
            "-std=c++17",
            "-O2",
            *flags,
            "-I" + str(ROOT / "src"),
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        [compiler, "-shared", str(obj), "-o", str(lib)],
    ]
    compiler_runs = []
    for command in commands:
        result = run_cached_compiler(
            command, 600.0, label="F12 contracted sources", launcher=launcher
        )
        compiler_runs.append({"command": command, **asdict(result)})
        if result.returncode or result.timed_out:
            raise RuntimeError(
                result.stderr or result.stdout or "Source compilation failed"
            )
    compile_seconds = time.perf_counter() - start
    function = ctypes.CDLL(str(lib)).f12_contracted_rows
    array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
    function.argtypes = (ctypes.c_int, array, array)
    function.restype = ctypes.c_int

    def execute(data: np.ndarray, accepted: bool = True) -> np.ndarray:
        output = np.zeros((len(data), 11))
        code = function(len(data), np.ascontiguousarray(data), output)
        if code:
            raise RuntimeError(f"Native contracted source failed with code {code}")
        np.testing.assert_array_equal(output[:, 0], int(accepted))
        return output[:, 1:]

    start = time.perf_counter()
    observed = execute(inputs)
    cold = time.perf_counter() - start
    start = time.perf_counter()
    np.testing.assert_array_equal(execute(inputs), observed)
    warm = time.perf_counter() - start
    expected = np.asarray([row["value_and_gradient"] for row in rows])
    scale = np.maximum(1.0, np.max(np.abs(expected), axis=1))
    error = (observed - expected) / scale[:, None]
    np.testing.assert_allclose(error, 0.0, atol=3e-10, rtol=0)
    translation = observed[:, 1:].reshape(-1, 3, 3).sum(axis=1) / scale[:, None]
    np.testing.assert_allclose(translation, 0.0, atol=3e-12, rtol=0)
    step_errors = {}
    for step in (2e-4, 1e-4):
        moved = []
        for point, row in zip(inputs, rows, strict=True):
            for coordinate in range(3 * len(row["counts"])):
                for sign in (-1, 1):
                    change = point.copy()
                    change[7 + coordinate] += sign * step
                    moved.append(change)
        values = iter(execute(np.asarray(moved))[:, 0])
        slopes = np.zeros((len(rows), 9))
        for i, row in enumerate(rows):
            for coordinate in range(3 * len(row["counts"])):
                minus, plus = next(values), next(values)
                slopes[i, coordinate] = (plus - minus) / (2 * step)
        delta = (slopes - observed[:, 1:]) / scale[:, None]
        np.testing.assert_allclose(delta, 0.0, atol=4e-7, rtol=0)
        step_errors[str(step)] = float(np.max(np.abs(delta)))
    nuclear_errors = {}
    # Move all Gaussian factors on the same atom together. Their source
    # derivatives must add before the molecular nuclear contraction.
    for step in (2e-4, 1e-4):
        moved, nuclear_slopes, nuclear_scale = [], [], []
        for i, (point, row) in enumerate(zip(inputs, rows, strict=True)):
            atoms = [s["atom"] for s in fixture["groups"][row["group"]]["shells"]]
            for atom in sorted(set(atoms)):
                for axis in range(3):
                    incidences = [c for c, a in enumerate(atoms) if a == atom]
                    nuclear_slopes.append(
                        sum(observed[i, 1 + 3 * c + axis] for c in incidences)
                    )
                    nuclear_scale.append(scale[i])
                    for sign in (-1, 1):
                        change = point.copy()
                        for c in incidences:
                            change[7 + 3 * c + axis] += sign * step
                        moved.append(change)
        values = execute(np.asarray(moved))[:, 0].reshape(-1, 2)
        slopes = (values[:, 1] - values[:, 0]) / (2 * step)
        delta = (slopes - nuclear_slopes) / nuclear_scale
        np.testing.assert_allclose(delta, 0.0, atol=4e-7, rtol=0)
        nuclear_errors[str(step)] = float(np.max(np.abs(delta)))
    for column, value in ((3, 0), (4, 0), (16, -1.0), (7, float("nan"))):
        invalid = inputs.copy()
        invalid[:, column] = value
        np.testing.assert_array_equal(execute(invalid, False), 123.5)
    after = subprocess.check_output([str(launcher.path), "--show-stats"], text=True)
    return {
        "backend": "cuda-fp64" if arch else "host-fp64",
        "scope": "contracted scalar integral/projection and independent-center derivatives; not full molecular E/F",
        "groups": [g["name"] for g in fixture["groups"]],
        "source_rows": len(rows),
        "max_value_and_gradient_scaled_error": float(np.max(np.abs(error))),
        "translation_scaled_error": float(np.max(np.abs(translation))),
        "two_step_errors": step_errors,
        "two_step_shared_atom_errors": nuclear_errors,
        "invalid_inputs_preserve_output": True,
        "primitive_cartesian_evaluations_per_pass": sum(work),
        "compile_seconds": compile_seconds,
        "cold_batch_seconds": cold,
        "warm_batch_seconds": warm,
        "source_bytes": source.stat().st_size,
        "cache": {
            "path": str(launcher.path),
            "version": launcher.version,
            "before": before,
            "after": after,
            "compiler_runs": compiler_runs,
            "statistics_scope": "launcher diagnostics; privately leased JIT execution is not a shared-server hit claim",
        },
        "rows": [
            {
                "group": row["group"],
                "operator": row["operator"],
                "beta": row["beta"],
                "native_value_and_gradient": value.tolist(),
                "reference_value_and_gradient": row["value_and_gradient"],
            }
            for row, value in zip(rows, observed, strict=True)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch")
    args = parser.parse_args()
    if os.environ.get("GENERATIVEQC_TEST_CUDA") == "1" and not args.arch:
        raise ValueError(
            "Allocated GPU qualification requires an explicit CUDA architecture"
        )
    result = check(json.loads(args.fixture.read_text()), args.cache, args.arch)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
