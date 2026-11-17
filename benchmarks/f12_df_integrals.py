"""Qualify generated geminal DF primitives against archived Libint2 values."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc_compiler.integral.f12_df import emit_f12_df_component

ROOT = Path(__file__).resolve().parents[1]


def adapter_source(components: list[tuple[str, ...]]) -> str:
    """A batched test consumer for the same generated host/device functions."""
    functions = [
        emit_f12_df_component(signature, f"component_{i}")
        for i, signature in enumerate(components)
    ]
    cases = [
        f"    case {i}: out[0] = component_{i}(in+3,in+6,op,fit,fit,out+1); break;"
        for i in range(len(components))
    ]
    return (
        "\n".join(
            [
                '#include "integrals/f12_geminal.hpp"',
                "#include <cmath>",
                "#ifdef __CUDACC__",
                "#define GQC_F12_COMPONENT_HD __host__ __device__",
                "#else",
                "#define GQC_F12_COMPONENT_HD",
                "#endif",
                *functions,
                "GQC_F12_COMPONENT_HD void evaluate(const double* in, double* out) {",
                "  namespace gi = generativeqc::integrals;",
                "  for (int i=0;i<11;++i) out[i]=0.;",
                "  gi::FittedSlater fit;",
                "  if (!gi::make_fitted_slater(in[2],gi::SlaterFitNormalization::negative_exponential,fit)) return;",
                "  const auto op = static_cast<gi::GeminalProduct>(static_cast<int>(in[1]));",
                "  switch (static_cast<int>(in[0])) {",
                *cases,
                "  }",
                "}",
                "#ifdef __CUDACC__",
                "__global__ void rows(int count,const double* in,double* out) {",
                "  int i=blockIdx.x*blockDim.x+threadIdx.x;",
                "  if(i<count) evaluate(in+15*i,out+11*i);",
                "}",
                "#endif",
                'extern "C" int f12_df_rows(int count,const double* in,double* out) {',
                "#ifdef __CUDACC__",
                "  double *device_in=nullptr,*device_out=nullptr;",
                "  const auto input_bytes=count*15*sizeof(double), output_bytes=count*11*sizeof(double);",
                "  auto status=cudaMalloc(&device_in,input_bytes);",
                "  if(status==cudaSuccess) status=cudaMalloc(&device_out,output_bytes);",
                "  if(status==cudaSuccess) status=cudaMemcpy(device_in,in,input_bytes,cudaMemcpyHostToDevice);",
                "  if(status==cudaSuccess) {",
                "    rows<<<(count+31)/32,32>>>(count,device_in,device_out);",
                "    status=cudaGetLastError();",
                "  }",
                "  if(status==cudaSuccess) status=cudaMemcpy(out,device_out,output_bytes,cudaMemcpyDeviceToHost);",
                "  if(device_out) cudaFree(device_out);",
                "  if(device_in) cudaFree(device_in);",
                "  return static_cast<int>(status);",
                "#else",
                "  for(int i=0;i<count;++i) evaluate(in+15*i,out+11*i);",
                "  return 0;",
                "#endif",
                "}",
            ]
        )
        + "\n"
    )


def check(fixture: dict[str, Any], build: Path, arch: str | None) -> dict[str, Any]:
    """Compile and execute a complete test batch on the explicitly chosen device."""
    compiler = shutil.which("nvcc" if arch else "c++")
    cache = shutil.which("sccache") or shutil.which("ccache")
    if compiler is None or cache is None:
        raise RuntimeError("Compiler and verified compiler cache are required")
    cache_version = subprocess.check_output([cache, "--version"], text=True).strip()
    cache_before = subprocess.check_output([cache, "--show-stats"], text=True)
    build.mkdir(parents=True, exist_ok=True)
    components = [tuple(row) for row in fixture["components"]]
    source = build / ("primitives.cu" if arch else "primitives.cpp")
    source.write_text(adapter_source(components))
    obj, lib = build / "primitives.o", build / "primitives.so"
    flags = ["-arch=" + arch, "-Xcompiler=-fPIC"] if arch else ["-fPIC"]
    start = time.perf_counter()
    subprocess.run(
        [
            cache,
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
        check=True,
    )
    subprocess.run([compiler, "-shared", str(obj), "-o", str(lib)], check=True)
    compilation_seconds = time.perf_counter() - start
    cache_after = subprocess.check_output([cache, "--show-stats"], text=True)
    function = ctypes.CDLL(str(lib)).f12_df_rows
    array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS")
    function.argtypes = (ctypes.c_int, array, array)
    function.restype = ctypes.c_int

    def execute(inputs: np.ndarray) -> np.ndarray:
        output = np.zeros((len(inputs), 11), dtype=np.float64)
        status = function(len(inputs), np.ascontiguousarray(inputs), output)
        if status:
            raise RuntimeError(f"Native primitive execution failed with code {status}")
        np.testing.assert_equal(output[:, 0], 1)
        return output[:, 1:]

    entries = fixture["rows"]
    inputs = np.asarray([row["input"] for row in entries])
    expected = np.asarray([row["value_and_gradient"] for row in entries])
    start = time.perf_counter()
    observed = execute(inputs)
    cold_seconds = time.perf_counter() - start
    start = time.perf_counter()
    repeated = execute(inputs)
    warm_seconds = time.perf_counter() - start
    np.testing.assert_array_equal(repeated, observed)
    scale = np.maximum(1.0, np.max(np.abs(expected), axis=1))
    residual = (observed - expected) / scale[:, None]
    np.testing.assert_allclose(residual, 0.0, atol=3e-10, rtol=0)
    step_errors = {}
    for step in (2e-4, 1e-4):
        shifted = []
        for row in inputs:
            for coordinate in range(3 * len(components[int(row[0])])):
                for sign in (-1, 1):
                    point = row.copy()
                    point[6 + coordinate] += sign * step
                    shifted.append(point)
        energies = iter(execute(np.asarray(shifted))[:, 0])
        slopes = np.zeros((len(inputs), 9))
        for i, row in enumerate(inputs):
            for coordinate in range(3 * len(components[int(row[0])])):
                minus, plus = next(energies), next(energies)
                slopes[i, coordinate] = (plus - minus) / (2 * step)
        error = (slopes - observed[:, 1:]) / scale[:, None]
        np.testing.assert_allclose(error, 0.0, atol=4e-7, rtol=0)
        step_errors[str(step)] = float(np.max(np.abs(error)))
    translation = np.sum(observed[:, 1:].reshape(-1, 3, 3), axis=1)
    np.testing.assert_allclose(translation / scale[:, None], 0.0, atol=2e-12, rtol=0)
    return {
        "backend": "cuda-fp64" if arch else "host-fp64",
        "cache": cache_version,
        "cache_before": cache_before,
        "cache_after": cache_after,
        "compiled_source_bytes": source.stat().st_size,
        "primitive_cases": len(inputs),
        "components": components,
        "max_value_and_derivative_scaled_error": float(np.max(np.abs(residual))),
        "max_scaled_translation_error": float(
            np.max(np.abs(translation) / scale[:, None])
        ),
        "two_step_fd_scaled_errors": step_errors,
        "compile_seconds": compilation_seconds,
        "cold_batch_seconds": cold_seconds,
        "warm_batch_seconds": warm_seconds,
        "rows": [
            {
                "input": row["input"],
                "native_value_and_gradient": value.tolist(),
                "reference_value_and_gradient": row["value_and_gradient"],
            }
            for row, value in zip(entries, observed, strict=True)
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
            "Allocated GPU validation requires an explicit CUDA architecture"
        )
    result = check(json.loads(args.fixture.read_text()), args.cache, args.arch)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
