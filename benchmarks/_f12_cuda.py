"""Compile and run F12 qualification graphs in FP64 on the allocated CUDA device."""

import shutil
from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from generativeqc_compiler.tensor.program import Program


def add_molecular_graph_budget(parser: ArgumentParser) -> None:
    """Set the per-graph numeric-buffer budget for molecular AO response tests."""

    def positive_mib(text: str) -> int:
        value = int(text)
        if value <= 0:
            raise ArgumentTypeError("budget must be positive")
        return value

    parser.add_argument(
        "--budget-mib",
        type=positive_mib,
        default=1536,
        help="Combined device and host staging budget per TensorIR graph in MiB "
        "(default: 1536, including the contracted water AO reverse graph).",
    )


def cuda_runner(
    arch: str, cache: Path, *, max_bytes: int = 512 << 20
) -> tuple[Callable, list[dict[str, Any]]]:
    """Return a source-cached graph runner and the list of actual backend measurements."""
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info
    from generativeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from generativeqc_compiler.tensor.cuda_plan import plan_cuda

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("CUDA qualification requires nvcc")
    compiler = CudaCompilerAdapter(Path(nvcc), cuda_target_info(arch))
    compiled = {}
    metrics = []

    def run(program: Program, inputs: Mapping[str, np.ndarray]) -> Any:
        key = program.logical_hash
        if key not in compiled:
            plan = plan_cuda(program, compiler.target, max_bytes=max_bytes)
            compiled[key] = (plan, compile_cuda(plan, compiler, cache))
        plan, artifact = compiled[key]
        with PreparedCuda(plan, artifact, device=0) as prepared:
            result = prepared.execute(inputs)
        if result.backend != "cuda-fp64-ordinary-stream":
            raise RuntimeError(f"unqualified F12 backend {result.backend}")
        metrics.append(
            {"program": key, "backend": result.backend, "metrics": result.metrics}
        )
        return result

    return run, metrics
