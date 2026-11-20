"""Run focal-point tensor graphs on device zero using the compiler's artifact cache."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpy as np
    from generativeqc_compiler.tensor import Execution, Program


def focal_cuda_runner(
    arch: str,
    cache: Path,
    *,
    max_bytes: int,
    record: Callable[[dict], None] | None = None,
) -> Callable:
    """Create an FP64 executor for the GPU assigned to this process."""
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info
    from generativeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from generativeqc_compiler.tensor.cuda_plan import plan_cuda

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("CUDA focal graph compilation requires nvcc")
    compiler = CudaCompilerAdapter(Path(nvcc), cuda_target_info(arch))
    compiled = {}

    def run(program: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        key = program.logical_hash
        if key not in compiled:
            plan = plan_cuda(program, compiler.target, max_bytes=max_bytes)
            compiled[key] = (plan, compile_cuda(plan, compiler, cache))
        plan, artifact = compiled[key]
        with PreparedCuda(plan, artifact, device=0) as prepared:
            result = prepared.execute(inputs)
        if result.backend != "cuda-fp64-ordinary-stream":
            raise RuntimeError(f"unqualified F12 backend {result.backend}")
        if record is not None:
            record(
                {"program": key, "backend": result.backend, "metrics": result.metrics}
            )
        return result

    return run
