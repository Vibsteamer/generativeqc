"""Run focal-point tensor graphs on device zero using the compiler's artifact cache."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpy as np
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.tensor import Execution, Program


def focal_cuda_compiler(arch: str) -> CudaCompilerAdapter:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("CUDA focal graph compilation requires nvcc")
    return CudaCompilerAdapter(Path(nvcc), cuda_target_info(arch))


def focal_cuda_runner(
    arch: str,
    cache: Path,
    *,
    max_bytes: int,
    record: Callable[[dict], None] | None = None,
) -> Callable:
    """Create an FP64 executor for the GPU assigned to this process."""
    from contextlib import contextmanager

    from generativeqc_compiler.common.cuda_target import cuda_architecture
    from generativeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from generativeqc_compiler.tensor.cuda_plan import plan_cuda

    cuda_architecture(arch)
    compiler: CudaCompilerAdapter | None = None
    compiled = {}
    stats = {
        "jit_cold_s": 0.0,
        "jit_compile_count": 0,
        "warm_reuse_s": 0.0,
        "warm_execute_count": 0,
    }

    def _compile(program: Program):
        nonlocal compiler
        key = program.logical_hash
        if key not in compiled:
            t0 = time.monotonic()
            if compiler is None:
                compiler = focal_cuda_compiler(arch)
            plan = plan_cuda(program, compiler.target, max_bytes=max_bytes)
            artifact = compile_cuda(plan, compiler, cache)
            stats["jit_cold_s"] += time.monotonic() - t0
            stats["jit_compile_count"] += 1
            compiled[key] = (plan, artifact)
        return key, compiled[key]

    @contextmanager
    def batch(program: Program):
        """Hold one PreparedCuda instance open across a batch of identical-graph executions."""
        key, (plan, artifact) = _compile(program)
        with PreparedCuda(plan, artifact, device=0) as prepared:
            def execute_one(inputs: Mapping[str, np.ndarray]) -> Execution:
                t0 = time.monotonic()
                result = prepared.execute(inputs)
                stats["warm_reuse_s"] += time.monotonic() - t0
                stats["warm_execute_count"] += 1
                if result.backend != "cuda-fp64-ordinary-stream":
                    raise RuntimeError(f"unqualified F12 backend {result.backend}")
                if record is not None:
                    record(
                        {"program": key, "backend": result.backend, "metrics": result.metrics}
                    )
                return result

            yield execute_one

    def run(program: Program, inputs: Mapping[str, np.ndarray]) -> Execution:
        with batch(program) as execute_one:
            return execute_one(inputs)

    run.batch = batch  # type: ignore[attr-defined]
    run.profile_snapshot = lambda: dict(stats)  # type: ignore[attr-defined]
    return run
