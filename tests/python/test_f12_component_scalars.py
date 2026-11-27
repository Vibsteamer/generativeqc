"""Procedural CUDA inputs must retain zero-dimensional FP64 array storage."""

import numpy as np
from generativeqc_compiler.tensor import Program, TensorSpec, execute, input_tensor

from benchmarks.f12_unrestricted_component import check


def test_scalar_forward_reverse_and_displacements() -> None:
    x = input_tensor("x", TensorSpec((), role="input", differentiable=True))
    graph = Program({"v": x})

    def strict_run(program: Program, inputs: dict) -> object:
        for value in inputs.values():
            assert (
                isinstance(value, np.ndarray)
                and value.shape == ()
                and value.dtype == np.float64
            )
        return execute(program, inputs)

    result = check(
        graph, {"x": np.array(0.42), "expected_v": np.array(0.42)}, strict_run, 1e-13
    )
    assert result["max_abs_errors"]["v"] == 0.0
