"""Bounded-memory Cartesian contraction of the molecular unrestricted F12 adjoint."""

import math
from collections.abc import Callable, Mapping

import numpy as np
from generativeqc_compiler.mp2.f12_unrestricted_force import (
    unrestricted_f12_nuclear_source_action,
)


def spin_source_force(
    weights: Mapping[str, np.ndarray],
    tangents: Mapping[str, np.ndarray],
    reference_gradient: float,
    run: Callable,
    *,
    block_elements: int,
) -> float:
    """Evaluate one force from a common ordered set of electronic weights and tangents.

    Inputs retain their original shape and may have non-contiguous strides.
    Each device dot consumes at most block_elements scalars per input. Sum the
    reference contribution once and accumulate the returned FP64 dots with fsum.
    """
    if type(block_elements) is not int or block_elements < 1:
        raise ValueError("Source block size must be a positive integer")
    size = sum(array.size for array in weights.values())
    if size == 0 or not np.isfinite(reference_gradient):
        raise ValueError("Expected nonempty sources and a finite reference gradient")
    width = min(block_elements, size)
    wbuf, dbuf = np.empty(width), np.empty(width)
    terms = []
    filled = 0

    def contract(count: int) -> None:
        value = run(
            unrestricted_f12_nuclear_source_action(count),
            {
                "weights": wbuf[:count],
                "tangents": dbuf[:count],
                "reference_gradient": np.array(
                    reference_gradient if not terms else 0.0
                ),
            },
        ).outputs["force"]
        terms.append(float(value))

    for name in sorted(weights):
        w, d = weights[name], tangents[name]
        if w.shape != d.shape or w.dtype != np.float64 or d.dtype != np.float64:
            raise ValueError(
                f"Inconsistent FP64 molecular source derivative for {name}"
            )
        offset = 0
        while offset < w.size:
            n = min(width - filled, w.size - offset)
            wbuf[filled : filled + n] = w.flat[offset : offset + n]
            dbuf[filled : filled + n] = d.flat[offset : offset + n]
            offset += n
            filled += n
            if filled == width:
                contract(filled)
                filled = 0
    if filled:
        contract(filled)
    result = math.fsum(terms)
    if not math.isfinite(result):
        raise ValueError("Nonfinite unrestricted F12 Cartesian force")
    return result
