"""Validate ownership of array storage passed to the F12 native bindings."""

import numpy as np


def check_disjoint_outputs(
    outputs: dict[str, np.ndarray | None], inputs: dict[str, np.ndarray | None]
) -> None:
    """Reject overwrites of another operand through a writable ctypes buffer."""
    buffers = dict(inputs)
    for name, output in outputs.items():
        if output is None:
            continue
        for other_name, other in buffers.items():
            if other is not None and np.shares_memory(output, other):
                raise ValueError(f"{name} cannot overlap {other_name}")
        buffers[name] = output
