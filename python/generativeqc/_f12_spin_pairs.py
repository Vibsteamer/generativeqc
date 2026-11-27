"""Shared scalar executor protocol for fitted post-HF response."""

from collections.abc import Callable
from contextlib import contextmanager

from generativeqc_compiler.tensor import Program


@contextmanager
def batch_execute(run: Callable, program: Program):
    """Reuse one prepared CUDA executor across repeated calls on the same program."""
    batch_fn = getattr(run, "batch", None)
    if callable(batch_fn):
        with batch_fn(program) as execute_one:
            yield execute_one
    else:
        yield lambda inputs: run(program, inputs)
