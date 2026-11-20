"""Compile and run F12 qualification graphs in FP64 on the allocated CUDA device."""

from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Callable
from pathlib import Path
from typing import Any

from generativeqc._f12_focal_runner import focal_cuda_runner


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
    metrics = []
    return focal_cuda_runner(
        arch, cache, max_bytes=max_bytes, record=metrics.append
    ), metrics
