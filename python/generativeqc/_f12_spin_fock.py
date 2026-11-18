"""Auxiliary-block execution of the extended UHF Fock source and its transpose."""

from collections.abc import Callable, Iterator, Mapping

import numpy as np
from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_extended_df_fock
from generativeqc_compiler.tensor import transpose_program
from generativeqc_compiler.tensor.program import Program


def _tiles(
    space: tuple[int, int, int, int],
    sources: Mapping[str, np.ndarray],
    batch_size: int | None,
) -> Iterator[tuple[Program, slice, dict[str, np.ndarray]]]:
    oa, ob, orbitals, auxiliaries = space
    width = auxiliaries if batch_size is None else batch_size
    if type(width) is not int or width < 1 or auxiliaries < 1:
        raise ValueError("The auxiliary count and batch size must be positive")
    for name, value in sources.items():
        if not name.startswith("hcore_") and value.shape[0] != auxiliaries:
            raise ValueError(f"Inconsistent shared auxiliary extent for {name}")
    for start in range(0, auxiliaries, width):
        rows = slice(start, min(start + width, auxiliaries))
        graph = unrestricted_extended_df_fock(oa, ob, orbitals, rows.stop - start)
        values = {}
        for key, data in sources.items():
            # The one-electron operator occurs once in the sum of Fock tiles.
            if key.startswith("hcore_"):
                values[key] = data if start == 0 else np.zeros_like(data)
            else:
                values[key] = data[rows]
        yield graph, rows, values


def spin_fock(
    space: tuple[int, int, int, int],
    sources: Mapping[str, np.ndarray],
    run: Callable,
    batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Sum Coulomb and spin exchange over a complete, already fitted auxiliary space.

    Both spin channels contribute to the charge in every tile. The full inverse
    fitting metric is applied before batching so off-diagonal metric couplings
    are retained. Each per-tile Fock contraction executes on the selected backend.
    """
    result = {}
    for graph, _, values in _tiles(space, sources, batch_size):
        part = run(graph, values).outputs
        for name, array in part.items():
            if name in result:
                result[name] += array
            else:
                result[name] = array.copy()
    return result


def spin_fock_pullback(
    space: tuple[int, int, int, int],
    sources: Mapping[str, np.ndarray],
    seeds: Mapping[str, np.ndarray],
    run: Callable,
    batch_size: int | None = None,
    *,
    accumulate: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Accumulate the tiled Fock transpose into weights of the global molecular inputs."""
    bars = (
        {key: np.zeros_like(value) for key, value in sources.items()}
        if accumulate is None
        else accumulate
    )
    for graph, rows, values in _tiles(space, sources, batch_size):
        reverse = transpose_program(graph, tuple(seeds)).program
        part = run(reverse, {**values, **{"bar_" + k: v for k, v in seeds.items()}})
        for output, array in part.outputs.items():
            key = output.removeprefix("bar_")
            if key.startswith("hcore_"):
                if rows.start == 0:
                    bars[key] += array
            else:
                bars[key][rows] += array
    return bars
