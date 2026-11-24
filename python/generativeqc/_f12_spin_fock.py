"""Auxiliary-block execution of the extended UHF Fock source and its transpose."""

from collections.abc import Callable, Iterator, Mapping

import numpy as np
from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_extended_df_fock
from generativeqc_compiler.tensor import transpose_program
from generativeqc_compiler.tensor.program import Program

from ._f12_spin_pairs import batch_execute

_TARGET_AUX_SLAB_BYTES = 256 << 20


def default_auxiliary_batch_size(
    auxiliaries: int, extent_a: int, extent_b: int | None = None
) -> int:
    """Choose an auxiliary chunk size keeping per-slab 3-center tensors <= 256 MiB."""
    if auxiliaries <= 1:
        return max(1, auxiliaries)
    eb = extent_a if extent_b is None else extent_b
    slice_bytes = max(1, extent_a * eb * 8)
    return max(1, min(auxiliaries, _TARGET_AUX_SLAB_BYTES // slice_bytes))


def _tile_groups(
    auxiliaries: int, width: int
) -> list[tuple[int, list[slice]]]:
    full = [slice(s, s + width) for s in range(0, (auxiliaries // width) * width, width)]
    groups: list[tuple[int, list[slice]]] = []
    if full:
        groups.append((width, full))
    rem_start = (auxiliaries // width) * width
    if rem_start < auxiliaries:
        groups.append((auxiliaries - rem_start, [slice(rem_start, auxiliaries)]))
    return groups


def _validate_tiles(
    space: tuple[int, int, int, int],
    sources: Mapping[str, np.ndarray],
    batch_size: int | None,
) -> int:
    _, _, orbitals, auxiliaries = space
    width = (
        default_auxiliary_batch_size(auxiliaries, orbitals)
        if batch_size is None
        else batch_size
    )
    if type(width) is not int or width < 1 or auxiliaries < 1:
        raise ValueError("The auxiliary count and batch size must be positive")
    for name, value in sources.items():
        if not name.startswith("hcore_") and value.shape[0] != auxiliaries:
            raise ValueError(f"Inconsistent shared auxiliary extent for {name}")
    return width


def _tile_values(sources: Mapping[str, np.ndarray], rows: slice) -> dict[str, np.ndarray]:
    values = {}
    for key, data in sources.items():
        if key.startswith("hcore_"):
            values[key] = data if rows.start == 0 else np.zeros_like(data)
        else:
            values[key] = np.ascontiguousarray(data[rows])
    return values


def _tiles(
    space: tuple[int, int, int, int],
    sources: Mapping[str, np.ndarray],
    batch_size: int | None,
) -> Iterator[tuple[Program, slice, dict[str, np.ndarray]]]:
    oa, ob, orbitals, auxiliaries = space
    width = _validate_tiles(space, sources, batch_size)
    for start in range(0, auxiliaries, width):
        rows = slice(start, min(start + width, auxiliaries))
        graph = unrestricted_extended_df_fock(oa, ob, orbitals, rows.stop - start)
        yield graph, rows, _tile_values(sources, rows)


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
    oa, ob, orbitals, auxiliaries = space
    width = _validate_tiles(space, sources, batch_size)
    result: dict[str, np.ndarray] = {}
    for tile_len, slices in _tile_groups(auxiliaries, width):
        graph = unrestricted_extended_df_fock(oa, ob, orbitals, tile_len)
        with batch_execute(run, graph) as exec_one:
            for rows in slices:
                part = exec_one(_tile_values(sources, rows)).outputs
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
    oa, ob, orbitals, auxiliaries = space
    width = _validate_tiles(space, sources, batch_size)
    share_beta_raw = (
        accumulate is None
        and oa == ob
        and "raw_coulomb_alpha" in sources
        and "raw_coulomb_beta" in sources
        and (
            sources["raw_coulomb_alpha"] is sources["raw_coulomb_beta"]
            or np.array_equal(sources["raw_coulomb_alpha"], sources["raw_coulomb_beta"])
        )
        and (
            "fit_alpha" not in sources
            or sources["fit_alpha"] is sources["fit_beta"]
            or np.array_equal(sources["fit_alpha"], sources["fit_beta"])
        )
        and (
            "hcore_alpha" not in sources
            or sources["hcore_alpha"] is sources["hcore_beta"]
            or np.array_equal(sources["hcore_alpha"], sources["hcore_beta"])
        )
        and all(
            (k.replace("_alpha", "_beta") in seeds and np.array_equal(seeds[k], seeds[k.replace("_alpha", "_beta")]))
            for k in seeds
            if k.endswith("_alpha")
        )
        and sum(1 for k in seeds if k.endswith("_alpha")) == sum(1 for k in seeds if k.endswith("_beta"))
    )
    if accumulate is None:
        bars = {
            key: np.zeros_like(value)
            for key, value in sources.items()
            if not (share_beta_raw and key == "raw_coulomb_beta")
        }
    else:
        bars = accumulate
    seed_inputs = {"bar_" + k: np.ascontiguousarray(v) for k, v in seeds.items()}
    for tile_len, slices in _tile_groups(auxiliaries, width):
        graph = unrestricted_extended_df_fock(oa, ob, orbitals, tile_len)
        reverse = transpose_program(graph, tuple(seeds)).program
        with batch_execute(run, reverse) as exec_one:
            for rows in slices:
                part = exec_one({**_tile_values(sources, rows), **seed_inputs})
                for output, array in part.outputs.items():
                    key = output.removeprefix("bar_")
                    if share_beta_raw and key == "raw_coulomb_beta":
                        continue
                    if key.startswith("hcore_"):
                        if rows.start == 0:
                            bars[key] += array
                    else:
                        bars[key][rows] += array
    if share_beta_raw:
        bars["raw_coulomb_beta"] = bars["raw_coulomb_alpha"]
    return bars
