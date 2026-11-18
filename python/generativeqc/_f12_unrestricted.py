"""Unrestricted DF-Fock and CABS singles composition in the alpha/beta frames."""

from collections.abc import Callable, Mapping
from typing import NamedTuple

import numpy as np
from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_extended_df_fock
from generativeqc_compiler.tensor import transpose_program

from ._f12_cabs import cabs_singles
from ._f12_spin_pairs import occupied_pair_blocks, pair_operands


class UnrestrictedCabsSingles(NamedTuple):
    energy: float
    spin_energies: tuple[float, float]
    source_cotangents: dict[str, np.ndarray] | None


def unrestricted_cabs_energy(
    dimensions: tuple[int, int, int, int],
    sources: Mapping[str, np.ndarray],
    run: Callable,
    *,
    pullback: bool = True,
    singles: Callable = cabs_singles,
) -> UnrestrictedCabsSingles:
    """Compute both singles contributions and their joint fixed-frame DF source VJP.

    Dimensions = alpha occupied, beta occupied, extended orbital count, fit count.
    The alpha/beta orbital frames can be distinct. The occupied plus extended-
    virtual Sylvester problem is separate for each spin. A shared Coulomb source
    couples their responses to both occupied densities through the Fock graph.
    Nuclear differentiation also needs frame, fitting and SCF response.
    """
    graph = unrestricted_extended_df_fock(*dimensions)
    forward = run(graph, sources).outputs
    seeds = {}
    energies = []
    for spin, occupied in zip(("alpha", "beta"), dimensions[:2], strict=True):
        key = "fock_" + spin
        fock = forward[key]
        # Physical F is symmetric. Define the derivative on the symmetric
        # subspace even if raw and metric-solved factor inputs vary separately.
        energy, bar = singles(
            (fock + fock.T) / 2, occupied, spin_weight=1, pullback=pullback
        )
        energies.append(float(energy))
        if pullback:
            if bar is None:
                raise ValueError("missing singles Fock cotangent")
            seeds["bar_" + key] = (bar + bar.T) / 2
    weights = None
    if pullback:
        reverse = transpose_program(graph, ("fock_alpha", "fock_beta")).program
        weights = {
            k.removeprefix("bar_"): v
            for k, v in run(reverse, {**sources, **seeds}).outputs.items()
        }
    return UnrestrictedCabsSingles(sum(energies), (energies[0], energies[1]), weights)


def unrestricted_mp2_f12_source_pullback(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    singles: Callable = cabs_singles,
) -> dict[str, np.ndarray]:
    """Pull the molecular UMP2, geminal and CABS energies back to their shared sources.

    One reverse per occupied pair provides weights reused for all nuclei. These
    are derivatives in the two moving OBS/CABS frames. Contract them with total
    physical source tangents including integral, metric, UHF and orbital motion,
    and add the reference gradient. The caller owns that geometry response.
    """
    oa, ob, _, ext, aux = space
    sources = {
        key.removeprefix("source/"): v
        for key, v in data.items()
        if key.startswith("source/")
    }
    fock_graph = unrestricted_extended_df_fock(oa, ob, ext, aux)
    fock = run(fock_graph, sources).outputs
    source_bars = {k: np.zeros_like(v) for k, v in sources.items()}
    raw_bars = {
        k: np.zeros_like(v)
        for k, v in data.items()
        if k == "cross_overlap" or k.startswith(("raw/", "metric/"))
    }
    fock_bars = {k: np.zeros_like(v) for k, v in fock.items()}
    for spin, occupied in (("alpha", oa), ("beta", ob)):
        key = "fock_" + spin
        matrix = fock[key]
        _, bar = singles(
            (matrix + matrix.T) / 2, occupied, spin_weight=1, pullback=True
        )
        if bar is None:
            raise ValueError("Missing UHF CABS singles Fock cotangent")
        fock_bars[key] += (bar + bar.T) / 2
    for sl, sr, nl, nr, graph, pairs in occupied_pair_blocks(
        space, frozen_occupied=frozen_occupied
    ):
        rev = transpose_program(graph, ("pair_energy", "ordinary_pair_energy")).program
        for i, j in pairs:
            primal = pair_operands(data, sources, fock, sl, sr, nl, nr, ext, i, j)
            feeds = {k: np.ascontiguousarray(v) for k, v in primal.items()}
            feeds["bar_pair_energy"] = np.array(1.0)
            feeds["bar_ordinary_pair_energy"] = np.array(1.0)
            result = run(rev, feeds).outputs
            # Pair operands are views into the molecular arrays. Shared metric,
            # Fock, and repeated occupied-leg entries accumulate independently.
            buffers = pair_operands(
                raw_bars, source_bars, fock_bars, sl, sr, nl, nr, ext, i, j
            )
            for key, value in result.items():
                buffers[key.removeprefix("bar_")] += value
    rev = transpose_program(fock_graph, tuple(fock)).program
    fock_weights = run(
        rev, {**sources, **{"bar_" + k: v for k, v in fock_bars.items()}}
    ).outputs
    for key, value in fock_weights.items():
        source_bars[key.removeprefix("bar_")] += value
    raw_bars.update({"source/" + k: v for k, v in source_bars.items()})
    return raw_bars
