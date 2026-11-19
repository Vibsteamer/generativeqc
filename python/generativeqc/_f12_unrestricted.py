"""Unrestricted DF-Fock and CABS singles composition in the alpha/beta frames."""

import math
from collections.abc import Callable, Mapping
from typing import NamedTuple

import numpy as np
from generativeqc_compiler.tensor import Program, transpose_program

from ._f12_cabs import cabs_singles
from ._f12_spin_ao import _orbital_sources
from ._f12_spin_fock import spin_fock, spin_fock_pullback
from ._f12_spin_pairs import occupied_pair_blocks, pair_operands


class UnrestrictedCabsSingles(NamedTuple):
    energy: float
    spin_energies: tuple[float, float]
    source_cotangents: dict[str, np.ndarray] | None


class UnrestrictedF12Energy(NamedTuple):
    ordinary_mp2: float
    f12_increment: float
    cabs_singles: tuple[float, float]
    active_pairs: int


def unrestricted_cabs_energy(
    dimensions: tuple[int, int, int, int],
    sources: Mapping[str, np.ndarray],
    run: Callable,
    *,
    pullback: bool = True,
    singles: Callable = cabs_singles,
    auxiliary_batch_size: int | None = None,
) -> UnrestrictedCabsSingles:
    """Compute both singles contributions and their joint fixed-frame DF source VJP.

    Dimensions = alpha occupied, beta occupied, extended orbital count, fit count.
    The alpha/beta orbital frames can be distinct. The occupied plus extended-
    virtual Sylvester problem is separate for each spin. A shared Coulomb source
    couples their responses to both occupied densities through the Fock graph.
    Nuclear differentiation also needs frame, fitting and SCF response.
    """
    forward = spin_fock(dimensions, sources, run, auxiliary_batch_size)
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
            seeds[key] = (bar + bar.T) / 2
    weights = None
    if pullback:
        weights = spin_fock_pullback(
            dimensions, sources, seeds, run, auxiliary_batch_size
        )
    return UnrestrictedCabsSingles(sum(energies), (energies[0], energies[1]), weights)


def unrestricted_mp2_f12_source_pullback(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    singles: Callable = cabs_singles,
    auxiliary_batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Pull the molecular UMP2, geminal and CABS energies back to their shared sources.

    One reverse per occupied pair provides weights reused for all nuclei. These
    are derivatives in the two moving OBS/CABS frames. Contract them with total
    physical source tangents including integral, metric, UHF and orbital motion,
    and add the reference gradient. The caller owns that geometry response.
    """
    weights, _ = _projected_response(
        space, data, run, frozen_occupied, singles, auxiliary_batch_size, energies=False
    )
    return weights


def unrestricted_mp2_f12_ao_response(
    space: tuple[int, int, int, int, int],
    ao: dict[str, np.ndarray],
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
    singles: Callable = cabs_singles,
    auxiliary_batch_size: int | None = None,
) -> tuple[UnrestrictedF12Energy, dict[str, np.ndarray]]:
    """Evaluate the correlation and singles energies and their common AO cotangents.

    Supply the two spin frames, raw operators, metrics and Coulomb whitener at
    one geometry. Each pair returns both its primal energies and its reverse
    contractions. The AO result includes both spin-coefficient cotangents and
    the physical Coulomb metric derivative. Nuclear and UHF response are
    applied by the force caller.
    """
    data, _ = _orbital_sources(space, ao, run, None, projected=None)
    weights, terms = _projected_response(
        space, data, run, frozen_occupied, singles, auxiliary_batch_size, energies=True
    )
    _, bars = _orbital_sources(space, ao, run, weights, projected=data)
    assert terms is not None and bars is not None
    return terms, bars


def _projected_response(
    space: tuple[int, int, int, int, int],
    data: dict[str, np.ndarray],
    run: Callable,
    frozen_occupied: tuple[int, int],
    singles: Callable,
    auxiliary_batch_size: int | None,
    *,
    energies: bool,
) -> tuple[dict[str, np.ndarray], UnrestrictedF12Energy | None]:
    oa, ob, _, ext, aux = space
    sources = {
        key.removeprefix("source/"): v
        for key, v in data.items()
        if key.startswith("source/")
    }
    fock_space = (oa, ob, ext, aux)
    fock = spin_fock(fock_space, sources, run, auxiliary_batch_size)
    source_bars = {k: np.zeros_like(v) for k, v in sources.items()}
    raw_bars = {
        k: np.zeros_like(v)
        for k, v in data.items()
        if k == "cross_overlap" or k.startswith(("raw/", "metric/"))
    }
    fock_bars = {k: np.zeros_like(v) for k, v in fock.items()}
    singles_energies = []
    mp2_terms, f12_terms = [], []
    for spin, occupied in (("alpha", oa), ("beta", ob)):
        key = "fock_" + spin
        matrix = fock[key]
        energy, bar = singles(
            (matrix + matrix.T) / 2, occupied, spin_weight=1, pullback=True
        )
        if bar is None:
            raise ValueError("Missing UHF CABS singles Fock cotangent")
        if energies:
            singles_energies.append(float(energy))
        fock_bars[key] += (bar + bar.T) / 2
    for sl, sr, nl, nr, graph, pairs in occupied_pair_blocks(
        space, frozen_occupied=frozen_occupied
    ):
        rev = transpose_program(graph, ("pair_energy", "ordinary_pair_energy")).program
        kernel = rev
        if energies:
            kernel = Program(
                {
                    **rev.outputs,
                    "ordinary_pair_energy": graph.outputs["ordinary_pair_energy"],
                    "pair_energy": graph.outputs["pair_energy"],
                }
            )
        for i, j in pairs:
            primal = pair_operands(data, sources, fock, sl, sr, nl, nr, ext, i, j)
            feeds = {k: np.ascontiguousarray(v) for k, v in primal.items()}
            feeds["bar_pair_energy"] = np.array(1.0)
            feeds["bar_ordinary_pair_energy"] = np.array(1.0)
            result = run(kernel, feeds).outputs
            if energies:
                mp2_terms.append(float(result["ordinary_pair_energy"]))
                f12_terms.append(float(result["pair_energy"]))
            # Pair operands are views into the molecular arrays. Shared metric,
            # Fock, and repeated occupied-leg entries accumulate independently.
            buffers = pair_operands(
                raw_bars, source_bars, fock_bars, sl, sr, nl, nr, ext, i, j
            )
            for key in rev.outputs:
                buffers[key.removeprefix("bar_")] += result[key]
    spin_fock_pullback(
        fock_space,
        sources,
        fock_bars,
        run,
        auxiliary_batch_size,
        accumulate=source_bars,
    )
    raw_bars.update({"source/" + k: v for k, v in source_bars.items()})
    terms = None
    if energies:
        mp2 = math.fsum(mp2_terms)
        f12 = math.fsum(f12_terms)
        alpha, beta = singles_energies
        if not all(math.isfinite(value) for value in (mp2, f12, alpha, beta)):
            raise ValueError("Nonfinite unrestricted MP2-F12 or CABS energy")
        terms = UnrestrictedF12Energy(mp2, f12, (alpha, beta), len(mp2_terms))
    return raw_bars, terms
