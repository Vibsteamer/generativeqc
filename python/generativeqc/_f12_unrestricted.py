"""Unrestricted DF-Fock and CABS singles composition in the alpha/beta frames."""

from collections.abc import Callable, Mapping
from typing import NamedTuple

import numpy as np
from generativeqc_compiler.mp2.f12_unrestricted import unrestricted_extended_df_fock
from generativeqc_compiler.tensor import transpose_program

from ._f12_cabs import cabs_singles


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
