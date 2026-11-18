"""Enumerate the occupied pairs of a canonical unrestricted 3C(FIX) reference."""

from collections.abc import Callable, Iterator
from itertools import combinations, product
from typing import NamedTuple

import numpy as np
from generativeqc_compiler.mp2.f12_opposite_spin import opposite_spin_projected_pair
from generativeqc_compiler.mp2.f12_unrestricted import projected_same_spin_f12_pair
from generativeqc_compiler.tensor import Program


class SpinPairBlock(NamedTuple):
    left: str
    right: str
    left_occupied: int
    right_occupied: int
    graph: Program
    pairs: Iterator[tuple[int, int]]


def occupied_pair_blocks(
    space: tuple[int, int, int, int, int],
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
) -> Iterator[SpinPairBlock]:
    """Yield one graph and its occupied pairs per nonempty spin channel.

    Space gives the alpha/beta occupied, OBS, extended, and fit dimensions.
    Optional frozen counts omit lowest occupied pair indices independently per
    spin. Every occupied orbital still participates in the UHF Fock and in the
    strong-orthogonality projector. Nuclear gradients must use the same frozen
    orbital convention and differentiate the complete parent reference.
    """
    oa, ob, obs, ext, aux = space
    fa, fb = frozen_occupied
    if not (0 <= fa <= oa < obs and 0 <= fb <= ob < obs < ext and aux > 0):
        raise ValueError(
            "Invalid occupied/frozen spin spaces or orbital/fitting dimensions"
        )
    for sl, sr, nl, nr, il, ir in (
        ("alpha", "alpha", oa, oa, fa, fa),
        ("alpha", "beta", oa, ob, fa, fb),
        ("beta", "beta", ob, ob, fb, fb),
    ):
        if sl == sr:
            if nl - il < 2:
                continue
            pairs = combinations(range(il, nl), 2)
            graph = projected_same_spin_f12_pair(
                nl, obs - nl, ext - obs, auxiliaries=aux
            )
        else:
            if nl == il or nr == ir:
                continue
            pairs = product(range(il, nl), range(ir, nr))
            graph = opposite_spin_projected_pair(nl, nr, obs, ext, auxiliaries=aux)
        yield SpinPairBlock(sl, sr, nl, nr, graph, pairs)


def pair_operands(
    data: dict,
    src: dict,
    fock: dict,
    sl: str,
    sr: str,
    nl: int,
    nr: int,
    n: int,
    i: int,
    j: int,
) -> dict[str, np.ndarray]:
    """Select the orbital and fit tensors for one occupied spin pair."""
    dl, dr = src["fit_" + sl][:, i], src["fit_" + sr][:, j]
    raw_c = src["raw_coulomb_" + sr][:, j]
    select_i = np.zeros(nl if sl == sr else n)
    select_j = np.zeros(nr if sl == sr else n)
    select_i[i] = select_j[j] = 1.0
    if sl == sr:
        vals = {
            "select_i": select_i,
            "select_j": select_j,
            "fock": fock["fock_" + sl],
            "exchange": fock["exchange_" + sl],
            "fit_i": dl,
            "fit_j": dr,
            "raw_coulomb_j": raw_c,
        }
    else:
        vals = {
            "select_i": select_i,
            "select_j": select_j,
            "cross_overlap": data["cross_overlap"],
            "fit_i_alpha": dl,
            "fit_j_beta": dr,
            "raw_coulomb_j_beta": raw_c,
            **fock,
        }
    for op, rel in (
        ("geminal", "f12"),
        ("squared_geminal", "f12_squared"),
        ("geminal_coulomb", "f12_over_coulomb"),
        ("double_commutator", "double_commutator"),
    ):
        la, rb = (
            data["raw/" + op + "/" + sl][:, i],
            data["raw/" + op + "/" + sr][:, j],
        )
        if sl == sr:
            vals.update(
                {
                    "raw_" + rel + "_i": la,
                    "raw_" + rel + "_j": rb,
                    "metric_" + rel: data["metric/" + op],
                }
            )
        else:
            vals.update(
                {
                    "raw_" + op + "_i_alpha": la,
                    "raw_" + op + "_j_beta": rb,
                    "metric_" + op: data["metric/" + op],
                }
            )
    return vals


class PairEnergy(NamedTuple):
    spins: tuple[str, str]
    occupied_indices: tuple[int, int]
    mp2: float
    f12: float


def molecular_pair_energies(
    space: tuple[int, int, int, int, int],
    data: dict,
    sources: dict,
    fock: dict,
    run: Callable,
    *,
    frozen_occupied: tuple[int, int] = (0, 0),
) -> Iterator[PairEnergy]:
    """Evaluate the orbital MP2 contribution and fixed-cusp increment per pair.

    The common auxiliary space is Coulomb fitted. Each spin retains its own
    extended Fock matrix and occupied/virtual MOs. The caller supplies the
    physical OBS/CABS integrals and the selected TensorIR execution backend.
    """
    for left, right, nl, nr, graph, pairs in occupied_pair_blocks(
        space, frozen_occupied=frozen_occupied
    ):
        for i, j in pairs:
            values = pair_operands(
                data, sources, fock, left, right, nl, nr, space[3], i, j
            )
            result = run(
                graph, {key: np.ascontiguousarray(v) for key, v in values.items()}
            ).outputs
            ordinary = float(result["ordinary_pair_energy"])
            increment = float(result["pair_energy"])
            if not np.isfinite(ordinary) or not np.isfinite(increment):
                raise ValueError(f"Nonfinite energy for {left}/{right} pair ({i}, {j})")
            yield PairEnergy((left, right), (i, j), ordinary, increment)
