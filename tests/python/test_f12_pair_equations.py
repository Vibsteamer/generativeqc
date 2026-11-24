"""Pair-local denominator regression against an independently replayed Psi4 routine."""

import numpy as np
import pytest
from generativeqc_compiler.mp2.f12_pairs import restricted_f12_pairs
from generativeqc_compiler.tensor import execute

from benchmarks.f12_pair_component import CASES, FIXTURE, evaluate_case, verify


@pytest.mark.parametrize("case", CASES)
def test_pair_energies_and_reverse(case: str) -> None:
    with np.load(FIXTURE, allow_pickle=False) as data:
        verify(evaluate_case(data, case, execute))


@pytest.mark.parametrize(
    "pairs,virtuals,diagonal", [(0, 3, True), (2, 0, False), (2, 3, 1)]
)
def test_pair_domain(pairs: int, virtuals: int, diagonal: bool) -> None:
    with pytest.raises(ValueError):
        restricted_f12_pairs(pairs, virtuals, diagonal=diagonal)
