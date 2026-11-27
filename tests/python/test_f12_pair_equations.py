"""Pair-local denominator regression against an independently replayed Psi4 routine."""

import numpy as np
import pytest
from generativeqc_compiler.mp2.f12_pairs import restricted_f12_pairs
from generativeqc_compiler.tensor import execute

from benchmarks.f12_pair_component import CASES, FIXTURE, evaluate_case


@pytest.mark.parametrize("case", CASES)
def test_pair_energies_and_reverse(case: str) -> None:
    with np.load(FIXTURE, allow_pickle=False) as data:
        occupied, _, _, frozen = map(int, data[case + "/space"])
        row = evaluate_case(data, case, execute)

    active = occupied - frozen
    expected_tiles = {"diagonal": active}
    if active > 1:
        expected_tiles["offdiag"] = active * (active - 1) // 2
    assert {tile["tile"]: tile["pairs"] for tile in row["timings"]} == expected_tiles

    # Psi4 prints its pair table to 12 decimals. Its total energy and the
    # independently displaced intermediate energies keep full precision.
    assert row["energy_Eh"] == pytest.approx(row["psi4_Eh"], rel=0, abs=2e-12)
    assert row["max_pair_error_Eh"] <= 8e-13
    assert len(row["psi4_derivatives"]) == 2
    for slope in row["psi4_derivatives"]:
        assert row["intermediate_derivative"] == pytest.approx(slope, rel=0, abs=2e-8)


@pytest.mark.parametrize(
    "pairs,virtuals,diagonal", [(0, 3, True), (2, 0, False), (2, 3, 1)]
)
def test_pair_domain(pairs: int, virtuals: int, diagonal: bool) -> None:
    with pytest.raises(ValueError):
        restricted_f12_pairs(pairs, virtuals, diagonal=diagonal)
