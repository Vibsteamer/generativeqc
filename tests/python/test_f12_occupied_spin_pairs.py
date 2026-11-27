"""Occupied determinant counting, including empty channels and frozen orbitals."""

from itertools import combinations

import pytest
from generativeqc._f12_spin_pairs import occupied_pair_blocks


@pytest.mark.parametrize(
    ("occupied", "frozen"),
    [
        ((1, 0), (0, 0)),
        ((0, 1), (0, 0)),
        ((1, 1), (0, 0)),
        ((2, 0), (0, 0)),
        ((0, 3), (0, 0)),
        ((5, 4), (0, 0)),
        ((5, 4), (1, 1)),
        ((4, 2), (2, 2)),
        ((1, 2), (1, 2)),
    ],
)
def test_unordered_active_spin_determinants(
    occupied: tuple[int, int], frozen: tuple[int, int]
) -> None:
    space = (*occupied, max(occupied) + 2, max(occupied) + 4, 3)
    blocks = occupied_pair_blocks(space, frozen_occupied=frozen)
    active = [
        (spin, i)
        for spin, count, first in zip(("alpha", "beta"), occupied, frozen, strict=True)
        for i in range(first, count)
    ]
    expected = set(combinations(active, 2))
    pairs = [
        ((block.left, i), (block.right, j)) for block in blocks for i, j in block.pairs
    ]
    assert len(pairs) == len(expected)
    assert set(pairs) == expected


def test_invalid_frozen_count() -> None:
    with pytest.raises(ValueError):
        list(occupied_pair_blocks((2, 1, 4, 5, 3), frozen_occupied=(1, 2)))
