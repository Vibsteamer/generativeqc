"""Independent exterior-product checks for explicit-triples spin coordinates."""

from fractions import Fraction
from itertools import permutations, product
from math import prod

import numpy as np
import pytest

from generativeqc_compiler.cc.explicit_triples_spin import ExplicitTriplesSpinLayout
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.cc.uccsd_triples import channel_shape


def test_typed_unrestricted_blocks_and_exact_ordered_weights():
    reference = UCCSDLayout(3, 2, 4, 3)
    layout = ExplicitTriplesSpinLayout(reference, (2, 1))
    physical = {block.channel: block for block in layout.blocks()}
    assert list(physical) == ["aaa", "aab", "abb", "bbb"]
    assert [b.ordered_prefactor for b in physical.values()] == [
        Fraction(1, 36),
        Fraction(1, 4),
        Fraction(1, 4),
        Fraction(1, 36),
    ]
    for channel, block in physical.items():
        assert block.spec.shape == channel_shape(reference, channel)
    assert physical["bbb"].independent_size == 0

    once = {(b.channel, b.particle_domains): b for b in layout.blocks(1)}
    separate = once["aab", "vcv"]
    same = once["aab", "vvc"]
    assert separate.spec.shape == (3, 3, 2, 4, 2, 3)
    assert [axis.space.name for axis in separate.spec.indices] == [
        "oa", "oa", "ob", "va", "ca", "vb"
    ]
    assert separate.independent_size == 144
    assert same.independent_size == 36
    assert separate.ordered_prefactor == Fraction(1, 2)
    assert same.ordered_prefactor == Fraction(1, 4)
    assert [symmetry.permutation for symmetry in separate.spec.symmetries] == [
        (1, 0, 2, 3, 4, 5)
    ]


def test_exchange_phase_agrees_with_independent_exterior_product():
    layout = ExplicitTriplesSpinLayout(UCCSDLayout(3, 3, 3, 3), (2, 2))
    holes = (("b", 1), ("a", 0), ("a", 1))
    particles = (("b", "v", 1), ("a", "v", 0), ("a", "c", 0))
    canonical = layout.coordinate("aab", (0, 1, 1), "aab", (0, 0, 1), "vcv")
    assert canonical is not None

    for hole_order, particle_order in product(permutations(holes), permutations(particles)):
        hspin, hindex = zip(*hole_order)
        pspin, pspace, pindex = zip(*particle_order)
        result = layout.coordinate("".join(hspin), hindex, "".join(pspin), pindex, "".join(pspace))
        assert result is not None
        assert (result.block, result.occupied, result.particles) == (
            canonical.block, canonical.occupied, canonical.particles,
        )
        hole_exterior = np.array([[a == b for b in sorted(holes)] for a in hole_order])
        particle_basis = (("a", "v", 0), ("a", "c", 0), ("b", "v", 1))
        particle_exterior = np.array([[a == b for b in particle_basis] for a in particle_order])
        assert result.sign == round(np.linalg.det(hole_exterior) * np.linalg.det(particle_exterior))

    assert layout.coordinate("aab", (0, 0, 1), "aab", (0, 0, 1), "vcv") is None
    assert layout.coordinate("aab", (0, 1, 1), "aab", (0, 0, 1), "vvv") is None
    assert layout.coordinate("aaa", (0, 1, 2), "aab", (0, 0, 1), "vcv") is None
    with pytest.raises(ValueError, match="out of range"):
        layout.coordinate("aab", (0, 1, 1), "aab", (0, 2, 1), "vcv")
    with pytest.raises(ValueError, match="out of range"):
        layout.coordinate("aab", (False, 1, 1), "aab", (0, 0, 1), "vcv")


def _spatial_spin_sum(left, right):
    """Spin-free reference from cross-permutation cycle counts.

    Contract two rank-three exterior products with identical spin restrictions.
    One binary spin sum survives per cycle in the relative permutation.
    """
    terms = []
    for p, q in product(permutations(range(3)), repeat=2):
        relative = tuple(q.index(p[i]) for i in range(3))
        cycles = 3 if relative == (0, 1, 2) else 2 if any(
            i == j for i, j in enumerate(relative)
        ) else 1
        phase = 1 if cycles % 2 else -1
        terms.append(phase * 2**cycles * prod(left[i, p[i]] * right[i, q[i]] for i in range(3)))
    return sum(terms)


@pytest.mark.parametrize("complement_legs", [0, 1, 2])
def test_matched_reference_spin_block_sum_matches_spatial_cycle_contraction(complement_legs):
    layout = ExplicitTriplesSpinLayout(UCCSDLayout(3, 3, 2, 2), (1, 1))
    # Spatial matrices are shared across alpha and beta. Entries for CABS use
    # a different spatial column even when their local index also starts at 0.
    left = np.array([[1, -2, 0], [3, 1, 2], [-2, 1, 1]])
    right = np.array([[3, 0, 4], [-1, 2, 1], [1, 4, -1]])
    via_blocks = Fraction(0)
    representatives = 0
    for block in layout.blocks(complement_legs):
        same_spin = np.equal.outer(tuple(block.channel), tuple(block.channel))
        dense = 0
        nonzero_coordinates = set()
        for idx in product(*(range(n) for n in block.spec.shape)):
            i, j, k, *virtuals = idx
            particles = tuple(a + 2 * (d == "c") for a, d in zip(virtuals, block.particle_domains))
            canonical = layout.coordinate(block.channel, (i, j, k), block.channel,
                                          tuple(virtuals), block.particle_domains)
            selecting = np.ix_((i, j, k), particles)
            v = round(np.linalg.det(left[selecting] * same_spin))
            w = round(np.linalg.det(right[selecting] * same_spin))
            if canonical is None:
                assert v == w == 0
            else:
                nonzero_coordinates.add((canonical.occupied, canonical.particles))
            dense += v * w
        assert len(nonzero_coordinates) == block.independent_size
        representatives += block.independent_size
        via_blocks += block.ordered_prefactor * dense

    via_spatial = sum(
        _spatial_spin_sum(left[np.ix_(holes, particles)], right[np.ix_(holes, particles)])
        for holes, particles in product(product(range(3), repeat=3), repeat=2)
        if particles.count(2) == complement_legs
    )
    assert representatives > 0
    assert via_blocks == Fraction(int(via_spatial), 36)
