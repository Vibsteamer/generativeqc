# Explicit triples spin coordinates

Status: implemented
Date: 2026-11-22

## Problem

Connected explicit triples require independent occupied and particle
permutations across four conserved UHF spin channels. A complementary
particle and an orbital virtual can share a local integer index while
representing different spin orbitals. The usual ordered-block contraction
weights for orbital-only triples cannot be reused for every complementary
sector.

## Decision

Use alpha-before-beta order on both sides of the excitation and orbital
virtuals before complementary particles within each spin. The signed
coordinate map independently orders the hole and particle determinants, so
the same phase applies to the forward value and reverse scatter. The inverse
of the product of each typed domain's factorial converts the ordered
block contraction to a sum over individual determinants.

## Invariants

The domains carry spin and the orbital/complement distinction even when
their populations coincide. Only repeated indices in the same typed domain
are Pauli zeros. A zero-population channel has a well-defined empty layout.
The equation inventory selects which complementary sectors participate.

## Evidence

The orbital-only blocks reproduce the existing conventional channel shapes
and their exact ordered contraction weights. An exterior-product determinant
checks the map for independent hole and particle reorderings. A spatial
relative-permutation cycle sum independently checks the equal-reference
closed-shell reduction with zero, one and two complementary particles.
See `tests/python/test_explicit_triples_spin.py`.

## Consequences

The compiler can bind distinct spin and particle-space axes before the
screened extended-SP terms are lowered. This change establishes the storage
and permutation conventions for those terms. Molecular energy, CUDA
execution, and stationary nuclear response have their own acceptance gates.
