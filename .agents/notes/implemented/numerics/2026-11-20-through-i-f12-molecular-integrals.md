# Decision: Through-i molecular Coulomb and geminal derivative sources

Status: implemented
Date: 2026-11-20

## Problem

Br and I PP-F12 use orbital-plus-CABS and fitting spaces with i shells. The
existing molecular sink's through-h workspace cannot evaluate the resulting
two-center metric blocks and three-center source blocks.

## Decision

The qualification builder selects an explicit FP64 `--cabs-i` domain for the
existing forward and weighted shell interfaces. Cartesian i/i/i values require
moments through order 18. An analytic center derivative raises one factor and
requires order 19. The corresponding axis coefficients and radial moments use
20 slots. The third center follows by simultaneous translation, and a
two-center metric retains a zero absent-center channel.

The i variant reuses one emitted axis polynomial under exchange of the two
factors on the same electronic coordinate. The branch swaps their powers and
mean offsets together. This bounds source size while evaluating either
derivative raise with the shared Gaussian recurrence.

## Invariants

The legacy and through-h builders retain their previous shell admission, buffer
shape, and arithmetic. Radial normalization, Cartesian CCA component order,
and positive derivative signs follow the existing sink ABI. The generated
source remains independent of the CPU oracle and public F12 admission.

## Evidence

The axis tests use independent tensor-product Gauss–Hermite quadrature through
total degree 19. The independent sink fixture takes i exponents from the Br/I
PP-F12 tables and integrates the original coupled Gaussian functions with
Gauss–Hermite quadrature. For Coulomb and f/r it adds adaptive radial
integration. Both two-center and three-center weighted values and all center
responses are checked for Coulomb, f, f/r, f² and the double-commutator
convention. Two independent scalar displacement steps form the reference
gradients.

## Consequences

The qualification library gives the heavy-element F12 integral sink values and
weighted nuclear pullbacks through i. Br/I molecular ECP derivatives, spin
response and full correlated energies and forces require separate integration
and validation. Kernel compilation and complete molecular performance need
separate device measurements.
