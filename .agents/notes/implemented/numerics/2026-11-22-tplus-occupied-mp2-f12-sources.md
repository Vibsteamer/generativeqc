# Decision: Share the canonical occupied MP2 and F12 sources of (T+)

Status: implemented
Date: 2026-11-22

## Problem

The Kállay (T+) ratio requires separate MP2 and fixed-geminal increments for each occupied orbital. Scaling by their total energies loses the fragment partition. A per-orbital response also places different weights on canonical occupied orbitals, so the existing invariant total MP2 denominator response cannot supply this derivative directly.

## Decision

Assign each distinct spin-orbital pair half of its contribution to each occupied leg. For the ordinary split-fit graph the ordered pair matrix already contains the same-spin factor of one quarter, and each axis marginal contributes one half. The reverse seed of an individual pair is the arithmetic mean of its two occupied cotangents. Group paired spin contributions by spatial orbital only for the restricted case. The ordinary canonical response differentiates the fitted factors and orbital energies, then uses the physical same-space Fock constraint before coupled occupied/virtual response. The selected F12 graph reuses its same canonical, CABS, and fitting response with the pair seed substituted.

## Invariants

- A partition belongs to the precise determinant, orbital ordering, fitting bases and correlation space from which it was constructed.
- Canonical stationary RHF and UHF have zero OBS singles value and differential in Eq. (9). HF+CABS singles belong to a different focal term.
- The existing focal pair owner supplies 3C(FIX). The paper's worked 2B fixed-amplitude protocol requires its own pair identity and independent comparison.
- Equal occupation numbers alone do not establish a restricted frame. Exact nonstationary torque at unresolved canonical degeneracies is rejected.
- This code provides MP2 and F12 parts of the chain rule. The weighted native triples and CC amplitude response are separate owners.

## Evidence

`tests/python/test_focal_tplus_mp2_sources.py` compares the spin partitions and their transpose to explicit sums and two finite-difference steps. It also checks a nonuniform analytic MP2/F12 force against recomputed displaced canonical frames when the native host shell library is available.

## References

Kállay et al., J. Chem. Phys. 155, 034107 (2021), Eqs. 7–13 and Appendix B, [doi:10.1063/5.0057426](https://doi.org/10.1063/5.0057426).
