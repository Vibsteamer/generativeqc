# Decision: Retain the RHF root for spatial focal triples energies

Status: implemented
Date: 2026-11-22

## Problem

The focal spin-resolved conventional triples forward pass takes substantially
longer than the spatial DF-RCCSD(T) owner on the same common-fit RHF water and
methane controls. An ordinary standalone spatial call would independently solve
SCF and could lose the occupied root tracked by a signed focal displacement.

## Decision

An all-electron, all-active, common-fit CUDA RHF energy request can use the
spatial DF-RCCSD(T) engine on the determinant already owned by the focal
reference. The native entry checks orbital Hamiltonian, fitting identity, and
device before forming the spatial DF correlation source. It returns the
conventional triples increment. Numeric or identity errors remain visible.
Missing ABI support and explicit resource refusal retain the bounded focal
spin-resolved route.

## Invariants

The public reference projector checks the signed point's occupied subspace before
this call. The spatial owner consumes the reference without another SCF solve.
Separate JK and correlation fits, ECPs, UHF, and force requests continue through
their existing owners. The spatial return does not introduce a scaled or
explicitly correlated triples term.

## Evidence

Matched conventional scalar differences were about 1.05e-12 Eh for water and
1.08e-13 Eh for methane in the common-fit L20 calibration. Those jobs compared
the prior owners and did not exercise the new retained spatial entry. The
CUDA opt-in signed LiH regression checks the independent central energy and
the center analytic increment at both finite-difference steps. The forward
center and each signed point record whether the spatial ABI published its
scalar and whether the spin-resolved ABI returned. The early retained-root
trace inside the solver records entry before the triples calculation, so it
does not establish which result the caller received.
