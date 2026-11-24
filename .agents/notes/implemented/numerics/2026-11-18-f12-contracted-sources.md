# Decision: contract F12 sources with explicit radial and angular conventions

Status: implemented; L20 qualification tracked independently
Date: 2026-11-18

## Problem

Primitive source derivatives do not validate normalized contracted functions or
spherical projections. Those factors must be applied identically to values
and nuclear derivatives before the molecular source adjoints can be consumed.

## Decision

Generate a bounded scalar two-/three-center consumer that calls the existing
physical-center F12 DAG. The caller supplies concatenated primitive exponents,
normalized radial coefficients and explicit Cartesian projection weights.
Every term shares its centers and angular shell signature. FP64 compensated
accumulation combines values and all physical-center derivatives in the same
order. Scratch consists of at most ten accumulated scalars, their compensation
and one primitive result; there is no primitive-by-molecular-coordinate tensor.

No pure-function phase/order or primitive normalization is guessed. The native
molecular basis owner must provide those conventions. Source generation stays
independent of the public runtime, PySCF, Psi4 and GPU availability.

The work budget explicitly counts the product of primitive counts times the
number of Cartesian projection terms. Checked multiplication precedes work.
Invalid counts, nonfinite data, exceeded budgets or failed integral evaluation
leave the caller's output unchanged. No partial scientific result is published.

## Scope and scheduling tradeoff

This establishes a scalar source contract, not a whole-shell or molecular
production schedule. Angular terms currently call the shared primitive source
individually and therefore repeat their radial moments. The benchmark reports
that work; it is not described as a shell-reuse performance improvement.
A future whole-shell scheduler should share geometry/radial work and contract
method cotangents early while preserving this numerical contract.

## Independent evidence

The reference uses Libint2 contracted shells with native pure transformations.
Radial coefficients are extracted from actual cc-pVDZ-F12 orbital and
aug-cc-pVTZ-RI fitting shells. The panel includes Cartesian s/p, pure d/g,
longer oxygen-core contraction, four geminal operators and three Slater ranges.
Derivative references use Gaussian raising/lowering evaluated independently by
Libint2. Native tests also check two displacement steps, translation invariance,
deterministic repetition and failure without publication. CPU reference/host
tests and physical L20 timings are kept separate; none qualifies a complete
SCF/MP2-F12 molecular analytic force.

## Revisit when

The molecular AO source provider owns shell tiles, public spherical ordering,
shared-center accumulation and resident method cotangents. Reuse this generated
mathematics there instead of introducing another recurrence implementation.
