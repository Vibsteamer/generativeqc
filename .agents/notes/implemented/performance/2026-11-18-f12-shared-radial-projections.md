# Decision: reuse radial moments across contracted F12 projection terms

Status: implemented
Date: 2026-11-18

## Problem

The first independently qualified contracted source expanded each Cartesian
projection term through a complete primitive evaluator. Its radial moments were
invariant across those terms, but the expensive fitted-geminal radial provider
was invoked again for every term. This amplification matters for pure transforms
and for externally weighted shell derivatives.

## Decision

Factor the existing F12 primitive lowering into a radial context and its unchanged
angular derivative DAG. The contracted consumer prepares one context per primitive
combination and operator, then reuses its moments across the projection. Retain
the original compensated accumulation order, normalization, explicit projection
coefficients, and primitive-Cartesian work budget. Only up to ten output scalars
are accumulated. No coordinate-major derivative tensor is materialized.

This is bounded sharing within a contracted projection, not a molecular AO cache
or a completed molecular F12 force path. Angular expressions and the caller's
different projections may still repeat work; a later shell schedule can share
those only after independent endpoint qualification.

## Rejected alternatives

Do not relax the Cartesian work budget merely because radial calls decrease.
Do not introduce a second integral recurrence, normalize coefficients implicitly,
or assume identical auxiliary/CABS spaces. Do not infer molecular speedup from
the source panel or replace the existing full molecular response gates.

## Invariants and evidence

The shared source uses the same generated derivative graph and radial provider as
the standalone primitive. Failure leaves output untouched, including insufficient
budgets and nonfinite inputs. Zero projection remains a valid zero contraction.
The 48 earlier independent Libint2 cases are preserved. A 72-row panel additionally
contracts every Cartesian component of a g–s metric and an s–p–d three-center shell
with signed weights; all four operators and three beta values remain. Both
independent-center and shared-atom two-step finite differences are required.

Qualification: `benchmarks/f12_contracted_integrals.py`, with
`tests/reference_data/f12/libint_contracted_full_shells.json`. The opt-in repeated
row timing includes memory allocation, transfers and synchronization. The L20
stage compares the previous emitter and this change in the same allocation with
the same harness, fixture, compiler and GPU. Timing is reported as source evidence
only, separately from compilation and from the outstanding molecular E/F work.
