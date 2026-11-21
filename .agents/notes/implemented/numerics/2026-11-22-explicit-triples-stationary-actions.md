# Decision: Separate the projected triples scalar from its stationary residual

Status: implemented
Date: 2026-11-22

## Problem

The explicit-triples formulation supplies a projected scalar and a linear
physical residual. Screening may retain terms in different places in those
two graphs. The existing conventional triples denominator is tied to another
equation and would lose off-diagonal action terms and their derivatives.

## Decision

The compiler differentiates one supplied TensorIR program in independent
physical coordinates. It emits the residual Jacobian action, its transpose,
and the negative amplitude derivative of the separately projected energy.
With `L = E + λ·R`, the response owner solves `R_t^T λ = −E_t` and seeds the
parameter reverse with one for `E` and `λ` for `R`. This includes the response
of both the inhomogeneous residual source and the selected linear action,
even when the action has cross-block couplings. The production runtime owns
the solves, resource scheduling and scatter through the physical source maps.

## Invariants

The equation provider must place amplitudes and residuals in a matching,
nonredundant Euclidean block basis. Any spin-permutation weighting or basis
metric belongs to the equation or an explicitly declared physical block map.
The scalar and residual must identify the same screened model and parent
state. Bind parent CC, orbital, CABS, DF and ECP derivatives only through the
parameter cotangents generated for that model.

## Evidence

`test_explicit_triples_stationary.py` checks a nonsymmetric residual action
against independent dense primal/transpose solves. It verifies source,
off-diagonal action, denominator and explicit scalar contributions against
analytic NumPy formulas and central differences at both 0.0004 and 0.0002.
An independent block fixture tests triangular coupling and zero direct
energy dependence. These tests qualify the stationary algebra only.

## Consequences

There is no dense solver dependency in this compiler module. Integration
requires the source-verified explicit contraction graph, a bounded native
stationary owner and matched-molecule energy and force evidence.
