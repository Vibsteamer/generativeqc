# Decision for geometry-first F12 focal dispatch

Status: implemented

Date: 2026-11-20

## Problem

The focal assembler combines precomputed energy and force terms using
caller-defined identifiers. Every caller validates geometry and basis identity.
The molecular provider for each electronic role can be qualified independently.

## Decision

The experimental geometry-first entry hands explicitly requested electronic
roles to a caller-selected provider. It resolves immutable orbital, JKFit,
MP2Fit, F12 fit and CABS records before work. The surface key incorporates the
ordered geometry, charge, multiplicity, scalar ECP data, root and correlated
space identifiers. The basis key includes the fitting roles and F12 exponent.
The dispatcher rejects any detected mismatch in a returned key or executed
backend. Both entry points use the same native-owned extrapolation coefficients
and force convention.

Electronic root continuity and response convergence are provider obligations.
The numerical composition remains an orchestration layer. Configured roles or
compiled integral kernels alone establish no new method capability.

The combined conventional correction uses the existing same-reference native
component group. Its single auxiliary record must describe both fitting roles.
The grouped result exposes a complete CCSD(T) force, so this provider owns the
combined CCSD(T) minus MP2 role. Split correction providers require separate
analytic force outputs.

## Evidence

The entry tests exercise same-geometry dispatch, stale result and backend
rejection, owned result storage, ECP changes and force differences for a known
analytic surface. The conventional provider test sends the high-order role
through native CPU `Calculator.singlepoint_components` and checks the
CCSD(T) minus MP2 force against a scalar energy difference. The same test case
supports an explicit CUDA allocation opt-in. CUDA molecular F12 endpoint
evidence remains with the corresponding provider tests.

## Independently returned components

`plan_singlepoint` resolves the same ordered requests as the synchronous
dispatcher. Each geometry has its own plan. Workers can evaluate independent
roles and cardinal numbers on separate devices and return results in any
order. `FocalPlan.collect` checks each return against its request, rejects
duplicates and missing components, and uses the original summation order.
An explicit `protocol_id` binds the campaign's method and implementation
revision to the archived surface identity. Physical execution and electronic
root tracking require evidence from the component provider.

## Geometry-owned one-electron sources

The combined-basis overlap and core Hamiltonian share contraction expansion,
radial normalization, spherical phases and OBS-first order with the molecular
integral sink. The one-electron provider packs only the orbital and OptRI
records. Enumerating fitting triples here would charge a cubic shell table
to a quadratic operator. The full-basis batch computes the positive nuclear
Jacobians for both Gaussian centers and for nuclear attraction.

The OH regression places OptRI functions on additional zero-charge centers
in its independent libcint calculation. Moving each zero-charge center with
its physical nucleus preserves the public AO order and checks the separate
basis-center contributions and their sum.

## Complement on the shared molecular overlap

The public CABS projector consumes the OBS-first overlap produced by the
one-electron provider. The providers must agree on geometry, basis and backend.
A fixed-rank kernel weight propagates through the projected OptRI overlap, its
retained subspace, and the removal of OBS components. The CPU reference uses
the existing symmetric pseudoinverse response for a truncated subspace and
the inverse-metric identity when all complementary directions survive.
Contract the returned overlap weight once with the one-electron nuclear
Jacobian. Contributions from a nuclear-dependent observable seed enter separately.
