# Decision: Matrix-free fixed-source UCCSD-F12b runtime

Status: implemented

Date: 2026-11-20

See the [explicit CUDA executor decision](2026-11-20-fixed-source-uccsd-f12b-cuda.md)
for the backend extension.

## Decision

A private tools owner admits canonical spin-MO Fock and integral blocks
together with the fixed geminal feeds. Its equation identity includes the
fixed-source TensorIR graph. It detaches the numerical feeds and binds their
hash to the orbital-energy vectors and caller-provided reference identities.
The caller qualifies the matching spin frame before constructing the owner.

The conventional and fixed-source tools owners reuse packed amplitude
iteration and the same Lambda admission and action policy. They remain
distinct types so that conventional UCCSD(T) and orbital force consumers
retain their method-specific admission. The fixed-source owner currently
executes with the native CPU TensorIR executor.

## Numerical contracts

DIIS proposes packed amplitude steps. The resulting state must pass fresh
energy and five-block residual checks. Lambda uses packed GMRES applied to
the compiler transpose graph, then checks fresh Lagrangian stationarity.
The full residual seed has a factor of one quarter in each same-spin
doubles block. The reported physical dual uses ordinary signed expansion.

The reduced-space reference can form a dense Jacobian for validation. The
full NH2 test space contains 52,937 excitations, so a dense FP64 Jacobian
would use over 22 GB. Production eligibility still depends on full-space
native execution, convergence at an explicit work budget, and separate
orbital and nuclear response.
