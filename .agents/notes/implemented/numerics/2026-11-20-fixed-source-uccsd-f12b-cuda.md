# Explicit CUDA execution for fixed-source UCCSD-F12b

Status: implemented

Date: 2026-11-20

The [matrix-free runtime](2026-11-20-fixed-source-uccsd-f12b-matrix-free-runtime.md)
establishes the packed amplitudes, Lambda coordinates and physical
convergence tests. The fixed-source tools owner also accepts an explicitly
supplied native CUDA tensor executor for those equations.

The constructor admits the primal plan against the selected CUDA target and
executor budget before copying the inputs. Its simultaneous numeric budget
includes both the host-owned iteration state and the planned device arena.
Lambda uses the existing per-program CUDA planning and semantic-work gates.
The CUDA executor releases its arena after each call and a backend mismatch
raises an error. DIIS and packed GMRES continue to run on the host.

These bindings admit canonical spin-MO inputs already qualified by their
caller. The molecular integrals, fixed source construction, orbital
response and nuclear derivatives remain separate from this solver.

## Orbital-gap preconditioning

The fixed-source Lambda owner can explicitly right-precondition its packed
GMRES iterations with the positive orbital-energy gaps of the unshifted
reference. The solver binds the immutable gap vector to its identity and
reserves five packed FP64 vectors for the diagonal, inverse and temporary
storage. The existing transpose solver still rechecks the unpreconditioned
residual and the owner still replays physical Lagrangian stationarity.

The 170-variable NH₂ replay with native CPU TensorIR used 25 transpose
actions under the preconditioned policy. The original unpreconditioned
replay used 114 actions. The full-space CUDA solve requires an independent
work and convergence qualification.

## Replaying checked checkpoint vectors

The private fixed-source owner can re-evaluate a checkpoint's exact packed
amplitudes against its bound canonical feeds. All residual blocks must pass
and the fresh energy must agree with the checkpoint before the owner binds
the detached vector as a converged state. A later parameter pullback can
likewise supply packed physical Lambda multipliers. The existing response
budget and fresh primal and Lagrangian stationarity checks apply before
the native parameter contraction. The replay reports fresh diagnostics for
the supplied vectors. The checkpoint caller owns external artifact and feed
identity checks.
