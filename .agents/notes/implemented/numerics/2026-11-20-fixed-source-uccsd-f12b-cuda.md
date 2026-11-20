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
