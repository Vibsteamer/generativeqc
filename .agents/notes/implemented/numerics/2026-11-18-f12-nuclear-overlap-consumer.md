# Decision: consume CABS overlap adjoints through native nuclear derivatives

Status: implemented, physical GPU qualification pending
Date: 2026-11-18

## Problem

The invariant CABS frame/kernel and overlap adjoints have passed L20 component
tests. That does not establish their contraction into physical atom coordinates.
Auxiliary and orbital shells can have distinct array ordering while sharing the
same physical centers; a doubled off-diagonal weight or duplicated atom would
produce an incorrect gradient even when the projector adjoint is correct.

## Decision

Add a thin private Python boundary for the existing
`generativeqc_system_one_electron_gradient_cuda` C ABI. It accepts fixed full
row-major FP64 overlap, kinetic and attraction weights and owns a private,
synchronous native context/system transaction. It calls the existing generated
CUDA derivative and contraction provider; no recurrence is copied and there is
no CPU fallback. All-zero channels can be null. Numeric staging stays bounded.

The bridge returns the energy-gradient sign, not the force sign. It excludes
nuclear repulsion, weight derivatives and orbital relaxation. These must be
assembled by the molecular caller. This addition does not expose a complete
F12 molecular endpoint.

## Independent qualification

The new physical probe is the norm of a fixed atom-centered probe projected
into CABS: `b(R)^T K(S(R)) b(R)`. Native projector reverse supplies the overlap
weight; the probe cross-overlap weight occupies only one off-diagonal block.
The native full-Frobenius contract adds the two block orientations once.
Explicit basis shells collapse all three source basis blocks onto the original
two/three physical atoms. Sources preserve OBS/OptRI/probe ordering.

PySCF/libcint supplies independent first overlap derivatives and complete
Cartesian displacements of both original H2 and H2O geometries. A Cholesky/QR
complement supplies independent displaced probe values, without reusing the
native Schur/eigenspace response. Both 4e-4 and 2e-4 Bohr steps must pass the
existing 3e-8 probe-gradient and 4e-10 probe-value gates; translation is checked.
The fixtures contain primal overlaps, so timings must not be described as full
SCF, full F12 energy/force or full integral-generation endpoint timings.

Local tests exercise ownership on native execution and system-creation errors,
FP64/shape/finite admission, full non-symmetric weights and bounded staging.
Actual L20 evidence is registered separately under
`validation/native-f12-nuclear-overlap-20261118` and batch
`18-cabs-nuclear-overlap`; the production molecular gates remain unchanged.

## Rejected alternatives

Do not expand a 3N-coordinate derivative tensor just to contract a fixed weight.
Do not infer atomic identities from AO offsets, invent duplicate nuclei for
auxiliary shells, or use the host overlap oracle as an unreported runtime path.
Do not infer full molecular forces from a probe-component pass.
