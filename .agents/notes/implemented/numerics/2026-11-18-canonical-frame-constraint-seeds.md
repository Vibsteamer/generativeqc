# Decision: complete canonical same-space constraints before RHF Z response

Status: implemented as a private component; complete molecular qualification pending
Date: 2026-11-18

## Problem

The F12 pair expression uses canonical orbital denominators. Differentiating its
AO/frame sources gives a coefficient cotangent, but that scalar extension is not
stationary under arbitrary occupied/occupied or virtual/virtual rotations.
Directly passing that weight to the existing physical RHF response violates its
same-space stationarity contract. The retained H2/H2O fixed-source audit found
maximum same-space torques of 2.30e-4 and 1.97e-3, respectively.

## Decision

For the same real canonical reference, compute M=C.T barC on the GPU. Let
D_pq=M_pq-M_qp and Delta_pq=epsilon_p-epsilon_q. A symmetric full-Frobenius
canonical-constraint seed is P_pq=-D_pq/(2 Delta_pq) within each occupied or
virtual block; its diagonal and occupied/virtual blocks are zero. Its rotation
derivative cancels D because each symmetric Fock entry contributes once.

Compiler-owned scalar IR generates this rule. Native CUDA owns storage,
finite-value checks, gap admission and failure-atomic output. It does not compute
an eigenvector derivative by differentiating iterative eigensolver operations.
The same-space seed is a constraint multiplier, not a second copy of the F12
extended-Fock cotangent already propagated through AO sources.

## Degeneracy and limits

For a gap below the declared threshold, require the corresponding torque to be
stationary within its separately declared tolerance. A stationary degenerate
block uses zero seed; a nonstationary block is refused. Do not clip denominators
or erase torques to make a molecular case pass. This establishes a locally
qualified first-order component, not smoothness across arbitrary root changes.

The caller owns canonical-reference validation, occupancy, same DF operator and
the complete source weights. The physical RHF Z and Pulay owner still follows.
Its existing reference electronic energy seed is one, so reference and nuclear
terms must not be added twice. Do not substitute Direct response for a DF state.

## Evidence

Independent small generalized eigenproblems re-diagonalize displaced Fock
matrices at two steps and evaluate a weighted quadratic orbital observable.
These check the sign and full-Frobenius factor two, including gauge-invariant
exactly degenerate blocks. The L20 qualification additionally checks CUDA frame
projection, explicit refusal of nonstationary degeneracy/nonfinite sources,
numeric budget refusal, 64-bit 43-GiB budget handling, and unchanged user outputs
on failures. CPU oracle results are not GPU performance or molecular forces.

The prior water full-scalar reference discrepancy and complete F12 force gates
remain independent. This private component does not advertise a public F12
energy/force capability.
