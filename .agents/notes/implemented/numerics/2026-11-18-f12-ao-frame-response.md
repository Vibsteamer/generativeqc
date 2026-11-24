# Decision: differentiate the shared OBS/CABS frame and all AO source maps

Status: implemented (source-map component; complete molecular force pending)
Date: 2026-11-18

## Problem

The molecular 3C(FIX) pair graph supplies adjoints on MO operators, fitted
Coulomb factors and the extended Fock matrix. An analytic nuclear force also
requires both occurrences of the same orbital coefficient matrix, all raw AO
operators and the fitting metric to remain live. Treating the CABS frame or
one leg of a transform as a constant drops terms.

## Decision

`molecular_orbital_sources` generates the one-electron transform and the five
three-center transforms with one common differentiable coefficient input.
Coulomb retains both complete OBS/CABS axes for the extended Fock; the other
four operators retain an occupied left axis. Each transform uses binary matrix
contractions, including the one-electron map. The TensorIR transpose sums all
uses of the shared frame, including occupied column slices. The existing metric
solve response owns the dependence of fitted coefficients on the Coulomb metric.

The benchmark has a source-map qualification and a separate diagnostic of the
entire fitted-energy chain. A source-map pass is not a molecular energy/force
pass. Reference AO tensors, reference orbitals and independent NumPy products
are fixtures for qualification, not a CPU production fallback. The real-device
adapter requires FP64 CUDA and records compilation, endpoint traffic and actual
device memory. Its explicit 2 GiB budget covers this small, materialized reference
panel; production must use integral/shell blocks rather than scale this layout.

## Invariants

- The common AO basis, row ordering and occupied/OBS/CABS column ordering are
  identical on every source. Matrix symmetry does not permit dropping one
  coefficient occurrence in reverse mode.
- All seven source inputs receive adjoints. Tests perturb coefficients, one-
  electron integrals, Coulomb integrals and a non-Coulomb operator independently.
- The source-map GPU gate uses both steps 2e-4 and 1e-4, error <=2e-8, and
  independent source values with atol 5e-10, rtol 2e-10.
- Retain the stricter full-chain energy gate and its failed diagnostics; no
  production label or complete analytic-force claim follows from this stage.

## Evidence and unresolved conditioning

Local H2 and water source checks use independent Psi4 1.11 cc-pVDZ-F12,
OptRI CABS, aug-cc-pVTZ-RI fitting, beta=0.9, and all-electron closed shells.
The original fixture geometries and hashes live in the validation stage
`native-f12-ao-response-20261118/oracle`; they are not substituted benchmark
members. Unit tests also use nonorthogonal random rectangular coefficient
matrices and independent matrix products.

The diagnostic rebuilding all MO sources and Coulomb fits from those AO arrays
has energy differences about -5.05e-12 Eh (H2) and +3.16e-10 Eh (water) against
stock Psi4. The water value exceeds the existing 5e-11 gate: it is not accepted.
The metric conditions are 4.60e6 and 8.96e6. Native equation residual maxima are
2.02e-12 and 4.88e-12, while reference fitted coefficients in the newly exported
full AO frame have residuals 1.31e-9 and 4.09e-9. Raw occupied-source transforms
agree to 7.45e-14 or better, but full virtual/CABS transforms and inversion can
amplify arithmetic differences. This is evidence to diagnose, not proof that
all of the energy discrepancy is harmless or a reason to widen the gate.

A combined arbitrary AO/frame perturbation of the water energy gives errors
1.62e-6 and 4.00e-7 at steps 1e-4 and 5e-5, consistent with a large truncation
term. Retain these failed full-chain checks. The independently qualified source
transpose does not resolve the energy-chain conditioning, CABS projector
response, SCF orbital response, or assembled molecular nuclear force.

## Next

Combine the source adjoints with native contracted/spherical nuclear integral
sources and existing projector/orbital response. Diagnose full-frame reference
conditioning and step-size convergence before claiming a complete H2O force.
Preserve the original benchmark members and their separate cloud identities.
