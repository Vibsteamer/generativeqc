# Decision: qualify canonical multipliers with actual molecular F12 weights

Status: implemented qualification harness; real-device composition pending
Date: 2026-11-18

## Problem

Passing an isolated canonical multiplier test does not establish that the full
F12 and CABS-singles pullbacks deliver the right occupied/virtual orbital weights.
The H2 and H2O fixtures have nonzero same-space torques even after consuming the
CABS columns in the overlap-projector reverse pass. Ignoring these torques before
the physical RHF response would omit part of the canonical orbital constraint.

## Decision

The benchmark f12_molecular_canonical.py joins the existing molecular CABS/F12
chain to the generated rotation-gradient map and CUDA canonical multiplier.
It retains only the original OBS columns in the reference response and checks
their corrected same-space stationarity without changing any denominator or
degeneracy threshold. The added seed is a constraint multiplier, not a duplicate
of the extended Fock source already differentiated by the molecular chain.

Two-level, same-space symmetric Fock perturbations are independently diagonalized
with SciPy for validation. The resulting orbital rotations are test inputs, not a
CPU eigensolver in a production endpoint. Native molecular scalar differences at
two step sizes check the contraction of the multiplier with each perturbation.
H2 checks a virtual direction; H2O checks occupied and virtual directions. Small
gap spectator blocks remain fixed, and their stationarity is explicitly audited.

## Qualification boundaries

- The new eigensolver differentiation is independent of the analytic multiplier.
  The CPU scalar preflight reuses previously qualified TensorIR equations; it is
  not an additional independent whole-method Psi4 implementation.
- The molecular AO integrals and orbital source frame are externally prescribed.
  This test is not a physical DF-RHF Z solve or a complete nuclear gradient.
- All native values, pullbacks, CABS operations and constraint multipliers must
  execute on an actual L20. Local reference time is not GPU performance.
- Keep the unresolved stock Psi4 water full-scalar tolerance and full Cartesian
  reference step-size sensitivity separate. A fixed-source composition pass
  does not waive either gate.
- Do not reduce a full-method acceptance criterion to these three directions.
  This is a targeted composition check, followed by same-reference DF-RHF Z,
  all integral/metric nuclear sources and final energy/force qualification.

The source/input/archive identities and result directory are registered through
the existing development watcher, independent of completed conventional MP2
baseline components. Existing successful and in-flight jobs are reused.
