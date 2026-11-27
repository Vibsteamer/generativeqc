# Decision: production focal assembly takes the composed valence result

Status: implemented
Date: 2026-11-26

The [device-specific precision decision](../numerics/2026-11-27-single-pass-fp32-triples-w-forces-by-device.md)
sets which triples force telemetry is eligible for assembly.

## Problem

Production assembly received the valence energy, forces and components as
separate arguments. A caller could pass an empty component tuple, or unrelated
production components, with the energy and forces of a diagnostic single-pass
FP32 W result. The diagnostic refusal checked only the supplied components, so
those values were assembled. The regression exercises the bypass with `valence_components=()`.

## Decision

- `assemble_production_focal_energy_and_forces` takes `valence`, the composed
  `FocalRunResult` (including `FocalMolecularResult`) for energy and forces, or
  a `FocalEnergyResult` for the energy only. The separate valence energy,
  forces and components arguments were removed.
- A result without components is refused before any other check. Diagnostic
  components and single-pass W force telemetry are refused as before.
- The valence energy and forces are recomposed from the components and weights
  with the focal composition arithmetic: `math.fsum` of weighted energies and
  `np.sum` over weighted FP64 forces in stored order. The stored totals must
  equal the recomposed values exactly.
- Components must be converged `FocalEvaluation` objects with a `FocalTerm`, or
  `FocalEnergyComponent` objects for an energy-only result. Each weight entry
  must name its component's role, basis and cardinal. Components share one
  surface identity and have finite values with one `(natoms, 3)` force shape.
- An energy-only result cannot be combined with Delta-CV or Delta-SR forces.
- Delta-CV and Delta-SR inputs are unchanged.

## Rejected alternatives

- Refusing only the empty tuple still allowed diagnostic values to be paired
  with any production component.
- A tolerance on recomposition would admit a substituted component whose
  difference is below the tolerance. Single-pass FP32 force differences are
  near 1e-7 Eh/bohr. Exact equality is attainable because assembly repeats the
  composition arithmetic in the same order.
- A separate identity token or hash would add state that every producer must
  carry. Recomposition binds the values to the components without new state.
- Keeping the bare-number keywords as an alternative entry would retain the
  bypass.

## Invariants

- Production assembly never uses valence energies or forces that are not the
  weighted sum of the result's own components.
- Recomposition uses the same arithmetic and order as focal composition. A
  change to composition arithmetic must change both together.
- An empty component tuple is never accepted.

## Evidence

`tests/python/test_triples_single_pass_fp32_force_policy.py` assembles a real
`singlepoint` water result with Delta-SR and reproduces its totals exactly. It
also composes a real tagged diagnostic triples evaluation into that result and
checks that assembly refuses it. Passing its values through the removed
keywords raises `TypeError`. Pairing them with an empty component tuple, or with
the production components, raises `ValueError`. Removing the empty-tuple
refusal fails 2 tests. Removing the stored-versus-recomposed check fails 5.
`tests/python/test_upstream_pr_hardening_gate1.py` assembles the Plan-A recipe
from a typed valence result.

## Consequences

Callers must keep the composed focal result. Staged pipelines that evaluate
components separately obtain it from `FocalPlan.collect`, keeping each
component's telemetry. External scripts that assembled bare valence numbers
must migrate.

`valence_correlation_space` is still declared separately. `FocalTerm` carries
only a surface hash that includes the correlation space, so assembly cannot
check the declared space against the result.

## Revisit when

Focal results carry their correlation-space identifier, so assembly can bind it
too, or focal composition changes its arithmetic.

## References

- `python/generativeqc/_f12_frozen_core.py`
- `python/generativeqc/experimental/f12_focal.py` (`_assemble`, `FocalPlan.collect`)
