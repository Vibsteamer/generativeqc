# Reject the initial generated nonzero-angular F12 shortcut

Status: rejected
Date: 2026-11-09

The [CPU s/p value qualification](../implemented/numerics/2026-11-09-f12-native-mo-space-and-order.md)
later superseded the s/p part of this decision.

## Problem

The initial generated CPU dispatcher compiled and reproduced a coincident
all-s Gaussian control. Cross-center p components disagreed with an independent
Psi4 AO reference. Comparing the dispatcher to its own scalar graph exercised
the same assumptions about component ordering and the radial recurrence.

## Decision

Use the shared Hermite recurrence for nonzero angular momentum in the initial
CPU implementation. Admit only the all-s generated case until the angular
components pass an independent comparison. Keep the physical Gaussian and
operator conventions fixed through that comparison.

## Rejected alternatives

- Admit p, d or f components on the strength of compilation or all-s parity.
- Fit a global scale or choose an arbitrary transpose to make the AO entries
  agree. Resolve the ordering, Gaussian prefactors and operator definitions at
  the primitive and contracted boundaries before checking molecular energies.

## Evidence

The coincident all-s value was `0.3535533905932739` in both the scalar graph
and the independent reference. The subsequent H₂ checks sampled cross-center
p and higher-angular AO entries against Psi4 for the geminal, its square, its
product with Coulomb and the gradient product. The linked follow-up records the
result and the narrower scope of the generated CPU path.

## Revisit when

Require independent cross-center values for the affected angular signatures
and operators. Where the change serves metric and three-center sources,
qualify both source ranks. Where it serves a molecular endpoint, also check its
orbital-space transforms and energy components.

The public tree exercises the same acceptance principle with independent
[Libint2 primitive fixtures](../../../tests/python/test_f12_df_integrals.py)
and [contracted shell fixtures](../../../tests/python/test_f12_contracted_integrals.py).
Those tests qualify the current generated density-fitting sources separately
from the historical CPU dispatcher.
