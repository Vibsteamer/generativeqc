# Qualify CPU F12 s/p values and molecular orbital ordering

Status: implemented
Date: 2026-11-09

This historical CPU result superseded the s/p part of the
[initial angular shortcut rejection](../../rejected/2026-11-09-unqualified-generated-f12-angular-shortcut.md).

## Problem

Matching isolated primitives still left disagreements in the molecular energy.
The combined orbital and complementary bases needed an explicit mapping from
the reference real-spherical order. Mixed orbital and complementary blocks
also needed the coefficient rows and dimensions selected independently on each
axis.

## Decision

Admit the generated s/p value kernels in the CPU reference path after
independent AO and molecular checks. The value emitter selected the value root.
In that implementation d/f and higher components used the shared Hermite
recurrence.

Build the native union with all orbital-basis shells before the complementary
shells, and use the same order for embedding. A chemists' AO block `(P,Q,R,S)`
selects the four axis spaces independently. After contraction, the permutation
`(0,2,1,3)` produces physicists' order for the intermediate builder. A complete
RI virtual axis uses the full RI coefficient space. Apply the permutation once.

## Evidence

The historical restricted H₂ comparison used a bond length of 1.4 bohr,
cc-pVDZ-F12 with OptRI and a six-Gaussian geminal at β = 0.9 bohr⁻¹. A
62-function combined overlap established the basis mapping. Separate tests
checked cross-center p and higher-angular samples for all four AO operators, a
mixed orbital/RI transform, and the ordinary MP2, F12 and CABS-singles energy
components against Psi4. The ordinary MP2 control was about
`-0.02958072647075` Eh. This qualification covered the CPU energy reference
at those settings.

The public tree has separate generated
[primitive](../../../../tests/python/test_f12_df_integrals.py) and
[contracted](../../../../tests/python/test_f12_contracted_integrals.py) checks
against independent Libint2 data, including a weighted full s–p–d shell
contraction. Its [robust fitting regression](../../../../tests/python/test_f12_df_blocks.py)
checks rectangular occupied/CABS projections against independent Psi4 H₂ and
H₂O fixtures. The current sources have their own angular and derivative coverage.
The [Gaussian DF decision](2026-11-17-f12-gaussian-df-moments.md) describes
those source boundaries and the separate operator metrics.
