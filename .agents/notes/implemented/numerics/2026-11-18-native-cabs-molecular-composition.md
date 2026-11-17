# Decision: compose the native CABS frame with shared F12 and singles sources

Status: implemented in the integration harness; L20 qualification tracked separately
Date: 2026-11-18

## Problem

Qualifying a projector with supplied weights does not establish that the
molecular F12 functional can generate those weights from the same native frame.
The combined orbital/OptRI AO basis also need not be OBS-first. Treating its
stored order as OBS-first changes the complement and the molecular scalar.

## Decision

The molecular integration harness accepts an explicit bijective AO permutation,
checks the embedded OBS columns, and evaluates the native CUDA forward frame in
OBS-first order. It restores that frame to the original AO source order before
the generated molecular contractions. No CPU complement calculation is used in
the GPU worker.

CABS singles receives the same extended Fock matrix as the F12 pair functional.
Its full Fock cotangent is added before the common Fock/fit/AO reverse pass.
The shared CABS coefficient cotangent is converted to an invariant projector
cotangent and passed through the native overlap response. The permutation is
then reversed. Only the OBS coefficient cotangent remains for subsequent SCF
response; sending the consumed CABS columns there again would double-count it.

The singles action is an explicit callback. Absence means no singles term,
not an implicit CPU or reference-program fallback. HF and ordinary MP2 energies
are not part of this increment and must not be added again by this harness.

## Evidence and limits

The H2/H2O qualification uses the original complete molecular AO sources and
unchanged geometry/Q/M. Independent Cholesky/QR/SVD complements and a Sylvester
singles solve check the new composition against the native eigenspace route.
Two overlap-direction steps keep the OBS block and supplied AO integrals fixed.
The scalar/frame parity gate is 5e-11 Eh and the overlap directional derivative
gate is 2e-8. Original source hashes and references remain in the batch receipt.

The original stock Psi4 water scalar discrepancy is retained as a separate,
unpassed full-chain gate. Stable metric attribution is not authority to weaken
it. These fixed-source derivatives are not physical nuclear forces, and local
CPU oracle timings are not GPU timings. The remaining native integral and SCF
response chain, independent molecular E/F, and endpoint performance still need
qualification before a production F12 API can be claimed.

## Revisit when

The resident source provider owns the complete molecular step. Move this
composition behind that provider, retain the explicit AO/physical-center maps,
and compare complete endpoint work and wall time with the prior CBS recipes.
