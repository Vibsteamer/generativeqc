# Additive core–valence corrections and geminal normalization

Status: implemented
Date: 2026-11-23

## Decision

A core–valence increment differences the active-core and frozen-core
correlation energies in the same core–valence basis, on the same DF-HF
determinant and fitted Hamiltonian. For MP2-F12 the two calculations also
share the fixed-rank CABS space. The HF and CABS singles cancel. The
ordinary MP2 variant has no geminal or CABS contribution. Both variants
use the matching analytic-force difference, and the forward pair
accumulation can collect the frozen subset in the same active-pair pass.

The production valence and core–valence roles must name matching frozen
spaces. A count that splits Br 3d or I 4d changes the correlated subspace.
The shell-gap guard verifies the selected count before accepting an
energy or force. The cardinal-specific Slater exponent is part of the
basis identity and uses the same convention for all elemental records
in the component.

The molecular Python and native shell defaults use the cusp-normalized
factor `-exp(-beta*r)/beta`. The unscaled Psi4 convention remains an
explicit reference choice. The two agree at beta equal to one. The
linear integral operators and derivatives scale by `1/beta`, while the
quadratic operators and derivatives scale by `1/beta^2`. Accept the
stored shell default only when it matches the binding's requested
normalization.

## Reference and rank invariants

A supported native orbital-angular-momentum refusal can use the focal
one-electron and JKFit sources to build the alternative DF-HF reference.
Both forward and force routes transport the same center seed. Occupied
columns projected to the displaced AO metric already satisfy
`C.T @ S @ C = I` and pass an occupied-overlap test. Euclidean QR in
these coordinates would change their density. The CUDA fallback reports
host-assisted SCF execution.

Equal frozen spaces cancel exactly. A returned F12 component nevertheless
verifies the actual CABS rank, including when all occupied orbitals are
frozen. A fixed-rank derivative keeps the corresponding frame at the
signed displaced geometries.

## Evidence

`tests/python/test_focal_cv_additive_gate1.py` compares the matched
ordinary differences against PySCF (including water and hydroxide) and
the MP2-F12 differences against Psi4 3C(FIX) with the same normalization.
It covers signed force differences and equality of chunked and unchunked
shell and fitting contractions. The earlier matched Psi4 water and HCl
core–valence increments agreed within 2.6e-10 and 1.1e-9 Eh respectively.
`tests/python/test_f12_sink_gpu_gate1.py` checks the default and explicit
normalizations and their signed response. The displaced-density, rank
and cancellation controls are in `tests/python/test_audit_fix4_endpoints_gate1.py`.

## References

- `python/generativeqc/_f12_focal_cv.py`
- `python/generativeqc/_f12_frozen_core.py`
- `src/integrals/f12_geminal.hpp`
- [Frozen-core shell-gap selection](2026-11-26-frozen-core-shell-gap-threshold.md)
