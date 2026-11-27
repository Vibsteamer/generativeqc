# Orbital recovery and the scope of focal Hessian searches

Status: implemented
Date: 2026-11-27

## Decision

Use the converged determinant and its prepared Fock plan to check real
internal orbital curvature. In UHF, run the established level-shifted and
damped recovery first after a failed primary. Restarting this ladder from
the caller's guess can find a different basin. If it fails, second-order
refinement uses the finite last primary density, trust-region Newton steps,
and exact per-spin orbital rotations followed by the usual DIIS polish.
Detected negative curvature can trigger at most four downhill restarts in
each recovery phase. Each accepted restart lowers the energy.

A converged RHF primary returns on the original path. A failed primary may
use the same refinement with paired alpha and beta rotations, followed by
the existing first-order fallback. The recovery energy guard compares with
the earlier evaluated idempotent determinant, and with the refined
determinant when available. It refuses a recovered energy more than
`1e-6 Eh` above that reference.

At up to 60 real occupied–virtual directions the Hessian calculation spans
the full coordinate space. Above that bound it uses block-Davidson with
deterministic starts and bounded residual expansion. A converged Ritz
residual measures an explored mode. The search trace records the scope
with `lowest_hessian_ritz_value`, `ritz_converged` and
`hessian_search_space`. Additional unstable directions can be orthogonal
to the explored space, and distinct stable solutions can coexist.

Report physical curvature in independent spin coordinates for UHF and a
common spatial rotation for RHF. The factors on the internal operator
`A'` are 2 and 4 respectively. The common-angle RHF derivative moves both
spins, so the unit paired-spin coordinate has half its curvature. The
downhill threshold remains `-1e-5 Eh` in the reported convention.

## Evidence

`tests/native/test_focal_hf_stability.cpp` builds synthetic DF Hamiltonians
and independently differences their variational energies. It checks full
search in a small rotation space, the common-angle RHF curvature near the
threshold, and a case with one alpha electron and 61 directions. On the
rotated 61-direction example the bounded search reported `+1.5 Eh` with
residual about `9.77e-16`, while the variational-energy difference found
a curvature of `-1 Eh`. The case fixes the Hamiltonian and verifies the
independent curvature, leaving room for a stronger search to find it.

The molecular tests in `tests/python/test_focal_uhf_reference_stability.py`
and `tests/python/test_focal_rhf_reference_stability.py` use matched PySCF
DF references. Same-basis energy, occupied-space overlap and signed
displacements are needed to qualify root selection. The small synthetic
and molecular controls establish no coverage for arbitrary large spaces.

## References

- `src/api/c_api_f12_focal_reference.cpp`
- `src/scf/solver/uhf_second_order.hpp`
- `tests/native/test_focal_hf_stability.cpp`
- [Orthonormal commutator evaluation](2026-11-27-focal-uhf-commutator-evaluation-order.md)
