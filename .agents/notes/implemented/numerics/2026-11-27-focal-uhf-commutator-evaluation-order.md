# Decision: Evaluate the focal UHF orthonormal commutator stop as M - M^T

Status: implemented
Date: 2026-11-27

## Problem

The opt-in focal DF-UHF stop (H2, `2026-11-26-focal-uhf-orthonormal-commutator.md`)
requires `max|X^T (F D S - S D F) X| <= density_tolerance` (`1e-11`) with
`X = S^-1/2`. It was evaluated by forming the AO commutator `R = F D S - S D F`
and then the congruence `X^T (R X)`. The two products in `R` cancel at the size of
`|F| |D| |S|`, and the congruence multiplies their rounding by `|X|^2 = 1/s_min`.
On diffuse bases with overlap eigenvalues near `1e-5` the evaluated value floors
above `1e-11` while the converged state's true value is about `1e-12`. Stationary
references were then refused with `canonical DF-HF did not converge` (C5H9O and
BrH2O at ha-Q; CH2Br on some cards), after the second-order refinement had
reached a maximum MO gradient of `1e-13`.

## Decision

Evaluate the same quantity in the antisymmetric order

    M = (X^T F) (D (S X)),    value = max|M - M^T|.

Because `F`, `D` and `S` are symmetric, `M - M^T = X^T F D S X - X^T S D F X` is
exactly `X^T (F D S - S D F) X`. No factor of `M` carries the cancellation, and
`S X = S^1/2` has norm `|S|^1/2`, so the rounding grows like `1/sqrt(s_min)`
instead of `1/s_min`. The tolerance stays `1e-11`; nothing is relaxed.

- `reference::orthonormal_commutator_max_abs` (`src/scf/reference/mean_field.cpp`)
  implements it; nonfinite input returns infinity, so the stop fails closed.
- `run_uhf_host_plan` uses it for both H2 checks (the CUDA-only metric fallback
  and the convergence acceptance), with the physical, not DIIS-extrapolated, spin
  Fock matrices and the evaluated state's spin densities. The `S X` product is
  computed once per solve and shared with the density-metric fallback.
- Every focal DF-UHF reference solves in `run_uhf_host_plan`, including those
  with CUDA Fock builds. The CUDA DF-UHF solvers (`src/scf/cuda/df_uhf_scf.cpp`
  and the host loops in `src/scf/rhf.cpp`) keep their own evaluation of the
  opt-in check; no production caller enables it there.

The restricted second-order polish uses the analogous orthonormal RMS,
measured on the same geometry and with the original AO residual checks.
The converged RHF primary continues to use its original evaluation order.

## Rejected alternatives

- **Floor-aware threshold.** A relaxation of the acceptance criterion; the
  ruling keeps `1e-11`.
- **Extended precision only at the gate.** It would also remove the floor, but
  needs non-BLAS long-double products of size `n^3` at every gate evaluation.
  The reordered FP64 value already has a margin of about 100 at `s_min = 1e-5`.
- **Dropping H2.** It would re-admit orbital gradients of order `1e-8` that leak
  into non-variational F12 energies, which is why H2 exists.

## Invariants

- Same quantity and tolerance as H2: both spins, `max|X^T (F D S - S D F) X|`
  `<= density_tolerance`, `X` the symmetric orthogonalizer used by the solve.
- Evaluated lazily, only with `enforce_orthonormal_uhf_commutator` and only
  after the cheaper criteria pass; solves without the flag are unchanged.
- A nonfinite value never passes the gate.

## Evidence

- Rounding of the gate value against long double, on PySCF lowest stable UHF
  states of CH2Br, C5H9O and BrH2O ha-Q (`s_min` 2.2e-5 to 8.0e-6): `1.2e-11` to
  `3.8e-11` in the AO-then-congruence order, `0.9e-13` to `1.8e-13` as
  `M - M^T`.
- Synthetic HBr+ (aug-cc-pVDZ plus a second Br p shell at 1.01 or 1.015 times
  the second most diffuse p exponent; `s_min` 4.3e-6 and 9.5e-6): `6.0e-11` and
  `3.0e-11` against `2.1e-13` and `1.2e-13` (alpha spin).
- `tests/native/test_focal_uhf_orthonormal_commutator.cpp`, section 5: on a
  synthetic bromine/hydrogen overlap with a duplicated diffuse p shell
  (`s_min` 8.2e-6), the value returned by `orthonormal_commutator_max_abs` agrees
  with an extended-precision evaluation of the definition to `4.9e-13` (bound
  `2e-12`), and nonfinite input fails closed. Synthetic core-Hamiltonian
  matrices do not reproduce the factor of about 100 between the two orders seen
  on SCF states (their diffuse-block elements are unscreened), so the test
  checks agreement with the reference rather than the old order's failure.
- A100, diagnostic build with only H2 reordered: C5H9O and BrH2O ha-Q converge
  to `-269.4513364174703` and `-491.5584863273785` Eh (PySCF lowest stable UHF
  `-269.45133641746895` and `-491.5584863273741`).
- An end-to-end CPU regression test on the HBr+ systems was tried and dropped:
  the `cpu_reference` backend does not use the stabilized metric density step,
  and from the lowest stable PySCF state the AO density change between DIIS
  cycles never falls below `1e-11` (tail minimum `4.7e-11` and `1.35e-11`). Those
  solves fail on the density criterion with either evaluation order.

## Consequences

- Two extra `n^3` products per spin at each gate evaluation, which happens only
  in the last iterations of a solve.
- The CUDA DF-UHF solvers and the host solve now evaluate the opt-in check in
  different orders. This matters only if a caller enables it on those solvers.

## Revisit when

- Overlap eigenvalues near `1e-9` reach focal references: the `M - M^T` rounding
  then approaches `1e-11`, and extended precision at the gate becomes necessary.
- A production caller enables the opt-in check on the CUDA DF-UHF solvers: port
  this evaluation order there first.
- The energy test (`1e-13` against the ulp of large `|E|`) is ruled on; it sits
  at the same kind of floor.
- Near-duplicate bases must converge on the `cpu_reference` backend: its AO
  density criterion floors above `1e-11` there, independently of H2.

## References

- `src/scf/reference/mean_field.cpp` (`orthonormal_commutator_max_abs`)
- `src/scf/solver/mean_field_driver.cpp` (`run_uhf_host_plan`)
- `tests/native/test_focal_uhf_orthonormal_commutator.cpp`
- `.agents/notes/implemented/numerics/2026-11-26-focal-uhf-orthonormal-commutator.md`
- [Orbital recovery and Hessian search](2026-11-27-focal-orbital-hessian-search.md)
