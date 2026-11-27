# Decision: Unpreconditioned DF-HF orbital response solves stop on the driver residual

Status: implemented
Date: 2026-11-26

## Problem

The host (CPU) branch of `relaxed_reference_weights` solves the coupled DF-HF
occupied/virtual equations with unpreconditioned SciPy MINRES at `rtol=1e-11`
and accepted any result whose residual was below `1e-9 max(1, ||driver||)`.
SciPy does not stop on `||r|| <= rtol ||b||`: its tests are backward errors
such as `||r|| <= rtol ||T_k||_F ||x||` with the Lanczos matrix `T_k`. On the
all-active HBr ECP pair-response gate (rank 286, eigenvalues 0.30 to 22.4)
both solves stopped after 37 iterations with true relative residuals 2.6e-10
and 1.4e-10 and forward errors 3.4e-11 and 1.9e-11 against a dense LU solve.
Two pipelines that should agree then differed by 1.17e-10 in the relaxed JKFit
AO Coulomb weights, above the unchanged 1e-10 gate.

## Decision

`_minres_to_residual` runs SciPy's unpreconditioned MINRES recurrences and
stops when the recurrence residual reaches `1e-11 ||driver||`.
`_coupled_response_minres` checks the true residual, runs at most one
continuation with the same absolute target, and raises when the target is not
met. Preconditioned matrix-free solves keep their SciPy `1e-12` criterion.

The target is absolute in the original driver. Orbital Hessians can have
symmetry null modes: the UHF OH gate (rank 85) has `|lambda| = 4.7e-13` next to
a spectrum starting at 0.168, and the driver's component along that mode is
rounding noise (1e-15). A residual target far above that noise never needs the
mode, so its amplitude in `z` stays at the SciPy single-pass level (2.9e-14
against 2.6e-14).

## Rejected alternatives

- Raise the parity tolerance: hides solver truncation in relaxed weights.
- Lower `rtol` only: still a backward-error test whose ratio to the relative
  residual depends on `||T_k||_F ||x|| / ||b||`; the declared tolerance would
  remain unverified.
- Restart SciPy MINRES on the true residual with a relative tolerance: each
  correction solve is judged against its own tiny `||x||`, so it over-solves to
  relative residuals near 1e-14 and resolves the rounding-level null-mode
  component. On UHF OH this put -3.7e-4 and -2.7e-4 along the null mode in two
  pipelines that should agree, and their overlap weights differed by 5.7e-6
  against the 1e-10 gate that the unrefined solve passed.
- Dense direct host solve: `O(r^2)` memory and `O(r^3)` work at QZ ranks, and it
  resolves null modes from rounding as well.

## Invariants

- A host response result has relative true residual at most `1e-11`, or the
  call fails.
- The stop never requires resolving a driver component smaller than
  `1e-11 ||driver||`, so a rounding-level component along a null mode is not
  amplified towards `|d_i| / |lambda_i|`.
- The matrix-free solver criterion and work are unchanged.

## Evidence

- `tests/python/test_f12_orbital_response_true_residual.py`: on an SPD operator
  with the HBr spectral range, plain SciPy MINRES returns relative residual
  1.3e-9 and the residual stop reaches at most 1e-11; with the UHF OH spectrum
  and a 1.2e-15 null-mode driver component the null amplitude stays at 9.7e-14,
  while the rejected restart put 1.1e-6 there; a noisy operator fails closed; the
  matrix-free result is bitwise unchanged.
- MP2-F12 pair-response gates, maximum pipeline differences with the same
  native libraries, before / after: HBr all-active JKFit AO Coulomb 1.173e-10 /
  3.8e-13 and relaxed hcore 1.5e-11 / 4.9e-14; UHF OH all-active overlap
  5.7e-12 / 1.4e-12 and JKFit AO Coulomb 1.0e-12 / 2.7e-13.

## Consequences

Across the 22 host solves of the HBr, OH and Br-atom pair-response parity gates,
the residual stop uses 1 to 7 more Hessian products than SciPy's backward-error
stop, counting the true-residual check: HBr 37-38 to 39-40, UHF OH 35-37 to
40-42, Br atom 10-24 to 11-27. Their relative residuals are 6.2e-14 to 8.7e-12;
the SciPy stop left 14 of the 22 above the nominal 1e-11, up to 4.4e-10 (HBr
FC4). Preconditioned matrix-free branches can still return true residuals above
their nominal `rtol` because the SciPy criterion applies there.

## Revisit when

Matrix-free (CUDA, large-rank) response parity needs the same guarantee; give
the preconditioned path the same absolute stop, with GPU timing and parity
evidence.
