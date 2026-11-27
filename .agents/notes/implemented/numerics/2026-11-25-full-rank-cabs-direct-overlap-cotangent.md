# Decision: Direct full-rank CABS overlap cotangent with compensated BLAS

Status: implemented
Date: 2026-11-25

## Problem

The complementary frame `C` satisfies `C.T @ S @ C = I`. Its gauge
depends on the chosen basis inside the retained subspace. Pair and Fock
energies use an invariant frame kernel `K = C @ C.T`. Directly
differentiating individual eigenvectors can be singular at an internal
degeneracy. For the kernel adjoint, let `D = S @ C` and
`M = sym(C.T @ G)`, with `G` the frame cotangent. The symmetric
representative `(G @ D.T + D @ G.T - D @ M @ D.T) / 2` reproduces frame
work when the gauge-skew and reconstruction checks pass.

On a diffuse full-rank complement, rounding this intermediate before
applying two kernel factors amplifies errors by `||K||^2`. Near
linearly dependent bases also make a plain FP64 Gram check look worse
than the actual orthonormality residual.

## Decision

For full-rank CABS complements (`rank == atomic - obs`), substitute $K = C C^T$
algebraically before forming $\bar{S}$ so the intermediate $\bar{K}$ and the
$K (\cdot) K$ amplification never occur:

$$\bar{S} = -\tfrac{1}{2}\big(O G C^T + C G^T O\big) - \tfrac{1}{2}\, C\,\mathrm{sym}(C^T G)\, C^T$$

with $O = \mathrm{embed}(S_{\text{obs}}^{-1})$. Shipped in
`python/generativeqc/_f12_focal_projector.py`, the implementation uses a three-tier precision hierarchy:

1. **FP64 fast path**: `reference_cabs_singles` (`_f12_focal_singles.py`) and
   `compensated_cabs_frame_cotangent_checks` (`_f12_focal_projector.py`) check
   orthonormality (`C^T S C`, `(S C)[:obs]`), `gauge_skew`, `dual_gram`, and
   `reconstruction_residual` in plain FP64 first against the strict `2e-7` gate,
   incurring zero extra work on ordinary well-conditioned bases.
2. **Ozaki-4 compensated double-double BLAS (`_ozaki_mm`, `_ozaki_mm_left_dd`)**:
   Whenever the FP64 fast path fails on ill-conditioned diffuse bases, or when
   computing the direct full-rank overlap cotangent, `_ozaki_mm` splits operands
   into exponent-bucketed FP64 slices (`slices=4` initial) and accumulates
   BLAS `dgemm` products in double-double `(hi, lo)` via error-free `_two_sum`
   and `_dd_add`. After the initial diagonal accumulation, `_ozaki_mm` checks an
   explicit remainder bound against `1e-15 * |result|` and dynamically extends
   slices up to `max_slices=16` whenever the unsliced tail is non-negligible.
   `compensated_cabs_orthonormality` computes `(C^T S C, (S C)[:obs])`,
   `compensated_cabs_frame_cotangent_checks` re-verifies any failing check
   against the same `2e-7` tolerance, and `full_rank_cabs_overlap_bar` evaluates
   $O G_{:\text{obs}}$ via Cholesky (`cho_factor`/`cho_solve`) and forms
   $\bar{S}$ in double-double.
3. **Exact rational fallback (`_exact_mm_dd`)**: If dynamic slice extension hits
   `max_slices` with deep scale cancellation still exceeding `1e-15 * |result|`,
   or if intermediate slice scaling would overflow FP64 exponents, `_ozaki_mm`
   escalates automatically to arbitrary-precision exact `fractions.Fraction`
   accumulation (`_exact_mm_dd`) rounded once to double-double `(hi, lo)`.

Wire `reference_cabs_singles` and `_relaxed_weights_aot` (`_f12_focal_singles.py`),
`relaxed_weights` (`_f12_focal_singles_response.py`), and
`relaxed_f12_source_weights` (`_f12_focal_increment.py`) to use
`compensated_cabs_orthonormality` / `compensated_cabs_frame_cotangent_checks`
and to dispatch to `full_rank_cabs_overlap_bar` when `rank == atomic - obs`.
Retain the existing `frame_kernel_cotangent` + `focal_cabs_projector(cotangent=cabs_kernel)`
path for rank-deficient CABS (`rank < atomic - obs`, where discarded-subspace
rotations contribute via the matrix-function VJP) and for unrelaxed
`reference_cabs_singles(..., relax=False)` inspection callers.

## Invariants

- Never relax the `2e-7` orthonormality/gauge/reconstruction gate to absorb FP64
  cancellation; re-verify the same tolerance in compensated precision.
- Retain the fixed-rank projector compatibility check and the rank-deficient
  (`rank < atomic - obs`) Fréchet derivative route unchanged.
- Escalate from Ozaki-4 to dynamic slice extension and `_exact_mm_dd` rather
  than silently truncating a non-negligible unsliced remainder.

## Evidence

In an earlier heavy-augmented QZ control, FP64 frame orthonormality error
of 2.94e-7 dropped to 1.05e-11 when recomputed with compensated products.
The indirect full-rank path produced signed force differences of
3.4e-8 to 6.1e-8 Eh/bohr on the sampled coordinates despite small net
force. This motivated the direct formula.

`tests/python/test_f12_frame_response.py` reproduces the full-rank
issue with an independent linear-solve oracle and exercises the
rank-truncated overlap response. `tests/python/test_audit_fix3_py_ffi_gate1.py`
and `tests/python/test_f12_patch_a2b_gate1.py` check compensated
multiplication and its exact fallback on controlled inputs.

## Rejected alternatives

- **Relaxing the `2e-7` `dual_gram` guard**: rejected because it masks real
  non-orthonormal frame bugs and does not fix the `3.4e-8` to `6.1e-8 Eh/bohr`
  force error from $-K \bar{K} K$.
- **Computing $\bar{K}$ in extended precision while still rounding $\bar{K}$ to
  FP64 before calling `focal_cabs_projector`**: rejected because the FP64
  rounding of $\bar{K}$ itself is amplified by $\|K\|^2 \sim 10^{16}$ inside
  $-K \bar{K} K$.
- **Using `np.longdouble` matmul instead of Ozaki BLAS**: rejected because
  the available precision is platform-dependent and the unaccelerated
  matrix multiply adds substantial work.

## References

- `python/generativeqc/_f12_focal_projector.py`
- `python/generativeqc/_f12_focal_singles.py`
- `python/generativeqc/_f12_focal_singles_response.py`
- `python/generativeqc/_f12_focal_increment.py`
