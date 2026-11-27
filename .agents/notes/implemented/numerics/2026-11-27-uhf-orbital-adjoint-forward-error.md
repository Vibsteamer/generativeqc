# Verifying forward error in dense DF-UHF orbital adjoints

Status: implemented
Date: 2026-11-27

Supersedes the publication rule in
[the original refinement note](2026-11-27-uhf-orbital-adjoint-refinement-certificate.md).

## Problem

The original threshold test could admit a correction before checking whether it
had halved. Even a probe of the correction's successor can miss a small
correction in a slow mode. For example, in units of 2^-40, an error of 7.5
along a mode with contraction rate 15/16 produces a correction of only 0.469.
A faster mode can make the largest measured correction 0.938 and its successor
0.440. Both scalar tests pass while the forward error exceeds the target.

## Decision

Check the observed halving requirement before the correction threshold at
every noninitial step. When the correction `dz` satisfies
`||dz||_inf <= 2^-40 ||z||_inf`, hold the current candidate `z` unchanged
and bound its error.

Let `H` be the exact Hessian of the packed FP64 inputs, `B = P^T L U` the
matrix of the returned pivoted LU factors, and `s = driver - H^T (z + dz)`
the residual of a hypothetical update. Compensated sums with directed
rounding enclose `s` without rounding the addition `z + dz` to FP64.
The same kernel bounds the column sums of `|P H - L U|`. Triangular
comparison systems with positive diagonal and negative off-diagonal
magnitudes bound both `t = ||B^-T s||_inf` and
`m = ||I - B^-T H^T||_inf`. The final permutation preserves these norms.
For an upper bound `m < 1` the Neumann series gives
`||z_exact - z||_inf <= ||dz||_inf + t / (1 - m)`.
Publish only if the outward-rounded bound is at most `2^-39 ||z||_inf`.

The first pass evaluates the product of the returned factors directly using
at most `n` round-to-nearest FP64 FMAs per entry. It encloses the rounding
error of those dot products by `2 n u |L| |U|`, where `u = 2^-53` and
`2 n u` overbounds `gamma_n` in the supported rank range. An additive
underflow allowance of `n^2 DBL_MIN` covers each column. If the resulting
bound is inconclusive, the second pass encloses each dot product with
compensated directed arithmetic. Both passes measure the returned factors
without a backward-error assumption about the factorizer.

## Evidence and costs

The public ABI test uses exact rational references. It includes slow
contraction just below the correction threshold, a mixed-rate example
where one probe would admit an inaccurate answer, an exact unexcited
mode and a shrinking sequence that stays within the forward-error bound.

The first verification pass adds one Hessian contraction with two terms
per rotation, up to `n(n+1)(2n+1)/6` FP64 triangular factor product terms,
and rank-squared triangular comparison work. The second pass repeats
these operations with compensated factor product terms. Both passes
reuse the existing factorization, with scratch sized
`max(lwork, 4 * rank)` doubles. At ordinary cuSOLVER workspace sizes the
rank-sized vectors fit in the factorization workspace. A smaller allocation
must fit the existing caller-supplied budget to proceed.

## References

- [NVIDIA cuSOLVER legacy LU contract](https://docs.nvidia.com/cuda/archive/12.6.0/cusolver/index.html)
