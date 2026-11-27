# Decision: Certify dense DF-UHF orbital adjoint solves by extra-precise refinement

Status: implemented
Date: 2026-11-27

The publication condition and workspace calculation were superseded by
[the verified residual and LU defect bound](2026-11-27-uhf-orbital-adjoint-forward-error.md).
The measurements below describe the original refinement implementation.

Supersedes in part
[2026-11-26-uhf-orbital-adjoint-amplification-admission.md](2026-11-26-uhf-orbital-adjoint-amplification-admission.md):
its residual risk and its invariant that admitted solves are bitwise identical to
the unguarded LU solve.

## Problem

`generativeqc_uhf_orbital_adjoint_cuda_v1` solves `H^T z = d` for the dense
DF-UHF orbital Hessian with cuSOLVER LU. The amplification gate
`rho = ||H||_1 ||z||_1 / ||d||_1 <= 2^26` measures how strongly the driver
excites near-null modes, but not the solver error. The LU backward error
`~eps ||H|| |z|` projects onto every soft mode with weight `1/lambda_min`, so a
driver orthogonal to an unprotected soft mode has `rho ~ 1` and still receives
an error along that mode. A force contraction whose right-hand side overlaps the
mode sees it. Examples from CPU emulation against exact rational solutions:

- 2x2 DF-UHF Hessian with `lambda_min = 1e-12`, driver along the stiff mode:
  `rho = 1.10`, success, relative error 8.3e-5 (2.5e-4 on a device).
- `x = y = 0.6`, `gap = fl(x^2)`, so `lambda_min = fl(x^2) - x^2 ~ 1e-17` is below
  the rounding of the entries; symmetric driver: `rho = 1.0`, success, relative
  error 0.44.

## Decision

After the LU solve the routine runs iterative refinement with extra-precise
residuals:

- `r = d - H^T z` is computed from the packed DF factors, not from the
  assembled FP64 Hessian. Every Hessian element is recomputed with exact
  products (FMA TwoProduct) and compensated sums (TwoSum), and is contracted
  with `z` in the same compensated arithmetic, so `r` is the residual of the
  exact Hessian of the given factors to about double-double accuracy.
- The correction `dz = (LU)^-T r` reuses the factorization. `z` is published
  once `|dz|_inf <= 2^-40 |z|_inf`. Each applied correction must shrink the
  next by at least one half, at most 8 corrections are applied, and otherwise
  the call fails closed with `ill-conditioned DF-UHF orbital Hessian: refined
  adjoint correction ...`. With a contraction factor below 1/2 the remaining
  error is at most `2 |dz|`, so a published `z` is within `2^-39 |z|_inf` of the
  exact solution.
- A first-step certificate leaves the LU solution bitwise unchanged.
- The amplification gate stays: it bounds sensitivity to the inputs.
  Refinement certifies the solve for given inputs, and above `rho = 2^26` the
  exact solution itself is not determined by rounding-level inputs.
- The residual reuses the LU workspace after `getrf`. The workspace is sized
  `max(lwork, rank)`, so the minimal budget changes only if cuSOLVER's `getrf`
  workspace were smaller than `rank` doubles.
- No outputs are published on any rejection.

## Rejected alternatives

- FERR-style forward error bounds (`|H^-T| |r|` estimates): they scale with
  `kappa(H) eps` and reject every OH-like system with a protected Goldstone
  mode (`kappa ~ 1e13`), although those solves refine to full accuracy.
- Refinement with FP64 residuals: the residual rounding `eps ||H|| |z|` is as
  large as the soft-mode error it should measure, so it improves the backward
  error but cannot certify the forward error.
- Projecting or deflating soft modes: needs a "soft" threshold or
  caller-supplied symmetry generators (ABI and Python changes), and the
  deflated response is a different response at near-symmetric geometries.
- Consumer-aware admission (bounding `g = rhs^T z` from the right-hand-side
  overlap with soft modes): covers only force-mode calls. The rotations-only
  consumers (`unrestricted_source_orbital_adjoint`, the focal DF-HF orbital
  response and the CABS singles relaxation) publish `z` itself.

## Invariants

- Residuals must come from the DF factors in error-free arithmetic. Residuals
  of the assembled Hessian cap the accuracy at `kappa(H) eps`.
- The TwoSum steps use the `__dadd_rn`/`__dsub_rn`/`__dmul_rn` intrinsics, which
  are never contracted into FMAs; plain `+`/`*` under `--fmad=true` breaks the
  error-free transformations.
- Published rotations are certified to `2^-39 |z|_inf` relative to the exact
  solution for the given factors, or the call fails without outputs.
- The amplification gate and its threshold are unchanged.

## Evidence

CPU emulation of the refined device solve: FP64 assembly, LAPACK
`getrf`/`getrs` transpose and a NumPy double-double residual. The oracle is the
exact rational Hessian of the FP64 factors, solved by 320-bit mpmath LU up to
rank 260 and by the P4 oracle above that. Errors are `|z - z_exact|_inf /
|z_exact|_inf`; "consumer" contracts with a right-hand side along the softest
mode.

| case | LU solve | refined | outcome |
|---|---|---|---|
| CH3, H2O cc-pVDZ, H2O cc-pVTZ (ranks 220, 190, 530) | 4.4e-16, 5.4e-16, 1.5e-15 | no correction, bitwise equal | admitted |
| OH cc-pVDZ, physical driver (Goldstone component removed) | 1.1e-6 (consumer 1.3e-5) | 1.4e-14 after 2 corrections (consumer 1.7e-13) | admitted, rho 16.7 |
| OH, 1e-2 Goldstone component | 1.1e-6 | 2.2e-16 after 3 corrections | rejected by rho 2.2e10 |
| 2x2 stiff driver, `lambda_min = 1e-12` | 8.3e-5 | 4.3e-15 after 3 corrections | admitted, rho 1.10 |
| 2x2 soft driver | 3.7e-4 | 2.0e-14 | rejected by rho 5.5e12 |
| 2x2 symmetric null mode, symmetric driver | 2.2e-5 | 4.0e-15 after 2 corrections | admitted, rho 1.0 |
| 2x2 `kappa ~ 1e15`, stiff or unit driver | | contraction 0.56 | rejected |
| 2x2 `lambda_min` below entry rounding | 0.44 | contraction 0.76 | rejected |

`tests/python/test_uhf_orbital_adjoint_admission.py` runs the public ABI on a
CUDA device: soft driver rejected without outputs; stiff driver and symmetric
null mode exact in rotations, orbital gradient and forces to `2^-38`;
entry-rounding-singular Hessian rejected without outputs; well-conditioned
solve exact.

### A100 device evidence

One A100 job ran three versions of `generativeqc_uhf_orbital_adjoint_cuda_v1`
on the same sealed inputs. The baseline used unguarded cuSOLVER LU. The
amplification-only version added the `rho` gate. The refined version used
that gate plus the compensated residual refinement described above.

- Sealed probe, 16 cases with exact oracles: the refined library publishes the
  eight expected admissions within 1.8e-15 of the exact solution and rejects
  the other eight without outputs. The amplification-only version publishes
  7 wrong solutions, with relative error up to 6.7 and 1.2e-6 for OH with the
  Goldstone component removed. The unguarded LU version publishes 12.
- The admission tests pass 5 of 5 with the refined library. The
  amplification-only version fails 3 (rank-deficient Hessian, stiff driver,
  symmetric null mode) and the unguarded LU version fails 4.
- Minimal device budgets are identical for the three libraries: H2O cc-pVDZ
  1,189,180 bytes, H2O cc-pVTZ 9,158,684, synthetic rank 3800 481,460,452. One
  byte less fails with the LU workspace budget detail.
- Wall time, minimum of repeats on synthetic factors:

| rank (auxiliary) | LU solve | refined |
|---|---|---|
| 950 (300) | 0.013 s | 0.016 s |
| 3800 (600) | 0.163 s | 0.271 s |
| 8550 (800) | 0.93 s | 2.34 s |
| 15200 (1000) | 3.30 s | 9.69 s |

- Force consumer oracle: the focal low-order and (T)+ forces for OH and H2O
  (ha-D-F12, frozen core) pass Richardson finite differences at 1e-8 Eh/bohr
  with both the refined and the base library. The OH dense calls (rank 355)
  carry Goldstone modes with `kappa` 7.4e13 and 2.1e14; the two libraries'
  rotations differ by up to 6.6e-5 in the infinity norm, while forces differ
  by at most 2.2e-14 Eh/bohr and energies by 2.5e-16 Eh.
- Production driver survey, 25 open-shell endpoints: all 33 dense calls come
  from `_f12_focal_singles_response.py:relaxed_reference_weights` without
  directions, and none is rejected. Seventeen calls in CF, CH, F, HF+, O, OF
  and OH carry symmetry soft modes. The largest is the F atom: rank 274,
  `kappa` 4.1e14, `|lambda| = 1.1e-13`. Refined and base endpoints differ by
  at most 1.7e-16 Eh and 1.6e-12 Eh/bohr. CH3O did not converge in DF-HF with
  either library and never reached the adjoint.

### Fail-closed margin for Goldstone modes

The OH cc-pVDZ Hessian above has one Goldstone mode with `lambda = 1.6e-12`
and `||H||_1 = 24.6`. Its eigenvalue was shifted to target values by adding
`(lambda_t - lambda) v v^T` to the double-double Hessian, and each shifted
system was solved with a driver whose Goldstone component is `1e-15` relative.
The CPU emulation was checked against 320-bit mpmath solves of the shifted
matrix.

| `lambda_t / ||H||_1` | corrections | contraction | refined error |
|---|---|---|---|
| 4e-15 (surveyed minimum, F atom: 2.2e-15) | 2 | < 5e-4 | 5.4e-15 |
| 8e-16 | 3 | < 5e-4 | 2.5e-16 |
| 2e-16 | 4 | 3e-3 | 2.0e-13 |
| 8e-17 | 7 | 0.02 | 3.4e-14 |

Every target certified, including a shift to zero, which leaves the eigenvalue
at the rounding level of the shift. The 2x2 cases fail at `kappa eps ~ 0.2`
because their soft eigenvalue is only about 25 times the rounding of the
matrix entries. In the OH Hessian, rounding perturbs the soft direction much
less.

## Consequences

- New fail-closed region: solves whose soft eigenvalue is not resolved by the
  FP64 rounding of the Hessian, where a correction exceeds half of the
  previous one or more than 8 corrections are needed. Such solves were previously
  published with an unbounded error whenever `rho` was small. Relative to
  `||H||_1`, the smallest surveyed production soft eigenvalue (F atom,
  2.2e-15) is 27 times the smallest certified OH target (8e-17). At the
  production level the OH emulation certifies after two corrections.
- Cost: each residual recomputes all `rank^2` Hessian elements in compensated
  arithmetic, about five times the FP64 assembly. Well-conditioned solves pay
  one residual and one extra `getrs`; soft-mode-limited solves pay one per
  correction. On A100 the refined solve costs 1.2 times the LU solve at rank
  950 and 2.9 times at rank 15200. Surveyed production ranks reach 1198, and
  the 33 calls together take 0.72 s refined against 0.63 s unrefined.

## Revisit when

- A production system fails in refinement. Then deflate caller-supplied
  symmetry generators rather than loosening the certificate.
- The residual becomes visible in endpoint timing. The Coulomb and exchange
  terms factor through the DF tensors (`O(q v o (v + o))` instead of
  `O(q v^2 o^2)`) if the intermediates are carried in double-double form.
- The dense solver changes from LU, or the Hessian assembly changes; the
  residual must follow `uhf::element` exactly.

## References

- [Driver amplification admission](2026-11-26-uhf-orbital-adjoint-amplification-admission.md)
  documents the stiff-driver failure with the `rho` gate alone.
- `src/posthf/uhf_orbital_adjoint_cuda.cu` implements the dense solve and
  compensated residual. `tests/python/test_uhf_orbital_adjoint_admission.py`
  exercises the public ABI with exact rational references, including a
  stiff driver and a symmetry-protected near-null mode.
