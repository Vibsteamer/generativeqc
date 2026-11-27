# Decision: Fail closed on amplified dense DF-UHF orbital adjoint solves

Status: implemented
Date: 2026-11-26

Superseded in part by
[2026-11-27-uhf-orbital-adjoint-refinement-certificate.md](2026-11-27-uhf-orbital-adjoint-refinement-certificate.md):
extra-precise refinement now certifies the solve, which closes the residual risk
below; solves that need corrections are no longer bitwise identical to the LU
solve.

## Problem

`generativeqc_uhf_orbital_adjoint_cuda_v1` assembles the dense DF-UHF orbital
Hessian `H = A + B` and solves `H^T z = d` with cuSOLVER LU. It rejected only
exact zero pivots and nonfinite results, so a nearly singular Hessian returned
status success with an inaccurate adjoint. A generic 2x2 DF-UHF Hessian
(`lambda_min = 3.8e-13`) driven along its soft mode gave a successful solve with
relative error 3.7e-4 in CPU emulation and 1.9e-4 on a device, against the exact
rational solution.

Open-shell UHF Hessians also carry symmetry-protected near-null modes: OH
cc-pVDZ has a Goldstone-like eigenvalue of 1.6e-12. A physical driver has no
component along that mode by symmetry, so its solve is accurate in the physical
subspace and must stay admitted, even though `kappa(H) ~ 1e13`.

## Decision

After the solve the routine forms `rho = ||H||_1 ||z||_1 / ||d||_1` and fails
when `rho > DBL_EPSILON^-1/2 = 2^26`. Since `z = H^-T d`,
`rho <= ||H||_1 ||H^-1||_inf`, which is `kappa_1(H)` for the symmetric Hessian,
so a rejection certifies ill-conditioning; `rho` itself measures the
amplification that this driver actually experiences. At the threshold the
expected relative error `rho u = 2^-27 ~ 7.5e-9` approaches the force accuracy
of the consuming pipelines. `||H||_1` is reduced on the device into the LU
scratch before factorization, so device memory and the minimal budget are
unchanged. A rejected call writes no rotations, gradients or forces.

## Rejected alternatives

- Admission on an estimate of `kappa(H)`: rejects every OH-like open-shell
  system with a protected null mode although their physical solves are exact.
- Detect soft modes by inverse iteration and test the driver overlap: needs
  heuristic thresholds for "soft" and "overlap" and falsely rejects imperfectly
  converged Goldstone modes.
- Regularized or deflated solves: change the response definition; they are a
  method change, not a guard.
- Warnings: the policy is fail-closed.

## Invariants

- Admitted solves are bitwise identical to the unguarded solve.
- The device budget contract is unchanged.
- No output is published on rejection.

## Evidence

CPU emulation of the device solve (FP64 assembly, LAPACK `getrf/getrs`
transpose) on DF-UHF factors from PySCF and on exact synthetic factors:

| case | rho | admitted | relative error |
|---|---|---|---|
| CH3 cc-pVDZ, random driver | 11.1 | yes | 3.5e-16 |
| H2O cc-pVDZ, random driver | 13.5 | yes | 2.8e-16 |
| H2O cc-pVTZ (rank 530), random driver | 13.5 | yes | 1.5e-15 |
| OH cc-pVDZ, Goldstone component removed | 16.7 | yes | 6.8e-7 along the null mode, 8e-17 projected |
| OH, 1e-2 Goldstone component | 2.2e10 | no | 1.1e-6 |
| synthetic soft driver | 5.5e12 | no | 3.7e-4 |
| synthetic stiff driver | 1.10 | yes | 8.3e-5 along the soft mode, 1.4e-16 projected |

Device runs on an NVIDIA L20 (sm_89, CUDA 12.9 runtime libraries) compared this
translation unit with its parent revision on the same factors and drivers.
Errors are relative to the exact solution; "projected" excludes the protected
null mode.

| case | rank | parent revision | this revision |
|---|---|---|---|
| CH3 cc-pVDZ, random driver | 220 | success, 8.8e-16 | success, bitwise equal |
| H2O cc-pVDZ, random driver | 190 | success, 6.6e-16 | success, bitwise equal |
| H2O cc-pVTZ, random driver | 530 | success, 1.6e-15 | success, bitwise equal |
| OH cc-pVDZ, Goldstone component removed | 130 | success, 8.0e-16 projected | success, bitwise equal |
| synthetic stiff driver | 2 | success, 0 projected | success, bitwise equal |
| OH, 1e-2 Goldstone component | 130 | success, 1.2e-4 | rejected at rho 2.2e10, outputs untouched |
| synthetic soft driver | 2 | success, 1.9e-4 | rejected at rho 5.5e12, outputs untouched |

The H2O cc-pVDZ solve needs 1,189,180 bytes of device budget in both revisions,
and one byte less fails with the budget error in both. The norm reduction does
not change solve time measurably (rank 530: 6.05 ms parent, 5.86 ms this
revision).

`tests/python/test_uhf_orbital_adjoint_admission.py` exercises the public ABI
on a CUDA device: soft driver rejected without outputs, protected null mode
admitted with an exact physical component, well-conditioned solve exact.

## Consequences

The residual risk is a driver orthogonal to an unprotected soft mode: `rho` is
small and the solve is admitted, while its error lies along that soft mode (the
synthetic stiff case). The error matters only to consumers whose right-hand
sides overlap the soft mode.

## Revisit when

- A production system shows `rho` within three orders of magnitude of `2^26`.
- Consumers need certification of soft-mode-confined errors; that requires
  consumer-aware admission or a deflated solve.
- The dense LU solver changes; the bound must be re-derived.
