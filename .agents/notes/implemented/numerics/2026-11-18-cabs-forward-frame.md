# Decision: expose the native GPU CABS frame with its invariant kernel

Status: implemented; physical L20 qualification tracked separately
Date: 2026-11-18

## Problem

The overlap provider returned the kernel K = C C^T and its overlap adjoint,
but the molecular F12 forward integral maps need C itself. Supplying a frame
from an independent CPU oracle is suitable for component qualification, not
for a native GPU molecular endpoint.

## Decision

After diagonalizing the projected auxiliary Gram matrix T, form
C = J U_keep diag(lambda_keep^(-1/2)) using the existing CUDA eigensolver and
cuBLAS stream. J removes the entire OBS component. Reuse the existing square
contraction scratch; the new frame adds no persistent device allocation.
Pack a rectangular row-major frame on the device before publishing it with K,
the optional overlap adjoint, retained rank and cutoff margin.

The private C ABI moves to v2 with an explicit frame capacity. Python and
repository benchmarks migrate together; no compatibility alias is retained.
Immutable already-submitted source snapshots remain on their original ABI.
Output arrays are published only after all numerical checks and copies succeed.

## Invariants

- C^T S C = I; S C has zero OBS rows; C C^T = K.
- OBS-prefix AO ordering is required by this provider. External combined
  bases must be explicitly permuted and their cotangents mapped back.
- The retained rank and cutoff guards are unchanged. Eigenvector signs and
  rotations inside a degenerate retained space are not physical observables.
- Nuclear response uses the invariant kernel cotangent described in
  [the frame-response note](2026-11-18-cabs-frame-kernel-cotangent.md), never
  a derivative of individually chosen eigenvectors.
- No CPU eigensolve, reference library or gauge alignment is used in the
  production forward provider.

## Evidence and scope

The rectangular ctypes boundary is exercised without CUDA. Device tests cover
retained degeneracy, truncated rank, exact linear dependence, scaling, invalid
rank and insufficient output capacity with unchanged output buffers.
The molecular benchmark checks the invariants against independent supplied
H2/H2O overlaps and keeps the original two-step physical probe derivative gates.
The composed-response benchmark now obtains its forward frame from CUDA and
checks the nonlinear observable against a Cholesky/QR/SVD reference at both
displacement steps. These are component qualifications, not complete analytic
molecular forces; full integral/SCF assembly and the water finite-difference
reference sensitivity remain separate work.

## Revisit when

A resident molecular endpoint can retain the frame on device across AO maps;
the current internal boundary returns host arrays like the existing K/VJP API.
