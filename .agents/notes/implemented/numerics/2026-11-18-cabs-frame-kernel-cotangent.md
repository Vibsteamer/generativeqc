# Decision: transfer CABS coefficient weights through an invariant kernel

Status: implemented (component algebra; molecular force integration pending)
Date: 2026-11-18

## Problem

The molecular AO-to-MO transpose supplies a weight B on CABS coefficients C.
The native spectral response instead differentiates K = C C^T with respect to
AO overlap S. Differentiating individual eigenvectors is gauge-dependent and
ill-defined inside a degenerate retained subspace, even when the observable is
smooth. Freezing the CABS coefficients drops a nuclear response term.

## Decision and derivation

For C^T S C = I set D = S C, and M = C^T B. An invariant scalar requires M to
be symmetric: the derivative along C Omega vanishes for every skew Omega.
Choose

    bar_K = (B D^T + D B^T - D M D^T) / 2.

Then bar_K is symmetric and 2 bar_K C = B. Consequently
<bar_K, d(C C^T)> = <B, dC> for any first-order frame motion. This weight is
not unique outside the tangent space of fixed-rank kernels; that freedom does
not change the derivative. A simultaneous orthogonal rotation of C and B must
leave bar_K unchanged.

The TensorIR graph uses only binary contractions and exports the metric-dual,
gauge-skew and reconstruction residuals. Its caller must check all three before
using the symmetric representative. The graph's symmetrization is not permission
to discard a nonzero gauge-dependent contribution. D is evaluated at the central
point: do not differentiate these adjoint weights again while assembling a
first derivative. This component does not implement a second derivative.

The overlap provider owns the derivative of K(S), including rotations across the
retained/discarded boundary. Its existing rank and spectral-gap guards remain.
The molecular caller owns other AO, SCF orbital and integral responses; this
transfer supplies only the complementary-frame contribution.

## Ordering and scope

The frame, overlap, coefficient weight and kernel weight must use one AO order.
The native CABS projector provider currently requires OBS rows first, followed
by complementary auxiliary functions. The AO export from an external program
may interleave shells: explicitly permute it before that provider, and map its
returned overlap weight back. The generic transfer does not impose that order
or silently choose a permutation.

F12 correlation, CABS singles and any other terms contribute their own weights.
The molecular fixtures here supply only the F12 correlation weight. Passing this
component is not full MP2-F12 or CCSD-F12b molecular analytic-force acceptance.
The source-level H2O energy-chain discrepancy recorded in the AO/frame note is
still unresolved and its original gate remains in force.

## Validation

Permanent tests use an independent nonlinear scalar of K, nonorthogonal AO
metrics, ranks 1/3/5, arbitrary frame displacements, two difference steps and
random column rotations. Invalid ranks and observable gauge dependence are
explicitly exposed. Molecular H2/H2O coefficient fixtures retain the original
AO/source geometry and protocol from `native-f12-ao-response-20261118`.

The separate L20 qualification composes this generated CUDA transfer with the
existing native CUDA projector/overlap response. Its reference uses an
independent Cholesky/QR/SVD construction. It includes full and truncated ranks,
a doubly degenerate retained eigenvalue, and two overlap-difference steps.
The fixture-to-orbital response is tested as a component; no reference-library
calculation is substituted into a production endpoint.

## Next

Integrate the overlap weight with native overlap nuclear derivatives and the
other AO integral weights. Obtain the forward CABS frame from the native
provider, preserve its retained subspace and rank, and add the SCF orbital
response before validating every Cartesian molecular force component.
