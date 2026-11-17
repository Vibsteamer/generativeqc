# Fixed-frame CABS singles and restricted 3C(FIX) occupied-pair response

Status: implemented at the intermediate-tensor boundary
Date: 2026-11-17

## CABS singles

Write the Fock matrix in the orthonormal occupied and combined orbital-virtual/complementary spaces as blocks A, C and B, where C is the occupied to virtual block. Solve A T − T B = C. The correction is w times the Frobenius inner product of C and T. Use w = 2 for one restricted spatial channel or w = 1 for one unrestricted spin channel. The nuclear dependence of both the Fock matrix and the basis of these blocks enters upstream.

The transpose maps a unit energy weight to −w TTᵀ in the occupied block, +w TᵀT in the virtual block and wT in the upper off-diagonal block of a full symmetric Fock cotangent. With symmetric perturbations both off-diagonal copies contribute. The CUDA provider diagonalizes each small block on device, rotates C using cuBLAS, divides by occupied-minus-virtual differences there, and rotates the solution back. This gives an invariant differential under rotations inside either subspace even when an eigenvalue repeats inside a block. Check for a closing interblock gap. The CABS energy has no geminal exponent.

## Restricted 3C(FIX) pairs

Keep the denominator D_ijab fixed to the external occupied pair (i,j) in both linear and quadratic contractions with the virtual coupling C. The historical reference tensor archive includes a water scalar reduced by an exploratory implementation. Summing D over independent occupied indices in the quadratic contraction reproduced its scalar but mixed distinct occupied-pair energies. A system with just one occupied pair cannot detect this error. Replaying the exact same water input tensors through Psi4's own method gave −0.08067307116871358 Eh; the exploratory scalar was −0.11005001488350676 Eh. This difference belongs to the 3C(FIX) correlation increment on those tensors.

The compiler now emits separate spatial tiles for i = j and i < j, accounts for both direct and exchanged couplings in the latter, and obtains their intermediate cotangents by reversing the same TensorIR program. The result is composable with upstream integral, orbital and CABS derivatives. An E/F endpoint also needs the geometry response of those inputs. The component tests exercise the fixed intermediate-frame derivative and independent occupied pairs.

## Tests and reconstruction

An allocated NVIDIA L20 with an FP64 CUDA build reproduced the four independent CABS energies within 6.01×10⁻¹⁴ Eh. For the water case, the standard binding to the complete compiled library returned −0.003249135051223353 Eh and a symmetric Fock adjoint of dimension 158. In a separate warm component measurement, the median time for energy and its Fock adjoint was 5.88 ms. Across the four restricted pair cases the largest energy discrepancy with the Psi4 replay was 1.39×10⁻¹⁷ Eh. The water case resolves all 15 independent occupied pairs.

Four small data records contain both molecular inputs to the reductions and independent Psi4 results. H₂O exercises five diagonal and ten off-diagonal occupied pairs. The independent oracle evaluates all pair energies and the response to a common perturbation direction at step sizes 3×10⁻⁴ and 10⁻⁴. The CABS tests additionally use a direct Kronecker-system solve, symmetric perturbations of all entries, degenerate blocks, spin multiplicities and numerical failure cases. Component timing covers input-tensor to result time. A complete single-structure geometry-to-energy/force benchmark remains a separate qualification. [Fixture provenance and commands](../../../../tests/reference_data/f12/README.md) · [Psi4 implementation](https://github.com/psi4/psi4/blob/v1.11/psi4/src/psi4/f12/mp2.cc)
