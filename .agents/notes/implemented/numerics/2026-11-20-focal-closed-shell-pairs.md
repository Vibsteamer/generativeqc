# Closed-shell orbital ownership in the focal pair term

The all-active, singlet RHF pair request starts from the canonical DF-RHF
solver. Its result has one spatial coefficient matrix, one Fock spectrum and
a density with two electrons per occupied spatial orbital. The unrestricted
pair algebra can represent this determinant exactly by placing the shared
spatial orbitals on both physical spin lines. The raw JK integrals are used
with one occupied electron per line. The retained αα, αβ and ββ classes
therefore use the same pair counting as the open-shell algebra.

Both derivatives of a shared coefficient matrix contribute to one RHF
coefficient cotangent. Each simultaneous spin rotation receives half of that
sum in the existing two-spin adjoint. The singlet solution has identical alpha
and beta rotations. We project onto that subspace and check the residual of
the full DF-HF response before applying the nuclear source transpose. The
action on a shared rotation contains the usual closed-shell Coulomb factor
four and both exchange contractions. A PySCF restricted water response,
formed by applying its DF Fock response to every spatial rotation and solving
for three electric-field directions, independently verifies this identity.

The full-basis molecular controls use cc-pVDZ-F12, its parent OptRI,
aug-cc-pVTZ-JKFit, aug-cc-pVTZ-RI and beta 0.9 per bohr. The independent
calculation solves PySCF DF-RHF and DF-RMP2, generates two- and three-center
operators with Psi4 Libint and uses spatial singlet and triplet pair
functionals. Each off-diagonal occupied spatial pair contributes one singlet
and three triplet states. A diagonal occupied pair contributes one singlet.
The independent force is obtained by changing every nuclear coordinate by
both signs at 0.0004 and 0.0002 bohr. H2 tests the single opposite-spin pair.
Bent H2O tests 45 occupied spin pairs including nonzero same-spin energy.
The observable contains ordinary and explicit 3C(FIX) pair correlation. The
reference and CABS singles remain separate focal components.

The scientific input, signed energy records and per-component nuclear forces
are retained in the validation workspace under
`focal-convergence-20261118/public_rhf_f12_pair`. This comparison used the
CPU reference backend. Independent qualification of the CUDA path requires
the native CUDA solver, the shell library and a physical device.
