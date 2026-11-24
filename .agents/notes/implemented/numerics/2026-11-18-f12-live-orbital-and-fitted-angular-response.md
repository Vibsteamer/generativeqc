# Molecular F12 response on the live DF-RHF reference

The molecular graph evaluates ordinary and complementary orbitals in a combined
orbital and OptRI AO basis. The DF-RHF solve owns the occupied and virtual
orbitals in its smaller basis. Map the actual live orbitals into the combined
basis by shell and public spherical-function order. Verify the overlap and
core Hamiltonian on that subspace before using the orbitals. The overlap and
core check together catches duplicated nuclear charges as well as permutations.

Use the same reference owner to consume the coefficient adjoint after the
complementary frame has been differentiated. The symmetric same-space
canonical multipliers belong to the Fock seed. The extended Fock has already
been differentiated through its AO construction. The native response supplies
the HF force including nuclear repulsion and the physical orbital/Pulay term.
Subtract the gradients from explicit source adjoints exactly once.

The Coulomb Gaussian angular polynomials can also propagate the four fitted
Slater operators. Their coefficients act on the radial moments
`(-d/dT)^n M0` in the same primitive prefactor convention. Expose the
existing compensated fitted-moment sum directly and share the angular and
center derivative equations. Signal the operator family in generated C++
namespaces and include guards. Retain the distinct orbital and auxiliary
angular limits and FP64 arithmetic.

The closed-shell test panel comprises nonsymmetric H2 and H2O in cc-pVDZ-F12,
cc-pVDZ-F12-OptRI and aug-cc-pVTZ-RI. Audit all five three-center contractions
and their five auxiliary metric contractions. Preserve atomic ordering, pure
angular conventions, both halves of the Frobenius contractions and signs for
physical forces. An independent ordinary DF-MP2 correlation derivative is
added once when comparing the complete DF-MP2-F12 energy and forces.

Run `python -m benchmarks.f12_df_integrals --polynomial` with the
Libint primitive fixture to qualify the bounded angular route through g,g,h.
Specify `--arch sm_89` on a physical L20. The separate
`benchmarks.f12_molecular_reference_forces` harness evaluates the true
molecular cotangents and the native reference response at one shared geometry.
It reports the outstanding two-/three-center sources explicitly.

The independent stock water numerical force changes appreciably with the
finite-difference stencil. Preserve both step estimates as well as the
analytic force comparison. A self-consistent directional derivative checks
the derivative of the implemented scalar. An independent energy, all-atom
force and physical device execution each require their own evidence.

In the offline qualification, the 108 Libint primitive cases reached a
maximum scaled value or gradient difference of 9.67e-15. The corresponding
real-molecule host test used all physical shell pairs and triples, including
every radial contraction and public pure function. For H2 and water the
largest error in any of the ten weighted, atom-resolved source gradients was
1.78e-13 hartree/bohr. This is a test of the AO sources on the host.

At the water geometry the complete molecular energy differed from stock Psi4
by 3.18e-10 hartree. Replacing the symmetric metric whitener by a Cholesky
factor of the same metric changed the F12 increment by only 1.74e-13 hartree.
Substitution of the independent implementation's explicit metric inverse
produced a much larger shift. This arithmetic sensitivity warrants distinct
checks of the fitting equations and the unchanged stock energy threshold.
