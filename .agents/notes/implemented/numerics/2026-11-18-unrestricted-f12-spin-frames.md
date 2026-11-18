# Unrestricted F12 spin frames and pair projectors

A semicanonical UHF reference gives distinct alpha and beta rotations. Occupied traces from both spins can only be added in the common fitting basis. Each spin Fock uses this total Coulomb source and the exchange from its own occupied population. Empty-spin exchange vanishes exactly. Both operators act on the entire orbital and complementary frame, including the virtual space of an empty spin.

For a spin-conserving ordered pair, the complement of the strong-orthogonality projector contains three disjoint blocks, OBS×OBS, occupied×CABS and CABS×occupied. Each electron uses its own spin-specific occupied and orbital-virtual projectors. The explicit scalar and these three contractions define the V and X matrix elements for either a diagonal pair or a supplied exchange ket. Rectangular spin blocks make distinct populations explicit and the TensorIR transpose yields all intermediate cotangents. Nuclear forces also require orbital, operator, metric and center pullbacks.

The restricted singlet and triplet weights assume the same spatial orbitals for alpha and beta; applying those weights to a single UHF orbital frame would change the projected equations. The remaining B term, the unrestricted cusp functional and end-to-end nuclear response require independent validation.

The molecular component development checks compare a one-electron doublet and bent NH2 with PySCF UHF in the complete parent double-zeta F12 orbital plus OptRI frame. A separate small-space control uses physical Psi4 four-center integrals for three UHF spin sectors. Permanent tests exercise the closed-shell limit, unequal populations, a fully polarized electron, full independent AO contractions and two displaced intermediate directions.

## Singles and geminal–orbital doubles coupling

Separate extended alpha/beta Fock matrices drive two weight-one singles solves.
The spin energies add. Their Fock cotangents feed a single reverse program so both
spin densities receive the derivative of their common Coulomb field. An empty spin
has zero singles energy. This derivative is at fixed orbital frames, with separate
raw and metric-solved source tensors; nuclear derivatives require their physical
geometry dependence. Physical cc-pVDZ-F12 H and bent NH2 doublets were checked
offline against PySCF extended Fock matrices and an independent Sylvester solve.

The C matrix is the orbital-virtual-pair projection of the sum of the two Fock
operators on a strongly orthogonal geminal. A one-body operator cannot take two
complementary virtuals to two orbital virtuals in one step. This leaves two
rectangular contractions, one per electron. Same-spin electrons share a Fock
source; opposite-spin electrons have their own occupied and virtual spaces.
The returned positive semicanonical gap is separate from the coupling. Dense
two-particle projectors, actual bent NH2 integrals and two-step ingredient
derivative checks exercise all three spin channels.

## Polarized (same-spin) fixed-cusp pair

The normalized antisymmetric spatial determinant of an alpha/alpha or beta/beta pair is a pure triplet with fixed amplitude 1/4. Use the spin-specific Fock and positive exchange in the spatial 3C projection, then antisymmetrize both Coulomb and doubles couplings. Each unordered virtual determinant occurs twice in a full ordered square. Eliminating it once gives the subtractions from both V and B; the energy factors for these dressed terms are 1/2 and 1/16. A two-electron H₂ triplet at 3 bohr tests this energy against an explicit antisymmetric determinant-space variational solve, including the B commutator. It does not exercise the mixed-spin cusp sector. The spin projectors and amplitude convention follow the fixed-amplitude method. [MP2-F12 equations](https://psi4.github.io/psi4docs/master/mp2f12.html)

The opposite-spin geminal must include the spatially exchanged column in distinct UHF spin frames. In a common complete extended subspace, an operator column M in alpha/beta coordinates flips as S_ab Mᵀ S_ab, with S_ab = C_alphaᵀ S_AO C_beta. The fixed spin weights are applied before strong orthogonalization. Applying F and K separately to each leg produces the 3C shifted B. A dense, common-spatial-frame determinant calculation verifies all 20 mixed-spin and 16 same-spin NH₂ pairs; summing the restricted-limit spin sectors also recovers independent Psi4 H₂ and H₂O F12 increments. The compact NH₂ orbital/RI basis is a component test. Production open-shell DF source integrals, UHF response and nuclear derivatives require further qualification. [Tew and Klopper, open-shell F12](https://doi.org/10.1080/00268970903449388)

For the H₂ triplet Cartesian test, the derivative of each source includes all four orbital legs, analytic Gaussian center derivatives, and the response of the stationary occupied density and CABS frame. An auxiliary Cartesian primitive basis spanning the original Gaussians and their angular raising/lowering partners reconstructs the four-center derivatives. The separate UHF reference gradient has the Pulay and nuclear-repulsion terms. A generated energy reverse plus the CABS singles Fock cotangent contracts these supplied tangents. At 3 bohr, reoptimizing both displaced UHF references along each of the six Cartesian coordinates checks the finite-difference force at two steps. The physical source oracle is external; the GPU contraction benchmark records this explicitly.

The fitted opposite-spin path accepts left-alpha and right-beta Coulomb-fitted orbital-pair columns and raw three-center sources plus each operator's two-center metric. It uses the shared robust density-fitting contraction and passes its spin-resolved columns to the same spatial projection. A full cc-pVDZ-F12/OptRI NH₂ control uses 198 aug-cc-pVTZ-RI functions and 158 extended orbitals, all 20 alpha/beta and 16 same-spin pairs. Its conventional DF-UMP2 total agrees independently with Psi4; the F12 contraction uses the Psi4/Libint operator tensors as source oracles. The extended HF orbitals and the underlying AO integrals in this test are supplied.

The molecular direction benchmark propagates an analytic external AO/orbital response through all unrestricted fitted energy graphs. Every Coulomb, geminal, geminal-squared, geminal–Coulomb and double-commutator three-center tensor and two-center metric is differentiated at the original geometry. Both spin rotations respond to the coupled UHF equations. Occupied/non-occupied CABS singles add one Fock cotangent per spin. An independent four-geometry control along the bent NH₂ doublet's first H z coordinate reoptimizes the UHF state at each displacement. The D-F12 derivative is compared separately for the UHF reference, ordinary UMP2, F12 correction, CABS singles, and total. The fixture supplies the AO and orbital response. This stage qualifies one nuclear direction and its generated electronic contraction.

The same physical coordinate is also solved by applying the density-fitted coupled UHF Hessian as a generated matrix-free program. The charge is built from both spins and the exchange contractions act on their own spin spaces. The existing GMRES controller checks the true residual; two independent AO spin densities check the action, including the fully polarized limit. The molecular D-F12 orbital solve has 391 rotations. This test receives Coulomb-metric-whitened factors, orbital gaps and the physical nuclear source. The native direct UHF response requires exact unscreened J/K, so it would change the DF Hamiltonian of this case.
