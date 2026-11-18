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
