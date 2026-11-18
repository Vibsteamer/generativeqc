# Unrestricted F12 spin frames and pair projectors

A semicanonical UHF reference gives distinct alpha and beta rotations. Occupied traces from both spins can only be added in the common fitting basis. Each spin Fock uses this total Coulomb source and the exchange from its own occupied population. Empty-spin exchange vanishes exactly. Both operators act on the entire orbital and complementary frame, including the virtual space of an empty spin.

For a spin-conserving ordered pair, the complement of the strong-orthogonality projector contains three disjoint blocks, OBS×OBS, occupied×CABS and CABS×occupied. Each electron uses its own spin-specific occupied and orbital-virtual projectors. The explicit scalar and these three contractions define the V and X matrix elements for either a diagonal pair or a supplied exchange ket. Rectangular spin blocks make distinct populations explicit and the TensorIR transpose yields all intermediate cotangents. Nuclear forces also require orbital, operator, metric and center pullbacks.

The restricted singlet and triplet weights assume the same spatial orbitals for alpha and beta; applying those weights to a single UHF orbital frame would change the projected equations. The remaining B and C terms, the unrestricted cusp functional and end-to-end nuclear response require independent validation.

The molecular component development checks compare a one-electron doublet and bent NH2 with PySCF UHF in the complete parent double-zeta F12 orbital plus OptRI frame. A separate small-space control uses physical Psi4 four-center integrals for three UHF spin sectors. Permanent tests exercise the closed-shell limit, unequal populations, a fully polarized electron, full independent AO contractions and two displaced intermediate directions.
