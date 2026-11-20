# Ordinary UMP2 with two independently fitted Coulomb operators

The UHF reference and its coupled orbital Hessian use JKFit. Correlation
uses MP2Fit. The ranks can differ and each fitting metric moves with nuclear
geometry. Each contracted shell derivative therefore receives weights only
from its own operator. Both sets of coefficient derivatives enter a single
coupled UHF adjoint before the AO weights are contracted.

The spin-pair force program already includes off-diagonal derivatives in
the occupied and virtual Fock blocks. Transposing its linear dependence on
the physical factor and Fock tangents returns the corresponding energy
weights with a negative unit force seed. Treating canonical eigenvalues as
the only moving Fock quantities would lose rotations among nearly degenerate
orbitals. The whitening kernel uses the symmetric metric differential.
Changes of orthonormal auxiliary gauge cancel in the ordinary energy, which
depends on the whitened factors through their pairwise products.

The independent regression differentiates a weighted Gram matrix through
a moving Cholesky factor, changing the AO coefficients, Coulomb integrals
and metric together. A second regression contracts the energy weights with
all dense spin-Fock and factor tangents and compares the result against
the direct force for each of the three spin classes.
