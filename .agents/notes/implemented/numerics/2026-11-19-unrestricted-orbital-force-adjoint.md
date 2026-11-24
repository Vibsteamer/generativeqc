# Cartesian forces from a DF-UHF orbital adjoint

The unrestricted force API solves the transpose of the shared-metric, coupled
orbital Hessian in alpha-then-beta occupied/virtual order. One solve supplies
every nuclear direction. An ordinary forward solve per coordinate remains an
independent check of the projection.

The input electronic driver is the derivative with respect to occupied/virtual
rotations. Nuclear right-hand sides use the forward convention `H x_R = b_R`.
The device computes the energy gradient `b_Rᵀ z` and subtracts it and the
supplied fixed-frame energy gradients from reference forces. Passing gradients
that already include occupied/virtual relaxation would count the rotation twice.

The dense CUDA implementation checks its explicit device allocation budget,
including the pivoted LU workspace. The existing tensor orbital action remains
available for matrix-free solves. The adjoint accepts independently supplied nuclear
right-hand sides and fixed gradients for callers with an existing response driver.

The molecular entry can construct all nuclear equations from AO overlap,
core-Hamiltonian, Coulomb three-center and Coulomb-metric tangents. The project
TensorIR projects them into both spin frames and forms the shared Coulomb and
same-spin exchange derivatives, the occupied metric-density response and six
fixed gradient contractions. A central whitener W and density-fitted factors B
are supplied in the same gauge. Using the symmetric differential
`dB = W dR - (W dJ Wᵀ) B / 2` preserves the exact derivative of the fitted
Coulomb product without differentiating a particular triangular factor.

A displaced inverse-metric Fock test moves the occupied density in its overlap
metric and also holds it fixed, isolating the three channels independently for
both spins and for either empty-spin limit. For the NH2 doublet, all 3,519
nuclear elements agree with a differentiated Cholesky control within 4.5e-14.
The four AO derivative arrays, central frame and electronic energy weights
remain caller inputs. Two additional direct F12/frame gradient contributions
and the reference force are supplied separately to the force assembly.

The reverse nuclear-source program transposes this same equation with the
solved orbital adjoint and unit fixed-gradient seeds. It emits symmetric
full-Frobenius weights for S, h, three-center Coulomb and the fitting metric.
With direct energy weights and both OBS/CABS overlap responses, it adds them
in the same generated program. A derivative integral provider can contract the
result in every nuclear direction using only the central electronic data.
The coefficient tests compare the four raw arrays with an independent direct
differentiation of the DF inverse metric, and their contraction with each of
the nine physical AO directions reproduces the forward orbital gradient.

For a resolved unrestricted reference, project the full AO coefficient
cotangent onto the canonical OBS columns in each spin. In the relaxed Fock
formulation every nondegenerate off-diagonal pair contributes to the Fock and
overlap multipliers, including occupied/virtual pairs. The projection shares
the generated scalar pair algebra with the restricted same-space implementation.
The full branch sets the stationary degenerate block to zero and checks the norm
of its unresolved antisymmetric torque, counting each pair once. This norm is
invariant under rotations of an exactly degenerate stationary subspace. A gap
across the occupation boundary below the configured resolution requires a
separately resolved electronic root before the calculation can proceed.

The Gaussian-force path can now build the electronic driver from the canonical
Fock cotangents. Both spin densities contribute to the Coulomb contraction, and
exchange uses the factors for the corresponding spin. The driver uses the same
MO ordering and auxiliary metric as the coupled Hessian. With the resulting
adjoint, the reverse AO integral contractions supply the nuclear projections
directly. A rotations-only solve therefore needs zero nuclear right-hand sides.
The public API for supplied nuclear directions still requires nonempty forces.

On the NH₂ doublet with the DZ-F12 basis, the L20 source-derived driver agreed
with the archived independent driver to 5.74e-16 across 391 elements. All nine
recomposed forces agreed with signed differences of 36 displaced energies to
2.65e-8 and 5.25e-8 hartree/bohr for steps of 0.0002 and 0.0001 bohr. This
qualification used supplied direct F12 AO cotangents, CABS overlap weights,
central DF factors, and a matched UHF reference.

The common-AO response entry forms UMP2, the F12 increment, and the two CABS
singles energies in the same spin frames as their cotangents. The reverse pair
program also returns its two primal energies. The AO response reuses the primal
projections owned by that evaluation, including the occupied legs needed for
the Coulomb metric derivative. Its internal reuse is confined to one call so
changing the geometry or orbital frame cannot reuse a stale projection.
Frozen occupations restrict correlated pairs while all occupied orbitals still
enter the Fock and singles equations. Small-space tests differentiate the
combined energy along raw-AO, metric and spin-coefficient perturbations.
