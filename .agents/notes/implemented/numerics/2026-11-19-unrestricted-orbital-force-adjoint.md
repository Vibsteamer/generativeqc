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
