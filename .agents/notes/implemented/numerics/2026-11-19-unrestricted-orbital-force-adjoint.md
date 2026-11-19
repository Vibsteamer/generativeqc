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
available for matrix-free solves. Formation of the molecular DF blocks, nuclear
right-hand sides, electronic driver and fixed gradients remains with the caller.
