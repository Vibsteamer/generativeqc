# Shell-generated Gaussian inputs on the live UHF root

The unrestricted energy qualification builds one common set of ten two- and
three-center tensors, the combined overlap and core Hamiltonian, and a geometric
CABS for both spin channels. The shell layout identifies the orbital-basis rows
in combined spherical order. Selecting those rows gives the orbital embedding
and the metric for comparing the live spin orbitals. Complementary frame signs
and rotations are arbitrary. Physical checks compare overlap, core, spin and the
two orbital-space Fock matrices. The geometric projector also checks its rank.

The earlier fixture-driven qualification can isolate faults in spin contractions.
The geometry-driven qualification takes shells, nuclear coordinates, and the
index maps. Independent target energies and UHF forces are used for acceptance.
The generated operators, orbital coefficients, and projector do not come from
the independent reference. Keeping the actual CUDA UHF snapshot with the result
lets a response calculation establish which root and orbital gauge produced the
energy. A qualified UHF gradient only covers the reference term. Validation of
a correlated analytic force requires the operator and orbital responses on this
root and independent reoptimized energy displacements.
