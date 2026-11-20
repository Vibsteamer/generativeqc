# Scalar ECP ownership in the focal DF-UHF reference

A normalized orbital handle contains the valence electron count, ionic charges,
and residual semilocal potentials used for both the physical Fock and its
nuclear derivative. A separately normalized JKFit handle has the same nuclear
positions and atomic numbers. Its electron count is incidental to its Coulomb
integrals. Comparing the two electron counts therefore excluded a valid ECP
reference. A second ECP on the JKFit is rejected.

The canonical physical-reference export and ordinary prepared CUDA UHF now
share native FP64, angular, charge, multiplicity and small-core hydride
admission, including the existing Br quadrature refinement. A focal caller
receives the spin orbitals, spectrum, physical density, Fock, overlap, core
Hamiltonian and optional stationary forces from one solve. The CPU fitting
provider retains its own angular limits. Iodine's diagnostic def2-universal
JKFit includes h shells.

The full bent IH₂ doublet D-zeta control uses 71 orbital functions, 254 JKFit
functions and ECP28MDF. An independent PySCF calculation checks the canonical
state and all nine nuclear forces against both analytic derivatives and 36
signed displacements in Bohr. The separately contracted 279-function
correlation basis defines subsequent ordinary MP2 and the captured independent
MP2 displacement controls.
