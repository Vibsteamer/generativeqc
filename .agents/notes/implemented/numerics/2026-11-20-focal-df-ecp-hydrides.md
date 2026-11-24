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

## IH₂ qualification

For the noncollinear neutral doublet, L20 job `2123744692456652800` took
18.37 s for a cold 107-iteration solve with nine analytic force components.
The occupied counts were (14,13) and the energy was −295.6725933394166 Eh.
The maximum energy and force differences from independent DF-UHF were
8.53e−13 Eh and 4.74e−8 Eh/Bohr. The generalized eigenvalue residual was
5.59e−13. The core Hamiltonian agreed with the previously measured public
OBS block of the 310-function OBS/OptRI operator to 1.07e−13 Eh.
The SHA-256 of the returned measurement archive was
`a0b7ad904b719bd1e40928b5627d25a5b94cf9ce34d452e13e7aefbebaf4073c`.

An offline analytic ordinary DF-UMP2 calculation used those returned orbitals,
the previous L20 residual ECP derivatives, and CPU unrestricted orbital, JKFit
and MP2Fit response. At 0.0004 and 0.0002 Bohr the largest differences between
all nine analytic correlation forces and independently displaced PySCF MP2
energies were 1.64e−8 and 4.09e−9 Eh/Bohr. For HF plus MP2 the corresponding
force differences were 2.69e−8 and 8.61e−9 Eh/Bohr. Resolving the nonvariational
MP2 displacements required a tighter occupied/virtual residual than the
stationary SCF energy check. The 36 signed controls reached residuals below
1.56e−13. The GPU measurement covered the canonical reference and its forces.
The subsequent mixed-source replay covered the all-active ordinary MP2 term.
