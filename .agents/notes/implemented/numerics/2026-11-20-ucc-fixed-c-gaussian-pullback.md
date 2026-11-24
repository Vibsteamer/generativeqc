# Spin-resolved C-source nuclear weights for fixed-cusp CC-F12

Each external C block contracts the virtual–CABS Fock coupling on either
spin leg with the cusp-dressed geminal. Source weights from the stationary
CC Lagrangian are assembled for each distinct occupied pair, including the
occupied and external exchange permutations. The geminal response produces
weights on the geminal and Coulomb metrics, three-center integrals and the
mixed-spin overlap. The Fock response uses the full extended UHF reference,
including occupied orbitals frozen in the CC equations. Both responses
contribute to each spin frame and to the common Coulomb fit.

A dense frozen-core test perturbs the core Hamiltonian, both functions of
the electron coordinates, their metrics, the overlap and each spin frame
separately. For bent NH₂ with 5 alpha and 4 beta occupied orbitals, 48 OBS
orbitals, 158 extended functions and 198 fitting functions, all nine
contracted AO derivatives agree with the stationary C-source force to
2.69e−14 Eh/Bohr. Finite differences of the six stationary-weighted C
sources use 36 signed nuclear displacements with ordinary UHF orbital
phases referred to the central root. Maximum errors over the 54 individual
force components are 2.60e−8 and 6.52e−9 Eh/Bohr at 0.0002 and 0.0001 Bohr.

L20 job `2123770743018749952` evaluated 20 distinct electronic CUDA graphs
in 92 calls. Two Coulomb and two geminal Gaussian derivative contractions
were evaluated on the physical geometry. The same spin frames consumed
the archived orbital response, overlap derivatives and the complete
158-by-158 core-Hamiltonian derivatives of the previously qualified L20 root.
Recontracting the returned weights independently with each source and
coordinate reproduced the separate forces to 3.16e−15 Eh/Bohr. The
nine-coordinate C contribution agreed with the six original stationary
source actions to 2.94e−14 Eh/Bohr. The original L20 archive SHA-256 is
`6f43f2501ac84a3a99db4b31f2e564dacedf2defce924c069cb1bdb2280e55af`.
