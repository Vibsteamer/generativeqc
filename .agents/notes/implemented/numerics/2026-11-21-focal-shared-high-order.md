# Shared canonical roots for focal CC and conventional triples

A single-cardinal CCSD-F12b minus MP2-F12 correction can reuse the low-order result's determinant, CABS rank, MP2-F12 correlation energy and analytic force. Signed displacements solve the physical CCSD-F12b amplitude equations and qualify the residual. The center also solves Lambda and contracts the ordinary and fixed source weights.

Conventional (T) uses a different parent CCSD Hamiltonian. Its native reference handle retains the immutable canonical spin orbitals from the same DF-HF solve as the low-order terms. The triples owner checks the nuclear Hamiltonian, JKFit, backend and device before reusing that state. Its CCSD(T) and CCSD force Lagrangians share one set of converged conventional amplitudes. Energy-only requests skip both responses. Closed-shell determinants supply identical orbitals and half of the spin-summed density to each spin channel. An empty excitation space has an exact zero energy and force at fixed occupations.

The NH2 split-fit tests compare the conventional increment with independently converged endpoints and signed energies. LiH checks the nonempty closed-shell spin expansion against DF-RCCSD(T). H3 checks the forward-only fixed-cusp correction against an independent dense amplitude solution.
