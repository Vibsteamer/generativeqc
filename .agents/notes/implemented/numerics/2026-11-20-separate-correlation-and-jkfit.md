# Independent correlation and Coulomb/exchange fitting spaces

Status: implemented
Date: 2026-11-20

Br/I PP-F12 uses OptRI to span the orbital complement. The fitted pair
operators use MP2Fit, while the reference and extended Fock matrices share
JKFit. A single fitting basis in these roles changes the Hamiltonian. On a
fixed full HBr/T determinant, using its 346-component correlation fit in
the Fock changed the canonical matrix by 0.0205 Eh. The diagnostic 284-function
JKFit reproduced the recorded canonical matrix to 1.68e-8 Eh.

The private spin-resolved Gaussian molecular driver takes an optional second
shell layout for JKFit. Both layouts must have the same ordered orbital AO
rows, primitives, spherical transforms, centers and geometry identity.
Their auxiliary shells and their ranks can differ. The Gaussian kernels
build the Coulomb factors for the Fock in the second layout. The pair equations
consume the primary layout's Coulomb fits, geminal sources, and metrics.

The Fock adjoint returns to JKFit, including its use by the pair equations
and singles. Pair-operator adjoints return to the correlation fit. Both
contribute to the same spin-frame weights before canonical and orbital
response. The SCF response contributes to JKFit. The two integral derivative
sinks and the shared one-electron sink each enter the force once.

The regression compares with independent four-index spin MP2 and singles
energies, perturbs both metrics and three-center integrals separately, and
recombines derivatives when the bases coincide. In a singles-only control
all direct correlation-fit derivatives vanish. A directional-orbital oracle
checks the separate nuclear contractions. Four heavy-augmented HBr and HI
projections used one occupied, one orbital virtual and one CABS function
with the full JKFit and MP2Fit spaces. The respective auxiliary dimensions
were 284/246 (HBr/D), 284/346 (HBr/T), 236/244 (HI/D) and 236/333 (HI/T).
Independent displacements of the heavy atom along z at two step sizes
reproduced the explicit fitting gradients to 2e-11 Eh/bohr across all four
cases. These projections used def2-universal-JKFit, a fixed orbital frame
and a fixed effective one-electron potential. The physical checks covered
the ordinary pair and two-spin singles terms. The nonzero geminal channels
were exercised in the algebraic regression.

Molecular controls with one auxiliary basis use the shared-fit route.
The ECP operator remains part of the one-electron Hamiltonian and its
adjoint. It enters the fixed-reference and fitted corrections only
through that shared physical Hamiltonian.

`tests/python/test_f12_separate_fitting_spaces.py` exercises these fit
assignments and their response.
