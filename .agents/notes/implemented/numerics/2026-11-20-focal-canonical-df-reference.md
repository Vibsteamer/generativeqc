# A single canonical DF-HF reference for focal components

A CABS or orbital-response contraction needs the physical Fock, its canonical
orbitals, the occupied density and a force consistent with that Slater
determinant. A second SCF calculation can choose another member of a
degenerate manifold, particularly for open-shell molecules. The focal
reference calls the shared native DF-HF solver with its physical-reference
export and optional force evaluation enabled in the same solve. That solver
validates the canonical state against the final physical Fock before export.
The Python result owns detached read-only array storage and the request's
surface, orbital and JKFit identities.

Both J and K use the same auxiliary basis and a relative Coulomb metric cutoff
of 1e-10. The cold solver uses zero integral screening, FP64, an energy
threshold of 1e-13 Eh and a density threshold of 1e-11. The independently
fitted CUDA route uses its existing device DF operators with native host SCF
control. A caller may use a separate RI basis for subsequent pair correlation.
The returned forces include the fitted reference's nuclear derivative. A full
HF+CABS force additionally requires the CABS singles response, including the
orbital and complementary-space terms.

The independent regression uses the same three hydrogen coordinates and
physical cc-pVDZ-F12 orbital and aug-cc-pVTZ-JKFit records for a closed-shell
H₃⁺ singlet and neutral H₃ doublet. PySCF checks all Cartesian forces, the
total energy, both AO matrices, every spin density, physical Fock and orbital
spectrum. Checking the generalized eigenvalue residual and the occupied
projector avoids fixing the unobservable signs of individual orbitals.
