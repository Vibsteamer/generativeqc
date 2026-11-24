Molecular F12 single points group components by the complete basis identity. At a center, an HF+CABS and MP2-F12 pair shares its DF determinant with a matched stationary CC correction. At a displacement, the public forward-only calculation transports the center determinant, enforces the same CABS rank, and recomputes the primal electronic energy. An RHF reference with one doubly occupied orbital has no three-hole determinants, so both conventional triples energy and its nuclear derivative vanish exactly. A two-electron validation consequently exercises composition and the remaining roles, while validation on other electron counts establishes nonzero triples behavior.

The retained-reference conventional triples force differentiates the triples
scalar at the converged parent amplitudes and orbitals. It solves one CC
adjoint with the triples cotangents and zero parent energy seeds. The orbital
response and integral contraction use the resulting increment weights with the
nuclear-repulsion seed zero. The old path solved the full CCSD(T) and CCSD
responses separately and subtracted them. The Jacobians are identical at the
retained parent state and their adjoints are linear in the seeds. Using the
increment seeds removes the second response and subtraction while retaining
the physical replay and stationarity gates. The standard UCCSD(T) total force
continues to use the total-energy seeds.
