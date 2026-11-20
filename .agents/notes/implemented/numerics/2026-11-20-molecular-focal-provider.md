# Fitted integrals in a geometry-driven molecular focal evaluation

The ordinary and explicitly correlated spin-pair contributions are stationary
at the same reference determinant. One public pair evaluation therefore uses
one DF-UHF root and one OBS/OptRI one-electron state. Fitted operators may
share a forward panel and reverse nuclear contraction when the complete
basis identities match. Equal dimensions alone do not identify the same
operator. OBS-only cotangents contribute to the leading two orbital axes of
a joint panel. The fitting-metric derivative is summed for each shared
operator as well.

The five possible equality patterns of JKFit, MP2Fit and F12Fit are exercised
by the molecular provider regression against separately assembled energies
and forces and by two displacement sizes. The HF+CABS regression uses an
independent SCF and finite differences along each Cartesian coordinate. For
physical qualification use a request with the desired spin root, a matching
backend and a separately identified high-order provider.

For a CUDA request, the final contractions of the overlap and core-Hamiltonian
Jacobian with their relaxed AO weights run as two FP64 TensorIR contractions
on the assigned card. The sign convention of the returned contraction is the
positive nuclear energy derivative. The caller applies the force sign along
with the Coulomb and geminal derivatives. The independent host full-matrix
contractions and both signed displacement sizes exercise AO order and sign.
