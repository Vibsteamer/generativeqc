# Combined-basis scalar ECP source bridge

Status: implemented
Date: 2026-11-20

The full-basis i-shell batch evaluates S and the point-charge core in the
source AO order. A scalar pseudopotential contributes a residual operator to
Hcore and a separate geometrical derivative. The private benchmark bridge
takes these two panels from an explicit evaluator. A supplied AO permutation
aligns both the matrix addition and the weighted residual pullback. Matching
geometry, effective charges and AO count are required. The input evaluator
must use matching real-spherical phases, normalization and atom order.

Keep the residual derivative in its original AO ordering and remap the
smaller Hcore cotangent. Both bare and residual nuclear contractions use the
already selected native weighted-Jacobian ABI. The two calls reuse the bare
backend and keep their individual staging bounds. The bridge retains an
owned scalar ECP panel and rejects derivative requests when the corresponding
bare batch is absent or has been replaced.

The focused regression compares the assembled source and positive nuclear
gradient to independently contracted residual contributions and signed
energies under nontrivial AO permutations. This checks the provider and its
derivative at fixed cotangents. Orbital response, the two-electron operator
and nuclear repulsion are composed separately by the molecular method.
