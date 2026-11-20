# Fock workspace for retained focal triples forces

A retained CUDA determinant supplies the stationary state for the conventional
triples correction. Orbital relaxation requires the reference J/K matrix action.
The force contraction evaluates overlap, one-electron, and fitted integral
derivatives through the unrestricted force owner, including the separate JKFit
source when fitting spaces differ.

Prepare the Fock action with derivative order zero for this retained CUDA
full-DF case. The previous plan charged its unused Fock derivative allowance
alongside the response executor. At 54 orbital functions with two alpha and one
beta occupied orbital, the original 40 GiB setting assigned about 10.4 GiB to
that allowance. The released executor needs 4,882,059,724 bytes for response.
The allowance alone leaves less than that executor needs after the host force
reservation, before any fitted-source storage is charged.

The correction retains analytic forces, the same canonical determinant and
fitting spaces, and the normal capacity and convergence checks. A same-root
physical CUDA comparison of all nine force components at the original two
steps is required before promoting the H3 endpoint.
