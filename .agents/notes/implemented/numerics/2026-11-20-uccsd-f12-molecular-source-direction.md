# Molecular fixed-source contribution to the UCCSD-F12b gradient

For a physical coordinate the same perturbation changes W, C, and the scalar
V+B−X in both spin frames. A derivative of an independent CC input, such as
one mixed-spin W block, captures only that block's contribution.

The molecular source builder applies a TensorIR JVP to the same pair graphs
and the same spin permutations used for the forward inputs. Projected MO
sources and extended Fock/exchange matrices carry their total tangents in the
moving frame. Occupation selectors stay fixed. The full OBS Coulomb fit uses
the equation J dD = dG − dJ D. Its inverse comes from the central full-rank
whitener, giving a result independent of the choice of whitening gauge.

The response owner freshly checks the fixed-source primal and Lambda before
evaluating the joint source direction of E + Lambda·R. The full residual seeds
come from the packed-dual lift, with a factor of one quarter on each same-spin
doubles block. The ordinary CC Hamiltonian, reference and separate CABS singles
contributions must be assembled to obtain the full gradient.

The unit tests compare every molecular source against two-step finite
differences with a nondiagonal fitting metric. Independent dense spin-orbital
solutions at the two source perturbations check the native stationary
contraction. On the retained bent NH2 cc-pVDZ-F12 H1-z tangent, all 36 pairs
gave a source gradient of +0.0000543050714 Eh/bohr. Central differences of the
source-interpolated stationary Lagrangian at steps 0.0002 and 0.0001 bohr
differed by 1.1e-9 and 2.6e-10 Eh/bohr. The molecular geometry and stationary
amplitudes were fixed during this interpolation.

The molecular directional derivative also accepts all ordinary canonical
Hamiltonian directions on this frame and the full UHF reference derivative.
A DF adapter differentiates each chemists' tile through its fitted and raw
legs, applying the same permutations and exchange signs used by the ordinary
UCC solver. Its output can then enter the joint correlation Lagrangian JVP
alongside the geminal sources. The reference derivative enters once, outside
the correlation graph. Separately computed CABS singles stay external.

For the retained NH2 H1-z direction, the ordinary CC contribution was
−0.007019536472 Eh/bohr and the reference derivative was −0.018665872204
Eh/bohr. Adding the fixed-source term gave −0.025631103605 Eh/bohr. Two
source-interpolation checks of the correlation derivative differed by 1.21e-9
and 2.81e-10 Eh/bohr. Independent displacements of converged molecular UCC
roots remain necessary to qualify this as a finite-difference nuclear check.
