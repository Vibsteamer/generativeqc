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
