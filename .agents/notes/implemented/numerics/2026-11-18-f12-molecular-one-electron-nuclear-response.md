# Physical-atom contraction of the F12 one-electron adjoints

The molecular reverse graph supplies two weights on the combined orbital and
OptRI basis. Its overlap weight includes the motion of the complementary
frame. Its core-Hamiltonian weight acts on kinetic energy and attraction with
the same coefficient. The CUDA bridge takes one contiguous core weight and
shares the device allocation across those operators. The fragment contributes
a gradient; the force sign belongs to the final assembly. The orbital, fitting,
two-electron and HF reference derivative owners contribute their terms separately.

Qualification uses an independent AO map from shell identity and magnetic
quantum number. Each physical atom owns both families of shells. The attraction
operator therefore contains each charge once. Comparing the full central
overlap and core-Hamiltonian matrices with Psi4 also tests phases and shell
normalization. An overlap-only comparison on coincident centers cannot detect
a duplicated nuclear charge.

For H₂ and H₂O, use independent libcint analytic derivatives and two nested
five-point differences for every Cartesian coordinate. Also re-evaluate the
nonlinear F12 plus CABS-singles scalar with only the two physical source
matrices displaced. Freezing the other sources makes this partial path stiff,
especially in water. Retain the raw three-point errors alongside the
fourth-order estimates from steps 0.000025 and 0.0000125 bohr. The absolute
gradient acceptance threshold is 3e-8 hartree/bohr for both estimates.

An all-electron component pass applies to the explicit overlap and
core-Hamiltonian contribution at this basis. The complete molecular endpoint,
open-shell and ECP tests have separate scientific receipts.
