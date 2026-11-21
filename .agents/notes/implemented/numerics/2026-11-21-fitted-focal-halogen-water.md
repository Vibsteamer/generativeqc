# Fitted scalar ECP for one Br or I and water

The O/H nuclei in `gscdb/21_H2O-Br_TA13` accompany one Br atom at the
unmodified author geometry. With the ten-electron scalar core removed,
neutral Br with water has 35 explicit electrons and the reference is
an unrestricted doublet with occupations 18 and 17.

The ordinary DF integral and nuclear derivative paths use Gaussian shells and
physical nuclear charges. The element restriction came from scalar-ECP
admission. The Python basis capability and native method now admit one
small-core Br or I with one all-electron O and two all-electron H atoms. The
charge, state and orbital angular range follow the existing small-core
monohalide rules. The Coulomb fitting basis carries no ECP. Bromine still uses
the refined ECP quadrature policy.

The host regression creates both orbital and fitting systems through the
native basis template, with a retained scalar ECP and an orbital g shell.
At the original TA13 geometry, it compares the native Br local and nonlocal
AO matrices with the external Libcint operator and checks signed numerical
derivatives on Br, O and H. This releases the numerical work to a physical
CUDA test. The full fitted CUDA determinant, F12 terms and focal forces have
separate qualification.

The public displaced entry validates all basis-specific retained determinants
against the same ordered center and weighted recipe before running a target
SCF. It also rejects a returned CABS rank inconsistent with the matched
analytic center before correlated work. A one-electron hydrogen host test
checks the complete four-role API with an empty beta channel. This is a
degenerate correlation case. Nonzero (T) is checked separately on LiH.
