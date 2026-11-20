# Unrestricted fixed-source CCSD-F12b response

Status: implemented for supplied geminal sources

The fixed-cusp additions use the same antisymmetrized spin-orbital amplitudes
and packed excitation coordinates as the ordinary UCCSD equations. Resolving
spin at graph construction makes the energy, residual, forward derivative and
transposed derivative consume the same equations. Same-spin dense residual
seeds have one quarter of the physical packed multiplier's signed expansion.
The mixed-spin block has unit weight.

The source slots `w_vo`, `w_vv` and `c_f12` use indices `ij,pq`, with both
pairs antisymmetrized and the cusp coefficients absorbed in the occupied-side
geminal. The `W` slots use the standard external projector from the
fixed-cusp equations in Knizia's 2010 thesis, equations 4.123–4.127. The
one-body Fock coupling goes in `C`. The amplitude-independent common
geminal energy includes `V`, `B`, and the occupied-Fock contraction of `X`.

The external projection is evaluated one ordered geminal column at a time.
Its left and right OBS dimensions can differ for opposite-spin pairs, and
both legs use one shared Coulomb fitting basis. The caller forms the
antisymmetrized occupied and target blocks from those ordered results.

The fixed-source Lagrangian derivative returns separate cotangents for all
supplied spin blocks. Molecular input construction must combine exchange-related
cotangents, differentiate cusp projection through the cross-spin overlap, and
propagate the amplitude-independent contribution through the occupied Fock,
integrals, and fitting metric. These are explicit dependencies of an analytic
molecular force.

## Molecular W, C, and linear V sources

The unrestricted molecular DF frame can now supply the dressed external W,
strong-projector Fock C, and linear occupied V contribution. For like-spin
pairs the spatial column is one quarter of its antisymmetrization. For
opposite-spin pairs the exchanged column uses the full cross-frame overlap
on both legs. Coulomb-fitted columns and non-Coulomb operator metrics share
one auxiliary space. Frozen occupied orbitals remain in the strong projector
and Fock, while the CC source arrays contain only active pairs. The adapter
visits each distinct like-spin or mixed-spin pair once, then expands the
fermionic signs required by the CC residual equations.

The external W uses the standard OBS projector. The linear V uses the
strong projector and contributes twice the occupied-bra contraction per
unique pair. C uses the off-diagonal virtual/CABS Fock coupling. Calculating
the projected external W first and using its occupied elements for the
common V would miss the occupied/CABS complement sectors.

The component tests compare to independent dense two-particle projectors
and concurrently perturb the upstream inputs at two steps, including the
nonidentity cross-spin overlap. An independent direct NumPy replay using
the archived NH2 doublet DZ frame (5 alpha and 4 beta occupied, 48 OBS,
158 extended and 198 auxiliary functions) covers all 36 active pairs.
The assembled W and C sources agreed within 5.22e-15 hartree and the
linear occupied V sum within 2.84e-15 hartree. The allocated W/C/V check
uses the same archived roots and verifies source derivatives separately
from a full relaxed molecular force.

## Quadratic source in the active occupied space

The pair graph evaluates B and the contraction of X with the full active
occupied Fock. The Fock acts on each occupied selection before cusp dressing.
Keeping its off-diagonal entries in the graph is necessary for derivatives
even when its value is diagonal at the canonical reference. The like-spin
quadratic expression uses half the norm of the ordered antisymmetric tensor,
which is the normalized triplet determinant convention. Mixed-spin pairs
use unit weight. Frozen-core projectors and the extended Fock still include
their original occupations.

The dense two-particle oracle checks both quadratic terms and their
two-step source transpose with non-diagonal occupied Fock matrices. On all
36 archived NH2/DZ UHF pairs the assembled common energy V+B−FX differs
by 1.45e-14 hartree from the existing MP2-F12 value with the conventional
doubles elimination accounted for explicitly. The next layers still need
to join this common scalar to the molecular UCCSD-F12b amplitude solver
and propagate its weights through the AO integrals, orbital relaxation,
and nuclear derivatives.
