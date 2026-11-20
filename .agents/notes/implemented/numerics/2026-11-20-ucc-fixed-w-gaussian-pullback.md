# Nuclear weights for the external W projector in fixed-cusp UCCSD-F12b

The stationary W_vo and W_vv source weights act on two views of each occupied
spin pair's external OBS projector. Their pullback scatters both weights
into the common matrix, flips the occupied exchange sign, and transposes
the mixed-spin external permutation with its minus sign. Each distinct
like-spin occupied pair is visited once. Every unlike-spin pair is visited
once. A missing stationary spin block contributes zero. Frozen occupied
columns stay in the full OBS projector.

The projector uses the Coulomb, geminal and geminal-Coulomb operators. Its
AO weights include the implicit response of the common Coulomb fitting
metric, the direct operator metrics, both coefficient legs, and the mixed
spin overlap. The caller contracts these weights with matching nuclear
integral derivatives and moving-frame directions. Fock-coupled C terms
and the occupied scalar V+B−X have separate weights.

An open-shell frozen-core test differentiates a dense spatial construction
with respect to every source independently. For the bent NH₂/D molecular
reference, the 11 retained W blocks span 36 active pairs on a 48-orbital
OBS, 158-orbital extended frame and 198 auxiliaries. A full AO replay
agreed with the original nine stationary W forces within 9.5e−13 Eh/Bohr.

L20 job `2123765046965186560` executed 12 distinct FP64 electronic graphs
in 73 calls. The six derivative contractions evaluated 1,596 fitting shell
pairs per metric and 83,160 shell triples per three-center operator. Its
returned AO and coefficient weights reproduced the separate integral and
orbital directions within 3.31e−16 Eh/Bohr on all nine coordinates.
An independent forward differentiation checked the signed contributions
of each W block and each coordinate to 5.36e−13 Eh/Bohr. With the
remaining 45 stationary source blocks and the original reference/CABS
forces, the recomposed total matched the parent to 9.40e−13 Eh/Bohr.

The archived input uses the accepted central AO integrals and the
independently measured moving orbital and overlap directions. The Gaussian
derivative kernels evaluated their shell values from the molecular
geometry. The original L20 archive SHA-256 is
`17bf7b2a37e062575859b7094b2d6be56d0b47686162117aa7833da4435e8fef`.
