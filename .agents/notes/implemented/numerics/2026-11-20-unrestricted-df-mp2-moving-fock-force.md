# Spin-pair DF-UMP2 forces on a moving canonical reference

Conventional DF-UMP2 contracts the two whitened occupied–virtual factors of
each spin pair. Its derivative includes both factor variation and the response
of the occupied and virtual Fock blocks in the chosen moving orbital gauge.
The central Fock blocks are diagonal. Their nuclear derivatives can be dense,
including within a degenerate subspace. A diagonal spectrum derivative alone
therefore misses a contribution when using parallel-transported orbitals.

The compiler exposes the energy, factor force and full Fock force for one
nonempty spin pair and an arbitrary number of simultaneous directions. The
complete ordered sums carry weights of one quarter for like spins and one
for unlike spins. The molecular sum uses alpha-alpha, alpha-beta and beta-beta
once each. The caller owns the physical orbitals, the correlation fitting
metric, and their common derivative convention. The optional reference force
enters the total molecular force separately.

A distorted-matrix regression recanonicalizes the occupied and virtual
blocks at each signed factor and Fock perturbation. It checks both pieces
and the sum for all three channels, including different alpha and beta
dimensions and nondiagonal perturbations. The physical control uses the
qualified bent IH₂/D reference with 14 and 13 occupied orbitals, 71 orbital
functions and 279 separate correlation fitting functions. It supplies all
nine directions from an independently checked orbital and integral response.

## IH₂/D L20 result

Job `2123755891713912832` executed three FP64 CUDA programs on the accepted
(14,13) IH₂ orbital frame. The alpha-alpha, alpha-beta and beta-beta energies
summed to −0.4803043355 Eh. The largest force deviation across the 27
spin/coordinate entries was 6.94e−18 Eh/Bohr against the CPU analytic
contraction. Independent spin-resolved semicanonical perturbations at 0.0004
and 0.0002 Bohr gave maximum errors of 1.34e−8 and 3.36e−9 Eh/Bohr. Adding
the previously measured L20 DF-UHF force gave maximum total force differences
of 2.69e−8 and 8.61e−9 Eh/Bohr against 36 independent nuclear displacements.
The returned archive SHA-256 was
`00acf9a3a3ef5fbd5613359c9b034e17b4d3f1741de8b58be2b4be56b3503087`.

The CUDA graphs consumed already formed derivative inputs. The orbital
response and Coulomb-fit factor tangents were prepared separately on the
same state. This stage establishes the spin-resolved terminal contractions.
