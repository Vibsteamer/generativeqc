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
