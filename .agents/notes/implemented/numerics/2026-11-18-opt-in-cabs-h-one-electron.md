# Q-F12 CABS h-shell overlap, kinetic and point-charge response

The private qualification bridge evaluates shell pairs containing h, through h/h, using the same
FP64 Hermite, Boys and Coulomb recurrences as the native SCF backend. The Hermite table accepts
an explicit dimension for the two raises of the kinetic operator. Existing instantiations retain
their storage dimensions. Center derivatives act on contracted primitives using their original
radial normalization. Differentiating the Hermite coefficients with respect to A−B and the
Coulomb state with respect to P−C accounts for both Gaussian motion and every moving point charge.
Contributions are combined by physical atom, including coincident shell and nuclear centers.

Each GPU thread computes a contracted Cartesian AO pair. The private ABI returns S, T and signed
nuclear potential V with positive matrix derivatives in the caller's atom order. The size gate
covers staged device inputs and outputs. The CUDA compiler reports per-thread local storage
separately. It is intended for validating high-angular blocks of the combined OBS+CABS.
The public basis admission and low-angular one-electron kernels retain their original paths.

The qualification panel takes four complete h-containing blocks from methane cc-pVQZ-F12 OptRI,
including separated and repeated centers. Psi4/Libint supplies the independent real-spherical
values and all 15 analytic derivatives, with independent five-point checks at two displacements.

## Combined-basis assembly

All h-containing shell pairs share one Cartesian evaluation launch and one spherical projection.
The primal reads compact S and Hcore. The central geometry retains the high-shell Jacobian in
source AO order on the host, then transfers it and all ordered weights together for a single CUDA
contraction. Displaced scalar geometries compile out nuclear responses. Low/low pairs retain the
qualified native value and gradient providers. Distinct shell pairs accumulate both W_ab and
the transpose of W_ba. A repeated shell consumes its full ordered block exactly once.
Source-row, angular, nuclear-center, primitive exponent and normalized radial identities are
checked before either cohort runs. The per-call byte gate accounts for the Cartesian scratch,
projected spherical output and staged inputs. The compiler separately reports per-thread storage.
