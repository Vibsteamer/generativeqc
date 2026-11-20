# Molecular Gaussian shell operators belong to the importable focal package

A wheel client with the native F12 shell and metric libraries must be able to
build the declared real-spherical AO bases, evaluate Gaussian integrals, and
contract the same channel derivatives on its own geometry. The Python owner
of these operations is `generativeqc._f12_focal_integrals`. The same shell
projection expands the forward tensor and transposes AO weights before the
nuclear derivative contraction. The integral bindings, operator codes,
triangle conventions and native full-rank Coulomb metric ABI have a single
package owner.

The benchmark modules retain their callable names, independent fixture
comparisons, CLI arguments and JSON output. Their benchmark-qualified
programs delegate to the packaged implementation. Existing frozen workers
carry their recorded source tree. New workers use the matching installed
package or source tree and supply the same physical native library paths.

The regression calls the numerical operations from an isolated Python
process outside the checkout. Independent Libint values and both atoms'
Cartesian derivatives are checked for the same real Gaussian shell input.
