# Scalar ECP core in the spin-resolved Gaussian center

The private molecular benchmark accepts a supplied joint-basis residual ECP
potential and its positive nuclear derivatives. For a basis containing i
shells it forms the bare full-basis core using the panel's effective charges.
The two spin occupations must account for the resulting valence electron count
and molecular multiplicity.

The same provider builds the spin Fock input and contracts the corrected
Hamiltonian and overlap weights after the canonical, occupied-orbital and CABS
frame response. The reference energy and forces, when included, come from the
matched snapshot. Adding an independent reference ECP force to the corrected
correlation weight would count the reference response twice.

A supplied central factorization can carry a whitener from another fitting
convention. In ECP mode the benchmark rebuilds both physical OBS Focks from the
joint Coulomb sources and tests canonical agreement to 2e-8 Eh before using
those factors in the coupled response. This implementation assumes a shared
Coulomb fit for the reference and extended Fock. A calculation using different
fits needs an explicit separation of their tensors and nuclear derivatives.

The host qualification evaluates HBr and neutral doublet IH2 with real
scalar pseudopotentials and compact orbital and i-shell bases. Libcint
displaced overlaps and core Hamiltonians check the fixed-weight gradient over
six and nine coordinates respectively. Two step lengths, 2e-4 and 1e-4 bohr,
give maximum errors below 4e-9 Eh/bohr. A separate spin-singles regression
checks the two-spin sign and the passage of the selected core through the
energy and force code. Full-basis, relaxed PP-F12 molecular forces require
root-matched integral and response measurements on device.

The private driver now also accepts a second shell layout for the independent
JKFit basis. See [independent fitting spaces](2026-11-20-separate-correlation-and-jkfit.md).
