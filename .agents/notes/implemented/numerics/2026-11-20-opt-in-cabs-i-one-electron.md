# Opt-in i-shell CABS overlap and point-charge core response

Status: implemented

Date: 2026-11-20

Br/I F12 auxiliary spaces can contain i shells. The one-electron
overlap, kinetic and point-charge attraction on pairs touching these shells
must use the same Cartesian ordering, radial normalization and nuclear
derivative signs as the existing h-shell cohort.

The separate `GQC_F12_CABS_I_ONE_ELECTRON` build instantiates the existing
Hermite and Coulomb recurrences at maximum shell angular momentum 6.
The first Coulomb derivative uses order 13, within the shared Boys
response bound. The new `gqc_f12_cabs_i_one_electron` symbols only
accept pairs with at least one i shell. Existing h builds have a separate
symbol family and still admit only pairs touching h with maximum angular
momentum 5. Keeping these cohorts distinct preserves the existing h
behavior and lets a combined provider avoid overlapping ownership.

The i kernel returns S, T and attraction to the supplied point charges.
For an ECP center the provider passes the effective charge and accounts
separately for residual local and nonlocal ECP terms. When supplied
effective charges differ from atomic numbers, the adapter requires a
matching provider for the lower-angular blocks and their pullbacks.

The host test uses independent Libcint Cartesian integrals with explicit
Br/I effective charges. It exercises both i/h orders, i/i and s/i, two
finite-difference steps over all nine nuclear coordinates, and the
compact real-spherical projection and weighted response. The CUDA
translation unit compiles for sm_89. Device execution and a correlated
Br/I molecular force are separate qualification steps.

A host replay used every i row of the archived HBr/T and HI/Q combined
OBS+OptRI bases and composed the point-charge core with the archived
L20 residual ECP. The maximum full Hcore error against libcint was
\(5.71\times10^{-12}\) Eh. The maximum fixed-weight nuclear gradient
error was \(6.01\times10^{-14}\) Eh/bohr. Lower-angular overlap and
bare core blocks came from the independent PySCF reference.
