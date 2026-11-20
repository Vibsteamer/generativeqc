# Fitted scalar ECP reference admission for C/H monohalides

Status: implemented
Date: 2026-11-21

The [methyl-halide reference decision](2026-11-20-fitted-methyl-halide-ecp.md)
admitted two molecular formulas. The reference and basis providers already
operate on atom-mapped scalar ECP terms, shell angular momenta and an explicit
Coulomb fitting basis. They can select the same providers for a C/H molecule
with a single small-core Br or I. Python admission, native basis expansion,
fitted HF preparation and the CUDA UHF spin gate now use this topology.

The orbitals are real spherical with full nuclear charges on C and H, a
ten-electron Br core or a twenty-eight-electron I core, and at most g shells.
The explicit fitting basis extends through h. Charge 0 or +1 must give an
even active electron count for an RHF singlet or an odd count for a UHF
doublet. Bromine continues to request refined scalar ECP quadrature. The
hydride states retain their own screening and angular admission.

The CLI selects the native SCF auxiliary handle for fitted ECP HF. Its
expanded ECP orbital constructor accepts through f. The CLI RHF dry run uses
the same budgeted DF inventory as Python in this ECP domain. Both include
the serialized ECP allocation bound. The shared DF inventory admits larger
RHF with an explicit sub-budget, subject to all declared resource limits.
It still refuses the large UHF shape. For the bromoacetylene fixture,
48 orbital shells and 116 auxiliary shells give 13,695 combined shell
pairs including the auxiliary unit shell. The generic host bound includes
2,048 bytes per combination of two shell pairs, which occurs once in
retained host bytes and twice in temporary host bytes. The resulting RHF
host inventory exceeds 1 TB even though the planned device peak is 1.57 GB.

The host tests use the NIST CCCBDB bromoacetylene geometry with the existing
Br/H/C orbital and JK element records. Compiled method checks admit the
38-electron singlet and 37-electron doublet and reject changed charge and
method. Independent PySCF/libcint results use the same Gaussian exponents,
contractions and scalar ECP, with analytic forces and two signed Br-axis
energy differences agreeing within 1e-6 Eh/bohr. Device execution on these
geometries requires a separate energy, force and occupied-space comparison.
F12 correlation and triples retain their own admission and validation gates.

[NIST CCCBDB bromoacetylene geometry](https://cccbdb.nist.gov/exp2x.asp?casno=593613&charge=0)
