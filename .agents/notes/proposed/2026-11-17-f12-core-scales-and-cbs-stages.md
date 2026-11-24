# F12 correlation factors for a single-structure CBS label

Status: proposed
Date: 2026-11-17

## What defines the target

The target is a smooth energy and its analytic forces for one geometry, on a declared electronic surface. Preserve the electronic Hamiltonian, correlated orbitals, ECP and electron count through all stages. Use raw geometric energies, with no ghost fragments or counterpoise terms. The initial accuracy study compares relative energies against like-for-like canonical CCSD(T) basis-limit references with a design tolerance of 0.1–0.2 kcal/mol. Absolute energies, element-matched atomic references and componentwise forces need separate checks.

The existing conventional CBS recipe sets the number of frozen orbitals to zero. Hydrogen has no inner shell. The occupied space of e.g. water includes the O 1s orbital. Effective core potentials remove only their declared electrons. A neutral iodine with a 28-electron ECP has 25 explicit electrons. Explicit Br with no ECP retains all 35. Charge and multiplicity must be recorded separately. The same-spin and opposite-spin electron cusp factors and the geminal range parameter control different aspects of the wavefunction.

## Choice of the Slater exponent

The range is proportional to 1/β for an exponential in the interelectronic distance. β enters the explicitly correlated MP2 and coupled-cluster terms. At fixed orbital and complementary bases, HF+CABS singles is determined by its Fock matrix, and the unscaled conventional triples contribution uses ordinary integrals. Cardinal extrapolation has its own exponent. The following geminal parameters are in inverse bohr, for the fixed-amplitude 3C(FIX) ansatz. They are starting points for matching a basis and an active space.

| Orbital family and electron space | DZ | TZ | QZ | 5Z |
| --- | ---: | ---: | ---: | ---: |
| Valence cc-pVnZ-F12 reference | 0.9 | 1.0 | 1.0 | 1.2 |
| cc-pCVnZ-F12, valence-only optimization | 0.9 | 1.0 | 1.0 | — |
| cc-pCVnZ-F12, inner-shell correlation included | 1.4 | 1.4 | 1.5 | — |
| aug-cc-pC(F12)VnZ, inner-shell correlation included | 1.6 | 1.5 | 1.6 | 1.5 |

The last row is a specifically optimized core-valence family. Ordinary diffuse augmentation of a valence basis does not supply its tight radial functions. Hill, Mazumder and Peterson's Table IV distinguishes both active spaces and ansätze. For an inner-shell increment calculated as the difference of two runs at a given cardinal, their tests recommend using the same inner-shell-optimized exponent and core-valence basis for both. Hold the geometry, reference, fitting and core definition fixed as well. The valence-only column describes a stand-alone valence calculation. [Core-valence exponents and tests, Tables IV–V](https://doi.org/10.1063/1.3308483) · [Valence and augmented F12 basis study](https://doi.org/10.1063/1.4998332)

Record which shells are included in the reference. The B–Ne and Al–Ar study and the separate study on Li, Be, Na and Mg provide shell-specific validation targets. For post-d elements with small-core pseudopotentials, the retained outer-core d shell has its own basis and scale requirements. Treat Na, deep all-electron Br shells and the retained I shells as separate convergence cases. Estimate a single-factor, matched-core-valence route first. A pair-specific alternative needs vv, cv and cc exponents and mixed-geminal matrix elements wherever different pair factors meet. The Molpro manual's generic input example uses (1.0, 1.7, 2.5); its Br₂ example uses (0.8, 1.7, 2.2) with ECP10MDF and 3d correlation. Neither fixes a parameter for all-electron Br or ECP28 iodine. Use a declared shell partition followed consistently across displaced geometries. Compare the spin-resolved and restricted reductions independently. [Li, Be, Na and Mg](https://doi.org/10.1039/C0CP00020E) · [Pair-specific exponents](https://doi.org/10.1080/00268976.2010.526641) · [Molpro Br₂ example](https://www.molpro.net/manual/doku.php?id=explicitly_correlated_methods#pair_specific_geminal_exponents) · [Post-d elements](https://doi.org/10.1063/1.4893989)

## Shell and normalization checks

For iodine there are molecular and atomic comparisons with a 28-electron pseudopotential. In the post-d basis paper, a single exponent of 1.4 was used on both sides of the outer-core d increment. The X40×10 halogen study used 1.5 specifically for core–valence pairs alongside the cardinal-dependent valence value. The later augmented PP-F12 atomic electron-affinity study used 1.0 on both sides. With its augmented TZ basis the iodine electron affinity changed by +0.42 kcal/mol on adding 4d correlation and by +0.21 kcal/mol on adding the diffuse functions. These shifts measure two separate changes to that atomic energy difference. Reproduce the published shell partitions as controls and separately test the additional 4s/4p shells retained by a fully active I ECP28 calculation. The Br references in those papers used a small-core PP, whereas an all-electron Br test retains deeper occupied shells. [Post-d molecular study](https://doi.org/10.1063/1.4893989) · [X40×10, computational details](https://doi.org/10.1021/acs.jpca.7b10958) · [Augmented PP-F12 study, Table VII](https://doi.org/10.1063/5.0197210)

The original inner-shell study tests N₂ 1s and P₂ 2s/2p contributions explicitly. A P₂ outer-core test alone does not establish convergence of a deep all-electron 1s contribution. Pair-specific optima also vary between elements with nominally similar roles. For example, the post-d study reports approximate valence/outer-core scales of (0.9, 1.6, 2.2) for As₂ and (0.65, 1.0, 1.2) for Bi₂ with its selected pseudopotentials. [Core-valence study, figures 3–4](https://doi.org/10.1063/1.3308483) · [Post-d study, figure 4](https://doi.org/10.1063/1.4893989)

Record the normalization as well as β. The Psi4 1.11 six-Gaussian input uses exponents proportional to β² and fixed coefficients. A literature convention with the same fixed cusp amplitudes uses −exp(−β r₁₂)/β. Our radial provider exposes both conventions by name. The first fixed-frame tensor tests reproduce precisely the independently replayed Psi4 inputs. Identifying a production convention requires matched end-to-end energies at multiple β values. [Psi4 1.11 correlation factor](https://github.com/psi4/psi4/blob/v1.11/psi4/src/psi4/libmints/mintshelper.cc) · [Explicitly correlated coupled-cluster treatment](https://doi.org/10.1063/1.4963748)

## Charged and open-shell references

Use the published valence cardinal prescription as the initial matched β for the a/ha basis comparison on anions. The augmented-F12 study explicitly tested electron affinities and anionic complexes. Its improvement concerns the flexibility of the one-particle basis. An ion's net charge alone gives no assignment of a new pair exponent. Check that the same electronic state remains bound and is followed at both displaced geometries. [Augmented-F12 study, computational details and anionic benchmarks](https://doi.org/10.1063/1.4998332)

For a spin-unrestricted reference, a fixed-amplitude F12 theory must represent the exchange of the spatial orbitals in an opposite-spin pair while preserving total spin projection. Tew and Klopper recommend the spin-projected fixed-amplitude ansatz of Bokhan and coworkers. The resulting spin-flipped geminal terms must be included when the alpha and beta orbital spaces differ. Test a one-electron doublet with zero pair correlation, a state with singly occupied and doubly occupied orbitals with a nonzero opposite-spin exchange term, and a spin-polarized state with only same-spin pairs. Check both the closed-shell reduction and agreement with an independent unrestricted F12 reference for energy and nuclear derivatives. A per-spin CABS singles unit test exercises only the one-electron reference correction. [Open-shell F12 analysis](https://doi.org/10.1080/00268970903449388) · [TURBOMOLE MP2-F12 implementation](https://doi.org/10.1002/jcc.21825)

## Allocation and reuse

Allocate one GPU to each independent combination of geometry, basis family, cardinal and method stage. Keep a reference determinant, CABS, density-fitting intermediates and the amplitudes needed by its triples correction together when shared. Different cardinals can start simultaneously. Persist the orbital basis, each fitting and complementary basis, ECP, active-space and spin definition, range and fitting of the correlation factor, geometry and root identity. Archive the stage energy and all Cartesian forces with units and atom order. Independently computed stages must agree on the reference where an energy difference assumes it.

Test HF+CABS, MP2-F12/3C(FIX), CCSD-F12b and a conventional (T) contribution separately. Compare D, T and Q against both [D,T] and [T,Q] extrapolations. Include a small conventional [Q,5] or appropriate independent F12 anchor when the full single-geometry calculation is practicable within 15 hours. Compare augmentation on every atom (a) with augmentation on heavy atoms (ha); allow the choice to differ by stage. Reuse identical finite-basis results in several extrapolations. In algebraically identical MP2 stages that cancel, reuse a single result. Freeze extrapolation coefficients before differentiating, then combine the same coefficient times each energy and each force. Publish actual device time, total GPU-hours, wall-clock critical path, build time and queue time separately.

## Additional W1-F12 terms recorded for reference

The original W1-F12 protocol estimates the inner-shell increment using 1.1 times the conventional CCSD increment at cc-pwCVTZ plus the conventional triples increment at cc-pwCVTZ without f functions. This entails differences of correlated spaces. The scalar relativistic term uses a second-order Douglas–Kroll–Hess CCSD(T) calculation and a matched nonrelativistic calculation at the A′VDZ level. The diagonal Born–Oppenheimer correction (DBOC) uses HF/A′VTZ and nuclear masses. Atomic first-order spin-orbit terms use experimental fine structure in the thermochemical construction. These are additional physical or core-correlation components. In an all-active calculation the same core contribution must be counted only once. Preserve the relativistic model already present in an ECP. [Original W1-F12 specification, section III F](https://doi.org/10.1063/1.3697678)

Published bottom-of-the-well **atomization** energy contributions illustrate the scale in kcal/mol. The examples use the high-level W4 component breakdown. They do not give a per-geometry absolute error or a force error.

| Molecule | Inner-shell | Scalar relativistic | DBOC |
| --- | ---: | ---: | ---: |
| H₂O | +0.38 | −0.27 | +0.13 |
| CH₄ | +1.27 | −0.19 | +0.10 |
| NH₃ | +0.65 | −0.25 | +0.14 |

Inner-shell calculations add correlated occupied orbitals to CCSD and triples work. The scalar correction requires relativistic and nonrelativistic endpoints. The HF DBOC requires electronic response to nuclear displacement and its force adds a further derivative. No measured GPU cost for these additions is assumed. This development phase records them as comparators. [W4 component energies, Table V](https://doi.org/10.1063/1.2348881)
