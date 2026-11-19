# Orbital g functions in scalar ECP quadrature

The public fitted-UHF chemistry checks and the mathematical domain of a raw
semilocal ECP matrix are separate decisions. A quadrature acting on a g
orbital still projects onto the angular channels listed by the potential.
Br ECP10MDF and I ECP28MDF use projectors through f. Their QZ orbital bases
contain g functions.

The host and CUDA quadratures now stage every term of a real g function in
the order returned by the molecule basis expansion. Its largest expansion
has six normalized Cartesian monomials. Each monomial receives its own
double-factorial normalization and generated value and first center
derivatives before contraction. An explicit shell check protects Cartesian
h inputs whose expansion length is one. Workspace admission accounts for
the expanded staging records.

The orbital regression compares all fifteen Cartesian and nine real spherical
g functions with normalized analytic Cartesian Gaussians and independent
real spherical harmonics. It checks the three nuclear derivatives at two
finite-difference steps. The maximum expansion occurs for the central real
harmonic. Molecular acceptance separately requires per-entry scalar ECP and
nine first-derivative matrices in both nonaxial BrH₂/Q and IH₂/Q geometries.
The two grids and independent signed displacements provide checks for
angular integration and nuclear translation. End-to-end SCF and correlated
nuclear derivatives retain their own qualification.

## Molecular evidence

On an FP64 L20, the finer BrH₂ grid (512 radial, 96 polar) gave maximum
matrix and nuclear derivative differences of 2.32 × 10⁻¹⁰ Eh and
8.34 × 10⁻⁹ Eh/bohr against the screened libcint control. The finer IH₂
grid (224 radial, 44 polar) gave 2.74 × 10⁻¹³ Eh and 4.38 × 10⁻¹² Eh/bohr.
Both full Q orbital bases contain 27 heavy-center g functions. Their
same-center g projections onto the lower-angular projectors vanish.
Use a displaced g center to measure finite g-projector matrix elements and
nuclear derivatives before extrapolating these data to other geometries.
