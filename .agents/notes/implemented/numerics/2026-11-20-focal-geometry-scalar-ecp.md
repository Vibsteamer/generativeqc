# Geometry-owned scalar ECP in focal OBS and OptRI

The focal one-electron CPU reference constructs the residual projector directly
from the orbital element's immutable scalar ECP record, the actual nuclear
positions, and the same normalized OBS-first shells and real solid harmonics as
the bare S/H kernel. The bare Coulomb attraction uses the nuclear charge less
the removed core. The supported ECP channel structure includes the Stuttgart
Br ECP10MDF and I ECP28MDF, whose residual local term is zero and whose
nonlocal projectors extend through f. The orbital and OptRI shells extend
through i.

For each AO center, rotate the quadrature's polar axis along its displacement
from the ECP center. At fixed polar cosine the shifted radial Gaussian is
independent of azimuth. The lab-frame orbital polynomial and its derivative
times a real f projector have maximum Fourier order ten. A 24-point uniform
azimuth rule integrates each such trigonometric polynomial. Binomial
translation preserves the original lab-frame spherical AO row and phase.
The radial map, residual radial integrand and normalized projector harmonics
come from the existing ECP generator graphs. A 384/80 and 512/96 radial/polar
comparison uses the ECP policy's absolute matrix and derivative gates.

The electron-position derivative of each AO projector contributes with
negative sign to its AO center and positive sign to the moving ECP center.
The transpose supplies the other AO index. Co-located AO and ECP centers
on the same atom cancel at scattering. This covers arbitrary orientations,
coincident OBS/OptRI owners and distinct centers in noncollinear hydrides.

The ordinary molecular ECP quadrature has a different orbital angular domain.
Two historical through-i hydride kernels require fixed center or axial
layouts. The experimental CUDA focal request still requires a combined-basis
device kernel for this residual. The CPU entry uses its explicitly selected
backend. Its matrices contain the geometry derivative of the one-electron
Hamiltonian; complete energy and gradient consumers also require their
fitted two-electron, orbital and response terms.
