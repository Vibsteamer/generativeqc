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
The molecular CUDA library now includes a general off-center kernel using
the through-i AO polynomial and radial factorization of the earlier
three-center experiment. The host stages basis rows, ECP records and fixed
quadrature constants. Each GPU radial tile evaluates AO values and electron
gradients in the laboratory frame, projects all sixteen angular channels with
FP64 cuBLAS, applies the radial potentials and scatters both AO-center and
operator-center derivatives on the device. This direct rule uses every
azimuth of each fixed grid and supports arbitrary atom ordering and multiple
projector centers. The tile accounts for the dense output, a bounded cuBLAS
workspace, and device input/staging. The two independent grid evaluations
must satisfy the same acceptance gates as the CPU reference.

The CUDA translation unit compiled and linked for sm_89. The shared scalar
AO and radial operations were checked on the host against independent libcint
derivatives on a tilted three-center through-i model. Real-device numerical
acceptance uses the molecular CUDA library and the public geometry request.
Complete energy and gradient consumers also require their fitted
two-electron, orbital and response terms.
