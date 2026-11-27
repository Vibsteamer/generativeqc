#pragma once

#include "integrals/eri_geometry.hpp"

#ifdef __CUDACC__
#define GQC_F12_HD __host__ __device__
#else
#define GQC_F12_HD
#endif

namespace generativeqc::integrals {

// The exponent z of exp(-z r12²) is in bohr^-2. A Slater fit supplies
// several z values and coefficients. The last operator also represents the
// cross gradient of two Gaussians after multiplication by 4*z1*z2 at z1+z2.
enum class GaussianGeminalKernel { gaussian, gaussian_over_distance, distance_squared_gaussian };

/** Hermite seeds M_n = (-d/dT)^n M_0, in the ordinary ERI prefactor convention. */
template <unsigned MaximumOrder>
GQC_F12_HD bool gaussian_geminal_moments(unsigned order, double t, double rho, double z,
                                         GaussianGeminalKernel kernel, double* moments) {
  if (!moments || order > MaximumOrder || !range_isfinite(t) || t < 0. || !range_isfinite(rho) ||
      rho <= 0. || !range_isfinite(z) || z < 0.)
    return false;
  const bool rho_is_larger = rho >= z;
  const double largest = rho_is_larger ? rho : z;
  const double ratio = (rho_is_larger ? z : rho) / largest;
  const double large_fraction = 1. / (1. + ratio);
  const double small_fraction = ratio * large_fraction;
  const double g = rho_is_larger ? small_fraction : large_fraction;  // z/(rho+z)
  const double h = rho_is_larger ? large_fraction : small_fraction;  // rho/(rho+z)
  const double decay = range_exp(-g * t);
  if (kernel == GaussianGeminalKernel::gaussian_over_distance) {
    // Integrate the Coulomb Gaussian transform after u = s/sqrt(rho+z+s²).
    // The resulting moment is h*exp(-gT) times the integral of
    // (g+h*u²)^n * exp(-h*T*u²) on [0,1]. Build its binomial average with
    // positive coefficients and a single ordinary Boys vector for all n.
    double boys[MaximumOrder + 1];
    if (!range_detail::long_moments(order, h * t, 1., boys)) return false;
    for (unsigned n = 0; n <= order; ++n) {
      moments[n] = h * decay * boys[0];
      for (unsigned k = 0; k < order - n; ++k) boys[k] = g * boys[k] + h * boys[k + 1];
    }
    return true;
  }
  if (kernel != GaussianGeminalKernel::gaussian &&
      kernel != GaussianGeminalKernel::distance_squared_gaussian)
    return false;
  const double inverse = large_fraction / largest;
  const double base = 0.5 * range_sqrt(0x1.921fb54442d18p+1) *
                      (range_sqrt(large_fraction) / range_sqrt(largest)) * h * decay;
  double power = 1., previous = 0.;
  for (unsigned n = 0; n <= order; ++n) {
    // Multiplication by r12² is exactly -d/dz on the primitive Gaussian.
    moments[n] = kernel == GaussianGeminalKernel::gaussian
                     ? base * power
                     : base * ((1.5 + h * t) * power - n * h * previous) * inverse;
    if (!range_isfinite(moments[n])) return false;
    previous = power;
    power *= g;
  }
  return true;
}

template <typename Geometry, unsigned MaximumOrder, unsigned UnitGaussianSlots = 0>
GQC_F12_HD bool make_geminal_geometry(const double* exponents, const double* centers,
                                      unsigned order, GaussianGeminalKernel kernel, double z,
                                      Geometry& geometry) {
  double argument;
  return bind_eri_centers<Geometry, MaximumOrder, UnitGaussianSlots>(exponents, centers, order,
                                                                     geometry, argument) &&
         gaussian_geminal_moments<MaximumOrder>(order, argument, geometry.rho, z, kernel,
                                                geometry.boys);
}

}  // namespace generativeqc::integrals

#undef GQC_F12_HD
