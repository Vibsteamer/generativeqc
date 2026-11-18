#pragma once

#include "integrals/f12_geometry.hpp"

#ifdef __CUDACC__
#define GQC_STG_HD __host__ __device__
#else
#define GQC_STG_HD
#endif

namespace generativeqc::integrals {

enum class SlaterFitNormalization { negative_exponential, negative_exponential_over_beta };
enum class GeminalProduct { factor, factor_over_distance, product, gradient_product };

struct FittedSlater {
  double exponent[6];
  double coefficient[6];
};

/** Prepare a geometry-independent six-Gaussian fit once per Slater range.
 * Raw coefficients follow Psi4 1.11 MintsHelper::f12_cgtg. The second
 * convention represents -exp(-beta*r)/beta with the same radial fit.
 * The primitive Gaussian exponents have units bohr^-2, beta has bohr^-1. */
GQC_STG_HD inline bool make_fitted_slater(double beta, SlaterFitNormalization normalization,
                                          FittedSlater& fit) {
  if (!range_isfinite(beta) || beta <= 0. ||
      (normalization != SlaterFitNormalization::negative_exponential &&
       normalization != SlaterFitNormalization::negative_exponential_over_beta))
    return false;
  constexpr double exponents[] = {0.22085085450735284, 1.0040191632019282, 3.6212173098378728,
                                  12.162483236221904,  45.855332448029337, 254.23460688554644};
  constexpr double coefficients[] = {-0.31442480597241274,  -0.30369575353387201,
                                     -0.16806968430232927,  -0.098115812152857612,
                                     -0.060246640234342785, -0.037263541968504843};
  for (unsigned k = 0; k < 6; ++k) {
    fit.exponent[k] = beta * (beta * exponents[k]);
    fit.coefficient[k] = normalization == SlaterFitNormalization::negative_exponential
                             ? coefficients[k]
                             : coefficients[k] / beta;
    if (!range_isfinite(fit.exponent[k]) || fit.exponent[k] <= 0. ||
        !range_isfinite(fit.coefficient[k]))
      return false;
  }
  return true;
}

/** Contract f, f/r, f1*f2 or grad_1(f1).grad_1(f2) into Hermite seeds.
 * For the two-electron kinetic energy, the latter is one half of
 * [f1,[T1+T2,f2]]. Shared fits may pass the same object for both arguments.
 * The pair multiplicities and geminal normalizations are explicit. */
template <unsigned MaximumOrder>
GQC_STG_HD bool fitted_geminal_moments(unsigned order, double t, double rho, GeminalProduct op,
                                       const FittedSlater& left, const FittedSlater& right,
                                       double* moments) {
  if (!moments || order > MaximumOrder) return false;
  const bool linear = op == GeminalProduct::factor || op == GeminalProduct::factor_over_distance;
  if (!linear && op != GeminalProduct::product && op != GeminalProduct::gradient_product)
    return false;
  const auto kernel =
      op == GeminalProduct::factor_over_distance ? GaussianGeminalKernel::gaussian_over_distance
      : op == GeminalProduct::gradient_product   ? GaussianGeminalKernel::distance_squared_gaussian
                                                 : GaussianGeminalKernel::gaussian;
  const bool symmetric = !linear && &left == &right;
  double sum[MaximumOrder + 1]{}, compensation[MaximumOrder + 1]{}, primitive[MaximumOrder + 1];
  for (unsigned i = 0; i < 6; ++i) {
    for (unsigned j = symmetric ? i : 0; j < (linear ? 1u : 6u); ++j) {
      double z = left.exponent[i], weight = left.coefficient[i];
      if (!linear) {
        z += right.exponent[j];
        weight *= right.coefficient[j];
        if (symmetric && i != j) weight *= 2.;
        if (op == GeminalProduct::gradient_product)
          weight *= 4. * left.exponent[i] * right.exponent[j];
      }
      if (!range_isfinite(weight) ||
          !gaussian_geminal_moments<MaximumOrder>(order, t, rho, z, kernel, primitive))
        return false;
      for (unsigned n = 0; n <= order; ++n) {
        const double addend = weight * primitive[n] - compensation[n];
        const double next = sum[n] + addend;
        compensation[n] = (next - sum[n]) - addend;
        sum[n] = next;
      }
    }
  }
  for (unsigned n = 0; n <= order; ++n) {
    if (!range_isfinite(sum[n])) return false;
    moments[n] = sum[n];
  }
  return true;
}

template <typename Geometry, unsigned MaximumOrder, unsigned UnitGaussianSlots = 0>
GQC_STG_HD bool make_fitted_geminal_geometry(const double* exponents, const double* centers,
                                             unsigned order, GeminalProduct op,
                                             const FittedSlater& left, const FittedSlater& right,
                                             Geometry& geometry) {
  double t;
  return bind_eri_centers<Geometry, MaximumOrder, UnitGaussianSlots>(exponents, centers, order,
                                                                    geometry, t) &&
         fitted_geminal_moments<MaximumOrder>(order, t, geometry.rho, op, left, right,
                                               geometry.boys);
}
}  // namespace generativeqc::integrals

#undef GQC_STG_HD
