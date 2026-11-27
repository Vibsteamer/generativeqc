#pragma once

#include <cstddef>

namespace generativeqc::cc::f12_cuda {

#if defined(__CUDACC__)
#define GQC_F12_ENERGY_ELEMENT __host__ __device__
#else
#define GQC_F12_ENERGY_ELEMENT
#endif

// Gradients of the spatial fixed-cusp energy using simultaneous pair-symmetric
// W_vv and C, in the same (i,j,a,b) order as the primal energy.
GQC_F12_ENERGY_ELEMENT inline double energy_gradient_one(
    std::size_t o, std::size_t v, const double* singles_constant, const double* w_vv,
    const double* t1, std::size_t ia) {
  const auto i = ia / v, a = ia % v;
  double sum = singles_constant[ia];
  for (std::size_t j = 0; j < o; ++j) {
    const auto pair = (i * o + j) * v * v;
    for (std::size_t b = 0; b < v; ++b)
      sum += (2.0 * w_vv[pair + a * v + b] - w_vv[pair + b * v + a]) * t1[j * v + b];
  }
  return 2.0 * sum;
}

GQC_F12_ENERGY_ELEMENT inline double energy_gradient_two(
    std::size_t v, const double* w_vv, const double* c_f12, std::size_t ijab) {
  const auto ab = ijab % (v * v);
  const auto swap = ijab - ab + (ab % v) * v + ab / v;
  return (2.0 * c_f12[ijab] - c_f12[swap]) + (2.0 * w_vv[ijab] - w_vv[swap]);
}

#undef GQC_F12_ENERGY_ELEMENT

}  // namespace generativeqc::cc::f12_cuda
