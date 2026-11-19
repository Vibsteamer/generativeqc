#pragma once

#include <cstddef>

#ifdef __CUDACC__
#define GENERATIVEQC_UHF_HD __host__ __device__
#else
#define GENERATIVEQC_UHF_HD
#endif

namespace generativeqc::posthf::uhf_response {
struct Spin {
  std::size_t occupied, virtuals;
  const double *vo, *oo, *vv, *gap;
  GENERATIVEQC_UHF_HD std::size_t rotations() const { return occupied * virtuals; }
};

GENERATIVEQC_UHF_HD inline Spin spin(const double* data, std::size_t o, std::size_t v,
                                     std::size_t q) {
  if (!o) return {0, v, nullptr, nullptr, nullptr, nullptr};
  const double* oo = data + q * v * o;
  const double* vv = oo + q * o * o;
  return {o, v, data, oo, vv, vv + q * v * v};
}

// Real canonical DF-UHF in virtual-major, occupied-minor rotation order.
// Coulomb couples both spins through the same auxiliary basis; exchange is spin diagonal.
GENERATIVEQC_UHF_HD inline double element(Spin alpha, Spin beta, std::size_t naux, std::size_t row,
                                          std::size_t col) {
  const bool rb = row >= alpha.rotations(), cb = col >= alpha.rotations();
  const Spin r = rb ? beta : alpha, c = cb ? beta : alpha;
  const auto ir = row - (rb ? alpha.rotations() : 0), ic = col - (cb ? alpha.rotations() : 0);
  const auto a = ir / r.occupied, i = ir % r.occupied, b = ic / c.occupied, j = ic % c.occupied;
  double value = row == col ? r.gap[ir] : 0.;
  for (std::size_t q = 0; q < naux; ++q) {
    value += 2. * r.vo[q * r.rotations() + ir] * c.vo[q * c.rotations() + ic];
    if (rb == cb) {
      value -=
          r.vv[(q * r.virtuals + a) * r.virtuals + b] *
              r.oo[(q * r.occupied + i) * r.occupied + j] +
          r.vo[(q * r.virtuals + a) * r.occupied + j] * r.vo[(q * r.virtuals + b) * r.occupied + i];
    }
  }
  return value;
}
}  // namespace generativeqc::posthf::uhf_response
#undef GENERATIVEQC_UHF_HD
