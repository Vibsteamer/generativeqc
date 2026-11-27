#pragma once

#include "posthf/capacity.hpp"
#include "scf/cuda_density_fitting_eigen.hpp"

namespace generativeqc::cc {

// This bounds the shared streamed J/K owner before its metric setup. It includes
// its conservative lazy SCF reservations even though this consumer never runs
// SCF on that owner. Validate against its reported peak before proceeding.
inline std::size_t df_streamed_metric_setup_bound(std::size_t n, std::size_t q,
                                                  std::size_t source_device,
                                                  std::size_t source_host_peak) {
  using posthf::checked_add;
  using posthf::checked_mul;
  auto count = checked_add(checked_mul(64, checked_mul(n, n)), checked_mul(8, checked_mul(n, q)));
  count = checked_add(count, checked_mul(8, checked_mul(q, q)));
  count = checked_add(count, checked_mul(8, checked_add(n, q)));
  auto peak = checked_add(checked_mul(sizeof(double), count), 1ULL << 20);
  peak = checked_add(peak, checked_mul(4, scf::df_eigen_workspace_allowance(n)));
  peak = checked_add(peak, checked_mul(2, scf::df_eigen_workspace_allowance(q)));
  return checked_add(peak, checked_add(source_device, checked_mul(2, source_host_peak)));
}

}  // namespace generativeqc::cc
