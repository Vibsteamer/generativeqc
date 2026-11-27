#pragma once

#include <limits>
#include <stdexcept>

#include "cc/df_ucc_v2_source_capacity.hpp"

inline void test_ucc_v2_source_capacity() {
  using namespace generativeqc::cc::v2_detail;
  const auto check_source_peak = [](const SpinBlockedDims& dims, std::size_t panel_doubles) {
    constexpr std::size_t blas_bytes = kUccBlasWorkspaceBytes;
    constexpr std::size_t plan_bytes = 512;
    const auto peak = ucc_source_panel_peak_bytes(dims);
    const auto cap = plan_bytes + blas_bytes + panel_doubles * sizeof(double);
    if (peak != cap - plan_bytes) throw std::runtime_error("incorrect DF source construction peak");
    require_ucc_source_panel_capacity(peak, plan_bytes, cap);
    try {
      require_ucc_source_panel_capacity(peak, plan_bytes, cap - 1);
    } catch (const std::length_error&) {
      return;
    }
    throw std::runtime_error("short DF source budget was admitted");
  };
  SpinBlockedDims d{};
  d.n = 4;
  d.q = 2;
  d.oa = 2;
  d.va = 2;
  d.ob = 1;
  d.vb = 3;
  // The MO panels retain 65 doubles and 50 factor doubles. Rotating T1 into T2
  // needs two 32-double intermediates simultaneously.
  check_source_peak(d, 65 + 50 + 64);
  // Serial VJP needs two 4x4 outputs, one 2x2 metric, one 4x2 row and
  // three 32-double intermediates. Retaining alpha U adds another 32 doubles.
  auto plan = ucc_source_pullback_capacity(d, 100, 200);
  if (plan.serial_bytes != 1420 || plan.extra_fused_bytes != 256)
    throw std::runtime_error("incorrect DF source sequential/fused pullback bounds");
  d.n = 3;
  d.q = 32;
  d.oa = 1;
  d.va = 2;
  d.ob = 0;
  d.vb = 3;
  // Both spins retain virtual panels. The rotation holds two 288-double intermediates.
  check_source_peak(d, 40 + 512 + 576);
  plan = ucc_source_pullback_capacity(d, 0, 0);
  // The separate metric VJP holds four QxQ matrices and its finite flag. Beta
  // has zero occupied orbitals but retains three virtual orbitals in the transform.
  if (plan.serial_bytes != (18 + 4 * 32 * 32) * sizeof(double) + sizeof(int) ||
      plan.extra_fused_bytes != 0)
    throw std::runtime_error("incorrect empty-occupation DF source pullback bound");
  d.q = std::numeric_limits<std::size_t>::max();
  try {
    (void)ucc_source_pullback_capacity(d, 0, 0);
  } catch (const std::overflow_error&) {
    return;
  }
  throw std::runtime_error("overflowing DF source metric capacity was admitted");
}
