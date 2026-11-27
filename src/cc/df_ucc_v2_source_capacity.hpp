#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "cc/df_ucc_v2_internal.hpp"
#include "posthf/capacity.hpp"

namespace generativeqc::cc::v2_detail {

// The borrowing consumers' DIIS dot reductions and split-K GEMMs use this
// workspace on the source stream. It matches the Ampere/Ada cuBLAS default
// pool; smaller workspaces force slower non-split-K kernels.
inline constexpr std::size_t kUccBlasWorkspaceBytes = 4ULL << 20;

// Owned device panels and construction scratch, in addition to the borrowed DF plan.
inline std::size_t ucc_source_panel_peak_bytes(const SpinBlockedDims& d) {
  using posthf::checked_add;
  using posthf::checked_mul;
  auto spin_fock_elems = [](std::size_t o, std::size_t v) {
    return checked_add(checked_mul(o, o), checked_add(checked_mul(o, v), checked_mul(v, v)));
  };
  const auto fock_elems = checked_add(spin_fock_elems(d.oa, d.va), spin_fock_elems(d.ob, d.vb));
  const auto eps_elems = checked_add(d.n_act_a(), d.n_act_b());
  const auto c_act_elems = checked_mul(d.n, eps_elems);
  const auto device_base = checked_mul(checked_add(checked_add(fock_elems, eps_elems), c_act_elems),
                                       sizeof(double));
  const auto prep_scratch = checked_mul(checked_mul(d.n, d.n), 4 * sizeof(double));
  const auto max_active = std::max(d.n_act_a(), d.n_act_b());
  const auto row = checked_mul(d.n, d.q);
  const auto t1 = checked_mul(max_active, row);
  const auto t2 = checked_mul(checked_mul(max_active, max_active), d.q);
  const auto scratch = max_active
      ? std::max({checked_mul(checked_add(checked_mul(2, row), t1), sizeof(double)),
                  checked_mul(checked_add(t1, t2), sizeof(double)), checked_mul(t2, 2 * sizeof(double)),
                  checked_add(checked_mul(t2, sizeof(double)), sizeof(int))})
      : 0;
  const auto factor_bytes = checked_mul(checked_mul(d.q, fock_elems), sizeof(double));
  const auto panel_scratch = std::max(prep_scratch, checked_add(factor_bytes, scratch));
  return checked_add(kUccBlasWorkspaceBytes, checked_add(device_base, panel_scratch));
}

inline void require_ucc_source_panel_capacity(std::size_t panel_peak, std::size_t plan_resident,
                                              std::size_t cap) {
  if (plan_resident > cap || panel_peak > cap - plan_resident)
    throw std::length_error("DF-UCC v2 CUDA source panels exceed memory budget");
}

struct UccSourcePullbackCapacity {
  std::size_t serial_bytes{};
  std::size_t extra_fused_bytes{};
};

// Combined source, factor cotangents, outputs and producer scratch, excluding the
// separately bounded nuclear sink. The serial schedule replays the derivative rows.
inline UccSourcePullbackCapacity ucc_source_pullback_capacity(
    const SpinBlockedDims& d, std::size_t resident, std::size_t factor_grad_bytes) {
  using posthf::checked_add;
  using posthf::checked_mul;
  const auto n = d.n, q = d.q;
  const auto metric = checked_mul(q, q), row = checked_mul(n, q);
  const auto fixed = checked_add(checked_add(resident, factor_grad_bytes),
                                 checked_mul(checked_mul(n, n), 2 * sizeof(double)));
  // Retain bar_W and one AO row. During a spin pass T1, T2 and bar_BMO coexist,
  // followed by T1, T2 and both host and device coefficient cotangents.
  const auto row_base = checked_mul(checked_add(metric, row), sizeof(double));
  auto spin_peak = [&](std::size_t active) {
    const auto c = checked_mul(n, active);
    const auto t1 = checked_mul(c, q);
    const auto t2 = checked_mul(checked_mul(active, active), q);
    const auto work = checked_add(checked_add(t1, t2), std::max(t2, checked_mul(2, c)));
    return checked_add(row_base, checked_mul(work, sizeof(double)));
  };
  const auto spin_a = spin_peak(d.n_act_a());
  const auto spin_b = spin_peak(d.n_act_b());
  const auto metric_peak = checked_add(checked_mul(metric, 4 * sizeof(double)), sizeof(int));
  const auto serial_peak = std::max({spin_a, spin_b, metric_peak});
  const auto ua_bytes = checked_mul(checked_mul(row, d.n_act_a()), sizeof(double));
  const auto room_with_beta = serial_peak - spin_b;
  const auto extra_fused = ua_bytes > room_with_beta ? ua_bytes - room_with_beta : 0;
  return {checked_add(fixed, serial_peak), extra_fused};
}

}  // namespace generativeqc::cc::v2_detail
