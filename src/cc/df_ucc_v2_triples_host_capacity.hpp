#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

#include "cc/df_ucc_v2_internal.hpp"
#include "posthf/capacity.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::cc::v2_detail {
namespace host_triples_capacity {
using posthf::checked_add;
using posthf::checked_mul;
using tensor::cpu_split3_gemm_workspace_bytes;

inline std::size_t sum(std::initializer_list<std::size_t> items) {
  std::size_t total = 0;
  for (const auto item : items) total = checked_add(total, item);
  return total;
}

inline std::size_t doubles(std::size_t count) { return checked_mul(count, sizeof(double)); }

// A page owns its two FP64 X/Y tensors and an active-occupied bit mask. Rounding
// the mask up to a whole 64-bit word also covers 32-bit vector<bool> word packing.
inline std::size_t page_bytes(std::size_t rows, std::size_t cube) {
  const auto mask = checked_mul(rows / 64 + (rows % 64 != 0), sizeof(std::uint64_t));
  return checked_add(doubles(checked_mul(2, checked_mul(rows, cube))), mask);
}

inline std::size_t forward_gemm_workspace(std::size_t m, std::size_t n, std::size_t k,
                                          bool compensated, bool emulated_fp32) {
  if (compensated) return cpu_split3_gemm_workspace_bytes(m, n, k);
  if (emulated_fp32) {
    return doubles(sum({checked_mul(m, k), checked_mul(k, n), checked_mul(m, n)}));
  }
  return 0;
}

inline std::size_t panel_workspace(std::size_t v, std::size_t pair, std::size_t q,
                                   bool compensated, bool emulated_fp32) {
  return checked_add(doubles(checked_mul(q, v)),
                     forward_gemm_workspace(v, pair, q, compensated, emulated_fp32));
}

// For A[m,k] * B[k,n], bar_A and bar_B have GEMM shapes (m,k,n) and
// (k,n,m). Transposing either result interchanges m and n, which preserves
// the split3 workspace (its row/column storage is symmetric).
inline std::size_t adjoint_workspace(std::size_t m, std::size_t n, std::size_t k,
                                     bool compensated) {
  if (!compensated) return 0;
  return std::max(cpu_split3_gemm_workspace_bytes(m, k, n),
                  cpu_split3_gemm_workspace_bytes(k, n, m));
}

inline std::size_t panel_pullback_workspace(std::size_t v, std::size_t pair, std::size_t q,
                                            bool compensated) {
  const auto packed = doubles(checked_mul(2, checked_mul(q, v)));
  return checked_add(packed, adjoint_workspace(v, pair, q, compensated));
}

inline std::size_t same_spin_bytes(std::size_t o, std::size_t v, std::size_t q, bool forces,
                                   bool compensated, bool emulated_fp32) {
  if (o < 3 || v < 3) return 0;
  const auto vv = checked_mul(v, v);
  const auto cube = checked_mul(v, vv);
  const auto slabs = checked_mul(o, cube);
  const auto setup =
      checked_add(doubles(slabs), panel_workspace(v, vv, q, compensated, emulated_fp32));
  const auto fixed = doubles(sum({checked_mul(forces ? 2 : 1, slabs), checked_mul(9, cube), vv}));
  const auto forward =
      forward_gemm_workspace(v, vv, std::max(v, o), compensated, emulated_fp32);
  std::size_t work = forward;
  if (forces) {
    const auto reverse = adjoint_workspace(v, vv, std::max(v, o), compensated);
    const auto panel_pullback = panel_pullback_workspace(v, vv, q, compensated);
    work = std::max({reverse, panel_pullback, checked_add(page_bytes(o, cube), forward)});
  }
  return std::max(setup, checked_add(fixed, work));
}

inline std::size_t mixed_spin_bytes(std::size_t o1, std::size_t o2, std::size_t v1,
                                    std::size_t v2, std::size_t q, bool forces, bool mirror,
                                    bool compensated, bool emulated_fp32) {
  if (o1 < 2 || !o2 || v1 < 2 || !v2) return 0;
  const auto v11 = checked_mul(v1, v1), v12 = checked_mul(v1, v2), v22 = checked_mul(v2, v2);
  const auto cube = checked_mul(v11, v2), o12 = checked_mul(o1, o2);
  const auto t2 = checked_mul(o12, v12);
  const auto slabs = sum({checked_mul(checked_mul(o1, v1), v11),
                          checked_mul(checked_mul(o1, v1), v22), checked_mul(o2, cube)});
  const auto setup = checked_add(
      doubles(slabs),
      std::max({panel_workspace(v1, v11, q, compensated, emulated_fp32),
                panel_workspace(v1, v22, q, compensated, emulated_fp32),
                panel_workspace(v2, v11, q, compensated, emulated_fp32)}));
  // g21_jk, g12_kj, and t2_12_k_m are retained through the occupied response.
  const auto permuted = sum({checked_mul(o12, checked_mul(v1, o2)),
                             checked_mul(o12, checked_mul(o1, v2)), t2});
  const auto fixed = doubles(sum({checked_mul(forces ? 2 : 1, checked_add(slabs, permuted)),
                                  checked_mul(10, cube), v11, v22}));
  // Every mixed W product has one of these two output shapes. The
  // allocation increases monotonically with its contracted dimension.
  const auto k1 = std::max({v1, v2, o1, o2}), k2 = std::max(v1, o1);
  const auto forward =
      std::max(forward_gemm_workspace(v1, v12, k1, compensated, emulated_fp32),
               forward_gemm_workspace(v11, v2, k2, compensated, emulated_fp32));
  auto work = forward;
  if (forces) {
    const auto reverse = std::max(adjoint_workspace(v1, v12, k1, compensated),
                                  adjoint_workspace(v11, v2, k2, compensated));
    const auto panel_pullback = std::max({panel_pullback_workspace(v1, v11, q, compensated),
                                          panel_pullback_workspace(v1, v22, q, compensated),
                                          panel_pullback_workspace(v2, v11, q, compensated)});
    work = std::max(
        {reverse, panel_pullback, checked_add(page_bytes(std::max(o1, o2), cube), forward)});
  }
  // The ABB caller transposes its two AB inputs and both cotangents outside the channel.
  const auto outer = mirror ? doubles(checked_mul(forces ? 4 : 2, t2)) : 0;
  return checked_add(outer, std::max(setup, checked_add(fixed, work)));
}

inline std::size_t cotangent_elements(const SpinBlockedDims& d) {
  const auto aa = checked_mul(d.oa, d.va), bb = checked_mul(d.ob, d.vb);
  const auto ab = checked_mul(d.oa, d.vb), ba = checked_mul(d.ob, d.va);
  const auto ooa = checked_mul(d.oa, d.oa), oob = checked_mul(d.ob, d.ob);
  const auto fock = sum({ooa, aa, checked_mul(d.va, d.va), oob, bb, checked_mul(d.vb, d.vb)});
  const auto pairs = sum({checked_mul(aa, aa), checked_mul(aa, bb), checked_mul(bb, bb)});
  const auto core = sum({pairs, checked_mul(ooa, ooa), checked_mul(ooa, oob), checked_mul(oob, oob),
                         checked_mul(checked_add(ooa, oob), checked_add(aa, bb)),
                         checked_mul(aa, aa), checked_mul(bb, bb), checked_mul(ab, ab),
                         checked_mul(ba, ba)});
  // All blocks of bar_t, bar_b, bar_core and bar_f are allocated, including zero cotangents.
  return sum({aa, bb, pairs, checked_mul(d.q, fock), core, fock});
}

inline std::size_t retained_elements(const SpinBlockedDFSource& s, const SpinBlockedFourIndexCore& c,
                                     const SpinBlockedUccAmplitudes& t, const DFUccV2Options& opt) {
  std::size_t elements = 0;
  for (const auto* block : {
           &s.boo_a, &s.bov_a, &s.bvv_a, &s.boo_b, &s.bov_b, &s.bvv_b,
           &s.foo_a, &s.fov_a, &s.fvv_a, &s.foo_b, &s.fov_b, &s.fvv_b,
           &s.eps_oa, &s.eps_va, &s.eps_ob, &s.eps_vb,
           &s.raw_three_center, &s.raw_metric, &s.metric_whitener, &s.t2_alpha, &s.t2_beta,
           &s.c_act_alpha, &s.c_act_beta, &c.goovv_aa, &c.goovv_ab, &c.goovv_bb,
           &c.goooo_aa, &c.goooo_ab, &c.goooo_bb, &c.gooov_aa, &c.gooov_bb, &c.gooov_ab, &c.gooov_ba,
           &c.govov_aa, &c.govov_bb, &c.govov_ab, &c.govov_ba, &t.t1a, &t.t1b,
           &t.t2aa, &t.t2ab, &t.t2bb, &opt.occupied_weights_alpha, &opt.occupied_weights_beta}) {
    elements = checked_add(elements, block->capacity());
  }
  if (const auto* f = opt.f12) {
    for (const auto* block : {&f->w_vo_aaaa, &f->w_vo_bbbb, &f->w_vo_abab,
                              &f->w_vo_abba, &f->w_vv_aaaa, &f->w_vv_abab,
                              &f->w_vv_bbbb, &f->c_f12_aaaa, &f->c_f12_abab, &f->c_f12_bbbb}) {
      elements = checked_add(elements, block->capacity());
    }
  }
  return elements;
}

inline std::size_t triples_host_bytes_impl(const SpinBlockedDFSource& src,
                                           const SpinBlockedFourIndexCore& core,
                                           const SpinBlockedUccAmplitudes& amp,
                                           const DFUccV2Options& opt, bool compensated,
                                           bool emulated_fp32) {
  const auto& d = src.dims;
  const bool forces = opt.compute_forces;
  const auto fixed = doubles(sum({retained_elements(src, core, amp, opt), d.oa, d.ob,
                                  forces ? cotangent_elements(d) : 0}));
  const auto channel = std::max({
      same_spin_bytes(d.oa, d.va, d.q, forces, compensated, emulated_fp32),
      same_spin_bytes(d.ob, d.vb, d.q, forces, compensated, emulated_fp32),
      mixed_spin_bytes(d.oa, d.ob, d.va, d.vb, d.q, forces, false, compensated, emulated_fp32),
      mixed_spin_bytes(d.ob, d.oa, d.vb, d.va, d.q, forces, true, compensated, emulated_fp32),
  });
  auto peak = channel;
  if (forces) {
    const auto pair = std::max(checked_mul(d.oa, d.va), checked_mul(d.ob, d.vb));
    peak = std::max(peak, doubles(checked_mul(pair, pair)));  // sequential T2 antisymmetrizations
  }
  return checked_add(fixed, peak);
}

}  // namespace host_triples_capacity

// Peak numeric host storage owned by the spin-blocked triples stage across all
// CPU W-contraction modes (FP64, emulated FP32, and compensated FP32).
inline std::size_t ucc_triples_host_bytes(const SpinBlockedDFSource& src,
                                          const SpinBlockedFourIndexCore& core,
                                          const SpinBlockedUccAmplitudes& amp,
                                          const DFUccV2Options& opt) {
  return host_triples_capacity::triples_host_bytes_impl(
      src, core, amp, opt, opt.triples_compensated_w,
      !opt.triples_compensated_w && opt.triples_fp32_w);
}

inline void admit_ucc_triples_host(const SpinBlockedDFSource& src,
                                   const SpinBlockedFourIndexCore& core,
                                   const SpinBlockedUccAmplitudes& amp,
                                   const DFUccV2Options& opt) {
  const auto bytes = ucc_triples_host_bytes(src, core, amp, opt);
  if (opt.max_bytes && bytes > opt.max_bytes)
    throw std::length_error("DF-UCC triples exceed the host memory budget");
}

inline std::size_t compensated_ucc_triples_bytes(const SpinBlockedDFSource& src,
                                                 const SpinBlockedFourIndexCore& core,
                                                 const SpinBlockedUccAmplitudes& amp,
                                                 const DFUccV2Options& opt) {
  return host_triples_capacity::triples_host_bytes_impl(src, core, amp, opt, true, false);
}

inline void admit_compensated_ucc_triples(const SpinBlockedDFSource& src,
                                          const SpinBlockedFourIndexCore& core,
                                          const SpinBlockedUccAmplitudes& amp,
                                          const DFUccV2Options& opt) {
  const auto bytes = compensated_ucc_triples_bytes(src, core, amp, opt);
  if (opt.max_bytes && bytes > opt.max_bytes)
    throw std::length_error("compensated DF-UCC triples exceed the host memory budget");
}

}  // namespace generativeqc::cc::v2_detail
