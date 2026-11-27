#pragma once

#include <algorithm>
#include <cstddef>
#include <initializer_list>

#include "cc/df_ucc_v2_internal.hpp"
#include "posthf/capacity.hpp"

namespace generativeqc::cc::v2_detail {

struct UccResidualCapacity {
  std::size_t non_ppl_peak_bytes{};
  std::size_t ppl_resident_bytes{};  // Includes same-spin packed input and output.
  std::size_t min_ppl_tile_bytes{};  // Additional scratch inside accumulate_df_ppl_tiled.
};

// Scratch in compute_uccsd_residual_device, excluding its caller-owned inputs and outputs.
// Keep the phases in execution order so changes to allocation lifetimes can be audited.
inline UccResidualCapacity ucc_residual_capacity(const SpinBlockedDims& d, std::size_t elem_size) {
  using posthf::checked_add;
  using posthf::checked_mul;
  const auto sum = [](std::initializer_list<std::size_t> counts) {
    std::size_t total = 0;
    for (auto count : counts) total = checked_add(total, count);
    return total;
  };

  const auto oa2 = checked_mul(d.oa, d.oa), ob2 = checked_mul(d.ob, d.ob);
  const auto va2 = checked_mul(d.va, d.va), vb2 = checked_mul(d.vb, d.vb);
  const auto ov_a = checked_mul(d.oa, d.va), ov_b = checked_mul(d.ob, d.vb);
  const auto aa = checked_mul(ov_a, ov_a), bb = checked_mul(ov_b, ov_b);
  const auto ab = checked_mul(ov_a, ov_b);
  const auto doubles = sum({aa, ab, bb});
  const auto oo = checked_add(oa2, ob2), vv = checked_add(va2, vb2);
  const auto ov = checked_add(ov_a, ov_b);
  const auto ooov_a = checked_mul(oa2, ov_a), ooov_b = checked_mul(ob2, ov_b);
  const auto ooov_ab = ab ? checked_mul(oa2, ov_b) : 0;
  const auto ooov_ba = ab ? checked_mul(ob2, ov_a) : 0;

  // j_Q lives to function exit. The x, y and bt arrays coexist until the PPL finishes.
  const auto j_x = checked_mul(d.q, checked_add(1, ov));
  const auto j_x_y = checked_add(j_x, checked_mul(d.q, oo));
  const auto df = checked_add(j_x_y, checked_mul(d.q, vv));

  // Fvv/Foo contractions retain both tau and half-tau. Each permutation makes a pair of
  // temporaries for one spin block. Only the mixed-spin permutation overlaps Foo.
  const auto tau_and_half = checked_mul(2, doubles);
  const auto build_fvv = sum({df, tau_and_half, vv, checked_mul(2, std::max({aa, bb, ab}))});
  const auto build_foo = sum({df, tau_and_half, vv, oo, checked_mul(2, ab)});

  // After half-tau is released, the residual keeps tau, permuted goovv, and permuted
  // t2 (both orientations of the mixed-spin block) through the occupied and virtual ladders.
  const auto permuted_t2 = checked_add(doubles, ab);
  const auto ladder_live = sum({df, doubles, doubles, permuted_t2});
  const auto q_ov_a = checked_mul(d.q, ov_a), q_ov_b = checked_mul(d.q, ov_b);
  const auto singles_scratch = std::max({
      checked_add(q_ov_a, aa), checked_mul(2, q_ov_a), checked_add(ooov_a, aa),
      checked_add(q_ov_b, bb), checked_mul(2, q_ov_b), checked_add(ooov_b, bb),
      checked_add(ooov_ab, ab), ooov_ba});
  const auto fock_squares = checked_add(oo, vv);
  const auto singles = sum({ladder_live, checked_mul(2, fock_squares), ov, singles_scratch});
  const auto doubles_fock = sum({ladder_live, fock_squares, std::max(aa, bb)});
  const auto occupied_ladder = checked_add(ladder_live, std::max({
      aa ? checked_mul(oa2, oa2) : 0, bb ? checked_mul(ob2, ob2) : 0,
      ab ? checked_mul(oa2, ob2) : 0}));

  const auto packed_aa = d.oa >= 2 ? checked_mul(checked_mul(d.oa, d.oa - 1) / 2, va2) : 0;
  const auto packed_bb = d.ob >= 2 ? checked_mul(checked_mul(d.ob, d.ob - 1) / 2, vb2) : 0;
  const auto ppl_resident = (packed_aa || packed_bb || ab)
      ? checked_add(ladder_live, checked_mul(2, std::max(packed_aa, packed_bb))) : 0;
  const auto min_tile = [&](std::size_t panel, std::size_t second_virtual_square) {
    // One transposed tau panel plus two Q-block panels and the transposed second DF factor.
    return d.q && panel ? checked_add(checked_mul(3, panel), second_virtual_square) : 0;
  };
  const auto ppl_tile = std::max({min_tile(packed_aa, va2), min_tile(packed_bb, vb2),
                                  min_tile(ab, vb2)});

  // W_abba and W_baab are allocated even when the mixed-spin contractions are skipped.
  const auto abba = checked_mul(oa2, vb2), baab = checked_mul(ob2, va2);
  const auto ring_blocks = sum({aa, bb, ab, ab, abba, baab});
  const auto ring_build_scratch = std::max({
      aa, bb, ab ? checked_add(std::max(abba, baab), checked_mul(2, ab)) : 0});
  const auto ring_build = sum({j_x_y, doubles, permuted_t2, ring_blocks, aa, bb,
                               ring_build_scratch});
  const auto ring_scatter = sum({j_x, permuted_t2, ring_blocks,
                                 std::max({aa, bb, checked_mul(5, ab)})});

  // Final doubles contractions release the ring and t2 permutations. The mixed-spin
  // branch drops u3 before allocating u4, while retaining u2 for the final kernel.
  const auto final_doubles = checked_add(j_x, std::max({
      sum({checked_mul(3, aa), ooov_a}), sum({checked_mul(3, bb), ooov_b}),
      sum({checked_mul(5, ab), ooov_ab, ooov_ba}),
      sum({checked_mul(6, ab), checked_mul(2, ooov_ba)})}));
  const auto non_ppl_peak = std::max({build_fvv, build_foo, singles, doubles_fock,
                                      occupied_ladder, ring_build, ring_scatter, final_doubles});
  return {checked_mul(non_ppl_peak, elem_size), checked_mul(ppl_resident, elem_size),
          checked_mul(ppl_tile, elem_size)};
}

}  // namespace generativeqc::cc::v2_detail
