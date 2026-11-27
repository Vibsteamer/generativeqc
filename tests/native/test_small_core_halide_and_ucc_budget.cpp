#include <iostream>
#include <stdexcept>
#include <string>

#include "cc/df_ucc_v2_internal.hpp"
#include "molecule/small_core_halide.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

using generativeqc::molecule::SmallCoreHalideComposition;

SmallCoreHalideComposition make_comp(std::initializer_list<std::pair<int, int>> atoms) {
  SmallCoreHalideComposition c;
  for (auto [z, core] : atoms) c.add(z, core);
  return c;
}

void test_small_core_halide_domain() {
  // 1. Validated hydrides and atomic Br/I.
  const auto hbr = make_comp({{35, 10}, {1, 0}});
  require(hbr.fitted_orbital_angular_limit(0, 1, 26) == 4U, "HBr(0,1) should admit l=4");
  require(hbr.fitted_orbital_angular_limit(1, 2, 25) == 4U, "HBr+(1,2) should admit l=4");
  require(hbr.fitted_orbital_angular_limit(0, 3, 26) == 0U, "HBr(0,3) triplet must be rejected");
  require(hbr.fitted_orbital_angular_limit(2, 1, 24) == 0U, "HBr(2+,1) must be rejected");
  require(hbr.fitted_orbital_angular_limit(-2, 1, 28) == 0U, "HBr(2-,1) must be rejected");

  const auto br = make_comp({{35, 10}});
  require(br.fitted_orbital_angular_limit(0, 2, 25) == 4U, "Br(0,2) should admit l=4");
  require(br.fitted_orbital_angular_limit(-1, 1, 26) == 4U, "Br-(1-,1) should admit l=4");
  require(br.fitted_orbital_angular_limit(-2, 2, 27) == 0U, "Br(2-,2) must be rejected");

  const auto brh2 = make_comp({{35, 10}, {1, 0}, {1, 0}});
  require(brh2.fitted_orbital_angular_limit(0, 2, 27) == 4U, "BrH2(0,2) should admit l=4");

  const auto brh3 = make_comp({{35, 10}, {1, 0}, {1, 0}, {1, 0}});
  require(brh3.fitted_orbital_angular_limit(0, 1, 28) == 0U,
          "BrH3(0,1) unvalidated hydride singlet must not fall through to l=4");

  const auto brh4 = make_comp({{35, 10}, {1, 0}, {1, 0}, {1, 0}, {1, 0}});
  require(brh4.fitted_orbital_angular_limit(0, 2, 29) == 3U,
          "BrH4(0,2) must remain capped at l=3");
  require(brh4.fitted_orbital_angular_limit(1, 1, 28) == 0U,
          "BrH4+(1,1) must be rejected");

  // 2. Molecular halides (C/H/Br, Br*H2O, CH3I) reject out-of-spec charges.
  const auto cbr = make_comp({{6, 0}, {35, 10}});
  require(cbr.fitted_orbital_angular_limit(0, 2, 31) == 4U, "CBr(0,2) should admit l=4");
  require(cbr.fitted_orbital_angular_limit(1, 1, 30) == 4U, "CBr+(1,1) should admit l=4");
  require(cbr.fitted_orbital_angular_limit(-2, 2, 33) == 0U, "CBr(2-,2) must be rejected");

  const auto ch3i = make_comp({{6, 0}, {53, 28}, {1, 0}, {1, 0}, {1, 0}});
  require(ch3i.fitted_orbital_angular_limit(0, 1, 34) == 4U, "CH3I(0,1) should admit l=4");
  require(ch3i.fitted_orbital_angular_limit(1, 2, 33) == 4U, "CH3I+(1,2) should admit l=4");
  require(ch3i.fitted_orbital_angular_limit(2, 1, 32) == 0U, "CH3I(2+,1) must be rejected");
  require(ch3i.fitted_orbital_angular_limit(3, 2, 31) == 0U, "CH3I(3+,2) must be rejected");

  const auto br_h2o = make_comp({{35, 10}, {8, 0}, {1, 0}, {1, 0}});
  require(br_h2o.fitted_orbital_angular_limit(0, 2, 35) == 4U, "Br*H2O(0,2) should admit l=4");
  require(br_h2o.fitted_orbital_angular_limit(1, 1, 34) == 4U, "Br*H2O+(1,1) should admit l=4");
  require(br_h2o.fitted_orbital_angular_limit(-2, 2, 37) == 0U, "Br*H2O(2-,2) must be rejected");

  // 3. Validated general small-core Br/I main-group molecules (H..Ar partners, |charge|<=1).
  const auto c4h6br2o = make_comp({{6, 0}, {6, 0}, {6, 0}, {6, 0}, {8, 0},
                                   {35, 10}, {35, 10},
                                   {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}});
  require(c4h6br2o.fitted_orbital_angular_limit(0, 1, 88) == 4U, "C4H6Br2O(0,1) should admit l=4");
  require(c4h6br2o.fitted_orbital_angular_limit(-2, 1, 90) == 0U, "C4H6Br2O(2-,1) must be rejected");
  require(c4h6br2o.fitted_orbital_angular_limit(0, 1, 86) == 0U,
          "C4H6Br2O mismatched electron count must be rejected");

  const auto br2 = make_comp({{35, 10}, {35, 10}});
  require(br2.fitted_orbital_angular_limit(0, 1, 50) == 4U, "Br2(0,1) should admit l=4");
  require(br2.fitted_orbital_angular_limit(2, 1, 48) == 0U, "Br2(2+,1) must be rejected");

  const auto ch2brcl = make_comp({{6, 0}, {1, 0}, {1, 0}, {35, 10}, {17, 0}});
  require(ch2brcl.fitted_orbital_angular_limit(0, 1, 50) == 4U, "CH2BrCl(0,1) should admit l=4");

  // 4. Reject unvalidated heavy all-electron partner atoms (Z > 18) and wrong ECP cores.
  const auto br_au = make_comp({{35, 10}, {79, 0}});
  require(br_au.fitted_orbital_angular_limit(0, 1, 104) == 0U,
          "Br + Au(all-electron Z=79) must be rejected");
  const auto bad_br = make_comp({{35, 2}, {1, 0}});
  require(bad_br.fitted_orbital_angular_limit(0, 1, 34) == 0U,
          "Br with non-10 core must be rejected");
}

void test_cpu_uhf_triples_enforces_max_bytes_all_modes() {
  using namespace generativeqc::cc::v2_detail;
  SpinBlockedDFSource src;
  SpinBlockedFourIndexCore core;
  SpinBlockedUccAmplitudes amp;
  // Dimensions large enough that o * v^3 slabs alone exceed 128 MiB, while
  // max_bytes is capped at 1 MiB. Admission must fail BEFORE allocating slabs.
  src.dims.q = 32;
  src.dims.oa = 4;
  src.dims.ob = 3;
  src.dims.va = 128;
  src.dims.vb = 128;

  for (bool fp32_w : {false, true}) {
    for (bool comp : {false, true}) {
      if (comp && !fp32_w) continue;
      generativeqc::cc::DFUccV2Options opt;
      opt.max_bytes = 1ULL << 20;
      opt.triples_fp32_w = fp32_w;
      opt.triples_compensated_w = comp;
      opt.compute_forces = false;
      bool rejected = false;
      try {
        (void)evaluate_df_ucc_triples_cpu(src, core, amp, opt);
      } catch (const std::length_error&) {
        rejected = true;
      }
      require(rejected, "evaluate_df_ucc_triples_cpu did not reject tight max_bytes upfront");
    }
  }
}

}  // namespace

int main() {
  try {
    test_small_core_halide_domain();
    test_cpu_uhf_triples_enforces_max_bytes_all_modes();
    std::cout << "test_small_core_halide_and_ucc_budget passed\n";
    return 0;
  } catch (const std::exception& err) {
    std::cerr << "test_small_core_halide_and_ucc_budget failed: " << err.what() << "\n";
    return 1;
  }
}
