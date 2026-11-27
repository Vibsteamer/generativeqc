#include <iostream>
#include <stdexcept>
#include <string>

#include "cc/df_ucc_v2_internal.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
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
    test_cpu_uhf_triples_enforces_max_bytes_all_modes();
    std::cout << "test_small_core_halide_and_ucc_budget passed\n";
    return 0;
  } catch (const std::exception& err) {
    std::cerr << "test_small_core_halide_and_ucc_budget failed: " << err.what() << "\n";
    return 1;
  }
}
