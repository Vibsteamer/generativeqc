#include "cc/df_ucc_v2_triples_host_capacity.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using namespace generativeqc::cc::v2_detail;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Error, class F>
void expect_rejection(F action, const char* message) {
  try {
    action();
  } catch (const Error&) {
    return;
  }
  throw std::runtime_error(message);
}

void test_compensated_host_admission() {
  SpinBlockedDFSource src;
  SpinBlockedFourIndexCore core;
  SpinBlockedUccAmplitudes amplitudes;
  generativeqc::cc::DFUccV2Options opt;
  src.dims.q = 65;
  src.dims.oa = 1;
  src.dims.ob = 2;
  src.dims.va = 3;
  src.dims.vb = 2;
  // Only ABB can run. Its two transposed inputs coexist with V11/V12/V21
  // slabs, one packed occupied slice and the split GEMM workspace.
  const auto energy = compensated_ucc_triples_bytes(src, core, amplitudes, opt);
  const auto ab = src.dims.oa * src.dims.ob * src.dims.va * src.dims.vb;
  const auto panel_setup = 2 * ab + 2 * 2 * 2 * 2 + 2 * 2 * 3 * 3 + 3 * 2 * 2 + 65 * 2;
  const auto minimum = (src.dims.oa + src.dims.ob + panel_setup) * sizeof(double) +
      generativeqc::tensor::cpu_split3_gemm_workspace_bytes(2, 9, 65);
  require(energy >= minimum, "mirrored mixed-spin panels and active SGEMM exceed the bound");

  src.raw_three_center.reserve(71);
  require(compensated_ucc_triples_bytes(src, core, amplitudes, opt) ==
              energy + src.raw_three_center.capacity() * sizeof(double),
          "borrowed source vector capacity missing from host admission");
  opt.compute_forces = true;
  opt.max_bytes = compensated_ucc_triples_bytes(src, core, amplitudes, opt);
  // Reverse DF-panel work publishes bar_Bvv[q,v_other^2], with both
  // packed Q*v slices, the complete cotangents and the outer ABB transpose live.
  const auto factor_bars = 65 * (1 + 3 + 9 + 4 + 4 + 4);
  const auto amplitude_bars = 3 + 4 + 9 + 12 + 16;
  const auto core_bars = 37 + 21 + 35 + 65;  // oovv, oooo, ooov, ovov
  const auto force_outputs = 3 + factor_bars + 25 + amplitude_bars + core_bars;
  const auto work = 4 * ab + 2 * (64 + 28) + 10 * 12 + 4 + 9 + 2 * 65 * 2;
  const auto response_minimum = (src.raw_three_center.capacity() + force_outputs + work) *
                                    sizeof(double) +
                                generativeqc::tensor::cpu_split3_gemm_workspace_bytes(65, 9, 2);
  require(opt.max_bytes >= response_minimum,
          "DF-factor adjoint SGEMM exceeds the compensated response bound");
  admit_compensated_ucc_triples(src, core, amplitudes, opt);
  --opt.max_bytes;
  expect_rejection<std::length_error>([&] { admit_compensated_ucc_triples(src, core, amplitudes, opt); },
                                      "admitted an insufficient compensated host budget");
  src.dims.va = std::numeric_limits<std::size_t>::max();
  expect_rejection<std::overflow_error>([&] { admit_compensated_ucc_triples(src, core, amplitudes, opt); },
                                        "admitted overflowing compensated host dimensions");

  // M8: non-compensated modes (FP64 and emulated FP32) must also enforce max_bytes
  // before allocating o * v^3 scratch buffers.
  src.dims = {};
  src.dims.q = 32;
  src.dims.oa = 4;
  src.dims.ob = 3;
  src.dims.va = 64;
  src.dims.vb = 64;
  src.raw_three_center.clear();
  src.raw_three_center.shrink_to_fit();
  for (bool fp32_w : {false, true}) {
    for (bool forces : {false, true}) {
      generativeqc::cc::DFUccV2Options mode_opt;
      mode_opt.triples_compensated_w = false;
      mode_opt.triples_fp32_w = fp32_w;
      mode_opt.compute_forces = forces;
      const auto needed = ucc_triples_host_bytes(src, core, amplitudes, mode_opt);
      const auto slab_floor = src.dims.oa * src.dims.va * src.dims.va * src.dims.va * sizeof(double);
      require(needed >= slab_floor, "host triples capacity omitted o*v^3 slabs");
      mode_opt.max_bytes = needed;
      admit_ucc_triples_host(src, core, amplitudes, mode_opt);
      mode_opt.max_bytes = needed - 1;
      expect_rejection<std::length_error>(
          [&] { admit_ucc_triples_host(src, core, amplitudes, mode_opt); },
          "non-compensated host triples admitted insufficient max_bytes");
    }
  }
}

void test_finite_partition() {
  generativeqc::cc::DFUccV2Options opts;
  opts.occupied_weights_alpha = {0., -2.};
  opts.occupied_weights_beta = {3.};
  std::array alpha{41., 3.};
  const std::array beta{2.};
  validate_ucc_triples_partition(0., alpha, beta, opts);
  expect_rejection<std::runtime_error>([&] { validate_ucc_triples_partition(1., alpha, beta, opts); },
                                       "nonclosing partition");
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  expect_rejection<std::runtime_error>([&] { validate_ucc_triples_partition(nan, alpha, beta, opts); },
                                       "nonfinite triples energy");
  for (auto value : {nan, std::numeric_limits<double>::infinity()}) {
    alpha[0] = value;
    expect_rejection<std::runtime_error>([&] { validate_ucc_triples_partition(0., alpha, beta, opts); },
                                         "nonfinite occupied output with zero weight");
  }
}

}  // namespace

int main() {
  try {
    test_finite_partition();
    test_compensated_host_admission();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
