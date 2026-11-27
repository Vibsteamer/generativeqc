#include "cc/df_ucc_v2_device_capacity.hpp"
#include "cc/df_ucc_v2_triples_capacity.hpp"
#include "cc/df_ucc_v2_triples_host_capacity.hpp"
#include "ucc_v2_source_capacity_cases.hpp"

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

void check_iteration(std::size_t q, std::size_t oa, std::size_t ob, std::size_t va,
                     std::size_t vb, std::size_t non_ppl, std::size_t resident, std::size_t tile) {
  SpinBlockedDims d{};
  d.q = q; d.oa = oa; d.ob = ob; d.va = va; d.vb = vb;
  const auto a = oa * va, b = ob * vb;
  const auto amplitudes = a + b + a * a + a * b + b * b;
  for (const auto width : {sizeof(float), sizeof(double)}) {
    const auto scratch = ucc_residual_capacity(d, width);
    require(scratch.non_ppl_peak_bytes == non_ppl * width, "residual peak bytes");
    require(scratch.ppl_resident_bytes == resident * width, "packed ladder resident bytes");
    require(scratch.min_ppl_tile_bytes == tile * width, "one-Q ladder tile bytes");

    const auto base = 100 + 3 * amplitudes * width;
    const auto minimum = base + std::max(non_ppl, resident + tile) * width;
    require(!ucc_iteration_budget(d, width, amplitudes, base, minimum - 1), "one-byte shortage");
    for (auto extra : {0U, 1U, 1024U, 65536U}) {
      const auto plan = ucc_iteration_budget(d, width, amplitudes, base, minimum + extra);
      require(plan && plan->ppl_tile_bytes >= tile * width, "feasible ladder lost");
      require(base + plan->diis_bytes + non_ppl * width <= minimum + extra, "residual over cap");
      require(base + plan->diis_bytes + resident * width + plan->ppl_tile_bytes <= minimum + extra,
              "packed ladder and DIIS over cap");
    }
  }
}

void test_iterations() {
  // Independent peak inventories for uneven spins, an empty alpha occupied space,
  // and a Q-dominated system. The last keeps all six DF panels at once.
  check_iteration(5, 2, 3, 4, 9, 5721, 4459, 810);
  check_iteration(25, 0, 3, 16, 2, 7852, 7032, 40);
  check_iteration(300, 1, 1, 2, 2, 5764, 4540, 16);
  const SpinBlockedDims empty{};
  require(!ucc_iteration_budget(empty, 8, 0, 0, 7), "empty FP64 energy sum requires eight bytes");
  require(ucc_iteration_budget(empty, 8, 0, 0, 8).has_value(), "empty energy sum fits");
  SpinBlockedDims huge{};
  huge.oa = std::numeric_limits<std::size_t>::max();
  expect_rejection<std::overflow_error>([&] { ucc_residual_capacity(huge, 8); }, "shape overflow");
}

void test_allocator_and_history() {
  expect_rejection<UccDeviceLimitError>([] { UccDeviceCapacityScope excessive(9, 10); },
                                        "source cap rejection");
  UccDeviceCapacityScope scope(100, 30);
  UccDeviceCharge history(40);
  {
    UccDeviceCharge work(30);
    require(!ucc_device_available(100), "all bytes occupied");
    expect_rejection<UccDeviceLimitError>([] { UccDeviceCharge excess(1); }, "over-cap allocation");
    history = std::move(work);
    require(ucc_device_available(100) == 40, "replacement releases only the old owner");
  }
  require(ucc_device_available(100) == 40, "moved-from owner double-credited");
  history.reset();
  require(ucc_device_available(100) == 70, "lease release");
  require(!try_ucc_warmup([&] { UccDeviceCharge excessive(71); }), "warmup cap fallback");
  require(try_ucc_warmup([&] { UccDeviceCharge exact(70); }), "warmup exact bound");
  require(!try_ucc_warmup([] { throw std::bad_alloc(); }), "warmup OOM fallback");
  expect_rejection<std::logic_error>([] {
    try_ucc_warmup([] { throw std::logic_error("calculation error"); });
  }, "optional warmup must propagate calculation errors");
  require(ucc_diis_space(6, 8, 39) == 0, "four vectors cannot extrapolate");
  require(ucc_diis_space(6, 8, 40) == 2, "two pairs and one combination fit exactly");
  require(ucc_diis_space(6, 8, std::numeric_limits<std::size_t>::max()) == 6, "slot narrowing");
}

void test_adjoint_tile() {
  require(ucc_lambda_ppl_q_tile(0, 99, 8, 0) == 0, "empty Lambda ladder");
  expect_rejection<UccDeviceLimitError>([] { ucc_lambda_ppl_q_tile(7, 15, 8, 479); },
                                        "adjoint transposes were omitted");
  require(ucc_lambda_ppl_q_tile(7, 15, 8, 480) == 1, "minimum Lambda tile");
  require(ucc_lambda_ppl_q_tile(7, 15, 8, 1200) == 4, "bounded Lambda Q batch");
}

void test_triples_pages() {
  // Each cross-page row holds four vectors plus two square Gram matrices.
  expect_rejection<std::length_error>([] { ucc_triples_page_rows(5, 11, 367); },
                                     "triples 1-row pages exceed budget");
  require(ucc_triples_page_rows(5, 11, 368) == 1, "minimum triples page");
  require(ucc_triples_page_rows(5, 11, 767) == 1, "second Gram row over budget");
  require(ucc_triples_page_rows(5, 11, 768) == 2, "second Gram row fits");
  require(ucc_triples_page_rows(5, 11, 1079) == 2, "single page too large");
  require(ucc_triples_page_rows(5, 11, 1080) == 5, "single-page layout fits");

  const auto cached = [](std::size_t available) {
    return ucc_triples_cache_slabs<float>({{27, 8, 3}}, true, available);
  };
  expect_rejection<std::length_error>([&] { cached(971); }, "recompute working set too large");
  require(!cached(972) && !cached(2591), "bounded triples recompute");
  require(cached(2592), "resident triples slabs fit");
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
    test_ucc_v2_source_capacity();
    test_iterations();
    test_allocator_and_history();
    test_adjoint_tile();
    test_triples_pages();
    test_finite_partition();
    test_compensated_host_admission();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
