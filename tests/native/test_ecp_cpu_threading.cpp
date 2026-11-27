#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "integrals/ecp.hpp"
#include "integrals/ecp_workers.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_usage.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

generativeqc::core::System ecp_quadrature_fixture() {
  generativeqc::core::System s;
  // Two-atom ECP fixture with an f-projector term (channel = 3) so
  // gaussian_ecp_supported() returns false and BOTH ecp_integrals(!deriv) and
  // contract_weighted_ecp_derivative exercise the multi-threaded quadrature path.
  s.atoms = {
      {35, {0.0, 0.0, 0.0}, 10},
      {1, {0.0, 0.0, 2.7}, 0},
  };
  s.shells = {
      {0, 0, {{4.2, 0.4}, {1.1, 0.6}}},
      {0, 1, {{2.3, 0.5}, {0.65, 0.5}}},
      {0, 2, {{1.4, 1.0}}},
      {1, 0, {{3.42525091, 0.15432897}, {0.62391373, 0.53532814}, {0.16885540, 0.44463454}}},
      {1, 1, {{0.8, 1.0}}},
  };
  s.ecp_terms = {
      {0, -1, 2, 1.85, -3.2},
      {0, 0, 2, 2.40, 14.5},
      {0, 1, 2, 1.95, 9.2},
      {0, 2, 2, 1.50, 4.1},
      {0, 3, 2, 1.25, -1.7},
  };
  s.charge = 0;
  s.multiplicity = 1;
  s.basis_representation = generativeqc::core::kSphericalBasis;
  std::string detail;
  require(generativeqc::molecule::validate_and_normalize(s, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "ECP fixture normalization failed");
  return s;
}

bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return false;
  if (a.empty()) return true;
  return std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

void test_static_partition() {
  using generativeqc::integrals::detail::static_radial_partition;
  for (std::size_t total : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{64},
                            std::size_t{127}}) {
    for (unsigned nthreads : {1U, 2U, 3U, 8U}) {
      std::size_t cursor = 0;
      for (unsigned tid = 0; tid < nthreads; ++tid) {
        const auto [b, e] = static_radial_partition(total, nthreads, tid);
        require(b == cursor && b <= e && e <= total, "static_radial_partition invalid range");
        cursor = e;
      }
      require(cursor == total, "static_radial_partition did not cover [0, total)");
    }
  }
}

void test_worker_exception_propagation() {
  using generativeqc::integrals::detail::run_ecp_workers;

  // 1. std::bad_alloc thrown in a worker thread must propagate cleanly without
  // std::terminate and signal cancellation to peer threads.
  std::atomic<unsigned> started{0};
  std::atomic<bool> saw_cancel{false};
  bool caught_bad_alloc = false;
  try {
    run_ecp_workers(8U, [&](unsigned tid, const std::atomic<bool>& cancelled) {
      started.fetch_add(1U, std::memory_order_relaxed);
      if (tid == 3U) throw std::bad_alloc();
      for (int iter = 0; iter < 1000000; ++iter) {
        if (cancelled.load(std::memory_order_acquire)) {
          saw_cancel.store(true, std::memory_order_relaxed);
          break;
        }
      }
    });
  } catch (const std::bad_alloc&) {
    caught_bad_alloc = true;
  }
  require(caught_bad_alloc, "std::bad_alloc did not propagate from ECP worker thread");
  require(started.load() == 8U, "not all ECP workers ran");
  require(saw_cancel.load(), "peer workers did not observe cancellation flag");

  // 2. Concurrent throws from multiple workers must also cleanly propagate one
  // exception without terminating or leaking joinable threads.
  bool caught_runtime = false;
  try {
    run_ecp_workers(6U, [&](unsigned tid, const std::atomic<bool>&) {
      if (tid % 2U == 0U) throw std::runtime_error("injected worker fault");
    });
  } catch (const std::runtime_error& err) {
    caught_runtime = std::string(err.what()) == "injected worker fault";
  }
  require(caught_runtime, "concurrent worker runtime_error did not propagate cleanly");
}

void test_ecp_bitwise_determinism_and_capacity() {
  const auto sys = ecp_quadrature_fixture();
  const auto n = generativeqc::molecule::ao_count(sys);
  const auto nn = n * n;

  // 1. Multi-threaded ecp_integrals(!derivatives) must be bitwise reproducible
  // across repeated calls and report nthreads-scaled CPU capacity samples.
  generativeqc::runtime::cpu_resource_observation = {};
  generativeqc::runtime::cpu_resource_observation.active = true;
  generativeqc::runtime::cpu_resource_observation.generation =
      generativeqc::runtime::next_cpu_resource_observation_generation();
  const auto ref_vals = generativeqc::integrals::ecp_integrals(sys, 64, 16, false);
  const auto multi_peak_vals = generativeqc::runtime::cpu_resource_observation.peak_bytes;
  const auto multi_samples_vals = generativeqc::runtime::cpu_resource_observation.samples;
  require(multi_samples_vals >= 1U && multi_peak_vals > 0U,
          "ecp_integrals(!derivatives) did not sample CPU capacity");
  for (int rep = 0; rep < 8; ++rep) {
    const auto cur = generativeqc::integrals::ecp_integrals(sys, 64, 16, false);
    require(bitwise_equal(ref_vals.local, cur.local),
            "ecp_integrals local matrix is not bitwise deterministic across repeated runs");
    require(bitwise_equal(ref_vals.nonlocal, cur.nonlocal),
            "ecp_integrals nonlocal matrix is not bitwise deterministic across repeated runs");
  }

  // Single-worker budget limit must agree numerically and record <= multi-worker peak.
  generativeqc::runtime::cpu_resource_observation = {};
  generativeqc::runtime::cpu_resource_observation.active = true;
  generativeqc::runtime::cpu_resource_observation.cpu_worker_limit = 1U;
  generativeqc::runtime::cpu_resource_observation.generation =
      generativeqc::runtime::next_cpu_resource_observation_generation();
  const auto single_vals = generativeqc::integrals::ecp_integrals(sys, 64, 16, false);
  const auto single_peak_vals = generativeqc::runtime::cpu_resource_observation.peak_bytes;
  require(single_peak_vals <= multi_peak_vals,
          "multi-worker ECP value capacity did not account for per-thread buffers");
  for (std::size_t k = 0; k < nn; ++k) {
    require(std::abs(ref_vals.local[k] - single_vals.local[k]) <= 1e-13,
            "multi-worker vs single-worker ECP local mismatch");
    require(std::abs(ref_vals.nonlocal[k] - single_vals.nonlocal[k]) <= 1e-13,
            "multi-worker vs single-worker ECP nonlocal mismatch");
  }

  // 2. Multi-threaded contract_weighted_ecp_derivative must be bitwise
  // reproducible across repeated calls and report nthreads-scaled CPU capacity.
  std::vector<double> weights(nn, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      weights[i * n + j] = 0.1 * std::cos(static_cast<double>(i + 1) * 0.7 +
                                          static_cast<double>(j + 1) * 0.3);
    }
  }
  generativeqc::runtime::cpu_resource_observation = {};
  generativeqc::runtime::cpu_resource_observation.active = true;
  generativeqc::runtime::cpu_resource_observation.generation =
      generativeqc::runtime::next_cpu_resource_observation_generation();
  const auto ref_grad = generativeqc::integrals::contract_weighted_ecp_derivative(sys, weights);
  const auto multi_peak_grad = generativeqc::runtime::cpu_resource_observation.peak_bytes;
  const auto multi_samples_grad = generativeqc::runtime::cpu_resource_observation.samples;
  require(multi_samples_grad >= 1U && multi_peak_grad > 0U,
          "contract_weighted_ecp_derivative did not sample CPU capacity");
  for (int rep = 0; rep < 6; ++rep) {
    const auto cur_grad = generativeqc::integrals::contract_weighted_ecp_derivative(sys, weights);
    require(bitwise_equal(ref_grad, cur_grad),
            "contract_weighted_ecp_derivative is not bitwise deterministic across repeated runs");
  }

  generativeqc::runtime::cpu_resource_observation = {};
  generativeqc::runtime::cpu_resource_observation.active = true;
  generativeqc::runtime::cpu_resource_observation.cpu_worker_limit = 1U;
  generativeqc::runtime::cpu_resource_observation.generation =
      generativeqc::runtime::next_cpu_resource_observation_generation();
  const auto single_grad = generativeqc::integrals::contract_weighted_ecp_derivative(sys, weights);
  const auto single_peak_grad = generativeqc::runtime::cpu_resource_observation.peak_bytes;
  generativeqc::runtime::cpu_resource_observation = {};
  require(single_peak_grad <= multi_peak_grad,
          "multi-worker ECP derivative capacity did not account for per-thread buffers");
  for (std::size_t c = 0; c < ref_grad.size(); ++c) {
    require(std::abs(ref_grad[c] - single_grad[c]) <= 1e-12,
            "multi-worker vs single-worker ECP weighted derivative mismatch");
  }
}

}  // namespace

int main() {
  try {
    test_static_partition();
    test_worker_exception_propagation();
    test_ecp_bitwise_determinism_and_capacity();
    std::cout << "test_ecp_cpu_threading passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_ecp_cpu_threading failed: " << error.what() << "\n";
    return 1;
  }
}
