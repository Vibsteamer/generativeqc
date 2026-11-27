#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "api/handles.hpp"
#include "cc/df_triples.hpp"
#include "cc/df_ucc_v2_internal.hpp"
#include "cc/lambda_response.hpp"
#include "generativeqc/ucc_fitting.h"
#include "hf/reference.hpp"
#include "integrals/x2c1e.hpp"
#include "methods/df_rccsdt_method.hpp"
#include "molecule/scalar_hamiltonian_identity.hpp"
#include "posthf/capacity.hpp"

namespace generativeqc::integrals::x2c1e_test_detail {
std::vector<double> eval_r_metric(const std::vector<double>& s, const std::vector<double>& s_nesc,
                                  std::size_t n);
std::vector<double> eval_forward_dr_metric(const std::vector<double>& s,
                                           const std::vector<double>& s_nesc,
                                           const std::vector<double>& ds,
                                           const std::vector<double>& ds_nesc, std::size_t n);
void eval_adjoint_r_metric(const std::vector<double>& s, const std::vector<double>& s_nesc,
                           const std::vector<double>& r_bar, std::size_t n,
                           std::vector<double>& s_bar_out, std::vector<double>& snesc_bar_out);
}  // namespace generativeqc::integrals::x2c1e_test_detail

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

double frob_dot(const std::vector<double>& a, const std::vector<double>& b) {
  require(a.size() == b.size(), "frob_dot size mismatch");
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  require(a.size() == b.size(), "max_abs_diff size mismatch");
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite value in comparison");
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

std::vector<double> matmul_nn(const std::vector<double>& a, const std::vector<double>& b,
                              std::size_t n) {
  std::vector<double> c(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < n; ++k)
      for (std::size_t j = 0; j < n; ++j) c[i * n + j] += a[i * n + k] * b[k * n + j];
  return c;
}

// =============================================================================
// M4 (Option A): Forward vs Reverse FP32-W Telemetry Invariants
// =============================================================================
void test_m4_triples_fp32_w_telemetry() {
  using generativeqc::cc::triples::DFCudaResult;
  {
    DFCudaResult fp64;
    require(!fp64.triples_fp32_w_executed(), "M4: default FP64 must report executed=false");
    require(!fp64.triples_fp32_w_forward_executed(), "M4: default FP64 forward must be false");
    require(!fp64.triples_fp32_w_reverse_executed(), "M4: default FP64 reverse must be false");
  }
  {
    DFCudaResult e_only;
    e_only.w_contraction_compute_bits = 32;
    e_only.w_reverse_contraction_compute_bits = 64;
    e_only.fp32_gemms = 10;
    e_only.fp32_forward_gemms = 10;
    e_only.fp32_reverse_gemms = 0;
    require(e_only.triples_fp32_w_executed(), "M4: E-only FP32-W must report executed=true");
    require(e_only.triples_fp32_w_forward_executed(), "M4: E-only FP32-W forward must be true");
    require(!e_only.triples_fp32_w_reverse_executed(), "M4: E-only FP32-W reverse must be false");
  }
  {
    DFCudaResult ef_fast;
    ef_fast.w_contraction_compute_bits = 32;
    ef_fast.w_reverse_contraction_compute_bits = 32;
    ef_fast.fp32_gemms = 24;
    ef_fast.fp32_forward_gemms = 12;
    ef_fast.fp32_reverse_gemms = 12;
    require(ef_fast.triples_fp32_w_executed(), "M4: E+F FP32-W must report executed=true");
    require(ef_fast.triples_fp32_w_forward_executed(), "M4: E+F FP32-W forward must be true");
    require(ef_fast.triples_fp32_w_reverse_executed(), "M4: E+F FP32-W reverse must be true");
  }
  {
    DFCudaResult bounded;
    bounded.resource_fallback = true;
    bounded.w_contraction_compute_bits = 32;
    // The prepared provider can fall back to generated FP32 W, with FP64
    // panels and reverse contractions. Only dispatched work proves precision.
    require(!bounded.triples_fp32_w_executed(), "M4: selected FP32 without work is not execution");
    bounded.fp32_forward_gemms = bounded.fp32_gemms = 2;
    require(bounded.triples_fp32_w_executed(), "M4: provider fallback retained mixed work");
    require(bounded.triples_fp32_w_forward_executed(), "M4: provider fallback retained FP32 W");
    require(!bounded.triples_fp32_w_reverse_executed(), "M4: bounded reverse W stayed FP64");
  }
  std::cout << "[PASS] M4: triples_fp32_w forward/reverse telemetry invariants\n";
}

// =============================================================================
// M5 (CPU ABI): generativeqc_rcc_f12b_cpu_solve_v1 upfront budget check
// Audit counterexample: o=1, v=128 (vvvv = 2 GiB) with max_bytes = 1 MiB
// must return OUT_OF_MEMORY immediately BEFORE allocating/reading vvvv!
// =============================================================================
void test_m5_cpu_f12b_upfront_budget() {
  double dummy = 0.0;
  double energy_out = 0.0;
  const std::size_t one_mib = 1ULL << 20;
  const auto status = generativeqc_rcc_f12b_cpu_solve_v1(
      /*nocc=*/1, /*nvir=*/128, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy,
      &dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy,
      /*fixed_geminal_energy=*/0.0, /*solve_lambda=*/0, /*max_bytes=*/one_mib, &energy_out, &dummy,
      &dummy, nullptr, nullptr, nullptr, nullptr, nullptr);
  require(status == GENERATIVEQC_STATUS_OUT_OF_MEMORY,
          "M5: generativeqc_rcc_f12b_cpu_solve_v1(o=1,v=128,1MiB) must reject upfront with OOM");
  std::cout << "[PASS] M5: generativeqc_rcc_f12b_cpu_solve_v1 rejects o=1,v=128,1MiB upfront with "
               "OUT_OF_MEMORY (zero 2GiB vvvv allocation)\n";
}

// =============================================================================
// M8 (C++): Fail-closed finite norms in DF-UCCSD v2 & DF-ULambda v2
// Audit counterexample: [0.0, NaN] MUST NOT return 0.0 or falsely converge/skip!
// =============================================================================
void test_m8_fail_closed_nan_norms() {
  using namespace generativeqc::cc::v2_detail;
  SpinBlockedDims d;
  d.n = 2;
  d.q = 1;
  d.oa_full = 1;
  d.ob_full = 0;
  d.fa = 0;
  d.fb = 0;
  d.oa = 1;
  d.ob = 0;
  d.va = 1;
  d.vb = 2;
  d.natoms = 1;

  // Test 1: DF-UCCSD v2 CPU residual contains [0.0, NaN] via fov_a = NaN while energy or other
  // elements are 0.0. Even with oa=1, va=2 so t1a has 2 elements [0.0, NaN]:
  {
    SpinBlockedDims d2 = d;
    d2.n = 3;
    d2.va = 2;
    d2.vb = 3;
    SpinBlockedDFSource src;
    src.dims = d2;
    src.boo_a.assign(d2.q * d2.oa * d2.oa, 0.0);
    src.bov_a.assign(d2.q * d2.oa * d2.va, 0.0);
    src.bvv_a.assign(d2.q * d2.va * d2.va, 0.0);
    src.boo_b.assign(d2.q * d2.ob * d2.ob, 0.0);
    src.bov_b.assign(d2.q * d2.ob * d2.vb, 0.0);
    src.bvv_b.assign(d2.q * d2.vb * d2.vb, 0.0);
    src.foo_a = {-1.0};
    src.fov_a = {0.0, std::numeric_limits<double>::quiet_NaN()};
    src.fvv_a = {1.0, 0.0, 0.0, 2.0};
    src.foo_b = {};
    src.fov_b = {};
    src.fvv_b.assign(d2.vb * d2.vb, 0.0);
    src.eps_oa = {-1.0};
    src.eps_va = {1.0, 2.0};
    src.eps_ob = {};
    src.eps_vb = {1.0, 2.0, 3.0};
    SpinBlockedFourIndexCore core;
    core.allocate_zero(d2);

    generativeqc::cc::DFUccV2Options opts;
    opts.max_ccsd_iterations = 5;
    bool threw = false;
    try {
      (void)solve_df_uccsd_cpu(src, core, opts);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    require(threw, "M8: solve_df_uccsd_cpu must fail closed on [0.0, NaN] residual");
  }

  // Test 2: DF-ULambda v2 CPU RHS contains [0.0, NaN] in triples->bar_t.t1a!
  // Before M8 fix, max_abs_amp(rhs) returned 0.0 and skipped the Lambda solve silently!
  {
    SpinBlockedDims d2 = d;
    d2.n = 3;
    d2.va = 2;
    d2.vb = 3;
    SpinBlockedDFSource src;
    src.dims = d2;
    src.boo_a.assign(d2.q * d2.oa * d2.oa, 0.0);
    src.bov_a.assign(d2.q * d2.oa * d2.va, 0.0);
    src.bvv_a.assign(d2.q * d2.va * d2.va, 0.0);
    src.boo_b.assign(d2.q * d2.ob * d2.ob, 0.0);
    src.bov_b.assign(d2.q * d2.ob * d2.vb, 0.0);
    src.bvv_b.assign(d2.q * d2.vb * d2.vb, 0.0);
    src.foo_a = {-1.0};
    src.fov_a = {0.0, 0.0};
    src.fvv_a = {1.0, 0.0, 0.0, 2.0};
    src.foo_b = {};
    src.fov_b = {};
    src.fvv_b.assign(d2.vb * d2.vb, 0.0);
    src.eps_oa = {-1.0};
    src.eps_va = {1.0, 2.0};
    src.eps_ob = {};
    src.eps_vb = {1.0, 2.0, 3.0};
    SpinBlockedFourIndexCore core;
    core.allocate_zero(d2);
    SpinBlockedUccAmplitudes amp;
    amp.allocate_zero(d2);
    SpinBlockedTriplesResult triples;
    triples.bar_t.allocate_zero(d2);
    triples.bar_t.t1a = {0.0, std::numeric_limits<double>::quiet_NaN()};
    triples.bar_f.allocate_zero(d2);
    triples.bar_b.allocate_zero(d2);
    triples.bar_core.allocate_zero(d2);

    generativeqc::cc::DFUccV2Options opts;
    opts.energy_source_only = true;
    bool threw = false;
    try {
      (void)solve_df_ulambda_cpu(src, core, amp, &triples, opts);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    require(threw, "M8: solve_df_ulambda_cpu must fail closed on [0.0, NaN] RHS (never skip!)");
  }
  std::cout << "[PASS] M8: DF-UCCSD v2 & DF-ULambda v2 fail closed on [0.0, NaN]\n";
}

// =============================================================================
// M16 (C++): Thread-local and Context-scoped Lambda Telemetry (zero race)
// =============================================================================
void test_m16_thread_and_context_lambda_telemetry() {
  using namespace generativeqc::cc;

  // Part 1: 4 concurrent threads hammering thread_local reset/record/read in lockstep
  constexpr int kThreads = 4;
  constexpr int kRounds = 200;
  std::array<generativeqc_context, kThreads> contexts;
  std::barrier sync_point(kThreads);
  std::atomic<bool> ok{true};
  std::vector<std::thread> threads;
  for (int tid = 0; tid < kThreads; ++tid) {
    threads.emplace_back([&, tid]() {
      const std::string tag = "thread_" + std::to_string(tid);
      for (int r = 0; r < kRounds; ++r) {
        generativeqc_cc_reset_last_lambda_telemetry_v1();
        sync_point.arrive_and_wait();
        double scalars[10]{};
        std::uint64_t counts[16]{};
        char reason[64]{};
        if (generativeqc_cc_last_lambda_telemetry_v1(scalars, 10, counts, 16, reason, 64) != 0)
          ok.store(false);
        sync_point.arrive_and_wait();
        LambdaDiagnostic d{};
        d.iterations = static_cast<std::size_t>(100 * (tid + 1) + r);
        d.lambda_residual_norm = 1e-12 * (tid + 1);
        d.df_mixed_handoff_reason = tag.c_str();
        detail::record_last_lambda_diagnostic(d);
        {
          std::lock_guard lock(contexts[tid].mutex);
          detail::capture_last_lambda_telemetry(contexts[tid].lambda_telemetry);
        }
        sync_point.arrive_and_wait();
        const int valid =
            generativeqc_cc_last_lambda_telemetry_v1(scalars, 10, counts, 16, reason, 64);
        if (valid != 1 || counts[0] != static_cast<std::uint64_t>(100 * (tid + 1) + r) ||
            std::abs(scalars[5] - 1e-12 * (tid + 1)) > 1e-24 || tag != reason) {
          ok.store(false);
        }
        const int peer = (tid + 1) % kThreads;
        if (generativeqc_cc_context_lambda_telemetry_v1(
                &contexts[peer], scalars, 10, counts, 16, reason, 64) != 1 ||
            counts[0] != static_cast<std::uint64_t>(100 * (peer + 1) + r) ||
            reason != "thread_" + std::to_string(peer)) {
          ok.store(false);
        }
        sync_point.arrive_and_wait();
      }
    });
  }
  for (auto& t : threads) t.join();
  require(ok.load(), "M16: concurrent threads must never clobber thread_local Lambda telemetry");

  // Part 2: Interleaved contexts on the same thread keep isolated telemetry
  generativeqc_context ctx_a, ctx_b;
  generativeqc_cc_reset_context_lambda_telemetry_v1(&ctx_a);
  generativeqc_cc_reset_context_lambda_telemetry_v1(&ctx_b);

  LambdaDiagnostic da{}, db{};
  da.iterations = 11;
  da.lambda_residual_norm = 1e-11;
  da.df_mixed_handoff_reason = "ctx_a_reason";
  db.iterations = 22;
  db.lambda_residual_norm = 2e-11;
  db.df_mixed_handoff_reason = "ctx_b_reason";
  detail::record_last_lambda_diagnostic(da);
  detail::capture_last_lambda_telemetry(ctx_a.lambda_telemetry);
  detail::record_last_lambda_diagnostic(db);
  detail::capture_last_lambda_telemetry(ctx_b.lambda_telemetry);

  generativeqc_cc_reset_context_lambda_telemetry_v1(&ctx_b);
  double sa[10]{}, sb[10]{};
  std::uint64_t ca[16]{}, cb[16]{};
  char ra[64]{}, rb[64]{};
  require(generativeqc_cc_context_lambda_telemetry_v1(&ctx_b, sb, 10, cb, 16, rb, 64) == 0,
          "M16: reset ctx_b must read as invalid");
  require(generativeqc_cc_context_lambda_telemetry_v1(&ctx_a, sa, 10, ca, 16, ra, 64) == 1,
          "M16: ctx_a must survive ctx_b update and reset");
  require(ca[0] == 11 && std::abs(sa[5] - 1e-11) < 1e-24 && std::string(ra) == "ctx_a_reason",
          "M16: ctx_a telemetry mismatch");
  // Live diagnostic slots survive unrelated contexts, including reset and reuse.
  std::vector<generativeqc_context> others(1025);
  for (auto& other : others) {
    detail::capture_last_lambda_telemetry(other.lambda_telemetry);
    generativeqc_cc_reset_context_lambda_telemetry_v1(&other);
    detail::capture_last_lambda_telemetry(other.lambda_telemetry);
  }
  require(generativeqc_cc_context_lambda_telemetry_v1(&ctx_a, nullptr, 0, ca, 1, nullptr, 0) == 1 &&
              ca[0] == da.iterations,
          "M16: a live context must retain its result under unrelated context churn");

  // Reconstruct the opaque handle at the identical address without an explicit reset.
  std::optional<generativeqc_context> storage(std::in_place);
  detail::capture_last_lambda_telemetry(storage->lambda_telemetry);
  require(generativeqc_cc_context_lambda_telemetry_v1(&*storage, nullptr, 0, cb, 1, nullptr, 0) == 1 &&
              cb[0] == db.iterations,
          "M16: the previous owner must have a published result");
  auto& fresh = storage.emplace();
  require(generativeqc_cc_context_lambda_telemetry_v1(&fresh, nullptr, 0, nullptr, 0, nullptr, 0) == 0,
          "M16: a new context at the previous address must start empty");
  std::cout << "[PASS] M16: thread and context telemetry survive concurrency, churn and reuse\n";
}

// =============================================================================
// M17: Rank-Reduced DLU-X2C Subspace Rotation Derivative
// =============================================================================
void test_m17_rank_reduced_x2c_r_derivative() {
  using namespace generativeqc::integrals::x2c1e_test_detail;

  // Test 1: Audit's EXACT 2x2 counterexample:
  // S(t) = U(t) diag(1, 1e-15) U(t)^T, S_nesc(t) = S(t), U(t) = 2D rotation
  // At t = 0: R'(0) = [[0, 1], [1, 0]] (off-diagonals MUST equal 1.0, NOT 0.0!).
  {
    const double eps = 1e-15;
    auto build_s = [&](double t) {
      const double c = std::cos(t), s = std::sin(t);
      const std::vector<double> u = {c, -s, s, c};
      const std::vector<double> ut = {c, s, -s, c};
      const std::vector<double> d = {1.0, 0.0, 0.0, eps};
      return matmul_nn(matmul_nn(u, d, 2), ut, 2);
    };
    const auto s0 = build_s(0.0);
    const std::vector<double> ds0 = {0.0, 1.0 - eps, 1.0 - eps, 0.0};
    const auto r0 = eval_r_metric(s0, s0, 2);
    require(max_abs_diff(r0, {1.0, 0.0, 0.0, 0.0}) < 1e-14, "M17: 2x2 R(0) mismatch");

    const auto dr = eval_forward_dr_metric(s0, s0, ds0, ds0, 2);
    const std::vector<double> expected_dr = {0.0, 1.0, 1.0, 0.0};
    const double err_exact = max_abs_diff(dr, expected_dr);
    require(err_exact < 1e-14, "M17: 2x2 R'(0) off-diagonals must equal 1.0!");

    const double h = 1e-5;
    const auto rp = eval_r_metric(build_s(h), build_s(h), 2);
    const auto rm = eval_r_metric(build_s(-h), build_s(-h), 2);
    std::vector<double> dr_fd(4);
    for (std::size_t k = 0; k < 4; ++k) dr_fd[k] = (rp[k] - rm[k]) / (2.0 * h);
    const double err_fd = max_abs_diff(dr, dr_fd);
    require(err_fd < 1e-10, "M17: 2x2 forward dR must match central FD to < 1e-10");

    // Verify adjoint against forward with asymmetric r_bar
    const std::vector<double> r_bar = {0.3, -1.7, 2.9, 0.5};
    std::vector<double> s_bar, snesc_bar;
    eval_adjoint_r_metric(s0, s0, r_bar, 2, s_bar, snesc_bar);
    const double fwd_ip = frob_dot(r_bar, dr);
    const double adj_ip = frob_dot(s_bar, ds0) + frob_dot(snesc_bar, ds0);
    require(std::abs(fwd_ip - adj_ip) < 1e-14, "M17: 2x2 adjoint must match forward dot product");
    std::cout << "[PASS] M17 (2x2 audit counterexample): R'(0)_01=" << dr[1]
              << " R'(0)_10=" << dr[2] << " err_exact=" << err_exact << " err_fd=" << err_fd
              << " adj_diff=" << std::abs(fwd_ip - adj_ip) << "\n";
  }

  // Test 2: General 4x4 rank-reduced S (r_s = 2, discarded evals = 4e-15, 1e-15) with
  // NON-TRIVIAL S_nesc != S (nonzero retained-discarded coupling V_1^T S_nesc V_0) and
  // smooth rank-preserving curve S(t) = Q(t) D(t) Q(t)^T, S_nesc(t) = S_nesc(0) + t * dS_nesc!
  {
    const std::size_t n = 4;
    // Skew-symmetric generator A for smooth orthogonal curve U(t) = exp(t * A)
    const std::vector<double> a_skew = {
        0.0, 0.3, -0.7, 0.5,
        -0.3, 0.0, 0.6, -0.4,
        0.7, -0.6, 0.0, 0.2,
        -0.5, 0.4, -0.2, 0.0,
    };
    // Exact Cayley orthogonal curve U(t) = (I + t/2 A)(I - t/2 A)^{-1}, U(0)=I, U'(0)=A.
    // Even simpler: use Taylor series of exp(t A) up to t^8 (exact to machine precision for |t|<=1e-4).
    auto exp_skew = [&](double t) {
      std::vector<double> res(n * n, 0.0), term(n * n, 0.0);
      for (std::size_t i = 0; i < n; ++i) res[i * n + i] = term[i * n + i] = 1.0;
      for (int k = 1; k <= 10; ++k) {
        auto next = matmul_nn(term, a_skew, n);
        for (double& x : next) x *= (t / static_cast<double>(k));
        for (std::size_t idx = 0; idx < n * n; ++idx) res[idx] += next[idx];
        term = std::move(next);
      }
      return res;
    };
    // Fixed base orthogonal matrix Q0 (Householder)
    const double u[4] = {1.0, -2.0, 0.5, 1.5};
    double u2 = 0.0;
    for (double x : u) u2 += x * x;
    std::vector<double> q0(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        q0[i * n + j] = (i == j ? 1.0 : 0.0) - 2.0 * u[i] * u[j] / u2;

    auto transpose_sq = [&](const std::vector<double>& m) {
      std::vector<double> mt(n * n);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) mt[j * n + i] = m[i * n + j];
      return mt;
    };
    auto build_s = [&](double t) {
      const auto q = matmul_nn(exp_skew(t), q0, n);
      const double ev[4] = {1.8 + 0.3 * t, 0.6 - 0.2 * t, 4e-15, 1e-15};
      std::vector<double> s(n * n, 0.0);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          for (std::size_t k = 0; k < n; ++k) s[i * n + j] += q[i * n + k] * ev[k] * q[j * n + k];
      return s;
    };
    const auto s0 = build_s(0.0);
    // Exact analytic S'(0) = A S(0) - S(0) A + Q0 diag(0.3, -0.2, 0, 0) Q0^T
    const auto a_s0 = matmul_nn(a_skew, s0, n);
    const auto s0_a = matmul_nn(s0, a_skew, n);
    const double dev[4] = {0.3, -0.2, 0.0, 0.0};
    std::vector<double> ds(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double diag_part = 0.0;
        for (std::size_t k = 0; k < n; ++k) diag_part += q0[i * n + k] * dev[k] * q0[j * n + k];
        ds[i * n + j] = a_s0[i * n + j] - s0_a[i * n + j] + diag_part;
      }
    (void)transpose_sq;

    // S_nesc = S + symmetric relativistic perturbation with strong cross-subspace coupling
    std::vector<double> snesc0 = s0;
    const double delta_n[16] = {
        0.12, -0.05, 0.08, -0.03,
        -0.05, 0.09, -0.04, 0.06,
        0.08, -0.04, 0.11, -0.02,
        -0.03, 0.06, -0.02, 0.07,
    };
    for (std::size_t k = 0; k < 16; ++k) snesc0[k] += delta_n[k];
    const std::vector<double> dsn = {
        0.05, 0.04, -0.06, 0.08,
        0.04, -0.09, 0.07, -0.03,
        -0.06, 0.07, 0.04, 0.09,
        0.08, -0.03, 0.09, -0.02,
    };

    const auto dr = eval_forward_dr_metric(s0, snesc0, ds, dsn, n);
    const double h = 1e-5;
    std::vector<double> np_mat(16), nm_mat(16);
    for (std::size_t k = 0; k < 16; ++k) {
      np_mat[k] = snesc0[k] + h * dsn[k];
      nm_mat[k] = snesc0[k] - h * dsn[k];
    }
    const auto rp = eval_r_metric(build_s(h), np_mat, n);
    const auto rm = eval_r_metric(build_s(-h), nm_mat, n);
    std::vector<double> dr_fd(16);
    for (std::size_t k = 0; k < 16; ++k) dr_fd[k] = (rp[k] - rm[k]) / (2.0 * h);
    const double err_fd = max_abs_diff(dr, dr_fd);
    require(err_fd < 1e-10, "M17: 4x4 general rank-reduced S forward dR must match FD to < 1e-10");

    // Asymmetric r_bar adjoint check
    const std::vector<double> r_bar = {
        0.4, -0.7, 1.1, -0.3,
        0.9, 0.2, -0.5, 0.8,
        -0.6, 1.3, 0.1, -0.9,
        0.5, -0.4, 0.7, 1.2,
    };
    std::vector<double> s_bar, snesc_bar;
    eval_adjoint_r_metric(s0, snesc0, r_bar, n, s_bar, snesc_bar);
    const double fwd_ip = frob_dot(r_bar, dr);
    const double adj_ip = frob_dot(s_bar, ds) + frob_dot(snesc_bar, dsn);
    require(std::abs(fwd_ip - adj_ip) < 1e-13,
            "M17: 4x4 general rank-reduced adjoint must match forward dot product");
    std::cout << "[PASS] M17 (4x4 general S!=S_nesc rank-reduced): err_fd=" << err_fd
              << " adj_diff=" << std::abs(fwd_ip - adj_ip) << "\n";
  }

  // Test 3: Rank-reduced K (n=3, r_s=3, r_k=2 with fixed discarded K eigenvalue 2e-15)
  {
    const std::size_t n = 3;
    const std::vector<double> sd = {1.5, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.1};
    const double ss[3] = {std::sqrt(1.5), std::sqrt(0.8), std::sqrt(1.1)};
    const double v[3] = {1.0, 2.0, -1.0};
    std::vector<double> qk0(9);
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        qk0[i * 3 + j] = (i == j ? 1.0 : 0.0) - 2.0 * v[i] * v[j] / 6.0;
    const std::vector<double> b_skew = {0.0, 0.4, -0.5, -0.4, 0.0, 0.3, 0.5, -0.3, 0.0};
    auto exp_b = [&](double t) {
      std::vector<double> res(9, 0.0), term(9, 0.0);
      for (std::size_t i = 0; i < 3; ++i) res[i * 3 + i] = term[i * 3 + i] = 1.0;
      for (int k = 1; k <= 10; ++k) {
        auto next = matmul_nn(term, b_skew, 3);
        for (double& x : next) x *= (t / static_cast<double>(k));
        for (std::size_t idx = 0; idx < 9; ++idx) res[idx] += next[idx];
        term = std::move(next);
      }
      return res;
    };
    auto build_k = [&](double t) {
      const auto qk = matmul_nn(exp_b(t), qk0, 3);
      const double ke[3] = {1.4 + 0.2 * t, 0.9 - 0.1 * t, 2e-15};
      std::vector<double> kmat(9, 0.0);
      for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
          for (std::size_t m = 0; m < 3; ++m) kmat[i * 3 + j] += qk[i * 3 + m] * ke[m] * qk[j * 3 + m];
      return kmat;
    };
    const std::vector<double> ds = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const auto k0 = build_k(0.0);
    const auto b_k0 = matmul_nn(b_skew, k0, 3);
    const auto k0_b = matmul_nn(k0, b_skew, 3);
    const double dke[3] = {0.2, -0.1, 0.0};
    std::vector<double> sn0(9, 0.0), dsn(9, 0.0);
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j) {
        double dkp = 0.0;
        for (std::size_t m = 0; m < 3; ++m) dkp += qk0[i * 3 + m] * dke[m] * qk0[j * 3 + m];
        const double dkij = b_k0[i * 3 + j] - k0_b[i * 3 + j] + dkp;
        sn0[i * 3 + j] = ss[i] * k0[i * 3 + j] * ss[j];
        dsn[i * 3 + j] = ss[i] * dkij * ss[j];
      }
    auto build_sn = [&](double t) {
      const auto kt = build_k(t);
      std::vector<double> sn(9);
      for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) sn[i * 3 + j] = ss[i] * kt[i * 3 + j] * ss[j];
      return sn;
    };
    const auto dr = eval_forward_dr_metric(sd, sn0, ds, dsn, n);
    const double h = 1e-5;
    const auto rp = eval_r_metric(sd, build_sn(h), n);
    const auto rm = eval_r_metric(sd, build_sn(-h), n);
    std::vector<double> dr_fd(9);
    for (std::size_t k = 0; k < 9; ++k) dr_fd[k] = (rp[k] - rm[k]) / (2.0 * h);
    const double err_fd = max_abs_diff(dr, dr_fd);
    require(err_fd < 1e-10, "M17: 3x3 rank-reduced K forward dR must match FD to < 1e-10");

    const std::vector<double> r_bar = {0.3, -0.5, 0.8, 0.2, 1.1, -0.4, -0.9, 0.6, 0.7};
    std::vector<double> s_bar, snesc_bar;
    eval_adjoint_r_metric(sd, sn0, r_bar, n, s_bar, snesc_bar);
    const double fwd_ip = frob_dot(r_bar, dr);
    const double adj_ip = frob_dot(s_bar, ds) + frob_dot(snesc_bar, dsn);
    require(std::abs(fwd_ip - adj_ip) < 1e-13,
            "M17: 3x3 rank-reduced K adjoint must match forward dot product");
    std::cout << "[PASS] M17 (3x3 rank-reduced K): err_fd=" << err_fd
              << " adj_diff=" << std::abs(fwd_ip - adj_ip) << "\n";
  }

  // Test 4: End-to-end molecular DLU-X2C with rank-reduced overlap (s_rank < n_prim)!
  // Two H atoms where atom 0 has two nearly-identical s primitives (overlap eigenvalue < 1e-14).
  {
    auto make_sys = [](double z1) {
      generativeqc::core::System sys;
      sys.atoms = {{1, {0.0, 0.0, 0.0}}, {1, {0.0, 0.0, z1}}};
      // Two nearly identical s exponents on atom 0 so S has an eigenvalue ~ 1e-16 < 1e-14!
      sys.shells = {
          {0, 0, {{1.0, 1.0}}},
          {0, 0, {{1.0 + 1e-9, 1.0}}},
          {1, 0, {{0.8, 1.0}}},
      };
      sys.electron_count = 2;
      sys.multiplicity = 1;
      sys.basis_representation = generativeqc::core::kSphericalBasis;
      return sys;
    };
    const double z0 = 1.4;
    const auto sys0 = make_sys(z0);
    // The copied state must retain both eigenspaces after its original owner closes.
    const auto state = *generativeqc::integrals::build_x2c1e_state(
        sys0, generativeqc::integrals::X2c1eMode::Dlu1e, generativeqc::integrals::kSpeedOfLightAu);
    require(state.metric.s_rank < state.n_prim,
            "M17: molecular DLU-X2C test must exercise rank-reduced s_rank < n_prim");
    const std::vector<double> wh = {0.7, -0.3, 0.5, -0.3, 0.9, -0.2, 0.5, -0.2, 0.6};
    const auto grad_adj =
        generativeqc::integrals::contract_weighted_x2c1e_hcore_derivative(state, wh, true);
    const auto dh = generativeqc::integrals::build_x2c1e_hcore_derivatives(state, true);
    // z-coordinate of atom 1 is coord index 5 (atom 1 * 3 + 2)
    double grad_fwd_z1 = 0.0;
    for (std::size_t k = 0; k < 9; ++k) grad_fwd_z1 += wh[k] * dh[5 * 9 + k];

    const double h = 1e-4;
    const auto sp = generativeqc::integrals::build_x2c1e_state(
        make_sys(z0 + h), generativeqc::integrals::X2c1eMode::Dlu1e,
        generativeqc::integrals::kSpeedOfLightAu);
    const auto sm = generativeqc::integrals::build_x2c1e_state(
        make_sys(z0 - h), generativeqc::integrals::X2c1eMode::Dlu1e,
        generativeqc::integrals::kSpeedOfLightAu);
    const double ep = frob_dot(wh, sp->hcore_delta);
    const double em = frob_dot(wh, sm->hcore_delta);
    const double grad_fd_z1 = (ep - em) / (2.0 * h);

    const double err_adj_fd = std::abs(grad_adj[5] - grad_fd_z1);
    const double err_fwd_adj = std::abs(grad_fwd_z1 - grad_adj[5]);
    require(err_adj_fd < 1e-9, "M17: molecular rank-reduced DLU-X2C adjoint must match FD");
    require(err_fwd_adj < 1e-13, "M17: molecular rank-reduced DLU-X2C forward must match adjoint");
    std::cout << "[PASS] M17 (molecular DLU-X2C s_rank=" << state.metric.s_rank << "/"
              << state.n_prim << "): err_adj_fd=" << err_adj_fd
              << " err_fwd_adj=" << err_fwd_adj << "\n";
  }
}

// =============================================================================
// M19: Restore v1 RHF Triples C ABI Common-Fit & All-Electron Contract
// =============================================================================
void test_m19_v1_common_fit_and_all_electron() {
  using namespace generativeqc;

  auto make_diatomic = [](double aux_exp, bool with_ecp) {
    core::System sys;
    sys.atoms = {{with_ecp ? 3 : 1, {0.0, 0.0, 0.0}}, {1, {0.0, 0.0, 1.4}}};
    if (with_ecp) sys.atoms[0].ecp_core = 2;
    sys.shells = {{0, 0, {{aux_exp, 1.0}}}, {1, 0, {{aux_exp, 1.0}}}};
    sys.electron_count = 2;
    sys.multiplicity = 1;
    sys.basis_representation = core::kSphericalBasis;
    return sys;
  };

  core::ContextState ctx_state;
  ctx_state.requested_backend = core::kCudaBackend;
  ctx_state.executed_backend = core::kCudaBackend;
  ctx_state.device_id = 0;

  methods::Capabilities caps{};
  generativeqc_system aux_handle;
  generativeqc_method_descriptor desc{};
  desc.struct_size = sizeof(desc);
  desc.method = GENERATIVEQC_METHOD_DF_RCCSD_T;
  desc.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_FULL_CUDA;
  desc.density_fitting_auxiliary_basis = &aux_handle;
  desc.precision_mode = GENERATIVEQC_PRECISION_FP64;

  auto dummy_rhf = std::make_shared<hf::PhysicalReference>();

  // A missing snapshot stops an admitted identity before launching any backend work.
  auto v2_accepts_identity = [](methods::PreparedCalculation& plan,
                                const molecule::ScalarHamiltonianIdentity& orbital,
                                const molecule::ScalarHamiltonianIdentity& jk) {
    try {
      (void)methods::detail::focal_rcc_triples_values(
          plan, {}, orbital, jk, 0, methods::detail::FocalRccTriplesOptions{});
    } catch (const methods::MethodError& e) {
      return e.status() == GENERATIVEQC_STATUS_NOT_IMPLEMENTED &&
             std::string(e.what()).find("retained RHF snapshot") != std::string::npos;
    }
    return false;
  };

  // Case 1: Split fit (owner JK != plan correlation fit) -> v1 MUST reject, v2 MUST allow past v1 gate!
  {
    const auto orbital = make_diatomic(1.0, false);
    const auto corr_fit = make_diatomic(2.0, false);
    const auto jk_fit = make_diatomic(3.5, false);
    auto plan =
        methods::detail::prepare_df_rccsdt_calculation(caps, ctx_state, orbital, desc, &corr_fit);
    const molecule::ScalarHamiltonianIdentity orb_id(orbital);
    const molecule::ScalarHamiltonianIdentity jk_id(jk_fit);

    bool v1_rejected_split = false;
    try {
      (void)methods::detail::focal_rcc_triples_values(*plan, dummy_rhf, orb_id, jk_id, 0, false);
    } catch (const std::invalid_argument& e) {
      v1_rejected_split = std::string(e.what()).find("common JK") != std::string::npos;
    }
    require(v1_rejected_split, "M19: v1 energy must reject split JK/correlation fit");

    bool v1_occ_rejected_split = false;
    try {
      (void)methods::detail::focal_rcc_triples_values(*plan, dummy_rhf, orb_id, jk_id, 0, true);
    } catch (const std::invalid_argument& e) {
      v1_occ_rejected_split = std::string(e.what()).find("common JK") != std::string::npos;
    }
    require(v1_occ_rejected_split, "M19: v1 occupied must reject split JK/correlation fit");

    require(v2_accepts_identity(*plan, orb_id, jk_id), "M19: v2 must retain split-fit support");
  }

  // Case 2: ECP system with common fit -> v1 MUST reject, v2 MUST allow past v1 gate!
  {
    const auto orbital = make_diatomic(1.0, true);
    auto common_fit = make_diatomic(2.0, true);
    common_fit.atoms[0].ecp_core = 0;
    common_fit.electron_count = 4;
    auto plan =
        methods::detail::prepare_df_rccsdt_calculation(caps, ctx_state, orbital, desc, &common_fit);
    const molecule::ScalarHamiltonianIdentity orb_id(orbital);
    const molecule::ScalarHamiltonianIdentity jk_id(common_fit);

    bool v1_rejected_ecp = false;
    try {
      (void)methods::detail::focal_rcc_triples_values(*plan, dummy_rhf, orb_id, jk_id, 0, false);
    } catch (const std::invalid_argument& e) {
      v1_rejected_ecp = std::string(e.what()).find("all-electron") != std::string::npos;
    }
    require(v1_rejected_ecp, "M19: v1 must reject ECP systems");

    require(v2_accepts_identity(*plan, orb_id, jk_id), "M19: v2 must retain scalar ECP support");
  }
  std::cout << "[PASS] M19: v1 enforces common-fit & all-electron while v2 retains split-fit & ECP\n";
}

// =============================================================================
// m5: Lambda mixed inner relative tolerance bounds consistency
// =============================================================================
void test_m5_minor_lambda_clamp_bounds() {
  using namespace generativeqc::cc;
  LambdaOptions opts;
  opts.df_mixed_precision = true;
  opts.df_mixed_precision_inner_relative_tolerance = 0.05;  // Audit counterexample (> 0.01!)
  bool rejected = false;
  try {
    validate_lambda_options(opts);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "m5: mixed_inner_relative=0.05 must be rejected by validate_lambda_options");

  opts.df_mixed_precision_inner_relative_tolerance = 1e-2;  // Boundary == hi
  validate_lambda_options(opts);
  opts.df_mixed_precision_inner_relative_tolerance = 5e-7;  // Default
  validate_lambda_options(opts);
  std::cout << "[PASS] m5: validate_lambda_options rejects 0.05 and accepts <= 1e-2\n";
}

// =============================================================================
// m7: X2C speed_of_light <= 0 sentinel & focal_df_reference OOM status mapping
// =============================================================================
void test_m7_x2c_sentinel_and_oom_mapping() {
  generativeqc_context ctx;
  ctx.state.requested_backend = generativeqc::core::kCpuReferenceBackend;
  ctx.state.executed_backend = generativeqc::core::kCpuReferenceBackend;

  generativeqc_system sys;
  sys.data.atoms = {{1, {0.0, 0.0, 0.0}}, {1, {0.0, 0.0, 1.4}}};
  sys.data.shells = {{0, 0, {{1.0, 1.0}}}, {1, 0, {{1.0, 1.0}}}};
  sys.data.electron_count = 2;
  sys.data.multiplicity = 1;
  sys.data.basis_representation = generativeqc::core::kSphericalBasis;

  // m7 Part 1: speed_of_light 0 selects CODATA 137.03599967994; negative and non-finite
  // values are rejected (matching header).
  require(generativeqc_system_set_x2c1e_v1(&ctx, &sys, 1, 0.0) == GENERATIVEQC_STATUS_SUCCESS,
          "m7: speed_of_light=0.0 must select CODATA default");
  require(sys.data.x2c1e_speed_of_light == generativeqc::integrals::kSpeedOfLightAu,
          "m7: speed_of_light=0.0 must store CODATA kSpeedOfLightAu");
  require(
      generativeqc_system_set_x2c1e_v1(&ctx, &sys, 1, -1.0) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
      "m7: negative speed_of_light must fail with INVALID_ARGUMENT");
  require(generativeqc_system_set_x2c1e_v1(&ctx, &sys, 1, std::numeric_limits<double>::quiet_NaN()) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "m7: NaN speed_of_light must fail with INVALID_ARGUMENT");
  require(generativeqc_system_set_x2c1e_v1(&ctx, &sys, 1, -std::numeric_limits<double>::infinity()) ==
              GENERATIVEQC_STATUS_INVALID_ARGUMENT,
          "m7: -Inf speed_of_light must fail with INVALID_ARGUMENT");

  generativeqc_system aux = sys;
  // The 2-AO restricted output has 20 matrix elements, 3 vector elements and a 3-value header.
  const std::vector<double> untouched(26, -37.0);
  auto out = untouched;
  ctx.last_detail.clear();
  ctx.last_detail.shrink_to_fit();
  const auto detail_capacity = ctx.last_detail.capacity();
  const auto check_oom = [&](std::size_t budget) {
    generativeqc_focal_reference* owner = nullptr;
    const auto status = generativeqc_focal_df_reference_owned_v1(
        &ctx, &sys, &aux, 1, 0, budget, nullptr, 0, out.data(), out.size(), &owner);
    require(status == GENERATIVEQC_STATUS_OUT_OF_MEMORY, "m7: focal budget must return OOM");
    require(owner == nullptr && out == untouched, "m7: over-budget result was published");
    require(!ctx.last_detail.empty() && ctx.last_detail.capacity() == detail_capacity,
            "m7: OOM reporting must fit the available detail buffer");
  };
  check_oom(16);
  const auto diis_history = generativeqc::scf::ScfOptions{}.diis_history;
  const auto expansion = generativeqc::posthf::uhf_reference_capacity(sys.data, diis_history);
  require(generativeqc::posthf::rhf_reference_capacity(sys.data, diis_history) < expansion,
          "m7: test budget must admit the underlying RHF reference");
  check_oom(expansion - 1);
  std::cout << "[PASS] m7: X2C sentinel and focal export/spin-expansion budgets\n";
}

}  // namespace

int main() {
  test_m4_triples_fp32_w_telemetry();
  test_m5_cpu_f12b_upfront_budget();
  test_m8_fail_closed_nan_norms();
  test_m16_thread_and_context_lambda_telemetry();
  test_m17_rank_reduced_x2c_r_derivative();
  test_m19_v1_common_fit_and_all_electron();
  test_m5_minor_lambda_clamp_bounds();
  test_m7_x2c_sentinel_and_oom_mapping();
  std::cout << "ALL AUDIT FIX-2 GATE 1 CHECKS PASSED!\n";
  return 0;
}
