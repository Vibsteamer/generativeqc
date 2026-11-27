#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "cc/lambda_krylov.hpp"

namespace {
using namespace generativeqc;
using namespace generativeqc::response;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

struct MatOwner {
  std::size_t dim{};
  std::vector<double> a64;
  std::vector<double> a32;
  bool mixed{}, active{}, suspended{}, finished{}, nonfinite{};
  std::size_t warm_actions{}, strict_actions{}, precision_switches{};
  std::size_t passes_recorded{}, strict_ref_recorded{};
  const char* reason_recorded{};

  bool begin_mixed_precision() {
    if (!mixed) return false;
    require(!active && !suspended && !finished, "double begin_mixed_precision");
    active = true;
    return true;
  }
  void suspend_mixed_precision() {
    require(active && !finished, "suspend requires active mixed epoch");
    active = false;
    suspended = true;
  }
  void resume_mixed_precision() {
    require(suspended && !active && !finished, "resume requires suspended mixed epoch");
    suspended = false;
    active = true;
  }
  void finish_mixed_precision() {
    if (!active && !suspended) return;
    active = false;
    suspended = false;
    finished = true;
    ++precision_switches;
  }
  void record_mixed_summary(std::size_t passes, std::size_t strict_ref, const char* reason) noexcept {
    passes_recorded = passes;
    strict_ref_recorded = strict_ref;
    reason_recorded = reason;
  }
  std::size_t mixed_gcr_history_limit() const { return 2; }

  void apply(std::span<const double> x, std::span<double> y) {
    if (!active && nonfinite) throw std::logic_error("strict retry after nonfinite mixed operator");
    if (active)
      ++warm_actions;
    else
      ++strict_actions;
    const auto& mat = active ? a32 : a64;
    for (std::size_t i = 0; i < dim; ++i) {
      double acc = 0.0;
      for (std::size_t j = 0; j < dim; ++j) acc += mat[i * dim + j] * x[j];
      y[i] = acc;
    }
    if (active && nonfinite) y[0] = std::numeric_limits<double>::quiet_NaN();
  }
};

struct RunReport {
  std::vector<double> solution;
  double residual{};
  std::size_t fp32_iters{};
  std::size_t fp64_iters{};
  std::size_t fp32_actions{};
  std::size_t fp64_actions{};
  std::size_t switches{};
  std::size_t passes{};
  std::string reason;
};

RunReport run_case(std::size_t dim, const std::vector<double>& a64,
                   const std::vector<double>& a32, const std::vector<double>& rhs,
                   const std::vector<double>& independent, bool mixed,
                   bool nonfinite = false) {
  cc::LambdaOptions options;
  options.df_mixed_precision = mixed;
  options.max_bytes = 1ULL << 30;
  options.gmres.absolute_tolerance = 1e-12;
  options.gmres.relative_tolerance = 0.0;
  options.gmres.max_iterations = 200;
  options.gmres.restart = 30;
  options.gmres.true_residual_every = 30;
  MatOwner owner;
  owner.dim = dim;
  owner.a64 = a64;
  owner.a32 = a32;
  owner.mixed = mixed;
  owner.nonfinite = nonfinite;
  const LinearResponseProblem response_problem(dim, [&](auto x, auto y) { owner.apply(x, y); });
  const auto response_plan = prepare_response(response_problem, options.gmres);
  const auto warm = cc::detail::prepare_lambda_warm_start(owner, options, response_problem, rhs,
                                                         independent);
  const auto solved =
      solve_response(response_plan, response_problem, rhs, warm.solution, independent);
  require(!owner.active && !owner.suspended, "mixed epoch not sealed");
  require(owner.finished == mixed, "mixed epoch finish mismatch");
  require(solved.converged() && solved.residual_norm <= 1e-12, "strict residual did not converge");
  const auto total_actions = solved.operator_actions + warm.operator_actions + owner.strict_ref_recorded;
  require(total_actions == owner.warm_actions + owner.strict_actions,
          "operator_actions accounting mismatch");
  RunReport rep;
  rep.solution.assign(solved.solution.begin(), solved.solution.end());
  rep.residual = solved.residual_norm;
  rep.fp32_iters = warm.iterations;
  rep.fp64_iters = solved.iterations;
  rep.fp32_actions = owner.warm_actions;
  rep.fp64_actions = owner.strict_actions;
  rep.switches = owner.precision_switches;
  rep.passes = owner.passes_recorded;
  rep.reason = owner.reason_recorded ? owner.reason_recorded : "not_requested";
  return rep;
}

std::vector<double> make_matrix(std::size_t n, double coupling, double min_diag, double max_diag) {
  std::vector<double> a(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const double t = n > 1 ? double(i) / double(n - 1) : 0.0;
    a[i * n + i] = -(min_diag + t * (max_diag - min_diag));
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) continue;
      // Non-symmetric banded + smooth coupling mimicking coupled-cluster J^T
      const double dist = std::abs(double(i) - double(j));
      const double sign = (i < j) ? 1.0 : -0.7;
      a[i * n + j] = coupling * sign / (1.0 + 0.35 * dist * dist);
    }
  }
  return a;
}

std::vector<double> perturb_matrix(const std::vector<double>& a64, double rel_noise) {
  std::vector<double> a32(a64.size());
  for (std::size_t k = 0; k < a64.size(); ++k) {
    const double s = std::sin(0.173 * double(k + 1) + 0.31);
    a32[k] = a64[k] * (1.0 + rel_noise * s);
  }
  return a32;
}

}  // namespace

int main() {
  // Case 1: Well-conditioned coupled-cluster-like non-symmetric system (n=48, FP32 noise 2e-7)
  const std::size_t n1 = 48;
  const auto a1_64 = make_matrix(n1, 0.32, 0.6, 8.0);
  const auto a1_32 = perturb_matrix(a1_64, 2e-7);
  std::vector<double> rhs1(n1), diag1(n1);
  for (std::size_t i = 0; i < n1; ++i) {
    rhs1[i] = std::cos(0.41 * double(i + 1));
    diag1[i] = a1_64[i * n1 + i];
  }
  const auto r1_fp64 = run_case(n1, a1_64, a1_32, rhs1, diag1, false);
  const auto r1_mix = run_case(n1, a1_64, a1_32, rhs1, diag1, true);
  double max_diff1 = 0.0;
  for (std::size_t i = 0; i < n1; ++i)
    max_diff1 = std::max(max_diff1, std::abs(r1_mix.solution[i] - r1_fp64.solution[i]));
  require(max_diff1 <= 1e-12, "Case 1 mixed vs FP64 parity exceeded 1e-12");
  require(r1_mix.switches == 1, "Case 1 precision switches != 1");
  require(r1_mix.fp64_actions <= 3 && r1_fp64.fp64_actions >= 12,
          "Case 1 did not achieve strong FP64 action reduction");

  // Case 2: Ill-conditioned anion-like system (small gap 0.08, strong coupling, FP32 noise 3e-6)
  const std::size_t n2 = 40;
  const auto a2_64 = make_matrix(n2, 0.22, 0.08, 12.0);
  const auto a2_32 = perturb_matrix(a2_64, 3e-6);
  std::vector<double> rhs2(n2), diag2(n2);
  for (std::size_t i = 0; i < n2; ++i) {
    rhs2[i] = std::sin(0.29 * double(i + 1)) / (1.0 + 0.05 * double(i));
    diag2[i] = a2_64[i * n2 + i];
  }
  const auto r2_fp64 = run_case(n2, a2_64, a2_32, rhs2, diag2, false);
  const auto r2_mix = run_case(n2, a2_64, a2_32, rhs2, diag2, true);
  double max_diff2 = 0.0;
  for (std::size_t i = 0; i < n2; ++i)
    max_diff2 = std::max(max_diff2, std::abs(r2_mix.solution[i] - r2_fp64.solution[i]));
  require(max_diff2 <= 1e-11, "Case 2 ill-conditioned parity exceeded 1e-11");
  require(r2_mix.residual <= 1e-12, "Case 2 final residual > 1e-12");
  require(r2_mix.fp64_actions < r2_fp64.fp64_actions / 2,
          "Case 2 did not reduce FP64 actions by >2x");

  // Case 3: Severe FP32 operator corruption (60% relative error) -> monotonic stagnation handoff
  const auto a3_32 = perturb_matrix(a2_64, 0.65);
  const auto r3_mix = run_case(n2, a2_64, a3_32, rhs2, diag2, true);
  double max_diff3 = 0.0;
  for (std::size_t i = 0; i < n2; ++i)
    max_diff3 = std::max(max_diff3, std::abs(r3_mix.solution[i] - r2_fp64.solution[i]));
  require(max_diff3 <= 1e-11, "Case 3 stagnation handoff parity failed");
  require(r3_mix.residual <= 1e-12, "Case 3 stagnation handoff residual failed");
  require(r3_mix.reason == "stagnated_handoff_fp64", "Case 3 did not report stagnation handoff");

  // Case 4: Nonfinite mixed operator must throw and never silently retry strict
  bool threw_nonfinite = false;
  try {
    (void)run_case(n1, a1_64, a1_32, rhs1, diag1, true, true);
  } catch (const std::runtime_error& error) {
    threw_nonfinite = std::string(error.what()).find("nonfinite") != std::string::npos;
  }
  require(threw_nonfinite, "Case 4 nonfinite mixed action did not throw");

  std::printf(
      "{\"case1\":{\"fp64_baseline_actions\":%zu,\"mixed_fp64_actions\":%zu,"
      "\"mixed_fp32_actions\":%zu,\"mixed_fp32_iters\":%zu,\"mixed_fp64_iters\":%zu,"
      "\"passes\":%zu,\"max_diff\":%.3e,\"residual\":%.3e,\"reason\":\"%s\"},"
      "\"case2\":{\"fp64_baseline_actions\":%zu,\"mixed_fp64_actions\":%zu,"
      "\"mixed_fp32_actions\":%zu,\"mixed_fp32_iters\":%zu,\"mixed_fp64_iters\":%zu,"
      "\"passes\":%zu,\"max_diff\":%.3e,\"residual\":%.3e,\"reason\":\"%s\"},"
      "\"case3\":{\"mixed_fp64_actions\":%zu,\"passes\":%zu,\"max_diff\":%.3e,"
      "\"residual\":%.3e,\"reason\":\"%s\"}}\n",
      r1_fp64.fp64_actions, r1_mix.fp64_actions, r1_mix.fp32_actions, r1_mix.fp32_iters,
      r1_mix.fp64_iters, r1_mix.passes, max_diff1, r1_mix.residual, r1_mix.reason.c_str(),
      r2_fp64.fp64_actions, r2_mix.fp64_actions, r2_mix.fp32_actions, r2_mix.fp32_iters,
      r2_mix.fp64_iters, r2_mix.passes, max_diff2, r2_mix.residual, r2_mix.reason.c_str(),
      r3_mix.fp64_actions, r3_mix.passes, max_diff3, r3_mix.residual, r3_mix.reason.c_str());
  return 0;
}
