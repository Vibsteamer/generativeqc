#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cc/lambda_response.hpp"
#include "response/solve.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/tracked_allocator.hpp"

namespace generativeqc::cc::detail {

/** Match the checked RCCSD size arithmetic used by the native ABI. */
inline std::size_t lambda_size_add(std::size_t a, std::size_t b) {
  std::size_t sum;
  if (!runtime::checked_add(a, b, sum)) throw std::length_error("RCCSD size overflow");
  return sum;
}

struct LambdaWarmStart {
  runtime::TrackedVector<double> solution;
  std::size_t iterations{}, operator_actions{}, preconditioner_actions{};
};

/** Build a mixed-precision proposal for the strict FP64 solve. The owner
 * switches the action precision and supplies its admitted GCR history size.
 * Only the proposal crosses into the strict solver.
 */
template <class ActionOwner>
LambdaWarmStart prepare_lambda_warm_start(ActionOwner& owner, const LambdaOptions& options,
                                         const response::LinearResponseProblem& response_problem,
                                         std::span<const double> rhs,
                                         std::span<const double> diagonal) {
  LambdaWarmStart result;
  if (!owner.begin_mixed_precision()) return result;
  if (options.df_mixed_precision_refinement_passes <= 1 ||
      options.df_mixed_precision_max_iterations < 16) {
    auto warmup_options = options.gmres;
    warmup_options.absolute_tolerance = options.df_mixed_precision_warmup_tolerance;
    warmup_options.relative_tolerance = 0.0;
    warmup_options.max_iterations =
        std::min(options.gmres.max_iterations, options.df_mixed_precision_max_iterations);
    warmup_options.restart = std::min(warmup_options.restart, warmup_options.max_iterations);
    const auto warmup_plan = response::prepare_response(response_problem, warmup_options);
    auto warmup = response::solve_response(warmup_plan, response_problem, rhs, {}, diagonal);
    owner.finish_mixed_precision();
    if (warmup.status == response::GmresStatus::nonfinite_input ||
        warmup.status == response::GmresStatus::nonfinite_operator ||
        warmup.status == response::GmresStatus::nonfinite_preconditioner)
      throw std::runtime_error("nonfinite RCCSD CUDA Lambda mixed warmup");
    result.iterations = warmup.iterations;
    result.operator_actions = warmup.operator_actions;
    result.preconditioner_actions = warmup.preconditioner_actions;
    // No approximate Krylov basis or residual crosses this boundary. The
    // original controller recomputes J^T x in FP64 from this finite proposal,
    // including after finite warmup exhaustion or optional workspace refusal.
    result.solution = std::move(warmup.solution);
    owner.record_mixed_summary(1, 0,
                               warmup.converged() ? "converged_warmup" : "max_iterations_handoff_fp64");
    return result;
  }
  const double rhs_norm = response::stable_norm(rhs);
  const double target = std::max(options.gmres.absolute_tolerance,
                                 options.gmres.relative_tolerance * rhs_norm);
  const std::size_t n = rhs.size();
  const std::size_t max_passes = options.df_mixed_precision_refinement_passes;
  const std::size_t per_pass_cap = std::min(
      options.gmres.max_iterations,
      std::max(options.df_mixed_precision_max_iterations, options.gmres.restart));
  const double inner_rel_floor = options.df_mixed_precision_inner_relative_tolerance;
  const double inner_rel_ceil =
      std::max(inner_rel_floor, LambdaOptions::kMaxMixedInnerRelativeTolerance);
  const std::size_t max_gcr_vectors = owner.mixed_gcr_history_limit();
  const double breakdown_tol = options.gmres.breakdown_tolerance;
  std::vector<double> x_cur(n, 0.0);
  std::vector<double> r_cur(rhs.begin(), rhs.end());
  std::vector<std::vector<double>> gcr_p;
  std::vector<std::vector<double>> gcr_u;
  std::vector<double> w_buf;
  double r_norm = rhs_norm;
  double prev_contraction = 1.0;
  std::size_t passes_done = 0;
  std::size_t strict_ref_actions = 0;
  const char* reason = "max_passes_handoff_fp64";
  bool mixed_closed = false;

  if (r_norm <= target) {
    owner.finish_mixed_precision();
    mixed_closed = true;
    reason = "initial_residual";
  } else {
    for (std::size_t pass = 0; pass < max_passes; ++pass) {
      ++passes_done;
      const double needed_rel = r_norm > 0.0 ? (0.1 * target / r_norm) : inner_rel_floor;
      const double inner_rel = std::clamp(needed_rel, inner_rel_floor, inner_rel_ceil);
      auto inner_options = options.gmres;
      inner_options.relative_tolerance = 0.0;
      inner_options.absolute_tolerance = inner_rel * r_norm;
      inner_options.max_iterations = per_pass_cap;
      inner_options.restart = std::min(inner_options.restart, inner_options.max_iterations);
      inner_options.true_residual_every = inner_options.max_iterations;
      auto inner_plan = response::prepare_response(response_problem, inner_options);
      if (inner_plan.workspace_bytes > inner_options.max_workspace_bytes) {
        inner_options.restart =
            std::min(options.gmres.restart, options.df_mixed_precision_max_iterations);
        inner_options.max_iterations = inner_options.restart;
        inner_options.true_residual_every = inner_options.max_iterations;
        inner_plan = response::prepare_response(response_problem, inner_options);
      }
      auto inner = response::solve_response(inner_plan, response_problem, r_cur, {}, diagonal);
      if (inner.status == response::GmresStatus::nonfinite_input ||
          inner.status == response::GmresStatus::nonfinite_operator ||
          inner.status == response::GmresStatus::nonfinite_preconditioner) {
        owner.finish_mixed_precision();
        throw std::runtime_error("nonfinite RCCSD CUDA Lambda mixed warmup");
      }
      result.iterations = lambda_size_add(result.iterations, inner.iterations);
      result.operator_actions = lambda_size_add(result.operator_actions, inner.operator_actions);
      result.preconditioner_actions =
          lambda_size_add(result.preconditioner_actions, inner.preconditioner_actions);
      if (inner.solution.size() != n) {
        reason = "workspace_refused_handoff_fp64";
        break;
      }
      double delta_norm = 0.0;
      try {
        delta_norm = response::stable_norm(inner.solution);
      } catch (const std::exception&) {
        owner.finish_mixed_precision();
        throw std::runtime_error("nonfinite RCCSD CUDA Lambda mixed warmup");
      }
      if (delta_norm == 0.0) {
        reason = "zero_correction_handoff_fp64";
        break;
      }

      const double predicted_norm =
          r_norm * std::max(prev_contraction, inner.relative_residual);
      const bool skip_inner_fp64_check =
          pass >= 1 && prev_contraction < 0.05 && inner.converged() &&
          (predicted_norm <= 2.0 * target || pass + 1 == max_passes);
      if (skip_inner_fp64_check) {
        for (std::size_t idx = 0; idx < n; ++idx) x_cur[idx] += inner.solution[idx];
        reason = (predicted_norm <= 2.0 * target) ? "converged_mixed_ir"
                                                  : "max_passes_handoff_fp64";
        break;
      }

      std::vector<double> delta(inner.solution.begin(), inner.solution.end());
      inner = {};
      owner.suspend_mixed_precision();
      w_buf.resize(n);
      response_problem.apply()(delta, w_buf);
      strict_ref_actions = lambda_size_add(strict_ref_actions, 1);

      // Exact FP64 GCR/FGMRES outer minimal-residual projection guarantees
      // monotonic decrease of ||b - A_64 x||_2 across refinement passes.
      for (unsigned mgs = 0; mgs < 2; ++mgs) {
        for (std::size_t j = 0; j < gcr_u.size(); ++j) {
          double proj = 0.0;
          for (std::size_t idx = 0; idx < n; ++idx) proj += gcr_u[j][idx] * w_buf[idx];
          for (std::size_t idx = 0; idx < n; ++idx) {
            w_buf[idx] -= proj * gcr_u[j][idx];
            delta[idx] -= proj * gcr_p[j][idx];
          }
        }
      }
      double w_norm = 0.0;
      try {
        w_norm = response::stable_norm(w_buf);
      } catch (const std::exception&) {
        owner.finish_mixed_precision();
        throw std::runtime_error("nonfinite RCCSD CUDA Lambda strict refinement residual");
      }
      if (!std::isfinite(w_norm) || w_norm <= breakdown_tol) {
        reason = "breakdown_handoff_fp64";
        break;
      }
      const double inv_w = 1.0 / w_norm;
      double alpha = 0.0;
      for (std::size_t idx = 0; idx < n; ++idx) {
        w_buf[idx] *= inv_w;
        delta[idx] *= inv_w;
        alpha += w_buf[idx] * r_cur[idx];
      }
      if (!std::isfinite(alpha)) {
        reason = "nonfinite_projection_handoff_fp64";
        break;
      }
      for (std::size_t idx = 0; idx < n; ++idx) {
        x_cur[idx] += alpha * delta[idx];
        r_cur[idx] -= alpha * w_buf[idx];
      }
      const double new_r_norm = response::stable_norm(r_cur);
      const double contraction = r_norm > 0.0 ? (new_r_norm / r_norm) : 0.0;
      r_norm = new_r_norm;
      prev_contraction = contraction;

      if (r_norm <= target) {
        reason = "converged_mixed_ir";
        break;
      }
      if (contraction > 0.35) {
        reason = "stagnated_handoff_fp64";
        break;
      }
      if (pass + 1 >= max_passes) {
        reason = "max_passes_handoff_fp64";
        break;
      }
      if (max_gcr_vectors > 0) {
        if (gcr_p.size() >= max_gcr_vectors) {
          gcr_p.erase(gcr_p.begin());
          gcr_u.erase(gcr_u.begin());
        }
        gcr_p.push_back(std::move(delta));
        gcr_u.push_back(std::move(w_buf));
      }
      owner.resume_mixed_precision();
    }
  }
  if (!mixed_closed) owner.finish_mixed_precision();
  gcr_p.clear();
  gcr_u.clear();
  w_buf.clear();
  r_cur.clear();
  result.solution.assign(x_cur.begin(), x_cur.end());
  owner.record_mixed_summary(passes_done, strict_ref_actions, reason);
  return result;
}

template <class Backend>
struct LambdaKrylovStorage {
  std::unique_ptr<Backend> backend;
  std::size_t bytes{}, provider_bytes{}, device_admission{}, extra_admission{};
  int provider_version{}, runtime_version{};
};

/** Admit optional resident Krylov storage after sealing the mixed warmup.
 * DF admission bounds both simultaneous numeric storage and device residency.
 */
template <class Backend, class Owner, class Layout>
LambdaKrylovStorage<Backend> prepare_lambda_krylov_storage(
    Owner& owner, const Layout& layout, const Problem& problem, const LambdaOptions& options,
    const response::ResponsePlan& plan, std::span<const double> diagonal,
    std::size_t provider_allowance) {
  LambdaKrylovStorage<Backend> storage;
  if (!options.cuda_resident_krylov ||
      layout.dimension() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    return storage;

  const auto required = Backend::required_bytes(plan, layout, !diagonal.empty());
  const auto extra = lambda_size_add(required, provider_allowance);
  const auto device_bytes = lambda_size_add(
      lambda_size_add(owner.owned_device_bytes(), problem.reference_retained_bytes), extra);
  const bool df_admitted =
      !problem.naux || (lambda_size_add(owner.numeric_capacity_bytes(), extra) <= options.max_bytes &&
                        device_bytes <= owner.device_limit_bytes());
  if (device_bytes > options.max_bytes || !df_admitted) return storage;
  try {
    storage.backend = std::make_unique<Backend>(owner, plan, diagonal);
  } catch (const std::bad_alloc&) {
    return storage;
  }
  storage.bytes = storage.backend->owned_resident_bytes();
  storage.device_admission = device_bytes;
  storage.extra_admission = extra;
  storage.provider_bytes = storage.backend->provider_observed_bytes();
  storage.provider_version = storage.backend->provider_version();
  storage.runtime_version = storage.backend->runtime_version();
  return storage;
}

}  // namespace generativeqc::cc::detail
