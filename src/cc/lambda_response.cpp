#include "cc/f12_fixed_cusp.hpp"
#include "cc/lambda_response.hpp"
#include "cc/lambda_telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#include "generated_rccsd_cpu.hpp"
#include "generativeqc/generativeqc.h"
#include "response/solve.hpp"

namespace generativeqc::cc {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b) { return generated::checked_add(a, b); }

std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("RCCSD Lambda size overflow");
  return a * b;
}

std::size_t bytes(std::size_t elements) { return checked_mul(elements, sizeof(double)); }

generated::Inputs inputs(const Problem& p, const SolverResult& cc) {
  return {p.foo.data(),  p.fov.data(),  p.fvv.data(),  p.ovov.data(), p.ovvo.data(),
          p.oovv.data(), p.ovvv.data(), p.ovoo.data(), p.oooo.data(), p.vvvv.data(),
          p.d1.data(),   p.d2.data(),   cc.t1.data(),  cc.t2.data()};
}

double max_abs(std::span<const double> values) {
  double result = 0.0;
  for (double value : values) {
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite RCCSD Lambda residual");
    result = std::max(result, std::abs(value));
  }
  return result;
}

struct AmplitudeLayout {
  std::size_t o{}, v{}, n1{}, n2{};
  std::vector<std::size_t> representatives;
  std::vector<std::size_t> partners;
  std::vector<double> sqrt_weights;

  explicit AmplitudeLayout(std::size_t occupied, std::size_t virtuals)
      : o(occupied),
        v(virtuals),
        n1(checked_mul(o, v)),
        n2(checked_mul(checked_mul(o, o), checked_mul(v, v))) {}

  [[nodiscard]] std::size_t pair_count() const { return checked_add(n2, n1) / 2; }
  [[nodiscard]] std::size_t dimension() const { return checked_add(n1, pair_count()); }
  [[nodiscard]] std::size_t storage_bytes() const {
    return checked_add(checked_mul(checked_mul(2, pair_count()), sizeof(std::size_t)),
                       bytes(dimension()));
  }

  void initialize() {
    sqrt_weights.assign(dimension(), 1.0);
    representatives.resize(pair_count());
    partners.resize(pair_count());
    std::size_t position = 0;
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b) {
            const auto flat = ((i * o + j) * v + a) * v + b;
            const auto mate = ((j * o + i) * v + b) * v + a;
            if (flat > mate) continue;
            representatives[position] = flat;
            partners[position] = mate;
            sqrt_weights[n1 + position] = flat == mate ? 1.0 : std::sqrt(2.0);
            ++position;
          }
  }

  void validate_dense(std::span<const double> one, std::span<const double> two) const {
    if (one.size() != n1 || two.size() != n2)
      throw std::invalid_argument("RCCSD Lambda amplitude shape mismatch");
    for (double value : one)
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite RCCSD Lambda singles");
    for (std::size_t k = 0; k < representatives.size(); ++k) {
      const double first = two[representatives[k]], second = two[partners[k]];
      if (!std::isfinite(first) || !std::isfinite(second) ||
          std::abs(first - second) > 1e-10 * (1.0 + std::max(std::abs(first), std::abs(second))))
        throw std::invalid_argument("RCCSD doubles violate simultaneous pair symmetry");
    }
  }

  void pack_weighted(std::span<const double> one, std::span<const double> two,
                     std::span<double> output) const {
    if (one.size() != n1 || two.size() != n2)
      throw std::invalid_argument("RCCSD Lambda amplitude shape mismatch");
    if (output.size() != dimension())
      throw std::invalid_argument("RCCSD Lambda packed output shape mismatch");
    for (std::size_t idx = 0; idx < n1; ++idx) {
      const double value = one[idx];
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite RCCSD Lambda singles");
      output[idx] = value;
    }
    const auto* reps = representatives.data();
    const auto* mates = partners.data();
    const auto* w = sqrt_weights.data() + n1;
    const auto* two_ptr = two.data();
    auto* out_pairs = output.data() + n1;
    const auto count = representatives.size();
    for (std::size_t k = 0; k < count; ++k) {
      const double first = two_ptr[reps[k]], second = two_ptr[mates[k]];
      if (!std::isfinite(first) || !std::isfinite(second) ||
          std::abs(first - second) > 1e-10 * (1.0 + std::max(std::abs(first), std::abs(second))))
        throw std::invalid_argument("RCCSD doubles violate simultaneous pair symmetry");
      out_pairs[k] = w[k] * first;
    }
  }

  void unpack_weighted(std::span<const double> packed, std::span<double> one,
                       std::span<double> two) const {
    if (packed.size() != dimension() || one.size() != n1 || two.size() != n2)
      throw std::invalid_argument("RCCSD Lambda unpack shape mismatch");
    std::copy(packed.begin(), packed.begin() + static_cast<std::ptrdiff_t>(n1), one.begin());
    const auto* reps = representatives.data();
    const auto* mates = partners.data();
    const auto* w = sqrt_weights.data() + n1;
    const auto* in_pairs = packed.data() + n1;
    auto* two_ptr = two.data();
    const auto count = representatives.size();
    for (std::size_t k = 0; k < count; ++k) {
      const double value = in_pairs[k] / w[k];
      two_ptr[reps[k]] = value;
      two_ptr[mates[k]] = value;
    }
  }

  void publish(std::span<const double> packed, std::vector<double>& one,
               std::vector<double>& two) const {
    one.resize(n1);
    two.resize(n2);
    unpack_weighted(packed, one, two);
  }
};

std::size_t vector_capacity_bytes(const std::vector<double>& values) {
  return bytes(values.capacity());
}

}  // namespace

namespace detail {

response::LinearResponseProblem make_lambda_response_problem(std::size_t dimension,
                                                             response::LinearOperator apply) {
  return {dimension, std::move(apply), response::LinearResponseSymmetry::General};
}

bool fill_lambda_diagonal_preconditioner(const Problem& p,
                                         std::span<const std::size_t> representatives,
                                         std::span<const std::size_t> partners,
                                         double breakdown_tolerance, std::span<double> diagonal) {
  if (representatives.size() != partners.size() ||
      diagonal.size() != checked_add(p.d1.size(), representatives.size()))
    throw std::invalid_argument("RCCSD Lambda preconditioner shape mismatch");
  // This is a solver convenience, not a new scientific denominator admission.
  // Small denominators disable it instead of clipping or changing the operator.
  const double threshold = std::max(1e-10, breakdown_tolerance);
  auto safe = [&](double value) { return std::isfinite(value) && std::abs(value) > threshold; };
  for (std::size_t i = 0; i < p.d1.size(); ++i) {
    if (!safe(p.d1[i])) return false;
    diagonal[i] = -p.d1[i];
  }
  const auto n2 = checked_mul(checked_mul(p.nocc, p.nocc), checked_mul(p.nvir, p.nvir));
  for (std::size_t k = 0; k < representatives.size(); ++k) {
    if (representatives[k] >= n2 || partners[k] >= n2)
      throw std::invalid_argument("RCCSD Lambda preconditioner pair index out of range");
    const auto first = doubles_denominator_at(p, representatives[k]),
               second = doubles_denominator_at(p, partners[k]);
    if (!safe(first) || !safe(second) ||
        std::abs(first - second) > 1e-10 * (1.0 + std::max(std::abs(first), std::abs(second))))
      return false;
    diagonal[p.d1.size() + k] = -first;
  }
  return true;
}

}  // namespace detail

void validate_lambda_options(const LambdaOptions& options) {
  options.convergence.validate();
  // Unused warmup controls must not narrow the original strict FP64 domain.
  if (!std::isfinite(options.cc_tolerance) || options.cc_tolerance <= 0.0 ||
      options.cc_tolerance > options.convergence.cc_residual() ||
      !std::isfinite(options.lambda_tolerance) || options.lambda_tolerance <= 0.0 ||
      options.lambda_tolerance > options.convergence.lambda_residual() || !options.max_bytes ||
      !options.df_auxiliary_batch_limit ||
      (options.df_mixed_precision &&
       (!std::isfinite(options.df_mixed_precision_warmup_tolerance) ||
        options.df_mixed_precision_warmup_tolerance <= 0.0 ||
        !options.df_mixed_precision_max_iterations ||
        !options.df_mixed_precision_refinement_passes ||
        !std::isfinite(options.df_mixed_precision_inner_relative_tolerance) ||
        options.df_mixed_precision_inner_relative_tolerance <= 0.0 ||
        options.df_mixed_precision_inner_relative_tolerance >
            LambdaOptions::kMaxMixedInnerRelativeTolerance ||
        options.df_mixed_precision_warmup_tolerance <= options.gmres.absolute_tolerance)))
    throw std::invalid_argument("invalid RCCSD Lambda tolerance/budget");
  (void)response::prepare_gmres(1, options.gmres);
}

std::size_t lambda_cpu_numeric_capacity(const Problem& p, const SolverResult& cc,
                                        const LambdaOptions& options, bool with_energy_source) {
  validate_lambda_options(options);
  AmplitudeLayout layout(p.nocc, p.nvir);
  const auto replay_elements = generated::replay_arena_elements(p.nocc, p.nvir);
  const auto shared_rhs_elements = generated::lambda_rhs_arena_elements(p.nocc, p.nvir);
  const auto shared_jt_elements = generated::lambda_transpose_arena_elements(p.nocc, p.nvir);
  const auto independent_rhs_elements =
      generated::lambda_independent_rhs_arena_elements(p.nocc, p.nvir);
  const auto independent_jt_elements =
      generated::lambda_independent_transpose_arena_elements(p.nocc, p.nvir);
  const auto arena_elements = std::max({replay_elements, shared_rhs_elements, shared_jt_elements,
                                        independent_rhs_elements, independent_jt_elements});
  const auto gmres_plan = response::prepare_gmres(layout.dimension(), options.gmres);

  std::size_t capacity = checked_add(p.reference_retained_bytes, problem_host_bytes(p));
  capacity = checked_add(capacity, vector_capacity_bytes(cc.t1));
  capacity = checked_add(capacity, vector_capacity_bytes(cc.t2));
  capacity = checked_add(capacity, bytes(arena_elements));
  // Dense seeds, packed RHS/audit and published dense Lambda can coexist.
  capacity = checked_add(capacity, bytes(checked_mul(3, layout.n1 + layout.n2)));
  capacity = checked_add(capacity, bytes(checked_mul(3, layout.dimension())));
  capacity = checked_add(capacity, layout.storage_bytes());
  capacity = checked_add(capacity, gmres_plan.workspace_bytes);
  if (with_energy_source || p.f12.active()) capacity = checked_add(capacity, bytes(layout.dimension()));
  if (p.f12.active())
    capacity = checked_add(capacity, bytes(checked_add(checked_mul(2, layout.n1), layout.n2)));
  return capacity;
}

static LambdaResult solve_lambda_cpu_impl(const Problem& p, const SolverResult& cc,
                                          std::span<const double> t1_source,
                                          std::span<const double> t2_source,
                                          const LambdaOptions& options) {
  using Clock = std::chrono::steady_clock;
  const auto elapsed = [](Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
  };
  const auto precompute_begin = Clock::now();
  validate_problem(p);
  validate_lambda_options(options);
  if (!cc.converged()) throw std::invalid_argument("RCCSD Lambda requires a converged CC result");
  AmplitudeLayout layout(p.nocc, p.nvir);
  if (cc.t1.size() != layout.n1 || cc.t2.size() != layout.n2)
    throw std::invalid_argument("RCCSD Lambda CC result shape mismatch");
  if (!std::isfinite(cc.correlation_energy))
    throw std::invalid_argument("nonfinite RCCSD Lambda primal energy");
  const bool with_source = !t1_source.empty() || !t2_source.empty();
  const bool f12_active = p.f12.active();
  if (with_source && (t1_source.size() != layout.n1 || t2_source.size() != layout.n2))
    throw std::invalid_argument("RCCSD Lambda energy-source shape mismatch");
  for (const auto source : {t1_source, t2_source})
    for (double value : source)
      if (!std::isfinite(value))
        throw std::invalid_argument("nonfinite RCCSD Lambda energy source");
  const auto capacity = lambda_cpu_numeric_capacity(p, cc, options, with_source);
  if (capacity > options.max_bytes)
    throw std::length_error("RCCSD Lambda exceeds simultaneous host budget");

  const auto arena_elements =
      std::max({generated::replay_arena_elements(p.nocc, p.nvir),
                generated::lambda_rhs_arena_elements(p.nocc, p.nvir),
                generated::lambda_transpose_arena_elements(p.nocc, p.nvir),
                generated::lambda_independent_rhs_arena_elements(p.nocc, p.nvir),
                generated::lambda_independent_transpose_arena_elements(p.nocc, p.nvir)});
  const auto in = inputs(p, cc);

  layout.initialize();
  layout.validate_dense(cc.t1, cc.t2);

  std::vector<double> f12_r1_const;
  if (f12_active) f12_r1_const = f12::singles_constant(p.nocc, p.nvir, p.f12.w_vo);

  std::vector<double> arena(arena_elements);
  const auto replay = generated::run_replay_cpu(p.nocc, p.nvir, in, arena.data(), arena.size());
  double replay_energy = replay.energy;
  double replay_r1 = 0.0, replay_r2 = 0.0;
  if (f12_active) {
    std::vector<double> r1_buf(replay.r1, replay.r1 + layout.n1);
    std::vector<double> r2_buf(replay.r2, replay.r2 + layout.n2);
    f12::add_residuals_cpu(p.nocc, p.nvir, p.f12, f12_r1_const, cc.t1.data(), r1_buf.data(),
                           r2_buf.data());
    replay_energy += f12::energy_cpu(p.nocc, p.nvir, p.f12, f12_r1_const, cc.t1.data(), cc.t2.data());
    replay_r1 = max_abs(r1_buf);
    replay_r2 = max_abs(r2_buf);
  } else {
    replay_r1 = max_abs({replay.r1, layout.n1});
    replay_r2 = max_abs({replay.r2, layout.n2});
  }
  if (std::max(replay_r1, replay_r2) > options.cc_tolerance ||
      std::abs(replay_energy - cc.correlation_energy) > 1e-10)
    throw std::runtime_error("RCCSD Lambda fresh primal replay gate failed");

  const double energy_seed = -1.0;
  auto rhs_outputs =
      generated::run_lambda_rhs_cpu(p.nocc, p.nvir, in, &energy_seed, arena.data(), arena.size());
  std::vector<double> rhs(layout.dimension());
  layout.pack_weighted({rhs_outputs.t1, layout.n1}, {rhs_outputs.t2, layout.n2}, rhs);

  std::vector<double> packed_source;
  if (with_source || f12_active) {
    std::vector<double> f12_g1, f12_g2;
    if (f12_active) f12::energy_gradient_cpu(p.nocc, p.nvir, p.f12, f12_r1_const, cc.t1, f12_g1, f12_g2);
    // Project directly into the one retained packed vector. Separate dense
    // source copies are unnecessary and would add another simultaneous owner.
    packed_source.resize(layout.dimension());
    for (std::size_t index = 0; index < layout.n1; ++index)
      packed_source[index] =
          (with_source ? t1_source[index] : 0.0) + (f12_active ? f12_g1[index] : 0.0);
    for (std::size_t k = 0; k < layout.representatives.size(); ++k) {
      const auto first = layout.representatives[k];
      const auto second = layout.partners[k];
      const double v_first =
          (with_source ? t2_source[first] : 0.0) + (f12_active ? f12_g2[first] : 0.0);
      const double v_second =
          (with_source ? t2_source[second] : 0.0) + (f12_active ? f12_g2[second] : 0.0);
      const double projected = first == second ? v_first : 0.5 * (v_first + v_second);
      packed_source[layout.n1 + k] = layout.sqrt_weights[layout.n1 + k] * projected;
      if (!std::isfinite(packed_source[layout.n1 + k]))
        throw std::invalid_argument("nonfinite projected RCCSD Lambda energy source");
    }
    for (std::size_t index = 0; index < rhs.size(); ++index) rhs[index] -= packed_source[index];
  }

  std::vector<double> dense_one(layout.n1), dense_two(layout.n2);
  std::vector<double> f12_action_one(f12_active ? layout.n1 : 0);
  auto apply = [&](std::span<const double> input, std::span<double> output) {
    layout.unpack_weighted(input, dense_one, dense_two);
    const auto action = generated::run_lambda_transpose_cpu(
        p.nocc, p.nvir, in, dense_one.data(), dense_two.data(), arena.data(), arena.size());
    if (f12_active) {
      std::copy_n(action.t1, layout.n1, f12_action_one.data());
      f12::add_lambda_transpose_cpu(p.nocc, p.nvir, p.f12.w_vo, dense_two, f12_action_one.data());
      layout.pack_weighted(f12_action_one, {action.t2, layout.n2}, output);
    } else {
      layout.pack_weighted({action.t1, layout.n1}, {action.t2, layout.n2}, output);
    }
  };

  const auto response_problem =
      detail::make_lambda_response_problem(layout.dimension(), std::move(apply));
  const auto response_plan = response::prepare_response(response_problem, options.gmres);
  // Reuse the already-admitted independent-audit vector for the diagonal while
  // GMRES runs. Its contents are overwritten by the physical audit afterward;
  // no new numeric owner or larger simultaneous-capacity bound is required.
  std::vector<double> independent(layout.dimension());
  const bool preconditioned =
      options.diagonal_preconditioning &&
      detail::fill_lambda_diagonal_preconditioner(p, layout.representatives, layout.partners,
                                                  options.gmres.breakdown_tolerance, independent);
  const double precompute_s = elapsed(precompute_begin);
  const auto strict_begin = Clock::now();
  auto solved = response::solve_response(
      response_plan, response_problem, rhs, {},
      preconditioned ? std::span<const double>(independent) : std::span<const double>{});
  const double strict_s = elapsed(strict_begin);
  if (!solved.converged()) throw std::runtime_error("RCCSD Lambda GMRES did not converge");

  const auto audit_begin = Clock::now();
  layout.unpack_weighted(solved.solution, dense_one, dense_two);
  const auto independent_action = generated::run_lambda_independent_transpose_cpu(
      p.nocc, p.nvir, in, dense_one.data(), dense_two.data(), arena.data(), arena.size());
  if (f12_active) {
    std::copy_n(independent_action.t1, layout.n1, f12_action_one.data());
    f12::add_lambda_transpose_cpu(p.nocc, p.nvir, p.f12.w_vo, dense_two, f12_action_one.data());
    layout.pack_weighted(f12_action_one, {independent_action.t2, layout.n2}, independent);
  } else {
    layout.pack_weighted({independent_action.t1, layout.n1}, {independent_action.t2, layout.n2},
                         independent);
  }

  const auto independent_rhs_output = generated::run_lambda_independent_rhs_cpu(
      p.nocc, p.nvir, in, &energy_seed, arena.data(), arena.size());
  std::vector<double> independent_rhs(layout.dimension());
  layout.pack_weighted({independent_rhs_output.t1, layout.n1},
                       {independent_rhs_output.t2, layout.n2}, independent_rhs);
  if (!packed_source.empty())
    for (std::size_t index = 0; index < independent_rhs.size(); ++index)
      independent_rhs[index] -= packed_source[index];
  for (std::size_t index = 0; index < independent.size(); ++index)
    independent[index] -= independent_rhs[index];

  const double independent_norm = response::stable_norm(independent);
  double independent_max = 0.0;
  for (std::size_t index = 0; index < independent.size(); ++index) {
    const double scaled = std::abs(independent[index] / layout.sqrt_weights[index]);
    if (!std::isfinite(scaled))
      throw std::runtime_error("nonfinite RCCSD Lambda independent physical residual");
    if (scaled > independent_max) independent_max = scaled;
  }
  if (!std::isfinite(solved.residual_norm) || !std::isfinite(independent_norm) ||
      !std::isfinite(independent_max) ||
      std::max({solved.residual_norm, independent_norm, independent_max}) >
          options.lambda_tolerance)
    throw std::runtime_error("RCCSD Lambda independent physical residual gate failed");

  const double audit_s = elapsed(audit_begin);
  LambdaResult result;
  layout.publish(solved.solution, result.lambda1, result.lambda2);
  result.reason = "shared GMRES and expanded physical Lambda residual passed";
  result.diagnostic.cc_r1_max = replay_r1;
  result.diagnostic.cc_r2_max = replay_r2;
  result.diagnostic.lambda_residual_norm = solved.residual_norm;
  result.diagnostic.independent_residual_norm = independent_norm;
  result.diagnostic.independent_residual_max = independent_max;
  result.diagnostic.iterations = solved.iterations;
  result.diagnostic.operator_actions = solved.operator_actions;
  result.diagnostic.df_strict_iterations = solved.iterations;
  result.diagnostic.df_strict_actions = solved.operator_actions;
  result.diagnostic.precompute_seconds = precompute_s;
  result.diagnostic.df_strict_seconds = strict_s;
  result.diagnostic.audit_seconds = audit_s;
  result.diagnostic.df_mixed_handoff_reason = "cpu_fp64";
  result.diagnostic.diagonal_preconditioned = preconditioned;
  result.diagnostic.preconditioner_actions = solved.preconditioner_actions;
  result.diagnostic.numeric_capacity_bytes = capacity;
  result.diagnostic.shared_program_hash = generated::lambda_transpose_program_hash;
  result.diagnostic.independent_program_hash =
      generated::lambda_independent_transpose_selected_program_hash(p.nocc, p.nvir);
  detail::record_last_lambda_diagnostic(result.diagnostic);
  return result;
}

namespace {
thread_local std::optional<detail::LambdaTelemetry> last_lambda_telemetry;
}  // namespace

namespace detail {
void record_last_lambda_diagnostic(const LambdaDiagnostic& d) noexcept {
  try {
    auto& state = last_lambda_telemetry.emplace();
    state.scalars = {
        d.df_mixed_seconds,          d.df_strict_seconds,         d.precompute_seconds,
        d.audit_seconds,             d.publication_seconds,       d.lambda_residual_norm,
        d.independent_residual_norm, d.independent_residual_max,  d.cc_r1_max,
        d.cc_r2_max,
    };
    state.counts = {
        static_cast<std::uint64_t>(d.iterations),
        static_cast<std::uint64_t>(d.operator_actions),
        static_cast<std::uint64_t>(d.df_mixed_iterations),
        static_cast<std::uint64_t>(d.df_strict_iterations),
        static_cast<std::uint64_t>(d.df_mixed_actions),
        static_cast<std::uint64_t>(d.df_strict_actions),
        static_cast<std::uint64_t>(d.df_mixed_precision ? 1 : 0),
        static_cast<std::uint64_t>(d.df_mixed_precision_refused ? 1 : 0),
        static_cast<std::uint64_t>(d.df_precision_switches),
        static_cast<std::uint64_t>(d.df_mixed_refinement_passes),
        static_cast<std::uint64_t>(d.df_auxiliary_batch_size),
        static_cast<std::uint64_t>(d.df_primal_batch_size),
        static_cast<std::uint64_t>(d.df_core_reuse ? 1 : 0),
        static_cast<std::uint64_t>(d.df_fp32_gemm_calls),
        static_cast<std::uint64_t>(d.df_gemm_calls),
        static_cast<std::uint64_t>(d.owned_device_bytes),
    };
    state.handoff_reason = d.df_mixed_handoff_reason ? d.df_mixed_handoff_reason : "unknown";
  } catch (...) {
    last_lambda_telemetry.reset();
  }
}

void reset_last_lambda_diagnostic() noexcept { last_lambda_telemetry.reset(); }

void capture_last_lambda_telemetry(std::optional<LambdaTelemetry>& destination) noexcept {
  try {
    destination = last_lambda_telemetry;
  } catch (...) {
    destination.reset();
  }
}

int copy_lambda_telemetry(const std::optional<LambdaTelemetry>& slot, double* scalars,
                          std::size_t scalar_count, std::uint64_t* counts, std::size_t count_len,
                          char* reason, std::size_t reason_len) noexcept {
  if (!slot) return 0;
  if (scalars)
    std::copy_n(slot->scalars.begin(), std::min(scalar_count, slot->scalars.size()), scalars);
  if (counts) std::copy_n(slot->counts.begin(), std::min(count_len, slot->counts.size()), counts);
  if (reason && reason_len) {
    const auto copy_len = std::min(reason_len - 1, slot->handoff_reason.size());
    std::memcpy(reason, slot->handoff_reason.data(), copy_len);
    reason[copy_len] = '\0';
  }
  return 1;
}
}  // namespace detail

LambdaResult solve_lambda_cpu(const Problem& problem, const SolverResult& cc_result,
                              const LambdaOptions& options) {
  return solve_lambda_cpu_impl(problem, cc_result, {}, {}, options);
}

LambdaResult solve_lambda_cpu_with_energy_source(const Problem& problem,
                                                 const SolverResult& cc_result,
                                                 std::span<const double> t1_source,
                                                 std::span<const double> t2_source,
                                                 const LambdaOptions& options) {
  return solve_lambda_cpu_impl(problem, cc_result, t1_source, t2_source, options);
}

}  // namespace generativeqc::cc

extern "C" GENERATIVEQC_API void generativeqc_cc_reset_last_lambda_telemetry_v1(void) {
  generativeqc::cc::detail::reset_last_lambda_diagnostic();
}

extern "C" GENERATIVEQC_API int generativeqc_cc_last_lambda_telemetry_v1(
    double* scalars, std::size_t scalar_count, std::uint64_t* counts, std::size_t count_len,
    char* reason, std::size_t reason_len) {
  return generativeqc::cc::detail::copy_lambda_telemetry(
      generativeqc::cc::last_lambda_telemetry, scalars, scalar_count, counts, count_len, reason,
      reason_len);
}
