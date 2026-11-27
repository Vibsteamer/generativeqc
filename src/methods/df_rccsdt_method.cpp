#include "methods/df_rccsdt_method.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include "methods/correlation_convergence.hpp"

#if GENERATIVEQC_HAS_CUDA
#include "methods/df_ccsdt_force.hpp"
#endif
#include "methods/rccsd_method.hpp"
#include "molecule/scalar_hamiltonian_identity.hpp"

namespace generativeqc::methods::detail {
namespace {

constexpr double kQualifiedMetricThreshold = 1e-10;

void validate_descriptor(const generativeqc_method_descriptor& descriptor,
                         const runtime::ExecutionContext& execution) {
  if (!execution.cuda_requested())
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "public DF-RCCSD(T) currently requires CUDA");
  if (descriptor.density_fitting_auxiliary_basis == nullptr)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DF-RCCSD(T) requires an explicit auxiliary basis");
  const bool correlation_mp2 =
      descriptor.method == GENERATIVEQC_METHOD_MP2 &&
      descriptor.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_CORRELATION_CUDA;
  if (!correlation_mp2 && descriptor.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_CUDA &&
      descriptor.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_AUTO &&
      descriptor.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_FULL_CUDA)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "DF-RCCSD(T) requires density_fitting='cuda' or 'auto'");
  const double threshold = descriptor.density_fitting_relative_threshold == 0.0
                               ? kQualifiedMetricThreshold
                               : descriptor.density_fitting_relative_threshold;
  if (!std::isfinite(threshold) || threshold <= 0.0 || threshold >= 1.0)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DF-RCCSD(T) metric threshold must lie in (0,1)");
  if (threshold != kQualifiedMetricThreshold)
    throw MethodError(
        GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
        "public DF-RCCSD(T) currently qualifies density_fitting_relative_threshold=1e-10");
  if (descriptor.density_fitting_memory_budget_bytes != 0)
    throw MethodError(
        GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
        "DF-RCCSD(T) uses correlation_memory_budget_bytes for complete endpoint admission; "
        "a separate density-fitting memory budget is not yet qualified");
  if (descriptor.screening_tolerance != 0.0)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DF-RCCSD(T) requires screening_tolerance=0");
  if (descriptor.precision_mode != GENERATIVEQC_PRECISION_FP64)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED, "DF-RCCSD(T) requires FP64 precision");
  if (descriptor.ccsd_frozen_core != 0)
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "DF-RCCSD(T) frozen-core references are not implemented");
}

class DfRccsdtPrepared final : public PreparedCalculation {
 public:
  DfRccsdtPrepared(Capabilities capabilities, runtime::ExecutionContext execution,
                   core::System system, core::System auxiliary,
                   const generativeqc_method_descriptor& descriptor)
      : PreparedCalculation(execution.backend(), execution.device_id()),
        capabilities_(capabilities),
        execution_(std::move(execution)),
        system_(std::move(system)),
        auxiliary_(std::move(auxiliary)),
        descriptor_(descriptor) {
    convergence_ = correlation_convergence(descriptor);
    preconvergence_ = correlation_preconvergence(descriptor);
    descriptor_.struct_size = sizeof(descriptor_);
    full_df_reference_ = descriptor.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_FULL_CUDA;
    // Preserve the legacy descriptor contract of the CC solver. The explicit
    // demand carries the RI-JK model through reference, companions and response.
    descriptor_.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
    descriptor_.density_fitting_auxiliary_basis = nullptr;
    descriptor_.density_fitting_memory_budget_bytes = 0;
    descriptor_.ks_options = nullptr;
    descriptor_.initial_guess = nullptr;
    if (descriptor_.method == GENERATIVEQC_METHOD_MP2) capabilities_.supports_batch = false;
  }

  std::size_t atom_count() const noexcept override { return system_.atoms.size(); }
  const Capabilities& capabilities() const noexcept override { return capabilities_; }

  runtime::ExecutionResourceSnapshot execution_resources() const noexcept override {
    std::lock_guard<std::mutex> lock(mutex_);
    return execution_.resources();
  }
  std::optional<generativeqc_correlation_diagnostic> correlation_diagnostic() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_correlation_;
  }
  std::optional<CcPerformanceDiagnostic> cc_performance_diagnostic() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_performance_;
  }
  void invalidate_result() override {
    std::lock_guard<std::mutex> lock(mutex_);
    last_correlation_.reset();
    last_performance_.reset();
  }

  Result execute_impl(bool compute_forces) override { return execute_group(compute_forces, 0); }
  Result execute_components_impl(bool compute_forces, unsigned companions) override {
    const auto allowed =
        descriptor_.method == GENERATIVEQC_METHOD_MP2 ? kHfComponent : kHfComponent | kMp2Component;
    if (companions & ~allowed)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "unsupported DF correlation companions");
    return execute_group(compute_forces, companions);
  }

  FocalRccTriplesValues triples_on_reference(
      const std::shared_ptr<const hf::PhysicalReference>& retained,
      const molecule::ScalarHamiltonianIdentity& orbital,
      const molecule::ScalarHamiltonianIdentity& jk, std::size_t retained_bytes,
      const FocalRccTriplesOptions& options) {
    if (!full_df_reference_ || descriptor_.method != GENERATIVEQC_METHOD_DF_RCCSD_T ||
        !orbital.matches(system_))
      throw std::invalid_argument(
          "focal spatial triples requires a matching full-DF CUDA RHF determinant");
    if (options.require_common_fit_all_electron) {
      const bool has_ecp =
          !system_.ecp_terms.empty() ||
          std::any_of(system_.atoms.begin(), system_.atoms.end(),
                      [](const auto& a) { return a.ecp_core > 0; });
      if (has_ecp)
        throw std::invalid_argument(
            "focal spatial triples v1 requires an all-electron system; use v2 for scalar ECP");
      if (!jk.matches(auxiliary_))
        throw std::invalid_argument(
            "focal spatial triples v1 requires a common JK and correlation fit; use v2 for split "
            "fits");
    }
    if (!retained)
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "focal spatial triples requires a retained RHF snapshot");
    FocalRccTriplesValues triples;
    execute_group(options.forces, 0, retained, retained_bytes, &triples, options);
    return triples;
  }

  FocalRccF12bValues f12b_on_reference(
      const std::shared_ptr<const hf::PhysicalReference>& retained,
      const molecule::ScalarHamiltonianIdentity& orbital,
      const molecule::ScalarHamiltonianIdentity& jk, std::size_t retained_bytes,
      FocalRccF12bOptions options) {
    (void)jk;
    if (!full_df_reference_ || descriptor_.method != GENERATIVEQC_METHOD_DF_RCCSD_T ||
        !orbital.matches(system_))
      throw std::invalid_argument(
          "focal spatial CCSD-F12b requires a matching full-DF CUDA RHF determinant");
    if (!retained)
      throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                        "focal spatial CCSD-F12b requires a retained RHF snapshot");
    std::lock_guard<std::mutex> lock(mutex_);
    last_correlation_.reset();
    last_performance_.reset();
#if GENERATIVEQC_HAS_CUDA
    DFCorrelationDemand demand{
        .mp2 = false,
        .companions = 0,
        .df_reference = full_df_reference_,
        .triples_occupied_partition = false,
        .retained_reference = retained,
        .retained_reference_bytes = retained_bytes,
        .frozen_occupied = options.frozen_occupied,
        .triples_occupied_weights = {},
        .focal_triples_sources = false,
        .triples_fp32_w = false,
        .focal_ccsdf12b_sources = true,
        .f12 = std::move(options.f12),
    };
    auto native = run_df_correlation_group_native(execution_, system_, auxiliary_, descriptor_,
                                                  options.forces, std::move(demand), convergence_,
                                                  options.preconvergence);
    last_correlation_ = native.correlation;
    last_performance_ = native.primal;
    FocalRccF12bValues out;
    out.energy = native.correlation_energy;
    out.t1 = std::move(native.ccsd_t1);
    out.t2 = std::move(native.ccsd_t2);
    out.lambda1 = std::move(native.lambda1);
    out.lambda2 = std::move(native.lambda2);
    out.correlation_gradient = std::move(native.correlation_gradient);
    out.bar_f = std::move(native.bar_f);
    out.bar_c = std::move(native.bar_c);
    out.ccsd_fp32_executed = native.primal.ccsd_fp32_executed != 0;
    out.lambda_fp32_executed = options.forces && (native.lambda.df_fp32_gemm_calls > 0);
    out.ccsd_iterations = native.solver.iterations;
    out.lambda_iterations = native.lambda.iterations;
    out.f12_cuda_preparations = native.solver.f12_cuda_preparations;
    out.f12_cuda_rhs_evaluations = native.lambda.f12_cuda_rhs_evaluations;
    out.ccsd_seconds = native.primal.solver_seconds;
    out.lambda_seconds = native.lambda_seconds;
    out.source_response_seconds = native.source_response_seconds;
    out.total_seconds = native.total_seconds;
    return out;
#else
    (void)retained_bytes;
    (void)options;
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "focal spatial CCSD-F12b requires a CUDA-enabled build");
#endif
  }

  Result execute_group(bool compute_forces, unsigned companions,
                       const std::shared_ptr<const hf::PhysicalReference>& retained = {},
                       std::size_t retained_bytes = 0, FocalRccTriplesValues* triples = nullptr,
                       const FocalRccTriplesOptions& options = {}) {
    std::lock_guard<std::mutex> lock(mutex_);
    last_correlation_.reset();
    last_performance_.reset();
#if GENERATIVEQC_HAS_CUDA
    const auto active_preconv =
        triples ? options.preconvergence : preconvergence_;
    DFCorrelationDemand demand{
        .mp2 = descriptor_.method == GENERATIVEQC_METHOD_MP2,
        .companions = companions,
        .df_reference = full_df_reference_,
        .triples_occupied_partition = triples ? options.occupied_partition : false,
        .retained_reference = retained,
        .retained_reference_bytes = retained_bytes,
        .frozen_occupied = triples ? options.frozen_occupied : 0,
        .triples_occupied_weights = triples ? options.occupied_weights : std::vector<double>{},
        .focal_triples_sources = triples != nullptr && compute_forces,
        .triples_fp32_w = triples && (options.triples_fp32_w || options.triples_compensated_w),
        .triples_compensated_w = triples && options.triples_compensated_w,
    };
    auto native =
        companions || descriptor_.method == GENERATIVEQC_METHOD_MP2 || full_df_reference_ ||
                convergence_.relaxed() || active_preconv != core::DfPreconvergence::Legacy
            ? run_df_correlation_group_native(execution_, system_, auxiliary_, descriptor_,
                                              compute_forces, std::move(demand), convergence_,
                                              active_preconv)
            : run_df_ccsdt_native(execution_, system_, auxiliary_, descriptor_, compute_forces);
    native.primal.triples_seconds = native.triples_seconds;
    // Publish detached response work only after the entire endpoint succeeds.
    // Action counters cover the shared transpose operator, not the separate
    // independent audit. Phase times include their nested preparation/replay.
    if (compute_forces) {
      auto& performance = native.primal;
      const auto& lambda = native.lambda;
      performance.df_response_available = true;
      performance.lambda_seconds = native.lambda_seconds;
      performance.source_response_seconds = native.source_response_seconds;
      performance.orbital_seconds = native.orbital_seconds;
      performance.lambda_fp32_executed = lambda.df_fp32_gemm_calls > 0;
      performance.lambda_fp32_refused = lambda.df_mixed_precision_refused;
      performance.lambda_fp32_iterations = lambda.df_mixed_iterations;
      performance.lambda_fp64_iterations = lambda.df_strict_iterations;
      performance.lambda_fp32_actions = lambda.df_mixed_actions;
      performance.lambda_fp64_actions = lambda.df_strict_actions;
      performance.lambda_fp32_gemm_calls = lambda.df_fp32_gemm_calls;
      performance.lambda_fp32_cast_elements = lambda.df_precision_cast_elements;
      performance.lambda_auxiliary_batch_size = lambda.df_auxiliary_batch_size;
      performance.lambda_numeric_capacity_bytes = lambda.numeric_capacity_bytes;
      performance.lambda_owned_device_bytes = lambda.owned_device_bytes;
    }
    last_correlation_ = native.correlation;
    last_performance_ = native.primal;
    if (triples) {
      triples->energy = native.correlation.ccsd_t_triples_energy;
      triples->occupied = std::move(native.triples.occupied_energies);
      triples->correlation_gradient = std::move(native.correlation_gradient);
      triples->bar_f = std::move(native.bar_f);
      triples->bar_c = std::move(native.bar_c);
      triples->ccsd_fp32_executed = native.primal.ccsd_fp32_executed != 0;
      triples->lambda_fp32_executed = native.primal.lambda_fp32_executed != 0;
      triples->triples_fp32_w_executed = native.triples.triples_fp32_w_executed();
      triples->triples_fp32_w_forward_executed = native.triples.triples_fp32_w_forward_executed();
      triples->triples_fp32_w_reverse_executed = native.triples.triples_fp32_w_reverse_executed();
      triples->triples_reverse_w_executed = native.triples.reverse_w_executed;
      triples->triples_compensated_w_executed = native.triples.compensated_fp32_w_executed;
      triples->triples_fp32_forward_gemms = native.triples.fp32_forward_gemms;
      triples->triples_fp32_reverse_gemms = native.triples.fp32_reverse_gemms;
      triples->triples_workspace_bytes = native.triples.workspace_bytes;
      triples->triples_fp32_w_resource_fallback = native.triples.resource_fallback;
      triples->triples_fp32_gemms = native.triples.fp32_gemms;
      triples->triples_fp64_gemms = native.triples.fp64_gemms;
      triples->ccsd_seconds = native.primal.solver_seconds;
      triples->triples_seconds = native.triples_seconds;
      triples->lambda_seconds = native.lambda_seconds;
      triples->source_response_seconds = native.source_response_seconds;
    }
    return std::move(native.method_result);
#else
    (void)retained;
    (void)retained_bytes;
    (void)triples;
    (void)options;
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "public DF-RCCSD(T) requires a CUDA-enabled build");
#endif
  }

 private:
  Capabilities capabilities_;
  runtime::ExecutionContext execution_;
  core::System system_;
  core::System auxiliary_;
  bool full_df_reference_{};
  generativeqc_method_descriptor descriptor_{};
  core::CorrelationConvergence convergence_;
  core::DfPreconvergence preconvergence_{core::DfPreconvergence::Legacy};
  std::optional<generativeqc_correlation_diagnostic> last_correlation_;
  std::optional<CcPerformanceDiagnostic> last_performance_;
  mutable std::mutex mutex_;
};

}  // namespace

FocalRccTriplesValues focal_rcc_triples_values(
    PreparedCalculation& plan, const std::shared_ptr<const hf::PhysicalReference>& retained,
    const molecule::ScalarHamiltonianIdentity& orbital,
    const molecule::ScalarHamiltonianIdentity& jk, std::size_t retained_reference_bytes,
    bool occupied_partition) {
  FocalRccTriplesOptions options;
  options.occupied_partition = occupied_partition;
  options.require_common_fit_all_electron = true;
  return focal_rcc_triples_values(plan, retained, orbital, jk, retained_reference_bytes, options);
}

FocalRccTriplesValues focal_rcc_triples_values(
    PreparedCalculation& plan, const std::shared_ptr<const hf::PhysicalReference>& retained,
    const molecule::ScalarHamiltonianIdentity& orbital,
    const molecule::ScalarHamiltonianIdentity& jk, std::size_t retained_reference_bytes,
    const FocalRccTriplesOptions& options) {
  auto* cc = dynamic_cast<DfRccsdtPrepared*>(&plan);
  if (!cc) throw std::invalid_argument("focal spatial triples requires a native DF-RCCSD(T) plan");
  return cc->triples_on_reference(retained, orbital, jk, retained_reference_bytes, options);
}

FocalRccF12bValues focal_rcc_f12b_values(
    PreparedCalculation& plan, const std::shared_ptr<const hf::PhysicalReference>& retained,
    const molecule::ScalarHamiltonianIdentity& orbital,
    const molecule::ScalarHamiltonianIdentity& jk, std::size_t retained_reference_bytes,
    FocalRccF12bOptions options) {
  auto* cc = dynamic_cast<DfRccsdtPrepared*>(&plan);
  if (!cc)
    throw std::invalid_argument("focal spatial CCSD-F12b requires a native DF-RCCSD(T) plan");
  return cc->f12b_on_reference(retained, orbital, jk, retained_reference_bytes, std::move(options));
}

generativeqc_status validate_df_rccsdt_system(generativeqc_method method,
                                              const core::System& system, std::string& detail) {
  return validate_rccsd_system(method, system, detail);
}

std::unique_ptr<PreparedCalculation> prepare_df_rccsdt_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const generativeqc_method_descriptor& descriptor, const core::System* auxiliary_basis) {
  runtime::ExecutionContext execution(context);
  validate_descriptor(descriptor, execution);
  if (auxiliary_basis == nullptr)
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "auxiliary basis must be resolved at the API boundary");
  core::System auxiliary = *auxiliary_basis;
  if (auxiliary.atoms.size() != system.atoms.size())
    throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "DF-RCCSD(T) auxiliary geometry differs from the orbital system");
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
    if (auxiliary.atoms[atom].atomic_number != system.atoms[atom].atomic_number ||
        auxiliary.atoms[atom].position != system.atoms[atom].position)
      throw MethodError(GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                        "DF-RCCSD(T) auxiliary geometry differs from the orbital system");
  return std::make_unique<DfRccsdtPrepared>(capabilities, std::move(execution), system,
                                            std::move(auxiliary), descriptor);
}

}  // namespace generativeqc::methods::detail
