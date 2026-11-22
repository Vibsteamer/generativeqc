#include "scf/solver/mean_field_driver.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "runtime/host_component_trace.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/fock_symmetry.hpp"
#include "scf/gradient/hf_gradient.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/diis.hpp"
#include "scf/solver/final_state.hpp"
#include "scf/solver/proposal_control.hpp"
#include "solver/self_consistent.hpp"

namespace generativeqc::scf::solver {
namespace {
using initial_guess::prepare_initial_density;
using initial_guess::prepare_initial_uhf_density;
using initial_guess::spin_occupations;
using reference::commutator_residual;
using reference::concatenate;
using reference::density_from_orbitals;
using reference::density_rms;
using reference::EigenResult;
using reference::electronic_energy;
using reference::generalized_eigen;
using reference::Matrix;
using reference::multiply;
using reference::residual_rms;
using reference::split_spin_matrices;
using reference::uhf_electronic_energy;

template <class... Vectors>
void sample_scf_buffers(const PreparedFockPlan& plan, const Diis& diis,
                        const Vectors&... vectors) noexcept {
  if (!runtime::cpu_resource_observation.active) return;
  runtime::sample_cpu_capacity(runtime::add_capacity(
      plan.cpu_observation_capacity(),
      runtime::add_capacity(diis.numeric_capacity(), runtime::vector_capacities(vectors...))));
}

void add_jk_in_place(DirectJkMatrices& target, const DirectJkMatrices& base) {
  if (target.nbf != base.nbf) throw std::logic_error("incremental J/K AO dimension mismatch");
  const auto add = [](std::vector<double>& values, const std::vector<double>& anchor) {
    if (values.size() != anchor.size())
      throw std::logic_error("incremental J/K selected-output mismatch");
    for (std::size_t i = 0; i < values.size(); ++i) values[i] += anchor[i];
  };
  add(target.coulomb, base.coulomb);
  add(target.exchange_alpha, base.exchange_alpha);
  add(target.exchange_beta, base.exchange_beta);
}

/** Exact accepted-iterate incremental J/K baseline for #990.
 *
 * Only the main SCF trajectory mutates the anchor. Proposal/audit builds and
 * final physical-state rebuilds use PreparedFockPlan directly, so a rejected
 * trial can never become the next anchor. This CPU lower declares exact-linear
 * execution with no density-weighted quartet skipping; activation and refresh
 * cadence come from the shared pre-lowering policy.
 */
class ExactIncrementalDirectJk {
 public:
  ExactIncrementalDirectJk(const PreparedFockPlan& plan, const ScfOptions& options,
                           IncrementalDirectJkDiagnostic& diagnostic)
      : plan_(plan),
        policy_(resolve_incremental_direct_jk_policy(
            options, {direct_jk_incremental_exact_eligible(plan.strategy()), false, false})),
        diagnostic_(diagnostic) {
    diagnostic_.requested = policy_.requested;
    diagnostic_.active = policy_.active;
  }

  DirectJkMatrices build(const Matrix& density, const Matrix& beta = {}) {
    if (!diagnostic_.active) return plan_.build(density, beta);
    const runtime::CpuRetainedCapacity anchor_capacity(numeric_capacity());
    if (direct_jk_incremental_requires_full_build(anchored_, delta_updates_since_full_, policy_)) {
      const bool refresh = anchored_;
      auto current = plan_.build(density, beta);
      anchor_density_ = density;
      anchor_beta_ = beta;
      anchor_jk_ = current;
      anchored_ = true;
      delta_updates_since_full_ = 0;
      ++diagnostic_.anchor_full_builds;
      if (refresh) ++diagnostic_.periodic_rebuilds;
      return current;
    }
    if (density.size() != anchor_density_.size() || beta.size() != anchor_beta_.size())
      throw std::logic_error("incremental J/K density shape changed within one SCF solve");
    Matrix delta(density.size());
    Matrix delta_beta(beta.size());
    const runtime::CpuRetainedCapacity delta_capacity(
        runtime::vector_capacities(delta, delta_beta));
    for (std::size_t i = 0; i < density.size(); ++i) {
      delta[i] = density[i] - anchor_density_[i];
      diagnostic_.max_abs_delta_density =
          std::max(diagnostic_.max_abs_delta_density, std::abs(delta[i]));
    }
    for (std::size_t i = 0; i < beta.size(); ++i) {
      delta_beta[i] = beta[i] - anchor_beta_[i];
      diagnostic_.max_abs_delta_density =
          std::max(diagnostic_.max_abs_delta_density, std::abs(delta_beta[i]));
    }
    auto current = plan_.build(delta, delta_beta);
    add_jk_in_place(current, anchor_jk_);
    anchor_density_ = density;
    anchor_beta_ = beta;
    anchor_jk_ = current;
    ++delta_updates_since_full_;
    ++diagnostic_.delta_builds;
    ++diagnostic_.anchor_updates;
    return current;
  }

  std::size_t numeric_capacity() const noexcept {
    return runtime::vector_capacities(anchor_density_, anchor_beta_, anchor_jk_.coulomb,
                                      anchor_jk_.exchange_alpha, anchor_jk_.exchange_beta);
  }

  void note_bypass_full_build() noexcept {
    if (diagnostic_.active) ++diagnostic_.bypass_full_builds;
  }
  void note_post_scf_full_builds(std::uint64_t count) noexcept {
    if (diagnostic_.active) diagnostic_.post_scf_full_builds += count;
  }

 private:
  const PreparedFockPlan& plan_;
  IncrementalDirectJkPolicy policy_{};
  IncrementalDirectJkDiagnostic& diagnostic_;
  bool anchored_{};
  unsigned delta_updates_since_full_{};
  Matrix anchor_density_;
  Matrix anchor_beta_;
  DirectJkMatrices anchor_jk_;
};

FockMatrices assemble_host_fock(const PreparedFockPlan& plan, const Matrix& hcore,
                                const DirectJkMatrices& jk) {
  const auto& strategy = plan.strategy();
  auto fock = assemble_fock(strategy, hcore, jk);
  if (strategy.backend == FockBackend::Cuda &&
      strategy.spec.coulomb.approximation == FockApproximation::DensityFitted &&
      strategy.spec.exchange.approximation == FockApproximation::DensityFitted) {
    // Keep independently accumulated CUDA J/K roundoff out of Pulay history
    // and use the same physical matrix convention for finalization.
    preserve_fock_symmetry(fock.alpha, jk.nbf);
    if (!fock.beta.empty()) preserve_fock_symmetry(fock.beta, jk.nbf);
  }
  return fock;
}

std::pair<Matrix, Matrix> build_uhf_focks(const PreparedFockPlan& plan, const Matrix& hcore,
                                          const Matrix& alpha_density, const Matrix& beta_density) {
  auto fock = assemble_host_fock(plan, hcore, plan.build(alpha_density, beta_density));
  return {std::move(fock.alpha), std::move(fock.beta)};
}

Matrix build_fock(const PreparedFockPlan& plan, const Matrix& hcore, const Matrix& density) {
  return assemble_host_fock(plan, hcore, plan.build(density)).alpha;
}

/** Substitute only the provider of this actual matrix. Iterative matrices may
 * be DIIS-extrapolated, so their frames never authorize physical-state reuse. */
EigenResult diagonalize(const PreparedFockPlan& plan, const Matrix& matrix,
                        const integrals::IntegralData& ints, const Matrix& orthogonalizer,
                        PreparedFockPlan::EigenUse use,
                        const initial_guess::EigenOperation& target_eigen) {
  namespace trace = runtime::host_trace;
  trace::Reason reason(use == PreparedFockPlan::EigenUse::Iteration
                           ? trace::EigenReason::iteration
                           : trace::EigenReason::final_fock);
  // Select on the actual provider backend. An internal CPU target callback must
  // never replace the existing CUDA ordinary-eigen callback.
  const auto operation = plan.strategy().backend == FockBackend::Cpu && target_eigen
                             ? target_eigen
                             : plan.eigen_operation(use);
  return operation ? operation(matrix, &ints.overlap, &orthogonalizer, ints.nbf)
                   : generalized_eigen(matrix, orthogonalizer, ints.nbf);
}

void finalize_scf(const PreparedFockPlan& plan, const integrals::IntegralData& ints,
                  const Matrix& orthogonalizer, std::size_t occupied, Matrix& density,
                  const ScfOptions& options, ScfResult& result,
                  const initial_guess::EigenOperation& target_eigen) {
  const std::size_t n = ints.nbf;
  runtime::host_trace::Region final_trace("host_finalization", n);
  Matrix final_fock = build_fock(plan, ints.hcore, density);
  EigenResult orbitals = diagonalize(plan, final_fock, ints, orthogonalizer,
                                     PreparedFockPlan::EigenUse::Finalization, target_eigen);
  density = density_from_orbitals(orbitals.vectors, n, occupied);
  final_fock = build_fock(plan, ints.hcore, density);
  const auto evaluate_energy = options.stabilize_focal_rhf_convergence
                                   ? reference::compensated_electronic_energy : electronic_energy;
  result.energy = evaluate_energy(density, ints.hcore, final_fock) + ints.nuclear_repulsion;
  if (options.compute_forces) {
    const Matrix weighted = reference::physical_weighted_density(density, final_fock, n, 2.0);
    result.forces =
        gradient::analytic_forces(ints, density, weighted, plan.energy_derivative(density));
  }
  result.density = density;
}

double congruence_rms(const Matrix& matrix, const Matrix& basis, std::size_t n) {
  return residual_rms(multiply(reference::transpose(basis, n), multiply(matrix, basis, n), n));
}

std::uint64_t next_host_final_state_epoch() {
  static std::atomic<std::uint64_t> next{1};
  const auto epoch = next.fetch_add(1, std::memory_order_relaxed);
  if (epoch == 0 || epoch == std::numeric_limits<std::uint64_t>::max())
    throw std::runtime_error("host final-state epoch exhausted");
  return epoch;
}

unsigned finalize_uhf(const PreparedFockPlan& plan, const integrals::IntegralData& ints,
                      const Matrix& orthogonalizer, std::size_t alpha_occupied,
                      std::size_t beta_occupied, Matrix& alpha_density, Matrix& beta_density,
                      const ScfOptions& options, ScfResult& result,
                      const initial_guess::EigenOperation& target_eigen) {
  const std::size_t n = ints.nbf;
  runtime::host_trace::Region final_trace("host_finalization", n);
  auto [alpha_fock, beta_fock] = build_uhf_focks(plan, ints.hcore, alpha_density, beta_density);
  EigenResult alpha_orbitals = diagonalize(plan, alpha_fock, ints, orthogonalizer,
                                           PreparedFockPlan::EigenUse::Finalization, target_eigen);
  EigenResult beta_orbitals = diagonalize(plan, beta_fock, ints, orthogonalizer,
                                          PreparedFockPlan::EigenUse::Finalization, target_eigen);
  unsigned physical_fock_builds = 1;

  if (options.compute_forces || options.export_physical_reference) {
    const auto epoch = next_host_final_state_epoch();
    FinalStateIdentity identity{
        {epoch, epoch, 2, 1}, epoch, plan.strategy(), {alpha_occupied, beta_occupied}};
    FinalFrameCandidate candidate{identity, 1, true, {alpha_orbitals, beta_orbitals}};
    const PhysicalFockOperation physical = [&](const auto& current, const auto& densities) {
      if (densities.size() != 2)
        throw std::runtime_error("UHF final-state provider lost a spin density");
      auto [alpha, beta] = build_uhf_focks(plan, ints.hcore, densities[0], densities[1]);
      return PhysicalFockFrame{current, true, {std::move(alpha), std::move(beta)}};
    };
    const initial_guess::EigenOperation eigen = [&](const auto& fock, const auto*, const auto*,
                                                    auto) {
      return diagonalize(plan, fock, ints, orthogonalizer, PreparedFockPlan::EigenUse::Finalization,
                         target_eigen);
    };
    // Validate the converged determinant before adopting a new projector.
    // Pre-projecting here can amplify a slow or unstable spin fixed-point
    // mode even when the retained determinant already passes the physical
    // proof. Rejected candidates still receive the same bounded correction:
    // at most 33 selector evaluations plus the initial physical Fock above.
    auto selected = select_final_state(identity, ints.overlap, ints.hcore, orthogonalizer,
                                       ints.nuclear_repulsion, {alpha_density, beta_density},
                                       &candidate, physical, eigen,
                                       {options.density_tolerance, options.energy_tolerance, 32,
                                        options.export_physical_reference, false},
                                       options.compute_forces);
    physical_fock_builds += selected.fock_evaluations;
    if (!selected.state)
      throw std::runtime_error("CPU UHF final-state correction failed: " + selected.detail);

    alpha_density = std::move(selected.state->density[0]);
    beta_density = std::move(selected.state->density[1]);
    alpha_fock = std::move(selected.state->fock[0]);
    beta_fock = std::move(selected.state->fock[1]);
    result.energy = selected.state->diagnostic.energy;
    if (options.export_physical_reference) {
      auto reference = std::make_shared<hf::UnrestrictedPhysicalReference>();
      reference->nbf = n;
      reference->nocc = {alpha_occupied, beta_occupied};
      reference->overlap = ints.overlap;
      reference->hcore = ints.hcore;
      reference->density = {alpha_density, beta_density};
      reference->fock = {alpha_fock, beta_fock};
      for (std::size_t spin = 0; spin < 2; ++spin) {
        reference->coefficients[spin] = std::move(selected.state->orbitals[spin].vectors);
        reference->orbital_energies[spin] = std::move(selected.state->orbitals[spin].values);
      }
      reference->energy = result.energy;
      reference->commutator_residual = selected.state->diagnostic.maximum_commutator;
      reference->canonical_density_drift = selected.state->diagnostic.maximum_density_error;
      reference->canonical_error = selected.state->diagnostic.maximum_canonical_error;
      reference->source_identity.emplace(plan.system());
      reference->source_charge = plan.system().charge;
      reference->source_electrons = plan.system().electron_count;
      reference->source_multiplicity = plan.system().multiplicity;
      core::validate_electronic_reference_shape(reference->electronic_reference());
      result.unrestricted_reference = std::move(reference);
    }
    if (options.compute_forces)
      result.forces = gradient::analytic_uhf_forces(
          ints, alpha_density, beta_density, selected.state->weighted_density[0],
          selected.state->weighted_density[1], plan.energy_derivative(alpha_density, beta_density));
  } else {
    alpha_density = density_from_orbitals(alpha_orbitals.vectors, n, alpha_occupied, 1.0);
    beta_density = density_from_orbitals(beta_orbitals.vectors, n, beta_occupied, 1.0);
    std::tie(alpha_fock, beta_fock) =
        build_uhf_focks(plan, ints.hcore, alpha_density, beta_density);
    ++physical_fock_builds;
    result.energy =
        uhf_electronic_energy(alpha_density, beta_density, ints.hcore, alpha_fock, beta_fock) +
        ints.nuclear_repulsion;
  }
  result.density = concatenate(alpha_density, beta_density);
  return physical_fock_builds;
}

Matrix apply_level_shift(Matrix fock, const Matrix& overlap, const Matrix& spin_density,
                         std::size_t n, double level_shift) {
  if (level_shift <= 0.0) return fock;
  // Virtual-space projector in AO basis: S - S * D_spin * S (zero on occupied space).
  const Matrix sds = multiply(overlap, multiply(spin_density, overlap, n), n);
  for (std::size_t i = 0; i < fock.size(); ++i)
    fock[i] += level_shift * (overlap[i] - sds[i]);
  return fock;
}

Matrix blend_density(const Matrix& current, const Matrix& proposed, double damping) {
  if (damping >= 1.0) return proposed;
  Matrix blended(current.size());
  const double keep = 1.0 - damping;
  for (std::size_t i = 0; i < current.size(); ++i)
    blended[i] = keep * current[i] + damping * proposed[i];
  return blended;
}

}  // namespace

ScfResult run_rhf_host_plan(const core::System& system, const ScfOptions& options,
                            const integrals::IntegralData& ints, const PreparedFockPlan& plan,
                            const std::vector<double>* initial_density,
                            initial_guess::OverlapOrthogonalizer* overlap_cache,
                            const initial_guess::EigenOperation& target_eigen) {
  const std::size_t n = ints.nbf;
  const std::size_t occupied = static_cast<std::size_t>(system.electron_count / 2);
  if (occupied > n) {
    throw std::runtime_error("basis has fewer orbitals than occupied electron pairs");
  }
  const Matrix orthogonalizer = plan.overlap_orthogonalizer(overlap_cache);
  std::optional<EigenResult> initial_orbitals;
  Matrix density = prepare_initial_density(system, ints, orthogonalizer, occupied, initial_density,
                                           initial_orbitals,
                                           initial_guess::InitialOrbitalRequest::ColdDensityOnly,
                                           plan.eigen_operation(PreparedFockPlan::EigenUse::Setup));
  // Only a cold seed carries a core frame. Warm consumers solve their first
  // target Fock before reading orbitals; RKS packs its initial factor cold-only.
  EigenResult orbitals = std::move(initial_orbitals).value_or(EigenResult{});
  if (options.strict_initial_density && initial_density) {
    validate_seed(ints.overlap, *initial_density, n, {static_cast<unsigned>(system.electron_count)},
                  2.0, plan.eigen_operation(PreparedFockPlan::EigenUse::Setup));
    density = *initial_density;
  }
  Diis diis(options.diis_history);
  const auto generation = new_scf_generation(options);
  unsigned proposal_failures = 0;

  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  ExactIncrementalDirectJk incremental_jk(plan, options, result.incremental_direct_jk);
  const bool require_residual = options.hooks && options.hooks->propose;
  const bool stabilize = options.stabilize_focal_rhf_convergence;
  const auto evaluate_energy = stabilize ? reference::compensated_electronic_energy : electronic_energy;
  Matrix density_metric_basis;

  struct RhfEvaluation {
    Matrix fock;
    Matrix residual;
    Matrix next_density;
    double energy{};
    double state_rms{};
    double residual_rms{};
    double ao_density_rms{};
  };

  const ::generativeqc::solver::SelfConsistentPolicy policy{
      options.max_iterations, options.energy_tolerance, options.density_tolerance,
      options.density_tolerance, require_residual};
  auto outcome = ::generativeqc::solver::run_self_consistent(
      std::move(density), policy,
      [&](const Matrix& current_density, unsigned) {
        ++result.fock_builds;
        Matrix fock =
            assemble_host_fock(plan, ints.hcore, incremental_jk.build(current_density)).alpha;
        const runtime::CpuRetainedCapacity anchor_capacity(incremental_jk.numeric_capacity());
        const double energy =
            evaluate_energy(current_density, ints.hcore, fock) + ints.nuclear_repulsion;
        Matrix residual = commutator_residual(fock, current_density, ints.overlap, n);
        Matrix effective_fock = options.enable_diis ? diis.update(fock, residual) : fock;
        if (options.level_shift > 0.0 && (result.iterations > 0 || initial_density != nullptr)) {
          Matrix spin_density(current_density.size());
          for (std::size_t i = 0; i < spin_density.size(); ++i)
            spin_density[i] = 0.5 * current_density[i];
          effective_fock =
              apply_level_shift(std::move(effective_fock), ints.overlap, spin_density, n,
                                options.level_shift);
        }
        orbitals = diagonalize(plan, effective_fock, ints, orthogonalizer,
                               PreparedFockPlan::EigenUse::Iteration, target_eigen);
        Matrix diag_density = density_from_orbitals(orbitals.vectors, n, occupied);

        const double ao_rms = density_rms(diag_density, current_density);
        double state_rms = ao_rms;
        // With diffuse OBS, a well-conditioned physical step can look large
        // in raw AO coefficients. Only consider the metric step when the
        // energy and the unextrapolated physical Fock are also stationary.
        if (stabilize && ao_rms >= options.density_tolerance && result.iterations != 0 &&
            std::abs(energy - result.energy) < options.energy_tolerance &&
            congruence_rms(residual, orthogonalizer, n) < options.density_tolerance) {
          if (density_metric_basis.empty())
            density_metric_basis = multiply(ints.overlap, orthogonalizer, n);
          Matrix delta(diag_density.size());
          for (std::size_t i = 0; i < delta.size(); ++i)
            delta[i] = diag_density[i] - current_density[i];
          const double metric_rms = congruence_rms(delta, density_metric_basis, n);
          if (metric_rms < options.density_tolerance) state_rms = metric_rms;
        }
        Matrix next_density =
            (options.density_damping > 0.0 && options.density_damping < 1.0 &&
             state_rms >= options.density_tolerance)
                ? blend_density(current_density, diag_density, options.density_damping)
                : std::move(diag_density);
        sample_scf_buffers(plan, diis, orthogonalizer, current_density, fock, residual,
                           effective_fock, orbitals.values, orbitals.vectors, next_density,
                           density_metric_basis);
        const double physical_residual_rms = require_residual ? residual_rms(residual) : 0.0;
        return RhfEvaluation{std::move(fock), std::move(residual), std::move(next_density),
                             energy, state_rms, physical_residual_rms, ao_rms};
      },
      [&](const Matrix& current_density, RhfEvaluation evaluation,
          const ::generativeqc::solver::SelfConsistentProgress& progress) {
        Matrix next_density = std::move(evaluation.next_density);
        if (options.hooks) {
          next_density = safeguarded_update(
              options, generation, progress.iteration, ints.overlap, current_density,
              evaluation.fock, evaluation.residual, std::move(next_density),
              {static_cast<unsigned>(system.electron_count)}, 2.0, result, diis, proposal_failures,
              progress.converged, [&](const Matrix& trial) {
                const runtime::CpuRetainedCapacity anchor_capacity(
                    incremental_jk.numeric_capacity());
                incremental_jk.note_bypass_full_build();
                const Matrix trial_fock = build_fock(plan, ints.hcore, trial);
                return std::make_pair(
                    electronic_energy(trial, ints.hcore, trial_fock) + ints.nuclear_repulsion,
                    commutator_residual(trial_fock, trial, ints.overlap, n));
              });
        }
        return next_density;
      },
      [&](const ::generativeqc::solver::SelfConsistentProgress& progress, const RhfEvaluation& evaluation) {
        result.iterations = progress.iteration;
        result.energy = progress.energy;
        result.energy_change = progress.energy_change;
        result.density_rms = evaluation.ao_density_rms;
      });
  density = std::move(outcome.state);
  result.converged = outcome.converged;

  if (!result.converged) {
    // Failed traces retain the last iterate, never a converged reference.
    result.density = density;
    return result;
  }
  result.fock_builds += 2;  // Physical rebuilds performed by finalization.
  incremental_jk.note_post_scf_full_builds(2);
  const runtime::CpuRetainedCapacity anchor_capacity(incremental_jk.numeric_capacity());

  // Rebuild and diagonalize the un-extrapolated converged Fock matrix. The
  // resulting orbitals define the energy-weighted density in the Pulay term.
  finalize_scf(plan, ints, orthogonalizer, occupied, density, options, result,
               target_eigen);
  return result;
}

ScfResult run_uhf_host_plan(const core::System& system, const ScfOptions& options,
                            const integrals::IntegralData& ints, const PreparedFockPlan& plan,
                            const std::vector<double>* initial_density,
                            initial_guess::OverlapOrthogonalizer* overlap_cache,
                            const initial_guess::EigenOperation& target_eigen) {
  const std::size_t n = ints.nbf;
  const auto [alpha_occupied, beta_occupied] = spin_occupations(system);
  if (alpha_occupied > n || beta_occupied > n) {
    throw std::runtime_error("basis has fewer orbitals than required UHF spin occupations");
  }
  const Matrix orthogonalizer = plan.overlap_orthogonalizer(overlap_cache);
  std::optional<EigenResult> initial_alpha, initial_beta;
  auto [alpha_density, beta_density] = prepare_initial_uhf_density(
      ints, orthogonalizer, alpha_occupied, beta_occupied, initial_density, initial_alpha,
      initial_beta, initial_guess::InitialOrbitalRequest::ColdDensityOnly,
      plan.eigen_operation(PreparedFockPlan::EigenUse::Setup));
  EigenResult alpha_orbitals = std::move(initial_alpha).value_or(EigenResult{});
  EigenResult beta_orbitals = std::move(initial_beta).value_or(EigenResult{});
  if (options.strict_initial_density && initial_density) {
    validate_seed(ints.overlap, *initial_density, n,
                  {static_cast<unsigned>(alpha_occupied), static_cast<unsigned>(beta_occupied)},
                  1.0, plan.eigen_operation(PreparedFockPlan::EigenUse::Setup));
    std::tie(alpha_density, beta_density) = split_spin_matrices(*initial_density, n * n);
  }
  Diis diis(options.diis_history);
  const auto generation = new_scf_generation(options);
  unsigned proposal_failures = 0;

  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  ExactIncrementalDirectJk incremental_jk(plan, options, result.incremental_direct_jk);
  const bool require_residual = options.hooks && options.hooks->propose;
  const bool stabilize = options.stabilize_focal_rhf_convergence;
  Matrix density_metric_basis;

  struct UhfState {
    Matrix alpha;
    Matrix beta;
  };
  struct UhfEvaluation {
    Matrix physical_fock;
    Matrix physical_residual;
    Matrix next_alpha;
    Matrix next_beta;
    double energy{};
    double state_rms{};
    double residual_rms{};
  };

  const ::generativeqc::solver::SelfConsistentPolicy policy{
      options.max_iterations, options.energy_tolerance, options.density_tolerance,
      options.density_tolerance, require_residual};
  auto outcome = ::generativeqc::solver::run_self_consistent(
      UhfState{std::move(alpha_density), std::move(beta_density)}, policy,
      [&](const UhfState& state, unsigned) {
        auto fock = assemble_host_fock(plan, ints.hcore, incremental_jk.build(state.alpha, state.beta));
        const runtime::CpuRetainedCapacity anchor_capacity(incremental_jk.numeric_capacity());
        auto alpha_fock = std::move(fock.alpha);
        auto beta_fock = std::move(fock.beta);
        ++result.fock_builds;
        const double energy =
            uhf_electronic_energy(state.alpha, state.beta, ints.hcore, alpha_fock, beta_fock) +
            ints.nuclear_repulsion;
        const Matrix alpha_residual = commutator_residual(alpha_fock, state.alpha, ints.overlap, n);
        const Matrix beta_residual = commutator_residual(beta_fock, state.beta, ints.overlap, n);
        Matrix physical_fock = concatenate(alpha_fock, beta_fock);
        Matrix physical_residual = concatenate(alpha_residual, beta_residual);
        const Matrix effective_joined =
            options.enable_diis ? diis.update(physical_fock, physical_residual) : physical_fock;
        std::tie(alpha_fock, beta_fock) = split_spin_matrices(effective_joined, n * n);
        if (options.level_shift > 0.0 && (result.iterations > 0 || initial_density != nullptr)) {
          alpha_fock = apply_level_shift(std::move(alpha_fock), ints.overlap, state.alpha, n,
                                         options.level_shift);
          beta_fock = apply_level_shift(std::move(beta_fock), ints.overlap, state.beta, n,
                                        options.level_shift);
        }
        alpha_orbitals = diagonalize(plan, alpha_fock, ints, orthogonalizer,
                                     PreparedFockPlan::EigenUse::Iteration, target_eigen);
        beta_orbitals = diagonalize(plan, beta_fock, ints, orthogonalizer,
                                    PreparedFockPlan::EigenUse::Iteration, target_eigen);
        Matrix diag_alpha = density_from_orbitals(alpha_orbitals.vectors, n, alpha_occupied, 1.0);
        Matrix diag_beta = density_from_orbitals(beta_orbitals.vectors, n, beta_occupied, 1.0);

        const double ao_rms =
            density_rms(concatenate(diag_alpha, diag_beta), concatenate(state.alpha, state.beta));
        double state_rms = ao_rms;
        if (stabilize && ao_rms >= options.density_tolerance && result.iterations != 0 &&
            std::abs(energy - result.energy) < options.energy_tolerance &&
            std::max(congruence_rms(alpha_residual, orthogonalizer, n),
                     congruence_rms(beta_residual, orthogonalizer, n)) <
                options.density_tolerance) {
          if (density_metric_basis.empty())
            density_metric_basis = multiply(ints.overlap, orthogonalizer, n);
          Matrix da(diag_alpha.size()), db(diag_beta.size());
          for (std::size_t i = 0; i < da.size(); ++i) {
            da[i] = diag_alpha[i] - state.alpha[i];
            db[i] = diag_beta[i] - state.beta[i];
          }
          const double metric_rms = std::max(congruence_rms(da, density_metric_basis, n),
                                             congruence_rms(db, density_metric_basis, n));
          if (metric_rms < options.density_tolerance) state_rms = metric_rms;
        }
        const bool damp = options.density_damping > 0.0 && options.density_damping < 1.0 &&
                          state_rms >= options.density_tolerance;
        Matrix next_alpha =
            damp ? blend_density(state.alpha, diag_alpha, options.density_damping)
                 : std::move(diag_alpha);
        Matrix next_beta =
            damp ? blend_density(state.beta, diag_beta, options.density_damping)
                 : std::move(diag_beta);

        sample_scf_buffers(plan, diis, orthogonalizer, state.alpha, state.beta, alpha_fock,
                           beta_fock, alpha_residual, beta_residual, physical_fock,
                           physical_residual, effective_joined, alpha_orbitals.values,
                           alpha_orbitals.vectors, beta_orbitals.values, beta_orbitals.vectors,
                           next_alpha, next_beta);

        const double physical_residual_rms =
            require_residual ? residual_rms(physical_residual) : 0.0;
        return UhfEvaluation{std::move(physical_fock),
                             std::move(physical_residual),
                             std::move(next_alpha),
                             std::move(next_beta),
                             energy,
                             state_rms,
                             physical_residual_rms};
      },
      [&](const UhfState& state, UhfEvaluation evaluation,
          const ::generativeqc::solver::SelfConsistentProgress& progress) {
        if (!options.hooks) {
          return UhfState{std::move(evaluation.next_alpha), std::move(evaluation.next_beta)};
        }

        const Matrix next = safeguarded_update(
            options, generation, progress.iteration, ints.overlap,
            concatenate(state.alpha, state.beta), evaluation.physical_fock,
            evaluation.physical_residual, concatenate(evaluation.next_alpha, evaluation.next_beta),
            {static_cast<unsigned>(alpha_occupied), static_cast<unsigned>(beta_occupied)}, 1.0,
            result, diis, proposal_failures, progress.converged, [&](const Matrix& trial) {
              const auto [a, b] = split_spin_matrices(trial, n * n);
              const runtime::CpuRetainedCapacity anchor_capacity(incremental_jk.numeric_capacity());
              incremental_jk.note_bypass_full_build();
              const auto [fa, fb] = build_uhf_focks(plan, ints.hcore, a, b);
              return std::make_pair(
                  uhf_electronic_energy(a, b, ints.hcore, fa, fb) + ints.nuclear_repulsion,
                  concatenate(commutator_residual(fa, a, ints.overlap, n),
                              commutator_residual(fb, b, ints.overlap, n)));
            });
        auto [next_alpha, next_beta] = split_spin_matrices(next, n * n);
        return UhfState{std::move(next_alpha), std::move(next_beta)};
      },
      [&](const ::generativeqc::solver::SelfConsistentProgress& progress, const UhfEvaluation&) {
        result.iterations = progress.iteration;
        result.energy = progress.energy;
        result.energy_change = progress.energy_change;
        result.density_rms = progress.state_rms;
      },
      [&](const ::generativeqc::solver::SelfConsistentProgress&, const UhfEvaluation& evaluation) {
        // Loose density controls cannot hand a nonstationary spin determinant
        // to the bounded final selector. Refine under the same SCF budget
        // until its existing physical commutator ceiling can pass; the full
        // canonical/fixed-point proof remains owned by finalization.
        const double raw_max = reference::residual_max_abs(evaluation.physical_residual);
        if (raw_max <= std::min(1e-8, options.density_tolerance)) return true;
        if (!stabilize || raw_max > 1e-8) return false;
        const auto [a_res, b_res] = split_spin_matrices(evaluation.physical_residual, n * n);
        return std::max(congruence_rms(a_res, orthogonalizer, n),
                        congruence_rms(b_res, orthogonalizer, n)) <= options.density_tolerance;
      });
  alpha_density = std::move(outcome.state.alpha);
  beta_density = std::move(outcome.state.beta);
  result.converged = outcome.converged;

  if (!result.converged) {
    // Failed traces retain the last iterate, never a converged reference.
    result.density = concatenate(alpha_density, beta_density);
    return result;
  }
  const runtime::CpuRetainedCapacity anchor_capacity(incremental_jk.numeric_capacity());

  // Rebuild from the un-extrapolated converged spin Fock matrices. Force
  // publication then uses the shared bounded physical final-state correction,
  // preserving the requested SCF controls while rejecting a nonstationary D/F
  // pair before constructing Pulay weights.
  const unsigned post_scf_builds =
      finalize_uhf(plan, ints, orthogonalizer, alpha_occupied, beta_occupied, alpha_density,
                   beta_density, options, result, target_eigen);
  result.fock_builds += post_scf_builds;
  incremental_jk.note_post_scf_full_builds(post_scf_builds);
  return result;
}

}  // namespace generativeqc::scf::solver
