// Python bridge to the native canonical DF-HF owner at one geometry.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "generativeqc/ucc_fitting.h"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/df_progress_trace.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/mean_field.hpp"
#include "scf/solver/uhf_second_order.hpp"

struct generativeqc_focal_reference {
  std::shared_ptr<const generativeqc::hf::UnrestrictedPhysicalReference> uhf;
  generativeqc::molecule::ScalarHamiltonianIdentity jk;
  generativeqc_backend backend;
  int device;
  std::shared_ptr<const generativeqc::hf::PhysicalReference> rhf;
};

namespace {
void record_oom_detail_noalloc(std::string& detail, const char* msg) noexcept {
  // Write only within existing string capacity (including SSO) so OOM reporting
  // never heap-allocates or throws across an extern "C" boundary.
  try {
    detail.clear();
    if (msg) {
      const std::size_t len = std::strlen(msg);
      if (len <= detail.capacity()) {
        detail.append(msg, len);
        return;
      }
    }
    constexpr char kFallback[] = "out of memory";
    if (sizeof(kFallback) - 1 <= detail.capacity()) detail.append(kFallback, sizeof(kFallback) - 1);
  } catch (...) {
    detail.clear();
  }
}

std::string scientific(double value) {
  char text[32];
  std::snprintf(text, sizeof text, "%.3e", value);
  return text;
}

std::string fixed(double value) {
  char text[32];
  std::snprintf(text, sizeof text, "%.10f", value);
  return text;
}

generativeqc_status focal_df_reference(generativeqc_context* context,
                                       const generativeqc_system* orbital,
                                       const generativeqc_system* auxiliary, int spins, int forces,
                                       std::size_t budget, const double* initial_density,
                                       std::size_t initial_count, double* output, std::size_t count,
                                       generativeqc_focal_reference** owner = nullptr) {
  using namespace generativeqc;
  if (!context || !orbital || !auxiliary || !output || !budget || (spins != 1 && spins != 2) ||
      (forces != 0 && forces != 1))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard lock(context->mutex);
  try {
    const auto& sys = orbital->data;
    const auto& aux = auxiliary->data;
    if (orbital->ks_auxiliary || orbital->correlation_auxiliary || orbital->scf_auxiliary ||
        auxiliary->ks_auxiliary || auxiliary->correlation_auxiliary)
      throw std::invalid_argument(
          "focal reference requires normalized orbital and SCF JKFit handles");
    if (sys.basis_representation != core::kSphericalBasis ||
        aux.basis_representation != core::kSphericalBasis)
      throw std::invalid_argument(
          "focal reference requires real-spherical orbital and JKFit representations");
    if (!aux.ecp_terms.empty())
      throw std::invalid_argument("the focal JKFit describes Coulomb fitting functions only");
    if (!sys.ecp_terms.empty())
      throw std::invalid_argument("focal reference requires an all-electron Hamiltonian");
    if (sys.atoms.size() != aux.atoms.size() || sys.charge != aux.charge ||
        sys.multiplicity != aux.multiplicity)
      throw std::invalid_argument(
          "focal reference systems differ in geometry, charge or multiplicity");
    for (std::size_t i = 0; i < sys.atoms.size(); ++i)
      if (sys.atoms[i].atomic_number != aux.atoms[i].atomic_number ||
          sys.atoms[i].position != aux.atoms[i].position || aux.atoms[i].ecp_core)
        throw std::invalid_argument("focal reference nuclei or effective charges differ");
    const auto n = molecule::ao_count(sys), nn = runtime::size_mul(n, n);
    const auto nforce = forces ? runtime::size_mul(3, sys.atoms.size()) : 0;
    const auto matrices = runtime::size_mul(2 + 3 * spins, nn);
    const auto vectors = runtime::size_mul(spins, runtime::size_add(n, 1));
    if (count !=
        runtime::size_add(runtime::size_add(matrices, vectors), runtime::size_add(3, nforce)))
      throw std::invalid_argument(
          "focal reference output shape differs from the normalized native OBS");
    if ((initial_density && initial_count != runtime::size_mul(spins, nn)) ||
        (!initial_density && initial_count))
      throw std::invalid_argument("initial DF-HF density has the wrong spin or AO dimensions");
    if (runtime::size_mul(runtime::size_add(count, initial_count), sizeof(double)) > budget) {
      record_oom_detail_noalloc(context->last_detail,
                                "focal reference export exceeds the numeric budget");
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }

    const auto backend = context->state.executed_backend;
    if (backend != context->state.requested_backend ||
        (backend != core::kCudaBackend && backend != core::kCpuReferenceBackend))
      throw std::invalid_argument("focal reference requires an explicit CPU or CUDA context");
    const bool cuda = backend == core::kCudaBackend;
    scf::ScfOptions options;
    options.stabilize_focal_rhf_convergence = cuda;
    auto capacity = spins == 1 ? posthf::rhf_reference_capacity(sys, options.diis_history, false)
                               : posthf::uhf_reference_capacity(sys, options.diis_history, false);
    if (options.stabilize_focal_rhf_convergence)
      capacity = runtime::size_add(capacity, runtime::size_mul(nn, 8 * sizeof(double)));
    if (capacity > budget) {
      record_oom_detail_noalloc(context->last_detail,
                                "DF-HF physical reference exceeds its numeric capacity budget");
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }
    if (owner && spins == 1 && posthf::uhf_reference_capacity(sys, options.diis_history) > budget) {
      record_oom_detail_noalloc(context->last_detail,
                                "closed-shell spin expansion exceeds focal capacity");
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }
    options.max_iterations = 140;
    options.energy_tolerance = 1e-13;
    options.density_tolerance = 1e-11;
    options.screening_tolerance = 0;
    options.density_fitting_relative_threshold = 1e-10;
    options.reference_memory_budget_bytes = budget;
    options.density_fitting_memory_budget_bytes = budget;
    options.precision_mode = GENERATIVEQC_PRECISION_FP64;
    options.compute_forces = forces;
    options.export_physical_reference = true;
    auto spec =
        scf::make_hf_fock_spec(spins == 1 ? scf::FockSpin::Restricted : scf::FockSpin::Unrestricted,
                               scf::FockApproximation::DensityFitted);
    spec.derivative_order = forces;
    options.resolved_fock_build = scf::resolve_fock_build(
        spec, cuda ? scf::FockBackend::Cuda : scf::FockBackend::Cpu, options.screening_tolerance,
        options.density_fitting_relative_threshold);
    // Only the orbital system defines occupations and the scalar Hamiltonian.
    // Keep the JKFit as supplied, with its own Gaussian shells on the same nuclei.
    const auto& physical = sys;
    std::vector<double> seed;
    if (initial_density) {
      seed.assign(initial_density, initial_density + initial_count);
      options.strict_initial_density = true;
    }
    const auto* start = initial_density ? &seed : nullptr;
    const auto valid_density = [&](const std::vector<double>& density) {
      return density.size() == runtime::size_mul(spins, nn) &&
             std::all_of(density.begin(), density.end(), [](double x) { return std::isfinite(x); });
    };
    std::string failure = "canonical DF-HF did not converge";
    // From its second iteration on, an undamped, unshifted DIIS run evaluates the
    // energy of a diagonalized (idempotent) determinant, so that energy bounds the
    // lowest solution of the same Hamiltonian from above.
    const auto upper_bound = [](const scf::ScfResult& st) {
      return st.iterations >= 2 && std::isfinite(st.energy)
                 ? st.energy
                 : std::numeric_limits<double>::infinity();
    };
    // Recovery restarts from the caller's seed and can converge to a different,
    // higher stationary point; that state is not the lowest solution.
    const auto reject_higher = [&](scf::ScfResult& st, double bound) {
      constexpr double kHigherSolutionTolerance = 1e-6;
      if (!st.converged || !(st.energy > bound + kHigherSolutionTolerance)) return false;
      failure = "canonical DF-HF recovery converged to a stationary point at " +
                fixed(st.energy) + " Eh, " + scientific(st.energy - bound) +
                " Eh above an earlier determinant of the same Hamiltonian; a higher solution "
                "is not returned; supply an initial density for the intended state";
      st.converged = false;
      return true;
    };
    auto recover = [&](const scf::PreparedFockPlan& plan) -> std::optional<scf::ScfResult> {
      // Multi-stage recovery on the SAME prepared Fock plan (no integral rebuild):
      // Stage 2a: virtual level-shift + density damping + DIIS to escape oscillations;
      // Stage 2b: stronger pure damped level-shifted descent if 2a still oscillates;
      // Stage 3a/3b: unshifted FP64 tight polish to full canonical stationarity.
      auto value_spec = spec;
      value_spec.derivative_order = 0;
      const auto value_build = scf::resolve_fock_build(
          value_spec, cuda ? scf::FockBackend::Cuda : scf::FockBackend::Cpu,
          options.screening_tolerance, options.density_fitting_relative_threshold);

      auto stage2 = options;
      stage2.compute_forces = false;
      stage2.export_physical_reference = false;
      stage2.strict_initial_density = false;
      stage2.resolved_fock_build = value_build;
      stage2.energy_tolerance = 1e-7;
      stage2.density_tolerance = 1e-4;
      stage2.max_iterations = 160;
      stage2.diis_history = 8;
      stage2.level_shift = 0.5;
      stage2.density_damping = 0.4;
      stage2.enable_diis = true;
      auto rec = scf::run_prepared_fock_strategy(plan, stage2, start);
      if (!rec.converged) {
        auto stage2b = stage2;
        stage2b.level_shift = 1.5;
        stage2b.density_damping = 0.25;
        stage2b.enable_diis = false;
        stage2b.max_iterations = 200;
        const bool rec_valid = rec.density.size() == runtime::size_mul(spins, nn) &&
                               std::all_of(rec.density.begin(), rec.density.end(),
                                           [](double x) { return std::isfinite(x); });
        rec = scf::run_prepared_fock_strategy(plan, stage2b, rec_valid ? &rec.density : start);
      }
      const bool have_seed = rec.density.size() == runtime::size_mul(spins, nn) &&
                             std::all_of(rec.density.begin(), rec.density.end(),
                                         [](double x) { return std::isfinite(x); });
      if (!have_seed) return std::nullopt;

      auto polish = options;
      polish.strict_initial_density = false;
      polish.level_shift = 0.0;
      polish.density_damping = 1.0;
      polish.enable_diis = true;
      polish.max_iterations = 200;
      auto st = scf::run_prepared_fock_strategy(plan, polish, &rec.density);
      if (!st.converged) {
        // Unshifted damped DIIS preserves the exact unshifted stationary fixed point.
        polish.density_damping = 0.55;
        polish.max_iterations = 240;
        const bool st_valid = st.density.size() == runtime::size_mul(spins, nn) &&
                              std::all_of(st.density.begin(), st.density.end(),
                                          [](double x) { return std::isfinite(x); });
        st = scf::run_prepared_fock_strategy(plan, polish, st_valid ? &st.density : &rec.density);
      }
      return st;
    };
    auto settle = [&](const scf::PreparedFockPlan& plan, scf::ScfResult st) {
      // First-order DIIS can stall on a soft orbital-rotation mode or stop at a
      // saddle. Exact second-order steps address the former; the search for
      // negative orbital curvature can drive a bounded downhill restart.
      // Every returned state is still accepted by the unchanged polish criteria.
      // Closed-shell determinants stay in RHF, so stability is internal only.
      namespace so = scf::solver::uhf_second_order;
      constexpr double kStabilityTolerance = 1e-5;
      constexpr unsigned kMaximumRestarts = 4;
      const bool restricted = spins == 1;
      const std::string kind = restricted ? "RHF" : "UHF";
      const auto& ints = plan.one_electron();
      const auto orthogonalizer = plan.overlap_orthogonalizer();
      const auto eigen = plan.eigen_operation(scf::PreparedFockPlan::EigenUse::Iteration);
      const std::vector<double> zero(nn, 0.0);
      const auto [alpha_occupied, beta_occupied] =
          scf::initial_guess::spin_occupations(plan.system());
      so::Problem problem;
      problem.n = n;
      problem.occupied = {alpha_occupied, beta_occupied};
      problem.hcore = &ints.hcore;
      problem.nuclear_repulsion = ints.nuclear_repulsion;
      problem.restricted = restricted;
      problem.two_electron = [&](const std::vector<double>& alpha,
                                 const std::vector<double>& beta) {
        if (restricted) {
          std::vector<double> total(nn);
          for (std::size_t k = 0; k < nn; ++k) total[k] = alpha[k] + beta[k];
          const auto g = scf::assemble_fock(plan.strategy(), zero, plan.build(total)).alpha;
          return so::SpinMatrices{g, g};
        }
        auto fock = scf::assemble_fock(plan.strategy(), zero, plan.build(alpha, beta));
        return so::SpinMatrices{std::move(fock.alpha), std::move(fock.beta)};
      };
      problem.eigen = [&](const std::vector<double>& fock) {
        return eigen ? eigen(fock, &ints.overlap, &orthogonalizer, n)
                     : scf::reference::generalized_eigen(fock, orthogonalizer, n);
      };
      const auto refine = [&](so::State state) {
        runtime::df_progress::Scope trace(restricted ? "focal_rhf_second_order"
                                                     : "focal_uhf_second_order");
        auto refined = so::refine(problem, std::move(state));
        trace.number("macro_iterations", refined.macro_iterations);
        trace.number("fock_builds", refined.fock_builds);
        trace.number("hessian_actions", refined.hessian_actions);
        trace.label("gradient_max", scientific(refined.gradient_max).c_str());
        return refined;
      };
      const auto probe = [&](const so::State& state) {
        runtime::df_progress::Scope trace(restricted ? "focal_rhf_stability"
                                                     : "focal_uhf_stability");
        auto mode = so::lowest_hessian_mode(problem, state);
        trace.number("hessian_actions", mode.hessian_actions);
        trace.label("lowest_hessian_ritz_value", scientific(mode.ritz_value).c_str());
        trace.number("ritz_converged", mode.ritz_converged);
        trace.label("hessian_search_space",
                    mode.complete_rotation_space ? "complete" : "davidson_subspace");
        return mode;
      };
      const auto polish = [&](const so::State& state) {
        auto polish_options = options;
        polish_options.strict_initial_density = false;
        polish_options.use_antisymmetric_rhf_commutator = restricted;
        polish_options.level_shift = 0.0;
        polish_options.density_damping = 1.0;
        polish_options.enable_diis = true;
        polish_options.max_iterations = 200;
        auto density = scf::reference::concatenate(state.density[0], state.density[1]);
        if (restricted) {
          for (std::size_t k = 0; k < nn; ++k) density[k] += density[nn + k];
          density.resize(nn);
        }
        return scf::run_prepared_fock_strategy(plan, polish_options, &density);
      };
      const auto check_stability = [&](scf::ScfResult& candidate) {
        double lowest = 0.0;
        unsigned restarts = 0;
        for (;; ++restarts) {
          if (restricted ? !candidate.reference : !candidate.unrestricted_reference)
            throw std::runtime_error("canonical DF-HF did not publish its physical reference");
          const auto state = so::from_coefficients(
              problem, restricted ? so::SpinMatrices{candidate.reference->coefficients,
                                                     candidate.reference->coefficients}
                                  : candidate.unrestricted_reference->coefficients);
          const auto mode = probe(state);
          lowest = mode.ritz_value;
          if (lowest >= -kStabilityTolerance) {
            if (mode.ritz_converged) return;
            failure = "canonical DF-" + kind +
                      " stability analysis did not converge (lowest orbital-Hessian Ritz value " +
                      scientific(lowest) + " Eh, residual " + scientific(mode.residual) + ")";
            candidate.converged = false;
            return;
          }
          if (restarts == kMaximumRestarts) break;
          unsigned probes = 0;
          auto lower = so::descend_along(problem, state, mode.vector, probes);
          if (!lower) break;
          auto next = polish(refine(std::move(*lower)).state);
          if (!next.converged || !(next.energy < candidate.energy)) break;
          candidate = std::move(next);
        }
        failure = "canonical DF-" + kind + " reached only an unstable stationary point (energy " +
                  fixed(candidate.energy) + " Eh, lowest real orbital-Hessian Ritz value " +
                  scientific(lowest) + " Eh after " + std::to_string(restarts) +
                  " downhill restarts); supply an initial density for the intended state";
        candidate.converged = false;
      };
      if (st.converged) {
        check_stability(st);
        return st;
      }

      double bound = upper_bound(st);
      std::string recovery_failure;
      if (!restricted) {
        // The shifted UHF restart can reach a different basin from the failed
        // primary. Give that established route priority over local refinement.
        if (auto recovered = recover(plan); recovered && recovered->converged) {
          if (!reject_higher(*recovered, bound)) check_stability(*recovered);
          if (recovered->converged) return std::move(*recovered);
          recovery_failure = failure;
        }
      }

      std::string second_order = "skipped for a nonfinite last iterate";
      if (valid_density(st.density)) {
        auto [alpha, beta] = restricted ? std::pair{st.density, st.density}
                                        : scf::reference::split_spin_matrices(st.density, nn);
        if (restricted)
          for (std::size_t k = 0; k < nn; ++k) alpha[k] = beta[k] = 0.5 * st.density[k];
        auto refined = refine(so::from_density(problem, alpha, beta));
        // Newton steps converge to a saddle as readily as to a minimum, and
        // the polish may hold either; leave a saddle downhill first.
        unsigned descents = 0;
        for (; descents < kMaximumRestarts; ++descents) {
          const auto mode = probe(refined.state);
          if (!(mode.ritz_value < -kStabilityTolerance)) break;
          unsigned probes = 0;
          auto lower = so::descend_along(problem, refined.state, mode.vector, probes);
          if (!lower) break;
          refined = refine(std::move(*lower));
        }
        if (std::isfinite(refined.state.energy)) bound = std::min(bound, refined.state.energy);
        second_order = std::to_string(refined.macro_iterations) +
                       " steps to maximum MO gradient " + scientific(refined.gradient_max) +
                       " after " + std::to_string(descents) + " downhill restarts";
        st = polish(refined.state);
      }
      if (!st.converged && restricted) {
        if (auto recovered = recover(plan)) st = std::move(*recovered);
      }
      if (reject_higher(st, bound)) return st;
      if (st.converged) {
        check_stability(st);
      } else if (!recovery_failure.empty()) {
        failure = std::move(recovery_failure);
      } else {
        failure = "canonical DF-HF did not converge: " + kind +
                  " DIIS, second-order refinement (" + second_order +
                  ") and first-order recovery all stopped short of the " +
                  "stationarity criteria (last energy change " + scientific(st.energy_change) +
                  " Eh, density RMS " + scientific(st.density_rms) +
                  "); supply an initial density for the intended state";
      }
      return st;
    };
    auto run_ladder = [&](const scf::PreparedFockPlan& plan) {
      auto st = scf::run_prepared_fock_strategy(plan, options, start);
      // Recover failed closed-shell primaries in the restricted space.
      if (spins != 1 || st.converged) return st;
      return settle(plan, std::move(st));
    };
    const auto strategy = scf::fock_strategy_for_execution(options);
    const scf::PreparedFockPlan plan(physical, &aux, strategy,
                                     cuda ? context->state.device_id : -1,
                                     options.density_fitting_memory_budget_bytes);
    const auto state = run_ladder(plan);
    if (start && !state.initial_density_used)
      throw std::runtime_error("DF-HF did not use the requested initial density");
    if (!state.converged) {
      context->last_detail = failure;
      return GENERATIVEQC_STATUS_SCF_NOT_CONVERGED;
    }
    if ((spins == 1 && !state.reference) || (spins == 2 && !state.unrestricted_reference))
      throw std::runtime_error("canonical DF-HF did not publish its physical reference");
    const auto view = spins == 1 ? state.reference->electronic_reference()
                                 : state.unrestricted_reference->electronic_reference();
    core::validate_electronic_reference_shape(view);
    if (n != view.basis_functions || std::size_t(spins) != view.spin_channels ||
        (forces && nforce != state.forces.size()) || !std::isfinite(view.energy))
      throw std::runtime_error("inconsistent canonical DF-HF result dimensions or energy");

    // All native views are borrowed from this solve. Copy the checked dimensions
    // into caller-owned storage while the same reference owns the spin frames.
    *output++ = view.energy;
    *output++ = state.iterations;
    *output++ = backend;
    for (int spin = 0; spin < spins; ++spin) *output++ = view.channels[spin].occupied;
    auto copy = [&](std::span<const double> src) {
      output = std::copy(src.begin(), src.end(), output);
    };
    copy(view.overlap);
    copy(view.hcore);
    for (int spin = 0; spin < spins; ++spin) {
      const auto& channel = view.channels[spin];
      copy(channel.coefficients);
      copy(channel.density);
      copy(channel.fock);
      copy(channel.orbital_energies);
    }
    if (forces) copy(state.forces);
    if (owner) {
      auto ref = state.unrestricted_reference;
      if (spins == 1) {
        const auto& rhf = *state.reference;
        auto expanded = std::make_shared<hf::UnrestrictedPhysicalReference>();
        expanded->nbf = n;
        expanded->nocc.fill(rhf.nocc);
        expanded->overlap = rhf.overlap;
        expanded->hcore = rhf.hcore;
        expanded->coefficients.fill(rhf.coefficients);
        expanded->fock.fill(rhf.fock);
        expanded->orbital_energies.fill(rhf.orbital_energies);
        auto density = rhf.density;
        for (auto& x : density) x *= 0.5;
        expanded->density.fill(density);
        expanded->energy = rhf.energy;
        expanded->commutator_residual = rhf.commutator_residual;
        expanded->canonical_density_drift = rhf.canonical_density_drift;
        expanded->canonical_error = rhf.eigen_residual;
        expanded->source_identity.emplace(physical);
        expanded->source_charge = physical.charge;
        expanded->source_electrons = physical.electron_count;
        expanded->source_multiplicity = physical.multiplicity;
        ref = std::move(expanded);
      }
      const bool spatial = spins == 1 && cuda;
      *owner = new generativeqc_focal_reference{
          std::move(ref), molecule::ScalarHamiltonianIdentity(aux),
          static_cast<generativeqc_backend>(backend), context->state.device_id,
          spatial ? state.reference : nullptr};
    }
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::length_error& error) {
    record_oom_detail_noalloc(context->last_detail, error.what());
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::bad_alloc& error) {
    record_oom_detail_noalloc(context->last_detail, error.what());
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return api::map_exception(&context->last_detail);
  }
}

}  // namespace

extern "C" GENERATIVEQC_API generativeqc_status generativeqc_focal_df_reference_v1(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* auxiliary, int spins, int forces, std::size_t budget,
    double* output, std::size_t count) {
  return focal_df_reference(context, orbital, auxiliary, spins, forces, budget, nullptr, 0,
                            output, count);
}

extern "C" GENERATIVEQC_API generativeqc_status generativeqc_focal_df_reference_seeded_v1(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* auxiliary, int spins, int forces, std::size_t budget,
    const double* initial_density, std::size_t initial_count, double* output, std::size_t count) {
  if (!initial_density || !initial_count) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  return focal_df_reference(context, orbital, auxiliary, spins, forces, budget,
                            initial_density, initial_count, output, count);
}

extern "C" generativeqc_status generativeqc_focal_df_reference_owned_v1(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* auxiliary, int spins, int forces, std::size_t budget,
    const double* initial_density, std::size_t initial_count, double* output, std::size_t count,
    generativeqc_focal_reference** owner) {
  if (!owner) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  *owner = nullptr;
  return focal_df_reference(context, orbital, auxiliary, spins, forces, budget, initial_density,
                            initial_count, output, count, owner);
}

extern "C" void generativeqc_focal_reference_destroy_v1(generativeqc_focal_reference* owner) {
  delete owner;
}
