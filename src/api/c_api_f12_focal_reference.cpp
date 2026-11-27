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
#include "cc/df_ucc_v2.hpp"
#include "cc/f12_fixed_cusp.hpp"
#include "cc/lambda_response.hpp"
#include "cc/solver.hpp"
#include "generated_rccsd_cpu.hpp"
#include "generativeqc/ucc_fitting.h"
#include "methods/df_rccsdt_method.hpp"
#include "methods/uccsd_method.hpp"
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
    options.enforce_orthonormal_uhf_commutator = true;
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
      stage2.enforce_orthonormal_uhf_commutator = false;
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

extern "C" generativeqc_status generativeqc_focal_ucc_triples_v1(
    generativeqc_calculation* calculation, const generativeqc_focal_reference* owner, int forces,
    double* output, std::size_t count) {
  if (!calculation || !calculation->context || !owner || !output || (forces != 0 && forces != 1))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  auto* context = calculation->context;
  std::lock_guard lock(context->mutex);
  try {
    using namespace generativeqc;
    runtime::df_progress::Scope trace(
        "focal_ucc_triples", context->state.executed_backend == core::kCudaBackend ? "cuda" : "host");
    if (owner->backend != context->state.executed_backend ||
        owner->device != context->state.device_id ||
        count != 1 + (forces ? 3 * calculation->plan->atom_count() : 0))
      throw std::invalid_argument("inconsistent focal triples output or backend");
    const auto result = generativeqc::methods::detail::focal_ucc_triples(
        *calculation->plan, owner->uhf, owner->jk, false, forces);
    if (!std::isfinite(result.energy) || result.forces.size() + 1 != count ||
        !std::all_of(result.forces.begin(), result.forces.end(),
                     [](double x) { return std::isfinite(x); }))
      throw std::runtime_error("invalid focal triples result");
    output[0] = result.energy;
    std::copy(result.forces.begin(), result.forces.end(), output + 1);
    trace.finish();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::length_error& error) {
    record_oom_detail_noalloc(context->last_detail, error.what());
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::bad_alloc& error) {
    record_oom_detail_noalloc(context->last_detail, error.what());
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

namespace {
generativeqc_status focal_rcc_triples_values(
    generativeqc_calculation* calculation, const generativeqc_focal_reference* owner,
    double* output, std::size_t count, bool occupied_partition) {
  if (!calculation || !calculation->context || !owner || !output)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  auto* context = calculation->context;
  std::lock_guard lock(context->mutex);
  try {
    using namespace generativeqc;
    runtime::df_progress::Scope trace(
        occupied_partition ? "focal_rcc_triples_occupied" : "focal_rcc_triples_energy", "cuda");
    if (!owner->uhf || !owner->uhf->source_identity ||
        owner->backend != core::kCudaBackend || owner->backend != context->state.executed_backend ||
        owner->device != context->state.device_id)
      throw std::invalid_argument("focal spatial triples requires a retained CUDA RHF owner");
    // The spatial source accounts for the RHF frame itself. Its existing
    // spin-expanded focal view and identity records also remain live.
    const auto& expanded = *owner->uhf;
    const auto occupied_count = occupied_partition ? expanded.nocc[0] : 0;
    if (count != runtime::size_add(1, occupied_count))
      throw std::invalid_argument("focal spatial triples output has the wrong occupied dimension");
    auto retained = runtime::size_add(owner->jk.storage_bytes(),
                                     expanded.source_identity->storage_bytes());
    auto charge = [&](const std::vector<double>& values) {
      retained = runtime::size_add(retained, runtime::size_mul(values.capacity(), sizeof(double)));
    };
    charge(expanded.overlap);
    charge(expanded.hcore);
    for (std::size_t spin = 0; spin < 2; ++spin) {
      charge(expanded.coefficients[spin]);
      charge(expanded.density[spin]);
      charge(expanded.fock[spin]);
      charge(expanded.orbital_energies[spin]);
    }
    const auto result = methods::detail::focal_rcc_triples_values(
        *calculation->plan, owner->rhf, *expanded.source_identity, owner->jk, retained,
        occupied_partition);
    if (!std::isfinite(result.energy) || result.occupied.size() != occupied_count)
      throw std::runtime_error("invalid focal spatial triples result");
    if (occupied_partition) {
      long double total = 0, magnitude = 0;
      for (const auto value : result.occupied) {
        if (!std::isfinite(value))
          throw std::runtime_error("nonfinite focal spatial occupied triples contribution");
        total += value;
        magnitude += std::abs(static_cast<long double>(value));
      }
      if (!std::isfinite(total) || !std::isfinite(magnitude) ||
          std::abs(total - result.energy) > 3e-12L * (1 + magnitude))
        throw std::runtime_error("focal spatial occupied triples do not close to their scalar");
      std::copy(result.occupied.begin(), result.occupied.end(), output + 1);
    }
    *output = result.energy;
    trace.finish();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::length_error& error) {
    record_oom_detail_noalloc(context->last_detail, error.what());
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::bad_alloc& error) {
    record_oom_detail_noalloc(context->last_detail, error.what());
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}
}  // namespace

extern "C" GENERATIVEQC_API generativeqc_status generativeqc_focal_rcc_triples_energy_v1(
    generativeqc_calculation* calculation, const generativeqc_focal_reference* owner,
    double* output) {
  return focal_rcc_triples_values(calculation, owner, output, 1, false);
}

extern "C" GENERATIVEQC_API generativeqc_status generativeqc_focal_rcc_triples_occupied_v1(
    generativeqc_calculation* calculation, const generativeqc_focal_reference* owner,
    double* output, std::size_t count) {
  return focal_rcc_triples_values(calculation, owner, output, count, true);
}

extern "C" GENERATIVEQC_API generativeqc_status generativeqc_focal_rcc_triples_v2(
    generativeqc_calculation* calculation, const generativeqc_focal_reference* owner, int forces,
    int occupied_partition, std::size_t frozen_occupied, const double* occupied_weights,
    std::size_t weights_count, std::uint32_t preconvergence, int triples_fp32_w, double* energy_out,
    double* occupied_out, std::size_t occupied_count, double* corr_gradient_out,
    std::size_t corr_gradient_count, double* bar_f_out, std::size_t bar_f_count, double* bar_c_out,
    std::size_t bar_c_count, double* telemetry_out, std::size_t telemetry_count) {
  if (!calculation || !calculation->context || !owner || !energy_out ||
      (forces != 0 && forces != 1) || (occupied_partition != 0 && occupied_partition != 1) ||
      (triples_fp32_w < 0 || triples_fp32_w > 2) || preconvergence > 3)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  auto* context = calculation->context;
  std::lock_guard lock(context->mutex);
  context->lambda_telemetry.reset();
  try {
    using namespace generativeqc;
    runtime::df_progress::Scope trace(
        forces ? "focal_rcc_triples_force_sources_v2"
               : (occupied_partition ? "focal_rcc_triples_occupied_v2"
                                     : "focal_rcc_triples_energy_v2"),
        "cuda");
    if (!owner->uhf || !owner->uhf->source_identity || !owner->rhf ||
        owner->backend != core::kCudaBackend || owner->backend != context->state.executed_backend ||
        owner->device != context->state.device_id)
      throw std::invalid_argument("focal spatial triples v2 requires a retained CUDA RHF owner");
    const auto& expanded = *owner->uhf;
    const std::size_t o_full = expanded.nocc[0];
    if (frozen_occupied >= o_full)
      throw std::invalid_argument("focal spatial triples v2 frozen_occupied exceeds occupied count");
    const std::size_t o_active = o_full - frozen_occupied;
    const std::size_t n = expanded.nbf;
    const std::size_t nn = runtime::size_mul(n, n);
    const std::size_t coords = runtime::size_mul(3, calculation->plan->atom_count());
    if (occupied_partition && (!occupied_out || occupied_count != o_active))
      throw std::invalid_argument("focal spatial triples v2 occupied buffer shape mismatch");
    if (weights_count && (!occupied_weights || weights_count != o_active))
      throw std::invalid_argument("focal spatial triples v2 weights buffer shape mismatch");
    if (forces && (!corr_gradient_out || corr_gradient_count != coords || !bar_f_out ||
                   bar_f_count != nn || !bar_c_out || bar_c_count != nn))
      throw std::invalid_argument("focal spatial triples v2 force source buffer shape mismatch");
    if (telemetry_out && telemetry_count < 8)
      throw std::invalid_argument("focal spatial triples v2 telemetry buffer requires >= 8 slots");

    auto retained = runtime::size_add(owner->jk.storage_bytes(),
                                     expanded.source_identity->storage_bytes());
    auto charge = [&](const std::vector<double>& values) {
      retained = runtime::size_add(retained, runtime::size_mul(values.capacity(), sizeof(double)));
    };
    charge(expanded.overlap);
    charge(expanded.hcore);
    for (std::size_t spin = 0; spin < 2; ++spin) {
      charge(expanded.coefficients[spin]);
      charge(expanded.density[spin]);
      charge(expanded.fock[spin]);
      charge(expanded.orbital_energies[spin]);
    }

    methods::detail::FocalRccTriplesOptions options;
    options.forces = forces != 0;
    options.occupied_partition = occupied_partition != 0;
    options.frozen_occupied = frozen_occupied;
    if (weights_count)
      options.occupied_weights.assign(occupied_weights, occupied_weights + weights_count);
    options.preconvergence = static_cast<core::DfPreconvergence>(preconvergence);
    options.triples_fp32_w = triples_fp32_w == 1;
    options.triples_compensated_w = triples_fp32_w == 2;

    cc::detail::reset_last_lambda_diagnostic();
    const auto result = methods::detail::focal_rcc_triples_values(
        *calculation->plan, owner->rhf, *expanded.source_identity, owner->jk, retained, options);
    if (!std::isfinite(result.energy))
      throw std::runtime_error("nonfinite focal spatial triples v2 energy");
    *energy_out = result.energy;
    if (occupied_partition) {
      if (result.occupied.size() != o_active)
        throw std::runtime_error("focal spatial triples v2 occupied partition size mismatch");
      std::copy(result.occupied.begin(), result.occupied.end(), occupied_out);
    }
    if (forces) {
      if (result.correlation_gradient.size() != coords || result.bar_f.size() != nn ||
          result.bar_c.size() != nn)
        throw std::runtime_error("focal spatial triples v2 force source size mismatch");
      std::copy(result.correlation_gradient.begin(), result.correlation_gradient.end(),
                corr_gradient_out);
      std::copy(result.bar_f.begin(), result.bar_f.end(), bar_f_out);
      std::copy(result.bar_c.begin(), result.bar_c.end(), bar_c_out);
    }
    if (telemetry_out) {
      telemetry_out[0] = result.ccsd_fp32_executed ? 1.0 : 0.0;
      telemetry_out[1] = result.lambda_fp32_executed ? 1.0 : 0.0;
      telemetry_out[2] = static_cast<double>(result.triples_fp32_gemms);
      telemetry_out[3] = static_cast<double>(result.triples_fp64_gemms);
      telemetry_out[4] = result.ccsd_seconds;
      telemetry_out[5] = result.triples_seconds;
      telemetry_out[6] = result.lambda_seconds;
      telemetry_out[7] = result.source_response_seconds;
      if (telemetry_count >= 9)
        telemetry_out[8] = result.triples_fp32_w_executed ? 1.0 : 0.0;
      if (telemetry_count >= 10)
        telemetry_out[9] = result.triples_fp32_w_resource_fallback ? 1.0 : 0.0;
      if (telemetry_count >= 11)
        telemetry_out[10] = result.triples_fp32_w_forward_executed ? 1.0 : 0.0;
      if (telemetry_count >= 12)
        telemetry_out[11] = result.triples_fp32_w_reverse_executed ? 1.0 : 0.0;
      if (telemetry_count >= 13)
        telemetry_out[12] = result.triples_reverse_w_executed ? 1.0 : 0.0;
      if (telemetry_count >= 14)
        telemetry_out[13] = result.triples_compensated_w_executed ? 1.0 : 0.0;
      if (telemetry_count >= 15)
        telemetry_out[14] = static_cast<double>(result.triples_fp32_forward_gemms);
      if (telemetry_count >= 16)
        telemetry_out[15] = static_cast<double>(result.triples_fp32_reverse_gemms);
      if (telemetry_count >= 17)
        telemetry_out[16] = static_cast<double>(result.triples_workspace_bytes);
    }
    if (forces) cc::detail::capture_last_lambda_telemetry(context->lambda_telemetry);
    trace.finish();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::length_error& error) {
    record_oom_detail_noalloc(context->last_detail, error.what());
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::bad_alloc& error) {
    record_oom_detail_noalloc(context->last_detail, error.what());
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

extern "C" GENERATIVEQC_API int generativeqc_focal_rcc_f12b_cuda_stage_counters_v1(void) {
#if GENERATIVEQC_HAS_CUDA
  return 1;
#else
  return 0;
#endif
}

extern "C" GENERATIVEQC_API generativeqc_status generativeqc_focal_rcc_f12b_v1(
    generativeqc_calculation* calculation, const generativeqc_focal_reference* owner, int forces,
    std::size_t frozen_occupied, const double* w_vo, std::size_t w_vo_count, const double* w_vv,
    std::size_t w_vv_count, const double* c_f12, std::size_t c_f12_count,
    double fixed_geminal_energy, std::uint32_t preconvergence, double* energy_out, double* t1_out,
    std::size_t t1_count, double* t2_out, std::size_t t2_count, double* lambda1_out,
    std::size_t lambda1_count, double* lambda2_out, std::size_t lambda2_count,
    double* corr_gradient_out, std::size_t corr_gradient_count, double* bar_f_out,
    std::size_t bar_f_count, double* bar_c_out, std::size_t bar_c_count, double* telemetry_out,
    std::size_t telemetry_count) {
  if (!calculation || !calculation->context || !owner || !energy_out ||
      (forces != 0 && forces != 1) || preconvergence > 3 || !w_vo || !w_vv || !c_f12 ||
      !std::isfinite(fixed_geminal_energy))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  auto* context = calculation->context;
  std::lock_guard lock(context->mutex);
  context->lambda_telemetry.reset();
  try {
    using namespace generativeqc;
    runtime::df_progress::Scope trace(
        forces ? "focal_rcc_f12b_force_sources_v1" : "focal_rcc_f12b_energy_v1", "cuda");
    if (!owner->uhf || !owner->uhf->source_identity || !owner->rhf ||
        owner->backend != core::kCudaBackend || owner->backend != context->state.executed_backend ||
        owner->device != context->state.device_id)
      throw std::invalid_argument("focal spatial CCSD-F12b requires a retained CUDA RHF owner");
    const auto& expanded = *owner->uhf;
    const std::size_t o_full = expanded.nocc[0];
    if (frozen_occupied >= o_full)
      throw std::invalid_argument(
          "focal spatial CCSD-F12b frozen_occupied exceeds occupied count");
    const std::size_t o_act = o_full - frozen_occupied;
    const std::size_t n = expanded.nbf;
    if (n <= o_full)
      throw std::invalid_argument("focal spatial CCSD-F12b requires nonempty virtual space");
    const std::size_t v = n - o_full;
    const std::size_t oo = runtime::size_mul(o_act, o_act);
    const std::size_t n1 = runtime::size_mul(o_act, v);
    const std::size_t n2 = runtime::size_mul(oo, runtime::size_mul(v, v));
    const std::size_t n_vo = runtime::size_mul(oo, runtime::size_mul(v, o_act));
    const std::size_t nn = runtime::size_mul(n, n);
    const std::size_t coords = runtime::size_mul(3, calculation->plan->atom_count());
    if (w_vo_count != n_vo || w_vv_count != n2 || c_f12_count != n2)
      throw std::invalid_argument("focal spatial CCSD-F12b fixed F12 buffer shape mismatch");
    if ((t1_out && t1_count != n1) || (!t1_out && t1_count) || (t2_out && t2_count != n2) ||
        (!t2_out && t2_count))
      throw std::invalid_argument("focal spatial CCSD-F12b amplitude output buffer shape mismatch");
    if ((lambda1_out && lambda1_count != n1) || (!lambda1_out && lambda1_count) ||
        (lambda2_out && lambda2_count != n2) || (!lambda2_out && lambda2_count))
      throw std::invalid_argument("focal spatial CCSD-F12b Lambda output buffer shape mismatch");
    if (forces && (!t1_out || !t2_out || !lambda1_out || !lambda2_out || !corr_gradient_out ||
                   corr_gradient_count != coords || !bar_f_out || bar_f_count != nn || !bar_c_out ||
                   bar_c_count != nn))
      throw std::invalid_argument("focal spatial CCSD-F12b force output buffer shape mismatch");
    if (telemetry_out && telemetry_count < 8)
      throw std::invalid_argument("focal spatial CCSD-F12b telemetry buffer requires >= 8 slots");

    auto retained = runtime::size_add(owner->jk.storage_bytes(),
                                     expanded.source_identity->storage_bytes());
    auto charge = [&](const std::vector<double>& values) {
      retained = runtime::size_add(retained, runtime::size_mul(values.capacity(), sizeof(double)));
    };
    charge(expanded.overlap);
    charge(expanded.hcore);
    for (std::size_t spin = 0; spin < 2; ++spin) {
      charge(expanded.coefficients[spin]);
      charge(expanded.density[spin]);
      charge(expanded.fock[spin]);
      charge(expanded.orbital_energies[spin]);
    }

    methods::detail::FocalRccF12bOptions options;
    options.forces = forces != 0;
    options.frozen_occupied = frozen_occupied;
    options.preconvergence = static_cast<core::DfPreconvergence>(preconvergence);
    options.f12.w_vo.assign(w_vo, w_vo + w_vo_count);
    options.f12.w_vv.assign(w_vv, w_vv + w_vv_count);
    options.f12.c_f12.assign(c_f12, c_f12 + c_f12_count);
    options.f12.fixed_geminal_energy = fixed_geminal_energy;

    cc::detail::reset_last_lambda_diagnostic();
    const auto result = methods::detail::focal_rcc_f12b_values(
        *calculation->plan, owner->rhf, *expanded.source_identity, owner->jk, retained,
        std::move(options));
    if (!std::isfinite(result.energy))
      throw std::runtime_error("nonfinite focal spatial CCSD-F12b energy");
    *energy_out = result.energy;
    if (t1_out) {
      if (result.t1.size() != n1)
        throw std::runtime_error("focal spatial CCSD-F12b t1 output size mismatch");
      std::copy(result.t1.begin(), result.t1.end(), t1_out);
    }
    if (t2_out) {
      if (result.t2.size() != n2)
        throw std::runtime_error("focal spatial CCSD-F12b t2 output size mismatch");
      std::copy(result.t2.begin(), result.t2.end(), t2_out);
    }
    if (forces) {
      if (result.lambda1.size() != n1 || result.lambda2.size() != n2 ||
          result.correlation_gradient.size() != coords || result.bar_f.size() != nn ||
          result.bar_c.size() != nn)
        throw std::runtime_error("focal spatial CCSD-F12b force output size mismatch");
      std::copy(result.lambda1.begin(), result.lambda1.end(), lambda1_out);
      std::copy(result.lambda2.begin(), result.lambda2.end(), lambda2_out);
      std::copy(result.correlation_gradient.begin(), result.correlation_gradient.end(),
                corr_gradient_out);
      std::copy(result.bar_f.begin(), result.bar_f.end(), bar_f_out);
      std::copy(result.bar_c.begin(), result.bar_c.end(), bar_c_out);
    }
    if (telemetry_out) {
      telemetry_out[0] = result.ccsd_fp32_executed ? 1.0 : 0.0;
      telemetry_out[1] = result.lambda_fp32_executed ? 1.0 : 0.0;
      telemetry_out[2] = static_cast<double>(result.ccsd_iterations);
      telemetry_out[3] = static_cast<double>(result.lambda_iterations);
      telemetry_out[4] = result.ccsd_seconds;
      telemetry_out[5] = result.lambda_seconds;
      telemetry_out[6] = result.source_response_seconds;
      telemetry_out[7] = result.total_seconds;
      if (telemetry_count >= 9) telemetry_out[8] = static_cast<double>(result.f12_cuda_preparations);
      if (telemetry_count >= 10) telemetry_out[9] = static_cast<double>(result.f12_cuda_rhs_evaluations);
    }
    if (forces) cc::detail::capture_last_lambda_telemetry(context->lambda_telemetry);
    trace.finish();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::length_error& error) {
    record_oom_detail_noalloc(context->last_detail, error.what());
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::bad_alloc& error) {
    record_oom_detail_noalloc(context->last_detail, error.what());
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

extern "C" GENERATIVEQC_API generativeqc_status generativeqc_rcc_f12b_cpu_solve_v1(
    std::size_t nocc, std::size_t nvir, const double* foo, const double* fov, const double* fvv,
    const double* ovov, const double* ovvo, const double* oovv, const double* ovvv,
    const double* ovoo, const double* oooo, const double* vvvv, const double* eps_o,
    const double* eps_v, const double* w_vo, const double* w_vv, const double* c_f12,
    double fixed_geminal_energy, int solve_lambda, std::size_t max_bytes, double* energy_out,
    double* t1_out, double* t2_out, double* lambda1_out, double* lambda2_out, double* bar_foo_out,
    double* bar_fov_out, double* bar_fvv_out) {
  if (!nocc || !nvir || !foo || !fov || !fvv || !ovov || !ovvo || !oovv || !ovvv || !ovoo ||
      !oooo || !vvvv || !eps_o || !eps_v || !w_vo || !w_vv || !c_f12 ||
      !std::isfinite(fixed_geminal_energy) || (solve_lambda != 0 && solve_lambda != 1) ||
      !max_bytes || !energy_out || !t1_out || !t2_out)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (solve_lambda && (!lambda1_out || !lambda2_out)) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
    using namespace generativeqc;
    const std::size_t o = nocc, v = nvir;
    const std::size_t oo = runtime::size_mul(o, o), vv = runtime::size_mul(v, v);
    const std::size_t n1 = runtime::size_mul(o, v), n2 = runtime::size_mul(oo, vv);
    const std::size_t n_vo = runtime::size_mul(oo, runtime::size_mul(v, o));
    const std::size_t n_ovvv = runtime::size_mul(n1, vv);
    const std::size_t n_ovoo = runtime::size_mul(n1, oo);
    const std::size_t n_oooo = runtime::size_mul(oo, oo);
    const std::size_t n_vvvv = runtime::size_mul(vv, vv);

    // Enforce the caller's byte budget BEFORE copying/allocating vvvv or any
    // other dense Problem/workspace buffer.
    std::size_t upfront_elems = 0;
    for (const std::size_t count :
         {oo, n1, vv, n2, n2, n2, n_ovvv, n_ovoo, n_oooo, n_vvvv, n_vo, n2, n2, n1, n2, n1, n2,
          cc::generated::iteration_arena_elements(o, v),
          cc::generated::replay_arena_elements(o, v)}) {
      upfront_elems = runtime::size_add(upfront_elems, count);
    }
    if (runtime::size_mul(upfront_elems, sizeof(double)) > max_bytes)
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;

    cc::Problem p;
    p.nocc = o;
    p.nvir = v;
    p.foo.assign(foo, foo + oo);
    p.fov.assign(fov, fov + n1);
    p.fvv.assign(fvv, fvv + vv);
    p.ovov.assign(ovov, ovov + n2);
    p.ovvo.assign(ovvo, ovvo + n2);
    p.oovv.assign(oovv, oovv + n2);
    p.ovvv.assign(ovvv, ovvv + n_ovvv);
    p.ovoo.assign(ovoo, ovoo + n_ovoo);
    p.oooo.assign(oooo, oooo + n_oooo);
    p.vvvv.assign(vvvv, vvvv + n_vvvv);
    p.f12.w_vo.assign(w_vo, w_vo + n_vo);
    p.f12.w_vv.assign(w_vv, w_vv + n2);
    p.f12.c_f12.assign(c_f12, c_f12 + n2);
    p.f12.fixed_geminal_energy = fixed_geminal_energy;
    cc::f12::validate(o, v, p.f12);
    p.f12.w_vv = cc::f12::symmetrized_pair_copy(o, v, p.f12.w_vv);
    p.f12.c_f12 = cc::f12::symmetrized_pair_copy(o, v, p.f12.c_f12);

    cc::SolverOptions opts;
    opts.max_iterations = 150;
    opts.energy_tolerance = 1e-13;
    opts.residual_tolerance = 1e-12;
    opts.max_bytes = max_bytes;
    std::vector<double> eps;
    eps.reserve(o + v);
    eps.insert(eps.end(), eps_o, eps_o + o);
    eps.insert(eps.end(), eps_v, eps_v + v);
    cc::initialize_canonical_denominators(p, eps, opts, false);

    p.initial_t1.assign(n1, 0.0);
    p.initial_t2.resize(n2);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b) {
            const auto t = ((i * o + j) * v + a) * v + b;
            const auto mate = ((j * o + i) * v + b) * v + a;
            if (t > mate) continue;
            const auto g = ((i * v + a) * o + j) * v + b;
            const auto g_mate = ((j * v + b) * o + i) * v + a;
            const double val =
                (0.5 * (p.ovov[g] + p.ovov[g_mate]) + p.f12.w_vv[t] + p.f12.c_f12[t]) / p.d2[t];
            p.initial_t2[t] = p.initial_t2[mate] = val;
          }

    const auto solved = cc::solve_cpu(p, opts);
    if (!solved.converged()) return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    *energy_out = solved.correlation_energy;
    std::copy(solved.t1.begin(), solved.t1.end(), t1_out);
    std::copy(solved.t2.begin(), solved.t2.end(), t2_out);

    if (solve_lambda) {
      cc::LambdaOptions lopts;
      lopts.cc_tolerance = 1e-10;
      lopts.lambda_tolerance = 1e-10;
      lopts.max_bytes = max_bytes;
      lopts.gmres.absolute_tolerance = 1e-13;
      lopts.gmres.max_workspace_bytes = max_bytes;
      const auto lambda = cc::solve_lambda_cpu(p, solved, lopts);
      if (!lambda.converged()) return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
      std::copy(lambda.lambda1.begin(), lambda.lambda1.end(), lambda1_out);
      std::copy(lambda.lambda2.begin(), lambda.lambda2.end(), lambda2_out);
      if (bar_foo_out || bar_fov_out || bar_fvv_out) {
        const auto arena_elems = std::max({cc::generated::parameter_foo_arena_elements(o, v),
                                           cc::generated::parameter_fov_arena_elements(o, v),
                                           cc::generated::parameter_fvv_arena_elements(o, v)});
        const auto live_bytes = runtime::size_add(
            cc::problem_host_bytes(p),
            runtime::size_mul(runtime::size_add(runtime::size_mul(2, runtime::size_add(n1, n2)),
                                                arena_elems),
                              sizeof(double)));
        if (live_bytes > max_bytes) return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
        std::vector<double> arena(arena_elems);
        const cc::generated::Inputs in{
            p.foo.data(),  p.fov.data(),  p.fvv.data(),  p.ovov.data(),   p.ovvo.data(),
            p.oovv.data(), p.ovvv.data(), p.ovoo.data(), p.oooo.data(),   p.vvvv.data(),
            p.d1.data(),   p.d2.data(),   solved.t1.data(), solved.t2.data()};
        const double energy_seed = 1.0;
        if (bar_foo_out) {
          const auto out = cc::generated::run_parameter_foo_cpu(
              o, v, in, &energy_seed, lambda.lambda1.data(), lambda.lambda2.data(), arena.data(),
              arena.size());
          std::copy_n(out.values, oo, bar_foo_out);
        }
        if (bar_fov_out) {
          const auto out = cc::generated::run_parameter_fov_cpu(
              o, v, in, &energy_seed, lambda.lambda1.data(), lambda.lambda2.data(), arena.data(),
              arena.size());
          std::copy_n(out.values, n1, bar_fov_out);
        }
        if (bar_fvv_out) {
          const auto out = cc::generated::run_parameter_fvv_cpu(
              o, v, in, &energy_seed, lambda.lambda1.data(), lambda.lambda2.data(), arena.data(),
              arena.size());
          std::copy_n(out.values, vv, bar_fvv_out);
        }
      }
    }
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::length_error&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::bad_alloc&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return generativeqc::api::map_exception(nullptr);
  }
}

extern "C" GENERATIVEQC_API generativeqc_status generativeqc_focal_ucc_triples_v2(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* correlation_fit, const generativeqc_focal_reference* owner,
    std::size_t budget_bytes, int forces, int occupied_partition, std::size_t frozen_alpha,
    std::size_t frozen_beta, const double* occupied_weights_alpha,
    std::size_t weights_alpha_count, const double* occupied_weights_beta,
    std::size_t weights_beta_count, std::uint32_t preconvergence, int triples_fp32_w,
    double* ccsd_energy_out, double* energy_out, double* occupied_out_alpha,
    std::size_t occupied_alpha_count, double* occupied_out_beta, std::size_t occupied_beta_count,
    double* corr_gradient_out, std::size_t corr_gradient_count, double* bar_f_alpha_out,
    std::size_t bar_f_alpha_count, double* bar_f_beta_out, std::size_t bar_f_beta_count,
    double* bar_c_alpha_out, std::size_t bar_c_alpha_count, double* bar_c_beta_out,
    std::size_t bar_c_beta_count, double* telemetry_out, std::size_t telemetry_count) {
  if (!context || !orbital || !correlation_fit || !owner || !energy_out ||
      (forces != 0 && forces != 1) || (occupied_partition != 0 && occupied_partition != 1) ||
      (triples_fp32_w < 0 || triples_fp32_w > 2) || preconvergence > 3)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard lock(context->mutex);
  try {
    using namespace generativeqc;
    const auto backend = context->state.executed_backend;
    if (backend != context->state.requested_backend ||
        (backend != core::kCudaBackend && backend != core::kCpuReferenceBackend))
      throw std::invalid_argument("focal UCC triples v2 requires an explicit CPU or CUDA context");
    const bool cuda = backend == core::kCudaBackend;
    runtime::df_progress::Scope trace(
        forces ? "focal_ucc_triples_force_sources_v2"
               : (occupied_partition ? "focal_ucc_triples_occupied_v2"
                                     : "focal_ucc_triples_energy_v2"),
        cuda ? "cuda" : "host");
    if (!owner->uhf || !owner->uhf->source_identity ||
        owner->backend != static_cast<generativeqc_backend>(backend) ||
        owner->device != context->state.device_id)
      throw std::invalid_argument(
          "focal UCC triples v2 requires a matching retained DF reference owner");
    if (orbital->ks_auxiliary || orbital->correlation_auxiliary || orbital->scf_auxiliary ||
        correlation_fit->ks_auxiliary || correlation_fit->scf_auxiliary)
      throw std::invalid_argument(
          "focal UCC triples v2 requires normalized orbital and correlation-fit system handles");
    const auto& sys = orbital->data;
    const auto& aux = correlation_fit->data;
    if (sys.basis_representation != core::kSphericalBasis ||
        aux.basis_representation != core::kSphericalBasis)
      throw std::invalid_argument(
          "focal UCC triples v2 requires real-spherical orbital and correlation-fit bases");
    if (!aux.ecp_terms.empty())
      throw std::invalid_argument(
          "focal UCC triples v2 correlation fit describes Coulomb fitting functions only");
    if (!owner->uhf->source_identity->matches(sys))
      throw std::invalid_argument(
          "focal UCC triples v2 orbital system differs from the retained DF determinant");
    if (sys.atoms.size() != aux.atoms.size())
      throw std::invalid_argument(
          "focal UCC triples v2 orbital and correlation-fit geometries differ");
    for (std::size_t i = 0; i < sys.atoms.size(); ++i)
      if (sys.atoms[i].atomic_number != aux.atoms[i].atomic_number ||
          sys.atoms[i].position != aux.atoms[i].position || aux.atoms[i].ecp_core)
        throw std::invalid_argument(
            "focal UCC triples v2 orbital and correlation-fit nuclei differ");

    const auto& ref = *owner->uhf;
    const std::size_t oa_full = ref.nocc[0];
    const std::size_t ob_full = ref.nocc[1];
    if (frozen_alpha > oa_full || frozen_beta > ob_full)
      throw std::invalid_argument("focal UCC triples v2 frozen count exceeds occupied count");
    const std::size_t oa_active = oa_full - frozen_alpha;
    const std::size_t ob_active = ob_full - frozen_beta;
    const std::size_t n = ref.nbf;
    const std::size_t nn = runtime::size_mul(n, n);
    const std::size_t coords = runtime::size_mul(3, sys.atoms.size());

    if (occupied_partition &&
        ((oa_active && !occupied_out_alpha) || occupied_alpha_count != oa_active ||
         (ob_active && !occupied_out_beta) || occupied_beta_count != ob_active))
      throw std::invalid_argument("focal UCC triples v2 occupied buffer shape mismatch");
    const bool has_wa = weights_alpha_count != 0;
    const bool has_wb = weights_beta_count != 0;
    if ((has_wa && (!occupied_weights_alpha || weights_alpha_count != oa_active)) ||
        (has_wb && (!occupied_weights_beta || weights_beta_count != ob_active)) ||
        (has_wa != has_wb && !(oa_active == 0 || ob_active == 0)))
      throw std::invalid_argument("focal UCC triples v2 occupied weights shape mismatch");
    if (forces && (!corr_gradient_out || corr_gradient_count != coords || !bar_f_alpha_out ||
                   bar_f_alpha_count != nn || !bar_f_beta_out || bar_f_beta_count != nn ||
                   !bar_c_alpha_out || bar_c_alpha_count != nn || !bar_c_beta_out ||
                   bar_c_beta_count != nn))
      throw std::invalid_argument("focal UCC triples v2 force source buffer shape mismatch");
    if (telemetry_out && telemetry_count < 10)
      throw std::invalid_argument("focal UCC triples v2 telemetry buffer requires >= 10 slots");

    cc::DFUccV2Options options;
    options.compute_forces = forces != 0;
    options.compute_triples = true;
    options.occupied_partition = occupied_partition != 0;
    options.energy_source_only = true;
    options.frozen_alpha = frozen_alpha;
    options.frozen_beta = frozen_beta;
    if (has_wa)
      options.occupied_weights_alpha.assign(occupied_weights_alpha,
                                            occupied_weights_alpha + weights_alpha_count);
    if (has_wb)
      options.occupied_weights_beta.assign(occupied_weights_beta,
                                           occupied_weights_beta + weights_beta_count);
    options.preconvergence = static_cast<core::DfPreconvergence>(preconvergence);
    options.triples_fp32_w = triples_fp32_w == 1;
    options.triples_compensated_w = triples_fp32_w == 2;
    if (budget_bytes) options.max_bytes = budget_bytes;

    const auto result =
        cc::run_df_ucc_v2(sys, aux, ref, cuda, context->state.device_id, options);
    if (!std::isfinite(result.ccsd_correlation_energy) || !std::isfinite(result.triples_energy))
      throw std::runtime_error("nonfinite focal UCC triples v2 energy");
    if (ccsd_energy_out) *ccsd_energy_out = result.ccsd_correlation_energy;
    *energy_out = result.triples_energy;
    if (occupied_partition) {
      if (result.occupied_alpha.size() != oa_active || result.occupied_beta.size() != ob_active)
        throw std::runtime_error("focal UCC triples v2 occupied partition size mismatch");
      if (oa_active) std::copy(result.occupied_alpha.begin(), result.occupied_alpha.end(), occupied_out_alpha);
      if (ob_active) std::copy(result.occupied_beta.begin(), result.occupied_beta.end(), occupied_out_beta);
    }
    if (forces) {
      if (result.correlation_gradient.size() != coords || result.bar_f_alpha.size() != nn ||
          result.bar_f_beta.size() != nn || result.bar_c_alpha.size() != nn ||
          result.bar_c_beta.size() != nn)
        throw std::runtime_error("focal UCC triples v2 force source size mismatch");
      std::copy(result.correlation_gradient.begin(), result.correlation_gradient.end(),
                corr_gradient_out);
      std::copy(result.bar_f_alpha.begin(), result.bar_f_alpha.end(), bar_f_alpha_out);
      std::copy(result.bar_f_beta.begin(), result.bar_f_beta.end(), bar_f_beta_out);
      std::copy(result.bar_c_alpha.begin(), result.bar_c_alpha.end(), bar_c_alpha_out);
      std::copy(result.bar_c_beta.begin(), result.bar_c_beta.end(), bar_c_beta_out);
    }
    if (telemetry_out) {
      telemetry_out[0] = result.ccsd_fp32_executed ? 1.0 : 0.0;
      telemetry_out[1] = result.lambda_fp32_executed ? 1.0 : 0.0;
      telemetry_out[2] = static_cast<double>(result.triples_fp32_gemms);
      telemetry_out[3] = static_cast<double>(result.triples_fp64_gemms);
      telemetry_out[4] = result.ccsd_seconds;
      telemetry_out[5] = result.triples_seconds;
      telemetry_out[6] = result.lambda_seconds;
      telemetry_out[7] = result.source_response_seconds;
      telemetry_out[8] = static_cast<double>(result.ccsd_iterations);
      telemetry_out[9] = static_cast<double>(result.lambda_iterations);
      if (telemetry_count >= 11)
        telemetry_out[10] = static_cast<double>(result.triples_fp32_reverse_gemms);
      if (telemetry_count >= 12)
        telemetry_out[11] = result.triples_compensated_w_executed ? 1.0 : 0.0;
    }
    trace.finish();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::length_error& error) {
    record_oom_detail_noalloc(context->last_detail, error.what());
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::bad_alloc& error) {
    record_oom_detail_noalloc(context->last_detail, error.what());
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

extern "C" GENERATIVEQC_API generativeqc_status generativeqc_focal_ucc_f12b_v1(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* correlation_fit, const generativeqc_focal_reference* owner,
    std::size_t budget_bytes, int forces, std::size_t frozen_alpha, std::size_t frozen_beta,
    std::uint32_t preconvergence, const generativeqc_focal_ucc_f12_coupling_v1* coupling,
    double* ccsd_f12b_energy_out, generativeqc_focal_ucc_f12_cotangents_v1* cotangents_out,
    double* corr_gradient_out, std::size_t corr_gradient_count, double* bar_f_alpha_out,
    std::size_t bar_f_alpha_count, double* bar_f_beta_out, std::size_t bar_f_beta_count,
    double* bar_c_alpha_out, std::size_t bar_c_alpha_count, double* bar_c_beta_out,
    std::size_t bar_c_beta_count, double* telemetry_out, std::size_t telemetry_count) {
  if (!context || !orbital || !correlation_fit || !owner || !coupling || !ccsd_f12b_energy_out ||
      (forces != 0 && forces != 1) || preconvergence > 3)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard lock(context->mutex);
  try {
    using namespace generativeqc;
    const auto backend = context->state.executed_backend;
    if (backend != context->state.requested_backend ||
        (backend != core::kCudaBackend && backend != core::kCpuReferenceBackend))
      throw std::invalid_argument("focal UCC-F12b v1 requires an explicit CPU or CUDA context");
    const bool cuda = backend == core::kCudaBackend;
    runtime::df_progress::Scope trace(
        forces ? "focal_ucc_f12b_force_sources_v1" : "focal_ucc_f12b_energy_v1",
        cuda ? "cuda" : "host");
    if (!owner->uhf || !owner->uhf->source_identity ||
        owner->backend != static_cast<generativeqc_backend>(backend) ||
        owner->device != context->state.device_id)
      throw std::invalid_argument(
          "focal UCC-F12b v1 requires a matching retained DF reference owner");
    if (orbital->ks_auxiliary || orbital->correlation_auxiliary || orbital->scf_auxiliary ||
        correlation_fit->ks_auxiliary || correlation_fit->scf_auxiliary)
      throw std::invalid_argument(
          "focal UCC-F12b v1 requires normalized orbital and correlation-fit system handles");
    const auto& sys = orbital->data;
    const auto& aux = correlation_fit->data;
    if (sys.basis_representation != core::kSphericalBasis ||
        aux.basis_representation != core::kSphericalBasis)
      throw std::invalid_argument(
          "focal UCC-F12b v1 requires real-spherical orbital and correlation-fit bases");
    if (!aux.ecp_terms.empty())
      throw std::invalid_argument(
          "focal UCC-F12b v1 correlation fit describes Coulomb fitting functions only");
    if (!owner->uhf->source_identity->matches(sys))
      throw std::invalid_argument(
          "focal UCC-F12b v1 orbital system differs from the retained DF determinant");
    if (sys.atoms.size() != aux.atoms.size())
      throw std::invalid_argument(
          "focal UCC-F12b v1 orbital and correlation-fit geometries differ");
    for (std::size_t i = 0; i < sys.atoms.size(); ++i)
      if (sys.atoms[i].atomic_number != aux.atoms[i].atomic_number ||
          sys.atoms[i].position != aux.atoms[i].position || aux.atoms[i].ecp_core)
        throw std::invalid_argument(
            "focal UCC-F12b v1 orbital and correlation-fit nuclei differ");

    const auto& ref = *owner->uhf;
    const std::size_t oa_full = ref.nocc[0];
    const std::size_t ob_full = ref.nocc[1];
    if (frozen_alpha > oa_full || frozen_beta > ob_full)
      throw std::invalid_argument("focal UCC-F12b v1 frozen count exceeds occupied count");
    const std::size_t oa = oa_full - frozen_alpha;
    const std::size_t ob = ob_full - frozen_beta;
    const std::size_t n = ref.nbf;
    const std::size_t va = n - oa_full;
    const std::size_t vb = n - ob_full;
    const std::size_t nn = runtime::size_mul(n, n);
    const std::size_t coords = runtime::size_mul(3, sys.atoms.size());

    auto mul4 = [](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
      return runtime::size_mul(runtime::size_mul(a, b), runtime::size_mul(c, d));
    };
    const std::size_t sz_w_vo_aaaa = mul4(oa, oa, va, oa);
    const std::size_t sz_w_vo_bbbb = mul4(ob, ob, vb, ob);
    const std::size_t sz_w_vo_abab = mul4(oa, ob, va, ob);
    const std::size_t sz_w_vo_abba = mul4(oa, ob, vb, oa);
    const std::size_t sz_w_vv_aaaa = mul4(oa, oa, va, va);
    const std::size_t sz_w_vv_abab = mul4(oa, ob, va, vb);
    const std::size_t sz_w_vv_bbbb = mul4(ob, ob, vb, vb);

    if (!std::isfinite(coupling->fixed_geminal_energy))
      throw std::invalid_argument("focal UCC-F12b v1 fixed_geminal_energy is nonfinite");

    cc::FixedF12Coupling f12;
    f12.fixed_geminal_energy = coupling->fixed_geminal_energy;
    auto load_in = [](const double* ptr, std::size_t count, std::size_t expected,
                      std::vector<double>& dst, const char* name) {
      if (count != expected || (expected && !ptr))
        throw std::invalid_argument(std::string("focal UCC-F12b v1 coupling shape mismatch for ") +
                                    name);
      if (expected) {
        dst.assign(ptr, ptr + expected);
        if (!std::all_of(dst.begin(), dst.end(), [](double x) { return std::isfinite(x); }))
          throw std::invalid_argument(std::string("focal UCC-F12b v1 nonfinite coupling in ") +
                                      name);
      }
    };
    load_in(coupling->w_vo_aaaa, coupling->w_vo_aaaa_count, sz_w_vo_aaaa, f12.w_vo_aaaa,
            "w_vo_aaaa");
    load_in(coupling->w_vo_bbbb, coupling->w_vo_bbbb_count, sz_w_vo_bbbb, f12.w_vo_bbbb,
            "w_vo_bbbb");
    load_in(coupling->w_vo_abab, coupling->w_vo_abab_count, sz_w_vo_abab, f12.w_vo_abab,
            "w_vo_abab");
    load_in(coupling->w_vo_abba, coupling->w_vo_abba_count, sz_w_vo_abba, f12.w_vo_abba,
            "w_vo_abba");
    load_in(coupling->w_vv_aaaa, coupling->w_vv_aaaa_count, sz_w_vv_aaaa, f12.w_vv_aaaa,
            "w_vv_aaaa");
    load_in(coupling->w_vv_abab, coupling->w_vv_abab_count, sz_w_vv_abab, f12.w_vv_abab,
            "w_vv_abab");
    load_in(coupling->w_vv_bbbb, coupling->w_vv_bbbb_count, sz_w_vv_bbbb, f12.w_vv_bbbb,
            "w_vv_bbbb");
    load_in(coupling->c_f12_aaaa, coupling->c_f12_aaaa_count, sz_w_vv_aaaa, f12.c_f12_aaaa,
            "c_f12_aaaa");
    load_in(coupling->c_f12_abab, coupling->c_f12_abab_count, sz_w_vv_abab, f12.c_f12_abab,
            "c_f12_abab");
    load_in(coupling->c_f12_bbbb, coupling->c_f12_bbbb_count, sz_w_vv_bbbb, f12.c_f12_bbbb,
            "c_f12_bbbb");

    if (forces) {
      if (!cotangents_out || !cotangents_out->bar_fixed_geminal_energy)
        throw std::invalid_argument("focal UCC-F12b v1 forces require cotangents_out struct");
      auto check_out = [](double* ptr, std::size_t count, std::size_t expected, const char* name) {
        if (count != expected || (expected && !ptr))
          throw std::invalid_argument(
              std::string("focal UCC-F12b v1 cotangent buffer shape mismatch for ") + name);
      };
      check_out(cotangents_out->bar_w_vo_aaaa, cotangents_out->bar_w_vo_aaaa_count, sz_w_vo_aaaa,
                "bar_w_vo_aaaa");
      check_out(cotangents_out->bar_w_vo_bbbb, cotangents_out->bar_w_vo_bbbb_count, sz_w_vo_bbbb,
                "bar_w_vo_bbbb");
      check_out(cotangents_out->bar_w_vo_abab, cotangents_out->bar_w_vo_abab_count, sz_w_vo_abab,
                "bar_w_vo_abab");
      check_out(cotangents_out->bar_w_vo_abba, cotangents_out->bar_w_vo_abba_count, sz_w_vo_abba,
                "bar_w_vo_abba");
      check_out(cotangents_out->bar_w_vv_aaaa, cotangents_out->bar_w_vv_aaaa_count, sz_w_vv_aaaa,
                "bar_w_vv_aaaa");
      check_out(cotangents_out->bar_w_vv_abab, cotangents_out->bar_w_vv_abab_count, sz_w_vv_abab,
                "bar_w_vv_abab");
      check_out(cotangents_out->bar_w_vv_bbbb, cotangents_out->bar_w_vv_bbbb_count, sz_w_vv_bbbb,
                "bar_w_vv_bbbb");
      check_out(cotangents_out->bar_c_f12_aaaa, cotangents_out->bar_c_f12_aaaa_count, sz_w_vv_aaaa,
                "bar_c_f12_aaaa");
      check_out(cotangents_out->bar_c_f12_abab, cotangents_out->bar_c_f12_abab_count, sz_w_vv_abab,
                "bar_c_f12_abab");
      check_out(cotangents_out->bar_c_f12_bbbb, cotangents_out->bar_c_f12_bbbb_count, sz_w_vv_bbbb,
                "bar_c_f12_bbbb");
      if (!corr_gradient_out || corr_gradient_count != coords || !bar_f_alpha_out ||
          bar_f_alpha_count != nn || !bar_f_beta_out || bar_f_beta_count != nn ||
          !bar_c_alpha_out || bar_c_alpha_count != nn || !bar_c_beta_out ||
          bar_c_beta_count != nn)
        throw std::invalid_argument("focal UCC-F12b v1 ordinary force source buffer shape mismatch");
    }
    if (telemetry_out && telemetry_count < 10)
      throw std::invalid_argument("focal UCC-F12b v1 telemetry buffer requires >= 10 slots");

    cc::DFUccV2Options options;
    options.compute_forces = forces != 0;
    options.compute_triples = false;
    options.occupied_partition = false;
    options.energy_source_only = false;
    options.frozen_alpha = frozen_alpha;
    options.frozen_beta = frozen_beta;
    options.preconvergence = static_cast<core::DfPreconvergence>(preconvergence);
    options.triples_fp32_w = false;
    options.f12 = &f12;
    if (budget_bytes) options.max_bytes = budget_bytes;

    const auto result =
        cc::run_df_ucc_v2(sys, aux, ref, cuda, context->state.device_id, options);
    if (!std::isfinite(result.ccsd_correlation_energy))
      throw std::runtime_error("nonfinite focal UCC-F12b v1 energy");
    *ccsd_f12b_energy_out = result.ccsd_correlation_energy;

    if (forces) {
      if (result.correlation_gradient.size() != coords || result.bar_f_alpha.size() != nn ||
          result.bar_f_beta.size() != nn || result.bar_c_alpha.size() != nn ||
          result.bar_c_beta.size() != nn)
        throw std::runtime_error("focal UCC-F12b v1 ordinary force source size mismatch");
      std::copy(result.correlation_gradient.begin(), result.correlation_gradient.end(),
                corr_gradient_out);
      std::copy(result.bar_f_alpha.begin(), result.bar_f_alpha.end(), bar_f_alpha_out);
      std::copy(result.bar_f_beta.begin(), result.bar_f_beta.end(), bar_f_beta_out);
      std::copy(result.bar_c_alpha.begin(), result.bar_c_alpha.end(), bar_c_alpha_out);
      std::copy(result.bar_c_beta.begin(), result.bar_c_beta.end(), bar_c_beta_out);

      const auto& fc = result.f12_cotangents;
      *cotangents_out->bar_fixed_geminal_energy = fc.bar_fixed_geminal_energy;
      auto copy_out = [](const std::vector<double>& src, double* dst, std::size_t expected,
                         const char* name) {
        if (src.size() != expected)
          throw std::runtime_error(std::string("focal UCC-F12b v1 cotangent size mismatch for ") +
                                   name);
        if (expected) std::copy(src.begin(), src.end(), dst);
      };
      copy_out(fc.bar_w_vo_aaaa, cotangents_out->bar_w_vo_aaaa, sz_w_vo_aaaa, "bar_w_vo_aaaa");
      copy_out(fc.bar_w_vo_bbbb, cotangents_out->bar_w_vo_bbbb, sz_w_vo_bbbb, "bar_w_vo_bbbb");
      copy_out(fc.bar_w_vo_abab, cotangents_out->bar_w_vo_abab, sz_w_vo_abab, "bar_w_vo_abab");
      copy_out(fc.bar_w_vo_abba, cotangents_out->bar_w_vo_abba, sz_w_vo_abba, "bar_w_vo_abba");
      copy_out(fc.bar_w_vv_aaaa, cotangents_out->bar_w_vv_aaaa, sz_w_vv_aaaa, "bar_w_vv_aaaa");
      copy_out(fc.bar_w_vv_abab, cotangents_out->bar_w_vv_abab, sz_w_vv_abab, "bar_w_vv_abab");
      copy_out(fc.bar_w_vv_bbbb, cotangents_out->bar_w_vv_bbbb, sz_w_vv_bbbb, "bar_w_vv_bbbb");
      copy_out(fc.bar_c_f12_aaaa, cotangents_out->bar_c_f12_aaaa, sz_w_vv_aaaa, "bar_c_f12_aaaa");
      copy_out(fc.bar_c_f12_abab, cotangents_out->bar_c_f12_abab, sz_w_vv_abab, "bar_c_f12_abab");
      copy_out(fc.bar_c_f12_bbbb, cotangents_out->bar_c_f12_bbbb, sz_w_vv_bbbb, "bar_c_f12_bbbb");
    }
    if (telemetry_out) {
      telemetry_out[0] = result.ccsd_fp32_executed ? 1.0 : 0.0;
      telemetry_out[1] = result.lambda_fp32_executed ? 1.0 : 0.0;
      telemetry_out[2] = static_cast<double>(result.triples_fp32_gemms);
      telemetry_out[3] = static_cast<double>(result.triples_fp64_gemms);
      telemetry_out[4] = result.ccsd_seconds;
      telemetry_out[5] = result.triples_seconds;
      telemetry_out[6] = result.lambda_seconds;
      telemetry_out[7] = result.source_response_seconds;
      telemetry_out[8] = static_cast<double>(result.ccsd_iterations);
      telemetry_out[9] = static_cast<double>(result.lambda_iterations);
    }
    trace.finish();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::length_error& error) {
    record_oom_detail_noalloc(context->last_detail, error.what());
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::bad_alloc& error) {
    record_oom_detail_noalloc(context->last_detail, error.what());
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

extern "C" GENERATIVEQC_API void generativeqc_cc_reset_context_lambda_telemetry_v1(
    generativeqc_context* context) {
  if (!context) return;
  std::lock_guard lock(context->mutex);
  context->lambda_telemetry.reset();
}

extern "C" GENERATIVEQC_API int generativeqc_cc_context_lambda_telemetry_v1(
    const generativeqc_context* context, double* scalars, std::size_t scalar_count,
    std::uint64_t* counts, std::size_t count_len, char* reason, std::size_t reason_len) {
  if (!context) return 0;
  std::lock_guard lock(context->mutex);
  return generativeqc::cc::detail::copy_lambda_telemetry(
      context->lambda_telemetry, scalars, scalar_count, counts, count_len, reason, reason_len);
}
