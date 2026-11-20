// Python bridge to the native canonical DF-HF owner at one geometry.
#include <algorithm>
#include <cmath>
#include <optional>
#include <span>
#include <vector>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "methods/hf_ecp_df.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/execution_context.hpp"
#include "scf/mean_field.hpp"

namespace {
generativeqc_status focal_df_reference(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* auxiliary, int spins, int forces, std::size_t budget,
    const double* initial_density, std::size_t initial_count, double* output, std::size_t count) {
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
    if (runtime::size_mul(runtime::size_add(count, initial_count), sizeof(double)) > budget)
      throw std::length_error("focal reference export exceeds the numeric budget");

    const auto backend = context->state.executed_backend;
    if (backend != context->state.requested_backend ||
        (backend != core::kCudaBackend && backend != core::kCpuReferenceBackend))
      throw std::invalid_argument("focal reference requires an explicit CPU or CUDA context");
    const bool cuda = backend == core::kCudaBackend;
    scf::ScfOptions options;
    const auto capacity = spins == 1
                              ? posthf::rhf_reference_capacity(sys, options.diis_history, false)
                              : posthf::uhf_reference_capacity(sys, options.diis_history, false);
    if (capacity > budget) {
      context->last_detail = "DF-HF physical reference exceeds its numeric capacity budget";
      return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
    }
    options.max_iterations = 240;
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
    std::optional<core::System> prepared_ecp;
    if (!sys.ecp_terms.empty()) {
      prepared_ecp = sys;
      options.density_fitting_mode = cuda ? GENERATIVEQC_DENSITY_FITTING_CUDA
                                          : GENERATIVEQC_DENSITY_FITTING_CPU_REFERENCE;
      methods::detail::prepare_ecp_df_hf(*prepared_ecp, options,
          spins == 1 ? GENERATIVEQC_METHOD_RHF : GENERATIVEQC_METHOD_UHF,
          runtime::ExecutionContext(context->state), &aux);
    }
    const auto& physical = prepared_ecp ? *prepared_ecp : sys;
    std::vector<double> seed;
    if (initial_density) {
      seed.assign(initial_density, initial_density + initial_count);
      options.strict_initial_density = true;
    }
    const auto* start = initial_density ? &seed : nullptr;
    const auto state = cuda ? scf::run_cuda_independent_fock_strategy(
                                  physical, &aux, options, context->state.device_id, start)
                            : scf::run_cpu_fock_strategy(physical, &aux, options, start);
    if (start && !state.initial_density_used)
      throw std::runtime_error("DF-HF did not use the requested initial density");
    if (!state.converged) {
      context->last_detail = "canonical DF-HF did not converge";
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
    return GENERATIVEQC_STATUS_SUCCESS;
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
