// Validation bridge for composing molecular F12 sources with the native DF
// reference owner. Geometry and complete basis coefficients are the only SCF
// inputs. The immutable reference and RI-JK plan survive every response call.
// This seam is not a public SDK ABI or a complete molecular F12 endpoint.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "hf/df_reference.hpp"
#include "hf/rhf_frame_response.hpp"
#include "molecule/basis.hpp"
#include "molecule/nuclear_gradient.hpp"
#include "posthf/capacity.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using namespace generativeqc;

struct Session {
  core::System orbital;
  std::shared_ptr<hf::DFReference> owner = std::make_shared<hf::DFReference>();
  std::shared_ptr<const hf::PhysicalReference> reference;
  std::size_t budget{};
  int device{};
  unsigned iterations{};
  double energy{}, scf_seconds{};
};

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

template <typename Action>
int guarded(Action action, char* error, std::size_t capacity) noexcept {
  try {
    action();
    if (error && capacity) error[0] = '\0';
    return 0;
  } catch (const std::exception& failure) {
    if (error && capacity) std::snprintf(error, capacity, "%s", failure.what());
  } catch (...) {
    if (error && capacity) std::snprintf(error, capacity, "unknown native reference failure");
  }
  return 1;
}

void shells(std::istream& input, core::System& system, std::size_t count) {
  system.shells.resize(count);
  for (auto& shell : system.shells) {
    std::size_t primitives{};
    input >> shell.atom_index >> shell.angular_momentum >> primitives;
    require(bool(input) && primitives && primitives <= 1000, "invalid reference shell");
    shell.primitives.resize(primitives);
    for (auto& p : shell.primitives) input >> p.exponent >> p.coefficient;
  }
  require(bool(input), "truncated reference basis");
  std::string detail;
  if (molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::invalid_argument(detail);
}

const Session& session(void* handle) {
  require(handle != nullptr, "closed or absent DF reference session");
  return *static_cast<Session*>(handle);
}
}  // namespace

extern "C" int gqc_df_session_create(const char* path, int device, void** output,
                                     std::size_t* dimensions, char* error,
                                     std::size_t error_size) noexcept {
  // Publication is atomic: a failed solve never leaves a partially usable owner.
  if (output) *output = nullptr;
  return guarded(
      [&] {
        require(path && output && dimensions && device >= 0, "invalid session arguments");
        auto state = std::make_unique<Session>();
        std::ifstream input(path);
        std::size_t atoms{}, orbital_shells{}, auxiliary_shells{};
        input >> atoms >> orbital_shells >> auxiliary_shells >> state->budget >>
            state->orbital.charge >> state->orbital.multiplicity;
        require(bool(input) && atoms && atoms <= 1000 && orbital_shells &&
                    orbital_shells <= 100000 && auxiliary_shells && auxiliary_shells <= 100000 &&
                    state->budget && state->orbital.multiplicity == 1,
                "invalid closed-shell reference metadata");
        state->device = device;
        state->orbital.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
        state->orbital.atoms.resize(atoms);
        for (auto& atom : state->orbital.atoms)
          input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
        auto auxiliary = state->orbital;
        shells(input, state->orbital, orbital_shells);
        shells(input, auxiliary, auxiliary_shells);
        input >> std::ws;
        require(input.eof(), "unexpected trailing reference input");
        require(state->orbital.electron_count >= 2 && state->orbital.electron_count % 2 == 0,
                "reference requires paired electrons");
        scf::ScfOptions options;
        options.max_iterations = 200;
        options.energy_tolerance = 1e-13;
        options.density_tolerance = 1e-11;
        options.density_fitting_relative_threshold = 1e-10;
        options.reference_memory_budget_bytes = state->budget;
        options.precision_mode = GENERATIVEQC_PRECISION_FP64;
        const auto started = Clock::now();
        auto result = state->owner->solve(state->orbital, auxiliary, options, device, nullptr);
        state->scf_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        require(result.converged && result.reference && std::isfinite(result.energy),
                "native DF-RHF did not publish a converged reference");
        state->reference = std::move(result.reference);
        state->energy = result.energy;
        state->iterations = result.iterations;
        require(state->owner->matches(state->orbital, *state->reference, device),
                "forward/reference owner identity mismatch");
        dimensions[0] = state->reference->nbf;
        dimensions[1] = state->reference->nocc;
        dimensions[2] = atoms;
        *output = state.release();
      },
      error, error_size);
}

extern "C" void gqc_df_session_destroy(void* handle) noexcept {
  delete static_cast<Session*>(handle);
}

extern "C" int gqc_df_session_snapshot(void* handle, double* matrices, std::size_t matrix_count,
                                       double* energies, std::size_t energy_count,
                                       double* diagnostics, std::size_t diagnostic_count,
                                       char* error, std::size_t error_size) noexcept {
  return guarded(
      [&] {
        const auto& state = session(handle);
        const auto& ref = *state.reference;
        const auto nn = posthf::checked_mul(ref.nbf, ref.nbf);
        require(matrices && energies && diagnostics && matrix_count == 5 * nn &&
                    energy_count == ref.nbf && diagnostic_count == 6,
                "invalid DF reference snapshot dimensions");
        std::size_t offset{};
        for (const auto* matrix :
             {&ref.coefficients, &ref.overlap, &ref.fock, &ref.hcore, &ref.density}) {
          require(matrix->size() == nn, "incomplete native reference matrix");
          std::copy(matrix->begin(), matrix->end(), matrices + offset);
          offset += nn;
        }
        require(ref.orbital_energies.size() == ref.nbf, "incomplete native orbital energies");
        std::copy(ref.orbital_energies.begin(), ref.orbital_energies.end(), energies);
        diagnostics[0] = state.energy;
        diagnostics[1] = state.scf_seconds;
        diagnostics[2] = state.iterations;
        diagnostics[3] = ref.commutator_residual;
        diagnostics[4] = ref.canonical_density_drift;
        diagnostics[5] = ref.eigen_residual;
      },
      error, error_size);
}

extern "C" int gqc_df_session_response(void* handle, const double* bar_fock,
                                       const double* bar_frame, std::size_t matrix_count,
                                       std::size_t maximum_bytes, std::size_t caller_bytes,
                                       double* forces, std::size_t force_count, double* diagnostics,
                                       std::size_t diagnostic_count, char* error,
                                       std::size_t error_size) noexcept {
  return guarded(
      [&] {
        const auto& state = session(handle);
        const auto nn = posthf::checked_mul(state.reference->nbf, state.reference->nbf);
        require(bar_fock && bar_frame && forces && diagnostics && matrix_count == nn &&
                    force_count == 3 * state.orbital.atoms.size() && diagnostic_count == 6 &&
                    maximum_bytes && maximum_bytes <= state.budget,
                "invalid DF response dimensions or allowance");
        hf::RHFFrameResponseOptions options;
        options.df_reference = state.owner;
        options.maximum_bytes = maximum_bytes;
        options.caller_bytes = caller_bytes;
        const auto started = Clock::now();
        auto result = hf::rhf_frame_response_cuda(state.orbital, *state.reference, {bar_fock, nn},
                                                  {bar_frame, nn}, state.device, options);
        // This response includes HF with seed one. Add explicit F12 integral /
        // metric gradients separately; never add a second HF force to this value.
        molecule::add_nuclear_repulsion_gradient(state.orbital, result.gradient);
        require(result.gradient.size() == force_count, "incomplete native response");
        for (double g : result.gradient) require(std::isfinite(g), "nonfinite native response");
        const double values[]{std::chrono::duration<double>(Clock::now() - started).count(),
                              result.orbital_residual,
                              result.maximum_stationarity,
                              result.reference_residual,
                              static_cast<double>(result.numeric_capacity_bytes),
                              static_cast<double>(result.orbital_actions)};
        for (double x : values) require(std::isfinite(x), "nonfinite response diagnostic");
        // All validation and device work finish before any caller output changes.
        std::transform(result.gradient.begin(), result.gradient.end(), forces,
                       [](double g) { return -g; });
        std::copy(std::begin(values), std::end(values), diagnostics);
      },
      error, error_size);
}
