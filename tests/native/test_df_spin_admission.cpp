#include <iostream>
#include <stdexcept>
#include <string>

#include "api/handles.hpp"
#include "generativeqc/fock.h"
#include "scf/cuda/df_source_domain.hpp"
#include "scf/cuda/df_spin_domain.hpp"
#include "scf/mean_field.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void public_plan_admission() {
  // Inject only the context's backend selection, so this host-only admission
  // test never needs to initialize CUDA. The real C adapter, resolver and
  // PreparedFockPlan constructor must reject before using the invalid device.
  generativeqc_context context;
  context.state.executed_backend = GENERATIVEQC_BACKEND_CUDA;
  context.state.device_id = -1;
  for (int representation : {GENERATIVEQC_BASIS_CARTESIAN, GENERATIVEQC_BASIS_SPHERICAL}) {
    for (unsigned derivative : {0U, 1U}) {
      for (bool restricted : {false, true}) {
        for (unsigned angular : {3U, 4U}) {
          for (int fitted_term : {0, 1, 2}) {
            generativeqc_system orbital;
            orbital.data.atoms = {{2, {0, 0, 0}}};
            orbital.data.shells = {{0, angular, {{1.0, 1.0}}}};
            orbital.data.electron_count = 2;
            orbital.data.multiplicity = 1;
            orbital.data.basis_representation = representation;
            generativeqc_system auxiliary = orbital;
            auxiliary.data.shells[0].angular_momentum = 0;
            generativeqc_fock_spec spec{
                sizeof(spec),
                GENERATIVEQC_ABI_VERSION,
                1,
                restricted ? GENERATIVEQC_FOCK_RESTRICTED : GENERATIVEQC_FOCK_UNRESTRICTED,
                derivative,
                {1, 1.0, GENERATIVEQC_FOCK_FULL_RANGE, 0.0,
                 fitted_term == 1 ? GENERATIVEQC_FOCK_EXACT : GENERATIVEQC_FOCK_DENSITY_FITTED},
                {1, restricted ? -0.5 : -1.0, GENERATIVEQC_FOCK_FULL_RANGE, 0.0,
                 fitted_term == 0 ? GENERATIVEQC_FOCK_EXACT : GENERATIVEQC_FOCK_DENSITY_FITTED}};
            generativeqc_fock_plan* plan = nullptr;
            const auto status = generativeqc_fock_plan_create(&context, &orbital, &auxiliary, &spec,
                                                              nullptr, &plan);
            require(plan == nullptr, "admission failure must not publish a plan");
            const bool admitted_all_electron_uhf_g =
                !restricted && angular == 4 && representation == GENERATIVEQC_BASIS_SPHERICAL &&
                fitted_term == 2;
            if (!restricted && angular == 4 && !admitted_all_electron_uhf_g) {
              require(status == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
                      "unqualified unrestricted g must fail native admission");
              require(context.last_detail.find("unrestricted CUDA DF") != std::string::npos,
                      "unrestricted g must fail before provider or device setup");
            } else {
              // CUDA-enabled builds reach the invalid-device sentinel. Host
              // builds reach provider availability instead. Neither is a claim
              // that these numerical endpoints were executed by this test.
              require(status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED ||
                          (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                           context.last_detail == "CUDA prepared Fock requires a device"),
                      "restricted g, unrestricted f, and spherical full-range DF-UHF g must pass "
                      "spin admission");
            }
          }
        }
      }
    }
  }
}

void resident_admission() {
#if GENERATIVEQC_HAS_CUDA
  using namespace generativeqc::scf;
  generativeqc::core::System orbital;
  orbital.atoms = {{2, {0, 0, 0}}};
  orbital.shells = {{0, 4, {{1.0, 1.0}}}};
  orbital.electron_count = 2;
  orbital.multiplicity = 1;
  const ScfOptions options;
  for (int entry : {0, 1, 2}) {
    bool rejected = false;
    try {
      if (entry == 0)
        (void)run_uhf_density_fitting_cuda(orbital, orbital, options, -1);
      else if (entry == 1)
        (void)run_uhf_density_fitting_cuda_bucket({orbital}, orbital, options, {nullptr}, -1);
      else {
        CudaDensityFittingJkPlan* plan = nullptr;
        (void)run_uhf_density_fitting_cuda_bucket_cached(&plan, {orbital}, orbital, options,
                                                         {nullptr}, -1);
      }
    } catch (const std::invalid_argument& error) {
      rejected = std::string(error.what()).find("unrestricted CUDA DF") != std::string::npos;
    }
    require(rejected, "resident UHF entries must reject g before preparation");
  }
#endif
}

void independent_source_limits() {
  using namespace generativeqc::scf;
  using namespace generativeqc::scf::cuda_execution;
  generativeqc::core::System orbital, auxiliary;
  orbital.shells = {{0, 4, {{1.0, 1.0}}}};
  auxiliary.shells = {{0, 5, {{1.0, 1.0}}}};
  CudaDfSourcePolicy policy;
  std::string detail;
  require_cuda_df_spin_domain(orbital, FockSpin::Restricted);
  require(cuda_df_value_domain(orbital, auxiliary, policy, detail),
          "qualified generic full-range g/h source must remain available");
  for (unsigned math : {1U, 2U, 3U}) {
    policy.value_math = math;
    require(!cuda_df_value_domain(orbital, auxiliary, policy, detail),
            "spin admission must not widen non-generic source math");
  }
  policy.value_math = 0;
  policy.range_omega = 0.3;
  require(!cuda_df_value_domain(orbital, auxiliary, policy, detail),
          "spin admission must not widen range-separated source qualification");
}
void fully_polarized_hydrogen_domain() {
  using namespace generativeqc;
  core::System dimer;
  dimer.atoms = {{1, {0.1, 0.2, 0.3}}, {1, {-0.2, 1.4, 0.8}}};
  dimer.basis_representation = core::kSphericalBasis;
  dimer.multiplicity = 3;
  dimer.shells = {{0, 4, {{1.0, 1.0}}}};
  auto admitted = [](const core::System& sys) {
    try {
      scf::cuda_execution::require_cuda_df_spin_domain(sys, scf::FockSpin::Unrestricted);
      return true;
    } catch (const std::invalid_argument&) {
      return false;
    }
  };
  require(admitted(dimer), "orbital g in all-electron triplet hydrogen dimer");
  auto other = dimer;
  other.multiplicity = 1;
  require(!admitted(other), "unqualified paired spin domain");
  other = dimer;
  other.charge = 1;
  require(!admitted(other), "unqualified charged dimer");
  other = dimer;
  other.atoms[1].atomic_number = 2;
  require(!admitted(other), "unqualified heteronuclear dimer");
  other = dimer;
  other.basis_representation = core::kCartesianBasis;
  require(!admitted(other), "unqualified Cartesian orbital g");
  other = dimer;
  other.shells.front().angular_momentum = 5;
  require(!admitted(other), "orbital h");
  other = dimer;
  other.ecp_terms.resize(1);
  require(!admitted(other), "unqualified pseudopotential");
}

void neutral_oh_value_domain() {
  using namespace generativeqc;
  using namespace generativeqc::scf;
  using namespace generativeqc::scf::cuda_execution;
  core::System oh;
  // O-H length of GSCDB W4-17_oh (CC BY 4.0, see THIRD_PARTY_NOTICES.md), in Bohr.
  oh.atoms = {{8, {0, 0, 0}}, {1, {0, 0, 1.834281919177214}}};
  oh.shells = {{0, 4, {{1.0, 1.0}}}};
  oh.electron_count = 9;
  oh.multiplicity = 2;
  oh.basis_representation = core::kSphericalBasis;
  auto admitted = [](const core::System& system, bool full_range_fully_fitted) {
    try {
      require_cuda_df_spin_domain(system, FockSpin::Unrestricted, full_range_fully_fitted);
      return true;
    } catch (const std::invalid_argument&) {
      return false;
    }
  };
  require(admitted(oh, true), "orbital g in spherical all-electron OH full-range DF-UHF");
  require(!admitted(oh, false), "OH orbital g rejects mixed exact/DF or range-separated plans");
  auto general_all_electron = oh;
  general_all_electron.charge = -1;
  general_all_electron.electron_count = 10;
  require(admitted(general_all_electron, true),
          "general all-electron spherical full-range DF-UHF admits orbital g");
  general_all_electron = oh;
  general_all_electron.atoms[1].atomic_number = 9;
  require(admitted(general_all_electron, true),
          "arbitrary all-electron nuclei admit orbital g on full-range DF-UHF");
  auto rejected = oh;
  rejected.basis_representation = core::kCartesianBasis;
  require(!admitted(rejected, true), "Cartesian OH orbital g is outside the domain");
  rejected = oh;
  rejected.shells[0].angular_momentum = 5;
  require(!admitted(rejected, true), "spherical OH orbital h is outside the domain");
  rejected = oh;
  rejected.ecp_terms.resize(1);
  require(!admitted(rejected, true), "unqualified ECP is outside the orbital-g domain");

  auto auxiliary = oh;
  auxiliary.shells[0].angular_momentum = 5;
  CudaDfSourcePolicy policy;
  std::string detail;
  require(cuda_df_value_domain(oh, auxiliary, policy, detail),
          "the existing generic Coulomb value source supports OH orbital g");
  policy.range_omega = 0.3;
  require(!cuda_df_value_domain(oh, auxiliary, policy, detail),
          "the OH spin-domain update retains the range-separated source guard");
}

void neutral_oh_public_plan_admission() {
  generativeqc_context context;
  context.state.executed_backend = GENERATIVEQC_BACKEND_CUDA;
  context.state.device_id = -1;
  generativeqc_system orbital;
  // O-H length of GSCDB W4-17_oh (CC BY 4.0, see THIRD_PARTY_NOTICES.md), in Bohr.
  orbital.data.atoms = {{8, {0, 0, 0}}, {1, {0, 0, 1.834281919177214}}};
  orbital.data.shells = {{0, 4, {{1.0, 1.0}}}};
  orbital.data.electron_count = 9;
  orbital.data.multiplicity = 2;
  orbital.data.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  auto auxiliary = orbital;
  auxiliary.data.shells[0].angular_momentum = 0;
  for (unsigned derivative : {0U, 1U}) {
    for (bool fully_fitted : {false, true}) {
      generativeqc_fock_spec spec{
          sizeof(spec), GENERATIVEQC_ABI_VERSION, 1, GENERATIVEQC_FOCK_UNRESTRICTED, derivative,
          {1, 1.0, GENERATIVEQC_FOCK_FULL_RANGE, 0.0, GENERATIVEQC_FOCK_DENSITY_FITTED},
          {1, -1.0, GENERATIVEQC_FOCK_FULL_RANGE, 0.0,
           fully_fitted ? GENERATIVEQC_FOCK_DENSITY_FITTED : GENERATIVEQC_FOCK_EXACT}};
      generativeqc_fock_plan* plan = nullptr;
      const auto status =
          generativeqc_fock_plan_create(&context, &orbital, &auxiliary, &spec, nullptr, &plan);
      require(plan == nullptr, "invalid device must not publish an OH plan");
      if (fully_fitted) {
        require(status == GENERATIVEQC_STATUS_NOT_IMPLEMENTED ||
                    (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                     context.last_detail == "CUDA prepared Fock requires a device"),
                "neutral spherical OH fully fitted values and derivatives must reach device setup");
      } else {
        require(status == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                    context.last_detail.find("unrestricted CUDA DF") != std::string::npos,
                "neutral OH g must reject mixed exact/fitted plans");
      }
    }
  }
}
}  // namespace

int main() {
  try {
    fully_polarized_hydrogen_domain();
    neutral_oh_value_domain();
    neutral_oh_public_plan_admission();
    public_plan_admission();
    independent_source_limits();
    resident_admission();
    std::cout << "native CUDA DF spin admission passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
