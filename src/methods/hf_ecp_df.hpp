#pragma once

#include <algorithm>

#include "methods/method.hpp"
#include "scf/cuda/df_spin_domain.hpp"

namespace generativeqc::methods::detail {

// Check the fitted scalar small-core halide domains. The native source
// and spin solver independently validate the actual fitting and orbital shells.
inline void prepare_ecp_df_hf(core::System& system, const scf::ScfOptions& options,
                              generativeqc_method method, const runtime::ExecutionContext& execution,
                              const core::System* auxiliary) {
  if (system.ecp_terms.empty() || options.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE)
    return;
  const auto composition = scf::cuda_execution::halide_composition(system);
  const bool restricted = method == GENERATIVEQC_METHOD_RHF && system.multiplicity == 1;
  const bool unrestricted = method == GENERATIVEQC_METHOD_UHF && system.multiplicity == 2;
  const bool hydride = composition.hydride() && system.charge == 0 &&
                       (unrestricted || (restricted && composition.hydrogen == 1 &&
                                         system.electron_count == 26));
  const bool methyl = composition.methyl_halide() &&
                      ((restricted && system.charge == 0 && system.electron_count == 34) ||
                       (unrestricted && system.charge == 1 && system.electron_count == 33));
  const unsigned limit = restricted || methyl ||
                         scf::cuda_execution::fitted_halogen_dihydride_doublet(system) ? 4U : 3U;
  if ((!hydride && !methyl) || !execution.cuda_requested() ||
      options.density_fitting_mode != GENERATIVEQC_DENSITY_FITTING_CUDA ||
      options.precision_mode != GENERATIVEQC_PRECISION_FP64 || !auxiliary ||
      system.basis_representation != core::kSphericalBasis ||
      std::any_of(system.shells.begin(), system.shells.end(),
                  [limit](const auto& s) { return s.angular_momentum > limit; }) ||
      std::any_of(auxiliary->shells.begin(), auxiliary->shells.end(),
                  [](const auto& s) { return s.angular_momentum > 5; }))
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "ECP density-fitting execution is not yet validated for this reference domain");
  system.refine_ecp_quadrature = composition.bromine == 1;
}

}  // namespace generativeqc::methods::detail
