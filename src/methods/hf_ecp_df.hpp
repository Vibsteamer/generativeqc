#pragma once

#include <algorithm>

#include "methods/method.hpp"
#include "scf/cuda/df_spin_domain.hpp"

namespace generativeqc::methods::detail {

// Admit the tested scalar small-core halogen hydride domain. The native source
// and spin solver independently validate the actual fitting and orbital shells.
inline void prepare_ecp_df_hf(core::System& system, const scf::ScfOptions& options,
                              generativeqc_method method, const runtime::ExecutionContext& execution,
                              const core::System* auxiliary) {
  if (system.ecp_terms.empty() || options.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_NONE)
    return;
  const bool restricted = method == GENERATIVEQC_METHOD_RHF && system.multiplicity == 1 &&
                          system.atoms.size() == 2 && system.electron_count == 26;
  const bool unrestricted = method == GENERATIVEQC_METHOD_UHF && system.multiplicity == 2;
  bool supported = (restricted || unrestricted) && execution.cuda_requested() &&
                   options.density_fitting_mode == GENERATIVEQC_DENSITY_FITTING_CUDA &&
                   options.precision_mode == GENERATIVEQC_PRECISION_FP64 && auxiliary &&
                   system.charge == 0 &&
                   system.basis_representation == core::kSphericalBasis;
  unsigned halogens = 0;
  bool bromine = false;
  for (const auto& atom : system.atoms) {
    if (atom.atomic_number == 1 && atom.ecp_core == 0) continue;
    ++halogens;
    bool br = atom.atomic_number == 35 && atom.ecp_core == 10;
    supported &= br || (atom.atomic_number == 53 && atom.ecp_core == 28);
    bromine |= br;
  }
  const unsigned limit =
      restricted || scf::cuda_execution::fitted_halogen_dihydride_doublet(system) ? 4U : 3U;
  if (!supported || halogens != 1 ||
      std::any_of(system.shells.begin(), system.shells.end(),
                  [limit](const auto& s) { return s.angular_momentum > limit; }) ||
      std::any_of(auxiliary->shells.begin(), auxiliary->shells.end(),
                  [](const auto& s) { return s.angular_momentum > 5; }))
    throw MethodError(GENERATIVEQC_STATUS_NOT_IMPLEMENTED,
                      "ECP density-fitting execution is not yet validated for this reference domain");
  system.refine_ecp_quadrature = bromine;
}

}  // namespace generativeqc::methods::detail
