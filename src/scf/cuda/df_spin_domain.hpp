#pragma once

#include <stdexcept>

#include "core/types.hpp"
#include "molecule/small_core_halide.hpp"
#include "scf/fock_build.hpp"

namespace generativeqc::scf::cuda_execution {

inline molecule::SmallCoreHalideComposition halide_composition(const core::System& orbital) {
  molecule::SmallCoreHalideComposition composition;
  for (const auto& atom : orbital.atoms) composition.add(atom.atomic_number, atom.ecp_core);
  return composition;
}

inline bool fitted_halogen_dihydride_doublet(const core::System& orbital) {
  if (orbital.atoms.size() != 3 || orbital.charge != 0 || orbital.multiplicity != 2 ||
      orbital.basis_representation != core::kSphericalBasis || orbital.ecp_terms.empty())
    return false;
  return halide_composition(orbital).hydride();
}

inline bool fitted_carbon_halide_doublet(const core::System& orbital) {
  return orbital.multiplicity == 2 && orbital.basis_representation == core::kSphericalBasis &&
         !orbital.ecp_terms.empty() && halide_composition(orbital).carbon_halide_reference(
             orbital.charge, orbital.multiplicity, orbital.electron_count);
}

inline bool all_electron_triplet_hydrogen_dimer(const core::System& orbital) {
  if (orbital.atoms.size() != 2 || orbital.charge != 0 || orbital.multiplicity != 3 ||
      orbital.basis_representation != core::kSphericalBasis || !orbital.ecp_terms.empty())
    return false;
  for (const auto& atom : orbital.atoms)
    if (atom.atomic_number != 1 || atom.ecp_core != 0) return false;
  return true;
}

inline void require_cuda_df_spin_domain(const core::System& orbital, FockSpin spin) {
  if (spin != FockSpin::Unrestricted) return;
  const bool supports_orbital_g = fitted_halogen_dihydride_doublet(orbital) ||
                                  fitted_carbon_halide_doublet(orbital) ||
                                  all_electron_triplet_hydrogen_dimer(orbital);
  const unsigned limit = supports_orbital_g ? 4U : 3U;
  for (const auto& shell : orbital.shells)
    if (shell.angular_momentum > limit)
      throw std::invalid_argument(
          limit == 3U ? "unrestricted CUDA DF execution supports orbital shells only through f (l <= 3)"
                      : "this fitted CUDA DF-UHF spin domain supports orbital shells through g");
}

}  // namespace generativeqc::scf::cuda_execution
