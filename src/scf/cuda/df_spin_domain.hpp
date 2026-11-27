#pragma once

#include <stdexcept>

#include "core/types.hpp"
#include "scf/fock_build.hpp"

namespace generativeqc::scf::cuda_execution {

inline bool all_electron_triplet_hydrogen_dimer(const core::System& orbital) {
  if (orbital.atoms.size() != 2 || orbital.charge != 0 || orbital.multiplicity != 3 ||
      orbital.basis_representation != core::kSphericalBasis || !orbital.ecp_terms.empty())
    return false;
  for (const auto& atom : orbital.atoms)
    if (atom.atomic_number != 1 || atom.ecp_core != 0) return false;
  return true;
}

inline bool all_electron_spherical_system(const core::System& orbital) {
  if (orbital.atoms.empty() || orbital.basis_representation != core::kSphericalBasis ||
      !orbital.ecp_terms.empty())
    return false;
  for (const auto& atom : orbital.atoms)
    if (atom.ecp_core != 0) return false;
  return true;
}

inline void require_cuda_df_spin_domain(const core::System& orbital, FockSpin spin,
                                        bool full_range_fully_fitted = false) {
  if (spin != FockSpin::Unrestricted) return;
  const bool supports_orbital_g = all_electron_triplet_hydrogen_dimer(orbital) ||
                                  (full_range_fully_fitted &&
                                   all_electron_spherical_system(orbital));
  const unsigned limit = supports_orbital_g ? 4U : 3U;
  for (const auto& shell : orbital.shells)
    if (shell.angular_momentum > limit)
      throw std::invalid_argument(
          limit == 3U ? "unrestricted CUDA DF execution supports orbital shells only through f (l <= 3)"
                      : "this fitted CUDA DF-UHF spin domain supports orbital shells through g");
}

}  // namespace generativeqc::scf::cuda_execution
