#pragma once

#include <span>
#include <vector>

#include "core/types.hpp"

namespace generativeqc::integrals {

// Residual Gaussian terms: c r^(power-2) exp(-exponent r^2).
// The Coulomb tail -Zeff/r is evaluated by nuclear attraction separately.
struct EcpData {
  std::size_t nbf{}, ncoord{};
  std::vector<double> local, nonlocal, local_derivative, nonlocal_derivative;
};

struct EcpSpherePoint {
  double x, y, z, weight;
  double harmonics[16];
};
// Largest internally admitted checked grid for this normalized system.
struct EcpGridShape {
  unsigned radial, polar;
};
EcpGridShape checked_ecp_grid_limit(const core::System& system);

struct EcpRadialPoint {
  double r, weight;
};
void ecp_quadrature(unsigned radial, unsigned angular, std::vector<EcpRadialPoint>& radii,
                    std::vector<EcpSpherePoint>& sphere);

EcpData ecp_integrals(const core::System& system, unsigned radial = 160, unsigned angular = 32,
                      bool derivatives = true);
EcpData checked_ecp_integrals(const core::System& system, bool derivatives);
/** Directly contract public-AO hcore weights against checked ECP nuclear derivatives. */
std::vector<double> contract_weighted_ecp_derivative(const core::System& system,
                                                     std::span<const double> hcore_weights);
/** Conservative numeric workspace for checked value/first-derivative consumers.
 * Includes both grids, public matrices, radial staging and host/device metadata;
 * callers reserve it in addition to their retained reference/correlation state. */
std::size_t ecp_workspace_capacity(const core::System& system);
void add_ecp(const EcpData& ecp, std::vector<double>& hcore, std::vector<double>& derivative);

}  // namespace generativeqc::integrals
