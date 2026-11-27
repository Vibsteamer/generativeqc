#include "integrals/ecp.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

#include "generated_ecp_ao.cuh"
#include "integrals/ecp_gaussian.hpp"
#include "integrals/ecp_workers.hpp"
#include "molecule/basis.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_usage.hpp"

namespace generativeqc::integrals {
namespace {
constexpr int projector_count = generated::ecp_projector_count;
struct Component {
  unsigned x, y, z;
  double coefficient;
};
struct AO {
  const core::Shell* shell;
  int primitive_offset{}, primitive_count{}, term_count{};
  double x, y, z;
  Component components[molecule::kMaximumAuxiliaryAoExpansionTerms];
};
struct Jet {
  double v[4];
};

unsigned resolve_ecp_worker_count(std::size_t radial_count) noexcept {
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) hw = 4;
  unsigned nthreads = std::min<unsigned>(hw, 8U);
  if (runtime::cpu_resource_observation.active &&
      runtime::cpu_resource_observation.cpu_worker_limit > 0) {
    nthreads = std::min(nthreads, runtime::cpu_resource_observation.cpu_worker_limit);
  }
  if (radial_count == 0) return 1U;
  return std::max<unsigned>(
      1U, static_cast<unsigned>(std::min<std::size_t>(nthreads, radial_count)));
}

std::size_t ecp_worker_scratch_bytes(std::size_t n, std::size_t sphere_size, unsigned nthreads) {
  const std::size_t values_bytes =
      runtime::size_mul(runtime::size_mul(n, sphere_size), sizeof(Jet));
  const std::size_t projections_bytes =
      runtime::size_mul(runtime::size_mul(n, static_cast<std::size_t>(projector_count)),
                        sizeof(Jet));
  const std::size_t active_bytes = runtime::size_mul(n, sizeof(std::size_t));
  const std::size_t per_worker = runtime::add_capacity(
      runtime::add_capacity(values_bytes, projections_bytes), active_bytes);
  return runtime::size_mul(static_cast<std::size_t>(nthreads), per_worker);
}
}  // namespace

void ecp_quadrature(unsigned radial, unsigned angular, std::vector<EcpRadialPoint>& radii,
                    std::vector<EcpSpherePoint>& sphere) {
  generated::ecp_make_grid(radial, angular, radii, sphere);
}

EcpData ecp_integrals(const core::System& system, unsigned radial, unsigned angular,
                      bool derivatives) {
  if (radial < 16 || radial > generated::ecp_cpu_max_radial_points || angular < 8 ||
      angular > generated::ecp_cpu_max_polar_points)
    throw std::invalid_argument(
        "CPU ECP quadrature requires 16..1536 radial and 8..192 polar points");
  const auto n = molecule::ao_count(system);
  const auto size = runtime::size_mul(n, n);
  const auto ncoord = derivatives ? runtime::size_mul(3, system.atoms.size()) : 0;
  const auto derivative_elements = runtime::size_mul(ncoord, size);
  std::vector<AO> aos;
  for (const auto& shell : system.shells) {
    if (shell.angular_momentum > generated::ecp_max_orbital_angular)
      throw std::invalid_argument("ECP AO shell exceeds the generated quadrature domain");
    if (shell.primitives.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("ECP primitive count exceeds the generated index type");
    const auto& position = system.atoms[shell.atom_index].position;
    const auto expansions =
        molecule::ao_expansions(shell.angular_momentum, system.basis_representation);
    if (std::any_of(expansions.begin(), expansions.end(),
                    [](const auto& expansion) {
                      return expansion.size() > molecule::kMaximumAuxiliaryAoExpansionTerms;
                    }))
      throw std::invalid_argument("ECP AO expansion exceeds validated domain");
    for (const auto& expansion : expansions) {
      AO ao{&shell,
            0,
            static_cast<int>(shell.primitives.size()),
            static_cast<int>(expansion.size()),
            position[0],
            position[1],
            position[2],
            {}};
      for (unsigned t = 0; t < expansion.size(); ++t) {
        const auto& v = expansion[t];
        ao.components[t] = {v.component[0], v.component[1], v.component[2],
                            generated::ecp_component_coefficient(v.component[0], v.component[1],
                                                                 v.component[2], v.coefficient)};
      }
      aos.push_back(ao);
    }
  }
  EcpData out{n, ncoord, std::vector<double>(size), std::vector<double>(size), {}, {}};
  out.local_derivative.resize(derivative_elements);
  out.nonlocal_derivative.resize(derivative_elements);
  if (system.ecp_terms.empty()) return out;
  std::vector<EcpRadialPoint> radii;
  std::vector<EcpSpherePoint> sphere;
  ecp_quadrature(radial, angular, radii, sphere);
  // A radial layer is still the lifetime boundary. Share its potentials and
  // projected AO jets across every pair, independently of radial grid length.
  if (!derivatives) {
    const unsigned nthreads = resolve_ecp_worker_count(radii.size());
    std::vector<std::vector<double>> loc_buf(nthreads, std::vector<double>(size, 0.0));
    std::vector<std::vector<double>> nloc_buf(nthreads, std::vector<double>(size, 0.0));
    std::size_t observed_bytes = runtime::vector_capacities(
        out.local, out.nonlocal, out.local_derivative, out.nonlocal_derivative, aos, radii, sphere);
    for (unsigned t = 0; t < nthreads; ++t) {
      observed_bytes = runtime::add_capacity(
          observed_bytes, runtime::vector_capacities(loc_buf[t], nloc_buf[t]));
    }
    observed_bytes = runtime::add_capacity(
        observed_bytes, ecp_worker_scratch_bytes(n, sphere.size(), nthreads));
    runtime::sample_cpu_capacity(observed_bytes);

    for (std::size_t center = 0; center < system.atoms.size(); ++center) {
      if (!system.atoms[center].ecp_core) continue;
      const auto& c = system.atoms[center].position;
      auto worker = [&](unsigned tid, const std::atomic<bool>& cancelled) {
        const auto [r_begin, r_end] =
            detail::static_radial_partition(radii.size(), nthreads, tid);
        if (r_begin >= r_end) return;
        std::vector<Jet> values(runtime::size_mul(n, sphere.size()));
        std::vector<Jet> projections(runtime::size_mul(n, projector_count));
        std::vector<std::size_t> active_aos;
        active_aos.reserve(n);
        auto& loc = loc_buf[tid];
        auto& nloc = nloc_buf[tid];
        for (std::size_t ridx = r_begin; ridx < r_end; ++ridx) {
          if (cancelled.load(std::memory_order_acquire)) break;
          const auto& radial_point = radii[ridx];
          double potentials[generated::ecp_max_projector_angular + 2];
          generated::ecp_potentials(system.ecp_terms.data(), system.ecp_terms.size(), radial_point,
                                    center, potentials);
          if (std::all_of(std::begin(potentials), std::end(potentials),
                          [](double v) { return v == 0; }))
            continue;
          active_aos.clear();
          for (std::size_t a = 0; a < n; ++a) {
            const double dx = aos[a].x - c[0];
            const double dy = aos[a].y - c[1];
            const double dz = aos[a].z - c[2];
            const double d_ac = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double min_dist = std::max(0.0, std::abs(d_ac - radial_point.r));
            const double d2 = min_dist * min_dist;
            bool nonzero = false;
            const auto* prims = aos[a].shell->primitives.data();
            for (int p = 0; p < aos[a].primitive_count; ++p) {
              if (prims[p].exponent * d2 <= 50.0) {
                nonzero = true;
                break;
              }
            }
            if (!nonzero) continue;
            active_aos.push_back(a);
            for (std::size_t q = 0; q < sphere.size(); ++q)
              values[a * sphere.size() + q] = generated::ecp_evaluate_ao<Jet>(
                  aos[a], prims, sphere[q], radial_point.r, c[0], c[1], c[2], false);
            for (int m = 0; m < projector_count; ++m)
              projections[a * projector_count + m] = generated::ecp_project(
                  values.data() + a * sphere.size(), sphere.data(), sphere.size(), m, false);
          }
          for (std::size_t ia = 0; ia < active_aos.size(); ++ia) {
            const std::size_t a = active_aos[ia];
            for (std::size_t ib = 0; ib <= ia; ++ib) {
              const std::size_t b = active_aos[ib];
              double parts[2][10];
              generated::ecp_contract_potentials(
                  potentials, sphere.data(), sphere.size(), values.data() + a * sphere.size(),
                  values.data() + b * sphere.size(), projections.data() + a * projector_count,
                  projections.data() + b * projector_count, false, parts);
              loc[a * n + b] += parts[0][0];
              nloc[a * n + b] += parts[1][0];
              if (a != b) {
                loc[b * n + a] += parts[0][0];
                nloc[b * n + a] += parts[1][0];
              }
            }
          }
        }
      };
      detail::run_ecp_workers(nthreads, worker);
    }
    for (unsigned t = 0; t < nthreads; ++t) {
      for (std::size_t k = 0; k < size; ++k) {
        out.local[k] += loc_buf[t][k];
        out.nonlocal[k] += nloc_buf[t][k];
      }
    }
    return out;
  }

  std::vector<Jet> values(runtime::size_mul(n, sphere.size())),
      projections(runtime::size_mul(n, projector_count));
  runtime::sample_cpu_capacity(
      runtime::vector_capacities(out.local, out.nonlocal, out.local_derivative,
                                 out.nonlocal_derivative, aos, radii, sphere, values, projections));
  for (std::size_t center = 0; center < system.atoms.size(); ++center) {
    if (!system.atoms[center].ecp_core) continue;
    const auto& c = system.atoms[center].position;
    for (const auto& radial_point : radii) {
      double potentials[generated::ecp_max_projector_angular + 2];
      generated::ecp_potentials(system.ecp_terms.data(), system.ecp_terms.size(), radial_point,
                                center, potentials);
      if (std::all_of(std::begin(potentials), std::end(potentials),
                      [](double v) { return v == 0; }))
        continue;
      for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t q = 0; q < sphere.size(); ++q)
          values[a * sphere.size() + q] =
              generated::ecp_evaluate_ao<Jet>(aos[a], aos[a].shell->primitives.data(), sphere[q],
                                              radial_point.r, c[0], c[1], c[2], derivatives);
        for (int m = 0; m < projector_count; ++m)
          projections[a * projector_count + m] = generated::ecp_project(
              values.data() + a * sphere.size(), sphere.data(), sphere.size(), m, derivatives);
      }
      for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b <= a; ++b) {
          double parts[2][10];
          generated::ecp_contract_potentials(
              potentials, sphere.data(), sphere.size(), values.data() + a * sphere.size(),
              values.data() + b * sphere.size(), projections.data() + a * projector_count,
              projections.data() + b * projector_count, derivatives, parts);
          for (unsigned part = 0; part < 2; ++part) {
            const auto& v = parts[part];
            auto& matrix = part == 0 ? out.local : out.nonlocal;
            auto& gradient = part == 0 ? out.local_derivative : out.nonlocal_derivative;
            for (unsigned transpose = 0; transpose < (a == b ? 1U : 2U); ++transpose) {
              const auto item = transpose ? b * n + a : a * n + b;
              matrix[item] += v[0];
              if (derivatives)
                for (unsigned d = 0; d < 3; ++d) {
                  gradient[(aos[a].shell->atom_index * 3 + d) * size + item] += v[d + 1];
                  gradient[(aos[b].shell->atom_index * 3 + d) * size + item] += v[d + 4];
                  gradient[(center * 3 + d) * size + item] += v[d + 7];
                }
            }
          }
        }
    }
  }
  return out;
}

EcpData checked_ecp_integrals(const core::System& system, bool derivatives) {
  if (gaussian_ecp_supported(system)) return gaussian_ecp_integrals(system, derivatives);
  auto accepted = [](const EcpData& a, const EcpData& b) {
    auto check_vec = [](const std::vector<double>& x, const std::vector<double>& y,
                        bool derivative) {
      for (std::size_t i = 0; i < x.size(); ++i)
        if (!generated::ecp_grid_pair_accepted(x[i], y[i], derivative)) return false;
      return true;
    };
    return check_vec(a.local, b.local, false) && check_vec(a.nonlocal, b.nonlocal, false) &&
           check_vec(a.local_derivative, b.local_derivative, true) &&
           check_vec(a.nonlocal_derivative, b.nonlocal_derivative, true);
  };

  auto coarse = ecp_integrals(system, generated::ecp_coarse_radial_points,
                              generated::ecp_coarse_polar_points, derivatives);
  runtime::CpuRetainedCapacity retained(runtime::vector_capacities(
      coarse.local, coarse.nonlocal, coarse.local_derivative, coarse.nonlocal_derivative));
  auto fine = ecp_integrals(system, generated::ecp_refined_radial_points,
                            generated::ecp_refined_polar_points, derivatives);
  if (!system.refine_ecp_quadrature && accepted(coarse, fine)) return fine;
  coarse = std::move(fine);
  fine = ecp_integrals(system, generated::ecp_extended_radial_points,
                       generated::ecp_extended_polar_points, derivatives);
  if (accepted(coarse, fine)) return fine;
  coarse = std::move(fine);
  fine = ecp_integrals(system, generated::ecp_max_cuda_radial_points,
                       generated::ecp_max_cuda_polar_points, derivatives);
  if (accepted(coarse, fine)) return fine;
  throw std::runtime_error(
      "ECP quadrature convergence gate failed; this input is outside the validated execution "
      "domain");
}

std::vector<double> contract_weighted_ecp_derivative(const core::System& system,
                                                     std::span<const double> hcore_weights) {
  const auto n = molecule::ao_count(system);
  const auto size = runtime::size_mul(n, n);
  const auto ncoord = runtime::size_mul(3, system.atoms.size());
  if (hcore_weights.size() != size)
    throw std::invalid_argument("weighted ECP derivative weight size mismatch");
  std::vector<double> result(ncoord, 0.0);
  if (system.ecp_terms.empty() ||
      std::none_of(hcore_weights.begin(), hcore_weights.end(), [](double x) { return x != 0.0; }))
    return result;
  if (gaussian_ecp_supported(system)) {
    const auto ecp = gaussian_ecp_integrals(system, true);
    for (std::size_t coord = 0; coord < ncoord; ++coord)
      for (std::size_t item = 0; item < size; ++item)
        result[coord] += hcore_weights[item] * (ecp.local_derivative[coord * size + item] +
                                                ecp.nonlocal_derivative[coord * size + item]);
    return result;
  }

  std::vector<AO> aos;
  aos.reserve(n);
  for (const auto& shell : system.shells) {
    if (shell.angular_momentum > generated::ecp_max_orbital_angular)
      throw std::invalid_argument("ECP AO shell exceeds the generated quadrature domain");
    const auto& position = system.atoms[shell.atom_index].position;
    const auto expansions =
        molecule::ao_expansions(shell.angular_momentum, system.basis_representation);
    for (const auto& expansion : expansions) {
      AO ao{&shell,
            0,
            static_cast<int>(shell.primitives.size()),
            static_cast<int>(expansion.size()),
            position[0],
            position[1],
            position[2],
            {}};
      for (unsigned t = 0; t < expansion.size(); ++t) {
        const auto& v = expansion[t];
        ao.components[t] = {v.component[0], v.component[1], v.component[2],
                            generated::ecp_component_coefficient(v.component[0], v.component[1],
                                                                 v.component[2], v.coefficient)};
      }
      aos.push_back(ao);
    }
  }

  auto eval_grid = [&](unsigned radial, unsigned angular) -> std::vector<double> {
    std::vector<EcpRadialPoint> radii;
    std::vector<EcpSpherePoint> sphere;
    ecp_quadrature(radial, angular, radii, sphere);
    const unsigned nthreads = resolve_ecp_worker_count(radii.size());
    std::vector<std::vector<double>> tgrad(nthreads, std::vector<double>(ncoord, 0.0));
    std::size_t observed_bytes = runtime::vector_capacities(result, aos, radii, sphere);
    for (unsigned t = 0; t < nthreads; ++t) {
      observed_bytes = runtime::add_capacity(observed_bytes, runtime::vector_bytes(tgrad[t]));
    }
    observed_bytes = runtime::add_capacity(
        observed_bytes, ecp_worker_scratch_bytes(n, sphere.size(), nthreads));
    runtime::sample_cpu_capacity(observed_bytes);

    for (std::size_t center = 0; center < system.atoms.size(); ++center) {
      if (!system.atoms[center].ecp_core) continue;
      const auto& c = system.atoms[center].position;
      auto worker = [&](unsigned tid, const std::atomic<bool>& cancelled) {
        const auto [r_begin, r_end] =
            detail::static_radial_partition(radii.size(), nthreads, tid);
        if (r_begin >= r_end) return;
        std::vector<Jet> values(runtime::size_mul(n, sphere.size()));
        std::vector<Jet> projections(runtime::size_mul(n, projector_count));
        std::vector<std::size_t> active_aos;
        active_aos.reserve(n);
        auto& g = tgrad[tid];
        for (std::size_t ridx = r_begin; ridx < r_end; ++ridx) {
          if (cancelled.load(std::memory_order_acquire)) break;
          const auto& radial_point = radii[ridx];
          double potentials[generated::ecp_max_projector_angular + 2];
          generated::ecp_potentials(system.ecp_terms.data(), system.ecp_terms.size(), radial_point,
                                    center, potentials);
          if (std::all_of(std::begin(potentials), std::end(potentials),
                          [](double v) { return v == 0; }))
            continue;
          active_aos.clear();
          for (std::size_t a = 0; a < n; ++a) {
            const double dx = aos[a].x - c[0];
            const double dy = aos[a].y - c[1];
            const double dz = aos[a].z - c[2];
            const double d_ac = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double min_dist = std::max(0.0, std::abs(d_ac - radial_point.r));
            const double d2 = min_dist * min_dist;
            bool nonzero = false;
            const auto* prims = aos[a].shell->primitives.data();
            for (int p = 0; p < aos[a].primitive_count; ++p) {
              if (prims[p].exponent * d2 <= 50.0) {
                nonzero = true;
                break;
              }
            }
            if (!nonzero) continue;
            active_aos.push_back(a);
            for (std::size_t q = 0; q < sphere.size(); ++q)
              values[a * sphere.size() + q] = generated::ecp_evaluate_ao<Jet>(
                  aos[a], prims, sphere[q], radial_point.r, c[0], c[1], c[2], true);
            for (int m = 0; m < projector_count; ++m)
              projections[a * projector_count + m] = generated::ecp_project(
                  values.data() + a * sphere.size(), sphere.data(), sphere.size(), m, true);
          }
          for (std::size_t ia = 0; ia < active_aos.size(); ++ia) {
            const std::size_t a = active_aos[ia];
            const std::size_t atom_a = aos[a].shell->atom_index;
            for (std::size_t ib = 0; ib <= ia; ++ib) {
              const std::size_t b = active_aos[ib];
              const double w =
                  (a == b) ? hcore_weights[a * n + b]
                           : (hcore_weights[a * n + b] + hcore_weights[b * n + a]);
              if (w == 0.0) continue;
              const std::size_t atom_b = aos[b].shell->atom_index;
              if (atom_a == center && atom_b == center) continue;
              double parts[2][10];
              generated::ecp_contract_potentials(
                  potentials, sphere.data(), sphere.size(), values.data() + a * sphere.size(),
                  values.data() + b * sphere.size(), projections.data() + a * projector_count,
                  projections.data() + b * projector_count, true, parts);
              for (unsigned d = 0; d < 3; ++d) {
                const double da = parts[0][d + 1] + parts[1][d + 1];
                const double db = parts[0][d + 4] + parts[1][d + 4];
                const double dc = parts[0][d + 7] + parts[1][d + 7];
                g[atom_a * 3 + d] += w * da;
                g[atom_b * 3 + d] += w * db;
                g[center * 3 + d] += w * dc;
              }
            }
          }
        }
      };
      detail::run_ecp_workers(nthreads, worker);
    }
    std::vector<double> out(ncoord, 0.0);
    for (unsigned t = 0; t < nthreads; ++t)
      for (std::size_t coord = 0; coord < ncoord; ++coord) out[coord] += tgrad[t][coord];
    return out;
  };

  double max_w = 0.0;
  for (double x : hcore_weights) max_w = std::max(max_w, std::abs(x));
  const double inv_scale = max_w > 0.0 ? (1.0 / max_w) : 1.0;
  auto accepted_grad = [inv_scale](const std::vector<double>& a, const std::vector<double>& b) {
    for (std::size_t i = 0; i < a.size(); ++i)
      if (!generated::ecp_grid_pair_accepted(a[i] * inv_scale, b[i] * inv_scale, true))
        return false;
    return true;
  };
  auto coarse = eval_grid(generated::ecp_coarse_radial_points, generated::ecp_coarse_polar_points);
  auto fine = eval_grid(generated::ecp_refined_radial_points, generated::ecp_refined_polar_points);
  if (!system.refine_ecp_quadrature && accepted_grad(coarse, fine)) return fine;
  coarse = std::move(fine);
  fine = eval_grid(generated::ecp_extended_radial_points, generated::ecp_extended_polar_points);
  if (accepted_grad(coarse, fine)) return fine;
  coarse = std::move(fine);
  fine = eval_grid(generated::ecp_max_cuda_radial_points, generated::ecp_max_cuda_polar_points);
  if (accepted_grad(coarse, fine)) return fine;
  throw std::runtime_error(
      "ECP quadrature derivative convergence gate failed; input outside validated domain");
}

EcpGridShape checked_ecp_grid_limit(const core::System& system) {
  if (!system.refine_ecp_quadrature)
    return {generated::ecp_refined_radial_points, generated::ecp_refined_polar_points};
  const bool orbital_g = std::any_of(system.shells.begin(), system.shells.end(),
                                     [](const auto& shell) { return shell.angular_momentum == 4; });
  if (orbital_g)
    return {generated::ecp_orbital_g_radial_points, generated::ecp_orbital_g_polar_points};
  return {generated::ecp_max_cuda_radial_points, generated::ecp_max_cuda_polar_points};
}

std::size_t ecp_workspace_capacity(const core::System& system) {
  if (system.ecp_terms.empty()) return 0;
  const auto ao_count = molecule::ao_count(system);
  const auto [radial, polar] = checked_ecp_grid_limit(system);
  const auto angular = runtime::size_mul(2 * polar, polar);
  const auto tile = generated::ecp_cuda_radial_tile(ao_count, polar);
  const auto coordinates = runtime::size_add(1, runtime::size_mul(3, system.atoms.size()));
  // Both checked grids can coexist with public value/coordinate output panels.
  auto bytes = runtime::size_mul(
      8 * sizeof(double), runtime::size_mul(runtime::size_mul(ao_count, ao_count), coordinates));
  // Four-component values/projections include simultaneous host/device staging.
  bytes = runtime::size_add(
      bytes,
      runtime::size_mul(
          64, runtime::size_mul(runtime::size_mul(tile, ao_count),
                                runtime::size_add(angular, generated::ecp_projector_count))));
  // Grid/metadata bounds include upload copies and geometric vector growth.
  bytes = runtime::size_add(bytes, runtime::size_mul(4 * sizeof(EcpSpherePoint), angular));
  bytes = runtime::size_add(
      bytes, runtime::size_mul(4 * sizeof(EcpRadialPoint), radial));
  bytes = runtime::size_add(bytes, runtime::size_mul(4 * sizeof(AO), ao_count));
  bytes = runtime::size_add(bytes,
                            runtime::size_mul(4 * sizeof(core::EcpTerm), system.ecp_terms.size()));
  for (const auto& shell : system.shells)
    bytes = runtime::size_add(bytes, runtime::size_mul(64, shell.primitives.size()));
  return gaussian_ecp_supported(system)
             ? std::max(bytes, gaussian_ecp_workspace_capacity(system, true))
             : bytes;
}

void add_ecp(const EcpData& ecp, std::vector<double>& hcore, std::vector<double>& derivative) {
  if (hcore.size() != ecp.local.size() || derivative.size() != ecp.local_derivative.size())
    throw std::invalid_argument("ECP and one-electron matrix layouts differ");
  for (std::size_t i = 0; i < hcore.size(); ++i)
    hcore[i] = generated::ecp_add_operator(hcore[i], ecp.local[i], ecp.nonlocal[i]);
  for (std::size_t i = 0; i < derivative.size(); ++i)
    derivative[i] = generated::ecp_add_operator(derivative[i], ecp.local_derivative[i],
                                                ecp.nonlocal_derivative[i]);
}
}  // namespace generativeqc::integrals
