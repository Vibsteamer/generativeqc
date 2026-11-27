#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "generativeqc/ucc_fitting.h"
#include "integrals/s_integrals.hpp"
#include "integrals/x2c1e.hpp"
#include "molecule/basis.hpp"

namespace {
double resolve_speed_of_light(double c) {
  if (c == 0.0) return generativeqc::integrals::kSpeedOfLightAu;
  if (!std::isfinite(c) || c < 0.0)
    throw std::invalid_argument(
        "X2C-1e speed of light must be finite and positive, or 0 for the CODATA value");
  return c;
}

generativeqc::integrals::X2c1eMode resolve_x2c1e_mode(uint32_t mode) {
  if (mode > 3U) throw std::invalid_argument("unsupported X2C-1e mode");
  return static_cast<generativeqc::integrals::X2c1eMode>(mode);
}
}  // namespace

extern "C" {

generativeqc_status generativeqc_system_set_x2c1e_v1(generativeqc_context* context,
                                                     generativeqc_system* system, uint32_t mode,
                                                     double speed_of_light) {
  if (!context || !system) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    const auto resolved_mode = resolve_x2c1e_mode(mode);
    const double c = resolve_speed_of_light(speed_of_light);
    if (resolved_mode == generativeqc::integrals::X2c1eMode::Global1e) {
      const bool has_ecp = !system->data.ecp_terms.empty() ||
                           std::any_of(system->data.atoms.begin(), system->data.atoms.end(),
                                       [](const auto& a) { return a.ecp_core > 0; });
      if (has_ecp) {
        throw std::invalid_argument(
            "Global X2C-1e is not valid on systems containing ECP atoms; use Dlu1e");
      }
    }
    system->data.x2c1e_mode = static_cast<std::uint32_t>(resolved_mode);
    system->data.x2c1e_speed_of_light = c;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

generativeqc_status generativeqc_system_stvw_integrals_v1(
    generativeqc_context* context, const generativeqc_system* system, double* overlap_out,
    double* kinetic_out, double* attraction_out, double* small_potential_out,
    size_t matrix_count) {
  if (!context || !system) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    const auto n = generativeqc::molecule::ao_count(system->data);
    if (!n || n > std::numeric_limits<std::size_t>::max() / n || matrix_count != n * n)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    const auto stvw = generativeqc::integrals::build_stvw_integrals(system->data);
    if (overlap_out) std::copy(stvw.overlap.begin(), stvw.overlap.end(), overlap_out);
    if (kinetic_out) std::copy(stvw.kinetic.begin(), stvw.kinetic.end(), kinetic_out);
    if (attraction_out) std::copy(stvw.attraction.begin(), stvw.attraction.end(), attraction_out);
    if (small_potential_out)
      std::copy(stvw.small_potential.begin(), stvw.small_potential.end(), small_potential_out);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

generativeqc_status generativeqc_system_x2c1e_hcore_v1(
    generativeqc_context* context, const generativeqc_system* system, uint32_t mode,
    double speed_of_light, int delta_only, double* hcore_out, size_t matrix_count) {
  if (!context || !system || !hcore_out) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    const auto n = generativeqc::molecule::ao_count(system->data);
    if (!n || n > std::numeric_limits<std::size_t>::max() / n || matrix_count != n * n)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    const auto state = generativeqc::integrals::build_x2c1e_state(
        system->data, resolve_x2c1e_mode(mode), resolve_speed_of_light(speed_of_light));
    const auto& src = delta_only ? state->hcore_delta : state->hcore_no_ecp;
    std::copy(src.begin(), src.end(), hcore_out);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

generativeqc_status generativeqc_system_x2c1e_hcore_derivatives_v1(
    generativeqc_context* context, const generativeqc_system* system, uint32_t mode,
    double speed_of_light, int delta_only, double* dhcore_out, size_t count) {
  if (!context || !system || !dhcore_out) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    const auto n = generativeqc::molecule::ao_count(system->data);
    const auto ncoord = system->data.atoms.size() * 3U;
    if (!n || n > std::numeric_limits<std::size_t>::max() / n ||
        (ncoord && n * n > std::numeric_limits<std::size_t>::max() / ncoord) ||
        count != ncoord * n * n)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    const auto state = generativeqc::integrals::build_x2c1e_state(
        system->data, resolve_x2c1e_mode(mode), resolve_speed_of_light(speed_of_light));
    const auto derivs =
        generativeqc::integrals::build_x2c1e_hcore_derivatives(*state, delta_only != 0);
    std::copy(derivs.begin(), derivs.end(), dhcore_out);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

generativeqc_status generativeqc_system_x2c1e_weighted_gradient_v1(
    generativeqc_context* context, const generativeqc_system* system, uint32_t mode,
    double speed_of_light, int delta_only, const double* hcore_weights, size_t matrix_count,
    double* gradient_out, size_t gradient_count) {
  if (!context || !system || !hcore_weights || !gradient_out)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    const auto n = generativeqc::molecule::ao_count(system->data);
    const auto ncoord = system->data.atoms.size() * 3U;
    if (!n || n > std::numeric_limits<std::size_t>::max() / n || matrix_count != n * n ||
        gradient_count != ncoord)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    const auto state = generativeqc::integrals::build_x2c1e_state(
        system->data, resolve_x2c1e_mode(mode), resolve_speed_of_light(speed_of_light));
    const auto grad = generativeqc::integrals::contract_weighted_x2c1e_hcore_derivative(
        *state, std::span<const double>(hcore_weights, matrix_count), delta_only != 0);
    std::copy(grad.begin(), grad.end(), gradient_out);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

generativeqc_status generativeqc_system_one_electron_gradient_cpu_v1(
    generativeqc_context* context, const generativeqc_system* system, const double* overlap_weights,
    const double* hcore_weights, size_t matrix_count, int include_nuclear_repulsion,
    double* gradient_out, size_t gradient_count) {
  if (!context || !system || !gradient_out) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    const auto n = generativeqc::molecule::ao_count(system->data);
    const auto ncoord = system->data.atoms.size() * 3U;
    if (!n || n > std::numeric_limits<std::size_t>::max() / n || matrix_count != n * n ||
        gradient_count != ncoord)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    std::vector<double> zero_buf;
    auto span_or_zero = [&](const double* ptr) -> std::span<const double> {
      if (ptr) return std::span<const double>(ptr, matrix_count);
      if (zero_buf.empty()) zero_buf.assign(matrix_count, 0.0);
      return std::span<const double>(zero_buf);
    };
    const auto grad = generativeqc::integrals::contract_weighted_one_electron_fast(
        system->data, span_or_zero(overlap_weights), span_or_zero(hcore_weights),
        include_nuclear_repulsion != 0);
    std::copy(grad.begin(), grad.end(), gradient_out);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

generativeqc_status generativeqc_system_df_integrals_cpu_v1(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* auxiliary, double* metric_out, size_t metric_count,
    double* three_center_out, size_t three_center_count) {
  if (!context || !orbital || !auxiliary || !metric_out || !three_center_out)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    const auto n = generativeqc::molecule::ao_count(orbital->data);
    const auto a = generativeqc::molecule::ao_count(auxiliary->data);
    const auto max_sz = std::numeric_limits<std::size_t>::max();
    if (!n || !a || a > max_sz / a || n > max_sz / n || n * n > max_sz / a ||
        metric_count != a * a || three_center_count != n * n * a)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    generativeqc::integrals::build_df_integrals_fast(
        orbital->data, auxiliary->data, std::span<double>(metric_out, metric_count),
        std::span<double>(three_center_out, three_center_count));
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

generativeqc_status generativeqc_system_df_gradient_cpu_v1(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* auxiliary, const double* metric_weights, size_t metric_count,
    const double* three_center_weights, size_t three_center_count, double* gradient_out,
    size_t gradient_count) {
  if (!context || !orbital || !auxiliary || !metric_weights || !three_center_weights ||
      !gradient_out)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    const auto n = generativeqc::molecule::ao_count(orbital->data);
    const auto a = generativeqc::molecule::ao_count(auxiliary->data);
    const auto ncoord = orbital->data.atoms.size() * 3U;
    const auto max_sz = std::numeric_limits<std::size_t>::max();
    if (!n || !a || a > max_sz / a || n > max_sz / n || n * n > max_sz / a ||
        metric_count != a * a || three_center_count != n * n * a || gradient_count != ncoord)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    const auto grad = generativeqc::integrals::contract_weighted_df_derivative_fast(
        orbital->data, auxiliary->data, std::span<const double>(metric_weights, metric_count),
        std::span<const double>(three_center_weights, three_center_count));
    std::copy(grad.begin(), grad.end(), gradient_out);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

generativeqc_status generativeqc_system_one_electron_integrals_cpu_v1(
    generativeqc_context* context, const generativeqc_system* system, double* overlap_out,
    double* hcore_out, size_t matrix_count, double* nuclear_repulsion_out) {
  if (!context || !system) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    const auto n = generativeqc::molecule::ao_count(system->data);
    if (!n || n > std::numeric_limits<std::size_t>::max() / n || matrix_count != n * n)
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    double enuc = 0.0;
    generativeqc::integrals::build_one_electron_values_fast(
        system->data,
        overlap_out ? std::span<double>(overlap_out, matrix_count) : std::span<double>{},
        hcore_out ? std::span<double>(hcore_out, matrix_count) : std::span<double>{}, enuc);
    if (nuclear_repulsion_out) *nuclear_repulsion_out = enuc;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

}  // extern "C"
