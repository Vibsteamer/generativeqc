// Molecular F12 qualification bridge: the same native CUDA one-electron value
// provider and Cartesian-to-public transform as the prepared RI-JK reference.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>

#include "api/handles.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"

extern "C" int gqc_f12_molecular_one_electron_forward(generativeqc_context* context,
                                                      const generativeqc_system* system,
                                                      std::size_t count, std::size_t maximum_bytes,
                                                      double* overlap, double* hcore,
                                                      std::uint64_t* metrics, char* error,
                                                      std::size_t error_capacity) {
  using namespace generativeqc;
  auto fail = [&](const char* message, int status) {
    if (error && error_capacity) std::snprintf(error, error_capacity, "%s", message);
    return status;
  };
  if (!context || !system || !count || !maximum_bytes || !overlap || !hcore || overlap == hcore ||
      !metrics || !error || !error_capacity) {
    return fail("missing or aliased one-electron argument", GENERATIVEQC_STATUS_INVALID_ARGUMENT);
  }
  try {
    std::lock_guard<std::recursive_mutex> lock(context->mutex);
    if (context->state.executed_backend != GENERATIVEQC_BACKEND_CUDA ||
        context->state.device_id < 0) {
      return fail("one-electron forward requires an active CUDA context",
                  GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
    }
    const auto& mol = system->data;
    if (system->correlation_auxiliary || system->ks_auxiliary || system->scf_auxiliary ||
        mol.atoms.empty() || mol.shells.empty() || !mol.ecp_terms.empty() ||
        mol.basis_representation != core::kSphericalBasis) {
      return fail("F12 combined basis requires real spherical all-electron orbital shells",
                  GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    }
    for (const auto& shell : mol.shells)
      if (shell.angular_momentum > 4)
        return fail("one-electron CUDA values support s through g",
                    GENERATIVEQC_STATUS_NOT_IMPLEMENTED);
    const auto n = molecule::ao_count(mol), cart_n = molecule::cartesian_ao_count(mol);
    if (!n || n > std::numeric_limits<std::size_t>::max() / n || count != n * n || !cart_n) {
      return fail("invalid one-electron matrix size", GENERATIVEQC_STATUS_INVALID_ARGUMENT);
    }
    if (count > maximum_bytes / (2 * sizeof(double)) ||
        cart_n > maximum_bytes / (2 * sizeof(double)) / cart_n) {
      return fail("combined one-electron matrices exceed staging budget",
                  GENERATIVEQC_STATUS_OUT_OF_MEMORY);
    }
    auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
    ledger->device = context->state.device_id;
    ledger->limit = maximum_bytes;
    ledger->admission = runtime::DeviceResourceAdmission::NumericCap;
    integrals::IntegralData cart;
    std::string detail;
    {
      runtime::DeviceResourceScope scope(ledger);
      const auto status =
          scf::build_cuda_one_electron_integrals(ledger->device, mol, cart, detail, false, false);
      if (status != GENERATIVEQC_STATUS_SUCCESS) return fail(detail.c_str(), status);
    }
    if (!ledger->peak || !ledger->allocations || ledger->live || ledger->peak > maximum_bytes) {
      return fail("CUDA one-electron resource ledger failed to close",
                  GENERATIVEQC_STATUS_INTERNAL_ERROR);
    }
    auto result = integrals::transform_integrals(cart, mol);
    const auto finite = [](double x) { return std::isfinite(x); };
    if (result.nbf != n || result.overlap.size() != count || result.hcore.size() != count ||
        !std::all_of(result.overlap.begin(), result.overlap.end(), finite) ||
        !std::all_of(result.hcore.begin(), result.hcore.end(), finite)) {
      return fail("invalid native overlap/Hcore", GENERATIVEQC_STATUS_INTERNAL_ERROR);
    }
    std::copy_n(result.overlap.data(), count, overlap);
    std::copy_n(result.hcore.data(), count, hcore);
    const std::uint64_t info[] = {ledger->peak, ledger->allocations, ledger->requested_bytes,
                                  cart_n, n};
    std::copy_n(info, 5, metrics);
    error[0] = '\0';
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    return fail("one-electron forward allocation failed", GENERATIVEQC_STATUS_OUT_OF_MEMORY);
  } catch (const std::exception& exc) {
    return fail(exc.what(), GENERATIVEQC_STATUS_INTERNAL_ERROR);
  } catch (...) {
    return fail("one-electron forward failed", GENERATIVEQC_STATUS_INTERNAL_ERROR);
  }
}
