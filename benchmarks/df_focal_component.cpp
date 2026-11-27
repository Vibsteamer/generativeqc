// Complete RI-JK/DF-CCSD response for a conventional focal correction.
// The input has geometry and full basis coefficients only. No imported
// electronic state, CPU integral evaluation, or reference-oracle fallback.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "methods/df_ccsdt_force.hpp"
#include "molecule/basis.hpp"
#include "runtime/execution_context.hpp"

namespace {
void shells(std::istream& input, generativeqc::core::System& system, std::size_t count) {
  system.shells.resize(count);
  for (auto& shell : system.shells) {
    std::size_t primitives{};
    input >> shell.atom_index >> shell.angular_momentum >> primitives;
    if (!input || !primitives || primitives > 1000)
      throw std::invalid_argument("invalid focal component shell");
    shell.primitives.resize(primitives);
    for (auto& p : shell.primitives) input >> p.exponent >> p.coefficient;
  }
  if (!input) throw std::invalid_argument("truncated focal component input");
  std::string detail;
  if (generativeqc::molecule::validate_and_normalize(system, detail) != GENERATIVEQC_STATUS_SUCCESS)
    throw std::invalid_argument(detail);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::invalid_argument("usage: df-focal-component INPUT OUTPUT_JSON");
    std::ifstream input(argv[1]);
    std::size_t atoms{}, orbital_shells{}, auxiliary_shells{}, budget{};
    int charge{};
    unsigned multiplicity{};
    input >> atoms >> orbital_shells >> auxiliary_shells >> budget >> charge >> multiplicity;
    if (!input || !atoms || atoms > 1000 || !orbital_shells || orbital_shells > 100000 ||
        !auxiliary_shells || auxiliary_shells > 100000 || !budget || multiplicity != 1)
      throw std::invalid_argument("invalid closed-shell component dimensions or state");
    generativeqc::core::System orbital;
    orbital.charge = charge;
    orbital.multiplicity = multiplicity;
    orbital.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
    orbital.atoms.resize(atoms);
    for (auto& atom : orbital.atoms)
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    auto auxiliary = orbital;
    shells(input, orbital, orbital_shells);
    shells(input, auxiliary, auxiliary_shells);
    input >> std::ws;
    if (!input.eof()) throw std::invalid_argument("unexpected trailing component input");
    if (orbital.electron_count < 2 || orbital.electron_count % 2)
      throw std::invalid_argument("component requires a paired closed-shell reference");

    generativeqc::core::ContextState context;
    context.requested_backend = GENERATIVEQC_BACKEND_CUDA;
    generativeqc::runtime::ExecutionContext execution(context);
    generativeqc_method_descriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = GENERATIVEQC_ABI_VERSION;
    descriptor.method = GENERATIVEQC_METHOD_RCCSD;
    descriptor.precision_mode = GENERATIVEQC_PRECISION_FP64;
    // This is the internal descriptor: the explicit demand below owns RI-JK.
    descriptor.density_fitting_mode = GENERATIVEQC_DENSITY_FITTING_NONE;
    descriptor.max_iterations = 200;
    descriptor.energy_tolerance = 1e-12;
    descriptor.density_tolerance = 1e-11;
    descriptor.ccsd_diis_history = 6;
    descriptor.ccsd_max_iterations = 150;
    descriptor.ccsd_energy_tolerance = 1e-12;
    descriptor.ccsd_residual_tolerance = 1e-10;
    descriptor.correlation_memory_budget_bytes = budget;
    generativeqc::methods::detail::DFCorrelationDemand demand;
    demand.df_reference = true;
    const auto result = generativeqc::methods::detail::run_df_ccsdt_native(
        execution, orbital, auxiliary, descriptor, true, false, true, {}, 32, 0, {}, true, false,
        true, false, true, {}, 30, true, nullptr, true, false, false, false,
        generativeqc::cc::SolverOptions{}.df_mixed_precision_switch_residual, demand);
    if (result.forces.size() != 3 * atoms || result.triples_energy != 0 ||
        !std::isfinite(result.energy) || !std::isfinite(result.reference_energy) ||
        !std::isfinite(result.correlation_energy))
      throw std::runtime_error("incomplete or non-finite CCSD response");
    for (double f : result.forces)
      if (!std::isfinite(f)) throw std::runtime_error("non-finite CCSD force");

    std::ofstream output(argv[2]);
    output << std::setprecision(17) << "{\n  \"energy_Eh\": " << result.energy
           << ",\n  \"reference_energy_Eh\": " << result.reference_energy
           << ",\n  \"correlation_energy_Eh\": " << result.correlation_energy
           << ",\n  \"triples_energy_Eh\": " << result.triples_energy
           << ",\n  \"endpoint_seconds\": " << result.total_seconds
           << ",\n  \"lambda_residual\": " << result.lambda.lambda_residual_norm
           << ",\n  \"lambda_independent_residual\": " << result.lambda.independent_residual_norm
           << ",\n  \"orbital_residual\": " << result.orbital.orbital_residual
           << ",\n  \"maximum_stationarity\": " << result.orbital.maximum_stationarity
           << ",\n  \"numeric_capacity_bytes\": " << result.numeric_capacity_bytes
           << ",\n  \"reference_model\": \"ri-jk\",\n  \"correlation_model\": \"df-ccsd\","
              "\n  \"backend\": \"cuda\",\n  \"forces_Eh_per_bohr\": [";
    for (std::size_t a = 0; a < atoms; ++a) {
      if (a) output << ',';
      output << '[' << result.forces[3 * a] << ',' << result.forces[3 * a + 1] << ','
             << result.forces[3 * a + 2] << ']';
    }
    output << "]\n}\n";
    output.close();
    if (!output) throw std::runtime_error("could not persist CCSD response");
    return 0;
  } catch (const std::exception& failure) {
    std::cerr << failure.what() << '\n';
    return 1;
  }
}
