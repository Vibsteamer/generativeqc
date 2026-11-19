// Physical UHF reference from the same native DF-J/DF-K Hamiltonian on both backends.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "molecule/basis.hpp"
#include "scf/mean_field.hpp"

namespace {
using namespace generativeqc;

void require(bool ok, const char* err) {
  if (!ok) throw std::runtime_error(err);
}

void read_shells(std::istream& input, core::System& sys, std::size_t count,
                 unsigned maximum_angular_momentum) {
  require(count > 0 && count <= 100000, "invalid reference shell count");
  sys.shells.resize(count);
  for (auto& shell : sys.shells) {
    std::size_t n{};
    input >> shell.atom_index >> shell.angular_momentum >> n;
    require(input && n > 0 && n <= 1000, "invalid shell");
    shell.primitives.resize(n);
    for (auto& p : shell.primitives) input >> p.exponent >> p.coefficient;
  }
  std::string detail;
  if (!input || molecule::validate_and_normalize(sys, detail, maximum_angular_momentum) !=
                    GENERATIVEQC_STATUS_SUCCESS)
    throw std::runtime_error("invalid basis: " + detail);
}

void vector(std::ostream& out, const std::vector<double>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    require(std::isfinite(values[i]), "nonfinite UHF physical reference");
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}
}  // namespace

int main(int argc, char** argv) try {
  using namespace generativeqc;
  require(argc == 4, "usage: snapshot cpu|cuda geometry.in output.json");
  const std::string backend = argv[1];
  require(backend == "cpu" || backend == "cuda", "unknown UHF reference backend");
  std::ifstream input(argv[2]);
  core::System sys;
  std::size_t natom{}, orbital_shells{}, aux_shells{}, budget{};
  input >> natom >> orbital_shells >> aux_shells >> budget >> sys.charge >> sys.multiplicity;
  require(input && natom && natom <= 1000 && budget && sys.multiplicity > 1,
          "invalid open-shell input");
  sys.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  sys.atoms.resize(natom);
  for (auto& a : sys.atoms)
    input >> a.atomic_number >> a.position[0] >> a.position[1] >> a.position[2];
  auto aux = sys;
  read_shells(input, sys, orbital_shells, 4);
  read_shells(input, aux, aux_shells, 5);
  input >> std::ws;
  require(input.eof(), "trailing reference data");
  scf::ScfOptions options;
  options.max_iterations = 240;
  options.energy_tolerance = 1e-13;
  options.density_tolerance = 1e-11;
  options.screening_tolerance = 0;
  options.density_fitting_relative_threshold = 1e-10;
  options.density_fitting_memory_budget_bytes = budget;
  options.reference_memory_budget_bytes = budget;
  options.precision_mode = GENERATIVEQC_PRECISION_FP64;
  options.compute_forces = true;
  options.export_physical_reference = true;
  const auto target = backend == "cpu" ? scf::FockBackend::Cpu : scf::FockBackend::Cuda;
  options.resolved_fock_build = scf::resolve_fock_build(
      scf::make_hf_fock_spec(scf::FockSpin::Unrestricted, scf::FockApproximation::DensityFitted),
      target, options.screening_tolerance, options.density_fitting_relative_threshold);
  const auto result = backend == "cpu"
                          ? scf::run_cpu_fock_strategy(sys, &aux, options)
                          : scf::run_cuda_independent_fock_strategy(sys, &aux, options, 0);
  require(result.converged && result.unrestricted_reference && std::isfinite(result.energy),
          "native DF-UHF did not converge to a physical reference");
  const auto& ref = *result.unrestricted_reference;
  require(ref.nocc[0] > ref.nocc[1] && result.forces.size() == 3 * natom,
          "missing spin channels or nuclear forces");
  std::ofstream out(argv[3]);
  require(bool(out), "cannot open snapshot output");
  out << std::setprecision(17) << "{\"backend\":\"" << backend << "\",\"energy\":" << result.energy
      << ",\"nbf\":" << ref.nbf << ",\"occupations\":[" << ref.nocc[0] << ',' << ref.nocc[1]
      << "],\"iterations\":" << result.iterations << ",\"Fock_builds\":" << result.fock_builds
      << ",\"commutator\":" << ref.commutator_residual
      << ",\"canonical_error\":" << ref.canonical_error
      << ",\"density_drift\":" << ref.canonical_density_drift << ",\"overlap\":";
  vector(out, ref.overlap);
  out << ",\"hcore\":";
  vector(out, ref.hcore);
  out << ",\"forces\":";
  vector(out, result.forces);
  for (std::size_t s = 0; s < 2; ++s) {
    out << ",\"" << (s ? "beta" : "alpha") << "\":{\"mo\":";
    vector(out, ref.coefficients[s]);
    out << ",\"eps\":";
    vector(out, ref.orbital_energies[s]);
    out << ",\"density\":";
    vector(out, ref.density[s]);
    out << ",\"fock\":";
    vector(out, ref.fock[s]);
    out << '}';
  }
  out << "}\n";
  require(bool(out), "incomplete snapshot output");
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
