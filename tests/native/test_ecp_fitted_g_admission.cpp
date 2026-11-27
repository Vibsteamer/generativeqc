#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "generativeqc/generativeqc.h"

namespace {

struct Atom {
  int32_t atomic_number;
  int32_t core;
};

struct Case {
  const char* name;
  std::vector<Atom> atoms;
  int32_t charge;
  uint32_t multiplicity;
  generativeqc_basis_representation representation;
  bool admits_g;
};

using ContextOwner = std::unique_ptr<generativeqc_context, decltype(&generativeqc_context_destroy)>;
using SystemOwner = std::unique_ptr<generativeqc_system, decltype(&generativeqc_system_destroy)>;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

generativeqc_status create(generativeqc_context* context, const Case& c, uint32_t ecp_shell_l,
                           SystemOwner& system) {
  std::vector<generativeqc_atom> atoms;
  std::vector<int32_t> cores;
  std::vector<generativeqc_shell> shells;
  std::vector<generativeqc_primitive> primitives;
  std::vector<generativeqc_ecp_term> terms;
  for (std::size_t a = 0; a < c.atoms.size(); ++a) {
    const auto index = static_cast<uint32_t>(a);
    atoms.push_back({c.atoms[a].atomic_number, 0.1 * static_cast<double>(a % 3),
                     -0.2 * static_cast<double>(a % 2), 3.1 * static_cast<double>(a)});
    cores.push_back(c.atoms[a].core);
    shells.push_back({index, 0, static_cast<uint32_t>(primitives.size()), 1});
    primitives.push_back({0.9, 1.0});
    if (c.atoms[a].core != 0) {
      shells.push_back({index, ecp_shell_l, static_cast<uint32_t>(primitives.size()), 1});
      primitives.push_back({0.6, 1.0});
      terms.push_back({index, -1, 2, 0.8, -2.0});
    }
  }
  const generativeqc_system_descriptor descriptor{
      sizeof(descriptor), GENERATIVEQC_ABI_VERSION,
      atoms.data(),       static_cast<uint32_t>(atoms.size()),
      shells.data(),      static_cast<uint32_t>(shells.size()),
      primitives.data(),  static_cast<uint32_t>(primitives.size()),
      c.charge,           c.multiplicity,
      c.representation};
  generativeqc_system* raw = nullptr;
  const auto status = generativeqc_system_create_ecp(context, &descriptor, cores.data(),
                                                     terms.data(), terms.size(), &raw);
  system.reset(raw);
  return status;
}

std::vector<Atom> repeat(std::vector<Atom> atoms, Atom atom, int count) {
  for (int i = 0; i < count; ++i) atoms.push_back(atom);
  return atoms;
}

std::vector<Case> cases() {
  constexpr auto sph = GENERATIVEQC_BASIS_SPHERICAL;
  constexpr auto cart = GENERATIVEQC_BASIS_CARTESIAN;
  const Atom h{1, 0}, c{6, 0}, o{8, 0}, na{11, 0}, na_ecp{11, 10}, cl{17, 0}, k{19, 0}, br{35, 10},
      i{53, 28}, br_all{35, 0}, i_large{53, 46};
  const auto c4h6br2o = repeat(repeat(repeat({o}, c, 4), h, 6), br, 2);
  return {
      {"HBr singlet", {h, br}, 0, 1, sph, true},
      {"HBr cation doublet", {h, br}, 1, 2, sph, true},
      {"HBr triplet", {h, br}, 0, 3, sph, false},
      {"HBr cation singlet parity mismatch", {h, br}, 1, 1, sph, false},
      {"HBr dication out of charge range", {h, br}, 2, 1, sph, false},
      {"HBr dianion out of charge range", {h, br}, -2, 1, sph, false},
      {"HBr Cartesian singlet", {h, br}, 0, 1, cart, false},
      {"Br atom doublet", {br}, 0, 2, sph, true},
      {"bromide singlet", {br}, -1, 1, sph, true},
      {"Br atom quartet", {br}, 0, 4, sph, false},
      {"BrH2 doublet", {br, h, h}, 0, 2, sph, true},
      {"BrH3 doublet parity mismatch", {br, h, h, h}, 0, 2, sph, false},
      {"BrH4 doublet with an f-only fitted limit", {br, h, h, h, h}, 0, 2, sph, false},
      {"HI singlet", {h, i}, 0, 1, sph, true},
      {"CH3I singlet", {c, h, h, h, i}, 0, 1, sph, true},
      {"CH3I cation doublet", {c, h, h, h, i}, 1, 2, sph, true},
      {"CH3I dication singlet", {c, h, h, h, i}, 2, 1, sph, false},
      {"CH3I anion doublet outside the molecular-halide charges",
       {c, h, h, h, i},
       -1,
       2,
       sph,
       false},
      {"CH3I triplet", {c, h, h, h, i}, 0, 3, sph, false},
      {"H2O-Br doublet", {o, h, h, br}, 0, 2, sph, true},
      {"C4H6Br2O singlet", c4h6br2o, 0, 1, sph, true},
      {"C4H6Br2O anion doublet", c4h6br2o, -1, 2, sph, true},
      {"C4H6Br2O anion singlet parity mismatch", c4h6br2o, -1, 1, sph, false},
      {"C4H6Br2O dianion", c4h6br2o, -2, 1, sph, false},
      {"C4H6Br2O dication", c4h6br2o, 2, 1, sph, false},
      {"Br2 singlet", {br, br}, 0, 1, sph, true},
      {"IBr singlet", {i, br}, 0, 1, sph, true},
      {"CH2BrCl singlet", {c, h, h, br, cl}, 0, 1, sph, true},
      {"NaI singlet", {na, i}, 0, 1, sph, true},
      {"NaI triplet", {na, i}, 0, 3, sph, false},
      {"NaI with Na ECP", {na_ecp, i}, 0, 1, sph, false},
      {"KI heavy partner", {k, i}, 0, 1, sph, false},
      {"all-electron Br partner", {br_all, i}, 0, 1, sph, false},
      {"large-core I", {h, i_large}, 0, 1, sph, false},
      {"Na ECP hydride", {na_ecp, h}, 0, 1, sph, false},
  };
}

void check_fitted_g_admission() {
  const generativeqc_context_descriptor context_desc{sizeof(context_desc), GENERATIVEQC_ABI_VERSION,
                                                     0, GENERATIVEQC_BACKEND_CPU_REFERENCE};
  generativeqc_context* raw = nullptr;
  require(generativeqc_context_create(&context_desc, &raw) == GENERATIVEQC_STATUS_SUCCESS,
          "context create");
  ContextOwner context(raw, generativeqc_context_destroy);
  for (const auto& c : cases()) {
    SystemOwner orbital_f(nullptr, generativeqc_system_destroy);
    require(create(context.get(), c, 3, orbital_f) == GENERATIVEQC_STATUS_SUCCESS && orbital_f,
            std::string(c.name) + ": orbital s/p/d/f ECP admission must not depend on the state");
    SystemOwner orbital_g(nullptr, generativeqc_system_destroy);
    const auto status = create(context.get(), c, 4, orbital_g);
    if (c.admits_g) {
      require(status == GENERATIVEQC_STATUS_SUCCESS && orbital_g,
              std::string(c.name) + ": fitted small-core halide state rejected orbital g shells");
    } else {
      require(status == GENERATIVEQC_STATUS_INVALID_ARGUMENT && !orbital_g,
              std::string(c.name) + ": orbital g shells admitted outside the fitted state");
      const std::string detail = generativeqc_context_get_last_detail(context.get());
      require(detail.find("supports orbital s/p/d/f shells") != std::string::npos,
              std::string(c.name) + ": unexpected g-shell rejection detail: " + detail);
    }
    SystemOwner orbital_h(nullptr, generativeqc_system_destroy);
    require(create(context.get(), c, 5, orbital_h) == GENERATIVEQC_STATUS_INVALID_ARGUMENT &&
                !orbital_h,
            std::string(c.name) + ": orbital h shells admitted");
  }
}

}  // namespace

int main() {
  try {
    check_fitted_g_admission();
    std::cout << "ECP orbital-g fitted-state admission matrix passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
