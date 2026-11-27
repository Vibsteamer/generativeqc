#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "generativeqc/generativeqc.h"
#include "generativeqc/ucc_fitting.h"

namespace {

constexpr double kCodata = 137.03599967994;
constexpr uint32_t kDlu1e = 1;
constexpr double kUntouched = -37.0;

using ContextOwner = std::unique_ptr<generativeqc_context, decltype(&generativeqc_context_destroy)>;
using SystemOwner = std::unique_ptr<generativeqc_system, decltype(&generativeqc_system_destroy)>;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

const std::array<double, 8> kRejected{
    -1.0,
    -kCodata,
    -std::numeric_limits<double>::min(),
    -std::numeric_limits<double>::denorm_min(),
    -std::numeric_limits<double>::max(),
    std::numeric_limits<double>::quiet_NaN(),
    std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity(),
};

struct Fixture {
  ContextOwner context{nullptr, generativeqc_context_destroy};
  SystemOwner system{nullptr, generativeqc_system_destroy};
  std::size_t n = 0;
  std::size_t natom = 0;
};

// All-electron NeH+ with tight s and p primitives, so X2C-1e changes hcore at c = 100.
Fixture make_fixture() {
  Fixture f;
  const generativeqc_context_descriptor context_desc{sizeof(context_desc), GENERATIVEQC_ABI_VERSION,
                                                     0, GENERATIVEQC_BACKEND_CPU_REFERENCE};
  generativeqc_context* context = nullptr;
  require(generativeqc_context_create(&context_desc, &context) == GENERATIVEQC_STATUS_SUCCESS,
          "context create");
  f.context.reset(context);
  const std::array<generativeqc_atom, 2> atoms{{{10, 0.0, 0.0, 0.0}, {1, 0.0, 0.3, 1.9}}};
  const std::array<generativeqc_primitive, 5> primitives{
      {{4000.0, 1.0}, {60.0, 1.0}, {25.0, 1.0}, {0.8, 1.0}, {1.2, 1.0}}};
  const std::array<generativeqc_shell, 5> shells{
      {{0, 0, 0, 1}, {0, 0, 1, 1}, {0, 1, 2, 1}, {0, 1, 3, 1}, {1, 0, 4, 1}}};
  const generativeqc_system_descriptor descriptor{sizeof(descriptor),
                                                  GENERATIVEQC_ABI_VERSION,
                                                  atoms.data(),
                                                  static_cast<uint32_t>(atoms.size()),
                                                  shells.data(),
                                                  static_cast<uint32_t>(shells.size()),
                                                  primitives.data(),
                                                  static_cast<uint32_t>(primitives.size()),
                                                  1,
                                                  1,
                                                  GENERATIVEQC_BASIS_SPHERICAL};
  generativeqc_system* system = nullptr;
  require(generativeqc_system_create(f.context.get(), &descriptor, &system) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "system create");
  f.system.reset(system);
  f.n = 1 + 1 + 3 + 3 + 1;
  f.natom = atoms.size();
  return f;
}

std::vector<double> configured_hcore(const Fixture& f) {
  std::vector<double> overlap(f.n * f.n), hcore(f.n * f.n);
  double nuclear_repulsion = 0.0;
  require(generativeqc_system_one_electron_integrals_cpu_v1(
              f.context.get(), f.system.get(), overlap.data(), hcore.data(), hcore.size(),
              &nuclear_repulsion) == GENERATIVEQC_STATUS_SUCCESS,
          "one-electron integrals");
  return hcore;
}

void check_configured_speed(Fixture& f) {
  require(generativeqc_system_set_x2c1e_v1(f.context.get(), f.system.get(), kDlu1e, 100.0) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "positive speed of light must be accepted");
  const auto custom = configured_hcore(f);
  for (const double c : kRejected) {
    require(generativeqc_system_set_x2c1e_v1(f.context.get(), f.system.get(), kDlu1e, c) ==
                GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            "set_x2c1e accepted speed of light " + std::to_string(c));
    require(configured_hcore(f) == custom, "rejected speed of light changed the system");
  }
  require(generativeqc_system_set_x2c1e_v1(f.context.get(), f.system.get(), kDlu1e, 0.0) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "zero must select the CODATA speed of light");
  const auto zero = configured_hcore(f);
  require(generativeqc_system_set_x2c1e_v1(f.context.get(), f.system.get(), kDlu1e, -0.0) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "negative zero must select the CODATA speed of light");
  require(configured_hcore(f) == zero, "negative zero differs from zero");
  require(generativeqc_system_set_x2c1e_v1(f.context.get(), f.system.get(), kDlu1e, kCodata) ==
              GENERATIVEQC_STATUS_SUCCESS,
          "explicit CODATA speed of light");
  require(configured_hcore(f) == zero, "zero does not select the CODATA speed of light");
  require(zero != custom, "fixture is insensitive to the speed of light");
}

template <class Call>
void check_evaluator(const char* name, std::size_t count, Call&& call) {
  std::vector<double> codata(count, kUntouched), zero(count, kUntouched);
  require(call(kCodata, codata.data(), count) == GENERATIVEQC_STATUS_SUCCESS,
          std::string(name) + " with explicit CODATA speed of light");
  require(call(0.0, zero.data(), count) == GENERATIVEQC_STATUS_SUCCESS,
          std::string(name) + " with zero speed of light");
  require(zero == codata, std::string(name) + ": zero does not select the CODATA value");
  std::vector<double> custom(count, kUntouched);
  require(call(100.0, custom.data(), count) == GENERATIVEQC_STATUS_SUCCESS && custom != codata,
          std::string(name) + " ignored a positive speed of light");
  for (const double c : kRejected) {
    std::vector<double> out(count, kUntouched);
    require(call(c, out.data(), count) == GENERATIVEQC_STATUS_INVALID_ARGUMENT,
            std::string(name) + " accepted speed of light " + std::to_string(c));
    require(out == std::vector<double>(count, kUntouched),
            std::string(name) + " wrote output for a rejected speed of light");
  }
}

void check_evaluators(const Fixture& f) {
  auto* context = f.context.get();
  const auto* system = f.system.get();
  const std::size_t matrix = f.n * f.n;
  for (const int delta_only : {0, 1}) {
    check_evaluator("x2c1e_hcore", matrix, [&](double c, double* out, std::size_t count) {
      return generativeqc_system_x2c1e_hcore_v1(context, system, kDlu1e, c, delta_only, out, count);
    });
    check_evaluator("x2c1e_hcore_derivatives", 3 * f.natom * matrix,
                    [&](double c, double* out, std::size_t count) {
                      return generativeqc_system_x2c1e_hcore_derivatives_v1(
                          context, system, kDlu1e, c, delta_only, out, count);
                    });
    std::vector<double> weights(matrix);
    for (std::size_t i = 0; i < f.n; ++i)
      for (std::size_t j = 0; j < f.n; ++j)
        weights[i * f.n + j] = 0.1 + 0.01 * static_cast<double>(i + j);
    check_evaluator(
        "x2c1e_weighted_gradient", 3 * f.natom, [&](double c, double* out, std::size_t count) {
          return generativeqc_system_x2c1e_weighted_gradient_v1(
              context, system, kDlu1e, c, delta_only, weights.data(), matrix, out, count);
        });
  }
}

}  // namespace

int main() {
  try {
    auto fixture = make_fixture();
    check_configured_speed(fixture);
    check_evaluators(fixture);
    std::cout << "X2C-1e speed-of-light ABI checks passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
