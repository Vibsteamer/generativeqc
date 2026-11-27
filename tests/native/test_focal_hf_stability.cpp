#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

#include "scf/reference/linalg.hpp"
#include "scf/solver/uhf_second_order.hpp"

namespace {

namespace so = generativeqc::scf::solver::uhf_second_order;
namespace ref = generativeqc::scf::reference;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

enum class SearchCase { Complete, Sampled, RotatedPair };

void check_hidden_unrestricted_mode(std::size_t n, SearchCase search) {
  // One alpha electron in an orthonormal basis. The positive semidefinite
  // rank-one DF tensor is (ij|kl) = L_ij L_kl.
  const auto first = n - 2;
  const auto second = n - 1;
  const bool rotated = search == SearchCase::RotatedPair;
  const double symmetric = std::sqrt(0.5);
  // The rotated direction is perpendicular to the final two coordinates of
  // the mixed start used when this fixture was introduced. Keep it fixed.
  const auto direction = rotated ? std::array{0.5852340465236672, -0.8108644219532227}
                                 : std::array{symmetric, -symmetric};
  so::Vector factor(n * n, 0.0);
  so::Vector fock(n * n, 0.0);
  factor[0] = 1.0;
  for (std::size_t i = 1; i < first; ++i) {
    factor[i * n + i] = 0.25;
    fock[i * n + i] = i <= 6 ? 1.0 + 0.125 * static_cast<double>(i - 1) : 2.0;
  }
  fock[first * n + first] = fock[second * n + second] = 3.0;
  factor[first * n + first] = 0.5 + 3.0 * direction[0] * direction[0];
  factor[second * n + second] = 0.5 + 3.0 * direction[1] * direction[1];
  factor[first * n + second] = factor[second * n + first] = 3.0 * direction[0] * direction[1];

  auto hcore = fock;
  for (std::size_t k = 0; k < hcore.size(); ++k) hcore[k] -= factor[k];
  hcore[0] += 1.0;
  so::Problem problem;
  problem.n = n;
  problem.occupied = {1, 0};
  problem.hcore = &hcore;
  problem.two_electron = [&](const so::Vector& alpha, const so::Vector& beta) {
    double trace = 0.0;
    for (std::size_t k = 0; k < factor.size(); ++k) trace += factor[k] * (alpha[k] + beta[k]);
    const auto alpha_exchange = ref::multiply(ref::multiply(factor, alpha, n), factor, n);
    const auto beta_exchange = ref::multiply(ref::multiply(factor, beta, n), factor, n);
    so::SpinMatrices result{factor, factor};
    for (std::size_t k = 0; k < factor.size(); ++k) {
      result[0][k] = trace * factor[k] - alpha_exchange[k];
      result[1][k] = trace * factor[k] - beta_exchange[k];
    }
    return result;
  };
  const auto frame = ref::identity(n);
  const auto state = so::from_coefficients(problem, {frame, frame});
  require(so::gradient_max(state) < 1e-14, "fixture is not stationary");
  for (std::size_t i = 0; i < fock.size(); ++i)
    require(std::abs(state.fock[0][i] - fock[i]) < 1e-14, "fixture is not an Aufbau determinant");

  // Coulomb and exchange cancel for the one-electron energy, so c^T h c
  // gives a finite difference independent of the Hessian implementation.
  const auto energy = [&](double angle) {
    const std::array aos{std::size_t{0}, first, second};
    const double sine = std::sin(angle);
    const std::array coefficients{std::cos(angle), sine * direction[0], sine * direction[1]};
    double value = 0.0;
    for (std::size_t i = 0; i < aos.size(); ++i)
      for (std::size_t j = 0; j < aos.size(); ++j)
        value += coefficients[i] * hcore[aos[i] * n + aos[j]] * coefficients[j];
    return value;
  };
  constexpr double step = 1e-4;
  const double curvature = (energy(step) + energy(-step) - 2.0 * energy(0.0)) / (step * step);
  require(std::abs(curvature + 1.0) < 1e-8, "fixture lost its negative UHF curvature");

  const so::StabilityControls controls;
  const auto mode = so::lowest_hessian_mode(problem, state, controls);
  require(std::isfinite(mode.ritz_value) && std::isfinite(mode.residual),
          "the stability probe did not report finite diagnostics");
  if (mode.ritz_converged) {
    const auto image = so::hessian_action(problem, state, mode.vector);
    double residual_squared = 0.0;
    for (std::size_t i = 0; i < image.size(); ++i) {
      const double entry = image[i] - 0.5 * mode.ritz_value * mode.vector[i];
      residual_squared += entry * entry;
    }
    require(mode.residual <= controls.residual_tolerance,
            "reported Ritz convergence exceeds its residual tolerance");
    require(std::sqrt(residual_squared) <= controls.residual_tolerance,
            "returned Ritz pair exceeds its residual tolerance");
  }
  if (!rotated)
    require(mode.complete_rotation_space == (search == SearchCase::Complete),
            "incorrect rotation-space scope");
  if (!rotated || mode.complete_rotation_space)
    require(mode.ritz_converged && std::abs(mode.ritz_value - curvature) < 1e-8,
            "the stability probe missed the independently known negative UHF mode");

  std::cout << "UHF rotations=" << (n - 1)
            << " rotated_pair=" << rotated
            << " complete_space=" << mode.complete_rotation_space
            << " ritz_converged=" << mode.ritz_converged
            << " Ritz=" << mode.ritz_value << " Ritz_residual=" << mode.residual
            << " independent_curvature=" << curvature
            << " Hessian_actions=" << mode.hessian_actions << '\n';
}

void check_restricted_curvature_units() {
  // Two spins share one spatial orbital. The independent energy change is
  // 2 * gap * sin(t)^2 when both orbitals rotate through the same angle t.
  constexpr double gap = -4e-6;
  const so::Vector hcore{0.0, 0.0, 0.0, gap};
  so::Problem problem;
  problem.n = 2;
  problem.occupied = {1, 1};
  problem.hcore = &hcore;
  problem.restricted = true;
  problem.two_electron = [](const so::Vector&, const so::Vector&) {
    return so::SpinMatrices{so::Vector(4, 0.0), so::Vector(4, 0.0)};
  };
  const auto frame = ref::identity(2);
  const auto state = so::from_coefficients(problem, {frame, frame});
  require(so::gradient_max(state) == 0.0, "restricted fixture is not stationary");
  constexpr double step = 1e-4;
  const double sine = std::sin(step);
  const double curvature = 4.0 * gap * sine * sine / (step * step);
  const auto mode = so::lowest_hessian_mode(problem, state);
  require(mode.ritz_converged && mode.ritz_value < -1e-5 &&
              std::abs(mode.ritz_value - curvature) < 1e-12,
          "the reported RHF curvature disagrees with a common spatial rotation");
  std::cout << "RHF common_rotation_Ritz=" << mode.ritz_value
            << " independent_curvature=" << curvature << '\n';
}

}  // namespace

int main() {
  try {
    check_hidden_unrestricted_mode(9, SearchCase::Complete);
    check_hidden_unrestricted_mode(62, SearchCase::Sampled);
    check_hidden_unrestricted_mode(62, SearchCase::RotatedPair);
    check_restricted_curvature_units();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_focal_hf_stability failed: " << error.what() << '\n';
    return 1;
  }
}
