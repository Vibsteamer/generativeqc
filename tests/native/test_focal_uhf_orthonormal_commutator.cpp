#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/mean_field.hpp"
#include "scf/proposals.hpp"
#include "scf/reference/linalg.hpp"
#include "scf/reference/mean_field.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

generativeqc::core::System diffuse_doublet_system() {
  generativeqc::core::System s;
  s.atoms = {{2, {0.0, 0.0, -0.75}}, {1, {0.0, 0.0, 0.75}}};
  // Valence + diffuse s/p shells giving a non-trivial overlap condition number
  // where max|X^T(FDS-SDF)X| significantly exceeds AO max|FDS-SDF|.
  s.shells = {
      {0, 0, {{6.36242139, 0.15432897}, {1.15892300, 0.53532814}, {0.31364979, 0.44463454}}},
      {0, 0, {{0.06, 1.0}}},
      {0, 0, {{0.052, 1.0}}},
      {1, 0, {{3.42525091, 0.15432897}, {0.62391373, 0.53532814}, {0.16885540, 0.44463454}}},
      {1, 0, {{0.055, 1.0}}},
      {1, 0, {{0.047, 1.0}}},
  };
  s.charge = 0;
  s.multiplicity = 2;
  std::string detail;
  require(generativeqc::molecule::validate_and_normalize(s, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "system normalization failed");
  return s;
}

double orthonormal_uhf_commutator_max_abs(const std::vector<double>& residual,
                                          const std::vector<double>& overlap, std::size_t n) {
  using namespace generativeqc::scf::reference;
  const Matrix x = symmetric_orthogonalizer(overlap, n);
  const Matrix xt = transpose(x, n);
  const auto [alpha_res, beta_res] = split_spin_matrices(residual, n * n);
  const double alpha_max = residual_max_abs(multiply(xt, multiply(alpha_res, x, n), n));
  const double beta_max = residual_max_abs(multiply(xt, multiply(beta_res, x, n), n));
  return std::max(alpha_max, beta_max);
}

struct ObservedStop {
  unsigned iterations{};
  double raw_max{std::numeric_limits<double>::infinity()};
  double ortho_max{std::numeric_limits<double>::infinity()};
};

ObservedStop run_observed_df_uhf(const generativeqc::scf::PreparedFockPlan& plan,
                                 generativeqc::scf::ScfOptions options,
                                 generativeqc::scf::ScfResult& out_result) {
  ObservedStop stop;
  generativeqc::scf::ScfHooks hooks;
  hooks.observe = [&](const generativeqc::scf::ScfSnapshot& snap,
                      const generativeqc::scf::ProposalDecision&) {
    stop.iterations = snap.iteration;
    stop.raw_max = generativeqc::scf::reference::residual_max_abs(snap.residual);
    stop.ortho_max = orthonormal_uhf_commutator_max_abs(snap.residual, snap.overlap, snap.nbf);
  };
  options.hooks = &hooks;
  out_result = generativeqc::scf::run_prepared_fock_strategy(plan, options);
  return stop;
}

// Synthetic even-tempered shells on a bromine nucleus and a hydrogen; repeating the most
// diffuse p exponent at `ratio` drives the smallest overlap eigenvalue towards 1e-5 while
// core-Hamiltonian elements stay near 5e2 Eh, the regime of the heavy-atom references.
generativeqc::core::System near_duplicate_heavy_atom_system(double ratio) {
  generativeqc::core::System s;
  s.atoms = {{35, {0.0, 0.0, 0.0}}, {1, {0.0, 0.0, 2.67}}};
  for (int k = 0; k < 10; ++k) s.shells.push_back({0, 0, {{0.05 * std::pow(3.0, k), 1.0}}});
  for (int k = 0; k < 7; ++k) s.shells.push_back({0, 1, {{0.04 * std::pow(3.0, k), 1.0}}});
  s.shells.push_back({0, 1, {{0.04 * ratio, 1.0}}});
  for (const double exponent : {0.1, 0.4, 1.6, 6.4}) s.shells.push_back({1, 0, {{exponent, 1.0}}});
  s.charge = 1;
  s.multiplicity = 2;
  std::string detail;
  require(generativeqc::molecule::validate_and_normalize(s, detail) == GENERATIVEQC_STATUS_SUCCESS,
          "near-duplicate system normalization failed");
  return s;
}

struct CommutatorNorms {
  long double maximum{};
  long double rms{};
};

/** X^T (F D S - S D F) X of the given FP64 inputs, evaluated as written in extended precision. */
CommutatorNorms extended_orthonormal_commutator(const std::vector<double>& fock,
                                               const std::vector<double>& density,
                                               const std::vector<double>& overlap,
                                               const std::vector<double>& orthogonalizer,
                                               std::size_t n) {
  using Extended = std::vector<long double>;
  const auto product = [n](const Extended& a, const Extended& b) {
    Extended c(n * n, 0.0L);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t k = 0; k < n; ++k)
        for (std::size_t j = 0; j < n; ++j) c[i * n + j] += a[i * n + k] * b[k * n + j];
    return c;
  };
  const Extended f(fock.begin(), fock.end()), d(density.begin(), density.end());
  const Extended s(overlap.begin(), overlap.end()), x(orthogonalizer.begin(), orthogonalizer.end());
  Extended xt(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) xt[i * n + j] = x[j * n + i];
  const Extended fds = product(product(f, d), s);
  const Extended sdf = product(product(s, d), f);
  Extended commutator(n * n);
  for (std::size_t i = 0; i < n * n; ++i) commutator[i] = fds[i] - sdf[i];
  CommutatorNorms result;
  long double square = 0.0L;
  for (const long double value : product(xt, product(commutator, x))) {
    result.maximum = std::max(result.maximum, std::fabs(value));
    square += value * value;
  }
  result.rms = std::sqrt(square / static_cast<long double>(n * n));
  return result;
}

}  // namespace

int main() {
  using namespace generativeqc::scf;
  try {
    // 1. Default option must stay false so upstream UHF consumers are untouched.
    const ScfOptions defaults;
    require(!defaults.enforce_orthonormal_uhf_commutator,
            "ScfOptions::enforce_orthonormal_uhf_commutator must default to false");

    const auto system = diffuse_doublet_system();

    // 2. With enforce_orthonormal_uhf_commutator == false, setting
    // export_physical_reference = true with tight density_tolerance (as in
    // UMP2/UCCSD) must preserve identical baseline SCF iteration stopping.
    ScfOptions plain_opts;
    plain_opts.max_iterations = 200;
    plain_opts.energy_tolerance = 1e-13;
    plain_opts.density_tolerance = 1e-11;
    plain_opts.screening_tolerance = 0.0;
    plain_opts.density_fitting_relative_threshold = 1e-10;
    plain_opts.compute_forces = false;
    plain_opts.export_physical_reference = false;
    plain_opts.enforce_orthonormal_uhf_commutator = false;
    auto spec = make_hf_fock_spec(FockSpin::Unrestricted, FockApproximation::DensityFitted);
    spec.derivative_order = 0;
    plain_opts.resolved_fock_build =
        resolve_fock_build(spec, FockBackend::Cpu, plain_opts.screening_tolerance,
                           plain_opts.density_fitting_relative_threshold);
    const PreparedFockPlan plan(system, &system, *plain_opts.resolved_fock_build);

    ScfResult plain_res;
    const ObservedStop plain_stop = run_observed_df_uhf(plan, plain_opts, plain_res);
    require(plain_res.converged, "plain baseline DF-UHF did not converge");

    ScfOptions ref_default_opts = plain_opts;
    ref_default_opts.export_physical_reference = true;
    require(!ref_default_opts.enforce_orthonormal_uhf_commutator,
            "ref_default_opts must keep enforce_orthonormal_uhf_commutator false");

    ScfResult ref_default_res;
    const ObservedStop ref_default_stop =
        run_observed_df_uhf(plan, ref_default_opts, ref_default_res);
    require(ref_default_res.converged, "default physical-reference DF-UHF did not converge");
    require(ref_default_res.iterations == plain_res.iterations,
            "export_physical_reference with default flag changed baseline UHF iteration count");
    require(ref_default_stop.iterations == plain_stop.iterations,
            "export_physical_reference with default flag changed observed stopping iterate");
    require(std::abs(ref_default_stop.raw_max - plain_stop.raw_max) <= 1e-15,
            "export_physical_reference with default flag changed baseline AO residual at stop");

    // 3. Opt-in focal flag must enforce max|X^T(FDS-SDF)X| <= density_tolerance
    // at convergence, even when AO max|FDS-SDF| already fell below tolerance.
    ScfOptions focal_opts = ref_default_opts;
    focal_opts.enforce_orthonormal_uhf_commutator = true;

    ScfResult focal_res;
    const ObservedStop focal_stop = run_observed_df_uhf(plan, focal_opts, focal_res);
    require(focal_res.converged, "focal orthonormal-commutator DF-UHF did not converge");
    require(focal_res.unrestricted_reference != nullptr,
            "focal DF-UHF did not export unrestricted_reference");
    require(focal_stop.ortho_max <= focal_opts.density_tolerance,
            "focal DF-UHF stopped with orthonormal commutator exceeding density_tolerance");

    // Also verify the finalized physical reference satisfies the same bound.
    const auto& ref = *focal_res.unrestricted_reference;
    const auto alpha_final_res =
        reference::commutator_residual(ref.fock[0], ref.density[0], ref.overlap, ref.nbf);
    const auto beta_final_res =
        reference::commutator_residual(ref.fock[1], ref.density[1], ref.overlap, ref.nbf);
    const double final_ortho_max = orthonormal_uhf_commutator_max_abs(
        reference::concatenate(alpha_final_res, beta_final_res), ref.overlap, ref.nbf);
    require(final_ortho_max <= focal_opts.density_tolerance,
            "finalized focal DF-UHF reference exceeded orthonormal commutator tolerance");

    // 4. Confirm strict separation between AO-only and orthonormal stopping on
    // the ill-conditioned diffuse fixture under geometric fixed-point descent:
    // baseline (false) stops as soon as AO max|FDS-SDF| <= tolerance while
    // max|X^T(FDS-SDF)X| is still > tolerance, whereas opt-in (true) refines
    // until max|X^T(FDS-SDF)X| <= tolerance.
    ScfOptions sep_base = ref_default_opts;
    sep_base.enable_diis = false;
    sep_base.density_damping = 0.75;
    sep_base.stabilize_focal_rhf_convergence = true;
    sep_base.energy_tolerance = 1e-10;
    sep_base.density_tolerance = 1e-9;
    sep_base.max_iterations = 400;
    sep_base.enforce_orthonormal_uhf_commutator = false;

    ScfResult sep_base_res;
    const ObservedStop sep_base_stop = run_observed_df_uhf(plan, sep_base, sep_base_res);
    require(sep_base_res.converged, "separation baseline DF-UHF did not converge");
    require(sep_base_stop.raw_max <= sep_base.density_tolerance,
            "separation baseline did not satisfy AO commutator bound");
    if (!(sep_base_stop.ortho_max > sep_base.density_tolerance)) {
      std::cerr << "diag: iters=" << sep_base_res.iterations
                << " drms=" << sep_base_res.density_rms
                << " raw_max=" << sep_base_stop.raw_max
                << " ortho_max=" << sep_base_stop.ortho_max << "\n";
    }
    require(sep_base_stop.ortho_max > sep_base.density_tolerance,
            "separation fixture did not expose orthonormal commutator gap");

    ScfOptions sep_focal = sep_base;
    sep_focal.enforce_orthonormal_uhf_commutator = true;
    ScfResult sep_focal_res;
    const ObservedStop sep_focal_stop = run_observed_df_uhf(plan, sep_focal, sep_focal_res);
    require(sep_focal_res.converged, "separation focal DF-UHF did not converge");
    require(sep_focal_stop.ortho_max <= sep_focal.density_tolerance,
            "opt-in orthonormal commutator guard failed to satisfy bound");
    require(sep_focal_res.iterations > sep_base_res.iterations,
            "opt-in orthonormal commutator guard did not refine beyond AO-only stop");

    // 5. The stopping value must agree with an extended-precision evaluation of its
    // definition, well below density_tolerance, on an overlap with s_min ~ 1e-5.
    const auto heavy = near_duplicate_heavy_atom_system(1.01);
    const auto one_electron = generativeqc::integrals::build_integrals(heavy, false, false);
    const std::size_t hn = one_electron.nbf;
    const auto overlap_spectrum = reference::symmetric_eigen(one_electron.overlap, hn);
    const double s_min = overlap_spectrum.values.front();
    require(s_min > 1e-6 && s_min < 3e-5, "near-duplicate fixture left the s_min ~ 1e-5 regime");
    const auto x = reference::symmetric_orthogonalizer(one_electron.overlap, hn);
    const auto sx = reference::multiply(one_electron.overlap, x, hn);
    const auto orbitals = reference::generalized_eigen(one_electron.hcore, x, hn);
    const auto density = reference::density_from_orbitals(orbitals.vectors, hn, 18, 1.0);
    const auto extended = extended_orthonormal_commutator(
        one_electron.hcore, density, one_electron.overlap, x, hn);
    const double evaluated = reference::orthonormal_commutator_max_abs(
        one_electron.hcore, density, x, sx, hn);
    std::cout << "evaluation order: s_min=" << s_min << " extended=" << extended.maximum
              << " evaluated_error=" << std::abs(evaluated - extended.maximum) << "\n";
    require(std::abs(evaluated - extended.maximum) <= 2e-12,
            "orthonormal commutator evaluation deviates from the extended reference");

    auto perturbed = one_electron.hcore;
    perturbed[1] += 1e-4;
    perturbed[hn] += 1e-4;
    const auto extended_perturbed = extended_orthonormal_commutator(
        perturbed, density, one_electron.overlap, x, hn);
    const double evaluated_rms = reference::residual_rms(
        reference::orthonormal_commutator_residual(perturbed, density, x, sx, hn));
    require(extended_perturbed.rms > 1e-11L, "perturbed commutator fixture is stationary");
    require(std::abs(evaluated_rms - extended_perturbed.rms) <= 2e-12L,
            "orthonormal commutator RMS deviates from the extended reference");
    auto poisoned = one_electron.hcore;
    poisoned[hn + 1] = std::numeric_limits<double>::quiet_NaN();
    require(std::isinf(reference::orthonormal_commutator_max_abs(
                poisoned, density, x, sx, hn)),
            "nonfinite orthonormal commutator input did not fail closed");

    std::cout << "test_focal_uhf_orthonormal_commutator passed: sep_base_iters="
              << sep_base_res.iterations << " sep_base_ortho_max=" << sep_base_stop.ortho_max
              << " sep_focal_iters=" << sep_focal_res.iterations
              << " sep_focal_ortho_max=" << sep_focal_stop.ortho_max << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_focal_uhf_orthonormal_commutator failed: " << error.what() << "\n";
    return 1;
  }
}
