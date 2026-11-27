#ifndef GENERATIVEQC_INTEGRALS_X2C1E_HPP
#define GENERATIVEQC_INTEGRALS_X2C1E_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/types.hpp"

namespace generativeqc::integrals {

struct IntegralData;

/** Speed of light in atomic units (CODATA / PySCF lib.param.LIGHT_SPEED). */
inline constexpr double kSpeedOfLightAu = 137.03599967994;

/** Scalar-relativistic 1-electron X2C model selection.
 *
 * - None: non-relativistic 1e Hamiltonian (h = T + V + V_ECP).
 * - Dlu1e: local atomic-block X2C-1e fold-down X_A on all-electron atoms only,
 *   identity on ECP atoms (zero ECP double-counting, V_ECP untouched), with
 *   exact molecular metric renormalization R (matches PySCF sfx2c1e ATOM1E on
 *   all-electron systems to machine precision and preserves exact potential
 *   gauge invariance on mixed AE+ECP molecules).
 * - Global1e: full molecular X2C-1e (all-electron systems only; matches PySCF
 *   sfx2c1e approx='1E' to machine precision).
 * - DluLocalR: strict atomic-block U_L = diag(R_A), U_S = diag(X_A R_A) on AE
 *   atoms and identity on ECP atoms (Peng-Reiher local-R comparator).
 */
enum class X2c1eMode : std::uint32_t {
  None = 0,
  Dlu1e = 1,
  Global1e = 2,
  DluLocalR = 3,
};

/** Public-AO matrices S, T, V, W = <nabla mu | V_nuc | nabla nu>. */
struct StvwIntegrals {
  std::size_t nbf{};
  std::vector<double> overlap;
  std::vector<double> kinetic;
  std::vector<double> attraction;
  std::vector<double> small_potential;
};

/** Spectral factors of the molecular overlap and renormalization metric. */
struct X2c1eMetricFactors {
  std::size_t n{};
  std::size_t s_rank{};
  std::vector<double> s_evals;         // s_rank
  std::vector<double> s_evecs;         // n x s_rank
  std::vector<double> s_disc_evals;    // n - s_rank
  std::vector<double> s_disc_evecs;    // n x (n - s_rank)
  std::vector<double> snesc_v;         // s_rank x s_rank
  std::vector<double> snesc_ret_disc;  // s_rank x (n - s_rank)
  std::size_t k_rank{};
  std::vector<double> k_evals;         // k_rank
  std::vector<double> k_evecs;         // s_rank x k_rank
  std::vector<double> k_disc_evals;    // s_rank - k_rank
  std::vector<double> k_disc_evecs;    // s_rank x (s_rank - k_rank)
  std::vector<double> r_mat;           // n x n
};

/** Retained forward X2C-1e state for O(1)-Sylvester adjoint nuclear gradients. */
struct X2c1eState {
  X2c1eMode mode{X2c1eMode::None};
  double speed_of_light{kSpeedOfLightAu};
  std::size_t n_contr{};
  std::size_t n_prim{};
  bool has_ae_atoms{};
  core::System uncontracted_system;
  // Row-major (n_prim x n_contr) contraction matrix C: M_contr = C^T M_prim C.
  std::vector<double> contraction;
  // Contracted public-AO matrices (excluding ECP):
  //   hcore_no_ecp = h_X2C (without V_ECP)
  //   hcore_delta  = h_X2C (without V_ECP) - (T + V)
  std::vector<double> hcore_no_ecp;
  std::vector<double> hcore_delta;

  // Uncontracted fold-down matrices retained for exact differentiation:
  std::vector<double> x_mat;        // n_prim x n_prim (identity on ECP blocks)
  std::vector<double> dx_mat;       // n_prim x n_prim (X - I; zero on ECP blocks)
  std::vector<double> xw_mat;       // n_prim x n_prim (zero on ECP blocks)
  std::vector<double> ul_mat;       // n_prim x n_prim (used by DluLocalR)
  std::vector<double> us_mat;       // n_prim x n_prim (used by DluLocalR)
  std::vector<double> uw_mat;       // n_prim x n_prim (used by DluLocalR)
  std::vector<double> h_nesc;       // n_prim x n_prim
  std::vector<double> t_prim;       // n_prim x n_prim (needed for Global1e X-response)
  std::vector<double> w_prim;       // n_prim x n_prim (needed for Global1e X-response)

  // Both eigenspaces belong to this geometry for forward and adjoint derivatives.
  X2c1eMetricFactors metric;

  // Global1e full Dirac eigensystem (only populated when mode == Global1e):
  std::size_t dirac_neg_count{};
  std::size_t dirac_pos_count{};
  std::vector<double> dirac_evals_neg;  // dirac_neg_count
  std::vector<double> dirac_evals_pos;  // dirac_pos_count
  std::vector<double> dirac_evecs_neg;  // (2*n_prim) x dirac_neg_count (row-major)
  std::vector<double> dirac_evecs_pos;  // (2*n_prim) x dirac_pos_count (row-major)
  std::vector<double> dirac_al_inv;     // n_prim x n_prim (A_L^{-1} for Global1e)
};

/** Evaluate public-AO S, T, V (ionic charges) and W = <nabla|V_nuc|nabla>. */
StvwIntegrals build_stvw_integrals(const core::System& system);

/** Build the complete scalar-relativistic X2C-1e state and hcore delta.
 *
 * The returned state excludes ECP matrices so callers can add `hcore_delta`
 * directly onto an existing NR+ECP core Hamiltonian without touching V_ECP.
 */
std::shared_ptr<const X2c1eState> build_x2c1e_state(const core::System& system, X2c1eMode mode,
                                                    double speed_of_light = kSpeedOfLightAu);

/** Exact analytic nuclear gradient +d/dR Tr(W_h * Delta h_SR) or +d/dR Tr(W_h * h_X2C_no_ecp).
 *
 * Uses O(1) Sylvester/eigenvector adjoint pullback into uncontracted (S, T, V, W)
 * weights followed by direct native McMurchie-Davidson derivative contraction.
 * Never materializes (3*natom, n, n) matrices.
 *
 * If `delta_only` is true, returns +d/dR Tr(W_h * (h_X2C_no_ecp - T - V)).
 * If `delta_only` is false, returns +d/dR Tr(W_h * h_X2C_no_ecp) (excluding ECP).
 */
std::vector<double> contract_weighted_x2c1e_hcore_derivative(
    const X2c1eState& state, std::span<const double> hcore_weights, bool delta_only);

/** Evaluate full coordinate-major derivative matrices d(h_X2C_no_ecp)/dR or d(Delta h_SR)/dR.
 *
 * Output shape: (3 * natom) * n_contr * n_contr, coordinate-major.
 * Intended for verification/diagnostics and explicit Jacobian inspection.
 */
std::vector<double> build_x2c1e_hcore_derivatives(const X2c1eState& state, bool delta_only);

/** Add scalar-relativistic X2C-1e delta hcore (and optional derivatives) onto
 * an existing public-AO IntegralData when system.x2c1e_mode != None. */
void apply_x2c1e_to_integrals(const core::System& system, IntegralData& integrals,
                              bool include_derivatives);

/** Fast multi-threaded scalar McMurchie-Davidson CPU DF values (s through h). */
void build_df_integrals_fast(const core::System& orbital, const core::System& auxiliary,
                             std::span<double> metric_out, std::span<double> three_center_out);

/** Fast multi-threaded scalar McMurchie-Davidson CPU weighted DF derivative (s through h). */
std::vector<double> contract_weighted_df_derivative_fast(
    const core::System& orbital, const core::System& auxiliary,
    std::span<const double> metric_weights, std::span<const double> three_center_weights);

/** Fast scalar CPU 1e + ECP + X2C-1e weighted nuclear derivative. */
std::vector<double> contract_weighted_one_electron_fast(
    const core::System& system, std::span<const double> overlap_weights,
    std::span<const double> hcore_weights, bool include_nuclear_repulsion);

/** Fast scalar CPU 1e + ECP + X2C-1e values (overlap, hcore, nuclear repulsion). */
void build_one_electron_values_fast(const core::System& system, std::span<double> overlap_out,
                                    std::span<double> hcore_out, double& nuclear_repulsion_out);

}  // namespace generativeqc::integrals

#endif
