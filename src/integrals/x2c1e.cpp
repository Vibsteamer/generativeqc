#include "integrals/x2c1e.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "integrals/ecp.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "molecule/nuclear_gradient.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::integrals {
namespace {

constexpr double kOverlapRankTol = 1.0e-14;
constexpr double kKineticRankTol = 1.0e-14;
constexpr double kExpDedupRelTol = 1.0e-9;

double radial_primitive_normalization(double exponent, unsigned angular_momentum) {
  return std::pow(2.0 * exponent / std::numbers::pi, 0.75) *
         std::pow(4.0 * exponent, 0.5 * static_cast<double>(angular_momentum));
}

std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::overflow_error("X2C-1e tensor extent overflows size_t");
  return a * b;
}

void symmetrize_inplace(std::vector<double>& mat, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double avg = 0.5 * (mat[i * n + j] + mat[j * n + i]);
      mat[i * n + j] = avg;
      mat[j * n + i] = avg;
    }
  }
}

std::vector<double> symmetrized_copy(std::span<const double> mat, std::size_t n) {
  std::vector<double> out(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i * n + i] = mat[i * n + i];
    for (std::size_t j = i + 1; j < n; ++j) {
      const double avg = 0.5 * (mat[i * n + j] + mat[j * n + i]);
      out[i * n + j] = avg;
      out[j * n + i] = avg;
    }
  }
  return out;
}

tensor::CpuLinalgPlan x2c_linalg_plan() noexcept {
  tensor::CpuLinalgPlan plan;
  plan.provider = tensor::CpuLinalgProvider::automatic;
  plan.thread_ownership = tensor::CpuLinalgThreadOwnership::provider_parallel;
  plan.provider_threads = 1;
  return plan;
}

tensor::CpuSymmetricEigenResult x2c_sym_eigen(std::vector<double> matrix, std::size_t n) {
  if (tensor::cpu_openblas_lapack_built()) {
    return tensor::cpu_symmetric_eigen(std::move(matrix), n, x2c_linalg_plan());
  }
  tensor::CpuLinalgPlan scalar_plan;
  scalar_plan.provider = tensor::CpuLinalgProvider::scalar;
  scalar_plan.thread_ownership = tensor::CpuLinalgThreadOwnership::task_parallel;
  scalar_plan.provider_threads = 1;
  return tensor::cpu_symmetric_eigen(std::move(matrix), n, scalar_plan, 1.0e-15);
}

std::vector<double> matmul(char ta, char tb, std::size_t m, std::size_t n, std::size_t k,
                           const std::vector<double>& a, const std::vector<double>& b,
                           double alpha = 1.0) {
  std::vector<double> c(checked_mul(m, n), 0.0);
  if (m && n && k)
    tensor::cpu_gemm(ta, tb, m, n, k, a.data(), b.data(), c.data(), alpha, 0.0, x2c_linalg_plan());
  return c;
}

// Compute C^T * A * C where C is (rows x cols) and A is (rows x rows).
std::vector<double> congruence_ct_a_c(const std::vector<double>& c, std::size_t rows,
                                      std::size_t cols, const std::vector<double>& a) {
  auto tmp = matmul('N', 'N', rows, cols, rows, a, c);
  return matmul('T', 'N', cols, cols, rows, c, tmp);
}

// Compute C * A * C^T where C is (rows x cols) and A is (cols x cols).
std::vector<double> congruence_c_a_ct(const std::vector<double>& c, std::size_t rows,
                                      std::size_t cols, const std::vector<double>& a) {
  auto tmp = matmul('N', 'T', cols, rows, cols, a, c);
  return matmul('N', 'N', rows, rows, cols, c, tmp);
}

// Invert a square non-singular row-major matrix using Gaussian elimination with partial pivoting.
std::vector<double> invert_square_matrix(std::vector<double> a, std::size_t n) {
  std::vector<double> inv(checked_mul(n, n), 0.0);
  for (std::size_t i = 0; i < n; ++i) inv[i * n + i] = 1.0;
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot_row = col;
    double max_abs = std::abs(a[col * n + col]);
    for (std::size_t r = col + 1; r < n; ++r) {
      const double v = std::abs(a[r * n + col]);
      if (v > max_abs) {
        max_abs = v;
        pivot_row = r;
      }
    }
    if (!(max_abs > 0.0) || !std::isfinite(max_abs))
      throw std::runtime_error("X2C-1e singular matrix inversion in Dirac fold-down");
    if (pivot_row != col) {
      for (std::size_t j = 0; j < n; ++j) {
        std::swap(a[col * n + j], a[pivot_row * n + j]);
        std::swap(inv[col * n + j], inv[pivot_row * n + j]);
      }
    }
    const double diag = a[col * n + col];
    const double inv_diag = 1.0 / diag;
    for (std::size_t j = col; j < n; ++j) a[col * n + j] *= inv_diag;
    for (std::size_t j = 0; j < n; ++j) inv[col * n + j] *= inv_diag;
    for (std::size_t r = 0; r < n; ++r) {
      if (r == col) continue;
      const double factor = a[r * n + col];
      if (factor == 0.0) continue;
      for (std::size_t j = col; j < n; ++j) a[r * n + j] -= factor * a[col * n + j];
      for (std::size_t j = 0; j < n; ++j) inv[r * n + j] -= factor * inv[col * n + j];
    }
  }
  return inv;
}

// Pseudoinverse A^+ = (A^T A)^{-1} A^T for tall/rank-deficient A (m x k, m >= k).
std::vector<double> pseudoinverse_tall(const std::vector<double>& a, std::size_t m,
                                       std::size_t k) {
  auto ata = matmul('T', 'N', k, k, m, a, a);
  symmetrize_inplace(ata, k);
  const auto eig = tensor::cpu_symmetric_eigen(std::move(ata), k);
  std::vector<double> ata_inv(k * k, 0.0);
  for (std::size_t p = 0; p < k; ++p) {
    if (eig.values[p] <= 1.0e-14) continue;
    const double inv_ev = 1.0 / eig.values[p];
    for (std::size_t i = 0; i < k; ++i) {
      const double vi = eig.vectors[i * k + p] * inv_ev;
      for (std::size_t j = 0; j < k; ++j) ata_inv[i * k + j] += vi * eig.vectors[j * k + p];
    }
  }
  return matmul('N', 'T', k, m, k, ata_inv, a);
}

// ==================== Scalar McMurchie-Davidson 1e STVW Kernels ====================

void boys_values_scalar(unsigned max_order, double t, std::vector<double>& out) {
  out.resize(max_order + 1);
  const double exponential = std::exp(-t);
  if (t < static_cast<double>(max_order) + 16.0) {
    for (unsigned n = 0; n <= max_order; ++n) {
      double term = 1.0 / (2.0 * n + 1.0);
      double sum = term;
      for (unsigned k = 1; k < 512; ++k) {
        term *= 2.0 * t / (2.0 * n + 2.0 * k + 1.0);
        sum += term;
        if (term <= sum * 2.0e-16) break;
      }
      out[n] = exponential * sum;
    }
  } else {
    out[0] = 0.5 * std::sqrt(std::numbers::pi / t) * std::erf(std::sqrt(t));
    for (unsigned n = 1; n <= max_order; ++n)
      out[n] = ((2.0 * n - 1.0) * out[n - 1] - exponential) / (2.0 * t);
  }
}

struct ScalarHermite {
  unsigned jdim{};
  unsigned tdim{};
  std::vector<double> data;

  double& at(unsigned i, unsigned j, unsigned t) {
    return data[(static_cast<std::size_t>(i) * jdim + j) * tdim + t];
  }
  double get(int i, int j, int t) const noexcept {
    if (i < 0 || j < 0 || t < 0 || t > i + j) return 0.0;
    return data[(static_cast<std::size_t>(i) * jdim + static_cast<std::size_t>(j)) * tdim +
                static_cast<std::size_t>(t)];
  }
};

void fill_hermite_scalar(unsigned max_i, unsigned max_j, double p_coord, double a_coord,
                         double b_coord, double alpha, double beta, ScalarHermite& coeff) {
  const unsigned idim = max_i + 1;
  coeff.jdim = max_j + 1;
  coeff.tdim = max_i + max_j + 2;
  coeff.data.assign(static_cast<std::size_t>(idim) * coeff.jdim * coeff.tdim, 0.0);
  const double p = alpha + beta;
  const double mu = alpha * beta / p;
  const double ab = a_coord - b_coord;
  coeff.at(0, 0, 0) = std::exp(-mu * ab * ab);
  const double pa = p_coord - a_coord;
  const double pb = p_coord - b_coord;
  const double inv_2p = 0.5 / p;
  for (unsigned i = 0; i <= max_i; ++i) {
    for (unsigned j = 0; j <= max_j; ++j) {
      if (i == 0 && j == 0) continue;
      if (i > 0) {
        coeff.at(i, j, 0) = pa * coeff.at(i - 1, j, 0) + coeff.at(i - 1, j, 1);
      } else {
        coeff.at(i, j, 0) = pb * coeff.at(i, j - 1, 0) + coeff.at(i, j - 1, 1);
      }
      for (unsigned t = 1; t <= i + j; ++t) {
        if (i > 0) {
          coeff.at(i, j, t) = pa * coeff.at(i - 1, j, t) + inv_2p * coeff.at(i - 1, j, t - 1) +
                              static_cast<double>(t + 1) * coeff.at(i - 1, j, t + 1);
        } else {
          coeff.at(i, j, t) = pb * coeff.at(i, j - 1, t) + inv_2p * coeff.at(i, j - 1, t - 1) +
                              static_cast<double>(t + 1) * coeff.at(i, j - 1, t + 1);
        }
      }
    }
  }
}

struct ScalarCoulomb {
  unsigned dim{};
  std::vector<double> data;

  double& at(unsigned n, unsigned t, unsigned u, unsigned v) {
    return data[(((static_cast<std::size_t>(n) * dim + t) * dim + u) * dim) + v];
  }
  double at(unsigned n, unsigned t, unsigned u, unsigned v) const {
    return data[(((static_cast<std::size_t>(n) * dim + t) * dim + u) * dim) + v];
  }
};

void fill_coulomb_scalar(unsigned max_order, double p, const std::array<double, 3>& product,
                         const std::array<double, 3>& center, std::vector<double>& boys_buf,
                         ScalarCoulomb& aux) {
  const double dx = product[0] - center[0];
  const double dy = product[1] - center[1];
  const double dz = product[2] - center[2];
  const double arg = p * (dx * dx + dy * dy + dz * dz);
  boys_values_scalar(max_order, arg, boys_buf);
  aux.dim = max_order + 1;
  const std::size_t d = aux.dim;
  aux.data.assign(d * d * d * d, 0.0);
  double factor = 1.0;
  for (unsigned n = 0; n <= max_order; ++n) {
    aux.at(n, 0, 0, 0) = factor * boys_buf[n];
    factor *= -2.0 * p;
  }
  for (unsigned v = 1; v <= max_order; ++v) {
    for (unsigned n = 0; n + v <= max_order; ++n) {
      double val = dz * aux.at(n + 1, 0, 0, v - 1);
      if (v > 1) val += static_cast<double>(v - 1) * aux.at(n + 1, 0, 0, v - 2);
      aux.at(n, 0, 0, v) = val;
    }
  }
  for (unsigned v = 0; v <= max_order; ++v) {
    for (unsigned u = 1; u + v <= max_order; ++u) {
      for (unsigned n = 0; n + u + v <= max_order; ++n) {
        double val = dy * aux.at(n + 1, 0, u - 1, v);
        if (u > 1) val += static_cast<double>(u - 1) * aux.at(n + 1, 0, u - 2, v);
        aux.at(n, 0, u, v) = val;
      }
    }
  }
  for (unsigned v = 0; v <= max_order; ++v) {
    for (unsigned u = 0; u + v <= max_order; ++u) {
      for (unsigned t = 1; t + u + v <= max_order; ++t) {
        for (unsigned n = 0; n + t + u + v <= max_order; ++n) {
          double val = dx * aux.at(n + 1, t - 1, u, v);
          if (t > 1) val += static_cast<double>(t - 1) * aux.at(n + 1, t - 2, u, v);
          aux.at(n, t, u, v) = val;
        }
      }
    }
  }
}

struct ShellProjection {
  std::size_t ao_offset{};
  std::size_t n_ao{};
  std::size_t n_cart{};
  std::vector<molecule::CartesianComponent> cart_comps;
  // Row-major (n_ao x n_cart): transforms normalized-radially Cartesian primitives to public AOs.
  std::vector<double> cart_to_pub;
};

std::vector<ShellProjection> build_shell_projections(const core::System& system,
                                                     std::size_t& total_ao) {
  std::vector<ShellProjection> out(system.shells.size());
  std::size_t offset = 0;
  for (std::size_t s = 0; s < system.shells.size(); ++s) {
    const unsigned l = system.shells[s].angular_momentum;
    auto comps = molecule::cartesian_components(l);
    const auto exps = molecule::ao_expansions(l, system.basis_representation);
    ShellProjection proj;
    proj.ao_offset = offset;
    proj.n_ao = exps.size();
    proj.n_cart = comps.size();
    proj.cart_to_pub.assign(proj.n_ao * proj.n_cart, 0.0);
    for (std::size_t m = 0; m < proj.n_ao; ++m) {
      for (const auto& term : exps[m]) {
        for (std::size_t c = 0; c < proj.n_cart; ++c) {
          if (comps[c] == term.component) {
            proj.cart_to_pub[m * proj.n_cart + c] +=
                term.coefficient * molecule::cartesian_component_normalization(term.component);
            break;
          }
        }
      }
    }
    proj.cart_comps = std::move(comps);
    offset += proj.n_ao;
    out[s] = std::move(proj);
  }
  total_ao = offset;
  return out;
}

// 1D helper operators evaluated on ScalarHermite for bra i, ket j, Hermite index t.
inline double hermite_f(const ScalarHermite& h, int i, int j, int t, double alpha, double beta) {
  double v = 4.0 * alpha * beta * h.get(i + 1, j + 1, t);
  if (i > 0) v -= 2.0 * beta * static_cast<double>(i) * h.get(i - 1, j + 1, t);
  if (j > 0) v -= 2.0 * alpha * static_cast<double>(j) * h.get(i + 1, j - 1, t);
  if (i > 0 && j > 0) v += static_cast<double>(i * j) * h.get(i - 1, j - 1, t);
  return v;
}

inline double hermite_da_e(const ScalarHermite& h, int i, int j, int t, double alpha) {
  double v = 2.0 * alpha * h.get(i + 1, j, t);
  if (i > 0) v -= static_cast<double>(i) * h.get(i - 1, j, t);
  return v;
}

inline double hermite_d2a_e(const ScalarHermite& h, int i, int j, int t, double alpha) {
  // (-d^2/dx^2 g_i) * g_j = -i(i-1) E(i-2,j,t) + 2*a*(2i+1) E(i,j,t) - 4*a^2 E(i+2,j,t)
  double v = 2.0 * alpha * static_cast<double>(2 * i + 1) * h.get(i, j, t) -
             4.0 * alpha * alpha * h.get(i + 2, j, t);
  if (i >= 2) v -= static_cast<double>(i * (i - 1)) * h.get(i - 2, j, t);
  return v;
}

inline double hermite_da_f(const ScalarHermite& h, int i, int j, int t, double alpha, double beta) {
  // (d/dA_x (d/dx g_i)) * (d/dx g_j) = (-d^2/dx^2 g_i) * (j g_{j-1} - 2*b g_{j+1})
  double v = -2.0 * beta * hermite_d2a_e(h, i, j + 1, t, alpha);
  if (j > 0) v += static_cast<double>(j) * hermite_d2a_e(h, i, j - 1, t, alpha);
  return v;
}

inline double kinetic_1d(const ScalarHermite& h, int i, int j, double beta) {
  double v = beta * static_cast<double>(2 * j + 1) * h.get(i, j, 0) -
             2.0 * beta * beta * h.get(i, j + 2, 0);
  if (j >= 2) v -= 0.5 * static_cast<double>(j * (j - 1)) * h.get(i, j - 2, 0);
  return v;
}

inline double da_kinetic_1d(const ScalarHermite& h, int i, int j, double alpha, double beta) {
  double v = 2.0 * alpha * kinetic_1d(h, i + 1, j, beta);
  if (i > 0) v -= static_cast<double>(i) * kinetic_1d(h, i - 1, j, beta);
  return v;
}

// Contract public-AO symmetric weights (W_S, W_T, W_V, W_W) against exact nuclear derivatives
// of (S, T, V, W) in the system's public AO basis.
std::vector<double> contract_weighted_stvw_derivative(const core::System& system,
                                                      std::span<const double> w_s,
                                                      std::span<const double> w_t,
                                                      std::span<const double> w_v,
                                                      std::span<const double> w_w) {
  std::size_t nbf = 0;
  const auto projs = build_shell_projections(system, nbf);
  const std::size_t natom = system.atoms.size();
  const std::size_t ncoord = natom * 3;
  std::vector<double> grad(ncoord, 0.0);
  if (nbf == 0 || natom == 0) return grad;

  std::vector<double> boys_buf;
  ScalarCoulomb aux;
  std::array<ScalarHermite, 3> hermite;

  for (std::size_t si = 0; si < system.shells.size(); ++si) {
    const auto& shell_i = system.shells[si];
    const auto& proj_i = projs[si];
    const unsigned li = shell_i.angular_momentum;
    const auto& center_a = system.atoms[shell_i.atom_index].position;

    for (std::size_t sj = si; sj < system.shells.size(); ++sj) {
      const auto& shell_j = system.shells[sj];
      const auto& proj_j = projs[sj];
      const unsigned lj = shell_j.angular_momentum;
      const auto& center_b = system.atoms[shell_j.atom_index].position;
      const double sym_scale = (si == sj) ? 1.0 : 2.0;

      // Project public-AO symmetric weights into Cartesian component weights for (si, sj).
      const std::size_t nci = proj_i.n_cart;
      const std::size_t ncj = proj_j.n_cart;
      std::vector<double> ws_cart(nci * ncj, 0.0);
      std::vector<double> wt_cart(nci * ncj, 0.0);
      std::vector<double> wv_cart(nci * ncj, 0.0);
      std::vector<double> ww_cart(nci * ncj, 0.0);

      double max_st = 0.0;
      double max_vw = 0.0;
      for (std::size_t mi = 0; mi < proj_i.n_ao; ++mi) {
        const std::size_t gi = proj_i.ao_offset + mi;
        for (std::size_t mj = 0; mj < proj_j.n_ao; ++mj) {
          const std::size_t gj = proj_j.ao_offset + mj;
          const double s_val =
              sym_scale * 0.5 * (w_s[gi * nbf + gj] + w_s[gj * nbf + gi]);
          const double t_val =
              sym_scale * 0.5 * (w_t[gi * nbf + gj] + w_t[gj * nbf + gi]);
          const double v_val =
              sym_scale * 0.5 * (w_v[gi * nbf + gj] + w_v[gj * nbf + gi]);
          const double w_val =
              sym_scale * 0.5 * (w_w[gi * nbf + gj] + w_w[gj * nbf + gi]);
          if (s_val == 0.0 && t_val == 0.0 && v_val == 0.0 && w_val == 0.0) continue;
          for (std::size_t ci = 0; ci < nci; ++ci) {
            const double ti = proj_i.cart_to_pub[mi * nci + ci];
            if (ti == 0.0) continue;
            for (std::size_t cj = 0; cj < ncj; ++cj) {
              const double tj = proj_j.cart_to_pub[mj * ncj + cj];
              if (tj == 0.0) continue;
              const double prod = ti * tj;
              const std::size_t slot = ci * ncj + cj;
              ws_cart[slot] += prod * s_val;
              wt_cart[slot] += prod * t_val;
              wv_cart[slot] += prod * v_val;
              ww_cart[slot] += prod * w_val;
            }
          }
        }
      }
      for (std::size_t k = 0; k < nci * ncj; ++k) {
        max_st = std::max(max_st, std::max(std::abs(ws_cart[k]), std::abs(wt_cart[k])));
        max_vw = std::max(max_vw, std::max(std::abs(wv_cart[k]), std::abs(ww_cart[k])));
      }
      if (max_st == 0.0 && max_vw == 0.0) continue;

      const unsigned max_order = li + lj + 3;
      const unsigned rdim = max_order + 1;
      const std::size_t rsize = static_cast<std::size_t>(rdim) * rdim * rdim;
      std::vector<double> r_total(rsize, 0.0);
      std::vector<double> h_tuv(rsize, 0.0);
      auto r_idx = [rdim](unsigned t, unsigned u, unsigned v) -> std::size_t {
        return (static_cast<std::size_t>(t) * rdim + u) * rdim + v;
      };

      for (const auto& pi : shell_i.primitives) {
        for (const auto& pj : shell_j.primitives) {
          const double alpha = pi.exponent;
          const double beta = pj.exponent;
          const double p = alpha + beta;
          const double prim_coeff = pi.coefficient * pj.coefficient;
          if (prim_coeff == 0.0) continue;

          const std::array<double, 3> product{
              (alpha * center_a[0] + beta * center_b[0]) / p,
              (alpha * center_a[1] + beta * center_b[1]) / p,
              (alpha * center_a[2] + beta * center_b[2]) / p,
          };
          for (std::size_t ax = 0; ax < 3; ++ax) {
            fill_hermite_scalar(li + 2, lj + 2, product[ax], center_a[ax], center_b[ax], alpha,
                                beta, hermite[ax]);
          }

          // 1. S and T nuclear derivatives (translationally invariant: dB = -dA).
          if (max_st > 0.0 && shell_i.atom_index != shell_j.atom_index) {
            const double st_pref = prim_coeff * std::pow(std::numbers::pi / p, 1.5);
            std::array<double, 3> da_st{0.0, 0.0, 0.0};
            for (std::size_t ci = 0; ci < nci; ++ci) {
              const auto& ai = proj_i.cart_comps[ci];
              for (std::size_t cj = 0; cj < ncj; ++cj) {
                const auto& bj = proj_j.cart_comps[cj];
                const double ws = ws_cart[ci * ncj + cj];
                const double wt = wt_cart[ci * ncj + cj];
                if (ws == 0.0 && wt == 0.0) continue;
                const double ex0 = hermite[0].get(ai[0], bj[0], 0);
                const double ey0 = hermite[1].get(ai[1], bj[1], 0);
                const double ez0 = hermite[2].get(ai[2], bj[2], 0);
                const double dax = hermite_da_e(hermite[0], ai[0], bj[0], 0, alpha);
                const double day = hermite_da_e(hermite[1], ai[1], bj[1], 0, alpha);
                const double daz = hermite_da_e(hermite[2], ai[2], bj[2], 0, alpha);
                if (ws != 0.0) {
                  da_st[0] += ws * dax * ey0 * ez0;
                  da_st[1] += ws * ex0 * day * ez0;
                  da_st[2] += ws * ex0 * ey0 * daz;
                }
                if (wt != 0.0) {
                  const double kx = kinetic_1d(hermite[0], ai[0], bj[0], beta);
                  const double ky = kinetic_1d(hermite[1], ai[1], bj[1], beta);
                  const double kz = kinetic_1d(hermite[2], ai[2], bj[2], beta);
                  const double dakx = da_kinetic_1d(hermite[0], ai[0], bj[0], alpha, beta);
                  const double daky = da_kinetic_1d(hermite[1], ai[1], bj[1], alpha, beta);
                  const double dakz = da_kinetic_1d(hermite[2], ai[2], bj[2], alpha, beta);
                  da_st[0] += wt * (dakx * ey0 * ez0 + dax * ky * ez0 + dax * ey0 * kz);
                  da_st[1] += wt * (kx * day * ez0 + ex0 * daky * ez0 + ex0 * day * kz);
                  da_st[2] += wt * (kx * ey0 * daz + ex0 * ky * daz + ex0 * ey0 * dakz);
                }
              }
            }
            for (std::size_t ax = 0; ax < 3; ++ax) {
              const double v = st_pref * da_st[ax];
              grad[3 * shell_i.atom_index + ax] += v;
              grad[3 * shell_j.atom_index + ax] -= v;
            }
          }

          // 2. V and W nuclear derivatives via combined Hermite weights + translational identity.
          if (max_vw > 0.0) {
            const unsigned vw_max = li + lj + 2;
            std::fill(h_tuv.begin(), h_tuv.end(), 0.0);
            std::fill(r_total.begin(), r_total.end(), 0.0);

            // Build combined Hermite tensor H_{t,u,v} for nucleus-C derivatives.
            for (std::size_t ci = 0; ci < nci; ++ci) {
              const auto& ai = proj_i.cart_comps[ci];
              for (std::size_t cj = 0; cj < ncj; ++cj) {
                const auto& bj = proj_j.cart_comps[cj];
                const double wv = wv_cart[ci * ncj + cj];
                const double ww = ww_cart[ci * ncj + cj];
                if (wv == 0.0 && ww == 0.0) continue;
                const unsigned tmax = ai[0] + bj[0] + (ww != 0.0 ? 2 : 0);
                const unsigned umax = ai[1] + bj[1] + (ww != 0.0 ? 2 : 0);
                const unsigned vmax = ai[2] + bj[2] + (ww != 0.0 ? 2 : 0);
                for (unsigned t = 0; t <= tmax; ++t) {
                  const double ex = hermite[0].get(ai[0], bj[0], t);
                  const double fx =
                      ww != 0.0 ? hermite_f(hermite[0], ai[0], bj[0], t, alpha, beta) : 0.0;
                  if (ex == 0.0 && fx == 0.0) continue;
                  for (unsigned u = 0; u <= umax && t + u <= vw_max; ++u) {
                    const double ey = hermite[1].get(ai[1], bj[1], u);
                    const double fy =
                        ww != 0.0 ? hermite_f(hermite[1], ai[1], bj[1], u, alpha, beta) : 0.0;
                    if (ey == 0.0 && fy == 0.0) continue;
                    const double c_e = wv * ex * ey;
                    const double c_w_xy = ww * (fx * ey + ex * fy);
                    const double c_w_z = ww * ex * ey;
                    for (unsigned v = 0; v <= vmax && t + u + v <= vw_max; ++v) {
                      const double ez = hermite[2].get(ai[2], bj[2], v);
                      double contrib = (c_e + c_w_xy) * ez;
                      if (ww != 0.0 && c_w_z != 0.0) {
                        contrib +=
                            c_w_z * hermite_f(hermite[2], ai[2], bj[2], v, alpha, beta);
                      }
                      h_tuv[r_idx(t, u, v)] += contrib;
                    }
                  }
                }
              }
            }

            // Sum over nuclei C: accumulate R_total and exact d/dC_gamma.
            const double two_pi_over_p = 2.0 * std::numbers::pi / p;
            std::array<double, 3> sum_dc{0.0, 0.0, 0.0};
            for (std::size_t atom_c = 0; atom_c < natom; ++atom_c) {
              const double zc = static_cast<double>(system.atoms[atom_c].ionic_charge());
              if (zc == 0.0) continue;
              fill_coulomb_scalar(max_order, p, product, system.atoms[atom_c].position, boys_buf,
                                  aux);
              const double neg_z_pref = -zc * two_pi_over_p;
              const double pos_z_pref = zc * two_pi_over_p * prim_coeff;
              std::array<double, 3> dc{0.0, 0.0, 0.0};
              for (unsigned t = 0; t <= max_order; ++t) {
                for (unsigned u = 0; t + u <= max_order; ++u) {
                  for (unsigned v = 0; t + u + v <= max_order; ++v) {
                    r_total[r_idx(t, u, v)] += neg_z_pref * aux.at(0, t, u, v);
                    if (t + u + v <= vw_max) {
                      const double hval = h_tuv[r_idx(t, u, v)];
                      if (hval != 0.0) {
                        dc[0] += hval * aux.at(0, t + 1, u, v);
                        dc[1] += hval * aux.at(0, t, u + 1, v);
                        dc[2] += hval * aux.at(0, t, u, v + 1);
                      }
                    }
                  }
                }
              }
              for (std::size_t ax = 0; ax < 3; ++ax) {
                const double val = pos_z_pref * dc[ax];
                grad[3 * atom_c + ax] += val;
                sum_dc[ax] += val;
              }
            }

            // Bra derivative d/dA_gamma contracted against R_total; then d/dB = -d/dA - sum_C d/dC.
            std::array<double, 3> da{0.0, 0.0, 0.0};
            if (shell_i.atom_index != shell_j.atom_index) {
              for (std::size_t ci = 0; ci < nci; ++ci) {
                const auto& ai = proj_i.cart_comps[ci];
                for (std::size_t cj = 0; cj < ncj; ++cj) {
                  const auto& bj = proj_j.cart_comps[cj];
                  const double wv = wv_cart[ci * ncj + cj];
                  const double ww = ww_cart[ci * ncj + cj];
                  if (wv == 0.0 && ww == 0.0) continue;
                  const unsigned tmax = ai[0] + bj[0] + (ww != 0.0 ? 3 : 1);
                  const unsigned umax = ai[1] + bj[1] + (ww != 0.0 ? 3 : 1);
                  const unsigned vmax = ai[2] + bj[2] + (ww != 0.0 ? 3 : 1);
                  double gx = 0.0, gy = 0.0, gz = 0.0;
                  for (unsigned t = 0; t <= tmax; ++t) {
                    const double ex = hermite[0].get(ai[0], bj[0], t);
                    const double dax = hermite_da_e(hermite[0], ai[0], bj[0], t, alpha);
                    const double fx =
                        ww != 0.0 ? hermite_f(hermite[0], ai[0], bj[0], t, alpha, beta) : 0.0;
                    const double dafx =
                        ww != 0.0 ? hermite_da_f(hermite[0], ai[0], bj[0], t, alpha, beta) : 0.0;
                    if (ex == 0.0 && dax == 0.0 && fx == 0.0 && dafx == 0.0) continue;
                    for (unsigned u = 0; u <= umax; ++u) {
                      const double ey = hermite[1].get(ai[1], bj[1], u);
                      const double day = hermite_da_e(hermite[1], ai[1], bj[1], u, alpha);
                      const double fy =
                          ww != 0.0 ? hermite_f(hermite[1], ai[1], bj[1], u, alpha, beta) : 0.0;
                      const double dafy =
                          ww != 0.0 ? hermite_da_f(hermite[1], ai[1], bj[1], u, alpha, beta) : 0.0;
                      if (ey == 0.0 && day == 0.0 && fy == 0.0 && dafy == 0.0) continue;
                      const double x_ez = wv * dax * ey + ww * (dafx * ey + dax * fy);
                      const double x_fz = ww * dax * ey;
                      const double y_ez = wv * ex * day + ww * (fx * day + ex * dafy);
                      const double y_fz = ww * ex * day;
                      const double z_daz = wv * ex * ey + ww * (fx * ey + ex * fy);
                      const double z_dafz = ww * ex * ey;
                      for (unsigned v = 0; v <= vmax && t + u + v <= max_order; ++v) {
                        const double rval = r_total[r_idx(t, u, v)];
                        if (rval == 0.0) continue;
                        const double ez = hermite[2].get(ai[2], bj[2], v);
                        const double daz = hermite_da_e(hermite[2], ai[2], bj[2], v, alpha);
                        const double fz =
                            ww != 0.0 ? hermite_f(hermite[2], ai[2], bj[2], v, alpha, beta) : 0.0;
                        const double dafz =
                            ww != 0.0 ? hermite_da_f(hermite[2], ai[2], bj[2], v, alpha, beta)
                                      : 0.0;
                        gx += rval * (x_ez * ez + x_fz * fz);
                        gy += rval * (y_ez * ez + y_fz * fz);
                        gz += rval * (z_daz * daz + z_dafz * dafz);
                      }
                    }
                  }
                  da[0] += gx;
                  da[1] += gy;
                  da[2] += gz;
                }
              }
              for (std::size_t ax = 0; ax < 3; ++ax) {
                const double va = prim_coeff * da[ax];
                grad[3 * shell_i.atom_index + ax] += va;
                grad[3 * shell_j.atom_index + ax] += (-va - sum_dc[ax]);
              }
            } else {
              // Same center A == B: d/dA + d/dB = -sum_C d/dC identically.
              for (std::size_t ax = 0; ax < 3; ++ax) {
                grad[3 * shell_i.atom_index + ax] -= sum_dc[ax];
              }
            }
          }
        }
      }
    }
  }
  return grad;
}

struct StvwDerivatives {
  std::size_t nbf{};
  std::size_t ncoord{};
  std::vector<double> d_overlap;
  std::vector<double> d_kinetic;
  std::vector<double> d_attraction;
  std::vector<double> d_small_potential;
};

StvwDerivatives build_stvw_derivatives(const core::System& system) {
  std::size_t nbf = 0;
  const auto projs = build_shell_projections(system, nbf);
  const std::size_t natom = system.atoms.size();
  const std::size_t ncoord = natom * 3;
  const std::size_t n2 = checked_mul(nbf, nbf);
  StvwDerivatives out;
  out.nbf = nbf;
  out.ncoord = ncoord;
  out.d_overlap.assign(checked_mul(ncoord, n2), 0.0);
  out.d_kinetic.assign(checked_mul(ncoord, n2), 0.0);
  out.d_attraction.assign(checked_mul(ncoord, n2), 0.0);
  out.d_small_potential.assign(checked_mul(ncoord, n2), 0.0);
  if (nbf == 0 || natom == 0) return out;

  std::vector<double> boys_buf;
  ScalarCoulomb aux;
  std::array<ScalarHermite, 3> hermite;

  for (std::size_t si = 0; si < system.shells.size(); ++si) {
    const auto& shell_i = system.shells[si];
    const auto& proj_i = projs[si];
    const unsigned li = shell_i.angular_momentum;
    const auto& center_a = system.atoms[shell_i.atom_index].position;

    for (std::size_t sj = si; sj < system.shells.size(); ++sj) {
      const auto& shell_j = system.shells[sj];
      const auto& proj_j = projs[sj];
      const unsigned lj = shell_j.angular_momentum;
      const auto& center_b = system.atoms[shell_j.atom_index].position;
      const std::size_t nci = proj_i.n_cart;
      const std::size_t ncj = proj_j.n_cart;
      const std::size_t nc_pair = nci * ncj;

      // Coordinate-major Cartesian derivative blocks for this shell pair.
      std::vector<double> ds_c(ncoord * nc_pair, 0.0);
      std::vector<double> dt_c(ncoord * nc_pair, 0.0);
      std::vector<double> dv_c(ncoord * nc_pair, 0.0);
      std::vector<double> dw_c(ncoord * nc_pair, 0.0);

      const unsigned max_order = li + lj + 3;
      const unsigned rdim = max_order + 1;
      const std::size_t rsize = static_cast<std::size_t>(rdim) * rdim * rdim;
      std::vector<double> r_total(rsize, 0.0);
      auto r_idx = [rdim](unsigned t, unsigned u, unsigned v) -> std::size_t {
        return (static_cast<std::size_t>(t) * rdim + u) * rdim + v;
      };

      for (const auto& pi : shell_i.primitives) {
        for (const auto& pj : shell_j.primitives) {
          const double alpha = pi.exponent;
          const double beta = pj.exponent;
          const double p = alpha + beta;
          const double prim_coeff = pi.coefficient * pj.coefficient;
          if (prim_coeff == 0.0) continue;

          const std::array<double, 3> product{
              (alpha * center_a[0] + beta * center_b[0]) / p,
              (alpha * center_a[1] + beta * center_b[1]) / p,
              (alpha * center_a[2] + beta * center_b[2]) / p,
          };
          for (std::size_t ax = 0; ax < 3; ++ax) {
            fill_hermite_scalar(li + 2, lj + 2, product[ax], center_a[ax], center_b[ax], alpha,
                                beta, hermite[ax]);
          }

          // S and T derivatives
          if (shell_i.atom_index != shell_j.atom_index) {
            const double st_pref = prim_coeff * std::pow(std::numbers::pi / p, 1.5);
            for (std::size_t ci = 0; ci < nci; ++ci) {
              const auto& ai = proj_i.cart_comps[ci];
              for (std::size_t cj = 0; cj < ncj; ++cj) {
                const auto& bj = proj_j.cart_comps[cj];
                const std::size_t slot = ci * ncj + cj;
                const double ex0 = hermite[0].get(ai[0], bj[0], 0);
                const double ey0 = hermite[1].get(ai[1], bj[1], 0);
                const double ez0 = hermite[2].get(ai[2], bj[2], 0);
                const double dax = hermite_da_e(hermite[0], ai[0], bj[0], 0, alpha);
                const double day = hermite_da_e(hermite[1], ai[1], bj[1], 0, alpha);
                const double daz = hermite_da_e(hermite[2], ai[2], bj[2], 0, alpha);
                const double kx = kinetic_1d(hermite[0], ai[0], bj[0], beta);
                const double ky = kinetic_1d(hermite[1], ai[1], bj[1], beta);
                const double kz = kinetic_1d(hermite[2], ai[2], bj[2], beta);
                const double dakx = da_kinetic_1d(hermite[0], ai[0], bj[0], alpha, beta);
                const double daky = da_kinetic_1d(hermite[1], ai[1], bj[1], alpha, beta);
                const double dakz = da_kinetic_1d(hermite[2], ai[2], bj[2], alpha, beta);
                const std::array<double, 3> ds_a{st_pref * dax * ey0 * ez0,
                                                 st_pref * ex0 * day * ez0,
                                                 st_pref * ex0 * ey0 * daz};
                const std::array<double, 3> dt_a{
                    st_pref * (dakx * ey0 * ez0 + dax * ky * ez0 + dax * ey0 * kz),
                    st_pref * (kx * day * ez0 + ex0 * daky * ez0 + ex0 * day * kz),
                    st_pref * (kx * ey0 * daz + ex0 * ky * daz + ex0 * ey0 * dakz)};
                for (std::size_t ax = 0; ax < 3; ++ax) {
                  ds_c[(3 * shell_i.atom_index + ax) * nc_pair + slot] += ds_a[ax];
                  ds_c[(3 * shell_j.atom_index + ax) * nc_pair + slot] -= ds_a[ax];
                  dt_c[(3 * shell_i.atom_index + ax) * nc_pair + slot] += dt_a[ax];
                  dt_c[(3 * shell_j.atom_index + ax) * nc_pair + slot] -= dt_a[ax];
                }
              }
            }
          }

          // V and W derivatives: nuclei C first, accumulating sum_C and R_total.
          const double two_pi_over_p = 2.0 * std::numbers::pi / p;
          std::fill(r_total.begin(), r_total.end(), 0.0);
          std::vector<double> sum_dvc(3 * nc_pair, 0.0);
          std::vector<double> sum_dwc(3 * nc_pair, 0.0);

          for (std::size_t atom_c = 0; atom_c < natom; ++atom_c) {
            const double zc = static_cast<double>(system.atoms[atom_c].ionic_charge());
            if (zc == 0.0) continue;
            fill_coulomb_scalar(max_order, p, product, system.atoms[atom_c].position, boys_buf,
                                aux);
            const double neg_z_pref = -zc * two_pi_over_p;
            const double pos_z_pref = zc * two_pi_over_p * prim_coeff;
            for (unsigned t = 0; t <= max_order; ++t)
              for (unsigned u = 0; t + u <= max_order; ++u)
                for (unsigned v = 0; t + u + v <= max_order; ++v)
                  r_total[r_idx(t, u, v)] += neg_z_pref * aux.at(0, t, u, v);

            for (std::size_t ci = 0; ci < nci; ++ci) {
              const auto& ai = proj_i.cart_comps[ci];
              for (std::size_t cj = 0; cj < ncj; ++cj) {
                const auto& bj = proj_j.cart_comps[cj];
                const std::size_t slot = ci * ncj + cj;
                const unsigned tmax = ai[0] + bj[0] + 2;
                const unsigned umax = ai[1] + bj[1] + 2;
                const unsigned vmax = ai[2] + bj[2] + 2;
                std::array<double, 3> vc{0.0, 0.0, 0.0};
                std::array<double, 3> wc{0.0, 0.0, 0.0};
                for (unsigned t = 0; t <= tmax; ++t) {
                  const double ex = hermite[0].get(ai[0], bj[0], t);
                  const double fx = hermite_f(hermite[0], ai[0], bj[0], t, alpha, beta);
                  if (ex == 0.0 && fx == 0.0) continue;
                  for (unsigned u = 0; u <= umax; ++u) {
                    const double ey = hermite[1].get(ai[1], bj[1], u);
                    const double fy = hermite_f(hermite[1], ai[1], bj[1], u, alpha, beta);
                    if (ey == 0.0 && fy == 0.0) continue;
                    const double exey = ex * ey;
                    const double f_xy = fx * ey + ex * fy;
                    for (unsigned v = 0; v <= vmax && t + u + v <= li + lj + 2; ++v) {
                      const double ez = hermite[2].get(ai[2], bj[2], v);
                      const double fz = hermite_f(hermite[2], ai[2], bj[2], v, alpha, beta);
                      const double ev = exey * ez;
                      const double ew = f_xy * ez + exey * fz;
                      if (ev != 0.0) {
                        vc[0] += ev * aux.at(0, t + 1, u, v);
                        vc[1] += ev * aux.at(0, t, u + 1, v);
                        vc[2] += ev * aux.at(0, t, u, v + 1);
                      }
                      if (ew != 0.0) {
                        wc[0] += ew * aux.at(0, t + 1, u, v);
                        wc[1] += ew * aux.at(0, t, u + 1, v);
                        wc[2] += ew * aux.at(0, t, u, v + 1);
                      }
                    }
                  }
                }
                for (std::size_t ax = 0; ax < 3; ++ax) {
                  const double vval = pos_z_pref * vc[ax];
                  const double wval = pos_z_pref * wc[ax];
                  dv_c[(3 * atom_c + ax) * nc_pair + slot] += vval;
                  dw_c[(3 * atom_c + ax) * nc_pair + slot] += wval;
                  sum_dvc[ax * nc_pair + slot] += vval;
                  sum_dwc[ax * nc_pair + slot] += wval;
                }
              }
            }
          }

          // Bra derivative d/dA and translational d/dB = -d/dA - sum_C d/dC.
          if (shell_i.atom_index != shell_j.atom_index) {
            for (std::size_t ci = 0; ci < nci; ++ci) {
              const auto& ai = proj_i.cart_comps[ci];
              for (std::size_t cj = 0; cj < ncj; ++cj) {
                const auto& bj = proj_j.cart_comps[cj];
                const std::size_t slot = ci * ncj + cj;
                const unsigned tmax = ai[0] + bj[0] + 3;
                const unsigned umax = ai[1] + bj[1] + 3;
                const unsigned vmax = ai[2] + bj[2] + 3;
                std::array<double, 3> va{0.0, 0.0, 0.0};
                std::array<double, 3> wa{0.0, 0.0, 0.0};
                for (unsigned t = 0; t <= tmax; ++t) {
                  const double ex = hermite[0].get(ai[0], bj[0], t);
                  const double dax = hermite_da_e(hermite[0], ai[0], bj[0], t, alpha);
                  const double fx = hermite_f(hermite[0], ai[0], bj[0], t, alpha, beta);
                  const double dafx = hermite_da_f(hermite[0], ai[0], bj[0], t, alpha, beta);
                  if (ex == 0.0 && dax == 0.0 && fx == 0.0 && dafx == 0.0) continue;
                  for (unsigned u = 0; u <= umax; ++u) {
                    const double ey = hermite[1].get(ai[1], bj[1], u);
                    const double day = hermite_da_e(hermite[1], ai[1], bj[1], u, alpha);
                    const double fy = hermite_f(hermite[1], ai[1], bj[1], u, alpha, beta);
                    const double dafy = hermite_da_f(hermite[1], ai[1], bj[1], u, alpha, beta);
                    if (ey == 0.0 && day == 0.0 && fy == 0.0 && dafy == 0.0) continue;
                    for (unsigned v = 0; v <= vmax && t + u + v <= max_order; ++v) {
                      const double rval = r_total[r_idx(t, u, v)];
                      if (rval == 0.0) continue;
                      const double ez = hermite[2].get(ai[2], bj[2], v);
                      const double daz = hermite_da_e(hermite[2], ai[2], bj[2], v, alpha);
                      const double fz = hermite_f(hermite[2], ai[2], bj[2], v, alpha, beta);
                      const double dafz = hermite_da_f(hermite[2], ai[2], bj[2], v, alpha, beta);
                      va[0] += rval * dax * ey * ez;
                      va[1] += rval * ex * day * ez;
                      va[2] += rval * ex * ey * daz;
                      wa[0] += rval * (dafx * ey * ez + dax * fy * ez + dax * ey * fz);
                      wa[1] += rval * (fx * day * ez + ex * dafy * ez + ex * day * fz);
                      wa[2] += rval * (fx * ey * daz + ex * fy * daz + ex * ey * dafz);
                    }
                  }
                }
                for (std::size_t ax = 0; ax < 3; ++ax) {
                  const double vval = prim_coeff * va[ax];
                  const double wval = prim_coeff * wa[ax];
                  dv_c[(3 * shell_i.atom_index + ax) * nc_pair + slot] += vval;
                  dv_c[(3 * shell_j.atom_index + ax) * nc_pair + slot] +=
                      (-vval - sum_dvc[ax * nc_pair + slot]);
                  dw_c[(3 * shell_i.atom_index + ax) * nc_pair + slot] += wval;
                  dw_c[(3 * shell_j.atom_index + ax) * nc_pair + slot] +=
                      (-wval - sum_dwc[ax * nc_pair + slot]);
                }
              }
            }
          } else {
            for (std::size_t ax = 0; ax < 3; ++ax) {
              for (std::size_t slot = 0; slot < nc_pair; ++slot) {
                dv_c[(3 * shell_i.atom_index + ax) * nc_pair + slot] -=
                    sum_dvc[ax * nc_pair + slot];
                dw_c[(3 * shell_i.atom_index + ax) * nc_pair + slot] -=
                    sum_dwc[ax * nc_pair + slot];
              }
            }
          }
        }
      }

      // Transform Cartesian derivative blocks to public AOs and scatter into output (both triangles).
      auto scatter = [&](const std::vector<double>& src, std::vector<double>& dst) {
        for (std::size_t coord = 0; coord < ncoord; ++coord) {
          const double* block = src.data() + coord * nc_pair;
          double* target = dst.data() + coord * n2;
          for (std::size_t mi = 0; mi < proj_i.n_ao; ++mi) {
            const std::size_t gi = proj_i.ao_offset + mi;
            for (std::size_t mj = 0; mj < proj_j.n_ao; ++mj) {
              const std::size_t gj = proj_j.ao_offset + mj;
              double val = 0.0;
              for (std::size_t ci = 0; ci < nci; ++ci) {
                const double ti = proj_i.cart_to_pub[mi * nci + ci];
                if (ti == 0.0) continue;
                for (std::size_t cj = 0; cj < ncj; ++cj) {
                  val += ti * proj_j.cart_to_pub[mj * ncj + cj] * block[ci * ncj + cj];
                }
              }
              target[gi * nbf + gj] = val;
              if (si != sj) target[gj * nbf + gi] = val;
            }
          }
        }
      };
      scatter(ds_c, out.d_overlap);
      scatter(dt_c, out.d_kinetic);
      scatter(dv_c, out.d_attraction);
      scatter(dw_c, out.d_small_potential);
    }
  }
  return out;
}

// ==================== Decontraction ====================

struct DecontractedBasis {
  core::System uncontracted;
  std::size_t n_contr{};
  std::size_t n_prim{};
  // Row-major (n_prim x n_contr) contraction matrix C: M_contr = C^T M_prim C.
  std::vector<double> contraction;
  // Per-atom contiguous primitive AO slice [start, end) and AE flag.
  std::vector<std::size_t> atom_ao_start;
  std::vector<std::size_t> atom_ao_end;
  std::vector<bool> atom_is_ae;
  bool has_ae_atoms{};
};

DecontractedBasis decontract_system(const core::System& system, X2c1eMode mode) {
  DecontractedBasis out;
  const std::size_t natom = system.atoms.size();
  out.n_contr = molecule::ao_count(system);
  out.uncontracted.atoms = system.atoms;
  out.uncontracted.charge = system.charge;
  out.uncontracted.multiplicity = system.multiplicity;
  out.uncontracted.electron_count = system.electron_count;
  out.uncontracted.basis_representation = system.basis_representation;
  // Uncontracted system never includes ECP terms or recursive X2C-1e flags.
  out.uncontracted.ecp_terms.clear();
  out.uncontracted.x2c1e_mode = 0;
  out.uncontracted.x2c1e_speed_of_light = kSpeedOfLightAu;

  // Record original shell offsets.
  std::vector<std::size_t> orig_offset(system.shells.size());
  std::size_t cursor = 0;
  for (std::size_t s = 0; s < system.shells.size(); ++s) {
    orig_offset[s] = cursor;
    cursor += molecule::shell_ao_count(system.shells[s].angular_momentum,
                                       system.basis_representation);
  }

  out.atom_ao_start.assign(natom, 0);
  out.atom_ao_end.assign(natom, 0);
  out.atom_is_ae.assign(natom, false);

  struct PrimEntry {
    std::size_t prim_ao_offset;
    std::size_t orig_ao_offset;
    std::size_t n_ao;
    double coeff;
  };
  std::vector<PrimEntry> entries;
  std::size_t prim_cursor = 0;

  for (std::size_t atom = 0; atom < natom; ++atom) {
    const bool is_ecp = system.atoms[atom].ecp_core > 0;
    const bool is_ae = !is_ecp;
    out.atom_is_ae[atom] = is_ae;
    if (is_ae) out.has_ae_atoms = true;
    out.atom_ao_start[atom] = prim_cursor;

    if (is_ecp && mode != X2c1eMode::Global1e) {
      // Keep ECP atoms contracted identically (1-to-1 identity block in contraction).
      for (std::size_t s = 0; s < system.shells.size(); ++s) {
        if (system.shells[s].atom_index != atom) continue;
        const std::size_t nao = molecule::shell_ao_count(system.shells[s].angular_momentum,
                                                         system.basis_representation);
        out.uncontracted.shells.push_back(system.shells[s]);
        entries.push_back({prim_cursor, orig_offset[s], nao, 1.0});
        prim_cursor += nao;
      }
      out.atom_ao_end[atom] = prim_cursor;
      continue;
    }

    // Group AE atom shells by angular momentum l.
    std::map<unsigned, std::vector<std::size_t>> shells_by_l;
    for (std::size_t s = 0; s < system.shells.size(); ++s) {
      if (system.shells[s].atom_index == atom) {
        shells_by_l[system.shells[s].angular_momentum].push_back(s);
      }
    }

    for (const auto& [l, s_indices] : shells_by_l) {
      const std::size_t nao = molecule::shell_ao_count(l, system.basis_representation);
      // Collect all primitive exponents on (atom, l) and deduplicate matching PySCF round(9).
      std::vector<double> all_exps;
      for (std::size_t s : s_indices) {
        for (const auto& p : system.shells[s].primitives) all_exps.push_back(p.exponent);
      }
      std::sort(all_exps.begin(), all_exps.end(), std::greater<double>());
      std::vector<double> uniq_exps;
      for (double e : all_exps) {
        if (uniq_exps.empty() ||
            std::abs(e - uniq_exps.back()) > kExpDedupRelTol * std::max(1.0, uniq_exps.back())) {
          uniq_exps.push_back(e);
        }
      }

      const std::size_t l_prim_base = prim_cursor;
      for (double e : uniq_exps) {
        core::Shell pshell;
        pshell.atom_index = static_cast<std::uint32_t>(atom);
        pshell.angular_momentum = l;
        pshell.primitives.push_back({e, radial_primitive_normalization(e, l)});
        out.uncontracted.shells.push_back(std::move(pshell));
        prim_cursor += nao;
      }

      for (std::size_t s : s_indices) {
        for (const auto& p : system.shells[s].primitives) {
          // Find matching unique exponent index.
          std::size_t match = uniq_exps.size();
          double best_diff = std::numeric_limits<double>::infinity();
          for (std::size_t u = 0; u < uniq_exps.size(); ++u) {
            const double d = std::abs(p.exponent - uniq_exps[u]);
            if (d < best_diff) {
              best_diff = d;
              match = u;
            }
          }
          const double rad_norm = radial_primitive_normalization(uniq_exps[match], l);
          entries.push_back(
              {l_prim_base + match * nao, orig_offset[s], nao, p.coefficient / rad_norm});
        }
      }
    }
    out.atom_ao_end[atom] = prim_cursor;
  }

  out.n_prim = prim_cursor;
  out.contraction.assign(checked_mul(out.n_prim, out.n_contr), 0.0);
  for (const auto& e : entries) {
    for (std::size_t m = 0; m < e.n_ao; ++m) {
      out.contraction[(e.prim_ao_offset + m) * out.n_contr + (e.orig_ao_offset + m)] += e.coeff;
    }
  }
  return out;
}

// ==================== Dirac Generalized Eigensolver & R Metric ====================

struct DiracSolution {
  std::size_t n_neg{};
  std::size_t n_pos{};
  std::vector<double> evals_neg;
  std::vector<double> evals_pos;
  std::vector<double> evecs_neg;  // (2*n) x n_neg
  std::vector<double> evecs_pos;  // (2*n) x n_pos
  std::vector<double> al_inv;     // n_pos x n
  std::vector<double> x_mat;      // n x n
};

DiracSolution solve_dirac_1e(const std::vector<double>& s, const std::vector<double>& t,
                             const std::vector<double>& v, const std::vector<double>& w,
                             std::size_t n, double c) {
  const double root2_c = std::sqrt(2.0) * c;
  const double two_c2 = 2.0 * c * c;

  std::vector<double> ls = s;
  std::vector<double> lt = t;
  const int info_s = tensor::cpu_cholesky_lower(ls.data(), n, x2c_linalg_plan());
  const int info_t = tensor::cpu_cholesky_lower(lt.data(), n, x2c_linalg_plan());

  std::vector<double> evals;
  std::vector<double> cl_all;  // n x m_tot
  std::vector<double> cs_all;  // n x m_tot
  std::size_t m_tot = 0;

  if (info_s == 0 && info_t == 0) {
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j) {
        ls[i * n + j] = 0.0;
        lt[i * n + j] = 0.0;
      }
    const auto plan = x2c_linalg_plan();
    std::vector<double> h11 = v;
    tensor::cpu_trsm('L', 'L', 'N', 'N', n, n, ls.data(), h11.data(), 1.0, plan);
    tensor::cpu_trsm('R', 'L', 'T', 'N', n, n, ls.data(), h11.data(), 1.0, plan);

    std::vector<double> h12 = lt;
    tensor::cpu_trsm('L', 'L', 'N', 'N', n, n, ls.data(), h12.data(), root2_c, plan);

    std::vector<double> h22 = w;
    tensor::cpu_trsm('L', 'L', 'N', 'N', n, n, lt.data(), h22.data(), 0.5, plan);
    tensor::cpu_trsm('R', 'L', 'T', 'N', n, n, lt.data(), h22.data(), 1.0, plan);
    for (std::size_t i = 0; i < n; ++i) h22[i * n + i] -= two_c2;

    m_tot = 2 * n;
    std::vector<double> htil(m_tot * m_tot, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        htil[i * m_tot + j] = h11[i * n + j];
        htil[i * m_tot + (n + j)] = h12[i * n + j];
        htil[(n + i) * m_tot + j] = h12[j * n + i];
        htil[(n + i) * m_tot + (n + j)] = h22[i * n + j];
      }
    }
    symmetrize_inplace(htil, m_tot);
    auto eig = tensor::cpu_symmetric_eigen(std::move(htil), m_tot, plan);
    evals = std::move(eig.values);
    cl_all.resize(n * m_tot);
    cs_all.resize(n * m_tot);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < m_tot; ++j) {
        cl_all[i * m_tot + j] = eig.vectors[i * m_tot + j];
        cs_all[i * m_tot + j] = eig.vectors[(n + i) * m_tot + j];
      }
    }
    tensor::cpu_trsm('L', 'L', 'T', 'N', n, m_tot, ls.data(), cl_all.data(), 1.0, plan);
    tensor::cpu_trsm('L', 'L', 'T', 'N', n, m_tot, lt.data(), cs_all.data(), root2_c, plan);
  } else {
    // Canonical orthogonalization fallback for linearly dependent uncontracted metrics.
    const auto plan = x2c_linalg_plan();
    const auto eig_s = tensor::cpu_symmetric_eigen(s, n, plan);
    const auto eig_t = tensor::cpu_symmetric_eigen(t, n, plan);
    std::vector<std::size_t> s_keep, t_keep;
    for (std::size_t i = 0; i < n; ++i) {
      if (eig_s.values[i] > kOverlapRankTol) s_keep.push_back(i);
      if (eig_t.values[i] > kKineticRankTol) t_keep.push_back(i);
    }
    const std::size_t rs = s_keep.size();
    const std::size_t rt = t_keep.size();
    std::vector<double> us(n * rs), ut(n * rt);
    for (std::size_t k = 0; k < rs; ++k) {
      const std::size_t idx = s_keep[k];
      const double inv_sq = 1.0 / std::sqrt(eig_s.values[idx]);
      for (std::size_t i = 0; i < n; ++i) us[i * rs + k] = eig_s.vectors[i * n + idx] * inv_sq;
    }
    for (std::size_t k = 0; k < rt; ++k) {
      const std::size_t idx = t_keep[k];
      const double inv_sq = 1.0 / std::sqrt(eig_t.values[idx]);
      for (std::size_t i = 0; i < n; ++i) ut[i * rt + k] = eig_t.vectors[i * n + idx] * inv_sq;
    }
    auto h11 = congruence_ct_a_c(us, n, rs, v);
    auto tut = matmul('N', 'N', n, rt, n, t, ut);
    auto h12 = matmul('T', 'N', rs, rt, n, us, tut, root2_c);
    auto h22 = congruence_ct_a_c(ut, n, rt, w);
    for (double& x : h22) x *= 0.5;
    for (std::size_t i = 0; i < rt; ++i) h22[i * rt + i] -= two_c2;

    m_tot = rs + rt;
    std::vector<double> htil(m_tot * m_tot, 0.0);
    for (std::size_t i = 0; i < rs; ++i) {
      for (std::size_t j = 0; j < rs; ++j) htil[i * m_tot + j] = h11[i * rs + j];
      for (std::size_t j = 0; j < rt; ++j) {
        htil[i * m_tot + (rs + j)] = h12[i * rt + j];
        htil[(rs + j) * m_tot + i] = h12[i * rt + j];
      }
    }
    for (std::size_t i = 0; i < rt; ++i)
      for (std::size_t j = 0; j < rt; ++j) htil[(rs + i) * m_tot + (rs + j)] = h22[i * rt + j];
    symmetrize_inplace(htil, m_tot);
    auto eig = tensor::cpu_symmetric_eigen(std::move(htil), m_tot, plan);
    evals = std::move(eig.values);
    std::vector<double> ql(rs * m_tot), qs(rt * m_tot);
    for (std::size_t i = 0; i < rs; ++i)
      for (std::size_t j = 0; j < m_tot; ++j) ql[i * m_tot + j] = eig.vectors[i * m_tot + j];
    for (std::size_t i = 0; i < rt; ++i)
      for (std::size_t j = 0; j < m_tot; ++j) qs[i * m_tot + j] = eig.vectors[(rs + i) * m_tot + j];
    cl_all = matmul('N', 'N', n, m_tot, rs, us, ql, 1.0);
    cs_all = matmul('N', 'N', n, m_tot, rt, ut, qs, root2_c);
  }

  std::vector<std::size_t> neg_idx, pos_idx;
  for (std::size_t i = 0; i < m_tot; ++i) {
    if (evals[i] > -c * c)
      pos_idx.push_back(i);
    else
      neg_idx.push_back(i);
  }
  DiracSolution sol;
  sol.n_neg = neg_idx.size();
  sol.n_pos = pos_idx.size();
  sol.evals_neg.resize(sol.n_neg);
  sol.evals_pos.resize(sol.n_pos);
  sol.evecs_neg.resize(2 * n * sol.n_neg);
  sol.evecs_pos.resize(2 * n * sol.n_pos);
  std::vector<double> al(n * sol.n_pos), as(n * sol.n_pos);
  for (std::size_t k = 0; k < sol.n_neg; ++k) {
    const std::size_t col = neg_idx[k];
    sol.evals_neg[k] = evals[col];
    for (std::size_t i = 0; i < n; ++i) {
      sol.evecs_neg[i * sol.n_neg + k] = cl_all[i * m_tot + col];
      sol.evecs_neg[(n + i) * sol.n_neg + k] = cs_all[i * m_tot + col];
    }
  }
  for (std::size_t k = 0; k < sol.n_pos; ++k) {
    const std::size_t col = pos_idx[k];
    sol.evals_pos[k] = evals[col];
    for (std::size_t i = 0; i < n; ++i) {
      const double vl = cl_all[i * m_tot + col];
      const double vs = cs_all[i * m_tot + col];
      sol.evecs_pos[i * sol.n_pos + k] = vl;
      sol.evecs_pos[(n + i) * sol.n_pos + k] = vs;
      al[i * sol.n_pos + k] = vl;
      as[i * sol.n_pos + k] = vs;
    }
  }
  if (sol.n_pos == n) {
    sol.al_inv = invert_square_matrix(al, n);
  } else {
    sol.al_inv = pseudoinverse_tall(al, n, sol.n_pos);
  }
  sol.x_mat = matmul('N', 'N', n, n, sol.n_pos, as, sol.al_inv);
  return sol;
}

X2c1eMetricFactors build_r_metric(const std::vector<double>& s, const std::vector<double>& s_nesc,
                                  std::size_t n) {
  X2c1eMetricFactors out;
  out.n = n;
  const auto eig_s = x2c_sym_eigen(s, n);
  std::vector<std::size_t> s_keep, s_drop;
  for (std::size_t i = 0; i < n; ++i) {
    if (eig_s.values[i] > kOverlapRankTol)
      s_keep.push_back(i);
    else
      s_drop.push_back(i);
  }
  const std::size_t rs = s_keep.size();
  const std::size_t ds_cnt = s_drop.size();
  out.s_rank = rs;
  out.s_evals.resize(rs);
  out.s_evecs.resize(n * rs);
  std::vector<double> ss(rs), si(rs);
  for (std::size_t k = 0; k < rs; ++k) {
    const std::size_t idx = s_keep[k];
    out.s_evals[k] = eig_s.values[idx];
    ss[k] = std::sqrt(out.s_evals[k]);
    si[k] = 1.0 / ss[k];
    for (std::size_t i = 0; i < n; ++i) out.s_evecs[i * rs + k] = eig_s.vectors[i * n + idx];
  }
  out.s_disc_evals.resize(ds_cnt);
  out.s_disc_evecs.resize(n * ds_cnt);
  for (std::size_t p = 0; p < ds_cnt; ++p) {
    const std::size_t idx = s_drop[p];
    out.s_disc_evals[p] = eig_s.values[idx];
    for (std::size_t i = 0; i < n; ++i) out.s_disc_evecs[i * ds_cnt + p] = eig_s.vectors[i * n + idx];
  }
  if (rs == 0) {
    out.r_mat.assign(n * n, 0.0);
    return out;
  }
  out.snesc_v = congruence_ct_a_c(out.s_evecs, n, rs, s_nesc);
  symmetrize_inplace(out.snesc_v, rs);
  if (ds_cnt > 0) {
    auto snesc_v0 = matmul('N', 'N', n, ds_cnt, n, s_nesc, out.s_disc_evecs);
    out.snesc_ret_disc = matmul('T', 'N', rs, ds_cnt, n, out.s_evecs, snesc_v0);
  }

  // Diagonalize delta_K = S^{-1/2} (S_nesc - S) S^{-1/2} = K - I so the +1 shift
  // never dilutes small relativistic eigenvalues or exact-zero ECP eigenvalues.
  std::vector<double> delta_s(n * n);
  for (std::size_t k = 0; k < n * n; ++k) delta_s[k] = s_nesc[k] - s[k];
  auto delta_sv = congruence_ct_a_c(out.s_evecs, n, rs, delta_s);
  symmetrize_inplace(delta_sv, rs);
  std::vector<double> delta_k(rs * rs);
  for (std::size_t i = 0; i < rs; ++i)
    for (std::size_t j = 0; j < rs; ++j) delta_k[i * rs + j] = si[i] * delta_sv[i * rs + j] * si[j];
  symmetrize_inplace(delta_k, rs);

  const auto eig_dk = x2c_sym_eigen(std::move(delta_k), rs);
  std::vector<std::size_t> k_keep, k_drop;
  for (std::size_t i = 0; i < rs; ++i) {
    if (1.0 + eig_dk.values[i] > kOverlapRankTol)
      k_keep.push_back(i);
    else
      k_drop.push_back(i);
  }
  const std::size_t rk = k_keep.size();
  const std::size_t dk_cnt = k_drop.size();
  out.k_rank = rk;
  out.k_evals.resize(rk);
  out.k_evecs.resize(rs * rk);
  std::vector<double> vk_scaled(rs * rk);
  for (std::size_t k = 0; k < rk; ++k) {
    const std::size_t idx = k_keep[k];
    out.k_evals[k] = 1.0 + eig_dk.values[idx];
    const double ki = 1.0 / std::sqrt(out.k_evals[k]);
    for (std::size_t i = 0; i < rs; ++i) {
      const double v = eig_dk.vectors[i * rs + idx];
      out.k_evecs[i * rk + k] = v;
      vk_scaled[i * rk + k] = v * ki;
    }
  }
  out.k_disc_evals.resize(dk_cnt);
  out.k_disc_evecs.resize(rs * dk_cnt);
  for (std::size_t c = 0; c < dk_cnt; ++c) {
    const std::size_t idx = k_drop[c];
    out.k_disc_evals[c] = 1.0 + eig_dk.values[idx];
    for (std::size_t i = 0; i < rs; ++i)
      out.k_disc_evecs[i * dk_cnt + c] = eig_dk.vectors[i * rs + idx];
  }
  if (rk == 0) {
    out.r_mat.assign(n * n, 0.0);
    return out;
  }
  auto m_mat = matmul('N', 'T', rs, rs, rk, vk_scaled, out.k_evecs);
  std::vector<double> r_v(rs * rs);
  for (std::size_t i = 0; i < rs; ++i)
    for (std::size_t j = 0; j < rs; ++j) r_v[i * rs + j] = si[i] * m_mat[i * rs + j] * ss[j];
  out.r_mat = congruence_c_a_ct(out.s_evecs, n, rs, r_v);
  return out;
}

// Reverse-mode adjoint of R = S^{-1/2} (S^{-1/2} S_nesc S^{-1/2})^{-1/2} S^{1/2},
// including exact retained-discarded subspace rotation divided differences.
void adjoint_r_metric(const X2c1eMetricFactors& rf, const std::vector<double>& r_bar,
                      std::vector<double>& s_bar_out, std::vector<double>& snesc_bar_out) {
  const std::size_t n = rf.n;
  const std::size_t rs = rf.s_rank;
  const std::size_t ds_cnt = n - rs;
  const std::size_t rk = rf.k_rank;
  const std::size_t dk_cnt = rs - rk;
  if (rs == 0 || rk == 0) {
    s_bar_out.assign(n * n, 0.0);
    snesc_bar_out.assign(n * n, 0.0);
    return;
  }
  std::vector<double> ss(rs), si(rs), ks(rk), ki(rk);
  for (std::size_t i = 0; i < rs; ++i) {
    ss[i] = std::sqrt(rf.s_evals[i]);
    si[i] = 1.0 / ss[i];
  }
  for (std::size_t a = 0; a < rk; ++a) {
    ks[a] = std::sqrt(rf.k_evals[a]);
    ki[a] = 1.0 / ks[a];
  }
  std::vector<double> vk_scaled(rs * rk);
  for (std::size_t i = 0; i < rs; ++i)
    for (std::size_t a = 0; a < rk; ++a) vk_scaled[i * rk + a] = rf.k_evecs[i * rk + a] * ki[a];
  auto m_mat = matmul('N', 'T', rs, rs, rk, vk_scaled, rf.k_evecs);

  auto rv_bar = congruence_ct_a_c(rf.s_evecs, n, rs, r_bar);

  // G_{s^-1/2}^{(1)} = sym((rv_bar * ss[None,:]) @ M)
  std::vector<double> rv_ss(rs * rs), si_rv(rs * rs), m_bar(rs * rs);
  for (std::size_t i = 0; i < rs; ++i) {
    for (std::size_t j = 0; j < rs; ++j) {
      const double rval = rv_bar[i * rs + j];
      rv_ss[i * rs + j] = rval * ss[j];
      si_rv[i * rs + j] = si[i] * rval;
      m_bar[i * rs + j] = si[i] * rval * ss[j];
    }
  }
  symmetrize_inplace(m_bar, rs);
  auto g_si_1 = matmul('N', 'N', rs, rs, rs, rv_ss, m_mat);
  symmetrize_inplace(g_si_1, rs);
  auto g_ss = matmul('N', 'N', rs, rs, rs, m_mat, si_rv);
  symmetrize_inplace(g_ss, rs);

  auto mt = congruence_ct_a_c(rf.k_evecs, rs, rk, m_bar);
  std::vector<double> kt(rk * rk);
  for (std::size_t a = 0; a < rk; ++a)
    for (std::size_t b = 0; b < rk; ++b)
      kt[a * rk + b] = -mt[a * rk + b] / (ks[a] * ks[b] * (ks[a] + ks[b]));
  auto k_bar = congruence_c_a_ct(rf.k_evecs, rs, rk, kt);
  if (dk_cnt > 0) {
    auto mbar_w0 = matmul('N', 'N', rs, dk_cnt, rs, m_bar, rf.k_disc_evecs);
    auto m_10 = matmul('T', 'N', rk, dk_cnt, rs, rf.k_evecs, mbar_w0);
    std::vector<double> k_10(rk * dk_cnt);
    for (std::size_t a = 0; a < rk; ++a) {
      for (std::size_t c = 0; c < dk_cnt; ++c) {
        const double gap = rf.k_evals[a] - rf.k_disc_evals[c];
        if (!(gap > 0.0) || !std::isfinite(gap))
          throw std::runtime_error("X2C-1e rank-reduced K eigenvalue gap is nonpositive");
        k_10[a * dk_cnt + c] = (ki[a] / gap) * m_10[a * dk_cnt + c];
      }
    }
    auto w1_k10 = matmul('N', 'N', rs, dk_cnt, rk, rf.k_evecs, k_10);
    auto k_cross = matmul('N', 'T', rs, rs, dk_cnt, w1_k10, rf.k_disc_evecs);
    for (std::size_t i = 0; i < rs; ++i)
      for (std::size_t j = 0; j < rs; ++j)
        k_bar[i * rs + j] += k_cross[i * rs + j] + k_cross[j * rs + i];
  }
  symmetrize_inplace(k_bar, rs);

  std::vector<double> snesc_v_bar(rs * rs);
  std::vector<double> k_bar_si(rs * rs);
  for (std::size_t i = 0; i < rs; ++i) {
    for (std::size_t j = 0; j < rs; ++j) {
      const double kv = k_bar[i * rs + j];
      snesc_v_bar[i * rs + j] = si[i] * kv * si[j];
      k_bar_si[i * rs + j] = kv * si[j];
    }
  }
  auto g_si_2 = matmul('N', 'N', rs, rs, rs, k_bar_si, rf.snesc_v, 2.0);
  symmetrize_inplace(g_si_2, rs);

  std::vector<double> sv_bar(rs * rs);
  for (std::size_t i = 0; i < rs; ++i) {
    for (std::size_t j = 0; j < rs; ++j) {
      const double g_si = g_si_1[i * rs + j] + g_si_2[i * rs + j];
      sv_bar[i * rs + j] = g_ss[i * rs + j] / (ss[i] + ss[j]) -
                           g_si / (ss[i] * ss[j] * (ss[i] + ss[j]));
    }
  }
  s_bar_out = congruence_c_a_ct(rf.s_evecs, n, rs, sv_bar);
  snesc_bar_out = congruence_c_a_ct(rf.s_evecs, n, rs, snesc_v_bar);

  if (ds_cnt > 0) {
    std::vector<double> r_v(rs * rs);
    for (std::size_t i = 0; i < rs; ++i)
      for (std::size_t j = 0; j < rs; ++j) r_v[i * rs + j] = si[i] * m_mat[i * rs + j] * ss[j];
    auto rbar_v0 = matmul('N', 'N', n, ds_cnt, n, r_bar, rf.s_disc_evecs);
    auto rbar_10 = matmul('T', 'N', rs, ds_cnt, n, rf.s_evecs, rbar_v0);
    auto rbar_v1 = matmul('N', 'N', n, rs, n, r_bar, rf.s_evecs);
    auto rbar_01 = matmul('T', 'N', ds_cnt, rs, n, rf.s_disc_evecs, rbar_v1);
    auto om1 = matmul('N', 'T', rs, ds_cnt, rs, r_v, rbar_01);
    auto om2 = matmul('T', 'N', rs, ds_cnt, rs, r_v, rbar_10);
    auto om3 = matmul('N', 'N', rs, ds_cnt, rs, snesc_v_bar, rf.snesc_ret_disc, 2.0);
    std::vector<double> s10_bar(rs * ds_cnt);
    for (std::size_t i = 0; i < rs; ++i) {
      for (std::size_t p = 0; p < ds_cnt; ++p) {
        const double gap = rf.s_evals[i] - rf.s_disc_evals[p];
        if (!(gap > 0.0) || !std::isfinite(gap))
          throw std::runtime_error("X2C-1e rank-reduced overlap eigenvalue gap is nonpositive");
        s10_bar[i * ds_cnt + p] =
            (om1[i * ds_cnt + p] + om2[i * ds_cnt + p] + om3[i * ds_cnt + p]) / gap;
      }
    }
    auto v1_s10 = matmul('N', 'N', n, ds_cnt, rs, rf.s_evecs, s10_bar);
    auto s_cross = matmul('N', 'T', n, n, ds_cnt, v1_s10, rf.s_disc_evecs);
    for (std::size_t k = 0; k < n * n; ++k) s_bar_out[k] += s_cross[k];
  }
  symmetrize_inplace(s_bar_out, n);
  symmetrize_inplace(snesc_bar_out, n);
}

// Forward-mode differential of R (JCP 135, 084114 (2011), Eq. 34),
// extended with exact retained-discarded subspace rotation divided differences.
std::vector<double> forward_dr_metric(const X2c1eMetricFactors& rf, const std::vector<double>& ds,
                                      const std::vector<double>& ds_nesc) {
  const std::size_t n = rf.n;
  const std::size_t rs = rf.s_rank;
  const std::size_t ds_cnt = n - rs;
  const std::size_t rk = rf.k_rank;
  const std::size_t dk_cnt = rs - rk;
  if (rs == 0 || rk == 0) return std::vector<double>(n * n, 0.0);

  std::vector<double> ss(rs), si(rs), ks(rk), ki(rk);
  for (std::size_t i = 0; i < rs; ++i) {
    ss[i] = std::sqrt(rf.s_evals[i]);
    si[i] = 1.0 / ss[i];
  }
  for (std::size_t a = 0; a < rk; ++a) {
    ks[a] = std::sqrt(rf.k_evals[a]);
    ki[a] = 1.0 / ks[a];
  }
  auto ds_v = congruence_ct_a_c(rf.s_evecs, n, rs, ds);
  auto dsn_v = congruence_ct_a_c(rf.s_evecs, n, rs, ds_nesc);

  std::vector<double> ds_sqrt(rs * rs), ds_invsqrt(rs * rs);
  for (std::size_t i = 0; i < rs; ++i) {
    for (std::size_t j = 0; j < rs; ++j) {
      const double v = ds_v[i * rs + j];
      ds_sqrt[i * rs + j] = v / (ss[i] + ss[j]);
      ds_invsqrt[i * rs + j] = -v / (ss[i] * ss[j] * (ss[i] + ss[j]));
    }
  }
  auto t1 = matmul('N', 'N', rs, rs, rs, ds_invsqrt, rf.snesc_v);
  std::vector<double> dk(rs * rs);
  for (std::size_t i = 0; i < rs; ++i) {
    for (std::size_t j = 0; j < rs; ++j) {
      dk[i * rs + j] =
          t1[i * rs + j] * si[j] + t1[j * rs + i] * si[i] + si[i] * dsn_v[i * rs + j] * si[j];
    }
  }
  std::vector<double> omega;
  if (ds_cnt > 0) {
    auto ds_v0 = matmul('N', 'N', n, ds_cnt, n, ds, rf.s_disc_evecs);
    auto ds_10 = matmul('T', 'N', rs, ds_cnt, n, rf.s_evecs, ds_v0);
    omega.resize(rs * ds_cnt);
    for (std::size_t i = 0; i < rs; ++i) {
      for (std::size_t p = 0; p < ds_cnt; ++p) {
        const double gap = rf.s_evals[i] - rf.s_disc_evals[p];
        if (!(gap > 0.0) || !std::isfinite(gap))
          throw std::runtime_error("X2C-1e rank-reduced overlap eigenvalue gap is nonpositive");
        omega[i * ds_cnt + p] = ds_10[i * ds_cnt + p] / gap;
      }
    }
    auto omega_n10t = matmul('N', 'T', rs, rs, ds_cnt, omega, rf.snesc_ret_disc);
    for (std::size_t i = 0; i < rs; ++i)
      for (std::size_t j = 0; j < rs; ++j)
        dk[i * rs + j] += si[i] * (omega_n10t[i * rs + j] + omega_n10t[j * rs + i]) * si[j];
  }

  auto dk_k = congruence_ct_a_c(rf.k_evecs, rs, rk, dk);
  std::vector<double> dm_k(rk * rk);
  for (std::size_t a = 0; a < rk; ++a)
    for (std::size_t b = 0; b < rk; ++b)
      dm_k[a * rk + b] = -dk_k[a * rk + b] / (ks[a] * ks[b] * (ks[a] + ks[b]));
  auto dm = congruence_c_a_ct(rf.k_evecs, rs, rk, dm_k);
  if (dk_cnt > 0) {
    auto dk_w0 = matmul('N', 'N', rs, dk_cnt, rs, dk, rf.k_disc_evecs);
    auto dk_10 = matmul('T', 'N', rk, dk_cnt, rs, rf.k_evecs, dk_w0);
    std::vector<double> dm_10(rk * dk_cnt);
    for (std::size_t a = 0; a < rk; ++a) {
      for (std::size_t c = 0; c < dk_cnt; ++c) {
        const double gap = rf.k_evals[a] - rf.k_disc_evals[c];
        if (!(gap > 0.0) || !std::isfinite(gap))
          throw std::runtime_error("X2C-1e rank-reduced K eigenvalue gap is nonpositive");
        dm_10[a * dk_cnt + c] = (ki[a] / gap) * dk_10[a * dk_cnt + c];
      }
    }
    auto w1_dm10 = matmul('N', 'N', rs, dk_cnt, rk, rf.k_evecs, dm_10);
    auto dm_cross = matmul('N', 'T', rs, rs, dk_cnt, w1_dm10, rf.k_disc_evecs);
    for (std::size_t i = 0; i < rs; ++i)
      for (std::size_t j = 0; j < rs; ++j)
        dm[i * rs + j] += dm_cross[i * rs + j] + dm_cross[j * rs + i];
  }

  std::vector<double> vk_scaled(rs * rk);
  for (std::size_t i = 0; i < rs; ++i)
    for (std::size_t a = 0; a < rk; ++a) vk_scaled[i * rk + a] = rf.k_evecs[i * rk + a] * ki[a];
  auto m_mat = matmul('N', 'T', rs, rs, rk, vk_scaled, rf.k_evecs);

  // dR_v = ds_invsqrt * M * ss + si * dm * ss + si * M * ds_sqrt
  auto term1 = matmul('N', 'N', rs, rs, rs, ds_invsqrt, m_mat);
  auto term3 = matmul('N', 'N', rs, rs, rs, m_mat, ds_sqrt);
  std::vector<double> dr_v(rs * rs);
  for (std::size_t i = 0; i < rs; ++i)
    for (std::size_t j = 0; j < rs; ++j)
      dr_v[i * rs + j] =
          term1[i * rs + j] * ss[j] + si[i] * dm[i * rs + j] * ss[j] + si[i] * term3[i * rs + j];
  auto dr = congruence_c_a_ct(rf.s_evecs, n, rs, dr_v);
  if (ds_cnt > 0) {
    std::vector<double> r_v(rs * rs);
    for (std::size_t i = 0; i < rs; ++i)
      for (std::size_t j = 0; j < rs; ++j) r_v[i * rs + j] = si[i] * m_mat[i * rs + j] * ss[j];
    auto omegat_rv = matmul('T', 'N', ds_cnt, rs, rs, omega, r_v);
    auto v0_omegat_rv = matmul('N', 'N', n, rs, ds_cnt, rf.s_disc_evecs, omegat_rv);
    auto left_cross = matmul('N', 'T', n, n, rs, v0_omegat_rv, rf.s_evecs);
    auto rv_omega = matmul('N', 'N', rs, ds_cnt, rs, r_v, omega);
    auto v1_rv_omega = matmul('N', 'N', n, ds_cnt, rs, rf.s_evecs, rv_omega);
    auto right_cross = matmul('N', 'T', n, n, ds_cnt, v1_rv_omega, rf.s_disc_evecs);
    for (std::size_t k = 0; k < n * n; ++k) dr[k] += left_cross[k] + right_cross[k];
  }
  return dr;
}

// Build atomic signature key to deduplicate identical AE atomic X_A / R_A blocks.
struct AtomicBasisKey {
  int atomic_number{};
  core::BasisRepresentation rep{};
  std::vector<std::pair<std::uint32_t, double>> shells;  // (l, exponent)
  bool operator<(const AtomicBasisKey& other) const {
    if (atomic_number != other.atomic_number) return atomic_number < other.atomic_number;
    if (rep != other.rep) return rep < other.rep;
    return shells < other.shells;
  }
};

}  // namespace

StvwIntegrals build_stvw_integrals(const core::System& system) {
  std::size_t nbf = 0;
  const auto projs = build_shell_projections(system, nbf);
  const std::size_t n2 = checked_mul(nbf, nbf);
  StvwIntegrals out;
  out.nbf = nbf;
  out.overlap.assign(n2, 0.0);
  out.kinetic.assign(n2, 0.0);
  out.attraction.assign(n2, 0.0);
  out.small_potential.assign(n2, 0.0);
  if (nbf == 0 || system.atoms.empty()) return out;

  std::vector<double> boys_buf;
  ScalarCoulomb aux;
  std::array<ScalarHermite, 3> hermite;

  for (std::size_t si = 0; si < system.shells.size(); ++si) {
    const auto& shell_i = system.shells[si];
    const auto& proj_i = projs[si];
    const unsigned li = shell_i.angular_momentum;
    const auto& center_a = system.atoms[shell_i.atom_index].position;

    for (std::size_t sj = si; sj < system.shells.size(); ++sj) {
      const auto& shell_j = system.shells[sj];
      const auto& proj_j = projs[sj];
      const unsigned lj = shell_j.angular_momentum;
      const auto& center_b = system.atoms[shell_j.atom_index].position;
      const std::size_t nci = proj_i.n_cart;
      const std::size_t ncj = proj_j.n_cart;
      const std::size_t nc_pair = nci * ncj;

      std::vector<double> s_c(nc_pair, 0.0);
      std::vector<double> t_c(nc_pair, 0.0);
      std::vector<double> v_c(nc_pair, 0.0);
      std::vector<double> w_c(nc_pair, 0.0);

      const unsigned max_order = li + lj + 2;
      const unsigned rdim = max_order + 1;
      const std::size_t rsize = static_cast<std::size_t>(rdim) * rdim * rdim;
      std::vector<double> r_total(rsize, 0.0);
      auto r_idx = [rdim](unsigned t, unsigned u, unsigned v) -> std::size_t {
        return (static_cast<std::size_t>(t) * rdim + u) * rdim + v;
      };

      for (const auto& pi : shell_i.primitives) {
        for (const auto& pj : shell_j.primitives) {
          const double alpha = pi.exponent;
          const double beta = pj.exponent;
          const double p = alpha + beta;
          const double prim_coeff = pi.coefficient * pj.coefficient;
          if (prim_coeff == 0.0) continue;

          const std::array<double, 3> product{
              (alpha * center_a[0] + beta * center_b[0]) / p,
              (alpha * center_a[1] + beta * center_b[1]) / p,
              (alpha * center_a[2] + beta * center_b[2]) / p,
          };
          for (std::size_t ax = 0; ax < 3; ++ax) {
            fill_hermite_scalar(li + 1, lj + 2, product[ax], center_a[ax], center_b[ax], alpha,
                                beta, hermite[ax]);
          }

          // Sum R_total over all nuclei C with effective ionic charges.
          const double two_pi_over_p = 2.0 * std::numbers::pi / p;
          std::fill(r_total.begin(), r_total.end(), 0.0);
          for (std::size_t atom_c = 0; atom_c < system.atoms.size(); ++atom_c) {
            const double zc = static_cast<double>(system.atoms[atom_c].ionic_charge());
            if (zc == 0.0) continue;
            fill_coulomb_scalar(max_order, p, product, system.atoms[atom_c].position, boys_buf,
                                aux);
            const double neg_z = -zc * two_pi_over_p;
            for (unsigned t = 0; t <= max_order; ++t)
              for (unsigned u = 0; t + u <= max_order; ++u)
                for (unsigned v = 0; t + u + v <= max_order; ++v)
                  r_total[r_idx(t, u, v)] += neg_z * aux.at(0, t, u, v);
          }

          const double st_pref = prim_coeff * std::pow(std::numbers::pi / p, 1.5);
          for (std::size_t ci = 0; ci < nci; ++ci) {
            const auto& ai = proj_i.cart_comps[ci];
            for (std::size_t cj = 0; cj < ncj; ++cj) {
              const auto& bj = proj_j.cart_comps[cj];
              const std::size_t slot = ci * ncj + cj;
              const double ex0 = hermite[0].get(ai[0], bj[0], 0);
              const double ey0 = hermite[1].get(ai[1], bj[1], 0);
              const double ez0 = hermite[2].get(ai[2], bj[2], 0);
              s_c[slot] += st_pref * ex0 * ey0 * ez0;
              const double kx = kinetic_1d(hermite[0], ai[0], bj[0], beta);
              const double ky = kinetic_1d(hermite[1], ai[1], bj[1], beta);
              const double kz = kinetic_1d(hermite[2], ai[2], bj[2], beta);
              t_c[slot] += st_pref * (kx * ey0 * ez0 + ex0 * ky * ez0 + ex0 * ey0 * kz);

              double vv = 0.0;
              double ww = 0.0;
              const unsigned tmax = ai[0] + bj[0] + 2;
              const unsigned umax = ai[1] + bj[1] + 2;
              const unsigned vmax = ai[2] + bj[2] + 2;
              for (unsigned t = 0; t <= tmax; ++t) {
                const double ex = hermite[0].get(ai[0], bj[0], t);
                const double fx = hermite_f(hermite[0], ai[0], bj[0], t, alpha, beta);
                if (ex == 0.0 && fx == 0.0) continue;
                for (unsigned u = 0; u <= umax; ++u) {
                  const double ey = hermite[1].get(ai[1], bj[1], u);
                  const double fy = hermite_f(hermite[1], ai[1], bj[1], u, alpha, beta);
                  if (ey == 0.0 && fy == 0.0) continue;
                  const double exey = ex * ey;
                  const double fxy = fx * ey + ex * fy;
                  for (unsigned v = 0; v <= vmax && t + u + v <= max_order; ++v) {
                    const double rval = r_total[r_idx(t, u, v)];
                    if (rval == 0.0) continue;
                    const double ez = hermite[2].get(ai[2], bj[2], v);
                    const double fz = hermite_f(hermite[2], ai[2], bj[2], v, alpha, beta);
                    vv += rval * exey * ez;
                    ww += rval * (fxy * ez + exey * fz);
                  }
                }
              }
              v_c[slot] += prim_coeff * vv;
              w_c[slot] += prim_coeff * ww;
            }
          }
        }
      }

      // Transform Cartesian block to public AOs and store both symmetric triangles.
      auto transform_and_store = [&](const std::vector<double>& src, std::vector<double>& dst) {
        for (std::size_t mi = 0; mi < proj_i.n_ao; ++mi) {
          const std::size_t gi = proj_i.ao_offset + mi;
          for (std::size_t mj = 0; mj < proj_j.n_ao; ++mj) {
            const std::size_t gj = proj_j.ao_offset + mj;
            double val = 0.0;
            for (std::size_t ci = 0; ci < nci; ++ci) {
              const double ti = proj_i.cart_to_pub[mi * nci + ci];
              if (ti == 0.0) continue;
              for (std::size_t cj = 0; cj < ncj; ++cj) {
                val += ti * proj_j.cart_to_pub[mj * ncj + cj] * src[ci * ncj + cj];
              }
            }
            dst[gi * nbf + gj] = val;
            if (si != sj) dst[gj * nbf + gi] = val;
          }
        }
      };
      transform_and_store(s_c, out.overlap);
      transform_and_store(t_c, out.kinetic);
      transform_and_store(v_c, out.attraction);
      transform_and_store(w_c, out.small_potential);
    }
  }
  return out;
}

std::shared_ptr<const X2c1eState> build_x2c1e_state(const core::System& system, X2c1eMode mode,
                                                    double speed_of_light) {
  if (!(speed_of_light > 0.0) || !std::isfinite(speed_of_light))
    throw std::invalid_argument("X2C-1e speed of light must be positive and finite");
  if (mode == X2c1eMode::Global1e) {
    const bool has_ecp = !system.ecp_terms.empty() ||
                         std::any_of(system.atoms.begin(), system.atoms.end(),
                                     [](const core::Atom& a) { return a.ecp_core > 0; });
    if (has_ecp) {
      throw std::invalid_argument(
          "Global X2C-1e is not valid on systems containing ECP atoms; use Dlu1e");
    }
  }

  auto state = std::make_shared<X2c1eState>();
  state->mode = mode;
  state->speed_of_light = speed_of_light;
  state->n_contr = molecule::ao_count(system);
  const std::size_t nc = state->n_contr;

  if (mode == X2c1eMode::None || nc == 0) {
    state->hcore_no_ecp.assign(nc * nc, 0.0);
    state->hcore_delta.assign(nc * nc, 0.0);
    return state;
  }

  auto dec = decontract_system(system, mode);
  state->n_prim = dec.n_prim;
  state->has_ae_atoms = dec.has_ae_atoms;
  state->uncontracted_system = std::move(dec.uncontracted);
  state->contraction = std::move(dec.contraction);
  const std::size_t np = state->n_prim;

  if (!state->has_ae_atoms) {
    // Pure ECP molecule (e.g., isolated Br/I atom with ECP): zero relativistic correction.
    const auto stvw = build_stvw_integrals(state->uncontracted_system);
    std::vector<double> h_nr_prim(np * np);
    for (std::size_t k = 0; k < np * np; ++k) h_nr_prim[k] = stvw.kinetic[k] + stvw.attraction[k];
    state->hcore_no_ecp = congruence_ct_a_c(state->contraction, np, nc, h_nr_prim);
    symmetrize_inplace(state->hcore_no_ecp, nc);
    state->hcore_delta.assign(nc * nc, 0.0);
    return state;
  }

  const auto stvw = build_stvw_integrals(state->uncontracted_system);
  const double a_s = 0.5 / (speed_of_light * speed_of_light);
  const double a_w = 0.25 / (speed_of_light * speed_of_light);

  state->x_mat.assign(np * np, 0.0);
  state->dx_mat.assign(np * np, 0.0);
  state->xw_mat.assign(np * np, 0.0);
  for (std::size_t i = 0; i < np; ++i) state->x_mat[i * np + i] = 1.0;

  if (mode == X2c1eMode::Global1e) {
    auto dirac =
        solve_dirac_1e(stvw.overlap, stvw.kinetic, stvw.attraction, stvw.small_potential, np,
                       speed_of_light);
    state->x_mat = std::move(dirac.x_mat);
    state->xw_mat = state->x_mat;
    for (std::size_t i = 0; i < np; ++i)
      for (std::size_t j = 0; j < np; ++j)
        state->dx_mat[i * np + j] = state->x_mat[i * np + j] - (i == j ? 1.0 : 0.0);
    state->t_prim = stvw.kinetic;
    state->w_prim = stvw.small_potential;
    state->dirac_neg_count = dirac.n_neg;
    state->dirac_pos_count = dirac.n_pos;
    state->dirac_evals_neg = std::move(dirac.evals_neg);
    state->dirac_evals_pos = std::move(dirac.evals_pos);
    state->dirac_evecs_neg = std::move(dirac.evecs_neg);
    state->dirac_evecs_pos = std::move(dirac.evecs_pos);
    state->dirac_al_inv = std::move(dirac.al_inv);
  } else {
    // Atomic-block fold-down on all-electron atoms only; ECP atoms stay X = I, X_W = 0.
    struct CachedAtomicBlock {
      std::vector<double> x;
      std::vector<double> r;
    };
    std::map<AtomicBasisKey, CachedAtomicBlock> cache;
    if (mode == X2c1eMode::DluLocalR) {
      state->ul_mat.assign(np * np, 0.0);
      state->us_mat.assign(np * np, 0.0);
      state->uw_mat.assign(np * np, 0.0);
      for (std::size_t i = 0; i < np; ++i) {
        state->ul_mat[i * np + i] = 1.0;
        state->us_mat[i * np + i] = 1.0;
      }
    }

    for (std::size_t a = 0; a < state->uncontracted_system.atoms.size(); ++a) {
      const std::size_t p0 = dec.atom_ao_start[a];
      const std::size_t p1 = dec.atom_ao_end[a];
      const std::size_t na = p1 - p0;
      if (!dec.atom_is_ae[a] || na == 0) continue;

      AtomicBasisKey key;
      key.atomic_number = state->uncontracted_system.atoms[a].atomic_number;
      key.rep = state->uncontracted_system.basis_representation;
      core::System atom_sys;
      atom_sys.atoms.push_back(
          {key.atomic_number, {0.0, 0.0, 0.0}, 0});
      atom_sys.charge = 0;
      atom_sys.multiplicity = 1;
      atom_sys.electron_count = key.atomic_number;
      atom_sys.basis_representation = key.rep;
      for (const auto& sh : state->uncontracted_system.shells) {
        if (sh.atom_index != a) continue;
        core::Shell ash = sh;
        ash.atom_index = 0;
        key.shells.emplace_back(sh.angular_momentum, sh.primitives[0].exponent);
        atom_sys.shells.push_back(std::move(ash));
      }

      auto it = cache.find(key);
      if (it == cache.end()) {
        const auto a_stvw = build_stvw_integrals(atom_sys);
        CachedAtomicBlock blk;
        if (key.rep == core::kSphericalBasis) {
          // Exact SO(3) symmetry: in spherical harmonics an isolated atom is
          // block-diagonal by (l, m) with identical radial blocks for all m=-l..+l.
          // Solving once per l eliminates degenerate Jacobi symmetry breaking and
          // guarantees exact rotational invariance (torque < 1e-13).
          blk.x.assign(na * na, 0.0);
          if (mode == X2c1eMode::DluLocalR) blk.r.assign(na * na, 0.0);
          std::map<unsigned, std::vector<std::size_t>> l_offsets;
          std::size_t ao_cur = 0;
          for (const auto& sh : atom_sys.shells) {
            l_offsets[sh.angular_momentum].push_back(ao_cur);
            ao_cur += 2 * sh.angular_momentum + 1;
          }
          for (const auto& [l, offs] : l_offsets) {
            const std::size_t k = offs.size();
            const std::size_t ncomp = 2 * l + 1;
            std::vector<double> sl(k * k), tl(k * k), vl(k * k), wl(k * k);
            for (std::size_t i = 0; i < k; ++i) {
              for (std::size_t j = 0; j < k; ++j) {
                const std::size_t gi = offs[i];
                const std::size_t gj = offs[j];
                sl[i * k + j] = a_stvw.overlap[gi * na + gj];
                tl[i * k + j] = a_stvw.kinetic[gi * na + gj];
                vl[i * k + j] = a_stvw.attraction[gi * na + gj];
                wl[i * k + j] = a_stvw.small_potential[gi * na + gj];
              }
            }
            auto dirac_l = solve_dirac_1e(sl, tl, vl, wl, k, speed_of_light);
            std::vector<double> rl;
            if (mode == X2c1eMode::DluLocalR) {
              auto xt_t_x = congruence_ct_a_c(dirac_l.x_mat, k, k, tl);
              std::vector<double> snesc_l(k * k);
              for (std::size_t idx = 0; idx < k * k; ++idx)
                snesc_l[idx] = sl[idx] + a_s * xt_t_x[idx];
              rl = build_r_metric(sl, snesc_l, k).r_mat;
            }
            for (std::size_t m = 0; m < ncomp; ++m) {
              for (std::size_t i = 0; i < k; ++i) {
                for (std::size_t j = 0; j < k; ++j) {
                  const std::size_t gi = offs[i] + m;
                  const std::size_t gj = offs[j] + m;
                  blk.x[gi * na + gj] = dirac_l.x_mat[i * k + j];
                  if (mode == X2c1eMode::DluLocalR) blk.r[gi * na + gj] = rl[i * k + j];
                }
              }
            }
          }
        } else {
          auto dirac = solve_dirac_1e(a_stvw.overlap, a_stvw.kinetic, a_stvw.attraction,
                                      a_stvw.small_potential, na, speed_of_light);
          blk.x = std::move(dirac.x_mat);
          if (mode == X2c1eMode::DluLocalR) {
            auto xt_t_x = congruence_ct_a_c(blk.x, na, na, a_stvw.kinetic);
            std::vector<double> snesc_a(na * na);
            for (std::size_t k = 0; k < na * na; ++k)
              snesc_a[k] = a_stvw.overlap[k] + a_s * xt_t_x[k];
            blk.r = build_r_metric(a_stvw.overlap, snesc_a, na).r_mat;
          }
        }
        it = cache.emplace(std::move(key), std::move(blk)).first;
      }

      const auto& blk = it->second;
      for (std::size_t i = 0; i < na; ++i) {
        for (std::size_t j = 0; j < na; ++j) {
          const double xij = blk.x[i * na + j];
          state->x_mat[(p0 + i) * np + (p0 + j)] = xij;
          state->xw_mat[(p0 + i) * np + (p0 + j)] = xij;
          state->dx_mat[(p0 + i) * np + (p0 + j)] = xij - (i == j ? 1.0 : 0.0);
        }
      }
      if (mode == X2c1eMode::DluLocalR) {
        auto xr = matmul('N', 'N', na, na, na, blk.x, blk.r);
        for (std::size_t i = 0; i < na; ++i) {
          for (std::size_t j = 0; j < na; ++j) {
            state->ul_mat[(p0 + i) * np + (p0 + j)] = blk.r[i * na + j];
            state->us_mat[(p0 + i) * np + (p0 + j)] = xr[i * na + j];
            state->uw_mat[(p0 + i) * np + (p0 + j)] = xr[i * na + j];
          }
        }
      }
    }
  }

  std::vector<double> h_prim(np * np, 0.0);
  if (mode == X2c1eMode::DluLocalR) {
    auto ul_v_ul = congruence_ct_a_c(state->ul_mat, np, np, stvw.attraction);
    auto t_us = matmul('N', 'N', np, np, np, stvw.kinetic, state->us_mat);
    auto ul_t_us = matmul('T', 'N', np, np, np, state->ul_mat, t_us);
    auto us_t_us = matmul('T', 'N', np, np, np, state->us_mat, t_us);
    auto uw_w_uw = congruence_ct_a_c(state->uw_mat, np, np, stvw.small_potential);
    for (std::size_t i = 0; i < np; ++i) {
      for (std::size_t j = 0; j < np; ++j) {
        h_prim[i * np + j] = ul_v_ul[i * np + j] + ul_t_us[i * np + j] + ul_t_us[j * np + i] -
                             us_t_us[i * np + j] + a_w * uw_w_uw[i * np + j];
      }
    }
  } else {
    // Dlu1e or Global1e:
    //   S_nesc = S + a_s * X_W^T * T * X_W
    //   H_nesc = V + T - dX^T * T * dX + a_w * X_W^T * W * X_W
    //   h_prim = R^T * H_nesc * R
    auto xw_t_xw = congruence_ct_a_c(state->xw_mat, np, np, stvw.kinetic);
    std::vector<double> s_nesc(np * np);
    for (std::size_t k = 0; k < np * np; ++k) s_nesc[k] = stvw.overlap[k] + a_s * xw_t_xw[k];
    symmetrize_inplace(s_nesc, np);

    auto dx_t_dx = congruence_ct_a_c(state->dx_mat, np, np, stvw.kinetic);
    auto xw_w_xw = congruence_ct_a_c(state->xw_mat, np, np, stvw.small_potential);
    state->h_nesc.resize(np * np);
    for (std::size_t k = 0; k < np * np; ++k) {
      state->h_nesc[k] =
          stvw.attraction[k] + stvw.kinetic[k] - dx_t_dx[k] + a_w * xw_w_xw[k];
    }
    symmetrize_inplace(state->h_nesc, np);

    state->metric = build_r_metric(stvw.overlap, s_nesc, np);
    h_prim = congruence_ct_a_c(state->metric.r_mat, np, np, state->h_nesc);
  }
  symmetrize_inplace(h_prim, np);

  std::vector<double> delta_prim(np * np);
  for (std::size_t k = 0; k < np * np; ++k)
    delta_prim[k] = h_prim[k] - stvw.kinetic[k] - stvw.attraction[k];

  state->hcore_no_ecp = congruence_ct_a_c(state->contraction, np, nc, h_prim);
  state->hcore_delta = congruence_ct_a_c(state->contraction, np, nc, delta_prim);
  symmetrize_inplace(state->hcore_no_ecp, nc);
  symmetrize_inplace(state->hcore_delta, nc);
  return state;
}

std::vector<double> contract_weighted_x2c1e_hcore_derivative(
    const X2c1eState& state, std::span<const double> hcore_weights, bool delta_only) {
  const std::size_t nc = state.n_contr;
  const std::size_t natom = state.uncontracted_system.atoms.size();
  const std::size_t ncoord = natom * 3;
  if (hcore_weights.size() != checked_mul(nc, nc))
    throw std::invalid_argument("X2C-1e weighted derivative received wrong hcore weight size");
  if (state.mode == X2c1eMode::None || nc == 0) return std::vector<double>(ncoord, 0.0);

  const std::size_t np = state.n_prim;
  const auto wh_sym = symmetrized_copy(hcore_weights, nc);
  auto wh_prim = congruence_c_a_ct(state.contraction, np, nc, wh_sym);
  symmetrize_inplace(wh_prim, np);

  if (!state.has_ae_atoms) {
    if (delta_only) return std::vector<double>(ncoord, 0.0);
    std::vector<double> zeros(np * np, 0.0);
    return contract_weighted_stvw_derivative(state.uncontracted_system, zeros, wh_prim, wh_prim,
                                             zeros);
  }

  const double c = state.speed_of_light;
  const double a_s = 0.5 / (c * c);
  const double a_w = 0.25 / (c * c);

  std::vector<double> s_bar(np * np, 0.0);
  std::vector<double> t_bar(np * np, 0.0);
  std::vector<double> v_bar(np * np, 0.0);
  std::vector<double> w_bar(np * np, 0.0);

  if (state.mode == X2c1eMode::DluLocalR) {
    v_bar = congruence_c_a_ct(state.ul_mat, np, np, wh_prim);
    auto ul_wh = matmul('N', 'N', np, np, np, state.ul_mat, wh_prim);
    auto ul_wh_ust = matmul('N', 'T', np, np, np, ul_wh, state.us_mat);
    auto us_wh_ust = congruence_c_a_ct(state.us_mat, np, np, wh_prim);
    for (std::size_t i = 0; i < np; ++i)
      for (std::size_t j = 0; j < np; ++j)
        t_bar[i * np + j] = ul_wh_ust[i * np + j] + ul_wh_ust[j * np + i] - us_wh_ust[i * np + j];
    w_bar = congruence_c_a_ct(state.uw_mat, np, np, wh_prim);
    for (double& x : w_bar) x *= a_w;
  } else {
    // Dlu1e or Global1e:
    //   h_prim = R^T H_nesc R -> H_nesc_bar = R wh_prim R^T, R_bar = 2 H_nesc R wh_prim.
    auto r_wh = matmul('N', 'N', np, np, np, state.metric.r_mat, wh_prim);
    auto hnesc_bar = matmul('N', 'T', np, np, np, r_wh, state.metric.r_mat);
    symmetrize_inplace(hnesc_bar, np);
    auto r_bar = matmul('N', 'N', np, np, np, state.h_nesc, r_wh, 2.0);

    std::vector<double> s_r_bar, snesc_bar;
    adjoint_r_metric(state.metric, r_bar, s_r_bar, snesc_bar);

    for (std::size_t k = 0; k < np * np; ++k) s_bar[k] = s_r_bar[k] + snesc_bar[k];
    v_bar = hnesc_bar;

    auto xw_snesc_xwt = congruence_c_a_ct(state.xw_mat, np, np, snesc_bar);
    auto dx_hnesc_dxt = congruence_c_a_ct(state.dx_mat, np, np, hnesc_bar);
    for (std::size_t k = 0; k < np * np; ++k)
      t_bar[k] = a_s * xw_snesc_xwt[k] + hnesc_bar[k] - dx_hnesc_dxt[k];

    w_bar = congruence_c_a_ct(state.xw_mat, np, np, hnesc_bar);
    for (double& x : w_bar) x *= a_w;

    if (state.mode == X2c1eMode::Global1e) {
      // X_bar = -2 * T * dX * hnesc_bar + 2 * a_w * W * X * hnesc_bar + 2 * a_s * T * X * snesc_bar
      auto t_dx = matmul('N', 'N', np, np, np, state.t_prim, state.dx_mat);
      auto x_bar = matmul('N', 'N', np, np, np, t_dx, hnesc_bar, -2.0);
      auto w_x = matmul('N', 'N', np, np, np, state.w_prim, state.x_mat);
      auto term2 = matmul('N', 'N', np, np, np, w_x, hnesc_bar, 2.0 * a_w);
      auto t_x = matmul('N', 'N', np, np, np, state.t_prim, state.x_mat);
      auto term3 = matmul('N', 'N', np, np, np, t_x, snesc_bar, 2.0 * a_s);
      for (std::size_t k = 0; k < np * np; ++k) x_bar[k] += term2[k] + term3[k];

      // Pull back X_bar through X = A_S * A_L^{-1} and the Dirac eigensystem:
      //   G = (B_S - X * B_L)^T * X_bar * (A_L^{-1})^T   (n_neg x n_pos)
      //   Y_{qp} = G_{qp} / (E_{+,p} - E_{-,q})
      const std::size_t n_neg = state.dirac_neg_count;
      const std::size_t n_pos = state.dirac_pos_count;
      std::vector<double> bl(np * n_neg), bs(np * n_neg);
      for (std::size_t i = 0; i < np; ++i) {
        for (std::size_t q = 0; q < n_neg; ++q) {
          bl[i * n_neg + q] = state.dirac_evecs_neg[i * n_neg + q];
          bs[i * n_neg + q] = state.dirac_evecs_neg[(np + i) * n_neg + q];
        }
      }
      auto x_bl = matmul('N', 'N', np, n_neg, np, state.x_mat, bl);
      std::vector<double> diff_b(np * n_neg);
      for (std::size_t k = 0; k < np * n_neg; ++k) diff_b[k] = bs[k] - x_bl[k];

      auto xbar_alinvt = matmul('N', 'T', np, n_pos, np, x_bar, state.dirac_al_inv);
      auto g_mat = matmul('T', 'N', n_neg, n_pos, np, diff_b, xbar_alinvt);

      std::vector<double> y_mat(n_neg * n_pos), y_epos(n_neg * n_pos);
      for (std::size_t q = 0; q < n_neg; ++q) {
        for (std::size_t p = 0; p < n_pos; ++p) {
          const double val = g_mat[q * n_pos + p] /
                             (state.dirac_evals_pos[p] - state.dirac_evals_neg[q]);
          y_mat[q * n_pos + p] = val;
          y_epos[q * n_pos + p] = val * state.dirac_evals_pos[p];
        }
      }
      const std::size_t n2 = 2 * np;
      auto cn_y = matmul('N', 'N', n2, n_pos, n_neg, state.dirac_evecs_neg, y_mat);
      auto h_d_bar = matmul('N', 'T', n2, n2, n_pos, cn_y, state.dirac_evecs_pos);
      symmetrize_inplace(h_d_bar, n2);

      auto cn_ye = matmul('N', 'N', n2, n_pos, n_neg, state.dirac_evecs_neg, y_epos);
      auto m_d_bar = matmul('N', 'T', n2, n2, n_pos, cn_ye, state.dirac_evecs_pos, -1.0);
      symmetrize_inplace(m_d_bar, n2);

      for (std::size_t i = 0; i < np; ++i) {
        for (std::size_t j = 0; j < np; ++j) {
          const std::size_t ij = i * np + j;
          s_bar[ij] += m_d_bar[i * n2 + j];
          v_bar[ij] += h_d_bar[i * n2 + j];
          t_bar[ij] += h_d_bar[i * n2 + (np + j)] + h_d_bar[(np + i) * n2 + j] -
                       h_d_bar[(np + i) * n2 + (np + j)] +
                       a_s * m_d_bar[(np + i) * n2 + (np + j)];
          w_bar[ij] += a_w * h_d_bar[(np + i) * n2 + (np + j)];
        }
      }
    }
  }

  if (delta_only) {
    for (std::size_t k = 0; k < np * np; ++k) {
      t_bar[k] -= wh_prim[k];
      v_bar[k] -= wh_prim[k];
    }
  }
  symmetrize_inplace(s_bar, np);
  symmetrize_inplace(t_bar, np);
  symmetrize_inplace(v_bar, np);
  symmetrize_inplace(w_bar, np);

  return contract_weighted_stvw_derivative(state.uncontracted_system, s_bar, t_bar, v_bar, w_bar);
}

std::vector<double> build_x2c1e_hcore_derivatives(const X2c1eState& state, bool delta_only) {
  const std::size_t nc = state.n_contr;
  const std::size_t natom = state.uncontracted_system.atoms.size();
  const std::size_t ncoord = natom * 3;
  const std::size_t nc2 = checked_mul(nc, nc);
  std::vector<double> out(checked_mul(ncoord, nc2), 0.0);
  if (state.mode == X2c1eMode::None || nc == 0 || ncoord == 0) return out;

  const std::size_t np = state.n_prim;
  const std::size_t np2 = checked_mul(np, np);
  if (!state.has_ae_atoms) {
    if (delta_only) return out;
    const auto derivs = build_stvw_derivatives(state.uncontracted_system);
    std::vector<double> dh_p(np2);
    for (std::size_t coord = 0; coord < ncoord; ++coord) {
      const double* dt = derivs.d_kinetic.data() + coord * np2;
      const double* dv = derivs.d_attraction.data() + coord * np2;
      for (std::size_t k = 0; k < np2; ++k) dh_p[k] = dt[k] + dv[k];
      auto dh_c = congruence_ct_a_c(state.contraction, np, nc, dh_p);
      symmetrize_inplace(dh_c, nc);
      std::copy(dh_c.begin(), dh_c.end(), out.begin() + coord * nc2);
    }
    return out;
  }

  const auto derivs = build_stvw_derivatives(state.uncontracted_system);
  const double c = state.speed_of_light;
  const double a_s = 0.5 / (c * c);
  const double a_w = 0.25 / (c * c);

  std::vector<double> hnesc_r;
  if (state.mode == X2c1eMode::Dlu1e || state.mode == X2c1eMode::Global1e) {
    hnesc_r = matmul('N', 'N', np, np, np, state.h_nesc, state.metric.r_mat);
  }

  // Precompute Global1e invariant Dirac projection factors.
  std::vector<double> diff_b;
  std::vector<double> t_x, t_dx, w_x;
  if (state.mode == X2c1eMode::Global1e) {
    const std::size_t n_neg = state.dirac_neg_count;
    std::vector<double> bl(np * n_neg), bs(np * n_neg);
    for (std::size_t i = 0; i < np; ++i) {
      for (std::size_t q = 0; q < n_neg; ++q) {
        bl[i * n_neg + q] = state.dirac_evecs_neg[i * n_neg + q];
        bs[i * n_neg + q] = state.dirac_evecs_neg[(np + i) * n_neg + q];
      }
    }
    auto x_bl = matmul('N', 'N', np, n_neg, np, state.x_mat, bl);
    diff_b.resize(np * n_neg);
    for (std::size_t k = 0; k < np * n_neg; ++k) diff_b[k] = bs[k] - x_bl[k];
    t_x = matmul('N', 'N', np, np, np, state.t_prim, state.x_mat);
    t_dx = matmul('N', 'N', np, np, np, state.t_prim, state.dx_mat);
    w_x = matmul('N', 'N', np, np, np, state.w_prim, state.x_mat);
  }

  std::vector<double> ds(np2), dt(np2), dv(np2), dw(np2);
  for (std::size_t coord = 0; coord < ncoord; ++coord) {
    std::copy_n(derivs.d_overlap.data() + coord * np2, np2, ds.begin());
    std::copy_n(derivs.d_kinetic.data() + coord * np2, np2, dt.begin());
    std::copy_n(derivs.d_attraction.data() + coord * np2, np2, dv.begin());
    std::copy_n(derivs.d_small_potential.data() + coord * np2, np2, dw.begin());

    std::vector<double> dh_p(np2, 0.0);
    if (state.mode == X2c1eMode::DluLocalR) {
      auto ul_dv_ul = congruence_ct_a_c(state.ul_mat, np, np, dv);
      auto dt_us = matmul('N', 'N', np, np, np, dt, state.us_mat);
      auto ul_dt_us = matmul('T', 'N', np, np, np, state.ul_mat, dt_us);
      auto us_dt_us = matmul('T', 'N', np, np, np, state.us_mat, dt_us);
      auto uw_dw_uw = congruence_ct_a_c(state.uw_mat, np, np, dw);
      for (std::size_t i = 0; i < np; ++i)
        for (std::size_t j = 0; j < np; ++j)
          dh_p[i * np + j] = ul_dv_ul[i * np + j] + ul_dt_us[i * np + j] + ul_dt_us[j * np + i] -
                             us_dt_us[i * np + j] + a_w * uw_dw_uw[i * np + j];
    } else {
      auto xw_dt_xw = congruence_ct_a_c(state.xw_mat, np, np, dt);
      auto dx_dt_dx = congruence_ct_a_c(state.dx_mat, np, np, dt);
      auto xw_dw_xw = congruence_ct_a_c(state.xw_mat, np, np, dw);
      std::vector<double> ds_nesc(np2), dh_nesc(np2);
      for (std::size_t k = 0; k < np2; ++k) {
        ds_nesc[k] = ds[k] + a_s * xw_dt_xw[k];
        dh_nesc[k] = dv[k] + dt[k] - dx_dt_dx[k] + a_w * xw_dw_xw[k];
      }

      if (state.mode == X2c1eMode::Global1e) {
        const std::size_t n_neg = state.dirac_neg_count;
        const std::size_t n_pos = state.dirac_pos_count;
        const std::size_t n2 = 2 * np;
        std::vector<double> dh_d(n2 * n2, 0.0), dm_d(n2 * n2, 0.0);
        for (std::size_t i = 0; i < np; ++i) {
          for (std::size_t j = 0; j < np; ++j) {
            const std::size_t ij = i * np + j;
            dh_d[i * n2 + j] = dv[ij];
            dh_d[i * n2 + (np + j)] = dt[ij];
            dh_d[(np + i) * n2 + j] = dt[ij];
            dh_d[(np + i) * n2 + (np + j)] = a_w * dw[ij] - dt[ij];
            dm_d[i * n2 + j] = ds[ij];
            dm_d[(np + i) * n2 + (np + j)] = a_s * dt[ij];
          }
        }
        auto dh_cp = matmul('N', 'N', n2, n_pos, n2, dh_d, state.dirac_evecs_pos);
        auto h1_np = matmul('T', 'N', n_neg, n_pos, n2, state.dirac_evecs_neg, dh_cp);
        auto dm_cp = matmul('N', 'N', n2, n_pos, n2, dm_d, state.dirac_evecs_pos);
        auto m1_np = matmul('T', 'N', n_neg, n_pos, n2, state.dirac_evecs_neg, dm_cp);

        std::vector<double> u_np(n_neg * n_pos);
        for (std::size_t q = 0; q < n_neg; ++q) {
          for (std::size_t p = 0; p < n_pos; ++p) {
            u_np[q * n_pos + p] =
                (h1_np[q * n_pos + p] - m1_np[q * n_pos + p] * state.dirac_evals_pos[p]) /
                (state.dirac_evals_pos[p] - state.dirac_evals_neg[q]);
          }
        }
        auto u_alinv = matmul('N', 'N', n_neg, np, n_pos, u_np, state.dirac_al_inv);
        auto dx_dir = matmul('N', 'N', np, np, n_neg, diff_b, u_alinv);

        auto dxt_tx = matmul('T', 'N', np, np, np, dx_dir, t_x);
        auto dxt_tdx = matmul('T', 'N', np, np, np, dx_dir, t_dx);
        auto dxt_wx = matmul('T', 'N', np, np, np, dx_dir, w_x);
        for (std::size_t i = 0; i < np; ++i) {
          for (std::size_t j = 0; j < np; ++j) {
            ds_nesc[i * np + j] += a_s * (dxt_tx[i * np + j] + dxt_tx[j * np + i]);
            dh_nesc[i * np + j] += -(dxt_tdx[i * np + j] + dxt_tdx[j * np + i]) +
                                   a_w * (dxt_wx[i * np + j] + dxt_wx[j * np + i]);
          }
        }
      }

      symmetrize_inplace(ds_nesc, np);
      symmetrize_inplace(dh_nesc, np);
      auto dr = forward_dr_metric(state.metric, ds, ds_nesc);
      auto drt_hr = matmul('T', 'N', np, np, np, dr, hnesc_r);
      auto rt_dh_r = congruence_ct_a_c(state.metric.r_mat, np, np, dh_nesc);
      for (std::size_t i = 0; i < np; ++i)
        for (std::size_t j = 0; j < np; ++j)
          dh_p[i * np + j] = drt_hr[i * np + j] + drt_hr[j * np + i] + rt_dh_r[i * np + j];
    }

    if (delta_only) {
      for (std::size_t k = 0; k < np2; ++k) dh_p[k] -= (dt[k] + dv[k]);
    }
    auto dh_c = congruence_ct_a_c(state.contraction, np, nc, dh_p);
    symmetrize_inplace(dh_c, nc);
    std::copy(dh_c.begin(), dh_c.end(), out.begin() + coord * nc2);
  }
  return out;
}

void apply_x2c1e_to_integrals(const core::System& system, IntegralData& integrals,
                              bool include_derivatives) {
  const auto mode = static_cast<X2c1eMode>(system.x2c1e_mode);
  if (mode == X2c1eMode::None) return;
  const double c =
      system.x2c1e_speed_of_light > 0.0 ? system.x2c1e_speed_of_light : kSpeedOfLightAu;
  const auto state = build_x2c1e_state(system, mode, c);
  const std::size_t n2 = checked_mul(integrals.nbf, integrals.nbf);
  if (state->hcore_delta.size() != n2 || integrals.hcore.size() != n2)
    throw std::runtime_error("X2C-1e delta hcore dimensions disagree with integral buffer");
  for (std::size_t k = 0; k < n2; ++k) integrals.hcore[k] += state->hcore_delta[k];
  if (include_derivatives && !integrals.hcore_derivative.empty()) {
    const auto d_delta = build_x2c1e_hcore_derivatives(*state, /*delta_only=*/true);
    if (d_delta.size() != integrals.hcore_derivative.size())
      throw std::runtime_error("X2C-1e derivative dimensions disagree with integral buffer");
    for (std::size_t k = 0; k < d_delta.size(); ++k) integrals.hcore_derivative[k] += d_delta[k];
  }
}

namespace {

struct TuvTriple {
  unsigned t{};
  unsigned u{};
  unsigned v{};
};

std::vector<TuvTriple> build_tuv_list(unsigned max_l) {
  std::vector<TuvTriple> list;
  for (unsigned t = 0; t <= max_l; ++t)
    for (unsigned u = 0; t + u <= max_l; ++u)
      for (unsigned v = 0; t + u + v <= max_l; ++v) list.push_back({t, u, v});
  return list;
}

struct KetTerm {
  unsigned tau{};
  unsigned nu{};
  unsigned phi{};
  double coeff{};  // includes (-1)^(tau+nu+phi) * prim_coeff * cart_to_pub
};

struct AuxPrimKet {
  double exponent{};
  std::size_t n_ao{};
  std::vector<std::vector<KetTerm>> ao_terms;  // [m_ao] -> sparse Hermite terms
};

struct AuxShellKet {
  std::size_t atom_index{};
  unsigned angular_momentum{};
  std::size_t ao_offset{};
  std::size_t n_ao{};
  std::vector<AuxPrimKet> primitives;
};

std::vector<AuxShellKet> build_aux_ket_table(const core::System& auxiliary,
                                             const std::vector<ShellProjection>& aux_projs) {
  std::vector<AuxShellKet> out(auxiliary.shells.size());
  ScalarHermite h1d;
  for (std::size_t sp = 0; sp < auxiliary.shells.size(); ++sp) {
    const auto& sh = auxiliary.shells[sp];
    const auto& proj = aux_projs[sp];
    const unsigned lc = sh.angular_momentum;
    AuxShellKet sk;
    sk.atom_index = sh.atom_index;
    sk.angular_momentum = lc;
    sk.ao_offset = proj.ao_offset;
    sk.n_ao = proj.n_ao;
    sk.primitives.reserve(sh.primitives.size());

    for (const auto& pk : sh.primitives) {
      AuxPrimKet prim;
      prim.exponent = pk.exponent;
      prim.n_ao = proj.n_ao;
      prim.ao_terms.resize(proj.n_ao);
      fill_hermite_scalar(lc, 0, 0.0, 0.0, 0.0, pk.exponent, 0.0, h1d);
      const unsigned dim = lc + 1;
      const std::size_t d3 = static_cast<std::size_t>(dim) * dim * dim;
      std::vector<double> accum(d3, 0.0);
      for (std::size_t m = 0; m < proj.n_ao; ++m) {
        std::fill(accum.begin(), accum.end(), 0.0);
        for (std::size_t ck = 0; ck < proj.n_cart; ++ck) {
          const double w = proj.cart_to_pub[m * proj.n_cart + ck] * pk.coefficient;
          if (w == 0.0) continue;
          const auto& ac = proj.cart_comps[ck];
          for (unsigned tau = 0; tau <= ac[0]; ++tau) {
            const double ex = h1d.get(ac[0], 0, tau);
            if (ex == 0.0) continue;
            for (unsigned nu = 0; nu <= ac[1]; ++nu) {
              const double ey = h1d.get(ac[1], 0, nu);
              if (ey == 0.0) continue;
              const double wxy = w * ex * ey;
              for (unsigned phi = 0; phi <= ac[2]; ++phi) {
                const double ez = h1d.get(ac[2], 0, phi);
                if (ez == 0.0) continue;
                accum[(static_cast<std::size_t>(tau) * dim + nu) * dim + phi] += wxy * ez;
              }
            }
          }
        }
        auto& terms = prim.ao_terms[m];
        for (unsigned tau = 0; tau <= lc; ++tau) {
          for (unsigned nu = 0; tau + nu <= lc; ++nu) {
            for (unsigned phi = 0; tau + nu + phi <= lc; ++phi) {
              const double v = accum[(static_cast<std::size_t>(tau) * dim + nu) * dim + phi];
              if (std::abs(v) > 1.0e-18) {
                const double sign = ((tau + nu + phi) & 1U) == 0 ? 1.0 : -1.0;
                terms.push_back({tau, nu, phi, sign * v});
              }
            }
          }
        }
      }
      sk.primitives.push_back(std::move(prim));
    }
    out[sp] = std::move(sk);
  }
  return out;
}

unsigned df_worker_count(std::size_t work_items) {
  unsigned hw = std::thread::hardware_concurrency();
  if (hw == 0) hw = 4;
  unsigned n = std::min<unsigned>(hw, 8U);
  if (work_items < n) n = static_cast<unsigned>(std::max<std::size_t>(1, work_items));
  return n;
}

}  // namespace

void build_df_integrals_fast(const core::System& orbital, const core::System& auxiliary,
                             std::span<double> metric_out, std::span<double> three_center_out) {
  std::size_t nbf = 0;
  std::size_t naux = 0;
  const auto orb_projs = build_shell_projections(orbital, nbf);
  const auto aux_projs = build_shell_projections(auxiliary, naux);
  if (metric_out.size() != checked_mul(naux, naux) ||
      three_center_out.size() != checked_mul(checked_mul(nbf, nbf), naux))
    throw std::invalid_argument("fast CPU DF value output sizes disagree with system dimensions");
  std::fill(metric_out.begin(), metric_out.end(), 0.0);
  std::fill(three_center_out.begin(), three_center_out.end(), 0.0);
  if (nbf == 0 || naux == 0) return;

  const auto aux_kets = build_aux_ket_table(auxiliary, aux_projs);
  const double two_pi_52 = 2.0 * std::pow(std::numbers::pi, 2.5);

  // 1. Two-center Coulomb metric (P|Q)
  {
    std::vector<double> boys_buf;
    ScalarCoulomb coul;
    for (std::size_t sp = 0; sp < aux_kets.size(); ++sp) {
      const auto& sk_p = aux_kets[sp];
      const auto& pos_p = auxiliary.atoms[sk_p.atom_index].position;
      for (std::size_t sq = sp; sq < aux_kets.size(); ++sq) {
        const auto& sk_q = aux_kets[sq];
        const auto& pos_q = auxiliary.atoms[sk_q.atom_index].position;
        const unsigned max_order = sk_p.angular_momentum + sk_q.angular_momentum;
        for (const auto& pp : sk_p.primitives) {
          for (const auto& pq : sk_q.primitives) {
            const double a = pp.exponent;
            const double b = pq.exponent;
            const double rho = a * b / (a + b);
            const double pref = two_pi_52 / (a * b * std::sqrt(a + b));
            fill_coulomb_scalar(max_order, rho, pos_p, pos_q, boys_buf, coul);
            for (std::size_t mp = 0; mp < sk_p.n_ao; ++mp) {
              const std::size_t gp = sk_p.ao_offset + mp;
              const auto& tp = pp.ao_terms[mp];
              for (std::size_t mq = 0; mq < sk_q.n_ao; ++mq) {
                const std::size_t gq = sk_q.ao_offset + mq;
                const auto& tq = pq.ao_terms[mq];
                double val = 0.0;
                for (const auto& bp : tp) {
                  // Undo ket (-1)^(t+u+v) sign for the bra center of (P|Q).
                  const double b_coeff =
                      (((bp.tau + bp.nu + bp.phi) & 1U) == 0 ? 1.0 : -1.0) * bp.coeff;
                  for (const auto& kq : tq) {
                    val += b_coeff * kq.coeff *
                           coul.at(0, bp.tau + kq.tau, bp.nu + kq.nu, bp.phi + kq.phi);
                  }
                }
                const double contrib = pref * val;
                metric_out[gp * naux + gq] += contrib;
                if (sp != sq) metric_out[gq * naux + gp] += contrib;
              }
            }
          }
        }
      }
    }
  }

  // 2. Three-center Coulomb (mu nu | P), parallelized across orbital shell pairs (si <= sj).
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  pairs.reserve(orbital.shells.size() * (orbital.shells.size() + 1) / 2);
  for (std::size_t si = 0; si < orbital.shells.size(); ++si)
    for (std::size_t sj = si; sj < orbital.shells.size(); ++sj) pairs.emplace_back(si, sj);

  const unsigned nthreads = df_worker_count(pairs.size());
  std::atomic<std::size_t> next_pair{0};

  auto worker = [&]() {
    std::vector<double> boys_buf;
    ScalarCoulomb coul;
    std::array<ScalarHermite, 3> hermite;
    std::vector<double> bra_pub;
    std::vector<double> r_ket;
    std::vector<double> block;

    while (true) {
      const std::size_t pidx = next_pair.fetch_add(1, std::memory_order_relaxed);
      if (pidx >= pairs.size()) break;
      const auto [si, sj] = pairs[pidx];
      const auto& shell_i = orbital.shells[si];
      const auto& shell_j = orbital.shells[sj];
      const auto& proj_i = orb_projs[si];
      const auto& proj_j = orb_projs[sj];
      const unsigned li = shell_i.angular_momentum;
      const unsigned lj = shell_j.angular_momentum;
      const auto& pos_a = orbital.atoms[shell_i.atom_index].position;
      const auto& pos_b = orbital.atoms[shell_j.atom_index].position;
      const double dx_ab = pos_a[0] - pos_b[0];
      const double dy_ab = pos_a[1] - pos_b[1];
      const double dz_ab = pos_a[2] - pos_b[2];
      const double r2_ab = dx_ab * dx_ab + dy_ab * dy_ab + dz_ab * dz_ab;

      const auto tuv_list = build_tuv_list(li + lj);
      const std::size_t ntuv = tuv_list.size();
      const std::size_t n_ao_pair = proj_i.n_ao * proj_j.n_ao;
      block.assign(n_ao_pair * naux, 0.0);
      bra_pub.resize(n_ao_pair * ntuv);
      r_ket.resize(naux * ntuv);

      for (const auto& pi : shell_i.primitives) {
        for (const auto& pj : shell_j.primitives) {
          const double alpha = pi.exponent;
          const double beta = pj.exponent;
          const double p = alpha + beta;
          const double mu = alpha * beta / p;
          if (mu * r2_ab > 50.0) continue;
          const double prim_coeff = pi.coefficient * pj.coefficient;
          if (prim_coeff == 0.0) continue;

          const std::array<double, 3> product{
              (alpha * pos_a[0] + beta * pos_b[0]) / p,
              (alpha * pos_a[1] + beta * pos_b[1]) / p,
              (alpha * pos_a[2] + beta * pos_b[2]) / p,
          };
          for (std::size_t ax = 0; ax < 3; ++ax)
            fill_hermite_scalar(li, lj, product[ax], pos_a[ax], pos_b[ax], alpha, beta,
                                hermite[ax]);

          // Build bra_pub (n_ao_pair x ntuv)
          std::fill(bra_pub.begin(), bra_pub.end(), 0.0);
          for (std::size_t ci = 0; ci < proj_i.n_cart; ++ci) {
            const auto& ai = proj_i.cart_comps[ci];
            for (std::size_t cj = 0; cj < proj_j.n_cart; ++cj) {
              const auto& bj = proj_j.cart_comps[cj];
              for (std::size_t k = 0; k < ntuv; ++k) {
                const auto [t, u, v] = tuv_list[k];
                if (t > ai[0] + bj[0] || u > ai[1] + bj[1] || v > ai[2] + bj[2]) continue;
                const double e_val = prim_coeff * hermite[0].get(ai[0], bj[0], t) *
                                     hermite[1].get(ai[1], bj[1], u) *
                                     hermite[2].get(ai[2], bj[2], v);
                if (e_val == 0.0) continue;
                for (std::size_t mi = 0; mi < proj_i.n_ao; ++mi) {
                  const double ti = proj_i.cart_to_pub[mi * proj_i.n_cart + ci];
                  if (ti == 0.0) continue;
                  const double tie = ti * e_val;
                  for (std::size_t mj = 0; mj < proj_j.n_ao; ++mj) {
                    const double tj = proj_j.cart_to_pub[mj * proj_j.n_cart + cj];
                    if (tj == 0.0) continue;
                    bra_pub[(mi * proj_j.n_ao + mj) * ntuv + k] += tie * tj;
                  }
                }
              }
            }
          }

          // Build r_ket (naux x ntuv) across all auxiliary shells
          std::fill(r_ket.begin(), r_ket.end(), 0.0);
          for (const auto& sk : aux_kets) {
            const auto& pos_c = auxiliary.atoms[sk.atom_index].position;
            const unsigned max_order = li + lj + sk.angular_momentum;
            for (const auto& pk : sk.primitives) {
              const double gamma = pk.exponent;
              const double rho = p * gamma / (p + gamma);
              const double pref = two_pi_52 / (p * gamma * std::sqrt(p + gamma));
              fill_coulomb_scalar(max_order, rho, product, pos_c, boys_buf, coul);
              for (std::size_t ma = 0; ma < sk.n_ao; ++ma) {
                double* row = r_ket.data() + (sk.ao_offset + ma) * ntuv;
                for (const auto& kt : pk.ao_terms[ma]) {
                  const double sc = pref * kt.coeff;
                  for (std::size_t k = 0; k < ntuv; ++k) {
                    const auto [t, u, v] = tuv_list[k];
                    row[k] += sc * coul.at(0, t + kt.tau, u + kt.nu, v + kt.phi);
                  }
                }
              }
            }
          }

          // Accumulate block += bra_pub @ r_ket^T
          for (std::size_t m = 0; m < n_ao_pair; ++m) {
            const double* b_row = bra_pub.data() + m * ntuv;
            double* out_row = block.data() + m * naux;
            for (std::size_t g = 0; g < naux; ++g) {
              const double* r_row = r_ket.data() + g * ntuv;
              double s = 0.0;
              for (std::size_t k = 0; k < ntuv; ++k) s += b_row[k] * r_row[k];
              out_row[g] += s;
            }
          }
        }
      }

      // Scatter block into public (nbf, nbf, naux) tensor (disjoint slots per (si, sj) pair).
      for (std::size_t mi = 0; mi < proj_i.n_ao; ++mi) {
        const std::size_t gi = proj_i.ao_offset + mi;
        for (std::size_t mj = 0; mj < proj_j.n_ao; ++mj) {
          const std::size_t gj = proj_j.ao_offset + mj;
          const double* src = block.data() + (mi * proj_j.n_ao + mj) * naux;
          double* dst1 = three_center_out.data() + (gi * nbf + gj) * naux;
          std::copy_n(src, naux, dst1);
          if (si != sj) {
            double* dst2 = three_center_out.data() + (gj * nbf + gi) * naux;
            std::copy_n(src, naux, dst2);
          }
        }
      }
    }
  };

  if (nthreads <= 1) {
    worker();
  } else {
    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
  }
}

std::vector<double> contract_weighted_df_derivative_fast(
    const core::System& orbital, const core::System& auxiliary,
    std::span<const double> metric_weights, std::span<const double> three_center_weights) {
  std::size_t nbf = 0;
  std::size_t naux = 0;
  const auto orb_projs = build_shell_projections(orbital, nbf);
  const auto aux_projs = build_shell_projections(auxiliary, naux);
  const std::size_t natom = orbital.atoms.size();
  const std::size_t ncoord = natom * 3;
  if (metric_weights.size() != checked_mul(naux, naux) ||
      three_center_weights.size() != checked_mul(checked_mul(nbf, nbf), naux))
    throw std::invalid_argument("fast CPU DF gradient weight sizes disagree with system");
  std::vector<double> grad(ncoord, 0.0);
  if (nbf == 0 || naux == 0 || natom == 0) return grad;

  const auto aux_kets = build_aux_ket_table(auxiliary, aux_projs);
  const double two_pi_52 = 2.0 * std::pow(std::numbers::pi, 2.5);

  // 1. Two-center metric derivative contraction (sp < sq, atom_p != atom_q)
  {
    std::vector<double> boys_buf;
    ScalarCoulomb coul;
    for (std::size_t sp = 0; sp < aux_kets.size(); ++sp) {
      const auto& sk_p = aux_kets[sp];
      const auto& pos_p = auxiliary.atoms[sk_p.atom_index].position;
      for (std::size_t sq = sp + 1; sq < aux_kets.size(); ++sq) {
        const auto& sk_q = aux_kets[sq];
        if (sk_p.atom_index == sk_q.atom_index) continue;
        const auto& pos_q = auxiliary.atoms[sk_q.atom_index].position;
        const unsigned max_order = sk_p.angular_momentum + sk_q.angular_momentum + 1;
        for (const auto& pp : sk_p.primitives) {
          for (const auto& pq : sk_q.primitives) {
            const double a = pp.exponent;
            const double b = pq.exponent;
            const double rho = a * b / (a + b);
            const double pref = two_pi_52 / (a * b * std::sqrt(a + b));
            fill_coulomb_scalar(max_order, rho, pos_p, pos_q, boys_buf, coul);
            std::array<double, 3> da{0.0, 0.0, 0.0};
            for (std::size_t mp = 0; mp < sk_p.n_ao; ++mp) {
              const std::size_t gp = sk_p.ao_offset + mp;
              const auto& tp = pp.ao_terms[mp];
              for (std::size_t mq = 0; mq < sk_q.n_ao; ++mq) {
                const std::size_t gq = sk_q.ao_offset + mq;
                // sym_scale = 2.0 for sp < sq with symmetrized metric weights.
                const double wm =
                    metric_weights[gp * naux + gq] + metric_weights[gq * naux + gp];
                if (wm == 0.0) continue;
                const auto& tq = pq.ao_terms[mq];
                for (const auto& bp : tp) {
                  const double wb =
                      wm * ((((bp.tau + bp.nu + bp.phi) & 1U) == 0 ? 1.0 : -1.0) * bp.coeff);
                  for (const auto& kq : tq) {
                    const double wbk = wb * kq.coeff;
                    const unsigned T = bp.tau + kq.tau;
                    const unsigned U = bp.nu + kq.nu;
                    const unsigned V = bp.phi + kq.phi;
                    da[0] += wbk * coul.at(0, T + 1, U, V);
                    da[1] += wbk * coul.at(0, T, U + 1, V);
                    da[2] += wbk * coul.at(0, T, U, V + 1);
                  }
                }
              }
            }
            for (std::size_t ax = 0; ax < 3; ++ax) {
              const double v = pref * da[ax];
              grad[3 * sk_p.atom_index + ax] += v;
              grad[3 * sk_q.atom_index + ax] -= v;
            }
          }
        }
      }
    }
  }

  // 2. Three-center weighted derivative, parallelized across orbital shell pairs (si <= sj).
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  pairs.reserve(orbital.shells.size() * (orbital.shells.size() + 1) / 2);
  for (std::size_t si = 0; si < orbital.shells.size(); ++si)
    for (std::size_t sj = si; sj < orbital.shells.size(); ++sj) pairs.emplace_back(si, sj);

  const unsigned nthreads = df_worker_count(pairs.size());
  std::vector<std::vector<double>> thread_grads(nthreads, std::vector<double>(ncoord, 0.0));
  std::atomic<std::size_t> next_pair{0};

  auto worker = [&](unsigned tid) {
    auto& g_loc = thread_grads[tid];
    std::vector<double> boys_buf;
    ScalarCoulomb coul;
    std::array<ScalarHermite, 3> hermite;
    std::vector<double> w_block;
    std::vector<double> bra_pub;
    std::array<std::vector<double>, 3> bra_da;
    std::vector<double> h_bra;
    std::array<std::vector<double>, 3> h_da;

    while (true) {
      const std::size_t pidx = next_pair.fetch_add(1, std::memory_order_relaxed);
      if (pidx >= pairs.size()) break;
      const auto [si, sj] = pairs[pidx];
      const auto& shell_i = orbital.shells[si];
      const auto& shell_j = orbital.shells[sj];
      const auto& proj_i = orb_projs[si];
      const auto& proj_j = orb_projs[sj];
      const unsigned li = shell_i.angular_momentum;
      const unsigned lj = shell_j.angular_momentum;
      const bool same_ab = (shell_i.atom_index == shell_j.atom_index);
      const double sym_scale = (si == sj) ? 1.0 : 2.0;

      const std::size_t n_ao_pair = proj_i.n_ao * proj_j.n_ao;
      w_block.resize(naux * n_ao_pair);
      double max_w = 0.0;
      for (std::size_t mi = 0; mi < proj_i.n_ao; ++mi) {
        const std::size_t gi = proj_i.ao_offset + mi;
        for (std::size_t mj = 0; mj < proj_j.n_ao; ++mj) {
          const std::size_t gj = proj_j.ao_offset + mj;
          const std::size_t m = mi * proj_j.n_ao + mj;
          const double* w1 = three_center_weights.data() + (gi * nbf + gj) * naux;
          const double* w2 = three_center_weights.data() + (gj * nbf + gi) * naux;
          for (std::size_t g = 0; g < naux; ++g) {
            const double val = sym_scale * 0.5 * (w1[g] + w2[g]);
            w_block[g * n_ao_pair + m] = val;
            max_w = std::max(max_w, std::abs(val));
          }
        }
      }
      if (max_w == 0.0) continue;

      const auto& pos_a = orbital.atoms[shell_i.atom_index].position;
      const auto& pos_b = orbital.atoms[shell_j.atom_index].position;
      const double dx_ab = pos_a[0] - pos_b[0];
      const double dy_ab = pos_a[1] - pos_b[1];
      const double dz_ab = pos_a[2] - pos_b[2];
      const double r2_ab = dx_ab * dx_ab + dy_ab * dy_ab + dz_ab * dz_ab;

      const auto tuv_list = build_tuv_list(li + lj);
      const std::size_t ntuv = tuv_list.size();
      const auto tuv_da_list = same_ab ? std::vector<TuvTriple>{} : build_tuv_list(li + lj + 1);
      const std::size_t ntuv_da = tuv_da_list.size();

      bra_pub.resize(n_ao_pair * ntuv);
      h_bra.resize(naux * ntuv);
      if (!same_ab) {
        for (std::size_t ax = 0; ax < 3; ++ax) {
          bra_da[ax].resize(n_ao_pair * ntuv_da);
          h_da[ax].resize(naux * ntuv_da);
        }
      }

      for (const auto& pi : shell_i.primitives) {
        for (const auto& pj : shell_j.primitives) {
          const double alpha = pi.exponent;
          const double beta = pj.exponent;
          const double p = alpha + beta;
          const double mu = alpha * beta / p;
          if (mu * r2_ab > 50.0) continue;
          const double prim_coeff = pi.coefficient * pj.coefficient;
          if (prim_coeff == 0.0) continue;

          const std::array<double, 3> product{
              (alpha * pos_a[0] + beta * pos_b[0]) / p,
              (alpha * pos_a[1] + beta * pos_b[1]) / p,
              (alpha * pos_a[2] + beta * pos_b[2]) / p,
          };
          for (std::size_t ax = 0; ax < 3; ++ax)
            fill_hermite_scalar(li + (same_ab ? 0 : 1), lj, product[ax], pos_a[ax], pos_b[ax],
                                alpha, beta, hermite[ax]);

          // Build bra_pub and (if !same_ab) bra_da[0..2]
          std::fill(bra_pub.begin(), bra_pub.end(), 0.0);
          if (!same_ab) {
            for (std::size_t ax = 0; ax < 3; ++ax)
              std::fill(bra_da[ax].begin(), bra_da[ax].end(), 0.0);
          }

          for (std::size_t ci = 0; ci < proj_i.n_cart; ++ci) {
            const auto& ai = proj_i.cart_comps[ci];
            for (std::size_t cj = 0; cj < proj_j.n_cart; ++cj) {
              const auto& bj = proj_j.cart_comps[cj];
              for (std::size_t k = 0; k < ntuv; ++k) {
                const auto [t, u, v] = tuv_list[k];
                if (t > ai[0] + bj[0] || u > ai[1] + bj[1] || v > ai[2] + bj[2]) continue;
                const double ev = prim_coeff * hermite[0].get(ai[0], bj[0], t) *
                                  hermite[1].get(ai[1], bj[1], u) *
                                  hermite[2].get(ai[2], bj[2], v);
                if (ev == 0.0) continue;
                for (std::size_t mi = 0; mi < proj_i.n_ao; ++mi) {
                  const double ti = proj_i.cart_to_pub[mi * proj_i.n_cart + ci];
                  if (ti == 0.0) continue;
                  const double tie = ti * ev;
                  for (std::size_t mj = 0; mj < proj_j.n_ao; ++mj) {
                    const double tj = proj_j.cart_to_pub[mj * proj_j.n_cart + cj];
                    if (tj != 0.0) bra_pub[(mi * proj_j.n_ao + mj) * ntuv + k] += tie * tj;
                  }
                }
              }
              if (!same_ab) {
                for (std::size_t k = 0; k < ntuv_da; ++k) {
                  const auto [t, u, v] = tuv_da_list[k];
                  if (t > ai[0] + bj[0] + 1 || u > ai[1] + bj[1] + 1 || v > ai[2] + bj[2] + 1)
                    continue;
                  const double ex = hermite[0].get(ai[0], bj[0], t);
                  const double ey = hermite[1].get(ai[1], bj[1], u);
                  const double ez = hermite[2].get(ai[2], bj[2], v);
                  const double gx =
                      prim_coeff * hermite_da_e(hermite[0], ai[0], bj[0], t, alpha) * ey * ez;
                  const double gy =
                      prim_coeff * ex * hermite_da_e(hermite[1], ai[1], bj[1], u, alpha) * ez;
                  const double gz =
                      prim_coeff * ex * ey * hermite_da_e(hermite[2], ai[2], bj[2], v, alpha);
                  if (gx == 0.0 && gy == 0.0 && gz == 0.0) continue;
                  for (std::size_t mi = 0; mi < proj_i.n_ao; ++mi) {
                    const double ti = proj_i.cart_to_pub[mi * proj_i.n_cart + ci];
                    if (ti == 0.0) continue;
                    for (std::size_t mj = 0; mj < proj_j.n_ao; ++mj) {
                      const double tj = proj_j.cart_to_pub[mj * proj_j.n_cart + cj];
                      if (tj == 0.0) continue;
                      const double w = ti * tj;
                      const std::size_t slot = (mi * proj_j.n_ao + mj) * ntuv_da + k;
                      bra_da[0][slot] += w * gx;
                      bra_da[1][slot] += w * gy;
                      bra_da[2][slot] += w * gz;
                    }
                  }
                }
              }
            }
          }

          // Contract w_block(naux x n_ao_pair) @ bra_pub(n_ao_pair x ntuv) -> h_bra(naux x ntuv)
          std::fill(h_bra.begin(), h_bra.end(), 0.0);
          for (std::size_t g = 0; g < naux; ++g) {
            const double* w_row = w_block.data() + g * n_ao_pair;
            double* h_row = h_bra.data() + g * ntuv;
            for (std::size_t m = 0; m < n_ao_pair; ++m) {
              const double wm = w_row[m];
              if (wm == 0.0) continue;
              const double* b_row = bra_pub.data() + m * ntuv;
              for (std::size_t k = 0; k < ntuv; ++k) h_row[k] += wm * b_row[k];
            }
          }
          if (!same_ab) {
            for (std::size_t ax = 0; ax < 3; ++ax) {
              std::fill(h_da[ax].begin(), h_da[ax].end(), 0.0);
              for (std::size_t g = 0; g < naux; ++g) {
                const double* w_row = w_block.data() + g * n_ao_pair;
                double* h_row = h_da[ax].data() + g * ntuv_da;
                for (std::size_t m = 0; m < n_ao_pair; ++m) {
                  const double wm = w_row[m];
                  if (wm == 0.0) continue;
                  const double* b_row = bra_da[ax].data() + m * ntuv_da;
                  for (std::size_t k = 0; k < ntuv_da; ++k) h_row[k] += wm * b_row[k];
                }
              }
            }
          }

          // Loop over auxiliary shells and contract against Coulomb auxiliary tensor R.
          for (const auto& sk : aux_kets) {
            if (same_ab && sk.atom_index == shell_i.atom_index) continue;
            const auto& pos_c = auxiliary.atoms[sk.atom_index].position;
            const unsigned max_order = li + lj + sk.angular_momentum + 1;
            for (const auto& pk : sk.primitives) {
              const double gamma = pk.exponent;
              const double rho = p * gamma / (p + gamma);
              const double pref = two_pi_52 / (p * gamma * std::sqrt(p + gamma));
              fill_coulomb_scalar(max_order, rho, product, pos_c, boys_buf, coul);

              std::array<double, 3> dc{0.0, 0.0, 0.0};
              std::array<double, 3> da{0.0, 0.0, 0.0};
              for (std::size_t ma = 0; ma < sk.n_ao; ++ma) {
                const std::size_t g = sk.ao_offset + ma;
                const double* hb_row = h_bra.data() + g * ntuv;
                for (const auto& kt : pk.ao_terms[ma]) {
                  const double kc = kt.coeff;
                  double cx = 0.0, cy = 0.0, cz = 0.0;
                  for (std::size_t k = 0; k < ntuv; ++k) {
                    const double hval = hb_row[k];
                    if (hval == 0.0) continue;
                    const auto [t, u, v] = tuv_list[k];
                    const unsigned T = t + kt.tau;
                    const unsigned U = u + kt.nu;
                    const unsigned V = v + kt.phi;
                    cx += hval * coul.at(0, T + 1, U, V);
                    cy += hval * coul.at(0, T, U + 1, V);
                    cz += hval * coul.at(0, T, U, V + 1);
                  }
                  dc[0] -= kc * cx;
                  dc[1] -= kc * cy;
                  dc[2] -= kc * cz;

                  if (!same_ab) {
                    const double* hdx = h_da[0].data() + g * ntuv_da;
                    const double* hdy = h_da[1].data() + g * ntuv_da;
                    const double* hdz = h_da[2].data() + g * ntuv_da;
                    double ax0 = 0.0, ay0 = 0.0, az0 = 0.0;
                    for (std::size_t k = 0; k < ntuv_da; ++k) {
                      const double vx = hdx[k], vy = hdy[k], vz = hdz[k];
                      if (vx == 0.0 && vy == 0.0 && vz == 0.0) continue;
                      const auto [t, u, v] = tuv_da_list[k];
                      const double rval = coul.at(0, t + kt.tau, u + kt.nu, v + kt.phi);
                      ax0 += vx * rval;
                      ay0 += vy * rval;
                      az0 += vz * rval;
                    }
                    da[0] += kc * ax0;
                    da[1] += kc * ay0;
                    da[2] += kc * az0;
                  }
                }
              }

              for (std::size_t ax = 0; ax < 3; ++ax) {
                const double vc = pref * dc[ax];
                g_loc[3 * sk.atom_index + ax] += vc;
                if (same_ab) {
                  g_loc[3 * shell_i.atom_index + ax] -= vc;
                } else {
                  const double va = pref * da[ax];
                  g_loc[3 * shell_i.atom_index + ax] += va;
                  g_loc[3 * shell_j.atom_index + ax] += (-va - vc);
                }
              }
            }
          }
        }
      }
    }
  };

  if (nthreads <= 1) {
    worker(0);
  } else {
    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
  }
  for (unsigned t = 0; t < nthreads; ++t)
    for (std::size_t c = 0; c < ncoord; ++c) grad[c] += thread_grads[t][c];
  return grad;
}

std::vector<double> contract_weighted_one_electron_fast(
    const core::System& system, std::span<const double> overlap_weights,
    std::span<const double> hcore_weights, bool include_nuclear_repulsion) {
  const std::size_t n = molecule::ao_count(system);
  const std::size_t n2 = checked_mul(n, n);
  const std::size_t ncoord = system.atoms.size() * 3U;
  if (overlap_weights.size() != n2 || hcore_weights.size() != n2)
    throw std::invalid_argument("fast one-electron gradient weight sizes mismatch");
  std::vector<double> zeros(n2, 0.0);
  auto grad = contract_weighted_stvw_derivative(system, overlap_weights, hcore_weights,
                                                hcore_weights, zeros);
  if (include_nuclear_repulsion) molecule::add_nuclear_repulsion_gradient(system, grad);
  const bool has_h =
      std::any_of(hcore_weights.begin(), hcore_weights.end(), [](double x) { return x != 0.0; });
  if (has_h && !system.ecp_terms.empty()) {
    const auto g_ecp = contract_weighted_ecp_derivative(system, hcore_weights);
    for (std::size_t c = 0; c < ncoord; ++c) grad[c] += g_ecp[c];
  }
  if (has_h && system.x2c1e_mode != 0) {
    const double c =
        system.x2c1e_speed_of_light > 0.0 ? system.x2c1e_speed_of_light : kSpeedOfLightAu;
    const auto st = build_x2c1e_state(system, static_cast<X2c1eMode>(system.x2c1e_mode), c);
    const auto g_x2c =
        contract_weighted_x2c1e_hcore_derivative(*st, hcore_weights, /*delta_only=*/true);
    for (std::size_t coord = 0; coord < ncoord; ++coord) grad[coord] += g_x2c[coord];
  }
  return grad;
}

void build_one_electron_values_fast(const core::System& system, std::span<double> overlap_out,
                                    std::span<double> hcore_out, double& nuclear_repulsion_out) {
  const std::size_t n = molecule::ao_count(system);
  const std::size_t n2 = checked_mul(n, n);
  if ((!overlap_out.empty() && overlap_out.size() != n2) ||
      (!hcore_out.empty() && hcore_out.size() != n2))
    throw std::invalid_argument("fast one-electron value output sizes mismatch");
  const auto stvw = build_stvw_integrals(system);
  if (!overlap_out.empty()) std::copy(stvw.overlap.begin(), stvw.overlap.end(), overlap_out.begin());
  if (!hcore_out.empty()) {
    for (std::size_t k = 0; k < n2; ++k) hcore_out[k] = stvw.kinetic[k] + stvw.attraction[k];
    if (!system.ecp_terms.empty()) {
      const auto ecp = checked_ecp_integrals(system, false);
      for (std::size_t k = 0; k < n2; ++k) hcore_out[k] += ecp.local[k] + ecp.nonlocal[k];
    }
    if (system.x2c1e_mode != 0) {
      const double c =
          system.x2c1e_speed_of_light > 0.0 ? system.x2c1e_speed_of_light : kSpeedOfLightAu;
      const auto st = build_x2c1e_state(system, static_cast<X2c1eMode>(system.x2c1e_mode), c);
      for (std::size_t k = 0; k < n2; ++k) hcore_out[k] += st->hcore_delta[k];
    }
  }
  double enuc = 0.0;
  for (std::size_t a = 0; a < system.atoms.size(); ++a) {
    for (std::size_t b = 0; b < a; ++b) {
      const double dx = system.atoms[a].position[0] - system.atoms[b].position[0];
      const double dy = system.atoms[a].position[1] - system.atoms[b].position[1];
      const double dz = system.atoms[a].position[2] - system.atoms[b].position[2];
      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      enuc += static_cast<double>(system.atoms[a].ionic_charge()) *
              static_cast<double>(system.atoms[b].ionic_charge()) / r;
    }
  }
  nuclear_repulsion_out = enuc;
}

namespace x2c1e_test_detail {

std::vector<double> eval_r_metric(const std::vector<double>& s, const std::vector<double>& s_nesc,
                                  std::size_t n) {
  return build_r_metric(s, s_nesc, n).r_mat;
}

std::vector<double> eval_forward_dr_metric(const std::vector<double>& s,
                                           const std::vector<double>& s_nesc,
                                           const std::vector<double>& ds,
                                           const std::vector<double>& ds_nesc, std::size_t n) {
  const auto rf = build_r_metric(s, s_nesc, n);
  return forward_dr_metric(rf, ds, ds_nesc);
}

void eval_adjoint_r_metric(const std::vector<double>& s, const std::vector<double>& s_nesc,
                           const std::vector<double>& r_bar, std::size_t n,
                           std::vector<double>& s_bar_out, std::vector<double>& snesc_bar_out) {
  const auto rf = build_r_metric(s, s_nesc, n);
  adjoint_r_metric(rf, r_bar, s_bar_out, snesc_bar_out);
}

}  // namespace x2c1e_test_detail

}  // namespace generativeqc::integrals
