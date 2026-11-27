#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "scf/reference/linalg.hpp"
#include "scf/reference/mean_field.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::scf::solver::uhf_second_order {

/** Real UHF orbital optimization in the rotation parameterization
 * C(x) = C exp(K), K = [[0, -x^T], [x, 0]] per spin, with x an nv-by-no block.
 * With g = C_v^T F C_o, the energy expansion is E(x) = E + 2 g.x + x.A'x + O(x^3),
 * A'x = F_vv x - x F_oo + C_v^T G[D1(x)] C_o and D1 = C_v x C_o^T + C_o x^T C_v^T,
 * so the physical orbital Hessian is 2A' for independent spin rotations.
 * For RHF, the physical curvature of a common rotation is twice that in
 * normalized paired-spin coordinates. These are the real internal stability
 * conventions used by PySCF for UHF and RHF respectively. */
using Vector = std::vector<double>;
using SpinMatrices = std::array<Vector, 2>;

/** Two-electron Fock part G_s = c_J J[D_a + D_b] + c_K K[D_s] of the prepared
 * operator. Hessian actions pass symmetric indefinite transition densities, so
 * the action must be linear in the supplied matrices. */
using TwoElectronAction = std::function<SpinMatrices(const Vector& alpha, const Vector& beta)>;
/** S-orthonormal generalized eigenvectors (columns, ascending energies). */
using Eigensolver = std::function<reference::EigenResult(const Vector& fock)>;

struct Problem {
  std::size_t n{};
  std::array<std::size_t, 2> occupied{};
  const Vector* hcore{};
  double nuclear_repulsion{};
  TwoElectronAction two_electron;
  Eigensolver eigen;
  /** Closed-shell determinants: both spins share one set of orbitals and every
   * rotation lies in the alpha = beta subspace, where the lowest Hessian
   * eigenvalue is that of real internal RHF stability analysis. */
  bool restricted{};
};

/** One determinant and every quantity that one Fock build at it determines. */
struct State {
  SpinMatrices occupied_orbitals;  // n x no, row-major
  SpinMatrices virtual_orbitals;   // n x nv, row-major
  SpinMatrices density, fock;      // n x n
  SpinMatrices fock_oo, fock_vv;   // no x no, nv x nv
  SpinMatrices gradient;           // nv x no
  double energy{};
};

struct Controls {
  unsigned max_macro_iterations{60};
  unsigned max_cg_iterations{60};
  /** Maximum MO-gradient entry; it bounds the orthonormal-basis commutator. */
  double gradient_tolerance{1e-12};
  double initial_radius{0.5};
  double max_radius{1.0};
  double min_radius{1e-9};
  double preconditioner_floor{0.05};
  /** Below this gradient, a run without halving it in the window has reached
   * the arithmetic floor of the operator and stops without converging. */
  double stagnation_gradient{1e-8};
  unsigned stagnation_window{4};
};

struct Refinement {
  State state;
  bool converged{};
  unsigned macro_iterations{}, fock_builds{}, hessian_actions{};
  double gradient_max{std::numeric_limits<double>::infinity()};
};

struct StabilityControls {
  unsigned max_iterations{80};
  unsigned max_subspace{60};
  unsigned roots{3};
  unsigned unit_guesses{6};
  double residual_tolerance{1e-5};
};

struct LowestMode {
  /** Lowest Ritz value in the physical spin or common RHF rotation coordinates. */
  double ritz_value{std::numeric_limits<double>::infinity()};
  Vector vector;
  /** Residual norm for the internally scaled operator A'. */
  double residual{};
  /** Convergence of the computed Ritz vectors to the residual tolerance. */
  bool ritz_converged{};
  bool complete_rotation_space{};
  unsigned hessian_actions{};
};

namespace detail {

inline Vector product(char ta, char tb, std::size_t m, std::size_t n, std::size_t k,
                      const Vector& a, const Vector& b) {
  Vector c(m * n, 0.0);
  if (m && n && k) tensor::cpu_gemm(ta, tb, m, n, k, a.data(), b.data(), c.data());
  return c;
}

inline double dot(const Vector& a, const Vector& b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

inline double norm(const Vector& a) { return std::sqrt(dot(a, a)); }

inline double max_abs(const Vector& a) {
  double result = 0.0;
  for (const double value : a) {
    if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
    result = std::max(result, std::abs(value));
  }
  return result;
}

inline void axpy(double alpha, const Vector& x, Vector& y) {
  for (std::size_t i = 0; i < y.size(); ++i) y[i] += alpha * x[i];
}

inline void symmetrize(Vector& a, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j)
      a[i * n + j] = a[j * n + i] = 0.5 * (a[i * n + j] + a[j * n + i]);
}

inline Vector columns(const Vector& matrix, std::size_t n, std::size_t first, std::size_t count) {
  Vector result(n * count);
  for (std::size_t row = 0; row < n; ++row)
    std::copy_n(matrix.begin() + row * n + first, count, result.begin() + row * count);
  return result;
}

inline double sinc(double s) {
  const double s2 = s * s;
  return s < 1e-2 ? 1.0 - s2 / 6.0 * (1.0 - s2 / 20.0 * (1.0 - s2 / 42.0)) : std::sin(s) / s;
}

/** (cos s - 1) / s^2 without cancellation near zero. */
inline double cosc(double s) {
  const double s2 = s * s;
  return s < 1e-2 ? -0.5 * (1.0 - s2 / 12.0 * (1.0 - s2 / 30.0 * (1.0 - s2 / 56.0)))
                  : (std::cos(s) - 1.0) / s2;
}

inline std::size_t virtuals(const Problem& p, int spin) { return p.n - p.occupied[spin]; }

inline std::size_t block(const Problem& p, int spin) {
  return virtuals(p, spin) * p.occupied[spin];
}

/** Spin blocks that are computed; a restricted beta block copies alpha. */
inline int computed_spins(const Problem& p) {
  if (p.restricted && p.occupied[0] != p.occupied[1])
    throw std::invalid_argument("restricted orbital optimization requires a closed shell");
  return p.restricted ? 1 : 2;
}

/** Packed alpha block of the alpha = beta projection of x. */
inline Vector restricted_block(const Problem& p, const Vector& x) {
  const auto m = block(p, 0);
  Vector y(m);
  for (std::size_t k = 0; k < m; ++k) y[k] = 0.5 * (x[k] + x[m + k]);
  return y;
}

/** Exact C exp(K) for one spin from the occupied-sized eigensystem of x^T x. */
inline std::pair<Vector, Vector> rotate(const Vector& co, const Vector& cv, std::size_t n,
                                        std::size_t no, std::size_t nv, const double* x) {
  if (!no || !nv) return {co, cv};
  const Vector xs(x, x + nv * no);
  auto xtx = product('T', 'N', no, no, nv, xs, xs);
  symmetrize(xtx, no);
  const auto eigen = tensor::cpu_symmetric_eigen(std::move(xtx), no);
  const auto& v = eigen.vectors;
  Vector v_cos(no * no), v_sinc(no * no), v_cosc(no * no);
  for (std::size_t k = 0; k < no; ++k) {
    const double s = std::sqrt(std::max(0.0, eigen.values[k]));
    const double c = std::cos(s), sc = sinc(s), cc = cosc(s);
    for (std::size_t row = 0; row < no; ++row) {
      v_cos[row * no + k] = v[row * no + k] * c;
      v_sinc[row * no + k] = v[row * no + k] * sc;
      v_cosc[row * no + k] = v[row * no + k] * cc;
    }
  }
  const auto oo = product('N', 'T', no, no, no, v_cos, v);  // cos(sqrt(x^T x))
  const auto a = product('N', 'N', nv, no, no, xs, product('N', 'T', no, no, no, v_sinc, v));
  const auto w = product('N', 'N', nv, no, no, xs, v);            // x V
  const auto w_cosc = product('N', 'N', nv, no, no, xs, v_cosc);  // x V cosc
  auto new_co = product('N', 'N', n, no, no, co, oo);
  tensor::cpu_gemm('N', 'N', n, no, nv, cv.data(), a.data(), new_co.data(), 1.0, 1.0);
  Vector new_cv = cv;
  const auto cvw = product('N', 'N', n, no, nv, cv, w_cosc);
  tensor::cpu_gemm('N', 'T', n, nv, no, cvw.data(), w.data(), new_cv.data(), 1.0, 1.0);
  tensor::cpu_gemm('N', 'T', n, nv, no, co.data(), a.data(), new_cv.data(), -1.0, 1.0);
  return {std::move(new_co), std::move(new_cv)};
}

}  // namespace detail

inline std::size_t rotation_count(const Problem& p) {
  return detail::block(p, 0) + detail::block(p, 1);
}

/** Energy, Fock matrices, MO blocks and gradient at S-orthonormal orbitals. */
inline State evaluate(const Problem& p, SpinMatrices occupied, SpinMatrices virtuals) {
  State st;
  const auto n = p.n;
  const int spins = detail::computed_spins(p);
  if (spins == 1) {
    occupied[1] = occupied[0];
    virtuals[1] = virtuals[0];
  }
  st.occupied_orbitals = std::move(occupied);
  st.virtual_orbitals = std::move(virtuals);
  for (int s = 0; s < spins; ++s) {
    st.density[s] = detail::product('N', 'T', n, n, p.occupied[s], st.occupied_orbitals[s],
                                    st.occupied_orbitals[s]);
    detail::symmetrize(st.density[s], n);
  }
  if (spins == 1) st.density[1] = st.density[0];
  auto g = p.two_electron(st.density[0], st.density[1]);
  for (int s = 0; s < 2; ++s) {
    if (g[s].size() != n * n)
      throw std::runtime_error("UHF two-electron action has the wrong shape");
    st.fock[s] = *p.hcore;
    detail::axpy(1.0, g[spins == 1 ? 0 : s], st.fock[s]);
    detail::symmetrize(st.fock[s], n);
  }
  st.energy = reference::uhf_electronic_energy(st.density[0], st.density[1], *p.hcore, st.fock[0],
                                               st.fock[1]) +
              p.nuclear_repulsion;
  for (int s = 0; s < spins; ++s) {
    const auto no = p.occupied[s], nv = detail::virtuals(p, s);
    const auto& co = st.occupied_orbitals[s];
    const auto& cv = st.virtual_orbitals[s];
    const auto fo = detail::product('N', 'N', n, no, n, st.fock[s], co);
    st.gradient[s] = detail::product('T', 'N', nv, no, n, cv, fo);
    st.fock_oo[s] = detail::product('T', 'N', no, no, n, co, fo);
    detail::symmetrize(st.fock_oo[s], no);
    st.fock_vv[s] = detail::product('T', 'N', nv, nv, n, cv,
                                    detail::product('N', 'N', n, nv, n, st.fock[s], cv));
    detail::symmetrize(st.fock_vv[s], nv);
  }
  if (spins == 1) {
    st.gradient[1] = st.gradient[0];
    st.fock_oo[1] = st.fock_oo[0];
    st.fock_vv[1] = st.fock_vv[0];
  }
  if (!std::isfinite(st.energy) || !std::isfinite(detail::max_abs(st.gradient[0])) ||
      !std::isfinite(detail::max_abs(st.gradient[1])))
    throw std::runtime_error("UHF orbital state is not finite");
  return st;
}

/** Split full n-by-n coefficient matrices (S-orthonormal columns) into blocks. */
inline State from_coefficients(const Problem& p, const SpinMatrices& coefficients) {
  SpinMatrices occupied, virtuals;
  for (int s = 0; s < 2; ++s) {
    if (coefficients[s].size() != p.n * p.n)
      throw std::invalid_argument("UHF orbital coefficients have the wrong shape");
    occupied[s] = detail::columns(coefficients[s], p.n, 0, p.occupied[s]);
    virtuals[s] = detail::columns(coefficients[s], p.n, p.occupied[s], detail::virtuals(p, s));
  }
  return evaluate(p, std::move(occupied), std::move(virtuals));
}

/** Aufbau orbitals of F[D] for spin densities D, then their own state. */
inline State from_density(const Problem& p, const Vector& alpha, const Vector& beta) {
  auto g = p.two_electron(alpha, beta);
  SpinMatrices coefficients;
  for (int s = 0; s < detail::computed_spins(p); ++s) {
    Vector fock = *p.hcore;
    detail::axpy(1.0, g[s], fock);
    detail::symmetrize(fock, p.n);
    coefficients[s] = p.eigen(fock).vectors;
  }
  if (p.restricted) coefficients[1] = coefficients[0];
  return from_coefficients(p, coefficients);
}

inline Vector gradient(const State& st) {
  Vector g = st.gradient[0];
  g.insert(g.end(), st.gradient[1].begin(), st.gradient[1].end());
  return g;
}

inline double gradient_max(const State& st) {
  return std::max(detail::max_abs(st.gradient[0]), detail::max_abs(st.gradient[1]));
}

/** Diagonal orbital-energy-difference part of A'. */
inline Vector hessian_diagonal(const Problem& p, const State& st) {
  Vector d;
  d.reserve(rotation_count(p));
  for (int s = 0; s < 2; ++s) {
    const auto no = p.occupied[s], nv = detail::virtuals(p, s);
    for (std::size_t a = 0; a < nv; ++a)
      for (std::size_t i = 0; i < no; ++i)
        d.push_back(st.fock_vv[s][a * nv + a] - st.fock_oo[s][i * no + i]);
  }
  return d;
}

/** Exact A'x with one two-electron build on the transition densities. */
inline Vector hessian_action(const Problem& p, const State& st, const Vector& x) {
  const auto n = p.n;
  const int spins = detail::computed_spins(p);
  SpinMatrices transition;
  std::array<Vector, 2> blocks;
  std::size_t offset = 0;
  for (int s = 0; s < spins; ++s) {
    const auto no = p.occupied[s], nv = detail::virtuals(p, s);
    if (spins == 1)
      blocks[s] = detail::restricted_block(p, x);
    else
      blocks[s].assign(x.begin() + offset, x.begin() + offset + nv * no);
    offset += nv * no;
    const auto t = detail::product('N', 'N', n, no, nv, st.virtual_orbitals[s], blocks[s]);
    auto d = detail::product('N', 'T', n, n, no, t, st.occupied_orbitals[s]);
    for (std::size_t i = 0; i < n; ++i) {
      d[i * n + i] *= 2.0;
      for (std::size_t j = i + 1; j < n; ++j)
        d[i * n + j] = d[j * n + i] = d[i * n + j] + d[j * n + i];
    }
    transition[s] = std::move(d);
  }
  if (spins == 1) transition[1] = transition[0];
  const auto g = p.two_electron(transition[0], transition[1]);
  Vector y;
  y.reserve(x.size());
  for (int s = 0; s < spins; ++s) {
    const auto no = p.occupied[s], nv = detail::virtuals(p, s);
    auto ys = detail::product('N', 'N', nv, no, nv, st.fock_vv[s], blocks[s]);
    const auto xo = detail::product('N', 'N', nv, no, no, blocks[s], st.fock_oo[s]);
    const auto gc = detail::product('N', 'N', n, no, n, g[s], st.occupied_orbitals[s]);
    const auto coupling = detail::product('T', 'N', nv, no, n, st.virtual_orbitals[s], gc);
    for (std::size_t k = 0; k < ys.size(); ++k) ys[k] += coupling[k] - xo[k];
    y.insert(y.end(), ys.begin(), ys.end());
  }
  if (spins == 1) {
    const auto m = y.size();
    y.resize(2 * m);
    std::copy_n(y.begin(), m, y.begin() + m);
  }
  return y;
}

/** Orbitals rotated by a packed (alpha then beta) step. */
inline std::pair<SpinMatrices, SpinMatrices> rotate(const Problem& p, const State& st,
                                                    const Vector& x) {
  SpinMatrices occupied, virtuals;
  if (detail::computed_spins(p) == 1) {
    const auto y = detail::restricted_block(p, x);
    std::tie(occupied[0], virtuals[0]) =
        detail::rotate(st.occupied_orbitals[0], st.virtual_orbitals[0], p.n, p.occupied[0],
                       detail::virtuals(p, 0), y.data());
    occupied[1] = occupied[0];
    virtuals[1] = virtuals[0];
    return {std::move(occupied), std::move(virtuals)};
  }
  std::size_t offset = 0;
  for (int s = 0; s < 2; ++s) {
    const auto no = p.occupied[s], nv = detail::virtuals(p, s);
    std::tie(occupied[s], virtuals[s]) = detail::rotate(
        st.occupied_orbitals[s], st.virtual_orbitals[s], p.n, no, nv, x.data() + offset);
    offset += nv * no;
  }
  return {std::move(occupied), std::move(virtuals)};
}

struct NewtonStep {
  Vector p;
  double predicted{};
  bool boundary{};
  unsigned hessian_actions{};
};

/** Steihaug-Toint truncated preconditioned CG for min 2g.p + p.A'p, |p| <= radius.
 * Negative curvature is followed to the trust-region boundary. A unit `fixed`
 * direction restricts the step to its orthogonal complement. */
inline NewtonStep truncated_newton_step(const Problem& pb, const State& st, Vector g,
                                        const Vector& preconditioner, double radius,
                                        double relative_tolerance, unsigned max_iterations,
                                        const Vector* fixed = nullptr) {
  const auto m = g.size();
  const auto project = [&](Vector& v) {
    if (fixed) detail::axpy(-detail::dot(*fixed, v), *fixed, v);
  };
  project(g);
  NewtonStep out;
  out.p.assign(m, 0.0);
  Vector ap(m, 0.0), r = g, z(m), d(m);
  for (std::size_t i = 0; i < m; ++i) z[i] = r[i] / preconditioner[i];
  project(z);
  for (std::size_t i = 0; i < m; ++i) d[i] = -z[i];
  double rz = detail::dot(r, z);
  const double stop = relative_tolerance * detail::norm(g);
  for (unsigned it = 0; it < max_iterations && rz > 0.0; ++it) {
    auto ad = hessian_action(pb, st, d);
    project(ad);
    ++out.hessian_actions;
    const double curvature = detail::dot(d, ad);
    const double dd = detail::dot(d, d), pd = detail::dot(out.p, d), pp = detail::dot(out.p, out.p);
    const auto to_boundary = [&] {
      return (-pd + std::sqrt(std::max(0.0, pd * pd + dd * (radius * radius - pp)))) / dd;
    };
    const double alpha = curvature > 0.0 ? rz / curvature : 0.0;
    if (!(curvature > 0.0) || pp + alpha * (2.0 * pd + alpha * dd) >= radius * radius) {
      const double tau = to_boundary();
      detail::axpy(tau, d, out.p);
      detail::axpy(tau, ad, ap);
      out.boundary = true;
      break;
    }
    detail::axpy(alpha, d, out.p);
    detail::axpy(alpha, ad, ap);
    detail::axpy(alpha, ad, r);
    if (detail::norm(r) <= stop) break;
    for (std::size_t i = 0; i < m; ++i) z[i] = r[i] / preconditioner[i];
    project(z);
    const double rz_next = detail::dot(r, z);
    const double beta = rz_next / rz;
    rz = rz_next;
    for (std::size_t i = 0; i < m; ++i) d[i] = beta * d[i] - z[i];
  }
  out.predicted = 2.0 * detail::dot(g, out.p) + detail::dot(out.p, ap);
  return out;
}

/** Trust-region Newton descent to a stationary determinant. */
inline Refinement refine(const Problem& pb, State state, const Controls& c = {}) {
  Refinement out;
  double radius = c.initial_radius;
  double best = std::numeric_limits<double>::infinity();
  unsigned since_best = 0;
  for (;;) {
    const auto g = gradient(state);
    out.gradient_max = detail::max_abs(g);
    if (out.gradient_max <= c.gradient_tolerance) {
      out.converged = true;
      break;
    }
    if (out.gradient_max < 0.5 * best) {
      best = out.gradient_max;
      since_best = 0;
    } else if (++since_best >= c.stagnation_window && out.gradient_max < c.stagnation_gradient) {
      break;
    }
    if (out.macro_iterations >= c.max_macro_iterations || radius < c.min_radius) break;
    ++out.macro_iterations;
    auto preconditioner = hessian_diagonal(pb, state);
    for (auto& value : preconditioner) value = std::max(value, c.preconditioner_floor);
    const double forcing = std::min(0.1, std::sqrt(detail::norm(g)));
    const auto step =
        truncated_newton_step(pb, state, g, preconditioner, radius, forcing, c.max_cg_iterations);
    out.hessian_actions += step.hessian_actions;
    auto [occupied, virtuals] = rotate(pb, state, step.p);
    auto trial = evaluate(pb, std::move(occupied), std::move(virtuals));
    ++out.fock_builds;
    const double actual = trial.energy - state.energy;
    // Energy differences below a few hundred ulps of |E| are rounding, not model error.
    const double resolution =
        512.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(state.energy));
    const bool resolved = std::abs(step.predicted) > resolution;
    double ratio = resolved ? actual / step.predicted : (actual <= resolution ? 1.0 : 0.0);
    const double length = detail::norm(step.p);
    if (resolved && ratio < 0.75) {
      // A step along a soft mode leaves the stiff modes displaced from a curved
      // valley, where the quadratic model fails even for short steps. Relax
      // them orthogonally to the step before judging it.
      Vector direction = step.p;
      for (auto& v : direction) v /= length;
      auto relax_preconditioner = hessian_diagonal(pb, trial);
      for (auto& value : relax_preconditioner) value = std::max(value, c.preconditioner_floor);
      const auto correction =
          truncated_newton_step(pb, trial, gradient(trial), relax_preconditioner, length, forcing,
                                c.max_cg_iterations, &direction);
      out.hessian_actions += correction.hessian_actions;
      auto [relaxed_occupied, relaxed_virtuals] = rotate(pb, trial, correction.p);
      auto relaxed = evaluate(pb, std::move(relaxed_occupied), std::move(relaxed_virtuals));
      ++out.fock_builds;
      const double relaxed_ratio = (relaxed.energy - state.energy) / step.predicted;
      if (relaxed_ratio > ratio) {
        trial = std::move(relaxed);
        ratio = relaxed_ratio;
      }
    }
    const bool accept = resolved ? ratio > 0.1 : ratio > 0.0;
    if (ratio < 0.25)
      radius = 0.25 * length;
    else if (ratio > 0.75 && step.boundary)
      radius = std::min(2.0 * radius, c.max_radius);
    if (accept) state = std::move(trial);
  }
  out.state = std::move(state);
  return out;
}

namespace detail {

/** Orthonormalize t against the basis; append it unless it is dependent. */
inline bool extend_basis(std::vector<Vector>& basis, Vector t) {
  const double initial = norm(t);
  if (!(initial > 0.0)) return false;
  for (int pass = 0; pass < 2; ++pass)
    for (const auto& b : basis) axpy(-dot(b, t), b, t);
  const double length = norm(t);
  if (!(length > 1e-8 * initial)) return false;
  for (auto& v : t) v /= length;
  basis.push_back(std::move(t));
  return true;
}

}  // namespace detail

/** Search for the lowest real internal orbital curvature. A full coordinate
 * basis covers spaces fitting the subspace bound. For larger spaces block
 * Davidson returns Ritz values in the explored space. */
inline LowestMode lowest_hessian_mode(const Problem& pb, const State& st,
                                      const StabilityControls& c = {}) {
  LowestMode out;
  const auto m = rotation_count(pb);
  if (!m) {
    out.ritz_converged = true;
    out.complete_rotation_space = true;
    return out;
  }
  const auto diagonal = hessian_diagonal(pb, st);
  std::vector<Vector> basis, images;
  // Restricted guesses pair each alpha rotation with its beta image, so the
  // subspace never acquires a triplet (RHF -> UHF) component.
  const auto candidates = detail::computed_spins(pb) == 1 ? detail::block(pb, 0) : m;
  out.complete_rotation_space = candidates <= c.max_subspace;
  const auto requested_roots = std::max(1u, c.roots);
  std::vector<std::size_t> order(candidates);
  std::iota(order.begin(), order.end(), std::size_t{0});
  auto units = candidates;
  if (!out.complete_rotation_space) {
    Vector dense(m);
    // Reproducible unequal amplitudes let the same dense starting vector
    // couple to antisymmetric modes of degenerate orbital pairs.
    std::minstd_rand weights(1);
    constexpr auto weight_range =
        static_cast<double>(std::minstd_rand::max() - std::minstd_rand::min());
    for (std::size_t i = 0; i < candidates; ++i) {
      const double weight = 0.5 + (weights() - weights.min()) / weight_range;
      dense[i] = weight / std::max(diagonal[i], 0.05);
      if (candidates != m) dense[candidates + i] = dense[i];
    }
    detail::extend_basis(basis, std::move(dense));
    // Evaluate the residual of every initial Ritz vector. Individually
    // converged unit directions can otherwise mask the mixed starting vector
    // before its nonzero residual has enlarged the search space.
    units = std::min({std::size_t(c.unit_guesses), std::size_t(requested_roots - 1), candidates});
    std::partial_sort(order.begin(), order.begin() + units, order.end(),
                      [&](std::size_t a, std::size_t b) { return diagonal[a] < diagonal[b]; });
  }
  for (std::size_t k = 0; k < units; ++k) {
    Vector e(m, 0.0);
    e[order[k]] = 1.0;
    if (candidates != m) e[candidates + order[k]] = 1.0;
    detail::extend_basis(basis, std::move(e));
  }
  for (unsigned it = 0; it < c.max_iterations; ++it) {
    while (images.size() < basis.size()) {
      images.push_back(hessian_action(pb, st, basis[images.size()]));
      ++out.hessian_actions;
    }
    const auto k = basis.size();
    Vector subspace(k * k);
    for (std::size_t i = 0; i < k; ++i)
      for (std::size_t j = 0; j < k; ++j) subspace[i * k + j] = detail::dot(basis[i], images[j]);
    detail::symmetrize(subspace, k);
    const auto eigen = tensor::cpu_symmetric_eigen(std::move(subspace), k);
    std::vector<std::size_t> ritz(k);
    std::iota(ritz.begin(), ritz.end(), std::size_t{0});
    std::sort(ritz.begin(), ritz.end(),
              [&](std::size_t a, std::size_t b) { return eigen.values[a] < eigen.values[b]; });
    const auto roots = std::min<std::size_t>(requested_roots, k);
    std::vector<Vector> vectors(roots, Vector(m, 0.0)), residuals(roots, Vector(m, 0.0));
    std::vector<double> norms(roots);
    for (std::size_t r = 0; r < roots; ++r) {
      const double theta = eigen.values[ritz[r]];
      for (std::size_t j = 0; j < k; ++j) {
        const double weight = eigen.vectors[j * k + ritz[r]];
        detail::axpy(weight, basis[j], vectors[r]);
        detail::axpy(weight, images[j], residuals[r]);
      }
      detail::axpy(-theta, vectors[r], residuals[r]);
      norms[r] = detail::norm(residuals[r]);
    }
    // A common spatial rotation moves both spin determinants through the same
    // angle. The resulting RHF curvature is twice the normalized spin-pair
    // curvature that is returned for UHF coordinates.
    const double physical_scale = pb.restricted ? 4.0 : 2.0;
    out.ritz_value = physical_scale * eigen.values[ritz[0]];
    out.vector = vectors[0];
    out.residual = *std::max_element(norms.begin(), norms.end());
    if (out.residual <= c.residual_tolerance) {
      out.ritz_converged = true;
      break;
    }
    if (out.complete_rotation_space) break;
    if (k + roots > c.max_subspace) {
      std::vector<Vector> collapsed, collapsed_images;
      for (std::size_t r = 0; r < roots; ++r) {
        Vector image(m, 0.0);
        for (std::size_t j = 0; j < k; ++j)
          detail::axpy(eigen.vectors[j * k + ritz[r]], images[j], image);
        collapsed.push_back(vectors[r]);
        collapsed_images.push_back(std::move(image));
      }
      basis = std::move(collapsed);
      images = std::move(collapsed_images);
    }
    bool extended = false;
    for (std::size_t r = 0; r < roots; ++r) {
      if (norms[r] <= c.residual_tolerance) continue;
      const double theta = eigen.values[ritz[r]];
      Vector t(m);
      for (std::size_t i = 0; i < m; ++i) {
        double denominator = theta - diagonal[i];
        if (std::abs(denominator) < 1e-4) denominator = denominator < 0.0 ? -1e-4 : 1e-4;
        t[i] = residuals[r][i] / denominator;
      }
      extended = detail::extend_basis(basis, std::move(t)) || extended;
    }
    if (!extended) break;
  }
  return out;
}

/** Lowest-energy determinant displaced along a Hessian mode, if any lies below
 * the reference state. Each probe costs one Fock build. */
inline std::optional<State> descend_along(const Problem& pb, const State& st, const Vector& mode,
                                          unsigned& fock_builds) {
  const double length = detail::norm(mode);
  if (!(length > 0.0)) return std::nullopt;
  const double resolution =
      512.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(st.energy));
  std::optional<State> best;
  double best_energy = st.energy - resolution;
  for (const double angle : {0.1, 0.3, 0.6, -0.1, -0.3, -0.6}) {
    Vector x = mode;
    for (auto& v : x) v *= angle / length;
    auto [occupied, virtuals] = rotate(pb, st, x);
    auto trial = evaluate(pb, std::move(occupied), std::move(virtuals));
    ++fock_builds;
    if (trial.energy < best_energy) {
      best_energy = trial.energy;
      best = std::move(trial);
    }
  }
  return best;
}

}  // namespace generativeqc::scf::solver::uhf_second_order
