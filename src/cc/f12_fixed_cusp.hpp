#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "cc/solver.hpp"

namespace generativeqc::cc::f12 {

inline std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("RCCSD-F12b size overflow");
  return a * b;
}

inline std::size_t checked_add(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::length_error("RCCSD-F12b size overflow");
  return a + b;
}

inline std::size_t host_bytes(const FixedCuspF12& f) {
  const auto elements = checked_add(checked_add(f.w_vo.capacity(), f.w_vv.capacity()),
                                    f.c_f12.capacity());
  return checked_mul(elements, sizeof(double));
}

inline void validate(std::size_t o, std::size_t v, const FixedCuspF12& f) {
  if (!f.active()) return;
  if (!o || !v) throw std::invalid_argument("RCCSD-F12b requires nonempty occupied and virtual spaces");
  if (!std::isfinite(f.fixed_geminal_energy))
    throw std::invalid_argument("nonfinite RCCSD-F12b fixed_geminal_energy");
  const auto oo = checked_mul(o, o);
  const auto vv = checked_mul(v, v);
  const auto n_vo = checked_mul(checked_mul(oo, v), o);
  const auto n_vv = checked_mul(oo, vv);
  auto check_vec = [](const std::vector<double>& x, std::size_t expected, const char* name) {
    if (x.size() != expected ||
        !std::all_of(x.begin(), x.end(), [](double y) { return std::isfinite(y); }))
      throw std::invalid_argument(std::string("invalid RCCSD-F12b input ") + name);
  };
  check_vec(f.w_vo, n_vo, "w_vo");
  check_vec(f.w_vv, n_vv, "w_vv");
  check_vec(f.c_f12, n_vv, "c_f12");
  auto check_pair_sym = [&](const std::vector<double>& x, const char* name) {
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b) {
            const auto flat = ((i * o + j) * v + a) * v + b;
            const auto mate = ((j * o + i) * v + b) * v + a;
            if (flat >= mate) continue;
            const double first = x[flat], second = x[mate];
            if (std::abs(first - second) >
                1e-10 * (1.0 + std::max(std::abs(first), std::abs(second))))
              throw std::invalid_argument(std::string("RCCSD-F12b ") + name +
                                          " violates simultaneous pair symmetry");
          }
  };
  check_pair_sym(f.w_vv, "w_vv");
  check_pair_sym(f.c_f12, "c_f12");
}

inline std::vector<double> symmetrized_pair_copy(std::size_t o, std::size_t v,
                                                 std::span<const double> src) {
  const auto n2 = checked_mul(checked_mul(o, o), checked_mul(v, v));
  if (src.size() != n2) throw std::invalid_argument("RCCSD-F12b pair tensor shape mismatch");
  std::vector<double> dst(n2);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const auto flat = ((i * o + j) * v + a) * v + b;
          const auto mate = ((j * o + i) * v + b) * v + a;
          if (flat > mate) continue;
          const double mean = flat == mate ? src[flat] : 0.5 * (src[flat] + src[mate]);
          dst[flat] = dst[mate] = mean;
        }
  return dst;
}

// Spatial closed-shell singles fixed-cusp constant:
//   Delta R1[i, a] = sum_j (2 * W_vo[i, j, a, j] - W_vo[j, i, a, j])
inline std::vector<double> singles_constant(std::size_t o, std::size_t v,
                                            std::span<const double> w_vo) {
  std::vector<double> r1(checked_mul(o, v), 0.0);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a) {
      double acc = 0.0;
      for (std::size_t j = 0; j < o; ++j) {
        const auto direct = ((i * o + j) * v + a) * o + j;
        const auto exch = ((j * o + i) * v + a) * o + j;
        acc += 2.0 * w_vo[direct] - w_vo[exch];
      }
      r1[i * v + a] = acc;
    }
  return r1;
}

inline void add_residuals_cpu(std::size_t o, std::size_t v, const FixedCuspF12& f,
                              std::span<const double> r1_const, const double* t1, double* r1,
                              double* r2) {
  const auto n1 = o * v;
  for (std::size_t idx = 0; idx < n1; ++idx) r1[idx] += r1_const[idx];
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a) {
        const double* w1 = f.w_vo.data() + ((i * o + j) * v + a) * o;
        for (std::size_t b = 0; b < v; ++b) {
          const auto flat = ((i * o + j) * v + a) * v + b;
          const double* w2 = f.w_vo.data() + ((j * o + i) * v + b) * o;
          double p = 0.0;
          for (std::size_t k = 0; k < o; ++k) p += w1[k] * t1[k * v + b] + w2[k] * t1[k * v + a];
          r2[flat] += (f.w_vv[flat] + f.c_f12[flat]) - p;
        }
      }
}

inline double energy_cpu(std::size_t o, std::size_t v, const FixedCuspF12& f,
                         std::span<const double> r1_const, const double* t1, const double* t2) {
  double e = f.fixed_geminal_energy;
  const auto n1 = o * v;
  for (std::size_t idx = 0; idx < n1; ++idx) e += 2.0 * r1_const[idx] * t1[idx];
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a) {
        const double t1_ia = t1[i * v + a];
        for (std::size_t b = 0; b < v; ++b) {
          const auto idx = ((i * o + j) * v + a) * v + b;
          const auto swap = ((i * o + j) * v + b) * v + a;
          const double tc = 2.0 * f.c_f12[idx] - f.c_f12[swap];
          const double tw = 2.0 * f.w_vv[idx] - f.w_vv[swap];
          const double t2_val = t2[idx];
          e += tc * t2_val + tw * (t2_val + t1_ia * t1[j * v + b]);
        }
      }
  return e;
}

// Positive derivatives d(Delta E_F12b) / d(t1, t2) for the spatial Lambda RHS.
inline void energy_gradient_cpu(std::size_t o, std::size_t v, const FixedCuspF12& f,
                                std::span<const double> r1_const, std::span<const double> t1,
                                std::vector<double>& g1, std::vector<double>& g2) {
  const auto n1 = checked_mul(o, v);
  const auto n2 = checked_mul(checked_mul(o, o), checked_mul(v, v));
  g1.assign(n1, 0.0);
  g2.assign(n2, 0.0);
  for (std::size_t k = 0; k < o; ++k)
    for (std::size_t c = 0; c < v; ++c) {
      double acc = r1_const[k * v + c];
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t b = 0; b < v; ++b) {
          const auto idx = ((k * o + j) * v + c) * v + b;
          const auto swap = ((k * o + j) * v + b) * v + c;
          acc += (2.0 * f.w_vv[idx] - f.w_vv[swap]) * t1[j * v + b];
        }
      g1[k * v + c] = 2.0 * acc;
    }
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const auto idx = ((i * o + j) * v + a) * v + b;
          const auto swap = ((i * o + j) * v + b) * v + a;
          g2[idx] = (2.0 * f.c_f12[idx] - f.c_f12[swap]) + (2.0 * f.w_vv[idx] - f.w_vv[swap]);
        }
}

// Spatial Lambda Jacobian-transpose F12b contribution:
//   out_one[k, c] -= 2 * sum_{i,j,a} W_vo[i, j, a, k] * lambda2[i, j, a, c]
inline void add_lambda_transpose_cpu(std::size_t o, std::size_t v, std::span<const double> w_vo,
                                     std::span<const double> lambda2, double* out_one) {
  const auto m = o * o * v;
  for (std::size_t k = 0; k < o; ++k)
    for (std::size_t c = 0; c < v; ++c) {
      double acc = 0.0;
      for (std::size_t p = 0; p < m; ++p) acc += w_vo[p * o + k] * lambda2[p * v + c];
      out_one[k * v + c] -= 2.0 * acc;
    }
}

}  // namespace generativeqc::cc::f12
