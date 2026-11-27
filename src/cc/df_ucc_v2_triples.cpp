#include "cc/df_ucc_v2_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "cc/df_ucc_v2_triples_host_capacity.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::cc::v2_detail {
namespace {

tensor::CpuLinalgPlan blas_plan() {
  return {tensor::CpuLinalgProvider::automatic, tensor::CpuLinalgThreadOwnership::provider_parallel,
          1};
}

void dgemm_counted(char ta, char tb, std::size_t m, std::size_t n, std::size_t k, const double* a,
                   const double* b, double* c, std::size_t& fp64_gemms, double alpha = 1.0,
                   double beta = 0.0) {
  if (!m || !n || !k) {
    if (beta == 0.0 && m && n) std::fill_n(c, m * n, 0.0);
    return;
  }
  tensor::cpu_gemm(ta, tb, m, n, k, a, b, c, alpha, beta, blas_plan());
  ++fp64_gemms;
}

// Apply one W-contraction policy consistently across all four spin channels.
// The compensated schedule uses a nine-SGEMM expansion for both DF panels and
// W products. The same expansion computes the two input cotangents for
// every occupied W product, virtual W product and DF panel. Resolvent and
// disconnected V contractions keep their independent FP64 policy.
struct WContractions {
  bool emulated_forward, compensated;
  std::size_t& fp32_gemms;
  std::size_t& fp32_reverse_gemms;
  std::size_t& fp64_gemms;

  void forward(char ta, char tb, std::size_t m, std::size_t n, std::size_t k, const double* a,
               const double* b, double* c, double alpha, double beta) const {
    if (compensated) {
      fp32_gemms += tensor::cpu_split3_gemm(ta, tb, m, n, k, a, b, c, alpha, beta, blas_plan());
      return;
    }
    if (!emulated_forward) {
      dgemm_counted(ta, tb, m, n, k, a, b, c, fp64_gemms, alpha, beta);
      return;
    }
    if (!m || !n || !k) {
      if (beta == 0.0 && m && n) std::fill_n(c, m * n, 0.0);
      return;
    }
    // The original CPU mode rounds FP64 DGEMM inputs and outputs to FP32.
    const std::size_t na = m * k, nb = k * n;
    std::vector<double> af(na), bf(nb), cf(m * n, 0.0);
    for (std::size_t idx = 0; idx < na; ++idx)
      af[idx] = static_cast<double>(static_cast<float>(a[idx]));
    for (std::size_t idx = 0; idx < nb; ++idx)
      bf[idx] = static_cast<double>(static_cast<float>(b[idx]));
    tensor::cpu_gemm(ta, tb, m, n, k, af.data(), bf.data(), cf.data(), alpha, 0.0, blas_plan());
    for (std::size_t idx = 0; idx < m * n; ++idx) {
      const double val = static_cast<double>(static_cast<float>(cf[idx]));
      c[idx] = (beta == 0.0) ? val : (beta * c[idx] + val);
    }
    ++fp32_gemms;
  }

  void pullback(char ta, char tb, std::size_t m, std::size_t n, std::size_t k,
                        const double* a, const double* b, double* c, double alpha,
                        double beta) const {
    if (compensated) {
      const auto count =
          tensor::cpu_split3_gemm(ta, tb, m, n, k, a, b, c, alpha, beta, blas_plan());
      fp32_gemms += count;
      fp32_reverse_gemms += count;
    } else {
      dgemm_counted(ta, tb, m, n, k, a, b, c, fp64_gemms, alpha, beta);
    }
  }
};

// Build single-occupied slab V_p[c, b, e] = sum_Q bov[Q, p, c] * bvv[Q, b, e]
void build_ovvv_slab(std::size_t q, std::size_t o_p, std::size_t v_c, std::size_t v_be,
                     std::size_t p, const double* bov, const double* bvv, double* slab,
                     const WContractions& contractions) {
  std::vector<double> bov_p(q * v_c, 0.0);
  for (std::size_t Q = 0; Q < q; ++Q)
    std::copy_n(bov + (Q * o_p + p) * v_c, v_c, bov_p.data() + Q * v_c);
  contractions.forward('T', 'N', v_c, v_be * v_be, q, bov_p.data(), bvv, slab, 1.0, 0.0);
}

void pullback_ovvv_slab(std::size_t q, std::size_t o_p, std::size_t v_c, std::size_t v_be,
                        std::size_t p, const double* bov, const double* bvv, const double* bar_slab,
                        double* bar_bov, double* bar_bvv, const WContractions& contractions) {
  std::vector<double> bov_p(q * v_c, 0.0);
  for (std::size_t Q = 0; Q < q; ++Q)
    std::copy_n(bov + (Q * o_p + p) * v_c, v_c, bov_p.data() + Q * v_c);
  std::vector<double> bar_bov_p(q * v_c, 0.0);
  contractions.pullback('N', 'T', q, v_c, v_be * v_be, bvv, bar_slab, bar_bov_p.data(), 1.0, 0.0);
  for (std::size_t Q = 0; Q < q; ++Q)
    for (std::size_t c = 0; c < v_c; ++c) bar_bov[(Q * o_p + p) * v_c + c] += bar_bov_p[Q * v_c + c];
  contractions.pullback('N', 'N', q, v_be * v_be, v_c, bov_p.data(), bar_slab, bar_bvv, 1.0, 1.0);
}

// Same-spin channel helper (used identically for aaa and bbb)
void run_same_spin_triples(std::size_t o, std::size_t v, std::size_t q, const double* bov,
                           const double* bvv, const double* goovv, const double* gooov,
                           const double* fov, const double* t1, const double* t2,
                           const double* eps_o, const double* eps_v,
                           const std::vector<double>& weights, bool has_weights,
                           bool compute_forces, const WContractions& contractions, double den_thresh, double& total_e,
                           std::vector<double>& occ_e, double* bar_t1, double* bar_t2,
                           double* bar_bov, double* bar_bvv, double* bar_goovv, double* bar_gooov,
                           double* bar_fov, double* bar_foo, double* bar_fvv) {
  if (o < 3 || v < 3) return;
  const std::size_t v2 = v * v, v3 = v * v2;

  // Build V_p[c, b, e] = (c p | b e) for each occupied p on demand (in bounded slabs)
  std::vector<double> v_slabs(o * v3, 0.0);
  for (std::size_t p = 0; p < o; ++p)
    build_ovvv_slab(q, o, v, v, p, bov, bvv, v_slabs.data() + p * v3, contractions);
  std::vector<double> bar_v_slabs(compute_forces ? o * v3 : 0, 0.0);

  // Helper: construct antisymmetric W[a,b,c] and V[a,b,c] for ordered (i, j, k)
  std::vector<double> s_acb(v3), s_occ(v3), w_abc(v3), v_abc(v3);
  auto build_wv = [&](std::size_t i, std::size_t j, std::size_t k, std::vector<double>& W,
                      std::vector<double>& V) {
    std::fill(s_acb.begin(), s_acb.end(), 0.0);
    std::fill(s_occ.begin(), s_occ.end(), 0.0);
    const std::size_t cyc[3][3] = {{i, j, k}, {j, k, i}, {k, i, j}};
    for (int m = 0; m < 3; ++m) {
      const std::size_t p = cyc[m][0], r = cyc[m][1], s = cyc[m][2];
      // s_acb[a, c, b] += sum_e t2[r, s, a, e] * V_p[c, b, e]
      contractions.forward('N', 'T', v, v2, v, t2 + (r * o + s) * v2, v_slabs.data() + p * v3,
                           s_acb.data(), 1.0, 1.0);
      // s_occ[a, b, c] += sum_n gooov[r, s, n, a] * t2[p, n, b, c]
      contractions.forward('T', 'N', v, v2, o, gooov + (r * o + s) * o * v, t2 + p * o * v2,
                           s_occ.data(), 1.0, 1.0);
    }
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t b = 0; b < v; ++b)
        for (std::size_t c = 0; c < v; ++c) {
          const double sv =
              s_acb[(a * v + c) * v + b] - s_acb[(a * v + b) * v + c] -
              s_acb[(b * v + c) * v + a] + s_acb[(b * v + a) * v + c] +
              s_acb[(c * v + b) * v + a] - s_acb[(c * v + a) * v + b];
          const double so =
              s_occ[(a * v + b) * v + c] - s_occ[(b * v + a) * v + c] -
              s_occ[(c * v + b) * v + a];
          W[(a * v + b) * v + c] = sv - so;
        }
    std::fill(V.begin(), V.end(), 0.0);
    for (int m = 0; m < 3; ++m) {
      const std::size_t p = cyc[m][0], r = cyc[m][1], s = cyc[m][2];
      const double* t1p = t1 + p * v;
      const double* fovp = fov + p * v;
      const double* g_rs = goovv + (r * o + s) * v2;
      const double* t2_rs = t2 + (r * o + s) * v2;
      for (std::size_t a = 0; a < v; ++a) {
        const double t1pa = t1p[a], fpa = fovp[a];
        for (std::size_t b = 0; b < v; ++b)
          for (std::size_t c = 0; c < v; ++c) {
            const double z = t1pa * g_rs[b * v + c] + fpa * t2_rs[b * v + c];
            V[(a * v + b) * v + c] += z;
            V[(b * v + a) * v + c] -= z;
            V[(c * v + b) * v + a] -= z;
          }
      }
    }
  };

  std::vector<double> X(v3), Y(v3), bar_W(v3), bar_V(v3), bar_s_acb(v3), xy_vv(v2);

  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = i + 1; j < o; ++j)
      for (std::size_t k = j + 1; k < o; ++k) {
        build_wv(i, j, k, w_abc, v_abc);
        const double eijk_o = eps_o[i] + eps_o[j] + eps_o[k];
        const double wt = has_weights ? (weights[i] + weights[j] + weights[k]) / 3.0 : 1.0;
        const double eff = (1.0 / 6.0) * wt;
        double e_unw = 0.0;
        double diag_foo = 0.0;
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b)
            for (std::size_t c = 0; c < v; ++c) {
              const std::size_t idx = (a * v + b) * v + c;
              const double den = eijk_o - eps_v[a] - eps_v[b] - eps_v[c];
              if (std::abs(den) < den_thresh)
                throw std::runtime_error("DF-UCCSD(T) v2 small same-spin triples denominator");
              const double w = w_abc[idx], vv = v_abc[idx];
              const double x = w / den;
              const double y = (w + vv) / den;
              e_unw += w * y;
              if (compute_forces) {
                X[idx] = x;
                Y[idx] = y;
                bar_W[idx] = eff * (x + y);
                bar_V[idx] = eff * x;
                diag_foo -= eff * x * y;
              }
            }
        e_unw /= 6.0;
        total_e += wt * e_unw;
        occ_e[i] += e_unw / 3.0;
        occ_e[j] += e_unw / 3.0;
        occ_e[k] += e_unw / 3.0;

        if (!compute_forces) continue;

        bar_foo[i * o + i] += diag_foo;
        bar_foo[j * o + j] += diag_foo;
        bar_foo[k * o + k] += diag_foo;

        // Virtual Fock resolvent bar_fvv[a, d] += 0.5 * wt * 0.5 * (X[a,bc] Y[d,bc] + Y[a,bc] X[d,bc])
        dgemm_counted('N', 'T', v, v, v2, X.data(), Y.data(), xy_vv.data(), contractions.fp64_gemms,
                      1.0, 0.0);
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t d = 0; d < v; ++d)
            bar_fvv[a * v + d] += 0.25 * wt * (xy_vv[a * v + d] + xy_vv[d * v + a]);

        // bar_s_acb[a, c, b] = 6.0 * bar_W[a, b, c]
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b)
            for (std::size_t c = 0; c < v; ++c)
              bar_s_acb[(a * v + c) * v + b] = 6.0 * bar_W[(a * v + b) * v + c];

        const std::size_t cyc[3][3] = {{i, j, k}, {j, k, i}, {k, i, j}};
        for (int m = 0; m < 3; ++m) {
          const std::size_t p = cyc[m][0], r = cyc[m][1], s = cyc[m][2];
          // Virtual pullback
          contractions.pullback('N', 'N', v, v, v2, bar_s_acb.data(), v_slabs.data() + p * v3,
                                bar_t2 + (r * o + s) * v2, 1.0, 1.0);
          contractions.pullback('T', 'N', v2, v, v, bar_s_acb.data(), t2 + (r * o + s) * v2,
                                bar_v_slabs.data() + p * v3, 1.0, 1.0);
          // Occupied pullback (factor -3.0 on bar_W since so = P(a/bc) s_occ and bar_W is antisym)
          contractions.pullback('N', 'T', o, v, v2, t2 + p * o * v2, bar_W.data(),
                                bar_gooov + (r * o + s) * o * v, -3.0, 1.0);
          contractions.pullback('N', 'N', o, v2, v, gooov + (r * o + s) * o * v, bar_W.data(),
                                bar_t2 + p * o * v2, -3.0, 1.0);
          // Disconnected V pullback (factor 3.0 on bar_V)
          const double* t1p = t1 + p * v;
          const double* fovp = fov + p * v;
          const double* g_rs = goovv + (r * o + s) * v2;
          const double* t2_rs = t2 + (r * o + s) * v2;
          double* bg_rs = bar_goovv + (r * o + s) * v2;
          double* bt2_rs = bar_t2 + (r * o + s) * v2;
          for (std::size_t a = 0; a < v; ++a) {
            double st1 = 0.0, sfov = 0.0;
            const double t1pa = 3.0 * t1p[a], fpa = 3.0 * fovp[a];
            const double* bv = bar_V.data() + a * v2;
            for (std::size_t bc = 0; bc < v2; ++bc) {
              const double bv_val = bv[bc];
              st1 += bv_val * g_rs[bc];
              sfov += bv_val * t2_rs[bc];
              bg_rs[bc] += t1pa * bv_val;
              bt2_rs[bc] += fpa * bv_val;
            }
            bar_t1[p * v + a] += 3.0 * st1;
            bar_fov[p * v + a] += 3.0 * sfov;
          }
        }
      }

  if (!compute_forces) return;

  // Pull bar_v_slabs back into bar_bov and bar_bvv
  for (std::size_t p = 0; p < o; ++p)
    pullback_ovvv_slab(q, o, v, v, p, bov, bvv, bar_v_slabs.data() + p * v3, bar_bov, bar_bvv,
                       contractions);

  // Same-space occupied Fock resolvent off-diagonals bar_foo[p, r] (p != r):
  // Full for unweighted (T), degenerate-only (|eps_p - eps_r| < 1e-6) for weighted (T+).
  bool need_offdiag = !has_weights;
  if (has_weights) {
    for (std::size_t p = 0; p < o && !need_offdiag; ++p)
      for (std::size_t r = p + 1; r < o; ++r)
        if (std::abs(eps_o[p] - eps_o[r]) < 1e-6) need_offdiag = true;
  }
  if (need_offdiag) {
    std::vector<double> xp(o * v3), yp(o * v3);
    std::vector<bool> active(o, false);
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t k = j + 1; k < o; ++k) {
        std::fill(active.begin(), active.end(), false);
        for (std::size_t p = 0; p < o; ++p) {
          if (p == j || p == k) continue;
          if (!has_weights) {
            active[p] = true;
          } else {
            for (std::size_t r = 0; r < o; ++r)
              if (r != p && r != j && r != k && std::abs(eps_o[p] - eps_o[r]) < 1e-6) {
                active[p] = true;
                break;
              }
          }
        }
        for (std::size_t p = 0; p < o; ++p) {
          if (!active[p]) continue;
          build_wv(p, j, k, w_abc, v_abc);
          const double epjk = eps_o[p] + eps_o[j] + eps_o[k];
          double* xptr = xp.data() + p * v3;
          double* yptr = yp.data() + p * v3;
          for (std::size_t a = 0; a < v; ++a)
            for (std::size_t b = 0; b < v; ++b)
              for (std::size_t c = 0; c < v; ++c) {
                const std::size_t idx = (a * v + b) * v + c;
                const double den = epjk - eps_v[a] - eps_v[b] - eps_v[c];
                xptr[idx] = w_abc[idx] / den;
                yptr[idx] = (w_abc[idx] + v_abc[idx]) / den;
              }
        }
        for (std::size_t p = 0; p < o; ++p) {
          if (!active[p]) continue;
          for (std::size_t r = p + 1; r < o; ++r) {
            if (!active[r]) continue;
            if (has_weights && std::abs(eps_o[p] - eps_o[r]) >= 1e-6) continue;
            const double wt = has_weights ? (weights[p] + weights[j] + weights[k]) / 3.0 : 1.0;
            const double* x_p = xp.data() + p * v3;
            const double* y_p = yp.data() + p * v3;
            const double* x_r = xp.data() + r * v3;
            const double* y_r = yp.data() + r * v3;
            double dot = 0.0;
            for (std::size_t idx = 0; idx < v3; ++idx) dot += x_p[idx] * y_r[idx] + y_p[idx] * x_r[idx];
            const double val = -(1.0 / 12.0) * wt * dot;
            bar_foo[p * o + r] += val;
            bar_foo[r * o + p] += val;
          }
        }
      }
  }
}

// Mixed-spin channel helper: 2 majority-spin (s1: i<j, a,b) + 1 minority-spin (s2: k, c).
// Used for BOTH aab (s1=alpha, s2=beta) and abb (s1=beta, s2=alpha).
void run_mixed_spin_triples(
    std::size_t o1, std::size_t o2, std::size_t v1, std::size_t v2, std::size_t q,
    const double* bov1, const double* bvv1, const double* bov2, const double* bvv2,
    const double* goovv11, const double* goovv12, const double* gooov11, const double* gooov12,
    const double* gooov21, const double* fov1, const double* fov2, const double* t1_1,
    const double* t1_2, const double* t2_11, const double* t2_12, const double* eps_o1,
    const double* eps_v1, const double* eps_o2, const double* eps_v2,
    const std::vector<double>& w1, const std::vector<double>& w2, bool has_weights,
    bool compute_forces, const WContractions& contractions, double den_thresh, double& total_e,
    std::vector<double>& occ1, std::vector<double>& occ2, double* bar_t1_1, double* bar_t1_2,
    double* bar_t2_11, double* bar_t2_12, double* bar_bov1, double* bar_bvv1, double* bar_bov2,
    double* bar_bvv2, double* bar_goovv11, double* bar_goovv12, double* bar_gooov11,
    double* bar_gooov12, double* bar_gooov21, double* bar_fov1, double* bar_fov2, double* bar_foo1,
    double* bar_foo2, double* bar_fvv1, double* bar_fvv2) {
  if (o1 < 2 || o2 < 1 || v1 < 2 || v2 < 1) return;
  const std::size_t v11 = v1 * v1, v12 = v1 * v2, v22 = v2 * v2, v_tot = v1 * v1 * v2;

  // Precompute 3-index slabs for mixed channel:
  // v11_slab[p, a, b, e] = (a p | b e)_11 = sum_Q bov1[Q, p, a] * bvv1[Q, b, e]  ([o1, v1, v1, v1])
  // v12_slab[p, b, c, e] = (b p | c e)_12 = sum_Q bov1[Q, p, b] * bvv2[Q, c, e]  ([o1, v1, v2, v2])
  // v21_slab[k, c, b, e] = (c k | b e)_21 = sum_Q bov2[Q, k, c] * bvv1[Q, b, e]  ([o2, v2, v1, v1])
  std::vector<double> v11_slabs(o1 * v1 * v11, 0.0);
  std::vector<double> v12_slabs(o1 * v1 * v22, 0.0);
  std::vector<double> v21_slabs(o2 * v2 * v11, 0.0);
  for (std::size_t p = 0; p < o1; ++p) {
    build_ovvv_slab(q, o1, v1, v1, p, bov1, bvv1, v11_slabs.data() + p * v1 * v11, contractions);
    build_ovvv_slab(q, o1, v1, v2, p, bov1, bvv2, v12_slabs.data() + p * v1 * v22, contractions);
  }
  for (std::size_t k = 0; k < o2; ++k) {
    build_ovvv_slab(q, o2, v2, v1, k, bov2, bvv1, v21_slabs.data() + k * v2 * v11, contractions);
  }
  std::vector<double> bar_v11(compute_forces ? v11_slabs.size() : 0, 0.0);
  std::vector<double> bar_v12(compute_forces ? v12_slabs.size() : 0, 0.0);
  std::vector<double> bar_v21(compute_forces ? v21_slabs.size() : 0, 0.0);

  // Transposed views of gooov21[m, j, k, a] -> g21_jk[j, k, a, m] and t2_12[m, k, b, c] -> t2_k_m[k, m, b, c]
  // and gooov12[m, k, j, c] -> g12_kj[k, j, m, c]
  std::vector<double> g21_jk(o1 * o2 * v1 * o2, 0.0);
  for (std::size_t m = 0; m < o2; ++m)
    for (std::size_t j = 0; j < o1; ++j)
      for (std::size_t k = 0; k < o2; ++k)
        for (std::size_t a = 0; a < v1; ++a)
          g21_jk[((j * o2 + k) * v1 + a) * o2 + m] =
              gooov21[((m * o1 + j) * o2 + k) * v1 + a];

  std::vector<double> g12_kj(o2 * o1 * o1 * v2, 0.0);
  for (std::size_t m = 0; m < o1; ++m)
    for (std::size_t k = 0; k < o2; ++k)
      for (std::size_t j = 0; j < o1; ++j)
        for (std::size_t c = 0; c < v2; ++c)
          g12_kj[((k * o1 + j) * o1 + m) * v2 + c] =
              gooov12[((m * o2 + k) * o1 + j) * v2 + c];

  std::vector<double> t2_12_k_m(o2 * o1 * v12, 0.0);
  for (std::size_t m = 0; m < o1; ++m)
    for (std::size_t k = 0; k < o2; ++k)
      std::copy_n(t2_12 + (m * o2 + k) * v12, v12, t2_12_k_m.data() + (k * o1 + m) * v12);

  std::vector<double> bar_g21_jk(compute_forces ? g21_jk.size() : 0, 0.0);
  std::vector<double> bar_g12_kj(compute_forces ? g12_kj.size() : 0, 0.0);
  std::vector<double> bar_t2_12_k_m(compute_forces ? t2_12_k_m.size() : 0, 0.0);

  std::vector<double> R(v_tot), tmp_acb(v_tot), w_abc(v_tot), v_abc(v_tot);
  auto build_wv = [&](std::size_t i, std::size_t j, std::size_t k, std::vector<double>& W,
                      std::vector<double>& V) {
    std::fill(R.begin(), R.end(), 0.0);
    // Term 1: + sum_e t2_11[i, j, a, e] * v21_slab[k, c, b, e] -> tmp_acb[a, c, b]
    contractions.forward('N', 'T', v1, v2 * v1, v1, t2_11 + (i * o1 + j) * v11,
                         v21_slabs.data() + k * v2 * v11, tmp_acb.data(), 1.0, 0.0);
    for (std::size_t a = 0; a < v1; ++a)
      for (std::size_t b = 0; b < v1; ++b)
        for (std::size_t c = 0; c < v2; ++c) R[(a * v1 + b) * v2 + c] += tmp_acb[(a * v2 + c) * v1 + b];

    // Term 2: - sum_e t2_12[j, k, a, e] * v12_slab[i, b, c, e] + (i<->j)
    contractions.forward('N', 'T', v1, v12, v2, t2_12 + (j * o2 + k) * v12,
                         v12_slabs.data() + i * v1 * v22, R.data(), -1.0, 1.0);
    contractions.forward('N', 'T', v1, v12, v2, t2_12 + (i * o2 + k) * v12,
                         v12_slabs.data() + j * v1 * v22, R.data(), 1.0, 1.0);

    // Term 3: + sum_e v11_slab[i, a, b, e] * t2_12[j, k, e, c] - (i<->j)
    contractions.forward('N', 'N', v11, v2, v1, v11_slabs.data() + i * v1 * v11,
                         t2_12 + (j * o2 + k) * v12, R.data(), 1.0, 1.0);
    contractions.forward('N', 'N', v11, v2, v1, v11_slabs.data() + j * v1 * v11,
                         t2_12 + (i * o2 + k) * v12, R.data(), -1.0, 1.0);

    // Term 4: + sum_m g21_jk[j, k, a, m] * t2_12[i, m, b, c] - (i<->j)
    contractions.forward('N', 'N', v1, v12, o2, g21_jk.data() + (j * o2 + k) * v1 * o2,
                         t2_12 + i * o2 * v12, R.data(), 1.0, 1.0);
    contractions.forward('N', 'N', v1, v12, o2, g21_jk.data() + (i * o2 + k) * v1 * o2,
                         t2_12 + j * o2 * v12, R.data(), -1.0, 1.0);

    // Term 6: + sum_m gooov11[i, j, m, a] * t2_12_k_m[k, m, b, c]
    contractions.forward('T', 'N', v1, v12, o1, gooov11 + (i * o1 + j) * o1 * v1,
                         t2_12_k_m.data() + k * o1 * v12, R.data(), 1.0, 1.0);

    // Term 5 (already antisymmetric in a,b): - sum_m t2_11[i, m, a, b] * g12_kj[k, j, m, c] + (i<->j)
    // Add directly into W after antisymmetrizing R!
    for (std::size_t a = 0; a < v1; ++a)
      for (std::size_t b = 0; b < v1; ++b)
        for (std::size_t c = 0; c < v2; ++c)
          W[(a * v1 + b) * v2 + c] = R[(a * v1 + b) * v2 + c] - R[(b * v1 + a) * v2 + c];

    contractions.forward('T', 'N', v11, v2, o1, t2_11 + i * o1 * v11,
                         g12_kj.data() + (k * o1 + j) * o1 * v2, W.data(), -1.0, 1.0);
    contractions.forward('T', 'N', v11, v2, o1, t2_11 + j * o1 * v11,
                         g12_kj.data() + (k * o1 + i) * o1 * v2, W.data(), 1.0, 1.0);

    // Disconnected V[a, b, c]
    const double* t1i = t1_1 + i * v1;
    const double* t1j = t1_1 + j * v1;
    const double* t1k = t1_2 + k * v2;
    const double* fovi = fov1 + i * v1;
    const double* fovj = fov1 + j * v1;
    const double* fovk = fov2 + k * v2;
    const double* g_jk = goovv12 + (j * o2 + k) * v12;
    const double* g_ik = goovv12 + (i * o2 + k) * v12;
    const double* g_ij = goovv11 + (i * o1 + j) * v11;
    const double* t_jk = t2_12 + (j * o2 + k) * v12;
    const double* t_ik = t2_12 + (i * o2 + k) * v12;
    const double* t_ij = t2_11 + (i * o1 + j) * v11;
    for (std::size_t a = 0; a < v1; ++a)
      for (std::size_t b = 0; b < v1; ++b) {
        const double g_ij_ab = g_ij[a * v1 + b];
        const double t_ij_ab = t_ij[a * v1 + b];
        for (std::size_t c = 0; c < v2; ++c) {
          const std::size_t bc = b * v2 + c, ac = a * v2 + c;
          V[(a * v1 + b) * v2 + c] =
              (t1i[a] * g_jk[bc] + fovi[a] * t_jk[bc]) - (t1i[b] * g_jk[ac] + fovi[b] * t_jk[ac]) -
              (t1j[a] * g_ik[bc] + fovj[a] * t_ik[bc]) + (t1j[b] * g_ik[ac] + fovj[b] * t_ik[ac]) +
              (t1k[c] * g_ij_ab + fovk[c] * t_ij_ab);
        }
      }
  };

  std::vector<double> X(v_tot), Y(v_tot), bar_W(v_tot), bar_V(v_tot), bar_R(v_tot),
      bar_R_acb(v_tot), xy1(v11), xy2(v22);

  for (std::size_t i = 0; i < o1; ++i)
    for (std::size_t j = i + 1; j < o1; ++j)
      for (std::size_t k = 0; k < o2; ++k) {
        build_wv(i, j, k, w_abc, v_abc);
        const double eijk_o = eps_o1[i] + eps_o1[j] + eps_o2[k];
        const double wt = has_weights ? (w1[i] + w1[j] + w2[k]) / 3.0 : 1.0;
        const double eff = 0.5 * wt;
        double e_unw = 0.0;
        double diag_foo = 0.0;
        for (std::size_t a = 0; a < v1; ++a)
          for (std::size_t b = 0; b < v1; ++b)
            for (std::size_t c = 0; c < v2; ++c) {
              const std::size_t idx = (a * v1 + b) * v2 + c;
              const double den = eijk_o - eps_v1[a] - eps_v1[b] - eps_v2[c];
              if (std::abs(den) < den_thresh)
                throw std::runtime_error("DF-UCCSD(T) v2 small mixed-spin triples denominator");
              const double w = w_abc[idx], vv = v_abc[idx];
              const double x = w / den;
              const double y = (w + vv) / den;
              e_unw += w * y;
              if (compute_forces) {
                X[idx] = x;
                Y[idx] = y;
                bar_W[idx] = eff * (x + y);
                bar_V[idx] = eff * x;
                diag_foo -= eff * x * y;
              }
            }
        e_unw *= 0.5;
        total_e += wt * e_unw;
        occ1[i] += e_unw / 3.0;
        occ1[j] += e_unw / 3.0;
        occ2[k] += e_unw / 3.0;

        if (!compute_forces) continue;

        bar_foo1[i * o1 + i] += diag_foo;
        bar_foo1[j * o1 + j] += diag_foo;
        bar_foo2[k * o2 + k] += diag_foo;

        // Virtual Fock resolvent:
        // s1 (2 legs * 0.5 = 1.0): bar_fvv1[a, d] += 0.5 * wt * (X[a,bc] Y[d,bc] + Y[a,bc] X[d,bc])
        dgemm_counted('N', 'T', v1, v1, v12, X.data(), Y.data(), xy1.data(), contractions.fp64_gemms,
                      1.0, 0.0);
        for (std::size_t a = 0; a < v1; ++a)
          for (std::size_t d = 0; d < v1; ++d)
            bar_fvv1[a * v1 + d] += 0.5 * wt * (xy1[a * v1 + d] + xy1[d * v1 + a]);
        // s2 (1 leg * 0.5 = 0.5): bar_fvv2[c, f] += 0.25 * wt * (X[ab,c] Y[ab,f] + Y[ab,c] X[ab,f])
        dgemm_counted('T', 'N', v2, v2, v11, X.data(), Y.data(), xy2.data(), contractions.fp64_gemms,
                      1.0, 0.0);
        for (std::size_t c = 0; c < v2; ++c)
          for (std::size_t f = 0; f < v2; ++f)
            bar_fvv2[c * v2 + f] += 0.25 * wt * (xy2[c * v2 + f] + xy2[f * v2 + c]);

        // bar_R = 2.0 * bar_W
        for (std::size_t idx = 0; idx < v_tot; ++idx) bar_R[idx] = 2.0 * bar_W[idx];
        for (std::size_t a = 0; a < v1; ++a)
          for (std::size_t b = 0; b < v1; ++b)
            for (std::size_t c = 0; c < v2; ++c)
              bar_R_acb[(a * v2 + c) * v1 + b] = bar_R[(a * v1 + b) * v2 + c];

        // Pullback Term 1: tmp_acb[a, c*b] = t2_11[i,j,a,e] @ v21[k,c*b,e]^T
        contractions.pullback('N', 'N', v1, v1, v12, bar_R_acb.data(),
                              v21_slabs.data() + k * v2 * v11, bar_t2_11 + (i * o1 + j) * v11, 1.0,
                              1.0);
        contractions.pullback('T', 'N', v12, v1, v1, bar_R_acb.data(), t2_11 + (i * o1 + j) * v11,
                              bar_v21.data() + k * v2 * v11, 1.0, 1.0);

        // Pullback Term 2: - t2_12[j,k,a,e] @ v12[i,b*c,e]^T + (i<->j)
        contractions.pullback('N', 'N', v1, v2, v12, bar_R.data(), v12_slabs.data() + i * v1 * v22,
                              bar_t2_12 + (j * o2 + k) * v12, -1.0, 1.0);
        contractions.pullback('T', 'N', v12, v2, v1, bar_R.data(), t2_12 + (j * o2 + k) * v12,
                              bar_v12.data() + i * v1 * v22, -1.0, 1.0);
        contractions.pullback('N', 'N', v1, v2, v12, bar_R.data(), v12_slabs.data() + j * v1 * v22,
                              bar_t2_12 + (i * o2 + k) * v12, 1.0, 1.0);
        contractions.pullback('T', 'N', v12, v2, v1, bar_R.data(), t2_12 + (i * o2 + k) * v12,
                              bar_v12.data() + j * v1 * v22, 1.0, 1.0);

        // Pullback Term 3: + v11[i, a*b, e] @ t2_12[j,k,e,c] - (i<->j)
        contractions.pullback('N', 'T', v11, v1, v2, bar_R.data(), t2_12 + (j * o2 + k) * v12,
                              bar_v11.data() + i * v1 * v11, 1.0, 1.0);
        contractions.pullback('T', 'N', v1, v2, v11, v11_slabs.data() + i * v1 * v11, bar_R.data(),
                              bar_t2_12 + (j * o2 + k) * v12, 1.0, 1.0);
        contractions.pullback('N', 'T', v11, v1, v2, bar_R.data(), t2_12 + (i * o2 + k) * v12,
                              bar_v11.data() + j * v1 * v11, -1.0, 1.0);
        contractions.pullback('T', 'N', v1, v2, v11, v11_slabs.data() + j * v1 * v11, bar_R.data(),
                              bar_t2_12 + (i * o2 + k) * v12, -1.0, 1.0);

        // Pullback Term 4: + g21_jk[j,k,a,m] @ t2_12[i,m,b*c] - (i<->j)
        contractions.pullback('N', 'T', v1, o2, v12, bar_R.data(), t2_12 + i * o2 * v12,
                              bar_g21_jk.data() + (j * o2 + k) * v1 * o2, 1.0, 1.0);
        contractions.pullback('T', 'N', o2, v12, v1, g21_jk.data() + (j * o2 + k) * v1 * o2,
                              bar_R.data(), bar_t2_12 + i * o2 * v12, 1.0, 1.0);
        contractions.pullback('N', 'T', v1, o2, v12, bar_R.data(), t2_12 + j * o2 * v12,
                              bar_g21_jk.data() + (i * o2 + k) * v1 * o2, -1.0, 1.0);
        contractions.pullback('T', 'N', o2, v12, v1, g21_jk.data() + (i * o2 + k) * v1 * o2,
                              bar_R.data(), bar_t2_12 + j * o2 * v12, -1.0, 1.0);

        // Pullback Term 6: + gooov11[i,j,m,a]^T @ t2_12_k_m[k,m,b*c]
        contractions.pullback('N', 'T', o1, v1, v12, t2_12_k_m.data() + k * o1 * v12, bar_R.data(),
                              bar_gooov11 + (i * o1 + j) * o1 * v1, 1.0, 1.0);
        contractions.pullback('N', 'N', o1, v12, v1, gooov11 + (i * o1 + j) * o1 * v1, bar_R.data(),
                              bar_t2_12_k_m.data() + k * o1 * v12, 1.0, 1.0);

        // Pullback Term 5 (directly from bar_W): - t2_11[i,m,a*b]^T @ g12_kj[k,j,m,c] + (i<->j)
        contractions.pullback('N', 'T', o1, v11, v2, g12_kj.data() + (k * o1 + j) * o1 * v2,
                              bar_W.data(), bar_t2_11 + i * o1 * v11, -1.0, 1.0);
        contractions.pullback('N', 'N', o1, v2, v11, t2_11 + i * o1 * v11, bar_W.data(),
                              bar_g12_kj.data() + (k * o1 + j) * o1 * v2, -1.0, 1.0);
        contractions.pullback('N', 'T', o1, v11, v2, g12_kj.data() + (k * o1 + i) * o1 * v2,
                              bar_W.data(), bar_t2_11 + j * o1 * v11, 1.0, 1.0);
        contractions.pullback('N', 'N', o1, v2, v11, t2_11 + j * o1 * v11, bar_W.data(),
                              bar_g12_kj.data() + (k * o1 + i) * o1 * v2, 1.0, 1.0);

        // Pullback disconnected V[a, b, c]
        const double* t1i = t1_1 + i * v1;
        const double* t1j = t1_1 + j * v1;
        const double* t1k = t1_2 + k * v2;
        const double* fovi = fov1 + i * v1;
        const double* fovj = fov1 + j * v1;
        const double* fovk = fov2 + k * v2;
        const double* g_jk = goovv12 + (j * o2 + k) * v12;
        const double* g_ik = goovv12 + (i * o2 + k) * v12;
        const double* g_ij = goovv11 + (i * o1 + j) * v11;
        const double* t_jk = t2_12 + (j * o2 + k) * v12;
        const double* t_ik = t2_12 + (i * o2 + k) * v12;
        const double* t_ij = t2_11 + (i * o1 + j) * v11;
        double* bg_jk = bar_goovv12 + (j * o2 + k) * v12;
        double* bg_ik = bar_goovv12 + (i * o2 + k) * v12;
        double* bg_ij = bar_goovv11 + (i * o1 + j) * v11;
        double* bt_jk = bar_t2_12 + (j * o2 + k) * v12;
        double* bt_ik = bar_t2_12 + (i * o2 + k) * v12;
        double* bt_ij = bar_t2_11 + (i * o1 + j) * v11;
        for (std::size_t a = 0; a < v1; ++a) {
          double st1i = 0.0, sfovi = 0.0, st1j = 0.0, sfovj = 0.0;
          const double t1ia = 2.0 * t1i[a], fia = 2.0 * fovi[a];
          const double t1ja = 2.0 * t1j[a], fja = 2.0 * fovj[a];
          const double* bv_a = bar_V.data() + a * v12;
          for (std::size_t bc = 0; bc < v12; ++bc) {
            const double bv_val = bv_a[bc];
            st1i += bv_val * g_jk[bc];
            sfovi += bv_val * t_jk[bc];
            st1j -= bv_val * g_ik[bc];
            sfovj -= bv_val * t_ik[bc];
            bg_jk[bc] += t1ia * bv_val;
            bt_jk[bc] += fia * bv_val;
            bg_ik[bc] -= t1ja * bv_val;
            bt_ik[bc] -= fja * bv_val;
          }
          bar_t1_1[i * v1 + a] += 2.0 * st1i;
          bar_fov1[i * v1 + a] += 2.0 * sfovi;
          bar_t1_1[j * v1 + a] += 2.0 * st1j;
          bar_fov1[j * v1 + a] += 2.0 * sfovj;
        }
        for (std::size_t a = 0; a < v1; ++a)
          for (std::size_t b = 0; b < v1; ++b) {
            const std::size_t ab = a * v1 + b;
            const double g_ab = g_ij[ab], t_ab = t_ij[ab];
            double s_g = 0.0, s_t = 0.0;
            const double* bv_ab = bar_V.data() + ab * v2;
            for (std::size_t c = 0; c < v2; ++c) {
              const double bv_val = bv_ab[c];
              bar_t1_2[k * v2 + c] += bv_val * g_ab;
              bar_fov2[k * v2 + c] += bv_val * t_ab;
              s_g += bv_val * t1k[c];
              s_t += bv_val * fovk[c];
            }
            bg_ij[ab] += s_g;
            bt_ij[ab] += s_t;
          }
      }

  if (!compute_forces) return;

  // Scatter transposed intermediate pullbacks back to canonical arrays
  for (std::size_t m = 0; m < o2; ++m)
    for (std::size_t j = 0; j < o1; ++j)
      for (std::size_t k = 0; k < o2; ++k)
        for (std::size_t a = 0; a < v1; ++a)
          bar_gooov21[((m * o1 + j) * o2 + k) * v1 + a] +=
              bar_g21_jk[((j * o2 + k) * v1 + a) * o2 + m];

  for (std::size_t m = 0; m < o1; ++m)
    for (std::size_t k = 0; k < o2; ++k)
      for (std::size_t j = 0; j < o1; ++j)
        for (std::size_t c = 0; c < v2; ++c)
          bar_gooov12[((m * o2 + k) * o1 + j) * v2 + c] +=
              bar_g12_kj[((k * o1 + j) * o1 + m) * v2 + c];

  for (std::size_t m = 0; m < o1; ++m)
    for (std::size_t k = 0; k < o2; ++k)
      for (std::size_t bc = 0; bc < v12; ++bc)
        bar_t2_12[(m * o2 + k) * v12 + bc] += bar_t2_12_k_m[(k * o1 + m) * v12 + bc];

  for (std::size_t p = 0; p < o1; ++p) {
    pullback_ovvv_slab(q, o1, v1, v1, p, bov1, bvv1, bar_v11.data() + p * v1 * v11, bar_bov1,
                       bar_bvv1, contractions);
    pullback_ovvv_slab(q, o1, v1, v2, p, bov1, bvv2, bar_v12.data() + p * v1 * v22, bar_bov1,
                       bar_bvv2, contractions);
  }
  for (std::size_t k = 0; k < o2; ++k) {
    pullback_ovvv_slab(q, o2, v2, v1, k, bov2, bvv1, bar_v21.data() + k * v2 * v11, bar_bov2,
                       bar_bvv1, contractions);
  }

  // Occupied same-space Fock resolvent off-diagonals for s1 (bar_foo1) and s2 (bar_foo2)
  bool need_s1 = !has_weights, need_s2 = !has_weights;
  if (has_weights) {
    for (std::size_t p = 0; p < o1 && !need_s1; ++p)
      for (std::size_t r = p + 1; r < o1; ++r)
        if (std::abs(eps_o1[p] - eps_o1[r]) < 1e-6) need_s1 = true;
    for (std::size_t p = 0; p < o2 && !need_s2; ++p)
      for (std::size_t r = p + 1; r < o2; ++r)
        if (std::abs(eps_o2[p] - eps_o2[r]) < 1e-6) need_s2 = true;
  }
  if (need_s1 && o1 >= 2) {
    std::vector<double> xp(o1 * v_tot), yp(o1 * v_tot);
    std::vector<bool> active(o1, false);
    for (std::size_t j = 0; j < o1; ++j)
      for (std::size_t k = 0; k < o2; ++k) {
        std::fill(active.begin(), active.end(), false);
        for (std::size_t p = 0; p < o1; ++p) {
          if (p == j) continue;
          if (!has_weights) {
            active[p] = true;
          } else {
            for (std::size_t r = 0; r < o1; ++r)
              if (r != p && r != j && std::abs(eps_o1[p] - eps_o1[r]) < 1e-6) {
                active[p] = true;
                break;
              }
          }
        }
        for (std::size_t p = 0; p < o1; ++p) {
          if (!active[p]) continue;
          build_wv(p, j, k, w_abc, v_abc);
          const double epjk = eps_o1[p] + eps_o1[j] + eps_o2[k];
          double* xptr = xp.data() + p * v_tot;
          double* yptr = yp.data() + p * v_tot;
          for (std::size_t a = 0; a < v1; ++a)
            for (std::size_t b = 0; b < v1; ++b)
              for (std::size_t c = 0; c < v2; ++c) {
                const std::size_t idx = (a * v1 + b) * v2 + c;
                const double den = epjk - eps_v1[a] - eps_v1[b] - eps_v2[c];
                xptr[idx] = w_abc[idx] / den;
                yptr[idx] = (w_abc[idx] + v_abc[idx]) / den;
              }
        }
        for (std::size_t p = 0; p < o1; ++p) {
          if (!active[p]) continue;
          for (std::size_t r = p + 1; r < o1; ++r) {
            if (!active[r]) continue;
            if (has_weights && std::abs(eps_o1[p] - eps_o1[r]) >= 1e-6) continue;
            const double wt = has_weights ? (w1[p] + w1[j] + w2[k]) / 3.0 : 1.0;
            const double* x_p = xp.data() + p * v_tot;
            const double* y_p = yp.data() + p * v_tot;
            const double* x_r = xp.data() + r * v_tot;
            const double* y_r = yp.data() + r * v_tot;
            double dot = 0.0;
            for (std::size_t idx = 0; idx < v_tot; ++idx)
              dot += x_p[idx] * y_r[idx] + y_p[idx] * x_r[idx];
            const double val = -0.25 * wt * dot;
            bar_foo1[p * o1 + r] += val;
            bar_foo1[r * o1 + p] += val;
          }
        }
      }
  }
  if (need_s2 && o2 >= 2) {
    std::vector<double> xq(o2 * v_tot), yq(o2 * v_tot);
    std::vector<bool> active(o2, false);
    for (std::size_t p = 0; p < o2; ++p) {
      if (!has_weights) {
        active[p] = true;
      } else {
        for (std::size_t r = 0; r < o2; ++r)
          if (r != p && std::abs(eps_o2[p] - eps_o2[r]) < 1e-6) {
            active[p] = true;
            break;
          }
      }
    }
    for (std::size_t i = 0; i < o1; ++i)
      for (std::size_t j = i + 1; j < o1; ++j) {
        for (std::size_t p = 0; p < o2; ++p) {
          if (!active[p]) continue;
          build_wv(i, j, p, w_abc, v_abc);
          const double eijp = eps_o1[i] + eps_o1[j] + eps_o2[p];
          double* xptr = xq.data() + p * v_tot;
          double* yptr = yq.data() + p * v_tot;
          for (std::size_t a = 0; a < v1; ++a)
            for (std::size_t b = 0; b < v1; ++b)
              for (std::size_t c = 0; c < v2; ++c) {
                const std::size_t idx = (a * v1 + b) * v2 + c;
                const double den = eijp - eps_v1[a] - eps_v1[b] - eps_v2[c];
                xptr[idx] = w_abc[idx] / den;
                yptr[idx] = (w_abc[idx] + v_abc[idx]) / den;
              }
        }
        for (std::size_t p = 0; p < o2; ++p) {
          if (!active[p]) continue;
          for (std::size_t r = p + 1; r < o2; ++r) {
            if (!active[r]) continue;
            if (has_weights && std::abs(eps_o2[p] - eps_o2[r]) >= 1e-6) continue;
            const double wt = has_weights ? (w1[i] + w1[j] + w2[p]) / 3.0 : 1.0;
            const double* x_p = xq.data() + p * v_tot;
            const double* y_p = yq.data() + p * v_tot;
            const double* x_r = xq.data() + r * v_tot;
            const double* y_r = yq.data() + r * v_tot;
            double dot = 0.0;
            for (std::size_t idx = 0; idx < v_tot; ++idx)
              dot += x_p[idx] * y_r[idx] + y_p[idx] * x_r[idx];
            const double val = -0.25 * wt * dot;
            bar_foo2[p * o2 + r] += val;
            bar_foo2[r * o2 + p] += val;
          }
        }
      }
  }
}

void antisymmetrize_t2(std::size_t o, std::size_t v, std::vector<double>& t2) {
  if (!o || !v) return;
  std::vector<double> out(t2.size(), 0.0);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b)
          out[((i * o + j) * v + a) * v + b] =
              0.25 * (t2[((i * o + j) * v + a) * v + b] - t2[((j * o + i) * v + a) * v + b] -
                      t2[((i * o + j) * v + b) * v + a] + t2[((j * o + i) * v + b) * v + a]);
  t2.swap(out);
}

}  // namespace

SpinBlockedTriplesResult evaluate_df_ucc_triples_cpu(const SpinBlockedDFSource& src,
                                                     const SpinBlockedFourIndexCore& core,
                                                     const SpinBlockedUccAmplitudes& amp,
                                                     const DFUccV2Options& options) {
  const auto& d = src.dims;
  const bool has_weights =
      !options.occupied_weights_alpha.empty() || !options.occupied_weights_beta.empty();
  if (has_weights) {
    auto check_deg = [](const std::vector<double>& eps, const std::vector<double>& w) {
      for (std::size_t a = 0; a < eps.size(); ++a) {
        if (!std::isfinite(w[a]))
          throw std::invalid_argument("DF-UCCSD(T+) v2 nonfinite occupied weight");
        for (std::size_t b = a + 1; b < eps.size(); ++b) {
          if (std::abs(eps[a] - eps[b]) < 1e-6 &&
              std::abs(w[a] - w[b]) > 1e-8 * (1.0 + std::max(std::abs(w[a]), std::abs(w[b]))))
            throw std::invalid_argument(
                "DF-UCCSD(T+) v2 requires equal weights across degenerate occupied orbitals");
        }
      }
    };
    check_deg(src.eps_oa, options.occupied_weights_alpha);
    check_deg(src.eps_ob, options.occupied_weights_beta);
  }

  admit_ucc_triples_host(src, core, amp, options);

  SpinBlockedTriplesResult out;
  out.occupied_alpha.assign(d.oa, 0.0);
  out.occupied_beta.assign(d.ob, 0.0);
  if (options.compute_forces) {
    out.bar_t.allocate_zero(d);
    out.bar_b.allocate_zero(d);
    out.bar_core.allocate_zero(d);
    out.bar_f.allocate_zero(d);
  }

  const WContractions contractions{options.triples_fp32_w, options.triples_compensated_w,
                                   out.fp32_gemms, out.fp32_reverse_gemms, out.fp64_gemms};
  // Channel 1: aaa
  run_same_spin_triples(
      d.oa, d.va, d.q, src.bov_a.data(), src.bvv_a.data(), core.goovv_aa.data(),
      core.gooov_aa.data(), src.fov_a.data(), amp.t1a.data(), amp.t2aa.data(), src.eps_oa.data(),
      src.eps_va.data(), options.occupied_weights_alpha, has_weights, options.compute_forces,
      contractions, options.denominator_threshold, out.energy, out.occupied_alpha,
      options.compute_forces ? out.bar_t.t1a.data() : nullptr,
      options.compute_forces ? out.bar_t.t2aa.data() : nullptr,
      options.compute_forces ? out.bar_b.bov_a.data() : nullptr,
      options.compute_forces ? out.bar_b.bvv_a.data() : nullptr,
      options.compute_forces ? out.bar_core.goovv_aa.data() : nullptr,
      options.compute_forces ? out.bar_core.gooov_aa.data() : nullptr,
      options.compute_forces ? out.bar_f.fov_a.data() : nullptr,
      options.compute_forces ? out.bar_f.foo_a.data() : nullptr,
      options.compute_forces ? out.bar_f.fvv_a.data() : nullptr);

  // Channel 2: bbb
  run_same_spin_triples(
      d.ob, d.vb, d.q, src.bov_b.data(), src.bvv_b.data(), core.goovv_bb.data(),
      core.gooov_bb.data(), src.fov_b.data(), amp.t1b.data(), amp.t2bb.data(), src.eps_ob.data(),
      src.eps_vb.data(), options.occupied_weights_beta, has_weights, options.compute_forces,
      contractions, options.denominator_threshold, out.energy, out.occupied_beta,
      options.compute_forces ? out.bar_t.t1b.data() : nullptr,
      options.compute_forces ? out.bar_t.t2bb.data() : nullptr,
      options.compute_forces ? out.bar_b.bov_b.data() : nullptr,
      options.compute_forces ? out.bar_b.bvv_b.data() : nullptr,
      options.compute_forces ? out.bar_core.goovv_bb.data() : nullptr,
      options.compute_forces ? out.bar_core.gooov_bb.data() : nullptr,
      options.compute_forces ? out.bar_f.fov_b.data() : nullptr,
      options.compute_forces ? out.bar_f.foo_b.data() : nullptr,
      options.compute_forces ? out.bar_f.fvv_b.data() : nullptr);

  // Channel 3: aab
  run_mixed_spin_triples(
      d.oa, d.ob, d.va, d.vb, d.q, src.bov_a.data(), src.bvv_a.data(), src.bov_b.data(),
      src.bvv_b.data(), core.goovv_aa.data(), core.goovv_ab.data(), core.gooov_aa.data(),
      core.gooov_ab.data(), core.gooov_ba.data(), src.fov_a.data(), src.fov_b.data(),
      amp.t1a.data(), amp.t1b.data(), amp.t2aa.data(), amp.t2ab.data(), src.eps_oa.data(),
      src.eps_va.data(), src.eps_ob.data(), src.eps_vb.data(), options.occupied_weights_alpha,
      options.occupied_weights_beta, has_weights, options.compute_forces, contractions,
      options.denominator_threshold, out.energy, out.occupied_alpha, out.occupied_beta,
      options.compute_forces ? out.bar_t.t1a.data() : nullptr,
      options.compute_forces ? out.bar_t.t1b.data() : nullptr,
      options.compute_forces ? out.bar_t.t2aa.data() : nullptr,
      options.compute_forces ? out.bar_t.t2ab.data() : nullptr,
      options.compute_forces ? out.bar_b.bov_a.data() : nullptr,
      options.compute_forces ? out.bar_b.bvv_a.data() : nullptr,
      options.compute_forces ? out.bar_b.bov_b.data() : nullptr,
      options.compute_forces ? out.bar_b.bvv_b.data() : nullptr,
      options.compute_forces ? out.bar_core.goovv_aa.data() : nullptr,
      options.compute_forces ? out.bar_core.goovv_ab.data() : nullptr,
      options.compute_forces ? out.bar_core.gooov_aa.data() : nullptr,
      options.compute_forces ? out.bar_core.gooov_ab.data() : nullptr,
      options.compute_forces ? out.bar_core.gooov_ba.data() : nullptr,
      options.compute_forces ? out.bar_f.fov_a.data() : nullptr,
      options.compute_forces ? out.bar_f.fov_b.data() : nullptr,
      options.compute_forces ? out.bar_f.foo_a.data() : nullptr,
      options.compute_forces ? out.bar_f.foo_b.data() : nullptr,
      options.compute_forces ? out.bar_f.fvv_a.data() : nullptr,
      options.compute_forces ? out.bar_f.fvv_b.data() : nullptr);

  // Channel 4: abb (exact mirror of aab via transposed t2ba and goovv_ba)
  if (d.ob >= 2 && d.oa >= 1 && d.vb >= 2 && d.va >= 1) {
    std::vector<double> t2ba(d.ob * d.oa * d.vb * d.va, 0.0);
    std::vector<double> goovv_ba(d.ob * d.oa * d.vb * d.va, 0.0);
    for (std::size_t i = 0; i < d.oa; ++i)
      for (std::size_t j = 0; j < d.ob; ++j)
        for (std::size_t a = 0; a < d.va; ++a)
          for (std::size_t b = 0; b < d.vb; ++b) {
            const std::size_t src_idx = ((i * d.ob + j) * d.va + a) * d.vb + b;
            const std::size_t dst_idx = ((j * d.oa + i) * d.vb + b) * d.va + a;
            t2ba[dst_idx] = amp.t2ab[src_idx];
            goovv_ba[dst_idx] = core.goovv_ab[src_idx];
          }
    std::vector<double> bar_t2ba(options.compute_forces ? t2ba.size() : 0, 0.0);
    std::vector<double> bar_goovv_ba(options.compute_forces ? goovv_ba.size() : 0, 0.0);
    run_mixed_spin_triples(
        d.ob, d.oa, d.vb, d.va, d.q, src.bov_b.data(), src.bvv_b.data(), src.bov_a.data(),
        src.bvv_a.data(), core.goovv_bb.data(), goovv_ba.data(), core.gooov_bb.data(),
        core.gooov_ba.data(), core.gooov_ab.data(), src.fov_b.data(), src.fov_a.data(),
        amp.t1b.data(), amp.t1a.data(), amp.t2bb.data(), t2ba.data(), src.eps_ob.data(),
        src.eps_vb.data(), src.eps_oa.data(), src.eps_va.data(), options.occupied_weights_beta,
        options.occupied_weights_alpha, has_weights, options.compute_forces, contractions,
        options.denominator_threshold, out.energy, out.occupied_beta, out.occupied_alpha,
        options.compute_forces ? out.bar_t.t1b.data() : nullptr,
        options.compute_forces ? out.bar_t.t1a.data() : nullptr,
        options.compute_forces ? out.bar_t.t2bb.data() : nullptr,
        options.compute_forces ? bar_t2ba.data() : nullptr,
        options.compute_forces ? out.bar_b.bov_b.data() : nullptr,
        options.compute_forces ? out.bar_b.bvv_b.data() : nullptr,
        options.compute_forces ? out.bar_b.bov_a.data() : nullptr,
        options.compute_forces ? out.bar_b.bvv_a.data() : nullptr,
        options.compute_forces ? out.bar_core.goovv_bb.data() : nullptr,
        options.compute_forces ? bar_goovv_ba.data() : nullptr,
        options.compute_forces ? out.bar_core.gooov_bb.data() : nullptr,
        options.compute_forces ? out.bar_core.gooov_ba.data() : nullptr,
        options.compute_forces ? out.bar_core.gooov_ab.data() : nullptr,
        options.compute_forces ? out.bar_f.fov_b.data() : nullptr,
        options.compute_forces ? out.bar_f.fov_a.data() : nullptr,
        options.compute_forces ? out.bar_f.foo_b.data() : nullptr,
        options.compute_forces ? out.bar_f.foo_a.data() : nullptr,
        options.compute_forces ? out.bar_f.fvv_b.data() : nullptr,
        options.compute_forces ? out.bar_f.fvv_a.data() : nullptr);
    if (options.compute_forces) {
      for (std::size_t i = 0; i < d.oa; ++i)
        for (std::size_t j = 0; j < d.ob; ++j)
          for (std::size_t a = 0; a < d.va; ++a)
            for (std::size_t b = 0; b < d.vb; ++b) {
              const std::size_t ab_idx = ((i * d.ob + j) * d.va + a) * d.vb + b;
              const std::size_t ba_idx = ((j * d.oa + i) * d.vb + b) * d.va + a;
              out.bar_t.t2ab[ab_idx] += bar_t2ba[ba_idx];
              out.bar_core.goovv_ab[ab_idx] += bar_goovv_ba[ba_idx];
            }
    }
  }

  validate_ucc_triples_partition(out.energy, out.occupied_alpha, out.occupied_beta, options);

  if (options.compute_forces) {
    antisymmetrize_t2(d.oa, d.va, out.bar_t.t2aa);
    antisymmetrize_t2(d.ob, d.vb, out.bar_t.t2bb);
  }
  return out;
}

}  // namespace generativeqc::cc::v2_detail
