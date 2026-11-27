#include "cc/explicit_tf12_sp.hpp"
#include "cc/explicit_tf12_sp_internal.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "posthf/capacity.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::cc::explicit_tf12 {
namespace {

using Clock = std::chrono::steady_clock;
using posthf::checked_add;
using posthf::checked_mul;
using tensor::cpu_gemm;

double elapsed_seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void check_finite_span(const double* ptr, std::size_t n, const char* name) {
  if (n && !ptr) throw std::invalid_argument(std::string("null explicit (T)-F12 input ") + name);
  for (std::size_t i = 0; i < n; ++i)
    if (!std::isfinite(ptr[i]))
      throw std::invalid_argument(std::string("nonfinite explicit (T)-F12 input ") + name);
}

}  // namespace

namespace detail {

// Evaluate all spatial closed-shell A-sector contractions from assembled spatial blocks.
// Inputs are active correlated spatial blocks (o = o_act, v = n_vir):
//   V7[o,v], V8[v,v], V9[o,o,o,v], V10_d[o,v,o,v], V10_e[o,v,o,v]
//   X2[o,v], X4[v,v], X6[o,o,o,v], X7_d[o,v,o,v],  X7_e[o,v,o,v]
//   B4[v,v], B7_d[o,v,o,v], B7_e[o,v,o,v], Bh4[v,v]
//   H7[o,o,o,v] (H7[k,l,j,c] = ovoo[l,c,k,j])
void evaluate_a_contractions_from_blocks(
    std::size_t o, std::size_t v, const double* fov, const double* t1, const double* t2,
    const double* eps_o, const double* eps_v, const double* ovoo, const double* V7,
    const double* V8, const double* V9, const double* V10_d, const double* V10_e, const double* X2,
    const double* X4, const double* X6, const double* X7_d, const double* X7_e, const double* B4,
    const double* B7_d, const double* B7_e, const double* Bh4, double t5_z2, Breakdown& out) {
  const std::size_t ov = checked_mul(o, v);
  const std::size_t oo = checked_mul(o, o);
  const std::size_t vv = checked_mul(v, v);
  const std::size_t n2 = checked_mul(oo, vv);

  // Build t2a[i,j,a,b] = t2[i,j,a,b] - t2[i,j,b,a], t2_ba = -t2_swap, t2t = 2*t2 - t2_swap.
  std::vector<double> t2a(n2), t2_ba(n2), t2t(n2);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const std::size_t idx = ((i * o + j) * v + a) * v + b;
          const std::size_t swp = ((i * o + j) * v + b) * v + a;
          const double val = t2[idx], sw = t2[swp];
          t2a[idx] = val - sw;
          t2_ba[idx] = -sw;
          t2t[idx] = 2.0 * val - sw;
        }

  // H7[k, l, j, c] = (l c | k j) = ovoo[l, c, k, j].
  auto h7 = [&](std::size_t k, std::size_t l, std::size_t j, std::size_t c) -> double {
    return ovoo[((l * v + c) * o + k) * o + j];
  };

  // t1 = +2 * sum_{i,j,a,b} fov[i,a] * X2[j,b] * t2t[i,j,a,b]
  // t7 = +2 * sum_{i,j,a,b} t1[i,a]  * V7[j,b] * t2t[i,j,a,b]
  double t1_val = 0.0, t7_val = 0.0;
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a) {
        const double f_ia = fov[i * v + a];
        const double t1_ia = t1[i * v + a];
        for (std::size_t b = 0; b < v; ++b) {
          const double tt = t2t[((i * o + j) * v + a) * v + b];
          t1_val += f_ia * X2[j * v + b] * tt;
          t7_val += t1_ia * V7[j * v + b] * tt;
        }
      }
  out.t1_h3_x2 = 2.0 * t1_val;
  out.t7_t1_v7 = 2.0 * t7_val;

  // m_hhhp_t1_t2_sp(U, M, t2t) = -2 * sum_{i,j,k,a,b} U[i,a] * M[j,k,i,b] * t2t[j,k,a,b]
  // Used for t2 (fov, X6), t4/t15 (X2, H7), t6 (t1, V9).
  // Contract (j, k, b) with t2t first: W_t2[i, a] = sum_{j,k,b} M[j,k,i,b] * t2t[j,k,a,b]
  auto m_hhhp_dot = [&](const double* U, const auto& M_fn) -> double {
    double acc = 0.0;
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t k = 0; k < o; ++k) {
        // S[i, a] = sum_b M[j, k, i, b] * t2t[j, k, a, b]
        for (std::size_t i = 0; i < o; ++i)
          for (std::size_t a = 0; a < v; ++a) {
            const double u_ia = U[i * v + a];
            if (u_ia == 0.0) continue;
            const double* t_row = t2t.data() + ((j * o + k) * v + a) * v;
            double s = 0.0;
            for (std::size_t b = 0; b < v; ++b) s += M_fn(j, k, i, b) * t_row[b];
            acc += u_ia * s;
          }
      }
    return -2.0 * acc;
  };
  out.t2_h3_x6 = m_hhhp_dot(fov, [&](std::size_t j, std::size_t k, std::size_t i, std::size_t b) {
    return X6[((j * o + k) * o + i) * v + b];
  });
  out.t4_x2_h7 = m_hhhp_dot(X2, h7);
  out.t15_x3_h9 = out.t4_x2_h7;
  out.t6_t1_v9 = m_hhhp_dot(t1, [&](std::size_t j, std::size_t k, std::size_t i, std::size_t b) {
    return V9[((j * o + k) * o + i) * v + b];
  });

  // t3 & t14: 2 * sum_{i,j,k,l,a,b} [X6a*H7a*T2a + X6s*H7s*T2a + X6*H7*T2 + X6a*H7s*T2s + X6s*H7a*T2s]
  // Factorize over (j, l): P[i, k, b, a] = sum_{j,l} X6(i,j,l,b) * H7(k,l,j,a) -> O(o^4 v^2)!
  auto contract_t3_pair = [&](const std::vector<double>& X_in, const std::vector<double>& H_in,
                              const std::vector<double>& T_in) -> double {
    // X_in[i, b, j, l] (shape o*v x o*o), H_in[k, a, j, l] (shape o*v x o*o)
    // Wait: sum_{j,l} X[i,b,j,l] * H[k,a,j,l] = C[i,b,k,a] -> dot with T[i,k,a,b]
    std::vector<double> C(checked_mul(ov, ov), 0.0);
    cpu_gemm('N', 'T', ov, ov, oo, X_in.data(), H_in.data(), C.data(), 1.0, 0.0);
    double s = 0.0;
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t b = 0; b < v; ++b)
        for (std::size_t k = 0; k < o; ++k)
          for (std::size_t a = 0; a < v; ++a)
            s += C[(i * v + b) * ov + (k * v + a)] * T_in[((i * o + k) * v + a) * v + b];
    return s;
  };
  std::vector<double> x6_jl(checked_mul(ov, oo)), x6s_jl(checked_mul(ov, oo)),
      x6a_jl(checked_mul(ov, oo));
  std::vector<double> h7_jl(checked_mul(ov, oo)), h7s_jl(checked_mul(ov, oo)),
      h7a_jl(checked_mul(ov, oo));
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t l = 0; l < o; ++l)
        for (std::size_t b = 0; b < v; ++b) {
          const double d = X6[((i * o + j) * o + l) * v + b];
          const double s = X6[((j * o + i) * o + l) * v + b];
          const std::size_t dst = (i * v + b) * oo + (j * o + l);
          x6_jl[dst] = d;
          x6s_jl[dst] = s;
          x6a_jl[dst] = d - s;
        }
  for (std::size_t k = 0; k < o; ++k)
    for (std::size_t l = 0; l < o; ++l)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t a = 0; a < v; ++a) {
          const double d = h7(k, l, j, a);
          const double s = h7(l, k, j, a);
          const std::size_t dst = (k * v + a) * oo + (j * o + l);
          h7_jl[dst] = d;
          h7s_jl[dst] = s;
          h7a_jl[dst] = d - s;
        }
  std::vector<double> t2_vec(t2, t2 + n2), t2s_vec(n2);
  for (std::size_t idx = 0; idx < n2; ++idx) t2s_vec[idx] = -t2_ba[idx];
  const double t3_val =
      2.0 * (contract_t3_pair(x6a_jl, h7a_jl, t2a) + contract_t3_pair(x6s_jl, h7s_jl, t2a) +
             contract_t3_pair(x6_jl, h7_jl, t2_vec) + contract_t3_pair(x6a_jl, h7s_jl, t2s_vec) +
             contract_t3_pair(x6s_jl, h7a_jl, t2s_vec));
  out.t3_xh_h7 = t3_val;
  out.t14_xh_h9 = t3_val;

  // 1-body particle contractions:
  // p1_gen(La, Lab, Lba, M_vv, Ra, Rab, Rba) = 2 * sum_{i,j,a,b,c} (La*M*Ra + Lab*M*Rab + Lba*M*Rba)
  // Build compact vv densities G_0[b, c], G_oj[b, c] (weighted by eps_o[j]), G_va[b, c] (eps_v[a]):
  std::vector<double> G_0(vv, 0.0), G_oj(vv, 0.0), G_va(vv, 0.0);
  // Pack T[ij*v + a, b] -> G = 2 * (La^T La + Lab^T Lab + Lba^T Lba)!
  // Note: unweighted G_0 via 3 GEMMs of (v, o*o*v) @ (o*o*v, v) = O(o^2 v^3)!
  const std::size_t oov = checked_mul(oo, v);
  cpu_gemm('T', 'N', v, v, oov, t2a.data(), t2a.data(), G_0.data(), 2.0, 0.0);
  cpu_gemm('T', 'N', v, v, oov, t2, t2, G_0.data(), 2.0, 1.0);
  cpu_gemm('T', 'N', v, v, oov, t2_ba.data(), t2_ba.data(), G_0.data(), 2.0, 1.0);

  std::vector<double> la_j(n2), lab_j(n2), lba_j(n2);
  std::vector<double> la_i(n2), lab_i(n2), lba_i(n2);
  std::vector<double> la_a(n2), lab_a(n2), lba_a(n2);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const std::size_t idx = ((i * o + j) * v + a) * v + b;
          const double ej = eps_o[j], ei = eps_o[i], ea = eps_v[a];
          la_j[idx] = t2a[idx] * ej;
          lab_j[idx] = t2[idx] * ej;
          lba_j[idx] = t2_ba[idx] * ej;
          la_i[idx] = t2a[idx] * ei;
          lab_i[idx] = t2[idx] * ei;
          lba_i[idx] = t2_ba[idx] * ei;
          la_a[idx] = t2a[idx] * ea;
          lab_a[idx] = t2[idx] * ea;
          lba_a[idx] = t2_ba[idx] * ea;
        }
  cpu_gemm('T', 'N', v, v, oov, la_j.data(), t2a.data(), G_oj.data(), 2.0, 0.0);
  cpu_gemm('T', 'N', v, v, oov, lab_j.data(), t2, G_oj.data(), 2.0, 1.0);
  cpu_gemm('T', 'N', v, v, oov, lba_j.data(), t2_ba.data(), G_oj.data(), 2.0, 1.0);

  cpu_gemm('T', 'N', v, v, oov, la_a.data(), t2a.data(), G_va.data(), 2.0, 0.0);
  cpu_gemm('T', 'N', v, v, oov, lab_a.data(), t2, G_va.data(), 2.0, 1.0);
  cpu_gemm('T', 'N', v, v, oov, lba_a.data(), t2_ba.data(), G_va.data(), 2.0, 1.0);

  auto dot_n = [](const double* a, const double* b, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
  };
  out.t8_v8 = dot_n(G_0.data(), V8, vv);
  out.t16_bh4 = 0.5 * dot_n(G_0.data(), Bh4, vv);
  out.t19_fo_x4 = -dot_n(G_oj.data(), X4, vv);
  out.t21_b4 = 0.5 * dot_n(G_0.data(), B4, vv);
  out.t23_fv_x4 = 0.5 * dot_n(G_va.data(), X4, vv);

  // 2-body ring contractions:
  // ring_density_gen(La, Lab, Lba, Ra, Rab, Rba) -> Dd[k,b,j,c], De[k,b,j,c]
  // Notice: pack L_perm[i, a, j, b] -> [ov, ov], R_perm[i, a, k, c] -> [ov, ov]:
  // Then A[j, b, k, c] = L_perm^T @ R_perm! Reindex to [k, b, j, c]!
  auto compute_ring_density = [&](const double* La, const double* Lab, const double* Lba,
                                  const double* Ra, const double* Rab, const double* Rba,
                                  std::vector<double>& Dd, std::vector<double>& De) {
    std::vector<double> pLa(n2), pLab(n2), pLba(n2), pRa(n2), pRab(n2), pRba(n2);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b) {
            const std::size_t src = ((i * o + j) * v + a) * v + b;
            const std::size_t dst = (i * v + a) * ov + (j * v + b);
            pLa[dst] = La[src];
            pLab[dst] = Lab[src];
            pLba[dst] = Lba[src];
            pRa[dst] = Ra[src];
            pRab[dst] = Rab[src];
            pRba[dst] = Rba[src];
          }
    std::vector<double> A1(n2, 0.0), A2(n2, 0.0), A3(n2, 0.0), A4(n2, 0.0), A5(n2, 0.0);
    cpu_gemm('T', 'N', ov, ov, ov, pLa.data(), pRa.data(), A1.data(), 1.0, 0.0);
    cpu_gemm('T', 'N', ov, ov, ov, pLa.data(), pRab.data(), A2.data(), 1.0, 0.0);
    cpu_gemm('T', 'N', ov, ov, ov, pLab.data(), pRa.data(), A3.data(), 1.0, 0.0);
    cpu_gemm('T', 'N', ov, ov, ov, pLab.data(), pRab.data(), A4.data(), 1.0, 0.0);
    cpu_gemm('T', 'N', ov, ov, ov, pLba.data(), pRba.data(), A5.data(), 1.0, 0.0);
    Dd.assign(n2, 0.0);
    De.assign(n2, 0.0);
    // A_m[(j*v + b)*ov + (k*v + c)] -> D[(k*v + b)*ov + (j*v + c)]
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t b = 0; b < v; ++b)
        for (std::size_t k = 0; k < o; ++k)
          for (std::size_t c = 0; c < v; ++c) {
            const std::size_t s = (j * v + b) * ov + (k * v + c);
            const std::size_t d = (k * v + b) * ov + (j * v + c);
            Dd[d] = 2.0 * (A1[s] + A4[s] + A5[s]);
            De[d] = -2.0 * (A1[s] + A2[s] + A3[s] + A4[s]);
          }
  };

  std::vector<double> Dd, De;
  compute_ring_density(t2a.data(), t2, t2_ba.data(), t2a.data(), t2, t2_ba.data(), Dd, De);
  out.t9_v10 = -2.0 * (dot_n(Dd.data(), V10_d, n2) + dot_n(De.data(), V10_e, n2));
  out.t22_b7 = -(dot_n(Dd.data(), B7_d, n2) + dot_n(De.data(), B7_e, n2));

  compute_ring_density(la_j.data(), lab_j.data(), lba_j.data(), t2a.data(), t2, t2_ba.data(), Dd,
                       De);
  out.t17_fo_x7 = dot_n(Dd.data(), X7_d, n2) + dot_n(De.data(), X7_e, n2);

  compute_ring_density(t2a.data(), t2, t2_ba.data(), la_j.data(), lab_j.data(), lba_j.data(), Dd,
                       De);
  out.t18_x7_fo = dot_n(Dd.data(), X7_d, n2) + dot_n(De.data(), X7_e, n2);

  compute_ring_density(la_i.data(), lab_i.data(), lba_i.data(), t2a.data(), t2, t2_ba.data(), Dd,
                       De);
  out.t20_fo_x7_ex = dot_n(Dd.data(), X7_d, n2) + dot_n(De.data(), X7_e, n2);

  compute_ring_density(la_a.data(), lab_a.data(), lba_a.data(), t2a.data(), t2, t2_ba.data(), Dd,
                       De);
  out.t24_fv_x7 = -(dot_n(Dd.data(), X7_d, n2) + dot_n(De.data(), X7_e, n2));

  out.t5_z2 = t5_z2;
  out.a_sum = out.t1_h3_x2 + out.t2_h3_x6 + out.t3_xh_h7 + out.t14_xh_h9 + out.t4_x2_h7 +
              out.t15_x3_h9 + out.t5_z2 + out.t6_t1_v9 + out.t7_t1_v7 + out.t8_v8 + out.t9_v10 +
              out.t16_bh4 + out.t17_fo_x7 + out.t18_x7_fo + out.t19_fo_x4 + out.t20_fo_x7_ex +
              out.t21_b4 + out.t22_b7 + out.t23_fv_x4 + out.t24_fv_x7;
}

// Compute Z2 scalar t5 = 2 * sum_{i,j,a,b} Z2_sp[i,j,a,b] * t2t[i,j,a,b]
// from dense spatial r_hh[ext,ext,o,o], r_hp[ext,ext,o,v], g_ex[o,ext,ext,v], g_co[o,ext,ext,v].
double z2_cabs_scalar_from_dense(std::size_t o, std::size_t v, std::size_t ext,
                                 const std::vector<double>& r_hh, const std::vector<double>& r_hp,
                                 const std::vector<double>& g_ex, const std::vector<double>& g_co,
                                 const std::vector<double>& t2t, bool include_exch,
                                 bool include_coul) {
  const std::size_t oo = checked_mul(o, o);
  const std::size_t ov = checked_mul(o, v);
  const std::size_t ee = checked_mul(ext, ext);
  const std::size_t n2 = checked_mul(oo, checked_mul(v, v));

  std::vector<double> r_hh_s(checked_mul(ee, oo)), r_hh_a(checked_mul(ee, oo));
  std::vector<double> r_hp_s(checked_mul(ee, ov)), r_hp_a(checked_mul(ee, ov));
  for (std::size_t al = 0; al < ext; ++al)
    for (std::size_t ga = 0; ga < ext; ++ga) {
      const std::size_t fg = (al * ext + ga) * oo;
      const std::size_t sg = (ga * ext + al) * oo;
      for (std::size_t p = 0; p < oo; ++p) {
        r_hh_s[fg + p] = r_hh[sg + p];
        r_hh_a[fg + p] = r_hh[fg + p] - r_hh[sg + p];
      }
      const std::size_t fp = (al * ext + ga) * ov;
      const std::size_t sp = (ga * ext + al) * ov;
      for (std::size_t p = 0; p < ov; ++p) {
        r_hp_s[fp + p] = r_hp[sp + p];
        r_hp_a[fp + p] = r_hp[fp + p] - r_hp[sp + p];
      }
    }

  std::vector<double> g_ex_minus_co(g_co.size(), 0.0);
  for (std::size_t i = 0; i < g_co.size(); ++i)
    g_ex_minus_co[i] = (include_exch ? g_ex[i] : 0.0) - (include_coul ? g_co[i] : 0.0);

  // m1_pair(rL[ext,ext,o,o], g4[o,ext,ext,v], rR[ext,ext,o,v]) -> out[o,o,v,v]
  auto m1_pair = [&](const std::vector<double>& rL, const std::vector<double>& g4,
                     const std::vector<double>& rR, double sign, std::vector<double>& acc) {
    std::vector<double> g_kb(ee), tmp(checked_mul(ee, oo)), rR_k(checked_mul(ee, v)),
        blk(checked_mul(oo, v));
    for (std::size_t k = 0; k < o; ++k) {
      for (std::size_t be = 0; be < ext; ++be)
        for (std::size_t ga = 0; ga < ext; ++ga)
          for (std::size_t a = 0; a < v; ++a)
            rR_k[(be * ext + ga) * v + a] = rR[((be * ext + ga) * o + k) * v + a];
      for (std::size_t b = 0; b < v; ++b) {
        // g_kb[be, al] = g4[k, al, be, b]
        for (std::size_t al = 0; al < ext; ++al)
          for (std::size_t be = 0; be < ext; ++be)
            g_kb[be * ext + al] = g4[((k * ext + al) * ext + be) * v + b];
        // tmp[be, ga*oo] = g_kb[be, al] @ rL[al, ga*oo]
        cpu_gemm('N', 'N', ext, checked_mul(ext, oo), ext, g_kb.data(), rL.data(), tmp.data(), 1.0,
                 0.0);
        // blk[oo, v] = tmp[ee, oo]^T @ rR_k[ee, v]
        cpu_gemm('T', 'N', oo, v, ee, tmp.data(), rR_k.data(), blk.data(), 1.0, 0.0);
        for (std::size_t ij = 0; ij < oo; ++ij)
          for (std::size_t a = 0; a < v; ++a) acc[(ij * v + a) * v + b] += sign * blk[ij * v + a];
      }
    }
  };

  // m2_pair(rL[ext,ext,o,o], g4[o,ext,ext,v], rR[ext,ext,o,v]) -> out[o,o,v,v]
  auto m2_pair = [&](const std::vector<double>& rL, const std::vector<double>& g4,
                     const std::vector<double>& rR, double sign, std::vector<double>& acc) {
    const std::size_t eo = checked_mul(ext, o);
    const std::size_t ev = checked_mul(ext, v);
    // Pack L[al*o + i, ga*o + k] and R[ga*o + k, be*v + a]
    std::vector<double> L_mat(checked_mul(eo, eo)), R_mat(checked_mul(eo, ev));
    for (std::size_t al = 0; al < ext; ++al)
      for (std::size_t ga = 0; ga < ext; ++ga)
        for (std::size_t i = 0; i < o; ++i)
          for (std::size_t k = 0; k < o; ++k)
            L_mat[(al * o + i) * eo + (ga * o + k)] = rL[((al * ext + ga) * o + i) * o + k];
    for (std::size_t be = 0; be < ext; ++be)
      for (std::size_t ga = 0; ga < ext; ++ga)
        for (std::size_t k = 0; k < o; ++k)
          for (std::size_t a = 0; a < v; ++a)
            R_mat[(ga * o + k) * ev + (be * v + a)] = rR[((be * ext + ga) * o + k) * v + a];
    std::vector<double> S(checked_mul(eo, ev), 0.0);
    cpu_gemm('N', 'N', eo, ev, eo, L_mat.data(), R_mat.data(), S.data(), 1.0, 0.0);
    // Reorder S to S2[i*v + a, al*ext + be] and G to G2[al*ext + be, j*v + b]
    std::vector<double> S2(checked_mul(ov, ee)), G2(checked_mul(ee, ov));
    for (std::size_t al = 0; al < ext; ++al)
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t be = 0; be < ext; ++be)
          for (std::size_t a = 0; a < v; ++a)
            S2[(i * v + a) * ee + (al * ext + be)] = S[(al * o + i) * ev + (be * v + a)];
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t al = 0; al < ext; ++al)
        for (std::size_t be = 0; be < ext; ++be)
          for (std::size_t b = 0; b < v; ++b)
            G2[(al * ext + be) * ov + (j * v + b)] = g4[((j * ext + al) * ext + be) * v + b];
    std::vector<double> M(checked_mul(ov, ov), 0.0);
    cpu_gemm('N', 'N', ov, ov, ee, S2.data(), G2.data(), M.data(), 1.0, 0.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t j = 0; j < o; ++j)
          for (std::size_t b = 0; b < v; ++b)
            acc[((i * o + j) * v + a) * v + b] += sign * M[(i * v + a) * ov + (j * v + b)];
  };

  std::vector<double> M1(n2, 0.0), M2d(n2, 0.0), M2e(n2, 0.0);
  if (include_coul) m1_pair(r_hh, g_co, r_hp_s, +1.0, M1);
  if (include_exch) m1_pair(r_hh_s, g_ex, r_hp_a, -1.0, M1);
  if (include_exch || include_coul) m1_pair(r_hh_s, g_ex_minus_co, r_hp, -1.0, M1);

  if (include_coul) {
    m2_pair(r_hh_a, g_co, r_hp_a, -1.0, M2d);
    m2_pair(r_hh, g_co, r_hp_s, +1.0, M2d);
  }
  if (include_exch || include_coul) m2_pair(r_hh_s, g_ex_minus_co, r_hp, -1.0, M2d);

  if (include_exch) {
    m2_pair(r_hh_a, g_ex, r_hp_s, -1.0, M2e);
    m2_pair(r_hh, g_ex, r_hp_a, +1.0, M2e);
  }

  double scalar = 0.0;
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const std::size_t ijab = ((i * o + j) * v + a) * v + b;
          const std::size_t jiba = ((j * o + i) * v + b) * v + a;
          const std::size_t jiab = ((j * o + i) * v + a) * v + b;
          const std::size_t ijba = ((i * o + j) * v + b) * v + a;
          const double z_val = -(M1[ijab] + M1[jiba]) +
                               (M2d[ijab] + M2d[jiba] - M2e[jiab] - M2e[ijba]);
          scalar += z_val * t2t[ijab];
        }
  return 2.0 * scalar;
}

// Compute exact spatial f^2 * g 1*1 Coulomb completion scalar t5_ffg_11:
// Uses bare ff[obs,obs,obs,obs] and v_sp[o,v,obs,obs] = -(k b | p q).
double z2_ffg_11_scalar_from_obs(std::size_t o_full, std::size_t frozen, std::size_t obs,
                                 const std::vector<double>& ff_obs,
                                 const std::vector<double>& v_sp, const std::vector<double>& t2t) {
  const std::size_t o = o_full - frozen;
  const std::size_t v = obs - o_full;
  const std::size_t oo = checked_mul(o, o);
  const std::size_t vv = checked_mul(v, v);
  const std::size_t n2 = checked_mul(oo, vv);
  const std::size_t obs2 = checked_mul(obs, obs);

  // Dressed ff_d[p,q,r,s] = (5/32)*ff[p,q,r,s] + (3/32)*ff[p,q,s,r]
  // ff_s[p,q,r,s] = ff_d[q,p,r,s], ff_a[p,q,r,s] = ff_d[p,q,r,s] - ff_d[q,p,r,s]
  // ff_M2 = ff_a - ff_s = ff_d[p,q,r,s] - 2*ff_d[q,p,r,s]
  auto ff_d_at = [&](std::size_t p, std::size_t q, std::size_t r, std::size_t s) -> double {
    return (5.0 / 32.0) * ff_obs[((p * obs + q) * obs + r) * obs + s] +
           (3.0 / 32.0) * ff_obs[((p * obs + q) * obs + s) * obs + r];
  };
  auto ff_s_at = [&](std::size_t p, std::size_t q, std::size_t r, std::size_t s) -> double {
    return ff_d_at(q, p, r, s);
  };
  auto ff_m2_at = [&](std::size_t p, std::size_t q, std::size_t r, std::size_t s) -> double {
    return ff_d_at(p, q, r, s) - 2.0 * ff_d_at(q, p, r, s);
  };
  auto v_at = [&](std::size_t k, std::size_t b, std::size_t p, std::size_t q) -> double {
    return v_sp[(k * v + b) * obs2 + p * obs + q];
  };

  std::vector<double> M1(n2, 0.0), M2(n2, 0.0);
  // Precompute contracted M2 helper: ff_m2_kk[p, a] = sum_k ff_m2(p, f+k, f+k, o_full+a)
  std::vector<double> ff_m2_kk(checked_mul(obs, v), 0.0);
  for (std::size_t p = 0; p < obs; ++p)
    for (std::size_t a = 0; a < v; ++a) {
      double s = 0.0;
      for (std::size_t k = 0; k < o; ++k) s += ff_m2_at(p, frozen + k, frozen + k, o_full + a);
      ff_m2_kk[p * v + a] = s;
    }

  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          double m1 = 0.0, m2 = 0.0;
          const std::size_t fi = frozen + i, fj = frozen + j, va = o_full + a;
          // M2 term 1: sum_p v[j,b,fi,p] * ff_m2_kk[p,a]
          for (std::size_t p = 0; p < obs; ++p) m2 += v_at(j, b, fi, p) * ff_m2_kk[p * v + a];
          for (std::size_t k = 0; k < o; ++k) {
            const std::size_t fk = frozen + k;
            for (std::size_t p = 0; p < obs; ++p) {
              m1 += v_at(k, b, fi, p) * ff_s_at(p, fj, fk, va) +
                    v_at(k, b, fj, p) * ff_s_at(fi, p, fk, va) +
                    v_at(k, b, p, fk) * ff_s_at(fi, fj, p, va) +
                    v_at(k, b, p, va) * ff_s_at(fi, fj, fk, p);
              m2 += v_at(j, b, fk, p) * ff_m2_at(fi, p, fk, va) +
                    v_at(j, b, p, fk) * ff_m2_at(fi, fk, p, va) +
                    v_at(j, b, p, va) * ff_m2_at(fi, fk, fk, p);
            }
          }
          const std::size_t idx = ((i * o + j) * v + a) * v + b;
          M1[idx] = -0.5 * m1;
          M2[idx] = 0.5 * m2;
        }

  double scalar = 0.0;
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const std::size_t ijab = ((i * o + j) * v + a) * v + b;
          const std::size_t jiba = ((j * o + i) * v + b) * v + a;
          const double z = -(M1[ijab] + M1[jiba]) + (M2[ijab] + M2[jiba]);
          scalar += z * t2t[ijab];
        }
  return 2.0 * scalar;
}

// Memory-bounded DF evaluator for the exact OBS 1*1 ff*g completion (zero obs^4 storage).
// Mathematically identical to z2_ffg_11_scalar_from_obs to machine precision (<1e-14 Eh),
// reducing memory on obs=353 from 124 GiB to <2 GiB.
double z2_ffg_11_df_cpu(std::size_t o_full, std::size_t frozen, std::size_t obs, std::size_t ext,
                        std::size_t q, const double* fit_ext_obs, const double* r_ff_ext_obs,
                        const double* b_g_ext_obs, double s2, const std::vector<double>& t2t) {
  const std::size_t o = o_full - frozen;
  const std::size_t v = obs - o_full;
  const std::size_t oo = checked_mul(o, o);
  const std::size_t ov = checked_mul(o, v);
  const std::size_t vv = checked_mul(v, v);
  const std::size_t n2 = checked_mul(oo, vv);
  const std::size_t obsv = checked_mul(obs, v);

  // Symmetrized weight W[i, j, a, b] = t2t[i, j, a, b] + t2t[j, i, b, a].
  // Total scalar = sum_{i,j,a,b} W[i,j,a,b] * (m1[i,j,a,b] + m2[i,j,a,b]).
  std::vector<double> W(n2), W_ja_ib(n2), W_ia_jb(n2);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const std::size_t idx = ((i * o + j) * v + a) * v + b;
          const std::size_t sym = ((j * o + i) * v + b) * v + a;
          const double wv = t2t[idx] + t2t[sym];
          W[idx] = wv;
          W_ja_ib[(j * v + a) * ov + (i * v + b)] = wv;
          W_ia_jb[(i * v + a) * ov + (j * v + b)] = wv;
        }

  // Compact OBS 3-index slices (avoiding any obs^4 tensor).
  std::vector<double> F_oo(checked_mul(oo, q)), R_oo(checked_mul(oo, q));
  std::vector<double> F_ov(checked_mul(ov, q)), R_ov(checked_mul(ov, q));
  std::vector<double> B_ov(checked_mul(ov, q));
  std::vector<double> F_oobs_t(checked_mul(checked_mul(o, obs), q)),
      R_oobs_t(checked_mul(checked_mul(o, obs), q)), B_oobs_q(checked_mul(q, checked_mul(o, obs)));
  std::vector<double> F_obso_q(checked_mul(q, checked_mul(obs, o))),
      R_obso_q(checked_mul(q, checked_mul(obs, o))), B_obso_q(checked_mul(q, checked_mul(obs, o)));
  std::vector<double> F_obsv_q(checked_mul(q, obsv)), R_obsv_q(checked_mul(q, obsv)),
      B_obsv_q(checked_mul(q, obsv));

  for (std::size_t Q = 0; Q < q; ++Q) {
    const double* fQ = fit_ext_obs + Q * ext * obs;
    const double* rQ = r_ff_ext_obs + Q * ext * obs;
    const double* bQ = b_g_ext_obs + Q * ext * obs;
    for (std::size_t i = 0; i < o; ++i) {
      const std::size_t fi = frozen + i;
      for (std::size_t k = 0; k < o; ++k) {
        const std::size_t fk = frozen + k;
        F_oo[(i * o + k) * q + Q] = fQ[fi * obs + fk];
        R_oo[(i * o + k) * q + Q] = rQ[fi * obs + fk];
      }
      for (std::size_t a = 0; a < v; ++a) {
        const std::size_t va = o_full + a;
        F_ov[(i * v + a) * q + Q] = fQ[fi * obs + va];
        R_ov[(i * v + a) * q + Q] = rQ[fi * obs + va];
        B_ov[(i * v + a) * q + Q] = bQ[fi * obs + va];
      }
      for (std::size_t p = 0; p < obs; ++p) {
        F_oobs_t[(i * obs + p) * q + Q] = fQ[fi * obs + p];
        R_oobs_t[(i * obs + p) * q + Q] = rQ[fi * obs + p];
        B_oobs_q[Q * (o * obs) + i * obs + p] = bQ[fi * obs + p];
        F_obso_q[Q * (o * obs) + i * obs + p] = fQ[p * obs + fi];
        R_obso_q[Q * (o * obs) + i * obs + p] = rQ[p * obs + fi];
        B_obso_q[Q * (o * obs) + i * obs + p] = bQ[p * obs + fi];
      }
    }
    for (std::size_t p = 0; p < obs; ++p)
      for (std::size_t a = 0; a < v; ++a) {
        const std::size_t va = o_full + a;
        F_obsv_q[Q * obsv + p * v + a] = fQ[p * obs + va];
        R_obsv_q[Q * obsv + p * v + a] = rQ[p * obs + va];
        B_obsv_q[Q * obsv + p * v + a] = bQ[p * obs + va];
      }
  }

  double scalar = 0.0;

  // Scratch buffers for the k-loop.
  std::vector<double> F_jk(checked_mul(o, q)), R_jk(checked_mul(o, q));
  std::vector<double> F_pfk(checked_mul(q, obs)), R_pfk(checked_mul(q, obs));
  std::vector<double> B_pfk(checked_mul(q, obs)), B_fkp(checked_mul(q, obs));
  std::vector<double> A_raw(checked_mul(o, obsv)), A_ovp(checked_mul(ov, obs)),
      B_ovp(checked_mul(ov, obs));
  std::vector<double> E_raw(checked_mul(checked_mul(o, obs), v)), E_ovp(checked_mul(ov, obs)),
      G_ovp(checked_mul(ov, obs));
  std::vector<double> v_ko_raw(checked_mul(v, checked_mul(o, obs))), v_ko_ibp(checked_mul(ov, obs));
  std::vector<double> v_k_jbp(checked_mul(ov, obs)), v_k_jbp2(checked_mul(ov, obs));
  std::vector<double> T_ovp(checked_mul(ov, obs)), T2_ovp(checked_mul(ov, obs));
  std::vector<double> ff_m2_kk(checked_mul(obs, v), 0.0);
  std::vector<double> F_kk(checked_mul(o, obs), 0.0);
  std::vector<double> V3(checked_mul(v, obs), 0.0);

  std::vector<double> H1(checked_mul(oo, obs)), H2(checked_mul(oo, obs)),
      F4(checked_mul(oo, obs));
  std::vector<double> WF4(checked_mul(vv, obs)), v_kv(checked_mul(v, obsv));
  std::vector<double> F_fkp(checked_mul(q, obs)), R_fkp(checked_mul(q, obs));
  std::vector<double> F_fka(checked_mul(q, v)), R_fka(checked_mul(q, v));

  for (std::size_t k = 0; k < o; ++k) {
    // Gather k-slices.
    for (std::size_t j = 0; j < o; ++j) {
      std::copy_n(F_oo.data() + (j * o + k) * q, q, F_jk.data() + j * q);
      std::copy_n(R_oo.data() + (j * o + k) * q, q, R_jk.data() + j * q);
    }
    for (std::size_t Q = 0; Q < q; ++Q) {
      std::copy_n(F_obso_q.data() + Q * (o * obs) + k * obs, obs, F_pfk.data() + Q * obs);
      std::copy_n(R_obso_q.data() + Q * (o * obs) + k * obs, obs, R_pfk.data() + Q * obs);
      std::copy_n(B_obso_q.data() + Q * (o * obs) + k * obs, obs, B_pfk.data() + Q * obs);
      std::copy_n(B_oobs_q.data() + Q * (o * obs) + k * obs, obs, B_fkp.data() + Q * obs);
      for (std::size_t p = 0; p < obs; ++p) {
        F_fkp[Q * obs + p] = F_oobs_t[(k * obs + p) * q + Q];
        R_fkp[Q * obs + p] = R_oobs_t[(k * obs + p) * q + Q];
      }
      for (std::size_t a = 0; a < v; ++a) {
        F_fka[Q * v + a] = F_ov[(k * v + a) * q + Q];
        R_fka[Q * v + a] = R_ov[(k * v + a) * q + Q];
      }
    }

    // V3[b, p] += v_sp(k, b, p, fk) = -sum_Q B_ov[k, b, Q] * B_pfk[Q, p]
    cpu_gemm('N', 'N', v, obs, q, B_ov.data() + k * v * q, B_pfk.data(), V3.data(), -1.0, 1.0);

    // A[j, a, p] = ff_phys(fj, p, fk, va) = s2*(F_jk[j,Q]*R_obsv[Q,p,a] + R_jk*F_obsv)
    cpu_gemm('N', 'N', o, obsv, q, F_jk.data(), R_obsv_q.data(), A_raw.data(), s2, 0.0);
    cpu_gemm('N', 'N', o, obsv, q, R_jk.data(), F_obsv_q.data(), A_raw.data(), s2, 1.0);
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t p = 0; p < obs; ++p)
        for (std::size_t a = 0; a < v; ++a)
          A_ovp[(j * v + a) * obs + p] = A_raw[j * obsv + p * v + a];

    // B[j, a, p] = ff_phys(fj, p, va, fk) = s2*(F_ov[j,a,Q]*R_pfk[Q,p] + R_ov*F_pfk)
    cpu_gemm('N', 'N', ov, obs, q, F_ov.data(), R_pfk.data(), B_ovp.data(), s2, 0.0);
    cpu_gemm('N', 'N', ov, obs, q, R_ov.data(), F_pfk.data(), B_ovp.data(), s2, 1.0);

    // Accumulate ff_m2_kk[p, a] += ff_m2(p, fk, fk, va).
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t p = 0; p < obs; ++p) {
        const double av = A_ovp[(k * v + a) * obs + p];
        const double bv = B_ovp[(k * v + a) * obs + p];
        const double ffd = (5.0 / 32.0) * bv + (3.0 / 32.0) * av;
        const double ffs = (5.0 / 32.0) * av + (3.0 / 32.0) * bv;
        ff_m2_kk[p * v + a] += ffd - 2.0 * ffs;
      }

    // M1 Term 1 & Term 2: v_ko[b, i, p] = v_sp(k, b, fi, p) = -B_ov[k,b,Q] @ B_oobs_q[Q, i, p]
    cpu_gemm('N', 'N', v, checked_mul(o, obs), q, B_ov.data() + k * v * q, B_oobs_q.data(),
             v_ko_raw.data(), -1.0, 0.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t b = 0; b < v; ++b)
        std::copy_n(v_ko_raw.data() + (b * o + i) * obs, obs,
                    v_ko_ibp.data() + (i * v + b) * obs);

    // M1 Term 1: T[j,a,p] = W_ja_ib @ v_ko_ibp; dot with ff_s(p, fj, fk, va) = 5/32*A + 3/32*B
    cpu_gemm('N', 'N', ov, obs, ov, W_ja_ib.data(), v_ko_ibp.data(), T_ovp.data(), 1.0, 0.0);
    // M1 Term 2: T2[i,a,p] = W_ia_jb @ v_ko_ibp; dot with ff_s(fi, p, fk, va) = 5/32*B + 3/32*A
    cpu_gemm('N', 'N', ov, obs, ov, W_ia_jb.data(), v_ko_ibp.data(), T2_ovp.data(), 1.0, 0.0);
    for (std::size_t idx = 0; idx < ov * obs; ++idx) {
      const double av = A_ovp[idx], bv = B_ovp[idx];
      scalar += T_ovp[idx] * ((5.0 / 32.0) * av + (3.0 / 32.0) * bv);
      scalar += T2_ovp[idx] * ((5.0 / 32.0) * bv + (3.0 / 32.0) * av);
    }

    // M2 Term 2: v_k_jbp[j, b, p] = v_sp(j, b, fk, p) = -B_ov[j,b,Q] @ B_fkp[Q, p]
    cpu_gemm('N', 'N', ov, obs, q, B_ov.data(), B_fkp.data(), v_k_jbp.data(), -1.0, 0.0);
    cpu_gemm('N', 'N', ov, obs, ov, W_ia_jb.data(), v_k_jbp.data(), T_ovp.data(), 1.0, 0.0);
    for (std::size_t idx = 0; idx < ov * obs; ++idx) {
      const double av = A_ovp[idx], bv = B_ovp[idx];
      const double ffd = (5.0 / 32.0) * av + (3.0 / 32.0) * bv;
      const double ffs = (5.0 / 32.0) * bv + (3.0 / 32.0) * av;
      scalar += T_ovp[idx] * (ffd - 2.0 * ffs);
    }

    // M2 Term 3: v_k_jbp2[j, b, p] = v_sp(j, b, p, fk) = -B_ov[j,b,Q] @ B_pfk[Q, p]
    cpu_gemm('N', 'N', ov, obs, q, B_ov.data(), B_pfk.data(), v_k_jbp2.data(), -1.0, 0.0);
    cpu_gemm('N', 'N', ov, obs, ov, W_ia_jb.data(), v_k_jbp2.data(), T2_ovp.data(), 1.0, 0.0);
    // E[i,a,p] = ff_phys(fi, fk, p, va) = s2*(F_oobs_t[i,p,Q]*R_fka[Q,a] + R_oobs_t*F_fka)
    cpu_gemm('N', 'N', checked_mul(o, obs), v, q, F_oobs_t.data(), R_fka.data(), E_raw.data(), s2,
             0.0);
    cpu_gemm('N', 'N', checked_mul(o, obs), v, q, R_oobs_t.data(), F_fka.data(), E_raw.data(), s2,
             1.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t p = 0; p < obs; ++p)
        for (std::size_t a = 0; a < v; ++a)
          E_ovp[(i * v + a) * obs + p] = E_raw[(i * obs + p) * v + a];
    // G[i,a,p] = ff_phys(fi, fk, va, p) = s2*(F_ov[i,a,Q]*R_fkp[Q,p] + R_ov*F_fkp)
    cpu_gemm('N', 'N', ov, obs, q, F_ov.data(), R_fkp.data(), G_ovp.data(), s2, 0.0);
    cpu_gemm('N', 'N', ov, obs, q, R_ov.data(), F_fkp.data(), G_ovp.data(), s2, 1.0);
    for (std::size_t idx = 0; idx < ov * obs; ++idx) {
      const double ev = E_ovp[idx], gv = G_ovp[idx];
      const double ffd = (5.0 / 32.0) * ev + (3.0 / 32.0) * gv;
      const double ffs = (5.0 / 32.0) * gv + (3.0 / 32.0) * ev;
      scalar += T2_ovp[idx] * (ffd - 2.0 * ffs);
    }

    // M1 Term 4 & accumulate F_kk[i, p] for M2 Term 4:
    // H1_raw[j, i, p] = ff_phys(fj, fi, fk, p) = s2*(F_jk[j,Q]*R_oobs_q[Q,i,p] + R_jk*F_oobs_q)
    // Build F_oobs_q_k[q, o*obs] inline? Reuse F_oobs_t[i*obs+p, q] via transB='T'!
    cpu_gemm('N', 'T', o, checked_mul(o, obs), q, F_jk.data(), R_oobs_t.data(), H1.data(), s2, 0.0);
    cpu_gemm('N', 'T', o, checked_mul(o, obs), q, R_jk.data(), F_oobs_t.data(), H1.data(), s2, 1.0);
    // H2_raw[j, p, i] -> H2[j, i, p] = ff_phys(fj, fi, p, fk) = H1[i, j, p]!
    // Check: ff_phys(fj, fi, p, fk) = s2*(F[fj,p]*R[fi,fk] + R[fj,p]*F[fi,fk]) = H1[i, j, p]!
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t p = 0; p < obs; ++p) {
          const double h1 = H1[(j * o + i) * obs + p];
          const double h2 = H1[(i * o + j) * obs + p];
          F4[(i * o + j) * obs + p] = (5.0 / 32.0) * h1 + (3.0 / 32.0) * h2;
          if (j == k) {
            const double ffd = (5.0 / 32.0) * h2 + (3.0 / 32.0) * h1;
            const double ffs = (5.0 / 32.0) * h1 + (3.0 / 32.0) * h2;
            F_kk[i * obs + p] += ffd - 2.0 * ffs;
          }
        }
    // WF4[a*v+b, p] = W[oo, vv]^T @ F4[oo, obs]
    cpu_gemm('T', 'N', vv, obs, oo, W.data(), F4.data(), WF4.data(), 1.0, 0.0);
    // v_kv[b, p, a] = v_sp(k, b, p, va) = -B_ov[k, b, Q] @ B_obsv_q[Q, p, a]
    cpu_gemm('N', 'N', v, obsv, q, B_ov.data() + k * v * q, B_obsv_q.data(), v_kv.data(), -1.0,
             0.0);
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t b = 0; b < v; ++b)
        for (std::size_t p = 0; p < obs; ++p)
          scalar += WF4[(a * v + b) * obs + p] * v_kv[b * obsv + p * v + a];
  }

  // M1 Term 3 (i-loop): sum_{i,j,a,b,p} W[i,j,a,b] * V3[b, p] * ff_s(fi, fj, p, va)
  // M2 Term 1 (i-loop): sum_{i,j,a,b,p} W[i,j,a,b] * v_sp(j, b, fi, p) * ff_m2_kk[p, a]
  std::vector<double> W_i_jab(checked_mul(ov, v)), WV3(checked_mul(ov, obs));
  std::vector<double> P1_raw(checked_mul(checked_mul(o, obs), v)), P1_ovp(checked_mul(ov, obs)),
      P2_ovp(checked_mul(ov, obs));
  std::vector<double> F_fia(checked_mul(q, v)), R_fia(checked_mul(q, v));
  std::vector<double> F_fip(checked_mul(q, obs)), R_fip(checked_mul(q, obs)),
      B_fip(checked_mul(q, obs));
  std::vector<double> T_m2_1(checked_mul(v, obs));
  for (std::size_t i = 0; i < o; ++i) {
    for (std::size_t Q = 0; Q < q; ++Q) {
      for (std::size_t a = 0; a < v; ++a) {
        F_fia[Q * v + a] = F_ov[(i * v + a) * q + Q];
        R_fia[Q * v + a] = R_ov[(i * v + a) * q + Q];
      }
      for (std::size_t p = 0; p < obs; ++p) {
        F_fip[Q * obs + p] = F_oobs_t[(i * obs + p) * q + Q];
        R_fip[Q * obs + p] = R_oobs_t[(i * obs + p) * q + Q];
        B_fip[Q * obs + p] = B_oobs_q[Q * (o * obs) + i * obs + p];
      }
    }
    // M1 Term 3:
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b)
          W_i_jab[(j * v + a) * v + b] = W[((i * o + j) * v + a) * v + b];
    cpu_gemm('N', 'N', ov, obs, v, W_i_jab.data(), V3.data(), WV3.data(), 1.0, 0.0);
    // P1[j,a,p] = ff_phys(fj, fi, p, va) = s2*(F_oobs_t[j,p,Q]*R_fia[Q,a] + R_oobs_t*F_fia)
    cpu_gemm('N', 'N', checked_mul(o, obs), v, q, F_oobs_t.data(), R_fia.data(), P1_raw.data(), s2,
             0.0);
    cpu_gemm('N', 'N', checked_mul(o, obs), v, q, R_oobs_t.data(), F_fia.data(), P1_raw.data(), s2,
             1.0);
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t p = 0; p < obs; ++p)
        for (std::size_t a = 0; a < v; ++a)
          P1_ovp[(j * v + a) * obs + p] = P1_raw[(j * obs + p) * v + a];
    // P2[j,a,p] = ff_phys(fj, fi, va, p) = s2*(F_ov[j,a,Q]*R_fip[Q,p] + R_ov*F_fip)
    cpu_gemm('N', 'N', ov, obs, q, F_ov.data(), R_fip.data(), P2_ovp.data(), s2, 0.0);
    cpu_gemm('N', 'N', ov, obs, q, R_ov.data(), F_fip.data(), P2_ovp.data(), s2, 1.0);
    for (std::size_t idx = 0; idx < ov * obs; ++idx)
      scalar += WV3[idx] * ((5.0 / 32.0) * P1_ovp[idx] + (3.0 / 32.0) * P2_ovp[idx]);

    // M2 Term 1: v_fi[j*v+b, p] = v_sp(j, b, fi, p) = -B_ov[j,b,Q] @ B_fip[Q, p]
    cpu_gemm('N', 'N', ov, obs, q, B_ov.data(), B_fip.data(), v_k_jbp.data(), -1.0, 0.0);
    cpu_gemm('N', 'N', v, obs, ov, W_ia_jb.data() + i * v * ov, v_k_jbp.data(), T_m2_1.data(), 1.0,
             0.0);
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t p = 0; p < obs; ++p) scalar += T_m2_1[a * obs + p] * ff_m2_kk[p * v + a];
  }

  // M2 Term 4 (j-loop): sum_{i,j,a,b,p} W[i,j,a,b] * F_kk[i, p] * v_sp(j, b, p, va)
  std::vector<double> W_j_i_ab(checked_mul(o, vv));
  for (std::size_t j = 0; j < o; ++j) {
    for (std::size_t i = 0; i < o; ++i)
      std::copy_n(W.data() + (i * o + j) * vv, vv, W_j_i_ab.data() + i * vv);
    cpu_gemm('T', 'N', vv, obs, o, W_j_i_ab.data(), F_kk.data(), WF4.data(), 1.0, 0.0);
    cpu_gemm('N', 'N', v, obsv, q, B_ov.data() + j * v * q, B_obsv_q.data(), v_kv.data(), -1.0,
             0.0);
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t b = 0; b < v; ++b)
        for (std::size_t p = 0; p < obs; ++p)
          scalar += WF4[(a * v + b) * obs + p] * v_kv[b * obsv + p * v + a];
  }

  return scalar;
}

}  // namespace detail

OwnedC2Factors build_c2_factors_cpu(std::size_t o_full, std::size_t frozen, std::size_t obs,
                                    std::size_t ext, std::size_t q_f12, const double* fit_ext_obs,
                                    const double* r_f_ext_obs, const double* f_ext,
                                    double geminal_scale) {
  OwnedC2Factors out;
  out.q_f12 = q_f12;
  out.geminal_scale = geminal_scale;
  if (!q_f12) return out;
  if (frozen >= o_full || o_full >= obs || obs >= ext)
    throw std::invalid_argument("invalid dimensions in build_c2_factors_cpu");
  const std::size_t o = o_full - frozen;
  const std::size_t v = obs - o_full;
  const std::size_t cabs = ext - obs;
  const std::size_t qov = checked_mul(q_f12, checked_mul(o, v));
  const std::size_t qvv = checked_mul(q_f12, checked_mul(v, v));
  out.f_vo.assign(qov, 0.0);
  out.r_vo.assign(qov, 0.0);
  out.f_vv.assign(qvv, 0.0);
  out.r_vv.assign(qvv, 0.0);
  out.af_vo.assign(qov, 0.0);
  out.ar_vo.assign(qov, 0.0);
  out.af_vv.assign(qvv, 0.0);
  out.ar_vv.assign(qvv, 0.0);

  // Extract F_vx[v, cabs] from f_ext[o_full..obs-1, obs..ext-1]
  std::vector<double> F_vx(checked_mul(v, cabs));
  for (std::size_t a = 0; a < v; ++a)
    for (std::size_t x = 0; x < cabs; ++x) F_vx[a * cabs + x] = f_ext[(o_full + a) * ext + obs + x];

  std::vector<double> x_o(checked_mul(cabs, o)), x_v(checked_mul(cabs, v));
  std::vector<double> tmp_vo(checked_mul(v, o)), tmp_vv(checked_mul(v, v));
  for (std::size_t Q = 0; Q < q_f12; ++Q) {
    const double* fQ = fit_ext_obs + Q * ext * obs;
    const double* rQ = r_f_ext_obs + Q * ext * obs;
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t a = 0; a < v; ++a) {
        out.f_vo[(Q * o + i) * v + a] = fQ[(o_full + a) * obs + (frozen + i)];
        out.r_vo[(Q * o + i) * v + a] = rQ[(o_full + a) * obs + (frozen + i)];
      }
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t b = 0; b < v; ++b) {
        out.f_vv[(Q * v + a) * v + b] = fQ[(o_full + a) * obs + (o_full + b)];
        out.r_vv[(Q * v + a) * v + b] = rQ[(o_full + a) * obs + (o_full + b)];
      }
    // af_vo[Q, i, a] = sum_x F_vx[a, x] * fQ[obs + x, frozen + i]
    for (std::size_t x = 0; x < cabs; ++x) {
      for (std::size_t i = 0; i < o; ++i) x_o[x * o + i] = fQ[(obs + x) * obs + (frozen + i)];
      for (std::size_t b = 0; b < v; ++b) x_v[x * v + b] = fQ[(obs + x) * obs + (o_full + b)];
    }
    cpu_gemm('N', 'N', v, o, cabs, F_vx.data(), x_o.data(), tmp_vo.data(), 1.0, 0.0);
    cpu_gemm('N', 'N', v, v, cabs, F_vx.data(), x_v.data(), tmp_vv.data(), 1.0, 0.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t a = 0; a < v; ++a) out.af_vo[(Q * o + i) * v + a] = tmp_vo[a * o + i];
    std::copy(tmp_vv.begin(), tmp_vv.end(), out.af_vv.begin() + Q * v * v);

    for (std::size_t x = 0; x < cabs; ++x) {
      for (std::size_t i = 0; i < o; ++i) x_o[x * o + i] = rQ[(obs + x) * obs + (frozen + i)];
      for (std::size_t b = 0; b < v; ++b) x_v[x * v + b] = rQ[(obs + x) * obs + (o_full + b)];
    }
    cpu_gemm('N', 'N', v, o, cabs, F_vx.data(), x_o.data(), tmp_vo.data(), 1.0, 0.0);
    cpu_gemm('N', 'N', v, v, cabs, F_vx.data(), x_v.data(), tmp_vv.data(), 1.0, 0.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t a = 0; a < v; ++a) out.ar_vo[(Q * o + i) * v + a] = tmp_vo[a * o + i];
    std::copy(tmp_vv.begin(), tmp_vv.end(), out.ar_vv.begin() + Q * v * v);
  }
  return out;
}

double evaluate_explicit_tf12_pt_cpu(const PtInputs& in, double* seconds_out) {
  const auto started = Clock::now();
  const std::size_t o = in.o, v = in.v, q = in.q_mp2;
  if (!o || !v || (!q && !in.ovvv_direct))
    throw std::invalid_argument("empty explicit (T)-F12 pt dimensions");
  const std::size_t ov = checked_mul(o, v), vv = checked_mul(v, v), v3 = checked_mul(v, vv);
  const std::size_t oo = checked_mul(o, o);
  if (in.ovvv_direct) {
    check_finite_span(in.ovvv_direct, checked_mul(o, v3), "ovvv_direct");
  } else {
    check_finite_span(in.bov, checked_mul(q, ov), "bov");
    check_finite_span(in.bvv, checked_mul(q, vv), "bvv");
  }
  check_finite_span(in.ovoo, checked_mul(ov, oo), "ovoo");
  check_finite_span(in.ovov, checked_mul(ov, ov), "ovov");
  check_finite_span(in.fov, ov, "fov");
  check_finite_span(in.t1, ov, "t1");
  check_finite_span(in.t2, checked_mul(oo, vv), "t2");
  check_finite_span(in.eps_o, o, "eps_o");
  check_finite_span(in.eps_v, v, "eps_v");

  const double max_o = *std::max_element(in.eps_o, in.eps_o + o);
  const double min_v = *std::min_element(in.eps_v, in.eps_v + v);
  if (!(max_o < min_v) || 3.0 * (min_v - max_o) <= in.denominator_threshold)
    throw std::invalid_argument("noncanonical or near-zero explicit (T)-F12 denominator");

  // Build all o panels on-the-fly into a compact 3-panel LRU cache (zero O*V^3 residency).
  std::array<std::vector<double>, 3> panel_bufs;
  for (auto& buf : panel_bufs) buf.resize(v3);
  std::array<std::size_t, 3> panel_ids{std::numeric_limits<std::size_t>::max(),
                                       std::numeric_limits<std::size_t>::max(),
                                       std::numeric_limits<std::size_t>::max()};
  std::array<std::size_t, 3> panel_ages{0, 0, 0};
  std::size_t epoch = 0;

  std::vector<double> bov_i(checked_mul(q, v));
  std::vector<double> c2_row_q(checked_mul(in.c2.q_f12, v));
  std::vector<double> c2_tmp(in.c2.q_f12 ? v3 : 0);

  auto get_panel = [&](std::size_t occ) -> const double* {
    for (std::size_t s = 0; s < 3; ++s) {
      if (panel_ids[s] == occ) {
        panel_ages[s] = ++epoch;
        return panel_bufs[s].data();
      }
    }
    const std::size_t slot =
        static_cast<std::size_t>(std::min_element(panel_ages.begin(), panel_ages.end()) -
                                 panel_ages.begin());
    double* dst = panel_bufs[slot].data();
    if (in.ovvv_direct) {
      std::copy_n(in.ovvv_direct + occ * v3, v3, dst);
    } else {
      for (std::size_t Q = 0; Q < q; ++Q)
        std::copy_n(in.bov + (Q * o + occ) * v, v, bov_i.data() + Q * v);
      // panel[a, b*v + f] = sum_Q bov_i[Q, a] * bvv[Q, b*v + f]
      cpu_gemm('T', 'N', v, vv, q, bov_i.data(), in.bvv, dst, 1.0, 0.0);
    }

    if (in.c2.c2_ab_direct != nullptr) {
      const double scale = in.c2.geminal_scale;
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b)
          for (std::size_t f = 0; f < v; ++f)
            dst[(a * v + b) * v + f] +=
                scale * in.c2.c2_ab_direct[((a * v + b) * o + occ) * v + f];
    } else if (in.c2.q_f12 > 0) {
      const std::size_t qf = in.c2.q_f12;
      const double s38 = 0.375 * in.c2.geminal_scale;
      const double s18 = 0.125 * in.c2.geminal_scale;
      auto extract_row = [&](const double* src) {
        for (std::size_t Q = 0; Q < qf; ++Q)
          std::copy_n(src + (Q * o + occ) * v, v, c2_row_q.data() + Q * v);
      };
      // Direct (3/8) terms directly into dst[a, b, f]:
      extract_row(in.c2.af_vo);
      cpu_gemm('T', 'N', v, vv, qf, c2_row_q.data(), in.c2.r_vv, dst, s38, 1.0);
      extract_row(in.c2.ar_vo);
      cpu_gemm('T', 'N', v, vv, qf, c2_row_q.data(), in.c2.f_vv, dst, s38, 1.0);
      extract_row(in.c2.f_vo);
      cpu_gemm('T', 'N', v, vv, qf, c2_row_q.data(), in.c2.ar_vv, dst, s38, 1.0);
      extract_row(in.c2.r_vo);
      cpu_gemm('T', 'N', v, vv, qf, c2_row_q.data(), in.c2.af_vv, dst, s38, 1.0);
      // Exchange (1/8) terms into c2_tmp[b, a, f], then transpose-add into dst[a, b, f]:
      extract_row(in.c2.r_vo);
      cpu_gemm('T', 'N', v, vv, qf, c2_row_q.data(), in.c2.af_vv, c2_tmp.data(), s18, 0.0);
      extract_row(in.c2.f_vo);
      cpu_gemm('T', 'N', v, vv, qf, c2_row_q.data(), in.c2.ar_vv, c2_tmp.data(), s18, 1.0);
      extract_row(in.c2.ar_vo);
      cpu_gemm('T', 'N', v, vv, qf, c2_row_q.data(), in.c2.f_vv, c2_tmp.data(), s18, 1.0);
      extract_row(in.c2.af_vo);
      cpu_gemm('T', 'N', v, vv, qf, c2_row_q.data(), in.c2.r_vv, c2_tmp.data(), s18, 1.0);
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b)
          for (std::size_t f = 0; f < v; ++f)
            dst[(a * v + b) * v + f] += c2_tmp[(b * v + a) * v + f];
    }
    panel_ids[slot] = occ;
    panel_ages[slot] = ++epoch;
    return dst;
  };

  std::array<std::vector<double>, 6> w_seeds;
  for (auto& w : w_seeds) w.resize(v3);
  std::vector<double> w_tot(v3);
  std::vector<double> ovoo_ij(checked_mul(v, o));
  std::vector<double> t2_mk(checked_mul(o, vv));

  auto build_seed = [&](std::size_t p0, std::size_t p1, std::size_t p2, double* dst) {
    const double* panel = get_panel(p0);
    const double* t2_kj = in.t2 + (p2 * o + p1) * vv;
    // W1[a, b, c] = sum_f panel[a, b, f] * t2[p2, p1, c, f]
    cpu_gemm('N', 'T', vv, v, v, panel, t2_kj, dst, 1.0, 0.0);
    // W2[a, b, c] = - sum_m ovoo[p0, a, p1, m] * t2[m, p2, b, c]
    for (std::size_t a = 0; a < v; ++a)
      std::copy_n(in.ovoo + ((p0 * v + a) * o + p1) * o, o, ovoo_ij.data() + a * o);
    for (std::size_t m = 0; m < o; ++m)
      std::copy_n(in.t2 + (m * o + p2) * vv, vv, t2_mk.data() + m * vv);
    cpu_gemm('N', 'N', v, vv, o, ovoo_ij.data(), t2_mk.data(), dst, -1.0, 1.0);
  };

  double total_energy = 0.0;
  for (std::size_t i = 0; i < o; ++i) {
    for (std::size_t j = i; j < o; ++j) {
      for (std::size_t k = j; k < o; ++k) {
        build_seed(i, j, k, w_seeds[0].data());
        build_seed(i, k, j, w_seeds[1].data());
        build_seed(j, i, k, w_seeds[2].data());
        build_seed(j, k, i, w_seeds[3].data());
        build_seed(k, i, j, w_seeds[4].data());
        build_seed(k, j, i, w_seeds[5].data());

        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b)
            for (std::size_t c = 0; c < v; ++c) {
              w_tot[(a * v + b) * v + c] =
                  w_seeds[0][(a * v + b) * v + c] + w_seeds[1][(a * v + c) * v + b] +
                  w_seeds[2][(b * v + a) * v + c] + w_seeds[3][(b * v + c) * v + a] +
                  w_seeds[4][(c * v + a) * v + b] + w_seeds[5][(c * v + b) * v + a];
            }

        const double e_ijk = in.eps_o[i] + in.eps_o[j] + in.eps_o[k];
        const double mult = (i == k) ? 1.0 : ((i == j || j == k) ? 3.0 : 6.0);
        const double* t2_ij = in.t2 + (i * o + j) * vv;
        const double* t2_ik = in.t2 + (i * o + k) * vv;
        const double* t2_jk = in.t2 + (j * o + k) * vv;
        double tile_sum = 0.0;
        for (std::size_t a = 0; a < v; ++a) {
          const double e_a = e_ijk - in.eps_v[a];
          const double t1_ia = in.t1[i * v + a];
          const double fov_ia = in.fov[i * v + a];
          for (std::size_t b = 0; b < v; ++b) {
            const double e_ab = e_a - in.eps_v[b];
            const double t1_jb = in.t1[j * v + b];
            const double fov_jb = in.fov[j * v + b];
            const double g_iajb = in.ovov[((i * v + a) * o + j) * v + b];
            const double t2_ijab = t2_ij[a * v + b];
            for (std::size_t c = 0; c < v; ++c) {
              const double w_abc = w_tot[(a * v + b) * v + c];
              const double z = 4.0 * w_abc + w_tot[(b * v + c) * v + a] +
                               w_tot[(c * v + a) * v + b] -
                               2.0 * (w_tot[(b * v + a) * v + c] + w_tot[(c * v + b) * v + a] +
                                      w_tot[(a * v + c) * v + b]);
              const double v_abc =
                  g_iajb * in.t1[k * v + c] +
                  in.ovov[((i * v + a) * o + k) * v + c] * t1_jb +
                  in.ovov[((j * v + b) * o + k) * v + c] * t1_ia +
                  t2_ijab * in.fov[k * v + c] + t2_ik[a * v + c] * fov_jb +
                  t2_jk[b * v + c] * fov_ia;
              tile_sum += z * (w_abc + v_abc) / (e_ab - in.eps_v[c]);
            }
          }
        }
        total_energy += (mult / 3.0) * tile_sum;
      }
    }
  }
  if (seconds_out) *seconds_out = elapsed_seconds(started);
  return total_energy;
}

Breakdown evaluate_explicit_tf12_a_cpu(const DfAInputs& in) {
  Breakdown out{};
  if (in.q_f12 == 0) return out;
  if (in.frozen >= in.o_full || in.o_full >= in.obs || in.obs >= in.ext)
    throw std::invalid_argument("invalid explicit (T)-F12 A-sector dimensions");
  const auto started = Clock::now();
  const std::size_t o_full = in.o_full, fz = in.frozen, obs = in.obs, ext = in.ext, q = in.q_f12;
  const std::size_t o = o_full - fz, v = obs - o_full;
  const std::size_t oo = checked_mul(o, o), ov = checked_mul(o, v), vv = checked_mul(v, v);
  const std::size_t o_obs = checked_mul(o, obs);
  const std::size_t n2 = checked_mul(oo, vv);
  const double s = in.geminal_scale;
  const double s2 = s * s;

  // Build spatial t2t[i,j,a,b] = 2*t2[i,j,a,b] - t2[i,j,b,a] for Z2.
  std::vector<double> t2t(n2);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const std::size_t idx = ((i * o + j) * v + a) * v + b;
          const std::size_t swp = ((i * o + j) * v + b) * v + a;
          t2t[idx] = 2.0 * in.t2[idx] - in.t2[swp];
        }

  // Helper: assemble robust DF operator matrix M[p1, q1, p2, q2] = sum_Q (F_L*R_R + R_L*F_R)
  // For V: V_a[i, p, k, b] = <i p | g Q f | k b>, V_b[i, p, b, k] = <i p | g Q f | b k>
  // Shape: [o*obs, o*v].
  std::vector<double> V_a(checked_mul(o_obs, ov), 0.0);
  std::vector<double> V_b(checked_mul(o_obs, ov), 0.0);
  std::vector<double> X_a(checked_mul(o_obs, ov), 0.0);
  std::vector<double> X_b(checked_mul(o_obs, ov), 0.0);
  std::vector<double> B_a(checked_mul(ov, ov), 0.0);
  std::vector<double> B_b(checked_mul(ov, ov), 0.0);

  // 1. Direct (1-part of Q) fg, ff, U, ff*(F+K) terms via 3-index DF!
  // For fg[i, p, k, b] = (i k | fg | p b) = sum_Q (F[Q, i, k] * R_fg[Q, p, b] + R_fg[Q, i, k] * F[Q, p, b])
  // Wait: contract Q for fixed (i, k) or pack (i, k) and (p, b) -> ONE GEMM of (o*o, q) @ (q, obs*v)!
  {
    std::vector<double> F_oo(checked_mul(oo, q)), Rfg_oo(checked_mul(oo, q)),
        Rff_oo(checked_mul(oo, q)), Ru_oo(checked_mul(oo, q));
    std::vector<double> F_obsv(checked_mul(q, checked_mul(obs, v))),
        Rfg_obsv(checked_mul(q, checked_mul(obs, v))),
        Rff_obsv(checked_mul(q, checked_mul(obs, v))),
        Ru_obsv(checked_mul(q, checked_mul(obs, v)));
    for (std::size_t Q = 0; Q < q; ++Q) {
      const double* fQ = in.fit_ext_obs + Q * ext * obs;
      const double* fgQ = in.r_fg_obs_obs + Q * obs * obs;
      const double* ffQ = in.r_ff_ext_obs + Q * ext * obs;
      const double* uQ = in.r_u_obs_obs + Q * obs * obs;
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t k = 0; k < o; ++k) {
          const std::size_t fi = fz + i, fk = fz + k;
          F_oo[(i * o + k) * q + Q] = fQ[fi * obs + fk];
          Rfg_oo[(i * o + k) * q + Q] = fgQ[fi * obs + fk];
          Rff_oo[(i * o + k) * q + Q] = ffQ[fi * obs + fk];
          Ru_oo[(i * o + k) * q + Q] = uQ[fi * obs + fk];
        }
      for (std::size_t p = 0; p < obs; ++p)
        for (std::size_t b = 0; b < v; ++b) {
          const std::size_t vb = o_full + b;
          F_obsv[Q * obs * v + p * v + b] = fQ[p * obs + vb];
          Rfg_obsv[Q * obs * v + p * v + b] = fgQ[p * obs + vb];
          Rff_obsv[Q * obs * v + p * v + b] = ffQ[p * obs + vb];
          Ru_obsv[Q * obs * v + p * v + b] = uQ[p * obs + vb];
        }
    }
    const std::size_t obsv = checked_mul(obs, v);
    std::vector<double> tmp(checked_mul(oo, obsv), 0.0);
    // V_a direct: (i k | fg | p b)
    cpu_gemm('N', 'N', oo, obsv, q, F_oo.data(), Rfg_obsv.data(), tmp.data(), 1.0, 0.0);
    cpu_gemm('N', 'N', oo, obsv, q, Rfg_oo.data(), F_obsv.data(), tmp.data(), 1.0, 1.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t p = 0; p < obs; ++p)
        for (std::size_t k = 0; k < o; ++k)
          for (std::size_t b = 0; b < v; ++b)
            V_a[(i * obs + p) * ov + (k * v + b)] = tmp[(i * o + k) * obsv + (p * v + b)];

    // X_a direct: (i k | ff | p b)
    cpu_gemm('N', 'N', oo, obsv, q, F_oo.data(), Rff_obsv.data(), tmp.data(), 1.0, 0.0);
    cpu_gemm('N', 'N', oo, obsv, q, Rff_oo.data(), F_obsv.data(), tmp.data(), 1.0, 1.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t p = 0; p < obs; ++p)
        for (std::size_t k = 0; k < o; ++k)
          for (std::size_t b = 0; b < v; ++b)
            X_a[(i * obs + p) * ov + (k * v + b)] = tmp[(i * o + k) * obsv + (p * v + b)];

    // B_a direct U: (i k | U | a b)
    cpu_gemm('N', 'N', oo, obsv, q, F_oo.data(), Ru_obsv.data(), tmp.data(), 1.0, 0.0);
    cpu_gemm('N', 'N', oo, obsv, q, Ru_oo.data(), F_obsv.data(), tmp.data(), 1.0, 1.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t k = 0; k < o; ++k)
          for (std::size_t b = 0; b < v; ++b)
            B_a[(i * v + a) * ov + (k * v + b)] = tmp[(i * o + k) * obsv + ((o_full + a) * v + b)];

    // Now swapped direct terms (i b | op | p k) for V_b, X_b, B_b:
    std::vector<double> F_ov(checked_mul(ov, q)), Rfg_ov(checked_mul(ov, q)),
        Rff_ov(checked_mul(ov, q)), Ru_ov(checked_mul(ov, q));
    std::vector<double> F_obso(checked_mul(q, o_obs)), Rfg_obso(checked_mul(q, o_obs)),
        Rff_obso(checked_mul(q, o_obs)), Ru_obso(checked_mul(q, o_obs));
    for (std::size_t Q = 0; Q < q; ++Q) {
      const double* fQ = in.fit_ext_obs + Q * ext * obs;
      const double* fgQ = in.r_fg_obs_obs + Q * obs * obs;
      const double* ffQ = in.r_ff_ext_obs + Q * ext * obs;
      const double* uQ = in.r_u_obs_obs + Q * obs * obs;
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t b = 0; b < v; ++b) {
          const std::size_t fi = fz + i, vb = o_full + b;
          F_ov[(i * v + b) * q + Q] = fQ[fi * obs + vb];
          Rfg_ov[(i * v + b) * q + Q] = fgQ[fi * obs + vb];
          Rff_ov[(i * v + b) * q + Q] = ffQ[fi * obs + vb];
          Ru_ov[(i * v + b) * q + Q] = uQ[fi * obs + vb];
        }
      for (std::size_t p = 0; p < obs; ++p)
        for (std::size_t k = 0; k < o; ++k) {
          const std::size_t fk = fz + k;
          F_obso[Q * o_obs + p * o + k] = fQ[p * obs + fk];
          Rfg_obso[Q * o_obs + p * o + k] = fgQ[p * obs + fk];
          Rff_obso[Q * o_obs + p * o + k] = ffQ[p * obs + fk];
          Ru_obso[Q * o_obs + p * o + k] = uQ[p * obs + fk];
        }
    }
    std::vector<double> tmp2(checked_mul(ov, o_obs), 0.0);
    cpu_gemm('N', 'N', ov, o_obs, q, F_ov.data(), Rfg_obso.data(), tmp2.data(), 1.0, 0.0);
    cpu_gemm('N', 'N', ov, o_obs, q, Rfg_ov.data(), F_obso.data(), tmp2.data(), 1.0, 1.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t p = 0; p < obs; ++p)
        for (std::size_t k = 0; k < o; ++k)
          for (std::size_t b = 0; b < v; ++b)
            V_b[(i * obs + p) * ov + (k * v + b)] = tmp2[(i * v + b) * o_obs + (p * o + k)];

    cpu_gemm('N', 'N', ov, o_obs, q, F_ov.data(), Rff_obso.data(), tmp2.data(), 1.0, 0.0);
    cpu_gemm('N', 'N', ov, o_obs, q, Rff_ov.data(), F_obso.data(), tmp2.data(), 1.0, 1.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t p = 0; p < obs; ++p)
        for (std::size_t k = 0; k < o; ++k)
          for (std::size_t b = 0; b < v; ++b)
            X_b[(i * obs + p) * ov + (k * v + b)] = tmp2[(i * v + b) * o_obs + (p * o + k)];

    cpu_gemm('N', 'N', ov, o_obs, q, F_ov.data(), Ru_obso.data(), tmp2.data(), 1.0, 0.0);
    cpu_gemm('N', 'N', ov, o_obs, q, Ru_ov.data(), F_obso.data(), tmp2.data(), 1.0, 1.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t k = 0; k < o; ++k)
          for (std::size_t b = 0; b < v; ++b)
            B_b[(i * v + a) * ov + (k * v + b)] = tmp2[(i * v + b) * o_obs + ((o_full + a) * o + k)];

    // Add symmetrized ff*(F+K) into B_a and B_b!
    // Precontract FpK[obs, ext] @ fit_ext_obs[Q, ext, obs] and @ r_ff_ext_obs[Q, ext, obs]:
    std::vector<double> FpK_oe(checked_mul(obs, ext));
    for (std::size_t p = 0; p < obs; ++p)
      for (std::size_t t = 0; t < ext; ++t) FpK_oe[p * ext + t] = in.f_ext[p * ext + t] + in.k_ext[p * ext + t];
    std::vector<double> FK_F(checked_mul(q, checked_mul(obs, obs))),
        FK_Rff(checked_mul(q, checked_mul(obs, obs)));
    for (std::size_t Q = 0; Q < q; ++Q) {
      cpu_gemm('N', 'N', obs, obs, ext, FpK_oe.data(), in.fit_ext_obs + Q * ext * obs,
               FK_F.data() + Q * obs * obs, 1.0, 0.0);
      cpu_gemm('N', 'N', obs, obs, ext, FpK_oe.data(), in.r_ff_ext_obs + Q * ext * obs,
               FK_Rff.data() + Q * obs * obs, 1.0, 0.0);
    }
    // Helper: ffbar_asym[p, q, r, s] = <p q | (FpK)_1 f^2 + (FpK)_2 f^2 | r s>
    //   = sum_Q [ FK_F[Q,p,r]*Rff[Q,q,s] + FK_Rff[Q,p,r]*F[Q,q,s] + F[Q,p,r]*FK_Rff[Q,q,s] + Rff[Q,p,r]*FK_F[Q,q,s] ]
    // Symmetrized: 0.5 * (ffbar_asym[p,q,r,s] + ffbar_asym[r,s,p,q]).
    std::vector<double> F_vv(checked_mul(q, vv)), Rff_vv(checked_mul(q, vv)),
        FKF_vv(checked_mul(q, vv)), FKR_vv(checked_mul(q, vv));
    std::vector<double> FKF_oo(checked_mul(oo, q)), FKR_oo(checked_mul(oo, q));
    std::vector<double> FKF_ov(checked_mul(ov, q)), FKR_ov(checked_mul(ov, q));
    std::vector<double> FKF_vo(checked_mul(q, ov)), FKR_vo(checked_mul(q, ov));
    std::vector<double> F_vo_q(checked_mul(q, ov)), Rff_vo_q(checked_mul(q, ov));
    for (std::size_t Q = 0; Q < q; ++Q) {
      const double* fQ = in.fit_ext_obs + Q * ext * obs;
      const double* rQ = in.r_ff_ext_obs + Q * ext * obs;
      const double* fkfQ = FK_F.data() + Q * obs * obs;
      const double* fkrQ = FK_Rff.data() + Q * obs * obs;
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t k = 0; k < o; ++k) {
          FKF_oo[(i * o + k) * q + Q] = fkfQ[(fz + i) * obs + (fz + k)];
          FKR_oo[(i * o + k) * q + Q] = fkrQ[(fz + i) * obs + (fz + k)];
        }
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          F_vv[Q * vv + a * v + b] = fQ[(o_full + a) * obs + (o_full + b)];
          Rff_vv[Q * vv + a * v + b] = rQ[(o_full + a) * obs + (o_full + b)];
          FKF_vv[Q * vv + a * v + b] = fkfQ[(o_full + a) * obs + (o_full + b)];
          FKR_vv[Q * vv + a * v + b] = fkrQ[(o_full + a) * obs + (o_full + b)];
        }
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t b = 0; b < v; ++b) {
          FKF_ov[(i * v + b) * q + Q] = fkfQ[(fz + i) * obs + (o_full + b)];
          FKR_ov[(i * v + b) * q + Q] = fkrQ[(fz + i) * obs + (o_full + b)];
        }
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t k = 0; k < o; ++k) {
          F_vo_q[Q * ov + a * o + k] = fQ[(o_full + a) * obs + (fz + k)];
          Rff_vo_q[Q * ov + a * o + k] = rQ[(o_full + a) * obs + (fz + k)];
          FKF_vo[Q * ov + a * o + k] = fkfQ[(o_full + a) * obs + (fz + k)];
          FKR_vo[Q * ov + a * o + k] = fkrQ[(o_full + a) * obs + (fz + k)];
        }
    }
    std::vector<double> fb_a(checked_mul(oo, vv), 0.0);
    cpu_gemm('N', 'N', oo, vv, q, FKF_oo.data(), Rff_vv.data(), fb_a.data(), 1.0, 0.0);
    cpu_gemm('N', 'N', oo, vv, q, FKR_oo.data(), F_vv.data(), fb_a.data(), 1.0, 1.0);
    cpu_gemm('N', 'N', oo, vv, q, F_oo.data(), FKR_vv.data(), fb_a.data(), 1.0, 1.0);
    cpu_gemm('N', 'N', oo, vv, q, Rff_oo.data(), FKF_vv.data(), fb_a.data(), 1.0, 1.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t k = 0; k < o; ++k)
          for (std::size_t b = 0; b < v; ++b) {
            const double v1 = fb_a[(i * o + k) * vv + (a * v + b)];
            const double v2 = fb_a[(k * o + i) * vv + (b * v + a)];
            B_a[(i * v + a) * ov + (k * v + b)] += 0.5 * (v1 + v2);
          }

    // For B_b[i, a, k, b] = <i a | B | b k> -> need both <i a | (FpK) f^2 | b k> and <b k | (FpK) f^2 | i a>!
    // Note: <b k | (FpK) f^2 | i a> = <k b | (FpK) f^2 | a i> (electron swap 1<->2!), which is just B_b[k, b, i, a]!
    std::vector<double> fb_b(checked_mul(ov, ov), 0.0);
    cpu_gemm('N', 'N', ov, ov, q, FKF_ov.data(), Rff_vo_q.data(), fb_b.data(), 1.0, 0.0);
    cpu_gemm('N', 'N', ov, ov, q, FKR_ov.data(), F_vo_q.data(), fb_b.data(), 1.0, 1.0);
    cpu_gemm('N', 'N', ov, ov, q, F_ov.data(), FKR_vo.data(), fb_b.data(), 1.0, 1.0);
    cpu_gemm('N', 'N', ov, ov, q, Rff_ov.data(), FKF_vo.data(), fb_b.data(), 1.0, 1.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t k = 0; k < o; ++k)
          for (std::size_t b = 0; b < v; ++b) {
            const double v1 = fb_b[(i * v + b) * ov + (a * o + k)];
            const double v2 = fb_b[(k * v + a) * ov + (b * o + i)];
            B_b[(i * v + a) * ov + (k * v + b)] += 0.5 * (v1 + v2);
          }
  }

  // 2. Subtract P12 = OBSxOBS + O_fullxX + XxO_full from V_a, V_b, X_a, X_b AND accumulate B_a, B_b!
  // Loop over s in [0, ext):
  //   Build G_s[i, p, r] = <r s | g | i p> (for r in ext, i in o_act, p in obs)
  //   Build F_s[i, p, r] = <r s | f | i p> (for r in ext, i in o_act, p in obs)
  //   AND swapped F_s_sw[k, b, r] = <s r | f | k b> = <r s | f | b k> (k in o_act, b in v)
  // Note: F_s[i, p, r] = sum_Q (F[Q, r, i] * R_f[Q, s, p] + R_f[Q, r, i] * F[Q, s, p])!
  {
    std::vector<double> F_eo(checked_mul(q, checked_mul(ext, o))),
        Rf_eo(checked_mul(q, checked_mul(ext, o))),
        Bg_eo(checked_mul(q, checked_mul(ext, o)));
    std::vector<double> F_ev(checked_mul(q, checked_mul(ext, v))),
        Rf_ev(checked_mul(q, checked_mul(ext, v)));
    for (std::size_t Q = 0; Q < q; ++Q) {
      const double* fQ = in.fit_ext_obs + Q * ext * obs;
      const double* rQ = in.r_f_ext_obs + Q * ext * obs;
      const double* bQ = in.b_g_ext_obs + Q * ext * obs;
      for (std::size_t r = 0; r < ext; ++r) {
        for (std::size_t i = 0; i < o; ++i) {
          F_eo[(Q * ext + r) * o + i] = fQ[r * obs + (fz + i)];
          Rf_eo[(Q * ext + r) * o + i] = rQ[r * obs + (fz + i)];
          Bg_eo[(Q * ext + r) * o + i] = bQ[r * obs + (fz + i)];
        }
        for (std::size_t b = 0; b < v; ++b) {
          F_ev[(Q * ext + r) * v + b] = fQ[r * obs + (o_full + b)];
          Rf_ev[(Q * ext + r) * v + b] = rQ[r * obs + (o_full + b)];
        }
      }
    }
    // Negated (K + F) matrix on [ext, ext]
    std::vector<double> neg_KF(checked_mul(ext, ext));
    for (std::size_t idx = 0; idx < ext * ext; ++idx) neg_KF[idx] = -(in.k_ext[idx] + in.f_ext[idx]);

    const std::size_t eo = checked_mul(ext, o), ev = checked_mul(ext, v);
    std::vector<double> f_sp(checked_mul(q, obs)), rf_sp(checked_mul(q, obs)),
        bg_sp(checked_mul(q, obs));
    std::vector<double> f_so(checked_mul(q, o)), rf_so(checked_mul(q, o));
    std::vector<double> g_mat(checked_mul(obs, eo)), f_mat(checked_mul(obs, eo)),
        f_sw_mat(checked_mul(o, ev));
    // Reordered buffers: [o*obs, ext], [o*v, ext]
    std::vector<double> G_ipr(checked_mul(o_obs, ext)), F_ipr(checked_mul(o_obs, ext));
    std::vector<double> F_kbr(checked_mul(ov, ext)), F_sw_kbr(checked_mul(ov, ext));
    std::vector<double> KF_L(checked_mul(ov, ext)), KF_R(checked_mul(ov, ext));
    std::vector<double> Q_L(checked_mul(ov, ext)), Q_R(checked_mul(ov, ext));
    std::vector<double> FQ_L(checked_mul(ov, ext)), FQ_R(checked_mul(ov, ext));

    for (std::size_t s_idx = 0; s_idx < ext; ++s_idx) {
      for (std::size_t Q = 0; Q < q; ++Q) {
        const double* fQ = in.fit_ext_obs + Q * ext * obs;
        const double* rQ = in.r_f_ext_obs + Q * ext * obs;
        const double* bQ = in.b_g_ext_obs + Q * ext * obs;
        std::copy_n(fQ + s_idx * obs, obs, f_sp.data() + Q * obs);
        std::copy_n(rQ + s_idx * obs, obs, rf_sp.data() + Q * obs);
        std::copy_n(bQ + s_idx * obs, obs, bg_sp.data() + Q * obs);
        for (std::size_t k = 0; k < o; ++k) {
          f_so[Q * o + k] = fQ[s_idx * obs + (fz + k)];
          rf_so[Q * o + k] = rQ[s_idx * obs + (fz + k)];
        }
      }
      // f_mat[p, r*o + i] = sum_Q (f_sp[Q,p]*Rf_eo[Q,r*o+i] + rf_sp[Q,p]*F_eo[Q,r*o+i])
      cpu_gemm('T', 'N', obs, eo, q, f_sp.data(), Rf_eo.data(), f_mat.data(), 1.0, 0.0);
      cpu_gemm('T', 'N', obs, eo, q, rf_sp.data(), F_eo.data(), f_mat.data(), 1.0, 1.0);
      // f_sw_mat[k, r*v + b] = <r s | f | b k> = sum_Q (f_so[Q,k]*Rf_ev[Q,r*v+b] + rf_so[Q,k]*F_ev[Q,r*v+b])
      cpu_gemm('T', 'N', o, ev, q, f_so.data(), Rf_ev.data(), f_sw_mat.data(), 1.0, 0.0);
      cpu_gemm('T', 'N', o, ev, q, rf_so.data(), F_ev.data(), f_sw_mat.data(), 1.0, 1.0);

      // Reorder into [o*obs, ext] and [o*v, ext]:
      for (std::size_t p = 0; p < obs; ++p)
        for (std::size_t r = 0; r < ext; ++r)
          for (std::size_t i = 0; i < o; ++i)
            F_ipr[(i * obs + p) * ext + r] = f_mat[p * eo + r * o + i];
      for (std::size_t k = 0; k < o; ++k)
        for (std::size_t b = 0; b < v; ++b)
          for (std::size_t r = 0; r < ext; ++r) {
            F_kbr[(k * v + b) * ext + r] = F_ipr[(k * obs + (o_full + b)) * ext + r];
            F_sw_kbr[(k * v + b) * ext + r] = f_sw_mat[k * ev + r * v + b];
          }

      // P12 projection for V and X:
      // Determine active r range in P12 for this s_idx:
      //   s_idx < o_full: all r in [0, ext) (O x OBS + O x X)
      //   o_full <= s_idx < obs: r in [0, obs) (V x OBS)
      //   s_idx >= obs: r in [0, o_full) (X x O)
      const std::size_t r_p12 = (s_idx < o_full) ? ext : ((s_idx < obs) ? obs : o_full);
      if (r_p12 > 0) {
        cpu_gemm('T', 'N', obs, eo, q, bg_sp.data(), Bg_eo.data(), g_mat.data(), 1.0, 0.0);
        // Pack truncated [o*obs, r_p12] and [o*v, r_p12] contiguous for fast GEMM
        std::vector<double> G_sub(checked_mul(o_obs, r_p12)), F_sub(checked_mul(o_obs, r_p12));
        std::vector<double> Fk_sub(checked_mul(ov, r_p12)), Fsw_sub(checked_mul(ov, r_p12));
        for (std::size_t p = 0; p < obs; ++p)
          for (std::size_t r = 0; r < r_p12; ++r)
            for (std::size_t i = 0; i < o; ++i) {
              G_sub[(i * obs + p) * r_p12 + r] = g_mat[p * eo + r * o + i];
              F_sub[(i * obs + p) * r_p12 + r] = F_ipr[(i * obs + p) * ext + r];
            }
        for (std::size_t kb = 0; kb < ov; ++kb) {
          std::copy_n(F_kbr.data() + kb * ext, r_p12, Fk_sub.data() + kb * r_p12);
          std::copy_n(F_sw_kbr.data() + kb * ext, r_p12, Fsw_sub.data() + kb * r_p12);
        }
        cpu_gemm('N', 'T', o_obs, ov, r_p12, G_sub.data(), Fk_sub.data(), V_a.data(), -1.0, 1.0);
        cpu_gemm('N', 'T', o_obs, ov, r_p12, G_sub.data(), Fsw_sub.data(), V_b.data(), -1.0, 1.0);
        cpu_gemm('N', 'T', o_obs, ov, r_p12, F_sub.data(), Fk_sub.data(), X_a.data(), -1.0, 1.0);
        cpu_gemm('N', 'T', o_obs, ov, r_p12, F_sub.data(), Fsw_sub.data(), X_b.data(), -1.0, 1.0);
      }

      // B_a and B_b: -<f | (K+F)_1 + (K+F)_2 | f> + <Qf | F_1 + F_2 | Qf>
      // Note: electron 1 has r (using F_kbr), electron 2 has r (using F_sw_kbr = f[s, r, k, b])!
      // Verify: sum_{r,r',s} f[r,s,i,a] M[r,r'] f[r',s,k,b]  = F_kbr @ M @ F_kbr^T
      //         sum_{r,r',s} f[s,r,i,a] M[r,r'] f[s,r',k,b]  = F_sw_kbr @ M @ F_sw_kbr^T
      //         sum_{r,r',s} f[r,s,i,a] M[r,r'] f[r',s,b,k]  = F_kbr @ M @ F_sw_kbr^T
      //         sum_{r,r',s} f[s,r,i,a] M[r,r'] f[s,r',b,k]  = F_sw_kbr @ M @ F_kbr^T
      cpu_gemm('N', 'N', ov, ext, ext, F_kbr.data(), neg_KF.data(), KF_L.data(), 1.0, 0.0);
      cpu_gemm('N', 'N', ov, ext, ext, F_sw_kbr.data(), neg_KF.data(), KF_R.data(), 1.0, 0.0);
      cpu_gemm('N', 'T', ov, ov, ext, KF_L.data(), F_kbr.data(), B_a.data(), 1.0, 1.0);
      cpu_gemm('N', 'T', ov, ov, ext, KF_R.data(), F_sw_kbr.data(), B_a.data(), 1.0, 1.0);
      cpu_gemm('N', 'T', ov, ov, ext, KF_L.data(), F_sw_kbr.data(), B_b.data(), 1.0, 1.0);
      cpu_gemm('N', 'T', ov, ov, ext, KF_R.data(), F_kbr.data(), B_b.data(), 1.0, 1.0);

      // Q12 mask on (r, s_idx): nonzero only if s_idx >= o_full AND (if s_idx < obs: r >= obs; else r >= o_full).
      if (s_idx >= o_full) {
        const std::size_t r_start = (s_idx < obs) ? obs : o_full;
        const std::size_t r_len = ext - r_start;
        // Extract F_sub_q[r_len, r_len] from in.f_ext[r_start..ext-1, r_start..ext-1]
        std::vector<double> F_qq(checked_mul(r_len, r_len));
        for (std::size_t r1 = 0; r1 < r_len; ++r1)
          std::copy_n(in.f_ext + (r_start + r1) * ext + r_start, r_len, F_qq.data() + r1 * r_len);
        std::vector<double> qL(checked_mul(ov, r_len)), qR(checked_mul(ov, r_len));
        std::vector<double> fqL(checked_mul(ov, r_len)), fqR(checked_mul(ov, r_len));
        for (std::size_t kb = 0; kb < ov; ++kb) {
          std::copy_n(F_kbr.data() + kb * ext + r_start, r_len, qL.data() + kb * r_len);
          std::copy_n(F_sw_kbr.data() + kb * ext + r_start, r_len, qR.data() + kb * r_len);
        }
        cpu_gemm('N', 'N', ov, r_len, r_len, qL.data(), F_qq.data(), fqL.data(), 1.0, 0.0);
        cpu_gemm('N', 'N', ov, r_len, r_len, qR.data(), F_qq.data(), fqR.data(), 1.0, 0.0);
        cpu_gemm('N', 'T', ov, ov, r_len, fqL.data(), qL.data(), B_a.data(), 1.0, 1.0);
        cpu_gemm('N', 'T', ov, ov, r_len, fqR.data(), qR.data(), B_a.data(), 1.0, 1.0);
        cpu_gemm('N', 'T', ov, ov, r_len, fqL.data(), qR.data(), B_b.data(), 1.0, 1.0);
        cpu_gemm('N', 'T', ov, ov, r_len, fqR.data(), qL.data(), B_b.data(), 1.0, 1.0);
      }
    }
  }

  // Extract dressed spatial blocks V7, V8, V9, V10_d, V10_e, X2, X4, X6, X7_d, X7_e, B4, B7_d, B7_e, Bh4!
  // Recall:
  //   V_sp[i, p, k, b] = s * (3/8 * V_a[i,p,k,b] + 1/8 * V_b[i,p,k,b])
  //   V_sp[p, i, k, b] = s * (3/8 * V_b[i,p,k,b] + 1/8 * V_a[i,p,k,b])
  //   X_sp[i, p, k, b] = s^2 * (5/32 * X_a + 3/32 * X_b)
  //   X_sp[p, i, k, b] = s^2 * (5/32 * X_b + 3/32 * X_a)
  //   B_sp[i, a, k, b] = s^2 * (5/32 * B_a + 3/32 * B_b)
  //   B_sp[a, i, k, b] = s^2 * (5/32 * B_b + 3/32 * B_a)
  std::vector<double> V7(ov, 0.0), V8(vv, 0.0), V9(checked_mul(oo, ov), 0.0),
      V10_d(n2, 0.0), V10_e(n2, 0.0);
  std::vector<double> X2(ov, 0.0), X4(vv, 0.0), X6(checked_mul(oo, ov), 0.0),
      X7_d(n2, 0.0), X7_e(n2, 0.0);
  std::vector<double> B4(vv, 0.0), B7_d(n2, 0.0), B7_e(n2, 0.0), Bh4(vv, 0.0);

  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t k = 0; k < o; ++k)
        for (std::size_t b = 0; b < v; ++b) {
          const std::size_t src = (i * obs + (fz + j)) * ov + (k * v + b);
          const double vd = s * (0.375 * V_a[src] + 0.125 * V_b[src]);
          const double ve = s * (0.375 * V_b[src] + 0.125 * V_a[src]);
          V9[((i * o + j) * o + k) * v + b] = vd;
          if (i == k) V7[j * v + b] += 2.0 * vd - ve;

          const double xd = s2 * ((5.0 / 32.0) * X_a[src] + (3.0 / 32.0) * X_b[src]);
          const double xe = s2 * ((5.0 / 32.0) * X_b[src] + (3.0 / 32.0) * X_a[src]);
          X6[((i * o + j) * o + k) * v + b] = xd;
          if (i == k) X2[j * v + b] += 2.0 * xd - xe;
        }

  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t k = 0; k < o; ++k)
        for (std::size_t b = 0; b < v; ++b) {
          const std::size_t src_obs = (i * obs + (o_full + a)) * ov + (k * v + b);
          const std::size_t src_ov = (i * v + a) * ov + (k * v + b);
          const double vd = s * (0.375 * V_a[src_obs] + 0.125 * V_b[src_obs]);
          const double ve = s * (0.375 * V_b[src_obs] + 0.125 * V_a[src_obs]);
          V10_d[src_ov] = vd;
          V10_e[src_ov] = ve;
          if (i == k) V8[a * v + b] += 2.0 * vd - ve;

          const double xd = s2 * ((5.0 / 32.0) * X_a[src_obs] + (3.0 / 32.0) * X_b[src_obs]);
          const double xe = s2 * ((5.0 / 32.0) * X_b[src_obs] + (3.0 / 32.0) * X_a[src_obs]);
          X7_d[src_ov] = xd;
          X7_e[src_ov] = xe;
          if (i == k) {
            const double x1 = 2.0 * xd - xe;
            X4[a * v + b] += x1;
            Bh4[a * v + b] -= in.eps_o[i] * x1;
          }

          const double bd = s2 * ((5.0 / 32.0) * B_a[src_ov] + (3.0 / 32.0) * B_b[src_ov]);
          const double be = s2 * ((5.0 / 32.0) * B_b[src_ov] + (3.0 / 32.0) * B_a[src_ov]);
          B7_d[src_ov] = bd;
          B7_e[src_ov] = be;
          if (i == k) B4[a * v + b] += 2.0 * bd - be;
        }

  out.seconds_a_vxbbh = elapsed_seconds(started);

  // 3. Evaluate Z2 (t5_z2) according to z2_mode!
  const auto z2_start = Clock::now();
  double t5_z2 = 0.0;
  if (in.z2_mode != Z2Mode::Zero) {
    if (!in.b_g_ext_ext)
      throw std::invalid_argument("explicit (T)-F12 Z2 requires b_g_ext_ext");
    const std::size_t ee = checked_mul(ext, ext);
    // Build spatial r_all_hh[ext,ext,o,o], r_all_hp[ext,ext,o,v] and Q-projected rQ_hh, rQ_hp!
    std::vector<double> r_hh(checked_mul(ee, oo), 0.0), r_hp(checked_mul(ee, ov), 0.0);
    std::vector<double> rQ_hh(checked_mul(ee, oo), 0.0), rQ_hp(checked_mul(ee, ov), 0.0);
    // Build raw f12[al, ga, fi, fj] = sum_Q (F[Q,al,fi]*Rf[Q,ga,fj] + Rf[Q,al,fi]*F[Q,ga,fj])
    // Pack F_eo[ext*o, q], Rf_eo[ext*o, q], F_ev[q, ext*v], Rf_ev[q, ext*v]
    const std::size_t eo = checked_mul(ext, o), ev = checked_mul(ext, v);
    std::vector<double> F_eo_t(checked_mul(eo, q)), Rf_eo_t(checked_mul(eo, q));
    std::vector<double> F_eo_q(checked_mul(q, eo)), Rf_eo_q(checked_mul(q, eo));
    std::vector<double> F_ev_q(checked_mul(q, ev)), Rf_ev_q(checked_mul(q, ev));
    for (std::size_t Q = 0; Q < q; ++Q) {
      const double* fQ = in.fit_ext_obs + Q * ext * obs;
      const double* rQ = in.r_f_ext_obs + Q * ext * obs;
      for (std::size_t al = 0; al < ext; ++al) {
        for (std::size_t i = 0; i < o; ++i) {
          const double fv = fQ[al * obs + (fz + i)];
          const double rv = rQ[al * obs + (fz + i)];
          F_eo_t[(al * o + i) * q + Q] = fv;
          Rf_eo_t[(al * o + i) * q + Q] = rv;
          F_eo_q[Q * eo + (al * o + i)] = fv;
          Rf_eo_q[Q * eo + (al * o + i)] = rv;
        }
        for (std::size_t a = 0; a < v; ++a) {
          F_ev_q[Q * ev + (al * v + a)] = fQ[al * obs + (o_full + a)];
          Rf_ev_q[Q * ev + (al * v + a)] = rQ[al * obs + (o_full + a)];
        }
      }
    }
    std::vector<double> f_hh_raw(checked_mul(eo, eo), 0.0);
    cpu_gemm('N', 'N', eo, eo, q, F_eo_t.data(), Rf_eo_q.data(), f_hh_raw.data(), 1.0, 0.0);
    cpu_gemm('N', 'N', eo, eo, q, Rf_eo_t.data(), F_eo_q.data(), f_hh_raw.data(), 1.0, 1.0);
    std::vector<double> f_hp_raw(checked_mul(eo, ev), 0.0);
    cpu_gemm('N', 'N', eo, ev, q, F_eo_t.data(), Rf_ev_q.data(), f_hp_raw.data(), 1.0, 0.0);
    cpu_gemm('N', 'N', eo, ev, q, Rf_eo_t.data(), F_ev_q.data(), f_hp_raw.data(), 1.0, 1.0);

    auto in_Q12 = [&](std::size_t al, std::size_t ga) -> bool {
      return (al >= o_full && ga >= o_full) && !(al < obs && ga < obs);
    };
    for (std::size_t al = 0; al < ext; ++al)
      for (std::size_t ga = 0; ga < ext; ++ga) {
        const bool qmask = in_Q12(al, ga);
        for (std::size_t i = 0; i < o; ++i)
          for (std::size_t j = 0; j < o; ++j) {
            const double d = f_hh_raw[(al * o + i) * eo + (ga * o + j)];
            const double x = f_hh_raw[(al * o + j) * eo + (ga * o + i)];
            const double val = s * (0.375 * d + 0.125 * x);
            const std::size_t dst = ((al * ext + ga) * o + i) * o + j;
            r_hh[dst] = val;
            if (qmask) rQ_hh[dst] = val;
          }
        for (std::size_t k = 0; k < o; ++k)
          for (std::size_t a = 0; a < v; ++a) {
            // f_hp_raw[(al*o+k)*ev + (ga*v+a)] = (al k | f | ga a) = <al ga | f | k a>
            // swapped ket <al ga | f | a k> = (al a | f | ga k) = (ga k | f | al a) = f_hp_raw[(ga*o+k)*ev + (al*v+a)]!
            const double d = f_hp_raw[(al * o + k) * ev + (ga * v + a)];
            const double x = f_hp_raw[(ga * o + k) * ev + (al * v + a)];
            const double val = s * (0.375 * d + 0.125 * x);
            const std::size_t dst = ((al * ext + ga) * o + k) * v + a;
            r_hp[dst] = val;
            if (qmask) rQ_hp[dst] = val;
          }
      }

    // Build g_ex[k, al, be, b] = (be k | al b) = sum_Q B[Q, be, k] * B[Q, al, b]
    //   and g_co[k, al, be, b] = (al be | k b) = sum_Q B[Q, al, be] * B[Q, k, b]
    std::vector<double> g_ex(checked_mul(ov, ee), 0.0), g_co(checked_mul(ov, ee), 0.0);
    {
      std::vector<double> B_ek_t(checked_mul(eo, q)), B_ab_q(checked_mul(q, ev));
      std::vector<double> B_kb_t(checked_mul(ov, q));
      for (std::size_t Q = 0; Q < q; ++Q) {
        const double* bQ = in.b_g_ext_obs + Q * ext * obs;
        for (std::size_t be = 0; be < ext; ++be)
          for (std::size_t k = 0; k < o; ++k) B_ek_t[(be * o + k) * q + Q] = bQ[be * obs + (fz + k)];
        for (std::size_t al = 0; al < ext; ++al)
          for (std::size_t b = 0; b < v; ++b)
            B_ab_q[Q * ev + (al * v + b)] = bQ[al * obs + (o_full + b)];
        for (std::size_t k = 0; k < o; ++k)
          for (std::size_t b = 0; b < v; ++b)
            B_kb_t[(k * v + b) * q + Q] = bQ[(fz + k) * obs + (o_full + b)];
      }
      std::vector<double> ex_tmp(checked_mul(eo, ev), 0.0);
      cpu_gemm('N', 'N', eo, ev, q, B_ek_t.data(), B_ab_q.data(), ex_tmp.data(), 1.0, 0.0);
      for (std::size_t k = 0; k < o; ++k)
        for (std::size_t al = 0; al < ext; ++al)
          for (std::size_t be = 0; be < ext; ++be)
            for (std::size_t b = 0; b < v; ++b)
              g_ex[((k * ext + al) * ext + be) * v + b] = ex_tmp[(be * o + k) * ev + (al * v + b)];

      std::vector<double> co_tmp(checked_mul(ov, ee), 0.0);
      cpu_gemm('N', 'N', ov, ee, q, B_kb_t.data(), in.b_g_ext_ext, co_tmp.data(), 1.0, 0.0);
      for (std::size_t k = 0; k < o; ++k)
        for (std::size_t b = 0; b < v; ++b)
          for (std::size_t al = 0; al < ext; ++al)
            for (std::size_t be = 0; be < ext; ++be)
              g_co[((k * ext + al) * ext + be) * v + b] = co_tmp[(k * v + b) * ee + (al * ext + be)];
    }

    t5_z2 = detail::z2_cabs_scalar_from_dense(o, v, ext, rQ_hh, rQ_hp, g_ex, g_co, t2t, true, true);
    if (in.z2_mode == Z2Mode::FfgCompleted) {
      const double t5_11_cabs =
          detail::z2_cabs_scalar_from_dense(o, v, ext, r_hh, r_hp, g_ex, g_co, t2t, false, true);
      const double t5_11_ffg = detail::z2_ffg_11_df_cpu(
          o_full, fz, obs, ext, q, in.fit_ext_obs, in.r_ff_ext_obs, in.b_g_ext_obs, s2, t2t);
      t5_z2 += (t5_11_ffg - t5_11_cabs);
    }
  }
  out.seconds_z2 = elapsed_seconds(z2_start);

  detail::evaluate_a_contractions_from_blocks(o, v, in.fov, in.t1, in.t2, in.eps_o, in.eps_v, in.ovoo,
                                      V7.data(), V8.data(), V9.data(), V10_d.data(), V10_e.data(),
                                      X2.data(), X4.data(), X6.data(), X7_d.data(), X7_e.data(),
                                      B4.data(), B7_d.data(), B7_e.data(), Bh4.data(), t5_z2, out);
  out.total_energy = out.a_sum;
  return out;
}

Breakdown evaluate_explicit_tf12_a_exact4_cpu(const Exact4AInputs& in) {
  Breakdown out{};
  if (in.frozen >= in.o_full || in.o_full >= in.obs || in.obs >= in.ext)
    throw std::invalid_argument("invalid explicit (T)-F12 exact4 dimensions");
  const auto started = Clock::now();
  const std::size_t o_full = in.o_full, fz = in.frozen, obs = in.obs, ext = in.ext;
  const std::size_t o = o_full - fz, v = obs - o_full;
  const std::size_t oo = checked_mul(o, o), ov = checked_mul(o, v), vv = checked_mul(v, v);
  const std::size_t n2 = checked_mul(oo, vv);
  const std::size_t obs2 = checked_mul(obs, obs), ee = checked_mul(ext, ext);
  const double s = in.geminal_scale, s2 = s * s;

  std::vector<double> t2t(n2);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const std::size_t idx = ((i * o + j) * v + a) * v + b;
          const std::size_t swp = ((i * o + j) * v + b) * v + a;
          t2t[idx] = 2.0 * in.t2[idx] - in.t2[swp];
        }

  auto in_Q12 = [&](std::size_t al, std::size_t ga) -> bool {
    return (al >= o_full && ga >= o_full) && !(al < obs && ga < obs);
  };
  auto f12_at = [&](std::size_t a, std::size_t b, std::size_t r, std::size_t s_i) -> double {
    return in.f12[((a * ext + b) * obs + r) * obs + s_i];
  };
  auto g_at = [&](std::size_t p, std::size_t q_i, std::size_t a, std::size_t b) -> double {
    return in.g_obs_obs_ext_ext[((p * obs + q_i) * ext + a) * ext + b];
  };

  // Build V_bare, X_bare, B_bare on [obs, obs, obs, obs] (small non-DF oracle systems like BH).
  std::vector<double> V_bare(in.fg, in.fg + checked_mul(obs2, obs2));
  std::vector<double> X_bare(in.ff, in.ff + checked_mul(obs2, obs2));
  std::vector<double> B_bare(checked_mul(obs2, obs2), 0.0);
  for (std::size_t p = 0; p < obs; ++p)
    for (std::size_t q_i = 0; q_i < obs; ++q_i)
      for (std::size_t r = 0; r < obs; ++r)
        for (std::size_t s_i = 0; s_i < obs; ++s_i) {
          const std::size_t i1 = ((p * obs + q_i) * obs + r) * obs + s_i;
          const std::size_t i2 = ((r * obs + s_i) * obs + p) * obs + q_i;
          B_bare[i1] = in.ftf[i1] + 0.5 * (in.ffbar[i1] + in.ffbar[i2]);
        }

  // Subtract P12 = OBSxOBS + O_fullxX + XxO_full from V_bare and X_bare via GEMM!
  std::vector<double> G_p12(checked_mul(obs2, ee), 0.0), F_p12(checked_mul(ee, obs2), 0.0);
  std::vector<double> F_full(checked_mul(ee, obs2), 0.0), FQ_full(checked_mul(ee, obs2), 0.0);
  for (std::size_t al = 0; al < ext; ++al)
    for (std::size_t ga = 0; ga < ext; ++ga) {
      const bool p12 = !in_Q12(al, ga);
      const std::size_t ag = al * ext + ga;
      for (std::size_t r = 0; r < obs; ++r)
        for (std::size_t s_i = 0; s_i < obs; ++s_i) {
          const double fv = f12_at(al, ga, r, s_i);
          F_full[ag * obs2 + r * obs + s_i] = fv;
          if (p12) {
            F_p12[ag * obs2 + r * obs + s_i] = fv;
            G_p12[(r * obs + s_i) * ee + ag] = g_at(r, s_i, al, ga);
          } else {
            FQ_full[ag * obs2 + r * obs + s_i] = fv;
          }
        }
    }
  cpu_gemm('N', 'N', obs2, obs2, ee, G_p12.data(), F_p12.data(), V_bare.data(), -1.0, 1.0);
  cpu_gemm('T', 'N', obs2, obs2, ee, F_p12.data(), F_p12.data(), X_bare.data(), -1.0, 1.0);

  // B_bare: -<f | (K+F)_1 + (K+F)_2 | f> + <Qf | F_1 + F_2 | Qf>
  std::vector<double> neg_KF(ee);
  for (std::size_t i = 0; i < ee; ++i) neg_KF[i] = -(in.k_ext[i] + in.f_ext[i]);
  // Apply 1-electron operator M on electron 1 and 2 of A[ext, ext, obs2]:
  auto add_one_body_sandwiched = [&](const std::vector<double>& A, const double* M) {
    // Electron 1: B += sum_{ga} (M @ A[:, ga, :])^T @ A[:, ga, :]
    std::vector<double> Ag(checked_mul(ext, obs2)), MAg(checked_mul(ext, obs2));
    for (std::size_t ga = 0; ga < ext; ++ga) {
      // Electron 1: A[al, ga, rs]
      for (std::size_t al = 0; al < ext; ++al)
        std::copy_n(A.data() + (al * ext + ga) * obs2, obs2, Ag.data() + al * obs2);
      cpu_gemm('N', 'N', ext, obs2, ext, M, Ag.data(), MAg.data(), 1.0, 0.0);
      cpu_gemm('T', 'N', obs2, obs2, ext, Ag.data(), MAg.data(), B_bare.data(), 1.0, 1.0);
      // Electron 2: A[ga, be, rs]
      const double* Abe = A.data() + ga * ext * obs2;
      cpu_gemm('N', 'N', ext, obs2, ext, M, Abe, MAg.data(), 1.0, 0.0);
      cpu_gemm('T', 'N', obs2, obs2, ext, Abe, MAg.data(), B_bare.data(), 1.0, 1.0);
    }
  };
  add_one_body_sandwiched(F_full, neg_KF.data());
  add_one_body_sandwiched(FQ_full, in.f_ext);

  // Dress and extract V, X, B, Bh blocks
  auto v_d = [&](std::size_t p, std::size_t q_i, std::size_t r, std::size_t s_i) {
    return s * (0.375 * V_bare[((p * obs + q_i) * obs + r) * obs + s_i] +
                0.125 * V_bare[((p * obs + q_i) * obs + s_i) * obs + r]);
  };
  auto x_d = [&](std::size_t p, std::size_t q_i, std::size_t r, std::size_t s_i) {
    return s2 * ((5.0 / 32.0) * X_bare[((p * obs + q_i) * obs + r) * obs + s_i] +
                 (3.0 / 32.0) * X_bare[((p * obs + q_i) * obs + s_i) * obs + r]);
  };
  auto b_d = [&](std::size_t p, std::size_t q_i, std::size_t r, std::size_t s_i) {
    return s2 * ((5.0 / 32.0) * B_bare[((p * obs + q_i) * obs + r) * obs + s_i] +
                 (3.0 / 32.0) * B_bare[((p * obs + q_i) * obs + s_i) * obs + r]);
  };

  std::vector<double> V7(ov, 0.0), V8(vv, 0.0), V9(checked_mul(oo, ov), 0.0),
      V10_d(n2, 0.0), V10_e(n2, 0.0);
  std::vector<double> X2(ov, 0.0), X4(vv, 0.0), X6(checked_mul(oo, ov), 0.0),
      X7_d(n2, 0.0), X7_e(n2, 0.0);
  std::vector<double> B4(vv, 0.0), B7_d(n2, 0.0), B7_e(n2, 0.0), Bh4(vv, 0.0);

  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t k = 0; k < o; ++k)
        for (std::size_t b = 0; b < v; ++b) {
          const std::size_t fi = fz + i, fj = fz + j, fk = fz + k, vb = o_full + b;
          const double vd = v_d(fi, fj, fk, vb), ve = v_d(fj, fi, fk, vb);
          V9[((i * o + j) * o + k) * v + b] = vd;
          if (i == k) V7[j * v + b] += 2.0 * vd - ve;

          const double xd = x_d(fi, fj, fk, vb), xe = x_d(fj, fi, fk, vb);
          X6[((i * o + j) * o + k) * v + b] = xd;
          if (i == k) X2[j * v + b] += 2.0 * xd - xe;
        }
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t k = 0; k < o; ++k)
        for (std::size_t b = 0; b < v; ++b) {
          const std::size_t fi = fz + i, va = o_full + a, fk = fz + k, vb = o_full + b;
          const std::size_t dst = (i * v + a) * ov + (k * v + b);
          const double vd = v_d(fi, va, fk, vb), ve = v_d(va, fi, fk, vb);
          V10_d[dst] = vd;
          V10_e[dst] = ve;
          if (i == k) V8[a * v + b] += 2.0 * vd - ve;

          const double xd = x_d(fi, va, fk, vb), xe = x_d(va, fi, fk, vb);
          X7_d[dst] = xd;
          X7_e[dst] = xe;
          if (i == k) {
            const double x1 = 2.0 * xd - xe;
            X4[a * v + b] += x1;
            Bh4[a * v + b] -= in.eps_o[i] * x1;
          }

          const double bd = b_d(fi, va, fk, vb), be = b_d(va, fi, fk, vb);
          B7_d[dst] = bd;
          B7_e[dst] = be;
          if (i == k) B4[a * v + b] += 2.0 * bd - be;
        }
  out.seconds_a_vxbbh = elapsed_seconds(started);

  const auto z2_start = Clock::now();
  double t5_z2 = 0.0;
  if (in.z2_mode != Z2Mode::Zero) {
    std::vector<double> r_hh(checked_mul(ee, oo), 0.0), r_hp(checked_mul(ee, ov), 0.0);
    std::vector<double> rQ_hh(checked_mul(ee, oo), 0.0), rQ_hp(checked_mul(ee, ov), 0.0);
    for (std::size_t al = 0; al < ext; ++al)
      for (std::size_t ga = 0; ga < ext; ++ga) {
        const bool qm = in_Q12(al, ga);
        for (std::size_t i = 0; i < o; ++i)
          for (std::size_t j = 0; j < o; ++j) {
            const double val = s * (0.375 * f12_at(al, ga, fz + i, fz + j) +
                                    0.125 * f12_at(al, ga, fz + j, fz + i));
            const std::size_t dst = ((al * ext + ga) * o + i) * o + j;
            r_hh[dst] = val;
            if (qm) rQ_hh[dst] = val;
          }
        for (std::size_t k = 0; k < o; ++k)
          for (std::size_t a = 0; a < v; ++a) {
            const double val = s * (0.375 * f12_at(al, ga, fz + k, o_full + a) +
                                    0.125 * f12_at(al, ga, o_full + a, fz + k));
            const std::size_t dst = ((al * ext + ga) * o + k) * v + a;
            r_hp[dst] = val;
            if (qm) rQ_hp[dst] = val;
          }
      }
    std::vector<double> g_ex(checked_mul(ov, ee)), g_co(checked_mul(ov, ee));
    for (std::size_t k = 0; k < o; ++k)
      for (std::size_t al = 0; al < ext; ++al)
        for (std::size_t be = 0; be < ext; ++be)
          for (std::size_t b = 0; b < v; ++b) {
            g_ex[((k * ext + al) * ext + be) * v + b] =
                in.g_chem_ext_o_ext_v[((be * o + k) * ext + al) * v + b];
            g_co[((k * ext + al) * ext + be) * v + b] =
                in.g_chem_ext_ext_o_v[((al * ext + be) * o + k) * v + b];
          }
    t5_z2 = detail::z2_cabs_scalar_from_dense(o, v, ext, rQ_hh, rQ_hp, g_ex, g_co, t2t, true, true);
    if (in.z2_mode == Z2Mode::FfgCompleted) {
      const double t5_11_cabs =
          detail::z2_cabs_scalar_from_dense(o, v, ext, r_hh, r_hp, g_ex, g_co, t2t, false, true);
      std::vector<double> ff_scaled(checked_mul(obs2, obs2));
      for (std::size_t i = 0; i < ff_scaled.size(); ++i) ff_scaled[i] = s2 * in.ff[i];
      std::vector<double> v_sp(checked_mul(ov, obs2));
      for (std::size_t k = 0; k < o; ++k)
        for (std::size_t b = 0; b < v; ++b)
          for (std::size_t p = 0; p < obs; ++p)
            for (std::size_t q_i = 0; q_i < obs; ++q_i)
              v_sp[(k * v + b) * obs2 + p * obs + q_i] =
                  -in.g_chem_ext_ext_o_v[((p * ext + q_i) * o + k) * v + b];
      const double t5_11_ffg = detail::z2_ffg_11_scalar_from_obs(o_full, fz, obs, ff_scaled, v_sp, t2t);
      t5_z2 += (t5_11_ffg - t5_11_cabs);
    }
  }
  out.seconds_z2 = elapsed_seconds(z2_start);

  detail::evaluate_a_contractions_from_blocks(o, v, in.fov, in.t1, in.t2, in.eps_o, in.eps_v, in.ovoo,
                                      V7.data(), V8.data(), V9.data(), V10_d.data(), V10_e.data(),
                                      X2.data(), X4.data(), X6.data(), X7_d.data(), X7_e.data(),
                                      B4.data(), B7_d.data(), B7_e.data(), Bh4.data(), t5_z2, out);
  out.total_energy = out.a_sum;
  return out;
}

Breakdown evaluate_explicit_tf12_df_cpu(const PtInputs& pt_in, const DfAInputs& a_in) {
  Breakdown out = evaluate_explicit_tf12_a_cpu(a_in);
  double sec_pt = 0.0;
  out.e_pt_with_c2 = evaluate_explicit_tf12_pt_cpu(pt_in, &sec_pt);
  out.seconds_pt = sec_pt;
  if (pt_in.c2.q_f12 == 0 && pt_in.c2.c2_ab_direct == nullptr) {
    out.e_pt_conv = out.e_pt_with_c2;
  } else {
    PtInputs conv = pt_in;
    conv.c2 = {};
    out.e_pt_conv = evaluate_explicit_tf12_pt_cpu(conv, nullptr);
  }
  out.total_energy = out.e_pt_with_c2 + out.a_sum;
  return out;
}

}  // namespace generativeqc::cc::explicit_tf12
