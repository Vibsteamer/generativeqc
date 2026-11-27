// AOT GPU-resident cuBLAS (and matching FP64 host reference) kernels for:
// 1. Ordinary DF-MP2 energy + unrelaxed response pullback (RHF spatial fast path + UHF dual-spin)
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#ifndef GQC_SINK_HOST_REFERENCE
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define GQC_HD __host__ __device__
#else
#define GQC_HD
#endif

namespace {

using Size = std::uint64_t;

constexpr int kErrBudgetExceeded = -3;
constexpr int kErrOom = -6;
constexpr int kErrInternal = -7;
constexpr Size kDefaultDeviceWorkspaceBytes = Size(512) << 20;

inline Size effective_device_budget(Size bytes) {
  return bytes ? bytes : kDefaultDeviceWorkspaceBytes;
}

inline Size sat_mul(Size a, Size b) {
  if (a == 0 || b == 0) return 0;
  if (a > std::numeric_limits<Size>::max() / b) return std::numeric_limits<Size>::max();
  return a * b;
}

inline Size sat_add(Size a, Size b) {
  if (a > std::numeric_limits<Size>::max() - b) return std::numeric_limits<Size>::max();
  return a + b;
}

inline Size bytes_of_doubles(Size count) { return sat_mul(count, sizeof(double)); }

bool all_finite_host(const double* ptr, Size n) {
  if (n == 0) return true;
  if (!ptr) return false;
  for (Size i = 0; i < n; ++i) {
    if (!std::isfinite(ptr[i])) return false;
  }
  return true;
}

GQC_HD inline Size tri_index(int i, int j, int n) {
  // Upper-triangular index for 0 <= i <= j < n.
  const Size ii = static_cast<Size>(i);
  const Size tri_skip = (ii > 0) ? ((ii * (ii - 1)) / 2) : 0;
  return ii * static_cast<Size>(n) - tri_skip + static_cast<Size>(j - i);
}

#ifdef GQC_SINK_HOST_REFERENCE

void rm_gemm_host(bool trans_a, bool trans_b, int M, int N, int K, double alpha, const double* A,
                  int lda, const double* B, int ldb, double beta, double* C, int ldc) {
  if (M <= 0 || N <= 0) return;
  if (beta == 0.0) {
    for (int i = 0; i < M; ++i) {
      std::fill_n(C + Size(i) * ldc, N, 0.0);
    }
  } else if (beta != 1.0) {
    for (int i = 0; i < M; ++i) {
      double* c_row = C + Size(i) * ldc;
      for (int j = 0; j < N; ++j) c_row[j] *= beta;
    }
  }
  if (K <= 0 || alpha == 0.0) return;

  if (!trans_a && !trans_b) {
    for (int i = 0; i < M; ++i) {
      const double* a_row = A + Size(i) * lda;
      double* c_row = C + Size(i) * ldc;
      for (int k = 0; k < K; ++k) {
        const double aik = alpha * a_row[k];
        if (aik == 0.0) continue;
        const double* b_row = B + Size(k) * ldb;
        for (int j = 0; j < N; ++j) c_row[j] += aik * b_row[j];
      }
    }
  } else if (!trans_a && trans_b) {
    for (int i = 0; i < M; ++i) {
      const double* a_row = A + Size(i) * lda;
      double* c_row = C + Size(i) * ldc;
      for (int j = 0; j < N; ++j) {
        const double* b_row = B + Size(j) * ldb;
        double acc = 0.0;
        for (int k = 0; k < K; ++k) acc += a_row[k] * b_row[k];
        c_row[j] += alpha * acc;
      }
    }
  } else if (trans_a && !trans_b) {
    for (int k = 0; k < K; ++k) {
      const double* a_row = A + Size(k) * lda;
      const double* b_row = B + Size(k) * ldb;
      for (int i = 0; i < M; ++i) {
        const double aki = alpha * a_row[i];
        if (aki == 0.0) continue;
        double* c_row = C + Size(i) * ldc;
        for (int j = 0; j < N; ++j) c_row[j] += aki * b_row[j];
      }
    }
  } else {
    for (int i = 0; i < M; ++i) {
      double* c_row = C + Size(i) * ldc;
      for (int j = 0; j < N; ++j) {
        const double* b_row = B + Size(j) * ldb;
        double acc = 0.0;
        for (int k = 0; k < K; ++k) acc += A[Size(k) * lda + i] * b_row[k];
        c_row[j] += alpha * acc;
      }
    }
  }
}

void extract_co_cv_host(int n, int o, int f, const double* coeff, std::vector<double>& Co,
                        std::vector<double>& Cv) {
  const int o_act = o - f;
  const int v = n - o;
  Co.resize(Size(n) * o_act);
  Cv.resize(Size(n) * v);
  for (int u = 0; u < n; ++u) {
    const double* row = coeff + Size(u) * n;
    for (int i = 0; i < o_act; ++i) Co[Size(u) * o_act + i] = row[f + i];
    for (int a = 0; a < v; ++a) Cv[Size(u) * v + a] = row[o + a];
  }
}

void build_ordinary_factors_host(int n, int q, int o, int f, const double* coeff,
                                 const double* ao_coul, const double* whitener,
                                 std::vector<double>& B_qov, std::vector<double>& B_ovq) {
  const int o_act = o - f;
  const int v = n - o;
  const Size ov = Size(o_act) * v;
  B_qov.assign(Size(q) * ov, 0.0);
  B_ovq.assign(ov * Size(q), 0.0);
  if (o_act <= 0 || v <= 0) return;

  std::vector<double> Co, Cv;
  extract_co_cv_host(n, o, f, coeff, Co, Cv);

  std::vector<double> mo_raw(Size(q) * ov, 0.0);
  std::vector<double> tmp(Size(o_act) * n, 0.0);
  const Size nn = Size(n) * n;
  for (int Q = 0; Q < q; ++Q) {
    const double* raw_Q = ao_coul + Size(Q) * nn;
    rm_gemm_host(true, false, o_act, n, n, 1.0, Co.data(), o_act, raw_Q, n, 0.0, tmp.data(), n);
    rm_gemm_host(false, false, o_act, v, n, 1.0, tmp.data(), n, Cv.data(), v, 0.0,
                 mo_raw.data() + Size(Q) * ov, v);
  }
  rm_gemm_host(false, false, q, static_cast<int>(ov), q, 1.0, whitener, q, mo_raw.data(),
               static_cast<int>(ov), 0.0, B_qov.data(), static_cast<int>(ov));
  for (int Q = 0; Q < q; ++Q) {
    const double* src = B_qov.data() + Size(Q) * ov;
    for (Size ia = 0; ia < ov; ++ia) {
      B_ovq[ia * Size(q) + Q] = src[ia];
    }
  }
}

void pullback_ordinary_factors_host(int n, int q, int o, int f, double ao_metric_scale,
                                    const double* coeff, const double* ao_coul,
                                    const double* whitener, const std::vector<double>& B_qov,
                                    const std::vector<double>& bar_B_ovq, double* bar_coeff,
                                    double* bar_ao_coul, double* bar_metric) {
  const int o_act = o - f;
  const int v = n - o;
  if (o_act <= 0 || v <= 0) return;
  const Size ov = Size(o_act) * v;
  const int ov_i = static_cast<int>(ov);

  std::vector<double> bar_B_qov(Size(q) * ov, 0.0);
  for (Size ia = 0; ia < ov; ++ia) {
    const double* src = bar_B_ovq.data() + ia * Size(q);
    for (int Q = 0; Q < q; ++Q) {
      bar_B_qov[Size(Q) * ov + ia] = src[Q];
    }
  }

  std::vector<double> bar_mo_raw(Size(q) * ov, 0.0);
  std::vector<double> fitted(Size(q) * ov, 0.0);
  rm_gemm_host(true, false, q, ov_i, q, 1.0, whitener, q, bar_B_qov.data(), ov_i, 0.0,
               bar_mo_raw.data(), ov_i);
  rm_gemm_host(true, false, q, ov_i, q, 1.0, whitener, q, B_qov.data(), ov_i, 0.0, fitted.data(),
               ov_i);

  std::vector<double> nonsym(Size(q) * q, 0.0);
  rm_gemm_host(false, true, q, q, ov_i, 1.0, bar_mo_raw.data(), ov_i, fitted.data(), ov_i, 0.0,
               nonsym.data(), q);
  const double m_coef = -0.25 * ao_metric_scale;
  for (int A = 0; A < q; ++A) {
    for (int B = 0; B < q; ++B) {
      bar_metric[Size(A) * q + B] += m_coef * (nonsym[Size(A) * q + B] + nonsym[Size(B) * q + A]);
    }
  }

  std::vector<double> Co, Cv;
  extract_co_cv_host(n, o, f, coeff, Co, Cv);
  std::vector<double> bar_Co(Size(n) * o_act, 0.0);
  std::vector<double> bar_Cv(Size(n) * v, 0.0);
  std::vector<double> Y(Size(o_act) * n, 0.0);
  std::vector<double> tmp(Size(o_act) * n, 0.0);
  const Size nn = Size(n) * n;

  for (int Q = 0; Q < q; ++Q) {
    const double* S_Q = bar_mo_raw.data() + Size(Q) * ov;
    const double* raw_Q = ao_coul + Size(Q) * nn;
    double* bar_raw_Q = bar_ao_coul + Size(Q) * nn;

    rm_gemm_host(false, true, o_act, n, v, 1.0, S_Q, v, Cv.data(), v, 0.0, Y.data(), n);
    rm_gemm_host(false, false, n, n, o_act, ao_metric_scale, Co.data(), o_act, Y.data(), n, 1.0,
                 bar_raw_Q, n);
    rm_gemm_host(false, true, n, o_act, n, 1.0, raw_Q, n, Y.data(), n, 1.0, bar_Co.data(), o_act);
    rm_gemm_host(true, false, o_act, n, n, 1.0, Co.data(), o_act, raw_Q, n, 0.0, tmp.data(), n);
    rm_gemm_host(true, false, n, v, o_act, 1.0, tmp.data(), n, S_Q, v, 1.0, bar_Cv.data(), v);
  }

  for (int u = 0; u < n; ++u) {
    double* dst = bar_coeff + Size(u) * n;
    for (int i = 0; i < o_act; ++i) dst[f + i] += bar_Co[Size(u) * o_act + i];
    for (int a = 0; a < v; ++a) dst[o + a] += bar_Cv[Size(u) * v + a];
  }
}

int run_uhf_same_spin_host(int n, int q, int o, int f, bool compute_response, bool has_occ_weights,
                           Size budget_bytes, const double* occ_w, const double* eps,
                           const std::vector<double>& B_ovq, std::vector<double>& bar_B_ovq,
                           double* fock, double& e_ss_out) {
  e_ss_out = 0.0;
  const int o_act = o - f;
  const int v = n - o;
  if (o_act <= 1 || v <= 0) return 0;

  const double* eo = eps + f;
  const double* ev = eps + o;
  const Size vv = Size(v) * v;
  const Size vq = Size(v) * q;
  const Size n_tri = (Size(o_act) * Size(o_act + 1)) / 2;
  const bool store_tri =
      compute_response && !has_occ_weights && (n_tri * vv * sizeof(double) <= budget_bytes / 2);

  std::vector<double> G(vv), Tss(vv);
  std::vector<double> bar_Fv;
  std::vector<double> Tss_tri;
  if (compute_response && !has_occ_weights) {
    bar_Fv.assign(vv, 0.0);
    if (store_tri) Tss_tri.assign(n_tri * vv, 0.0);
  }

  double e_ss = 0.0;
  for (int i = 0; i < o_act; ++i) {
    const double* Bi = B_ovq.data() + Size(i) * vq;
    double* bBi = compute_response ? (bar_B_ovq.data() + Size(i) * vq) : nullptr;
    for (int j = i + 1; j < o_act; ++j) {
      const double* Bj = B_ovq.data() + Size(j) * vq;
      double* bBj = compute_response ? (bar_B_ovq.data() + Size(j) * vq) : nullptr;
      rm_gemm_host(false, true, v, v, q, 1.0, Bi, q, Bj, q, 0.0, G.data(), v);

      const double eij = eo[i] + eo[j];
      const double wij = has_occ_weights ? 0.5 * (occ_w[i] + occ_w[j]) : 1.0;
      double pair_e = 0.0;
      double pair_sq = 0.0;
      for (int a = 0; a < v; ++a) {
        const double ea = ev[a];
        double row_sq = 0.0;
        for (int b = 0; b < v; ++b) {
          const double d = eij - ea - ev[b];
          if (!std::isfinite(d) || d == 0.0) return -4;
          const double num = G[Size(a) * v + b] - G[Size(b) * v + a];
          const double t = num / d;
          Tss[Size(a) * v + b] = t;
          pair_e += t * num;
          row_sq += t * t;
        }
        pair_sq += row_sq;
        if (compute_response && has_occ_weights) {
          fock[Size(o + a) * n + (o + a)] += wij * row_sq;
        }
      }
      e_ss += 0.5 * wij * pair_e;

      if (compute_response) {
        const double b_scale = 2.0 * wij;
        rm_gemm_host(false, false, v, q, v, b_scale, Tss.data(), v, Bj, q, 1.0, bBi, q);
        rm_gemm_host(true, false, v, q, v, b_scale, Tss.data(), v, Bi, q, 1.0, bBj, q);
        if (has_occ_weights) {
          fock[Size(f + i) * n + (f + i)] -= 0.5 * wij * pair_sq;
          fock[Size(f + j) * n + (f + j)] -= 0.5 * wij * pair_sq;
        } else {
          rm_gemm_host(false, true, v, v, v, 1.0, Tss.data(), v, Tss.data(), v, 1.0, bar_Fv.data(),
                       v);
          if (store_tri) {
            std::copy_n(Tss.data(), vv, Tss_tri.data() + tri_index(i, j, o_act) * vv);
          }
        }
      }
    }
  }
  e_ss_out = e_ss;

  if (compute_response && !has_occ_weights) {
    std::vector<double> bar_Fo(Size(o_act) * o_act, 0.0);
    if (store_tri) {
      std::vector<double> Tcol(Size(o_act) * vv, 0.0);
      for (int j = 0; j < o_act; ++j) {
        for (int i = 0; i < o_act; ++i) {
          double* dst = Tcol.data() + Size(i) * vv;
          if (i == j) {
            std::fill_n(dst, vv, 0.0);
          } else if (i < j) {
            std::copy_n(Tss_tri.data() + tri_index(i, j, o_act) * vv, vv, dst);
          } else {
            const double* src = Tss_tri.data() + tri_index(j, i, o_act) * vv;
            for (Size p = 0; p < vv; ++p) dst[p] = -src[p];
          }
        }
        rm_gemm_host(false, true, o_act, o_act, static_cast<int>(vv), -0.5, Tcol.data(),
                     static_cast<int>(vv), Tcol.data(), static_cast<int>(vv), 1.0, bar_Fo.data(),
                     o_act);
      }
    } else {
      // Bounded-memory fallback for bar_Fo[i, k] = -0.5 * sum_{j,a,b} Tss[i,j,a,b] * Tss[k,j,a,b].
      std::vector<double> Ti(vv), Tk(vv);
      auto build_tss_ij = [&](int i, int j, double* out) -> int {
        if (i == j) {
          std::fill_n(out, vv, 0.0);
          return 0;
        }
        const double* Bi = B_ovq.data() + Size(i) * vq;
        const double* Bj = B_ovq.data() + Size(j) * vq;
        rm_gemm_host(false, true, v, v, q, 1.0, Bi, q, Bj, q, 0.0, G.data(), v);
        const double eij = eo[i] + eo[j];
        for (int a = 0; a < v; ++a) {
          for (int b = 0; b < v; ++b) {
            const double d = eij - ev[a] - ev[b];
            if (!std::isfinite(d) || d == 0.0) return -4;
            out[Size(a) * v + b] = (G[Size(a) * v + b] - G[Size(b) * v + a]) / d;
          }
        }
        return 0;
      };
      for (int j = 0; j < o_act; ++j) {
        for (int i = 0; i < o_act; ++i) {
          if (i == j) continue;
          if (int rc = build_tss_ij(i, j, Ti.data()); rc != 0) return rc;
          for (int k = i; k < o_act; ++k) {
            if (k == j) continue;
            const double* pTk = Ti.data();
            if (k != i) {
              if (int rc = build_tss_ij(k, j, Tk.data()); rc != 0) return rc;
              pTk = Tk.data();
            }
            double dot = 0.0;
            for (Size p = 0; p < vv; ++p) dot += Ti[p] * pTk[p];
            bar_Fo[Size(i) * o_act + k] -= 0.5 * dot;
            if (k != i) bar_Fo[Size(k) * o_act + i] -= 0.5 * dot;
          }
        }
      }
    }
    for (int i = 0; i < o_act; ++i) {
      for (int k = 0; k < o_act; ++k) {
        fock[Size(f + i) * n + (f + k)] += bar_Fo[Size(i) * o_act + k];
      }
    }
    for (int a = 0; a < v; ++a) {
      for (int c = 0; c < v; ++c) {
        fock[Size(o + a) * n + (o + c)] += bar_Fv[Size(a) * v + c];
      }
    }
  }
  return 0;
}

int run_uhf_opp_spin_host(int n, int q, int oa, int fa, int ob, int fb, bool compute_response,
                          bool has_occ_weights, Size budget_bytes, const double* occ_w_a,
                          const double* occ_w_b, const double* eps_a, const double* eps_b,
                          const std::vector<double>& Ba_ovq, const std::vector<double>& Bb_ovq,
                          std::vector<double>& bar_Ba_ovq, std::vector<double>& bar_Bb_ovq,
                          double* fock_a, double* fock_b, double& e_ab_out) {
  e_ab_out = 0.0;
  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  const int va = n - oa;
  const int vb = n - ob;
  if (oa_act <= 0 || ob_act <= 0 || va <= 0 || vb <= 0) return 0;

  const double* eoa = eps_a + fa;
  const double* eva = eps_a + oa;
  const double* eob = eps_b + fb;
  const double* evb = eps_b + ob;
  const Size vab = Size(va) * vb;
  const Size vaq = Size(va) * q;
  const Size vbq = Size(vb) * q;

  const bool store_all = compute_response && !has_occ_weights &&
                         (Size(oa_act) * Size(ob_act) * vab * sizeof(double) <= budget_bytes / 2);

  std::vector<double> G(vab), T(vab);
  std::vector<double> bar_Fva, bar_Fvb, bar_Foa, bar_Fob;
  std::vector<double> T_all;
  if (compute_response && !has_occ_weights) {
    bar_Fva.assign(Size(va) * va, 0.0);
    bar_Fvb.assign(Size(vb) * vb, 0.0);
    bar_Foa.assign(Size(oa_act) * oa_act, 0.0);
    bar_Fob.assign(Size(ob_act) * ob_act, 0.0);
    if (store_all) T_all.assign(Size(oa_act) * Size(ob_act) * vab, 0.0);
  }

  double e_ab = 0.0;
  std::vector<double> Tj_col;
  if (compute_response && !has_occ_weights && !store_all &&
      Size(oa_act) * vab * sizeof(double) <= budget_bytes / 2) {
    Tj_col.assign(Size(oa_act) * vab, 0.0);
  }
  const bool use_j_col = !Tj_col.empty();

  for (int j = 0; j < ob_act; ++j) {
    const double* Bj = Bb_ovq.data() + Size(j) * vbq;
    double* bBj = compute_response ? (bar_Bb_ovq.data() + Size(j) * vbq) : nullptr;
    for (int i = 0; i < oa_act; ++i) {
      const double* Bi = Ba_ovq.data() + Size(i) * vaq;
      double* bBi = compute_response ? (bar_Ba_ovq.data() + Size(i) * vaq) : nullptr;
      rm_gemm_host(false, true, va, vb, q, 1.0, Bi, q, Bj, q, 0.0, G.data(), vb);

      const double eij = eoa[i] + eob[j];
      const double wij = has_occ_weights ? 0.5 * (occ_w_a[i] + occ_w_b[j]) : 1.0;
      double pair_e = 0.0;
      double pair_sq = 0.0;
      for (int a = 0; a < va; ++a) {
        const double ea = eva[a];
        double row_sq = 0.0;
        for (int b = 0; b < vb; ++b) {
          const double d = eij - ea - evb[b];
          if (!std::isfinite(d) || d == 0.0) return -4;
          const double g = G[Size(a) * vb + b];
          const double t = g / d;
          T[Size(a) * vb + b] = t;
          pair_e += t * g;
          const double t2 = t * t;
          row_sq += t2;
          if (compute_response && has_occ_weights) {
            fock_b[Size(ob + b) * n + (ob + b)] += wij * t2;
          }
        }
        pair_sq += row_sq;
        if (compute_response && has_occ_weights) {
          fock_a[Size(oa + a) * n + (oa + a)] += wij * row_sq;
        }
      }
      e_ab += wij * pair_e;

      if (compute_response) {
        const double b_scale = 2.0 * wij;
        rm_gemm_host(false, false, va, q, vb, b_scale, T.data(), vb, Bj, q, 1.0, bBi, q);
        rm_gemm_host(true, false, vb, q, va, b_scale, T.data(), vb, Bi, q, 1.0, bBj, q);
        if (has_occ_weights) {
          fock_a[Size(fa + i) * n + (fa + i)] -= wij * pair_sq;
          fock_b[Size(fb + j) * n + (fb + j)] -= wij * pair_sq;
        } else {
          rm_gemm_host(false, true, va, va, vb, 1.0, T.data(), vb, T.data(), vb, 1.0, bar_Fva.data(),
                       va);
          rm_gemm_host(true, false, vb, vb, va, 1.0, T.data(), vb, T.data(), vb, 1.0, bar_Fvb.data(),
                       vb);
          if (store_all) {
            std::copy_n(T.data(), vab, T_all.data() + (Size(i) * ob_act + j) * vab);
          } else if (use_j_col) {
            std::copy_n(T.data(), vab, Tj_col.data() + Size(i) * vab);
          }
        }
      }
    }
    if (compute_response && !has_occ_weights && !store_all && use_j_col) {
      rm_gemm_host(false, true, oa_act, oa_act, static_cast<int>(vab), -1.0, Tj_col.data(),
                   static_cast<int>(vab), Tj_col.data(), static_cast<int>(vab), 1.0, bar_Foa.data(),
                   oa_act);
    }
  }
  e_ab_out = e_ab;

  if (compute_response && !has_occ_weights) {
    if (store_all) {
      // bar_Fob[j, l] -= sum_{i,a,b} T_all[i, j, ab] * T_all[i, l, ab]
      for (int i = 0; i < oa_act; ++i) {
        const double* Ti = T_all.data() + Size(i) * ob_act * vab;
        rm_gemm_host(false, true, ob_act, ob_act, static_cast<int>(vab), -1.0, Ti,
                     static_cast<int>(vab), Ti, static_cast<int>(vab), 1.0, bar_Fob.data(), ob_act);
      }
      // bar_Foa[i, k] -= sum_{j,a,b} T_all[i, j, ab] * T_all[k, j, ab]
      std::vector<double> Tj(Size(oa_act) * vab);
      for (int j = 0; j < ob_act; ++j) {
        for (int i = 0; i < oa_act; ++i) {
          std::copy_n(T_all.data() + (Size(i) * ob_act + j) * vab, vab, Tj.data() + Size(i) * vab);
        }
        rm_gemm_host(false, true, oa_act, oa_act, static_cast<int>(vab), -1.0, Tj.data(),
                     static_cast<int>(vab), Tj.data(), static_cast<int>(vab), 1.0, bar_Foa.data(),
                     oa_act);
      }
    } else {
      auto build_t_ij = [&](int i, int j, double* out) -> int {
        const double* Bi = Ba_ovq.data() + Size(i) * vaq;
        const double* Bj = Bb_ovq.data() + Size(j) * vbq;
        rm_gemm_host(false, true, va, vb, q, 1.0, Bi, q, Bj, q, 0.0, G.data(), vb);
        const double eij = eoa[i] + eob[j];
        for (int a = 0; a < va; ++a) {
          for (int b = 0; b < vb; ++b) {
            const double d = eij - eva[a] - evb[b];
            if (!std::isfinite(d) || d == 0.0) return -4;
            out[Size(a) * vb + b] = G[Size(a) * vb + b] / d;
          }
        }
        return 0;
      };
      std::vector<double> T1(vab), T2(vab);
      if (!use_j_col) {
        for (int j = 0; j < ob_act; ++j) {
          for (int i = 0; i < oa_act; ++i) {
            if (int rc = build_t_ij(i, j, T1.data()); rc != 0) return rc;
            for (int k = i; k < oa_act; ++k) {
              const double* p2 = T1.data();
              if (k != i) {
                if (int rc = build_t_ij(k, j, T2.data()); rc != 0) return rc;
                p2 = T2.data();
              }
              double dot = 0.0;
              for (Size p = 0; p < vab; ++p) dot += T1[p] * p2[p];
              bar_Foa[Size(i) * oa_act + k] -= dot;
              if (k != i) bar_Foa[Size(k) * oa_act + i] -= dot;
            }
          }
        }
      }
      for (int i = 0; i < oa_act; ++i) {
        for (int j = 0; j < ob_act; ++j) {
          if (int rc = build_t_ij(i, j, T1.data()); rc != 0) return rc;
          for (int l = j; l < ob_act; ++l) {
            const double* p2 = T1.data();
            if (l != j) {
              if (int rc = build_t_ij(i, l, T2.data()); rc != 0) return rc;
              p2 = T2.data();
            }
            double dot = 0.0;
            for (Size p = 0; p < vab; ++p) dot += T1[p] * p2[p];
            bar_Fob[Size(j) * ob_act + l] -= dot;
            if (l != j) bar_Fob[Size(l) * ob_act + j] -= dot;
          }
        }
      }
    }

    for (int i = 0; i < oa_act; ++i) {
      for (int k = 0; k < oa_act; ++k) {
        fock_a[Size(fa + i) * n + (fa + k)] += bar_Foa[Size(i) * oa_act + k];
      }
    }
    for (int j = 0; j < ob_act; ++j) {
      for (int l = 0; l < ob_act; ++l) {
        fock_b[Size(fb + j) * n + (fb + l)] += bar_Fob[Size(j) * ob_act + l];
      }
    }
    for (int a = 0; a < va; ++a) {
      for (int c = 0; c < va; ++c) {
        fock_a[Size(oa + a) * n + (oa + c)] += bar_Fva[Size(a) * va + c];
      }
    }
    for (int b = 0; b < vb; ++b) {
      for (int d = 0; d < vb; ++d) {
        fock_b[Size(ob + b) * n + (ob + d)] += bar_Fvb[Size(b) * vb + d];
      }
    }
  }
  return 0;
}

#else

// The preflight plans the minimum viable schedule. Also meter actual live allocations
// so changes in a helper cannot overrun the caller's cap. Calls on other threads and
// nested native calls have independent ownership. CUDA driver/context memory is shared.
struct DeviceBudgetExceeded {};
struct DeviceWorkspace {
  Size available;
  DeviceWorkspace* previous;
  static thread_local DeviceWorkspace* current;

  explicit DeviceWorkspace(Size budget)
      : available(effective_device_budget(budget)), previous(current) {
    current = this;
  }
  ~DeviceWorkspace() { current = previous; }
};
thread_local DeviceWorkspace* DeviceWorkspace::current = nullptr;

template <typename T> struct WorkspaceBuffer {
  T* ptr = nullptr;
  Size cap = 0;
  WorkspaceBuffer() = default;
  WorkspaceBuffer(const WorkspaceBuffer&) = delete;
  WorkspaceBuffer& operator=(const WorkspaceBuffer&) = delete;
  ~WorkspaceBuffer() { reset(); }
  void reset() {
    if (ptr) {
      cudaFree(ptr);
      DeviceWorkspace::current->available += cap * sizeof(T);
      ptr = nullptr;
      cap = 0;
    }
  }
  cudaError_t alloc(Size count, bool optional = false) {
    if (count <= cap) return cudaSuccess;
    reset();
    auto& available = DeviceWorkspace::current->available;
    if (count > available / sizeof(T)) {
      if (optional) return cudaErrorMemoryAllocation;
      throw DeviceBudgetExceeded{};
    }
    const cudaError_t status = cudaMalloc(&ptr, count * sizeof(T));
    if (status == cudaSuccess) {
      cap = count;
      available -= cap * sizeof(T);
    }
    return status;
  }
};
using DeviceBuf = WorkspaceBuffer<double>;
using DeviceIntBuf = WorkspaceBuffer<int>;

struct CublasGuard {
  cublasHandle_t handle = nullptr;
  CublasGuard() = default;
  CublasGuard(const CublasGuard&) = delete;
  CublasGuard& operator=(const CublasGuard&) = delete;
  ~CublasGuard() {
    if (handle) cublasDestroy(handle);
  }
  cublasStatus_t init() {
    const auto status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    // These FP64 primitives accept exact-fit budgets; no uncharged BLAS operation workspace.
    return cublasSetWorkspace(handle, nullptr, 0);
  }
};

inline cublasStatus_t rm_gemm(cublasHandle_t h, bool trans_a, bool trans_b, int M, int N, int K,
                              double alpha, const double* A, int lda, const double* B, int ldb,
                              double beta, double* C, int ldc) {
  if (M <= 0 || N <= 0) return CUBLAS_STATUS_SUCCESS;
  return cublasDgemm(h, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N, trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                     N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc);
}

inline cublasStatus_t rm_gemm_batched(cublasHandle_t h, bool trans_a, bool trans_b, int M, int N,
                                      int K, double alpha, const double* A, int lda,
                                      long long stride_a, const double* B, int ldb,
                                      long long stride_b, double beta, double* C, int ldc,
                                      long long stride_c, int batch_count) {
  if (M <= 0 || N <= 0 || batch_count <= 0) return CUBLAS_STATUS_SUCCESS;
  return cublasDgemmStridedBatched(h, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                                   trans_a ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B, ldb,
                                   stride_b, A, lda, stride_a, &beta, C, ldc, stride_c,
                                   batch_count);
}

inline unsigned grid_for(Size n, int blk = 256) {
  return static_cast<unsigned>((n + blk - 1) / blk);
}

__global__ void k_transpose_2d(Size rows, Size cols, const double* __restrict__ src,
                               double* __restrict__ dst) {
  __shared__ double tile[32][33];
  const Size x = Size(blockIdx.x) * 32 + threadIdx.x;
  const Size y = Size(blockIdx.y) * 32 + threadIdx.y;
  if (x < cols && y < rows) {
    tile[threadIdx.y][threadIdx.x] = src[y * cols + x];
  }
  __syncthreads();
  const Size tx = Size(blockIdx.y) * 32 + threadIdx.x;
  const Size ty = Size(blockIdx.x) * 32 + threadIdx.y;
  if (tx < rows && ty < cols) {
    dst[ty * rows + tx] = tile[threadIdx.x][threadIdx.y];
  }
}

void launch_transpose_2d(Size rows, Size cols, const double* src, double* dst) {
  if (rows == 0 || cols == 0) return;
  dim3 block(32, 32);
  dim3 grid(static_cast<unsigned>((cols + 31) / 32), static_cast<unsigned>((rows + 31) / 32));
  k_transpose_2d<<<grid, block>>>(rows, cols, src, dst);
}

__global__ void k_extract_co_cv(int n, int o, int f, const double* __restrict__ coeff,
                                double* __restrict__ Co, double* __restrict__ Cv) {
  const int o_act = o - f;
  const int v = n - o;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(n) * (o_act + v);
  if (idx >= total) return;
  const int u = static_cast<int>(idx / (o_act + v));
  const int c = static_cast<int>(idx % (o_act + v));
  if (c < o_act) {
    Co[Size(u) * o_act + c] = coeff[Size(u) * n + (f + c)];
  } else {
    const int a = c - o_act;
    Cv[Size(u) * v + a] = coeff[Size(u) * n + (o + a)];
  }
}

__global__ void k_scatter_bar_co_cv(int n, int o, int f, const double* __restrict__ bar_Co,
                                    const double* __restrict__ bar_Cv,
                                    double* __restrict__ bar_coeff) {
  const int o_act = o - f;
  const int v = n - o;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(n) * (o_act + v);
  if (idx >= total) return;
  const int u = static_cast<int>(idx / (o_act + v));
  const int c = static_cast<int>(idx % (o_act + v));
  if (c < o_act) {
    bar_coeff[Size(u) * n + (f + c)] += bar_Co[Size(u) * o_act + c];
  } else {
    const int a = c - o_act;
    bar_coeff[Size(u) * n + (o + a)] += bar_Cv[Size(u) * v + a];
  }
}

__global__ void k_symmetrize_metric_acc(int q, double scale, const double* __restrict__ nonsym,
                                        double* __restrict__ bar_metric) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(q) * q;
  if (idx >= total) return;
  const int A = static_cast<int>(idx / q);
  const int B = static_cast<int>(idx % q);
  bar_metric[idx] += scale * (nonsym[idx] + nonsym[Size(B) * q + A]);
}

__global__ void k_scatter_fock_full(int n, int o, int f, const double* __restrict__ bar_Fo,
                                    const double* __restrict__ bar_Fv, double* __restrict__ fock) {
  const int o_act = o - f;
  const int v = n - o;
  const Size oo = Size(o_act) * o_act;
  const Size vv = Size(v) * v;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < oo) {
    const int i = static_cast<int>(idx / o_act);
    const int k = static_cast<int>(idx % o_act);
    fock[Size(f + i) * n + (f + k)] += bar_Fo[idx];
  } else if (idx < oo + vv) {
    const Size vidx = idx - oo;
    const int a = static_cast<int>(vidx / v);
    const int c = static_cast<int>(vidx % v);
    fock[Size(o + a) * n + (o + c)] += bar_Fv[vidx];
  }
}

__global__ void k_scatter_fock_diag(int n, int o, int f, const double* __restrict__ diag_o,
                                    const double* __restrict__ diag_v, double* __restrict__ fock) {
  const int o_act = o - f;
  const int v = n - o;
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < o_act) {
    fock[Size(f + idx) * n + (f + idx)] += diag_o[idx];
  } else if (idx < o_act + v) {
    const int a = idx - o_act;
    fock[Size(o + a) * n + (o + a)] += diag_v[a];
  }
}

// -------------------------------------------------------------------------
// RHF spatial fast-path CUDA kernels
// -------------------------------------------------------------------------

__global__ void k_rhf_g_to_t(int i0, int j0, int bi, int bj, int o_act, int v, bool store_tri,
                             const double* __restrict__ eo, const double* __restrict__ ev,
                             const double* __restrict__ G_blk, double* __restrict__ T_blk,
                             double* __restrict__ T_tri, int* __restrict__ err_flag) {
  const Size ld = Size(bj) * v;
  const Size vv = Size(v) * v;
  const Size block_id = Size(blockIdx.x);
  const int i_loc = static_cast<int>(block_id / (Size(bj) * v));
  const Size rem = block_id % (Size(bj) * v);
  const int j_loc = static_cast<int>(rem / v);
  const int a = static_cast<int>(rem % v);
  const int gi = i0 + i_loc;
  const int gj = j0 + j_loc;
  const double e_ija = eo[gi] + eo[gj] - ev[a];
  const Size row_off = Size(i_loc * v + a) * ld + Size(j_loc * v);
  const bool write_tri = store_tri && (gi <= gj);
  double* tri_row = write_tri ? (T_tri + tri_index(gi, gj, o_act) * vv + Size(a) * v) : nullptr;

  for (int b = threadIdx.x; b < v; b += blockDim.x) {
    const double d = e_ija - ev[b];
    if (!isfinite(d) || d == 0.0) {
      atomicExch(err_flag, -4);
      T_blk[row_off + b] = 0.0;
      if (tri_row) tri_row[b] = 0.0;
      continue;
    }
    const double t = G_blk[row_off + b] / d;
    T_blk[row_off + b] = t;
    if (tri_row) tri_row[b] = t;
  }
}

__global__ void k_rhf_t_to_u_and_energy(
    int i0, int j0, int bi, int bj, int v, bool off_diag, bool compute_response,
    bool has_occ_weights, const double* __restrict__ occ_w, const double* __restrict__ T_blk,
    double* __restrict__ G_to_U_blk, double* __restrict__ d_energies,
    double* __restrict__ d_diag_o, double* __restrict__ d_diag_v) {
  __shared__ double s_eaa[256];
  __shared__ double s_eab[256];
  __shared__ double s_tu[256];

  const Size ld = Size(bj) * v;
  const Size block_id = Size(blockIdx.x);
  const int i_loc = static_cast<int>(block_id / (Size(bj) * v));
  const Size rem = block_id % (Size(bj) * v);
  const int j_loc = static_cast<int>(rem / v);
  const int a = static_cast<int>(rem % v);
  const int gi = i0 + i_loc;
  const int gj = j0 + j_loc;
  const double wij = has_occ_weights ? 0.5 * (occ_w[gi] + occ_w[gj]) : 1.0;

  const Size row_ab = Size(i_loc * v + a) * ld + Size(j_loc * v);
  const Size col_a = Size(j_loc * v + a);
  const Size base_i = Size(i_loc * v);

  double local_eaa = 0.0;
  double local_eab = 0.0;
  double local_tu = 0.0;

  for (int b = threadIdx.x; b < v; b += blockDim.x) {
    const double tab = T_blk[row_ab + b];
    const double tba = T_blk[(base_i + b) * ld + col_a];
    const double gab = G_to_U_blk[row_ab + b];
    const double uab = 2.0 * tab - tba;
    if (off_diag) {
      local_eab += 2.0 * wij * tab * gab;
      local_eaa += wij * (tab - tba) * gab;
    } else {
      local_eab += wij * tab * gab;
      if (gi != gj) {
        local_eaa += 0.5 * wij * (tab - tba) * gab;
      }
    }
    if (compute_response) {
      G_to_U_blk[row_ab + b] = has_occ_weights ? (wij * uab) : uab;
      if (has_occ_weights) {
        const double tu = wij * tab * uab;
        local_tu += tu;
        if (off_diag) {
          atomicAdd(&d_diag_v[b], tu);
        }
      }
    }
  }

  s_eaa[threadIdx.x] = local_eaa;
  s_eab[threadIdx.x] = local_eab;
  s_tu[threadIdx.x] = local_tu;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_eaa[threadIdx.x] += s_eaa[threadIdx.x + stride];
      s_eab[threadIdx.x] += s_eab[threadIdx.x + stride];
      s_tu[threadIdx.x] += s_tu[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(&d_energies[0], s_eaa[0]);
    atomicAdd(&d_energies[1], s_eab[0]);
    if (compute_response && has_occ_weights) {
      atomicAdd(&d_diag_o[gi], -s_tu[0]);
      atomicAdd(&d_diag_v[a], s_tu[0]);
      if (off_diag) {
        atomicAdd(&d_diag_o[gj], -s_tu[0]);
      }
    }
  }
}

__global__ void k_rhf_gather_tri_cols(int o_act, int j0, int jb, int v,
                                      const double* __restrict__ T_tri,
                                      double* __restrict__ T_col, double* __restrict__ U_col) {
  const Size vv = Size(v) * v;
  const Size col_stride = Size(jb) * vv;
  const Size block_id = Size(blockIdx.x);
  const int i = static_cast<int>(block_id / (Size(jb) * v));
  const Size rem = block_id % (Size(jb) * v);
  const int j_loc = static_cast<int>(rem / v);
  const int a = static_cast<int>(rem % v);
  const int j = j0 + j_loc;

  const double* src = (i <= j) ? (T_tri + tri_index(i, j, o_act) * vv)
                               : (T_tri + tri_index(j, i, o_act) * vv);
  const Size dst_row = Size(i) * col_stride + Size(j_loc) * vv + Size(a) * v;
  for (int b = threadIdx.x; b < v; b += blockDim.x) {
    const double tab = (i <= j) ? src[Size(a) * v + b] : src[Size(b) * v + a];
    const double tba = (i <= j) ? src[Size(b) * v + a] : src[Size(a) * v + b];
    T_col[dst_row + b] = tab;
    U_col[dst_row + b] = 2.0 * tab - tba;
  }
}

__global__ void k_rhf_t_to_u_plain(int bi, int bj, int v, const double* __restrict__ T_blk,
                                   double* __restrict__ U_blk) {
  const Size ld = Size(bj) * v;
  const Size block_id = Size(blockIdx.x);
  const int i_loc = static_cast<int>(block_id / (Size(bj) * v));
  const Size rem = block_id % (Size(bj) * v);
  const int j_loc = static_cast<int>(rem / v);
  const int a = static_cast<int>(rem % v);
  const Size row_ab = Size(i_loc * v + a) * ld + Size(j_loc * v);
  const Size col_a = Size(j_loc * v + a);
  const Size base_i = Size(i_loc * v);
  for (int b = threadIdx.x; b < v; b += blockDim.x) {
    const double tab = T_blk[row_ab + b];
    const double tba = T_blk[(base_i + b) * ld + col_a];
    U_blk[row_ab + b] = 2.0 * tab - tba;
  }
}

// -------------------------------------------------------------------------
// UHF same-spin & opposite-spin CUDA kernels
// -------------------------------------------------------------------------

__global__ void k_uhf_ss_g_to_tss(int i0, int j0, int bi, int bj, int o_act, int v, bool store_tri,
                                  const double* __restrict__ eo, const double* __restrict__ ev,
                                  const double* __restrict__ G_blk, double* __restrict__ Tss_blk,
                                  double* __restrict__ Tss_tri, int* __restrict__ err_flag) {
  const Size ld = Size(bj) * v;
  const Size vv = Size(v) * v;
  const Size block_id = Size(blockIdx.x);
  const int i_loc = static_cast<int>(block_id / (Size(bj) * v));
  const Size rem = block_id % (Size(bj) * v);
  const int j_loc = static_cast<int>(rem / v);
  const int a = static_cast<int>(rem % v);
  const int gi = i0 + i_loc;
  const int gj = j0 + j_loc;

  const Size row_ab = Size(i_loc * v + a) * ld + Size(j_loc * v);
  const Size col_a = Size(j_loc * v + a);
  const Size base_i = Size(i_loc * v);
  const bool write_tri = store_tri && (gi < gj);
  double* tri_row = write_tri ? (Tss_tri + tri_index(gi, gj, o_act) * vv + Size(a) * v) : nullptr;

  if (gi == gj) {
    for (int b = threadIdx.x; b < v; b += blockDim.x) {
      Tss_blk[row_ab + b] = 0.0;
    }
    return;
  }

  const double e_ija = eo[gi] + eo[gj] - ev[a];
  for (int b = threadIdx.x; b < v; b += blockDim.x) {
    const double d = e_ija - ev[b];
    if (!isfinite(d) || d == 0.0) {
      atomicExch(err_flag, -4);
      Tss_blk[row_ab + b] = 0.0;
      if (tri_row) tri_row[b] = 0.0;
      continue;
    }
    const double num = G_blk[row_ab + b] - G_blk[(base_i + b) * ld + col_a];
    const double t = num / d;
    Tss_blk[row_ab + b] = t;
    if (tri_row) tri_row[b] = t;
  }
}

__global__ void k_uhf_ss_energy_and_w(int i0, int j0, int bi, int bj, int v, bool off_diag,
                                      bool compute_response, bool has_occ_weights,
                                      const double* __restrict__ occ_w,
                                      const double* __restrict__ Tss_blk,
                                      double* __restrict__ G_to_wTss_blk,
                                      double* __restrict__ d_e_ss, double* __restrict__ d_diag_o,
                                      double* __restrict__ d_diag_v) {
  __shared__ double s_e[256];
  __shared__ double s_sq[256];

  const Size ld = Size(bj) * v;
  const Size block_id = Size(blockIdx.x);
  const int i_loc = static_cast<int>(block_id / (Size(bj) * v));
  const Size rem = block_id % (Size(bj) * v);
  const int j_loc = static_cast<int>(rem / v);
  const int a = static_cast<int>(rem % v);
  const int gi = i0 + i_loc;
  const int gj = j0 + j_loc;
  const double wij = has_occ_weights ? 0.5 * (occ_w[gi] + occ_w[gj]) : 1.0;

  const Size row_ab = Size(i_loc * v + a) * ld + Size(j_loc * v);
  double local_e = 0.0;
  double local_sq = 0.0;

  // Note: sum_{a,b} Tss[a,b] * (G[a,b] - G[b,a]) = 2 * sum_{a,b} Tss[a,b] * G[a,b]
  // because Tss is antisymmetric in (a, b).
  const double e_coef = off_diag ? wij : (0.5 * wij);
  for (int b = threadIdx.x; b < v; b += blockDim.x) {
    const double t = Tss_blk[row_ab + b];
    const double g = G_to_wTss_blk[row_ab + b];
    local_e += e_coef * t * g;
    if (compute_response && has_occ_weights) {
      G_to_wTss_blk[row_ab + b] = wij * t;
      local_sq += wij * t * t;
    }
  }

  s_e[threadIdx.x] = local_e;
  s_sq[threadIdx.x] = local_sq;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_e[threadIdx.x] += s_e[threadIdx.x + stride];
      s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(d_e_ss, s_e[0]);
    if (compute_response && has_occ_weights) {
      atomicAdd(&d_diag_o[gi], -0.5 * s_sq[0]);
      if (off_diag) {
        atomicAdd(&d_diag_o[gj], -0.5 * s_sq[0]);
        atomicAdd(&d_diag_v[a], s_sq[0]);
      } else {
        atomicAdd(&d_diag_v[a], 0.5 * s_sq[0]);
      }
    }
  }
}

__global__ void k_uhf_ss_gather_tri_cols(int o_act, int j0, int jb, int v,
                                         const double* __restrict__ Tss_tri,
                                         double* __restrict__ Tss_col) {
  const Size vv = Size(v) * v;
  const Size col_stride = Size(jb) * vv;
  const Size block_id = Size(blockIdx.x);
  const int i = static_cast<int>(block_id / (Size(jb) * v));
  const Size rem = block_id % (Size(jb) * v);
  const int j_loc = static_cast<int>(rem / v);
  const int a = static_cast<int>(rem % v);
  const int j = j0 + j_loc;

  const Size dst_row = Size(i) * col_stride + Size(j_loc) * vv + Size(a) * v;
  if (i == j) {
    for (int b = threadIdx.x; b < v; b += blockDim.x) Tss_col[dst_row + b] = 0.0;
    return;
  }
  const double* src = (i < j) ? (Tss_tri + tri_index(i, j, o_act) * vv + Size(a) * v)
                              : (Tss_tri + tri_index(j, i, o_act) * vv + Size(a) * v);
  const double sign = (i < j) ? 1.0 : -1.0;
  for (int b = threadIdx.x; b < v; b += blockDim.x) {
    Tss_col[dst_row + b] = sign * src[b];
  }
}

__global__ void k_uhf_ab_g_to_t(int i0, int j0, int bi, int bj, int ob_act, int va, int vb,
                                bool compute_response, bool has_occ_weights, bool store_all,
                                const double* __restrict__ occ_w_a,
                                const double* __restrict__ occ_w_b, const double* __restrict__ eoa,
                                const double* __restrict__ eob, const double* __restrict__ eva,
                                const double* __restrict__ evb, double* __restrict__ G_to_wT_blk,
                                double* __restrict__ T_blk, double* __restrict__ T_all,
                                double* __restrict__ d_e_ab, double* __restrict__ d_diag_oa,
                                double* __restrict__ d_diag_ob, double* __restrict__ d_diag_va,
                                double* __restrict__ d_diag_vb, int* __restrict__ err_flag) {
  __shared__ double s_e[256];
  __shared__ double s_sq[256];

  const Size ld = Size(bj) * vb;
  const Size vab = Size(va) * vb;
  const Size block_id = Size(blockIdx.x);
  const int i_loc = static_cast<int>(block_id / (Size(bj) * va));
  const Size rem = block_id % (Size(bj) * va);
  const int j_loc = static_cast<int>(rem / va);
  const int a = static_cast<int>(rem % va);
  const int gi = i0 + i_loc;
  const int gj = j0 + j_loc;

  const double e_ija = eoa[gi] + eob[gj] - eva[a];
  const double wij = has_occ_weights ? 0.5 * (occ_w_a[gi] + occ_w_b[gj]) : 1.0;
  const Size row_ab = Size(i_loc * va + a) * ld + Size(j_loc * vb);
  double* all_row =
      store_all ? (T_all + (Size(gi) * ob_act + gj) * vab + Size(a) * vb) : nullptr;

  double local_e = 0.0;
  double local_sq = 0.0;
  for (int b = threadIdx.x; b < vb; b += blockDim.x) {
    const double d = e_ija - evb[b];
    if (!isfinite(d) || d == 0.0) {
      atomicExch(err_flag, -4);
      T_blk[row_ab + b] = 0.0;
      if (all_row) all_row[b] = 0.0;
      continue;
    }
    const double g = G_to_wT_blk[row_ab + b];
    const double t = g / d;
    T_blk[row_ab + b] = t;
    if (all_row) all_row[b] = t;
    local_e += wij * t * g;
    if (compute_response && has_occ_weights) {
      G_to_wT_blk[row_ab + b] = wij * t;
      const double wt2 = wij * t * t;
      local_sq += wt2;
      atomicAdd(&d_diag_vb[b], wt2);
    }
  }

  s_e[threadIdx.x] = local_e;
  s_sq[threadIdx.x] = local_sq;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_e[threadIdx.x] += s_e[threadIdx.x + stride];
      s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(d_e_ab, s_e[0]);
    if (compute_response && has_occ_weights) {
      atomicAdd(&d_diag_oa[gi], -s_sq[0]);
      atomicAdd(&d_diag_ob[gj], -s_sq[0]);
      atomicAdd(&d_diag_va[a], s_sq[0]);
    }
  }
}

__global__ void k_uhf_ab_gather_j_col(int oa_act, int ob_act, int j0, int jb, Size vab,
                                      const double* __restrict__ T_all,
                                      double* __restrict__ T_col) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(oa_act) * Size(jb) * vab;
  if (idx >= total) return;
  const Size jvab = Size(jb) * vab;
  const int i = static_cast<int>(idx / jvab);
  const Size rem = idx % jvab;
  const int j_loc = static_cast<int>(rem / vab);
  const Size ab = rem % vab;
  T_col[idx] = T_all[(Size(i) * ob_act + (j0 + j_loc)) * vab + ab];
}

__global__ void k_uhf_ab_g_to_t_plain(int i0, int j0, int bi, int bj, int va, int vb,
                                      const double* __restrict__ eoa,
                                      const double* __restrict__ eob,
                                      const double* __restrict__ eva,
                                      const double* __restrict__ evb,
                                      const double* __restrict__ G_blk, double* __restrict__ T_blk,
                                      int* __restrict__ err_flag) {
  const Size ld = Size(bj) * vb;
  const Size block_id = Size(blockIdx.x);
  const int i_loc = static_cast<int>(block_id / (Size(bj) * va));
  const Size rem = block_id % (Size(bj) * va);
  const int j_loc = static_cast<int>(rem / va);
  const int a = static_cast<int>(rem % va);
  const double e_ija = eoa[i0 + i_loc] + eob[j0 + j_loc] - eva[a];
  const Size row_ab = Size(i_loc * va + a) * ld + Size(j_loc * vb);
  for (int b = threadIdx.x; b < vb; b += blockDim.x) {
    const double d = e_ija - evb[b];
    if (!isfinite(d) || d == 0.0) {
      atomicExch(err_flag, -4);
      T_blk[row_ab + b] = 0.0;
      continue;
    }
    T_blk[row_ab + b] = G_blk[row_ab + b] / d;
  }
}

// -------------------------------------------------------------------------
// GPU Factor Build & Factor Pullback (Q-slabbed, bounded VRAM)
// -------------------------------------------------------------------------

bool build_ordinary_factors_gpu(cublasHandle_t h, int n, int q, int o, int f, Size slab_cap,
                                const double* h_coeff, const double* h_ao_coul,
                                const double* d_whitener, DeviceBuf& d_B_ovq) {
  const int o_act = o - f;
  const int v = n - o;
  if (o_act <= 0 || v <= 0) return true;
  const Size ov = Size(o_act) * v;
  const Size nn = Size(n) * n;

  const Size bytes_per_q = (nn + Size(o_act) * n) * sizeof(double);
  const int q_slab = std::max(
      1, std::min(q, static_cast<int>(slab_cap / std::max<Size>(1, bytes_per_q))));

  DeviceBuf d_coeff, d_Co, d_Cv, d_mo_raw, d_B_qov, d_ao_slab, d_tmp_slab;
  if (d_coeff.alloc(nn) != cudaSuccess || d_Co.alloc(Size(n) * o_act) != cudaSuccess ||
      d_Cv.alloc(Size(n) * v) != cudaSuccess || d_mo_raw.alloc(Size(q) * ov) != cudaSuccess ||
      d_B_qov.alloc(Size(q) * ov) != cudaSuccess || d_B_ovq.alloc(ov * Size(q)) != cudaSuccess ||
      d_ao_slab.alloc(Size(q_slab) * nn) != cudaSuccess ||
      d_tmp_slab.alloc(Size(q_slab) * Size(o_act) * n) != cudaSuccess) {
    return false;
  }
  if (cudaMemcpy(d_coeff.ptr, h_coeff, nn * sizeof(double), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    return false;
  }
  k_extract_co_cv<<<grid_for(Size(n) * (o_act + v)), 256>>>(n, o, f, d_coeff.ptr, d_Co.ptr,
                                                            d_Cv.ptr);

  for (int q0 = 0; q0 < q; q0 += q_slab) {
    const int cur = std::min(q_slab, q - q0);
    if (cudaMemcpy(d_ao_slab.ptr, h_ao_coul + Size(q0) * nn, Size(cur) * nn * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return false;
    }
    if (rm_gemm_batched(h, true, false, o_act, n, n, 1.0, d_Co.ptr, o_act, 0, d_ao_slab.ptr, n,
                        static_cast<long long>(nn), 0.0, d_tmp_slab.ptr, n,
                        static_cast<long long>(Size(o_act) * n), cur) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    if (rm_gemm(h, false, false, cur * o_act, v, n, 1.0, d_tmp_slab.ptr, n, d_Cv.ptr, v, 0.0,
                d_mo_raw.ptr + Size(q0) * ov, v) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
  }

  if (rm_gemm(h, false, false, q, static_cast<int>(ov), q, 1.0, d_whitener, q, d_mo_raw.ptr,
              static_cast<int>(ov), 0.0, d_B_qov.ptr, static_cast<int>(ov)) !=
      CUBLAS_STATUS_SUCCESS) {
    return false;
  }
  launch_transpose_2d(Size(q), ov, d_B_qov.ptr, d_B_ovq.ptr);
  return cudaGetLastError() == cudaSuccess;
}

bool pullback_ordinary_factors_gpu(cublasHandle_t h, int n, int q, int o, int f,
                                   double ao_metric_scale, bool add_to_host_ao, Size slab_cap,
                                   const double* h_coeff, const double* h_ao_coul,
                                   const double* d_whitener, const double* d_B_ovq,
                                   const double* d_bar_B_ovq, double* d_bar_metric,
                                   double* h_bar_coeff, double* h_bar_ao_coul) {
  const int o_act = o - f;
  const int v = n - o;
  if (o_act <= 0 || v <= 0) return true;
  const Size ov = Size(o_act) * v;
  const Size nn = Size(n) * n;
  const int ov_i = static_cast<int>(ov);

  // Stage 1: Whitener pullback & metric_bar (in its own scope so d_fitted/d_tmp_qov are freed
  // before allocating Q-slab buffers!).
  DeviceBuf d_bar_mo_raw;
  if (d_bar_mo_raw.alloc(Size(q) * ov) != cudaSuccess) return false;
  {
    DeviceBuf d_qov_tmp, d_fitted, d_nonsym;
    if (d_qov_tmp.alloc(Size(q) * ov) != cudaSuccess ||
        d_fitted.alloc(Size(q) * ov) != cudaSuccess ||
        d_nonsym.alloc(Size(q) * q) != cudaSuccess) {
      return false;
    }
    launch_transpose_2d(ov, Size(q), d_bar_B_ovq, d_qov_tmp.ptr);
    if (rm_gemm(h, true, false, q, ov_i, q, 1.0, d_whitener, q, d_qov_tmp.ptr, ov_i, 0.0,
                d_bar_mo_raw.ptr, ov_i) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    launch_transpose_2d(ov, Size(q), d_B_ovq, d_qov_tmp.ptr);
    if (rm_gemm(h, true, false, q, ov_i, q, 1.0, d_whitener, q, d_qov_tmp.ptr, ov_i, 0.0,
                d_fitted.ptr, ov_i) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    if (rm_gemm(h, false, true, q, q, ov_i, 1.0, d_bar_mo_raw.ptr, ov_i, d_fitted.ptr, ov_i, 0.0,
                d_nonsym.ptr, q) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    k_symmetrize_metric_acc<<<grid_for(Size(q) * q), 256>>>(q, -0.25 * ao_metric_scale,
                                                            d_nonsym.ptr, d_bar_metric);
    if (cudaGetLastError() != cudaSuccess) return false;
  }

  // Stage 2: Q-slabbed AO & MO-coefficient pullback (reuses d_ao_slab in-place for bar_ao_slab
  // and d_Z_slab for both bar_Co and bar_Cv intermediates!).
  const Size bytes_per_q = (nn + 2 * Size(o_act) * n) * sizeof(double);
  const int q_slab = std::max(
      1, std::min(q, static_cast<int>(slab_cap / std::max<Size>(1, bytes_per_q))));

  DeviceBuf d_coeff, d_Co, d_Cv, d_bar_Co, d_bar_Cv, d_bar_coeff;
  DeviceBuf d_ao_slab, d_Y_slab, d_Z_slab, d_ones;
  if (d_coeff.alloc(nn) != cudaSuccess || d_Co.alloc(Size(n) * o_act) != cudaSuccess ||
      d_Cv.alloc(Size(n) * v) != cudaSuccess || d_bar_Co.alloc(Size(n) * o_act) != cudaSuccess ||
      d_bar_Cv.alloc(Size(n) * v) != cudaSuccess || d_bar_coeff.alloc(nn) != cudaSuccess ||
      d_ao_slab.alloc(Size(q_slab) * nn) != cudaSuccess ||
      d_Y_slab.alloc(Size(q_slab) * Size(o_act) * n) != cudaSuccess ||
      d_Z_slab.alloc(Size(q_slab) * Size(o_act) * n) != cudaSuccess ||
      d_ones.alloc(q_slab) != cudaSuccess) {
    return false;
  }

  std::vector<double> ones(q_slab, 1.0);
  if (cudaMemcpy(d_ones.ptr, ones.data(), Size(q_slab) * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemcpy(d_coeff.ptr, h_coeff, nn * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemset(d_bar_Co.ptr, 0, Size(n) * o_act * sizeof(double)) != cudaSuccess ||
      cudaMemset(d_bar_Cv.ptr, 0, Size(n) * v * sizeof(double)) != cudaSuccess ||
      cudaMemset(d_bar_coeff.ptr, 0, nn * sizeof(double)) != cudaSuccess) {
    return false;
  }
  k_extract_co_cv<<<grid_for(Size(n) * (o_act + v)), 256>>>(n, o, f, d_coeff.ptr, d_Co.ptr,
                                                            d_Cv.ptr);

  std::vector<double> h_slab_tmp;
  if (add_to_host_ao) h_slab_tmp.resize(Size(q_slab) * nn);

  for (int q0 = 0; q0 < q; q0 += q_slab) {
    const int cur = std::min(q_slab, q - q0);
    const Size cur_qnn = Size(cur) * nn;
    const double* S_slab = d_bar_mo_raw.ptr + Size(q0) * ov;

    if (cudaMemcpy(d_ao_slab.ptr, h_ao_coul + Size(q0) * nn, cur_qnn * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return false;
    }
    // Y[cur, o_act, n] = S[cur, o_act, v] @ Cv^T
    if (rm_gemm(h, false, true, cur * o_act, n, v, 1.0, S_slab, v, d_Cv.ptr, v, 0.0, d_Y_slab.ptr,
                n) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    // bar_Co[n, o_act] += sum_q raw[q] @ Y[q]^T
    if (rm_gemm_batched(h, false, true, n, o_act, n, 1.0, d_ao_slab.ptr, n,
                        static_cast<long long>(nn), d_Y_slab.ptr, n,
                        static_cast<long long>(Size(o_act) * n), 0.0, d_Z_slab.ptr, o_act,
                        static_cast<long long>(Size(n) * o_act), cur) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    if (rm_gemm(h, false, false, 1, n * o_act, cur, 1.0, d_ones.ptr, cur, d_Z_slab.ptr, n * o_act,
                1.0, d_bar_Co.ptr, n * o_act) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    // tmp[cur, o_act, n] = Co^T @ raw[q], then bar_Cv[n, v] += tmp^T @ S_slab
    if (rm_gemm_batched(h, true, false, o_act, n, n, 1.0, d_Co.ptr, o_act, 0, d_ao_slab.ptr, n,
                        static_cast<long long>(nn), 0.0, d_Z_slab.ptr, n,
                        static_cast<long long>(Size(o_act) * n), cur) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    if (rm_gemm(h, true, false, n, v, cur * o_act, 1.0, d_Z_slab.ptr, n, S_slab, v, 1.0,
                d_bar_Cv.ptr, v) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    // Overwrite d_ao_slab in-place with bar_ao_slab[q] = ao_metric_scale * (Co @ Y[q])
    if (rm_gemm_batched(h, false, false, n, n, o_act, ao_metric_scale, d_Co.ptr, o_act, 0,
                        d_Y_slab.ptr, n, static_cast<long long>(Size(o_act) * n), 0.0,
                        d_ao_slab.ptr, n, static_cast<long long>(nn),
                        cur) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    if (!add_to_host_ao) {
      if (cudaMemcpy(h_bar_ao_coul + Size(q0) * nn, d_ao_slab.ptr, cur_qnn * sizeof(double),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
      }
    } else {
      if (cudaMemcpy(h_slab_tmp.data(), d_ao_slab.ptr, cur_qnn * sizeof(double),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
      }
      double* dst = h_bar_ao_coul + Size(q0) * nn;
      for (Size p = 0; p < cur_qnn; ++p) dst[p] += h_slab_tmp[p];
    }
  }

  k_scatter_bar_co_cv<<<grid_for(Size(n) * (o_act + v)), 256>>>(n, o, f, d_bar_Co.ptr, d_bar_Cv.ptr,
                                                                d_bar_coeff.ptr);
  if (cudaGetLastError() != cudaSuccess) return false;
  if (cudaMemcpy(h_bar_coeff, d_bar_coeff.ptr, nn * sizeof(double), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return false;
  }
  return true;
}

// -------------------------------------------------------------------------
// GPU Pair Engines: RHF Spatial Fast Path + UHF Same-Spin + UHF Opp-Spin
// -------------------------------------------------------------------------

// Unweighted response processes one occupied pair per launch on the handle's
// default stream. Only lane zero writes the two symmetric Fock entries.
__global__ void k_accum_occupied_fock_dot(Size count, const double* left, const double* right,
                                          int nocc, int row, int col, double scale, double* fock) {
  if (count == 0 || nocc <= 0 || row < 0 || row >= nocc || col < 0 || col >= nocc) return;
  __shared__ double partial[256];
  double sum = 0.0;
  for (Size i = threadIdx.x; i < count; i += blockDim.x) sum += left[i] * right[i];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double update = scale * partial[0];
    fock[Size(row) * nocc + col] += update;
    if (row != col) fock[Size(col) * nocc + row] += update;
  }
}

inline cudaError_t launch_accum_occupied_fock_dot(Size count, const double* left,
                                                  const double* right, int nocc, int row, int col,
                                                  double scale, double* fock) {
  if (count == 0 || nocc <= 0) return cudaSuccess;
  if (!left || !right || !fock || row < 0 || row >= nocc || col < 0 || col >= nocc) {
    return cudaErrorInvalidValue;
  }
  k_accum_occupied_fock_dot<<<1, 256>>>(count, left, right, nocc, row, col, scale, fock);
  return cudaGetLastError();
}

int run_rhf_pairs_gpu(cublasHandle_t h, int n, int q, int o, int f, bool compute_response,
                      bool has_occ_weights, Size tile_cap, Size tri_cap, const double* h_occ_w,
                      const double* h_eps, const double* d_B_ovq, double* d_bar_B_ovq,
                      double* d_fock_a, double& e_aa_out, double& e_ab_out) {
  e_aa_out = 0.0;
  e_ab_out = 0.0;
  const int o_act = o - f;
  const int v = n - o;
  if (o_act <= 0 || v <= 0) return 0;

  const Size vv = Size(v) * v;
  const Size vq = Size(v) * q;
  const Size n_tri = (Size(o_act) * Size(o_act + 1)) / 2;
  bool store_tri =
      compute_response && !has_occ_weights && (bytes_of_doubles(sat_mul(n_tri, vv)) <= tri_cap);

  // Choose occupied tile size b_occ so 2 * b_occ^2 * vv * 8 <= tile_cap (min 1, max o_act).
  const Size max_pairs_in_tile =
      std::max<Size>(1, tile_cap / std::max<Size>(1, 2 * vv * sizeof(double)));
  int b_occ = static_cast<int>(std::floor(std::sqrt(static_cast<double>(max_pairs_in_tile))));
  b_occ = std::max(1, std::min(o_act, b_occ));
  // Ensure tile workspace can also hold at least 1 full column [o_act, vv] if store_tri and it fits.
  Size tile_elems = Size(b_occ) * Size(b_occ) * vv;
  const bool col_fits_tile = bytes_of_doubles(sat_mul(2 * Size(o_act), vv)) <= tile_cap;
  store_tri = store_tri && col_fits_tile;
  if (compute_response && !has_occ_weights && col_fits_tile && tile_elems < Size(o_act) * vv) {
    tile_elems = Size(o_act) * vv;
  }

  DeviceBuf d_eo, d_ev, d_occ_w, d_T_blk, d_U_blk, d_T_tri;
  DeviceBuf d_bar_Fo, d_bar_Fv, d_diag_o, d_diag_v, d_energies;
  DeviceIntBuf d_err;

  if (d_eo.alloc(o_act) != cudaSuccess || d_ev.alloc(v) != cudaSuccess ||
      d_T_blk.alloc(tile_elems) != cudaSuccess || d_U_blk.alloc(tile_elems) != cudaSuccess ||
      d_energies.alloc(2) != cudaSuccess || d_err.alloc(1) != cudaSuccess) {
    return 2;
  }
  if (has_occ_weights && d_occ_w.alloc(o_act) != cudaSuccess) return 2;

  if (compute_response) {
    if (has_occ_weights) {
      if (d_diag_o.alloc(o_act) != cudaSuccess || d_diag_v.alloc(v) != cudaSuccess) return 2;
    } else {
      if (d_bar_Fo.alloc(Size(o_act) * o_act) != cudaSuccess || d_bar_Fv.alloc(vv) != cudaSuccess) {
        return 2;
      }
    }
  }
  if (store_tri && d_T_tri.alloc(n_tri * vv, true) != cudaSuccess) {
    // Optional triangular cache failed allocation; clear sticky error and fall back to exact recompute.
    (void)cudaGetLastError();
    d_T_tri.reset();
    store_tri = false;
  }

  if (cudaMemcpy(d_eo.ptr, h_eps + f, Size(o_act) * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemcpy(d_ev.ptr, h_eps + o, Size(v) * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemset(d_energies.ptr, 0, 2 * sizeof(double)) != cudaSuccess ||
      cudaMemset(d_err.ptr, 0, sizeof(int)) != cudaSuccess) {
    return 3;
  }
  if (has_occ_weights && cudaMemcpy(d_occ_w.ptr, h_occ_w, Size(o_act) * sizeof(double),
                                    cudaMemcpyHostToDevice) != cudaSuccess) {
    return 3;
  }
  if (compute_response) {
    if (has_occ_weights) {
      if (cudaMemset(d_diag_o.ptr, 0, Size(o_act) * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_diag_v.ptr, 0, Size(v) * sizeof(double)) != cudaSuccess) {
        return 3;
      }
    } else {
      if (cudaMemset(d_bar_Fo.ptr, 0, Size(o_act) * o_act * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_Fv.ptr, 0, vv * sizeof(double)) != cudaSuccess) {
        return 3;
      }
    }
  }

  // Main upper-triangular occupied-block loop (i0 <= j0): halves G GEMM FLOPs!
  for (int i0 = 0; i0 < o_act; i0 += b_occ) {
    const int bi = std::min(b_occ, o_act - i0);
    const double* Bi = d_B_ovq + Size(i0) * vq;
    double* bBi = compute_response ? (d_bar_B_ovq + Size(i0) * vq) : nullptr;
    for (int j0 = i0; j0 < o_act; j0 += b_occ) {
      const int bj = std::min(b_occ, o_act - j0);
      const bool off_diag = (i0 < j0);
      const double* Bj = d_B_ovq + Size(j0) * vq;
      double* bBj = compute_response ? (d_bar_B_ovq + Size(j0) * vq) : nullptr;
      const int m_dim = bi * v;
      const int n_dim = bj * v;

      // 1. G_blk[bi*v, bj*v] = Bi[bi*v, q] @ Bj[bj*v, q]^T (stored in d_U_blk)
      if (rm_gemm(h, false, true, m_dim, n_dim, q, 1.0, Bi, q, Bj, q, 0.0, d_U_blk.ptr, n_dim) !=
          CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      // 2. T_blk = G_blk / denom (+ optional save into d_T_tri for i <= j)
      const unsigned nblocks = static_cast<unsigned>(Size(bi) * Size(bj) * Size(v));
      k_rhf_g_to_t<<<nblocks, 256>>>(i0, j0, bi, bj, o_act, v, store_tri, d_eo.ptr, d_ev.ptr,
                                     d_U_blk.ptr, d_T_blk.ptr, d_T_tri.ptr, d_err.ptr);
      // 3. U_blk = 2*T_blk - T_blk^T (in-place into d_U_blk) + E_aa/E_ab + weighted diag Fock
      k_rhf_t_to_u_and_energy<<<nblocks, 256>>>(i0, j0, bi, bj, v, off_diag, compute_response,
                                                has_occ_weights, d_occ_w.ptr, d_T_blk.ptr,
                                                d_U_blk.ptr, d_energies.ptr, d_diag_o.ptr,
                                                d_diag_v.ptr);

      if (compute_response) {
        // 4. bar_B update (single giant DGEMM per leg)
        if (rm_gemm(h, false, false, m_dim, q, n_dim, 2.0, d_U_blk.ptr, n_dim, Bj, q, 1.0, bBi,
                    q) != CUBLAS_STATUS_SUCCESS) {
          return 5;
        }
        if (off_diag) {
          if (rm_gemm(h, true, false, n_dim, q, m_dim, 2.0, d_U_blk.ptr, n_dim, Bi, q, 1.0, bBj,
                      q) != CUBLAS_STATUS_SUCCESS) {
            return 5;
          }
        }
        // 5. bar_Fv update (zero-copy row & column slice DGEMMs)
        if (!has_occ_weights) {
          for (int i_loc = 0; i_loc < bi; ++i_loc) {
            const Size off = Size(i_loc) * v * n_dim;
            if (rm_gemm(h, false, true, v, v, n_dim, 1.0, d_T_blk.ptr + off, n_dim,
                        d_U_blk.ptr + off, n_dim, 1.0, d_bar_Fv.ptr, v) != CUBLAS_STATUS_SUCCESS) {
              return 5;
            }
          }
          if (off_diag) {
            for (int j_loc = 0; j_loc < bj; ++j_loc) {
              const Size off = Size(j_loc) * v;
              if (rm_gemm(h, true, false, v, v, m_dim, 1.0, d_T_blk.ptr + off, n_dim,
                          d_U_blk.ptr + off, n_dim, 1.0, d_bar_Fv.ptr, v) !=
                  CUBLAS_STATUS_SUCCESS) {
                return 5;
              }
            }
          }
        }
      }
    }
  }

  // 6. Full bar_Fo[o_act, o_act] = -sum_j T_j @ U_j^T (unweighted response mode)
  if (compute_response && !has_occ_weights) {
    const int k_jb_cap =
        std::max(1, static_cast<int>(std::min<Size>(o_act, Size(INT_MAX) / std::max<Size>(1, vv))));
    if (store_tri && col_fits_tile) {
      const int jb_max =
          std::max(1, std::min(k_jb_cap, static_cast<int>(tile_elems / (Size(o_act) * vv))));
      for (int j0 = 0; j0 < o_act; j0 += jb_max) {
        const int jb = std::min(jb_max, o_act - j0);
        const unsigned nb = static_cast<unsigned>(Size(o_act) * Size(jb) * Size(v));
        k_rhf_gather_tri_cols<<<nb, 256>>>(o_act, j0, jb, v, d_T_tri.ptr, d_T_blk.ptr, d_U_blk.ptr);
        const int k_dim = static_cast<int>(Size(jb) * vv);
        if (rm_gemm(h, false, true, o_act, o_act, k_dim, -1.0, d_T_blk.ptr, k_dim, d_U_blk.ptr,
                    k_dim, 1.0, d_bar_Fo.ptr, o_act) != CUBLAS_STATUS_SUCCESS) {
          return 6;
        }
      }
    } else if (col_fits_tile) {
      // Regime B1: column-tiled recompute without storing T_tri (zero permutation needed!).
      const int jb_max =
          std::max(1, std::min(k_jb_cap, static_cast<int>(tile_elems / (Size(o_act) * vv))));
      for (int j0 = 0; j0 < o_act; j0 += jb_max) {
        const int jb = std::min(jb_max, o_act - j0);
        const double* Bj = d_B_ovq + Size(j0) * vq;
        if (rm_gemm(h, false, true, o_act * v, jb * v, q, 1.0, d_B_ovq, q, Bj, q, 0.0, d_U_blk.ptr,
                    jb * v) != CUBLAS_STATUS_SUCCESS) {
          return 6;
        }
        const unsigned nb = static_cast<unsigned>(Size(o_act) * Size(jb) * Size(v));
        k_rhf_g_to_t<<<nb, 256>>>(0, j0, o_act, jb, o_act, v, false, d_eo.ptr, d_ev.ptr,
                                  d_U_blk.ptr, d_T_blk.ptr, nullptr, d_err.ptr);
        k_rhf_t_to_u_plain<<<nb, 256>>>(o_act, jb, v, d_T_blk.ptr, d_U_blk.ptr);
        const int k_dim = static_cast<int>(Size(jb) * vv);
        if (rm_gemm(h, false, true, o_act, o_act, k_dim, -1.0, d_T_blk.ptr, k_dim, d_U_blk.ptr,
                    k_dim, 1.0, d_bar_Fo.ptr, o_act) != CUBLAS_STATUS_SUCCESS) {
          return 6;
        }
      }
    } else {
      // Regime B2: ultra-bounded O(v^2) fallback for extreme budget limits.
      DeviceBuf d_Ti;
      if (d_Ti.alloc(vv) != cudaSuccess) return 6;
      for (int j = 0; j < o_act; ++j) {
        const double* Bj = d_B_ovq + Size(j) * vq;
        for (int i = 0; i < o_act; ++i) {
          const double* Bi = d_B_ovq + Size(i) * vq;
          if (rm_gemm(h, false, true, v, v, q, 1.0, Bi, q, Bj, q, 0.0, d_U_blk.ptr, v) !=
              CUBLAS_STATUS_SUCCESS) {
            return 6;
          }
          k_rhf_g_to_t<<<static_cast<unsigned>(v), 256>>>(i, j, 1, 1, o_act, v, false, d_eo.ptr,
                                                          d_ev.ptr, d_U_blk.ptr, d_Ti.ptr, nullptr,
                                                          d_err.ptr);
          for (int k = i; k < o_act; ++k) {
            const double* Bk = d_B_ovq + Size(k) * vq;
            if (rm_gemm(h, false, true, v, v, q, 1.0, Bk, q, Bj, q, 0.0, d_U_blk.ptr, v) !=
                CUBLAS_STATUS_SUCCESS) {
              return 6;
            }
            k_rhf_g_to_t<<<static_cast<unsigned>(v), 256>>>(k, j, 1, 1, o_act, v, false, d_eo.ptr,
                                                            d_ev.ptr, d_U_blk.ptr, d_T_blk.ptr,
                                                            nullptr, d_err.ptr);
            k_rhf_t_to_u_plain<<<static_cast<unsigned>(v), 256>>>(1, 1, v, d_T_blk.ptr,
                                                                  d_U_blk.ptr);
            if (launch_accum_occupied_fock_dot(vv, d_Ti.ptr, d_U_blk.ptr, o_act, i, k, -1.0,
                                               d_bar_Fo.ptr) != cudaSuccess) {
              return 6;
            }
          }
        }
      }
    }
  }

  if (compute_response) {
    if (has_occ_weights) {
      k_scatter_fock_diag<<<grid_for(Size(o_act) + v), 256>>>(n, o, f, d_diag_o.ptr, d_diag_v.ptr,
                                                              d_fock_a);
    } else {
      k_scatter_fock_full<<<grid_for(Size(o_act) * o_act + vv), 256>>>(n, o, f, d_bar_Fo.ptr,
                                                                       d_bar_Fv.ptr, d_fock_a);
    }
  }

  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 7;
  int h_err = 0;
  double h_e[2] = {0.0, 0.0};
  if (cudaMemcpy(&h_err, d_err.ptr, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(h_e, d_energies.ptr, 2 * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
    return 8;
  }
  if (h_err != 0) return h_err;
  e_aa_out = h_e[0];
  e_ab_out = h_e[1];
  return 0;
}

int run_uhf_same_spin_gpu(cublasHandle_t h, int n, int q, int o, int f, bool compute_response,
                          bool has_occ_weights, Size tile_cap, Size tri_cap, const double* h_occ_w,
                          const double* h_eps, const double* d_B_ovq, double* d_bar_B_ovq,
                          double* d_fock, double& e_ss_out) {
  e_ss_out = 0.0;
  const int o_act = o - f;
  const int v = n - o;
  if (o_act <= 1 || v <= 0) return 0;

  const Size vv = Size(v) * v;
  const Size vq = Size(v) * q;
  const Size n_tri = (Size(o_act) * Size(o_act + 1)) / 2;
  bool store_tri =
      compute_response && !has_occ_weights && (bytes_of_doubles(sat_mul(n_tri, vv)) <= tri_cap);

  const Size max_pairs_in_tile =
      std::max<Size>(1, tile_cap / std::max<Size>(1, 2 * vv * sizeof(double)));
  int b_occ = static_cast<int>(std::floor(std::sqrt(static_cast<double>(max_pairs_in_tile))));
  b_occ = std::max(1, std::min(o_act, b_occ));
  Size tile_elems = Size(b_occ) * Size(b_occ) * vv;
  const bool col_fits_tile = bytes_of_doubles(sat_mul(2 * Size(o_act), vv)) <= tile_cap;
  store_tri = store_tri && col_fits_tile;
  if (compute_response && !has_occ_weights && col_fits_tile && tile_elems < Size(o_act) * vv) {
    tile_elems = Size(o_act) * vv;
  }

  DeviceBuf d_eo, d_ev, d_occ_w, d_T_blk, d_G_blk, d_T_tri;
  DeviceBuf d_bar_Fo, d_bar_Fv, d_diag_o, d_diag_v, d_e_ss;
  DeviceIntBuf d_err;

  if (d_eo.alloc(o_act) != cudaSuccess || d_ev.alloc(v) != cudaSuccess ||
      d_T_blk.alloc(tile_elems) != cudaSuccess || d_G_blk.alloc(tile_elems) != cudaSuccess ||
      d_e_ss.alloc(1) != cudaSuccess || d_err.alloc(1) != cudaSuccess) {
    return 2;
  }
  if (has_occ_weights && d_occ_w.alloc(o_act) != cudaSuccess) return 2;

  if (compute_response) {
    if (has_occ_weights) {
      if (d_diag_o.alloc(o_act) != cudaSuccess || d_diag_v.alloc(v) != cudaSuccess) return 2;
    } else {
      if (d_bar_Fo.alloc(Size(o_act) * o_act) != cudaSuccess || d_bar_Fv.alloc(vv) != cudaSuccess) {
        return 2;
      }
    }
  }
  if (store_tri && d_T_tri.alloc(n_tri * vv, true) != cudaSuccess) {
    (void)cudaGetLastError();
    d_T_tri.reset();
    store_tri = false;
  }

  if (cudaMemcpy(d_eo.ptr, h_eps + f, Size(o_act) * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemcpy(d_ev.ptr, h_eps + o, Size(v) * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemset(d_e_ss.ptr, 0, sizeof(double)) != cudaSuccess ||
      cudaMemset(d_err.ptr, 0, sizeof(int)) != cudaSuccess) {
    return 3;
  }
  if (has_occ_weights && cudaMemcpy(d_occ_w.ptr, h_occ_w, Size(o_act) * sizeof(double),
                                    cudaMemcpyHostToDevice) != cudaSuccess) {
    return 3;
  }
  if (compute_response) {
    if (has_occ_weights) {
      if (cudaMemset(d_diag_o.ptr, 0, Size(o_act) * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_diag_v.ptr, 0, Size(v) * sizeof(double)) != cudaSuccess) {
        return 3;
      }
    } else {
      if (cudaMemset(d_bar_Fo.ptr, 0, Size(o_act) * o_act * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_Fv.ptr, 0, vv * sizeof(double)) != cudaSuccess) {
        return 3;
      }
    }
  }

  for (int i0 = 0; i0 < o_act; i0 += b_occ) {
    const int bi = std::min(b_occ, o_act - i0);
    const double* Bi = d_B_ovq + Size(i0) * vq;
    double* bBi = compute_response ? (d_bar_B_ovq + Size(i0) * vq) : nullptr;
    for (int j0 = i0; j0 < o_act; j0 += b_occ) {
      const int bj = std::min(b_occ, o_act - j0);
      const bool off_diag = (i0 < j0);
      if (!off_diag && bi == 1) continue;  // Single diagonal pair (i, i) has Tss == 0.
      const double* Bj = d_B_ovq + Size(j0) * vq;
      double* bBj = compute_response ? (d_bar_B_ovq + Size(j0) * vq) : nullptr;
      const int m_dim = bi * v;
      const int n_dim = bj * v;

      if (rm_gemm(h, false, true, m_dim, n_dim, q, 1.0, Bi, q, Bj, q, 0.0, d_G_blk.ptr, n_dim) !=
          CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      const unsigned nblocks = static_cast<unsigned>(Size(bi) * Size(bj) * Size(v));
      k_uhf_ss_g_to_tss<<<nblocks, 256>>>(i0, j0, bi, bj, o_act, v, store_tri, d_eo.ptr, d_ev.ptr,
                                          d_G_blk.ptr, d_T_blk.ptr, d_T_tri.ptr, d_err.ptr);
      k_uhf_ss_energy_and_w<<<nblocks, 256>>>(i0, j0, bi, bj, v, off_diag, compute_response,
                                              has_occ_weights, d_occ_w.ptr, d_T_blk.ptr,
                                              d_G_blk.ptr, d_e_ss.ptr, d_diag_o.ptr, d_diag_v.ptr);

      if (compute_response) {
        const double* p_wT = has_occ_weights ? d_G_blk.ptr : d_T_blk.ptr;
        if (rm_gemm(h, false, false, m_dim, q, n_dim, 2.0, p_wT, n_dim, Bj, q, 1.0, bBi, q) !=
            CUBLAS_STATUS_SUCCESS) {
          return 5;
        }
        if (off_diag) {
          if (rm_gemm(h, true, false, n_dim, q, m_dim, 2.0, p_wT, n_dim, Bi, q, 1.0, bBj, q) !=
              CUBLAS_STATUS_SUCCESS) {
            return 5;
          }
        }
        if (!has_occ_weights) {
          const double fv_scale = off_diag ? 1.0 : 0.5;
          for (int i_loc = 0; i_loc < bi; ++i_loc) {
            const Size off = Size(i_loc) * v * n_dim;
            if (rm_gemm(h, false, true, v, v, n_dim, fv_scale, d_T_blk.ptr + off, n_dim,
                        d_T_blk.ptr + off, n_dim, 1.0, d_bar_Fv.ptr, v) != CUBLAS_STATUS_SUCCESS) {
              return 5;
            }
          }
        }
      }
    }
  }

  if (compute_response && !has_occ_weights) {
    const int k_jb_cap =
        std::max(1, static_cast<int>(std::min<Size>(o_act, Size(INT_MAX) / std::max<Size>(1, vv))));
    if (store_tri && col_fits_tile) {
      const int jb_max =
          std::max(1, std::min(k_jb_cap, static_cast<int>(tile_elems / (Size(o_act) * vv))));
      for (int j0 = 0; j0 < o_act; j0 += jb_max) {
        const int jb = std::min(jb_max, o_act - j0);
        const unsigned nb = static_cast<unsigned>(Size(o_act) * Size(jb) * Size(v));
        k_uhf_ss_gather_tri_cols<<<nb, 256>>>(o_act, j0, jb, v, d_T_tri.ptr, d_T_blk.ptr);
        const int k_dim = static_cast<int>(Size(jb) * vv);
        if (rm_gemm(h, false, true, o_act, o_act, k_dim, -0.5, d_T_blk.ptr, k_dim, d_T_blk.ptr,
                    k_dim, 1.0, d_bar_Fo.ptr, o_act) != CUBLAS_STATUS_SUCCESS) {
          return 6;
        }
      }
    } else if (col_fits_tile) {
      const int jb_max =
          std::max(1, std::min(k_jb_cap, static_cast<int>(tile_elems / (Size(o_act) * vv))));
      for (int j0 = 0; j0 < o_act; j0 += jb_max) {
        const int jb = std::min(jb_max, o_act - j0);
        const double* Bj = d_B_ovq + Size(j0) * vq;
        if (rm_gemm(h, false, true, o_act * v, jb * v, q, 1.0, d_B_ovq, q, Bj, q, 0.0, d_G_blk.ptr,
                    jb * v) != CUBLAS_STATUS_SUCCESS) {
          return 6;
        }
        const unsigned nb = static_cast<unsigned>(Size(o_act) * Size(jb) * Size(v));
        k_uhf_ss_g_to_tss<<<nb, 256>>>(0, j0, o_act, jb, o_act, v, false, d_eo.ptr, d_ev.ptr,
                                       d_G_blk.ptr, d_T_blk.ptr, nullptr, d_err.ptr);
        const int k_dim = static_cast<int>(Size(jb) * vv);
        if (rm_gemm(h, false, true, o_act, o_act, k_dim, -0.5, d_T_blk.ptr, k_dim, d_T_blk.ptr,
                    k_dim, 1.0, d_bar_Fo.ptr, o_act) != CUBLAS_STATUS_SUCCESS) {
          return 6;
        }
      }
    } else {
      DeviceBuf d_Ti;
      if (d_Ti.alloc(vv) != cudaSuccess) return 6;
      for (int j = 0; j < o_act; ++j) {
        const double* Bj = d_B_ovq + Size(j) * vq;
        for (int i = 0; i < o_act; ++i) {
          if (i == j) continue;
          const double* Bi = d_B_ovq + Size(i) * vq;
          if (rm_gemm(h, false, true, v, v, q, 1.0, Bi, q, Bj, q, 0.0, d_G_blk.ptr, v) !=
              CUBLAS_STATUS_SUCCESS) {
            return 6;
          }
          k_uhf_ss_g_to_tss<<<static_cast<unsigned>(v), 256>>>(i, j, 1, 1, o_act, v, false,
                                                               d_eo.ptr, d_ev.ptr, d_G_blk.ptr,
                                                               d_Ti.ptr, nullptr, d_err.ptr);
          for (int k = i; k < o_act; ++k) {
            if (k == j) continue;
            const double* pTk = d_Ti.ptr;
            if (k != i) {
              const double* Bk = d_B_ovq + Size(k) * vq;
              if (rm_gemm(h, false, true, v, v, q, 1.0, Bk, q, Bj, q, 0.0, d_G_blk.ptr, v) !=
                  CUBLAS_STATUS_SUCCESS) {
                return 6;
              }
              k_uhf_ss_g_to_tss<<<static_cast<unsigned>(v), 256>>>(k, j, 1, 1, o_act, v, false,
                                                                   d_eo.ptr, d_ev.ptr, d_G_blk.ptr,
                                                                   d_T_blk.ptr, nullptr, d_err.ptr);
              pTk = d_T_blk.ptr;
            }
            if (launch_accum_occupied_fock_dot(vv, d_Ti.ptr, pTk, o_act, i, k, -0.5,
                                               d_bar_Fo.ptr) != cudaSuccess) {
              return 6;
            }
          }
        }
      }
    }
  }

  if (compute_response) {
    if (has_occ_weights) {
      k_scatter_fock_diag<<<grid_for(Size(o_act) + v), 256>>>(n, o, f, d_diag_o.ptr, d_diag_v.ptr,
                                                              d_fock);
    } else {
      k_scatter_fock_full<<<grid_for(Size(o_act) * o_act + vv), 256>>>(n, o, f, d_bar_Fo.ptr,
                                                                       d_bar_Fv.ptr, d_fock);
    }
  }

  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 7;
  int h_err = 0;
  double h_e = 0.0;
  if (cudaMemcpy(&h_err, d_err.ptr, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(&h_e, d_e_ss.ptr, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
    return 8;
  }
  if (h_err != 0) return h_err;
  e_ss_out = h_e;
  return 0;
}

int run_uhf_opp_spin_gpu(cublasHandle_t h, int n, int q, int oa, int fa, int ob, int fb,
                         bool compute_response, bool has_occ_weights, Size tile_cap, Size tri_cap,
                         const double* h_occ_w_a, const double* h_occ_w_b, const double* h_eps_a,
                         const double* h_eps_b, const double* d_Ba_ovq, const double* d_Bb_ovq,
                         double* d_bar_Ba_ovq, double* d_bar_Bb_ovq, double* d_fock_a,
                         double* d_fock_b, double& e_ab_out) {
  e_ab_out = 0.0;
  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  const int va = n - oa;
  const int vb = n - ob;
  if (oa_act <= 0 || ob_act <= 0 || va <= 0 || vb <= 0) return 0;

  const Size vab = Size(va) * vb;
  const Size vaq = Size(va) * q;
  const Size vbq = Size(vb) * q;
  bool store_all =
      compute_response && !has_occ_weights &&
      (bytes_of_doubles(sat_mul(Size(oa_act) * Size(ob_act), vab)) <= tri_cap);

  // Prefer keeping bi_max == oa_act so bar_Foa can be computed in-flight with zero transpose.
  const Size max_pairs = std::max<Size>(1, tile_cap / std::max<Size>(1, 2 * vab * sizeof(double)));
  const int k_vab_cap =
      std::max(1, static_cast<int>(std::min<Size>(ob_act, Size(INT_MAX) / std::max<Size>(1, vab))));
  int bi_max = 1;
  int bj_max = 1;
  if (max_pairs >= Size(oa_act)) {
    bi_max = oa_act;
    bj_max = std::max(1, std::min(k_vab_cap, static_cast<int>(max_pairs / Size(oa_act))));
  } else {
    bi_max = std::max(1, std::min(oa_act, static_cast<int>(max_pairs)));
    bj_max = 1;
  }
  const Size tile_elems = Size(bi_max) * Size(bj_max) * vab;

  DeviceBuf d_eoa, d_eva, d_eob, d_evb, d_wa, d_wb;
  DeviceBuf d_T_blk, d_G_blk, d_T_all;
  DeviceBuf d_bar_Foa, d_bar_Fob, d_bar_Fva, d_bar_Fvb;
  DeviceBuf d_diag_oa, d_diag_ob, d_diag_va, d_diag_vb, d_e_ab;
  DeviceIntBuf d_err;

  if (d_eoa.alloc(oa_act) != cudaSuccess || d_eva.alloc(va) != cudaSuccess ||
      d_eob.alloc(ob_act) != cudaSuccess || d_evb.alloc(vb) != cudaSuccess ||
      d_T_blk.alloc(tile_elems) != cudaSuccess || d_G_blk.alloc(tile_elems) != cudaSuccess ||
      d_e_ab.alloc(1) != cudaSuccess || d_err.alloc(1) != cudaSuccess) {
    return 2;
  }
  if (has_occ_weights &&
      (d_wa.alloc(oa_act) != cudaSuccess || d_wb.alloc(ob_act) != cudaSuccess)) {
    return 2;
  }

  if (compute_response) {
    if (has_occ_weights) {
      if (d_diag_oa.alloc(oa_act) != cudaSuccess || d_diag_ob.alloc(ob_act) != cudaSuccess ||
          d_diag_va.alloc(va) != cudaSuccess || d_diag_vb.alloc(vb) != cudaSuccess) {
        return 2;
      }
    } else {
      if (d_bar_Foa.alloc(Size(oa_act) * oa_act) != cudaSuccess ||
          d_bar_Fob.alloc(Size(ob_act) * ob_act) != cudaSuccess ||
          d_bar_Fva.alloc(Size(va) * va) != cudaSuccess ||
          d_bar_Fvb.alloc(Size(vb) * vb) != cudaSuccess) {
        return 2;
      }
    }
  }
  if (store_all && d_T_all.alloc(Size(oa_act) * Size(ob_act) * vab, true) != cudaSuccess) {
    (void)cudaGetLastError();
    d_T_all.reset();
    store_all = false;
  }

  if (cudaMemcpy(d_eoa.ptr, h_eps_a + fa, Size(oa_act) * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemcpy(d_eva.ptr, h_eps_a + oa, Size(va) * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemcpy(d_eob.ptr, h_eps_b + fb, Size(ob_act) * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemcpy(d_evb.ptr, h_eps_b + ob, Size(vb) * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemset(d_e_ab.ptr, 0, sizeof(double)) != cudaSuccess ||
      cudaMemset(d_err.ptr, 0, sizeof(int)) != cudaSuccess) {
    return 3;
  }
  if (has_occ_weights &&
      (cudaMemcpy(d_wa.ptr, h_occ_w_a, Size(oa_act) * sizeof(double), cudaMemcpyHostToDevice) !=
           cudaSuccess ||
       cudaMemcpy(d_wb.ptr, h_occ_w_b, Size(ob_act) * sizeof(double), cudaMemcpyHostToDevice) !=
           cudaSuccess)) {
    return 3;
  }
  if (compute_response) {
    if (has_occ_weights) {
      if (cudaMemset(d_diag_oa.ptr, 0, Size(oa_act) * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_diag_ob.ptr, 0, Size(ob_act) * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_diag_va.ptr, 0, Size(va) * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_diag_vb.ptr, 0, Size(vb) * sizeof(double)) != cudaSuccess) {
        return 3;
      }
    } else {
      if (cudaMemset(d_bar_Foa.ptr, 0, Size(oa_act) * oa_act * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_Fob.ptr, 0, Size(ob_act) * ob_act * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_Fva.ptr, 0, Size(va) * va * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_Fvb.ptr, 0, Size(vb) * vb * sizeof(double)) != cudaSuccess) {
        return 3;
      }
    }
  }

  const bool foa_done_in_loop = compute_response && !has_occ_weights && (bi_max == oa_act);

  for (int j0 = 0; j0 < ob_act; j0 += bj_max) {
    const int bj = std::min(bj_max, ob_act - j0);
    const double* Bj = d_Bb_ovq + Size(j0) * vbq;
    double* bBj = compute_response ? (d_bar_Bb_ovq + Size(j0) * vbq) : nullptr;
    const int n_dim = bj * vb;

    for (int i0 = 0; i0 < oa_act; i0 += bi_max) {
      const int bi = std::min(bi_max, oa_act - i0);
      const double* Bi = d_Ba_ovq + Size(i0) * vaq;
      double* bBi = compute_response ? (d_bar_Ba_ovq + Size(i0) * vaq) : nullptr;
      const int m_dim = bi * va;

      if (rm_gemm(h, false, true, m_dim, n_dim, q, 1.0, Bi, q, Bj, q, 0.0, d_G_blk.ptr, n_dim) !=
          CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      const unsigned nblocks = static_cast<unsigned>(Size(bi) * Size(bj) * Size(va));
      k_uhf_ab_g_to_t<<<nblocks, 256>>>(i0, j0, bi, bj, ob_act, va, vb, compute_response,
                                        has_occ_weights, store_all, d_wa.ptr, d_wb.ptr, d_eoa.ptr,
                                        d_eob.ptr, d_eva.ptr, d_evb.ptr, d_G_blk.ptr, d_T_blk.ptr,
                                        d_T_all.ptr, d_e_ab.ptr, d_diag_oa.ptr, d_diag_ob.ptr,
                                        d_diag_va.ptr, d_diag_vb.ptr, d_err.ptr);

      if (compute_response) {
        const double* p_wT = has_occ_weights ? d_G_blk.ptr : d_T_blk.ptr;
        if (rm_gemm(h, false, false, m_dim, q, n_dim, 2.0, p_wT, n_dim, Bj, q, 1.0, bBi, q) !=
                CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, true, false, n_dim, q, m_dim, 2.0, p_wT, n_dim, Bi, q, 1.0, bBj, q) !=
                CUBLAS_STATUS_SUCCESS) {
          return 5;
        }
        if (!has_occ_weights) {
          for (int i_loc = 0; i_loc < bi; ++i_loc) {
            const Size off = Size(i_loc) * va * n_dim;
            if (rm_gemm(h, false, true, va, va, n_dim, 1.0, d_T_blk.ptr + off, n_dim,
                        d_T_blk.ptr + off, n_dim, 1.0, d_bar_Fva.ptr, va) !=
                CUBLAS_STATUS_SUCCESS) {
              return 5;
            }
          }
          for (int j_loc = 0; j_loc < bj; ++j_loc) {
            const Size off = Size(j_loc) * vb;
            if (rm_gemm(h, true, false, vb, vb, m_dim, 1.0, d_T_blk.ptr + off, n_dim,
                        d_T_blk.ptr + off, n_dim, 1.0, d_bar_Fvb.ptr, vb) !=
                CUBLAS_STATUS_SUCCESS) {
              return 5;
            }
          }
          if (foa_done_in_loop) {
            const int k_dim = static_cast<int>(Size(va) * n_dim);
            if (rm_gemm(h, false, true, oa_act, oa_act, k_dim, -1.0, d_T_blk.ptr, k_dim,
                        d_T_blk.ptr, k_dim, 1.0, d_bar_Foa.ptr, oa_act) != CUBLAS_STATUS_SUCCESS) {
              return 5;
            }
          }
        }
      }
    }
  }

  if (compute_response && !has_occ_weights) {
    if (store_all) {
      // bar_Fob[j, l] -= sum_i T_all[i, j, ab] * T_all[i, l, ab]
      for (int i = 0; i < oa_act; ++i) {
        const double* Ti = d_T_all.ptr + Size(i) * ob_act * vab;
        if (rm_gemm(h, false, true, ob_act, ob_act, static_cast<int>(vab), -1.0, Ti,
                    static_cast<int>(vab), Ti, static_cast<int>(vab), 1.0, d_bar_Fob.ptr,
                    ob_act) != CUBLAS_STATUS_SUCCESS) {
          return 6;
        }
      }
      if (!foa_done_in_loop) {
        // Gather j-columns from d_T_all into d_T_blk if it fits, else per-pair fallback.
        if (tile_elems >= Size(oa_act) * vab) {
          const int jb_g = std::max(
              1, std::min(k_vab_cap, static_cast<int>(tile_elems / (Size(oa_act) * vab))));
          for (int j0 = 0; j0 < ob_act; j0 += jb_g) {
            const int jb = std::min(jb_g, ob_act - j0);
            const Size total = Size(oa_act) * Size(jb) * vab;
            k_uhf_ab_gather_j_col<<<grid_for(total), 256>>>(oa_act, ob_act, j0, jb, vab,
                                                            d_T_all.ptr, d_T_blk.ptr);
            const int k_dim = static_cast<int>(Size(jb) * vab);
            if (rm_gemm(h, false, true, oa_act, oa_act, k_dim, -1.0, d_T_blk.ptr, k_dim,
                        d_T_blk.ptr, k_dim, 1.0, d_bar_Foa.ptr, oa_act) != CUBLAS_STATUS_SUCCESS) {
              return 6;
            }
          }
        } else {
          for (int j = 0; j < ob_act; ++j) {
            for (int i = 0; i < oa_act; ++i) {
              const double* Ti = d_T_all.ptr + (Size(i) * ob_act + j) * vab;
              for (int k = i; k < oa_act; ++k) {
                const double* Tk = d_T_all.ptr + (Size(k) * ob_act + j) * vab;
                if (launch_accum_occupied_fock_dot(vab, Ti, Tk, oa_act, i, k, -1.0,
                                                   d_bar_Foa.ptr) != cudaSuccess) {
                  return 6;
                }
              }
            }
          }
        }
      }
    } else {
      // Bounded-memory fallback without d_T_all.
      DeviceBuf d_T1;
      if (d_T1.alloc(vab) != cudaSuccess) return 6;
      if (!foa_done_in_loop) {
        for (int j = 0; j < ob_act; ++j) {
          const double* Bj = d_Bb_ovq + Size(j) * vbq;
          for (int i = 0; i < oa_act; ++i) {
            const double* Bi = d_Ba_ovq + Size(i) * vaq;
            if (rm_gemm(h, false, true, va, vb, q, 1.0, Bi, q, Bj, q, 0.0, d_G_blk.ptr, vb) !=
                CUBLAS_STATUS_SUCCESS) {
              return 6;
            }
            k_uhf_ab_g_to_t_plain<<<static_cast<unsigned>(va), 256>>>(
                i, j, 1, 1, va, vb, d_eoa.ptr, d_eob.ptr, d_eva.ptr, d_evb.ptr, d_G_blk.ptr,
                d_T1.ptr, d_err.ptr);
            for (int k = i; k < oa_act; ++k) {
              const double* pTk = d_T1.ptr;
              if (k != i) {
                const double* Bk = d_Ba_ovq + Size(k) * vaq;
                if (rm_gemm(h, false, true, va, vb, q, 1.0, Bk, q, Bj, q, 0.0, d_G_blk.ptr, vb) !=
                    CUBLAS_STATUS_SUCCESS) {
                  return 6;
                }
                k_uhf_ab_g_to_t_plain<<<static_cast<unsigned>(va), 256>>>(
                    k, j, 1, 1, va, vb, d_eoa.ptr, d_eob.ptr, d_eva.ptr, d_evb.ptr, d_G_blk.ptr,
                    d_T_blk.ptr, d_err.ptr);
                pTk = d_T_blk.ptr;
              }
              if (launch_accum_occupied_fock_dot(vab, d_T1.ptr, pTk, oa_act, i, k, -1.0,
                                                 d_bar_Foa.ptr) != cudaSuccess) {
                return 6;
              }
            }
          }
        }
      }
      // bar_Fob: if [ob_act, vab] fits in tile_elems, compute one giant GEMM per i!
      if (tile_elems >= Size(ob_act) * vab) {
        const int k_oa_vab_cap = std::max(
            1, static_cast<int>(std::min<Size>(oa_act, Size(INT_MAX) / std::max<Size>(1, vab))));
        const int ib_fob = std::max(
            1, std::min(k_oa_vab_cap, static_cast<int>(tile_elems / (Size(ob_act) * vab))));
        for (int i0 = 0; i0 < oa_act; i0 += ib_fob) {
          const int bi = std::min(ib_fob, oa_act - i0);
          const double* Bi = d_Ba_ovq + Size(i0) * vaq;
          // Compute G_trans[ob_act * vb, bi * va] = Bb @ Bi^T -> row j is [vb, bi*va] contiguous!
          if (rm_gemm(h, false, true, ob_act * vb, bi * va, q, 1.0, d_Bb_ovq, q, Bi, q, 0.0,
                      d_G_blk.ptr, bi * va) != CUBLAS_STATUS_SUCCESS) {
            return 6;
          }
          // Swap (a, b) roles in k_uhf_ab_g_to_t_plain: rows are (j, b), cols are (i_loc, a)!
          const unsigned nb = static_cast<unsigned>(Size(ob_act) * Size(bi) * Size(vb));
          k_uhf_ab_g_to_t_plain<<<nb, 256>>>(0, i0, ob_act, bi, vb, va, d_eob.ptr, d_eoa.ptr,
                                             d_evb.ptr, d_eva.ptr, d_G_blk.ptr, d_T_blk.ptr,
                                             d_err.ptr);
          const int k_dim = static_cast<int>(Size(bi) * vab);
          if (rm_gemm(h, false, true, ob_act, ob_act, k_dim, -1.0, d_T_blk.ptr, k_dim, d_T_blk.ptr,
                      k_dim, 1.0, d_bar_Fob.ptr, ob_act) != CUBLAS_STATUS_SUCCESS) {
            return 6;
          }
        }
      } else {
        for (int i = 0; i < oa_act; ++i) {
          const double* Bi = d_Ba_ovq + Size(i) * vaq;
          for (int j = 0; j < ob_act; ++j) {
            const double* Bj = d_Bb_ovq + Size(j) * vbq;
            if (rm_gemm(h, false, true, va, vb, q, 1.0, Bi, q, Bj, q, 0.0, d_G_blk.ptr, vb) !=
                CUBLAS_STATUS_SUCCESS) {
              return 6;
            }
            k_uhf_ab_g_to_t_plain<<<static_cast<unsigned>(va), 256>>>(
                i, j, 1, 1, va, vb, d_eoa.ptr, d_eob.ptr, d_eva.ptr, d_evb.ptr, d_G_blk.ptr,
                d_T1.ptr, d_err.ptr);
            for (int l = j; l < ob_act; ++l) {
              const double* pTl = d_T1.ptr;
              if (l != j) {
                const double* Bl = d_Bb_ovq + Size(l) * vbq;
                if (rm_gemm(h, false, true, va, vb, q, 1.0, Bi, q, Bl, q, 0.0, d_G_blk.ptr, vb) !=
                    CUBLAS_STATUS_SUCCESS) {
                  return 6;
                }
                k_uhf_ab_g_to_t_plain<<<static_cast<unsigned>(va), 256>>>(
                    i, l, 1, 1, va, vb, d_eoa.ptr, d_eob.ptr, d_eva.ptr, d_evb.ptr, d_G_blk.ptr,
                    d_T_blk.ptr, d_err.ptr);
                pTl = d_T_blk.ptr;
              }
              if (launch_accum_occupied_fock_dot(vab, d_T1.ptr, pTl, ob_act, j, l, -1.0,
                                                 d_bar_Fob.ptr) != cudaSuccess) {
                return 6;
              }
            }
          }
        }
      }
    }
  }

  if (compute_response) {
    if (has_occ_weights) {
      k_scatter_fock_diag<<<grid_for(Size(oa_act) + va), 256>>>(n, oa, fa, d_diag_oa.ptr,
                                                                d_diag_va.ptr, d_fock_a);
      k_scatter_fock_diag<<<grid_for(Size(ob_act) + vb), 256>>>(n, ob, fb, d_diag_ob.ptr,
                                                                d_diag_vb.ptr, d_fock_b);
    } else {
      k_scatter_fock_full<<<grid_for(Size(oa_act) * oa_act + Size(va) * va), 256>>>(
          n, oa, fa, d_bar_Foa.ptr, d_bar_Fva.ptr, d_fock_a);
      k_scatter_fock_full<<<grid_for(Size(ob_act) * ob_act + Size(vb) * vb), 256>>>(
          n, ob, fb, d_bar_Fob.ptr, d_bar_Fvb.ptr, d_fock_b);
    }
  }

  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 7;
  int h_err = 0;
  double h_e = 0.0;
  if (cudaMemcpy(&h_err, d_err.ptr, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(&h_e, d_e_ab.ptr, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
    return 8;
  }
  if (h_err != 0) return h_err;
  e_ab_out = h_e;
  return 0;
}

#endif

}  // namespace

extern "C" {

#ifdef GQC_SINK_HOST_REFERENCE

int gqc_f12_ordinary_df_mp2_host_reference(
    int same_spin_int, int compute_response_int, int has_occ_weights_int, int n_orb, int n_aux,
    int oa, int fa, int ob, int fb, std::uint64_t budget_bytes, const double* occ_w_a,
    const double* occ_w_b, const double* coeff_a, const double* coeff_b, const double* eps_a,
    const double* eps_b, const double* ao_coul, const double* whitener, double* spin_energies,
    double* fock_a, double* fock_b, double* bar_coeff_a, double* bar_coeff_b, double* bar_ao_coul,
    double* bar_metric) {
  try {
  const bool same_spin = (same_spin_int != 0);
  const bool compute_response = (compute_response_int != 0);
  const bool has_occ_weights = (has_occ_weights_int != 0);

  if (n_orb < 2 || n_aux < 1 || oa < 0 || oa >= n_orb || ob < 0 || ob >= n_orb || fa < 0 ||
      fa > oa || fb < 0 || fb > ob || !coeff_a || !eps_a || !ao_coul || !whitener ||
      !spin_energies) {
    return -1;
  }
  if (same_spin && (oa != ob || fa != fb)) return -1;
  if (!same_spin && (!coeff_b || !eps_b)) return -1;

  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  if (has_occ_weights) {
    if (oa_act > 0 && !occ_w_a) return -1;
    if (!same_spin && ob_act > 0 && !occ_w_b) return -1;
  }
  if (compute_response) {
    if (!fock_a || !bar_coeff_a || !bar_ao_coul || !bar_metric) return -1;
    if (!same_spin && (!fock_b || !bar_coeff_b)) return -1;
  }

  const Size nn = Size(n_orb) * n_orb;
  const Size qq = Size(n_aux) * n_aux;
  const Size qnn = Size(n_aux) * nn;
  if (!all_finite_host(coeff_a, nn) || !all_finite_host(eps_a, n_orb) ||
      !all_finite_host(ao_coul, qnn) || !all_finite_host(whitener, qq)) {
    return -2;
  }
  if (!same_spin && (!all_finite_host(coeff_b, nn) || !all_finite_host(eps_b, n_orb))) {
    return -2;
  }
  if (has_occ_weights) {
    if (oa_act > 0 && !all_finite_host(occ_w_a, oa_act)) return -2;
    if (!same_spin && ob_act > 0 && !all_finite_host(occ_w_b, ob_act)) return -2;
  }

  spin_energies[0] = 0.0;
  spin_energies[1] = 0.0;
  spin_energies[2] = 0.0;
  if (compute_response) {
    std::fill_n(fock_a, nn, 0.0);
    if (fock_b && fock_b != fock_a) std::fill_n(fock_b, nn, 0.0);
    std::fill_n(bar_coeff_a, nn, 0.0);
    if (bar_coeff_b && bar_coeff_b != bar_coeff_a) std::fill_n(bar_coeff_b, nn, 0.0);
    std::fill_n(bar_ao_coul, qnn, 0.0);
    std::fill_n(bar_metric, qq, 0.0);
  }

  const Size eff_budget = effective_device_budget(budget_bytes);

  if (same_spin) {
    if (oa_act == 0) return 0;
    const int o = oa;
    const int f = fa;
    const int o_act = oa_act;
    const int v = n_orb - o;
    const Size vv = Size(v) * v;
    const Size vq = Size(v) * n_aux;

    std::vector<double> B_qov, B_ovq;
    build_ordinary_factors_host(n_orb, n_aux, o, f, coeff_a, ao_coul, whitener, B_qov, B_ovq);
    std::vector<double> bar_B_ovq;
    if (compute_response) bar_B_ovq.assign(Size(o_act) * vq, 0.0);

    const double* eo = eps_a + f;
    const double* ev = eps_a + o;
    const Size n_tri = (Size(o_act) * Size(o_act + 1)) / 2;
    const bool store_tri =
        compute_response && !has_occ_weights && (n_tri * vv * sizeof(double) <= eff_budget / 2);

    std::vector<double> G(vv), T(vv), U(vv);
    std::vector<double> bar_Fv, T_tri;
    if (compute_response && !has_occ_weights) {
      bar_Fv.assign(vv, 0.0);
      if (store_tri) T_tri.assign(n_tri * vv, 0.0);
    }

    double e_aa = 0.0;
    double e_ab = 0.0;
    for (int i = 0; i < o_act; ++i) {
      const double* Bi = B_ovq.data() + Size(i) * vq;
      double* bBi = compute_response ? (bar_B_ovq.data() + Size(i) * vq) : nullptr;
      for (int j = i; j < o_act; ++j) {
        const double* Bj = B_ovq.data() + Size(j) * vq;
        double* bBj = compute_response ? (bar_B_ovq.data() + Size(j) * vq) : nullptr;
        rm_gemm_host(false, true, v, v, n_aux, 1.0, Bi, n_aux, Bj, n_aux, 0.0, G.data(), v);

        const double eij = eo[i] + eo[j];
        const double wij = has_occ_weights ? 0.5 * (occ_w_a[i] + occ_w_a[j]) : 1.0;
        for (int a = 0; a < v; ++a) {
          const double ea = ev[a];
          for (int b = 0; b < v; ++b) {
            const double d = eij - ea - ev[b];
            if (!std::isfinite(d) || d == 0.0) return -4;
            T[Size(a) * v + b] = G[Size(a) * v + b] / d;
          }
        }
        double dot_tg = 0.0;
        double dot_ttg = 0.0;
        double dot_tu = 0.0;
        for (int a = 0; a < v; ++a) {
          double row_tu = 0.0;
          for (int b = 0; b < v; ++b) {
            const double tab = T[Size(a) * v + b];
            const double tba = T[Size(b) * v + a];
            const double uab = 2.0 * tab - tba;
            U[Size(a) * v + b] = uab;
            const double gab = G[Size(a) * v + b];
            dot_tg += tab * gab;
            dot_ttg += tba * gab;
            const double tu = tab * uab;
            row_tu += tu;
            if (compute_response && has_occ_weights && i < j) {
              fock_a[Size(o + b) * n_orb + (o + b)] += wij * tu;
            }
          }
          dot_tu += row_tu;
          if (compute_response && has_occ_weights) {
            fock_a[Size(o + a) * n_orb + (o + a)] += wij * row_tu;
          }
        }

        if (i == j) {
          e_ab += wij * dot_tg;
        } else {
          e_ab += 2.0 * wij * dot_tg;
          e_aa += wij * (dot_tg - dot_ttg);
        }

        if (compute_response) {
          const double b_scale = 2.0 * wij;
          rm_gemm_host(false, false, v, n_aux, v, b_scale, U.data(), v, Bj, n_aux, 1.0, bBi, n_aux);
          if (i < j) {
            rm_gemm_host(true, false, v, n_aux, v, b_scale, U.data(), v, Bi, n_aux, 1.0, bBj,
                         n_aux);
          }
          if (has_occ_weights) {
            fock_a[Size(f + i) * n_orb + (f + i)] -= wij * dot_tu;
            if (i < j) fock_a[Size(f + j) * n_orb + (f + j)] -= wij * dot_tu;
          } else {
            rm_gemm_host(false, true, v, v, v, 1.0, T.data(), v, U.data(), v, 1.0, bar_Fv.data(), v);
            if (i < j) {
              rm_gemm_host(true, false, v, v, v, 1.0, T.data(), v, U.data(), v, 1.0, bar_Fv.data(),
                           v);
            }
            if (store_tri) {
              std::copy_n(T.data(), vv, T_tri.data() + tri_index(i, j, o_act) * vv);
            }
          }
        }
      }
    }

    spin_energies[0] = e_aa;
    spin_energies[1] = e_ab;
    spin_energies[2] = e_aa;

    if (compute_response) {
      if (!has_occ_weights) {
        std::vector<double> bar_Fo(Size(o_act) * o_act, 0.0);
        if (store_tri) {
          std::vector<double> Tcol(Size(o_act) * vv), Ucol(Size(o_act) * vv);
          for (int j = 0; j < o_act; ++j) {
            for (int i = 0; i < o_act; ++i) {
              double* t_dst = Tcol.data() + Size(i) * vv;
              double* u_dst = Ucol.data() + Size(i) * vv;
              if (i <= j) {
                const double* src = T_tri.data() + tri_index(i, j, o_act) * vv;
                for (int a = 0; a < v; ++a) {
                  for (int b = 0; b < v; ++b) {
                    const double tab = src[Size(a) * v + b];
                    const double tba = src[Size(b) * v + a];
                    t_dst[Size(a) * v + b] = tab;
                    u_dst[Size(a) * v + b] = 2.0 * tab - tba;
                  }
                }
              } else {
                const double* src = T_tri.data() + tri_index(j, i, o_act) * vv;
                for (int a = 0; a < v; ++a) {
                  for (int b = 0; b < v; ++b) {
                    const double tab = src[Size(b) * v + a];
                    const double tba = src[Size(a) * v + b];
                    t_dst[Size(a) * v + b] = tab;
                    u_dst[Size(a) * v + b] = 2.0 * tab - tba;
                  }
                }
              }
            }
            rm_gemm_host(false, true, o_act, o_act, static_cast<int>(vv), -1.0, Tcol.data(),
                         static_cast<int>(vv), Ucol.data(), static_cast<int>(vv), 1.0,
                         bar_Fo.data(), o_act);
          }
        } else {
          // Bounded-memory fallback when T_tri exceeds budget: O(v^2) scratch.
          std::vector<double> Ti(vv), Uk(vv);
          auto build_t_or_u = [&](int i, int j, bool make_u, double* out) -> int {
            const double* Bi = B_ovq.data() + Size(i) * vq;
            const double* Bj = B_ovq.data() + Size(j) * vq;
            rm_gemm_host(false, true, v, v, n_aux, 1.0, Bi, n_aux, Bj, n_aux, 0.0, G.data(), v);
            const double eij = eo[i] + eo[j];
            for (int a = 0; a < v; ++a) {
              for (int b = 0; b < v; ++b) {
                const double d = eij - ev[a] - ev[b];
                if (!std::isfinite(d) || d == 0.0) return -4;
                T[Size(a) * v + b] = G[Size(a) * v + b] / d;
              }
            }
            if (!make_u) {
              std::copy_n(T.data(), vv, out);
            } else {
              for (int a = 0; a < v; ++a) {
                for (int b = 0; b < v; ++b) {
                  out[Size(a) * v + b] = 2.0 * T[Size(a) * v + b] - T[Size(b) * v + a];
                }
              }
            }
            return 0;
          };
          for (int j = 0; j < o_act; ++j) {
            for (int i = 0; i < o_act; ++i) {
              if (int rc = build_t_or_u(i, j, false, Ti.data()); rc != 0) return rc;
              for (int k = i; k < o_act; ++k) {
                if (int rc = build_t_or_u(k, j, true, Uk.data()); rc != 0) return rc;
                double dot = 0.0;
                for (Size p = 0; p < vv; ++p) dot += Ti[p] * Uk[p];
                bar_Fo[Size(i) * o_act + k] -= dot;
                if (k != i) bar_Fo[Size(k) * o_act + i] -= dot;
              }
            }
          }
        }
        for (int i = 0; i < o_act; ++i) {
          for (int k = 0; k < o_act; ++k) {
            fock_a[Size(f + i) * n_orb + (f + k)] += bar_Fo[Size(i) * o_act + k];
          }
        }
        for (int a = 0; a < v; ++a) {
          for (int c = 0; c < v; ++c) {
            fock_a[Size(o + a) * n_orb + (o + c)] += bar_Fv[Size(a) * v + c];
          }
        }
      }
      if (fock_b && fock_b != fock_a) std::copy_n(fock_a, nn, fock_b);
      pullback_ordinary_factors_host(n_orb, n_aux, o, f, 2.0, coeff_a, ao_coul, whitener, B_qov,
                                     bar_B_ovq, bar_coeff_a, bar_ao_coul, bar_metric);
      if (bar_coeff_b && bar_coeff_b != bar_coeff_a) std::copy_n(bar_coeff_a, nn, bar_coeff_b);
    }
  } else {
    // General UHF dual-spin path (aa, ab, bb)
    std::vector<double> Ba_qov, Ba_ovq, Bb_qov, Bb_ovq;
    if (oa_act > 0) {
      build_ordinary_factors_host(n_orb, n_aux, oa, fa, coeff_a, ao_coul, whitener, Ba_qov, Ba_ovq);
    }
    if (ob_act > 0) {
      build_ordinary_factors_host(n_orb, n_aux, ob, fb, coeff_b, ao_coul, whitener, Bb_qov, Bb_ovq);
    }
    std::vector<double> bar_Ba_ovq, bar_Bb_ovq;
    if (compute_response) {
      if (oa_act > 0) bar_Ba_ovq.assign(Size(oa_act) * Size(n_orb - oa) * n_aux, 0.0);
      if (ob_act > 0) bar_Bb_ovq.assign(Size(ob_act) * Size(n_orb - ob) * n_aux, 0.0);
    }

    double e_aa = 0.0, e_ab = 0.0, e_bb = 0.0;
    if (int rc = run_uhf_same_spin_host(n_orb, n_aux, oa, fa, compute_response, has_occ_weights,
                                        eff_budget, occ_w_a, eps_a, Ba_ovq, bar_Ba_ovq, fock_a,
                                        e_aa);
        rc != 0) {
      return rc;
    }
    if (int rc = run_uhf_opp_spin_host(n_orb, n_aux, oa, fa, ob, fb, compute_response,
                                       has_occ_weights, eff_budget, occ_w_a, occ_w_b, eps_a, eps_b,
                                       Ba_ovq, Bb_ovq, bar_Ba_ovq, bar_Bb_ovq, fock_a, fock_b,
                                       e_ab);
        rc != 0) {
      return rc;
    }
    if (int rc = run_uhf_same_spin_host(n_orb, n_aux, ob, fb, compute_response, has_occ_weights,
                                        eff_budget, occ_w_b, eps_b, Bb_ovq, bar_Bb_ovq, fock_b,
                                        e_bb);
        rc != 0) {
      return rc;
    }

    spin_energies[0] = e_aa;
    spin_energies[1] = e_ab;
    spin_energies[2] = e_bb;

    if (compute_response) {
      if (oa_act > 0) {
        pullback_ordinary_factors_host(n_orb, n_aux, oa, fa, 1.0, coeff_a, ao_coul, whitener,
                                       Ba_qov, bar_Ba_ovq, bar_coeff_a, bar_ao_coul, bar_metric);
      }
      if (ob_act > 0) {
        pullback_ordinary_factors_host(n_orb, n_aux, ob, fb, 1.0, coeff_b, ao_coul, whitener,
                                       Bb_qov, bar_Bb_ovq, bar_coeff_b, bar_ao_coul, bar_metric);
      }
    }
  }

  if (!all_finite_host(spin_energies, 3)) return -5;
  if (compute_response) {
    if (!all_finite_host(fock_a, nn) || !all_finite_host(bar_coeff_a, nn) ||
        !all_finite_host(bar_ao_coul, qnn) || !all_finite_host(bar_metric, qq)) {
      return -5;
    }
    if (!same_spin && (!all_finite_host(fock_b, nn) || !all_finite_host(bar_coeff_b, nn))) {
      return -5;
    }
  }
  return 0;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

#else

int gqc_f12_ordinary_df_mp2_cuda(
    int same_spin_int, int compute_response_int, int has_occ_weights_int, int n_orb, int n_aux,
    int oa, int fa, int ob, int fb, std::uint64_t budget_bytes, const double* occ_w_a,
    const double* occ_w_b, const double* coeff_a, const double* coeff_b, const double* eps_a,
    const double* eps_b, const double* ao_coul, const double* whitener, double* spin_energies,
    double* fock_a, double* fock_b, double* bar_coeff_a, double* bar_coeff_b, double* bar_ao_coul,
    double* bar_metric) {
  try {
  const bool same_spin = (same_spin_int != 0);
  const bool compute_response = (compute_response_int != 0);
  const bool has_occ_weights = (has_occ_weights_int != 0);

  if (n_orb < 2 || n_aux < 1 || oa < 0 || oa >= n_orb || ob < 0 || ob >= n_orb || fa < 0 ||
      fa > oa || fb < 0 || fb > ob || !coeff_a || !eps_a || !ao_coul || !whitener ||
      !spin_energies) {
    return -1;
  }
  if (same_spin && (oa != ob || fa != fb)) return -1;
  if (!same_spin && (!coeff_b || !eps_b)) return -1;

  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  if (has_occ_weights) {
    if (oa_act > 0 && !occ_w_a) return -1;
    if (!same_spin && ob_act > 0 && !occ_w_b) return -1;
  }
  if (compute_response) {
    if (!fock_a || !bar_coeff_a || !bar_ao_coul || !bar_metric) return -1;
    if (!same_spin && (!fock_b || !bar_coeff_b)) return -1;
  }

  const Size nn = Size(n_orb) * n_orb;
  const Size qq = Size(n_aux) * n_aux;
  const Size qnn = Size(n_aux) * nn;
  if (!all_finite_host(coeff_a, nn) || !all_finite_host(eps_a, n_orb) ||
      !all_finite_host(ao_coul, qnn) || !all_finite_host(whitener, qq)) {
    return -2;
  }
  if (!same_spin && (!all_finite_host(coeff_b, nn) || !all_finite_host(eps_b, n_orb))) {
    return -2;
  }
  if (has_occ_weights) {
    if (oa_act > 0 && !all_finite_host(occ_w_a, oa_act)) return -2;
    if (!same_spin && ob_act > 0 && !all_finite_host(occ_w_b, ob_act)) return -2;
  }

  spin_energies[0] = 0.0;
  spin_energies[1] = 0.0;
  spin_energies[2] = 0.0;
  if (compute_response) {
    std::fill_n(fock_a, nn, 0.0);
    if (fock_b && fock_b != fock_a) std::fill_n(fock_b, nn, 0.0);
    std::fill_n(bar_coeff_a, nn, 0.0);
    if (bar_coeff_b && bar_coeff_b != bar_coeff_a) std::fill_n(bar_coeff_b, nn, 0.0);
    std::fill_n(bar_ao_coul, qnn, 0.0);
    std::fill_n(bar_metric, qq, 0.0);
  }

  if (oa_act == 0 && ob_act == 0) return 0;

  // Hard VRAM budget caps: pre-check minimum streaming working set BEFORE allocating,
  // and size optional triangular caches strictly from remaining budget headroom.
  const Size eff_budget = effective_device_budget(budget_bytes);
  const auto rem_cap = [](Size budget, Size used, Size max_cap) -> Size {
    return (budget > used) ? std::min<Size>(budget - used, max_cap) : 0;
  };

  Size build_a_slab_cap = std::min<Size>(eff_budget, Size(512) << 20);
  Size build_b_slab_cap = build_a_slab_cap;
  Size pb_a_slab_cap = build_a_slab_cap;
  Size pb_b_slab_cap = build_a_slab_cap;
  Size tile_cap_main = std::min<Size>(eff_budget, Size(512) << 20);
  Size tri_cap_main = 0;

  if (same_spin) {
    const int o_act = oa_act;
    const int v = n_orb - oa;
    const Size ov = sat_mul(Size(o_act), Size(v));
    const Size ovq = sat_mul(ov, Size(n_aux));
    const Size vv = sat_mul(Size(v), Size(v));

    const Size build_fixed = sat_add(
        sat_add(qq, nn),
        sat_add(sat_mul(Size(n_orb), Size(o_act + v)), sat_mul(Size(3), ovq)));
    const Size build_min = sat_add(build_fixed, sat_add(nn, sat_mul(Size(o_act), Size(n_orb))));

    Size pairs_fixed = sat_add(qq, ovq);
    if (compute_response) pairs_fixed = sat_add(pairs_fixed, sat_add(ovq, nn));
    pairs_fixed = sat_add(pairs_fixed, Size(o_act) + Size(v) + Size(2));
    if (has_occ_weights) pairs_fixed = sat_add(pairs_fixed, Size(o_act));
    if (compute_response) {
      pairs_fixed = sat_add(
          pairs_fixed, has_occ_weights ? (Size(o_act) + Size(v))
                                       : sat_add(sat_mul(Size(o_act), Size(o_act)), vv));
    }
    const Size pairs_min = sat_add(
        pairs_fixed,
        sat_add(sat_mul(Size(2), vv), (compute_response && !has_occ_weights) ? vv : 0));

    const Size pb1 = compute_response
                         ? sat_add(sat_mul(Size(3), qq), sat_mul(Size(5), ovq))
                         : 0;
    const Size pb2_fixed = compute_response
                               ? sat_add(sat_add(sat_mul(Size(2), qq), sat_mul(Size(3), ovq)),
                                         sat_add(sat_mul(Size(2), nn),
                                                 sat_mul(Size(2), sat_mul(Size(n_orb), Size(o_act + v)))))
                               : 0;
    const Size pb2_min = compute_response
                             ? sat_add(pb2_fixed,
                                       sat_add(nn, sat_add(sat_mul(Size(2), sat_mul(Size(o_act), Size(n_orb))), Size(1))))
                             : 0;

    const Size min_required_bytes =
        sat_add(bytes_of_doubles(std::max({build_min, pairs_min, pb1, pb2_min})), sizeof(int));
    if (min_required_bytes > eff_budget) {
      return kErrBudgetExceeded;
    }

    build_a_slab_cap = rem_cap(eff_budget, bytes_of_doubles(build_fixed), Size(512) << 20);
    pb_a_slab_cap = rem_cap(eff_budget, bytes_of_doubles(pb2_fixed), Size(512) << 20);
    const Size pairs_fixed_bytes = sat_add(bytes_of_doubles(pairs_fixed), sizeof(int));
    const Size pairs_rem = (eff_budget > pairs_fixed_bytes) ? (eff_budget - pairs_fixed_bytes) : 0;
    const Size min_tile_bytes = bytes_of_doubles(sat_mul(Size(2), vv));
    if (compute_response && !has_occ_weights) {
      tile_cap_main = std::min<Size>(pairs_rem / 2, Size(512) << 20);
      if (tile_cap_main < min_tile_bytes) {
        // Keep the third single-pair buffer available for exact recomputation.
        tile_cap_main = std::min(pairs_rem, min_tile_bytes);
        tri_cap_main = 0;
      } else {
        tri_cap_main = std::min<Size>(pairs_rem - tile_cap_main, Size(2048) << 20);
      }
    } else {
      tile_cap_main = std::min<Size>(pairs_rem, Size(512) << 20);
      tri_cap_main = 0;
    }
  } else {
    const int va = n_orb - oa;
    const int vb = n_orb - ob;
    const Size a_ovq = sat_mul(sat_mul(Size(oa_act), Size(va)), Size(n_aux));
    const Size b_ovq = sat_mul(sat_mul(Size(ob_act), Size(vb)), Size(n_aux));
    const Size vva = sat_mul(Size(va), Size(va));
    const Size vvb = sat_mul(Size(vb), Size(vb));
    const Size vab = sat_mul(Size(va), Size(vb));

    const Size build_a_fixed =
        (oa_act > 0)
            ? sat_add(sat_add(qq, nn),
                      sat_add(sat_mul(Size(n_orb), Size(oa_act + va)), sat_mul(Size(3), a_ovq)))
            : 0;
    const Size build_a_min =
        (oa_act > 0) ? sat_add(build_a_fixed, sat_add(nn, sat_mul(Size(oa_act), Size(n_orb)))) : 0;

    const Size build_b_fixed =
        (ob_act > 0)
            ? sat_add(sat_add(sat_add(qq, a_ovq), nn),
                      sat_add(sat_mul(Size(n_orb), Size(ob_act + vb)), sat_mul(Size(3), b_ovq)))
            : 0;
    const Size build_b_min =
        (ob_act > 0) ? sat_add(build_b_fixed, sat_add(nn, sat_mul(Size(ob_act), Size(n_orb)))) : 0;

    Size pairs_base = sat_add(qq, sat_add(a_ovq, b_ovq));
    if (compute_response) {
      pairs_base = sat_add(pairs_base, sat_add(sat_add(a_ovq, b_ovq), sat_mul(Size(2), nn)));
    }
    const Size aa_fixed =
        (oa_act > 1 && va > 0)
            ? sat_add(pairs_base,
                      sat_add(Size(oa_act) + Size(va) + Size(1) + (has_occ_weights ? Size(oa_act) : 0),
                              compute_response
                                  ? (has_occ_weights ? (Size(oa_act) + Size(va))
                                                     : sat_add(sat_mul(Size(oa_act), Size(oa_act)), vva))
                                  : 0))
            : pairs_base;
    const Size aa_min =
        (oa_act > 1 && va > 0)
            ? sat_add(aa_fixed, sat_add(sat_mul(Size(2), vva), (compute_response && !has_occ_weights) ? vva : 0))
            : pairs_base;

    const Size ab_fixed =
        (oa_act > 0 && ob_act > 0 && va > 0 && vb > 0)
            ? sat_add(pairs_base,
                      sat_add(Size(oa_act) + Size(va) + Size(ob_act) + Size(vb) + Size(1) +
                                  (has_occ_weights ? (Size(oa_act) + Size(ob_act)) : 0),
                              compute_response
                                  ? (has_occ_weights
                                         ? (Size(oa_act) + Size(ob_act) + Size(va) + Size(vb))
                                         : sat_add(sat_add(sat_mul(Size(oa_act), Size(oa_act)),
                                                           sat_mul(Size(ob_act), Size(ob_act))),
                                                   sat_add(vva, vvb)))
                                  : 0))
            : pairs_base;
    const Size ab_min =
        (oa_act > 0 && ob_act > 0 && va > 0 && vb > 0)
            ? sat_add(ab_fixed, sat_add(sat_mul(Size(2), vab), (compute_response && !has_occ_weights) ? vab : 0))
            : pairs_base;

    const Size bb_fixed =
        (ob_act > 1 && vb > 0)
            ? sat_add(pairs_base,
                      sat_add(Size(ob_act) + Size(vb) + Size(1) + (has_occ_weights ? Size(ob_act) : 0),
                              compute_response
                                  ? (has_occ_weights ? (Size(ob_act) + Size(vb))
                                                     : sat_add(sat_mul(Size(ob_act), Size(ob_act)), vvb))
                                  : 0))
            : pairs_base;
    const Size bb_min =
        (ob_act > 1 && vb > 0)
            ? sat_add(bb_fixed, sat_add(sat_mul(Size(2), vvb), (compute_response && !has_occ_weights) ? vvb : 0))
            : pairs_base;

    const Size pb_a1 =
        (compute_response && oa_act > 0)
            ? sat_add(sat_mul(Size(3), qq), sat_add(sat_mul(Size(5), a_ovq), sat_mul(Size(2), b_ovq)))
            : 0;
    const Size pb_a2_fixed =
        (compute_response && oa_act > 0)
            ? sat_add(sat_add(sat_mul(Size(2), qq),
                              sat_add(sat_mul(Size(3), a_ovq), sat_mul(Size(2), b_ovq))),
                      sat_add(sat_mul(Size(2), nn),
                              sat_mul(Size(2), sat_mul(Size(n_orb), Size(oa_act + va)))))
            : 0;
    const Size pb_a2_min =
        (compute_response && oa_act > 0)
            ? sat_add(pb_a2_fixed,
                      sat_add(nn, sat_add(sat_mul(Size(2), sat_mul(Size(oa_act), Size(n_orb))), Size(1))))
            : 0;

    const Size pb_b1 =
        (compute_response && ob_act > 0)
            ? sat_add(sat_mul(Size(3), qq), sat_mul(Size(5), b_ovq))
            : 0;
    const Size pb_b2_fixed =
        (compute_response && ob_act > 0)
            ? sat_add(sat_add(sat_mul(Size(2), qq), sat_mul(Size(3), b_ovq)),
                      sat_add(sat_mul(Size(2), nn),
                              sat_mul(Size(2), sat_mul(Size(n_orb), Size(ob_act + vb)))))
            : 0;
    const Size pb_b2_min =
        (compute_response && ob_act > 0)
            ? sat_add(pb_b2_fixed,
                      sat_add(nn, sat_add(sat_mul(Size(2), sat_mul(Size(ob_act), Size(n_orb))), Size(1))))
            : 0;

    const Size min_required_bytes = sat_add(
        bytes_of_doubles(std::max({build_a_min, build_b_min, aa_min, ab_min, bb_min, pb_a1,
                                   pb_a2_min, pb_b1, pb_b2_min})),
        sizeof(int));
    if (min_required_bytes > eff_budget) {
      return kErrBudgetExceeded;
    }

    build_a_slab_cap = rem_cap(eff_budget, bytes_of_doubles(build_a_fixed), Size(512) << 20);
    build_b_slab_cap = rem_cap(eff_budget, bytes_of_doubles(build_b_fixed), Size(512) << 20);
    pb_a_slab_cap = rem_cap(eff_budget, bytes_of_doubles(pb_a2_fixed), Size(512) << 20);
    pb_b_slab_cap = rem_cap(eff_budget, bytes_of_doubles(pb_b2_fixed), Size(512) << 20);
    const Size max_pairs_fixed = std::max({aa_fixed, ab_fixed, bb_fixed});
    const Size pairs_fixed_bytes = sat_add(bytes_of_doubles(max_pairs_fixed), sizeof(int));
    const Size pairs_rem = (eff_budget > pairs_fixed_bytes) ? (eff_budget - pairs_fixed_bytes) : 0;
    const Size min_tile_bytes = bytes_of_doubles(sat_mul(Size(2), std::max({vva, vvb, vab})));
    if (compute_response && !has_occ_weights) {
      tile_cap_main = std::min<Size>(pairs_rem / 2, Size(512) << 20);
      if (tile_cap_main < min_tile_bytes) {
        // Keep the third single-pair buffer available for exact recomputation.
        tile_cap_main = std::min(pairs_rem, min_tile_bytes);
        tri_cap_main = 0;
      } else {
        tri_cap_main = std::min<Size>(pairs_rem - tile_cap_main, Size(2048) << 20);
      }
    } else {
      tile_cap_main = std::min<Size>(pairs_rem, Size(512) << 20);
      tri_cap_main = 0;
    }
  }

  DeviceWorkspace workspace(budget_bytes);
  CublasGuard cg;
  if (cg.init() != CUBLAS_STATUS_SUCCESS) return 1;
  cublasHandle_t h = cg.handle;

  DeviceBuf d_whitener;
  if (d_whitener.alloc(qq) != cudaSuccess ||
      cudaMemcpy(d_whitener.ptr, whitener, qq * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess) {
    return 2;
  }

  if (same_spin) {
    const int o = oa;
    const int f = fa;
    const int o_act = oa_act;
    const int v = n_orb - o;

    DeviceBuf d_B_ovq, d_bar_B_ovq, d_fock_a;
    if (!build_ordinary_factors_gpu(h, n_orb, n_aux, o, f, build_a_slab_cap, coeff_a, ao_coul,
                                    d_whitener.ptr, d_B_ovq)) {
      return 3;
    }
    if (compute_response) {
      const Size ovq = Size(o_act) * Size(v) * Size(n_aux);
      if (d_bar_B_ovq.alloc(ovq) != cudaSuccess || d_fock_a.alloc(nn) != cudaSuccess ||
          cudaMemset(d_bar_B_ovq.ptr, 0, ovq * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_fock_a.ptr, 0, nn * sizeof(double)) != cudaSuccess) {
        return 3;
      }
    }

    double e_aa = 0.0, e_ab = 0.0;
    if (int rc = run_rhf_pairs_gpu(h, n_orb, n_aux, o, f, compute_response, has_occ_weights,
                                   tile_cap_main, tri_cap_main, occ_w_a, eps_a, d_B_ovq.ptr,
                                   d_bar_B_ovq.ptr, d_fock_a.ptr, e_aa, e_ab);
        rc != 0) {
      return rc;
    }
    spin_energies[0] = e_aa;
    spin_energies[1] = e_ab;
    spin_energies[2] = e_aa;

    if (compute_response) {
      if (cudaMemcpy(fock_a, d_fock_a.ptr, nn * sizeof(double), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        return 9;
      }
      d_fock_a.reset();
      if (fock_b && fock_b != fock_a) std::copy_n(fock_a, nn, fock_b);

      DeviceBuf d_bar_metric;
      if (d_bar_metric.alloc(qq) != cudaSuccess ||
          cudaMemset(d_bar_metric.ptr, 0, qq * sizeof(double)) != cudaSuccess) {
        return 9;
      }
      if (!pullback_ordinary_factors_gpu(h, n_orb, n_aux, o, f, 2.0, false, pb_a_slab_cap, coeff_a,
                                         ao_coul, d_whitener.ptr, d_B_ovq.ptr, d_bar_B_ovq.ptr,
                                         d_bar_metric.ptr, bar_coeff_a, bar_ao_coul)) {
        return 10;
      }
      if (cudaDeviceSynchronize() != cudaSuccess ||
          cudaMemcpy(bar_metric, d_bar_metric.ptr, qq * sizeof(double), cudaMemcpyDeviceToHost) !=
              cudaSuccess) {
        return 11;
      }
      if (bar_coeff_b && bar_coeff_b != bar_coeff_a) std::copy_n(bar_coeff_a, nn, bar_coeff_b);
    }
  } else {
    DeviceBuf d_Ba_ovq, d_Bb_ovq, d_bar_Ba_ovq, d_bar_Bb_ovq, d_fock_a, d_fock_b;
    if (oa_act > 0 && !build_ordinary_factors_gpu(h, n_orb, n_aux, oa, fa, build_a_slab_cap,
                                                  coeff_a, ao_coul, d_whitener.ptr, d_Ba_ovq)) {
      return 3;
    }
    if (ob_act > 0 && !build_ordinary_factors_gpu(h, n_orb, n_aux, ob, fb, build_b_slab_cap,
                                                  coeff_b, ao_coul, d_whitener.ptr, d_Bb_ovq)) {
      return 3;
    }
    if (compute_response) {
      const Size a_ovq = Size(oa_act) * Size(n_orb - oa) * Size(n_aux);
      const Size b_ovq = Size(ob_act) * Size(n_orb - ob) * Size(n_aux);
      if ((oa_act > 0 &&
           (d_bar_Ba_ovq.alloc(a_ovq) != cudaSuccess ||
            cudaMemset(d_bar_Ba_ovq.ptr, 0, a_ovq * sizeof(double)) != cudaSuccess)) ||
          (ob_act > 0 &&
           (d_bar_Bb_ovq.alloc(b_ovq) != cudaSuccess ||
            cudaMemset(d_bar_Bb_ovq.ptr, 0, b_ovq * sizeof(double)) != cudaSuccess)) ||
          d_fock_a.alloc(nn) != cudaSuccess || d_fock_b.alloc(nn) != cudaSuccess ||
          cudaMemset(d_fock_a.ptr, 0, nn * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_fock_b.ptr, 0, nn * sizeof(double)) != cudaSuccess) {
        return 3;
      }
    }

    double e_aa = 0.0, e_ab = 0.0, e_bb = 0.0;
    if (int rc = run_uhf_same_spin_gpu(h, n_orb, n_aux, oa, fa, compute_response, has_occ_weights,
                                       tile_cap_main, tri_cap_main, occ_w_a, eps_a, d_Ba_ovq.ptr,
                                       d_bar_Ba_ovq.ptr, d_fock_a.ptr, e_aa);
        rc != 0) {
      return rc;
    }
    if (int rc = run_uhf_opp_spin_gpu(h, n_orb, n_aux, oa, fa, ob, fb, compute_response,
                                      has_occ_weights, tile_cap_main, tri_cap_main, occ_w_a,
                                      occ_w_b, eps_a, eps_b, d_Ba_ovq.ptr, d_Bb_ovq.ptr,
                                      d_bar_Ba_ovq.ptr, d_bar_Bb_ovq.ptr, d_fock_a.ptr,
                                      d_fock_b.ptr, e_ab);
        rc != 0) {
      return rc;
    }
    if (int rc = run_uhf_same_spin_gpu(h, n_orb, n_aux, ob, fb, compute_response, has_occ_weights,
                                       tile_cap_main, tri_cap_main, occ_w_b, eps_b, d_Bb_ovq.ptr,
                                       d_bar_Bb_ovq.ptr, d_fock_b.ptr, e_bb);
        rc != 0) {
      return rc;
    }

    spin_energies[0] = e_aa;
    spin_energies[1] = e_ab;
    spin_energies[2] = e_bb;

    if (compute_response) {
      if (cudaMemcpy(fock_a, d_fock_a.ptr, nn * sizeof(double), cudaMemcpyDeviceToHost) !=
              cudaSuccess ||
          cudaMemcpy(fock_b, d_fock_b.ptr, nn * sizeof(double), cudaMemcpyDeviceToHost) !=
              cudaSuccess) {
        return 9;
      }
      d_fock_a.reset();
      d_fock_b.reset();

      DeviceBuf d_bar_metric;
      if (d_bar_metric.alloc(qq) != cudaSuccess ||
          cudaMemset(d_bar_metric.ptr, 0, qq * sizeof(double)) != cudaSuccess) {
        return 9;
      }
      if (oa_act > 0) {
        if (!pullback_ordinary_factors_gpu(h, n_orb, n_aux, oa, fa, 1.0, false, pb_a_slab_cap,
                                           coeff_a, ao_coul, d_whitener.ptr, d_Ba_ovq.ptr,
                                           d_bar_Ba_ovq.ptr, d_bar_metric.ptr, bar_coeff_a,
                                           bar_ao_coul)) {
          return 10;
        }
        d_Ba_ovq.reset();
        d_bar_Ba_ovq.reset();
      }
      if (ob_act > 0) {
        if (!pullback_ordinary_factors_gpu(h, n_orb, n_aux, ob, fb, 1.0, (oa_act > 0),
                                           pb_b_slab_cap, coeff_b, ao_coul, d_whitener.ptr,
                                           d_Bb_ovq.ptr, d_bar_Bb_ovq.ptr, d_bar_metric.ptr,
                                           bar_coeff_b, bar_ao_coul)) {
          return 10;
        }
        d_Bb_ovq.reset();
        d_bar_Bb_ovq.reset();
      }
      if (cudaDeviceSynchronize() != cudaSuccess ||
          cudaMemcpy(bar_metric, d_bar_metric.ptr, qq * sizeof(double), cudaMemcpyDeviceToHost) !=
              cudaSuccess) {
        return 11;
      }
    }
  }

  if (!all_finite_host(spin_energies, 3)) return -5;
  if (compute_response) {
    if (!all_finite_host(fock_a, nn) || !all_finite_host(bar_coeff_a, nn) ||
        !all_finite_host(bar_ao_coul, qnn) || !all_finite_host(bar_metric, qq)) {
      return -5;
    }
    if (!same_spin && (!all_finite_host(fock_b, nn) || !all_finite_host(bar_coeff_b, nn))) {
      return -5;
    }
  }
  return 0;
  } catch (const DeviceBudgetExceeded&) {
    return kErrBudgetExceeded;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

#endif

}  // extern "C"
