// AOT GPU-resident cuBLAS (and matching FP64 host reference) kernels for:
// 1. Ordinary DF-MP2 energy + unrelaxed response pullback (RHF spatial fast path + UHF dual-spin)
// 2. Fixed-cusp 3C(FIX) MP2-F12 pair energy + reverse MO-source pullback (RHF fast path + UHF)
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
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

Size sum_sizes(std::initializer_list<Size> sizes) {
  Size sum = 0;
  for (Size size : sizes) sum = sat_add(sum, size);
  return sum;
}

// All workspace sizes below are FP64 element counts until the byte conversion at the ABI.
Size ab_projection_scratch(int oa_act, int ob_act, int ext, int aux, Size output_scalars = 0) {
  const Size left = 2 * Size(ob_act) + 3 * Size(oa_act);
  const Size right = 2 * Size(oa_act) + 3 * Size(ob_act);
  const Size panels = sat_mul(left + right, Size(ext));
  const Size transforms = sat_mul(10, sat_mul(Size(ext), Size(ext)));
  const Size contractions = sat_mul(2 * Size(aux), sum_sizes({
      Size(oa_act), Size(ob_act), sat_mul(oa_act, left), sat_mul(ob_act, right)}));
  return sat_add(panels, std::max(transforms, sat_add(contractions, output_scalars)));
}

struct PairBatchWorkspace {
  Size sub_a;
  Size sub_b;
  Size elements;
};

PairBatchWorkspace ss_batch_workspace(int o, int obs, int ext) {
  const Size v = obs - o;
  const Size c = ext - obs;
  const Size sub_a = std::max({Size(o) * ext, v * obs, c * ext, v * c});
  const Size sub_b = std::max({Size(o) * ext, c * o, v * ext});
  return {sub_a, sub_b, sum_sizes({sat_mul(3, Size(ext) * ext), Size(obs) * obs,
                                   sat_mul(2, Size(o) * c), v * v, sub_a, sub_b, 6})};
}

PairBatchWorkspace ab_batch_workspace(int oa, int ob, int obs, int ext) {
  const Size va = obs - oa;
  const Size vb = obs - ob;
  const Size c = ext - obs;
  const Size sub_a = std::max({Size(ext) * ob, Size(obs) * vb, c * vb, Size(ext) * c,
                               Size(oa) * ext, va * obs, va * c});
  const Size sub_b = std::max({Size(ext) * ob, Size(obs) * vb, c * vb, Size(oa) * c,
                               Size(ext - oa) * c, Size(oa) * ext, va * obs, va * c,
                               c * ob, c * (ext - ob)});
  return {sub_a, sub_b, sum_sizes({sat_mul(2, Size(ext) * ext), Size(obs) * obs,
                                   Size(oa) * c, c * ob, Size(obs) * vb, va * obs,
                                   sub_a, sub_b, 3})};
}

Size estimate_mp2_pair_response_device_bytes(
    bool rhf_spatial, bool include_f12, bool include_ordinary, bool has_pair_f12_out,
    bool has_pair_ord_out, bool has_bar_cross_overlap, int oa, int ob, int fa, int fb, int obs,
    int ext, int aux) {
  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  const Size qq = sat_mul(Size(aux), Size(aux));
  const Size ee = sat_mul(Size(ext), Size(ext));
  const Size qoe_a = sat_mul(sat_mul(Size(aux), Size(oa_act)), Size(ext));
  const Size qoe_b = sat_mul(sat_mul(Size(aux), Size(ob_act)), Size(ext));
  const Size max_qoe = std::max(qoe_a, qoe_b);
  const Size n_aa = (oa_act >= 2) ? (Size(oa_act) * (oa_act - 1)) / 2 : 0;
  const Size n_ab = sat_mul(Size(oa_act), Size(ob_act));
  const Size n_bb = (ob_act >= 2) ? (Size(ob_act) * (ob_act - 1)) / 2 : 0;
  const Size n_total = sat_add(sat_add(n_aa, n_ab), n_bb);

  Size base = 6;
  if (include_f12 && has_pair_f12_out) base = sat_add(base, n_total);
  if (include_ordinary && has_pair_ord_out) base = sat_add(base, n_total);
  base = sat_add(base, max_qoe);
  base = sat_add(base, sat_mul(Size(3), qq));  // d_metric_buf, d_bar_metric_buf, d_m_tmp

  const Size obs2 = sat_mul(Size(obs), Size(obs));
  const int cabs = ext - obs;

  if (rhf_spatial) {
    Size fixed = sat_mul(Size(5 + ((include_f12 && has_bar_cross_overlap) ? 1 : 0)), ee);
    fixed = sat_add(fixed, sat_mul(Size(4), qoe_a));
    if (include_f12) {
      fixed = sat_add(fixed, sat_mul(Size(8), sat_mul(Size(oa_act), Size(oa_act))));
    }
    Size s1a = 0;
    Size s1b = 0;
    if (include_f12) {
      s1a = sat_mul(Size(6), sat_mul(sat_mul(Size(aux), Size(oa_act)), Size(oa)));
      s1b = sat_add(sat_mul(Size(5), qoe_a),
                    sat_add(ee, sat_add(sat_mul(Size(18), Size(ext)), sat_mul(Size(2), Size(aux)))));
    }
    Size s2 = sat_mul(Size(include_f12 ? 7 : 2), qoe_a);
    const Size va = Size(obs - oa);
    const Size vv = sat_mul(va, va);
    if (include_f12) {
      const Size oc = sat_mul(Size(oa), Size(cabs));
      const auto batch = ss_batch_workspace(oa, obs, ext);
      const Size s2_ws = sum_sizes({sat_mul(13, ee), sat_mul(3, obs2), sat_mul(6, oc),
                                    sat_mul(6, vv), batch.sub_a, batch.sub_b, 6});
      s2 = sat_add(s2, s2_ws);
    } else if (include_ordinary) {
      s2 = sat_add(s2, sat_mul(Size(3), vv));
    }
    const Size peak = sat_add(sat_add(base, fixed), std::max({s1a, s1b, s2}));
    return sat_add(bytes_of_doubles(peak), sizeof(int));
  }

  // General UHF path.
  Size fixed = sat_mul(Size(14), ee);
  if (oa_act > 0 && ob_act > 0 && include_f12) {
    fixed = sat_add(fixed, sat_mul(Size(has_bar_cross_overlap ? 4 : 3), ee));
  }
  fixed = sat_add(fixed, sat_mul(Size(4), sat_add(qoe_a, qoe_b)));
  if (include_f12) {
    if (oa_act >= 2) fixed = sat_add(fixed, sat_mul(Size(8), sat_mul(Size(oa_act), Size(oa_act))));
    if (ob_act >= 2) fixed = sat_add(fixed, sat_mul(Size(8), sat_mul(Size(ob_act), Size(ob_act))));
    if (oa_act > 0 && ob_act > 0) fixed = sat_add(fixed, sat_mul(Size(4), n_ab));
  }
  Size s1a = 0;
  Size s1b = 0;
  if (include_f12) {
    const Size ss_a = (oa_act >= 2) ? sat_mul(Size(6), sat_mul(sat_mul(Size(aux), Size(oa_act)), Size(oa))) : 0;
    const Size ss_b = (ob_act >= 2) ? sat_mul(Size(6), sat_mul(sat_mul(Size(aux), Size(ob_act)), Size(ob))) : 0;
    const Size ab_ws = (oa_act > 0 && ob_act > 0) ? ab_projection_scratch(oa_act, ob_act, ext, aux) : 0;
    s1a = std::max({ss_a, ss_b, ab_ws});
    const Size q_s1b_a = (oa_act >= 2) ? sat_mul(Size(4), qoe_a) : (oa_act > 0 ? sat_mul(Size(2), qoe_a) : 0);
    const Size q_s1b_b = (ob_act > 0) ? sat_mul(Size(4), qoe_b) : 0;
    const Size scratch_s1b = std::max(
        sat_add(ee, sat_add(sat_mul(Size(18), Size(ext)), sat_mul(Size(2), Size(aux)))),
        sat_add(sat_mul(Size(4), Size(aux)), sat_mul(Size(22), Size(ext))));
    s1b = sat_add(sat_add(q_s1b_a, q_s1b_b), scratch_s1b);
  }
  const Size q_s2_a = (oa_act > 0) ? sat_mul(Size(2 + (include_f12 ? (oa_act >= 2 ? 4 : 2) : 0)), qoe_a) : 0;
  const Size q_s2_b = (ob_act > 0) ? sat_mul(Size(2 + (include_f12 ? 4 : 0)), qoe_b) : 0;
  const Size va = Size(obs - oa);
  const Size vb = Size(obs - ob);
  const Size vab = sat_mul(va, vb);
  Size s2_ws = 0;
  if (include_f12) {
    const Size ab_s2 = sum_sizes({sat_mul(8, ee), sat_mul(2, obs2),
        sat_mul(2 * Size(cabs), Size(oa) + Size(ob)), sat_mul(2, vab), 3});
    auto same_spin_scratch = [&](int o, Size v) {
      const auto batch = ss_batch_workspace(o, obs, ext);
      return sum_sizes({sat_mul(7, ee), sat_mul(2, obs2), sat_mul(4 * Size(o), Size(cabs)),
                        sat_mul(5, v * v), batch.sub_a, batch.sub_b, 6});
    };
    s2_ws = std::max({oa_act > 0 && ob_act > 0 ? ab_s2 : 0,
                      oa_act >= 2 ? same_spin_scratch(oa, va) : 0,
                      ob_act >= 2 ? same_spin_scratch(ob, vb) : 0});
  } else if (include_ordinary) {
    s2_ws = sat_mul(Size(2), std::max({sat_mul(va, va), sat_mul(vb, vb), vab}));
  }
  const Size s2 = sat_add(sat_add(q_s2_a, q_s2_b), s2_ws);
  const Size peak = sat_add(sat_add(base, fixed), std::max({s1a, s1b, s2}));
  return sat_add(bytes_of_doubles(peak), sizeof(int));
}

Size estimate_mp2_pair_energies_device_bytes(
    bool rhf_spatial, bool include_ordinary, bool has_occ_weights, bool has_pair_f12_out,
    bool has_pair_ord_out, int oa, int ob, int fa, int fb, int obs, int ext, int aux) {
  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  const Size qq = sat_mul(Size(aux), Size(aux));
  const Size ee = sat_mul(Size(ext), Size(ext));
  const Size qoe_a = sat_mul(sat_mul(Size(aux), Size(oa_act)), Size(ext));
  const Size qoe_b = sat_mul(sat_mul(Size(aux), Size(ob_act)), Size(ext));
  const Size max_qoe = std::max(qoe_a, qoe_b);
  const Size n_aa = (oa_act >= 2) ? (Size(oa_act) * (oa_act - 1)) / 2 : 0;
  const Size n_ab = sat_mul(Size(oa_act), Size(ob_act));
  const Size n_bb = (ob_act >= 2) ? (Size(ob_act) * (ob_act - 1)) / 2 : 0;
  const Size n_total = sat_add(sat_add(n_aa, n_ab), n_bb);

  Size base = 6;
  if (has_pair_f12_out) base = sat_add(base, n_total);
  if (include_ordinary && has_pair_ord_out) base = sat_add(base, n_total);
  if (has_occ_weights) base = sat_add(base, Size(oa_act) + (rhf_spatial ? 0 : Size(ob_act)));

  if (rhf_spatial) {
    const Size stage1_elems = sat_add(
        sat_add(base, sat_mul(Size(3), ee)),
        sat_add(sat_mul(Size(2), qoe_a),
                sat_add(qq, sat_add(sat_mul(Size(8), sat_mul(Size(oa_act), Size(oa_act))),
                                    sat_mul(Size(6), sat_mul(sat_mul(Size(aux), Size(oa_act)), Size(oa)))))));
    // Repacking fit into OQE briefly retains five panels after the metric is released.
    const Size prep_elems = sum_sizes({base, sat_mul(2, ee), sat_mul(4, qoe_a),
        std::max(qq, qoe_a), sat_mul(8, Size(oa_act) * oa_act)});
    const Size stage2_fixed = sat_add(
        sat_add(base, sat_mul(Size(2), ee)),
        sat_add(sat_mul(Size(4), qoe_a), sat_mul(Size(8), sat_mul(Size(oa_act), Size(oa_act)))));
    const Size min_batch = ss_batch_workspace(oa, obs, ext).elements;
    const Size peak = std::max({stage1_elems, prep_elems, sat_add(stage2_fixed, min_batch)});
    return sat_add(bytes_of_doubles(peak), sizeof(int));
  }

  const bool aa = oa_act >= 2, ab = oa_act > 0 && ob_act > 0, bb = ob_act >= 2;
  const Size root = sat_add(base, sat_mul(4 + Size(ab), ee));
  const Size inputs = sum_sizes({root, qoe_a, qoe_b, max_qoe, qq});
  const Size aa_scalars = aa ? sat_mul(8, Size(oa_act) * oa_act) : 0;
  const Size ab_scalars = ab ? sat_mul(4, n_ab) : 0;
  const Size bb_scalars = bb ? sat_mul(8, Size(ob_act) * ob_act) : 0;
  Size peak = inputs;
  if (aa) {
    const Size contracted = sat_mul(6 * Size(aux), Size(oa_act) * oa);
    peak = std::max(peak, sum_sizes({inputs, aa_scalars, ee, contracted}));
  }
  if (ab) {
    peak = std::max(peak, sum_sizes({inputs, aa_scalars,
        ab_projection_scratch(oa_act, ob_act, ext, aux, ab_scalars)}));
  }
  if (bb) {
    const Size contracted = sat_mul(6 * Size(aux), Size(ob_act) * ob);
    peak = std::max(peak, sum_sizes({inputs, aa_scalars, ab_scalars, bb_scalars, ee, contracted}));
  }

  // The AA preparation retains the beta QOE panel and the metric through its batch;
  // AB releases the AA scalars before repacking beta. BB releases alpha and Sab.
  const Size all_scalars = sum_sizes({aa_scalars, ab_scalars, bb_scalars});
  if (oa_act > 0) {
    peak = std::max(peak, sum_sizes({inputs, all_scalars, sat_mul(aa ? 3 : 2, qoe_a)}));
  }
  if (aa) {
    peak = std::max(peak, sum_sizes({root, all_scalars, sat_mul(4, qoe_a),
                                     qoe_b, qq, ss_batch_workspace(oa, obs, ext).elements}));
  }
  if (ab || bb) {
    const Size alpha_panels = sat_mul(2, qoe_a);
    const Size retained = sum_sizes({root, ab_scalars, bb_scalars, alpha_panels});
    const Size scratch = aa ? qoe_b : max_qoe;
    peak = std::max(peak, sum_sizes({retained, scratch,
          std::max(sat_add(qq, sat_mul(2, qoe_b)), sat_mul(3, qoe_b))}));
    if (ab) {
      peak = std::max(peak, sum_sizes({retained, sat_mul(3, qoe_b),
                                       ab_batch_workspace(oa, ob, obs, ext).elements}));
    }
    if (bb) {
      const Size beta_root = sum_sizes({base, sat_mul(4, ee), bb_scalars, sat_mul(4, qoe_b)});
      peak = std::max(peak, sat_add(beta_root, std::max(qoe_b, ss_batch_workspace(ob, obs, ext).elements)));
    }
  }
  return sat_add(bytes_of_doubles(peak), sizeof(int));
}

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

GQC_HD inline Size strict_tri_index(int i, int j, int n) {
  // Strict upper-triangular index for 0 <= i < j < n.
  const Size ii = static_cast<Size>(i);
  const Size tri_skip = (ii * (ii + 1)) / 2;
  return ii * static_cast<Size>(n) - tri_skip + static_cast<Size>(j - i - 1);
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

// -------------------------------------------------------------------------
// Host-Reference MP2-F12 Forward Pair Energies (V / B / X / C, 3C(FIX) SP)
// -------------------------------------------------------------------------

double dot_rect_host(const double* A, int lda, const double* B, int ldb, int rows, int cols) {
  double acc = 0.0;
  for (int r = 0; r < rows; ++r) {
    const double* ar = A + Size(r) * lda;
    const double* br = B + Size(r) * ldb;
    for (int c = 0; c < cols; ++c) acc += ar[c] * br[c];
  }
  return acc;
}

double sqnorm_rect_host(const double* A, int lda, int rows, int cols) {
  double acc = 0.0;
  for (int r = 0; r < rows; ++r) {
    const double* ar = A + Size(r) * lda;
    for (int c = 0; c < cols; ++c) acc += ar[c] * ar[c];
  }
  return acc;
}

void pack_active_qoe_host(const double* src, int aux, int n1, int f, int o_act, int ext,
                          std::vector<double>& out_qoe) {
  out_qoe.resize(Size(aux) * Size(o_act) * ext);
  for (int A = 0; A < aux; ++A) {
    const double* s_base = src + (Size(A) * n1 + f) * ext;
    double* d_base = out_qoe.data() + Size(A) * o_act * ext;
    std::copy_n(s_base, Size(o_act) * ext, d_base);
  }
}

void transpose_qoe_to_oqe_host(const std::vector<double>& in_qoe, int aux, int o_act, int ext,
                               std::vector<double>& out_oqe) {
  out_oqe.resize(Size(o_act) * Size(aux) * ext);
  for (int A = 0; A < aux; ++A) {
    for (int i = 0; i < o_act; ++i) {
      const double* s = in_qoe.data() + (Size(A) * o_act + i) * ext;
      double* d = out_oqe.data() + (Size(i) * aux + A) * ext;
      std::copy_n(s, ext, d);
    }
  }
}

bool pack_active_oqe_checked_host(const double* src, int aux, int n1, int f, int o_act, int ext,
                                  std::vector<double>& out_oqe,
                                  std::vector<double>* out_qoe_opt = nullptr) {
  std::vector<double> qoe;
  pack_active_qoe_host(src, aux, n1, f, o_act, ext, qoe);
  if (!all_finite_host(qoe.data(), qoe.size())) return false;
  transpose_qoe_to_oqe_host(qoe, aux, o_act, ext, out_oqe);
  if (out_qoe_opt) *out_qoe_opt = std::move(qoe);
  return true;
}

void build_res_oqe_host(const std::vector<double>& raw_qoe, const std::vector<double>& fit_qoe,
                        const double* metric, int aux, int o_act, int ext,
                        std::vector<double>& out_res_oqe) {
  std::vector<double> res_qoe = raw_qoe;
  const int cols = o_act * ext;
  rm_gemm_host(false, false, aux, cols, aux, -1.0, metric, aux, fit_qoe.data(), cols, 1.0,
               res_qoe.data(), cols);
  transpose_qoe_to_oqe_host(res_qoe, aux, o_act, ext, out_res_oqe);
}

inline double robust_elem_host(const double* fit_i, const double* raw_i, const double* fit_j,
                               const double* res_j, int aux, int ext, int p, int r) {
  double acc = 0.0;
  for (int A = 0; A < aux; ++A) {
    const Size row = Size(A) * ext;
    acc += fit_i[row + p] * res_j[row + r] + raw_i[row + p] * fit_j[row + r];
  }
  return acc;
}

double robust_bilin_host(const double* fit_i, const double* raw_i, const double* fit_j,
                         const double* res_j, int aux, int ext, const double* x, const double* y,
                         std::vector<double>& tmp_a, std::vector<double>& tmp_b) {
  // Computes x^T (fit_i^T res_j + raw_i^T fit_j) y in O(aux * ext).
  double acc = 0.0;
  for (int A = 0; A < aux; ++A) {
    const double* fi = fit_i + Size(A) * ext;
    const double* ri = raw_i + Size(A) * ext;
    const double* fj = fit_j + Size(A) * ext;
    const double* sj = res_j + Size(A) * ext;
    double fx = 0.0, rx = 0.0, fy = 0.0, sy = 0.0;
    for (int p = 0; p < ext; ++p) {
      const double xp = x[p];
      const double yp = y[p];
      fx += fi[p] * xp;
      rx += ri[p] * xp;
      fy += fj[p] * yp;
      sy += sj[p] * yp;
    }
    tmp_a[A] = fx * sy;
    tmp_b[A] = rx * fy;
    acc += tmp_a[A] + tmp_b[A];
  }
  return acc;
}

struct BLeftScratchHost {
  std::vector<double> L0, L1, L2, L35, L4;
};

void compute_b_left_host(const double* A, const double* F, const double* K, int o, int obs, int ext,
                         BLeftScratchHost& out) {
  const int v = obs - o;
  const int c = ext - obs;
  out.L0.resize(Size(ext) * ext);
  out.L1.resize(Size(o) * ext);
  out.L2.resize(Size(c) * o);
  out.L35.resize(Size(v) * ext);
  out.L4.resize(Size(c) * o);

  rm_gemm_host(false, false, ext, ext, ext, 1.0, A, ext, K, ext, 0.0, out.L0.data(), ext);
  rm_gemm_host(false, false, o, ext, ext, 1.0, A, ext, F, ext, 0.0, out.L1.data(), ext);
  rm_gemm_host(false, false, c, o, o, 1.0, A + Size(obs) * ext, ext, F, ext, 0.0, out.L2.data(), o);
  rm_gemm_host(false, false, v, ext, obs, 1.0, A + Size(o) * ext, ext, F, ext, 0.0, out.L35.data(),
               ext);
  rm_gemm_host(false, true, c, o, ext, 1.0, A + Size(obs) * ext, ext, F, ext, 0.0, out.L4.data(),
               o);
}

double eval_b_overlap_host(const BLeftScratchHost& L, const double* B, int o, int obs, int ext) {
  const int v = obs - o;
  const int c = ext - obs;
  double acc = 0.0;
  acc -= dot_rect_host(L.L0.data(), ext, B, ext, ext, ext);
  acc -= dot_rect_host(L.L1.data(), ext, B, ext, o, ext);
  acc += dot_rect_host(L.L2.data(), o, B + Size(obs) * ext, ext, c, o);
  acc -= dot_rect_host(L.L35.data(), ext, B + Size(o) * ext, ext, v, obs);
  acc -= 2.0 * dot_rect_host(L.L4.data(), o, B + Size(obs) * ext, ext, c, o);
  acc -= 2.0 * dot_rect_host(L.L35.data() + obs, ext, B + Size(o) * ext + obs, ext, v, c);
  return acc;
}

int eval_same_spin_or_rhf_pair_host(
    int i, int j, bool compute_both_rhf, bool include_ordinary, int o, int obs, int ext, int aux,
    const double* fit_i, const double* fit_j, const double* raw_coul_j, const double* raw_g_i,
    const double* res_g_j, const double* raw_sq_i, const double* res_sq_j, const double* raw_gc_i,
    const double* res_gc_j, const double* raw_dc_i, const double* res_dc_j, const double* F,
    const double* K, const double* H, double& e_f12_ss, double& e_ord_ss, double& e_f12_os,
    double& e_ord_os) {
  const int v = obs - o;
  const int c = ext - obs;
  const bool diag = (i == j);
  e_f12_ss = 0.0;
  e_ord_ss = 0.0;
  e_f12_os = 0.0;
  e_ord_os = 0.0;

  std::vector<double> f0(Size(ext) * ext, 0.0);
  rm_gemm_host(true, false, ext, ext, aux, 1.0, fit_i, ext, res_g_j, ext, 0.0, f0.data(), ext);
  rm_gemm_host(true, false, ext, ext, aux, 1.0, raw_g_i, ext, fit_j, ext, 1.0, f0.data(), ext);

  std::vector<double> ft;
  if (!diag) {
    ft.resize(Size(ext) * ext);
    for (int p = 0; p < ext; ++p) {
      for (int r = 0; r < ext; ++r) ft[Size(p) * ext + r] = f0[Size(r) * ext + p];
    }
  }

  std::vector<double> G_obs(Size(obs) * obs, 0.0);
  std::vector<double> G_oc(Size(o) * c, 0.0);
  std::vector<double> G_co(Size(c) * o, 0.0);
  rm_gemm_host(true, false, obs, obs, aux, 1.0, fit_i, ext, raw_coul_j, ext, 0.0, G_obs.data(),
               obs);
  rm_gemm_host(true, false, o, c, aux, 1.0, fit_i, ext, raw_coul_j + obs, ext, 0.0, G_oc.data(), c);
  rm_gemm_host(true, false, c, o, aux, 1.0, fit_i + obs, ext, raw_coul_j, ext, 0.0, G_co.data(), o);

  const double fg_ij = robust_elem_host(fit_i, raw_gc_i, fit_j, res_gc_j, aux, ext, i, j);
  const double f2_ij = robust_elem_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, i, j);
  const double uf_ij = robust_elem_host(fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, i, j);

  // dot(squared_m, fk_n) + dot(squared_n, fk_m) for direct (m=i, n=j) and exchange (m=j, n=i).
  const double* hi = H + Size(i) * ext;
  const double* hj = H + Size(j) * ext;
  double sq_fk_d = 0.0;
  double sq_fk_x = 0.0;
  for (int A = 0; A < aux; ++A) {
    const double* fi = fit_i + Size(A) * ext;
    const double* ri = raw_sq_i + Size(A) * ext;
    const double* fj = fit_j + Size(A) * ext;
    const double* sj = res_sq_j + Size(A) * ext;
    double sj_hj = 0.0, fj_hj = 0.0, fi_hi = 0.0, ri_hi = 0.0;
    double sj_hi = 0.0, fj_hi = 0.0, fi_hj = 0.0, ri_hj = 0.0;
    for (int p = 0; p < ext; ++p) {
      const double hip = hi[p];
      const double hjp = hj[p];
      sj_hj += sj[p] * hjp;
      fj_hj += fj[p] * hjp;
      fi_hi += fi[p] * hip;
      ri_hi += ri[p] * hip;
      if (!diag) {
        sj_hi += sj[p] * hip;
        fj_hi += fj[p] * hip;
        fi_hj += fi[p] * hjp;
        ri_hj += ri[p] * hjp;
      }
    }
    sq_fk_d += fi[i] * sj_hj + ri[i] * fj_hj + fi_hi * sj[j] + ri_hi * fj[j];
    if (!diag) {
      sq_fk_x += fi[j] * sj_hi + ri[j] * fj_hi + fi_hj * sj[i] + ri_hj * fj[i];
    }
  }

  const double V_d = fg_ij - dot_rect_host(G_obs.data(), obs, f0.data(), ext, obs, obs) -
                     dot_rect_host(G_oc.data(), c, f0.data() + obs, ext, o, c) -
                     dot_rect_host(G_co.data(), o, f0.data() + Size(obs) * ext, ext, c, o);
  const double X_d = f2_ij - sqnorm_rect_host(f0.data(), ext, obs, obs) -
                     sqnorm_rect_host(f0.data() + obs, ext, o, c) -
                     sqnorm_rect_host(f0.data() + Size(obs) * ext, ext, c, o);

  BLeftScratchHost Lf, Lft;
  compute_b_left_host(f0.data(), F, K, o, obs, ext, Lf);
  double B_d = 0.0;
  double V_x = 0.0, X_x = 0.0, B_x = 0.0;
  if (diag) {
    B_d = uf_ij + sq_fk_d + 2.0 * eval_b_overlap_host(Lf, f0.data(), o, obs, ext);
  } else {
    const double fg_ji = robust_elem_host(fit_i, raw_gc_i, fit_j, res_gc_j, aux, ext, j, i);
    const double f2_ji = robust_elem_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, j, i);
    const double uf_ji = robust_elem_host(fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, j, i);
    V_x = fg_ji - dot_rect_host(G_obs.data(), obs, ft.data(), ext, obs, obs) -
          dot_rect_host(G_oc.data(), c, ft.data() + obs, ext, o, c) -
          dot_rect_host(G_co.data(), o, ft.data() + Size(obs) * ext, ext, c, o);
    X_x = f2_ji - dot_rect_host(f0.data(), ext, ft.data(), ext, obs, obs) -
          dot_rect_host(f0.data() + obs, ext, ft.data() + obs, ext, o, c) -
          dot_rect_host(f0.data() + Size(obs) * ext, ext, ft.data() + Size(obs) * ext, ext, c, o);
    compute_b_left_host(ft.data(), F, K, o, obs, ext, Lft);
    B_d = uf_ij + sq_fk_d + eval_b_overlap_host(Lf, f0.data(), o, obs, ext) +
          eval_b_overlap_host(Lft, ft.data(), o, obs, ext);
    B_x = uf_ji + sq_fk_x + eval_b_overlap_host(Lf, ft.data(), o, obs, ext) +
          eval_b_overlap_host(Lft, f0.data(), o, obs, ext);
  }

  std::vector<double> C_d(Size(v) * v, 0.0);
  rm_gemm_host(false, true, v, v, c, 1.0, f0.data() + Size(o) * ext + obs, ext,
               F + Size(o) * ext + obs, ext, 0.0, C_d.data(), v);
  rm_gemm_host(false, false, v, v, c, 1.0, F + Size(o) * ext + obs, ext,
               f0.data() + Size(obs) * ext + o, ext, 1.0, C_d.data(), v);

  const double occ_sum = F[Size(i) * ext + i] + F[Size(j) * ext + j];
  if (diag) {
    double vd_cg = 0.0, bd_cc = 0.0, os_gg = 0.0;
    for (int a = 0; a < v; ++a) {
      const double ea = F[Size(o + a) * ext + (o + a)];
      for (int b = 0; b < v; ++b) {
        const double gap = ea + F[Size(o + b) * ext + (o + b)] - occ_sum;
        if (!std::isfinite(gap) || gap == 0.0) return -4;
        const double inv_g = 1.0 / gap;
        const double gab = G_obs[Size(o + a) * obs + (o + b)];
        const double cab = C_d[Size(a) * v + b];
        vd_cg += cab * gab * inv_g;
        bd_cc += cab * cab * inv_g;
        if (include_ordinary) os_gg += gab * gab * inv_g;
      }
    }
    e_f12_os = (V_d - vd_cg) + 0.25 * (B_d - occ_sum * X_d - bd_cc);
    if (include_ordinary) e_ord_os = -os_gg;
  } else {
    double ss_gc = 0.0, ss_cc = 0.0, ss_gg = 0.0;
    double os_term = 0.0, os_gg = 0.0;
    for (int a = 0; a < v; ++a) {
      const double ea = F[Size(o + a) * ext + (o + a)];
      for (int b = 0; b < v; ++b) {
        const double gap = ea + F[Size(o + b) * ext + (o + b)] - occ_sum;
        if (!std::isfinite(gap) || gap == 0.0) return -4;
        const double inv_g = 1.0 / gap;
        const double gab = G_obs[Size(o + a) * obs + (o + b)];
        const double gba = G_obs[Size(o + b) * obs + (o + a)];
        const double cab = C_d[Size(a) * v + b];
        const double cba = C_d[Size(b) * v + a];
        const double ag = gab - gba;
        const double ac = cab - cba;
        ss_gc += ag * ac * inv_g;
        ss_cc += ac * ac * inv_g;
        if (include_ordinary) ss_gg += ag * ag * inv_g;
        if (compute_both_rhf) {
          const double c_os = 0.375 * cab + 0.125 * cba;
          os_term += c_os * (2.0 * gab + c_os) * inv_g;
          if (include_ordinary) os_gg += gab * gab * inv_g;
        }
      }
    }
    const double lin_ss = (V_d - V_x) - 0.5 * ss_gc;
    const double quad_ss = ((B_d - B_x) - occ_sum * (X_d - X_x)) - 0.5 * ss_cc;
    e_f12_ss = 0.5 * lin_ss + (1.0 / 16.0) * quad_ss;
    if (include_ordinary) e_ord_ss = -0.5 * ss_gg;
    if (compute_both_rhf) {
      const double lin_os = 0.375 * V_d + 0.125 * V_x;
      const double norm_os = (5.0 / 32.0) * X_d + (3.0 / 32.0) * X_x;
      const double b_os = (5.0 / 32.0) * B_d + (3.0 / 32.0) * B_x;
      e_f12_os = 2.0 * lin_os + b_os - occ_sum * norm_os - os_term;
      if (include_ordinary) e_ord_os = -os_gg;
    }
  }
  return 0;
}

int eval_uhf_opp_spin_channel_host(
    bool include_ordinary, bool has_occ_weights, int oa, int ob, int fa, int fb, int obs, int ext,
    int aux, const double* occ_w_a, const double* occ_w_b, const std::vector<double>& fit_a_oqe,
    const std::vector<double>& fit_b_oqe, const std::vector<double>& raw_coul_b_oqe,
    const std::vector<double>& raw_g_a_oqe, const std::vector<double>& res_g_b_oqe,
    const std::vector<double>& raw_sq_a_oqe, const std::vector<double>& res_sq_b_oqe,
    const std::vector<double>& raw_gc_a_oqe, const std::vector<double>& res_gc_b_oqe,
    const std::vector<double>& raw_dc_a_oqe, const std::vector<double>& res_dc_b_oqe,
    const double* Fa, const double* Fb, const double* Ka, const double* Kb, const double* Sab,
    double& e_f12_tot, double& e_ord_tot, double* pair_f12_ab, double* pair_ord_ab) {
  e_f12_tot = 0.0;
  e_ord_tot = 0.0;
  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  if (oa_act <= 0 || ob_act <= 0) return 0;

  const int va = obs - oa;
  const int vb = obs - ob;
  const int c = ext - obs;
  const Size ee = Size(ext) * ext;
  const Size qe = Size(aux) * ext;

  // Precompute cross-overlap / Fock+K helper matrices once for the entire AB channel.
  std::vector<double> Ha(ee), Hb(ee);
  for (Size p = 0; p < ee; ++p) {
    Ha[p] = Fa[p] + Ka[p];
    Hb[p] = Fb[p] + Kb[p];
  }
  std::vector<double> SS_T(ee, 0.0), ST_S(ee, 0.0);
  std::vector<double> Ha_S(ee, 0.0), Hb_ST(ee, 0.0);
  std::vector<double> ST_Ha(ee, 0.0), S_Hb(ee, 0.0);
  std::vector<double> ST_Ha_S(ee, 0.0), S_Hb_ST(ee, 0.0);
  rm_gemm_host(false, true, ext, ext, ext, 1.0, Sab, ext, Sab, ext, 0.0, SS_T.data(), ext);
  rm_gemm_host(true, false, ext, ext, ext, 1.0, Sab, ext, Sab, ext, 0.0, ST_S.data(), ext);
  rm_gemm_host(false, false, ext, ext, ext, 1.0, Ha.data(), ext, Sab, ext, 0.0, Ha_S.data(), ext);
  rm_gemm_host(false, true, ext, ext, ext, 1.0, Hb.data(), ext, Sab, ext, 0.0, Hb_ST.data(), ext);
  rm_gemm_host(true, false, ext, ext, ext, 1.0, Sab, ext, Ha.data(), ext, 0.0, ST_Ha.data(), ext);
  rm_gemm_host(false, false, ext, ext, ext, 1.0, Sab, ext, Hb.data(), ext, 0.0, S_Hb.data(), ext);
  rm_gemm_host(true, false, ext, ext, ext, 1.0, Sab, ext, Ha_S.data(), ext, 0.0, ST_Ha_S.data(),
               ext);
  rm_gemm_host(false, false, ext, ext, ext, 1.0, Sab, ext, Hb_ST.data(), ext, 0.0, S_Hb_ST.data(),
               ext);

  std::vector<double> u_j(ext), v_i(ext), a_i(ext), b_j(ext);
  std::vector<double> ha_i(ext), hb_j(ext), comb_rhs_i(ext), comb_lhs_j(ext);
  std::vector<double> st_ha_s_j(ext), s_hb_st_i(ext);
  std::vector<double> tmp_aux1(aux), tmp_aux2(aux);
  std::vector<double> f0(ee), tmp_f(ee), f(ee);
  std::vector<double> G_obs(Size(obs) * obs), G_oc(Size(oa) * c), G_co(Size(c) * ob);
  std::vector<double> A1(Size(obs) * vb), A2(Size(obs) * vb), A3(Size(c) * vb);
  std::vector<double> B1(Size(oa) * c), B2(Size(ext - oa) * c);
  std::vector<double> C1(Size(va) * obs), C2(Size(va) * obs), C3(Size(va) * c);
  std::vector<double> D1(Size(c) * ob), D2(Size(c) * (ext - ob));
  std::vector<double> occ_a_tmp(Size(oa) * ext), occ_b_tmp(Size(ext) * ob);
  std::vector<double> C_vv(Size(va) * vb);

  for (int i_loc = 0; i_loc < oa_act; ++i_loc) {
    const int i = fa + i_loc;
    const double* fit_i = fit_a_oqe.data() + Size(i_loc) * qe;
    const double* raw_g_i = raw_g_a_oqe.data() + Size(i_loc) * qe;
    const double* raw_sq_i = raw_sq_a_oqe.data() + Size(i_loc) * qe;
    const double* raw_gc_i = raw_gc_a_oqe.data() + Size(i_loc) * qe;
    const double* raw_dc_i = raw_dc_a_oqe.data() + Size(i_loc) * qe;
    for (int p = 0; p < ext; ++p) {
      v_i[p] = Sab[Size(i) * ext + p];
      a_i[p] = SS_T[Size(p) * ext + i];
      ha_i[p] = Ha[Size(p) * ext + i];
      comb_rhs_i[p] = ST_Ha[Size(p) * ext + i] + Hb_ST[Size(p) * ext + i];
      s_hb_st_i[p] = S_Hb_ST[Size(p) * ext + i];
    }

    for (int j_loc = 0; j_loc < ob_act; ++j_loc) {
      const int j = fb + j_loc;
      const double* fit_j = fit_b_oqe.data() + Size(j_loc) * qe;
      const double* raw_coul_j = raw_coul_b_oqe.data() + Size(j_loc) * qe;
      const double* res_g_j = res_g_b_oqe.data() + Size(j_loc) * qe;
      const double* res_sq_j = res_sq_b_oqe.data() + Size(j_loc) * qe;
      const double* res_gc_j = res_gc_b_oqe.data() + Size(j_loc) * qe;
      const double* res_dc_j = res_dc_b_oqe.data() + Size(j_loc) * qe;
      for (int p = 0; p < ext; ++p) {
        u_j[p] = Sab[Size(p) * ext + j];
        b_j[p] = ST_S[Size(j) * ext + p];
        hb_j[p] = Hb[Size(p) * ext + j];
        comb_lhs_j[p] = S_Hb[Size(p) * ext + j] + Ha_S[Size(p) * ext + j];
        st_ha_s_j[p] = ST_Ha_S[Size(p) * ext + j];
      }

      // 1. O(aux * ext) rank-structured scalars for gc, dc, sq_norm, sq_fk.
      const double gc_val =
          0.375 * robust_elem_host(fit_i, raw_gc_i, fit_j, res_gc_j, aux, ext, i, j) +
          0.125 * robust_bilin_host(fit_i, raw_gc_i, fit_j, res_gc_j, aux, ext, u_j.data(),
                                    v_i.data(), tmp_aux1, tmp_aux2);

      const double dc_val =
          (9.0 / 64.0) * robust_elem_host(fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, i, j) +
          (6.0 / 64.0) * robust_bilin_host(fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, u_j.data(),
                                           v_i.data(), tmp_aux1, tmp_aux2) +
          (1.0 / 64.0) * robust_bilin_host(fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, a_i.data(),
                                           b_j.data(), tmp_aux1, tmp_aux2);

      const double sq_norm_val =
          (9.0 / 64.0) * robust_elem_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, i, j) +
          (6.0 / 64.0) * robust_bilin_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, u_j.data(),
                                           v_i.data(), tmp_aux1, tmp_aux2) +
          (1.0 / 64.0) * robust_bilin_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, a_i.data(),
                                           b_j.data(), tmp_aux1, tmp_aux2);

      // <M^(sq), W(Y)>: 6 rank-1 terms (2 single-column/row + 4 general bilinear).
      double sq_t1 = 0.0, sq_t3 = 0.0;
      for (int A = 0; A < aux; ++A) {
        const double* fi = fit_i + Size(A) * ext;
        const double* ri = raw_sq_i + Size(A) * ext;
        const double* fj = fit_j + Size(A) * ext;
        const double* sj = res_sq_j + Size(A) * ext;
        double fi_ha = 0.0, ri_ha = 0.0, sj_hb = 0.0, fj_hb = 0.0;
        for (int p = 0; p < ext; ++p) {
          fi_ha += fi[p] * ha_i[p];
          ri_ha += ri[p] * ha_i[p];
          sj_hb += sj[p] * hb_j[p];
          fj_hb += fj[p] * hb_j[p];
        }
        sq_t1 += fi_ha * sj[j] + ri_ha * fj[j];
        sq_t3 += fi[i] * sj_hb + ri[i] * fj_hb;
      }
      const double sq_fk_val =
          (9.0 / 64.0) * (sq_t1 + sq_t3) +
          (3.0 / 64.0) * robust_bilin_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, u_j.data(),
                                           comb_rhs_i.data(), tmp_aux1, tmp_aux2) +
          (3.0 / 64.0) * robust_bilin_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext,
                                           comb_lhs_j.data(), v_i.data(), tmp_aux1, tmp_aux2) +
          (1.0 / 64.0) * robust_bilin_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, a_i.data(),
                                           st_ha_s_j.data(), tmp_aux1, tmp_aux2) +
          (1.0 / 64.0) * robust_bilin_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext,
                                           s_hb_st_i.data(), b_j.data(), tmp_aux1, tmp_aux2);

      // 2. Geminal f = (3/8) f0 + (1/8) Sab f0^T Sab.
      rm_gemm_host(true, false, ext, ext, aux, 1.0, fit_i, ext, res_g_j, ext, 0.0, f0.data(), ext);
      rm_gemm_host(true, false, ext, ext, aux, 1.0, raw_g_i, ext, fit_j, ext, 1.0, f0.data(), ext);
      rm_gemm_host(false, true, ext, ext, ext, 1.0, Sab, ext, f0.data(), ext, 0.0, tmp_f.data(),
                   ext);
      std::copy_n(f0.data(), ee, f.data());
      rm_gemm_host(false, false, ext, ext, ext, 0.125, tmp_f.data(), ext, Sab, ext, 0.375, f.data(),
                   ext);

      // 3. Coulomb blocks G_obs, G_oc, G_co.
      rm_gemm_host(true, false, obs, obs, aux, 1.0, fit_i, ext, raw_coul_j, ext, 0.0, G_obs.data(),
                   obs);
      rm_gemm_host(true, false, oa, c, aux, 1.0, fit_i, ext, raw_coul_j + obs, ext, 0.0,
                   G_oc.data(), c);
      rm_gemm_host(true, false, c, ob, aux, 1.0, fit_i + obs, ext, raw_coul_j, ext, 0.0,
                   G_co.data(), ob);

      const double linear =
          gc_val - dot_rect_host(G_obs.data(), obs, f.data(), ext, obs, obs) -
          dot_rect_host(G_oc.data(), c, f.data() + obs, ext, oa, c) -
          dot_rect_host(G_co.data(), ob, f.data() + Size(obs) * ext, ext, c, ob);
      const double norm = sq_norm_val - sqnorm_rect_host(f.data(), ext, obs, obs) -
                          sqnorm_rect_host(f.data() + obs, ext, oa, c) -
                          sqnorm_rect_host(f.data() + Size(obs) * ext, ext, c, ob);

      // 4. Exchange terms -<f, Ka @ f> - <f, f @ Kb^T> (reusing tmp_f).
      rm_gemm_host(false, false, ext, ext, ext, 1.0, Ka, ext, f.data(), ext, 0.0, tmp_f.data(),
                   ext);
      double b_val = dc_val + sq_fk_val - dot_rect_host(f.data(), ext, tmp_f.data(), ext, ext, ext);
      rm_gemm_host(false, true, ext, ext, ext, 1.0, f.data(), ext, Kb, ext, 0.0, tmp_f.data(), ext);
      b_val -= dot_rect_host(f.data(), ext, tmp_f.data(), ext, ext, ext);

      // 5. Complement-block cancellation <qf, Fa qf> - <f, Fa f> (alpha side).
      rm_gemm_host(false, false, ext, ob, ext, 1.0, Fa, ext, f.data(), ext, 0.0, occ_b_tmp.data(),
                   ob);
      b_val -= dot_rect_host(f.data(), ext, occ_b_tmp.data(), ob, ext, ob);

      rm_gemm_host(false, false, obs, vb, obs, 1.0, Fa, ext, f.data() + ob, ext, 0.0, A1.data(), vb);
      rm_gemm_host(false, false, obs, vb, c, 1.0, Fa + obs, ext, f.data() + Size(obs) * ext + ob,
                   ext, 0.0, A2.data(), vb);
      rm_gemm_host(false, false, c, vb, obs, 1.0, Fa + Size(obs) * ext, ext, f.data() + ob, ext,
                   0.0, A3.data(), vb);
      b_val -= dot_rect_host(f.data() + ob, ext, A1.data(), vb, obs, vb) +
               dot_rect_host(f.data() + ob, ext, A2.data(), vb, obs, vb) +
               dot_rect_host(f.data() + Size(obs) * ext + ob, ext, A3.data(), vb, c, vb);

      rm_gemm_host(false, false, oa, c, ext, 1.0, Fa, ext, f.data() + obs, ext, 0.0, B1.data(), c);
      rm_gemm_host(false, false, ext - oa, c, oa, 1.0, Fa + Size(oa) * ext, ext, f.data() + obs,
                   ext, 0.0, B2.data(), c);
      b_val -= dot_rect_host(f.data() + obs, ext, B1.data(), c, oa, c) +
               dot_rect_host(f.data() + Size(oa) * ext + obs, ext, B2.data(), c, ext - oa, c);

      // 6. Complement-block cancellation <qf, qf Fb^T> - <f, f Fb^T> (beta side).
      rm_gemm_host(false, true, oa, ext, ext, 1.0, f.data(), ext, Fb, ext, 0.0, occ_a_tmp.data(),
                   ext);
      b_val -= dot_rect_host(f.data(), ext, occ_a_tmp.data(), ext, oa, ext);

      rm_gemm_host(false, true, va, obs, obs, 1.0, f.data() + Size(oa) * ext, ext, Fb, ext, 0.0,
                   C1.data(), obs);
      rm_gemm_host(false, true, va, obs, c, 1.0, f.data() + Size(oa) * ext + obs, ext, Fb + obs,
                   ext, 0.0, C2.data(), obs);
      rm_gemm_host(false, true, va, c, obs, 1.0, f.data() + Size(oa) * ext, ext,
                   Fb + Size(obs) * ext, ext, 0.0, C3.data(), c);
      b_val -= dot_rect_host(f.data() + Size(oa) * ext, ext, C1.data(), obs, va, obs) +
               dot_rect_host(f.data() + Size(oa) * ext, ext, C2.data(), obs, va, obs) +
               dot_rect_host(f.data() + Size(oa) * ext + obs, ext, C3.data(), c, va, c);

      rm_gemm_host(false, true, c, ob, ext, 1.0, f.data() + Size(obs) * ext, ext, Fb, ext, 0.0,
                   D1.data(), ob);
      rm_gemm_host(false, true, c, ext - ob, ob, 1.0, f.data() + Size(obs) * ext, ext,
                   Fb + Size(ob) * ext, ext, 0.0, D2.data(), ext - ob);
      b_val -= dot_rect_host(f.data() + Size(obs) * ext, ext, D1.data(), ob, c, ob) +
               dot_rect_host(f.data() + Size(obs) * ext + ob, ext, D2.data(), ext - ob, c,
                             ext - ob);

      // 7. Coupling C[a, b] = A2[oa + a, b] + C2[a, ob + b] and denominator contraction.
      const double occ_sum = Fa[Size(i) * ext + i] + Fb[Size(j) * ext + j];
      double os_term = 0.0, os_gg = 0.0;
      for (int a = 0; a < va; ++a) {
        const double ea = Fa[Size(oa + a) * ext + (oa + a)];
        const double* a2_row = A2.data() + Size(oa + a) * vb;
        const double* c2_row = C2.data() + Size(a) * obs + ob;
        const double* g_row = G_obs.data() + Size(oa + a) * obs + ob;
        for (int b = 0; b < vb; ++b) {
          const double gap = ea + Fb[Size(ob + b) * ext + (ob + b)] - occ_sum;
          if (!std::isfinite(gap) || gap == 0.0) return -4;
          const double inv_g = 1.0 / gap;
          const double cab = a2_row[b] + c2_row[b];
          const double gab = g_row[b];
          os_term += cab * (2.0 * gab + cab) * inv_g;
          if (include_ordinary) os_gg += gab * gab * inv_g;
        }
      }

      const double e_f12 = 2.0 * linear + b_val - occ_sum * norm - os_term;
      const double e_ord = include_ordinary ? (-os_gg) : 0.0;
      const Size idx = Size(i_loc) * ob_act + j_loc;
      if (pair_f12_ab) pair_f12_ab[idx] = e_f12;
      if (include_ordinary && pair_ord_ab) pair_ord_ab[idx] = e_ord;
      const double wij = has_occ_weights ? 0.5 * (occ_w_a[i_loc] + occ_w_b[j_loc]) : 1.0;
      e_f12_tot += wij * e_f12;
      if (include_ordinary) e_ord_tot += wij * e_ord;
    }
  }
  return 0;
}

// -------------------------------------------------------------------------
// Host-Reference MP2-F12 Pair Response Pullback (General UHF + RHF Spatial)
// -------------------------------------------------------------------------

void transpose_oqe_to_qoe_host(const std::vector<double>& in_oqe, int aux, int o_act, int ext,
                               std::vector<double>& out_qoe) {
  out_qoe.resize(Size(aux) * Size(o_act) * ext);
  for (int i = 0; i < o_act; ++i) {
    for (int A = 0; A < aux; ++A) {
      const double* s = in_oqe.data() + (Size(i) * aux + A) * ext;
      double* d = out_qoe.data() + (Size(A) * o_act + i) * ext;
      std::copy_n(s, ext, d);
    }
  }
}

void scatter_active_qoe_host(const std::vector<double>& in_qoe, int aux, int occ, int f, int o_act,
                             int ext, double* dst) {
  if (!dst || o_act <= 0) return;
  for (int A = 0; A < aux; ++A) {
    const double* s = in_qoe.data() + Size(A) * o_act * ext;
    double* d = dst + (Size(A) * occ + f) * ext;
    std::copy_n(s, Size(o_act) * ext, d);
  }
}

void scatter_active_oqe_host(const std::vector<double>& in_oqe, int aux, int occ, int f, int o_act,
                             int ext, double* dst) {
  if (!dst || o_act <= 0) return;
  for (int A = 0; A < aux; ++A) {
    for (int i = 0; i < o_act; ++i) {
      const double* s = in_oqe.data() + (Size(i) * aux + A) * ext;
      double* d = dst + (Size(A) * occ + (f + i)) * ext;
      std::copy_n(s, ext, d);
    }
  }
}

void pullback_res_oqe_host(const std::vector<double>& bar_res_oqe,
                           const std::vector<double>& fit_qoe, const double* metric,
                           double metric_scale, bool sym_metric, int aux, int o_act, int ext,
                           std::vector<double>& bar_raw_oqe, std::vector<double>& bar_fit_qoe,
                           double* bar_metric) {
  if (o_act <= 0) return;
  const Size n_tot = bar_res_oqe.size();
  for (Size idx = 0; idx < n_tot; ++idx) bar_raw_oqe[idx] += bar_res_oqe[idx];
  std::vector<double> bar_res_qoe;
  transpose_oqe_to_qoe_host(bar_res_oqe, aux, o_act, ext, bar_res_qoe);
  const int cols = o_act * ext;
  rm_gemm_host(true, false, aux, cols, aux, -1.0, metric, aux, bar_res_qoe.data(), cols, 1.0,
               bar_fit_qoe.data(), cols);
  if (bar_metric) {
    if (!sym_metric) {
      rm_gemm_host(false, true, aux, aux, cols, -metric_scale, bar_res_qoe.data(), cols,
                   fit_qoe.data(), cols, 1.0, bar_metric, aux);
    } else {
      std::vector<double> m_tmp(Size(aux) * aux, 0.0);
      rm_gemm_host(false, true, aux, aux, cols, -0.5 * metric_scale, bar_res_qoe.data(), cols,
                   fit_qoe.data(), cols, 0.0, m_tmp.data(), aux);
      for (int p = 0; p < aux; ++p) {
        for (int q = 0; q < aux; ++q) {
          bar_metric[Size(p) * aux + q] += m_tmp[Size(p) * aux + q] + m_tmp[Size(q) * aux + p];
        }
      }
    }
  }
}

inline void robust_elem_pullback_host(double alpha, int p, int r, int aux, int ext,
                                      const double* fit_i, const double* raw_i,
                                      const double* fit_j, const double* res_j, double* bar_fit_i,
                                      double* bar_raw_i, double* bar_fit_j, double* bar_res_j) {
  if (alpha == 0.0) return;
  for (int A = 0; A < aux; ++A) {
    const Size row = Size(A) * ext;
    bar_fit_i[row + p] += alpha * res_j[row + r];
    bar_res_j[row + r] += alpha * fit_i[row + p];
    bar_raw_i[row + p] += alpha * fit_j[row + r];
    bar_fit_j[row + r] += alpha * raw_i[row + p];
  }
}

void robust_bilin_pullback_host(double alpha, const double* x, const double* y, int aux, int ext,
                                const double* fit_i, const double* raw_i, const double* fit_j,
                                const double* res_j, double* bar_fit_i, double* bar_raw_i,
                                double* bar_fit_j, double* bar_res_j) {
  if (alpha == 0.0) return;
  for (int A = 0; A < aux; ++A) {
    const Size row = Size(A) * ext;
    const double* fi = fit_i + row;
    const double* ri = raw_i + row;
    const double* fj = fit_j + row;
    const double* sj = res_j + row;
    double fx = 0.0, rx = 0.0, fy = 0.0, sy = 0.0;
    for (int p = 0; p < ext; ++p) {
      const double xp = x[p];
      const double yp = y[p];
      fx += fi[p] * xp;
      rx += ri[p] * xp;
      fy += fj[p] * yp;
      sy += sj[p] * yp;
    }
    const double a_sy = alpha * sy;
    const double a_fx = alpha * fx;
    const double a_fy = alpha * fy;
    const double a_rx = alpha * rx;
    double* bfi = bar_fit_i + row;
    double* bri = bar_raw_i + row;
    double* bfj = bar_fit_j + row;
    double* bsj = bar_res_j + row;
    for (int p = 0; p < ext; ++p) {
      const double xp = x[p];
      const double yp = y[p];
      bfi[p] += a_sy * xp;
      bsj[p] += a_fx * yp;
      bri[p] += a_fy * xp;
      bfj[p] += a_rx * yp;
    }
  }
}

void robust_row_col_pullback_host(double alpha_row, int row_idx, const double* y_row,
                                  double alpha_col, int col_idx, const double* x_col, int aux,
                                  int ext, const double* fit_i, const double* raw_i,
                                  const double* fit_j, const double* res_j, double* bar_fit_i,
                                  double* bar_raw_i, double* bar_fit_j, double* bar_res_j) {
  // Accumulates bar_M += alpha_row * e_{row_idx} y_row^T + alpha_col * x_col e_{col_idx}^T.
  if (alpha_row == 0.0 && alpha_col == 0.0) return;
  for (int A = 0; A < aux; ++A) {
    const Size row = Size(A) * ext;
    const double* fi = fit_i + row;
    const double* ri = raw_i + row;
    const double* fj = fit_j + row;
    const double* sj = res_j + row;
    double* bfi = bar_fit_i + row;
    double* bri = bar_raw_i + row;
    double* bfj = bar_fit_j + row;
    double* bsj = bar_res_j + row;
    if (alpha_row != 0.0) {
      double sy = 0.0, fy = 0.0;
      for (int p = 0; p < ext; ++p) {
        const double yp = y_row[p];
        sy += sj[p] * yp;
        fy += fj[p] * yp;
      }
      bfi[row_idx] += alpha_row * sy;
      bri[row_idx] += alpha_row * fy;
      const double afi = alpha_row * fi[row_idx];
      const double ari = alpha_row * ri[row_idx];
      for (int p = 0; p < ext; ++p) {
        const double yp = y_row[p];
        bsj[p] += afi * yp;
        bfj[p] += ari * yp;
      }
    }
    if (alpha_col != 0.0) {
      double fx = 0.0, rx = 0.0;
      for (int p = 0; p < ext; ++p) {
        const double xp = x_col[p];
        fx += fi[p] * xp;
        rx += ri[p] * xp;
      }
      bsj[col_idx] += alpha_col * fx;
      bfj[col_idx] += alpha_col * rx;
      const double asj = alpha_col * sj[col_idx];
      const double afj = alpha_col * fj[col_idx];
      for (int p = 0; p < ext; ++p) {
        const double xp = x_col[p];
        bfi[p] += asj * xp;
        bri[p] += afj * xp;
      }
    }
  }
}

void robust_rmatvec_host(const double* fit_i, const double* raw_i, const double* fit_j,
                         const double* res_j, int aux, int ext, const double* y, double* out) {
  // out = (fit_i^T res_j + raw_i^T fit_j) y
  std::fill_n(out, ext, 0.0);
  for (int A = 0; A < aux; ++A) {
    const Size row = Size(A) * ext;
    const double* fi = fit_i + row;
    const double* ri = raw_i + row;
    const double* fj = fit_j + row;
    const double* sj = res_j + row;
    double sy = 0.0, fy = 0.0;
    for (int p = 0; p < ext; ++p) {
      const double yp = y[p];
      sy += sj[p] * yp;
      fy += fj[p] * yp;
    }
    for (int p = 0; p < ext; ++p) out[p] += fi[p] * sy + ri[p] * fy;
  }
}

void robust_lmatvec_host(const double* fit_i, const double* raw_i, const double* fit_j,
                         const double* res_j, int aux, int ext, const double* x, double* out) {
  // out = (fit_i^T res_j + raw_i^T fit_j)^T x
  std::fill_n(out, ext, 0.0);
  for (int A = 0; A < aux; ++A) {
    const Size row = Size(A) * ext;
    const double* fi = fit_i + row;
    const double* ri = raw_i + row;
    const double* fj = fit_j + row;
    const double* sj = res_j + row;
    double fx = 0.0, rx = 0.0;
    for (int p = 0; p < ext; ++p) {
      const double xp = x[p];
      fx += fi[p] * xp;
      rx += ri[p] * xp;
    }
    for (int p = 0; p < ext; ++p) out[p] += fx * sj[p] + rx * fj[p];
  }
}

void robust_row_host(const double* fit_i, const double* raw_i, const double* fit_j,
                     const double* res_j, int aux, int ext, int row_idx, double* out) {
  // out[q] = M[row_idx, q]
  std::fill_n(out, ext, 0.0);
  for (int A = 0; A < aux; ++A) {
    const Size row = Size(A) * ext;
    const double f_val = fit_i[row + row_idx];
    const double r_val = raw_i[row + row_idx];
    const double* fj = fit_j + row;
    const double* sj = res_j + row;
    for (int p = 0; p < ext; ++p) out[p] += f_val * sj[p] + r_val * fj[p];
  }
}

void robust_col_host(const double* fit_i, const double* raw_i, const double* fit_j,
                     const double* res_j, int aux, int ext, int col_idx, double* out) {
  // out[p] = M[p, col_idx]
  std::fill_n(out, ext, 0.0);
  for (int A = 0; A < aux; ++A) {
    const Size row = Size(A) * ext;
    const double s_val = res_j[row + col_idx];
    const double f_val = fit_j[row + col_idx];
    const double* fi = fit_i + row;
    const double* ri = raw_i + row;
    for (int p = 0; p < ext; ++p) out[p] += fi[p] * s_val + ri[p] * f_val;
  }
}

void matvec_host(bool trans, const double* M, int ext, const double* x, double* y) {
  std::fill_n(y, ext, 0.0);
  if (!trans) {
    for (int p = 0; p < ext; ++p) {
      const double* mp = M + Size(p) * ext;
      double acc = 0.0;
      for (int q = 0; q < ext; ++q) acc += mp[q] * x[q];
      y[p] = acc;
    }
  } else {
    for (int p = 0; p < ext; ++p) {
      const double xp = x[p];
      if (xp == 0.0) continue;
      const double* mp = M + Size(p) * ext;
      for (int q = 0; q < ext; ++q) y[q] += mp[q] * xp;
    }
  }
}

void add_outer_host(double alpha, const double* x, const double* y, int ext, double* M) {
  if (alpha == 0.0) return;
  for (int p = 0; p < ext; ++p) {
    const double ax = alpha * x[p];
    if (ax == 0.0) continue;
    double* mp = M + Size(p) * ext;
    for (int q = 0; q < ext; ++q) mp[q] += ax * y[q];
  }
}

void zero_p_blocks_host(double* M, int oa, int ob, int obs, int ext) {
  // Zeroes the strong-orthogonality complement P = (obs,obs) U (oa,c) U (c,ob).
  const int c = ext - obs;
  for (int p = 0; p < obs; ++p) std::fill_n(M + Size(p) * ext, obs, 0.0);
  for (int p = 0; p < oa; ++p) std::fill_n(M + Size(p) * ext + obs, c, 0.0);
  for (int p = obs; p < ext; ++p) std::fill_n(M + Size(p) * ext, ob, 0.0);
}

void b_overlap_pullback_host(double scale, const double* A, const double* B, const double* F,
                             const double* K, int o, int obs, int ext, double* bar_A, double* bar_B,
                             double* bar_F, double* bar_K) {
  // Exact adjoint of scale * eval_b_overlap_host(compute_b_left(A, F, K), B).
  if (scale == 0.0) return;
  const int v = obs - o;
  const int c = ext - obs;
  // Term 0: -scale * <A @ K, B>
  rm_gemm_host(false, true, ext, ext, ext, -scale, B, ext, K, ext, 1.0, bar_A, ext);
  rm_gemm_host(false, false, ext, ext, ext, -scale, A, ext, K, ext, 1.0, bar_B, ext);
  rm_gemm_host(true, false, ext, ext, ext, -scale, A, ext, B, ext, 1.0, bar_K, ext);
  // Term 1: -scale * <A[:o, :] @ F, B[:o, :]>
  rm_gemm_host(false, true, o, ext, ext, -scale, B, ext, F, ext, 1.0, bar_A, ext);
  rm_gemm_host(false, false, o, ext, ext, -scale, A, ext, F, ext, 1.0, bar_B, ext);
  rm_gemm_host(true, false, ext, ext, o, -scale, A, ext, B, ext, 1.0, bar_F, ext);
  // Term 2: +scale * <A[obs:, :o] @ F[:o, :o], B[obs:, :o]>
  rm_gemm_host(false, true, c, o, o, scale, B + Size(obs) * ext, ext, F, ext, 1.0,
               bar_A + Size(obs) * ext, ext);
  rm_gemm_host(false, false, c, o, o, scale, A + Size(obs) * ext, ext, F, ext, 1.0,
               bar_B + Size(obs) * ext, ext);
  rm_gemm_host(true, false, o, o, c, scale, A + Size(obs) * ext, ext, B + Size(obs) * ext, ext, 1.0,
               bar_F, ext);
  // Term 3: -scale * <A[o:obs, :obs] @ F[:obs, :obs], B[o:obs, :obs]>
  rm_gemm_host(false, true, v, obs, obs, -scale, B + Size(o) * ext, ext, F, ext, 1.0,
               bar_A + Size(o) * ext, ext);
  rm_gemm_host(false, false, v, obs, obs, -scale, A + Size(o) * ext, ext, F, ext, 1.0,
               bar_B + Size(o) * ext, ext);
  rm_gemm_host(true, false, obs, obs, v, -scale, A + Size(o) * ext, ext, B + Size(o) * ext, ext,
               1.0, bar_F, ext);
  // Term 4: -2 * scale * <A[obs:, :] @ F[:o, :]^T, B[obs:, :o]>
  rm_gemm_host(false, false, c, ext, o, -2.0 * scale, B + Size(obs) * ext, ext, F, ext, 1.0,
               bar_A + Size(obs) * ext, ext);
  rm_gemm_host(false, true, c, o, ext, -2.0 * scale, A + Size(obs) * ext, ext, F, ext, 1.0,
               bar_B + Size(obs) * ext, ext);
  rm_gemm_host(true, false, o, ext, c, -2.0 * scale, B + Size(obs) * ext, ext,
               A + Size(obs) * ext, ext, 1.0, bar_F, ext);
  // Term 5: -2 * scale * <A[o:obs, :obs] @ F[:obs, obs:], B[o:obs, obs:]>
  rm_gemm_host(false, true, v, obs, c, -2.0 * scale, B + Size(o) * ext + obs, ext, F + obs, ext,
               1.0, bar_A + Size(o) * ext, ext);
  rm_gemm_host(false, false, v, c, obs, -2.0 * scale, A + Size(o) * ext, ext, F + obs, ext, 1.0,
               bar_B + Size(o) * ext + obs, ext);
  rm_gemm_host(true, false, obs, c, v, -2.0 * scale, A + Size(o) * ext, ext,
               B + Size(o) * ext + obs, ext, 1.0, bar_F + obs, ext);
}

void weight_rank1_pullback_host(double coeff, const double* x, const double* y, const double* Sab,
                                int aux, int ext, const double* fit_i, const double* raw_i,
                                const double* fit_j, const double* res_j, double* bar_fit_i,
                                double* bar_raw_i, double* bar_fit_j, double* bar_res_j,
                                double* bar_Sab, std::vector<double>& sy_buf,
                                std::vector<double>& stx_buf, std::vector<double>& mv1_buf,
                                std::vector<double>& mv2_buf) {
  // Exact adjoint of <coeff * x y^T, W(M)> wrt M and Sab, where W(M) = (3/8)M + (1/8) Sab M^T Sab.
  if (coeff == 0.0) return;
  matvec_host(false, Sab, ext, y, sy_buf.data());
  matvec_host(true, Sab, ext, x, stx_buf.data());
  // bar_M += (3/8) coeff * x y^T + (1/8) coeff * (Sab y)(Sab^T x)^T
  robust_bilin_pullback_host(0.375 * coeff, x, y, aux, ext, fit_i, raw_i, fit_j, res_j, bar_fit_i,
                             bar_raw_i, bar_fit_j, bar_res_j);
  robust_bilin_pullback_host(0.125 * coeff, sy_buf.data(), stx_buf.data(), aux, ext, fit_i, raw_i,
                             fit_j, res_j, bar_fit_i, bar_raw_i, bar_fit_j, bar_res_j);
  // bar_Sab += (1/8) coeff * ( x (M^T (Sab y))^T + (M (Sab^T x)) y^T )
  if (bar_Sab) {
    robust_lmatvec_host(fit_i, raw_i, fit_j, res_j, aux, ext, sy_buf.data(), mv1_buf.data());
    robust_rmatvec_host(fit_i, raw_i, fit_j, res_j, aux, ext, stx_buf.data(), mv2_buf.data());
    add_outer_host(0.125 * coeff, x, mv1_buf.data(), ext, bar_Sab);
    add_outer_host(0.125 * coeff, mv2_buf.data(), y, ext, bar_Sab);
  }
}

void weight_rmatvec_host(const double* Sab, const double* fit_i, const double* raw_i,
                         const double* fit_j, const double* res_j, int aux, int ext,
                         const double* y, double* out, std::vector<double>& tmp1,
                         std::vector<double>& tmp2) {
  // out = W(M) y = (3/8) M y + (1/8) Sab (M^T (Sab y))
  robust_rmatvec_host(fit_i, raw_i, fit_j, res_j, aux, ext, y, out);
  matvec_host(false, Sab, ext, y, tmp1.data());
  robust_lmatvec_host(fit_i, raw_i, fit_j, res_j, aux, ext, tmp1.data(), tmp2.data());
  matvec_host(false, Sab, ext, tmp2.data(), tmp1.data());
  for (int p = 0; p < ext; ++p) out[p] = 0.375 * out[p] + 0.125 * tmp1[p];
}

void weight_lmatvec_host(const double* Sab, const double* fit_i, const double* raw_i,
                         const double* fit_j, const double* res_j, int aux, int ext,
                         const double* x, double* out, std::vector<double>& tmp1,
                         std::vector<double>& tmp2) {
  // out = W(M)^T x = (3/8) M^T x + (1/8) Sab^T (M (Sab^T x))
  robust_lmatvec_host(fit_i, raw_i, fit_j, res_j, aux, ext, x, out);
  matvec_host(true, Sab, ext, x, tmp1.data());
  robust_rmatvec_host(fit_i, raw_i, fit_j, res_j, aux, ext, tmp1.data(), tmp2.data());
  matvec_host(true, Sab, ext, tmp2.data(), tmp1.data());
  for (int p = 0; p < ext; ++p) out[p] = 0.375 * out[p] + 0.125 * tmp1[p];
}

int eval_same_spin_or_rhf_pair_response_host(
    int i, int j, bool compute_both_rhf, bool include_f12, bool include_ordinary, double wij, int o,
    int obs, int ext, int aux, const double* fit_i, const double* fit_j, const double* raw_coul_i,
    const double* raw_coul_j, const double* raw_g_i, const double* res_g_j, const double* raw_sq_i,
    const double* res_sq_j, const double* raw_gc_i, const double* res_gc_j, const double* raw_dc_i,
    const double* res_dc_j, const double* F, const double* K, const double* H, double& e_f12_ss,
    double& e_ord_ss, double& e_f12_os, double& e_ord_os, double* bar_fit_i, double* bar_fit_j,
    double* bar_raw_coul_i, double* bar_raw_coul_j, double* bar_raw_g_i, double* bar_res_g_j,
    double* bar_res_os_g_j, double* bar_raw_sq_i, double* bar_res_sq_j, double* bar_res_os_sq_j,
    double* bar_raw_gc_i, double* bar_res_gc_j, double* bar_res_os_gc_j, double* bar_raw_dc_i,
    double* bar_res_dc_j, double* bar_res_os_dc_j, double* bar_F, double* bar_K,
    double* bar_cross_overlap) {
  const int v = obs - o;
  const int c = ext - obs;
  const bool diag = (i == j);
  const Size ee = Size(ext) * ext;
  e_f12_ss = 0.0;
  e_ord_ss = 0.0;
  e_f12_os = 0.0;
  e_ord_os = 0.0;
  if (!include_f12 && !include_ordinary) return 0;

  const double s_f12 = include_f12 ? wij : 0.0;
  const double s_ord = include_ordinary ? wij : 0.0;
  // For closed-shell RHF diagonal (i == j), AB(i, i) is a single spin pair whose
  // spin-symmetrized alpha/beta pullback shares 0.5x into alpha (mirrored to beta),
  // whereas off-diagonal i < j already pairs AA+BB and AB(i,j)+AB(j,i) at 1.0x alpha.
  const double spin_scale = (compute_both_rhf && diag) ? 0.5 : 1.0;

  const double occ_sum = F[Size(i) * ext + i] + F[Size(j) * ext + j];

  if (!include_f12) {
    // Ordinary-only fast path (no geminal / Fock / exchange / cross_overlap work).
    std::vector<double> G_vv(Size(v) * v, 0.0);
    rm_gemm_host(true, false, v, v, aux, 1.0, fit_i + o, ext, raw_coul_j + o, ext, 0.0,
                 G_vv.data(), v);
    std::vector<double> bar_G_vv(Size(v) * v, 0.0);
    std::vector<double> bar_G_vv_os(compute_both_rhf && !diag ? Size(v) * v : 0, 0.0);
    double bar_occ_sum = 0.0;
    if (diag) {
      double os_gg = 0.0;
      for (int a = 0; a < v; ++a) {
        const double ea = F[Size(o + a) * ext + (o + a)];
        for (int b = 0; b < v; ++b) {
          const double gap = ea + F[Size(o + b) * ext + (o + b)] - occ_sum;
          if (!std::isfinite(gap) || gap == 0.0) return -4;
          const double inv_g = 1.0 / gap;
          const double gab = G_vv[Size(a) * v + b];
          os_gg += gab * gab * inv_g;
          bar_G_vv[Size(a) * v + b] = -2.0 * s_ord * gab * inv_g;
          const double bar_gap = s_ord * gab * gab * inv_g * inv_g;
          bar_F[Size(o + a) * ext + (o + a)] += spin_scale * bar_gap;
          bar_F[Size(o + b) * ext + (o + b)] += spin_scale * bar_gap;
          bar_occ_sum -= bar_gap;
        }
      }
      e_ord_os = -os_gg;
      bar_F[Size(i) * ext + i] += spin_scale * 2.0 * bar_occ_sum;
    } else {
      double ss_gg = 0.0, os_gg = 0.0;
      for (int a = 0; a < v; ++a) {
        const double ea = F[Size(o + a) * ext + (o + a)];
        for (int b = 0; b < v; ++b) {
          const double gap = ea + F[Size(o + b) * ext + (o + b)] - occ_sum;
          if (!std::isfinite(gap) || gap == 0.0) return -4;
          const double inv_g = 1.0 / gap;
          const double gab = G_vv[Size(a) * v + b];
          const double gba = G_vv[Size(b) * v + a];
          const double ag = gab - gba;
          ss_gg += ag * ag * inv_g;
          const double bg_ss = -2.0 * s_ord * ag * inv_g;
          double bgap = 0.5 * s_ord * ag * ag * inv_g * inv_g;
          bar_G_vv[Size(a) * v + b] = bg_ss;
          if (compute_both_rhf) {
            os_gg += gab * gab * inv_g;
            bar_G_vv_os[Size(a) * v + b] = -2.0 * s_ord * gab * inv_g;
            bgap += s_ord * gab * gab * inv_g * inv_g;
          }
          bar_F[Size(o + a) * ext + (o + a)] += bgap;
          bar_F[Size(o + b) * ext + (o + b)] += bgap;
          bar_occ_sum -= bgap;
        }
      }
      e_ord_ss = -0.5 * ss_gg;
      if (compute_both_rhf) e_ord_os = -os_gg;
      bar_F[Size(i) * ext + i] += bar_occ_sum;
      bar_F[Size(j) * ext + j] += bar_occ_sum;
    }
    if (compute_both_rhf && !diag) {
      for (Size idx = 0; idx < bar_G_vv.size(); ++idx) bar_G_vv[idx] += 0.5 * bar_G_vv_os[idx];
      rm_gemm_host(false, true, aux, v, v, 1.0, raw_coul_j + o, ext, bar_G_vv.data(), v, 1.0,
                   bar_fit_i + o, ext);
      rm_gemm_host(false, false, aux, v, v, 1.0, fit_i + o, ext, bar_G_vv.data(), v, 1.0,
                   bar_raw_coul_j + o, ext);
      rm_gemm_host(false, false, aux, v, v, 0.5, raw_coul_i + o, ext, bar_G_vv_os.data(), v, 1.0,
                   bar_fit_j + o, ext);
      rm_gemm_host(false, true, aux, v, v, 0.5, fit_j + o, ext, bar_G_vv_os.data(), v, 1.0,
                   bar_raw_coul_i + o, ext);
    } else {
      rm_gemm_host(false, true, aux, v, v, spin_scale, raw_coul_j + o, ext, bar_G_vv.data(), v, 1.0,
                   bar_fit_i + o, ext);
      rm_gemm_host(false, false, aux, v, v, spin_scale, fit_i + o, ext, bar_G_vv.data(), v, 1.0,
                   bar_raw_coul_j + o, ext);
    }
    return 0;
  }

  // Forward pass (identical to eval_same_spin_or_rhf_pair_host).
  std::vector<double> f0(ee, 0.0);
  rm_gemm_host(true, false, ext, ext, aux, 1.0, fit_i, ext, res_g_j, ext, 0.0, f0.data(), ext);
  rm_gemm_host(true, false, ext, ext, aux, 1.0, raw_g_i, ext, fit_j, ext, 1.0, f0.data(), ext);

  std::vector<double> ft;
  if (!diag) {
    ft.resize(ee);
    for (int p = 0; p < ext; ++p) {
      for (int r = 0; r < ext; ++r) ft[Size(p) * ext + r] = f0[Size(r) * ext + p];
    }
  }

  std::vector<double> G_obs(Size(obs) * obs, 0.0);
  std::vector<double> G_oc(Size(o) * c, 0.0);
  std::vector<double> G_co(Size(c) * o, 0.0);
  rm_gemm_host(true, false, obs, obs, aux, 1.0, fit_i, ext, raw_coul_j, ext, 0.0, G_obs.data(),
               obs);
  rm_gemm_host(true, false, o, c, aux, 1.0, fit_i, ext, raw_coul_j + obs, ext, 0.0, G_oc.data(), c);
  rm_gemm_host(true, false, c, o, aux, 1.0, fit_i + obs, ext, raw_coul_j, ext, 0.0, G_co.data(), o);

  const double fg_ij = robust_elem_host(fit_i, raw_gc_i, fit_j, res_gc_j, aux, ext, i, j);
  const double f2_ij = robust_elem_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, i, j);
  const double uf_ij = robust_elem_host(fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, i, j);

  // Extract rows/columns of M_sq needed for both forward sq_fk and reverse bar_F/bar_K/bar_Sab.
  std::vector<double> m_row_i(ext), m_col_j(ext), m_row_j(ext), m_col_i(ext);
  robust_row_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, i, m_row_i.data());
  robust_col_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, j, m_col_j.data());
  if (!diag) {
    robust_row_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, j, m_row_j.data());
    robust_col_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, i, m_col_i.data());
  } else {
    m_row_j = m_row_i;
    m_col_i = m_col_j;
  }

  const double* hi = H + Size(i) * ext;
  const double* hj = H + Size(j) * ext;
  double sq_fk_d = 0.0, sq_fk_x = 0.0;
  for (int p = 0; p < ext; ++p) {
    sq_fk_d += m_row_i[p] * hj[p] + m_col_j[p] * hi[p];
    if (!diag) sq_fk_x += m_row_j[p] * hi[p] + m_col_i[p] * hj[p];
  }

  const double V_d = fg_ij - dot_rect_host(G_obs.data(), obs, f0.data(), ext, obs, obs) -
                     dot_rect_host(G_oc.data(), c, f0.data() + obs, ext, o, c) -
                     dot_rect_host(G_co.data(), o, f0.data() + Size(obs) * ext, ext, c, o);
  const double X_d = f2_ij - sqnorm_rect_host(f0.data(), ext, obs, obs) -
                     sqnorm_rect_host(f0.data() + obs, ext, o, c) -
                     sqnorm_rect_host(f0.data() + Size(obs) * ext, ext, c, o);

  BLeftScratchHost Lf, Lft;
  compute_b_left_host(f0.data(), F, K, o, obs, ext, Lf);
  double B_d = 0.0, V_x = 0.0, X_x = 0.0, B_x = 0.0;
  if (diag) {
    B_d = uf_ij + sq_fk_d + 2.0 * eval_b_overlap_host(Lf, f0.data(), o, obs, ext);
  } else {
    const double fg_ji = robust_elem_host(fit_i, raw_gc_i, fit_j, res_gc_j, aux, ext, j, i);
    const double f2_ji = robust_elem_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, j, i);
    const double uf_ji = robust_elem_host(fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, j, i);
    V_x = fg_ji - dot_rect_host(G_obs.data(), obs, ft.data(), ext, obs, obs) -
          dot_rect_host(G_oc.data(), c, ft.data() + obs, ext, o, c) -
          dot_rect_host(G_co.data(), o, ft.data() + Size(obs) * ext, ext, c, o);
    X_x = f2_ji - dot_rect_host(f0.data(), ext, ft.data(), ext, obs, obs) -
          dot_rect_host(f0.data() + obs, ext, ft.data() + obs, ext, o, c) -
          dot_rect_host(f0.data() + Size(obs) * ext, ext, ft.data() + Size(obs) * ext, ext, c, o);
    compute_b_left_host(ft.data(), F, K, o, obs, ext, Lft);
    B_d = uf_ij + sq_fk_d + eval_b_overlap_host(Lf, f0.data(), o, obs, ext) +
          eval_b_overlap_host(Lft, ft.data(), o, obs, ext);
    B_x = uf_ji + sq_fk_x + eval_b_overlap_host(Lf, ft.data(), o, obs, ext) +
          eval_b_overlap_host(Lft, f0.data(), o, obs, ext);
  }

  std::vector<double> C_d(Size(v) * v, 0.0);
  rm_gemm_host(false, true, v, v, c, 1.0, f0.data() + Size(o) * ext + obs, ext,
               F + Size(o) * ext + obs, ext, 0.0, C_d.data(), v);
  rm_gemm_host(false, false, v, v, c, 1.0, F + Size(o) * ext + obs, ext,
               f0.data() + Size(obs) * ext + o, ext, 1.0, C_d.data(), v);

  // Reverse Step 1: virtual denominator contraction & scalar weights.
  std::vector<double> bar_C_d(Size(v) * v, 0.0);
  std::vector<double> bar_C_os;
  if (compute_both_rhf) bar_C_os.assign(Size(v) * v, 0.0);
  std::vector<double> bar_G_obs(Size(obs) * obs, 0.0);
  std::vector<double> bar_G_oc(Size(o) * c, 0.0);
  std::vector<double> bar_G_co(Size(c) * o, 0.0);
  const bool split_rhf_offdiag = (compute_both_rhf && !diag);
  std::vector<double> bar_G_obs_os(split_rhf_offdiag ? Size(obs) * obs : 0, 0.0);
  std::vector<double> bar_G_oc_os(split_rhf_offdiag ? Size(o) * c : 0, 0.0);
  std::vector<double> bar_G_co_os(split_rhf_offdiag ? Size(c) * o : 0, 0.0);

  double bar_V_d = 0.0, bar_V_x = 0.0;
  double bar_X_d = 0.0, bar_X_x = 0.0;
  double bar_B_d_ss = 0.0, bar_B_d_os = 0.0, bar_B_x_os = 0.0;
  double bar_occ_sum = 0.0;

  if (diag) {
    // Closed-shell RHF diagonal AB(i, i) only.
    double vd_cg = 0.0, bd_cc = 0.0, os_gg = 0.0;
    bar_V_d = s_f12;
    bar_B_d_os = 0.25 * s_f12;
    bar_X_d = -0.25 * s_f12 * occ_sum;
    bar_occ_sum = -0.25 * s_f12 * X_d;
    for (int a = 0; a < v; ++a) {
      const double ea = F[Size(o + a) * ext + (o + a)];
      for (int b = 0; b < v; ++b) {
        const double gap = ea + F[Size(o + b) * ext + (o + b)] - occ_sum;
        if (!std::isfinite(gap) || gap == 0.0) return -4;
        const double inv_g = 1.0 / gap;
        const double gab = G_obs[Size(o + a) * obs + (o + b)];
        const double cab = C_d[Size(a) * v + b];
        vd_cg += cab * gab * inv_g;
        bd_cc += cab * cab * inv_g;
        if (include_ordinary) os_gg += gab * gab * inv_g;

        const double c_os = 0.5 * cab;
        const double bc_os = -2.0 * s_f12 * (gab + c_os) * inv_g;
        bar_C_os[Size(a) * v + b] = bc_os;
        bar_C_d[Size(a) * v + b] = 0.5 * bc_os;
        bar_G_obs[Size(o + a) * obs + (o + b)] = (-s_f12 * cab - 2.0 * s_ord * gab) * inv_g;
        const double bar_inv_g =
            -s_f12 * (cab * gab + 0.25 * cab * cab) - s_ord * gab * gab;
        const double bar_gap = -bar_inv_g * inv_g * inv_g;
        bar_F[Size(o + a) * ext + (o + a)] += spin_scale * bar_gap;
        bar_F[Size(o + b) * ext + (o + b)] += spin_scale * bar_gap;
        bar_occ_sum -= bar_gap;
      }
    }
    e_f12_os = (V_d - vd_cg) + 0.25 * (B_d - occ_sum * X_d - bd_cc);
    if (include_ordinary) e_ord_os = -os_gg;
    bar_F[Size(i) * ext + i] += spin_scale * 2.0 * bar_occ_sum;
  } else {
    double ss_gc = 0.0, ss_cc = 0.0, ss_gg = 0.0;
    double os_term = 0.0, os_gg = 0.0;
    bar_V_d = 0.5 * s_f12;
    bar_V_x = -0.5 * s_f12;
    bar_B_d_ss = (1.0 / 16.0) * s_f12;
    bar_X_d = -(1.0 / 16.0) * s_f12 * occ_sum;
    bar_X_x = (1.0 / 16.0) * s_f12 * occ_sum;
    bar_occ_sum = -(1.0 / 16.0) * s_f12 * (X_d - X_x);
    if (compute_both_rhf) {
      bar_V_d += 0.75 * s_f12;
      bar_V_x += 0.25 * s_f12;
      bar_B_d_os = (5.0 / 32.0) * s_f12;
      bar_B_x_os = (3.0 / 32.0) * s_f12;
      bar_X_d -= (5.0 / 32.0) * s_f12 * occ_sum;
      bar_X_x -= (3.0 / 32.0) * s_f12 * occ_sum;
      bar_occ_sum -= s_f12 * ((5.0 / 32.0) * X_d + (3.0 / 32.0) * X_x);
    }

    std::vector<double> bar_ag(Size(v) * v, 0.0);
    std::vector<double> bar_ac(Size(v) * v, 0.0);
    for (int a = 0; a < v; ++a) {
      const double ea = F[Size(o + a) * ext + (o + a)];
      for (int b = 0; b < v; ++b) {
        const double gap = ea + F[Size(o + b) * ext + (o + b)] - occ_sum;
        if (!std::isfinite(gap) || gap == 0.0) return -4;
        const double inv_g = 1.0 / gap;
        const double gab = G_obs[Size(o + a) * obs + (o + b)];
        const double gba = G_obs[Size(o + b) * obs + (o + a)];
        const double cab = C_d[Size(a) * v + b];
        const double cba = C_d[Size(b) * v + a];
        const double ag = gab - gba;
        const double ac = cab - cba;
        ss_gc += ag * ac * inv_g;
        ss_cc += ac * ac * inv_g;
        if (include_ordinary) ss_gg += ag * ag * inv_g;

        bar_ag[Size(a) * v + b] = (-0.25 * s_f12 * ac - s_ord * ag) * inv_g;
        bar_ac[Size(a) * v + b] = (-0.25 * s_f12 * ag - (1.0 / 16.0) * s_f12 * ac) * inv_g;
        double bar_inv_g =
            -0.25 * s_f12 * ag * ac - (1.0 / 32.0) * s_f12 * ac * ac - 0.5 * s_ord * ag * ag;

        if (compute_both_rhf) {
          const double c_os = 0.375 * cab + 0.125 * cba;
          os_term += c_os * (2.0 * gab + c_os) * inv_g;
          if (include_ordinary) os_gg += gab * gab * inv_g;

          bar_C_os[Size(a) * v + b] = -2.0 * s_f12 * (gab + c_os) * inv_g;
          bar_G_obs_os[Size(o + a) * obs + (o + b)] +=
              (-2.0 * s_f12 * c_os - 2.0 * s_ord * gab) * inv_g;
          bar_inv_g += -s_f12 * c_os * (2.0 * gab + c_os) - s_ord * gab * gab;
        }
        const double bar_gap = -bar_inv_g * inv_g * inv_g;
        bar_F[Size(o + a) * ext + (o + a)] += bar_gap;
        bar_F[Size(o + b) * ext + (o + b)] += bar_gap;
        bar_occ_sum -= bar_gap;
      }
    }
    for (int a = 0; a < v; ++a) {
      for (int b = 0; b < v; ++b) {
        bar_G_obs[Size(o + a) * obs + (o + b)] +=
            bar_ag[Size(a) * v + b] - bar_ag[Size(b) * v + a];
        double bc = bar_ac[Size(a) * v + b] - bar_ac[Size(b) * v + a];
        if (compute_both_rhf) {
          bc += 0.375 * bar_C_os[Size(a) * v + b] + 0.125 * bar_C_os[Size(b) * v + a];
        }
        bar_C_d[Size(a) * v + b] = bc;
      }
    }
    bar_F[Size(i) * ext + i] += bar_occ_sum;
    bar_F[Size(j) * ext + j] += bar_occ_sum;

    const double lin_ss = (V_d - V_x) - 0.5 * ss_gc;
    const double quad_ss = ((B_d - B_x) - occ_sum * (X_d - X_x)) - 0.5 * ss_cc;
    e_f12_ss = 0.5 * lin_ss + (1.0 / 16.0) * quad_ss;
    if (include_ordinary) e_ord_ss = -0.5 * ss_gg;
    if (compute_both_rhf) {
      const double lin_os = 0.375 * V_d + 0.125 * V_x;
      const double norm_os = (5.0 / 32.0) * X_d + (3.0 / 32.0) * X_x;
      const double b_os = (5.0 / 32.0) * B_d + (3.0 / 32.0) * B_x;
      e_f12_os = 2.0 * lin_os + b_os - occ_sum * norm_os - os_term;
      if (include_ordinary) e_ord_os = -os_gg;
    }
  }

  // Reverse Step 2: Coupling C_d pullback -> bar_f0, bar_F[o:obs, obs:].
  std::vector<double> bar_f0(ee, 0.0);
  std::vector<double> bar_ft(ee, 0.0);
  if (!compute_both_rhf) {
    rm_gemm_host(false, false, v, c, v, 1.0, bar_C_d.data(), v, F + Size(o) * ext + obs, ext, 1.0,
                 bar_f0.data() + Size(o) * ext + obs, ext);
    rm_gemm_host(true, false, c, v, v, 1.0, F + Size(o) * ext + obs, ext, bar_C_d.data(), v, 1.0,
                 bar_f0.data() + Size(obs) * ext + o, ext);
    rm_gemm_host(true, false, v, c, v, 1.0, bar_C_d.data(), v, f0.data() + Size(o) * ext + obs,
                 ext, 1.0, bar_F + Size(o) * ext + obs, ext);
    rm_gemm_host(false, true, v, c, v, 1.0, bar_C_d.data(), v, f0.data() + Size(obs) * ext + o,
                 ext, 1.0, bar_F + Size(o) * ext + obs, ext);
  } else {
    // In RHF spatial, bar_C_d already sums SS + OS for bar_F[o:obs, obs:] (with spin_scale for diag).
    // For bar_f0: SS uses bar_C_d_ss into bar_f0, while OS accumulates into bar_f_os (also needed for bar_Sab).
    rm_gemm_host(true, false, v, c, v, spin_scale, bar_C_d.data(), v,
                 f0.data() + Size(o) * ext + obs, ext, 1.0, bar_F + Size(o) * ext + obs, ext);
    rm_gemm_host(false, true, v, c, v, spin_scale, bar_C_d.data(), v,
                 f0.data() + Size(obs) * ext + o, ext, 1.0, bar_F + Size(o) * ext + obs, ext);
    if (!diag) {
      // Same-spin only portion of bar_C_d for bar_f0; OS portion is handled cleanly via bar_f_os below.
      std::vector<double> bar_C_ss(Size(v) * v);
      for (int a = 0; a < v; ++a) {
        for (int b = 0; b < v; ++b) {
          bar_C_ss[Size(a) * v + b] =
              bar_C_d[Size(a) * v + b] -
              (0.375 * bar_C_os[Size(a) * v + b] + 0.125 * bar_C_os[Size(b) * v + a]);
        }
      }
      rm_gemm_host(false, false, v, c, v, 1.0, bar_C_ss.data(), v, F + Size(o) * ext + obs, ext,
                   1.0, bar_f0.data() + Size(o) * ext + obs, ext);
      rm_gemm_host(true, false, c, v, v, 1.0, F + Size(o) * ext + obs, ext, bar_C_ss.data(), v, 1.0,
                   bar_f0.data() + Size(obs) * ext + o, ext);
    }
  }

  // Reverse Step 3a: Same-spin B_overlap pullback (for i < j):
  // B_d - B_x overlap part equals B_overlap(f0 - ft, f0 - ft).
  if (!diag && bar_B_d_ss != 0.0) {
    std::vector<double> diff_f(ee);
    for (Size idx = 0; idx < ee; ++idx) diff_f[idx] = f0[idx] - ft[idx];
    std::vector<double> bar_diff(ee, 0.0);
    b_overlap_pullback_host(bar_B_d_ss, diff_f.data(), diff_f.data(), F, K, o, obs, ext,
                            bar_diff.data(), bar_diff.data(), bar_F, bar_K);
    for (Size idx = 0; idx < ee; ++idx) {
      bar_f0[idx] += bar_diff[idx];
      bar_ft[idx] -= bar_diff[idx];
    }
  }

  // Reverse Step 3b: Closed-shell RHF opposite-spin pullback (exact f_os, bar_f_os, bar_F, bar_K, bar_Sab).
  if (compute_both_rhf) {
    std::vector<double> f_os(ee);
    if (diag) {
      for (Size idx = 0; idx < ee; ++idx) f_os[idx] = 0.5 * f0[idx];
    } else {
      for (Size idx = 0; idx < ee; ++idx) f_os[idx] = 0.375 * f0[idx] + 0.125 * ft[idx];
    }
    std::vector<double> qf_os = f_os;
    zero_p_blocks_host(qf_os.data(), o, o, obs, ext);

    // bar_F, bar_K from -s_f12 (<f_os, H f_os + f_os H^T>) + s_f12 (<qf_os, F qf_os + qf_os F^T>).
    // Note for RHF spatial, alpha receives spin_scale * (bar_Fa + bar_Fb).
    std::vector<double> ff_mat(ee, 0.0);
    rm_gemm_host(false, true, ext, ext, ext, 1.0, f_os.data(), ext, f_os.data(), ext, 0.0,
                 ff_mat.data(), ext);
    rm_gemm_host(true, false, ext, ext, ext, 1.0, f_os.data(), ext, f_os.data(), ext, 1.0,
                 ff_mat.data(), ext);
    for (Size idx = 0; idx < ee; ++idx) {
      const double v_sub = spin_scale * s_f12 * ff_mat[idx];
      bar_K[idx] -= v_sub;
      bar_F[idx] -= v_sub;
    }
    rm_gemm_host(false, true, ext, ext, ext, spin_scale * s_f12, qf_os.data(), ext, qf_os.data(),
                 ext, 1.0, bar_F, ext);
    rm_gemm_host(true, false, ext, ext, ext, spin_scale * s_f12, qf_os.data(), ext, qf_os.data(),
                 ext, 1.0, bar_F, ext);

    // Build total bar_f_os for AB(i, j) (for both W(bar_f_os)->bar_f0 and bar_Sab!).
    std::vector<double> bar_f_os(ee, 0.0);
    // From coupling C_os:
    rm_gemm_host(false, false, v, c, v, 1.0, bar_C_os.data(), v, F + Size(o) * ext + obs, ext, 1.0,
                 bar_f_os.data() + Size(o) * ext + obs, ext);
    rm_gemm_host(true, false, c, v, v, 1.0, F + Size(o) * ext + obs, ext, bar_C_os.data(), v, 1.0,
                 bar_f_os.data() + Size(obs) * ext + o, ext);
    // From -s_f12 (<f_os, H f_os + f_os H^T>) (with H = H^T in RHF):
    rm_gemm_host(false, false, ext, ext, ext, -2.0 * s_f12, H, ext, f_os.data(), ext, 1.0,
                 bar_f_os.data(), ext);
    rm_gemm_host(false, false, ext, ext, ext, -2.0 * s_f12, f_os.data(), ext, H, ext, 1.0,
                 bar_f_os.data(), ext);
    // From +s_f12 (<qf_os, F qf_os + qf_os F^T>) (with F = F^T in RHF):
    std::vector<double> bar_qf_os(ee, 0.0);
    rm_gemm_host(false, false, ext, ext, ext, 2.0 * s_f12, F, ext, qf_os.data(), ext, 0.0,
                 bar_qf_os.data(), ext);
    rm_gemm_host(false, false, ext, ext, ext, 2.0 * s_f12, qf_os.data(), ext, F, ext, 1.0,
                 bar_qf_os.data(), ext);
    zero_p_blocks_host(bar_qf_os.data(), o, o, obs, ext);
    for (Size idx = 0; idx < ee; ++idx) bar_f_os[idx] += bar_qf_os[idx];
    // From linear (-2 * s_f12 * <G, pf_os>) and norm (+2 * s_f12 * occ_sum * <f_os, pf_os>):
    const double n_coef = 2.0 * s_f12 * occ_sum;
    for (int p = 0; p < obs; ++p) {
      for (int r = 0; r < obs; ++r) {
        bar_f_os[Size(p) * ext + r] +=
            -2.0 * s_f12 * G_obs[Size(p) * obs + r] + n_coef * f_os[Size(p) * ext + r];
      }
    }
    for (int p = 0; p < o; ++p) {
      for (int q = 0; q < c; ++q) {
        bar_f_os[Size(p) * ext + (obs + q)] +=
            -2.0 * s_f12 * G_oc[Size(p) * c + q] + n_coef * f_os[Size(p) * ext + (obs + q)];
      }
    }
    for (int p = 0; p < c; ++p) {
      for (int q = 0; q < o; ++q) {
        bar_f_os[Size(obs + p) * ext + q] +=
            -2.0 * s_f12 * G_co[Size(p) * o + q] + n_coef * f_os[Size(obs + p) * ext + q];
      }
    }

    // Keep bar_f0_os = W(bar_f_os) separate for bar_res_os_g_j (symmetrized AB metric pullback).
    std::vector<double> bar_f0_os(ee, 0.0);
    for (int p = 0; p < ext; ++p) {
      for (int r = 0; r < ext; ++r) {
        const double v_os =
            0.375 * bar_f_os[Size(p) * ext + r] + 0.125 * bar_f_os[Size(r) * ext + p];
        bar_f0_os[Size(p) * ext + r] = v_os;
      }
    }
    rm_gemm_host(false, true, aux, ext, ext, spin_scale, res_g_j, ext, bar_f0_os.data(), ext, 1.0,
                 bar_fit_i, ext);
    rm_gemm_host(false, false, aux, ext, ext, spin_scale, fit_i, ext, bar_f0_os.data(), ext, 1.0,
                 bar_res_os_g_j, ext);
    rm_gemm_host(false, true, aux, ext, ext, spin_scale, fit_j, ext, bar_f0_os.data(), ext, 1.0,
                 bar_raw_g_i, ext);
    rm_gemm_host(false, false, aux, ext, ext, spin_scale, raw_g_i, ext, bar_f0_os.data(), ext, 1.0,
                 bar_fit_j, ext);

    // Exact bar_cross_overlap at Sab = I for AB(i, j) (and mirrored transpose for AB(j, i) when i < j).
    if (bar_cross_overlap) {
      std::vector<double> bar_Sab_ij(ee, 0.0);
      // 1. From f = W(f0): (1/8) (bar_f_os @ f0 + f0 @ bar_f_os)
      rm_gemm_host(false, false, ext, ext, ext, 0.125, bar_f_os.data(), ext, f0.data(), ext, 1.0,
                   bar_Sab_ij.data(), ext);
      rm_gemm_host(false, false, ext, ext, ext, 0.125, f0.data(), ext, bar_f_os.data(), ext, 1.0,
                   bar_Sab_ij.data(), ext);

      std::vector<double> bw_v(ext, 0.0), bwT_u(ext, 0.0);
      // 2. From gc0: bar_w += 2 * s_f12 * gc0
      std::vector<double> gc_col_i(ext), gc_row_j(ext);
      robust_col_host(fit_i, raw_gc_i, fit_j, res_gc_j, aux, ext, i, gc_col_i.data());
      robust_row_host(fit_i, raw_gc_i, fit_j, res_gc_j, aux, ext, j, gc_row_j.data());
      for (int p = 0; p < ext; ++p) {
        bw_v[p] += 2.0 * s_f12 * gc_col_i[p];
        bwT_u[p] += 2.0 * s_f12 * gc_row_j[p];
      }

      // 3. From uf = W(uf0)
      std::vector<double> u_col_i(ext), u_row_j(ext), u_col_j(ext), u_row_i(ext);
      robust_col_host(fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, i, u_col_i.data());
      robust_row_host(fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, j, u_row_j.data());
      if (!diag) {
        robust_col_host(fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, j, u_col_j.data());
        robust_row_host(fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, i, u_row_i.data());
      } else {
        u_col_j = u_col_i;
        u_row_i = u_row_j;
      }
      const double c3_64 = (3.0 / 64.0) * s_f12;
      const double c1_64 = (1.0 / 64.0) * s_f12;
      for (int p = 0; p < ext; ++p) {
        bar_Sab_ij[Size(i) * ext + p] += c3_64 * u_row_j[p];
        bar_Sab_ij[Size(p) * ext + j] += c3_64 * u_col_i[p];
        bar_Sab_ij[Size(j) * ext + p] += c1_64 * u_row_i[p];
        bar_Sab_ij[Size(p) * ext + i] += c1_64 * u_col_j[p];
        bw_v[p] += s_f12 * (0.375 * u_col_i[p] + 0.125 * u_row_i[p]);
        bwT_u[p] += s_f12 * (0.375 * u_row_j[p] + 0.125 * u_col_j[p]);
      }

      // 4. From ff = W(ff0)
      std::vector<double> ff0_hi(ext), ff0T_hi(ext), ff0_hj(ext), ff0T_hj(ext);
      robust_rmatvec_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, hi, ff0_hi.data());
      robust_lmatvec_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, hi, ff0T_hi.data());
      if (!diag) {
        robust_rmatvec_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, hj, ff0_hj.data());
        robust_lmatvec_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, hj, ff0T_hj.data());
      } else {
        ff0_hj = ff0_hi;
        ff0T_hj = ff0T_hi;
      }
      std::vector<double> x1(ext), x3(ext), y2(ext), y4(ext), ff0_x1(ext), ff0_x3(ext);
      for (int p = 0; p < ext; ++p) {
        x1[p] = s_f12 * (hi[p] - (p == i ? occ_sum : 0.0));
        x3[p] = s_f12 * (hj[p] - (p == j ? occ_sum : 0.0));
        y2[p] = s_f12 * hj[p];
        y4[p] = s_f12 * hi[p];
        ff0_x1[p] = s_f12 * (ff0_hi[p] - occ_sum * m_col_i[p]);
        ff0_x3[p] = s_f12 * (ff0_hj[p] - occ_sum * m_col_j[p]);
      }
      // (3/64) * (x1 m_row_j^T + (ff0 x1) e_j^T + e_i (ff0^T y2)^T + m_col_i y2^T)
      add_outer_host(3.0 / 64.0, x1.data(), m_row_j.data(), ext, bar_Sab_ij.data());
      add_outer_host(3.0 / 64.0, m_col_i.data(), y2.data(), ext, bar_Sab_ij.data());
      // (1/64) * (x3 m_row_i^T + (ff0 x3) e_i^T + e_j (ff0^T y4)^T + m_col_j y4^T)
      add_outer_host(1.0 / 64.0, x3.data(), m_row_i.data(), ext, bar_Sab_ij.data());
      add_outer_host(1.0 / 64.0, m_col_j.data(), y4.data(), ext, bar_Sab_ij.data());
      for (int p = 0; p < ext; ++p) {
        bar_Sab_ij[Size(p) * ext + j] += (3.0 / 64.0) * ff0_x1[p];
        bar_Sab_ij[Size(i) * ext + p] += (3.0 / 64.0) * s_f12 * ff0T_hj[p];
        bar_Sab_ij[Size(p) * ext + i] += (1.0 / 64.0) * ff0_x3[p];
        bar_Sab_ij[Size(j) * ext + p] += (1.0 / 64.0) * s_f12 * ff0T_hi[p];
      }

      // bar_w += -s_f12 * occ_sum * ff + s_f12 * (H @ ff + ff @ H)
      std::vector<double> ff_ei(ext), ffT_ej(ext), h_ff_ei(ext), h_ffT_ej(ext);
      for (int p = 0; p < ext; ++p) {
        ff_ei[p] = 0.375 * m_col_i[p] + 0.125 * m_row_i[p];
        ffT_ej[p] = 0.375 * m_row_j[p] + 0.125 * m_col_j[p];
      }
      matvec_host(false, H, ext, ff_ei.data(), h_ff_ei.data());
      matvec_host(false, H, ext, ffT_ej.data(), h_ffT_ej.data());
      for (int p = 0; p < ext; ++p) {
        const double ff_hi_p = 0.375 * ff0_hi[p] + 0.125 * ff0T_hi[p];
        const double ffT_hj_p = 0.375 * ff0T_hj[p] + 0.125 * ff0_hj[p];
        bw_v[p] += s_f12 * (-occ_sum * ff_ei[p] + h_ff_ei[p] + ff_hi_p);
        bwT_u[p] += s_f12 * (-occ_sum * ffT_ej[p] + ffT_hj_p + h_ffT_ej[p]);
      }

      // Fold bw_v and bwT_u into bar_Sab_ij
      for (int p = 0; p < ext; ++p) {
        bar_Sab_ij[Size(p) * ext + j] += 0.125 * bw_v[p];
        bar_Sab_ij[Size(i) * ext + p] += 0.125 * bwT_u[p];
      }
      if (diag) {
        for (Size idx = 0; idx < ee; ++idx) bar_cross_overlap[idx] += bar_Sab_ij[idx];
      } else {
        for (int p = 0; p < ext; ++p) {
          for (int r = 0; r < ext; ++r) {
            bar_cross_overlap[Size(p) * ext + r] +=
                bar_Sab_ij[Size(p) * ext + r] + bar_Sab_ij[Size(r) * ext + p];
          }
        }
      }
    }
  }

  // Reverse Step 4: V_d/V_x/X_d/X_x -> bar_G_obs/oc/co and SS part of bar_f0/bar_ft.
  // (Note: for compute_both_rhf, OS part of bar_f0 was already pulled back via bar_f0_os above.)
  const double bar_V_d_ss = (!diag) ? (0.5 * s_f12) : 0.0;
  const double bar_V_x_ss = (!diag) ? (-0.5 * s_f12) : 0.0;
  const double bar_V_d_os = compute_both_rhf ? (diag ? s_f12 : (0.75 * s_f12)) : 0.0;
  const double bar_V_x_os = split_rhf_offdiag ? (0.25 * s_f12) : 0.0;
  const double bar_X_d_ss = (!diag) ? (-(1.0 / 16.0) * s_f12 * occ_sum) : 0.0;
  const double bar_X_x_ss = (!diag) ? ((1.0 / 16.0) * s_f12 * occ_sum) : 0.0;
  const double bar_X_d_os =
      compute_both_rhf ? (-(diag ? 0.25 : (5.0 / 32.0)) * s_f12 * occ_sum) : 0.0;
  const double bar_X_x_os = split_rhf_offdiag ? (-(3.0 / 32.0) * s_f12 * occ_sum) : 0.0;

  for (int p = 0; p < obs; ++p) {
    for (int r = 0; r < obs; ++r) {
      const double fp = f0[Size(p) * ext + r];
      const double ftp = diag ? fp : ft[Size(p) * ext + r];
      const double gpr = G_obs[Size(p) * obs + r];
      if (split_rhf_offdiag) {
        bar_G_obs[Size(p) * obs + r] -= bar_V_d_ss * fp + bar_V_x_ss * ftp;
        bar_G_obs_os[Size(p) * obs + r] -= bar_V_d_os * fp + bar_V_x_os * ftp;
      } else {
        bar_G_obs[Size(p) * obs + r] -= bar_V_d * fp + bar_V_x * ftp;
      }
      if (!diag) {
        bar_f0[Size(p) * ext + r] -= bar_V_d_ss * gpr + 2.0 * bar_X_d_ss * fp + bar_X_x_ss * ftp;
        bar_ft[Size(p) * ext + r] -= bar_V_x_ss * gpr + bar_X_x_ss * fp;
      }
    }
  }
  for (int p = 0; p < o; ++p) {
    for (int q = 0; q < c; ++q) {
      const double fp = f0[Size(p) * ext + (obs + q)];
      const double ftp = diag ? fp : ft[Size(p) * ext + (obs + q)];
      const double gpq = G_oc[Size(p) * c + q];
      if (split_rhf_offdiag) {
        bar_G_oc[Size(p) * c + q] -= bar_V_d_ss * fp + bar_V_x_ss * ftp;
        bar_G_oc_os[Size(p) * c + q] -= bar_V_d_os * fp + bar_V_x_os * ftp;
      } else {
        bar_G_oc[Size(p) * c + q] -= bar_V_d * fp + bar_V_x * ftp;
      }
      if (!diag) {
        bar_f0[Size(p) * ext + (obs + q)] -=
            bar_V_d_ss * gpq + 2.0 * bar_X_d_ss * fp + bar_X_x_ss * ftp;
        bar_ft[Size(p) * ext + (obs + q)] -= bar_V_x_ss * gpq + bar_X_x_ss * fp;
      }
    }
  }
  for (int p = 0; p < c; ++p) {
    for (int q = 0; q < o; ++q) {
      const double fp = f0[Size(obs + p) * ext + q];
      const double ftp = diag ? fp : ft[Size(obs + p) * ext + q];
      const double gpq = G_co[Size(p) * o + q];
      if (split_rhf_offdiag) {
        bar_G_co[Size(p) * o + q] -= bar_V_d_ss * fp + bar_V_x_ss * ftp;
        bar_G_co_os[Size(p) * o + q] -= bar_V_d_os * fp + bar_V_x_os * ftp;
      } else {
        bar_G_co[Size(p) * o + q] -= bar_V_d * fp + bar_V_x * ftp;
      }
      if (!diag) {
        bar_f0[Size(obs + p) * ext + q] -=
            bar_V_d_ss * gpq + 2.0 * bar_X_d_ss * fp + bar_X_x_ss * ftp;
        bar_ft[Size(obs + p) * ext + q] -= bar_V_x_ss * gpq + bar_X_x_ss * fp;
      }
    }
  }
  if (!diag) {
    for (int p = 0; p < ext; ++p) {
      for (int r = 0; r < ext; ++r) bar_f0[Size(p) * ext + r] += bar_ft[Size(r) * ext + p];
    }
  }

  // Reverse Step 5: Pullback of SS f0, G_obs/G_oc/G_co, and robust-DF scalars into fit/raw/res.
  if (!diag) {
    rm_gemm_host(false, true, aux, ext, ext, 1.0, res_g_j, ext, bar_f0.data(), ext, 1.0, bar_fit_i,
                 ext);
    rm_gemm_host(false, false, aux, ext, ext, 1.0, fit_i, ext, bar_f0.data(), ext, 1.0, bar_res_g_j,
                 ext);
    rm_gemm_host(false, true, aux, ext, ext, 1.0, fit_j, ext, bar_f0.data(), ext, 1.0, bar_raw_g_i,
                 ext);
    rm_gemm_host(false, false, aux, ext, ext, 1.0, raw_g_i, ext, bar_f0.data(), ext, 1.0, bar_fit_j,
                 ext);
  }

  if (split_rhf_offdiag) {
    for (Size idx = 0; idx < bar_G_obs.size(); ++idx) bar_G_obs[idx] += 0.5 * bar_G_obs_os[idx];
    for (Size idx = 0; idx < bar_G_oc.size(); ++idx) bar_G_oc[idx] += 0.5 * bar_G_oc_os[idx];
    for (Size idx = 0; idx < bar_G_co.size(); ++idx) bar_G_co[idx] += 0.5 * bar_G_co_os[idx];
    rm_gemm_host(false, true, aux, obs, obs, 1.0, raw_coul_j, ext, bar_G_obs.data(), obs, 1.0,
                 bar_fit_i, ext);
    rm_gemm_host(false, false, aux, obs, obs, 1.0, fit_i, ext, bar_G_obs.data(), obs, 1.0,
                 bar_raw_coul_j, ext);
    rm_gemm_host(false, true, aux, o, c, 1.0, raw_coul_j + obs, ext, bar_G_oc.data(), c, 1.0,
                 bar_fit_i, ext);
    rm_gemm_host(false, false, aux, c, o, 1.0, fit_i, ext, bar_G_oc.data(), c, 1.0,
                 bar_raw_coul_j + obs, ext);
    rm_gemm_host(false, true, aux, c, o, 1.0, raw_coul_j, ext, bar_G_co.data(), o, 1.0,
                 bar_fit_i + obs, ext);
    rm_gemm_host(false, false, aux, o, c, 1.0, fit_i + obs, ext, bar_G_co.data(), o, 1.0,
                 bar_raw_coul_j, ext);
    // Symmetric AB(j, i) half of Coulomb into (fit_j, raw_coul_i):
    rm_gemm_host(false, false, aux, obs, obs, 0.5, raw_coul_i, ext, bar_G_obs_os.data(), obs, 1.0,
                 bar_fit_j, ext);
    rm_gemm_host(false, true, aux, obs, obs, 0.5, fit_j, ext, bar_G_obs_os.data(), obs, 1.0,
                 bar_raw_coul_i, ext);
    rm_gemm_host(false, false, aux, o, c, 0.5, raw_coul_i + obs, ext, bar_G_co_os.data(), o, 1.0,
                 bar_fit_j, ext);
    rm_gemm_host(false, true, aux, c, o, 0.5, fit_j, ext, bar_G_co_os.data(), o, 1.0,
                 bar_raw_coul_i + obs, ext);
    rm_gemm_host(false, false, aux, c, o, 0.5, raw_coul_i, ext, bar_G_oc_os.data(), c, 1.0,
                 bar_fit_j + obs, ext);
    rm_gemm_host(false, true, aux, o, c, 0.5, fit_j + obs, ext, bar_G_oc_os.data(), c, 1.0,
                 bar_raw_coul_i, ext);
  } else {
    rm_gemm_host(false, true, aux, obs, obs, spin_scale, raw_coul_j, ext, bar_G_obs.data(), obs,
                 1.0, bar_fit_i, ext);
    rm_gemm_host(false, false, aux, obs, obs, spin_scale, fit_i, ext, bar_G_obs.data(), obs, 1.0,
                 bar_raw_coul_j, ext);
    rm_gemm_host(false, true, aux, o, c, spin_scale, raw_coul_j + obs, ext, bar_G_oc.data(), c, 1.0,
                 bar_fit_i, ext);
    rm_gemm_host(false, false, aux, c, o, spin_scale, fit_i, ext, bar_G_oc.data(), c, 1.0,
                 bar_raw_coul_j + obs, ext);
    rm_gemm_host(false, true, aux, c, o, spin_scale, raw_coul_j, ext, bar_G_co.data(), o, 1.0,
                 bar_fit_i + obs, ext);
    rm_gemm_host(false, false, aux, o, c, spin_scale, fit_i + obs, ext, bar_G_co.data(), o, 1.0,
                 bar_raw_coul_j, ext);
  }

  // gc, dc, sq scalar elements & sq_fk: split SS (-> bar_res_*) and OS (-> bar_res_os_*).
  if (!diag) {
    robust_elem_pullback_host(bar_V_d_ss, i, j, aux, ext, fit_i, raw_gc_i, fit_j, res_gc_j,
                              bar_fit_i, bar_raw_gc_i, bar_fit_j, bar_res_gc_j);
    robust_elem_pullback_host(bar_V_x_ss, j, i, aux, ext, fit_i, raw_gc_i, fit_j, res_gc_j,
                              bar_fit_i, bar_raw_gc_i, bar_fit_j, bar_res_gc_j);
    robust_elem_pullback_host(bar_B_d_ss, i, j, aux, ext, fit_i, raw_dc_i, fit_j, res_dc_j,
                              bar_fit_i, bar_raw_dc_i, bar_fit_j, bar_res_dc_j);
    robust_elem_pullback_host(-bar_B_d_ss, j, i, aux, ext, fit_i, raw_dc_i, fit_j, res_dc_j,
                              bar_fit_i, bar_raw_dc_i, bar_fit_j, bar_res_dc_j);
    robust_elem_pullback_host(bar_X_d_ss, i, j, aux, ext, fit_i, raw_sq_i, fit_j, res_sq_j,
                              bar_fit_i, bar_raw_sq_i, bar_fit_j, bar_res_sq_j);
    robust_elem_pullback_host(bar_X_x_ss, j, i, aux, ext, fit_i, raw_sq_i, fit_j, res_sq_j,
                              bar_fit_i, bar_raw_sq_i, bar_fit_j, bar_res_sq_j);
    robust_row_col_pullback_host(bar_B_d_ss, i, hj, bar_B_d_ss, j, hi, aux, ext, fit_i, raw_sq_i,
                                 fit_j, res_sq_j, bar_fit_i, bar_raw_sq_i, bar_fit_j, bar_res_sq_j);
    robust_row_col_pullback_host(-bar_B_d_ss, j, hi, -bar_B_d_ss, i, hj, aux, ext, fit_i, raw_sq_i,
                                 fit_j, res_sq_j, bar_fit_i, bar_raw_sq_i, bar_fit_j, bar_res_sq_j);
  }
  if (compute_both_rhf) {
    robust_elem_pullback_host(spin_scale * bar_V_d_os, i, j, aux, ext, fit_i, raw_gc_i, fit_j,
                              res_gc_j, bar_fit_i, bar_raw_gc_i, bar_fit_j, bar_res_os_gc_j);
    robust_elem_pullback_host(spin_scale * bar_B_d_os, i, j, aux, ext, fit_i, raw_dc_i, fit_j,
                              res_dc_j, bar_fit_i, bar_raw_dc_i, bar_fit_j, bar_res_os_dc_j);
    robust_elem_pullback_host(spin_scale * bar_X_d_os, i, j, aux, ext, fit_i, raw_sq_i, fit_j,
                              res_sq_j, bar_fit_i, bar_raw_sq_i, bar_fit_j, bar_res_os_sq_j);
    robust_row_col_pullback_host(spin_scale * bar_B_d_os, i, hj, spin_scale * bar_B_d_os, j, hi,
                                 aux, ext, fit_i, raw_sq_i, fit_j, res_sq_j, bar_fit_i,
                                 bar_raw_sq_i, bar_fit_j, bar_res_os_sq_j);
    if (!diag) {
      robust_elem_pullback_host(bar_V_x_os, j, i, aux, ext, fit_i, raw_gc_i, fit_j, res_gc_j,
                                bar_fit_i, bar_raw_gc_i, bar_fit_j, bar_res_os_gc_j);
      robust_elem_pullback_host(bar_B_x_os, j, i, aux, ext, fit_i, raw_dc_i, fit_j, res_dc_j,
                                bar_fit_i, bar_raw_dc_i, bar_fit_j, bar_res_os_dc_j);
      robust_elem_pullback_host(bar_X_x_os, j, i, aux, ext, fit_i, raw_sq_i, fit_j, res_sq_j,
                                bar_fit_i, bar_raw_sq_i, bar_fit_j, bar_res_os_sq_j);
      robust_row_col_pullback_host(bar_B_x_os, j, hi, bar_B_x_os, i, hj, aux, ext, fit_i, raw_sq_i,
                                   fit_j, res_sq_j, bar_fit_i, bar_raw_sq_i, bar_fit_j,
                                   bar_res_os_sq_j);
    }
  }
  // Exact row vs column orientation for bar_F, bar_K:
  // - Same-spin (AA) writes to ROWS i, j
  // - Opposite-spin (AB) writes to COLUMNS i, j
  if (!diag && bar_B_d_ss != 0.0) {
    for (int p = 0; p < ext; ++p) {
      const double ri_val = bar_B_d_ss * (m_col_j[p] - m_row_j[p]);
      const double rj_val = bar_B_d_ss * (m_row_i[p] - m_col_i[p]);
      bar_F[Size(i) * ext + p] += ri_val;
      bar_K[Size(i) * ext + p] += ri_val;
      bar_F[Size(j) * ext + p] += rj_val;
      bar_K[Size(j) * ext + p] += rj_val;
    }
  }
  if (compute_both_rhf) {
    if (diag) {
      // AB(i, i): 0.5 * (0.25 * m_col_i + 0.25 * m_row_i) into column i
      for (int p = 0; p < ext; ++p) {
        const double ci_val = 0.125 * s_f12 * (m_col_i[p] + m_row_i[p]);
        bar_F[Size(p) * ext + i] += ci_val;
        bar_K[Size(p) * ext + i] += ci_val;
      }
    } else {
      for (int p = 0; p < ext; ++p) {
        const double ci_val = bar_B_d_os * m_col_j[p] + bar_B_x_os * m_row_j[p];
        const double cj_val = bar_B_d_os * m_row_i[p] + bar_B_x_os * m_col_i[p];
        bar_F[Size(p) * ext + i] += ci_val;
        bar_K[Size(p) * ext + i] += ci_val;
        bar_F[Size(p) * ext + j] += cj_val;
        bar_K[Size(p) * ext + j] += cj_val;
      }
    }
  }
  return 0;
}

int eval_uhf_opp_spin_channel_response_host(
    bool include_f12, bool include_ordinary, bool has_occ_weights, int oa, int ob, int fa, int fb,
    int obs, int ext, int aux, const double* occ_w_a, const double* occ_w_b,
    const std::vector<double>& fit_a_oqe, const std::vector<double>& fit_b_oqe,
    const std::vector<double>& raw_coul_b_oqe, const std::vector<double>& raw_g_a_oqe,
    const std::vector<double>& res_g_b_oqe, const std::vector<double>& raw_sq_a_oqe,
    const std::vector<double>& res_sq_b_oqe, const std::vector<double>& raw_gc_a_oqe,
    const std::vector<double>& res_gc_b_oqe, const std::vector<double>& raw_dc_a_oqe,
    const std::vector<double>& res_dc_b_oqe, const double* Fa, const double* Fb, const double* Ka,
    const double* Kb, const double* Sab, double& e_f12_tot, double& e_ord_tot, double* pair_f12_ab,
    double* pair_ord_ab, std::vector<double>& bar_fit_a_oqe, std::vector<double>& bar_fit_b_oqe,
    std::vector<double>& bar_raw_coul_b_oqe, std::vector<double>& bar_raw_g_a_oqe,
    std::vector<double>& bar_res_g_b_oqe, std::vector<double>& bar_raw_sq_a_oqe,
    std::vector<double>& bar_res_sq_b_oqe, std::vector<double>& bar_raw_gc_a_oqe,
    std::vector<double>& bar_res_gc_b_oqe, std::vector<double>& bar_raw_dc_a_oqe,
    std::vector<double>& bar_res_dc_b_oqe, double* bar_Fa, double* bar_Fb, double* bar_Ka,
    double* bar_Kb, double* bar_Sab) {
  e_f12_tot = 0.0;
  e_ord_tot = 0.0;
  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  if (oa_act <= 0 || ob_act <= 0 || (!include_f12 && !include_ordinary)) return 0;

  const int va = obs - oa;
  const int vb = obs - ob;
  const int c = ext - obs;
  const Size ee = Size(ext) * ext;
  const Size qe = Size(aux) * ext;

  if (!include_f12) {
    // Ordinary-only opposite-spin fast path.
    std::vector<double> G_vv(Size(va) * vb), bar_G_vv(Size(va) * vb);
    for (int i_loc = 0; i_loc < oa_act; ++i_loc) {
      const int i = fa + i_loc;
      const double* fit_i = fit_a_oqe.data() + Size(i_loc) * qe;
      double* bfit_i = bar_fit_a_oqe.data() + Size(i_loc) * qe;
      for (int j_loc = 0; j_loc < ob_act; ++j_loc) {
        const int j = fb + j_loc;
        const double* raw_coul_j = raw_coul_b_oqe.data() + Size(j_loc) * qe;
        double* braw_coul_j = bar_raw_coul_b_oqe.data() + Size(j_loc) * qe;
        rm_gemm_host(true, false, va, vb, aux, 1.0, fit_i + oa, ext, raw_coul_j + ob, ext, 0.0,
                     G_vv.data(), vb);
        const double wij = has_occ_weights ? 0.5 * (occ_w_a[i_loc] + occ_w_b[j_loc]) : 1.0;
        const double occ_sum = Fa[Size(i) * ext + i] + Fb[Size(j) * ext + j];
        double os_gg = 0.0, bar_occ_sum = 0.0;
        for (int a = 0; a < va; ++a) {
          const double ea = Fa[Size(oa + a) * ext + (oa + a)];
          for (int b = 0; b < vb; ++b) {
            const double gap = ea + Fb[Size(ob + b) * ext + (ob + b)] - occ_sum;
            if (!std::isfinite(gap) || gap == 0.0) return -4;
            const double inv_g = 1.0 / gap;
            const double gab = G_vv[Size(a) * vb + b];
            os_gg += gab * gab * inv_g;
            bar_G_vv[Size(a) * vb + b] = -2.0 * wij * gab * inv_g;
            const double bar_gap = wij * gab * gab * inv_g * inv_g;
            bar_Fa[Size(oa + a) * ext + (oa + a)] += bar_gap;
            bar_Fb[Size(ob + b) * ext + (ob + b)] += bar_gap;
            bar_occ_sum -= bar_gap;
          }
        }
        bar_Fa[Size(i) * ext + i] += bar_occ_sum;
        bar_Fb[Size(j) * ext + j] += bar_occ_sum;
        rm_gemm_host(false, true, aux, va, vb, 1.0, raw_coul_j + ob, ext, bar_G_vv.data(), vb, 1.0,
                     bfit_i + oa, ext);
        rm_gemm_host(false, false, aux, vb, va, 1.0, fit_i + oa, ext, bar_G_vv.data(), vb, 1.0,
                     braw_coul_j + ob, ext);
        const double e_ord = -os_gg;
        if (pair_ord_ab) pair_ord_ab[Size(i_loc) * ob_act + j_loc] = e_ord;
        e_ord_tot += wij * e_ord;
      }
    }
    return 0;
  }

  // Precompute cross-overlap & symmetrized Fock/exchange helper matrices once for AB channel.
  std::vector<double> Ha(ee), Hb(ee), Fa_sym(ee), Fb_sym(ee), Ha_sym(ee), Hb_sym(ee);
  for (int p = 0; p < ext; ++p) {
    for (int q = 0; q < ext; ++q) {
      const Size pq = Size(p) * ext + q;
      const Size qp = Size(q) * ext + p;
      Ha[pq] = Fa[pq] + Ka[pq];
      Hb[pq] = Fb[pq] + Kb[pq];
      Fa_sym[pq] = Fa[pq] + Fa[qp];
      Fb_sym[pq] = Fb[pq] + Fb[qp];
      Ha_sym[pq] = Fa_sym[pq] + Ka[pq] + Ka[qp];
      Hb_sym[pq] = Fb_sym[pq] + Kb[pq] + Kb[qp];
    }
  }
  std::vector<double> SS_T(ee, 0.0), ST_S(ee, 0.0);
  std::vector<double> Ha_S(ee, 0.0), Hb_ST(ee, 0.0);
  std::vector<double> ST_Ha(ee, 0.0), S_Hb(ee, 0.0);
  std::vector<double> ST_Ha_S(ee, 0.0), S_Hb_ST(ee, 0.0);
  rm_gemm_host(false, true, ext, ext, ext, 1.0, Sab, ext, Sab, ext, 0.0, SS_T.data(), ext);
  rm_gemm_host(true, false, ext, ext, ext, 1.0, Sab, ext, Sab, ext, 0.0, ST_S.data(), ext);
  rm_gemm_host(false, false, ext, ext, ext, 1.0, Ha.data(), ext, Sab, ext, 0.0, Ha_S.data(), ext);
  rm_gemm_host(false, true, ext, ext, ext, 1.0, Hb.data(), ext, Sab, ext, 0.0, Hb_ST.data(), ext);
  rm_gemm_host(true, false, ext, ext, ext, 1.0, Sab, ext, Ha.data(), ext, 0.0, ST_Ha.data(), ext);
  rm_gemm_host(false, false, ext, ext, ext, 1.0, Sab, ext, Hb.data(), ext, 0.0, S_Hb.data(), ext);
  rm_gemm_host(true, false, ext, ext, ext, 1.0, Sab, ext, Ha_S.data(), ext, 0.0, ST_Ha_S.data(),
               ext);
  rm_gemm_host(false, false, ext, ext, ext, 1.0, Sab, ext, Hb_ST.data(), ext, 0.0, S_Hb_ST.data(),
               ext);

  std::vector<double> u_j(ext), v_i(ext), a_i(ext), b_j(ext);
  std::vector<double> ha_i(ext), hb_j(ext), comb_rhs_i(ext), comb_lhs_j(ext);
  std::vector<double> st_ha_s_j(ext), s_hb_st_i(ext);
  std::vector<double> ha_uj(ext), hb_vi(ext);
  std::vector<double> e_i_vec(ext, 0.0), e_j_vec(ext, 0.0);
  std::vector<double> x1(ext), x3(ext), y2(ext), y4(ext);
  std::vector<double> bw_v(ext), bwT_u(ext);
  std::vector<double> sy_buf(ext), stx_buf(ext), mv1_buf(ext), mv2_buf(ext);
  std::vector<double> tmp_v1(ext), tmp_v2(ext), ff_ej(ext), ff_vi(ext), ffT_ei(ext), ffT_uj(ext);
  std::vector<double> tmp_aux1(aux), tmp_aux2(aux);

  std::vector<double> f0(ee), tmp_f(ee), f(ee), qf(ee);
  std::vector<double> bar_f(ee), bar_qf(ee), bar_f0(ee), ff_tmp(ee);
  std::vector<double> G_obs(Size(obs) * obs), G_oc(Size(oa) * c), G_co(Size(c) * ob);
  std::vector<double> bar_G_obs(Size(obs) * obs), bar_G_oc(Size(oa) * c), bar_G_co(Size(c) * ob);
  std::vector<double> C_vv(Size(va) * vb), bar_C_vv(Size(va) * vb);

  for (int i_loc = 0; i_loc < oa_act; ++i_loc) {
    const int i = fa + i_loc;
    const double* fit_i = fit_a_oqe.data() + Size(i_loc) * qe;
    const double* raw_g_i = raw_g_a_oqe.data() + Size(i_loc) * qe;
    const double* raw_sq_i = raw_sq_a_oqe.data() + Size(i_loc) * qe;
    const double* raw_gc_i = raw_gc_a_oqe.data() + Size(i_loc) * qe;
    const double* raw_dc_i = raw_dc_a_oqe.data() + Size(i_loc) * qe;
    double* bfit_i = bar_fit_a_oqe.data() + Size(i_loc) * qe;
    double* braw_g_i = bar_raw_g_a_oqe.data() + Size(i_loc) * qe;
    double* braw_sq_i = bar_raw_sq_a_oqe.data() + Size(i_loc) * qe;
    double* braw_gc_i = bar_raw_gc_a_oqe.data() + Size(i_loc) * qe;
    double* braw_dc_i = bar_raw_dc_a_oqe.data() + Size(i_loc) * qe;

    std::fill_n(e_i_vec.data(), ext, 0.0);
    e_i_vec[i] = 1.0;
    for (int p = 0; p < ext; ++p) {
      v_i[p] = Sab[Size(i) * ext + p];
      a_i[p] = SS_T[Size(p) * ext + i];
      ha_i[p] = Ha[Size(p) * ext + i];
      comb_rhs_i[p] = ST_Ha[Size(p) * ext + i] + Hb_ST[Size(p) * ext + i];
      s_hb_st_i[p] = S_Hb_ST[Size(p) * ext + i];
      hb_vi[p] = Hb_ST[Size(p) * ext + i];
    }

    for (int j_loc = 0; j_loc < ob_act; ++j_loc) {
      const int j = fb + j_loc;
      const double* fit_j = fit_b_oqe.data() + Size(j_loc) * qe;
      const double* raw_coul_j = raw_coul_b_oqe.data() + Size(j_loc) * qe;
      const double* res_g_j = res_g_b_oqe.data() + Size(j_loc) * qe;
      const double* res_sq_j = res_sq_b_oqe.data() + Size(j_loc) * qe;
      const double* res_gc_j = res_gc_b_oqe.data() + Size(j_loc) * qe;
      const double* res_dc_j = res_dc_b_oqe.data() + Size(j_loc) * qe;
      double* bfit_j = bar_fit_b_oqe.data() + Size(j_loc) * qe;
      double* braw_coul_j = bar_raw_coul_b_oqe.data() + Size(j_loc) * qe;
      double* bres_g_j = bar_res_g_b_oqe.data() + Size(j_loc) * qe;
      double* bres_sq_j = bar_res_sq_b_oqe.data() + Size(j_loc) * qe;
      double* bres_gc_j = bar_res_gc_b_oqe.data() + Size(j_loc) * qe;
      double* bres_dc_j = bar_res_dc_b_oqe.data() + Size(j_loc) * qe;

      std::fill_n(e_j_vec.data(), ext, 0.0);
      e_j_vec[j] = 1.0;
      for (int p = 0; p < ext; ++p) {
        u_j[p] = Sab[Size(p) * ext + j];
        b_j[p] = ST_S[Size(j) * ext + p];
        hb_j[p] = Hb[Size(p) * ext + j];
        comb_lhs_j[p] = S_Hb[Size(p) * ext + j] + Ha_S[Size(p) * ext + j];
        st_ha_s_j[p] = ST_Ha_S[Size(p) * ext + j];
        ha_uj[p] = Ha_S[Size(p) * ext + j];
      }

      // Forward Step 1: O(aux * ext) rank-structured scalars.
      const double gc_val =
          0.375 * robust_elem_host(fit_i, raw_gc_i, fit_j, res_gc_j, aux, ext, i, j) +
          0.125 * robust_bilin_host(fit_i, raw_gc_i, fit_j, res_gc_j, aux, ext, u_j.data(),
                                    v_i.data(), tmp_aux1, tmp_aux2);
      const double dc_val =
          (9.0 / 64.0) * robust_elem_host(fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, i, j) +
          (6.0 / 64.0) * robust_bilin_host(fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, u_j.data(),
                                           v_i.data(), tmp_aux1, tmp_aux2) +
          (1.0 / 64.0) * robust_bilin_host(fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, a_i.data(),
                                           b_j.data(), tmp_aux1, tmp_aux2);
      const double sq_norm_val =
          (9.0 / 64.0) * robust_elem_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, i, j) +
          (6.0 / 64.0) * robust_bilin_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, u_j.data(),
                                           v_i.data(), tmp_aux1, tmp_aux2) +
          (1.0 / 64.0) * robust_bilin_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, a_i.data(),
                                           b_j.data(), tmp_aux1, tmp_aux2);

      double sq_t1 = 0.0, sq_t3 = 0.0;
      for (int A = 0; A < aux; ++A) {
        const double* fi = fit_i + Size(A) * ext;
        const double* ri = raw_sq_i + Size(A) * ext;
        const double* fj = fit_j + Size(A) * ext;
        const double* sj = res_sq_j + Size(A) * ext;
        double fi_ha = 0.0, ri_ha = 0.0, sj_hb = 0.0, fj_hb = 0.0;
        for (int p = 0; p < ext; ++p) {
          fi_ha += fi[p] * ha_i[p];
          ri_ha += ri[p] * ha_i[p];
          sj_hb += sj[p] * hb_j[p];
          fj_hb += fj[p] * hb_j[p];
        }
        sq_t1 += fi_ha * sj[j] + ri_ha * fj[j];
        sq_t3 += fi[i] * sj_hb + ri[i] * fj_hb;
      }
      const double sq_fk_val =
          (9.0 / 64.0) * (sq_t1 + sq_t3) +
          (3.0 / 64.0) * robust_bilin_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, u_j.data(),
                                           comb_rhs_i.data(), tmp_aux1, tmp_aux2) +
          (3.0 / 64.0) * robust_bilin_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext,
                                           comb_lhs_j.data(), v_i.data(), tmp_aux1, tmp_aux2) +
          (1.0 / 64.0) * robust_bilin_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, a_i.data(),
                                           st_ha_s_j.data(), tmp_aux1, tmp_aux2) +
          (1.0 / 64.0) * robust_bilin_host(fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext,
                                           s_hb_st_i.data(), b_j.data(), tmp_aux1, tmp_aux2);

      // Forward Step 2: f0, f = W(f0), qf = Q(f), Coulomb blocks.
      rm_gemm_host(true, false, ext, ext, aux, 1.0, fit_i, ext, res_g_j, ext, 0.0, f0.data(), ext);
      rm_gemm_host(true, false, ext, ext, aux, 1.0, raw_g_i, ext, fit_j, ext, 1.0, f0.data(), ext);
      rm_gemm_host(false, true, ext, ext, ext, 1.0, Sab, ext, f0.data(), ext, 0.0, tmp_f.data(),
                   ext);
      std::copy_n(f0.data(), ee, f.data());
      rm_gemm_host(false, false, ext, ext, ext, 0.125, tmp_f.data(), ext, Sab, ext, 0.375, f.data(),
                   ext);
      std::copy_n(f.data(), ee, qf.data());
      zero_p_blocks_host(qf.data(), oa, ob, obs, ext);

      rm_gemm_host(true, false, obs, obs, aux, 1.0, fit_i, ext, raw_coul_j, ext, 0.0, G_obs.data(),
                   obs);
      rm_gemm_host(true, false, oa, c, aux, 1.0, fit_i, ext, raw_coul_j + obs, ext, 0.0,
                   G_oc.data(), c);
      rm_gemm_host(true, false, c, ob, aux, 1.0, fit_i + obs, ext, raw_coul_j, ext, 0.0,
                   G_co.data(), ob);

      const double linear =
          gc_val - dot_rect_host(G_obs.data(), obs, f.data(), ext, obs, obs) -
          dot_rect_host(G_oc.data(), c, f.data() + obs, ext, oa, c) -
          dot_rect_host(G_co.data(), ob, f.data() + Size(obs) * ext, ext, c, ob);
      const double norm = sq_norm_val - sqnorm_rect_host(f.data(), ext, obs, obs) -
                          sqnorm_rect_host(f.data() + obs, ext, oa, c) -
                          sqnorm_rect_host(f.data() + Size(obs) * ext, ext, c, ob);

      // Forward Step 3: b_val = dc_val + sq_fk_val - <f, Ha f + f Hb^T> + <qf, Fa qf + qf Fb^T>
      double b_val = dc_val + sq_fk_val;
      rm_gemm_host(false, false, ext, ext, ext, 1.0, Ha.data(), ext, f.data(), ext, 0.0,
                   tmp_f.data(), ext);
      b_val -= dot_rect_host(f.data(), ext, tmp_f.data(), ext, ext, ext);
      rm_gemm_host(false, true, ext, ext, ext, 1.0, f.data(), ext, Hb.data(), ext, 0.0,
                   tmp_f.data(), ext);
      b_val -= dot_rect_host(f.data(), ext, tmp_f.data(), ext, ext, ext);
      rm_gemm_host(false, false, ext, ext, ext, 1.0, Fa, ext, qf.data(), ext, 0.0, tmp_f.data(),
                   ext);
      b_val += dot_rect_host(qf.data(), ext, tmp_f.data(), ext, ext, ext);
      rm_gemm_host(false, true, ext, ext, ext, 1.0, qf.data(), ext, Fb, ext, 0.0, tmp_f.data(),
                   ext);
      b_val += dot_rect_host(qf.data(), ext, tmp_f.data(), ext, ext, ext);

      // Forward Step 4: Coupling C_vv[a, b] = Fa[oa+a, obs:] f[obs:, ob+b] + f[oa+a, obs:] Fb[ob+b, obs:]^T
      rm_gemm_host(false, false, va, vb, c, 1.0, Fa + Size(oa) * ext + obs, ext,
                   f.data() + Size(obs) * ext + ob, ext, 0.0, C_vv.data(), vb);
      rm_gemm_host(false, true, va, vb, c, 1.0, f.data() + Size(oa) * ext + obs, ext,
                   Fb + Size(ob) * ext + obs, ext, 1.0, C_vv.data(), vb);

      const double wij = has_occ_weights ? 0.5 * (occ_w_a[i_loc] + occ_w_b[j_loc]) : 1.0;
      const double s_f12 = wij;
      const double s_ord = include_ordinary ? wij : 0.0;
      const double occ_sum = Fa[Size(i) * ext + i] + Fb[Size(j) * ext + j];

      // Reverse Step U1: energy & denominator contraction.
      std::fill(bar_G_obs.begin(), bar_G_obs.end(), 0.0);
      std::fill(bar_G_oc.begin(), bar_G_oc.end(), 0.0);
      std::fill(bar_G_co.begin(), bar_G_co.end(), 0.0);
      double os_term = 0.0, os_gg = 0.0;
      double bar_occ_sum = -s_f12 * norm;
      for (int a = 0; a < va; ++a) {
        const double ea = Fa[Size(oa + a) * ext + (oa + a)];
        for (int b = 0; b < vb; ++b) {
          const double gap = ea + Fb[Size(ob + b) * ext + (ob + b)] - occ_sum;
          if (!std::isfinite(gap) || gap == 0.0) return -4;
          const double inv_g = 1.0 / gap;
          const double cab = C_vv[Size(a) * vb + b];
          const double gab = G_obs[Size(oa + a) * obs + (ob + b)];
          os_term += cab * (2.0 * gab + cab) * inv_g;
          if (include_ordinary) os_gg += gab * gab * inv_g;

          bar_C_vv[Size(a) * vb + b] = -2.0 * s_f12 * (gab + cab) * inv_g;
          bar_G_obs[Size(oa + a) * obs + (ob + b)] =
              (-2.0 * s_f12 * cab - 2.0 * s_ord * gab) * inv_g;
          const double bar_inv_g =
              -s_f12 * cab * (2.0 * gab + cab) - s_ord * gab * gab;
          const double bar_gap = -bar_inv_g * inv_g * inv_g;
          bar_Fa[Size(oa + a) * ext + (oa + a)] += bar_gap;
          bar_Fb[Size(ob + b) * ext + (ob + b)] += bar_gap;
          bar_occ_sum -= bar_gap;
        }
      }
      bar_Fa[Size(i) * ext + i] += bar_occ_sum;
      bar_Fb[Size(j) * ext + j] += bar_occ_sum;

      const double e_f12 = 2.0 * linear + b_val - occ_sum * norm - os_term;
      const double e_ord = include_ordinary ? (-os_gg) : 0.0;
      const Size idx = Size(i_loc) * ob_act + j_loc;
      if (pair_f12_ab) pair_f12_ab[idx] = e_f12;
      if (include_ordinary && pair_ord_ab) pair_ord_ab[idx] = e_ord;
      e_f12_tot += wij * e_f12;
      if (include_ordinary) e_ord_tot += wij * e_ord;

      // Reverse Step U2: Coupling C_vv -> bar_f, bar_Fa, bar_Fb.
      std::fill(bar_f.begin(), bar_f.end(), 0.0);
      rm_gemm_host(false, true, va, c, vb, 1.0, bar_C_vv.data(), vb,
                   f.data() + Size(obs) * ext + ob, ext, 1.0, bar_Fa + Size(oa) * ext + obs, ext);
      rm_gemm_host(true, false, c, vb, va, 1.0, Fa + Size(oa) * ext + obs, ext, bar_C_vv.data(), vb,
                   1.0, bar_f.data() + Size(obs) * ext + ob, ext);
      rm_gemm_host(false, false, va, c, vb, 1.0, bar_C_vv.data(), vb, Fb + Size(ob) * ext + obs,
                   ext, 1.0, bar_f.data() + Size(oa) * ext + obs, ext);
      rm_gemm_host(true, false, vb, c, va, 1.0, bar_C_vv.data(), vb,
                   f.data() + Size(oa) * ext + obs, ext, 1.0, bar_Fb + Size(ob) * ext + obs, ext);

      // Reverse Step U3: b terms in f and qf -> bar_Fa, bar_Fb, bar_Ka, bar_Kb, bar_f.
      rm_gemm_host(false, true, ext, ext, ext, 1.0, f.data(), ext, f.data(), ext, 0.0,
                   ff_tmp.data(), ext);
      for (Size p = 0; p < ee; ++p) {
        const double v_sub = s_f12 * ff_tmp[p];
        bar_Ka[p] -= v_sub;
        bar_Fa[p] -= v_sub;
      }
      rm_gemm_host(true, false, ext, ext, ext, 1.0, f.data(), ext, f.data(), ext, 0.0,
                   ff_tmp.data(), ext);
      for (Size p = 0; p < ee; ++p) {
        const double v_sub = s_f12 * ff_tmp[p];
        bar_Kb[p] -= v_sub;
        bar_Fb[p] -= v_sub;
      }
      rm_gemm_host(false, true, ext, ext, ext, s_f12, qf.data(), ext, qf.data(), ext, 1.0, bar_Fa,
                   ext);
      rm_gemm_host(true, false, ext, ext, ext, s_f12, qf.data(), ext, qf.data(), ext, 1.0, bar_Fb,
                   ext);

      rm_gemm_host(false, false, ext, ext, ext, -s_f12, Ha_sym.data(), ext, f.data(), ext, 1.0,
                   bar_f.data(), ext);
      rm_gemm_host(false, false, ext, ext, ext, -s_f12, f.data(), ext, Hb_sym.data(), ext, 1.0,
                   bar_f.data(), ext);

      rm_gemm_host(false, false, ext, ext, ext, s_f12, Fa_sym.data(), ext, qf.data(), ext, 0.0,
                   bar_qf.data(), ext);
      rm_gemm_host(false, false, ext, ext, ext, s_f12, qf.data(), ext, Fb_sym.data(), ext, 1.0,
                   bar_qf.data(), ext);
      zero_p_blocks_host(bar_qf.data(), oa, ob, obs, ext);
      for (Size p = 0; p < ee; ++p) bar_f[p] += bar_qf[p];

      // Reverse Step U4: linear and norm terms in g and f.
      const double bar_linear = 2.0 * s_f12;
      const double bar_norm = -s_f12 * occ_sum;
      for (int p = 0; p < obs; ++p) {
        for (int r = 0; r < obs; ++r) {
          const double fpr = f[Size(p) * ext + r];
          bar_G_obs[Size(p) * obs + r] -= bar_linear * fpr;
          bar_f[Size(p) * ext + r] -= bar_linear * G_obs[Size(p) * obs + r] + 2.0 * bar_norm * fpr;
        }
      }
      for (int p = 0; p < oa; ++p) {
        for (int q = 0; q < c; ++q) {
          const double fpq = f[Size(p) * ext + (obs + q)];
          bar_G_oc[Size(p) * c + q] -= bar_linear * fpq;
          bar_f[Size(p) * ext + (obs + q)] -=
              bar_linear * G_oc[Size(p) * c + q] + 2.0 * bar_norm * fpq;
        }
      }
      for (int p = 0; p < c; ++p) {
        for (int q = 0; q < ob; ++q) {
          const double fpq = f[Size(obs + p) * ext + q];
          bar_G_co[Size(p) * ob + q] -= bar_linear * fpq;
          bar_f[Size(obs + p) * ext + q] -=
              bar_linear * G_co[Size(p) * ob + q] + 2.0 * bar_norm * fpq;
        }
      }

      // Reverse Step U5: Pullback of f = W(f0) -> bar_f0, bar_Sab.
      rm_gemm_host(false, true, ext, ext, ext, 1.0, Sab, ext, bar_f.data(), ext, 0.0, tmp_f.data(),
                   ext);
      std::copy_n(bar_f.data(), ee, bar_f0.data());
      rm_gemm_host(false, false, ext, ext, ext, 0.125, tmp_f.data(), ext, Sab, ext, 0.375,
                   bar_f0.data(), ext);
      if (bar_Sab) {
        // bar_Sab += (1/8) * ((bar_f @ Sab^T) @ f0 + (f0 @ Sab^T) @ bar_f)
        rm_gemm_host(false, true, ext, ext, ext, 1.0, bar_f.data(), ext, Sab, ext, 0.0,
                     tmp_f.data(), ext);
        rm_gemm_host(false, false, ext, ext, ext, 0.125, tmp_f.data(), ext, f0.data(), ext, 1.0,
                     bar_Sab, ext);
        rm_gemm_host(false, true, ext, ext, ext, 1.0, f0.data(), ext, Sab, ext, 0.0, tmp_f.data(),
                     ext);
        rm_gemm_host(false, false, ext, ext, ext, 0.125, tmp_f.data(), ext, bar_f.data(), ext, 1.0,
                     bar_Sab, ext);
      }

      // Reverse Step U6: Pullback of f0 and G_obs/G_oc/G_co into fit/raw/res.
      rm_gemm_host(false, true, aux, ext, ext, 1.0, res_g_j, ext, bar_f0.data(), ext, 1.0, bfit_i,
                   ext);
      rm_gemm_host(false, false, aux, ext, ext, 1.0, fit_i, ext, bar_f0.data(), ext, 1.0, bres_g_j,
                   ext);
      rm_gemm_host(false, true, aux, ext, ext, 1.0, fit_j, ext, bar_f0.data(), ext, 1.0, braw_g_i,
                   ext);
      rm_gemm_host(false, false, aux, ext, ext, 1.0, raw_g_i, ext, bar_f0.data(), ext, 1.0, bfit_j,
                   ext);

      rm_gemm_host(false, true, aux, obs, obs, 1.0, raw_coul_j, ext, bar_G_obs.data(), obs, 1.0,
                   bfit_i, ext);
      rm_gemm_host(false, false, aux, obs, obs, 1.0, fit_i, ext, bar_G_obs.data(), obs, 1.0,
                   braw_coul_j, ext);
      rm_gemm_host(false, true, aux, oa, c, 1.0, raw_coul_j + obs, ext, bar_G_oc.data(), c, 1.0,
                   bfit_i, ext);
      rm_gemm_host(false, false, aux, c, oa, 1.0, fit_i, ext, bar_G_oc.data(), c, 1.0,
                   braw_coul_j + obs, ext);
      rm_gemm_host(false, true, aux, c, ob, 1.0, raw_coul_j, ext, bar_G_co.data(), ob, 1.0,
                   bfit_i + obs, ext);
      rm_gemm_host(false, false, aux, ob, c, 1.0, fit_i + obs, ext, bar_G_co.data(), ob, 1.0,
                   braw_coul_j, ext);

      // Reverse Step U7: Exact O(aux * ext) rank-structured pullback of gc0, uf, ff, Ha, Hb, w_mat.
      std::fill(bw_v.begin(), bw_v.end(), 0.0);
      std::fill(bwT_u.begin(), bwT_u.end(), 0.0);

      // 7a. gc0: bar_gc0 = bar_linear * w_mat, bar_w += bar_linear * gc0
      robust_elem_pullback_host(0.375 * bar_linear, i, j, aux, ext, fit_i, raw_gc_i, fit_j,
                                res_gc_j, bfit_i, braw_gc_i, bfit_j, bres_gc_j);
      robust_bilin_pullback_host(0.125 * bar_linear, u_j.data(), v_i.data(), aux, ext, fit_i,
                                 raw_gc_i, fit_j, res_gc_j, bfit_i, braw_gc_i, bfit_j, bres_gc_j);
      if (bar_Sab) {
        robust_rmatvec_host(fit_i, raw_gc_i, fit_j, res_gc_j, aux, ext, v_i.data(), tmp_v1.data());
        robust_lmatvec_host(fit_i, raw_gc_i, fit_j, res_gc_j, aux, ext, u_j.data(), tmp_v2.data());
        for (int p = 0; p < ext; ++p) {
          bw_v[p] += bar_linear * tmp_v1[p];
          bwT_u[p] += bar_linear * tmp_v2[p];
        }
      }

      // 7b. uf = W(uf0): bar_uf = s_f12 * w_mat, bar_w += s_f12 * uf
      weight_rank1_pullback_host(0.375 * s_f12, e_i_vec.data(), e_j_vec.data(), Sab, aux, ext,
                                 fit_i, raw_dc_i, fit_j, res_dc_j, bfit_i, braw_dc_i, bfit_j,
                                 bres_dc_j, bar_Sab, sy_buf, stx_buf, mv1_buf, mv2_buf);
      weight_rank1_pullback_host(0.125 * s_f12, u_j.data(), v_i.data(), Sab, aux, ext, fit_i,
                                 raw_dc_i, fit_j, res_dc_j, bfit_i, braw_dc_i, bfit_j, bres_dc_j,
                                 bar_Sab, sy_buf, stx_buf, mv1_buf, mv2_buf);
      if (bar_Sab) {
        weight_rmatvec_host(Sab, fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, v_i.data(),
                            mv1_buf.data(), tmp_v1, tmp_v2);
        weight_lmatvec_host(Sab, fit_i, raw_dc_i, fit_j, res_dc_j, aux, ext, u_j.data(),
                            mv2_buf.data(), tmp_v1, tmp_v2);
        for (int p = 0; p < ext; ++p) {
          bw_v[p] += s_f12 * mv1_buf[p];
          bwT_u[p] += s_f12 * mv2_buf[p];
        }
      }

      // 7c. ff = W(ff0): bar_ff = bar_norm * w_mat + s_f12 * (Ha @ w_mat + w_mat @ Hb^T)
      for (int p = 0; p < ext; ++p) {
        x1[p] = s_f12 * ha_i[p] + (p == i ? bar_norm : 0.0);
        y2[p] = s_f12 * hb_j[p];
        x3[p] = s_f12 * ha_uj[p] + bar_norm * u_j[p];
        y4[p] = s_f12 * hb_vi[p];
      }
      weight_rank1_pullback_host(0.375, x1.data(), e_j_vec.data(), Sab, aux, ext, fit_i, raw_sq_i,
                                 fit_j, res_sq_j, bfit_i, braw_sq_i, bfit_j, bres_sq_j, bar_Sab,
                                 sy_buf, stx_buf, mv1_buf, mv2_buf);
      weight_rank1_pullback_host(0.375, e_i_vec.data(), y2.data(), Sab, aux, ext, fit_i, raw_sq_i,
                                 fit_j, res_sq_j, bfit_i, braw_sq_i, bfit_j, bres_sq_j, bar_Sab,
                                 sy_buf, stx_buf, mv1_buf, mv2_buf);
      weight_rank1_pullback_host(0.125, x3.data(), v_i.data(), Sab, aux, ext, fit_i, raw_sq_i,
                                 fit_j, res_sq_j, bfit_i, braw_sq_i, bfit_j, bres_sq_j, bar_Sab,
                                 sy_buf, stx_buf, mv1_buf, mv2_buf);
      weight_rank1_pullback_host(0.125, u_j.data(), y4.data(), Sab, aux, ext, fit_i, raw_sq_i,
                                 fit_j, res_sq_j, bfit_i, braw_sq_i, bfit_j, bres_sq_j, bar_Sab,
                                 sy_buf, stx_buf, mv1_buf, mv2_buf);

      // bar_Ha += s_f12 * (ff @ w_mat^T), bar_Hb += s_f12 * (ff^T @ w_mat)
      weight_rmatvec_host(Sab, fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, e_j_vec.data(),
                          ff_ej.data(), tmp_v1, tmp_v2);
      weight_rmatvec_host(Sab, fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, v_i.data(), ff_vi.data(),
                          tmp_v1, tmp_v2);
      weight_lmatvec_host(Sab, fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, e_i_vec.data(),
                          ffT_ei.data(), tmp_v1, tmp_v2);
      weight_lmatvec_host(Sab, fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, u_j.data(),
                          ffT_uj.data(), tmp_v1, tmp_v2);
      for (int p = 0; p < ext; ++p) {
        const double ha_col_i = 0.375 * s_f12 * ff_ej[p];
        const double hb_col_j = 0.375 * s_f12 * ffT_ei[p];
        bar_Fa[Size(p) * ext + i] += ha_col_i;
        bar_Ka[Size(p) * ext + i] += ha_col_i;
        bar_Fb[Size(p) * ext + j] += hb_col_j;
        bar_Kb[Size(p) * ext + j] += hb_col_j;
      }
      add_outer_host(0.125 * s_f12, ff_vi.data(), u_j.data(), ext, bar_Fa);
      add_outer_host(0.125 * s_f12, ff_vi.data(), u_j.data(), ext, bar_Ka);
      add_outer_host(0.125 * s_f12, ffT_uj.data(), v_i.data(), ext, bar_Fb);
      add_outer_host(0.125 * s_f12, ffT_uj.data(), v_i.data(), ext, bar_Kb);

      // bar_w += bar_norm * ff + s_f12 * (Ha^T @ ff + ff @ Hb) -> bar_Sab
      if (bar_Sab) {
        matvec_host(true, Ha.data(), ext, ff_vi.data(), mv1_buf.data());
        weight_rmatvec_host(Sab, fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, hb_vi.data(),
                            mv2_buf.data(), tmp_v1, tmp_v2);
        for (int p = 0; p < ext; ++p) {
          bw_v[p] += bar_norm * ff_vi[p] + s_f12 * (mv1_buf[p] + mv2_buf[p]);
        }
        matvec_host(true, Hb.data(), ext, ffT_uj.data(), mv1_buf.data());
        weight_lmatvec_host(Sab, fit_i, raw_sq_i, fit_j, res_sq_j, aux, ext, ha_uj.data(),
                            mv2_buf.data(), tmp_v1, tmp_v2);
        for (int p = 0; p < ext; ++p) {
          bwT_u[p] += bar_norm * ffT_uj[p] + s_f12 * (mv1_buf[p] + mv2_buf[p]);
        }
        for (int p = 0; p < ext; ++p) {
          bar_Sab[Size(p) * ext + j] += 0.125 * bw_v[p];
          bar_Sab[Size(i) * ext + p] += 0.125 * bwT_u[p];
        }
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

// -------------------------------------------------------------------------
// GPU MP2-F12 Forward Pair Energies (V / B / X / C, 3C(FIX) SP)
// -------------------------------------------------------------------------

bool pack_active_qoe_checked_cuda_host(const double* src, int aux, int n1, int f, int o_act,
                                       int ext, std::vector<double>& out_qoe) {
  out_qoe.resize(Size(aux) * Size(o_act) * ext);
  for (int A = 0; A < aux; ++A) {
    const double* s_base = src + (Size(A) * n1 + f) * ext;
    double* d_base = out_qoe.data() + Size(A) * o_act * ext;
    std::copy_n(s_base, Size(o_act) * ext, d_base);
  }
  return all_finite_host(out_qoe.data(), out_qoe.size());
}

__global__ void k_add_2d(Size total, const double* __restrict__ A, const double* __restrict__ B,
                         double* __restrict__ C) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) C[idx] = A[idx] + B[idx];
}

__global__ void k_qoe_to_oqe(int aux, int o_act, int ext, const double* __restrict__ qoe,
                             double* __restrict__ oqe) {
  const Size total = Size(aux) * Size(o_act) * Size(ext);
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int p = static_cast<int>(idx % ext);
  const Size ao = idx / ext;
  const int i = static_cast<int>(ao % o_act);
  const int A = static_cast<int>(ao / o_act);
  oqe[(Size(i) * aux + A) * ext + p] = qoe[idx];
}

__global__ void k_extract_qoe_occ(int aux, int o_act, int ext, int o,
                                  const double* __restrict__ qoe, double* __restrict__ oqo) {
  const Size total = Size(aux) * Size(o_act) * Size(o);
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int m = static_cast<int>(idx % o);
  const Size ao = idx / o;
  const int i = static_cast<int>(ao % o_act);
  const int A = static_cast<int>(ao / o_act);
  oqo[(Size(i) * aux + A) * o + m] = qoe[(Size(A) * o_act + i) * ext + m];
}

__global__ void k_extract_qoe_diag(int aux, int o_act, int ext, int f,
                                   const double* __restrict__ qoe, double* __restrict__ diag) {
  const Size total = Size(aux) * Size(o_act);
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int i = static_cast<int>(idx % o_act);
  const int A = static_cast<int>(idx / o_act);
  diag[idx] = qoe[(Size(A) * o_act + i) * ext + (f + i)];
}

__global__ void k_ss_robust_diag_elems(int aux, int o_act, int o, int f,
                                       const double* __restrict__ fit_oqo,
                                       const double* __restrict__ raw_oqo,
                                       const double* __restrict__ res_oqo,
                                       double* __restrict__ out_d, double* __restrict__ out_x) {
  __shared__ double sd[256];
  __shared__ double sx[256];
  const int pair_id = static_cast<int>(blockIdx.x);
  const int i_loc = pair_id / o_act;
  const int j_loc = pair_id % o_act;
  const int i = f + i_loc;
  const int j = f + j_loc;
  const Size base_i = Size(i_loc) * aux * o;
  const Size base_j = Size(j_loc) * aux * o;

  double ld = 0.0;
  double lx = 0.0;
  for (int A = threadIdx.x; A < aux; A += blockDim.x) {
    const Size ri = base_i + Size(A) * o;
    const Size rj = base_j + Size(A) * o;
    ld += fit_oqo[ri + i] * res_oqo[rj + j] + raw_oqo[ri + i] * fit_oqo[rj + j];
    lx += fit_oqo[ri + j] * res_oqo[rj + i] + raw_oqo[ri + j] * fit_oqo[rj + i];
  }
  sd[threadIdx.x] = ld;
  sx[threadIdx.x] = lx;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sd[threadIdx.x] += sd[threadIdx.x + stride];
      sx[threadIdx.x] += sx[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out_d[pair_id] = sd[0];
    out_x[pair_id] = sx[0];
  }
}

__global__ void k_ss_sq_fk(int aux, int o_act, int o, int f, const double* __restrict__ fit_oqo,
                           const double* __restrict__ raw_oqo, const double* __restrict__ res_oqo,
                           const double* __restrict__ fit_H, const double* __restrict__ raw_H,
                           const double* __restrict__ res_H, double* __restrict__ out_d,
                           double* __restrict__ out_x) {
  __shared__ double sd[256];
  __shared__ double sx[256];
  const int pair_id = static_cast<int>(blockIdx.x);
  const int i_loc = pair_id / o_act;
  const int j_loc = pair_id % o_act;
  const int i = f + i_loc;
  const int j = f + j_loc;
  const Size base_i = Size(i_loc) * aux * o;
  const Size base_j = Size(j_loc) * aux * o;

  double ld = 0.0;
  double lx = 0.0;
  for (int A = threadIdx.x; A < aux; A += blockDim.x) {
    const Size ri = base_i + Size(A) * o;
    const Size rj = base_j + Size(A) * o;
    const Size hi = (Size(A) * o_act + i_loc) * o;
    const Size hj = (Size(A) * o_act + j_loc) * o;
    ld += fit_oqo[ri + i] * res_H[hj + j] + raw_oqo[ri + i] * fit_H[hj + j] +
          fit_H[hi + i] * res_oqo[rj + j] + raw_H[hi + i] * fit_oqo[rj + j];
    lx += fit_oqo[ri + j] * res_H[hj + i] + raw_oqo[ri + j] * fit_H[hj + i] +
          fit_H[hi + j] * res_oqo[rj + i] + raw_H[hi + j] * fit_oqo[rj + i];
  }
  sd[threadIdx.x] = ld;
  sx[threadIdx.x] = lx;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sd[threadIdx.x] += sd[threadIdx.x + stride];
      sx[threadIdx.x] += sx[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out_d[pair_id] = sd[0];
    out_x[pair_id] = sx[0];
  }
}

__global__ void k_build_ab_proj_L(int oa_act, int ob_act, int fa, int fb, int ext,
                                  const double* __restrict__ Sab, const double* __restrict__ SS_T,
                                  const double* __restrict__ Ha, const double* __restrict__ Ha_S,
                                  const double* __restrict__ S_Hb,
                                  const double* __restrict__ S_Hb_ST, double* __restrict__ P_L) {
  const int n_vec_L = 2 * ob_act + 3 * oa_act;
  const Size total = Size(n_vec_L) * ext;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int row = static_cast<int>(idx / ext);
  const int p = static_cast<int>(idx % ext);
  if (row < ob_act) {
    const int j = fb + row;
    P_L[idx] = Sab[Size(p) * ext + j];
  } else if (row < 2 * ob_act) {
    const int j = fb + (row - ob_act);
    P_L[idx] = S_Hb[Size(p) * ext + j] + Ha_S[Size(p) * ext + j];
  } else if (row < 2 * ob_act + oa_act) {
    const int i = fa + (row - 2 * ob_act);
    P_L[idx] = SS_T[Size(p) * ext + i];
  } else if (row < 2 * ob_act + 2 * oa_act) {
    const int i = fa + (row - 2 * ob_act - oa_act);
    P_L[idx] = Ha[Size(p) * ext + i];
  } else {
    const int i = fa + (row - 2 * ob_act - 2 * oa_act);
    P_L[idx] = S_Hb_ST[Size(p) * ext + i];
  }
}

__global__ void k_build_ab_proj_R(int oa_act, int ob_act, int fa, int fb, int ext,
                                  const double* __restrict__ Sab, const double* __restrict__ ST_S,
                                  const double* __restrict__ Hb, const double* __restrict__ ST_Ha,
                                  const double* __restrict__ Hb_ST,
                                  const double* __restrict__ ST_Ha_S, double* __restrict__ P_R) {
  const int n_vec_R = 2 * oa_act + 3 * ob_act;
  const Size total = Size(n_vec_R) * ext;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int row = static_cast<int>(idx / ext);
  const int q = static_cast<int>(idx % ext);
  if (row < oa_act) {
    const int i = fa + row;
    P_R[idx] = Sab[Size(i) * ext + q];
  } else if (row < 2 * oa_act) {
    const int i = fa + (row - oa_act);
    P_R[idx] = ST_Ha[Size(q) * ext + i] + Hb_ST[Size(q) * ext + i];
  } else if (row < 2 * oa_act + ob_act) {
    const int j = fb + (row - 2 * oa_act);
    P_R[idx] = ST_S[Size(j) * ext + q];
  } else if (row < 2 * oa_act + 2 * ob_act) {
    const int j = fb + (row - 2 * oa_act - ob_act);
    P_R[idx] = Hb[Size(q) * ext + j];
  } else {
    const int j = fb + (row - 2 * oa_act - 2 * ob_act);
    P_R[idx] = ST_Ha_S[Size(q) * ext + j];
  }
}

__global__ void k_ab_stage1_scalars(
    int op_mode, int aux, int oa_act, int ob_act, const double* __restrict__ fit_a_diag,
    const double* __restrict__ raw_a_diag, const double* __restrict__ fit_b_diag,
    const double* __restrict__ res_b_diag, const double* __restrict__ fit_a_PL,
    const double* __restrict__ raw_a_PL, const double* __restrict__ fit_b_PR,
    const double* __restrict__ res_b_PR, double* __restrict__ out1, double* __restrict__ out2) {
  __shared__ double s1[256];
  __shared__ double s2[256];
  const int pair_id = static_cast<int>(blockIdx.x);
  const int i_loc = pair_id / ob_act;
  const int j_loc = pair_id % ob_act;
  const int n_vec_L = 2 * ob_act + 3 * oa_act;
  const int n_vec_R = 2 * oa_act + 3 * ob_act;
  const int off_L_U = 0;
  const int off_L_Comb = ob_act;
  const int off_L_A = 2 * ob_act;
  const int off_L_Ha = 2 * ob_act + oa_act;
  const int off_L_SHbST = 2 * ob_act + 2 * oa_act;
  const int off_R_V = 0;
  const int off_R_Comb = oa_act;
  const int off_R_B = 2 * oa_act;
  const int off_R_Hb = 2 * oa_act + ob_act;
  const int off_R_STHaS = 2 * oa_act + 2 * ob_act;

  double v1 = 0.0;
  double v2 = 0.0;
  for (int A = threadIdx.x; A < aux; A += blockDim.x) {
    const double fa_d = fit_a_diag[Size(A) * oa_act + i_loc];
    const double ra_d = raw_a_diag[Size(A) * oa_act + i_loc];
    const double fb_d = fit_b_diag[Size(A) * ob_act + j_loc];
    const double sb_d = res_b_diag[Size(A) * ob_act + j_loc];
    const double e_ij = fa_d * sb_d + ra_d * fb_d;

    const Size l_base = (Size(A) * oa_act + i_loc) * n_vec_L;
    const Size r_base = (Size(A) * ob_act + j_loc) * n_vec_R;
    auto bilin = [&](int lr, int rr) -> double {
      return fit_a_PL[l_base + lr] * res_b_PR[r_base + rr] +
             raw_a_PL[l_base + lr] * fit_b_PR[r_base + rr];
    };

    const double uv = bilin(off_L_U + j_loc, off_R_V + i_loc);
    if (op_mode == 0) {
      v1 += 0.375 * e_ij + 0.125 * uv;
    } else {
      const double ab_t = bilin(off_L_A + i_loc, off_R_B + j_loc);
      v1 += (9.0 / 64.0) * e_ij + (6.0 / 64.0) * uv + (1.0 / 64.0) * ab_t;
      if (op_mode == 2) {
        const double sq_t1 = fit_a_PL[l_base + off_L_Ha + i_loc] * sb_d +
                             raw_a_PL[l_base + off_L_Ha + i_loc] * fb_d;
        const double sq_t3 = fa_d * res_b_PR[r_base + off_R_Hb + j_loc] +
                             ra_d * fit_b_PR[r_base + off_R_Hb + j_loc];
        const double t27 = bilin(off_L_U + j_loc, off_R_Comb + i_loc);
        const double t45 = bilin(off_L_Comb + j_loc, off_R_V + i_loc);
        const double t6 = bilin(off_L_A + i_loc, off_R_STHaS + j_loc);
        const double t8 = bilin(off_L_SHbST + i_loc, off_R_B + j_loc);
        v2 += (9.0 / 64.0) * (sq_t1 + sq_t3) + (3.0 / 64.0) * (t27 + t45) +
              (1.0 / 64.0) * (t6 + t8);
      }
    }
  }
  s1[threadIdx.x] = v1;
  s2[threadIdx.x] = v2;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s1[threadIdx.x] += s1[threadIdx.x + stride];
      s2[threadIdx.x] += s2[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out1[pair_id] = s1[0];
    if (op_mode == 2 && out2) out2[pair_id] = s2[0];
  }
}

__global__ void k_batched_transpose(int batch, int n, const double* __restrict__ src,
                                    double* __restrict__ dst) {
  __shared__ double tile[32][33];
  const int b = static_cast<int>(blockIdx.z);
  if (b >= batch) return;
  const Size nn = Size(n) * n;
  const double* s = src + Size(b) * nn;
  double* d = dst + Size(b) * nn;
  const int x = static_cast<int>(blockIdx.x) * 32 + threadIdx.x;
  const int y = static_cast<int>(blockIdx.y) * 32 + threadIdx.y;
  if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = s[Size(y) * n + x];
  __syncthreads();
  const int tx = static_cast<int>(blockIdx.y) * 32 + threadIdx.x;
  const int ty = static_cast<int>(blockIdx.x) * 32 + threadIdx.y;
  if (tx < n && ty < n) d[Size(ty) * n + tx] = tile[threadIdx.x][threadIdx.y];
}

void launch_batched_transpose(int batch, int n, const double* src, double* dst) {
  if (batch <= 0 || n <= 0) return;
  dim3 block(32, 32);
  dim3 grid(static_cast<unsigned>((n + 31) / 32), static_cast<unsigned>((n + 31) / 32),
            static_cast<unsigned>(batch));
  k_batched_transpose<<<grid, block>>>(batch, n, src, dst);
}

__global__ void k_pack_rect_batch(int batch, int src_rows, int src_cols, int r0, int c0, int rows,
                                  int cols, const double* __restrict__ src,
                                  double* __restrict__ dst) {
  const Size per_b = Size(rows) * cols;
  const Size total = Size(batch) * per_b;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const Size b = idx / per_b;
  const Size rem = idx % per_b;
  const int r = static_cast<int>(rem / cols);
  const int c = static_cast<int>(rem % cols);
  dst[idx] = src[(b * src_rows + (r0 + r)) * src_cols + (c0 + c)];
}

void launch_pack_rect_batch(int batch, int src_rows, int src_cols, int r0, int c0, int rows,
                            int cols, const double* src, double* dst) {
  const Size total = Size(batch) * Size(rows) * Size(cols);
  if (total == 0) return;
  k_pack_rect_batch<<<grid_for(total), 256>>>(batch, src_rows, src_cols, r0, c0, rows, cols, src,
                                              dst);
}

__global__ void k_ss_vx_reduce(
    int i_loc, int j0, int o_act, int o, int obs, int ext, const double* __restrict__ fg_d,
    const double* __restrict__ fg_x, const double* __restrict__ f2_d,
    const double* __restrict__ f2_x, const double* __restrict__ f0_b,
    const double* __restrict__ ft_b, const double* __restrict__ G_obs_b,
    const double* __restrict__ G_oc_b, const double* __restrict__ G_co_b,
    double* __restrict__ d_vx, double* __restrict__ d_b_ov) {
  __shared__ double svd[256];
  __shared__ double svx[256];
  __shared__ double sxd[256];
  __shared__ double sxx[256];

  const int b = static_cast<int>(blockIdx.x);
  const int j_loc = j0 + b;
  const int c = ext - obs;
  const double* f0 = f0_b + Size(b) * ext * ext;
  const double* ft = ft_b + Size(b) * ext * ext;
  const double* gobs = G_obs_b + Size(b) * obs * obs;
  const double* goc = G_oc_b + Size(b) * o * c;
  const double* gco = G_co_b + Size(b) * c * o;

  double vd = 0.0, vx = 0.0, xd = 0.0, xx = 0.0;
  const Size n_obs2 = Size(obs) * obs;
  for (Size idx = threadIdx.x; idx < n_obs2; idx += blockDim.x) {
    const int p = static_cast<int>(idx / obs);
    const int q = static_cast<int>(idx % obs);
    const double fp = f0[Size(p) * ext + q];
    const double fq = ft[Size(p) * ext + q];
    const double gv = gobs[idx];
    vd += gv * fp;
    vx += gv * fq;
    xd += fp * fp;
    xx += fp * fq;
  }
  const Size n_oc = Size(o) * c;
  for (Size idx = threadIdx.x; idx < n_oc; idx += blockDim.x) {
    const int p = static_cast<int>(idx / c);
    const int q = static_cast<int>(idx % c);
    const double fp_oc = f0[Size(p) * ext + (obs + q)];
    const double fq_oc = ft[Size(p) * ext + (obs + q)];
    const double gv_oc = goc[idx];
    vd += gv_oc * fp_oc;
    vx += gv_oc * fq_oc;
    xd += fp_oc * fp_oc;
    xx += fp_oc * fq_oc;

    const double fp_co = f0[Size(obs + q) * ext + p];
    const double fq_co = ft[Size(obs + q) * ext + p];
    const double gv_co = gco[Size(q) * o + p];
    vd += gv_co * fp_co;
    vx += gv_co * fq_co;
    xd += fp_co * fp_co;
    xx += fp_co * fq_co;
  }

  svd[threadIdx.x] = vd;
  svx[threadIdx.x] = vx;
  sxd[threadIdx.x] = xd;
  sxx[threadIdx.x] = xx;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      svd[threadIdx.x] += svd[threadIdx.x + stride];
      svx[threadIdx.x] += svx[threadIdx.x + stride];
      sxd[threadIdx.x] += sxd[threadIdx.x + stride];
      sxx[threadIdx.x] += sxx[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int pid = i_loc * o_act + j_loc;
    d_vx[Size(b) * 4 + 0] = fg_d[pid] - svd[0];
    d_vx[Size(b) * 4 + 1] = fg_x[pid] - svx[0];
    d_vx[Size(b) * 4 + 2] = f2_d[pid] - sxd[0];
    d_vx[Size(b) * 4 + 3] = f2_x[pid] - sxx[0];
    d_b_ov[Size(b) * 2 + 0] = 0.0;
    d_b_ov[Size(b) * 2 + 1] = 0.0;
  }
}

__global__ void k_accum_dot_pair_rect(int rows, int cols, int ldL, int L_rows, int col_off_L,
                                      int r0, int c0, int ext, double scale,
                                      const double* __restrict__ L_b,
                                      const double* __restrict__ B_dir_b,
                                      const double* __restrict__ B_exc_b,
                                      double* __restrict__ d_b_ov) {
  __shared__ double sd[256];
  __shared__ double sx[256];
  const int b = static_cast<int>(blockIdx.x);
  const double* L = L_b + Size(b) * L_rows * ldL;
  const double* Bd = B_dir_b + Size(b) * ext * ext;
  const double* Bx = B_exc_b + Size(b) * ext * ext;
  const Size total = Size(rows) * cols;

  double ld = 0.0;
  double lx = 0.0;
  for (Size idx = threadIdx.x; idx < total; idx += blockDim.x) {
    const int r = static_cast<int>(idx / cols);
    const int c = static_cast<int>(idx % cols);
    const double lv = L[Size(r) * ldL + (col_off_L + c)];
    const Size boff = Size(r0 + r) * ext + (c0 + c);
    ld += lv * Bd[boff];
    lx += lv * Bx[boff];
  }
  sd[threadIdx.x] = ld;
  sx[threadIdx.x] = lx;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sd[threadIdx.x] += sd[threadIdx.x + stride];
      sx[threadIdx.x] += sx[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    d_b_ov[Size(b) * 2 + 0] += scale * sd[0];
    d_b_ov[Size(b) * 2 + 1] += scale * sx[0];
  }
}

__global__ void k_ss_or_rhf_finalize(
    int i_loc, int j0, int o_act, int o, int f, int obs, int ext, bool compute_both_rhf,
    bool include_ordinary, bool has_occ_weights, Size out_pair_offset, int scalar_f12_idx,
    int scalar_ord_idx, const double* __restrict__ occ_w, const double* __restrict__ F,
    const double* __restrict__ uf_d, const double* __restrict__ uf_x,
    const double* __restrict__ sq_fk_d, const double* __restrict__ sq_fk_x,
    const double* __restrict__ d_vx, const double* __restrict__ d_b_ov,
    const double* __restrict__ G_obs_b, const double* __restrict__ C_d_b,
    double* __restrict__ d_out_scalars, double* __restrict__ d_pair_f12,
    double* __restrict__ d_pair_ord, int* __restrict__ d_err) {
  __shared__ double s0[256];
  __shared__ double s1[256];
  __shared__ double s2[256];
  __shared__ double s3[256];
  __shared__ double s4[256];

  const int b = static_cast<int>(blockIdx.x);
  const int j_loc = j0 + b;
  const int i = f + i_loc;
  const int j = f + j_loc;
  const bool diag = (i_loc == j_loc);
  const int v = obs - o;
  const double occ_sum = F[Size(i) * ext + i] + F[Size(j) * ext + j];
  const double* gobs = G_obs_b + Size(b) * obs * obs;
  const double* cd = C_d_b + Size(b) * v * v;
  const Size vv = Size(v) * v;

  double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0, a4 = 0.0;
  for (Size idx = threadIdx.x; idx < vv; idx += blockDim.x) {
    const int a = static_cast<int>(idx / v);
    const int bv = static_cast<int>(idx % v);
    const double gap = F[Size(o + a) * ext + (o + a)] + F[Size(o + bv) * ext + (o + bv)] - occ_sum;
    if (!isfinite(gap) || gap == 0.0) {
      atomicExch(d_err, -4);
      continue;
    }
    const double inv_g = 1.0 / gap;
    const double gab = gobs[Size(o + a) * obs + (o + bv)];
    const double cab = cd[idx];
    if (diag) {
      a0 += cab * gab * inv_g;
      a1 += cab * cab * inv_g;
      if (include_ordinary) a2 += gab * gab * inv_g;
    } else {
      const double gba = gobs[Size(o + bv) * obs + (o + a)];
      const double cba = cd[Size(bv) * v + a];
      const double ag = gab - gba;
      const double ac = cab - cba;
      a0 += ag * ac * inv_g;
      a1 += ac * ac * inv_g;
      if (include_ordinary) a2 += ag * ag * inv_g;
      if (compute_both_rhf) {
        const double c_os = 0.375 * cab + 0.125 * cba;
        a3 += c_os * (2.0 * gab + c_os) * inv_g;
        if (include_ordinary) a4 += gab * gab * inv_g;
      }
    }
  }
  s0[threadIdx.x] = a0;
  s1[threadIdx.x] = a1;
  s2[threadIdx.x] = a2;
  s3[threadIdx.x] = a3;
  s4[threadIdx.x] = a4;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s0[threadIdx.x] += s0[threadIdx.x + stride];
      s1[threadIdx.x] += s1[threadIdx.x + stride];
      s2[threadIdx.x] += s2[threadIdx.x + stride];
      s3[threadIdx.x] += s3[threadIdx.x + stride];
      s4[threadIdx.x] += s4[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int pid = i_loc * o_act + j_loc;
    const double V_d = d_vx[Size(b) * 4 + 0];
    const double V_x = d_vx[Size(b) * 4 + 1];
    const double X_d = d_vx[Size(b) * 4 + 2];
    const double X_x = d_vx[Size(b) * 4 + 3];
    const double B_d = uf_d[pid] + sq_fk_d[pid] + d_b_ov[Size(b) * 2 + 0];
    const double B_x = uf_x[pid] + sq_fk_x[pid] + d_b_ov[Size(b) * 2 + 1];
    const double wij = has_occ_weights ? 0.5 * (occ_w[i_loc] + occ_w[j_loc]) : 1.0;

    if (compute_both_rhf) {
      const Size n_aa = (o_act >= 2) ? (Size(o_act) * (o_act - 1)) / 2 : 0;
      const Size n_ab = Size(o_act) * o_act;
      const Size off_ab = n_aa;
      const Size off_bb = n_aa + n_ab;
      if (diag) {
        const double ef_os = (V_d - s0[0]) + 0.25 * (B_d - occ_sum * X_d - s1[0]);
        const double eo_os = include_ordinary ? (-s2[0]) : 0.0;
        const Size ab_idx = off_ab + Size(i_loc) * o_act + i_loc;
        if (d_pair_f12) d_pair_f12[ab_idx] = ef_os;
        if (include_ordinary && d_pair_ord) d_pair_ord[ab_idx] = eo_os;
        atomicAdd(&d_out_scalars[1], wij * ef_os);
        if (include_ordinary) atomicAdd(&d_out_scalars[4], wij * eo_os);
      } else {
        const double lin_ss = (V_d - V_x) - 0.5 * s0[0];
        const double quad_ss = ((B_d - B_x) - occ_sum * (X_d - X_x)) - 0.5 * s1[0];
        const double ef_ss = 0.5 * lin_ss + (1.0 / 16.0) * quad_ss;
        const double eo_ss = include_ordinary ? (-0.5 * s2[0]) : 0.0;

        const double lin_os = 0.375 * V_d + 0.125 * V_x;
        const double norm_os = (5.0 / 32.0) * X_d + (3.0 / 32.0) * X_x;
        const double b_os = (5.0 / 32.0) * B_d + (3.0 / 32.0) * B_x;
        const double ef_os = 2.0 * lin_os + b_os - occ_sum * norm_os - s3[0];
        const double eo_os = include_ordinary ? (-s4[0]) : 0.0;

        const Size tri_idx = strict_tri_index(i_loc, j_loc, o_act);
        const Size ab_ij = off_ab + Size(i_loc) * o_act + j_loc;
        const Size ab_ji = off_ab + Size(j_loc) * o_act + i_loc;
        if (d_pair_f12) {
          d_pair_f12[tri_idx] = ef_ss;
          d_pair_f12[ab_ij] = ef_os;
          d_pair_f12[ab_ji] = ef_os;
          d_pair_f12[off_bb + tri_idx] = ef_ss;
        }
        if (include_ordinary && d_pair_ord) {
          d_pair_ord[tri_idx] = eo_ss;
          d_pair_ord[ab_ij] = eo_os;
          d_pair_ord[ab_ji] = eo_os;
          d_pair_ord[off_bb + tri_idx] = eo_ss;
        }
        atomicAdd(&d_out_scalars[0], wij * ef_ss);
        atomicAdd(&d_out_scalars[2], wij * ef_ss);
        atomicAdd(&d_out_scalars[1], 2.0 * wij * ef_os);
        if (include_ordinary) {
          atomicAdd(&d_out_scalars[3], wij * eo_ss);
          atomicAdd(&d_out_scalars[5], wij * eo_ss);
          atomicAdd(&d_out_scalars[4], 2.0 * wij * eo_os);
        }
      }
    } else {
      const double lin_ss = (V_d - V_x) - 0.5 * s0[0];
      const double quad_ss = ((B_d - B_x) - occ_sum * (X_d - X_x)) - 0.5 * s1[0];
      const double ef_ss = 0.5 * lin_ss + (1.0 / 16.0) * quad_ss;
      const double eo_ss = include_ordinary ? (-0.5 * s2[0]) : 0.0;
      const Size idx = out_pair_offset + strict_tri_index(i_loc, j_loc, o_act);
      if (d_pair_f12) d_pair_f12[idx] = ef_ss;
      if (include_ordinary && d_pair_ord) d_pair_ord[idx] = eo_ss;
      atomicAdd(&d_out_scalars[scalar_f12_idx], wij * ef_ss);
      if (include_ordinary) atomicAdd(&d_out_scalars[scalar_ord_idx], wij * eo_ss);
    }
  }
}

__global__ void k_ab_lin_norm_init(int i_loc, int j0, int ob_act, int oa, int ob, int obs, int ext,
                                   const double* __restrict__ gc_ab,
                                   const double* __restrict__ dc_ab,
                                   const double* __restrict__ sq_norm_ab,
                                   const double* __restrict__ sq_fk_ab,
                                   const double* __restrict__ f_b,
                                   const double* __restrict__ G_obs_b,
                                   const double* __restrict__ G_oc_b,
                                   const double* __restrict__ G_co_b,
                                   double* __restrict__ d_lin_norm, double* __restrict__ d_b_acc) {
  __shared__ double slin[256];
  __shared__ double snorm[256];

  const int b = static_cast<int>(blockIdx.x);
  const int j_loc = j0 + b;
  const int c = ext - obs;
  const double* f = f_b + Size(b) * ext * ext;
  const double* gobs = G_obs_b + Size(b) * obs * obs;
  const double* goc = G_oc_b + Size(b) * oa * c;
  const double* gco = G_co_b + Size(b) * c * ob;

  double lin_sub = 0.0;
  double norm_sub = 0.0;
  const Size n_obs2 = Size(obs) * obs;
  for (Size idx = threadIdx.x; idx < n_obs2; idx += blockDim.x) {
    const int p = static_cast<int>(idx / obs);
    const int q = static_cast<int>(idx % obs);
    const double fp = f[Size(p) * ext + q];
    lin_sub += gobs[idx] * fp;
    norm_sub += fp * fp;
  }
  const Size n_oc = Size(oa) * c;
  for (Size idx = threadIdx.x; idx < n_oc; idx += blockDim.x) {
    const int p = static_cast<int>(idx / c);
    const int q = static_cast<int>(idx % c);
    const double fp = f[Size(p) * ext + (obs + q)];
    lin_sub += goc[idx] * fp;
    norm_sub += fp * fp;
  }
  const Size n_co = Size(c) * ob;
  for (Size idx = threadIdx.x; idx < n_co; idx += blockDim.x) {
    const int p = static_cast<int>(idx / ob);
    const int q = static_cast<int>(idx % ob);
    const double fp = f[Size(obs + p) * ext + q];
    lin_sub += gco[idx] * fp;
    norm_sub += fp * fp;
  }

  slin[threadIdx.x] = lin_sub;
  snorm[threadIdx.x] = norm_sub;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      slin[threadIdx.x] += slin[threadIdx.x + stride];
      snorm[threadIdx.x] += snorm[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int pid = i_loc * ob_act + j_loc;
    d_lin_norm[Size(b) * 2 + 0] = gc_ab[pid] - slin[0];
    d_lin_norm[Size(b) * 2 + 1] = sq_norm_ab[pid] - snorm[0];
    d_b_acc[b] = dc_ab[pid] + sq_fk_ab[pid];
  }
}

__global__ void k_accum_dot_single_rect(int rows, int cols, int A_rows, int ldA, int r0_A, int c0_A,
                                        int B_rows, int ldB, int r0_B, int c0_B, double scale,
                                        const double* __restrict__ A_b,
                                        const double* __restrict__ B_b,
                                        double* __restrict__ d_acc) {
  __shared__ double s[256];
  const int b = static_cast<int>(blockIdx.x);
  const double* A = A_b + Size(b) * A_rows * ldA;
  const double* B = B_b + Size(b) * B_rows * ldB;
  const Size total = Size(rows) * cols;

  double acc = 0.0;
  for (Size idx = threadIdx.x; idx < total; idx += blockDim.x) {
    const int r = static_cast<int>(idx / cols);
    const int c = static_cast<int>(idx % cols);
    acc += A[Size(r0_A + r) * ldA + (c0_A + c)] * B[Size(r0_B + r) * ldB + (c0_B + c)];
  }
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) d_acc[b] += scale * s[0];
}

__global__ void k_ab_finalize(
    int i_loc, int j0, int ob_act, int oa, int ob, int fa, int fb, int obs, int ext,
    bool include_ordinary, bool has_occ_weights, Size off_ab, const double* __restrict__ occ_w_a,
    const double* __restrict__ occ_w_b, const double* __restrict__ Fa,
    const double* __restrict__ Fb, const double* __restrict__ d_lin_norm,
    const double* __restrict__ d_b_acc, const double* __restrict__ G_obs_b,
    const double* __restrict__ A2_b, const double* __restrict__ C2_b,
    double* __restrict__ d_out_scalars, double* __restrict__ d_pair_f12,
    double* __restrict__ d_pair_ord, int* __restrict__ d_err) {
  __shared__ double st[256];
  __shared__ double sg[256];

  const int b = static_cast<int>(blockIdx.x);
  const int j_loc = j0 + b;
  const int i = fa + i_loc;
  const int j = fb + j_loc;
  const int va = obs - oa;
  const int vb = obs - ob;
  const double occ_sum = Fa[Size(i) * ext + i] + Fb[Size(j) * ext + j];
  const double* gobs = G_obs_b + Size(b) * obs * obs;
  const double* a2 = A2_b + Size(b) * obs * vb;
  const double* c2 = C2_b + Size(b) * va * obs;
  const Size vab = Size(va) * vb;

  double os_term = 0.0;
  double os_gg = 0.0;
  for (Size idx = threadIdx.x; idx < vab; idx += blockDim.x) {
    const int a = static_cast<int>(idx / vb);
    const int bv = static_cast<int>(idx % vb);
    const double gap = Fa[Size(oa + a) * ext + (oa + a)] + Fb[Size(ob + bv) * ext + (ob + bv)] - occ_sum;
    if (!isfinite(gap) || gap == 0.0) {
      atomicExch(d_err, -4);
      continue;
    }
    const double inv_g = 1.0 / gap;
    const double cab = a2[Size(oa + a) * vb + bv] + c2[Size(a) * obs + (ob + bv)];
    const double gab = gobs[Size(oa + a) * obs + (ob + bv)];
    os_term += cab * (2.0 * gab + cab) * inv_g;
    if (include_ordinary) os_gg += gab * gab * inv_g;
  }
  st[threadIdx.x] = os_term;
  sg[threadIdx.x] = os_gg;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      st[threadIdx.x] += st[threadIdx.x + stride];
      sg[threadIdx.x] += sg[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double lin = d_lin_norm[Size(b) * 2 + 0];
    const double norm = d_lin_norm[Size(b) * 2 + 1];
    const double b_val = d_b_acc[b];
    const double ef = 2.0 * lin + b_val - occ_sum * norm - st[0];
    const double eo = include_ordinary ? (-sg[0]) : 0.0;
    const Size idx = off_ab + Size(i_loc) * ob_act + j_loc;
    if (d_pair_f12) d_pair_f12[idx] = ef;
    if (include_ordinary && d_pair_ord) d_pair_ord[idx] = eo;
    const double wij = has_occ_weights ? 0.5 * (occ_w_a[i_loc] + occ_w_b[j_loc]) : 1.0;
    atomicAdd(&d_out_scalars[1], wij * ef);
    if (include_ordinary) atomicAdd(&d_out_scalars[4], wij * eo);
  }
}

int build_ss_stage1_scalars_gpu(
    cublasHandle_t h, int o, int f, int ext, int aux, const double* d_fit_qoe,
    const double* d_H, const double* raw_gc, const double* raw_dc, const double* raw_sq,
    const double* metric_gc, const double* metric_dc, const double* metric_sq,
    DeviceBuf& d_qoe_scratch, DeviceBuf& d_metric_buf, DeviceBuf& d_fg_d, DeviceBuf& d_fg_x,
    DeviceBuf& d_uf_d, DeviceBuf& d_uf_x, DeviceBuf& d_f2_d, DeviceBuf& d_f2_x,
    DeviceBuf& d_sq_fk_d, DeviceBuf& d_sq_fk_x) {
  const int o_act = o - f;
  const Size qoe_n = Size(aux) * Size(o_act) * ext;
  const Size qq = Size(aux) * aux;
  const Size qoo = Size(aux) * Size(o_act) * o;
  const Size oo_act = Size(o_act) * o_act;

  DeviceBuf d_fit_oqo, d_raw_oqo, d_res_oqo;
  if (d_fit_oqo.alloc(qoo) != cudaSuccess || d_raw_oqo.alloc(qoo) != cudaSuccess ||
      d_res_oqo.alloc(qoo) != cudaSuccess || d_fg_d.alloc(oo_act) != cudaSuccess ||
      d_fg_x.alloc(oo_act) != cudaSuccess || d_uf_d.alloc(oo_act) != cudaSuccess ||
      d_uf_x.alloc(oo_act) != cudaSuccess || d_f2_d.alloc(oo_act) != cudaSuccess ||
      d_f2_x.alloc(oo_act) != cudaSuccess || d_sq_fk_d.alloc(oo_act) != cudaSuccess ||
      d_sq_fk_x.alloc(oo_act) != cudaSuccess) {
    return 2;
  }
  k_extract_qoe_occ<<<grid_for(qoo), 256>>>(aux, o_act, ext, o, d_fit_qoe, d_fit_oqo.ptr);

  std::vector<double> h_qoe;
  auto run_op = [&](const double* raw_src, const double* metric_src, double* out_d, double* out_x,
                    bool do_sq_fk) -> int {
    if (!pack_active_qoe_checked_cuda_host(raw_src, aux, o, f, o_act, ext, h_qoe)) return -2;
    if (cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_n * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_metric_buf.ptr, metric_src, qq * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
      return 3;
    }
    k_extract_qoe_occ<<<grid_for(qoo), 256>>>(aux, o_act, ext, o, d_qoe_scratch.ptr, d_raw_oqo.ptr);

    DeviceBuf d_fit_H, d_raw_H, d_res_H;
    if (do_sq_fk) {
      if (d_fit_H.alloc(qoo) != cudaSuccess || d_raw_H.alloc(qoo) != cudaSuccess ||
          d_res_H.alloc(qoo) != cudaSuccess) {
        return 2;
      }
      if (rm_gemm(h, false, true, aux * o_act, o, ext, 1.0, d_fit_qoe, ext, d_H, ext, 0.0,
                  d_fit_H.ptr, o) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, true, aux * o_act, o, ext, 1.0, d_qoe_scratch.ptr, ext, d_H, ext, 0.0,
                  d_raw_H.ptr, o) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
    }

    if (rm_gemm(h, false, false, aux, o_act * ext, aux, -1.0, d_metric_buf.ptr, aux, d_fit_qoe,
                o_act * ext, 1.0, d_qoe_scratch.ptr, o_act * ext) != CUBLAS_STATUS_SUCCESS) {
      return 4;
    }
    k_extract_qoe_occ<<<grid_for(qoo), 256>>>(aux, o_act, ext, o, d_qoe_scratch.ptr, d_res_oqo.ptr);
    k_ss_robust_diag_elems<<<static_cast<unsigned>(oo_act), 256>>>(
        aux, o_act, o, f, d_fit_oqo.ptr, d_raw_oqo.ptr, d_res_oqo.ptr, out_d, out_x);

    if (do_sq_fk) {
      if (rm_gemm(h, false, true, aux * o_act, o, ext, 1.0, d_qoe_scratch.ptr, ext, d_H, ext, 0.0,
                  d_res_H.ptr, o) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_ss_sq_fk<<<static_cast<unsigned>(oo_act), 256>>>(
          aux, o_act, o, f, d_fit_oqo.ptr, d_raw_oqo.ptr, d_res_oqo.ptr, d_fit_H.ptr, d_raw_H.ptr,
          d_res_H.ptr, d_sq_fk_d.ptr, d_sq_fk_x.ptr);
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : 7;
  };

  if (int rc = run_op(raw_gc, metric_gc, d_fg_d.ptr, d_fg_x.ptr, false); rc != 0) return rc;
  if (int rc = run_op(raw_dc, metric_dc, d_uf_d.ptr, d_uf_x.ptr, false); rc != 0) return rc;
  if (int rc = run_op(raw_sq, metric_sq, d_f2_d.ptr, d_f2_x.ptr, true); rc != 0) return rc;
  return 0;
}

int run_ss_or_rhf_stage2_gpu(
    cublasHandle_t h, bool compute_both_rhf, bool include_ordinary, bool has_occ_weights, int o,
    int f, int obs, int ext, int aux, Size ws_cap, Size out_pair_offset, int scalar_f12_idx,
    int scalar_ord_idx, const double* d_occ_w, const double* d_fit_oqe,
    const double* d_raw_coul_oqe, const double* d_raw_g_oqe, const double* d_res_g_oqe,
    const double* d_F, const double* d_K, const double* d_fg_d, const double* d_fg_x,
    const double* d_f2_d, const double* d_f2_x, const double* d_uf_d, const double* d_uf_x,
    const double* d_sq_fk_d, const double* d_sq_fk_x, double* d_out_scalars, double* d_pair_f12,
    double* d_pair_ord, int* d_err) {
  const int o_act = o - f;
  if (compute_both_rhf ? (o_act <= 0) : (o_act < 2)) return 0;
  const int v = obs - o;
  const int c = ext - obs;
  const Size ee = Size(ext) * ext;
  const Size qe = Size(aux) * ext;
  const Size obs2 = Size(obs) * obs;
  const Size oc = Size(o) * c;
  const Size vv = Size(v) * v;

  const auto batch = ss_batch_workspace(o, obs, ext);
  const Size sub_a_per = batch.sub_a, sub_l_per = batch.sub_b;
  const Size bytes_per_pair = bytes_of_doubles(batch.elements);
  const Size remaining = DeviceWorkspace::current->available;
  if (bytes_per_pair > remaining) return kErrBudgetExceeded;
  const Size available = std::max(bytes_per_pair, std::min(ws_cap, remaining));
  int b_max = static_cast<int>(std::min<Size>(o_act, available / bytes_per_pair));
  b_max = std::min<int>(b_max, std::max(1, static_cast<int>((Size(INT_MAX) / 4) / ext)));

  // Allocate ALL pair-batch workspace ONCE before the pair loop (zero cudaMalloc/cudaFree inside).
  DeviceBuf d_f0, d_ft, d_L0, d_G_obs, d_G_oc, d_G_co, d_C_d, d_sub_A, d_sub_L, d_vx, d_b_ov;
  if (d_f0.alloc(Size(b_max) * ee) != cudaSuccess || d_ft.alloc(Size(b_max) * ee) != cudaSuccess ||
      d_L0.alloc(Size(b_max) * ee) != cudaSuccess ||
      d_G_obs.alloc(Size(b_max) * obs2) != cudaSuccess ||
      d_G_oc.alloc(Size(b_max) * oc) != cudaSuccess ||
      d_G_co.alloc(Size(b_max) * oc) != cudaSuccess ||
      d_C_d.alloc(Size(b_max) * vv) != cudaSuccess ||
      d_sub_A.alloc(Size(b_max) * sub_a_per) != cudaSuccess ||
      d_sub_L.alloc(Size(b_max) * sub_l_per) != cudaSuccess ||
      d_vx.alloc(Size(b_max) * 4) != cudaSuccess || d_b_ov.alloc(Size(b_max) * 2) != cudaSuccess) {
    return 2;
  }

  for (int i_loc = 0; i_loc < o_act; ++i_loc) {
    const int j_start = compute_both_rhf ? i_loc : (i_loc + 1);
    const double* fit_i = d_fit_oqe + Size(i_loc) * qe;
    const double* raw_g_i = d_raw_g_oqe + Size(i_loc) * qe;

    for (int j0 = j_start; j0 < o_act; j0 += b_max) {
      const int bj = std::min(b_max, o_act - j0);
      const double* fit_j0 = d_fit_oqe + Size(j0) * qe;
      const double* raw_coul_j0 = d_raw_coul_oqe + Size(j0) * qe;
      const double* res_g_j0 = d_res_g_oqe + Size(j0) * qe;

      // 1. Batched f0[b] = fit_i^T @ res_g_j + raw_g_i^T @ fit_j & batched transpose -> ft[b].
      if (rm_gemm_batched(h, true, false, ext, ext, aux, 1.0, fit_i, ext, 0, res_g_j0, ext,
                          static_cast<long long>(qe), 0.0, d_f0.ptr, ext,
                          static_cast<long long>(ee), bj) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm_batched(h, true, false, ext, ext, aux, 1.0, raw_g_i, ext, 0, fit_j0, ext,
                          static_cast<long long>(qe), 1.0, d_f0.ptr, ext,
                          static_cast<long long>(ee), bj) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      launch_batched_transpose(bj, ext, d_f0.ptr, d_ft.ptr);

      // 2. Batched Coulomb blocks G_obs, G_oc, G_co.
      if (rm_gemm_batched(h, true, false, obs, obs, aux, 1.0, fit_i, ext, 0, raw_coul_j0, ext,
                          static_cast<long long>(qe), 0.0, d_G_obs.ptr, obs,
                          static_cast<long long>(obs2), bj) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm_batched(h, true, false, o, c, aux, 1.0, fit_i, ext, 0, raw_coul_j0 + obs, ext,
                          static_cast<long long>(qe), 0.0, d_G_oc.ptr, c,
                          static_cast<long long>(oc), bj) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm_batched(h, true, false, c, o, aux, 1.0, fit_i + obs, ext, 0, raw_coul_j0, ext,
                          static_cast<long long>(qe), 0.0, d_G_co.ptr, o,
                          static_cast<long long>(oc), bj) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }

      // 3. Fused V_d, V_x, X_d, X_x reduction (also zeros d_b_ov).
      k_ss_vx_reduce<<<static_cast<unsigned>(bj), 256>>>(
          i_loc, j0, o_act, o, obs, ext, d_fg_d, d_fg_x, d_f2_d, d_f2_x, d_f0.ptr, d_ft.ptr,
          d_G_obs.ptr, d_G_oc.ptr, d_G_co.ptr, d_vx.ptr, d_b_ov.ptr);

      // 4. Batched b_overlap for A = f0 (pass 0) and A = ft (pass 1).
      for (int pass = 0; pass < 2; ++pass) {
        const double* A_b = (pass == 0) ? d_f0.ptr : d_ft.ptr;
        const double* B_dir = (pass == 0) ? d_f0.ptr : d_ft.ptr;
        const double* B_exc = (pass == 0) ? d_ft.ptr : d_f0.ptr;

        // Term 0: L0 = A @ K (-1)
        if (rm_gemm(h, false, false, bj * ext, ext, ext, 1.0, A_b, ext, d_K, ext, 0.0, d_L0.ptr,
                    ext) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
        k_accum_dot_pair_rect<<<static_cast<unsigned>(bj), 256>>>(
            ext, ext, ext, ext, 0, 0, 0, ext, -1.0, d_L0.ptr, B_dir, B_exc, d_b_ov.ptr);

        // Term 1: L1 = A[0:o, 0:ext] @ F (-1)
        launch_pack_rect_batch(bj, ext, ext, 0, 0, o, ext, A_b, d_sub_A.ptr);
        if (rm_gemm(h, false, false, bj * o, ext, ext, 1.0, d_sub_A.ptr, ext, d_F, ext, 0.0,
                    d_sub_L.ptr, ext) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
        k_accum_dot_pair_rect<<<static_cast<unsigned>(bj), 256>>>(
            o, ext, ext, o, 0, 0, 0, ext, -1.0, d_sub_L.ptr, B_dir, B_exc, d_b_ov.ptr);

        // Term 2: L2 = A[obs:ext, 0:o] @ F[0:o, 0:o] (+1)
        launch_pack_rect_batch(bj, ext, ext, obs, 0, c, o, A_b, d_sub_A.ptr);
        if (rm_gemm(h, false, false, bj * c, o, o, 1.0, d_sub_A.ptr, o, d_F, ext, 0.0, d_sub_L.ptr,
                    o) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
        k_accum_dot_pair_rect<<<static_cast<unsigned>(bj), 256>>>(
            c, o, o, c, 0, obs, 0, ext, 1.0, d_sub_L.ptr, B_dir, B_exc, d_b_ov.ptr);

        // Terms 3 & 5: L35 = A[o:obs, 0:obs] @ F[0:obs, 0:ext] (-1 on 0:obs, -2 on obs:ext)
        launch_pack_rect_batch(bj, ext, ext, o, 0, v, obs, A_b, d_sub_A.ptr);
        if (rm_gemm(h, false, false, bj * v, ext, obs, 1.0, d_sub_A.ptr, obs, d_F, ext, 0.0,
                    d_sub_L.ptr, ext) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
        k_accum_dot_pair_rect<<<static_cast<unsigned>(bj), 256>>>(
            v, obs, ext, v, 0, o, 0, ext, -1.0, d_sub_L.ptr, B_dir, B_exc, d_b_ov.ptr);
        k_accum_dot_pair_rect<<<static_cast<unsigned>(bj), 256>>>(
            v, c, ext, v, obs, o, obs, ext, -2.0, d_sub_L.ptr, B_dir, B_exc, d_b_ov.ptr);

        // Term 4: L4 = A[obs:ext, 0:ext] @ F[0:o, 0:ext]^T (-2)
        launch_pack_rect_batch(bj, ext, ext, obs, 0, c, ext, A_b, d_sub_A.ptr);
        if (rm_gemm(h, false, true, bj * c, o, ext, 1.0, d_sub_A.ptr, ext, d_F, ext, 0.0,
                    d_sub_L.ptr, o) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
        k_accum_dot_pair_rect<<<static_cast<unsigned>(bj), 256>>>(
            c, o, o, c, 0, obs, 0, ext, -2.0, d_sub_L.ptr, B_dir, B_exc, d_b_ov.ptr);
      }

      // 5. Batched coupling C_d = f0[vir, comp] @ F[vir, comp]^T + F[vir, comp] @ f0[comp, vir].
      launch_pack_rect_batch(bj, ext, ext, o, obs, v, c, d_f0.ptr, d_sub_A.ptr);
      if (rm_gemm(h, false, true, bj * v, v, c, 1.0, d_sub_A.ptr, c, d_F + Size(o) * ext + obs, ext,
                  0.0, d_C_d.ptr, v) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      launch_pack_rect_batch(bj, ext, ext, obs, o, c, v, d_f0.ptr, d_sub_A.ptr);
      if (rm_gemm_batched(h, false, false, v, v, c, 1.0, d_F + Size(o) * ext + obs, ext, 0,
                          d_sub_A.ptr, v, static_cast<long long>(Size(c) * v), 1.0, d_C_d.ptr, v,
                          static_cast<long long>(vv), bj) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }

      // 6. Finalize pair energies and weighted spin totals.
      k_ss_or_rhf_finalize<<<static_cast<unsigned>(bj), 256>>>(
          i_loc, j0, o_act, o, f, obs, ext, compute_both_rhf, include_ordinary, has_occ_weights,
          out_pair_offset, scalar_f12_idx, scalar_ord_idx, d_occ_w, d_F, d_uf_d, d_uf_x, d_sq_fk_d,
          d_sq_fk_x, d_vx.ptr, d_b_ov.ptr, d_G_obs.ptr, d_C_d.ptr, d_out_scalars, d_pair_f12,
          d_pair_ord, d_err);
    }
  }
  return (cudaGetLastError() == cudaSuccess) ? 0 : 7;
}

int build_ab_stage1_scalars_gpu(
    cublasHandle_t h, int oa, int ob, int fa, int fb, int ext, int aux, const double* d_fit_a_qoe,
    const double* d_fit_b_qoe, const double* d_Fa, const double* d_Fb, const double* d_Ka,
    const double* d_Kb, const double* d_Sab, const double* raw_gc_a, const double* raw_gc_b,
    const double* raw_dc_a, const double* raw_dc_b, const double* raw_sq_a, const double* raw_sq_b,
    const double* metric_gc, const double* metric_dc, const double* metric_sq,
    DeviceBuf& d_qoe_scratch, DeviceBuf& d_metric_buf, DeviceBuf& d_gc_ab, DeviceBuf& d_dc_ab,
    DeviceBuf& d_sq_norm_ab, DeviceBuf& d_sq_fk_ab) {
  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  const Size ee = Size(ext) * ext;
  const Size qq = Size(aux) * aux;
  const Size ab_pairs = Size(oa_act) * ob_act;
  const int n_vec_L = 2 * ob_act + 3 * oa_act;
  const int n_vec_R = 2 * oa_act + 3 * ob_act;

  // Build P_L and P_R inside a scoped block so helper (ext x ext) matrices are freed immediately.
  DeviceBuf d_P_L, d_P_R;
  if (d_P_L.alloc(Size(n_vec_L) * ext) != cudaSuccess ||
      d_P_R.alloc(Size(n_vec_R) * ext) != cudaSuccess) {
    return 2;
  }
  {
    DeviceBuf d_Ha, d_Hb, d_SS_T, d_ST_S, d_Ha_S, d_Hb_ST, d_ST_Ha, d_S_Hb, d_ST_Ha_S, d_S_Hb_ST;
    if (d_Ha.alloc(ee) != cudaSuccess || d_Hb.alloc(ee) != cudaSuccess ||
        d_SS_T.alloc(ee) != cudaSuccess || d_ST_S.alloc(ee) != cudaSuccess ||
        d_Ha_S.alloc(ee) != cudaSuccess || d_Hb_ST.alloc(ee) != cudaSuccess ||
        d_ST_Ha.alloc(ee) != cudaSuccess || d_S_Hb.alloc(ee) != cudaSuccess ||
        d_ST_Ha_S.alloc(ee) != cudaSuccess || d_S_Hb_ST.alloc(ee) != cudaSuccess) {
      return 2;
    }
    k_add_2d<<<grid_for(ee), 256>>>(ee, d_Fa, d_Ka, d_Ha.ptr);
    k_add_2d<<<grid_for(ee), 256>>>(ee, d_Fb, d_Kb, d_Hb.ptr);
    if (rm_gemm(h, false, true, ext, ext, ext, 1.0, d_Sab, ext, d_Sab, ext, 0.0, d_SS_T.ptr, ext) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, true, false, ext, ext, ext, 1.0, d_Sab, ext, d_Sab, ext, 0.0, d_ST_S.ptr, ext) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, false, ext, ext, ext, 1.0, d_Ha.ptr, ext, d_Sab, ext, 0.0, d_Ha_S.ptr,
                ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, true, ext, ext, ext, 1.0, d_Hb.ptr, ext, d_Sab, ext, 0.0, d_Hb_ST.ptr,
                ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, true, false, ext, ext, ext, 1.0, d_Sab, ext, d_Ha.ptr, ext, 0.0, d_ST_Ha.ptr,
                ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, false, ext, ext, ext, 1.0, d_Sab, ext, d_Hb.ptr, ext, 0.0, d_S_Hb.ptr,
                ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, true, false, ext, ext, ext, 1.0, d_Sab, ext, d_Ha_S.ptr, ext, 0.0, d_ST_Ha_S.ptr,
                ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, false, ext, ext, ext, 1.0, d_Sab, ext, d_Hb_ST.ptr, ext, 0.0,
                d_S_Hb_ST.ptr, ext) != CUBLAS_STATUS_SUCCESS) {
      return 4;
    }
    k_build_ab_proj_L<<<grid_for(Size(n_vec_L) * ext), 256>>>(
        oa_act, ob_act, fa, fb, ext, d_Sab, d_SS_T.ptr, d_Ha.ptr, d_Ha_S.ptr, d_S_Hb.ptr,
        d_S_Hb_ST.ptr, d_P_L.ptr);
    k_build_ab_proj_R<<<grid_for(Size(n_vec_R) * ext), 256>>>(
        oa_act, ob_act, fa, fb, ext, d_Sab, d_ST_S.ptr, d_Hb.ptr, d_ST_Ha.ptr, d_Hb_ST.ptr,
        d_ST_Ha_S.ptr, d_P_R.ptr);
    if (cudaGetLastError() != cudaSuccess) return 7;
  }

  DeviceBuf d_fa_diag, d_ra_diag, d_fb_diag, d_sb_diag;
  DeviceBuf d_fa_PL, d_ra_PL, d_fb_PR, d_sb_PR;
  if (d_fa_diag.alloc(Size(aux) * oa_act) != cudaSuccess ||
      d_ra_diag.alloc(Size(aux) * oa_act) != cudaSuccess ||
      d_fb_diag.alloc(Size(aux) * ob_act) != cudaSuccess ||
      d_sb_diag.alloc(Size(aux) * ob_act) != cudaSuccess ||
      d_fa_PL.alloc(Size(aux) * oa_act * n_vec_L) != cudaSuccess ||
      d_ra_PL.alloc(Size(aux) * oa_act * n_vec_L) != cudaSuccess ||
      d_fb_PR.alloc(Size(aux) * ob_act * n_vec_R) != cudaSuccess ||
      d_sb_PR.alloc(Size(aux) * ob_act * n_vec_R) != cudaSuccess ||
      d_gc_ab.alloc(ab_pairs) != cudaSuccess || d_dc_ab.alloc(ab_pairs) != cudaSuccess ||
      d_sq_norm_ab.alloc(ab_pairs) != cudaSuccess || d_sq_fk_ab.alloc(ab_pairs) != cudaSuccess) {
    return 2;
  }

  k_extract_qoe_diag<<<grid_for(Size(aux) * oa_act), 256>>>(aux, oa_act, ext, fa, d_fit_a_qoe,
                                                            d_fa_diag.ptr);
  k_extract_qoe_diag<<<grid_for(Size(aux) * ob_act), 256>>>(aux, ob_act, ext, fb, d_fit_b_qoe,
                                                            d_fb_diag.ptr);
  if (rm_gemm(h, false, true, aux * oa_act, n_vec_L, ext, 1.0, d_fit_a_qoe, ext, d_P_L.ptr, ext,
              0.0, d_fa_PL.ptr, n_vec_L) != CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, false, true, aux * ob_act, n_vec_R, ext, 1.0, d_fit_b_qoe, ext, d_P_R.ptr, ext,
              0.0, d_fb_PR.ptr, n_vec_R) != CUBLAS_STATUS_SUCCESS) {
    return 4;
  }

  std::vector<double> h_qoe;
  auto run_ab_op = [&](int op_mode, const double* raw_a, const double* raw_b,
                       const double* metric_src, double* out1, double* out2) -> int {
    if (!pack_active_qoe_checked_cuda_host(raw_a, aux, oa, fa, oa_act, ext, h_qoe)) return -2;
    if (cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), h_qoe.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return 3;
    }
    k_extract_qoe_diag<<<grid_for(Size(aux) * oa_act), 256>>>(aux, oa_act, ext, fa,
                                                              d_qoe_scratch.ptr, d_ra_diag.ptr);
    if (rm_gemm(h, false, true, aux * oa_act, n_vec_L, ext, 1.0, d_qoe_scratch.ptr, ext, d_P_L.ptr,
                ext, 0.0, d_ra_PL.ptr, n_vec_L) != CUBLAS_STATUS_SUCCESS) {
      return 4;
    }

    if (!pack_active_qoe_checked_cuda_host(raw_b, aux, ob, fb, ob_act, ext, h_qoe)) return -2;
    if (cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), h_qoe.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_metric_buf.ptr, metric_src, qq * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
      return 3;
    }
    if (rm_gemm(h, false, false, aux, ob_act * ext, aux, -1.0, d_metric_buf.ptr, aux, d_fit_b_qoe,
                ob_act * ext, 1.0, d_qoe_scratch.ptr, ob_act * ext) != CUBLAS_STATUS_SUCCESS) {
      return 4;
    }
    k_extract_qoe_diag<<<grid_for(Size(aux) * ob_act), 256>>>(aux, ob_act, ext, fb,
                                                              d_qoe_scratch.ptr, d_sb_diag.ptr);
    if (rm_gemm(h, false, true, aux * ob_act, n_vec_R, ext, 1.0, d_qoe_scratch.ptr, ext, d_P_R.ptr,
                ext, 0.0, d_sb_PR.ptr, n_vec_R) != CUBLAS_STATUS_SUCCESS) {
      return 4;
    }

    k_ab_stage1_scalars<<<static_cast<unsigned>(ab_pairs), 256>>>(
        op_mode, aux, oa_act, ob_act, d_fa_diag.ptr, d_ra_diag.ptr, d_fb_diag.ptr, d_sb_diag.ptr,
        d_fa_PL.ptr, d_ra_PL.ptr, d_fb_PR.ptr, d_sb_PR.ptr, out1, out2);
    return (cudaGetLastError() == cudaSuccess) ? 0 : 7;
  };

  if (int rc = run_ab_op(0, raw_gc_a, raw_gc_b, metric_gc, d_gc_ab.ptr, nullptr); rc != 0) return rc;
  if (int rc = run_ab_op(1, raw_dc_a, raw_dc_b, metric_dc, d_dc_ab.ptr, nullptr); rc != 0) return rc;
  if (int rc = run_ab_op(2, raw_sq_a, raw_sq_b, metric_sq, d_sq_norm_ab.ptr, d_sq_fk_ab.ptr);
      rc != 0) {
    return rc;
  }
  return 0;
}

int run_ab_stage2_gpu(
    cublasHandle_t h, bool include_ordinary, bool has_occ_weights, int oa, int ob, int fa, int fb,
    int obs, int ext, int aux, Size ws_cap, Size off_ab, const double* d_occ_w_a,
    const double* d_occ_w_b, const double* d_fit_a_oqe, const double* d_raw_g_a_oqe,
    const double* d_fit_b_oqe, const double* d_raw_coul_b_oqe, const double* d_res_g_b_oqe,
    const double* d_Fa, const double* d_Fb, const double* d_Ka, const double* d_Kb,
    const double* d_Sab, const double* d_gc_ab, const double* d_dc_ab, const double* d_sq_norm_ab,
    const double* d_sq_fk_ab, double* d_out_scalars, double* d_pair_f12, double* d_pair_ord,
    int* d_err) {
  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  if (oa_act <= 0 || ob_act <= 0) return 0;
  const int va = obs - oa;
  const int vb = obs - ob;
  const int c = ext - obs;
  const Size ee = Size(ext) * ext;
  const Size qe = Size(aux) * ext;
  const Size obs2 = Size(obs) * obs;
  const Size oac = Size(oa) * c;
  const Size cob = Size(c) * ob;
  const Size obs_vb = Size(obs) * vb;
  const Size va_obs = Size(va) * obs;

  const auto batch = ab_batch_workspace(oa, ob, obs, ext);
  const Size sub_a_per = batch.sub_a, sub_b_per = batch.sub_b;
  const Size bytes_per_pair = bytes_of_doubles(batch.elements);
  const Size remaining = DeviceWorkspace::current->available;
  if (bytes_per_pair > remaining) return kErrBudgetExceeded;
  const Size available = std::max(bytes_per_pair, std::min(ws_cap, remaining));
  int b_max = static_cast<int>(std::min<Size>(ob_act, available / bytes_per_pair));
  b_max = std::min<int>(b_max, std::max(1, static_cast<int>((Size(INT_MAX) / 4) / ext)));

  // Allocate ALL pair-batch workspace ONCE before the pair loop (zero cudaMalloc/cudaFree inside).
  DeviceBuf d_f, d_tmp1, d_G_obs, d_G_oc, d_G_co, d_A2, d_C2, d_sub_A, d_sub_B, d_lin_norm, d_b_acc;
  if (d_f.alloc(Size(b_max) * ee) != cudaSuccess ||
      d_tmp1.alloc(Size(b_max) * ee) != cudaSuccess ||
      d_G_obs.alloc(Size(b_max) * obs2) != cudaSuccess ||
      d_G_oc.alloc(Size(b_max) * oac) != cudaSuccess ||
      d_G_co.alloc(Size(b_max) * cob) != cudaSuccess ||
      d_A2.alloc(Size(b_max) * obs_vb) != cudaSuccess ||
      d_C2.alloc(Size(b_max) * va_obs) != cudaSuccess ||
      d_sub_A.alloc(Size(b_max) * sub_a_per) != cudaSuccess ||
      d_sub_B.alloc(Size(b_max) * sub_b_per) != cudaSuccess ||
      d_lin_norm.alloc(Size(b_max) * 2) != cudaSuccess ||
      d_b_acc.alloc(Size(b_max)) != cudaSuccess) {
    return 2;
  }

  for (int i_loc = 0; i_loc < oa_act; ++i_loc) {
    const double* fit_a_i = d_fit_a_oqe + Size(i_loc) * qe;
    const double* raw_g_a_i = d_raw_g_a_oqe + Size(i_loc) * qe;

    for (int j0 = 0; j0 < ob_act; j0 += b_max) {
      const int bj = std::min(b_max, ob_act - j0);
      const double* fit_b_j0 = d_fit_b_oqe + Size(j0) * qe;
      const double* raw_coul_b_j0 = d_raw_coul_b_oqe + Size(j0) * qe;
      const double* res_g_b_j0 = d_res_g_b_oqe + Size(j0) * qe;

      // 1. Batched f0 -> f = (3/8) f0 + (1/8) (Sab @ f0^T) @ Sab in-place in d_f.
      if (rm_gemm_batched(h, true, false, ext, ext, aux, 1.0, fit_a_i, ext, 0, res_g_b_j0, ext,
                          static_cast<long long>(qe), 0.0, d_f.ptr, ext,
                          static_cast<long long>(ee), bj) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm_batched(h, true, false, ext, ext, aux, 1.0, raw_g_a_i, ext, 0, fit_b_j0, ext,
                          static_cast<long long>(qe), 1.0, d_f.ptr, ext,
                          static_cast<long long>(ee), bj) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm_batched(h, false, true, ext, ext, ext, 1.0, d_Sab, ext, 0, d_f.ptr, ext,
                          static_cast<long long>(ee), 0.0, d_tmp1.ptr, ext,
                          static_cast<long long>(ee), bj) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, bj * ext, ext, ext, 0.125, d_tmp1.ptr, ext, d_Sab, ext, 0.375,
                  d_f.ptr, ext) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }

      // 2. Batched Coulomb blocks G_obs, G_oc, G_co.
      if (rm_gemm_batched(h, true, false, obs, obs, aux, 1.0, fit_a_i, ext, 0, raw_coul_b_j0, ext,
                          static_cast<long long>(qe), 0.0, d_G_obs.ptr, obs,
                          static_cast<long long>(obs2), bj) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm_batched(h, true, false, oa, c, aux, 1.0, fit_a_i, ext, 0, raw_coul_b_j0 + obs,
                          ext, static_cast<long long>(qe), 0.0, d_G_oc.ptr, c,
                          static_cast<long long>(oac), bj) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm_batched(h, true, false, c, ob, aux, 1.0, fit_a_i + obs, ext, 0, raw_coul_b_j0,
                          ext, static_cast<long long>(qe), 0.0, d_G_co.ptr, ob,
                          static_cast<long long>(cob), bj) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }

      // 3. Fused linear & norm reduction + initialize d_b_acc = dc + sq_fk.
      k_ab_lin_norm_init<<<static_cast<unsigned>(bj), 256>>>(
          i_loc, j0, ob_act, oa, ob, obs, ext, d_gc_ab, d_dc_ab, d_sq_norm_ab, d_sq_fk_ab, d_f.ptr,
          d_G_obs.ptr, d_G_oc.ptr, d_G_co.ptr, d_lin_norm.ptr, d_b_acc.ptr);

      // 4. Exchange terms: -<f, Ka @ f> - <f, f @ Kb^T>.
      if (rm_gemm_batched(h, false, false, ext, ext, ext, 1.0, d_Ka, ext, 0, d_f.ptr, ext,
                          static_cast<long long>(ee), 0.0, d_tmp1.ptr, ext,
                          static_cast<long long>(ee), bj) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_accum_dot_single_rect<<<static_cast<unsigned>(bj), 256>>>(
          ext, ext, ext, ext, 0, 0, ext, ext, 0, 0, -1.0, d_f.ptr, d_tmp1.ptr, d_b_acc.ptr);
      if (rm_gemm(h, false, true, bj * ext, ext, ext, 1.0, d_f.ptr, ext, d_Kb, ext, 0.0, d_tmp1.ptr,
                  ext) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_accum_dot_single_rect<<<static_cast<unsigned>(bj), 256>>>(
          ext, ext, ext, ext, 0, 0, ext, ext, 0, 0, -1.0, d_f.ptr, d_tmp1.ptr, d_b_acc.ptr);

      // 5. Alpha complement-block cancellation <qf, Fa qf> - <f, Fa f>.
      launch_pack_rect_batch(bj, ext, ext, 0, 0, ext, ob, d_f.ptr, d_sub_A.ptr);
      if (rm_gemm_batched(h, false, false, ext, ob, ext, 1.0, d_Fa, ext, 0, d_sub_A.ptr, ob,
                          static_cast<long long>(Size(ext) * ob), 0.0, d_sub_B.ptr, ob,
                          static_cast<long long>(Size(ext) * ob), bj) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_accum_dot_single_rect<<<static_cast<unsigned>(bj), 256>>>(
          ext, ob, ext, ob, 0, 0, ext, ob, 0, 0, -1.0, d_sub_A.ptr, d_sub_B.ptr, d_b_acc.ptr);

      launch_pack_rect_batch(bj, ext, ext, 0, ob, obs, vb, d_f.ptr, d_sub_A.ptr);
      if (rm_gemm_batched(h, false, false, obs, vb, obs, 1.0, d_Fa, ext, 0, d_sub_A.ptr, vb,
                          static_cast<long long>(obs_vb), 0.0, d_sub_B.ptr, vb,
                          static_cast<long long>(obs_vb), bj) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_accum_dot_single_rect<<<static_cast<unsigned>(bj), 256>>>(
          obs, vb, obs, vb, 0, 0, obs, vb, 0, 0, -1.0, d_sub_A.ptr, d_sub_B.ptr, d_b_acc.ptr);

      if (rm_gemm_batched(h, false, false, c, vb, obs, 1.0, d_Fa + Size(obs) * ext, ext, 0,
                          d_sub_A.ptr, vb, static_cast<long long>(obs_vb), 0.0, d_sub_B.ptr, vb,
                          static_cast<long long>(Size(c) * vb), bj) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_accum_dot_single_rect<<<static_cast<unsigned>(bj), 256>>>(
          c, vb, ext, ext, obs, ob, c, vb, 0, 0, -1.0, d_f.ptr, d_sub_B.ptr, d_b_acc.ptr);

      launch_pack_rect_batch(bj, ext, ext, obs, ob, c, vb, d_f.ptr, d_sub_A.ptr);
      if (rm_gemm_batched(h, false, false, obs, vb, c, 1.0, d_Fa + obs, ext, 0, d_sub_A.ptr, vb,
                          static_cast<long long>(Size(c) * vb), 0.0, d_A2.ptr, vb,
                          static_cast<long long>(obs_vb), bj) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_accum_dot_single_rect<<<static_cast<unsigned>(bj), 256>>>(
          obs, vb, ext, ext, 0, ob, obs, vb, 0, 0, -1.0, d_f.ptr, d_A2.ptr, d_b_acc.ptr);

      launch_pack_rect_batch(bj, ext, ext, 0, obs, ext, c, d_f.ptr, d_sub_A.ptr);
      if (rm_gemm_batched(h, false, false, oa, c, ext, 1.0, d_Fa, ext, 0, d_sub_A.ptr, c,
                          static_cast<long long>(Size(ext) * c), 0.0, d_sub_B.ptr, c,
                          static_cast<long long>(oac), bj) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_accum_dot_single_rect<<<static_cast<unsigned>(bj), 256>>>(
          oa, c, ext, ext, 0, obs, oa, c, 0, 0, -1.0, d_f.ptr, d_sub_B.ptr, d_b_acc.ptr);

      launch_pack_rect_batch(bj, ext, ext, 0, obs, oa, c, d_f.ptr, d_sub_A.ptr);
      if (rm_gemm_batched(h, false, false, ext - oa, c, oa, 1.0, d_Fa + Size(oa) * ext, ext, 0,
                          d_sub_A.ptr, c, static_cast<long long>(oac), 0.0, d_sub_B.ptr, c,
                          static_cast<long long>(Size(ext - oa) * c), bj) !=
          CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_accum_dot_single_rect<<<static_cast<unsigned>(bj), 256>>>(
          ext - oa, c, ext, ext, oa, obs, ext - oa, c, 0, 0, -1.0, d_f.ptr, d_sub_B.ptr,
          d_b_acc.ptr);

      // 6. Beta complement-block cancellation <qf, qf Fb^T> - <f, f Fb^T>.
      launch_pack_rect_batch(bj, ext, ext, 0, 0, oa, ext, d_f.ptr, d_sub_A.ptr);
      if (rm_gemm(h, false, true, bj * oa, ext, ext, 1.0, d_sub_A.ptr, ext, d_Fb, ext, 0.0,
                  d_sub_B.ptr, ext) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_accum_dot_single_rect<<<static_cast<unsigned>(bj), 256>>>(
          oa, ext, oa, ext, 0, 0, oa, ext, 0, 0, -1.0, d_sub_A.ptr, d_sub_B.ptr, d_b_acc.ptr);

      launch_pack_rect_batch(bj, ext, ext, oa, 0, va, obs, d_f.ptr, d_sub_A.ptr);
      if (rm_gemm(h, false, true, bj * va, obs, obs, 1.0, d_sub_A.ptr, obs, d_Fb, ext, 0.0,
                  d_sub_B.ptr, obs) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_accum_dot_single_rect<<<static_cast<unsigned>(bj), 256>>>(
          va, obs, va, obs, 0, 0, va, obs, 0, 0, -1.0, d_sub_A.ptr, d_sub_B.ptr, d_b_acc.ptr);

      if (rm_gemm(h, false, true, bj * va, c, obs, 1.0, d_sub_A.ptr, obs, d_Fb + Size(obs) * ext,
                  ext, 0.0, d_sub_B.ptr, c) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_accum_dot_single_rect<<<static_cast<unsigned>(bj), 256>>>(
          va, c, ext, ext, oa, obs, va, c, 0, 0, -1.0, d_f.ptr, d_sub_B.ptr, d_b_acc.ptr);

      launch_pack_rect_batch(bj, ext, ext, oa, obs, va, c, d_f.ptr, d_sub_A.ptr);
      if (rm_gemm(h, false, true, bj * va, obs, c, 1.0, d_sub_A.ptr, c, d_Fb + obs, ext, 0.0,
                  d_C2.ptr, obs) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_accum_dot_single_rect<<<static_cast<unsigned>(bj), 256>>>(
          va, obs, ext, ext, oa, 0, va, obs, 0, 0, -1.0, d_f.ptr, d_C2.ptr, d_b_acc.ptr);

      launch_pack_rect_batch(bj, ext, ext, obs, 0, c, ext, d_f.ptr, d_sub_A.ptr);
      if (rm_gemm(h, false, true, bj * c, ob, ext, 1.0, d_sub_A.ptr, ext, d_Fb, ext, 0.0,
                  d_sub_B.ptr, ob) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_accum_dot_single_rect<<<static_cast<unsigned>(bj), 256>>>(
          c, ob, ext, ext, obs, 0, c, ob, 0, 0, -1.0, d_f.ptr, d_sub_B.ptr, d_b_acc.ptr);

      launch_pack_rect_batch(bj, ext, ext, obs, 0, c, ob, d_f.ptr, d_sub_A.ptr);
      if (rm_gemm(h, false, true, bj * c, ext - ob, ob, 1.0, d_sub_A.ptr, ob, d_Fb + Size(ob) * ext,
                  ext, 0.0, d_sub_B.ptr, ext - ob) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_accum_dot_single_rect<<<static_cast<unsigned>(bj), 256>>>(
          c, ext - ob, ext, ext, obs, ob, c, ext - ob, 0, 0, -1.0, d_f.ptr, d_sub_B.ptr,
          d_b_acc.ptr);

      // 7. Finalize AB pair energies using C = A2[oa:obs, :] + C2[:, ob:obs].
      k_ab_finalize<<<static_cast<unsigned>(bj), 256>>>(
          i_loc, j0, ob_act, oa, ob, fa, fb, obs, ext, include_ordinary, has_occ_weights, off_ab,
          d_occ_w_a, d_occ_w_b, d_Fa, d_Fb, d_lin_norm.ptr, d_b_acc.ptr, d_G_obs.ptr, d_A2.ptr,
          d_C2.ptr, d_out_scalars, d_pair_f12, d_pair_ord, d_err);
    }
  }
  return (cudaGetLastError() == cudaSuccess) ? 0 : 7;
}

// -------------------------------------------------------------------------
// GPU MP2-F12 Reverse Pair Response (V / B / X / C, 3C(FIX) SP, RHF + UHF)
// -------------------------------------------------------------------------

inline cublasStatus_t rm_gemv(cublasHandle_t h, bool trans, int M, int N, double alpha,
                              const double* A, int lda, const double* x, int incx, double beta,
                              double* y, int incy) {
  if (M <= 0 || N <= 0) return CUBLAS_STATUS_SUCCESS;
  return cublasDgemv(h, trans ? CUBLAS_OP_N : CUBLAS_OP_T, N, M, &alpha, A, lda, x, incx, &beta, y,
                     incy);
}

inline cublasStatus_t rm_ger(cublasHandle_t h, int M, int N, double alpha, const double* u,
                             int incu, const double* v, int incv, double* A, int lda) {
  if (M <= 0 || N <= 0 || alpha == 0.0) return CUBLAS_STATUS_SUCCESS;
  return cublasDger(h, N, M, &alpha, v, incv, u, incu, A, lda);
}

__global__ void k_oqe_to_qoe(int aux, int o_act, int ext, const double* __restrict__ oqe,
                             double* __restrict__ qoe) {
  const Size total = Size(aux) * Size(o_act) * Size(ext);
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int p = static_cast<int>(idx % ext);
  const Size ia = idx / ext;
  const int A = static_cast<int>(ia % aux);
  const int i = static_cast<int>(ia / aux);
  qoe[(Size(A) * o_act + i) * ext + p] = oqe[idx];
}

__global__ void k_axpy(Size total, double alpha, const double* __restrict__ x,
                       double* __restrict__ y) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) y[idx] += alpha * x[idx];
}

__global__ void k_lincomb2(Size total, double alpha, const double* __restrict__ a, double beta,
                           const double* __restrict__ b, double* __restrict__ out) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) out[idx] = alpha * a[idx] + beta * b[idx];
}

__global__ void k_lincomb2_inplace(Size total, double alpha, double beta,
                                   const double* __restrict__ x, double* __restrict__ y) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) y[idx] = alpha * y[idx] + beta * x[idx];
}

__global__ void k_sub_scale_both(Size total, double alpha, const double* __restrict__ M,
                                 double* __restrict__ out1, double* __restrict__ out2) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    const double v = alpha * M[idx];
    out1[idx] -= v;
    out2[idx] -= v;
  }
}

__global__ void k_zero_p_blocks(int oa, int ob, int obs, int ext, double* __restrict__ M) {
  const Size total = Size(ext) * ext;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int p = static_cast<int>(idx / ext);
  const int r = static_cast<int>(idx % ext);
  if ((p < obs && r < obs) || (p < oa && r >= obs) || (p >= obs && r < ob)) {
    M[idx] = 0.0;
  }
}

__global__ void k_sym_add_transpose(int n, double alpha, const double* __restrict__ src,
                                    double* __restrict__ dst) {
  const Size total = Size(n) * n;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int p = static_cast<int>(idx / n);
  const int r = static_cast<int>(idx % n);
  dst[idx] += alpha * (src[idx] + src[Size(r) * n + p]);
}

__global__ void k_add_transpose_scaled(int n, double alpha, const double* __restrict__ src,
                                       double* __restrict__ dst) {
  const Size total = Size(n) * n;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int p = static_cast<int>(idx / n);
  const int r = static_cast<int>(idx % n);
  dst[idx] += alpha * src[Size(r) * n + p];
}

__global__ void k_w_sym_mat(int ext, double alpha, double beta, const double* __restrict__ src,
                            double* __restrict__ dst) {
  const Size total = Size(ext) * ext;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int p = static_cast<int>(idx / ext);
  const int r = static_cast<int>(idx % ext);
  dst[idx] = alpha * src[idx] + beta * src[Size(r) * ext + p];
}

__global__ void k_build_ab_sym_mats(int ext, const double* __restrict__ Fa,
                                    const double* __restrict__ Fb, const double* __restrict__ Ka,
                                    const double* __restrict__ Kb, double* __restrict__ Ha,
                                    double* __restrict__ Hb, double* __restrict__ Fa_sym,
                                    double* __restrict__ Fb_sym, double* __restrict__ Ha_sym,
                                    double* __restrict__ Hb_sym) {
  const Size total = Size(ext) * ext;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int p = static_cast<int>(idx / ext);
  const int r = static_cast<int>(idx % ext);
  const Size qp = Size(r) * ext + p;
  Ha[idx] = Fa[idx] + Ka[idx];
  Hb[idx] = Fb[idx] + Kb[idx];
  const double fas = Fa[idx] + Fa[qp];
  const double fbs = Fb[idx] + Fb[qp];
  Fa_sym[idx] = fas;
  Fb_sym[idx] = fbs;
  Ha_sym[idx] = fas + Ka[idx] + Ka[qp];
  Hb_sym[idx] = fbs + Kb[idx] + Kb[qp];
}

__global__ void k_add_col_scaled_vec(int rows, int cols, int col_idx, double alpha,
                                     const double* __restrict__ vec, double* __restrict__ M) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p < rows) M[Size(p) * cols + col_idx] += alpha * vec[p];
}

__global__ void k_add_row_scaled_vec(int cols, int row_idx, double alpha,
                                     const double* __restrict__ vec, double* __restrict__ M) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p < cols) M[Size(row_idx) * cols + p] += alpha * vec[p];
}

__global__ void k_set_unit_vec(int ext, int pos, double* __restrict__ v) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p < ext) v[p] = (p == pos) ? 1.0 : 0.0;
}

__global__ void k_extract_col(int rows, int cols, int col_idx, const double* __restrict__ M,
                              double* __restrict__ out) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p < rows) out[p] = M[Size(p) * cols + col_idx];
}

__global__ void k_robust_elem_pullback(double alpha, int p, int r, int aux, int ext,
                                       const double* fit_i, const double* raw_i,
                                       const double* fit_j, const double* res_j,
                                       double* bar_fit_i, double* bar_raw_i,
                                       double* bar_fit_j, double* bar_res_j) {
  const int A = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (A >= aux) return;
  const Size row = Size(A) * ext;
  const double fi = fit_i[row + p];
  const double ri = raw_i[row + p];
  const double fj = fit_j[row + r];
  const double sj = res_j[row + r];
  if (bar_fit_i == bar_fit_j && p == r) {
    bar_fit_i[row + p] = (bar_fit_i[row + p] + alpha * sj) + alpha * ri;
    bar_res_j[row + r] += alpha * fi;
    bar_raw_i[row + p] += alpha * fj;
  } else {
    bar_fit_i[row + p] += alpha * sj;
    bar_res_j[row + r] += alpha * fi;
    bar_raw_i[row + p] += alpha * fj;
    bar_fit_j[row + r] += alpha * ri;
  }
}

__global__ void k_add_scaled_two_vecs_to_row_both(int ext, int row_idx, double a1,
                                                  const double* __restrict__ v1, double a2,
                                                  const double* __restrict__ v2,
                                                  double* __restrict__ M1,
                                                  double* __restrict__ M2) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p < ext) {
    const double val = a1 * v1[p] + a2 * v2[p];
    M1[Size(row_idx) * ext + p] += val;
    M2[Size(row_idx) * ext + p] += val;
  }
}

__global__ void k_add_scaled_two_vecs_to_col_both(int ext, int col_idx, double a1,
                                                  const double* __restrict__ v1, double a2,
                                                  const double* __restrict__ v2,
                                                  double* __restrict__ M1,
                                                  double* __restrict__ M2) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p < ext) {
    const double val = a1 * v1[p] + a2 * v2[p];
    M1[Size(p) * ext + col_idx] += val;
    M2[Size(p) * ext + col_idx] += val;
  }
}

__global__ void k_rhf_sq_sab_build_vecs(int ext, int i, int j, double s_f12, double occ_sum,
                                        const double* __restrict__ hi,
                                        const double* __restrict__ hj,
                                        const double* __restrict__ m_col_i,
                                        const double* __restrict__ m_col_j,
                                        const double* __restrict__ m_row_i,
                                        const double* __restrict__ m_row_j,
                                        const double* __restrict__ ff0_hi,
                                        const double* __restrict__ ff0_hj,
                                        double* __restrict__ x1, double* __restrict__ x3,
                                        double* __restrict__ y2, double* __restrict__ y4,
                                        double* __restrict__ ff0_x1, double* __restrict__ ff0_x3,
                                        double* __restrict__ ff_ei, double* __restrict__ ffT_ej) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= ext) return;
  x1[p] = s_f12 * (hi[p] - (p == i ? occ_sum : 0.0));
  x3[p] = s_f12 * (hj[p] - (p == j ? occ_sum : 0.0));
  y2[p] = s_f12 * hj[p];
  y4[p] = s_f12 * hi[p];
  ff0_x1[p] = s_f12 * (ff0_hi[p] - occ_sum * m_col_i[p]);
  ff0_x3[p] = s_f12 * (ff0_hj[p] - occ_sum * m_col_j[p]);
  ff_ei[p] = 0.375 * m_col_i[p] + 0.125 * m_row_i[p];
  ffT_ej[p] = 0.375 * m_row_j[p] + 0.125 * m_col_j[p];
}

// Two sequential race-free 1D passes: pass 1 writes columns j & i (unique row p per thread),
// pass 2 writes rows i & j (unique col p per thread), eliminating all (i,j)/(j,i)/(i,i)/(j,j) races.
__global__ void k_rhf_sq_sab_fold_cols(int ext, int i, int j, double s_f12, double occ_sum,
                                       const double* __restrict__ ff0_x1,
                                       const double* __restrict__ ff0_x3,
                                       const double* __restrict__ ff0_hi,
                                       const double* __restrict__ ff0T_hi,
                                       const double* __restrict__ ff_ei,
                                       const double* __restrict__ h_ff_ei,
                                       double* __restrict__ bar_Sab_ij) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= ext) return;
  const double ff_hi_p = 0.375 * ff0_hi[p] + 0.125 * ff0T_hi[p];
  const double bw_v = s_f12 * (-occ_sum * ff_ei[p] + h_ff_ei[p] + ff_hi_p);
  const double col_j = (3.0 / 64.0) * ff0_x1[p] + 0.125 * bw_v;
  const double col_i = (1.0 / 64.0) * ff0_x3[p];
  if (i == j) {
    bar_Sab_ij[Size(p) * ext + j] = (bar_Sab_ij[Size(p) * ext + j] + col_j) + col_i;
  } else {
    bar_Sab_ij[Size(p) * ext + j] += col_j;
    bar_Sab_ij[Size(p) * ext + i] += col_i;
  }
}

__global__ void k_rhf_sq_sab_fold_rows(int ext, int i, int j, double s_f12, double occ_sum,
                                       const double* __restrict__ ff0T_hi,
                                       const double* __restrict__ ff0_hj,
                                       const double* __restrict__ ff0T_hj,
                                       const double* __restrict__ ffT_ej,
                                       const double* __restrict__ h_ffT_ej,
                                       double* __restrict__ bar_Sab_ij) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= ext) return;
  const double ffT_hj_p = 0.375 * ff0T_hj[p] + 0.125 * ff0_hj[p];
  const double bwT_u = s_f12 * (-occ_sum * ffT_ej[p] + ffT_hj_p + h_ffT_ej[p]);
  const double row_i = (3.0 / 64.0) * s_f12 * ff0T_hj[p] + 0.125 * bwT_u;
  const double row_j = (1.0 / 64.0) * s_f12 * ff0T_hi[p];
  if (i == j) {
    bar_Sab_ij[Size(i) * ext + p] = (bar_Sab_ij[Size(i) * ext + p] + row_i) + row_j;
  } else {
    bar_Sab_ij[Size(i) * ext + p] += row_i;
    bar_Sab_ij[Size(j) * ext + p] += row_j;
  }
}

__global__ void k_rhf_dc_sab_fold_cols(int ext, int i, int j, double s_f12,
                                       const double* __restrict__ u_col_i,
                                       const double* __restrict__ u_col_j,
                                       const double* __restrict__ u_row_i,
                                       double* __restrict__ bar_Sab_ij) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= ext) return;
  const double c3_64 = (3.0 / 64.0) * s_f12;
  const double c1_64 = (1.0 / 64.0) * s_f12;
  const double bw_v = s_f12 * (0.375 * u_col_i[p] + 0.125 * u_row_i[p]);
  const double col_j = c3_64 * u_col_i[p] + 0.125 * bw_v;
  const double col_i = c1_64 * u_col_j[p];
  if (i == j) {
    bar_Sab_ij[Size(p) * ext + j] = (bar_Sab_ij[Size(p) * ext + j] + col_j) + col_i;
  } else {
    bar_Sab_ij[Size(p) * ext + j] += col_j;
    bar_Sab_ij[Size(p) * ext + i] += col_i;
  }
}

__global__ void k_rhf_dc_sab_fold_rows(int ext, int i, int j, double s_f12,
                                       const double* __restrict__ u_row_j,
                                       const double* __restrict__ u_col_j,
                                       const double* __restrict__ u_row_i,
                                       double* __restrict__ bar_Sab_ij) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= ext) return;
  const double c3_64 = (3.0 / 64.0) * s_f12;
  const double c1_64 = (1.0 / 64.0) * s_f12;
  const double bwT_u = s_f12 * (0.375 * u_row_j[p] + 0.125 * u_col_j[p]);
  const double row_i = c3_64 * u_row_j[p] + 0.125 * bwT_u;
  const double row_j = c1_64 * u_row_i[p];
  if (i == j) {
    bar_Sab_ij[Size(i) * ext + p] = (bar_Sab_ij[Size(i) * ext + p] + row_i) + row_j;
  } else {
    bar_Sab_ij[Size(i) * ext + p] += row_i;
    bar_Sab_ij[Size(j) * ext + p] += row_j;
  }
}

__global__ void k_ab_sq_build_vecs(int ext, int i, double s_f12, double bar_norm,
                                   const double* __restrict__ ha_i,
                                   const double* __restrict__ hb_j,
                                   const double* __restrict__ ha_uj,
                                   const double* __restrict__ hb_vi,
                                   const double* __restrict__ u_j, double* __restrict__ x1,
                                   double* __restrict__ y2, double* __restrict__ x3,
                                   double* __restrict__ y4) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= ext) return;
  x1[p] = s_f12 * ha_i[p] + (p == i ? bar_norm : 0.0);
  y2[p] = s_f12 * hb_j[p];
  x3[p] = s_f12 * ha_uj[p] + bar_norm * u_j[p];
  y4[p] = s_f12 * hb_vi[p];
}

__global__ void k_ab_sq_bar_sab_bw_col(int ext, int j, double s_f12, double bar_norm,
                                       const double* __restrict__ ff_vi,
                                       const double* __restrict__ mv1_v,
                                       const double* __restrict__ mv2_v,
                                       double* __restrict__ bar_Sab) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= ext) return;
  const double bw_v = bar_norm * ff_vi[p] + s_f12 * (mv1_v[p] + mv2_v[p]);
  bar_Sab[Size(p) * ext + j] += 0.125 * bw_v;
}

__global__ void k_ab_sq_bar_sab_bw_row(int ext, int i, double s_f12, double bar_norm,
                                       const double* __restrict__ ffT_uj,
                                       const double* __restrict__ mv1_u,
                                       const double* __restrict__ mv2_u,
                                       double* __restrict__ bar_Sab) {
  const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= ext) return;
  const double bwT_u = bar_norm * ffT_uj[p] + s_f12 * (mv1_u[p] + mv2_u[p]);
  bar_Sab[Size(i) * ext + p] += 0.125 * bwT_u;
}

int robust_bilin_pullback_gpu(cublasHandle_t h, double alpha, const double* d_x, const double* d_y,
                              int aux, int ext, const double* d_fit_i, const double* d_raw_i,
                              const double* d_fit_j, const double* d_res_j, double* d_bfit_i,
                              double* d_braw_i, double* d_bfit_j, double* d_bres_j, double* d_fx,
                              double* d_rx, double* d_sy, double* d_fy) {
  if (alpha == 0.0) return 0;
  if (rm_gemv(h, false, aux, ext, 1.0, d_fit_i, ext, d_x, 1, 0.0, d_fx, 1) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemv(h, false, aux, ext, 1.0, d_raw_i, ext, d_x, 1, 0.0, d_rx, 1) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemv(h, false, aux, ext, 1.0, d_res_j, ext, d_y, 1, 0.0, d_sy, 1) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemv(h, false, aux, ext, 1.0, d_fit_j, ext, d_y, 1, 0.0, d_fy, 1) !=
          CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  if (rm_ger(h, aux, ext, alpha, d_sy, 1, d_x, 1, d_bfit_i, ext) != CUBLAS_STATUS_SUCCESS ||
      rm_ger(h, aux, ext, alpha, d_fy, 1, d_x, 1, d_braw_i, ext) != CUBLAS_STATUS_SUCCESS ||
      rm_ger(h, aux, ext, alpha, d_fx, 1, d_y, 1, d_bres_j, ext) != CUBLAS_STATUS_SUCCESS ||
      rm_ger(h, aux, ext, alpha, d_rx, 1, d_y, 1, d_bfit_j, ext) != CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  return 0;
}

int robust_row_col_pullback_gpu(cublasHandle_t h, double alpha_row, int row_idx,
                                const double* d_y_row, double alpha_col, int col_idx,
                                const double* d_x_col, int aux, int ext, const double* d_fit_i,
                                const double* d_raw_i, const double* d_fit_j,
                                const double* d_res_j, double* d_bfit_i, double* d_braw_i,
                                double* d_bfit_j, double* d_bres_j, double* d_aux1,
                                double* d_aux2) {
  if (alpha_row != 0.0) {
    if (rm_gemv(h, false, aux, ext, 1.0, d_res_j, ext, d_y_row, 1, 0.0, d_aux1, 1) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemv(h, false, aux, ext, 1.0, d_fit_j, ext, d_y_row, 1, 0.0, d_aux2, 1) !=
            CUBLAS_STATUS_SUCCESS) {
      return 4;
    }
    k_add_col_scaled_vec<<<grid_for(aux), 256>>>(aux, ext, row_idx, alpha_row, d_aux1, d_bfit_i);
    k_add_col_scaled_vec<<<grid_for(aux), 256>>>(aux, ext, row_idx, alpha_row, d_aux2, d_braw_i);
    if (rm_ger(h, aux, ext, alpha_row, d_fit_i + row_idx, ext, d_y_row, 1, d_bres_j, ext) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_ger(h, aux, ext, alpha_row, d_raw_i + row_idx, ext, d_y_row, 1, d_bfit_j, ext) !=
            CUBLAS_STATUS_SUCCESS) {
      return 4;
    }
  }
  if (alpha_col != 0.0) {
    if (rm_gemv(h, false, aux, ext, 1.0, d_fit_i, ext, d_x_col, 1, 0.0, d_aux1, 1) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemv(h, false, aux, ext, 1.0, d_raw_i, ext, d_x_col, 1, 0.0, d_aux2, 1) !=
            CUBLAS_STATUS_SUCCESS) {
      return 4;
    }
    k_add_col_scaled_vec<<<grid_for(aux), 256>>>(aux, ext, col_idx, alpha_col, d_aux1, d_bres_j);
    k_add_col_scaled_vec<<<grid_for(aux), 256>>>(aux, ext, col_idx, alpha_col, d_aux2, d_bfit_j);
    if (rm_ger(h, aux, ext, alpha_col, d_res_j + col_idx, ext, d_x_col, 1, d_bfit_i, ext) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_ger(h, aux, ext, alpha_col, d_fit_j + col_idx, ext, d_x_col, 1, d_braw_i, ext) !=
            CUBLAS_STATUS_SUCCESS) {
      return 4;
    }
  }
  return 0;
}

int robust_rmatvec_gpu(cublasHandle_t h, const double* d_fit_i, const double* d_raw_i,
                       const double* d_fit_j, const double* d_res_j, int aux, int ext,
                       const double* d_y, double* d_out, double* d_sy, double* d_fy) {
  if (rm_gemv(h, false, aux, ext, 1.0, d_res_j, ext, d_y, 1, 0.0, d_sy, 1) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemv(h, false, aux, ext, 1.0, d_fit_j, ext, d_y, 1, 0.0, d_fy, 1) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemv(h, true, aux, ext, 1.0, d_fit_i, ext, d_sy, 1, 0.0, d_out, 1) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemv(h, true, aux, ext, 1.0, d_raw_i, ext, d_fy, 1, 1.0, d_out, 1) !=
          CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  return 0;
}

int robust_lmatvec_gpu(cublasHandle_t h, const double* d_fit_i, const double* d_raw_i,
                       const double* d_fit_j, const double* d_res_j, int aux, int ext,
                       const double* d_x, double* d_out, double* d_fx, double* d_rx) {
  if (rm_gemv(h, false, aux, ext, 1.0, d_fit_i, ext, d_x, 1, 0.0, d_fx, 1) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemv(h, false, aux, ext, 1.0, d_raw_i, ext, d_x, 1, 0.0, d_rx, 1) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemv(h, true, aux, ext, 1.0, d_res_j, ext, d_fx, 1, 0.0, d_out, 1) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemv(h, true, aux, ext, 1.0, d_fit_j, ext, d_rx, 1, 1.0, d_out, 1) !=
          CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  return 0;
}

int robust_row_gpu(cublasHandle_t h, const double* d_fit_i, const double* d_raw_i,
                   const double* d_fit_j, const double* d_res_j, int aux, int ext, int row_idx,
                   double* d_out) {
  if (rm_gemv(h, true, aux, ext, 1.0, d_res_j, ext, d_fit_i + row_idx, ext, 0.0, d_out, 1) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemv(h, true, aux, ext, 1.0, d_fit_j, ext, d_raw_i + row_idx, ext, 1.0, d_out, 1) !=
          CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  return 0;
}

int robust_col_gpu(cublasHandle_t h, const double* d_fit_i, const double* d_raw_i,
                   const double* d_fit_j, const double* d_res_j, int aux, int ext, int col_idx,
                   double* d_out) {
  if (rm_gemv(h, true, aux, ext, 1.0, d_fit_i, ext, d_res_j + col_idx, ext, 0.0, d_out, 1) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemv(h, true, aux, ext, 1.0, d_raw_i, ext, d_fit_j + col_idx, ext, 1.0, d_out, 1) !=
          CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  return 0;
}

int weight_rank1_pullback_gpu(cublasHandle_t h, double coeff, const double* d_x, const double* d_y,
                              const double* d_Sab, int aux, int ext, const double* d_fit_i,
                              const double* d_raw_i, const double* d_fit_j, const double* d_res_j,
                              double* d_bfit_i, double* d_braw_i, double* d_bfit_j,
                              double* d_bres_j, double* d_bar_Sab, double* d_sy_v, double* d_stx_v,
                              double* d_mv1, double* d_mv2, double* d_a1, double* d_a2,
                              double* d_a3, double* d_a4) {
  if (coeff == 0.0) return 0;
  if (rm_gemv(h, false, ext, ext, 1.0, d_Sab, ext, d_y, 1, 0.0, d_sy_v, 1) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemv(h, true, ext, ext, 1.0, d_Sab, ext, d_x, 1, 0.0, d_stx_v, 1) !=
          CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  if (int rc = robust_bilin_pullback_gpu(h, 0.375 * coeff, d_x, d_y, aux, ext, d_fit_i, d_raw_i,
                                         d_fit_j, d_res_j, d_bfit_i, d_braw_i, d_bfit_j, d_bres_j,
                                         d_a1, d_a2, d_a3, d_a4);
      rc != 0) {
    return rc;
  }
  if (int rc = robust_bilin_pullback_gpu(h, 0.125 * coeff, d_sy_v, d_stx_v, aux, ext, d_fit_i,
                                         d_raw_i, d_fit_j, d_res_j, d_bfit_i, d_braw_i, d_bfit_j,
                                         d_bres_j, d_a1, d_a2, d_a3, d_a4);
      rc != 0) {
    return rc;
  }
  if (d_bar_Sab) {
    if (int rc = robust_lmatvec_gpu(h, d_fit_i, d_raw_i, d_fit_j, d_res_j, aux, ext, d_sy_v, d_mv1,
                                    d_a1, d_a2);
        rc != 0) {
      return rc;
    }
    if (int rc = robust_rmatvec_gpu(h, d_fit_i, d_raw_i, d_fit_j, d_res_j, aux, ext, d_stx_v, d_mv2,
                                    d_a1, d_a2);
        rc != 0) {
      return rc;
    }
    if (rm_ger(h, ext, ext, 0.125 * coeff, d_x, 1, d_mv1, 1, d_bar_Sab, ext) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_ger(h, ext, ext, 0.125 * coeff, d_mv2, 1, d_y, 1, d_bar_Sab, ext) !=
            CUBLAS_STATUS_SUCCESS) {
      return 4;
    }
  }
  return 0;
}

int weight_rmatvec_gpu(cublasHandle_t h, const double* d_Sab, const double* d_fit_i,
                       const double* d_raw_i, const double* d_fit_j, const double* d_res_j,
                       int aux, int ext, const double* d_y, double* d_out, double* d_tmp1,
                       double* d_tmp2, double* d_a1, double* d_a2) {
  if (int rc = robust_rmatvec_gpu(h, d_fit_i, d_raw_i, d_fit_j, d_res_j, aux, ext, d_y, d_out, d_a1,
                                  d_a2);
      rc != 0) {
    return rc;
  }
  if (rm_gemv(h, false, ext, ext, 1.0, d_Sab, ext, d_y, 1, 0.0, d_tmp1, 1) !=
      CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  if (int rc = robust_lmatvec_gpu(h, d_fit_i, d_raw_i, d_fit_j, d_res_j, aux, ext, d_tmp1, d_tmp2,
                                  d_a1, d_a2);
      rc != 0) {
    return rc;
  }
  if (rm_gemv(h, false, ext, ext, 1.0, d_Sab, ext, d_tmp2, 1, 0.0, d_tmp1, 1) !=
      CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  k_lincomb2_inplace<<<grid_for(ext), 256>>>(ext, 0.375, 0.125, d_tmp1, d_out);
  return 0;
}

int weight_lmatvec_gpu(cublasHandle_t h, const double* d_Sab, const double* d_fit_i,
                       const double* d_raw_i, const double* d_fit_j, const double* d_res_j,
                       int aux, int ext, const double* d_x, double* d_out, double* d_tmp1,
                       double* d_tmp2, double* d_a1, double* d_a2) {
  if (int rc = robust_lmatvec_gpu(h, d_fit_i, d_raw_i, d_fit_j, d_res_j, aux, ext, d_x, d_out, d_a1,
                                  d_a2);
      rc != 0) {
    return rc;
  }
  if (rm_gemv(h, true, ext, ext, 1.0, d_Sab, ext, d_x, 1, 0.0, d_tmp1, 1) !=
      CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  if (int rc = robust_rmatvec_gpu(h, d_fit_i, d_raw_i, d_fit_j, d_res_j, aux, ext, d_tmp1, d_tmp2,
                                  d_a1, d_a2);
      rc != 0) {
    return rc;
  }
  if (rm_gemv(h, true, ext, ext, 1.0, d_Sab, ext, d_tmp2, 1, 0.0, d_tmp1, 1) !=
      CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  k_lincomb2_inplace<<<grid_for(ext), 256>>>(ext, 0.375, 0.125, d_tmp1, d_out);
  return 0;
}

int pullback_res_oqe_gpu(cublasHandle_t h, const double* d_bar_res_oqe, const double* d_fit_qoe,
                         const double* d_metric, double metric_scale, bool sym_metric, int aux,
                         int o_act, int ext, double* d_bar_raw_oqe, double* d_bar_fit_qoe,
                         double* d_bar_metric, double* d_qoe_tmp, double* d_m_tmp) {
  if (o_act <= 0) return 0;
  const Size n_tot = Size(aux) * Size(o_act) * ext;
  const int cols = o_act * ext;
  k_axpy<<<grid_for(n_tot), 256>>>(n_tot, 1.0, d_bar_res_oqe, d_bar_raw_oqe);
  k_oqe_to_qoe<<<grid_for(n_tot), 256>>>(aux, o_act, ext, d_bar_res_oqe, d_qoe_tmp);
  if (rm_gemm(h, true, false, aux, cols, aux, -1.0, d_metric, aux, d_qoe_tmp, cols, 1.0,
              d_bar_fit_qoe, cols) != CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  if (d_bar_metric) {
    if (!sym_metric) {
      if (rm_gemm(h, false, true, aux, aux, cols, -metric_scale, d_qoe_tmp, cols, d_fit_qoe, cols,
                  1.0, d_bar_metric, aux) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
    } else {
      if (rm_gemm(h, false, true, aux, aux, cols, -0.5 * metric_scale, d_qoe_tmp, cols, d_fit_qoe,
                  cols, 0.0, d_m_tmp, aux) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_sym_add_transpose<<<grid_for(Size(aux) * aux), 256>>>(aux, 1.0, d_m_tmp, d_bar_metric);
    }
  }
  return (cudaGetLastError() == cudaSuccess) ? 0 : 7;
}

int scatter_active_qoe_cuda_host(const double* d_qoe, int aux, int occ, int f, int o_act, int ext,
                                 double* dst_host, std::vector<double>& h_qoe) {
  if (!dst_host || o_act <= 0) return 0;
  const Size n_tot = Size(aux) * Size(o_act) * ext;
  h_qoe.resize(n_tot);
  if (cudaMemcpy(h_qoe.data(), d_qoe, n_tot * sizeof(double), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    return 6;
  }
  for (int A = 0; A < aux; ++A) {
    const double* s = h_qoe.data() + Size(A) * o_act * ext;
    double* d = dst_host + (Size(A) * occ + f) * ext;
    std::copy_n(s, Size(o_act) * ext, d);
  }
  return 0;
}

int scatter_active_oqe_cuda_host(const double* d_oqe, int aux, int occ, int f, int o_act, int ext,
                                 double* dst_host, std::vector<double>& h_qoe, double* d_qoe_tmp) {
  if (!dst_host || o_act <= 0) return 0;
  const Size n_tot = Size(aux) * Size(o_act) * ext;
  k_oqe_to_qoe<<<grid_for(n_tot), 256>>>(aux, o_act, ext, d_oqe, d_qoe_tmp);
  if (cudaGetLastError() != cudaSuccess) return 7;
  return scatter_active_qoe_cuda_host(d_qoe_tmp, aux, occ, f, o_act, ext, dst_host, h_qoe);
}

int b_overlap_pullback_gpu(cublasHandle_t h, double scale, const double* d_A, const double* d_B,
                           const double* d_F, const double* d_K, int o, int obs, int ext,
                           double* d_bar_A, double* d_bar_B, double* d_bar_F, double* d_bar_K) {
  if (scale == 0.0) return 0;
  const int v = obs - o;
  const int c = ext - obs;
  // Term 0: -scale * <A @ K, B>
  if (rm_gemm(h, false, true, ext, ext, ext, -scale, d_B, ext, d_K, ext, 1.0, d_bar_A, ext) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, false, false, ext, ext, ext, -scale, d_A, ext, d_K, ext, 1.0, d_bar_B, ext) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, true, false, ext, ext, ext, -scale, d_A, ext, d_B, ext, 1.0, d_bar_K, ext) !=
          CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  // Term 1: -scale * <A[:o, :] @ F, B[:o, :]>
  if (rm_gemm(h, false, true, o, ext, ext, -scale, d_B, ext, d_F, ext, 1.0, d_bar_A, ext) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, false, false, o, ext, ext, -scale, d_A, ext, d_F, ext, 1.0, d_bar_B, ext) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, true, false, ext, ext, o, -scale, d_A, ext, d_B, ext, 1.0, d_bar_F, ext) !=
          CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  // Term 2: +scale * <A[obs:, :o] @ F[:o, :o], B[obs:, :o]>
  if (rm_gemm(h, false, true, c, o, o, scale, d_B + Size(obs) * ext, ext, d_F, ext, 1.0,
              d_bar_A + Size(obs) * ext, ext) != CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, false, false, c, o, o, scale, d_A + Size(obs) * ext, ext, d_F, ext, 1.0,
              d_bar_B + Size(obs) * ext, ext) != CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, true, false, o, o, c, scale, d_A + Size(obs) * ext, ext, d_B + Size(obs) * ext,
              ext, 1.0, d_bar_F, ext) != CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  // Term 3: -scale * <A[o:obs, :obs] @ F[:obs, :obs], B[o:obs, :obs]>
  if (rm_gemm(h, false, true, v, obs, obs, -scale, d_B + Size(o) * ext, ext, d_F, ext, 1.0,
              d_bar_A + Size(o) * ext, ext) != CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, false, false, v, obs, obs, -scale, d_A + Size(o) * ext, ext, d_F, ext, 1.0,
              d_bar_B + Size(o) * ext, ext) != CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, true, false, obs, obs, v, -scale, d_A + Size(o) * ext, ext, d_B + Size(o) * ext,
              ext, 1.0, d_bar_F, ext) != CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  // Term 4: -2 * scale * <A[obs:, :] @ F[:o, :]^T, B[obs:, :o]>
  if (rm_gemm(h, false, false, c, ext, o, -2.0 * scale, d_B + Size(obs) * ext, ext, d_F, ext, 1.0,
              d_bar_A + Size(obs) * ext, ext) != CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, false, true, c, o, ext, -2.0 * scale, d_A + Size(obs) * ext, ext, d_F, ext, 1.0,
              d_bar_B + Size(obs) * ext, ext) != CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, true, false, o, ext, c, -2.0 * scale, d_B + Size(obs) * ext, ext,
              d_A + Size(obs) * ext, ext, 1.0, d_bar_F, ext) != CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  // Term 5: -2 * scale * <A[o:obs, :obs] @ F[:obs, obs:], B[o:obs, obs:]>
  if (rm_gemm(h, false, true, v, obs, c, -2.0 * scale, d_B + Size(o) * ext + obs, ext, d_F + obs,
              ext, 1.0, d_bar_A + Size(o) * ext, ext) != CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, false, false, v, c, obs, -2.0 * scale, d_A + Size(o) * ext, ext, d_F + obs, ext,
              1.0, d_bar_B + Size(o) * ext + obs, ext) != CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, true, false, obs, c, v, -2.0 * scale, d_A + Size(o) * ext, ext,
              d_B + Size(o) * ext + obs, ext, 1.0, d_bar_F + obs, ext) != CUBLAS_STATUS_SUCCESS) {
    return 4;
  }
  return 0;
}

// Stage 1 Reverse for one operator (op_kind: 0=gc, 1=dc, 2=sq) on SS or RHF spatial channel.
int run_ss_or_rhf_stage1_op_rev_gpu(
    cublasHandle_t h, int op_kind, bool compute_both_rhf, bool has_occ_weights, int o, int f,
    int ext, int aux, const double* occ_w_host, const double* fock_host, const double* d_fit_oqe,
    const double* d_raw_op_oqe, const double* d_res_op_oqe, const double* d_H,
    double* d_bar_fit_oqe, double* d_bar_raw_op_oqe, double* d_bar_res_op_oqe,
    double* d_bar_res_os_op_oqe, double* d_bar_F, double* d_bar_K, double* d_bar_Sab) {
  const int o_act = o - f;
  if (compute_both_rhf ? (o_act <= 0) : (o_act < 2)) return 0;
  const Size qe = Size(aux) * ext;
  const Size ee = Size(ext) * ext;

  DeviceBuf d_a1, d_a2, d_v1, d_v2, d_v3, d_v4, d_v5, d_v6, d_v7, d_v8, d_v9, d_sab_ij;
  if (d_a1.alloc(aux) != cudaSuccess || d_a2.alloc(aux) != cudaSuccess ||
      d_v1.alloc(ext) != cudaSuccess || d_v2.alloc(ext) != cudaSuccess ||
      d_v3.alloc(ext) != cudaSuccess || d_v4.alloc(ext) != cudaSuccess) {
    return 2;
  }
  if (op_kind == 2 || (compute_both_rhf && d_bar_Sab != nullptr)) {
    if (d_v5.alloc(ext) != cudaSuccess || d_v6.alloc(ext) != cudaSuccess ||
        d_v7.alloc(ext) != cudaSuccess || d_v8.alloc(ext) != cudaSuccess ||
        d_v9.alloc(Size(10) * ext) != cudaSuccess || d_sab_ij.alloc(ee) != cudaSuccess) {
      return 2;
    }
  }

  for (int i_loc = 0; i_loc < o_act; ++i_loc) {
    const int i = f + i_loc;
    const int j_start = compute_both_rhf ? i_loc : (i_loc + 1);
    const double* fit_i = d_fit_oqe + Size(i_loc) * qe;
    const double* raw_i = d_raw_op_oqe + Size(i_loc) * qe;
    double* bfit_i = d_bar_fit_oqe + Size(i_loc) * qe;
    double* braw_i = d_bar_raw_op_oqe + Size(i_loc) * qe;
    const double* d_hi = d_H + Size(i) * ext;

    for (int j_loc = j_start; j_loc < o_act; ++j_loc) {
      const int j = f + j_loc;
      const bool diag = (i_loc == j_loc);
      const double s_f12 = has_occ_weights ? 0.5 * (occ_w_host[i_loc] + occ_w_host[j_loc]) : 1.0;
      const double spin_scale = (compute_both_rhf && diag) ? 0.5 : 1.0;
      const double occ_sum = fock_host[Size(i) * ext + i] + fock_host[Size(j) * ext + j];
      const double* fit_j = d_fit_oqe + Size(j_loc) * qe;
      const double* res_j = d_res_op_oqe + Size(j_loc) * qe;
      double* bfit_j = d_bar_fit_oqe + Size(j_loc) * qe;
      double* bres_j = d_bar_res_op_oqe + Size(j_loc) * qe;
      double* bres_os_j = compute_both_rhf ? (d_bar_res_os_op_oqe + Size(j_loc) * qe) : nullptr;
      const double* d_hj = d_H + Size(j) * ext;

      if (op_kind == 0) {
        // geminal_coulomb (gc)
        if (!diag) {
          k_robust_elem_pullback<<<grid_for(aux), 256>>>(0.5 * s_f12, i, j, aux, ext, fit_i, raw_i,
                                                         fit_j, res_j, bfit_i, braw_i, bfit_j,
                                                         bres_j);
          k_robust_elem_pullback<<<grid_for(aux), 256>>>(-0.5 * s_f12, j, i, aux, ext, fit_i, raw_i,
                                                         fit_j, res_j, bfit_i, braw_i, bfit_j,
                                                         bres_j);
        }
        if (compute_both_rhf) {
          const double bar_V_d_os = diag ? s_f12 : (0.75 * s_f12);
          k_robust_elem_pullback<<<grid_for(aux), 256>>>(spin_scale * bar_V_d_os, i, j, aux, ext,
                                                         fit_i, raw_i, fit_j, res_j, bfit_i, braw_i,
                                                         bfit_j, bres_os_j);
          if (!diag) {
            k_robust_elem_pullback<<<grid_for(aux), 256>>>(0.25 * s_f12, j, i, aux, ext, fit_i,
                                                           raw_i, fit_j, res_j, bfit_i, braw_i,
                                                           bfit_j, bres_os_j);
          }
          if (d_bar_Sab) {
            if (int rc = robust_col_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, i, d_v1.ptr);
                rc != 0)
              return rc;
            if (int rc = robust_row_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, j, d_v2.ptr);
                rc != 0)
              return rc;
            const double c_gc = 0.25 * s_f12;
            k_add_col_scaled_vec<<<grid_for(ext), 256>>>(ext, ext, j, c_gc, d_v1.ptr, d_bar_Sab);
            k_add_row_scaled_vec<<<grid_for(ext), 256>>>(ext, i, c_gc, d_v2.ptr, d_bar_Sab);
            if (!diag) {
              k_add_row_scaled_vec<<<grid_for(ext), 256>>>(ext, j, c_gc, d_v1.ptr, d_bar_Sab);
              k_add_col_scaled_vec<<<grid_for(ext), 256>>>(ext, ext, i, c_gc, d_v2.ptr, d_bar_Sab);
            }
          }
        }
      } else if (op_kind == 1) {
        // double_commutator (dc)
        const double bar_B_d_ss = (!diag) ? ((1.0 / 16.0) * s_f12) : 0.0;
        const double bar_B_d_os = (diag ? 0.25 : (5.0 / 32.0)) * s_f12;
        const double bar_B_x_os = (!diag) ? ((3.0 / 32.0) * s_f12) : 0.0;
        if (!diag) {
          k_robust_elem_pullback<<<grid_for(aux), 256>>>(bar_B_d_ss, i, j, aux, ext, fit_i, raw_i,
                                                         fit_j, res_j, bfit_i, braw_i, bfit_j,
                                                         bres_j);
          k_robust_elem_pullback<<<grid_for(aux), 256>>>(-bar_B_d_ss, j, i, aux, ext, fit_i, raw_i,
                                                         fit_j, res_j, bfit_i, braw_i, bfit_j,
                                                         bres_j);
        }
        if (compute_both_rhf) {
          k_robust_elem_pullback<<<grid_for(aux), 256>>>(spin_scale * bar_B_d_os, i, j, aux, ext,
                                                         fit_i, raw_i, fit_j, res_j, bfit_i, braw_i,
                                                         bfit_j, bres_os_j);
          if (!diag) {
            k_robust_elem_pullback<<<grid_for(aux), 256>>>(bar_B_x_os, j, i, aux, ext, fit_i, raw_i,
                                                           fit_j, res_j, bfit_i, braw_i, bfit_j,
                                                           bres_os_j);
          }
          if (d_bar_Sab) {
            if (int rc = robust_col_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, i, d_v1.ptr);
                rc != 0)
              return rc;
            if (int rc = robust_row_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, j, d_v2.ptr);
                rc != 0)
              return rc;
            const double* u_col_j = d_v1.ptr;
            const double* u_row_i = d_v2.ptr;
            if (!diag) {
              if (int rc = robust_col_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, j, d_v3.ptr);
                  rc != 0)
                return rc;
              if (int rc = robust_row_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, i, d_v4.ptr);
                  rc != 0)
                return rc;
              u_col_j = d_v3.ptr;
              u_row_i = d_v4.ptr;
            }
            if (cudaMemset(d_sab_ij.ptr, 0, ee * sizeof(double)) != cudaSuccess) return 2;
            k_rhf_dc_sab_fold_cols<<<grid_for(ext), 256>>>(ext, i, j, s_f12, d_v1.ptr, u_col_j,
                                                           u_row_i, d_sab_ij.ptr);
            k_rhf_dc_sab_fold_rows<<<grid_for(ext), 256>>>(ext, i, j, s_f12, d_v2.ptr, u_col_j,
                                                           u_row_i, d_sab_ij.ptr);
            if (diag) {
              k_axpy<<<grid_for(ee), 256>>>(ee, 1.0, d_sab_ij.ptr, d_bar_Sab);
            } else {
              k_sym_add_transpose<<<grid_for(ee), 256>>>(ext, 1.0, d_sab_ij.ptr, d_bar_Sab);
            }
          }
        }
      } else {
        // squared_geminal (sq)
        const double bar_B_d_ss = (!diag) ? ((1.0 / 16.0) * s_f12) : 0.0;
        const double bar_B_d_os = (diag ? 0.25 : (5.0 / 32.0)) * s_f12;
        const double bar_B_x_os = (!diag) ? ((3.0 / 32.0) * s_f12) : 0.0;
        const double bar_X_d_ss = (!diag) ? (-(1.0 / 16.0) * s_f12 * occ_sum) : 0.0;
        const double bar_X_x_ss = (!diag) ? ((1.0 / 16.0) * s_f12 * occ_sum) : 0.0;
        const double bar_X_d_os = -(diag ? 0.25 : (5.0 / 32.0)) * s_f12 * occ_sum;
        const double bar_X_x_os = (!diag) ? (-(3.0 / 32.0) * s_f12 * occ_sum) : 0.0;

        // m_row_i -> d_v1, m_col_j -> d_v2, m_row_j -> d_v3 (or d_v1), m_col_i -> d_v4 (or d_v2)
        if (int rc = robust_row_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, i, d_v1.ptr); rc != 0)
          return rc;
        if (int rc = robust_col_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, j, d_v2.ptr); rc != 0)
          return rc;
        const double* m_row_j = d_v1.ptr;
        const double* m_col_i = d_v2.ptr;
        if (!diag) {
          if (int rc = robust_row_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, j, d_v3.ptr);
              rc != 0)
            return rc;
          if (int rc = robust_col_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, i, d_v4.ptr);
              rc != 0)
            return rc;
          m_row_j = d_v3.ptr;
          m_col_i = d_v4.ptr;
        }

        if (!diag) {
          k_robust_elem_pullback<<<grid_for(aux), 256>>>(bar_X_d_ss, i, j, aux, ext, fit_i, raw_i,
                                                         fit_j, res_j, bfit_i, braw_i, bfit_j,
                                                         bres_j);
          k_robust_elem_pullback<<<grid_for(aux), 256>>>(bar_X_x_ss, j, i, aux, ext, fit_i, raw_i,
                                                         fit_j, res_j, bfit_i, braw_i, bfit_j,
                                                         bres_j);
          if (int rc = robust_row_col_pullback_gpu(h, bar_B_d_ss, i, d_hj, bar_B_d_ss, j, d_hi, aux,
                                                   ext, fit_i, raw_i, fit_j, res_j, bfit_i, braw_i,
                                                   bfit_j, bres_j, d_a1.ptr, d_a2.ptr);
              rc != 0)
            return rc;
          if (int rc = robust_row_col_pullback_gpu(h, -bar_B_d_ss, j, d_hi, -bar_B_d_ss, i, d_hj,
                                                   aux, ext, fit_i, raw_i, fit_j, res_j, bfit_i,
                                                   braw_i, bfit_j, bres_j, d_a1.ptr, d_a2.ptr);
              rc != 0)
            return rc;
          // bar_F, bar_K rows i, j
          k_add_scaled_two_vecs_to_row_both<<<grid_for(ext), 256>>>(
              ext, i, bar_B_d_ss, d_v2.ptr, -bar_B_d_ss, m_row_j, d_bar_F, d_bar_K);
          k_add_scaled_two_vecs_to_row_both<<<grid_for(ext), 256>>>(
              ext, j, bar_B_d_ss, d_v1.ptr, -bar_B_d_ss, m_col_i, d_bar_F, d_bar_K);
        }

        if (compute_both_rhf) {
          k_robust_elem_pullback<<<grid_for(aux), 256>>>(spin_scale * bar_X_d_os, i, j, aux, ext,
                                                         fit_i, raw_i, fit_j, res_j, bfit_i, braw_i,
                                                         bfit_j, bres_os_j);
          if (int rc = robust_row_col_pullback_gpu(
                  h, spin_scale * bar_B_d_os, i, d_hj, spin_scale * bar_B_d_os, j, d_hi, aux, ext,
                  fit_i, raw_i, fit_j, res_j, bfit_i, braw_i, bfit_j, bres_os_j, d_a1.ptr,
                  d_a2.ptr);
              rc != 0)
            return rc;
          if (!diag) {
            k_robust_elem_pullback<<<grid_for(aux), 256>>>(bar_X_x_os, j, i, aux, ext, fit_i, raw_i,
                                                           fit_j, res_j, bfit_i, braw_i, bfit_j,
                                                           bres_os_j);
            if (int rc = robust_row_col_pullback_gpu(
                    h, bar_B_x_os, j, d_hi, bar_B_x_os, i, d_hj, aux, ext, fit_i, raw_i, fit_j,
                    res_j, bfit_i, braw_i, bfit_j, bres_os_j, d_a1.ptr, d_a2.ptr);
                rc != 0)
              return rc;
            k_add_scaled_two_vecs_to_col_both<<<grid_for(ext), 256>>>(
                ext, i, bar_B_d_os, d_v2.ptr, bar_B_x_os, m_row_j, d_bar_F, d_bar_K);
            k_add_scaled_two_vecs_to_col_both<<<grid_for(ext), 256>>>(
                ext, j, bar_B_d_os, d_v1.ptr, bar_B_x_os, m_col_i, d_bar_F, d_bar_K);
          } else {
            k_add_scaled_two_vecs_to_col_both<<<grid_for(ext), 256>>>(
                ext, i, 0.125 * s_f12, m_col_i, 0.125 * s_f12, d_v1.ptr, d_bar_F, d_bar_K);
          }

          if (d_bar_Sab) {
            // ff0_hi -> d_v5, ff0T_hi -> d_v6, ff0_hj -> d_v7 (or d_v5), ff0T_hj -> d_v8 (or d_v6)
            if (int rc = robust_rmatvec_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, d_hi, d_v5.ptr,
                                            d_a1.ptr, d_a2.ptr);
                rc != 0)
              return rc;
            if (int rc = robust_lmatvec_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, d_hi, d_v6.ptr,
                                            d_a1.ptr, d_a2.ptr);
                rc != 0)
              return rc;
            const double* ff0_hj = d_v5.ptr;
            const double* ff0T_hj = d_v6.ptr;
            if (!diag) {
              if (int rc = robust_rmatvec_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, d_hj,
                                              d_v7.ptr, d_a1.ptr, d_a2.ptr);
                  rc != 0)
                return rc;
              if (int rc = robust_lmatvec_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, d_hj,
                                              d_v8.ptr, d_a1.ptr, d_a2.ptr);
                  rc != 0)
                return rc;
              ff0_hj = d_v7.ptr;
              ff0T_hj = d_v8.ptr;
            }
            if (cudaMemset(d_sab_ij.ptr, 0, ee * sizeof(double)) != cudaSuccess) return 2;
            double* d_x1 = d_v9.ptr + Size(0) * ext;
            double* d_x3 = d_v9.ptr + Size(1) * ext;
            double* d_y2 = d_v9.ptr + Size(2) * ext;
            double* d_y4 = d_v9.ptr + Size(3) * ext;
            double* d_ff0_x1 = d_v9.ptr + Size(4) * ext;
            double* d_ff0_x3 = d_v9.ptr + Size(5) * ext;
            double* d_ff_ei = d_v9.ptr + Size(6) * ext;
            double* d_ffT_ej = d_v9.ptr + Size(7) * ext;
            double* d_h_ff_ei = d_v9.ptr + Size(8) * ext;
            double* d_h_ffT_ej = d_v9.ptr + Size(9) * ext;
            k_rhf_sq_sab_build_vecs<<<grid_for(ext), 256>>>(
                ext, i, j, s_f12, occ_sum, d_hi, d_hj, m_col_i, d_v2.ptr, d_v1.ptr, m_row_j,
                d_v5.ptr, ff0_hj, d_x1, d_x3, d_y2, d_y4, d_ff0_x1, d_ff0_x3, d_ff_ei, d_ffT_ej);
            if (rm_ger(h, ext, ext, 3.0 / 64.0, d_x1, 1, m_row_j, 1, d_sab_ij.ptr, ext) !=
                    CUBLAS_STATUS_SUCCESS ||
                rm_ger(h, ext, ext, 3.0 / 64.0, m_col_i, 1, d_y2, 1, d_sab_ij.ptr, ext) !=
                    CUBLAS_STATUS_SUCCESS ||
                rm_ger(h, ext, ext, 1.0 / 64.0, d_x3, 1, d_v1.ptr, 1, d_sab_ij.ptr, ext) !=
                    CUBLAS_STATUS_SUCCESS ||
                rm_ger(h, ext, ext, 1.0 / 64.0, d_v2.ptr, 1, d_y4, 1, d_sab_ij.ptr, ext) !=
                    CUBLAS_STATUS_SUCCESS ||
                rm_gemv(h, false, ext, ext, 1.0, d_H, ext, d_ff_ei, 1, 0.0, d_h_ff_ei, 1) !=
                    CUBLAS_STATUS_SUCCESS ||
                rm_gemv(h, false, ext, ext, 1.0, d_H, ext, d_ffT_ej, 1, 0.0, d_h_ffT_ej, 1) !=
                    CUBLAS_STATUS_SUCCESS) {
              return 4;
            }
            k_rhf_sq_sab_fold_cols<<<grid_for(ext), 256>>>(
                ext, i, j, s_f12, occ_sum, d_ff0_x1, d_ff0_x3, d_v5.ptr, d_v6.ptr, d_ff_ei,
                d_h_ff_ei, d_sab_ij.ptr);
            k_rhf_sq_sab_fold_rows<<<grid_for(ext), 256>>>(
                ext, i, j, s_f12, occ_sum, d_v6.ptr, ff0_hj, ff0T_hj, d_ffT_ej, d_h_ffT_ej,
                d_sab_ij.ptr);
            if (diag) {
              k_axpy<<<grid_for(ee), 256>>>(ee, 1.0, d_sab_ij.ptr, d_bar_Sab);
            } else {
              k_sym_add_transpose<<<grid_for(ee), 256>>>(ext, 1.0, d_sab_ij.ptr, d_bar_Sab);
            }
          }
        }
      }
    }
  }
  return (cudaGetLastError() == cudaSuccess) ? 0 : 7;
}

// Stage 1 Reverse for one operator (op_kind: 0=gc, 1=dc, 2=sq) on UHF Opposite-Spin (AB) channel.
int run_ab_stage1_op_rev_gpu(
    cublasHandle_t h, int op_kind, bool has_occ_weights, int oa, int ob, int fa, int fb, int ext,
    int aux, const double* occ_w_a_host, const double* occ_w_b_host, const double* fock_a_host,
    const double* fock_b_host, const double* d_fit_a_oqe, const double* d_fit_b_oqe,
    const double* d_raw_op_a_oqe, const double* d_res_op_b_oqe, const double* d_Ha,
    const double* d_Hb, const double* d_Sab, const double* d_Ha_S, const double* d_Hb_ST,
    double* d_bar_fit_a_oqe, double* d_bar_fit_b_oqe, double* d_bar_raw_op_a_oqe,
    double* d_bar_res_op_b_oqe, double* d_bar_Fa, double* d_bar_Fb, double* d_bar_Ka,
    double* d_bar_Kb, double* d_bar_Sab) {
  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  if (oa_act <= 0 || ob_act <= 0) return 0;
  const Size qe = Size(aux) * ext;

  // Allocate all scratch vectors ONCE before pair loops.
  DeviceBuf d_a, d_v;
  if (d_a.alloc(Size(4) * aux) != cudaSuccess || d_v.alloc(Size(22) * ext) != cudaSuccess) {
    return 2;
  }
  double* d_a1 = d_a.ptr + Size(0) * aux;
  double* d_a2 = d_a.ptr + Size(1) * aux;
  double* d_a3 = d_a.ptr + Size(2) * aux;
  double* d_a4 = d_a.ptr + Size(3) * aux;

  double* d_ei = d_v.ptr + Size(0) * ext;
  double* d_ej = d_v.ptr + Size(1) * ext;
  double* d_vi = d_v.ptr + Size(2) * ext;
  double* d_uj = d_v.ptr + Size(3) * ext;
  double* d_ha_i = d_v.ptr + Size(4) * ext;
  double* d_hb_j = d_v.ptr + Size(5) * ext;
  double* d_ha_uj = d_v.ptr + Size(6) * ext;
  double* d_hb_vi = d_v.ptr + Size(7) * ext;
  double* d_x1 = d_v.ptr + Size(8) * ext;
  double* d_y2 = d_v.ptr + Size(9) * ext;
  double* d_x3 = d_v.ptr + Size(10) * ext;
  double* d_y4 = d_v.ptr + Size(11) * ext;
  double* d_sy_v = d_v.ptr + Size(12) * ext;
  double* d_stx_v = d_v.ptr + Size(13) * ext;
  double* d_mv1 = d_v.ptr + Size(14) * ext;
  double* d_mv2 = d_v.ptr + Size(15) * ext;
  double* d_tmp1 = d_v.ptr + Size(16) * ext;
  double* d_tmp2 = d_v.ptr + Size(17) * ext;
  double* d_ff_ej = d_v.ptr + Size(18) * ext;
  double* d_ff_vi = d_v.ptr + Size(19) * ext;
  double* d_ffT_ei = d_v.ptr + Size(20) * ext;
  double* d_ffT_uj = d_v.ptr + Size(21) * ext;

  for (int i_loc = 0; i_loc < oa_act; ++i_loc) {
    const int i = fa + i_loc;
    const double* fit_i = d_fit_a_oqe + Size(i_loc) * qe;
    const double* raw_i = d_raw_op_a_oqe + Size(i_loc) * qe;
    double* bfit_i = d_bar_fit_a_oqe + Size(i_loc) * qe;
    double* braw_i = d_bar_raw_op_a_oqe + Size(i_loc) * qe;

    k_set_unit_vec<<<grid_for(ext), 256>>>(ext, i, d_ei);
    // v_i = Sab[i, :] (row i of Sab)
    if (cudaMemcpy(d_vi, d_Sab + Size(i) * ext, Size(ext) * sizeof(double),
                   cudaMemcpyDeviceToDevice) != cudaSuccess) {
      return 3;
    }
    if (op_kind == 2) {
      k_extract_col<<<grid_for(ext), 256>>>(ext, ext, i, d_Ha, d_ha_i);
      k_extract_col<<<grid_for(ext), 256>>>(ext, ext, i, d_Hb_ST, d_hb_vi);
    }

    for (int j_loc = 0; j_loc < ob_act; ++j_loc) {
      const int j = fb + j_loc;
      const double* fit_j = d_fit_b_oqe + Size(j_loc) * qe;
      const double* res_j = d_res_op_b_oqe + Size(j_loc) * qe;
      double* bfit_j = d_bar_fit_b_oqe + Size(j_loc) * qe;
      double* bres_j = d_bar_res_op_b_oqe + Size(j_loc) * qe;

      const double s_f12 =
          has_occ_weights ? 0.5 * (occ_w_a_host[i_loc] + occ_w_b_host[j_loc]) : 1.0;
      const double occ_sum = fock_a_host[Size(i) * ext + i] + fock_b_host[Size(j) * ext + j];
      const double bar_linear = 2.0 * s_f12;
      const double bar_norm = -s_f12 * occ_sum;

      k_set_unit_vec<<<grid_for(ext), 256>>>(ext, j, d_ej);
      k_extract_col<<<grid_for(ext), 256>>>(ext, ext, j, d_Sab, d_uj);

      if (op_kind == 0) {
        // 7a. gc0
        k_robust_elem_pullback<<<grid_for(aux), 256>>>(0.375 * bar_linear, i, j, aux, ext, fit_i,
                                                       raw_i, fit_j, res_j, bfit_i, braw_i, bfit_j,
                                                       bres_j);
        if (int rc = robust_bilin_pullback_gpu(h, 0.125 * bar_linear, d_uj, d_vi, aux, ext, fit_i,
                                               raw_i, fit_j, res_j, bfit_i, braw_i, bfit_j, bres_j,
                                               d_a1, d_a2, d_a3, d_a4);
            rc != 0)
          return rc;
        if (d_bar_Sab) {
          if (int rc = robust_rmatvec_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, d_vi, d_tmp1,
                                          d_a1, d_a2);
              rc != 0)
            return rc;
          if (int rc = robust_lmatvec_gpu(h, fit_i, raw_i, fit_j, res_j, aux, ext, d_uj, d_tmp2,
                                          d_a1, d_a2);
              rc != 0)
            return rc;
          k_add_col_scaled_vec<<<grid_for(ext), 256>>>(ext, ext, j, 0.125 * bar_linear, d_tmp1,
                                                       d_bar_Sab);
          k_add_row_scaled_vec<<<grid_for(ext), 256>>>(ext, i, 0.125 * bar_linear, d_tmp2,
                                                       d_bar_Sab);
        }
      } else if (op_kind == 1) {
        // 7b. dc (uf = W(uf0))
        if (int rc = weight_rank1_pullback_gpu(h, 0.375 * s_f12, d_ei, d_ej, d_Sab, aux, ext, fit_i,
                                               raw_i, fit_j, res_j, bfit_i, braw_i, bfit_j, bres_j,
                                               d_bar_Sab, d_sy_v, d_stx_v, d_mv1, d_mv2, d_a1, d_a2,
                                               d_a3, d_a4);
            rc != 0)
          return rc;
        if (int rc = weight_rank1_pullback_gpu(h, 0.125 * s_f12, d_uj, d_vi, d_Sab, aux, ext, fit_i,
                                               raw_i, fit_j, res_j, bfit_i, braw_i, bfit_j, bres_j,
                                               d_bar_Sab, d_sy_v, d_stx_v, d_mv1, d_mv2, d_a1, d_a2,
                                               d_a3, d_a4);
            rc != 0)
          return rc;
        if (d_bar_Sab) {
          if (int rc = weight_rmatvec_gpu(h, d_Sab, fit_i, raw_i, fit_j, res_j, aux, ext, d_vi,
                                          d_mv1, d_tmp1, d_tmp2, d_a1, d_a2);
              rc != 0)
            return rc;
          if (int rc = weight_lmatvec_gpu(h, d_Sab, fit_i, raw_i, fit_j, res_j, aux, ext, d_uj,
                                          d_mv2, d_tmp1, d_tmp2, d_a1, d_a2);
              rc != 0)
            return rc;
          k_add_col_scaled_vec<<<grid_for(ext), 256>>>(ext, ext, j, 0.125 * s_f12, d_mv1,
                                                       d_bar_Sab);
          k_add_row_scaled_vec<<<grid_for(ext), 256>>>(ext, i, 0.125 * s_f12, d_mv2, d_bar_Sab);
        }
      } else {
        // 7c. sq (ff = W(ff0))
        k_extract_col<<<grid_for(ext), 256>>>(ext, ext, j, d_Hb, d_hb_j);
        k_extract_col<<<grid_for(ext), 256>>>(ext, ext, j, d_Ha_S, d_ha_uj);
        k_ab_sq_build_vecs<<<grid_for(ext), 256>>>(ext, i, s_f12, bar_norm, d_ha_i, d_hb_j, d_ha_uj,
                                                   d_hb_vi, d_uj, d_x1, d_y2, d_x3, d_y4);
        if (int rc = weight_rank1_pullback_gpu(h, 0.375, d_x1, d_ej, d_Sab, aux, ext, fit_i, raw_i,
                                               fit_j, res_j, bfit_i, braw_i, bfit_j, bres_j,
                                               d_bar_Sab, d_sy_v, d_stx_v, d_mv1, d_mv2, d_a1, d_a2,
                                               d_a3, d_a4);
            rc != 0)
          return rc;
        if (int rc = weight_rank1_pullback_gpu(h, 0.375, d_ei, d_y2, d_Sab, aux, ext, fit_i, raw_i,
                                               fit_j, res_j, bfit_i, braw_i, bfit_j, bres_j,
                                               d_bar_Sab, d_sy_v, d_stx_v, d_mv1, d_mv2, d_a1, d_a2,
                                               d_a3, d_a4);
            rc != 0)
          return rc;
        if (int rc = weight_rank1_pullback_gpu(h, 0.125, d_x3, d_vi, d_Sab, aux, ext, fit_i, raw_i,
                                               fit_j, res_j, bfit_i, braw_i, bfit_j, bres_j,
                                               d_bar_Sab, d_sy_v, d_stx_v, d_mv1, d_mv2, d_a1, d_a2,
                                               d_a3, d_a4);
            rc != 0)
          return rc;
        if (int rc = weight_rank1_pullback_gpu(h, 0.125, d_uj, d_y4, d_Sab, aux, ext, fit_i, raw_i,
                                               fit_j, res_j, bfit_i, braw_i, bfit_j, bres_j,
                                               d_bar_Sab, d_sy_v, d_stx_v, d_mv1, d_mv2, d_a1, d_a2,
                                               d_a3, d_a4);
            rc != 0)
          return rc;

        if (int rc = weight_rmatvec_gpu(h, d_Sab, fit_i, raw_i, fit_j, res_j, aux, ext, d_ej,
                                        d_ff_ej, d_tmp1, d_tmp2, d_a1, d_a2);
            rc != 0)
          return rc;
        if (int rc = weight_rmatvec_gpu(h, d_Sab, fit_i, raw_i, fit_j, res_j, aux, ext, d_vi,
                                        d_ff_vi, d_tmp1, d_tmp2, d_a1, d_a2);
            rc != 0)
          return rc;
        if (int rc = weight_lmatvec_gpu(h, d_Sab, fit_i, raw_i, fit_j, res_j, aux, ext, d_ei,
                                        d_ffT_ei, d_tmp1, d_tmp2, d_a1, d_a2);
            rc != 0)
          return rc;
        if (int rc = weight_lmatvec_gpu(h, d_Sab, fit_i, raw_i, fit_j, res_j, aux, ext, d_uj,
                                        d_ffT_uj, d_tmp1, d_tmp2, d_a1, d_a2);
            rc != 0)
          return rc;

        k_add_scaled_two_vecs_to_col_both<<<grid_for(ext), 256>>>(
            ext, i, 0.375 * s_f12, d_ff_ej, 0.0, d_ff_ej, d_bar_Fa, d_bar_Ka);
        k_add_scaled_two_vecs_to_col_both<<<grid_for(ext), 256>>>(
            ext, j, 0.375 * s_f12, d_ffT_ei, 0.0, d_ffT_ei, d_bar_Fb, d_bar_Kb);
        if (rm_ger(h, ext, ext, 0.125 * s_f12, d_ff_vi, 1, d_uj, 1, d_bar_Fa, ext) !=
                CUBLAS_STATUS_SUCCESS ||
            rm_ger(h, ext, ext, 0.125 * s_f12, d_ff_vi, 1, d_uj, 1, d_bar_Ka, ext) !=
                CUBLAS_STATUS_SUCCESS ||
            rm_ger(h, ext, ext, 0.125 * s_f12, d_ffT_uj, 1, d_vi, 1, d_bar_Fb, ext) !=
                CUBLAS_STATUS_SUCCESS ||
            rm_ger(h, ext, ext, 0.125 * s_f12, d_ffT_uj, 1, d_vi, 1, d_bar_Kb, ext) !=
                CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
        if (d_bar_Sab) {
          if (rm_gemv(h, true, ext, ext, 1.0, d_Ha, ext, d_ff_vi, 1, 0.0, d_mv1, 1) !=
              CUBLAS_STATUS_SUCCESS)
            return 4;
          if (int rc = weight_rmatvec_gpu(h, d_Sab, fit_i, raw_i, fit_j, res_j, aux, ext, d_hb_vi,
                                          d_mv2, d_tmp1, d_tmp2, d_a1, d_a2);
              rc != 0)
            return rc;
          if (rm_gemv(h, true, ext, ext, 1.0, d_Hb, ext, d_ffT_uj, 1, 0.0, d_x1, 1) !=
              CUBLAS_STATUS_SUCCESS)
            return 4;
          if (int rc = weight_lmatvec_gpu(h, d_Sab, fit_i, raw_i, fit_j, res_j, aux, ext, d_ha_uj,
                                          d_y2, d_tmp1, d_tmp2, d_a1, d_a2);
              rc != 0)
            return rc;
          k_ab_sq_bar_sab_bw_col<<<grid_for(ext), 256>>>(ext, j, s_f12, bar_norm, d_ff_vi, d_mv1,
                                                         d_mv2, d_bar_Sab);
          k_ab_sq_bar_sab_bw_row<<<grid_for(ext), 256>>>(ext, i, s_f12, bar_norm, d_ffT_uj, d_x1,
                                                         d_y2, d_bar_Sab);
        }
      }
    }
  }
  return (cudaGetLastError() == cudaSuccess) ? 0 : 7;
}

// -------------------------------------------------------------------------
// Stage 2 Reverse Kernels & Drivers (SS/RHF Spatial + UHF Opposite-Spin AB)
// -------------------------------------------------------------------------

__global__ void k_ss_rhf_ord_only_step(
    int i_loc, int j_loc, int o_act, int o, int f, int obs, int ext, bool compute_both_rhf,
    double wij, Size out_pair_offset, int scalar_ord_idx, const double* __restrict__ F,
    const double* __restrict__ G_vv, double* __restrict__ bar_G_vv,
    double* __restrict__ bar_G_vv_os, double* __restrict__ bar_F,
    double* __restrict__ d_out_scalars, double* __restrict__ d_pair_ord, int* __restrict__ d_err) {
  __shared__ double s0[256];
  __shared__ double s1[256];
  __shared__ double s2[256];
  const int v = obs - o;
  const int i = f + i_loc;
  const int j = f + j_loc;
  const bool diag = (i_loc == j_loc);
  const double spin_scale = (compute_both_rhf && diag) ? 0.5 : 1.0;
  const double occ_sum = F[Size(i) * ext + i] + F[Size(j) * ext + j];
  const Size vv = Size(v) * v;

  double ss_gg = 0.0, os_gg = 0.0, sum_bgap = 0.0;
  for (Size idx = threadIdx.x; idx < vv; idx += blockDim.x) {
    const int a = static_cast<int>(idx / v);
    const int b = static_cast<int>(idx % v);
    const double gap = F[Size(o + a) * ext + (o + a)] + F[Size(o + b) * ext + (o + b)] - occ_sum;
    if (!isfinite(gap) || gap == 0.0) {
      atomicExch(d_err, -4);
      continue;
    }
    const double inv_g = 1.0 / gap;
    const double gab = G_vv[idx];
    if (diag) {
      os_gg += gab * gab * inv_g;
      bar_G_vv[idx] = -2.0 * wij * gab * inv_g;
      const double bgap = wij * gab * gab * inv_g * inv_g;
      atomicAdd(&bar_F[Size(o + a) * ext + (o + a)], spin_scale * bgap);
      atomicAdd(&bar_F[Size(o + b) * ext + (o + b)], spin_scale * bgap);
      sum_bgap += bgap;
    } else {
      const double gba = G_vv[Size(b) * v + a];
      const double ag = gab - gba;
      ss_gg += ag * ag * inv_g;
      bar_G_vv[idx] = -2.0 * wij * ag * inv_g;
      double bgap = 0.5 * wij * ag * ag * inv_g * inv_g;
      if (compute_both_rhf) {
        os_gg += gab * gab * inv_g;
        bar_G_vv_os[idx] = -2.0 * wij * gab * inv_g;
        bgap += wij * gab * gab * inv_g * inv_g;
      }
      atomicAdd(&bar_F[Size(o + a) * ext + (o + a)], bgap);
      atomicAdd(&bar_F[Size(o + b) * ext + (o + b)], bgap);
      sum_bgap += bgap;
    }
  }
  s0[threadIdx.x] = ss_gg;
  s1[threadIdx.x] = os_gg;
  s2[threadIdx.x] = sum_bgap;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s0[threadIdx.x] += s0[threadIdx.x + stride];
      s1[threadIdx.x] += s1[threadIdx.x + stride];
      s2[threadIdx.x] += s2[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double bar_occ_sum = -s2[0];
    if (diag) {
      bar_F[Size(i) * ext + i] += spin_scale * 2.0 * bar_occ_sum;
      const double eo_os = -s1[0];
      const Size n_aa = (o_act >= 2) ? (Size(o_act) * (o_act - 1)) / 2 : 0;
      const Size ab_idx = n_aa + Size(i_loc) * o_act + i_loc;
      if (d_pair_ord) d_pair_ord[ab_idx] = eo_os;
      d_out_scalars[4] += wij * eo_os;
    } else {
      bar_F[Size(i) * ext + i] += bar_occ_sum;
      bar_F[Size(j) * ext + j] += bar_occ_sum;
      const double eo_ss = -0.5 * s0[0];
      const double eo_os = compute_both_rhf ? (-s1[0]) : 0.0;
      if (compute_both_rhf) {
        const Size n_aa = (Size(o_act) * (o_act - 1)) / 2;
        const Size n_ab = Size(o_act) * o_act;
        const Size tri = strict_tri_index(i_loc, j_loc, o_act);
        if (d_pair_ord) {
          d_pair_ord[tri] = eo_ss;
          d_pair_ord[n_aa + Size(i_loc) * o_act + j_loc] = eo_os;
          d_pair_ord[n_aa + Size(j_loc) * o_act + i_loc] = eo_os;
          d_pair_ord[n_aa + n_ab + tri] = eo_ss;
        }
        d_out_scalars[3] += wij * eo_ss;
        d_out_scalars[5] += wij * eo_ss;
        d_out_scalars[4] += 2.0 * wij * eo_os;
      } else {
        if (d_pair_ord) d_pair_ord[out_pair_offset + strict_tri_index(i_loc, j_loc, o_act)] = eo_ss;
        d_out_scalars[scalar_ord_idx] += wij * eo_ss;
      }
    }
  }
}

__global__ void k_rhf_diag_rev_step1_denom(
    int i_loc, int o_act, int o, int f, int obs, int ext, bool include_ordinary, double wij,
    const double* __restrict__ F, const double* __restrict__ d_uf_d,
    const double* __restrict__ d_sq_fk_d, const double* __restrict__ d_vx,
    const double* __restrict__ d_b_ov, const double* __restrict__ G_obs,
    const double* __restrict__ C_d, double* __restrict__ bar_C_d, double* __restrict__ bar_C_os,
    double* __restrict__ bar_G_obs, double* __restrict__ bar_F, double* __restrict__ d_out_scalars,
    double* __restrict__ d_pair_f12, double* __restrict__ d_pair_ord, int* __restrict__ d_err) {
  __shared__ double s0[256];
  __shared__ double s1[256];
  __shared__ double s2[256];
  __shared__ double s3[256];

  const int v = obs - o;
  const int i = f + i_loc;
  const double s_f12 = wij;
  const double s_ord = include_ordinary ? wij : 0.0;
  const double spin_scale = 0.5;
  const double occ_sum = 2.0 * F[Size(i) * ext + i];
  const Size vv = Size(v) * v;

  double vd_cg = 0.0, bd_cc = 0.0, os_gg = 0.0, sum_bgap = 0.0;
  for (Size idx = threadIdx.x; idx < vv; idx += blockDim.x) {
    const int a = static_cast<int>(idx / v);
    const int b = static_cast<int>(idx % v);
    const double gap = F[Size(o + a) * ext + (o + a)] + F[Size(o + b) * ext + (o + b)] - occ_sum;
    if (!isfinite(gap) || gap == 0.0) {
      atomicExch(d_err, -4);
      continue;
    }
    const double inv_g = 1.0 / gap;
    const double gab = G_obs[Size(o + a) * obs + (o + b)];
    const double cab = C_d[idx];
    vd_cg += cab * gab * inv_g;
    bd_cc += cab * cab * inv_g;
    if (include_ordinary) os_gg += gab * gab * inv_g;

    const double c_os = 0.5 * cab;
    const double bc_os = -2.0 * s_f12 * (gab + c_os) * inv_g;
    bar_C_os[idx] = bc_os;
    bar_C_d[idx] = 0.5 * bc_os;
    bar_G_obs[Size(o + a) * obs + (o + b)] = (-s_f12 * cab - 2.0 * s_ord * gab) * inv_g;
    const double bar_inv_g = -s_f12 * (cab * gab + 0.25 * cab * cab) - s_ord * gab * gab;
    const double bar_gap = -bar_inv_g * inv_g * inv_g;
    atomicAdd(&bar_F[Size(o + a) * ext + (o + a)], spin_scale * bar_gap);
    atomicAdd(&bar_F[Size(o + b) * ext + (o + b)], spin_scale * bar_gap);
    sum_bgap += bar_gap;
  }
  s0[threadIdx.x] = vd_cg;
  s1[threadIdx.x] = bd_cc;
  s2[threadIdx.x] = os_gg;
  s3[threadIdx.x] = sum_bgap;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s0[threadIdx.x] += s0[threadIdx.x + stride];
      s1[threadIdx.x] += s1[threadIdx.x + stride];
      s2[threadIdx.x] += s2[threadIdx.x + stride];
      s3[threadIdx.x] += s3[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int pid = i_loc * o_act + i_loc;
    const double V_d = d_vx[0];
    const double X_d = d_vx[2];
    const double B_d = d_uf_d[pid] + d_sq_fk_d[pid] + 2.0 * d_b_ov[0];
    const double ef_os = (V_d - s0[0]) + 0.25 * (B_d - occ_sum * X_d - s1[0]);
    const double eo_os = include_ordinary ? (-s2[0]) : 0.0;
    const double bar_occ_sum = -0.25 * s_f12 * X_d - s3[0];
    bar_F[Size(i) * ext + i] += spin_scale * 2.0 * bar_occ_sum;
    const Size n_aa = (o_act >= 2) ? (Size(o_act) * (o_act - 1)) / 2 : 0;
    const Size ab_idx = n_aa + Size(i_loc) * o_act + i_loc;
    if (d_pair_f12) d_pair_f12[ab_idx] = ef_os;
    if (include_ordinary && d_pair_ord) d_pair_ord[ab_idx] = eo_os;
    d_out_scalars[1] += wij * ef_os;
    if (include_ordinary) d_out_scalars[4] += wij * eo_os;
  }
}

__global__ void k_ss_rhf_rev_step1_denom(
    int i_loc, int j_loc, int o_act, int o, int f, int obs, int ext, bool compute_both_rhf,
    bool include_ordinary, double wij, Size out_pair_offset, int scalar_f12_idx, int scalar_ord_idx,
    const double* __restrict__ F, const double* __restrict__ d_uf_d,
    const double* __restrict__ d_uf_x, const double* __restrict__ d_sq_fk_d,
    const double* __restrict__ d_sq_fk_x, const double* __restrict__ d_vx,
    const double* __restrict__ d_b_ov, const double* __restrict__ G_obs,
    const double* __restrict__ C_d, double* __restrict__ bar_ag, double* __restrict__ bar_ac,
    double* __restrict__ bar_C_os, double* __restrict__ bar_G_obs_os, double* __restrict__ bar_F,
    double* __restrict__ d_out_scalars, double* __restrict__ d_pair_f12,
    double* __restrict__ d_pair_ord, int* __restrict__ d_err) {
  __shared__ double s0[256];
  __shared__ double s1[256];
  __shared__ double s2[256];
  __shared__ double s3[256];
  __shared__ double s4[256];
  __shared__ double s5[256];

  const int v = obs - o;
  const int i = f + i_loc;
  const int j = f + j_loc;
  const double s_f12 = wij;
  const double s_ord = include_ordinary ? wij : 0.0;
  const double occ_sum = F[Size(i) * ext + i] + F[Size(j) * ext + j];
  const Size vv = Size(v) * v;

  double ss_gc = 0.0, ss_cc = 0.0, ss_gg = 0.0, os_term = 0.0, os_gg = 0.0, sum_bgap = 0.0;
  for (Size idx = threadIdx.x; idx < vv; idx += blockDim.x) {
    const int a = static_cast<int>(idx / v);
    const int b = static_cast<int>(idx % v);
    const double gap = F[Size(o + a) * ext + (o + a)] + F[Size(o + b) * ext + (o + b)] - occ_sum;
    if (!isfinite(gap) || gap == 0.0) {
      atomicExch(d_err, -4);
      continue;
    }
    const double inv_g = 1.0 / gap;
    const double gab = G_obs[Size(o + a) * obs + (o + b)];
    const double gba = G_obs[Size(o + b) * obs + (o + a)];
    const double cab = C_d[idx];
    const double cba = C_d[Size(b) * v + a];
    const double ag = gab - gba;
    const double ac = cab - cba;
    ss_gc += ag * ac * inv_g;
    ss_cc += ac * ac * inv_g;
    if (include_ordinary) ss_gg += ag * ag * inv_g;

    bar_ag[idx] = (-0.25 * s_f12 * ac - s_ord * ag) * inv_g;
    bar_ac[idx] = (-0.25 * s_f12 * ag - (1.0 / 16.0) * s_f12 * ac) * inv_g;
    double bar_inv_g =
        -0.25 * s_f12 * ag * ac - (1.0 / 32.0) * s_f12 * ac * ac - 0.5 * s_ord * ag * ag;
    if (compute_both_rhf) {
      const double c_os = 0.375 * cab + 0.125 * cba;
      os_term += c_os * (2.0 * gab + c_os) * inv_g;
      if (include_ordinary) os_gg += gab * gab * inv_g;
      bar_C_os[idx] = -2.0 * s_f12 * (gab + c_os) * inv_g;
      bar_G_obs_os[Size(o + a) * obs + (o + b)] = (-2.0 * s_f12 * c_os - 2.0 * s_ord * gab) * inv_g;
      bar_inv_g += -s_f12 * c_os * (2.0 * gab + c_os) - s_ord * gab * gab;
    }
    const double bar_gap = -bar_inv_g * inv_g * inv_g;
    atomicAdd(&bar_F[Size(o + a) * ext + (o + a)], bar_gap);
    atomicAdd(&bar_F[Size(o + b) * ext + (o + b)], bar_gap);
    sum_bgap += bar_gap;
  }
  s0[threadIdx.x] = ss_gc;
  s1[threadIdx.x] = ss_cc;
  s2[threadIdx.x] = ss_gg;
  s3[threadIdx.x] = os_term;
  s4[threadIdx.x] = os_gg;
  s5[threadIdx.x] = sum_bgap;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s0[threadIdx.x] += s0[threadIdx.x + stride];
      s1[threadIdx.x] += s1[threadIdx.x + stride];
      s2[threadIdx.x] += s2[threadIdx.x + stride];
      s3[threadIdx.x] += s3[threadIdx.x + stride];
      s4[threadIdx.x] += s4[threadIdx.x + stride];
      s5[threadIdx.x] += s5[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int pid = i_loc * o_act + j_loc;
    const double V_d = d_vx[0];
    const double V_x = d_vx[1];
    const double X_d = d_vx[2];
    const double X_x = d_vx[3];
    const double B_d = d_uf_d[pid] + d_sq_fk_d[pid] + d_b_ov[0];
    const double B_x = d_uf_x[pid] + d_sq_fk_x[pid] + d_b_ov[1];
    double bar_occ_sum = -(1.0 / 16.0) * s_f12 * (X_d - X_x) - s5[0];
    if (compute_both_rhf) {
      bar_occ_sum -= s_f12 * ((5.0 / 32.0) * X_d + (3.0 / 32.0) * X_x);
    }
    bar_F[Size(i) * ext + i] += bar_occ_sum;
    bar_F[Size(j) * ext + j] += bar_occ_sum;

    const double lin_ss = (V_d - V_x) - 0.5 * s0[0];
    const double quad_ss = ((B_d - B_x) - occ_sum * (X_d - X_x)) - 0.5 * s1[0];
    const double ef_ss = 0.5 * lin_ss + (1.0 / 16.0) * quad_ss;
    const double eo_ss = include_ordinary ? (-0.5 * s2[0]) : 0.0;
    if (compute_both_rhf) {
      const double lin_os = 0.375 * V_d + 0.125 * V_x;
      const double norm_os = (5.0 / 32.0) * X_d + (3.0 / 32.0) * X_x;
      const double b_os = (5.0 / 32.0) * B_d + (3.0 / 32.0) * B_x;
      const double ef_os = 2.0 * lin_os + b_os - occ_sum * norm_os - s3[0];
      const double eo_os = include_ordinary ? (-s4[0]) : 0.0;
      const Size n_aa = (Size(o_act) * (o_act - 1)) / 2;
      const Size n_ab = Size(o_act) * o_act;
      const Size tri = strict_tri_index(i_loc, j_loc, o_act);
      if (d_pair_f12) {
        d_pair_f12[tri] = ef_ss;
        d_pair_f12[n_aa + Size(i_loc) * o_act + j_loc] = ef_os;
        d_pair_f12[n_aa + Size(j_loc) * o_act + i_loc] = ef_os;
        d_pair_f12[n_aa + n_ab + tri] = ef_ss;
      }
      if (include_ordinary && d_pair_ord) {
        d_pair_ord[tri] = eo_ss;
        d_pair_ord[n_aa + Size(i_loc) * o_act + j_loc] = eo_os;
        d_pair_ord[n_aa + Size(j_loc) * o_act + i_loc] = eo_os;
        d_pair_ord[n_aa + n_ab + tri] = eo_ss;
      }
      d_out_scalars[0] += wij * ef_ss;
      d_out_scalars[2] += wij * ef_ss;
      d_out_scalars[1] += 2.0 * wij * ef_os;
      if (include_ordinary) {
        d_out_scalars[3] += wij * eo_ss;
        d_out_scalars[5] += wij * eo_ss;
        d_out_scalars[4] += 2.0 * wij * eo_os;
      }
    } else {
      const Size idx = out_pair_offset + strict_tri_index(i_loc, j_loc, o_act);
      if (d_pair_f12) d_pair_f12[idx] = ef_ss;
      if (include_ordinary && d_pair_ord) d_pair_ord[idx] = eo_ss;
      d_out_scalars[scalar_f12_idx] += wij * ef_ss;
      if (include_ordinary) d_out_scalars[scalar_ord_idx] += wij * eo_ss;
    }
  }
}

__global__ void k_ss_rhf_rev_step1_fold(int o, int obs, bool compute_both_rhf,
                                        const double* __restrict__ bar_ag,
                                        const double* __restrict__ bar_ac,
                                        const double* __restrict__ bar_C_os,
                                        double* __restrict__ bar_G_obs,
                                        double* __restrict__ bar_C_d,
                                        double* __restrict__ bar_C_ss) {
  const int v = obs - o;
  const Size vv = Size(v) * v;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= vv) return;
  const int a = static_cast<int>(idx / v);
  const int b = static_cast<int>(idx % v);
  const Size ba = Size(b) * v + a;
  bar_G_obs[Size(o + a) * obs + (o + b)] = bar_ag[idx] - bar_ag[ba];
  const double bcss = bar_ac[idx] - bar_ac[ba];
  bar_C_ss[idx] = bcss;
  double bcd = bcss;
  if (compute_both_rhf) bcd += 0.375 * bar_C_os[idx] + 0.125 * bar_C_os[ba];
  bar_C_d[idx] = bcd;
}

__global__ void k_rhf_bar_f_os_p_blocks(int o, int obs, int ext, double s_f12, double occ_sum,
                                        const double* __restrict__ G_obs,
                                        const double* __restrict__ G_oc,
                                        const double* __restrict__ G_co,
                                        const double* __restrict__ f_os,
                                        double* __restrict__ bar_f_os) {
  const int c = ext - obs;
  const Size n_obs2 = Size(obs) * obs;
  const Size n_oc = Size(o) * c;
  const Size total = n_obs2 + 2 * n_oc;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const double n_coef = 2.0 * s_f12 * occ_sum;
  if (idx < n_obs2) {
    const int p = static_cast<int>(idx / obs);
    const int r = static_cast<int>(idx % obs);
    const Size pr = Size(p) * ext + r;
    bar_f_os[pr] += -2.0 * s_f12 * G_obs[idx] + n_coef * f_os[pr];
  } else if (idx < n_obs2 + n_oc) {
    const Size rem = idx - n_obs2;
    const int p = static_cast<int>(rem / c);
    const int q = static_cast<int>(rem % c);
    const Size pq = Size(p) * ext + (obs + q);
    bar_f_os[pq] += -2.0 * s_f12 * G_oc[rem] + n_coef * f_os[pq];
  } else {
    const Size rem = idx - n_obs2 - n_oc;
    const int p = static_cast<int>(rem / o);
    const int q = static_cast<int>(rem % o);
    const Size pq = Size(obs + p) * ext + q;
    bar_f_os[pq] += -2.0 * s_f12 * G_co[rem] + n_coef * f_os[pq];
  }
}

__global__ void k_ss_rhf_rev_step4_vx(
    int o, int obs, int ext, bool diag, bool split_rhf_offdiag, double bar_V_d, double bar_V_x,
    double bar_V_d_ss, double bar_V_x_ss, double bar_V_d_os, double bar_V_x_os, double bar_X_d_ss,
    double bar_X_x_ss, const double* __restrict__ f0, const double* __restrict__ ft,
    const double* __restrict__ G_obs, const double* __restrict__ G_oc,
    const double* __restrict__ G_co, double* __restrict__ bar_G_obs, double* __restrict__ bar_G_oc,
    double* __restrict__ bar_G_co, double* __restrict__ bar_G_obs_os,
    double* __restrict__ bar_G_oc_os, double* __restrict__ bar_G_co_os,
    double* __restrict__ bar_f0, double* __restrict__ bar_ft) {
  const int c = ext - obs;
  const Size n_obs2 = Size(obs) * obs;
  const Size n_oc = Size(o) * c;
  const Size total = n_obs2 + 2 * n_oc;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  if (idx < n_obs2) {
    const int p = static_cast<int>(idx / obs);
    const int r = static_cast<int>(idx % obs);
    const Size pr = Size(p) * ext + r;
    const double fp = f0[pr];
    const double ftp = diag ? fp : ft[pr];
    const double gpr = G_obs[idx];
    if (split_rhf_offdiag) {
      bar_G_obs[idx] -= bar_V_d_ss * fp + bar_V_x_ss * ftp;
      bar_G_obs_os[idx] -= bar_V_d_os * fp + bar_V_x_os * ftp;
    } else {
      bar_G_obs[idx] -= bar_V_d * fp + bar_V_x * ftp;
    }
    if (!diag) {
      bar_f0[pr] -= bar_V_d_ss * gpr + 2.0 * bar_X_d_ss * fp + bar_X_x_ss * ftp;
      bar_ft[pr] -= bar_V_x_ss * gpr + bar_X_x_ss * fp;
    }
  } else if (idx < n_obs2 + n_oc) {
    const Size rem = idx - n_obs2;
    const int p = static_cast<int>(rem / c);
    const int q = static_cast<int>(rem % c);
    const Size pq = Size(p) * ext + (obs + q);
    const double fp = f0[pq];
    const double ftp = diag ? fp : ft[pq];
    const double gpq = G_oc[rem];
    if (split_rhf_offdiag) {
      bar_G_oc[rem] -= bar_V_d_ss * fp + bar_V_x_ss * ftp;
      bar_G_oc_os[rem] -= bar_V_d_os * fp + bar_V_x_os * ftp;
    } else {
      bar_G_oc[rem] -= bar_V_d * fp + bar_V_x * ftp;
    }
    if (!diag) {
      bar_f0[pq] -= bar_V_d_ss * gpq + 2.0 * bar_X_d_ss * fp + bar_X_x_ss * ftp;
      bar_ft[pq] -= bar_V_x_ss * gpq + bar_X_x_ss * fp;
    }
  } else {
    const Size rem = idx - n_obs2 - n_oc;
    const int p = static_cast<int>(rem / o);
    const int q = static_cast<int>(rem % o);
    const Size pq = Size(obs + p) * ext + q;
    const double fp = f0[pq];
    const double ftp = diag ? fp : ft[pq];
    const double gpq = G_co[rem];
    if (split_rhf_offdiag) {
      bar_G_co[rem] -= bar_V_d_ss * fp + bar_V_x_ss * ftp;
      bar_G_co_os[rem] -= bar_V_d_os * fp + bar_V_x_os * ftp;
    } else {
      bar_G_co[rem] -= bar_V_d * fp + bar_V_x * ftp;
    }
    if (!diag) {
      bar_f0[pq] -= bar_V_d_ss * gpq + 2.0 * bar_X_d_ss * fp + bar_X_x_ss * ftp;
      bar_ft[pq] -= bar_V_x_ss * gpq + bar_X_x_ss * fp;
    }
  }
}

int run_ss_or_rhf_stage2_response_gpu(
    cublasHandle_t h, bool compute_both_rhf, bool include_f12, bool include_ordinary,
    bool has_occ_weights, int o, int f, int obs, int ext, int aux, Size out_pair_offset,
    int scalar_f12_idx, int scalar_ord_idx, const double* occ_w_host, const double* fock_host,
    const double* d_fit_oqe, const double* d_raw_coul_oqe, const double* d_raw_g_oqe,
    const double* d_res_g_oqe, const double* d_F, const double* d_K, const double* d_H,
    const double* d_fg_d, const double* d_fg_x, const double* d_f2_d, const double* d_f2_x,
    const double* d_uf_d, const double* d_uf_x, const double* d_sq_fk_d, const double* d_sq_fk_x,
    double* d_out_scalars, double* d_pair_f12, double* d_pair_ord, double* d_bar_fit_oqe,
    double* d_bar_raw_coul_oqe, double* d_bar_raw_g_oqe, double* d_bar_res_g_oqe,
    double* d_bar_res_os_g_oqe, double* d_bar_F, double* d_bar_K, double* d_bar_Sab, int* d_err) {
  const int o_act = o - f;
  if ((compute_both_rhf ? (o_act <= 0) : (o_act < 2)) || (!include_f12 && !include_ordinary)) {
    return 0;
  }
  const int v = obs - o;
  const int c = ext - obs;
  const Size ee = Size(ext) * ext;
  const Size qe = Size(aux) * ext;
  const Size obs2 = Size(obs) * obs;
  const Size oc = Size(o) * c;
  const Size vv = Size(v) * v;

  if (!include_f12) {
    DeviceBuf d_G_vv, d_bar_G_vv, d_bar_G_vv_os;
    if (d_G_vv.alloc(vv) != cudaSuccess || d_bar_G_vv.alloc(vv) != cudaSuccess ||
        (compute_both_rhf && d_bar_G_vv_os.alloc(vv) != cudaSuccess)) {
      return 2;
    }
    for (int i_loc = 0; i_loc < o_act; ++i_loc) {
      const int j_start = compute_both_rhf ? i_loc : (i_loc + 1);
      const double* fit_i = d_fit_oqe + Size(i_loc) * qe;
      const double* raw_coul_i = d_raw_coul_oqe + Size(i_loc) * qe;
      double* bfit_i = d_bar_fit_oqe + Size(i_loc) * qe;
      double* braw_coul_i = d_bar_raw_coul_oqe + Size(i_loc) * qe;
      for (int j_loc = j_start; j_loc < o_act; ++j_loc) {
        const bool diag = (i_loc == j_loc);
        const double wij = has_occ_weights ? 0.5 * (occ_w_host[i_loc] + occ_w_host[j_loc]) : 1.0;
        const double spin_scale = (compute_both_rhf && diag) ? 0.5 : 1.0;
        const double* fit_j = d_fit_oqe + Size(j_loc) * qe;
        const double* raw_coul_j = d_raw_coul_oqe + Size(j_loc) * qe;
        double* bfit_j = d_bar_fit_oqe + Size(j_loc) * qe;
        double* braw_coul_j = d_bar_raw_coul_oqe + Size(j_loc) * qe;
        if (rm_gemm(h, true, false, v, v, aux, 1.0, fit_i + o, ext, raw_coul_j + o, ext, 0.0,
                    d_G_vv.ptr, v) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
        k_ss_rhf_ord_only_step<<<1, 256>>>(
            i_loc, j_loc, o_act, o, f, obs, ext, compute_both_rhf, wij, out_pair_offset,
            scalar_ord_idx, d_F, d_G_vv.ptr, d_bar_G_vv.ptr, d_bar_G_vv_os.ptr, d_bar_F,
            d_out_scalars, d_pair_ord, d_err);
        if (compute_both_rhf && !diag) {
          k_axpy<<<grid_for(vv), 256>>>(vv, 0.5, d_bar_G_vv_os.ptr, d_bar_G_vv.ptr);
          if (rm_gemm(h, false, true, aux, v, v, 1.0, raw_coul_j + o, ext, d_bar_G_vv.ptr, v, 1.0,
                      bfit_i + o, ext) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm(h, false, false, aux, v, v, 1.0, fit_i + o, ext, d_bar_G_vv.ptr, v, 1.0,
                      braw_coul_j + o, ext) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm(h, false, false, aux, v, v, 0.5, raw_coul_i + o, ext, d_bar_G_vv_os.ptr, v,
                      1.0, bfit_j + o, ext) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm(h, false, true, aux, v, v, 0.5, fit_j + o, ext, d_bar_G_vv_os.ptr, v, 1.0,
                      braw_coul_i + o, ext) != CUBLAS_STATUS_SUCCESS) {
            return 4;
          }
        } else {
          if (rm_gemm(h, false, true, aux, v, v, spin_scale, raw_coul_j + o, ext, d_bar_G_vv.ptr, v,
                      1.0, bfit_i + o, ext) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm(h, false, false, aux, v, v, spin_scale, fit_i + o, ext, d_bar_G_vv.ptr, v,
                      1.0, braw_coul_j + o, ext) != CUBLAS_STATUS_SUCCESS) {
            return 4;
          }
        }
      }
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : 7;
  }

  // Allocate ALL single-pair scratch buffers ONCE before the loop (zero inner-loop allocations).
  const Size sub_a_per = std::max<Size>({Size(o) * ext, Size(v) * obs, Size(c) * ext, Size(v) * c});
  const Size sub_l_per = std::max<Size>({Size(o) * ext, Size(c) * o, Size(v) * ext});
  DeviceBuf d_f0, d_ft, d_L0, d_sub_A, d_sub_L, d_G_obs, d_G_oc, d_G_co, d_C_d;
  DeviceBuf d_bar_C_d, d_bar_C_ss, d_bar_C_os, d_bar_ag, d_bar_ac;
  DeviceBuf d_bar_G_obs, d_bar_G_oc, d_bar_G_co, d_bar_G_obs_os, d_bar_G_oc_os, d_bar_G_co_os;
  DeviceBuf d_bar_f0, d_bar_ft, d_diff_f, d_bar_diff;
  DeviceBuf d_f_os, d_qf_os, d_bar_f_os, d_bar_qf_os, d_bar_f0_os, d_ff_mat;
  DeviceBuf d_vx, d_b_ov;

  if (d_f0.alloc(ee) != cudaSuccess || d_ft.alloc(ee) != cudaSuccess ||
      d_L0.alloc(ee) != cudaSuccess || d_sub_A.alloc(sub_a_per) != cudaSuccess ||
      d_sub_L.alloc(sub_l_per) != cudaSuccess || d_G_obs.alloc(obs2) != cudaSuccess ||
      d_G_oc.alloc(oc) != cudaSuccess || d_G_co.alloc(oc) != cudaSuccess ||
      d_C_d.alloc(vv) != cudaSuccess || d_bar_C_d.alloc(vv) != cudaSuccess ||
      d_bar_C_ss.alloc(vv) != cudaSuccess || d_bar_ag.alloc(vv) != cudaSuccess ||
      d_bar_ac.alloc(vv) != cudaSuccess || d_bar_G_obs.alloc(obs2) != cudaSuccess ||
      d_bar_G_oc.alloc(oc) != cudaSuccess || d_bar_G_co.alloc(oc) != cudaSuccess ||
      d_bar_f0.alloc(ee) != cudaSuccess || d_bar_ft.alloc(ee) != cudaSuccess ||
      d_diff_f.alloc(ee) != cudaSuccess || d_bar_diff.alloc(ee) != cudaSuccess ||
      d_vx.alloc(4) != cudaSuccess || d_b_ov.alloc(2) != cudaSuccess) {
    return 2;
  }
  if (compute_both_rhf) {
    if (d_bar_C_os.alloc(vv) != cudaSuccess || d_bar_G_obs_os.alloc(obs2) != cudaSuccess ||
        d_bar_G_oc_os.alloc(oc) != cudaSuccess || d_bar_G_co_os.alloc(oc) != cudaSuccess ||
        d_f_os.alloc(ee) != cudaSuccess || d_qf_os.alloc(ee) != cudaSuccess ||
        d_bar_f_os.alloc(ee) != cudaSuccess || d_bar_qf_os.alloc(ee) != cudaSuccess ||
        d_bar_f0_os.alloc(ee) != cudaSuccess || d_ff_mat.alloc(ee) != cudaSuccess) {
      return 2;
    }
  }

  for (int i_loc = 0; i_loc < o_act; ++i_loc) {
    const int i = f + i_loc;
    const int j_start = compute_both_rhf ? i_loc : (i_loc + 1);
    const double* fit_i = d_fit_oqe + Size(i_loc) * qe;
    const double* raw_coul_i = d_raw_coul_oqe + Size(i_loc) * qe;
    const double* raw_g_i = d_raw_g_oqe + Size(i_loc) * qe;
    double* bfit_i = d_bar_fit_oqe + Size(i_loc) * qe;
    double* braw_coul_i = d_bar_raw_coul_oqe + Size(i_loc) * qe;
    double* braw_g_i = d_bar_raw_g_oqe + Size(i_loc) * qe;

    for (int j_loc = j_start; j_loc < o_act; ++j_loc) {
      const int j = f + j_loc;
      const bool diag = (i_loc == j_loc);
      const bool split_rhf_offdiag = (compute_both_rhf && !diag);
      const double wij = has_occ_weights ? 0.5 * (occ_w_host[i_loc] + occ_w_host[j_loc]) : 1.0;
      const double s_f12 = wij;
      const double spin_scale = (compute_both_rhf && diag) ? 0.5 : 1.0;
      const double occ_sum = fock_host[Size(i) * ext + i] + fock_host[Size(j) * ext + j];

      const double* fit_j = d_fit_oqe + Size(j_loc) * qe;
      const double* raw_coul_j = d_raw_coul_oqe + Size(j_loc) * qe;
      const double* res_g_j = d_res_g_oqe + Size(j_loc) * qe;
      double* bfit_j = d_bar_fit_oqe + Size(j_loc) * qe;
      double* braw_coul_j = d_bar_raw_coul_oqe + Size(j_loc) * qe;
      double* bres_g_j = d_bar_res_g_oqe + Size(j_loc) * qe;
      double* bres_os_g_j = compute_both_rhf ? (d_bar_res_os_g_oqe + Size(j_loc) * qe) : nullptr;

      // Forward 1: f0, ft, Coulomb blocks.
      if (rm_gemm(h, true, false, ext, ext, aux, 1.0, fit_i, ext, res_g_j, ext, 0.0, d_f0.ptr,
                  ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, true, false, ext, ext, aux, 1.0, raw_g_i, ext, fit_j, ext, 1.0, d_f0.ptr,
                  ext) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      if (!diag) launch_transpose_2d(ext, ext, d_f0.ptr, d_ft.ptr);
      const double* d_ft_eff = diag ? d_f0.ptr : d_ft.ptr;

      if (rm_gemm(h, true, false, obs, obs, aux, 1.0, fit_i, ext, raw_coul_j, ext, 0.0, d_G_obs.ptr,
                  obs) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, true, false, o, c, aux, 1.0, fit_i, ext, raw_coul_j + obs, ext, 0.0,
                  d_G_oc.ptr, c) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, true, false, c, o, aux, 1.0, fit_i + obs, ext, raw_coul_j, ext, 0.0,
                  d_G_co.ptr, o) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }

      // Forward 2: V_d, V_x, X_d, X_x (also zeroes d_b_ov[0..1]).
      k_ss_vx_reduce<<<1, 256>>>(i_loc, j_loc, o_act, o, obs, ext, d_fg_d, d_fg_x, d_f2_d, d_f2_x,
                                 d_f0.ptr, d_ft_eff, d_G_obs.ptr, d_G_oc.ptr, d_G_co.ptr, d_vx.ptr,
                                 d_b_ov.ptr);

      // Forward 3: b_overlap (pass=0 only for diag, since diag doubles d_b_ov[0]).
      const int n_passes = diag ? 1 : 2;
      for (int pass = 0; pass < n_passes; ++pass) {
        const double* A_b = (pass == 0) ? d_f0.ptr : d_ft.ptr;
        const double* B_dir = (pass == 0) ? d_f0.ptr : d_ft.ptr;
        const double* B_exc = (pass == 0) ? d_ft_eff : d_f0.ptr;
        if (rm_gemm(h, false, false, ext, ext, ext, 1.0, A_b, ext, d_K, ext, 0.0, d_L0.ptr, ext) !=
            CUBLAS_STATUS_SUCCESS)
          return 4;
        k_accum_dot_pair_rect<<<1, 256>>>(ext, ext, ext, ext, 0, 0, 0, ext, -1.0, d_L0.ptr, B_dir,
                                          B_exc, d_b_ov.ptr);
        launch_pack_rect_batch(1, ext, ext, 0, 0, o, ext, A_b, d_sub_A.ptr);
        if (rm_gemm(h, false, false, o, ext, ext, 1.0, d_sub_A.ptr, ext, d_F, ext, 0.0, d_sub_L.ptr,
                    ext) != CUBLAS_STATUS_SUCCESS)
          return 4;
        k_accum_dot_pair_rect<<<1, 256>>>(o, ext, ext, o, 0, 0, 0, ext, -1.0, d_sub_L.ptr, B_dir,
                                          B_exc, d_b_ov.ptr);
        launch_pack_rect_batch(1, ext, ext, obs, 0, c, o, A_b, d_sub_A.ptr);
        if (rm_gemm(h, false, false, c, o, o, 1.0, d_sub_A.ptr, o, d_F, ext, 0.0, d_sub_L.ptr, o) !=
            CUBLAS_STATUS_SUCCESS)
          return 4;
        k_accum_dot_pair_rect<<<1, 256>>>(c, o, o, c, 0, obs, 0, ext, 1.0, d_sub_L.ptr, B_dir,
                                          B_exc, d_b_ov.ptr);
        launch_pack_rect_batch(1, ext, ext, o, 0, v, obs, A_b, d_sub_A.ptr);
        if (rm_gemm(h, false, false, v, ext, obs, 1.0, d_sub_A.ptr, obs, d_F, ext, 0.0, d_sub_L.ptr,
                    ext) != CUBLAS_STATUS_SUCCESS)
          return 4;
        k_accum_dot_pair_rect<<<1, 256>>>(v, obs, ext, v, 0, o, 0, ext, -1.0, d_sub_L.ptr, B_dir,
                                          B_exc, d_b_ov.ptr);
        k_accum_dot_pair_rect<<<1, 256>>>(v, c, ext, v, obs, o, obs, ext, -2.0, d_sub_L.ptr, B_dir,
                                          B_exc, d_b_ov.ptr);
        launch_pack_rect_batch(1, ext, ext, obs, 0, c, ext, A_b, d_sub_A.ptr);
        if (rm_gemm(h, false, true, c, o, ext, 1.0, d_sub_A.ptr, ext, d_F, ext, 0.0, d_sub_L.ptr,
                    o) != CUBLAS_STATUS_SUCCESS)
          return 4;
        k_accum_dot_pair_rect<<<1, 256>>>(c, o, o, c, 0, obs, 0, ext, -2.0, d_sub_L.ptr, B_dir,
                                          B_exc, d_b_ov.ptr);
      }

      // Forward 4: C_d.
      if (rm_gemm(h, false, true, v, v, c, 1.0, d_f0.ptr + Size(o) * ext + obs, ext,
                  d_F + Size(o) * ext + obs, ext, 0.0, d_C_d.ptr, v) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, v, v, c, 1.0, d_F + Size(o) * ext + obs, ext,
                  d_f0.ptr + Size(obs) * ext + o, ext, 1.0, d_C_d.ptr, v) !=
              CUBLAS_STATUS_SUCCESS) {
        return 4;
      }

      // Zero per-pair bar buffers.
      if (cudaMemset(d_bar_G_obs.ptr, 0, obs2 * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_G_oc.ptr, 0, oc * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_G_co.ptr, 0, oc * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_f0.ptr, 0, ee * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_ft.ptr, 0, ee * sizeof(double)) != cudaSuccess) {
        return 2;
      }
      if (split_rhf_offdiag) {
        if (cudaMemset(d_bar_G_obs_os.ptr, 0, obs2 * sizeof(double)) != cudaSuccess ||
            cudaMemset(d_bar_G_oc_os.ptr, 0, oc * sizeof(double)) != cudaSuccess ||
            cudaMemset(d_bar_G_co_os.ptr, 0, oc * sizeof(double)) != cudaSuccess) {
          return 2;
        }
      }

      // Reverse Step 1: denominator contraction.
      if (diag) {
        k_rhf_diag_rev_step1_denom<<<1, 256>>>(
            i_loc, o_act, o, f, obs, ext, include_ordinary, wij, d_F, d_uf_d, d_sq_fk_d, d_vx.ptr,
            d_b_ov.ptr, d_G_obs.ptr, d_C_d.ptr, d_bar_C_d.ptr, d_bar_C_os.ptr, d_bar_G_obs.ptr,
            d_bar_F, d_out_scalars, d_pair_f12, d_pair_ord, d_err);
      } else {
        k_ss_rhf_rev_step1_denom<<<1, 256>>>(
            i_loc, j_loc, o_act, o, f, obs, ext, compute_both_rhf, include_ordinary, wij,
            out_pair_offset, scalar_f12_idx, scalar_ord_idx, d_F, d_uf_d, d_uf_x, d_sq_fk_d,
            d_sq_fk_x, d_vx.ptr, d_b_ov.ptr, d_G_obs.ptr, d_C_d.ptr, d_bar_ag.ptr, d_bar_ac.ptr,
            d_bar_C_os.ptr, d_bar_G_obs_os.ptr, d_bar_F, d_out_scalars, d_pair_f12, d_pair_ord,
            d_err);
        k_ss_rhf_rev_step1_fold<<<grid_for(vv), 256>>>(
            o, obs, compute_both_rhf, d_bar_ag.ptr, d_bar_ac.ptr, d_bar_C_os.ptr, d_bar_G_obs.ptr,
            d_bar_C_d.ptr, d_bar_C_ss.ptr);
      }

      // Reverse Step 2: Coupling C_d pullback.
      if (!compute_both_rhf) {
        if (rm_gemm(h, false, false, v, c, v, 1.0, d_bar_C_d.ptr, v, d_F + Size(o) * ext + obs, ext,
                    1.0, d_bar_f0.ptr + Size(o) * ext + obs, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, true, false, c, v, v, 1.0, d_F + Size(o) * ext + obs, ext, d_bar_C_d.ptr, v,
                    1.0, d_bar_f0.ptr + Size(obs) * ext + o, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, true, false, v, c, v, 1.0, d_bar_C_d.ptr, v, d_f0.ptr + Size(o) * ext + obs,
                    ext, 1.0, d_bar_F + Size(o) * ext + obs, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, true, v, c, v, 1.0, d_bar_C_d.ptr, v, d_f0.ptr + Size(obs) * ext + o,
                    ext, 1.0, d_bar_F + Size(o) * ext + obs, ext) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
      } else {
        if (rm_gemm(h, true, false, v, c, v, spin_scale, d_bar_C_d.ptr, v,
                    d_f0.ptr + Size(o) * ext + obs, ext, 1.0, d_bar_F + Size(o) * ext + obs,
                    ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, true, v, c, v, spin_scale, d_bar_C_d.ptr, v,
                    d_f0.ptr + Size(obs) * ext + o, ext, 1.0, d_bar_F + Size(o) * ext + obs,
                    ext) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
        if (!diag) {
          if (rm_gemm(h, false, false, v, c, v, 1.0, d_bar_C_ss.ptr, v, d_F + Size(o) * ext + obs,
                      ext, 1.0, d_bar_f0.ptr + Size(o) * ext + obs, ext) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm(h, true, false, c, v, v, 1.0, d_F + Size(o) * ext + obs, ext, d_bar_C_ss.ptr,
                      v, 1.0, d_bar_f0.ptr + Size(obs) * ext + o, ext) != CUBLAS_STATUS_SUCCESS) {
            return 4;
          }
        }
      }

      // Reverse Step 3a: Same-spin B_overlap pullback (for i < j).
      if (!diag && s_f12 != 0.0) {
        const double bar_B_d_ss = (1.0 / 16.0) * s_f12;
        k_lincomb2<<<grid_for(ee), 256>>>(ee, 1.0, d_f0.ptr, -1.0, d_ft.ptr, d_diff_f.ptr);
        if (cudaMemset(d_bar_diff.ptr, 0, ee * sizeof(double)) != cudaSuccess) return 2;
        if (int rc = b_overlap_pullback_gpu(h, bar_B_d_ss, d_diff_f.ptr, d_diff_f.ptr, d_F, d_K, o,
                                            obs, ext, d_bar_diff.ptr, d_bar_diff.ptr, d_bar_F,
                                            d_bar_K);
            rc != 0)
          return rc;
        k_axpy<<<grid_for(ee), 256>>>(ee, 1.0, d_bar_diff.ptr, d_bar_f0.ptr);
        k_axpy<<<grid_for(ee), 256>>>(ee, -1.0, d_bar_diff.ptr, d_bar_ft.ptr);
      }

      // Reverse Step 3b: Closed-shell RHF opposite-spin pullback.
      if (compute_both_rhf) {
        if (diag) {
          k_lincomb2<<<grid_for(ee), 256>>>(ee, 0.5, d_f0.ptr, 0.0, d_f0.ptr, d_f_os.ptr);
        } else {
          k_lincomb2<<<grid_for(ee), 256>>>(ee, 0.375, d_f0.ptr, 0.125, d_ft.ptr, d_f_os.ptr);
        }
        if (cudaMemcpy(d_qf_os.ptr, d_f_os.ptr, ee * sizeof(double), cudaMemcpyDeviceToDevice) !=
            cudaSuccess)
          return 3;
        k_zero_p_blocks<<<grid_for(ee), 256>>>(o, o, obs, ext, d_qf_os.ptr);

        if (rm_gemm(h, false, true, ext, ext, ext, 1.0, d_f_os.ptr, ext, d_f_os.ptr, ext, 0.0,
                    d_ff_mat.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, true, false, ext, ext, ext, 1.0, d_f_os.ptr, ext, d_f_os.ptr, ext, 1.0,
                    d_ff_mat.ptr, ext) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
        k_sub_scale_both<<<grid_for(ee), 256>>>(ee, spin_scale * s_f12, d_ff_mat.ptr, d_bar_K,
                                                d_bar_F);
        if (rm_gemm(h, false, true, ext, ext, ext, spin_scale * s_f12, d_qf_os.ptr, ext,
                    d_qf_os.ptr, ext, 1.0, d_bar_F, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, true, false, ext, ext, ext, spin_scale * s_f12, d_qf_os.ptr, ext,
                    d_qf_os.ptr, ext, 1.0, d_bar_F, ext) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }

        if (cudaMemset(d_bar_f_os.ptr, 0, ee * sizeof(double)) != cudaSuccess) return 2;
        if (rm_gemm(h, false, false, v, c, v, 1.0, d_bar_C_os.ptr, v, d_F + Size(o) * ext + obs,
                    ext, 1.0, d_bar_f_os.ptr + Size(o) * ext + obs, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, true, false, c, v, v, 1.0, d_F + Size(o) * ext + obs, ext, d_bar_C_os.ptr, v,
                    1.0, d_bar_f_os.ptr + Size(obs) * ext + o, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, ext, ext, ext, -2.0 * s_f12, d_H, ext, d_f_os.ptr, ext, 1.0,
                    d_bar_f_os.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, ext, ext, ext, -2.0 * s_f12, d_f_os.ptr, ext, d_H, ext, 1.0,
                    d_bar_f_os.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, ext, ext, ext, 2.0 * s_f12, d_F, ext, d_qf_os.ptr, ext, 0.0,
                    d_bar_qf_os.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, ext, ext, ext, 2.0 * s_f12, d_qf_os.ptr, ext, d_F, ext, 1.0,
                    d_bar_qf_os.ptr, ext) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
        k_zero_p_blocks<<<grid_for(ee), 256>>>(o, o, obs, ext, d_bar_qf_os.ptr);
        k_axpy<<<grid_for(ee), 256>>>(ee, 1.0, d_bar_qf_os.ptr, d_bar_f_os.ptr);
        k_rhf_bar_f_os_p_blocks<<<grid_for(obs2 + 2 * oc), 256>>>(
            o, obs, ext, s_f12, occ_sum, d_G_obs.ptr, d_G_oc.ptr, d_G_co.ptr, d_f_os.ptr,
            d_bar_f_os.ptr);

        k_w_sym_mat<<<grid_for(ee), 256>>>(ext, 0.375, 0.125, d_bar_f_os.ptr, d_bar_f0_os.ptr);
        if (rm_gemm(h, false, true, aux, ext, ext, spin_scale, res_g_j, ext, d_bar_f0_os.ptr, ext,
                    1.0, bfit_i, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, aux, ext, ext, spin_scale, fit_i, ext, d_bar_f0_os.ptr, ext,
                    1.0, bres_os_g_j, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, true, aux, ext, ext, spin_scale, fit_j, ext, d_bar_f0_os.ptr, ext,
                    1.0, braw_g_i, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, aux, ext, ext, spin_scale, raw_g_i, ext, d_bar_f0_os.ptr, ext,
                    1.0, bfit_j, ext) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
        if (d_bar_Sab) {
          // Reuse d_ff_mat for bar_Sab_ij = 0.125 * (bar_f_os @ f0 + f0 @ bar_f_os)
          if (rm_gemm(h, false, false, ext, ext, ext, 0.125, d_bar_f_os.ptr, ext, d_f0.ptr, ext,
                      0.0, d_ff_mat.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm(h, false, false, ext, ext, ext, 0.125, d_f0.ptr, ext, d_bar_f_os.ptr, ext,
                      1.0, d_ff_mat.ptr, ext) != CUBLAS_STATUS_SUCCESS) {
            return 4;
          }
          if (diag) {
            k_axpy<<<grid_for(ee), 256>>>(ee, 1.0, d_ff_mat.ptr, d_bar_Sab);
          } else {
            k_sym_add_transpose<<<grid_for(ee), 256>>>(ext, 1.0, d_ff_mat.ptr, d_bar_Sab);
          }
        }
      }

      // Reverse Step 4: V/X pullback into bar_G and SS bar_f0/bar_ft.
      const double bar_V_d_ss = (!diag) ? (0.5 * s_f12) : 0.0;
      const double bar_V_x_ss = (!diag) ? (-0.5 * s_f12) : 0.0;
      const double bar_V_d_os = compute_both_rhf ? (diag ? s_f12 : (0.75 * s_f12)) : 0.0;
      const double bar_V_x_os = split_rhf_offdiag ? (0.25 * s_f12) : 0.0;
      const double bar_V_d = bar_V_d_ss + bar_V_d_os;
      const double bar_V_x = bar_V_x_ss + bar_V_x_os;
      const double bar_X_d_ss = (!diag) ? (-(1.0 / 16.0) * s_f12 * occ_sum) : 0.0;
      const double bar_X_x_ss = (!diag) ? ((1.0 / 16.0) * s_f12 * occ_sum) : 0.0;

      k_ss_rhf_rev_step4_vx<<<grid_for(obs2 + 2 * oc), 256>>>(
          o, obs, ext, diag, split_rhf_offdiag, bar_V_d, bar_V_x, bar_V_d_ss, bar_V_x_ss,
          bar_V_d_os, bar_V_x_os, bar_X_d_ss, bar_X_x_ss, d_f0.ptr, d_ft_eff, d_G_obs.ptr,
          d_G_oc.ptr, d_G_co.ptr, d_bar_G_obs.ptr, d_bar_G_oc.ptr, d_bar_G_co.ptr,
          d_bar_G_obs_os.ptr, d_bar_G_oc_os.ptr, d_bar_G_co_os.ptr, d_bar_f0.ptr, d_bar_ft.ptr);
      if (!diag) {
        k_add_transpose_scaled<<<grid_for(ee), 256>>>(ext, 1.0, d_bar_ft.ptr, d_bar_f0.ptr);
      }

      // Reverse Step 5: Pullback of SS f0 and G_obs/oc/co into fit/raw/res.
      if (!diag) {
        if (rm_gemm(h, false, true, aux, ext, ext, 1.0, res_g_j, ext, d_bar_f0.ptr, ext, 1.0,
                    bfit_i, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, aux, ext, ext, 1.0, fit_i, ext, d_bar_f0.ptr, ext, 1.0,
                    bres_g_j, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, true, aux, ext, ext, 1.0, fit_j, ext, d_bar_f0.ptr, ext, 1.0,
                    braw_g_i, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, aux, ext, ext, 1.0, raw_g_i, ext, d_bar_f0.ptr, ext, 1.0,
                    bfit_j, ext) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
      }
      if (split_rhf_offdiag) {
        k_axpy<<<grid_for(obs2), 256>>>(obs2, 0.5, d_bar_G_obs_os.ptr, d_bar_G_obs.ptr);
        k_axpy<<<grid_for(oc), 256>>>(oc, 0.5, d_bar_G_oc_os.ptr, d_bar_G_oc.ptr);
        k_axpy<<<grid_for(oc), 256>>>(oc, 0.5, d_bar_G_co_os.ptr, d_bar_G_co.ptr);
        if (rm_gemm(h, false, true, aux, obs, obs, 1.0, raw_coul_j, ext, d_bar_G_obs.ptr, obs, 1.0,
                    bfit_i, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, aux, obs, obs, 1.0, fit_i, ext, d_bar_G_obs.ptr, obs, 1.0,
                    braw_coul_j, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, true, aux, o, c, 1.0, raw_coul_j + obs, ext, d_bar_G_oc.ptr, c, 1.0,
                    bfit_i, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, aux, c, o, 1.0, fit_i, ext, d_bar_G_oc.ptr, c, 1.0,
                    braw_coul_j + obs, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, true, aux, c, o, 1.0, raw_coul_j, ext, d_bar_G_co.ptr, o, 1.0,
                    bfit_i + obs, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, aux, o, c, 1.0, fit_i + obs, ext, d_bar_G_co.ptr, o, 1.0,
                    braw_coul_j, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, aux, obs, obs, 0.5, raw_coul_i, ext, d_bar_G_obs_os.ptr, obs,
                    1.0, bfit_j, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, true, aux, obs, obs, 0.5, fit_j, ext, d_bar_G_obs_os.ptr, obs, 1.0,
                    braw_coul_i, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, aux, o, c, 0.5, raw_coul_i + obs, ext, d_bar_G_co_os.ptr, o,
                    1.0, bfit_j, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, true, aux, c, o, 0.5, fit_j, ext, d_bar_G_co_os.ptr, o, 1.0,
                    braw_coul_i + obs, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, aux, c, o, 0.5, raw_coul_i, ext, d_bar_G_oc_os.ptr, c, 1.0,
                    bfit_j + obs, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, true, aux, o, c, 0.5, fit_j + obs, ext, d_bar_G_oc_os.ptr, c, 1.0,
                    braw_coul_i, ext) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
      } else {
        if (rm_gemm(h, false, true, aux, obs, obs, spin_scale, raw_coul_j, ext, d_bar_G_obs.ptr,
                    obs, 1.0, bfit_i, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, aux, obs, obs, spin_scale, fit_i, ext, d_bar_G_obs.ptr, obs,
                    1.0, braw_coul_j, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, true, aux, o, c, spin_scale, raw_coul_j + obs, ext, d_bar_G_oc.ptr, c,
                    1.0, bfit_i, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, aux, c, o, spin_scale, fit_i, ext, d_bar_G_oc.ptr, c, 1.0,
                    braw_coul_j + obs, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, true, aux, c, o, spin_scale, raw_coul_j, ext, d_bar_G_co.ptr, o, 1.0,
                    bfit_i + obs, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, aux, o, c, spin_scale, fit_i + obs, ext, d_bar_G_co.ptr, o,
                    1.0, braw_coul_j, ext) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
      }
    }
  }
  return (cudaGetLastError() == cudaSuccess) ? 0 : 7;
}

__global__ void k_ab_ord_only_step(
    int i_loc, int j_loc, int ob_act, int oa, int ob, int fa, int fb, int obs, int ext, double wij,
    Size off_ab, const double* __restrict__ Fa, const double* __restrict__ Fb,
    const double* __restrict__ G_vv, double* __restrict__ bar_G_vv, double* __restrict__ bar_Fa,
    double* __restrict__ bar_Fb, double* __restrict__ d_out_scalars, double* __restrict__ d_pair_ord,
    int* __restrict__ d_err) {
  __shared__ double s0[256];
  __shared__ double s1[256];
  const int va = obs - oa;
  const int vb = obs - ob;
  const int i = fa + i_loc;
  const int j = fb + j_loc;
  const double occ_sum = Fa[Size(i) * ext + i] + Fb[Size(j) * ext + j];
  const Size vab = Size(va) * vb;

  double os_gg = 0.0, sum_bgap = 0.0;
  for (Size idx = threadIdx.x; idx < vab; idx += blockDim.x) {
    const int a = static_cast<int>(idx / vb);
    const int b = static_cast<int>(idx % vb);
    const double gap =
        Fa[Size(oa + a) * ext + (oa + a)] + Fb[Size(ob + b) * ext + (ob + b)] - occ_sum;
    if (!isfinite(gap) || gap == 0.0) {
      atomicExch(d_err, -4);
      continue;
    }
    const double inv_g = 1.0 / gap;
    const double gab = G_vv[idx];
    os_gg += gab * gab * inv_g;
    bar_G_vv[idx] = -2.0 * wij * gab * inv_g;
    const double bgap = wij * gab * gab * inv_g * inv_g;
    atomicAdd(&bar_Fa[Size(oa + a) * ext + (oa + a)], bgap);
    atomicAdd(&bar_Fb[Size(ob + b) * ext + (ob + b)], bgap);
    sum_bgap += bgap;
  }
  s0[threadIdx.x] = os_gg;
  s1[threadIdx.x] = sum_bgap;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s0[threadIdx.x] += s0[threadIdx.x + stride];
      s1[threadIdx.x] += s1[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    bar_Fa[Size(i) * ext + i] -= s1[0];
    bar_Fb[Size(j) * ext + j] -= s1[0];
    const double eo = -s0[0];
    if (d_pair_ord) d_pair_ord[off_ab + Size(i_loc) * ob_act + j_loc] = eo;
    d_out_scalars[4] += wij * eo;
  }
}

__global__ void k_ab_rev_step1_denom(
    int i_loc, int j_loc, int ob_act, int oa, int ob, int fa, int fb, int obs, int ext,
    bool include_ordinary, double wij, Size off_ab, const double* __restrict__ Fa,
    const double* __restrict__ Fb, const double* __restrict__ d_lin_norm,
    const double* __restrict__ d_b_acc, const double* __restrict__ G_obs,
    const double* __restrict__ C_vv, double* __restrict__ bar_C_vv, double* __restrict__ bar_G_obs,
    double* __restrict__ bar_Fa, double* __restrict__ bar_Fb, double* __restrict__ d_out_scalars,
    double* __restrict__ d_pair_f12, double* __restrict__ d_pair_ord, int* __restrict__ d_err) {
  __shared__ double s0[256];
  __shared__ double s1[256];
  __shared__ double s2[256];

  const int va = obs - oa;
  const int vb = obs - ob;
  const int i = fa + i_loc;
  const int j = fb + j_loc;
  const double s_f12 = wij;
  const double s_ord = include_ordinary ? wij : 0.0;
  const double occ_sum = Fa[Size(i) * ext + i] + Fb[Size(j) * ext + j];
  const Size vab = Size(va) * vb;

  double os_term = 0.0, os_gg = 0.0, sum_bgap = 0.0;
  for (Size idx = threadIdx.x; idx < vab; idx += blockDim.x) {
    const int a = static_cast<int>(idx / vb);
    const int b = static_cast<int>(idx % vb);
    const double gap =
        Fa[Size(oa + a) * ext + (oa + a)] + Fb[Size(ob + b) * ext + (ob + b)] - occ_sum;
    if (!isfinite(gap) || gap == 0.0) {
      atomicExch(d_err, -4);
      continue;
    }
    const double inv_g = 1.0 / gap;
    const double cab = C_vv[idx];
    const double gab = G_obs[Size(oa + a) * obs + (ob + b)];
    os_term += cab * (2.0 * gab + cab) * inv_g;
    if (include_ordinary) os_gg += gab * gab * inv_g;

    bar_C_vv[idx] = -2.0 * s_f12 * (gab + cab) * inv_g;
    bar_G_obs[Size(oa + a) * obs + (ob + b)] = (-2.0 * s_f12 * cab - 2.0 * s_ord * gab) * inv_g;
    const double bar_inv_g = -s_f12 * cab * (2.0 * gab + cab) - s_ord * gab * gab;
    const double bar_gap = -bar_inv_g * inv_g * inv_g;
    atomicAdd(&bar_Fa[Size(oa + a) * ext + (oa + a)], bar_gap);
    atomicAdd(&bar_Fb[Size(ob + b) * ext + (ob + b)], bar_gap);
    sum_bgap += bar_gap;
  }
  s0[threadIdx.x] = os_term;
  s1[threadIdx.x] = os_gg;
  s2[threadIdx.x] = sum_bgap;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s0[threadIdx.x] += s0[threadIdx.x + stride];
      s1[threadIdx.x] += s1[threadIdx.x + stride];
      s2[threadIdx.x] += s2[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double lin = d_lin_norm[0];
    const double norm = d_lin_norm[1];
    const double b_val = d_b_acc[0];
    const double bar_occ_sum = -s_f12 * norm - s2[0];
    bar_Fa[Size(i) * ext + i] += bar_occ_sum;
    bar_Fb[Size(j) * ext + j] += bar_occ_sum;
    const double ef = 2.0 * lin + b_val - occ_sum * norm - s0[0];
    const double eo = include_ordinary ? (-s1[0]) : 0.0;
    const Size idx = off_ab + Size(i_loc) * ob_act + j_loc;
    if (d_pair_f12) d_pair_f12[idx] = ef;
    if (include_ordinary && d_pair_ord) d_pair_ord[idx] = eo;
    d_out_scalars[1] += wij * ef;
    if (include_ordinary) d_out_scalars[4] += wij * eo;
  }
}

__global__ void k_ab_rev_step4_lin_norm(int oa, int ob, int obs, int ext, double bar_linear,
                                        double bar_norm, const double* __restrict__ f,
                                        const double* __restrict__ G_obs,
                                        const double* __restrict__ G_oc,
                                        const double* __restrict__ G_co,
                                        double* __restrict__ bar_G_obs,
                                        double* __restrict__ bar_G_oc,
                                        double* __restrict__ bar_G_co,
                                        double* __restrict__ bar_f) {
  const int c = ext - obs;
  const Size n_obs2 = Size(obs) * obs;
  const Size n_oc = Size(oa) * c;
  const Size n_co = Size(c) * ob;
  const Size total = n_obs2 + n_oc + n_co;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  if (idx < n_obs2) {
    const int p = static_cast<int>(idx / obs);
    const int r = static_cast<int>(idx % obs);
    const Size pr = Size(p) * ext + r;
    const double fpr = f[pr];
    bar_G_obs[idx] -= bar_linear * fpr;
    bar_f[pr] -= bar_linear * G_obs[idx] + 2.0 * bar_norm * fpr;
  } else if (idx < n_obs2 + n_oc) {
    const Size rem = idx - n_obs2;
    const int p = static_cast<int>(rem / c);
    const int q = static_cast<int>(rem % c);
    const Size pq = Size(p) * ext + (obs + q);
    const double fpq = f[pq];
    bar_G_oc[rem] -= bar_linear * fpq;
    bar_f[pq] -= bar_linear * G_oc[rem] + 2.0 * bar_norm * fpq;
  } else {
    const Size rem = idx - n_obs2 - n_oc;
    const int p = static_cast<int>(rem / ob);
    const int q = static_cast<int>(rem % ob);
    const Size pq = Size(obs + p) * ext + q;
    const double fpq = f[pq];
    bar_G_co[rem] -= bar_linear * fpq;
    bar_f[pq] -= bar_linear * G_co[rem] + 2.0 * bar_norm * fpq;
  }
}

int run_ab_stage2_response_gpu(
    cublasHandle_t h, bool include_f12, bool include_ordinary, bool has_occ_weights, int oa, int ob,
    int fa, int fb, int obs, int ext, int aux, Size off_ab, const double* occ_w_a_host,
    const double* occ_w_b_host, const double* fock_a_host, const double* fock_b_host,
    const double* d_fit_a_oqe, const double* d_fit_b_oqe, const double* d_raw_coul_b_oqe,
    const double* d_raw_g_a_oqe, const double* d_res_g_b_oqe, const double* d_Fa,
    const double* d_Fb, const double* d_Ha, const double* d_Hb, const double* d_Fa_sym,
    const double* d_Fb_sym, const double* d_Ha_sym, const double* d_Hb_sym, const double* d_Sab,
    const double* d_gc_ab, const double* d_dc_ab, const double* d_sq_norm_ab,
    const double* d_sq_fk_ab, double* d_out_scalars, double* d_pair_f12, double* d_pair_ord,
    double* d_bar_fit_a_oqe, double* d_bar_fit_b_oqe, double* d_bar_raw_coul_b_oqe,
    double* d_bar_raw_g_a_oqe, double* d_bar_res_g_b_oqe, double* d_bar_Fa, double* d_bar_Fb,
    double* d_bar_Ka, double* d_bar_Kb, double* d_bar_Sab, int* d_err) {
  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  if (oa_act <= 0 || ob_act <= 0 || (!include_f12 && !include_ordinary)) return 0;

  const int va = obs - oa;
  const int vb = obs - ob;
  const int c = ext - obs;
  const Size ee = Size(ext) * ext;
  const Size qe = Size(aux) * ext;
  const Size obs2 = Size(obs) * obs;
  const Size oac = Size(oa) * c;
  const Size cob = Size(c) * ob;
  const Size vab = Size(va) * vb;

  if (!include_f12) {
    DeviceBuf d_G_vv, d_bar_G_vv;
    if (d_G_vv.alloc(vab) != cudaSuccess || d_bar_G_vv.alloc(vab) != cudaSuccess) return 2;
    for (int i_loc = 0; i_loc < oa_act; ++i_loc) {
      const double* fit_i = d_fit_a_oqe + Size(i_loc) * qe;
      double* bfit_i = d_bar_fit_a_oqe + Size(i_loc) * qe;
      for (int j_loc = 0; j_loc < ob_act; ++j_loc) {
        const double wij =
            has_occ_weights ? 0.5 * (occ_w_a_host[i_loc] + occ_w_b_host[j_loc]) : 1.0;
        const double* raw_coul_j = d_raw_coul_b_oqe + Size(j_loc) * qe;
        double* braw_coul_j = d_bar_raw_coul_b_oqe + Size(j_loc) * qe;
        if (rm_gemm(h, true, false, va, vb, aux, 1.0, fit_i + oa, ext, raw_coul_j + ob, ext, 0.0,
                    d_G_vv.ptr, vb) != CUBLAS_STATUS_SUCCESS)
          return 4;
        k_ab_ord_only_step<<<1, 256>>>(i_loc, j_loc, ob_act, oa, ob, fa, fb, obs, ext, wij, off_ab,
                                       d_Fa, d_Fb, d_G_vv.ptr, d_bar_G_vv.ptr, d_bar_Fa, d_bar_Fb,
                                       d_out_scalars, d_pair_ord, d_err);
        if (rm_gemm(h, false, true, aux, va, vb, 1.0, raw_coul_j + ob, ext, d_bar_G_vv.ptr, vb, 1.0,
                    bfit_i + oa, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, aux, vb, va, 1.0, fit_i + oa, ext, d_bar_G_vv.ptr, vb, 1.0,
                    braw_coul_j + ob, ext) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
      }
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : 7;
  }

  DeviceBuf d_f0, d_tmp_f, d_f, d_qf, d_bar_f, d_bar_qf, d_bar_f0, d_ff_tmp;
  DeviceBuf d_G_obs, d_G_oc, d_G_co, d_bar_G_obs, d_bar_G_oc, d_bar_G_co, d_C_vv, d_bar_C_vv;
  DeviceBuf d_lin_norm, d_b_acc;
  if (d_f0.alloc(ee) != cudaSuccess || d_tmp_f.alloc(ee) != cudaSuccess ||
      d_f.alloc(ee) != cudaSuccess || d_qf.alloc(ee) != cudaSuccess ||
      d_bar_f.alloc(ee) != cudaSuccess || d_bar_qf.alloc(ee) != cudaSuccess ||
      d_bar_f0.alloc(ee) != cudaSuccess || d_ff_tmp.alloc(ee) != cudaSuccess ||
      d_G_obs.alloc(obs2) != cudaSuccess || d_G_oc.alloc(oac) != cudaSuccess ||
      d_G_co.alloc(cob) != cudaSuccess || d_bar_G_obs.alloc(obs2) != cudaSuccess ||
      d_bar_G_oc.alloc(oac) != cudaSuccess || d_bar_G_co.alloc(cob) != cudaSuccess ||
      d_C_vv.alloc(vab) != cudaSuccess || d_bar_C_vv.alloc(vab) != cudaSuccess ||
      d_lin_norm.alloc(2) != cudaSuccess || d_b_acc.alloc(1) != cudaSuccess) {
    return 2;
  }

  for (int i_loc = 0; i_loc < oa_act; ++i_loc) {
    const int i = fa + i_loc;
    const double* fit_i = d_fit_a_oqe + Size(i_loc) * qe;
    const double* raw_g_i = d_raw_g_a_oqe + Size(i_loc) * qe;
    double* bfit_i = d_bar_fit_a_oqe + Size(i_loc) * qe;
    double* braw_g_i = d_bar_raw_g_a_oqe + Size(i_loc) * qe;

    for (int j_loc = 0; j_loc < ob_act; ++j_loc) {
      const int j = fb + j_loc;
      const double wij =
          has_occ_weights ? 0.5 * (occ_w_a_host[i_loc] + occ_w_b_host[j_loc]) : 1.0;
      const double s_f12 = wij;
      const double occ_sum = fock_a_host[Size(i) * ext + i] + fock_b_host[Size(j) * ext + j];
      const double* fit_j = d_fit_b_oqe + Size(j_loc) * qe;
      const double* raw_coul_j = d_raw_coul_b_oqe + Size(j_loc) * qe;
      const double* res_g_j = d_res_g_b_oqe + Size(j_loc) * qe;
      double* bfit_j = d_bar_fit_b_oqe + Size(j_loc) * qe;
      double* braw_coul_j = d_bar_raw_coul_b_oqe + Size(j_loc) * qe;
      double* bres_g_j = d_bar_res_g_b_oqe + Size(j_loc) * qe;

      // Forward Step 2: f0, f = W(f0), qf = Q(f), Coulomb blocks.
      if (rm_gemm(h, true, false, ext, ext, aux, 1.0, fit_i, ext, res_g_j, ext, 0.0, d_f0.ptr,
                  ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, true, false, ext, ext, aux, 1.0, raw_g_i, ext, fit_j, ext, 1.0, d_f0.ptr,
                  ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, true, ext, ext, ext, 1.0, d_Sab, ext, d_f0.ptr, ext, 0.0, d_tmp_f.ptr,
                  ext) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      if (cudaMemcpy(d_f.ptr, d_f0.ptr, ee * sizeof(double), cudaMemcpyDeviceToDevice) !=
          cudaSuccess)
        return 3;
      if (rm_gemm(h, false, false, ext, ext, ext, 0.125, d_tmp_f.ptr, ext, d_Sab, ext, 0.375,
                  d_f.ptr, ext) != CUBLAS_STATUS_SUCCESS)
        return 4;
      if (cudaMemcpy(d_qf.ptr, d_f.ptr, ee * sizeof(double), cudaMemcpyDeviceToDevice) !=
          cudaSuccess)
        return 3;
      k_zero_p_blocks<<<grid_for(ee), 256>>>(oa, ob, obs, ext, d_qf.ptr);

      if (rm_gemm(h, true, false, obs, obs, aux, 1.0, fit_i, ext, raw_coul_j, ext, 0.0, d_G_obs.ptr,
                  obs) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, true, false, oa, c, aux, 1.0, fit_i, ext, raw_coul_j + obs, ext, 0.0,
                  d_G_oc.ptr, c) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, true, false, c, ob, aux, 1.0, fit_i + obs, ext, raw_coul_j, ext, 0.0,
                  d_G_co.ptr, ob) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }

      // Linear, norm, and b_val = dc_ab + sq_fk_ab - <f, Ha f + f Hb^T> + <qf, Fa qf + qf Fb^T>.
      k_ab_lin_norm_init<<<1, 256>>>(i_loc, j_loc, ob_act, oa, ob, obs, ext, d_gc_ab, d_dc_ab,
                                     d_sq_norm_ab, d_sq_fk_ab, d_f.ptr, d_G_obs.ptr, d_G_oc.ptr,
                                     d_G_co.ptr, d_lin_norm.ptr, d_b_acc.ptr);
      if (rm_gemm(h, false, false, ext, ext, ext, 1.0, d_Ha, ext, d_f.ptr, ext, 0.0, d_tmp_f.ptr,
                  ext) != CUBLAS_STATUS_SUCCESS)
        return 4;
      k_accum_dot_single_rect<<<1, 256>>>(ext, ext, ext, ext, 0, 0, ext, ext, 0, 0, -1.0, d_f.ptr,
                                          d_tmp_f.ptr, d_b_acc.ptr);
      if (rm_gemm(h, false, true, ext, ext, ext, 1.0, d_f.ptr, ext, d_Hb, ext, 0.0, d_tmp_f.ptr,
                  ext) != CUBLAS_STATUS_SUCCESS)
        return 4;
      k_accum_dot_single_rect<<<1, 256>>>(ext, ext, ext, ext, 0, 0, ext, ext, 0, 0, -1.0, d_f.ptr,
                                          d_tmp_f.ptr, d_b_acc.ptr);
      if (rm_gemm(h, false, false, ext, ext, ext, 1.0, d_Fa, ext, d_qf.ptr, ext, 0.0, d_tmp_f.ptr,
                  ext) != CUBLAS_STATUS_SUCCESS)
        return 4;
      k_accum_dot_single_rect<<<1, 256>>>(ext, ext, ext, ext, 0, 0, ext, ext, 0, 0, 1.0, d_qf.ptr,
                                          d_tmp_f.ptr, d_b_acc.ptr);
      if (rm_gemm(h, false, true, ext, ext, ext, 1.0, d_qf.ptr, ext, d_Fb, ext, 0.0, d_tmp_f.ptr,
                  ext) != CUBLAS_STATUS_SUCCESS)
        return 4;
      k_accum_dot_single_rect<<<1, 256>>>(ext, ext, ext, ext, 0, 0, ext, ext, 0, 0, 1.0, d_qf.ptr,
                                          d_tmp_f.ptr, d_b_acc.ptr);

      // Forward Step 4: C_vv.
      if (rm_gemm(h, false, false, va, vb, c, 1.0, d_Fa + Size(oa) * ext + obs, ext,
                  d_f.ptr + Size(obs) * ext + ob, ext, 0.0, d_C_vv.ptr, vb) !=
              CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, true, va, vb, c, 1.0, d_f.ptr + Size(oa) * ext + obs, ext,
                  d_Fb + Size(ob) * ext + obs, ext, 1.0, d_C_vv.ptr, vb) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }

      // Reverse U1: energy & denominator contraction.
      if (cudaMemset(d_bar_G_obs.ptr, 0, obs2 * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_G_oc.ptr, 0, oac * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_G_co.ptr, 0, cob * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_f.ptr, 0, ee * sizeof(double)) != cudaSuccess) {
        return 2;
      }
      k_ab_rev_step1_denom<<<1, 256>>>(i_loc, j_loc, ob_act, oa, ob, fa, fb, obs, ext,
                                       include_ordinary, wij, off_ab, d_Fa, d_Fb, d_lin_norm.ptr,
                                       d_b_acc.ptr, d_G_obs.ptr, d_C_vv.ptr, d_bar_C_vv.ptr,
                                       d_bar_G_obs.ptr, d_bar_Fa, d_bar_Fb, d_out_scalars,
                                       d_pair_f12, d_pair_ord, d_err);

      // Reverse U2: C_vv pullback.
      if (rm_gemm(h, false, true, va, c, vb, 1.0, d_bar_C_vv.ptr, vb,
                  d_f.ptr + Size(obs) * ext + ob, ext, 1.0, d_bar_Fa + Size(oa) * ext + obs, ext) !=
              CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, true, false, c, vb, va, 1.0, d_Fa + Size(oa) * ext + obs, ext, d_bar_C_vv.ptr,
                  vb, 1.0, d_bar_f.ptr + Size(obs) * ext + ob, ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, va, c, vb, 1.0, d_bar_C_vv.ptr, vb, d_Fb + Size(ob) * ext + obs,
                  ext, 1.0, d_bar_f.ptr + Size(oa) * ext + obs, ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, true, false, vb, c, va, 1.0, d_bar_C_vv.ptr, vb,
                  d_f.ptr + Size(oa) * ext + obs, ext, 1.0, d_bar_Fb + Size(ob) * ext + obs, ext) !=
              CUBLAS_STATUS_SUCCESS) {
        return 4;
      }

      // Reverse U3: b terms in f and qf.
      if (rm_gemm(h, false, true, ext, ext, ext, 1.0, d_f.ptr, ext, d_f.ptr, ext, 0.0, d_ff_tmp.ptr,
                  ext) != CUBLAS_STATUS_SUCCESS)
        return 4;
      k_sub_scale_both<<<grid_for(ee), 256>>>(ee, s_f12, d_ff_tmp.ptr, d_bar_Ka, d_bar_Fa);
      if (rm_gemm(h, true, false, ext, ext, ext, 1.0, d_f.ptr, ext, d_f.ptr, ext, 0.0, d_ff_tmp.ptr,
                  ext) != CUBLAS_STATUS_SUCCESS)
        return 4;
      k_sub_scale_both<<<grid_for(ee), 256>>>(ee, s_f12, d_ff_tmp.ptr, d_bar_Kb, d_bar_Fb);

      if (rm_gemm(h, false, true, ext, ext, ext, s_f12, d_qf.ptr, ext, d_qf.ptr, ext, 1.0, d_bar_Fa,
                  ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, true, false, ext, ext, ext, s_f12, d_qf.ptr, ext, d_qf.ptr, ext, 1.0, d_bar_Fb,
                  ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, ext, ext, ext, -s_f12, d_Ha_sym, ext, d_f.ptr, ext, 1.0,
                  d_bar_f.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, ext, ext, ext, -s_f12, d_f.ptr, ext, d_Hb_sym, ext, 1.0,
                  d_bar_f.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, ext, ext, ext, s_f12, d_Fa_sym, ext, d_qf.ptr, ext, 0.0,
                  d_bar_qf.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, ext, ext, ext, s_f12, d_qf.ptr, ext, d_Fb_sym, ext, 1.0,
                  d_bar_qf.ptr, ext) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_zero_p_blocks<<<grid_for(ee), 256>>>(oa, ob, obs, ext, d_bar_qf.ptr);
      k_axpy<<<grid_for(ee), 256>>>(ee, 1.0, d_bar_qf.ptr, d_bar_f.ptr);

      // Reverse U4: linear and norm P-block terms.
      k_ab_rev_step4_lin_norm<<<grid_for(obs2 + oac + cob), 256>>>(
          oa, ob, obs, ext, 2.0 * s_f12, -s_f12 * occ_sum, d_f.ptr, d_G_obs.ptr, d_G_oc.ptr,
          d_G_co.ptr, d_bar_G_obs.ptr, d_bar_G_oc.ptr, d_bar_G_co.ptr, d_bar_f.ptr);

      // Reverse U5: f = W(f0) -> bar_f0, bar_Sab.
      if (rm_gemm(h, false, true, ext, ext, ext, 1.0, d_Sab, ext, d_bar_f.ptr, ext, 0.0,
                  d_tmp_f.ptr, ext) != CUBLAS_STATUS_SUCCESS)
        return 4;
      if (cudaMemcpy(d_bar_f0.ptr, d_bar_f.ptr, ee * sizeof(double), cudaMemcpyDeviceToDevice) !=
          cudaSuccess)
        return 3;
      if (rm_gemm(h, false, false, ext, ext, ext, 0.125, d_tmp_f.ptr, ext, d_Sab, ext, 0.375,
                  d_bar_f0.ptr, ext) != CUBLAS_STATUS_SUCCESS)
        return 4;
      if (d_bar_Sab) {
        if (rm_gemm(h, false, true, ext, ext, ext, 1.0, d_bar_f.ptr, ext, d_Sab, ext, 0.0,
                    d_tmp_f.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, ext, ext, ext, 0.125, d_tmp_f.ptr, ext, d_f0.ptr, ext, 1.0,
                    d_bar_Sab, ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, true, ext, ext, ext, 1.0, d_f0.ptr, ext, d_Sab, ext, 0.0, d_tmp_f.ptr,
                    ext) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, false, ext, ext, ext, 0.125, d_tmp_f.ptr, ext, d_bar_f.ptr, ext, 1.0,
                    d_bar_Sab, ext) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
      }

      // Reverse U6: f0 & G pullback into fit/raw/res.
      if (rm_gemm(h, false, true, aux, ext, ext, 1.0, res_g_j, ext, d_bar_f0.ptr, ext, 1.0, bfit_i,
                  ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, aux, ext, ext, 1.0, fit_i, ext, d_bar_f0.ptr, ext, 1.0, bres_g_j,
                  ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, true, aux, ext, ext, 1.0, fit_j, ext, d_bar_f0.ptr, ext, 1.0, braw_g_i,
                  ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, aux, ext, ext, 1.0, raw_g_i, ext, d_bar_f0.ptr, ext, 1.0, bfit_j,
                  ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, true, aux, obs, obs, 1.0, raw_coul_j, ext, d_bar_G_obs.ptr, obs, 1.0,
                  bfit_i, ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, aux, obs, obs, 1.0, fit_i, ext, d_bar_G_obs.ptr, obs, 1.0,
                  braw_coul_j, ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, true, aux, oa, c, 1.0, raw_coul_j + obs, ext, d_bar_G_oc.ptr, c, 1.0,
                  bfit_i, ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, aux, c, oa, 1.0, fit_i, ext, d_bar_G_oc.ptr, c, 1.0,
                  braw_coul_j + obs, ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, true, aux, c, ob, 1.0, raw_coul_j, ext, d_bar_G_co.ptr, ob, 1.0,
                  bfit_i + obs, ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, aux, ob, c, 1.0, fit_i + obs, ext, d_bar_G_co.ptr, ob, 1.0,
                  braw_coul_j, ext) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
    }
  }
  return (cudaGetLastError() == cudaSuccess) ? 0 : 7;
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

int gqc_f12_mp2_pair_response_host_reference(
    int rhf_spatial_int, int include_f12_int, int include_ordinary_int, int has_occ_weights_int,
    int oa, int ob, int fa, int fb, int obs, int ext, int aux, int raw_coul_n1_a,
    int raw_coul_n1_b, std::uint64_t budget_bytes, const double* occ_w_a, const double* occ_w_b,
    const double* fit_a, const double* fit_b, const double* raw_coul_a, const double* raw_coul_b,
    const double* raw_g_a, const double* raw_g_b, const double* raw_sq_a, const double* raw_sq_b,
    const double* raw_gc_a, const double* raw_gc_b, const double* raw_dc_a, const double* raw_dc_b,
    const double* metric_g, const double* metric_sq, const double* metric_gc,
    const double* metric_dc, const double* fock_a, const double* fock_b, const double* exch_a,
    const double* exch_b, const double* cross_overlap, double* out_scalars, double* pair_f12_out,
    double* pair_ord_out, double* bar_fit_a, double* bar_fit_b, double* bar_raw_coul_a,
    double* bar_raw_coul_b, double* bar_raw_g_a, double* bar_raw_g_b, double* bar_raw_sq_a,
    double* bar_raw_sq_b, double* bar_raw_gc_a, double* bar_raw_gc_b, double* bar_raw_dc_a,
    double* bar_raw_dc_b, double* bar_metric_g, double* bar_metric_sq, double* bar_metric_gc,
    double* bar_metric_dc, double* bar_fock_a, double* bar_fock_b, double* bar_exch_a,
    double* bar_exch_b, double* bar_cross_overlap) {
  try {
  const bool rhf_spatial = (rhf_spatial_int != 0);
  const bool include_f12 = (include_f12_int != 0);
  const bool include_ordinary = (include_ordinary_int != 0);
  const bool has_occ_weights = (has_occ_weights_int != 0);

  if (fa < 0 || fa > oa || oa >= obs || fb < 0 || fb > ob || ob >= obs || obs >= ext || aux < 1) {
    return -1;
  }
  if (rhf_spatial && (oa != ob || fa != fb)) return -1;
  if (!metric_g || !metric_sq || !metric_gc || !metric_dc || !fock_a || !exch_a) return -1;
  if (!rhf_spatial && (!fock_b || !exch_b)) return -1;
  if (!bar_fock_a || !bar_exch_a) return -1;
  if (!rhf_spatial && (!bar_fock_b || !bar_exch_b)) return -1;
  if (include_f12 && (!bar_metric_g || !bar_metric_sq || !bar_metric_gc || !bar_metric_dc)) {
    return -1;
  }

  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  if (has_occ_weights) {
    if (oa_act > 0 && !occ_w_a) return -1;
    if (!rhf_spatial && ob_act > 0 && !occ_w_b) return -1;
  }
  if (oa_act > 0) {
    if (raw_coul_n1_a < oa || !fit_a || !raw_coul_a || !raw_g_a || !raw_sq_a || !raw_gc_a ||
        !raw_dc_a) {
      return -1;
    }
    if (!bar_fit_a || !bar_raw_coul_a) return -1;
    if (include_f12 && (!bar_raw_g_a || !bar_raw_sq_a || !bar_raw_gc_a || !bar_raw_dc_a)) {
      return -1;
    }
  }
  if (!rhf_spatial && ob_act > 0) {
    if (raw_coul_n1_b < ob || !fit_b || !raw_coul_b || !raw_g_b || !raw_sq_b || !raw_gc_b ||
        !raw_dc_b) {
      return -1;
    }
    if (!bar_fit_b || !bar_raw_coul_b) return -1;
    if (include_f12 && (!bar_raw_g_b || !bar_raw_sq_b || !bar_raw_gc_b || !bar_raw_dc_b)) {
      return -1;
    }
  }
  if (!rhf_spatial && oa_act > 0 && ob_act > 0 && !cross_overlap) return -1;
  if (include_f12 && oa_act > 0 && ob_act > 0 && !bar_cross_overlap) return -1;

  const Size qq = Size(aux) * aux;
  const Size ee = Size(ext) * ext;
  const Size qoa_e = Size(aux) * Size(oa) * ext;
  const Size qob_e = Size(aux) * Size(ob) * ext;

  if (!all_finite_host(metric_g, qq) || !all_finite_host(metric_sq, qq) ||
      !all_finite_host(metric_gc, qq) || !all_finite_host(metric_dc, qq) ||
      !all_finite_host(fock_a, ee) || !all_finite_host(exch_a, ee)) {
    return -2;
  }
  if (!rhf_spatial && (!all_finite_host(fock_b, ee) || !all_finite_host(exch_b, ee))) {
    return -2;
  }
  if (!rhf_spatial && oa_act > 0 && ob_act > 0 && !all_finite_host(cross_overlap, ee)) {
    return -2;
  }
  if (has_occ_weights) {
    if (oa_act > 0 && !all_finite_host(occ_w_a, oa_act)) return -2;
    if (!rhf_spatial && ob_act > 0 && !all_finite_host(occ_w_b, ob_act)) return -2;
  }

  const Size n_aa = (oa_act >= 2) ? (Size(oa_act) * (oa_act - 1)) / 2 : 0;
  const Size n_ab = Size(oa_act) * Size(ob_act);
  const Size n_bb = (ob_act >= 2) ? (Size(ob_act) * (ob_act - 1)) / 2 : 0;
  const Size n_total = n_aa + n_ab + n_bb;

  // Zero-initialize all provided output buffers before early exits or accumulation.
  if (out_scalars) std::fill_n(out_scalars, 6, 0.0);
  if (pair_f12_out && n_total > 0) std::fill_n(pair_f12_out, n_total, 0.0);
  if (include_ordinary && pair_ord_out && n_total > 0) std::fill_n(pair_ord_out, n_total, 0.0);
  if (qoa_e > 0) {
    if (bar_fit_a) std::fill_n(bar_fit_a, qoa_e, 0.0);
    if (bar_raw_coul_a) std::fill_n(bar_raw_coul_a, qoa_e, 0.0);
    if (bar_raw_g_a) std::fill_n(bar_raw_g_a, qoa_e, 0.0);
    if (bar_raw_sq_a) std::fill_n(bar_raw_sq_a, qoa_e, 0.0);
    if (bar_raw_gc_a) std::fill_n(bar_raw_gc_a, qoa_e, 0.0);
    if (bar_raw_dc_a) std::fill_n(bar_raw_dc_a, qoa_e, 0.0);
  }
  if (qob_e > 0) {
    if (bar_fit_b && bar_fit_b != bar_fit_a) std::fill_n(bar_fit_b, qob_e, 0.0);
    if (bar_raw_coul_b && bar_raw_coul_b != bar_raw_coul_a) std::fill_n(bar_raw_coul_b, qob_e, 0.0);
    if (bar_raw_g_b && bar_raw_g_b != bar_raw_g_a) std::fill_n(bar_raw_g_b, qob_e, 0.0);
    if (bar_raw_sq_b && bar_raw_sq_b != bar_raw_sq_a) std::fill_n(bar_raw_sq_b, qob_e, 0.0);
    if (bar_raw_gc_b && bar_raw_gc_b != bar_raw_gc_a) std::fill_n(bar_raw_gc_b, qob_e, 0.0);
    if (bar_raw_dc_b && bar_raw_dc_b != bar_raw_dc_a) std::fill_n(bar_raw_dc_b, qob_e, 0.0);
  }
  if (bar_metric_g) std::fill_n(bar_metric_g, qq, 0.0);
  if (bar_metric_sq) std::fill_n(bar_metric_sq, qq, 0.0);
  if (bar_metric_gc) std::fill_n(bar_metric_gc, qq, 0.0);
  if (bar_metric_dc) std::fill_n(bar_metric_dc, qq, 0.0);
  std::fill_n(bar_fock_a, ee, 0.0);
  std::fill_n(bar_exch_a, ee, 0.0);
  if (bar_fock_b && bar_fock_b != bar_fock_a) std::fill_n(bar_fock_b, ee, 0.0);
  if (bar_exch_b && bar_exch_b != bar_exch_a) std::fill_n(bar_exch_b, ee, 0.0);
  if (bar_cross_overlap) std::fill_n(bar_cross_overlap, ee, 0.0);

  if (n_total > 0 && (include_f12 || include_ordinary)) {
    const Size req_bytes = estimate_mp2_pair_response_device_bytes(
        rhf_spatial, include_f12, include_ordinary, pair_f12_out != nullptr,
        pair_ord_out != nullptr, bar_cross_overlap != nullptr, oa, ob, fa, fb, obs, ext, aux);
    if (req_bytes > effective_device_budget(budget_bytes)) return kErrBudgetExceeded;
  }

  // Pack active slices & verify finiteness even when n_total == 0.
  std::vector<double> fit_a_qoe, fit_a_oqe, raw_coul_a_oqe;
  std::vector<double> raw_g_a_qoe, raw_g_a_oqe, raw_sq_a_qoe, raw_sq_a_oqe;
  std::vector<double> raw_gc_a_qoe, raw_gc_a_oqe, raw_dc_a_qoe, raw_dc_a_oqe;
  if (oa_act > 0) {
    if (!pack_active_oqe_checked_host(fit_a, aux, oa, fa, oa_act, ext, fit_a_oqe, &fit_a_qoe) ||
        !pack_active_oqe_checked_host(raw_coul_a, aux, raw_coul_n1_a, fa, oa_act, ext,
                                      raw_coul_a_oqe) ||
        !pack_active_oqe_checked_host(raw_g_a, aux, oa, fa, oa_act, ext, raw_g_a_oqe,
                                      &raw_g_a_qoe) ||
        !pack_active_oqe_checked_host(raw_sq_a, aux, oa, fa, oa_act, ext, raw_sq_a_oqe,
                                      &raw_sq_a_qoe) ||
        !pack_active_oqe_checked_host(raw_gc_a, aux, oa, fa, oa_act, ext, raw_gc_a_oqe,
                                      &raw_gc_a_qoe) ||
        !pack_active_oqe_checked_host(raw_dc_a, aux, oa, fa, oa_act, ext, raw_dc_a_oqe,
                                      &raw_dc_a_qoe)) {
      return -2;
    }
  }
  if (n_total == 0 || (!include_f12 && !include_ordinary)) {
    if (!rhf_spatial && ob_act > 0) {
      std::vector<double> chk;
      if (!pack_active_oqe_checked_host(fit_b, aux, ob, fb, ob_act, ext, chk) ||
          !pack_active_oqe_checked_host(raw_coul_b, aux, raw_coul_n1_b, fb, ob_act, ext, chk) ||
          !pack_active_oqe_checked_host(raw_g_b, aux, ob, fb, ob_act, ext, chk) ||
          !pack_active_oqe_checked_host(raw_sq_b, aux, ob, fb, ob_act, ext, chk) ||
          !pack_active_oqe_checked_host(raw_gc_b, aux, ob, fb, ob_act, ext, chk) ||
          !pack_active_oqe_checked_host(raw_dc_b, aux, ob, fb, ob_act, ext, chk)) {
        return -2;
      }
    }
    return 0;
  }

  const Size qe = Size(aux) * ext;
  const Size oaqe = Size(oa_act) * qe;
  const Size obqe = Size(ob_act) * qe;
  double e_f12_aa = 0.0, e_f12_ab = 0.0, e_f12_bb = 0.0;
  double e_ord_aa = 0.0, e_ord_ab = 0.0, e_ord_bb = 0.0;

  std::vector<double> bar_fit_a_oqe(oaqe, 0.0);
  std::vector<double> bar_raw_coul_a_oqe(oaqe, 0.0);
  std::vector<double> bar_raw_g_a_oqe(include_f12 ? oaqe : 0, 0.0);
  std::vector<double> bar_res_g_a_oqe(include_f12 ? oaqe : 0, 0.0);
  std::vector<double> bar_raw_sq_a_oqe(include_f12 ? oaqe : 0, 0.0);
  std::vector<double> bar_res_sq_a_oqe(include_f12 ? oaqe : 0, 0.0);
  std::vector<double> bar_raw_gc_a_oqe(include_f12 ? oaqe : 0, 0.0);
  std::vector<double> bar_res_gc_a_oqe(include_f12 ? oaqe : 0, 0.0);
  std::vector<double> bar_raw_dc_a_oqe(include_f12 ? oaqe : 0, 0.0);
  std::vector<double> bar_res_dc_a_oqe(include_f12 ? oaqe : 0, 0.0);

  if (rhf_spatial) {
    std::vector<double> res_g_oqe, res_sq_oqe, res_gc_oqe, res_dc_oqe;
    std::vector<double> bar_res_os_g_oqe(include_f12 ? oaqe : 0, 0.0);
    std::vector<double> bar_res_os_sq_oqe(include_f12 ? oaqe : 0, 0.0);
    std::vector<double> bar_res_os_gc_oqe(include_f12 ? oaqe : 0, 0.0);
    std::vector<double> bar_res_os_dc_oqe(include_f12 ? oaqe : 0, 0.0);
    std::vector<double> Ha;
    if (include_f12) {
      build_res_oqe_host(raw_g_a_qoe, fit_a_qoe, metric_g, aux, oa_act, ext, res_g_oqe);
      build_res_oqe_host(raw_sq_a_qoe, fit_a_qoe, metric_sq, aux, oa_act, ext, res_sq_oqe);
      build_res_oqe_host(raw_gc_a_qoe, fit_a_qoe, metric_gc, aux, oa_act, ext, res_gc_oqe);
      build_res_oqe_host(raw_dc_a_qoe, fit_a_qoe, metric_dc, aux, oa_act, ext, res_dc_oqe);
      Ha.resize(ee);
      for (Size p = 0; p < ee; ++p) Ha[p] = fock_a[p] + exch_a[p];
    }

    const Size off_ab = n_aa;
    const Size off_bb = n_aa + n_ab;
    for (int i_loc = 0; i_loc < oa_act; ++i_loc) {
      const int i = fa + i_loc;
      for (int j_loc = i_loc; j_loc < oa_act; ++j_loc) {
        const int j = fa + j_loc;
        const double wij = has_occ_weights ? 0.5 * (occ_w_a[i_loc] + occ_w_a[j_loc]) : 1.0;
        double ef_ss = 0.0, eo_ss = 0.0, ef_os = 0.0, eo_os = 0.0;
        const int rc = eval_same_spin_or_rhf_pair_response_host(
            i, j, true, include_f12, include_ordinary, wij, oa, obs, ext, aux,
            fit_a_oqe.data() + Size(i_loc) * qe, fit_a_oqe.data() + Size(j_loc) * qe,
            raw_coul_a_oqe.data() + Size(i_loc) * qe, raw_coul_a_oqe.data() + Size(j_loc) * qe,
            include_f12 ? (raw_g_a_oqe.data() + Size(i_loc) * qe) : nullptr,
            include_f12 ? (res_g_oqe.data() + Size(j_loc) * qe) : nullptr,
            include_f12 ? (raw_sq_a_oqe.data() + Size(i_loc) * qe) : nullptr,
            include_f12 ? (res_sq_oqe.data() + Size(j_loc) * qe) : nullptr,
            include_f12 ? (raw_gc_a_oqe.data() + Size(i_loc) * qe) : nullptr,
            include_f12 ? (res_gc_oqe.data() + Size(j_loc) * qe) : nullptr,
            include_f12 ? (raw_dc_a_oqe.data() + Size(i_loc) * qe) : nullptr,
            include_f12 ? (res_dc_oqe.data() + Size(j_loc) * qe) : nullptr, fock_a, exch_a,
            include_f12 ? Ha.data() : nullptr, ef_ss, eo_ss, ef_os, eo_os,
            bar_fit_a_oqe.data() + Size(i_loc) * qe, bar_fit_a_oqe.data() + Size(j_loc) * qe,
            bar_raw_coul_a_oqe.data() + Size(i_loc) * qe,
            bar_raw_coul_a_oqe.data() + Size(j_loc) * qe,
            include_f12 ? (bar_raw_g_a_oqe.data() + Size(i_loc) * qe) : nullptr,
            include_f12 ? (bar_res_g_a_oqe.data() + Size(j_loc) * qe) : nullptr,
            include_f12 ? (bar_res_os_g_oqe.data() + Size(j_loc) * qe) : nullptr,
            include_f12 ? (bar_raw_sq_a_oqe.data() + Size(i_loc) * qe) : nullptr,
            include_f12 ? (bar_res_sq_a_oqe.data() + Size(j_loc) * qe) : nullptr,
            include_f12 ? (bar_res_os_sq_oqe.data() + Size(j_loc) * qe) : nullptr,
            include_f12 ? (bar_raw_gc_a_oqe.data() + Size(i_loc) * qe) : nullptr,
            include_f12 ? (bar_res_gc_a_oqe.data() + Size(j_loc) * qe) : nullptr,
            include_f12 ? (bar_res_os_gc_oqe.data() + Size(j_loc) * qe) : nullptr,
            include_f12 ? (bar_raw_dc_a_oqe.data() + Size(i_loc) * qe) : nullptr,
            include_f12 ? (bar_res_dc_a_oqe.data() + Size(j_loc) * qe) : nullptr,
            include_f12 ? (bar_res_os_dc_oqe.data() + Size(j_loc) * qe) : nullptr, bar_fock_a,
            bar_exch_a, include_f12 ? bar_cross_overlap : nullptr);
        if (rc != 0) return rc;

        if (i_loc == j_loc) {
          const Size ab_idx = off_ab + Size(i_loc) * oa_act + i_loc;
          if (include_f12 && pair_f12_out) pair_f12_out[ab_idx] = ef_os;
          if (include_ordinary && pair_ord_out) pair_ord_out[ab_idx] = eo_os;
          if (include_f12) e_f12_ab += wij * ef_os;
          if (include_ordinary) e_ord_ab += wij * eo_os;
        } else {
          const Size tri_idx = strict_tri_index(i_loc, j_loc, oa_act);
          const Size ab_ij = off_ab + Size(i_loc) * oa_act + j_loc;
          const Size ab_ji = off_ab + Size(j_loc) * oa_act + i_loc;
          if (include_f12 && pair_f12_out) {
            pair_f12_out[tri_idx] = ef_ss;
            pair_f12_out[ab_ij] = ef_os;
            pair_f12_out[ab_ji] = ef_os;
            pair_f12_out[off_bb + tri_idx] = ef_ss;
          }
          if (include_ordinary && pair_ord_out) {
            pair_ord_out[tri_idx] = eo_ss;
            pair_ord_out[ab_ij] = eo_os;
            pair_ord_out[ab_ji] = eo_os;
            pair_ord_out[off_bb + tri_idx] = eo_ss;
          }
          if (include_f12) {
            e_f12_aa += wij * ef_ss;
            e_f12_bb += wij * ef_ss;
            e_f12_ab += 2.0 * wij * ef_os;
          }
          if (include_ordinary) {
            e_ord_aa += wij * eo_ss;
            e_ord_bb += wij * eo_ss;
            e_ord_ab += 2.0 * wij * eo_os;
          }
        }
      }
    }

    // Pullback residuals once for the single spatial channel:
    // - AA+BB residuals contribute -2.0 * (bar_res_ss @ fit^T) (asymmetric)
    // - AB(i,j)+AB(j,i) and AB(i,i) residuals contribute symmetric -(M_os + M_os^T).
    std::vector<double> bar_fit_a_qoe;
    transpose_oqe_to_qoe_host(bar_fit_a_oqe, aux, oa_act, ext, bar_fit_a_qoe);
    if (include_f12) {
      pullback_res_oqe_host(bar_res_g_a_oqe, fit_a_qoe, metric_g, 2.0, false, aux, oa_act, ext,
                            bar_raw_g_a_oqe, bar_fit_a_qoe, bar_metric_g);
      pullback_res_oqe_host(bar_res_os_g_oqe, fit_a_qoe, metric_g, 2.0, true, aux, oa_act, ext,
                            bar_raw_g_a_oqe, bar_fit_a_qoe, bar_metric_g);
      pullback_res_oqe_host(bar_res_sq_a_oqe, fit_a_qoe, metric_sq, 2.0, false, aux, oa_act, ext,
                            bar_raw_sq_a_oqe, bar_fit_a_qoe, bar_metric_sq);
      pullback_res_oqe_host(bar_res_os_sq_oqe, fit_a_qoe, metric_sq, 2.0, true, aux, oa_act, ext,
                            bar_raw_sq_a_oqe, bar_fit_a_qoe, bar_metric_sq);
      pullback_res_oqe_host(bar_res_gc_a_oqe, fit_a_qoe, metric_gc, 2.0, false, aux, oa_act, ext,
                            bar_raw_gc_a_oqe, bar_fit_a_qoe, bar_metric_gc);
      pullback_res_oqe_host(bar_res_os_gc_oqe, fit_a_qoe, metric_gc, 2.0, true, aux, oa_act, ext,
                            bar_raw_gc_a_oqe, bar_fit_a_qoe, bar_metric_gc);
      pullback_res_oqe_host(bar_res_dc_a_oqe, fit_a_qoe, metric_dc, 2.0, false, aux, oa_act, ext,
                            bar_raw_dc_a_oqe, bar_fit_a_qoe, bar_metric_dc);
      pullback_res_oqe_host(bar_res_os_dc_oqe, fit_a_qoe, metric_dc, 2.0, true, aux, oa_act, ext,
                            bar_raw_dc_a_oqe, bar_fit_a_qoe, bar_metric_dc);
    }

    scatter_active_qoe_host(bar_fit_a_qoe, aux, oa, fa, oa_act, ext, bar_fit_a);
    scatter_active_oqe_host(bar_raw_coul_a_oqe, aux, oa, fa, oa_act, ext, bar_raw_coul_a);
    if (include_f12) {
      scatter_active_oqe_host(bar_raw_g_a_oqe, aux, oa, fa, oa_act, ext, bar_raw_g_a);
      scatter_active_oqe_host(bar_raw_sq_a_oqe, aux, oa, fa, oa_act, ext, bar_raw_sq_a);
      scatter_active_oqe_host(bar_raw_gc_a_oqe, aux, oa, fa, oa_act, ext, bar_raw_gc_a);
      scatter_active_oqe_host(bar_raw_dc_a_oqe, aux, oa, fa, oa_act, ext, bar_raw_dc_a);
    }
    // Mirror alpha outputs to beta outputs when distinct beta pointers are provided.
    if (bar_fit_b && bar_fit_b != bar_fit_a) std::copy_n(bar_fit_a, qoa_e, bar_fit_b);
    if (bar_raw_coul_b && bar_raw_coul_b != bar_raw_coul_a) {
      std::copy_n(bar_raw_coul_a, qoa_e, bar_raw_coul_b);
    }
    if (include_f12) {
      if (bar_raw_g_b && bar_raw_g_b != bar_raw_g_a) std::copy_n(bar_raw_g_a, qoa_e, bar_raw_g_b);
      if (bar_raw_sq_b && bar_raw_sq_b != bar_raw_sq_a) {
        std::copy_n(bar_raw_sq_a, qoa_e, bar_raw_sq_b);
      }
      if (bar_raw_gc_b && bar_raw_gc_b != bar_raw_gc_a) {
        std::copy_n(bar_raw_gc_a, qoa_e, bar_raw_gc_b);
      }
      if (bar_raw_dc_b && bar_raw_dc_b != bar_raw_dc_a) {
        std::copy_n(bar_raw_dc_a, qoa_e, bar_raw_dc_b);
      }
    }
    if (bar_fock_b && bar_fock_b != bar_fock_a) std::copy_n(bar_fock_a, ee, bar_fock_b);
    if (bar_exch_b && bar_exch_b != bar_exch_a) std::copy_n(bar_exch_a, ee, bar_exch_b);
  } else {
    // General UHF dual-spin path: AA, AB, BB.
    std::vector<double> fit_b_qoe, fit_b_oqe, raw_coul_b_oqe;
    std::vector<double> raw_g_b_qoe, raw_g_b_oqe, raw_sq_b_qoe, raw_sq_b_oqe;
    std::vector<double> raw_gc_b_qoe, raw_gc_b_oqe, raw_dc_b_qoe, raw_dc_b_oqe;
    if (ob_act > 0) {
      if (!pack_active_oqe_checked_host(fit_b, aux, ob, fb, ob_act, ext, fit_b_oqe, &fit_b_qoe) ||
          !pack_active_oqe_checked_host(raw_coul_b, aux, raw_coul_n1_b, fb, ob_act, ext,
                                        raw_coul_b_oqe) ||
          !pack_active_oqe_checked_host(raw_g_b, aux, ob, fb, ob_act, ext, raw_g_b_oqe,
                                        &raw_g_b_qoe) ||
          !pack_active_oqe_checked_host(raw_sq_b, aux, ob, fb, ob_act, ext, raw_sq_b_oqe,
                                        &raw_sq_b_qoe) ||
          !pack_active_oqe_checked_host(raw_gc_b, aux, ob, fb, ob_act, ext, raw_gc_b_oqe,
                                        &raw_gc_b_qoe) ||
          !pack_active_oqe_checked_host(raw_dc_b, aux, ob, fb, ob_act, ext, raw_dc_b_oqe,
                                        &raw_dc_b_qoe)) {
        return -2;
      }
    }

    std::vector<double> bar_fit_b_oqe(obqe, 0.0);
    std::vector<double> bar_raw_coul_b_oqe(obqe, 0.0);
    std::vector<double> bar_raw_g_b_oqe(include_f12 ? obqe : 0, 0.0);
    std::vector<double> bar_res_g_b_oqe(include_f12 ? obqe : 0, 0.0);
    std::vector<double> bar_raw_sq_b_oqe(include_f12 ? obqe : 0, 0.0);
    std::vector<double> bar_res_sq_b_oqe(include_f12 ? obqe : 0, 0.0);
    std::vector<double> bar_raw_gc_b_oqe(include_f12 ? obqe : 0, 0.0);
    std::vector<double> bar_res_gc_b_oqe(include_f12 ? obqe : 0, 0.0);
    std::vector<double> bar_raw_dc_b_oqe(include_f12 ? obqe : 0, 0.0);
    std::vector<double> bar_res_dc_b_oqe(include_f12 ? obqe : 0, 0.0);

    // Channel 1: alpha-alpha (i < j)
    if (oa_act >= 2) {
      std::vector<double> res_g_a_oqe, res_sq_a_oqe, res_gc_a_oqe, res_dc_a_oqe;
      std::vector<double> Ha;
      if (include_f12) {
        build_res_oqe_host(raw_g_a_qoe, fit_a_qoe, metric_g, aux, oa_act, ext, res_g_a_oqe);
        build_res_oqe_host(raw_sq_a_qoe, fit_a_qoe, metric_sq, aux, oa_act, ext, res_sq_a_oqe);
        build_res_oqe_host(raw_gc_a_qoe, fit_a_qoe, metric_gc, aux, oa_act, ext, res_gc_a_oqe);
        build_res_oqe_host(raw_dc_a_qoe, fit_a_qoe, metric_dc, aux, oa_act, ext, res_dc_a_oqe);
        Ha.resize(ee);
        for (Size p = 0; p < ee; ++p) Ha[p] = fock_a[p] + exch_a[p];
      }
      for (int i_loc = 0; i_loc < oa_act; ++i_loc) {
        for (int j_loc = i_loc + 1; j_loc < oa_act; ++j_loc) {
          const double wij = has_occ_weights ? 0.5 * (occ_w_a[i_loc] + occ_w_a[j_loc]) : 1.0;
          double ef_ss = 0.0, eo_ss = 0.0, dummy1 = 0.0, dummy2 = 0.0;
          const int rc = eval_same_spin_or_rhf_pair_response_host(
              fa + i_loc, fa + j_loc, false, include_f12, include_ordinary, wij, oa, obs, ext, aux,
              fit_a_oqe.data() + Size(i_loc) * qe, fit_a_oqe.data() + Size(j_loc) * qe, nullptr,
              raw_coul_a_oqe.data() + Size(j_loc) * qe,
              include_f12 ? (raw_g_a_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (res_g_a_oqe.data() + Size(j_loc) * qe) : nullptr,
              include_f12 ? (raw_sq_a_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (res_sq_a_oqe.data() + Size(j_loc) * qe) : nullptr,
              include_f12 ? (raw_gc_a_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (res_gc_a_oqe.data() + Size(j_loc) * qe) : nullptr,
              include_f12 ? (raw_dc_a_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (res_dc_a_oqe.data() + Size(j_loc) * qe) : nullptr, fock_a, exch_a,
              include_f12 ? Ha.data() : nullptr, ef_ss, eo_ss, dummy1, dummy2,
              bar_fit_a_oqe.data() + Size(i_loc) * qe, bar_fit_a_oqe.data() + Size(j_loc) * qe,
              nullptr, bar_raw_coul_a_oqe.data() + Size(j_loc) * qe,
              include_f12 ? (bar_raw_g_a_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (bar_res_g_a_oqe.data() + Size(j_loc) * qe) : nullptr, nullptr,
              include_f12 ? (bar_raw_sq_a_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (bar_res_sq_a_oqe.data() + Size(j_loc) * qe) : nullptr, nullptr,
              include_f12 ? (bar_raw_gc_a_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (bar_res_gc_a_oqe.data() + Size(j_loc) * qe) : nullptr, nullptr,
              include_f12 ? (bar_raw_dc_a_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (bar_res_dc_a_oqe.data() + Size(j_loc) * qe) : nullptr, nullptr,
              bar_fock_a, bar_exch_a, nullptr);
          if (rc != 0) return rc;
          const Size idx = strict_tri_index(i_loc, j_loc, oa_act);
          if (include_f12 && pair_f12_out) pair_f12_out[idx] = ef_ss;
          if (include_ordinary && pair_ord_out) pair_ord_out[idx] = eo_ss;
          if (include_f12) e_f12_aa += wij * ef_ss;
          if (include_ordinary) e_ord_aa += wij * eo_ss;
        }
      }
    }

    // Beta residuals shared by AB and BB channels.
    std::vector<double> res_g_b_oqe, res_sq_b_oqe, res_gc_b_oqe, res_dc_b_oqe;
    if (include_f12 && ob_act > 0 && (oa_act > 0 || ob_act >= 2)) {
      build_res_oqe_host(raw_g_b_qoe, fit_b_qoe, metric_g, aux, ob_act, ext, res_g_b_oqe);
      build_res_oqe_host(raw_sq_b_qoe, fit_b_qoe, metric_sq, aux, ob_act, ext, res_sq_b_oqe);
      build_res_oqe_host(raw_gc_b_qoe, fit_b_qoe, metric_gc, aux, ob_act, ext, res_gc_b_oqe);
      build_res_oqe_host(raw_dc_b_qoe, fit_b_qoe, metric_dc, aux, ob_act, ext, res_dc_b_oqe);
    }

    // Channel 2: alpha-beta (all i, j)
    if (oa_act > 0 && ob_act > 0) {
      const int rc = eval_uhf_opp_spin_channel_response_host(
          include_f12, include_ordinary, has_occ_weights, oa, ob, fa, fb, obs, ext, aux, occ_w_a,
          occ_w_b, fit_a_oqe, fit_b_oqe, raw_coul_b_oqe, raw_g_a_oqe, res_g_b_oqe, raw_sq_a_oqe,
          res_sq_b_oqe, raw_gc_a_oqe, res_gc_b_oqe, raw_dc_a_oqe, res_dc_b_oqe, fock_a, fock_b,
          exch_a, exch_b, cross_overlap, e_f12_ab, e_ord_ab,
          (include_f12 && pair_f12_out) ? (pair_f12_out + n_aa) : nullptr,
          (include_ordinary && pair_ord_out) ? (pair_ord_out + n_aa) : nullptr, bar_fit_a_oqe,
          bar_fit_b_oqe, bar_raw_coul_b_oqe, bar_raw_g_a_oqe, bar_res_g_b_oqe, bar_raw_sq_a_oqe,
          bar_res_sq_b_oqe, bar_raw_gc_a_oqe, bar_res_gc_b_oqe, bar_raw_dc_a_oqe, bar_res_dc_b_oqe,
          bar_fock_a, bar_fock_b, bar_exch_a, bar_exch_b,
          include_f12 ? bar_cross_overlap : nullptr);
      if (rc != 0) return rc;
    }

    // Channel 3: beta-beta (i < j)
    if (ob_act >= 2) {
      std::vector<double> Hb;
      if (include_f12) {
        Hb.resize(ee);
        for (Size p = 0; p < ee; ++p) Hb[p] = fock_b[p] + exch_b[p];
      }
      const Size off_bb = n_aa + n_ab;
      for (int i_loc = 0; i_loc < ob_act; ++i_loc) {
        for (int j_loc = i_loc + 1; j_loc < ob_act; ++j_loc) {
          const double wij = has_occ_weights ? 0.5 * (occ_w_b[i_loc] + occ_w_b[j_loc]) : 1.0;
          double ef_ss = 0.0, eo_ss = 0.0, dummy1 = 0.0, dummy2 = 0.0;
          const int rc = eval_same_spin_or_rhf_pair_response_host(
              fb + i_loc, fb + j_loc, false, include_f12, include_ordinary, wij, ob, obs, ext, aux,
              fit_b_oqe.data() + Size(i_loc) * qe, fit_b_oqe.data() + Size(j_loc) * qe, nullptr,
              raw_coul_b_oqe.data() + Size(j_loc) * qe,
              include_f12 ? (raw_g_b_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (res_g_b_oqe.data() + Size(j_loc) * qe) : nullptr,
              include_f12 ? (raw_sq_b_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (res_sq_b_oqe.data() + Size(j_loc) * qe) : nullptr,
              include_f12 ? (raw_gc_b_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (res_gc_b_oqe.data() + Size(j_loc) * qe) : nullptr,
              include_f12 ? (raw_dc_b_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (res_dc_b_oqe.data() + Size(j_loc) * qe) : nullptr, fock_b, exch_b,
              include_f12 ? Hb.data() : nullptr, ef_ss, eo_ss, dummy1, dummy2,
              bar_fit_b_oqe.data() + Size(i_loc) * qe, bar_fit_b_oqe.data() + Size(j_loc) * qe,
              nullptr, bar_raw_coul_b_oqe.data() + Size(j_loc) * qe,
              include_f12 ? (bar_raw_g_b_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (bar_res_g_b_oqe.data() + Size(j_loc) * qe) : nullptr, nullptr,
              include_f12 ? (bar_raw_sq_b_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (bar_res_sq_b_oqe.data() + Size(j_loc) * qe) : nullptr, nullptr,
              include_f12 ? (bar_raw_gc_b_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (bar_res_gc_b_oqe.data() + Size(j_loc) * qe) : nullptr, nullptr,
              include_f12 ? (bar_raw_dc_b_oqe.data() + Size(i_loc) * qe) : nullptr,
              include_f12 ? (bar_res_dc_b_oqe.data() + Size(j_loc) * qe) : nullptr, nullptr,
              bar_fock_b, bar_exch_b, nullptr);
          if (rc != 0) return rc;
          const Size idx = off_bb + strict_tri_index(i_loc, j_loc, ob_act);
          if (include_f12 && pair_f12_out) pair_f12_out[idx] = ef_ss;
          if (include_ordinary && pair_ord_out) pair_ord_out[idx] = eo_ss;
          if (include_f12) e_f12_bb += wij * ef_ss;
          if (include_ordinary) e_ord_bb += wij * eo_ss;
        }
      }
    }

    // Pullback residuals once per spin channel & scatter into full [aux, occ, ext] outputs.
    if (oa_act > 0) {
      std::vector<double> bar_fit_a_qoe;
      transpose_oqe_to_qoe_host(bar_fit_a_oqe, aux, oa_act, ext, bar_fit_a_qoe);
      if (include_f12 && oa_act >= 2) {
        pullback_res_oqe_host(bar_res_g_a_oqe, fit_a_qoe, metric_g, 1.0, false, aux, oa_act, ext,
                              bar_raw_g_a_oqe, bar_fit_a_qoe, bar_metric_g);
        pullback_res_oqe_host(bar_res_sq_a_oqe, fit_a_qoe, metric_sq, 1.0, false, aux, oa_act, ext,
                              bar_raw_sq_a_oqe, bar_fit_a_qoe, bar_metric_sq);
        pullback_res_oqe_host(bar_res_gc_a_oqe, fit_a_qoe, metric_gc, 1.0, false, aux, oa_act, ext,
                              bar_raw_gc_a_oqe, bar_fit_a_qoe, bar_metric_gc);
        pullback_res_oqe_host(bar_res_dc_a_oqe, fit_a_qoe, metric_dc, 1.0, false, aux, oa_act, ext,
                              bar_raw_dc_a_oqe, bar_fit_a_qoe, bar_metric_dc);
      }
      scatter_active_qoe_host(bar_fit_a_qoe, aux, oa, fa, oa_act, ext, bar_fit_a);
      scatter_active_oqe_host(bar_raw_coul_a_oqe, aux, oa, fa, oa_act, ext, bar_raw_coul_a);
      if (include_f12) {
        scatter_active_oqe_host(bar_raw_g_a_oqe, aux, oa, fa, oa_act, ext, bar_raw_g_a);
        scatter_active_oqe_host(bar_raw_sq_a_oqe, aux, oa, fa, oa_act, ext, bar_raw_sq_a);
        scatter_active_oqe_host(bar_raw_gc_a_oqe, aux, oa, fa, oa_act, ext, bar_raw_gc_a);
        scatter_active_oqe_host(bar_raw_dc_a_oqe, aux, oa, fa, oa_act, ext, bar_raw_dc_a);
      }
    }
    if (ob_act > 0) {
      std::vector<double> bar_fit_b_qoe;
      transpose_oqe_to_qoe_host(bar_fit_b_oqe, aux, ob_act, ext, bar_fit_b_qoe);
      if (include_f12 && (oa_act > 0 || ob_act >= 2)) {
        pullback_res_oqe_host(bar_res_g_b_oqe, fit_b_qoe, metric_g, 1.0, false, aux, ob_act, ext,
                              bar_raw_g_b_oqe, bar_fit_b_qoe, bar_metric_g);
        pullback_res_oqe_host(bar_res_sq_b_oqe, fit_b_qoe, metric_sq, 1.0, false, aux, ob_act, ext,
                              bar_raw_sq_b_oqe, bar_fit_b_qoe, bar_metric_sq);
        pullback_res_oqe_host(bar_res_gc_b_oqe, fit_b_qoe, metric_gc, 1.0, false, aux, ob_act, ext,
                              bar_raw_gc_b_oqe, bar_fit_b_qoe, bar_metric_gc);
        pullback_res_oqe_host(bar_res_dc_b_oqe, fit_b_qoe, metric_dc, 1.0, false, aux, ob_act, ext,
                              bar_raw_dc_b_oqe, bar_fit_b_qoe, bar_metric_dc);
      }
      scatter_active_qoe_host(bar_fit_b_qoe, aux, ob, fb, ob_act, ext, bar_fit_b);
      scatter_active_oqe_host(bar_raw_coul_b_oqe, aux, ob, fb, ob_act, ext, bar_raw_coul_b);
      if (include_f12) {
        scatter_active_oqe_host(bar_raw_g_b_oqe, aux, ob, fb, ob_act, ext, bar_raw_g_b);
        scatter_active_oqe_host(bar_raw_sq_b_oqe, aux, ob, fb, ob_act, ext, bar_raw_sq_b);
        scatter_active_oqe_host(bar_raw_gc_b_oqe, aux, ob, fb, ob_act, ext, bar_raw_gc_b);
        scatter_active_oqe_host(bar_raw_dc_b_oqe, aux, ob, fb, ob_act, ext, bar_raw_dc_b);
      }
    }
  }

  if (out_scalars) {
    out_scalars[0] = e_f12_aa;
    out_scalars[1] = e_f12_ab;
    out_scalars[2] = e_f12_bb;
    out_scalars[3] = e_ord_aa;
    out_scalars[4] = e_ord_ab;
    out_scalars[5] = e_ord_bb;
    if (!all_finite_host(out_scalars, 6)) return -5;
  }
  if (include_f12 && pair_f12_out && !all_finite_host(pair_f12_out, n_total)) return -5;
  if (include_ordinary && pair_ord_out && !all_finite_host(pair_ord_out, n_total)) return -5;
  if (!all_finite_host(bar_fock_a, ee) || !all_finite_host(bar_exch_a, ee)) return -5;
  if (!rhf_spatial && (!all_finite_host(bar_fock_b, ee) || !all_finite_host(bar_exch_b, ee))) {
    return -5;
  }
  if (include_f12 && bar_cross_overlap && !all_finite_host(bar_cross_overlap, ee)) return -5;
  if (include_f12 && (!all_finite_host(bar_metric_g, qq) || !all_finite_host(bar_metric_sq, qq) ||
                      !all_finite_host(bar_metric_gc, qq) || !all_finite_host(bar_metric_dc, qq))) {
    return -5;
  }
  if (oa_act > 0) {
    if (!all_finite_host(bar_fit_a, qoa_e) || !all_finite_host(bar_raw_coul_a, qoa_e)) return -5;
    if (include_f12 &&
        (!all_finite_host(bar_raw_g_a, qoa_e) || !all_finite_host(bar_raw_sq_a, qoa_e) ||
         !all_finite_host(bar_raw_gc_a, qoa_e) || !all_finite_host(bar_raw_dc_a, qoa_e))) {
      return -5;
    }
  }
  if (!rhf_spatial && ob_act > 0) {
    if (!all_finite_host(bar_fit_b, qob_e) || !all_finite_host(bar_raw_coul_b, qob_e)) return -5;
    if (include_f12 &&
        (!all_finite_host(bar_raw_g_b, qob_e) || !all_finite_host(bar_raw_sq_b, qob_e) ||
         !all_finite_host(bar_raw_gc_b, qob_e) || !all_finite_host(bar_raw_dc_b, qob_e))) {
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

int gqc_f12_mp2_pair_energies_host_reference(
    int rhf_spatial_int, int include_ordinary_int, int has_occ_weights_int, int oa, int ob, int fa,
    int fb, int obs, int ext, int aux, int raw_coul_n1_a, int raw_coul_n1_b,
    std::uint64_t budget_bytes, const double* occ_w_a, const double* occ_w_b, const double* fit_a,
    const double* fit_b, const double* raw_coul_a, const double* raw_coul_b, const double* raw_g_a,
    const double* raw_g_b, const double* raw_sq_a, const double* raw_sq_b, const double* raw_gc_a,
    const double* raw_gc_b, const double* raw_dc_a, const double* raw_dc_b, const double* metric_g,
    const double* metric_sq, const double* metric_gc, const double* metric_dc, const double* fock_a,
    const double* fock_b, const double* exch_a, const double* exch_b, const double* cross_overlap,
    double* out_scalars, double* pair_f12_out, double* pair_ord_out) {
  try {
  const bool rhf_spatial = (rhf_spatial_int != 0);
  const bool include_ordinary = (include_ordinary_int != 0);
  const bool has_occ_weights = (has_occ_weights_int != 0);

  if (fa < 0 || fa > oa || oa >= obs || fb < 0 || fb > ob || ob >= obs || obs >= ext || aux < 1) {
    return -1;
  }
  if (rhf_spatial && (oa != ob || fa != fb)) return -1;
  if (!metric_g || !metric_sq || !metric_gc || !metric_dc || !fock_a || !exch_a) return -1;
  if (!rhf_spatial && (!fock_b || !exch_b)) return -1;

  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  if (has_occ_weights) {
    if (oa_act > 0 && !occ_w_a) return -1;
    if (!rhf_spatial && ob_act > 0 && !occ_w_b) return -1;
  }
  if (oa_act > 0) {
    if (raw_coul_n1_a < oa || !fit_a || !raw_coul_a || !raw_g_a || !raw_sq_a || !raw_gc_a ||
        !raw_dc_a) {
      return -1;
    }
  }
  if (!rhf_spatial && ob_act > 0) {
    if (raw_coul_n1_b < ob || !fit_b || !raw_coul_b || !raw_g_b || !raw_sq_b || !raw_gc_b ||
        !raw_dc_b) {
      return -1;
    }
  }
  if (!rhf_spatial && oa_act > 0 && ob_act > 0 && !cross_overlap) return -1;

  const Size qq = Size(aux) * aux;
  const Size ee = Size(ext) * ext;
  if (!all_finite_host(metric_g, qq) || !all_finite_host(metric_sq, qq) ||
      !all_finite_host(metric_gc, qq) || !all_finite_host(metric_dc, qq) ||
      !all_finite_host(fock_a, ee) || !all_finite_host(exch_a, ee)) {
    return -2;
  }
  if (!rhf_spatial && (!all_finite_host(fock_b, ee) || !all_finite_host(exch_b, ee))) {
    return -2;
  }
  if (!rhf_spatial && oa_act > 0 && ob_act > 0 && !all_finite_host(cross_overlap, ee)) {
    return -2;
  }
  if (has_occ_weights) {
    if (oa_act > 0 && !all_finite_host(occ_w_a, oa_act)) return -2;
    if (!rhf_spatial && ob_act > 0 && !all_finite_host(occ_w_b, ob_act)) return -2;
  }

  const Size n_aa = (oa_act >= 2) ? (Size(oa_act) * (oa_act - 1)) / 2 : 0;
  const Size n_ab = Size(oa_act) * Size(ob_act);
  const Size n_bb = (ob_act >= 2) ? (Size(ob_act) * (ob_act - 1)) / 2 : 0;
  const Size n_total = n_aa + n_ab + n_bb;

  if (out_scalars) std::fill_n(out_scalars, 6, 0.0);
  if (pair_f12_out && n_total > 0) std::fill_n(pair_f12_out, n_total, 0.0);
  if (include_ordinary && pair_ord_out && n_total > 0) std::fill_n(pair_ord_out, n_total, 0.0);

  if (n_total > 0) {
    const Size req_bytes = estimate_mp2_pair_energies_device_bytes(
        rhf_spatial, include_ordinary, has_occ_weights, pair_f12_out != nullptr,
        pair_ord_out != nullptr, oa, ob, fa, fb, obs, ext, aux);
    if (req_bytes > effective_device_budget(budget_bytes)) return kErrBudgetExceeded;
  }

  // Pack active slices & verify finiteness even when n_total == 0 (e.g., H atom).
  std::vector<double> fit_a_qoe, fit_a_oqe, raw_coul_a_oqe;
  std::vector<double> raw_g_a_qoe, raw_g_a_oqe, raw_sq_a_qoe, raw_sq_a_oqe;
  std::vector<double> raw_gc_a_qoe, raw_gc_a_oqe, raw_dc_a_qoe, raw_dc_a_oqe;
  if (oa_act > 0) {
    if (!pack_active_oqe_checked_host(fit_a, aux, oa, fa, oa_act, ext, fit_a_oqe, &fit_a_qoe) ||
        !pack_active_oqe_checked_host(raw_coul_a, aux, raw_coul_n1_a, fa, oa_act, ext,
                                      raw_coul_a_oqe) ||
        !pack_active_oqe_checked_host(raw_g_a, aux, oa, fa, oa_act, ext, raw_g_a_oqe,
                                      &raw_g_a_qoe) ||
        !pack_active_oqe_checked_host(raw_sq_a, aux, oa, fa, oa_act, ext, raw_sq_a_oqe,
                                      &raw_sq_a_qoe) ||
        !pack_active_oqe_checked_host(raw_gc_a, aux, oa, fa, oa_act, ext, raw_gc_a_oqe,
                                      &raw_gc_a_qoe) ||
        !pack_active_oqe_checked_host(raw_dc_a, aux, oa, fa, oa_act, ext, raw_dc_a_oqe,
                                      &raw_dc_a_qoe)) {
      return -2;
    }
  }
  if (n_total == 0) {
    if (!rhf_spatial && ob_act > 0) {
      std::vector<double> chk;
      if (!pack_active_oqe_checked_host(fit_b, aux, ob, fb, ob_act, ext, chk) ||
          !pack_active_oqe_checked_host(raw_coul_b, aux, raw_coul_n1_b, fb, ob_act, ext, chk) ||
          !pack_active_oqe_checked_host(raw_g_b, aux, ob, fb, ob_act, ext, chk) ||
          !pack_active_oqe_checked_host(raw_sq_b, aux, ob, fb, ob_act, ext, chk) ||
          !pack_active_oqe_checked_host(raw_gc_b, aux, ob, fb, ob_act, ext, chk) ||
          !pack_active_oqe_checked_host(raw_dc_b, aux, ob, fb, ob_act, ext, chk)) {
        return -2;
      }
    }
    return 0;
  }

  const Size qe = Size(aux) * ext;
  double e_f12_aa = 0.0, e_f12_ab = 0.0, e_f12_bb = 0.0;
  double e_ord_aa = 0.0, e_ord_ab = 0.0, e_ord_bb = 0.0;

  if (rhf_spatial) {
    std::vector<double> res_g_oqe, res_sq_oqe, res_gc_oqe, res_dc_oqe;
    build_res_oqe_host(raw_g_a_qoe, fit_a_qoe, metric_g, aux, oa_act, ext, res_g_oqe);
    build_res_oqe_host(raw_sq_a_qoe, fit_a_qoe, metric_sq, aux, oa_act, ext, res_sq_oqe);
    build_res_oqe_host(raw_gc_a_qoe, fit_a_qoe, metric_gc, aux, oa_act, ext, res_gc_oqe);
    build_res_oqe_host(raw_dc_a_qoe, fit_a_qoe, metric_dc, aux, oa_act, ext, res_dc_oqe);

    std::vector<double> Ha(ee);
    for (Size p = 0; p < ee; ++p) Ha[p] = fock_a[p] + exch_a[p];

    const Size off_ab = n_aa;
    const Size off_bb = n_aa + n_ab;
    for (int i_loc = 0; i_loc < oa_act; ++i_loc) {
      const int i = fa + i_loc;
      for (int j_loc = i_loc; j_loc < oa_act; ++j_loc) {
        const int j = fa + j_loc;
        double ef_ss = 0.0, eo_ss = 0.0, ef_os = 0.0, eo_os = 0.0;
        const int rc = eval_same_spin_or_rhf_pair_host(
            i, j, true, include_ordinary, oa, obs, ext, aux, fit_a_oqe.data() + Size(i_loc) * qe,
            fit_a_oqe.data() + Size(j_loc) * qe, raw_coul_a_oqe.data() + Size(j_loc) * qe,
            raw_g_a_oqe.data() + Size(i_loc) * qe, res_g_oqe.data() + Size(j_loc) * qe,
            raw_sq_a_oqe.data() + Size(i_loc) * qe, res_sq_oqe.data() + Size(j_loc) * qe,
            raw_gc_a_oqe.data() + Size(i_loc) * qe, res_gc_oqe.data() + Size(j_loc) * qe,
            raw_dc_a_oqe.data() + Size(i_loc) * qe, res_dc_oqe.data() + Size(j_loc) * qe, fock_a,
            exch_a, Ha.data(), ef_ss, eo_ss, ef_os, eo_os);
        if (rc != 0) return rc;

        const double wij = has_occ_weights ? 0.5 * (occ_w_a[i_loc] + occ_w_a[j_loc]) : 1.0;
        if (i_loc == j_loc) {
          const Size ab_idx = off_ab + Size(i_loc) * oa_act + i_loc;
          if (pair_f12_out) pair_f12_out[ab_idx] = ef_os;
          if (include_ordinary && pair_ord_out) pair_ord_out[ab_idx] = eo_os;
          e_f12_ab += wij * ef_os;
          if (include_ordinary) e_ord_ab += wij * eo_os;
        } else {
          const Size tri_idx = strict_tri_index(i_loc, j_loc, oa_act);
          const Size ab_ij = off_ab + Size(i_loc) * oa_act + j_loc;
          const Size ab_ji = off_ab + Size(j_loc) * oa_act + i_loc;
          if (pair_f12_out) {
            pair_f12_out[tri_idx] = ef_ss;
            pair_f12_out[ab_ij] = ef_os;
            pair_f12_out[ab_ji] = ef_os;
            pair_f12_out[off_bb + tri_idx] = ef_ss;
          }
          if (include_ordinary && pair_ord_out) {
            pair_ord_out[tri_idx] = eo_ss;
            pair_ord_out[ab_ij] = eo_os;
            pair_ord_out[ab_ji] = eo_os;
            pair_ord_out[off_bb + tri_idx] = eo_ss;
          }
          e_f12_aa += wij * ef_ss;
          e_f12_bb += wij * ef_ss;
          e_f12_ab += 2.0 * wij * ef_os;
          if (include_ordinary) {
            e_ord_aa += wij * eo_ss;
            e_ord_bb += wij * eo_ss;
            e_ord_ab += 2.0 * wij * eo_os;
          }
        }
      }
    }
  } else {
    // General UHF dual-spin path: AA, AB, BB.
    std::vector<double> fit_b_qoe, fit_b_oqe, raw_coul_b_oqe;
    std::vector<double> raw_g_b_qoe, raw_g_b_oqe, raw_sq_b_qoe, raw_sq_b_oqe;
    std::vector<double> raw_gc_b_qoe, raw_gc_b_oqe, raw_dc_b_qoe, raw_dc_b_oqe;
    if (ob_act > 0) {
      if (!pack_active_oqe_checked_host(fit_b, aux, ob, fb, ob_act, ext, fit_b_oqe, &fit_b_qoe) ||
          !pack_active_oqe_checked_host(raw_coul_b, aux, raw_coul_n1_b, fb, ob_act, ext,
                                        raw_coul_b_oqe) ||
          !pack_active_oqe_checked_host(raw_g_b, aux, ob, fb, ob_act, ext, raw_g_b_oqe,
                                        &raw_g_b_qoe) ||
          !pack_active_oqe_checked_host(raw_sq_b, aux, ob, fb, ob_act, ext, raw_sq_b_oqe,
                                        &raw_sq_b_qoe) ||
          !pack_active_oqe_checked_host(raw_gc_b, aux, ob, fb, ob_act, ext, raw_gc_b_oqe,
                                        &raw_gc_b_qoe) ||
          !pack_active_oqe_checked_host(raw_dc_b, aux, ob, fb, ob_act, ext, raw_dc_b_oqe,
                                        &raw_dc_b_qoe)) {
        return -2;
      }
    }

    // Channel 1: alpha-alpha (i < j)
    if (oa_act >= 2) {
      std::vector<double> res_g_a_oqe, res_sq_a_oqe, res_gc_a_oqe, res_dc_a_oqe;
      build_res_oqe_host(raw_g_a_qoe, fit_a_qoe, metric_g, aux, oa_act, ext, res_g_a_oqe);
      build_res_oqe_host(raw_sq_a_qoe, fit_a_qoe, metric_sq, aux, oa_act, ext, res_sq_a_oqe);
      build_res_oqe_host(raw_gc_a_qoe, fit_a_qoe, metric_gc, aux, oa_act, ext, res_gc_a_oqe);
      build_res_oqe_host(raw_dc_a_qoe, fit_a_qoe, metric_dc, aux, oa_act, ext, res_dc_a_oqe);
      std::vector<double> Ha(ee);
      for (Size p = 0; p < ee; ++p) Ha[p] = fock_a[p] + exch_a[p];
      for (int i_loc = 0; i_loc < oa_act; ++i_loc) {
        for (int j_loc = i_loc + 1; j_loc < oa_act; ++j_loc) {
          double ef_ss = 0.0, eo_ss = 0.0, dummy1 = 0.0, dummy2 = 0.0;
          const int rc = eval_same_spin_or_rhf_pair_host(
              fa + i_loc, fa + j_loc, false, include_ordinary, oa, obs, ext, aux,
              fit_a_oqe.data() + Size(i_loc) * qe, fit_a_oqe.data() + Size(j_loc) * qe,
              raw_coul_a_oqe.data() + Size(j_loc) * qe, raw_g_a_oqe.data() + Size(i_loc) * qe,
              res_g_a_oqe.data() + Size(j_loc) * qe, raw_sq_a_oqe.data() + Size(i_loc) * qe,
              res_sq_a_oqe.data() + Size(j_loc) * qe, raw_gc_a_oqe.data() + Size(i_loc) * qe,
              res_gc_a_oqe.data() + Size(j_loc) * qe, raw_dc_a_oqe.data() + Size(i_loc) * qe,
              res_dc_a_oqe.data() + Size(j_loc) * qe, fock_a, exch_a, Ha.data(), ef_ss, eo_ss,
              dummy1, dummy2);
          if (rc != 0) return rc;
          const Size idx = strict_tri_index(i_loc, j_loc, oa_act);
          if (pair_f12_out) pair_f12_out[idx] = ef_ss;
          if (include_ordinary && pair_ord_out) pair_ord_out[idx] = eo_ss;
          const double wij = has_occ_weights ? 0.5 * (occ_w_a[i_loc] + occ_w_a[j_loc]) : 1.0;
          e_f12_aa += wij * ef_ss;
          if (include_ordinary) e_ord_aa += wij * eo_ss;
        }
      }
    }

    // Beta residuals are shared by AB and BB channels.
    std::vector<double> res_g_b_oqe, res_sq_b_oqe, res_gc_b_oqe, res_dc_b_oqe;
    if (ob_act > 0 && (oa_act > 0 || ob_act >= 2)) {
      build_res_oqe_host(raw_g_b_qoe, fit_b_qoe, metric_g, aux, ob_act, ext, res_g_b_oqe);
      build_res_oqe_host(raw_sq_b_qoe, fit_b_qoe, metric_sq, aux, ob_act, ext, res_sq_b_oqe);
      build_res_oqe_host(raw_gc_b_qoe, fit_b_qoe, metric_gc, aux, ob_act, ext, res_gc_b_oqe);
      build_res_oqe_host(raw_dc_b_qoe, fit_b_qoe, metric_dc, aux, ob_act, ext, res_dc_b_oqe);
    }

    // Channel 2: alpha-beta (all i, j)
    if (oa_act > 0 && ob_act > 0) {
      const int rc = eval_uhf_opp_spin_channel_host(
          include_ordinary, has_occ_weights, oa, ob, fa, fb, obs, ext, aux, occ_w_a, occ_w_b,
          fit_a_oqe, fit_b_oqe, raw_coul_b_oqe, raw_g_a_oqe, res_g_b_oqe, raw_sq_a_oqe,
          res_sq_b_oqe, raw_gc_a_oqe, res_gc_b_oqe, raw_dc_a_oqe, res_dc_b_oqe, fock_a, fock_b,
          exch_a, exch_b, cross_overlap, e_f12_ab, e_ord_ab,
          pair_f12_out ? (pair_f12_out + n_aa) : nullptr,
          (include_ordinary && pair_ord_out) ? (pair_ord_out + n_aa) : nullptr);
      if (rc != 0) return rc;
    }

    // Channel 3: beta-beta (i < j)
    if (ob_act >= 2) {
      std::vector<double> Hb(ee);
      for (Size p = 0; p < ee; ++p) Hb[p] = fock_b[p] + exch_b[p];
      const Size off_bb = n_aa + n_ab;
      for (int i_loc = 0; i_loc < ob_act; ++i_loc) {
        for (int j_loc = i_loc + 1; j_loc < ob_act; ++j_loc) {
          double ef_ss = 0.0, eo_ss = 0.0, dummy1 = 0.0, dummy2 = 0.0;
          const int rc = eval_same_spin_or_rhf_pair_host(
              fb + i_loc, fb + j_loc, false, include_ordinary, ob, obs, ext, aux,
              fit_b_oqe.data() + Size(i_loc) * qe, fit_b_oqe.data() + Size(j_loc) * qe,
              raw_coul_b_oqe.data() + Size(j_loc) * qe, raw_g_b_oqe.data() + Size(i_loc) * qe,
              res_g_b_oqe.data() + Size(j_loc) * qe, raw_sq_b_oqe.data() + Size(i_loc) * qe,
              res_sq_b_oqe.data() + Size(j_loc) * qe, raw_gc_b_oqe.data() + Size(i_loc) * qe,
              res_gc_b_oqe.data() + Size(j_loc) * qe, raw_dc_b_oqe.data() + Size(i_loc) * qe,
              res_dc_b_oqe.data() + Size(j_loc) * qe, fock_b, exch_b, Hb.data(), ef_ss, eo_ss,
              dummy1, dummy2);
          if (rc != 0) return rc;
          const Size idx = off_bb + strict_tri_index(i_loc, j_loc, ob_act);
          if (pair_f12_out) pair_f12_out[idx] = ef_ss;
          if (include_ordinary && pair_ord_out) pair_ord_out[idx] = eo_ss;
          const double wij = has_occ_weights ? 0.5 * (occ_w_b[i_loc] + occ_w_b[j_loc]) : 1.0;
          e_f12_bb += wij * ef_ss;
          if (include_ordinary) e_ord_bb += wij * eo_ss;
        }
      }
    }
  }

  if (out_scalars) {
    out_scalars[0] = e_f12_aa;
    out_scalars[1] = e_f12_ab;
    out_scalars[2] = e_f12_bb;
    out_scalars[3] = e_ord_aa;
    out_scalars[4] = e_ord_ab;
    out_scalars[5] = e_ord_bb;
    if (!all_finite_host(out_scalars, 6)) return -5;
  }
  if (pair_f12_out && !all_finite_host(pair_f12_out, n_total)) return -5;
  if (include_ordinary && pair_ord_out && !all_finite_host(pair_ord_out, n_total)) return -5;
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

int gqc_f12_mp2_pair_response_cuda(
    int rhf_spatial_int, int include_f12_int, int include_ordinary_int, int has_occ_weights_int,
    int oa, int ob, int fa, int fb, int obs, int ext, int aux, int raw_coul_n1_a,
    int raw_coul_n1_b, std::uint64_t budget_bytes, const double* occ_w_a, const double* occ_w_b,
    const double* fit_a, const double* fit_b, const double* raw_coul_a, const double* raw_coul_b,
    const double* raw_g_a, const double* raw_g_b, const double* raw_sq_a, const double* raw_sq_b,
    const double* raw_gc_a, const double* raw_gc_b, const double* raw_dc_a, const double* raw_dc_b,
    const double* metric_g, const double* metric_sq, const double* metric_gc,
    const double* metric_dc, const double* fock_a, const double* fock_b, const double* exch_a,
    const double* exch_b, const double* cross_overlap, double* out_scalars, double* pair_f12_out,
    double* pair_ord_out, double* bar_fit_a, double* bar_fit_b, double* bar_raw_coul_a,
    double* bar_raw_coul_b, double* bar_raw_g_a, double* bar_raw_g_b, double* bar_raw_sq_a,
    double* bar_raw_sq_b, double* bar_raw_gc_a, double* bar_raw_gc_b, double* bar_raw_dc_a,
    double* bar_raw_dc_b, double* bar_metric_g, double* bar_metric_sq, double* bar_metric_gc,
    double* bar_metric_dc, double* bar_fock_a, double* bar_fock_b, double* bar_exch_a,
    double* bar_exch_b, double* bar_cross_overlap) {
  try {
  const bool rhf_spatial = (rhf_spatial_int != 0);
  const bool include_f12 = (include_f12_int != 0);
  const bool include_ordinary = (include_ordinary_int != 0);
  const bool has_occ_weights = (has_occ_weights_int != 0);

  if (fa < 0 || fa > oa || oa >= obs || fb < 0 || fb > ob || ob >= obs || obs >= ext || aux < 1) {
    return -1;
  }
  if (rhf_spatial && (oa != ob || fa != fb)) return -1;
  if (!metric_g || !metric_sq || !metric_gc || !metric_dc || !fock_a || !exch_a) return -1;
  if (!rhf_spatial && (!fock_b || !exch_b)) return -1;
  if (!bar_fock_a || !bar_exch_a) return -1;
  if (!rhf_spatial && (!bar_fock_b || !bar_exch_b)) return -1;
  if (include_f12 && (!bar_metric_g || !bar_metric_sq || !bar_metric_gc || !bar_metric_dc)) {
    return -1;
  }

  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  if (has_occ_weights) {
    if (oa_act > 0 && !occ_w_a) return -1;
    if (!rhf_spatial && ob_act > 0 && !occ_w_b) return -1;
  }
  if (oa_act > 0) {
    if (raw_coul_n1_a < oa || !fit_a || !raw_coul_a || !raw_g_a || !raw_sq_a || !raw_gc_a ||
        !raw_dc_a) {
      return -1;
    }
    if (!bar_fit_a || !bar_raw_coul_a) return -1;
    if (include_f12 && (!bar_raw_g_a || !bar_raw_sq_a || !bar_raw_gc_a || !bar_raw_dc_a)) {
      return -1;
    }
  }
  if (!rhf_spatial && ob_act > 0) {
    if (raw_coul_n1_b < ob || !fit_b || !raw_coul_b || !raw_g_b || !raw_sq_b || !raw_gc_b ||
        !raw_dc_b) {
      return -1;
    }
    if (!bar_fit_b || !bar_raw_coul_b) return -1;
    if (include_f12 && (!bar_raw_g_b || !bar_raw_sq_b || !bar_raw_gc_b || !bar_raw_dc_b)) {
      return -1;
    }
  }
  if (!rhf_spatial && oa_act > 0 && ob_act > 0 && !cross_overlap) return -1;
  if (include_f12 && oa_act > 0 && ob_act > 0 && !bar_cross_overlap) return -1;

  const Size qq = Size(aux) * aux;
  const Size ee = Size(ext) * ext;
  const Size qoa_e = Size(aux) * Size(oa) * ext;
  const Size qob_e = Size(aux) * Size(ob) * ext;

  if (!all_finite_host(metric_g, qq) || !all_finite_host(metric_sq, qq) ||
      !all_finite_host(metric_gc, qq) || !all_finite_host(metric_dc, qq) ||
      !all_finite_host(fock_a, ee) || !all_finite_host(exch_a, ee)) {
    return -2;
  }
  if (!rhf_spatial && (!all_finite_host(fock_b, ee) || !all_finite_host(exch_b, ee))) {
    return -2;
  }
  if (!rhf_spatial && oa_act > 0 && ob_act > 0 && !all_finite_host(cross_overlap, ee)) {
    return -2;
  }
  if (has_occ_weights) {
    if (oa_act > 0 && !all_finite_host(occ_w_a, oa_act)) return -2;
    if (!rhf_spatial && ob_act > 0 && !all_finite_host(occ_w_b, ob_act)) return -2;
  }

  const Size n_aa = (oa_act >= 2) ? (Size(oa_act) * (oa_act - 1)) / 2 : 0;
  const Size n_ab = Size(oa_act) * Size(ob_act);
  const Size n_bb = (ob_act >= 2) ? (Size(ob_act) * (ob_act - 1)) / 2 : 0;
  const Size n_total = n_aa + n_ab + n_bb;

  // Zero-initialize all provided output buffers.
  if (out_scalars) std::fill_n(out_scalars, 6, 0.0);
  if (pair_f12_out && n_total > 0) std::fill_n(pair_f12_out, n_total, 0.0);
  if (include_ordinary && pair_ord_out && n_total > 0) std::fill_n(pair_ord_out, n_total, 0.0);
  if (qoa_e > 0) {
    if (bar_fit_a) std::fill_n(bar_fit_a, qoa_e, 0.0);
    if (bar_raw_coul_a) std::fill_n(bar_raw_coul_a, qoa_e, 0.0);
    if (bar_raw_g_a) std::fill_n(bar_raw_g_a, qoa_e, 0.0);
    if (bar_raw_sq_a) std::fill_n(bar_raw_sq_a, qoa_e, 0.0);
    if (bar_raw_gc_a) std::fill_n(bar_raw_gc_a, qoa_e, 0.0);
    if (bar_raw_dc_a) std::fill_n(bar_raw_dc_a, qoa_e, 0.0);
  }
  if (qob_e > 0) {
    if (bar_fit_b && bar_fit_b != bar_fit_a) std::fill_n(bar_fit_b, qob_e, 0.0);
    if (bar_raw_coul_b && bar_raw_coul_b != bar_raw_coul_a) std::fill_n(bar_raw_coul_b, qob_e, 0.0);
    if (bar_raw_g_b && bar_raw_g_b != bar_raw_g_a) std::fill_n(bar_raw_g_b, qob_e, 0.0);
    if (bar_raw_sq_b && bar_raw_sq_b != bar_raw_sq_a) std::fill_n(bar_raw_sq_b, qob_e, 0.0);
    if (bar_raw_gc_b && bar_raw_gc_b != bar_raw_gc_a) std::fill_n(bar_raw_gc_b, qob_e, 0.0);
    if (bar_raw_dc_b && bar_raw_dc_b != bar_raw_dc_a) std::fill_n(bar_raw_dc_b, qob_e, 0.0);
  }
  if (bar_metric_g) std::fill_n(bar_metric_g, qq, 0.0);
  if (bar_metric_sq) std::fill_n(bar_metric_sq, qq, 0.0);
  if (bar_metric_gc) std::fill_n(bar_metric_gc, qq, 0.0);
  if (bar_metric_dc) std::fill_n(bar_metric_dc, qq, 0.0);
  std::fill_n(bar_fock_a, ee, 0.0);
  std::fill_n(bar_exch_a, ee, 0.0);
  if (bar_fock_b && bar_fock_b != bar_fock_a) std::fill_n(bar_fock_b, ee, 0.0);
  if (bar_exch_b && bar_exch_b != bar_exch_a) std::fill_n(bar_exch_b, ee, 0.0);
  if (bar_cross_overlap) std::fill_n(bar_cross_overlap, ee, 0.0);

  // Validate active 3-center inputs even on early exit.
  std::vector<double> h_qoe;
  if (n_total == 0 || (!include_f12 && !include_ordinary)) {
    if (oa_act > 0) {
      if (!pack_active_qoe_checked_cuda_host(fit_a, aux, oa, fa, oa_act, ext, h_qoe) ||
          !pack_active_qoe_checked_cuda_host(raw_coul_a, aux, raw_coul_n1_a, fa, oa_act, ext,
                                             h_qoe) ||
          !pack_active_qoe_checked_cuda_host(raw_g_a, aux, oa, fa, oa_act, ext, h_qoe) ||
          !pack_active_qoe_checked_cuda_host(raw_sq_a, aux, oa, fa, oa_act, ext, h_qoe) ||
          !pack_active_qoe_checked_cuda_host(raw_gc_a, aux, oa, fa, oa_act, ext, h_qoe) ||
          !pack_active_qoe_checked_cuda_host(raw_dc_a, aux, oa, fa, oa_act, ext, h_qoe)) {
        return -2;
      }
    }
    if (!rhf_spatial && ob_act > 0) {
      if (!pack_active_qoe_checked_cuda_host(fit_b, aux, ob, fb, ob_act, ext, h_qoe) ||
          !pack_active_qoe_checked_cuda_host(raw_coul_b, aux, raw_coul_n1_b, fb, ob_act, ext,
                                             h_qoe) ||
          !pack_active_qoe_checked_cuda_host(raw_g_b, aux, ob, fb, ob_act, ext, h_qoe) ||
          !pack_active_qoe_checked_cuda_host(raw_sq_b, aux, ob, fb, ob_act, ext, h_qoe) ||
          !pack_active_qoe_checked_cuda_host(raw_gc_b, aux, ob, fb, ob_act, ext, h_qoe) ||
          !pack_active_qoe_checked_cuda_host(raw_dc_b, aux, ob, fb, ob_act, ext, h_qoe)) {
        return -2;
      }
    }
    return 0;
  }

  // Enforce caller-specified hard device VRAM budget BEFORE allocating any device memory.
  {
    const Size req_device_bytes = estimate_mp2_pair_response_device_bytes(
        rhf_spatial, include_f12, include_ordinary, pair_f12_out != nullptr,
        pair_ord_out != nullptr, bar_cross_overlap != nullptr, oa, ob, fa, fb, obs, ext, aux);
    if (req_device_bytes > effective_device_budget(budget_bytes)) {
      return kErrBudgetExceeded;
    }
  }

  DeviceWorkspace workspace(budget_bytes);
  CublasGuard cg;
  if (cg.init() != CUBLAS_STATUS_SUCCESS) return 1;
  cublasHandle_t h = cg.handle;

  DeviceBuf d_out_scalars, d_pair_f12, d_pair_ord;
  DeviceIntBuf d_err;
  if (d_out_scalars.alloc(6) != cudaSuccess || d_err.alloc(1) != cudaSuccess ||
      cudaMemset(d_out_scalars.ptr, 0, 6 * sizeof(double)) != cudaSuccess ||
      cudaMemset(d_err.ptr, 0, sizeof(int)) != cudaSuccess) {
    return 2;
  }
  if (include_f12 && pair_f12_out) {
    if (d_pair_f12.alloc(n_total) != cudaSuccess ||
        cudaMemset(d_pair_f12.ptr, 0, n_total * sizeof(double)) != cudaSuccess) {
      return 2;
    }
  }
  if (include_ordinary && pair_ord_out) {
    if (d_pair_ord.alloc(n_total) != cudaSuccess ||
        cudaMemset(d_pair_ord.ptr, 0, n_total * sizeof(double)) != cudaSuccess) {
      return 2;
    }
  }

  const Size qoe_a = Size(aux) * Size(oa_act) * ext;
  const Size qoe_b = Size(aux) * Size(ob_act) * ext;
  const Size max_qoe = std::max(qoe_a, qoe_b);

  DeviceBuf d_qoe_scratch, d_metric_buf, d_bar_metric_buf, d_m_tmp;
  if (d_qoe_scratch.alloc(max_qoe) != cudaSuccess || d_metric_buf.alloc(qq) != cudaSuccess ||
      d_bar_metric_buf.alloc(qq) != cudaSuccess || d_m_tmp.alloc(qq) != cudaSuccess) {
    return 2;
  }

  if (rhf_spatial) {
    DeviceBuf d_Fa, d_Ka, d_Ha, d_bar_Fa, d_bar_Ka, d_bar_Sab;
    DeviceBuf d_fit_qoe, d_fit_oqe, d_bar_fit_qoe, d_bar_fit_oqe;
    if (d_Fa.alloc(ee) != cudaSuccess || d_Ka.alloc(ee) != cudaSuccess ||
        d_Ha.alloc(ee) != cudaSuccess || d_bar_Fa.alloc(ee) != cudaSuccess ||
        d_bar_Ka.alloc(ee) != cudaSuccess || d_fit_qoe.alloc(qoe_a) != cudaSuccess ||
        d_fit_oqe.alloc(qoe_a) != cudaSuccess || d_bar_fit_qoe.alloc(qoe_a) != cudaSuccess ||
        d_bar_fit_oqe.alloc(qoe_a) != cudaSuccess ||
        cudaMemcpy(d_Fa.ptr, fock_a, ee * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_Ka.ptr, exch_a, ee * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(d_bar_Fa.ptr, 0, ee * sizeof(double)) != cudaSuccess ||
        cudaMemset(d_bar_Ka.ptr, 0, ee * sizeof(double)) != cudaSuccess ||
        cudaMemset(d_bar_fit_qoe.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess ||
        cudaMemset(d_bar_fit_oqe.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess) {
      return 2;
    }
    if (include_f12 && bar_cross_overlap) {
      if (d_bar_Sab.alloc(ee) != cudaSuccess ||
          cudaMemset(d_bar_Sab.ptr, 0, ee * sizeof(double)) != cudaSuccess) {
        return 2;
      }
    }
    k_add_2d<<<grid_for(ee), 256>>>(ee, d_Fa.ptr, d_Ka.ptr, d_Ha.ptr);
    if (!pack_active_qoe_checked_cuda_host(fit_a, aux, oa, fa, oa_act, ext, h_qoe)) return -2;
    if (cudaMemcpy(d_fit_qoe.ptr, h_qoe.data(), qoe_a * sizeof(double), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
      return 3;
    }
    k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_fit_qoe.ptr, d_fit_oqe.ptr);

    DeviceBuf d_fg_d, d_fg_x, d_uf_d, d_uf_x, d_f2_d, d_f2_x, d_sq_fk_d, d_sq_fk_x;
    if (include_f12) {
      // Stage 1a: Forward scalars for gc, dc, sq (reuses validated build_ss_stage1_scalars_gpu).
      if (int rc = build_ss_stage1_scalars_gpu(
              h, oa, fa, ext, aux, d_fit_qoe.ptr, d_Ha.ptr, raw_gc_a, raw_dc_a, raw_sq_a, metric_gc,
              metric_dc, metric_sq, d_qoe_scratch, d_metric_buf, d_fg_d, d_fg_x, d_uf_d, d_uf_x,
              d_f2_d, d_f2_x, d_sq_fk_d, d_sq_fk_x);
          rc != 0) {
        return rc;
      }
      // Stage 1b: Sequential 1-operator-at-a-time reverse pullback (gc, dc, sq) reusing ONE set of buffers.
      {
        DeviceBuf d_raw_op_oqe, d_res_op_oqe, d_bar_raw_op_oqe, d_bar_res_op_oqe,
            d_bar_res_os_op_oqe;
        if (d_raw_op_oqe.alloc(qoe_a) != cudaSuccess || d_res_op_oqe.alloc(qoe_a) != cudaSuccess ||
            d_bar_raw_op_oqe.alloc(qoe_a) != cudaSuccess ||
            d_bar_res_op_oqe.alloc(qoe_a) != cudaSuccess ||
            d_bar_res_os_op_oqe.alloc(qoe_a) != cudaSuccess) {
          return 2;
        }
        const double* raw_ops[3] = {raw_gc_a, raw_dc_a, raw_sq_a};
        const double* met_ops[3] = {metric_gc, metric_dc, metric_sq};
        double* bar_raw_ops[3] = {bar_raw_gc_a, bar_raw_dc_a, bar_raw_sq_a};
        double* bar_met_ops[3] = {bar_metric_gc, bar_metric_dc, bar_metric_sq};
        for (int op_kind = 0; op_kind < 3; ++op_kind) {
          if (!pack_active_qoe_checked_cuda_host(raw_ops[op_kind], aux, oa, fa, oa_act, ext, h_qoe))
            return -2;
          if (cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_a * sizeof(double),
                         cudaMemcpyHostToDevice) != cudaSuccess ||
              cudaMemcpy(d_metric_buf.ptr, met_ops[op_kind], qq * sizeof(double),
                         cudaMemcpyHostToDevice) != cudaSuccess ||
              cudaMemset(d_bar_raw_op_oqe.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess ||
              cudaMemset(d_bar_res_op_oqe.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess ||
              cudaMemset(d_bar_res_os_op_oqe.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess ||
              cudaMemset(d_bar_metric_buf.ptr, 0, qq * sizeof(double)) != cudaSuccess) {
            return 3;
          }
          k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr,
                                                 d_raw_op_oqe.ptr);
          if (rm_gemm(h, false, false, aux, oa_act * ext, aux, -1.0, d_metric_buf.ptr, aux,
                      d_fit_qoe.ptr, oa_act * ext, 1.0, d_qoe_scratch.ptr,
                      oa_act * ext) != CUBLAS_STATUS_SUCCESS) {
            return 4;
          }
          k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr,
                                                 d_res_op_oqe.ptr);
          if (int rc = run_ss_or_rhf_stage1_op_rev_gpu(
                  h, op_kind, true, has_occ_weights, oa, fa, ext, aux, occ_w_a, fock_a,
                  d_fit_oqe.ptr, d_raw_op_oqe.ptr, d_res_op_oqe.ptr, d_Ha.ptr, d_bar_fit_oqe.ptr,
                  d_bar_raw_op_oqe.ptr, d_bar_res_op_oqe.ptr, d_bar_res_os_op_oqe.ptr, d_bar_Fa.ptr,
                  d_bar_Ka.ptr, d_bar_Sab.ptr);
              rc != 0) {
            return rc;
          }
          if (int rc = pullback_res_oqe_gpu(h, d_bar_res_op_oqe.ptr, d_fit_qoe.ptr,
                                            d_metric_buf.ptr, 2.0, false, aux, oa_act, ext,
                                            d_bar_raw_op_oqe.ptr, d_bar_fit_qoe.ptr,
                                            d_bar_metric_buf.ptr, d_qoe_scratch.ptr, d_m_tmp.ptr);
              rc != 0) {
            return rc;
          }
          if (int rc = pullback_res_oqe_gpu(h, d_bar_res_os_op_oqe.ptr, d_fit_qoe.ptr,
                                            d_metric_buf.ptr, 2.0, true, aux, oa_act, ext,
                                            d_bar_raw_op_oqe.ptr, d_bar_fit_qoe.ptr,
                                            d_bar_metric_buf.ptr, d_qoe_scratch.ptr, d_m_tmp.ptr);
              rc != 0) {
            return rc;
          }
          if (int rc = scatter_active_oqe_cuda_host(d_bar_raw_op_oqe.ptr, aux, oa, fa, oa_act, ext,
                                                    bar_raw_ops[op_kind], h_qoe, d_qoe_scratch.ptr);
              rc != 0) {
            return rc;
          }
          if (cudaMemcpy(bar_met_ops[op_kind], d_bar_metric_buf.ptr, qq * sizeof(double),
                         cudaMemcpyDeviceToHost) != cudaSuccess) {
            return 6;
          }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) return 7;
      }
    }

    // Stage 2: Core Coulomb + geminal g response.
    DeviceBuf d_raw_coul_oqe, d_bar_raw_coul_oqe;
    DeviceBuf d_raw_g_oqe, d_res_g_oqe, d_bar_raw_g_oqe, d_bar_res_g_oqe, d_bar_res_os_g_oqe;
    if (d_raw_coul_oqe.alloc(qoe_a) != cudaSuccess ||
        d_bar_raw_coul_oqe.alloc(qoe_a) != cudaSuccess ||
        cudaMemset(d_bar_raw_coul_oqe.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess) {
      return 2;
    }
    if (!pack_active_qoe_checked_cuda_host(raw_coul_a, aux, raw_coul_n1_a, fa, oa_act, ext, h_qoe))
      return -2;
    if (cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_a * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return 3;
    }
    k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr, d_raw_coul_oqe.ptr);

    if (include_f12) {
      if (d_raw_g_oqe.alloc(qoe_a) != cudaSuccess || d_res_g_oqe.alloc(qoe_a) != cudaSuccess ||
          d_bar_raw_g_oqe.alloc(qoe_a) != cudaSuccess ||
          d_bar_res_g_oqe.alloc(qoe_a) != cudaSuccess ||
          d_bar_res_os_g_oqe.alloc(qoe_a) != cudaSuccess ||
          cudaMemset(d_bar_raw_g_oqe.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_res_g_oqe.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_res_os_g_oqe.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_metric_buf.ptr, 0, qq * sizeof(double)) != cudaSuccess) {
        return 2;
      }
      if (!pack_active_qoe_checked_cuda_host(raw_g_a, aux, oa, fa, oa_act, ext, h_qoe)) return -2;
      if (cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_a * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(d_metric_buf.ptr, metric_g, qq * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess) {
        return 3;
      }
      k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr, d_raw_g_oqe.ptr);
      if (rm_gemm(h, false, false, aux, oa_act * ext, aux, -1.0, d_metric_buf.ptr, aux,
                  d_fit_qoe.ptr, oa_act * ext, 1.0, d_qoe_scratch.ptr,
                  oa_act * ext) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr, d_res_g_oqe.ptr);
    }

    if (int rc = run_ss_or_rhf_stage2_response_gpu(
            h, true, include_f12, include_ordinary, has_occ_weights, oa, fa, obs, ext, aux, 0, 0, 3,
            occ_w_a, fock_a, d_fit_oqe.ptr, d_raw_coul_oqe.ptr, d_raw_g_oqe.ptr, d_res_g_oqe.ptr,
            d_Fa.ptr, d_Ka.ptr, d_Ha.ptr, d_fg_d.ptr, d_fg_x.ptr, d_f2_d.ptr, d_f2_x.ptr,
            d_uf_d.ptr, d_uf_x.ptr, d_sq_fk_d.ptr, d_sq_fk_x.ptr, d_out_scalars.ptr, d_pair_f12.ptr,
            d_pair_ord.ptr, d_bar_fit_oqe.ptr, d_bar_raw_coul_oqe.ptr, d_bar_raw_g_oqe.ptr,
            d_bar_res_g_oqe.ptr, d_bar_res_os_g_oqe.ptr, d_bar_Fa.ptr, d_bar_Ka.ptr, d_bar_Sab.ptr,
            d_err.ptr);
        rc != 0) {
      return rc;
    }

    if (include_f12) {
      if (int rc = pullback_res_oqe_gpu(h, d_bar_res_g_oqe.ptr, d_fit_qoe.ptr, d_metric_buf.ptr,
                                        2.0, false, aux, oa_act, ext, d_bar_raw_g_oqe.ptr,
                                        d_bar_fit_qoe.ptr, d_bar_metric_buf.ptr, d_qoe_scratch.ptr,
                                        d_m_tmp.ptr);
          rc != 0) {
        return rc;
      }
      if (int rc = pullback_res_oqe_gpu(h, d_bar_res_os_g_oqe.ptr, d_fit_qoe.ptr, d_metric_buf.ptr,
                                        2.0, true, aux, oa_act, ext, d_bar_raw_g_oqe.ptr,
                                        d_bar_fit_qoe.ptr, d_bar_metric_buf.ptr, d_qoe_scratch.ptr,
                                        d_m_tmp.ptr);
          rc != 0) {
        return rc;
      }
      if (int rc = scatter_active_oqe_cuda_host(d_bar_raw_g_oqe.ptr, aux, oa, fa, oa_act, ext,
                                                bar_raw_g_a, h_qoe, d_qoe_scratch.ptr);
          rc != 0) {
        return rc;
      }
      if (cudaMemcpy(bar_metric_g, d_bar_metric_buf.ptr, qq * sizeof(double),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 6;
      }
    }

    // Fold d_bar_fit_oqe into d_bar_fit_qoe & scatter all outputs to host.
    k_oqe_to_qoe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_bar_fit_oqe.ptr, d_qoe_scratch.ptr);
    k_axpy<<<grid_for(qoe_a), 256>>>(qoe_a, 1.0, d_qoe_scratch.ptr, d_bar_fit_qoe.ptr);
    if (cudaDeviceSynchronize() != cudaSuccess) return 7;
    if (int rc = scatter_active_qoe_cuda_host(d_bar_fit_qoe.ptr, aux, oa, fa, oa_act, ext,
                                              bar_fit_a, h_qoe);
        rc != 0) {
      return rc;
    }
    if (int rc = scatter_active_oqe_cuda_host(d_bar_raw_coul_oqe.ptr, aux, oa, fa, oa_act, ext,
                                              bar_raw_coul_a, h_qoe, d_qoe_scratch.ptr);
        rc != 0) {
      return rc;
    }
    if (cudaMemcpy(bar_fock_a, d_bar_Fa.ptr, ee * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(bar_exch_a, d_bar_Ka.ptr, ee * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
      return 6;
    }
    if (include_f12 && bar_cross_overlap &&
        cudaMemcpy(bar_cross_overlap, d_bar_Sab.ptr, ee * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
      return 6;
    }

    // Mirror alpha outputs to beta outputs when distinct pointers are provided.
    if (bar_fit_b && bar_fit_b != bar_fit_a) std::copy_n(bar_fit_a, qoa_e, bar_fit_b);
    if (bar_raw_coul_b && bar_raw_coul_b != bar_raw_coul_a) {
      std::copy_n(bar_raw_coul_a, qoa_e, bar_raw_coul_b);
    }
    if (include_f12) {
      if (bar_raw_g_b && bar_raw_g_b != bar_raw_g_a) std::copy_n(bar_raw_g_a, qoa_e, bar_raw_g_b);
      if (bar_raw_sq_b && bar_raw_sq_b != bar_raw_sq_a) {
        std::copy_n(bar_raw_sq_a, qoa_e, bar_raw_sq_b);
      }
      if (bar_raw_gc_b && bar_raw_gc_b != bar_raw_gc_a) {
        std::copy_n(bar_raw_gc_a, qoa_e, bar_raw_gc_b);
      }
      if (bar_raw_dc_b && bar_raw_dc_b != bar_raw_dc_a) {
        std::copy_n(bar_raw_dc_a, qoa_e, bar_raw_dc_b);
      }
    }
    if (bar_fock_b && bar_fock_b != bar_fock_a) std::copy_n(bar_fock_a, ee, bar_fock_b);
    if (bar_exch_b && bar_exch_b != bar_exch_a) std::copy_n(bar_exch_a, ee, bar_exch_b);
  } else {
    // General UHF dual-spin path: AA, AB, BB.
    DeviceBuf d_Fa, d_Fb, d_Ka, d_Kb, d_Ha, d_Hb, d_Fa_sym, d_Fb_sym, d_Ha_sym, d_Hb_sym;
    DeviceBuf d_Sab, d_Ha_S, d_Hb_ST;
    DeviceBuf d_bar_Fa, d_bar_Fb, d_bar_Ka, d_bar_Kb, d_bar_Sab;
    DeviceBuf d_fit_a_qoe, d_fit_a_oqe, d_bar_fit_a_qoe, d_bar_fit_a_oqe;
    DeviceBuf d_fit_b_qoe, d_fit_b_oqe, d_bar_fit_b_qoe, d_bar_fit_b_oqe;

    if (d_Fa.alloc(ee) != cudaSuccess || d_Fb.alloc(ee) != cudaSuccess ||
        d_Ka.alloc(ee) != cudaSuccess || d_Kb.alloc(ee) != cudaSuccess ||
        d_Ha.alloc(ee) != cudaSuccess || d_Hb.alloc(ee) != cudaSuccess ||
        d_Fa_sym.alloc(ee) != cudaSuccess || d_Fb_sym.alloc(ee) != cudaSuccess ||
        d_Ha_sym.alloc(ee) != cudaSuccess || d_Hb_sym.alloc(ee) != cudaSuccess ||
        d_bar_Fa.alloc(ee) != cudaSuccess || d_bar_Fb.alloc(ee) != cudaSuccess ||
        d_bar_Ka.alloc(ee) != cudaSuccess || d_bar_Kb.alloc(ee) != cudaSuccess ||
        cudaMemcpy(d_Fa.ptr, fock_a, ee * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_Fb.ptr, fock_b, ee * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_Ka.ptr, exch_a, ee * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_Kb.ptr, exch_b, ee * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(d_bar_Fa.ptr, 0, ee * sizeof(double)) != cudaSuccess ||
        cudaMemset(d_bar_Fb.ptr, 0, ee * sizeof(double)) != cudaSuccess ||
        cudaMemset(d_bar_Ka.ptr, 0, ee * sizeof(double)) != cudaSuccess ||
        cudaMemset(d_bar_Kb.ptr, 0, ee * sizeof(double)) != cudaSuccess) {
      return 2;
    }
    k_build_ab_sym_mats<<<grid_for(ee), 256>>>(ext, d_Fa.ptr, d_Fb.ptr, d_Ka.ptr, d_Kb.ptr,
                                               d_Ha.ptr, d_Hb.ptr, d_Fa_sym.ptr, d_Fb_sym.ptr,
                                               d_Ha_sym.ptr, d_Hb_sym.ptr);

    if (oa_act > 0 && ob_act > 0 && include_f12) {
      if (d_Sab.alloc(ee) != cudaSuccess || d_Ha_S.alloc(ee) != cudaSuccess ||
          d_Hb_ST.alloc(ee) != cudaSuccess ||
          cudaMemcpy(d_Sab.ptr, cross_overlap, ee * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess) {
        return 2;
      }
      if (rm_gemm(h, false, false, ext, ext, ext, 1.0, d_Ha.ptr, ext, d_Sab.ptr, ext, 0.0,
                  d_Ha_S.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, true, ext, ext, ext, 1.0, d_Hb.ptr, ext, d_Sab.ptr, ext, 0.0,
                  d_Hb_ST.ptr, ext) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      if (bar_cross_overlap) {
        if (d_bar_Sab.alloc(ee) != cudaSuccess ||
            cudaMemset(d_bar_Sab.ptr, 0, ee * sizeof(double)) != cudaSuccess) {
          return 2;
        }
      }
    }

    if (oa_act > 0) {
      if (d_fit_a_qoe.alloc(qoe_a) != cudaSuccess || d_fit_a_oqe.alloc(qoe_a) != cudaSuccess ||
          d_bar_fit_a_qoe.alloc(qoe_a) != cudaSuccess ||
          d_bar_fit_a_oqe.alloc(qoe_a) != cudaSuccess ||
          cudaMemset(d_bar_fit_a_qoe.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_fit_a_oqe.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess) {
        return 2;
      }
      if (!pack_active_qoe_checked_cuda_host(fit_a, aux, oa, fa, oa_act, ext, h_qoe)) return -2;
      if (cudaMemcpy(d_fit_a_qoe.ptr, h_qoe.data(), qoe_a * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return 3;
      }
      k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_fit_a_qoe.ptr, d_fit_a_oqe.ptr);
    }
    if (ob_act > 0) {
      if (d_fit_b_qoe.alloc(qoe_b) != cudaSuccess || d_fit_b_oqe.alloc(qoe_b) != cudaSuccess ||
          d_bar_fit_b_qoe.alloc(qoe_b) != cudaSuccess ||
          d_bar_fit_b_oqe.alloc(qoe_b) != cudaSuccess ||
          cudaMemset(d_bar_fit_b_qoe.ptr, 0, qoe_b * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bar_fit_b_oqe.ptr, 0, qoe_b * sizeof(double)) != cudaSuccess) {
        return 2;
      }
      if (!pack_active_qoe_checked_cuda_host(fit_b, aux, ob, fb, ob_act, ext, h_qoe)) return -2;
      if (cudaMemcpy(d_fit_b_qoe.ptr, h_qoe.data(), qoe_b * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return 3;
      }
      k_qoe_to_oqe<<<grid_for(qoe_b), 256>>>(aux, ob_act, ext, d_fit_b_qoe.ptr, d_fit_b_oqe.ptr);
    }

    // Stage 1: Forward scalars + 1-operator-at-a-time reverse pullback across AA, AB, BB.
    DeviceBuf d_fg_d_a, d_fg_x_a, d_uf_d_a, d_uf_x_a, d_f2_d_a, d_f2_x_a, d_sq_fk_d_a, d_sq_fk_x_a;
    DeviceBuf d_fg_d_b, d_fg_x_b, d_uf_d_b, d_uf_x_b, d_f2_d_b, d_f2_x_b, d_sq_fk_d_b, d_sq_fk_x_b;
    DeviceBuf d_gc_ab, d_dc_ab, d_sq_norm_ab, d_sq_fk_ab;
    if (include_f12) {
      if (oa_act >= 2) {
        if (int rc = build_ss_stage1_scalars_gpu(
                h, oa, fa, ext, aux, d_fit_a_qoe.ptr, d_Ha.ptr, raw_gc_a, raw_dc_a, raw_sq_a,
                metric_gc, metric_dc, metric_sq, d_qoe_scratch, d_metric_buf, d_fg_d_a, d_fg_x_a,
                d_uf_d_a, d_uf_x_a, d_f2_d_a, d_f2_x_a, d_sq_fk_d_a, d_sq_fk_x_a);
            rc != 0) {
          return rc;
        }
      }
      if (oa_act > 0 && ob_act > 0) {
        if (int rc = build_ab_stage1_scalars_gpu(
                h, oa, ob, fa, fb, ext, aux, d_fit_a_qoe.ptr, d_fit_b_qoe.ptr, d_Fa.ptr, d_Fb.ptr,
                d_Ka.ptr, d_Kb.ptr, d_Sab.ptr, raw_gc_a, raw_gc_b, raw_dc_a, raw_dc_b, raw_sq_a,
                raw_sq_b, metric_gc, metric_dc, metric_sq, d_qoe_scratch, d_metric_buf, d_gc_ab,
                d_dc_ab, d_sq_norm_ab, d_sq_fk_ab);
            rc != 0) {
          return rc;
        }
      }
      if (ob_act >= 2) {
        if (int rc = build_ss_stage1_scalars_gpu(
                h, ob, fb, ext, aux, d_fit_b_qoe.ptr, d_Hb.ptr, raw_gc_b, raw_dc_b, raw_sq_b,
                metric_gc, metric_dc, metric_sq, d_qoe_scratch, d_metric_buf, d_fg_d_b, d_fg_x_b,
                d_uf_d_b, d_uf_x_b, d_f2_d_b, d_f2_x_b, d_sq_fk_d_b, d_sq_fk_x_b);
            rc != 0) {
          return rc;
        }
      }

      // Sequential 1-operator-at-a-time reverse pullback for gc, dc, sq across AA, AB, BB.
      {
        DeviceBuf d_raw_a, d_res_a, d_braw_a, d_bres_a;
        DeviceBuf d_raw_b, d_res_b, d_braw_b, d_bres_b;
        if (oa_act > 0) {
          if (d_raw_a.alloc(qoe_a) != cudaSuccess || d_braw_a.alloc(qoe_a) != cudaSuccess) return 2;
          if (oa_act >= 2 &&
              (d_res_a.alloc(qoe_a) != cudaSuccess || d_bres_a.alloc(qoe_a) != cudaSuccess))
            return 2;
        }
        if (ob_act > 0) {
          if (d_raw_b.alloc(qoe_b) != cudaSuccess || d_res_b.alloc(qoe_b) != cudaSuccess ||
              d_braw_b.alloc(qoe_b) != cudaSuccess || d_bres_b.alloc(qoe_b) != cudaSuccess) {
            return 2;
          }
        }
        const double* raw_ops_a[3] = {raw_gc_a, raw_dc_a, raw_sq_a};
        const double* raw_ops_b[3] = {raw_gc_b, raw_dc_b, raw_sq_b};
        const double* met_ops[3] = {metric_gc, metric_dc, metric_sq};
        double* braw_ops_a[3] = {bar_raw_gc_a, bar_raw_dc_a, bar_raw_sq_a};
        double* braw_ops_b[3] = {bar_raw_gc_b, bar_raw_dc_b, bar_raw_sq_b};
        double* bmet_ops[3] = {bar_metric_gc, bar_metric_dc, bar_metric_sq};

        for (int op_kind = 0; op_kind < 3; ++op_kind) {
          if (cudaMemcpy(d_metric_buf.ptr, met_ops[op_kind], qq * sizeof(double),
                         cudaMemcpyHostToDevice) != cudaSuccess ||
              cudaMemset(d_bar_metric_buf.ptr, 0, qq * sizeof(double)) != cudaSuccess) {
            return 3;
          }
          if (oa_act > 0) {
            if (!pack_active_qoe_checked_cuda_host(raw_ops_a[op_kind], aux, oa, fa, oa_act, ext,
                                                   h_qoe))
              return -2;
            if (cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_a * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemset(d_braw_a.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess) {
              return 3;
            }
            k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr,
                                                   d_raw_a.ptr);
            if (oa_act >= 2) {
              if (cudaMemset(d_bres_a.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess) return 2;
              if (rm_gemm(h, false, false, aux, oa_act * ext, aux, -1.0, d_metric_buf.ptr, aux,
                          d_fit_a_qoe.ptr, oa_act * ext, 1.0, d_qoe_scratch.ptr,
                          oa_act * ext) != CUBLAS_STATUS_SUCCESS) {
                return 4;
              }
              k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr,
                                                     d_res_a.ptr);
              if (int rc = run_ss_or_rhf_stage1_op_rev_gpu(
                      h, op_kind, false, has_occ_weights, oa, fa, ext, aux, occ_w_a, fock_a,
                      d_fit_a_oqe.ptr, d_raw_a.ptr, d_res_a.ptr, d_Ha.ptr, d_bar_fit_a_oqe.ptr,
                      d_braw_a.ptr, d_bres_a.ptr, nullptr, d_bar_Fa.ptr, d_bar_Ka.ptr, nullptr);
                  rc != 0) {
                return rc;
              }
            }
          }
          if (ob_act > 0) {
            if (!pack_active_qoe_checked_cuda_host(raw_ops_b[op_kind], aux, ob, fb, ob_act, ext,
                                                   h_qoe))
              return -2;
            if (cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_b * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemset(d_braw_b.ptr, 0, qoe_b * sizeof(double)) != cudaSuccess ||
                cudaMemset(d_bres_b.ptr, 0, qoe_b * sizeof(double)) != cudaSuccess) {
              return 3;
            }
            k_qoe_to_oqe<<<grid_for(qoe_b), 256>>>(aux, ob_act, ext, d_qoe_scratch.ptr,
                                                   d_raw_b.ptr);
            if (rm_gemm(h, false, false, aux, ob_act * ext, aux, -1.0, d_metric_buf.ptr, aux,
                        d_fit_b_qoe.ptr, ob_act * ext, 1.0, d_qoe_scratch.ptr,
                        ob_act * ext) != CUBLAS_STATUS_SUCCESS) {
              return 4;
            }
            k_qoe_to_oqe<<<grid_for(qoe_b), 256>>>(aux, ob_act, ext, d_qoe_scratch.ptr,
                                                   d_res_b.ptr);
            if (oa_act > 0) {
              if (int rc = run_ab_stage1_op_rev_gpu(
                      h, op_kind, has_occ_weights, oa, ob, fa, fb, ext, aux, occ_w_a, occ_w_b,
                      fock_a, fock_b, d_fit_a_oqe.ptr, d_fit_b_oqe.ptr, d_raw_a.ptr, d_res_b.ptr,
                      d_Ha.ptr, d_Hb.ptr, d_Sab.ptr, d_Ha_S.ptr, d_Hb_ST.ptr, d_bar_fit_a_oqe.ptr,
                      d_bar_fit_b_oqe.ptr, d_braw_a.ptr, d_bres_b.ptr, d_bar_Fa.ptr, d_bar_Fb.ptr,
                      d_bar_Ka.ptr, d_bar_Kb.ptr, d_bar_Sab.ptr);
                  rc != 0) {
                return rc;
              }
            }
            if (ob_act >= 2) {
              if (int rc = run_ss_or_rhf_stage1_op_rev_gpu(
                      h, op_kind, false, has_occ_weights, ob, fb, ext, aux, occ_w_b, fock_b,
                      d_fit_b_oqe.ptr, d_raw_b.ptr, d_res_b.ptr, d_Hb.ptr, d_bar_fit_b_oqe.ptr,
                      d_braw_b.ptr, d_bres_b.ptr, nullptr, d_bar_Fb.ptr, d_bar_Kb.ptr, nullptr);
                  rc != 0) {
                return rc;
              }
            }
          }

          if (oa_act >= 2) {
            if (int rc = pullback_res_oqe_gpu(h, d_bres_a.ptr, d_fit_a_qoe.ptr, d_metric_buf.ptr,
                                              1.0, false, aux, oa_act, ext, d_braw_a.ptr,
                                              d_bar_fit_a_qoe.ptr, d_bar_metric_buf.ptr,
                                              d_qoe_scratch.ptr, d_m_tmp.ptr);
                rc != 0) {
              return rc;
            }
          }
          if (ob_act > 0 && (oa_act > 0 || ob_act >= 2)) {
            if (int rc = pullback_res_oqe_gpu(h, d_bres_b.ptr, d_fit_b_qoe.ptr, d_metric_buf.ptr,
                                              1.0, false, aux, ob_act, ext, d_braw_b.ptr,
                                              d_bar_fit_b_qoe.ptr, d_bar_metric_buf.ptr,
                                              d_qoe_scratch.ptr, d_m_tmp.ptr);
                rc != 0) {
              return rc;
            }
          }
          if (oa_act > 0) {
            if (int rc = scatter_active_oqe_cuda_host(d_braw_a.ptr, aux, oa, fa, oa_act, ext,
                                                      braw_ops_a[op_kind], h_qoe,
                                                      d_qoe_scratch.ptr);
                rc != 0) {
              return rc;
            }
          }
          if (ob_act > 0) {
            if (int rc = scatter_active_oqe_cuda_host(d_braw_b.ptr, aux, ob, fb, ob_act, ext,
                                                      braw_ops_b[op_kind], h_qoe,
                                                      d_qoe_scratch.ptr);
                rc != 0) {
              return rc;
            }
          }
          if (cudaMemcpy(bmet_ops[op_kind], d_bar_metric_buf.ptr, qq * sizeof(double),
                         cudaMemcpyDeviceToHost) != cudaSuccess) {
            return 6;
          }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) return 7;
      }
    }

    // Stage 2: Core Coulomb + geminal g across AA, AB, BB.
    DeviceBuf d_raw_coul_a_oqe, d_bar_raw_coul_a_oqe, d_raw_g_a_oqe, d_res_g_a_oqe,
        d_bar_raw_g_a_oqe, d_bar_res_g_a_oqe;
    DeviceBuf d_raw_coul_b_oqe, d_bar_raw_coul_b_oqe, d_raw_g_b_oqe, d_res_g_b_oqe,
        d_bar_raw_g_b_oqe, d_bar_res_g_b_oqe;

    if (include_f12) {
      if (cudaMemcpy(d_metric_buf.ptr, metric_g, qq * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess ||
          cudaMemset(d_bar_metric_buf.ptr, 0, qq * sizeof(double)) != cudaSuccess) {
        return 3;
      }
    }
    if (oa_act > 0) {
      if (d_raw_coul_a_oqe.alloc(qoe_a) != cudaSuccess ||
          d_bar_raw_coul_a_oqe.alloc(qoe_a) != cudaSuccess ||
          cudaMemset(d_bar_raw_coul_a_oqe.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess) {
        return 2;
      }
      if (!pack_active_qoe_checked_cuda_host(raw_coul_a, aux, raw_coul_n1_a, fa, oa_act, ext,
                                             h_qoe))
        return -2;
      if (cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_a * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess)
        return 3;
      k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr,
                                             d_raw_coul_a_oqe.ptr);
      if (include_f12) {
        if (d_raw_g_a_oqe.alloc(qoe_a) != cudaSuccess ||
            d_bar_raw_g_a_oqe.alloc(qoe_a) != cudaSuccess ||
            cudaMemset(d_bar_raw_g_a_oqe.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess) {
          return 2;
        }
        if (!pack_active_qoe_checked_cuda_host(raw_g_a, aux, oa, fa, oa_act, ext, h_qoe)) return -2;
        if (cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_a * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess)
          return 3;
        k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr,
                                               d_raw_g_a_oqe.ptr);
        if (oa_act >= 2) {
          if (d_res_g_a_oqe.alloc(qoe_a) != cudaSuccess ||
              d_bar_res_g_a_oqe.alloc(qoe_a) != cudaSuccess ||
              cudaMemset(d_bar_res_g_a_oqe.ptr, 0, qoe_a * sizeof(double)) != cudaSuccess)
            return 2;
          if (rm_gemm(h, false, false, aux, oa_act * ext, aux, -1.0, d_metric_buf.ptr, aux,
                      d_fit_a_qoe.ptr, oa_act * ext, 1.0, d_qoe_scratch.ptr,
                      oa_act * ext) != CUBLAS_STATUS_SUCCESS)
            return 4;
          k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr,
                                                 d_res_g_a_oqe.ptr);
        }
      }
    }
    if (ob_act > 0) {
      if (d_raw_coul_b_oqe.alloc(qoe_b) != cudaSuccess ||
          d_bar_raw_coul_b_oqe.alloc(qoe_b) != cudaSuccess ||
          cudaMemset(d_bar_raw_coul_b_oqe.ptr, 0, qoe_b * sizeof(double)) != cudaSuccess) {
        return 2;
      }
      if (!pack_active_qoe_checked_cuda_host(raw_coul_b, aux, raw_coul_n1_b, fb, ob_act, ext,
                                             h_qoe))
        return -2;
      if (cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_b * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess)
        return 3;
      k_qoe_to_oqe<<<grid_for(qoe_b), 256>>>(aux, ob_act, ext, d_qoe_scratch.ptr,
                                             d_raw_coul_b_oqe.ptr);
      if (include_f12) {
        if (d_raw_g_b_oqe.alloc(qoe_b) != cudaSuccess ||
            d_res_g_b_oqe.alloc(qoe_b) != cudaSuccess ||
            d_bar_raw_g_b_oqe.alloc(qoe_b) != cudaSuccess ||
            d_bar_res_g_b_oqe.alloc(qoe_b) != cudaSuccess ||
            cudaMemset(d_bar_raw_g_b_oqe.ptr, 0, qoe_b * sizeof(double)) != cudaSuccess ||
            cudaMemset(d_bar_res_g_b_oqe.ptr, 0, qoe_b * sizeof(double)) != cudaSuccess) {
          return 2;
        }
        if (!pack_active_qoe_checked_cuda_host(raw_g_b, aux, ob, fb, ob_act, ext, h_qoe)) return -2;
        if (cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_b * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess)
          return 3;
        k_qoe_to_oqe<<<grid_for(qoe_b), 256>>>(aux, ob_act, ext, d_qoe_scratch.ptr,
                                               d_raw_g_b_oqe.ptr);
        if (rm_gemm(h, false, false, aux, ob_act * ext, aux, -1.0, d_metric_buf.ptr, aux,
                    d_fit_b_qoe.ptr, ob_act * ext, 1.0, d_qoe_scratch.ptr,
                    ob_act * ext) != CUBLAS_STATUS_SUCCESS)
          return 4;
        k_qoe_to_oqe<<<grid_for(qoe_b), 256>>>(aux, ob_act, ext, d_qoe_scratch.ptr,
                                               d_res_g_b_oqe.ptr);
      }
    }

    // Channel 1: AA
    if (oa_act >= 2) {
      if (int rc = run_ss_or_rhf_stage2_response_gpu(
              h, false, include_f12, include_ordinary, has_occ_weights, oa, fa, obs, ext, aux, 0, 0,
              3, occ_w_a, fock_a, d_fit_a_oqe.ptr, d_raw_coul_a_oqe.ptr, d_raw_g_a_oqe.ptr,
              d_res_g_a_oqe.ptr, d_Fa.ptr, d_Ka.ptr, d_Ha.ptr, d_fg_d_a.ptr, d_fg_x_a.ptr,
              d_f2_d_a.ptr, d_f2_x_a.ptr, d_uf_d_a.ptr, d_uf_x_a.ptr, d_sq_fk_d_a.ptr,
              d_sq_fk_x_a.ptr, d_out_scalars.ptr, d_pair_f12.ptr, d_pair_ord.ptr,
              d_bar_fit_a_oqe.ptr, d_bar_raw_coul_a_oqe.ptr, d_bar_raw_g_a_oqe.ptr,
              d_bar_res_g_a_oqe.ptr, nullptr, d_bar_Fa.ptr, d_bar_Ka.ptr, nullptr, d_err.ptr);
          rc != 0) {
        return rc;
      }
    }
    // Channel 2: AB
    if (oa_act > 0 && ob_act > 0) {
      if (int rc = run_ab_stage2_response_gpu(
              h, include_f12, include_ordinary, has_occ_weights, oa, ob, fa, fb, obs, ext, aux,
              n_aa, occ_w_a, occ_w_b, fock_a, fock_b, d_fit_a_oqe.ptr, d_fit_b_oqe.ptr,
              d_raw_coul_b_oqe.ptr, d_raw_g_a_oqe.ptr, d_res_g_b_oqe.ptr, d_Fa.ptr, d_Fb.ptr,
              d_Ha.ptr, d_Hb.ptr, d_Fa_sym.ptr, d_Fb_sym.ptr, d_Ha_sym.ptr, d_Hb_sym.ptr,
              d_Sab.ptr, d_gc_ab.ptr, d_dc_ab.ptr, d_sq_norm_ab.ptr, d_sq_fk_ab.ptr,
              d_out_scalars.ptr, d_pair_f12.ptr, d_pair_ord.ptr, d_bar_fit_a_oqe.ptr,
              d_bar_fit_b_oqe.ptr, d_bar_raw_coul_b_oqe.ptr, d_bar_raw_g_a_oqe.ptr,
              d_bar_res_g_b_oqe.ptr, d_bar_Fa.ptr, d_bar_Fb.ptr, d_bar_Ka.ptr, d_bar_Kb.ptr,
              d_bar_Sab.ptr, d_err.ptr);
          rc != 0) {
        return rc;
      }
    }
    // Channel 3: BB
    if (ob_act >= 2) {
      if (int rc = run_ss_or_rhf_stage2_response_gpu(
              h, false, include_f12, include_ordinary, has_occ_weights, ob, fb, obs, ext, aux,
              n_aa + n_ab, 2, 5, occ_w_b, fock_b, d_fit_b_oqe.ptr, d_raw_coul_b_oqe.ptr,
              d_raw_g_b_oqe.ptr, d_res_g_b_oqe.ptr, d_Fb.ptr, d_Kb.ptr, d_Hb.ptr, d_fg_d_b.ptr,
              d_fg_x_b.ptr, d_f2_d_b.ptr, d_f2_x_b.ptr, d_uf_d_b.ptr, d_uf_x_b.ptr, d_sq_fk_d_b.ptr,
              d_sq_fk_x_b.ptr, d_out_scalars.ptr, d_pair_f12.ptr, d_pair_ord.ptr,
              d_bar_fit_b_oqe.ptr, d_bar_raw_coul_b_oqe.ptr, d_bar_raw_g_b_oqe.ptr,
              d_bar_res_g_b_oqe.ptr, nullptr, d_bar_Fb.ptr, d_bar_Kb.ptr, nullptr, d_err.ptr);
          rc != 0) {
        return rc;
      }
    }

    if (include_f12) {
      if (oa_act >= 2) {
        if (int rc = pullback_res_oqe_gpu(h, d_bar_res_g_a_oqe.ptr, d_fit_a_qoe.ptr,
                                          d_metric_buf.ptr, 1.0, false, aux, oa_act, ext,
                                          d_bar_raw_g_a_oqe.ptr, d_bar_fit_a_qoe.ptr,
                                          d_bar_metric_buf.ptr, d_qoe_scratch.ptr, d_m_tmp.ptr);
            rc != 0) {
          return rc;
        }
      }
      if (ob_act > 0 && (oa_act > 0 || ob_act >= 2)) {
        if (int rc = pullback_res_oqe_gpu(h, d_bar_res_g_b_oqe.ptr, d_fit_b_qoe.ptr,
                                          d_metric_buf.ptr, 1.0, false, aux, ob_act, ext,
                                          d_bar_raw_g_b_oqe.ptr, d_bar_fit_b_qoe.ptr,
                                          d_bar_metric_buf.ptr, d_qoe_scratch.ptr, d_m_tmp.ptr);
            rc != 0) {
          return rc;
        }
      }
      if (cudaMemcpy(bar_metric_g, d_bar_metric_buf.ptr, qq * sizeof(double),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 6;
      }
    }

    if (oa_act > 0) {
      k_oqe_to_qoe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_bar_fit_a_oqe.ptr,
                                             d_qoe_scratch.ptr);
      k_axpy<<<grid_for(qoe_a), 256>>>(qoe_a, 1.0, d_qoe_scratch.ptr, d_bar_fit_a_qoe.ptr);
      if (cudaDeviceSynchronize() != cudaSuccess) return 7;
      if (int rc = scatter_active_qoe_cuda_host(d_bar_fit_a_qoe.ptr, aux, oa, fa, oa_act, ext,
                                                bar_fit_a, h_qoe);
          rc != 0)
        return rc;
      if (int rc = scatter_active_oqe_cuda_host(d_bar_raw_coul_a_oqe.ptr, aux, oa, fa, oa_act, ext,
                                                bar_raw_coul_a, h_qoe, d_qoe_scratch.ptr);
          rc != 0)
        return rc;
      if (include_f12) {
        if (int rc = scatter_active_oqe_cuda_host(d_bar_raw_g_a_oqe.ptr, aux, oa, fa, oa_act, ext,
                                                  bar_raw_g_a, h_qoe, d_qoe_scratch.ptr);
            rc != 0)
          return rc;
      }
    }
    if (ob_act > 0) {
      k_oqe_to_qoe<<<grid_for(qoe_b), 256>>>(aux, ob_act, ext, d_bar_fit_b_oqe.ptr,
                                             d_qoe_scratch.ptr);
      k_axpy<<<grid_for(qoe_b), 256>>>(qoe_b, 1.0, d_qoe_scratch.ptr, d_bar_fit_b_qoe.ptr);
      if (cudaDeviceSynchronize() != cudaSuccess) return 7;
      if (int rc = scatter_active_qoe_cuda_host(d_bar_fit_b_qoe.ptr, aux, ob, fb, ob_act, ext,
                                                bar_fit_b, h_qoe);
          rc != 0)
        return rc;
      if (int rc = scatter_active_oqe_cuda_host(d_bar_raw_coul_b_oqe.ptr, aux, ob, fb, ob_act, ext,
                                                bar_raw_coul_b, h_qoe, d_qoe_scratch.ptr);
          rc != 0)
        return rc;
      if (include_f12) {
        if (int rc = scatter_active_oqe_cuda_host(d_bar_raw_g_b_oqe.ptr, aux, ob, fb, ob_act, ext,
                                                  bar_raw_g_b, h_qoe, d_qoe_scratch.ptr);
            rc != 0)
          return rc;
      }
    }
    if (cudaMemcpy(bar_fock_a, d_bar_Fa.ptr, ee * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(bar_fock_b, d_bar_Fb.ptr, ee * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(bar_exch_a, d_bar_Ka.ptr, ee * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(bar_exch_b, d_bar_Kb.ptr, ee * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
      return 6;
    }
    if (include_f12 && oa_act > 0 && ob_act > 0 && bar_cross_overlap &&
        cudaMemcpy(bar_cross_overlap, d_bar_Sab.ptr, ee * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
      return 6;
    }
  }

  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 7;
  int h_err = 0;
  if (cudaMemcpy(&h_err, d_err.ptr, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 6;
  if (h_err != 0) return h_err;

  if (out_scalars) {
    if (cudaMemcpy(out_scalars, d_out_scalars.ptr, 6 * sizeof(double), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      return 6;
    }
    if (!all_finite_host(out_scalars, 6)) return -5;
  }
  if (include_f12 && pair_f12_out) {
    if (cudaMemcpy(pair_f12_out, d_pair_f12.ptr, n_total * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return 6;
    }
    if (!all_finite_host(pair_f12_out, n_total)) return -5;
  }
  if (include_ordinary && pair_ord_out) {
    if (cudaMemcpy(pair_ord_out, d_pair_ord.ptr, n_total * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return 6;
    }
    if (!all_finite_host(pair_ord_out, n_total)) return -5;
  }
  if (!all_finite_host(bar_fock_a, ee) || !all_finite_host(bar_exch_a, ee)) return -5;
  if (!rhf_spatial && (!all_finite_host(bar_fock_b, ee) || !all_finite_host(bar_exch_b, ee))) {
    return -5;
  }
  if (include_f12 && bar_cross_overlap && !all_finite_host(bar_cross_overlap, ee)) return -5;
  if (include_f12 && (!all_finite_host(bar_metric_g, qq) || !all_finite_host(bar_metric_sq, qq) ||
                      !all_finite_host(bar_metric_gc, qq) || !all_finite_host(bar_metric_dc, qq))) {
    return -5;
  }
  if (oa_act > 0) {
    if (!all_finite_host(bar_fit_a, qoa_e) || !all_finite_host(bar_raw_coul_a, qoa_e)) return -5;
    if (include_f12 &&
        (!all_finite_host(bar_raw_g_a, qoa_e) || !all_finite_host(bar_raw_sq_a, qoa_e) ||
         !all_finite_host(bar_raw_gc_a, qoa_e) || !all_finite_host(bar_raw_dc_a, qoa_e))) {
      return -5;
    }
  }
  if (!rhf_spatial && ob_act > 0) {
    if (!all_finite_host(bar_fit_b, qob_e) || !all_finite_host(bar_raw_coul_b, qob_e)) return -5;
    if (include_f12 &&
        (!all_finite_host(bar_raw_g_b, qob_e) || !all_finite_host(bar_raw_sq_b, qob_e) ||
         !all_finite_host(bar_raw_gc_b, qob_e) || !all_finite_host(bar_raw_dc_b, qob_e))) {
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

int gqc_f12_mp2_pair_energies_cuda(
    int rhf_spatial_int, int include_ordinary_int, int has_occ_weights_int, int oa, int ob, int fa,
    int fb, int obs, int ext, int aux, int raw_coul_n1_a, int raw_coul_n1_b,
    std::uint64_t budget_bytes, const double* occ_w_a, const double* occ_w_b, const double* fit_a,
    const double* fit_b, const double* raw_coul_a, const double* raw_coul_b, const double* raw_g_a,
    const double* raw_g_b, const double* raw_sq_a, const double* raw_sq_b, const double* raw_gc_a,
    const double* raw_gc_b, const double* raw_dc_a, const double* raw_dc_b, const double* metric_g,
    const double* metric_sq, const double* metric_gc, const double* metric_dc, const double* fock_a,
    const double* fock_b, const double* exch_a, const double* exch_b, const double* cross_overlap,
    double* out_scalars, double* pair_f12_out, double* pair_ord_out) {
  try {
  const bool rhf_spatial = (rhf_spatial_int != 0);
  const bool include_ordinary = (include_ordinary_int != 0);
  const bool has_occ_weights = (has_occ_weights_int != 0);

  if (fa < 0 || fa > oa || oa >= obs || fb < 0 || fb > ob || ob >= obs || obs >= ext || aux < 1) {
    return -1;
  }
  if (rhf_spatial && (oa != ob || fa != fb)) return -1;
  if (!metric_g || !metric_sq || !metric_gc || !metric_dc || !fock_a || !exch_a) return -1;
  if (!rhf_spatial && (!fock_b || !exch_b)) return -1;

  const int oa_act = oa - fa;
  const int ob_act = ob - fb;
  if (has_occ_weights) {
    if (oa_act > 0 && !occ_w_a) return -1;
    if (!rhf_spatial && ob_act > 0 && !occ_w_b) return -1;
  }
  if (oa_act > 0) {
    if (raw_coul_n1_a < oa || !fit_a || !raw_coul_a || !raw_g_a || !raw_sq_a || !raw_gc_a ||
        !raw_dc_a) {
      return -1;
    }
  }
  if (!rhf_spatial && ob_act > 0) {
    if (raw_coul_n1_b < ob || !fit_b || !raw_coul_b || !raw_g_b || !raw_sq_b || !raw_gc_b ||
        !raw_dc_b) {
      return -1;
    }
  }
  if (!rhf_spatial && oa_act > 0 && ob_act > 0 && !cross_overlap) return -1;

  const Size qq = Size(aux) * aux;
  const Size ee = Size(ext) * ext;
  if (!all_finite_host(metric_g, qq) || !all_finite_host(metric_sq, qq) ||
      !all_finite_host(metric_gc, qq) || !all_finite_host(metric_dc, qq) ||
      !all_finite_host(fock_a, ee) || !all_finite_host(exch_a, ee)) {
    return -2;
  }
  if (!rhf_spatial && (!all_finite_host(fock_b, ee) || !all_finite_host(exch_b, ee))) {
    return -2;
  }
  if (!rhf_spatial && oa_act > 0 && ob_act > 0 && !all_finite_host(cross_overlap, ee)) {
    return -2;
  }
  if (has_occ_weights) {
    if (oa_act > 0 && !all_finite_host(occ_w_a, oa_act)) return -2;
    if (!rhf_spatial && ob_act > 0 && !all_finite_host(occ_w_b, ob_act)) return -2;
  }

  const Size n_aa = (oa_act >= 2) ? (Size(oa_act) * (oa_act - 1)) / 2 : 0;
  const Size n_ab = Size(oa_act) * Size(ob_act);
  const Size n_bb = (ob_act >= 2) ? (Size(ob_act) * (ob_act - 1)) / 2 : 0;
  const Size n_total = n_aa + n_ab + n_bb;

  if (out_scalars) std::fill_n(out_scalars, 6, 0.0);
  if (pair_f12_out && n_total > 0) std::fill_n(pair_f12_out, n_total, 0.0);
  if (include_ordinary && pair_ord_out && n_total > 0) std::fill_n(pair_ord_out, n_total, 0.0);

  if (n_total == 0) {
    std::vector<double> chk;
    if (oa_act > 0) {
      if (!pack_active_qoe_checked_cuda_host(fit_a, aux, oa, fa, oa_act, ext, chk) ||
          !pack_active_qoe_checked_cuda_host(raw_coul_a, aux, raw_coul_n1_a, fa, oa_act, ext,
                                             chk) ||
          !pack_active_qoe_checked_cuda_host(raw_g_a, aux, oa, fa, oa_act, ext, chk) ||
          !pack_active_qoe_checked_cuda_host(raw_sq_a, aux, oa, fa, oa_act, ext, chk) ||
          !pack_active_qoe_checked_cuda_host(raw_gc_a, aux, oa, fa, oa_act, ext, chk) ||
          !pack_active_qoe_checked_cuda_host(raw_dc_a, aux, oa, fa, oa_act, ext, chk)) {
        return -2;
      }
    }
    if (!rhf_spatial && ob_act > 0) {
      if (!pack_active_qoe_checked_cuda_host(fit_b, aux, ob, fb, ob_act, ext, chk) ||
          !pack_active_qoe_checked_cuda_host(raw_coul_b, aux, raw_coul_n1_b, fb, ob_act, ext,
                                             chk) ||
          !pack_active_qoe_checked_cuda_host(raw_g_b, aux, ob, fb, ob_act, ext, chk) ||
          !pack_active_qoe_checked_cuda_host(raw_sq_b, aux, ob, fb, ob_act, ext, chk) ||
          !pack_active_qoe_checked_cuda_host(raw_gc_b, aux, ob, fb, ob_act, ext, chk) ||
          !pack_active_qoe_checked_cuda_host(raw_dc_b, aux, ob, fb, ob_act, ext, chk)) {
        return -2;
      }
    }
    return 0;
  }

  // The effective cap bounds allocations; the target controls multi-pair batching.
  const Size eff_budget = effective_device_budget(budget_bytes);
  const Size req_device_bytes = estimate_mp2_pair_energies_device_bytes(
      rhf_spatial, include_ordinary, has_occ_weights, pair_f12_out != nullptr,
      pair_ord_out != nullptr, oa, ob, fa, fb, obs, ext, aux);
  if (req_device_bytes > eff_budget) {
    return kErrBudgetExceeded;
  }
  const Size ws_cap = std::min<Size>(eff_budget, Size(512) << 20);

  DeviceWorkspace workspace(budget_bytes);
  CublasGuard cg;
  if (cg.init() != CUBLAS_STATUS_SUCCESS) return 1;
  cublasHandle_t h = cg.handle;

  DeviceBuf d_out_scalars, d_pair_f12, d_pair_ord, d_occ_w_a, d_occ_w_b;
  DeviceIntBuf d_err;
  if (d_out_scalars.alloc(6) != cudaSuccess || d_err.alloc(1) != cudaSuccess ||
      cudaMemset(d_out_scalars.ptr, 0, 6 * sizeof(double)) != cudaSuccess ||
      cudaMemset(d_err.ptr, 0, sizeof(int)) != cudaSuccess) {
    return 2;
  }
  if (pair_f12_out) {
    if (d_pair_f12.alloc(n_total) != cudaSuccess ||
        cudaMemset(d_pair_f12.ptr, 0, n_total * sizeof(double)) != cudaSuccess) {
      return 2;
    }
  }
  if (include_ordinary && pair_ord_out) {
    if (d_pair_ord.alloc(n_total) != cudaSuccess ||
        cudaMemset(d_pair_ord.ptr, 0, n_total * sizeof(double)) != cudaSuccess) {
      return 2;
    }
  }
  if (has_occ_weights) {
    if (oa_act > 0 && (d_occ_w_a.alloc(oa_act) != cudaSuccess ||
                       cudaMemcpy(d_occ_w_a.ptr, occ_w_a, Size(oa_act) * sizeof(double),
                                  cudaMemcpyHostToDevice) != cudaSuccess)) {
      return 2;
    }
    if (!rhf_spatial && ob_act > 0 &&
        (d_occ_w_b.alloc(ob_act) != cudaSuccess ||
         cudaMemcpy(d_occ_w_b.ptr, occ_w_b, Size(ob_act) * sizeof(double),
                    cudaMemcpyHostToDevice) != cudaSuccess)) {
      return 2;
    }
  }

  std::vector<double> h_qoe;
  const Size qoe_a = Size(aux) * Size(oa_act) * ext;
  const Size qoe_b = Size(aux) * Size(ob_act) * ext;

  if (rhf_spatial) {
    DeviceBuf d_Fa, d_Ka, d_Ha, d_fit_qoe, d_qoe_scratch, d_metric_buf;
    if (d_Fa.alloc(ee) != cudaSuccess || d_Ka.alloc(ee) != cudaSuccess ||
        d_Ha.alloc(ee) != cudaSuccess || d_fit_qoe.alloc(qoe_a) != cudaSuccess ||
        d_qoe_scratch.alloc(qoe_a) != cudaSuccess || d_metric_buf.alloc(qq) != cudaSuccess) {
      return 2;
    }
    if (!pack_active_qoe_checked_cuda_host(fit_a, aux, oa, fa, oa_act, ext, h_qoe)) return -2;
    if (cudaMemcpy(d_Fa.ptr, fock_a, ee * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_Ka.ptr, exch_a, ee * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_fit_qoe.ptr, h_qoe.data(), qoe_a * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
      return 3;
    }
    k_add_2d<<<grid_for(ee), 256>>>(ee, d_Fa.ptr, d_Ka.ptr, d_Ha.ptr);

    DeviceBuf d_fg_d, d_fg_x, d_uf_d, d_uf_x, d_f2_d, d_f2_x, d_sq_fk_d, d_sq_fk_x;
    if (int rc = build_ss_stage1_scalars_gpu(
            h, oa, fa, ext, aux, d_fit_qoe.ptr, d_Ha.ptr, raw_gc_a, raw_dc_a, raw_sq_a, metric_gc,
            metric_dc, metric_sq, d_qoe_scratch, d_metric_buf, d_fg_d, d_fg_x, d_uf_d, d_uf_x,
            d_f2_d, d_f2_x, d_sq_fk_d, d_sq_fk_x);
        rc != 0) {
      return rc;
    }
    d_Ha.reset();

    // Build GPU-resident OQE geminal, fit, and Coulomb 3-center tensors (<= 4 tensors in loop).
    DeviceBuf d_fit_oqe, d_raw_g_oqe, d_res_g_oqe, d_raw_coul_oqe;
    if (d_raw_g_oqe.alloc(qoe_a) != cudaSuccess || d_res_g_oqe.alloc(qoe_a) != cudaSuccess) {
      return 2;
    }
    if (!pack_active_qoe_checked_cuda_host(raw_g_a, aux, oa, fa, oa_act, ext, h_qoe)) return -2;
    if (cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_a * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_metric_buf.ptr, metric_g, qq * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
      return 3;
    }
    k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr, d_raw_g_oqe.ptr);
    if (rm_gemm(h, false, false, aux, oa_act * ext, aux, -1.0, d_metric_buf.ptr, aux, d_fit_qoe.ptr,
                oa_act * ext, 1.0, d_qoe_scratch.ptr, oa_act * ext) != CUBLAS_STATUS_SUCCESS) {
      return 4;
    }
    k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr, d_res_g_oqe.ptr);
    d_metric_buf.reset();

    if (d_fit_oqe.alloc(qoe_a) != cudaSuccess) return 2;
    k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_fit_qoe.ptr, d_fit_oqe.ptr);
    d_fit_qoe.reset();

    if (d_raw_coul_oqe.alloc(qoe_a) != cudaSuccess) return 2;
    if (!pack_active_qoe_checked_cuda_host(raw_coul_a, aux, raw_coul_n1_a, fa, oa_act, ext,
                                           h_qoe)) {
      return -2;
    }
    if (cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_a * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return 3;
    }
    k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr, d_raw_coul_oqe.ptr);
    d_qoe_scratch.reset();

    if (int rc = run_ss_or_rhf_stage2_gpu(
            h, true, include_ordinary, has_occ_weights, oa, fa, obs, ext, aux, ws_cap, 0, 0, 3,
            d_occ_w_a.ptr, d_fit_oqe.ptr, d_raw_coul_oqe.ptr, d_raw_g_oqe.ptr, d_res_g_oqe.ptr,
            d_Fa.ptr, d_Ka.ptr, d_fg_d.ptr, d_fg_x.ptr, d_f2_d.ptr, d_f2_x.ptr, d_uf_d.ptr,
            d_uf_x.ptr, d_sq_fk_d.ptr, d_sq_fk_x.ptr, d_out_scalars.ptr, d_pair_f12.ptr,
            d_pair_ord.ptr, d_err.ptr);
        rc != 0) {
      return rc;
    }
  } else {
    // General UHF dual-spin path (AA, AB, BB) with <= 5 simultaneous 3c tensors on GPU.
    DeviceBuf d_Fa, d_Fb, d_Ka, d_Kb, d_Sab;
    if (d_Fa.alloc(ee) != cudaSuccess || d_Fb.alloc(ee) != cudaSuccess ||
        d_Ka.alloc(ee) != cudaSuccess || d_Kb.alloc(ee) != cudaSuccess ||
        cudaMemcpy(d_Fa.ptr, fock_a, ee * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_Fb.ptr, fock_b, ee * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_Ka.ptr, exch_a, ee * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_Kb.ptr, exch_b, ee * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
      return 3;
    }
    if (oa_act > 0 && ob_act > 0) {
      if (d_Sab.alloc(ee) != cudaSuccess ||
          cudaMemcpy(d_Sab.ptr, cross_overlap, ee * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess) {
        return 3;
      }
    }

    DeviceBuf d_fit_a_qoe, d_fit_b_qoe, d_qoe_scratch, d_metric_buf;
    const Size qoe_max = std::max(qoe_a, qoe_b);
    if (d_qoe_scratch.alloc(qoe_max) != cudaSuccess || d_metric_buf.alloc(qq) != cudaSuccess) {
      return 2;
    }
    if (oa_act > 0) {
      if (!pack_active_qoe_checked_cuda_host(fit_a, aux, oa, fa, oa_act, ext, h_qoe)) return -2;
      if (d_fit_a_qoe.alloc(qoe_a) != cudaSuccess ||
          cudaMemcpy(d_fit_a_qoe.ptr, h_qoe.data(), qoe_a * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return 3;
      }
    }
    if (ob_act > 0) {
      if (!pack_active_qoe_checked_cuda_host(fit_b, aux, ob, fb, ob_act, ext, h_qoe)) return -2;
      if (d_fit_b_qoe.alloc(qoe_b) != cudaSuccess ||
          cudaMemcpy(d_fit_b_qoe.ptr, h_qoe.data(), qoe_b * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return 3;
      }
    }

    // Stage 1: precompute ALL gc/dc/sq scalars for AA, AB, BB once while fit_qoe is resident.
    DeviceBuf d_fg_a_d, d_fg_a_x, d_uf_a_d, d_uf_a_x, d_f2_a_d, d_f2_a_x, d_sq_fk_a_d, d_sq_fk_a_x;
    DeviceBuf d_fg_b_d, d_fg_b_x, d_uf_b_d, d_uf_b_x, d_f2_b_d, d_f2_b_x, d_sq_fk_b_d, d_sq_fk_b_x;
    DeviceBuf d_gc_ab, d_dc_ab, d_sq_norm_ab, d_sq_fk_ab;

    if (oa_act >= 2) {
      DeviceBuf d_Ha;
      if (d_Ha.alloc(ee) != cudaSuccess) return 2;
      k_add_2d<<<grid_for(ee), 256>>>(ee, d_Fa.ptr, d_Ka.ptr, d_Ha.ptr);
      if (int rc = build_ss_stage1_scalars_gpu(
              h, oa, fa, ext, aux, d_fit_a_qoe.ptr, d_Ha.ptr, raw_gc_a, raw_dc_a, raw_sq_a,
              metric_gc, metric_dc, metric_sq, d_qoe_scratch, d_metric_buf, d_fg_a_d, d_fg_a_x,
              d_uf_a_d, d_uf_a_x, d_f2_a_d, d_f2_a_x, d_sq_fk_a_d, d_sq_fk_a_x);
          rc != 0) {
        return rc;
      }
    }
    if (oa_act > 0 && ob_act > 0) {
      if (int rc = build_ab_stage1_scalars_gpu(
              h, oa, ob, fa, fb, ext, aux, d_fit_a_qoe.ptr, d_fit_b_qoe.ptr, d_Fa.ptr, d_Fb.ptr,
              d_Ka.ptr, d_Kb.ptr, d_Sab.ptr, raw_gc_a, raw_gc_b, raw_dc_a, raw_dc_b, raw_sq_a,
              raw_sq_b, metric_gc, metric_dc, metric_sq, d_qoe_scratch, d_metric_buf, d_gc_ab,
              d_dc_ab, d_sq_norm_ab, d_sq_fk_ab);
          rc != 0) {
        return rc;
      }
    }
    if (ob_act >= 2) {
      DeviceBuf d_Hb;
      if (d_Hb.alloc(ee) != cudaSuccess) return 2;
      k_add_2d<<<grid_for(ee), 256>>>(ee, d_Fb.ptr, d_Kb.ptr, d_Hb.ptr);
      if (int rc = build_ss_stage1_scalars_gpu(
              h, ob, fb, ext, aux, d_fit_b_qoe.ptr, d_Hb.ptr, raw_gc_b, raw_dc_b, raw_sq_b,
              metric_gc, metric_dc, metric_sq, d_qoe_scratch, d_metric_buf, d_fg_b_d, d_fg_b_x,
              d_uf_b_d, d_uf_b_x, d_f2_b_d, d_f2_b_x, d_sq_fk_b_d, d_sq_fk_b_x);
          rc != 0) {
        return rc;
      }
    }

    // Stage 2: run AA -> AB -> BB, freeing unused 3c tensors immediately between channels.
    DeviceBuf d_fit_a_oqe, d_raw_g_a_oqe;
    if (oa_act > 0) {
      if (!pack_active_qoe_checked_cuda_host(raw_g_a, aux, oa, fa, oa_act, ext, h_qoe)) return -2;
      if (d_raw_g_a_oqe.alloc(qoe_a) != cudaSuccess ||
          cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_a * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return 3;
      }
      k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr,
                                             d_raw_g_a_oqe.ptr);

      DeviceBuf d_res_g_a_oqe, d_raw_coul_a_oqe;
      if (oa_act >= 2) {
        if (d_res_g_a_oqe.alloc(qoe_a) != cudaSuccess ||
            cudaMemcpy(d_metric_buf.ptr, metric_g, qq * sizeof(double), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
          return 3;
        }
        if (rm_gemm(h, false, false, aux, oa_act * ext, aux, -1.0, d_metric_buf.ptr, aux,
                    d_fit_a_qoe.ptr, oa_act * ext, 1.0, d_qoe_scratch.ptr, oa_act * ext) !=
            CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
        k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr,
                                               d_res_g_a_oqe.ptr);
      }
      if (d_fit_a_oqe.alloc(qoe_a) != cudaSuccess) return 2;
      k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_fit_a_qoe.ptr, d_fit_a_oqe.ptr);
      d_fit_a_qoe.reset();

      if (oa_act >= 2) {
        if (!pack_active_qoe_checked_cuda_host(raw_coul_a, aux, raw_coul_n1_a, fa, oa_act, ext,
                                               h_qoe)) {
          return -2;
        }
        if (d_raw_coul_a_oqe.alloc(qoe_a) != cudaSuccess ||
            cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_a * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return 3;
        }
        k_qoe_to_oqe<<<grid_for(qoe_a), 256>>>(aux, oa_act, ext, d_qoe_scratch.ptr,
                                               d_raw_coul_a_oqe.ptr);
        d_qoe_scratch.reset();
        if (int rc = run_ss_or_rhf_stage2_gpu(
                h, false, include_ordinary, has_occ_weights, oa, fa, obs, ext, aux, ws_cap, 0, 0, 3,
                d_occ_w_a.ptr, d_fit_a_oqe.ptr, d_raw_coul_a_oqe.ptr, d_raw_g_a_oqe.ptr,
                d_res_g_a_oqe.ptr, d_Fa.ptr, d_Ka.ptr, d_fg_a_d.ptr, d_fg_a_x.ptr, d_f2_a_d.ptr,
                d_f2_a_x.ptr, d_uf_a_d.ptr, d_uf_a_x.ptr, d_sq_fk_a_d.ptr, d_sq_fk_a_x.ptr,
                d_out_scalars.ptr, d_pair_f12.ptr, d_pair_ord.ptr, d_err.ptr);
            rc != 0) {
          return rc;
        }
      }
      // d_res_g_a_oqe and d_raw_coul_a_oqe are automatically freed leaving this scope!
      d_fg_a_d.reset();
      d_fg_a_x.reset();
      d_uf_a_d.reset();
      d_uf_a_x.reset();
      d_f2_a_d.reset();
      d_f2_a_x.reset();
      d_sq_fk_a_d.reset();
      d_sq_fk_a_x.reset();
    }

    // Build shared beta OQE tensors (fit_b, raw_coul_b, res_g_b) for AB and BB.
    DeviceBuf d_fit_b_oqe, d_raw_coul_b_oqe, d_res_g_b_oqe;
    if (ob_act > 0 && (oa_act > 0 || ob_act >= 2)) {
      if (!pack_active_qoe_checked_cuda_host(raw_g_b, aux, ob, fb, ob_act, ext, h_qoe)) return -2;
      if (d_qoe_scratch.alloc(qoe_b) != cudaSuccess || d_res_g_b_oqe.alloc(qoe_b) != cudaSuccess ||
          cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_b * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(d_metric_buf.ptr, metric_g, qq * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess) {
        return 3;
      }
      if (rm_gemm(h, false, false, aux, ob_act * ext, aux, -1.0, d_metric_buf.ptr, aux,
                  d_fit_b_qoe.ptr, ob_act * ext, 1.0, d_qoe_scratch.ptr, ob_act * ext) !=
          CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      k_qoe_to_oqe<<<grid_for(qoe_b), 256>>>(aux, ob_act, ext, d_qoe_scratch.ptr,
                                             d_res_g_b_oqe.ptr);
      d_metric_buf.reset();

      if (d_fit_b_oqe.alloc(qoe_b) != cudaSuccess) return 2;
      k_qoe_to_oqe<<<grid_for(qoe_b), 256>>>(aux, ob_act, ext, d_fit_b_qoe.ptr, d_fit_b_oqe.ptr);
      d_fit_b_qoe.reset();

      if (!pack_active_qoe_checked_cuda_host(raw_coul_b, aux, raw_coul_n1_b, fb, ob_act, ext,
                                             h_qoe)) {
        return -2;
      }
      if (d_raw_coul_b_oqe.alloc(qoe_b) != cudaSuccess ||
          cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_b * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return 3;
      }
      k_qoe_to_oqe<<<grid_for(qoe_b), 256>>>(aux, ob_act, ext, d_qoe_scratch.ptr,
                                             d_raw_coul_b_oqe.ptr);
      d_qoe_scratch.reset();
    }

    // Channel 2: AB
    if (oa_act > 0 && ob_act > 0) {
      if (int rc = run_ab_stage2_gpu(
              h, include_ordinary, has_occ_weights, oa, ob, fa, fb, obs, ext, aux, ws_cap, n_aa,
              d_occ_w_a.ptr, d_occ_w_b.ptr, d_fit_a_oqe.ptr, d_raw_g_a_oqe.ptr, d_fit_b_oqe.ptr,
              d_raw_coul_b_oqe.ptr, d_res_g_b_oqe.ptr, d_Fa.ptr, d_Fb.ptr, d_Ka.ptr, d_Kb.ptr,
              d_Sab.ptr, d_gc_ab.ptr, d_dc_ab.ptr, d_sq_norm_ab.ptr, d_sq_fk_ab.ptr,
              d_out_scalars.ptr, d_pair_f12.ptr, d_pair_ord.ptr, d_err.ptr);
          rc != 0) {
        return rc;
      }
    }
    d_fit_a_oqe.reset();
    d_raw_g_a_oqe.reset();
    d_gc_ab.reset();
    d_dc_ab.reset();
    d_sq_norm_ab.reset();
    d_sq_fk_ab.reset();
    d_Sab.reset();

    // Channel 3: BB
    if (ob_act >= 2) {
      DeviceBuf d_raw_g_b_oqe;
      if (!pack_active_qoe_checked_cuda_host(raw_g_b, aux, ob, fb, ob_act, ext, h_qoe)) return -2;
      if (d_raw_g_b_oqe.alloc(qoe_b) != cudaSuccess ||
          d_qoe_scratch.alloc(qoe_b) != cudaSuccess ||
          cudaMemcpy(d_qoe_scratch.ptr, h_qoe.data(), qoe_b * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return 3;
      }
      k_qoe_to_oqe<<<grid_for(qoe_b), 256>>>(aux, ob_act, ext, d_qoe_scratch.ptr,
                                             d_raw_g_b_oqe.ptr);
      d_qoe_scratch.reset();

      if (int rc = run_ss_or_rhf_stage2_gpu(
              h, false, include_ordinary, has_occ_weights, ob, fb, obs, ext, aux, ws_cap,
              n_aa + n_ab, 2, 5, d_occ_w_b.ptr, d_fit_b_oqe.ptr, d_raw_coul_b_oqe.ptr,
              d_raw_g_b_oqe.ptr, d_res_g_b_oqe.ptr, d_Fb.ptr, d_Kb.ptr, d_fg_b_d.ptr, d_fg_b_x.ptr,
              d_f2_b_d.ptr, d_f2_b_x.ptr, d_uf_b_d.ptr, d_uf_b_x.ptr, d_sq_fk_b_d.ptr,
              d_sq_fk_b_x.ptr, d_out_scalars.ptr, d_pair_f12.ptr, d_pair_ord.ptr, d_err.ptr);
          rc != 0) {
        return rc;
      }
    }
  }

  if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 7;
  int h_err = 0;
  if (cudaMemcpy(&h_err, d_err.ptr, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 8;
  if (h_err != 0) return h_err;

  if (out_scalars) {
    if (cudaMemcpy(out_scalars, d_out_scalars.ptr, 6 * sizeof(double), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      return 8;
    }
    if (!all_finite_host(out_scalars, 6)) return -5;
  }
  if (pair_f12_out) {
    if (cudaMemcpy(pair_f12_out, d_pair_f12.ptr, n_total * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return 8;
    }
    if (!all_finite_host(pair_f12_out, n_total)) return -5;
  }
  if (include_ordinary && pair_ord_out) {
    if (cudaMemcpy(pair_ord_out, d_pair_ord.ptr, n_total * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return 8;
    }
    if (!all_finite_host(pair_ord_out, n_total)) return -5;
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
