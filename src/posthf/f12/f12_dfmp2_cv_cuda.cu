// Dynamic-dimension AOT DF-MP2 Delta-CV host-reference (FP64) and CUDA cuBLAS ABI kernels.
// Covers all 4 compute sites in generativeqc._f12_focal_cv:
//   1. Q-chunked DF-HF J/K Fock build (fallback SCF)
//   2. Q-chunked MP2Fit MO factor forward & reverse (with metric bar)
//   3. Exact tiled cv-active vs frozen-core* DF-MP2 pair energy & Delta-CV cotangents
//      (closed-shell RHF spatial 2*t - t^T fast path + general UHF aa/ab/bb path)
//   4. OBS-only DF-HF MO Fock forward, Fock pullback, matrix-free CPHF matvec,
//      and nuclear source weights (eliminating all TensorIR JIT on Delta-CV E & E+F).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#ifndef GQC_SINK_HOST_REFERENCE
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace {

using Size = std::uint64_t;

constexpr int kErrOom = 4;
constexpr int kErrInternal = 9;

[[maybe_unused]] inline Size sat_mul(Size a, Size b) {
  if (a == 0 || b == 0) return 0;
  if (a > std::numeric_limits<Size>::max() / b) return std::numeric_limits<Size>::max();
  return a * b;
}

[[maybe_unused]] inline Size sat_add(Size a, Size b) {
  if (a > std::numeric_limits<Size>::max() - b) return std::numeric_limits<Size>::max();
  return a + b;
}

constexpr Size kDefaultChunkBudgetBytes = Size(1024) << 20;
constexpr Size kDefaultPairBudgetBytes = Size(2560) << 20;
constexpr Size kDefaultJkBudgetBytes = Size(2800) << 20;

[[maybe_unused]] inline Size effective_budget(Size budget_bytes, Size default_bytes) {
  return budget_bytes > 0 ? budget_bytes : default_bytes;
}

// All entry points keep two device int flags alongside their FP64 storage.
[[maybe_unused]] Size workspace_after_fixed(Size cap_bytes, Size fixed_doubles) {
  const Size fixed = sat_add(sat_mul(fixed_doubles, sizeof(double)), 2 * sizeof(int));
  return fixed < cap_bytes ? cap_bytes - fixed : 0;
}

[[maybe_unused]] Size pair_factor_doubles(int q, int o, int v, bool cotangents) {
  if (o == 0 || v == 0) return 0;
  const Size packed_and_raw = sat_mul(Size(2) * q, Size(o) * v);
  const Size bars = cotangents ? sat_add(Size(o) * o, Size(v) * v) : 0;
  return sat_add(packed_and_raw, sat_add(Size(o) + v, bars));
}

struct JkWorkspace {
  int chunk = 0;
  Size slab_bytes = 0;
  bool resident = false;
};

// A full DF cache can fail to allocate. Reserve enough for the streamed
// algorithm, including its whitening slab, regardless of cache eligibility.
[[maybe_unused]] JkWorkspace jk_mo_workspace(bool reverse, int q, int n, int max_occ,
                                             bool shared, int hint, Size budget_bytes) {
  const Size cap = effective_budget(budget_bytes, kDefaultJkBudgetBytes);
  const Size slab_cap = budget_bytes > 0 ? cap : kDefaultChunkBudgetBytes;
  const Size rev_chunk_cap = budget_bytes > 0 ? cap : (Size(768) << 20);
  const Size nn = Size(n) * n, qq = Size(q) * q, no = Size(n) * max_occ;
  const Size fixed = reverse ? sat_add(sat_mul(2, qq), Size(2) * q)
                             : sat_add(qq, Size(q));
  const Size matrices = reverse ? (shared ? 7 : 10) : (shared ? 4 : 6);
  const Size available = workspace_after_fixed(
      cap, sat_add(fixed, sat_mul(matrices, nn)));
  const Size slab_min = Size(2) * q * sizeof(double);
  if (available <= slab_min) return {};
  const Size df_bytes = sat_mul(sat_mul(q, nn), sizeof(double));
  const Size width = std::min(q, hint > 0 ? hint : q);
  const Size blas_width = std::numeric_limits<int>::max() / Size(n);
  JkWorkspace plan;
  if (reverse) {
    // Final congruence reuses bar_M and needs one additional metric matrix
    // after the spin scratch is released. Streamed phase B has four AO tiles.
    if (sat_mul(qq, sizeof(double)) > available) return {};
    const Size row_bytes = sat_mul(sat_add(sat_mul(4, nn), 1), sizeof(double));
    const Size rows = std::min({width, blas_width, (available - slab_min) / sizeof(double),
                                std::min(rev_chunk_cap, available) / row_bytes});
    plan.chunk = static_cast<int>(rows);
    const Size ones_bytes = rows * sizeof(double);
    plan.slab_bytes = std::min(slab_cap, available - ones_bytes);
    const Size phase_a_row = sat_add(nn, 2 * no);
    const Size a = sat_add(sat_mul(2, df_bytes),
                           sat_mul(phase_a_row, rows * sizeof(double)));
    const Size b = sat_add(df_bytes, sat_mul(3 * nn, rows * sizeof(double)));
    plan.resident = std::max(a, b) <= available - ones_bytes;
  } else {
    const Size row_bytes = sat_mul(sat_add(sat_mul(2, nn), no), sizeof(double));
    const Size rows = std::min({width, blas_width,
                                std::min(slab_cap, available - slab_min) / row_bytes});
    plan.chunk = static_cast<int>(rows);
    const Size remaining = available - rows * row_bytes;
    plan.slab_bytes = std::min(slab_cap, remaining);
    plan.resident = df_bytes <= remaining;
  }
  return plan;
}

[[maybe_unused]] inline bool all_finite(const double* ptr, Size count) {
  if (count == 0) return true;
  if (!ptr) return false;
  for (Size i = 0; i < count; ++i) {
    if (!std::isfinite(ptr[i])) return false;
  }
  return true;
}

#ifdef GQC_SINK_HOST_REFERENCE

// Row-major DGEMM: C = alpha * op(A) * op(B) + beta * C
// A has shape (M, K) if !trans_a else (K, M) with row stride lda.
// B has shape (K, N) if !trans_b else (N, K) with row stride ldb.
// C has shape (M, N) with row stride ldc.
void rm_gemm_host(bool trans_a, bool trans_b, int M, int N, int K, double alpha,
                  const double* A, int lda, const double* B, int ldb,
                  double beta, double* C, int ldc) {
  if (M <= 0 || N <= 0) return;
  if (beta == 0.0) {
    for (int i = 0; i < M; ++i) {
      double* c_row = C + Size(i) * ldc;
      std::fill(c_row, c_row + N, 0.0);
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
        const double a_ik = alpha * a_row[k];
        if (a_ik == 0.0) continue;
        const double* b_row = B + Size(k) * ldb;
        for (int j = 0; j < N; ++j) {
          c_row[j] += a_ik * b_row[j];
        }
      }
    }
  } else if (!trans_a && trans_b) {
    for (int i = 0; i < M; ++i) {
      const double* a_row = A + Size(i) * lda;
      double* c_row = C + Size(i) * ldc;
      for (int j = 0; j < N; ++j) {
        const double* b_row = B + Size(j) * ldb;
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
          acc += a_row[k] * b_row[k];
        }
        c_row[j] += alpha * acc;
      }
    }
  } else if (trans_a && !trans_b) {
    for (int k = 0; k < K; ++k) {
      const double* a_row = A + Size(k) * lda;
      const double* b_row = B + Size(k) * ldb;
      for (int i = 0; i < M; ++i) {
        const double a_ki = alpha * a_row[i];
        if (a_ki == 0.0) continue;
        double* c_row = C + Size(i) * ldc;
        for (int j = 0; j < N; ++j) {
          c_row[j] += a_ki * b_row[j];
        }
      }
    }
  } else {
    for (int i = 0; i < M; ++i) {
      double* c_row = C + Size(i) * ldc;
      for (int j = 0; j < N; ++j) {
        const double* b_row = B + Size(j) * ldb;
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
          acc += A[Size(k) * lda + i] * b_row[k];
        }
        c_row[j] += alpha * acc;
      }
    }
  }
}

void symmetrize_inplace(int n, double* mat) {
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const double avg = 0.5 * (mat[Size(i) * n + j] + mat[Size(j) * n + i]);
      mat[Size(i) * n + j] = avg;
      mat[Size(j) * n + i] = avg;
    }
  }
}

// Compute C_left^T @ M @ C_right into out (shape [nl, nr]) using tmp [nl, n].
void transform_ao_slice_host(int n, int nl, int nr, const double* C_left, int ld_l,
                             const double* M, const double* C_right, int ld_r,
                             double* tmp, double* out) {
  rm_gemm_host(true, false, nl, n, n, 1.0, C_left, ld_l, M, n, 0.0, tmp, n);
  rm_gemm_host(false, false, nl, nr, n, 1.0, tmp, n, C_right, ld_r, 0.0, out, nr);
}

// Accumulate one UHF spin channel (same==1: aa or bb; same==0: ab) for exact cv-active vs FC.
void accumulate_uhf_channel_host(
    bool same, bool need_cotangents, int q, int ol, int orr, int vl, int vr,
    int sh_l, int sh_r, int j_tile,
    const double* Bl, const double* Br,
    const double* eo_l, const double* eo_r,
    const double* ev_l, const double* ev_r,
    double& e_cv_out, double& e_fc_out,
    double* bar_Bl, double* bar_Br,
    double* bar_Fo_l, double* bar_Fo_r,
    double* bar_Fv_l, double* bar_Fv_r) {
  e_cv_out = 0.0;
  e_fc_out = 0.0;
  if (ol <= 0 || orr <= 0 || vl <= 0 || vr <= 0 || (same && ol < 2)) return;

  const Size vv = Size(vl) * vr;
  const int tile = std::max(1, std::min(orr, j_tile > 0 ? j_tile : orr));

  // Build tile intervals preserving pure-core [0, sh_r) vs pure-valence [sh_r, orr) tiles.
  struct TileSpec {
    int j0;
    int j1;
    bool is_core_j;
  };
  std::vector<TileSpec> tiles;
  for (int j0 = 0; j0 < sh_r; j0 += tile) {
    tiles.push_back({j0, std::min(sh_r, j0 + tile), true});
  }
  for (int j0 = sh_r; j0 < orr; j0 += tile) {
    tiles.push_back({j0, std::min(orr, j0 + tile), false});
  }

  // Reusable per-tile buffers
  std::vector<double> Bl_packed(Size(ol) * vl * q);
  for (int i = 0; i < ol; ++i) {
    for (int a = 0; a < vl; ++a) {
      double* dst = Bl_packed.data() + (Size(i) * vl + a) * q;
      for (int k = 0; k < q; ++k) {
        dst[k] = Bl[(Size(k) * ol + i) * vl + a];
      }
    }
  }
  std::vector<double> Br_tile(Size(tile) * vr * q);
  std::vector<double> g_buf(Size(ol) * vl * tile * vr);  // layout: [i, a, j_loc, b]
  std::vector<double> t_buf;
  std::vector<double> tm_buf;
  std::vector<double> fo_tmp;
  if (need_cotangents) {
    t_buf.resize(Size(ol) * tile * vv);
    tm_buf.resize(Size(ol) * tile * vv);
    fo_tmp.resize(Size(ol) * ol);
  }

  // For opposite-spin bar_Fo_r we also accumulate over i-tiles in the same pass if needed,
  // or compute directly per i-slice cleanly without extra pair recomputation.
  std::vector<double> t_i_slice;
  std::vector<double> fo_r_tmp;
  if (need_cotangents && !same && bar_Fo_r != nullptr) {
    t_i_slice.resize(Size(orr) * vv);
    fo_r_tmp.resize(Size(orr) * orr);
  }

  const double e_weight = same ? 0.25 : 1.0;
  const double b_weight = same ? 1.0 : 2.0;
  const double fv_weight = same ? 0.25 : 1.0;
  const double fo_weight = same ? -0.5 : -1.0;  // same gets 2 * (-0.25) = -0.5

  for (const auto& ts : tiles) {
    const int j0 = ts.j0;
    const int actual = ts.j1 - j0;
    // Pack Br[:, j0:j1, :] -> [actual * vr, q]
    for (int jl = 0; jl < actual; ++jl) {
      for (int b = 0; b < vr; ++b) {
        double* dst = Br_tile.data() + (Size(jl) * vr + b) * q;
        for (int k = 0; k < q; ++k) {
          dst[k] = Br[(Size(k) * orr + (j0 + jl)) * vr + b];
        }
      }
    }
    // g_buf[i*vl + a, jl*vr + b] = sum_k Bl_packed[i*vl + a, k] * Br_tile[jl*vr + b, k]
    rm_gemm_host(false, true, ol * vl, actual * vr, q, 1.0,
                 Bl_packed.data(), q, Br_tile.data(), q,
                 0.0, g_buf.data(), actual * vr);

    for (int i = 0; i < ol; ++i) {
      const double ei = eo_l[i];
      const bool i_is_core = i < sh_l;
      for (int jl = 0; jl < actual; ++jl) {
        const int j = j0 + jl;
        const double eij = ei + eo_r[j];
        const bool pair_in_delta = ts.is_core_j || i_is_core;
        double pe = 0.0;
        double* t_ij = need_cotangents ? (t_buf.data() + (Size(i) * actual + jl) * vv) : nullptr;
        double* tm_ij = need_cotangents ? (tm_buf.data() + (Size(i) * actual + jl) * vv) : nullptr;

        if (same) {
          for (int a = 0; a < vl; ++a) {
            const double* g_row_a = g_buf.data() + Size(i * vl + a) * (actual * vr) + Size(jl) * vr;
            const double eija = eij - ev_l[a];
            for (int b = 0; b < vr; ++b) {
              const double g_ab = g_row_a[b];
              const double g_ba = g_buf[Size(i * vl + b) * (actual * vr) + Size(jl) * vr + a];
              const double num = g_ab - g_ba;
              const double denom = eija - ev_r[b];
              const double t_val = num / denom;
              pe += t_val * num;
              if (need_cotangents) {
                const Size ab = Size(a) * vr + b;
                t_ij[ab] = t_val;
                tm_ij[ab] = pair_in_delta ? t_val : 0.0;
              }
            }
          }
        } else {
          for (int a = 0; a < vl; ++a) {
            const double* g_row_a = g_buf.data() + Size(i * vl + a) * (actual * vr) + Size(jl) * vr;
            const double eija = eij - ev_l[a];
            for (int b = 0; b < vr; ++b) {
              const double g_ab = g_row_a[b];
              const double denom = eija - ev_r[b];
              const double t_val = g_ab / denom;
              pe += t_val * g_ab;
              if (need_cotangents) {
                const Size ab = Size(a) * vr + b;
                t_ij[ab] = t_val;
                tm_ij[ab] = pair_in_delta ? t_val : 0.0;
              }
            }
          }
        }
        pe *= e_weight;
        e_cv_out += pe;
        if (!ts.is_core_j && !i_is_core) {
          e_fc_out += pe;
        }
      }
    }

    if (!need_cotangents) continue;

    // 1) bar_Bl[Q, i, a] += b_weight * sum_{jl, b} tm[i, jl, a, b] * Br[Q, j0+jl, b]
    //    bar_Br[Q, j0+jl, b] += b_weight * sum_{i, a} tm[i, jl, a, b] * Bl[Q, i, a]
    // Reuse g_buf as tm_mat with layout [i*vl + a, jl*vr + b] for two clean DGEMMs!
    for (int i = 0; i < ol; ++i) {
      for (int jl = 0; jl < actual; ++jl) {
        const double* tm_ij = tm_buf.data() + (Size(i) * actual + jl) * vv;
        for (int a = 0; a < vl; ++a) {
          double* dst = g_buf.data() + Size(i * vl + a) * (actual * vr) + Size(jl) * vr;
          std::memcpy(dst, tm_ij + Size(a) * vr, sizeof(double) * vr);
        }
      }
    }
    // bar_Bl_part[Q, i*vl + a] = b_weight * sum_{jb} Br_qjb[Q, jb] * tm_mat[ia, jb]
    // Br_tile is [jb, Q] -> Br_tile^T is [Q, jb]; tm_mat^T is [jb, ia].
    // Compute into temporary [ol*vl, q] then scatter-add to bar_Bl[q, ol, vl],
    // or directly: (Q, ia) += b_weight * (Br_tile^T [Q, jb]) @ (tm_mat^T [jb, ia])!
    rm_gemm_host(true, true, q, ol * vl, actual * vr, b_weight,
                 Br_tile.data(), q, g_buf.data(), actual * vr,
                 1.0, bar_Bl, ol * vl);

    // bar_Br_tile[Q, jb] += b_weight * (Bl_packed^T [Q, ia]) @ (tm_mat [ia, jb])
    // Since bar_Br has stride (orr * vr), accumulate into a compact [q, actual*vr] buffer or per-Q row!
    std::vector<double> bar_Br_chunk(Size(q) * actual * vr, 0.0);
    rm_gemm_host(true, false, q, actual * vr, ol * vl, b_weight,
                 Bl_packed.data(), q, g_buf.data(), actual * vr,
                 0.0, bar_Br_chunk.data(), actual * vr);
    for (int k = 0; k < q; ++k) {
      double* dst = bar_Br + (Size(k) * orr + j0) * vr;
      const double* src = bar_Br_chunk.data() + Size(k) * (actual * vr);
      for (int idx = 0; idx < actual * vr; ++idx) {
        dst[idx] += src[idx];
      }
    }

    // 2) bar_Fv_l[a, c] += fv_weight * sum_{i, jl, b} tm[i, jl, a, b] * tm[i, jl, c, b]
    //    bar_Fv_r[b, c] += fv_weight * sum_{i, jl, a} tm[i, jl, a, b] * tm[i, jl, a, c]
    for (int i = 0; i < ol; ++i) {
      for (int jl = 0; jl < actual; ++jl) {
        if (!ts.is_core_j && i >= sh_l) continue;
        const double* tm_ij = tm_buf.data() + (Size(i) * actual + jl) * vv;
        rm_gemm_host(false, true, vl, vl, vr, fv_weight, tm_ij, vr, tm_ij, vr, 1.0, bar_Fv_l, vl);
        rm_gemm_host(true, false, vr, vr, vl, fv_weight, tm_ij, vr, tm_ij, vr, 1.0, bar_Fv_r, vr);
      }
    }

    // 3) bar_Fo_l[i, k] += fo_weight * sum_{jl} sum_{a,b} t[i, jl, a, b] * t[k, jl, a, b]
    //    (with [sh_l:, sh_l:] zeroed when !ts.is_core_j!)
    rm_gemm_host(false, true, ol, ol, int(Size(actual) * vv), fo_weight,
                 t_buf.data(), int(Size(actual) * vv),
                 t_buf.data(), int(Size(actual) * vv),
                 0.0, fo_tmp.data(), ol);
    if (!ts.is_core_j) {
      for (int i = sh_l; i < ol; ++i) {
        for (int k = sh_l; k < ol; ++k) {
          fo_tmp[Size(i) * ol + k] = 0.0;
        }
      }
    }
    for (Size idx = 0; idx < Size(ol) * ol; ++idx) {
      bar_Fo_l[idx] += fo_tmp[idx];
    }
  }

  // 4) For opposite-spin UHF (same == false), also compute exact bar_Fo_r[j, l]
  //    using i-slices so we never materialize full [ol, orr, vl, vr].
  if (need_cotangents && !same && bar_Fo_r != nullptr) {
    std::vector<double> Br_full(Size(orr) * vr * q);
    for (int j = 0; j < orr; ++j) {
      for (int b = 0; b < vr; ++b) {
        double* dst = Br_full.data() + (Size(j) * vr + b) * q;
        for (int k = 0; k < q; ++k) {
          dst[k] = Br[(Size(k) * orr + j) * vr + b];
        }
      }
    }
    std::vector<double> g_i(Size(vl) * orr * vr);
    for (int i = 0; i < ol; ++i) {
      const bool is_core_i = i < sh_l;
      const double ei = eo_l[i];
      // g_i[a, j*vr + b] = Bl_packed[i*vl + a, :] @ Br_full[j*vr + b, :]^T
      rm_gemm_host(false, true, vl, orr * vr, q, 1.0,
                   Bl_packed.data() + Size(i) * vl * q, q,
                   Br_full.data(), q,
                   0.0, g_i.data(), orr * vr);
      for (int j = 0; j < orr; ++j) {
        const double eij = ei + eo_r[j];
        double* tj = t_i_slice.data() + Size(j) * vv;
        for (int a = 0; a < vl; ++a) {
          const double eija = eij - ev_l[a];
          const double* g_row = g_i.data() + Size(a) * (orr * vr) + Size(j) * vr;
          for (int b = 0; b < vr; ++b) {
            tj[Size(a) * vr + b] = g_row[b] / (eija - ev_r[b]);
          }
        }
      }
      rm_gemm_host(false, true, orr, orr, int(vv), -1.0,
                   t_i_slice.data(), int(vv), t_i_slice.data(), int(vv),
                   0.0, fo_r_tmp.data(), orr);
      if (!is_core_i) {
        for (int j = sh_r; j < orr; ++j) {
          for (int l = sh_r; l < orr; ++l) {
            fo_r_tmp[Size(j) * orr + l] = 0.0;
          }
        }
      }
      for (Size idx = 0; idx < Size(orr) * orr; ++idx) {
        bar_Fo_r[idx] += fo_r_tmp[idx];
      }
    }
  }
}

// Closed-shell RHF spatial fast path: merges aa + ab + bb into ONE spatial g_ab = B_i^T @ B_j
// pass using u = 2*t - t^T. Mathematically identical to general UHF with alpha == beta.
void accumulate_rhf_spatial_host(
    bool need_cotangents, int q, int o, int v, int sh, int j_tile,
    const double* B, const double* eo, const double* ev,
    double* cv_spin_e, double* fc_spin_e,
    double* bar_B, double* bar_Fo, double* bar_Fv) {
  cv_spin_e[0] = cv_spin_e[1] = cv_spin_e[2] = 0.0;
  fc_spin_e[0] = fc_spin_e[1] = fc_spin_e[2] = 0.0;
  if (o <= 0 || v <= 0) return;

  const Size vv = Size(v) * v;
  const int tile = std::max(1, std::min(o, j_tile > 0 ? j_tile : o));

  struct TileSpec {
    int j0;
    int j1;
    bool is_core_j;
  };
  std::vector<TileSpec> tiles;
  for (int j0 = 0; j0 < sh; j0 += tile) {
    tiles.push_back({j0, std::min(sh, j0 + tile), true});
  }
  for (int j0 = sh; j0 < o; j0 += tile) {
    tiles.push_back({j0, std::min(o, j0 + tile), false});
  }

  std::vector<double> B_packed(Size(o) * v * q);
  for (int i = 0; i < o; ++i) {
    for (int a = 0; a < v; ++a) {
      double* dst = B_packed.data() + (Size(i) * v + a) * q;
      for (int k = 0; k < q; ++k) {
        dst[k] = B[(Size(k) * o + i) * v + a];
      }
    }
  }
  std::vector<double> g_buf(Size(o) * v * tile * v);
  std::vector<double> t_buf;
  std::vector<double> u_buf;
  std::vector<double> um_mat;
  std::vector<double> fo_tmp;
  if (need_cotangents) {
    t_buf.resize(Size(o) * tile * vv);
    u_buf.resize(Size(o) * tile * vv);
    um_mat.resize(Size(o) * v * tile * v);
    fo_tmp.resize(Size(o) * o);
  }

  double e_ss_cv = 0.0, e_os_cv = 0.0;
  double e_ss_fc = 0.0, e_os_fc = 0.0;

  for (const auto& ts : tiles) {
    const int j0 = ts.j0;
    const int actual = ts.j1 - j0;
    const double* B_j_packed = B_packed.data() + Size(j0) * v * q;

    // g_buf[i*v + a, jl*v + b] = B_packed[i*v + a, :] @ B_j_packed[jl*v + b, :]^T
    rm_gemm_host(false, true, o * v, actual * v, q, 1.0,
                 B_packed.data(), q, B_j_packed, q,
                 0.0, g_buf.data(), actual * v);

    for (int i = 0; i < o; ++i) {
      const double ei = eo[i];
      const bool i_is_core = i < sh;
      for (int jl = 0; jl < actual; ++jl) {
        const int j = j0 + jl;
        const double eij = ei + eo[j];
        const bool pair_in_delta = ts.is_core_j || i_is_core;
        double os_ij = 0.0;
        double ss_ij = 0.0;
        double* t_ij = need_cotangents ? (t_buf.data() + (Size(i) * actual + jl) * vv) : nullptr;
        double* u_ij = need_cotangents ? (u_buf.data() + (Size(i) * actual + jl) * vv) : nullptr;

        for (int a = 0; a < v; ++a) {
          const double* g_row_a = g_buf.data() + Size(i * v + a) * (actual * v) + Size(jl) * v;
          const double eija = eij - ev[a];
          double* um_dst = need_cotangents
              ? (um_mat.data() + Size(i * v + a) * (actual * v) + Size(jl) * v)
              : nullptr;
          for (int b = 0; b < v; ++b) {
            const double g_ab = g_row_a[b];
            const double g_ba = g_buf[Size(i * v + b) * (actual * v) + Size(jl) * v + a];
            const double denom = eija - ev[b];
            const double t_ab = g_ab / denom;
            const double t_ba = g_ba / denom;
            const double u_ab = 2.0 * t_ab - t_ba;
            os_ij += t_ab * g_ab;
            ss_ij += 0.5 * (t_ab - t_ba) * g_ab;
            if (need_cotangents) {
              const Size ab = Size(a) * v + b;
              t_ij[ab] = t_ab;
              u_ij[ab] = u_ab;
              um_dst[b] = pair_in_delta ? u_ab : 0.0;
            }
          }
        }
        e_os_cv += os_ij;
        e_ss_cv += ss_ij;
        if (!ts.is_core_j && !i_is_core) {
          e_os_fc += os_ij;
          e_ss_fc += ss_ij;
        }
      }
    }

    if (!need_cotangents) continue;

    // 1) bar_B[Q, i, a] += 4.0 * sum_{jl, b} u_m[i, jl, a, b] * B[Q, j0+jl, b]
    rm_gemm_host(true, true, q, o * v, actual * v, 4.0,
                 B_j_packed, q, um_mat.data(), actual * v,
                 1.0, bar_B, o * v);

    // 2) bar_Fv[a, c] += sum_{i, jl in delta, b} t[i, jl, a, b] * u[i, jl, c, b]
    for (int i = 0; i < o; ++i) {
      for (int jl = 0; jl < actual; ++jl) {
        if (!ts.is_core_j && i >= sh) continue;
        const double* t_ij = t_buf.data() + (Size(i) * actual + jl) * vv;
        const double* u_ij = u_buf.data() + (Size(i) * actual + jl) * vv;
        rm_gemm_host(false, true, v, v, v, 1.0, t_ij, v, u_ij, v, 1.0, bar_Fv, v);
      }
    }

    // 3) bar_Fo[i, k] += -sum_{jl} sum_{a,b} t[i, jl, a, b] * u[k, jl, a, b]
    //    with [sh:, sh:] zeroed for valence j-tiles!
    rm_gemm_host(false, true, o, o, int(Size(actual) * vv), -1.0,
                 t_buf.data(), int(Size(actual) * vv),
                 u_buf.data(), int(Size(actual) * vv),
                 0.0, fo_tmp.data(), o);
    if (!ts.is_core_j) {
      for (int i = sh; i < o; ++i) {
        for (int k = sh; k < o; ++k) {
          fo_tmp[Size(i) * o + k] = 0.0;
        }
      }
    }
    for (Size idx = 0; idx < Size(o) * o; ++idx) {
      bar_Fo[idx] += fo_tmp[idx];
    }
  }

  cv_spin_e[0] = e_ss_cv;
  cv_spin_e[1] = e_os_cv;
  cv_spin_e[2] = e_ss_cv;
  fc_spin_e[0] = e_ss_fc;
  fc_spin_e[1] = e_os_fc;
  fc_spin_e[2] = e_ss_fc;
}

#else  // !GQC_SINK_HOST_REFERENCE

struct CudaStream {
  cudaStream_t stream = nullptr;
  CudaStream() = default;
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  ~CudaStream() {
    if (stream) cudaStreamDestroy(stream);
  }
  cudaError_t init() { return cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); }
};

struct CublasHandle {
  cublasHandle_t handle = nullptr;
  CublasHandle() = default;
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
  ~CublasHandle() {
    if (handle) cublasDestroy(handle);
  }
  cublasStatus_t init(cudaStream_t stream) {
    cublasStatus_t s = cublasCreate(&handle);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = cublasSetStream(handle, stream);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    return cublasSetWorkspace(handle, nullptr, 0);
  }
};

template <typename T>
struct DeviceBuf {
  T* ptr = nullptr;
  Size cap = 0;
  DeviceBuf() = default;
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
  ~DeviceBuf() { reset(); }
  void reset() {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
      cap = 0;
    }
  }
  cudaError_t alloc(Size count) {
    reset();
    if (count == 0) return cudaSuccess;
    cudaError_t e = cudaMalloc(&ptr, count * sizeof(T));
    if (e == cudaSuccess) cap = count;
    return e;
  }
  cudaError_t zero_async(Size count, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    return cudaMemsetAsync(ptr, 0, count * sizeof(T), stream);
  }
};

using DoubleBuf = DeviceBuf<double>;
using IntBuf = DeviceBuf<int>;

inline bool fits_int(Size x) {
  return x <= static_cast<Size>(std::numeric_limits<int>::max());
}

inline unsigned grid_for_cap(Size n, unsigned blk = 256) {
  if (n == 0) return 0;
  const Size blocks = (n + blk - 1) / blk;
  return static_cast<unsigned>(std::min<Size>(blocks, 65535ULL));
}

inline bool launch_ok() { return cudaGetLastError() == cudaSuccess; }

template <typename T>
inline bool h2d_async(T* dst, const T* src, Size count, cudaStream_t stream) {
  if (count == 0) return true;
  return cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

template <typename T>
inline bool d2h_async(T* dst, const T* src, Size count, cudaStream_t stream) {
  if (count == 0) return true;
  return cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

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

__global__ void k_check_finite(const double* ptr, Size n, int* bad_flag) {
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += Size(gridDim.x) * blockDim.x) {
    if (!isfinite(ptr[idx])) {
      atomicOr(bad_flag, 1);
    }
  }
}

inline bool check_finite_gpu(const double* ptr, Size n, int* d_bad, cudaStream_t stream) {
  if (n == 0) return true;
  k_check_finite<<<grid_for_cap(n), 256, 0, stream>>>(ptr, n, d_bad);
  return launch_ok();
}

__global__ void k_transpose_2d(int rows, int cols, const double* src, double* dst) {
  __shared__ double tile[32][33];
  for (int by = blockIdx.y; by * 32 < rows; by += gridDim.y) {
    for (int bx = blockIdx.x; bx * 32 < cols; bx += gridDim.x) {
      const int x = bx * 32 + threadIdx.x;
      const int y = by * 32 + threadIdx.y;
      if (x < cols) {
        for (int dy = 0; dy < 32; dy += 8) {
          if (y + dy < rows) {
            tile[threadIdx.y + dy][threadIdx.x] = src[Size(y + dy) * cols + x];
          }
        }
      }
      __syncthreads();
      const int tx = by * 32 + threadIdx.x;
      const int ty = bx * 32 + threadIdx.y;
      if (tx < rows) {
        for (int dy = 0; dy < 32; dy += 8) {
          if (ty + dy < cols) {
            dst[Size(ty + dy) * rows + tx] = tile[threadIdx.x][threadIdx.y + dy];
          }
        }
      }
      __syncthreads();
    }
  }
}

inline bool transpose_2d_gpu(int rows, int cols, const double* src, double* dst,
                             cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return true;
  const dim3 block(32, 8);
  const dim3 grid(std::min<unsigned>((cols + 31) / 32, 65535U),
                  std::min<unsigned>((rows + 31) / 32, 65535U));
  k_transpose_2d<<<grid, block, 0, stream>>>(rows, cols, src, dst);
  return launch_ok();
}

__global__ void k_symmetrize_batched(int batch, int n, double* mat) {
  const Size nn = Size(n) * n;
  const Size total = Size(batch) * nn;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const Size b = idx / nn;
    const Size rem = idx - b * nn;
    const int i = static_cast<int>(rem / n);
    const int j = static_cast<int>(rem - Size(i) * n);
    if (i < j) {
      const Size p_ij = b * nn + Size(i) * n + j;
      const Size p_ji = b * nn + Size(j) * n + i;
      const double avg = 0.5 * (mat[p_ij] + mat[p_ji]);
      mat[p_ij] = avg;
      mat[p_ji] = avg;
    }
  }
}

inline bool symmetrize_batched_gpu(int batch, int n, double* mat, cudaStream_t stream) {
  if (batch <= 0 || n <= 1) return true;
  const Size total = Size(batch) * Size(n) * n;
  k_symmetrize_batched<<<grid_for_cap(total), 256, 0, stream>>>(batch, n, mat);
  return launch_ok();
}

__global__ void k_make_sym_sum(int n, const double* S, double* S_sym) {
  const Size total = Size(n) * n;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const int i = static_cast<int>(idx / n);
    const int j = static_cast<int>(idx - Size(i) * n);
    S_sym[idx] = S[idx] + S[Size(j) * n + i];
  }
}

__global__ void k_trace_occ_accum(int batch, int n, int occ, double scale, const double* df_ch,
                                  double* charge_out) {
  const Size nn = Size(n) * n;
  for (int al = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); al < batch;
       al += static_cast<int>(gridDim.x * blockDim.x)) {
    const double* sl = df_ch + Size(al) * nn;
    double tr = 0.0;
    for (int i = 0; i < occ; ++i) tr += sl[Size(i) * n + i];
    charge_out[al] += scale * tr;
  }
}

__global__ void k_pack_leading_rows(int batch, int rows, int n, const double* src_bnn,
                                    double* dst_brn) {
  const Size nn = Size(n) * n;
  const Size rn = Size(rows) * n;
  const Size total = Size(batch) * rn;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const Size b = idx / rn;
    const Size rem = idx - b * rn;
    dst_brn[idx] = src_bnn[b * nn + rem];
  }
}

__global__ void k_sub_leading_rows(int batch, int rows, int n, const double* src_brn,
                                   double* dst_bnn) {
  const Size nn = Size(n) * n;
  const Size rn = Size(rows) * n;
  const Size total = Size(batch) * rn;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const Size b = idx / rn;
    const Size rem = idx - b * rn;
    dst_bnn[b * nn + rem] -= src_brn[idx];
  }
}

__global__ void k_extract_subblock(int batch, int sub_r, int sub_c, int r0, int c0, int n,
                                   const double* src_bnn, double* dst_brc) {
  const Size nn = Size(n) * n;
  const Size rc = Size(sub_r) * sub_c;
  const Size total = Size(batch) * rc;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const Size b = idx / rc;
    const Size rem = idx - b * rc;
    const int r = static_cast<int>(rem / sub_c);
    const int c = static_cast<int>(rem - Size(r) * sub_c);
    dst_brc[idx] = src_bnn[b * nn + Size(r0 + r) * n + (c0 + c)];
  }
}

__global__ void k_add_subblock_single(int sub_r, int sub_c, int r0, int c0, int ld_dst,
                                      double scale, const double* src_rc, double* dst_nn) {
  const Size total = Size(sub_r) * sub_c;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const int r = static_cast<int>(idx / sub_c);
    const int c = static_cast<int>(idx - Size(r) * sub_c);
    dst_nn[Size(r0 + r) * ld_dst + (c0 + c)] += scale * src_rc[idx];
  }
}

__global__ void k_init_bar_df_chunk(int batch, int n, int occ, const double* ch, const double* bch,
                                    const double* mat, double* dst_bnn) {
  const Size nn = Size(n) * n;
  const Size total = Size(batch) * nn;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const Size b = idx / nn;
    const Size rem = idx - b * nn;
    const int p = static_cast<int>(rem / n);
    const int r = static_cast<int>(rem - Size(p) * n);
    double val = ch[b] * mat[rem];
    if (p == r && p < occ) val += bch[b];
    dst_bnn[idx] = val;
  }
}

__global__ void k_mul_elemwise(Size n, const double* a, const double* b, double* out) {
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += Size(gridDim.x) * blockDim.x) {
    out[idx] = a[idx] * b[idx];
  }
}

__global__ void k_cphf_jcoeff(int batch, int restricted, const double* ch_a, const double* ch_b,
                              double* jcoeff) {
  for (int al = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); al < batch;
       al += static_cast<int>(gridDim.x * blockDim.x)) {
    const double ca = ch_a ? ch_a[al] : 0.0;
    const double cb = ch_b ? ch_b[al] : 0.0;
    jcoeff[al] = restricted ? (4.0 * ca) : (2.0 * (ca + cb));
  }
}

// --- MP2 Pair Kernels ---

__global__ void k_rhf_mp2_energy_block(int ni, int nj, int v, double scale, int is_fc_block,
                                       const double* g, const double* eo_i, const double* eo_j,
                                       const double* ev, double* e_acc) {
  __shared__ double s_ss_cv[256];
  __shared__ double s_os_cv[256];
  __shared__ double s_ss_fc[256];
  __shared__ double s_os_fc[256];

  const Size njv = Size(nj) * v;
  const Size total = Size(ni) * nj * Size(v) * v;
  double loc_ss_cv = 0.0;
  double loc_os_cv = 0.0;
  double loc_ss_fc = 0.0;
  double loc_os_fc = 0.0;

  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const int b = static_cast<int>(idx % v);
    const Size t1 = idx / v;
    const int a = static_cast<int>(t1 % v);
    const Size t2 = t1 / v;
    const int jl = static_cast<int>(t2 % nj);
    const int il = static_cast<int>(t2 / nj);

    const double g_ab = g[Size(il * v + a) * njv + Size(jl) * v + b];
    const double g_ba = g[Size(il * v + b) * njv + Size(jl) * v + a];
    const double denom = eo_i[il] + eo_j[jl] - ev[a] - ev[b];
    const double t_ab = g_ab / denom;
    const double t_ba = g_ba / denom;
    const double os_val = scale * t_ab * g_ab;
    const double ss_val = scale * 0.5 * (t_ab - t_ba) * g_ab;
    loc_os_cv += os_val;
    loc_ss_cv += ss_val;
    if (is_fc_block) {
      loc_os_fc += os_val;
      loc_ss_fc += ss_val;
    }
  }

  s_ss_cv[threadIdx.x] = loc_ss_cv;
  s_os_cv[threadIdx.x] = loc_os_cv;
  s_ss_fc[threadIdx.x] = loc_ss_fc;
  s_os_fc[threadIdx.x] = loc_os_fc;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_ss_cv[threadIdx.x] += s_ss_cv[threadIdx.x + s];
      s_os_cv[threadIdx.x] += s_os_cv[threadIdx.x + s];
      s_ss_fc[threadIdx.x] += s_ss_fc[threadIdx.x + s];
      s_os_fc[threadIdx.x] += s_os_fc[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(&e_acc[0], s_ss_cv[0]);
    atomicAdd(&e_acc[1], s_os_cv[0]);
    atomicAdd(&e_acc[2], s_ss_fc[0]);
    atomicAdd(&e_acc[3], s_os_fc[0]);
  }
}

__global__ void k_rhf_mp2_tile_cot(int o, int actual, int v, int j0, int sh, int is_core_j,
                                   const double* g, const double* eo, const double* ev,
                                   double* e_acc, double* t_out, double* u_out) {
  __shared__ double s_ss_cv[256];
  __shared__ double s_os_cv[256];
  __shared__ double s_ss_fc[256];
  __shared__ double s_os_fc[256];

  const Size av = Size(actual) * v;
  const Size total = Size(o) * actual * Size(v) * v;
  double loc_ss_cv = 0.0;
  double loc_os_cv = 0.0;
  double loc_ss_fc = 0.0;
  double loc_os_fc = 0.0;

  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const int b = static_cast<int>(idx % v);
    const Size t1 = idx / v;
    const int a = static_cast<int>(t1 % v);
    const Size t2 = t1 / v;
    const int jl = static_cast<int>(t2 % actual);
    const int i = static_cast<int>(t2 / actual);

    const double g_ab = g[Size(i * v + a) * av + Size(jl) * v + b];
    const double g_ba = g[Size(i * v + b) * av + Size(jl) * v + a];
    const double denom = eo[i] + eo[j0 + jl] - ev[a] - ev[b];
    const double t_ab = g_ab / denom;
    const double t_ba = g_ba / denom;
    const double u_ab = 2.0 * t_ab - t_ba;
    const double os_val = t_ab * g_ab;
    const double ss_val = 0.5 * (t_ab - t_ba) * g_ab;

    loc_os_cv += os_val;
    loc_ss_cv += ss_val;
    if (!is_core_j && i >= sh) {
      loc_os_fc += os_val;
      loc_ss_fc += ss_val;
    }
    t_out[idx] = t_ab;
    u_out[idx] = u_ab;
  }

  s_ss_cv[threadIdx.x] = loc_ss_cv;
  s_os_cv[threadIdx.x] = loc_os_cv;
  s_ss_fc[threadIdx.x] = loc_ss_fc;
  s_os_fc[threadIdx.x] = loc_os_fc;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_ss_cv[threadIdx.x] += s_ss_cv[threadIdx.x + s];
      s_os_cv[threadIdx.x] += s_os_cv[threadIdx.x + s];
      s_ss_fc[threadIdx.x] += s_ss_fc[threadIdx.x + s];
      s_os_fc[threadIdx.x] += s_os_fc[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(&e_acc[0], s_ss_cv[0]);
    atomicAdd(&e_acc[1], s_os_cv[0]);
    atomicAdd(&e_acc[2], s_ss_fc[0]);
    atomicAdd(&e_acc[3], s_os_fc[0]);
  }
}

template <bool SAME, bool NEED_COT>
__global__ void k_uhf_mp2_tile(int ol, int actual, int vl, int vr, int j0, int sh_l, int is_core_j,
                               const double* g, const double* eo_l, const double* eo_r,
                               const double* ev_l, const double* ev_r, double* e_acc,
                               double* t_out) {
  __shared__ double s_cv[256];
  __shared__ double s_fc[256];

  const Size avr = Size(actual) * vr;
  const Size total = Size(ol) * actual * Size(vl) * vr;
  double loc_cv = 0.0;
  double loc_fc = 0.0;

  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const int b = static_cast<int>(idx % vr);
    const Size t1 = idx / vr;
    const int a = static_cast<int>(t1 % vl);
    const Size t2 = t1 / vl;
    const int jl = static_cast<int>(t2 % actual);
    const int i = static_cast<int>(t2 / actual);

    const double g_ab = g[Size(i * vl + a) * avr + Size(jl) * vr + b];
    const double denom = eo_l[i] + eo_r[j0 + jl] - ev_l[a] - ev_r[b];
    double t_val = 0.0;
    double pe = 0.0;
    if (SAME) {
      const double g_ba = g[Size(i * vl + b) * avr + Size(jl) * vr + a];
      const double num = g_ab - g_ba;
      t_val = num / denom;
      pe = 0.25 * t_val * num;
    } else {
      t_val = g_ab / denom;
      pe = t_val * g_ab;
    }
    loc_cv += pe;
    if (!is_core_j && i >= sh_l) {
      loc_fc += pe;
    }
    if (NEED_COT) {
      t_out[idx] = t_val;
    }
  }

  s_cv[threadIdx.x] = loc_cv;
  s_fc[threadIdx.x] = loc_fc;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      s_cv[threadIdx.x] += s_cv[threadIdx.x + s];
      s_fc[threadIdx.x] += s_fc[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(&e_acc[0], s_cv[0]);
    atomicAdd(&e_acc[1], s_fc[0]);
  }
}

__global__ void k_uhf_ab_for_tile(int orr, int actual_i, int vr, int vl, int i0, const double* g,
                                  const double* eo_l, const double* eo_r, const double* ev_l,
                                  const double* ev_r, double* t_out) {
  const Size aivl = Size(actual_i) * vl;
  const Size total = Size(orr) * actual_i * Size(vr) * vl;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const int a = static_cast<int>(idx % vl);
    const Size t1 = idx / vl;
    const int b = static_cast<int>(t1 % vr);
    const Size t2 = t1 / vr;
    const int il = static_cast<int>(t2 % actual_i);
    const int j = static_cast<int>(t2 / actual_i);

    const double g_val = g[Size(j * vr + b) * aivl + Size(il) * vl + a];
    const double denom = eo_l[i0 + il] + eo_r[j] - ev_l[a] - ev_r[b];
    t_out[idx] = g_val / denom;
  }
}

// Repack [ni, nj, vl, vr] -> [ni * vl, nj * vr]
__global__ void k_pack_t_to_mat(int ni, int nj, int vl, int vr, const double* t_src,
                                double* mat_dst) {
  const Size njvr = Size(nj) * vr;
  const Size total = Size(ni) * nj * Size(vl) * vr;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const int b = static_cast<int>(idx % vr);
    const Size t1 = idx / vr;
    const int a = static_cast<int>(t1 % vl);
    const Size t2 = t1 / vl;
    const int jl = static_cast<int>(t2 % nj);
    const int i = static_cast<int>(t2 / nj);
    mat_dst[Size(i * vl + a) * njvr + Size(jl) * vr + b] = t_src[idx];
  }
}

// Repack [np, vl, vr] -> [vl, np, vr]
__global__ void k_pack_pab_to_apb(int np, int vl, int vr, const double* src, double* dst) {
  const Size total = Size(np) * vl * vr;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const int b = static_cast<int>(idx % vr);
    const Size t1 = idx / vr;
    const int a = static_cast<int>(t1 % vl);
    const int p = static_cast<int>(t1 / vl);
    dst[(Size(a) * np + p) * vr + b] = src[idx];
  }
}

// In-place GPU whitener transform on host matrix [q, cols]: host_mat = op(W) @ host_mat
// Bounded to <= 1 GiB device staging via 2D column slabs (zero extra host RAM).
bool whiten_host_inplace_slabbed_gpu(cublasHandle_t h, cudaStream_t stream, bool trans_w, int q,
                                     Size cols, const double* d_W, double* host_mat,
                                     int* d_out_bad, Size slab_budget) {
  if (q <= 0 || cols == 0) return true;
  const Size bytes_per_col = sat_mul(sat_mul(Size(2), Size(q)), sizeof(double));
  if (bytes_per_col > slab_budget) return false;
  Size slab_w = slab_budget / std::max<Size>(1, bytes_per_col);
  slab_w = std::min<Size>(slab_w, cols);
  slab_w = std::min<Size>(slab_w, static_cast<Size>(std::numeric_limits<int>::max()));

  DoubleBuf d_in, d_out;
  if (d_in.alloc(Size(q) * slab_w) != cudaSuccess ||
      d_out.alloc(Size(q) * slab_w) != cudaSuccess) {
    return false;
  }
  const size_t host_pitch = static_cast<size_t>(cols) * sizeof(double);
  for (Size c0 = 0; c0 < cols; c0 += slab_w) {
    const Size w_c = std::min<Size>(slab_w, cols - c0);
    const size_t width_bytes = static_cast<size_t>(w_c) * sizeof(double);
    if (cudaMemcpy2DAsync(d_in.ptr, width_bytes, host_mat + c0, host_pitch, width_bytes,
                          static_cast<size_t>(q), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      return false;
    }
    if (rm_gemm(h, trans_w, false, q, static_cast<int>(w_c), q, 1.0, d_W, q, d_in.ptr,
                static_cast<int>(w_c), 0.0, d_out.ptr, static_cast<int>(w_c)) !=
        CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    if (!check_finite_gpu(d_out.ptr, Size(q) * w_c, d_out_bad, stream)) return false;
    if (cudaMemcpy2DAsync(host_mat + c0, host_pitch, d_out.ptr, width_bytes, width_bytes,
                          static_cast<size_t>(q), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
      return false;
    }
  }
  return cudaStreamSynchronize(stream) == cudaSuccess;
}

bool check_uhf_pair_workspace_budget(Size scratch_bytes, bool same, bool need_cotangents, int ol, int orr, int vl,
                                     int vr, bool need_bar_fo_r) {
  if (ol <= 0 || orr <= 0 || vl <= 0 || vr <= 0 || (same && ol < 2)) return true;
  const Size vv = sat_mul(Size(vl), Size(vr));
  const Size n_bufs = need_cotangents ? Size(2) : Size(1);
  const Size bytes_per_j = sat_mul(sat_mul(n_bufs, Size(ol)), sat_mul(vv, sizeof(double)));
  if (bytes_per_j > scratch_bytes) return false;
  if (need_cotangents && !same && need_bar_fo_r) {
    const Size bytes_per_i = sat_mul(sat_mul(Size(2), Size(orr)), sat_mul(vv, sizeof(double)));
    if (bytes_per_i > scratch_bytes) return false;
  }
  return true;
}

bool check_rhf_pair_workspace_budget(Size scratch_bytes, bool need_cotangents, int o, int v) {
  if (o <= 0 || v <= 0) return true;
  const Size vv = sat_mul(Size(v), Size(v));
  if (!need_cotangents) {
    return sat_mul(vv, sizeof(double)) <= scratch_bytes;
  }
  const Size bytes_per_j = sat_mul(sat_mul(Size(3), Size(o)), sat_mul(vv, sizeof(double)));
  return bytes_per_j <= scratch_bytes;
}

// GPU UHF single-spin-channel MP2 accumulator (same==1: aa/bb; same==0: ab).
bool accumulate_uhf_channel_gpu(
    cublasHandle_t h, cudaStream_t stream,
    bool same, bool need_cotangents, int q, int ol, int orr, int vl, int vr,
    int sh_l, int sh_r, int tile_limit, Size scratch_bytes,
    const double* d_Bl_packed, const double* d_Br_packed,
    const double* d_eo_l, const double* d_eo_r,
    const double* d_ev_l, const double* d_ev_r,
    double* d_e_acc,
    double* d_bar_Bl, double* d_bar_Br,
    double* d_bar_Fo_l, double* d_bar_Fo_r,
    double* d_bar_Fv_l, double* d_bar_Fv_r,
    int* err_code_out = nullptr) {
  if (ol <= 0 || orr <= 0 || vl <= 0 || vr <= 0 || (same && ol < 2)) return true;

  const Size vv = sat_mul(Size(vl), Size(vr));
  const Size n_bufs = need_cotangents ? Size(2) : Size(1);
  const Size bytes_per_j = sat_mul(sat_mul(n_bufs, Size(ol)), sat_mul(vv, sizeof(double)));
  if (bytes_per_j > scratch_bytes) {
    if (err_code_out) *err_code_out = kErrOom;
    return false;
  }
  Size max_tile = scratch_bytes / std::max<Size>(1, bytes_per_j);
  max_tile = std::min<Size>(max_tile, static_cast<Size>(orr));
  if (tile_limit > 0) max_tile = std::min<Size>(max_tile, static_cast<Size>(tile_limit));
  // Ensure cuBLAS int dimensions stay strictly within INT_MAX
  while (max_tile > 1 &&
         (!fits_int(max_tile * vv) || !fits_int(Size(ol) * max_tile * std::max(vl, vr)))) {
    max_tile /= 2;
  }
  if (!fits_int(max_tile * vv) || !fits_int(Size(ol) * vl) || !fits_int(max_tile * vr) ||
      !fits_int(Size(ol) * max_tile * std::max(vl, vr))) {
    return false;
  }
  const int j_tile = static_cast<int>(max_tile);

  // Also size i_tile for opposite-spin bar_Fo_r so we allocate d_g and d_t ONCE.
  int i_tile = 0;
  Size buf_elems = Size(ol) * j_tile * vv;
  if (need_cotangents && !same && d_bar_Fo_r != nullptr) {
    const Size bytes_per_i = sat_mul(sat_mul(Size(2), Size(orr)), sat_mul(vv, sizeof(double)));
    if (bytes_per_i > scratch_bytes) {
      if (err_code_out) *err_code_out = kErrOom;
      return false;
    }
    Size max_it = scratch_bytes / std::max<Size>(1, bytes_per_i);
    max_it = std::min<Size>(max_it, static_cast<Size>(ol));
    if (tile_limit > 0) max_it = std::min<Size>(max_it, static_cast<Size>(tile_limit));
    while (max_it > 1 &&
           (!fits_int(max_it * vv) || !fits_int(Size(orr) * vr) || !fits_int(max_it * vl))) {
      max_it /= 2;
    }
    if (!fits_int(max_it * vv) || !fits_int(Size(orr) * vr) || !fits_int(max_it * vl)) {
      return false;
    }
    i_tile = static_cast<int>(max_it);
    buf_elems = std::max<Size>(buf_elems, Size(orr) * i_tile * vv);
  }

  DoubleBuf d_g, d_t;
  if (d_g.alloc(buf_elems) != cudaSuccess ||
      (need_cotangents && d_t.alloc(buf_elems) != cudaSuccess)) {
    if (err_code_out) *err_code_out = kErrOom;
    return false;
  }

  struct TileSpec {
    int start;
    int stop;
    bool is_core;
  };
  std::vector<TileSpec> j_tiles;
  for (int j0 = 0; j0 < sh_r; j0 += j_tile) {
    j_tiles.push_back({j0, std::min(sh_r, j0 + j_tile), true});
  }
  for (int j0 = sh_r; j0 < orr; j0 += j_tile) {
    j_tiles.push_back({j0, std::min(orr, j0 + j_tile), false});
  }

  const double b_weight = same ? 1.0 : 2.0;
  const double fv_weight = same ? 0.25 : 1.0;
  const double fo_weight = same ? -0.5 : -1.0;

  for (const auto& ts : j_tiles) {
    const int j0 = ts.start;
    const int actual = ts.stop - j0;
    const double* d_Br_j = d_Br_packed + Size(j0) * vr * q;

    // G[i*vl + a, jl*vr + b] = Bl_packed[i*vl + a, :] @ Br_j[jl*vr + b, :]^T
    if (rm_gemm(h, false, true, ol * vl, actual * vr, q, 1.0,
                d_Bl_packed, q, d_Br_j, q,
                0.0, d_g.ptr, actual * vr) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }

    const Size tile_elems = Size(ol) * actual * vv;
    const unsigned grid = grid_for_cap(tile_elems);
    if (same) {
      if (need_cotangents) {
        k_uhf_mp2_tile<true, true><<<grid, 256, 0, stream>>>(
            ol, actual, vl, vr, j0, sh_l, ts.is_core ? 1 : 0,
            d_g.ptr, d_eo_l, d_eo_r, d_ev_l, d_ev_r, d_e_acc, d_t.ptr);
      } else {
        k_uhf_mp2_tile<true, false><<<grid, 256, 0, stream>>>(
            ol, actual, vl, vr, j0, sh_l, ts.is_core ? 1 : 0,
            d_g.ptr, d_eo_l, d_eo_r, d_ev_l, d_ev_r, d_e_acc, nullptr);
      }
    } else {
      if (need_cotangents) {
        k_uhf_mp2_tile<false, true><<<grid, 256, 0, stream>>>(
            ol, actual, vl, vr, j0, sh_l, ts.is_core ? 1 : 0,
            d_g.ptr, d_eo_l, d_eo_r, d_ev_l, d_ev_r, d_e_acc, d_t.ptr);
      } else {
        k_uhf_mp2_tile<false, false><<<grid, 256, 0, stream>>>(
            ol, actual, vl, vr, j0, sh_l, ts.is_core ? 1 : 0,
            d_g.ptr, d_eo_l, d_eo_r, d_ev_l, d_ev_r, d_e_acc, nullptr);
      }
    }
    if (!launch_ok()) return false;

    if (!need_cotangents) continue;

    // 1) bar_Fo_l[i, k] += fo_weight * sum_{jl, a, b} t[i, jl, a, b] * t[k, jl, a, b]
    //    (omitting [sh_l:, sh_l:] block when !ts.is_core!)
    const int k_fo = static_cast<int>(Size(actual) * vv);
    if (ts.is_core) {
      if (rm_gemm(h, false, true, ol, ol, k_fo, fo_weight,
                  d_t.ptr, k_fo, d_t.ptr, k_fo,
                  1.0, d_bar_Fo_l, ol) != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
    } else if (sh_l > 0) {
      if (rm_gemm(h, false, true, sh_l, ol, k_fo, fo_weight,
                  d_t.ptr, k_fo, d_t.ptr, k_fo,
                  1.0, d_bar_Fo_l, ol) != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
      if (ol > sh_l) {
        if (rm_gemm(h, false, true, ol - sh_l, sh_l, k_fo, fo_weight,
                    d_t.ptr + Size(sh_l) * k_fo, k_fo, d_t.ptr, k_fo,
                    1.0, d_bar_Fo_l + Size(sh_l) * ol, ol) != CUBLAS_STATUS_SUCCESS) {
          return false;
        }
      }
    }

    const int n_delta_i = ts.is_core ? ol : sh_l;
    if (n_delta_i <= 0) continue;

    // 2) bar_Bl and bar_Br: reuse d_g as tm_mat [n_delta_i * vl, actual * vr]
    const Size delta_elems = Size(n_delta_i) * actual * vv;
    k_pack_t_to_mat<<<grid_for_cap(delta_elems), 256, 0, stream>>>(
        n_delta_i, actual, vl, vr, d_t.ptr, d_g.ptr);
    if (!launch_ok()) return false;

    // bar_Bl[Q, 0 : n_delta_i*vl] += b_weight * (Br_j^T [Q, actual*vr]) @ (tm_mat^T [actual*vr, n_delta_i*vl])
    if (rm_gemm(h, true, true, q, n_delta_i * vl, actual * vr, b_weight,
                d_Br_j, q, d_g.ptr, actual * vr,
                1.0, d_bar_Bl, ol * vl) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    // bar_Br[Q, j0*vr : (j0+actual)*vr] += b_weight * (Bl[0:n_delta_i*vl]^T) @ tm_mat
    if (rm_gemm(h, true, false, q, actual * vr, n_delta_i * vl, b_weight,
                d_Bl_packed, q, d_g.ptr, actual * vr,
                1.0, d_bar_Br + Size(j0) * vr, orr * vr) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }

    // 3) bar_Fv_l and bar_Fv_r
    const int n_delta = n_delta_i * actual;
    if (same) {
      // For same-spin, t[p, a, b] = -t[p, b, a], so T @ T^T == T^T @ T identically;
      // bar_Fv_l == bar_Fv_r receives 2 * fv_weight = 0.5 in one DGEMM with zero repack.
      if (rm_gemm(h, true, false, vr, vr, n_delta * vl, 2.0 * fv_weight,
                  d_t.ptr, vr, d_t.ptr, vr,
                  1.0, d_bar_Fv_r, vr) != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
    } else {
      // bar_Fv_r[b, c] += fv_weight * sum_{p, a} t[p, a, b] * t[p, a, c]
      if (rm_gemm(h, true, false, vr, vr, n_delta * vl, fv_weight,
                  d_t.ptr, vr, d_t.ptr, vr,
                  1.0, d_bar_Fv_r, vr) != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
      // Repack t[p, a, b] -> d_g[a, p, b] for single DGEMM bar_Fv_l[a, c]
      k_pack_pab_to_apb<<<grid_for_cap(delta_elems), 256, 0, stream>>>(
          n_delta, vl, vr, d_t.ptr, d_g.ptr);
      if (!launch_ok()) return false;
      if (rm_gemm(h, false, true, vl, vl, n_delta * vr, fv_weight,
                  d_g.ptr, n_delta * vr, d_g.ptr, n_delta * vr,
                  1.0, d_bar_Fv_l, vl) != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
    }
  }

  // 4) Opposite-spin UHF bar_Fo_r[j, l] via i-tiled pass (exact Delta-CV mask)
  if (need_cotangents && !same && d_bar_Fo_r != nullptr) {
    std::vector<TileSpec> i_tiles;
    for (int i0 = 0; i0 < sh_l; i0 += i_tile) {
      i_tiles.push_back({i0, std::min(sh_l, i0 + i_tile), true});
    }
    for (int i0 = sh_l; i0 < ol; i0 += i_tile) {
      i_tiles.push_back({i0, std::min(ol, i0 + i_tile), false});
    }
    for (const auto& ts : i_tiles) {
      const int i0 = ts.start;
      const int actual_i = ts.stop - i0;
      const double* d_Bl_i = d_Bl_packed + Size(i0) * vl * q;

      // d_g[j*vr + b, il*vl + a] = Br_packed[j*vr + b, :] @ Bl_i[il*vl + a, :]^T
      if (rm_gemm(h, false, true, orr * vr, actual_i * vl, q, 1.0,
                  d_Br_packed, q, d_Bl_i, q,
                  0.0, d_g.ptr, actual_i * vl) != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
      const Size itile_elems = Size(orr) * actual_i * vv;
      k_uhf_ab_for_tile<<<grid_for_cap(itile_elems), 256, 0, stream>>>(
          orr, actual_i, vr, vl, i0, d_g.ptr, d_eo_l, d_eo_r, d_ev_l, d_ev_r, d_t.ptr);
      if (!launch_ok()) return false;

      const int k_for = static_cast<int>(Size(actual_i) * vv);
      if (ts.is_core) {
        if (rm_gemm(h, false, true, orr, orr, k_for, -1.0,
                    d_t.ptr, k_for, d_t.ptr, k_for,
                    1.0, d_bar_Fo_r, orr) != CUBLAS_STATUS_SUCCESS) {
          return false;
        }
      } else if (sh_r > 0) {
        if (rm_gemm(h, false, true, sh_r, orr, k_for, -1.0,
                    d_t.ptr, k_for, d_t.ptr, k_for,
                    1.0, d_bar_Fo_r, orr) != CUBLAS_STATUS_SUCCESS) {
          return false;
        }
        if (orr > sh_r) {
          if (rm_gemm(h, false, true, orr - sh_r, sh_r, k_for, -1.0,
                      d_t.ptr + Size(sh_r) * k_for, k_for, d_t.ptr, k_for,
                      1.0, d_bar_Fo_r + Size(sh_r) * orr, orr) != CUBLAS_STATUS_SUCCESS) {
            return false;
          }
        }
      }
    }
  }
  return cudaStreamSynchronize(stream) == cudaSuccess;
}

// GPU closed-shell RHF spatial fast path (2*t - t^T).
// Energy-only uses i <= j block symmetry (2x fewer FLOPs); cotangents use pure-core/valence j-tiles.
bool accumulate_rhf_spatial_gpu(
    cublasHandle_t h, cudaStream_t stream,
    bool need_cotangents, int q, int o, int v, int sh, int tile_limit, Size scratch_bytes,
    const double* d_B_packed, const double* d_eo, const double* d_ev,
    double* d_e_acc, double* d_bar_B, double* d_bar_Fo, double* d_bar_Fv,
    int* err_code_out = nullptr) {
  if (o <= 0 || v <= 0) return true;
  const Size vv = sat_mul(Size(v), Size(v));

  struct BlockSpec {
    int start;
    int stop;
    bool is_core;
  };

  if (!need_cotangents) {
    // Energy-only: exploit (i, j) symmetry over pure-core and pure-valence blocks (bi <= bj)
    if (sat_mul(vv, sizeof(double)) > scratch_bytes) {
      if (err_code_out) *err_code_out = kErrOom;
      return false;
    }
    const Size max_elems = scratch_bytes / sizeof(double);
    int max_blk = static_cast<int>(std::sqrt(static_cast<double>(max_elems / std::max<Size>(1, vv))));
    max_blk = std::max(1, std::min(o, max_blk));
    if (tile_limit > 0) max_blk = std::max(1, std::min(max_blk, tile_limit));
    while (max_blk > 1 && !fits_int(Size(max_blk) * v)) {
      max_blk /= 2;
    }
    if (!fits_int(Size(max_blk) * v)) return false;

    DoubleBuf d_g;
    if (d_g.alloc(Size(max_blk) * max_blk * vv) != cudaSuccess) {
      if (err_code_out) *err_code_out = kErrOom;
      return false;
    }

    std::vector<BlockSpec> blocks;
    for (int i0 = 0; i0 < sh; i0 += max_blk) {
      blocks.push_back({i0, std::min(sh, i0 + max_blk), true});
    }
    for (int i0 = sh; i0 < o; i0 += max_blk) {
      blocks.push_back({i0, std::min(o, i0 + max_blk), false});
    }

    for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
      const int i0 = blocks[bi].start;
      const int ni = blocks[bi].stop - i0;
      const double* d_Bi = d_B_packed + Size(i0) * v * q;
      for (std::size_t bj = bi; bj < blocks.size(); ++bj) {
        const int j0 = blocks[bj].start;
        const int nj = blocks[bj].stop - j0;
        const double* d_Bj = d_B_packed + Size(j0) * v * q;

        if (rm_gemm(h, false, true, ni * v, nj * v, q, 1.0,
                    d_Bi, q, d_Bj, q,
                    0.0, d_g.ptr, nj * v) != CUBLAS_STATUS_SUCCESS) {
          return false;
        }
        const double scale = (bi == bj) ? 1.0 : 2.0;
        const int is_fc_block = (!blocks[bi].is_core && !blocks[bj].is_core) ? 1 : 0;
        const Size elems = Size(ni) * nj * vv;
        k_rhf_mp2_energy_block<<<grid_for_cap(elems), 256, 0, stream>>>(
            ni, nj, v, scale, is_fc_block, d_g.ptr, d_eo + i0, d_eo + j0, d_ev, d_e_acc);
        if (!launch_ok()) return false;
      }
    }
    return cudaStreamSynchronize(stream) == cudaSuccess;
  }

  // Three cotangent tiles share the budget left after the fixed factor buffers.
  const Size bytes_per_j = sat_mul(sat_mul(Size(3), Size(o)), sat_mul(vv, sizeof(double)));
  if (bytes_per_j > scratch_bytes) {
    if (err_code_out) *err_code_out = kErrOom;
    return false;
  }
  Size max_tile = scratch_bytes / std::max<Size>(1, bytes_per_j);
  max_tile = std::min<Size>(max_tile, static_cast<Size>(o));
  if (tile_limit > 0) max_tile = std::min<Size>(max_tile, static_cast<Size>(tile_limit));
  while (max_tile > 1 &&
         (!fits_int(max_tile * vv) || !fits_int(Size(o) * max_tile * v))) {
    max_tile /= 2;
  }
  if (!fits_int(max_tile * vv) || !fits_int(Size(o) * v) || !fits_int(max_tile * v) ||
      !fits_int(Size(o) * max_tile * v)) {
    return false;
  }
  const int j_tile = static_cast<int>(max_tile);
  const Size buf_elems = Size(o) * j_tile * vv;

  DoubleBuf d_g, d_t, d_u;
  if (d_g.alloc(buf_elems) != cudaSuccess || d_t.alloc(buf_elems) != cudaSuccess ||
      d_u.alloc(buf_elems) != cudaSuccess) {
    if (err_code_out) *err_code_out = kErrOom;
    return false;
  }

  std::vector<BlockSpec> j_tiles;
  for (int j0 = 0; j0 < sh; j0 += j_tile) {
    j_tiles.push_back({j0, std::min(sh, j0 + j_tile), true});
  }
  for (int j0 = sh; j0 < o; j0 += j_tile) {
    j_tiles.push_back({j0, std::min(o, j0 + j_tile), false});
  }

  for (const auto& ts : j_tiles) {
    const int j0 = ts.start;
    const int actual = ts.stop - j0;
    const double* d_Bj = d_B_packed + Size(j0) * v * q;

    // G[i*v + a, jl*v + b] = B_packed[i*v + a, :] @ B_j[jl*v + b, :]^T
    if (rm_gemm(h, false, true, o * v, actual * v, q, 1.0,
                d_B_packed, q, d_Bj, q,
                0.0, d_g.ptr, actual * v) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }

    const Size tile_elems = Size(o) * actual * vv;
    k_rhf_mp2_tile_cot<<<grid_for_cap(tile_elems), 256, 0, stream>>>(
        o, actual, v, j0, sh, ts.is_core ? 1 : 0,
        d_g.ptr, d_eo, d_ev, d_e_acc, d_t.ptr, d_u.ptr);
    if (!launch_ok()) return false;

    // 1) bar_Fo[i, k] += -sum_{jl, a, b} t[i, jl, a, b] * u[k, jl, a, b]
    //    (omitting [sh:, sh:] for valence j-tiles!)
    const int k_fo = static_cast<int>(Size(actual) * vv);
    if (ts.is_core) {
      if (rm_gemm(h, false, true, o, o, k_fo, -1.0,
                  d_t.ptr, k_fo, d_u.ptr, k_fo,
                  1.0, d_bar_Fo, o) != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
    } else if (sh > 0) {
      if (rm_gemm(h, false, true, sh, o, k_fo, -1.0,
                  d_t.ptr, k_fo, d_u.ptr, k_fo,
                  1.0, d_bar_Fo, o) != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
      if (o > sh) {
        if (rm_gemm(h, false, true, o - sh, sh, k_fo, -1.0,
                    d_t.ptr + Size(sh) * k_fo, k_fo, d_u.ptr, k_fo,
                    1.0, d_bar_Fo + Size(sh) * o, o) != CUBLAS_STATUS_SUCCESS) {
          return false;
        }
      }
    }

    const int n_delta_i = ts.is_core ? o : sh;
    if (n_delta_i <= 0) continue;

    // 2) bar_B[Q, 0 : n_delta_i*v] += 4.0 * (B_j^T) @ (um_mat^T), reusing d_g as um_mat
    const Size delta_elems = Size(n_delta_i) * actual * vv;
    k_pack_t_to_mat<<<grid_for_cap(delta_elems), 256, 0, stream>>>(
        n_delta_i, actual, v, v, d_u.ptr, d_g.ptr);
    if (!launch_ok()) return false;

    if (rm_gemm(h, true, true, q, n_delta_i * v, actual * v, 4.0,
                d_Bj, q, d_g.ptr, actual * v,
                1.0, d_bar_B, o * v) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }

    // 3) bar_Fv[a, c] += sum_{p in delta, b} t[p, a, b] * u[p, c, b]
    //    Repack t[p,a,b] -> d_g[a,p,b], then u[p,c,b] -> d_t[c,p,b], then ONE giant DGEMM!
    const int n_delta = n_delta_i * actual;
    k_pack_pab_to_apb<<<grid_for_cap(delta_elems), 256, 0, stream>>>(
        n_delta, v, v, d_t.ptr, d_g.ptr);
    if (!launch_ok()) return false;
    k_pack_pab_to_apb<<<grid_for_cap(delta_elems), 256, 0, stream>>>(
        n_delta, v, v, d_u.ptr, d_t.ptr);
    if (!launch_ok()) return false;
    if (rm_gemm(h, false, true, v, v, n_delta * v, 1.0,
                d_g.ptr, n_delta * v, d_t.ptr, n_delta * v,
                1.0, d_bar_Fv, v) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }
  }
  return cudaStreamSynchronize(stream) == cudaSuccess;
}

#endif  // GQC_SINK_HOST_REFERENCE

}  // namespace

extern "C" {

#ifdef GQC_SINK_HOST_REFERENCE

int gqc_f12_cv_df_jk_fock_host_reference(
    int q, int n, int oa, int ob, int restricted, int q_chunk, std::uint64_t budget_bytes,
    const double* B_jk, const double* D_tot,
    const double* Co_a, const double* Co_b,
    double* J_out, double* Ka_out, double* Kb_out) {
  try {
  (void)budget_bytes;
  if (q <= 0 || n <= 0 || oa < 0 || ob < 0 || oa > n || ob > n) return 1;
  if (!B_jk || !D_tot || !J_out || !Ka_out) return 1;
  if (oa > 0 && !Co_a) return 1;
  if (!restricted && (!Kb_out || (ob > 0 && !Co_b))) return 1;

  const Size nn = Size(n) * n;
  if (!all_finite(B_jk, Size(q) * nn) || !all_finite(D_tot, nn) ||
      (oa > 0 && !all_finite(Co_a, Size(n) * oa)) ||
      (!restricted && ob > 0 && !all_finite(Co_b, Size(n) * ob))) {
    return 1;
  }
  std::fill(J_out, J_out + nn, 0.0);
  std::fill(Ka_out, Ka_out + nn, 0.0);
  if (!restricted) std::fill(Kb_out, Kb_out + nn, 0.0);

  const int chunk = std::max(1, std::min(q, q_chunk > 0 ? q_chunk : q));
  std::vector<double> rho(chunk);
  std::vector<double> x_a(oa > 0 ? Size(n) * oa : 1);
  std::vector<double> x_b((!restricted && ob > 0) ? Size(n) * ob : 1);

  for (int q0 = 0; q0 < q; q0 += chunk) {
    const int actual = std::min(chunk, q - q0);
    const double* B_ch = B_jk + Size(q0) * nn;
    // rho[a] = sum_{p,q} B_ch[a, pq] * D_tot[pq]
    rm_gemm_host(false, false, actual, 1, int(nn), 1.0, B_ch, int(nn), D_tot, 1, 0.0, rho.data(), 1);
    // J[pq] += sum_a rho[a] * B_ch[a, pq]
    rm_gemm_host(true, false, 1, int(nn), actual, 1.0, rho.data(), 1, B_ch, int(nn), 1.0, J_out, int(nn));

    for (int a = 0; a < actual; ++a) {
      const double* B_a = B_ch + Size(a) * nn;
      if (oa > 0) {
        rm_gemm_host(false, false, n, oa, n, 1.0, B_a, n, Co_a, oa, 0.0, x_a.data(), oa);
        rm_gemm_host(false, true, n, n, oa, 1.0, x_a.data(), oa, x_a.data(), oa, 1.0, Ka_out, n);
      }
      if (!restricted && ob > 0) {
        rm_gemm_host(false, false, n, ob, n, 1.0, B_a, n, Co_b, ob, 0.0, x_b.data(), ob);
        rm_gemm_host(false, true, n, n, ob, 1.0, x_b.data(), ob, x_b.data(), ob, 1.0, Kb_out, n);
      }
    }
  }

  if (!all_finite(J_out, nn) || !all_finite(Ka_out, nn)) return 2;
  if (!restricted && !all_finite(Kb_out, nn)) return 2;
  return 0;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_cv_mp2_factors_fwd_host_reference(
    int q, int n, int o_act, int n_vir, int q_chunk, std::uint64_t budget_bytes,
    const double* ao_coulomb, const double* whitener,
    const double* co, const double* cv, double* B_out) {
  try {
  (void)budget_bytes;
  if (q <= 0 || n <= 0 || o_act <= 0 || n_vir <= 0) return 1;
  if (!ao_coulomb || !whitener || !co || !cv || !B_out) return 1;

  const Size nn = Size(n) * n;
  const Size ov = Size(o_act) * n_vir;
  if (!all_finite(ao_coulomb, Size(q) * nn) || !all_finite(whitener, Size(q) * q) ||
      !all_finite(co, Size(n) * o_act) || !all_finite(cv, Size(n) * n_vir)) {
    return 1;
  }
  const int chunk = std::max(1, std::min(q, q_chunk > 0 ? q_chunk : q));
  std::vector<double> mo_raw(Size(q) * ov);
  std::vector<double> tmp(Size(o_act) * n);

  for (int q0 = 0; q0 < q; q0 += chunk) {
    const int q1 = std::min(q, q0 + chunk);
    for (int a = q0; a < q1; ++a) {
      transform_ao_slice_host(n, o_act, n_vir, co, o_act, ao_coulomb + Size(a) * nn,
                              cv, n_vir, tmp.data(), mo_raw.data() + Size(a) * ov);
    }
  }

  // B_out[q, ov] = whitener[q, q] @ mo_raw[q, ov]
  rm_gemm_host(false, false, q, int(ov), q, 1.0, whitener, q, mo_raw.data(), int(ov),
               0.0, B_out, int(ov));
  return all_finite(B_out, Size(q) * ov) ? 0 : 2;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_cv_mp2_factors_rev_host_reference(
    int q, int n, int o_act, int n_vir, int q_chunk, std::uint64_t budget_bytes,
    const double* ao_coulomb, const double* whitener,
    const double* co, const double* cv,
    const double* B, const double* bar_B,
    double* bar_co_out, double* bar_cv_out,
    double* bar_ao_out, double* metric_bar_out) {
  try {
  (void)budget_bytes;
  if (q <= 0 || n <= 0 || o_act <= 0 || n_vir <= 0) return 1;
  if (!ao_coulomb || !whitener || !co || !cv || !B || !bar_B) return 1;
  if (!bar_co_out || !bar_cv_out || !bar_ao_out || !metric_bar_out) return 1;

  const Size nn = Size(n) * n;
  const Size ov = Size(o_act) * n_vir;
  if (!all_finite(ao_coulomb, Size(q) * nn) || !all_finite(whitener, Size(q) * q) ||
      !all_finite(co, Size(n) * o_act) || !all_finite(cv, Size(n) * n_vir) ||
      !all_finite(B, Size(q) * ov) || !all_finite(bar_B, Size(q) * ov)) {
    return 1;
  }
  std::fill(bar_co_out, bar_co_out + Size(n) * o_act, 0.0);
  std::fill(bar_cv_out, bar_cv_out + Size(n) * n_vir, 0.0);

  std::vector<double> bar_mo_raw(Size(q) * ov);
  std::vector<double> fitted(Size(q) * ov);
  // bar_mo_raw = W^T @ bar_B; fitted = W^T @ B
  rm_gemm_host(true, false, q, int(ov), q, 1.0, whitener, q, bar_B, int(ov),
               0.0, bar_mo_raw.data(), int(ov));
  rm_gemm_host(true, false, q, int(ov), q, 1.0, whitener, q, B, int(ov),
               0.0, fitted.data(), int(ov));

  // metric_bar = -0.25 * (bar_mo_raw @ fitted^T + (bar_mo_raw @ fitted^T)^T)
  rm_gemm_host(false, true, q, q, int(ov), -0.5, bar_mo_raw.data(), int(ov),
               fitted.data(), int(ov), 0.0, metric_bar_out, q);
  symmetrize_inplace(q, metric_bar_out);

  const int chunk = std::max(1, std::min(q, q_chunk > 0 ? q_chunk : q));
  std::vector<double> t_slice(Size(o_act) * n);
  std::vector<double> bar_t(Size(o_act) * n);

  for (int q0 = 0; q0 < q; q0 += chunk) {
    const int q1 = std::min(q, q0 + chunk);
    for (int a = q0; a < q1; ++a) {
      const double* ao_a = ao_coulomb + Size(a) * nn;
      const double* bm_a = bar_mo_raw.data() + Size(a) * ov;
      double* bao_a = bar_ao_out + Size(a) * nn;

      // Forward intermediate: t_slice = co^T @ ao_a  [o_act, n]
      rm_gemm_host(true, false, o_act, n, n, 1.0, co, o_act, ao_a, n, 0.0, t_slice.data(), n);
      // bar_cv += t_slice^T @ bm_a  [n, n_vir]
      rm_gemm_host(true, false, n, n_vir, o_act, 1.0, t_slice.data(), n, bm_a, n_vir,
                   1.0, bar_cv_out, n_vir);
      // bar_t = bm_a @ cv^T  [o_act, n]
      rm_gemm_host(false, true, o_act, n, n_vir, 1.0, bm_a, n_vir, cv, n_vir,
                   0.0, bar_t.data(), n);
      // bar_co += ao_a @ bar_t^T  [n, o_act]
      rm_gemm_host(false, true, n, o_act, n, 1.0, ao_a, n, bar_t.data(), n,
                   1.0, bar_co_out, o_act);
      // bar_ao[a] = co @ bar_t  [n, n]
      rm_gemm_host(false, false, n, n, o_act, 1.0, co, o_act, bar_t.data(), n,
                   0.0, bao_a, n);
    }
  }

  if (!all_finite(bar_co_out, Size(n) * o_act) ||
      !all_finite(bar_cv_out, Size(n) * n_vir) ||
      !all_finite(bar_ao_out, Size(q) * nn) ||
      !all_finite(metric_bar_out, Size(q) * q)) {
    return 2;
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

int gqc_f12_cv_mp2_pairs_host_reference(
    int restricted, int need_cotangents, int q,
    int oa, int ob, int va, int vb, int sh_a, int sh_b, int tile_limit,
    std::uint64_t budget_bytes,
    const double* B_a, const double* B_b,
    const double* eo_a, const double* eo_b,
    const double* ev_a, const double* ev_b,
    double* cv_spin_e_out, double* fc_spin_e_out,
    double* bar_B_a_out, double* bar_B_b_out,
    double* bar_Fo_a_out, double* bar_Fo_b_out,
    double* bar_Fv_a_out, double* bar_Fv_b_out) {
  try {
  (void)budget_bytes;
  if (q <= 0 || oa < 0 || ob < 0 || va < 0 || vb < 0) return 1;
  if (sh_a < 0 || sh_a > oa || sh_b < 0 || sh_b > ob) return 1;
  if (!cv_spin_e_out || !fc_spin_e_out) return 1;
  if (oa > 0 && va > 0 && (!B_a || !eo_a || !ev_a)) return 1;
  if (!restricted && ob > 0 && vb > 0 && (!B_b || !eo_b || !ev_b)) return 1;
  if (restricted && (oa != ob || va != vb || sh_a != sh_b)) return 1;
  if (oa > 0 && va > 0 &&
      (!all_finite(B_a, Size(q) * oa * va) || !all_finite(eo_a, Size(oa)) ||
       !all_finite(ev_a, Size(va)))) {
    return 1;
  }
  if (!restricted && ob > 0 && vb > 0 &&
      (!all_finite(B_b, Size(q) * ob * vb) || !all_finite(eo_b, Size(ob)) ||
       !all_finite(ev_b, Size(vb)))) {
    return 1;
  }

  if (need_cotangents) {
    if (oa > 0 && va > 0) {
      if (!bar_B_a_out || !bar_Fo_a_out || !bar_Fv_a_out) return 1;
      std::fill(bar_B_a_out, bar_B_a_out + Size(q) * oa * va, 0.0);
      std::fill(bar_Fo_a_out, bar_Fo_a_out + Size(oa) * oa, 0.0);
      std::fill(bar_Fv_a_out, bar_Fv_a_out + Size(va) * va, 0.0);
    }
    if (!restricted && ob > 0 && vb > 0) {
      if (!bar_B_b_out || !bar_Fo_b_out || !bar_Fv_b_out) return 1;
      std::fill(bar_B_b_out, bar_B_b_out + Size(q) * ob * vb, 0.0);
      std::fill(bar_Fo_b_out, bar_Fo_b_out + Size(ob) * ob, 0.0);
      std::fill(bar_Fv_b_out, bar_Fv_b_out + Size(vb) * vb, 0.0);
    }
  }

  if (restricted) {
    accumulate_rhf_spatial_host(
        need_cotangents != 0, q, oa, va, sh_a, tile_limit,
        B_a, eo_a, ev_a,
        cv_spin_e_out, fc_spin_e_out,
        bar_B_a_out, bar_Fo_a_out, bar_Fv_a_out);
  } else {
    cv_spin_e_out[0] = cv_spin_e_out[1] = cv_spin_e_out[2] = 0.0;
    fc_spin_e_out[0] = fc_spin_e_out[1] = fc_spin_e_out[2] = 0.0;
    // aa
    accumulate_uhf_channel_host(
        true, need_cotangents != 0, q, oa, oa, va, va, sh_a, sh_a, tile_limit,
        B_a, B_a, eo_a, eo_a, ev_a, ev_a,
        cv_spin_e_out[0], fc_spin_e_out[0],
        bar_B_a_out, bar_B_a_out, bar_Fo_a_out, nullptr, bar_Fv_a_out, bar_Fv_a_out);
    // ab
    accumulate_uhf_channel_host(
        false, need_cotangents != 0, q, oa, ob, va, vb, sh_a, sh_b, tile_limit,
        B_a, B_b, eo_a, eo_b, ev_a, ev_b,
        cv_spin_e_out[1], fc_spin_e_out[1],
        bar_B_a_out, bar_B_b_out, bar_Fo_a_out, bar_Fo_b_out, bar_Fv_a_out, bar_Fv_b_out);
    // bb
    accumulate_uhf_channel_host(
        true, need_cotangents != 0, q, ob, ob, vb, vb, sh_b, sh_b, tile_limit,
        B_b, B_b, eo_b, eo_b, ev_b, ev_b,
        cv_spin_e_out[2], fc_spin_e_out[2],
        bar_B_b_out, bar_B_b_out, bar_Fo_b_out, nullptr, bar_Fv_b_out, bar_Fv_b_out);
  }

  if (!all_finite(cv_spin_e_out, 3) || !all_finite(fc_spin_e_out, 3)) return 2;
  if (need_cotangents) {
    if (oa > 0 && va > 0 &&
        (!all_finite(bar_B_a_out, Size(q) * oa * va) ||
         !all_finite(bar_Fo_a_out, Size(oa) * oa) ||
         !all_finite(bar_Fv_a_out, Size(va) * va))) {
      return 2;
    }
    if (!restricted && ob > 0 && vb > 0 &&
        (!all_finite(bar_B_b_out, Size(q) * ob * vb) ||
         !all_finite(bar_Fo_b_out, Size(ob) * ob) ||
         !all_finite(bar_Fv_b_out, Size(vb) * vb))) {
      return 2;
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

int gqc_f12_cv_jk_mo_fock_fwd_host_reference(
    int q, int n, int oa, int ob, int restricted, int q_chunk, std::uint64_t budget_bytes,
    const double* ao_coulomb, const double* hcore_ao, const double* whitener,
    const double* C_a, const double* C_b,
    double* df_a_out, double* df_b_out,
    double* fock_a_out, double* fock_b_out) {
  try {
  (void)budget_bytes;
  if (q <= 0 || n <= 0 || oa < 0 || ob < 0 || oa >= n || ob >= n || (oa + ob) <= 0) return 1;
  if (!ao_coulomb || !hcore_ao || !whitener || !C_a || !df_a_out || !fock_a_out) return 1;
  if (!restricted && (!C_b || !df_b_out || !fock_b_out)) return 1;

  const Size nn = Size(n) * n;
  if (!all_finite(ao_coulomb, Size(q) * nn) || !all_finite(hcore_ao, nn) ||
      !all_finite(whitener, Size(q) * q) || !all_finite(C_a, nn) ||
      (!restricted && !all_finite(C_b, nn))) {
    return 1;
  }
  const int chunk = std::max(1, std::min(q, q_chunk > 0 ? q_chunk : q));
  std::vector<double> tmp(nn);
  std::vector<double> raw_s(Size(q) * nn);

  auto project_and_whiten = [&](const double* C, double* df_out) {
    for (int q0 = 0; q0 < q; q0 += chunk) {
      const int q1 = std::min(q, q0 + chunk);
      for (int a = q0; a < q1; ++a) {
        transform_ao_slice_host(n, n, n, C, n, ao_coulomb + Size(a) * nn,
                                C, n, tmp.data(), raw_s.data() + Size(a) * nn);
      }
    }
    rm_gemm_host(false, false, q, int(nn), q, 1.0, whitener, q, raw_s.data(), int(nn),
                 0.0, df_out, int(nn));
  };

  project_and_whiten(C_a, df_a_out);
  if (!restricted) {
    project_and_whiten(C_b, df_b_out);
  }

  // Compute shared charge_df[A] = sum_{s, i < o_s} df_s[A, i, i]
  std::vector<double> charge_df(q, 0.0);
  const double a_scale = restricted ? 2.0 : 1.0;
  for (int a = 0; a < q; ++a) {
    const double* df_a_sl = df_a_out + Size(a) * nn;
    double tr = 0.0;
    for (int i = 0; i < oa; ++i) tr += df_a_sl[Size(i) * n + i];
    charge_df[a] += a_scale * tr;
    if (!restricted && ob > 0) {
      const double* df_b_sl = df_b_out + Size(a) * nn;
      double tr_b = 0.0;
      for (int i = 0; i < ob; ++i) tr_b += df_b_sl[Size(i) * n + i];
      charge_df[a] += tr_b;
    }
  }

  auto build_fock = [&](const double* C, const double* df_s, int occ, double* fock_out) {
    // hcore_mo = C^T @ hcore_ao @ C
    transform_ao_slice_host(n, n, n, C, n, hcore_ao, C, n, tmp.data(), fock_out);
    // + J = sum_A charge_df[A] * df_s[A]
    rm_gemm_host(true, false, 1, int(nn), q, 1.0, charge_df.data(), 1, df_s, int(nn),
                 1.0, fock_out, int(nn));
    // - K = -sum_{A, i < occ} df_s[A, i, :]^T @ df_s[A, i, :]
    if (occ > 0) {
      for (int a = 0; a < q; ++a) {
        const double* df_occ = df_s + Size(a) * nn;  // leading [occ, n] rows
        rm_gemm_host(true, false, n, n, occ, -1.0, df_occ, n, df_occ, n, 1.0, fock_out, n);
      }
    }
  };

  build_fock(C_a, df_a_out, oa, fock_a_out);
  if (!restricted) {
    build_fock(C_b, df_b_out, ob, fock_b_out);
  }

  if (!all_finite(df_a_out, Size(q) * nn) || !all_finite(fock_a_out, nn)) return 2;
  if (!restricted && (!all_finite(df_b_out, Size(q) * nn) || !all_finite(fock_b_out, nn))) return 2;
  return 0;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_cv_jk_fock_pullback_host_reference(
    int q, int n, int oa, int ob, int restricted, int q_chunk, std::uint64_t budget_bytes,
    const double* ao_coulomb, const double* hcore_ao, const double* whitener,
    const double* C_a, const double* C_b,
    const double* df_a, const double* df_b,
    const double* S_a, const double* S_b,
    double* bar_C_a_out, double* bar_C_b_out,
    double* bar_hcore_out, double* bar_ao_out, double* bar_metric_out) {
  try {
  (void)budget_bytes;
  if (q <= 0 || n <= 0 || oa < 0 || ob < 0 || oa >= n || ob >= n || (oa + ob) <= 0) return 1;
  if (!ao_coulomb || !hcore_ao || !whitener || !C_a || !df_a || !S_a) return 1;
  if (!bar_C_a_out || !bar_hcore_out || !bar_ao_out || !bar_metric_out) return 1;
  const bool share_spin = (restricted != 0) && (S_b == nullptr);
  if (!share_spin && (!C_b || !df_b || !S_b || !bar_C_b_out)) return 1;

  const Size nn = Size(n) * n;
  if (!all_finite(ao_coulomb, Size(q) * nn) || !all_finite(hcore_ao, nn) ||
      !all_finite(whitener, Size(q) * q) || !all_finite(C_a, nn) ||
      !all_finite(df_a, Size(q) * nn) || !all_finite(S_a, nn) ||
      (!share_spin && (!all_finite(C_b, nn) || !all_finite(df_b, Size(q) * nn) ||
                       !all_finite(S_b, nn)))) {
    return 1;
  }
  std::fill(bar_C_a_out, bar_C_a_out + nn, 0.0);
  if (!share_spin && bar_C_b_out) std::fill(bar_C_b_out, bar_C_b_out + nn, 0.0);
  std::fill(bar_hcore_out, bar_hcore_out + nn, 0.0);
  std::fill(bar_ao_out, bar_ao_out + Size(q) * nn, 0.0);

  // 1) charge_df[A] and bar_charge_df[A]
  std::vector<double> charge_df(q, 0.0);
  std::vector<double> bar_charge_df(q, 0.0);
  const double share_scale = share_spin ? 2.0 : 1.0;

  rm_gemm_host(false, false, q, 1, int(nn), share_scale, df_a, int(nn), S_a, 1,
               0.0, bar_charge_df.data(), 1);
  for (int a = 0; a < q; ++a) {
    const double* sl = df_a + Size(a) * nn;
    double tr = 0.0;
    for (int i = 0; i < oa; ++i) tr += sl[Size(i) * n + i];
    charge_df[a] += share_scale * tr;
  }
  if (!share_spin) {
    rm_gemm_host(false, false, q, 1, int(nn), 1.0, df_b, int(nn), S_b, 1,
                 1.0, bar_charge_df.data(), 1);
    for (int a = 0; a < q; ++a) {
      const double* sl = df_b + Size(a) * nn;
      double tr = 0.0;
      for (int i = 0; i < ob; ++i) tr += sl[Size(i) * n + i];
      charge_df[a] += tr;
    }
  }

  std::vector<double> bar_M(Size(q) * q, 0.0);
  std::vector<double> bar_df_s(Size(q) * nn);
  std::vector<double> bar_raw_s(Size(q) * nn);
  std::vector<double> S_sym(nn);
  std::vector<double> tmp1(nn);
  std::vector<double> tmp2(nn);
  const int chunk = std::max(1, std::min(q, q_chunk > 0 ? q_chunk : q));

  auto process_spin = [&](const double* C, const double* df_s, const double* S, int occ,
                          double scale, double* bar_C) {
    // hcore pullback:
    //   bar_hcore += scale * C @ S @ C^T
    //   bar_C += hcore_ao @ C @ S^T + hcore_ao^T @ C @ S
    rm_gemm_host(false, false, n, n, n, 1.0, C, n, S, n, 0.0, tmp1.data(), n);
    rm_gemm_host(false, true, n, n, n, scale, tmp1.data(), n, C, n, 1.0, bar_hcore_out, n);
    rm_gemm_host(false, false, n, n, n, 1.0, hcore_ao, n, C, n, 0.0, tmp1.data(), n);
    rm_gemm_host(false, true, n, n, n, 1.0, tmp1.data(), n, S, n, 1.0, bar_C, n);
    rm_gemm_host(true, false, n, n, n, 1.0, hcore_ao, n, C, n, 0.0, tmp1.data(), n);
    rm_gemm_host(false, false, n, n, n, 1.0, tmp1.data(), n, S, n, 1.0, bar_C, n);

    for (int p = 0; p < n; ++p) {
      for (int r = 0; r < n; ++r) {
        S_sym[Size(p) * n + r] = S[Size(p) * n + r] + S[Size(r) * n + p];
      }
    }

    // Build bar_df_s[A, p, q]
    for (int a = 0; a < q; ++a) {
      double* bdf = bar_df_s.data() + Size(a) * nn;
      const double ch = charge_df[a];
      for (Size idx = 0; idx < nn; ++idx) bdf[idx] = ch * S[idx];
      if (occ > 0) {
        const double bch = bar_charge_df[a];
        for (int i = 0; i < occ; ++i) bdf[Size(i) * n + i] += bch;
        // bdf[:occ, :] -= df_s[a, :occ, :] @ S_sym
        const double* df_occ = df_s + Size(a) * nn;
        rm_gemm_host(false, false, occ, n, n, -1.0, df_occ, n, S_sym.data(), n, 1.0, bdf, n);
      }
    }

    // bar_M[A, B] += (-0.5 * scale) * sum_{p,q} bar_df_s[A, pq] * df_s[B, pq]
    rm_gemm_host(false, true, q, q, int(nn), -0.5 * scale, bar_df_s.data(), int(nn),
                 df_s, int(nn), 1.0, bar_M.data(), q);

    // bar_raw_s = W^T @ bar_df_s  [q, nn]
    rm_gemm_host(true, false, q, int(nn), q, 1.0, whitener, q, bar_df_s.data(), int(nn),
                 0.0, bar_raw_s.data(), int(nn));

    // Q-chunked back-projection to bar_ao and bar_C
    for (int q0 = 0; q0 < q; q0 += chunk) {
      const int q1 = std::min(q, q0 + chunk);
      for (int a = q0; a < q1; ++a) {
        const double* ao_a = ao_coulomb + Size(a) * nn;
        const double* br_a = bar_raw_s.data() + Size(a) * nn;
        double* bao_a = bar_ao_out + Size(a) * nn;

        // bar_ao[a] += scale * (C @ br_a @ C^T)
        rm_gemm_host(false, false, n, n, n, 1.0, C, n, br_a, n, 0.0, tmp1.data(), n);
        rm_gemm_host(false, true, n, n, n, scale, tmp1.data(), n, C, n, 1.0, bao_a, n);

        // bar_C += ao_a @ C @ br_a^T + ao_a^T @ C @ br_a
        rm_gemm_host(false, false, n, n, n, 1.0, ao_a, n, C, n, 0.0, tmp1.data(), n);
        rm_gemm_host(false, true, n, n, n, 1.0, tmp1.data(), n, br_a, n, 1.0, bar_C, n);
        rm_gemm_host(true, false, n, n, n, 1.0, ao_a, n, C, n, 0.0, tmp2.data(), n);
        rm_gemm_host(false, false, n, n, n, 1.0, tmp2.data(), n, br_a, n, 1.0, bar_C, n);
      }
    }
  };

  process_spin(C_a, df_a, S_a, oa, share_scale, bar_C_a_out);
  if (!share_spin) {
    process_spin(C_b, df_b, S_b, ob, 1.0, bar_C_b_out);
  }

  // bar_metric = sym(W^T @ bar_M @ W)
  std::vector<double> mw(Size(q) * q);
  rm_gemm_host(false, false, q, q, q, 1.0, bar_M.data(), q, whitener, q, 0.0, mw.data(), q);
  rm_gemm_host(true, false, q, q, q, 1.0, whitener, q, mw.data(), q, 0.0, bar_metric_out, q);
  symmetrize_inplace(q, bar_metric_out);

  if (!all_finite(bar_C_a_out, nn) || !all_finite(bar_hcore_out, nn) ||
      !all_finite(bar_ao_out, Size(q) * nn) || !all_finite(bar_metric_out, Size(q) * q)) {
    return 2;
  }
  if (!share_spin && !all_finite(bar_C_b_out, nn)) return 2;
  return 0;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_cv_cphf_matvec_host_reference(
    int q, int n, int oa, int ob, int restricted, int q_chunk, std::uint64_t budget_bytes,
    const double* df_a, const double* df_b,
    const double* gap_a, const double* gap_b,
    const double* trial_a, const double* trial_b,
    double* out_a, double* out_b) {
  try {
  (void)budget_bytes;
  if (q <= 0 || n <= 0 || oa < 0 || ob < 0 || oa >= n || ob >= n || (oa + ob) <= 0) return 1;
  if (oa > 0 && (!df_a || !gap_a || !trial_a || !out_a)) return 1;
  if (!restricted && ob > 0 && (!df_b || !gap_b || !trial_b || !out_b)) return 1;

  const Size nn = Size(n) * n;
  const int va = n - oa;
  const int vb = n - ob;
  if (oa > 0 && (!all_finite(df_a, Size(q) * nn) || !all_finite(gap_a, Size(va) * oa) ||
                 !all_finite(trial_a, Size(va) * oa))) {
    return 1;
  }
  if (!restricted && ob > 0 &&
      (!all_finite(df_b, Size(q) * nn) || !all_finite(gap_b, Size(vb) * ob) ||
       !all_finite(trial_b, Size(vb) * ob))) {
    return 1;
  }
  const int chunk = std::max(1, std::min(q, q_chunk > 0 ? q_chunk : q));
  (void)chunk;

  if (oa > 0) {
    for (int idx = 0; idx < va * oa; ++idx) {
      out_a[idx] = gap_a[idx] * trial_a[idx];
    }
  }
  if (!restricted && ob > 0) {
    for (int idx = 0; idx < vb * ob; ++idx) {
      out_b[idx] = gap_b[idx] * trial_b[idx];
    }
  }

  std::vector<double> kv_a(oa > 0 ? Size(va) * oa : 1);
  std::vector<double> ki_a(oa > 0 ? Size(oa) * oa : 1);
  std::vector<double> kv_b((!restricted && ob > 0) ? Size(vb) * ob : 1);
  std::vector<double> ki_b((!restricted && ob > 0) ? Size(ob) * ob : 1);

  for (int a = 0; a < q; ++a) {
    double ch_a = 0.0;
    double ch_b = 0.0;

    if (oa > 0) {
      const double* df_sl = df_a + Size(a) * nn;
      const double* vo = df_sl + Size(oa) * n;  // [va, oa] with stride n
      for (int r = 0; r < va; ++r) {
        const double* vo_r = vo + Size(r) * n;
        const double* tr_r = trial_a + Size(r) * oa;
        for (int i = 0; i < oa; ++i) ch_a += vo_r[i] * tr_r[i];
      }
    }
    if (!restricted && ob > 0) {
      const double* df_sl = df_b + Size(a) * nn;
      const double* vo = df_sl + Size(ob) * n;  // [vb, ob] with stride n
      for (int r = 0; r < vb; ++r) {
        const double* vo_r = vo + Size(r) * n;
        const double* tr_r = trial_b + Size(r) * ob;
        for (int i = 0; i < ob; ++i) ch_b += vo_r[i] * tr_r[i];
      }
    }

    const double j_coeff = restricted ? (4.0 * ch_a) : (2.0 * (ch_a + ch_b));

    if (oa > 0) {
      const double* df_sl = df_a + Size(a) * nn;
      const double* oo = df_sl;                         // [oa, oa] stride n
      const double* vo = df_sl + Size(oa) * n;          // [va, oa] stride n
      const double* vv = df_sl + Size(oa) * n + oa;     // [va, va] stride n

      // + j_coeff * vo
      for (int r = 0; r < va; ++r) {
        const double* vo_r = vo + Size(r) * n;
        double* out_r = out_a + Size(r) * oa;
        for (int i = 0; i < oa; ++i) out_r[i] += j_coeff * vo_r[i];
      }
      // kv[a, j] = sum_b vv[a, b] * trial[b, j]
      rm_gemm_host(false, false, va, oa, va, 1.0, vv, n, trial_a, oa, 0.0, kv_a.data(), oa);
      // out[a, i] -= sum_j kv[a, j] * oo[i, j]
      rm_gemm_host(false, true, va, oa, oa, -1.0, kv_a.data(), oa, oo, n, 1.0, out_a, oa);
      // ki[i, j] = sum_b vo[b, i] * trial[b, j]
      rm_gemm_host(true, false, oa, oa, va, 1.0, vo, n, trial_a, oa, 0.0, ki_a.data(), oa);
      // out[a, i] -= sum_j vo[a, j] * ki[i, j]
      rm_gemm_host(false, true, va, oa, oa, -1.0, vo, n, ki_a.data(), oa, 1.0, out_a, oa);
    }

    if (!restricted && ob > 0) {
      const double* df_sl = df_b + Size(a) * nn;
      const double* oo = df_sl;
      const double* vo = df_sl + Size(ob) * n;
      const double* vv = df_sl + Size(ob) * n + ob;

      for (int r = 0; r < vb; ++r) {
        const double* vo_r = vo + Size(r) * n;
        double* out_r = out_b + Size(r) * ob;
        for (int i = 0; i < ob; ++i) out_r[i] += j_coeff * vo_r[i];
      }
      rm_gemm_host(false, false, vb, ob, vb, 1.0, vv, n, trial_b, ob, 0.0, kv_b.data(), ob);
      rm_gemm_host(false, true, vb, ob, ob, -1.0, kv_b.data(), ob, oo, n, 1.0, out_b, ob);
      rm_gemm_host(true, false, ob, ob, vb, 1.0, vo, n, trial_b, ob, 0.0, ki_b.data(), ob);
      rm_gemm_host(false, true, vb, ob, ob, -1.0, vo, n, ki_b.data(), ob, 1.0, out_b, ob);
    }
  }

  if (oa > 0 && !all_finite(out_a, Size(va) * oa)) return 2;
  if (!restricted && ob > 0 && !all_finite(out_b, Size(vb) * ob)) return 2;
  return 0;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_cv_nuclear_source_weights_host_reference(
    int q, int n, int oa, int ob, int restricted, int q_chunk, std::uint64_t budget_bytes,
    const double* whitener,
    const double* C_a, const double* C_b,
    const double* df_a, const double* df_b,
    const double* p_a, const double* p_b,
    const double* sm_a, const double* sm_b,
    double* bar_ds_out, double* bar_dh_out,
    double* bar_dr_out, double* bar_dj_out) {
  try {
  (void)budget_bytes;
  if (q <= 0 || n <= 0 || oa < 0 || ob < 0 || oa >= n || ob >= n || (oa + ob) <= 0) return 1;
  if (!whitener || !C_a || !df_a || !p_a || !sm_a) return 1;
  if (!bar_ds_out || !bar_dh_out || !bar_dr_out || !bar_dj_out) return 1;
  if (!restricted && (!C_b || !df_b || !p_b || !sm_b)) return 1;

  const Size nn = Size(n) * n;
  if (!all_finite(whitener, Size(q) * q) || !all_finite(C_a, nn) ||
      !all_finite(df_a, Size(q) * nn) || !all_finite(p_a, nn) || !all_finite(sm_a, nn) ||
      (!restricted && (!all_finite(C_b, nn) || !all_finite(df_b, Size(q) * nn) ||
                       !all_finite(p_b, nn) || !all_finite(sm_b, nn)))) {
    return 1;
  }
  std::fill(bar_ds_out, bar_ds_out + nn, 0.0);
  std::fill(bar_dh_out, bar_dh_out + nn, 0.0);
  std::fill(bar_dr_out, bar_dr_out + Size(q) * nn, 0.0);

  const double share_scale = restricted ? 2.0 : 1.0;

  // 1) charge[A] and bar_dcharge[A]
  std::vector<double> charge(q, 0.0);
  std::vector<double> bar_dcharge(q, 0.0);
  rm_gemm_host(false, false, q, 1, int(nn), share_scale, df_a, int(nn), p_a, 1,
               0.0, bar_dcharge.data(), 1);
  for (int a = 0; a < q; ++a) {
    const double* sl = df_a + Size(a) * nn;
    double tr = 0.0;
    for (int i = 0; i < oa; ++i) tr += sl[Size(i) * n + i];
    charge[a] += share_scale * tr;
  }
  if (!restricted) {
    rm_gemm_host(false, false, q, 1, int(nn), 1.0, df_b, int(nn), p_b, 1,
                 1.0, bar_dcharge.data(), 1);
    for (int a = 0; a < q; ++a) {
      const double* sl = df_b + Size(a) * nn;
      double tr = 0.0;
      for (int i = 0; i < ob; ++i) tr += sl[Size(i) * n + i];
      charge[a] += tr;
    }
  }

  // 2) Complete bar_sm_s and assemble bar_dh, bar_ds
  std::vector<double> tmp1(nn);
  auto finish_one_body = [&](const double* C, const double* df_s, const double* p_s,
                             const double* sm_in, int occ, double scale) {
    std::vector<double> bar_sm(sm_in, sm_in + nn);
    if (occ > 0) {
      std::vector<double> pb(Size(n) * occ);
      std::vector<double> s_ex(Size(occ) * occ);
      for (int a = 0; a < q; ++a) {
        const double* sl = df_s + Size(a) * nn;
        const double mc = -bar_dcharge[a];
        for (int i = 0; i < occ; ++i) {
          for (int j = 0; j < occ; ++j) {
            bar_sm[Size(i) * n + j] += mc * sl[Size(i) * n + j];
          }
        }
        // pb[q, i] = sum_p p_s[p, q] * sl[p, i]  (i.e. p_s^T @ sl[:, :occ])
        rm_gemm_host(true, false, n, occ, n, 1.0, p_s, n, sl, n, 0.0, pb.data(), occ);
        // s_ex[i, j] = sum_q pb[q, i] * sl[q, j]
        rm_gemm_host(true, false, occ, occ, n, 1.0, pb.data(), occ, sl, n, 0.0, s_ex.data(), occ);
        for (int i = 0; i < occ; ++i) {
          for (int j = 0; j < occ; ++j) {
            bar_sm[Size(i) * n + j] += s_ex[Size(i) * occ + j];
          }
        }
      }
    }
    // bar_dh += scale * (C @ p_s @ C^T)
    rm_gemm_host(false, false, n, n, n, 1.0, C, n, p_s, n, 0.0, tmp1.data(), n);
    rm_gemm_host(false, true, n, n, n, scale, tmp1.data(), n, C, n, 1.0, bar_dh_out, n);
    // bar_ds += scale * (C @ bar_sm @ C^T)
    rm_gemm_host(false, false, n, n, n, 1.0, C, n, bar_sm.data(), n, 0.0, tmp1.data(), n);
    rm_gemm_host(false, true, n, n, n, scale, tmp1.data(), n, C, n, 1.0, bar_ds_out, n);
  };

  finish_one_body(C_a, df_a, p_a, sm_a, oa, share_scale);
  if (!restricted) {
    finish_one_body(C_b, df_b, p_b, sm_b, ob, 1.0);
  }
  symmetrize_inplace(n, bar_dh_out);
  symmetrize_inplace(n, bar_ds_out);

  // 3) Q-chunked bar_dB -> bar_M [q, q] and ao_db_sym [q, n, n] -> bar_dr = W^T @ ao_db_sym
  const int chunk = std::max(1, std::min(q, q_chunk > 0 ? q_chunk : q));
  std::vector<double> bar_M(Size(q) * q, 0.0);
  std::vector<double> psym_a(nn), psym_b(nn);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      psym_a[Size(i) * n + j] = p_a[Size(i) * n + j] + p_a[Size(j) * n + i];
      if (!restricted) {
        psym_b[Size(i) * n + j] = p_b[Size(i) * n + j] + p_b[Size(j) * n + i];
      }
    }
  }
  std::vector<double> bar_db_ch(Size(chunk) * nn);
  std::vector<double> ao_db_ch(Size(chunk) * nn);
  std::vector<double> ex_term(Size(n) * std::max(1, std::max(oa, ob)));

  for (int q0 = 0; q0 < q; q0 += chunk) {
    const int actual = std::min(chunk, q - q0);
    std::fill(ao_db_ch.begin(), ao_db_ch.begin() + Size(actual) * nn, 0.0);

    auto accum_spin_chunk = [&](const double* C, const double* df_s, const double* p_s,
                                const double* psym, int occ, double scale) {
      for (int al = 0; al < actual; ++al) {
        const int a = q0 + al;
        double* bdb = bar_db_ch.data() + Size(al) * nn;
        const double ch = charge[a];
        for (Size idx = 0; idx < nn; ++idx) bdb[idx] = ch * p_s[idx];
        if (occ > 0) {
          const double bd = bar_dcharge[a];
          for (int i = 0; i < occ; ++i) bdb[Size(i) * n + i] += bd;
          // ex_term[p, i] = sum_q psym[p, q] * df_s[a, q, i]
          const double* sl = df_s + Size(a) * nn;
          rm_gemm_host(false, false, n, occ, n, 1.0, psym, n, sl, n, 0.0, ex_term.data(), occ);
          for (int p = 0; p < n; ++p) {
            for (int i = 0; i < occ; ++i) {
              bdb[Size(p) * n + i] -= ex_term[Size(p) * occ + i];
            }
          }
        }
        // ao_db_ch[al] += scale * (C @ bdb @ C^T)
        rm_gemm_host(false, false, n, n, n, 1.0, C, n, bdb, n, 0.0, tmp1.data(), n);
        rm_gemm_host(false, true, n, n, n, scale, tmp1.data(), n, C, n,
                     1.0, ao_db_ch.data() + Size(al) * nn, n);
      }
      // bar_M[q0:q0+actual, :] += (-0.5 * scale) * bar_db_ch @ df_s^T
      rm_gemm_host(false, true, actual, q, int(nn), -0.5 * scale,
                   bar_db_ch.data(), int(nn), df_s, int(nn),
                   1.0, bar_M.data() + Size(q0) * q, q);
    };

    accum_spin_chunk(C_a, df_a, p_a, psym_a.data(), oa, share_scale);
    if (!restricted) {
      accum_spin_chunk(C_b, df_b, p_b, psym_b.data(), ob, 1.0);
    }

    for (int al = 0; al < actual; ++al) {
      symmetrize_inplace(n, ao_db_ch.data() + Size(al) * nn);
    }
    // bar_dr[T, mn] += sum_{al} W[q0 + al, T] * ao_db_ch[al, mn]
    rm_gemm_host(true, false, q, int(nn), actual, 1.0,
                 whitener + Size(q0) * q, q,
                 ao_db_ch.data(), int(nn),
                 1.0, bar_dr_out, int(nn));
  }

  // 4) bar_dj = sym(W^T @ bar_M @ W)
  std::vector<double> mw(Size(q) * q);
  rm_gemm_host(false, false, q, q, q, 1.0, bar_M.data(), q, whitener, q, 0.0, mw.data(), q);
  rm_gemm_host(true, false, q, q, q, 1.0, whitener, q, mw.data(), q, 0.0, bar_dj_out, q);
  symmetrize_inplace(q, bar_dj_out);

  if (!all_finite(bar_ds_out, nn) || !all_finite(bar_dh_out, nn) ||
      !all_finite(bar_dr_out, Size(q) * nn) || !all_finite(bar_dj_out, Size(q) * q)) {
    return 2;
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

#else  // !GQC_SINK_HOST_REFERENCE (CUDA cuBLAS implementations)

int gqc_f12_cv_df_jk_fock_cuda(
    int q, int n, int oa, int ob, int restricted, int q_chunk, std::uint64_t budget_bytes,
    const double* B_jk, const double* D_tot,
    const double* Co_a, const double* Co_b,
    double* J_out, double* Ka_out, double* Kb_out) {
  try {
  if (q <= 0 || n <= 0 || oa < 0 || ob < 0 || oa > n || ob > n) return 1;
  if (!B_jk || !D_tot || !J_out || !Ka_out) return 1;
  if (oa > 0 && !Co_a) return 1;
  if (!restricted && (!Kb_out || (ob > 0 && !Co_b))) return 1;

  const Size nn = Size(n) * n;
  if (!fits_int(nn)) return 1;

  const int max_occ = restricted ? oa : std::max(oa, ob);
  const Size bytes_per_q = sat_mul(sat_add(sat_add(nn, Size(1)), sat_mul(Size(max_occ), Size(n))), sizeof(double));
  const Size fixed = sat_add(sat_mul(restricted ? 3 : 4, nn),
                             sat_mul(n, Size(oa) + (restricted ? 0 : ob)));
  const Size cap = effective_budget(budget_bytes, kDefaultChunkBudgetBytes);
  const Size available = workspace_after_fixed(cap, fixed);
  if (bytes_per_q > available) return kErrOom;
  Size max_chunk = available / std::max<Size>(1, bytes_per_q);
  max_chunk = std::min<Size>(max_chunk, static_cast<Size>(q));
  if (q_chunk > 0) max_chunk = std::min<Size>(max_chunk, static_cast<Size>(q_chunk));
  while (max_chunk > 1 && max_occ > 0 && !fits_int(max_chunk * max_occ)) {
    max_chunk /= 2;
  }
  if (max_occ > 0 && !fits_int(max_chunk * max_occ)) return 1;
  const int chunk = static_cast<int>(max_chunk);

  CudaStream cs;
  CublasHandle ch;
  if (cs.init() != cudaSuccess || ch.init(cs.stream) != CUBLAS_STATUS_SUCCESS) return 3;
  cudaStream_t s = cs.stream;
  cublasHandle_t h = ch.handle;

  IntBuf d_bad;
  DoubleBuf d_D, d_Coa, d_Cob, d_J, d_Ka, d_Kb;
  if (d_bad.alloc(2) != cudaSuccess || d_bad.zero_async(2, s) != cudaSuccess ||
      d_D.alloc(nn) != cudaSuccess || d_J.alloc(nn) != cudaSuccess ||
      d_Ka.alloc(nn) != cudaSuccess ||
      (oa > 0 && d_Coa.alloc(Size(n) * oa) != cudaSuccess) ||
      (!restricted && d_Kb.alloc(nn) != cudaSuccess) ||
      (!restricted && ob > 0 && d_Cob.alloc(Size(n) * ob) != cudaSuccess)) {
    return 4;
  }

  if (d_J.zero_async(nn, s) != cudaSuccess || d_Ka.zero_async(nn, s) != cudaSuccess ||
      (!restricted && d_Kb.zero_async(nn, s) != cudaSuccess) ||
      !h2d_async(d_D.ptr, D_tot, nn, s) ||
      (oa > 0 && !h2d_async(d_Coa.ptr, Co_a, Size(n) * oa, s)) ||
      (!restricted && ob > 0 && !h2d_async(d_Cob.ptr, Co_b, Size(n) * ob, s))) {
    return 5;
  }
  if (!check_finite_gpu(d_D.ptr, nn, d_bad.ptr, s) ||
      (oa > 0 && !check_finite_gpu(d_Coa.ptr, Size(n) * oa, d_bad.ptr, s)) ||
      (!restricted && ob > 0 && !check_finite_gpu(d_Cob.ptr, Size(n) * ob, d_bad.ptr, s))) {
    return 6;
  }

  DoubleBuf d_Bch, d_rho, d_Y;
  if (d_Bch.alloc(Size(chunk) * nn) != cudaSuccess ||
      d_rho.alloc(static_cast<Size>(chunk)) != cudaSuccess ||
      (max_occ > 0 && d_Y.alloc(Size(chunk) * max_occ * n) != cudaSuccess)) {
    return 4;
  }

  for (int q0 = 0; q0 < q; q0 += chunk) {
    const int actual = std::min(chunk, q - q0);
    if (!h2d_async(d_Bch.ptr, B_jk + Size(q0) * nn, Size(actual) * nn, s) ||
        !check_finite_gpu(d_Bch.ptr, Size(actual) * nn, d_bad.ptr, s)) {
      return 5;
    }
    // rho[a] = sum_{pq} B_ch[a, pq] * D_tot[pq]
    if (rm_gemm(h, false, false, actual, 1, static_cast<int>(nn), 1.0,
                d_Bch.ptr, static_cast<int>(nn), d_D.ptr, 1,
                0.0, d_rho.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
    // J[pq] += sum_a rho[a] * B_ch[a, pq]
    if (rm_gemm(h, true, false, 1, static_cast<int>(nn), actual, 1.0,
                d_rho.ptr, 1, d_Bch.ptr, static_cast<int>(nn),
                1.0, d_J.ptr, static_cast<int>(nn)) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
    // Exchange K_a: Y_a[a, oa, n] = Co_a^T @ B_a^T, then K_a += Y_a^T @ Y_a in ONE giant DGEMM!
    if (oa > 0) {
      if (rm_gemm_batched(h, true, true, oa, n, n, 1.0,
                          d_Coa.ptr, oa, 0,
                          d_Bch.ptr, n, static_cast<long long>(nn),
                          0.0, d_Y.ptr, n, static_cast<long long>(Size(oa) * n),
                          actual) != CUBLAS_STATUS_SUCCESS) {
        return 7;
      }
      if (rm_gemm(h, true, false, n, n, actual * oa, 1.0,
                  d_Y.ptr, n, d_Y.ptr, n,
                  1.0, d_Ka.ptr, n) != CUBLAS_STATUS_SUCCESS) {
        return 7;
      }
    }
    if (!restricted && ob > 0) {
      if (rm_gemm_batched(h, true, true, ob, n, n, 1.0,
                          d_Cob.ptr, ob, 0,
                          d_Bch.ptr, n, static_cast<long long>(nn),
                          0.0, d_Y.ptr, n, static_cast<long long>(Size(ob) * n),
                          actual) != CUBLAS_STATUS_SUCCESS) {
        return 7;
      }
      if (rm_gemm(h, true, false, n, n, actual * ob, 1.0,
                  d_Y.ptr, n, d_Y.ptr, n,
                  1.0, d_Kb.ptr, n) != CUBLAS_STATUS_SUCCESS) {
        return 7;
      }
    }
  }

  if (!check_finite_gpu(d_J.ptr, nn, d_bad.ptr + 1, s) ||
      !check_finite_gpu(d_Ka.ptr, nn, d_bad.ptr + 1, s) ||
      (!restricted && !check_finite_gpu(d_Kb.ptr, nn, d_bad.ptr + 1, s))) {
    return 6;
  }

  int h_bad[2] = {0, 0};
  if (!d2h_async(J_out, d_J.ptr, nn, s) || !d2h_async(Ka_out, d_Ka.ptr, nn, s) ||
      (!restricted && !d2h_async(Kb_out, d_Kb.ptr, nn, s)) ||
      !d2h_async(h_bad, d_bad.ptr, 2, s) || cudaStreamSynchronize(s) != cudaSuccess) {
    return 8;
  }
  if (h_bad[0] != 0) return 1;
  if (h_bad[1] != 0) return 2;
  return 0;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_cv_mp2_factors_fwd_cuda(
    int q, int n, int o_act, int n_vir, int q_chunk, std::uint64_t budget_bytes,
    const double* ao_coulomb, const double* whitener,
    const double* co, const double* cv, double* B_out) {
  try {
  if (q <= 0 || n <= 0 || o_act <= 0 || n_vir <= 0) return 1;
  if (!ao_coulomb || !whitener || !co || !cv || !B_out) return 1;

  const Size nn = Size(n) * n;
  const Size ov = Size(o_act) * n_vir;
  if (!fits_int(ov)) return 1;

  const Size no = Size(n) * o_act;
  const Size nv = Size(n) * n_vir;
  const Size qov = Size(q) * ov;
  // W, both coefficient matrices and B remain resident throughout the AO/MO transform.
  const Size fixed = sat_add(sat_add(Size(q) * q, qov), sat_add(no, nv));
  const Size cap = effective_budget(budget_bytes, kDefaultChunkBudgetBytes);
  const Size room = workspace_after_fixed(cap, fixed);
  const Size bytes_per_q = sat_mul(sat_add(nn, sat_add(no, ov)), sizeof(double));
  const int chunk = static_cast<int>(std::min({
      room / bytes_per_q, Size(q_chunk > 0 ? std::min(q, q_chunk) : q),
      Size(std::numeric_limits<int>::max() / o_act)}));
  if (chunk == 0) return kErrOom;

  CudaStream cs;
  CublasHandle ch;
  if (cs.init() != cudaSuccess || ch.init(cs.stream) != CUBLAS_STATUS_SUCCESS) return 3;
  cudaStream_t s = cs.stream;
  cublasHandle_t h = ch.handle;

  IntBuf d_bad;
  DoubleBuf d_W, d_co, d_cv, d_B;
  if (d_bad.alloc(2) != cudaSuccess || d_bad.zero_async(2, s) != cudaSuccess ||
      d_W.alloc(Size(q) * q) != cudaSuccess || d_co.alloc(Size(n) * o_act) != cudaSuccess ||
      d_cv.alloc(Size(n) * n_vir) != cudaSuccess || d_B.alloc(Size(q) * ov) != cudaSuccess) {
    return 4;
  }
  if (!h2d_async(d_W.ptr, whitener, Size(q) * q, s) ||
      !h2d_async(d_co.ptr, co, Size(n) * o_act, s) ||
      !h2d_async(d_cv.ptr, cv, Size(n) * n_vir, s) ||
      !check_finite_gpu(d_W.ptr, Size(q) * q, d_bad.ptr, s) ||
      !check_finite_gpu(d_co.ptr, Size(n) * o_act, d_bad.ptr, s) ||
      !check_finite_gpu(d_cv.ptr, Size(n) * n_vir, d_bad.ptr, s)) {
    return 5;
  }


  DoubleBuf d_ao_ch, d_tmp, d_mo_ch;
  if (d_ao_ch.alloc(Size(chunk) * nn) != cudaSuccess ||
      d_tmp.alloc(Size(chunk) * o_act * n) != cudaSuccess ||
      d_mo_ch.alloc(Size(chunk) * ov) != cudaSuccess) {
    return 4;
  }

  for (int q0 = 0; q0 < q; q0 += chunk) {
    const int actual = std::min(chunk, q - q0);
    if (!h2d_async(d_ao_ch.ptr, ao_coulomb + Size(q0) * nn, Size(actual) * nn, s) ||
        !check_finite_gpu(d_ao_ch.ptr, Size(actual) * nn, d_bad.ptr, s)) {
      return 5;
    }
    // tmp[a, o_act, n] = co^T @ ao_ch[a, n, n]
    if (rm_gemm_batched(h, true, false, o_act, n, n, 1.0,
                        d_co.ptr, o_act, 0,
                        d_ao_ch.ptr, n, static_cast<long long>(nn),
                        0.0, d_tmp.ptr, n, static_cast<long long>(Size(o_act) * n),
                        actual) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
    // mo_ch[actual*o_act, n_vir] = tmp[actual*o_act, n] @ cv[n, n_vir]
    if (rm_gemm(h, false, false, actual * o_act, n_vir, n, 1.0,
                d_tmp.ptr, n, d_cv.ptr, n_vir,
                0.0, d_mo_ch.ptr, n_vir) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
    // Accumulate whitened B[q, ov] += W[:, q0:q1] @ mo_ch[actual, ov]
    if (rm_gemm(h, false, false, q, static_cast<int>(ov), actual, 1.0,
                d_W.ptr + q0, q, d_mo_ch.ptr, static_cast<int>(ov),
                (q0 == 0 ? 0.0 : 1.0), d_B.ptr, static_cast<int>(ov)) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
  }

  if (!check_finite_gpu(d_B.ptr, Size(q) * ov, d_bad.ptr + 1, s)) return 6;

  int h_bad[2] = {0, 0};
  if (!d2h_async(B_out, d_B.ptr, Size(q) * ov, s) ||
      !d2h_async(h_bad, d_bad.ptr, 2, s) || cudaStreamSynchronize(s) != cudaSuccess) {
    return 8;
  }
  if (h_bad[0] != 0) return 1;
  if (h_bad[1] != 0) return 2;
  return 0;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_cv_mp2_factors_rev_cuda(
    int q, int n, int o_act, int n_vir, int q_chunk, std::uint64_t budget_bytes,
    const double* ao_coulomb, const double* whitener,
    const double* co, const double* cv,
    const double* B, const double* bar_B,
    double* bar_co_out, double* bar_cv_out,
    double* bar_ao_out, double* metric_bar_out) {
  try {
  if (q <= 0 || n <= 0 || o_act <= 0 || n_vir <= 0) return 1;
  if (!ao_coulomb || !whitener || !co || !cv || !B || !bar_B) return 1;
  if (!bar_co_out || !bar_cv_out || !bar_ao_out || !metric_bar_out) return 1;

  const Size nn = Size(n) * n;
  const Size ov = Size(o_act) * n_vir;
  if (!fits_int(ov) || !fits_int(Size(n) * o_act)) return 1;

  const Size no = Size(n) * o_act;
  const Size nv = Size(n) * n_vir;
  const Size qov = Size(q) * ov;
  const Size qq = Size(q) * q;
  const Size coefficients = sat_add(no, nv);
  const Size common = sat_add(qov, coefficients);  // bar_mo, co and cv in both stages.
  const Size cap = effective_budget(budget_bytes, kDefaultChunkBudgetBytes);
  const Size metric_extra = sat_mul(sat_add(qq, qov), 2 * sizeof(double));
  if (metric_extra > workspace_after_fixed(cap, common)) return kErrOom;

  // After the metric stage, both coefficient cotangents live with two AO blocks,
  // two half-transformed blocks and one reduction weight per auxiliary row.
  const Size room = workspace_after_fixed(cap, sat_add(common, coefficients));
  const Size bytes_per_q = sat_mul(sat_add(sat_mul(2, sat_add(nn, no)), 1), sizeof(double));
  const int chunk = static_cast<int>(std::min({
      room / bytes_per_q, Size(q_chunk > 0 ? std::min(q, q_chunk) : q),
      Size(std::numeric_limits<int>::max() / o_act)}));
  if (chunk == 0) return kErrOom;

  CudaStream cs;
  CublasHandle ch;
  if (cs.init() != cudaSuccess || ch.init(cs.stream) != CUBLAS_STATUS_SUCCESS) return 3;
  cudaStream_t s = cs.stream;
  cublasHandle_t h = ch.handle;

  IntBuf d_bad;
  DoubleBuf d_bar_mo, d_co, d_cv;
  if (d_bad.alloc(2) != cudaSuccess || d_bad.zero_async(2, s) != cudaSuccess ||
      d_bar_mo.alloc(Size(q) * ov) != cudaSuccess ||
      d_co.alloc(Size(n) * o_act) != cudaSuccess ||
      d_cv.alloc(Size(n) * n_vir) != cudaSuccess) {
    return 4;
  }

  // Stage 1: Whitener pullback & metric_bar (scoped so B/bar_B/W are freed before Stage 2).
  {
    DoubleBuf d_W, d_B, d_bar_B, d_metric_bar;
    if (d_W.alloc(Size(q) * q) != cudaSuccess || d_B.alloc(Size(q) * ov) != cudaSuccess ||
        d_bar_B.alloc(Size(q) * ov) != cudaSuccess ||
        d_metric_bar.alloc(Size(q) * q) != cudaSuccess) {
      return 4;
    }
    if (!h2d_async(d_W.ptr, whitener, Size(q) * q, s) ||
        !h2d_async(d_B.ptr, B, Size(q) * ov, s) ||
        !h2d_async(d_bar_B.ptr, bar_B, Size(q) * ov, s) ||
        !h2d_async(d_co.ptr, co, Size(n) * o_act, s) ||
        !h2d_async(d_cv.ptr, cv, Size(n) * n_vir, s) ||
        !check_finite_gpu(d_W.ptr, Size(q) * q, d_bad.ptr, s) ||
        !check_finite_gpu(d_B.ptr, Size(q) * ov, d_bad.ptr, s) ||
        !check_finite_gpu(d_bar_B.ptr, Size(q) * ov, d_bad.ptr, s) ||
        !check_finite_gpu(d_co.ptr, Size(n) * o_act, d_bad.ptr, s) ||
        !check_finite_gpu(d_cv.ptr, Size(n) * n_vir, d_bad.ptr, s)) {
      return 5;
    }
    // bar_mo_raw = W^T @ bar_B
    if (rm_gemm(h, true, false, q, static_cast<int>(ov), q, 1.0,
                d_W.ptr, q, d_bar_B.ptr, static_cast<int>(ov),
                0.0, d_bar_mo.ptr, static_cast<int>(ov)) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
    // Reuse d_bar_B as fitted = W^T @ B
    if (rm_gemm(h, true, false, q, static_cast<int>(ov), q, 1.0,
                d_W.ptr, q, d_B.ptr, static_cast<int>(ov),
                0.0, d_bar_B.ptr, static_cast<int>(ov)) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
    // metric_bar = sym(-0.5 * bar_mo_raw @ fitted^T) = -0.25 * (X + X^T)
    if (rm_gemm(h, false, true, q, q, static_cast<int>(ov), -0.5,
                d_bar_mo.ptr, static_cast<int>(ov), d_bar_B.ptr, static_cast<int>(ov),
                0.0, d_metric_bar.ptr, q) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
    if (!symmetrize_batched_gpu(1, q, d_metric_bar.ptr, s) ||
        !check_finite_gpu(d_metric_bar.ptr, Size(q) * q, d_bad.ptr + 1, s) ||
        !d2h_async(metric_bar_out, d_metric_bar.ptr, Size(q) * q, s) ||
        cudaStreamSynchronize(s) != cudaSuccess) {
      return 8;
    }
  }

  // Stage 2: Q-chunked AO/MO back-projection with zero inner-loop allocations.

  DoubleBuf d_bar_co, d_bar_cv, d_ao_ch, d_bar_ao_ch, d_t, d_bar_t, d_ones;
  if (d_bar_co.alloc(Size(n) * o_act) != cudaSuccess ||
      d_bar_cv.alloc(Size(n) * n_vir) != cudaSuccess ||
      d_ao_ch.alloc(Size(chunk) * nn) != cudaSuccess ||
      d_bar_ao_ch.alloc(Size(chunk) * nn) != cudaSuccess ||
      d_t.alloc(Size(chunk) * o_act * n) != cudaSuccess ||
      d_bar_t.alloc(Size(chunk) * o_act * n) != cudaSuccess ||
      d_ones.alloc(static_cast<Size>(chunk)) != cudaSuccess) {
    return 4;
  }
  std::vector<double> ones(chunk, 1.0);
  if (d_bar_co.zero_async(Size(n) * o_act, s) != cudaSuccess ||
      d_bar_cv.zero_async(Size(n) * n_vir, s) != cudaSuccess ||
      !h2d_async(d_ones.ptr, ones.data(), static_cast<Size>(chunk), s)) {
    return 5;
  }

  for (int q0 = 0; q0 < q; q0 += chunk) {
    const int actual = std::min(chunk, q - q0);
    const double* bm_ch = d_bar_mo.ptr + Size(q0) * ov;
    if (!h2d_async(d_ao_ch.ptr, ao_coulomb + Size(q0) * nn, Size(actual) * nn, s) ||
        !check_finite_gpu(d_ao_ch.ptr, Size(actual) * nn, d_bad.ptr, s)) {
      return 5;
    }
    // 1) t_slice[a, o_act, n] = co^T @ ao_ch[a]
    if (rm_gemm_batched(h, true, false, o_act, n, n, 1.0,
                        d_co.ptr, o_act, 0,
                        d_ao_ch.ptr, n, static_cast<long long>(nn),
                        0.0, d_t.ptr, n, static_cast<long long>(Size(o_act) * n),
                        actual) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
    // 2) bar_cv[n, n_vir] += t_slice^T @ bm_ch (flattened across chunk!)
    if (rm_gemm(h, true, false, n, n_vir, actual * o_act, 1.0,
                d_t.ptr, n, bm_ch, n_vir,
                1.0, d_bar_cv.ptr, n_vir) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
    // 3) bar_t[actual*o_act, n] = bm_ch @ cv^T (flattened across chunk!)
    if (rm_gemm(h, false, true, actual * o_act, n, n_vir, 1.0,
                bm_ch, n_vir, d_cv.ptr, n_vir,
                0.0, d_bar_t.ptr, n) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
    // 4) bar_co += sum_a ao_a @ bar_t[a]^T (reusing dead d_t as [actual, n, o_act]!)
    if (rm_gemm_batched(h, false, true, n, o_act, n, 1.0,
                        d_ao_ch.ptr, n, static_cast<long long>(nn),
                        d_bar_t.ptr, n, static_cast<long long>(Size(o_act) * n),
                        0.0, d_t.ptr, o_act, static_cast<long long>(Size(n) * o_act),
                        actual) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
    if (rm_gemm(h, false, false, 1, n * o_act, actual, 1.0,
                d_ones.ptr, actual, d_t.ptr, n * o_act,
                1.0, d_bar_co.ptr, n * o_act) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
    // 5) bar_ao[a, n, n] = co @ bar_t[a]
    if (rm_gemm_batched(h, false, false, n, n, o_act, 1.0,
                        d_co.ptr, o_act, 0,
                        d_bar_t.ptr, n, static_cast<long long>(Size(o_act) * n),
                        0.0, d_bar_ao_ch.ptr, n, static_cast<long long>(nn),
                        actual) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
    if (!check_finite_gpu(d_bar_ao_ch.ptr, Size(actual) * nn, d_bad.ptr + 1, s) ||
        !d2h_async(bar_ao_out + Size(q0) * nn, d_bar_ao_ch.ptr, Size(actual) * nn, s)) {
      return 8;
    }
  }

  if (!check_finite_gpu(d_bar_co.ptr, Size(n) * o_act, d_bad.ptr + 1, s) ||
      !check_finite_gpu(d_bar_cv.ptr, Size(n) * n_vir, d_bad.ptr + 1, s)) {
    return 6;
  }

  int h_bad[2] = {0, 0};
  if (!d2h_async(bar_co_out, d_bar_co.ptr, Size(n) * o_act, s) ||
      !d2h_async(bar_cv_out, d_bar_cv.ptr, Size(n) * n_vir, s) ||
      !d2h_async(h_bad, d_bad.ptr, 2, s) || cudaStreamSynchronize(s) != cudaSuccess) {
    return 8;
  }
  if (h_bad[0] != 0) return 1;
  if (h_bad[1] != 0) return 2;
  return 0;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_cv_mp2_pairs_cuda(
    int restricted, int need_cotangents, int q,
    int oa, int ob, int va, int vb, int sh_a, int sh_b, int tile_limit,
    std::uint64_t budget_bytes,
    const double* B_a, const double* B_b,
    const double* eo_a, const double* eo_b,
    const double* ev_a, const double* ev_b,
    double* cv_spin_e_out, double* fc_spin_e_out,
    double* bar_B_a_out, double* bar_B_b_out,
    double* bar_Fo_a_out, double* bar_Fo_b_out,
    double* bar_Fv_a_out, double* bar_Fv_b_out) {
  try {
  if (q <= 0 || oa < 0 || ob < 0 || va < 0 || vb < 0) return 1;
  if (sh_a < 0 || sh_a > oa || sh_b < 0 || sh_b > ob) return 1;
  if (!cv_spin_e_out || !fc_spin_e_out) return 1;
  if (oa > 0 && va > 0 && (!B_a || !eo_a || !ev_a)) return 1;
  if (!restricted && ob > 0 && vb > 0 && (!B_b || !eo_b || !ev_b)) return 1;
  if (restricted && (oa != ob || va != vb || sh_a != sh_b)) return 1;

  const bool has_a = (oa > 0 && va > 0);
  const bool has_b = (!restricted && ob > 0 && vb > 0);
  if (need_cotangents) {
    if (has_a && (!bar_B_a_out || !bar_Fo_a_out || !bar_Fv_a_out)) return 1;
    if (has_b && (!bar_B_b_out || !bar_Fo_b_out || !bar_Fv_b_out)) return 1;
  }

  cv_spin_e_out[0] = cv_spin_e_out[1] = cv_spin_e_out[2] = 0.0;
  fc_spin_e_out[0] = fc_spin_e_out[1] = fc_spin_e_out[2] = 0.0;
  if (!has_a && !has_b) return 0;

  // Fixed factors and cotangents coexist with each sequential spin channel.
  // d_B*_raw doubles as the factor cotangent, so count that buffer only once.
  Size fixed = sat_add(restricted ? 4 : 6, pair_factor_doubles(q, oa, va, need_cotangents));
  if (has_b) fixed = sat_add(fixed, pair_factor_doubles(q, ob, vb, need_cotangents));
  const Size cap = effective_budget(budget_bytes, kDefaultPairBudgetBytes);
  const Size scratch_bytes = workspace_after_fixed(cap, fixed);
  if (scratch_bytes == 0) return kErrOom;
  if (restricted) {
    if (!check_rhf_pair_workspace_budget(scratch_bytes, need_cotangents != 0, oa, va)) return kErrOom;
  } else {
    if (has_a && !check_uhf_pair_workspace_budget(scratch_bytes, true, need_cotangents != 0, oa, oa, va, va, false)) {
      return kErrOom;
    }
    if (has_a && has_b &&
        !check_uhf_pair_workspace_budget(scratch_bytes, false, need_cotangents != 0, oa, ob, va, vb, true)) {
      return kErrOom;
    }
    if (has_b && !check_uhf_pair_workspace_budget(scratch_bytes, true, need_cotangents != 0, ob, ob, vb, vb, false)) {
      return kErrOom;
    }
  }

  CudaStream cs;
  CublasHandle ch;
  if (cs.init() != cudaSuccess || ch.init(cs.stream) != CUBLAS_STATUS_SUCCESS) return 3;
  cudaStream_t s = cs.stream;
  cublasHandle_t h = ch.handle;

  IntBuf d_bad;
  if (d_bad.alloc(2) != cudaSuccess || d_bad.zero_async(2, s) != cudaSuccess) return 4;

  if (restricted) {
    const Size ov = Size(oa) * va;
    if (!fits_int(ov)) return 1;
    DoubleBuf d_Ba_raw, d_Ba_packed, d_eoa, d_eva, d_e_acc;
    DoubleBuf d_bar_Fo, d_bar_Fv;
    if (d_Ba_raw.alloc(Size(q) * ov) != cudaSuccess ||
        d_Ba_packed.alloc(Size(q) * ov) != cudaSuccess ||
        d_eoa.alloc(static_cast<Size>(oa)) != cudaSuccess ||
        d_eva.alloc(static_cast<Size>(va)) != cudaSuccess ||
        d_e_acc.alloc(4) != cudaSuccess || d_e_acc.zero_async(4, s) != cudaSuccess) {
      return 4;
    }
    if (!h2d_async(d_Ba_raw.ptr, B_a, Size(q) * ov, s) ||
        !h2d_async(d_eoa.ptr, eo_a, static_cast<Size>(oa), s) ||
        !h2d_async(d_eva.ptr, ev_a, static_cast<Size>(va), s) ||
        !check_finite_gpu(d_Ba_raw.ptr, Size(q) * ov, d_bad.ptr, s) ||
        !check_finite_gpu(d_eoa.ptr, static_cast<Size>(oa), d_bad.ptr, s) ||
        !check_finite_gpu(d_eva.ptr, static_cast<Size>(va), d_bad.ptr, s)) {
      return 5;
    }
    if (!transpose_2d_gpu(q, static_cast<int>(ov), d_Ba_raw.ptr, d_Ba_packed.ptr, s)) return 6;

    // Reuse d_Ba_raw as d_bar_B when cotangents are needed!
    double* d_bar_B_ptr = nullptr;
    if (need_cotangents) {
      d_bar_B_ptr = d_Ba_raw.ptr;
      if (d_Ba_raw.zero_async(Size(q) * ov, s) != cudaSuccess ||
          d_bar_Fo.alloc(Size(oa) * oa) != cudaSuccess ||
          d_bar_Fv.alloc(Size(va) * va) != cudaSuccess ||
          d_bar_Fo.zero_async(Size(oa) * oa, s) != cudaSuccess ||
          d_bar_Fv.zero_async(Size(va) * va, s) != cudaSuccess) {
        return 4;
      }
    }

    int acc_rc = 7;
    if (!accumulate_rhf_spatial_gpu(h, s, need_cotangents != 0, q, oa, va, sh_a, tile_limit, scratch_bytes,
                                    d_Ba_packed.ptr, d_eoa.ptr, d_eva.ptr, d_e_acc.ptr,
                                    d_bar_B_ptr, d_bar_Fo.ptr, d_bar_Fv.ptr, &acc_rc)) {
      return acc_rc;
    }

    if (!check_finite_gpu(d_e_acc.ptr, 4, d_bad.ptr + 1, s)) return 6;
    if (need_cotangents) {
      if (!check_finite_gpu(d_bar_B_ptr, Size(q) * ov, d_bad.ptr + 1, s) ||
          !check_finite_gpu(d_bar_Fo.ptr, Size(oa) * oa, d_bad.ptr + 1, s) ||
          !check_finite_gpu(d_bar_Fv.ptr, Size(va) * va, d_bad.ptr + 1, s)) {
        return 6;
      }
    }

    double h_e[4] = {0.0, 0.0, 0.0, 0.0};
    int h_bad[2] = {0, 0};
    if (!d2h_async(h_e, d_e_acc.ptr, 4, s) ||
        (need_cotangents && (!d2h_async(bar_B_a_out, d_bar_B_ptr, Size(q) * ov, s) ||
                             !d2h_async(bar_Fo_a_out, d_bar_Fo.ptr, Size(oa) * oa, s) ||
                             !d2h_async(bar_Fv_a_out, d_bar_Fv.ptr, Size(va) * va, s))) ||
        !d2h_async(h_bad, d_bad.ptr, 2, s) || cudaStreamSynchronize(s) != cudaSuccess) {
      return 8;
    }
    if (h_bad[0] != 0) return 1;
    if (h_bad[1] != 0) return 2;

    cv_spin_e_out[0] = h_e[0];
    cv_spin_e_out[1] = h_e[1];
    cv_spin_e_out[2] = h_e[0];
    fc_spin_e_out[0] = h_e[2];
    fc_spin_e_out[1] = h_e[3];
    fc_spin_e_out[2] = h_e[2];
    return 0;
  }

  // General UHF path (aa, ab, bb)
  const Size ova = Size(oa) * va;
  const Size ovb = Size(ob) * vb;
  if ((has_a && !fits_int(ova)) || (has_b && !fits_int(ovb))) return 1;

  DoubleBuf d_Ba_raw, d_Ba_packed, d_eoa, d_eva, d_bar_Fo_a, d_bar_Fv_a;
  DoubleBuf d_Bb_raw, d_Bb_packed, d_eob, d_evb, d_bar_Fo_b, d_bar_Fv_b;
  DoubleBuf d_e_acc;
  if (d_e_acc.alloc(6) != cudaSuccess || d_e_acc.zero_async(6, s) != cudaSuccess) return 4;

  double* d_bar_Ba_ptr = nullptr;
  double* d_bar_Bb_ptr = nullptr;

  if (has_a) {
    if (d_Ba_raw.alloc(Size(q) * ova) != cudaSuccess ||
        d_Ba_packed.alloc(Size(q) * ova) != cudaSuccess ||
        d_eoa.alloc(static_cast<Size>(oa)) != cudaSuccess ||
        d_eva.alloc(static_cast<Size>(va)) != cudaSuccess) {
      return 4;
    }
    if (!h2d_async(d_Ba_raw.ptr, B_a, Size(q) * ova, s) ||
        !h2d_async(d_eoa.ptr, eo_a, static_cast<Size>(oa), s) ||
        !h2d_async(d_eva.ptr, ev_a, static_cast<Size>(va), s) ||
        !check_finite_gpu(d_Ba_raw.ptr, Size(q) * ova, d_bad.ptr, s) ||
        !check_finite_gpu(d_eoa.ptr, static_cast<Size>(oa), d_bad.ptr, s) ||
        !check_finite_gpu(d_eva.ptr, static_cast<Size>(va), d_bad.ptr, s)) {
      return 5;
    }
    if (!transpose_2d_gpu(q, static_cast<int>(ova), d_Ba_raw.ptr, d_Ba_packed.ptr, s)) return 6;
    if (need_cotangents) {
      d_bar_Ba_ptr = d_Ba_raw.ptr;
      if (d_Ba_raw.zero_async(Size(q) * ova, s) != cudaSuccess ||
          d_bar_Fo_a.alloc(Size(oa) * oa) != cudaSuccess ||
          d_bar_Fv_a.alloc(Size(va) * va) != cudaSuccess ||
          d_bar_Fo_a.zero_async(Size(oa) * oa, s) != cudaSuccess ||
          d_bar_Fv_a.zero_async(Size(va) * va, s) != cudaSuccess) {
        return 4;
      }
    }
  }

  if (has_b) {
    if (d_Bb_raw.alloc(Size(q) * ovb) != cudaSuccess ||
        d_Bb_packed.alloc(Size(q) * ovb) != cudaSuccess ||
        d_eob.alloc(static_cast<Size>(ob)) != cudaSuccess ||
        d_evb.alloc(static_cast<Size>(vb)) != cudaSuccess) {
      return 4;
    }
    if (!h2d_async(d_Bb_raw.ptr, B_b, Size(q) * ovb, s) ||
        !h2d_async(d_eob.ptr, eo_b, static_cast<Size>(ob), s) ||
        !h2d_async(d_evb.ptr, ev_b, static_cast<Size>(vb), s) ||
        !check_finite_gpu(d_Bb_raw.ptr, Size(q) * ovb, d_bad.ptr, s) ||
        !check_finite_gpu(d_eob.ptr, static_cast<Size>(ob), d_bad.ptr, s) ||
        !check_finite_gpu(d_evb.ptr, static_cast<Size>(vb), d_bad.ptr, s)) {
      return 5;
    }
    if (!transpose_2d_gpu(q, static_cast<int>(ovb), d_Bb_raw.ptr, d_Bb_packed.ptr, s)) return 6;
    if (need_cotangents) {
      d_bar_Bb_ptr = d_Bb_raw.ptr;
      if (d_Bb_raw.zero_async(Size(q) * ovb, s) != cudaSuccess ||
          d_bar_Fo_b.alloc(Size(ob) * ob) != cudaSuccess ||
          d_bar_Fv_b.alloc(Size(vb) * vb) != cudaSuccess ||
          d_bar_Fo_b.zero_async(Size(ob) * ob, s) != cudaSuccess ||
          d_bar_Fv_b.zero_async(Size(vb) * vb, s) != cudaSuccess) {
        return 4;
      }
    }
  }

  // aa
  if (has_a) {
    int acc_rc = 7;
    if (!accumulate_uhf_channel_gpu(h, s, true, need_cotangents != 0, q, oa, oa, va, va,
                                    sh_a, sh_a, tile_limit, scratch_bytes,
                                    d_Ba_packed.ptr, d_Ba_packed.ptr,
                                    d_eoa.ptr, d_eoa.ptr, d_eva.ptr, d_eva.ptr,
                                    d_e_acc.ptr,
                                    d_bar_Ba_ptr, d_bar_Ba_ptr,
                                    d_bar_Fo_a.ptr, nullptr,
                                    d_bar_Fv_a.ptr, d_bar_Fv_a.ptr, &acc_rc)) {
      return acc_rc;
    }
  }
  // ab
  if (has_a && has_b) {
    int acc_rc = 7;
    if (!accumulate_uhf_channel_gpu(h, s, false, need_cotangents != 0, q, oa, ob, va, vb,
                                    sh_a, sh_b, tile_limit, scratch_bytes,
                                    d_Ba_packed.ptr, d_Bb_packed.ptr,
                                    d_eoa.ptr, d_eob.ptr, d_eva.ptr, d_evb.ptr,
                                    d_e_acc.ptr + 2,
                                    d_bar_Ba_ptr, d_bar_Bb_ptr,
                                    d_bar_Fo_a.ptr, d_bar_Fo_b.ptr,
                                    d_bar_Fv_a.ptr, d_bar_Fv_b.ptr, &acc_rc)) {
      return acc_rc;
    }
  }
  // bb
  if (has_b) {
    int acc_rc = 7;
    if (!accumulate_uhf_channel_gpu(h, s, true, need_cotangents != 0, q, ob, ob, vb, vb,
                                    sh_b, sh_b, tile_limit, scratch_bytes,
                                    d_Bb_packed.ptr, d_Bb_packed.ptr,
                                    d_eob.ptr, d_eob.ptr, d_evb.ptr, d_evb.ptr,
                                    d_e_acc.ptr + 4,
                                    d_bar_Bb_ptr, d_bar_Bb_ptr,
                                    d_bar_Fo_b.ptr, nullptr,
                                    d_bar_Fv_b.ptr, d_bar_Fv_b.ptr, &acc_rc)) {
      return acc_rc;
    }
  }

  if (!check_finite_gpu(d_e_acc.ptr, 6, d_bad.ptr + 1, s)) return 6;
  if (need_cotangents) {
    if (has_a && (!check_finite_gpu(d_bar_Ba_ptr, Size(q) * ova, d_bad.ptr + 1, s) ||
                  !check_finite_gpu(d_bar_Fo_a.ptr, Size(oa) * oa, d_bad.ptr + 1, s) ||
                  !check_finite_gpu(d_bar_Fv_a.ptr, Size(va) * va, d_bad.ptr + 1, s))) {
      return 6;
    }
    if (has_b && (!check_finite_gpu(d_bar_Bb_ptr, Size(q) * ovb, d_bad.ptr + 1, s) ||
                  !check_finite_gpu(d_bar_Fo_b.ptr, Size(ob) * ob, d_bad.ptr + 1, s) ||
                  !check_finite_gpu(d_bar_Fv_b.ptr, Size(vb) * vb, d_bad.ptr + 1, s))) {
      return 6;
    }
  }

  double h_e[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int h_bad[2] = {0, 0};
  if (!d2h_async(h_e, d_e_acc.ptr, 6, s) ||
      (need_cotangents && has_a &&
       (!d2h_async(bar_B_a_out, d_bar_Ba_ptr, Size(q) * ova, s) ||
        !d2h_async(bar_Fo_a_out, d_bar_Fo_a.ptr, Size(oa) * oa, s) ||
        !d2h_async(bar_Fv_a_out, d_bar_Fv_a.ptr, Size(va) * va, s))) ||
      (need_cotangents && has_b &&
       (!d2h_async(bar_B_b_out, d_bar_Bb_ptr, Size(q) * ovb, s) ||
        !d2h_async(bar_Fo_b_out, d_bar_Fo_b.ptr, Size(ob) * ob, s) ||
        !d2h_async(bar_Fv_b_out, d_bar_Fv_b.ptr, Size(vb) * vb, s))) ||
      !d2h_async(h_bad, d_bad.ptr, 2, s) || cudaStreamSynchronize(s) != cudaSuccess) {
    return 8;
  }
  if (h_bad[0] != 0) return 1;
  if (h_bad[1] != 0) return 2;

  cv_spin_e_out[0] = h_e[0];
  fc_spin_e_out[0] = h_e[1];
  cv_spin_e_out[1] = h_e[2];
  fc_spin_e_out[1] = h_e[3];
  cv_spin_e_out[2] = h_e[4];
  fc_spin_e_out[2] = h_e[5];
  return 0;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_cv_jk_mo_fock_fwd_cuda(
    int q, int n, int oa, int ob, int restricted, int q_chunk, std::uint64_t budget_bytes,
    const double* ao_coulomb, const double* hcore_ao, const double* whitener,
    const double* C_a, const double* C_b,
    double* df_a_out, double* df_b_out,
    double* fock_a_out, double* fock_b_out) {
  try {
  if (q <= 0 || n <= 0 || oa < 0 || ob < 0 || oa >= n || ob >= n || (oa + ob) <= 0) return 1;
  if (!ao_coulomb || !hcore_ao || !whitener || !C_a || !df_a_out || !fock_a_out) return 1;
  if (!restricted && (!C_b || !df_b_out || !fock_b_out)) return 1;

  const Size nn = Size(n) * n;
  if (!fits_int(nn)) return 1;

  const int max_occ = restricted ? oa : std::max(oa, ob);
  const auto plan = jk_mo_workspace(false, q, n, max_occ, restricted != 0, q_chunk, budget_bytes);
  const int chunk = plan.chunk;
  if (!chunk) return kErrOom;

  CudaStream cs;
  CublasHandle ch;
  if (cs.init() != cudaSuccess || ch.init(cs.stream) != CUBLAS_STATUS_SUCCESS) return 3;
  cudaStream_t s = cs.stream;
  cublasHandle_t h = ch.handle;

  IntBuf d_bad;
  DoubleBuf d_W, d_hcore, d_Ca, d_Cb, d_charge, d_fock_a, d_fock_b, d_tmp_nn;
  if (d_bad.alloc(2) != cudaSuccess || d_bad.zero_async(2, s) != cudaSuccess ||
      d_W.alloc(Size(q) * q) != cudaSuccess || d_hcore.alloc(nn) != cudaSuccess ||
      d_Ca.alloc(nn) != cudaSuccess || d_charge.alloc(static_cast<Size>(q)) != cudaSuccess ||
      d_charge.zero_async(static_cast<Size>(q), s) != cudaSuccess ||
      d_fock_a.alloc(nn) != cudaSuccess || d_tmp_nn.alloc(nn) != cudaSuccess ||
      (!restricted && (d_Cb.alloc(nn) != cudaSuccess || d_fock_b.alloc(nn) != cudaSuccess))) {
    return 4;
  }
  if (!h2d_async(d_W.ptr, whitener, Size(q) * q, s) ||
      !h2d_async(d_hcore.ptr, hcore_ao, nn, s) ||
      !h2d_async(d_Ca.ptr, C_a, nn, s) ||
      (!restricted && !h2d_async(d_Cb.ptr, C_b, nn, s)) ||
      !check_finite_gpu(d_W.ptr, Size(q) * q, d_bad.ptr, s) ||
      !check_finite_gpu(d_hcore.ptr, nn, d_bad.ptr, s) ||
      !check_finite_gpu(d_Ca.ptr, nn, d_bad.ptr, s) ||
      (!restricted && !check_finite_gpu(d_Cb.ptr, nn, d_bad.ptr, s))) {
    return 5;
  }

  // Initialize fock_s = C_s^T @ hcore_ao @ C_s
  if (rm_gemm(h, true, false, n, n, n, 1.0, d_Ca.ptr, n, d_hcore.ptr, n, 0.0, d_tmp_nn.ptr, n) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, false, false, n, n, n, 1.0, d_tmp_nn.ptr, n, d_Ca.ptr, n, 0.0, d_fock_a.ptr, n) !=
          CUBLAS_STATUS_SUCCESS) {
    return 7;
  }
  if (!restricted) {
    if (rm_gemm(h, true, false, n, n, n, 1.0, d_Cb.ptr, n, d_hcore.ptr, n, 0.0, d_tmp_nn.ptr, n) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, false, n, n, n, 1.0, d_tmp_nn.ptr, n, d_Cb.ptr, n, 0.0, d_fock_b.ptr,
                n) != CUBLAS_STATUS_SUCCESS) {
      return 7;
    }
  }

  DoubleBuf d_ao_ch, d_tmp_ch, d_occ_pack;
  if (d_ao_ch.alloc(Size(chunk) * nn) != cudaSuccess ||
      d_tmp_ch.alloc(Size(chunk) * nn) != cudaSuccess ||
      (max_occ > 0 && d_occ_pack.alloc(Size(chunk) * max_occ * n) != cudaSuccess)) {
    return 4;
  }

  // Optional resident d_df cache: only keep resident if it fits budget AND cudaMalloc succeeds;
  // otherwise clear sticky OOM and fall back cleanly to the exact streaming branch below.
  const Size df_bytes = sat_mul(sat_mul(static_cast<Size>(q), nn), sizeof(double));
  size_t free_mem = 0, total_mem = 0;
  const bool have_mem_info = (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess);
  const bool try_df_resident =
      plan.resident &&
      (!have_mem_info || sat_add(df_bytes, Size(64) << 20) <= static_cast<Size>(free_mem));
  DoubleBuf d_df;
  bool keep_df_resident = false;
  if (try_df_resident) {
    if (d_df.alloc(Size(q) * nn) == cudaSuccess) {
      keep_df_resident = true;
    } else {
      (void)cudaGetLastError();
      d_df.reset();
    }
  }
  if (keep_df_resident) {

    auto process_spin_res = [&](const double* d_C, int occ, double ch_scale, double* df_out,
                                double* d_fock, bool check_ao, bool compute_j_now) -> bool {
      for (int q0 = 0; q0 < q; q0 += chunk) {
        const int actual = std::min(chunk, q - q0);
        if (!h2d_async(d_ao_ch.ptr, ao_coulomb + Size(q0) * nn, Size(actual) * nn, s)) return false;
        if (check_ao && !check_finite_gpu(d_ao_ch.ptr, Size(actual) * nn, d_bad.ptr, s)) return false;
        // d_tmp_ch[a] = C^T @ ao_ch[a]
        if (rm_gemm_batched(h, true, false, n, n, n, 1.0,
                            d_C, n, 0,
                            d_ao_ch.ptr, n, static_cast<long long>(nn),
                            0.0, d_tmp_ch.ptr, n, static_cast<long long>(nn),
                            actual) != CUBLAS_STATUS_SUCCESS) {
          return false;
        }
        // raw_ch = d_tmp_ch @ C (reusing d_ao_ch as raw_ch)
        if (rm_gemm(h, false, false, actual * n, n, n, 1.0,
                    d_tmp_ch.ptr, n, d_C, n,
                    0.0, d_ao_ch.ptr, n) != CUBLAS_STATUS_SUCCESS) {
          return false;
        }
        // d_df[q, nn] += W[:, q0:q1] @ raw_ch[actual, nn]
        if (rm_gemm(h, false, false, q, static_cast<int>(nn), actual, 1.0,
                    d_W.ptr + q0, q, d_ao_ch.ptr, static_cast<int>(nn),
                    (q0 == 0 ? 0.0 : 1.0), d_df.ptr, static_cast<int>(nn)) != CUBLAS_STATUS_SUCCESS) {
          return false;
        }
      }
      if (!check_finite_gpu(d_df.ptr, Size(q) * nn, d_bad.ptr + 1, s)) return false;
      if (occ > 0) {
        k_trace_occ_accum<<<grid_for_cap(static_cast<Size>(q)), 256, 0, s>>>(
            q, n, occ, ch_scale, d_df.ptr, d_charge.ptr);
        if (!launch_ok()) return false;
        for (int q0 = 0; q0 < q; q0 += chunk) {
          const int actual = std::min(chunk, q - q0);
          const Size pack_elems = Size(actual) * occ * n;
          k_pack_leading_rows<<<grid_for_cap(pack_elems), 256, 0, s>>>(
              actual, occ, n, d_df.ptr + Size(q0) * nn, d_occ_pack.ptr);
          if (!launch_ok()) return false;
          if (rm_gemm(h, true, false, n, n, actual * occ, -1.0,
                      d_occ_pack.ptr, n, d_occ_pack.ptr, n,
                      1.0, d_fock, n) != CUBLAS_STATUS_SUCCESS) {
            return false;
          }
        }
      }
      if (compute_j_now) {
        if (rm_gemm(h, true, false, 1, static_cast<int>(nn), q, 1.0,
                    d_charge.ptr, 1, d_df.ptr, static_cast<int>(nn),
                    1.0, d_fock, static_cast<int>(nn)) != CUBLAS_STATUS_SUCCESS) {
          return false;
        }
      }
      return d2h_async(df_out, d_df.ptr, Size(q) * nn, s);
    };

    if (restricted) {
      if (!process_spin_res(d_Ca.ptr, oa, 2.0, df_a_out, d_fock_a.ptr, true, true)) return 7;
    } else {
      if (!process_spin_res(d_Ca.ptr, oa, 1.0, df_a_out, d_fock_a.ptr, true, false)) return 7;
      if (cudaStreamSynchronize(s) != cudaSuccess) return 8;
      if (!process_spin_res(d_Cb.ptr, ob, 1.0, df_b_out, d_fock_b.ptr, false, true)) return 7;
      // Accumulate J_a = sum_A charge_df[A] * df_a[A] by streaming df_a_out in Q-chunks
      for (int q0 = 0; q0 < q; q0 += chunk) {
        const int actual = std::min(chunk, q - q0);
        if (!h2d_async(d_ao_ch.ptr, df_a_out + Size(q0) * nn, Size(actual) * nn, s)) return 5;
        if (rm_gemm(h, true, false, 1, static_cast<int>(nn), actual, 1.0,
                    d_charge.ptr + q0, 1, d_ao_ch.ptr, static_cast<int>(nn),
                    1.0, d_fock_a.ptr, static_cast<int>(nn)) != CUBLAS_STATUS_SUCCESS) {
          return 7;
        }
      }
    }
    // Sync before freeing d_df after async D2H of df_a_out / df_b_out.
    if (cudaStreamSynchronize(s) != cudaSuccess) return 8;
    d_df.reset();
  } else {
    // Stream the DF panel through host memory when the resident cache is unavailable.
    auto build_spin_streamed = [&](const double* d_C, int occ, double ch_scale, double* df_out,
                                   double* d_fock, bool check_ao) -> bool {
      for (int q0 = 0; q0 < q; q0 += chunk) {
        const int actual = std::min(chunk, q - q0);
        if (!h2d_async(d_ao_ch.ptr, ao_coulomb + Size(q0) * nn, Size(actual) * nn, s)) return false;
        if (check_ao && !check_finite_gpu(d_ao_ch.ptr, Size(actual) * nn, d_bad.ptr, s)) return false;
        if (rm_gemm_batched(h, true, false, n, n, n, 1.0,
                            d_C, n, 0,
                            d_ao_ch.ptr, n, static_cast<long long>(nn),
                            0.0, d_tmp_ch.ptr, n, static_cast<long long>(nn),
                            actual) != CUBLAS_STATUS_SUCCESS) {
          return false;
        }
        if (rm_gemm(h, false, false, actual * n, n, n, 1.0,
                    d_tmp_ch.ptr, n, d_C, n,
                    0.0, d_ao_ch.ptr, n) != CUBLAS_STATUS_SUCCESS) {
          return false;
        }
        if (!d2h_async(df_out + Size(q0) * nn, d_ao_ch.ptr, Size(actual) * nn, s)) return false;
      }
      if (cudaStreamSynchronize(s) != cudaSuccess) return false;
      if (!whiten_host_inplace_slabbed_gpu(h, s, false, q, nn, d_W.ptr, df_out, d_bad.ptr + 1, plan.slab_bytes)) {
        return false;
      }
      if (cudaStreamSynchronize(s) != cudaSuccess) return false;
      if (occ > 0) {
        for (int q0 = 0; q0 < q; q0 += chunk) {
          const int actual = std::min(chunk, q - q0);
          if (!h2d_async(d_ao_ch.ptr, df_out + Size(q0) * nn, Size(actual) * nn, s)) return false;
          k_trace_occ_accum<<<grid_for_cap(static_cast<Size>(actual)), 256, 0, s>>>(
              actual, n, occ, ch_scale, d_ao_ch.ptr, d_charge.ptr + q0);
          if (!launch_ok()) return false;
          const Size pack_elems = Size(actual) * occ * n;
          k_pack_leading_rows<<<grid_for_cap(pack_elems), 256, 0, s>>>(
              actual, occ, n, d_ao_ch.ptr, d_occ_pack.ptr);
          if (!launch_ok()) return false;
          if (rm_gemm(h, true, false, n, n, actual * occ, -1.0,
                      d_occ_pack.ptr, n, d_occ_pack.ptr, n,
                      1.0, d_fock, n) != CUBLAS_STATUS_SUCCESS) {
            return false;
          }
        }
      }
      return true;
    };

    if (!build_spin_streamed(d_Ca.ptr, oa, restricted ? 2.0 : 1.0, df_a_out, d_fock_a.ptr, true)) {
      return 7;
    }
    if (!restricted) {
      if (!build_spin_streamed(d_Cb.ptr, ob, 1.0, df_b_out, d_fock_b.ptr, false)) return 7;
    }
    auto add_j_streamed = [&](const double* df_host, double* d_fock) -> bool {
      for (int q0 = 0; q0 < q; q0 += chunk) {
        const int actual = std::min(chunk, q - q0);
        if (!h2d_async(d_ao_ch.ptr, df_host + Size(q0) * nn, Size(actual) * nn, s)) return false;
        if (rm_gemm(h, true, false, 1, static_cast<int>(nn), actual, 1.0,
                    d_charge.ptr + q0, 1, d_ao_ch.ptr, static_cast<int>(nn),
                    1.0, d_fock, static_cast<int>(nn)) != CUBLAS_STATUS_SUCCESS) {
          return false;
        }
      }
      return true;
    };
    if (!add_j_streamed(df_a_out, d_fock_a.ptr)) return 7;
    if (!restricted && !add_j_streamed(df_b_out, d_fock_b.ptr)) return 7;
  }

  if (!check_finite_gpu(d_fock_a.ptr, nn, d_bad.ptr + 1, s) ||
      (!restricted && !check_finite_gpu(d_fock_b.ptr, nn, d_bad.ptr + 1, s))) {
    return 6;
  }

  int h_bad[2] = {0, 0};
  if (!d2h_async(fock_a_out, d_fock_a.ptr, nn, s) ||
      (!restricted && !d2h_async(fock_b_out, d_fock_b.ptr, nn, s)) ||
      !d2h_async(h_bad, d_bad.ptr, 2, s) || cudaStreamSynchronize(s) != cudaSuccess) {
    return 8;
  }
  if (h_bad[0] != 0) return 1;
  if (h_bad[1] != 0) return 2;
  return 0;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_cv_jk_fock_pullback_cuda(
    int q, int n, int oa, int ob, int restricted, int q_chunk, std::uint64_t budget_bytes,
    const double* ao_coulomb, const double* hcore_ao, const double* whitener,
    const double* C_a, const double* C_b,
    const double* df_a, const double* df_b,
    const double* S_a, const double* S_b,
    double* bar_C_a_out, double* bar_C_b_out,
    double* bar_hcore_out, double* bar_ao_out, double* bar_metric_out) {
  try {
  if (q <= 0 || n <= 0 || oa < 0 || ob < 0 || oa >= n || ob >= n || (oa + ob) <= 0) return 1;
  if (!ao_coulomb || !hcore_ao || !whitener || !C_a || !df_a || !S_a) return 1;
  if (!bar_C_a_out || !bar_hcore_out || !bar_ao_out || !bar_metric_out) return 1;
  const bool share_spin = (restricted != 0) && (S_b == nullptr);
  if (!share_spin && (!C_b || !df_b || !S_b || !bar_C_b_out)) return 1;

  const Size nn = Size(n) * n;
  if (!fits_int(nn)) return 1;

  const int max_occ = share_spin ? oa : std::max(oa, ob);
  const auto plan = jk_mo_workspace(true, q, n, max_occ, share_spin, q_chunk, budget_bytes);
  const int chunk = plan.chunk;
  if (!chunk) return kErrOom;

  CudaStream cs;
  CublasHandle ch;
  if (cs.init() != cudaSuccess || ch.init(cs.stream) != CUBLAS_STATUS_SUCCESS) return 3;
  cudaStream_t s = cs.stream;
  cublasHandle_t h = ch.handle;

  IntBuf d_bad;
  DoubleBuf d_W, d_hcore, d_Ca, d_Cb, d_Sa, d_Sb;
  DoubleBuf d_charge, d_bar_charge, d_bar_M, d_bar_hcore, d_bar_Ca, d_bar_Cb;
  DoubleBuf d_S_sym, d_tmp_nn;
  if (d_bad.alloc(2) != cudaSuccess || d_bad.zero_async(2, s) != cudaSuccess ||
      d_W.alloc(Size(q) * q) != cudaSuccess || d_hcore.alloc(nn) != cudaSuccess ||
      d_Ca.alloc(nn) != cudaSuccess || d_Sa.alloc(nn) != cudaSuccess ||
      d_charge.alloc(static_cast<Size>(q)) != cudaSuccess ||
      d_bar_charge.alloc(static_cast<Size>(q)) != cudaSuccess ||
      d_bar_M.alloc(Size(q) * q) != cudaSuccess || d_bar_hcore.alloc(nn) != cudaSuccess ||
      d_bar_Ca.alloc(nn) != cudaSuccess || d_S_sym.alloc(nn) != cudaSuccess ||
      d_tmp_nn.alloc(nn) != cudaSuccess ||
      (!share_spin && (d_Cb.alloc(nn) != cudaSuccess || d_Sb.alloc(nn) != cudaSuccess ||
                       d_bar_Cb.alloc(nn) != cudaSuccess))) {
    return 4;
  }

  if (d_charge.zero_async(static_cast<Size>(q), s) != cudaSuccess ||
      d_bar_charge.zero_async(static_cast<Size>(q), s) != cudaSuccess ||
      d_bar_M.zero_async(Size(q) * q, s) != cudaSuccess ||
      d_bar_hcore.zero_async(nn, s) != cudaSuccess ||
      d_bar_Ca.zero_async(nn, s) != cudaSuccess ||
      (!share_spin && d_bar_Cb.zero_async(nn, s) != cudaSuccess) ||
      !h2d_async(d_W.ptr, whitener, Size(q) * q, s) ||
      !h2d_async(d_hcore.ptr, hcore_ao, nn, s) ||
      !h2d_async(d_Ca.ptr, C_a, nn, s) ||
      !h2d_async(d_Sa.ptr, S_a, nn, s) ||
      (!share_spin && (!h2d_async(d_Cb.ptr, C_b, nn, s) || !h2d_async(d_Sb.ptr, S_b, nn, s))) ||
      !check_finite_gpu(d_W.ptr, Size(q) * q, d_bad.ptr, s) ||
      !check_finite_gpu(d_hcore.ptr, nn, d_bad.ptr, s) ||
      !check_finite_gpu(d_Ca.ptr, nn, d_bad.ptr, s) ||
      !check_finite_gpu(d_Sa.ptr, nn, d_bad.ptr, s) ||
      (!share_spin && (!check_finite_gpu(d_Cb.ptr, nn, d_bad.ptr, s) ||
                       !check_finite_gpu(d_Sb.ptr, nn, d_bad.ptr, s)))) {
    return 5;
  }

  const double share_scale = share_spin ? 2.0 : 1.0;
  // 1) Compute charge_df[A] and bar_charge_df[A] via streamed Q-chunks (zero full-df allocation yet)
  {
    DoubleBuf d_df_ch;
    if (d_df_ch.alloc(Size(chunk) * nn) != cudaSuccess) return 4;
    auto accum_charges = [&](const double* df_host, const double* d_S, int occ, double sc) -> bool {
      for (int q0 = 0; q0 < q; q0 += chunk) {
        const int actual = std::min(chunk, q - q0);
        if (!h2d_async(d_df_ch.ptr, df_host + Size(q0) * nn, Size(actual) * nn, s) ||
            !check_finite_gpu(d_df_ch.ptr, Size(actual) * nn, d_bad.ptr, s)) {
          return false;
        }
        if (rm_gemm(h, false, false, actual, 1, static_cast<int>(nn), sc,
                    d_df_ch.ptr, static_cast<int>(nn), d_S, 1,
                    1.0, d_bar_charge.ptr + q0, 1) != CUBLAS_STATUS_SUCCESS) {
          return false;
        }
        if (occ > 0) {
          k_trace_occ_accum<<<grid_for_cap(static_cast<Size>(actual)), 256, 0, s>>>(
              actual, n, occ, sc, d_df_ch.ptr, d_charge.ptr + q0);
          if (!launch_ok()) return false;
        }
      }
      return true;
    };
    if (!accum_charges(df_a, d_Sa.ptr, oa, share_scale)) return 7;
    if (!share_spin && !accum_charges(df_b, d_Sb.ptr, ob, 1.0)) return 7;
    if (cudaStreamSynchronize(s) != cudaSuccess) return 8;
  }

  std::vector<double> ones(chunk, 1.0);
  DoubleBuf d_ones;
  if (d_ones.alloc(static_cast<Size>(chunk)) != cudaSuccess ||
      !h2d_async(d_ones.ptr, ones.data(), static_cast<Size>(chunk), s)) {
    return 4;
  }

  auto process_spin = [&](int spin_idx, const double* d_C, const double* df_host, const double* d_S,
                          int occ, double scale, double* d_bar_C) -> bool {
    // Hcore pullback:
    //   bar_hcore += scale * C @ S @ C^T
    //   bar_C += hcore_ao @ C @ S^T + hcore_ao^T @ C @ S
    if (rm_gemm(h, false, false, n, n, n, 1.0, d_C, n, d_S, n, 0.0, d_tmp_nn.ptr, n) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, true, n, n, n, scale, d_tmp_nn.ptr, n, d_C, n, 1.0, d_bar_hcore.ptr, n) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, false, n, n, n, 1.0, d_hcore.ptr, n, d_C, n, 0.0, d_tmp_nn.ptr, n) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, true, n, n, n, 1.0, d_tmp_nn.ptr, n, d_S, n, 1.0, d_bar_C, n) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, true, false, n, n, n, 1.0, d_hcore.ptr, n, d_C, n, 0.0, d_tmp_nn.ptr, n) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, false, n, n, n, 1.0, d_tmp_nn.ptr, n, d_S, n, 1.0, d_bar_C, n) !=
            CUBLAS_STATUS_SUCCESS) {
      return false;
    }

    k_make_sym_sum<<<grid_for_cap(nn), 256, 0, s>>>(n, d_S, d_S_sym.ptr);
    if (!launch_ok()) return false;

    if (plan.resident) {
      DoubleBuf d_bar_raw;
      DoubleBuf d_df_full, d_bdf_ch, d_occ_pack, d_ex;
      bool resident_ok = (d_bar_raw.alloc(Size(q) * nn) == cudaSuccess) &&
                         (d_df_full.alloc(Size(q) * nn) == cudaSuccess) &&
                         (d_bdf_ch.alloc(Size(chunk) * nn) == cudaSuccess) &&
                         (occ == 0 || (d_occ_pack.alloc(Size(chunk) * occ * n) == cudaSuccess &&
                                       d_ex.alloc(Size(chunk) * occ * n) == cudaSuccess));
      if (!resident_ok) {
        (void)cudaGetLastError();
        d_ex.reset();
        d_occ_pack.reset();
        d_bdf_ch.reset();
        d_df_full.reset();
        d_bar_raw.reset();
      } else {
        // Phase A: build bar_df_s chunks from resident d_df_full, accumulate bar_M & bar_raw_s.
        // Release d_df_full and Phase A scratch BEFORE Phase B allocates chunk buffers!
        if (!h2d_async(d_df_full.ptr, df_host, Size(q) * nn, s)) return false;

        for (int q0 = 0; q0 < q; q0 += chunk) {
          const int actual = std::min(chunk, q - q0);
          k_init_bar_df_chunk<<<grid_for_cap(Size(actual) * nn), 256, 0, s>>>(
              actual, n, occ, d_charge.ptr + q0, d_bar_charge.ptr + q0, d_S, d_bdf_ch.ptr);
          if (!launch_ok()) return false;
          if (occ > 0) {
            const Size rn_tot = Size(actual) * occ * n;
            k_pack_leading_rows<<<grid_for_cap(rn_tot), 256, 0, s>>>(
                actual, occ, n, d_df_full.ptr + Size(q0) * nn, d_occ_pack.ptr);
            if (!launch_ok()) return false;
            if (rm_gemm(h, false, false, actual * occ, n, n, 1.0,
                        d_occ_pack.ptr, n, d_S_sym.ptr, n,
                        0.0, d_ex.ptr, n) != CUBLAS_STATUS_SUCCESS) {
              return false;
            }
            k_sub_leading_rows<<<grid_for_cap(rn_tot), 256, 0, s>>>(
                actual, occ, n, d_ex.ptr, d_bdf_ch.ptr);
            if (!launch_ok()) return false;
          }
          // bar_M[q0:q1, :] += (-0.5 * scale) * bar_df_ch @ df_full^T
          if (rm_gemm(h, false, true, actual, q, static_cast<int>(nn), -0.5 * scale,
                      d_bdf_ch.ptr, static_cast<int>(nn), d_df_full.ptr, static_cast<int>(nn),
                      1.0, d_bar_M.ptr + Size(q0) * q, q) != CUBLAS_STATUS_SUCCESS) {
            return false;
          }
          // bar_raw_s[q, nn] += W[q0:q1, :]^T @ bar_df_ch[actual, nn]
          if (rm_gemm(h, true, false, q, static_cast<int>(nn), actual, 1.0,
                      d_W.ptr + Size(q0) * q, q, d_bdf_ch.ptr, static_cast<int>(nn),
                      (q0 == 0 ? 0.0 : 1.0), d_bar_raw.ptr, static_cast<int>(nn)) !=
              CUBLAS_STATUS_SUCCESS) {
            return false;
          }
        }
        if (cudaStreamSynchronize(s) != cudaSuccess) return false;
        d_ex.reset();
        d_occ_pack.reset();
        d_bdf_ch.reset();
        d_df_full.reset();

        // Phase B: Q-chunked back-projection to bar_ao_out and bar_C
        DoubleBuf d_ao_ch, d_tmp1, d_tmp2;
        if (d_ao_ch.alloc(Size(chunk) * nn) != cudaSuccess ||
            d_tmp1.alloc(Size(chunk) * nn) != cudaSuccess ||
            d_tmp2.alloc(Size(chunk) * nn) != cudaSuccess) {
          return false;
        }
        for (int q0 = 0; q0 < q; q0 += chunk) {
          const int actual = std::min(chunk, q - q0);
          const double* br_ch = d_bar_raw.ptr + Size(q0) * nn;
          if (!h2d_async(d_ao_ch.ptr, ao_coulomb + Size(q0) * nn, Size(actual) * nn, s)) return false;
          if (spin_idx == 0 && !check_finite_gpu(d_ao_ch.ptr, Size(actual) * nn, d_bad.ptr, s)) {
            return false;
          }
          if (spin_idx > 0 && !h2d_async(d_tmp2.ptr, bar_ao_out + Size(q0) * nn, Size(actual) * nn, s)) {
            return false;
          }
          // bar_ao[a] += scale * (C @ br_a @ C^T)
          if (rm_gemm_batched(h, false, false, n, n, n, 1.0,
                              d_C, n, 0,
                              br_ch, n, static_cast<long long>(nn),
                              0.0, d_tmp1.ptr, n, static_cast<long long>(nn),
                              actual) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm(h, false, true, actual * n, n, n, scale,
                      d_tmp1.ptr, n, d_C, n,
                      (spin_idx == 0 ? 0.0 : 1.0), d_tmp2.ptr, n) != CUBLAS_STATUS_SUCCESS) {
            return false;
          }
          if (!check_finite_gpu(d_tmp2.ptr, Size(actual) * nn, d_bad.ptr + 1, s) ||
              !d2h_async(bar_ao_out + Size(q0) * nn, d_tmp2.ptr, Size(actual) * nn, s)) {
            return false;
          }
          // bar_C += sum_a (ao_a @ C) @ br_a^T
          if (rm_gemm(h, false, false, actual * n, n, n, 1.0,
                      d_ao_ch.ptr, n, d_C, n,
                      0.0, d_tmp1.ptr, n) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm_batched(h, false, true, n, n, n, 1.0,
                              d_tmp1.ptr, n, static_cast<long long>(nn),
                              br_ch, n, static_cast<long long>(nn),
                              0.0, d_tmp2.ptr, n, static_cast<long long>(nn),
                              actual) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm(h, false, false, 1, static_cast<int>(nn), actual, 1.0,
                      d_ones.ptr, actual, d_tmp2.ptr, static_cast<int>(nn),
                      1.0, d_bar_C, static_cast<int>(nn)) != CUBLAS_STATUS_SUCCESS) {
            return false;
          }
          // bar_C += sum_a (ao_a^T @ C) @ br_a
          if (rm_gemm_batched(h, true, false, n, n, n, 1.0,
                              d_ao_ch.ptr, n, static_cast<long long>(nn),
                              d_C, n, 0,
                              0.0, d_tmp1.ptr, n, static_cast<long long>(nn),
                              actual) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm_batched(h, false, false, n, n, n, 1.0,
                              d_tmp1.ptr, n, static_cast<long long>(nn),
                              br_ch, n, static_cast<long long>(nn),
                              0.0, d_tmp2.ptr, n, static_cast<long long>(nn),
                              actual) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm(h, false, false, 1, static_cast<int>(nn), actual, 1.0,
                      d_ones.ptr, actual, d_tmp2.ptr, static_cast<int>(nn),
                      1.0, d_bar_C, static_cast<int>(nn)) != CUBLAS_STATUS_SUCCESS) {
            return false;
          }
        }
        return cudaStreamSynchronize(s) == cudaSuccess;
      }
    }

    // Stream the DF panel through host memory when the resident cache is unavailable.
    std::vector<double> h_bar_raw(Size(q) * nn);
    {
      DoubleBuf d_df_ch, d_bdf_ch, d_occ_pack, d_ex;
      if (d_df_ch.alloc(Size(chunk) * nn) != cudaSuccess ||
          d_bdf_ch.alloc(Size(chunk) * nn) != cudaSuccess ||
          (occ > 0 && (d_occ_pack.alloc(Size(chunk) * occ * n) != cudaSuccess ||
                       d_ex.alloc(Size(chunk) * occ * n) != cudaSuccess))) {
        return false;
      }
      for (int q0 = 0; q0 < q; q0 += chunk) {
        const int actual = std::min(chunk, q - q0);
        if (!h2d_async(d_df_ch.ptr, df_host + Size(q0) * nn, Size(actual) * nn, s)) return false;
        k_init_bar_df_chunk<<<grid_for_cap(Size(actual) * nn), 256, 0, s>>>(
            actual, n, occ, d_charge.ptr + q0, d_bar_charge.ptr + q0, d_S, d_bdf_ch.ptr);
        if (!launch_ok()) return false;
        if (occ > 0) {
          const Size rn_tot = Size(actual) * occ * n;
          k_pack_leading_rows<<<grid_for_cap(rn_tot), 256, 0, s>>>(
              actual, occ, n, d_df_ch.ptr, d_occ_pack.ptr);
          if (!launch_ok()) return false;
          if (rm_gemm(h, false, false, actual * occ, n, n, 1.0,
                      d_occ_pack.ptr, n, d_S_sym.ptr, n,
                      0.0, d_ex.ptr, n) != CUBLAS_STATUS_SUCCESS) {
            return false;
          }
          k_sub_leading_rows<<<grid_for_cap(rn_tot), 256, 0, s>>>(
              actual, occ, n, d_ex.ptr, d_bdf_ch.ptr);
          if (!launch_ok()) return false;
        }
        if (!d2h_async(h_bar_raw.data() + Size(q0) * nn, d_bdf_ch.ptr, Size(actual) * nn, s)) {
          return false;
        }
        // Tiled inner loop over df_host to accumulate bar_M[q0:q1, k0:k1]
        int loaded_k0 = q0;
        for (int k0 = 0; k0 < q; k0 += chunk) {
          const int act_k = std::min(chunk, q - k0);
          if (k0 != loaded_k0) {
            if (!h2d_async(d_df_ch.ptr, df_host + Size(k0) * nn, Size(act_k) * nn, s)) return false;
            loaded_k0 = k0;
          }
          if (rm_gemm(h, false, true, actual, act_k, static_cast<int>(nn), -0.5 * scale,
                      d_bdf_ch.ptr, static_cast<int>(nn), d_df_ch.ptr, static_cast<int>(nn),
                      1.0, d_bar_M.ptr + Size(q0) * q + k0, q) != CUBLAS_STATUS_SUCCESS) {
            return false;
          }
        }
      }
      if (cudaStreamSynchronize(s) != cudaSuccess) return false;
    }
    if (!whiten_host_inplace_slabbed_gpu(h, s, true, q, nn, d_W.ptr, h_bar_raw.data(),
                                         d_bad.ptr + 1, plan.slab_bytes) ||
        cudaStreamSynchronize(s) != cudaSuccess) {
      return false;
    }
    DoubleBuf d_ao_ch, d_br_ch, d_tmp1, d_tmp2;
    if (d_ao_ch.alloc(Size(chunk) * nn) != cudaSuccess ||
        d_br_ch.alloc(Size(chunk) * nn) != cudaSuccess ||
        d_tmp1.alloc(Size(chunk) * nn) != cudaSuccess ||
        d_tmp2.alloc(Size(chunk) * nn) != cudaSuccess) {
      return false;
    }
    for (int q0 = 0; q0 < q; q0 += chunk) {
      const int actual = std::min(chunk, q - q0);
      if (!h2d_async(d_ao_ch.ptr, ao_coulomb + Size(q0) * nn, Size(actual) * nn, s) ||
          !h2d_async(d_br_ch.ptr, h_bar_raw.data() + Size(q0) * nn, Size(actual) * nn, s)) {
        return false;
      }
      if (spin_idx == 0 && !check_finite_gpu(d_ao_ch.ptr, Size(actual) * nn, d_bad.ptr, s)) {
        return false;
      }
      if (spin_idx > 0 && !h2d_async(d_tmp2.ptr, bar_ao_out + Size(q0) * nn, Size(actual) * nn, s)) {
        return false;
      }
      if (rm_gemm_batched(h, false, false, n, n, n, 1.0,
                          d_C, n, 0,
                          d_br_ch.ptr, n, static_cast<long long>(nn),
                          0.0, d_tmp1.ptr, n, static_cast<long long>(nn),
                          actual) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, true, actual * n, n, n, scale,
                  d_tmp1.ptr, n, d_C, n,
                  (spin_idx == 0 ? 0.0 : 1.0), d_tmp2.ptr, n) != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
      if (!check_finite_gpu(d_tmp2.ptr, Size(actual) * nn, d_bad.ptr + 1, s) ||
          !d2h_async(bar_ao_out + Size(q0) * nn, d_tmp2.ptr, Size(actual) * nn, s)) {
        return false;
      }
      if (rm_gemm(h, false, false, actual * n, n, n, 1.0,
                  d_ao_ch.ptr, n, d_C, n,
                  0.0, d_tmp1.ptr, n) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm_batched(h, false, true, n, n, n, 1.0,
                          d_tmp1.ptr, n, static_cast<long long>(nn),
                          d_br_ch.ptr, n, static_cast<long long>(nn),
                          0.0, d_tmp2.ptr, n, static_cast<long long>(nn),
                          actual) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, 1, static_cast<int>(nn), actual, 1.0,
                  d_ones.ptr, actual, d_tmp2.ptr, static_cast<int>(nn),
                  1.0, d_bar_C, static_cast<int>(nn)) != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
      if (rm_gemm_batched(h, true, false, n, n, n, 1.0,
                          d_ao_ch.ptr, n, static_cast<long long>(nn),
                          d_C, n, 0,
                          0.0, d_tmp1.ptr, n, static_cast<long long>(nn),
                          actual) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm_batched(h, false, false, n, n, n, 1.0,
                          d_tmp1.ptr, n, static_cast<long long>(nn),
                          d_br_ch.ptr, n, static_cast<long long>(nn),
                          0.0, d_tmp2.ptr, n, static_cast<long long>(nn),
                          actual) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, 1, static_cast<int>(nn), actual, 1.0,
                  d_ones.ptr, actual, d_tmp2.ptr, static_cast<int>(nn),
                  1.0, d_bar_C, static_cast<int>(nn)) != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
    }
    return cudaStreamSynchronize(s) == cudaSuccess;
  };

  if (!process_spin(0, d_Ca.ptr, df_a, d_Sa.ptr, oa, share_scale, d_bar_Ca.ptr)) return 7;
  if (!share_spin) {
    if (!process_spin(1, d_Cb.ptr, df_b, d_Sb.ptr, ob, 1.0, d_bar_Cb.ptr)) return 7;
  }

  d_ones.reset();
  // bar_metric = sym(W^T @ bar_M @ W)
  DoubleBuf d_mw;
  if (d_mw.alloc(Size(q) * q) != cudaSuccess) {
    return 4;
  }
  if (rm_gemm(h, false, false, q, q, q, 1.0, d_bar_M.ptr, q, d_W.ptr, q, 0.0, d_mw.ptr, q) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, true, false, q, q, q, 1.0, d_W.ptr, q, d_mw.ptr, q, 0.0, d_bar_M.ptr, q) !=
          CUBLAS_STATUS_SUCCESS ||
      !symmetrize_batched_gpu(1, q, d_bar_M.ptr, s)) {
    return 7;
  }

  if (!check_finite_gpu(d_bar_Ca.ptr, nn, d_bad.ptr + 1, s) ||
      (!share_spin && !check_finite_gpu(d_bar_Cb.ptr, nn, d_bad.ptr + 1, s)) ||
      !check_finite_gpu(d_bar_hcore.ptr, nn, d_bad.ptr + 1, s) ||
      !check_finite_gpu(d_bar_M.ptr, Size(q) * q, d_bad.ptr + 1, s)) {
    return 6;
  }

  int h_bad[2] = {0, 0};
  if (!d2h_async(bar_C_a_out, d_bar_Ca.ptr, nn, s) ||
      (!share_spin && !d2h_async(bar_C_b_out, d_bar_Cb.ptr, nn, s)) ||
      !d2h_async(bar_hcore_out, d_bar_hcore.ptr, nn, s) ||
      !d2h_async(bar_metric_out, d_bar_M.ptr, Size(q) * q, s) ||
      !d2h_async(h_bad, d_bad.ptr, 2, s) || cudaStreamSynchronize(s) != cudaSuccess) {
    return 8;
  }
  if (h_bad[0] != 0) return 1;
  if (h_bad[1] != 0) return 2;
  return 0;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_cv_cphf_matvec_cuda(
    int q, int n, int oa, int ob, int restricted, int q_chunk, std::uint64_t budget_bytes,
    const double* df_a, const double* df_b,
    const double* gap_a, const double* gap_b,
    const double* trial_a, const double* trial_b,
    double* out_a, double* out_b) {
  try {
  if (q <= 0 || n <= 0 || oa < 0 || ob < 0 || oa >= n || ob >= n || (oa == 0 && ob == 0)) return 1;
  if (oa > 0 && (!df_a || !gap_a || !trial_a || !out_a)) return 1;
  if (!restricted && ob > 0 && (!df_b || !gap_b || !trial_b || !out_b)) return 1;

  const Size nn = Size(n) * n;
  const int va = n - oa;
  const int vb = n - ob;
  const Size vo_a = Size(va) * oa;
  const Size vo_b = Size(vb) * ob;
  if (!fits_int(nn) || !fits_int(vo_a) || !fits_int(vo_b)) return 1;

  const bool do_a = (oa > 0);
  const bool do_b = (!restricted && ob > 0);

  const Size active_vo = sat_add(do_a ? vo_a : 0, do_b ? vo_b : 0);
  const Size max_vo = std::max(do_a ? vo_a : Size(0), do_b ? vo_b : Size(0));
  const Size max_oo = std::max(do_a ? Size(oa) * oa : Size(0), do_b ? Size(ob) * ob : Size(0));
  const Size spins = Size(do_a) + Size(do_b);
  // Three resident VO matrices per spin (gap, trial, result). A streamed row
  // holds each spin's DF, VO and charge, two exchange VO blocks, OO and two scalars.
  const Size cap = effective_budget(budget_bytes, kDefaultChunkBudgetBytes);
  const Size room = workspace_after_fixed(cap, sat_mul(3, active_vo));
  const Size spin_rows = sat_add(sat_mul(spins, nn), sat_add(active_vo, spins));
  const Size exchange = sat_add(sat_mul(2, max_vo), max_oo);
  const Size bytes_per_q = sat_mul(sat_add(spin_rows, sat_add(exchange, 2)), sizeof(double));
  const int chunk = static_cast<int>(std::min(room / bytes_per_q,
      Size(q_chunk > 0 ? std::min(q, q_chunk) : q)));
  if (chunk == 0) return kErrOom;

  CudaStream cs;
  CublasHandle ch;
  if (cs.init() != cudaSuccess || ch.init(cs.stream) != CUBLAS_STATUS_SUCCESS) return 3;
  cudaStream_t s = cs.stream;
  cublasHandle_t h = ch.handle;

  IntBuf d_bad;
  DoubleBuf d_gap_a, d_trial_a, d_out_a;
  DoubleBuf d_gap_b, d_trial_b, d_out_b;
  if (d_bad.alloc(2) != cudaSuccess || d_bad.zero_async(2, s) != cudaSuccess) return 4;

  if (do_a) {
    if (d_gap_a.alloc(vo_a) != cudaSuccess || d_trial_a.alloc(vo_a) != cudaSuccess ||
        d_out_a.alloc(vo_a) != cudaSuccess) {
      return 4;
    }
    if (!h2d_async(d_gap_a.ptr, gap_a, vo_a, s) || !h2d_async(d_trial_a.ptr, trial_a, vo_a, s) ||
        !check_finite_gpu(d_gap_a.ptr, vo_a, d_bad.ptr, s) ||
        !check_finite_gpu(d_trial_a.ptr, vo_a, d_bad.ptr, s)) {
      return 5;
    }
    k_mul_elemwise<<<grid_for_cap(vo_a), 256, 0, s>>>(vo_a, d_gap_a.ptr, d_trial_a.ptr, d_out_a.ptr);
    if (!launch_ok()) return 6;
  }
  if (do_b) {
    if (d_gap_b.alloc(vo_b) != cudaSuccess || d_trial_b.alloc(vo_b) != cudaSuccess ||
        d_out_b.alloc(vo_b) != cudaSuccess) {
      return 4;
    }
    if (!h2d_async(d_gap_b.ptr, gap_b, vo_b, s) || !h2d_async(d_trial_b.ptr, trial_b, vo_b, s) ||
        !check_finite_gpu(d_gap_b.ptr, vo_b, d_bad.ptr, s) ||
        !check_finite_gpu(d_trial_b.ptr, vo_b, d_bad.ptr, s)) {
      return 5;
    }
    k_mul_elemwise<<<grid_for_cap(vo_b), 256, 0, s>>>(vo_b, d_gap_b.ptr, d_trial_b.ptr, d_out_b.ptr);
    if (!launch_ok()) return 6;
  }

  DoubleBuf d_df_a_ch, d_df_b_ch, d_vo_a, d_vo_b, d_ch_a, d_ch_b, d_jcoeff;
  DoubleBuf d_kv, d_ki, d_tmp_vo, d_ones;
  if ((do_a && (d_df_a_ch.alloc(Size(chunk) * nn) != cudaSuccess ||
                d_vo_a.alloc(Size(chunk) * vo_a) != cudaSuccess ||
                d_ch_a.alloc(static_cast<Size>(chunk)) != cudaSuccess)) ||
      (do_b && (d_df_b_ch.alloc(Size(chunk) * nn) != cudaSuccess ||
                d_vo_b.alloc(Size(chunk) * vo_b) != cudaSuccess ||
                d_ch_b.alloc(static_cast<Size>(chunk)) != cudaSuccess)) ||
      d_jcoeff.alloc(static_cast<Size>(chunk)) != cudaSuccess ||
      d_kv.alloc(Size(chunk) * max_vo) != cudaSuccess ||
      d_ki.alloc(Size(chunk) * max_oo) != cudaSuccess ||
      d_tmp_vo.alloc(Size(chunk) * max_vo) != cudaSuccess ||
      d_ones.alloc(static_cast<Size>(chunk)) != cudaSuccess) {
    return 4;
  }
  std::vector<double> ones(chunk, 1.0);
  if (!h2d_async(d_ones.ptr, ones.data(), static_cast<Size>(chunk), s)) return 5;

  for (Size q0 = 0; q0 < Size(q); q0 += Size(chunk)) {
    const int actual = static_cast<int>(std::min(Size(chunk), Size(q) - q0));
    if (do_a) {
      if (!h2d_async(d_df_a_ch.ptr, df_a + Size(q0) * nn, Size(actual) * nn, s) ||
          !check_finite_gpu(d_df_a_ch.ptr, Size(actual) * nn, d_bad.ptr, s)) {
        return 5;
      }
      k_extract_subblock<<<grid_for_cap(Size(actual) * vo_a), 256, 0, s>>>(
          actual, va, oa, oa, 0, n, d_df_a_ch.ptr, d_vo_a.ptr);
      if (!launch_ok()) return 6;
      if (rm_gemm(h, false, false, actual, 1, static_cast<int>(vo_a), 1.0,
                  d_vo_a.ptr, static_cast<int>(vo_a), d_trial_a.ptr, 1,
                  0.0, d_ch_a.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
        return 7;
      }
    }
    if (do_b) {
      if (!h2d_async(d_df_b_ch.ptr, df_b + Size(q0) * nn, Size(actual) * nn, s) ||
          !check_finite_gpu(d_df_b_ch.ptr, Size(actual) * nn, d_bad.ptr, s)) {
        return 5;
      }
      k_extract_subblock<<<grid_for_cap(Size(actual) * vo_b), 256, 0, s>>>(
          actual, vb, ob, ob, 0, n, d_df_b_ch.ptr, d_vo_b.ptr);
      if (!launch_ok()) return 6;
      if (rm_gemm(h, false, false, actual, 1, static_cast<int>(vo_b), 1.0,
                  d_vo_b.ptr, static_cast<int>(vo_b), d_trial_b.ptr, 1,
                  0.0, d_ch_b.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
        return 7;
      }
    }

    k_cphf_jcoeff<<<grid_for_cap(static_cast<Size>(actual)), 256, 0, s>>>(
        actual, restricted, do_a ? d_ch_a.ptr : nullptr, do_b ? d_ch_b.ptr : nullptr, d_jcoeff.ptr);
    if (!launch_ok()) return 6;

    auto apply_spin_chunk = [&](int occ, int vir, const double* d_df_ch, const double* d_vo,
                                const double* d_trial, double* d_out) -> bool {
      const Size vo = Size(vir) * occ;
      const Size oo = Size(occ) * occ;
      // + sum_a j_coeff[a] * vo[a]
      if (rm_gemm(h, true, false, 1, static_cast<int>(vo), actual, 1.0,
                  d_jcoeff.ptr, 1, d_vo, static_cast<int>(vo),
                  1.0, d_out, static_cast<int>(vo)) != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
      // kv[a] = vv[a] @ trial; out -= sum_a kv[a] @ oo[a]^T
      if (rm_gemm_batched(h, false, false, vir, occ, vir, 1.0,
                          d_df_ch + Size(occ) * n + occ, n, static_cast<long long>(nn),
                          d_trial, occ, 0,
                          0.0, d_kv.ptr, occ, static_cast<long long>(vo),
                          actual) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm_batched(h, false, true, vir, occ, occ, 1.0,
                          d_kv.ptr, occ, static_cast<long long>(vo),
                          d_df_ch, n, static_cast<long long>(nn),
                          0.0, d_tmp_vo.ptr, occ, static_cast<long long>(vo),
                          actual) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, 1, static_cast<int>(vo), actual, -1.0,
                  d_ones.ptr, actual, d_tmp_vo.ptr, static_cast<int>(vo),
                  1.0, d_out, static_cast<int>(vo)) != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
      // ki[a] = vo[a]^T @ trial; out -= sum_a vo[a] @ ki[a]^T
      if (rm_gemm_batched(h, true, false, occ, occ, vir, 1.0,
                          d_vo, occ, static_cast<long long>(vo),
                          d_trial, occ, 0,
                          0.0, d_ki.ptr, occ, static_cast<long long>(oo),
                          actual) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm_batched(h, false, true, vir, occ, occ, 1.0,
                          d_vo, occ, static_cast<long long>(vo),
                          d_ki.ptr, occ, static_cast<long long>(oo),
                          0.0, d_tmp_vo.ptr, occ, static_cast<long long>(vo),
                          actual) != CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, 1, static_cast<int>(vo), actual, -1.0,
                  d_ones.ptr, actual, d_tmp_vo.ptr, static_cast<int>(vo),
                  1.0, d_out, static_cast<int>(vo)) != CUBLAS_STATUS_SUCCESS) {
        return false;
      }
      return true;
    };

    if (do_a && !apply_spin_chunk(oa, va, d_df_a_ch.ptr, d_vo_a.ptr, d_trial_a.ptr, d_out_a.ptr)) {
      return 7;
    }
    if (do_b && !apply_spin_chunk(ob, vb, d_df_b_ch.ptr, d_vo_b.ptr, d_trial_b.ptr, d_out_b.ptr)) {
      return 7;
    }
  }

  if ((do_a && !check_finite_gpu(d_out_a.ptr, vo_a, d_bad.ptr + 1, s)) ||
      (do_b && !check_finite_gpu(d_out_b.ptr, vo_b, d_bad.ptr + 1, s))) {
    return 6;
  }

  int h_bad[2] = {0, 0};
  if ((do_a && !d2h_async(out_a, d_out_a.ptr, vo_a, s)) ||
      (do_b && !d2h_async(out_b, d_out_b.ptr, vo_b, s)) ||
      !d2h_async(h_bad, d_bad.ptr, 2, s) || cudaStreamSynchronize(s) != cudaSuccess) {
    return 8;
  }
  if (h_bad[0] != 0) return 1;
  if (h_bad[1] != 0) return 2;
  return 0;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_cv_nuclear_source_weights_cuda(
    int q, int n, int oa, int ob, int restricted, int q_chunk, std::uint64_t budget_bytes,
    const double* whitener,
    const double* C_a, const double* C_b,
    const double* df_a, const double* df_b,
    const double* p_a, const double* p_b,
    const double* sm_a, const double* sm_b,
    double* bar_ds_out, double* bar_dh_out,
    double* bar_dr_out, double* bar_dj_out) {
  try {
  if (q <= 0 || n <= 0 || oa < 0 || ob < 0 || oa >= n || ob >= n || (oa == 0 && ob == 0)) return 1;
  if (!whitener || !C_a || !df_a || !p_a || !sm_a) return 1;
  if (!bar_ds_out || !bar_dh_out || !bar_dr_out || !bar_dj_out) return 1;
  if (!restricted && (!C_b || !df_b || !p_b || !sm_b)) return 1;

  const Size nn = Size(n) * n;
  if (!fits_int(nn)) return 1;

  const Size qq = Size(q) * q;
  const int max_occ = restricted ? oa : std::max(oa, ob);
  const Size no = Size(n) * max_occ, oo = Size(max_occ) * max_occ;
  const Size cap = effective_budget(budget_bytes, kDefaultJkBudgetBytes);
  const Size slab_cap = budget_bytes > 0 ? cap : kDefaultChunkBudgetBytes;
  const Size spin_chunk_cap = budget_bytes > 0 ? cap : (Size(768) << 20);
  // After the spin stage only W, bar_M, bar_ds and bar_dh remain live.
  // Whitening needs a pair of q-vector slabs. The final two GEMMs reuse
  // bar_M as their output and need one additional q-by-q matrix.
  const Size final_fixed = sat_add(sat_mul(2, qq), sat_mul(2, nn));
  const Size final_room = workspace_after_fixed(cap, final_fixed);
  const Size slab_budget = std::min(slab_cap, final_room);
  if (sat_mul(Size(2) * q, sizeof(double)) > slab_budget ||
      sat_mul(qq, sizeof(double)) > final_room) return kErrOom;

  // W, bar_M, two charge vectors, 8 (restricted) or 11 (unrestricted)
  // n-by-n matrices, and an occupied-block accumulator coexist here.
  const Size spin_fixed = sat_add(sat_add(sat_mul(2, qq), Size(2) * q),
                                  sat_add(sat_mul(restricted ? 8 : 11, nn), oo));
  const Size spin_room = workspace_after_fixed(cap, spin_fixed);
  // The DF staging slab can double as GEMM scratch: W/M and density
  // contractions finish reading it on the same stream before C @ bdb writes it.
  const Size plane = sat_add(sat_mul(3, nn), sat_add(no, sat_add(oo, 1)));
  const Size plane_bytes = sat_mul(plane, sizeof(double));
  const Size streamed_room = std::min(spin_chunk_cap, spin_room);
  if (plane_bytes > streamed_room) return kErrOom;

  const Size df_bytes = sat_mul(sat_mul(Size(q), nn), sizeof(double));
  bool resident_df = df_bytes <= spin_room - plane_bytes;
  const Size chunk_room = resident_df ? std::min(spin_chunk_cap, spin_room - df_bytes) : streamed_room;
  Size max_chunk = chunk_room / plane_bytes;
  max_chunk = std::min(max_chunk, Size(q_chunk > 0 ? std::min(q, q_chunk) : q));
  max_chunk = std::min(max_chunk, Size(std::numeric_limits<int>::max()) / Size(n));
  const int chunk = static_cast<int>(max_chunk);

  CudaStream cs;
  CublasHandle ch;
  if (cs.init() != cudaSuccess || ch.init(cs.stream) != CUBLAS_STATUS_SUCCESS) return 3;
  cudaStream_t s = cs.stream;
  cublasHandle_t h = ch.handle;

  IntBuf d_bad;
  DoubleBuf d_W, d_Ca, d_Cb, d_pa, d_pb, d_sma, d_smb;
  DoubleBuf d_charge, d_bar_dcharge, d_bar_ds, d_bar_dh, d_bar_M;
  DoubleBuf d_psym, d_bar_sm, d_tmp_nn;
  if (d_bad.alloc(2) != cudaSuccess || d_bad.zero_async(2, s) != cudaSuccess ||
      d_W.alloc(Size(q) * q) != cudaSuccess || d_Ca.alloc(nn) != cudaSuccess ||
      d_pa.alloc(nn) != cudaSuccess || d_sma.alloc(nn) != cudaSuccess ||
      d_charge.alloc(static_cast<Size>(q)) != cudaSuccess ||
      d_bar_dcharge.alloc(static_cast<Size>(q)) != cudaSuccess ||
      d_bar_ds.alloc(nn) != cudaSuccess || d_bar_dh.alloc(nn) != cudaSuccess ||
      d_bar_M.alloc(Size(q) * q) != cudaSuccess || d_psym.alloc(nn) != cudaSuccess ||
      d_bar_sm.alloc(nn) != cudaSuccess || d_tmp_nn.alloc(nn) != cudaSuccess ||
      (!restricted && (d_Cb.alloc(nn) != cudaSuccess || d_pb.alloc(nn) != cudaSuccess ||
                       d_smb.alloc(nn) != cudaSuccess))) {
    return 4;
  }

  if (d_charge.zero_async(static_cast<Size>(q), s) != cudaSuccess ||
      d_bar_dcharge.zero_async(static_cast<Size>(q), s) != cudaSuccess ||
      d_bar_ds.zero_async(nn, s) != cudaSuccess || d_bar_dh.zero_async(nn, s) != cudaSuccess ||
      d_bar_M.zero_async(Size(q) * q, s) != cudaSuccess ||
      !h2d_async(d_W.ptr, whitener, Size(q) * q, s) ||
      !h2d_async(d_Ca.ptr, C_a, nn, s) || !h2d_async(d_pa.ptr, p_a, nn, s) ||
      !h2d_async(d_sma.ptr, sm_a, nn, s) ||
      (!restricted && (!h2d_async(d_Cb.ptr, C_b, nn, s) || !h2d_async(d_pb.ptr, p_b, nn, s) ||
                       !h2d_async(d_smb.ptr, sm_b, nn, s))) ||
      !check_finite_gpu(d_W.ptr, Size(q) * q, d_bad.ptr, s) ||
      !check_finite_gpu(d_Ca.ptr, nn, d_bad.ptr, s) ||
      !check_finite_gpu(d_pa.ptr, nn, d_bad.ptr, s) ||
      !check_finite_gpu(d_sma.ptr, nn, d_bad.ptr, s) ||
      (!restricted && (!check_finite_gpu(d_Cb.ptr, nn, d_bad.ptr, s) ||
                       !check_finite_gpu(d_pb.ptr, nn, d_bad.ptr, s) ||
                       !check_finite_gpu(d_smb.ptr, nn, d_bad.ptr, s)))) {
    return 5;
  }

  const double share_scale = restricted ? 2.0 : 1.0;
  // Release the ones vector and spin workspace before the final whitening slabs.
  {
    std::vector<double> ones(chunk, 1.0);
    DoubleBuf d_ones;
    if (d_ones.alloc(Size(chunk)) != cudaSuccess) return kErrOom;
    if (!h2d_async(d_ones.ptr, ones.data(), Size(chunk), s)) return 5;
    DoubleBuf d_df_full;
    DoubleBuf d_bdb_ch, d_ao_db_ch, d_tmp1, d_pb_ch, d_oo_ch, d_oo_acc;
    if (d_bdb_ch.alloc(Size(chunk) * nn) != cudaSuccess ||
        d_ao_db_ch.alloc(Size(chunk) * nn) != cudaSuccess ||
        d_tmp1.alloc(Size(chunk) * nn) != cudaSuccess ||
        (max_occ > 0 && (d_pb_ch.alloc(Size(chunk) * n * max_occ) != cudaSuccess ||
                         d_oo_ch.alloc(Size(chunk) * max_occ * max_occ) != cudaSuccess ||
                         d_oo_acc.alloc(Size(max_occ) * max_occ) != cudaSuccess))) {
      return 4;
    }
    if (resident_df) {
      const cudaError_t alloc_status = d_df_full.alloc(Size(q) * nn);
      if (alloc_status == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();
        std::fputs("gqc_f12_cv_nuclear_source_weights_cuda: DF cache allocation OOM; streaming exact slabs\n", stderr);
        resident_df = false;
      } else if (alloc_status != cudaSuccess) {
        return kErrInternal;
      }
    }
    // Streaming reuses d_tmp1 for the DF input until the final AO GEMMs.
    double* const df_slab = d_tmp1.ptr;

    // 1) Compute charge[A] and bar_dcharge[A]
    if (restricted && resident_df) {
      // Single upload of df_a reused for steps 1, 2, and 3!
      if (!h2d_async(d_df_full.ptr, df_a, Size(q) * nn, s) ||
          !check_finite_gpu(d_df_full.ptr, Size(q) * nn, d_bad.ptr, s)) {
        return 5;
      }
      if (rm_gemm(h, false, false, q, 1, static_cast<int>(nn), share_scale,
                  d_df_full.ptr, static_cast<int>(nn), d_pa.ptr, 1,
                  0.0, d_bar_dcharge.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
        return 7;
      }
      if (oa > 0) {
        k_trace_occ_accum<<<grid_for_cap(static_cast<Size>(q)), 256, 0, s>>>(
            q, n, oa, share_scale, d_df_full.ptr, d_charge.ptr);
        if (!launch_ok()) return 6;
      }
    } else {
      auto accum_charges = [&](const double* df_host, const double* d_p, int occ,
                               double sc) -> bool {
        for (int q0 = 0; q0 < q; q0 += chunk) {
          const int actual = std::min(chunk, q - q0);
          if (!h2d_async(df_slab, df_host + Size(q0) * nn, Size(actual) * nn, s) ||
              !check_finite_gpu(df_slab, Size(actual) * nn, d_bad.ptr, s)) {
            return false;
          }
          if (rm_gemm(h, false, false, actual, 1, static_cast<int>(nn), sc,
                      df_slab, static_cast<int>(nn), d_p, 1,
                      1.0, d_bar_dcharge.ptr + q0, 1) != CUBLAS_STATUS_SUCCESS) {
            return false;
          }
          if (occ > 0) {
            k_trace_occ_accum<<<grid_for_cap(static_cast<Size>(actual)), 256, 0, s>>>(
                actual, n, occ, sc, df_slab, d_charge.ptr + q0);
            if (!launch_ok()) return false;
          }
        }
        return true;
      };
      if (!accum_charges(df_a, d_pa.ptr, oa, share_scale)) return 7;
      if (!restricted && !accum_charges(df_b, d_pb.ptr, ob, 1.0)) return 7;
    }

    // 2) & 3) Per-spin one-body bar_dh/bar_ds, bar_M, and ao_db_sym (staged in bar_dr_out)
    auto process_spin = [&](int spin_idx, bool is_last_spin, const double* d_C,
                            const double* df_host, const double* d_p, const double* d_sm_in,
                            int occ, double scale, bool already_loaded) -> bool {
      if (resident_df && !already_loaded) {
        if (!h2d_async(d_df_full.ptr, df_host, Size(q) * nn, s)) return false;
      }

      // Finish one-body bar_sm, bar_dh, bar_ds
      if (cudaMemcpyAsync(d_bar_sm.ptr, d_sm_in, nn * sizeof(double), cudaMemcpyDeviceToDevice, s) !=
          cudaSuccess) {
        return false;
      }
      if (occ > 0) {
        if (d_oo_acc.zero_async(Size(occ) * occ, s) != cudaSuccess) return false;
        const int oo_int = occ * occ;
        for (int q0 = 0; q0 < q; q0 += chunk) {
          const int actual = std::min(chunk, q - q0);
          const double* df_ptr = nullptr;
          if (resident_df) {
            df_ptr = d_df_full.ptr + Size(q0) * nn;
          } else {
            if (!h2d_async(df_slab, df_host + Size(q0) * nn, Size(actual) * nn, s)) {
              return false;
            }
            df_ptr = df_slab;
          }
          // -bar_dcharge[a] * sl[:occ, :occ]
          k_extract_subblock<<<grid_for_cap(Size(actual) * oo_int), 256, 0, s>>>(
              actual, occ, occ, 0, 0, n, df_ptr, d_oo_ch.ptr);
          if (!launch_ok()) return false;
          if (rm_gemm(h, true, false, 1, oo_int, actual, -1.0,
                      d_bar_dcharge.ptr + q0, 1, d_oo_ch.ptr, oo_int,
                      1.0, d_oo_acc.ptr, oo_int) != CUBLAS_STATUS_SUCCESS) {
            return false;
          }
          // pb[a, n, occ] = p^T @ sl[:, :occ]; s_ex[a, occ, occ] = pb[a]^T @ sl[:, :occ]
          if (rm_gemm_batched(h, true, false, n, occ, n, 1.0,
                              d_p, n, 0,
                              df_ptr, n, static_cast<long long>(nn),
                              0.0, d_pb_ch.ptr, occ, static_cast<long long>(Size(n) * occ),
                              actual) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm_batched(h, true, false, occ, occ, n, 1.0,
                              d_pb_ch.ptr, occ, static_cast<long long>(Size(n) * occ),
                              df_ptr, n, static_cast<long long>(nn),
                              0.0, d_oo_ch.ptr, occ, static_cast<long long>(oo_int),
                              actual) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm(h, false, false, 1, oo_int, actual, 1.0,
                      d_ones.ptr, actual, d_oo_ch.ptr, oo_int,
                      1.0, d_oo_acc.ptr, oo_int) != CUBLAS_STATUS_SUCCESS) {
            return false;
          }
        }
        k_add_subblock_single<<<grid_for_cap(Size(occ) * occ), 256, 0, s>>>(
            occ, occ, 0, 0, n, 1.0, d_oo_acc.ptr, d_bar_sm.ptr);
        if (!launch_ok()) return false;
      }

      // bar_dh += scale * (C @ p @ C^T); bar_ds += scale * (C @ bar_sm @ C^T)
      if (rm_gemm(h, false, false, n, n, n, 1.0, d_C, n, d_p, n, 0.0, d_tmp_nn.ptr, n) !=
              CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, true, n, n, n, scale, d_tmp_nn.ptr, n, d_C, n, 1.0, d_bar_dh.ptr, n) !=
              CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, false, n, n, n, 1.0, d_C, n, d_bar_sm.ptr, n, 0.0, d_tmp_nn.ptr, n) !=
              CUBLAS_STATUS_SUCCESS ||
          rm_gemm(h, false, true, n, n, n, scale, d_tmp_nn.ptr, n, d_C, n, 1.0, d_bar_ds.ptr, n) !=
              CUBLAS_STATUS_SUCCESS) {
        return false;
      }

      // Q-chunked bar_dB -> bar_M and ao_db_ch (staged into bar_dr_out)
      k_make_sym_sum<<<grid_for_cap(nn), 256, 0, s>>>(n, d_p, d_psym.ptr);
      if (!launch_ok()) return false;

      for (int q0 = 0; q0 < q; q0 += chunk) {
        const int actual = std::min(chunk, q - q0);
        const double* df_ptr = nullptr;
        if (resident_df) {
          df_ptr = d_df_full.ptr + Size(q0) * nn;
        } else {
          if (!h2d_async(df_slab, df_host + Size(q0) * nn, Size(actual) * nn, s)) return false;
          df_ptr = df_slab;
        }

        // bdb[al, p, r] = charge[a] * p_s[p, r] + (p==r && p<occ ? bar_dcharge[a] : 0)
        k_init_bar_df_chunk<<<grid_for_cap(Size(actual) * nn), 256, 0, s>>>(
            actual, n, occ, d_charge.ptr + q0, d_bar_dcharge.ptr + q0, d_p, d_bdb_ch.ptr);
        if (!launch_ok()) return false;

        // bdb[al, :, :occ] -= psym @ df_s[a, :, :occ]
        if (occ > 0) {
          if (rm_gemm_batched(h, false, false, n, occ, n, -1.0,
                              d_psym.ptr, n, 0,
                              df_ptr, n, static_cast<long long>(nn),
                              1.0, d_bdb_ch.ptr, n, static_cast<long long>(nn),
                              actual) != CUBLAS_STATUS_SUCCESS) {
            return false;
          }
        }

        // bar_M[q0:q1, :] += (-0.5 * scale) * bar_db_ch @ df_s^T
        if (resident_df) {
          if (rm_gemm(h, false, true, actual, q, static_cast<int>(nn), -0.5 * scale,
                      d_bdb_ch.ptr, static_cast<int>(nn), d_df_full.ptr, static_cast<int>(nn),
                      1.0, d_bar_M.ptr + Size(q0) * q, q) != CUBLAS_STATUS_SUCCESS) {
            return false;
          }
        } else {
          int loaded_k0 = q0;
          for (int k0 = 0; k0 < q; k0 += chunk) {
            const int act_k = std::min(chunk, q - k0);
            if (k0 != loaded_k0) {
              if (!h2d_async(df_slab, df_host + Size(k0) * nn, Size(act_k) * nn, s)) {
                return false;
              }
              loaded_k0 = k0;
            }
            if (rm_gemm(h, false, true, actual, act_k, static_cast<int>(nn), -0.5 * scale,
                        d_bdb_ch.ptr, static_cast<int>(nn), df_slab, static_cast<int>(nn),
                        1.0, d_bar_M.ptr + Size(q0) * q + k0, q) != CUBLAS_STATUS_SUCCESS) {
              return false;
            }
          }
        }

        // ao_db_ch[al] += scale * (C @ bdb[al] @ C^T)
        if (spin_idx > 0) {
          if (!h2d_async(d_ao_db_ch.ptr, bar_dr_out + Size(q0) * nn, Size(actual) * nn, s)) {
            return false;
          }
        }
        if (rm_gemm_batched(h, false, false, n, n, n, 1.0,
                            d_C, n, 0,
                            d_bdb_ch.ptr, n, static_cast<long long>(nn),
                            0.0, d_tmp1.ptr, n, static_cast<long long>(nn),
                            actual) != CUBLAS_STATUS_SUCCESS ||
            rm_gemm(h, false, true, actual * n, n, n, scale,
                    d_tmp1.ptr, n, d_C, n,
                    (spin_idx == 0 ? 0.0 : 1.0), d_ao_db_ch.ptr, n) != CUBLAS_STATUS_SUCCESS) {
          return false;
        }
        if (is_last_spin) {
          if (!symmetrize_batched_gpu(actual, n, d_ao_db_ch.ptr, s)) return false;
        }
        if (!d2h_async(bar_dr_out + Size(q0) * nn, d_ao_db_ch.ptr, Size(actual) * nn, s)) {
          return false;
        }
      }
      return cudaStreamSynchronize(s) == cudaSuccess;
    };

    if (!process_spin(0, restricted != 0, d_Ca.ptr, df_a, d_pa.ptr, d_sma.ptr, oa, share_scale,
                      restricted && resident_df)) {
      return 7;
    }
    if (!restricted) {
      if (!process_spin(1, true, d_Cb.ptr, df_b, d_pb.ptr, d_smb.ptr, ob, 1.0, false)) {
        return 7;
      }
    }
  }

  d_Ca.reset();
  d_Cb.reset();
  d_pa.reset();
  d_pb.reset();
  d_sma.reset();
  d_smb.reset();
  d_charge.reset();
  d_bar_dcharge.reset();
  d_psym.reset();
  d_bar_sm.reset();
  d_tmp_nn.reset();

  // Finalize bar_dr = W^T @ ao_db_sym in-place via <= 1 GiB 2D column slabs!
  if (!whiten_host_inplace_slabbed_gpu(h, s, true, q, nn, d_W.ptr, bar_dr_out, d_bad.ptr + 1, slab_budget)) {
    return 7;
  }

  // Symmetrize bar_dh, bar_ds and compute bar_dj = sym(W^T @ bar_M @ W)
  DoubleBuf d_mw;
  if (d_mw.alloc(qq) != cudaSuccess) {
    return 4;
  }
  if (!symmetrize_batched_gpu(1, n, d_bar_dh.ptr, s) ||
      !symmetrize_batched_gpu(1, n, d_bar_ds.ptr, s) ||
      rm_gemm(h, false, false, q, q, q, 1.0, d_bar_M.ptr, q, d_W.ptr, q, 0.0, d_mw.ptr, q) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemm(h, true, false, q, q, q, 1.0, d_W.ptr, q, d_mw.ptr, q, 0.0, d_bar_M.ptr, q) !=
          CUBLAS_STATUS_SUCCESS ||
      !symmetrize_batched_gpu(1, q, d_bar_M.ptr, s)) {
    return 7;
  }

  if (!check_finite_gpu(d_bar_ds.ptr, nn, d_bad.ptr + 1, s) ||
      !check_finite_gpu(d_bar_dh.ptr, nn, d_bad.ptr + 1, s) ||
      !check_finite_gpu(d_bar_M.ptr, Size(q) * q, d_bad.ptr + 1, s)) {
    return 6;
  }

  int h_bad[2] = {0, 0};
  if (!d2h_async(bar_ds_out, d_bar_ds.ptr, nn, s) ||
      !d2h_async(bar_dh_out, d_bar_dh.ptr, nn, s) ||
      !d2h_async(bar_dj_out, d_bar_M.ptr, Size(q) * q, s) ||
      !d2h_async(h_bad, d_bad.ptr, 2, s) || cudaStreamSynchronize(s) != cudaSuccess) {
    return 8;
  }
  if (h_bad[0] != 0) return 1;
  if (h_bad[1] != 0) return 2;
  return 0;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

#endif  // GQC_SINK_HOST_REFERENCE

}  // extern "C"
