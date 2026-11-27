// GPU-resident cuBLAS (and matching host reference) AOT primitives for F12 CPHF
// Step 1 + Step 2: 3-center whitening, orbital rotation/metric pullback, CPHF
// orbital Hessian action (RHF + UHF), DF canonical orbital driver, and DF
// nuclear source weights (single-tile and Q-tiled).
#include <algorithm>
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
#endif

namespace {

using Size = std::uint64_t;

constexpr int kOk = 0;
constexpr int kErrInvalidArg = -1;
constexpr int kErrOom = -2;
#ifndef GQC_SINK_HOST_REFERENCE
constexpr int kErrCuda = -3;
constexpr int kErrCublas = -4;
#endif
constexpr int kErrBudget = -5;
constexpr int kErrNonFinite = -6;
constexpr int kErrInternal = -7;

constexpr Size kMaxBudgetBytes = Size(512) << 20;

inline bool fits_int(Size v) {
  return v <= static_cast<Size>(std::numeric_limits<int>::max());
}

inline bool checked_mul(Size a, Size b, Size& out) {
  if (a != 0 && b > std::numeric_limits<Size>::max() / a) return false;
  out = a * b;
  return true;
}

inline bool check_finite(const double* ptr, Size count) {
  if (!ptr || count == 0) return true;
  for (Size i = 0; i < count; ++i) {
    if (!std::isfinite(ptr[i])) return false;
  }
  return true;
}

inline Size effective_budget_bytes(Size budget_bytes) {
  if (budget_bytes == 0) return kMaxBudgetBytes;
  return std::min(budget_bytes, kMaxBudgetBytes);
}

struct WhitenTiles {
  Size a_tile = 0;
  Size b_tile = 0;
  Size c_tile = 0;
};

inline WhitenTiles compute_whiten_tiles(Size s_aout, Size s_ain, Size s_ncols, Size budget_bytes) {
  const Size budget_doubles = effective_budget_bytes(budget_bytes) / sizeof(double);
  const Size max_c_tile = std::min<Size>(s_ncols, std::max<Size>(1, budget_doubles / 4));
  const Size c_tile = std::min<Size>(max_c_tile, static_cast<Size>(1 << 24));
  if (c_tile == 0) return {};

  Size a_tile = s_aout;
  Size b_tile = s_ain;
  auto fits = [&](Size a, Size b) -> bool {
    return a * b + (a + b) * c_tile <= budget_doubles;
  };
  if (!fits(a_tile, b_tile)) {
    const Size side_w = std::max<Size>(
        1, static_cast<Size>(std::sqrt(static_cast<double>(budget_doubles / 4))));
    const Size side_slab = std::max<Size>(1, (budget_doubles * 3) / (8 * c_tile));
    a_tile = std::min<Size>(s_aout, std::min(side_w, side_slab));
    b_tile = std::min<Size>(s_ain, std::min(side_w, side_slab));
    while (!fits(a_tile, b_tile) && (a_tile > 1 || b_tile > 1)) {
      if (a_tile >= b_tile && a_tile > 1) {
        a_tile = std::max<Size>(1, a_tile / 2);
      } else if (b_tile > 1) {
        b_tile = std::max<Size>(1, b_tile / 2);
      }
    }
    if (!fits(a_tile, b_tile)) return {};
  }
  return {a_tile, b_tile, c_tile};
}

inline int compute_cphf_action_q_tile(int oa, int ob, int obs, int aux, Size budget_bytes) {
  const int va = (oa > 0) ? (obs - oa) : 0;
  const int vb = (ob > 0) ? (obs - ob) : 0;
  const Size n_vo_a = Size(va) * static_cast<Size>(oa);
  const Size n_vo_b = Size(vb) * static_cast<Size>(ob);
  const Size n_oo_a = Size(oa) * static_cast<Size>(oa);
  const Size n_oo_b = Size(ob) * static_cast<Size>(ob);
  const Size n_vv_a = Size(va) * static_cast<Size>(va);
  const Size n_vv_b = Size(vb) * static_cast<Size>(vb);
  const Size max_vo = std::max(n_vo_a, n_vo_b);
  const Size max_oo = std::max(n_oo_a, n_oo_b);
  const Size max_vv = std::max(n_vv_a, n_vv_b);
  const Size budget_doubles = effective_budget_bytes(budget_bytes) / sizeof(double);
  const Size fixed_doubles = 3 * (n_vo_a + n_vo_b);
  const Size per_q_doubles = n_vo_a + n_vo_b + max_vv + max_oo + 2 * max_vo + 2 * max_oo + 1;
  if (fixed_doubles + per_q_doubles > budget_doubles) return 0;
  const Size max_by_int =
      static_cast<Size>(std::numeric_limits<int>::max()) / static_cast<Size>(obs);
  if (max_by_int == 0) return 0;
  return static_cast<int>(std::min<Size>(
      static_cast<Size>(aux), std::min<Size>((budget_doubles - fixed_doubles) / per_q_doubles,
                                             max_by_int)));
}

inline int compute_df_driver_q_tile(int oa, int ob, bool has_charge_a, bool has_charge_b,
                                    int obs, int aux, Size budget_bytes) {
  const Size n = static_cast<Size>(obs);
  const Size n2 = n * n;
  const Size n_vo_a = (oa > 0) ? Size(obs - oa) * static_cast<Size>(oa) : 0;
  const Size n_vo_b = (ob > 0) ? Size(obs - ob) * static_cast<Size>(ob) : 0;
  const Size budget_doubles = effective_budget_bytes(budget_bytes) / sizeof(double);
  const Size fixed_a = (oa > 0) ? (2 * n2 + n_vo_a) : (has_charge_a ? n2 : 0);
  const Size fixed_b = (ob > 0) ? (2 * n2 + n_vo_b) : (has_charge_b ? n2 : 0);
  const Size fixed_doubles = fixed_a + fixed_b;
  const Size per_q_doubles = 3 * n2 + 1;
  if (fixed_doubles + per_q_doubles > budget_doubles) return 0;
  const Size max_by_int = static_cast<Size>(std::numeric_limits<int>::max()) / n;
  if (max_by_int == 0) return 0;
  return static_cast<int>(std::min<Size>(
      static_cast<Size>(aux), std::min<Size>((budget_doubles - fixed_doubles) / per_q_doubles,
                                             max_by_int)));
}

inline int compute_df_nuclear_weights_q_tile(int oa, int ob, int obs, int aux, int n_eff,
                                             bool same_spin_all, Size budget_bytes) {
  const Size n = static_cast<Size>(obs);
  const Size ne = static_cast<Size>(n_eff);
  const Size n2 = n * n;
  const Size ne2 = ne * ne;
  const Size n_spins = same_spin_all ? 1 : 2;
  const Size budget_doubles = effective_budget_bytes(budget_bytes) / sizeof(double);
  const Size fixed_doubles =
      n_spins * (ne * n + 2 * n2) + Size(oa) * Size(oa) +
      (same_spin_all ? Size(0) : Size(ob) * Size(ob)) + 2 * static_cast<Size>(aux);
  const Size per_q_doubles = 2 * n2 + 3 * ne2;
  if (fixed_doubles + per_q_doubles > budget_doubles) return 0;
  const Size max_by_int = static_cast<Size>(std::numeric_limits<int>::max()) / n;
  if (max_by_int == 0) return 0;
  return static_cast<int>(std::min<Size>(
      static_cast<Size>(aux), std::min<Size>((budget_doubles - fixed_doubles) / per_q_doubles,
                                             max_by_int)));
}

struct NuclearWeightsSetup {
  int n_eff = 0;
  bool same_spin_all = false;
  std::vector<double> p_a;
  std::vector<double> p_b;
  std::vector<double> psym_a;
  std::vector<double> psym_b;
  std::vector<double> bar_sm_a;
  std::vector<double> bar_sm_b;
};

inline bool rows_tail_all_zero(const double* coeff, int nao, int obs) {
  const Size n = static_cast<Size>(obs);
  const Size tail_start = n * n;
  const Size total = static_cast<Size>(nao) * n;
  for (Size idx = tail_start; idx < total; ++idx) {
    if (coeff[idx] != 0.0) return false;
  }
  return true;
}

inline NuclearWeightsSetup prepare_nuclear_weights_setup(
    int oa, int ob, int obs, int nao, const double* coeff_a, const double* coeff_b,
    const double* occ_e_a, const double* occ_e_b, const double* df_a, const double* df_b,
    const double* fock_seed_a, const double* fock_seed_b, const double* overlap_seed_a,
    const double* overlap_seed_b, const double* z_a, const double* z_b) {
  const Size n = static_cast<Size>(obs);
  const Size n2 = n * n;
  NuclearWeightsSetup s;
  s.p_a.assign(fock_seed_a, fock_seed_a + n2);
  s.p_b.assign(fock_seed_b, fock_seed_b + n2);
  s.bar_sm_a.assign(overlap_seed_a, overlap_seed_a + n2);
  s.bar_sm_b.assign(overlap_seed_b, overlap_seed_b + n2);

  auto apply_z = [&](int occ, const double* z, const double* eps, std::vector<double>& p,
                     std::vector<double>& sm, std::vector<double>& psym) {
    if (occ <= 0) return;
    const int virt = obs - occ;
    const Size so = static_cast<Size>(occ);
    for (int v = 0; v < virt; ++v) {
      const Size row = static_cast<Size>(occ + v);
      for (int i = 0; i < occ; ++i) {
        const Size si = static_cast<Size>(i);
        const double zvi = z[Size(v) * so + si];
        p[row * n + si] -= zvi;
        sm[row * n + si] += zvi * eps[i];
      }
    }
    psym.resize(n2);
    for (Size r = 0; r < n; ++r) {
      for (Size c = 0; c < n; ++c) {
        psym[r * n + c] = p[r * n + c] + p[c * n + r];
      }
    }
  };

  apply_z(oa, z_a, occ_e_a, s.p_a, s.bar_sm_a, s.psym_a);
  apply_z(ob, z_b, occ_e_b, s.p_b, s.bar_sm_b, s.psym_b);

  const bool obs_only =
      (nao > obs) && rows_tail_all_zero(coeff_a, nao, obs) &&
      (coeff_b == coeff_a || rows_tail_all_zero(coeff_b, nao, obs));
  s.n_eff = obs_only ? obs : nao;

  const Size ne_n = static_cast<Size>(s.n_eff) * n;
  const bool same_c =
      (coeff_a == coeff_b) || (std::memcmp(coeff_a, coeff_b, ne_n * sizeof(double)) == 0);
  const bool same_p = std::equal(s.p_a.begin(), s.p_a.end(), s.p_b.begin());
  s.same_spin_all = (oa == ob) && (df_a == df_b) && same_c && same_p;
  return s;
}

inline void symmetrize_square_host(Size n, double* mat) {
  for (Size r = 0; r < n; ++r) {
    for (Size c = r + 1; c < n; ++c) {
      const Size rc = r * n + c;
      const Size cr = c * n + r;
      const double avg = 0.5 * (mat[rc] + mat[cr]);
      mat[rc] = avg;
      mat[cr] = avg;
    }
  }
}

inline void accumulate_dr_tile_host(int a0, int t0, int tl, int nao, int n_eff, bool include_direct,
                                    const double* direct_ao_coulomb, const double* dr_tile,
                                    double* ao_coulomb_out) {
  const Size snao = static_cast<Size>(nao);
  const Size snao2 = snao * snao;
  const Size sne = static_cast<Size>(n_eff);
  const Size sne2 = sne * sne;
  if (a0 == 0) {
    if (include_direct) {
      if (ao_coulomb_out != direct_ao_coulomb) {
        std::memcpy(ao_coulomb_out + Size(t0) * snao2, direct_ao_coulomb + Size(t0) * snao2,
                    Size(tl) * snao2 * sizeof(double));
      }
    } else {
      std::memset(ao_coulomb_out + Size(t0) * snao2, 0, Size(tl) * snao2 * sizeof(double));
    }
  }
  if (n_eff == nao) {
    double* dst = ao_coulomb_out + Size(t0) * snao2;
    const Size count = Size(tl) * snao2;
    for (Size idx = 0; idx < count; ++idx) dst[idx] += dr_tile[idx];
  } else {
    for (int t = 0; t < tl; ++t) {
      double* dst_slice = ao_coulomb_out + Size(t0 + t) * snao2;
      const double* src_slice = dr_tile + Size(t) * sne2;
      for (Size m = 0; m < sne; ++m) {
        double* dst_row = dst_slice + m * snao;
        const double* src_row = src_slice + m * sne;
        for (Size k = 0; k < sne; ++k) dst_row[k] += src_row[k];
      }
    }
  }
}

#ifdef GQC_SINK_HOST_REFERENCE

void rm_gemm_host(bool trans_a, bool trans_b, Size M, Size N, Size K, double alpha,
                  const double* A, Size lda, const double* B, Size ldb, double beta, double* C,
                  Size ldc) {
  if (beta == 0.0) {
    for (Size i = 0; i < M; ++i) {
      std::memset(C + i * ldc, 0, N * sizeof(double));
    }
  } else if (beta != 1.0) {
    for (Size i = 0; i < M; ++i) {
      double* ci = C + i * ldc;
      for (Size j = 0; j < N; ++j) ci[j] *= beta;
    }
  }
  if (!trans_a && !trans_b) {
    for (Size i = 0; i < M; ++i) {
      const double* ai = A + i * lda;
      double* ci = C + i * ldc;
      for (Size k = 0; k < K; ++k) {
        const double aik = alpha * ai[k];
        if (aik == 0.0) continue;
        const double* bk = B + k * ldb;
        for (Size j = 0; j < N; ++j) ci[j] += aik * bk[j];
      }
    }
  } else if (!trans_a && trans_b) {
    for (Size i = 0; i < M; ++i) {
      const double* ai = A + i * lda;
      double* ci = C + i * ldc;
      for (Size j = 0; j < N; ++j) {
        const double* bj = B + j * ldb;
        double acc = 0.0;
        for (Size k = 0; k < K; ++k) acc += ai[k] * bj[k];
        ci[j] += alpha * acc;
      }
    }
  } else if (trans_a && !trans_b) {
    for (Size k = 0; k < K; ++k) {
      const double* ak = A + k * lda;
      const double* bk = B + k * ldb;
      for (Size i = 0; i < M; ++i) {
        const double aki = alpha * ak[i];
        if (aki == 0.0) continue;
        double* ci = C + i * ldc;
        for (Size j = 0; j < N; ++j) ci[j] += aki * bk[j];
      }
    }
  } else {
    for (Size i = 0; i < M; ++i) {
      double* ci = C + i * ldc;
      for (Size j = 0; j < N; ++j) {
        const double* bj = B + j * ldb;
        double acc = 0.0;
        for (Size k = 0; k < K; ++k) acc += A[k * lda + i] * bj[k];
        ci[j] += alpha * acc;
      }
    }
  }
}

void cphf_exchange_spin_host(int occ, int virt, int aux, const double* vo, const double* oo,
                             const double* vv, const double* trial, double* out) {
  const Size o = static_cast<Size>(occ);
  const Size v = static_cast<Size>(virt);
  const Size vo_stride = v * o;
  const Size oo_stride = o * o;
  const Size vv_stride = v * v;
  std::vector<double> kv(vo_stride);
  std::vector<double> ki(oo_stride);
  for (int q = 0; q < aux; ++q) {
    const double* vo_q = vo + Size(q) * vo_stride;
    const double* oo_q = oo + Size(q) * oo_stride;
    const double* vv_q = vv + Size(q) * vv_stride;
    // kv[a, j] = sum_b vv[q, a, b] * trial[b, j]
    rm_gemm_host(false, false, v, o, v, 1.0, vv_q, v, trial, o, 0.0, kv.data(), o);
    // out[a, i] -= sum_j kv[a, j] * oo[q, i, j]
    rm_gemm_host(false, true, v, o, o, -1.0, kv.data(), o, oo_q, o, 1.0, out, o);
    // ki[i, j] = sum_b vo[q, b, i] * trial[b, j]
    rm_gemm_host(true, false, o, o, v, 1.0, vo_q, o, trial, o, 0.0, ki.data(), o);
    // out[a, i] -= sum_j vo[q, a, j] * ki[i, j]
    rm_gemm_host(false, true, v, o, o, -1.0, vo_q, o, ki.data(), o, 1.0, out, o);
  }
}

void df_driver_exchange_spin_host(int obs, int aux, const double* df, const double* seed,
                                  double* field) {
  const Size n = static_cast<Size>(obs);
  const Size n2 = n * n;
  std::vector<double> weighted(n2);
  for (int q = 0; q < aux; ++q) {
    const double* df_q = df + Size(q) * n2;
    // weighted[p, j] = sum_i df[q, p, i] * seed[i, j]
    rm_gemm_host(false, false, n, n, n, 1.0, df_q, n, seed, n, 0.0, weighted.data(), n);
    // field[p, r] -= sum_j weighted[p, j] * df[q, r, j]
    rm_gemm_host(false, true, n, n, n, -1.0, weighted.data(), n, df_q, n, 1.0, field, n);
  }
}

#else

struct DeviceBuf {
  double* ptr = nullptr;
  ~DeviceBuf() {
    if (ptr) cudaFree(ptr);
  }
  cudaError_t alloc(Size count) {
    if (count == 0) return cudaSuccess;
    return cudaMalloc(&ptr, count * sizeof(double));
  }
};

struct CublasGuard {
  cublasHandle_t handle = nullptr;
  ~CublasGuard() {
    if (handle) cublasDestroy(handle);
  }
  cublasStatus_t init() {
    const auto status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    return cublasSetWorkspace(handle, nullptr, 0);
  }
};

inline cublasStatus_t rm_gemm(cublasHandle_t h, bool trans_a, bool trans_b, int M, int N, int K,
                              double alpha, const double* A, int lda, const double* B, int ldb,
                              double beta, double* C, int ldc) {
  return cublasDgemm(h, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N, trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                     N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc);
}

inline cublasStatus_t rm_gemm_batched(cublasHandle_t h, bool trans_a, bool trans_b, int M, int N,
                                      int K, double alpha, const double* A, int lda,
                                      long long stride_a, const double* B, int ldb,
                                      long long stride_b, double beta, double* C, int ldc,
                                      long long stride_c, int batch_count) {
  return cublasDgemmStridedBatched(h, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                                   trans_a ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B, ldb,
                                   stride_b, A, lda, stride_a, &beta, C, ldc, stride_c,
                                   batch_count);
}

inline unsigned int grid_1d(Size n) {
  return static_cast<unsigned int>((n + 255) / 256);
}

__global__ void k_transpose_3d_012_to_102(int d0, int d1, int d2, const double* in, double* out) {
  const Size total = Size(d0) * Size(d1) * Size(d2);
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const Size s2 = static_cast<Size>(d2);
  const Size c = idx % s2;
  const Size tmp = idx / s2;
  const Size b = tmp % static_cast<Size>(d1);
  const Size a = tmp / static_cast<Size>(d1);
  out[(b * static_cast<Size>(d0) + a) * s2 + c] = in[idx];
}

__global__ void k_mask_symmetrize(int n, const double* rot, const double* mask, double* frame) {
  const Size sn = static_cast<Size>(n);
  const Size total = sn * sn;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const Size p = idx / sn;
  const Size q = idx % sn;
  const Size pq = p * sn + q;
  const Size qp = q * sn + p;
  frame[pq] = 0.5 * (mask[pq] * rot[pq] + mask[qp] * rot[qp]);
}

__global__ void k_symmetrize_inplace(int n, double* mat) {
  const Size sn = static_cast<Size>(n);
  const Size total = sn * sn;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const Size p = idx / sn;
  const Size q = idx % sn;
  if (p < q) {
    const Size pq = p * sn + q;
    const Size qp = q * sn + p;
    const double avg = 0.5 * (mat[pq] + mat[qp]);
    mat[pq] = avg;
    mat[qp] = avg;
  }
}

__global__ void k_mul_init(Size n, const double* a, const double* b, double* out) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = a[idx] * b[idx];
}

__global__ void k_sym_slice_vo(int n, int occ, const double* field, double* out) {
  const int va = n - occ;
  const Size total = Size(va) * static_cast<Size>(occ);
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const Size s_occ = static_cast<Size>(occ);
  const Size s_n = static_cast<Size>(n);
  const Size a = idx / s_occ;
  const Size i = idx % s_occ;
  const Size p = s_occ + a;
  out[idx] = field[p * s_n + i] + field[i * s_n + p];
}

int cphf_exchange_tile_cuda(cublasHandle_t handle, int ql, int occ, int virt, const double* d_vo,
                            const double* d_oo, const double* d_vv, const double* d_trial,
                            double* d_kv, double* d_t1, double* d_t2, double* d_t3, double* d_out) {
  // kv[q, a, j] = sum_b vv[q, a, b] * trial[b, j]
  if (rm_gemm(handle, false, false, ql * virt, occ, virt, 1.0, d_vv, virt, d_trial, occ, 0.0, d_kv,
              occ) != CUBLAS_STATUS_SUCCESS) {
    return -4;
  }
  const Size n_vo = Size(ql) * static_cast<Size>(virt) * static_cast<Size>(occ);
  const Size n_oo = Size(ql) * static_cast<Size>(occ) * static_cast<Size>(occ);
  k_transpose_3d_012_to_102<<<grid_1d(n_vo), 256>>>(ql, virt, occ, d_kv, d_t1);
  k_transpose_3d_012_to_102<<<grid_1d(n_oo), 256>>>(ql, occ, occ, d_oo, d_t2);
  if (cudaGetLastError() != cudaSuccess) return -3;
  // out[a, i] -= sum_{q, j} kv[q, a, j] * oo[q, i, j]
  if (rm_gemm(handle, false, true, virt, occ, ql * occ, -1.0, d_t1, ql * occ, d_t2, ql * occ, 1.0,
              d_out, occ) != CUBLAS_STATUS_SUCCESS) {
    return -4;
  }
  // ki[q, i, j] = sum_b vo[q, b, i] * trial[b, j]
  const long long stride_vo = static_cast<long long>(virt) * static_cast<long long>(occ);
  const long long stride_oo = static_cast<long long>(occ) * static_cast<long long>(occ);
  if (rm_gemm_batched(handle, true, false, occ, occ, virt, 1.0, d_vo, occ, stride_vo, d_trial, occ,
                      0LL, 0.0, d_t2, occ, stride_oo, ql) != CUBLAS_STATUS_SUCCESS) {
    return -4;
  }
  k_transpose_3d_012_to_102<<<grid_1d(n_vo), 256>>>(ql, virt, occ, d_vo, d_t1);
  k_transpose_3d_012_to_102<<<grid_1d(n_oo), 256>>>(ql, occ, occ, d_t2, d_t3);
  if (cudaGetLastError() != cudaSuccess) return -3;
  // out[a, i] -= sum_{q, j} vo[q, a, j] * ki[q, i, j]
  if (rm_gemm(handle, false, true, virt, occ, ql * occ, -1.0, d_t1, ql * occ, d_t3, ql * occ, 1.0,
              d_out, occ) != CUBLAS_STATUS_SUCCESS) {
    return -4;
  }
  return 0;
}

int df_driver_exchange_tile_cuda(cublasHandle_t handle, int ql, int obs, double* d_buf1,
                                 double* d_buf2, double* d_buf3, const double* d_seed,
                                 double* d_field) {
  // d_buf1 currently holds df[ql, obs, obs].
  // d_buf2[q, p, j] = sum_i df[q, p, i] * seed[i, j]
  if (rm_gemm(handle, false, false, ql * obs, obs, obs, 1.0, d_buf1, obs, d_seed, obs, 0.0, d_buf2,
              obs) != CUBLAS_STATUS_SUCCESS) {
    return -4;
  }
  const Size n_df = Size(ql) * static_cast<Size>(obs) * static_cast<Size>(obs);
  // d_buf3 holds df_t[obs, ql, obs]; then d_buf1 is overwritten with weighted_t[obs, ql, obs].
  k_transpose_3d_012_to_102<<<grid_1d(n_df), 256>>>(ql, obs, obs, d_buf1, d_buf3);
  k_transpose_3d_012_to_102<<<grid_1d(n_df), 256>>>(ql, obs, obs, d_buf2, d_buf1);
  if (cudaGetLastError() != cudaSuccess) return -3;
  // field[p, r] -= sum_{q, j} weighted[q, p, j] * df[q, r, j]
  if (rm_gemm(handle, false, true, obs, obs, ql * obs, -1.0, d_buf1, ql * obs, d_buf3, ql * obs,
              1.0, d_field, obs) != CUBLAS_STATUS_SUCCESS) {
    return -4;
  }
  return 0;
}

__global__ void k_add_occ_trace(int ql, int n, int occ, double scale, const double* b,
                                double* charge_out) {
  const int a = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (a >= ql) return;
  const Size sn = static_cast<Size>(n);
  const double* ba = b + Size(a) * sn * sn;
  double tr = 0.0;
  for (int i = 0; i < occ; ++i) {
    tr += ba[Size(i) * sn + static_cast<Size>(i)];
  }
  charge_out[a] += scale * tr;
}

__global__ void k_extract_bo_and_oo(int ql, int n, int occ, const double* b, double* bo,
                                    double* oo) {
  const Size sn = static_cast<Size>(n);
  const Size so = static_cast<Size>(occ);
  const Size total = Size(ql) * sn * so;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const Size i = idx % so;
  const Size tmp = idx / so;
  const Size p = tmp % sn;
  const Size a = tmp / sn;
  const double val = b[(a * sn + p) * sn + i];
  bo[idx] = val;
  if (p < so) {
    oo[(a * so + p) * so + i] = val;
  }
}

__global__ void k_init_bar_db(int ql, int n, int occ, const double* charge,
                              const double* bar_dcharge, const double* p_mat, double* db) {
  const Size sn = static_cast<Size>(n);
  const Size total = Size(ql) * sn * sn;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const Size q_i = idx % sn;
  const Size tmp = idx / sn;
  const Size p_i = tmp % sn;
  const Size a = tmp / sn;
  double val = charge[a] * p_mat[p_i * sn + q_i];
  if (p_i == q_i && p_i < static_cast<Size>(occ)) {
    val += bar_dcharge[a];
  }
  db[idx] = val;
}

__global__ void k_sub_ex_from_db(int ql, int n, int occ, const double* ex, double* db) {
  const Size sn = static_cast<Size>(n);
  const Size so = static_cast<Size>(occ);
  const Size total = Size(ql) * sn * so;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const Size i = idx % so;
  const Size tmp = idx / so;
  const Size p_i = tmp % sn;
  const Size a = tmp / sn;
  db[(a * sn + p_i) * sn + i] -= ex[idx];
}

__global__ void k_symmetrize_3d_inplace(int ql, int m, double* arr) {
  const Size sm = static_cast<Size>(m);
  const Size sm2 = sm * sm;
  const Size total = Size(ql) * sm2;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const Size c = idx % sm;
  const Size tmp = idx / sm;
  const Size r = tmp % sm;
  const Size a = tmp / sm;
  if (r < c) {
    const Size rc = a * sm2 + r * sm + c;
    const Size cr = a * sm2 + c * sm + r;
    const double avg = 0.5 * (arr[rc] + arr[cr]);
    arr[rc] = avg;
    arr[cr] = avg;
  }
}

int tiled_gemm_cuda(cublasHandle_t handle, bool trans_a, bool trans_b, Size M, Size N, Size K,
                    const double* A, Size lda, const double* B, Size ldb, double* C, Size ldc,
                    Size budget_doubles, bool accumulate_c = false) {
  Size blk = static_cast<Size>(std::sqrt(static_cast<double>(budget_doubles / 3)));
  if (blk < 1) blk = 1;
  while (blk > 1 && 3 * blk * blk > budget_doubles) --blk;
  const Size m_tile = std::min(M, blk);
  const Size n_tile = std::min(N, blk);
  const Size k_tile = std::min(K, blk);
  if (m_tile * k_tile + k_tile * n_tile + m_tile * n_tile > budget_doubles) return -5;

  DeviceBuf d_a, d_b, d_c;
  if (d_a.alloc(m_tile * k_tile) != cudaSuccess || d_b.alloc(k_tile * n_tile) != cudaSuccess ||
      d_c.alloc(m_tile * n_tile) != cudaSuccess) {
    return -2;
  }
  std::vector<double> h_c_tile;
  if (accumulate_c) h_c_tile.resize(m_tile * n_tile);

  for (Size i0 = 0; i0 < M; i0 += m_tile) {
    const int il = static_cast<int>(std::min(m_tile, M - i0));
    for (Size j0 = 0; j0 < N; j0 += n_tile) {
      const int jl = static_cast<int>(std::min(n_tile, N - j0));
      for (Size k0 = 0; k0 < K; k0 += k_tile) {
        const int kl = static_cast<int>(std::min(k_tile, K - k0));
        const int a_rows = trans_a ? kl : il;
        const int a_cols = trans_a ? il : kl;
        const Size a_row0 = trans_a ? k0 : i0;
        const Size a_col0 = trans_a ? i0 : k0;
        if (cudaMemcpy2D(d_a.ptr, Size(a_cols) * sizeof(double), A + a_row0 * lda + a_col0,
                         lda * sizeof(double), Size(a_cols) * sizeof(double),
                         static_cast<Size>(a_rows), cudaMemcpyHostToDevice) != cudaSuccess) {
          return -3;
        }
        const int b_rows = trans_b ? jl : kl;
        const int b_cols = trans_b ? kl : jl;
        const Size b_row0 = trans_b ? j0 : k0;
        const Size b_col0 = trans_b ? k0 : j0;
        if (cudaMemcpy2D(d_b.ptr, Size(b_cols) * sizeof(double), B + b_row0 * ldb + b_col0,
                         ldb * sizeof(double), Size(b_cols) * sizeof(double),
                         static_cast<Size>(b_rows), cudaMemcpyHostToDevice) != cudaSuccess) {
          return -3;
        }
        const double beta = (k0 == 0) ? 0.0 : 1.0;
        if (rm_gemm(handle, trans_a, trans_b, il, jl, kl, 1.0, d_a.ptr, a_cols, d_b.ptr, b_cols,
                    beta, d_c.ptr, jl) != CUBLAS_STATUS_SUCCESS) {
          return -4;
        }
      }
      if (accumulate_c) {
        if (cudaMemcpy(h_c_tile.data(), d_c.ptr, Size(il) * Size(jl) * sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          return -3;
        }
        for (int i = 0; i < il; ++i) {
          double* dst = C + (i0 + Size(i)) * ldc + j0;
          const double* src = h_c_tile.data() + Size(i) * Size(jl);
          for (int j = 0; j < jl; ++j) dst[j] += src[j];
        }
      } else {
        if (cudaMemcpy2D(C + i0 * ldc + j0, ldc * sizeof(double), d_c.ptr,
                         Size(jl) * sizeof(double), Size(jl) * sizeof(double),
                         static_cast<Size>(il), cudaMemcpyDeviceToHost) != cudaSuccess) {
          return -3;
        }
      }
    }
  }
  return 0;
}

#endif

}  // namespace

extern "C" {

#ifdef GQC_SINK_HOST_REFERENCE

int gqc_f12_cphf_whiten_3c_host_reference(int aux_out, int aux_in, std::int64_t ncols,
                                          std::uint64_t budget_bytes, const double* whitener,
                                          const double* raw, double* out) {
  try {
    if (aux_out < 1 || aux_in < 1 || ncols < 1 || !whitener || !raw || !out) {
      return kErrInvalidArg;
    }
    const Size s_aout = static_cast<Size>(aux_out);
    const Size s_ain = static_cast<Size>(aux_in);
    const Size s_ncols = static_cast<Size>(ncols);
    Size w_elems = 0, r_elems = 0, out_elems = 0;
    if (!checked_mul(s_aout, s_ain, w_elems) || !checked_mul(s_ain, s_ncols, r_elems) ||
        !checked_mul(s_aout, s_ncols, out_elems)) {
      return kErrInvalidArg;
    }
    const WhitenTiles tiles = compute_whiten_tiles(s_aout, s_ain, s_ncols, budget_bytes);
    if (tiles.a_tile == 0 || tiles.b_tile == 0 || tiles.c_tile == 0) return kErrBudget;
    if (!check_finite(whitener, w_elems) || !check_finite(raw, r_elems)) return kErrNonFinite;
    for (Size c0 = 0; c0 < s_ncols; c0 += tiles.c_tile) {
      const Size cl = std::min(tiles.c_tile, s_ncols - c0);
      for (Size a0 = 0; a0 < s_aout; a0 += tiles.a_tile) {
        const Size al = std::min(tiles.a_tile, s_aout - a0);
        for (Size b0 = 0; b0 < s_ain; b0 += tiles.b_tile) {
          const Size bl = std::min(tiles.b_tile, s_ain - b0);
          const double beta = (b0 == 0) ? 0.0 : 1.0;
          rm_gemm_host(false, false, al, cl, bl, 1.0, whitener + a0 * s_ain + b0, s_ain,
                       raw + b0 * s_ncols + c0, s_ncols, beta, out + a0 * s_ncols + c0, s_ncols);
        }
      }
    }
    if (!check_finite(out, out_elems)) return kErrNonFinite;
    return kOk;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_orbital_rot_metric_host_reference(int nao, int nmo, std::uint64_t budget_bytes,
                                              const double* coefficients, const double* frame_bar,
                                              const double* mask, double* rotation_out,
                                              double* overlap_bar_out) {
  try {
    (void)budget_bytes;
    if (nao < 1 || nmo < 1 || nmo > nao || !coefficients || !frame_bar) return kErrInvalidArg;
    if (!rotation_out && !overlap_bar_out) return kErrInvalidArg;
    if (overlap_bar_out && !mask) return kErrInvalidArg;

    const Size sa = static_cast<Size>(nao);
    const Size sm = static_cast<Size>(nmo);
    std::vector<double> rot_buf;
    double* rot = rotation_out;
    if (!rot) {
      rot_buf.resize(sm * sm);
      rot = rot_buf.data();
    }
    // rotation[p, q] = sum_a C[a, p] * frame_bar[a, q]
    rm_gemm_host(true, false, sm, sm, sa, 1.0, coefficients, sm, frame_bar, sm, 0.0, rot, sm);

    if (overlap_bar_out) {
      std::vector<double> frame(sm * sm);
      for (Size p = 0; p < sm; ++p) {
        for (Size q = 0; q < sm; ++q) {
          const Size pq = p * sm + q;
          const Size qp = q * sm + p;
          frame[pq] = 0.5 * (mask[pq] * rot[pq] + mask[qp] * rot[qp]);
        }
      }
      std::vector<double> rows(sa * sm);
      rm_gemm_host(false, false, sa, sm, sm, 1.0, coefficients, sm, frame.data(), sm, 0.0,
                   rows.data(), sm);
      rm_gemm_host(false, true, sa, sa, sm, 1.0, rows.data(), sm, coefficients, sm, 0.0,
                   overlap_bar_out, sa);
      for (Size u = 0; u < sa; ++u) {
        for (Size v = u + 1; v < sa; ++v) {
          const Size uv = u * sa + v;
          const Size vu = v * sa + u;
          const double avg = 0.5 * (overlap_bar_out[uv] + overlap_bar_out[vu]);
          overlap_bar_out[uv] = avg;
          overlap_bar_out[vu] = avg;
        }
      }
    }
    if ((rotation_out && !check_finite(rotation_out, sm * sm)) ||
        (overlap_bar_out && !check_finite(overlap_bar_out, sa * sa))) {
      return kErrNonFinite;
    }
    return kOk;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_cphf_action_host_reference(int is_rhf, int oa, int ob, int obs, int aux,
                                       std::uint64_t budget_bytes, const double* vo_a,
                                       const double* oo_a, const double* vv_a, const double* gap_a,
                                       const double* trial_a, double* out_a, const double* vo_b,
                                       const double* oo_b, const double* vv_b, const double* gap_b,
                                       const double* trial_b, double* out_b) {
  try {
    if (obs < 2 || aux < 1) return kErrInvalidArg;
    const bool rhf = (is_rhf != 0);
    const int eff_oa = oa;
    const int eff_ob = rhf ? 0 : ob;
    if (rhf) {
      if (eff_oa < 1 || eff_oa >= obs || !vo_a || !oo_a || !vv_a || !gap_a || !trial_a || !out_a) {
        return kErrInvalidArg;
      }
    } else {
      if (eff_oa < 0 || eff_ob < 0 || eff_oa + eff_ob == 0 || eff_oa >= obs || eff_ob >= obs) {
        return kErrInvalidArg;
      }
      if (eff_oa > 0 && (!vo_a || !oo_a || !vv_a || !gap_a || !trial_a || !out_a)) {
        return kErrInvalidArg;
      }
      if (eff_ob > 0 && (!vo_b || !oo_b || !vv_b || !gap_b || !trial_b || !out_b)) {
        return kErrInvalidArg;
      }
    }

    const int va = (eff_oa > 0) ? (obs - eff_oa) : 0;
    const int vb = (eff_ob > 0) ? (obs - eff_ob) : 0;
    const Size n_vo_a = Size(va) * static_cast<Size>(eff_oa);
    const Size n_vo_b = Size(vb) * static_cast<Size>(eff_ob);
    if (!fits_int(n_vo_a) || !fits_int(n_vo_b)) return kErrInvalidArg;

    const int q_tile = compute_cphf_action_q_tile(eff_oa, eff_ob, obs, aux, budget_bytes);
    if (q_tile < 1) return kErrBudget;

    const Size n_oo_a = Size(eff_oa) * static_cast<Size>(eff_oa);
    const Size n_oo_b = Size(eff_ob) * static_cast<Size>(eff_ob);
    const Size n_vv_a = Size(va) * static_cast<Size>(va);
    const Size n_vv_b = Size(vb) * static_cast<Size>(vb);

    if (eff_oa > 0) {
      for (Size idx = 0; idx < n_vo_a; ++idx) out_a[idx] = gap_a[idx] * trial_a[idx];
    }
    if (eff_ob > 0) {
      for (Size idx = 0; idx < n_vo_b; ++idx) out_b[idx] = gap_b[idx] * trial_b[idx];
    }

    const double j_scale = rhf ? 4.0 : 2.0;
    std::vector<double> charge(static_cast<Size>(q_tile));

    for (int q0 = 0; q0 < aux; q0 += q_tile) {
      const int ql = std::min(q_tile, aux - q0);
      const Size sql = static_cast<Size>(ql);
      const Size sq0 = static_cast<Size>(q0);
      std::fill_n(charge.data(), sql, 0.0);
      if (eff_oa > 0) {
        rm_gemm_host(false, false, sql, 1, n_vo_a, 1.0, vo_a + sq0 * n_vo_a, n_vo_a, trial_a, 1,
                     1.0, charge.data(), 1);
      }
      if (eff_ob > 0) {
        rm_gemm_host(false, false, sql, 1, n_vo_b, 1.0, vo_b + sq0 * n_vo_b, n_vo_b, trial_b, 1,
                     1.0, charge.data(), 1);
      }
      if (eff_oa > 0) {
        rm_gemm_host(true, false, n_vo_a, 1, sql, j_scale, vo_a + sq0 * n_vo_a, n_vo_a,
                     charge.data(), 1, 1.0, out_a, 1);
        cphf_exchange_spin_host(eff_oa, va, ql, vo_a + sq0 * n_vo_a, oo_a + sq0 * n_oo_a,
                                vv_a + sq0 * n_vv_a, trial_a, out_a);
      }
      if (eff_ob > 0) {
        rm_gemm_host(true, false, n_vo_b, 1, sql, j_scale, vo_b + sq0 * n_vo_b, n_vo_b,
                     charge.data(), 1, 1.0, out_b, 1);
        cphf_exchange_spin_host(eff_ob, vb, ql, vo_b + sq0 * n_vo_b, oo_b + sq0 * n_oo_b,
                                vv_b + sq0 * n_vv_b, trial_b, out_b);
      }
    }
    if ((eff_oa > 0 && !check_finite(out_a, n_vo_a)) ||
        (eff_ob > 0 && !check_finite(out_b, n_vo_b))) {
      return kErrNonFinite;
    }
    return kOk;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_df_canonical_driver_host_reference(int oa, int ob, int obs, int aux,
                                               std::uint64_t budget_bytes, const double* df_a,
                                               const double* fock_seed_a, double* out_a,
                                               const double* df_b, const double* fock_seed_b,
                                               double* out_b) {
  try {
    if (obs < 2 || aux < 1 || oa < 0 || ob < 0 || oa + ob == 0 || oa >= obs || ob >= obs) {
      return kErrInvalidArg;
    }
    if (oa > 0 && (!df_a || !fock_seed_a || !out_a)) return kErrInvalidArg;
    if (ob > 0 && (!df_b || !fock_seed_b || !out_b)) return kErrInvalidArg;
    if (oa == 0 && ((df_a != nullptr) != (fock_seed_a != nullptr))) return kErrInvalidArg;
    if (ob == 0 && ((df_b != nullptr) != (fock_seed_b != nullptr))) return kErrInvalidArg;

    const bool has_charge_a = (df_a != nullptr && fock_seed_a != nullptr);
    const bool has_charge_b = (df_b != nullptr && fock_seed_b != nullptr);

    const Size n = static_cast<Size>(obs);
    const Size n2 = n * n;
    if (!fits_int(n2)) return kErrInvalidArg;

    const int q_tile =
        compute_df_driver_q_tile(oa, ob, has_charge_a, has_charge_b, obs, aux, budget_bytes);
    if (q_tile < 1) return kErrBudget;

    std::vector<double> field_a(oa > 0 ? n2 : 0, 0.0);
    std::vector<double> field_b(ob > 0 ? n2 : 0, 0.0);
    std::vector<double> charge(static_cast<Size>(q_tile));

    for (int q0 = 0; q0 < aux; q0 += q_tile) {
      const int ql = std::min(q_tile, aux - q0);
      const Size sql = static_cast<Size>(ql);
      const Size sq0 = static_cast<Size>(q0);
      std::fill_n(charge.data(), sql, 0.0);
      if (has_charge_a) {
        rm_gemm_host(false, false, sql, 1, n2, 1.0, df_a + sq0 * n2, n2, fock_seed_a, 1, 1.0,
                     charge.data(), 1);
      }
      if (has_charge_b) {
        rm_gemm_host(false, false, sql, 1, n2, 1.0, df_b + sq0 * n2, n2, fock_seed_b, 1, 1.0,
                     charge.data(), 1);
      }
      if (oa > 0) {
        rm_gemm_host(true, false, n2, 1, sql, 1.0, df_a + sq0 * n2, n2, charge.data(), 1, 1.0,
                     field_a.data(), 1);
        df_driver_exchange_spin_host(obs, ql, df_a + sq0 * n2, fock_seed_a, field_a.data());
      }
      if (ob > 0) {
        rm_gemm_host(true, false, n2, 1, sql, 1.0, df_b + sq0 * n2, n2, charge.data(), 1, 1.0,
                     field_b.data(), 1);
        df_driver_exchange_spin_host(obs, ql, df_b + sq0 * n2, fock_seed_b, field_b.data());
      }
    }

    auto finish_vo = [&](int occ, const std::vector<double>& field, double* out) {
      const int va = obs - occ;
      for (int a = 0; a < va; ++a) {
        const Size p = static_cast<Size>(occ + a);
        for (int i = 0; i < occ; ++i) {
          const Size si = static_cast<Size>(i);
          out[Size(a) * static_cast<Size>(occ) + si] = field[p * n + si] + field[si * n + p];
        }
      }
    };
    if (oa > 0) {
      finish_vo(oa, field_a, out_a);
      if (!check_finite(out_a, Size(obs - oa) * static_cast<Size>(oa))) return kErrNonFinite;
    }
    if (ob > 0) {
      finish_vo(ob, field_b, out_b);
      if (!check_finite(out_b, Size(obs - ob) * static_cast<Size>(ob))) return kErrNonFinite;
    }
    return kOk;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_df_nuclear_weights_host_reference(
    int oa, int ob, int obs, int aux, int nao, int include_direct, std::uint64_t budget_bytes,
    const double* coeff_a, const double* coeff_b, const double* occ_e_a, const double* occ_e_b,
    const double* df_a, const double* df_b, const double* fock_seed_a, const double* fock_seed_b,
    const double* overlap_seed_a, const double* overlap_seed_b, const double* whitener,
    const double* z_a, const double* z_b, const double* direct_overlap_ao,
    const double* direct_hcore_ao, const double* direct_ao_coulomb,
    const double* direct_coulomb_metric, const double* frame_alpha, const double* frame_beta,
    double* overlap_ao_out, double* hcore_ao_out, double* ao_coulomb_out,
    double* metric_coulomb_out) {
  try {
    if (obs < 2 || aux < 1 || nao < obs || oa < 0 || ob < 0 || oa + ob == 0 || oa >= obs ||
        ob >= obs) {
      return kErrInvalidArg;
    }
    if (!coeff_a || !coeff_b || !df_a || !df_b || !fock_seed_a || !fock_seed_b || !overlap_seed_a ||
        !overlap_seed_b || !whitener || !overlap_ao_out || !hcore_ao_out || !ao_coulomb_out ||
        !metric_coulomb_out) {
      return kErrInvalidArg;
    }
    if (oa > 0 && (!occ_e_a || !z_a)) return kErrInvalidArg;
    if (ob > 0 && (!occ_e_b || !z_b)) return kErrInvalidArg;
    const bool inc_dir = (include_direct != 0);
    if (inc_dir && (!direct_overlap_ao || !direct_hcore_ao || !direct_ao_coulomb ||
                    !direct_coulomb_metric || !frame_alpha || !frame_beta)) {
      return kErrInvalidArg;
    }

    NuclearWeightsSetup st = prepare_nuclear_weights_setup(
        oa, ob, obs, nao, coeff_a, coeff_b, occ_e_a, occ_e_b, df_a, df_b, fock_seed_a, fock_seed_b,
        overlap_seed_a, overlap_seed_b, z_a, z_b);

    const int n_eff = st.n_eff;
    const bool same_spin_all = st.same_spin_all;
    const Size n = static_cast<Size>(obs);
    const Size n2 = n * n;
    const Size ne = static_cast<Size>(n_eff);
    const Size ne2 = ne * ne;
    if (!fits_int(n2) || !fits_int(ne2)) return kErrInvalidArg;

    const int q_tile =
        compute_df_nuclear_weights_q_tile(oa, ob, obs, aux, n_eff, same_spin_all, budget_bytes);
    if (q_tile < 1) return kErrBudget;

    const Size snao = static_cast<Size>(nao);
    const Size s_aux = static_cast<Size>(aux);
    const Size sq = static_cast<Size>(q_tile);

    struct SpinView {
      int occ;
      const double* coeff;
      const double* df;
      const std::vector<double>* p;
      const std::vector<double>* psym;
      std::vector<double>* bar_sm;
    };
    SpinView spins[2] = {
        {oa, coeff_a, df_a, &st.p_a, &st.psym_a, &st.bar_sm_a},
        {ob, coeff_b, df_b, &st.p_b, &st.psym_b, &st.bar_sm_b},
    };
    const int n_loop_spins = same_spin_all ? 1 : 2;
    const double spin_scale = same_spin_all ? 2.0 : 1.0;

    // Phase 1: charge[q] and bar_dcharge[q].
    std::vector<double> charge(s_aux, 0.0);
    std::vector<double> bar_dcharge(s_aux, 0.0);
    for (int q0 = 0; q0 < aux; q0 += q_tile) {
      const int ql = std::min(q_tile, aux - q0);
      const Size sql = static_cast<Size>(ql);
      const Size sq0 = static_cast<Size>(q0);
      for (int s = 0; s < n_loop_spins; ++s) {
        const SpinView& sp = spins[s];
        rm_gemm_host(false, false, sql, 1, n2, spin_scale, sp.df + sq0 * n2, n2, sp.p->data(), 1,
                     1.0, bar_dcharge.data() + sq0, 1);
        if (sp.occ > 0) {
          for (int a = 0; a < ql; ++a) {
            const double* ba = sp.df + (sq0 + Size(a)) * n2;
            double tr = 0.0;
            for (int i = 0; i < sp.occ; ++i) tr += ba[Size(i) * n + static_cast<Size>(i)];
            charge[sq0 + Size(a)] += spin_scale * tr;
          }
        }
      }
    }

    // Phase 2 (fused) + Phase 3: bar_sm_oo, bar_db, bar_M, ao_db_chunk, bar_dr.
    std::vector<double> sm_oo[2];
    if (oa > 0) sm_oo[0].assign(Size(oa) * Size(oa), 0.0);
    if (ob > 0 && !same_spin_all) sm_oo[1].assign(Size(ob) * Size(ob), 0.0);

    const int b_tile = static_cast<int>(std::min<Size>(sq, ne2));
    const int t_tile = static_cast<int>(std::min<Size>(sq, ne2));
    const Size max_occ = static_cast<Size>(std::max(oa, ob));

    std::vector<double> bo_buf(sq * n * max_occ);
    std::vector<double> oo_buf(sq * max_occ * max_occ);
    std::vector<double> pb_buf(sq * n * max_occ);
    std::vector<double> ex_buf(n * max_occ);
    std::vector<double> db_buf(sq * n2);
    std::vector<double> tmp_ao(sq * n * ne);
    std::vector<double> ao_chunk(sq * ne2);
    std::vector<double> dr_tile(Size(t_tile) * ne2);
    std::vector<double> bar_m(s_aux * s_aux, 0.0);

    for (int a0 = 0; a0 < aux; a0 += q_tile) {
      const int ql = std::min(q_tile, aux - a0);
      const Size sql = static_cast<Size>(ql);
      const Size sa0 = static_cast<Size>(a0);

      for (int s = 0; s < n_loop_spins; ++s) {
        const SpinView& sp = spins[s];
        const int occ = sp.occ;
        const Size so = static_cast<Size>(occ);

        if (occ > 0) {
          const Size no = n * so;
          const Size o2 = so * so;
          for (int a = 0; a < ql; ++a) {
            const double* ba = sp.df + (sa0 + Size(a)) * n2;
            double* bo_a = bo_buf.data() + Size(a) * no;
            double* oo_a = oo_buf.data() + Size(a) * o2;
            for (Size p = 0; p < n; ++p) {
              std::memcpy(bo_a + p * so, ba + p * n, so * sizeof(double));
              if (p < so) {
                std::memcpy(oo_a + p * so, ba + p * n, so * sizeof(double));
              }
            }
          }
          // s_mc: sm_oo += -bar_dcharge * B_oo
          rm_gemm_host(true, false, o2, 1, sql, -1.0, oo_buf.data(), o2, bar_dcharge.data() + sa0,
                       1, 1.0, sm_oo[s].data(), 1);
          // pb[a, q, i] = sum_p P[p, q] * B_o[a, p, i]
          for (int a = 0; a < ql; ++a) {
            rm_gemm_host(true, false, n, so, n, 1.0, sp.p->data(), n, bo_buf.data() + Size(a) * no,
                         so, 0.0, pb_buf.data() + Size(a) * no, so);
          }
          // s_ex: sm_oo[i, j] += sum_{a, q} pb[a, q, i] * B_o[a, q, j]
          rm_gemm_host(true, false, so, so, sql * n, 1.0, pb_buf.data(), so, bo_buf.data(), so, 1.0,
                       sm_oo[s].data(), so);
        }

        // Build bar_db[a, p, q]
        const double* p_data = sp.p->data();
        for (int a = 0; a < ql; ++a) {
          const double ca = charge[sa0 + Size(a)];
          const double bdc = bar_dcharge[sa0 + Size(a)];
          double* db_a = db_buf.data() + Size(a) * n2;
          for (Size idx = 0; idx < n2; ++idx) db_a[idx] = ca * p_data[idx];
          if (occ > 0) {
            for (int i = 0; i < occ; ++i) db_a[Size(i) * n + Size(i)] += bdc;
            rm_gemm_host(false, false, n, so, n, 1.0, sp.psym->data(), n,
                         bo_buf.data() + Size(a) * n * so, so, 0.0, ex_buf.data(), so);
            for (Size p = 0; p < n; ++p) {
              double* row = db_a + p * n;
              const double* ex_row = ex_buf.data() + p * so;
              for (Size i = 0; i < so; ++i) row[i] -= ex_row[i];
            }
          }
        }

        // Accumulate bar_M[a0:a0+ql, :]
        for (int b0 = 0; b0 < aux; b0 += b_tile) {
          const int bl = std::min(b_tile, aux - b0);
          rm_gemm_host(false, true, sql, static_cast<Size>(bl), n2, -0.5 * spin_scale,
                       db_buf.data(), n2, sp.df + Size(b0) * n2, n2, 1.0,
                       bar_m.data() + sa0 * s_aux + Size(b0), s_aux);
        }

        // Back-project bar_db to AO space: ao_chunk += spin_scale * C @ bar_db @ C^T
        rm_gemm_host(false, true, sql * n, ne, n, 1.0, db_buf.data(), n, sp.coeff, n, 0.0,
                     tmp_ao.data(), ne);
        const double beta_ao = (s == 0) ? 0.0 : 1.0;
        for (int a = 0; a < ql; ++a) {
          rm_gemm_host(false, false, ne, ne, n, spin_scale, sp.coeff, n,
                       tmp_ao.data() + Size(a) * n * ne, ne, beta_ao,
                       ao_chunk.data() + Size(a) * ne2, ne);
        }
      }

      // Symmetrize ao_chunk slices in-place
      for (int a = 0; a < ql; ++a) {
        symmetrize_square_host(ne, ao_chunk.data() + Size(a) * ne2);
      }

      // Accumulate bar_dr[t0:t0+tl, :n_eff, :n_eff] += W[a0:a0+ql, t0:t0+tl]^T @ ao_chunk
      for (int t0 = 0; t0 < aux; t0 += t_tile) {
        const int tl = std::min(t_tile, aux - t0);
        rm_gemm_host(true, false, static_cast<Size>(tl), ne2, sql, 1.0,
                     whitener + sa0 * s_aux + Size(t0), s_aux, ao_chunk.data(), ne2, 0.0,
                     dr_tile.data(), ne2);
        accumulate_dr_tile_host(a0, t0, tl, nao, n_eff, inc_dir, direct_ao_coulomb, dr_tile.data(),
                                ao_coulomb_out);
      }
    }

    // Finish bar_sm[:occ, :occ]
    auto add_sm_oo = [&](int occ, const std::vector<double>& src, std::vector<double>& dst) {
      const Size so = static_cast<Size>(occ);
      for (Size i = 0; i < so; ++i) {
        for (Size j = 0; j < so; ++j) {
          dst[i * n + j] += src[i * so + j];
        }
      }
    };
    if (oa > 0) {
      add_sm_oo(oa, sm_oo[0], st.bar_sm_a);
      if (same_spin_all) add_sm_oo(ob, sm_oo[0], st.bar_sm_b);
    }
    if (ob > 0 && !same_spin_all) {
      add_sm_oo(ob, sm_oo[1], st.bar_sm_b);
    }

    // Phase 4a: bar_dh and bar_ds (2D AO back-projections)
    std::memset(hcore_ao_out, 0, snao * snao * sizeof(double));
    std::memset(overlap_ao_out, 0, snao * snao * sizeof(double));
    std::vector<double> tmp_cm(ne * n);
    for (int s = 0; s < 2; ++s) {
      const SpinView& sp = spins[s];
      rm_gemm_host(false, false, ne, n, n, 1.0, sp.coeff, n, sp.p->data(), n, 0.0, tmp_cm.data(),
                   n);
      rm_gemm_host(false, true, ne, ne, n, 1.0, tmp_cm.data(), n, sp.coeff, n, 1.0, hcore_ao_out,
                   snao);

      rm_gemm_host(false, false, ne, n, n, 1.0, sp.coeff, n, sp.bar_sm->data(), n, 0.0,
                   tmp_cm.data(), n);
      rm_gemm_host(false, true, ne, ne, n, 1.0, tmp_cm.data(), n, sp.coeff, n, 1.0, overlap_ao_out,
                   snao);
    }
    symmetrize_square_host(snao, hcore_ao_out);
    symmetrize_square_host(snao, overlap_ao_out);

    // Phase 4b: bar_dJ = 0.5 * (W^T @ bar_M @ W + (W^T @ bar_M @ W)^T)
    std::vector<double> tmp_m(s_aux * s_aux);
    rm_gemm_host(false, false, s_aux, s_aux, s_aux, 1.0, bar_m.data(), s_aux, whitener, s_aux, 0.0,
                 tmp_m.data(), s_aux);
    rm_gemm_host(true, false, s_aux, s_aux, s_aux, 1.0, whitener, s_aux, tmp_m.data(), s_aux, 0.0,
                 metric_coulomb_out, s_aux);
    symmetrize_square_host(s_aux, metric_coulomb_out);

    const Size snao2 = snao * snao;
    const Size sq2 = s_aux * s_aux;
    if (inc_dir) {
      for (Size idx = 0; idx < snao2; ++idx) {
        hcore_ao_out[idx] += direct_hcore_ao[idx];
        overlap_ao_out[idx] += direct_overlap_ao[idx] + frame_alpha[idx] + frame_beta[idx];
      }
      for (Size idx = 0; idx < sq2; ++idx) {
        metric_coulomb_out[idx] += direct_coulomb_metric[idx];
      }
    }
    if (!check_finite(overlap_ao_out, snao2) || !check_finite(hcore_ao_out, snao2) ||
        !check_finite(ao_coulomb_out, s_aux * snao2) || !check_finite(metric_coulomb_out, sq2)) {
      return kErrNonFinite;
    }
    return kOk;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

#else

int gqc_f12_cphf_whiten_3c_cuda(int aux_out, int aux_in, std::int64_t ncols,
                                std::uint64_t budget_bytes, const double* whitener,
                                const double* raw, double* out) {
  try {
    if (aux_out < 1 || aux_in < 1 || ncols < 1 || !whitener || !raw || !out) {
      return kErrInvalidArg;
    }

    const Size s_ncols = static_cast<Size>(ncols);
    const Size s_aout = static_cast<Size>(aux_out);
    const Size s_ain = static_cast<Size>(aux_in);
    Size w_elems = 0, r_elems = 0, out_elems = 0;
    if (!checked_mul(s_aout, s_ain, w_elems) || !checked_mul(s_ain, s_ncols, r_elems) ||
        !checked_mul(s_aout, s_ncols, out_elems)) {
      return kErrInvalidArg;
    }

    const WhitenTiles tiles = compute_whiten_tiles(s_aout, s_ain, s_ncols, budget_bytes);
    if (tiles.a_tile == 0 || tiles.b_tile == 0 || tiles.c_tile == 0) return kErrBudget;
    if (!check_finite(whitener, w_elems) || !check_finite(raw, r_elems)) return kErrNonFinite;
    const Size a_tile = tiles.a_tile;
    const Size b_tile = tiles.b_tile;
    const Size c_tile = tiles.c_tile;

    CublasGuard cublas;
    if (cublas.init() != CUBLAS_STATUS_SUCCESS) return kErrCublas;

    DeviceBuf d_w, d_raw, d_out;
    if (d_w.alloc(a_tile * b_tile) != cudaSuccess || d_raw.alloc(b_tile * c_tile) != cudaSuccess ||
        d_out.alloc(a_tile * c_tile) != cudaSuccess) {
      return kErrOom;
    }

    const bool full_c = (c_tile == s_ncols);
    const bool full_b = (b_tile == s_ain);

    for (Size c0 = 0; c0 < s_ncols; c0 += c_tile) {
      const int cl = static_cast<int>(std::min(c_tile, s_ncols - c0));
      for (Size a0 = 0; a0 < s_aout; a0 += a_tile) {
        const int al = static_cast<int>(std::min(a_tile, s_aout - a0));
        for (Size b0 = 0; b0 < s_ain; b0 += b_tile) {
          const int bl = static_cast<int>(std::min(b_tile, s_ain - b0));
          if (full_b) {
            if (cudaMemcpy(d_w.ptr, whitener + a0 * s_ain, Size(al) * Size(bl) * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
              return kErrCuda;
            }
          } else {
            if (cudaMemcpy2D(d_w.ptr, Size(bl) * sizeof(double), whitener + a0 * s_ain + b0,
                             s_ain * sizeof(double), Size(bl) * sizeof(double),
                             static_cast<Size>(al), cudaMemcpyHostToDevice) != cudaSuccess) {
              return kErrCuda;
            }
          }
          if (full_c) {
            if (cudaMemcpy(d_raw.ptr, raw + b0 * s_ncols, Size(bl) * Size(cl) * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
              return kErrCuda;
            }
          } else {
            if (cudaMemcpy2D(d_raw.ptr, Size(cl) * sizeof(double), raw + b0 * s_ncols + c0,
                             s_ncols * sizeof(double), Size(cl) * sizeof(double),
                             static_cast<Size>(bl), cudaMemcpyHostToDevice) != cudaSuccess) {
              return kErrCuda;
            }
          }
          const double beta = (b0 == 0) ? 0.0 : 1.0;
          if (rm_gemm(cublas.handle, false, false, al, cl, bl, 1.0, d_w.ptr, bl, d_raw.ptr, cl,
                      beta, d_out.ptr, cl) != CUBLAS_STATUS_SUCCESS) {
            return kErrCublas;
          }
        }
        if (full_c) {
          if (cudaMemcpy(out + a0 * s_ncols, d_out.ptr, Size(al) * Size(cl) * sizeof(double),
                         cudaMemcpyDeviceToHost) != cudaSuccess) {
            return kErrCuda;
          }
        } else {
          if (cudaMemcpy2D(out + a0 * s_ncols + c0, s_ncols * sizeof(double), d_out.ptr,
                           Size(cl) * sizeof(double), Size(cl) * sizeof(double),
                           static_cast<Size>(al), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return kErrCuda;
          }
        }
      }
    }
    if (!check_finite(out, out_elems)) return kErrNonFinite;
    return kOk;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_orbital_rot_metric_cuda(int nao, int nmo, std::uint64_t budget_bytes,
                                    const double* coefficients, const double* frame_bar,
                                    const double* mask, double* rotation_out,
                                    double* overlap_bar_out) {
  try {
    if (nao < 1 || nmo < 1 || nmo > nao || !coefficients || !frame_bar) return kErrInvalidArg;
    if (!rotation_out && !overlap_bar_out) return kErrInvalidArg;
    if (overlap_bar_out && !mask) return kErrInvalidArg;

    const Size sa = static_cast<Size>(nao);
    const Size sm = static_cast<Size>(nmo);
    const Size budget_doubles = effective_budget_bytes(budget_bytes) / sizeof(double);

    CublasGuard cublas;
    if (cublas.init() != CUBLAS_STATUS_SUCCESS) return kErrCublas;

    const Size full_need = overlap_bar_out ? (2 * sa * sm + 2 * sm * sm + sa * sa)
                                           : (2 * sa * sm + sm * sm);

    if (full_need <= budget_doubles) {
      DeviceBuf d_c, d_tmp, d_rot, d_frame, d_ov;
      if (d_c.alloc(sa * sm) != cudaSuccess || d_tmp.alloc(sa * sm) != cudaSuccess ||
          d_rot.alloc(sm * sm) != cudaSuccess) {
        return kErrOom;
      }
      if (cudaMemcpy(d_c.ptr, coefficients, sa * sm * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess ||
          cudaMemcpy(d_tmp.ptr, frame_bar, sa * sm * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess) {
        return kErrCuda;
      }
      if (rm_gemm(cublas.handle, true, false, nmo, nmo, nao, 1.0, d_c.ptr, nmo, d_tmp.ptr, nmo, 0.0,
                  d_rot.ptr, nmo) != CUBLAS_STATUS_SUCCESS) {
        return kErrCublas;
      }
      if (rotation_out) {
        if (cudaMemcpy(rotation_out, d_rot.ptr, sm * sm * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
          return kErrCuda;
        }
      }
      if (overlap_bar_out) {
        if (d_frame.alloc(sm * sm) != cudaSuccess || d_ov.alloc(sa * sa) != cudaSuccess) {
          return kErrOom;
        }
        // Upload mask into d_ov (since sa >= sm => sa*sa >= sm*sm), write frame into d_frame.
        if (cudaMemcpy(d_ov.ptr, mask, sm * sm * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
          return kErrCuda;
        }
        k_mask_symmetrize<<<grid_1d(sm * sm), 256>>>(nmo, d_rot.ptr, d_ov.ptr, d_frame.ptr);
        if (cudaGetLastError() != cudaSuccess) return kErrCuda;
        // rows = C @ frame (stored in d_tmp)
        if (rm_gemm(cublas.handle, false, false, nao, nmo, nmo, 1.0, d_c.ptr, nmo, d_frame.ptr, nmo,
                    0.0, d_tmp.ptr, nmo) != CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
        // weight = rows @ C^T (stored in d_ov)
        if (rm_gemm(cublas.handle, false, true, nao, nao, nmo, 1.0, d_tmp.ptr, nmo, d_c.ptr, nmo,
                    0.0, d_ov.ptr, nao) != CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
        k_symmetrize_inplace<<<grid_1d(sa * sa), 256>>>(nao, d_ov.ptr);
        if (cudaGetLastError() != cudaSuccess) return kErrCuda;
        if (cudaMemcpy(overlap_bar_out, d_ov.ptr, sa * sa * sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          return kErrCuda;
        }
      }
      if ((rotation_out && !check_finite(rotation_out, sm * sm)) ||
          (overlap_bar_out && !check_finite(overlap_bar_out, sa * sa))) {
        return kErrNonFinite;
      }
      return kOk;
    }

    // Budget-bounded tiled fallback for very large (nao, nmo).
    const Size base_need = overlap_bar_out ? (2 * sm * sm) : (sm * sm);
    if (base_need >= budget_doubles) return kErrBudget;
    const Size rem = budget_doubles - base_need;

    DeviceBuf d_rot;
    if (d_rot.alloc(sm * sm) != cudaSuccess) return kErrOom;

    const Size a_tile_rot = std::min<Size>(sa, rem / (2 * sm));
    if (a_tile_rot < 1) return kErrBudget;
    {
      DeviceBuf d_ca, d_ba;
      if (d_ca.alloc(a_tile_rot * sm) != cudaSuccess ||
          d_ba.alloc(a_tile_rot * sm) != cudaSuccess) {
        return kErrOom;
      }
      for (Size a0 = 0; a0 < sa; a0 += a_tile_rot) {
        const int al = static_cast<int>(std::min(a_tile_rot, sa - a0));
        if (cudaMemcpy(d_ca.ptr, coefficients + a0 * sm, Size(al) * sm * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_ba.ptr, frame_bar + a0 * sm, Size(al) * sm * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return kErrCuda;
        }
        const double beta = (a0 == 0) ? 0.0 : 1.0;
        if (rm_gemm(cublas.handle, true, false, nmo, nmo, al, 1.0, d_ca.ptr, nmo, d_ba.ptr, nmo,
                    beta, d_rot.ptr, nmo) != CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
      }
    }
    if (rotation_out) {
      if (cudaMemcpy(rotation_out, d_rot.ptr, sm * sm * sizeof(double), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        return kErrCuda;
      }
    }

    if (overlap_bar_out) {
      DeviceBuf d_frame;
      if (d_frame.alloc(sm * sm) != cudaSuccess) return kErrOom;
      // The elementwise product may overwrite its mask. A second launch gives
      // every symmetric pair a single writer after both products are ready.
      if (cudaMemcpy(d_frame.ptr, mask, sm * sm * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess) return kErrCuda;
      k_mul_init<<<grid_1d(sm * sm), 256>>>(sm * sm, d_frame.ptr, d_rot.ptr, d_frame.ptr);
      if (cudaGetLastError() != cudaSuccess) return kErrCuda;
      k_symmetrize_inplace<<<grid_1d(sm * sm), 256>>>(nmo, d_frame.ptr);
      if (cudaGetLastError() != cudaSuccess) return kErrCuda;
      // Only the symmetrized frame is needed by the tiled AO contractions.
      if (d_rot.ptr) {
        cudaFree(d_rot.ptr);
        d_rot.ptr = nullptr;
      }
      const Size rem_ov = budget_doubles - sm * sm;
      // Need d_cu (u*sm) + d_rows (u*sm) + d_cv (u*sm) + d_blk (u*u) <= rem_ov.
      Size u_tile = std::min<Size>(sa, std::max<Size>(1, static_cast<Size>(std::sqrt(
                                                             static_cast<double>(rem_ov / 2)))));
      // After each off-diagonal GEMM the coefficient tile holds its transpose.
      u_tile = std::min(u_tile, sm);
      while (u_tile > 1 && (3 * u_tile * sm + u_tile * u_tile) > rem_ov) {
        u_tile = std::max<Size>(1, u_tile / 2);
      }
      if (3 * u_tile * sm + u_tile * u_tile > rem_ov) return kErrBudget;

      DeviceBuf d_cu, d_rows, d_cv, d_blk;
      if (d_cu.alloc(u_tile * sm) != cudaSuccess || d_rows.alloc(u_tile * sm) != cudaSuccess ||
          d_cv.alloc(u_tile * sm) != cudaSuccess || d_blk.alloc(u_tile * u_tile) != cudaSuccess) {
        return kErrOom;
      }
      for (Size u0 = 0; u0 < sa; u0 += u_tile) {
        const int ul = static_cast<int>(std::min(u_tile, sa - u0));
        if (cudaMemcpy(d_cu.ptr, coefficients + u0 * sm, Size(ul) * sm * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return kErrCuda;
        }
        if (rm_gemm(cublas.handle, false, false, ul, nmo, nmo, 1.0, d_cu.ptr, nmo, d_frame.ptr, nmo,
                    0.0, d_rows.ptr, nmo) != CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
        for (Size v0 = u0; v0 < sa; v0 += u_tile) {
          const int vl = static_cast<int>(std::min(u_tile, sa - v0));
          if (cudaMemcpy(d_cv.ptr, coefficients + v0 * sm, Size(vl) * sm * sizeof(double),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return kErrCuda;
          }
          if (rm_gemm(cublas.handle, false, true, ul, vl, nmo, 1.0, d_rows.ptr, nmo, d_cv.ptr, nmo,
                      0.0, d_blk.ptr, vl) != CUBLAS_STATUS_SUCCESS) {
            return kErrCublas;
          }
          if (u0 == v0) {
            k_symmetrize_inplace<<<grid_1d(Size(ul) * ul), 256>>>(ul, d_blk.ptr);
          } else {
            // C frame C^T is symmetric. Return exactly the same rounded value
            // for both orientations of a pair without computing a second GEMM.
            k_transpose_3d_012_to_102<<<grid_1d(Size(ul) * vl), 256>>>(
                ul, vl, 1, d_blk.ptr, d_cv.ptr);
          }
          if (cudaGetLastError() != cudaSuccess) return kErrCuda;
          if (cudaMemcpy2D(overlap_bar_out + u0 * sa + v0, sa * sizeof(double), d_blk.ptr,
                           Size(vl) * sizeof(double), Size(vl) * sizeof(double),
                           static_cast<Size>(ul), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return kErrCuda;
          }
          if (u0 != v0 &&
              cudaMemcpy2D(overlap_bar_out + v0 * sa + u0, sa * sizeof(double), d_cv.ptr,
                           Size(ul) * sizeof(double), Size(ul) * sizeof(double), Size(vl),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
            return kErrCuda;
          }
        }
      }
    }
    if ((rotation_out && !check_finite(rotation_out, sm * sm)) ||
        (overlap_bar_out && !check_finite(overlap_bar_out, sa * sa))) {
      return kErrNonFinite;
    }
    return kOk;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_cphf_action_cuda(int is_rhf, int oa, int ob, int obs, int aux,
                             std::uint64_t budget_bytes, const double* vo_a, const double* oo_a,
                             const double* vv_a, const double* gap_a, const double* trial_a,
                             double* out_a, const double* vo_b, const double* oo_b,
                             const double* vv_b, const double* gap_b, const double* trial_b,
                             double* out_b) {
  try {
    if (obs < 2 || aux < 1) return kErrInvalidArg;

    const bool rhf = (is_rhf != 0);
    const int eff_oa = oa;
    const int eff_ob = rhf ? 0 : ob;
    if (rhf) {
      if (eff_oa < 1 || eff_oa >= obs || !vo_a || !oo_a || !vv_a || !gap_a || !trial_a || !out_a) {
        return kErrInvalidArg;
      }
    } else {
      if (eff_oa < 0 || eff_ob < 0 || eff_oa + eff_ob == 0 || eff_oa >= obs || eff_ob >= obs) {
        return kErrInvalidArg;
      }
      if (eff_oa > 0 && (!vo_a || !oo_a || !vv_a || !gap_a || !trial_a || !out_a)) {
        return kErrInvalidArg;
      }
      if (eff_ob > 0 && (!vo_b || !oo_b || !vv_b || !gap_b || !trial_b || !out_b)) {
        return kErrInvalidArg;
      }
    }

    const int va = (eff_oa > 0) ? (obs - eff_oa) : 0;
    const int vb = (eff_ob > 0) ? (obs - eff_ob) : 0;
    const Size n_vo_a = Size(va) * static_cast<Size>(eff_oa);
    const Size n_vo_b = Size(vb) * static_cast<Size>(eff_ob);
    if (!fits_int(n_vo_a) || !fits_int(n_vo_b)) return kErrInvalidArg;

    const Size n_oo_a = Size(eff_oa) * static_cast<Size>(eff_oa);
    const Size n_oo_b = Size(eff_ob) * static_cast<Size>(eff_ob);
    const Size n_vv_a = Size(va) * static_cast<Size>(va);
    const Size n_vv_b = Size(vb) * static_cast<Size>(vb);

    const Size max_vo = std::max(n_vo_a, n_vo_b);
    const Size max_oo = std::max(n_oo_a, n_oo_b);
    const Size max_vv = std::max(n_vv_a, n_vv_b);

    const int q_tile = compute_cphf_action_q_tile(eff_oa, eff_ob, obs, aux, budget_bytes);
    if (q_tile < 1) return kErrBudget;

    CublasGuard cublas;
    if (cublas.init() != CUBLAS_STATUS_SUCCESS) return kErrCublas;

    DeviceBuf d_trial_a, d_gap_a, d_out_a, d_trial_b, d_gap_b, d_out_b;
    if (eff_oa > 0) {
      if (d_trial_a.alloc(n_vo_a) != cudaSuccess || d_gap_a.alloc(n_vo_a) != cudaSuccess ||
          d_out_a.alloc(n_vo_a) != cudaSuccess) {
        return kErrOom;
      }
      if (cudaMemcpy(d_trial_a.ptr, trial_a, n_vo_a * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess ||
          cudaMemcpy(d_gap_a.ptr, gap_a, n_vo_a * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess) {
        return kErrCuda;
      }
      k_mul_init<<<grid_1d(n_vo_a), 256>>>(n_vo_a, d_gap_a.ptr, d_trial_a.ptr, d_out_a.ptr);
      if (cudaGetLastError() != cudaSuccess) return kErrCuda;
    }
    if (eff_ob > 0) {
      if (d_trial_b.alloc(n_vo_b) != cudaSuccess || d_gap_b.alloc(n_vo_b) != cudaSuccess ||
          d_out_b.alloc(n_vo_b) != cudaSuccess) {
        return kErrOom;
      }
      if (cudaMemcpy(d_trial_b.ptr, trial_b, n_vo_b * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess ||
          cudaMemcpy(d_gap_b.ptr, gap_b, n_vo_b * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess) {
        return kErrCuda;
      }
      k_mul_init<<<grid_1d(n_vo_b), 256>>>(n_vo_b, d_gap_b.ptr, d_trial_b.ptr, d_out_b.ptr);
      if (cudaGetLastError() != cudaSuccess) return kErrCuda;
    }

    const Size sq = static_cast<Size>(q_tile);
    DeviceBuf d_vo_a, d_vo_b, d_vv, d_oo, d_kv, d_t1, d_t2, d_t3, d_charge;
    if ((eff_oa > 0 && d_vo_a.alloc(sq * n_vo_a) != cudaSuccess) ||
        (eff_ob > 0 && d_vo_b.alloc(sq * n_vo_b) != cudaSuccess) ||
        d_vv.alloc(sq * max_vv) != cudaSuccess || d_oo.alloc(sq * max_oo) != cudaSuccess ||
        d_kv.alloc(sq * max_vo) != cudaSuccess || d_t1.alloc(sq * max_vo) != cudaSuccess ||
        d_t2.alloc(sq * max_oo) != cudaSuccess || d_t3.alloc(sq * max_oo) != cudaSuccess ||
        d_charge.alloc(sq) != cudaSuccess) {
      return kErrOom;
    }

    const double j_scale = rhf ? 4.0 : 2.0;

    for (int q0 = 0; q0 < aux; q0 += q_tile) {
      const int ql = std::min(q_tile, aux - q0);
      const Size sql = static_cast<Size>(ql);
      const Size sq0 = static_cast<Size>(q0);

      double charge_beta = 0.0;
      if (eff_oa > 0) {
        if (cudaMemcpy(d_vo_a.ptr, vo_a + sq0 * n_vo_a, sql * n_vo_a * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return kErrCuda;
        }
        if (rm_gemm(cublas.handle, false, false, ql, 1, static_cast<int>(n_vo_a), 1.0, d_vo_a.ptr,
                    static_cast<int>(n_vo_a), d_trial_a.ptr, 1, charge_beta, d_charge.ptr, 1) !=
            CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
        charge_beta = 1.0;
      }
      if (eff_ob > 0) {
        if (cudaMemcpy(d_vo_b.ptr, vo_b + sq0 * n_vo_b, sql * n_vo_b * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return kErrCuda;
        }
        if (rm_gemm(cublas.handle, false, false, ql, 1, static_cast<int>(n_vo_b), 1.0, d_vo_b.ptr,
                    static_cast<int>(n_vo_b), d_trial_b.ptr, 1, charge_beta, d_charge.ptr, 1) !=
            CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
      }

      if (eff_oa > 0) {
        if (rm_gemm(cublas.handle, true, false, static_cast<int>(n_vo_a), 1, ql, j_scale,
                    d_vo_a.ptr, static_cast<int>(n_vo_a), d_charge.ptr, 1, 1.0, d_out_a.ptr, 1) !=
            CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
        if (cudaMemcpy(d_oo.ptr, oo_a + sq0 * n_oo_a, sql * n_oo_a * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_vv.ptr, vv_a + sq0 * n_vv_a, sql * n_vv_a * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return kErrCuda;
        }
        const int rc = cphf_exchange_tile_cuda(cublas.handle, ql, eff_oa, va, d_vo_a.ptr, d_oo.ptr,
                                               d_vv.ptr, d_trial_a.ptr, d_kv.ptr, d_t1.ptr,
                                               d_t2.ptr, d_t3.ptr, d_out_a.ptr);
        if (rc != 0) return rc;
      }

      if (eff_ob > 0) {
        if (rm_gemm(cublas.handle, true, false, static_cast<int>(n_vo_b), 1, ql, j_scale,
                    d_vo_b.ptr, static_cast<int>(n_vo_b), d_charge.ptr, 1, 1.0, d_out_b.ptr, 1) !=
            CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
        if (cudaMemcpy(d_oo.ptr, oo_b + sq0 * n_oo_b, sql * n_oo_b * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_vv.ptr, vv_b + sq0 * n_vv_b, sql * n_vv_b * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return kErrCuda;
        }
        const int rc = cphf_exchange_tile_cuda(cublas.handle, ql, eff_ob, vb, d_vo_b.ptr, d_oo.ptr,
                                               d_vv.ptr, d_trial_b.ptr, d_kv.ptr, d_t1.ptr,
                                               d_t2.ptr, d_t3.ptr, d_out_b.ptr);
        if (rc != 0) return rc;
      }
    }

    if (eff_oa > 0 && cudaMemcpy(out_a, d_out_a.ptr, n_vo_a * sizeof(double),
                                 cudaMemcpyDeviceToHost) != cudaSuccess) {
      return kErrCuda;
    }
    if (eff_ob > 0 && cudaMemcpy(out_b, d_out_b.ptr, n_vo_b * sizeof(double),
                                 cudaMemcpyDeviceToHost) != cudaSuccess) {
      return kErrCuda;
    }
    if ((eff_oa > 0 && !check_finite(out_a, n_vo_a)) ||
        (eff_ob > 0 && !check_finite(out_b, n_vo_b))) {
      return kErrNonFinite;
    }
    return kOk;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_df_canonical_driver_cuda(int oa, int ob, int obs, int aux, std::uint64_t budget_bytes,
                                     const double* df_a, const double* fock_seed_a, double* out_a,
                                     const double* df_b, const double* fock_seed_b, double* out_b) {
  try {
    if (obs < 2 || aux < 1 || oa < 0 || ob < 0 || oa + ob == 0 || oa >= obs || ob >= obs) {
      return kErrInvalidArg;
    }
    if (oa > 0 && (!df_a || !fock_seed_a || !out_a)) return kErrInvalidArg;
    if (ob > 0 && (!df_b || !fock_seed_b || !out_b)) return kErrInvalidArg;
    if (oa == 0 && ((df_a != nullptr) != (fock_seed_a != nullptr))) return kErrInvalidArg;
    if (ob == 0 && ((df_b != nullptr) != (fock_seed_b != nullptr))) return kErrInvalidArg;

    const bool has_charge_a = (df_a != nullptr && fock_seed_a != nullptr);
    const bool has_charge_b = (df_b != nullptr && fock_seed_b != nullptr);

    const Size n = static_cast<Size>(obs);
    const Size n2 = n * n;
    if (!fits_int(n2)) return kErrInvalidArg;
    const int n2_int = static_cast<int>(n2);
    const Size n_vo_a = (oa > 0) ? Size(obs - oa) * static_cast<Size>(oa) : 0;
    const Size n_vo_b = (ob > 0) ? Size(obs - ob) * static_cast<Size>(ob) : 0;

    const int q_tile =
        compute_df_driver_q_tile(oa, ob, has_charge_a, has_charge_b, obs, aux, budget_bytes);
    if (q_tile < 1) return kErrBudget;

    CublasGuard cublas;
    if (cublas.init() != CUBLAS_STATUS_SUCCESS) return kErrCublas;

    DeviceBuf d_seed_a, d_field_a, d_out_a, d_seed_b, d_field_b, d_out_b;
    if (has_charge_a) {
      if (d_seed_a.alloc(n2) != cudaSuccess) return kErrOom;
      if (cudaMemcpy(d_seed_a.ptr, fock_seed_a, n2 * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess) {
        return kErrCuda;
      }
    }
    if (oa > 0) {
      if (d_field_a.alloc(n2) != cudaSuccess || d_out_a.alloc(n_vo_a) != cudaSuccess) {
        return kErrOom;
      }
      if (cudaMemset(d_field_a.ptr, 0, n2 * sizeof(double)) != cudaSuccess) {
        return kErrCuda;
      }
    }
    if (has_charge_b) {
      if (d_seed_b.alloc(n2) != cudaSuccess) return kErrOom;
      if (cudaMemcpy(d_seed_b.ptr, fock_seed_b, n2 * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess) {
        return kErrCuda;
      }
    }
    if (ob > 0) {
      if (d_field_b.alloc(n2) != cudaSuccess || d_out_b.alloc(n_vo_b) != cudaSuccess) {
        return kErrOom;
      }
      if (cudaMemset(d_field_b.ptr, 0, n2 * sizeof(double)) != cudaSuccess) {
        return kErrCuda;
      }
    }

    const Size sq = static_cast<Size>(q_tile);
    DeviceBuf d_buf1, d_buf2, d_buf3, d_charge;
    if (d_buf1.alloc(sq * n2) != cudaSuccess || d_buf2.alloc(sq * n2) != cudaSuccess ||
        d_buf3.alloc(sq * n2) != cudaSuccess || d_charge.alloc(sq) != cudaSuccess) {
      return kErrOom;
    }

    for (int q0 = 0; q0 < aux; q0 += q_tile) {
      const int ql = std::min(q_tile, aux - q0);
      const Size sql = static_cast<Size>(ql);
      const Size sq0 = static_cast<Size>(q0);

      if (oa > 0 && ob == 0) {
        double charge_beta = 0.0;
        if (has_charge_b) {
          if (cudaMemcpy(d_buf1.ptr, df_b + sq0 * n2, sql * n2 * sizeof(double),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return kErrCuda;
          }
          if (rm_gemm(cublas.handle, false, false, ql, 1, n2_int, 1.0, d_buf1.ptr, n2_int,
                      d_seed_b.ptr, 1, 0.0, d_charge.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
            return kErrCublas;
          }
          charge_beta = 1.0;
        }
        if (cudaMemcpy(d_buf1.ptr, df_a + sq0 * n2, sql * n2 * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return kErrCuda;
        }
        if (rm_gemm(cublas.handle, false, false, ql, 1, n2_int, 1.0, d_buf1.ptr, n2_int,
                    d_seed_a.ptr, 1, charge_beta, d_charge.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
        if (rm_gemm(cublas.handle, true, false, n2_int, 1, ql, 1.0, d_buf1.ptr, n2_int,
                    d_charge.ptr, 1, 1.0, d_field_a.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
        const int rc = df_driver_exchange_tile_cuda(cublas.handle, ql, obs, d_buf1.ptr, d_buf2.ptr,
                                                    d_buf3.ptr, d_seed_a.ptr, d_field_a.ptr);
        if (rc != 0) return rc;
      } else if (oa == 0 && ob > 0) {
        double charge_beta = 0.0;
        if (has_charge_a) {
          if (cudaMemcpy(d_buf1.ptr, df_a + sq0 * n2, sql * n2 * sizeof(double),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return kErrCuda;
          }
          if (rm_gemm(cublas.handle, false, false, ql, 1, n2_int, 1.0, d_buf1.ptr, n2_int,
                      d_seed_a.ptr, 1, 0.0, d_charge.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
            return kErrCublas;
          }
          charge_beta = 1.0;
        }
        if (cudaMemcpy(d_buf1.ptr, df_b + sq0 * n2, sql * n2 * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return kErrCuda;
        }
        if (rm_gemm(cublas.handle, false, false, ql, 1, n2_int, 1.0, d_buf1.ptr, n2_int,
                    d_seed_b.ptr, 1, charge_beta, d_charge.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
        if (rm_gemm(cublas.handle, true, false, n2_int, 1, ql, 1.0, d_buf1.ptr, n2_int,
                    d_charge.ptr, 1, 1.0, d_field_b.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
        const int rc = df_driver_exchange_tile_cuda(cublas.handle, ql, obs, d_buf1.ptr, d_buf2.ptr,
                                                    d_buf3.ptr, d_seed_b.ptr, d_field_b.ptr);
        if (rc != 0) return rc;
      } else {
        // Both alpha and beta are active.
        // 1) Alpha: upload df_a -> charge_a + exchange_a.
        if (cudaMemcpy(d_buf1.ptr, df_a + sq0 * n2, sql * n2 * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return kErrCuda;
        }
        if (rm_gemm(cublas.handle, false, false, ql, 1, n2_int, 1.0, d_buf1.ptr, n2_int,
                    d_seed_a.ptr, 1, 0.0, d_charge.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
        int rc = df_driver_exchange_tile_cuda(cublas.handle, ql, obs, d_buf1.ptr, d_buf2.ptr,
                                              d_buf3.ptr, d_seed_a.ptr, d_field_a.ptr);
        if (rc != 0) return rc;

        // 2) Beta: upload df_b -> charge += charge_b, J_b, exchange_b.
        if (cudaMemcpy(d_buf1.ptr, df_b + sq0 * n2, sql * n2 * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return kErrCuda;
        }
        if (rm_gemm(cublas.handle, false, false, ql, 1, n2_int, 1.0, d_buf1.ptr, n2_int,
                    d_seed_b.ptr, 1, 1.0, d_charge.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
        if (rm_gemm(cublas.handle, true, false, n2_int, 1, ql, 1.0, d_buf1.ptr, n2_int,
                    d_charge.ptr, 1, 1.0, d_field_b.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
        rc = df_driver_exchange_tile_cuda(cublas.handle, ql, obs, d_buf1.ptr, d_buf2.ptr,
                                          d_buf3.ptr, d_seed_b.ptr, d_field_b.ptr);
        if (rc != 0) return rc;

        // 3) Alpha J term using total charge = charge_a + charge_b.
        if (cudaMemcpy(d_buf1.ptr, df_a + sq0 * n2, sql * n2 * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return kErrCuda;
        }
        if (rm_gemm(cublas.handle, true, false, n2_int, 1, ql, 1.0, d_buf1.ptr, n2_int,
                    d_charge.ptr, 1, 1.0, d_field_a.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
      }
    }

    if (oa > 0) {
      k_sym_slice_vo<<<grid_1d(n_vo_a), 256>>>(obs, oa, d_field_a.ptr, d_out_a.ptr);
      if (cudaGetLastError() != cudaSuccess) return kErrCuda;
      if (cudaMemcpy(out_a, d_out_a.ptr, n_vo_a * sizeof(double), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        return kErrCuda;
      }
      if (!check_finite(out_a, n_vo_a)) return kErrNonFinite;
    }
    if (ob > 0) {
      k_sym_slice_vo<<<grid_1d(n_vo_b), 256>>>(obs, ob, d_field_b.ptr, d_out_b.ptr);
      if (cudaGetLastError() != cudaSuccess) return kErrCuda;
      if (cudaMemcpy(out_b, d_out_b.ptr, n_vo_b * sizeof(double), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        return kErrCuda;
      }
      if (!check_finite(out_b, n_vo_b)) return kErrNonFinite;
    }
    return kOk;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

int gqc_f12_df_nuclear_weights_cuda(
    int oa, int ob, int obs, int aux, int nao, int include_direct, std::uint64_t budget_bytes,
    const double* coeff_a, const double* coeff_b, const double* occ_e_a, const double* occ_e_b,
    const double* df_a, const double* df_b, const double* fock_seed_a, const double* fock_seed_b,
    const double* overlap_seed_a, const double* overlap_seed_b, const double* whitener,
    const double* z_a, const double* z_b, const double* direct_overlap_ao,
    const double* direct_hcore_ao, const double* direct_ao_coulomb,
    const double* direct_coulomb_metric, const double* frame_alpha, const double* frame_beta,
    double* overlap_ao_out, double* hcore_ao_out, double* ao_coulomb_out,
    double* metric_coulomb_out) {
  try {
    if (obs < 2 || aux < 1 || nao < obs || oa < 0 || ob < 0 || oa + ob == 0 || oa >= obs ||
        ob >= obs) {
      return kErrInvalidArg;
    }
    if (!coeff_a || !coeff_b || !df_a || !df_b || !fock_seed_a || !fock_seed_b || !overlap_seed_a ||
        !overlap_seed_b || !whitener || !overlap_ao_out || !hcore_ao_out || !ao_coulomb_out ||
        !metric_coulomb_out) {
      return kErrInvalidArg;
    }
    if (oa > 0 && (!occ_e_a || !z_a)) return kErrInvalidArg;
    if (ob > 0 && (!occ_e_b || !z_b)) return kErrInvalidArg;
    const bool inc_dir = (include_direct != 0);
    if (inc_dir && (!direct_overlap_ao || !direct_hcore_ao || !direct_ao_coulomb ||
                    !direct_coulomb_metric || !frame_alpha || !frame_beta)) {
      return kErrInvalidArg;
    }

    NuclearWeightsSetup st = prepare_nuclear_weights_setup(
        oa, ob, obs, nao, coeff_a, coeff_b, occ_e_a, occ_e_b, df_a, df_b, fock_seed_a, fock_seed_b,
        overlap_seed_a, overlap_seed_b, z_a, z_b);

    const int n_eff = st.n_eff;
    const bool same_spin_all = st.same_spin_all;
    const Size n = static_cast<Size>(obs);
    const Size n2 = n * n;
    const Size ne = static_cast<Size>(n_eff);
    const Size ne2 = ne * ne;
    if (!fits_int(n2) || !fits_int(ne2)) return kErrInvalidArg;

    const int q_tile =
        compute_df_nuclear_weights_q_tile(oa, ob, obs, aux, n_eff, same_spin_all, budget_bytes);
    if (q_tile < 1) return kErrBudget;

    const Size snao = static_cast<Size>(nao);
    const Size s_aux = static_cast<Size>(aux);
    const Size sq = static_cast<Size>(q_tile);
    const Size budget_doubles = effective_budget_bytes(budget_bytes) / sizeof(double);

    CublasGuard cublas;
    if (cublas.init() != CUBLAS_STATUS_SUCCESS) return kErrCublas;

    std::vector<double> bar_m(s_aux * s_aux, 0.0);

    // Scoped Phases 1-3 so all Q-tile device buffers free before Phase 4.
    {
      DeviceBuf d_c_a, d_p_a, d_psym_a, d_sm_oo_a;
      DeviceBuf d_c_b, d_p_b, d_psym_b, d_sm_oo_b;
      DeviceBuf d_charge, d_bardcharge;
      DeviceBuf d_b_a, d_db, d_work1, d_work2, d_ao_chunk;

      if (d_c_a.alloc(ne * n) != cudaSuccess || d_p_a.alloc(n2) != cudaSuccess ||
          (oa > 0 && (d_psym_a.alloc(n2) != cudaSuccess ||
                      d_sm_oo_a.alloc(Size(oa) * Size(oa)) != cudaSuccess)) ||
          d_charge.alloc(s_aux) != cudaSuccess || d_bardcharge.alloc(s_aux) != cudaSuccess ||
          d_b_a.alloc(sq * n2) != cudaSuccess || d_db.alloc(sq * n2) != cudaSuccess ||
          d_work1.alloc(sq * ne2) != cudaSuccess || d_work2.alloc(sq * ne2) != cudaSuccess ||
          d_ao_chunk.alloc(sq * ne2) != cudaSuccess) {
        return kErrOom;
      }
      if (cudaMemcpy(d_c_a.ptr, coeff_a, ne * n * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess ||
          cudaMemcpy(d_p_a.ptr, st.p_a.data(), n2 * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess ||
          cudaMemset(d_charge.ptr, 0, s_aux * sizeof(double)) != cudaSuccess ||
          cudaMemset(d_bardcharge.ptr, 0, s_aux * sizeof(double)) != cudaSuccess) {
        return kErrCuda;
      }
      if (oa > 0) {
        if (cudaMemcpy(d_psym_a.ptr, st.psym_a.data(), n2 * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemset(d_sm_oo_a.ptr, 0, Size(oa) * Size(oa) * sizeof(double)) != cudaSuccess) {
          return kErrCuda;
        }
      }
      if (!same_spin_all) {
        if (d_c_b.alloc(ne * n) != cudaSuccess || d_p_b.alloc(n2) != cudaSuccess ||
            (ob > 0 && (d_psym_b.alloc(n2) != cudaSuccess ||
                        d_sm_oo_b.alloc(Size(ob) * Size(ob)) != cudaSuccess))) {
          return kErrOom;
        }
        if (cudaMemcpy(d_c_b.ptr, coeff_b, ne * n * sizeof(double), cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaMemcpy(d_p_b.ptr, st.p_b.data(), n2 * sizeof(double), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
          return kErrCuda;
        }
        if (ob > 0) {
          if (cudaMemcpy(d_psym_b.ptr, st.psym_b.data(), n2 * sizeof(double),
                         cudaMemcpyHostToDevice) != cudaSuccess ||
              cudaMemset(d_sm_oo_b.ptr, 0, Size(ob) * Size(ob) * sizeof(double)) != cudaSuccess) {
            return kErrCuda;
          }
        }
      }

      struct SpinDevView {
        int occ;
        const double* h_df;
        const double* d_c;
        const double* d_p;
        const double* d_psym;
        double* d_sm_oo;
      };
      SpinDevView spins[2] = {
          {oa, df_a, d_c_a.ptr, d_p_a.ptr, d_psym_a.ptr, d_sm_oo_a.ptr},
          {ob, df_b, d_c_b.ptr, d_p_b.ptr, d_psym_b.ptr, d_sm_oo_b.ptr},
      };
      const int n_loop_spins = same_spin_all ? 1 : 2;
      const double spin_scale = same_spin_all ? 2.0 : 1.0;

      // Phase 1: charge[q] and bar_dcharge[q].
      for (int q0 = 0; q0 < aux; q0 += q_tile) {
        const int ql = std::min(q_tile, aux - q0);
        const Size sql = static_cast<Size>(ql);
        const Size sq0 = static_cast<Size>(q0);
        for (int s = 0; s < n_loop_spins; ++s) {
          const SpinDevView& sp = spins[s];
          if (cudaMemcpy(d_b_a.ptr, sp.h_df + sq0 * n2, sql * n2 * sizeof(double),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return kErrCuda;
          }
          if (rm_gemm(cublas.handle, false, false, ql, 1, static_cast<int>(n2), spin_scale,
                      d_b_a.ptr, static_cast<int>(n2), sp.d_p, 1, 1.0, d_bardcharge.ptr + sq0,
                      1) != CUBLAS_STATUS_SUCCESS) {
            return kErrCublas;
          }
          if (sp.occ > 0) {
            k_add_occ_trace<<<grid_1d(sql), 256>>>(ql, obs, sp.occ, spin_scale, d_b_a.ptr,
                                                   d_charge.ptr + sq0);
            if (cudaGetLastError() != cudaSuccess) return kErrCuda;
          }
        }
      }

      const int b_tile = static_cast<int>(std::min<Size>(sq, ne2));
      const int t_tile = static_cast<int>(std::min<Size>(sq, ne2));
      std::vector<double> h_m_blk(sq * static_cast<Size>(b_tile));
      std::vector<double> h_dr_tile;
      if (inc_dir || n_eff != nao || q_tile < aux) {
        h_dr_tile.resize(static_cast<Size>(t_tile) * ne2);
      }

      // Phase 2 (fused) + Phase 3: bar_sm_oo, bar_db, bar_M, ao_db_chunk, bar_dr.
      for (int a0 = 0; a0 < aux; a0 += q_tile) {
        const int ql = std::min(q_tile, aux - a0);
        const Size sql = static_cast<Size>(ql);
        const Size sa0 = static_cast<Size>(a0);

        for (int s = 0; s < n_loop_spins; ++s) {
          const SpinDevView& sp = spins[s];
          const int occ = sp.occ;
          const Size so = static_cast<Size>(occ);

          if (cudaMemcpy(d_b_a.ptr, sp.h_df + sa0 * n2, sql * n2 * sizeof(double),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return kErrCuda;
          }

          if (occ > 0) {
            const long long stride_bo = static_cast<long long>(obs) * static_cast<long long>(occ);
            k_extract_bo_and_oo<<<grid_1d(sql * n * so), 256>>>(ql, obs, occ, d_b_a.ptr,
                                                                d_work1.ptr, d_db.ptr);
            if (cudaGetLastError() != cudaSuccess) return kErrCuda;
            // s_mc: d_sm_oo += -bar_dcharge * B_oo
            if (rm_gemm(cublas.handle, true, false, occ * occ, 1, ql, -1.0, d_db.ptr, occ * occ,
                        d_bardcharge.ptr + sa0, 1, 1.0, sp.d_sm_oo, 1) != CUBLAS_STATUS_SUCCESS) {
              return kErrCublas;
            }
            // pb[a, q, i] = sum_p P[p, q] * B_o[a, p, i] -> d_work2
            if (rm_gemm_batched(cublas.handle, true, false, obs, occ, obs, 1.0, sp.d_p, obs, 0LL,
                                d_work1.ptr, occ, stride_bo, 0.0, d_work2.ptr, occ, stride_bo,
                                ql) != CUBLAS_STATUS_SUCCESS) {
              return kErrCublas;
            }
            // s_ex: d_sm_oo[i, j] += sum_{a, q} pb[a, q, i] * B_o[a, q, j]
            if (rm_gemm(cublas.handle, true, false, occ, occ, ql * obs, 1.0, d_work2.ptr, occ,
                        d_work1.ptr, occ, 1.0, sp.d_sm_oo, occ) != CUBLAS_STATUS_SUCCESS) {
              return kErrCublas;
            }
          }

          // Build bar_db[a, p, q] into d_db
          k_init_bar_db<<<grid_1d(sql * n2), 256>>>(ql, obs, occ, d_charge.ptr + sa0,
                                                    d_bardcharge.ptr + sa0, sp.d_p, d_db.ptr);
          if (cudaGetLastError() != cudaSuccess) return kErrCuda;

          if (occ > 0) {
            // d_work1 still holds B_o[ql, obs, occ]
            const long long stride_bo = static_cast<long long>(obs) * static_cast<long long>(occ);
            if (rm_gemm_batched(cublas.handle, false, false, obs, occ, obs, 1.0, sp.d_psym, obs,
                                0LL, d_work1.ptr, occ, stride_bo, 0.0, d_work2.ptr, occ, stride_bo,
                                ql) != CUBLAS_STATUS_SUCCESS) {
              return kErrCublas;
            }
            k_sub_ex_from_db<<<grid_1d(sql * n * so), 256>>>(ql, obs, occ, d_work2.ptr, d_db.ptr);
            if (cudaGetLastError() != cudaSuccess) return kErrCuda;
          }

          // Accumulate bar_M[a0:a0+ql, :]
          for (int b0 = 0; b0 < aux; b0 += b_tile) {
            const int bl = std::min(b_tile, aux - b0);
            const double* b_ptr = nullptr;
            if (b0 == a0 && bl == ql) {
              b_ptr = d_b_a.ptr;
            } else {
              if (cudaMemcpy(d_work1.ptr, sp.h_df + Size(b0) * n2, Size(bl) * n2 * sizeof(double),
                             cudaMemcpyHostToDevice) != cudaSuccess) {
                return kErrCuda;
              }
              b_ptr = d_work1.ptr;
            }
            if (rm_gemm(cublas.handle, false, true, ql, bl, static_cast<int>(n2), -0.5 * spin_scale,
                        d_db.ptr, static_cast<int>(n2), b_ptr, static_cast<int>(n2), 0.0,
                        d_work2.ptr, bl) != CUBLAS_STATUS_SUCCESS) {
              return kErrCublas;
            }
            if (cudaMemcpy(h_m_blk.data(), d_work2.ptr, sql * Size(bl) * sizeof(double),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
              return kErrCuda;
            }
            for (int a = 0; a < ql; ++a) {
              double* dst = bar_m.data() + Size(a0 + a) * s_aux + Size(b0);
              const double* src = h_m_blk.data() + Size(a) * Size(bl);
              for (int b = 0; b < bl; ++b) dst[b] += src[b];
            }
          }

          // Back-project d_db to AO space: d_ao_chunk += spin_scale * C @ d_db @ C^T
          if (rm_gemm(cublas.handle, false, true, ql * obs, n_eff, obs, 1.0, d_db.ptr, obs, sp.d_c,
                      obs, 0.0, d_work1.ptr, n_eff) != CUBLAS_STATUS_SUCCESS) {
            return kErrCublas;
          }
          const long long stride_tmp = static_cast<long long>(obs) * static_cast<long long>(n_eff);
          const long long stride_ao = static_cast<long long>(n_eff) * static_cast<long long>(n_eff);
          const double beta_ao = (s == 0) ? 0.0 : 1.0;
          if (rm_gemm_batched(cublas.handle, false, false, n_eff, n_eff, obs, spin_scale, sp.d_c,
                              obs, 0LL, d_work1.ptr, n_eff, stride_tmp, beta_ao, d_ao_chunk.ptr,
                              n_eff, stride_ao, ql) != CUBLAS_STATUS_SUCCESS) {
            return kErrCublas;
          }
        }

        // Symmetrize d_ao_chunk[ql, n_eff, n_eff] in-place
        k_symmetrize_3d_inplace<<<grid_1d(sql * ne2), 256>>>(ql, n_eff, d_ao_chunk.ptr);
        if (cudaGetLastError() != cudaSuccess) return kErrCuda;

        // Accumulate bar_dr[t0:t0+tl, :n_eff, :n_eff] += W[a0:a0+ql, t0:t0+tl]^T @ d_ao_chunk
        for (int t0 = 0; t0 < aux; t0 += t_tile) {
          const int tl = std::min(t_tile, aux - t0);
          if (tl == aux) {
            if (cudaMemcpy(d_work1.ptr, whitener + sa0 * s_aux, sql * s_aux * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
              return kErrCuda;
            }
          } else {
            if (cudaMemcpy2D(d_work1.ptr, Size(tl) * sizeof(double),
                             whitener + sa0 * s_aux + Size(t0), s_aux * sizeof(double),
                             Size(tl) * sizeof(double), sql, cudaMemcpyHostToDevice) !=
                cudaSuccess) {
              return kErrCuda;
            }
          }
          if (rm_gemm(cublas.handle, true, false, tl, static_cast<int>(ne2), ql, 1.0, d_work1.ptr,
                      tl, d_ao_chunk.ptr, static_cast<int>(ne2), 0.0, d_work2.ptr,
                      static_cast<int>(ne2)) != CUBLAS_STATUS_SUCCESS) {
            return kErrCublas;
          }
          if (a0 == 0 && !inc_dir && n_eff == nao) {
            if (cudaMemcpy(ao_coulomb_out + Size(t0) * snao * snao, d_work2.ptr,
                           Size(tl) * ne2 * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
              return kErrCuda;
            }
          } else {
            if (cudaMemcpy(h_dr_tile.data(), d_work2.ptr, Size(tl) * ne2 * sizeof(double),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
              return kErrCuda;
            }
            accumulate_dr_tile_host(a0, t0, tl, nao, n_eff, inc_dir, direct_ao_coulomb,
                                    h_dr_tile.data(), ao_coulomb_out);
          }
        }
      }

      // Download bar_sm_oo increments and fold into st.bar_sm_{a,b}
      auto download_and_add_sm = [&](int occ, const double* d_src, std::vector<double>& dst1,
                                     std::vector<double>* dst2) -> int {
        const Size so = static_cast<Size>(occ);
        std::vector<double> h_sm(so * so);
        if (cudaMemcpy(h_sm.data(), d_src, so * so * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
          return kErrCuda;
        }
        for (Size i = 0; i < so; ++i) {
          for (Size j = 0; j < so; ++j) {
            const double v = h_sm[i * so + j];
            dst1[i * n + j] += v;
            if (dst2) (*dst2)[i * n + j] += v;
          }
        }
        return kOk;
      };
      if (oa > 0) {
        const int rc = download_and_add_sm(oa, d_sm_oo_a.ptr, st.bar_sm_a,
                                           same_spin_all ? &st.bar_sm_b : nullptr);
        if (rc != 0) return rc;
      }
      if (ob > 0 && !same_spin_all) {
        const int rc = download_and_add_sm(ob, d_sm_oo_b.ptr, st.bar_sm_b, nullptr);
        if (rc != 0) return rc;
      }
    }

    // Phase 4a: 2D AO projections bar_dh and bar_ds (budget-bounded GPU GEMMs)
    std::memset(hcore_ao_out, 0, snao * snao * sizeof(double));
    std::memset(overlap_ao_out, 0, snao * snao * sizeof(double));
    std::vector<double> tmp_cm(ne * n);
    const double* coeffs[2] = {coeff_a, coeff_b};
    const std::vector<double>* p_vecs[2] = {&st.p_a, &st.p_b};
    const std::vector<double>* sm_vecs[2] = {&st.bar_sm_a, &st.bar_sm_b};
    for (int s = 0; s < 2; ++s) {
      int rc = tiled_gemm_cuda(cublas.handle, false, false, ne, n, n, coeffs[s], n,
                               p_vecs[s]->data(), n, tmp_cm.data(), n, budget_doubles, false);
      if (rc != 0) return rc;
      rc = tiled_gemm_cuda(cublas.handle, false, true, ne, ne, n, tmp_cm.data(), n, coeffs[s], n,
                           hcore_ao_out, snao, budget_doubles, true);
      if (rc != 0) return rc;

      rc = tiled_gemm_cuda(cublas.handle, false, false, ne, n, n, coeffs[s], n, sm_vecs[s]->data(),
                           n, tmp_cm.data(), n, budget_doubles, false);
      if (rc != 0) return rc;
      rc = tiled_gemm_cuda(cublas.handle, false, true, ne, ne, n, tmp_cm.data(), n, coeffs[s], n,
                           overlap_ao_out, snao, budget_doubles, true);
      if (rc != 0) return rc;
    }
    symmetrize_square_host(snao, hcore_ao_out);
    symmetrize_square_host(snao, overlap_ao_out);

    // Phase 4b: bar_dJ = 0.5 * (W^T @ bar_M @ W + (W^T @ bar_M @ W)^T)
    std::vector<double> tmp_m(s_aux * s_aux);
    int rc = tiled_gemm_cuda(cublas.handle, false, false, s_aux, s_aux, s_aux, bar_m.data(), s_aux,
                             whitener, s_aux, tmp_m.data(), s_aux, budget_doubles, false);
    if (rc != 0) return rc;
    rc = tiled_gemm_cuda(cublas.handle, true, false, s_aux, s_aux, s_aux, whitener, s_aux,
                         tmp_m.data(), s_aux, metric_coulomb_out, s_aux, budget_doubles, false);
    if (rc != 0) return rc;
    symmetrize_square_host(s_aux, metric_coulomb_out);

    const Size snao2 = snao * snao;
    const Size sq2 = s_aux * s_aux;
    if (inc_dir) {
      for (Size idx = 0; idx < snao2; ++idx) {
        hcore_ao_out[idx] += direct_hcore_ao[idx];
        overlap_ao_out[idx] += direct_overlap_ao[idx] + frame_alpha[idx] + frame_beta[idx];
      }
      for (Size idx = 0; idx < sq2; ++idx) {
        metric_coulomb_out[idx] += direct_coulomb_metric[idx];
      }
    }
    if (!check_finite(overlap_ao_out, snao2) || !check_finite(hcore_ao_out, snao2) ||
        !check_finite(ao_coulomb_out, s_aux * snao2) || !check_finite(metric_coulomb_out, sq2)) {
      return kErrNonFinite;
    }
    return kOk;
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
