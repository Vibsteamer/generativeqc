// GPU-resident cuBLAS (and matching host reference) kernels for fixed-cusp CC-F12
// Coulomb density-fitting solve (forward and implicit metric/three-center reverse).
// Column-chunked over RHS columns so device VRAM stays bounded <= 512 MiB.
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
constexpr int kErrNonFinite = -2;
constexpr int kErrOom = -5;
constexpr int kErrInternal = -6;
constexpr Size kDefaultBudgetBytes = Size(480) << 20;  // Leave >= 32 MiB headroom under 512 MiB.

inline bool checked_mul(Size a, Size b, Size& out) {
  if (a != 0 && b > std::numeric_limits<Size>::max() / a) return false;
  out = a * b;
  return true;
}

inline bool checked_add(Size a, Size b, Size& out) {
  if (a > std::numeric_limits<Size>::max() - b) return false;
  out = a + b;
  return true;
}

inline bool check_finite(const double* ptr, Size count) {
  if (!ptr || count == 0) return true;
  for (Size i = 0; i < count; ++i) {
    if (!std::isfinite(ptr[i])) return false;
  }
  return true;
}

int compute_chunk_cols(int aux, std::int64_t ncols, int fixed_mats, int chunk_bufs,
                       std::int64_t max_workspace_bytes) {
  const Size budget =
      (max_workspace_bytes > 0) ? static_cast<Size>(max_workspace_bytes) : kDefaultBudgetBytes;
  const Size budget_doubles = budget / sizeof(double);
  const Size s_aux = static_cast<Size>(aux);
  Size aa = 0, fixed_doubles = 0, per_col_doubles = 0;
  if (!checked_mul(s_aux, s_aux, aa) ||
      !checked_mul(static_cast<Size>(fixed_mats), aa, fixed_doubles) ||
      !checked_mul(static_cast<Size>(chunk_bufs), s_aux, per_col_doubles)) {
    return 0;
  }
  if (per_col_doubles == 0 || fixed_doubles >= budget_doubles) return 0;
  const Size avail_doubles = budget_doubles - fixed_doubles;
  if (avail_doubles < per_col_doubles) return 0;
  const Size max_by_mem = avail_doubles / per_col_doubles;
  const Size max_int = static_cast<Size>(std::numeric_limits<int>::max());
  const Size capped =
      std::min<Size>(static_cast<Size>(ncols), std::min<Size>(max_by_mem, max_int));
  return static_cast<int>(capped);
}

int compute_residual_chunk(int aux, int ext, int nact_r, Size budget_bytes) {
  const Size budget_doubles = budget_bytes / sizeof(double);
  Size aa = 0, ae = 0, per_orb_doubles = 0;
  if (!checked_mul(Size(aux), Size(aux), aa) || !checked_mul(Size(aux), Size(ext), ae) ||
      !checked_mul(Size(2), ae, per_orb_doubles)) {
    return 0;
  }
  if (per_orb_doubles == 0 || aa >= budget_doubles) return 0;
  const Size avail_doubles = budget_doubles - aa;
  if (avail_doubles < per_orb_doubles) return 0;
  return static_cast<int>(std::min<Size>(Size(nact_r), avail_doubles / per_orb_doubles));
}

int compute_fock_mix_chunk(int nact, Size ae, Size budget_bytes) {
  const Size budget_doubles = budget_bytes / sizeof(double);
  Size ff = 0, per_col_doubles = 0;
  if (!checked_mul(Size(nact), Size(nact), ff) ||
      !checked_mul(Size(2), Size(nact), per_col_doubles)) {
    return 0;
  }
  if (per_col_doubles == 0 || ff >= budget_doubles) return 0;
  const Size avail_doubles = budget_doubles - ff;
  if (avail_doubles < per_col_doubles) return 0;
  const Size max_int = static_cast<Size>(std::numeric_limits<int>::max());
  return static_cast<int>(std::min<Size>(ae, std::min<Size>(max_int, avail_doubles / per_col_doubles)));
}

bool fixed_pair_sources_fits_budget(bool same_spin, bool compute_wc, bool compute_quad, int nact_l,
                                    int nact_r, int obs, int ext, int aux, int nv_l, int nv_r,
                                    Size budget_bytes) {
  const Size budget_doubles = budget_bytes / sizeof(double);
  Size ae = 0, ee = 0, oo = 0, vv = 0;
  if (!checked_mul(Size(aux), Size(ext), ae) || !checked_mul(Size(ext), Size(ext), ee) ||
      !checked_mul(Size(obs), Size(obs), oo) || !checked_mul(Size(nv_l), Size(nv_r), vv)) {
    return false;
  }
  // Stage 0: residual and Fock-mix minimum chunks.
  if (compute_residual_chunk(aux, ext, nact_r, budget_bytes) < 1) return false;
  if (compute_quad) {
    if (compute_fock_mix_chunk(nact_l, ae, budget_bytes) < 1) return false;
    if (!same_spin && compute_fock_mix_chunk(nact_r, ae, budget_bytes) < 1) return false;
  }
  // Stage 1: fixed resident matrices + minimum 4-slice scratch (Mode B).
  Size fixed_ee_count = 4;
  if (compute_wc || compute_quad) fixed_ee_count += same_spin ? 1 : 2;
  if (!same_spin) fixed_ee_count += 1;
  if (compute_quad) fixed_ee_count += (same_spin ? 4 : 6);
  Size fixed_ee = 0, fixed_elems = 0, four_ae = 0, stage1_min = 0;
  const Size sub_elems = compute_wc ? std::max(oo, vv) : Size(0);
  if (!checked_mul(fixed_ee_count, ee, fixed_ee) ||
      !checked_add(fixed_ee, Size(4) + sub_elems, fixed_elems) ||
      !checked_mul(Size(4), ae, four_ae) ||
      !checked_add(fixed_elems, four_ae, stage1_min) ||
      stage1_min > budget_doubles) {
    return false;
  }
  // Stage 2: minimum 5*oo matrices (B=1, P=1) when compute_wc is enabled.
  if (compute_wc) {
    Size stage2_min = 0;
    if (!checked_mul(Size(5), oo, stage2_min) || stage2_min > budget_doubles) return false;
  }
  return true;
}

void pack_cols_row_major(int rows, std::int64_t total_cols, std::int64_t col_start, int chunk_cols,
                         const double* src, double* dst) {
  const Size bytes = Size(chunk_cols) * sizeof(double);
  for (int r = 0; r < rows; ++r) {
    std::memcpy(dst + Size(r) * chunk_cols, src + Size(r) * total_cols + col_start, bytes);
  }
}

void unpack_cols_row_major(int rows, std::int64_t total_cols, std::int64_t col_start,
                           int chunk_cols, const double* src, double* dst) {
  const Size bytes = Size(chunk_cols) * sizeof(double);
  for (int r = 0; r < rows; ++r) {
    std::memcpy(dst + Size(r) * total_cols + col_start, src + Size(r) * chunk_cols, bytes);
  }
}

#ifdef GQC_SINK_HOST_REFERENCE

void rm_gemm_host(bool trans_a, bool trans_b, int M, int N, int K, double alpha, const double* A,
                  int lda, const double* B, int ldb, double beta, double* C, int ldc) {
  if (!trans_b) {
    for (int i = 0; i < M; ++i) {
      double* c_row = C + Size(i) * ldc;
      if (beta == 0.0) {
        std::fill_n(c_row, N, 0.0);
      } else if (beta != 1.0) {
        for (int j = 0; j < N; ++j) c_row[j] *= beta;
      }
    }
    if (!trans_a) {
      for (int i = 0; i < M; ++i) {
        double* c_row = C + Size(i) * ldc;
        const double* a_row = A + Size(i) * lda;
        for (int k = 0; k < K; ++k) {
          const double aik = alpha * a_row[k];
          const double* b_row = B + Size(k) * ldb;
          for (int j = 0; j < N; ++j) c_row[j] += aik * b_row[j];
        }
      }
    } else {
      for (int k = 0; k < K; ++k) {
        const double* a_row = A + Size(k) * lda;
        const double* b_row = B + Size(k) * ldb;
        for (int i = 0; i < M; ++i) {
          const double aki = alpha * a_row[i];
          double* c_row = C + Size(i) * ldc;
          for (int j = 0; j < N; ++j) c_row[j] += aki * b_row[j];
        }
      }
    }
    return;
  }

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      if (!trans_a) {
        const double* a_row = A + Size(i) * lda;
        const double* b_row = B + Size(j) * ldb;
        for (int k = 0; k < K; ++k) acc += a_row[k] * b_row[k];
      } else {
        const double* b_row = B + Size(j) * ldb;
        for (int k = 0; k < K; ++k) acc += A[Size(k) * lda + i] * b_row[k];
      }
      double& dst = C[Size(i) * ldc + j];
      dst = (beta == 0.0) ? (alpha * acc) : (beta * dst + alpha * acc);
    }
  }
}

double dot_host(Size n, const double* x, const double* y) {
  double s = 0.0;
  for (Size i = 0; i < n; ++i) s += x[i] * y[i];
  return s;
}

void copy_subblock_host(int rows, int cols, const double* src, int src_ld, double* dst,
                        int dst_ld) {
  const Size row_bytes = Size(cols) * sizeof(double);
  for (int r = 0; r < rows; ++r) {
    std::memcpy(dst + Size(r) * dst_ld, src + Size(r) * src_ld, row_bytes);
  }
}

void robust_from_res_host(int aux, int ext, const double* Dl, const double* Ol, const double* Dr,
                          const double* Kr, double beta_out, double* G_raw) {
  rm_gemm_host(true, false, ext, ext, aux, 1.0, Dl, ext, Kr, ext, beta_out, G_raw, ext);
  rm_gemm_host(true, false, ext, ext, aux, 1.0, Ol, ext, Dr, ext, 1.0, G_raw, ext);
}

void dress_fwd_host(bool same_spin, int ext, const double* S, const double* in, double* out,
                    double* tmp) {
  if (same_spin) {
    for (int p = 0; p < ext; ++p) {
      for (int q = 0; q < ext; ++q) {
        out[Size(p) * ext + q] = 0.25 * (in[Size(p) * ext + q] - in[Size(q) * ext + p]);
      }
    }
    return;
  }
  const Size n2 = Size(ext) * ext;
  std::copy_n(in, n2, out);
  rm_gemm_host(false, true, ext, ext, ext, 1.0, S, ext, in, ext, 0.0, tmp, ext);
  rm_gemm_host(false, false, ext, ext, ext, 0.125, tmp, ext, S, ext, 0.375, out, ext);
}

void strong_mask_host(int ext, int nl, int nr, int obs, const double* in, double* out,
                      bool invert) {
  for (int p = 0; p < ext; ++p) {
    for (int q = 0; q < ext; ++q) {
      const bool in_q = (p >= nl && q >= nr) && !(p < obs && q < obs);
      const bool keep = invert ? !in_q : in_q;
      out[Size(p) * ext + q] = keep ? in[Size(p) * ext + q] : 0.0;
    }
  }
}

double v_mask_dot_host(int ext, int nl, int nr, int obs, const double* C, const double* f) {
  double acc = 0.0;
  for (int p = 0; p < ext; ++p) {
    for (int q = 0; q < ext; ++q) {
      const bool in_m = (p < obs && q < obs) || (p < nl && q >= obs) || (p >= obs && q < nr);
      if (in_m) {
        const Size idx = Size(p) * ext + q;
        acc += C[idx] * f[idx];
      }
    }
  }
  return acc;
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
    // These FP64 primitives accept exact-fit budgets; no uncharged BLAS operation workspace.
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

inline unsigned grid_for(Size n, unsigned blk = 256) {
  return static_cast<unsigned>((n + blk - 1) / blk);
}

__global__ void k_symmetrize_neg_half_inplace(int aux, double* mat) {
  const Size total = Size(aux) * Size(aux);
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int a = static_cast<int>(idx / aux);
  const int b = static_cast<int>(idx % aux);
  if (a <= b) {
    const Size ab = Size(a) * aux + b;
    const Size ba = Size(b) * aux + a;
    const double v = -0.5 * (mat[ab] + mat[ba]);
    mat[ab] = v;
    mat[ba] = v;
  }
}

__global__ void k_antisym_quarter(int ext, const double* in, double* out) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(ext) * ext;
  if (idx >= total) return;
  const int p = static_cast<int>(idx / ext);
  const int q = static_cast<int>(idx % ext);
  out[idx] = 0.25 * (in[idx] - in[Size(q) * ext + p]);
}

__global__ void k_add_two(Size n, const double* a, const double* b, double* out) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void k_strong_mask(int ext, int nl, int nr, int obs, const double* in, double* out,
                              bool invert) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(ext) * ext;
  if (idx >= total) return;
  const int p = static_cast<int>(idx / ext);
  const int q = static_cast<int>(idx % ext);
  const bool in_q = (p >= nl && q >= nr) && !(p < obs && q < obs);
  const bool keep = invert ? !in_q : in_q;
  out[idx] = keep ? in[idx] : 0.0;
}

__global__ void k_copy_subblock(int rows, int cols, const double* src, int src_ld, double* dst,
                                int dst_ld) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(rows) * cols;
  if (idx >= total) return;
  const int r = static_cast<int>(idx / cols);
  const int c = static_cast<int>(idx % cols);
  dst[Size(r) * dst_ld + c] = src[Size(r) * src_ld + c];
}

__global__ void k_set_chi_same_spin(int ext, int i, int j, double* mat) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(ext) * ext;
  if (idx >= total) return;
  const int p = static_cast<int>(idx / ext);
  const int q = static_cast<int>(idx % ext);
  double v = 0.0;
  if (p == i && q == j) v += 0.25;
  if (p == j && q == i) v -= 0.25;
  mat[idx] = v;
}

__global__ void k_set_chi_opp_spin(int ext, int i, int j, const double* S, double* mat) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(ext) * ext;
  if (idx >= total) return;
  const int p = static_cast<int>(idx / ext);
  const int q = static_cast<int>(idx % ext);
  double v = 0.125 * S[Size(p) * ext + j] * S[Size(i) * ext + q];
  if (p == i && q == j) v += 0.375;
  mat[idx] = v;
}

__global__ void k_dot_accum(Size n, double scale, const double* x, const double* y, double* accum) {
  __shared__ double sdata[256];
  double local = 0.0;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += Size(gridDim.x) * blockDim.x) {
    local += x[idx] * y[idx];
  }
  sdata[threadIdx.x] = local;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(accum, scale * sdata[0]);
}

__global__ void k_v_energy_accum(int ext, int nl, int nr, int obs, int i, int j, double mult,
                                 const double* fg, const double* C, const double* f,
                                 double* accum) {
  __shared__ double sdata[256];
  const Size total = Size(ext) * ext;
  double local = 0.0;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const int p = static_cast<int>(idx / ext);
    const int q = static_cast<int>(idx % ext);
    const bool in_m = (p < obs && q < obs) || (p < nl && q >= obs) || (p >= obs && q < nr);
    if (in_m) local += C[idx] * f[idx];
  }
  sdata[threadIdx.x] = local;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    double contrib = -2.0 * mult * sdata[0];
    if (blockIdx.x == 0) contrib += 2.0 * mult * fg[Size(i) * ext + j];
    atomicAdd(accum, contrib);
  }
}

bool robust_from_res_gpu(cublasHandle_t h, int aux, int ext, const double* Dl, const double* Ol,
                         const double* Dr, const double* Kr, double beta_out, double* G_raw) {
  if (rm_gemm(h, true, false, ext, ext, aux, 1.0, Dl, ext, Kr, ext, beta_out, G_raw, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  if (rm_gemm(h, true, false, ext, ext, aux, 1.0, Ol, ext, Dr, ext, 1.0, G_raw, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  return true;
}

bool dress_fwd_gpu(cublasHandle_t h, bool same_spin, int ext, const double* S, const double* in,
                   double* out, double* tmp) {
  const Size n2 = Size(ext) * ext;
  if (same_spin) {
    k_antisym_quarter<<<grid_for(n2), 256>>>(ext, in, out);
    return cudaGetLastError() == cudaSuccess;
  }
  if (cudaMemcpy(out, in, n2 * sizeof(double), cudaMemcpyDeviceToDevice) != cudaSuccess)
    return false;
  if (rm_gemm(h, false, true, ext, ext, ext, 1.0, S, ext, in, ext, 0.0, tmp, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  if (rm_gemm(h, false, false, ext, ext, ext, 0.125, tmp, ext, S, ext, 0.375, out, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  return true;
}

bool compute_residual_gpu(cublasHandle_t h, int aux, int ext, int nact_r, const double* metric_host,
                          const double* fit_r_host, const double* raw_r_host, double* res_r_host,
                          Size budget_bytes) {
  const int j_chunk = compute_residual_chunk(aux, ext, nact_r, budget_bytes);
  if (j_chunk < 1) return false;
  const Size aa = Size(aux) * aux;
  const Size ae = Size(aux) * ext;

  DeviceBuf d_M, d_D, d_K;
  if (d_M.alloc(aa) != cudaSuccess || d_D.alloc(Size(j_chunk) * ae) != cudaSuccess ||
      d_K.alloc(Size(j_chunk) * ae) != cudaSuccess)
    return false;
  if (cudaMemcpy(d_M.ptr, metric_host, aa * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    return false;

  for (int j0 = 0; j0 < nact_r; j0 += j_chunk) {
    const int cur = std::min(j_chunk, nact_r - j0);
    const Size elems = Size(cur) * ae;
    if (cudaMemcpy(d_D.ptr, fit_r_host + Size(j0) * ae, elems * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_K.ptr, raw_r_host + Size(j0) * ae, elems * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
    if (rm_gemm_batched(h, false, false, aux, ext, aux, -1.0, d_M.ptr, aux, 0, d_D.ptr, ext,
                        static_cast<long long>(ae), 1.0, d_K.ptr, ext, static_cast<long long>(ae),
                        cur) != CUBLAS_STATUS_SUCCESS)
      return false;
    if (cudaMemcpy(res_r_host + Size(j0) * ae, d_K.ptr, elems * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      return false;
  }
  return true;
}

bool compute_fock_mix_gpu(cublasHandle_t h, int nact, Size ae, const double* foo_host,
                          const double* src_host, double* dst_host, Size budget_bytes) {
  const int col_chunk = compute_fock_mix_chunk(nact, ae, budget_bytes);
  if (col_chunk < 1) return false;
  const Size ff = Size(nact) * nact;
  const bool single = (static_cast<Size>(col_chunk) == ae);

  DeviceBuf d_F, d_src, d_dst;
  if (d_F.alloc(ff) != cudaSuccess || d_src.alloc(Size(nact) * col_chunk) != cudaSuccess ||
      d_dst.alloc(Size(nact) * col_chunk) != cudaSuccess)
    return false;
  if (cudaMemcpy(d_F.ptr, foo_host, ff * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
    return false;

  std::vector<double> h_stage(single ? Size(0) : Size(nact) * col_chunk);
  for (Size c0 = 0; c0 < ae; c0 += static_cast<Size>(col_chunk)) {
    const int cur = static_cast<int>(std::min<Size>(static_cast<Size>(col_chunk), ae - c0));
    const Size elems = Size(nact) * cur;
    const double* in_ptr = src_host;
    if (!single) {
      pack_cols_row_major(nact, static_cast<std::int64_t>(ae), static_cast<std::int64_t>(c0), cur,
                          src_host, h_stage.data());
      in_ptr = h_stage.data();
    }
    if (cudaMemcpy(d_src.ptr, in_ptr, elems * sizeof(double), cudaMemcpyHostToDevice) !=
        cudaSuccess)
      return false;
    if (rm_gemm(h, true, false, nact, cur, nact, 1.0, d_F.ptr, nact, d_src.ptr, cur, 0.0,
                d_dst.ptr, cur) != CUBLAS_STATUS_SUCCESS)
      return false;
    double* out_ptr = single ? dst_host : h_stage.data();
    if (cudaMemcpy(out_ptr, d_dst.ptr, elems * sizeof(double), cudaMemcpyDeviceToHost) !=
        cudaSuccess)
      return false;
    if (!single) {
      unpack_cols_row_major(nact, static_cast<std::int64_t>(ae), static_cast<std::int64_t>(c0), cur,
                            h_stage.data(), dst_host);
    }
  }
  return true;
}

#endif

}  // namespace

#ifdef GQC_SINK_HOST_REFERENCE
#define COULOMB_FIT_FWD_EXPORT gqc_f12_cc_coulomb_fit_fwd_host_reference
#define COULOMB_FIT_REV_EXPORT gqc_f12_cc_coulomb_fit_rev_host_reference
#define FIXED_PAIR_SOURCES_EXPORT gqc_f12_cc_fixed_pair_sources_host_reference
#else
#define COULOMB_FIT_FWD_EXPORT gqc_f12_cc_coulomb_fit_fwd_cuda
#define COULOMB_FIT_REV_EXPORT gqc_f12_cc_coulomb_fit_rev_cuda
#define FIXED_PAIR_SOURCES_EXPORT gqc_f12_cc_fixed_pair_sources_cuda
#endif

// =========================================================================
// 1. Coulomb density-fitting forward solve: fit = W^T @ (W @ three_center)
//    whitener:     [aux, aux] row-major
//    three_center: [aux, ncols] row-major
//    fit_out:      [aux, ncols] row-major (overwritten)
// =========================================================================
extern "C" int COULOMB_FIT_FWD_EXPORT(int aux, std::int64_t ncols, const double* whitener,
                                      const double* three_center, double* fit_out,
                                      std::int64_t max_workspace_bytes) {
  try {
    if (aux <= 0 || ncols <= 0 || max_workspace_bytes < 0 || !whitener || !three_center || !fit_out)
      return kErrInvalidArg;

    Size total_elems = 0;
    if (!checked_mul(Size(aux), static_cast<Size>(ncols), total_elems)) return kErrInvalidArg;

    const int chunk_cols = compute_chunk_cols(aux, ncols, 1, 2, max_workspace_bytes);
    if (chunk_cols <= 0) return kErrOom;
    const bool single_chunk = (chunk_cols == ncols);

#ifdef GQC_SINK_HOST_REFERENCE
    std::vector<double> buf1(single_chunk ? Size(0) : Size(aux) * chunk_cols);
    std::vector<double> buf2(Size(aux) * chunk_cols);

    for (std::int64_t c0 = 0; c0 < ncols; c0 += chunk_cols) {
      const int cur = static_cast<int>(std::min<std::int64_t>(chunk_cols, ncols - c0));
      const double* in_ptr = three_center;
      double* out_ptr = fit_out;
      if (!single_chunk) {
        pack_cols_row_major(aux, ncols, c0, cur, three_center, buf1.data());
        in_ptr = buf1.data();
        out_ptr = buf1.data();
      }
      rm_gemm_host(false, false, aux, cur, aux, 1.0, whitener, aux, in_ptr, cur, 0.0, buf2.data(),
                   cur);
      rm_gemm_host(true, false, aux, cur, aux, 1.0, whitener, aux, buf2.data(), cur, 0.0, out_ptr,
                   cur);
      if (!single_chunk) {
        unpack_cols_row_major(aux, ncols, c0, cur, out_ptr, fit_out);
      }
    }
#else
    CublasGuard cg;
    if (cg.init() != CUBLAS_STATUS_SUCCESS) return 1;
    cublasHandle_t h = cg.handle;

    const Size aa = Size(aux) * aux;
    const Size chunk_elems = Size(aux) * chunk_cols;
    DeviceBuf d_W, d_buf1, d_buf2;
    if (d_W.alloc(aa) != cudaSuccess || d_buf1.alloc(chunk_elems) != cudaSuccess ||
        d_buf2.alloc(chunk_elems) != cudaSuccess)
      return kErrOom;
    if (cudaMemcpy(d_W.ptr, whitener, aa * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
      return 3;

    std::vector<double> h_stage(single_chunk ? Size(0) : chunk_elems);
    for (std::int64_t c0 = 0; c0 < ncols; c0 += chunk_cols) {
      const int cur = static_cast<int>(std::min<std::int64_t>(chunk_cols, ncols - c0));
      const Size cur_elems = Size(aux) * cur;
      const double* src = three_center;
      if (!single_chunk) {
        pack_cols_row_major(aux, ncols, c0, cur, three_center, h_stage.data());
        src = h_stage.data();
      }
      if (cudaMemcpy(d_buf1.ptr, src, cur_elems * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess)
        return 3;
      if (rm_gemm(h, false, false, aux, cur, aux, 1.0, d_W.ptr, aux, d_buf1.ptr, cur, 0.0,
                  d_buf2.ptr, cur) != CUBLAS_STATUS_SUCCESS)
        return 4;
      if (rm_gemm(h, true, false, aux, cur, aux, 1.0, d_W.ptr, aux, d_buf2.ptr, cur, 0.0,
                  d_buf1.ptr, cur) != CUBLAS_STATUS_SUCCESS)
        return 4;
      double* dst = single_chunk ? fit_out : h_stage.data();
      if (cudaMemcpy(dst, d_buf1.ptr, cur_elems * sizeof(double), cudaMemcpyDeviceToHost) !=
          cudaSuccess)
        return 5;
      if (!single_chunk) {
        unpack_cols_row_major(aux, ncols, c0, cur, h_stage.data(), fit_out);
      }
    }
#endif
    if (!check_finite(fit_out, total_elems)) return kErrNonFinite;
    return kOk;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

// =========================================================================
// 2. Coulomb density-fitting implicit reverse:
//    three_center_bar = W^T @ (W @ fit_bar)
//    fit = fit_precomputed if non-null else W^T @ (W @ three_center)
//    weight = fit @ three_center_bar^T
//    metric_bar = -0.5 * (weight + weight^T)
// =========================================================================
extern "C" int COULOMB_FIT_REV_EXPORT(int aux, std::int64_t ncols, const double* whitener,
                                      const double* three_center, const double* fit_precomputed,
                                      const double* fit_bar, double* three_center_bar_out,
                                      double* metric_bar_out, std::int64_t max_workspace_bytes) {
  try {
    if (aux <= 0 || ncols <= 0 || max_workspace_bytes < 0 || !whitener ||
        (!three_center && !fit_precomputed) || !fit_bar || !three_center_bar_out ||
        !metric_bar_out)
      return kErrInvalidArg;

    Size total_elems = 0;
    const Size aa = Size(aux) * Size(aux);
    if (!checked_mul(Size(aux), static_cast<Size>(ncols), total_elems)) return kErrInvalidArg;

    const bool has_fit = (fit_precomputed != nullptr);
    const int chunk_bufs = has_fit ? 2 : 3;
    const int chunk_cols = compute_chunk_cols(aux, ncols, 2, chunk_bufs, max_workspace_bytes);
    if (chunk_cols <= 0) return kErrOom;
    const bool single_chunk = (chunk_cols == ncols);

#ifdef GQC_SINK_HOST_REFERENCE
    const Size chunk_elems = Size(aux) * chunk_cols;
    std::vector<double> buf1(single_chunk ? Size(0) : chunk_elems);
    std::vector<double> buf2(chunk_elems);
    std::vector<double> buf3(has_fit ? Size(0) : chunk_elems);

    for (std::int64_t c0 = 0; c0 < ncols; c0 += chunk_cols) {
      const int cur = static_cast<int>(std::min<std::int64_t>(chunk_cols, ncols - c0));
      const double* fb_ptr = fit_bar;
      double* raw_bar_ptr = three_center_bar_out;
      if (!single_chunk) {
        pack_cols_row_major(aux, ncols, c0, cur, fit_bar, buf1.data());
        fb_ptr = buf1.data();
        raw_bar_ptr = buf1.data();
      }
      rm_gemm_host(false, false, aux, cur, aux, 1.0, whitener, aux, fb_ptr, cur, 0.0, buf2.data(),
                   cur);
      rm_gemm_host(true, false, aux, cur, aux, 1.0, whitener, aux, buf2.data(), cur, 0.0,
                   raw_bar_ptr, cur);
      if (!single_chunk) {
        unpack_cols_row_major(aux, ncols, c0, cur, raw_bar_ptr, three_center_bar_out);
      }

      const double* fit_ptr = nullptr;
      if (has_fit) {
        if (single_chunk) {
          fit_ptr = fit_precomputed;
        } else {
          pack_cols_row_major(aux, ncols, c0, cur, fit_precomputed, buf2.data());
          fit_ptr = buf2.data();
        }
      } else {
        const double* tc_ptr = three_center;
        if (!single_chunk) {
          pack_cols_row_major(aux, ncols, c0, cur, three_center, buf2.data());
          tc_ptr = buf2.data();
        }
        rm_gemm_host(false, false, aux, cur, aux, 1.0, whitener, aux, tc_ptr, cur, 0.0, buf3.data(),
                     cur);
        rm_gemm_host(true, false, aux, cur, aux, 1.0, whitener, aux, buf3.data(), cur, 0.0,
                     buf2.data(), cur);
        fit_ptr = buf2.data();
      }

      const double beta = (c0 == 0) ? 0.0 : 1.0;
      rm_gemm_host(false, true, aux, aux, cur, 1.0, fit_ptr, cur, raw_bar_ptr, cur, beta,
                   metric_bar_out, aux);
    }

    for (int a = 0; a < aux; ++a) {
      for (int b = a; b < aux; ++b) {
        const Size ab = Size(a) * aux + b;
        const Size ba = Size(b) * aux + a;
        const double v = -0.5 * (metric_bar_out[ab] + metric_bar_out[ba]);
        metric_bar_out[ab] = v;
        metric_bar_out[ba] = v;
      }
    }
#else
    CublasGuard cg;
    if (cg.init() != CUBLAS_STATUS_SUCCESS) return 1;
    cublasHandle_t h = cg.handle;

    const Size chunk_elems = Size(aux) * chunk_cols;
    DeviceBuf d_W, d_weight, d_buf1, d_buf2, d_buf3;
    if (d_W.alloc(aa) != cudaSuccess || d_weight.alloc(aa) != cudaSuccess ||
        d_buf1.alloc(chunk_elems) != cudaSuccess || d_buf2.alloc(chunk_elems) != cudaSuccess)
      return kErrOom;
    if (!has_fit && d_buf3.alloc(chunk_elems) != cudaSuccess) return kErrOom;
    if (cudaMemcpy(d_W.ptr, whitener, aa * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(d_weight.ptr, 0, aa * sizeof(double)) != cudaSuccess)
      return 3;

    std::vector<double> h_stage(single_chunk ? Size(0) : chunk_elems);
    for (std::int64_t c0 = 0; c0 < ncols; c0 += chunk_cols) {
      const int cur = static_cast<int>(std::min<std::int64_t>(chunk_cols, ncols - c0));
      const Size cur_elems = Size(aux) * cur;

      const double* fb_src = fit_bar;
      if (!single_chunk) {
        pack_cols_row_major(aux, ncols, c0, cur, fit_bar, h_stage.data());
        fb_src = h_stage.data();
      }
      if (cudaMemcpy(d_buf1.ptr, fb_src, cur_elems * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess)
        return 3;
      if (rm_gemm(h, false, false, aux, cur, aux, 1.0, d_W.ptr, aux, d_buf1.ptr, cur, 0.0,
                  d_buf2.ptr, cur) != CUBLAS_STATUS_SUCCESS)
        return 4;
      if (rm_gemm(h, true, false, aux, cur, aux, 1.0, d_W.ptr, aux, d_buf2.ptr, cur, 0.0,
                  d_buf1.ptr, cur) != CUBLAS_STATUS_SUCCESS)
        return 4;
      double* rb_dst = single_chunk ? three_center_bar_out : h_stage.data();
      if (cudaMemcpy(rb_dst, d_buf1.ptr, cur_elems * sizeof(double), cudaMemcpyDeviceToHost) !=
          cudaSuccess)
        return 5;
      if (!single_chunk) {
        unpack_cols_row_major(aux, ncols, c0, cur, h_stage.data(), three_center_bar_out);
      }

      const double* rhs_src = has_fit ? fit_precomputed : three_center;
      const double* upload_src = rhs_src;
      if (!single_chunk) {
        pack_cols_row_major(aux, ncols, c0, cur, rhs_src, h_stage.data());
        upload_src = h_stage.data();
      }
      if (cudaMemcpy(d_buf2.ptr, upload_src, cur_elems * sizeof(double), cudaMemcpyHostToDevice) !=
          cudaSuccess)
        return 6;
      if (!has_fit) {
        if (rm_gemm(h, false, false, aux, cur, aux, 1.0, d_W.ptr, aux, d_buf2.ptr, cur, 0.0,
                    d_buf3.ptr, cur) != CUBLAS_STATUS_SUCCESS)
          return 7;
        if (rm_gemm(h, true, false, aux, cur, aux, 1.0, d_W.ptr, aux, d_buf3.ptr, cur, 0.0,
                    d_buf2.ptr, cur) != CUBLAS_STATUS_SUCCESS)
          return 7;
      }

      const double beta = (c0 == 0) ? 0.0 : 1.0;
      if (rm_gemm(h, false, true, aux, aux, cur, 1.0, d_buf2.ptr, cur, d_buf1.ptr, cur, beta,
                  d_weight.ptr, aux) != CUBLAS_STATUS_SUCCESS)
        return 8;
    }

    k_symmetrize_neg_half_inplace<<<grid_for(aa), 256>>>(aux, d_weight.ptr);
    if (cudaGetLastError() != cudaSuccess) return 9;
    if (cudaMemcpy(metric_bar_out, d_weight.ptr, aa * sizeof(double), cudaMemcpyDeviceToHost) !=
        cudaSuccess)
      return 10;
#endif
    if (!check_finite(three_center_bar_out, total_elems) || !check_finite(metric_bar_out, aa)) {
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

// =========================================================================
// 3. Fixed-cusp CC-F12 pair sources (W_obs, C_vv, and scalar V + B - X)
//    Evaluates all active pairs (pairs_i[k], pairs_j[k]) in one spin channel
//    with prefactored operator residuals and strictly VRAM-bounded tiling
//    (<= 512 MiB default).
//
//    Input factor arrays fit_*, raw_* have ACTIVE-only shape [nact, aux, ext]
//    indexed by ii = pairs_i[k] - fl and jj = pairs_j[k] - fr.
//    energies_out[0..2] is overwritten with [linear_v, b_energy, x_fock_energy].
// =========================================================================
extern "C" int FIXED_PAIR_SOURCES_EXPORT(
    int same_spin_int, int compute_wc_int, int compute_quad_int, int nl, int nr, int fl, int fr,
    int obs, int ext, int aux, int n_pairs, const std::int32_t* pairs_i,
    const std::int32_t* pairs_j, const double* pair_mult, const double* fit_l, const double* fit_r,
    const double* raw_coul_r, const double* raw_g_l, const double* raw_g_r, const double* raw_gc_l,
    const double* raw_gc_r, const double* raw_sq_l, const double* raw_sq_r, const double* raw_dc_l,
    const double* raw_dc_r, const double* metric_g, const double* metric_gc,
    const double* metric_sq, const double* metric_dc, const double* fock_l, const double* fock_r,
    const double* exch_l, const double* exch_r, const double* cross_overlap,
    const double* fit_obs_l, const double* raw_obs_r, double* w_obs_out, double* c_vv_out,
    double* energies_out, std::int64_t max_workspace_bytes) {
  try {
    const bool same_spin = (same_spin_int != 0);
    const bool compute_wc = (compute_wc_int != 0);
    const bool compute_quad = (compute_quad_int != 0);

    if (nl <= 0 || nr <= 0 || fl < 0 || fl >= nl || fr < 0 || fr >= nr || obs <= nl || obs <= nr ||
        ext <= obs || aux <= 0 || n_pairs < 0 || max_workspace_bytes < 0 || !energies_out)
      return kErrInvalidArg;
    if (same_spin && (nl != nr || fl != fr)) return kErrInvalidArg;
    if (n_pairs == 0) {
      energies_out[0] = 0.0;
      energies_out[1] = 0.0;
      energies_out[2] = 0.0;
      return kOk;
    }

    const double* p_fit_r = same_spin ? fit_l : fit_r;
    const double* p_raw_g_r = same_spin ? raw_g_l : raw_g_r;
    const double* p_raw_gc_r = same_spin ? raw_gc_l : raw_gc_r;
    const double* p_raw_sq_r = same_spin ? raw_sq_l : raw_sq_r;
    const double* p_raw_dc_r = same_spin ? raw_dc_l : raw_dc_r;
    const double* p_fock_r = same_spin ? fock_l : fock_r;
    const double* p_exch_r = same_spin ? exch_l : exch_r;

    if (!pairs_i || !pairs_j || !fit_l || !p_fit_r || !raw_coul_r || !raw_g_l || !p_raw_g_r ||
        !raw_gc_l || !p_raw_gc_r || !metric_g || !metric_gc)
      return kErrInvalidArg;
    if (!same_spin && !cross_overlap) return kErrInvalidArg;
    if (compute_wc &&
        (!fit_obs_l || !raw_obs_r || !fock_l || !p_fock_r || !w_obs_out || !c_vv_out))
      return kErrInvalidArg;
    if (compute_quad && (!fock_l || !p_fock_r || !exch_l || !p_exch_r || !raw_sq_l || !p_raw_sq_r ||
                         !raw_dc_l || !p_raw_dc_r || !metric_sq || !metric_dc))
      return kErrInvalidArg;

    for (int k = 0; k < n_pairs; ++k) {
      if (pairs_i[k] < fl || pairs_i[k] >= nl || pairs_j[k] < fr || pairs_j[k] >= nr)
        return kErrInvalidArg;
    }

    const int nact_l = nl - fl;
    const int nact_r = nr - fr;
    const int nv_l = obs - nl;
    const int nv_r = obs - nr;
    const int nc = ext - obs;
    const Size ae = Size(aux) * ext;
    const Size ee = Size(ext) * ext;
    const Size oo = Size(obs) * obs;
    const Size vv = Size(nv_l) * nv_r;
    const double norm = same_spin ? 0.5 : 1.0;

    const Size budget =
        (max_workspace_bytes > 0) ? static_cast<Size>(max_workspace_bytes) : kDefaultBudgetBytes;
    if (!fixed_pair_sources_fits_budget(same_spin, compute_wc, compute_quad, nact_l, nact_r, obs,
                                        ext, aux, nv_l, nv_r, budget)) {
      return kErrOom;
    }

    std::vector<double> F_oo_l, F_oo_r;
    if (compute_quad) {
      F_oo_l.resize(Size(nact_l) * nact_l);
      for (int r = 0; r < nact_l; ++r) {
        for (int c = 0; c < nact_l; ++c) {
          F_oo_l[Size(r) * nact_l + c] = fock_l[Size(fl + r) * ext + (fl + c)];
        }
      }
      if (!same_spin) {
        F_oo_r.resize(Size(nact_r) * nact_r);
        for (int r = 0; r < nact_r; ++r) {
          for (int c = 0; c < nact_r; ++c) {
            F_oo_r[Size(r) * nact_r + c] = p_fock_r[Size(fr + r) * ext + (fr + c)];
          }
        }
      }
    }

#ifdef GQC_SINK_HOST_REFERENCE
    // Stage 0: prefactor operator residuals K_op_r[jj] = raw_op_r[jj] - M_op @ fit_r[jj]
    auto make_res_host = [&](const double* metric_op, const double* raw_r) {
      std::vector<double> K_r(Size(nact_r) * ae);
      for (int jj = 0; jj < nact_r; ++jj) {
        const double* Dr = p_fit_r + Size(jj) * ae;
        const double* Or = raw_r + Size(jj) * ae;
        double* Kr = K_r.data() + Size(jj) * ae;
        std::copy_n(Or, ae, Kr);
        rm_gemm_host(false, false, aux, ext, aux, -1.0, metric_op, aux, Dr, ext, 1.0, Kr, ext);
      }
      return K_r;
    };

    std::vector<double> K_g_r = make_res_host(metric_g, p_raw_g_r);
    std::vector<double> K_gc_r = make_res_host(metric_gc, p_raw_gc_r);
    std::vector<double> K_sq_r, K_dc_r;
    std::vector<double> XD_l, XD_r, XO_g_l, XK_g_r, XO_sq_l, XK_sq_r;

    if (compute_quad) {
      K_sq_r = make_res_host(metric_sq, p_raw_sq_r);
      K_dc_r = make_res_host(metric_dc, p_raw_dc_r);

      auto make_fmix_host = [&](int nact, const double* foo, const double* src) {
        std::vector<double> dst(Size(nact) * ae);
        const Size max_int = static_cast<Size>(std::numeric_limits<int>::max());
        if (ae <= max_int) {
          const int ae_int = static_cast<int>(ae);
          rm_gemm_host(true, false, nact, ae_int, nact, 1.0, foo, nact, src, ae_int, 0.0,
                       dst.data(), ae_int);
        } else {
          const int col_chunk = static_cast<int>(std::min<Size>(max_int, Size(1) << 20));
          std::vector<double> stage_in(Size(nact) * col_chunk);
          std::vector<double> stage_out(Size(nact) * col_chunk);
          for (Size c0 = 0; c0 < ae; c0 += static_cast<Size>(col_chunk)) {
            const int cur = static_cast<int>(std::min<Size>(static_cast<Size>(col_chunk), ae - c0));
            pack_cols_row_major(nact, static_cast<std::int64_t>(ae),
                                static_cast<std::int64_t>(c0), cur, src, stage_in.data());
            rm_gemm_host(true, false, nact, cur, nact, 1.0, foo, nact, stage_in.data(), cur, 0.0,
                         stage_out.data(), cur);
            unpack_cols_row_major(nact, static_cast<std::int64_t>(ae),
                                  static_cast<std::int64_t>(c0), cur, stage_out.data(), dst.data());
          }
        }
        return dst;
      };
      const double* p_foo_r = same_spin ? F_oo_l.data() : F_oo_r.data();
      XD_l = make_fmix_host(nact_l, F_oo_l.data(), fit_l);
      XO_g_l = make_fmix_host(nact_l, F_oo_l.data(), raw_g_l);
      XO_sq_l = make_fmix_host(nact_l, F_oo_l.data(), raw_sq_l);
      if (!same_spin) XD_r = make_fmix_host(nact_r, p_foo_r, p_fit_r);
      XK_g_r = make_fmix_host(nact_r, p_foo_r, K_g_r.data());
      XK_sq_r = make_fmix_host(nact_r, p_foo_r, K_sq_r.data());
    }

    std::vector<double> FK_l, FK_r;
    if (compute_quad) {
      FK_l.resize(ee);
      for (Size t = 0; t < ee; ++t) FK_l[t] = fock_l[t] + exch_l[t];
      if (!same_spin) {
        FK_r.resize(ee);
        for (Size t = 0; t < ee; ++t) FK_r[t] = p_fock_r[t] + p_exch_r[t];
      }
    }
    const double* p_FK_r = same_spin ? FK_l.data() : FK_r.data();
    const double* p_XD_r = same_spin ? XD_l.data() : XD_r.data();

    // Stage 1: per-pair forward contractions
    std::vector<double> f_obs_all(compute_wc ? Size(n_pairs) * oo : Size(0));
    std::vector<double> G_raw(ee), f(ee), tmp1(ee), tmp2(ee);
    std::vector<double> qf(compute_quad ? ee : Size(0)), chi(compute_quad ? ee : Size(0));

    double e_v = 0.0, e_b = 0.0, e_x = 0.0;

    for (int k = 0; k < n_pairs; ++k) {
      const int i = pairs_i[k];
      const int j = pairs_j[k];
      const int ii = i - fl;
      const int jj = j - fr;
      const double mult = pair_mult ? pair_mult[k] : 1.0;

      const double* Dl = fit_l + Size(ii) * ae;
      const double* Dr = p_fit_r + Size(jj) * ae;
      const double* Rcr = raw_coul_r + Size(jj) * ae;
      const double* Ogl = raw_g_l + Size(ii) * ae;
      const double* Kgr = K_g_r.data() + Size(jj) * ae;
      const double* Ogcl = raw_gc_l + Size(ii) * ae;
      const double* Kgcr = K_gc_r.data() + Size(jj) * ae;

      // 1. Geminal f
      robust_from_res_host(aux, ext, Dl, Ogl, Dr, Kgr, 0.0, G_raw.data());
      dress_fwd_host(same_spin, ext, cross_overlap, G_raw.data(), f.data(), tmp1.data());

      if (compute_wc) {
        copy_subblock_host(obs, obs, f.data(), ext, f_obs_all.data() + Size(k) * oo, obs);
        double* ck = c_vv_out + Size(k) * vv;
        rm_gemm_host(false, false, nv_l, nv_r, nc, 1.0, fock_l + Size(nl) * ext + obs, ext,
                     f.data() + Size(obs) * ext + nr, ext, 0.0, ck, nv_r);
        rm_gemm_host(false, true, nv_l, nv_r, nc, 1.0, f.data() + Size(nl) * ext + obs, ext,
                     p_fock_r + Size(nr) * ext + obs, ext, 1.0, ck, nv_r);
      }

      // 2. Geminal-Coulomb fg
      robust_from_res_host(aux, ext, Dl, Ogcl, Dr, Kgcr, 0.0, G_raw.data());
      dress_fwd_host(same_spin, ext, cross_overlap, G_raw.data(), tmp2.data(), tmp1.data());
      if (compute_wc) {
        copy_subblock_host(obs, obs, tmp2.data(), ext, w_obs_out + Size(k) * oo, obs);
      }

      // 3. Linear V energy
      rm_gemm_host(true, false, ext, ext, aux, 1.0, Dl, ext, Rcr, ext, 0.0, tmp1.data(), ext);
      const double v_dot = v_mask_dot_host(ext, nl, nr, obs, tmp1.data(), f.data());
      e_v += mult * 2.0 * (tmp2[Size(i) * ext + j] - v_dot);

      // 4. Quadratic B and X energies
      if (compute_quad) {
        const double* Osql = raw_sq_l + Size(ii) * ae;
        const double* Ksqr = K_sq_r.data() + Size(jj) * ae;
        const double* Odcl = raw_dc_l + Size(ii) * ae;
        const double* Kdcr = K_dc_r.data() + Size(jj) * ae;
        const double* XDl = XD_l.data() + Size(ii) * ae;
        const double* XDr = p_XD_r + Size(jj) * ae;
        const double* XOgl = XO_g_l.data() + Size(ii) * ae;
        const double* XKgr = XK_g_r.data() + Size(jj) * ae;
        const double* XOsql = XO_sq_l.data() + Size(ii) * ae;
        const double* XKsqr = XK_sq_r.data() + Size(jj) * ae;

        if (same_spin) {
          std::fill(chi.begin(), chi.end(), 0.0);
          chi[Size(i) * ext + j] += 0.25;
          chi[Size(j) * ext + i] -= 0.25;
        } else {
          for (int p = 0; p < ext; ++p) {
            for (int q = 0; q < ext; ++q) {
              chi[Size(p) * ext + q] =
                  0.125 * cross_overlap[Size(p) * ext + j] * cross_overlap[Size(i) * ext + q] +
                  ((p == i && q == j) ? 0.375 : 0.0);
            }
          }
        }
        strong_mask_host(ext, nl, nr, obs, f.data(), qf.data(), false);

        // B term 1: +dot(chi, uf)
        robust_from_res_host(aux, ext, Dl, Odcl, Dr, Kdcr, 0.0, G_raw.data());
        dress_fwd_host(same_spin, ext, cross_overlap, G_raw.data(), tmp2.data(), tmp1.data());
        double eb_k = dot_host(ee, chi.data(), tmp2.data());

        // B term 2: +dot(ff, action(FK_l, FK_r, chi))
        robust_from_res_host(aux, ext, Dl, Osql, Dr, Ksqr, 0.0, G_raw.data());
        dress_fwd_host(same_spin, ext, cross_overlap, G_raw.data(), tmp2.data(), tmp1.data());
        rm_gemm_host(false, false, ext, ext, ext, 1.0, FK_l.data(), ext, chi.data(), ext, 0.0,
                     G_raw.data(), ext);
        rm_gemm_host(false, true, ext, ext, ext, 1.0, chi.data(), ext, p_FK_r, ext, 1.0,
                     G_raw.data(), ext);
        eb_k += dot_host(ee, tmp2.data(), G_raw.data());

        // B terms 3 + 5: -dot(f, action(FK_l, FK_r, f))
        rm_gemm_host(false, false, ext, ext, ext, 1.0, FK_l.data(), ext, f.data(), ext, 0.0,
                     G_raw.data(), ext);
        rm_gemm_host(false, true, ext, ext, ext, 1.0, f.data(), ext, p_FK_r, ext, 1.0, G_raw.data(),
                     ext);
        eb_k -= dot_host(ee, f.data(), G_raw.data());

        // B term 4: +dot(qf, action(F_l, F_r, qf))
        rm_gemm_host(false, false, ext, ext, ext, 1.0, fock_l, ext, qf.data(), ext, 0.0,
                     G_raw.data(), ext);
        rm_gemm_host(false, true, ext, ext, ext, 1.0, qf.data(), ext, p_fock_r, ext, 1.0,
                     G_raw.data(), ext);
        eb_k += dot_host(ee, qf.data(), G_raw.data());

        e_b += mult * norm * eb_k;

        // X term 1: +dot(chi, ffx)
        robust_from_res_host(aux, ext, XDl, XOsql, Dr, Ksqr, 0.0, G_raw.data());
        robust_from_res_host(aux, ext, Dl, Osql, XDr, XKsqr, 1.0, G_raw.data());
        dress_fwd_host(same_spin, ext, cross_overlap, G_raw.data(), tmp2.data(), tmp1.data());
        double ex_k = dot_host(ee, chi.data(), tmp2.data());

        // X term 2: -dot(f - qf, fx)
        robust_from_res_host(aux, ext, XDl, XOgl, Dr, Kgr, 0.0, G_raw.data());
        robust_from_res_host(aux, ext, Dl, Ogl, XDr, XKgr, 1.0, G_raw.data());
        dress_fwd_host(same_spin, ext, cross_overlap, G_raw.data(), tmp2.data(), tmp1.data());
        strong_mask_host(ext, nl, nr, obs, f.data(), qf.data(), true);
        ex_k -= dot_host(ee, qf.data(), tmp2.data());

        e_x += mult * norm * ex_k;
      }
    }

    // Stage 2: OBS Coulomb projection w_obs[k] -= sum_a fit_obs_l[a] @ f_obs[k] @ raw_obs_r[a]^T
    if (compute_wc) {
      std::vector<double> T(oo);
      for (int k = 0; k < n_pairs; ++k) {
        const double* fk = f_obs_all.data() + Size(k) * oo;
        double* wk = w_obs_out + Size(k) * oo;
        for (int a = 0; a < aux; ++a) {
          const double* Da = fit_obs_l + Size(a) * oo;
          const double* Ra = raw_obs_r + Size(a) * oo;
          rm_gemm_host(false, true, obs, obs, obs, 1.0, fk, obs, Ra, obs, 0.0, T.data(), obs);
          rm_gemm_host(false, false, obs, obs, obs, -1.0, Da, obs, T.data(), obs, 1.0, wk, obs);
        }
      }
    }

    energies_out[0] = e_v;
    energies_out[1] = e_b;
    energies_out[2] = e_x;
#else
    CublasGuard cg;
    if (cg.init() != CUBLAS_STATUS_SUCCESS) return 1;
    cublasHandle_t h = cg.handle;

    // Stage 0: prefactor operator residuals K_op_r and F_oo-mixed tensors on GPU (bounded VRAM).
    std::vector<double> K_g_r(Size(nact_r) * ae);
    std::vector<double> K_gc_r(Size(nact_r) * ae);
    if (!compute_residual_gpu(h, aux, ext, nact_r, metric_g, p_fit_r, p_raw_g_r, K_g_r.data(),
                              budget) ||
        !compute_residual_gpu(h, aux, ext, nact_r, metric_gc, p_fit_r, p_raw_gc_r, K_gc_r.data(),
                              budget))
      return 2;

    std::vector<double> K_sq_r, K_dc_r;
    std::vector<double> XD_l, XD_r, XO_g_l, XK_g_r, XO_sq_l, XK_sq_r;
    if (compute_quad) {
      K_sq_r.resize(Size(nact_r) * ae);
      K_dc_r.resize(Size(nact_r) * ae);
      if (!compute_residual_gpu(h, aux, ext, nact_r, metric_sq, p_fit_r, p_raw_sq_r, K_sq_r.data(),
                                budget) ||
          !compute_residual_gpu(h, aux, ext, nact_r, metric_dc, p_fit_r, p_raw_dc_r, K_dc_r.data(),
                                budget))
        return 2;

      const double* p_foo_r = same_spin ? F_oo_l.data() : F_oo_r.data();
      XD_l.resize(Size(nact_l) * ae);
      XO_g_l.resize(Size(nact_l) * ae);
      XO_sq_l.resize(Size(nact_l) * ae);
      XK_g_r.resize(Size(nact_r) * ae);
      XK_sq_r.resize(Size(nact_r) * ae);
      if (!compute_fock_mix_gpu(h, nact_l, ae, F_oo_l.data(), fit_l, XD_l.data(), budget) ||
          !compute_fock_mix_gpu(h, nact_l, ae, F_oo_l.data(), raw_g_l, XO_g_l.data(), budget) ||
          !compute_fock_mix_gpu(h, nact_l, ae, F_oo_l.data(), raw_sq_l, XO_sq_l.data(), budget) ||
          !compute_fock_mix_gpu(h, nact_r, ae, p_foo_r, K_g_r.data(), XK_g_r.data(), budget) ||
          !compute_fock_mix_gpu(h, nact_r, ae, p_foo_r, K_sq_r.data(), XK_sq_r.data(), budget))
        return 2;
      if (!same_spin) {
        XD_r.resize(Size(nact_r) * ae);
        if (!compute_fock_mix_gpu(h, nact_r, ae, p_foo_r, p_fit_r, XD_r.data(), budget)) return 2;
      }
    }
    const double* h_XD_r = same_spin ? XD_l.data() : XD_r.data();

    // Stage 1: GPU-resident (or orbital-tiled / 4-slice bounded) pair forward pass.
    std::vector<double> f_obs_all(compute_wc ? Size(n_pairs) * oo : Size(0));
    double h_energies[3] = {0.0, 0.0, 0.0};

    {
      auto chk = [](cudaError_t e) { return e == cudaSuccess; };
      auto h2d = [](double* dst, const double* src, Size n) {
        return cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess;
      };

      DeviceBuf d_fl, d_fr, d_kl, d_kr, d_FKl, d_FKr, d_S;
      DeviceBuf d_G_raw, d_f, d_tmp1, d_tmp2, d_qf, d_chi, d_sub, d_scalar_acc;

      Size fixed_elems = 4 * ee + 4;
      if (compute_wc || compute_quad) fixed_elems += same_spin ? ee : 2 * ee;
      if (!same_spin) fixed_elems += ee;
      if (compute_quad) fixed_elems += (same_spin ? 2 * ee : 4 * ee) + 2 * ee;
      const Size sub_elems = compute_wc ? std::max(oo, vv) : Size(0);
      fixed_elems += sub_elems;

      const Size fixed_bytes = fixed_elems * sizeof(double);
      const Size slice_bytes = ae * sizeof(double);
      const Size avail_bytes = (budget > fixed_bytes) ? (budget - fixed_bytes) : Size(0);
      const Size avail_slices = avail_bytes / std::max<Size>(Size(1), slice_bytes);
      if (fixed_bytes > budget || avail_slices < Size(4)) return kErrOom;

      if (!chk(d_G_raw.alloc(ee)) || !chk(d_f.alloc(ee)) || !chk(d_tmp1.alloc(ee)) ||
          !chk(d_tmp2.alloc(ee)) || !chk(d_scalar_acc.alloc(4)))
        return kErrOom;
      if (cudaMemset(d_scalar_acc.ptr, 0, 4 * sizeof(double)) != cudaSuccess) return 3;
      if (compute_wc && !chk(d_sub.alloc(sub_elems))) return kErrOom;
      if (compute_wc || compute_quad) {
        if (!chk(d_fl.alloc(ee)) || !h2d(d_fl.ptr, fock_l, ee)) return 3;
        if (!same_spin) {
          if (!chk(d_fr.alloc(ee)) || !h2d(d_fr.ptr, p_fock_r, ee)) return 3;
        }
      }
      if (!same_spin) {
        if (!chk(d_S.alloc(ee)) || !h2d(d_S.ptr, cross_overlap, ee)) return 3;
      }
      if (compute_quad) {
        if (!chk(d_kl.alloc(ee)) || !chk(d_FKl.alloc(ee)) || !chk(d_qf.alloc(ee)) ||
            !chk(d_chi.alloc(ee)) || !h2d(d_kl.ptr, exch_l, ee))
          return 3;
        k_add_two<<<grid_for(ee), 256>>>(ee, d_fl.ptr, d_kl.ptr, d_FKl.ptr);
        if (!same_spin) {
          if (!chk(d_kr.alloc(ee)) || !chk(d_FKr.alloc(ee)) || !h2d(d_kr.ptr, p_exch_r, ee))
            return 3;
          k_add_two<<<grid_for(ee), 256>>>(ee, d_fr.ptr, d_kr.ptr, d_FKr.ptr);
        }
        if (cudaGetLastError() != cudaSuccess) return 3;
      }

      const double* d_p_fr = same_spin ? d_fl.ptr : d_fr.ptr;
      const double* d_p_FKr = same_spin ? d_FKl.ptr : d_FKr.ptr;

      const Size L_cnt = compute_quad ? Size(8) : Size(3);
      const Size R_cnt = compute_quad ? Size(9) : Size(4);
      const bool tile_mode = (avail_slices >= L_cnt + R_cnt);

      int ti = 1, tj = 1;
      if (tile_mode) {
        if (L_cnt * Size(nact_l) + R_cnt * Size(nact_r) <= avail_slices) {
          ti = nact_l;
          tj = nact_r;
        } else {
          tj = 1;
          const Size ti_sz =
              std::min<Size>(Size(nact_l), std::max<Size>(Size(1), (avail_slices - R_cnt) / L_cnt));
          ti = static_cast<int>(ti_sz);
          if (ti == nact_l) {
            const Size tj_sz = std::min<Size>(
                Size(nact_r), std::max<Size>(Size(1), (avail_slices - L_cnt * Size(ti)) / R_cnt));
            tj = static_cast<int>(tj_sz);
          }
        }
      }

      // Tile buffers (Mode A) or 4-slice scratch buffers (Mode B).
      DeviceBuf d_fit_l, d_g_l, d_gc_l, d_sq_l, d_dc_l, d_xf_l, d_xg_l, d_xsq_l;
      DeviceBuf d_fit_r, d_rc_r, d_Kg_r, d_Kgc_r, d_Ksq_r, d_Kdc_r, d_xf_r, d_XKg_r, d_XKsq_r;
      DeviceBuf d_s0, d_s1, d_s2, d_s3;

      if (tile_mode) {
        const Size l_elems = Size(ti) * ae;
        const Size r_elems = Size(tj) * ae;
        if (!chk(d_fit_l.alloc(l_elems)) || !chk(d_g_l.alloc(l_elems)) ||
            !chk(d_gc_l.alloc(l_elems)) || !chk(d_fit_r.alloc(r_elems)) ||
            !chk(d_rc_r.alloc(r_elems)) || !chk(d_Kg_r.alloc(r_elems)) ||
            !chk(d_Kgc_r.alloc(r_elems)))
          return kErrOom;
        if (compute_quad) {
          if (!chk(d_sq_l.alloc(l_elems)) || !chk(d_dc_l.alloc(l_elems)) ||
              !chk(d_xf_l.alloc(l_elems)) || !chk(d_xg_l.alloc(l_elems)) ||
              !chk(d_xsq_l.alloc(l_elems)) || !chk(d_Ksq_r.alloc(r_elems)) ||
              !chk(d_Kdc_r.alloc(r_elems)) || !chk(d_xf_r.alloc(r_elems)) ||
              !chk(d_XKg_r.alloc(r_elems)) || !chk(d_XKsq_r.alloc(r_elems)))
            return kErrOom;
        }
      } else {
        if (!chk(d_s0.alloc(ae)) || !chk(d_s1.alloc(ae)) || !chk(d_s2.alloc(ae)) ||
            !chk(d_s3.alloc(ae)))
          return kErrOom;
      }

      const unsigned g_ee = grid_for(ee);
      const unsigned g_red = std::min<unsigned>(g_ee, 256u);

      auto eval_pair = [&](int k, int li, int lj) -> int {
        const int i = pairs_i[k];
        const int j = pairs_j[k];
        const int ii = i - fl;
        const int jj = j - fr;
        const double mult = pair_mult ? pair_mult[k] : 1.0;

        bool ok = true;
        auto get_l = [&](const DeviceBuf& buf, const double* h_arr, double* slot) -> const double* {
          if (tile_mode) return buf.ptr + Size(li) * ae;
          if (!h2d(slot, h_arr + Size(ii) * ae, ae)) ok = false;
          return slot;
        };
        auto get_r = [&](const DeviceBuf& buf, const double* h_arr, double* slot) -> const double* {
          if (tile_mode) return buf.ptr + Size(lj) * ae;
          if (!h2d(slot, h_arr + Size(jj) * ae, ae)) ok = false;
          return slot;
        };

        // Load Dl (s0) and Dr (s2) once for g, gc, C, dc, sq.
        const double* Dl = get_l(d_fit_l, fit_l, d_s0.ptr);
        const double* Dr = get_r(d_fit_r, p_fit_r, d_s2.ptr);
        if (!ok) return 4;

        // 1. Geminal f
        {
          const double* Ogl = get_l(d_g_l, raw_g_l, d_s1.ptr);
          const double* Kgr = get_r(d_Kg_r, K_g_r.data(), d_s3.ptr);
          if (!ok || !robust_from_res_gpu(h, aux, ext, Dl, Ogl, Dr, Kgr, 0.0, d_G_raw.ptr) ||
              !dress_fwd_gpu(h, same_spin, ext, d_S.ptr, d_G_raw.ptr, d_f.ptr, d_tmp1.ptr))
            return 4;
        }

        if (compute_wc) {
          k_copy_subblock<<<grid_for(oo), 256>>>(obs, obs, d_f.ptr, ext, d_sub.ptr, obs);
          if (cudaGetLastError() != cudaSuccess ||
              cudaMemcpy(f_obs_all.data() + Size(k) * oo, d_sub.ptr, oo * sizeof(double),
                         cudaMemcpyDeviceToHost) != cudaSuccess)
            return 4;
          if (rm_gemm(h, false, false, nv_l, nv_r, nc, 1.0, d_fl.ptr + Size(nl) * ext + obs, ext,
                      d_f.ptr + Size(obs) * ext + nr, ext, 0.0, d_sub.ptr, nv_r) !=
                  CUBLAS_STATUS_SUCCESS ||
              rm_gemm(h, false, true, nv_l, nv_r, nc, 1.0, d_f.ptr + Size(nl) * ext + obs, ext,
                      d_p_fr + Size(nr) * ext + obs, ext, 1.0, d_sub.ptr, nv_r) !=
                  CUBLAS_STATUS_SUCCESS ||
              cudaMemcpy(c_vv_out + Size(k) * vv, d_sub.ptr, vv * sizeof(double),
                         cudaMemcpyDeviceToHost) != cudaSuccess)
            return 4;
        }

        // 2. Geminal-Coulomb fg -> d_tmp2
        {
          const double* Ogcl = get_l(d_gc_l, raw_gc_l, d_s1.ptr);
          const double* Kgcr = get_r(d_Kgc_r, K_gc_r.data(), d_s3.ptr);
          if (!ok || !robust_from_res_gpu(h, aux, ext, Dl, Ogcl, Dr, Kgcr, 0.0, d_G_raw.ptr) ||
              !dress_fwd_gpu(h, same_spin, ext, d_S.ptr, d_G_raw.ptr, d_tmp2.ptr, d_tmp1.ptr))
            return 4;
        }
        if (compute_wc) {
          k_copy_subblock<<<grid_for(oo), 256>>>(obs, obs, d_tmp2.ptr, ext, d_sub.ptr, obs);
          if (cudaGetLastError() != cudaSuccess ||
              cudaMemcpy(w_obs_out + Size(k) * oo, d_sub.ptr, oo * sizeof(double),
                         cudaMemcpyDeviceToHost) != cudaSuccess)
            return 4;
        }

        // 3. Linear V energy: C = Dl^T @ Rcr -> d_tmp1
        {
          const double* Rcr = get_r(d_rc_r, raw_coul_r, d_s3.ptr);
          if (!ok || rm_gemm(h, true, false, ext, ext, aux, 1.0, Dl, ext, Rcr, ext, 0.0, d_tmp1.ptr,
                             ext) != CUBLAS_STATUS_SUCCESS)
            return 4;
          k_v_energy_accum<<<g_red, 256>>>(ext, nl, nr, obs, i, j, mult, d_tmp2.ptr, d_tmp1.ptr,
                                           d_f.ptr, d_scalar_acc.ptr + 0);
          if (cudaGetLastError() != cudaSuccess) return 4;
        }

        // 4. Quadratic B and X energies
        if (compute_quad) {
          const double w_q = mult * norm;
          if (same_spin) {
            k_set_chi_same_spin<<<g_ee, 256>>>(ext, i, j, d_chi.ptr);
          } else {
            k_set_chi_opp_spin<<<g_ee, 256>>>(ext, i, j, d_S.ptr, d_chi.ptr);
          }
          k_strong_mask<<<g_ee, 256>>>(ext, nl, nr, obs, d_f.ptr, d_qf.ptr, false);
          if (cudaGetLastError() != cudaSuccess) return 4;

          // B term 1: +dot(chi, uf)
          {
            const double* Odcl = get_l(d_dc_l, raw_dc_l, d_s1.ptr);
            const double* Kdcr = get_r(d_Kdc_r, K_dc_r.data(), d_s3.ptr);
            if (!ok || !robust_from_res_gpu(h, aux, ext, Dl, Odcl, Dr, Kdcr, 0.0, d_G_raw.ptr) ||
                !dress_fwd_gpu(h, same_spin, ext, d_S.ptr, d_G_raw.ptr, d_tmp2.ptr, d_tmp1.ptr))
              return 4;
            k_dot_accum<<<g_red, 256>>>(ee, w_q, d_chi.ptr, d_tmp2.ptr, d_scalar_acc.ptr + 1);
          }

          // B term 2: +dot(ff, action(FK_l, FK_r, chi))
          // Keep Osql in s1 and Ksqr in s3 so X-ffx can reuse Ksqr in Mode B.
          const double* Osql = get_l(d_sq_l, raw_sq_l, d_s1.ptr);
          const double* Ksqr = get_r(d_Ksq_r, K_sq_r.data(), d_s3.ptr);
          if (!ok || !robust_from_res_gpu(h, aux, ext, Dl, Osql, Dr, Ksqr, 0.0, d_G_raw.ptr) ||
              !dress_fwd_gpu(h, same_spin, ext, d_S.ptr, d_G_raw.ptr, d_tmp2.ptr, d_tmp1.ptr))
            return 4;
          if (rm_gemm(h, false, false, ext, ext, ext, 1.0, d_FKl.ptr, ext, d_chi.ptr, ext, 0.0,
                      d_G_raw.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm(h, false, true, ext, ext, ext, 1.0, d_chi.ptr, ext, d_p_FKr, ext, 1.0,
                      d_G_raw.ptr, ext) != CUBLAS_STATUS_SUCCESS)
            return 4;
          k_dot_accum<<<g_red, 256>>>(ee, w_q, d_tmp2.ptr, d_G_raw.ptr, d_scalar_acc.ptr + 1);

          // B terms 3 + 5: -dot(f, action(FK_l, FK_r, f))
          if (rm_gemm(h, false, false, ext, ext, ext, 1.0, d_FKl.ptr, ext, d_f.ptr, ext, 0.0,
                      d_G_raw.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm(h, false, true, ext, ext, ext, 1.0, d_f.ptr, ext, d_p_FKr, ext, 1.0,
                      d_G_raw.ptr, ext) != CUBLAS_STATUS_SUCCESS)
            return 4;
          k_dot_accum<<<g_red, 256>>>(ee, -w_q, d_f.ptr, d_G_raw.ptr, d_scalar_acc.ptr + 1);

          // B term 4: +dot(qf, action(F_l, F_r, qf))
          if (rm_gemm(h, false, false, ext, ext, ext, 1.0, d_fl.ptr, ext, d_qf.ptr, ext, 0.0,
                      d_G_raw.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
              rm_gemm(h, false, true, ext, ext, ext, 1.0, d_qf.ptr, ext, d_p_fr, ext, 1.0,
                      d_G_raw.ptr, ext) != CUBLAS_STATUS_SUCCESS)
            return 4;
          k_dot_accum<<<g_red, 256>>>(ee, w_q, d_qf.ptr, d_G_raw.ptr, d_scalar_acc.ptr + 1);

          // X term 1: +dot(chi, ffx)
          // Part 1: (XDl, XOsql, Dr, Ksqr) — Dr is still in s2, Ksqr is still in s3!
          {
            const double* XDl = get_l(d_xf_l, XD_l.data(), d_s0.ptr);
            const double* XOsql = get_l(d_xsq_l, XO_sq_l.data(), d_s1.ptr);
            if (!ok || !robust_from_res_gpu(h, aux, ext, XDl, XOsql, Dr, Ksqr, 0.0, d_G_raw.ptr))
              return 4;
          }
          // Part 2: (Dl, Osql, XDr, XKsqr)
          {
            const double* Dl_p = get_l(d_fit_l, fit_l, d_s0.ptr);
            const double* Osql_p = get_l(d_sq_l, raw_sq_l, d_s1.ptr);
            const double* XDr = get_r(d_xf_r, h_XD_r, d_s2.ptr);
            const double* XKsqr = get_r(d_XKsq_r, XK_sq_r.data(), d_s3.ptr);
            if (!ok ||
                !robust_from_res_gpu(h, aux, ext, Dl_p, Osql_p, XDr, XKsqr, 1.0, d_G_raw.ptr) ||
                !dress_fwd_gpu(h, same_spin, ext, d_S.ptr, d_G_raw.ptr, d_tmp2.ptr, d_tmp1.ptr))
              return 4;
            k_dot_accum<<<g_red, 256>>>(ee, w_q, d_chi.ptr, d_tmp2.ptr, d_scalar_acc.ptr + 2);

            // X term 2: -dot(f - qf, fx)
            // Part 2 first: (Dl_p in s0, Ogl in s1, XDr in s2, XKgr in s3)
            const double* Ogl_p = get_l(d_g_l, raw_g_l, d_s1.ptr);
            const double* XKgr = get_r(d_XKg_r, XK_g_r.data(), d_s3.ptr);
            if (!ok || !robust_from_res_gpu(h, aux, ext, Dl_p, Ogl_p, XDr, XKgr, 0.0, d_G_raw.ptr))
              return 4;
          }
          // Part 1: (XDl, XOgl, Dr, Kgr)
          {
            const double* XDl = get_l(d_xf_l, XD_l.data(), d_s0.ptr);
            const double* XOgl = get_l(d_xg_l, XO_g_l.data(), d_s1.ptr);
            const double* Dr_p = get_r(d_fit_r, p_fit_r, d_s2.ptr);
            const double* Kgr_p = get_r(d_Kg_r, K_g_r.data(), d_s3.ptr);
            if (!ok ||
                !robust_from_res_gpu(h, aux, ext, XDl, XOgl, Dr_p, Kgr_p, 1.0, d_G_raw.ptr) ||
                !dress_fwd_gpu(h, same_spin, ext, d_S.ptr, d_G_raw.ptr, d_tmp2.ptr, d_tmp1.ptr))
              return 4;
          }
          k_strong_mask<<<g_ee, 256>>>(ext, nl, nr, obs, d_f.ptr, d_qf.ptr, true);
          k_dot_accum<<<g_red, 256>>>(ee, -w_q, d_qf.ptr, d_tmp2.ptr, d_scalar_acc.ptr + 2);
          if (cudaGetLastError() != cudaSuccess) return 4;
        }

        return kOk;
      };

      if (!tile_mode) {
        for (int k = 0; k < n_pairs; ++k) {
          const int rc = eval_pair(k, 0, 0);
          if (rc != 0) return rc;
        }
      } else {
        std::vector<int> tile_pairs;
        for (int i0 = 0; i0 < nact_l; i0 += ti) {
          const int cur_i = std::min(ti, nact_l - i0);
          bool any_i = false;
          for (int k = 0; k < n_pairs; ++k) {
            const int ii = pairs_i[k] - fl;
            if (ii >= i0 && ii < i0 + cur_i) {
              any_i = true;
              break;
            }
          }
          if (!any_i) continue;

          const Size l_cur = Size(cur_i) * ae;
          const Size l_off = Size(i0) * ae;
          if (!h2d(d_fit_l.ptr, fit_l + l_off, l_cur) || !h2d(d_g_l.ptr, raw_g_l + l_off, l_cur) ||
              !h2d(d_gc_l.ptr, raw_gc_l + l_off, l_cur))
            return 4;
          if (compute_quad) {
            if (!h2d(d_sq_l.ptr, raw_sq_l + l_off, l_cur) ||
                !h2d(d_dc_l.ptr, raw_dc_l + l_off, l_cur) ||
                !h2d(d_xf_l.ptr, XD_l.data() + l_off, l_cur) ||
                !h2d(d_xg_l.ptr, XO_g_l.data() + l_off, l_cur) ||
                !h2d(d_xsq_l.ptr, XO_sq_l.data() + l_off, l_cur))
              return 4;
          }

          for (int j0 = 0; j0 < nact_r; j0 += tj) {
            const int cur_j = std::min(tj, nact_r - j0);
            tile_pairs.clear();
            for (int k = 0; k < n_pairs; ++k) {
              const int ii = pairs_i[k] - fl;
              const int jj = pairs_j[k] - fr;
              if (ii >= i0 && ii < i0 + cur_i && jj >= j0 && jj < j0 + cur_j) {
                tile_pairs.push_back(k);
              }
            }
            if (tile_pairs.empty()) continue;

            const Size r_cur = Size(cur_j) * ae;
            const Size r_off = Size(j0) * ae;
            if (!h2d(d_fit_r.ptr, p_fit_r + r_off, r_cur) ||
                !h2d(d_rc_r.ptr, raw_coul_r + r_off, r_cur) ||
                !h2d(d_Kg_r.ptr, K_g_r.data() + r_off, r_cur) ||
                !h2d(d_Kgc_r.ptr, K_gc_r.data() + r_off, r_cur))
              return 4;
            if (compute_quad) {
              if (!h2d(d_Ksq_r.ptr, K_sq_r.data() + r_off, r_cur) ||
                  !h2d(d_Kdc_r.ptr, K_dc_r.data() + r_off, r_cur) ||
                  !h2d(d_xf_r.ptr, h_XD_r + r_off, r_cur) ||
                  !h2d(d_XKg_r.ptr, XK_g_r.data() + r_off, r_cur) ||
                  !h2d(d_XKsq_r.ptr, XK_sq_r.data() + r_off, r_cur))
                return 4;
            }

            for (int k : tile_pairs) {
              const int rc = eval_pair(k, (pairs_i[k] - fl) - i0, (pairs_j[k] - fr) - j0);
              if (rc != 0) return rc;
            }
          }
        }
      }

      if (cudaMemcpy(h_energies, d_scalar_acc.ptr, 3 * sizeof(double), cudaMemcpyDeviceToHost) !=
          cudaSuccess)
        return 5;
    }

    // Stage 2: Batched OBS Coulomb projection w_obs[k] -= sum_a D_a @ f_obs[k] @ R_a^T
    // Stage 1 GPU allocations are freed above so Stage 2 uses the full workspace budget.
    if (compute_wc) {
      const Size mat_bytes = oo * sizeof(double);
      const Size max_mats = budget / std::max<Size>(Size(1), mat_bytes);
      if (max_mats < Size(5)) return kErrOom;

      const Size max_b_by_int =
          static_cast<Size>(std::numeric_limits<int>::max()) / static_cast<Size>(obs);
      if (max_b_by_int < Size(1)) return kErrInvalidArg;
      const Size b_cap = std::min<Size>(Size(n_pairs), max_b_by_int);

      int B = 1, P = 1;
      if (Size(2) * Size(aux) + Size(3) * b_cap <= max_mats) {
        P = aux;
        B = static_cast<int>(b_cap);
      } else {
        const Size b_sz = std::min<Size>(b_cap, std::max<Size>(Size(1), (max_mats * 2 / 5) / 3));
        B = static_cast<int>(b_sz);
        const Size p_sz =
            std::min<Size>(Size(aux), std::max<Size>(Size(1), (max_mats - Size(3) * Size(B)) / 2));
        P = static_cast<int>(p_sz);
        if (P == aux) {
          const Size b2 =
              std::min<Size>(b_cap, std::max<Size>(Size(1), (max_mats - Size(2) * Size(P)) / 3));
          B = static_cast<int>(b2);
        } else if (Size(B) == b_cap) {
          const Size p2 = std::min<Size>(
              Size(aux), std::max<Size>(Size(1), (max_mats - Size(3) * Size(B)) / 2));
          P = static_cast<int>(p2);
        }
      }

      DeviceBuf d_D, d_R, d_fb, d_wb, d_T;
      if (d_D.alloc(Size(P) * oo) != cudaSuccess || d_R.alloc(Size(P) * oo) != cudaSuccess ||
          d_fb.alloc(Size(B) * oo) != cudaSuccess || d_wb.alloc(Size(B) * oo) != cudaSuccess ||
          d_T.alloc(Size(B) * oo) != cudaSuccess)
        return kErrOom;

      auto run_proj_tile = [&](int cur_b, int cur_p) -> bool {
        for (int a = 0; a < cur_p; ++a) {
          const double* Da = d_D.ptr + Size(a) * oo;
          const double* Ra = d_R.ptr + Size(a) * oo;
          if (rm_gemm(h, false, true, cur_b * obs, obs, obs, 1.0, d_fb.ptr, obs, Ra, obs, 0.0,
                      d_T.ptr, obs) != CUBLAS_STATUS_SUCCESS)
            return false;
          if (rm_gemm_batched(h, false, false, obs, obs, obs, -1.0, Da, obs, 0, d_T.ptr, obs,
                              static_cast<long long>(oo), 1.0, d_wb.ptr, obs,
                              static_cast<long long>(oo), cur_b) != CUBLAS_STATUS_SUCCESS)
            return false;
        }
        return true;
      };

      const Size n_b_tiles = Size((n_pairs + B - 1) / B);
      const Size n_p_tiles = Size((aux + P - 1) / P);
      const bool outer_pair =
          (n_b_tiles == 1) ||
          (n_p_tiles > 1 && n_b_tiles * 2 * Size(aux) <= n_p_tiles * 3 * Size(n_pairs));

      if (outer_pair) {
        for (int b0 = 0; b0 < n_pairs; b0 += B) {
          const int cur_b = std::min(B, n_pairs - b0);
          const Size b_bytes = Size(cur_b) * oo * sizeof(double);
          if (cudaMemcpy(d_fb.ptr, f_obs_all.data() + Size(b0) * oo, b_bytes,
                         cudaMemcpyHostToDevice) != cudaSuccess ||
              cudaMemcpy(d_wb.ptr, w_obs_out + Size(b0) * oo, b_bytes, cudaMemcpyHostToDevice) !=
                  cudaSuccess)
            return 7;
          for (int p0 = 0; p0 < aux; p0 += P) {
            const int cur_p = std::min(P, aux - p0);
            const Size p_bytes = Size(cur_p) * oo * sizeof(double);
            if (cudaMemcpy(d_D.ptr, fit_obs_l + Size(p0) * oo, p_bytes, cudaMemcpyHostToDevice) !=
                    cudaSuccess ||
                cudaMemcpy(d_R.ptr, raw_obs_r + Size(p0) * oo, p_bytes, cudaMemcpyHostToDevice) !=
                    cudaSuccess)
              return 7;
            if (!run_proj_tile(cur_b, cur_p)) return 8;
          }
          if (cudaMemcpy(w_obs_out + Size(b0) * oo, d_wb.ptr, b_bytes, cudaMemcpyDeviceToHost) !=
              cudaSuccess)
            return 9;
        }
      } else {
        for (int p0 = 0; p0 < aux; p0 += P) {
          const int cur_p = std::min(P, aux - p0);
          const Size p_bytes = Size(cur_p) * oo * sizeof(double);
          if (cudaMemcpy(d_D.ptr, fit_obs_l + Size(p0) * oo, p_bytes, cudaMemcpyHostToDevice) !=
                  cudaSuccess ||
              cudaMemcpy(d_R.ptr, raw_obs_r + Size(p0) * oo, p_bytes, cudaMemcpyHostToDevice) !=
                  cudaSuccess)
            return 7;
          for (int b0 = 0; b0 < n_pairs; b0 += B) {
            const int cur_b = std::min(B, n_pairs - b0);
            const Size b_bytes = Size(cur_b) * oo * sizeof(double);
            if (cudaMemcpy(d_fb.ptr, f_obs_all.data() + Size(b0) * oo, b_bytes,
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(d_wb.ptr, w_obs_out + Size(b0) * oo, b_bytes, cudaMemcpyHostToDevice) !=
                    cudaSuccess)
              return 7;
            if (!run_proj_tile(cur_b, cur_p)) return 8;
            if (cudaMemcpy(w_obs_out + Size(b0) * oo, d_wb.ptr, b_bytes, cudaMemcpyDeviceToHost) !=
                cudaSuccess)
              return 9;
          }
        }
      }
    }

    energies_out[0] = h_energies[0];
    energies_out[1] = h_energies[1];
    energies_out[2] = h_energies[2];
#endif
    if (!check_finite(energies_out, 3) ||
        (compute_wc && (!check_finite(w_obs_out, Size(n_pairs) * oo) ||
                        !check_finite(c_vv_out, Size(n_pairs) * vv)))) {
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
