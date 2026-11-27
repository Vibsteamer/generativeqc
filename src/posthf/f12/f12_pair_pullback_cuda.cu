// GPU-resident cuBLAS (and matching host reference) MO-space pair pullbacks for fixed-cusp F12
// W_vo/W_vv, C_vv, and scalar V + B - X nuclear force contractions.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#ifndef GQC_SINK_HOST_REFERENCE
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace {

using Index = std::int32_t;
using Size = std::uint64_t;

constexpr int kErrInvalidArg = -1;
constexpr int kErrNonFiniteOutput = -2;
constexpr int kErrOom = -3;
constexpr int kErrInternal = -4;

[[maybe_unused]] constexpr Size kDefaultDeviceBudgetBytes = Size(4096) << 20;

[[maybe_unused]] inline Size sat_mul(Size a, Size b) noexcept {
  if (a == 0 || b == 0) return 0;
  if (a > std::numeric_limits<Size>::max() / b) return std::numeric_limits<Size>::max();
  return a * b;
}

[[maybe_unused]] inline Size sat_add(Size a, Size b) noexcept {
  if (a > std::numeric_limits<Size>::max() - b) return std::numeric_limits<Size>::max();
  return a + b;
}

[[maybe_unused]] inline Size effective_device_budget(Size budget_bytes) noexcept {
  return budget_bytes > 0 ? budget_bytes : kDefaultDeviceBudgetBytes;
}

inline bool fits_int(Size x) noexcept {
  return x <= static_cast<Size>(std::numeric_limits<int>::max());
}

inline bool all_finite(const double* ptr, Size count) noexcept {
  if (count == 0) return true;
  if (!ptr) return false;
  for (Size i = 0; i < count; ++i) {
    if (!std::isfinite(ptr[i])) return false;
  }
  return true;
}

inline void add_inplace(double* dst, const double* src, Size count) noexcept {
  for (Size i = 0; i < count; ++i) dst[i] += src[i];
}

#ifdef GQC_SINK_HOST_REFERENCE

void rm_gemm_host(bool trans_a, bool trans_b, int M, int N, int K, double alpha, const double* A,
                  int lda, const double* B, int ldb, double beta, double* C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        const double a_ik = trans_a ? A[Size(k) * lda + i] : A[Size(i) * lda + k];
        const double b_kj = trans_b ? B[Size(j) * ldb + k] : B[Size(k) * ldb + j];
        acc += a_ik * b_kj;
      }
      if (beta == 0.0)
        C[Size(i) * ldc + j] = alpha * acc;
      else
        C[Size(i) * ldc + j] = beta * C[Size(i) * ldc + j] + alpha * acc;
    }
  }
}

void axpy_host(Size n, double alpha, const double* x, double* y) {
  for (Size i = 0; i < n; ++i) y[i] += alpha * x[i];
}

void scale_add_host(Size n, double a, const double* x, double b, const double* y, double* out) {
  for (Size i = 0; i < n; ++i) out[i] = a * x[i] + b * y[i];
}

double dot_host(Size n, const double* x, const double* y) {
  double s = 0.0;
  for (Size i = 0; i < n; ++i) s += x[i] * y[i];
  return s;
}

void antisym_quarter_host(int ext, const double* in, double* out) {
  for (int p = 0; p < ext; ++p) {
    for (int q = 0; q < ext; ++q) {
      out[Size(p) * ext + q] = 0.25 * (in[Size(p) * ext + q] - in[Size(q) * ext + p]);
    }
  }
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

void add_strong_masked_host(int ext, int nl, int nr, int obs, double scale, const double* in,
                            double* dst, bool invert) {
  for (int p = 0; p < ext; ++p) {
    for (int q = 0; q < ext; ++q) {
      const bool in_q = (p >= nl && q >= nr) && !(p < obs && q < obs);
      const bool keep = invert ? !in_q : in_q;
      if (keep) dst[Size(p) * ext + q] += scale * in[Size(p) * ext + q];
    }
  }
}

void v_mask_acc_host(int ext, int nl, int nr, int obs, double scale, const double* C,
                     const double* f, double* bar_C, double* bar_f, double& dot_out) {
  double acc = 0.0;
  for (int p = 0; p < ext; ++p) {
    for (int q = 0; q < ext; ++q) {
      const bool in_m = (p < obs && q < obs) || (p < nl && q >= obs) || (p >= obs && q < nr);
      const Size idx = Size(p) * ext + q;
      if (in_m) {
        acc += C[idx] * f[idx];
        bar_C[idx] = scale * f[idx];
        bar_f[idx] += scale * C[idx];
      } else {
        bar_C[idx] = 0.0;
      }
    }
  }
  dot_out = acc;
}

void copy_subblock_host(int rows, int cols, const double* src, int src_ld, double* dst, int dst_ld,
                        bool add_mode) {
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (add_mode)
        dst[Size(r) * dst_ld + c] += src[Size(r) * src_ld + c];
      else
        dst[Size(r) * dst_ld + c] = src[Size(r) * src_ld + c];
    }
  }
}

void robust_fwd_host(int aux, int ext, const double* Dl, const double* Ol, const double* Dr,
                     const double* Or, const double* M, double* Kr, double* G_raw) {
  std::copy_n(Or, Size(aux) * ext, Kr);
  rm_gemm_host(false, false, aux, ext, aux, -1.0, M, aux, Dr, ext, 1.0, Kr, ext);
  rm_gemm_host(true, false, ext, ext, aux, 1.0, Dl, ext, Kr, ext, 0.0, G_raw, ext);
  rm_gemm_host(true, false, ext, ext, aux, 1.0, Ol, ext, Dr, ext, 1.0, G_raw, ext);
}

void robust_adj_host(int aux, int ext, const double* Dl, const double* Ol, const double* Dr,
                     const double* M, const double* Kr, const double* bar_G, double* bar_Dl,
                     double* bar_Ol, double* bar_Dr, double* bar_Or, double* bar_M,
                     double* bar_Kr) {
  rm_gemm_host(false, true, aux, ext, ext, 1.0, Kr, ext, bar_G, ext, 1.0, bar_Dl, ext);
  rm_gemm_host(false, false, aux, ext, ext, 1.0, Dl, ext, bar_G, ext, 0.0, bar_Kr, ext);
  rm_gemm_host(false, true, aux, ext, ext, 1.0, Dr, ext, bar_G, ext, 1.0, bar_Ol, ext);
  rm_gemm_host(false, false, aux, ext, ext, 1.0, Ol, ext, bar_G, ext, 1.0, bar_Dr, ext);
  rm_gemm_host(true, false, aux, ext, aux, -1.0, M, aux, bar_Kr, ext, 1.0, bar_Dr, ext);
  axpy_host(Size(aux) * ext, 1.0, bar_Kr, bar_Or);
  rm_gemm_host(false, true, aux, aux, ext, -1.0, bar_Kr, ext, Dr, ext, 1.0, bar_M, aux);
}

void dress_fwd_host(bool same_spin, int ext, const double* S, const double* in, double* out,
                    double* tmp) {
  const Size n2 = Size(ext) * ext;
  if (same_spin) {
    antisym_quarter_host(ext, in, out);
    return;
  }
  std::copy_n(in, n2, out);
  rm_gemm_host(false, true, ext, ext, ext, 1.0, S, ext, in, ext, 0.0, tmp, ext);
  rm_gemm_host(false, false, ext, ext, ext, 0.125, tmp, ext, S, ext, 0.375, out, ext);
}

void dress_adj_host(bool same_spin, int ext, const double* S, const double* in_raw,
                    const double* bar_out, double* bar_in, double* bar_S, double* tmp1,
                    double* tmp2) {
  const Size n2 = Size(ext) * ext;
  if (same_spin) {
    antisym_quarter_host(ext, bar_out, bar_in);
    return;
  }
  rm_gemm_host(false, true, ext, ext, ext, 1.0, bar_out, ext, S, ext, 0.0, tmp1, ext);
  rm_gemm_host(false, false, ext, ext, ext, 0.125, tmp1, ext, in_raw, ext, 1.0, bar_S, ext);
  rm_gemm_host(false, true, ext, ext, ext, 1.0, in_raw, ext, S, ext, 0.0, tmp2, ext);
  rm_gemm_host(false, false, ext, ext, ext, 0.125, tmp2, ext, bar_out, ext, 1.0, bar_S, ext);
  std::copy_n(bar_out, n2, bar_in);
  rm_gemm_host(true, false, ext, ext, ext, 0.125, tmp1, ext, S, ext, 0.375, bar_in, ext);
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
  cublasStatus_t init() { return cublasCreate(&handle); }
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

__global__ void k_antisym_quarter(int ext, const double* in, double* out) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(ext) * ext;
  if (idx >= total) return;
  const int p = static_cast<int>(idx / ext);
  const int q = static_cast<int>(idx % ext);
  out[idx] = 0.25 * (in[idx] - in[Size(q) * ext + p]);
}

__global__ void k_axpy(Size n, double alpha, const double* x, double* y) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) y[idx] += alpha * x[idx];
}

__global__ void k_scale_add(Size n, double a, const double* x, double b, const double* y,
                            double* out) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = a * x[idx] + b * y[idx];
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

__global__ void k_add_strong_masked(int ext, int nl, int nr, int obs, double scale,
                                    const double* in, double* dst, bool invert) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(ext) * ext;
  if (idx >= total) return;
  const int p = static_cast<int>(idx / ext);
  const int q = static_cast<int>(idx % ext);
  const bool in_q = (p >= nl && q >= nr) && !(p < obs && q < obs);
  const bool keep = invert ? !in_q : in_q;
  if (keep) dst[idx] += scale * in[idx];
}

__global__ void k_v_mask_acc(int ext, int nl, int nr, int obs, double scale, const double* C,
                             const double* f, double* bar_C, double* bar_f, double* dot_accum) {
  __shared__ double sdata[256];
  const Size total = Size(ext) * ext;
  double local = 0.0;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(gridDim.x) * blockDim.x) {
    const int p = static_cast<int>(idx / ext);
    const int q = static_cast<int>(idx % ext);
    const bool in_m = (p < obs && q < obs) || (p < nl && q >= obs) || (p >= obs && q < nr);
    if (in_m) {
      local += C[idx] * f[idx];
      bar_C[idx] = scale * f[idx];
      bar_f[idx] += scale * C[idx];
    } else {
      bar_C[idx] = 0.0;
    }
  }
  sdata[threadIdx.x] = local;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(dot_accum, sdata[0]);
}

__global__ void k_copy_subblock(int rows, int cols, const double* src, int src_ld, double* dst,
                                int dst_ld, bool add_mode) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(rows) * cols;
  if (idx >= total) return;
  const int r = static_cast<int>(idx / cols);
  const int c = static_cast<int>(idx % cols);
  if (add_mode)
    dst[Size(r) * dst_ld + c] += src[Size(r) * src_ld + c];
  else
    dst[Size(r) * dst_ld + c] = src[Size(r) * src_ld + c];
}

__global__ void k_set_point(double* mat, int ld, int i, int j, double val, bool add_mode) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    if (add_mode)
      mat[Size(i) * ld + j] += val;
    else
      mat[Size(i) * ld + j] = val;
  }
}

__global__ void k_read_point_accum(const double* mat, int ld, int i, int j, double scale,
                                   double* accum) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *accum += scale * mat[Size(i) * ld + j];
  }
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

__global__ void k_bar_chi_to_bar_S(int ext, int i, int j, const double* bar_chi, const double* S,
                                   double* bar_S) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < ext) {
    double acc_pj = 0.0;
    for (int q = 0; q < ext; ++q) acc_pj += bar_chi[Size(k) * ext + q] * S[Size(i) * ext + q];
    atomicAdd(&bar_S[Size(k) * ext + j], 0.125 * acc_pj);

    double acc_iq = 0.0;
    for (int p = 0; p < ext; ++p) acc_iq += bar_chi[Size(p) * ext + k] * S[Size(p) * ext + j];
    atomicAdd(&bar_S[Size(i) * ext + k], 0.125 * acc_iq);
  }
}

inline unsigned grid_for(Size n, int blk = 256) {
  return static_cast<unsigned>((n + blk - 1) / blk);
}

struct PairScratch {
  DeviceBuf Kr, bar_Kr, G_raw, tmp1, tmp2;
  cudaError_t init(int aux, int ext) {
    cudaError_t s = Kr.alloc(Size(aux) * ext);
    if (s == cudaSuccess) s = bar_Kr.alloc(Size(aux) * ext);
    if (s == cudaSuccess) s = G_raw.alloc(Size(ext) * ext);
    if (s == cudaSuccess) s = tmp1.alloc(Size(ext) * ext);
    if (s == cudaSuccess) s = tmp2.alloc(Size(ext) * ext);
    return s;
  }
};

bool robust_fwd_gpu(cublasHandle_t h, int aux, int ext, const double* Dl, const double* Ol,
                    const double* Dr, const double* Or, const double* M, double* Kr,
                    double* G_raw) {
  if (cudaMemcpy(Kr, Or, Size(aux) * ext * sizeof(double), cudaMemcpyDeviceToDevice) != cudaSuccess)
    return false;
  if (rm_gemm(h, false, false, aux, ext, aux, -1.0, M, aux, Dr, ext, 1.0, Kr, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  if (rm_gemm(h, true, false, ext, ext, aux, 1.0, Dl, ext, Kr, ext, 0.0, G_raw, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  if (rm_gemm(h, true, false, ext, ext, aux, 1.0, Ol, ext, Dr, ext, 1.0, G_raw, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  return true;
}

bool robust_adj_gpu(cublasHandle_t h, int aux, int ext, const double* Dl, const double* Ol,
                    const double* Dr, const double* M, const double* Kr, const double* bar_G,
                    double* bar_Dl, double* bar_Ol, double* bar_Dr, double* bar_Or, double* bar_M,
                    double* bar_Kr) {
  if (rm_gemm(h, false, true, aux, ext, ext, 1.0, Kr, ext, bar_G, ext, 1.0, bar_Dl, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  if (rm_gemm(h, false, false, aux, ext, ext, 1.0, Dl, ext, bar_G, ext, 0.0, bar_Kr, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  if (rm_gemm(h, false, true, aux, ext, ext, 1.0, Dr, ext, bar_G, ext, 1.0, bar_Ol, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  if (rm_gemm(h, false, false, aux, ext, ext, 1.0, Ol, ext, bar_G, ext, 1.0, bar_Dr, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  if (rm_gemm(h, true, false, aux, ext, aux, -1.0, M, aux, bar_Kr, ext, 1.0, bar_Dr, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  k_axpy<<<grid_for(Size(aux) * ext), 256>>>(Size(aux) * ext, 1.0, bar_Kr, bar_Or);
  if (rm_gemm(h, false, true, aux, aux, ext, -1.0, bar_Kr, ext, Dr, ext, 1.0, bar_M, aux) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  return true;
}

bool dress_fwd_gpu(cublasHandle_t h, bool same_spin, int ext, const double* S, const double* in,
                   double* out, double* tmp) {
  const Size n2 = Size(ext) * ext;
  if (same_spin) {
    k_antisym_quarter<<<grid_for(n2), 256>>>(ext, in, out);
    return true;
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

bool dress_adj_gpu(cublasHandle_t h, bool same_spin, int ext, const double* S, const double* in_raw,
                   const double* bar_out, double* bar_in, double* bar_S, double* tmp1,
                   double* tmp2) {
  const Size n2 = Size(ext) * ext;
  if (same_spin) {
    k_antisym_quarter<<<grid_for(n2), 256>>>(ext, bar_out, bar_in);
    return true;
  }
  if (rm_gemm(h, false, true, ext, ext, ext, 1.0, bar_out, ext, S, ext, 0.0, tmp1, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  if (rm_gemm(h, false, false, ext, ext, ext, 0.125, tmp1, ext, in_raw, ext, 1.0, bar_S, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  if (rm_gemm(h, false, true, ext, ext, ext, 1.0, in_raw, ext, S, ext, 0.0, tmp2, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  if (rm_gemm(h, false, false, ext, ext, ext, 0.125, tmp2, ext, bar_out, ext, 1.0, bar_S, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  if (cudaMemcpy(bar_in, bar_out, n2 * sizeof(double), cudaMemcpyDeviceToDevice) != cudaSuccess)
    return false;
  if (rm_gemm(h, true, false, ext, ext, ext, 0.125, tmp1, ext, S, ext, 0.375, bar_in, ext) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  return true;
}

bool d2h_staged(const double* dev_src, Size n, std::vector<double>& out) {
  out.resize(n);
  if (n == 0) return true;
  if (cudaMemcpy(out.data(), dev_src, n * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
    return false;
  return true;
}

bool launch_ok() noexcept {
  return cudaPeekAtLastError() == cudaSuccess;
}

#endif

}  // namespace

#ifdef GQC_SINK_HOST_REFERENCE
#define W_PULLBACK_EXPORT gqc_f12_cc_w_mo_pullback_host_reference
#define C_PULLBACK_EXPORT gqc_f12_cc_c_mo_pullback_host_reference
#define SCALAR_PULLBACK_EXPORT gqc_f12_cc_scalar_mo_pullback_host_reference
#else
#define W_PULLBACK_EXPORT gqc_f12_cc_w_mo_pullback_cuda
#define C_PULLBACK_EXPORT gqc_f12_cc_c_mo_pullback_cuda
#define SCALAR_PULLBACK_EXPORT gqc_f12_cc_scalar_mo_pullback_cuda
#endif

// =========================================================================
// 1. W_vo / W_vv MO-space pullback (all pairs in one spin channel)
//    All [occ, aux, ext] arrays are contiguous per orbital slice.
//    fit_obs_l, raw_obs_r, bar_fit_obs_l, bar_raw_obs_r have shape [aux, obs, obs].
//    Outputs are ACCUMULATED (+=) into caller-supplied host arrays!
// =========================================================================
extern "C" int W_PULLBACK_EXPORT(
    int same_spin_int, int nl, int nr, int obs, int ext, int aux, int n_pairs,
    std::uint64_t budget_bytes, const Index* pairs_i, const Index* pairs_j,
    const double* fit_obs_l, const double* raw_obs_r, const double* fit_l, const double* fit_r,
    const double* raw_g_l, const double* raw_g_r, const double* raw_gc_l, const double* raw_gc_r,
    const double* metric_g, const double* metric_gc, const double* cross_overlap,
    const double* bar_w_all, double* bar_fit_obs_l, double* bar_raw_obs_r, double* bar_fit_l,
    double* bar_fit_r, double* bar_raw_g_l, double* bar_raw_g_r, double* bar_raw_gc_l,
    double* bar_raw_gc_r, double* bar_metric_g, double* bar_metric_gc, double* bar_cross_overlap) {
  try {
  if (nl <= 0 || nr <= 0 || obs <= 0 || obs < nl || obs < nr || ext <= obs || aux <= 0 ||
      n_pairs <= 0 || !pairs_i || !pairs_j || !fit_obs_l || !raw_obs_r || !fit_l || !fit_r ||
      !raw_g_l || !raw_g_r || !raw_gc_l || !raw_gc_r || !metric_g || !metric_gc || !bar_w_all ||
      !bar_fit_obs_l || !bar_raw_obs_r || !bar_fit_l || !bar_fit_r || !bar_raw_g_l ||
      !bar_raw_g_r || !bar_raw_gc_l || !bar_raw_gc_r || !bar_metric_g || !bar_metric_gc)
    return kErrInvalidArg;
  const bool same_spin = (same_spin_int != 0);
  if (same_spin && nl != nr) return kErrInvalidArg;
  if (!same_spin && (!cross_overlap || !bar_cross_overlap)) return kErrInvalidArg;
  for (int k = 0; k < n_pairs; ++k) {
    if (pairs_i[k] < 0 || pairs_i[k] >= nl || pairs_j[k] < 0 || pairs_j[k] >= nr)
      return kErrInvalidArg;
  }

  const Size ae = sat_mul(Size(aux), Size(ext));
  const Size ee = sat_mul(Size(ext), Size(ext));
  const Size oo = sat_mul(Size(obs), Size(obs));
  const Size aa = sat_mul(Size(aux), Size(aux));
  const Size aoo = sat_mul(Size(aux), oo);
  const Size nlae = sat_mul(Size(nl), ae);
  const Size nrae = sat_mul(Size(nr), ae);
  const Size npoo = sat_mul(Size(n_pairs), oo);
  if (!fits_int(oo) || !fits_int(ee) || !fits_int(aa) || !fits_int(ae) ||
      aoo == std::numeric_limits<Size>::max() || nlae == std::numeric_limits<Size>::max() ||
      nrae == std::numeric_limits<Size>::max() || npoo == std::numeric_limits<Size>::max())
    return kErrInvalidArg;

  if (!all_finite(fit_obs_l, aoo) || !all_finite(raw_obs_r, aoo) || !all_finite(fit_l, nlae) ||
      !all_finite(raw_g_l, nlae) || !all_finite(raw_gc_l, nlae) || !all_finite(metric_g, aa) ||
      !all_finite(metric_gc, aa) || !all_finite(bar_w_all, npoo))
    return kErrInvalidArg;
  if (!same_spin) {
    if (!all_finite(fit_r, nrae) || !all_finite(raw_g_r, nrae) || !all_finite(raw_gc_r, nrae) ||
        !all_finite(cross_overlap, ee))
      return kErrInvalidArg;
  }

#ifdef GQC_SINK_HOST_REFERENCE
  (void)budget_bytes;
  std::vector<double> Kr(ae), bar_Kr(ae), G_raw(ee), G_dress(ee), tmp1(ee), tmp2(ee);
  std::vector<double> bar_G_dress(ee), bar_G_raw(ee);
  std::vector<double> f_obs(oo), bar_f_obs(oo), Right(oo), B_l(oo);

  std::vector<double> loc_bfo_l(aoo, 0.0), loc_bro_r(aoo, 0.0);
  std::vector<double> loc_bfl(nlae, 0.0), loc_bgl(nlae, 0.0), loc_bgcl(nlae, 0.0);
  std::vector<double> loc_bfr(same_spin ? 0 : nrae, 0.0);
  std::vector<double> loc_bgr(same_spin ? 0 : nrae, 0.0);
  std::vector<double> loc_bgcr(same_spin ? 0 : nrae, 0.0);
  std::vector<double> loc_bmg(aa, 0.0), loc_bmgc(aa, 0.0);
  std::vector<double> loc_bS(same_spin ? 0 : ee, 0.0);

  double* const p_bfr = same_spin ? loc_bfl.data() : loc_bfr.data();
  double* const p_bgr = same_spin ? loc_bgl.data() : loc_bgr.data();
  double* const p_bgcr = same_spin ? loc_bgcl.data() : loc_bgcr.data();
  double* const p_bS = same_spin ? nullptr : loc_bS.data();

  for (int k = 0; k < n_pairs; ++k) {
    const int i = pairs_i[k], j = pairs_j[k];
    const double* Dl = fit_l + Size(i) * ae;
    const double* Dr = (same_spin ? fit_l : fit_r) + Size(j) * ae;
    const double* Ogl = raw_g_l + Size(i) * ae;
    const double* Ogr = (same_spin ? raw_g_l : raw_g_r) + Size(j) * ae;
    const double* Ogcl = raw_gc_l + Size(i) * ae;
    const double* Ogcr = (same_spin ? raw_gc_l : raw_gc_r) + Size(j) * ae;
    double* bDl = loc_bfl.data() + Size(i) * ae;
    double* bDr = p_bfr + Size(j) * ae;
    double* bOgl = loc_bgl.data() + Size(i) * ae;
    double* bOgr = p_bgr + Size(j) * ae;
    double* bOgcl = loc_bgcl.data() + Size(i) * ae;
    double* bOgcr = p_bgcr + Size(j) * ae;

    robust_fwd_host(aux, ext, Dl, Ogl, Dr, Ogr, metric_g, Kr.data(), G_raw.data());
    dress_fwd_host(same_spin, ext, cross_overlap, G_raw.data(), G_dress.data(), tmp1.data());
    copy_subblock_host(obs, obs, G_dress.data(), ext, f_obs.data(), obs, false);

    const double* bw = bar_w_all + Size(k) * oo;
    std::fill(bar_f_obs.begin(), bar_f_obs.end(), 0.0);
    for (int a = 0; a < aux; ++a) {
      const double* Da = fit_obs_l + Size(a) * oo;
      const double* Ra = raw_obs_r + Size(a) * oo;
      double* bDa = loc_bfo_l.data() + Size(a) * oo;
      double* bRa = loc_bro_r.data() + Size(a) * oo;
      rm_gemm_host(false, true, obs, obs, obs, 1.0, f_obs.data(), obs, Ra, obs, 0.0, Right.data(),
                   obs);
      rm_gemm_host(false, true, obs, obs, obs, -1.0, bw, obs, Right.data(), obs, 1.0, bDa, obs);
      rm_gemm_host(true, false, obs, obs, obs, 1.0, Da, obs, bw, obs, 0.0, B_l.data(), obs);
      rm_gemm_host(true, false, obs, obs, obs, -1.0, B_l.data(), obs, f_obs.data(), obs, 1.0, bRa,
                   obs);
      rm_gemm_host(false, false, obs, obs, obs, -1.0, B_l.data(), obs, Ra, obs, 1.0,
                   bar_f_obs.data(), obs);
    }

    std::fill(bar_G_dress.begin(), bar_G_dress.end(), 0.0);
    copy_subblock_host(obs, obs, bar_f_obs.data(), obs, bar_G_dress.data(), ext, false);
    dress_adj_host(same_spin, ext, cross_overlap, G_raw.data(), bar_G_dress.data(),
                   bar_G_raw.data(), p_bS, tmp1.data(), tmp2.data());
    robust_adj_host(aux, ext, Dl, Ogl, Dr, metric_g, Kr.data(), bar_G_raw.data(), bDl, bOgl, bDr,
                    bOgr, loc_bmg.data(), bar_Kr.data());

    robust_fwd_host(aux, ext, Dl, Ogcl, Dr, Ogcr, metric_gc, Kr.data(), G_raw.data());
    std::fill(bar_G_dress.begin(), bar_G_dress.end(), 0.0);
    copy_subblock_host(obs, obs, bw, obs, bar_G_dress.data(), ext, false);
    dress_adj_host(same_spin, ext, cross_overlap, G_raw.data(), bar_G_dress.data(),
                   bar_G_raw.data(), p_bS, tmp1.data(), tmp2.data());
    robust_adj_host(aux, ext, Dl, Ogcl, Dr, metric_gc, Kr.data(), bar_G_raw.data(), bDl, bOgcl, bDr,
                    bOgcr, loc_bmgc.data(), bar_Kr.data());
  }

  if (!all_finite(loc_bfo_l.data(), aoo) || !all_finite(loc_bro_r.data(), aoo) ||
      !all_finite(loc_bfl.data(), nlae) || !all_finite(loc_bgl.data(), nlae) ||
      !all_finite(loc_bgcl.data(), nlae) || !all_finite(loc_bmg.data(), aa) ||
      !all_finite(loc_bmgc.data(), aa) ||
      (!same_spin && (!all_finite(loc_bfr.data(), nrae) || !all_finite(loc_bgr.data(), nrae) ||
                      !all_finite(loc_bgcr.data(), nrae) || !all_finite(loc_bS.data(), ee))))
    return kErrNonFiniteOutput;

  add_inplace(bar_fit_obs_l, loc_bfo_l.data(), aoo);
  add_inplace(bar_raw_obs_r, loc_bro_r.data(), aoo);
  add_inplace(bar_fit_l, loc_bfl.data(), nlae);
  add_inplace(bar_raw_g_l, loc_bgl.data(), nlae);
  add_inplace(bar_raw_gc_l, loc_bgcl.data(), nlae);
  add_inplace(bar_metric_g, loc_bmg.data(), aa);
  add_inplace(bar_metric_gc, loc_bmgc.data(), aa);
  if (!same_spin) {
    add_inplace(bar_fit_r, loc_bfr.data(), nrae);
    add_inplace(bar_raw_g_r, loc_bgr.data(), nrae);
    add_inplace(bar_raw_gc_r, loc_bgcr.data(), nrae);
    add_inplace(bar_cross_overlap, loc_bS.data(), ee);
  }
  return 0;
#else
  const Size cap_bytes = effective_device_budget(budget_bytes);
  Size fixed_doubles = sat_add(sat_mul(6, nlae), sat_mul(4, aa));
  fixed_doubles = sat_add(fixed_doubles, sat_mul(3, npoo));
  fixed_doubles = sat_add(fixed_doubles, sat_add(sat_mul(6, ee), sat_mul(2, ae)));
  if (!same_spin) {
    fixed_doubles = sat_add(fixed_doubles, sat_add(sat_mul(6, nrae), sat_mul(2, ee)));
  }
  const Size fixed_bytes = sat_mul(fixed_doubles, sizeof(double));
  const Size one_aux_bytes = sat_mul(sat_add(sat_mul(6, oo), 1), sizeof(double));
  if (fixed_bytes >= cap_bytes || one_aux_bytes > cap_bytes - fixed_bytes) return kErrOom;

  const Size remaining_bytes = cap_bytes - fixed_bytes;
  const Size slab_cap = budget_bytes > 0 ? remaining_bytes : std::min(remaining_bytes, Size(2048) << 20);
  const Size max_slab = slab_cap / std::max<Size>(1, one_aux_bytes);
  if (max_slab == 0) return kErrOom;
  const int aux_slab = static_cast<int>(std::min<Size>(static_cast<Size>(aux), max_slab));

  CublasGuard cg;
  if (cg.init() != CUBLAS_STATUS_SUCCESS) return 1;
  cublasHandle_t h = cg.handle;

  DeviceBuf d_fit_l, d_fit_r, d_raw_g_l, d_raw_g_r, d_raw_gc_l, d_raw_gc_r;
  DeviceBuf d_mg, d_mgc, d_S, d_bar_w_all, d_f_obs_all, d_bar_f_obs_all;
  DeviceBuf d_bfit_l, d_bfit_r, d_braw_g_l, d_braw_g_r, d_braw_gc_l, d_braw_gc_r;
  DeviceBuf d_bmg, d_bmgc, d_bS;
  DeviceBuf d_G_dress, d_bar_G_dress, d_bar_G_raw, d_ones;
  PairScratch sc;

  auto chk = [](cudaError_t e) { return e == cudaSuccess; };
  if (!chk(d_fit_l.alloc(nlae)) || !chk(d_raw_g_l.alloc(nlae)) ||
      !chk(d_raw_gc_l.alloc(nlae)) || !chk(d_bfit_l.alloc(nlae)) ||
      !chk(d_braw_g_l.alloc(nlae)) || !chk(d_braw_gc_l.alloc(nlae)) ||
      !chk(d_mg.alloc(aa)) || !chk(d_mgc.alloc(aa)) || !chk(d_bmg.alloc(aa)) ||
      !chk(d_bmgc.alloc(aa)) || !chk(d_bar_w_all.alloc(npoo)) ||
      !chk(d_f_obs_all.alloc(npoo)) ||
      !chk(d_bar_f_obs_all.alloc(npoo)) || !chk(d_G_dress.alloc(ee)) ||
      !chk(d_bar_G_dress.alloc(ee)) || !chk(d_bar_G_raw.alloc(ee)) || !chk(d_ones.alloc(aux_slab)) ||
      !chk(sc.init(aux, ext)))
    return kErrOom;
  if (!same_spin) {
    if (!chk(d_fit_r.alloc(nrae)) || !chk(d_raw_g_r.alloc(nrae)) ||
        !chk(d_raw_gc_r.alloc(nrae)) || !chk(d_bfit_r.alloc(nrae)) ||
        !chk(d_braw_g_r.alloc(nrae)) || !chk(d_braw_gc_r.alloc(nrae)) ||
        !chk(d_S.alloc(ee)) || !chk(d_bS.alloc(ee)))
      return kErrOom;
  }

  auto h2d = [](double* dst, const double* src, Size n) {
    return cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess;
  };
  auto zero = [](double* dst, Size n) {
    return cudaMemset(dst, 0, n * sizeof(double)) == cudaSuccess;
  };
  std::vector<double> ones(aux_slab, 1.0);
  if (!h2d(d_ones.ptr, ones.data(), aux_slab) || !h2d(d_fit_l.ptr, fit_l, nlae) ||
      !h2d(d_raw_g_l.ptr, raw_g_l, nlae) ||
      !h2d(d_raw_gc_l.ptr, raw_gc_l, nlae) || !h2d(d_mg.ptr, metric_g, aa) ||
      !h2d(d_mgc.ptr, metric_gc, aa) || !h2d(d_bar_w_all.ptr, bar_w_all, npoo) ||
      !zero(d_bfit_l.ptr, nlae) || !zero(d_braw_g_l.ptr, nlae) ||
      !zero(d_braw_gc_l.ptr, nlae) || !zero(d_bmg.ptr, aa) || !zero(d_bmgc.ptr, aa) ||
      !zero(d_bar_f_obs_all.ptr, npoo))
    return 3;
  if (!same_spin) {
    if (!h2d(d_fit_r.ptr, fit_r, nrae) || !h2d(d_raw_g_r.ptr, raw_g_r, nrae) ||
        !h2d(d_raw_gc_r.ptr, raw_gc_r, nrae) || !h2d(d_S.ptr, cross_overlap, ee) ||
        !zero(d_bfit_r.ptr, nrae) || !zero(d_braw_g_r.ptr, nrae) ||
        !zero(d_braw_gc_r.ptr, nrae) || !zero(d_bS.ptr, ee))
      return 3;
  }

  const double* p_fit_r = same_spin ? d_fit_l.ptr : d_fit_r.ptr;
  const double* p_raw_g_r = same_spin ? d_raw_g_l.ptr : d_raw_g_r.ptr;
  const double* p_raw_gc_r = same_spin ? d_raw_gc_l.ptr : d_raw_gc_r.ptr;
  double* p_bfit_r = same_spin ? d_bfit_l.ptr : d_bfit_r.ptr;
  double* p_braw_g_r = same_spin ? d_braw_g_l.ptr : d_braw_g_r.ptr;
  double* p_braw_gc_r = same_spin ? d_braw_gc_l.ptr : d_braw_gc_r.ptr;

  // Stage 1: Compute f_obs_all[k, obs, obs] for all pairs on GPU.
  for (int k = 0; k < n_pairs; ++k) {
    const int i = pairs_i[k], j = pairs_j[k];
    if (!robust_fwd_gpu(h, aux, ext, d_fit_l.ptr + Size(i) * ae, d_raw_g_l.ptr + Size(i) * ae,
                        p_fit_r + Size(j) * ae, p_raw_g_r + Size(j) * ae, d_mg.ptr, sc.Kr.ptr,
                        sc.G_raw.ptr))
      return 4;
    if (!dress_fwd_gpu(h, same_spin, ext, d_S.ptr, sc.G_raw.ptr, d_G_dress.ptr, sc.tmp1.ptr))
      return 4;
    k_copy_subblock<<<grid_for(oo), 256>>>(obs, obs, d_G_dress.ptr, ext,
                                           d_f_obs_all.ptr + Size(k) * oo, obs, false);
    if (!launch_ok()) return 4;
  }

  // Stage 2: Auxiliary-slabbed OBS Coulomb projection reverse across ALL pairs on GPU.
  // Stage host outputs locally so caller arrays stay untouched if any output is nonfinite.
  std::vector<double> stage_bfo_l(aoo, 0.0), stage_bro_r(aoo, 0.0);
  std::vector<double> h_tmp;
  {
    DeviceBuf d_D, d_R, d_bD, d_bR, d_Right, d_Bl;
    const Size slab_elems = Size(aux_slab) * oo;
    if (!chk(d_D.alloc(slab_elems)) || !chk(d_R.alloc(slab_elems)) ||
        !chk(d_bD.alloc(slab_elems)) || !chk(d_bR.alloc(slab_elems)) ||
        !chk(d_Right.alloc(slab_elems)) || !chk(d_Bl.alloc(slab_elems)))
      return kErrOom;
    for (int a0 = 0; a0 < aux; a0 += aux_slab) {
      const int cur = std::min(aux_slab, aux - a0);
      const Size cur_elems = Size(cur) * oo;
      if (!h2d(d_D.ptr, fit_obs_l + Size(a0) * oo, cur_elems) ||
          !h2d(d_R.ptr, raw_obs_r + Size(a0) * oo, cur_elems) || !zero(d_bD.ptr, cur_elems) ||
          !zero(d_bR.ptr, cur_elems))
        return 6;
      for (int k = 0; k < n_pairs; ++k) {
        const double* fk = d_f_obs_all.ptr + Size(k) * oo;
        const double* bwk = d_bar_w_all.ptr + Size(k) * oo;
        double* bfk = d_bar_f_obs_all.ptr + Size(k) * oo;
        if (rm_gemm_batched(h, false, true, obs, obs, obs, 1.0, fk, obs, 0, d_R.ptr, obs,
                            static_cast<long long>(oo), 0.0, d_Right.ptr, obs,
                            static_cast<long long>(oo), cur) != CUBLAS_STATUS_SUCCESS)
          return 7;
        if (rm_gemm_batched(h, false, true, obs, obs, obs, -1.0, bwk, obs, 0, d_Right.ptr, obs,
                            static_cast<long long>(oo), 1.0, d_bD.ptr, obs,
                            static_cast<long long>(oo), cur) != CUBLAS_STATUS_SUCCESS)
          return 7;
        if (rm_gemm_batched(h, true, false, obs, obs, obs, 1.0, d_D.ptr, obs,
                            static_cast<long long>(oo), bwk, obs, 0, 0.0, d_Bl.ptr, obs,
                            static_cast<long long>(oo), cur) != CUBLAS_STATUS_SUCCESS)
          return 7;
        if (rm_gemm_batched(h, true, false, obs, obs, obs, -1.0, d_Bl.ptr, obs,
                            static_cast<long long>(oo), fk, obs, 0, 1.0, d_bR.ptr, obs,
                            static_cast<long long>(oo), cur) != CUBLAS_STATUS_SUCCESS)
          return 7;
        if (rm_gemm_batched(h, false, false, obs, obs, obs, 1.0, d_Bl.ptr, obs,
                            static_cast<long long>(oo), d_R.ptr, obs, static_cast<long long>(oo),
                            0.0, d_Right.ptr, obs, static_cast<long long>(oo),
                            cur) != CUBLAS_STATUS_SUCCESS)
          return 7;
        if (rm_gemm(h, false, false, 1, static_cast<int>(oo), cur, -1.0, d_ones.ptr, cur,
                    d_Right.ptr, static_cast<int>(oo), 1.0, bfk, static_cast<int>(oo)) !=
            CUBLAS_STATUS_SUCCESS)
          return 7;
      }
      if (!d2h_staged(d_bD.ptr, cur_elems, h_tmp)) return 8;
      if (!all_finite(h_tmp.data(), cur_elems)) return kErrNonFiniteOutput;
      add_inplace(stage_bfo_l.data() + Size(a0) * oo, h_tmp.data(), cur_elems);
      if (!d2h_staged(d_bR.ptr, cur_elems, h_tmp)) return 8;
      if (!all_finite(h_tmp.data(), cur_elems)) return kErrNonFiniteOutput;
      add_inplace(stage_bro_r.data() + Size(a0) * oo, h_tmp.data(), cur_elems);
    }
  }

  // Stage 3: Per-pair reverse back to fit, raw_g, raw_gc, metric_g, metric_gc, cross_overlap.
  for (int k = 0; k < n_pairs; ++k) {
    const int i = pairs_i[k], j = pairs_j[k];
    const double* Dl = d_fit_l.ptr + Size(i) * ae;
    const double* Dr = p_fit_r + Size(j) * ae;
    const double* Ogl = d_raw_g_l.ptr + Size(i) * ae;
    const double* Ogr = p_raw_g_r + Size(j) * ae;
    const double* Ogcl = d_raw_gc_l.ptr + Size(i) * ae;
    const double* Ogcr = p_raw_gc_r + Size(j) * ae;
    double* bDl = d_bfit_l.ptr + Size(i) * ae;
    double* bDr = p_bfit_r + Size(j) * ae;
    double* bOgl = d_braw_g_l.ptr + Size(i) * ae;
    double* bOgr = p_braw_g_r + Size(j) * ae;
    double* bOgcl = d_braw_gc_l.ptr + Size(i) * ae;
    double* bOgcr = p_braw_gc_r + Size(j) * ae;

    if (!robust_fwd_gpu(h, aux, ext, Dl, Ogl, Dr, Ogr, d_mg.ptr, sc.Kr.ptr, sc.G_raw.ptr))
      return 9;
    if (!zero(d_bar_G_dress.ptr, ee)) return 9;
    k_copy_subblock<<<grid_for(oo), 256>>>(obs, obs, d_bar_f_obs_all.ptr + Size(k) * oo, obs,
                                           d_bar_G_dress.ptr, ext, false);
    if (!launch_ok()) return 9;
    if (!dress_adj_gpu(h, same_spin, ext, d_S.ptr, sc.G_raw.ptr, d_bar_G_dress.ptr,
                       d_bar_G_raw.ptr, d_bS.ptr, sc.tmp1.ptr, sc.tmp2.ptr))
      return 9;
    if (!robust_adj_gpu(h, aux, ext, Dl, Ogl, Dr, d_mg.ptr, sc.Kr.ptr, d_bar_G_raw.ptr, bDl, bOgl,
                        bDr, bOgr, d_bmg.ptr, sc.bar_Kr.ptr))
      return 9;

    if (!robust_fwd_gpu(h, aux, ext, Dl, Ogcl, Dr, Ogcr, d_mgc.ptr, sc.Kr.ptr, sc.G_raw.ptr))
      return 9;
    if (!zero(d_bar_G_dress.ptr, ee)) return 9;
    k_copy_subblock<<<grid_for(oo), 256>>>(obs, obs, d_bar_w_all.ptr + Size(k) * oo, obs,
                                           d_bar_G_dress.ptr, ext, false);
    if (!launch_ok()) return 9;
    if (!dress_adj_gpu(h, same_spin, ext, d_S.ptr, sc.G_raw.ptr, d_bar_G_dress.ptr,
                       d_bar_G_raw.ptr, d_bS.ptr, sc.tmp1.ptr, sc.tmp2.ptr))
      return 9;
    if (!robust_adj_gpu(h, aux, ext, Dl, Ogcl, Dr, d_mgc.ptr, sc.Kr.ptr, d_bar_G_raw.ptr, bDl,
                        bOgcl, bDr, bOgcr, d_bmgc.ptr, sc.bar_Kr.ptr))
      return 9;
  }

  if (cudaDeviceSynchronize() != cudaSuccess) return 10;
  std::vector<double> s_bfl, s_bgl, s_bgcl, s_bmg, s_bmgc, s_bfr, s_bgr, s_bgcr, s_bS;
  if (!d2h_staged(d_bfit_l.ptr, nlae, s_bfl) || !d2h_staged(d_braw_g_l.ptr, nlae, s_bgl) ||
      !d2h_staged(d_braw_gc_l.ptr, nlae, s_bgcl) || !d2h_staged(d_bmg.ptr, aa, s_bmg) ||
      !d2h_staged(d_bmgc.ptr, aa, s_bmgc))
    return 11;
  if (!same_spin) {
    if (!d2h_staged(d_bfit_r.ptr, nrae, s_bfr) || !d2h_staged(d_braw_g_r.ptr, nrae, s_bgr) ||
        !d2h_staged(d_braw_gc_r.ptr, nrae, s_bgcr) || !d2h_staged(d_bS.ptr, ee, s_bS))
      return 11;
  }
  if (!all_finite(s_bfl.data(), nlae) || !all_finite(s_bgl.data(), nlae) ||
      !all_finite(s_bgcl.data(), nlae) || !all_finite(s_bmg.data(), aa) ||
      !all_finite(s_bmgc.data(), aa) ||
      (!same_spin && (!all_finite(s_bfr.data(), nrae) || !all_finite(s_bgr.data(), nrae) ||
                      !all_finite(s_bgcr.data(), nrae) || !all_finite(s_bS.data(), ee))))
    return kErrNonFiniteOutput;

  add_inplace(bar_fit_obs_l, stage_bfo_l.data(), aoo);
  add_inplace(bar_raw_obs_r, stage_bro_r.data(), aoo);
  add_inplace(bar_fit_l, s_bfl.data(), nlae);
  add_inplace(bar_raw_g_l, s_bgl.data(), nlae);
  add_inplace(bar_raw_gc_l, s_bgcl.data(), nlae);
  add_inplace(bar_metric_g, s_bmg.data(), aa);
  add_inplace(bar_metric_gc, s_bmgc.data(), aa);
  if (!same_spin) {
    add_inplace(bar_fit_r, s_bfr.data(), nrae);
    add_inplace(bar_raw_g_r, s_bgr.data(), nrae);
    add_inplace(bar_raw_gc_r, s_bgcr.data(), nrae);
    add_inplace(bar_cross_overlap, s_bS.data(), ee);
  }
  return 0;
#endif
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (const std::exception&) {
    return kErrInternal;
  } catch (...) {
    return kErrInternal;
  }
}

// =========================================================================
// 2. C_vv MO-space pullback (all pairs in one spin channel)
//    bar_c_all has shape [n_pairs, obs - nl, obs - nr].
//    Outputs are ACCUMULATED (+=) into caller-supplied host arrays!
// =========================================================================
extern "C" int C_PULLBACK_EXPORT(
    int same_spin_int, int nl, int nr, int obs, int ext, int aux, int n_pairs,
    std::uint64_t budget_bytes, const Index* pairs_i, const Index* pairs_j, const double* fit_l,
    const double* fit_r, const double* raw_g_l, const double* raw_g_r, const double* metric_g,
    const double* fock_l, const double* fock_r, const double* cross_overlap,
    const double* bar_c_all, double* bar_fit_l, double* bar_fit_r, double* bar_raw_g_l,
    double* bar_raw_g_r, double* bar_metric_g, double* bar_fock_l, double* bar_fock_r,
    double* bar_cross_overlap) {
  try {
  if (nl <= 0 || nr <= 0 || obs <= nl || obs <= nr || ext <= obs || aux <= 0 || n_pairs <= 0 ||
      !pairs_i || !pairs_j || !fit_l || !fit_r || !raw_g_l || !raw_g_r || !metric_g || !fock_l ||
      !fock_r || !bar_c_all || !bar_fit_l || !bar_fit_r || !bar_raw_g_l || !bar_raw_g_r ||
      !bar_metric_g || !bar_fock_l || !bar_fock_r)
    return kErrInvalidArg;
  const bool same_spin = (same_spin_int != 0);
  if (same_spin && nl != nr) return kErrInvalidArg;
  if (!same_spin && (!cross_overlap || !bar_cross_overlap)) return kErrInvalidArg;
  for (int k = 0; k < n_pairs; ++k) {
    if (pairs_i[k] < 0 || pairs_i[k] >= nl || pairs_j[k] < 0 || pairs_j[k] >= nr)
      return kErrInvalidArg;
  }

  const int nvl = obs - nl;
  const int nvr = obs - nr;
  const int ncabs = ext - obs;
  const Size ae = sat_mul(Size(aux), Size(ext));
  const Size ee = sat_mul(Size(ext), Size(ext));
  const Size aa = sat_mul(Size(aux), Size(aux));
  const Size vv = sat_mul(Size(nvl), Size(nvr));
  const Size nlae = sat_mul(Size(nl), ae);
  const Size nrae = sat_mul(Size(nr), ae);
  const Size npvv = sat_mul(Size(n_pairs), vv);
  if (!fits_int(ee) || !fits_int(aa) || !fits_int(ae) || !fits_int(vv) ||
      nlae == std::numeric_limits<Size>::max() || nrae == std::numeric_limits<Size>::max() ||
      npvv == std::numeric_limits<Size>::max())
    return kErrInvalidArg;

  if (!all_finite(fit_l, nlae) || !all_finite(raw_g_l, nlae) || !all_finite(metric_g, aa) ||
      !all_finite(fock_l, ee) || !all_finite(bar_c_all, npvv))
    return kErrInvalidArg;
  if (!same_spin) {
    if (!all_finite(fit_r, nrae) || !all_finite(raw_g_r, nrae) || !all_finite(fock_r, ee) ||
        !all_finite(cross_overlap, ee))
      return kErrInvalidArg;
  }

#ifdef GQC_SINK_HOST_REFERENCE
  (void)budget_bytes;
  std::vector<double> Kr(ae), bar_Kr(ae), G_raw(ee), G_dress(ee), tmp1(ee), tmp2(ee);
  std::vector<double> bar_G_dress(ee), bar_G_raw(ee);

  std::vector<double> loc_bfl(nlae, 0.0), loc_bgl(nlae, 0.0), loc_bmg(aa, 0.0), loc_bfock_l(ee, 0.0);
  std::vector<double> loc_bfr(same_spin ? 0 : nrae, 0.0);
  std::vector<double> loc_bgr(same_spin ? 0 : nrae, 0.0);
  std::vector<double> loc_bfock_r(same_spin ? 0 : ee, 0.0);
  std::vector<double> loc_bS(same_spin ? 0 : ee, 0.0);

  double* const p_bfr = same_spin ? loc_bfl.data() : loc_bfr.data();
  double* const p_bgr = same_spin ? loc_bgl.data() : loc_bgr.data();
  double* const p_bfock_r = same_spin ? loc_bfock_l.data() : loc_bfock_r.data();
  double* const p_bS = same_spin ? nullptr : loc_bS.data();

  for (int k = 0; k < n_pairs; ++k) {
    const int i = pairs_i[k], j = pairs_j[k];
    const double* Dl = fit_l + Size(i) * ae;
    const double* Dr = (same_spin ? fit_l : fit_r) + Size(j) * ae;
    const double* Ogl = raw_g_l + Size(i) * ae;
    const double* Ogr = (same_spin ? raw_g_l : raw_g_r) + Size(j) * ae;
    double* bDl = loc_bfl.data() + Size(i) * ae;
    double* bDr = p_bfr + Size(j) * ae;
    double* bOgl = loc_bgl.data() + Size(i) * ae;
    double* bOgr = p_bgr + Size(j) * ae;

    robust_fwd_host(aux, ext, Dl, Ogl, Dr, Ogr, metric_g, Kr.data(), G_raw.data());
    dress_fwd_host(same_spin, ext, cross_overlap, G_raw.data(), G_dress.data(), tmp1.data());

    const double* bc = bar_c_all + Size(k) * vv;
    std::fill(bar_G_dress.begin(), bar_G_dress.end(), 0.0);
    rm_gemm_host(false, true, nvl, ncabs, nvr, 1.0, bc, nvr, G_dress.data() + Size(obs) * ext + nr,
                 ext, 1.0, loc_bfock_l.data() + Size(nl) * ext + obs, ext);
    rm_gemm_host(true, false, ncabs, nvr, nvl, 1.0, fock_l + Size(nl) * ext + obs, ext, bc, nvr,
                 1.0, bar_G_dress.data() + Size(obs) * ext + nr, ext);
    rm_gemm_host(true, false, nvr, ncabs, nvl, 1.0, bc, nvr, G_dress.data() + Size(nl) * ext + obs,
                 ext, 1.0, p_bfock_r + Size(nr) * ext + obs, ext);
    rm_gemm_host(false, false, nvl, ncabs, nvr, 1.0, bc, nvr,
                 (same_spin ? fock_l : fock_r) + Size(nr) * ext + obs, ext,
                 1.0, bar_G_dress.data() + Size(nl) * ext + obs, ext);

    dress_adj_host(same_spin, ext, cross_overlap, G_raw.data(), bar_G_dress.data(),
                   bar_G_raw.data(), p_bS, tmp1.data(), tmp2.data());
    robust_adj_host(aux, ext, Dl, Ogl, Dr, metric_g, Kr.data(), bar_G_raw.data(), bDl, bOgl, bDr,
                    bOgr, loc_bmg.data(), bar_Kr.data());
  }

  if (!all_finite(loc_bfl.data(), nlae) || !all_finite(loc_bgl.data(), nlae) ||
      !all_finite(loc_bmg.data(), aa) || !all_finite(loc_bfock_l.data(), ee) ||
      (!same_spin && (!all_finite(loc_bfr.data(), nrae) || !all_finite(loc_bgr.data(), nrae) ||
                      !all_finite(loc_bfock_r.data(), ee) || !all_finite(loc_bS.data(), ee))))
    return kErrNonFiniteOutput;

  add_inplace(bar_fit_l, loc_bfl.data(), nlae);
  add_inplace(bar_raw_g_l, loc_bgl.data(), nlae);
  add_inplace(bar_metric_g, loc_bmg.data(), aa);
  add_inplace(bar_fock_l, loc_bfock_l.data(), ee);
  if (!same_spin) {
    add_inplace(bar_fit_r, loc_bfr.data(), nrae);
    add_inplace(bar_raw_g_r, loc_bgr.data(), nrae);
    add_inplace(bar_fock_r, loc_bfock_r.data(), ee);
    add_inplace(bar_cross_overlap, loc_bS.data(), ee);
  }
  return 0;
#else
  const Size cap_bytes = effective_device_budget(budget_bytes);
  Size req_doubles = sat_add(sat_mul(4, nlae), sat_mul(2, aa));
  req_doubles = sat_add(req_doubles, sat_add(npvv, sat_add(sat_mul(8, ee), sat_mul(2, ae))));
  if (!same_spin) {
    req_doubles = sat_add(req_doubles, sat_add(sat_mul(4, nrae), sat_mul(4, ee)));
  }
  if (sat_mul(req_doubles, sizeof(double)) > cap_bytes) return kErrOom;

  CublasGuard cg;
  if (cg.init() != CUBLAS_STATUS_SUCCESS) return 1;
  cublasHandle_t h = cg.handle;

  DeviceBuf d_fit_l, d_fit_r, d_raw_g_l, d_raw_g_r, d_mg, d_fl, d_fr, d_S, d_bc_all;
  DeviceBuf d_bfit_l, d_bfit_r, d_braw_g_l, d_braw_g_r, d_bmg, d_bfl, d_bfr, d_bS;
  DeviceBuf d_G_dress, d_bar_G_dress, d_bar_G_raw;
  PairScratch sc;

  auto chk = [](cudaError_t e) { return e == cudaSuccess; };
  if (!chk(d_fit_l.alloc(nlae)) || !chk(d_raw_g_l.alloc(nlae)) ||
      !chk(d_bfit_l.alloc(nlae)) || !chk(d_braw_g_l.alloc(nlae)) ||
      !chk(d_mg.alloc(aa)) || !chk(d_bmg.alloc(aa)) || !chk(d_fl.alloc(ee)) ||
      !chk(d_bfl.alloc(ee)) || !chk(d_bc_all.alloc(npvv)) ||
      !chk(d_G_dress.alloc(ee)) || !chk(d_bar_G_dress.alloc(ee)) || !chk(d_bar_G_raw.alloc(ee)) ||
      !chk(sc.init(aux, ext)))
    return kErrOom;
  if (!same_spin) {
    if (!chk(d_fit_r.alloc(nrae)) || !chk(d_raw_g_r.alloc(nrae)) ||
        !chk(d_bfit_r.alloc(nrae)) || !chk(d_braw_g_r.alloc(nrae)) ||
        !chk(d_fr.alloc(ee)) || !chk(d_bfr.alloc(ee)) || !chk(d_S.alloc(ee)) ||
        !chk(d_bS.alloc(ee)))
      return kErrOom;
  }

  auto h2d = [](double* dst, const double* src, Size n) {
    return cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess;
  };
  auto zero = [](double* dst, Size n) {
    return cudaMemset(dst, 0, n * sizeof(double)) == cudaSuccess;
  };
  if (!h2d(d_fit_l.ptr, fit_l, nlae) || !h2d(d_raw_g_l.ptr, raw_g_l, nlae) ||
      !h2d(d_mg.ptr, metric_g, aa) || !h2d(d_fl.ptr, fock_l, ee) ||
      !h2d(d_bc_all.ptr, bar_c_all, npvv) || !zero(d_bfit_l.ptr, nlae) ||
      !zero(d_braw_g_l.ptr, nlae) || !zero(d_bmg.ptr, aa) || !zero(d_bfl.ptr, ee))
    return 3;
  if (!same_spin) {
    if (!h2d(d_fit_r.ptr, fit_r, nrae) || !h2d(d_raw_g_r.ptr, raw_g_r, nrae) ||
        !h2d(d_fr.ptr, fock_r, ee) || !h2d(d_S.ptr, cross_overlap, ee) ||
        !zero(d_bfit_r.ptr, nrae) || !zero(d_braw_g_r.ptr, nrae) ||
        !zero(d_bfr.ptr, ee) || !zero(d_bS.ptr, ee))
      return 3;
  }

  const double* p_fit_r = same_spin ? d_fit_l.ptr : d_fit_r.ptr;
  const double* p_raw_g_r = same_spin ? d_raw_g_l.ptr : d_raw_g_r.ptr;
  const double* p_fr = same_spin ? d_fl.ptr : d_fr.ptr;
  double* p_bfit_r = same_spin ? d_bfit_l.ptr : d_bfit_r.ptr;
  double* p_braw_g_r = same_spin ? d_braw_g_l.ptr : d_braw_g_r.ptr;
  double* p_bfr = same_spin ? d_bfl.ptr : d_bfr.ptr;

  for (int k = 0; k < n_pairs; ++k) {
    const int i = pairs_i[k], j = pairs_j[k];
    const double* Dl = d_fit_l.ptr + Size(i) * ae;
    const double* Dr = p_fit_r + Size(j) * ae;
    const double* Ogl = d_raw_g_l.ptr + Size(i) * ae;
    const double* Ogr = p_raw_g_r + Size(j) * ae;
    double* bDl = d_bfit_l.ptr + Size(i) * ae;
    double* bDr = p_bfit_r + Size(j) * ae;
    double* bOgl = d_braw_g_l.ptr + Size(i) * ae;
    double* bOgr = p_braw_g_r + Size(j) * ae;

    if (!robust_fwd_gpu(h, aux, ext, Dl, Ogl, Dr, Ogr, d_mg.ptr, sc.Kr.ptr, sc.G_raw.ptr))
      return 4;
    if (!dress_fwd_gpu(h, same_spin, ext, d_S.ptr, sc.G_raw.ptr, d_G_dress.ptr, sc.tmp1.ptr))
      return 4;

    const double* bc = d_bc_all.ptr + Size(k) * vv;
    if (!zero(d_bar_G_dress.ptr, ee)) return 4;
    if (rm_gemm(h, false, true, nvl, ncabs, nvr, 1.0, bc, nvr,
                d_G_dress.ptr + Size(obs) * ext + nr, ext, 1.0, d_bfl.ptr + Size(nl) * ext + obs,
                ext) != CUBLAS_STATUS_SUCCESS)
      return 5;
    if (rm_gemm(h, true, false, ncabs, nvr, nvl, 1.0, d_fl.ptr + Size(nl) * ext + obs, ext, bc,
                nvr, 1.0, d_bar_G_dress.ptr + Size(obs) * ext + nr, ext) != CUBLAS_STATUS_SUCCESS)
      return 5;
    if (rm_gemm(h, true, false, nvr, ncabs, nvl, 1.0, bc, nvr,
                d_G_dress.ptr + Size(nl) * ext + obs, ext, 1.0, p_bfr + Size(nr) * ext + obs,
                ext) != CUBLAS_STATUS_SUCCESS)
      return 5;
    if (rm_gemm(h, false, false, nvl, ncabs, nvr, 1.0, bc, nvr, p_fr + Size(nr) * ext + obs, ext,
                1.0, d_bar_G_dress.ptr + Size(nl) * ext + obs, ext) != CUBLAS_STATUS_SUCCESS)
      return 5;

    if (!dress_adj_gpu(h, same_spin, ext, d_S.ptr, sc.G_raw.ptr, d_bar_G_dress.ptr,
                       d_bar_G_raw.ptr, d_bS.ptr, sc.tmp1.ptr, sc.tmp2.ptr))
      return 6;
    if (!robust_adj_gpu(h, aux, ext, Dl, Ogl, Dr, d_mg.ptr, sc.Kr.ptr, d_bar_G_raw.ptr, bDl, bOgl,
                        bDr, bOgr, d_bmg.ptr, sc.bar_Kr.ptr))
      return 6;
  }

  if (cudaDeviceSynchronize() != cudaSuccess) return 7;
  std::vector<double> s_bfl, s_bgl, s_bmg, s_bfock_l, s_bfr, s_bgr, s_bfock_r, s_bS;
  if (!d2h_staged(d_bfit_l.ptr, nlae, s_bfl) || !d2h_staged(d_braw_g_l.ptr, nlae, s_bgl) ||
      !d2h_staged(d_bmg.ptr, aa, s_bmg) || !d2h_staged(d_bfl.ptr, ee, s_bfock_l))
    return 8;
  if (!same_spin) {
    if (!d2h_staged(d_bfit_r.ptr, nrae, s_bfr) || !d2h_staged(d_braw_g_r.ptr, nrae, s_bgr) ||
        !d2h_staged(d_bfr.ptr, ee, s_bfock_r) || !d2h_staged(d_bS.ptr, ee, s_bS))
      return 8;
  }
  if (!all_finite(s_bfl.data(), nlae) || !all_finite(s_bgl.data(), nlae) ||
      !all_finite(s_bmg.data(), aa) || !all_finite(s_bfock_l.data(), ee) ||
      (!same_spin && (!all_finite(s_bfr.data(), nrae) || !all_finite(s_bgr.data(), nrae) ||
                      !all_finite(s_bfock_r.data(), ee) || !all_finite(s_bS.data(), ee))))
    return kErrNonFiniteOutput;

  add_inplace(bar_fit_l, s_bfl.data(), nlae);
  add_inplace(bar_raw_g_l, s_bgl.data(), nlae);
  add_inplace(bar_metric_g, s_bmg.data(), aa);
  add_inplace(bar_fock_l, s_bfock_l.data(), ee);
  if (!same_spin) {
    add_inplace(bar_fit_r, s_bfr.data(), nrae);
    add_inplace(bar_raw_g_r, s_bgr.data(), nrae);
    add_inplace(bar_fock_r, s_bfock_r.data(), ee);
    add_inplace(bar_cross_overlap, s_bS.data(), ee);
  }
  return 0;
#endif
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (const std::exception&) {
    return kErrInternal;
  } catch (...) {
    return kErrInternal;
  }
}

// =========================================================================
// 3. Scalar (linear V + B - X) MO-space pullback (all pairs in one spin channel)
//    All [occ, aux, ext] arrays are contiguous per orbital slice!
//    fock_mixed arrays have shape [n_act, aux, ext] (indexed by i - fl, j - fr).
//    Returns energies_out[3] = {e_v, e_b, e_x} (unweighted energies for this channel)
//    and ACCUMULATES (+=) weighted derivatives into all caller-supplied host bar arrays!
// =========================================================================
extern "C" int SCALAR_PULLBACK_EXPORT(
    int same_spin_int, int nl, int nr, int fl, int fr, int obs, int ext, int aux, int n_pairs,
    std::uint64_t budget_bytes, double scalar_weight, const Index* pairs_i, const Index* pairs_j,
    const double* fit_l, const double* fit_r, const double* raw_coul_r, const double* raw_g_l,
    const double* raw_g_r, const double* raw_gc_l, const double* raw_gc_r, const double* raw_sq_l,
    const double* raw_sq_r, const double* raw_dc_l, const double* raw_dc_r,
    const double* fmix_fit_l, const double* fmix_fit_r, const double* fmix_g_l,
    const double* fmix_g_r, const double* fmix_sq_l, const double* fmix_sq_r,
    const double* metric_g, const double* metric_gc, const double* metric_sq,
    const double* metric_dc, const double* fock_l, const double* fock_r, const double* exch_l,
    const double* exch_r, const double* cross_overlap, double* energies_out, double* bar_fit_l,
    double* bar_fit_r, double* bar_raw_coul_r, double* bar_raw_g_l, double* bar_raw_g_r,
    double* bar_raw_gc_l, double* bar_raw_gc_r, double* bar_raw_sq_l, double* bar_raw_sq_r,
    double* bar_raw_dc_l, double* bar_raw_dc_r, double* bar_fmix_fit_l, double* bar_fmix_fit_r,
    double* bar_fmix_g_l, double* bar_fmix_g_r, double* bar_fmix_sq_l, double* bar_fmix_sq_r,
    double* bar_metric_g, double* bar_metric_gc, double* bar_metric_sq, double* bar_metric_dc,
    double* bar_fock_l, double* bar_fock_r, double* bar_exch_l, double* bar_exch_r,
    double* bar_cross_overlap) {
  try {
  if (nl <= fl || nr <= fr || fl < 0 || fr < 0 || obs <= nl || obs <= nr || ext <= obs ||
      aux <= 0 || n_pairs <= 0 || !std::isfinite(scalar_weight) || !pairs_i || !pairs_j ||
      !energies_out || !fit_l || !fit_r || !raw_coul_r || !raw_g_l || !raw_g_r || !raw_gc_l ||
      !raw_gc_r || !raw_sq_l || !raw_sq_r || !raw_dc_l || !raw_dc_r || !fmix_fit_l ||
      !fmix_fit_r || !fmix_g_l || !fmix_g_r || !fmix_sq_l || !fmix_sq_r || !metric_g ||
      !metric_gc || !metric_sq || !metric_dc || !fock_l || !fock_r || !exch_l || !exch_r ||
      !bar_fit_l || !bar_fit_r || !bar_raw_coul_r || !bar_raw_g_l || !bar_raw_g_r ||
      !bar_raw_gc_l || !bar_raw_gc_r || !bar_raw_sq_l || !bar_raw_sq_r || !bar_raw_dc_l ||
      !bar_raw_dc_r || !bar_fmix_fit_l || !bar_fmix_fit_r || !bar_fmix_g_l || !bar_fmix_g_r ||
      !bar_fmix_sq_l || !bar_fmix_sq_r || !bar_metric_g || !bar_metric_gc || !bar_metric_sq ||
      !bar_metric_dc || !bar_fock_l || !bar_fock_r || !bar_exch_l || !bar_exch_r)
    return kErrInvalidArg;
  const bool same_spin = (same_spin_int != 0);
  if (same_spin && (nl != nr || fl != fr)) return kErrInvalidArg;
  if (!same_spin && (!cross_overlap || !bar_cross_overlap)) return kErrInvalidArg;
  for (int k = 0; k < n_pairs; ++k) {
    if (pairs_i[k] < fl || pairs_i[k] >= nl || pairs_j[k] < fr || pairs_j[k] >= nr)
      return kErrInvalidArg;
  }

  const int nact_l = nl - fl;
  const int nact_r = nr - fr;
  const Size ae = sat_mul(Size(aux), Size(ext));
  const Size ee = sat_mul(Size(ext), Size(ext));
  const Size aa = sat_mul(Size(aux), Size(aux));
  const Size nlae = sat_mul(Size(nl), ae);
  const Size nrae = sat_mul(Size(nr), ae);
  const Size nacl_ae = sat_mul(Size(nact_l), ae);
  const Size nacr_ae = sat_mul(Size(nact_r), ae);
  if (!fits_int(ee) || !fits_int(aa) || !fits_int(ae) ||
      nlae == std::numeric_limits<Size>::max() || nrae == std::numeric_limits<Size>::max() ||
      nacl_ae == std::numeric_limits<Size>::max() || nacr_ae == std::numeric_limits<Size>::max())
    return kErrInvalidArg;

  if (!all_finite(fit_l, nlae) || !all_finite(raw_coul_r, nrae) || !all_finite(raw_g_l, nlae) ||
      !all_finite(raw_gc_l, nlae) || !all_finite(raw_sq_l, nlae) || !all_finite(raw_dc_l, nlae) ||
      !all_finite(fmix_fit_l, nacl_ae) || !all_finite(fmix_g_l, nacl_ae) ||
      !all_finite(fmix_sq_l, nacl_ae) || !all_finite(metric_g, aa) || !all_finite(metric_gc, aa) ||
      !all_finite(metric_sq, aa) || !all_finite(metric_dc, aa) || !all_finite(fock_l, ee) ||
      !all_finite(exch_l, ee))
    return kErrInvalidArg;
  if (!same_spin) {
    if (!all_finite(fit_r, nrae) || !all_finite(raw_g_r, nrae) || !all_finite(raw_gc_r, nrae) ||
        !all_finite(raw_sq_r, nrae) || !all_finite(raw_dc_r, nrae) ||
        !all_finite(fmix_fit_r, nacr_ae) || !all_finite(fmix_g_r, nacr_ae) ||
        !all_finite(fmix_sq_r, nacr_ae) || !all_finite(fock_r, ee) || !all_finite(exch_r, ee) ||
        !all_finite(cross_overlap, ee))
      return kErrInvalidArg;
  }

  const double norm = same_spin ? 0.5 : 1.0;
  const double w_v = scalar_weight;
  const double w_b = scalar_weight * norm;
  const double w_x = -scalar_weight * norm;

#ifdef GQC_SINK_HOST_REFERENCE
  (void)budget_bytes;
  std::vector<double> Kr_g(ae), Kr_sq(ae), Kr_dc(ae), Kr_tmp(ae), bar_Kr(ae);
  std::vector<double> Kr_xg_r(ae), Kr_xsq_r(ae);
  std::vector<double> f_raw(ee), f(ee), qf(ee), f_mq(ee), ff_raw(ee), ff(ee), uf_raw(ee), uf(ee);
  std::vector<double> fg_raw(ee), fg(ee), Cmat(ee), chi(ee);
  std::vector<double> fx_raw(ee), fx(ee), ffx_raw(ee), ffx(ee);
  std::vector<double> FK_l(ee), FK_r(ee), act_FK_chi(ee), act_K_f(ee), act_F_qf(ee), act_F_f(ee);
  std::vector<double> bar_f(ee), bar_ff(ee), bar_uf(ee), bar_fg(ee), bar_chi(ee), bar_C(ee);
  std::vector<double> bar_tmp(ee), bar_raw(ee), tmp1(ee), tmp2(ee);

  // Stage outputs locally so caller arrays stay untouched on non-finite failure.
  std::vector<double> loc_bfl(nlae, 0.0), loc_brcr(nrae, 0.0);
  std::vector<double> loc_bgl(nlae, 0.0), loc_bgcl(nlae, 0.0), loc_bsql(nlae, 0.0), loc_bdcl(nlae, 0.0);
  std::vector<double> loc_bxfl(nacl_ae, 0.0), loc_bxgl(nacl_ae, 0.0), loc_bxsql(nacl_ae, 0.0);
  std::vector<double> loc_bmg(aa, 0.0), loc_bmgc(aa, 0.0), loc_bmsq(aa, 0.0), loc_bmdc(aa, 0.0);
  std::vector<double> loc_bfock_l(ee, 0.0), loc_bexch_l(ee, 0.0);

  std::vector<double> loc_bfr(same_spin ? 0 : nrae, 0.0);
  std::vector<double> loc_bgr(same_spin ? 0 : nrae, 0.0);
  std::vector<double> loc_bgcr(same_spin ? 0 : nrae, 0.0);
  std::vector<double> loc_bsqr(same_spin ? 0 : nrae, 0.0);
  std::vector<double> loc_bdcr(same_spin ? 0 : nrae, 0.0);
  std::vector<double> loc_bxfr(same_spin ? 0 : nacr_ae, 0.0);
  std::vector<double> loc_bxgr(same_spin ? 0 : nacr_ae, 0.0);
  std::vector<double> loc_bxsqr(same_spin ? 0 : nacr_ae, 0.0);
  std::vector<double> loc_bfock_r(same_spin ? 0 : ee, 0.0);
  std::vector<double> loc_bexch_r(same_spin ? 0 : ee, 0.0);
  std::vector<double> loc_bS(same_spin ? 0 : ee, 0.0);

  double* const p_bfl = loc_bfl.data();
  double* const p_bfr = same_spin ? loc_bfl.data() : loc_bfr.data();
  double* const p_bgl = loc_bgl.data();
  double* const p_bgr = same_spin ? loc_bgl.data() : loc_bgr.data();
  double* const p_bgcl = loc_bgcl.data();
  double* const p_bgcr = same_spin ? loc_bgcl.data() : loc_bgcr.data();
  double* const p_bsql = loc_bsql.data();
  double* const p_bsqr = same_spin ? loc_bsql.data() : loc_bsqr.data();
  double* const p_bdcl = loc_bdcl.data();
  double* const p_bdcr = same_spin ? loc_bdcl.data() : loc_bdcr.data();
  double* const p_bxfl = loc_bxfl.data();
  double* const p_bxfr = same_spin ? loc_bxfl.data() : loc_bxfr.data();
  double* const p_bxgl = loc_bxgl.data();
  double* const p_bxgr = same_spin ? loc_bxgl.data() : loc_bxgr.data();
  double* const p_bxsql = loc_bxsql.data();
  double* const p_bxsqr = same_spin ? loc_bxsql.data() : loc_bxsqr.data();
  double* const p_bfock_l = loc_bfock_l.data();
  double* const p_bfock_r = same_spin ? loc_bfock_l.data() : loc_bfock_r.data();
  double* const p_bexch_l = loc_bexch_l.data();
  double* const p_bexch_r = same_spin ? loc_bexch_l.data() : loc_bexch_r.data();
  double* const p_bS = same_spin ? nullptr : loc_bS.data();

  scale_add_host(ee, 1.0, fock_l, 1.0, exch_l, FK_l.data());
  if (!same_spin) scale_add_host(ee, 1.0, fock_r, 1.0, exch_r, FK_r.data());
  const double* p_FK_r = same_spin ? FK_l.data() : FK_r.data();
  const double* p_fock_r = same_spin ? fock_l : fock_r;
  const double* p_exch_r = same_spin ? exch_l : exch_r;

  double e_v = 0.0, e_b = 0.0, e_x = 0.0;

  for (int k = 0; k < n_pairs; ++k) {
    const int i = pairs_i[k], j = pairs_j[k];
    const int ii = i - fl, jj = j - fr;

    const double* Dl = fit_l + Size(i) * ae;
    const double* Dr = (same_spin ? fit_l : fit_r) + Size(j) * ae;
    const double* Rcr = raw_coul_r + Size(j) * ae;
    const double* Ogl = raw_g_l + Size(i) * ae;
    const double* Ogr = (same_spin ? raw_g_l : raw_g_r) + Size(j) * ae;
    const double* Ogcl = raw_gc_l + Size(i) * ae;
    const double* Ogcr = (same_spin ? raw_gc_l : raw_gc_r) + Size(j) * ae;
    const double* Osql = raw_sq_l + Size(i) * ae;
    const double* Osqr = (same_spin ? raw_sq_l : raw_sq_r) + Size(j) * ae;
    const double* Odcl = raw_dc_l + Size(i) * ae;
    const double* Odcr = (same_spin ? raw_dc_l : raw_dc_r) + Size(j) * ae;

    const double* XDl = fmix_fit_l + Size(ii) * ae;
    const double* XDr = (same_spin ? fmix_fit_l : fmix_fit_r) + Size(jj) * ae;
    const double* XOgl = fmix_g_l + Size(ii) * ae;
    const double* XOgr = (same_spin ? fmix_g_l : fmix_g_r) + Size(jj) * ae;
    const double* XOsql = fmix_sq_l + Size(ii) * ae;
    const double* XOsqr = (same_spin ? fmix_sq_l : fmix_sq_r) + Size(jj) * ae;

    double* bDl = p_bfl + Size(i) * ae;
    double* bDr = p_bfr + Size(j) * ae;
    double* bRcr = loc_brcr.data() + Size(j) * ae;
    double* bOgl = p_bgl + Size(i) * ae;
    double* bOgr = p_bgr + Size(j) * ae;
    double* bOgcl = p_bgcl + Size(i) * ae;
    double* bOgcr = p_bgcr + Size(j) * ae;
    double* bOsql = p_bsql + Size(i) * ae;
    double* bOsqr = p_bsqr + Size(j) * ae;
    double* bOdcl = p_bdcl + Size(i) * ae;
    double* bOdcr = p_bdcr + Size(j) * ae;
    double* bXDl = p_bxfl + Size(ii) * ae;
    double* bXDr = p_bxfr + Size(jj) * ae;
    double* bXOgl = p_bxgl + Size(ii) * ae;
    double* bXOgr = p_bxgr + Size(jj) * ae;
    double* bXOsql = p_bxsql + Size(ii) * ae;
    double* bXOsqr = p_bxsqr + Size(jj) * ae;

    // 1. Forward operators
    robust_fwd_host(aux, ext, Dl, Ogl, Dr, Ogr, metric_g, Kr_g.data(), f_raw.data());
    dress_fwd_host(same_spin, ext, cross_overlap, f_raw.data(), f.data(), tmp1.data());
    strong_mask_host(ext, nl, nr, obs, f.data(), qf.data(), false);
    strong_mask_host(ext, nl, nr, obs, f.data(), f_mq.data(), true);

    robust_fwd_host(aux, ext, Dl, Ogcl, Dr, Ogcr, metric_gc, Kr_tmp.data(), fg_raw.data());
    dress_fwd_host(same_spin, ext, cross_overlap, fg_raw.data(), fg.data(), tmp1.data());

    // Linear V energy & adjoint
    rm_gemm_host(true, false, ext, ext, aux, 1.0, Dl, ext, Rcr, ext, 0.0, Cmat.data(), ext);
    std::fill(bar_f.begin(), bar_f.end(), 0.0);
    double v_dot = 0.0;
    v_mask_acc_host(ext, nl, nr, obs, -2.0 * w_v, Cmat.data(), f.data(), bar_C.data(),
                    bar_f.data(), v_dot);
    e_v += 2.0 * (fg[Size(i) * ext + j] - v_dot);

    rm_gemm_host(false, true, aux, ext, ext, 1.0, Rcr, ext, bar_C.data(), ext, 1.0, bDl, ext);
    rm_gemm_host(false, false, aux, ext, ext, 1.0, Dl, ext, bar_C.data(), ext, 1.0, bRcr, ext);

    std::fill(bar_fg.begin(), bar_fg.end(), 0.0);
    bar_fg[Size(i) * ext + j] = 2.0 * w_v;
    dress_adj_host(same_spin, ext, cross_overlap, fg_raw.data(), bar_fg.data(), bar_raw.data(),
                   p_bS, tmp1.data(), tmp2.data());
    robust_adj_host(aux, ext, Dl, Ogcl, Dr, metric_gc, Kr_tmp.data(), bar_raw.data(), bDl, bOgcl,
                    bDr, bOgcr, loc_bmgc.data(), bar_Kr.data());

    // 2. Quadratic B forward
    if (same_spin) {
      std::fill(chi.begin(), chi.end(), 0.0);
      chi[Size(i) * ext + j] += 0.25;
      chi[Size(j) * ext + i] -= 0.25;
    } else {
      for (int p = 0; p < ext; ++p)
        for (int q = 0; q < ext; ++q)
          chi[Size(p) * ext + q] =
              0.125 * cross_overlap[Size(p) * ext + j] * cross_overlap[Size(i) * ext + q] +
              ((p == i && q == j) ? 0.375 : 0.0);
    }
    robust_fwd_host(aux, ext, Dl, Osql, Dr, Osqr, metric_sq, Kr_sq.data(), ff_raw.data());
    dress_fwd_host(same_spin, ext, cross_overlap, ff_raw.data(), ff.data(), tmp1.data());

    robust_fwd_host(aux, ext, Dl, Odcl, Dr, Odcr, metric_dc, Kr_dc.data(), uf_raw.data());
    dress_fwd_host(same_spin, ext, cross_overlap, uf_raw.data(), uf.data(), tmp1.data());

    // Actions: A_l @ M + M @ A_r^T
    rm_gemm_host(false, false, ext, ext, ext, 1.0, FK_l.data(), ext, chi.data(), ext, 0.0,
                 act_FK_chi.data(), ext);
    rm_gemm_host(false, true, ext, ext, ext, 1.0, chi.data(), ext, p_FK_r, ext, 1.0,
                 act_FK_chi.data(), ext);

    rm_gemm_host(false, false, ext, ext, ext, 1.0, exch_l, ext, f.data(), ext, 0.0, act_K_f.data(),
                 ext);
    rm_gemm_host(false, true, ext, ext, ext, 1.0, f.data(), ext, p_exch_r, ext, 1.0, act_K_f.data(),
                 ext);

    rm_gemm_host(false, false, ext, ext, ext, 1.0, fock_l, ext, qf.data(), ext, 0.0,
                 act_F_qf.data(), ext);
    rm_gemm_host(false, true, ext, ext, ext, 1.0, qf.data(), ext, p_fock_r, ext, 1.0, act_F_qf.data(),
                 ext);

    rm_gemm_host(false, false, ext, ext, ext, 1.0, fock_l, ext, f.data(), ext, 0.0, act_F_f.data(),
                 ext);
    rm_gemm_host(false, true, ext, ext, ext, 1.0, f.data(), ext, p_fock_r, ext, 1.0, act_F_f.data(),
                 ext);

    const double eb_k = dot_host(ee, chi.data(), uf.data()) +
                        dot_host(ee, ff.data(), act_FK_chi.data()) -
                        dot_host(ee, f.data(), act_K_f.data()) +
                        dot_host(ee, qf.data(), act_F_qf.data()) -
                        dot_host(ee, f.data(), act_F_f.data());
    e_b += norm * eb_k;

    // Quadratic B adjoints
    // Term 1: w_b * dot(chi, uf)
    for (Size t = 0; t < ee; ++t) {
      bar_uf[t] = w_b * chi[t];
      bar_chi[t] = w_b * uf[t];
      bar_ff[t] = w_b * act_FK_chi[t];
    }
    // Term 2: w_b * dot(ff, FK_l @ chi + chi @ FK_r^T)
    rm_gemm_host(false, true, ext, ext, ext, w_b, ff.data(), ext, chi.data(), ext, 0.0, tmp1.data(),
                 ext);
    axpy_host(ee, 1.0, tmp1.data(), p_bfock_l);
    axpy_host(ee, 1.0, tmp1.data(), p_bexch_l);
    rm_gemm_host(true, false, ext, ext, ext, w_b, ff.data(), ext, chi.data(), ext, 0.0, tmp1.data(),
                 ext);
    axpy_host(ee, 1.0, tmp1.data(), p_bfock_r);
    axpy_host(ee, 1.0, tmp1.data(), p_bexch_r);
    rm_gemm_host(true, false, ext, ext, ext, w_b, FK_l.data(), ext, ff.data(), ext, 1.0,
                 bar_chi.data(), ext);
    rm_gemm_host(false, false, ext, ext, ext, w_b, ff.data(), ext, p_FK_r, ext, 1.0, bar_chi.data(),
                 ext);

    // Term 3: -w_b * dot(f, K_l @ f + f @ K_r^T)
    rm_gemm_host(false, true, ext, ext, ext, -w_b, f.data(), ext, f.data(), ext, 1.0, p_bexch_l,
                 ext);
    rm_gemm_host(true, false, ext, ext, ext, -w_b, f.data(), ext, f.data(), ext, 1.0, p_bexch_r,
                 ext);
    axpy_host(ee, -w_b, act_K_f.data(), bar_f.data());
    rm_gemm_host(true, false, ext, ext, ext, -w_b, exch_l, ext, f.data(), ext, 1.0, bar_f.data(),
                 ext);
    rm_gemm_host(false, false, ext, ext, ext, -w_b, f.data(), ext, p_exch_r, ext, 1.0, bar_f.data(),
                 ext);

    // Term 4: +w_b * dot(qf, F_l @ qf + qf @ F_r^T)
    rm_gemm_host(false, true, ext, ext, ext, w_b, qf.data(), ext, qf.data(), ext, 1.0, p_bfock_l,
                 ext);
    rm_gemm_host(true, false, ext, ext, ext, w_b, qf.data(), ext, qf.data(), ext, 1.0, p_bfock_r,
                 ext);
    scale_add_host(ee, w_b, act_F_qf.data(), 0.0, act_F_qf.data(), bar_tmp.data());
    rm_gemm_host(true, false, ext, ext, ext, w_b, fock_l, ext, qf.data(), ext, 1.0, bar_tmp.data(),
                 ext);
    rm_gemm_host(false, false, ext, ext, ext, w_b, qf.data(), ext, p_fock_r, ext, 1.0, bar_tmp.data(),
                 ext);
    add_strong_masked_host(ext, nl, nr, obs, 1.0, bar_tmp.data(), bar_f.data(), false);

    // Term 5: -w_b * dot(f, F_l @ f + f @ F_r^T)
    rm_gemm_host(false, true, ext, ext, ext, -w_b, f.data(), ext, f.data(), ext, 1.0, p_bfock_l,
                 ext);
    rm_gemm_host(true, false, ext, ext, ext, -w_b, f.data(), ext, f.data(), ext, 1.0, p_bfock_r,
                 ext);
    axpy_host(ee, -w_b, act_F_f.data(), bar_f.data());
    rm_gemm_host(true, false, ext, ext, ext, -w_b, fock_l, ext, f.data(), ext, 1.0, bar_f.data(),
                 ext);
    rm_gemm_host(false, false, ext, ext, ext, -w_b, f.data(), ext, p_fock_r, ext, 1.0, bar_f.data(),
                 ext);

    // 3. Quadratic X forward & adjoint
    // f_x_raw = fitted(g, xl, cr) + fitted(g, cl, xr)
    robust_fwd_host(aux, ext, XDl, XOgl, Dr, Ogr, metric_g, Kr_tmp.data(), fx_raw.data());
    robust_fwd_host(aux, ext, Dl, Ogl, XDr, XOgr, metric_g, Kr_xg_r.data(), tmp1.data());
    axpy_host(ee, 1.0, tmp1.data(), fx_raw.data());
    dress_fwd_host(same_spin, ext, cross_overlap, fx_raw.data(), fx.data(), tmp1.data());

    // ff_x_raw = fitted(sq, xl, cr) + fitted(sq, cl, xr)
    robust_fwd_host(aux, ext, XDl, XOsql, Dr, Osqr, metric_sq, Kr_tmp.data(), ffx_raw.data());
    robust_fwd_host(aux, ext, Dl, Osql, XDr, XOsqr, metric_sq, Kr_xsq_r.data(), tmp1.data());
    axpy_host(ee, 1.0, tmp1.data(), ffx_raw.data());
    dress_fwd_host(same_spin, ext, cross_overlap, ffx_raw.data(), ffx.data(), tmp1.data());

    const double ex_k = dot_host(ee, chi.data(), ffx.data()) - dot_host(ee, f_mq.data(), fx.data());
    e_x += norm * ex_k;

    // X adjoints: w_x * (dot(chi, ffx) - dot(f_mq, fx))
    axpy_host(ee, w_x, ffx.data(), bar_chi.data());
    scale_add_host(ee, w_x, chi.data(), 0.0, chi.data(), bar_tmp.data());
    dress_adj_host(same_spin, ext, cross_overlap, ffx_raw.data(), bar_tmp.data(), bar_raw.data(),
                   p_bS, tmp1.data(), tmp2.data());
    // Note Kr_sq holds Osqr - M_sq @ Dr; Kr_xsq_r holds XOsqr - M_sq @ XDr
    robust_adj_host(aux, ext, XDl, XOsql, Dr, metric_sq, Kr_sq.data(), bar_raw.data(), bXDl, bXOsql,
                    bDr, bOsqr, loc_bmsq.data(), bar_Kr.data());
    robust_adj_host(aux, ext, Dl, Osql, XDr, metric_sq, Kr_xsq_r.data(), bar_raw.data(), bDl, bOsql,
                    bXDr, bXOsqr, loc_bmsq.data(), bar_Kr.data());

    add_strong_masked_host(ext, nl, nr, obs, -w_x, fx.data(), bar_f.data(), true);
    scale_add_host(ee, -w_x, f_mq.data(), 0.0, f_mq.data(), bar_tmp.data());
    dress_adj_host(same_spin, ext, cross_overlap, fx_raw.data(), bar_tmp.data(), bar_raw.data(),
                   p_bS, tmp1.data(), tmp2.data());
    robust_adj_host(aux, ext, XDl, XOgl, Dr, metric_g, Kr_g.data(), bar_raw.data(), bXDl, bXOgl,
                    bDr, bOgr, loc_bmg.data(), bar_Kr.data());
    robust_adj_host(aux, ext, Dl, Ogl, XDr, metric_g, Kr_xg_r.data(), bar_raw.data(), bDl, bOgl,
                    bXDr, bXOgr, loc_bmg.data(), bar_Kr.data());

    // 4. Propagate bar_chi -> bar_S (only for opposite spin)
    if (!same_spin) {
      for (int p = 0; p < ext; ++p) {
        double acc_pj = 0.0;
        for (int q = 0; q < ext; ++q)
          acc_pj += bar_chi[Size(p) * ext + q] * cross_overlap[Size(i) * ext + q];
        p_bS[Size(p) * ext + j] += 0.125 * acc_pj;
      }
      for (int q = 0; q < ext; ++q) {
        double acc_iq = 0.0;
        for (int p = 0; p < ext; ++p)
          acc_iq += bar_chi[Size(p) * ext + q] * cross_overlap[Size(p) * ext + j];
        p_bS[Size(i) * ext + q] += 0.125 * acc_iq;
      }
    }

    // 5. Propagate bar_f, bar_ff, bar_uf back through dress_adj + robust_adj
    dress_adj_host(same_spin, ext, cross_overlap, f_raw.data(), bar_f.data(), bar_raw.data(),
                   p_bS, tmp1.data(), tmp2.data());
    robust_adj_host(aux, ext, Dl, Ogl, Dr, metric_g, Kr_g.data(), bar_raw.data(), bDl, bOgl, bDr,
                    bOgr, loc_bmg.data(), bar_Kr.data());

    dress_adj_host(same_spin, ext, cross_overlap, ff_raw.data(), bar_ff.data(), bar_raw.data(),
                   p_bS, tmp1.data(), tmp2.data());
    robust_adj_host(aux, ext, Dl, Osql, Dr, metric_sq, Kr_sq.data(), bar_raw.data(), bDl, bOsql,
                    bDr, bOsqr, loc_bmsq.data(), bar_Kr.data());

    dress_adj_host(same_spin, ext, cross_overlap, uf_raw.data(), bar_uf.data(), bar_raw.data(),
                   p_bS, tmp1.data(), tmp2.data());
    robust_adj_host(aux, ext, Dl, Odcl, Dr, metric_dc, Kr_dc.data(), bar_raw.data(), bDl, bOdcl,
                    bDr, bOdcr, loc_bmdc.data(), bar_Kr.data());
  }

  if (!std::isfinite(e_v) || !std::isfinite(e_b) || !std::isfinite(e_x) ||
      !all_finite(loc_bfl.data(), nlae) || !all_finite(loc_brcr.data(), nrae) ||
      !all_finite(loc_bgl.data(), nlae) || !all_finite(loc_bgcl.data(), nlae) ||
      !all_finite(loc_bsql.data(), nlae) || !all_finite(loc_bdcl.data(), nlae) ||
      !all_finite(loc_bxfl.data(), nacl_ae) || !all_finite(loc_bxgl.data(), nacl_ae) ||
      !all_finite(loc_bxsql.data(), nacl_ae) || !all_finite(loc_bmg.data(), aa) ||
      !all_finite(loc_bmgc.data(), aa) || !all_finite(loc_bmsq.data(), aa) ||
      !all_finite(loc_bmdc.data(), aa) || !all_finite(loc_bfock_l.data(), ee) ||
      !all_finite(loc_bexch_l.data(), ee) ||
      (!same_spin &&
       (!all_finite(loc_bfr.data(), nrae) || !all_finite(loc_bgr.data(), nrae) ||
        !all_finite(loc_bgcr.data(), nrae) || !all_finite(loc_bsqr.data(), nrae) ||
        !all_finite(loc_bdcr.data(), nrae) || !all_finite(loc_bxfr.data(), nacr_ae) ||
        !all_finite(loc_bxgr.data(), nacr_ae) || !all_finite(loc_bxsqr.data(), nacr_ae) ||
        !all_finite(loc_bfock_r.data(), ee) || !all_finite(loc_bexch_r.data(), ee) ||
        !all_finite(loc_bS.data(), ee))))
    return kErrNonFiniteOutput;

  add_inplace(bar_fit_l, loc_bfl.data(), nlae);
  add_inplace(bar_raw_coul_r, loc_brcr.data(), nrae);
  add_inplace(bar_raw_g_l, loc_bgl.data(), nlae);
  add_inplace(bar_raw_gc_l, loc_bgcl.data(), nlae);
  add_inplace(bar_raw_sq_l, loc_bsql.data(), nlae);
  add_inplace(bar_raw_dc_l, loc_bdcl.data(), nlae);
  add_inplace(bar_fmix_fit_l, loc_bxfl.data(), nacl_ae);
  add_inplace(bar_fmix_g_l, loc_bxgl.data(), nacl_ae);
  add_inplace(bar_fmix_sq_l, loc_bxsql.data(), nacl_ae);
  add_inplace(bar_metric_g, loc_bmg.data(), aa);
  add_inplace(bar_metric_gc, loc_bmgc.data(), aa);
  add_inplace(bar_metric_sq, loc_bmsq.data(), aa);
  add_inplace(bar_metric_dc, loc_bmdc.data(), aa);
  add_inplace(bar_fock_l, loc_bfock_l.data(), ee);
  add_inplace(bar_exch_l, loc_bexch_l.data(), ee);
  if (!same_spin) {
    add_inplace(bar_fit_r, loc_bfr.data(), nrae);
    add_inplace(bar_raw_g_r, loc_bgr.data(), nrae);
    add_inplace(bar_raw_gc_r, loc_bgcr.data(), nrae);
    add_inplace(bar_raw_sq_r, loc_bsqr.data(), nrae);
    add_inplace(bar_raw_dc_r, loc_bdcr.data(), nrae);
    add_inplace(bar_fmix_fit_r, loc_bxfr.data(), nacr_ae);
    add_inplace(bar_fmix_g_r, loc_bxgr.data(), nacr_ae);
    add_inplace(bar_fmix_sq_r, loc_bxsqr.data(), nacr_ae);
    add_inplace(bar_fock_r, loc_bfock_r.data(), ee);
    add_inplace(bar_exch_r, loc_bexch_r.data(), ee);
    add_inplace(bar_cross_overlap, loc_bS.data(), ee);
  }
  energies_out[0] = e_v;
  energies_out[1] = e_b;
  energies_out[2] = e_x;
  return 0;
#else
  const Size cap_bytes = effective_device_budget(budget_bytes);
  Size req_doubles = sat_add(sat_mul(10, nlae), sat_mul(2, nrae));
  req_doubles = sat_add(req_doubles, sat_mul(6, nacl_ae));
  req_doubles = sat_add(req_doubles, sat_add(sat_mul(8, aa), sat_mul(5, ee)));
  if (!same_spin) {
    req_doubles = sat_add(req_doubles, sat_mul(10, nrae));
    req_doubles = sat_add(req_doubles, sat_mul(6, nacr_ae));
    req_doubles = sat_add(req_doubles, sat_mul(7, ee));
  }
  req_doubles = sat_add(req_doubles, sat_add(sat_mul(7, ae), sat_add(sat_mul(30, ee), 4)));
  if (sat_mul(req_doubles, sizeof(double)) > cap_bytes) return kErrOom;

  CublasGuard cg;
  if (cg.init() != CUBLAS_STATUS_SUCCESS) return 1;
  cublasHandle_t h = cg.handle;

  DeviceBuf d_fit_l, d_fit_r, d_rcoul_r;
  DeviceBuf d_g_l, d_g_r, d_gc_l, d_gc_r, d_sq_l, d_sq_r, d_dc_l, d_dc_r;
  DeviceBuf d_xf_l, d_xf_r, d_xg_l, d_xg_r, d_xsq_l, d_xsq_r;
  DeviceBuf d_mg, d_mgc, d_msq, d_mdc, d_fl, d_fr, d_kl, d_kr, d_S;
  DeviceBuf d_FKl, d_FKr;

  DeviceBuf d_bfit_l, d_bfit_r, d_brcoul_r;
  DeviceBuf d_bg_l, d_bg_r, d_bgc_l, d_bgc_r, d_bsq_l, d_bsq_r, d_bdc_l, d_bdc_r;
  DeviceBuf d_bxf_l, d_bxf_r, d_bxg_l, d_bxg_r, d_bxsq_l, d_bxsq_r;
  DeviceBuf d_bmg, d_bmgc, d_bmsq, d_bmdc, d_bfl, d_bfr, d_bkl, d_bkr, d_bS;

  DeviceBuf d_Kr_g, d_Kr_sq, d_Kr_dc, d_Kr_tmp, d_Kr_xg_r, d_Kr_xsq_r, d_bar_Kr;
  DeviceBuf d_f_raw, d_f, d_qf, d_fmq, d_ff_raw, d_ff, d_uf_raw, d_uf, d_fg_raw, d_fg;
  DeviceBuf d_C, d_chi, d_fx_raw, d_fx, d_ffx_raw, d_ffx;
  DeviceBuf d_act_FK_chi, d_act_K_f, d_act_F_qf, d_act_F_f;
  DeviceBuf d_bar_f, d_bar_ff, d_bar_uf, d_bar_fg, d_bar_chi, d_bar_C, d_bar_tmp, d_bar_raw;
  DeviceBuf d_tmp1, d_tmp2, d_scalar_acc;

  auto chk = [](cudaError_t e) { return e == cudaSuccess; };
  auto h2d = [](double* dst, const double* src, Size n) {
    return cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess;
  };
  auto zero = [](double* dst, Size n) {
    return cudaMemset(dst, 0, n * sizeof(double)) == cudaSuccess;
  };

  // Allocate left-spin & shared buffers
  if (!chk(d_fit_l.alloc(Size(nl) * ae)) || !chk(d_rcoul_r.alloc(Size(nr) * ae)) ||
      !chk(d_g_l.alloc(Size(nl) * ae)) || !chk(d_gc_l.alloc(Size(nl) * ae)) ||
      !chk(d_sq_l.alloc(Size(nl) * ae)) || !chk(d_dc_l.alloc(Size(nl) * ae)) ||
      !chk(d_xf_l.alloc(Size(nact_l) * ae)) || !chk(d_xg_l.alloc(Size(nact_l) * ae)) ||
      !chk(d_xsq_l.alloc(Size(nact_l) * ae)) || !chk(d_mg.alloc(aa)) || !chk(d_mgc.alloc(aa)) ||
      !chk(d_msq.alloc(aa)) || !chk(d_mdc.alloc(aa)) || !chk(d_fl.alloc(ee)) ||
      !chk(d_kl.alloc(ee)) || !chk(d_FKl.alloc(ee)) || !chk(d_bfit_l.alloc(Size(nl) * ae)) ||
      !chk(d_brcoul_r.alloc(Size(nr) * ae)) || !chk(d_bg_l.alloc(Size(nl) * ae)) ||
      !chk(d_bgc_l.alloc(Size(nl) * ae)) || !chk(d_bsq_l.alloc(Size(nl) * ae)) ||
      !chk(d_bdc_l.alloc(Size(nl) * ae)) || !chk(d_bxf_l.alloc(Size(nact_l) * ae)) ||
      !chk(d_bxg_l.alloc(Size(nact_l) * ae)) || !chk(d_bxsq_l.alloc(Size(nact_l) * ae)) ||
      !chk(d_bmg.alloc(aa)) || !chk(d_bmgc.alloc(aa)) || !chk(d_bmsq.alloc(aa)) ||
      !chk(d_bmdc.alloc(aa)) || !chk(d_bfl.alloc(ee)) || !chk(d_bkl.alloc(ee)))
    return 2;

  if (!same_spin) {
    if (!chk(d_fit_r.alloc(Size(nr) * ae)) || !chk(d_g_r.alloc(Size(nr) * ae)) ||
        !chk(d_gc_r.alloc(Size(nr) * ae)) || !chk(d_sq_r.alloc(Size(nr) * ae)) ||
        !chk(d_dc_r.alloc(Size(nr) * ae)) || !chk(d_xf_r.alloc(Size(nact_r) * ae)) ||
        !chk(d_xg_r.alloc(Size(nact_r) * ae)) || !chk(d_xsq_r.alloc(Size(nact_r) * ae)) ||
        !chk(d_fr.alloc(ee)) || !chk(d_kr.alloc(ee)) || !chk(d_FKr.alloc(ee)) ||
        !chk(d_S.alloc(ee)) || !chk(d_bfit_r.alloc(Size(nr) * ae)) ||
        !chk(d_bg_r.alloc(Size(nr) * ae)) || !chk(d_bgc_r.alloc(Size(nr) * ae)) ||
        !chk(d_bsq_r.alloc(Size(nr) * ae)) || !chk(d_bdc_r.alloc(Size(nr) * ae)) ||
        !chk(d_bxf_r.alloc(Size(nact_r) * ae)) || !chk(d_bxg_r.alloc(Size(nact_r) * ae)) ||
        !chk(d_bxsq_r.alloc(Size(nact_r) * ae)) || !chk(d_bfr.alloc(ee)) || !chk(d_bkr.alloc(ee)) ||
        !chk(d_bS.alloc(ee)))
      return 2;
  }

  // Scratch buffers
  if (!chk(d_Kr_g.alloc(ae)) || !chk(d_Kr_sq.alloc(ae)) || !chk(d_Kr_dc.alloc(ae)) ||
      !chk(d_Kr_tmp.alloc(ae)) || !chk(d_Kr_xg_r.alloc(ae)) || !chk(d_Kr_xsq_r.alloc(ae)) ||
      !chk(d_bar_Kr.alloc(ae)) || !chk(d_f_raw.alloc(ee)) || !chk(d_f.alloc(ee)) ||
      !chk(d_qf.alloc(ee)) || !chk(d_fmq.alloc(ee)) || !chk(d_ff_raw.alloc(ee)) ||
      !chk(d_ff.alloc(ee)) || !chk(d_uf_raw.alloc(ee)) || !chk(d_uf.alloc(ee)) ||
      !chk(d_fg_raw.alloc(ee)) || !chk(d_fg.alloc(ee)) || !chk(d_C.alloc(ee)) ||
      !chk(d_chi.alloc(ee)) || !chk(d_fx_raw.alloc(ee)) || !chk(d_fx.alloc(ee)) ||
      !chk(d_ffx_raw.alloc(ee)) || !chk(d_ffx.alloc(ee)) || !chk(d_act_FK_chi.alloc(ee)) ||
      !chk(d_act_K_f.alloc(ee)) || !chk(d_act_F_qf.alloc(ee)) || !chk(d_act_F_f.alloc(ee)) ||
      !chk(d_bar_f.alloc(ee)) || !chk(d_bar_ff.alloc(ee)) || !chk(d_bar_uf.alloc(ee)) ||
      !chk(d_bar_fg.alloc(ee)) || !chk(d_bar_chi.alloc(ee)) || !chk(d_bar_C.alloc(ee)) ||
      !chk(d_bar_tmp.alloc(ee)) || !chk(d_bar_raw.alloc(ee)) || !chk(d_tmp1.alloc(ee)) ||
      !chk(d_tmp2.alloc(ee)) || !chk(d_scalar_acc.alloc(4)))
    return 2;

  // Upload inputs ONCE
  if (!h2d(d_fit_l.ptr, fit_l, Size(nl) * ae) || !h2d(d_rcoul_r.ptr, raw_coul_r, Size(nr) * ae) ||
      !h2d(d_g_l.ptr, raw_g_l, Size(nl) * ae) || !h2d(d_gc_l.ptr, raw_gc_l, Size(nl) * ae) ||
      !h2d(d_sq_l.ptr, raw_sq_l, Size(nl) * ae) || !h2d(d_dc_l.ptr, raw_dc_l, Size(nl) * ae) ||
      !h2d(d_xf_l.ptr, fmix_fit_l, Size(nact_l) * ae) ||
      !h2d(d_xg_l.ptr, fmix_g_l, Size(nact_l) * ae) ||
      !h2d(d_xsq_l.ptr, fmix_sq_l, Size(nact_l) * ae) || !h2d(d_mg.ptr, metric_g, aa) ||
      !h2d(d_mgc.ptr, metric_gc, aa) || !h2d(d_msq.ptr, metric_sq, aa) ||
      !h2d(d_mdc.ptr, metric_dc, aa) || !h2d(d_fl.ptr, fock_l, ee) || !h2d(d_kl.ptr, exch_l, ee) ||
      !zero(d_bfit_l.ptr, Size(nl) * ae) || !zero(d_brcoul_r.ptr, Size(nr) * ae) ||
      !zero(d_bg_l.ptr, Size(nl) * ae) || !zero(d_bgc_l.ptr, Size(nl) * ae) ||
      !zero(d_bsq_l.ptr, Size(nl) * ae) || !zero(d_bdc_l.ptr, Size(nl) * ae) ||
      !zero(d_bxf_l.ptr, Size(nact_l) * ae) || !zero(d_bxg_l.ptr, Size(nact_l) * ae) ||
      !zero(d_bxsq_l.ptr, Size(nact_l) * ae) || !zero(d_bmg.ptr, aa) || !zero(d_bmgc.ptr, aa) ||
      !zero(d_bmsq.ptr, aa) || !zero(d_bmdc.ptr, aa) || !zero(d_bfl.ptr, ee) ||
      !zero(d_bkl.ptr, ee))
    return 3;
  k_scale_add<<<grid_for(ee), 256>>>(ee, 1.0, d_fl.ptr, 1.0, d_kl.ptr, d_FKl.ptr);

  if (!same_spin) {
    if (!h2d(d_fit_r.ptr, fit_r, Size(nr) * ae) || !h2d(d_g_r.ptr, raw_g_r, Size(nr) * ae) ||
        !h2d(d_gc_r.ptr, raw_gc_r, Size(nr) * ae) || !h2d(d_sq_r.ptr, raw_sq_r, Size(nr) * ae) ||
        !h2d(d_dc_r.ptr, raw_dc_r, Size(nr) * ae) ||
        !h2d(d_xf_r.ptr, fmix_fit_r, Size(nact_r) * ae) ||
        !h2d(d_xg_r.ptr, fmix_g_r, Size(nact_r) * ae) ||
        !h2d(d_xsq_r.ptr, fmix_sq_r, Size(nact_r) * ae) || !h2d(d_fr.ptr, fock_r, ee) ||
        !h2d(d_kr.ptr, exch_r, ee) || !h2d(d_S.ptr, cross_overlap, ee) ||
        !zero(d_bfit_r.ptr, Size(nr) * ae) || !zero(d_bg_r.ptr, Size(nr) * ae) ||
        !zero(d_bgc_r.ptr, Size(nr) * ae) || !zero(d_bsq_r.ptr, Size(nr) * ae) ||
        !zero(d_bdc_r.ptr, Size(nr) * ae) || !zero(d_bxf_r.ptr, Size(nact_r) * ae) ||
        !zero(d_bxg_r.ptr, Size(nact_r) * ae) || !zero(d_bxsq_r.ptr, Size(nact_r) * ae) ||
        !zero(d_bfr.ptr, ee) || !zero(d_bkr.ptr, ee) || !zero(d_bS.ptr, ee))
      return 3;
    k_scale_add<<<grid_for(ee), 256>>>(ee, 1.0, d_fr.ptr, 1.0, d_kr.ptr, d_FKr.ptr);
  }

  const double* p_fit_r = same_spin ? d_fit_l.ptr : d_fit_r.ptr;
  const double* p_g_r = same_spin ? d_g_l.ptr : d_g_r.ptr;
  const double* p_gc_r = same_spin ? d_gc_l.ptr : d_gc_r.ptr;
  const double* p_sq_r = same_spin ? d_sq_l.ptr : d_sq_r.ptr;
  const double* p_dc_r = same_spin ? d_dc_l.ptr : d_dc_r.ptr;
  const double* p_xf_r = same_spin ? d_xf_l.ptr : d_xf_r.ptr;
  const double* p_xg_r = same_spin ? d_xg_l.ptr : d_xg_r.ptr;
  const double* p_xsq_r = same_spin ? d_xsq_l.ptr : d_xsq_r.ptr;
  const double* p_fr = same_spin ? d_fl.ptr : d_fr.ptr;
  const double* p_kr = same_spin ? d_kl.ptr : d_kr.ptr;
  const double* p_FKr = same_spin ? d_FKl.ptr : d_FKr.ptr;

  double* p_bfit_r = same_spin ? d_bfit_l.ptr : d_bfit_r.ptr;
  double* p_bg_r = same_spin ? d_bg_l.ptr : d_bg_r.ptr;
  double* p_bgc_r = same_spin ? d_bgc_l.ptr : d_bgc_r.ptr;
  double* p_bsq_r = same_spin ? d_bsq_l.ptr : d_bsq_r.ptr;
  double* p_bdc_r = same_spin ? d_bdc_l.ptr : d_bdc_r.ptr;
  double* p_bxf_r = same_spin ? d_bxf_l.ptr : d_bxf_r.ptr;
  double* p_bxg_r = same_spin ? d_bxg_l.ptr : d_bxg_r.ptr;
  double* p_bxsq_r = same_spin ? d_bxsq_l.ptr : d_bxsq_r.ptr;
  double* p_bfr = same_spin ? d_bfl.ptr : d_bfr.ptr;
  double* p_bkr = same_spin ? d_bkl.ptr : d_bkr.ptr;

  double e_v = 0.0, e_b = 0.0, e_x = 0.0;
  const int ee_int = static_cast<int>(ee);
  const unsigned g_ee = grid_for(ee);

  for (int k = 0; k < n_pairs; ++k) {
    const int i = pairs_i[k], j = pairs_j[k];
    const int ii = i - fl, jj = j - fr;

    const double* Dl = d_fit_l.ptr + Size(i) * ae;
    const double* Dr = p_fit_r + Size(j) * ae;
    const double* Rcr = d_rcoul_r.ptr + Size(j) * ae;
    const double* Ogl = d_g_l.ptr + Size(i) * ae;
    const double* Ogr = p_g_r + Size(j) * ae;
    const double* Ogcl = d_gc_l.ptr + Size(i) * ae;
    const double* Ogcr = p_gc_r + Size(j) * ae;
    const double* Osql = d_sq_l.ptr + Size(i) * ae;
    const double* Osqr = p_sq_r + Size(j) * ae;
    const double* Odcl = d_dc_l.ptr + Size(i) * ae;
    const double* Odcr = p_dc_r + Size(j) * ae;

    const double* XDl = d_xf_l.ptr + Size(ii) * ae;
    const double* XDr = p_xf_r + Size(jj) * ae;
    const double* XOgl = d_xg_l.ptr + Size(ii) * ae;
    const double* XOgr = p_xg_r + Size(jj) * ae;
    const double* XOsql = d_xsq_l.ptr + Size(ii) * ae;
    const double* XOsqr = p_xsq_r + Size(jj) * ae;

    double* bDl = d_bfit_l.ptr + Size(i) * ae;
    double* bDr = p_bfit_r + Size(j) * ae;
    double* bRcr = d_brcoul_r.ptr + Size(j) * ae;
    double* bOgl = d_bg_l.ptr + Size(i) * ae;
    double* bOgr = p_bg_r + Size(j) * ae;
    double* bOgcl = d_bgc_l.ptr + Size(i) * ae;
    double* bOgcr = p_bgc_r + Size(j) * ae;
    double* bOsql = d_bsq_l.ptr + Size(i) * ae;
    double* bOsqr = p_bsq_r + Size(j) * ae;
    double* bOdcl = d_bdc_l.ptr + Size(i) * ae;
    double* bOdcr = p_bdc_r + Size(j) * ae;
    double* bXDl = d_bxf_l.ptr + Size(ii) * ae;
    double* bXDr = p_bxf_r + Size(jj) * ae;
    double* bXOgl = d_bxg_l.ptr + Size(ii) * ae;
    double* bXOgr = p_bxg_r + Size(jj) * ae;
    double* bXOsql = d_bxsq_l.ptr + Size(ii) * ae;
    double* bXOsqr = p_bxsq_r + Size(jj) * ae;

    // 1. Forward f, qf, f_mq, fg, C
    if (!robust_fwd_gpu(h, aux, ext, Dl, Ogl, Dr, Ogr, d_mg.ptr, d_Kr_g.ptr, d_f_raw.ptr)) return 4;
    if (!dress_fwd_gpu(h, same_spin, ext, d_S.ptr, d_f_raw.ptr, d_f.ptr, d_tmp1.ptr)) return 4;
    k_strong_mask<<<g_ee, 256>>>(ext, nl, nr, obs, d_f.ptr, d_qf.ptr, false);
    k_strong_mask<<<g_ee, 256>>>(ext, nl, nr, obs, d_f.ptr, d_fmq.ptr, true);

    if (!robust_fwd_gpu(h, aux, ext, Dl, Ogcl, Dr, Ogcr, d_mgc.ptr, d_Kr_tmp.ptr, d_fg_raw.ptr))
      return 4;
    if (!dress_fwd_gpu(h, same_spin, ext, d_S.ptr, d_fg_raw.ptr, d_fg.ptr, d_tmp1.ptr)) return 4;

    if (rm_gemm(h, true, false, ext, ext, aux, 1.0, Dl, ext, Rcr, ext, 0.0, d_C.ptr, ext) !=
        CUBLAS_STATUS_SUCCESS)
      return 4;

    if (!zero(d_bar_f.ptr, ee) || !zero(d_scalar_acc.ptr, 2)) return 4;
    k_read_point_accum<<<1, 1>>>(d_fg.ptr, ext, i, j, 2.0, d_scalar_acc.ptr);
    k_v_mask_acc<<<std::min<unsigned>(g_ee, 256), 256>>>(
        ext, nl, nr, obs, -2.0 * w_v, d_C.ptr, d_f.ptr, d_bar_C.ptr, d_bar_f.ptr,
        d_scalar_acc.ptr + 1);
    double v_pair[2];
    if (cudaMemcpy(v_pair, d_scalar_acc.ptr, 2 * sizeof(double), cudaMemcpyDeviceToHost) !=
        cudaSuccess)
      return 4;
    e_v += v_pair[0] - 2.0 * v_pair[1];

    if (rm_gemm(h, false, true, aux, ext, ext, 1.0, Rcr, ext, d_bar_C.ptr, ext, 1.0, bDl, ext) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, false, aux, ext, ext, 1.0, Dl, ext, d_bar_C.ptr, ext, 1.0, bRcr, ext) !=
            CUBLAS_STATUS_SUCCESS)
      return 4;

    if (!zero(d_bar_fg.ptr, ee)) return 4;
    k_set_point<<<1, 1>>>(d_bar_fg.ptr, ext, i, j, 2.0 * w_v, false);
    if (!dress_adj_gpu(h, same_spin, ext, d_S.ptr, d_fg_raw.ptr, d_bar_fg.ptr, d_bar_raw.ptr,
                       d_bS.ptr, d_tmp1.ptr, d_tmp2.ptr) ||
        !robust_adj_gpu(h, aux, ext, Dl, Ogcl, Dr, d_mgc.ptr, d_Kr_tmp.ptr, d_bar_raw.ptr, bDl,
                        bOgcl, bDr, bOgcr, d_bmgc.ptr, d_bar_Kr.ptr))
      return 4;

    // 2. Quadratic B forward
    if (same_spin)
      k_set_chi_same_spin<<<g_ee, 256>>>(ext, i, j, d_chi.ptr);
    else
      k_set_chi_opp_spin<<<g_ee, 256>>>(ext, i, j, d_S.ptr, d_chi.ptr);

    if (!robust_fwd_gpu(h, aux, ext, Dl, Osql, Dr, Osqr, d_msq.ptr, d_Kr_sq.ptr, d_ff_raw.ptr) ||
        !dress_fwd_gpu(h, same_spin, ext, d_S.ptr, d_ff_raw.ptr, d_ff.ptr, d_tmp1.ptr) ||
        !robust_fwd_gpu(h, aux, ext, Dl, Odcl, Dr, Odcr, d_mdc.ptr, d_Kr_dc.ptr, d_uf_raw.ptr) ||
        !dress_fwd_gpu(h, same_spin, ext, d_S.ptr, d_uf_raw.ptr, d_uf.ptr, d_tmp1.ptr))
      return 5;

    if (rm_gemm(h, false, false, ext, ext, ext, 1.0, d_FKl.ptr, ext, d_chi.ptr, ext, 0.0,
                d_act_FK_chi.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, true, ext, ext, ext, 1.0, d_chi.ptr, ext, p_FKr, ext, 1.0,
                d_act_FK_chi.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, false, ext, ext, ext, 1.0, d_kl.ptr, ext, d_f.ptr, ext, 0.0,
                d_act_K_f.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, true, ext, ext, ext, 1.0, d_f.ptr, ext, p_kr, ext, 1.0, d_act_K_f.ptr,
                ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, false, ext, ext, ext, 1.0, d_fl.ptr, ext, d_qf.ptr, ext, 0.0,
                d_act_F_qf.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, true, ext, ext, ext, 1.0, d_qf.ptr, ext, p_fr, ext, 1.0, d_act_F_qf.ptr,
                ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, false, ext, ext, ext, 1.0, d_fl.ptr, ext, d_f.ptr, ext, 0.0,
                d_act_F_f.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, true, ext, ext, ext, 1.0, d_f.ptr, ext, p_fr, ext, 1.0, d_act_F_f.ptr,
                ext) != CUBLAS_STATUS_SUCCESS)
      return 5;

    double d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0;
    if (cublasDdot(h, ee_int, d_chi.ptr, 1, d_uf.ptr, 1, &d1) != CUBLAS_STATUS_SUCCESS ||
        cublasDdot(h, ee_int, d_ff.ptr, 1, d_act_FK_chi.ptr, 1, &d2) != CUBLAS_STATUS_SUCCESS ||
        cublasDdot(h, ee_int, d_f.ptr, 1, d_act_K_f.ptr, 1, &d3) != CUBLAS_STATUS_SUCCESS ||
        cublasDdot(h, ee_int, d_qf.ptr, 1, d_act_F_qf.ptr, 1, &d4) != CUBLAS_STATUS_SUCCESS ||
        cublasDdot(h, ee_int, d_f.ptr, 1, d_act_F_f.ptr, 1, &d5) != CUBLAS_STATUS_SUCCESS)
      return 5;
    e_b += norm * (d1 + d2 - d3 + d4 - d5);

    // Quadratic B adjoints
    k_scale_add<<<g_ee, 256>>>(ee, w_b, d_chi.ptr, 0.0, d_chi.ptr, d_bar_uf.ptr);
    k_scale_add<<<g_ee, 256>>>(ee, w_b, d_uf.ptr, 0.0, d_uf.ptr, d_bar_chi.ptr);
    k_scale_add<<<g_ee, 256>>>(ee, w_b, d_act_FK_chi.ptr, 0.0, d_act_FK_chi.ptr, d_bar_ff.ptr);

    if (rm_gemm(h, false, true, ext, ext, ext, w_b, d_ff.ptr, ext, d_chi.ptr, ext, 0.0, d_tmp1.ptr,
                ext) != CUBLAS_STATUS_SUCCESS)
      return 6;
    k_axpy<<<g_ee, 256>>>(ee, 1.0, d_tmp1.ptr, d_bfl.ptr);
    k_axpy<<<g_ee, 256>>>(ee, 1.0, d_tmp1.ptr, d_bkl.ptr);
    if (rm_gemm(h, true, false, ext, ext, ext, w_b, d_ff.ptr, ext, d_chi.ptr, ext, 0.0, d_tmp1.ptr,
                ext) != CUBLAS_STATUS_SUCCESS)
      return 6;
    k_axpy<<<g_ee, 256>>>(ee, 1.0, d_tmp1.ptr, p_bfr);
    k_axpy<<<g_ee, 256>>>(ee, 1.0, d_tmp1.ptr, p_bkr);

    if (rm_gemm(h, true, false, ext, ext, ext, w_b, d_FKl.ptr, ext, d_ff.ptr, ext, 1.0,
                d_bar_chi.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, false, ext, ext, ext, w_b, d_ff.ptr, ext, p_FKr, ext, 1.0, d_bar_chi.ptr,
                ext) != CUBLAS_STATUS_SUCCESS)
      return 6;

    // Term 3: -w_b * dot(f, K_l @ f + f @ K_r^T)
    if (rm_gemm(h, false, true, ext, ext, ext, -w_b, d_f.ptr, ext, d_f.ptr, ext, 1.0, d_bkl.ptr,
                ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, true, false, ext, ext, ext, -w_b, d_f.ptr, ext, d_f.ptr, ext, 1.0, p_bkr, ext) !=
            CUBLAS_STATUS_SUCCESS)
      return 6;
    k_axpy<<<g_ee, 256>>>(ee, -w_b, d_act_K_f.ptr, d_bar_f.ptr);
    if (rm_gemm(h, true, false, ext, ext, ext, -w_b, d_kl.ptr, ext, d_f.ptr, ext, 1.0, d_bar_f.ptr,
                ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, false, ext, ext, ext, -w_b, d_f.ptr, ext, p_kr, ext, 1.0, d_bar_f.ptr,
                ext) != CUBLAS_STATUS_SUCCESS)
      return 6;

    // Term 4: +w_b * dot(qf, F_l @ qf + qf @ F_r^T)
    if (rm_gemm(h, false, true, ext, ext, ext, w_b, d_qf.ptr, ext, d_qf.ptr, ext, 1.0, d_bfl.ptr,
                ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, true, false, ext, ext, ext, w_b, d_qf.ptr, ext, d_qf.ptr, ext, 1.0, p_bfr,
                ext) != CUBLAS_STATUS_SUCCESS)
      return 6;
    k_scale_add<<<g_ee, 256>>>(ee, w_b, d_act_F_qf.ptr, 0.0, d_act_F_qf.ptr, d_bar_tmp.ptr);
    if (rm_gemm(h, true, false, ext, ext, ext, w_b, d_fl.ptr, ext, d_qf.ptr, ext, 1.0,
                d_bar_tmp.ptr, ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, false, ext, ext, ext, w_b, d_qf.ptr, ext, p_fr, ext, 1.0, d_bar_tmp.ptr,
                ext) != CUBLAS_STATUS_SUCCESS)
      return 6;
    k_add_strong_masked<<<g_ee, 256>>>(ext, nl, nr, obs, 1.0, d_bar_tmp.ptr, d_bar_f.ptr, false);

    // Term 5: -w_b * dot(f, F_l @ f + f @ F_r^T)
    if (rm_gemm(h, false, true, ext, ext, ext, -w_b, d_f.ptr, ext, d_f.ptr, ext, 1.0, d_bfl.ptr,
                ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, true, false, ext, ext, ext, -w_b, d_f.ptr, ext, d_f.ptr, ext, 1.0, p_bfr, ext) !=
            CUBLAS_STATUS_SUCCESS)
      return 6;
    k_axpy<<<g_ee, 256>>>(ee, -w_b, d_act_F_f.ptr, d_bar_f.ptr);
    if (rm_gemm(h, true, false, ext, ext, ext, -w_b, d_fl.ptr, ext, d_f.ptr, ext, 1.0, d_bar_f.ptr,
                ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, false, false, ext, ext, ext, -w_b, d_f.ptr, ext, p_fr, ext, 1.0, d_bar_f.ptr,
                ext) != CUBLAS_STATUS_SUCCESS)
      return 6;

    // 3. Quadratic X forward & adjoint
    if (!robust_fwd_gpu(h, aux, ext, XDl, XOgl, Dr, Ogr, d_mg.ptr, d_Kr_tmp.ptr, d_fx_raw.ptr) ||
        !robust_fwd_gpu(h, aux, ext, Dl, Ogl, XDr, XOgr, d_mg.ptr, d_Kr_xg_r.ptr, d_tmp1.ptr))
      return 7;
    k_axpy<<<g_ee, 256>>>(ee, 1.0, d_tmp1.ptr, d_fx_raw.ptr);
    if (!dress_fwd_gpu(h, same_spin, ext, d_S.ptr, d_fx_raw.ptr, d_fx.ptr, d_tmp1.ptr)) return 7;

    if (!robust_fwd_gpu(h, aux, ext, XDl, XOsql, Dr, Osqr, d_msq.ptr, d_Kr_tmp.ptr,
                        d_ffx_raw.ptr) ||
        !robust_fwd_gpu(h, aux, ext, Dl, Osql, XDr, XOsqr, d_msq.ptr, d_Kr_xsq_r.ptr, d_tmp1.ptr))
      return 7;
    k_axpy<<<g_ee, 256>>>(ee, 1.0, d_tmp1.ptr, d_ffx_raw.ptr);
    if (!dress_fwd_gpu(h, same_spin, ext, d_S.ptr, d_ffx_raw.ptr, d_ffx.ptr, d_tmp1.ptr)) return 7;

    double x1 = 0, x2 = 0;
    if (cublasDdot(h, ee_int, d_chi.ptr, 1, d_ffx.ptr, 1, &x1) != CUBLAS_STATUS_SUCCESS ||
        cublasDdot(h, ee_int, d_fmq.ptr, 1, d_fx.ptr, 1, &x2) != CUBLAS_STATUS_SUCCESS)
      return 7;
    e_x += norm * (x1 - x2);

    k_axpy<<<g_ee, 256>>>(ee, w_x, d_ffx.ptr, d_bar_chi.ptr);
    k_scale_add<<<g_ee, 256>>>(ee, w_x, d_chi.ptr, 0.0, d_chi.ptr, d_bar_tmp.ptr);
    if (!dress_adj_gpu(h, same_spin, ext, d_S.ptr, d_ffx_raw.ptr, d_bar_tmp.ptr, d_bar_raw.ptr,
                       d_bS.ptr, d_tmp1.ptr, d_tmp2.ptr) ||
        !robust_adj_gpu(h, aux, ext, XDl, XOsql, Dr, d_msq.ptr, d_Kr_sq.ptr, d_bar_raw.ptr, bXDl,
                        bXOsql, bDr, bOsqr, d_bmsq.ptr, d_bar_Kr.ptr) ||
        !robust_adj_gpu(h, aux, ext, Dl, Osql, XDr, d_msq.ptr, d_Kr_xsq_r.ptr, d_bar_raw.ptr, bDl,
                        bOsql, bXDr, bXOsqr, d_bmsq.ptr, d_bar_Kr.ptr))
      return 8;

    k_add_strong_masked<<<g_ee, 256>>>(ext, nl, nr, obs, -w_x, d_fx.ptr, d_bar_f.ptr, true);
    k_scale_add<<<g_ee, 256>>>(ee, -w_x, d_fmq.ptr, 0.0, d_fmq.ptr, d_bar_tmp.ptr);
    if (!dress_adj_gpu(h, same_spin, ext, d_S.ptr, d_fx_raw.ptr, d_bar_tmp.ptr, d_bar_raw.ptr,
                       d_bS.ptr, d_tmp1.ptr, d_tmp2.ptr) ||
        !robust_adj_gpu(h, aux, ext, XDl, XOgl, Dr, d_mg.ptr, d_Kr_g.ptr, d_bar_raw.ptr, bXDl,
                        bXOgl, bDr, bOgr, d_bmg.ptr, d_bar_Kr.ptr) ||
        !robust_adj_gpu(h, aux, ext, Dl, Ogl, XDr, d_mg.ptr, d_Kr_xg_r.ptr, d_bar_raw.ptr, bDl,
                        bOgl, bXDr, bXOgr, d_bmg.ptr, d_bar_Kr.ptr))
      return 8;

    // 4. Propagate bar_chi -> bar_S for opposite spin
    if (!same_spin) {
      k_bar_chi_to_bar_S<<<grid_for(ext), 256>>>(ext, i, j, d_bar_chi.ptr, d_S.ptr, d_bS.ptr);
    }

    // 5. Propagate bar_f, bar_ff, bar_uf back through dress_adj + robust_adj
    if (!dress_adj_gpu(h, same_spin, ext, d_S.ptr, d_f_raw.ptr, d_bar_f.ptr, d_bar_raw.ptr,
                       d_bS.ptr, d_tmp1.ptr, d_tmp2.ptr) ||
        !robust_adj_gpu(h, aux, ext, Dl, Ogl, Dr, d_mg.ptr, d_Kr_g.ptr, d_bar_raw.ptr, bDl, bOgl,
                        bDr, bOgr, d_bmg.ptr, d_bar_Kr.ptr) ||
        !dress_adj_gpu(h, same_spin, ext, d_S.ptr, d_ff_raw.ptr, d_bar_ff.ptr, d_bar_raw.ptr,
                       d_bS.ptr, d_tmp1.ptr, d_tmp2.ptr) ||
        !robust_adj_gpu(h, aux, ext, Dl, Osql, Dr, d_msq.ptr, d_Kr_sq.ptr, d_bar_raw.ptr, bDl,
                        bOsql, bDr, bOsqr, d_bmsq.ptr, d_bar_Kr.ptr) ||
        !dress_adj_gpu(h, same_spin, ext, d_S.ptr, d_uf_raw.ptr, d_bar_uf.ptr, d_bar_raw.ptr,
                       d_bS.ptr, d_tmp1.ptr, d_tmp2.ptr) ||
        !robust_adj_gpu(h, aux, ext, Dl, Odcl, Dr, d_mdc.ptr, d_Kr_dc.ptr, d_bar_raw.ptr, bDl,
                        bOdcl, bDr, bOdcr, d_bmdc.ptr, d_bar_Kr.ptr))
      return 9;
  }

  if (cudaDeviceSynchronize() != cudaSuccess) return 10;
  std::vector<double> s_bfl, s_brcr, s_bgl, s_bgcl, s_bsql, s_bdcl;
  std::vector<double> s_bxfl, s_bxgl, s_bxsql, s_bmg, s_bmgc, s_bmsq, s_bmdc, s_bfock_l, s_bexch_l;
  std::vector<double> s_bfr, s_bgr, s_bgcr, s_bsqr, s_bdcr, s_bxfr, s_bxgr, s_bxsqr;
  std::vector<double> s_bfock_r, s_bexch_r, s_bS;
  if (!d2h_staged(d_bfit_l.ptr, nlae, s_bfl) || !d2h_staged(d_brcoul_r.ptr, nrae, s_brcr) ||
      !d2h_staged(d_bg_l.ptr, nlae, s_bgl) || !d2h_staged(d_bgc_l.ptr, nlae, s_bgcl) ||
      !d2h_staged(d_bsq_l.ptr, nlae, s_bsql) || !d2h_staged(d_bdc_l.ptr, nlae, s_bdcl) ||
      !d2h_staged(d_bxf_l.ptr, nacl_ae, s_bxfl) || !d2h_staged(d_bxg_l.ptr, nacl_ae, s_bxgl) ||
      !d2h_staged(d_bxsq_l.ptr, nacl_ae, s_bxsql) || !d2h_staged(d_bmg.ptr, aa, s_bmg) ||
      !d2h_staged(d_bmgc.ptr, aa, s_bmgc) || !d2h_staged(d_bmsq.ptr, aa, s_bmsq) ||
      !d2h_staged(d_bmdc.ptr, aa, s_bmdc) || !d2h_staged(d_bfl.ptr, ee, s_bfock_l) ||
      !d2h_staged(d_bkl.ptr, ee, s_bexch_l))
    return 11;
  if (!same_spin) {
    if (!d2h_staged(d_bfit_r.ptr, nrae, s_bfr) || !d2h_staged(d_bg_r.ptr, nrae, s_bgr) ||
        !d2h_staged(d_bgc_r.ptr, nrae, s_bgcr) || !d2h_staged(d_bsq_r.ptr, nrae, s_bsqr) ||
        !d2h_staged(d_bdc_r.ptr, nrae, s_bdcr) || !d2h_staged(d_bxf_r.ptr, nacr_ae, s_bxfr) ||
        !d2h_staged(d_bxg_r.ptr, nacr_ae, s_bxgr) || !d2h_staged(d_bxsq_r.ptr, nacr_ae, s_bxsqr) ||
        !d2h_staged(d_bfr.ptr, ee, s_bfock_r) || !d2h_staged(d_bkr.ptr, ee, s_bexch_r) ||
        !d2h_staged(d_bS.ptr, ee, s_bS))
      return 11;
  }

  if (!std::isfinite(e_v) || !std::isfinite(e_b) || !std::isfinite(e_x) ||
      !all_finite(s_bfl.data(), nlae) || !all_finite(s_brcr.data(), nrae) ||
      !all_finite(s_bgl.data(), nlae) || !all_finite(s_bgcl.data(), nlae) ||
      !all_finite(s_bsql.data(), nlae) || !all_finite(s_bdcl.data(), nlae) ||
      !all_finite(s_bxfl.data(), nacl_ae) || !all_finite(s_bxgl.data(), nacl_ae) ||
      !all_finite(s_bxsql.data(), nacl_ae) || !all_finite(s_bmg.data(), aa) ||
      !all_finite(s_bmgc.data(), aa) || !all_finite(s_bmsq.data(), aa) ||
      !all_finite(s_bmdc.data(), aa) || !all_finite(s_bfock_l.data(), ee) ||
      !all_finite(s_bexch_l.data(), ee) ||
      (!same_spin &&
       (!all_finite(s_bfr.data(), nrae) || !all_finite(s_bgr.data(), nrae) ||
        !all_finite(s_bgcr.data(), nrae) || !all_finite(s_bsqr.data(), nrae) ||
        !all_finite(s_bdcr.data(), nrae) || !all_finite(s_bxfr.data(), nacr_ae) ||
        !all_finite(s_bxgr.data(), nacr_ae) || !all_finite(s_bxsqr.data(), nacr_ae) ||
        !all_finite(s_bfock_r.data(), ee) || !all_finite(s_bexch_r.data(), ee) ||
        !all_finite(s_bS.data(), ee))))
    return kErrNonFiniteOutput;

  add_inplace(bar_fit_l, s_bfl.data(), nlae);
  add_inplace(bar_raw_coul_r, s_brcr.data(), nrae);
  add_inplace(bar_raw_g_l, s_bgl.data(), nlae);
  add_inplace(bar_raw_gc_l, s_bgcl.data(), nlae);
  add_inplace(bar_raw_sq_l, s_bsql.data(), nlae);
  add_inplace(bar_raw_dc_l, s_bdcl.data(), nlae);
  add_inplace(bar_fmix_fit_l, s_bxfl.data(), nacl_ae);
  add_inplace(bar_fmix_g_l, s_bxgl.data(), nacl_ae);
  add_inplace(bar_fmix_sq_l, s_bxsql.data(), nacl_ae);
  add_inplace(bar_metric_g, s_bmg.data(), aa);
  add_inplace(bar_metric_gc, s_bmgc.data(), aa);
  add_inplace(bar_metric_sq, s_bmsq.data(), aa);
  add_inplace(bar_metric_dc, s_bmdc.data(), aa);
  add_inplace(bar_fock_l, s_bfock_l.data(), ee);
  add_inplace(bar_exch_l, s_bexch_l.data(), ee);
  if (!same_spin) {
    add_inplace(bar_fit_r, s_bfr.data(), nrae);
    add_inplace(bar_raw_g_r, s_bgr.data(), nrae);
    add_inplace(bar_raw_gc_r, s_bgcr.data(), nrae);
    add_inplace(bar_raw_sq_r, s_bsqr.data(), nrae);
    add_inplace(bar_raw_dc_r, s_bdcr.data(), nrae);
    add_inplace(bar_fmix_fit_r, s_bxfr.data(), nacr_ae);
    add_inplace(bar_fmix_g_r, s_bxgr.data(), nacr_ae);
    add_inplace(bar_fmix_sq_r, s_bxsqr.data(), nacr_ae);
    add_inplace(bar_fock_r, s_bfock_r.data(), ee);
    add_inplace(bar_exch_r, s_bexch_r.data(), ee);
    add_inplace(bar_cross_overlap, s_bS.data(), ee);
  }
  energies_out[0] = e_v;
  energies_out[1] = e_b;
  energies_out[2] = e_x;
  return 0;
#endif
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (const std::exception&) {
    return kErrInternal;
  } catch (...) {
    return kErrInternal;
  }
}
