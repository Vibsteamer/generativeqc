// Dynamic-dimension AOT cuBLAS (and matching FP64 host reference) kernels for
// F12 AO->MO forward/reverse three-center projections, 2D bilinear transforms
// (cross_overlap, hcore), and full-rank Coulomb metric fit forward/reverse.
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

constexpr int kStatusOk = 0;
constexpr int kStatusInvalidArg = -1;
constexpr int kStatusNonFinite = -2;
constexpr int kStatusBudgetExceeded = -3;
constexpr int kStatusCudaError = -4;
constexpr int kStatusCublasError = -5;
constexpr int kStatusOom = -6;
constexpr int kStatusInternalError = -7;

// Target workspace per call (1 GiB) and hard upper cap (4 GiB << 36 GiB L20 limit).
// For CNa3O3 ha-Q (nc ~ 750, ext ~ 750, aux ~ 700, occ ~ 26):
//   - project_fwd worst case (left_dim=750): 4.5 MB coeffs + B * 13.5 MB <= 1.01 GiB (B=75)
//   - project_rev worst case (mode=both, active_rows=750, tc_rows=26):
//     9.0 MB coeffs/bar_coeffs + B * 18.16 MB <= 1.01 GiB (B=55, d_ao_tile reused for d_bar_ao)
//   - coulomb_fit_rev (aux=700, pairs=19500): 7.8 MB W/weight + 327 MB <= 0.34 GiB (single chunk)
constexpr Size kTargetDeviceWorkspaceBytes = Size(1024) << 20;
constexpr Size kMaxDeviceWorkspaceBytes = Size(4096) << 20;
constexpr Size kFitWorkspaceBytes = Size(512) << 20;

inline bool is_finite_buffer(const double* ptr, Size n) {
  if (n == 0) return true;
  if (ptr == nullptr) return false;
  for (Size i = 0; i < n; ++i) {
    if (!std::isfinite(ptr[i])) return false;
  }
  return true;
}

#ifdef GQC_SINK_HOST_REFERENCE

// Cache-friendly row-major FP64 GEMM: C = alpha * op(A) * op(B) + beta * C.
void rm_gemm_host(bool trans_a, bool trans_b, int M, int N, int K, double alpha,
                  const double* A, int lda, const double* B, int ldb,
                  double beta, double* C, int ldc) {
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
        for (int j = 0; j < N; ++j) {
          c_row[j] += aik * b_row[j];
        }
      }
    }
  } else if (trans_a && !trans_b) {
    for (int k = 0; k < K; ++k) {
      const double* a_row = A + Size(k) * lda;
      const double* b_row = B + Size(k) * ldb;
      for (int i = 0; i < M; ++i) {
        const double aik = alpha * a_row[i];
        if (aik == 0.0) continue;
        double* c_row = C + Size(i) * ldc;
        for (int j = 0; j < N; ++j) {
          c_row[j] += aik * b_row[j];
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
  } else {
    // trans_a && trans_b: A is K x M, B is N x K
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

void build_effective_seed_slice(int active_rows, int ext, int seed_rows,
                                int tc_rows, double seed_scale,
                                const double* seed_q, const double* tc_q,
                                double* s_buf) {
  const Size total = Size(active_rows) * ext;
  std::fill_n(s_buf, total, 0.0);
  if (seed_rows > 0 && seed_q != nullptr) {
    std::copy_n(seed_q, Size(seed_rows) * ext, s_buf);
  }
  if (tc_rows > 0 && tc_q != nullptr) {
    const Size tc_n = Size(tc_rows) * ext;
    for (Size idx = 0; idx < tc_n; ++idx) {
      s_buf[idx] += tc_q[idx];
    }
  }
  if (seed_scale != 1.0) {
    for (Size idx = 0; idx < total; ++idx) {
      s_buf[idx] *= seed_scale;
    }
  }
}

#else  // CUDA + cuBLAS implementation

struct DeviceBuf {
  double* ptr = nullptr;
  Size count = 0;

  DeviceBuf() = default;
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
  ~DeviceBuf() { reset(); }

  void reset() {
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
      count = 0;
    }
  }

  cudaError_t alloc(Size n) {
    reset();
    if (n == 0) return cudaSuccess;
    const Size bytes = n * sizeof(double);
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
      ptr = nullptr;
      count = 0;
      return err;
    }
    count = n;
    return cudaSuccess;
  }
};

struct CudaStreamGuard {
  cudaStream_t stream = nullptr;

  CudaStreamGuard() = default;
  CudaStreamGuard(const CudaStreamGuard&) = delete;
  CudaStreamGuard& operator=(const CudaStreamGuard&) = delete;
  ~CudaStreamGuard() {
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }

  cudaError_t init() {
    return cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
};

struct CublasGuard {
  cublasHandle_t handle = nullptr;

  CublasGuard() = default;
  CublasGuard(const CublasGuard&) = delete;
  CublasGuard& operator=(const CublasGuard&) = delete;
  ~CublasGuard() {
    if (handle != nullptr) {
      cublasDestroy(handle);
    }
  }

  cublasStatus_t init(cudaStream_t stream) {
    cublasStatus_t st = cublasCreate(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    st = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    return cublasSetStream(handle, stream);
  }
};

// Row-major FP64 GEMM via cuBLAS: C[M, N] = alpha * op(A) * op(B) + beta * C[M, N].
inline cublasStatus_t rm_gemm_cuda(cublasHandle_t handle, bool trans_a,
                                   bool trans_b, int M, int N, int K,
                                   double alpha, const double* A, int lda,
                                   const double* B, int ldb, double beta,
                                   double* C, int ldc) {
  const cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  return cublasDgemm(handle, op_b, op_a, N, M, K, &alpha, B, ldb, A, lda,
                     &beta, C, ldc);
}

inline cublasStatus_t rm_gemm_strided_batched_cuda(
    cublasHandle_t handle, bool trans_a, bool trans_b, int M, int N, int K,
    double alpha, const double* A, int lda, long long stride_a,
    const double* B, int ldb, long long stride_b, double beta, double* C,
    int ldc, long long stride_c, int batch_count) {
  const cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  return cublasDgemmStridedBatched(handle, op_b, op_a, N, M, K, &alpha, B, ldb,
                                   stride_b, A, lda, stride_a, &beta, C, ldc,
                                   stride_c, batch_count);
}

__global__ void scale_buffer_kernel(double* __restrict__ buf, Size n,
                                    double scale) {
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += Size(blockDim.x) * gridDim.x) {
    buf[idx] *= scale;
  }
}

__global__ void add_tc_and_scale_kernel(double* __restrict__ s_tile,
                                        const double* __restrict__ tc_tile,
                                        int b, int active_rows, int tc_rows,
                                        int ext, double seed_scale) {
  const Size per_q = Size(active_rows) * ext;
  const Size tc_per_q = Size(tc_rows) * ext;
  const Size total = Size(b) * per_q;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(blockDim.x) * gridDim.x) {
    const Size q = idx / per_q;
    const Size rem = idx - q * per_q;
    const int p = static_cast<int>(rem / ext);
    const int r = static_cast<int>(rem - Size(p) * ext);
    double v = s_tile[idx];
    if (p < tc_rows) {
      v += tc_tile[q * tc_per_q + Size(p) * ext + r];
    }
    s_tile[idx] = v * seed_scale;
  }
}

// Deterministic Q-reduction: bar_coeffs[u, p] += sum_{q=0}^{b-1} delta_q[q, u, p]
// where delta_q has shape [b, nc, active_rows] and bar_coeffs has shape [nc, ext].
__global__ void reduce_q_add_strided_kernel(
    const double* __restrict__ delta_q, double* __restrict__ bar_coeffs, int b,
    int nc, int active_rows, int ext) {
  const Size total = Size(nc) * active_rows;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(blockDim.x) * gridDim.x) {
    const int u = static_cast<int>(idx / active_rows);
    const int p = static_cast<int>(idx - Size(u) * active_rows);
    double acc = 0.0;
    for (int q = 0; q < b; ++q) {
      acc += delta_q[Size(q) * total + idx];
    }
    bar_coeffs[Size(u) * ext + p] += acc;
  }
}

inline int grid_blocks(Size total, int threads = 256) {
  if (total == 0) return 0;
  const Size raw = (total + Size(threads) - 1) / Size(threads);
  return static_cast<int>(std::min<Size>(raw, Size(65535)));
}

// Compute safe Q-chunk width satisfying both the target/max VRAM caps and 32-bit
// cuBLAS flattened dimension limits (b * rows_per_q <= INT_MAX).
inline int choose_q_chunk(int aux, int rows_per_q, Size fixed_bytes,
                          Size per_q_bytes, int q_chunk_hint) {
  if (fixed_bytes + per_q_bytes > kMaxDeviceWorkspaceBytes) {
    return 0;  // Even 1 slice exceeds hard VRAM cap.
  }
  Size avail = kTargetDeviceWorkspaceBytes > fixed_bytes
                   ? (kTargetDeviceWorkspaceBytes - fixed_bytes)
                   : 0;
  Size cap_by_mem = per_q_bytes > 0 ? std::max<Size>(1, avail / per_q_bytes)
                                    : Size(aux);
  const Size max_int_q =
      rows_per_q > 0
          ? Size(std::numeric_limits<int>::max() / rows_per_q)
          : Size(aux);
  Size q_chunk = std::min<Size>({Size(aux), cap_by_mem, max_int_q});
  if (q_chunk_hint > 0) {
    q_chunk = std::min<Size>(q_chunk, Size(q_chunk_hint));
  }
  return static_cast<int>(std::max<Size>(1, q_chunk));
}

// ============================================================================
// CABS Singles E+F CUDA helper kernels (deterministic, zero atomicAdd)
// ============================================================================

__global__ void transpose_bno_to_bon_kernel(const double* __restrict__ src,
                                            double* __restrict__ dst, int b,
                                            int n, int o) {
  const Size per_q = Size(n) * o;
  const Size total = Size(b) * per_q;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(blockDim.x) * gridDim.x) {
    const Size q = idx / per_q;
    const Size rem = idx - q * per_q;
    const int m = static_cast<int>(rem / o);
    const int i = static_cast<int>(rem - Size(m) * o);
    dst[q * per_q + Size(i) * n + m] = src[idx];
  }
}

__global__ void transpose_bon_to_bno_kernel(const double* __restrict__ src,
                                            double* __restrict__ dst, int b,
                                            int o, int n) {
  const Size per_q = Size(o) * n;
  const Size total = Size(b) * per_q;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(blockDim.x) * gridDim.x) {
    const Size q = idx / per_q;
    const Size rem = idx - q * per_q;
    const int i = static_cast<int>(rem / n);
    const int m = static_cast<int>(rem - Size(i) * n);
    dst[q * per_q + Size(m) * o + i] = src[idx];
  }
}

// One block of 256 threads per auxiliary index a in [0, q):
// rho[a] = (accumulate ? rho[a] : 0) + scale * sum_{i<o, m<occ_rows} fit[a,i,m] * occ[m,i]
__global__ void cabs_rho_from_fit_kernel(const double* __restrict__ fit,
                                         const double* __restrict__ occ,
                                         double* __restrict__ rho, int q, int o,
                                         int n, int occ_rows, double scale,
                                         int accumulate) {
  __shared__ double smem[256];
  const int a = blockIdx.x;
  if (a >= q) return;
  const Size on = Size(o) * n;
  const Size pairs = Size(o) * occ_rows;
  const double* fit_a = fit + Size(a) * on;
  double acc = 0.0;
  for (Size idx = threadIdx.x; idx < pairs; idx += blockDim.x) {
    const int i = static_cast<int>(idx / occ_rows);
    const int m = static_cast<int>(idx - Size(i) * occ_rows);
    acc += fit_a[Size(i) * n + m] * occ[Size(m) * o + i];
  }
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (accumulate) {
      rho[a] += scale * smem[0];
    } else {
      rho[a] = scale * smem[0];
    }
  }
}

__global__ void cabs_fock_combine_inplace_kernel(const double* __restrict__ h,
                                                 const double* __restrict__ j,
                                                 double* __restrict__ k_inout,
                                                 Size nn) {
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < nn;
       idx += Size(blockDim.x) * gridDim.x) {
    k_inout[idx] = h[idx] + j[idx] - k_inout[idx];
  }
}

__global__ void symmetrize_copy_kernel(const double* __restrict__ src,
                                       double* __restrict__ dst, int n) {
  const Size nn = Size(n) * n;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < nn;
       idx += Size(blockDim.x) * gridDim.x) {
    const int i = static_cast<int>(idx / n);
    const int j = static_cast<int>(idx - Size(i) * n);
    dst[idx] = 0.5 * (src[idx] + src[Size(j) * n + i]);
  }
}

__global__ void add_transpose_kernel(const double* __restrict__ src,
                                     double* __restrict__ dst, int n) {
  const Size nn = Size(n) * n;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < nn;
       idx += Size(blockDim.x) * gridDim.x) {
    const int i = static_cast<int>(idx / n);
    const int j = static_cast<int>(idx - Size(i) * n);
    dst[idx] = src[idx] + src[Size(j) * n + i];
  }
}

__global__ void add_sym_scaled_kernel(const double* __restrict__ src,
                                      double* __restrict__ dst, int n,
                                      double scale) {
  const Size nn = Size(n) * n;
  const double half_s = 0.5 * scale;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < nn;
       idx += Size(blockDim.x) * gridDim.x) {
    const int i = static_cast<int>(idx / n);
    const int j = static_cast<int>(idx - Size(i) * n);
    dst[idx] += half_s * (src[idx] + src[Size(j) * n + i]);
  }
}

__global__ void add_scaled_pair_kernel(const double* __restrict__ a,
                                       const double* __restrict__ b,
                                       double* __restrict__ out, Size n,
                                       double sa, double sb) {
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += Size(blockDim.x) * gridDim.x) {
    out[idx] = sa * a[idx] + (b != nullptr ? sb * b[idx] : 0.0);
  }
}

__global__ void cabs_bar_occ_from_rho_kernel(const double* __restrict__ bar_rho,
                                             const double* __restrict__ fit,
                                             double* __restrict__ bar_occ,
                                             int q, int o, int n,
                                             int occ_rows) {
  const Size total = Size(occ_rows) * o;
  const Size on = Size(o) * n;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(blockDim.x) * gridDim.x) {
    const int m = static_cast<int>(idx / o);
    const int i = static_cast<int>(idx - Size(m) * o);
    double acc = 0.0;
    for (int a = 0; a < q; ++a) {
      acc += bar_rho[a] * fit[Size(a) * on + Size(i) * n + m];
    }
    bar_occ[idx] += acc;
  }
}

__global__ void cabs_add_bar_rho_occ_to_bar_d_kernel(
    double* __restrict__ bar_d, const double* __restrict__ bar_rho,
    const double* __restrict__ occ, int q, int o, int n, int occ_rows) {
  const Size per_q = Size(o) * occ_rows;
  const Size total = Size(q) * per_q;
  const Size on = Size(o) * n;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(blockDim.x) * gridDim.x) {
    const Size a = idx / per_q;
    const Size rem = idx - a * per_q;
    const int i = static_cast<int>(rem / occ_rows);
    const int m = static_cast<int>(rem - Size(i) * occ_rows);
    bar_d[a * on + Size(i) * n + m] += bar_rho[a] * occ[Size(m) * o + i];
  }
}

__global__ void cabs_init_raw_bar_j_kernel(double* __restrict__ raw_bar_tile,
                                           const double* __restrict__ rho_tile,
                                           const double* __restrict__ bar_g,
                                           int b, int n) {
  const Size nn = Size(n) * n;
  const Size total = Size(b) * nn;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(blockDim.x) * gridDim.x) {
    const Size q = idx / nn;
    const Size elem = idx - q * nn;
    raw_bar_tile[idx] = rho_tile[q] * bar_g[elem];
  }
}

__global__ void cabs_add_bar_occ_to_frame_kernel(
    const double* __restrict__ bar_occ, double* __restrict__ g_out,
    int occ_rows, int o, int ext) {
  const Size total = Size(occ_rows) * o;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(blockDim.x) * gridDim.x) {
    const int m = static_cast<int>(idx / o);
    const int i = static_cast<int>(idx - Size(m) * o);
    g_out[Size(m) * ext + i] += bar_occ[idx];
  }
}

__global__ void cabs_fk_sym_skew_kernel(const double* __restrict__ moment,
                                        double* __restrict__ sym_m,
                                        double* __restrict__ skew, int rank) {
  const Size rr = Size(rank) * rank;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < rr;
       idx += Size(blockDim.x) * gridDim.x) {
    const int r = static_cast<int>(idx / rank);
    const int s = static_cast<int>(idx - Size(r) * rank);
    const double m_rs = moment[idx];
    const double m_sr = moment[Size(s) * rank + r];
    sym_m[idx] = 0.5 * (m_rs + m_sr);
    skew[idx] = m_rs - m_sr;
  }
}

__global__ void cabs_fk_kernel_bar_kernel(const double* __restrict__ cross,
                                          const double* __restrict__ proj,
                                          double* __restrict__ kernel_bar,
                                          int atomic) {
  const Size aa = Size(atomic) * atomic;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < aa;
       idx += Size(blockDim.x) * gridDim.x) {
    const int u = static_cast<int>(idx / atomic);
    const int v = static_cast<int>(idx - Size(u) * atomic);
    kernel_bar[idx] =
        0.5 * (cross[idx] + cross[Size(v) * atomic + u] - proj[idx]);
  }
}

__global__ void sub_inplace_kernel(double* __restrict__ dst,
                                   const double* __restrict__ src, Size n) {
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < n;
       idx += Size(blockDim.x) * gridDim.x) {
    dst[idx] -= src[idx];
  }
}

// One block of 256 threads per slice a in [0, b):
// charge_tile[a] = (accumulate ? charge_tile[a] : 0) + sum_{bi<v, i<o} df[a, o+bi, i] * trial[bi, i]
__global__ void cphf_vo_dot_kernel(const double* __restrict__ df_tile,
                                   const double* __restrict__ trial,
                                   double* __restrict__ charge_tile, int b,
                                   int o, int v, int obs, int accumulate) {
  __shared__ double smem[256];
  const int a = blockIdx.x;
  if (a >= b) return;
  const Size oo_full = Size(obs) * obs;
  const Size vo_n = Size(v) * o;
  const double* vo_a = df_tile + Size(a) * oo_full + Size(o) * obs;
  double acc = 0.0;
  for (Size idx = threadIdx.x; idx < vo_n; idx += blockDim.x) {
    const int bi = static_cast<int>(idx / o);
    const int i = static_cast<int>(idx - Size(bi) * o);
    acc += vo_a[Size(bi) * obs + i] * trial[idx];
  }
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (accumulate) {
      charge_tile[a] += smem[0];
    } else {
      charge_tile[a] = smem[0];
    }
  }
}

__global__ void cphf_init_gap_trial_kernel(const double* __restrict__ gap,
                                           const double* __restrict__ trial,
                                           double* __restrict__ out,
                                           Size vo_n) {
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < vo_n;
       idx += Size(blockDim.x) * gridDim.x) {
    out[idx] = gap[idx] * trial[idx];
  }
}

__global__ void cphf_reduce_coul_exch_kernel(
    const double* __restrict__ df_tile, const double* __restrict__ charge_tile,
    const double* __restrict__ exch_tile, double* __restrict__ out, int b,
    int o, int v, int obs, double coul_scale) {
  const Size vo_n = Size(v) * o;
  const Size oo_full = Size(obs) * obs;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < vo_n;
       idx += Size(blockDim.x) * gridDim.x) {
    const int bi = static_cast<int>(idx / o);
    const int i = static_cast<int>(idx - Size(bi) * o);
    const Size vo_off = Size(o + bi) * obs + i;
    double acc = 0.0;
    for (int a = 0; a < b; ++a) {
      acc += (coul_scale * charge_tile[a]) * df_tile[Size(a) * oo_full + vo_off] -
             exch_tile[Size(a) * vo_n + idx];
    }
    out[idx] += acc;
  }
}

// One block of 256 threads per a in [0, b):
// bar_dcharge[a] += scale * <p_s, df_tile[a]>, charge[a] += scale * tr_occ(df_tile[a])
__global__ void nuc_charges_kernel(const double* __restrict__ df_tile,
                                   const double* __restrict__ p_s,
                                   double* __restrict__ charge_tile,
                                   double* __restrict__ bar_dcharge_tile, int b,
                                   int o, int obs, double scale) {
  __shared__ double smem_d[256];
  __shared__ double smem_c[256];
  const int a = blockIdx.x;
  if (a >= b) return;
  const Size nn_obs = Size(obs) * obs;
  const double* b_a = df_tile + Size(a) * nn_obs;
  double d_acc = 0.0;
  for (Size idx = threadIdx.x; idx < nn_obs; idx += blockDim.x) {
    d_acc += p_s[idx] * b_a[idx];
  }
  double c_acc = 0.0;
  for (int i = threadIdx.x; i < o; i += blockDim.x) {
    c_acc += b_a[Size(i) * obs + i];
  }
  smem_d[threadIdx.x] = d_acc;
  smem_c[threadIdx.x] = c_acc;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem_d[threadIdx.x] += smem_d[threadIdx.x + stride];
      smem_c[threadIdx.x] += smem_c[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    bar_dcharge_tile[a] += scale * smem_d[0];
    if (o > 0) charge_tile[a] += scale * smem_c[0];
  }
}

__global__ void nuc_accum_sm_kernel(const double* __restrict__ df_tile,
                                    const double* __restrict__ sex_tile,
                                    const double* __restrict__ bar_dcharge_tile,
                                    double* __restrict__ sm_work, int b, int o,
                                    int obs) {
  const Size oo = Size(o) * o;
  const Size nn_obs = Size(obs) * obs;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < oo;
       idx += Size(blockDim.x) * gridDim.x) {
    const int i = static_cast<int>(idx / o);
    const int j = static_cast<int>(idx - Size(i) * o);
    const Size obs_off = Size(i) * obs + j;
    double acc = 0.0;
    for (int a = 0; a < b; ++a) {
      acc += (-bar_dcharge_tile[a]) * df_tile[Size(a) * nn_obs + obs_off] +
             sex_tile[Size(a) * oo + idx];
    }
    sm_work[obs_off] += acc;
  }
}

__global__ void nuc_init_bar_db_kernel(
    double* __restrict__ bar_db_tile, const double* __restrict__ charge_tile,
    const double* __restrict__ bar_dcharge_tile, const double* __restrict__ p_s,
    int b, int o, int obs) {
  const Size nn_obs = Size(obs) * obs;
  const Size total = Size(b) * nn_obs;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(blockDim.x) * gridDim.x) {
    const Size a = idx / nn_obs;
    const Size elem = idx - a * nn_obs;
    const int p = static_cast<int>(elem / obs);
    const int q_i = static_cast<int>(elem - Size(p) * obs);
    double v = charge_tile[a] * p_s[elem];
    if (p == q_i && p < o) {
      v += bar_dcharge_tile[a];
    }
    bar_db_tile[idx] = v;
  }
}

__global__ void nuc_sub_ex_kernel(double* __restrict__ bar_db_tile,
                                  const double* __restrict__ ex_tile, int b,
                                  int o, int obs) {
  const Size per_q = Size(obs) * o;
  const Size nn_obs = Size(obs) * obs;
  const Size total = Size(b) * per_q;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(blockDim.x) * gridDim.x) {
    const Size a = idx / per_q;
    const Size rem = idx - a * per_q;
    const int p = static_cast<int>(rem / o);
    const int i = static_cast<int>(rem - Size(p) * o);
    bar_db_tile[a * nn_obs + Size(p) * obs + i] -= ex_tile[idx];
  }
}

__global__ void nuc_accum_ao_sym_kernel(const double* __restrict__ ao_raw_tile,
                                        double* __restrict__ ao_sym_tile, int b,
                                        int nao, double scale) {
  const Size nn_ao = Size(nao) * nao;
  const Size total = Size(b) * nn_ao;
  const double half_s = 0.5 * scale;
  for (Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += Size(blockDim.x) * gridDim.x) {
    const Size a = idx / nn_ao;
    const Size elem = idx - a * nn_ao;
    const int m = static_cast<int>(elem / nao);
    const int v = static_cast<int>(elem - Size(m) * nao);
    const Size base = a * nn_ao;
    ao_sym_tile[idx] +=
        half_s * (ao_raw_tile[idx] + ao_raw_tile[base + Size(v) * nao + m]);
  }
}

#endif  // GQC_SINK_HOST_REFERENCE

}  // namespace

#ifdef GQC_SINK_HOST_REFERENCE

extern "C" int gqc_f12_ao_mo_project_fwd_host_reference(
    int nc, int ext, int left_dim, int aux, int q_chunk_hint,
    const double* coeffs, const double* ao, double* out) {
  try {
  (void)q_chunk_hint;
  if (nc <= 0 || ext <= 0 || ext > nc || left_dim < 0 || left_dim > ext ||
      aux <= 0) {
    return kStatusInvalidArg;
  }
  if (left_dim == 0) {
    return kStatusOk;
  }
  if (coeffs == nullptr || ao == nullptr || out == nullptr) {
    return kStatusInvalidArg;
  }
  if (!is_finite_buffer(coeffs, Size(nc) * ext) ||
      !is_finite_buffer(ao, Size(aux) * nc * nc)) {
    return kStatusNonFinite;
  }

  std::vector<double> tmp(Size(left_dim) * nc);
  const Size ao_stride = Size(nc) * nc;
  const Size out_stride = Size(left_dim) * ext;
  for (int q = 0; q < aux; ++q) {
    const double* ao_q = ao + Size(q) * ao_stride;
    double* out_q = out + Size(q) * out_stride;
    // tmp[left_dim, nc] = C[:, :left_dim]^T @ AO[q]
    rm_gemm_host(true, false, left_dim, nc, nc, 1.0, coeffs, ext, ao_q, nc,
                 0.0, tmp.data(), nc);
    // out_q[left_dim, ext] = tmp @ C[:, :ext]
    rm_gemm_host(false, false, left_dim, ext, nc, 1.0, tmp.data(), nc, coeffs,
                 ext, 0.0, out_q, ext);
  }
  if (!is_finite_buffer(out, Size(aux) * out_stride)) return kStatusNonFinite;
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_ao_mo_project_rev_host_reference(
    int nc, int ext, int left_dim, int aux, int seed_rows, int tc_rows,
    int mode, int accumulate_bar_ao, int q_chunk_hint, double seed_scale,
    const double* coeffs, const double* ao, const double* seed,
    const double* tc_bar, double* bar_coeffs, double* bar_ao) {
  try {
  (void)q_chunk_hint;
  if (nc <= 0 || ext <= 0 || ext > nc || left_dim < 0 || left_dim > ext ||
      aux <= 0 || seed_rows < 0 || seed_rows > left_dim || tc_rows < 0 ||
      tc_rows > left_dim || mode < 0 || mode > 2) {
    return kStatusInvalidArg;
  }
  if (!std::isfinite(seed_scale)) {
    return kStatusNonFinite;
  }
  const bool want_coeffs = (mode == 0 || mode == 1) && bar_coeffs != nullptr;
  const bool want_ao = (mode == 0 || mode == 2) && bar_ao != nullptr;
  if (want_ao && !accumulate_bar_ao) {
    std::fill_n(bar_ao, Size(aux) * nc * nc, 0.0);
  }
  const int active_rows = std::max(seed_rows, tc_rows);
  if (active_rows == 0 || (!want_coeffs && !want_ao)) {
    return kStatusOk;
  }
  if (coeffs == nullptr) return kStatusInvalidArg;
  if (want_coeffs && ao == nullptr) return kStatusInvalidArg;
  if (seed_rows > 0 && seed == nullptr) return kStatusInvalidArg;
  if (tc_rows > 0 && tc_bar == nullptr) return kStatusInvalidArg;

  if (!is_finite_buffer(coeffs, Size(nc) * ext)) return kStatusNonFinite;
  if (want_coeffs && !is_finite_buffer(ao, Size(aux) * nc * nc)) {
    return kStatusNonFinite;
  }
  if (seed_rows > 0 && !is_finite_buffer(seed, Size(aux) * seed_rows * ext)) {
    return kStatusNonFinite;
  }
  if (tc_rows > 0 && !is_finite_buffer(tc_bar, Size(aux) * tc_rows * ext)) {
    return kStatusNonFinite;
  }

  std::vector<double> s_buf(Size(active_rows) * ext);
  std::vector<double> u_buf(Size(active_rows) * nc);
  std::vector<double> l_buf;
  if (want_coeffs) {
    l_buf.resize(Size(active_rows) * nc);
  }

  const Size ao_stride = Size(nc) * nc;
  const Size seed_stride = Size(seed_rows) * ext;
  const Size tc_stride = Size(tc_rows) * ext;

  for (int q = 0; q < aux; ++q) {
    const double* seed_q =
        seed_rows > 0 ? seed + Size(q) * seed_stride : nullptr;
    const double* tc_q = tc_rows > 0 ? tc_bar + Size(q) * tc_stride : nullptr;
    build_effective_seed_slice(active_rows, ext, seed_rows, tc_rows, seed_scale,
                               seed_q, tc_q, s_buf.data());

    // U[active_rows, nc] = S[active_rows, ext] @ C^T[ext, nc]
    rm_gemm_host(false, true, active_rows, nc, ext, 1.0, s_buf.data(), ext,
                 coeffs, ext, 0.0, u_buf.data(), nc);

    if (want_ao) {
      double* bar_ao_q = bar_ao + Size(q) * ao_stride;
      // bar_ao[q] += C[:, :active_rows] @ U
      rm_gemm_host(false, false, nc, nc, active_rows, 1.0, coeffs, ext,
                   u_buf.data(), nc, 1.0, bar_ao_q, nc);
    }

    if (want_coeffs) {
      const double* ao_q = ao + Size(q) * ao_stride;
      // bar_coeffs[:, :active_rows] += AO[q] @ U^T
      rm_gemm_host(false, true, nc, active_rows, nc, 1.0, ao_q, nc,
                   u_buf.data(), nc, 1.0, bar_coeffs, ext);
      // L[active_rows, nc] = C[:, :active_rows]^T @ AO[q]
      rm_gemm_host(true, false, active_rows, nc, nc, 1.0, coeffs, ext, ao_q,
                   nc, 0.0, l_buf.data(), nc);
      // bar_coeffs[:, :ext] += L^T @ S
      rm_gemm_host(true, false, nc, ext, active_rows, 1.0, l_buf.data(), nc,
                   s_buf.data(), ext, 1.0, bar_coeffs, ext);
    }
  }
  if (want_coeffs && !is_finite_buffer(bar_coeffs, Size(nc) * ext)) {
    return kStatusNonFinite;
  }
  if (want_ao && !is_finite_buffer(bar_ao, Size(aux) * ao_stride)) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_ao_mo_bilin_fwd_host_reference(
    int nc, int n_left, int n_right, const double* A, const double* C_left,
    const double* C_right, double* out) {
  try {
  if (nc <= 0 || n_left <= 0 || n_right <= 0 || A == nullptr ||
      C_left == nullptr || C_right == nullptr || out == nullptr) {
    return kStatusInvalidArg;
  }
  if (!is_finite_buffer(A, Size(nc) * nc) ||
      !is_finite_buffer(C_left, Size(nc) * n_left) ||
      !is_finite_buffer(C_right, Size(nc) * n_right)) {
    return kStatusNonFinite;
  }
  std::vector<double> tmp(Size(nc) * n_right);
  // tmp = A @ C_right
  rm_gemm_host(false, false, nc, n_right, nc, 1.0, A, nc, C_right, n_right,
               0.0, tmp.data(), n_right);
  // out = C_left^T @ tmp
  rm_gemm_host(true, false, n_left, n_right, nc, 1.0, C_left, n_left,
               tmp.data(), n_right, 0.0, out, n_right);
  if (!is_finite_buffer(out, Size(n_left) * n_right)) return kStatusNonFinite;
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_ao_mo_bilin_rev_host_reference(
    int nc, int n_left, int n_right, const double* A, const double* C_left,
    const double* C_right, const double* bar_out, double* bar_A,
    double* bar_C_left, double* bar_C_right) {
  try {
  if (nc <= 0 || n_left <= 0 || n_right <= 0 || A == nullptr ||
      C_left == nullptr || C_right == nullptr || bar_out == nullptr) {
    return kStatusInvalidArg;
  }
  if (!is_finite_buffer(A, Size(nc) * nc) ||
      !is_finite_buffer(C_left, Size(nc) * n_left) ||
      !is_finite_buffer(C_right, Size(nc) * n_right) ||
      !is_finite_buffer(bar_out, Size(n_left) * n_right)) {
    return kStatusNonFinite;
  }
  if (bar_A != nullptr) {
    // U[n_left, nc] = bar_out @ C_right^T
    std::vector<double> u(Size(n_left) * nc);
    rm_gemm_host(false, true, n_left, nc, n_right, 1.0, bar_out, n_right,
                 C_right, n_right, 0.0, u.data(), nc);
    // bar_A += C_left @ U
    rm_gemm_host(false, false, nc, nc, n_left, 1.0, C_left, n_left, u.data(),
                 nc, 1.0, bar_A, nc);
  }
  if (bar_C_left != nullptr) {
    // tmp1[nc, n_right] = A @ C_right
    std::vector<double> tmp1(Size(nc) * n_right);
    rm_gemm_host(false, false, nc, n_right, nc, 1.0, A, nc, C_right, n_right,
                 0.0, tmp1.data(), n_right);
    // bar_C_left += tmp1 @ bar_out^T
    rm_gemm_host(false, true, nc, n_left, n_right, 1.0, tmp1.data(), n_right,
                 bar_out, n_right, 1.0, bar_C_left, n_left);
  }
  if (bar_C_right != nullptr) {
    // tmp2[nc, n_left] = A^T @ C_left
    std::vector<double> tmp2(Size(nc) * n_left);
    rm_gemm_host(true, false, nc, n_left, nc, 1.0, A, nc, C_left, n_left, 0.0,
                 tmp2.data(), n_left);
    // bar_C_right += tmp2 @ bar_out
    rm_gemm_host(false, false, nc, n_right, n_left, 1.0, tmp2.data(), n_left,
                 bar_out, n_right, 1.0, bar_C_right, n_right);
  }
  if ((bar_A != nullptr && !is_finite_buffer(bar_A, Size(nc) * nc)) ||
      (bar_C_left != nullptr && !is_finite_buffer(bar_C_left, Size(nc) * n_left)) ||
      (bar_C_right != nullptr && !is_finite_buffer(bar_C_right, Size(nc) * n_right))) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_coulomb_fit_fwd_host_reference(
    int aux, int pairs, const double* whitener, const double* raw,
    double* fit) {
  try {
  if (aux <= 0 || pairs < 0) return kStatusInvalidArg;
  if (pairs == 0) return kStatusOk;
  if (whitener == nullptr || raw == nullptr || fit == nullptr) {
    return kStatusInvalidArg;
  }
  if (!is_finite_buffer(whitener, Size(aux) * aux) ||
      !is_finite_buffer(raw, Size(aux) * pairs)) {
    return kStatusNonFinite;
  }
  std::vector<double> tmp(Size(aux) * pairs);
  // tmp = W @ raw
  rm_gemm_host(false, false, aux, pairs, aux, 1.0, whitener, aux, raw, pairs,
               0.0, tmp.data(), pairs);
  // fit = W^T @ tmp
  rm_gemm_host(true, false, aux, pairs, aux, 1.0, whitener, aux, tmp.data(),
               pairs, 0.0, fit, pairs);
  if (!is_finite_buffer(fit, Size(aux) * pairs)) return kStatusNonFinite;
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_coulomb_fit_rev_host_reference(
    int aux, int pairs, double metric_scale, const double* whitener,
    const double* raw, const double* fit_bar, double* three_center_bar,
    double* metric_bar) {
  try {
  if (aux <= 0 || pairs < 0) return kStatusInvalidArg;
  if (!std::isfinite(metric_scale)) return kStatusNonFinite;
  if (pairs == 0) return kStatusOk;
  if (whitener == nullptr || raw == nullptr || fit_bar == nullptr ||
      three_center_bar == nullptr || metric_bar == nullptr) {
    return kStatusInvalidArg;
  }
  if (!is_finite_buffer(whitener, Size(aux) * aux) ||
      !is_finite_buffer(raw, Size(aux) * pairs) ||
      !is_finite_buffer(fit_bar, Size(aux) * pairs)) {
    return kStatusNonFinite;
  }
  std::vector<double> tmp(Size(aux) * pairs);
  std::vector<double> fit(Size(aux) * pairs);
  // fit = W^T @ (W @ raw)
  rm_gemm_host(false, false, aux, pairs, aux, 1.0, whitener, aux, raw, pairs,
               0.0, tmp.data(), pairs);
  rm_gemm_host(true, false, aux, pairs, aux, 1.0, whitener, aux, tmp.data(),
               pairs, 0.0, fit.data(), pairs);
  // three_center_bar = W^T @ (W @ fit_bar)
  rm_gemm_host(false, false, aux, pairs, aux, 1.0, whitener, aux, fit_bar,
               pairs, 0.0, tmp.data(), pairs);
  rm_gemm_host(true, false, aux, pairs, aux, 1.0, whitener, aux, tmp.data(),
               pairs, 0.0, three_center_bar, pairs);
  // weight = fit @ three_center_bar^T; metric_bar += -0.5 * metric_scale * (W + W^T)
  std::vector<double> weight(Size(aux) * aux);
  rm_gemm_host(false, true, aux, aux, pairs, 1.0, fit.data(), pairs,
               three_center_bar, pairs, 0.0, weight.data(), aux);
  const double half_neg = -0.5 * metric_scale;
  for (int a = 0; a < aux; ++a) {
    for (int b = 0; b < aux; ++b) {
      metric_bar[Size(a) * aux + b] +=
          half_neg * (weight[Size(a) * aux + b] + weight[Size(b) * aux + a]);
    }
  }
  if (!is_finite_buffer(three_center_bar, Size(aux) * pairs) ||
      !is_finite_buffer(metric_bar, Size(aux) * aux)) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

// ============================================================================
// CABS Singles E+F dynamic-dimension AOT kernels (Host Reference FP64)
// ============================================================================

extern "C" int gqc_f12_cabs_fock_fwd_host_reference(
    int n, int q, int oa, int ob, int occ_rows, int same_occ, int q_chunk_hint,
    const double* hcore, const double* raw, const double* whitener,
    const double* occ_a, const double* occ_b, double* fock_ao_a,
    double* fock_ao_b, double* rho_out, double* raw_a_out, double* fit_a_out,
    double* raw_b_out, double* fit_b_out) {
  try {
  (void)q_chunk_hint;
  if (n <= 0 || q <= 0 || oa < 0 || oa > n || ob < 0 || ob > n ||
      occ_rows <= 0 || occ_rows > n) {
    return kStatusInvalidArg;
  }
  if (same_occ && oa != ob) {
    return kStatusInvalidArg;
  }
  if (hcore == nullptr || raw == nullptr || whitener == nullptr ||
      fock_ao_a == nullptr || fock_ao_b == nullptr) {
    return kStatusInvalidArg;
  }
  if (oa > 0 && occ_a == nullptr) return kStatusInvalidArg;
  if (!same_occ && ob > 0 && occ_b == nullptr) return kStatusInvalidArg;

  const Size nn = Size(n) * n;
  const Size qq = Size(q) * q;
  if (!is_finite_buffer(hcore, nn) || !is_finite_buffer(raw, Size(q) * nn) ||
      !is_finite_buffer(whitener, qq)) {
    return kStatusNonFinite;
  }
  if (oa > 0 && !is_finite_buffer(occ_a, Size(occ_rows) * oa)) {
    return kStatusNonFinite;
  }
  if (!same_occ && ob > 0 && !is_finite_buffer(occ_b, Size(occ_rows) * ob)) {
    return kStatusNonFinite;
  }

  auto build_spin = [&](int o, const double* occ, std::vector<double>& k_mat,
                        std::vector<double>& rho_s, double* r_out,
                        double* d_out) {
    k_mat.assign(nn, 0.0);
    rho_s.assign(Size(q), 0.0);
    if (o == 0) return;

    const Size on = Size(o) * n;
    const Size qon = Size(q) * on;
    std::vector<double> R(qon);
    std::vector<double> T(Size(n) * o);
    for (int a = 0; a < q; ++a) {
      const double* raw_a = raw + Size(a) * nn;
      // T[n, o] = raw[a, :, :occ_rows] @ occ[occ_rows, o]
      rm_gemm_host(false, false, n, o, occ_rows, 1.0, raw_a, n, occ, o, 0.0,
                   T.data(), o);
      double* r_slice = R.data() + Size(a) * on;
      for (int m = 0; m < n; ++m) {
        const double* t_row = T.data() + Size(m) * o;
        for (int i = 0; i < o; ++i) {
          r_slice[Size(i) * n + m] = t_row[i];
        }
      }
    }
    if (r_out != nullptr) {
      std::copy(R.begin(), R.end(), r_out);
    }

    // B[q, o*n] = W @ R
    std::vector<double> B(qon);
    rm_gemm_host(false, false, q, static_cast<int>(on), q, 1.0, whitener, q,
                 R.data(), static_cast<int>(on), 0.0, B.data(),
                 static_cast<int>(on));
    // K[n, n] = B_flat^T @ B_flat where B_flat has shape [q*o, n]
    rm_gemm_host(true, false, n, n, q * o, 1.0, B.data(), n, B.data(), n, 0.0,
                 k_mat.data(), n);
    // D[q, o*n] = W^T @ B (overwrite R)
    rm_gemm_host(true, false, q, static_cast<int>(on), q, 1.0, whitener, q,
                 B.data(), static_cast<int>(on), 0.0, R.data(),
                 static_cast<int>(on));
    if (d_out != nullptr) {
      std::copy(R.begin(), R.end(), d_out);
    }
    for (int a = 0; a < q; ++a) {
      const double* d_slice = R.data() + Size(a) * on;
      double acc = 0.0;
      for (int i = 0; i < o; ++i) {
        const double* d_i = d_slice + Size(i) * n;
        for (int m = 0; m < occ_rows; ++m) {
          acc += d_i[m] * occ[Size(m) * o + i];
        }
      }
      rho_s[a] = acc;
    }
  };

  std::vector<double> k_a, k_b, rho_a, rho_b;
  build_spin(oa, occ_a, k_a, rho_a, raw_a_out, fit_a_out);
  std::vector<double> rho(Size(q), 0.0);
  if (same_occ) {
    for (int a = 0; a < q; ++a) rho[a] = 2.0 * rho_a[a];
  } else {
    build_spin(ob, occ_b, k_b, rho_b, raw_b_out, fit_b_out);
    for (int a = 0; a < q; ++a) rho[a] = rho_a[a] + rho_b[a];
  }
  if (rho_out != nullptr) {
    std::copy(rho.begin(), rho.end(), rho_out);
  }

  // J[m, n] = sum_a rho[a] * raw[a, m, n]
  std::vector<double> j_mat(nn, 0.0);
  for (int a = 0; a < q; ++a) {
    const double ra = rho[a];
    if (ra == 0.0) continue;
    const double* raw_a = raw + Size(a) * nn;
    for (Size idx = 0; idx < nn; ++idx) {
      j_mat[idx] += ra * raw_a[idx];
    }
  }

  for (Size idx = 0; idx < nn; ++idx) {
    fock_ao_a[idx] = hcore[idx] + j_mat[idx] - k_a[idx];
  }
  if (same_occ) {
    std::copy_n(fock_ao_a, nn, fock_ao_b);
  } else {
    for (Size idx = 0; idx < nn; ++idx) {
      fock_ao_b[idx] = hcore[idx] + j_mat[idx] - k_b[idx];
    }
  }
  if (!is_finite_buffer(fock_ao_a, nn) || !is_finite_buffer(fock_ao_b, nn) ||
      (rho_out != nullptr && !is_finite_buffer(rho_out, Size(q))) ||
      (oa > 0 && raw_a_out != nullptr && !is_finite_buffer(raw_a_out, Size(q) * oa * n)) ||
      (oa > 0 && fit_a_out != nullptr && !is_finite_buffer(fit_a_out, Size(q) * oa * n)) ||
      (!same_occ && ob > 0 && raw_b_out != nullptr &&
       !is_finite_buffer(raw_b_out, Size(q) * ob * n)) ||
      (!same_occ && ob > 0 && fit_b_out != nullptr &&
       !is_finite_buffer(fit_b_out, Size(q) * ob * n))) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_cabs_fock_rev_host_reference(
    int n, int ext, int q, int oa, int ob, int occ_rows, int same_spin,
    int q_chunk_hint, const double* fock_ao_a, const double* fock_ao_b,
    const double* frame_a, const double* frame_b, const double* seed_a,
    const double* seed_b, const double* raw, const double* whitener,
    const double* rho, const double* raw_a, const double* fit_a,
    const double* raw_b, const double* fit_b, const double* occ_a,
    const double* occ_b, double* ga_out, double* gb_out, double* core_bar_out,
    double* raw_bar_out, double* metric_bar_out) {
  try {
  (void)q_chunk_hint;
  if (n <= 0 || ext <= 0 || ext > n || q <= 0 || oa < 0 || oa > ext ||
      ob < 0 || ob > ext || occ_rows <= 0 || occ_rows > n) {
    return kStatusInvalidArg;
  }
  if (same_spin && oa != ob) return kStatusInvalidArg;
  if (fock_ao_a == nullptr || frame_a == nullptr || seed_a == nullptr ||
      raw == nullptr || whitener == nullptr || rho == nullptr ||
      ga_out == nullptr || gb_out == nullptr || core_bar_out == nullptr ||
      raw_bar_out == nullptr || metric_bar_out == nullptr) {
    return kStatusInvalidArg;
  }
  if (!same_spin &&
      (fock_ao_b == nullptr || frame_b == nullptr || seed_b == nullptr)) {
    return kStatusInvalidArg;
  }
  if (oa > 0 && (raw_a == nullptr || fit_a == nullptr || occ_a == nullptr)) {
    return kStatusInvalidArg;
  }
  if (!same_spin && ob > 0 &&
      (raw_b == nullptr || fit_b == nullptr || occ_b == nullptr)) {
    return kStatusInvalidArg;
  }

  const Size nn = Size(n) * n;
  const Size ne = Size(n) * ext;
  const Size ee = Size(ext) * ext;
  const Size qq = Size(q) * q;
  if (!is_finite_buffer(fock_ao_a, nn) || !is_finite_buffer(frame_a, ne) ||
      !is_finite_buffer(seed_a, ee) || !is_finite_buffer(raw, Size(q) * nn) ||
      !is_finite_buffer(whitener, qq) || !is_finite_buffer(rho, Size(q))) {
    return kStatusNonFinite;
  }
  if (!same_spin &&
      (!is_finite_buffer(fock_ao_b, nn) || !is_finite_buffer(frame_b, ne) ||
       !is_finite_buffer(seed_b, ee))) {
    return kStatusNonFinite;
  }
  if (oa > 0 &&
      (!is_finite_buffer(raw_a, Size(q) * oa * n) ||
       !is_finite_buffer(fit_a, Size(q) * oa * n) ||
       !is_finite_buffer(occ_a, Size(occ_rows) * oa))) {
    return kStatusNonFinite;
  }
  if (!same_spin && ob > 0 &&
      (!is_finite_buffer(raw_b, Size(q) * ob * n) ||
       !is_finite_buffer(fit_b, Size(q) * ob * n) ||
       !is_finite_buffer(occ_b, Size(occ_rows) * ob))) {
    return kStatusNonFinite;
  }

  std::fill_n(ga_out, ne, 0.0);
  std::fill_n(gb_out, ne, 0.0);
  std::fill_n(metric_bar_out, qq, 0.0);

  auto congruence_rev = [&](const double* fock_ao, const double* frame,
                            const double* seed, std::vector<double>& ao_seed,
                            double* g_out) {
    std::vector<double> m_sym(nn);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        m_sym[Size(i) * n + j] =
            0.5 * (fock_ao[Size(i) * n + j] + fock_ao[Size(j) * n + i]);
      }
    }
    // U[ext, n] = seed @ frame^T; bar_mat = frame @ U; ao_seed = sym(bar_mat)
    std::vector<double> u(Size(ext) * n);
    rm_gemm_host(false, true, ext, n, ext, 1.0, seed, ext, frame, ext, 0.0,
                 u.data(), n);
    std::vector<double> bar_mat(nn);
    rm_gemm_host(false, false, n, n, ext, 1.0, frame, ext, u.data(), n, 0.0,
                 bar_mat.data(), n);
    ao_seed.resize(nn);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        ao_seed[Size(i) * n + j] =
            0.5 * (bar_mat[Size(i) * n + j] + bar_mat[Size(j) * n + i]);
      }
    }
    // g_out = m_sym @ frame @ (seed + seed^T)
    std::vector<double> s_sym(ee);
    for (int p = 0; p < ext; ++p) {
      for (int r = 0; r < ext; ++r) {
        s_sym[Size(p) * ext + r] =
            seed[Size(p) * ext + r] + seed[Size(r) * ext + p];
      }
    }
    std::vector<double> mc(ne);
    rm_gemm_host(false, false, n, ext, n, 1.0, m_sym.data(), n, frame, ext, 0.0,
                 mc.data(), ext);
    rm_gemm_host(false, false, n, ext, ext, 1.0, mc.data(), ext, s_sym.data(),
                 ext, 1.0, g_out, ext);
  };

  std::vector<double> ao_seed_a, ao_seed_b;
  congruence_rev(fock_ao_a, frame_a, seed_a, ao_seed_a, ga_out);
  std::vector<double> bar_g(nn);
  if (same_spin) {
    for (Size idx = 0; idx < nn; ++idx) {
      bar_g[idx] = 2.0 * ao_seed_a[idx];
    }
  } else {
    congruence_rev(fock_ao_b, frame_b, seed_b, ao_seed_b, gb_out);
    for (Size idx = 0; idx < nn; ++idx) {
      bar_g[idx] = ao_seed_a[idx] + ao_seed_b[idx];
    }
  }
  std::copy(bar_g.begin(), bar_g.end(), core_bar_out);

  std::vector<double> bar_rho(Size(q), 0.0);
  for (int a = 0; a < q; ++a) {
    const double* raw_slice = raw + Size(a) * nn;
    double* rbar_slice = raw_bar_out + Size(a) * nn;
    const double ra = rho[a];
    double dot = 0.0;
    for (Size idx = 0; idx < nn; ++idx) {
      dot += raw_slice[idx] * bar_g[idx];
      rbar_slice[idx] = ra * bar_g[idx];
    }
    bar_rho[a] = dot;
  }

  auto pullback_spin = [&](int o, double spin_scale,
                           const std::vector<double>& ao_seed,
                           const double* R_s, const double* D_s,
                           const double* occ_s, double* g_out) {
    if (o == 0) return;
    const Size on = Size(o) * n;
    const Size qon = Size(q) * on;

    std::vector<double> k_seed(nn);
    for (Size idx = 0; idx < nn; ++idx) k_seed[idx] = -ao_seed[idx];

    // 1. bar_O_from_rho[m, i] = sum_a bar_rho[a] * D_s[a, i, m] (m < occ_rows)
    for (int a = 0; a < q; ++a) {
      const double br = bar_rho[a];
      if (br == 0.0) continue;
      const double* d_slice = D_s + Size(a) * on;
      for (int i = 0; i < o; ++i) {
        const double* d_i = d_slice + Size(i) * n;
        for (int m = 0; m < occ_rows; ++m) {
          g_out[Size(m) * ext + i] += br * d_i[m];
        }
      }
    }

    // 2. bar_D[a, i, m] = (m < occ_rows ? bar_rho[a] * occ_s[m, i] : 0) + (R_s @ k_seed^T)[a, i, m]
    std::vector<double> bar_D(qon);
    rm_gemm_host(false, true, q * o, n, n, 1.0, R_s, n, k_seed.data(), n, 0.0,
                 bar_D.data(), n);
    for (int a = 0; a < q; ++a) {
      const double br = bar_rho[a];
      if (br == 0.0) continue;
      double* bd_slice = bar_D.data() + Size(a) * on;
      for (int i = 0; i < o; ++i) {
        double* bd_i = bd_slice + Size(i) * n;
        for (int m = 0; m < occ_rows; ++m) {
          bd_i[m] += br * occ_s[Size(m) * o + i];
        }
      }
    }

    // 3. tc_bar = W^T @ (W @ bar_D)
    std::vector<double> tmp(qon);
    std::vector<double> tc_bar(qon);
    rm_gemm_host(false, false, q, static_cast<int>(on), q, 1.0, whitener, q,
                 bar_D.data(), static_cast<int>(on), 0.0, tmp.data(),
                 static_cast<int>(on));
    rm_gemm_host(true, false, q, static_cast<int>(on), q, 1.0, whitener, q,
                 tmp.data(), static_cast<int>(on), 0.0, tc_bar.data(),
                 static_cast<int>(on));

    // 4. metric_bar += (-0.5 * spin_scale) * (W_m + W_m^T), W_m = D_s @ tc_bar^T
    std::vector<double> w_m(qq);
    rm_gemm_host(false, true, q, q, static_cast<int>(on), 1.0, D_s,
                 static_cast<int>(on), tc_bar.data(), static_cast<int>(on), 0.0,
                 w_m.data(), q);
    const double half_neg = -0.5 * spin_scale;
    for (int a = 0; a < q; ++a) {
      for (int b = 0; b < q; ++b) {
        metric_bar_out[Size(a) * q + b] +=
            half_neg * (w_m[Size(a) * q + b] + w_m[Size(b) * q + a]);
      }
    }

    // 5. seed_R = tc_bar + D_s @ k_seed (accumulated into tc_bar)
    rm_gemm_host(false, false, q * o, n, n, 1.0, D_s, n, k_seed.data(), n, 1.0,
                 tc_bar.data(), n);

    // 6. Half-transform pullback:
    //    raw_bar[a, :, :occ_rows] += spin_scale * seed_R[a]^T @ occ_s^T
    //    bar_occ_half[occ_rows, o] += sum_a raw[a, :, :occ_rows]^T @ seed_R[a]^T
    std::vector<double> bar_occ_half(Size(occ_rows) * o, 0.0);
    for (int a = 0; a < q; ++a) {
      const double* s_a = tc_bar.data() + Size(a) * on;
      const double* raw_slice = raw + Size(a) * nn;
      double* rbar_slice = raw_bar_out + Size(a) * nn;
      rm_gemm_host(true, true, n, occ_rows, o, spin_scale, s_a, n, occ_s, o,
                   1.0, rbar_slice, n);
      rm_gemm_host(true, true, occ_rows, o, n, 1.0, raw_slice, n, s_a, n, 1.0,
                   bar_occ_half.data(), o);
    }
    for (int v = 0; v < occ_rows; ++v) {
      for (int i = 0; i < o; ++i) {
        g_out[Size(v) * ext + i] += bar_occ_half[Size(v) * o + i];
      }
    }
  };

  if (same_spin) {
    pullback_spin(oa, 2.0, ao_seed_a, raw_a, fit_a, occ_a, ga_out);
    std::copy_n(ga_out, ne, gb_out);
  } else {
    pullback_spin(oa, 1.0, ao_seed_a, raw_a, fit_a, occ_a, ga_out);
    pullback_spin(ob, 1.0, ao_seed_b, raw_b, fit_b, occ_b, gb_out);
  }
  if (!is_finite_buffer(ga_out, ne) || !is_finite_buffer(gb_out, ne) ||
      !is_finite_buffer(core_bar_out, nn) ||
      !is_finite_buffer(raw_bar_out, Size(q) * nn) ||
      !is_finite_buffer(metric_bar_out, qq)) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_cabs_frame_kernel_cotangent_host_reference(
    int atomic, int rank, const double* overlap, const double* frame,
    const double* frame_bar, double* kernel_bar_out, double* dual_gram_out,
    double* gauge_skew_out, double* reconstruction_residual_out) {
  try {
  if (atomic <= 0 || rank <= 0 || rank > atomic || overlap == nullptr ||
      frame == nullptr || frame_bar == nullptr || kernel_bar_out == nullptr ||
      dual_gram_out == nullptr || gauge_skew_out == nullptr ||
      reconstruction_residual_out == nullptr) {
    return kStatusInvalidArg;
  }
  const Size aa = Size(atomic) * atomic;
  const Size ar = Size(atomic) * rank;
  const Size rr = Size(rank) * rank;
  if (!is_finite_buffer(overlap, aa) || !is_finite_buffer(frame, ar) ||
      !is_finite_buffer(frame_bar, ar)) {
    return kStatusNonFinite;
  }

  std::vector<double> dual(ar);
  rm_gemm_host(false, false, atomic, rank, atomic, 1.0, overlap, atomic, frame,
               rank, 0.0, dual.data(), rank);

  std::vector<double> moment(rr);
  rm_gemm_host(true, false, rank, rank, atomic, 1.0, frame, rank, frame_bar,
               rank, 0.0, moment.data(), rank);

  rm_gemm_host(true, false, rank, rank, atomic, 1.0, frame, rank, dual.data(),
               rank, 0.0, dual_gram_out, rank);

  std::vector<double> sym_m(rr);
  for (int r = 0; r < rank; ++r) {
    for (int s = 0; s < rank; ++s) {
      const double m_rs = moment[Size(r) * rank + s];
      const double m_sr = moment[Size(s) * rank + r];
      sym_m[Size(r) * rank + s] = 0.5 * (m_rs + m_sr);
      gauge_skew_out[Size(r) * rank + s] = m_rs - m_sr;
    }
  }

  std::vector<double> cross(aa);
  rm_gemm_host(false, true, atomic, atomic, rank, 1.0, frame_bar, rank,
               dual.data(), rank, 0.0, cross.data(), atomic);

  std::vector<double> dsym(ar);
  rm_gemm_host(false, false, atomic, rank, rank, 1.0, dual.data(), rank,
               sym_m.data(), rank, 0.0, dsym.data(), rank);
  std::vector<double> proj(aa);
  rm_gemm_host(false, true, atomic, atomic, rank, 1.0, dsym.data(), rank,
               dual.data(), rank, 0.0, proj.data(), atomic);

  for (int u = 0; u < atomic; ++u) {
    for (int v = 0; v < atomic; ++v) {
      kernel_bar_out[Size(u) * atomic + v] =
          0.5 * (cross[Size(u) * atomic + v] + cross[Size(v) * atomic + u] -
                 proj[Size(u) * atomic + v]);
    }
  }

  rm_gemm_host(false, false, atomic, rank, atomic, 2.0, kernel_bar_out, atomic,
               frame, rank, 0.0, reconstruction_residual_out, rank);
  for (Size idx = 0; idx < ar; ++idx) {
    reconstruction_residual_out[idx] -= frame_bar[idx];
  }
  if (!is_finite_buffer(kernel_bar_out, aa) ||
      !is_finite_buffer(dual_gram_out, rr) ||
      !is_finite_buffer(gauge_skew_out, rr) ||
      !is_finite_buffer(reconstruction_residual_out, ar)) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_cabs_whiten_3c_host_reference(
    int aux, int pairs, const double* whitener, const double* raw,
    double* df_out) {
  try {
  if (aux <= 0 || pairs < 0) return kStatusInvalidArg;
  if (pairs == 0) return kStatusOk;
  if (whitener == nullptr || raw == nullptr || df_out == nullptr) {
    return kStatusInvalidArg;
  }
  if (!is_finite_buffer(whitener, Size(aux) * aux) ||
      !is_finite_buffer(raw, Size(aux) * pairs)) {
    return kStatusNonFinite;
  }
  rm_gemm_host(false, false, aux, pairs, aux, 1.0, whitener, aux, raw, pairs,
               0.0, df_out, pairs);
  if (!is_finite_buffer(df_out, Size(aux) * pairs)) return kStatusNonFinite;
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_cabs_cphf_action_host_reference(
    int oa, int ob, int obs, int aux, int is_rhf, int q_chunk_hint,
    const double* df_a, const double* df_b, const double* gap_a,
    const double* gap_b, const double* trial_a, const double* trial_b,
    double* out_a, double* out_b) {
  try {
  (void)q_chunk_hint;
  if (obs <= 0 || aux <= 0 || oa < 0 || ob < 0 || oa >= obs || ob >= obs ||
      oa + ob == 0) {
    return kStatusInvalidArg;
  }
  if (is_rhf && (oa == 0 || oa != ob)) {
    return kStatusInvalidArg;
  }
  const int va = obs - oa;
  const int vb = obs - ob;
  const Size oo_full = Size(obs) * obs;
  if (oa > 0 && (df_a == nullptr || gap_a == nullptr || trial_a == nullptr ||
                 out_a == nullptr)) {
    return kStatusInvalidArg;
  }
  if (!is_rhf && ob > 0 &&
      (df_b == nullptr || gap_b == nullptr || trial_b == nullptr ||
       out_b == nullptr)) {
    return kStatusInvalidArg;
  }
  if (oa > 0 && (!is_finite_buffer(df_a, Size(aux) * oo_full) ||
                 !is_finite_buffer(gap_a, Size(va) * oa) ||
                 !is_finite_buffer(trial_a, Size(va) * oa))) {
    return kStatusNonFinite;
  }
  if (!is_rhf && ob > 0 &&
      (!is_finite_buffer(df_b, Size(aux) * oo_full) ||
       !is_finite_buffer(gap_b, Size(vb) * ob) ||
       !is_finite_buffer(trial_b, Size(vb) * ob))) {
    return kStatusNonFinite;
  }

  std::vector<double> total_charge(Size(aux), 0.0);
  auto accum_charge = [&](int o, int v, const double* df_s,
                          const double* trial_s) {
    if (o == 0) return;
    for (int a = 0; a < aux; ++a) {
      const double* vo_a = df_s + Size(a) * oo_full + Size(o) * obs;
      double dot = 0.0;
      for (int b = 0; b < v; ++b) {
        const double* vo_row = vo_a + Size(b) * obs;
        const double* tr_row = trial_s + Size(b) * o;
        for (int i = 0; i < o; ++i) {
          dot += vo_row[i] * tr_row[i];
        }
      }
      total_charge[a] += dot;
    }
  };
  accum_charge(oa, va, df_a, trial_a);
  if (!is_rhf) {
    accum_charge(ob, vb, df_b, trial_b);
  }

  const double coul_scale = is_rhf ? 4.0 : 2.0;
  auto apply_spin = [&](int o, int v, const double* df_s, const double* gap_s,
                        const double* trial_s, double* out_s) {
    if (o == 0) return;
    const Size vo_n = Size(v) * o;
    for (Size idx = 0; idx < vo_n; ++idx) {
      out_s[idx] = gap_s[idx] * trial_s[idx];
    }
    std::vector<double> kv(vo_n);
    std::vector<double> ki(Size(o) * o);
    for (int a = 0; a < aux; ++a) {
      const double* df_slice = df_s + Size(a) * oo_full;
      const double* oo_a = df_slice;
      const double* vo_a = df_slice + Size(o) * obs;
      const double* vv_a = df_slice + Size(o) * obs + o;
      const double c_a = coul_scale * total_charge[a];
      if (c_a != 0.0) {
        for (int b = 0; b < v; ++b) {
          const double* vo_row = vo_a + Size(b) * obs;
          double* out_row = out_s + Size(b) * o;
          for (int i = 0; i < o; ++i) {
            out_row[i] += c_a * vo_row[i];
          }
        }
      }
      // kv[v, o] = vv[v, v] @ trial[v, o]
      rm_gemm_host(false, false, v, o, v, 1.0, vv_a, obs, trial_s, o, 0.0,
                   kv.data(), o);
      // ki[o, o] = vo^T[o, v] @ trial[v, o]
      rm_gemm_host(true, false, o, o, v, 1.0, vo_a, obs, trial_s, o, 0.0,
                   ki.data(), o);
      // out -= kv @ oo^T + vo @ ki^T
      rm_gemm_host(false, true, v, o, o, -1.0, kv.data(), o, oo_a, obs, 1.0,
                   out_s, o);
      rm_gemm_host(false, true, v, o, o, -1.0, vo_a, obs, ki.data(), o, 1.0,
                   out_s, o);
    }
  };

  apply_spin(oa, va, df_a, gap_a, trial_a, out_a);
  if (!is_rhf) {
    apply_spin(ob, vb, df_b, gap_b, trial_b, out_b);
  }
  if ((oa > 0 && !is_finite_buffer(out_a, Size(va) * oa)) ||
      (!is_rhf && ob > 0 && !is_finite_buffer(out_b, Size(vb) * ob))) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_cabs_nuclear_weights_host_reference(
    int oa, int ob, int obs, int aux, int nao, int same_spin, int q_chunk_hint,
    const double* whitener, const double* c_a, const double* c_b,
    const double* df_a, const double* df_b, const double* p_a,
    const double* p_b, const double* sm_a, const double* sm_b,
    double* bar_ds_out, double* bar_dh_out, double* bar_dr_out,
    double* bar_dj_out) {
  try {
  (void)q_chunk_hint;
  if (obs <= 0 || aux <= 0 || nao < obs || oa < 0 || oa > obs || ob < 0 ||
      ob > obs) {
    return kStatusInvalidArg;
  }
  if (same_spin && oa != ob) return kStatusInvalidArg;
  if (whitener == nullptr || c_a == nullptr || df_a == nullptr ||
      p_a == nullptr || sm_a == nullptr || bar_ds_out == nullptr ||
      bar_dh_out == nullptr || bar_dr_out == nullptr ||
      bar_dj_out == nullptr) {
    return kStatusInvalidArg;
  }
  if (!same_spin && (c_b == nullptr || df_b == nullptr || p_b == nullptr ||
                     sm_b == nullptr)) {
    return kStatusInvalidArg;
  }

  const Size qq = Size(aux) * aux;
  const Size nn_obs = Size(obs) * obs;
  const Size nn_ao = Size(nao) * nao;
  const Size n_c = Size(nao) * obs;
  if (!is_finite_buffer(whitener, qq) || !is_finite_buffer(c_a, n_c) ||
      !is_finite_buffer(df_a, Size(aux) * nn_obs) ||
      !is_finite_buffer(p_a, nn_obs) || !is_finite_buffer(sm_a, nn_obs)) {
    return kStatusNonFinite;
  }
  if (!same_spin &&
      (!is_finite_buffer(c_b, n_c) ||
       !is_finite_buffer(df_b, Size(aux) * nn_obs) ||
       !is_finite_buffer(p_b, nn_obs) || !is_finite_buffer(sm_b, nn_obs))) {
    return kStatusNonFinite;
  }

  std::fill_n(bar_ds_out, nn_ao, 0.0);
  std::fill_n(bar_dh_out, nn_ao, 0.0);
  std::fill_n(bar_dr_out, Size(aux) * nn_ao, 0.0);
  std::fill_n(bar_dj_out, qq, 0.0);

  // 1. charge[a] and bar_dcharge[a]
  std::vector<double> charge(Size(aux), 0.0);
  std::vector<double> bar_dcharge(Size(aux), 0.0);
  auto accum_charges = [&](int o, double scale, const double* df_s,
                           const double* p_s) {
    for (int a = 0; a < aux; ++a) {
      const double* b_a = df_s + Size(a) * nn_obs;
      double d_dot = 0.0;
      for (Size idx = 0; idx < nn_obs; ++idx) {
        d_dot += p_s[idx] * b_a[idx];
      }
      bar_dcharge[a] += scale * d_dot;
      if (o > 0) {
        double tr = 0.0;
        for (int i = 0; i < o; ++i) tr += b_a[Size(i) * obs + i];
        charge[a] += scale * tr;
      }
    }
  };
  if (same_spin) {
    accum_charges(oa, 2.0, df_a, p_a);
  } else {
    accum_charges(oa, 1.0, df_a, p_a);
    accum_charges(ob, 1.0, df_b, p_b);
  }
  std::vector<double> bar_metric_charge(static_cast<Size>(aux), 0.0);
  for (int a = 0; a < aux; ++a) bar_metric_charge[a] = -bar_dcharge[a];

  // 2. Complete bar_sm and assemble bar_dh, bar_ds
  auto accum_2d = [&](int o, double scale, const double* c_s,
                      const double* df_s, const double* p_s,
                      const double* sm_s) {
    std::vector<double> sm_work(sm_s, sm_s + nn_obs);
    if (o > 0) {
      std::vector<double> pb(Size(obs) * o);
      std::vector<double> s_ex(Size(o) * o);
      for (int a = 0; a < aux; ++a) {
        const double* b_a = df_s + Size(a) * nn_obs;
        // pb[obs, o] = p_s^T @ b_a[:, :o]
        rm_gemm_host(true, false, obs, o, obs, 1.0, p_s, obs, b_a, obs, 0.0,
                     pb.data(), o);
        // s_ex[o, o] = pb^T @ b_a[:, :o]
        rm_gemm_host(true, false, o, o, obs, 1.0, pb.data(), o, b_a, obs, 0.0,
                     s_ex.data(), o);
        const double mc_a = bar_metric_charge[a];
        for (int i = 0; i < o; ++i) {
          for (int j = 0; j < o; ++j) {
            sm_work[Size(i) * obs + j] +=
                mc_a * b_a[Size(i) * obs + j] + s_ex[Size(i) * o + j];
          }
        }
      }
    }
    std::vector<double> tmp(n_c);
    // bar_dh += scale * (c_s @ p_s @ c_s^T)
    rm_gemm_host(false, false, nao, obs, obs, 1.0, c_s, obs, p_s, obs, 0.0,
                 tmp.data(), obs);
    rm_gemm_host(false, true, nao, nao, obs, scale, tmp.data(), obs, c_s, obs,
                 1.0, bar_dh_out, nao);
    // bar_ds += scale * (c_s @ sm_work @ c_s^T)
    rm_gemm_host(false, false, nao, obs, obs, 1.0, c_s, obs, sm_work.data(),
                 obs, 0.0, tmp.data(), obs);
    rm_gemm_host(false, true, nao, nao, obs, scale, tmp.data(), obs, c_s, obs,
                 1.0, bar_ds_out, nao);
  };
  if (same_spin) {
    accum_2d(oa, 2.0, c_a, df_a, p_a, sm_a);
  } else {
    accum_2d(oa, 1.0, c_a, df_a, p_a, sm_a);
    accum_2d(ob, 1.0, c_b, df_b, p_b, sm_b);
  }
  for (int i = 0; i < nao; ++i) {
    for (int j = i + 1; j < nao; ++j) {
      const double h_avg =
          0.5 * (bar_dh_out[Size(i) * nao + j] + bar_dh_out[Size(j) * nao + i]);
      bar_dh_out[Size(i) * nao + j] = h_avg;
      bar_dh_out[Size(j) * nao + i] = h_avg;
      const double s_avg =
          0.5 * (bar_ds_out[Size(i) * nao + j] + bar_ds_out[Size(j) * nao + i]);
      bar_ds_out[Size(i) * nao + j] = s_avg;
      bar_ds_out[Size(j) * nao + i] = s_avg;
    }
  }

  // 3. bar_dB -> bar_M and ao_db_sym -> bar_dr and bar_dj
  std::vector<double> bar_m(qq, 0.0);
  std::vector<double> ao_db_sym(Size(aux) * nn_ao, 0.0);
  auto accum_3c = [&](int o, double scale, const double* c_s,
                      const double* df_s, const double* p_s) {
    std::vector<double> psym(nn_obs);
    for (int p = 0; p < obs; ++p) {
      for (int q_i = 0; q_i < obs; ++q_i) {
        psym[Size(p) * obs + q_i] =
            p_s[Size(p) * obs + q_i] + p_s[Size(q_i) * obs + p];
      }
    }
    std::vector<double> bar_db(Size(aux) * nn_obs);
    std::vector<double> ex(Size(obs) * std::max(1, o));
    std::vector<double> tmp_c(n_c);
    std::vector<double> ao_a(nn_ao);
    for (int a = 0; a < aux; ++a) {
      double* db_a = bar_db.data() + Size(a) * nn_obs;
      const double ch_a = charge[a];
      for (Size idx = 0; idx < nn_obs; ++idx) {
        db_a[idx] = ch_a * p_s[idx];
      }
      if (o > 0) {
        const double bd_a = bar_dcharge[a];
        for (int i = 0; i < o; ++i) {
          db_a[Size(i) * obs + i] += bd_a;
        }
        const double* b_a = df_s + Size(a) * nn_obs;
        rm_gemm_host(false, false, obs, o, obs, 1.0, psym.data(), obs, b_a, obs,
                     0.0, ex.data(), o);
        for (int p = 0; p < obs; ++p) {
          double* db_row = db_a + Size(p) * obs;
          const double* ex_row = ex.data() + Size(p) * o;
          for (int i = 0; i < o; ++i) {
            db_row[i] -= ex_row[i];
          }
        }
      }
      // ao_a = c_s @ db_a @ c_s^T
      rm_gemm_host(false, false, nao, obs, obs, 1.0, c_s, obs, db_a, obs, 0.0,
                   tmp_c.data(), obs);
      rm_gemm_host(false, true, nao, nao, obs, 1.0, tmp_c.data(), obs, c_s, obs,
                   0.0, ao_a.data(), nao);
      double* dst_a = ao_db_sym.data() + Size(a) * nn_ao;
      const double half_s = 0.5 * scale;
      for (int m = 0; m < nao; ++m) {
        for (int v = 0; v < nao; ++v) {
          dst_a[Size(m) * nao + v] +=
              half_s * (ao_a[Size(m) * nao + v] + ao_a[Size(v) * nao + m]);
        }
      }
    }
    // bar_m += (-0.5 * scale) * bar_db_flat @ df_s_flat^T
    rm_gemm_host(false, true, aux, aux, static_cast<int>(nn_obs), -0.5 * scale,
                 bar_db.data(), static_cast<int>(nn_obs), df_s,
                 static_cast<int>(nn_obs), 1.0, bar_m.data(), aux);
  };
  if (same_spin) {
    accum_3c(oa, 2.0, c_a, df_a, p_a);
  } else {
    accum_3c(oa, 1.0, c_a, df_a, p_a);
    accum_3c(ob, 1.0, c_b, df_b, p_b);
  }

  // bar_dr_out[aux, nao*nao] = whitener^T @ ao_db_sym[aux, nao*nao]
  rm_gemm_host(true, false, aux, static_cast<int>(nn_ao), aux, 1.0, whitener,
               aux, ao_db_sym.data(), static_cast<int>(nn_ao), 0.0, bar_dr_out,
               static_cast<int>(nn_ao));

  // bar_dj_out = sym(whitener^T @ bar_m @ whitener)
  std::vector<double> tmp_m(qq);
  std::vector<double> dj_raw(qq);
  rm_gemm_host(false, false, aux, aux, aux, 1.0, bar_m.data(), aux, whitener,
               aux, 0.0, tmp_m.data(), aux);
  rm_gemm_host(true, false, aux, aux, aux, 1.0, whitener, aux, tmp_m.data(),
               aux, 0.0, dj_raw.data(), aux);
  for (int a = 0; a < aux; ++a) {
    for (int b = 0; b < aux; ++b) {
      bar_dj_out[Size(a) * aux + b] =
          0.5 * (dj_raw[Size(a) * aux + b] + dj_raw[Size(b) * aux + a]);
    }
  }
  if (!is_finite_buffer(bar_ds_out, nn_ao) ||
      !is_finite_buffer(bar_dh_out, nn_ao) ||
      !is_finite_buffer(bar_dr_out, Size(aux) * nn_ao) ||
      !is_finite_buffer(bar_dj_out, qq)) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

#else  // CUDA backend

extern "C" int gqc_f12_ao_mo_project_fwd_cuda(
    int nc, int ext, int left_dim, int aux, int q_chunk_hint,
    const double* coeffs, const double* ao, double* out) {
  try {
  if (nc <= 0 || ext <= 0 || ext > nc || left_dim < 0 || left_dim > ext ||
      aux <= 0) {
    return kStatusInvalidArg;
  }
  if (left_dim == 0) {
    return kStatusOk;
  }
  if (coeffs == nullptr || ao == nullptr || out == nullptr) {
    return kStatusInvalidArg;
  }
  if (!is_finite_buffer(coeffs, Size(nc) * ext) ||
      !is_finite_buffer(ao, Size(aux) * nc * nc)) {
    return kStatusNonFinite;
  }

  const Size coeff_elems = Size(nc) * ext;
  const Size ao_slice = Size(nc) * nc;
  const Size tmp_slice = Size(left_dim) * nc;
  const Size out_slice = Size(left_dim) * ext;

  const Size fixed_bytes = coeff_elems * sizeof(double);
  const Size per_q_bytes = (ao_slice + tmp_slice + out_slice) * sizeof(double);
  const int q_chunk =
      choose_q_chunk(aux, left_dim, fixed_bytes, per_q_bytes, q_chunk_hint);
  if (q_chunk <= 0) {
    return kStatusBudgetExceeded;
  }

  CudaStreamGuard sg;
  if (sg.init() != cudaSuccess) return kStatusCudaError;
  CublasGuard cg;
  if (cg.init(sg.stream) != CUBLAS_STATUS_SUCCESS) return kStatusCublasError;

  // Persistent allocations outside the Q-tile loop: ZERO cudaMalloc/cudaFree inside loop.
  DeviceBuf d_coeffs, d_ao_tile, d_tmp_tile, d_out_tile;
  if (d_coeffs.alloc(coeff_elems) != cudaSuccess ||
      d_ao_tile.alloc(Size(q_chunk) * ao_slice) != cudaSuccess ||
      d_tmp_tile.alloc(Size(q_chunk) * tmp_slice) != cudaSuccess ||
      d_out_tile.alloc(Size(q_chunk) * out_slice) != cudaSuccess) {
    return kStatusCudaError;
  }

  // Upload coefficients once and keep device-resident across all Q-tiles.
  if (cudaMemcpyAsync(d_coeffs.ptr, coeffs, coeff_elems * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }

  for (int q0 = 0; q0 < aux; q0 += q_chunk) {
    const int b = std::min(q_chunk, aux - q0);
    const Size b_ao = Size(b) * ao_slice;
    const Size b_out = Size(b) * out_slice;

    if (cudaMemcpyAsync(d_ao_tile.ptr, ao + Size(q0) * ao_slice,
                        b_ao * sizeof(double), cudaMemcpyHostToDevice,
                        sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }

    // Step 1: T[q, left_dim, nc] = C[:, :left_dim]^T @ AO[q, nc, nc]
    //         strideA = 0 reuses L2-resident C across all b slices in one launch.
    if (rm_gemm_strided_batched_cuda(
            cg.handle, true, false, left_dim, nc, nc, 1.0, d_coeffs.ptr, ext,
            0LL, d_ao_tile.ptr, nc, static_cast<long long>(ao_slice), 0.0,
            d_tmp_tile.ptr, nc, static_cast<long long>(tmp_slice), b) !=
        CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }

    // Step 2: M_flat[b * left_dim, ext] = T_flat[b * left_dim, nc] @ C[nc, ext]
    //         Single large DGEMM saturates FP64 throughput even when left_dim == occ.
    if (rm_gemm_cuda(cg.handle, false, false, b * left_dim, ext, nc, 1.0,
                     d_tmp_tile.ptr, nc, d_coeffs.ptr, ext, 0.0,
                     d_out_tile.ptr, ext) != CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }

    if (cudaMemcpyAsync(out + Size(q0) * out_slice, d_out_tile.ptr,
                        b_out * sizeof(double), cudaMemcpyDeviceToHost,
                        sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
    if (cudaStreamSynchronize(sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
  }
  if (!is_finite_buffer(out, Size(aux) * out_slice)) return kStatusNonFinite;
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_ao_mo_project_rev_cuda(
    int nc, int ext, int left_dim, int aux, int seed_rows, int tc_rows,
    int mode, int accumulate_bar_ao, int q_chunk_hint, double seed_scale,
    const double* coeffs, const double* ao, const double* seed,
    const double* tc_bar, double* bar_coeffs, double* bar_ao) {
  try {
  if (nc <= 0 || ext <= 0 || ext > nc || left_dim < 0 || left_dim > ext ||
      aux <= 0 || seed_rows < 0 || seed_rows > left_dim || tc_rows < 0 ||
      tc_rows > left_dim || mode < 0 || mode > 2) {
    return kStatusInvalidArg;
  }
  if (!std::isfinite(seed_scale)) {
    return kStatusNonFinite;
  }
  const bool want_coeffs = (mode == 0 || mode == 1) && bar_coeffs != nullptr;
  const bool want_ao = (mode == 0 || mode == 2) && bar_ao != nullptr;
  const int active_rows = std::max(seed_rows, tc_rows);
  if (active_rows == 0 || (!want_coeffs && !want_ao)) {
    if (want_ao && !accumulate_bar_ao) {
      std::fill_n(bar_ao, Size(aux) * nc * nc, 0.0);
    }
    return kStatusOk;
  }
  if (coeffs == nullptr) return kStatusInvalidArg;
  if (want_coeffs && ao == nullptr) return kStatusInvalidArg;
  if (seed_rows > 0 && seed == nullptr) return kStatusInvalidArg;
  if (tc_rows > 0 && tc_bar == nullptr) return kStatusInvalidArg;

  if (!is_finite_buffer(coeffs, Size(nc) * ext)) return kStatusNonFinite;
  if (want_coeffs && !is_finite_buffer(ao, Size(aux) * nc * nc)) {
    return kStatusNonFinite;
  }
  if (seed_rows > 0 && !is_finite_buffer(seed, Size(aux) * seed_rows * ext)) {
    return kStatusNonFinite;
  }
  if (tc_rows > 0 && !is_finite_buffer(tc_bar, Size(aux) * tc_rows * ext)) {
    return kStatusNonFinite;
  }

  const Size coeff_elems = Size(nc) * ext;
  const Size ao_slice = Size(nc) * nc;
  const Size s_slice = Size(active_rows) * ext;
  const Size u_slice = Size(active_rows) * nc;
  const Size seed_slice = Size(seed_rows) * ext;
  const Size tc_slice = Size(tc_rows) * ext;

  // d_ao_tile is reused for d_bar_ao_tile in mode == 0 (bar_ao is computed and
  // copied D2H before ao[q0..q1] is uploaded for bar_coeffs).
  const Size fixed_bytes =
      (coeff_elems + (want_coeffs ? coeff_elems : 0)) * sizeof(double);
  const Size per_q_elems =
      ao_slice + s_slice + u_slice + (want_coeffs ? u_slice : 0) + tc_slice;
  const Size per_q_bytes = per_q_elems * sizeof(double);
  const int q_chunk =
      choose_q_chunk(aux, active_rows, fixed_bytes, per_q_bytes, q_chunk_hint);
  if (q_chunk <= 0) {
    return kStatusBudgetExceeded;
  }

  CudaStreamGuard sg;
  if (sg.init() != cudaSuccess) return kStatusCudaError;
  CublasGuard cg;
  if (cg.init(sg.stream) != CUBLAS_STATUS_SUCCESS) return kStatusCublasError;

  // Persistent allocations outside the Q-tile loop.
  DeviceBuf d_coeffs, d_bar_coeffs, d_ao_tile, d_s_tile, d_u_tile, d_l_tile,
      d_tc_tile;
  if (d_coeffs.alloc(coeff_elems) != cudaSuccess ||
      d_ao_tile.alloc(Size(q_chunk) * ao_slice) != cudaSuccess ||
      d_s_tile.alloc(Size(q_chunk) * s_slice) != cudaSuccess ||
      d_u_tile.alloc(Size(q_chunk) * u_slice) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (want_coeffs) {
    if (d_bar_coeffs.alloc(coeff_elems) != cudaSuccess ||
        d_l_tile.alloc(Size(q_chunk) * u_slice) != cudaSuccess) {
      return kStatusCudaError;
    }
    if (cudaMemsetAsync(d_bar_coeffs.ptr, 0, coeff_elems * sizeof(double),
                        sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
  }
  if (tc_rows > 0) {
    if (d_tc_tile.alloc(Size(q_chunk) * tc_slice) != cudaSuccess) {
      return kStatusCudaError;
    }
  }

  // Upload coefficients once across all Q-tiles.
  if (cudaMemcpyAsync(d_coeffs.ptr, coeffs, coeff_elems * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }

  // Bounded host staging buffer only when accumulating onto pre-existing bar_ao
  // (avoids H2D transfer of old bar_ao, halving PCIe traffic).
  std::vector<double> h_bar_ao_stage;
  if (want_ao && accumulate_bar_ao) {
    h_bar_ao_stage.resize(Size(q_chunk) * ao_slice);
  }

  for (int q0 = 0; q0 < aux; q0 += q_chunk) {
    const int b = std::min(q_chunk, aux - q0);
    const Size b_s = Size(b) * s_slice;
    const Size b_ao = Size(b) * ao_slice;

    // 1. Construct effective seed S[b, active_rows, ext] on device.
    if (seed_rows < active_rows) {
      if (cudaMemsetAsync(d_s_tile.ptr, 0, b_s * sizeof(double), sg.stream) !=
          cudaSuccess) {
        return kStatusCudaError;
      }
    }
    if (seed_rows == active_rows) {
      if (cudaMemcpyAsync(d_s_tile.ptr, seed + Size(q0) * seed_slice,
                          b_s * sizeof(double), cudaMemcpyHostToDevice,
                          sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    } else if (seed_rows > 0) {
      if (cudaMemcpy2DAsync(d_s_tile.ptr, s_slice * sizeof(double),
                            seed + Size(q0) * seed_slice,
                            seed_slice * sizeof(double),
                            seed_slice * sizeof(double), static_cast<size_t>(b),
                            cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    }

    if (tc_rows > 0) {
      if (cudaMemcpyAsync(d_tc_tile.ptr, tc_bar + Size(q0) * tc_slice,
                          Size(b) * tc_slice * sizeof(double),
                          cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
      add_tc_and_scale_kernel<<<grid_blocks(b_s), 256, 0, sg.stream>>>(
          d_s_tile.ptr, d_tc_tile.ptr, b, active_rows, tc_rows, ext,
          seed_scale);
      if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
    } else if (seed_scale != 1.0) {
      scale_buffer_kernel<<<grid_blocks(b_s), 256, 0, sg.stream>>>(
          d_s_tile.ptr, b_s, seed_scale);
      if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
    }

    // 2. U_flat[b * active_rows, nc] = S_flat[b * active_rows, ext] @ C^T[ext, nc]
    //    Reused by BOTH bar_ao and Delta1 bar_coeffs!
    if (rm_gemm_cuda(cg.handle, false, true, b * active_rows, nc, ext, 1.0,
                     d_s_tile.ptr, ext, d_coeffs.ptr, ext, 0.0, d_u_tile.ptr,
                     nc) != CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }

    // 3. Branch A: bar_ao[q] = C[:, :active_rows] @ U[q] (computed into d_ao_tile).
    if (want_ao) {
      if (rm_gemm_strided_batched_cuda(
              cg.handle, false, false, nc, nc, active_rows, 1.0, d_coeffs.ptr,
              ext, 0LL, d_u_tile.ptr, nc, static_cast<long long>(u_slice), 0.0,
              d_ao_tile.ptr, nc, static_cast<long long>(ao_slice), b) !=
          CUBLAS_STATUS_SUCCESS) {
        return kStatusCublasError;
      }
      double* dst_host = accumulate_bar_ao ? h_bar_ao_stage.data()
                                           : (bar_ao + Size(q0) * ao_slice);
      if (cudaMemcpyAsync(dst_host, d_ao_tile.ptr, b_ao * sizeof(double),
                          cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
      if (cudaStreamSynchronize(sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
      if (accumulate_bar_ao) {
        double* out_slice = bar_ao + Size(q0) * ao_slice;
        for (Size i = 0; i < b_ao; ++i) {
          out_slice[i] += h_bar_ao_stage[i];
        }
      }
    }

    // 4. Branch B: bar_coeffs += sum_q (AO[q] @ U[q]^T) + (C_act^T @ AO[q])^T @ S[q]
    if (want_coeffs) {
      if (cudaMemcpyAsync(d_ao_tile.ptr, ao + Size(q0) * ao_slice,
                          b_ao * sizeof(double), cudaMemcpyHostToDevice,
                          sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }

      // B1a: Delta1_q[q, nc, active_rows] = AO[q, nc, nc] @ U[q, active_rows, nc]^T
      //      Temporarily stored in d_l_tile (identical element count b * nc * active_rows).
      if (rm_gemm_strided_batched_cuda(
              cg.handle, false, true, nc, active_rows, nc, 1.0, d_ao_tile.ptr,
              nc, static_cast<long long>(ao_slice), d_u_tile.ptr, nc,
              static_cast<long long>(u_slice), 0.0, d_l_tile.ptr, active_rows,
              static_cast<long long>(u_slice), b) != CUBLAS_STATUS_SUCCESS) {
        return kStatusCublasError;
      }

      // B1b: Deterministic Q-reduction into d_bar_coeffs[:, :active_rows].
      reduce_q_add_strided_kernel<<<grid_blocks(u_slice), 256, 0, sg.stream>>>(
          d_l_tile.ptr, d_bar_coeffs.ptr, b, nc, active_rows, ext);
      if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;

      // B2a: L[q, active_rows, nc] = C[:, :active_rows]^T @ AO[q, nc, nc]
      if (rm_gemm_strided_batched_cuda(
              cg.handle, true, false, active_rows, nc, nc, 1.0, d_coeffs.ptr,
              ext, 0LL, d_ao_tile.ptr, nc, static_cast<long long>(ao_slice),
              0.0, d_l_tile.ptr, nc, static_cast<long long>(u_slice), b) !=
          CUBLAS_STATUS_SUCCESS) {
        return kStatusCublasError;
      }

      // B2b: d_bar_coeffs[nc, ext] += L_flat^T[nc, b*active_rows] @ S_flat[b*active_rows, ext]
      if (rm_gemm_cuda(cg.handle, true, false, nc, ext, b * active_rows, 1.0,
                       d_l_tile.ptr, nc, d_s_tile.ptr, ext, 1.0,
                       d_bar_coeffs.ptr, ext) != CUBLAS_STATUS_SUCCESS) {
        return kStatusCublasError;
      }

      // Ensure host buffers for this tile are consumed before the next tile overwrites
      // device buffers (single sync per tile when want_coeffs is active).
      if (cudaStreamSynchronize(sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    }
  }

  if (want_coeffs) {
    std::vector<double> h_delta(coeff_elems);
    if (cudaMemcpyAsync(h_delta.data(), d_bar_coeffs.ptr,
                        coeff_elems * sizeof(double), cudaMemcpyDeviceToHost,
                        sg.stream) != cudaSuccess ||
        cudaStreamSynchronize(sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
    for (Size i = 0; i < coeff_elems; ++i) {
      bar_coeffs[i] += h_delta[i];
    }
  }
  if (want_coeffs && !is_finite_buffer(bar_coeffs, coeff_elems)) {
    return kStatusNonFinite;
  }
  if (want_ao && !is_finite_buffer(bar_ao, Size(aux) * ao_slice)) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_ao_mo_bilin_fwd_cuda(
    int nc, int n_left, int n_right, const double* A, const double* C_left,
    const double* C_right, double* out) {
  try {
  if (nc <= 0 || n_left <= 0 || n_right <= 0 || A == nullptr ||
      C_left == nullptr || C_right == nullptr || out == nullptr) {
    return kStatusInvalidArg;
  }
  const Size a_n = Size(nc) * nc;
  const Size cl_n = Size(nc) * n_left;
  const Size cr_n = Size(nc) * n_right;
  const Size out_n = Size(n_left) * n_right;
  if (!is_finite_buffer(A, a_n) || !is_finite_buffer(C_left, cl_n) ||
      !is_finite_buffer(C_right, cr_n)) {
    return kStatusNonFinite;
  }
  const Size total_bytes = (a_n + cl_n + cr_n + cr_n + out_n) * sizeof(double);
  if (total_bytes > kMaxDeviceWorkspaceBytes) {
    return kStatusBudgetExceeded;
  }

  CudaStreamGuard sg;
  if (sg.init() != cudaSuccess) return kStatusCudaError;
  CublasGuard cg;
  if (cg.init(sg.stream) != CUBLAS_STATUS_SUCCESS) return kStatusCublasError;

  DeviceBuf d_A, d_Cl, d_Cr, d_tmp, d_out;
  if (d_A.alloc(a_n) != cudaSuccess || d_Cl.alloc(cl_n) != cudaSuccess ||
      d_Cr.alloc(cr_n) != cudaSuccess || d_tmp.alloc(cr_n) != cudaSuccess ||
      d_out.alloc(out_n) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (cudaMemcpyAsync(d_A.ptr, A, a_n * sizeof(double), cudaMemcpyHostToDevice,
                      sg.stream) != cudaSuccess ||
      cudaMemcpyAsync(d_Cl.ptr, C_left, cl_n * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
      cudaMemcpyAsync(d_Cr.ptr, C_right, cr_n * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }
  // tmp = A @ C_right
  if (rm_gemm_cuda(cg.handle, false, false, nc, n_right, nc, 1.0, d_A.ptr, nc,
                   d_Cr.ptr, n_right, 0.0, d_tmp.ptr, n_right) !=
      CUBLAS_STATUS_SUCCESS) {
    return kStatusCublasError;
  }
  // out = C_left^T @ tmp
  if (rm_gemm_cuda(cg.handle, true, false, n_left, n_right, nc, 1.0, d_Cl.ptr,
                   n_left, d_tmp.ptr, n_right, 0.0, d_out.ptr, n_right) !=
      CUBLAS_STATUS_SUCCESS) {
    return kStatusCublasError;
  }
  if (cudaMemcpyAsync(out, d_out.ptr, out_n * sizeof(double),
                      cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess ||
      cudaStreamSynchronize(sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (!is_finite_buffer(out, out_n)) return kStatusNonFinite;
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_ao_mo_bilin_rev_cuda(
    int nc, int n_left, int n_right, const double* A, const double* C_left,
    const double* C_right, const double* bar_out, double* bar_A,
    double* bar_C_left, double* bar_C_right) {
  try {
  if (nc <= 0 || n_left <= 0 || n_right <= 0 || A == nullptr ||
      C_left == nullptr || C_right == nullptr || bar_out == nullptr) {
    return kStatusInvalidArg;
  }
  const Size a_n = Size(nc) * nc;
  const Size cl_n = Size(nc) * n_left;
  const Size cr_n = Size(nc) * n_right;
  const Size bo_n = Size(n_left) * n_right;
  if (!is_finite_buffer(A, a_n) || !is_finite_buffer(C_left, cl_n) ||
      !is_finite_buffer(C_right, cr_n) || !is_finite_buffer(bar_out, bo_n)) {
    return kStatusNonFinite;
  }
  if (bar_A == nullptr && bar_C_left == nullptr && bar_C_right == nullptr) {
    return kStatusOk;
  }

  const Size max_scratch = std::max({a_n, cl_n, cr_n});
  const Size total_bytes =
      (a_n + cl_n + cr_n + bo_n + 2 * max_scratch) * sizeof(double);
  if (total_bytes > kMaxDeviceWorkspaceBytes) {
    return kStatusBudgetExceeded;
  }

  CudaStreamGuard sg;
  if (sg.init() != cudaSuccess) return kStatusCudaError;
  CublasGuard cg;
  if (cg.init(sg.stream) != CUBLAS_STATUS_SUCCESS) return kStatusCublasError;

  DeviceBuf d_A, d_Cl, d_Cr, d_bo, d_tmp, d_out;
  if (d_A.alloc(a_n) != cudaSuccess || d_Cl.alloc(cl_n) != cudaSuccess ||
      d_Cr.alloc(cr_n) != cudaSuccess || d_bo.alloc(bo_n) != cudaSuccess ||
      d_tmp.alloc(max_scratch) != cudaSuccess ||
      d_out.alloc(max_scratch) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (cudaMemcpyAsync(d_A.ptr, A, a_n * sizeof(double), cudaMemcpyHostToDevice,
                      sg.stream) != cudaSuccess ||
      cudaMemcpyAsync(d_Cl.ptr, C_left, cl_n * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
      cudaMemcpyAsync(d_Cr.ptr, C_right, cr_n * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
      cudaMemcpyAsync(d_bo.ptr, bar_out, bo_n * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }

  std::vector<double> h_stage(max_scratch);

  if (bar_A != nullptr) {
    // d_tmp[n_left, nc] = bar_out @ C_right^T
    if (rm_gemm_cuda(cg.handle, false, true, n_left, nc, n_right, 1.0,
                     d_bo.ptr, n_right, d_Cr.ptr, n_right, 0.0, d_tmp.ptr,
                     nc) != CUBLAS_STATUS_SUCCESS ||
        // d_out[nc, nc] = C_left @ d_tmp
        rm_gemm_cuda(cg.handle, false, false, nc, nc, n_left, 1.0, d_Cl.ptr,
                     n_left, d_tmp.ptr, nc, 0.0, d_out.ptr, nc) !=
            CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
    if (cudaMemcpyAsync(h_stage.data(), d_out.ptr, a_n * sizeof(double),
                        cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess ||
        cudaStreamSynchronize(sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
    for (Size i = 0; i < a_n; ++i) bar_A[i] += h_stage[i];
  }

  if (bar_C_left != nullptr) {
    // d_tmp[nc, n_right] = A @ C_right
    if (rm_gemm_cuda(cg.handle, false, false, nc, n_right, nc, 1.0, d_A.ptr,
                     nc, d_Cr.ptr, n_right, 0.0, d_tmp.ptr, n_right) !=
            CUBLAS_STATUS_SUCCESS ||
        // d_out[nc, n_left] = d_tmp @ bar_out^T
        rm_gemm_cuda(cg.handle, false, true, nc, n_left, n_right, 1.0,
                     d_tmp.ptr, n_right, d_bo.ptr, n_right, 0.0, d_out.ptr,
                     n_left) != CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
    if (cudaMemcpyAsync(h_stage.data(), d_out.ptr, cl_n * sizeof(double),
                        cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess ||
        cudaStreamSynchronize(sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
    for (Size i = 0; i < cl_n; ++i) bar_C_left[i] += h_stage[i];
  }

  if (bar_C_right != nullptr) {
    // d_tmp[nc, n_left] = A^T @ C_left
    if (rm_gemm_cuda(cg.handle, true, false, nc, n_left, nc, 1.0, d_A.ptr, nc,
                     d_Cl.ptr, n_left, 0.0, d_tmp.ptr, n_left) !=
            CUBLAS_STATUS_SUCCESS ||
        // d_out[nc, n_right] = d_tmp @ bar_out
        rm_gemm_cuda(cg.handle, false, false, nc, n_right, n_left, 1.0,
                     d_tmp.ptr, n_left, d_bo.ptr, n_right, 0.0, d_out.ptr,
                     n_right) != CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
    if (cudaMemcpyAsync(h_stage.data(), d_out.ptr, cr_n * sizeof(double),
                        cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess ||
        cudaStreamSynchronize(sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
    for (Size i = 0; i < cr_n; ++i) bar_C_right[i] += h_stage[i];
  }
  if ((bar_A != nullptr && !is_finite_buffer(bar_A, a_n)) ||
      (bar_C_left != nullptr && !is_finite_buffer(bar_C_left, cl_n)) ||
      (bar_C_right != nullptr && !is_finite_buffer(bar_C_right, cr_n))) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_coulomb_fit_fwd_cuda(
    int aux, int pairs, const double* whitener, const double* raw,
    double* fit) {
  try {
  if (aux <= 0 || pairs < 0) return kStatusInvalidArg;
  if (pairs == 0) return kStatusOk;
  if (whitener == nullptr || raw == nullptr || fit == nullptr) {
    return kStatusInvalidArg;
  }
  const Size aa = Size(aux) * aux;
  if (!is_finite_buffer(whitener, aa) ||
      !is_finite_buffer(raw, Size(aux) * pairs)) {
    return kStatusNonFinite;
  }

  const Size fixed_bytes = aa * sizeof(double);
  const Size per_col_bytes = Size(2) * aux * sizeof(double);
  if (fixed_bytes + per_col_bytes > kMaxDeviceWorkspaceBytes) {
    return kStatusBudgetExceeded;
  }
  const Size avail = kFitWorkspaceBytes > fixed_bytes
                         ? (kFitWorkspaceBytes - fixed_bytes)
                         : 0;
  const int p_chunk = static_cast<int>(
      std::min<Size>(Size(pairs), std::max<Size>(1, avail / per_col_bytes)));

  CudaStreamGuard sg;
  if (sg.init() != cudaSuccess) return kStatusCudaError;
  CublasGuard cg;
  if (cg.init(sg.stream) != CUBLAS_STATUS_SUCCESS) return kStatusCublasError;

  DeviceBuf d_W, d_in, d_tmp;
  if (d_W.alloc(aa) != cudaSuccess ||
      d_in.alloc(Size(aux) * p_chunk) != cudaSuccess ||
      d_tmp.alloc(Size(aux) * p_chunk) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (cudaMemcpyAsync(d_W.ptr, whitener, aa * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }

  for (int p0 = 0; p0 < pairs; p0 += p_chunk) {
    const int p_len = std::min(p_chunk, pairs - p0);
    if (p_len == pairs) {
      if (cudaMemcpyAsync(d_in.ptr, raw, Size(aux) * pairs * sizeof(double),
                          cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    } else {
      if (cudaMemcpy2DAsync(d_in.ptr, Size(p_len) * sizeof(double), raw + p0,
                            Size(pairs) * sizeof(double),
                            Size(p_len) * sizeof(double),
                            static_cast<size_t>(aux), cudaMemcpyHostToDevice,
                            sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    }

    // tmp = W @ raw_chunk; in = W^T @ tmp
    if (rm_gemm_cuda(cg.handle, false, false, aux, p_len, aux, 1.0, d_W.ptr,
                     aux, d_in.ptr, p_len, 0.0, d_tmp.ptr, p_len) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm_cuda(cg.handle, true, false, aux, p_len, aux, 1.0, d_W.ptr, aux,
                     d_tmp.ptr, p_len, 0.0, d_in.ptr, p_len) !=
            CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }

    if (p_len == pairs) {
      if (cudaMemcpyAsync(fit, d_in.ptr, Size(aux) * pairs * sizeof(double),
                          cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    } else {
      if (cudaMemcpy2DAsync(fit + p0, Size(pairs) * sizeof(double), d_in.ptr,
                            Size(p_len) * sizeof(double),
                            Size(p_len) * sizeof(double),
                            static_cast<size_t>(aux), cudaMemcpyDeviceToHost,
                            sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    }
    if (cudaStreamSynchronize(sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
  }
  if (!is_finite_buffer(fit, Size(aux) * pairs)) return kStatusNonFinite;
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_coulomb_fit_rev_cuda(
    int aux, int pairs, double metric_scale, const double* whitener,
    const double* raw, const double* fit_bar, double* three_center_bar,
    double* metric_bar) {
  try {
  if (aux <= 0 || pairs < 0) return kStatusInvalidArg;
  if (!std::isfinite(metric_scale)) return kStatusNonFinite;
  if (pairs == 0) return kStatusOk;
  if (whitener == nullptr || raw == nullptr || fit_bar == nullptr ||
      three_center_bar == nullptr || metric_bar == nullptr) {
    return kStatusInvalidArg;
  }
  const Size aa = Size(aux) * aux;
  if (!is_finite_buffer(whitener, aa) ||
      !is_finite_buffer(raw, Size(aux) * pairs) ||
      !is_finite_buffer(fit_bar, Size(aux) * pairs)) {
    return kStatusNonFinite;
  }

  const Size fixed_bytes = Size(2) * aa * sizeof(double);
  const Size per_col_bytes = Size(3) * aux * sizeof(double);
  if (fixed_bytes + per_col_bytes > kMaxDeviceWorkspaceBytes) {
    return kStatusBudgetExceeded;
  }
  const Size avail = kFitWorkspaceBytes > fixed_bytes
                         ? (kFitWorkspaceBytes - fixed_bytes)
                         : 0;
  const int p_chunk = static_cast<int>(
      std::min<Size>(Size(pairs), std::max<Size>(1, avail / per_col_bytes)));

  CudaStreamGuard sg;
  if (sg.init() != cudaSuccess) return kStatusCudaError;
  CublasGuard cg;
  if (cg.init(sg.stream) != CUBLAS_STATUS_SUCCESS) return kStatusCublasError;

  DeviceBuf d_W, d_weight, d_fit, d_tcb, d_tmp;
  if (d_W.alloc(aa) != cudaSuccess || d_weight.alloc(aa) != cudaSuccess ||
      d_fit.alloc(Size(aux) * p_chunk) != cudaSuccess ||
      d_tcb.alloc(Size(aux) * p_chunk) != cudaSuccess ||
      d_tmp.alloc(Size(aux) * p_chunk) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (cudaMemcpyAsync(d_W.ptr, whitener, aa * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
      cudaMemsetAsync(d_weight.ptr, 0, aa * sizeof(double), sg.stream) !=
          cudaSuccess) {
    return kStatusCudaError;
  }

  for (int p0 = 0; p0 < pairs; p0 += p_chunk) {
    const int p_len = std::min(p_chunk, pairs - p0);
    if (p_len == pairs) {
      if (cudaMemcpyAsync(d_fit.ptr, raw, Size(aux) * pairs * sizeof(double),
                          cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
          cudaMemcpyAsync(d_tcb.ptr, fit_bar,
                          Size(aux) * pairs * sizeof(double),
                          cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    } else {
      if (cudaMemcpy2DAsync(d_fit.ptr, Size(p_len) * sizeof(double), raw + p0,
                            Size(pairs) * sizeof(double),
                            Size(p_len) * sizeof(double),
                            static_cast<size_t>(aux), cudaMemcpyHostToDevice,
                            sg.stream) != cudaSuccess ||
          cudaMemcpy2DAsync(d_tcb.ptr, Size(p_len) * sizeof(double),
                            fit_bar + p0, Size(pairs) * sizeof(double),
                            Size(p_len) * sizeof(double),
                            static_cast<size_t>(aux), cudaMemcpyHostToDevice,
                            sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    }

    // fit = W^T @ (W @ raw_chunk)
    if (rm_gemm_cuda(cg.handle, false, false, aux, p_len, aux, 1.0, d_W.ptr,
                     aux, d_fit.ptr, p_len, 0.0, d_tmp.ptr, p_len) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm_cuda(cg.handle, true, false, aux, p_len, aux, 1.0, d_W.ptr, aux,
                     d_tmp.ptr, p_len, 0.0, d_fit.ptr, p_len) !=
            CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }

    // three_center_bar = W^T @ (W @ fit_bar_chunk)
    if (rm_gemm_cuda(cg.handle, false, false, aux, p_len, aux, 1.0, d_W.ptr,
                     aux, d_tcb.ptr, p_len, 0.0, d_tmp.ptr, p_len) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm_cuda(cg.handle, true, false, aux, p_len, aux, 1.0, d_W.ptr, aux,
                     d_tmp.ptr, p_len, 0.0, d_tcb.ptr, p_len) !=
            CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }

    // weight += fit_chunk @ three_center_bar_chunk^T
    if (rm_gemm_cuda(cg.handle, false, true, aux, aux, p_len, 1.0, d_fit.ptr,
                     p_len, d_tcb.ptr, p_len, 1.0, d_weight.ptr, aux) !=
        CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }

    if (p_len == pairs) {
      if (cudaMemcpyAsync(three_center_bar, d_tcb.ptr,
                          Size(aux) * pairs * sizeof(double),
                          cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    } else {
      if (cudaMemcpy2DAsync(three_center_bar + p0,
                            Size(pairs) * sizeof(double), d_tcb.ptr,
                            Size(p_len) * sizeof(double),
                            Size(p_len) * sizeof(double),
                            static_cast<size_t>(aux), cudaMemcpyDeviceToHost,
                            sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    }
    if (cudaStreamSynchronize(sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
  }

  std::vector<double> h_weight(aa);
  if (cudaMemcpyAsync(h_weight.data(), d_weight.ptr, aa * sizeof(double),
                      cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess ||
      cudaStreamSynchronize(sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }
  const double half_neg = -0.5 * metric_scale;
  for (int a = 0; a < aux; ++a) {
    for (int b = 0; b < aux; ++b) {
      metric_bar[Size(a) * aux + b] +=
          half_neg *
          (h_weight[Size(a) * aux + b] + h_weight[Size(b) * aux + a]);
    }
  }
  if (!is_finite_buffer(three_center_bar, Size(aux) * pairs) ||
      !is_finite_buffer(metric_bar, aa)) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

// ============================================================================
// CABS Singles E+F dynamic-dimension AOT kernels (CUDA + cuBLAS)
// ============================================================================

extern "C" int gqc_f12_cabs_fock_fwd_cuda(
    int n, int q, int oa, int ob, int occ_rows, int same_occ, int q_chunk_hint,
    const double* hcore, const double* raw, const double* whitener,
    const double* occ_a, const double* occ_b, double* fock_ao_a,
    double* fock_ao_b, double* rho_out, double* raw_a_out, double* fit_a_out,
    double* raw_b_out, double* fit_b_out) {
  try {
  if (n <= 0 || q <= 0 || oa < 0 || oa > n || ob < 0 || ob > n ||
      occ_rows <= 0 || occ_rows > n) {
    return kStatusInvalidArg;
  }
  if (same_occ && oa != ob) return kStatusInvalidArg;
  if (hcore == nullptr || raw == nullptr || whitener == nullptr ||
      fock_ao_a == nullptr || fock_ao_b == nullptr) {
    return kStatusInvalidArg;
  }
  if (oa > 0 && occ_a == nullptr) return kStatusInvalidArg;
  if (!same_occ && ob > 0 && occ_b == nullptr) return kStatusInvalidArg;

  const Size nn = Size(n) * n;
  const Size qq = Size(q) * q;
  if (!is_finite_buffer(hcore, nn) || !is_finite_buffer(raw, Size(q) * nn) ||
      !is_finite_buffer(whitener, qq)) {
    return kStatusNonFinite;
  }
  if (oa > 0 && !is_finite_buffer(occ_a, Size(occ_rows) * oa)) {
    return kStatusNonFinite;
  }
  if (!same_occ && ob > 0 && !is_finite_buffer(occ_b, Size(occ_rows) * ob)) {
    return kStatusNonFinite;
  }

  // Degenerate zero-electron case: F_AO = H_core, rho = 0.
  if (oa == 0 && ob == 0) {
    std::copy_n(hcore, nn, fock_ao_a);
    std::copy_n(hcore, nn, fock_ao_b);
    if (rho_out != nullptr) std::fill_n(rho_out, Size(q), 0.0);
    return kStatusOk;
  }

  const int o_max = std::max(oa, ob);
  const Size max_on = Size(o_max) * n;
  const Size max_qon = Size(q) * max_on;
  if (nn > Size(std::numeric_limits<int>::max()) ||
      max_on > Size(std::numeric_limits<int>::max()) ||
      Size(q) * o_max > Size(std::numeric_limits<int>::max())) {
    return kStatusBudgetExceeded;
  }

  const Size fixed_elems =
      qq + Size(4) * nn + Size(occ_rows) * o_max + Size(q) + Size(2) * max_qon;
  const Size fixed_bytes = fixed_elems * sizeof(double);
  const Size per_q_bytes = (nn + Size(n) * o_max) * sizeof(double);
  const int q_chunk =
      choose_q_chunk(q, n, fixed_bytes, per_q_bytes, q_chunk_hint);
  if (q_chunk <= 0) return kStatusBudgetExceeded;

  CudaStreamGuard sg;
  if (sg.init() != cudaSuccess) return kStatusCudaError;
  CublasGuard cg;
  if (cg.init(sg.stream) != CUBLAS_STATUS_SUCCESS) return kStatusCublasError;

  // Persistent allocations outside all Q-tile loops (ZERO loop cudaMalloc/Free).
  DeviceBuf d_W, d_h, d_J, d_K_a, d_K_b, d_occ, d_rho, d_R, d_tmp, d_raw_tile,
      d_T_tile;
  if (d_W.alloc(qq) != cudaSuccess || d_h.alloc(nn) != cudaSuccess ||
      d_J.alloc(nn) != cudaSuccess || d_K_a.alloc(nn) != cudaSuccess ||
      d_occ.alloc(Size(occ_rows) * o_max) != cudaSuccess ||
      d_rho.alloc(Size(q)) != cudaSuccess || d_R.alloc(max_qon) != cudaSuccess ||
      d_tmp.alloc(max_qon) != cudaSuccess ||
      d_raw_tile.alloc(Size(q_chunk) * nn) != cudaSuccess ||
      d_T_tile.alloc(Size(q_chunk) * n * o_max) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (!same_occ) {
    if (d_K_b.alloc(nn) != cudaSuccess) return kStatusCudaError;
  }

  if (cudaMemcpyAsync(d_W.ptr, whitener, qq * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
      cudaMemcpyAsync(d_h.ptr, hcore, nn * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
      cudaMemsetAsync(d_rho.ptr, 0, Size(q) * sizeof(double), sg.stream) !=
          cudaSuccess ||
      cudaMemsetAsync(d_J.ptr, 0, nn * sizeof(double), sg.stream) !=
          cudaSuccess) {
    return kStatusCudaError;
  }

  auto build_spin_cuda = [&](int o, const double* h_occ, double rho_scale,
                             int accum_rho, double* d_K_target, double* r_out,
                             double* d_out) -> int {
    if (o == 0) {
      if (cudaMemsetAsync(d_K_target, 0, nn * sizeof(double), sg.stream) !=
          cudaSuccess) {
        return kStatusCudaError;
      }
      return kStatusOk;
    }
    const Size on = Size(o) * n;
    const Size qon = Size(q) * on;

    if (cudaMemcpyAsync(d_occ.ptr, h_occ, Size(occ_rows) * o * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }

    // Pass 1: Q-chunked half-transform R[q, o, n] = raw[q, :, :occ_rows] @ occ
    for (int q0 = 0; q0 < q; q0 += q_chunk) {
      const int b = std::min(q_chunk, q - q0);
      if (cudaMemcpyAsync(d_raw_tile.ptr, raw + Size(q0) * nn,
                          Size(b) * nn * sizeof(double), cudaMemcpyHostToDevice,
                          sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
      // Flattened [b*n, occ_rows] @ [occ_rows, o] -> d_T_tile[b*n, o]
      if (rm_gemm_cuda(cg.handle, false, false, b * n, o, occ_rows, 1.0,
                       d_raw_tile.ptr, n, d_occ.ptr, o, 0.0, d_T_tile.ptr,
                       o) != CUBLAS_STATUS_SUCCESS) {
        return kStatusCublasError;
      }
      transpose_bno_to_bon_kernel<<<grid_blocks(Size(b) * on), 256, 0,
                                    sg.stream>>>(
          d_T_tile.ptr, d_R.ptr + Size(q0) * on, b, n, o);
      if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
    }

    // Optional D2H save of R before overwriting d_R with D = W^T @ B.
    if (r_out != nullptr) {
      if (cudaMemcpyAsync(r_out, d_R.ptr, qon * sizeof(double),
                          cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    }

    // B[q, o*n] = W @ R -> d_tmp
    if (rm_gemm_cuda(cg.handle, false, false, q, static_cast<int>(on), q, 1.0,
                     d_W.ptr, q, d_R.ptr, static_cast<int>(on), 0.0, d_tmp.ptr,
                     static_cast<int>(on)) != CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }

    // K[n, n] = B_flat^T @ B_flat where B_flat is [q*o, n]
    if (rm_gemm_cuda(cg.handle, true, false, n, n, q * o, 1.0, d_tmp.ptr, n,
                     d_tmp.ptr, n, 0.0, d_K_target, n) !=
        CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }

    // D[q, o*n] = W^T @ B -> overwrite d_R (stream order ensures r_out D2H finishes first)
    if (rm_gemm_cuda(cg.handle, true, false, q, static_cast<int>(on), q, 1.0,
                     d_W.ptr, q, d_tmp.ptr, static_cast<int>(on), 0.0, d_R.ptr,
                     static_cast<int>(on)) != CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }

    if (d_out != nullptr) {
      if (cudaMemcpyAsync(d_out, d_R.ptr, qon * sizeof(double),
                          cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    }

    cabs_rho_from_fit_kernel<<<q, 256, 0, sg.stream>>>(
        d_R.ptr, d_occ.ptr, d_rho.ptr, q, o, n, occ_rows, rho_scale, accum_rho);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
    return kStatusOk;
  };

  int rc = build_spin_cuda(oa, occ_a, same_occ ? 2.0 : 1.0, 0, d_K_a.ptr,
                           raw_a_out, fit_a_out);
  if (rc != kStatusOk) return rc;
  if (!same_occ) {
    rc = build_spin_cuda(ob, occ_b, 1.0, 1, d_K_b.ptr, raw_b_out, fit_b_out);
    if (rc != kStatusOk) return rc;
  }

  if (rho_out != nullptr) {
    if (cudaMemcpyAsync(rho_out, d_rho.ptr, Size(q) * sizeof(double),
                        cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
  }

  // Pass 2: J[n, n] = sum_q rho[q] * raw[q] via single cublasDgemv per Q-tile.
  const double one = 1.0;
  for (int q0 = 0; q0 < q; q0 += q_chunk) {
    const int b = std::min(q_chunk, q - q0);
    if (cudaMemcpyAsync(d_raw_tile.ptr, raw + Size(q0) * nn,
                        Size(b) * nn * sizeof(double), cudaMemcpyHostToDevice,
                        sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
    if (cublasDgemv(cg.handle, CUBLAS_OP_N, static_cast<int>(nn), b, &one,
                    d_raw_tile.ptr, static_cast<int>(nn), d_rho.ptr + q0, 1,
                    &one, d_J.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
  }

  // Combine F_AO_a = H + J - K_a (in-place into d_K_a), copy D2H.
  cabs_fock_combine_inplace_kernel<<<grid_blocks(nn), 256, 0, sg.stream>>>(
      d_h.ptr, d_J.ptr, d_K_a.ptr, nn);
  if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
  if (cudaMemcpyAsync(fock_ao_a, d_K_a.ptr, nn * sizeof(double),
                      cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }

  if (!same_occ) {
    cabs_fock_combine_inplace_kernel<<<grid_blocks(nn), 256, 0, sg.stream>>>(
        d_h.ptr, d_J.ptr, d_K_b.ptr, nn);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
    if (cudaMemcpyAsync(fock_ao_b, d_K_b.ptr, nn * sizeof(double),
                        cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
  }

  if (cudaStreamSynchronize(sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (same_occ) {
    std::copy_n(fock_ao_a, nn, fock_ao_b);
  }
  if (!is_finite_buffer(fock_ao_a, nn) || !is_finite_buffer(fock_ao_b, nn) ||
      (rho_out != nullptr && !is_finite_buffer(rho_out, Size(q))) ||
      (oa > 0 && raw_a_out != nullptr && !is_finite_buffer(raw_a_out, Size(q) * oa * n)) ||
      (oa > 0 && fit_a_out != nullptr && !is_finite_buffer(fit_a_out, Size(q) * oa * n)) ||
      (!same_occ && ob > 0 && raw_b_out != nullptr &&
       !is_finite_buffer(raw_b_out, Size(q) * ob * n)) ||
      (!same_occ && ob > 0 && fit_b_out != nullptr &&
       !is_finite_buffer(fit_b_out, Size(q) * ob * n))) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_cabs_fock_rev_cuda(
    int n, int ext, int q, int oa, int ob, int occ_rows, int same_spin,
    int q_chunk_hint, const double* fock_ao_a, const double* fock_ao_b,
    const double* frame_a, const double* frame_b, const double* seed_a,
    const double* seed_b, const double* raw, const double* whitener,
    const double* rho, const double* raw_a, const double* fit_a,
    const double* raw_b, const double* fit_b, const double* occ_a,
    const double* occ_b, double* ga_out, double* gb_out, double* core_bar_out,
    double* raw_bar_out, double* metric_bar_out) {
  try {
  if (n <= 0 || ext <= 0 || ext > n || q <= 0 || oa < 0 || oa > ext ||
      ob < 0 || ob > ext || occ_rows <= 0 || occ_rows > n) {
    return kStatusInvalidArg;
  }
  if (same_spin && oa != ob) return kStatusInvalidArg;
  if (fock_ao_a == nullptr || frame_a == nullptr || seed_a == nullptr ||
      raw == nullptr || whitener == nullptr || rho == nullptr ||
      ga_out == nullptr || gb_out == nullptr || core_bar_out == nullptr ||
      raw_bar_out == nullptr || metric_bar_out == nullptr) {
    return kStatusInvalidArg;
  }
  if (!same_spin &&
      (fock_ao_b == nullptr || frame_b == nullptr || seed_b == nullptr)) {
    return kStatusInvalidArg;
  }
  if (oa > 0 && (raw_a == nullptr || fit_a == nullptr || occ_a == nullptr)) {
    return kStatusInvalidArg;
  }
  if (!same_spin && ob > 0 &&
      (raw_b == nullptr || fit_b == nullptr || occ_b == nullptr)) {
    return kStatusInvalidArg;
  }

  const Size nn = Size(n) * n;
  const Size ne = Size(n) * ext;
  const Size ee = Size(ext) * ext;
  const Size qq = Size(q) * q;
  if (!is_finite_buffer(fock_ao_a, nn) || !is_finite_buffer(frame_a, ne) ||
      !is_finite_buffer(seed_a, ee) || !is_finite_buffer(raw, Size(q) * nn) ||
      !is_finite_buffer(whitener, qq) || !is_finite_buffer(rho, Size(q))) {
    return kStatusNonFinite;
  }
  if (!same_spin &&
      (!is_finite_buffer(fock_ao_b, nn) || !is_finite_buffer(frame_b, ne) ||
       !is_finite_buffer(seed_b, ee))) {
    return kStatusNonFinite;
  }
  if (oa > 0 &&
      (!is_finite_buffer(raw_a, Size(q) * oa * n) ||
       !is_finite_buffer(fit_a, Size(q) * oa * n) ||
       !is_finite_buffer(occ_a, Size(occ_rows) * oa))) {
    return kStatusNonFinite;
  }
  if (!same_spin && ob > 0 &&
      (!is_finite_buffer(raw_b, Size(q) * ob * n) ||
       !is_finite_buffer(fit_b, Size(q) * ob * n) ||
       !is_finite_buffer(occ_b, Size(occ_rows) * ob))) {
    return kStatusNonFinite;
  }

  const int o_max = std::max(oa, ob);
  const Size max_on = Size(o_max) * n;
  const Size max_qon = Size(q) * max_on;
  if (nn > Size(std::numeric_limits<int>::max()) ||
      max_on > Size(std::numeric_limits<int>::max()) ||
      Size(q) * o_max > Size(std::numeric_limits<int>::max())) {
    return kStatusBudgetExceeded;
  }

  // Fixed GPU memory: W + metric_bar + w_tmp + congruence scratch + 3 * [q, o_max, n]
  const Size fixed_elems =
      Size(3) * qq + Size(6) * nn + Size(4) * ne + Size(2) * ee +
      Size(2) * Size(q) + Size(4) * Size(occ_rows) * o_max + Size(3) * max_qon;
  const Size fixed_bytes = fixed_elems * sizeof(double);
  const Size per_q_bytes = (nn + Size(n) * o_max) * sizeof(double);
  const int q_chunk =
      choose_q_chunk(q, n, fixed_bytes, per_q_bytes, q_chunk_hint);
  if (q_chunk <= 0) return kStatusBudgetExceeded;

  CudaStreamGuard sg;
  if (sg.init() != cudaSuccess) return kStatusCudaError;
  CublasGuard cg;
  if (cg.init(sg.stream) != CUBLAS_STATUS_SUCCESS) return kStatusCublasError;

  // Persistent GPU allocations outside all loops.
  DeviceBuf d_W, d_metric_bar, d_w_tmp, d_rho, d_bar_rho;
  DeviceBuf d_ao_seed_a, d_ao_seed_b, d_bar_g, d_k_seed, d_m_raw, d_m_sym;
  DeviceBuf d_frame, d_seed, d_s_sym, d_u_cong, d_mc, d_g_a, d_g_b;
  DeviceBuf d_raw_tile, d_S_T;
  DeviceBuf d_buf1, d_buf2, d_buf3;
  DeviceBuf d_occ_a, d_occ_b, d_bar_occ_a, d_bar_occ_b;

  if (d_W.alloc(qq) != cudaSuccess || d_metric_bar.alloc(qq) != cudaSuccess ||
      d_w_tmp.alloc(qq) != cudaSuccess || d_rho.alloc(Size(q)) != cudaSuccess ||
      d_bar_rho.alloc(Size(q)) != cudaSuccess ||
      d_ao_seed_a.alloc(nn) != cudaSuccess || d_bar_g.alloc(nn) != cudaSuccess ||
      d_k_seed.alloc(nn) != cudaSuccess || d_m_raw.alloc(nn) != cudaSuccess ||
      d_m_sym.alloc(nn) != cudaSuccess || d_frame.alloc(ne) != cudaSuccess ||
      d_seed.alloc(ee) != cudaSuccess || d_s_sym.alloc(ee) != cudaSuccess ||
      d_u_cong.alloc(ne) != cudaSuccess || d_mc.alloc(ne) != cudaSuccess ||
      d_g_a.alloc(ne) != cudaSuccess ||
      d_raw_tile.alloc(Size(q_chunk) * nn) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (!same_spin) {
    if (d_ao_seed_b.alloc(nn) != cudaSuccess ||
        d_g_b.alloc(ne) != cudaSuccess) {
      return kStatusCudaError;
    }
  }
  if (o_max > 0) {
    if (d_buf1.alloc(max_qon) != cudaSuccess ||
        d_buf2.alloc(max_qon) != cudaSuccess ||
        d_buf3.alloc(max_qon) != cudaSuccess ||
        d_S_T.alloc(Size(q_chunk) * n * o_max) != cudaSuccess) {
      return kStatusCudaError;
    }
  }
  if (oa > 0) {
    if (d_occ_a.alloc(Size(occ_rows) * oa) != cudaSuccess ||
        d_bar_occ_a.alloc(Size(occ_rows) * oa) != cudaSuccess) {
      return kStatusCudaError;
    }
    if (cudaMemcpyAsync(d_occ_a.ptr, occ_a, Size(occ_rows) * oa * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
        cudaMemsetAsync(d_bar_occ_a.ptr, 0, Size(occ_rows) * oa * sizeof(double),
                        sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
  }
  if (!same_spin && ob > 0) {
    if (d_occ_b.alloc(Size(occ_rows) * ob) != cudaSuccess ||
        d_bar_occ_b.alloc(Size(occ_rows) * ob) != cudaSuccess) {
      return kStatusCudaError;
    }
    if (cudaMemcpyAsync(d_occ_b.ptr, occ_b, Size(occ_rows) * ob * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
        cudaMemsetAsync(d_bar_occ_b.ptr, 0, Size(occ_rows) * ob * sizeof(double),
                        sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
  }

  if (cudaMemcpyAsync(d_W.ptr, whitener, qq * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
      cudaMemcpyAsync(d_rho.ptr, rho, Size(q) * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
      cudaMemsetAsync(d_metric_bar.ptr, 0, qq * sizeof(double), sg.stream) !=
          cudaSuccess) {
    return kStatusCudaError;
  }

  // Step 1: Congruence pullback on GPU for spin alpha (and beta if !same_spin).
  auto congruence_rev_cuda = [&](const double* h_fock, const double* h_frame,
                                 const double* h_seed, double* d_ao_seed,
                                 double* d_g_target) -> int {
    if (cudaMemcpyAsync(d_m_raw.ptr, h_fock, nn * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
        cudaMemcpyAsync(d_frame.ptr, h_frame, ne * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
        cudaMemcpyAsync(d_seed.ptr, h_seed, ee * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
    symmetrize_copy_kernel<<<grid_blocks(nn), 256, 0, sg.stream>>>(
        d_m_raw.ptr, d_m_sym.ptr, n);
    add_transpose_kernel<<<grid_blocks(ee), 256, 0, sg.stream>>>(
        d_seed.ptr, d_s_sym.ptr, ext);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;

    // U[ext, n] = seed @ frame^T; m_raw = frame @ U; ao_seed = sym(m_raw)
    if (rm_gemm_cuda(cg.handle, false, true, ext, n, ext, 1.0, d_seed.ptr, ext,
                     d_frame.ptr, ext, 0.0, d_u_cong.ptr, n) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm_cuda(cg.handle, false, false, n, n, ext, 1.0, d_frame.ptr, ext,
                     d_u_cong.ptr, n, 0.0, d_m_raw.ptr, n) !=
            CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
    symmetrize_copy_kernel<<<grid_blocks(nn), 256, 0, sg.stream>>>(
        d_m_raw.ptr, d_ao_seed, n);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;

    // g_target = m_sym @ frame @ (seed + seed^T)
    if (rm_gemm_cuda(cg.handle, false, false, n, ext, n, 1.0, d_m_sym.ptr, n,
                     d_frame.ptr, ext, 0.0, d_mc.ptr, ext) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm_cuda(cg.handle, false, false, n, ext, ext, 1.0, d_mc.ptr, ext,
                     d_s_sym.ptr, ext, 0.0, d_g_target, ext) !=
            CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
    return kStatusOk;
  };

  int rc = congruence_rev_cuda(fock_ao_a, frame_a, seed_a, d_ao_seed_a.ptr,
                               d_g_a.ptr);
  if (rc != kStatusOk) return rc;
  if (same_spin) {
    add_scaled_pair_kernel<<<grid_blocks(nn), 256, 0, sg.stream>>>(
        d_ao_seed_a.ptr, nullptr, d_bar_g.ptr, nn, 2.0, 0.0);
  } else {
    rc = congruence_rev_cuda(fock_ao_b, frame_b, seed_b, d_ao_seed_b.ptr,
                             d_g_b.ptr);
    if (rc != kStatusOk) return rc;
    add_scaled_pair_kernel<<<grid_blocks(nn), 256, 0, sg.stream>>>(
        d_ao_seed_a.ptr, d_ao_seed_b.ptr, d_bar_g.ptr, nn, 1.0, 1.0);
  }
  if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;

  // core_bar = bar_g
  if (cudaMemcpyAsync(core_bar_out, d_bar_g.ptr, nn * sizeof(double),
                      cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }

  // Step 2 (Pass 1 over raw): bar_rho[q] = <raw[q], bar_g> via cublasDgemv.
  // Skip if oa == 0 && ob == 0 (bar_rho not needed; raw_bar = rho * bar_g only).
  const double one = 1.0;
  const double zero = 0.0;
  if (o_max > 0) {
    for (int q0 = 0; q0 < q; q0 += q_chunk) {
      const int b = std::min(q_chunk, q - q0);
      if (cudaMemcpyAsync(d_raw_tile.ptr, raw + Size(q0) * nn,
                          Size(b) * nn * sizeof(double), cudaMemcpyHostToDevice,
                          sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
      if (cublasDgemv(cg.handle, CUBLAS_OP_T, static_cast<int>(nn), b, &one,
                      d_raw_tile.ptr, static_cast<int>(nn), d_bar_g.ptr, 1,
                      &zero, d_bar_rho.ptr + q0, 1) != CUBLAS_STATUS_SUCCESS) {
        return kStatusCublasError;
      }
    }
  }

  // Step 3: Per-spin implicit metric solve & seed_R build on GPU (reusing d_buf1..3).
  auto build_seed_r_cuda = [&](int o, double spin_scale,
                               const double* d_ao_seed, const double* h_R,
                               const double* h_D, const double* d_occ,
                               double* d_bar_occ,
                               std::vector<double>* h_seed_r_spill) -> int {
    if (o == 0) return kStatusOk;
    const Size on = Size(o) * n;
    const Size qon = Size(q) * on;

    // k_seed = -ao_seed
    add_scaled_pair_kernel<<<grid_blocks(nn), 256, 0, sg.stream>>>(
        d_ao_seed, nullptr, d_k_seed.ptr, nn, -1.0, 0.0);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;

    // Upload R -> d_buf1, D -> d_buf2
    if (cudaMemcpyAsync(d_buf1.ptr, h_R, qon * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
        cudaMemcpyAsync(d_buf2.ptr, h_D, qon * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }

    // bar_occ += sum_a bar_rho[a] * D[a, i, m] (m < occ_rows)
    cabs_bar_occ_from_rho_kernel<<<grid_blocks(Size(occ_rows) * o), 256, 0,
                                   sg.stream>>>(
        d_bar_rho.ptr, d_buf2.ptr, d_bar_occ, q, o, n, occ_rows);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;

    // d_buf3 = bar_D = R @ k_seed^T + (bar_rho * occ on m < occ_rows)
    if (rm_gemm_cuda(cg.handle, false, true, q * o, n, n, 1.0, d_buf1.ptr, n,
                     d_k_seed.ptr, n, 0.0, d_buf3.ptr, n) !=
        CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
    cabs_add_bar_rho_occ_to_bar_d_kernel<<<
        grid_blocks(Size(q) * o * occ_rows), 256, 0, sg.stream>>>(
        d_buf3.ptr, d_bar_rho.ptr, d_occ, q, o, n, occ_rows);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;

    // tc_bar = W^T @ (W @ bar_D): overwrite d_buf1 with W @ bar_D, then d_buf3 with tc_bar
    if (rm_gemm_cuda(cg.handle, false, false, q, static_cast<int>(on), q, 1.0,
                     d_W.ptr, q, d_buf3.ptr, static_cast<int>(on), 0.0,
                     d_buf1.ptr, static_cast<int>(on)) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm_cuda(cg.handle, true, false, q, static_cast<int>(on), q, 1.0,
                     d_W.ptr, q, d_buf1.ptr, static_cast<int>(on), 0.0,
                     d_buf3.ptr, static_cast<int>(on)) !=
            CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }

    // w_tmp[q, q] = D @ tc_bar^T; metric_bar += (-0.5 * spin_scale) * (w_tmp + w_tmp^T)
    if (rm_gemm_cuda(cg.handle, false, true, q, q, static_cast<int>(on), 1.0,
                     d_buf2.ptr, static_cast<int>(on), d_buf3.ptr,
                     static_cast<int>(on), 0.0, d_w_tmp.ptr, q) !=
        CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
    add_sym_scaled_kernel<<<grid_blocks(qq), 256, 0, sg.stream>>>(
        d_w_tmp.ptr, d_metric_bar.ptr, q, -spin_scale);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;

    // seed_R = tc_bar + D @ k_seed -> accumulated into d_buf3
    if (rm_gemm_cuda(cg.handle, false, false, q * o, n, n, 1.0, d_buf2.ptr, n,
                     d_k_seed.ptr, n, 1.0, d_buf3.ptr, n) !=
        CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }

    if (h_seed_r_spill != nullptr) {
      h_seed_r_spill->resize(qon);
      if (cudaMemcpyAsync(h_seed_r_spill->data(), d_buf3.ptr,
                          qon * sizeof(double), cudaMemcpyDeviceToHost,
                          sg.stream) != cudaSuccess ||
          cudaStreamSynchronize(sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    }
    return kStatusOk;
  };

  // Keep seed_R GPU-resident whenever only one active spin exists (RHF or ob==0/oa==0).
  // When UHF has both oa > 0 and ob > 0, spill alpha seed_R to host and reuse d_buf1..3 for beta!
  const bool need_uhf_both = (!same_spin && oa > 0 && ob > 0);
  std::vector<double> h_seed_r_a;
  if (same_spin) {
    rc = build_seed_r_cuda(oa, 2.0, d_ao_seed_a.ptr, raw_a, fit_a, d_occ_a.ptr,
                           d_bar_occ_a.ptr, nullptr);
    if (rc != kStatusOk) return rc;
  } else {
    rc = build_seed_r_cuda(oa, 1.0, d_ao_seed_a.ptr, raw_a, fit_a, d_occ_a.ptr,
                           d_bar_occ_a.ptr,
                           need_uhf_both ? &h_seed_r_a : nullptr);
    if (rc != kStatusOk) return rc;
    rc = build_seed_r_cuda(ob, 1.0, d_ao_seed_b.ptr, raw_b, fit_b, d_occ_b.ptr,
                           d_bar_occ_b.ptr, nullptr);
    if (rc != kStatusOk) return rc;
  }

  // Step 4 (Pass 2 over raw Q-tiles):
  //   - Compute bar_occ_half += raw[:, :, :occ_rows]^T @ S_T (flattened DGEMM)
  //   - Overwrite d_raw_tile in-place with raw_bar = rho * bar_g + sum_s spin_scale * S_T @ occ^T
  //   - Copy d_raw_tile D2H to raw_bar_out[q0:q1]
  const Size oa_n = Size(oa) * n;
  const Size ob_n = Size(ob) * n;
  for (int q0 = 0; q0 < q; q0 += q_chunk) {
    const int b = std::min(q_chunk, q - q0);

    if (o_max > 0) {
      if (cudaMemcpyAsync(d_raw_tile.ptr, raw + Size(q0) * nn,
                          Size(b) * nn * sizeof(double), cudaMemcpyHostToDevice,
                          sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    }

    // Prepare alpha S_T (if oa > 0) and accumulate bar_occ_a
    if (oa > 0) {
      const double* seed_a_dev = nullptr;
      if (need_uhf_both) {
        if (cudaMemcpyAsync(d_buf1.ptr, h_seed_r_a.data() + Size(q0) * oa_n,
                            Size(b) * oa_n * sizeof(double),
                            cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
          return kStatusCudaError;
        }
        seed_a_dev = d_buf1.ptr;
      } else {
        seed_a_dev = d_buf3.ptr + Size(q0) * oa_n;
      }
      transpose_bon_to_bno_kernel<<<grid_blocks(Size(b) * oa_n), 256, 0,
                                    sg.stream>>>(seed_a_dev, d_S_T.ptr, b, oa,
                                                 n);
      if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
      if (rm_gemm_cuda(cg.handle, true, false, occ_rows, oa, b * n, 1.0,
                       d_raw_tile.ptr, n, d_S_T.ptr, oa, 1.0, d_bar_occ_a.ptr,
                       oa) != CUBLAS_STATUS_SUCCESS) {
        return kStatusCublasError;
      }
    }

    // If UHF with ob > 0, accumulate bar_occ_b before overwriting d_raw_tile!
    if (!same_spin && ob > 0) {
      const double* seed_b_dev = d_buf3.ptr + Size(q0) * ob_n;
      double* s_t_b = need_uhf_both ? d_buf2.ptr : d_S_T.ptr;
      transpose_bon_to_bno_kernel<<<grid_blocks(Size(b) * ob_n), 256, 0,
                                    sg.stream>>>(seed_b_dev, s_t_b, b, ob, n);
      if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
      if (rm_gemm_cuda(cg.handle, true, false, occ_rows, ob, b * n, 1.0,
                       d_raw_tile.ptr, n, s_t_b, ob, 1.0, d_bar_occ_b.ptr,
                       ob) != CUBLAS_STATUS_SUCCESS) {
        return kStatusCublasError;
      }
    }

    // Overwrite d_raw_tile in-place with raw_bar[q0:q1] = rho[q] * bar_g
    cabs_init_raw_bar_j_kernel<<<grid_blocks(Size(b) * nn), 256, 0,
                                 sg.stream>>>(d_raw_tile.ptr, d_rho.ptr + q0,
                                              d_bar_g.ptr, b, n);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;

    // Add alpha S_T @ occ_a^T (scaled by 2.0 on RHF same_spin)
    if (oa > 0) {
      const double scale_a = same_spin ? 2.0 : 1.0;
      if (rm_gemm_cuda(cg.handle, false, true, b * n, occ_rows, oa, scale_a,
                       d_S_T.ptr, oa, d_occ_a.ptr, oa, 1.0, d_raw_tile.ptr,
                       n) != CUBLAS_STATUS_SUCCESS) {
        return kStatusCublasError;
      }
    }

    // Add beta S_T @ occ_b^T for UHF
    if (!same_spin && ob > 0) {
      const double* s_t_b = need_uhf_both ? d_buf2.ptr : d_S_T.ptr;
      if (rm_gemm_cuda(cg.handle, false, true, b * n, occ_rows, ob, 1.0, s_t_b,
                       ob, d_occ_b.ptr, ob, 1.0, d_raw_tile.ptr, n) !=
          CUBLAS_STATUS_SUCCESS) {
        return kStatusCublasError;
      }
    }

    if (cudaMemcpyAsync(raw_bar_out + Size(q0) * nn, d_raw_tile.ptr,
                        Size(b) * nn * sizeof(double), cudaMemcpyDeviceToHost,
                        sg.stream) != cudaSuccess ||
        cudaStreamSynchronize(sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
  }

  // Merge bar_occ into d_g_a / d_g_b and copy final outputs D2H.
  if (oa > 0) {
    cabs_add_bar_occ_to_frame_kernel<<<grid_blocks(Size(occ_rows) * oa), 256, 0,
                                       sg.stream>>>(d_bar_occ_a.ptr, d_g_a.ptr,
                                                    occ_rows, oa, ext);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
  }
  if (!same_spin && ob > 0) {
    cabs_add_bar_occ_to_frame_kernel<<<grid_blocks(Size(occ_rows) * ob), 256, 0,
                                       sg.stream>>>(d_bar_occ_b.ptr, d_g_b.ptr,
                                                    occ_rows, ob, ext);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
  }

  if (cudaMemcpyAsync(ga_out, d_g_a.ptr, ne * sizeof(double),
                      cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess ||
      cudaMemcpyAsync(metric_bar_out, d_metric_bar.ptr, qq * sizeof(double),
                      cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (!same_spin) {
    if (cudaMemcpyAsync(gb_out, d_g_b.ptr, ne * sizeof(double),
                        cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
  }
  if (cudaStreamSynchronize(sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (same_spin) {
    std::copy_n(ga_out, ne, gb_out);
  }
  if (!is_finite_buffer(ga_out, ne) || !is_finite_buffer(gb_out, ne) ||
      !is_finite_buffer(core_bar_out, nn) ||
      !is_finite_buffer(raw_bar_out, Size(q) * nn) ||
      !is_finite_buffer(metric_bar_out, qq)) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_cabs_frame_kernel_cotangent_cuda(
    int atomic, int rank, const double* overlap, const double* frame,
    const double* frame_bar, double* kernel_bar_out, double* dual_gram_out,
    double* gauge_skew_out, double* reconstruction_residual_out) {
  try {
  if (atomic <= 0 || rank <= 0 || rank > atomic || overlap == nullptr ||
      frame == nullptr || frame_bar == nullptr || kernel_bar_out == nullptr ||
      dual_gram_out == nullptr || gauge_skew_out == nullptr ||
      reconstruction_residual_out == nullptr) {
    return kStatusInvalidArg;
  }
  const Size aa = Size(atomic) * atomic;
  const Size ar = Size(atomic) * rank;
  const Size rr = Size(rank) * rank;
  if (!is_finite_buffer(overlap, aa) || !is_finite_buffer(frame, ar) ||
      !is_finite_buffer(frame_bar, ar)) {
    return kStatusNonFinite;
  }
  const Size total_bytes =
      (Size(4) * aa + Size(5) * ar + Size(4) * rr) * sizeof(double);
  if (total_bytes > kMaxDeviceWorkspaceBytes) {
    return kStatusBudgetExceeded;
  }

  CudaStreamGuard sg;
  if (sg.init() != cudaSuccess) return kStatusCudaError;
  CublasGuard cg;
  if (cg.init(sg.stream) != CUBLAS_STATUS_SUCCESS) return kStatusCublasError;

  DeviceBuf d_S, d_C, d_G, d_dual, d_moment, d_sym, d_skew, d_dg, d_cross,
      d_dsym, d_proj, d_kb, d_recon;
  if (d_S.alloc(aa) != cudaSuccess || d_C.alloc(ar) != cudaSuccess ||
      d_G.alloc(ar) != cudaSuccess || d_dual.alloc(ar) != cudaSuccess ||
      d_moment.alloc(rr) != cudaSuccess || d_sym.alloc(rr) != cudaSuccess ||
      d_skew.alloc(rr) != cudaSuccess || d_dg.alloc(rr) != cudaSuccess ||
      d_cross.alloc(aa) != cudaSuccess || d_dsym.alloc(ar) != cudaSuccess ||
      d_proj.alloc(aa) != cudaSuccess || d_kb.alloc(aa) != cudaSuccess ||
      d_recon.alloc(ar) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (cudaMemcpyAsync(d_S.ptr, overlap, aa * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
      cudaMemcpyAsync(d_C.ptr, frame, ar * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
      cudaMemcpyAsync(d_G.ptr, frame_bar, ar * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }

  // dual = S @ C; moment = C^T @ G; dual_gram = C^T @ dual
  if (rm_gemm_cuda(cg.handle, false, false, atomic, rank, atomic, 1.0, d_S.ptr,
                   atomic, d_C.ptr, rank, 0.0, d_dual.ptr, rank) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemm_cuda(cg.handle, true, false, rank, rank, atomic, 1.0, d_C.ptr,
                   rank, d_G.ptr, rank, 0.0, d_moment.ptr, rank) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemm_cuda(cg.handle, true, false, rank, rank, atomic, 1.0, d_C.ptr,
                   rank, d_dual.ptr, rank, 0.0, d_dg.ptr, rank) !=
          CUBLAS_STATUS_SUCCESS) {
    return kStatusCublasError;
  }

  cabs_fk_sym_skew_kernel<<<grid_blocks(rr), 256, 0, sg.stream>>>(
      d_moment.ptr, d_sym.ptr, d_skew.ptr, rank);
  if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;

  // cross = G @ dual^T; dsym = dual @ sym; proj = dsym @ dual^T
  if (rm_gemm_cuda(cg.handle, false, true, atomic, atomic, rank, 1.0, d_G.ptr,
                   rank, d_dual.ptr, rank, 0.0, d_cross.ptr, atomic) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemm_cuda(cg.handle, false, false, atomic, rank, rank, 1.0, d_dual.ptr,
                   rank, d_sym.ptr, rank, 0.0, d_dsym.ptr, rank) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemm_cuda(cg.handle, false, true, atomic, atomic, rank, 1.0,
                   d_dsym.ptr, rank, d_dual.ptr, rank, 0.0, d_proj.ptr,
                   atomic) != CUBLAS_STATUS_SUCCESS) {
    return kStatusCublasError;
  }

  cabs_fk_kernel_bar_kernel<<<grid_blocks(aa), 256, 0, sg.stream>>>(
      d_cross.ptr, d_proj.ptr, d_kb.ptr, atomic);
  if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;

  // recon = 2.0 * kernel_bar @ C - G
  if (rm_gemm_cuda(cg.handle, false, false, atomic, rank, atomic, 2.0, d_kb.ptr,
                   atomic, d_C.ptr, rank, 0.0, d_recon.ptr, rank) !=
      CUBLAS_STATUS_SUCCESS) {
    return kStatusCublasError;
  }
  sub_inplace_kernel<<<grid_blocks(ar), 256, 0, sg.stream>>>(d_recon.ptr,
                                                             d_G.ptr, ar);
  if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;

  if (cudaMemcpyAsync(kernel_bar_out, d_kb.ptr, aa * sizeof(double),
                      cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess ||
      cudaMemcpyAsync(dual_gram_out, d_dg.ptr, rr * sizeof(double),
                      cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess ||
      cudaMemcpyAsync(gauge_skew_out, d_skew.ptr, rr * sizeof(double),
                      cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess ||
      cudaMemcpyAsync(reconstruction_residual_out, d_recon.ptr,
                      ar * sizeof(double), cudaMemcpyDeviceToHost, sg.stream) !=
          cudaSuccess ||
      cudaStreamSynchronize(sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (!is_finite_buffer(kernel_bar_out, aa) ||
      !is_finite_buffer(dual_gram_out, rr) ||
      !is_finite_buffer(gauge_skew_out, rr) ||
      !is_finite_buffer(reconstruction_residual_out, ar)) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_cabs_whiten_3c_cuda(
    int aux, int pairs, const double* whitener, const double* raw,
    double* df_out) {
  try {
  if (aux <= 0 || pairs < 0) return kStatusInvalidArg;
  if (pairs == 0) return kStatusOk;
  if (whitener == nullptr || raw == nullptr || df_out == nullptr) {
    return kStatusInvalidArg;
  }
  const Size aa = Size(aux) * aux;
  if (!is_finite_buffer(whitener, aa) ||
      !is_finite_buffer(raw, Size(aux) * pairs)) {
    return kStatusNonFinite;
  }

  const Size fixed_bytes = aa * sizeof(double);
  const Size per_col_bytes = Size(2) * aux * sizeof(double);
  if (fixed_bytes + per_col_bytes > kMaxDeviceWorkspaceBytes) {
    return kStatusBudgetExceeded;
  }
  const Size avail = kFitWorkspaceBytes > fixed_bytes
                         ? (kFitWorkspaceBytes - fixed_bytes)
                         : 0;
  const int p_chunk = static_cast<int>(
      std::min<Size>(Size(pairs), std::max<Size>(1, avail / per_col_bytes)));

  CudaStreamGuard sg;
  if (sg.init() != cudaSuccess) return kStatusCudaError;
  CublasGuard cg;
  if (cg.init(sg.stream) != CUBLAS_STATUS_SUCCESS) return kStatusCublasError;

  DeviceBuf d_W, d_in, d_out;
  if (d_W.alloc(aa) != cudaSuccess ||
      d_in.alloc(Size(aux) * p_chunk) != cudaSuccess ||
      d_out.alloc(Size(aux) * p_chunk) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (cudaMemcpyAsync(d_W.ptr, whitener, aa * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }

  for (int p0 = 0; p0 < pairs; p0 += p_chunk) {
    const int p_len = std::min(p_chunk, pairs - p0);
    if (p_len == pairs) {
      if (cudaMemcpyAsync(d_in.ptr, raw, Size(aux) * pairs * sizeof(double),
                          cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    } else {
      if (cudaMemcpy2DAsync(d_in.ptr, Size(p_len) * sizeof(double), raw + p0,
                            Size(pairs) * sizeof(double),
                            Size(p_len) * sizeof(double),
                            static_cast<size_t>(aux), cudaMemcpyHostToDevice,
                            sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    }
    if (rm_gemm_cuda(cg.handle, false, false, aux, p_len, aux, 1.0, d_W.ptr,
                     aux, d_in.ptr, p_len, 0.0, d_out.ptr, p_len) !=
        CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
    if (p_len == pairs) {
      if (cudaMemcpyAsync(df_out, d_out.ptr, Size(aux) * pairs * sizeof(double),
                          cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    } else {
      if (cudaMemcpy2DAsync(df_out + p0, Size(pairs) * sizeof(double),
                            d_out.ptr, Size(p_len) * sizeof(double),
                            Size(p_len) * sizeof(double),
                            static_cast<size_t>(aux), cudaMemcpyDeviceToHost,
                            sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
    }
    if (cudaStreamSynchronize(sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
  }
  if (!is_finite_buffer(df_out, Size(aux) * pairs)) return kStatusNonFinite;
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_cabs_cphf_action_cuda(
    int oa, int ob, int obs, int aux, int is_rhf, int q_chunk_hint,
    const double* df_a, const double* df_b, const double* gap_a,
    const double* gap_b, const double* trial_a, const double* trial_b,
    double* out_a, double* out_b) {
  try {
  if (obs <= 0 || aux <= 0 || oa < 0 || ob < 0 || oa >= obs || ob >= obs ||
      oa + ob == 0) {
    return kStatusInvalidArg;
  }
  if (is_rhf && (oa == 0 || oa != ob)) return kStatusInvalidArg;
  const int va = obs - oa;
  const int vb = obs - ob;
  const Size oo_full = Size(obs) * obs;
  const Size vo_a = Size(va) * oa;
  const Size vo_b = Size(vb) * ob;
  const bool has_a = (oa > 0);
  const bool has_b = (!is_rhf && ob > 0);

  if (has_a && (df_a == nullptr || gap_a == nullptr || trial_a == nullptr ||
                out_a == nullptr)) {
    return kStatusInvalidArg;
  }
  if (has_b && (df_b == nullptr || gap_b == nullptr || trial_b == nullptr ||
                out_b == nullptr)) {
    return kStatusInvalidArg;
  }
  if (has_a && (!is_finite_buffer(df_a, Size(aux) * oo_full) ||
                !is_finite_buffer(gap_a, vo_a) ||
                !is_finite_buffer(trial_a, vo_a))) {
    return kStatusNonFinite;
  }
  if (has_b && (!is_finite_buffer(df_b, Size(aux) * oo_full) ||
                !is_finite_buffer(gap_b, vo_b) ||
                !is_finite_buffer(trial_b, vo_b))) {
    return kStatusNonFinite;
  }

  const Size max_vo = std::max(has_a ? vo_a : Size(0), has_b ? vo_b : Size(0));
  const Size max_oo = std::max(has_a ? Size(oa) * oa : Size(0),
                               has_b ? Size(ob) * ob : Size(0));
  const Size fixed_bytes =
      (Size(3) * (has_a ? vo_a : Size(0)) + Size(3) * (has_b ? vo_b : Size(0))) *
      sizeof(double);
  const Size per_q_elems =
      (has_a ? oo_full : Size(0)) + (has_b ? oo_full : Size(0)) +
      Size(2) * max_vo + max_oo + Size(1);
  const Size per_q_bytes = per_q_elems * sizeof(double);
  const int q_chunk =
      choose_q_chunk(aux, obs, fixed_bytes, per_q_bytes, q_chunk_hint);
  if (q_chunk <= 0) return kStatusBudgetExceeded;

  CudaStreamGuard sg;
  if (sg.init() != cudaSuccess) return kStatusCudaError;
  CublasGuard cg;
  if (cg.init(sg.stream) != CUBLAS_STATUS_SUCCESS) return kStatusCublasError;

  DeviceBuf d_df_a, d_df_b, d_gap_a, d_gap_b, d_tr_a, d_tr_b, d_out_a, d_out_b;
  DeviceBuf d_kv, d_ki, d_exch, d_charge;
  if (d_kv.alloc(Size(q_chunk) * max_vo) != cudaSuccess ||
      d_ki.alloc(Size(q_chunk) * max_oo) != cudaSuccess ||
      d_exch.alloc(Size(q_chunk) * max_vo) != cudaSuccess ||
      d_charge.alloc(Size(q_chunk)) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (has_a) {
    if (d_df_a.alloc(Size(q_chunk) * oo_full) != cudaSuccess ||
        d_gap_a.alloc(vo_a) != cudaSuccess ||
        d_tr_a.alloc(vo_a) != cudaSuccess ||
        d_out_a.alloc(vo_a) != cudaSuccess) {
      return kStatusCudaError;
    }
    if (cudaMemcpyAsync(d_gap_a.ptr, gap_a, vo_a * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
        cudaMemcpyAsync(d_tr_a.ptr, trial_a, vo_a * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
    cphf_init_gap_trial_kernel<<<grid_blocks(vo_a), 256, 0, sg.stream>>>(
        d_gap_a.ptr, d_tr_a.ptr, d_out_a.ptr, vo_a);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
  }
  if (has_b) {
    if (d_df_b.alloc(Size(q_chunk) * oo_full) != cudaSuccess ||
        d_gap_b.alloc(vo_b) != cudaSuccess ||
        d_tr_b.alloc(vo_b) != cudaSuccess ||
        d_out_b.alloc(vo_b) != cudaSuccess) {
      return kStatusCudaError;
    }
    if (cudaMemcpyAsync(d_gap_b.ptr, gap_b, vo_b * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
        cudaMemcpyAsync(d_tr_b.ptr, trial_b, vo_b * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
    cphf_init_gap_trial_kernel<<<grid_blocks(vo_b), 256, 0, sg.stream>>>(
        d_gap_b.ptr, d_tr_b.ptr, d_out_b.ptr, vo_b);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
  }

  const double coul_scale = is_rhf ? 4.0 : 2.0;

  auto apply_spin_tile = [&](int b, int o, int v, const double* d_df_tile,
                             const double* d_tr, double* d_out) -> int {
    const Size vo_n = Size(v) * o;
    const Size oo_n = Size(o) * o;
    const double* d_oo = d_df_tile;
    const double* d_vo = d_df_tile + Size(o) * obs;
    const double* d_vv = d_df_tile + Size(o) * obs + o;
    // kv[a] = vv[a] @ trial
    if (rm_gemm_strided_batched_cuda(
            cg.handle, false, false, v, o, v, 1.0, d_vv, obs,
            static_cast<long long>(oo_full), d_tr, o, 0LL, 0.0, d_kv.ptr, o,
            static_cast<long long>(vo_n), b) != CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
    // ki[a] = vo[a]^T @ trial
    if (rm_gemm_strided_batched_cuda(
            cg.handle, true, false, o, o, v, 1.0, d_vo, obs,
            static_cast<long long>(oo_full), d_tr, o, 0LL, 0.0, d_ki.ptr, o,
            static_cast<long long>(oo_n), b) != CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
    // exch[a] = kv[a] @ oo[a]^T + vo[a] @ ki[a]^T
    if (rm_gemm_strided_batched_cuda(
            cg.handle, false, true, v, o, o, 1.0, d_kv.ptr, o,
            static_cast<long long>(vo_n), d_oo, obs,
            static_cast<long long>(oo_full), 0.0, d_exch.ptr, o,
            static_cast<long long>(vo_n), b) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm_strided_batched_cuda(
            cg.handle, false, true, v, o, o, 1.0, d_vo, obs,
            static_cast<long long>(oo_full), d_ki.ptr, o,
            static_cast<long long>(oo_n), 1.0, d_exch.ptr, o,
            static_cast<long long>(vo_n), b) != CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
    cphf_reduce_coul_exch_kernel<<<grid_blocks(vo_n), 256, 0, sg.stream>>>(
        d_df_tile, d_charge.ptr, d_exch.ptr, d_out, b, o, v, obs, coul_scale);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
    return kStatusOk;
  };

  // Single pass over Q-tiles: total_charge[a] = charge_a[a] + charge_b[a] is local to a!
  for (int q0 = 0; q0 < aux; q0 += q_chunk) {
    const int b = std::min(q_chunk, aux - q0);
    int accum = 0;
    if (has_a) {
      if (cudaMemcpyAsync(d_df_a.ptr, df_a + Size(q0) * oo_full,
                          Size(b) * oo_full * sizeof(double),
                          cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
      cphf_vo_dot_kernel<<<b, 256, 0, sg.stream>>>(
          d_df_a.ptr, d_tr_a.ptr, d_charge.ptr, b, oa, va, obs, accum);
      if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
      accum = 1;
    }
    if (has_b) {
      if (cudaMemcpyAsync(d_df_b.ptr, df_b + Size(q0) * oo_full,
                          Size(b) * oo_full * sizeof(double),
                          cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
      cphf_vo_dot_kernel<<<b, 256, 0, sg.stream>>>(
          d_df_b.ptr, d_tr_b.ptr, d_charge.ptr, b, ob, vb, obs, accum);
      if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
    }
    if (has_a) {
      int rc = apply_spin_tile(b, oa, va, d_df_a.ptr, d_tr_a.ptr, d_out_a.ptr);
      if (rc != kStatusOk) return rc;
    }
    if (has_b) {
      int rc = apply_spin_tile(b, ob, vb, d_df_b.ptr, d_tr_b.ptr, d_out_b.ptr);
      if (rc != kStatusOk) return rc;
    }
  }

  if (has_a) {
    if (cudaMemcpyAsync(out_a, d_out_a.ptr, vo_a * sizeof(double),
                        cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
  }
  if (has_b) {
    if (cudaMemcpyAsync(out_b, d_out_b.ptr, vo_b * sizeof(double),
                        cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
  }
  if (cudaStreamSynchronize(sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }
  if ((has_a && !is_finite_buffer(out_a, vo_a)) ||
      (has_b && !is_finite_buffer(out_b, vo_b))) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

extern "C" int gqc_f12_cabs_nuclear_weights_cuda(
    int oa, int ob, int obs, int aux, int nao, int same_spin, int q_chunk_hint,
    const double* whitener, const double* c_a, const double* c_b,
    const double* df_a, const double* df_b, const double* p_a,
    const double* p_b, const double* sm_a, const double* sm_b,
    double* bar_ds_out, double* bar_dh_out, double* bar_dr_out,
    double* bar_dj_out) {
  try {
  if (obs <= 0 || aux <= 0 || nao < obs || oa < 0 || oa > obs || ob < 0 ||
      ob > obs) {
    return kStatusInvalidArg;
  }
  if (same_spin && oa != ob) return kStatusInvalidArg;
  if (whitener == nullptr || c_a == nullptr || df_a == nullptr ||
      p_a == nullptr || sm_a == nullptr || bar_ds_out == nullptr ||
      bar_dh_out == nullptr || bar_dr_out == nullptr ||
      bar_dj_out == nullptr) {
    return kStatusInvalidArg;
  }
  if (!same_spin && (c_b == nullptr || df_b == nullptr || p_b == nullptr ||
                     sm_b == nullptr)) {
    return kStatusInvalidArg;
  }

  const Size qq = Size(aux) * aux;
  const Size nn_obs = Size(obs) * obs;
  const Size nn_ao = Size(nao) * nao;
  const Size n_c = Size(nao) * obs;
  if (!is_finite_buffer(whitener, qq) || !is_finite_buffer(c_a, n_c) ||
      !is_finite_buffer(df_a, Size(aux) * nn_obs) ||
      !is_finite_buffer(p_a, nn_obs) || !is_finite_buffer(sm_a, nn_obs)) {
    return kStatusNonFinite;
  }
  if (!same_spin &&
      (!is_finite_buffer(c_b, n_c) ||
       !is_finite_buffer(df_b, Size(aux) * nn_obs) ||
       !is_finite_buffer(p_b, nn_obs) || !is_finite_buffer(sm_b, nn_obs))) {
    return kStatusNonFinite;
  }
  if (nn_obs > Size(std::numeric_limits<int>::max()) ||
      nn_ao > Size(std::numeric_limits<int>::max())) {
    return kStatusBudgetExceeded;
  }

  const int o_max = std::max(oa, ob);
  const Size fixed_elems =
      Size(3) * qq + Size(3) * nn_ao + Size(3) * nn_obs + Size(2) * n_c +
      Size(2) * Size(aux);
  const Size fixed_bytes = fixed_elems * sizeof(double);
  const Size per_q_elems =
      Size(3) * nn_obs + Size(3) * nn_ao + n_c + Size(obs) * o_max +
      Size(o_max) * o_max;
  const Size per_q_bytes = per_q_elems * sizeof(double);
  const int q_chunk =
      choose_q_chunk(aux, nao, fixed_bytes, per_q_bytes, q_chunk_hint);
  if (q_chunk <= 0) return kStatusBudgetExceeded;

  CudaStreamGuard sg;
  if (sg.init() != cudaSuccess) return kStatusCudaError;
  CublasGuard cg;
  if (cg.init(sg.stream) != CUBLAS_STATUS_SUCCESS) return kStatusCublasError;

  // Persistent GPU allocations outside all Q-tile loops.
  DeviceBuf d_W, d_bar_M, d_tmp_m, d_charge, d_bar_dcharge;
  DeviceBuf d_c_s, d_p_s, d_psym_s, d_sm_work, d_tmp_2d, d_bar_dh, d_bar_ds,
      d_sym_2d;
  DeviceBuf d_df_tile, d_df_tile_b, d_bar_db_tile, d_ao_sym_tile, d_ao_raw_tile,
      d_tmp_c_tile, d_dr_tile;
  DeviceBuf d_ex_tile, d_sex_tile;

  if (d_W.alloc(qq) != cudaSuccess || d_bar_M.alloc(qq) != cudaSuccess ||
      d_tmp_m.alloc(qq) != cudaSuccess ||
      d_charge.alloc(Size(aux)) != cudaSuccess ||
      d_bar_dcharge.alloc(Size(aux)) != cudaSuccess ||
      d_c_s.alloc(n_c) != cudaSuccess || d_p_s.alloc(nn_obs) != cudaSuccess ||
      d_psym_s.alloc(nn_obs) != cudaSuccess ||
      d_sm_work.alloc(nn_obs) != cudaSuccess ||
      d_tmp_2d.alloc(n_c) != cudaSuccess ||
      d_bar_dh.alloc(nn_ao) != cudaSuccess ||
      d_bar_ds.alloc(nn_ao) != cudaSuccess ||
      d_sym_2d.alloc(nn_ao) != cudaSuccess ||
      d_df_tile.alloc(Size(q_chunk) * nn_obs) != cudaSuccess ||
      d_df_tile_b.alloc(Size(q_chunk) * nn_obs) != cudaSuccess ||
      d_bar_db_tile.alloc(Size(q_chunk) * nn_obs) != cudaSuccess ||
      d_ao_sym_tile.alloc(Size(q_chunk) * nn_ao) != cudaSuccess ||
      d_ao_raw_tile.alloc(Size(q_chunk) * nn_ao) != cudaSuccess ||
      d_tmp_c_tile.alloc(Size(q_chunk) * n_c) != cudaSuccess ||
      d_dr_tile.alloc(Size(q_chunk) * nn_ao) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (o_max > 0) {
    if (d_ex_tile.alloc(Size(q_chunk) * obs * o_max) != cudaSuccess ||
        d_sex_tile.alloc(Size(q_chunk) * o_max * o_max) != cudaSuccess) {
      return kStatusCudaError;
    }
  }

  if (cudaMemcpyAsync(d_W.ptr, whitener, qq * sizeof(double),
                      cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
      cudaMemsetAsync(d_bar_M.ptr, 0, qq * sizeof(double), sg.stream) !=
          cudaSuccess ||
      cudaMemsetAsync(d_charge.ptr, 0, Size(aux) * sizeof(double), sg.stream) !=
          cudaSuccess ||
      cudaMemsetAsync(d_bar_dcharge.ptr, 0, Size(aux) * sizeof(double),
                      sg.stream) != cudaSuccess ||
      cudaMemsetAsync(d_bar_dh.ptr, 0, nn_ao * sizeof(double), sg.stream) !=
          cudaSuccess ||
      cudaMemsetAsync(d_bar_ds.ptr, 0, nn_ao * sizeof(double), sg.stream) !=
          cudaSuccess) {
    return kStatusCudaError;
  }

  // Pass 1: charge[aux] and bar_dcharge[aux]
  auto accum_charges_cuda = [&](int o, double scale, const double* h_df,
                                const double* h_p) -> int {
    if (cudaMemcpyAsync(d_p_s.ptr, h_p, nn_obs * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
    for (int q0 = 0; q0 < aux; q0 += q_chunk) {
      const int b = std::min(q_chunk, aux - q0);
      if (cudaMemcpyAsync(d_df_tile.ptr, h_df + Size(q0) * nn_obs,
                          Size(b) * nn_obs * sizeof(double),
                          cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
      nuc_charges_kernel<<<b, 256, 0, sg.stream>>>(
          d_df_tile.ptr, d_p_s.ptr, d_charge.ptr + q0, d_bar_dcharge.ptr + q0,
          b, o, obs, scale);
      if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
    }
    return kStatusOk;
  };
  if (same_spin) {
    int rc = accum_charges_cuda(oa, 2.0, df_a, p_a);
    if (rc != kStatusOk) return rc;
  } else {
    int rc = accum_charges_cuda(oa, 1.0, df_a, p_a);
    if (rc != kStatusOk) return rc;
    rc = accum_charges_cuda(ob, 1.0, df_b, p_b);
    if (rc != kStatusOk) return rc;
  }

  // Pass 2: Complete sm_work and assemble bar_dh, bar_ds
  auto accum_2d_cuda = [&](int o, double scale, const double* h_c,
                           const double* h_df, const double* h_p,
                           const double* h_sm) -> int {
    if (cudaMemcpyAsync(d_c_s.ptr, h_c, n_c * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
        cudaMemcpyAsync(d_p_s.ptr, h_p, nn_obs * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
        cudaMemcpyAsync(d_sm_work.ptr, h_sm, nn_obs * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
    if (o > 0) {
      const Size oo = Size(o) * o;
      const Size ob_o = Size(obs) * o;
      for (int q0 = 0; q0 < aux; q0 += q_chunk) {
        const int b = std::min(q_chunk, aux - q0);
        if (cudaMemcpyAsync(d_df_tile.ptr, h_df + Size(q0) * nn_obs,
                            Size(b) * nn_obs * sizeof(double),
                            cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
          return kStatusCudaError;
        }
        // pb[a, obs, o] = p_s^T @ df_tile[a, :, :o] -> d_ex_tile
        if (rm_gemm_strided_batched_cuda(
                cg.handle, true, false, obs, o, obs, 1.0, d_p_s.ptr, obs, 0LL,
                d_df_tile.ptr, obs, static_cast<long long>(nn_obs), 0.0,
                d_ex_tile.ptr, o, static_cast<long long>(ob_o), b) !=
            CUBLAS_STATUS_SUCCESS) {
          return kStatusCublasError;
        }
        // s_ex[a, o, o] = pb[a]^T @ df_tile[a, :, :o] -> d_sex_tile
        if (rm_gemm_strided_batched_cuda(
                cg.handle, true, false, o, o, obs, 1.0, d_ex_tile.ptr, o,
                static_cast<long long>(ob_o), d_df_tile.ptr, obs,
                static_cast<long long>(nn_obs), 0.0, d_sex_tile.ptr, o,
                static_cast<long long>(oo), b) != CUBLAS_STATUS_SUCCESS) {
          return kStatusCublasError;
        }
        nuc_accum_sm_kernel<<<grid_blocks(oo), 256, 0, sg.stream>>>(
            d_df_tile.ptr, d_sex_tile.ptr, d_bar_dcharge.ptr + q0,
            d_sm_work.ptr, b, o, obs);
        if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
      }
    }
    // bar_dh += scale * (c_s @ p_s @ c_s^T)
    if (rm_gemm_cuda(cg.handle, false, false, nao, obs, obs, 1.0, d_c_s.ptr,
                     obs, d_p_s.ptr, obs, 0.0, d_tmp_2d.ptr, obs) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm_cuda(cg.handle, false, true, nao, nao, obs, scale, d_tmp_2d.ptr,
                     obs, d_c_s.ptr, obs, 1.0, d_bar_dh.ptr, nao) !=
            CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
    // bar_ds += scale * (c_s @ sm_work @ c_s^T)
    if (rm_gemm_cuda(cg.handle, false, false, nao, obs, obs, 1.0, d_c_s.ptr,
                     obs, d_sm_work.ptr, obs, 0.0, d_tmp_2d.ptr, obs) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm_cuda(cg.handle, false, true, nao, nao, obs, scale, d_tmp_2d.ptr,
                     obs, d_c_s.ptr, obs, 1.0, d_bar_ds.ptr, nao) !=
            CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
    return kStatusOk;
  };
  if (same_spin) {
    int rc = accum_2d_cuda(oa, 2.0, c_a, df_a, p_a, sm_a);
    if (rc != kStatusOk) return rc;
  } else {
    int rc = accum_2d_cuda(oa, 1.0, c_a, df_a, p_a, sm_a);
    if (rc != kStatusOk) return rc;
    rc = accum_2d_cuda(ob, 1.0, c_b, df_b, p_b, sm_b);
    if (rc != kStatusOk) return rc;
  }

  symmetrize_copy_kernel<<<grid_blocks(nn_ao), 256, 0, sg.stream>>>(
      d_bar_dh.ptr, d_sym_2d.ptr, nao);
  if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
  if (cudaMemcpyAsync(bar_dh_out, d_sym_2d.ptr, nn_ao * sizeof(double),
                      cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }
  symmetrize_copy_kernel<<<grid_blocks(nn_ao), 256, 0, sg.stream>>>(
      d_bar_ds.ptr, d_sym_2d.ptr, nao);
  if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
  if (cudaMemcpyAsync(bar_ds_out, d_sym_2d.ptr, nn_ao * sizeof(double),
                      cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }

  // Pass 3: Q-chunked bar_dB -> bar_M and ao_db_sym -> bar_dr
  std::fill_n(bar_dr_out, Size(aux) * nn_ao, 0.0);
  std::vector<double> h_dr_stage(Size(q_chunk) * nn_ao);

  auto process_spin_a_tile = [&](int a0, int b, int o, double scale,
                                 const double* h_c, const double* h_df,
                                 const double* h_p) -> int {
    if (cudaMemcpyAsync(d_c_s.ptr, h_c, n_c * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
        cudaMemcpyAsync(d_p_s.ptr, h_p, nn_obs * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess ||
        cudaMemcpyAsync(d_df_tile.ptr, h_df + Size(a0) * nn_obs,
                        Size(b) * nn_obs * sizeof(double),
                        cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
    nuc_init_bar_db_kernel<<<grid_blocks(Size(b) * nn_obs), 256, 0,
                             sg.stream>>>(d_bar_db_tile.ptr, d_charge.ptr + a0,
                                          d_bar_dcharge.ptr + a0, d_p_s.ptr, b,
                                          o, obs);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;

    if (o > 0) {
      const Size ob_o = Size(obs) * o;
      add_transpose_kernel<<<grid_blocks(nn_obs), 256, 0, sg.stream>>>(
          d_p_s.ptr, d_psym_s.ptr, obs);
      if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
      if (rm_gemm_strided_batched_cuda(
              cg.handle, false, false, obs, o, obs, 1.0, d_psym_s.ptr, obs, 0LL,
              d_df_tile.ptr, obs, static_cast<long long>(nn_obs), 0.0,
              d_ex_tile.ptr, o, static_cast<long long>(ob_o), b) !=
          CUBLAS_STATUS_SUCCESS) {
        return kStatusCublasError;
      }
      nuc_sub_ex_kernel<<<grid_blocks(Size(b) * ob_o), 256, 0, sg.stream>>>(
          d_bar_db_tile.ptr, d_ex_tile.ptr, b, o, obs);
      if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
    }

    // bar_M[a0:a1, b0:b1] += (-0.5 * scale) * bar_db[a0:a1] @ df[b0:b1]^T
    for (int b0 = 0; b0 < aux; b0 += q_chunk) {
      const int b_len = std::min(q_chunk, aux - b0);
      const double* df_b_ptr = d_df_tile.ptr;
      if (b0 != a0) {
        if (cudaMemcpyAsync(d_df_tile_b.ptr, h_df + Size(b0) * nn_obs,
                            Size(b_len) * nn_obs * sizeof(double),
                            cudaMemcpyHostToDevice, sg.stream) != cudaSuccess) {
          return kStatusCudaError;
        }
        df_b_ptr = d_df_tile_b.ptr;
      }
      if (rm_gemm_cuda(cg.handle, false, true, b, b_len,
                       static_cast<int>(nn_obs), -0.5 * scale,
                       d_bar_db_tile.ptr, static_cast<int>(nn_obs), df_b_ptr,
                       static_cast<int>(nn_obs), 1.0,
                       d_bar_M.ptr + Size(a0) * aux + b0, aux) !=
          CUBLAS_STATUS_SUCCESS) {
        return kStatusCublasError;
      }
    }

    // AO back-projection: ao_raw[a] = c_s @ bar_db[a] @ c_s^T
    if (rm_gemm_strided_batched_cuda(
            cg.handle, false, false, nao, obs, obs, 1.0, d_c_s.ptr, obs, 0LL,
            d_bar_db_tile.ptr, obs, static_cast<long long>(nn_obs), 0.0,
            d_tmp_c_tile.ptr, obs, static_cast<long long>(n_c), b) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm_cuda(cg.handle, false, true, b * nao, nao, obs, 1.0,
                     d_tmp_c_tile.ptr, obs, d_c_s.ptr, obs, 0.0,
                     d_ao_raw_tile.ptr, nao) != CUBLAS_STATUS_SUCCESS) {
      return kStatusCublasError;
    }
    nuc_accum_ao_sym_kernel<<<grid_blocks(Size(b) * nn_ao), 256, 0,
                              sg.stream>>>(d_ao_raw_tile.ptr, d_ao_sym_tile.ptr,
                                           b, nao, scale);
    if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
    return kStatusOk;
  };

  for (int a0 = 0; a0 < aux; a0 += q_chunk) {
    const int b = std::min(q_chunk, aux - a0);
    if (cudaMemsetAsync(d_ao_sym_tile.ptr, 0, Size(b) * nn_ao * sizeof(double),
                        sg.stream) != cudaSuccess) {
      return kStatusCudaError;
    }
    if (same_spin) {
      int rc = process_spin_a_tile(a0, b, oa, 2.0, c_a, df_a, p_a);
      if (rc != kStatusOk) return rc;
    } else {
      int rc = process_spin_a_tile(a0, b, oa, 1.0, c_a, df_a, p_a);
      if (rc != kStatusOk) return rc;
      rc = process_spin_a_tile(a0, b, ob, 1.0, c_b, df_b, p_b);
      if (rc != kStatusOk) return rc;
    }

    // Accumulate bar_dr[t0:t1] += W[a0:a1, t0:t1]^T @ ao_sym_tile[b, nn_ao]
    for (int t0 = 0; t0 < aux; t0 += q_chunk) {
      const int t_len = std::min(q_chunk, aux - t0);
      if (rm_gemm_cuda(cg.handle, true, false, t_len, static_cast<int>(nn_ao),
                       b, 1.0, d_W.ptr + Size(a0) * aux + t0, aux,
                       d_ao_sym_tile.ptr, static_cast<int>(nn_ao), 0.0,
                       d_dr_tile.ptr, static_cast<int>(nn_ao)) !=
          CUBLAS_STATUS_SUCCESS) {
        return kStatusCublasError;
      }
      const Size t_elems = Size(t_len) * nn_ao;
      if (cudaMemcpyAsync(h_dr_stage.data(), d_dr_tile.ptr,
                          t_elems * sizeof(double), cudaMemcpyDeviceToHost,
                          sg.stream) != cudaSuccess ||
          cudaStreamSynchronize(sg.stream) != cudaSuccess) {
        return kStatusCudaError;
      }
      double* dst_dr = bar_dr_out + Size(t0) * nn_ao;
      for (Size idx = 0; idx < t_elems; ++idx) {
        dst_dr[idx] += h_dr_stage[idx];
      }
    }
  }

  // Step 4: bar_dj = sym(W^T @ bar_M @ W)
  if (rm_gemm_cuda(cg.handle, false, false, aux, aux, aux, 1.0, d_bar_M.ptr,
                   aux, d_W.ptr, aux, 0.0, d_tmp_m.ptr, aux) !=
          CUBLAS_STATUS_SUCCESS ||
      rm_gemm_cuda(cg.handle, true, false, aux, aux, aux, 1.0, d_W.ptr, aux,
                   d_tmp_m.ptr, aux, 0.0, d_bar_M.ptr, aux) !=
          CUBLAS_STATUS_SUCCESS) {
    return kStatusCublasError;
  }
  symmetrize_copy_kernel<<<grid_blocks(qq), 256, 0, sg.stream>>>(
      d_bar_M.ptr, d_tmp_m.ptr, aux);
  if (cudaGetLastError() != cudaSuccess) return kStatusCudaError;
  if (cudaMemcpyAsync(bar_dj_out, d_tmp_m.ptr, qq * sizeof(double),
                      cudaMemcpyDeviceToHost, sg.stream) != cudaSuccess ||
      cudaStreamSynchronize(sg.stream) != cudaSuccess) {
    return kStatusCudaError;
  }
  if (!is_finite_buffer(bar_ds_out, nn_ao) ||
      !is_finite_buffer(bar_dh_out, nn_ao) ||
      !is_finite_buffer(bar_dr_out, Size(aux) * nn_ao) ||
      !is_finite_buffer(bar_dj_out, qq)) {
    return kStatusNonFinite;
  }
  return kStatusOk;
  } catch (const std::bad_alloc&) {
    return kStatusOom;
  } catch (const std::length_error&) {
    return kStatusOom;
  } catch (...) {
    return kStatusInternalError;
  }
}

#endif  // GQC_SINK_HOST_REFERENCE
