#include "cc/explicit_tf12_sp.hpp"
#include "cc/explicit_tf12_sp_internal.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cc/df_triples_traversal.hpp"
#include "generated_df_occupied_triples_cuda.cuh"
#include "posthf/capacity.hpp"
#include "runtime/cuda_resources.cuh"
#include "tensor/cuda_runtime.cuh"

namespace generativeqc::cc::explicit_tf12 {
namespace {

using Clock = std::chrono::steady_clock;
using posthf::checked_add;
using posthf::checked_mul;
using generativeqc_tensor::blas_check;
using generativeqc_tensor::cuda_check;

double elapsed_seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::size_t bytes_f64(std::size_t n) { return checked_mul(n, sizeof(double)); }
std::size_t bytes_f32(std::size_t n) { return checked_mul(n, sizeof(float)); }

struct DeviceArenaTracker {
  std::size_t max_bytes{};
  std::size_t live_bytes{};
  std::size_t peak_bytes{};

  void charge(std::size_t n) {
    const auto next = checked_add(live_bytes, n);
    if (next > max_bytes)
      throw std::length_error("explicit (T)-F12 CUDA workspace exceeds numeric memory budget");
    live_bytes = next;
    if (live_bytes > peak_bytes) peak_bytes = live_bytes;
  }
  void release(std::size_t n) noexcept { live_bytes = live_bytes >= n ? live_bytes - n : 0; }
};

template <typename T>
class DeviceBuf {
 public:
  DeviceBuf() = default;
  DeviceBuf(DeviceArenaTracker& tracker, std::size_t count)
      : tracker_(&tracker), count_(count), nbytes_(checked_mul(count, sizeof(T))) {
    if (nbytes_ > 0) {
      tracker_->charge(nbytes_);
      const cudaError_t st = cudaMalloc(reinterpret_cast<void**>(&ptr_), nbytes_);
      if (st != cudaSuccess) {
        tracker_->release(nbytes_);
        ptr_ = nullptr;
        nbytes_ = 0;
        count_ = 0;
        cuda_check(st);
      }
    }
  }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
  DeviceBuf(DeviceBuf&& o) noexcept
      : tracker_(o.tracker_), ptr_(o.ptr_), count_(o.count_), nbytes_(o.nbytes_) {
    o.tracker_ = nullptr;
    o.ptr_ = nullptr;
    o.count_ = 0;
    o.nbytes_ = 0;
  }
  DeviceBuf& operator=(DeviceBuf&& o) noexcept {
    if (this != &o) {
      reset();
      tracker_ = o.tracker_;
      ptr_ = o.ptr_;
      count_ = o.count_;
      nbytes_ = o.nbytes_;
      o.tracker_ = nullptr;
      o.ptr_ = nullptr;
      o.count_ = 0;
      o.nbytes_ = 0;
    }
    return *this;
  }
  ~DeviceBuf() { reset(); }

  void reset() noexcept {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    if (tracker_ && nbytes_) tracker_->release(nbytes_);
    nbytes_ = 0;
    count_ = 0;
  }

  void upload(const T* host, std::size_t n, cudaStream_t stream) {
    if (n > count_) throw std::length_error("DeviceBuf upload overflow");
    if (n) cuda_check(cudaMemcpyAsync(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice, stream));
  }
  void download(T* host, std::size_t n, cudaStream_t stream) const {
    if (n > count_) throw std::length_error("DeviceBuf download overflow");
    if (n) cuda_check(cudaMemcpyAsync(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost, stream));
  }
  void zero(cudaStream_t stream) {
    if (nbytes_) cuda_check(cudaMemsetAsync(ptr_, 0, nbytes_, stream));
  }

  T* data() noexcept { return ptr_; }
  const T* data() const noexcept { return ptr_; }
  std::size_t size() const noexcept { return count_; }

 private:
  DeviceArenaTracker* tracker_{};
  T* ptr_{};
  std::size_t count_{};
  std::size_t nbytes_{};
};

struct BlasStreamOwner {
  cudaStream_t stream{};
  cublasHandle_t handle{};
  explicit BlasStreamOwner() {
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
      blas_check(cublasCreate(&handle));
      blas_check(cublasSetStream(handle, stream));
      blas_check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
      blas_check(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    } catch (...) {
      if (handle) cublasDestroy(handle);
      if (stream) cudaStreamDestroy(stream);
      throw;
    }
  }
  BlasStreamOwner(const BlasStreamOwner&) = delete;
  BlasStreamOwner& operator=(const BlasStreamOwner&) = delete;
  ~BlasStreamOwner() {
    if (stream) cudaStreamSynchronize(stream);
    if (handle) cublasDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
  }
};

// Row-major FP64 GEMM via cuBLAS: C[m, n] = alpha * op(A)[m, k] * op(B)[k, n] + beta * C[m, n].
inline void gpu_gemm_rm(cublasHandle_t handle, char a_trans, char b_trans, std::size_t m,
                        std::size_t n, std::size_t k, const double* a, std::size_t lda,
                        const double* b, std::size_t ldb, double* c, std::size_t ldc,
                        double alpha = 1.0, double beta = 0.0) {
  if (!m || !n || !k) return;
  const auto op_a = (a_trans == 'T' || a_trans == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  const auto op_b = (b_trans == 'T' || b_trans == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
  blas_check(cublasDgemm(handle, op_b, op_a, static_cast<int>(n), static_cast<int>(m),
                         static_cast<int>(k), &alpha, b, static_cast<int>(ldb), a,
                         static_cast<int>(lda), &beta, c, static_cast<int>(ldc)));
}

inline void gpu_gemm_rm(cublasHandle_t handle, char a_trans, char b_trans, std::size_t m,
                        std::size_t n, std::size_t k, const double* a, const double* b, double* c,
                        double alpha = 1.0, double beta = 0.0) {
  const std::size_t lda = (a_trans == 'T' || a_trans == 't') ? m : k;
  const std::size_t ldb = (b_trans == 'T' || b_trans == 't') ? k : n;
  gpu_gemm_rm(handle, a_trans, b_trans, m, n, k, a, lda, b, ldb, c, n, alpha, beta);
}

inline void gpu_gemm_cm_f64(cublasHandle_t handle, char ta, char tb, std::size_t m, std::size_t n,
                            std::size_t k, double alpha, const double* a, std::size_t lda,
                            const double* b, std::size_t ldb, double beta, double* c,
                            std::size_t ldc) {
  const auto op_a = (ta != 'N' && ta != 'n') ? CUBLAS_OP_T : CUBLAS_OP_N;
  const auto op_b = (tb != 'N' && tb != 'n') ? CUBLAS_OP_T : CUBLAS_OP_N;
  blas_check(cublasDgemm(handle, op_a, op_b, static_cast<int>(m), static_cast<int>(n),
                         static_cast<int>(k), &alpha, a, static_cast<int>(lda), b,
                         static_cast<int>(ldb), &beta, c, static_cast<int>(ldc)));
}

inline void gpu_gemm_cm_f32(cublasHandle_t handle, char ta, char tb, std::size_t m, std::size_t n,
                            std::size_t k, float alpha, const float* a, std::size_t lda,
                            const float* b, std::size_t ldb, float beta, float* c,
                            std::size_t ldc) {
  const auto op_a = (ta != 'N' && ta != 'n') ? CUBLAS_OP_T : CUBLAS_OP_N;
  const auto op_b = (tb != 'N' && tb != 'n') ? CUBLAS_OP_T : CUBLAS_OP_N;
  blas_check(cublasSgemm(handle, op_a, op_b, static_cast<int>(m), static_cast<int>(n),
                         static_cast<int>(k), &alpha, a, static_cast<int>(lda), b,
                         static_cast<int>(ldb), &beta, c, static_cast<int>(ldc)));
}

unsigned grid_blocks(std::size_t n, int threads = 256) {
  if (!n) return 1;
  return static_cast<unsigned>(std::min<std::size_t>((n + threads - 1) / threads, 65535));
}

// ---------------- CUDA Kernels ----------------

__global__ void add_c2_sym_v3_kernel(std::size_t v, double s38, double s18, const double* tmp,
                                     double* dst) {
  const std::size_t vv = v * v, v3 = v * vv;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < v3;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t a = idx / vv;
    const std::size_t rem = idx % vv;
    const std::size_t b = rem / v;
    const std::size_t f = rem % v;
    dst[idx] += s38 * tmp[idx] + s18 * tmp[(b * v + a) * v + f];
  }
}

__global__ void extract_qov_row_kernel(std::size_t q, std::size_t o, std::size_t v, std::size_t occ,
                                       const double* src, double* dst) {
  const std::size_t qv = q * v;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < qv;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t Q = idx / v;
    const std::size_t a = idx % v;
    dst[idx] = src[(Q * o + occ) * v + a];
  }
}

__global__ void cast_f64_to_f32_kernel(std::size_t n, const double* src, float* dst, int* error) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += std::size_t(blockDim.x) * gridDim.x) {
    const double val = src[i];
    if (!isfinite(val)) atomicCAS(error, 0, 11);
    dst[i] = static_cast<float>(val);
  }
}

__global__ void accumulate_f32_to_f64_kernel(std::size_t n, const float* src, double alpha,
                                             double beta, double* dst, int* error) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += std::size_t(blockDim.x) * gridDim.x) {
    const double s = static_cast<double>(src[i]);
    const double prev = (beta == 0.0) ? 0.0 : dst[i];
    const double val = alpha * s + beta * prev;
    if (!isfinite(val)) atomicCAS(error, 0, 12);
    dst[i] = val;
  }
}

__device__ void store_sum_block(double sum, double* output, int* error) {
  __shared__ double sums[256];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) *output = generativeqc_tensor::finite(sums[0], error, 3);
}

__global__ void reduce_sum_kernel(const double* values, std::size_t count, double* output,
                                  int* error) {
  double sum = 0.0;
  for (std::size_t x = threadIdx.x; x < count; x += blockDim.x) sum += values[x];
  store_sum_block(sum, output, error);
}

__global__ void extract_s_slice_kernel(std::size_t q, std::size_t ext, std::size_t obs,
                                       std::size_t fz, std::size_t o, std::size_t s_idx,
                                       const double* f_eo, const double* rf_eo, const double* bg_eo,
                                       double* f_sp, double* rf_sp, double* bg_sp, double* f_so,
                                       double* rf_so) {
  const std::size_t q_obs = q * obs;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < q_obs;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t Q = idx / obs;
    const std::size_t p = idx % obs;
    const std::size_t off = (Q * ext + s_idx) * obs + p;
    f_sp[idx] = f_eo[off];
    rf_sp[idx] = rf_eo[off];
    bg_sp[idx] = bg_eo[off];
    if (p >= fz && p < fz + o) {
      const std::size_t k = p - fz;
      f_so[Q * o + k] = f_eo[off];
      rf_so[Q * o + k] = rf_eo[off];
    }
  }
}

__global__ void reorder_s_mats_kernel(std::size_t o, std::size_t v, std::size_t o_full,
                                      std::size_t obs, std::size_t ext, const double* f_mat,
                                      const double* f_sw_mat, double* F_ipr, double* F_kbr,
                                      double* F_sw_kbr) {
  const std::size_t eo = ext * o;
  const std::size_t total = o * obs * ext;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t r = idx % ext;
    const std::size_t ip = idx / ext;
    const std::size_t i = ip / obs;
    const std::size_t p = ip % obs;
    const double val = f_mat[p * eo + r * o + i];
    F_ipr[idx] = val;
    if (p >= o_full) {
      const std::size_t b = p - o_full;
      const std::size_t kb = i * v + b;
      F_kbr[kb * ext + r] = val;
      F_sw_kbr[kb * ext + r] = f_sw_mat[i * (ext * v) + r * v + b];
    }
  }
}

__global__ void pack_p12_sub_kernel(std::size_t o, std::size_t v, std::size_t o_full,
                                    std::size_t obs, std::size_t ext, std::size_t r_p12,
                                    const double* g_mat, const double* F_ipr, const double* F_kbr,
                                    const double* F_sw_kbr, double* G_sub, double* F_sub,
                                    double* Fk_sub, double* Fsw_sub) {
  const std::size_t eo = ext * o;
  const std::size_t total = o * obs * r_p12;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t r = idx % r_p12;
    const std::size_t ip = idx / r_p12;
    const std::size_t i = ip / obs;
    const std::size_t p = ip % obs;
    G_sub[idx] = g_mat[p * eo + r * o + i];
    F_sub[idx] = F_ipr[ip * ext + r];
    if (p >= o_full) {
      const std::size_t kb = i * v + (p - o_full);
      Fk_sub[kb * r_p12 + r] = F_kbr[kb * ext + r];
      Fsw_sub[kb * r_p12 + r] = F_sw_kbr[kb * ext + r];
    }
  }
}

__global__ void slice_q_cols_kernel(std::size_t ov, std::size_t ext, std::size_t r_start,
                                    std::size_t r_len, const double* F_kbr, const double* F_sw_kbr,
                                    double* qL, double* qR) {
  const std::size_t total = ov * r_len;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t kb = idx / r_len;
    const std::size_t c = idx % r_len;
    qL[idx] = F_kbr[kb * ext + (r_start + c)];
    qR[idx] = F_sw_kbr[kb * ext + (r_start + c)];
  }
}

__global__ void slice_2d_sub_kernel(std::size_t d1_full, std::size_t s0, std::size_t n0,
                                    std::size_t s1, std::size_t n1, const double* src,
                                    double* dst) {
  const std::size_t total = n0 * n1;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t c = idx % n1;
    const std::size_t r = idx / n1;
    dst[idx] = src[(s0 + r) * d1_full + (s1 + c)];
  }
}

__global__ void dress_r_hh_al_tile_kernel(std::size_t o_full, std::size_t obs, std::size_t ext,
                                          std::size_t o, std::size_t al0, std::size_t al_len,
                                          double scale, int proj_mode, const double* f_tile_raw,
                                          double* r_hh) {
  const std::size_t eo = ext * o;
  const std::size_t oo = o * o;
  const std::size_t total = al_len * ext * oo;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t ij = idx % oo;
    const std::size_t ag = idx / oo;
    const std::size_t al_loc = ag / ext;
    const std::size_t ga = ag % ext;
    const std::size_t al = al0 + al_loc;
    const std::size_t dst_idx = (al * ext + ga) * oo + ij;
    const bool in_q = (al >= o_full && ga >= o_full) && !(al < obs && ga < obs);
    if ((proj_mode == 1 && !in_q) || (proj_mode == 2 && in_q)) {
      r_hh[dst_idx] = 0.0;
      continue;
    }
    const std::size_t i = ij / o;
    const std::size_t j = ij % o;
    const double d = f_tile_raw[(al_loc * o + i) * eo + (ga * o + j)];
    const double x = f_tile_raw[(al_loc * o + j) * eo + (ga * o + i)];
    r_hh[dst_idx] = scale * (0.375 * d + 0.125 * x);
  }
}

__global__ void transpose_ee_v_kernel(std::size_t ext, std::size_t v, const double* src,
                                      double* dst) {
  const std::size_t total = ext * ext * v;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t a = idx % v;
    const std::size_t ag = idx / v;
    const std::size_t al = ag / ext;
    const std::size_t ga = ag % ext;
    dst[idx] = src[(ga * ext + al) * v + a];
  }
}

// Dress single-k r_hp_k[ext, ext, v] (and r_hp_k_s, r_hp_k_a) from f_hp_k_raw[ext, ext*v].
// proj_mode: 0=full, 1=Q12, 2=P12.
__global__ void dress_r_hp_k_kernel(std::size_t o_full, std::size_t obs, std::size_t ext,
                                    std::size_t v, double scale, int proj_mode,
                                    const double* f_hp_k_raw, double* r_hp_k, double* r_hp_k_s,
                                    double* r_hp_k_a) {
  const std::size_t ev = ext * v;
  const std::size_t total = ext * ext * v;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t a = idx % v;
    const std::size_t bg = idx / v;
    const std::size_t be = bg / ext;
    const std::size_t ga = bg % ext;
    const bool in_q = (be >= o_full && ga >= o_full) && !(be < obs && ga < obs);
    if ((proj_mode == 1 && !in_q) || (proj_mode == 2 && in_q)) {
      r_hp_k[idx] = 0.0;
      if (r_hp_k_s) r_hp_k_s[idx] = 0.0;
      if (r_hp_k_a) r_hp_k_a[idx] = 0.0;
      continue;
    }
    const double d = f_hp_k_raw[be * ev + ga * v + a];
    const double x = f_hp_k_raw[ga * ev + be * v + a];
    const double val = scale * (0.375 * d + 0.125 * x);
    const double swp = scale * (0.375 * x + 0.125 * d);
    r_hp_k[idx] = val;
    if (r_hp_k_s) r_hp_k_s[idx] = swp;
    if (r_hp_k_a) r_hp_k_a[idx] = val - swp;
  }
}

__global__ void reorder_hp_k_for_ga_kernel(std::size_t ext, std::size_t v, const double* src,
                                           double* dst_ga_bev) {
  const std::size_t ev = ext * v;
  const std::size_t total = ext * ext * v;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t a = idx % v;
    const std::size_t bg = idx / v;
    const std::size_t be = bg / ext;
    const std::size_t ga = bg % ext;
    dst_ga_bev[ga * ev + be * v + a] = src[idx];
  }
}

__global__ void pack_r_hh_ik_tile_kernel(std::size_t ext, std::size_t o, std::size_t i0,
                                         std::size_t i_len, std::size_t k, int hh_variant,
                                         const double* r_hh, double* dst) {
  const std::size_t oo = o * o;
  const std::size_t total = i_len * ext * ext;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t ga = idx % ext;
    const std::size_t ial = idx / ext;
    const std::size_t i_loc = ial / ext;
    const std::size_t al = ial % ext;
    const std::size_t ik = (i0 + i_loc) * o + k;
    const double d = r_hh[(al * ext + ga) * oo + ik];
    if (hh_variant == 0) {
      dst[idx] = d;
    } else {
      const double sw = r_hh[(ga * ext + al) * oo + ik];
      dst[idx] = (hh_variant == 1) ? sw : (d - sw);
    }
  }
}

__global__ void reorder_s_tile_kernel(std::size_t i_len, std::size_t ext, std::size_t v,
                                      const double* src, double* dst) {
  const std::size_t ee = ext * ext;
  const std::size_t total = i_len * ee * v;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t a = idx % v;
    const std::size_t tmp = idx / v;
    const std::size_t be = tmp % ext;
    const std::size_t ial = tmp / ext;
    const std::size_t i_loc = ial / ext;
    const std::size_t al = ial % ext;
    dst[(i_loc * v + a) * ee + (al * ext + be)] = src[idx];
  }
}

__global__ void lincomb2_kernel(std::size_t n, double a, const double* x, double b, const double* y,
                                double* dst) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += std::size_t(blockDim.x) * gridDim.x)
    dst[i] = a * x[i] + b * y[i];
}

__global__ void extract_qev_b_kernel(std::size_t q, std::size_t ext, std::size_t v, std::size_t b,
                                     const double* B_ev, double* B_b) {
  const std::size_t qe = q * ext;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < qe;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t Q = idx / ext;
    const std::size_t al = idx % ext;
    B_b[idx] = B_ev[(Q * ext + al) * v + b];
  }
}

__global__ void extract_qoe_j_trans_kernel(std::size_t q, std::size_t o, std::size_t ext,
                                           std::size_t j, const double* B_oe, double* B_j_eq) {
  const std::size_t qe = q * ext;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < qe;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t Q = idx / ext;
    const std::size_t be = idx % ext;
    B_j_eq[be * q + Q] = B_oe[(Q * o + j) * ext + be];
  }
}

__global__ void extract_qoe_k_kernel(std::size_t q, std::size_t o, std::size_t ext, std::size_t k,
                                     const double* B_oe, double* B_k_qe) {
  const std::size_t qe = q * ext;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < qe;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t Q = idx / ext;
    const std::size_t be = idx % ext;
    B_k_qe[Q * ext + be] = B_oe[(Q * o + k) * ext + be];
  }
}

__global__ void reorder_bev_to_baq_kernel(std::size_t q, std::size_t ext, std::size_t v,
                                          const double* B_ev, double* B_baq) {
  const std::size_t total = q * ext * v;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t b = idx % v;
    const std::size_t tmp = idx / v;
    const std::size_t al = tmp % ext;
    const std::size_t Q = tmp / ext;
    B_baq[(b * ext + al) * q + Q] = B_ev[idx];
  }
}

// Slice 3D subblock [n0, n1, d2] from [d0_full, d1_full, d2] at offset (s0, s1).
__global__ void slice_3d_sub_kernel(std::size_t d1_full, std::size_t d2, std::size_t s0,
                                    std::size_t n0, std::size_t s1, std::size_t n1,
                                    const double* src, double* dst) {
  const std::size_t total = n0 * n1 * d2;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t k2 = idx % d2;
    const std::size_t tmp = idx / d2;
    const std::size_t k1 = tmp % n1;
    const std::size_t k0 = tmp / n1;
    dst[idx] = src[((s0 + k0) * d1_full + (s1 + k1)) * d2 + k2];
  }
}

// Transposed slice: dst[k0, k1, k2] = src[s1 + k1, s0 + k0, k2] (avoids storing W2_tile!).
__global__ void slice_3d_sub_trans_kernel(std::size_t d1_full, std::size_t d2, std::size_t s0,
                                          std::size_t n0, std::size_t s1, std::size_t n1,
                                          const double* src, double* dst) {
  const std::size_t total = n0 * n1 * d2;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t k2 = idx % d2;
    const std::size_t tmp = idx / d2;
    const std::size_t k1 = tmp % n1;
    const std::size_t k0 = tmp / n1;
    dst[idx] = src[((s1 + k1) * d1_full + (s0 + k0)) * d2 + k2];
  }
}

// Slice B_kb_sub[k, q_len] for fixed b_idx and q-range [q0, q0+q_len) from B_ov[q, o, v].
__global__ void extract_b_ov_for_b_qslice_kernel(std::size_t o, std::size_t v, std::size_t b,
                                                 std::size_t q0, std::size_t q_len,
                                                 const double* B_ov, double* B_kq) {
  const std::size_t oq = o * q_len;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < oq;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t k = idx / q_len;
    const std::size_t ql = idx % q_len;
    B_kq[idx] = B_ov[((q0 + ql) * o + k) * v + b];
  }
}

// Pack w_b[oo, v] = w_dir[i, j, a, b] for fixed b.
__global__ void extract_w_for_b_kernel(std::size_t oo, std::size_t v, std::size_t b,
                                       const double* w_4d, double* w_b) {
  const std::size_t oov = oo * v;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < oov;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t ij = idx / v;
    const std::size_t a = idx % v;
    w_b[idx] = w_4d[(ij * v + a) * v + b];
  }
}

// Extract w_ex_j[v, i_len*v] = w_4d[i0+i_loc, j, a, b] for fixed j and i-tile!
__global__ void extract_w_for_j_itile_kernel(std::size_t o, std::size_t v, std::size_t i0,
                                             std::size_t i_len, std::size_t j, const double* w_4d,
                                             double* dst_b_ia) {
  const std::size_t total = v * i_len * v;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t ia = idx % (i_len * v);
    const std::size_t b = idx / (i_len * v);
    const std::size_t i_loc = ia / v;
    const std::size_t a = ia % v;
    dst_b_ia[idx] = w_4d[(((i0 + i_loc) * o + j) * v + a) * v + b];
  }
}

// Extract D_tile[q, i_len*v] from D_co[q, o*v] at offset i0.
__global__ void extract_d_co_tile_kernel(std::size_t q, std::size_t o, std::size_t v,
                                         std::size_t i0, std::size_t i_len, const double* D_co,
                                         double* D_tile) {
  const std::size_t iv = i_len * v;
  const std::size_t total = q * iv;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t ia = idx % iv;
    const std::size_t Q = idx / iv;
    D_tile[idx] = D_co[Q * (o * v) + i0 * v + ia];
  }
}

// Extract G_sub[n_al, n_be] from G_kb[k, ext, ext] at offsets (al0, be0) for fixed k.
__global__ void extract_g_sub_for_k_kernel(std::size_t ext, std::size_t k, std::size_t al0,
                                           std::size_t n_al, std::size_t be0, std::size_t n_be,
                                           const double* G_k_ee, double* G_sub) {
  const std::size_t total = n_al * n_be;
  const std::size_t base = k * ext * ext;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t be = idx % n_be;
    const std::size_t al = idx / n_be;
    G_sub[idx] = G_k_ee[base + (al0 + al) * ext + (be0 + be)];
  }
}

// Dot product of two device arrays of length n via block reduction.
__global__ void dot_partial_kernel(std::size_t n, const double* a, const double* b,
                                   double* partials) {
  double sum = 0.0;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += std::size_t(blockDim.x) * gridDim.x)
    sum += a[i] * b[i];
  __shared__ double sh[256];
  sh[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = sh[0];
}

double gpu_dot(std::size_t n, const double* d_a, const double* d_b, double* d_partials,
               std::vector<double>& h_partials, cudaStream_t stream) {
  if (!n) return 0.0;
  const unsigned nb = static_cast<unsigned>(std::min<std::size_t>((n + 255) / 256, 1024));
  dot_partial_kernel<<<nb, 256, 0, stream>>>(n, d_a, d_b, d_partials);
  cuda_check(cudaGetLastError());
  h_partials.resize(nb);
  cuda_check(cudaMemcpyAsync(h_partials.data(), d_partials, nb * sizeof(double),
                             cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaStreamSynchronize(stream));
  double s = 0.0;
  for (unsigned i = 0; i < nb; ++i) s += h_partials[i];
  return s;
}

// Helper: build single-k dressed r_hp_k, r_hp_k_s, r_hp_k_a on GPU from pre-packed d_F_eo_k[o, ext, q]
// and d_F_ev_q[q, ext*v].
void build_r_hp_k_gpu(cublasHandle_t handle, cudaStream_t stream, std::size_t o_full,
                      std::size_t obs, std::size_t ext, std::size_t v, std::size_t q, std::size_t k,
                      double scale, int proj_mode, const double* d_F_eo_k, const double* d_Rf_eo_k,
                      const double* d_F_ev_q, const double* d_Rf_ev_q, double* d_f_hp_k_raw,
                      double* d_r_hp_k, double* d_r_hp_k_s, double* d_r_hp_k_a) {
  const std::size_t eq = checked_mul(ext, q), ev = checked_mul(ext, v);
  const double* Fk = d_F_eo_k + k * eq;
  const double* Rfk = d_Rf_eo_k + k * eq;
  // f_hp_k_raw[ext, ext*v] = Fk[ext, q] @ Rf_ev_q[q, ev] + Rfk[ext, q] @ F_ev_q[q, ev]
  gpu_gemm_rm(handle, 'N', 'N', ext, ev, q, Fk, d_Rf_ev_q, d_f_hp_k_raw, 1.0, 0.0);
  gpu_gemm_rm(handle, 'N', 'N', ext, ev, q, Rfk, d_F_ev_q, d_f_hp_k_raw, 1.0, 1.0);
  dress_r_hp_k_kernel<<<grid_blocks(ext * ev), 256, 0, stream>>>(
      o_full, obs, ext, v, scale, proj_mode, d_f_hp_k_raw, d_r_hp_k, d_r_hp_k_s, d_r_hp_k_a);
  cuda_check(cudaGetLastError());
}

// Adaptive GPU-resident or q-tiled view of b_g_ext_ext[q, ext*ext].
struct BgEeSource {
  const double* d_full{};  // non-null when full b_g_ee fits in VRAM
  const double* h_full{};  // host pointer when q-streaming is needed
  std::size_t q_tile{};
};

// Build dressed r_hh[ext, ext, o, o] in-place into d_r_hh using al-tiling (zero full f_hh_raw!).
void build_r_hh_gpu(cublasHandle_t handle, cudaStream_t stream, DeviceArenaTracker& tracker,
                    std::size_t o_full, std::size_t fz, std::size_t obs, std::size_t ext,
                    std::size_t q, double scale, int proj_mode,
                    const std::vector<double>& F_eo_t, const std::vector<double>& Rf_eo_t,
                    const std::vector<double>& F_eo_q, const std::vector<double>& Rf_eo_q,
                    double* d_r_hh) {
  const std::size_t o = o_full - fz;
  const std::size_t oo = checked_mul(o, o), eo = checked_mul(ext, o);
  DeviceBuf<double> d_F_eo_q(tracker, F_eo_q.size()), d_Rf_eo_q(tracker, Rf_eo_q.size());
  d_F_eo_q.upload(F_eo_q.data(), F_eo_q.size(), stream);
  d_Rf_eo_q.upload(Rf_eo_q.data(), Rf_eo_q.size(), stream);

  const std::size_t one_al_bytes = bytes_f64(checked_mul(ext, oo));
  const std::size_t avail = tracker.max_bytes > tracker.live_bytes
                                ? (tracker.max_bytes - tracker.live_bytes)
                                : 0;
  const std::size_t cap = std::min<std::size_t>(512ULL << 20, avail / 3);
  std::size_t al_tile = std::max<std::size_t>(1, cap / std::max<std::size_t>(1, one_al_bytes));
  al_tile = std::min(al_tile, ext);

  DeviceBuf<double> d_F_tile(tracker, checked_mul(checked_mul(al_tile, o), q));
  DeviceBuf<double> d_Rf_tile(tracker, checked_mul(checked_mul(al_tile, o), q));
  DeviceBuf<double> d_raw_tile(tracker, checked_mul(checked_mul(al_tile, o), eo));

  for (std::size_t al0 = 0; al0 < ext; al0 += al_tile) {
    const std::size_t al_len = std::min(al_tile, ext - al0);
    const std::size_t alo = checked_mul(al_len, o);
    d_F_tile.upload(F_eo_t.data() + al0 * o * q, checked_mul(alo, q), stream);
    d_Rf_tile.upload(Rf_eo_t.data() + al0 * o * q, checked_mul(alo, q), stream);
    gpu_gemm_rm(handle, 'N', 'N', alo, eo, q, d_F_tile.data(), d_Rf_eo_q.data(), d_raw_tile.data(),
                1.0, 0.0);
    gpu_gemm_rm(handle, 'N', 'N', alo, eo, q, d_Rf_tile.data(), d_F_eo_q.data(), d_raw_tile.data(),
                1.0, 1.0);
    dress_r_hh_al_tile_kernel<<<grid_blocks(al_len * ext * oo), 256, 0, stream>>>(
        o_full, obs, ext, o, al0, al_len, scale, proj_mode, d_raw_tile.data(), d_r_hh);
    cuda_check(cudaGetLastError());
  }
  cuda_check(cudaStreamSynchronize(stream));
}

struct M2GpuTerm {
  int hh_variant;    // 0 = r_hh, 1 = r_hh_s, 2 = r_hh_a
  int r_hp_variant;  // 0 = r_hp, 1 = r_hp_s, 2 = r_hp_a
  double co_weight;
  double ex_weight;
  bool ex_uses_swp;
};

double evaluate_z2_m2_gpu(cublasHandle_t handle, cudaStream_t stream, DeviceArenaTracker& tracker,
                          std::size_t o_full, std::size_t fz, std::size_t obs, std::size_t ext,
                          std::size_t q, double scale, int hp_proj_mode, const double* d_r_hh,
                          const std::vector<M2GpuTerm>& terms, const double* d_F_eo_k,
                          const double* d_Rf_eo_k, const double* d_F_ev_q, const double* d_Rf_ev_q,
                          const BgEeSource& bg_ee, const double* d_D_co, const double* d_B_oe,
                          const double* d_B_baq, const double* d_w_dir, const double* d_w_swp,
                          double* d_partials, std::vector<double>& h_partials) {
  if (terms.empty()) return 0.0;
  const std::size_t o = o_full - fz, v = obs - o_full;
  const std::size_t ee = checked_mul(ext, ext), ev = checked_mul(ext, v);
  const bool any_co = std::any_of(terms.begin(), terms.end(),
                                  [](const M2GpuTerm& t) { return t.co_weight != 0.0; });
  const bool any_ex = std::any_of(terms.begin(), terms.end(),
                                  [](const M2GpuTerm& t) { return t.ex_weight != 0.0; });

  DeviceBuf<double> d_bg_chunk;
  std::size_t q_chunk = q;
  if (any_co && !bg_ee.d_full) {
    q_chunk = std::min(q, std::max<std::size_t>(1, bg_ee.q_tile));
    d_bg_chunk = DeviceBuf<double>(tracker, checked_mul(q_chunk, ee));
  }

  // Size i_tile to keep S_tile + S_col under ~1.4 GiB (or 25% of remaining budget).
  const std::size_t s_one = bytes_f64(checked_mul(ee, v));
  const std::size_t avail = tracker.max_bytes > tracker.live_bytes
                                ? (tracker.max_bytes - tracker.live_bytes)
                                : 0;
  const std::size_t cap_bytes = std::min<std::size_t>(1400ULL << 20, avail / 3);
  std::size_t i_tile =
      std::max<std::size_t>(1, cap_bytes / std::max<std::size_t>(1, 2 * s_one));
  i_tile = std::min(i_tile, o);

  DeviceBuf<double> d_f_hp_k_raw(tracker, checked_mul(ext, ev));
  DeviceBuf<double> d_r_hp_k(tracker, checked_mul(ee, v));
  DeviceBuf<double> d_r_hp_k_s(tracker, checked_mul(ee, v));
  DeviceBuf<double> d_r_hp_k_a(tracker, checked_mul(ee, v));
  DeviceBuf<double> d_r_hp_ga(tracker, checked_mul(ee, v));
  DeviceBuf<double> d_r_hh_ik(tracker, checked_mul(i_tile, ee));
  DeviceBuf<double> d_S_tile(tracker, checked_mul(i_tile, checked_mul(ee, v)));
  DeviceBuf<double> d_S_col(tracker, checked_mul(i_tile, checked_mul(ee, v)));

  DeviceBuf<double> d_L_tile, d_D_tile;
  if (any_co) {
    d_L_tile = DeviceBuf<double>(tracker, checked_mul(q_chunk, checked_mul(i_tile, v)));
    d_D_tile = DeviceBuf<double>(tracker, checked_mul(q, checked_mul(i_tile, v)));
  }

  DeviceBuf<double> d_U_j, d_B_j_eq, d_X_j, d_w_slice;
  if (any_ex) {
    d_U_j = DeviceBuf<double>(tracker, checked_mul(v, ee));
    d_B_j_eq = DeviceBuf<double>(tracker, checked_mul(ext, q));
    d_X_j = DeviceBuf<double>(tracker, checked_mul(ev, q));
    d_w_slice = DeviceBuf<double>(tracker, checked_mul(v, checked_mul(i_tile, v)));
  }

  double total_scalar = 0.0;
  for (std::size_t i0 = 0; i0 < o; i0 += i_tile) {
    const std::size_t i_len = std::min(i_tile, o - i0);
    const std::size_t iv = checked_mul(i_len, v);
    const std::size_t ie = checked_mul(i_len, ext);

    if (any_co) {
      extract_d_co_tile_kernel<<<grid_blocks(q * iv), 256, 0, stream>>>(
          q, o, v, i0, i_len, d_D_co, d_D_tile.data());
      cuda_check(cudaGetLastError());
    }

    for (const auto& term : terms) {
      d_S_tile.zero(stream);
      for (std::size_t k = 0; k < o; ++k) {
        build_r_hp_k_gpu(handle, stream, o_full, obs, ext, v, q, k, scale, hp_proj_mode, d_F_eo_k,
                         d_Rf_eo_k, d_F_ev_q, d_Rf_ev_q, d_f_hp_k_raw.data(), d_r_hp_k.data(),
                         d_r_hp_k_s.data(), d_r_hp_k_a.data());
        const double* rR_src = (term.r_hp_variant == 0)
                                   ? d_r_hp_k.data()
                                   : (term.r_hp_variant == 1 ? d_r_hp_k_s.data()
                                                             : d_r_hp_k_a.data());
        reorder_hp_k_for_ga_kernel<<<grid_blocks(ee * v), 256, 0, stream>>>(ext, v, rR_src,
                                                                            d_r_hp_ga.data());
        pack_r_hh_ik_tile_kernel<<<grid_blocks(i_len * ee), 256, 0, stream>>>(
            ext, o, i0, i_len, k, term.hh_variant, d_r_hh, d_r_hh_ik.data());
        cuda_check(cudaGetLastError());
        gpu_gemm_rm(handle, 'N', 'N', ie, ev, ext, d_r_hh_ik.data(), d_r_hp_ga.data(),
                    d_S_tile.data(), 1.0, 1.0);
      }

      reorder_s_tile_kernel<<<grid_blocks(i_len * ee * v), 256, 0, stream>>>(
          i_len, ext, v, d_S_tile.data(), d_S_col.data());
      cuda_check(cudaGetLastError());

      if (term.co_weight != 0.0) {
        double d_val = 0.0;
        if (bg_ee.d_full) {
          gpu_gemm_rm(handle, 'N', 'T', q, iv, ee, bg_ee.d_full, d_S_col.data(), d_L_tile.data(),
                      1.0, 0.0);
          d_val = gpu_dot(checked_mul(q, iv), d_L_tile.data(), d_D_tile.data(), d_partials,
                          h_partials, stream);
        } else {
          for (std::size_t q0 = 0; q0 < q; q0 += q_chunk) {
            const std::size_t q_len = std::min(q_chunk, q - q0);
            d_bg_chunk.upload(bg_ee.h_full + q0 * ee, checked_mul(q_len, ee), stream);
            gpu_gemm_rm(handle, 'N', 'T', q_len, iv, ee, d_bg_chunk.data(), d_S_col.data(),
                        d_L_tile.data(), 1.0, 0.0);
            d_val += gpu_dot(checked_mul(q_len, iv), d_L_tile.data(), d_D_tile.data() + q0 * iv,
                             d_partials, h_partials, stream);
          }
        }
        total_scalar += term.co_weight * d_val;
      }

      if (term.ex_weight != 0.0) {
        const double* w_use = term.ex_uses_swp ? d_w_swp : d_w_dir;
        double ex_acc = 0.0;
        for (std::size_t j = 0; j < o; ++j) {
          extract_w_for_j_itile_kernel<<<grid_blocks(v * iv), 256, 0, stream>>>(
              o, v, i0, i_len, j, w_use, d_w_slice.data());
          cuda_check(cudaGetLastError());
          gpu_gemm_rm(handle, 'N', 'N', v, ee, iv, d_w_slice.data(), d_S_col.data(), d_U_j.data(),
                      1.0, 0.0);
          extract_qoe_j_trans_kernel<<<grid_blocks(q * ext), 256, 0, stream>>>(
              q, o, ext, j, d_B_oe, d_B_j_eq.data());
          cuda_check(cudaGetLastError());
          gpu_gemm_rm(handle, 'N', 'N', ev, q, ext, d_U_j.data(), d_B_j_eq.data(), d_X_j.data(),
                      1.0, 0.0);
          ex_acc +=
              gpu_dot(checked_mul(ev, q), d_X_j.data(), d_B_baq, d_partials, h_partials, stream);
        }
        total_scalar += term.ex_weight * ex_acc;
      }
    }
  }
  return total_scalar;
}

// GPU helper for M1_ex: computes +4.0 * m1(rQ_hh_s, g_ex, 2*rQ_hp - rQ_hp_s) * w_dir
// using d_rQ_hh directly + tiny 10 MiB transpose (zero d_rQ_hh_s allocation!).
double evaluate_z2_m1_ex_gpu(cublasHandle_t handle, cudaStream_t stream,
                             DeviceArenaTracker& tracker, std::size_t o_full, std::size_t fz,
                             std::size_t obs, std::size_t ext, std::size_t q, double scale,
                             const double* d_rQ_hh, const double* d_F_eo_k,
                             const double* d_Rf_eo_k, const double* d_F_ev_q,
                             const double* d_Rf_ev_q, const double* d_B_oe, const double* d_B_ev,
                             const double* d_w_dir, double* d_partials,
                             std::vector<double>& h_partials) {
  const std::size_t o = o_full - fz, v = obs - o_full;
  const std::size_t oo = checked_mul(o, o), ee = checked_mul(ext, ext), ev = checked_mul(ext, v);

  DeviceBuf<double> d_v_ex(tracker, checked_mul(q, ev));
  d_v_ex.zero(stream);
  {
    DeviceBuf<double> d_f_hp_k_raw(tracker, checked_mul(ext, ev));
    DeviceBuf<double> d_r_hp_k(tracker, checked_mul(ee, v));
    DeviceBuf<double> d_r_hp_k_s(tracker, checked_mul(ee, v));
    DeviceBuf<double> d_r_comb(tracker, checked_mul(ee, v));
    DeviceBuf<double> d_B_k_qe(tracker, checked_mul(q, ext));
    for (std::size_t k = 0; k < o; ++k) {
      build_r_hp_k_gpu(handle, stream, o_full, obs, ext, v, q, k, scale, /*Q12*/ 1, d_F_eo_k,
                       d_Rf_eo_k, d_F_ev_q, d_Rf_ev_q, d_f_hp_k_raw.data(), d_r_hp_k.data(),
                       d_r_hp_k_s.data(), nullptr);
      lincomb2_kernel<<<grid_blocks(ee * v), 256, 0, stream>>>(
          ee * v, 2.0, d_r_hp_k.data(), -1.0, d_r_hp_k_s.data(), d_r_comb.data());
      extract_qoe_k_kernel<<<grid_blocks(q * ext), 256, 0, stream>>>(
          q, o, ext, k, d_B_oe, d_B_k_qe.data());
      cuda_check(cudaGetLastError());
      gpu_gemm_rm(handle, 'N', 'N', q, ev, ext, d_B_k_qe.data(), d_r_comb.data(), d_v_ex.data(),
                  1.0, 1.0);
    }
  }

  DeviceBuf<double> d_w_b(tracker, checked_mul(oo, v));
  DeviceBuf<double> d_W_b(tracker, checked_mul(ee, v));
  DeviceBuf<double> d_W_bs(tracker, checked_mul(ee, v));
  DeviceBuf<double> d_B_b(tracker, checked_mul(q, ext));
  DeviceBuf<double> d_Y_b(tracker, checked_mul(q, ev));
  double acc = 0.0;
  for (std::size_t b = 0; b < v; ++b) {
    extract_w_for_b_kernel<<<grid_blocks(oo * v), 256, 0, stream>>>(oo, v, b, d_w_dir, d_w_b.data());
    extract_qev_b_kernel<<<grid_blocks(q * ext), 256, 0, stream>>>(q, ext, v, b, d_B_ev,
                                                                   d_B_b.data());
    cuda_check(cudaGetLastError());
    gpu_gemm_rm(handle, 'N', 'N', ee, v, oo, d_rQ_hh, d_w_b.data(), d_W_b.data(), 1.0, 0.0);
    transpose_ee_v_kernel<<<grid_blocks(ee * v), 256, 0, stream>>>(ext, v, d_W_b.data(),
                                                                   d_W_bs.data());
    cuda_check(cudaGetLastError());
    gpu_gemm_rm(handle, 'N', 'N', q, ev, ext, d_B_b.data(), d_W_bs.data(), d_Y_b.data(), 1.0, 0.0);
    acc += gpu_dot(checked_mul(q, ev), d_Y_b.data(), d_v_ex.data(), d_partials, h_partials, stream);
  }
  return 4.0 * acc;
}

struct SubBlock3 {
  std::size_t al0, n_al;
  std::size_t ga0, n_ga;
  std::size_t be0, n_be;
};

double evaluate_z2_m1_co_pair_gpu(cublasHandle_t handle, cudaStream_t stream,
                                  DeviceArenaTracker& tracker, std::size_t o_full, std::size_t fz,
                                  std::size_t obs, std::size_t ext, std::size_t q, double scale,
                                  int r_proj_mode, const std::vector<SubBlock3>& blocks,
                                  double outer_sign, const double* d_rL_hh,
                                  const double* d_F_eo_k, const double* d_Rf_eo_k,
                                  const double* d_F_ev_q, const double* d_Rf_ev_q,
                                  const double* d_B_ov, const BgEeSource& bg_ee,
                                  const double* d_w_dir, double* d_partials,
                                  std::vector<double>& h_partials) {
  const std::size_t o = o_full - fz, v = obs - o_full;
  const std::size_t oo = checked_mul(o, o), ee = checked_mul(ext, ext), ev = checked_mul(ext, v);
  if (blocks.empty()) return 0.0;

  std::size_t max_al_ga = 0, max_be_ga = 0, max_al_be = 0;
  for (const auto& blk : blocks) {
    max_al_ga = std::max(max_al_ga, checked_mul(blk.n_al, blk.n_ga));
    max_be_ga = std::max(max_be_ga, checked_mul(blk.n_be, blk.n_ga));
    max_al_be = std::max(max_al_be, checked_mul(blk.n_al, blk.n_be));
  }

  DeviceBuf<double> d_bg_chunk;
  std::size_t q_chunk = q;
  if (!bg_ee.d_full) {
    q_chunk = std::min(q, std::max<std::size_t>(1, bg_ee.q_tile));
    d_bg_chunk = DeviceBuf<double>(tracker, checked_mul(q_chunk, ee));
  }

  const std::size_t w_one_bytes = bytes_f64(checked_mul(ee, v));
  const std::size_t g_one_bytes = bytes_f64(checked_mul(o, ee));
  const std::size_t avail = tracker.max_bytes > tracker.live_bytes
                                ? (tracker.max_bytes - tracker.live_bytes)
                                : 0;
  const std::size_t b_cap_bytes = std::min<std::size_t>(2400ULL << 20, avail / 2);
  std::size_t b_tile = std::max<std::size_t>(
      1, b_cap_bytes / std::max<std::size_t>(1, checked_add(w_one_bytes, g_one_bytes)));
  b_tile = std::min(b_tile, v);

  DeviceBuf<double> d_w_b(tracker, checked_mul(oo, v));
  DeviceBuf<double> d_W1_tile(tracker, checked_mul(b_tile, checked_mul(ee, v)));
  DeviceBuf<double> d_B_kq(tracker, checked_mul(o, q_chunk));
  DeviceBuf<double> d_G_tile(tracker, checked_mul(b_tile, checked_mul(o, ee)));
  DeviceBuf<double> d_f_hp_k_raw(tracker, checked_mul(ext, ev));
  DeviceBuf<double> d_r_hp_k(tracker, checked_mul(ee, v));
  DeviceBuf<double> d_r_hp_k_s(tracker, checked_mul(ee, v));

  DeviceBuf<double> d_W_sub(tracker, checked_mul(max_al_ga, v));
  DeviceBuf<double> d_R_sub(tracker, checked_mul(max_be_ga, v));
  DeviceBuf<double> d_T_sub(tracker, max_al_be);
  DeviceBuf<double> d_G_sub(tracker, max_al_be);

  double acc = 0.0;
  for (std::size_t b0 = 0; b0 < v; b0 += b_tile) {
    const std::size_t b_len = std::min(b_tile, v - b0);
    // Build W1_b for all b in [b0, b0 + b_len).
    for (std::size_t bl = 0; bl < b_len; ++bl) {
      const std::size_t b = b0 + bl;
      extract_w_for_b_kernel<<<grid_blocks(oo * v), 256, 0, stream>>>(oo, v, b, d_w_dir,
                                                                      d_w_b.data());
      cuda_check(cudaGetLastError());
      gpu_gemm_rm(handle, 'N', 'N', ee, v, oo, d_rL_hh, d_w_b.data(),
                  d_W1_tile.data() + bl * ee * v, 1.0, 0.0);
    }

    // Build G_tile[bl, k, ee] = B_kq[k, q] @ b_g_ee[q, ee] (resident OR q-tiled!).
    if (bg_ee.d_full) {
      for (std::size_t bl = 0; bl < b_len; ++bl) {
        const std::size_t b = b0 + bl;
        extract_b_ov_for_b_qslice_kernel<<<grid_blocks(o * q), 256, 0, stream>>>(
            o, v, b, 0, q, d_B_ov, d_B_kq.data());
        cuda_check(cudaGetLastError());
        gpu_gemm_rm(handle, 'N', 'N', o, ee, q, d_B_kq.data(), bg_ee.d_full,
                    d_G_tile.data() + bl * o * ee, 1.0, 0.0);
      }
    } else {
      for (std::size_t q0 = 0; q0 < q; q0 += q_chunk) {
        const std::size_t q_len = std::min(q_chunk, q - q0);
        d_bg_chunk.upload(bg_ee.h_full + q0 * ee, checked_mul(q_len, ee), stream);
        const double beta_acc = (q0 == 0) ? 0.0 : 1.0;
        for (std::size_t bl = 0; bl < b_len; ++bl) {
          const std::size_t b = b0 + bl;
          extract_b_ov_for_b_qslice_kernel<<<grid_blocks(o * q_len), 256, 0, stream>>>(
              o, v, b, q0, q_len, d_B_ov, d_B_kq.data());
          cuda_check(cudaGetLastError());
          gpu_gemm_rm(handle, 'N', 'N', o, ee, q_len, d_B_kq.data(), d_bg_chunk.data(),
                      d_G_tile.data() + bl * o * ee, 1.0, beta_acc);
        }
      }
    }

    for (std::size_t k = 0; k < o; ++k) {
      build_r_hp_k_gpu(handle, stream, o_full, obs, ext, v, q, k, scale, r_proj_mode, d_F_eo_k,
                       d_Rf_eo_k, d_F_ev_q, d_Rf_ev_q, d_f_hp_k_raw.data(), d_r_hp_k.data(),
                       d_r_hp_k_s.data(), nullptr);
      for (const auto& blk : blocks) {
        if (!blk.n_al || !blk.n_ga || !blk.n_be) continue;
        const std::size_t k_dim = checked_mul(blk.n_ga, v);
        const std::size_t ab_sz = checked_mul(blk.n_al, blk.n_be);

        slice_3d_sub_kernel<<<grid_blocks(blk.n_be * k_dim), 256, 0, stream>>>(
            ext, v, blk.be0, blk.n_be, blk.ga0, blk.n_ga, d_r_hp_k_s.data(), d_R_sub.data());
        cuda_check(cudaGetLastError());
        for (std::size_t bl = 0; bl < b_len; ++bl) {
          slice_3d_sub_kernel<<<grid_blocks(blk.n_al * k_dim), 256, 0, stream>>>(
              ext, v, blk.al0, blk.n_al, blk.ga0, blk.n_ga, d_W1_tile.data() + bl * ee * v,
              d_W_sub.data());
          extract_g_sub_for_k_kernel<<<grid_blocks(ab_sz), 256, 0, stream>>>(
              ext, k, blk.al0, blk.n_al, blk.be0, blk.n_be, d_G_tile.data() + bl * o * ee,
              d_G_sub.data());
          cuda_check(cudaGetLastError());
          gpu_gemm_rm(handle, 'N', 'T', blk.n_al, blk.n_be, k_dim, d_W_sub.data(), d_R_sub.data(),
                      d_T_sub.data(), 1.0, 0.0);
          acc += gpu_dot(ab_sz, d_T_sub.data(), d_G_sub.data(), d_partials, h_partials, stream);
        }

        slice_3d_sub_kernel<<<grid_blocks(blk.n_be * k_dim), 256, 0, stream>>>(
            ext, v, blk.be0, blk.n_be, blk.ga0, blk.n_ga, d_r_hp_k.data(), d_R_sub.data());
        cuda_check(cudaGetLastError());
        for (std::size_t bl = 0; bl < b_len; ++bl) {
          slice_3d_sub_trans_kernel<<<grid_blocks(blk.n_al * k_dim), 256, 0, stream>>>(
              ext, v, blk.al0, blk.n_al, blk.ga0, blk.n_ga, d_W1_tile.data() + bl * ee * v,
              d_W_sub.data());
          extract_g_sub_for_k_kernel<<<grid_blocks(ab_sz), 256, 0, stream>>>(
              ext, k, blk.al0, blk.n_al, blk.be0, blk.n_be, d_G_tile.data() + bl * o * ee,
              d_G_sub.data());
          cuda_check(cudaGetLastError());
          gpu_gemm_rm(handle, 'N', 'T', blk.n_al, blk.n_be, k_dim, d_W_sub.data(), d_R_sub.data(),
                      d_T_sub.data(), 1.0, 0.0);
          acc += gpu_dot(ab_sz, d_T_sub.data(), d_G_sub.data(), d_partials, h_partials, stream);
        }
      }
    }
  }
  return outer_sign * acc;
}

}  // namespace

double evaluate_explicit_tf12_pt_cuda(const PtInputs& in, std::size_t max_bytes, int device,
                                      std::size_t max_panel_buffers,
                                      runtime::PrecisionDirective admitted_w,
                                      Breakdown* telemetry_out) {
  const auto started = Clock::now();
  const std::size_t o = in.o, v = in.v, q = in.q_mp2;
  if (!o || !v || !q || !max_bytes || device < 0 || !std::isfinite(in.denominator_threshold) ||
      in.denominator_threshold <= 0.0 || !max_panel_buffers || max_panel_buffers > 3)
    throw std::invalid_argument("invalid explicit (T)-F12 CUDA pt dimensions or budget");
  if (in.ovvv_direct != nullptr || in.c2.c2_ab_direct != nullptr)
    throw std::invalid_argument(
        "CUDA explicit (T)-F12 requires 3-index DF factors, not direct 4-index");

  const std::size_t oo = checked_mul(o, o);
  const std::size_t ov = checked_mul(o, v);
  const std::size_t vv = checked_mul(v, v);
  const std::size_t v3 = checked_mul(v, vv);
  const std::size_t tiles = checked_mul(checked_mul(o, checked_add(o, 1)), checked_add(o, 2)) / 6;
  const unsigned epilogue_blocks =
      static_cast<unsigned>(std::min<std::size_t>(1 + (v3 - 1) / 256, 65535));
  const std::size_t qf = in.c2.q_f12;

  const double max_o = *std::max_element(in.eps_o, in.eps_o + o);
  const double min_v = *std::min_element(in.eps_v, in.eps_v + v);
  if (!(max_o < min_v) || 3.0 * (min_v - max_o) <= in.denominator_threshold)
    throw std::invalid_argument("noncanonical or near-zero explicit (T)-F12 denominator");

  const bool use_fp32_w = (admitted_w.compute_dtype == runtime::PrecisionDtype::Fp32);
  constexpr std::size_t provider_allowance = 128ULL << 20;
  if (max_bytes <= provider_allowance)
    throw std::length_error("explicit (T)-F12 CUDA budget below provider allowance");

  runtime::CudaDeviceScope device_scope(device, cuda_check);
  BlasStreamOwner blas;
  DeviceArenaTracker tracker{max_bytes - provider_allowance, 0, 0};

  DeviceBuf<double> d_bov(tracker, checked_mul(q, ov));
  DeviceBuf<double> d_bvv(tracker, checked_mul(q, vv));
  DeviceBuf<double> d_ovoo(tracker, checked_mul(ov, oo));
  DeviceBuf<double> d_ovov(tracker, checked_mul(ov, ov));
  DeviceBuf<double> d_fov(tracker, ov);
  DeviceBuf<double> d_t1(tracker, ov);
  DeviceBuf<double> d_t2(tracker, checked_mul(oo, vv));
  DeviceBuf<double> d_eps_o(tracker, o);
  DeviceBuf<double> d_eps_v(tracker, v);
  d_bov.upload(in.bov, d_bov.size(), blas.stream);
  d_bvv.upload(in.bvv, d_bvv.size(), blas.stream);
  d_ovoo.upload(in.ovoo, d_ovoo.size(), blas.stream);
  d_ovov.upload(in.ovov, d_ovov.size(), blas.stream);
  d_fov.upload(in.fov, d_fov.size(), blas.stream);
  d_t1.upload(in.t1, d_t1.size(), blas.stream);
  d_t2.upload(in.t2, d_t2.size(), blas.stream);
  d_eps_o.upload(in.eps_o, o, blas.stream);
  d_eps_v.upload(in.eps_v, v, blas.stream);

  DeviceBuf<double> d_f_vo, d_r_vo, d_f_vv, d_r_vv, d_af_vo, d_ar_vo, d_af_vv, d_ar_vv;
  DeviceBuf<double> d_c2_row, d_c2_tmp;
  if (qf > 0) {
    const std::size_t qov = checked_mul(qf, ov), qvv = checked_mul(qf, vv);
    d_f_vo = DeviceBuf<double>(tracker, qov);
    d_r_vo = DeviceBuf<double>(tracker, qov);
    d_f_vv = DeviceBuf<double>(tracker, qvv);
    d_r_vv = DeviceBuf<double>(tracker, qvv);
    d_af_vo = DeviceBuf<double>(tracker, qov);
    d_ar_vo = DeviceBuf<double>(tracker, qov);
    d_af_vv = DeviceBuf<double>(tracker, qvv);
    d_ar_vv = DeviceBuf<double>(tracker, qvv);
    d_c2_row = DeviceBuf<double>(tracker, checked_mul(qf, v));
    d_c2_tmp = DeviceBuf<double>(tracker, v3);
    d_f_vo.upload(in.c2.f_vo, qov, blas.stream);
    d_r_vo.upload(in.c2.r_vo, qov, blas.stream);
    d_f_vv.upload(in.c2.f_vv, qvv, blas.stream);
    d_r_vv.upload(in.c2.r_vv, qvv, blas.stream);
    d_af_vo.upload(in.c2.af_vo, qov, blas.stream);
    d_ar_vo.upload(in.c2.ar_vo, qov, blas.stream);
    d_af_vv.upload(in.c2.af_vv, qvv, blas.stream);
    d_ar_vv.upload(in.c2.ar_vv, qvv, blas.stream);
  }

  std::size_t panel_cap = std::min(o, max_panel_buffers);
  const std::size_t fixed_ws =
      bytes_f64(checked_mul(6, v3) + epilogue_blocks + tiles + 1) + sizeof(int) +
      (use_fp32_w ? bytes_f32(checked_mul(ov, oo) + checked_mul(oo, vv) + v3) : 0);
  if (checked_add(
          tracker.live_bytes,
          checked_add(fixed_ws, checked_mul(panel_cap, bytes_f64(v3) +
                                                           (use_fp32_w ? bytes_f32(v3) : 0)))) >
      tracker.max_bytes) {
    panel_cap = 1;
  }

  DeviceBuf<double> d_panels(tracker, checked_mul(panel_cap, v3));
  DeviceBuf<double> d_moments(tracker, checked_mul(6, v3));
  DeviceBuf<double> d_partials(tracker, epilogue_blocks);
  DeviceBuf<double> d_energies(tracker, tiles);
  DeviceBuf<double> d_energy(tracker, 1);
  DeviceBuf<int> d_error(tracker, 1);
  d_error.zero(blas.stream);

  DeviceBuf<float> d_ovoo_f32, d_t2_f32, d_panels_f32, d_scratch_f32;
  if (use_fp32_w) {
    d_ovoo_f32 = DeviceBuf<float>(tracker, checked_mul(ov, oo));
    d_t2_f32 = DeviceBuf<float>(tracker, checked_mul(oo, vv));
    d_panels_f32 = DeviceBuf<float>(tracker, checked_mul(panel_cap, v3));
    d_scratch_f32 = DeviceBuf<float>(tracker, v3);
    cast_f64_to_f32_kernel<<<grid_blocks(d_ovoo.size()), 256, 0, blas.stream>>>(
        d_ovoo.size(), d_ovoo.data(), d_ovoo_f32.data(), d_error.data());
    cast_f64_to_f32_kernel<<<grid_blocks(d_t2.size()), 256, 0, blas.stream>>>(
        d_t2.size(), d_t2.data(), d_t2_f32.data(), d_error.data());
    cuda_check(cudaGetLastError());
  }

  triples::generated_df::Inputs gen_in{
      .bov = d_bov.data(),
      .bvv = d_bvv.data(),
      .ovoo = d_ovoo.data(),
      .ovov = d_ovov.data(),
      .fov = d_fov.data(),
      .t1 = d_t1.data(),
      .t2 = d_t2.data(),
      .eps_o = d_eps_o.data(),
      .eps_v = d_eps_v.data(),
  };

  auto fp64_gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t k, double alpha,
                       const double* a, std::size_t lda, const double* b, std::size_t ldb,
                       double beta, double* c, std::size_t ldc) {
    gpu_gemm_cm_f64(blas.handle, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  };
  auto fp32_gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t k, float alpha,
                       const float* a, std::size_t lda, const float* b, std::size_t ldb, float beta,
                       float* c, std::size_t ldc) {
    gpu_gemm_cm_f32(blas.handle, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  };
  auto accum_f32 = [&](const float* src, std::size_t n, double alpha, double beta, double* dst) {
    accumulate_f32_to_f64_kernel<<<grid_blocks(n), 256, 0, blas.stream>>>(
        n, src, alpha, beta, dst, d_error.data());
    cuda_check(cudaGetLastError());
  };

  std::array<std::size_t, 3> identities{std::numeric_limits<std::size_t>::max(),
                                        std::numeric_limits<std::size_t>::max(),
                                        std::numeric_limits<std::size_t>::max()};
  std::array<std::size_t, 3> ages{0, 0, 0};
  std::size_t epoch = 0;
  std::size_t fp32_gemms = 0, fp64_gemms = 0;

  auto panel_slot_for = [&](std::size_t occupied) -> std::size_t {
    std::size_t slot = 0;
    for (; slot < panel_cap; ++slot)
      if (identities[slot] == occupied) break;
    if (slot == panel_cap) {
      slot = static_cast<std::size_t>(
          std::min_element(ages.begin(), ages.begin() + panel_cap) - ages.begin());
      double* panel_dst = d_panels.data() + slot * v3;
      triples::generated_df::build_panel(o, v, q, occupied, gen_in, panel_dst, fp64_gemm);
      ++fp64_gemms;
      if (qf > 0) {
        const double s38 = 0.375 * in.c2.geminal_scale;
        const double s18 = 0.125 * in.c2.geminal_scale;
        auto extract_row = [&](const DeviceBuf<double>& src) {
          extract_qov_row_kernel<<<grid_blocks(qf * v), 256, 0, blas.stream>>>(
              qf, o, v, occupied, src.data(), d_c2_row.data());
          cuda_check(cudaGetLastError());
        };
        extract_row(d_af_vo);
        gpu_gemm_rm(blas.handle, 'T', 'N', v, vv, qf, d_c2_row.data(), d_r_vv.data(),
                    d_c2_tmp.data(), 1.0, 0.0);
        extract_row(d_ar_vo);
        gpu_gemm_rm(blas.handle, 'T', 'N', v, vv, qf, d_c2_row.data(), d_f_vv.data(),
                    d_c2_tmp.data(), 1.0, 1.0);
        extract_row(d_f_vo);
        gpu_gemm_rm(blas.handle, 'T', 'N', v, vv, qf, d_c2_row.data(), d_ar_vv.data(),
                    d_c2_tmp.data(), 1.0, 1.0);
        extract_row(d_r_vo);
        gpu_gemm_rm(blas.handle, 'T', 'N', v, vv, qf, d_c2_row.data(), d_af_vv.data(),
                    d_c2_tmp.data(), 1.0, 1.0);
        add_c2_sym_v3_kernel<<<grid_blocks(v3), 256, 0, blas.stream>>>(v, s38, s18, d_c2_tmp.data(),
                                                                       panel_dst);
        cuda_check(cudaGetLastError());
        fp64_gemms += 4;
      }
      if (use_fp32_w) {
        cast_f64_to_f32_kernel<<<grid_blocks(v3), 256, 0, blas.stream>>>(
            v3, panel_dst, d_panels_f32.data() + slot * v3, d_error.data());
        cuda_check(cudaGetLastError());
      }
      identities[slot] = occupied;
    }
    ages[slot] = ++epoch;
    return slot;
  };

  const bool distinct_moments = !use_fp32_w;
  auto visit_tile = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t canonical_tile) {
    const std::array<std::size_t, 3> occupied{i, j, k};
    const bool alias_tile = distinct_moments && (i == j || j == k);
    const auto sources = alias_tile ? triples::generated_df::occupied_moment_sources(i, j, k)
                                    : triples::generated_df::MomentSourceMap{};
    for (std::size_t idx = 0; idx < 3; ++idx) {
      if (std::find(occupied.begin(), occupied.begin() + idx, occupied[idx]) !=
          occupied.begin() + idx)
        continue;
      const auto slot = panel_slot_for(occupied[idx]);
      for (std::size_t perm = 0; perm < 6; ++perm) {
        const auto* order = triples::generated_df::permutations[perm];
        if (occupied[order[0]] != occupied[idx]) continue;
        if (sources.index[perm] != perm) continue;
        double* moment_dst = d_moments.data() + perm * v3;
        if (use_fp32_w) {
          triples::generated_df::build_w_fp32(
              o, v, occupied[order[0]], occupied[order[1]], occupied[order[2]], d_ovoo_f32.data(),
              d_t2_f32.data(), d_panels_f32.data() + slot * v3, d_scratch_f32.data(), moment_dst,
              fp32_gemm, accum_f32);
          fp32_gemms += 2;
        } else {
          triples::generated_df::build_w(o, v, occupied[order[0]], occupied[order[1]],
                                         occupied[order[2]], gen_in, d_panels.data() + slot * v3,
                                         moment_dst, fp64_gemm);
          fp64_gemms += 2;
        }
      }
    }
    const double degeneracy = i == k ? 6.0 : (i == j || j == k ? 2.0 : 1.0);
    if (alias_tile)
      triples::generated_df::energy_distinct_tile(o, v, i, j, k, degeneracy,
                                                  in.denominator_threshold, gen_in,
                                                  d_moments.data(), sources, epilogue_blocks,
                                                  d_partials.data(), d_error.data(), blas.stream);
    else
      triples::generated_df::energy_tile(o, v, i, j, k, degeneracy, in.denominator_threshold,
                                         gen_in, d_moments.data(), epilogue_blocks,
                                         d_partials.data(), d_error.data(), blas.stream);
    reduce_sum_kernel<<<1, 256, 0, blas.stream>>>(d_partials.data(), epilogue_blocks,
                                                  d_energies.data() + canonical_tile,
                                                  d_error.data());
    cuda_check(cudaGetLastError());
  };

  triples::detail::visit_occupied_tiles(o, distinct_moments && panel_cap == 3, visit_tile);
  reduce_sum_kernel<<<1, 256, 0, blas.stream>>>(d_energies.data(), tiles, d_energy.data(),
                                                d_error.data());
  cuda_check(cudaGetLastError());

  double energy_val = 0.0;
  int failed = 0;
  d_energy.download(&energy_val, 1, blas.stream);
  d_error.download(&failed, 1, blas.stream);
  cuda_check(cudaStreamSynchronize(blas.stream));
  if (failed || !std::isfinite(energy_val))
    throw std::runtime_error("nonfinite or unsafe explicit (T)-F12 CUDA pt arithmetic");

  if (telemetry_out) {
    telemetry_out->seconds_pt = elapsed_seconds(started);
    telemetry_out->peak_workspace_bytes =
        std::max(telemetry_out->peak_workspace_bytes, tracker.peak_bytes + provider_allowance);
    telemetry_out->triples_fp32_gemms += fp32_gemms;
    telemetry_out->triples_fp64_gemms += fp64_gemms;
  }
  return energy_val;
}

Breakdown evaluate_explicit_tf12_a_cuda(const DfAInputs& in, std::size_t max_bytes, int device) {
  Breakdown out{};
  if (in.q_f12 == 0) return out;
  if (in.frozen >= in.o_full || in.o_full >= in.obs || in.obs >= in.ext || device < 0 || !max_bytes)
    throw std::invalid_argument("invalid explicit (T)-F12 CUDA A-sector dimensions");

  constexpr std::size_t provider_allowance = 128ULL << 20;
  if (max_bytes <= provider_allowance)
    throw std::length_error("explicit (T)-F12 CUDA A-sector budget below provider allowance");

  const auto started = Clock::now();
  const std::size_t o_full = in.o_full, fz = in.frozen, obs = in.obs, ext = in.ext, q = in.q_f12;
  const std::size_t o = o_full - fz, v = obs - o_full, cabs = ext - obs;
  const std::size_t oo = checked_mul(o, o), ov = checked_mul(o, v), vv = checked_mul(v, v);
  const std::size_t o_obs = checked_mul(o, obs), obs2 = checked_mul(obs, obs);
  const std::size_t ee = checked_mul(ext, ext), eo = checked_mul(ext, o), ev = checked_mul(ext, v);
  const std::size_t n2 = checked_mul(oo, vv);
  const double s = in.geminal_scale, s2 = s * s;

  std::vector<double> t2t(n2), w_swp(n2);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const std::size_t idx = ((i * o + j) * v + a) * v + b;
          const std::size_t swp = ((i * o + j) * v + b) * v + a;
          t2t[idx] = 2.0 * in.t2[idx] - in.t2[swp];
        }
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b)
          w_swp[((i * o + j) * v + a) * v + b] = t2t[((i * o + j) * v + b) * v + a];

  runtime::CudaDeviceScope device_scope(device, cuda_check);
  BlasStreamOwner blas;
  DeviceArenaTracker tracker{max_bytes - provider_allowance, 0, 0};

  // Host results from Stage 2A.
  std::vector<double> V_a(checked_mul(o_obs, ov), 0.0), V_b(checked_mul(o_obs, ov), 0.0);
  std::vector<double> X_a(checked_mul(o_obs, ov), 0.0), X_b(checked_mul(o_obs, ov), 0.0);
  std::vector<double> B_a(checked_mul(ov, ov), 0.0), B_b(checked_mul(ov, ov), 0.0);

  // ==========================================================================
  // STAGE 2A: GPU-RESIDENT V, X, B (direct + P12 + -(K+F) + QFQ)
  // ==========================================================================
  {
    const std::size_t obsv = checked_mul(obs, v);
    DeviceBuf<double> d_V_a(tracker, checked_mul(o_obs, ov));
    DeviceBuf<double> d_V_b(tracker, checked_mul(o_obs, ov));
    DeviceBuf<double> d_X_a(tracker, checked_mul(o_obs, ov));
    DeviceBuf<double> d_X_b(tracker, checked_mul(o_obs, ov));
    DeviceBuf<double> d_B_a(tracker, checked_mul(ov, ov));
    DeviceBuf<double> d_B_b(tracker, checked_mul(ov, ov));

    // Direct 1-part terms (fg, ff, U, ff*(F+K)) via GPU GEMMs (host-packed small slices -> GPU).
    {
      std::vector<double> F_oo(checked_mul(oo, q)), Rfg_oo(checked_mul(oo, q)),
          Rff_oo(checked_mul(oo, q)), Ru_oo(checked_mul(oo, q));
      std::vector<double> F_obsv(checked_mul(q, obsv)), Rfg_obsv(checked_mul(q, obsv)),
          Rff_obsv(checked_mul(q, obsv)), Ru_obsv(checked_mul(q, obsv));
      std::vector<double> F_ov(checked_mul(ov, q)), Rfg_ov(checked_mul(ov, q)),
          Rff_ov(checked_mul(ov, q)), Ru_ov(checked_mul(ov, q));
      std::vector<double> F_obso(checked_mul(q, o_obs)), Rfg_obso(checked_mul(q, o_obs)),
          Rff_obso(checked_mul(q, o_obs)), Ru_obso(checked_mul(q, o_obs));
      for (std::size_t Q = 0; Q < q; ++Q) {
        const double* fQ = in.fit_ext_obs + Q * ext * obs;
        const double* fgQ = in.r_fg_obs_obs + Q * obs2;
        const double* ffQ = in.r_ff_ext_obs + Q * ext * obs;
        const double* uQ = in.r_u_obs_obs + Q * obs2;
        for (std::size_t i = 0; i < o; ++i)
          for (std::size_t k = 0; k < o; ++k) {
            const std::size_t fi = fz + i, fk = fz + k;
            F_oo[(i * o + k) * q + Q] = fQ[fi * obs + fk];
            Rfg_oo[(i * o + k) * q + Q] = fgQ[fi * obs + fk];
            Rff_oo[(i * o + k) * q + Q] = ffQ[fi * obs + fk];
            Ru_oo[(i * o + k) * q + Q] = uQ[fi * obs + fk];
          }
        for (std::size_t p = 0; p < obs; ++p) {
          for (std::size_t b = 0; b < v; ++b) {
            const std::size_t vb = o_full + b;
            F_obsv[Q * obsv + p * v + b] = fQ[p * obs + vb];
            Rfg_obsv[Q * obsv + p * v + b] = fgQ[p * obs + vb];
            Rff_obsv[Q * obsv + p * v + b] = ffQ[p * obs + vb];
            Ru_obsv[Q * obsv + p * v + b] = uQ[p * obs + vb];
          }
          for (std::size_t k = 0; k < o; ++k) {
            const std::size_t fk = fz + k;
            F_obso[Q * o_obs + p * o + k] = fQ[p * obs + fk];
            Rfg_obso[Q * o_obs + p * o + k] = fgQ[p * obs + fk];
            Rff_obso[Q * o_obs + p * o + k] = ffQ[p * obs + fk];
            Ru_obso[Q * o_obs + p * o + k] = uQ[p * obs + fk];
          }
        }
        for (std::size_t i = 0; i < o; ++i)
          for (std::size_t b = 0; b < v; ++b) {
            const std::size_t fi = fz + i, vb = o_full + b;
            F_ov[(i * v + b) * q + Q] = fQ[fi * obs + vb];
            Rfg_ov[(i * v + b) * q + Q] = fgQ[fi * obs + vb];
            Rff_ov[(i * v + b) * q + Q] = ffQ[fi * obs + vb];
            Ru_ov[(i * v + b) * q + Q] = uQ[fi * obs + vb];
          }
      }

      DeviceBuf<double> d_L1(tracker, checked_mul(oo, q)), d_L2(tracker, checked_mul(oo, q));
      DeviceBuf<double> d_R1(tracker, checked_mul(q, obsv)), d_R2(tracker, checked_mul(q, obsv));
      DeviceBuf<double> d_tmp(tracker, checked_mul(oo, obsv));
      std::vector<double> h_tmp(checked_mul(oo, obsv));

      auto run_dir_a = [&](const std::vector<double>& R_oo, const std::vector<double>& R_obsv) {
        d_L1.upload(F_oo.data(), d_L1.size(), blas.stream);
        d_L2.upload(R_oo.data(), d_L2.size(), blas.stream);
        d_R1.upload(F_obsv.data(), d_R1.size(), blas.stream);
        d_R2.upload(R_obsv.data(), d_R2.size(), blas.stream);
        gpu_gemm_rm(blas.handle, 'N', 'N', oo, obsv, q, d_L1.data(), d_R2.data(), d_tmp.data(), 1.0,
                    0.0);
        gpu_gemm_rm(blas.handle, 'N', 'N', oo, obsv, q, d_L2.data(), d_R1.data(), d_tmp.data(), 1.0,
                    1.0);
        d_tmp.download(h_tmp.data(), h_tmp.size(), blas.stream);
        cuda_check(cudaStreamSynchronize(blas.stream));
      };
      run_dir_a(Rfg_oo, Rfg_obsv);
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t p = 0; p < obs; ++p)
          for (std::size_t k = 0; k < o; ++k)
            for (std::size_t b = 0; b < v; ++b)
              V_a[(i * obs + p) * ov + (k * v + b)] = h_tmp[(i * o + k) * obsv + (p * v + b)];

      run_dir_a(Rff_oo, Rff_obsv);
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t p = 0; p < obs; ++p)
          for (std::size_t k = 0; k < o; ++k)
            for (std::size_t b = 0; b < v; ++b)
              X_a[(i * obs + p) * ov + (k * v + b)] = h_tmp[(i * o + k) * obsv + (p * v + b)];

      run_dir_a(Ru_oo, Ru_obsv);
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t k = 0; k < o; ++k)
            for (std::size_t b = 0; b < v; ++b)
              B_a[(i * v + a) * ov + (k * v + b)] =
                  h_tmp[(i * o + k) * obsv + ((o_full + a) * v + b)];

      d_L1.reset();
      d_L2.reset();
      d_R1.reset();
      d_R2.reset();
      d_tmp.reset();

      DeviceBuf<double> d_Lb1(tracker, checked_mul(ov, q)), d_Lb2(tracker, checked_mul(ov, q));
      DeviceBuf<double> d_Rb1(tracker, checked_mul(q, o_obs)), d_Rb2(tracker, checked_mul(q, o_obs));
      DeviceBuf<double> d_tmp2(tracker, checked_mul(ov, o_obs));
      std::vector<double> h_tmp2(checked_mul(ov, o_obs));
      auto run_dir_b = [&](const std::vector<double>& R_ov, const std::vector<double>& R_obso) {
        d_Lb1.upload(F_ov.data(), d_Lb1.size(), blas.stream);
        d_Lb2.upload(R_ov.data(), d_Lb2.size(), blas.stream);
        d_Rb1.upload(F_obso.data(), d_Rb1.size(), blas.stream);
        d_Rb2.upload(R_obso.data(), d_Rb2.size(), blas.stream);
        gpu_gemm_rm(blas.handle, 'N', 'N', ov, o_obs, q, d_Lb1.data(), d_Rb2.data(), d_tmp2.data(),
                    1.0, 0.0);
        gpu_gemm_rm(blas.handle, 'N', 'N', ov, o_obs, q, d_Lb2.data(), d_Rb1.data(), d_tmp2.data(),
                    1.0, 1.0);
        d_tmp2.download(h_tmp2.data(), h_tmp2.size(), blas.stream);
        cuda_check(cudaStreamSynchronize(blas.stream));
      };
      run_dir_b(Rfg_ov, Rfg_obso);
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t p = 0; p < obs; ++p)
          for (std::size_t k = 0; k < o; ++k)
            for (std::size_t b = 0; b < v; ++b)
              V_b[(i * obs + p) * ov + (k * v + b)] = h_tmp2[(i * v + b) * o_obs + (p * o + k)];

      run_dir_b(Rff_ov, Rff_obso);
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t p = 0; p < obs; ++p)
          for (std::size_t k = 0; k < o; ++k)
            for (std::size_t b = 0; b < v; ++b)
              X_b[(i * obs + p) * ov + (k * v + b)] = h_tmp2[(i * v + b) * o_obs + (p * o + k)];

      run_dir_b(Ru_ov, Ru_obso);
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t k = 0; k < o; ++k)
            for (std::size_t b = 0; b < v; ++b)
              B_b[(i * v + a) * ov + (k * v + b)] =
                  h_tmp2[(i * v + b) * o_obs + ((o_full + a) * o + k)];

      d_Lb1.reset();
      d_Lb2.reset();
      d_Rb1.reset();
      d_Rb2.reset();
      d_tmp2.reset();

      // Symmetrized ff*(F+K) on GPU: FK_F[Q, obs, obs] and FK_Rff[Q, obs, obs]
      std::vector<double> FpK_oe(checked_mul(obs, ext));
      for (std::size_t p = 0; p < obs; ++p)
        for (std::size_t t = 0; t < ext; ++t)
          FpK_oe[p * ext + t] = in.f_ext[p * ext + t] + in.k_ext[p * ext + t];
      DeviceBuf<double> d_FpK(tracker, FpK_oe.size());
      DeviceBuf<double> d_in_q(tracker, checked_mul(ext, obs));
      DeviceBuf<double> d_out_q(tracker, obs2);
      d_FpK.upload(FpK_oe.data(), FpK_oe.size(), blas.stream);
      std::vector<double> FK_F(checked_mul(q, obs2)), FK_Rff(checked_mul(q, obs2));
      for (std::size_t Q = 0; Q < q; ++Q) {
        d_in_q.upload(in.fit_ext_obs + Q * ext * obs, ext * obs, blas.stream);
        gpu_gemm_rm(blas.handle, 'N', 'N', obs, obs, ext, d_FpK.data(), d_in_q.data(),
                    d_out_q.data(), 1.0, 0.0);
        d_out_q.download(FK_F.data() + Q * obs2, obs2, blas.stream);

        d_in_q.upload(in.r_ff_ext_obs + Q * ext * obs, ext * obs, blas.stream);
        gpu_gemm_rm(blas.handle, 'N', 'N', obs, obs, ext, d_FpK.data(), d_in_q.data(),
                    d_out_q.data(), 1.0, 0.0);
        d_out_q.download(FK_Rff.data() + Q * obs2, obs2, blas.stream);
      }
      cuda_check(cudaStreamSynchronize(blas.stream));
      d_FpK.reset();
      d_in_q.reset();
      d_out_q.reset();

      std::vector<double> F_vv(checked_mul(q, vv)), Rff_vv(checked_mul(q, vv)),
          FKF_vv(checked_mul(q, vv)), FKR_vv(checked_mul(q, vv));
      std::vector<double> FKF_oo(checked_mul(oo, q)), FKR_oo(checked_mul(oo, q));
      std::vector<double> FKF_ov(checked_mul(ov, q)), FKR_ov(checked_mul(ov, q));
      std::vector<double> FKF_vo(checked_mul(q, ov)), FKR_vo(checked_mul(q, ov));
      std::vector<double> F_vo_q(checked_mul(q, ov)), Rff_vo_q(checked_mul(q, ov));
      for (std::size_t Q = 0; Q < q; ++Q) {
        const double* fQ = in.fit_ext_obs + Q * ext * obs;
        const double* rQ = in.r_ff_ext_obs + Q * ext * obs;
        const double* fkfQ = FK_F.data() + Q * obs2;
        const double* fkrQ = FK_Rff.data() + Q * obs2;
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
      DeviceBuf<double> d_a1(tracker, checked_mul(oo, q)), d_a2(tracker, checked_mul(q, vv)),
          d_fba(tracker, n2);
      std::vector<double> h_fba(n2);
      auto add_fba = [&](const std::vector<double>& L, const std::vector<double>& R, double beta) {
        d_a1.upload(L.data(), L.size(), blas.stream);
        d_a2.upload(R.data(), R.size(), blas.stream);
        gpu_gemm_rm(blas.handle, 'N', 'N', oo, vv, q, d_a1.data(), d_a2.data(), d_fba.data(), 1.0,
                    beta);
      };
      add_fba(FKF_oo, Rff_vv, 0.0);
      add_fba(FKR_oo, F_vv, 1.0);
      add_fba(F_oo, FKR_vv, 1.0);
      add_fba(Rff_oo, FKF_vv, 1.0);
      d_fba.download(h_fba.data(), n2, blas.stream);
      cuda_check(cudaStreamSynchronize(blas.stream));
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t k = 0; k < o; ++k)
            for (std::size_t b = 0; b < v; ++b) {
              const double v1 = h_fba[(i * o + k) * vv + (a * v + b)];
              const double v2 = h_fba[(k * o + i) * vv + (b * v + a)];
              B_a[(i * v + a) * ov + (k * v + b)] += 0.5 * (v1 + v2);
            }
      d_a1.reset();
      d_a2.reset();
      d_fba.reset();

      DeviceBuf<double> d_b1(tracker, checked_mul(ov, q)), d_b2(tracker, checked_mul(q, ov)),
          d_fbb(tracker, n2);
      std::vector<double> h_fbb(n2);
      auto add_fbb = [&](const std::vector<double>& L, const std::vector<double>& R, double beta) {
        d_b1.upload(L.data(), L.size(), blas.stream);
        d_b2.upload(R.data(), R.size(), blas.stream);
        gpu_gemm_rm(blas.handle, 'N', 'N', ov, ov, q, d_b1.data(), d_b2.data(), d_fbb.data(), 1.0,
                    beta);
      };
      add_fbb(FKF_ov, Rff_vo_q, 0.0);
      add_fbb(FKR_ov, F_vo_q, 1.0);
      add_fbb(F_ov, FKR_vo, 1.0);
      add_fbb(Rff_ov, FKF_vo, 1.0);
      d_fbb.download(h_fbb.data(), n2, blas.stream);
      cuda_check(cudaStreamSynchronize(blas.stream));
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t k = 0; k < o; ++k)
            for (std::size_t b = 0; b < v; ++b) {
              const double v1 = h_fbb[(i * v + b) * ov + (a * o + k)];
              const double v2 = h_fbb[(k * v + a) * ov + (b * o + i)];
              B_b[(i * v + a) * ov + (k * v + b)] += 0.5 * (v1 + v2);
            }
    }

    // Upload initial direct V_a, V_b, X_a, X_b, B_a, B_b to GPU and run s_idx loop fully on GPU!
    d_V_a.upload(V_a.data(), V_a.size(), blas.stream);
    d_V_b.upload(V_b.data(), V_b.size(), blas.stream);
    d_X_a.upload(X_a.data(), X_a.size(), blas.stream);
    d_X_b.upload(X_b.data(), X_b.size(), blas.stream);
    d_B_a.upload(B_a.data(), B_a.size(), blas.stream);
    d_B_b.upload(B_b.data(), B_b.size(), blas.stream);

    DeviceBuf<double> d_fit_eo(tracker, checked_mul(q, checked_mul(ext, obs)));
    DeviceBuf<double> d_rf_eo(tracker, checked_mul(q, checked_mul(ext, obs)));
    DeviceBuf<double> d_bg_eo(tracker, checked_mul(q, checked_mul(ext, obs)));
    d_fit_eo.upload(in.fit_ext_obs, d_fit_eo.size(), blas.stream);
    d_rf_eo.upload(in.r_f_ext_obs, d_rf_eo.size(), blas.stream);
    d_bg_eo.upload(in.b_g_ext_obs, d_bg_eo.size(), blas.stream);

    // Pack F_eo[q, ext*o], Rf_eo[q, ext*o], Bg_eo[q, ext*o], F_ev[q, ext*v], Rf_ev[q, ext*v].
    std::vector<double> F_eo(checked_mul(q, eo)), Rf_eo(checked_mul(q, eo)),
        Bg_eo(checked_mul(q, eo));
    std::vector<double> F_ev(checked_mul(q, ev)), Rf_ev(checked_mul(q, ev));
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
    DeviceBuf<double> d_F_eo(tracker, F_eo.size()), d_Rf_eo(tracker, Rf_eo.size()),
        d_Bg_eo(tracker, Bg_eo.size());
    DeviceBuf<double> d_F_ev(tracker, F_ev.size()), d_Rf_ev(tracker, Rf_ev.size());
    d_F_eo.upload(F_eo.data(), F_eo.size(), blas.stream);
    d_Rf_eo.upload(Rf_eo.data(), Rf_eo.size(), blas.stream);
    d_Bg_eo.upload(Bg_eo.data(), Bg_eo.size(), blas.stream);
    d_F_ev.upload(F_ev.data(), F_ev.size(), blas.stream);
    d_Rf_ev.upload(Rf_ev.data(), Rf_ev.size(), blas.stream);

    std::vector<double> neg_KF(ee);
    for (std::size_t i = 0; i < ee; ++i) neg_KF[i] = -(in.k_ext[i] + in.f_ext[i]);
    DeviceBuf<double> d_neg_KF(tracker, ee), d_f_ext(tracker, ee);
    d_neg_KF.upload(neg_KF.data(), ee, blas.stream);
    d_f_ext.upload(in.f_ext, ee, blas.stream);

    DeviceBuf<double> d_f_sp(tracker, checked_mul(q, obs)), d_rf_sp(tracker, checked_mul(q, obs)),
        d_bg_sp(tracker, checked_mul(q, obs));
    DeviceBuf<double> d_f_so(tracker, checked_mul(q, o)), d_rf_so(tracker, checked_mul(q, o));
    DeviceBuf<double> d_g_mat(tracker, checked_mul(obs, eo)), d_f_mat(tracker, checked_mul(obs, eo)),
        d_f_sw_mat(tracker, checked_mul(o, ev));
    DeviceBuf<double> d_F_ipr(tracker, checked_mul(o_obs, ext));
    DeviceBuf<double> d_F_kbr(tracker, checked_mul(ov, ext)),
        d_F_sw_kbr(tracker, checked_mul(ov, ext));
    DeviceBuf<double> d_G_sub(tracker, checked_mul(o_obs, ext)),
        d_F_sub(tracker, checked_mul(o_obs, ext));
    DeviceBuf<double> d_Fk_sub(tracker, checked_mul(ov, ext)),
        d_Fsw_sub(tracker, checked_mul(ov, ext));
    DeviceBuf<double> d_KF_L(tracker, checked_mul(ov, ext)), d_KF_R(tracker, checked_mul(ov, ext));
    DeviceBuf<double> d_F_qq(tracker, ee);
    DeviceBuf<double> d_qL(tracker, checked_mul(ov, ext)), d_qR(tracker, checked_mul(ov, ext));
    DeviceBuf<double> d_fqL(tracker, checked_mul(ov, ext)), d_fqR(tracker, checked_mul(ov, ext));

    for (std::size_t s_idx = 0; s_idx < ext; ++s_idx) {
      extract_s_slice_kernel<<<grid_blocks(q * obs), 256, 0, blas.stream>>>(
          q, ext, obs, fz, o, s_idx, d_fit_eo.data(), d_rf_eo.data(), d_bg_eo.data(),
          d_f_sp.data(), d_rf_sp.data(), d_bg_sp.data(), d_f_so.data(), d_rf_so.data());
      cuda_check(cudaGetLastError());

      gpu_gemm_rm(blas.handle, 'T', 'N', obs, eo, q, d_f_sp.data(), d_Rf_eo.data(), d_f_mat.data(),
                  1.0, 0.0);
      gpu_gemm_rm(blas.handle, 'T', 'N', obs, eo, q, d_rf_sp.data(), d_F_eo.data(), d_f_mat.data(),
                  1.0, 1.0);
      gpu_gemm_rm(blas.handle, 'T', 'N', o, ev, q, d_f_so.data(), d_Rf_ev.data(),
                  d_f_sw_mat.data(), 1.0, 0.0);
      gpu_gemm_rm(blas.handle, 'T', 'N', o, ev, q, d_rf_so.data(), d_F_ev.data(),
                  d_f_sw_mat.data(), 1.0, 1.0);

      reorder_s_mats_kernel<<<grid_blocks(o_obs * ext), 256, 0, blas.stream>>>(
          o, v, o_full, obs, ext, d_f_mat.data(), d_f_sw_mat.data(), d_F_ipr.data(),
          d_F_kbr.data(), d_F_sw_kbr.data());
      cuda_check(cudaGetLastError());

      const std::size_t r_p12 = (s_idx < o_full) ? ext : ((s_idx < obs) ? obs : o_full);
      if (r_p12 > 0) {
        gpu_gemm_rm(blas.handle, 'T', 'N', obs, eo, q, d_bg_sp.data(), d_Bg_eo.data(),
                    d_g_mat.data(), 1.0, 0.0);
        pack_p12_sub_kernel<<<grid_blocks(o_obs * r_p12), 256, 0, blas.stream>>>(
            o, v, o_full, obs, ext, r_p12, d_g_mat.data(), d_F_ipr.data(), d_F_kbr.data(),
            d_F_sw_kbr.data(), d_G_sub.data(), d_F_sub.data(), d_Fk_sub.data(), d_Fsw_sub.data());
        cuda_check(cudaGetLastError());
        gpu_gemm_rm(blas.handle, 'N', 'T', o_obs, ov, r_p12, d_G_sub.data(), d_Fk_sub.data(),
                    d_V_a.data(), -1.0, 1.0);
        gpu_gemm_rm(blas.handle, 'N', 'T', o_obs, ov, r_p12, d_G_sub.data(), d_Fsw_sub.data(),
                    d_V_b.data(), -1.0, 1.0);
        gpu_gemm_rm(blas.handle, 'N', 'T', o_obs, ov, r_p12, d_F_sub.data(), d_Fk_sub.data(),
                    d_X_a.data(), -1.0, 1.0);
        gpu_gemm_rm(blas.handle, 'N', 'T', o_obs, ov, r_p12, d_F_sub.data(), d_Fsw_sub.data(),
                    d_X_b.data(), -1.0, 1.0);
      }

      gpu_gemm_rm(blas.handle, 'N', 'N', ov, ext, ext, d_F_kbr.data(), d_neg_KF.data(),
                  d_KF_L.data(), 1.0, 0.0);
      gpu_gemm_rm(blas.handle, 'N', 'N', ov, ext, ext, d_F_sw_kbr.data(), d_neg_KF.data(),
                  d_KF_R.data(), 1.0, 0.0);
      gpu_gemm_rm(blas.handle, 'N', 'T', ov, ov, ext, d_KF_L.data(), d_F_kbr.data(), d_B_a.data(),
                  1.0, 1.0);
      gpu_gemm_rm(blas.handle, 'N', 'T', ov, ov, ext, d_KF_R.data(), d_F_sw_kbr.data(),
                  d_B_a.data(), 1.0, 1.0);
      gpu_gemm_rm(blas.handle, 'N', 'T', ov, ov, ext, d_KF_L.data(), d_F_sw_kbr.data(),
                  d_B_b.data(), 1.0, 1.0);
      gpu_gemm_rm(blas.handle, 'N', 'T', ov, ov, ext, d_KF_R.data(), d_F_kbr.data(), d_B_b.data(),
                  1.0, 1.0);

      if (s_idx >= o_full) {
        const std::size_t r_start = (s_idx < obs) ? obs : o_full;
        const std::size_t r_len = ext - r_start;
        slice_2d_sub_kernel<<<grid_blocks(r_len * r_len), 256, 0, blas.stream>>>(
            ext, r_start, r_len, r_start, r_len, d_f_ext.data(), d_F_qq.data());
        slice_q_cols_kernel<<<grid_blocks(ov * r_len), 256, 0, blas.stream>>>(
            ov, ext, r_start, r_len, d_F_kbr.data(), d_F_sw_kbr.data(), d_qL.data(), d_qR.data());
        cuda_check(cudaGetLastError());
        gpu_gemm_rm(blas.handle, 'N', 'N', ov, r_len, r_len, d_qL.data(), d_F_qq.data(),
                    d_fqL.data(), 1.0, 0.0);
        gpu_gemm_rm(blas.handle, 'N', 'N', ov, r_len, r_len, d_qR.data(), d_F_qq.data(),
                    d_fqR.data(), 1.0, 0.0);
        gpu_gemm_rm(blas.handle, 'N', 'T', ov, ov, r_len, d_fqL.data(), d_qL.data(), d_B_a.data(),
                    1.0, 1.0);
        gpu_gemm_rm(blas.handle, 'N', 'T', ov, ov, r_len, d_fqR.data(), d_qR.data(), d_B_a.data(),
                    1.0, 1.0);
        gpu_gemm_rm(blas.handle, 'N', 'T', ov, ov, r_len, d_fqL.data(), d_qR.data(), d_B_b.data(),
                    1.0, 1.0);
        gpu_gemm_rm(blas.handle, 'N', 'T', ov, ov, r_len, d_fqR.data(), d_qL.data(), d_B_b.data(),
                    1.0, 1.0);
      }
    }

    d_V_a.download(V_a.data(), V_a.size(), blas.stream);
    d_V_b.download(V_b.data(), V_b.size(), blas.stream);
    d_X_a.download(X_a.data(), X_a.size(), blas.stream);
    d_X_b.download(X_b.data(), X_b.size(), blas.stream);
    d_B_a.download(B_a.data(), B_a.size(), blas.stream);
    d_B_b.download(B_b.data(), B_b.size(), blas.stream);
    cuda_check(cudaStreamSynchronize(blas.stream));
  }  // All Stage 2A device buffers freed cleanly before Stage 2B!

  std::vector<double> V7(ov, 0.0), V8(vv, 0.0), V9(checked_mul(oo, ov), 0.0), V10_d(n2, 0.0),
      V10_e(n2, 0.0);
  std::vector<double> X2(ov, 0.0), X4(vv, 0.0), X6(checked_mul(oo, ov), 0.0), X7_d(n2, 0.0),
      X7_e(n2, 0.0);
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

  // ==========================================================================
  // STAGE 2B: GPU-RESIDENT Z2 (Direct OR exact FfgCompleted via P12/Q12 split!)
  // ==========================================================================
  const auto z2_start = Clock::now();
  double t5_z2 = 0.0;
  if (in.z2_mode != Z2Mode::Zero) {
    if (!in.b_g_ext_ext)
      throw std::invalid_argument("explicit (T)-F12 CUDA Z2 requires b_g_ext_ext");

    // Pre-pack compact host factors for Z2:
    //   F_eo_k[o, ext, q], Rf_eo_k[o, ext, q]
    //   F_eo_t[ext*o, q], Rf_eo_t[ext*o, q], F_eo_q[q, ext*o], Rf_eo_q[q, ext*o]
    //   F_ev_q[q, ext*v], Rf_ev_q[q, ext*v]
    //   B_oe[q, o, ext], B_ev[q, ext, v], B_ov[q, o, v]
    std::vector<double> F_eo_k(checked_mul(o, checked_mul(ext, q))),
        Rf_eo_k(checked_mul(o, checked_mul(ext, q)));
    std::vector<double> F_eo_t(checked_mul(eo, q)), Rf_eo_t(checked_mul(eo, q));
    std::vector<double> F_eo_q(checked_mul(q, eo)), Rf_eo_q(checked_mul(q, eo));
    std::vector<double> F_ev_q(checked_mul(q, ev)), Rf_ev_q(checked_mul(q, ev));
    std::vector<double> B_oe(checked_mul(q, eo)), B_ev(checked_mul(q, ev)),
        B_ov(checked_mul(q, ov));
    for (std::size_t Q = 0; Q < q; ++Q) {
      const double* fQ = in.fit_ext_obs + Q * ext * obs;
      const double* rQ = in.r_f_ext_obs + Q * ext * obs;
      const double* bQ = in.b_g_ext_obs + Q * ext * obs;
      for (std::size_t al = 0; al < ext; ++al) {
        for (std::size_t i = 0; i < o; ++i) {
          const double fv = fQ[al * obs + (fz + i)];
          const double rv = rQ[al * obs + (fz + i)];
          F_eo_k[(i * ext + al) * q + Q] = fv;
          Rf_eo_k[(i * ext + al) * q + Q] = rv;
          F_eo_t[(al * o + i) * q + Q] = fv;
          Rf_eo_t[(al * o + i) * q + Q] = rv;
          F_eo_q[Q * eo + (al * o + i)] = fv;
          Rf_eo_q[Q * eo + (al * o + i)] = rv;
          B_oe[(Q * o + i) * ext + al] = bQ[al * obs + (fz + i)];
        }
        for (std::size_t a = 0; a < v; ++a) {
          F_ev_q[Q * ev + (al * v + a)] = fQ[al * obs + (o_full + a)];
          Rf_ev_q[Q * ev + (al * v + a)] = rQ[al * obs + (o_full + a)];
          B_ev[(Q * ext + al) * v + a] = bQ[al * obs + (o_full + a)];
        }
      }
      for (std::size_t k = 0; k < o; ++k)
        for (std::size_t b = 0; b < v; ++b)
          B_ov[(Q * o + k) * v + b] = bQ[(fz + k) * obs + (o_full + b)];
    }

    // Upload shared Z2 GPU tensors (keeping Exchange and Coulomb buffer lifetimes strictly disjoint!).
    {
      DeviceBuf<double> d_F_eo_k(tracker, F_eo_k.size()), d_Rf_eo_k(tracker, Rf_eo_k.size());
      DeviceBuf<double> d_F_ev_q(tracker, F_ev_q.size()), d_Rf_ev_q(tracker, Rf_ev_q.size());
      DeviceBuf<double> d_B_ov(tracker, B_ov.size());
      DeviceBuf<double> d_w_dir(tracker, n2), d_w_swp(tracker, n2);
      DeviceBuf<double> d_D_co(tracker, checked_mul(q, ov));
      DeviceBuf<double> d_partials(tracker, 1024);
      std::vector<double> h_partials;

      d_F_eo_k.upload(F_eo_k.data(), F_eo_k.size(), blas.stream);
      d_Rf_eo_k.upload(Rf_eo_k.data(), Rf_eo_k.size(), blas.stream);
      d_F_ev_q.upload(F_ev_q.data(), F_ev_q.size(), blas.stream);
      d_Rf_ev_q.upload(Rf_ev_q.data(), Rf_ev_q.size(), blas.stream);
      d_B_ov.upload(B_ov.data(), B_ov.size(), blas.stream);
      d_w_dir.upload(t2t.data(), n2, blas.stream);
      d_w_swp.upload(w_swp.data(), n2, blas.stream);

      // D_co[Q, i, a] = sum_{j, b} B_ov[Q, j, b] * w_dir[i, j, a, b]
      {
        std::vector<double> w_ia_jb(n2);
        for (std::size_t i = 0; i < o; ++i)
          for (std::size_t j = 0; j < o; ++j)
            for (std::size_t a = 0; a < v; ++a)
              for (std::size_t b = 0; b < v; ++b)
                w_ia_jb[(i * v + a) * ov + (j * v + b)] = t2t[((i * o + j) * v + a) * v + b];
        DeviceBuf<double> d_w_ia_jb(tracker, n2);
        d_w_ia_jb.upload(w_ia_jb.data(), n2, blas.stream);
        gpu_gemm_rm(blas.handle, 'N', 'T', q, ov, ov, d_B_ov.data(), d_w_ia_jb.data(),
                    d_D_co.data(), 1.0, 0.0);
        cuda_check(cudaStreamSynchronize(blas.stream));
      }

      // Single reusable [ext, ext, o, o] GPU buffer for rQ_hh (and later overwritten with rP_hh)!
      const std::size_t ee_oo = checked_mul(ee, oo);
      DeviceBuf<double> d_r_hh(tracker, ee_oo);
      build_r_hh_gpu(blas.handle, blas.stream, tracker, o_full, fz, obs, ext, q, s, /*Q12*/ 1,
                     F_eo_t, Rf_eo_t, F_eo_q, Rf_eo_q, d_r_hh.data());

      // ---- PHASE A (Exchange): M1_ex + M2_ex (both identical for Direct and FfgCompleted!) ----
      // B_oe, B_ev, B_baq are allocated ONLY during Exchange and freed BEFORE b_g_ee!
      {
        DeviceBuf<double> d_B_oe(tracker, B_oe.size());
        DeviceBuf<double> d_B_ev(tracker, B_ev.size());
        d_B_oe.upload(B_oe.data(), B_oe.size(), blas.stream);
        d_B_ev.upload(B_ev.data(), B_ev.size(), blas.stream);

        t5_z2 += evaluate_z2_m1_ex_gpu(blas.handle, blas.stream, tracker, o_full, fz, obs, ext, q,
                                       s, d_r_hh.data(), d_F_eo_k.data(), d_Rf_eo_k.data(),
                                       d_F_ev_q.data(), d_Rf_ev_q.data(), d_B_oe.data(),
                                       d_B_ev.data(), d_w_dir.data(), d_partials.data(), h_partials);

        DeviceBuf<double> d_B_baq(tracker, checked_mul(ev, q));
        reorder_bev_to_baq_kernel<<<grid_blocks(q * ev), 256, 0, blas.stream>>>(
            q, ext, v, d_B_ev.data(), d_B_baq.data());
        cuda_check(cudaStreamSynchronize(blas.stream));
        d_B_ev.reset();

        const BgEeSource no_bg{};
        const std::vector<M2GpuTerm> m2_ex_terms{
            {/*rQ_hh_s*/ 1, /*rQ_hp*/ 0, 0.0, -4.0, false},
            {/*rQ_hh_a*/ 2, /*rQ_hp_s*/ 1, 0.0, +4.0, true},
            {/*rQ_hh*/ 0, /*rQ_hp_a*/ 2, 0.0, -4.0, true},
        };
        t5_z2 += evaluate_z2_m2_gpu(blas.handle, blas.stream, tracker, o_full, fz, obs, ext, q, s,
                                    /*Q12*/ 1, d_r_hh.data(), m2_ex_terms, d_F_eo_k.data(),
                                    d_Rf_eo_k.data(), d_F_ev_q.data(), d_Rf_ev_q.data(), no_bg,
                                    d_D_co.data(), d_B_oe.data(), d_B_baq.data(), d_w_dir.data(),
                                    d_w_swp.data(), d_partials.data(), h_partials);
        cuda_check(cudaStreamSynchronize(blas.stream));
      }  // d_B_oe, d_B_ev, d_B_baq freed before Coulomb!

      // ---- PHASE B (Coulomb): resident d_b_g_ee if fits easily, else q-tiled <= 2 GiB! ----
      DeviceBuf<double> d_b_g_ee_full;
      BgEeSource bg_src{};
      const std::size_t bg_bytes = bytes_f64(checked_mul(q, ee));
      const std::size_t avail_for_bg = tracker.max_bytes > tracker.live_bytes
                                           ? (tracker.max_bytes - tracker.live_bytes)
                                           : 0;
      if (bg_bytes <= avail_for_bg / 2 && avail_for_bg > (bg_bytes + (3000ULL << 20))) {
        d_b_g_ee_full = DeviceBuf<double>(tracker, checked_mul(q, ee));
        d_b_g_ee_full.upload(in.b_g_ext_ext, d_b_g_ee_full.size(), blas.stream);
        bg_src = {d_b_g_ee_full.data(), nullptr, q};
      } else {
        const std::size_t q_cap =
            std::min<std::size_t>(2048ULL << 20, avail_for_bg / 3);
        const std::size_t q_tile =
            std::max<std::size_t>(1, q_cap / std::max<std::size_t>(1, bytes_f64(ee)));
        bg_src = {nullptr, in.b_g_ext_ext, std::min(q, q_tile)};
      }

      if (in.z2_mode == Z2Mode::Direct) {
        const std::vector<M2GpuTerm> m2_co_terms{
            {/*rQ_hh_a*/ 2, /*rQ_hp_a*/ 2, -4.0, 0.0, false},
            {/*rQ_hh*/ 0, /*rQ_hp_s*/ 1, +4.0, 0.0, false},
            {/*rQ_hh_s*/ 1, /*rQ_hp*/ 0, +4.0, 0.0, false},
        };
        t5_z2 += evaluate_z2_m2_gpu(blas.handle, blas.stream, tracker, o_full, fz, obs, ext, q, s,
                                    /*Q12*/ 1, d_r_hh.data(), m2_co_terms, d_F_eo_k.data(),
                                    d_Rf_eo_k.data(), d_F_ev_q.data(), d_Rf_ev_q.data(), bg_src,
                                    d_D_co.data(), nullptr, nullptr, d_w_dir.data(), d_w_swp.data(),
                                    d_partials.data(), h_partials);

        const std::vector<SubBlock3> q_q_blocks{
            {obs, cabs, o_full, v, obs, cabs},
            {o_full, v + cabs, obs, cabs, o_full, v + cabs},
        };
        t5_z2 += evaluate_z2_m1_co_pair_gpu(
            blas.handle, blas.stream, tracker, o_full, fz, obs, ext, q, s, /*Q12*/ 1, q_q_blocks,
            -4.0, d_r_hh.data(), d_F_eo_k.data(), d_Rf_eo_k.data(), d_F_ev_q.data(),
            d_Rf_ev_q.data(), d_B_ov.data(), bg_src, d_w_dir.data(), d_partials.data(), h_partials);
      } else {
        // Exact FfgCompleted (GeCCo Z2_appr=none): Coulomb = -(Q,P) - (P,1) + ff*g_11 (zero X*X*X!).
        const std::vector<M2GpuTerm> m2_qp_terms{
            {/*rL_hh_a*/ 2, /*rR_hp_a*/ 2, +4.0, 0.0, false},
            {/*rL_hh*/ 0, /*rR_hp_s*/ 1, -4.0, 0.0, false},
            {/*rL_hh_s*/ 1, /*rR_hp*/ 0, -4.0, 0.0, false},
        };
        t5_z2 += evaluate_z2_m2_gpu(blas.handle, blas.stream, tracker, o_full, fz, obs, ext, q, s,
                                    /*P12*/ 2, d_r_hh.data(), m2_qp_terms, d_F_eo_k.data(),
                                    d_Rf_eo_k.data(), d_F_ev_q.data(), d_Rf_ev_q.data(), bg_src,
                                    d_D_co.data(), nullptr, nullptr, d_w_dir.data(), d_w_swp.data(),
                                    d_partials.data(), h_partials);

        const std::vector<SubBlock3> q_p_blocks{
            {obs, cabs, o_full, v, 0, obs},
            {o_full, v + cabs, obs, cabs, 0, o_full},
        };
        t5_z2 += evaluate_z2_m1_co_pair_gpu(
            blas.handle, blas.stream, tracker, o_full, fz, obs, ext, q, s, /*P12*/ 2, q_p_blocks,
            +4.0, d_r_hh.data(), d_F_eo_k.data(), d_Rf_eo_k.data(), d_F_ev_q.data(),
            d_Rf_ev_q.data(), d_B_ov.data(), bg_src, d_w_dir.data(), d_partials.data(), h_partials);

        // Overwrite d_r_hh in-place with P12 via al-tiling (zero extra [ext,ext,o,o] VRAM!).
        build_r_hh_gpu(blas.handle, blas.stream, tracker, o_full, fz, obs, ext, q, s, /*P12*/ 2,
                       F_eo_t, Rf_eo_t, F_eo_q, Rf_eo_q, d_r_hh.data());

        t5_z2 += evaluate_z2_m2_gpu(blas.handle, blas.stream, tracker, o_full, fz, obs, ext, q, s,
                                    /*full 1*/ 0, d_r_hh.data(), m2_qp_terms, d_F_eo_k.data(),
                                    d_Rf_eo_k.data(), d_F_ev_q.data(), d_Rf_ev_q.data(), bg_src,
                                    d_D_co.data(), nullptr, nullptr, d_w_dir.data(), d_w_swp.data(),
                                    d_partials.data(), h_partials);

        const std::vector<SubBlock3> p_1_blocks{
            {0, ext, 0, o_full, 0, ext},
            {0, obs, o_full, v, 0, ext},
            {0, o_full, obs, cabs, 0, ext},
        };
        t5_z2 += evaluate_z2_m1_co_pair_gpu(
            blas.handle, blas.stream, tracker, o_full, fz, obs, ext, q, s, /*full 1*/ 0, p_1_blocks,
            +4.0, d_r_hh.data(), d_F_eo_k.data(), d_Rf_eo_k.data(), d_F_ev_q.data(),
            d_Rf_ev_q.data(), d_B_ov.data(), bg_src, d_w_dir.data(), d_partials.data(), h_partials);
      }
      cuda_check(cudaStreamSynchronize(blas.stream));
    }  // All Stage 2B GPU buffers freed cleanly before Step 2f!

    if (in.z2_mode == Z2Mode::FfgCompleted) {
      // Zero-obs^4 DF evaluator for the exact OBS 1*1 ff*g completion (<1.8 GiB host RAM, 0 VRAM).
      t5_z2 += detail::z2_ffg_11_df_cpu(o_full, fz, obs, ext, q, in.fit_ext_obs, in.r_ff_ext_obs,
                                        in.b_g_ext_obs, s2, t2t);
    }
  }
  out.seconds_z2 = elapsed_seconds(z2_start);
  out.peak_workspace_bytes = tracker.peak_bytes + provider_allowance;

  detail::evaluate_a_contractions_from_blocks(
      o, v, in.fov, in.t1, in.t2, in.eps_o, in.eps_v, in.ovoo, V7.data(), V8.data(), V9.data(),
      V10_d.data(), V10_e.data(), X2.data(), X4.data(), X6.data(), X7_d.data(), X7_e.data(),
      B4.data(), B7_d.data(), B7_e.data(), Bh4.data(), t5_z2, out);
  out.total_energy = out.a_sum;
  return out;
}

Breakdown evaluate_explicit_tf12_df_cuda(const PtInputs& pt_in, const DfAInputs& a_in,
                                         std::size_t max_bytes, int device,
                                         std::size_t max_panel_buffers,
                                         runtime::PrecisionDirective admitted_w) {
  Breakdown out = evaluate_explicit_tf12_a_cuda(a_in, max_bytes, device);
  out.e_pt_with_c2 = evaluate_explicit_tf12_pt_cuda(pt_in, max_bytes, device, max_panel_buffers,
                                                    admitted_w, &out);
  if (pt_in.c2.q_f12 == 0 && pt_in.c2.c2_ab_direct == nullptr) {
    out.e_pt_conv = out.e_pt_with_c2;
  } else if (pt_in.o <= 10) {
    PtInputs conv = pt_in;
    conv.c2 = {};
    out.e_pt_conv = evaluate_explicit_tf12_pt_cuda(conv, max_bytes, device, max_panel_buffers,
                                                   admitted_w, nullptr);
  } else {
    // The optional second conventional evaluation is omitted for large parents.
    out.e_pt_conv = std::numeric_limits<double>::quiet_NaN();
  }
  out.total_energy = out.e_pt_with_c2 + out.a_sum;
  return out;
}

}  // namespace generativeqc::cc::explicit_tf12
