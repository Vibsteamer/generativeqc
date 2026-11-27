#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cc/df_ucc_v2_device_capacity.hpp"
#include "cc/df_ucc_v2_internal.hpp"
#include "posthf/capacity.hpp"
#include "runtime/cuda_resources.cuh"
#include "scf/cuda/df_plan_internal.hpp"

namespace generativeqc::cc::v2_detail::cuda_detail {

using posthf::checked_add;
using posthf::checked_mul;

inline void cuda_check(cudaError_t status, const char* context) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("DF-UCC v2 CUDA ") + context + ": " +
                             cudaGetErrorString(status));
  }
}

inline void cublas_check(cublasStatus_t status, const char* context) {
  if (status == CUBLAS_STATUS_ALLOC_FAILED) throw std::bad_alloc();
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("DF-UCC v2 cuBLAS ") + context + ": status " +
                             std::to_string(static_cast<int>(status)));
  }
}

class CublasMathModeScope {
 public:
  CublasMathModeScope(cublasHandle_t handle, cublasMath_t mode, bool enabled = true)
      : handle_(handle) {
    if (!enabled) return;
    cublas_check(cublasGetMathMode(handle_, &previous_), "get math mode");
    if (previous_ != mode) {
      cublas_check(cublasSetMathMode(handle_, mode), "set math mode");
      changed_ = true;
    }
  }
  CublasMathModeScope(const CublasMathModeScope&) = delete;
  CublasMathModeScope& operator=(const CublasMathModeScope&) = delete;
  ~CublasMathModeScope() {
    if (changed_) (void)cublasSetMathMode(handle_, previous_);
  }

 private:
  cublasHandle_t handle_{};
  cublasMath_t previous_{CUBLAS_DEFAULT_MATH};
  bool changed_{false};
};

inline unsigned grid_1d(std::size_t count, int threads = 256) noexcept {
  if (!count) return 0U;
  const std::size_t blocks = (count + static_cast<std::size_t>(threads) - 1U) /
                             static_cast<std::size_t>(threads);
  return static_cast<unsigned>(std::min<std::size_t>(blocks, 65535ULL));
}

inline std::size_t query_device_cap_bytes(int device, std::size_t requested_max_bytes) {
  runtime::CudaDeviceScope scope(device);
  std::size_t avail = 0, total = 0;
  cuda_check(cudaMemGetInfo(&avail, &total), "query device memory");
  const std::size_t available_device_bytes =
      std::min(avail, runtime::device_resource_available(device));
  return requested_max_bytes ? std::min(requested_max_bytes, available_device_bytes)
                             : available_device_bytes;
}

inline std::size_t query_free_device_bytes(int device) noexcept {
  int prev = 0;
  const bool restore = cudaGetDevice(&prev) == cudaSuccess;
  if (cudaSetDevice(device) != cudaSuccess) return 0;
  std::size_t avail = 0, total = 0;
  const auto status = cudaMemGetInfo(&avail, &total);
  if (restore) (void)cudaSetDevice(prev);
  return status == cudaSuccess ? avail : 0;
}

// Zero-safe RAII device buffer wrapper over OwnedCudaBuffer.
template <class T>
class DeviceBuf {
 public:
  DeviceBuf() = default;
  DeviceBuf(int device, std::size_t count, cudaStream_t stream) {
    allocate(device, count, stream);
  }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
  DeviceBuf(DeviceBuf&&) noexcept = default;
  DeviceBuf& operator=(DeviceBuf&&) noexcept = default;

  void allocate(int device, std::size_t count, cudaStream_t stream) {
    reset();
    if (count == 0) return;
    UccDeviceCharge charge(checked_mul(count, sizeof(T)));
    buf_.allocate(device, count, stream);
    charge_ = std::move(charge);
  }

  void allocate_zero(int device, std::size_t count, cudaStream_t stream) {
    allocate(device, count, stream);
    zero(stream);
  }

  void zero(cudaStream_t stream) {
    if (!buf_.size()) return;
    cuda_check(cudaMemsetAsync(buf_.get(), 0, checked_mul(buf_.size(), sizeof(T)), stream),
               "memset");
  }

  void copy_from_host(const T* src, std::size_t count, cudaStream_t stream) {
    if (!count) return;
    if (count > buf_.size()) throw std::logic_error("DeviceBuf H2D overflow");
    cuda_check(cudaMemcpyAsync(buf_.get(), src, checked_mul(count, sizeof(T)),
                               cudaMemcpyHostToDevice, stream),
               "H2D");
  }

  void copy_to_host(T* dst, std::size_t count, cudaStream_t stream) const {
    if (!count) return;
    if (count > buf_.size()) throw std::logic_error("DeviceBuf D2H overflow");
    cuda_check(cudaMemcpyAsync(dst, buf_.get(), checked_mul(count, sizeof(T)),
                               cudaMemcpyDeviceToHost, stream),
               "D2H");
  }

  void copy_from_device(const T* src, std::size_t count, cudaStream_t stream) {
    if (!count) return;
    if (count > buf_.size()) throw std::logic_error("DeviceBuf D2D overflow");
    cuda_check(cudaMemcpyAsync(buf_.get(), src, checked_mul(count, sizeof(T)),
                               cudaMemcpyDeviceToDevice, stream),
               "D2D");
  }

  void reset() noexcept {
    buf_.reset();
    charge_.reset();
  }
  [[nodiscard]] T* get() noexcept { return buf_.get(); }
  [[nodiscard]] const T* get() const noexcept { return buf_.get(); }
  [[nodiscard]] std::size_t size() const noexcept { return buf_.size(); }
  [[nodiscard]] std::size_t bytes() const noexcept { return buf_.size() * sizeof(T); }
  [[nodiscard]] bool empty() const noexcept { return buf_.size() == 0; }

 private:
  UccDeviceCharge charge_;
  runtime::OwnedCudaBuffer<T> buf_;
};

// --- Row-major cuBLAS GEMM / Batched GEMM / BLAS-1 wrappers ---
inline int checked_blas_int(std::size_t v) {
  if (v > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("DF-UCC v2 CUDA BLAS dimension exceeds 32-bit int");
  return static_cast<int>(v);
}

template <class Real>
inline void rm_gemm(cublasHandle_t handle, cudaStream_t stream, char ta, char tb, std::size_t m,
                    std::size_t n, std::size_t k, const Real* a, const Real* b, Real* c,
                    Real alpha = Real(1), Real beta = Real(0)) {
  if (!m || !n) return;
  if (!k) {
    if (beta == Real(0)) {
      cuda_check(cudaMemsetAsync(c, 0, checked_mul(checked_mul(m, n), sizeof(Real)), stream),
                 "gemm zero");
    }
    return;
  }
  const int mi = checked_blas_int(m);
  const int ni = checked_blas_int(n);
  const int ki = checked_blas_int(k);
  const cublasOperation_t opa = (ta == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t opb = (tb == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int lda = (ta == 'N') ? ki : mi;
  const int ldb = (tb == 'N') ? ni : ki;
  if constexpr (std::is_same_v<Real, double>) {
    cublas_check(cublasDgemm(handle, opb, opa, ni, mi, ki, &alpha, b, ldb, a, lda, &beta, c, ni),
                 "dgemm");
  } else {
    cublas_check(cublasSgemm(handle, opb, opa, ni, mi, ki, &alpha, b, ldb, a, lda, &beta, c, ni),
                 "sgemm");
  }
}

template <class Real>
inline void rm_gemm_batched(cublasHandle_t handle, cudaStream_t stream, char ta, char tb,
                            std::size_t m, std::size_t n, std::size_t k, const Real* a,
                            long long stride_a, const Real* b, long long stride_b, Real* c,
                            long long stride_c, std::size_t batches, Real alpha = Real(1),
                            Real beta = Real(0)) {
  if (!batches || !m || !n) return;
  if (!k) {
    if (beta == Real(0)) {
      for (std::size_t bi = 0; bi < batches; ++bi) {
        cuda_check(cudaMemsetAsync(c + bi * static_cast<std::size_t>(stride_c), 0,
                                   checked_mul(checked_mul(m, n), sizeof(Real)), stream),
                   "batched gemm zero");
      }
    }
    return;
  }
  const int mi = checked_blas_int(m);
  const int ni = checked_blas_int(n);
  const int ki = checked_blas_int(k);
  const int bi = checked_blas_int(batches);
  const cublasOperation_t opa = (ta == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  const cublasOperation_t opb = (tb == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int lda = (ta == 'N') ? ki : mi;
  const int ldb = (tb == 'N') ? ni : ki;
  if constexpr (std::is_same_v<Real, double>) {
    cublas_check(cublasDgemmStridedBatched(handle, opb, opa, ni, mi, ki, &alpha, b, ldb, stride_b,
                                           a, lda, stride_a, &beta, c, ni, stride_c, bi),
                 "dgemmStridedBatched");
  } else {
    cublas_check(cublasSgemmStridedBatched(handle, opb, opa, ni, mi, ki, &alpha, b, ldb, stride_b,
                                           a, lda, stride_a, &beta, c, ni, stride_c, bi),
                 "sgemmStridedBatched");
  }
}

template <class Real>
inline double rm_dot(cublasHandle_t handle, std::size_t count, const Real* x, const Real* y) {
  if (!count) return 0.0;
  constexpr std::size_t kChunk = static_cast<std::size_t>(std::numeric_limits<int>::max());
  double total = 0.0;
  for (std::size_t off = 0; off < count; off += kChunk) {
    const int n = static_cast<int>(std::min(kChunk, count - off));
    if constexpr (std::is_same_v<Real, double>) {
      double part = 0.0;
      cublas_check(cublasDdot(handle, n, x + off, 1, y + off, 1, &part), "ddot");
      total += part;
    } else {
      float part = 0.0f;
      cublas_check(cublasSdot(handle, n, x + off, 1, y + off, 1, &part), "sdot");
      total += static_cast<double>(part);
    }
  }
  return total;
}

template <class Real>
inline void rm_axpy(cublasHandle_t handle, std::size_t count, Real alpha, const Real* x, Real* y) {
  if (!count || alpha == Real(0)) return;
  constexpr std::size_t kChunk = static_cast<std::size_t>(std::numeric_limits<int>::max());
  for (std::size_t off = 0; off < count; off += kChunk) {
    const int n = static_cast<int>(std::min(kChunk, count - off));
    if constexpr (std::is_same_v<Real, double>) {
      cublas_check(cublasDaxpy(handle, n, &alpha, x + off, 1, y + off, 1), "daxpy");
    } else {
      cublas_check(cublasSaxpy(handle, n, &alpha, x + off, 1, y + off, 1), "saxpy");
    }
  }
}

// --- Common Device Kernels ---
template <class Src, class Dst>
__global__ void cast_array_kernel(const Src* src, Dst* dst, std::size_t count) {
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    dst[i] = static_cast<Dst>(src[i]);
  }
}

template <class Src, class Dst>
inline void launch_cast(const Src* src, Dst* dst, std::size_t count, cudaStream_t stream) {
  if (!count) return;
  cast_array_kernel<Src, Dst><<<grid_1d(count), 256, 0, stream>>>(src, dst, count);
  cuda_check(cudaGetLastError(), "cast_array");
}

template <class Real>
__global__ void add_scaled_kernel(Real* dst, const Real* src, Real scale, std::size_t count) {
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    dst[i] += scale * src[i];
  }
}

template <class Real>
inline void launch_add_scaled(Real* dst, const Real* src, Real scale, std::size_t count,
                              cudaStream_t stream) {
  if (!count || scale == Real(0)) return;
  add_scaled_kernel<Real><<<grid_1d(count), 256, 0, stream>>>(dst, src, scale, count);
  cuda_check(cudaGetLastError(), "add_scaled");
}

template <class Real>
__global__ void max_abs_finite_stage1_kernel(const Real* values, std::size_t count,
                                             double* block_max, int* nonfinite) {
  __shared__ double smem[256];
  double local = 0.0;
  int bad = 0;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const double v = static_cast<double>(values[i]);
    if (!isfinite(v)) {
      bad = 1;
    } else {
      const double a = fabs(v);
      if (a > local) local = a;
    }
  }
  if (bad) atomicExch(nonfinite, 1);
  smem[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = 128U; stride > 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      if (smem[threadIdx.x + stride] > smem[threadIdx.x])
        smem[threadIdx.x] = smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) block_max[blockIdx.x] = smem[0];
}

static __global__ void max_abs_finite_stage2_kernel(const double* block_max, unsigned nblocks,
                                                    double* out_max) {
  __shared__ double smem[256];
  double local = 0.0;
  for (unsigned i = threadIdx.x; i < nblocks; i += blockDim.x) {
    if (block_max[i] > local) local = block_max[i];
  }
  smem[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = 128U; stride > 0U; stride >>= 1U) {
    if (threadIdx.x < stride) {
      if (smem[threadIdx.x + stride] > smem[threadIdx.x])
        smem[threadIdx.x] = smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *out_max = smem[0];
}

template <class Real>
inline double device_max_abs(const Real* values, std::size_t count, int device,
                             cudaStream_t stream) {
  if (!count) return 0.0;
  const unsigned nblocks = std::min<unsigned>(grid_1d(count, 256), 1024U);
  DeviceBuf<double> d_blocks(device, nblocks, stream);
  DeviceBuf<double> d_out(device, 1, stream);
  DeviceBuf<int> d_bad(device, 1, stream);
  d_bad.zero(stream);
  max_abs_finite_stage1_kernel<Real><<<nblocks, 256, 0, stream>>>(values, count, d_blocks.get(),
                                                                  d_bad.get());
  cuda_check(cudaGetLastError(), "max_abs stage1");
  max_abs_finite_stage2_kernel<<<1, 256, 0, stream>>>(d_blocks.get(), nblocks, d_out.get());
  cuda_check(cudaGetLastError(), "max_abs stage2");
  double host_max = 0.0;
  int host_bad = 0;
  d_out.copy_to_host(&host_max, 1, stream);
  d_bad.copy_to_host(&host_bad, 1, stream);
  cuda_check(cudaStreamSynchronize(stream), "max_abs sync");
  if (host_bad || !std::isfinite(host_max)) return std::numeric_limits<double>::quiet_NaN();
  return host_max;
}

template <class Real>
__global__ void antisymmetrize_t2_kernel(std::size_t o, std::size_t v, const Real* src,
                                         Real* dst) {
  const std::size_t total = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t rem1 = idx / v;
    const std::size_t a = rem1 % v;
    const std::size_t rem2 = rem1 / v;
    const std::size_t j = rem2 % o;
    const std::size_t i = rem2 / o;
    const Real v00 = src[((i * o + j) * v + a) * v + b];
    const Real v10 = src[((j * o + i) * v + a) * v + b];
    const Real v01 = src[((i * o + j) * v + b) * v + a];
    const Real v11 = src[((j * o + i) * v + b) * v + a];
    dst[idx] = Real(0.25) * (v00 - v10 - v01 + v11);
  }
}

template <class Real>
inline void launch_antisymmetrize_t2(std::size_t o, std::size_t v, Real* t2, int device,
                                     cudaStream_t stream) {
  const std::size_t total = o * o * v * v;
  if (!total) return;
  DeviceBuf<Real> tmp(device, total, stream);
  antisymmetrize_t2_kernel<Real><<<grid_1d(total), 256, 0, stream>>>(o, v, t2, tmp.get());
  cuda_check(cudaGetLastError(), "antisymmetrize_t2");
  cuda_check(cudaMemcpyAsync(t2, tmp.get(), checked_mul(total, sizeof(Real)),
                             cudaMemcpyDeviceToDevice, stream),
             "antisymmetrize_t2 copy");
}

template <class Real>
__global__ void build_tau_ss_kernel(std::size_t o, std::size_t v, const Real* t2, const Real* t1,
                                    Real factor, Real* tau) {
  const std::size_t total = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t rem1 = idx / v;
    const std::size_t a = rem1 % v;
    const std::size_t rem2 = rem1 / v;
    const std::size_t j = rem2 % o;
    const std::size_t i = rem2 / o;
    tau[idx] = t2[idx] + factor * (t1[i * v + a] * t1[j * v + b] - t1[i * v + b] * t1[j * v + a]);
  }
}

template <class Real>
__global__ void build_tau_ab_kernel(std::size_t oa, std::size_t ob, std::size_t va, std::size_t vb,
                                    const Real* t2ab, const Real* t1a, const Real* t1b, Real factor,
                                    Real* tau_ab) {
  const std::size_t total = oa * ob * va * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % vb;
    const std::size_t rem1 = idx / vb;
    const std::size_t a = rem1 % va;
    const std::size_t rem2 = rem1 / va;
    const std::size_t j = rem2 % ob;
    const std::size_t i = rem2 / ob;
    tau_ab[idx] = t2ab[idx] + factor * t1a[i * va + a] * t1b[j * vb + b];
  }
}

// Generic 4D permutation: dst[p0, p1, p2, p3] = src[i0, i1, i2, i3]
// where dst axis k reads src axis perm[k].
template <class Real, int P0, int P1, int P2, int P3>
__global__ void permute4d_kernel(std::size_t d0, std::size_t d1, std::size_t d2, std::size_t d3,
                                 const Real* src, Real* dst, bool add_into) {
  const std::size_t dims[4] = {d0, d1, d2, d3};
  const std::size_t sdims[4] = {dims[P0], dims[P1], dims[P2], dims[P3]};
  const std::size_t total = d0 * d1 * d2 * d3;
  for (std::size_t dst_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       dst_idx < total; dst_idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::size_t tmp = dst_idx;
    const std::size_t o3 = tmp % sdims[3]; tmp /= sdims[3];
    const std::size_t o2 = tmp % sdims[2]; tmp /= sdims[2];
    const std::size_t o1 = tmp % sdims[1];
    const std::size_t o0 = tmp / sdims[1];
    std::size_t src_coords[4];
    src_coords[P0] = o0;
    src_coords[P1] = o1;
    src_coords[P2] = o2;
    src_coords[P3] = o3;
    const std::size_t src_idx =
        ((src_coords[0] * d1 + src_coords[1]) * d2 + src_coords[2]) * d3 + src_coords[3];
    if (add_into) {
      dst[dst_idx] += src[src_idx];
    } else {
      dst[dst_idx] = src[src_idx];
    }
  }
}

template <class Real, int P0, int P1, int P2, int P3>
inline void launch_permute4d(std::size_t d0, std::size_t d1, std::size_t d2, std::size_t d3,
                             const Real* src, Real* dst, cudaStream_t stream,
                             bool add_into = false) {
  const std::size_t total = d0 * d1 * d2 * d3;
  if (!total) return;
  permute4d_kernel<Real, P0, P1, P2, P3>
      <<<grid_1d(total), 256, 0, stream>>>(d0, d1, d2, d3, src, dst, add_into);
  cuda_check(cudaGetLastError(), "permute4d");
}

// Generic 3D permutation: dst[p0, p1, p2] = src[i0, i1, i2]
template <class Real, int P0, int P1, int P2>
__global__ void permute3d_kernel(std::size_t d0, std::size_t d1, std::size_t d2, const Real* src,
                                 Real* dst, Real scale, bool add_into) {
  const std::size_t dims[3] = {d0, d1, d2};
  const std::size_t sdims[3] = {dims[P0], dims[P1], dims[P2]};
  const std::size_t total = d0 * d1 * d2;
  for (std::size_t dst_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       dst_idx < total; dst_idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::size_t tmp = dst_idx;
    const std::size_t o2 = tmp % sdims[2]; tmp /= sdims[2];
    const std::size_t o1 = tmp % sdims[1];
    const std::size_t o0 = tmp / sdims[1];
    std::size_t sc[3];
    sc[P0] = o0; sc[P1] = o1; sc[P2] = o2;
    const std::size_t src_idx = (sc[0] * d1 + sc[1]) * d2 + sc[2];
    const Real val = scale * src[src_idx];
    if (add_into) {
      dst[dst_idx] += val;
    } else {
      dst[dst_idx] = val;
    }
  }
}

template <class Real, int P0, int P1, int P2>
inline void launch_permute3d(std::size_t d0, std::size_t d1, std::size_t d2, const Real* src,
                             Real* dst, cudaStream_t stream, Real scale = Real(1),
                             bool add_into = false) {
  const std::size_t total = d0 * d1 * d2;
  if (!total) return;
  permute3d_kernel<Real, P0, P1, P2>
      <<<grid_1d(total), 256, 0, stream>>>(d0, d1, d2, src, dst, scale, add_into);
  cuda_check(cudaGetLastError(), "permute3d");
}

// --- Core GPU Data Containers ---
template <class Real>
struct DeviceUccAmplitudes {
  SpinBlockedDims dims{};
  DeviceBuf<Real> flat;
  std::size_t off_1a{}, off_1b{}, off_2aa{}, off_2ab{}, off_2bb{};
  std::size_t sz_1a{}, sz_1b{}, sz_2aa{}, sz_2ab{}, sz_2bb{}, total{};

  void allocate_zero(int device, const SpinBlockedDims& d, cudaStream_t stream) {
    dims = d;
    sz_1a = d.oa * d.va;
    sz_1b = d.ob * d.vb;
    sz_2aa = d.oa * d.oa * d.va * d.va;
    sz_2ab = d.oa * d.ob * d.va * d.vb;
    sz_2bb = d.ob * d.ob * d.vb * d.vb;
    off_1a = 0;
    off_1b = sz_1a;
    off_2aa = off_1b + sz_1b;
    off_2ab = off_2aa + sz_2aa;
    off_2bb = off_2ab + sz_2ab;
    total = off_2bb + sz_2bb;
    flat.allocate_zero(device, total, stream);
  }

  void zero(cudaStream_t stream) { flat.zero(stream); }

  Real* t1a() noexcept { return sz_1a ? flat.get() + off_1a : nullptr; }
  Real* t1b() noexcept { return sz_1b ? flat.get() + off_1b : nullptr; }
  Real* t2aa() noexcept { return sz_2aa ? flat.get() + off_2aa : nullptr; }
  Real* t2ab() noexcept { return sz_2ab ? flat.get() + off_2ab : nullptr; }
  Real* t2bb() noexcept { return sz_2bb ? flat.get() + off_2bb : nullptr; }

  const Real* t1a() const noexcept { return sz_1a ? flat.get() + off_1a : nullptr; }
  const Real* t1b() const noexcept { return sz_1b ? flat.get() + off_1b : nullptr; }
  const Real* t2aa() const noexcept { return sz_2aa ? flat.get() + off_2aa : nullptr; }
  const Real* t2ab() const noexcept { return sz_2ab ? flat.get() + off_2ab : nullptr; }
  const Real* t2bb() const noexcept { return sz_2bb ? flat.get() + off_2bb : nullptr; }

  const Real* t1(int s) const noexcept { return s == 0 ? t1a() : t1b(); }
  Real* t1(int s) noexcept { return s == 0 ? t1a() : t1b(); }
  const Real* t2(int s) const noexcept { return s == 0 ? t2aa() : t2bb(); }
  Real* t2(int s) noexcept { return s == 0 ? t2aa() : t2bb(); }
  std::size_t n1(int s) const noexcept { return s == 0 ? sz_1a : sz_1b; }
  std::size_t n2(int s) const noexcept { return s == 0 ? sz_2aa : sz_2bb; }
};

template <class Real>
struct DeviceFourIndexCore {
  DeviceBuf<Real> goovv_aa, goovv_ab, goovv_bb;
  DeviceBuf<Real> goooo_aa, goooo_ab, goooo_bb;
  DeviceBuf<Real> gooov_aa, gooov_bb, gooov_ab, gooov_ba;
  DeviceBuf<Real> govov_aa, govov_bb, govov_ab, govov_ba;

  void allocate_zero(int device, const SpinBlockedDims& d, cudaStream_t stream) {
    goovv_aa.allocate_zero(device, d.oa * d.oa * d.va * d.va, stream);
    goovv_ab.allocate_zero(device, d.oa * d.ob * d.va * d.vb, stream);
    goovv_bb.allocate_zero(device, d.ob * d.ob * d.vb * d.vb, stream);
    goooo_aa.allocate_zero(device, d.oa * d.oa * d.oa * d.oa, stream);
    goooo_ab.allocate_zero(device, d.oa * d.ob * d.oa * d.ob, stream);
    goooo_bb.allocate_zero(device, d.ob * d.ob * d.ob * d.ob, stream);
    gooov_aa.allocate_zero(device, d.oa * d.oa * d.oa * d.va, stream);
    gooov_bb.allocate_zero(device, d.ob * d.ob * d.ob * d.vb, stream);
    gooov_ab.allocate_zero(device, d.oa * d.ob * d.oa * d.vb, stream);
    gooov_ba.allocate_zero(device, d.ob * d.oa * d.ob * d.va, stream);
    govov_aa.allocate_zero(device, d.oa * d.va * d.oa * d.va, stream);
    govov_bb.allocate_zero(device, d.ob * d.vb * d.ob * d.vb, stream);
    govov_ab.allocate_zero(device, d.oa * d.vb * d.oa * d.vb, stream);
    govov_ba.allocate_zero(device, d.ob * d.va * d.ob * d.va, stream);
  }

  std::size_t total_bytes() const noexcept {
    return goovv_aa.bytes() + goovv_ab.bytes() + goovv_bb.bytes() + goooo_aa.bytes() +
           goooo_ab.bytes() + goooo_bb.bytes() + gooov_aa.bytes() + gooov_bb.bytes() +
           gooov_ab.bytes() + gooov_ba.bytes() + govov_aa.bytes() + govov_bb.bytes() +
           govov_ab.bytes() + govov_ba.bytes();
  }

  const Real* goovv(int s) const noexcept { return s == 0 ? goovv_aa.get() : goovv_bb.get(); }
  Real* goovv(int s) noexcept { return s == 0 ? goovv_aa.get() : goovv_bb.get(); }
  const Real* goooo(int s) const noexcept { return s == 0 ? goooo_aa.get() : goooo_bb.get(); }
  Real* goooo(int s) noexcept { return s == 0 ? goooo_aa.get() : goooo_bb.get(); }
  const Real* gooov(int s) const noexcept { return s == 0 ? gooov_aa.get() : gooov_bb.get(); }
  Real* gooov(int s) noexcept { return s == 0 ? gooov_aa.get() : gooov_bb.get(); }
  const Real* govov(int s) const noexcept { return s == 0 ? govov_aa.get() : govov_bb.get(); }
  Real* govov(int s) noexcept { return s == 0 ? govov_aa.get() : govov_bb.get(); }
};

template <class Real>
inline DeviceFourIndexCore<Real> cast_four_index_core(const DeviceFourIndexCore<double>& src,
                                                      int device, const SpinBlockedDims& d,
                                                      cudaStream_t stream) {
  DeviceFourIndexCore<Real> out;
  out.allocate_zero(device, d, stream);
  launch_cast(src.goovv_aa.get(), out.goovv_aa.get(), src.goovv_aa.size(), stream);
  launch_cast(src.goovv_ab.get(), out.goovv_ab.get(), src.goovv_ab.size(), stream);
  launch_cast(src.goovv_bb.get(), out.goovv_bb.get(), src.goovv_bb.size(), stream);
  launch_cast(src.goooo_aa.get(), out.goooo_aa.get(), src.goooo_aa.size(), stream);
  launch_cast(src.goooo_ab.get(), out.goooo_ab.get(), src.goooo_ab.size(), stream);
  launch_cast(src.goooo_bb.get(), out.goooo_bb.get(), src.goooo_bb.size(), stream);
  launch_cast(src.gooov_aa.get(), out.gooov_aa.get(), src.gooov_aa.size(), stream);
  launch_cast(src.gooov_bb.get(), out.gooov_bb.get(), src.gooov_bb.size(), stream);
  launch_cast(src.gooov_ab.get(), out.gooov_ab.get(), src.gooov_ab.size(), stream);
  launch_cast(src.gooov_ba.get(), out.gooov_ba.get(), src.gooov_ba.size(), stream);
  launch_cast(src.govov_aa.get(), out.govov_aa.get(), src.govov_aa.size(), stream);
  launch_cast(src.govov_bb.get(), out.govov_bb.get(), src.govov_bb.size(), stream);
  launch_cast(src.govov_ab.get(), out.govov_ab.get(), src.govov_ab.size(), stream);
  launch_cast(src.govov_ba.get(), out.govov_ba.get(), src.govov_ba.size(), stream);
  return out;
}

template <class Real>
struct DeviceFactorSet {
  DeviceBuf<Real> boo_a, bov_a, bvv_a;
  DeviceBuf<Real> boo_b, bov_b, bvv_b;

  void allocate_zero(int device, const SpinBlockedDims& d, cudaStream_t stream) {
    boo_a.allocate_zero(device, d.q * d.oa * d.oa, stream);
    bov_a.allocate_zero(device, d.q * d.oa * d.va, stream);
    bvv_a.allocate_zero(device, d.q * d.va * d.va, stream);
    boo_b.allocate_zero(device, d.q * d.ob * d.ob, stream);
    bov_b.allocate_zero(device, d.q * d.ob * d.vb, stream);
    bvv_b.allocate_zero(device, d.q * d.vb * d.vb, stream);
  }

  std::size_t total_bytes() const noexcept {
    return boo_a.bytes() + bov_a.bytes() + bvv_a.bytes() + boo_b.bytes() + bov_b.bytes() +
           bvv_b.bytes();
  }

  Real* boo(int s) noexcept { return s == 0 ? boo_a.get() : boo_b.get(); }
  Real* bov(int s) noexcept { return s == 0 ? bov_a.get() : bov_b.get(); }
  Real* bvv(int s) noexcept { return s == 0 ? bvv_a.get() : bvv_b.get(); }
  const Real* boo(int s) const noexcept { return s == 0 ? boo_a.get() : boo_b.get(); }
  const Real* bov(int s) const noexcept { return s == 0 ? bov_a.get() : bov_b.get(); }
  const Real* bvv(int s) const noexcept { return s == 0 ? bvv_a.get() : bvv_b.get(); }
};

using DeviceFactorGrads = DeviceFactorSet<double>;

template <class Real>
struct DeviceFockSet {
  DeviceBuf<Real> foo_a, fov_a, fvv_a;
  DeviceBuf<Real> foo_b, fov_b, fvv_b;

  void allocate_zero(int device, const SpinBlockedDims& d, cudaStream_t stream) {
    foo_a.allocate_zero(device, d.oa * d.oa, stream);
    fov_a.allocate_zero(device, d.oa * d.va, stream);
    fvv_a.allocate_zero(device, d.va * d.va, stream);
    foo_b.allocate_zero(device, d.ob * d.ob, stream);
    fov_b.allocate_zero(device, d.ob * d.vb, stream);
    fvv_b.allocate_zero(device, d.vb * d.vb, stream);
  }

  Real* foo(int s) noexcept { return s == 0 ? foo_a.get() : foo_b.get(); }
  Real* fov(int s) noexcept { return s == 0 ? fov_a.get() : fov_b.get(); }
  Real* fvv(int s) noexcept { return s == 0 ? fvv_a.get() : fvv_b.get(); }
  const Real* foo(int s) const noexcept { return s == 0 ? foo_a.get() : foo_b.get(); }
  const Real* fov(int s) const noexcept { return s == 0 ? fov_a.get() : fov_b.get(); }
  const Real* fvv(int s) const noexcept { return s == 0 ? fvv_a.get() : fvv_b.get(); }
};

using DeviceFockGrads = DeviceFockSet<double>;

template <class Real>
struct DeviceFixedF12Coupling {
  double fixed_geminal_energy{};
  DeviceBuf<Real> w_vo_aaaa, w_vo_bbbb, w_vo_abab, w_vo_abba;
  DeviceBuf<Real> w_vv_aaaa, w_vv_abab, w_vv_bbbb;
  DeviceBuf<Real> c_f12_aaaa, c_f12_abab, c_f12_bbbb;

  void upload(int device, const FixedF12Coupling& host, cudaStream_t stream) {
    fixed_geminal_energy = host.fixed_geminal_energy;
    auto up = [&](const std::vector<double>& src, DeviceBuf<Real>& dst) {
      dst.allocate(device, src.size(), stream);
      if (src.empty()) return;
      if constexpr (std::is_same_v<Real, double>) {
        dst.copy_from_host(src.data(), src.size(), stream);
      } else {
        std::vector<float> tmp(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) tmp[i] = static_cast<float>(src[i]);
        dst.copy_from_host(tmp.data(), tmp.size(), stream);
        cuda_check(cudaStreamSynchronize(stream), "f12 f32 upload sync");
      }
    };
    up(host.w_vo_aaaa, w_vo_aaaa);
    up(host.w_vo_bbbb, w_vo_bbbb);
    up(host.w_vo_abab, w_vo_abab);
    up(host.w_vo_abba, w_vo_abba);
    up(host.w_vv_aaaa, w_vv_aaaa);
    up(host.w_vv_abab, w_vv_abab);
    up(host.w_vv_bbbb, w_vv_bbbb);
    up(host.c_f12_aaaa, c_f12_aaaa);
    up(host.c_f12_abab, c_f12_abab);
    up(host.c_f12_bbbb, c_f12_bbbb);
  }
};

struct DeviceFixedF12Cotangents {
  double bar_fixed_geminal_energy{};
  DeviceBuf<double> bar_w_vo_aaaa, bar_w_vo_bbbb, bar_w_vo_abab, bar_w_vo_abba;
  DeviceBuf<double> bar_w_vv_aaaa, bar_w_vv_abab, bar_w_vv_bbbb;
  DeviceBuf<double> bar_c_f12_aaaa, bar_c_f12_abab, bar_c_f12_bbbb;

  void allocate_zero(int device, const FixedF12Coupling& host, cudaStream_t stream) {
    bar_fixed_geminal_energy = 0.0;
    bar_w_vo_aaaa.allocate_zero(device, host.w_vo_aaaa.size(), stream);
    bar_w_vo_bbbb.allocate_zero(device, host.w_vo_bbbb.size(), stream);
    bar_w_vo_abab.allocate_zero(device, host.w_vo_abab.size(), stream);
    bar_w_vo_abba.allocate_zero(device, host.w_vo_abba.size(), stream);
    bar_w_vv_aaaa.allocate_zero(device, host.w_vv_aaaa.size(), stream);
    bar_w_vv_abab.allocate_zero(device, host.w_vv_abab.size(), stream);
    bar_w_vv_bbbb.allocate_zero(device, host.w_vv_bbbb.size(), stream);
    bar_c_f12_aaaa.allocate_zero(device, host.c_f12_aaaa.size(), stream);
    bar_c_f12_abab.allocate_zero(device, host.c_f12_abab.size(), stream);
    bar_c_f12_bbbb.allocate_zero(device, host.c_f12_bbbb.size(), stream);
  }

  FixedF12Cotangents download(cudaStream_t stream) const {
    FixedF12Cotangents out;
    out.bar_fixed_geminal_energy = bar_fixed_geminal_energy;
    auto dl = [&](const DeviceBuf<double>& src, std::vector<double>& dst) {
      dst.assign(src.size(), 0.0);
      if (!src.empty()) src.copy_to_host(dst.data(), src.size(), stream);
    };
    dl(bar_w_vo_aaaa, out.bar_w_vo_aaaa);
    dl(bar_w_vo_bbbb, out.bar_w_vo_bbbb);
    dl(bar_w_vo_abab, out.bar_w_vo_abab);
    dl(bar_w_vo_abba, out.bar_w_vo_abba);
    dl(bar_w_vv_aaaa, out.bar_w_vv_aaaa);
    dl(bar_w_vv_abab, out.bar_w_vv_abab);
    dl(bar_w_vv_bbbb, out.bar_w_vv_bbbb);
    dl(bar_c_f12_aaaa, out.bar_c_f12_aaaa);
    dl(bar_c_f12_abab, out.bar_c_f12_abab);
    dl(bar_c_f12_bbbb, out.bar_c_f12_bbbb);
    cuda_check(cudaStreamSynchronize(stream), "f12 cotangents download");
    return out;
  }
};

template <class Real>
struct DeviceSourceView {
  SpinBlockedDims dims{};
  const Real* boo[2]{};
  const Real* bov[2]{};
  const Real* bvv[2]{};
  const Real* foo[2]{};
  const Real* fov[2]{};
  const Real* fvv[2]{};
  const Real* eps_o[2]{};
  const Real* eps_v[2]{};
  std::size_t o(int s) const noexcept { return s == 0 ? dims.oa : dims.ob; }
  std::size_t v(int s) const noexcept { return s == 0 ? dims.va : dims.vb; }
};

struct PlanDelete {
  void operator()(scf::CudaDensityFittingJkPlan* p) const noexcept {
    if (!p) return;
    int prev = 0;
    const bool restore = cudaGetDevice(&prev) == cudaSuccess;
    scf::destroy_cuda_density_fitting_jk_plan(p);
    if (restore) (void)cudaSetDevice(prev);
  }
};

// Complete GPU-resident spin-blocked DF source owner.
// Declaration order ensures all stream-bound buffers drain and die BEFORE owned_plan.
struct CudaSpinBlockedSource {
  SpinBlockedDims dims{};
  int device{0};
  std::size_t device_cap{0};
  std::size_t plan_resident_bytes{0};
  double metric_relative_threshold{1e-10};

  std::unique_ptr<scf::CudaDensityFittingJkPlan, PlanDelete> owned_plan;
  cudaStream_t stream{};
  cublasHandle_t blas{};
  DeviceBuf<unsigned char> blas_workspace;

  // Device active 3-index whitened DF factors (Q-major)
  DeviceFactorSet<double> factors;
  // Device active MO Fock blocks
  DeviceFockSet<double> fock;
  // Device active canonical orbital energies
  DeviceBuf<double> eps_oa, eps_va, eps_ob, eps_vb;
  // Device active MO coefficient matrices [n, n_act_a] and [n, n_act_b] (retained for pullback)
  DeviceBuf<double> c_act_a, c_act_b;

  // Host copies of active orbital energies (small O(n) vectors for degeneracy/gap checks)
  std::vector<double> host_eps_oa, host_eps_va;
  std::vector<double> host_eps_ob, host_eps_vb;

  DeviceSourceView<double> view() const noexcept {
    DeviceSourceView<double> sv;
    sv.dims = dims;
    sv.boo[0] = factors.boo_a.get(); sv.boo[1] = factors.boo_b.get();
    sv.bov[0] = factors.bov_a.get(); sv.bov[1] = factors.bov_b.get();
    sv.bvv[0] = factors.bvv_a.get(); sv.bvv[1] = factors.bvv_b.get();
    sv.foo[0] = fock.foo_a.get(); sv.foo[1] = fock.foo_b.get();
    sv.fov[0] = fock.fov_a.get(); sv.fov[1] = fock.fov_b.get();
    sv.fvv[0] = fock.fvv_a.get(); sv.fvv[1] = fock.fvv_b.get();
    sv.eps_o[0] = eps_oa.get(); sv.eps_o[1] = eps_ob.get();
    sv.eps_v[0] = eps_va.get(); sv.eps_v[1] = eps_vb.get();
    return sv;
  }
};

std::size_t source_resident_bytes(const CudaSpinBlockedSource& src);

// Owned FP32 cast of CudaSpinBlockedSource for mixed-precision CCSD / Lambda warmup.
struct DeviceSourceFp32 {
  SpinBlockedDims dims{};
  DeviceFactorSet<float> factors;
  DeviceFockSet<float> fock;
  DeviceBuf<float> eps_oa, eps_va, eps_ob, eps_vb;

  void cast_from(const CudaSpinBlockedSource& src) {
    dims = src.dims;
    const int dev = src.device;
    const cudaStream_t st = src.stream;
    factors.allocate_zero(dev, dims, st);
    fock.allocate_zero(dev, dims, st);
    eps_oa.allocate(dev, dims.oa, st);
    eps_va.allocate(dev, dims.va, st);
    eps_ob.allocate(dev, dims.ob, st);
    eps_vb.allocate(dev, dims.vb, st);
    launch_cast(src.factors.boo_a.get(), factors.boo_a.get(), src.factors.boo_a.size(), st);
    launch_cast(src.factors.bov_a.get(), factors.bov_a.get(), src.factors.bov_a.size(), st);
    launch_cast(src.factors.bvv_a.get(), factors.bvv_a.get(), src.factors.bvv_a.size(), st);
    launch_cast(src.factors.boo_b.get(), factors.boo_b.get(), src.factors.boo_b.size(), st);
    launch_cast(src.factors.bov_b.get(), factors.bov_b.get(), src.factors.bov_b.size(), st);
    launch_cast(src.factors.bvv_b.get(), factors.bvv_b.get(), src.factors.bvv_b.size(), st);
    launch_cast(src.fock.foo_a.get(), fock.foo_a.get(), src.fock.foo_a.size(), st);
    launch_cast(src.fock.fov_a.get(), fock.fov_a.get(), src.fock.fov_a.size(), st);
    launch_cast(src.fock.fvv_a.get(), fock.fvv_a.get(), src.fock.fvv_a.size(), st);
    launch_cast(src.fock.foo_b.get(), fock.foo_b.get(), src.fock.foo_b.size(), st);
    launch_cast(src.fock.fov_b.get(), fock.fov_b.get(), src.fock.fov_b.size(), st);
    launch_cast(src.fock.fvv_b.get(), fock.fvv_b.get(), src.fock.fvv_b.size(), st);
    launch_cast(src.eps_oa.get(), eps_oa.get(), dims.oa, st);
    launch_cast(src.eps_va.get(), eps_va.get(), dims.va, st);
    launch_cast(src.eps_ob.get(), eps_ob.get(), dims.ob, st);
    launch_cast(src.eps_vb.get(), eps_vb.get(), dims.vb, st);
  }

  DeviceSourceView<float> view() const noexcept {
    DeviceSourceView<float> sv;
    sv.dims = dims;
    sv.boo[0] = factors.boo_a.get(); sv.boo[1] = factors.boo_b.get();
    sv.bov[0] = factors.bov_a.get(); sv.bov[1] = factors.bov_b.get();
    sv.bvv[0] = factors.bvv_a.get(); sv.bvv[1] = factors.bvv_b.get();
    sv.foo[0] = fock.foo_a.get(); sv.foo[1] = fock.foo_b.get();
    sv.fov[0] = fock.fov_a.get(); sv.fov[1] = fock.fov_b.get();
    sv.fvv[0] = fock.fvv_a.get(); sv.fvv[1] = fock.fvv_b.get();
    sv.eps_o[0] = eps_oa.get(); sv.eps_o[1] = eps_ob.get();
    sv.eps_v[0] = eps_va.get(); sv.eps_v[1] = eps_vb.get();
    return sv;
  }
};

// --- GPU-Resident Packed 5-Block DIIS ---
// Packs antisymmetric t2aa & t2bb into unique (i<j, a<b) triangles (4x smaller)
// with weight 2.0 on error norms so the Gram matrix is exact to machine precision.
inline bool solve_small_linear_system(std::vector<double> a, std::vector<double> b, std::size_t n,
                                      std::vector<double>& x) {
  x.assign(n, 0.0);
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot = col;
    double best = std::abs(a[col * n + col]);
    for (std::size_t row = col + 1; row < n; ++row) {
      const double v = std::abs(a[row * n + col]);
      if (v > best) {
        best = v;
        pivot = row;
      }
    }
    if (!(best > 1e-18) || !std::isfinite(best)) return false;
    if (pivot != col) {
      for (std::size_t k = col; k < n; ++k) std::swap(a[col * n + k], a[pivot * n + k]);
      std::swap(b[col], b[pivot]);
    }
    const double diag = a[col * n + col];
    for (std::size_t row = col + 1; row < n; ++row) {
      const double factor = a[row * n + col] / diag;
      a[row * n + col] = 0.0;
      for (std::size_t k = col + 1; k < n; ++k) a[row * n + k] -= factor * a[col * n + k];
      b[row] -= factor * b[col];
    }
  }
  for (std::size_t ii = n; ii-- > 0;) {
    double sum = b[ii];
    for (std::size_t k = ii + 1; k < n; ++k) sum -= a[ii * n + k] * x[k];
    x[ii] = sum / a[ii * n + ii];
    if (!std::isfinite(x[ii])) return false;
  }
  return true;
}

template <class Real>
__global__ void pack_ss_t2_kernel(std::size_t o, std::size_t v, const Real* full, Real* packed,
                                  Real scale) {
  const std::size_t np_o = (o >= 2) ? (o * (o - 1) / 2) : 0;
  const std::size_t np_v = (v >= 2) ? (v * (v - 1) / 2) : 0;
  const std::size_t total = np_o * np_v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t po = idx / np_v;
    const std::size_t pv = idx % np_v;
    // Map triangular index to (i < j) via row scan formula
    std::size_t j = static_cast<std::size_t>((sqrt(8.0 * static_cast<double>(po) + 1.0) + 1.0) * 0.5);
    while (j * (j - 1) / 2 > po) --j;
    while ((j + 1) * j / 2 <= po) ++j;
    const std::size_t i = po - j * (j - 1) / 2;

    std::size_t b = static_cast<std::size_t>((sqrt(8.0 * static_cast<double>(pv) + 1.0) + 1.0) * 0.5);
    while (b * (b - 1) / 2 > pv) --b;
    while ((b + 1) * b / 2 <= pv) ++b;
    const std::size_t a = pv - b * (b - 1) / 2;

    packed[idx] = scale * full[((i * o + j) * v + a) * v + b];
  }
}

template <class Real>
__global__ void unpack_ss_t2_kernel(std::size_t o, std::size_t v, const Real* packed, Real* full) {
  const std::size_t np_o = (o >= 2) ? (o * (o - 1) / 2) : 0;
  const std::size_t np_v = (v >= 2) ? (v * (v - 1) / 2) : 0;
  const std::size_t total = np_o * np_v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t po = idx / np_v;
    const std::size_t pv = idx % np_v;
    std::size_t j = static_cast<std::size_t>((sqrt(8.0 * static_cast<double>(po) + 1.0) + 1.0) * 0.5);
    while (j * (j - 1) / 2 > po) --j;
    while ((j + 1) * j / 2 <= po) ++j;
    const std::size_t i = po - j * (j - 1) / 2;

    std::size_t b = static_cast<std::size_t>((sqrt(8.0 * static_cast<double>(pv) + 1.0) + 1.0) * 0.5);
    while (b * (b - 1) / 2 > pv) --b;
    while ((b + 1) * b / 2 <= pv) ++b;
    const std::size_t a = pv - b * (b - 1) / 2;

    const Real val = packed[idx];
    full[((i * o + j) * v + a) * v + b] = val;
    full[((j * o + i) * v + a) * v + b] = -val;
    full[((i * o + j) * v + b) * v + a] = -val;
    full[((j * o + i) * v + b) * v + a] = val;
  }
}

template <class Real>
class DeviceDiis {
 public:
  DeviceDiis(int device, cublasHandle_t blas, cudaStream_t stream, const SpinBlockedDims& d,
             int requested_space, std::size_t budget_bytes)
      : device_(device), blas_(blas), stream_(stream), dims_(d) {
    n1a_ = d.oa * d.va;
    n1b_ = d.ob * d.vb;
    n2aa_p_ = (d.oa >= 2 && d.va >= 2) ? (d.oa * (d.oa - 1) / 2) * (d.va * (d.va - 1) / 2) : 0;
    n2ab_ = d.oa * d.ob * d.va * d.vb;
    n2bb_p_ = (d.ob >= 2 && d.vb >= 2) ? (d.ob * (d.ob - 1) / 2) * (d.vb * (d.vb - 1) / 2) : 0;
    packed_len_ = n1a_ + n1b_ + n2aa_p_ + n2ab_ + n2bb_p_;
    max_space_ = ucc_diis_space(requested_space, checked_mul(packed_len_, sizeof(Real)), budget_bytes);
  }

  void clear() {
    vecs_.clear();
    errs_.clear();
    gram_.clear();
  }

  bool disable() {
    const bool had_history = !vecs_.empty();
    max_space_ = 0;
    clear();
    return had_history;
  }

  void push(const DeviceUccAmplitudes<Real>& vec, const DeviceUccAmplitudes<Real>& err) {
    if (max_space_ < 2 || packed_len_ == 0) return;
    if (static_cast<int>(vecs_.size()) >= max_space_) {
      vecs_.pop_front();
      errs_.pop_front();
      const std::size_t old_m = gram_.size();
      std::vector<std::vector<double>> next_gram(old_m - 1, std::vector<double>(old_m - 1, 0.0));
      for (std::size_t i = 1; i < old_m; ++i)
        for (std::size_t j = 1; j < old_m; ++j) next_gram[i - 1][j - 1] = gram_[i][j];
      gram_.swap(next_gram);
    }

    DeviceBuf<Real> pv(device_, packed_len_, stream_);
    DeviceBuf<Real> pe(device_, packed_len_, stream_);
    pack(vec, pv, Real(1));
    pack(err, pe, Real(2));  // Weight 2.0 on unique (i<j,a<b) matches full 4-fold dense norm.

    const std::size_t idx = vecs_.size();
    vecs_.push_back(std::move(pv));
    errs_.push_back(std::move(pe));

    // Update symmetric Gram matrix with only (idx + 1) GPU dot products.
    for (auto& row : gram_) row.push_back(0.0);
    gram_.emplace_back(idx + 1, 0.0);
    for (std::size_t j = 0; j <= idx; ++j) {
      const double dot = rm_dot<Real>(blas_, packed_len_, errs_[idx].get(), errs_[j].get());
      gram_[idx][j] = dot;
      gram_[j][idx] = dot;
    }
  }

  bool extrapolate(DeviceUccAmplitudes<Real>& out) const {
    const std::size_t m = vecs_.size();
    if (m < 2 || packed_len_ == 0) return false;
    const std::size_t dim = m + 1;
    std::vector<double> mat(dim * dim, 0.0);
    std::vector<double> rhs(dim, 0.0);
    double scale = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < m; ++j) mat[i * dim + j] = gram_[i][j];
      scale = std::max(scale, std::abs(gram_[i][i]));
      mat[i * dim + m] = -1.0;
      mat[m * dim + i] = -1.0;
    }
    rhs[m] = -1.0;
    if (!(scale > 0.0) || !std::isfinite(scale)) return false;
    const double inv_scale = 1.0 / scale;
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < m; ++j) mat[i * dim + j] *= inv_scale;
      mat[i * dim + i] += 1e-14;
    }
    std::vector<double> coeffs;
    if (!solve_small_linear_system(std::move(mat), std::move(rhs), dim, coeffs)) return false;

    DeviceBuf<Real> comb(device_, packed_len_, stream_);
    comb.zero(stream_);
    for (std::size_t i = 0; i < m; ++i) {
      rm_axpy<Real>(blas_, packed_len_, static_cast<Real>(coeffs[i]), vecs_[i].get(), comb.get());
    }
    unpack(comb, out);
    return true;
  }

 private:
  void pack(const DeviceUccAmplitudes<Real>& a, DeviceBuf<Real>& dst, Real ss_scale) const {
    std::size_t off = 0;
    if (n1a_) {
      cuda_check(cudaMemcpyAsync(dst.get() + off, a.t1a(), n1a_ * sizeof(Real),
                                 cudaMemcpyDeviceToDevice, stream_),
                 "diis pack t1a");
      off += n1a_;
    }
    if (n1b_) {
      cuda_check(cudaMemcpyAsync(dst.get() + off, a.t1b(), n1b_ * sizeof(Real),
                                 cudaMemcpyDeviceToDevice, stream_),
                 "diis pack t1b");
      off += n1b_;
    }
    if (n2aa_p_) {
      pack_ss_t2_kernel<Real><<<grid_1d(n2aa_p_), 256, 0, stream_>>>(dims_.oa, dims_.va, a.t2aa(),
                                                                     dst.get() + off, ss_scale);
      cuda_check(cudaGetLastError(), "diis pack t2aa");
      off += n2aa_p_;
    }
    if (n2ab_) {
      cuda_check(cudaMemcpyAsync(dst.get() + off, a.t2ab(), n2ab_ * sizeof(Real),
                                 cudaMemcpyDeviceToDevice, stream_),
                 "diis pack t2ab");
      off += n2ab_;
    }
    if (n2bb_p_) {
      pack_ss_t2_kernel<Real><<<grid_1d(n2bb_p_), 256, 0, stream_>>>(dims_.ob, dims_.vb, a.t2bb(),
                                                                     dst.get() + off, ss_scale);
      cuda_check(cudaGetLastError(), "diis pack t2bb");
    }
  }

  void unpack(const DeviceBuf<Real>& src, DeviceUccAmplitudes<Real>& a) const {
    a.zero(stream_);
    std::size_t off = 0;
    if (n1a_) {
      cuda_check(cudaMemcpyAsync(a.t1a(), src.get() + off, n1a_ * sizeof(Real),
                                 cudaMemcpyDeviceToDevice, stream_),
                 "diis unpack t1a");
      off += n1a_;
    }
    if (n1b_) {
      cuda_check(cudaMemcpyAsync(a.t1b(), src.get() + off, n1b_ * sizeof(Real),
                                 cudaMemcpyDeviceToDevice, stream_),
                 "diis unpack t1b");
      off += n1b_;
    }
    if (n2aa_p_) {
      unpack_ss_t2_kernel<Real><<<grid_1d(n2aa_p_), 256, 0, stream_>>>(
          dims_.oa, dims_.va, src.get() + off, a.t2aa());
      cuda_check(cudaGetLastError(), "diis unpack t2aa");
      off += n2aa_p_;
    }
    if (n2ab_) {
      cuda_check(cudaMemcpyAsync(a.t2ab(), src.get() + off, n2ab_ * sizeof(Real),
                                 cudaMemcpyDeviceToDevice, stream_),
                 "diis unpack t2ab");
      off += n2ab_;
    }
    if (n2bb_p_) {
      unpack_ss_t2_kernel<Real><<<grid_1d(n2bb_p_), 256, 0, stream_>>>(
          dims_.ob, dims_.vb, src.get() + off, a.t2bb());
      cuda_check(cudaGetLastError(), "diis unpack t2bb");
    }
  }

  int device_{0};
  cublasHandle_t blas_{};
  cudaStream_t stream_{};
  SpinBlockedDims dims_{};
  std::size_t n1a_{}, n1b_{}, n2aa_p_{}, n2ab_{}, n2bb_p_{}, packed_len_{};
  int max_space_{0};
  std::deque<DeviceBuf<Real>> vecs_;
  std::deque<DeviceBuf<Real>> errs_;
  std::vector<std::vector<double>> gram_;
};

// --- Inter-module CUDA Stage Results and Declarations ---
struct CudaCcsdSolveResult {
  double energy{};
  int iterations{};
  bool fp32_executed{};
  DeviceUccAmplitudes<double> amplitudes;
};

struct CudaTriplesResult {
  double energy{};
  std::vector<double> occupied_alpha;
  std::vector<double> occupied_beta;
  DeviceUccAmplitudes<double> bar_t;
  DeviceFactorGrads bar_b;
  DeviceFourIndexCore<double> bar_core;
  DeviceFockGrads bar_f;
  std::size_t fp32_gemms{};
  std::size_t fp64_gemms{};
};

struct CudaLambdaSolveResult {
  int iterations{};
  bool fp32_executed{};
  DeviceFactorGrads bar_b;
  DeviceFockGrads bar_f;
  FixedF12Cotangents f12_cotangents;
};

// df_ucc_v2_source_cuda.cu
CudaSpinBlockedSource build_spin_blocked_df_source_cuda_impl(
    const core::System& orbital, const core::System& auxiliary,
    const hf::UnrestrictedPhysicalReference& reference, int device, const DFUccV2Options& options);

DeviceFourIndexCore<double> build_four_index_core_cuda_impl(const CudaSpinBlockedSource& src);

void pullback_four_index_core_cuda_impl(const CudaSpinBlockedSource& src,
                                        const DeviceFourIndexCore<double>& bar_core,
                                        DeviceFactorGrads& bar_b);

SpinBlockedSourcePullback pullback_spin_blocked_df_source_cuda_impl(
    const core::System& orbital, const core::System& auxiliary, const CudaSpinBlockedSource& src,
    const DeviceFactorGrads& bar_b, std::size_t budget_bytes);

// df_ucc_v2_ccsd_cuda.cu
CudaCcsdSolveResult solve_df_uccsd_cuda_impl(const CudaSpinBlockedSource& src,
                                             const DeviceFourIndexCore<double>& core,
                                             const DFUccV2Options& options);

// df_ucc_v2_triples_cuda.cu
CudaTriplesResult evaluate_df_ucc_triples_cuda_impl(const CudaSpinBlockedSource& src,
                                                    const DeviceFourIndexCore<double>& core,
                                                    const DeviceUccAmplitudes<double>& amp,
                                                    const DFUccV2Options& options);

// df_ucc_v2_lambda_cuda.cu
CudaLambdaSolveResult solve_df_ulambda_cuda_impl(const CudaSpinBlockedSource& src,
                                                 const DeviceFourIndexCore<double>& core,
                                                 const DeviceUccAmplitudes<double>& amp,
                                                 const CudaTriplesResult* triples,
                                                 const DFUccV2Options& options);

}  // namespace generativeqc::cc::v2_detail::cuda_detail
