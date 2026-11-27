#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

#include "tensor/cuda_error.hpp"
#include "tensor/cuda_runtime.cuh"

namespace generativeqc::cc::f12_cuda {

static __global__ void singles_constant_kernel(std::size_t o, std::size_t v,
                                               const double* __restrict__ w_vo,
                                               double* __restrict__ r1_const, int* error) {
  for (std::size_t ia = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; ia < o * v;
       ia += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto i = ia / v, a = ia % v;
    double sum = 0.0;
    for (std::size_t j = 0; j < o; ++j) {
      const double direct = w_vo[((i * o + j) * v + a) * o + j];
      const double exchange = w_vo[((j * o + i) * v + a) * o + j];
      sum = __dadd_rn(sum, __dsub_rn(__dmul_rn(2.0, direct), exchange));
    }
    r1_const[ia] = generativeqc_tensor::finite(sum, error, 0);
  }
}

static __global__ void symmetrize_pairs_kernel(std::size_t o, std::size_t v,
                                              double* __restrict__ w_vv,
                                              double* __restrict__ c_f12, int* error) {
  const auto n2 = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n2;
       idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto ij = idx / (v * v);
    const auto mate = (((ij % o) * o + ij / o) * v + idx % v) * v + (idx / v) % v;
    if (idx > mate) continue;
    // Only this lane reads and writes both members of the simultaneous-pair orbit.
    const double w = idx == mate ? w_vv[idx] : __dmul_rn(0.5, __dadd_rn(w_vv[idx], w_vv[mate]));
    const double c = idx == mate ? c_f12[idx] : __dmul_rn(0.5, __dadd_rn(c_f12[idx], c_f12[mate]));
    w_vv[idx] = w_vv[mate] = generativeqc_tensor::finite(w, error, 0);
    c_f12[idx] = c_f12[mate] = generativeqc_tensor::finite(c, error, 0);
  }
}

inline void launch_prepare_constants(std::size_t o, std::size_t v, const double* w_vo,
                                     double* w_vv, double* c_f12, double* r1_const,
                                     int* error, cudaStream_t stream) {
  if (!o || !v) return;
  singles_constant_kernel<<<generativeqc_tensor::blocks(o * v, 256), 256, 0, stream>>>(
      o, v, w_vo, r1_const, error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
  symmetrize_pairs_kernel<<<generativeqc_tensor::blocks(o * o * v * v, 256), 256, 0, stream>>>(
      o, v, w_vv, c_f12, error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
}

static __global__ void add_singles_residual_kernel(std::size_t n1,
                                                   const double* __restrict__ r1_const,
                                                   double* __restrict__ r1_acc, int* error) {
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n1;
       idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    r1_acc[idx] = generativeqc_tensor::finite(__dadd_rn(r1_acc[idx], r1_const[idx]), error, 0);
}

// Preserves bit-exact simultaneous pair symmetry (i,j,a,b) <-> (j,i,b,a) when
// w_vv and c_f12 are pair-symmetric, with zero temporary device memory.
static __global__ void add_doubles_residual_kernel(std::size_t o, std::size_t v,
                                                   const double* __restrict__ w_vo,
                                                   const double* __restrict__ w_vv,
                                                   const double* __restrict__ c_f12,
                                                   const double* __restrict__ t1,
                                                   double* __restrict__ r2_acc, int* error) {
  const std::size_t vv = v * v;
  const std::size_t n2 = o * o * vv;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n2;
       idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t b = idx % v;
    const std::size_t tmp0 = idx / v;
    const std::size_t a = tmp0 % v;
    const std::size_t ij = tmp0 / v;
    const std::size_t j = ij % o;
    const std::size_t i = ij / o;
    const double* w1 = w_vo + ((i * o + j) * v + a) * o;
    const double* w2 = w_vo + ((j * o + i) * v + b) * o;
    double p = 0.0;
    for (std::size_t k = 0; k < o; ++k)
      p = __dadd_rn(p, __dadd_rn(__dmul_rn(w1[k], t1[k * v + b]), __dmul_rn(w2[k], t1[k * v + a])));
    const double delta = __dsub_rn(__dadd_rn(w_vv[idx], c_f12[idx]), p);
    r2_acc[idx] = generativeqc_tensor::finite(__dadd_rn(r2_acc[idx], delta), error, 0);
  }
}

static __global__ void energy_partials_kernel(std::size_t o, std::size_t v,
                                              const double* __restrict__ r1_const,
                                              const double* __restrict__ w_vv,
                                              const double* __restrict__ c_f12,
                                              const double* __restrict__ t1,
                                              const double* __restrict__ t2,
                                              double* __restrict__ partials) {
  __shared__ double sdata[256];
  const std::size_t n1 = o * v;
  const std::size_t vv = v * v;
  const std::size_t n2 = o * o * vv;
  double acc = 0.0;
  if (blockIdx.x == 0) {
    for (std::size_t ia = threadIdx.x; ia < n1; ia += blockDim.x)
      acc += 2.0 * r1_const[ia] * t1[ia];
  }
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n2;
       idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t b = idx % v;
    const std::size_t tmp0 = idx / v;
    const std::size_t a = tmp0 % v;
    const std::size_t ij = tmp0 / v;
    const std::size_t j = ij % o;
    const std::size_t i = ij / o;
    const std::size_t swap = (ij * v + b) * v + a;
    const double tc = 2.0 * c_f12[idx] - c_f12[swap];
    const double tw = 2.0 * w_vv[idx] - w_vv[swap];
    const double t2_val = t2[idx];
    acc += tc * t2_val + tw * (t2_val + t1[i * v + a] * t1[j * v + b]);
  }
  sdata[threadIdx.x] = acc;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = sdata[0];
}

static __global__ void energy_finish_kernel(const double* __restrict__ partials, int nblocks,
                                            double fixed_geminal_energy,
                                            const double* __restrict__ ord_energy,
                                            double* __restrict__ out_energy, int* error) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  double acc = fixed_geminal_energy + *ord_energy;
  for (int i = 0; i < nblocks; ++i) acc += partials[i];
  *out_energy = generativeqc_tensor::finite(acc, error, 0);
}

// Tiled 16x16 shared-memory matrix product:
//   sum1[k, c] -= 2.0 * sum_{p=0}^{o*o*v-1} w_vo[p * o + k] * lambda2[p * v + c]
static __global__ void lambda_transpose_kernel(std::size_t o, std::size_t v,
                                               const double* __restrict__ w_vo,
                                               const double* __restrict__ lambda2,
                                               double* __restrict__ sum1, int* error) {
  __shared__ double sw[16][17];
  __shared__ double sl[16][17];
  const std::size_t c0 = static_cast<std::size_t>(blockIdx.x) * 16;
  const std::size_t k0 = static_cast<std::size_t>(blockIdx.y) * 16;
  const std::size_t tx = threadIdx.x;
  const std::size_t ty = threadIdx.y;
  const std::size_t c = c0 + tx;
  const std::size_t k = k0 + ty;
  const std::size_t m = o * o * v;
  double acc = 0.0;
  for (std::size_t p0 = 0; p0 < m; p0 += 16) {
    const std::size_t p = p0 + ty;
    sw[ty][tx] = (p < m && k0 + tx < o) ? w_vo[p * o + (k0 + tx)] : 0.0;
    sl[ty][tx] = (p < m && c < v) ? lambda2[p * v + c] : 0.0;
    __syncthreads();
    for (int t = 0; t < 16; ++t) acc += sw[t][ty] * sl[t][tx];
    __syncthreads();
  }
  if (k < o && c < v)
    sum1[k * v + c] =
        generativeqc_tensor::finite(__dsub_rn(sum1[k * v + c], __dmul_rn(2.0, acc)), error, 1);
}

inline std::size_t energy_partial_blocks(std::size_t n2) {
  const auto needed = (n2 + 255) / 256;
  return std::max<std::size_t>(1, std::min<std::size_t>(needed, 1024));
}

inline void launch_add_residuals(std::size_t o, std::size_t v, const double* r1_const,
                                 const double* w_vo, const double* w_vv, const double* c_f12,
                                 const double* t1, double* r1_acc, double* r2_acc, int* error,
                                 cudaStream_t stream) {
  const auto n1 = o * v;
  const auto n2 = o * o * v * v;
  add_singles_residual_kernel<<<generativeqc_tensor::blocks(n1, 256), 256, 0, stream>>>(
      n1, r1_const, r1_acc, error);
  add_doubles_residual_kernel<<<generativeqc_tensor::blocks(n2, 256), 256, 0, stream>>>(
      o, v, w_vo, w_vv, c_f12, t1, r2_acc, error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
}

inline void launch_add_energy(std::size_t o, std::size_t v, const double* r1_const,
                              const double* w_vv, const double* c_f12, const double* t1,
                              const double* t2, double fixed_geminal_energy,
                              const double* ord_energy, double* partials, std::size_t nblocks,
                              double* out_energy, int* error, cudaStream_t stream) {
  energy_partials_kernel<<<static_cast<unsigned>(nblocks), 256, 0, stream>>>(
      o, v, r1_const, w_vv, c_f12, t1, t2, partials);
  energy_finish_kernel<<<1, 1, 0, stream>>>(partials, static_cast<int>(nblocks),
                                            fixed_geminal_energy, ord_energy, out_energy, error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
}

inline void launch_lambda_transpose(std::size_t o, std::size_t v, const double* w_vo,
                                    const double* lambda2, double* sum1, int* error,
                                    cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid(static_cast<unsigned>((v + 15) / 16), static_cast<unsigned>((o + 15) / 16));
  lambda_transpose_kernel<<<grid, block, 0, stream>>>(o, v, w_vo, lambda2, sum1, error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
}

}  // namespace generativeqc::cc::f12_cuda
