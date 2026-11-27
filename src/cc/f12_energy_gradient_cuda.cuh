#pragma once

#include "cc/f12_energy_gradient.hpp"
#include "tensor/cuda_runtime.cuh"

namespace generativeqc::cc::f12_cuda {

// The ordinary generated RHS is the negative energy derivative. Subtract the
// fixed-cusp derivative into the response owner's detached result buffers.
static __global__ void subtract_singles_energy_gradient(
    std::size_t o, std::size_t v, const double* r1_const, const double* w_vv,
    const double* t1, const double* base, double* rhs, int* error) {
  for (std::size_t ia = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       ia < o * v; ia += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    rhs[ia] = generativeqc_tensor::finite(
        base[ia] - energy_gradient_one(o, v, r1_const, w_vv, t1, ia), error, 0);
}

static __global__ void subtract_doubles_energy_gradient(
    std::size_t n2, std::size_t v, const double* w_vv, const double* c_f12,
    const double* base, double* rhs, int* error) {
  for (std::size_t ijab = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       ijab < n2; ijab += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    rhs[ijab] = generativeqc_tensor::finite(
        base[ijab] - energy_gradient_two(v, w_vv, c_f12, ijab), error, 0);
}

inline void launch_subtract_energy_gradient(
    std::size_t o, std::size_t v, const double* r1_const, const double* w_vv,
    const double* c_f12, const double* t1, const double* base1, const double* base2,
    double* rhs1, double* rhs2, int* error, cudaStream_t stream) {
  const auto n1 = o * v, n2 = n1 * n1;  // Checked at response-owner construction.
  subtract_singles_energy_gradient<<<generativeqc_tensor::blocks(n1, 256), 256, 0, stream>>>(
      o, v, r1_const, w_vv, t1, base1, rhs1, error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
  subtract_doubles_energy_gradient<<<generativeqc_tensor::blocks(n2, 256), 256, 0, stream>>>(
      n2, v, w_vv, c_f12, base2, rhs2, error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
}

}  // namespace generativeqc::cc::f12_cuda
