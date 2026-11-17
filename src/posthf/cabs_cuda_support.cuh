#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>

#include "runtime/cuda_resources.cuh"
#include "solver/cuda/symmetric_eigen_handles.hpp"
#include "tensor/cuda_runtime.cuh"

namespace generativeqc::posthf::cabs {
namespace gpu = generativeqc::runtime;
using generativeqc_tensor::blas_check;

struct Blas {
  cublasHandle_t handle{};
  Blas() { blas_check(cublasCreate(&handle)); }
  ~Blas() { (void)cublasDestroy(handle); }
  Blas(const Blas&) = delete;
  Blas& operator=(const Blas&) = delete;
};

inline void solver_check(std::uint32_t status) {
  if (status) throw std::runtime_error("CABS cuSOLVER status " + std::to_string(status));
}

inline unsigned blocks(std::size_t count) {
  return static_cast<unsigned>(std::min<std::size_t>(65535, (count - 1) / 256 + 1));
}

inline void check_flags(const gpu::OwnedCudaStream& stream, const int* flags, int count,
                        const char* message) {
  if (count < 1 || count > 3) throw std::logic_error("invalid CABS flag count");
  int host[3]{};
  gpu::cuda_resource_check(cudaGetLastError());
  gpu::cuda_resource_check(
      cudaMemcpyAsync(host, flags, count * sizeof(int), cudaMemcpyDeviceToHost, stream.get()));
  stream.synchronize();
  if (std::any_of(host, host + count, [](int x) { return x != 0; }))
    throw std::runtime_error(message);
}

}  // namespace generativeqc::posthf::cabs
