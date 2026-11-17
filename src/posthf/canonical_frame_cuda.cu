#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "generated_canonical_frame.cuh"
#include "posthf/canonical_frame_cuda.hpp"
#include "runtime/cuda_resources.cuh"
#include "tensor/cuda_runtime.cuh"

namespace {
struct StagingDrain {
  cudaStream_t stream;
  ~StagingDrain() {
    if (stream) (void)cudaStreamSynchronize(stream);
  }
};
}  // namespace

// Private real-FP64 component ABI. The full physical reference and canonical
// residual are qualified by the caller before applying this response seed.
extern "C" int generativeqc_canonical_frame_seed_cuda_v1(
    int device, int dimension, int occupied, double gap_threshold, double stationarity_threshold,
    std::size_t maximum_bytes, const double* energies, const double* rotation_gradient,
    double* fock_seed, std::size_t* device_bytes, char* error, std::size_t error_size) {
  namespace gpu = generativeqc::runtime;
  try {
    if (device < 0 || dimension < 2 || occupied < 1 || occupied >= dimension ||
        !std::isfinite(gap_threshold) || gap_threshold <= 0 ||
        !std::isfinite(stationarity_threshold) || stationarity_threshold <= 0 || !energies ||
        !rotation_gradient || !fock_seed || !device_bytes)
      throw std::invalid_argument("invalid canonical-frame response arguments");
    const auto n = static_cast<std::size_t>(dimension);
    const auto nn = gpu::size_mul(n, n, "canonical frame dimension overflow");
    const auto elements = gpu::size_add(gpu::size_mul(nn, 2, "canonical frame overflow"), n,
                                        "canonical frame overflow");
    const auto allocation =
        gpu::size_add(gpu::size_mul(elements, sizeof(double), "canonical frame bytes overflow"),
                      2 * sizeof(int), "canonical frame bytes overflow");
    // Count device storage, detached staging, and borrowed caller inputs/outputs.
    const auto host = gpu::size_add(gpu::size_mul(nn, 3, "canonical frame host overflow"), n,
                                    "canonical frame host overflow");
    const auto total = gpu::size_add(
        allocation, gpu::size_mul(host, sizeof(double), "canonical frame host overflow"),
        "canonical frame total overflow");
    if (total > maximum_bytes) throw std::runtime_error("canonical-frame numeric budget exceeded");
    gpu::CudaDeviceScope scope(device);
    gpu::OwnedCudaStream stream(device);
    gpu::OwnedCudaBuffer<double> arena(device, elements, stream.get());
    gpu::OwnedCudaBuffer<int> flags(device, 2, stream.get());
    auto* m = arena.get();
    auto* e = m + nn;
    auto* out = e + n;
    gpu::cuda_resource_check(cudaMemcpyAsync(m, rotation_gradient, nn * sizeof(double),
                                             cudaMemcpyHostToDevice, stream.get()));
    gpu::cuda_resource_check(
        cudaMemcpyAsync(e, energies, n * sizeof(double), cudaMemcpyHostToDevice, stream.get()));
    gpu::cuda_resource_check(cudaMemsetAsync(flags.get(), 0, 2 * sizeof(int), stream.get()));
    generativeqc::tensor::launch_canonical_frame_seed(
        n, occupied, e, m, gap_threshold, stationarity_threshold, out, flags.get(), stream.get());
    gpu::cuda_resource_check(cudaGetLastError());
    int host_flags[2]{};
    gpu::cuda_resource_check(cudaMemcpyAsync(host_flags, flags.get(), sizeof(host_flags),
                                             cudaMemcpyDeviceToHost, stream.get()));
    stream.synchronize();
    if (host_flags[0]) throw std::runtime_error("nonfinite canonical-frame pair response");
    if (host_flags[1])
      throw std::runtime_error("nonstationary degenerate canonical same-space block");
    std::vector<double> staged(nn);
    StagingDrain fence{stream.get()};
    gpu::cuda_resource_check(cudaMemcpyAsync(staged.data(), out, nn * sizeof(double),
                                             cudaMemcpyDeviceToHost, stream.get()));
    stream.synchronize();
    fence.stream = nullptr;
    std::copy(staged.begin(), staged.end(), fock_seed);
    *device_bytes = allocation;
    return 0;
  } catch (const std::exception& failure) {
    generativeqc_tensor::error_text(error, error_size, failure.what());
    return 1;
  }
}
