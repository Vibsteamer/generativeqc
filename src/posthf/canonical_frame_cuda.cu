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

// Private real-FP64 component ABI. The full physical reference and canonical
// residual are qualified by the caller before applying this response seed.
int canonical_seeds(int device, int dimension, int occupied, double gap_threshold,
                    double stationarity_threshold, std::size_t maximum_bytes,
                    const double* energies, const double* rotation_gradient, double* fock_seed,
                    double* overlap_seed, double* unresolved_torque, std::size_t* device_bytes,
                    char* error, std::size_t error_size) {
  namespace gpu = generativeqc::runtime;
  try {
    const bool full = overlap_seed != nullptr;
    if (device < 0 || dimension < 2 || occupied < (full ? 0 : 1) || occupied >= dimension ||
        (full != (unresolved_torque != nullptr)) || !std::isfinite(gap_threshold) ||
        gap_threshold <= 0 || !std::isfinite(stationarity_threshold) ||
        stationarity_threshold <= 0 || !energies || !rotation_gradient || !fock_seed ||
        !device_bytes)
      throw std::invalid_argument("invalid canonical-frame response arguments");
    const auto n = static_cast<std::size_t>(dimension);
    const auto nn = gpu::size_mul(n, n, "canonical frame dimension overflow");
    const auto flags_count = full ? 3 : 2;
    const auto elements = gpu::size_add(gpu::size_mul(nn, full ? 3 : 2, "canonical frame overflow"),
                                        n + (full ? 1 : 0), "canonical frame overflow");
    const auto allocation =
        gpu::size_add(gpu::size_mul(elements, sizeof(double), "canonical frame bytes overflow"),
                      flags_count * sizeof(int), "canonical frame bytes overflow");
    // Count device storage, detached staging, and borrowed caller inputs/outputs.
    const auto host =
        gpu::size_add(gpu::size_mul(nn, full ? 5 : 3, "canonical frame host overflow"),
                      n + (full ? 2 : 0), "canonical frame host overflow");
    const auto total = gpu::size_add(
        allocation, gpu::size_mul(host, sizeof(double), "canonical frame host overflow"),
        "canonical frame total overflow");
    if (total > maximum_bytes) throw std::runtime_error("canonical-frame numeric budget exceeded");
    gpu::CudaDeviceScope scope(device);
    gpu::OwnedCudaStream stream(device);
    gpu::OwnedCudaBuffer<double> arena(device, elements, stream.get());
    gpu::OwnedCudaBuffer<int> flags(device, flags_count, stream.get());
    auto* m = arena.get();
    auto* e = m + nn;
    auto* out = e + n;
    auto* overlap = full ? out + nn : nullptr;
    auto* norm2 = full ? overlap + nn : nullptr;
    if (full) gpu::cuda_resource_check(cudaMemsetAsync(norm2, 0, sizeof(double), stream.get()));
    gpu::cuda_resource_check(cudaMemcpyAsync(m, rotation_gradient, nn * sizeof(double),
                                             cudaMemcpyHostToDevice, stream.get()));
    gpu::cuda_resource_check(
        cudaMemcpyAsync(e, energies, n * sizeof(double), cudaMemcpyHostToDevice, stream.get()));
    gpu::cuda_resource_check(
        cudaMemsetAsync(flags.get(), 0, flags_count * sizeof(int), stream.get()));
    generativeqc::tensor::launch_canonical_frame_seed(n, occupied, e, m, gap_threshold,
                                                      stationarity_threshold, out, flags.get(),
                                                      stream.get(), overlap, norm2);
    gpu::cuda_resource_check(cudaGetLastError());
    int host_flags[3]{};
    double host_norm2 = 0;
    if (full)
      gpu::cuda_resource_check(cudaMemcpyAsync(&host_norm2, norm2, sizeof(double),
                                               cudaMemcpyDeviceToHost, stream.get()));
    gpu::cuda_resource_check(cudaMemcpyAsync(host_flags, flags.get(), flags_count * sizeof(int),
                                             cudaMemcpyDeviceToHost, stream.get()));
    stream.synchronize();
    if (host_flags[0]) throw std::runtime_error("nonfinite canonical-frame pair response");
    if (full && host_flags[2])
      throw std::runtime_error(
          "near-degenerate occupied/virtual frontier requires a resolved electronic root");
    if (host_flags[1] ||
        (full && (!std::isfinite(host_norm2) || std::sqrt(host_norm2) > stationarity_threshold)))
      throw std::runtime_error("nonstationary degenerate canonical same-space block");
    std::vector<double> staged(full ? 2 * nn : nn);
    StagingDrain fence{stream.get()};
    gpu::cuda_resource_check(cudaMemcpyAsync(staged.data(), out, staged.size() * sizeof(double),
                                             cudaMemcpyDeviceToHost, stream.get()));
    stream.synchronize();
    fence.stream = nullptr;
    std::copy_n(staged.data(), nn, fock_seed);
    if (full) {
      std::copy_n(staged.data() + nn, nn, overlap_seed);
      *unresolved_torque = std::sqrt(host_norm2);
    }
    *device_bytes = allocation;
    return 0;
  } catch (const std::exception& failure) {
    generativeqc_tensor::error_text(error, error_size, failure.what());
    return 1;
  }
}

}  // namespace

extern "C" int generativeqc_canonical_frame_seed_cuda_v1(
    int device, int dimension, int occupied, double gap_threshold, double stationarity_threshold,
    std::size_t maximum_bytes, const double* energies, const double* rotation_gradient,
    double* fock_seed, std::size_t* device_bytes, char* error, std::size_t error_size) {
  return canonical_seeds(device, dimension, occupied, gap_threshold, stationarity_threshold,
                         maximum_bytes, energies, rotation_gradient, fock_seed, nullptr, nullptr,
                         device_bytes, error, error_size);
}

extern "C" int generativeqc_canonical_full_frame_seeds_cuda_v1(
    int device, int dimension, int occupied, double gap_threshold, double stationarity_threshold,
    std::size_t maximum_bytes, const double* energies, const double* rotation_gradient,
    double* fock_seed, double* overlap_seed, double* unresolved_torque, std::size_t* device_bytes,
    char* error, std::size_t error_size) {
  if (!overlap_seed || !unresolved_torque) {
    generativeqc_tensor::error_text(error, error_size, "missing full canonical-frame outputs");
    return 1;
  }
  return canonical_seeds(device, dimension, occupied, gap_threshold, stationarity_threshold,
                         maximum_bytes, energies, rotation_gradient, fock_seed, overlap_seed,
                         unresolved_torque, device_bytes, error, error_size);
}
