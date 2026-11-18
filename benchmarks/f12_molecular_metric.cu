// Symmetric M^(-1/2) for the molecular closed-shell F12 qualification.
// Full-rank branch only. Input and returned whitener are row-major FP64.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "posthf/cabs_cuda_support.cuh"

struct GqcF12MetricInfo {
  std::uint64_t device_bytes{}, solver_device_workspace_bytes{}, solver_host_workspace_bytes{},
      host_to_device_bytes{}, device_to_host_bytes{}, rank{};
  double smallest{}, largest{}, relative_margin{}, roundoff_guard{};
};

namespace {
namespace gpu = generativeqc::runtime;
namespace eig = generativeqc::solver::cuda;
using generativeqc::posthf::cabs::Blas;
using generativeqc::posthf::cabs::blocks;
using generativeqc::posthf::cabs::solver_check;
using generativeqc_tensor::blas_check;
using gpu::cuda_resource_check;

__global__ void scale_columns(const double* vectors, const double* eigenvalues, double* scaled,
                              std::size_t n) {
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n * n;
       k += std::size_t(blockDim.x) * gridDim.x) {
    scaled[k] = vectors[k] / sqrt(eigenvalues[k / n]);
  }
}

__global__ void symmetrize(double* data, std::size_t n) {
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n * n;
       k += std::size_t(blockDim.x) * gridDim.x) {
    const auto i = k % n, j = k / n;
    if (i < j) {
      const auto t = i * n + j;
      data[k] = data[t] = 0.5 * data[k] + 0.5 * data[t];
    }
  }
}

void whiten(int device, int n, const double* metric, double threshold, std::uint64_t budget,
            double* output, GqcF12MetricInfo& report) {
  if (n <= 0 || device < 0 || !metric || !output || !std::isfinite(threshold) ||
      !(0. < threshold && threshold < 1.))
    throw std::invalid_argument("invalid metric dimensions or rank threshold");
  const auto nn = gpu::size_mul(n, n, "metric element count overflow");
  const auto bytes = gpu::size_mul(nn, sizeof(double), "metric matrix byte count overflow");
  const auto eigenbytes = std::size_t(n) * sizeof(double);
  const auto storage = gpu::size_add(gpu::size_mul(3, bytes, "metric arena overflow"),
                                     eigenbytes + sizeof(int), "metric arena overflow");
  if (storage > budget) throw std::length_error("device budget too small for metric arrays");
  for (std::size_t i = 0; i < std::size_t(n); ++i)
    for (std::size_t j = 0; j <= i; ++j) {
      const double a = metric[i * n + j], b = metric[j * n + i];
      if (!std::isfinite(a) || !std::isfinite(b) ||
          std::abs(a - b) > 1e-12 * std::max({1., std::abs(a), std::abs(b)}))
        throw std::invalid_argument("metric must be finite and symmetric");
    }

  gpu::CudaDeviceScope scope(device);
  gpu::OwnedCudaStream stream(device);
  eig::PreparedSymmetricEigenHandles solver;
  solver_check(solver.create());
  solver_check(solver.bind_stream(stream.get()));
  solver_check(solver.create_parameters());
  auto resources = solver.view();
  gpu::OwnedCudaBuffer<double> arena(device, 3 * nn + std::size_t(n), stream.get());
  gpu::OwnedCudaBuffer<int> dev_info(device, 1, stream.get());
  double* vectors = arena.get();
  double* scaled = vectors + nn;
  double* result = scaled + nn;
  double* values = result + nn;
  constexpr auto family = eig::SymmetricEigenFamily::xsyevd;
  eig::SymmetricEigenWorkspace sizes{};
  solver_check(eig::query_symmetric_eigen(resources, family, {n, 1}, vectors, values, sizes));
  report.device_bytes =
      gpu::size_add(storage, sizes.device_bytes, "eigensolver workspace overflow");
  report.solver_device_workspace_bytes = sizes.device_bytes;
  report.solver_host_workspace_bytes = sizes.host_bytes;
  if (report.device_bytes > budget)
    throw std::length_error("device budget too small for eigensolver workspace");
  gpu::OwnedCudaBuffer<unsigned char> workspace;
  if (sizes.device_bytes) workspace.allocate(device, sizes.device_bytes, stream.get());
  std::vector<unsigned char> host_workspace(sizes.host_bytes);
  resources.device_workspace = workspace.get();
  resources.device_workspace_bytes = sizes.device_bytes;
  resources.host_workspace = host_workspace.data();
  resources.host_workspace_bytes = sizes.host_bytes;
  cuda_resource_check(
      cudaMemcpyAsync(vectors, metric, bytes, cudaMemcpyHostToDevice, stream.get()));
  report.host_to_device_bytes = bytes;
  // clang-format off
  symmetrize<<<blocks(nn), 256, 0, stream.get()>>>(vectors, n);
  // clang-format on
  cuda_resource_check(cudaGetLastError());
  solver_check(
      eig::launch_symmetric_eigen(resources, family, {n, 1}, vectors, values, dev_info.get(), 0));
  std::vector<double> spectrum(n);
  int info{};
  cuda_resource_check(
      cudaMemcpyAsync(&info, dev_info.get(), sizeof(int), cudaMemcpyDeviceToHost, stream.get()));
  cuda_resource_check(
      cudaMemcpyAsync(spectrum.data(), values, eigenbytes, cudaMemcpyDeviceToHost, stream.get()));
  stream.synchronize();
  report.device_to_host_bytes = eigenbytes + sizeof(int);
  if (info ||
      !std::all_of(spectrum.begin(), spectrum.end(), [](double x) { return std::isfinite(x); }))
    throw std::runtime_error("cuSOLVER failed on the RI metric");
  report.smallest = spectrum.front();
  report.largest = spectrum.back();
  if (report.largest <= 0.) throw std::domain_error("RI metric has no positive spectrum");
  const auto cutoff = threshold * report.largest;
  report.rank = spectrum.end() - std::upper_bound(spectrum.begin(), spectrum.end(), cutoff);
  report.relative_margin = report.smallest / report.largest - threshold;
  // Reject a branch whose measured gap to the relative rank cutoff is within
  // the FP64 eigensolver's normwise roundoff scale.
  report.roundoff_guard = 8. * n * std::numeric_limits<double>::epsilon();
  if (report.rank != std::uint64_t(n) || report.relative_margin <= report.roundoff_guard)
    throw std::domain_error("RI metric full-rank branch lost or unresolved near cutoff");
  Blas blas;
  blas_check(cublasSetStream(blas.handle, stream.get()));
  // clang-format off
  scale_columns<<<blocks(nn), 256, 0, stream.get()>>>(vectors, values, scaled, n);
  // clang-format on
  cuda_resource_check(cudaGetLastError());
  const double one = 1., zero = 0.;
  blas_check(generativeqc_tensor::submit_column_gemm(
      {blas.handle, false, true, n, n, n, &one, scaled, n, vectors, n, &zero, result, n}));
  // clang-format off
  symmetrize<<<blocks(nn), 256, 0, stream.get()>>>(result, n);
  // clang-format on
  cuda_resource_check(cudaGetLastError());
  std::vector<double> staged(nn);
  cuda_resource_check(
      cudaMemcpyAsync(staged.data(), result, bytes, cudaMemcpyDeviceToHost, stream.get()));
  stream.synchronize();
  report.device_to_host_bytes += bytes;
  if (!std::all_of(staged.begin(), staged.end(), [](double x) { return std::isfinite(x); }))
    throw std::runtime_error("nonfinite RI metric whitener");
  std::copy(staged.begin(), staged.end(), output);
}
}  // namespace

extern "C" int gqc_f12_molecular_metric_whitener(int device, int n, const double* metric,
                                                 double relative_threshold, std::uint64_t budget,
                                                 double* output, GqcF12MetricInfo* report,
                                                 char* error, std::size_t error_size) noexcept {
  try {
    if (!report) throw std::invalid_argument("missing metric diagnostic record");
    *report = {};
    whiten(device, n, metric, relative_threshold, budget, output, *report);
    return 0;
  } catch (const std::exception& ex) {
    generativeqc_tensor::error_text(error, error_size, ex.what());
  } catch (...) {
    generativeqc_tensor::error_text(error, error_size, "unknown molecular metric CUDA failure");
  }
  return 1;
}
