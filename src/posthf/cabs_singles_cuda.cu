#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "posthf/cabs_singles_cuda.hpp"
#include "runtime/cuda_resources.cuh"
#include "solver/cuda/symmetric_eigen_handles.hpp"
#include "tensor/cuda_runtime.cuh"

namespace {
namespace gpu = generativeqc::runtime;
namespace eig = generativeqc::solver::cuda;
using generativeqc_tensor::blas_check;

struct Blas {
  cublasHandle_t handle{};
  Blas() { blas_check(cublasCreate(&handle)); }
  ~Blas() { (void)cublasDestroy(handle); }
  Blas(const Blas&) = delete;
  Blas& operator=(const Blas&) = delete;
};

void solver_check(std::uint32_t status) {
  if (status) throw std::runtime_error("CABS cuSOLVER status " + std::to_string(status));
}

unsigned blocks(std::size_t count) {
  return static_cast<unsigned>(std::min<std::size_t>(65535, (count - 1) / 256 + 1));
}

// Pack column-major provider blocks from the two triangles of a symmetric
// row-major Fock. Tiny last-bit AO-transform asymmetries use their mean.
__global__ void unpack(const double* f, std::size_t n, std::size_t o, double* a, double* b,
                       double* c, int* bad) {
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n * n;
       k += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t i = k / n, j = k % n, v = n - o;
    const double x = f[k], y = f[j * n + i];
    if (!isfinite(x) || fabs(x - y) > 1e-9 * fmax(1., fmax(fabs(x), fabs(y)))) atomicExch(bad, 1);
    const double value = .5 * x + .5 * y;
    if (i < o && j < o)
      a[i + j * o] = value;
    else if (i >= o && j >= o)
      b[(i - o) + (j - o) * v] = value;
    else if (i < o)
      c[i + (j - o) * o] = value;
  }
}

__global__ void solve_diagonal(double* t, const double* eo, const double* ev, std::size_t o,
                               std::size_t count, double min_gap, int* bad) {
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < count;
       k += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const double gap = eo[k % o] - ev[k / o];
    if (!isfinite(gap) || fabs(gap) <= min_gap)
      atomicExch(bad, 1);
    else
      t[k] /= gap;
  }
}

__global__ void write_pullback(double* f, std::size_t n, std::size_t o, const double* oo,
                               const double* vv, const double* t, double weight, int* bad) {
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n * n;
       k += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t i = k / n, j = k % n;
    double value;
    if (i < o && j < o)
      value = oo[i + o * j];
    else if (i >= o && j >= o)
      value = vv[i - o + (n - o) * (j - o)];
    else if (i < o)
      value = weight * t[i + o * (j - o)];
    else
      value = weight * t[j + o * (i - o)];
    f[k] = value;
    if (!isfinite(value)) atomicExch(bad, 1);
  }
}

void check_flags(const gpu::OwnedCudaStream& stream, const int* flags, int count,
                 const char* message) {
  int host[3]{};
  gpu::cuda_resource_check(cudaGetLastError());
  gpu::cuda_resource_check(
      cudaMemcpyAsync(host, flags, count * sizeof(int), cudaMemcpyDeviceToHost, stream.get()));
  stream.synchronize();
  if (std::any_of(host, host + count, [](int x) { return x != 0; }))
    throw std::runtime_error(message);
}

void cabs_singles(int device, int n, int o, int weight, double min_gap, const double* fock,
                  double* energy, double* fock_bar) {
  if (device < 0 || n < 1 || o < 0 || o > n || (weight != 1 && weight != 2) ||
      !std::isfinite(min_gap) || min_gap <= 0 || !fock || !energy)
    throw std::invalid_argument("invalid CABS dimension, spin weight, gap, or host buffer");
  const int v = n - o;
  const std::size_t square = gpu::size_mul(n, n, "CABS union dimension overflow");
  const std::size_t bytes = gpu::size_mul(square, sizeof(double), "CABS union size overflow");
  gpu::CudaDeviceScope scope(device);
  if (!o || !v) {
    for (std::size_t k = 0; k < square; ++k) {
      const double x = fock[k], y = fock[(k % n) * n + k / n];
      if (!std::isfinite(x) || std::abs(x - y) > 1e-9 * std::max({1., std::abs(x), std::abs(y)}))
        throw std::invalid_argument("invalid empty-channel Fock matrix");
    }
    if (fock_bar) std::fill_n(fock_bar, square, 0.);
    *energy = 0.;
    return;
  }
  const std::size_t oo = std::size_t(o) * o, vv = std::size_t(v) * v, ov = std::size_t(o) * v;
  gpu::OwnedCudaStream stream(device);
  Blas blas;
  blas_check(cublasSetStream(blas.handle, stream.get()));
  eig::PreparedSymmetricEigenHandles handles;
  solver_check(handles.create());
  solver_check(handles.bind_stream(stream.get()));
  solver_check(handles.create_parameters());
  // A single arena supplies the input/output, both eigenspaces, three
  // ov-sized tiles and both spectra. oo + vv + 2*ov == square.
  auto count =
      gpu::size_add(gpu::size_mul(square, 2, "CABS arena overflow"), ov, "CABS arena overflow");
  count = gpu::size_add(count, n, "CABS arena overflow");
  gpu::OwnedCudaBuffer<double> arena(device, count, stream.get());
  gpu::OwnedCudaBuffer<int> flags(device, 3, stream.get());
  double* f = arena.get();
  double* a = f + square;
  double* b = a + oo;
  double* c = b + vv;
  double* tmp = c + ov;
  double* t = tmp + ov;
  double* eo = t + ov;
  double* ev = eo + o;
  gpu::cuda_resource_check(cudaMemcpyAsync(f, fock, bytes, cudaMemcpyHostToDevice, stream.get()));
  gpu::cuda_resource_check(cudaMemsetAsync(flags.get(), 0, 3 * sizeof(int), stream.get()));
  unpack<<<blocks(square), 256, 0, stream.get()>>>(f, n, o, a, b, c, flags.get());
  check_flags(stream, flags.get(), 1, "nonfinite or asymmetric CABS Fock matrix");

  constexpr auto family = eig::SymmetricEigenFamily::xsyevd;
  const eig::SymmetricEigenProblem occ{o, 1}, virt{v, 1};
  auto resources = handles.view();
  eig::SymmetricEigenWorkspace wo{}, wv{};
  solver_check(eig::query_symmetric_eigen(resources, family, occ, a, eo, wo));
  solver_check(eig::query_symmetric_eigen(resources, family, virt, b, ev, wv));
  const auto device_bytes = std::max(wo.device_bytes, wv.device_bytes);
  gpu::OwnedCudaBuffer<unsigned char> work;
  if (device_bytes) work.allocate(device, device_bytes, stream.get());
  std::vector<unsigned char> host(std::max(wo.host_bytes, wv.host_bytes));
  resources.device_workspace = work.get();
  resources.device_workspace_bytes = device_bytes;
  resources.host_workspace = host.data();
  resources.host_workspace_bytes = host.size();
  solver_check(eig::launch_symmetric_eigen(resources, family, occ, a, eo, flags.get() + 1, 0));
  solver_check(eig::launch_symmetric_eigen(resources, family, virt, b, ev, flags.get() + 2, 0));
  check_flags(stream, flags.get() + 1, 2, "CABS occupied/virtual eigensolver failed");

  const double zero = 0.;
  auto gemm = [&](bool tl, bool tr, int rows, int cols, int inner, const double* l, const double* r,
                  double* dest, double scale = 1.) {
    blas_check(generativeqc_tensor::submit_column_gemm({blas.handle, tl, tr, rows, cols, inner,
                                                        &scale, l, tl ? inner : rows, r,
                                                        tr ? cols : inner, &zero, dest, rows}));
  };
  gemm(true, false, o, v, o, a, c, tmp);
  gemm(false, false, o, v, v, tmp, b, t);
  solve_diagonal<<<blocks(ov), 256, 0, stream.get()>>>(t, eo, ev, o, ov, min_gap, flags.get());
  check_flags(stream, flags.get(), 1, "CABS occupied/virtual denominator below threshold");
  gemm(false, false, o, v, o, a, t, tmp);
  gemm(false, true, o, v, v, tmp, b, t);
  // A*T - T*B = C. Only the occupied/virtual gap enters the derivative.
  double result = 0.;
  for (std::size_t i = 0; i < ov;) {
    const auto length = static_cast<int>(std::min<std::size_t>(ov - i, INT_MAX));
    double dot = 0.;
    blas_check(cublasDdot(blas.handle, length, c + i, 1, t + i, 1, &dot));
    result += weight * dot;
    i += length;
  }
  if (!std::isfinite(result)) throw std::runtime_error("nonfinite CABS singles energy");
  if (fock_bar) {
    gemm(false, true, o, o, v, t, t, a, -weight);
    gemm(true, false, v, v, o, t, t, b, weight);
    write_pullback<<<blocks(square), 256, 0, stream.get()>>>(f, n, o, a, b, t, weight, flags.get());
    check_flags(stream, flags.get(), 1, "nonfinite CABS singles Fock response");
    std::vector<double> staged(square);
    gpu::cuda_resource_check(
        cudaMemcpyAsync(staged.data(), f, bytes, cudaMemcpyDeviceToHost, stream.get()));
    stream.synchronize();
    std::copy(staged.begin(), staged.end(), fock_bar);
  } else {
    stream.synchronize();
  }
  *energy = result;
}
}  // namespace

extern "C" int generativeqc_f12_cabs_singles_cuda_v1(int device, int dimension, int occupied,
                                                     int spin_weight, double min_gap,
                                                     const double* fock, double* energy,
                                                     double* fock_bar, char* error,
                                                     std::size_t error_size) {
  try {
    cabs_singles(device, dimension, occupied, spin_weight, min_gap, fock, energy, fock_bar);
    return 0;
  } catch (const std::exception& ex) {
    generativeqc_tensor::error_text(error, error_size, ex.what());
  } catch (...) {
    generativeqc_tensor::error_text(error, error_size, "unknown CABS CUDA failure");
  }
  return 1;
}
