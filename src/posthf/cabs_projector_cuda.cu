#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "generated_symmetric_matrix_function.cuh"
#include "posthf/cabs_cuda_support.cuh"
#include "posthf/cabs_projector_cuda.hpp"

namespace {
namespace gpu = generativeqc::runtime;
namespace eig = generativeqc::solver::cuda;
using generativeqc::posthf::cabs::Blas;
using generativeqc::posthf::cabs::blocks;
using generativeqc::posthf::cabs::check_flags;
using generativeqc::posthf::cabs::solver_check;
using generativeqc_tensor::blas_check;
using gpu::cuda_resource_check;

__global__ void symmetrize(double* a, std::size_t n, bool require_symmetric, int* bad) {
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n * n;
       k += std::size_t(blockDim.x) * gridDim.x) {
    const auto i = k % n, j = k / n;
    if (i > j) continue;
    const auto t = j + i * n;
    const double x = a[k], y = a[t];
    if (!isfinite(x) || !isfinite(y) ||
        (require_symmetric && fabs(x - y) > 1e-12 * fmax(1., fmax(fabs(x), fabs(y)))))
      atomicExch(bad, 1);
    a[k] = a[t] = .5 * x + .5 * y;
  }
}

__global__ void scale_inverse_columns(const double* vectors, const double* values, std::size_t n,
                                      double cutoff, double* scaled) {
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n * n;
       k += std::size_t(blockDim.x) * gridDim.x) {
    const double v = values[k / n];
    scaled[k] = v > cutoff ? vectors[k] / v : 0.;
  }
}

__global__ void identity_tail(double* lift, std::size_t n, std::size_t o) {
  const auto x = n - o;
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < x * x;
       k += std::size_t(blockDim.x) * gridDim.x)
    lift[o + k % x + n * (k / x)] = k % x == k / x;
}

__global__ void normalize_retained_columns(const double* vectors, const double* values,
                                           std::size_t rows, std::size_t rank, double* scaled) {
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < rows * rank;
       k += std::size_t(blockDim.x) * gridDim.x)
    scaled[k] = vectors[k] / sqrt(values[k / rows]);
}

__global__ void pack_frame(const double* column_major, std::size_t rows, std::size_t cols,
                           double* row_major, int* bad) {
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < rows * cols;
       k += std::size_t(blockDim.x) * gridDim.x) {
    const double value = column_major[k / cols + rows * (k % cols)];
    if (!isfinite(value)) atomicExch(bad, 1);
    row_major[k] = value;
  }
}

__global__ void assemble_response(const double* from_metric, const double* from_lift, std::size_t n,
                                  std::size_t o, double* out) {
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n * n;
       k += std::size_t(blockDim.x) * gridDim.x) {
    const auto i = k % n, j = k / n;
    out[k] = .5 * from_metric[k] + .5 * from_metric[j + i * n] -
             (i < o ? from_lift[i + o * j] : 0.) - (j < o ? from_lift[j + o * i] : 0.);
  }
}

void evaluate(int device, int n, int o, double cutoff, double guard, int expected,
              const double* overlap, const double* seed, double* kernel, double* frame,
              std::size_t frame_capacity, double* bar, int* rank, double* margin) {
  if (device < 0 || n <= 1 || o <= 0 || o >= n || !(0. < guard && guard < cutoff && cutoff < 1.) ||
      expected < -1 || expected > n - o || !overlap || !kernel || !frame || !rank || !margin ||
      (seed == nullptr) != (bar == nullptr))
    throw std::invalid_argument("invalid CABS projector dimensions, cutoff, rank or buffers");
  const int x = n - o;
  const auto nn = gpu::size_mul(n, n, "CABS overlap size overflow");
  const auto oo = std::size_t(o) * o, xx = std::size_t(x) * x, nx = std::size_t(n) * x;
  if (frame_capacity < nx) throw std::invalid_argument("insufficient CABS frame capacity");
  if (seed && xx > 256ull * std::numeric_limits<int>::max())
    throw std::invalid_argument("CABS spectral-response CUDA grid is too large");
  gpu::CudaDeviceScope device_scope(device);
  gpu::OwnedCudaStream stream(device);
  Blas blas;
  blas_check(cublasSetStream(blas.handle, stream.get()));
  eig::PreparedSymmetricEigenHandles solver;
  solver_check(solver.create());
  solver_check(solver.bind_stream(stream.get()));
  solver_check(solver.create_parameters());
  auto resources = solver.view();
  // S (later seed), K, two contraction tiles, eigensystems and the lift.
  std::size_t count = gpu::size_mul(4, nn, "CABS projector arena overflow");
  for (auto size : {oo, oo, xx, xx, nx, std::size_t(n)})
    count = gpu::size_add(count, size, "CABS projector arena overflow");
  gpu::OwnedCudaBuffer<double> arena(device, count, stream.get());
  gpu::OwnedCudaBuffer<int> info(device, 2, stream.get());
  double* s = arena.get();
  double* k = s + nn;
  double* tmp = k + nn;
  double* tmp2 = tmp + nn;
  double* a = tmp2 + nn;
  double* inv_a = a + oo;
  double* t = inv_a + oo;
  double* inv_t = t + xx;
  double* j = inv_t + xx;
  double* ea = j + nx;
  double* et = ea + o;
  const auto bytes = gpu::size_mul(nn, sizeof(double), "CABS overlap bytes overflow");
  cuda_resource_check(cudaMemsetAsync(info.get(), 0, 2 * sizeof(int), stream.get()));
  auto load = [&](const double* src, bool symmetric) {
    cuda_resource_check(cudaMemcpyAsync(s, src, bytes, cudaMemcpyHostToDevice, stream.get()));
    symmetrize<<<blocks(nn), 256, 0, stream.get()>>>(s, n, symmetric, info.get());
    check_flags(stream, info.get(), 1, "nonfinite input or asymmetric CABS overlap");
  };
  load(overlap, true);
  auto block_copy = [&](double* dest, const double* src, int d) {
    cuda_resource_check(cudaMemcpy2DAsync(
        dest, std::size_t(d) * sizeof(double), src, std::size_t(n) * sizeof(double),
        std::size_t(d) * sizeof(double), d, cudaMemcpyDeviceToDevice, stream.get()));
  };
  block_copy(a, s, o);
  block_copy(t, s + o + std::size_t(o) * n, x);
  constexpr auto family = eig::SymmetricEigenFamily::xsyevd;
  eig::SymmetricEigenWorkspace wa{}, wt{};
  solver_check(eig::query_symmetric_eigen(resources, family, {o, 1}, a, ea, wa));
  solver_check(eig::query_symmetric_eigen(resources, family, {x, 1}, t, et, wt));
  const auto work_bytes = std::max(wa.device_bytes, wt.device_bytes);
  gpu::OwnedCudaBuffer<unsigned char> workspace;
  if (work_bytes) workspace.allocate(device, work_bytes, stream.get());
  std::vector<unsigned char> host_workspace(std::max(wa.host_bytes, wt.host_bytes));
  resources.device_workspace = workspace.get();
  resources.device_workspace_bytes = work_bytes;
  resources.host_workspace = host_workspace.data();
  resources.host_workspace_bytes = host_workspace.size();
  auto eigen = [&](int d, double* matrix, double* values) {
    solver_check(
        eig::launch_symmetric_eigen(resources, family, {d, 1}, matrix, values, info.get() + 1, 0));
    std::vector<double> spectrum(d);
    cuda_resource_check(cudaMemcpyAsync(spectrum.data(), values, d * sizeof(double),
                                        cudaMemcpyDeviceToHost, stream.get()));
    check_flags(stream, info.get() + 1, 1, "CABS overlap eigensolver failed");
    if (!std::all_of(spectrum.begin(), spectrum.end(), [](double v) { return std::isfinite(v); }))
      throw std::runtime_error("nonfinite CABS overlap spectrum");
    return spectrum;
  };
  auto gemm = [&](bool tl, bool tr, int m, int cols, int inner, const double* l, int ld_l,
                  const double* r, int ld_r, double* out, int ld_out, double alpha = 1.,
                  double beta = 0.) {
    blas_check(generativeqc_tensor::submit_column_gemm(
        {blas.handle, tl, tr, m, cols, inner, &alpha, l, ld_l, r, ld_r, &beta, out, ld_out}));
  };
  auto invert = [&](int d, const double* q, const double* values, double threshold, double* out) {
    scale_inverse_columns<<<blocks(std::size_t(d) * d), 256, 0, stream.get()>>>(q, values, d,
                                                                                threshold, tmp);
    gemm(false, true, d, d, d, tmp, d, q, d, out, d);
  };
  auto obs_spectrum = eigen(o, a, ea);
  if (obs_spectrum.back() <= 0. || obs_spectrum.front() <= guard * obs_spectrum.back())
    throw std::runtime_error("unresolved OBS overlap rank");
  invert(o, a, ea, 0., inv_a);
  // J=[-A^-1 D; I], T=B-D^T A^-1 D. OBS directions are preserved exactly.
  const double* cross = s + std::size_t(o) * n;
  gemm(false, false, o, x, o, inv_a, o, cross, n, j, n, -1.);
  identity_tail<<<blocks(xx), 256, 0, stream.get()>>>(j, n, o);
  gemm(true, false, x, x, o, cross, n, j, n, t, x, 1., 1.);
  symmetrize<<<blocks(xx), 256, 0, stream.get()>>>(t, x, false, info.get());
  check_flags(stream, info.get(), 1, "nonfinite CABS projected overlap");
  auto spectrum = eigen(x, t, et);
  double aux_scale = 0.;
  for (int i = o; i < n; i++) aux_scale = std::max(aux_scale, overlap[std::size_t(i) * (n + 1ull)]);
  double largest = spectrum.back();
  if (largest <= guard * aux_scale || largest <= 0. ||
      spectrum.front() < -guard * std::max(aux_scale, largest))
    throw std::runtime_error("indefinite or unresolved CABS projected overlap");
  double threshold = cutoff * largest;
  double distance = std::numeric_limits<double>::infinity();
  int kept = 0;
  for (double v : spectrum) {
    kept += v > threshold;
    distance = std::min(distance, std::abs(v - threshold));
  }
  if (distance <= guard * std::max(largest, aux_scale))
    throw std::runtime_error("CABS eigenvalue inside the cutoff guard");
  if (expected >= 0 && kept != expected) throw std::runtime_error("CABS retained rank changed");
  invert(x, t, et, threshold, inv_t);
  gemm(false, false, n, x, x, j, n, inv_t, x, tmp, n);
  gemm(false, true, n, n, x, tmp, n, j, n, k, n);
  symmetrize<<<blocks(nn), 256, 0, stream.get()>>>(k, n, false, info.get());
  if (seed) {
    load(seed, false);
    // G=frechet(T^+)[J^T U J]. The generated rule includes retained/discarded rotations.
    gemm(false, false, n, x, n, s, n, j, n, tmp, n);
    gemm(true, false, x, x, n, j, n, tmp, n, inv_t, x);
    generativeqc::tensor::launch_symmetric_pseudoinverse_vjp(x, t, et, cutoff, inv_t, tmp, tmp2,
                                                             inv_t, stream.get());
    gemm(false, false, n, x, x, j, n, inv_t, x, tmp, n);
    gemm(false, true, n, n, x, tmp, n, j, n, tmp2, n);
    // bar_S = J G J^T - O U K - K U O, O contains A^-1 in the OBS block.
    gemm(false, false, o, n, n, s, n, k, n, tmp, o);
    gemm(false, false, o, n, o, inv_a, o, tmp, o, s, o);
    assemble_response<<<blocks(nn), 256, 0, stream.get()>>>(tmp2, s, n, o, tmp);
    symmetrize<<<blocks(nn), 256, 0, stream.get()>>>(tmp, n, false, info.get());
  }
  check_flags(stream, info.get(), 1, "nonfinite CABS projector or overlap response");
  const auto nr = gpu::size_mul(n, kept, "CABS frame size overflow");
  std::vector<double> staged(nn), staged_bar(seed ? nn : 0), staged_frame(nr);
  cuda_resource_check(
      cudaMemcpyAsync(staged.data(), k, bytes, cudaMemcpyDeviceToHost, stream.get()));
  if (seed)
    cuda_resource_check(
        cudaMemcpyAsync(staged_bar.data(), tmp, bytes, cudaMemcpyDeviceToHost, stream.get()));
  // cuSOLVER orders eigenvalues increasingly: the retained columns form a suffix.
  // Reuse contraction tiles only after the optional response copy on this stream.
  normalize_retained_columns<<<blocks(std::size_t(x) * kept), 256, 0, stream.get()>>>(
      t + std::size_t(x) * (x - kept), et + x - kept, x, kept, inv_t);
  gemm(false, false, n, kept, x, j, n, inv_t, x, tmp2, n);
  pack_frame<<<blocks(nr), 256, 0, stream.get()>>>(tmp2, n, kept, tmp, info.get());
  check_flags(stream, info.get(), 1, "nonfinite CABS frame");
  cuda_resource_check(cudaMemcpyAsync(staged_frame.data(), tmp, nr * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream.get()));
  stream.synchronize();
  std::copy(staged.begin(), staged.end(), kernel);
  if (seed) std::copy(staged_bar.begin(), staged_bar.end(), bar);
  std::copy(staged_frame.begin(), staged_frame.end(), frame);
  *rank = kept;
  *margin = distance / largest;
}
}  // namespace

extern "C" int generativeqc_f12_cabs_projector_cuda_v2(int device, int n, int o, double cutoff,
                                                       double guard, int expected, const double* s,
                                                       const double* seed, double* k, double* frame,
                                                       std::size_t frame_capacity, double* bar,
                                                       int* rank, double* margin, char* error,
                                                       std::size_t error_size) {
  try {
    evaluate(device, n, o, cutoff, guard, expected, s, seed, k, frame, frame_capacity, bar, rank,
             margin);
    return 0;
  } catch (const std::exception& ex) {
    generativeqc_tensor::error_text(error, error_size, ex.what());
  } catch (...) {
    generativeqc_tensor::error_text(error, error_size, "unknown CABS projector CUDA failure");
  }
  return 1;
}
