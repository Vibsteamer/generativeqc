#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "generated_symmetric_matrix_function.cuh"
#include "posthf/cabs_compensated.cuh"
#include "posthf/cabs_cuda_support.cuh"
#include "posthf/cabs_projector_cuda.hpp"
#include "solver/cuda/cholesky_provider.hpp"

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

// C contains retained, provisionally normalized columns. A compensated
// physical Gram keeps the final orthonormalization stable for large lift coefficients.
using generativeqc::posthf::cabs::Extended;
__global__ void metric_times_frame(const double* s, const double* c, size_t n, size_t x,
                                   Extended* work) {
  for (size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n * x;
       k += size_t(blockDim.x) * gridDim.x) {
    const size_t i = k % n, j = k / n;
    Extended sum{};
    for (size_t a = 0; a < n; ++a) sum.add_product(s[i + a * n], c[a + j * n]);
    work[k] = sum;
  }
}
__global__ void newton_normalizer(const double* c, const Extended* sc, size_t n, size_t x,
                                  double* m) {
  for (size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < x * x;
       k += size_t(blockDim.x) * gridDim.x) {
    const size_t i = k % x, j = k / x;
    Extended sum{};
    for (size_t a = 0; a < n; ++a) sum.add_product(sc[a + j * n], c[a + i * n]);
    m[k] = (i == j ? 1.5 : 0.) - 0.5 * (sum.hi + sum.lo);
  }
}

// A diffuse full-rank complement can have large inverse-metric elements.
// Preserve cancellation in the two contractions of its kernel cotangent.
__global__ void compensated_square_product(const double* left, const double* right, size_t n,
                                           double scale, double* product) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < n * n;
       pos += size_t(blockDim.x) * gridDim.x) {
    const size_t row = pos % n, col = pos / n;
    Extended sum{};
    for (size_t j = 0; j < n; ++j) sum.add_product(left[row + j * n], right[j + col * n]);
    product[pos] = scale * (sum.hi + sum.lo);
  }
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
                                  std::size_t o, std::size_t lift_stride, double* out) {
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n * n;
       k += std::size_t(blockDim.x) * gridDim.x) {
    const auto i = k % n, j = k / n;
    out[k] = .5 * from_metric[k] + .5 * from_metric[j + i * n] -
             (i < o ? from_lift[i + lift_stride * j] : 0.) - (j < o ? from_lift[j + lift_stride * i] : 0.);
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
  for (auto size : {oo, xx, xx, nx, std::size_t(n)})
    count = gpu::size_add(count, size, "CABS projector arena overflow");
  gpu::OwnedCudaBuffer<double> arena(device, count, stream.get());
  gpu::OwnedCudaBuffer<int> info(device, 2, stream.get());
  double* s = arena.get();
  double* k = s + nn;
  double* tmp = k + nn;
  double* tmp2 = tmp + nn;
  double* a = tmp2 + nn;
  double* t = a + oo;
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
  int cholesky_elements = 0;
  solver_check(eig::query_lower_cholesky(resources.solver, o, a, cholesky_elements));
  if (cholesky_elements < 0) throw std::runtime_error("invalid OBS Cholesky workspace");
  const auto work_bytes = std::max({wa.device_bytes, wt.device_bytes,
                                    gpu::size_mul(std::size_t(cholesky_elements), sizeof(double),
                                                  "CABS Cholesky workspace overflow")});
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
  // Solve against A directly. Multiplying a spectral inverse by D can lose
  // significant digits before the Schur subtraction and frame normalization.
  block_copy(a, s, o);
  solver_check(eig::launch_lower_cholesky(resources.solver, o, a,
                                          reinterpret_cast<double*>(workspace.get()),
                                          cholesky_elements, info.get() + 1));
  check_flags(stream, info.get() + 1, 1, "CABS OBS Cholesky factorization failed");
  auto solve_obs = [&](double* rhs, int cols, int ld_rhs, double scale) {
    const double one = 1.;
    blas_check(cublasDtrsm(blas.handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                           CUBLAS_DIAG_NON_UNIT, o, cols, &scale, a, o, rhs, ld_rhs));
    blas_check(cublasDtrsm(blas.handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                           CUBLAS_DIAG_NON_UNIT, o, cols, &one, a, o, rhs, ld_rhs));
  };
  // J=[-A^-1 D; I], T=B+D^T J_obs. OBS directions are preserved exactly.
  const double* cross = s + std::size_t(o) * n;
  cuda_resource_check(cudaMemcpy2DAsync(
      j, std::size_t(n) * sizeof(double), cross, std::size_t(n) * sizeof(double),
      std::size_t(o) * sizeof(double), x, cudaMemcpyDeviceToDevice, stream.get()));
  solve_obs(j, x, n, -1.);
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
  const bool full_rank = kept == x;
  if (full_rank) {
    // Full rank fixes the subspace independently of its frame gauge.
    // Correct the realized frame's physical Gram and form K from these same columns.
    // clang-format off
    normalize_retained_columns<<<blocks(xx),256,0,stream.get()>>>(t, et, x, x, inv_t);
    // clang-format on
    gemm(false, false, n, x, x, j, n, inv_t, x, tmp2, n);
    gpu::OwnedCudaBuffer<Extended> metric_frame(device, nx, stream.get());
    // clang-format off
    metric_times_frame<<<blocks(nx),256,0,stream.get()>>>(s,tmp2,n,x,metric_frame.get());
    newton_normalizer<<<blocks(xx),256,0,stream.get()>>>(tmp2,metric_frame.get(),n,x,inv_t);
    // clang-format on
    cuda_resource_check(cudaGetLastError());
    gemm(false, false, n, x, x, tmp2, n, inv_t, x, j, n);
    gemm(false, true, n, n, x, j, n, j, n, k, n);
  } else {
    invert(x, t, et, threshold, inv_t);
    gemm(false, false, n, x, x, j, n, inv_t, x, tmp, n);
    gemm(false, true, n, n, x, tmp, n, j, n, k, n);
  }
  symmetrize<<<blocks(nn), 256, 0, stream.get()>>>(k, n, false, info.get());
  if (seed) {
    load(seed, false);
    if (full_rank) {
      // K = S^-1 - O, O=embed(A^-1). Its full-rank derivative uses the same
      // realized K as the forward evaluation. The two cross terms follow below.
      compensated_square_product<<<blocks(nn), 256, 0, stream.get()>>>(s, k, n, 1., tmp);
      compensated_square_product<<<blocks(nn), 256, 0, stream.get()>>>(k, tmp, n, -1., tmp2);
    } else {
      // G=frechet(T^+)[J^T U J]. Include retained/discarded rotations.
      gemm(false, false, n, x, n, s, n, j, n, tmp, n);
      gemm(true, false, x, x, n, j, n, tmp, n, inv_t, x);
      generativeqc::tensor::launch_symmetric_pseudoinverse_vjp(x, t, et, cutoff, inv_t, tmp, tmp2,
                                                               inv_t, stream.get());
      gemm(false, false, n, x, x, j, n, inv_t, x, tmp, n);
      gemm(false, true, n, n, x, tmp, n, j, n, tmp2, n);
    }
    // bar_S = J G J^T - O U K - K U O, O contains A^-1 in the OBS block.
    const int lift_stride = full_rank ? n : o;
    if (!full_rank) gemm(false, false, o, n, n, s, n, k, n, tmp, o);
    solve_obs(tmp, n, lift_stride, 1.);
    assemble_response<<<blocks(nn), 256, 0, stream.get()>>>(tmp2, tmp, n, o, lift_stride, s);
    symmetrize<<<blocks(nn), 256, 0, stream.get()>>>(s, n, false, info.get());
  }
  check_flags(stream, info.get(), 1, "nonfinite CABS projector or overlap response");
  const auto nr = gpu::size_mul(n, kept, "CABS frame size overflow");
  std::vector<double> staged(nn), staged_bar(seed ? nn : 0), staged_frame(nr);
  cuda_resource_check(
      cudaMemcpyAsync(staged.data(), k, bytes, cudaMemcpyDeviceToHost, stream.get()));
  if (seed)
    cuda_resource_check(
        cudaMemcpyAsync(staged_bar.data(), s, bytes, cudaMemcpyDeviceToHost, stream.get()));
  // cuSOLVER orders eigenvalues increasingly: the retained columns form a suffix.
  // Reuse contraction tiles only after the optional response copy on this stream.
  const double* retained = j;
  if (!full_rank) {
    normalize_retained_columns<<<blocks(std::size_t(x) * kept), 256, 0, stream.get()>>>(
        t + std::size_t(x) * (x - kept), et + x - kept, x, kept, inv_t);
    gemm(false, false, n, kept, x, j, n, inv_t, x, tmp2, n);
    retained = tmp2;
  }
  pack_frame<<<blocks(nr), 256, 0, stream.get()>>>(retained, n, kept, tmp, info.get());
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

namespace {
// Row-major C is column-major C^T; explicitly index its two independent
// contractions so a second leg never discards the low word of the first.
__global__ void operator_times_frame(const double* ao, const double* c, size_t n, size_t m,
                                     Extended* work) {
  for (size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n * m;
       k += size_t(blockDim.x) * gridDim.x) {
    const size_t row = k / m, col = k % m;
    Extended sum{};
    for (size_t a = 0; a < n; ++a) sum.add_product(ao[row * n + a], c[a * m + col]);
    work[k] = sum;
  }
}
__global__ void frame_times_product(const double* c, const Extended* work, size_t n, size_t m,
                                    double* output) {
  for (size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < m * m;
       k += size_t(blockDim.x) * gridDim.x) {
    const size_t row = k / m, col = k % m;
    Extended sum{};
    for (size_t a = 0; a < n; ++a) sum.add_product(work[a * m + col], c[a * m + row]);
    output[k] = sum.hi + sum.lo;
  }
}
}  // namespace

extern "C" int generativeqc_f12_cabs_congruence_cuda_v1(int device, int n, int columns,
                                                        const double* ao, const double* frame,
                                                        double* transformed, char* error,
                                                        size_t error_size) {
  try {
    if (device < 0 || n < 1 || columns < 1 || columns > n || !ao || !frame || !transformed)
      throw std::invalid_argument("invalid physical CABS congruence buffers or dimensions");
    const auto nn = gpu::size_mul(n, n, "AO operator extent overflow");
    const auto nc = gpu::size_mul(n, columns, "CABS frame extent overflow");
    const auto cc = gpu::size_mul(columns, columns, "extended Fock extent overflow");
    gpu::CudaDeviceScope scope(device);
    gpu::OwnedCudaStream stream(device);
    gpu::OwnedCudaBuffer<double> a(device, nn, stream.get()), c(device, nc, stream.get()),
        f(device, cc, stream.get());
    gpu::OwnedCudaBuffer<Extended> work(device, nc, stream.get());
    cuda_resource_check(
        cudaMemcpyAsync(a.get(), ao, nn * sizeof(double), cudaMemcpyHostToDevice, stream.get()));
    cuda_resource_check(
        cudaMemcpyAsync(c.get(), frame, nc * sizeof(double), cudaMemcpyHostToDevice, stream.get()));
    operator_times_frame<<<blocks(nc), 256, 0, stream.get()>>>(a.get(), c.get(), n, columns,
                                                               work.get());
    frame_times_product<<<blocks(cc), 256, 0, stream.get()>>>(c.get(), work.get(), n, columns,
                                                              f.get());
    cuda_resource_check(cudaGetLastError());
    std::vector<double> staged(cc);
    cuda_resource_check(cudaMemcpyAsync(staged.data(), f.get(), cc * sizeof(double),
                                        cudaMemcpyDeviceToHost, stream.get()));
    stream.synchronize();
    if (!std::all_of(staged.begin(), staged.end(), [](double x) { return std::isfinite(x); }))
      throw std::runtime_error("nonfinite compensated CABS congruence");
    std::copy(staged.begin(), staged.end(), transformed);
    return 0;
  } catch (const std::exception& ex) {
    generativeqc_tensor::error_text(error, error_size, ex.what());
  } catch (...) {
    generativeqc_tensor::error_text(error, error_size, "unknown CABS congruence CUDA failure");
  }
  return 1;
}
