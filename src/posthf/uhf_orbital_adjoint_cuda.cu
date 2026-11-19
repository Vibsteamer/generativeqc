#include <cusolverDn.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "posthf/cabs_cuda_support.cuh"
#include "posthf/uhf_orbital_adjoint_cuda.hpp"
#include "posthf/uhf_orbital_hessian.hpp"

namespace {
namespace gpu = generativeqc::runtime;
namespace uhf = generativeqc::posthf::uhf_response;
using generativeqc::posthf::cabs::Blas;
using generativeqc::posthf::cabs::blocks;
using generativeqc::posthf::cabs::check_flags;
using generativeqc_tensor::blas_check;
using gpu::cuda_resource_check;

std::size_t mul(std::size_t a, std::size_t b) {
  return gpu::size_mul(a, b, "UHF adjoint size overflow");
}
std::size_t add(std::size_t a, std::size_t b) {
  return gpu::size_add(a, b, "UHF adjoint size overflow");
}
std::size_t factors(std::size_t o, std::size_t v, std::size_t q) {
  if (!o) return 0;
  return add(mul(q, add(mul(v, o), add(mul(o, o), mul(v, v)))), mul(v, o));
}
void solver_check(cusolverStatus_t status) {
  if (status != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error("DF-UHF cuSOLVER status " + std::to_string(static_cast<int>(status)));
}
struct Solver {
  cusolverDnHandle_t handle{};
  Solver() { solver_check(cusolverDnCreate(&handle)); }
  ~Solver() { (void)cusolverDnDestroy(handle); }
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;
};

__global__ void assemble(uhf::Spin alpha, uhf::Spin beta, std::size_t q, std::size_t rank,
                         double* matrix) {
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < rank * rank;
       k += static_cast<std::size_t>(blockDim.x) * gridDim.x)
    matrix[k] = uhf::element(alpha, beta, q, k % rank, k / rank);
}

__global__ void finish_forces(double* reference, const double* orbital_gradient,
                              std::size_t directions, int terms) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < directions;
       i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    double gradient = orbital_gradient[i];
    for (int term = 0; term < terms; ++term)
      gradient += reference[(std::size_t(term) + 1) * directions + i];
    reference[i] -= gradient;
  }
}

void solve(int device, int oa, int ob, int orbitals, int q, int k, int terms, const double* host,
           std::size_t count, std::size_t budget, double* z, double* g, double* force) {
  if (device < 0 || oa < 0 || ob < 0 || !(oa || ob) || std::max(oa, ob) >= orbitals || q < 1 ||
      k < 1 || terms < 0 || !host || !z || !g || !force)
    throw std::invalid_argument("invalid DF-UHF orbital response dimensions or buffers");
  const int va = orbitals - oa, vb = orbitals - ob;
  const auto rank = add(mul(oa, va), mul(ob, vb)), fa = factors(oa, va, q),
             both = add(fa, factors(ob, vb, q));
  const auto square = mul(rank, rank), rhs_count = mul(k, rank);
  if (rank > INT_MAX || count != add(both, add(rank, add(rhs_count, mul(k, add(terms, 1))))))
    throw std::invalid_argument("DF-UHF packed size or cuSOLVER dimension mismatch");
  const auto fixed_bytes =
      add(mul(add(count, add(square, k)), sizeof(double)), mul(add(rank, 1), sizeof(int)));
  if (fixed_bytes > budget)
    throw std::length_error("DF-UHF response exceeds device workspace budget");
  if (!std::all_of(host, host + count, [](double x) { return std::isfinite(x); }))
    throw std::invalid_argument("nonfinite DF-UHF orbital response input");
  gpu::CudaDeviceScope scope(device);
  gpu::OwnedCudaStream stream(device);
  Solver solver;
  Blas blas;
  solver_check(cusolverDnSetStream(solver.handle, stream.get()));
  blas_check(cublasSetStream(blas.handle, stream.get()));
  gpu::OwnedCudaBuffer<double> matrix(device, square, stream.get());
  const int n = static_cast<int>(rank);
  int lwork = 0;
  solver_check(cusolverDnDgetrf_bufferSize(solver.handle, n, n, matrix.get(), n, &lwork));
  if (lwork < 1 || add(fixed_bytes, mul(lwork, sizeof(double))) > budget)
    throw std::length_error("DF-UHF LU workspace exceeds device budget");
  gpu::OwnedCudaBuffer<double> data(device, count, stream.get()), work(device, lwork, stream.get()),
      grad(device, k, stream.get());
  gpu::OwnedCudaBuffer<int> flags(device, rank + 1, stream.get());
  int* info = flags.get() + rank;
  cuda_resource_check(cudaMemcpyAsync(data.get(), host, mul(count, sizeof(double)),
                                      cudaMemcpyHostToDevice, stream.get()));
  assemble<<<blocks(square), 256, 0, stream.get()>>>(uhf::spin(data.get(), oa, va, q),
                                                     uhf::spin(data.get() + fa, ob, vb, q), q, rank,
                                                     matrix.get());
  cuda_resource_check(cudaGetLastError());
  double* driver = data.get() + both;
  double* rhs = driver + rank;
  double* fixed = rhs + rhs_count;
  solver_check(
      cusolverDnDgetrf(solver.handle, n, n, matrix.get(), n, work.get(), flags.get(), info));
  check_flags(stream, info, 1, "singular DF-UHF orbital Hessian");
  solver_check(cusolverDnDgetrs(solver.handle, CUBLAS_OP_T, n, 1, matrix.get(), n, flags.get(),
                                driver, n, info));
  check_flags(stream, info, 1, "invalid DF-UHF transpose solve");
  const double one = 1., zero = 0.;
  blas_check(
      cublasDgemv(blas.handle, CUBLAS_OP_T, n, k, &one, rhs, n, driver, 1, &zero, grad.get(), 1));
  finish_forces<<<blocks(k), 256, 0, stream.get()>>>(fixed, grad.get(), k, terms);
  cuda_resource_check(cudaGetLastError());
  std::vector<double> result(add(rank, mul(2, k)));
  const auto read = [&](double* dest, const double* src, std::size_t nvalue) {
    cuda_resource_check(cudaMemcpyAsync(dest, src, mul(nvalue, sizeof(double)),
                                        cudaMemcpyDeviceToHost, stream.get()));
  };
  read(result.data(), driver, rank);
  read(result.data() + rank, grad.get(), k);
  read(result.data() + rank + k, fixed, k);
  stream.synchronize();
  if (!std::all_of(result.begin(), result.end(), [](double x) { return std::isfinite(x); }))
    throw std::runtime_error("nonfinite DF-UHF adjoint or molecular force");
  std::copy_n(result.data(), rank, z);
  std::copy_n(result.data() + rank, k, g);
  std::copy_n(result.data() + rank + k, k, force);
}
}  // namespace

extern "C" int generativeqc_uhf_orbital_adjoint_cuda_v1(
    int device, int alpha, int beta, int orbitals, int auxiliary, int directions, int fixed_terms,
    const double* packed, std::size_t count, std::size_t maximum_bytes, double* rotations,
    double* orbital_gradient, double* forces, char* error, std::size_t error_size) {
  try {
    solve(device, alpha, beta, orbitals, auxiliary, directions, fixed_terms, packed, count,
          maximum_bytes, rotations, orbital_gradient, forces);
    return 0;
  } catch (const std::exception& ex) {
    generativeqc_tensor::error_text(error, error_size, ex.what());
  } catch (...) {
    generativeqc_tensor::error_text(error, error_size, "unknown DF-UHF orbital adjoint failure");
  }
  return 1;
}
