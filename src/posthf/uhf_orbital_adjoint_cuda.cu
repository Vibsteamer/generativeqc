#include <cusolverDn.h>
#include <math_constants.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
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

// Nonnegative IEEE doubles order like their unsigned bit patterns, so atomicMax keeps max_j.
__global__ void one_norm(const double* matrix, std::size_t rank, unsigned long long* norm) {
  __shared__ double partial[256];
  for (std::size_t col = blockIdx.x; col < rank; col += gridDim.x) {
    double sum = 0.;
    for (std::size_t row = threadIdx.x; row < rank; row += blockDim.x)
      sum += fabs(matrix[col * rank + row]);
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
      if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
      __syncthreads();
    }
    if (!threadIdx.x)
      atomicMax(norm, static_cast<unsigned long long>(__double_as_longlong(partial[0])));
    __syncthreads();
  }
}

// Error-free transformations. The _rn intrinsics are never contracted into FMAs, which would
// break the exact TwoSum/TwoProduct identities.
__device__ void two_sum(double a, double b, double& sum, double& error) {
  sum = __dadd_rn(a, b);
  const double v = __dsub_rn(sum, a);
  error = __dadd_rn(__dsub_rn(a, __dsub_rn(sum, v)), __dsub_rn(b, v));
}
struct Compensated {
  double sum = 0., error = 0.;
  __device__ void add(double x) {
    double e;
    two_sum(sum, x, sum, e);
    error = __dadd_rn(error, e);
  }
  __device__ void add_product(double a, double b) {
    const double p = __dmul_rn(a, b);
    add(p);
    error = __dadd_rn(error, fma(a, b, -p));
  }
};

// The high word is accumulated by TwoSum. Directed operations enclose its low word,
// including a product error too small to represent exactly. A nonfinite interval fails closed.
struct EnclosedSum {
  double sum = 0., lower = 0., upper = 0.;
  __device__ void add(double x) {
    double error;
    two_sum(sum, x, sum, error);
    lower = __dadd_rd(lower, error);
    upper = __dadd_ru(upper, error);
  }
  __device__ void add_product(double a, double b) {
    const double product = __dmul_rn(a, b);
    add(product);
    lower = __dadd_rd(lower, __fma_rd(a, b, -product));
    upper = __dadd_ru(upper, __fma_ru(a, b, -product));
  }
  __device__ void add_scaled(const EnclosedSum& x, double factor) {
    add_product(factor, x.sum);
    lower = __fma_rd(factor, factor < 0. ? x.upper : x.lower, lower);
    upper = __fma_ru(factor, factor < 0. ? x.lower : x.upper, upper);
  }
  __device__ void merge(const EnclosedSum& other) {
    add(other.sum);
    lower = __dadd_rd(lower, other.lower);
    upper = __dadd_ru(upper, other.upper);
  }
  __device__ double magnitude() const {
    const double lo = __dadd_rd(sum, lower), hi = __dadd_ru(sum, upper);
    return lo <= hi ? fmax(fabs(lo), fabs(hi)) : CUDART_INF;
  }
};

// Recompute the Hessian of the supplied factors in compensated arithmetic, independently
// of the FP64 assembly used for the initial LU solve.
template <typename Sum>
__device__ Sum hessian_element_sum(uhf::Spin alpha, uhf::Spin beta, std::size_t naux,
                                   std::size_t row, std::size_t col) {
  const bool rb = row >= alpha.rotations(), cb = col >= alpha.rotations();
  const uhf::Spin r = rb ? beta : alpha, c = cb ? beta : alpha;
  const auto ir = row - (rb ? alpha.rotations() : 0), ic = col - (cb ? alpha.rotations() : 0);
  const auto a = ir / r.occupied, i = ir % r.occupied, b = ic / c.occupied, j = ic % c.occupied;
  Sum value;
  if (row == col) value.add(r.gap[ir]);
  for (std::size_t q = 0; q < naux; ++q) {
    value.add_product(2. * r.vo[q * r.rotations() + ir], c.vo[q * c.rotations() + ic]);
    if (rb == cb) {
      value.add_product(-r.vv[(q * r.virtuals + a) * r.virtuals + b],
                        r.oo[(q * r.occupied + i) * r.occupied + j]);
      value.add_product(-r.vo[(q * r.virtuals + a) * r.occupied + j],
                        r.vo[(q * r.virtuals + b) * r.occupied + i]);
    }
  }
  return value;
}

// out[col] = driver[col] - sum_row H(row, col) z[row]: the transpose system that getrs solves.
// out holds the driver on entry.
__global__ void residual(uhf::Spin alpha, uhf::Spin beta, std::size_t q, std::size_t rank,
                         const double* z, double* out) {
  __shared__ double sums[256], errors[256];
  for (std::size_t col = blockIdx.x; col < rank; col += gridDim.x) {
    Compensated acc;
    for (std::size_t row = threadIdx.x; row < rank; row += blockDim.x) {
      const auto h = hessian_element_sum<Compensated>(alpha, beta, q, row, col);
      acc.add_product(h.sum, z[row]);
      acc.error = fma(h.error, z[row], acc.error);
    }
    sums[threadIdx.x] = acc.sum;
    errors[threadIdx.x] = acc.error;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
      if (threadIdx.x < stride) {
        Compensated merged{sums[threadIdx.x], errors[threadIdx.x]};
        merged.add(sums[threadIdx.x + stride]);
        sums[threadIdx.x] = merged.sum;
        errors[threadIdx.x] = __dadd_rn(merged.error, errors[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    if (!threadIdx.x) {
      Compensated total{out[col], 0.};
      total.add(-sums[0]);
      out[col] = __dadd_rn(total.sum, __dsub_rn(total.error, errors[0]));
    }
    __syncthreads();
  }
}

__device__ double sum_upper(double value, double* partial) {
  partial[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      partial[threadIdx.x] = __dadd_ru(partial[threadIdx.x], partial[threadIdx.x + stride]);
    __syncthreads();
  }
  const double result = partial[0];
  __syncthreads();
  return result;
}

__global__ void lower_column_sums(const double* lu, std::size_t rank, double* sums) {
  __shared__ double partial[256];
  for (std::size_t col = blockIdx.x; col < rank; col += gridDim.x) {
    double total = 0.;
    for (std::size_t row = col + 1 + threadIdx.x; row < rank; row += blockDim.x)
      total = __dadd_ru(total, fabs(lu[col * rank + row]));
    total = __dadd_ru(1., sum_upper(total, partial));
    if (!threadIdx.x) sums[col] = total;
  }
}

// Bound |d - H^T(z + dz)| and the column sums of |PH - LU|. Evaluate the
// returned LU factors directly, using either an FP64 dot product with a
// rounding bound or a tighter compensated interval. pivot_rows maps each
// original row to its row in PH.
template <bool compensated_product>
__global__ void refinement_enclosures(uhf::Spin alpha, uhf::Spin beta, std::size_t q,
                                       std::size_t rank, const double* lu,
                                       const int* pivot_rows, const double* z, const double* dz,
                                       const double* lower_sums, double* rhs, double* defect) {
  __shared__ EnclosedSum residual_parts[256];
  __shared__ double partial[256];
  for (std::size_t col = blockIdx.x; col < rank; col += gridDim.x) {
    EnclosedSum residual;
    double defect_sum = 0., factor_sum = 0.;
    for (std::size_t row = threadIdx.x; row < rank; row += blockDim.x) {
      auto h = hessian_element_sum<EnclosedSum>(alpha, beta, q, row, col);
      residual.add_scaled(h, -z[row]);
      residual.add_scaled(h, -dz[row]);
      const auto pivoted = static_cast<std::size_t>(pivot_rows[row]);
      const double unit_lower_term = pivoted <= col ? lu[col * rank + pivoted] : 0.;
      if constexpr (compensated_product) {
        h.add(-unit_lower_term);
        for (std::size_t k = 0; k < pivoted && k <= col; ++k)
          h.add_product(-lu[k * rank + pivoted], lu[col * rank + k]);
      } else {
        double product = unit_lower_term;
        for (std::size_t k = 0; k < pivoted && k <= col; ++k)
          product = __fma_rn(lu[k * rank + pivoted], lu[col * rank + k], product);
        h.add(-product);
        if (row <= col)
          factor_sum = __fma_ru(lower_sums[row], fabs(lu[col * rank + row]), factor_sum);
      }
      defect_sum = __dadd_ru(defect_sum, h.magnitude());
    }
    residual_parts[threadIdx.x] = residual;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
      if (threadIdx.x < stride)
        residual_parts[threadIdx.x].merge(residual_parts[threadIdx.x + stride]);
      __syncthreads();
    }
    defect_sum = sum_upper(defect_sum, partial);
    if constexpr (!compensated_product) {
      // The dot has at most n rounded FMAs. 2 n u bounds gamma_n for n <= INT_MAX.
      // The n^2 DBL_MIN term bounds additive underflow across all dots in this column.
      factor_sum = sum_upper(factor_sum, partial);
      const double n = static_cast<double>(rank);
      const double underflow_and_defect = __fma_ru(n, n * 0x1p-1022, defect_sum);
      defect_sum = __fma_ru(n * 0x1p-52, factor_sum, underflow_and_defect);
    }
    if (!threadIdx.x) {
      auto total = residual_parts[0];
      total.add(rhs[col]);
      rhs[col] = total.magnitude();
      defect[col] = defect_sum;
    }
    __syncthreads();
  }
}

// For nonnegative b, solve the two triangular comparison systems. Their
// upper-rounded solution bounds |(P^T LU)^-T b| up to the final permutation.
__global__ void check_forward_error(const double* lu, std::size_t rank, double change,
                                      double scale, double* rhs, double* defect) {
  __shared__ double partial_rhs[256], partial_defect[256];
  for (std::size_t col = 0; col < rank; ++col) {
    double value = 0., perturbation = 0.;
    for (std::size_t row = threadIdx.x; row < col; row += blockDim.x) {
      const double factor = fabs(lu[col * rank + row]);
      if (factor == 0.) continue;
      value = __fma_ru(factor, rhs[row], value);
      perturbation = __fma_ru(factor, defect[row], perturbation);
    }
    value = sum_upper(value, partial_rhs);
    perturbation = sum_upper(perturbation, partial_defect);
    if (!threadIdx.x) {
      const double pivot = fabs(lu[col * rank + col]);
      rhs[col] = __ddiv_ru(__dadd_ru(rhs[col], value), pivot);
      defect[col] = __ddiv_ru(__dadd_ru(defect[col], perturbation), pivot);
    }
    __syncthreads();
  }
  for (std::size_t col = rank; col--;) {
    double value = 0., perturbation = 0.;
    for (std::size_t row = col + 1 + threadIdx.x; row < rank; row += blockDim.x) {
      const double factor = fabs(lu[col * rank + row]);
      if (factor == 0.) continue;
      value = __fma_ru(factor, rhs[row], value);
      perturbation = __fma_ru(factor, defect[row], perturbation);
    }
    value = sum_upper(value, partial_rhs);
    perturbation = sum_upper(perturbation, partial_defect);
    if (!threadIdx.x) {
      rhs[col] = __dadd_ru(rhs[col], value);
      defect[col] = __dadd_ru(defect[col], perturbation);
    }
    __syncthreads();
  }
  double maximum_rhs = 0., maximum_defect = 0.;
  for (std::size_t col = threadIdx.x; col < rank; col += blockDim.x) {
    maximum_rhs = isfinite(rhs[col]) ? fmax(maximum_rhs, rhs[col]) : CUDART_INF;
    maximum_defect = isfinite(defect[col]) ? fmax(maximum_defect, defect[col]) : CUDART_INF;
  }
  partial_rhs[threadIdx.x] = maximum_rhs;
  partial_defect[threadIdx.x] = maximum_defect;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) {
      partial_rhs[threadIdx.x] = fmax(partial_rhs[threadIdx.x], partial_rhs[threadIdx.x + stride]);
      partial_defect[threadIdx.x] =
          fmax(partial_defect[threadIdx.x], partial_defect[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    const double gap = __dsub_rd(1., partial_defect[0]);
    const double bound =
        gap > 0. ? __dadd_ru(change, __ddiv_ru(partial_rhs[0], gap)) : CUDART_INF;
    rhs[0] = isfinite(bound) && bound <= __dmul_rd(0x1p-39, scale) ? 1. : 0.;
  }
}

// Let B = P^T LU, M = I - B^-T H^T and s = d - H^T(z + dz). The comparison
// solves give m >= ||M||_inf and t >= ||B^-T s||_inf. If m < 1, the Neumann
// series bounds ||z_exact - z||_inf by ||dz||_inf + t / (1 - m).
bool certify(const gpu::OwnedCudaStream& stream, uhf::Spin alpha, uhf::Spin beta,
             std::size_t q, std::size_t rank, const double* lu, int* pivots, const double* driver,
             const double* z, double* work, double change, double scale) {
  const auto bytes = mul(rank, sizeof(double));
  const auto grid = static_cast<unsigned>(std::min<std::size_t>(rank, 65535));
  double* rhs = work + rank;
  double* defect = rhs + rank;
  double* lower_sums = defect + rank;
  const auto reset_rhs = [&] {
    cuda_resource_check(cudaMemcpyAsync(rhs, driver, bytes, cudaMemcpyHostToDevice, stream.get()));
  };
  const auto admit = [&] {
    check_forward_error<<<1, 256, 0, stream.get()>>>(lu, rank, change, scale, rhs, defect);
    cuda_resource_check(cudaGetLastError());
    double success = 0.;
    cuda_resource_check(
        cudaMemcpyAsync(&success, rhs, sizeof(success), cudaMemcpyDeviceToHost, stream.get()));
    stream.synchronize();
    return success == 1.;
  };
  // No further LU solves run after this point, so the pivot buffer is reusable.
  const auto pivot_bytes = mul(rank, sizeof(int));
  std::vector<int> swaps(rank), order(rank);
  std::iota(order.begin(), order.end(), 0);
  cuda_resource_check(
      cudaMemcpyAsync(swaps.data(), pivots, pivot_bytes, cudaMemcpyDeviceToHost, stream.get()));
  stream.synchronize();
  for (std::size_t row = 0; row < rank; ++row) {
    const int target = swaps[row];
    if (target < 1 || static_cast<std::size_t>(target) > rank)
      throw std::runtime_error("invalid DF-UHF LU pivot");
    std::swap(order[row], order[static_cast<std::size_t>(target - 1)]);
  }
  for (std::size_t row = 0; row < rank; ++row)
    swaps[static_cast<std::size_t>(order[row])] = static_cast<int>(row);
  cuda_resource_check(
      cudaMemcpyAsync(pivots, swaps.data(), pivot_bytes, cudaMemcpyHostToDevice, stream.get()));
  reset_rhs();
  lower_column_sums<<<grid, 256, 0, stream.get()>>>(lu, rank, lower_sums);
  cuda_resource_check(cudaGetLastError());
  refinement_enclosures<false><<<grid, 256, 0, stream.get()>>>(
      alpha, beta, q, rank, lu, pivots, z, work, lower_sums, rhs, defect);
  cuda_resource_check(cudaGetLastError());
  if (admit()) return true;

  reset_rhs();
  refinement_enclosures<true><<<grid, 256, 0, stream.get()>>>(
      alpha, beta, q, rank, lu, pivots, z, work, nullptr, rhs, defect);
  cuda_resource_check(cudaGetLastError());
  return admit();
}

double max_abs(const std::vector<double>& x) {
  double peak = 0.;
  for (const double v : x) peak = std::max(peak, std::abs(v));
  return peak;
}

// Observed corrections must at least halve before a candidate can pass the threshold.
// Certify every candidate, including the first one, before publishing z without its final dz.
void refine(const Solver& solver, const gpu::OwnedCudaStream& stream, uhf::Spin alpha,
            uhf::Spin beta, std::size_t q, int n, const double* lu, int* pivots, int* info,
            const double* driver, double* z_device, double* scratch, std::vector<double>& z) {
  constexpr int max_corrections = 8;
  const auto rank = static_cast<std::size_t>(n);
  const auto bytes = mul(rank, sizeof(double));
  std::vector<double> correction(rank);
  cuda_resource_check(
      cudaMemcpyAsync(z.data(), z_device, bytes, cudaMemcpyDeviceToHost, stream.get()));
  double previous = 0.;
  for (int step = 0;; ++step) {
    cuda_resource_check(
        cudaMemcpyAsync(scratch, driver, bytes, cudaMemcpyHostToDevice, stream.get()));
    residual<<<static_cast<unsigned>(std::min<std::size_t>(rank, 65535)), 256, 0, stream.get()>>>(
        alpha, beta, q, rank, z_device, scratch);
    cuda_resource_check(cudaGetLastError());
    solver_check(
        cusolverDnDgetrs(solver.handle, CUBLAS_OP_T, n, 1, lu, n, pivots, scratch, n, info));
    check_flags(stream, info, 1, "invalid DF-UHF refinement solve");
    cuda_resource_check(
        cudaMemcpyAsync(correction.data(), scratch, bytes, cudaMemcpyDeviceToHost, stream.get()));
    stream.synchronize();
    if (!std::all_of(correction.begin(), correction.end(),
                     [](double v) { return std::isfinite(v); }) ||
        !std::all_of(z.begin(), z.end(), [](double v) { return std::isfinite(v); }))
      throw std::runtime_error("nonfinite DF-UHF adjoint refinement");
    const double size = max_abs(z), change = max_abs(correction);
    if (step && 2. * change > previous) {
      char detail[180];
      std::snprintf(
          detail, sizeof(detail),
          "ill-conditioned DF-UHF orbital Hessian: refined adjoint correction ratio %.3f "
          "exceeds 0.5 at %.3e of |z| after %d corrections",
          change / previous, change / size, step);
      throw std::runtime_error(detail);
    }
    if (change <= 0x1p-40 * size) {
      if (!certify(stream, alpha, beta, q, rank, lu, pivots, driver, z_device, scratch, change,
                   size))
        throw std::runtime_error(
            "ill-conditioned DF-UHF orbital Hessian: unable to bound refined adjoint error");
      return;
    }
    if (step == max_corrections) {
      char detail[200];
      std::snprintf(detail, sizeof(detail),
                    "ill-conditioned DF-UHF orbital Hessian: refined adjoint correction %.3e of "
                    "|z| exceeds %.3e after %d corrections",
                    change / size, 0x1p-40, step);
      throw std::runtime_error(detail);
    }
    for (std::size_t i = 0; i < rank; ++i) z[i] += correction[i];
    cuda_resource_check(
        cudaMemcpyAsync(z_device, z.data(), bytes, cudaMemcpyHostToDevice, stream.get()));
    previous = change;
  }
}

// rho = ||H||_1 ||z||_1 / ||d||_1 <= kappa_1(H) for z = H^-T d. Above eps^-1/2 the response is
// dominated by near-null modes whose eigenvalues rounding-level input errors already shift, so
// even an exact solve is not determined by its inputs; symmetry-protected null modes keep rho
// small.
void admit_amplification(double hessian_norm, const double* driver, const double* z,
                         std::size_t rank) {
  double driver_norm = 0., solution_norm = 0.;
  for (std::size_t i = 0; i < rank; ++i) {
    driver_norm += std::abs(driver[i]);
    solution_norm += std::abs(z[i]);
  }
  if (driver_norm == 0.) return;
  const double limit = 1. / std::sqrt(std::numeric_limits<double>::epsilon());
  if (!(hessian_norm * solution_norm <= limit * driver_norm)) {
    char detail[160];
    std::snprintf(detail, sizeof(detail),
                  "ill-conditioned DF-UHF orbital Hessian: driver amplification %.3e exceeds %.3e",
                  hessian_norm * solution_norm / driver_norm, limit);
    throw std::runtime_error(detail);
  }
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
      k < 0 || terms < 0 || (!k && terms) || !host || !z || (k && (!g || !force)))
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
  // Verification reuses the LU scratch for the correction and three rank-sized bounds.
  const auto scratch = std::max<std::size_t>(std::max(lwork, 0), mul(4, rank));
  if (lwork < 1 || add(fixed_bytes, mul(scratch, sizeof(double))) > budget)
    throw std::length_error("DF-UHF LU workspace exceeds device budget");
  gpu::OwnedCudaBuffer<double> data(device, count, stream.get()),
      work(device, scratch, stream.get()), grad;
  if (k) grad.allocate(device, k, stream.get());
  gpu::OwnedCudaBuffer<int> flags(device, rank + 1, stream.get());
  int* info = flags.get() + rank;
  cuda_resource_check(cudaMemcpyAsync(data.get(), host, mul(count, sizeof(double)),
                                      cudaMemcpyHostToDevice, stream.get()));
  const auto alpha = uhf::spin(data.get(), oa, va, q), beta = uhf::spin(data.get() + fa, ob, vb, q);
  assemble<<<blocks(square), 256, 0, stream.get()>>>(alpha, beta, q, rank, matrix.get());
  cuda_resource_check(cudaGetLastError());
  // ||H||_1 is reduced into the LU scratch before getrf reuses it.
  auto* norm_bits = reinterpret_cast<unsigned long long*>(work.get());
  unsigned long long host_norm_bits = 0;
  cuda_resource_check(cudaMemsetAsync(norm_bits, 0, sizeof(*norm_bits), stream.get()));
  one_norm<<<static_cast<unsigned>(std::min<std::size_t>(rank, 65535)), 256, 0, stream.get()>>>(
      matrix.get(), rank, norm_bits);
  cuda_resource_check(cudaGetLastError());
  cuda_resource_check(cudaMemcpyAsync(&host_norm_bits, norm_bits, sizeof(host_norm_bits),
                                      cudaMemcpyDeviceToHost, stream.get()));
  double* driver = data.get() + both;
  double* rhs = driver + rank;
  double* fixed = rhs + rhs_count;
  solver_check(
      cusolverDnDgetrf(solver.handle, n, n, matrix.get(), n, work.get(), flags.get(), info));
  check_flags(stream, info, 1, "singular DF-UHF orbital Hessian");
  solver_check(cusolverDnDgetrs(solver.handle, CUBLAS_OP_T, n, 1, matrix.get(), n, flags.get(),
                                driver, n, info));
  check_flags(stream, info, 1, "invalid DF-UHF transpose solve");
  std::vector<double> solution(rank);
  refine(solver, stream, alpha, beta, q, n, matrix.get(), flags.get(), info, host + both, driver,
         work.get(), solution);
  if (k) {
    const double one = 1., zero = 0.;
    blas_check(
        cublasDgemv(blas.handle, CUBLAS_OP_T, n, k, &one, rhs, n, driver, 1, &zero, grad.get(), 1));
    finish_forces<<<blocks(k), 256, 0, stream.get()>>>(fixed, grad.get(), k, terms);
    cuda_resource_check(cudaGetLastError());
  }
  std::vector<double> result(add(rank, mul(2, k)));
  const auto read = [&](double* dest, const double* src, std::size_t nvalue) {
    cuda_resource_check(cudaMemcpyAsync(dest, src, mul(nvalue, sizeof(double)),
                                        cudaMemcpyDeviceToHost, stream.get()));
  };
  std::copy(solution.begin(), solution.end(), result.begin());
  if (k) {
    read(result.data() + rank, grad.get(), k);
    read(result.data() + rank + k, fixed, k);
  }
  stream.synchronize();
  if (!std::all_of(result.begin(), result.end(), [](double x) { return std::isfinite(x); }))
    throw std::runtime_error("nonfinite DF-UHF adjoint or molecular force");
  admit_amplification(std::bit_cast<double>(host_norm_bits), host + both, result.data(), rank);
  std::copy_n(result.data(), rank, z);
  if (k) {
    std::copy_n(result.data() + rank, k, g);
    std::copy_n(result.data() + rank + k, k, force);
  }
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
