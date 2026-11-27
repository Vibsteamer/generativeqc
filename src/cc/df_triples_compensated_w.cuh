#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "posthf/capacity.hpp"
#include "tensor/cuda_contraction.cuh"

namespace generativeqc::cc::triples {

/** Pedantic non-TF32 IEEE FP32 strided row-major GEMM through shared Tensor provider.
 * Uses CudaContractionContext (CUBLAS_PEDANTIC_MATH) with complete finiteness audit.
 */
inline void execute_strided_fp32_product(const tensor::CudaContractionContext& provider,
                                         cudaStream_t stream, char a_trans, char b_trans,
                                         std::size_t m, std::size_t n, std::size_t k, float alpha,
                                         const float* a, std::size_t lda, const float* b,
                                         std::size_t ldb, float beta, float* c, std::size_t ldc,
                                         int* error,
                                         std::string_view qualification =
                                             "issue1764/df-triples-w-fp32-candidate-v1") {
  tensor::ContractionRequest req{};
  req.precision = runtime::PrecisionDirective{
      runtime::PrecisionDtype::Fp32, runtime::PrecisionDtype::Fp32, runtime::PrecisionDtype::Fp32,
      qualification};
  req.publication_dtype = runtime::PrecisionDtype::Fp32;
  req.a_trans = a_trans;
  req.b_trans = b_trans;
  req.batches = 1;
  req.m = m;
  req.n = n;
  req.k = k;
  req.coefficient = static_cast<double>(alpha);
  req.beta = static_cast<double>(beta);
  req.leading_dimensions = {lda, ldb, ldc};
  req.operands[0] = tensor::ContractionOperand::matrix_view(
      {0, 1},
      a_trans == 'N' ? std::initializer_list<std::size_t>{m, k}
                     : std::initializer_list<std::size_t>{k, m},
      runtime::PrecisionDtype::Fp32, 1, lda);
  req.operands[1] = tensor::ContractionOperand::matrix_view(
      {1, 2},
      b_trans == 'N' ? std::initializer_list<std::size_t>{k, n}
                     : std::initializer_list<std::size_t>{n, k},
      runtime::PrecisionDtype::Fp32, 1, ldb);
  req.operands[2] =
      tensor::ContractionOperand::matrix_view({0, 2}, {m, n}, runtime::PrecisionDtype::Fp32, 1, ldc);
  tensor::execute_matrix_contraction<float>(provider, req, tensor::ContractionAlgorithm::PedanticBlas,
                                            stream, a, b, c, error);
}

/** Dense row-major strided-batched pedantic FP32 GEMM, C[b] = A[b] B[b]. */
inline void execute_batched_fp32_product(const tensor::CudaContractionContext& provider,
                                         cudaStream_t stream, std::size_t batches, std::size_t m,
                                         std::size_t n, std::size_t k, const float* a,
                                         const float* b, float* c, int* error,
                                         std::string_view qualification) {
  tensor::ContractionRequest req{};
  req.precision =
      runtime::PrecisionDirective{runtime::PrecisionDtype::Fp32, runtime::PrecisionDtype::Fp32,
                                  runtime::PrecisionDtype::Fp32, qualification};
  req.publication_dtype = runtime::PrecisionDtype::Fp32;
  req.batches = batches;
  req.m = m;
  req.n = n;
  req.k = k;
  req.operands[0] =
      tensor::ContractionOperand::dense({0, 1, 2}, {batches, m, k}, runtime::PrecisionDtype::Fp32);
  req.operands[1] =
      tensor::ContractionOperand::dense({0, 2, 3}, {batches, k, n}, runtime::PrecisionDtype::Fp32);
  req.operands[2] =
      tensor::ContractionOperand::dense({0, 1, 3}, {batches, m, n}, runtime::PrecisionDtype::Fp32);
  tensor::execute_matrix_contraction<float>(
      provider, req, tensor::ContractionAlgorithm::PedanticBlas, stream, a, b, c, error);
}

namespace compensated_w {
inline constexpr std::string_view schedule =
    "spatial-triples/three-limb-fp32-sgemm-k64-high8-fp64-combine-v1";
inline constexpr std::size_t tile = 64;
inline constexpr std::size_t limbs = 3;

inline std::size_t workspace_bytes(std::size_t m, std::size_t n, std::size_t k) {
  if (!m || !n || !k || std::max({m, n, k}) > std::numeric_limits<int>::max())
    throw std::invalid_argument("invalid compensated triples W dimensions");
  const auto vectors = posthf::checked_add(m, n);
  const auto digits = posthf::checked_mul(limbs * std::min(k, tile), vectors);
  const auto floats = posthf::checked_add(digits, posthf::checked_mul(m, n));
  return posthf::checked_add(posthf::checked_mul(floats, sizeof(float)),
                             posthf::checked_mul(vectors, limbs * sizeof(int)));
}

inline std::size_t cast_elements(std::size_t m, std::size_t n, std::size_t k) {
  const auto packed = posthf::checked_mul(limbs, posthf::checked_mul(posthf::checked_add(m, n), k));
  const auto segments = k / tile + (k % tile != 0);
  const auto partials = posthf::checked_mul(limbs * limbs, segments);
  return posthf::checked_add(packed, posthf::checked_mul(partials, posthf::checked_mul(m, n)));
}

inline unsigned launch_blocks(std::size_t count) {
  return static_cast<unsigned>(std::min<std::size_t>(count, 65535));
}

// A full warp pair owns each row (or column) segment. Integer high and
// middle digits share a power-of-two grid in each vector. With |digit| <=
// 256, a 64-term high/middle dot fits the exact integer range of binary32.
static __device__ int grid_exponent(double x, double* maximum) {
  const auto lane = threadIdx.x;
  maximum[lane] = fabs(x);
  __syncthreads();
  for (unsigned stride = tile / 2; stride; stride /= 2) {
    if (lane < stride) maximum[lane] = fmax(maximum[lane], maximum[lane + stride]);
    __syncthreads();
  }
  const auto exponent = maximum[0] ? ilogb(maximum[0]) + 1 : 0;
  __syncthreads();
  return exponent;
}

template<bool Left>
static __global__ void pack(const double* source, bool transposed, std::size_t ld,
                            std::size_t first, std::size_t width, std::size_t vectors,
                            float* digits, std::size_t pitch, int* exponents, int* error) {
  __shared__ double maximum[tile];
  const auto lane = static_cast<std::size_t>(threadIdx.x);
  for (std::size_t i = blockIdx.x; i < vectors; i += gridDim.x) {
    double x = 0;
    if (lane < width) {
      const auto t = first + lane;
      x = source[(Left != transposed) ? i * ld + t : t * ld + i];
      if (!isfinite(x)) {
        atomicCAS(error, 0, 26);
        x = 0;
      }
    }
    for (std::size_t limb = 0; limb < limbs; ++limb) {
      const int exponent = grid_exponent(x, maximum);
      const bool last = limb + 1 == limbs;
      // The fractional low limb uses the normal FP32 range even in products
      // with another low limb. 64 terms bounded by 2^120 sum to at most 2^126.
      constexpr int low_lift = 60;
      if (lane == 0) exponents[limb * vectors + i] = last ? exponent - low_lift : exponent;
      if (lane < width) {
        const double normalized = ldexp(x, -exponent);
        float digit;
        if (last) {
          digit = __double2float_rn(ldexp(normalized, low_lift));
          if (x != 0 && fabsf(digit) < 0x1p-63f) atomicCAS(error, 0, 26);
        } else {
          digit = __double2float_rn(rint(normalized * 256.0) / 256.0);
          // A tiny component can first become significant in a later grid.
          if (digit != 0) x = ldexp(normalized - digit, exponent);
        }
        const auto slot = Left ? i * width + lane : lane * vectors + i;
        digits[limb * pitch + slot] = digit;
      }
    }
  }
}

static __global__ void combine(const float* partial, const int* ea, const int* eb,
                               double alpha, double beta, std::size_t m, std::size_t n,
                               double* c, std::size_t ldc, int* error) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < m*n;
       i += std::size_t(gridDim.x) * blockDim.x) {
    const auto row = i / n, col = i % n, offset = row * ldc + col;
    int alpha_exponent = 0;
    const double fraction = frexp(alpha, &alpha_exponent);
    const double product = ldexp(__dmul_rn(fraction, double(partial[i])),
                                 ea[row] + eb[col] + alpha_exponent);
    const double prior = beta == 0 ? 0 : __dmul_rn(beta, c[offset]);
    c[offset] = generativeqc_tensor::finite(__dadd_rn(prior, product), error, 26);
  }
}

// The stacked layouts hold segment s of a batch as a row-major left block
// [limb * m + row, t] and a row-major right block [t, limb * n + column].
// One SGEMM then forms all nine limb pairs of a segment, and a contiguous
// left limb against the right block forms three of them.
template <bool Left>
static __global__ void pack_stacked(const double* source, bool transposed, std::size_t ld,
                                    std::size_t first, std::size_t width, std::size_t vectors,
                                    std::size_t segments, float* digits, int* exponents,
                                    int* error) {
  __shared__ double maximum[tile];
  const auto lane = static_cast<std::size_t>(threadIdx.x);
  const auto block = limbs * vectors * width;
  for (std::size_t s = blockIdx.y; s < segments; s += gridDim.y) {
    for (std::size_t i = blockIdx.x; i < vectors; i += gridDim.x) {
      double x = 0;
      if (lane < width) {
        const auto t = first + s * width + lane;
        x = source[(Left != transposed) ? i * ld + t : t * ld + i];
        if (!isfinite(x)) {
          atomicCAS(error, 0, 26);
          x = 0;
        }
      }
      for (std::size_t limb = 0; limb < limbs; ++limb) {
        const int exponent = grid_exponent(x, maximum);
        const bool last = limb + 1 == limbs;
        constexpr int low_lift = 60;
        if (lane == 0)
          exponents[(s * limbs + limb) * vectors + i] = last ? exponent - low_lift : exponent;
        if (lane < width) {
          const double normalized = ldexp(x, -exponent);
          float digit;
          if (last) {
            digit = __double2float_rn(ldexp(normalized, low_lift));
            if (x != 0 && fabsf(digit) < 0x1p-63f) atomicCAS(error, 0, 26);
          } else {
            digit = __double2float_rn(rint(normalized * 256.0) / 256.0);
            if (digit != 0) x = ldexp(normalized - digit, exponent);
          }
          const auto slot = Left ? (limb * vectors + i) * width + lane
                                 : lane * (limbs * vectors) + limb * vectors + i;
          digits[s * block + slot] = digit;
        }
      }
    }
  }
}

// Adds the stacked partials of `segments` segments and `left_count` left
// limbs starting at `left_first` in the per-pair order: segment, left limb,
// right limb. Only the first term reads the seeded `beta * c`.
static __global__ void combine_stacked(const float* partial, const int* ea, const int* eb,
                                       std::size_t segments, std::size_t left_first,
                                       std::size_t left_count, double alpha, double beta,
                                       std::size_t m, std::size_t n, double* c, std::size_t ldc,
                                       int* error) {
  int alpha_exponent = 0;
  const double fraction = frexp(alpha, &alpha_exponent);
  const auto columns = limbs * n;
  const auto stride = left_count * m * columns;
  for (std::size_t e = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; e < m * n;
       e += std::size_t(gridDim.x) * blockDim.x) {
    const auto row = e / n, col = e % n, offset = row * ldc + col;
    double value = c[offset];
    bool first = true;
    for (std::size_t s = 0; s < segments; ++s) {
      for (std::size_t l = 0; l < left_count; ++l) {
        const auto* p = partial + s * stride + (l * m + row) * columns + col;
        const int left_exponent = ea[(s * limbs + left_first + l) * m + row] + alpha_exponent;
        for (std::size_t j = 0; j < limbs; ++j) {
          const double product = ldexp(__dmul_rn(fraction, double(p[j * n])),
                                       left_exponent + eb[(s * limbs + j) * n + col]);
          const double prior = !first ? value : beta == 0 ? 0 : __dmul_rn(beta, value);
          value = generativeqc_tensor::finite(__dadd_rn(prior, product), error, 26);
          first = false;
        }
      }
    }
    c[offset] = value;
  }
}

enum class Mode { pairs, rows, batched };

struct Plan {
  Mode mode{Mode::pairs};
  std::size_t segments_per_batch{1};
};

inline constexpr std::size_t stacked_alignment = 256;

inline std::size_t align_up(std::size_t bytes) {
  const auto padded = posthf::checked_add(bytes, stacked_alignment - 1);
  return padded / stacked_alignment * stacked_alignment;
}

/** Scratch for `segments` stacked segments of `width` whose partials hold
 * `left_count` left limbs against all right limbs. */
inline std::size_t stacked_bytes(std::size_t m, std::size_t n, std::size_t width,
                                 std::size_t segments, std::size_t left_count) {
  using posthf::checked_add;
  using posthf::checked_mul;
  const auto digits = [&](std::size_t vectors) {
    const auto count = checked_mul(checked_mul(segments, limbs * width), vectors);
    return align_up(checked_mul(count, sizeof(float)));
  };
  const auto pairs = checked_mul(segments, left_count * limbs);
  const auto partial = align_up(checked_mul(checked_mul(pairs, checked_mul(m, n)), sizeof(float)));
  const auto scales = checked_mul(checked_mul(segments, limbs), checked_add(m, n));
  return checked_add(checked_add(digits(m), digits(n)),
                     checked_add(partial, checked_mul(scales, sizeof(int))));
}

/** Choose the fewest launches that fit the reserved scratch. Every schedule
 * forms the same digits and adds the same FP64 terms in the same order; the
 * per-pair schedule needs only workspace_bytes(m, n, k). */
inline Plan plan(std::size_t m, std::size_t n, std::size_t k, std::size_t limit) {
  constexpr auto int_limit = std::size_t(std::numeric_limits<int>::max());
  const auto cap = std::min(k, tile);
  const auto full = k / cap;
  if (std::max(m, n) > int_limit / limbs) return {};
  if (stacked_bytes(m, n, cap, 1, limbs) <= limit) {
    std::size_t lower = 1, upper = full;
    while (lower < upper) {
      const auto middle = upper - (upper - lower) / 2;
      if (stacked_bytes(m, n, cap, middle, limbs) <= limit)
        lower = middle;
      else
        upper = middle - 1;
    }
    return {Mode::batched, lower};
  }
  if (stacked_bytes(m, n, cap, 1, 1) <= limit) return {Mode::rows, 1};
  return {};
}

struct StackedScratch {
  float* left;
  float* right;
  float* partial;
  int* left_exponents;
  int* right_exponents;
};

inline StackedScratch stacked_scratch(void* scratch, std::size_t m, std::size_t n,
                                      std::size_t width, std::size_t segments,
                                      std::size_t left_count) {
  auto* base = static_cast<unsigned char*>(scratch);
  StackedScratch view{};
  view.left = reinterpret_cast<float*>(base);
  base += align_up(segments * limbs * width * m * sizeof(float));
  view.right = reinterpret_cast<float*>(base);
  base += align_up(segments * limbs * width * n * sizeof(float));
  view.partial = reinterpret_cast<float*>(base);
  base += align_up(segments * left_count * limbs * m * n * sizeof(float));
  view.left_exponents = reinterpret_cast<int*>(base);
  view.right_exponents = view.left_exponents + segments * limbs * m;
  return view;
}

/** One batch of equal-width segments in the stacked layout. */
inline std::size_t execute_stacked(const tensor::CudaContractionContext& provider,
                                   cudaStream_t stream, Mode mode, char ta, char tb, std::size_t m,
                                   std::size_t n, std::size_t first, std::size_t width,
                                   std::size_t segments, double alpha, const double* a,
                                   std::size_t lda, const double* b, std::size_t ldb, double beta,
                                   double* c, std::size_t ldc, void* scratch, int* error) {
  const auto left_count = mode == Mode::batched ? limbs : 1;
  const auto view = stacked_scratch(scratch, m, n, width, segments, left_count);
  const dim3 left_grid(launch_blocks(m), launch_blocks(segments));
  const dim3 right_grid(launch_blocks(n), launch_blocks(segments));
  pack_stacked<true><<<left_grid, tile, 0, stream>>>(a, ta == 'T', lda, first, width, m, segments,
                                                     view.left, view.left_exponents, error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
  pack_stacked<false><<<right_grid, tile, 0, stream>>>(b, tb == 'T', ldb, first, width, n, segments,
                                                       view.right, view.right_exponents, error);
  generativeqc_tensor::cuda_check(cudaGetLastError());
  const auto combine_blocks = launch_blocks((m * n + 255) / 256);
  if (mode == Mode::batched) {
    execute_batched_fp32_product(provider, stream, segments, limbs * m, limbs * n, width, view.left,
                                 view.right, view.partial, error, schedule);
    combine_stacked<<<combine_blocks, 256, 0, stream>>>(view.partial, view.left_exponents,
                                                        view.right_exponents, segments, 0, limbs,
                                                        alpha, beta, m, n, c, ldc, error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    return limbs * limbs * segments;
  }
  for (std::size_t i = 0; i < limbs; ++i) {
    execute_strided_fp32_product(provider, stream, 'N', 'N', m, limbs * n, width, 1.0f,
                                 view.left + i * m * width, width, view.right, limbs * n, 0.0f,
                                 view.partial, limbs * n, error, schedule);
    combine_stacked<<<combine_blocks, 256, 0, stream>>>(view.partial, view.left_exponents,
                                                        view.right_exponents, 1, i, 1, alpha,
                                                        i ? 1.0 : beta, m, n, c, ldc, error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
  }
  return limbs * limbs;
}

/** Issue nine real pedantic FP32 products for each K segment. Only the final
 * digit is fractional. FP64 restoration and addition keep all three digits
 * visible to the next contraction and to the outer adjoint accumulation.
 * Batched and row schedules stack these products into fewer SGEMMs inside
 * the same reserved scratch; the returned count is always nine per segment. */
inline std::size_t execute(const tensor::CudaContractionContext& provider, cudaStream_t stream,
                           char ta, char tb, std::size_t m, std::size_t n, std::size_t k,
                           double alpha, const double* a, std::size_t lda,
                           const double* b, std::size_t ldb, double beta,
                           double* c, std::size_t ldc, void* scratch, std::size_t limit, int* error) {
  if (workspace_bytes(m, n, k) > limit)
    throw std::length_error("compensated triples W scratch exceeds budget");
  if (!scratch || !a || !b || !c || !error || (ta != 'N' && ta != 'T') ||
      (tb != 'N' && tb != 'T') || lda < (ta == 'N' ? k : m) ||
      ldb < (tb == 'N' ? n : k) || ldc < n)
    throw std::invalid_argument("invalid compensated triples W binding");
  const auto cap = std::min(k, tile);
  if (const auto selected = plan(m, n, k, limit); selected.mode != Mode::pairs) {
    std::size_t calls = 0;
    const auto full = k / cap;
    for (std::size_t segment = 0; segment < full;) {
      const auto count = std::min(selected.segments_per_batch, full - segment);
      calls +=
          execute_stacked(provider, stream, selected.mode, ta, tb, m, n, segment * cap, cap, count,
                          alpha, a, lda, b, ldb, calls ? 1.0 : beta, c, ldc, scratch, error);
      segment += count;
    }
    if (const auto tail = k - full * cap)
      calls += execute_stacked(provider, stream, selected.mode, ta, tb, m, n, full * cap, tail, 1,
                               alpha, a, lda, b, ldb, calls ? 1.0 : beta, c, ldc, scratch, error);
    return calls;
  }
  const auto pitch_a = m * cap, pitch_b = n * cap;
  auto* digits_a = static_cast<float*>(scratch);
  auto* digits_b = digits_a + limbs * pitch_a;
  auto* partial = digits_b + limbs * pitch_b;
  auto* ea = reinterpret_cast<int*>(partial + m*n);
  auto* eb = ea + limbs * m;
  std::size_t calls = 0;
  for (std::size_t first = 0; first < k; first += cap) {
    const auto width = std::min(cap, k - first);
    pack<true><<<launch_blocks(m), tile, 0, stream>>>(
        a, ta == 'T', lda, first, width, m, digits_a, pitch_a, ea, error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    pack<false><<<launch_blocks(n), tile, 0, stream>>>(
        b, tb == 'T', ldb, first, width, n, digits_b, pitch_b, eb, error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    for (std::size_t i = 0; i < limbs; ++i) {
      for (std::size_t j = 0; j < limbs; ++j) {
        execute_strided_fp32_product(provider, stream, 'N', 'N', m, n, width, 1.0f,
                                     digits_a + i*pitch_a, width, digits_b + j*pitch_b, n,
                                     0.0f, partial, n, error, schedule);
        combine<<<launch_blocks((m*n+255)/256), 256, 0, stream>>>(
            partial, ea + i*m, eb + j*n, alpha, calls ? 1.0 : beta,
            m, n, c, ldc, error);
        generativeqc_tensor::cuda_check(cudaGetLastError());
        ++calls;
      }
    }
  }
  return calls;
}
}  // namespace compensated_w
}  // namespace generativeqc::cc::triples
