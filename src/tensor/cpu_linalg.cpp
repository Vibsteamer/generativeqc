#include "tensor/cpu_linalg.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>

#include "solver/cpu/symmetric_eigen.hpp"

#ifndef GENERATIVEQC_HAS_OPENBLAS
#define GENERATIVEQC_HAS_OPENBLAS 0
#endif
#ifndef GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
#define GENERATIVEQC_OPENBLAS_SCIPY_PREFIX 0
#endif
#ifndef GENERATIVEQC_OPENBLAS_HAS_LAPACKE
#define GENERATIVEQC_OPENBLAS_HAS_LAPACKE 0
#endif
#ifndef GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS
#define GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS 0
#endif
#ifndef GENERATIVEQC_OPENBLAS_LOCAL_THREADS_UNPREFIXED
#define GENERATIVEQC_OPENBLAS_LOCAL_THREADS_UNPREFIXED 0
#endif
#ifndef GENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS
#define GENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS 0
#endif

#if GENERATIVEQC_HAS_OPENBLAS
#include <cblas.h>
#if GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS && GENERATIVEQC_OPENBLAS_LOCAL_THREADS_UNPREFIXED
// Some prefixed distributions declare only the prefixed spelling while their
// shared library exports the ordinary extension. CMake verifies this exact ABI.
extern "C" int openblas_set_num_threads_local(int);
#endif
#if GENERATIVEQC_OPENBLAS_HAS_LAPACKE
#include "tensor/openblas_lapacke.hpp"
#endif
#endif

namespace generativeqc::tensor {
namespace {

using detail::split3_limbs;
using detail::split3_max_inner;

// An inner block has at most 64 products. A high or middle limb is a signed
// 8-bit magnitude (rounded up to 256) times one shared power of two. Thus each
// high/high, high/middle, middle/high and middle/middle partial sum contains
// at most 64 * 256^2 = 2^22 grid units, exactly representable in binary32.
#if GENERATIVEQC_HAS_OPENBLAS
bool split3_pack(const double* source, std::size_t input_stride, std::size_t count,
                 float* packed, std::size_t output_stride, std::size_t plane, int* exponents) {
  std::array<double, split3_max_inner> residual;
  for (std::size_t t = 0; t < count; ++t) {
    residual[t] = source[t * input_stride];
    if (!std::isfinite(residual[t])) throw std::domain_error("nonfinite split3 GEMM input");
  }
  for (std::size_t limb = 0; limb < split3_limbs; ++limb) {
    double largest = 0.0;
    for (std::size_t t = 0; t < count; ++t) largest = std::max(largest, std::abs(residual[t]));
    int exponent = 0;
    std::frexp(largest, &exponent);
    // Lifting the last limb by 60 bits leaves headroom for a sum of 64
    // products (<= 2^126). Its smallest admitted factors, 2^-63, still
    // multiply to a normal binary32 value. This prevents flush-to-zero
    // behavior from erasing a dot of tiny, but physically rescaled, factors.
    const bool last = limb + 1 == split3_limbs;
    constexpr int low_lift = 60;
    exponents[limb] = last ? exponent - low_lift : exponent;
    for (std::size_t t = 0; t < count; ++t) {
      const double normalized = std::scalbn(residual[t], -exponent);
      float value;
      if (!last) {
        value = static_cast<float>(std::nearbyint(normalized * 256.0) / 256.0);
        // Preserve tiny values for the next limb even if scaling rounded them to zero.
        if (value != 0.0f) residual[t] = std::scalbn(normalized - value, exponent);
      } else {
        value = static_cast<float>(std::scalbn(normalized, low_lift));
        if (residual[t] != 0.0 && std::abs(value) < 0x1p-63f)
          return false;
      }
      packed[limb * plane + t * output_stride] = value;
    }
  }
  return true;
}
#endif

bool transpose(char value) {
  if (value == 'N' || value == 'n') return false;
  if (value == 'T' || value == 't') return true;
  throw std::invalid_argument("CPU GEMM transpose must be N or T");
}

bool gemv_transpose(char value) {
  if (value == 'N' || value == 'n') return false;
  if (value == 'T' || value == 't') return true;
  throw std::invalid_argument("CPU GEMV transpose must be N or T");
}

bool syrk_transpose(char value) {
  if (value == 'N' || value == 'n') return false;
  if (value == 'T' || value == 't') return true;
  throw std::invalid_argument("CPU SYRK transpose must be N or T");
}

bool syr2k_transpose(char value) {
  if (value == 'N' || value == 'n') return false;
  if (value == 'T' || value == 't') return true;
  throw std::invalid_argument("CPU SYR2K transpose must be N or T");
}

bool symm_left_side(char value) {
  if (value == 'L' || value == 'l') return true;
  if (value == 'R' || value == 'r') return false;
  throw std::invalid_argument("CPU SYMM side must be L or R");
}

bool symm_upper_triangle(char value) {
  if (value == 'U' || value == 'u') return true;
  if (value == 'L' || value == 'l') return false;
  throw std::invalid_argument("CPU SYMM triangle must be U or L");
}

bool syr_upper_triangle(char value) {
  if (value == 'U' || value == 'u') return true;
  if (value == 'L' || value == 'l') return false;
  throw std::invalid_argument("CPU SYR triangle must be U or L");
}

bool syr2_upper_triangle(char value) {
  if (value == 'U' || value == 'u') return true;
  if (value == 'L' || value == 'l') return false;
  throw std::invalid_argument("CPU SYR2 triangle must be U or L");
}

bool upper_triangle(char value) {
  if (value == 'U' || value == 'u') return true;
  if (value == 'L' || value == 'l') return false;
  throw std::invalid_argument("CPU SYRK triangle must be U or L");
}

bool syr2k_upper_triangle(char value) {
  if (value == 'U' || value == 'u') return true;
  if (value == 'L' || value == 'l') return false;
  throw std::invalid_argument("CPU SYR2K triangle must be U or L");
}

bool trsm_left_side(char value) {
  if (value == 'L' || value == 'l') return true;
  if (value == 'R' || value == 'r') return false;
  throw std::invalid_argument("CPU TRSM side must be L or R");
}

bool trsm_upper_triangle(char value) {
  if (value == 'U' || value == 'u') return true;
  if (value == 'L' || value == 'l') return false;
  throw std::invalid_argument("CPU TRSM triangle must be U or L");
}

bool trsm_transpose(char value) {
  if (value == 'N' || value == 'n') return false;
  if (value == 'T' || value == 't') return true;
  throw std::invalid_argument("CPU TRSM transpose must be N or T");
}

bool trsm_unit_diagonal(char value) {
  if (value == 'U' || value == 'u') return true;
  if (value == 'N' || value == 'n') return false;
  throw std::invalid_argument("CPU TRSM diagonal must be U or N");
}

bool trmm_left_side(char value) {
  if (value == 'L' || value == 'l') return true;
  if (value == 'R' || value == 'r') return false;
  throw std::invalid_argument("CPU TRMM side must be L or R");
}

bool trmm_upper_triangle(char value) {
  if (value == 'U' || value == 'u') return true;
  if (value == 'L' || value == 'l') return false;
  throw std::invalid_argument("CPU TRMM triangle must be U or L");
}

bool trmm_transpose(char value) {
  if (value == 'N' || value == 'n') return false;
  if (value == 'T' || value == 't') return true;
  throw std::invalid_argument("CPU TRMM transpose must be N or T");
}

bool trmm_unit_diagonal(char value) {
  if (value == 'U' || value == 'u') return true;
  if (value == 'N' || value == 'n') return false;
  throw std::invalid_argument("CPU TRMM diagonal must be U or N");
}

void validate_plan(const CpuLinalgPlan& plan) {
  if (plan.provider_threads < 1)
    throw std::invalid_argument("CPU linear algebra thread count must be positive");
  if (plan.thread_ownership == CpuLinalgThreadOwnership::task_parallel &&
      plan.provider_threads != 1)
    throw std::invalid_argument("task-parallel CPU linear algebra requires single-thread provider");
}

std::size_t checked_matrix_elements(std::size_t rows, std::size_t columns) {
  if (columns && rows > std::numeric_limits<std::size_t>::max() / sizeof(double) / columns)
    throw std::length_error("CPU matrix storage extent overflows");
  return rows * columns;
}

void scalar_gemm(bool ta, bool tb, std::size_t m, std::size_t n, std::size_t k, const double* a,
                 const double* b, double* c, double alpha, double beta) {
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (std::size_t p = 0; p < k; ++p) {
        const double av = ta ? a[p * m + i] : a[i * k + p];
        const double bv = tb ? b[j * k + p] : b[p * n + j];
        sum += av * bv;
      }
      c[i * n + j] = beta == 0.0 ? alpha * sum : alpha * sum + beta * c[i * n + j];
    }
  }
}

void scalar_gemv(bool trans, std::size_t m, std::size_t n, const double* a, const double* x,
                 double* y, double alpha, double beta) {
  const std::size_t output_size = trans ? n : m;
  const std::size_t input_size = trans ? m : n;
  for (std::size_t i = 0; i < output_size; ++i) {
    double sum = 0.0;
    for (std::size_t j = 0; j < input_size; ++j) {
      const double av = trans ? a[j * n + i] : a[i * n + j];
      sum += av * x[j];
    }
    y[i] = beta == 0.0 ? alpha * sum : alpha * sum + beta * y[i];
  }
}

void scalar_ger(std::size_t m, std::size_t n, const double* x, const double* y, double* a,
                double alpha) {
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = 0; j < n; ++j) a[i * n + j] += alpha * x[i] * y[j];
}

void scalar_symm(bool left, bool upper, std::size_t m, std::size_t n, const double* a,
                 const double* b, double* c, double alpha, double beta) {
  const std::size_t order = left ? m : n;
  const auto symmetric_value = [&](std::size_t row, std::size_t column) {
    if (upper) {
      if (row > column) std::swap(row, column);
    } else if (row < column) {
      std::swap(row, column);
    }
    return a[row * order + column];
  };
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      if (left) {
        for (std::size_t p = 0; p < m; ++p) sum += symmetric_value(i, p) * b[p * n + j];
      } else {
        for (std::size_t p = 0; p < n; ++p) sum += b[i * n + p] * symmetric_value(p, j);
      }
      const std::size_t index = i * n + j;
      c[index] = beta == 0.0 ? alpha * sum : alpha * sum + beta * c[index];
    }
  }
}

void scalar_syr(bool upper, std::size_t n, const double* x, double* a, double alpha) {
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t first_column = upper ? i : 0;
    const std::size_t last_column = upper ? n : i + 1;
    for (std::size_t j = first_column; j < last_column; ++j) a[i * n + j] += alpha * x[i] * x[j];
  }
}

void scalar_syr2(bool upper, std::size_t n, const double* x, const double* y, double* a,
                 double alpha) {
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t first_column = upper ? i : 0;
    const std::size_t last_column = upper ? n : i + 1;
    // Scale first, as in DSYR2: unscaled products can overflow or underflow
    // even when the requested rank update is representable in binary64.
    const double scaled_x = alpha * x[i];
    const double scaled_y = alpha * y[i];
    for (std::size_t j = first_column; j < last_column; ++j)
      a[i * n + j] = a[i * n + j] + scaled_x * y[j] + scaled_y * x[j];
  }
}

void scalar_syrk(bool upper, bool trans, std::size_t n, std::size_t k, const double* a, double* c,
                 double alpha, double beta) {
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t first_column = upper ? i : 0;
    const std::size_t last_column = upper ? n : i + 1;
    for (std::size_t j = first_column; j < last_column; ++j) {
      double sum = 0.0;
      for (std::size_t p = 0; p < k; ++p) {
        const double ai = trans ? a[p * n + i] : a[i * k + p];
        const double aj = trans ? a[p * n + j] : a[j * k + p];
        sum += ai * aj;
      }
      const std::size_t index = i * n + j;
      c[index] = beta == 0.0 ? alpha * sum : alpha * sum + beta * c[index];
    }
  }
}

void scalar_syr2k(bool upper, bool trans, std::size_t n, std::size_t k, const double* a,
                  const double* b, double* c, double alpha, double beta) {
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t first_column = upper ? i : 0;
    const std::size_t last_column = upper ? n : i + 1;
    for (std::size_t j = first_column; j < last_column; ++j) {
      double sum = 0.0;
      for (std::size_t p = 0; p < k; ++p) {
        const double ai = trans ? a[p * n + i] : a[i * k + p];
        const double aj = trans ? a[p * n + j] : a[j * k + p];
        const double bi = trans ? b[p * n + i] : b[i * k + p];
        const double bj = trans ? b[p * n + j] : b[j * k + p];
        sum += ai * bj + bi * aj;
      }
      const std::size_t index = i * n + j;
      c[index] = beta == 0.0 ? alpha * sum : alpha * sum + beta * c[index];
    }
  }
}

void scalar_trsm(bool left, bool upper, bool trans, bool unit_diagonal, std::size_t m,
                 std::size_t n, const double* a, double* b, double alpha) {
  const std::size_t order = left ? m : n;
  const bool effective_upper = trans ? !upper : upper;
  const auto a_value = [&](std::size_t row, std::size_t column) {
    return trans ? a[column * order + row] : a[row * order + column];
  };

  if (alpha != 1.0)
    for (std::size_t index = 0; index < m * n; ++index) b[index] *= alpha;

  if (left) {
    for (std::size_t column = 0; column < n; ++column) {
      for (std::size_t step = 0; step < m; ++step) {
        const std::size_t row = effective_upper ? m - 1 - step : step;
        double value = b[row * n + column];
        if (effective_upper) {
          for (std::size_t p = row + 1; p < m; ++p) value -= a_value(row, p) * b[p * n + column];
        } else {
          for (std::size_t p = 0; p < row; ++p) value -= a_value(row, p) * b[p * n + column];
        }
        if (!unit_diagonal) value /= a_value(row, row);
        b[row * n + column] = value;
      }
    }
    return;
  }

  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t step = 0; step < n; ++step) {
      const std::size_t column = effective_upper ? step : n - 1 - step;
      double value = b[row * n + column];
      if (effective_upper) {
        for (std::size_t p = 0; p < column; ++p) value -= b[row * n + p] * a_value(p, column);
      } else {
        for (std::size_t p = column + 1; p < n; ++p) value -= b[row * n + p] * a_value(p, column);
      }
      if (!unit_diagonal) value /= a_value(column, column);
      b[row * n + column] = value;
    }
  }
}

void scalar_trmm(bool left, bool upper, bool trans, bool unit_diagonal, std::size_t m,
                 std::size_t n, const double* a, double* b, double alpha) {
  const std::size_t order = left ? m : n;
  const auto a_value = [&](std::size_t row, std::size_t column) {
    if (row == column && unit_diagonal) return 1.0;
    const std::size_t stored_row = trans ? column : row;
    const std::size_t stored_column = trans ? row : column;
    const bool stored = upper ? stored_column >= stored_row : stored_column <= stored_row;
    return stored ? a[stored_row * order + stored_column] : 0.0;
  };
  const std::vector<double> input(b, b + m * n);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      if (left) {
        for (std::size_t p = 0; p < m; ++p) sum += a_value(i, p) * input[p * n + j];
      } else {
        for (std::size_t p = 0; p < n; ++p) sum += input[i * n + p] * a_value(p, j);
      }
      b[i * n + j] = alpha * sum;
    }
  }
}

int scalar_cholesky_lower(double* matrix, std::size_t n) {
  for (std::size_t j = 0; j < n; ++j) {
    double diagonal = matrix[j * n + j];
    for (std::size_t p = 0; p < j; ++p) {
      const double value = matrix[j * n + p];
      diagonal -= value * value;
    }
    if (!(diagonal > 0.0) || !std::isfinite(diagonal)) return static_cast<int>(j + 1);
    const double root = std::sqrt(diagonal);
    matrix[j * n + j] = root;
    for (std::size_t i = j + 1; i < n; ++i) {
      double value = matrix[i * n + j];
      for (std::size_t p = 0; p < j; ++p) value -= matrix[i * n + p] * matrix[j * n + p];
      matrix[i * n + j] = value / root;
    }
  }
  return 0;
}

#if GENERATIVEQC_HAS_OPENBLAS
[[maybe_unused]] void openblas_set_local_threads(int threads) {
#if GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX && !GENERATIVEQC_OPENBLAS_LOCAL_THREADS_UNPREFIXED
  (void)scipy_openblas_set_num_threads_local(threads);
#else
  (void)openblas_set_num_threads_local(threads);
#endif
#else
  (void)threads;
#endif
}

[[maybe_unused]] int openblas_set_local_threads_return_previous(int threads) {
#if GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX && !GENERATIVEQC_OPENBLAS_LOCAL_THREADS_UNPREFIXED
  return scipy_openblas_set_num_threads_local(threads);
#else
  return openblas_set_num_threads_local(threads);
#endif
#else
  (void)threads;
  return 0;
#endif
}

[[maybe_unused]] int openblas_get_global_threads() {
#if GENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
  return scipy_openblas_get_num_threads();
#else
  return openblas_get_num_threads();
#endif
#else
  return 0;
#endif
}

[[maybe_unused]] void openblas_set_global_threads(int threads) {
#if GENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
  scipy_openblas_set_num_threads(threads);
#else
  openblas_set_num_threads(threads);
#endif
#else
  (void)threads;
#endif
}

[[maybe_unused]] std::mutex& openblas_global_thread_mutex() {
  static std::mutex mutex;
  return mutex;
}

class OpenBlasThreadGuard {
 public:
  explicit OpenBlasThreadGuard(const CpuLinalgPlan& plan) {
#if GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS
    previous_ = openblas_set_local_threads_return_previous(plan.provider_threads);
    local_ = true;
#elif GENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS
    if (plan.thread_ownership != CpuLinalgThreadOwnership::provider_parallel)
      throw std::runtime_error(
          "OpenBLAS build lacks thread-local control; use provider-parallel ownership");
    global_lock_ = std::unique_lock<std::mutex>(openblas_global_thread_mutex());
    previous_ = openblas_get_global_threads();
    if (previous_ != plan.provider_threads) {
      openblas_set_global_threads(plan.provider_threads);
      global_changed_ = true;
    }
#else
    (void)plan;
    throw std::runtime_error("OpenBLAS provider lacks runtime thread control");
#endif
  }
  OpenBlasThreadGuard(const OpenBlasThreadGuard&) = delete;
  OpenBlasThreadGuard& operator=(const OpenBlasThreadGuard&) = delete;
  ~OpenBlasThreadGuard() {
    if (local_) openblas_set_local_threads(previous_);
    if (global_changed_) openblas_set_global_threads(previous_);
  }

 private:
  int previous_{1};
  bool local_{};
  bool global_changed_{};
  std::unique_lock<std::mutex> global_lock_;
};

void openblas_gemm(bool ta, bool tb, std::size_t m, std::size_t n, std::size_t k, const double* a,
                   const double* b, double* c, double alpha, double beta,
                   const CpuLinalgPlan& plan) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (m > limit || n > limit || k > limit)
    throw std::length_error("OpenBLAS GEMM dimensions exceed int range");
  OpenBlasThreadGuard guard(plan);
  const auto trans_a = ta ? CblasTrans : CblasNoTrans;
  const auto trans_b = tb ? CblasTrans : CblasNoTrans;
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
  scipy_cblas_dgemm(CblasRowMajor, trans_a, trans_b, static_cast<int>(m), static_cast<int>(n),
                    static_cast<int>(k), alpha, a, static_cast<int>(ta ? m : k), b,
                    static_cast<int>(tb ? k : n), beta, c, static_cast<int>(n));
#else
  cblas_dgemm(CblasRowMajor, trans_a, trans_b, static_cast<int>(m), static_cast<int>(n),
              static_cast<int>(k), alpha, a, static_cast<int>(ta ? m : k), b,
              static_cast<int>(tb ? k : n), beta, c, static_cast<int>(n));
#endif
}

void openblas_gemv(bool trans, std::size_t m, std::size_t n, const double* a, const double* x,
                   double* y, double alpha, double beta, const CpuLinalgPlan& plan) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (m > limit || n > limit) throw std::length_error("OpenBLAS GEMV dimensions exceed int range");
  OpenBlasThreadGuard guard(plan);
  const auto transpose_a = trans ? CblasTrans : CblasNoTrans;
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
  scipy_cblas_dgemv(CblasRowMajor, transpose_a, static_cast<int>(m), static_cast<int>(n), alpha, a,
                    static_cast<int>(n), x, 1, beta, y, 1);
#else
  cblas_dgemv(CblasRowMajor, transpose_a, static_cast<int>(m), static_cast<int>(n), alpha, a,
              static_cast<int>(n), x, 1, beta, y, 1);
#endif
}

void openblas_ger(std::size_t m, std::size_t n, const double* x, const double* y, double* a,
                  double alpha, const CpuLinalgPlan& plan) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (m > limit || n > limit) throw std::length_error("OpenBLAS GER dimensions exceed int range");
  OpenBlasThreadGuard guard(plan);
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
  scipy_cblas_dger(CblasRowMajor, static_cast<int>(m), static_cast<int>(n), alpha, x, 1, y, 1, a,
                   static_cast<int>(n));
#else
  cblas_dger(CblasRowMajor, static_cast<int>(m), static_cast<int>(n), alpha, x, 1, y, 1, a,
             static_cast<int>(n));
#endif
}

void openblas_symm(bool left, bool upper, std::size_t m, std::size_t n, const double* a,
                   const double* b, double* c, double alpha, double beta,
                   const CpuLinalgPlan& plan) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (m > limit || n > limit) throw std::length_error("OpenBLAS SYMM dimensions exceed int range");
  OpenBlasThreadGuard guard(plan);
  const auto side = left ? CblasLeft : CblasRight;
  const auto triangle = upper ? CblasUpper : CblasLower;
  const int order = static_cast<int>(left ? m : n);
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
  scipy_cblas_dsymm(CblasRowMajor, side, triangle, static_cast<int>(m), static_cast<int>(n), alpha,
                    a, order, b, static_cast<int>(n), beta, c, static_cast<int>(n));
#else
  cblas_dsymm(CblasRowMajor, side, triangle, static_cast<int>(m), static_cast<int>(n), alpha, a,
              order, b, static_cast<int>(n), beta, c, static_cast<int>(n));
#endif
}

void openblas_syr(bool upper, std::size_t n, const double* x, double* a, double alpha,
                  const CpuLinalgPlan& plan) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (n > limit) throw std::length_error("OpenBLAS SYR dimension exceeds int range");
  OpenBlasThreadGuard guard(plan);
  const auto triangle = upper ? CblasUpper : CblasLower;
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
  scipy_cblas_dsyr(CblasRowMajor, triangle, static_cast<int>(n), alpha, x, 1, a,
                   static_cast<int>(n));
#else
  cblas_dsyr(CblasRowMajor, triangle, static_cast<int>(n), alpha, x, 1, a, static_cast<int>(n));
#endif
}

void openblas_syr2(bool upper, std::size_t n, const double* x, const double* y, double* a,
                   double alpha, const CpuLinalgPlan& plan) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (n > limit) throw std::length_error("OpenBLAS SYR2 dimension exceeds int range");
  OpenBlasThreadGuard guard(plan);
  const auto triangle = upper ? CblasUpper : CblasLower;
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
  scipy_cblas_dsyr2(CblasRowMajor, triangle, static_cast<int>(n), alpha, x, 1, y, 1, a,
                    static_cast<int>(n));
#else
  cblas_dsyr2(CblasRowMajor, triangle, static_cast<int>(n), alpha, x, 1, y, 1, a,
              static_cast<int>(n));
#endif
}

void openblas_syrk(bool upper, bool trans, std::size_t n, std::size_t k, const double* a, double* c,
                   double alpha, double beta, const CpuLinalgPlan& plan) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (n > limit || k > limit) throw std::length_error("OpenBLAS SYRK dimensions exceed int range");
  OpenBlasThreadGuard guard(plan);
  const auto triangle = upper ? CblasUpper : CblasLower;
  const auto transpose_a = trans ? CblasTrans : CblasNoTrans;
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
  scipy_cblas_dsyrk(CblasRowMajor, triangle, transpose_a, static_cast<int>(n), static_cast<int>(k),
                    alpha, a, static_cast<int>(trans ? n : k), beta, c, static_cast<int>(n));
#else
  cblas_dsyrk(CblasRowMajor, triangle, transpose_a, static_cast<int>(n), static_cast<int>(k), alpha,
              a, static_cast<int>(trans ? n : k), beta, c, static_cast<int>(n));
#endif
}

void openblas_syr2k(bool upper, bool trans, std::size_t n, std::size_t k, const double* a,
                    const double* b, double* c, double alpha, double beta,
                    const CpuLinalgPlan& plan) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (n > limit || k > limit) throw std::length_error("OpenBLAS SYR2K dimensions exceed int range");
  OpenBlasThreadGuard guard(plan);
  const auto triangle = upper ? CblasUpper : CblasLower;
  const auto transpose_ab = trans ? CblasTrans : CblasNoTrans;
  const int leading_dimension = static_cast<int>(trans ? n : k);
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
  scipy_cblas_dsyr2k(CblasRowMajor, triangle, transpose_ab, static_cast<int>(n),
                     static_cast<int>(k), alpha, a, leading_dimension, b, leading_dimension, beta,
                     c, static_cast<int>(n));
#else
  cblas_dsyr2k(CblasRowMajor, triangle, transpose_ab, static_cast<int>(n), static_cast<int>(k),
               alpha, a, leading_dimension, b, leading_dimension, beta, c, static_cast<int>(n));
#endif
}

void openblas_trsm(bool left, bool upper, bool trans, bool unit_diagonal, std::size_t m,
                   std::size_t n, const double* a, double* b, double alpha,
                   const CpuLinalgPlan& plan) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (m > limit || n > limit) throw std::length_error("OpenBLAS TRSM dimensions exceed int range");
  OpenBlasThreadGuard guard(plan);
  const auto side = left ? CblasLeft : CblasRight;
  const auto triangle = upper ? CblasUpper : CblasLower;
  const auto transpose_a = trans ? CblasTrans : CblasNoTrans;
  const auto diagonal = unit_diagonal ? CblasUnit : CblasNonUnit;
  const int order = static_cast<int>(left ? m : n);
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
  scipy_cblas_dtrsm(CblasRowMajor, side, triangle, transpose_a, diagonal, static_cast<int>(m),
                    static_cast<int>(n), alpha, a, order, b, static_cast<int>(n));
#else
  cblas_dtrsm(CblasRowMajor, side, triangle, transpose_a, diagonal, static_cast<int>(m),
              static_cast<int>(n), alpha, a, order, b, static_cast<int>(n));
#endif
}

void openblas_trmm(bool left, bool upper, bool trans, bool unit_diagonal, std::size_t m,
                   std::size_t n, const double* a, double* b, double alpha,
                   const CpuLinalgPlan& plan) {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (m > limit || n > limit) throw std::length_error("OpenBLAS TRMM dimensions exceed int range");
  OpenBlasThreadGuard guard(plan);
  const auto side = left ? CblasLeft : CblasRight;
  const auto triangle = upper ? CblasUpper : CblasLower;
  const auto transpose_a = trans ? CblasTrans : CblasNoTrans;
  const auto diagonal = unit_diagonal ? CblasUnit : CblasNonUnit;
  const int order = static_cast<int>(left ? m : n);
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
  scipy_cblas_dtrmm(CblasRowMajor, side, triangle, transpose_a, diagonal, static_cast<int>(m),
                    static_cast<int>(n), alpha, a, order, b, static_cast<int>(n));
#else
  cblas_dtrmm(CblasRowMajor, side, triangle, transpose_a, diagonal, static_cast<int>(m),
              static_cast<int>(n), alpha, a, order, b, static_cast<int>(n));
#endif
}

int openblas_cholesky_lower(double* matrix, std::size_t n, const CpuLinalgPlan& plan) {
#if GENERATIVEQC_OPENBLAS_HAS_LAPACKE
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("OpenBLAS Cholesky dimension exceeds int range");
  OpenBlasThreadGuard guard(plan);
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
  return static_cast<int>(scipy_LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', static_cast<int>(n), matrix,
                                               static_cast<int>(n)));
#else
  return static_cast<int>(
      LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', static_cast<int>(n), matrix, static_cast<int>(n)));
#endif
#else
  (void)matrix;
  (void)n;
  (void)plan;
  throw std::runtime_error("OpenBLAS provider was built without LAPACKE");
#endif
}

CpuSymmetricEigenResult openblas_symmetric_eigen(std::vector<double> matrix, std::size_t n,
                                                 const CpuLinalgPlan& plan) {
#if GENERATIVEQC_OPENBLAS_HAS_LAPACKE
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("OpenBLAS eigensolver dimension exceeds int range");
  std::vector<double> values(n);
  OpenBlasThreadGuard guard(plan);
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
  const int info =
      static_cast<int>(scipy_LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', static_cast<int>(n),
                                            matrix.data(), static_cast<int>(n), values.data()));
#else
  const int info =
      static_cast<int>(LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', static_cast<int>(n),
                                      matrix.data(), static_cast<int>(n), values.data()));
#endif
  if (info < 0) throw std::invalid_argument("OpenBLAS symmetric eigensolver rejected an argument");
  if (info > 0) throw std::runtime_error("OpenBLAS symmetric eigensolver did not converge");
  return {std::move(values), std::move(matrix)};
#else
  (void)matrix;
  (void)n;
  (void)plan;
  throw std::runtime_error("OpenBLAS provider was built without LAPACKE");
#endif
}
#endif

bool fits_openblas(std::size_t m, std::size_t n, std::size_t k) noexcept {
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  return m <= limit && n <= limit && k <= limit;
}

}  // namespace

bool cpu_openblas_built() noexcept { return GENERATIVEQC_HAS_OPENBLAS != 0; }
bool cpu_openblas_lapack_built() noexcept {
  return GENERATIVEQC_HAS_OPENBLAS != 0 && GENERATIVEQC_OPENBLAS_HAS_LAPACKE != 0;
}
bool cpu_openblas_local_thread_control_built() noexcept {
  return GENERATIVEQC_HAS_OPENBLAS != 0 && GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS != 0;
}
bool cpu_openblas_global_thread_control_built() noexcept {
  return GENERATIVEQC_HAS_OPENBLAS != 0 && GENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS != 0;
}

CpuLinalgProvider resolve_cpu_linalg_provider(const CpuLinalgPlan& plan, bool require_lapack) {
  validate_plan(plan);
  if (plan.provider == CpuLinalgProvider::scalar) return CpuLinalgProvider::scalar;
  const bool thread_control = plan.thread_ownership == CpuLinalgThreadOwnership::task_parallel
                                  ? cpu_openblas_local_thread_control_built()
                                  : (cpu_openblas_local_thread_control_built() ||
                                     cpu_openblas_global_thread_control_built());
  const bool usable =
      cpu_openblas_built() && thread_control && (!require_lapack || cpu_openblas_lapack_built());
  if (plan.provider == CpuLinalgProvider::openblas) {
    if (!usable)
      throw std::runtime_error(
          require_lapack
              ? "requested OpenBLAS LAPACK provider is unavailable"
              : "requested OpenBLAS provider is unavailable for this thread-ownership mode");
    return CpuLinalgProvider::openblas;
  }
  return usable ? CpuLinalgProvider::openblas : CpuLinalgProvider::scalar;
}

std::string_view cpu_linalg_target_name() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX512F__) && defined(__FMA__)
  return "x86_64-avx512f-fma";
#elif defined(__AVX2__) && defined(__FMA__)
  return "x86_64-avx2-fma";
#elif defined(__AVX2__)
  return "x86_64-avx2";
#else
  return "x86_64-generic";
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
  return "aarch64-generic";
#else
  return "generic";
#endif
}

std::string_view cpu_linalg_provider_name(CpuLinalgProvider provider) noexcept {
  switch (provider) {
    case CpuLinalgProvider::automatic:
      return "automatic";
    case CpuLinalgProvider::scalar:
      return "scalar";
    case CpuLinalgProvider::openblas:
      return "openblas";
  }
  return "unknown";
}

CpuLinalgDiagnostic cpu_linalg_diagnostic(const CpuLinalgPlan& plan) {
  const auto provider = resolve_cpu_linalg_provider(plan);
  return {.provider = provider,
          .thread_ownership = plan.thread_ownership,
          .provider_threads = plan.provider_threads,
          .external_provider = provider == CpuLinalgProvider::openblas,
          .lapack_available = cpu_openblas_lapack_built(),
          .local_thread_control = cpu_openblas_local_thread_control_built(),
          .global_thread_control = cpu_openblas_global_thread_control_built(),
          .cpu_target = cpu_linalg_target_name()};
}

void cpu_gemm(char a_trans, char b_trans, std::size_t m, std::size_t n, std::size_t k,
              const double* a, const double* b, double* c, double alpha, double beta,
              const CpuLinalgPlan& plan) {
  const bool ta = transpose(a_trans);
  const bool tb = transpose(b_trans);
  validate_plan(plan);
  if (!m || !n) return;
  const auto elements = checked_matrix_elements(m, n);
  if (!c) throw std::invalid_argument("CPU GEMM received null storage");
  if (!k || alpha == 0.0) {
    if (beta == 0.0)
      std::fill(c, c + elements, 0.0);
    else if (beta != 1.0)
      for (std::size_t i = 0; i < elements; ++i) c[i] *= beta;
    return;
  }
  checked_matrix_elements(m, k);
  checked_matrix_elements(k, n);
  if (!a || !b) throw std::invalid_argument("CPU GEMM received null storage");

  CpuLinalgProvider provider = plan.provider;
  if (provider == CpuLinalgProvider::automatic) {
    provider =
        fits_openblas(m, n, k) ? resolve_cpu_linalg_provider(plan) : CpuLinalgProvider::scalar;
  } else {
    provider = resolve_cpu_linalg_provider(plan);
  }
#if GENERATIVEQC_HAS_OPENBLAS
  if (provider == CpuLinalgProvider::openblas) {
    openblas_gemm(ta, tb, m, n, k, a, b, c, alpha, beta, plan);
    return;
  }
#endif
  scalar_gemm(ta, tb, m, n, k, a, b, c, alpha, beta);
}

std::size_t cpu_split3_gemm(char a_trans, char b_trans, std::size_t m, std::size_t n,
                            std::size_t k, const double* a, const double* b, double* c,
                            double alpha, double beta, const CpuLinalgPlan& plan) {
  const bool ta = transpose(a_trans);
  const bool tb = transpose(b_trans);
  validate_plan(plan);
  if (!std::isfinite(alpha) || !std::isfinite(beta))
    throw std::domain_error("nonfinite split3 GEMM coefficient");
  if (!m || !n) return 0;
  [[maybe_unused]] const auto output = checked_matrix_elements(m, n);
  if (!c) throw std::invalid_argument("split3 GEMM received null output");
  if (!k || alpha == 0.0) {
    cpu_gemm(a_trans, b_trans, m, n, 0, nullptr, nullptr, c, alpha, beta, plan);
    return 0;
  }
  checked_matrix_elements(m, k);
  checked_matrix_elements(k, n);
  if (!a || !b) throw std::invalid_argument("split3 GEMM received null input");
  if (resolve_cpu_linalg_provider(plan) != CpuLinalgProvider::openblas)
    throw std::runtime_error("split3 GEMM requires an available OpenBLAS SGEMM provider");
#if GENERATIVEQC_HAS_OPENBLAS
  if (!fits_openblas(m, n, split3_max_inner))
    throw std::length_error("split3 GEMM dimensions exceed OpenBLAS int range");
  (void)cpu_split3_gemm_workspace_bytes(m, n, k);
  const auto max_inner = std::min(k, split3_max_inner);
  const auto a_plane = m * max_inner;
  const auto b_plane = n * max_inner;
  std::vector<float> a_parts(split3_limbs * a_plane), b_parts(split3_limbs * b_plane);
  std::vector<float> partial(output);
  std::vector<int> a_exponents(split3_limbs * m), b_exponents(split3_limbs * n);
  std::vector<double> result(output);
  if (beta != 0.0)
    for (std::size_t i = 0; i < output; ++i) result[i] = beta * c[i];
  int alpha_exponent = 0;
  const double alpha_fraction = std::frexp(alpha, &alpha_exponent);
  OpenBlasThreadGuard guard(plan);
  const auto pack_block = [&](std::size_t start, std::size_t inner) {
    for (std::size_t i = 0; i < m; ++i) {
      if (!split3_pack(a + (ta ? start * m + i : i * k + start), ta ? m : 1, inner,
                       a_parts.data() + i * inner, 1, a_plane, a_exponents.data() + i * split3_limbs))
        return false;
    }
    for (std::size_t j = 0; j < n; ++j) {
      if (!split3_pack(b + (tb ? j * k + start : start * n + j), tb ? 1 : n, inner,
                       b_parts.data() + j, n, b_plane, b_exponents.data() + j * split3_limbs))
        return false;
    }
    return true;
  };
  std::size_t calls = 0;
  for (std::size_t start = 0; start < k;) {
    auto inner = std::min(split3_max_inner, k - start);
    // A wide residual exponent spread can make a block unsafe for FP32.
    // Repack the left prefix until every nonzero low limb is representable.
    // At a singleton its own exponent normalizes it, even for FP64 subnormals.
    while (!pack_block(start, inner)) {
      if (inner == 1) throw std::logic_error("split3 GEMM failed to normalize a finite singleton");
      inner = (inner + 1) / 2;
    }
    for (std::size_t left = 0; left < split3_limbs; ++left) {
      for (std::size_t right = 0; right < split3_limbs; ++right) {
#if GENERATIVEQC_OPENBLAS_SCIPY_PREFIX
        scipy_cblas_sgemm(
#else
        cblas_sgemm(
#endif
            CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(m), static_cast<int>(n),
            static_cast<int>(inner), 1.0f, a_parts.data() + left * a_plane, static_cast<int>(inner),
            b_parts.data() + right * b_plane, static_cast<int>(n), 0.0f, partial.data(),
            static_cast<int>(n));
        for (std::size_t i = 0; i < m; ++i) {
          for (std::size_t j = 0; j < n; ++j) {
            const auto offset = i * n + j;
            const int exponent = a_exponents[i * split3_limbs + left] +
                                 b_exponents[j * split3_limbs + right];
            result[offset] += std::scalbn(alpha_fraction * static_cast<double>(partial[offset]),
                                          exponent + alpha_exponent);
          }
        }
      }
    }
    calls = runtime::size_add(calls, split3_limbs * split3_limbs);
    start += inner;
  }
  if (!std::all_of(result.begin(), result.end(), [](double value) { return std::isfinite(value); }))
    throw std::overflow_error("split3 GEMM result exceeds FP64 range");
  std::copy(result.begin(), result.end(), c);
  return calls;
#else
  (void)ta;
  (void)tb;
  throw std::runtime_error("split3 GEMM requires an OpenBLAS build");
#endif
}

void cpu_congruence(char coefficient_transpose, std::size_t n, const double* coefficients,
                    const double* matrix, double* result, double* workspace,
                    const CpuLinalgPlan& plan) {
  bool transposed = false;
  if (coefficient_transpose == 'T' || coefficient_transpose == 't')
    transposed = true;
  else if (coefficient_transpose != 'N' && coefficient_transpose != 'n')
    throw std::invalid_argument("CPU congruence transpose must be N or T");
  validate_plan(plan);
  if (!n) return;
  checked_matrix_elements(n, n);
  if (!coefficients || !matrix || !result || !workspace)
    throw std::invalid_argument("CPU congruence received null storage");
  if (workspace == coefficients || workspace == matrix || workspace == result ||
      result == coefficients || result == matrix)
    throw std::invalid_argument("CPU congruence requires non-aliasing input/output/workspace");

  if (transposed) {
    cpu_gemm('N', 'N', n, n, n, matrix, coefficients, workspace, 1.0, 0.0, plan);
    cpu_gemm('T', 'N', n, n, n, coefficients, workspace, result, 1.0, 0.0, plan);
  } else {
    cpu_gemm('N', 'N', n, n, n, coefficients, matrix, workspace, 1.0, 0.0, plan);
    cpu_gemm('N', 'T', n, n, n, workspace, coefficients, result, 1.0, 0.0, plan);
  }
}

void cpu_gemv(char trans, std::size_t m, std::size_t n, const double* a, const double* x, double* y,
              double alpha, double beta, const CpuLinalgPlan& plan) {
  const bool transposed = gemv_transpose(trans);
  validate_plan(plan);
  const std::size_t output_size = transposed ? n : m;
  const std::size_t input_size = transposed ? m : n;
  if (!output_size) return;
  checked_matrix_elements(output_size, 1);
  if (!y) throw std::invalid_argument("CPU GEMV received null output storage");
  if (!input_size || alpha == 0.0) {
    if (beta == 0.0)
      std::fill(y, y + output_size, 0.0);
    else if (beta != 1.0)
      for (std::size_t i = 0; i < output_size; ++i) y[i] *= beta;
    return;
  }
  checked_matrix_elements(m, n);
  if (!a || !x) throw std::invalid_argument("CPU GEMV received null input storage");

  CpuLinalgProvider provider = plan.provider;
  if (provider == CpuLinalgProvider::automatic) {
    provider =
        fits_openblas(m, n, 1) ? resolve_cpu_linalg_provider(plan) : CpuLinalgProvider::scalar;
  } else {
    provider = resolve_cpu_linalg_provider(plan);
  }
#if GENERATIVEQC_HAS_OPENBLAS
  if (provider == CpuLinalgProvider::openblas) {
    openblas_gemv(transposed, m, n, a, x, y, alpha, beta, plan);
    return;
  }
#endif
  scalar_gemv(transposed, m, n, a, x, y, alpha, beta);
}

void cpu_ger(std::size_t m, std::size_t n, const double* x, const double* y, double* a,
             double alpha, const CpuLinalgPlan& plan) {
  validate_plan(plan);
  if (!m || !n) return;
  checked_matrix_elements(m, n);
  if (alpha == 0.0) return;
  if (!x || !y || !a) throw std::invalid_argument("CPU GER received null storage");

  CpuLinalgProvider provider = plan.provider;
  if (provider == CpuLinalgProvider::automatic) {
    provider =
        fits_openblas(m, n, 1) ? resolve_cpu_linalg_provider(plan) : CpuLinalgProvider::scalar;
  } else {
    provider = resolve_cpu_linalg_provider(plan);
  }
#if GENERATIVEQC_HAS_OPENBLAS
  if (provider == CpuLinalgProvider::openblas) {
    openblas_ger(m, n, x, y, a, alpha, plan);
    return;
  }
#endif
  scalar_ger(m, n, x, y, a, alpha);
}

void cpu_symm(char side, char uplo, std::size_t m, std::size_t n, const double* a, const double* b,
              double* c, double alpha, double beta, const CpuLinalgPlan& plan) {
  const bool left = symm_left_side(side);
  const bool upper = symm_upper_triangle(uplo);
  validate_plan(plan);
  if (!m || !n) return;
  const auto elements = checked_matrix_elements(m, n);
  if (!c) throw std::invalid_argument("CPU SYMM received null output storage");
  if (alpha == 0.0) {
    if (beta == 0.0)
      std::fill(c, c + elements, 0.0);
    else if (beta != 1.0)
      for (std::size_t i = 0; i < elements; ++i) c[i] *= beta;
    return;
  }
  const std::size_t order = left ? m : n;
  checked_matrix_elements(order, order);
  if (!a || !b) throw std::invalid_argument("CPU SYMM received null input storage");

  CpuLinalgProvider provider = plan.provider;
  if (provider == CpuLinalgProvider::automatic) {
    provider =
        fits_openblas(m, n, order) ? resolve_cpu_linalg_provider(plan) : CpuLinalgProvider::scalar;
  } else {
    provider = resolve_cpu_linalg_provider(plan);
  }
#if GENERATIVEQC_HAS_OPENBLAS
  if (provider == CpuLinalgProvider::openblas) {
    openblas_symm(left, upper, m, n, a, b, c, alpha, beta, plan);
    return;
  }
#endif
  scalar_symm(left, upper, m, n, a, b, c, alpha, beta);
}

void cpu_syr(char uplo, std::size_t n, const double* x, double* a, double alpha,
             const CpuLinalgPlan& plan) {
  const bool upper = syr_upper_triangle(uplo);
  validate_plan(plan);
  if (!n) return;
  checked_matrix_elements(n, n);
  if (alpha == 0.0) return;
  if (!x || !a) throw std::invalid_argument("CPU SYR received null storage");

  CpuLinalgProvider provider = plan.provider;
  if (provider == CpuLinalgProvider::automatic) {
    provider =
        fits_openblas(n, n, 1) ? resolve_cpu_linalg_provider(plan) : CpuLinalgProvider::scalar;
  } else {
    provider = resolve_cpu_linalg_provider(plan);
  }
#if GENERATIVEQC_HAS_OPENBLAS
  if (provider == CpuLinalgProvider::openblas) {
    openblas_syr(upper, n, x, a, alpha, plan);
    return;
  }
#endif
  scalar_syr(upper, n, x, a, alpha);
}

void cpu_syr2(char uplo, std::size_t n, const double* x, const double* y, double* a, double alpha,
              const CpuLinalgPlan& plan) {
  const bool upper = syr2_upper_triangle(uplo);
  validate_plan(plan);
  if (!n) return;
  checked_matrix_elements(n, n);
  if (alpha == 0.0) return;
  if (!x || !y || !a) throw std::invalid_argument("CPU SYR2 received null storage");

  CpuLinalgProvider provider = plan.provider;
  if (provider == CpuLinalgProvider::automatic) {
    provider =
        fits_openblas(n, n, 1) ? resolve_cpu_linalg_provider(plan) : CpuLinalgProvider::scalar;
  } else {
    provider = resolve_cpu_linalg_provider(plan);
  }
#if GENERATIVEQC_HAS_OPENBLAS
  if (provider == CpuLinalgProvider::openblas) {
    openblas_syr2(upper, n, x, y, a, alpha, plan);
    return;
  }
#endif
  scalar_syr2(upper, n, x, y, a, alpha);
}

void cpu_syrk(char uplo, char trans, std::size_t n, std::size_t k, const double* a, double* c,
              double alpha, double beta, const CpuLinalgPlan& plan) {
  const bool upper = upper_triangle(uplo);
  const bool transposed = syrk_transpose(trans);
  validate_plan(plan);
  if (!n) return;
  checked_matrix_elements(n, n);
  if (!c) throw std::invalid_argument("CPU SYRK received null storage");
  if (!k || alpha == 0.0) {
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t first_column = upper ? i : 0;
      const std::size_t last_column = upper ? n : i + 1;
      for (std::size_t j = first_column; j < last_column; ++j) {
        const std::size_t index = i * n + j;
        if (beta == 0.0)
          c[index] = 0.0;
        else if (beta != 1.0)
          c[index] *= beta;
      }
    }
    return;
  }
  checked_matrix_elements(transposed ? k : n, transposed ? n : k);
  if (!a) throw std::invalid_argument("CPU SYRK received null storage");

  CpuLinalgProvider provider = plan.provider;
  if (provider == CpuLinalgProvider::automatic) {
    provider =
        fits_openblas(n, n, k) ? resolve_cpu_linalg_provider(plan) : CpuLinalgProvider::scalar;
  } else {
    provider = resolve_cpu_linalg_provider(plan);
  }
#if GENERATIVEQC_HAS_OPENBLAS
  if (provider == CpuLinalgProvider::openblas) {
    openblas_syrk(upper, transposed, n, k, a, c, alpha, beta, plan);
    return;
  }
#endif
  scalar_syrk(upper, transposed, n, k, a, c, alpha, beta);
}

void cpu_syr2k(char uplo, char trans, std::size_t n, std::size_t k, const double* a,
               const double* b, double* c, double alpha, double beta, const CpuLinalgPlan& plan) {
  const bool upper = syr2k_upper_triangle(uplo);
  const bool transposed = syr2k_transpose(trans);
  validate_plan(plan);
  if (!n) return;
  checked_matrix_elements(n, n);
  if (!c) throw std::invalid_argument("CPU SYR2K received null output storage");
  if (!k || alpha == 0.0) {
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t first_column = upper ? i : 0;
      const std::size_t last_column = upper ? n : i + 1;
      for (std::size_t j = first_column; j < last_column; ++j) {
        const std::size_t index = i * n + j;
        if (beta == 0.0)
          c[index] = 0.0;
        else if (beta != 1.0)
          c[index] *= beta;
      }
    }
    return;
  }
  checked_matrix_elements(transposed ? k : n, transposed ? n : k);
  if (!a || !b) throw std::invalid_argument("CPU SYR2K received null input storage");

  CpuLinalgProvider provider = plan.provider;
  if (provider == CpuLinalgProvider::automatic) {
    provider =
        fits_openblas(n, n, k) ? resolve_cpu_linalg_provider(plan) : CpuLinalgProvider::scalar;
  } else {
    provider = resolve_cpu_linalg_provider(plan);
  }
#if GENERATIVEQC_HAS_OPENBLAS
  if (provider == CpuLinalgProvider::openblas) {
    openblas_syr2k(upper, transposed, n, k, a, b, c, alpha, beta, plan);
    return;
  }
#endif
  scalar_syr2k(upper, transposed, n, k, a, b, c, alpha, beta);
}

void cpu_trsm(char side, char uplo, char trans, char diag, std::size_t m, std::size_t n,
              const double* a, double* b, double alpha, const CpuLinalgPlan& plan) {
  const bool left = trsm_left_side(side);
  const bool upper = trsm_upper_triangle(uplo);
  const bool transposed = trsm_transpose(trans);
  const bool unit_diagonal = trsm_unit_diagonal(diag);
  validate_plan(plan);
  if (!m || !n) return;
  const auto elements = checked_matrix_elements(m, n);
  if (!b) throw std::invalid_argument("CPU TRSM received null output storage");
  if (alpha == 0.0) {
    std::fill(b, b + elements, 0.0);
    return;
  }
  const std::size_t order = left ? m : n;
  checked_matrix_elements(order, order);
  if (!a) throw std::invalid_argument("CPU TRSM received null triangular storage");

  CpuLinalgProvider provider = plan.provider;
  if (provider == CpuLinalgProvider::automatic) {
    provider =
        fits_openblas(m, n, order) ? resolve_cpu_linalg_provider(plan) : CpuLinalgProvider::scalar;
  } else {
    provider = resolve_cpu_linalg_provider(plan);
  }
#if GENERATIVEQC_HAS_OPENBLAS
  if (provider == CpuLinalgProvider::openblas) {
    openblas_trsm(left, upper, transposed, unit_diagonal, m, n, a, b, alpha, plan);
    return;
  }
#endif
  scalar_trsm(left, upper, transposed, unit_diagonal, m, n, a, b, alpha);
}

void cpu_trmm(char side, char uplo, char trans, char diag, std::size_t m, std::size_t n,
              const double* a, double* b, double alpha, const CpuLinalgPlan& plan) {
  const bool left = trmm_left_side(side);
  const bool upper = trmm_upper_triangle(uplo);
  const bool transposed = trmm_transpose(trans);
  const bool unit_diagonal = trmm_unit_diagonal(diag);
  validate_plan(plan);
  if (!m || !n) return;
  const auto elements = checked_matrix_elements(m, n);
  if (!b) throw std::invalid_argument("CPU TRMM received null output storage");
  if (alpha == 0.0) {
    std::fill(b, b + elements, 0.0);
    return;
  }
  const std::size_t order = left ? m : n;
  checked_matrix_elements(order, order);
  if (!a) throw std::invalid_argument("CPU TRMM received null triangular storage");

  CpuLinalgProvider provider = plan.provider;
  if (provider == CpuLinalgProvider::automatic) {
    provider =
        fits_openblas(m, n, order) ? resolve_cpu_linalg_provider(plan) : CpuLinalgProvider::scalar;
  } else {
    provider = resolve_cpu_linalg_provider(plan);
  }
#if GENERATIVEQC_HAS_OPENBLAS
  if (provider == CpuLinalgProvider::openblas) {
    openblas_trmm(left, upper, transposed, unit_diagonal, m, n, a, b, alpha, plan);
    return;
  }
#endif
  scalar_trmm(left, upper, transposed, unit_diagonal, m, n, a, b, alpha);
}

int cpu_cholesky_lower(double* matrix, std::size_t n, const CpuLinalgPlan& plan) {
  validate_plan(plan);
  if (!n) return 0;
  checked_matrix_elements(n, n);
  if (!matrix) throw std::invalid_argument("CPU Cholesky received null storage");
#if GENERATIVEQC_HAS_OPENBLAS
  if (resolve_cpu_linalg_provider(plan, true) == CpuLinalgProvider::openblas)
    return openblas_cholesky_lower(matrix, n, plan);
#else
  (void)resolve_cpu_linalg_provider(plan, true);
#endif
  return scalar_cholesky_lower(matrix, n);
}

CpuSymmetricEigenResult cpu_symmetric_eigen(std::vector<double> matrix, std::size_t n,
                                            const CpuLinalgPlan& plan) {
  return cpu_symmetric_eigen(std::move(matrix), n, plan, 0.0);
}

CpuSymmetricEigenResult cpu_symmetric_eigen(std::vector<double> matrix, std::size_t n,
                                            const CpuLinalgPlan& plan,
                                            double absolute_off_diagonal_tolerance) {
  validate_plan(plan);
  if (!std::isfinite(absolute_off_diagonal_tolerance) || absolute_off_diagonal_tolerance < 0.0)
    throw std::invalid_argument(
        "CPU symmetric eigensolver absolute tolerance must be finite and nonnegative");
  if (absolute_off_diagonal_tolerance > 0.0 && plan.provider != CpuLinalgProvider::scalar)
    throw std::invalid_argument(
        "CPU symmetric eigensolver absolute tolerance requires explicit scalar provider");
  if (!n || n > std::numeric_limits<std::size_t>::max() / n || matrix.size() != n * n)
    throw std::invalid_argument("CPU symmetric eigensolver dimensions are inconsistent");
  if (!std::all_of(matrix.begin(), matrix.end(), [](double x) { return std::isfinite(x); }))
    throw std::invalid_argument("CPU symmetric eigensolver requires finite input");
#if GENERATIVEQC_HAS_OPENBLAS
  if (resolve_cpu_linalg_provider(plan, true) == CpuLinalgProvider::openblas) {
    const auto prepared = solver::cpu::prepare_owned_symmetric_eigen(
        n, solver::cpu::SymmetricEigenFamily::lapack_owned_row_major);
    return solver::cpu::execute_symmetric_eigen(
        prepared, std::move(matrix),
        [](std::vector<double> input, std::size_t order, const void* context) {
          return openblas_symmetric_eigen(std::move(input), order,
                                          *static_cast<const CpuLinalgPlan*>(context));
        },
        &plan);
  }
#else
  (void)resolve_cpu_linalg_provider(plan, true);
#endif
  const auto prepared = solver::cpu::prepare_owned_symmetric_eigen(
      n, solver::cpu::SymmetricEigenFamily::scalar_owned_row_major,
      absolute_off_diagonal_tolerance);
  return solver::cpu::execute_symmetric_eigen(prepared, std::move(matrix));
}

}  // namespace generativeqc::tensor
