#ifndef GENERATIVEQC_TENSOR_CPU_LINALG_HPP
#define GENERATIVEQC_TENSOR_CPU_LINALG_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "runtime/bounded_workspace.hpp"

namespace generativeqc::tensor {

enum class CpuLinalgProvider : std::uint8_t { automatic, scalar, openblas };
enum class CpuLinalgThreadOwnership : std::uint8_t { task_parallel, provider_parallel };

struct CpuLinalgPlan {
  CpuLinalgProvider provider{CpuLinalgProvider::automatic};
  CpuLinalgThreadOwnership thread_ownership{CpuLinalgThreadOwnership::task_parallel};
  int provider_threads{1};
};

struct CpuLinalgDiagnostic {
  CpuLinalgProvider provider{CpuLinalgProvider::scalar};
  CpuLinalgThreadOwnership thread_ownership{CpuLinalgThreadOwnership::task_parallel};
  int provider_threads{1};
  bool external_provider{};
  bool lapack_available{};
  bool local_thread_control{};
  bool global_thread_control{};
  // GenerativeQC host-code build target. External BLAS may dispatch internally.
  std::string_view cpu_target{};
};

struct CpuSymmetricEigenResult {
  std::vector<double> values;
  std::vector<double> vectors;  // row-major matrix, eigenvectors in columns
};

[[nodiscard]] bool cpu_openblas_built() noexcept;
[[nodiscard]] bool cpu_openblas_lapack_built() noexcept;
[[nodiscard]] bool cpu_openblas_local_thread_control_built() noexcept;
[[nodiscard]] bool cpu_openblas_global_thread_control_built() noexcept;
[[nodiscard]] CpuLinalgProvider resolve_cpu_linalg_provider(const CpuLinalgPlan& plan,
                                                            bool require_lapack = false);
[[nodiscard]] CpuLinalgDiagnostic cpu_linalg_diagnostic(const CpuLinalgPlan& plan = {});
[[nodiscard]] std::string_view cpu_linalg_provider_name(CpuLinalgProvider provider) noexcept;
[[nodiscard]] std::string_view cpu_linalg_target_name() noexcept;

void cpu_gemm(char a_trans, char b_trans, std::size_t m, std::size_t n, std::size_t k,
              const double* a, const double* b, double* c, double alpha = 1.0, double beta = 0.0,
              const CpuLinalgPlan& plan = {});

namespace detail {
inline constexpr std::size_t split3_max_inner = 64;
inline constexpr std::size_t split3_limbs = 3;
}  // namespace detail

/** Numeric workspace for the packed operands, binary exponents, SGEMM output,
 * transactional FP64 result and fixed split buffer. */
[[nodiscard]] inline std::size_t cpu_split3_gemm_workspace_bytes(std::size_t m, std::size_t n,
                                                                 std::size_t k) {
  if (!m || !n || !k) return 0;
  using runtime::size_add;
  using runtime::size_mul;
  const auto output = size_mul(m, n);
  const auto lines = size_mul(detail::split3_limbs, size_add(m, n));
  const auto floats = size_add(size_mul(lines, std::min(k, detail::split3_max_inner)), output);
  const auto dynamic = size_add(size_mul(floats, sizeof(float)), size_mul(lines, sizeof(int)));
  return size_add(dynamic, size_mul(size_add(output, detail::split3_max_inner), sizeof(double)));
}

/** Three binary32 limbs per operand, contracted by nine OpenBLAS SGEMMs per
 * inner block of at most 64 rows. Blocks with wide residual exponent ranges
 * subdivide before any matrix product executes. Factors use separate binary exponents per
 * row/column and limb. Recombination and publication use binary64.
 * Returns the number of executed binary32 matrix products.
 */
[[nodiscard]] std::size_t cpu_split3_gemm(char a_trans, char b_trans, std::size_t m, std::size_t n,
                                          std::size_t k, const double* a, const double* b,
                                          double* c, double alpha = 1.0, double beta = 0.0,
                                          const CpuLinalgPlan& plan = {});

/** Square rank-2 congruence transform with row-major storage.
 * `coefficient_transpose == 'T'` computes result := C^T * A * C.
 * `coefficient_transpose == 'N'` computes result := C * A * C^T.
 * `workspace` must contain n*n doubles and must not alias C, A, or result.
 */
void cpu_congruence(char coefficient_transpose, std::size_t n, const double* coefficients,
                    const double* matrix, double* result, double* workspace,
                    const CpuLinalgPlan& plan = {});

/** Matrix-vector product with row-major A.
 * `trans == 'N'` consumes an m-by-n A and n-vector x, producing an m-vector y.
 * `trans == 'T'` consumes the same storage and an m-vector x, producing an n-vector y.
 */
void cpu_gemv(char trans, std::size_t m, std::size_t n, const double* a, const double* x, double* y,
              double alpha = 1.0, double beta = 0.0, const CpuLinalgPlan& plan = {});

/** Rank-1 update of a row-major m-by-n matrix: A := alpha * x * y^T + A. */
void cpu_ger(std::size_t m, std::size_t n, const double* x, const double* y, double* a,
             double alpha = 1.0, const CpuLinalgPlan& plan = {});

/** Symmetric matrix-matrix product with row-major storage.
 * `side == 'L'` computes C := alpha * A * B + beta * C with m-by-m symmetric A.
 * `side == 'R'` computes C := alpha * B * A + beta * C with n-by-n symmetric A.
 * Only the triangle selected by `uplo` is read from A.
 */
void cpu_symm(char side, char uplo, std::size_t m, std::size_t n, const double* a, const double* b,
              double* c, double alpha = 1.0, double beta = 0.0, const CpuLinalgPlan& plan = {});

/** Symmetric rank-1 update of the selected triangle: A := alpha * x * x^T + A.
 * Only the triangle selected by `uplo` is read from or written to in A.
 */
void cpu_syr(char uplo, std::size_t n, const double* x, double* a, double alpha = 1.0,
             const CpuLinalgPlan& plan = {});

/** Symmetric rank-2 update of the selected triangle.
 * A := alpha * x * y^T + alpha * y * x^T + A.
 * Only the triangle selected by `uplo` is read from or written to in A.
 */
void cpu_syr2(char uplo, std::size_t n, const double* x, const double* y, double* a,
              double alpha = 1.0, const CpuLinalgPlan& plan = {});

/** Symmetric rank-k update of the selected triangle.
 * `trans == 'N'` consumes an n-by-k row-major A; `trans == 'T'` consumes k-by-n.
 * Only the triangle selected by `uplo` is read from or written to in C.
 */
void cpu_syrk(char uplo, char trans, std::size_t n, std::size_t k, const double* a, double* c,
              double alpha = 1.0, double beta = 0.0, const CpuLinalgPlan& plan = {});

/** Symmetric rank-2k update of the selected triangle.
 * `trans == 'N'` consumes n-by-k row-major A/B and computes
 * C := alpha * A * B^T + alpha * B * A^T + beta * C.
 * `trans == 'T'` consumes k-by-n A/B and computes the transposed-form update.
 * Only the triangle selected by `uplo` is read from or written to in C.
 */
void cpu_syr2k(char uplo, char trans, std::size_t n, std::size_t k, const double* a,
               const double* b, double* c, double alpha = 1.0, double beta = 0.0,
               const CpuLinalgPlan& plan = {});

/** Solve a triangular matrix equation in place using row-major storage.
 * `side == 'L'` computes B := alpha * op(A)^-1 * B and requires an m-by-m A.
 * `side == 'R'` computes B := alpha * B * op(A)^-1 and requires an n-by-n A.
 * `diag == 'U'` treats the diagonal as unit and does not read it.
 */
void cpu_trsm(char side, char uplo, char trans, char diag, std::size_t m, std::size_t n,
              const double* a, double* b, double alpha = 1.0, const CpuLinalgPlan& plan = {});

/** Multiply a general matrix by a triangular matrix in place using row-major storage.
 * `side == 'L'` computes B := alpha * op(A) * B and requires an m-by-m A.
 * `side == 'R'` computes B := alpha * B * op(A) and requires an n-by-n A.
 * `diag == 'U'` treats the diagonal as unit and does not read it.
 */
void cpu_trmm(char side, char uplo, char trans, char diag, std::size_t m, std::size_t n,
              const double* a, double* b, double alpha = 1.0, const CpuLinalgPlan& plan = {});

/** In-place lower Cholesky factorization. Returns LAPACK-style info:
 * 0 on success, j>0 when the leading minor of order j is not positive definite.
 */
int cpu_cholesky_lower(double* matrix, std::size_t n, const CpuLinalgPlan& plan = {});
[[nodiscard]] CpuSymmetricEigenResult cpu_symmetric_eigen(std::vector<double> matrix, std::size_t n,
                                                          const CpuLinalgPlan& plan = {});
/** Opt-in absolute off-diagonal stopping cap for the explicit scalar provider.
 * A positive cap is in the input matrix's units and can only tighten its usual
 * relative tolerance. Zero preserves the default; nonfinite/negative caps and
 * non-scalar plans with a positive cap are rejected. Storage is unchanged.
 */
[[nodiscard]] CpuSymmetricEigenResult cpu_symmetric_eigen(std::vector<double> matrix, std::size_t n,
                                                          const CpuLinalgPlan& plan,
                                                          double absolute_off_diagonal_tolerance);

}  // namespace generativeqc::tensor

#endif
