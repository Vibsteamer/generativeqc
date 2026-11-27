// Native FP64 cuBLAS (and matching host reference) implementation of extended
// physical AO density-fitted Fock/exchange operators and their full-rank metric
// and occupied-orbital pullback (replaces all TensorIR graphs in _f12_cabs_fock.py).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#ifndef GQC_SINK_HOST_REFERENCE
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace {

using Size = std::uint64_t;

constexpr int kOk = 0;
constexpr int kErrInvalidArg = -1;
constexpr int kErrNullPtr = -2;
constexpr int kErrAlias = -3;
constexpr int kErrNonFinite = -4;
constexpr int kErrOom = -5;
constexpr int kErrInternal = -6;

constexpr Size kVramTileBudgetBytes = Size(256) * 1024 * 1024;
constexpr Size kVramCoreTotalBudgetBytes = Size(384) * 1024 * 1024;
constexpr Size kVramTileBudgetDoubles = kVramTileBudgetBytes / sizeof(double);
constexpr Size kVramCoreTotalBudgetDoubles = kVramCoreTotalBudgetBytes / sizeof(double);

inline bool fits_int(Size v) {
  return v <= static_cast<Size>(std::numeric_limits<int>::max());
}

inline bool checked_mul(Size a, Size b, Size& out) {
  if (a != 0 && b > std::numeric_limits<Size>::max() / a) return false;
  out = a * b;
  return true;
}

inline bool checked_add(Size a, Size b, Size& out) {
  if (a > std::numeric_limits<Size>::max() - b) return false;
  out = a + b;
  return true;
}

inline bool check_finite(const double* ptr, Size count) {
  if (!ptr || count == 0) return true;
  for (Size i = 0; i < count; ++i) {
    if (!std::isfinite(ptr[i])) return false;
  }
  return true;
}

// Pack O[:, i0:i0+co]^T into [co, n] row-major buffer.
void pack_occ_t_chunk(int n, int o, int i0, int co, const double* O, double* O_t) {
  for (int m = 0; m < n; ++m) {
    const double* row = O + Size(m) * o + i0;
    for (int i = 0; i < co; ++i) {
      O_t[Size(i) * n + m] = row[i];
    }
  }
}

// Add [co, n] row-major buffer transposed into bar_O[:, i0:i0+co] ([n, o]).
void add_occ_t_chunk(int n, int o, int i0, int co, const double* bar_O_t, double* bar_O) {
  for (int m = 0; m < n; ++m) {
    double* row = bar_O + Size(m) * o + i0;
    for (int i = 0; i < co; ++i) {
      row[i] += bar_O_t[Size(i) * n + m];
    }
  }
}

#ifndef GQC_SINK_HOST_REFERENCE
// Pack a contiguous [q, co, n] chunk from [q, o, n] starting at orbital i0.
void pack_qon_chunk(int q, int o, int n, int i0, int co, const double* src, double* dst) {
  const Size src_q_stride = Size(o) * n;
  const Size dst_q_stride = Size(co) * n;
  const Size row_elems = Size(co) * n;
  for (int a = 0; a < q; ++a) {
    const double* s = src + Size(a) * src_q_stride + Size(i0) * n;
    double* d = dst + Size(a) * dst_q_stride;
    std::memcpy(d, s, row_elems * sizeof(double));
  }
}

// Unpack a contiguous [q, co, n] chunk back into [q, o, n] starting at orbital i0.
void unpack_qon_chunk(int q, int o, int n, int i0, int co, const double* src, double* dst) {
  const Size src_q_stride = Size(co) * n;
  const Size dst_q_stride = Size(o) * n;
  const Size row_elems = Size(co) * n;
  for (int a = 0; a < q; ++a) {
    const double* s = src + Size(a) * src_q_stride;
    double* d = dst + Size(a) * dst_q_stride + Size(i0) * n;
    std::memcpy(d, s, row_elems * sizeof(double));
  }
}
#endif

#ifdef GQC_SINK_HOST_REFERENCE

void rm_gemm_host(bool trans_a, bool trans_b, int M, int N, int K, double alpha, const double* A,
                  int lda, const double* B, int ldb, double beta, double* C, int ldc) {
  if (M == 0 || N == 0) return;
  if (K == 0) {
    if (beta == 0.0) {
      for (int i = 0; i < M; ++i) {
        std::fill_n(C + Size(i) * ldc, N, 0.0);
      }
    } else if (beta != 1.0) {
      for (int i = 0; i < M; ++i) {
        double* row = C + Size(i) * ldc;
        for (int j = 0; j < N; ++j) row[j] *= beta;
      }
    }
    return;
  }
  if (!trans_a && !trans_b) {
    for (int i = 0; i < M; ++i) {
      double* c_row = C + Size(i) * ldc;
      if (beta == 0.0) {
        std::fill_n(c_row, N, 0.0);
      } else if (beta != 1.0) {
        for (int j = 0; j < N; ++j) c_row[j] *= beta;
      }
      const double* a_row = A + Size(i) * lda;
      for (int k = 0; k < K; ++k) {
        const double aik = alpha * a_row[k];
        const double* b_row = B + Size(k) * ldb;
        for (int j = 0; j < N; ++j) c_row[j] += aik * b_row[j];
      }
    }
  } else if (trans_a && !trans_b) {
    for (int i = 0; i < M; ++i) {
      double* c_row = C + Size(i) * ldc;
      if (beta == 0.0) {
        std::fill_n(c_row, N, 0.0);
      } else if (beta != 1.0) {
        for (int j = 0; j < N; ++j) c_row[j] *= beta;
      }
    }
    for (int k = 0; k < K; ++k) {
      const double* a_row = A + Size(k) * lda;
      const double* b_row = B + Size(k) * ldb;
      for (int i = 0; i < M; ++i) {
        const double aki = alpha * a_row[i];
        double* c_row = C + Size(i) * ldc;
        for (int j = 0; j < N; ++j) c_row[j] += aki * b_row[j];
      }
    }
  } else if (!trans_a && trans_b) {
    for (int i = 0; i < M; ++i) {
      const double* a_row = A + Size(i) * lda;
      double* c_row = C + Size(i) * ldc;
      for (int j = 0; j < N; ++j) {
        const double* b_row = B + Size(j) * ldb;
        double acc = 0.0;
        for (int k = 0; k < K; ++k) acc += a_row[k] * b_row[k];
        c_row[j] = (beta == 0.0) ? alpha * acc : beta * c_row[j] + alpha * acc;
      }
    }
  } else {
    for (int i = 0; i < M; ++i) {
      double* c_row = C + Size(i) * ldc;
      for (int j = 0; j < N; ++j) {
        const double* b_row = B + Size(j) * ldb;
        double acc = 0.0;
        for (int k = 0; k < K; ++k) acc += A[Size(k) * lda + i] * b_row[k];
        c_row[j] = (beta == 0.0) ? alpha * acc : beta * c_row[j] + alpha * acc;
      }
    }
  }
}

void sym_half_host(int n, const double* in, double* out) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      out[Size(i) * n + j] = 0.5 * (in[Size(i) * n + j] + in[Size(j) * n + i]);
    }
  }
}

int run_fwd_spin_host(int q, int n, int o, const double* W, const double* O, const double* R,
                      double* D, double* K, double* rho_acc) {
  const Size p_sz = Size(o) * Size(n);
  const Size qo_sz = Size(q) * Size(o);
  if (!fits_int(p_sz) || !fits_int(qo_sz)) return kErrInvalidArg;
  const int P = static_cast<int>(p_sz);
  std::vector<double> B(Size(q) * p_sz);
  std::vector<double> O_t(p_sz);
  pack_occ_t_chunk(n, o, 0, o, O, O_t.data());

  // B = W @ R, D = W^T @ B
  rm_gemm_host(false, false, q, P, q, 1.0, W, q, R, P, 0.0, B.data(), P);
  rm_gemm_host(true, false, q, P, q, 1.0, W, q, B.data(), P, 0.0, D, P);

  // K = B^T @ B where B is viewed as [q * o, n]
  rm_gemm_host(true, false, n, n, static_cast<int>(qo_sz), 1.0, B.data(), n, B.data(), n, 0.0, K,
               n);

  // rho_acc[a] += sum_p D[a, p] * O_t[p]
  for (int a = 0; a < q; ++a) {
    const double* d_row = D + Size(a) * p_sz;
    double acc = 0.0;
    for (Size p = 0; p < p_sz; ++p) acc += d_row[p] * O_t[p];
    rho_acc[a] += acc;
  }
  return kOk;
}

int run_rev_spin_host(int q, int n, int o, const double* W, const double* O, const double* R,
                      const double* D, const double* bar_rho, const double* k_seed,
                      double* bar_R_out, double* bar_O_acc, double* weight_acc) {
  const Size p_sz = Size(o) * Size(n);
  const Size qo_sz = Size(q) * Size(o);
  if (!fits_int(p_sz) || !fits_int(qo_sz)) return kErrInvalidArg;
  const int P = static_cast<int>(p_sz);
  const int qo = static_cast<int>(qo_sz);
  std::vector<double> O_t(p_sz);
  pack_occ_t_chunk(n, o, 0, o, O, O_t.data());

  // 1. bar_O_core[m, i] += sum_a D[a, i, m] * bar_rho[a]
  std::vector<double> bar_O_t(p_sz, 0.0);
  for (int a = 0; a < q; ++a) {
    const double br = bar_rho[a];
    const double* d_row = D + Size(a) * p_sz;
    for (Size p = 0; p < p_sz; ++p) bar_O_t[p] += br * d_row[p];
  }
  add_occ_t_chunk(n, o, 0, o, bar_O_t.data(), bar_O_acc);

  // 2. bar_D = R @ k_seed^T + bar_rho[:, None] * O_t[None, :]
  std::vector<double> bar_D(Size(q) * p_sz);
  rm_gemm_host(false, true, qo, n, n, 1.0, R, n, k_seed, n, 0.0, bar_D.data(), n);
  for (int a = 0; a < q; ++a) {
    const double br = bar_rho[a];
    double* row = bar_D.data() + Size(a) * p_sz;
    for (Size p = 0; p < p_sz; ++p) row[p] += br * O_t[p];
  }

  // 3. U = W @ bar_D, three_center_bar = W^T @ U
  std::vector<double> U(Size(q) * p_sz);
  rm_gemm_host(false, false, q, P, q, 1.0, W, q, bar_D.data(), P, 0.0, U.data(), P);
  rm_gemm_host(true, false, q, P, q, 1.0, W, q, U.data(), P, 0.0, bar_R_out, P);

  // 4. weight_acc += D @ three_center_bar^T
  rm_gemm_host(false, true, q, q, P, 1.0, D, P, bar_R_out, P, 1.0, weight_acc, q);

  // 5. bar_R_out += D @ k_seed
  rm_gemm_host(false, false, qo, n, n, 1.0, D, n, k_seed, n, 1.0, bar_R_out, n);
  return kOk;
}

#else

struct DeviceBuffer {
  double* ptr = nullptr;
  Size count = 0;
  ~DeviceBuffer() {
    if (ptr) cudaFree(ptr);
  }
  bool alloc(Size n) {
    if (n == 0) return true;
    count = n;
    return cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(double)) == cudaSuccess;
  }
  bool zero() {
    if (!ptr || count == 0) return true;
    return cudaMemset(ptr, 0, count * sizeof(double)) == cudaSuccess;
  }
};

struct CublasHandle {
  cublasHandle_t handle = nullptr;
  ~CublasHandle() {
    if (handle) cublasDestroy(handle);
  }
  cublasStatus_t init() {
    const auto status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    // These FP64 primitives accept exact-fit budgets; no uncharged BLAS operation workspace.
    return cublasSetWorkspace(handle, nullptr, 0);
  }
};

inline unsigned int grid_for(Size n, unsigned int block = 256) {
  return static_cast<unsigned int>((n + block - 1) / block);
}

inline cublasStatus_t rm_gemm(cublasHandle_t h, bool trans_a, bool trans_b, int M, int N, int K,
                              double alpha, const double* A, int lda, const double* B, int ldb,
                              double beta, double* C, int ldc) {
  return cublasDgemm(h, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N, trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                     N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc);
}

inline cublasStatus_t rm_gemm_batched(cublasHandle_t h, bool trans_a, bool trans_b, int M, int N,
                                      int K, double alpha, const double* A, int lda,
                                      long long stride_a, const double* B, int ldb,
                                      long long stride_b, double beta, double* C, int ldc,
                                      long long stride_c, int batch_count) {
  return cublasDgemmStridedBatched(h, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                                   trans_a ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B, ldb,
                                   stride_b, A, lda, stride_a, &beta, C, ldc, stride_c,
                                   batch_count);
}

__global__ void k_sym_scale(int n, double scale, const double* in, double* out) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(n) * n;
  if (idx >= total) return;
  const int r = static_cast<int>(idx / n);
  const int c = static_cast<int>(idx % n);
  out[idx] = scale * (in[idx] + in[Size(c) * n + r]);
}

__global__ void k_add_rank1_q_p(int q, int P, const double* vec_q, const double* vec_p,
                                double* mat_qp) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(q) * P;
  if (idx >= total) return;
  const int a = static_cast<int>(idx / P);
  const int p = static_cast<int>(idx % P);
  mat_qp[idx] += vec_q[a] * vec_p[p];
}

__global__ void k_outer_q_nn(int qt, int nn, const double* rho_q, const double* bar_g_nn,
                             double* out_qnn, bool add_mode) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(qt) * nn;
  if (idx >= total) return;
  const int a = static_cast<int>(idx / nn);
  const int p = static_cast<int>(idx % nn);
  const double val = rho_q[a] * bar_g_nn[p];
  if (add_mode)
    out_qnn[idx] += val;
  else
    out_qnn[idx] = val;
}

__global__ void k_transpose_qon_to_qno(int qt, int o, int n, const double* in, double* out) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const Size total = Size(qt) * o * n;
  if (idx >= total) return;
  const Size on = Size(o) * n;
  const int a = static_cast<int>(idx / on);
  const Size rem = idx % on;
  const int i = static_cast<int>(rem / n);
  const int m = static_cast<int>(rem % n);
  out[Size(a) * on + Size(m) * o + i] = in[idx];
}

int run_fwd_spin_gpu(cublasHandle_t h, int q, int n, int o, const double* d_W, const double* O,
                     const double* R, double* D, double* d_K, double* d_rho_acc,
                     Size chunk_budget_doubles) {
  Size qn = 0, three_qn = 0, per_o_doubles = 0;
  if (!checked_mul(Size(q), Size(n), qn) || !checked_mul(Size(3), qn, three_qn) ||
      !checked_add(three_qn, Size(n), per_o_doubles)) {
    return kErrOom;
  }
  if (per_o_doubles == 0 || chunk_budget_doubles < per_o_doubles) return kErrOom;
  const Size max_co_mem = chunk_budget_doubles / per_o_doubles;
  const Size max_co_int_n = static_cast<Size>(std::numeric_limits<int>::max()) / Size(n);
  const Size max_co_int_q = static_cast<Size>(std::numeric_limits<int>::max()) / Size(q);
  const Size max_co_sz =
      std::min<Size>(Size(o), std::min<Size>(max_co_mem, std::min<Size>(max_co_int_n, max_co_int_q)));
  if (max_co_sz == 0) return kErrInvalidArg;
  const int max_co = static_cast<int>(max_co_sz);

  DeviceBuffer d_R, d_B, d_D, d_O_t;
  const Size max_qpn = qn * max_co_sz;
  if (!d_R.alloc(max_qpn) || !d_B.alloc(max_qpn) || !d_D.alloc(max_qpn) ||
      !d_O_t.alloc(max_co_sz * Size(n))) {
    return kErrOom;
  }

  std::vector<double> h_R_chunk, h_D_chunk;
  if (max_co < o) {
    h_R_chunk.resize(max_qpn);
    h_D_chunk.resize(max_qpn);
  }
  std::vector<double> h_O_t(max_co_sz * Size(n));

  if (cudaMemset(d_K, 0, Size(n) * n * sizeof(double)) != cudaSuccess) return 3;

  for (int i0 = 0; i0 < o; i0 += max_co) {
    const int co = std::min(max_co, o - i0);
    const int P = static_cast<int>(Size(co) * Size(n));
    const int q_co = static_cast<int>(Size(q) * Size(co));
    const Size qpn = Size(q) * Size(P);

    const double* r_src = R;
    if (co < o) {
      pack_qon_chunk(q, o, n, i0, co, R, h_R_chunk.data());
      r_src = h_R_chunk.data();
    }
    pack_occ_t_chunk(n, o, i0, co, O, h_O_t.data());

    if (cudaMemcpy(d_R.ptr, r_src, qpn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_O_t.ptr, h_O_t.data(), Size(P) * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
      return 3;
    }

    // B = W @ R, D = W^T @ B
    if (rm_gemm(h, false, false, q, P, q, 1.0, d_W, q, d_R.ptr, P, 0.0, d_B.ptr, P) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, true, false, q, P, q, 1.0, d_W, q, d_B.ptr, P, 0.0, d_D.ptr, P) !=
            CUBLAS_STATUS_SUCCESS) {
      return 4;
    }

    // K += B^T @ B (B viewed as [q * co, n])
    if (rm_gemm(h, true, false, n, n, q_co, 1.0, d_B.ptr, n, d_B.ptr, n, 1.0, d_K, n) !=
        CUBLAS_STATUS_SUCCESS) {
      return 4;
    }

    // rho_acc += D @ O_t (D row-major [q, P] => col-major [P, q], OP_T gives [q, P] @ [P])
    const double one = 1.0;
    if (cublasDgemv(h, CUBLAS_OP_T, P, q, &one, d_D.ptr, P, d_O_t.ptr, 1, &one, d_rho_acc, 1) !=
        CUBLAS_STATUS_SUCCESS) {
      return 4;
    }

    if (co < o) {
      if (cudaMemcpy(h_D_chunk.data(), d_D.ptr, qpn * sizeof(double), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        return 5;
      }
      unpack_qon_chunk(q, o, n, i0, co, h_D_chunk.data(), D);
    } else {
      if (cudaMemcpy(D, d_D.ptr, qpn * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 5;
      }
    }
  }
  return kOk;
}

int run_rev_spin_gpu(cublasHandle_t h, int q, int n, int o, const double* d_W,
                     const double* d_bar_rho, const double* O, const double* R, const double* D,
                     const double* k_seed, double* bar_R_out, double* bar_O_acc,
                     double* d_weight_acc, Size chunk_budget_doubles) {
  Size qn = 0, three_qn = 0, per_o_doubles = 0;
  if (!checked_mul(Size(q), Size(n), qn) || !checked_mul(Size(3), qn, three_qn) ||
      !checked_add(three_qn, Size(2) * Size(n), per_o_doubles)) {
    return kErrOom;
  }
  if (per_o_doubles == 0 || chunk_budget_doubles < per_o_doubles) return kErrOom;
  const Size max_co_mem = chunk_budget_doubles / per_o_doubles;
  const Size max_co_int_n = static_cast<Size>(std::numeric_limits<int>::max()) / Size(n);
  const Size max_co_int_q = static_cast<Size>(std::numeric_limits<int>::max()) / Size(q);
  const Size max_co_sz =
      std::min<Size>(Size(o), std::min<Size>(max_co_mem, std::min<Size>(max_co_int_n, max_co_int_q)));
  if (max_co_sz == 0) return kErrInvalidArg;
  const int max_co = static_cast<int>(max_co_sz);

  DeviceBuffer d_R, d_D, d_tmp, d_O_t, d_bar_O_t, d_k_seed;
  const Size max_qpn = qn * max_co_sz;
  if (!d_R.alloc(max_qpn) || !d_D.alloc(max_qpn) || !d_tmp.alloc(max_qpn) ||
      !d_O_t.alloc(max_co_sz * Size(n)) || !d_bar_O_t.alloc(max_co_sz * Size(n)) ||
      !d_k_seed.alloc(Size(n) * n)) {
    return kErrOom;
  }
  if (cudaMemcpy(d_k_seed.ptr, k_seed, Size(n) * n * sizeof(double), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    return 3;
  }

  std::vector<double> h_R_chunk, h_D_chunk, h_bar_R_chunk;
  if (max_co < o) {
    h_R_chunk.resize(max_qpn);
    h_D_chunk.resize(max_qpn);
    h_bar_R_chunk.resize(max_qpn);
  }
  std::vector<double> h_O_t(max_co_sz * Size(n));
  std::vector<double> h_bar_O_t(max_co_sz * Size(n));

  for (int i0 = 0; i0 < o; i0 += max_co) {
    const int co = std::min(max_co, o - i0);
    const int P = static_cast<int>(Size(co) * Size(n));
    const int q_co = static_cast<int>(Size(q) * Size(co));
    const Size qpn = Size(q) * Size(P);

    const double* r_src = R;
    const double* d_src = D;
    if (co < o) {
      pack_qon_chunk(q, o, n, i0, co, R, h_R_chunk.data());
      pack_qon_chunk(q, o, n, i0, co, D, h_D_chunk.data());
      r_src = h_R_chunk.data();
      d_src = h_D_chunk.data();
    }
    pack_occ_t_chunk(n, o, i0, co, O, h_O_t.data());

    if (cudaMemcpy(d_R.ptr, r_src, qpn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_D.ptr, d_src, qpn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_O_t.ptr, h_O_t.data(), Size(P) * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
      return 3;
    }

    // 1. bar_O_t = D^T @ bar_rho (D row-major [q, P] => col-major [P, q], OP_N gives [P, q] @ [q])
    const double one = 1.0;
    const double zero = 0.0;
    if (cublasDgemv(h, CUBLAS_OP_N, P, q, &one, d_D.ptr, P, d_bar_rho, 1, &zero, d_bar_O_t.ptr,
                    1) != CUBLAS_STATUS_SUCCESS) {
      return 4;
    }
    if (cudaMemcpy(h_bar_O_t.data(), d_bar_O_t.ptr, Size(P) * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return 5;
    }
    add_occ_t_chunk(n, o, i0, co, h_bar_O_t.data(), bar_O_acc);

    // 2. d_tmp = bar_D = R @ k_seed^T + bar_rho[:, None] * O_t[None, :]
    if (rm_gemm(h, false, true, q_co, n, n, 1.0, d_R.ptr, n, d_k_seed.ptr, n, 0.0, d_tmp.ptr, n) !=
        CUBLAS_STATUS_SUCCESS) {
      return 4;
    }
    k_add_rank1_q_p<<<grid_for(qpn), 256>>>(q, P, d_bar_rho, d_O_t.ptr, d_tmp.ptr);
    if (cudaGetLastError() != cudaSuccess) return 4;

    // 3. Reuse d_R as U = W @ bar_D, then d_tmp = three_center_bar = W^T @ U
    if (rm_gemm(h, false, false, q, P, q, 1.0, d_W, q, d_tmp.ptr, P, 0.0, d_R.ptr, P) !=
            CUBLAS_STATUS_SUCCESS ||
        rm_gemm(h, true, false, q, P, q, 1.0, d_W, q, d_R.ptr, P, 0.0, d_tmp.ptr, P) !=
            CUBLAS_STATUS_SUCCESS) {
      return 4;
    }

    // 4. weight_acc += D @ three_center_bar^T
    if (rm_gemm(h, false, true, q, q, P, 1.0, d_D.ptr, P, d_tmp.ptr, P, 1.0, d_weight_acc, q) !=
        CUBLAS_STATUS_SUCCESS) {
      return 4;
    }

    // 5. d_tmp += D @ k_seed => total bar_R_s
    if (rm_gemm(h, false, false, q_co, n, n, 1.0, d_D.ptr, n, d_k_seed.ptr, n, 1.0, d_tmp.ptr, n) !=
        CUBLAS_STATUS_SUCCESS) {
      return 4;
    }

    if (co < o) {
      if (cudaMemcpy(h_bar_R_chunk.data(), d_tmp.ptr, qpn * sizeof(double),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 5;
      }
      unpack_qon_chunk(q, o, n, i0, co, h_bar_R_chunk.data(), bar_R_out);
    } else {
      if (cudaMemcpy(bar_R_out, d_tmp.ptr, qpn * sizeof(double), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        return 5;
      }
    }
  }
  return kOk;
}

#endif

}  // namespace

#ifdef GQC_SINK_HOST_REFERENCE
#define GQC_EXT_FOCK_SYM(name) name##_host_reference
#else
#define GQC_EXT_FOCK_SYM(name) name##_cuda
#endif

extern "C" {

// 1. Q-tiled half-transform forward: R[a, i, m] = sum_v A[a, m, v] * O[v, i]
int GQC_EXT_FOCK_SYM(gqc_f12_ext_fock_half_fwd)(int qt, int n, int o, const double* raw_tile,
                                                const double* occ, double* R_tile) {
  try {
    if (qt < 0 || n < 0 || o < 0) return kErrInvalidArg;
    if (qt == 0 || n == 0 || o == 0) return kOk;
    if (!raw_tile || !occ || !R_tile) return kErrNullPtr;

    const Size nn = Size(n) * Size(n);
    const Size on = Size(o) * Size(n);
    Size per_q_doubles = 0;
    if (!checked_add(nn, on, per_q_doubles) || on > kVramTileBudgetDoubles ||
        per_q_doubles > kVramTileBudgetDoubles - on) {
      return kErrOom;
    }
    Size total_out = 0;
    if (!checked_mul(Size(qt), on, total_out)) return kErrInvalidArg;

#ifdef GQC_SINK_HOST_REFERENCE
    for (int a = 0; a < qt; ++a) {
      rm_gemm_host(true, true, o, n, n, 1.0, occ, o, raw_tile + Size(a) * nn, n, 0.0,
                   R_tile + Size(a) * on, n);
    }
    if (!check_finite(R_tile, total_out)) return kErrNonFinite;
    return kOk;
#else
    const int max_sub =
        static_cast<int>(std::min<Size>(Size(qt), (kVramTileBudgetDoubles - on) / per_q_doubles));
    CublasHandle ch;
    if (ch.init() != CUBLAS_STATUS_SUCCESS) return 1;

    DeviceBuffer d_occ, d_raw, d_R;
    if (!d_occ.alloc(on) || !d_raw.alloc(Size(max_sub) * nn) || !d_R.alloc(Size(max_sub) * on)) {
      return kErrOom;
    }
    if (cudaMemcpy(d_occ.ptr, occ, on * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
      return 3;
    }

    for (int q0 = 0; q0 < qt; q0 += max_sub) {
      const int sub = std::min(max_sub, qt - q0);
      if (cudaMemcpy(d_raw.ptr, raw_tile + Size(q0) * nn, Size(sub) * nn * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return 3;
      }
      if (rm_gemm_batched(ch.handle, true, true, o, n, n, 1.0, d_occ.ptr, o, 0, d_raw.ptr, n,
                          static_cast<long long>(nn), 0.0, d_R.ptr, n, static_cast<long long>(on),
                          sub) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      if (cudaMemcpy(R_tile + Size(q0) * on, d_R.ptr, Size(sub) * on * sizeof(double),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 5;
      }
    }
    if (!check_finite(R_tile, total_out)) return kErrNonFinite;
    return kOk;
#endif
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

// 2. Core forward: R_s -> (D_s, K_s, rho)
int GQC_EXT_FOCK_SYM(gqc_f12_ext_fock_fwd)(int q, int n, int oa, int ob, int same_occ_frame,
                                           const double* whitener, const double* occ_a,
                                           const double* occ_b, const double* R_a,
                                           const double* R_b, double* D_a, double* D_b,
                                           double* K_a, double* K_b, double* rho) {
  try {
    if (q < 0 || n < 0 || oa < 0 || ob < 0) return kErrInvalidArg;
    if (q == 0 || n == 0) return kOk;
    if (!whitener || !K_a || !K_b || !rho) return kErrNullPtr;
    if (oa > 0 && (!occ_a || !R_a || !D_a)) return kErrNullPtr;
    if (ob > 0 && !same_occ_frame && (!occ_b || !R_b || !D_b)) return kErrNullPtr;

    const Size qq = Size(q) * Size(q);
    const Size nn = Size(n) * Size(n);
    const Size qn = Size(q) * Size(n);

    if (oa == 0 && ob == 0) {
      std::fill_n(rho, q, 0.0);
      std::fill_n(K_a, nn, 0.0);
      std::fill_n(K_b, nn, 0.0);
      return kOk;
    }

    // Hard VRAM budget check (fixed resident + minimum 1-orbital chunk) before allocation.
    Size fixed_doubles = 0, tmp_sum = 0, three_qn = 0, per_o_doubles = 0;
    if (!checked_add(qq, nn, tmp_sum) || !checked_add(tmp_sum, Size(q), fixed_doubles) ||
        !checked_mul(Size(3), qn, three_qn) || !checked_add(three_qn, Size(n), per_o_doubles) ||
        fixed_doubles > kVramCoreTotalBudgetDoubles ||
        per_o_doubles > kVramCoreTotalBudgetDoubles - fixed_doubles) {
      return kErrOom;
    }

    std::fill_n(rho, q, 0.0);
    if (oa == 0) std::fill_n(K_a, nn, 0.0);
    if (ob == 0) std::fill_n(K_b, nn, 0.0);

#ifdef GQC_SINK_HOST_REFERENCE
    if (oa > 0) {
      const int rc = run_fwd_spin_host(q, n, oa, whitener, occ_a, R_a, D_a, K_a, rho);
      if (rc != 0) return rc;
    }
    if (same_occ_frame && oa > 0) {
      std::memcpy(K_b, K_a, nn * sizeof(double));
      for (int a = 0; a < q; ++a) rho[a] *= 2.0;
      if (D_b && D_b != D_a) {
        std::memcpy(D_b, D_a, qn * Size(oa) * sizeof(double));
      }
    } else if (ob > 0) {
      const int rc = run_fwd_spin_host(q, n, ob, whitener, occ_b, R_b, D_b, K_b, rho);
      if (rc != 0) return rc;
    }
#else
    CublasHandle ch;
    if (ch.init() != CUBLAS_STATUS_SUCCESS) return 1;

    const Size chunk_budget_doubles = kVramCoreTotalBudgetDoubles - fixed_doubles;

    DeviceBuffer d_W, d_K, d_rho;
    if (!d_W.alloc(qq) || !d_K.alloc(nn) || !d_rho.alloc(Size(q))) return kErrOom;
    if (cudaMemcpy(d_W.ptr, whitener, qq * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        !d_rho.zero()) {
      return 3;
    }

    if (oa > 0) {
      const int rc = run_fwd_spin_gpu(ch.handle, q, n, oa, d_W.ptr, occ_a, R_a, D_a, d_K.ptr,
                                      d_rho.ptr, chunk_budget_doubles);
      if (rc != 0) return rc;
      if (cudaMemcpy(K_a, d_K.ptr, nn * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 5;
      }
    }
    if (same_occ_frame && oa > 0) {
      std::memcpy(K_b, K_a, nn * sizeof(double));
      if (cudaMemcpy(rho, d_rho.ptr, Size(q) * sizeof(double), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        return 5;
      }
      for (int a = 0; a < q; ++a) rho[a] *= 2.0;
      if (D_b && D_b != D_a) {
        std::memcpy(D_b, D_a, qn * Size(oa) * sizeof(double));
      }
    } else {
      if (ob > 0) {
        const int rc = run_fwd_spin_gpu(ch.handle, q, n, ob, d_W.ptr, occ_b, R_b, D_b, d_K.ptr,
                                        d_rho.ptr, chunk_budget_doubles);
        if (rc != 0) return rc;
        if (cudaMemcpy(K_b, d_K.ptr, nn * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
          return 5;
        }
      }
      if (cudaMemcpy(rho, d_rho.ptr, Size(q) * sizeof(double), cudaMemcpyDeviceToHost) !=
          cudaSuccess) {
        return 5;
      }
    }
#endif
    if (!check_finite(K_a, nn) || !check_finite(K_b, nn) || !check_finite(rho, Size(q)) ||
        (oa > 0 && !check_finite(D_a, qn * Size(oa))) ||
        (ob > 0 && D_b && (!same_occ_frame || D_b != D_a) &&
         !check_finite(D_b, qn * Size(ob)))) {
      return kErrNonFinite;
    }
    return kOk;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

// 3. Q-tiled J forward: J_acc[m, v] += sum_a rho_tile[a] * raw_tile[a, m, v]
int GQC_EXT_FOCK_SYM(gqc_f12_ext_fock_j_fwd)(int qt, int n, const double* raw_tile,
                                             const double* rho_tile, double* J_acc) {
  try {
    if (qt < 0 || n < 0) return kErrInvalidArg;
    if (qt == 0 || n == 0) return kOk;
    if (!raw_tile || !rho_tile || !J_acc) return kErrNullPtr;

    const Size nn = Size(n) * Size(n);
    if (!fits_int(nn)) return kErrInvalidArg;

    Size per_q_doubles = 0;
    if (!checked_add(nn, Size(1), per_q_doubles) || nn > kVramTileBudgetDoubles ||
        per_q_doubles > kVramTileBudgetDoubles - nn) {
      return kErrOom;
    }

#ifdef GQC_SINK_HOST_REFERENCE
    for (int a = 0; a < qt; ++a) {
      const double r = rho_tile[a];
      const double* slice = raw_tile + Size(a) * nn;
      for (Size p = 0; p < nn; ++p) J_acc[p] += r * slice[p];
    }
    if (!check_finite(J_acc, nn)) return kErrNonFinite;
    return kOk;
#else
    CublasHandle ch;
    if (ch.init() != CUBLAS_STATUS_SUCCESS) return 1;
    const int nn_int = static_cast<int>(nn);
    const int max_sub =
        static_cast<int>(std::min<Size>(Size(qt), (kVramTileBudgetDoubles - nn) / per_q_doubles));

    DeviceBuffer d_raw, d_rho, d_J;
    if (!d_raw.alloc(Size(max_sub) * nn) || !d_rho.alloc(Size(max_sub)) || !d_J.alloc(nn)) {
      return kErrOom;
    }
    if (cudaMemcpy(d_J.ptr, J_acc, nn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
      return 3;
    }

    const double one = 1.0;
    for (int q0 = 0; q0 < qt; q0 += max_sub) {
      const int sub = std::min(max_sub, qt - q0);
      if (cudaMemcpy(d_raw.ptr, raw_tile + Size(q0) * nn, Size(sub) * nn * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(d_rho.ptr, rho_tile + q0, Size(sub) * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return 3;
      }
      // raw_tile row-major [sub, nn] => col-major [nn, sub]; OP_N computes [nn, sub] @ [sub]
      if (cublasDgemv(ch.handle, CUBLAS_OP_N, nn_int, sub, &one, d_raw.ptr, nn_int, d_rho.ptr, 1,
                      &one, d_J.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
    }
    if (cudaMemcpy(J_acc, d_J.ptr, nn * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
      return 5;
    }
    if (!check_finite(J_acc, nn)) return kErrNonFinite;
    return kOk;
#endif
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

// 4. Congruence pullback for M = C^T F_sym C with arbitrary (possibly non-symmetric) bar_M:
//    S = 0.5 * (bar_M + bar_M^T), T = C @ S,
//    bar_F_sym = 0.5 * (T @ C^T + (T @ C^T)^T), bar_C_acc += 2 * F_sym @ T.
int GQC_EXT_FOCK_SYM(gqc_f12_ext_fock_congruence_rev)(int n, int ext, const double* C,
                                                      const double* F_primal, const double* bar_M,
                                                      double* bar_F_sym, double* bar_C_acc) {
  try {
    if (n < 0 || ext < 0) return kErrInvalidArg;
    if (n == 0 || ext == 0) return kOk;
    if (!C || !F_primal || !bar_M || !bar_F_sym || !bar_C_acc) return kErrNullPtr;

    const Size nn = Size(n) * Size(n);
    const Size ne = Size(n) * Size(ext);
    const Size ee = Size(ext) * Size(ext);

    // 8 resident device matrices: 3*nn + 3*ne + 2*ee doubles must fit in VRAM budget.
    Size t1 = 0, t2 = 0, t3 = 0, t12 = 0, total_doubles = 0;
    if (!checked_mul(Size(3), nn, t1) || !checked_mul(Size(3), ne, t2) ||
        !checked_mul(Size(2), ee, t3) || !checked_add(t1, t2, t12) ||
        !checked_add(t12, t3, total_doubles) || total_doubles > kVramCoreTotalBudgetDoubles) {
      return kErrOom;
    }

#ifdef GQC_SINK_HOST_REFERENCE
    std::vector<double> F_sym(nn);
    std::vector<double> S(ee);
    std::vector<double> T(ne);
    std::vector<double> tmp_F(nn);
    sym_half_host(n, F_primal, F_sym.data());
    sym_half_host(ext, bar_M, S.data());
    rm_gemm_host(false, false, n, ext, ext, 1.0, C, ext, S.data(), ext, 0.0, T.data(), ext);
    rm_gemm_host(false, true, n, n, ext, 1.0, T.data(), ext, C, ext, 0.0, tmp_F.data(), n);
    sym_half_host(n, tmp_F.data(), bar_F_sym);
    rm_gemm_host(false, false, n, ext, n, 2.0, F_sym.data(), n, T.data(), ext, 1.0, bar_C_acc, ext);
    if (!check_finite(bar_F_sym, nn) || !check_finite(bar_C_acc, ne)) return kErrNonFinite;
    return kOk;
#else
    CublasHandle ch;
    if (ch.init() != CUBLAS_STATUS_SUCCESS) return 1;

    DeviceBuffer d_C, d_F_in, d_F_sym, d_M_in, d_S, d_T, d_bar_F, d_bar_C;
    if (!d_C.alloc(ne) || !d_F_in.alloc(nn) || !d_F_sym.alloc(nn) || !d_M_in.alloc(ee) ||
        !d_S.alloc(ee) || !d_T.alloc(ne) || !d_bar_F.alloc(nn) || !d_bar_C.alloc(ne)) {
      return kErrOom;
    }
    if (cudaMemcpy(d_C.ptr, C, ne * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_F_in.ptr, F_primal, nn * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(d_M_in.ptr, bar_M, ee * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_bar_C.ptr, bar_C_acc, ne * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
      return 3;
    }

    k_sym_scale<<<grid_for(nn), 256>>>(n, 0.5, d_F_in.ptr, d_F_sym.ptr);
    k_sym_scale<<<grid_for(ee), 256>>>(ext, 0.5, d_M_in.ptr, d_S.ptr);
    if (cudaGetLastError() != cudaSuccess) return 4;

    // T = C @ S; d_F_in (reused as scratch) = T @ C^T; bar_F_sym = 0.5 * (tmp + tmp^T)
    if (rm_gemm(ch.handle, false, false, n, ext, ext, 1.0, d_C.ptr, ext, d_S.ptr, ext, 0.0, d_T.ptr,
                ext) != CUBLAS_STATUS_SUCCESS ||
        rm_gemm(ch.handle, false, true, n, n, ext, 1.0, d_T.ptr, ext, d_C.ptr, ext, 0.0, d_F_in.ptr,
                n) != CUBLAS_STATUS_SUCCESS) {
      return 4;
    }
    k_sym_scale<<<grid_for(nn), 256>>>(n, 0.5, d_F_in.ptr, d_bar_F.ptr);
    if (cudaGetLastError() != cudaSuccess) return 4;

    // bar_C_acc += 2 * F_sym @ T
    if (rm_gemm(ch.handle, false, false, n, ext, n, 2.0, d_F_sym.ptr, n, d_T.ptr, ext, 1.0,
                d_bar_C.ptr, ext) != CUBLAS_STATUS_SUCCESS) {
      return 4;
    }

    if (cudaMemcpy(bar_F_sym, d_bar_F.ptr, nn * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(bar_C_acc, d_bar_C.ptr, ne * sizeof(double), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
      return 5;
    }
    if (!check_finite(bar_F_sym, nn) || !check_finite(bar_C_acc, ne)) return kErrNonFinite;
    return kOk;
#endif
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

// 5. Q-tiled J reverse:
//    bar_rho_tile[a] = sum_{m, v} raw_tile[a, m, v] * bar_g[m, v]
//    if bar_raw_tile != nullptr: bar_raw_tile[a, m, v] += rho_tile[a] * bar_g[m, v]
int GQC_EXT_FOCK_SYM(gqc_f12_ext_fock_j_rev)(int qt, int n, const double* raw_tile,
                                             const double* rho_tile, const double* bar_g,
                                             double* bar_rho_tile, double* bar_raw_tile) {
  try {
    if (qt < 0 || n < 0) return kErrInvalidArg;
    if (qt == 0 || n == 0) return kOk;
    if (!raw_tile || !bar_g || !bar_rho_tile) return kErrNullPtr;
    if (bar_raw_tile && !rho_tile) return kErrNullPtr;

    const Size nn = Size(n) * Size(n);
    if (!fits_int(nn)) return kErrInvalidArg;
    const bool write_bar = (bar_raw_tile != nullptr);

    Size per_q_raw = 0, per_q_doubles = 0;
    if (!checked_mul(write_bar ? Size(2) : Size(1), nn, per_q_raw) ||
        !checked_add(per_q_raw, write_bar ? Size(2) : Size(1), per_q_doubles) ||
        nn > kVramTileBudgetDoubles || per_q_doubles > kVramTileBudgetDoubles - nn) {
      return kErrOom;
    }

#ifdef GQC_SINK_HOST_REFERENCE
    for (int a = 0; a < qt; ++a) {
      const double* slice = raw_tile + Size(a) * nn;
      double acc = 0.0;
      for (Size p = 0; p < nn; ++p) acc += slice[p] * bar_g[p];
      bar_rho_tile[a] = acc;
      if (bar_raw_tile) {
        const double r = rho_tile[a];
        double* out_slice = bar_raw_tile + Size(a) * nn;
        for (Size p = 0; p < nn; ++p) out_slice[p] += r * bar_g[p];
      }
    }
    if (!check_finite(bar_rho_tile, Size(qt)) ||
        (write_bar && !check_finite(bar_raw_tile, Size(qt) * nn))) {
      return kErrNonFinite;
    }
    return kOk;
#else
    CublasHandle ch;
    if (ch.init() != CUBLAS_STATUS_SUCCESS) return 1;
    const int nn_int = static_cast<int>(nn);
    const int max_sub =
        static_cast<int>(std::min<Size>(Size(qt), (kVramTileBudgetDoubles - nn) / per_q_doubles));

    DeviceBuffer d_raw, d_bar_g, d_bar_rho, d_rho, d_bar_raw;
    if (!d_raw.alloc(Size(max_sub) * nn) || !d_bar_g.alloc(nn) || !d_bar_rho.alloc(Size(max_sub))) {
      return kErrOom;
    }
    if (write_bar && (!d_rho.alloc(Size(max_sub)) || !d_bar_raw.alloc(Size(max_sub) * nn))) {
      return kErrOom;
    }
    if (cudaMemcpy(d_bar_g.ptr, bar_g, nn * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
      return 3;
    }

    const double one = 1.0;
    const double zero = 0.0;
    for (int q0 = 0; q0 < qt; q0 += max_sub) {
      const int sub = std::min(max_sub, qt - q0);
      const Size sub_nn = Size(sub) * nn;
      if (cudaMemcpy(d_raw.ptr, raw_tile + Size(q0) * nn, sub_nn * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return 3;
      }
      // raw_tile row-major [sub, nn] => col-major [nn, sub]; OP_T computes [sub, nn] @ [nn]
      if (cublasDgemv(ch.handle, CUBLAS_OP_T, nn_int, sub, &one, d_raw.ptr, nn_int, d_bar_g.ptr, 1,
                      &zero, d_bar_rho.ptr, 1) != CUBLAS_STATUS_SUCCESS) {
        return 4;
      }
      if (cudaMemcpy(bar_rho_tile + q0, d_bar_rho.ptr, Size(sub) * sizeof(double),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 5;
      }
      if (write_bar) {
        if (cudaMemcpy(d_rho.ptr, rho_tile + q0, Size(sub) * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_bar_raw.ptr, bar_raw_tile + Size(q0) * nn, sub_nn * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return 3;
        }
        k_outer_q_nn<<<grid_for(sub_nn), 256>>>(sub, nn_int, d_rho.ptr, d_bar_g.ptr, d_bar_raw.ptr,
                                                true);
        if (cudaGetLastError() != cudaSuccess) return 4;
        if (cudaMemcpy(bar_raw_tile + Size(q0) * nn, d_bar_raw.ptr, sub_nn * sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          return 5;
        }
      }
    }
    if (!check_finite(bar_rho_tile, Size(qt)) ||
        (write_bar && !check_finite(bar_raw_tile, Size(qt) * nn))) {
      return kErrNonFinite;
    }
    return kOk;
#endif
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

// 6. Core reverse + Coulomb metric implicit solve pullback (exact match to coulomb_fit_response).
int GQC_EXT_FOCK_SYM(gqc_f12_ext_fock_rev)(int q, int n, int oa, int ob, const double* whitener,
                                           const double* occ_a, const double* occ_b,
                                           const double* R_a, const double* R_b, const double* D_a,
                                           const double* D_b, const double* bar_rho,
                                           const double* k_seed_a, const double* k_seed_b,
                                           double* bar_R_a, double* bar_R_b, double* bar_occ_a_acc,
                                           double* bar_occ_b_acc, double* metric_bar) {
  try {
    if (q < 0 || n < 0 || oa < 0 || ob < 0) return kErrInvalidArg;
    if (q == 0 || n == 0) return kOk;
    if (!whitener || !bar_rho || !metric_bar) return kErrNullPtr;
    if (oa > 0 && (!occ_a || !R_a || !D_a || !k_seed_a || !bar_R_a || !bar_occ_a_acc)) {
      return kErrNullPtr;
    }
    if (ob > 0 && (!occ_b || !R_b || !D_b || !k_seed_b || !bar_R_b || !bar_occ_b_acc)) {
      return kErrNullPtr;
    }

    const Size qq = Size(q) * Size(q);
    if (oa == 0 && ob == 0) {
      std::fill_n(metric_bar, qq, 0.0);
      return kOk;
    }

    const Size nn = Size(n) * Size(n);
    const Size qn = Size(q) * Size(n);

    // Hard VRAM budget check (3*q^2 + q + n^2 fixed + 3*q*n + 2*n min 1-orbital chunk).
    Size three_qq = 0, tmp1 = 0, fixed_doubles = 0, three_qn = 0, per_o_doubles = 0;
    if (!checked_mul(Size(3), qq, three_qq) || !checked_add(three_qq, Size(q), tmp1) ||
        !checked_add(tmp1, nn, fixed_doubles) || !checked_mul(Size(3), qn, three_qn) ||
        !checked_add(three_qn, Size(2) * Size(n), per_o_doubles) ||
        fixed_doubles > kVramCoreTotalBudgetDoubles ||
        per_o_doubles > kVramCoreTotalBudgetDoubles - fixed_doubles) {
      return kErrOom;
    }

#ifdef GQC_SINK_HOST_REFERENCE
    std::vector<double> weight(qq, 0.0);
    if (oa > 0) {
      const int rc = run_rev_spin_host(q, n, oa, whitener, occ_a, R_a, D_a, bar_rho, k_seed_a,
                                       bar_R_a, bar_occ_a_acc, weight.data());
      if (rc != 0) return rc;
    }
    if (ob > 0) {
      const int rc = run_rev_spin_host(q, n, ob, whitener, occ_b, R_b, D_b, bar_rho, k_seed_b,
                                       bar_R_b, bar_occ_b_acc, weight.data());
      if (rc != 0) return rc;
    }
    for (int a = 0; a < q; ++a) {
      for (int b = 0; b < q; ++b) {
        metric_bar[Size(a) * q + b] = -0.5 * (weight[Size(a) * q + b] + weight[Size(b) * q + a]);
      }
    }
#else
    CublasHandle ch;
    if (ch.init() != CUBLAS_STATUS_SUCCESS) return 1;

    const Size chunk_budget_doubles = kVramCoreTotalBudgetDoubles - fixed_doubles;

    DeviceBuffer d_W, d_weight, d_metric_bar, d_bar_rho;
    if (!d_W.alloc(qq) || !d_weight.alloc(qq) || !d_metric_bar.alloc(qq) ||
        !d_bar_rho.alloc(Size(q))) {
      return kErrOom;
    }
    if (cudaMemcpy(d_W.ptr, whitener, qq * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_bar_rho.ptr, bar_rho, Size(q) * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        !d_weight.zero()) {
      return 3;
    }

    if (oa > 0) {
      const int rc = run_rev_spin_gpu(ch.handle, q, n, oa, d_W.ptr, d_bar_rho.ptr, occ_a, R_a, D_a,
                                      k_seed_a, bar_R_a, bar_occ_a_acc, d_weight.ptr,
                                      chunk_budget_doubles);
      if (rc != 0) return rc;
    }
    if (ob > 0) {
      const int rc = run_rev_spin_gpu(ch.handle, q, n, ob, d_W.ptr, d_bar_rho.ptr, occ_b, R_b, D_b,
                                      k_seed_b, bar_R_b, bar_occ_b_acc, d_weight.ptr,
                                      chunk_budget_doubles);
      if (rc != 0) return rc;
    }

    k_sym_scale<<<grid_for(qq), 256>>>(q, -0.5, d_weight.ptr, d_metric_bar.ptr);
    if (cudaGetLastError() != cudaSuccess) return 4;
    if (cudaMemcpy(metric_bar, d_metric_bar.ptr, qq * sizeof(double), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      return 5;
    }
#endif
    if (!check_finite(metric_bar, qq) ||
        (oa > 0 && (!check_finite(bar_R_a, qn * Size(oa)) ||
                    !check_finite(bar_occ_a_acc, Size(n) * Size(oa)))) ||
        (ob > 0 && (!check_finite(bar_R_b, qn * Size(ob)) ||
                    !check_finite(bar_occ_b_acc, Size(n) * Size(ob))))) {
      return kErrNonFinite;
    }
    return kOk;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

// 7. Q-tiled half-transform reverse for 0, 1, or 2 spins + optional rho[a] * bar_g[m, v].
//    Safe when bar_raw_tile == raw_tile (for inplace_raw=True streaming mode) when
//    add_to_bar_raw == 0.
int GQC_EXT_FOCK_SYM(gqc_f12_ext_fock_half_rev)(int qt, int n, int oa, int ob,
                                                const double* raw_tile, const double* rho_tile,
                                                const double* bar_g, const double* occ_a,
                                                const double* occ_b, const double* bar_R_a,
                                                const double* bar_R_b, double* bar_raw_tile,
                                                int add_to_bar_raw, double* bar_occ_a_acc,
                                                double* bar_occ_b_acc) {
  try {
    if (qt < 0 || n < 0 || oa < 0 || ob < 0) return kErrInvalidArg;
    if (qt == 0 || n == 0) return kOk;
    if (!raw_tile || !bar_raw_tile) return kErrNullPtr;
    if (oa > 0 && (!occ_a || !bar_R_a || !bar_occ_a_acc)) return kErrNullPtr;
    if (ob > 0 && (!occ_b || !bar_R_b || !bar_occ_b_acc)) return kErrNullPtr;
    if ((rho_tile != nullptr) != (bar_g != nullptr)) return kErrNullPtr;
    if (add_to_bar_raw && bar_raw_tile == raw_tile) return kErrAlias;

    const Size nn = Size(n) * Size(n);
    if (!fits_int(nn)) return kErrInvalidArg;
    const bool has_outer = (rho_tile != nullptr && bar_g != nullptr);
    const int omax = std::max(oa, ob);

    // Hard VRAM budget check (fixed resident + minimum 1-q tile) before allocation.
    Size sum_o = 0, occ_elems = 0, fixed_doubles = 0;
    Size two_nn = 0, two_omax_n = 0, per_q_tmp = 0, per_q_doubles = 0;
    if (!checked_add(Size(oa), Size(ob), sum_o) ||
        !checked_mul(Size(2) * Size(n), sum_o, occ_elems) ||
        !checked_add(has_outer ? nn : Size(0), occ_elems, fixed_doubles) ||
        !checked_mul(Size(2), nn, two_nn) ||
        !checked_mul(Size(2) * Size(omax), Size(n), two_omax_n) ||
        !checked_add(two_nn, two_omax_n, per_q_tmp) ||
        !checked_add(per_q_tmp, has_outer ? Size(1) : Size(0), per_q_doubles) ||
        fixed_doubles > kVramTileBudgetDoubles ||
        per_q_doubles > kVramTileBudgetDoubles - fixed_doubles) {
      return kErrOom;
    }

    const Size max_sub_mem = (kVramTileBudgetDoubles - fixed_doubles) / per_q_doubles;
    const Size max_sub_int = static_cast<Size>(std::numeric_limits<int>::max()) / Size(n);
    const int max_sub =
        static_cast<int>(std::min<Size>(Size(qt), std::min<Size>(max_sub_mem, max_sub_int)));
    if (max_sub < 1) return kErrInvalidArg;

#ifdef GQC_SINK_HOST_REFERENCE
    // Step 1: Compute bar_occ_s FIRST (reads raw_tile only) so overwriting bar_raw_tile == raw_tile
    // in Step 2 is 100% alias-safe with zero extra copy.
    const auto acc_bar_occ = [qt, n, nn, max_sub, raw_tile](int o, const double* bar_R,
                                                            double* bar_occ_acc) {
      if (o == 0) return;
      const Size on = Size(o) * Size(n);
      std::vector<double> bar_R_t(Size(max_sub) * on);
      for (int q0 = 0; q0 < qt; q0 += max_sub) {
        const int sub = std::min(max_sub, qt - q0);
        for (int a = 0; a < sub; ++a) {
          const double* src = bar_R + Size(q0 + a) * on;
          double* dst = bar_R_t.data() + Size(a) * on;
          for (int i = 0; i < o; ++i) {
            for (int m = 0; m < n; ++m) {
              dst[Size(m) * o + i] = src[Size(i) * n + m];
            }
          }
        }
        rm_gemm_host(true, false, n, o, static_cast<int>(Size(sub) * Size(n)), 1.0,
                     raw_tile + Size(q0) * nn, n, bar_R_t.data(), o, 1.0, bar_occ_acc, o);
      }
    };
    acc_bar_occ(oa, bar_R_a, bar_occ_a_acc);
    acc_bar_occ(ob, bar_R_b, bar_occ_b_acc);

    // Step 2: Populate or accumulate bar_raw_tile[a, m, v] = rho[a]*bar_g[m, v] + sum_s bar_R_s^T @ O_s^T.
    for (int a = 0; a < qt; ++a) {
      double* out_slice = bar_raw_tile + Size(a) * nn;
      if (has_outer) {
        const double r = rho_tile[a];
        if (add_to_bar_raw) {
          for (Size p = 0; p < nn; ++p) out_slice[p] += r * bar_g[p];
        } else {
          for (Size p = 0; p < nn; ++p) out_slice[p] = r * bar_g[p];
        }
      } else if (!add_to_bar_raw) {
        std::fill_n(out_slice, nn, 0.0);
      }
      if (oa > 0) {
        rm_gemm_host(true, true, n, n, oa, 1.0, bar_R_a + Size(a) * Size(oa) * n, n, occ_a, oa, 1.0,
                     out_slice, n);
      }
      if (ob > 0) {
        rm_gemm_host(true, true, n, n, ob, 1.0, bar_R_b + Size(a) * Size(ob) * n, n, occ_b, ob, 1.0,
                     out_slice, n);
      }
    }
#else
    CublasHandle ch;
    if (ch.init() != CUBLAS_STATUS_SUCCESS) return 1;

    const int nn_int = static_cast<int>(nn);

    DeviceBuffer d_raw, d_bar_raw, d_bar_g, d_rho;
    DeviceBuffer d_occ_a, d_occ_b, d_bar_occ_a, d_bar_occ_b, d_bar_R, d_bar_R_t;
    if (!d_raw.alloc(Size(max_sub) * nn) || !d_bar_raw.alloc(Size(max_sub) * nn)) return kErrOom;
    if (has_outer && (!d_bar_g.alloc(nn) || !d_rho.alloc(Size(max_sub)))) return kErrOom;
    if (oa > 0 && (!d_occ_a.alloc(Size(n) * oa) || !d_bar_occ_a.alloc(Size(n) * oa))) {
      return kErrOom;
    }
    if (ob > 0 && (!d_occ_b.alloc(Size(n) * ob) || !d_bar_occ_b.alloc(Size(n) * ob))) {
      return kErrOom;
    }
    if (omax > 0 &&
        (!d_bar_R.alloc(Size(max_sub) * omax * n) || !d_bar_R_t.alloc(Size(max_sub) * omax * n))) {
      return kErrOom;
    }

    if (has_outer &&
        cudaMemcpy(d_bar_g.ptr, bar_g, nn * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
      return 3;
    }
    if (oa > 0) {
      if (cudaMemcpy(d_occ_a.ptr, occ_a, Size(n) * oa * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess ||
          !d_bar_occ_a.zero()) {
        return 3;
      }
    }
    if (ob > 0) {
      if (cudaMemcpy(d_occ_b.ptr, occ_b, Size(n) * ob * sizeof(double), cudaMemcpyHostToDevice) !=
              cudaSuccess ||
          !d_bar_occ_b.zero()) {
        return 3;
      }
    }

    for (int q0 = 0; q0 < qt; q0 += max_sub) {
      const int sub = std::min(max_sub, qt - q0);
      const Size sub_nn = Size(sub) * nn;

      if (cudaMemcpy(d_raw.ptr, raw_tile + Size(q0) * nn, sub_nn * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return 3;
      }
      if (add_to_bar_raw) {
        if (cudaMemcpy(d_bar_raw.ptr, bar_raw_tile + Size(q0) * nn, sub_nn * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return 3;
        }
      }
      if (has_outer) {
        if (cudaMemcpy(d_rho.ptr, rho_tile + q0, Size(sub) * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return 3;
        }
        k_outer_q_nn<<<grid_for(sub_nn), 256>>>(sub, nn_int, d_rho.ptr, d_bar_g.ptr, d_bar_raw.ptr,
                                                add_to_bar_raw != 0);
        if (cudaGetLastError() != cudaSuccess) return 4;
      } else if (!add_to_bar_raw) {
        if (cudaMemset(d_bar_raw.ptr, 0, sub_nn * sizeof(double)) != cudaSuccess) return 3;
      }

      const auto do_spin = [&](int o, const double* h_bar_R, const double* d_occ,
                               double* d_bar_occ) -> int {
        if (o == 0) return kOk;
        const Size on = Size(o) * Size(n);
        const Size sub_on = Size(sub) * on;
        if (cudaMemcpy(d_bar_R.ptr, h_bar_R + Size(q0) * on, sub_on * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return 3;
        }
        k_transpose_qon_to_qno<<<grid_for(sub_on), 256>>>(sub, o, n, d_bar_R.ptr, d_bar_R_t.ptr);
        if (cudaGetLastError() != cudaSuccess) return 4;
        if (rm_gemm(ch.handle, true, false, n, o, static_cast<int>(Size(sub) * Size(n)), 1.0,
                    d_raw.ptr, n, d_bar_R_t.ptr, o, 1.0, d_bar_occ, o) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
        if (rm_gemm_batched(ch.handle, true, true, n, n, o, 1.0, d_bar_R.ptr, n,
                            static_cast<long long>(on), d_occ, o, 0, 1.0, d_bar_raw.ptr, n,
                            static_cast<long long>(nn), sub) != CUBLAS_STATUS_SUCCESS) {
          return 4;
        }
        return kOk;
      };

      if (const int rc = do_spin(oa, bar_R_a, d_occ_a.ptr, d_bar_occ_a.ptr); rc != 0) return rc;
      if (const int rc = do_spin(ob, bar_R_b, d_occ_b.ptr, d_bar_occ_b.ptr); rc != 0) return rc;

      if (cudaMemcpy(bar_raw_tile + Size(q0) * nn, d_bar_raw.ptr, sub_nn * sizeof(double),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 5;
      }
    }

    if (oa > 0) {
      std::vector<double> h_tmp(Size(n) * oa);
      if (cudaMemcpy(h_tmp.data(), d_bar_occ_a.ptr, Size(n) * oa * sizeof(double),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 5;
      }
      for (Size k = 0; k < Size(n) * oa; ++k) bar_occ_a_acc[k] += h_tmp[k];
    }
    if (ob > 0) {
      std::vector<double> h_tmp(Size(n) * ob);
      if (cudaMemcpy(h_tmp.data(), d_bar_occ_b.ptr, Size(n) * ob * sizeof(double),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 5;
      }
      for (Size k = 0; k < Size(n) * ob; ++k) bar_occ_b_acc[k] += h_tmp[k];
    }
#endif
    if (!check_finite(bar_raw_tile, Size(qt) * nn) ||
        (oa > 0 && !check_finite(bar_occ_a_acc, Size(n) * Size(oa))) ||
        (ob > 0 && !check_finite(bar_occ_b_acc, Size(n) * Size(ob)))) {
      return kErrNonFinite;
    }
    return kOk;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}

}  // extern "C"
