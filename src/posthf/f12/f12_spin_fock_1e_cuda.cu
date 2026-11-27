// Dynamic-dimension AOT C++/cuBLAS kernels for F12 one-electron nuclear gradient
// and extended UHF/RHF density-fitted spin Fock forward & pullback.
//
// Exports paired extern "C" *_host_reference (always built) and *_cuda (built
// under __CUDACC__) FP64 symbols with bounded GPU slab memory (<= 256 MiB/slab).

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#ifdef __CUDACC__
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace {

using Size = std::uint64_t;

constexpr int kOk = 0;
constexpr int kErrInvalidArg = -1;
constexpr int kErrNonFinite = -2;
constexpr int kErrOom = -3;
#ifdef __CUDACC__
constexpr int kErrCuda = -4;
constexpr int kErrCublas = -5;
#endif
constexpr int kErrInternal = -6;

constexpr std::size_t kMaxSlabBytes = std::size_t(256) << 20;  // 256 MiB

inline int compute_tile_width(int total_count, std::size_t fixed_bytes,
                              std::size_t bytes_per_item, int hint) {
  if (bytes_per_item == 0 || fixed_bytes >= kMaxSlabBytes) return 0;
  const std::size_t avail = kMaxSlabBytes - fixed_bytes;
  if (avail < bytes_per_item) return 0;
  const std::size_t cap = avail / bytes_per_item;
  int width = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(total_count), cap));
  if (hint > 0 && hint < width) {
    width = hint;
  }
  return width;
}

inline bool check_finite(const double* ptr, Size count) {
  if (!ptr || count == 0) return true;
  for (Size i = 0; i < count; ++i) {
    if (!std::isfinite(ptr[i])) return false;
  }
  return true;
}

// Host row-major GEMM: C = beta * C + alpha * op(A) @ op(B)
// A is (M, K) if !trans_a else (K, M) with row stride lda.
// B is (K, N) if !trans_b else (N, K) with row stride ldb.
// C is (M, N) with row stride ldc.
void rm_gemm_host(bool trans_a, bool trans_b, int M, int N, int K, double alpha,
                  const double* A, int lda, const double* B, int ldb,
                  double beta, double* C, int ldc) {
  if (beta == 0.0) {
    for (int i = 0; i < M; ++i) {
      std::memset(C + Size(i) * ldc, 0, Size(N) * sizeof(double));
    }
  } else if (beta != 1.0) {
    for (int i = 0; i < M; ++i) {
      double* ci = C + Size(i) * ldc;
      for (int j = 0; j < N; ++j) ci[j] *= beta;
    }
  }
  if (M == 0 || N == 0 || K == 0 || alpha == 0.0) return;

  if (!trans_a && !trans_b) {
    // C[i, j] += alpha * sum_k A[i, k] * B[k, j]
    for (int i = 0; i < M; ++i) {
      const double* ai = A + Size(i) * lda;
      double* ci = C + Size(i) * ldc;
      for (int k = 0; k < K; ++k) {
        const double aik = alpha * ai[k];
        if (aik == 0.0) continue;
        const double* bk = B + Size(k) * ldb;
        for (int j = 0; j < N; ++j) {
          ci[j] += aik * bk[j];
        }
      }
    }
  } else if (trans_a && !trans_b) {
    // C[i, j] += alpha * sum_k A[k, i] * B[k, j]
    for (int k = 0; k < K; ++k) {
      const double* ak = A + Size(k) * lda;
      const double* bk = B + Size(k) * ldb;
      for (int i = 0; i < M; ++i) {
        const double aki = alpha * ak[i];
        if (aki == 0.0) continue;
        double* ci = C + Size(i) * ldc;
        for (int j = 0; j < N; ++j) {
          ci[j] += aki * bk[j];
        }
      }
    }
  } else if (!trans_a && trans_b) {
    // C[i, j] += alpha * sum_k A[i, k] * B[j, k]
    for (int i = 0; i < M; ++i) {
      const double* ai = A + Size(i) * lda;
      double* ci = C + Size(i) * ldc;
      for (int j = 0; j < N; ++j) {
        const double* bj = B + Size(j) * ldb;
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
          acc += ai[k] * bj[k];
        }
        ci[j] += alpha * acc;
      }
    }
  } else {
    // C[i, j] += alpha * sum_k A[k, i] * B[j, k]
    for (int i = 0; i < M; ++i) {
      double* ci = C + Size(i) * ldc;
      for (int j = 0; j < N; ++j) {
        const double* bj = B + Size(j) * ldb;
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
          acc += A[Size(k) * lda + i] * bj[k];
        }
        ci[j] += alpha * acc;
      }
    }
  }
}

void compute_tile_charge_host(int oa, int ob, int n, int qt, int share_beta,
                              const double* fit_a_tile, const double* fit_b_tile,
                              double* charge_tile) {
  const Size stride_a = Size(oa) * n;
  const Size stride_b = Size(ob) * n;
  for (int a = 0; a < qt; ++a) {
    double c = 0.0;
    if (oa > 0) {
      const double* fa = fit_a_tile + Size(a) * stride_a;
      double ta = 0.0;
      for (int i = 0; i < oa; ++i) ta += fa[Size(i) * n + i];
      c += share_beta ? (2.0 * ta) : ta;
    }
    if (!share_beta && ob > 0) {
      const double* fb = fit_b_tile + Size(a) * stride_b;
      double tb = 0.0;
      for (int i = 0; i < ob; ++i) tb += fb[Size(i) * n + i];
      c += tb;
    }
    charge_tile[a] = c;
  }
}

#ifdef __CUDACC__

struct DevBuf {
  double* ptr = nullptr;
  ~DevBuf() {
    if (ptr) cudaFree(ptr);
  }
  cudaError_t alloc(std::size_t count) {
    if (count == 0) return cudaSuccess;
    return cudaMalloc(&ptr, count * sizeof(double));
  }
};

struct CublasGuard {
  cublasHandle_t handle = nullptr;
  ~CublasGuard() {
    if (handle) cublasDestroy(handle);
  }
  cublasStatus_t init() { return cublasCreate(&handle); }
};

inline cublasStatus_t rm_gemm_cuda(cublasHandle_t h, bool trans_a, bool trans_b,
                                   int M, int N, int K, double alpha,
                                   const double* A, int lda, const double* B,
                                   int ldb, double beta, double* C, int ldc) {
  return cublasDgemm(h, trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                     trans_a ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B,
                     ldb, A, lda, &beta, C, ldc);
}

__global__ void k_slice_occ_raw(int qt, int occ, int n, const double* raw,
                                double* occ_raw) {
  const Size total = Size(qt) * occ * n;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int r = static_cast<int>(idx % n);
  const Size ai = idx / n;
  const int i = static_cast<int>(ai % occ);
  const int a = static_cast<int>(ai / occ);
  occ_raw[idx] = raw[(Size(a) * n + i) * n + r];
}

__global__ void k_add_occ_raw(int qt, int occ, int n, const double* bar_occ_raw,
                              double* bar_raw) {
  const Size total = Size(qt) * occ * n;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int r = static_cast<int>(idx % n);
  const Size ai = idx / n;
  const int i = static_cast<int>(ai % occ);
  const int a = static_cast<int>(ai / occ);
  bar_raw[(Size(a) * n + i) * n + r] += bar_occ_raw[idx];
}

__global__ void k_outer_charge_bar_g(int qt, Size nn, const double* charge,
                                     const double* bar_g, double* bar_raw,
                                     int accumulate_out) {
  const Size total = Size(qt) * nn;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int a = static_cast<int>(idx / nn);
  const Size pr = idx % nn;
  const double val = bar_g ? (charge[a] * bar_g[pr]) : 0.0;
  if (accumulate_out) {
    bar_raw[idx] += val;
  } else {
    bar_raw[idx] = val;
  }
}

__global__ void k_add_diag_bar_c(int qt, int occ, int n, const double* bar_c,
                                 double* bar_fit) {
  const Size total = Size(qt) * occ;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int a = static_cast<int>(idx / occ);
  const int i = static_cast<int>(idx % occ);
  bar_fit[(Size(a) * occ + i) * n + i] += bar_c[a];
}

inline int launch_blocks(Size total, int threads) {
  return static_cast<int>((total + Size(threads) - 1) / Size(threads));
}

#endif  // __CUDACC__

}  // namespace

// ============================================================================
// 1. One-electron nuclear gradient
// ============================================================================

extern "C" int gqc_f12_one_electron_grad_host_reference(
    int natoms,
    int n,
    const double* d_overlap,
    const double* d_hcore,
    const double* overlap_w,
    const double* hcore_w,
    double* grad_out) {
  try {
    if (natoms <= 0 || n <= 0) return kErrInvalidArg;
    if (!d_overlap || !d_hcore || !overlap_w || !hcore_w || !grad_out) {
      return kErrInvalidArg;
    }
    const Size rows = Size(3) * natoms;
    const Size nn = Size(n) * n;
    if (rows > Size(INT_MAX) || nn > Size(INT_MAX)) return kErrInvalidArg;
    if (!check_finite(overlap_w, nn) || !check_finite(hcore_w, nn)) {
      return kErrNonFinite;
    }
    for (Size r = 0; r < rows; ++r) {
      const double* ov_row = d_overlap + r * nn;
      const double* hc_row = d_hcore + r * nn;
      double sum_ov = 0.0, comp_ov = 0.0;
      double sum_hc = 0.0, comp_hc = 0.0;
      for (Size k = 0; k < nn; ++k) {
        const double x_ov = ov_row[k] * overlap_w[k];
        const double t_ov = sum_ov + x_ov;
        comp_ov += (std::abs(sum_ov) >= std::abs(x_ov))
                       ? ((sum_ov - t_ov) + x_ov)
                       : ((x_ov - t_ov) + sum_ov);
        sum_ov = t_ov;

        const double x_hc = hc_row[k] * hcore_w[k];
        const double t_hc = sum_hc + x_hc;
        comp_hc += (std::abs(sum_hc) >= std::abs(x_hc))
                       ? ((sum_hc - t_hc) + x_hc)
                       : ((x_hc - t_hc) + sum_hc);
        sum_hc = t_hc;
      }
      const double acc = (sum_ov + comp_ov) + (sum_hc + comp_hc);
      if (!std::isfinite(acc)) return kErrNonFinite;
      grad_out[r] = acc;
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

#ifdef __CUDACC__
extern "C" int gqc_f12_one_electron_grad_cuda(
    int natoms,
    int n,
    const double* d_overlap,
    const double* d_hcore,
    const double* overlap_w,
    const double* hcore_w,
    double* grad_out) {
  try {
    if (natoms <= 0 || n <= 0) return kErrInvalidArg;
    if (!d_overlap || !d_hcore || !overlap_w || !hcore_w || !grad_out) {
      return kErrInvalidArg;
    }
    const Size rows64 = Size(3) * natoms;
    const Size nn64 = Size(n) * n;
    if (rows64 > Size(INT_MAX) || nn64 > Size(INT_MAX)) return kErrInvalidArg;
    const int rows = static_cast<int>(rows64);
    const int nn = static_cast<int>(nn64);
    if (!check_finite(overlap_w, nn64) || !check_finite(hcore_w, nn64)) {
      return kErrNonFinite;
    }

    const std::size_t fixed_bytes = (Size(2) * nn64 + rows64) * sizeof(double);
    const int chunk_cap = compute_tile_width(rows, fixed_bytes, nn64 * sizeof(double), 0);
    if (chunk_cap < 1) return kErrOom;

    CublasGuard cg;
    if (cg.init() != CUBLAS_STATUS_SUCCESS) return kErrCublas;

    DevBuf dev_ov_w, dev_hc_w, dev_grad, dev_chunk;
    if (dev_ov_w.alloc(nn64) != cudaSuccess ||
        dev_hc_w.alloc(nn64) != cudaSuccess ||
        dev_grad.alloc(rows64) != cudaSuccess ||
        dev_chunk.alloc(Size(chunk_cap) * nn64) != cudaSuccess) {
      return kErrOom;
    }

    if (cudaMemcpy(dev_ov_w.ptr, overlap_w, nn64 * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_hc_w.ptr, hcore_w, nn64 * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return kErrCuda;
    }

    const double one = 1.0;
    const double zero = 0.0;

    for (int r0 = 0; r0 < rows; r0 += chunk_cap) {
      const int cr = std::min(chunk_cap, rows - r0);
      const std::size_t chunk_bytes = Size(cr) * nn64 * sizeof(double);

      // Row-major (cr, nn) == col-major (nn, cr); CUBLAS_OP_T computes dot per row.
      if (cudaMemcpy(dev_chunk.ptr, d_overlap + Size(r0) * nn64, chunk_bytes,
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return kErrCuda;
      }
      if (cublasDgemv(cg.handle, CUBLAS_OP_T, nn, cr, &one, dev_chunk.ptr, nn,
                      dev_ov_w.ptr, 1, &zero, dev_grad.ptr + r0, 1) !=
          CUBLAS_STATUS_SUCCESS) {
        return kErrCublas;
      }

      if (cudaMemcpy(dev_chunk.ptr, d_hcore + Size(r0) * nn64, chunk_bytes,
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return kErrCuda;
      }
      if (cublasDgemv(cg.handle, CUBLAS_OP_T, nn, cr, &one, dev_chunk.ptr, nn,
                      dev_hc_w.ptr, 1, &one, dev_grad.ptr + r0, 1) !=
          CUBLAS_STATUS_SUCCESS) {
        return kErrCublas;
      }
    }

    if (cudaMemcpy(grad_out, dev_grad.ptr, rows64 * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return kErrCuda;
    }
    if (!check_finite(grad_out, rows64)) return kErrNonFinite;
    return kOk;
  } catch (const std::bad_alloc&) {
    return kErrOom;
  } catch (const std::length_error&) {
    return kErrOom;
  } catch (...) {
    return kErrInternal;
  }
}
#endif

// ============================================================================
// 2. Spin Fock forward
// ============================================================================

extern "C" int gqc_f12_spin_fock_fwd_host_reference(
    int oa,
    int ob,
    int n,
    int q,
    int has_hcore_a,
    int has_hcore_b,
    int share_beta,
    int q_tile_hint,
    const double* hcore_a,
    const double* hcore_b,
    const double* raw_a,
    const double* raw_b,
    const double* fit_a,
    const double* fit_b,
    double* fock_a,
    double* fock_b,
    double* exchange_a,
    double* exchange_b) {
  try {
    if (oa < 0 || ob < 0 || oa + ob < 1 || n <= 0 || oa > n || ob > n || q <= 0) {
      return kErrInvalidArg;
    }
    if (share_beta && oa != ob) return kErrInvalidArg;
    if (!raw_a || !fock_a || !fock_b) return kErrInvalidArg;
    if (!share_beta && !raw_b) return kErrInvalidArg;
    if (oa > 0 && !fit_a) return kErrInvalidArg;
    if (!share_beta && ob > 0 && !fit_b) return kErrInvalidArg;
    if (has_hcore_a && !hcore_a) return kErrInvalidArg;
    if (has_hcore_b && !hcore_b) return kErrInvalidArg;

    const Size nn = Size(n) * n;
    if (nn > Size(INT_MAX)) return kErrInvalidArg;

    const int max_o = std::max(oa, ob);
    const std::size_t fixed_mats =
        std::size_t(1) + (oa > 0 ? 1 : 0) + (!share_beta ? 1 : 0) + (!share_beta && ob > 0 ? 1 : 0);
    const std::size_t fixed_bytes = fixed_mats * nn * sizeof(double);
    const std::size_t per_q_bytes = (nn + Size(2) * max_o * n + 1) * sizeof(double);
    int qt_cap = compute_tile_width(q, fixed_bytes, per_q_bytes, q_tile_hint);
    if (qt_cap < 1) return kErrOom;
    if (max_o > 0) {
      qt_cap = std::min(qt_cap, INT_MAX / max_o);
      if (qt_cap < 1) return kErrInvalidArg;
    }

    std::vector<double> J_a(nn, 0.0);
    std::vector<double> K_a(nn, 0.0);
    std::vector<double> J_b(share_beta ? 0 : nn, 0.0);
    std::vector<double> K_b(share_beta ? 0 : nn, 0.0);
    std::vector<double> charge_tile(static_cast<std::size_t>(qt_cap), 0.0);
    std::vector<double> occ_raw(Size(qt_cap) * max_o * n);

    for (int q0 = 0; q0 < q; q0 += qt_cap) {
      const int qt = std::min(qt_cap, q - q0);
      const double* raw_a_tile = raw_a + Size(q0) * nn;
      const double* fit_a_tile = oa > 0 ? (fit_a + Size(q0) * oa * n) : nullptr;
      const double* fit_b_tile = (!share_beta && ob > 0) ? (fit_b + Size(q0) * ob * n) : nullptr;

      compute_tile_charge_host(oa, ob, n, qt, share_beta, fit_a_tile, fit_b_tile,
                               charge_tile.data());

      // Alpha Coulomb: J_a[pr] += sum_a raw_a[a, pr] * charge[a]
      for (int a = 0; a < qt; ++a) {
        const double ca = charge_tile[a];
        if (ca == 0.0) continue;
        const double* ra = raw_a_tile + Size(a) * nn;
        for (Size pr = 0; pr < nn; ++pr) J_a[pr] += ca * ra[pr];
      }
      // Alpha Exchange: K_a[p, r] += sum_{a, i} fit_a[a, i, p] * raw_a[a, i, r]
      if (oa > 0) {
        for (int a = 0; a < qt; ++a) {
          std::memcpy(occ_raw.data() + Size(a) * oa * n, raw_a_tile + Size(a) * nn,
                      Size(oa) * n * sizeof(double));
        }
        rm_gemm_host(true, false, n, n, qt * oa, 1.0, fit_a_tile, n,
                     occ_raw.data(), n, 1.0, K_a.data(), n);
      }

      if (!share_beta) {
        const double* raw_b_tile = raw_b + Size(q0) * nn;
        for (int a = 0; a < qt; ++a) {
          const double ca = charge_tile[a];
          if (ca == 0.0) continue;
          const double* rb = raw_b_tile + Size(a) * nn;
          for (Size pr = 0; pr < nn; ++pr) J_b[pr] += ca * rb[pr];
        }
        if (ob > 0) {
          for (int a = 0; a < qt; ++a) {
            std::memcpy(occ_raw.data() + Size(a) * ob * n, raw_b_tile + Size(a) * nn,
                        Size(ob) * n * sizeof(double));
          }
          rm_gemm_host(true, false, n, n, qt * ob, 1.0, fit_b_tile, n,
                       occ_raw.data(), n, 1.0, K_b.data(), n);
        }
      }
    }

    for (Size pr = 0; pr < nn; ++pr) {
      const double ha = has_hcore_a ? hcore_a[pr] : 0.0;
      const double hb = has_hcore_b ? hcore_b[pr] : 0.0;
      const double ja = J_a[pr];
      const double ka = K_a[pr];
      const double jb = share_beta ? ja : J_b[pr];
      const double kb = share_beta ? ka : K_b[pr];
      fock_a[pr] = ha + ja - ka;
      fock_b[pr] = hb + jb - kb;
      if (exchange_a) exchange_a[pr] = ka;
      if (exchange_b) exchange_b[pr] = kb;
    }

    if (!check_finite(fock_a, nn) || !check_finite(fock_b, nn) ||
        (exchange_a && !check_finite(exchange_a, nn)) ||
        (exchange_b && !check_finite(exchange_b, nn))) {
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

#ifdef __CUDACC__
extern "C" int gqc_f12_spin_fock_fwd_cuda(
    int oa,
    int ob,
    int n,
    int q,
    int has_hcore_a,
    int has_hcore_b,
    int share_beta,
    int q_tile_hint,
    const double* hcore_a,
    const double* hcore_b,
    const double* raw_a,
    const double* raw_b,
    const double* fit_a,
    const double* fit_b,
    double* fock_a,
    double* fock_b,
    double* exchange_a,
    double* exchange_b) {
  try {
    if (oa < 0 || ob < 0 || oa + ob < 1 || n <= 0 || oa > n || ob > n || q <= 0) {
      return kErrInvalidArg;
    }
    if (share_beta && oa != ob) return kErrInvalidArg;
    if (!raw_a || !fock_a || !fock_b) return kErrInvalidArg;
    if (!share_beta && !raw_b) return kErrInvalidArg;
    if (oa > 0 && !fit_a) return kErrInvalidArg;
    if (!share_beta && ob > 0 && !fit_b) return kErrInvalidArg;
    if (has_hcore_a && !hcore_a) return kErrInvalidArg;
    if (has_hcore_b && !hcore_b) return kErrInvalidArg;

    const Size nn64 = Size(n) * n;
    if (nn64 > Size(INT_MAX)) return kErrInvalidArg;
    const int nn = static_cast<int>(nn64);

    const int max_o = std::max(oa, ob);
    const std::size_t fixed_mats =
        std::size_t(1) + (oa > 0 ? 1 : 0) + (!share_beta ? 1 : 0) + (!share_beta && ob > 0 ? 1 : 0);
    const std::size_t fixed_bytes = fixed_mats * nn64 * sizeof(double);
    const std::size_t per_q_bytes = (nn64 + Size(2) * max_o * n + 1) * sizeof(double);
    int qt_cap = compute_tile_width(q, fixed_bytes, per_q_bytes, q_tile_hint);
    if (qt_cap < 1) return kErrOom;
    if (max_o > 0) {
      qt_cap = std::min(qt_cap, INT_MAX / max_o);
      if (qt_cap < 1) return kErrInvalidArg;
    }

    CublasGuard cg;
    if (cg.init() != CUBLAS_STATUS_SUCCESS) return kErrCublas;

    DevBuf d_raw, d_fit, d_occ_raw, d_charge;
    DevBuf d_J_a, d_K_a, d_J_b, d_K_b;
    if (d_raw.alloc(Size(qt_cap) * nn64) != cudaSuccess ||
        d_fit.alloc(Size(qt_cap) * max_o * n) != cudaSuccess ||
        d_occ_raw.alloc(Size(qt_cap) * max_o * n) != cudaSuccess ||
        d_charge.alloc(static_cast<std::size_t>(qt_cap)) != cudaSuccess ||
        d_J_a.alloc(nn64) != cudaSuccess ||
        (oa > 0 && d_K_a.alloc(nn64) != cudaSuccess) ||
        (!share_beta && d_J_b.alloc(nn64) != cudaSuccess) ||
        (!share_beta && ob > 0 && d_K_b.alloc(nn64) != cudaSuccess)) {
      return kErrOom;
    }

    std::vector<double> h_charge(static_cast<std::size_t>(qt_cap), 0.0);
    const double one = 1.0;
    constexpr int kThreads = 256;

    for (int q0 = 0; q0 < q; q0 += qt_cap) {
      const int qt = std::min(qt_cap, q - q0);
      const double beta_acc = (q0 == 0) ? 0.0 : 1.0;

      const double* raw_a_tile = raw_a + Size(q0) * nn64;
      const double* fit_a_tile = oa > 0 ? (fit_a + Size(q0) * oa * n) : nullptr;
      const double* fit_b_tile = (!share_beta && ob > 0) ? (fit_b + Size(q0) * ob * n) : nullptr;

      compute_tile_charge_host(oa, ob, n, qt, share_beta, fit_a_tile, fit_b_tile,
                               h_charge.data());
      if (cudaMemcpy(d_charge.ptr, h_charge.data(), Size(qt) * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return kErrCuda;
      }

      // Alpha spin
      if (cudaMemcpy(d_raw.ptr, raw_a_tile, Size(qt) * nn64 * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return kErrCuda;
      }
      if (cublasDgemv(cg.handle, CUBLAS_OP_N, nn, qt, &one, d_raw.ptr, nn,
                      d_charge.ptr, 1, &beta_acc, d_J_a.ptr, 1) !=
          CUBLAS_STATUS_SUCCESS) {
        return kErrCublas;
      }
      if (oa > 0) {
        if (cudaMemcpy(d_fit.ptr, fit_a_tile, Size(qt) * oa * n * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return kErrCuda;
        }
        const Size occ_total = Size(qt) * oa * n;
        k_slice_occ_raw<<<launch_blocks(occ_total, kThreads), kThreads>>>(
            qt, oa, n, d_raw.ptr, d_occ_raw.ptr);
        if (cudaGetLastError() != cudaSuccess) return kErrCuda;
        if (rm_gemm_cuda(cg.handle, true, false, n, n, qt * oa, 1.0, d_fit.ptr, n,
                         d_occ_raw.ptr, n, beta_acc, d_K_a.ptr, n) !=
            CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
      }

      // Beta spin (skipped when share_beta != 0)
      if (!share_beta) {
        const double* raw_b_tile = raw_b + Size(q0) * nn64;
        if (cudaMemcpy(d_raw.ptr, raw_b_tile, Size(qt) * nn64 * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return kErrCuda;
        }
        if (cublasDgemv(cg.handle, CUBLAS_OP_N, nn, qt, &one, d_raw.ptr, nn,
                        d_charge.ptr, 1, &beta_acc, d_J_b.ptr, 1) !=
            CUBLAS_STATUS_SUCCESS) {
          return kErrCublas;
        }
        if (ob > 0) {
          if (cudaMemcpy(d_fit.ptr, fit_b_tile, Size(qt) * ob * n * sizeof(double),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return kErrCuda;
          }
          const Size occ_total = Size(qt) * ob * n;
          k_slice_occ_raw<<<launch_blocks(occ_total, kThreads), kThreads>>>(
              qt, ob, n, d_raw.ptr, d_occ_raw.ptr);
          if (cudaGetLastError() != cudaSuccess) return kErrCuda;
          if (rm_gemm_cuda(cg.handle, true, false, n, n, qt * ob, 1.0, d_fit.ptr, n,
                           d_occ_raw.ptr, n, beta_acc, d_K_b.ptr, n) !=
              CUBLAS_STATUS_SUCCESS) {
            return kErrCublas;
          }
        }
      }
    }

    std::vector<double> J_a(nn64, 0.0), K_a(nn64, 0.0);
    std::vector<double> J_b(share_beta ? 0 : nn64, 0.0);
    std::vector<double> K_b(share_beta ? 0 : nn64, 0.0);

    if (cudaMemcpy(J_a.data(), d_J_a.ptr, nn64 * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return kErrCuda;
    }
    if (oa > 0 && cudaMemcpy(K_a.data(), d_K_a.ptr, nn64 * sizeof(double),
                             cudaMemcpyDeviceToHost) != cudaSuccess) {
      return kErrCuda;
    }
    if (!share_beta) {
      if (cudaMemcpy(J_b.data(), d_J_b.ptr, nn64 * sizeof(double),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return kErrCuda;
      }
      if (ob > 0 && cudaMemcpy(K_b.data(), d_K_b.ptr, nn64 * sizeof(double),
                               cudaMemcpyDeviceToHost) != cudaSuccess) {
        return kErrCuda;
      }
    }

    for (Size pr = 0; pr < nn64; ++pr) {
      const double ha = has_hcore_a ? hcore_a[pr] : 0.0;
      const double hb = has_hcore_b ? hcore_b[pr] : 0.0;
      const double ja = J_a[pr];
      const double ka = K_a[pr];
      const double jb = share_beta ? ja : J_b[pr];
      const double kb = share_beta ? ka : K_b[pr];
      fock_a[pr] = ha + ja - ka;
      fock_b[pr] = hb + jb - kb;
      if (exchange_a) exchange_a[pr] = ka;
      if (exchange_b) exchange_b[pr] = kb;
    }

    if (!check_finite(fock_a, nn64) || !check_finite(fock_b, nn64) ||
        (exchange_a && !check_finite(exchange_a, nn64)) ||
        (exchange_b && !check_finite(exchange_b, nn64))) {
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
#endif

// ============================================================================
// 3. Spin Fock reverse (pullback)
// ============================================================================

extern "C" int gqc_f12_spin_fock_rev_host_reference(
    int oa,
    int ob,
    int n,
    int q,
    int share_beta_raw,
    int accumulate_out,
    int q_tile_hint,
    const double* raw_a,
    const double* raw_b,
    const double* fit_a,
    const double* fit_b,
    const double* bar_fock_a,
    const double* bar_fock_b,
    const double* bar_exchange_a,
    const double* bar_exchange_b,
    double* bar_hcore_a,
    double* bar_hcore_b,
    double* bar_raw_a,
    double* bar_raw_b,
    double* bar_fit_a,
    double* bar_fit_b) {
  try {
    if (oa < 0 || ob < 0 || oa + ob < 1 || n <= 0 || oa > n || ob > n || q <= 0) {
      return kErrInvalidArg;
    }
    if (share_beta_raw && (oa != ob || accumulate_out != 0)) {
      return kErrInvalidArg;
    }
    if (!raw_a) return kErrInvalidArg;
    if (!share_beta_raw && !raw_b) return kErrInvalidArg;
    if (oa > 0 && !fit_a) return kErrInvalidArg;
    if (!share_beta_raw && ob > 0 && !fit_b) return kErrInvalidArg;

    const Size nn = Size(n) * n;
    if (nn > Size(INT_MAX)) return kErrInvalidArg;

    const bool has_g_a = (bar_fock_a != nullptr);
    const bool has_g_b = share_beta_raw ? has_g_a : (bar_fock_b != nullptr);
    const bool has_k_a = (bar_exchange_a != nullptr);
    const bool has_k_b = share_beta_raw ? has_k_a : (bar_exchange_b != nullptr);
    const bool has_keff_a = has_g_a || has_k_a;
    const bool has_keff_b = has_g_b || has_k_b;

    const int max_o = std::max(oa, ob);
    const std::size_t fixed_mats =
        (has_g_a ? 1 : 0) + (!share_beta_raw && has_g_b ? 1 : 0) +
        (has_keff_a ? 1 : 0) + (!share_beta_raw && has_keff_b ? 1 : 0);
    const std::size_t fixed_bytes = fixed_mats * nn * sizeof(double);
    const std::size_t per_q_bytes = (nn + Size(3) * max_o * n + 2) * sizeof(double);
    int qt_cap = compute_tile_width(q, fixed_bytes, per_q_bytes, q_tile_hint);
    if (qt_cap < 1) return kErrOom;
    if (max_o > 0) {
      qt_cap = std::min(qt_cap, INT_MAX / max_o);
      if (qt_cap < 1) return kErrInvalidArg;
    }

    // Build effective exchange cotangents: bar_K_eff_s = bar_exchange_s - bar_fock_s
    std::vector<double> bar_keff_a(has_keff_a ? nn : 0, 0.0);
    if (has_keff_a) {
      for (Size pr = 0; pr < nn; ++pr) {
        bar_keff_a[pr] = (has_k_a ? bar_exchange_a[pr] : 0.0) -
                         (has_g_a ? bar_fock_a[pr] : 0.0);
      }
    }
    std::vector<double> bar_keff_b((!share_beta_raw && has_keff_b) ? nn : 0, 0.0);
    if (!share_beta_raw && has_keff_b) {
      for (Size pr = 0; pr < nn; ++pr) {
        bar_keff_b[pr] = (has_k_b ? bar_exchange_b[pr] : 0.0) -
                         (has_g_b ? bar_fock_b[pr] : 0.0);
      }
    }

    // hcore cotangents: bar_hcore_s (+)= bar_fock_s
    if (bar_hcore_a) {
      for (Size pr = 0; pr < nn; ++pr) {
        const double v = has_g_a ? bar_fock_a[pr] : 0.0;
        if (accumulate_out) bar_hcore_a[pr] += v;
        else bar_hcore_a[pr] = v;
      }
    }
    if (bar_hcore_b) {
      const double* src_gb = share_beta_raw ? bar_fock_a : bar_fock_b;
      for (Size pr = 0; pr < nn; ++pr) {
        const double v = src_gb ? src_gb[pr] : 0.0;
        if (accumulate_out) bar_hcore_b[pr] += v;
        else bar_hcore_b[pr] = v;
      }
    }

    std::vector<double> charge_tile(static_cast<std::size_t>(qt_cap), 0.0);
    std::vector<double> bar_c_tile(static_cast<std::size_t>(qt_cap), 0.0);
    std::vector<double> occ_buf(Size(qt_cap) * max_o * n);

    for (int q0 = 0; q0 < q; q0 += qt_cap) {
      const int qt = std::min(qt_cap, q - q0);
      const double* raw_a_tile = raw_a + Size(q0) * nn;
      const double* fit_a_tile = oa > 0 ? (fit_a + Size(q0) * oa * n) : nullptr;
      const double* raw_b_tile = !share_beta_raw ? (raw_b + Size(q0) * nn) : nullptr;
      const double* fit_b_tile = (!share_beta_raw && ob > 0) ? (fit_b + Size(q0) * ob * n) : nullptr;

      compute_tile_charge_host(oa, ob, n, qt, share_beta_raw, fit_a_tile, fit_b_tile,
                               charge_tile.data());

      // bar_c[a] = dot(raw_a[a], bar_fock_a) + dot(raw_b[a], bar_fock_b)
      for (int a = 0; a < qt; ++a) {
        double bc = 0.0;
        if (has_g_a) {
          const double* ra = raw_a_tile + Size(a) * nn;
          double dot_a = 0.0;
          for (Size pr = 0; pr < nn; ++pr) dot_a += ra[pr] * bar_fock_a[pr];
          bc += share_beta_raw ? (2.0 * dot_a) : dot_a;
        }
        if (!share_beta_raw && has_g_b) {
          const double* rb = raw_b_tile + Size(a) * nn;
          double dot_b = 0.0;
          for (Size pr = 0; pr < nn; ++pr) dot_b += rb[pr] * bar_fock_b[pr];
          bc += dot_b;
        }
        bar_c_tile[a] = bc;
      }

      // Alpha outputs (compute bar_fit_a before bar_raw_a so bar_raw_a == raw_a is alias-safe!)
      if (oa > 0 && bar_fit_a) {
        double* out_fa = bar_fit_a + Size(q0) * oa * n;
        if (has_keff_a) {
          for (int a = 0; a < qt; ++a) {
            std::memcpy(occ_buf.data() + Size(a) * oa * n, raw_a_tile + Size(a) * nn,
                        Size(oa) * n * sizeof(double));
          }
          rm_gemm_host(false, true, qt * oa, n, n, 1.0, occ_buf.data(), n,
                       bar_keff_a.data(), n, accumulate_out ? 1.0 : 0.0, out_fa, n);
        } else if (!accumulate_out) {
          std::memset(out_fa, 0, Size(qt) * oa * n * sizeof(double));
        }
        if (has_g_a || has_g_b) {
          for (int a = 0; a < qt; ++a) {
            const double bc = bar_c_tile[a];
            double* dst = out_fa + Size(a) * oa * n;
            for (int i = 0; i < oa; ++i) dst[Size(i) * n + i] += bc;
          }
        }
        if (share_beta_raw && bar_fit_b && bar_fit_b != bar_fit_a) {
          std::memcpy(bar_fit_b + Size(q0) * oa * n, out_fa,
                      Size(qt) * oa * n * sizeof(double));
        }
      }
      if (bar_raw_a) {
        double* out_ra = bar_raw_a + Size(q0) * nn;
        for (int a = 0; a < qt; ++a) {
          double* dst = out_ra + Size(a) * nn;
          const double ca = charge_tile[a];
          if (has_g_a) {
            for (Size pr = 0; pr < nn; ++pr) {
              const double v = ca * bar_fock_a[pr];
              if (accumulate_out) dst[pr] += v;
              else dst[pr] = v;
            }
          } else if (!accumulate_out) {
            std::memset(dst, 0, nn * sizeof(double));
          }
        }
        if (oa > 0 && has_keff_a) {
          rm_gemm_host(false, false, qt * oa, n, n, 1.0, fit_a_tile, n,
                       bar_keff_a.data(), n, 0.0, occ_buf.data(), n);
          for (int a = 0; a < qt; ++a) {
            double* dst = out_ra + Size(a) * nn;
            const double* src = occ_buf.data() + Size(a) * oa * n;
            for (Size ir = 0; ir < Size(oa) * n; ++ir) dst[ir] += src[ir];
          }
        }
      }

      // Beta outputs (when !share_beta_raw)
      if (!share_beta_raw) {
        if (bar_raw_b) {
          double* out_rb = bar_raw_b + Size(q0) * nn;
          for (int a = 0; a < qt; ++a) {
            double* dst = out_rb + Size(a) * nn;
            const double ca = charge_tile[a];
            if (has_g_b) {
              for (Size pr = 0; pr < nn; ++pr) {
                const double v = ca * bar_fock_b[pr];
                if (accumulate_out) dst[pr] += v;
                else dst[pr] = v;
              }
            } else if (!accumulate_out) {
              std::memset(dst, 0, nn * sizeof(double));
            }
          }
          if (ob > 0 && has_keff_b) {
            rm_gemm_host(false, false, qt * ob, n, n, 1.0, fit_b_tile, n,
                         bar_keff_b.data(), n, 0.0, occ_buf.data(), n);
            for (int a = 0; a < qt; ++a) {
              double* dst = out_rb + Size(a) * nn;
              const double* src = occ_buf.data() + Size(a) * ob * n;
              for (Size ir = 0; ir < Size(ob) * n; ++ir) dst[ir] += src[ir];
            }
          }
        }
        if (ob > 0 && bar_fit_b) {
          double* out_fb = bar_fit_b + Size(q0) * ob * n;
          if (has_keff_b) {
            for (int a = 0; a < qt; ++a) {
              std::memcpy(occ_buf.data() + Size(a) * ob * n, raw_b_tile + Size(a) * nn,
                          Size(ob) * n * sizeof(double));
            }
            rm_gemm_host(false, true, qt * ob, n, n, 1.0, occ_buf.data(), n,
                         bar_keff_b.data(), n, accumulate_out ? 1.0 : 0.0, out_fb, n);
          } else if (!accumulate_out) {
            std::memset(out_fb, 0, Size(qt) * ob * n * sizeof(double));
          }
          if (has_g_a || has_g_b) {
            for (int a = 0; a < qt; ++a) {
              const double bc = bar_c_tile[a];
              double* dst = out_fb + Size(a) * ob * n;
              for (int i = 0; i < ob; ++i) dst[Size(i) * n + i] += bc;
            }
          }
        }
      }
    }

    if ((bar_hcore_a && !check_finite(bar_hcore_a, nn)) ||
        (bar_hcore_b && !check_finite(bar_hcore_b, nn)) ||
        (bar_raw_a && !check_finite(bar_raw_a, Size(q) * nn)) ||
        (!share_beta_raw && bar_raw_b && !check_finite(bar_raw_b, Size(q) * nn)) ||
        (oa > 0 && bar_fit_a && !check_finite(bar_fit_a, Size(q) * oa * n)) ||
        (ob > 0 && bar_fit_b && (!share_beta_raw || bar_fit_b != bar_fit_a) &&
         !check_finite(bar_fit_b, Size(q) * ob * n))) {
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

#ifdef __CUDACC__
extern "C" int gqc_f12_spin_fock_rev_cuda(
    int oa,
    int ob,
    int n,
    int q,
    int share_beta_raw,
    int accumulate_out,
    int q_tile_hint,
    const double* raw_a,
    const double* raw_b,
    const double* fit_a,
    const double* fit_b,
    const double* bar_fock_a,
    const double* bar_fock_b,
    const double* bar_exchange_a,
    const double* bar_exchange_b,
    double* bar_hcore_a,
    double* bar_hcore_b,
    double* bar_raw_a,
    double* bar_raw_b,
    double* bar_fit_a,
    double* bar_fit_b) {
  try {
    if (oa < 0 || ob < 0 || oa + ob < 1 || n <= 0 || oa > n || ob > n || q <= 0) {
      return kErrInvalidArg;
    }
    if (share_beta_raw && (oa != ob || accumulate_out != 0)) {
      return kErrInvalidArg;
    }
    if (!raw_a) return kErrInvalidArg;
    if (!share_beta_raw && !raw_b) return kErrInvalidArg;
    if (oa > 0 && !fit_a) return kErrInvalidArg;
    if (!share_beta_raw && ob > 0 && !fit_b) return kErrInvalidArg;

    const Size nn64 = Size(n) * n;
    if (nn64 > Size(INT_MAX)) return kErrInvalidArg;
    const int nn = static_cast<int>(nn64);

    const bool has_g_a = (bar_fock_a != nullptr);
    const bool has_g_b = share_beta_raw ? has_g_a : (bar_fock_b != nullptr);
    const bool has_k_a = (bar_exchange_a != nullptr);
    const bool has_k_b = share_beta_raw ? has_k_a : (bar_exchange_b != nullptr);
    const bool has_keff_a = has_g_a || has_k_a;
    const bool has_keff_b = has_g_b || has_k_b;
    const bool need_bar_c = (has_g_a || has_g_b) &&
                            ((oa > 0 && bar_fit_a != nullptr) ||
                             (ob > 0 && bar_fit_b != nullptr));

    const int max_o = std::max(oa, ob);
    const std::size_t fixed_mats =
        (has_g_a ? 1 : 0) + (!share_beta_raw && has_g_b ? 1 : 0) +
        (has_keff_a ? 1 : 0) + (!share_beta_raw && has_keff_b ? 1 : 0);
    const std::size_t fixed_bytes = fixed_mats * nn64 * sizeof(double);
    const std::size_t per_q_bytes = (nn64 + Size(3) * max_o * n + 2) * sizeof(double);
    int qt_cap = compute_tile_width(q, fixed_bytes, per_q_bytes, q_tile_hint);
    if (qt_cap < 1) return kErrOom;
    if (max_o > 0) {
      qt_cap = std::min(qt_cap, INT_MAX / max_o);
      if (qt_cap < 1) return kErrInvalidArg;
    }

    // Build effective exchange cotangents on host (small n x n matrices)
    std::vector<double> h_keff_a(has_keff_a ? nn64 : 0, 0.0);
    if (has_keff_a) {
      for (Size pr = 0; pr < nn64; ++pr) {
        h_keff_a[pr] = (has_k_a ? bar_exchange_a[pr] : 0.0) -
                       (has_g_a ? bar_fock_a[pr] : 0.0);
      }
    }
    std::vector<double> h_keff_b((!share_beta_raw && has_keff_b) ? nn64 : 0, 0.0);
    if (!share_beta_raw && has_keff_b) {
      for (Size pr = 0; pr < nn64; ++pr) {
        h_keff_b[pr] = (has_k_b ? bar_exchange_b[pr] : 0.0) -
                       (has_g_b ? bar_fock_b[pr] : 0.0);
      }
    }

    // hcore cotangents on host (small n x n)
    if (bar_hcore_a) {
      for (Size pr = 0; pr < nn64; ++pr) {
        const double v = has_g_a ? bar_fock_a[pr] : 0.0;
        if (accumulate_out) bar_hcore_a[pr] += v;
        else bar_hcore_a[pr] = v;
      }
    }
    if (bar_hcore_b) {
      const double* src_gb = share_beta_raw ? bar_fock_a : bar_fock_b;
      for (Size pr = 0; pr < nn64; ++pr) {
        const double v = src_gb ? src_gb[pr] : 0.0;
        if (accumulate_out) bar_hcore_b[pr] += v;
        else bar_hcore_b[pr] = v;
      }
    }

    CublasGuard cg;
    if (cg.init() != CUBLAS_STATUS_SUCCESS) return kErrCublas;

    DevBuf d_raw, d_fit, d_occ_buf, d_bar_fit, d_charge, d_bar_c;
    DevBuf d_bar_ga, d_bar_gb, d_keff_a, d_keff_b;
    if (d_raw.alloc(Size(qt_cap) * nn64) != cudaSuccess ||
        d_fit.alloc(Size(qt_cap) * max_o * n) != cudaSuccess ||
        d_occ_buf.alloc(Size(qt_cap) * max_o * n) != cudaSuccess ||
        d_bar_fit.alloc(Size(qt_cap) * max_o * n) != cudaSuccess ||
        d_charge.alloc(static_cast<std::size_t>(qt_cap)) != cudaSuccess ||
        d_bar_c.alloc(static_cast<std::size_t>(qt_cap)) != cudaSuccess ||
        (has_g_a && d_bar_ga.alloc(nn64) != cudaSuccess) ||
        (!share_beta_raw && has_g_b && d_bar_gb.alloc(nn64) != cudaSuccess) ||
        (has_keff_a && d_keff_a.alloc(nn64) != cudaSuccess) ||
        (!share_beta_raw && has_keff_b && d_keff_b.alloc(nn64) != cudaSuccess)) {
      return kErrOom;
    }

    if (has_g_a && cudaMemcpy(d_bar_ga.ptr, bar_fock_a, nn64 * sizeof(double),
                              cudaMemcpyHostToDevice) != cudaSuccess) {
      return kErrCuda;
    }
    if (!share_beta_raw && has_g_b &&
        cudaMemcpy(d_bar_gb.ptr, bar_fock_b, nn64 * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return kErrCuda;
    }
    if (has_keff_a && cudaMemcpy(d_keff_a.ptr, h_keff_a.data(), nn64 * sizeof(double),
                                 cudaMemcpyHostToDevice) != cudaSuccess) {
      return kErrCuda;
    }
    if (!share_beta_raw && has_keff_b &&
        cudaMemcpy(d_keff_b.ptr, h_keff_b.data(), nn64 * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return kErrCuda;
    }

    std::vector<double> h_charge(static_cast<std::size_t>(qt_cap), 0.0);
    std::vector<double> h_bar_c(static_cast<std::size_t>(qt_cap), 0.0);
    const double one = 1.0;
    const double two = 2.0;
    const double zero = 0.0;
    constexpr int kThreads = 256;

    for (int q0 = 0; q0 < q; q0 += qt_cap) {
      const int qt = std::min(qt_cap, q - q0);
      const Size tile_raw_elems = Size(qt) * nn64;

      const double* raw_a_tile = raw_a + Size(q0) * nn64;
      const double* fit_a_tile = oa > 0 ? (fit_a + Size(q0) * oa * n) : nullptr;
      const double* raw_b_tile = !share_beta_raw ? (raw_b + Size(q0) * nn64) : nullptr;
      const double* fit_b_tile = (!share_beta_raw && ob > 0) ? (fit_b + Size(q0) * ob * n) : nullptr;

      compute_tile_charge_host(oa, ob, n, qt, share_beta_raw, fit_a_tile, fit_b_tile,
                               h_charge.data());
      if (cudaMemcpy(d_charge.ptr, h_charge.data(), Size(qt) * sizeof(double),
                     cudaMemcpyHostToDevice) != cudaSuccess) {
        return kErrCuda;
      }

      // ----- Alpha pass -----
      const bool need_raw_a_gpu = (need_bar_c && has_g_a) ||
                                  (oa > 0 && bar_fit_a != nullptr && has_keff_a);
      if (need_raw_a_gpu) {
        if (cudaMemcpy(d_raw.ptr, raw_a_tile, tile_raw_elems * sizeof(double),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
          return kErrCuda;
        }
      }

      if (need_bar_c) {
        if (has_g_a) {
          const double alpha_c = share_beta_raw ? two : one;
          if (cublasDgemv(cg.handle, CUBLAS_OP_T, nn, qt, &alpha_c, d_raw.ptr, nn,
                          d_bar_ga.ptr, 1, &zero, d_bar_c.ptr, 1) !=
              CUBLAS_STATUS_SUCCESS) {
            return kErrCublas;
          }
        } else {
          if (cudaMemset(d_bar_c.ptr, 0, Size(qt) * sizeof(double)) != cudaSuccess) {
            return kErrCuda;
          }
        }
      }

      if (oa > 0 && bar_fit_a) {
        const Size fit_elems = Size(qt) * oa * n;
        double* out_fa = bar_fit_a + Size(q0) * oa * n;
        if (accumulate_out) {
          if (cudaMemcpy(d_bar_fit.ptr, out_fa, fit_elems * sizeof(double),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return kErrCuda;
          }
        }
        if (has_keff_a) {
          k_slice_occ_raw<<<launch_blocks(fit_elems, kThreads), kThreads>>>(
              qt, oa, n, d_raw.ptr, d_occ_buf.ptr);
          if (cudaGetLastError() != cudaSuccess) return kErrCuda;
          if (rm_gemm_cuda(cg.handle, false, true, qt * oa, n, n, 1.0,
                           d_occ_buf.ptr, n, d_keff_a.ptr, n,
                           accumulate_out ? 1.0 : 0.0, d_bar_fit.ptr, n) !=
              CUBLAS_STATUS_SUCCESS) {
            return kErrCublas;
          }
        } else if (!accumulate_out) {
          if (cudaMemset(d_bar_fit.ptr, 0, fit_elems * sizeof(double)) != cudaSuccess) {
            return kErrCuda;
          }
        }
        if (share_beta_raw && need_bar_c) {
          const Size diag_total = Size(qt) * oa;
          k_add_diag_bar_c<<<launch_blocks(diag_total, kThreads), kThreads>>>(
              qt, oa, n, d_bar_c.ptr, d_bar_fit.ptr);
          if (cudaGetLastError() != cudaSuccess) return kErrCuda;
        }
        if (cudaMemcpy(out_fa, d_bar_fit.ptr, fit_elems * sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          return kErrCuda;
        }
        if (share_beta_raw && bar_fit_b && bar_fit_b != bar_fit_a) {
          std::memcpy(bar_fit_b + Size(q0) * oa * n, out_fa,
                      fit_elems * sizeof(double));
        }
      }

      if (bar_raw_a) {
        double* out_ra = bar_raw_a + Size(q0) * nn64;
        if (accumulate_out) {
          if (cudaMemcpy(d_raw.ptr, out_ra, tile_raw_elems * sizeof(double),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return kErrCuda;
          }
        }
        k_outer_charge_bar_g<<<launch_blocks(tile_raw_elems, kThreads), kThreads>>>(
            qt, nn64, d_charge.ptr, has_g_a ? d_bar_ga.ptr : nullptr, d_raw.ptr,
            accumulate_out);
        if (cudaGetLastError() != cudaSuccess) return kErrCuda;

        if (oa > 0 && has_keff_a) {
          const Size fit_elems = Size(qt) * oa * n;
          if (cudaMemcpy(d_fit.ptr, fit_a_tile, fit_elems * sizeof(double),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return kErrCuda;
          }
          if (rm_gemm_cuda(cg.handle, false, false, qt * oa, n, n, 1.0,
                           d_fit.ptr, n, d_keff_a.ptr, n, 0.0, d_occ_buf.ptr, n) !=
              CUBLAS_STATUS_SUCCESS) {
            return kErrCublas;
          }
          k_add_occ_raw<<<launch_blocks(fit_elems, kThreads), kThreads>>>(
              qt, oa, n, d_occ_buf.ptr, d_raw.ptr);
          if (cudaGetLastError() != cudaSuccess) return kErrCuda;
        }
        if (cudaMemcpy(out_ra, d_raw.ptr, tile_raw_elems * sizeof(double),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          return kErrCuda;
        }
      }

      // ----- Beta pass (skipped when share_beta_raw != 0) -----
      if (!share_beta_raw) {
        const bool need_raw_b_gpu = (need_bar_c && has_g_b) ||
                                    (ob > 0 && bar_fit_b != nullptr && has_keff_b);
        if (need_raw_b_gpu) {
          if (cudaMemcpy(d_raw.ptr, raw_b_tile, tile_raw_elems * sizeof(double),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return kErrCuda;
          }
        }

        if (need_bar_c && has_g_b) {
          if (cublasDgemv(cg.handle, CUBLAS_OP_T, nn, qt, &one, d_raw.ptr, nn,
                          d_bar_gb.ptr, 1, &one, d_bar_c.ptr, 1) !=
              CUBLAS_STATUS_SUCCESS) {
            return kErrCublas;
          }
        }

        if (ob > 0 && bar_fit_b) {
          const Size fit_elems = Size(qt) * ob * n;
          double* out_fb = bar_fit_b + Size(q0) * ob * n;
          if (accumulate_out) {
            if (cudaMemcpy(d_bar_fit.ptr, out_fb, fit_elems * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
              return kErrCuda;
            }
          }
          if (has_keff_b) {
            k_slice_occ_raw<<<launch_blocks(fit_elems, kThreads), kThreads>>>(
                qt, ob, n, d_raw.ptr, d_occ_buf.ptr);
            if (cudaGetLastError() != cudaSuccess) return kErrCuda;
            if (rm_gemm_cuda(cg.handle, false, true, qt * ob, n, n, 1.0,
                             d_occ_buf.ptr, n, d_keff_b.ptr, n,
                             accumulate_out ? 1.0 : 0.0, d_bar_fit.ptr, n) !=
                CUBLAS_STATUS_SUCCESS) {
              return kErrCublas;
            }
          } else if (!accumulate_out) {
            if (cudaMemset(d_bar_fit.ptr, 0, fit_elems * sizeof(double)) != cudaSuccess) {
              return kErrCuda;
            }
          }
          if (need_bar_c) {
            const Size diag_total = Size(qt) * ob;
            k_add_diag_bar_c<<<launch_blocks(diag_total, kThreads), kThreads>>>(
                qt, ob, n, d_bar_c.ptr, d_bar_fit.ptr);
            if (cudaGetLastError() != cudaSuccess) return kErrCuda;
          }
          if (cudaMemcpy(out_fb, d_bar_fit.ptr, fit_elems * sizeof(double),
                         cudaMemcpyDeviceToHost) != cudaSuccess) {
            return kErrCuda;
          }
        }

        if (bar_raw_b) {
          double* out_rb = bar_raw_b + Size(q0) * nn64;
          if (accumulate_out) {
            if (cudaMemcpy(d_raw.ptr, out_rb, tile_raw_elems * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
              return kErrCuda;
            }
          }
          k_outer_charge_bar_g<<<launch_blocks(tile_raw_elems, kThreads), kThreads>>>(
              qt, nn64, d_charge.ptr, has_g_b ? d_bar_gb.ptr : nullptr, d_raw.ptr,
              accumulate_out);
          if (cudaGetLastError() != cudaSuccess) return kErrCuda;

          if (ob > 0 && has_keff_b) {
            const Size fit_elems = Size(qt) * ob * n;
            if (cudaMemcpy(d_fit.ptr, fit_b_tile, fit_elems * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
              return kErrCuda;
            }
            if (rm_gemm_cuda(cg.handle, false, false, qt * ob, n, n, 1.0,
                             d_fit.ptr, n, d_keff_b.ptr, n, 0.0, d_occ_buf.ptr, n) !=
                CUBLAS_STATUS_SUCCESS) {
              return kErrCublas;
            }
            k_add_occ_raw<<<launch_blocks(fit_elems, kThreads), kThreads>>>(
                qt, ob, n, d_occ_buf.ptr, d_raw.ptr);
            if (cudaGetLastError() != cudaSuccess) return kErrCuda;
          }
          if (cudaMemcpy(out_rb, d_raw.ptr, tile_raw_elems * sizeof(double),
                         cudaMemcpyDeviceToHost) != cudaSuccess) {
            return kErrCuda;
          }
        }

        // Finish alpha fit diagonal with complete bar_c = bar_c_alpha + bar_c_beta
        if (need_bar_c && oa > 0 && bar_fit_a) {
          if (cudaMemcpy(h_bar_c.data(), d_bar_c.ptr, Size(qt) * sizeof(double),
                         cudaMemcpyDeviceToHost) != cudaSuccess) {
            return kErrCuda;
          }
          double* out_fa = bar_fit_a + Size(q0) * oa * n;
          for (int a = 0; a < qt; ++a) {
            const double bc = h_bar_c[a];
            double* dst = out_fa + Size(a) * oa * n;
            for (int i = 0; i < oa; ++i) dst[Size(i) * n + i] += bc;
          }
        }
      }
    }

    if ((bar_hcore_a && !check_finite(bar_hcore_a, nn64)) ||
        (bar_hcore_b && !check_finite(bar_hcore_b, nn64)) ||
        (bar_raw_a && !check_finite(bar_raw_a, Size(q) * nn64)) ||
        (!share_beta_raw && bar_raw_b && !check_finite(bar_raw_b, Size(q) * nn64)) ||
        (oa > 0 && bar_fit_a && !check_finite(bar_fit_a, Size(q) * oa * n)) ||
        (ob > 0 && bar_fit_b && (!share_beta_raw || bar_fit_b != bar_fit_a) &&
         !check_finite(bar_fit_b, Size(q) * ob * n))) {
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
#endif
