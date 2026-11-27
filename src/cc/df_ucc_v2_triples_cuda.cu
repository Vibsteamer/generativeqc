#include "cc/df_ucc_v2_cuda_internal.cuh"
#include "cc/df_ucc_v2_triples_capacity.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace generativeqc::cc::v2_detail::cuda_detail {
namespace {

template <class Real>
void gemm_counted(cublasHandle_t blas, cudaStream_t st, char ta, char tb, std::size_t m,
                  std::size_t n, std::size_t k, const Real* a, const Real* b, Real* c,
                  std::size_t& counter, Real alpha = Real(1), Real beta = Real(0)) {
  if (!m || !n) return;
  if (!k) {
    if (beta == Real(0))
      cuda_check(cudaMemsetAsync(c, 0, checked_mul(checked_mul(m, n), sizeof(Real)), st),
                 "triples gemm zero");
    return;
  }
  rm_gemm<Real>(blas, st, ta, tb, m, n, k, a, b, c, alpha, beta);
  ++counter;
}

template <class Real>
__global__ void extract_bov_slice_kernel(std::size_t q, std::size_t o_p, std::size_t v_c,
                                         std::size_t p, const Real* bov, Real* bov_p) {
  const std::size_t total = q * v_c;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t Q = idx / v_c, c = idx % v_c;
    bov_p[idx] = bov[(Q * o_p + p) * v_c + c];
  }
}

__global__ void scatter_add_bov_slice_kernel(std::size_t q, std::size_t o_p, std::size_t v_c,
                                             std::size_t p, const double* bar_bov_p,
                                             double* bar_bov) {
  const std::size_t total = q * v_c;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t Q = idx / v_c, c = idx % v_c;
    bar_bov[(Q * o_p + p) * v_c + c] += bar_bov_p[idx];
  }
}

template <class Real>
void build_ovvv_slab_device(int dev, cublasHandle_t blas, cudaStream_t st, std::size_t q,
                            std::size_t o_p, std::size_t v_c, std::size_t v_be, std::size_t p,
                            const Real* bov, const Real* bvv, Real* slab, Real* bov_p_scratch,
                            std::size_t& gemm_count) {
  (void)dev;
  extract_bov_slice_kernel<Real><<<grid_1d(q * v_c), 256, 0, st>>>(q, o_p, v_c, p, bov,
                                                                   bov_p_scratch);
  cuda_check(cudaGetLastError(), "extract_bov_slice");
  CublasMathModeScope math_guard(blas, CUBLAS_PEDANTIC_MATH, std::is_same_v<Real, float>);
  gemm_counted<Real>(blas, st, 'T', 'N', v_c, v_be * v_be, q, bov_p_scratch, bvv, slab, gemm_count,
                     Real(1), Real(0));
}

void pullback_ovvv_slab_device(cublasHandle_t blas, cudaStream_t st, std::size_t q, std::size_t o_p,
                               std::size_t v_c, std::size_t v_be, std::size_t p, const double* bov,
                               const double* bvv, const double* bar_slab, double* bar_bov,
                               double* bar_bvv, double* bov_p_scratch, double* bar_bov_p_scratch,
                               std::size_t& fp64_gemms) {
  extract_bov_slice_kernel<double><<<grid_1d(q * v_c), 256, 0, st>>>(q, o_p, v_c, p, bov,
                                                                     bov_p_scratch);
  cuda_check(cudaGetLastError(), "pullback extract_bov_slice");
  gemm_counted<double>(blas, st, 'N', 'T', q, v_c, v_be * v_be, bvv, bar_slab, bar_bov_p_scratch,
                       fp64_gemms, 1.0, 0.0);
  scatter_add_bov_slice_kernel<<<grid_1d(q * v_c), 256, 0, st>>>(q, o_p, v_c, p, bar_bov_p_scratch,
                                                                 bar_bov);
  cuda_check(cudaGetLastError(), "scatter_add_bov_slice");
  gemm_counted<double>(blas, st, 'N', 'N', q, v_be * v_be, v_c, bov_p_scratch, bar_slab, bar_bvv,
                       fp64_gemms, 1.0, 1.0);
}

// --- Same-Spin (aaa / bbb) Kernels ---
template <class RealW>
__global__ void assemble_w_same_spin_kernel(std::size_t v, const RealW* s_acb, const RealW* s_occ,
                                            double* W) {
  const std::size_t v2 = v * v, v3 = v * v2;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < v3;
       idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t c = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t b = r1 % v;
    const std::size_t a = r1 / v;
    const double sv =
        static_cast<double>(s_acb[(a * v + c) * v + b]) -
        static_cast<double>(s_acb[(a * v + b) * v + c]) -
        static_cast<double>(s_acb[(b * v + c) * v + a]) +
        static_cast<double>(s_acb[(b * v + a) * v + c]) +
        static_cast<double>(s_acb[(c * v + b) * v + a]) -
        static_cast<double>(s_acb[(c * v + a) * v + b]);
    const double so = static_cast<double>(s_occ[(a * v + b) * v + c]) -
                      static_cast<double>(s_occ[(b * v + a) * v + c]) -
                      static_cast<double>(s_occ[(c * v + b) * v + a]);
    W[idx] = sv - so;
  }
}

__device__ inline double eval_v_same_spin_elem(std::size_t v, std::size_t a, std::size_t b,
                                               std::size_t c, const double* t1_0,
                                               const double* fov_0, const double* g_12,
                                               const double* t2_12) {
  const auto term = [&](std::size_t x, std::size_t y, std::size_t z) -> double {
    return t1_0[x] * g_12[y * v + z] + fov_0[x] * t2_12[y * v + z];
  };
  return term(a, b, c) - term(b, a, c) - term(c, b, a);
}

__global__ void eval_same_spin_triple_kernel(
    std::size_t o, std::size_t v, std::size_t i, std::size_t j, std::size_t k, double eijk_o,
    double wt, double den_thresh, bool compute_forces, const double* W, const double* t1,
    const double* fov, const double* goovv, const double* t2, const double* eps_v, double* X,
    double* Y, double* bar_W, double* bar_V, double* d_total_e, double* d_occ_e, double* d_bar_foo,
    int* d_err) {
  __shared__ double smem_e[256];
  __shared__ double smem_f[256];
  const std::size_t v2 = v * v, v3 = v * v2;
  const double eff = (1.0 / 6.0) * wt;
  const double* t1i = t1 + i * v;
  const double* t1j = t1 + j * v;
  const double* t1k = t1 + k * v;
  const double* fovi = fov + i * v;
  const double* fovj = fov + j * v;
  const double* fovk = fov + k * v;
  const double* g_jk = goovv + (j * o + k) * v2;
  const double* g_ki = goovv + (k * o + i) * v2;
  const double* g_ij = goovv + (i * o + j) * v2;
  const double* t_jk = t2 + (j * o + k) * v2;
  const double* t_ki = t2 + (k * o + i) * v2;
  const double* t_ij = t2 + (i * o + j) * v2;

  double local_e = 0.0;
  double local_f = 0.0;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < v3;
       idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t c = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t b = r1 % v;
    const std::size_t a = r1 / v;
    const double den = eijk_o - eps_v[a] - eps_v[b] - eps_v[c];
    if (fabs(den) < den_thresh) {
      atomicExch(d_err, 1);
      continue;
    }
    const double w = W[idx];
    const double vv = eval_v_same_spin_elem(v, a, b, c, t1i, fovi, g_jk, t_jk) +
                      eval_v_same_spin_elem(v, a, b, c, t1j, fovj, g_ki, t_ki) +
                      eval_v_same_spin_elem(v, a, b, c, t1k, fovk, g_ij, t_ij);
    const double x = w / den;
    const double y = (w + vv) / den;
    local_e += w * y;
    if (compute_forces) {
      X[idx] = x;
      Y[idx] = y;
      bar_W[idx] = eff * (x + y);
      bar_V[idx] = eff * x;
      local_f -= eff * x * y;
    }
  }
  smem_e[threadIdx.x] = local_e;
  smem_f[threadIdx.x] = local_f;
  __syncthreads();
  for (unsigned s = 128U; s > 0U; s >>= 1U) {
    if (threadIdx.x < s) {
      smem_e[threadIdx.x] += smem_e[threadIdx.x + s];
      smem_f[threadIdx.x] += smem_f[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double e_unw = smem_e[0] / 6.0;
    atomicAdd(d_total_e, wt * e_unw);
    const double third = e_unw / 3.0;
    atomicAdd(d_occ_e + i, third);
    atomicAdd(d_occ_e + j, third);
    atomicAdd(d_occ_e + k, third);
    if (compute_forces) {
      const double df = smem_f[0];
      atomicAdd(d_bar_foo + i * o + i, df);
      atomicAdd(d_bar_foo + j * o + j, df);
      atomicAdd(d_bar_foo + k * o + k, df);
    }
  }
}

__global__ void eval_same_spin_xy_only_kernel(
    std::size_t o, std::size_t v, std::size_t p, std::size_t j, std::size_t k, double epjk,
    const double* W, const double* t1, const double* fov, const double* goovv, const double* t2,
    const double* eps_v, double* X, double* Y) {
  const std::size_t v2 = v * v, v3 = v * v2;
  const double* t1p = t1 + p * v;
  const double* t1j = t1 + j * v;
  const double* t1k = t1 + k * v;
  const double* fovp = fov + p * v;
  const double* fovj = fov + j * v;
  const double* fovk = fov + k * v;
  const double* g_jk = goovv + (j * o + k) * v2;
  const double* g_kp = goovv + (k * o + p) * v2;
  const double* g_pj = goovv + (p * o + j) * v2;
  const double* t_jk = t2 + (j * o + k) * v2;
  const double* t_kp = t2 + (k * o + p) * v2;
  const double* t_pj = t2 + (p * o + j) * v2;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < v3;
       idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t c = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t b = r1 % v;
    const std::size_t a = r1 / v;
    const double den = epjk - eps_v[a] - eps_v[b] - eps_v[c];
    const double w = W[idx];
    const double vv = eval_v_same_spin_elem(v, a, b, c, t1p, fovp, g_jk, t_jk) +
                      eval_v_same_spin_elem(v, a, b, c, t1j, fovj, g_kp, t_kp) +
                      eval_v_same_spin_elem(v, a, b, c, t1k, fovk, g_pj, t_pj);
    X[idx] = w / den;
    Y[idx] = (w + vv) / den;
  }
}

__global__ void add_sym_xy_fvv_kernel(std::size_t v, double coeff, const double* xy,
                                      double* bar_fvv) {
  const std::size_t total = v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t a = idx / v, d = idx % v;
    bar_fvv[idx] += coeff * (xy[a * v + d] + xy[d * v + a]);
  }
}

__global__ void pullback_v_same_spin_a_kernel(std::size_t v, const double* bar_V,
                                              const double* g_rs, const double* t2_rs,
                                              double* bar_t1_p, double* bar_fov_p) {
  const std::size_t v2 = v * v;
  for (std::size_t a = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; a < v;
       a += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    double st1 = 0.0, sfov = 0.0;
    const double* bv = bar_V + a * v2;
    for (std::size_t bc = 0; bc < v2; ++bc) {
      const double bvv = bv[bc];
      st1 += bvv * g_rs[bc];
      sfov += bvv * t2_rs[bc];
    }
    bar_t1_p[a] += 3.0 * st1;
    bar_fov_p[a] += 3.0 * sfov;
  }
}

__global__ void pullback_v_same_spin_bc_kernel(std::size_t v, const double* bar_V,
                                               const double* t1_p, const double* fov_p,
                                               double* bar_g_rs, double* bar_t2_rs) {
  const std::size_t v2 = v * v;
  for (std::size_t bc = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; bc < v2;
       bc += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    double sg = 0.0, st = 0.0;
    for (std::size_t a = 0; a < v; ++a) {
      const double bvv = bar_V[a * v2 + bc];
      sg += t1_p[a] * bvv;
      st += fov_p[a] * bvv;
    }
    bar_g_rs[bc] += 3.0 * sg;
    bar_t2_rs[bc] += 3.0 * st;
  }
}

// --- Mixed-Spin (aab / abb) Kernels ---
template <class RealW>
__global__ void add_acb_to_abc_kernel(std::size_t v1, std::size_t v2, const RealW* tmp_acb,
                                      RealW* R) {
  const std::size_t total = v1 * v1 * v2;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t c = idx % v2;
    const std::size_t r1 = idx / v2;
    const std::size_t b = r1 % v1;
    const std::size_t a = r1 / v1;
    R[idx] = tmp_acb[(a * v2 + c) * v1 + b];
  }
}

template <class RealW>
__global__ void antisym_ab_mixed_kernel(std::size_t v1, std::size_t v2, const RealW* R, RealW* W) {
  const std::size_t total = v1 * v1 * v2;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t c = idx % v2;
    const std::size_t r1 = idx / v2;
    const std::size_t b = r1 % v1;
    const std::size_t a = r1 / v1;
    W[idx] = R[idx] - R[(b * v1 + a) * v2 + c];
  }
}

template <class RealW>
__global__ void eval_mixed_spin_triple_kernel(
    std::size_t o1, std::size_t o2, std::size_t v1, std::size_t v2, std::size_t i, std::size_t j,
    std::size_t k, double eijk_o, double wt, double den_thresh, bool compute_forces, const RealW* W,
    const double* t1_1, const double* t1_2, const double* fov1, const double* fov2,
    const double* goovv11, const double* goovv12, const double* t2_11, const double* t2_12,
    const double* eps_v1, const double* eps_v2, double* X, double* Y, double* bar_W, double* bar_V,
    double* d_total_e, double* d_occ1, double* d_occ2, double* d_bar_foo1, double* d_bar_foo2,
    int* d_err) {
  __shared__ double smem_e[256];
  __shared__ double smem_f[256];
  const std::size_t v11 = v1 * v1, v12 = v1 * v2, v_tot = v11 * v2;
  const double eff = 0.5 * wt;
  const double* t1i = t1_1 + i * v1;
  const double* t1j = t1_1 + j * v1;
  const double* t1k = t1_2 + k * v2;
  const double* fovi = fov1 + i * v1;
  const double* fovj = fov1 + j * v1;
  const double* fovk = fov2 + k * v2;
  const double* g_jk = goovv12 + (j * o2 + k) * v12;
  const double* g_ik = goovv12 + (i * o2 + k) * v12;
  const double* g_ij = goovv11 + (i * o1 + j) * v11;
  const double* t_jk = t2_12 + (j * o2 + k) * v12;
  const double* t_ik = t2_12 + (i * o2 + k) * v12;
  const double* t_ij = t2_11 + (i * o1 + j) * v11;

  double local_e = 0.0;
  double local_f = 0.0;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < v_tot; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t c = idx % v2;
    const std::size_t r1 = idx / v2;
    const std::size_t b = r1 % v1;
    const std::size_t a = r1 / v1;
    const double den = eijk_o - eps_v1[a] - eps_v1[b] - eps_v2[c];
    if (fabs(den) < den_thresh) {
      atomicExch(d_err, 1);
      continue;
    }
    const double w = static_cast<double>(W[idx]);
    const std::size_t bc = b * v2 + c, ac = a * v2 + c, ab = a * v1 + b;
    const double vv =
        (t1i[a] * g_jk[bc] + fovi[a] * t_jk[bc]) - (t1i[b] * g_jk[ac] + fovi[b] * t_jk[ac]) -
        (t1j[a] * g_ik[bc] + fovj[a] * t_ik[bc]) + (t1j[b] * g_ik[ac] + fovj[b] * t_ik[ac]) +
        (t1k[c] * g_ij[ab] + fovk[c] * t_ij[ab]);
    const double x = w / den;
    const double y = (w + vv) / den;
    local_e += w * y;
    if (compute_forces) {
      X[idx] = x;
      Y[idx] = y;
      bar_W[idx] = eff * (x + y);
      bar_V[idx] = eff * x;
      local_f -= eff * x * y;
    }
  }
  smem_e[threadIdx.x] = local_e;
  smem_f[threadIdx.x] = local_f;
  __syncthreads();
  for (unsigned s = 128U; s > 0U; s >>= 1U) {
    if (threadIdx.x < s) {
      smem_e[threadIdx.x] += smem_e[threadIdx.x + s];
      smem_f[threadIdx.x] += smem_f[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const double e_unw = 0.5 * smem_e[0];
    atomicAdd(d_total_e, wt * e_unw);
    const double third = e_unw / 3.0;
    atomicAdd(d_occ1 + i, third);
    atomicAdd(d_occ1 + j, third);
    atomicAdd(d_occ2 + k, third);
    if (compute_forces) {
      const double df = smem_f[0];
      atomicAdd(d_bar_foo1 + i * o1 + i, df);
      atomicAdd(d_bar_foo1 + j * o1 + j, df);
      atomicAdd(d_bar_foo2 + k * o2 + k, df);
    }
  }
}

template <class RealW>
__global__ void eval_mixed_spin_xy_only_kernel(
    std::size_t o1, std::size_t o2, std::size_t v1, std::size_t v2, std::size_t i, std::size_t j,
    std::size_t k, double eijk_o, const RealW* W, const double* t1_1, const double* t1_2,
    const double* fov1, const double* fov2, const double* goovv11, const double* goovv12,
    const double* t2_11, const double* t2_12, const double* eps_v1, const double* eps_v2, double* X,
    double* Y) {
  const std::size_t v11 = v1 * v1, v12 = v1 * v2, v_tot = v11 * v2;
  const double* t1i = t1_1 + i * v1;
  const double* t1j = t1_1 + j * v1;
  const double* t1k = t1_2 + k * v2;
  const double* fovi = fov1 + i * v1;
  const double* fovj = fov1 + j * v1;
  const double* fovk = fov2 + k * v2;
  const double* g_jk = goovv12 + (j * o2 + k) * v12;
  const double* g_ik = goovv12 + (i * o2 + k) * v12;
  const double* g_ij = goovv11 + (i * o1 + j) * v11;
  const double* t_jk = t2_12 + (j * o2 + k) * v12;
  const double* t_ik = t2_12 + (i * o2 + k) * v12;
  const double* t_ij = t2_11 + (i * o1 + j) * v11;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < v_tot; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t c = idx % v2;
    const std::size_t r1 = idx / v2;
    const std::size_t b = r1 % v1;
    const std::size_t a = r1 / v1;
    const double den = eijk_o - eps_v1[a] - eps_v1[b] - eps_v2[c];
    const double w = static_cast<double>(W[idx]);
    const std::size_t bc = b * v2 + c, ac = a * v2 + c, ab = a * v1 + b;
    const double vv =
        (t1i[a] * g_jk[bc] + fovi[a] * t_jk[bc]) - (t1i[b] * g_jk[ac] + fovi[b] * t_jk[ac]) -
        (t1j[a] * g_ik[bc] + fovj[a] * t_ik[bc]) + (t1j[b] * g_ik[ac] + fovj[b] * t_ik[ac]) +
        (t1k[c] * g_ij[ab] + fovk[c] * t_ij[ab]);
    X[idx] = w / den;
    Y[idx] = (w + vv) / den;
  }
}

__global__ void pullback_v_mixed_a_kernel(std::size_t v1, std::size_t v2, const double* bar_V,
                                          const double* g_jk, const double* g_ik,
                                          const double* t_jk, const double* t_ik, double* bar_t1_i,
                                          double* bar_fov_i, double* bar_t1_j, double* bar_fov_j) {
  const std::size_t v12 = v1 * v2;
  for (std::size_t a = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; a < v1;
       a += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    double st1i = 0.0, sfovi = 0.0, st1j = 0.0, sfovj = 0.0;
    const double* bv = bar_V + a * v12;
    for (std::size_t bc = 0; bc < v12; ++bc) {
      const double val = bv[bc];
      st1i += val * g_jk[bc];
      sfovi += val * t_jk[bc];
      st1j -= val * g_ik[bc];
      sfovj -= val * t_ik[bc];
    }
    bar_t1_i[a] += 2.0 * st1i;
    bar_fov_i[a] += 2.0 * sfovi;
    bar_t1_j[a] += 2.0 * st1j;
    bar_fov_j[a] += 2.0 * sfovj;
  }
}

__global__ void pullback_v_mixed_bc_kernel(std::size_t v1, std::size_t v2, const double* bar_V,
                                           const double* t1_i, const double* fov_i,
                                           const double* t1_j, const double* fov_j, double* bg_jk,
                                           double* bt_jk, double* bg_ik, double* bt_ik) {
  const std::size_t v12 = v1 * v2;
  for (std::size_t bc = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; bc < v12;
       bc += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    double sg_jk = 0.0, st_jk = 0.0, sg_ik = 0.0, st_ik = 0.0;
    for (std::size_t a = 0; a < v1; ++a) {
      const double val = bar_V[a * v12 + bc];
      sg_jk += t1_i[a] * val;
      st_jk += fov_i[a] * val;
      sg_ik -= t1_j[a] * val;
      st_ik -= fov_j[a] * val;
    }
    bg_jk[bc] += 2.0 * sg_jk;
    bt_jk[bc] += 2.0 * st_jk;
    bg_ik[bc] += 2.0 * sg_ik;
    bt_ik[bc] += 2.0 * st_ik;
  }
}

__global__ void pullback_v_mixed_c_kernel(std::size_t v1, std::size_t v2, const double* bar_V,
                                          const double* g_ij, const double* t_ij, double* bar_t1_k,
                                          double* bar_fov_k) {
  const std::size_t v11 = v1 * v1;
  for (std::size_t c = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; c < v2;
       c += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    double st = 0.0, sf = 0.0;
    for (std::size_t ab = 0; ab < v11; ++ab) {
      const double val = bar_V[ab * v2 + c];
      st += val * g_ij[ab];
      sf += val * t_ij[ab];
    }
    bar_t1_k[c] += st;
    bar_fov_k[c] += sf;
  }
}

__global__ void pullback_v_mixed_ab_kernel(std::size_t v1, std::size_t v2, const double* bar_V,
                                           const double* t1_k, const double* fov_k, double* bg_ij,
                                           double* bt_ij) {
  const std::size_t v11 = v1 * v1;
  for (std::size_t ab = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; ab < v11;
       ab += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    double sg = 0.0, st = 0.0;
    const double* bv = bar_V + ab * v2;
    for (std::size_t c = 0; c < v2; ++c) {
      const double val = bv[c];
      sg += val * t1_k[c];
      st += val * fov_k[c];
    }
    bg_ij[ab] += sg;
    bt_ij[ab] += st;
  }
}

__global__ void scale_copy_kernel(std::size_t n, double scale, const double* src, double* dst) {
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    dst[i] = scale * src[i];
  }
}

std::size_t triples_headroom(int device, std::size_t device_cap) {
  return std::min(ucc_device_available(device_cap), query_free_device_bytes(device));
}

// The virtual Fock contraction buffer is idle during occupied-response paging. Stage
// the result through the same allocation, including when there are more occupied than
// virtual orbitals and the result needs multiple transfers.
void add_occupied_response(const std::vector<double>& values, DeviceBuf<double>& staging,
                           double* bar_foo, cudaStream_t st) {
  for (std::size_t offset = 0; offset < values.size();) {
    const auto count = std::min(staging.size(), values.size() - offset);
    staging.copy_from_host(values.data() + offset, count, st);
    launch_add_scaled<double>(bar_foo + offset, staging.get(), 1.0, count, st);
    offset += count;
  }
  cuda_check(cudaStreamSynchronize(st), "triples occupied Fock response sync");
}

// --- Paged X/Y off-diagonal occupied Fock resolvent helper ---
// Evaluates M[p, r] = sum_idx (X_p[idx]*Y_r[idx] + Y_p[idx]*X_r[idx]) for all active pairs (p < r)
// using bounded GPU pages (respecting < 24 GiB cap) and cuBLAS DGEMM.
template <class BuildXYFn, class WeightFn>
void accumulate_paged_foo_offdiag(int dev, cublasHandle_t blas, cudaStream_t st,
                                  std::size_t o_dim, std::size_t v_tot,
                                  const std::vector<std::size_t>& active_list,
                                  const std::vector<double>& host_eps, bool has_weights,
                                  double coeff, std::size_t device_cap,
                                  BuildXYFn&& build_xy_into, WeightFn&& pair_weight,
                                  std::vector<double>& host_bar_foo, std::size_t& fp64_gemms) {
  const std::size_t n_act = active_list.size();
  if (n_act < 2 || !v_tot) return;

  // Forward W operands, permutation scratch and the slabs used by build_xy_into
  // already own their buffers. Only X/Y pages and Grams are newly allocated here.
  const auto page_cap = ucc_triples_page_rows(n_act, v_tot, triples_headroom(dev, device_cap));
  const bool single_page = page_cap == n_act;
  const auto xy_elems = checked_mul(page_cap, v_tot);
  const auto gram_elems = checked_mul(page_cap, page_cap);
  DeviceBuf<double> x_a(dev, xy_elems, st), y_a(dev, xy_elems, st);
  DeviceBuf<double> x_b(dev, single_page ? 0 : xy_elems, st);
  DeviceBuf<double> y_b(dev, single_page ? 0 : xy_elems, st);
  DeviceBuf<double> d_g1(dev, gram_elems, st), d_g2(dev, single_page ? 0 : gram_elems, st);
  std::vector<double> h_g1(gram_elems), h_g2(d_g2.size());

  for (std::size_t a0 = 0; a0 < n_act; a0 += page_cap) {
    const std::size_t na = std::min(page_cap, n_act - a0);
    for (std::size_t u = 0; u < na; ++u) {
      build_xy_into(active_list[a0 + u], x_a.get() + u * v_tot, y_a.get() + u * v_tot);
    }
    // Within-page Gram: G1[u, w] = X_A[u] . Y_A[w]
    gemm_counted<double>(blas, st, 'N', 'T', na, na, v_tot, x_a.get(), y_a.get(), d_g1.get(),
                         fp64_gemms, 1.0, 0.0);
    d_g1.copy_to_host(h_g1.data(), na * na, st);
    cuda_check(cudaStreamSynchronize(st), "foo offdiag page sync");
    for (std::size_t u = 0; u < na; ++u) {
      const std::size_t p = active_list[a0 + u];
      for (std::size_t w = u + 1; w < na; ++w) {
        const std::size_t r = active_list[a0 + w];
        if (has_weights && std::abs(host_eps[p] - host_eps[r]) >= 1e-6) continue;
        const double dot = h_g1[u * na + w] + h_g1[w * na + u];
        const double val = coeff * pair_weight(p) * dot;
        host_bar_foo[p * o_dim + r] += val;
        host_bar_foo[r * o_dim + p] += val;
      }
    }
    // Cross-page Grams
    for (std::size_t b0 = a0 + page_cap; b0 < n_act; b0 += page_cap) {
      const std::size_t nb = std::min(page_cap, n_act - b0);
      for (std::size_t w = 0; w < nb; ++w) {
        build_xy_into(active_list[b0 + w], x_b.get() + w * v_tot, y_b.get() + w * v_tot);
      }
      gemm_counted<double>(blas, st, 'N', 'T', na, nb, v_tot, x_a.get(), y_b.get(), d_g1.get(),
                           fp64_gemms, 1.0, 0.0);
      gemm_counted<double>(blas, st, 'N', 'T', na, nb, v_tot, y_a.get(), x_b.get(), d_g2.get(),
                           fp64_gemms, 1.0, 0.0);
      d_g1.copy_to_host(h_g1.data(), na * nb, st);
      d_g2.copy_to_host(h_g2.data(), na * nb, st);
      cuda_check(cudaStreamSynchronize(st), "foo offdiag cross sync");
      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t p = active_list[a0 + u];
        for (std::size_t w = 0; w < nb; ++w) {
          const std::size_t r = active_list[b0 + w];
          if (has_weights && std::abs(host_eps[p] - host_eps[r]) >= 1e-6) continue;
          const double dot = h_g1[u * nb + w] + h_g2[u * nb + w];
          const double val = coeff * pair_weight(p) * dot;
          host_bar_foo[p * o_dim + r] += val;
          host_bar_foo[r * o_dim + p] += val;
        }
      }
    }
  }
}

// --- Same-Spin Channel (aaa / bbb) ---
template <class RealW>
void run_same_spin_triples_cuda(
    int dev, cublasHandle_t blas, cudaStream_t st, std::size_t device_cap, std::size_t o,
    std::size_t v, std::size_t q, const double* bov, const double* bvv, const double* goovv,
    const double* gooov, const double* fov, const double* t1, const double* t2,
    const double* d_eps_v, const std::vector<double>& host_eps_o,
    const std::vector<double>& weights, bool has_weights, bool compute_forces, double den_thresh,
    double& total_e, std::vector<double>& occ_e, double* bar_t1, double* bar_t2, double* bar_bov,
    double* bar_bvv, double* bar_goovv, double* bar_gooov, double* bar_fov, double* bar_foo,
    double* bar_fvv, std::size_t& fp32_gemms, std::size_t& fp64_gemms) {
  if (o < 3 || v < 3) return;
  constexpr bool kFp32W = std::is_same_v<RealW, float>;
  std::size_t& w_gemm_counter = kFp32W ? fp32_gemms : fp64_gemms;
  const std::size_t v2 = checked_mul(v, v), v3 = checked_mul(v, v2);

  // Optional FP32 copies of compact forward-W inputs (bov, bvv, t2, gooov)
  DeviceBuf<RealW> bov_w_buf, bvv_w_buf, t2_w_buf, gooov_w_buf;
  const RealW* bov_w = nullptr;
  const RealW* bvv_w = nullptr;
  const RealW* t2_w = nullptr;
  const RealW* gooov_w = nullptr;
  if constexpr (kFp32W) {
    bov_w_buf.allocate(dev, q * o * v, st);
    bvv_w_buf.allocate(dev, q * v2, st);
    t2_w_buf.allocate(dev, o * o * v2, st);
    gooov_w_buf.allocate(dev, o * o * o * v, st);
    launch_cast(bov, bov_w_buf.get(), q * o * v, st);
    launch_cast(bvv, bvv_w_buf.get(), q * v2, st);
    launch_cast(t2, t2_w_buf.get(), o * o * v2, st);
    launch_cast(gooov, gooov_w_buf.get(), o * o * o * v, st);
    bov_w = bov_w_buf.get();
    bvv_w = bvv_w_buf.get();
    t2_w = t2_w_buf.get();
    gooov_w = gooov_w_buf.get();
  } else {
    bov_w = bov;
    bvv_w = bvv;
    t2_w = t2;
    gooov_w = gooov;
  }

  DeviceBuf<RealW> bov_p_w(dev, q * v, st);
  DeviceBuf<double> bov_p_f64(dev, compute_forces ? q * v : 0, st);
  DeviceBuf<double> bar_bov_p_f64(dev, compute_forces ? q * v : 0, st);

  DeviceBuf<RealW> v_slabs_all, v_slab_i, v_slab_j, v_slab_k;
  DeviceBuf<double> bar_v_all, bar_v_i, bar_v_j, bar_v_k;
  DeviceBuf<double> v_slab_f64_scratch(dev, (compute_forces && kFp32W) ? v3 : 0, st);

  DeviceBuf<RealW> s_acb(dev, v3, st), s_occ(dev, v3, st);
  DeviceBuf<double> W(dev, v3, st);
  DeviceBuf<double> X(dev, compute_forces ? v3 : 0, st), Y(dev, compute_forces ? v3 : 0, st);
  DeviceBuf<double> bar_W(dev, compute_forces ? v3 : 0, st), bar_V(dev, compute_forces ? v3 : 0, st);
  DeviceBuf<double> bar_s_acb(dev, compute_forces ? v3 : 0, st);
  DeviceBuf<double> xy_vv(dev, compute_forces ? v2 : 0, st);

  DeviceBuf<double> d_total_e(dev, 1, st);
  DeviceBuf<double> d_occ_e(dev, o, st);
  DeviceBuf<int> d_err(dev, 1, st);
  d_total_e.zero(st);
  d_occ_e.zero(st);
  d_err.zero(st);

  const bool cache_all_slabs = ucc_triples_cache_slabs<RealW>(
      {{v3, o, 3}}, compute_forces, triples_headroom(dev, device_cap));
  if (cache_all_slabs) {
    v_slabs_all.allocate(dev, o * v3, st);
    {
      CublasMathModeScope math_guard(blas, CUBLAS_PEDANTIC_MATH, kFp32W);
      for (std::size_t p = 0; p < o; ++p) {
        build_ovvv_slab_device<RealW>(dev, blas, st, q, o, v, v, p, bov_w, bvv_w,
                                      v_slabs_all.get() + p * v3, bov_p_w.get(), w_gemm_counter);
      }
    }
    if (compute_forces) bar_v_all.allocate_zero(dev, o * v3, st);
  } else {
    v_slab_i.allocate(dev, v3, st);
    v_slab_j.allocate(dev, v3, st);
    v_slab_k.allocate(dev, v3, st);
    if (compute_forces) {
      bar_v_i.allocate_zero(dev, v3, st);
      bar_v_j.allocate_zero(dev, v3, st);
      bar_v_k.allocate_zero(dev, v3, st);
    }
  }

  auto build_w_with_slabs = [&](std::size_t i, std::size_t j, std::size_t k, const RealW* vi,
                                const RealW* vj, const RealW* vk) {
    s_acb.zero(st);
    s_occ.zero(st);
    const std::size_t cyc[3][3] = {{i, j, k}, {j, k, i}, {k, i, j}};
    const RealW* vptrs[3] = {vi, vj, vk};
    {
      CublasMathModeScope math_guard(blas, CUBLAS_PEDANTIC_MATH, kFp32W);
      for (int m = 0; m < 3; ++m) {
        const std::size_t p = cyc[m][0], r = cyc[m][1], s = cyc[m][2];
        gemm_counted<RealW>(blas, st, 'N', 'T', v, v2, v, t2_w + (r * o + s) * v2, vptrs[m],
                            s_acb.get(), w_gemm_counter, RealW(1), RealW(1));
        gemm_counted<RealW>(blas, st, 'T', 'N', v, v2, o, gooov_w + (r * o + s) * o * v,
                            t2_w + p * o * v2, s_occ.get(), w_gemm_counter, RealW(1), RealW(1));
      }
    }
    assemble_w_same_spin_kernel<RealW><<<grid_1d(v3), 256, 0, st>>>(v, s_acb.get(), s_occ.get(),
                                                                    W.get());
    cuda_check(cudaGetLastError(), "assemble_w_same_spin");
  };

  for (std::size_t i = 0; i < o; ++i) {
    if (!cache_all_slabs) {
      build_ovvv_slab_device<RealW>(dev, blas, st, q, o, v, v, i, bov_w, bvv_w, v_slab_i.get(),
                                    bov_p_w.get(), w_gemm_counter);
      if (compute_forces) bar_v_i.zero(st);
    }
    for (std::size_t j = i + 1; j < o; ++j) {
      if (!cache_all_slabs) {
        build_ovvv_slab_device<RealW>(dev, blas, st, q, o, v, v, j, bov_w, bvv_w, v_slab_j.get(),
                                      bov_p_w.get(), w_gemm_counter);
        if (compute_forces) bar_v_j.zero(st);
      }
      for (std::size_t k = j + 1; k < o; ++k) {
        const RealW* vi = cache_all_slabs ? (v_slabs_all.get() + i * v3) : v_slab_i.get();
        const RealW* vj = cache_all_slabs ? (v_slabs_all.get() + j * v3) : v_slab_j.get();
        const RealW* vk = cache_all_slabs ? (v_slabs_all.get() + k * v3) : v_slab_k.get();
        if (!cache_all_slabs) {
          build_ovvv_slab_device<RealW>(dev, blas, st, q, o, v, v, k, bov_w, bvv_w, v_slab_k.get(),
                                        bov_p_w.get(), w_gemm_counter);
          if (compute_forces) bar_v_k.zero(st);
        }

        build_w_with_slabs(i, j, k, vi, vj, vk);
        const double eijk_o = host_eps_o[i] + host_eps_o[j] + host_eps_o[k];
        const double wt = has_weights ? (weights[i] + weights[j] + weights[k]) / 3.0 : 1.0;

        eval_same_spin_triple_kernel<<<grid_1d(v3), 256, 0, st>>>(
            o, v, i, j, k, eijk_o, wt, den_thresh, compute_forces, W.get(), t1, fov, goovv, t2,
            d_eps_v, X.get(), Y.get(), bar_W.get(), bar_V.get(), d_total_e.get(), d_occ_e.get(),
            bar_foo, d_err.get());
        cuda_check(cudaGetLastError(), "eval_same_spin_triple");

        if (!compute_forces) continue;

        // Virtual Fock resolvent bar_fvv
        gemm_counted<double>(blas, st, 'N', 'T', v, v, v2, X.get(), Y.get(), xy_vv.get(),
                             fp64_gemms, 1.0, 0.0);
        add_sym_xy_fvv_kernel<<<grid_1d(v2), 256, 0, st>>>(v, 0.25 * wt, xy_vv.get(), bar_fvv);
        cuda_check(cudaGetLastError(), "same_spin bar_fvv");

        // bar_s_acb[a, c, b] = 6.0 * bar_W[a, b, c]
        launch_permute3d<double, 0, 2, 1>(v, v, v, bar_W.get(), bar_s_acb.get(), st, 6.0, false);

        const std::size_t cyc[3][3] = {{i, j, k}, {j, k, i}, {k, i, j}};
        const RealW* vptrs[3] = {vi, vj, vk};
        double* bvptrs[3] = {cache_all_slabs ? (bar_v_all.get() + i * v3) : bar_v_i.get(),
                             cache_all_slabs ? (bar_v_all.get() + j * v3) : bar_v_j.get(),
                             cache_all_slabs ? (bar_v_all.get() + k * v3) : bar_v_k.get()};

        for (int m = 0; m < 3; ++m) {
          const std::size_t p = cyc[m][0], r = cyc[m][1], s = cyc[m][2];
          const double* vp_f64 = nullptr;
          if constexpr (kFp32W) {
            launch_cast(vptrs[m], v_slab_f64_scratch.get(), v3, st);
            vp_f64 = v_slab_f64_scratch.get();
          } else {
            vp_f64 = vptrs[m];
          }
          gemm_counted<double>(blas, st, 'N', 'N', v, v, v2, bar_s_acb.get(), vp_f64,
                               bar_t2 + (r * o + s) * v2, fp64_gemms, 1.0, 1.0);
          gemm_counted<double>(blas, st, 'T', 'N', v2, v, v, bar_s_acb.get(), t2 + (r * o + s) * v2,
                               bvptrs[m], fp64_gemms, 1.0, 1.0);
          gemm_counted<double>(blas, st, 'N', 'T', o, v, v2, t2 + p * o * v2, bar_W.get(),
                               bar_gooov + (r * o + s) * o * v, fp64_gemms, -3.0, 1.0);
          gemm_counted<double>(blas, st, 'N', 'N', o, v2, v, gooov + (r * o + s) * o * v,
                               bar_W.get(), bar_t2 + p * o * v2, fp64_gemms, -3.0, 1.0);
          pullback_v_same_spin_a_kernel<<<grid_1d(v), 256, 0, st>>>(
              v, bar_V.get(), goovv + (r * o + s) * v2, t2 + (r * o + s) * v2, bar_t1 + p * v,
              bar_fov + p * v);
          pullback_v_same_spin_bc_kernel<<<grid_1d(v2), 256, 0, st>>>(
              v, bar_V.get(), t1 + p * v, fov + p * v, bar_goovv + (r * o + s) * v2,
              bar_t2 + (r * o + s) * v2);
          cuda_check(cudaGetLastError(), "pullback_v_same_spin");
        }
        if (!cache_all_slabs) {
          pullback_ovvv_slab_device(blas, st, q, o, v, v, k, bov, bvv, bar_v_k.get(), bar_bov,
                                    bar_bvv, bov_p_f64.get(), bar_bov_p_f64.get(), fp64_gemms);
        }
      }
      if (!cache_all_slabs && compute_forces) {
        pullback_ovvv_slab_device(blas, st, q, o, v, v, j, bov, bvv, bar_v_j.get(), bar_bov,
                                  bar_bvv, bov_p_f64.get(), bar_bov_p_f64.get(), fp64_gemms);
      }
    }
    if (!cache_all_slabs && compute_forces) {
      pullback_ovvv_slab_device(blas, st, q, o, v, v, i, bov, bvv, bar_v_i.get(), bar_bov, bar_bvv,
                                bov_p_f64.get(), bar_bov_p_f64.get(), fp64_gemms);
    }
  }

  double ch_e = 0.0;
  std::vector<double> ch_occ(o, 0.0);
  int err = 0;
  d_total_e.copy_to_host(&ch_e, 1, st);
  d_occ_e.copy_to_host(ch_occ.data(), o, st);
  d_err.copy_to_host(&err, 1, st);
  cuda_check(cudaStreamSynchronize(st), "same_spin finish sync");
  if (err) throw std::runtime_error("DF-UCCSD(T) v2 CUDA small same-spin triples denominator");
  total_e += ch_e;
  for (std::size_t idx = 0; idx < o; ++idx) occ_e[idx] += ch_occ[idx];

  if (!compute_forces) return;

  if (cache_all_slabs) {
    for (std::size_t p = 0; p < o; ++p) {
      pullback_ovvv_slab_device(blas, st, q, o, v, v, p, bov, bvv, bar_v_all.get() + p * v3,
                                bar_bov, bar_bvv, bov_p_f64.get(), bar_bov_p_f64.get(), fp64_gemms);
    }
  }
  // Free backward slab buffers before paged X/Y occupied Fock resolvent.
  bar_v_all.reset(); bar_v_i.reset(); bar_v_j.reset(); bar_v_k.reset();
  bar_W.reset(); bar_V.reset(); bar_s_acb.reset(); X.reset(); Y.reset();

  bool need_offdiag = !has_weights;
  if (has_weights) {
    for (std::size_t p = 0; p < o && !need_offdiag; ++p)
      for (std::size_t r = p + 1; r < o; ++r)
        if (std::abs(host_eps_o[p] - host_eps_o[r]) < 1e-6) need_offdiag = true;
  }
  if (need_offdiag && o >= 4) {
    std::vector<double> host_bar_foo_off(o * o, 0.0);
    for (std::size_t j = 0; j < o; ++j) {
      if (!cache_all_slabs) {
        build_ovvv_slab_device<RealW>(dev, blas, st, q, o, v, v, j, bov_w, bvv_w, v_slab_j.get(),
                                      bov_p_w.get(), w_gemm_counter);
      }
      for (std::size_t k = j + 1; k < o; ++k) {
        std::vector<std::size_t> active;
        for (std::size_t p = 0; p < o; ++p) {
          if (p == j || p == k) continue;
          if (!has_weights) {
            active.push_back(p);
          } else {
            for (std::size_t r = 0; r < o; ++r) {
              if (r != p && r != j && r != k && std::abs(host_eps_o[p] - host_eps_o[r]) < 1e-6) {
                active.push_back(p);
                break;
              }
            }
          }
        }
        if (active.size() < 2) continue;
        if (!cache_all_slabs) {
          build_ovvv_slab_device<RealW>(dev, blas, st, q, o, v, v, k, bov_w, bvv_w, v_slab_k.get(),
                                        bov_p_w.get(), w_gemm_counter);
        }
        const RealW* vj = cache_all_slabs ? (v_slabs_all.get() + j * v3) : v_slab_j.get();
        const RealW* vk = cache_all_slabs ? (v_slabs_all.get() + k * v3) : v_slab_k.get();
        accumulate_paged_foo_offdiag(
            dev, blas, st, o, v3, active, host_eps_o, has_weights, -(1.0 / 12.0), device_cap,
            [&](std::size_t p, double* x_out, double* y_out) {
              const RealW* vp = cache_all_slabs ? (v_slabs_all.get() + p * v3) : v_slab_i.get();
              if (!cache_all_slabs) {
                build_ovvv_slab_device<RealW>(dev, blas, st, q, o, v, v, p, bov_w, bvv_w,
                                              v_slab_i.get(), bov_p_w.get(), w_gemm_counter);
              }
              build_w_with_slabs(p, j, k, vp, vj, vk);
              const double epjk = host_eps_o[p] + host_eps_o[j] + host_eps_o[k];
              eval_same_spin_xy_only_kernel<<<grid_1d(v3), 256, 0, st>>>(
                  o, v, p, j, k, epjk, W.get(), t1, fov, goovv, t2, d_eps_v, x_out, y_out);
              cuda_check(cudaGetLastError(), "same_spin xy_only");
            },
            [&](std::size_t p) {
              return has_weights ? (weights[p] + weights[j] + weights[k]) / 3.0 : 1.0;
            },
            host_bar_foo_off, fp64_gemms);
      }
    }
    add_occupied_response(host_bar_foo_off, xy_vv, bar_foo, st);
  }
}

// --- Mixed-Spin Channel (aab / abb) ---
template <class RealW>
void run_mixed_spin_triples_cuda(
    int dev, cublasHandle_t blas, cudaStream_t st, std::size_t device_cap, std::size_t o1,
    std::size_t o2, std::size_t v1, std::size_t v2, std::size_t q, const double* bov1,
    const double* bvv1, const double* bov2, const double* bvv2, const double* goovv11,
    const double* goovv12, const double* gooov11, const double* gooov12, const double* gooov21,
    const double* fov1, const double* fov2, const double* t1_1, const double* t1_2,
    const double* t2_11, const double* t2_12, const double* d_eps_v1, const double* d_eps_v2,
    const std::vector<double>& host_eps_o1, const std::vector<double>& host_eps_o2,
    const std::vector<double>& w1, const std::vector<double>& w2, bool has_weights,
    bool compute_forces, double den_thresh, double& total_e, std::vector<double>& occ1,
    std::vector<double>& occ2, double* bar_t1_1, double* bar_t1_2, double* bar_t2_11,
    double* bar_t2_12, double* bar_bov1, double* bar_bvv1, double* bar_bov2, double* bar_bvv2,
    double* bar_goovv11, double* bar_goovv12, double* bar_gooov11, double* bar_gooov12,
    double* bar_gooov21, double* bar_fov1, double* bar_fov2, double* bar_foo1, double* bar_foo2,
    double* bar_fvv1, double* bar_fvv2, std::size_t& fp32_gemms, std::size_t& fp64_gemms) {
  if (o1 < 2 || o2 < 1 || v1 < 2 || v2 < 1) return;
  constexpr bool kFp32W = std::is_same_v<RealW, float>;
  std::size_t& w_gemm_counter = kFp32W ? fp32_gemms : fp64_gemms;
  const std::size_t v11 = checked_mul(v1, v1), v12 = checked_mul(v1, v2),
                    v22 = checked_mul(v2, v2), v_tot = checked_mul(v11, v2);

  // Transposed views on GPU (FP64)
  // g21_jk[j, k, a, m] from gooov21[m, j, k, a]
  DeviceBuf<double> g21_jk(dev, o1 * o2 * v1 * o2, st);
  launch_permute4d<double, 1, 2, 3, 0>(o2, o1, o2, v1, gooov21, g21_jk.get(), st);
  // g12_kj[k, j, m, c] from gooov12[m, k, j, c]
  DeviceBuf<double> g12_kj(dev, o2 * o1 * o1 * v2, st);
  launch_permute4d<double, 1, 2, 0, 3>(o1, o2, o1, v2, gooov12, g12_kj.get(), st);
  // t2_12_k_m[k, m, bc] from t2_12[m, k, bc]
  DeviceBuf<double> t2_12_k_m(dev, o2 * o1 * v12, st);
  launch_permute3d<double, 1, 0, 2>(o1, o2, v12, t2_12, t2_12_k_m.get(), st);

  DeviceBuf<double> bar_g21_jk(dev, compute_forces ? g21_jk.size() : 0, st);
  DeviceBuf<double> bar_g12_kj(dev, compute_forces ? g12_kj.size() : 0, st);
  DeviceBuf<double> bar_t2_12_k_m(dev, compute_forces ? t2_12_k_m.size() : 0, st);
  if (compute_forces) {
    bar_g21_jk.zero(st);
    bar_g12_kj.zero(st);
    bar_t2_12_k_m.zero(st);
  }

  // Forward-W typed operands
  DeviceBuf<RealW> bov1_wb, bvv1_wb, bov2_wb, bvv2_wb, t2_11_wb, t2_12_wb, t2_12_km_wb,
      gooov11_wb, g21_jk_wb, g12_kj_wb;
  const RealW *bov1_w{}, *bvv1_w{}, *bov2_w{}, *bvv2_w{}, *t2_11_w{}, *t2_12_w{}, *t2_12_km_w{},
      *gooov11_w{}, *g21_jk_w{}, *g12_kj_w{};
  if constexpr (kFp32W) {
    bov1_wb.allocate(dev, q * o1 * v1, st); launch_cast(bov1, bov1_wb.get(), q * o1 * v1, st);
    bvv1_wb.allocate(dev, q * v11, st); launch_cast(bvv1, bvv1_wb.get(), q * v11, st);
    bov2_wb.allocate(dev, q * o2 * v2, st); launch_cast(bov2, bov2_wb.get(), q * o2 * v2, st);
    bvv2_wb.allocate(dev, q * v22, st); launch_cast(bvv2, bvv2_wb.get(), q * v22, st);
    t2_11_wb.allocate(dev, o1 * o1 * v11, st); launch_cast(t2_11, t2_11_wb.get(), o1 * o1 * v11, st);
    t2_12_wb.allocate(dev, o1 * o2 * v12, st); launch_cast(t2_12, t2_12_wb.get(), o1 * o2 * v12, st);
    t2_12_km_wb.allocate(dev, o2 * o1 * v12, st);
    launch_cast(t2_12_k_m.get(), t2_12_km_wb.get(), o2 * o1 * v12, st);
    gooov11_wb.allocate(dev, o1 * o1 * o1 * v1, st);
    launch_cast(gooov11, gooov11_wb.get(), o1 * o1 * o1 * v1, st);
    g21_jk_wb.allocate(dev, g21_jk.size(), st);
    launch_cast(g21_jk.get(), g21_jk_wb.get(), g21_jk.size(), st);
    g12_kj_wb.allocate(dev, g12_kj.size(), st);
    launch_cast(g12_kj.get(), g12_kj_wb.get(), g12_kj.size(), st);
    bov1_w = bov1_wb.get(); bvv1_w = bvv1_wb.get(); bov2_w = bov2_wb.get(); bvv2_w = bvv2_wb.get();
    t2_11_w = t2_11_wb.get(); t2_12_w = t2_12_wb.get(); t2_12_km_w = t2_12_km_wb.get();
    gooov11_w = gooov11_wb.get(); g21_jk_w = g21_jk_wb.get(); g12_kj_w = g12_kj_wb.get();
  } else {
    bov1_w = bov1; bvv1_w = bvv1; bov2_w = bov2; bvv2_w = bvv2;
    t2_11_w = t2_11; t2_12_w = t2_12; t2_12_km_w = t2_12_k_m.get();
    gooov11_w = gooov11; g21_jk_w = g21_jk.get(); g12_kj_w = g12_kj.get();
  }

  const std::size_t s11 = checked_mul(v1, v11), s12 = checked_mul(v1, v22), s21 = v_tot;
  const std::size_t max_qv = checked_mul(q, std::max(v1, v2));
  DeviceBuf<RealW> bov_p_w(dev, max_qv, st);
  DeviceBuf<double> bov_p_f64(dev, compute_forces ? max_qv : 0, st);
  DeviceBuf<double> bar_bov_p_f64(dev, compute_forces ? max_qv : 0, st);

  DeviceBuf<RealW> v11_all, v12_all, v21_all;
  DeviceBuf<RealW> v11_i, v11_j, v12_i, v12_j, v21_k;
  DeviceBuf<double> bv11_all, bv12_all, bv21_all;
  DeviceBuf<double> bv11_i, bv11_j, bv12_i, bv12_j, bv21_k;
  DeviceBuf<double> slab_f64_scratch(
      dev, (compute_forces && kFp32W) ? std::max({s11, s12, s21}) : 0, st);

  DeviceBuf<RealW> R(dev, v_tot, st), tmp_acb(dev, v_tot, st), W(dev, v_tot, st);
  DeviceBuf<double> X(dev, compute_forces ? v_tot : 0, st), Y(dev, compute_forces ? v_tot : 0, st);
  DeviceBuf<double> bar_W(dev, compute_forces ? v_tot : 0, st),
      bar_V(dev, compute_forces ? v_tot : 0, st);
  DeviceBuf<double> bar_R(dev, compute_forces ? v_tot : 0, st),
      bar_R_acb(dev, compute_forces ? v_tot : 0, st);
  DeviceBuf<double> xy1(dev, compute_forces ? v11 : 0, st), xy2(dev, compute_forces ? v22 : 0, st);

  DeviceBuf<double> d_total_e(dev, 1, st), d_occ1(dev, o1, st), d_occ2(dev, o2, st);
  DeviceBuf<int> d_err(dev, 1, st);
  d_total_e.zero(st); d_occ1.zero(st); d_occ2.zero(st); d_err.zero(st);

  const bool cache_all_slabs = ucc_triples_cache_slabs<RealW>(
      {{s11, o1, 2}, {s12, o1, 2}, {s21, o2, 1}}, compute_forces,
      triples_headroom(dev, device_cap));
  if (cache_all_slabs) {
    v11_all.allocate(dev, o1 * s11, st);
    v12_all.allocate(dev, o1 * s12, st);
    v21_all.allocate(dev, o2 * s21, st);
    {
      CublasMathModeScope math_guard(blas, CUBLAS_PEDANTIC_MATH, kFp32W);
      for (std::size_t p = 0; p < o1; ++p) {
        build_ovvv_slab_device<RealW>(dev, blas, st, q, o1, v1, v1, p, bov1_w, bvv1_w,
                                      v11_all.get() + p * s11, bov_p_w.get(), w_gemm_counter);
        build_ovvv_slab_device<RealW>(dev, blas, st, q, o1, v1, v2, p, bov1_w, bvv2_w,
                                      v12_all.get() + p * s12, bov_p_w.get(), w_gemm_counter);
      }
      for (std::size_t k = 0; k < o2; ++k) {
        build_ovvv_slab_device<RealW>(dev, blas, st, q, o2, v2, v1, k, bov2_w, bvv1_w,
                                      v21_all.get() + k * s21, bov_p_w.get(), w_gemm_counter);
      }
    }
    if (compute_forces) {
      bv11_all.allocate_zero(dev, o1 * s11, st);
      bv12_all.allocate_zero(dev, o1 * s12, st);
      bv21_all.allocate_zero(dev, o2 * s21, st);
    }
  } else {
    v11_i.allocate(dev, s11, st); v11_j.allocate(dev, s11, st);
    v12_i.allocate(dev, s12, st); v12_j.allocate(dev, s12, st);
    v21_k.allocate(dev, s21, st);
    if (compute_forces) {
      bv11_i.allocate_zero(dev, s11, st); bv11_j.allocate_zero(dev, s11, st);
      bv12_i.allocate_zero(dev, s12, st); bv12_j.allocate_zero(dev, s12, st);
      bv21_k.allocate_zero(dev, s21, st);
    }
  }

  auto build_w_mixed = [&](std::size_t i, std::size_t j, std::size_t k, const RealW* s11_i,
                           const RealW* s11_j, const RealW* s12_i, const RealW* s12_j,
                           const RealW* s21_k) {
    CublasMathModeScope math_guard(blas, CUBLAS_PEDANTIC_MATH, kFp32W);
    // Term 1: t2_11[i,j,a,e] @ v21[k, c*b, e]^T -> tmp_acb -> R
    gemm_counted<RealW>(blas, st, 'N', 'T', v1, v2 * v1, v1, t2_11_w + (i * o1 + j) * v11, s21_k,
                        tmp_acb.get(), w_gemm_counter, RealW(1), RealW(0));
    add_acb_to_abc_kernel<RealW><<<grid_1d(v_tot), 256, 0, st>>>(v1, v2, tmp_acb.get(), R.get());
    // Term 2: - t2_12[j,k] @ v12[i]^T + t2_12[i,k] @ v12[j]^T
    gemm_counted<RealW>(blas, st, 'N', 'T', v1, v12, v2, t2_12_w + (j * o2 + k) * v12, s12_i,
                        R.get(), w_gemm_counter, RealW(-1), RealW(1));
    gemm_counted<RealW>(blas, st, 'N', 'T', v1, v12, v2, t2_12_w + (i * o2 + k) * v12, s12_j,
                        R.get(), w_gemm_counter, RealW(1), RealW(1));
    // Term 3: + v11[i] @ t2_12[j,k] - v11[j] @ t2_12[i,k]
    gemm_counted<RealW>(blas, st, 'N', 'N', v11, v2, v1, s11_i, t2_12_w + (j * o2 + k) * v12,
                        R.get(), w_gemm_counter, RealW(1), RealW(1));
    gemm_counted<RealW>(blas, st, 'N', 'N', v11, v2, v1, s11_j, t2_12_w + (i * o2 + k) * v12,
                        R.get(), w_gemm_counter, RealW(-1), RealW(1));
    // Term 4: + g21_jk[j,k] @ t2_12[i] - g21_jk[i,k] @ t2_12[j]
    gemm_counted<RealW>(blas, st, 'N', 'N', v1, v12, o2, g21_jk_w + (j * o2 + k) * v1 * o2,
                        t2_12_w + i * o2 * v12, R.get(), w_gemm_counter, RealW(1), RealW(1));
    gemm_counted<RealW>(blas, st, 'N', 'N', v1, v12, o2, g21_jk_w + (i * o2 + k) * v1 * o2,
                        t2_12_w + j * o2 * v12, R.get(), w_gemm_counter, RealW(-1), RealW(1));
    // Term 6: + gooov11[i,j]^T @ t2_12_k_m[k]
    gemm_counted<RealW>(blas, st, 'T', 'N', v1, v12, o1, gooov11_w + (i * o1 + j) * o1 * v1,
                        t2_12_km_w + k * o1 * v12, R.get(), w_gemm_counter, RealW(1), RealW(1));
    // Antisymmetrize R -> W
    antisym_ab_mixed_kernel<RealW><<<grid_1d(v_tot), 256, 0, st>>>(v1, v2, R.get(), W.get());
    // Term 5 directly into W: - t2_11[i]^T @ g12_kj[k,j] + t2_11[j]^T @ g12_kj[k,i]
    gemm_counted<RealW>(blas, st, 'T', 'N', v11, v2, o1, t2_11_w + i * o1 * v11,
                        g12_kj_w + (k * o1 + j) * o1 * v2, W.get(), w_gemm_counter, RealW(-1),
                        RealW(1));
    gemm_counted<RealW>(blas, st, 'T', 'N', v11, v2, o1, t2_11_w + j * o1 * v11,
                        g12_kj_w + (k * o1 + i) * o1 * v2, W.get(), w_gemm_counter, RealW(1),
                        RealW(1));
    cuda_check(cudaGetLastError(), "build_w_mixed");
  };

  auto as_f64_slab = [&](const RealW* ptr, std::size_t elems) -> const double* {
    if constexpr (kFp32W) {
      launch_cast(ptr, slab_f64_scratch.get(), elems, st);
      return slab_f64_scratch.get();
    } else {
      return ptr;
    }
  };

  for (std::size_t i = 0; i < o1; ++i) {
    if (!cache_all_slabs) {
      build_ovvv_slab_device<RealW>(dev, blas, st, q, o1, v1, v1, i, bov1_w, bvv1_w, v11_i.get(),
                                    bov_p_w.get(), w_gemm_counter);
      build_ovvv_slab_device<RealW>(dev, blas, st, q, o1, v1, v2, i, bov1_w, bvv2_w, v12_i.get(),
                                    bov_p_w.get(), w_gemm_counter);
      if (compute_forces) { bv11_i.zero(st); bv12_i.zero(st); }
    }
    for (std::size_t j = i + 1; j < o1; ++j) {
      if (!cache_all_slabs) {
        build_ovvv_slab_device<RealW>(dev, blas, st, q, o1, v1, v1, j, bov1_w, bvv1_w, v11_j.get(),
                                      bov_p_w.get(), w_gemm_counter);
        build_ovvv_slab_device<RealW>(dev, blas, st, q, o1, v1, v2, j, bov1_w, bvv2_w, v12_j.get(),
                                      bov_p_w.get(), w_gemm_counter);
        if (compute_forces) { bv11_j.zero(st); bv12_j.zero(st); }
      }
      for (std::size_t k = 0; k < o2; ++k) {
        const RealW* p11_i = cache_all_slabs ? (v11_all.get() + i * s11) : v11_i.get();
        const RealW* p11_j = cache_all_slabs ? (v11_all.get() + j * s11) : v11_j.get();
        const RealW* p12_i = cache_all_slabs ? (v12_all.get() + i * s12) : v12_i.get();
        const RealW* p12_j = cache_all_slabs ? (v12_all.get() + j * s12) : v12_j.get();
        const RealW* p21_k = cache_all_slabs ? (v21_all.get() + k * s21) : v21_k.get();
        if (!cache_all_slabs) {
          build_ovvv_slab_device<RealW>(dev, blas, st, q, o2, v2, v1, k, bov2_w, bvv1_w,
                                        v21_k.get(), bov_p_w.get(), w_gemm_counter);
          if (compute_forces) bv21_k.zero(st);
        }

        build_w_mixed(i, j, k, p11_i, p11_j, p12_i, p12_j, p21_k);
        const double eijk_o = host_eps_o1[i] + host_eps_o1[j] + host_eps_o2[k];
        const double wt = has_weights ? (w1[i] + w1[j] + w2[k]) / 3.0 : 1.0;

        eval_mixed_spin_triple_kernel<RealW><<<grid_1d(v_tot), 256, 0, st>>>(
            o1, o2, v1, v2, i, j, k, eijk_o, wt, den_thresh, compute_forces, W.get(), t1_1, t1_2,
            fov1, fov2, goovv11, goovv12, t2_11, t2_12, d_eps_v1, d_eps_v2, X.get(), Y.get(),
            bar_W.get(), bar_V.get(), d_total_e.get(), d_occ1.get(), d_occ2.get(), bar_foo1,
            bar_foo2, d_err.get());
        cuda_check(cudaGetLastError(), "eval_mixed_spin_triple");

        if (!compute_forces) continue;

        // Virtual Fock resolvents
        gemm_counted<double>(blas, st, 'N', 'T', v1, v1, v12, X.get(), Y.get(), xy1.get(),
                             fp64_gemms, 1.0, 0.0);
        add_sym_xy_fvv_kernel<<<grid_1d(v11), 256, 0, st>>>(v1, 0.5 * wt, xy1.get(), bar_fvv1);
        gemm_counted<double>(blas, st, 'T', 'N', v2, v2, v11, X.get(), Y.get(), xy2.get(),
                             fp64_gemms, 1.0, 0.0);
        add_sym_xy_fvv_kernel<<<grid_1d(v22), 256, 0, st>>>(v2, 0.25 * wt, xy2.get(), bar_fvv2);

        scale_copy_kernel<<<grid_1d(v_tot), 256, 0, st>>>(v_tot, 2.0, bar_W.get(), bar_R.get());
        launch_permute3d<double, 0, 2, 1>(v1, v1, v2, bar_R.get(), bar_R_acb.get(), st);

        double* pb11_i = cache_all_slabs ? (bv11_all.get() + i * s11) : bv11_i.get();
        double* pb11_j = cache_all_slabs ? (bv11_all.get() + j * s11) : bv11_j.get();
        double* pb12_i = cache_all_slabs ? (bv12_all.get() + i * s12) : bv12_i.get();
        double* pb12_j = cache_all_slabs ? (bv12_all.get() + j * s12) : bv12_j.get();
        double* pb21_k = cache_all_slabs ? (bv21_all.get() + k * s21) : bv21_k.get();

        // Pullback Term 1
        gemm_counted<double>(blas, st, 'N', 'N', v1, v1, v12, bar_R_acb.get(),
                             as_f64_slab(p21_k, s21), bar_t2_11 + (i * o1 + j) * v11, fp64_gemms,
                             1.0, 1.0);
        gemm_counted<double>(blas, st, 'T', 'N', v12, v1, v1, bar_R_acb.get(),
                             t2_11 + (i * o1 + j) * v11, pb21_k, fp64_gemms, 1.0, 1.0);
        // Pullback Term 2
        gemm_counted<double>(blas, st, 'N', 'N', v1, v2, v12, bar_R.get(), as_f64_slab(p12_i, s12),
                             bar_t2_12 + (j * o2 + k) * v12, fp64_gemms, -1.0, 1.0);
        gemm_counted<double>(blas, st, 'T', 'N', v12, v2, v1, bar_R.get(),
                             t2_12 + (j * o2 + k) * v12, pb12_i, fp64_gemms, -1.0, 1.0);
        gemm_counted<double>(blas, st, 'N', 'N', v1, v2, v12, bar_R.get(), as_f64_slab(p12_j, s12),
                             bar_t2_12 + (i * o2 + k) * v12, fp64_gemms, 1.0, 1.0);
        gemm_counted<double>(blas, st, 'T', 'N', v12, v2, v1, bar_R.get(),
                             t2_12 + (i * o2 + k) * v12, pb12_j, fp64_gemms, 1.0, 1.0);
        // Pullback Term 3
        gemm_counted<double>(blas, st, 'N', 'T', v11, v1, v2, bar_R.get(),
                             t2_12 + (j * o2 + k) * v12, pb11_i, fp64_gemms, 1.0, 1.0);
        gemm_counted<double>(blas, st, 'T', 'N', v1, v2, v11, as_f64_slab(p11_i, s11), bar_R.get(),
                             bar_t2_12 + (j * o2 + k) * v12, fp64_gemms, 1.0, 1.0);
        gemm_counted<double>(blas, st, 'N', 'T', v11, v1, v2, bar_R.get(),
                             t2_12 + (i * o2 + k) * v12, pb11_j, fp64_gemms, -1.0, 1.0);
        gemm_counted<double>(blas, st, 'T', 'N', v1, v2, v11, as_f64_slab(p11_j, s11), bar_R.get(),
                             bar_t2_12 + (i * o2 + k) * v12, fp64_gemms, -1.0, 1.0);
        // Pullback Term 4
        gemm_counted<double>(blas, st, 'N', 'T', v1, o2, v12, bar_R.get(), t2_12 + i * o2 * v12,
                             bar_g21_jk.get() + (j * o2 + k) * v1 * o2, fp64_gemms, 1.0, 1.0);
        gemm_counted<double>(blas, st, 'T', 'N', o2, v12, v1, g21_jk.get() + (j * o2 + k) * v1 * o2,
                             bar_R.get(), bar_t2_12 + i * o2 * v12, fp64_gemms, 1.0, 1.0);
        gemm_counted<double>(blas, st, 'N', 'T', v1, o2, v12, bar_R.get(), t2_12 + j * o2 * v12,
                             bar_g21_jk.get() + (i * o2 + k) * v1 * o2, fp64_gemms, -1.0, 1.0);
        gemm_counted<double>(blas, st, 'T', 'N', o2, v12, v1, g21_jk.get() + (i * o2 + k) * v1 * o2,
                             bar_R.get(), bar_t2_12 + j * o2 * v12, fp64_gemms, -1.0, 1.0);
        // Pullback Term 6
        gemm_counted<double>(blas, st, 'N', 'T', o1, v1, v12, t2_12_k_m.get() + k * o1 * v12,
                             bar_R.get(), bar_gooov11 + (i * o1 + j) * o1 * v1, fp64_gemms, 1.0,
                             1.0);
        gemm_counted<double>(blas, st, 'N', 'N', o1, v12, v1, gooov11 + (i * o1 + j) * o1 * v1,
                             bar_R.get(), bar_t2_12_k_m.get() + k * o1 * v12, fp64_gemms, 1.0, 1.0);
        // Pullback Term 5
        gemm_counted<double>(blas, st, 'N', 'T', o1, v11, v2, g12_kj.get() + (k * o1 + j) * o1 * v2,
                             bar_W.get(), bar_t2_11 + i * o1 * v11, fp64_gemms, -1.0, 1.0);
        gemm_counted<double>(blas, st, 'N', 'N', o1, v2, v11, t2_11 + i * o1 * v11, bar_W.get(),
                             bar_g12_kj.get() + (k * o1 + j) * o1 * v2, fp64_gemms, -1.0, 1.0);
        gemm_counted<double>(blas, st, 'N', 'T', o1, v11, v2, g12_kj.get() + (k * o1 + i) * o1 * v2,
                             bar_W.get(), bar_t2_11 + j * o1 * v11, fp64_gemms, 1.0, 1.0);
        gemm_counted<double>(blas, st, 'N', 'N', o1, v2, v11, t2_11 + j * o1 * v11, bar_W.get(),
                             bar_g12_kj.get() + (k * o1 + i) * o1 * v2, fp64_gemms, 1.0, 1.0);

        // Pullback disconnected V[a, b, c]
        pullback_v_mixed_a_kernel<<<grid_1d(v1), 256, 0, st>>>(
            v1, v2, bar_V.get(), goovv12 + (j * o2 + k) * v12, goovv12 + (i * o2 + k) * v12,
            t2_12 + (j * o2 + k) * v12, t2_12 + (i * o2 + k) * v12, bar_t1_1 + i * v1,
            bar_fov1 + i * v1, bar_t1_1 + j * v1, bar_fov1 + j * v1);
        pullback_v_mixed_bc_kernel<<<grid_1d(v12), 256, 0, st>>>(
            v1, v2, bar_V.get(), t1_1 + i * v1, fov1 + i * v1, t1_1 + j * v1, fov1 + j * v1,
            bar_goovv12 + (j * o2 + k) * v12, bar_t2_12 + (j * o2 + k) * v12,
            bar_goovv12 + (i * o2 + k) * v12, bar_t2_12 + (i * o2 + k) * v12);
        pullback_v_mixed_c_kernel<<<grid_1d(v2), 256, 0, st>>>(
            v1, v2, bar_V.get(), goovv11 + (i * o1 + j) * v11, t2_11 + (i * o1 + j) * v11,
            bar_t1_2 + k * v2, bar_fov2 + k * v2);
        pullback_v_mixed_ab_kernel<<<grid_1d(v11), 256, 0, st>>>(
            v1, v2, bar_V.get(), t1_2 + k * v2, fov2 + k * v2, bar_goovv11 + (i * o1 + j) * v11,
            bar_t2_11 + (i * o1 + j) * v11);
        cuda_check(cudaGetLastError(), "pullback_v_mixed");

        if (!cache_all_slabs) {
          pullback_ovvv_slab_device(blas, st, q, o2, v2, v1, k, bov2, bvv1, bv21_k.get(), bar_bov2,
                                    bar_bvv1, bov_p_f64.get(), bar_bov_p_f64.get(), fp64_gemms);
        }
      }
      if (!cache_all_slabs && compute_forces) {
        pullback_ovvv_slab_device(blas, st, q, o1, v1, v1, j, bov1, bvv1, bv11_j.get(), bar_bov1,
                                  bar_bvv1, bov_p_f64.get(), bar_bov_p_f64.get(), fp64_gemms);
        pullback_ovvv_slab_device(blas, st, q, o1, v1, v2, j, bov1, bvv2, bv12_j.get(), bar_bov1,
                                  bar_bvv2, bov_p_f64.get(), bar_bov_p_f64.get(), fp64_gemms);
      }
    }
    if (!cache_all_slabs && compute_forces) {
      pullback_ovvv_slab_device(blas, st, q, o1, v1, v1, i, bov1, bvv1, bv11_i.get(), bar_bov1,
                                bar_bvv1, bov_p_f64.get(), bar_bov_p_f64.get(), fp64_gemms);
      pullback_ovvv_slab_device(blas, st, q, o1, v1, v2, i, bov1, bvv2, bv12_i.get(), bar_bov1,
                                bar_bvv2, bov_p_f64.get(), bar_bov_p_f64.get(), fp64_gemms);
    }
  }

  double ch_e = 0.0;
  std::vector<double> ch_occ1(o1, 0.0), ch_occ2(o2, 0.0);
  int err = 0;
  d_total_e.copy_to_host(&ch_e, 1, st);
  d_occ1.copy_to_host(ch_occ1.data(), o1, st);
  d_occ2.copy_to_host(ch_occ2.data(), o2, st);
  d_err.copy_to_host(&err, 1, st);
  cuda_check(cudaStreamSynchronize(st), "mixed_spin finish sync");
  if (err) throw std::runtime_error("DF-UCCSD(T) v2 CUDA small mixed-spin triples denominator");
  total_e += ch_e;
  for (std::size_t idx = 0; idx < o1; ++idx) occ1[idx] += ch_occ1[idx];
  for (std::size_t idx = 0; idx < o2; ++idx) occ2[idx] += ch_occ2[idx];

  if (!compute_forces) return;

  // Scatter transposed intermediate pullbacks back to canonical arrays
  launch_permute4d<double, 3, 0, 1, 2>(o1, o2, v1, o2, bar_g21_jk.get(), bar_gooov21, st, true);
  launch_permute4d<double, 2, 0, 1, 3>(o2, o1, o1, v2, bar_g12_kj.get(), bar_gooov12, st, true);
  launch_permute3d<double, 1, 0, 2>(o2, o1, v12, bar_t2_12_k_m.get(), bar_t2_12, st, 1.0, true);
  bar_g21_jk.reset(); bar_g12_kj.reset(); bar_t2_12_k_m.reset();

  if (cache_all_slabs) {
    for (std::size_t p = 0; p < o1; ++p) {
      pullback_ovvv_slab_device(blas, st, q, o1, v1, v1, p, bov1, bvv1, bv11_all.get() + p * s11,
                                bar_bov1, bar_bvv1, bov_p_f64.get(), bar_bov_p_f64.get(),
                                fp64_gemms);
      pullback_ovvv_slab_device(blas, st, q, o1, v1, v2, p, bov1, bvv2, bv12_all.get() + p * s12,
                                bar_bov1, bar_bvv2, bov_p_f64.get(), bar_bov_p_f64.get(),
                                fp64_gemms);
    }
    for (std::size_t k = 0; k < o2; ++k) {
      pullback_ovvv_slab_device(blas, st, q, o2, v2, v1, k, bov2, bvv1, bv21_all.get() + k * s21,
                                bar_bov2, bar_bvv1, bov_p_f64.get(), bar_bov_p_f64.get(),
                                fp64_gemms);
    }
  }
  bv11_all.reset(); bv12_all.reset(); bv21_all.reset();
  bv11_i.reset(); bv11_j.reset(); bv12_i.reset(); bv12_j.reset(); bv21_k.reset();
  bar_W.reset(); bar_V.reset(); bar_R.reset(); bar_R_acb.reset(); X.reset(); Y.reset();

  // Occupied same-space Fock resolvent off-diagonals for s1 (bar_foo1) and s2 (bar_foo2)
  bool need_s1 = !has_weights, need_s2 = !has_weights;
  if (has_weights) {
    for (std::size_t p = 0; p < o1 && !need_s1; ++p)
      for (std::size_t r = p + 1; r < o1; ++r)
        if (std::abs(host_eps_o1[p] - host_eps_o1[r]) < 1e-6) need_s1 = true;
    for (std::size_t p = 0; p < o2 && !need_s2; ++p)
      for (std::size_t r = p + 1; r < o2; ++r)
        if (std::abs(host_eps_o2[p] - host_eps_o2[r]) < 1e-6) need_s2 = true;
  }
  if (need_s1 && o1 >= 3) {
    std::vector<double> host_foo1_off(o1 * o1, 0.0);
    for (std::size_t j = 0; j < o1; ++j) {
      if (!cache_all_slabs) {
        build_ovvv_slab_device<RealW>(dev, blas, st, q, o1, v1, v1, j, bov1_w, bvv1_w, v11_j.get(),
                                      bov_p_w.get(), w_gemm_counter);
        build_ovvv_slab_device<RealW>(dev, blas, st, q, o1, v1, v2, j, bov1_w, bvv2_w, v12_j.get(),
                                      bov_p_w.get(), w_gemm_counter);
      }
      for (std::size_t k = 0; k < o2; ++k) {
        std::vector<std::size_t> active;
        for (std::size_t p = 0; p < o1; ++p) {
          if (p == j) continue;
          if (!has_weights) {
            active.push_back(p);
          } else {
            for (std::size_t r = 0; r < o1; ++r) {
              if (r != p && r != j && std::abs(host_eps_o1[p] - host_eps_o1[r]) < 1e-6) {
                active.push_back(p);
                break;
              }
            }
          }
        }
        if (active.size() < 2) continue;
        if (!cache_all_slabs) {
          build_ovvv_slab_device<RealW>(dev, blas, st, q, o2, v2, v1, k, bov2_w, bvv1_w,
                                        v21_k.get(), bov_p_w.get(), w_gemm_counter);
        }
        const RealW* p11_j = cache_all_slabs ? (v11_all.get() + j * s11) : v11_j.get();
        const RealW* p12_j = cache_all_slabs ? (v12_all.get() + j * s12) : v12_j.get();
        const RealW* p21_k = cache_all_slabs ? (v21_all.get() + k * s21) : v21_k.get();
        accumulate_paged_foo_offdiag(
            dev, blas, st, o1, v_tot, active, host_eps_o1, has_weights, -0.25, device_cap,
            [&](std::size_t p, double* x_out, double* y_out) {
              const RealW* p11_p = cache_all_slabs ? (v11_all.get() + p * s11) : v11_i.get();
              const RealW* p12_p = cache_all_slabs ? (v12_all.get() + p * s12) : v12_i.get();
              if (!cache_all_slabs) {
                build_ovvv_slab_device<RealW>(dev, blas, st, q, o1, v1, v1, p, bov1_w, bvv1_w,
                                              v11_i.get(), bov_p_w.get(), w_gemm_counter);
                build_ovvv_slab_device<RealW>(dev, blas, st, q, o1, v1, v2, p, bov1_w, bvv2_w,
                                              v12_i.get(), bov_p_w.get(), w_gemm_counter);
              }
              build_w_mixed(p, j, k, p11_p, p11_j, p12_p, p12_j, p21_k);
              const double epjk = host_eps_o1[p] + host_eps_o1[j] + host_eps_o2[k];
              eval_mixed_spin_xy_only_kernel<RealW><<<grid_1d(v_tot), 256, 0, st>>>(
                  o1, o2, v1, v2, p, j, k, epjk, W.get(), t1_1, t1_2, fov1, fov2, goovv11, goovv12,
                  t2_11, t2_12, d_eps_v1, d_eps_v2, x_out, y_out);
              cuda_check(cudaGetLastError(), "mixed_spin s1 xy_only");
            },
            [&](std::size_t p) { return has_weights ? (w1[p] + w1[j] + w2[k]) / 3.0 : 1.0; },
            host_foo1_off, fp64_gemms);
      }
    }
    add_occupied_response(host_foo1_off, xy1, bar_foo1, st);
  }

  if (need_s2 && o2 >= 2) {
    std::vector<std::size_t> active;
    for (std::size_t p = 0; p < o2; ++p) {
      if (!has_weights) {
        active.push_back(p);
      } else {
        for (std::size_t r = 0; r < o2; ++r) {
          if (r != p && std::abs(host_eps_o2[p] - host_eps_o2[r]) < 1e-6) {
            active.push_back(p);
            break;
          }
        }
      }
    }
    if (active.size() >= 2) {
      std::vector<double> host_foo2_off(o2 * o2, 0.0);
      for (std::size_t i = 0; i < o1; ++i) {
        if (!cache_all_slabs) {
          build_ovvv_slab_device<RealW>(dev, blas, st, q, o1, v1, v1, i, bov1_w, bvv1_w,
                                        v11_i.get(), bov_p_w.get(), w_gemm_counter);
          build_ovvv_slab_device<RealW>(dev, blas, st, q, o1, v1, v2, i, bov1_w, bvv2_w,
                                        v12_i.get(), bov_p_w.get(), w_gemm_counter);
        }
        for (std::size_t j = i + 1; j < o1; ++j) {
          if (!cache_all_slabs) {
            build_ovvv_slab_device<RealW>(dev, blas, st, q, o1, v1, v1, j, bov1_w, bvv1_w,
                                          v11_j.get(), bov_p_w.get(), w_gemm_counter);
            build_ovvv_slab_device<RealW>(dev, blas, st, q, o1, v1, v2, j, bov1_w, bvv2_w,
                                          v12_j.get(), bov_p_w.get(), w_gemm_counter);
          }
          const RealW* p11_i = cache_all_slabs ? (v11_all.get() + i * s11) : v11_i.get();
          const RealW* p11_j = cache_all_slabs ? (v11_all.get() + j * s11) : v11_j.get();
          const RealW* p12_i = cache_all_slabs ? (v12_all.get() + i * s12) : v12_i.get();
          const RealW* p12_j = cache_all_slabs ? (v12_all.get() + j * s12) : v12_j.get();
          accumulate_paged_foo_offdiag(
              dev, blas, st, o2, v_tot, active, host_eps_o2, has_weights, -0.25, device_cap,
              [&](std::size_t p, double* x_out, double* y_out) {
                const RealW* p21_p = cache_all_slabs ? (v21_all.get() + p * s21) : v21_k.get();
                if (!cache_all_slabs) {
                  build_ovvv_slab_device<RealW>(dev, blas, st, q, o2, v2, v1, p, bov2_w, bvv1_w,
                                                v21_k.get(), bov_p_w.get(), w_gemm_counter);
                }
                build_w_mixed(i, j, p, p11_i, p11_j, p12_i, p12_j, p21_p);
                const double eijp = host_eps_o1[i] + host_eps_o1[j] + host_eps_o2[p];
                eval_mixed_spin_xy_only_kernel<RealW><<<grid_1d(v_tot), 256, 0, st>>>(
                    o1, o2, v1, v2, i, j, p, eijp, W.get(), t1_1, t1_2, fov1, fov2, goovv11,
                    goovv12, t2_11, t2_12, d_eps_v1, d_eps_v2, x_out, y_out);
                cuda_check(cudaGetLastError(), "mixed_spin s2 xy_only");
              },
              [&](std::size_t p) { return has_weights ? (w1[i] + w1[j] + w2[p]) / 3.0 : 1.0; },
              host_foo2_off, fp64_gemms);
        }
      }
      add_occupied_response(host_foo2_off, xy2, bar_foo2, st);
    }
  }
}

template <class RealW>
void dispatch_all_four_channels(const CudaSpinBlockedSource& src,
                                const DeviceFourIndexCore<double>& core,
                                const DeviceUccAmplitudes<double>& amp,
                                const DFUccV2Options& options, bool has_weights,
                                CudaTriplesResult& out) {
  const auto& d = src.dims;
  const int dev = src.device;
  const cudaStream_t st = src.stream;
  const cublasHandle_t blas = src.blas;
  const bool cf = options.compute_forces;

  // Channel 1: aaa
  run_same_spin_triples_cuda<RealW>(
      dev, blas, st, src.device_cap, d.oa, d.va, d.q, src.factors.bov_a.get(),
      src.factors.bvv_a.get(), core.goovv_aa.get(), core.gooov_aa.get(), src.fock.fov_a.get(),
      amp.t1a(), amp.t2aa(), src.eps_va.get(), src.host_eps_oa, options.occupied_weights_alpha,
      has_weights, cf, options.denominator_threshold, out.energy, out.occupied_alpha,
      cf ? out.bar_t.t1a() : nullptr, cf ? out.bar_t.t2aa() : nullptr,
      cf ? out.bar_b.bov_a.get() : nullptr, cf ? out.bar_b.bvv_a.get() : nullptr,
      cf ? out.bar_core.goovv_aa.get() : nullptr, cf ? out.bar_core.gooov_aa.get() : nullptr,
      cf ? out.bar_f.fov_a.get() : nullptr, cf ? out.bar_f.foo_a.get() : nullptr,
      cf ? out.bar_f.fvv_a.get() : nullptr, out.fp32_gemms, out.fp64_gemms);

  // Channel 2: bbb
  run_same_spin_triples_cuda<RealW>(
      dev, blas, st, src.device_cap, d.ob, d.vb, d.q, src.factors.bov_b.get(),
      src.factors.bvv_b.get(), core.goovv_bb.get(), core.gooov_bb.get(), src.fock.fov_b.get(),
      amp.t1b(), amp.t2bb(), src.eps_vb.get(), src.host_eps_ob, options.occupied_weights_beta,
      has_weights, cf, options.denominator_threshold, out.energy, out.occupied_beta,
      cf ? out.bar_t.t1b() : nullptr, cf ? out.bar_t.t2bb() : nullptr,
      cf ? out.bar_b.bov_b.get() : nullptr, cf ? out.bar_b.bvv_b.get() : nullptr,
      cf ? out.bar_core.goovv_bb.get() : nullptr, cf ? out.bar_core.gooov_bb.get() : nullptr,
      cf ? out.bar_f.fov_b.get() : nullptr, cf ? out.bar_f.foo_b.get() : nullptr,
      cf ? out.bar_f.fvv_b.get() : nullptr, out.fp32_gemms, out.fp64_gemms);

  // Channel 3: aab
  run_mixed_spin_triples_cuda<RealW>(
      dev, blas, st, src.device_cap, d.oa, d.ob, d.va, d.vb, d.q, src.factors.bov_a.get(),
      src.factors.bvv_a.get(), src.factors.bov_b.get(), src.factors.bvv_b.get(),
      core.goovv_aa.get(), core.goovv_ab.get(), core.gooov_aa.get(), core.gooov_ab.get(),
      core.gooov_ba.get(), src.fock.fov_a.get(), src.fock.fov_b.get(), amp.t1a(), amp.t1b(),
      amp.t2aa(), amp.t2ab(), src.eps_va.get(), src.eps_vb.get(), src.host_eps_oa, src.host_eps_ob,
      options.occupied_weights_alpha, options.occupied_weights_beta, has_weights, cf,
      options.denominator_threshold, out.energy, out.occupied_alpha, out.occupied_beta,
      cf ? out.bar_t.t1a() : nullptr, cf ? out.bar_t.t1b() : nullptr,
      cf ? out.bar_t.t2aa() : nullptr, cf ? out.bar_t.t2ab() : nullptr,
      cf ? out.bar_b.bov_a.get() : nullptr, cf ? out.bar_b.bvv_a.get() : nullptr,
      cf ? out.bar_b.bov_b.get() : nullptr, cf ? out.bar_b.bvv_b.get() : nullptr,
      cf ? out.bar_core.goovv_aa.get() : nullptr, cf ? out.bar_core.goovv_ab.get() : nullptr,
      cf ? out.bar_core.gooov_aa.get() : nullptr, cf ? out.bar_core.gooov_ab.get() : nullptr,
      cf ? out.bar_core.gooov_ba.get() : nullptr, cf ? out.bar_f.fov_a.get() : nullptr,
      cf ? out.bar_f.fov_b.get() : nullptr, cf ? out.bar_f.foo_a.get() : nullptr,
      cf ? out.bar_f.foo_b.get() : nullptr, cf ? out.bar_f.fvv_a.get() : nullptr,
      cf ? out.bar_f.fvv_b.get() : nullptr, out.fp32_gemms, out.fp64_gemms);

  // Channel 4: abb (exact mirror of aab via GPU-transposed t2ba and goovv_ba)
  if (d.ob >= 2 && d.oa >= 1 && d.vb >= 2 && d.va >= 1) {
    const std::size_t sz_ab = d.oa * d.ob * d.va * d.vb;
    DeviceBuf<double> t2ba(dev, sz_ab, st), goovv_ba(dev, sz_ab, st);
    launch_permute4d<double, 1, 0, 3, 2>(d.oa, d.ob, d.va, d.vb, amp.t2ab(), t2ba.get(), st);
    launch_permute4d<double, 1, 0, 3, 2>(d.oa, d.ob, d.va, d.vb, core.goovv_ab.get(),
                                         goovv_ba.get(), st);
    DeviceBuf<double> bar_t2ba(dev, cf ? sz_ab : 0, st);
    DeviceBuf<double> bar_goovv_ba(dev, cf ? sz_ab : 0, st);
    if (cf) {
      bar_t2ba.zero(st);
      bar_goovv_ba.zero(st);
    }
    run_mixed_spin_triples_cuda<RealW>(
        dev, blas, st, src.device_cap, d.ob, d.oa, d.vb, d.va, d.q, src.factors.bov_b.get(),
        src.factors.bvv_b.get(), src.factors.bov_a.get(), src.factors.bvv_a.get(),
        core.goovv_bb.get(), goovv_ba.get(), core.gooov_bb.get(), core.gooov_ba.get(),
        core.gooov_ab.get(), src.fock.fov_b.get(), src.fock.fov_a.get(), amp.t1b(), amp.t1a(),
        amp.t2bb(), t2ba.get(), src.eps_vb.get(), src.eps_va.get(), src.host_eps_ob,
        src.host_eps_oa, options.occupied_weights_beta, options.occupied_weights_alpha, has_weights,
        cf, options.denominator_threshold, out.energy, out.occupied_beta, out.occupied_alpha,
        cf ? out.bar_t.t1b() : nullptr, cf ? out.bar_t.t1a() : nullptr,
        cf ? out.bar_t.t2bb() : nullptr, cf ? bar_t2ba.get() : nullptr,
        cf ? out.bar_b.bov_b.get() : nullptr, cf ? out.bar_b.bvv_b.get() : nullptr,
        cf ? out.bar_b.bov_a.get() : nullptr, cf ? out.bar_b.bvv_a.get() : nullptr,
        cf ? out.bar_core.goovv_bb.get() : nullptr, cf ? bar_goovv_ba.get() : nullptr,
        cf ? out.bar_core.gooov_bb.get() : nullptr, cf ? out.bar_core.gooov_ba.get() : nullptr,
        cf ? out.bar_core.gooov_ab.get() : nullptr, cf ? out.bar_f.fov_b.get() : nullptr,
        cf ? out.bar_f.fov_a.get() : nullptr, cf ? out.bar_f.foo_b.get() : nullptr,
        cf ? out.bar_f.foo_a.get() : nullptr, cf ? out.bar_f.fvv_b.get() : nullptr,
        cf ? out.bar_f.fvv_a.get() : nullptr, out.fp32_gemms, out.fp64_gemms);
    if (cf) {
      launch_permute4d<double, 1, 0, 3, 2>(d.ob, d.oa, d.vb, d.va, bar_t2ba.get(), out.bar_t.t2ab(),
                                           st, true);
      launch_permute4d<double, 1, 0, 3, 2>(d.ob, d.oa, d.vb, d.va, bar_goovv_ba.get(),
                                           out.bar_core.goovv_ab.get(), st, true);
    }
  }
}

}  // namespace

CudaTriplesResult evaluate_df_ucc_triples_cuda_impl(const CudaSpinBlockedSource& src,
                                                    const DeviceFourIndexCore<double>& core,
                                                    const DeviceUccAmplitudes<double>& amp,
                                                    const DFUccV2Options& options) {
  const auto& d = src.dims;
  const bool has_weights =
      !options.occupied_weights_alpha.empty() || !options.occupied_weights_beta.empty();
  if (has_weights) {
    auto check_deg = [](const std::vector<double>& eps, const std::vector<double>& w) {
      for (std::size_t a = 0; a < eps.size(); ++a) {
        if (!std::isfinite(w[a]))
          throw std::invalid_argument("DF-UCCSD(T+) v2 CUDA nonfinite occupied weight");
        for (std::size_t b = a + 1; b < eps.size(); ++b) {
          if (std::abs(eps[a] - eps[b]) < 1e-6 &&
              std::abs(w[a] - w[b]) > 1e-8 * (1.0 + std::max(std::abs(w[a]), std::abs(w[b]))))
            throw std::invalid_argument(
                "DF-UCCSD(T+) v2 CUDA requires equal weights across degenerate occupied orbitals");
        }
      }
    };
    check_deg(src.host_eps_oa, options.occupied_weights_alpha);
    check_deg(src.host_eps_ob, options.occupied_weights_beta);
  }

  runtime::CudaDeviceScope dev_scope(src.device);
  CudaTriplesResult out;
  out.occupied_alpha.assign(d.oa, 0.0);
  out.occupied_beta.assign(d.ob, 0.0);
  if (options.compute_forces) {
    out.bar_t.allocate_zero(src.device, d, src.stream);
    out.bar_b.allocate_zero(src.device, d, src.stream);
    out.bar_core.allocate_zero(src.device, d, src.stream);
    out.bar_f.allocate_zero(src.device, d, src.stream);
  }

  if (options.triples_fp32_w) {
    dispatch_all_four_channels<float>(src, core, amp, options, has_weights, out);
  } else {
    dispatch_all_four_channels<double>(src, core, amp, options, has_weights, out);
  }

  validate_ucc_triples_partition(out.energy, out.occupied_alpha, out.occupied_beta, options);

  if (options.compute_forces) {
    launch_antisymmetrize_t2<double>(d.oa, d.va, out.bar_t.t2aa(), src.device, src.stream);
    launch_antisymmetrize_t2<double>(d.ob, d.vb, out.bar_t.t2bb(), src.device, src.stream);
  }
  return out;
}

}  // namespace generativeqc::cc::v2_detail::cuda_detail
