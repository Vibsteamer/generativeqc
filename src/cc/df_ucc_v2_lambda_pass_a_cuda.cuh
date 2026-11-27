#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "cc/df_ucc_v2_lambda_vjp_cuda.cuh"

namespace generativeqc::cc::v2_detail::cuda_detail::lambda_vjp_cuda {

// Intermediate adjoint tensors passed from Pass A to Pass B.
template <class Real>
struct DeviceLambdaIntermediates {
  DeviceBuf<Real> bar_Fvv[2], bar_Foo[2], bar_Fov[2], bar_Fvv_d[2], bar_Foo_d[2];
  DeviceBuf<Real> bar_tau[2], bar_tau_ab;
  DeviceBuf<Real> bar_x[2], bar_bt[2];
  DeviceBuf<Real> bar_W_ss[2], bar_W_abab, bar_W_baba, bar_W_abba, bar_W_baab;

  void allocate_zero(int dev, const SpinBlockedDims& d, cudaStream_t st) {
    const std::size_t os[2] = {d.oa, d.ob}, vs[2] = {d.va, d.vb};
    for (int s = 0; s < 2; ++s) {
      const std::size_t o = os[s], v = vs[s];
      bar_Fvv[s].allocate_zero(dev, v * v, st);
      bar_Foo[s].allocate_zero(dev, o * o, st);
      bar_Fov[s].allocate_zero(dev, o * v, st);
      bar_Fvv_d[s].allocate_zero(dev, v * v, st);
      bar_Foo_d[s].allocate_zero(dev, o * o, st);
      bar_tau[s].allocate_zero(dev, o * o * v * v, st);
      bar_x[s].allocate_zero(dev, d.q * o * v, st);
      bar_bt[s].allocate_zero(dev, d.q * v * v, st);
      bar_W_ss[s].allocate_zero(dev, o * v * o * v, st);
    }
    bar_tau_ab.allocate_zero(dev, d.oa * d.ob * d.va * d.vb, st);
    bar_W_abab.allocate_zero(dev, d.oa * d.va * d.ob * d.vb, st);
    bar_W_baba.allocate_zero(dev, d.ob * d.vb * d.oa * d.va, st);
    bar_W_abba.allocate_zero(dev, d.oa * d.vb * d.oa * d.vb, st);
    bar_W_baab.allocate_zero(dev, d.ob * d.va * d.ob * d.va, st);
  }
};

// --- Pass A helper kernels (zero atomics, fully deterministic) ---

template <class Real>
__global__ void f12_r1_wvo_cotangent_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                            std::size_t vb, const Real* l1a, const Real* l1b,
                                            Real* bw_aaaa, Real* bw_abab, Real* bw_bbbb,
                                            Real* bw_abba) {
  const std::size_t n0 = oa * oa * va;
  const std::size_t n1 = oa * ob * va;
  const std::size_t n2 = ob * ob * vb;
  const std::size_t n3 = ob * oa * vb;
  const std::size_t total = n0 + n1 + n2 + n3;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (idx < n0) {
      const std::size_t a = idx % va;
      const std::size_t r = idx / va;
      const std::size_t j = r % oa, i = r / oa;
      bw_aaaa[((i * oa + j) * va + a) * oa + j] += l1a[i * va + a];
    } else if (idx < n0 + n1) {
      const std::size_t k = idx - n0;
      const std::size_t a = k % va;
      const std::size_t r = k / va;
      const std::size_t j = r % ob, i = r / ob;
      bw_abab[((i * ob + j) * va + a) * ob + j] += l1a[i * va + a];
    } else if (idx < n0 + n1 + n2) {
      const std::size_t k = idx - n0 - n1;
      const std::size_t a = k % vb;
      const std::size_t r = k / vb;
      const std::size_t j = r % ob, i = r / ob;
      bw_bbbb[((i * ob + j) * vb + a) * ob + j] += l1b[i * vb + a];
    } else {
      const std::size_t k = idx - n0 - n1 - n2;
      const std::size_t a = k % vb;
      const std::size_t r = k / vb;
      const std::size_t j = r % oa, i = r / oa;
      bw_abba[((j * ob + i) * vb + a) * oa + j] -= l1b[i * vb + a];
    }
  }
}

template <class Real>
__global__ void build_l2_ab_asym_kernel(std::size_t o, std::size_t v, const Real* L2, Real* out) {
  const std::size_t total = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t a = r1 % v;
    const std::size_t ij = r1 / v;
    out[idx] = L2[(ij * v + a) * v + b] - L2[(ij * v + b) * v + a];
  }
}

template <class Real>
__global__ void build_l2_ij_asym_neg_kernel(std::size_t o, std::size_t v, const Real* L2,
                                            Real* out) {
  const std::size_t vv = v * v;
  const std::size_t total = o * o * vv;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t ab = idx % vv;
    const std::size_t ij = idx / vv;
    const std::size_t j = ij % o, i = ij / o;
    out[idx] = -L2[(i * o + j) * vv + ab] + L2[(j * o + i) * vv + ab];
  }
}

template <class Real>
__global__ void build_a2_ss_bars_kernel(std::size_t o, std::size_t v, const Real* L2, Real* bar_xb,
                                        Real* bar_tg, Real* bar_to) {
  const std::size_t total = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t a = r1 % v;
    const std::size_t r2 = r1 / v;
    const std::size_t j = r2 % o, i = r2 / o;
    const Real v00 = L2[((i * o + j) * v + a) * v + b];
    const Real v01 = L2[((i * o + j) * v + b) * v + a];
    const Real v10 = L2[((j * o + i) * v + a) * v + b];
    const Real v11 = L2[((j * o + i) * v + b) * v + a];
    const Real l_asym = v00 - v01 - v10 + v11;
    bar_xb[((i * v + a) * o + j) * v + b] = l_asym;
    bar_tg[((a * v + b) * o + j) * o + i] = l_asym;
    bar_to[idx] = -v00 + v01;
  }
}

template <class Real>
__global__ void build_a2_ab_bars_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                        std::size_t vb, const Real* L2ab, Real* bar_xb,
                                        Real* bar_to1, Real* bar_to2, Real* bar_tg1, Real* bar_tg3,
                                        Real* bar_tg4) {
  const std::size_t total = oa * ob * va * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % vb;
    const std::size_t r1 = idx / vb;
    const std::size_t a = r1 % va;
    const std::size_t r2 = r1 / va;
    const std::size_t j = r2 % ob, i = r2 / ob;
    const Real l2 = L2ab[idx];
    bar_xb[((i * va + a) * ob + j) * vb + b] = l2;
    bar_to1[idx] = -l2;
    bar_to2[((j * oa + i) * vb + b) * va + a] = -l2;
    bar_tg1[((a * ob + j) * vb + b) * oa + i] = -l2;
    bar_tg3[((a * vb + b) * oa + i) * ob + j] = -l2;
    bar_tg4[((b * va + a) * ob + j) * oa + i] = -l2;
  }
}

template <class Real>
__global__ void build_bar_ring_ss_kernel(std::size_t o, std::size_t v, const Real* L2,
                                         Real* bar_ring) {
  const std::size_t total = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t a = r1 % v;
    const std::size_t r2 = r1 / v;
    const std::size_t j = r2 % o, i = r2 / o;
    bar_ring[((i * v + a) * o + j) * v + b] =
        L2[((i * o + j) * v + a) * v + b] - L2[((j * o + i) * v + a) * v + b] -
        L2[((i * o + j) * v + b) * v + a] + L2[((j * o + i) * v + b) * v + a];
  }
}

template <class Real>
__global__ void antisym_bw_ss_kernel(std::size_t o, const Real* bw, Real* bw_asym) {
  const std::size_t oo = o * o;
  const std::size_t total = oo * oo;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t j = idx % o;
    const std::size_t r1 = idx / o;
    const std::size_t i = r1 % o;
    const std::size_t mn = r1 / o;
    bw_asym[idx] = bw[(mn * o + i) * o + j] - bw[(mn * o + j) * o + i];
  }
}

template <class Real>
__global__ void r1_ss_l1_fov_t2_kernel(std::size_t o, std::size_t v, const Real* L1,
                                       const Real* Fov, Real* bt2) {
  const std::size_t total = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t e = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t a = r1 % v;
    const std::size_t r2 = r1 / v;
    const std::size_t m = r2 % o, i = r2 / o;
    bt2[idx] += L1[i * v + a] * Fov[m * v + e];
  }
}

template <class Real>
__global__ void r1_ss_l1_t1_govov_kernel(std::size_t o, std::size_t v, const Real* L1,
                                         const Real* t1, Real* bc_ovov) {
  const std::size_t total = o * v * o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t e = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t i = r1 % o;
    const std::size_t r2 = r1 / o;
    const std::size_t a = r2 % v, m = r2 / v;
    bc_ovov[idx] -= L1[i * v + a] * t1[m * v + e];
  }
}

template <class Real>
__global__ void r1_cross_l1_fov_t2ab_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                            std::size_t vb, const Real* L1a, const Real* L1b,
                                            const Real* Fov_a, const Real* Fov_b, Real* bt2ab) {
  const std::size_t total = oa * ob * va * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t e = idx % vb;
    const std::size_t r1 = idx / vb;
    const std::size_t a = r1 % va;
    const std::size_t r2 = r1 / va;
    const std::size_t m = r2 % ob, i = r2 / ob;
    bt2ab[idx] += L1a[i * va + a] * Fov_b[m * vb + e] + L1b[m * vb + e] * Fov_a[i * va + a];
  }
}

template <class Real>
__global__ void r1_cross_l1_t1_goovv_ab_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                               std::size_t vb, const Real* L1a, const Real* L1b,
                                               const Real* t1a, const Real* t1b,
                                               Real* bc_goovv_ab) {
  const std::size_t total = oa * ob * va * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t e = idx % vb;
    const std::size_t r1 = idx / vb;
    const std::size_t a = r1 % va;
    const std::size_t r2 = r1 / va;
    const std::size_t m = r2 % ob, i = r2 / ob;
    bc_goovv_ab[idx] += L1a[i * va + a] * t1b[m * vb + e] + L1b[m * vb + e] * t1a[i * va + a];
  }
}

// =========================================================================
// PASS A: Residual outputs (r1, r2, optional f12) -> intermediate adjoints
// =========================================================================
template <class Real>
inline void evaluate_lambda_pass_a_cuda(
    const DeviceForwardCache<Real>& c, const DeviceSourceView<Real>& src,
    const DeviceFourIndexCore<Real>& core, const DeviceUccAmplitudes<Real>& amp,
    const DeviceFixedF12Coupling<Real>* f12, const DeviceUccAmplitudes<Real>& lam,
    DeviceUccAmplitudes<Real>& bar_t, DeviceLambdaIntermediates<Real>& inter,
    DeviceFockGrads* bar_f, DeviceFactorGrads* bar_b, DeviceFourIndexCore<double>* bar_core,
    DeviceFixedF12Cotangents* bar_f12, int dev, cublasHandle_t blas, cudaStream_t st,
    std::size_t ppl_tile_budget) {
  (void)bar_f;
  (void)bar_b;
  (void)bar_core;
  (void)bar_f12;
  const auto& d = src.dims;
  const std::size_t q = d.q, oa = d.oa, ob = d.ob, va = d.va, vb = d.vb;

  bar_t.zero(st);
  inter.allocate_zero(dev, d, st);

  // A1. Optional FixedF12Coupling residual terms
  if (f12) {
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_f12) {
        const std::size_t n_wvo = oa * oa * va + oa * ob * va + ob * ob * vb + ob * oa * vb;
        if (n_wvo) {
          f12_r1_wvo_cotangent_kernel<double><<<grid_1d(n_wvo), 256, 0, st>>>(
              oa, ob, va, vb, lam.t1a(), lam.t1b(), bar_f12->bar_w_vo_aaaa.get(),
              bar_f12->bar_w_vo_abab.get(), bar_f12->bar_w_vo_bbbb.get(),
              bar_f12->bar_w_vo_abba.get());
        }
        launch_add_scaled(bar_f12->bar_w_vv_aaaa.get(), lam.t2aa(), 1.0, oa * oa * va * va, st);
        launch_add_scaled(bar_f12->bar_c_f12_aaaa.get(), lam.t2aa(), 1.0, oa * oa * va * va, st);
        launch_add_scaled(bar_f12->bar_w_vv_bbbb.get(), lam.t2bb(), 1.0, ob * ob * vb * vb, st);
        launch_add_scaled(bar_f12->bar_c_f12_bbbb.get(), lam.t2bb(), 1.0, ob * ob * vb * vb, st);
        launch_add_scaled(bar_f12->bar_w_vv_abab.get(), lam.t2ab(), 1.0, oa * ob * va * vb, st);
        launch_add_scaled(bar_f12->bar_c_f12_abab.get(), lam.t2ab(), 1.0, oa * ob * va * vb, st);
      }
    }
    const Real* w_vo_ss[2] = {f12->w_vo_aaaa.get(), f12->w_vo_bbbb.get()};
    for (int s = 0; s < 2; ++s) {
      const std::size_t o = src.o(s), v = src.v(s);
      if (!o || !v) continue;
      DeviceBuf<Real> l2_asym(dev, o * o * v * v, st);
      build_l2_ab_asym_kernel<Real><<<grid_1d(l2_asym.size()), 256, 0, st>>>(
          o, v, lam.t2(s), l2_asym.get());
      rm_gemm<Real>(blas, st, 'T', 'N', o, v, o * o * v, w_vo_ss[s], l2_asym.get(), bar_t.t1(s),
                    Real(-1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_f12) {
          double* bw = (s == 0) ? bar_f12->bar_w_vo_aaaa.get() : bar_f12->bar_w_vo_bbbb.get();
          rm_gemm<double>(blas, st, 'N', 'T', o * o * v, o, v, l2_asym.get(), amp.t1(s), bw, -1.0,
                          1.0);
        }
      }
    }
    if (oa && ob && va && vb) {
      rm_gemm<Real>(blas, st, 'T', 'N', ob, vb, oa * ob * va, f12->w_vo_abab.get(), lam.t2ab(),
                    bar_t.t1b(), Real(-1), Real(1));
      DeviceBuf<Real> l2_ijba(dev, oa * ob * vb * va, st);
      launch_permute3d<Real, 0, 2, 1>(oa * ob, va, vb, lam.t2ab(), l2_ijba.get(), st);
      rm_gemm<Real>(blas, st, 'T', 'N', oa, va, oa * ob * vb, f12->w_vo_abba.get(), l2_ijba.get(),
                    bar_t.t1a(), Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_f12) {
          rm_gemm<double>(blas, st, 'N', 'T', oa * ob * va, ob, vb, lam.t2ab(), amp.t1b(),
                          bar_f12->bar_w_vo_abab.get(), -1.0, 1.0);
          rm_gemm<double>(blas, st, 'N', 'T', oa * ob * vb, oa, va, l2_ijba.get(), amp.t1a(),
                          bar_f12->bar_w_vo_abba.get(), 1.0, 1.0);
        }
      }
    }
    cuda_check(cudaGetLastError(), "lambda pass A1");
  }

  // A2. r2: bare t1*govvv (xb), t1*t1*govov (tg), -t1*gooov (to)
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if (!o || !v) continue;
    DeviceBuf<Real> bar_xb(dev, o * v * o * v, st), bar_tg(dev, v * v * o * o, st),
        bar_to(dev, o * o * v * v, st);
    build_a2_ss_bars_kernel<Real><<<grid_1d(o * o * v * v), 256, 0, st>>>(
        o, v, lam.t2(s), bar_xb.get(), bar_tg.get(), bar_to.get());
    rm_gemm<Real>(blas, st, 'N', 'T', q, o * v, o * v, src.bov[s], bar_xb.get(), inter.bar_x[s].get(),
                  Real(1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_b)
        rm_gemm<double>(blas, st, 'N', 'N', q, o * v, o * v, c.x_s[s].get(), bar_xb.get(),
                        bar_b->bov(s), 1.0, 1.0);
    }
    bar_xb.reset();

    rm_gemm<Real>(blas, st, 'N', 'T', o, v, v * o * o, c.u_tg_s[s].get(), bar_tg.get(), bar_t.t1(s),
                  Real(1), Real(1));
    DeviceBuf<Real> bar_u(dev, o * v * o * o, st);
    rm_gemm<Real>(blas, st, 'N', 'N', o, v * o * o, v, amp.t1(s), bar_tg.get(), bar_u.get(),
                  Real(1), Real(0));
    bar_tg.reset();
    rm_gemm<Real>(blas, st, 'T', 'N', o, v, o * v * o, bar_u.get(), core.govov(s), bar_t.t1(s),
                  Real(1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        rm_gemm<double>(blas, st, 'N', 'N', o * v * o, v, o, bar_u.get(), amp.t1(s),
                        bar_core->govov(s), 1.0, 1.0);
    }
    bar_u.reset();

    DeviceBuf<Real> g_p(dev, o * o * o * v, st), b_p(dev, v * o * o * v, st);
    launch_permute3d<Real, 1, 0, 2>(o * o, o, v, core.gooov(s), g_p.get(), st);
    launch_permute3d<Real, 1, 0, 2>(o * o, v, v, bar_to.get(), b_p.get(), st);
    rm_gemm<Real>(blas, st, 'N', 'T', o, v, o * o * v, g_p.get(), b_p.get(), bar_t.t1(s), Real(1),
                  Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        rm_gemm_batched<double>(blas, st, 'N', 'N', o, v, v, amp.t1(s), 0, bar_to.get(),
                                static_cast<long long>(v * v), bar_core->gooov(s),
                                static_cast<long long>(o * v), o * o, 1.0, 1.0);
    }
  }
  if (oa && ob && va && vb) {
    DeviceBuf<Real> bar_xb(dev, oa * va * ob * vb, st), bar_to1(dev, oa * ob * va * vb, st),
        bar_to2(dev, ob * oa * vb * va, st), bar_tg1(dev, va * ob * vb * oa, st),
        bar_tg3(dev, va * vb * oa * ob, st), bar_tg4(dev, vb * va * ob * oa, st);
    build_a2_ab_bars_kernel<Real><<<grid_1d(oa * ob * va * vb), 256, 0, st>>>(
        oa, ob, va, vb, lam.t2ab(), bar_xb.get(), bar_to1.get(), bar_to2.get(), bar_tg1.get(),
        bar_tg3.get(), bar_tg4.get());
    rm_gemm<Real>(blas, st, 'N', 'T', q, oa * va, ob * vb, src.bov[1], bar_xb.get(),
                  inter.bar_x[0].get(), Real(1), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', q, ob * vb, oa * va, src.bov[0], bar_xb.get(),
                  inter.bar_x[1].get(), Real(1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_b) {
        rm_gemm<double>(blas, st, 'N', 'N', q, ob * vb, oa * va, c.x_s[0].get(), bar_xb.get(),
                        bar_b->bov_b.get(), 1.0, 1.0);
        rm_gemm<double>(blas, st, 'N', 'T', q, oa * va, ob * vb, c.x_s[1].get(), bar_xb.get(),
                        bar_b->bov_a.get(), 1.0, 1.0);
      }
    }
    bar_xb.reset();

    {
      DeviceBuf<Real> g_p(dev, oa * oa * ob * vb, st), b_p(dev, va * oa * ob * vb, st);
      launch_permute3d<Real, 1, 0, 2>(oa * ob, oa, vb, core.gooov_ab.get(), g_p.get(), st);
      launch_permute3d<Real, 1, 0, 2>(oa * ob, va, vb, bar_to1.get(), b_p.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', oa, va, oa * ob * vb, g_p.get(), b_p.get(), bar_t.t1a(),
                    Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_core)
          rm_gemm_batched<double>(blas, st, 'N', 'N', oa, vb, va, amp.t1a(), 0, bar_to1.get(),
                                  static_cast<long long>(va * vb), bar_core->gooov_ab.get(),
                                  static_cast<long long>(oa * vb), oa * ob, 1.0, 1.0);
      }
    }
    bar_to1.reset();
    {
      DeviceBuf<Real> g_p(dev, ob * ob * oa * va, st), b_p(dev, vb * ob * oa * va, st);
      launch_permute3d<Real, 1, 0, 2>(ob * oa, ob, va, core.gooov_ba.get(), g_p.get(), st);
      launch_permute3d<Real, 1, 0, 2>(ob * oa, vb, va, bar_to2.get(), b_p.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', ob, vb, ob * oa * va, g_p.get(), b_p.get(), bar_t.t1b(),
                    Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_core)
          rm_gemm_batched<double>(blas, st, 'N', 'N', ob, va, vb, amp.t1b(), 0, bar_to2.get(),
                                  static_cast<long long>(vb * va), bar_core->gooov_ba.get(),
                                  static_cast<long long>(ob * va), ob * oa, 1.0, 1.0);
      }
    }
    bar_to2.reset();

    // -tg1
    rm_gemm<Real>(blas, st, 'N', 'T', oa, va, ob * vb * oa, c.t1_mjb_i.get(), bar_tg1.get(),
                  bar_t.t1a(), Real(1), Real(1));
    {
      DeviceBuf<Real> bar_mjb(dev, oa * ob * vb * oa, st), g_mjbe(dev, oa * ob * vb * va, st);
      rm_gemm<Real>(blas, st, 'N', 'N', oa, ob * vb * oa, va, amp.t1a(), bar_tg1.get(),
                    bar_mjb.get(), Real(1), Real(0));
      launch_permute3d<Real, 0, 2, 1>(oa * ob, va, vb, core.goovv_ab.get(), g_mjbe.get(), st);
      rm_gemm<Real>(blas, st, 'T', 'N', oa, va, oa * ob * vb, bar_mjb.get(), g_mjbe.get(),
                    bar_t.t1a(), Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_core) {
          rm_gemm<double>(blas, st, 'N', 'N', oa * ob * vb, va, oa, bar_mjb.get(), amp.t1a(),
                          g_mjbe.get(), 1.0, 0.0);
          launch_permute3d<double, 0, 2, 1>(oa * ob, vb, va, g_mjbe.get(), bar_core->goovv_ab.get(),
                                            st, 1.0, true);
        }
      }
    }
    bar_tg1.reset();

    // -tg2
    {
      DeviceBuf<Real> l_iajb(dev, oa * va * ob * vb, st), u2_m_iaj(dev, ob * oa * va * ob, st),
          bu2_m_iaj(dev, ob * oa * va * ob, st), bar_u2(dev, oa * ob * va * ob, st);
      launch_permute4d<Real, 0, 2, 1, 3>(oa, ob, va, vb, lam.t2ab(), l_iajb.get(), st);
      launch_permute4d<Real, 1, 0, 2, 3>(oa, ob, va, ob, c.u2_ab.get(), u2_m_iaj.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'N', ob, vb, oa * va * ob, u2_m_iaj.get(), l_iajb.get(),
                    bar_t.t1b(), Real(-1), Real(1));
      rm_gemm<Real>(blas, st, 'N', 'T', ob, oa * va * ob, vb, amp.t1b(), l_iajb.get(),
                    bu2_m_iaj.get(), Real(-1), Real(0));
      launch_permute4d<Real, 1, 0, 2, 3>(ob, oa, va, ob, bu2_m_iaj.get(), bar_u2.get(), st);
      rm_gemm<Real>(blas, st, 'T', 'N', ob, vb, oa * ob * va, bar_u2.get(), core.goovv_ab.get(),
                    bar_t.t1b(), Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_core)
          rm_gemm<double>(blas, st, 'N', 'N', oa * ob * va, vb, ob, bar_u2.get(), amp.t1b(),
                          bar_core->goovv_ab.get(), 1.0, 1.0);
      }
    }

    // -tg3
    rm_gemm<Real>(blas, st, 'N', 'T', oa, va, vb * oa * ob, c.u3_ab.get(), bar_tg3.get(),
                  bar_t.t1a(), Real(1), Real(1));
    {
      DeviceBuf<Real> bar_u3(dev, oa * vb * oa * ob, st);
      rm_gemm<Real>(blas, st, 'N', 'N', oa, vb * oa * ob, va, amp.t1a(), bar_tg3.get(),
                    bar_u3.get(), Real(1), Real(0));
      rm_gemm<Real>(blas, st, 'T', 'N', ob, vb, oa * vb * oa, bar_u3.get(), core.govov_ab.get(),
                    bar_t.t1b(), Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_core)
          rm_gemm<double>(blas, st, 'N', 'N', oa * vb * oa, vb, ob, bar_u3.get(), amp.t1b(),
                          bar_core->govov_ab.get(), 1.0, 1.0);
      }
    }
    bar_tg3.reset();

    // -tg4
    rm_gemm<Real>(blas, st, 'N', 'T', ob, vb, va * ob * oa, c.u4_ab.get(), bar_tg4.get(),
                  bar_t.t1b(), Real(1), Real(1));
    {
      DeviceBuf<Real> bar_u4(dev, ob * va * ob * oa, st);
      rm_gemm<Real>(blas, st, 'N', 'N', ob, va * ob * oa, vb, amp.t1b(), bar_tg4.get(),
                    bar_u4.get(), Real(1), Real(0));
      rm_gemm<Real>(blas, st, 'T', 'N', oa, va, ob * va * ob, bar_u4.get(), core.govov_ba.get(),
                    bar_t.t1a(), Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_core)
          rm_gemm<double>(blas, st, 'N', 'N', ob * va * ob, va, oa, bar_u4.get(), amp.t1a(),
                          bar_core->govov_ba.get(), 1.0, 1.0);
      }
    }
  }
  cuda_check(cudaGetLastError(), "lambda pass A2");

  // A3. Ring contractions: t2 * Wovvo -> r2 (keep t2_ia_me_s, tab_ia_me, tab_jb_me for A7)
  DeviceBuf<Real> t2_ia_me_s[2], tab_ia_me, tab_jb_me;
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if (!o || !v) continue;
    t2_ia_me_s[s].allocate(dev, o * v * o * v, st);
    launch_permute4d<Real, 0, 2, 1, 3>(o, o, v, v, amp.t2(s), t2_ia_me_s[s].get(), st);
  }
  if (oa && ob && va && vb) {
    tab_ia_me.allocate(dev, oa * va * ob * vb, st);
    tab_jb_me.allocate(dev, ob * vb * oa * va, st);
    launch_permute4d<Real, 0, 2, 1, 3>(oa, ob, va, vb, amp.t2ab(), tab_ia_me.get(), st);
    launch_permute4d<Real, 1, 3, 0, 2>(oa, ob, va, vb, amp.t2ab(), tab_jb_me.get(), st);
  }
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s), o2 = src.o(1 - s), v2 = src.v(1 - s);
    if (!o || !v) continue;
    DeviceBuf<Real> bar_ring(dev, o * v * o * v, st), bar_t2_me(dev, o * v * o * v, st);
    build_bar_ring_ss_kernel<Real><<<grid_1d(o * o * v * v), 256, 0, st>>>(
        o, v, lam.t2(s), bar_ring.get());
    rm_gemm<Real>(blas, st, 'N', 'T', o * v, o * v, o * v, bar_ring.get(), c.W_ss[s].get(),
                  bar_t2_me.get(), Real(1), Real(0));
    rm_gemm<Real>(blas, st, 'T', 'N', o * v, o * v, o * v, t2_ia_me_s[s].get(), bar_ring.get(),
                  inter.bar_W_ss[s].get(), Real(1), Real(1));
    launch_permute4d<Real, 0, 2, 1, 3>(o, v, o, v, bar_t2_me.get(), bar_t.t2(s), st, true);
    bar_t2_me.reset();
    if (o2 && v2) {
      const Real* W_cross = (s == 0) ? c.W_baba.get() : c.W_abab.get();
      Real* bar_W_cross = (s == 0) ? inter.bar_W_baba.get() : inter.bar_W_abab.get();
      const Real* tab_view = (s == 0) ? tab_ia_me.get() : tab_jb_me.get();
      DeviceBuf<Real> bar_tab(dev, o * v * o2 * v2, st);
      rm_gemm<Real>(blas, st, 'N', 'T', o * v, o2 * v2, o * v, bar_ring.get(), W_cross,
                    bar_tab.get(), Real(1), Real(0));
      rm_gemm<Real>(blas, st, 'T', 'N', o2 * v2, o * v, o * v, tab_view, bar_ring.get(),
                    bar_W_cross, Real(1), Real(1));
      if (s == 0) {
        launch_permute4d<Real, 0, 2, 1, 3>(oa, va, ob, vb, bar_tab.get(), bar_t.t2ab(), st, true);
      } else {
        launch_permute4d<Real, 2, 0, 3, 1>(ob, vb, oa, va, bar_tab.get(), bar_t.t2ab(), st, true);
      }
    }
  }
  if (oa && ob && va && vb) {
    {
      DeviceBuf<Real> br12(dev, oa * va * ob * vb, st), btaa(dev, oa * va * oa * va, st),
          btab1(dev, oa * va * ob * vb, st);
      launch_permute4d<Real, 0, 2, 1, 3>(oa, ob, va, vb, lam.t2ab(), br12.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', oa * va, oa * va, ob * vb, br12.get(), c.W_abab.get(),
                    btaa.get(), Real(1), Real(0));
      rm_gemm<Real>(blas, st, 'T', 'N', oa * va, ob * vb, oa * va, t2_ia_me_s[0].get(), br12.get(),
                    inter.bar_W_abab.get(), Real(1), Real(1));
      rm_gemm<Real>(blas, st, 'N', 'T', oa * va, ob * vb, ob * vb, br12.get(), c.W_ss[1].get(),
                    btab1.get(), Real(1), Real(0));
      rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, ob * vb, oa * va, tab_ia_me.get(), br12.get(),
                    inter.bar_W_ss[1].get(), Real(1), Real(1));
      launch_permute4d<Real, 0, 2, 1, 3>(oa, va, oa, va, btaa.get(), bar_t.t2aa(), st, true);
      launch_permute4d<Real, 0, 2, 1, 3>(oa, va, ob, vb, btab1.get(), bar_t.t2ab(), st, true);
    }
    {
      DeviceBuf<Real> br2(dev, ob * vb * oa * va, st), btbb(dev, ob * vb * ob * vb, st),
          btab2(dev, ob * vb * oa * va, st);
      launch_permute4d<Real, 1, 3, 0, 2>(oa, ob, va, vb, lam.t2ab(), br2.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', ob * vb, ob * vb, oa * va, br2.get(), c.W_baba.get(),
                    btbb.get(), Real(1), Real(0));
      rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, oa * va, ob * vb, t2_ia_me_s[1].get(), br2.get(),
                    inter.bar_W_baba.get(), Real(1), Real(1));
      rm_gemm<Real>(blas, st, 'N', 'T', ob * vb, oa * va, oa * va, br2.get(), c.W_ss[0].get(),
                    btab2.get(), Real(1), Real(0));
      rm_gemm<Real>(blas, st, 'T', 'N', oa * va, oa * va, ob * vb, tab_jb_me.get(), br2.get(),
                    inter.bar_W_ss[0].get(), Real(1), Real(1));
      launch_permute4d<Real, 0, 2, 1, 3>(ob, vb, ob, vb, btbb.get(), bar_t.t2bb(), st, true);
      launch_permute4d<Real, 2, 0, 3, 1>(ob, vb, oa, va, btab2.get(), bar_t.t2ab(), st, true);
    }
    {
      DeviceBuf<Real> br3(dev, ob * va * oa * vb, st), tab_ja_me(dev, ob * va * oa * vb, st),
          btab3(dev, ob * va * oa * vb, st);
      launch_permute4d<Real, 1, 2, 0, 3>(oa, ob, va, vb, lam.t2ab(), br3.get(), st);
      launch_permute4d<Real, 1, 2, 0, 3>(oa, ob, va, vb, amp.t2ab(), tab_ja_me.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', ob * va, oa * vb, oa * vb, br3.get(), c.W_abba.get(),
                    btab3.get(), Real(1), Real(0));
      rm_gemm<Real>(blas, st, 'T', 'N', oa * vb, oa * vb, ob * va, tab_ja_me.get(), br3.get(),
                    inter.bar_W_abba.get(), Real(1), Real(1));
      launch_permute4d<Real, 2, 0, 1, 3>(ob, va, oa, vb, btab3.get(), bar_t.t2ab(), st, true);
    }
    {
      DeviceBuf<Real> br4(dev, oa * vb * ob * va, st), tab_ib_me(dev, oa * vb * ob * va, st),
          btab4(dev, oa * vb * ob * va, st);
      launch_permute4d<Real, 0, 3, 1, 2>(oa, ob, va, vb, lam.t2ab(), br4.get(), st);
      launch_permute4d<Real, 0, 3, 1, 2>(oa, ob, va, vb, amp.t2ab(), tab_ib_me.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', oa * vb, ob * va, ob * va, br4.get(), c.W_baab.get(),
                    btab4.get(), Real(1), Real(0));
      rm_gemm<Real>(blas, st, 'T', 'N', ob * va, ob * va, oa * vb, tab_ib_me.get(), br4.get(),
                    inter.bar_W_baab.get(), Real(1), Real(1));
      launch_permute4d<Real, 0, 2, 3, 1>(oa, vb, ob, va, btab4.get(), bar_t.t2ab(), st, true);
    }
  }
  cuda_check(cudaGetLastError(), "lambda pass A3");

  // A4. On-the-fly T1-dressed DF Virtual Ladder adjoint (ZERO vvvv materialized!)
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if (o < 2 || !v) continue;
    const std::size_t np = o * (o - 1) / 2;
    DeviceBuf<Real> tau_lt(dev, np * v * v, st), l2_lt(dev, np * v * v, st),
        btau_lt(dev, np * v * v, st);
    btau_lt.zero(st);
    pack_occ_lt_kernel<Real><<<grid_1d(np * v * v), 256, 0, st>>>(o, v, c.tau_s[s].get(),
                                                                   tau_lt.get());
    pack_occ_lt_kernel<Real><<<grid_1d(np * v * v), 256, 0, st>>>(o, v, lam.t2(s), l2_lt.get());
    accumulate_df_ppl_vjp_tiled<Real>(dev, blas, st, q, np, v, v, c.bt_s[s].get(), c.bt_s[s].get(),
                                      tau_lt.get(), l2_lt.get(), inter.bar_bt[s].get(),
                                      inter.bar_bt[s].get(), btau_lt.get(), Real(2),
                                      ppl_tile_budget);
    unpack_occ_lt_add_kernel<Real><<<grid_1d(np * v * v), 256, 0, st>>>(o, v, btau_lt.get(),
                                                                        inter.bar_tau[s].get());
  }
  if (oa && ob && va && vb) {
    accumulate_df_ppl_vjp_tiled<Real>(dev, blas, st, q, oa * ob, va, vb, c.bt_s[0].get(),
                                      c.bt_s[1].get(), c.tau_ab.get(), lam.t2ab(),
                                      inter.bar_bt[0].get(), inter.bar_bt[1].get(),
                                      inter.bar_tau_ab.get(), Real(1), ppl_tile_budget);
  }
  cuda_check(cudaGetLastError(), "lambda pass A4");

  // A5. Woooo_bare + tau_oovv * t2 into r2
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if (!o || !v) continue;
    DeviceBuf<Real> bw(dev, o * o * o * o, st), bw_asym(dev, o * o * o * o, st);
    rm_gemm<Real>(blas, st, 'N', 'T', o * o, o * o, v * v, c.tau_s[s].get(), lam.t2(s), bw.get(),
                  Real(0.5), Real(0));
    rm_gemm<Real>(blas, st, 'N', 'N', o * o, v * v, o * o, c.w_bare_s[s].get(), lam.t2(s),
                  inter.bar_tau[s].get(), Real(0.5), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core) launch_add_scaled(bar_core->goooo(s), bw.get(), 1.0, o * o * o * o, st);
    }
    antisym_bw_ss_kernel<Real><<<grid_1d(o * o * o * o), 256, 0, st>>>(o, bw.get(), bw_asym.get());
    rm_gemm<Real>(blas, st, 'T', 'N', o, v, o * o * o, bw_asym.get(), core.gooov(s), bar_t.t1(s),
                  Real(1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        rm_gemm<double>(blas, st, 'N', 'N', o * o * o, v, o, bw_asym.get(), amp.t1(s),
                        bar_core->gooov(s), 1.0, 1.0);
    }
    bw_asym.reset();

    rm_gemm<Real>(blas, st, 'N', 'T', o * o, o * o, v * v, amp.t2(s), lam.t2(s), bw.get(),
                  Real(0.25), Real(0));
    rm_gemm<Real>(blas, st, 'N', 'N', o * o, v * v, o * o, c.tau_oovv_s[s].get(), lam.t2(s),
                  bar_t.t2(s), Real(0.25), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', o * o, v * v, o * o, bw.get(), core.goovv(s),
                  inter.bar_tau[s].get(), Real(1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        rm_gemm<double>(blas, st, 'N', 'N', o * o, v * v, o * o, bw.get(), c.tau_s[s].get(),
                        bar_core->goovv(s), 1.0, 1.0);
    }
  }
  if (oa && ob && va && vb) {
    DeviceBuf<Real> bw(dev, oa * ob * oa * ob, st), bw_nmji(dev, ob * oa * ob * oa, st);
    rm_gemm<Real>(blas, st, 'N', 'T', oa * ob, oa * ob, va * vb, c.tau_ab.get(), lam.t2ab(),
                  bw.get(), Real(1), Real(0));
    rm_gemm<Real>(blas, st, 'N', 'N', oa * ob, va * vb, oa * ob, c.w_bare_ab.get(), lam.t2ab(),
                  inter.bar_tau_ab.get(), Real(1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core) launch_add_scaled(bar_core->goooo_ab.get(), bw.get(), 1.0, bw.size(), st);
    }
    rm_gemm<Real>(blas, st, 'T', 'N', ob, vb, oa * ob * oa, bw.get(), core.gooov_ab.get(),
                  bar_t.t1b(), Real(1), Real(1));
    launch_permute4d<Real, 1, 0, 3, 2>(oa, ob, oa, ob, bw.get(), bw_nmji.get(), st);
    rm_gemm<Real>(blas, st, 'T', 'N', oa, va, ob * oa * ob, bw_nmji.get(), core.gooov_ba.get(),
                  bar_t.t1a(), Real(1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core) {
        rm_gemm<double>(blas, st, 'N', 'N', oa * ob * oa, vb, ob, bw.get(), amp.t1b(),
                        bar_core->gooov_ab.get(), 1.0, 1.0);
        rm_gemm<double>(blas, st, 'N', 'N', ob * oa * ob, va, oa, bw_nmji.get(), amp.t1a(),
                        bar_core->gooov_ba.get(), 1.0, 1.0);
      }
    }
    bw_nmji.reset();

    rm_gemm<Real>(blas, st, 'N', 'T', oa * ob, oa * ob, va * vb, amp.t2ab(), lam.t2ab(), bw.get(),
                  Real(1), Real(0));
    rm_gemm<Real>(blas, st, 'N', 'N', oa * ob, va * vb, oa * ob, c.g_ab.get(), lam.t2ab(),
                  bar_t.t2ab(), Real(1), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', oa * ob, va * vb, oa * ob, bw.get(), core.goovv_ab.get(),
                  inter.bar_tau_ab.get(), Real(1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        rm_gemm<double>(blas, st, 'N', 'N', oa * ob, va * vb, oa * ob, bw.get(), c.tau_ab.get(),
                        bar_core->goovv_ab.get(), 1.0, 1.0);
    }
  }
  cuda_check(cudaGetLastError(), "lambda pass A5");

  // A6. r2 base goovv + Fvv_d / Foo_d
  if constexpr (std::is_same_v<Real, double>) {
    if (bar_core) {
      launch_add_scaled(bar_core->goovv_aa.get(), lam.t2aa(), 1.0, oa * oa * va * va, st);
      launch_add_scaled(bar_core->goovv_bb.get(), lam.t2bb(), 1.0, ob * ob * vb * vb, st);
      launch_add_scaled(bar_core->goovv_ab.get(), lam.t2ab(), 1.0, oa * ob * va * vb, st);
    }
  }
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if (!o || !v) continue;
    DeviceBuf<Real> btmp(dev, o * o * v * v, st);
    build_l2_ab_asym_kernel<Real><<<grid_1d(btmp.size()), 256, 0, st>>>(o, v, lam.t2(s),
                                                                        btmp.get());
    rm_gemm<Real>(blas, st, 'N', 'N', o * o * v, v, v, btmp.get(), c.Fvv_d_s[s].get(), bar_t.t2(s),
                  Real(1), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', v, v, o * o * v, btmp.get(), amp.t2(s),
                  inter.bar_Fvv_d[s].get(), Real(1), Real(1));

    build_l2_ij_asym_neg_kernel<Real><<<grid_1d(btmp.size()), 256, 0, st>>>(o, v, lam.t2(s),
                                                                            btmp.get());
    DeviceBuf<Real> t_p(dev, o * o * v * v, st), b_p(dev, o * o * v * v, st);
    launch_permute3d<Real, 1, 0, 2>(o, o, v * v, amp.t2(s), t_p.get(), st);
    launch_permute3d<Real, 1, 0, 2>(o, o, v * v, btmp.get(), b_p.get(), st);
    rm_gemm<Real>(blas, st, 'N', 'T', o, o, o * v * v, t_p.get(), b_p.get(),
                  inter.bar_Foo_d[s].get(), Real(1), Real(1));
    rm_gemm_batched<Real>(blas, st, 'N', 'N', o, v * v, o, c.Foo_d_s[s].get(), 0, btmp.get(),
                          static_cast<long long>(o * v * v), bar_t.t2(s),
                          static_cast<long long>(o * v * v), o, Real(1), Real(1));
  }
  if (oa && ob && va && vb) {
    rm_gemm<Real>(blas, st, 'N', 'N', oa * ob * va, vb, vb, lam.t2ab(), c.Fvv_d_s[1].get(),
                  bar_t.t2ab(), Real(1), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', vb, vb, oa * ob * va, lam.t2ab(), amp.t2ab(),
                  inter.bar_Fvv_d[1].get(), Real(1), Real(1));
    {
      DeviceBuf<Real> l_p(dev, va * oa * ob * vb, st), t_p(dev, va * oa * ob * vb, st);
      launch_permute3d<Real, 1, 0, 2>(oa * ob, va, vb, lam.t2ab(), l_p.get(), st);
      launch_permute3d<Real, 1, 0, 2>(oa * ob, va, vb, amp.t2ab(), t_p.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', va, va, oa * ob * vb, l_p.get(), t_p.get(),
                    inter.bar_Fvv_d[0].get(), Real(1), Real(1));
      rm_gemm_batched<Real>(blas, st, 'T', 'N', va, vb, va, c.Fvv_d_s[0].get(), 0, lam.t2ab(),
                            static_cast<long long>(va * vb), bar_t.t2ab(),
                            static_cast<long long>(va * vb), oa * ob, Real(1), Real(1));
    }
    {
      DeviceBuf<Real> t_p(dev, ob * oa * va * vb, st), l_p(dev, ob * oa * va * vb, st);
      launch_permute3d<Real, 1, 0, 2>(oa, ob, va * vb, amp.t2ab(), t_p.get(), st);
      launch_permute3d<Real, 1, 0, 2>(oa, ob, va * vb, lam.t2ab(), l_p.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', ob, ob, oa * va * vb, t_p.get(), l_p.get(),
                    inter.bar_Foo_d[1].get(), Real(-1), Real(1));
      rm_gemm_batched<Real>(blas, st, 'N', 'N', ob, va * vb, ob, c.Foo_d_s[1].get(), 0, lam.t2ab(),
                            static_cast<long long>(ob * va * vb), bar_t.t2ab(),
                            static_cast<long long>(ob * va * vb), oa, Real(-1), Real(1));
    }
    rm_gemm<Real>(blas, st, 'N', 'T', oa, oa, ob * va * vb, amp.t2ab(), lam.t2ab(),
                  inter.bar_Foo_d[0].get(), Real(-1), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', oa, ob * va * vb, oa, c.Foo_d_s[0].get(), lam.t2ab(),
                  bar_t.t2ab(), Real(-1), Real(1));
  }
  cuda_check(cudaGetLastError(), "lambda pass A6");

  // A7. Singles residuals r1a, r1b
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s), o2 = src.o(1 - s), v2 = src.v(1 - s);
    if (!o || !v) continue;
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_f) launch_add_scaled(bar_f->fov(s), lam.t1(s), 1.0, o * v, st);
    }
    rm_gemm<Real>(blas, st, 'N', 'N', o, v, v, lam.t1(s), c.Fvv_s[s].get(), bar_t.t1(s), Real(1),
                  Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', v, v, o, lam.t1(s), amp.t1(s), inter.bar_Fvv[s].get(),
                  Real(1), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'T', o, o, v, amp.t1(s), lam.t1(s), inter.bar_Foo[s].get(),
                  Real(-1), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', o, v, o, c.Foo_s[s].get(), lam.t1(s), bar_t.t1(s), Real(-1),
                  Real(1));

    r1_ss_l1_fov_t2_kernel<Real><<<grid_1d(o * o * v * v), 256, 0, st>>>(
        o, v, lam.t1(s), c.Fov_s[s].get(), bar_t.t2(s));
    rm_gemm<Real>(blas, st, 'T', 'N', o * v, 1, o * v, t2_ia_me_s[s].get(), lam.t1(s),
                  inter.bar_Fov[s].get(), Real(1), Real(1));
    {
      DeviceBuf<Real> g_meia(dev, o * v * o * v, st);
      launch_permute4d<Real, 0, 3, 2, 1>(o, v, o, v, core.govov(s), g_meia.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'N', o * v, 1, o * v, g_meia.get(), lam.t1(s), bar_t.t1(s),
                    Real(-1), Real(1));
    }
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        r1_ss_l1_t1_govov_kernel<double><<<grid_1d(o * v * o * v), 256, 0, st>>>(
            o, v, lam.t1(s), amp.t1(s), bar_core->govov(s));
    }

    // DF govvv -> r1
    {
      DeviceBuf<Real> bar_z(dev, q * o * v, st);
      rm_gemm_batched<Real>(blas, st, 'N', 'N', o, v, v, lam.t1(s), 0, src.bvv[s],
                            static_cast<long long>(v * v), bar_z.get(),
                            static_cast<long long>(o * v), q, Real(1), Real(0));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_b)
          rm_gemm_batched<double>(blas, st, 'T', 'N', v, v, o, lam.t1(s), 0, c.z_s[s].get(),
                                  static_cast<long long>(o * v), bar_b->bvv(s),
                                  static_cast<long long>(v * v), q, 1.0, 1.0);
      }
      {
        DeviceBuf<Real> bt2_ifme(dev, o * v * o * v, st);
        rm_gemm<Real>(blas, st, 'T', 'N', o * v, o * v, q, bar_z.get(), src.bov[s], bt2_ifme.get(),
                      Real(-1), Real(0));
        launch_permute4d<Real, 0, 2, 3, 1>(o, v, o, v, bt2_ifme.get(), bar_t.t2(s), st, true);
        if constexpr (std::is_same_v<Real, double>) {
          if (bar_b) {
            launch_permute4d<double, 0, 3, 1, 2>(o, o, v, v, amp.t2(s), bt2_ifme.get(), st);
            rm_gemm<double>(blas, st, 'N', 'N', q, o * v, o * v, bar_z.get(), bt2_ifme.get(),
                            bar_b->bov(s), -1.0, 1.0);
          }
        }
      }
      if (o2 && v2) {
        DeviceBuf<Real> btab(dev, o * v * o2 * v2, st);
        rm_gemm<Real>(blas, st, 'T', 'N', o * v, o2 * v2, q, bar_z.get(), src.bov[1 - s],
                      btab.get(), Real(1), Real(0));
        if (s == 0) {
          launch_permute4d<Real, 0, 2, 1, 3>(oa, va, ob, vb, btab.get(), bar_t.t2ab(), st, true);
        } else {
          launch_permute4d<Real, 2, 0, 3, 1>(ob, vb, oa, va, btab.get(), bar_t.t2ab(), st, true);
        }
        if constexpr (std::is_same_v<Real, double>) {
          if (bar_b) {
            const double* tab_view = (s == 0) ? tab_ia_me.get() : tab_jb_me.get();
            rm_gemm<double>(blas, st, 'N', 'N', q, o2 * v2, o * v, bar_z.get(), tab_view,
                            bar_b->bov(1 - s), 1.0, 1.0);
          }
        }
      }
    }

    // -0.5 * gooov_ss * t2_ss -> r1
    rm_gemm_batched<Real>(blas, st, 'T', 'N', v, v, o, lam.t1(s), 0, core.gooov(s),
                          static_cast<long long>(o * v), bar_t.t2(s),
                          static_cast<long long>(v * v), o * o, Real(-0.5), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        rm_gemm_batched<double>(blas, st, 'N', 'N', o, v, v, lam.t1(s), 0, amp.t2(s),
                                static_cast<long long>(v * v), bar_core->gooov(s),
                                static_cast<long long>(o * v), o * o, -0.5, 1.0);
    }
  }
  if (oa && ob && va && vb) {
    r1_cross_l1_fov_t2ab_kernel<Real><<<grid_1d(oa * ob * va * vb), 256, 0, st>>>(
        oa, ob, va, vb, lam.t1a(), lam.t1b(), c.Fov_s[0].get(), c.Fov_s[1].get(), bar_t.t2ab());
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, 1, oa * va, tab_ia_me.get(), lam.t1a(),
                  inter.bar_Fov[1].get(), Real(1), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, 1, ob * vb, tab_ia_me.get(), lam.t1b(),
                  inter.bar_Fov[0].get(), Real(1), Real(1));
    {
      DeviceBuf<Real> gab_menf(dev, oa * va * ob * vb, st);
      launch_permute4d<Real, 0, 2, 1, 3>(oa, ob, va, vb, core.goovv_ab.get(), gab_menf.get(), st);
      rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, 1, oa * va, gab_menf.get(), lam.t1a(),
                    bar_t.t1b(), Real(1), Real(1));
      rm_gemm<Real>(blas, st, 'N', 'N', oa * va, 1, ob * vb, gab_menf.get(), lam.t1b(),
                    bar_t.t1a(), Real(1), Real(1));
    }
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        r1_cross_l1_t1_goovv_ab_kernel<double><<<grid_1d(oa * ob * va * vb), 256, 0, st>>>(
            oa, ob, va, vb, lam.t1a(), lam.t1b(), amp.t1a(), amp.t1b(), bar_core->goovv_ab.get());
    }
    rm_gemm_batched<Real>(blas, st, 'T', 'N', va, vb, oa, lam.t1a(), 0, core.gooov_ab.get(),
                          static_cast<long long>(oa * vb), bar_t.t2ab(),
                          static_cast<long long>(va * vb), oa * ob, Real(-1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        rm_gemm_batched<double>(blas, st, 'N', 'N', oa, vb, va, lam.t1a(), 0, amp.t2ab(),
                                static_cast<long long>(va * vb), bar_core->gooov_ab.get(),
                                static_cast<long long>(oa * vb), oa * ob, -1.0, 1.0);
    }
    {
      DeviceBuf<Real> gba_nm(dev, oa * ob * ob * va, st);
      launch_permute3d<Real, 1, 0, 2>(ob, oa, ob * va, core.gooov_ba.get(), gba_nm.get(), st);
      rm_gemm_batched<Real>(blas, st, 'T', 'N', va, vb, ob, gba_nm.get(),
                            static_cast<long long>(ob * va), lam.t1b(), 0, bar_t.t2ab(),
                            static_cast<long long>(va * vb), oa * ob, Real(-1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_core) {
          DeviceBuf<double> t2ab_mn(dev, ob * oa * va * vb, st);
          launch_permute3d<double, 1, 0, 2>(oa, ob, va * vb, amp.t2ab(), t2ab_mn.get(), st);
          rm_gemm_batched<double>(blas, st, 'N', 'T', ob, va, vb, lam.t1b(), 0, t2ab_mn.get(),
                                  static_cast<long long>(va * vb), bar_core->gooov_ba.get(),
                                  static_cast<long long>(ob * va), ob * oa, -1.0, 1.0);
        }
      }
    }
  }
  cuda_check(cudaGetLastError(), "lambda pass A7");
}

}  // namespace generativeqc::cc::v2_detail::cuda_detail::lambda_vjp_cuda
