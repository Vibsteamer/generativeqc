#pragma once

#include <cstddef>
#include <type_traits>

#include "cc/df_ucc_v2_lambda_pass_a_cuda.cuh"

namespace generativeqc::cc::v2_detail::cuda_detail::lambda_vjp_cuda {

// --- Pass B helper kernels (zero atomics, fully deterministic) ---

template <class Real>
__global__ void wovvo_bar_to_by_bvv_kernel(std::size_t o, std::size_t v, const Real* bar_W,
                                           Real* by_bvv) {
  const std::size_t total = o * v * o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t j = r1 % o;
    const std::size_t r2 = r1 / o;
    const std::size_t e = r2 % v, m = r2 / v;
    by_bvv[((m * o + j) * v + b) * v + e] = -bar_W[idx];
  }
}

template <class Real>
__global__ void wovvo_bar_sub_govov_kernel(std::size_t o, std::size_t v, const Real* bar_W,
                                           Real* bc_ovov) {
  const std::size_t total = o * v * o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t j = r1 % o;
    const std::size_t r2 = r1 / o;
    const std::size_t e = r2 % v, m = r2 / v;
    bc_ovov[((m * v + b) * o + j) * v + e] -= bar_W[idx];
  }
}

template <class Real>
__global__ void scatter_half_btab_x_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                           std::size_t vb, const Real* btab_x, Real* bt2ab) {
  const std::size_t total = ob * va * oa * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % vb;
    const std::size_t r1 = idx / vb;
    const std::size_t j = r1 % oa;
    const std::size_t r2 = r1 / oa;
    const std::size_t f = r2 % va, n = r2 / va;
    bt2ab[((j * ob + n) * va + f) * vb + b] += Real(0.5) * btab_x[idx];
  }
}

template <class Real>
__global__ void scatter_half_btba_x_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                           std::size_t vb, const Real* btba_x, Real* bt2ab) {
  const std::size_t total = oa * vb * ob * va;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % va;
    const std::size_t r1 = idx / va;
    const std::size_t j = r1 % ob;
    const std::size_t r2 = r1 / ob;
    const std::size_t f = r2 % vb, n = r2 / vb;
    bt2ab[((n * ob + j) * va + b) * vb + f] += Real(0.5) * btba_x[idx];
  }
}

template <class Real>
__global__ void scatter_half_bt_nfjb_ss_kernel(std::size_t o, std::size_t v, const Real* bt_nfjb,
                                               Real* bt2) {
  const std::size_t total = o * v * o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t j = r1 % o;
    const std::size_t r2 = r1 / o;
    const std::size_t f = r2 % v, n = r2 / v;
    bt2[((j * o + n) * v + f) * v + b] += Real(0.5) * bt_nfjb[idx];
  }
}

template <class Real>
__global__ void pullback_fov_goovv_ss_kernel(std::size_t o, std::size_t v, const Real* bar_Fov,
                                             const Real* t1, Real* bc_oovv) {
  const std::size_t total = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t f = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t e = r1 % v;
    const std::size_t r2 = r1 / v;
    const std::size_t n = r2 % o, m = r2 / o;
    bc_oovv[idx] += bar_Fov[m * v + e] * t1[n * v + f];
  }
}

template <class Real>
__global__ void pullback_fov_goovv_ab_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                             std::size_t vb, const Real* bar_Fov_a,
                                             const Real* bar_Fov_b, const Real* t1a,
                                             const Real* t1b, Real* bc_oovv_ab) {
  const std::size_t total = oa * ob * va * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t f = idx % vb;
    const std::size_t r1 = idx / vb;
    const std::size_t e = r1 % va;
    const std::size_t r2 = r1 / va;
    const std::size_t n = r2 % ob, m = r2 / ob;
    bc_oovv_ab[idx] +=
        bar_Fov_a[m * va + e] * t1b[n * vb + f] + bar_Fov_b[n * vb + f] * t1a[m * va + e];
  }
}

template <class Real>
__global__ void pullback_foo_gooov_t1_kernel(std::size_t o1, std::size_t o2, std::size_t v,
                                             const Real* bar_Foo, const Real* gooov, Real* bt1) {
  const std::size_t total = o2 * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t n = idx / v, e = idx % v;
    Real acc = Real(0);
    for (std::size_t m = 0; m < o1; ++m) {
      for (std::size_t i = 0; i < o1; ++i) {
        acc += bar_Foo[m * o1 + i] * gooov[((m * o2 + n) * o1 + i) * v + e];
      }
    }
    bt1[idx] += acc;
  }
}

template <class Real>
__global__ void pullback_foo_gooov_core_kernel(std::size_t o1, std::size_t o2, std::size_t v,
                                               const Real* bar_Foo, const Real* t1,
                                               Real* bc_ooov) {
  const std::size_t total = o1 * o2 * o1 * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t e = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t i = r1 % o1;
    const std::size_t r2 = r1 / o1;
    const std::size_t n = r2 % o2, m = r2 / o2;
    bc_ooov[idx] += bar_Foo[m * o1 + i] * t1[n * v + e];
  }
}

template <class Real>
__global__ void pullback_tau_ss_t1_kernel(std::size_t o, std::size_t v, const Real* btau,
                                          const Real* t1, Real* bt1) {
  const std::size_t total = o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t i = idx / v, a = idx % v;
    Real acc = Real(0);
    for (std::size_t j = 0; j < o; ++j) {
      for (std::size_t b = 0; b < v; ++b) {
        acc += (btau[((i * o + j) * v + a) * v + b] + btau[((j * o + i) * v + b) * v + a] -
                btau[((i * o + j) * v + b) * v + a] - btau[((j * o + i) * v + a) * v + b]) *
               t1[j * v + b];
      }
    }
    bt1[idx] += acc;
  }
}

// =========================================================================
// PASS B: Pull intermediate adjoints back to bar_t and (if requested) params
// =========================================================================
template <class Real>
inline void evaluate_lambda_pass_b_cuda(
    const DeviceForwardCache<Real>& c, const DeviceSourceView<Real>& src,
    const DeviceFourIndexCore<Real>& core, const DeviceUccAmplitudes<Real>& amp,
    DeviceLambdaIntermediates<Real>& inter, DeviceUccAmplitudes<Real>& bar_t,
    DeviceFockGrads* bar_f, DeviceFactorGrads* bar_b, DeviceFourIndexCore<double>* bar_core,
    int dev, cublasHandle_t blas, cudaStream_t st) {
  (void)bar_f;
  (void)bar_b;
  (void)bar_core;
  const auto& d = src.dims;
  const std::size_t q = d.q, oa = d.oa, ob = d.ob, va = d.va, vb = d.vb;

  // B1. Pull back Wovvo (all 6 blocks)
  DeviceBuf<Real> bar_y[2], g_menf_s[2], bt_nfjb_s[2];
  DeviceBuf<double> t_nfjb_s[2], tab_ia_me, tab_jb_me;
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    bar_y[s].allocate_zero(dev, q * o * o, st);
    if (!o || !v) continue;
    g_menf_s[s].allocate(dev, o * v * o * v, st);
    bt_nfjb_s[s].allocate_zero(dev, o * v * o * v, st);
    launch_permute4d<Real, 0, 2, 1, 3>(o, o, v, v, core.goovv(s), g_menf_s[s].get(), st);
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core) {
        t_nfjb_s[s].allocate(dev, o * v * o * v, st);
        build_t_nfjb_ss_kernel<double><<<grid_1d(t_nfjb_s[s].size()), 256, 0, st>>>(
            o, v, amp.t2(s), amp.t1(s), t_nfjb_s[s].get());
      }
    }
  }
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if (!o || !v) continue;
    rm_gemm<Real>(blas, st, 'N', 'N', q, o * v, o * v, src.bov[s], inter.bar_W_ss[s].get(),
                  inter.bar_x[s].get(), Real(1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_b)
        rm_gemm<double>(blas, st, 'N', 'T', q, o * v, o * v, c.x_s[s].get(),
                        inter.bar_W_ss[s].get(), bar_b->bov(s), 1.0, 1.0);
    }
    {
      DeviceBuf<Real> by_bvv(dev, o * o * v * v, st);
      wovvo_bar_to_by_bvv_kernel<Real><<<grid_1d(o * v * o * v), 256, 0, st>>>(
          o, v, inter.bar_W_ss[s].get(), by_bvv.get());
      rm_gemm<Real>(blas, st, 'N', 'T', q, o * o, v * v, src.bvv[s], by_bvv.get(), bar_y[s].get(),
                    Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_b)
          rm_gemm<double>(blas, st, 'N', 'N', q, v * v, o * o, c.y_s[s].get(), by_bvv.get(),
                          bar_b->bvv(s), 1.0, 1.0);
        if (bar_core)
          wovvo_bar_sub_govov_kernel<double><<<grid_1d(o * v * o * v), 256, 0, st>>>(
              o, v, inter.bar_W_ss[s].get(), bar_core->govov(s));
      }
    }
    {
      DeviceBuf<Real> bw_mje_b(dev, o * o * v * v, st), g_n_mje(dev, o * o * o * v, st);
      launch_permute4d<Real, 0, 2, 1, 3>(o, v, o, v, inter.bar_W_ss[s].get(), bw_mje_b.get(), st);
      launch_permute4d<Real, 1, 0, 2, 3>(o, o, o, v, core.gooov(s), g_n_mje.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'N', o, v, o * o * v, g_n_mje.get(), bw_mje_b.get(), bar_t.t1(s),
                    Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_core) {
          rm_gemm<double>(blas, st, 'N', 'T', o, o * o * v, v, amp.t1(s), bw_mje_b.get(),
                          g_n_mje.get(), 1.0, 0.0);
          launch_permute4d<double, 1, 0, 2, 3>(o, o, o, v, g_n_mje.get(), bar_core->gooov(s), st,
                                               true);
        }
      }
    }
    rm_gemm<Real>(blas, st, 'T', 'N', o * v, o * v, o * v, g_menf_s[s].get(),
                  inter.bar_W_ss[s].get(), bt_nfjb_s[s].get(), Real(-1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core) {
        DeviceBuf<double> bg(dev, o * v * o * v, st);
        rm_gemm<double>(blas, st, 'N', 'T', o * v, o * v, o * v, inter.bar_W_ss[s].get(),
                        t_nfjb_s[s].get(), bg.get(), -1.0, 0.0);
        launch_permute4d<double, 0, 2, 1, 3>(o, v, o, v, bg.get(), bar_core->goovv(s), st, true);
      }
    }
  }

  DeviceBuf<Real> gab_menf;
  if (oa && ob && va && vb) {
    gab_menf.allocate(dev, oa * va * ob * vb, st);
    launch_permute4d<Real, 0, 2, 1, 3>(oa, ob, va, vb, core.goovv_ab.get(), gab_menf.get(), st);
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core) {
        tab_ia_me.allocate(dev, oa * va * ob * vb, st);
        tab_jb_me.allocate(dev, ob * vb * oa * va, st);
        launch_permute4d<double, 0, 2, 1, 3>(oa, ob, va, vb, amp.t2ab(), tab_ia_me.get(), st);
        launch_permute4d<double, 1, 3, 0, 2>(oa, ob, va, vb, amp.t2ab(), tab_jb_me.get(), st);
      }
    }

    // Cross terms in W_aaaa and W_bbbb
    DeviceBuf<Real> btab_nfb_jaa(dev, ob * vb * oa * va, st),
        btab_nfa_jbb(dev, oa * va * ob * vb, st);
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, oa * va, oa * va, gab_menf.get(),
                  inter.bar_W_ss[0].get(), btab_nfb_jaa.get(), Real(0.5), Real(0));
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, ob * vb, ob * vb, gab_menf.get(),
                  inter.bar_W_ss[1].get(), btab_nfa_jbb.get(), Real(0.5), Real(0));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core) {
        DeviceBuf<double> bgab(dev, oa * va * ob * vb, st);
        rm_gemm<double>(blas, st, 'N', 'T', oa * va, ob * vb, oa * va, inter.bar_W_ss[0].get(),
                        tab_jb_me.get(), bgab.get(), 0.5, 0.0);
        rm_gemm<double>(blas, st, 'N', 'T', oa * va, ob * vb, ob * vb, tab_ia_me.get(),
                        inter.bar_W_ss[1].get(), bgab.get(), 0.5, 1.0);
        launch_permute4d<double, 0, 2, 1, 3>(oa, va, ob, vb, bgab.get(), bar_core->goovv_ab.get(),
                                             st, true);
      }
    }

    // W_abab
    rm_gemm<Real>(blas, st, 'N', 'N', q, ob * vb, oa * va, src.bov[0], inter.bar_W_abab.get(),
                  inter.bar_x[1].get(), Real(1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_b)
        rm_gemm<double>(blas, st, 'N', 'T', q, oa * va, ob * vb, c.x_s[1].get(),
                        inter.bar_W_abab.get(), bar_b->bov_a.get(), 1.0, 1.0);
      if (bar_core)
        launch_permute4d<double, 0, 2, 1, 3>(oa, va, ob, vb, inter.bar_W_abab.get(),
                                             bar_core->goovv_ab.get(), st, true);
    }
    {
      DeviceBuf<Real> bw_mje_b(dev, oa * ob * va * vb, st);
      launch_permute4d<Real, 0, 2, 1, 3>(oa, va, ob, vb, inter.bar_W_abab.get(), bw_mje_b.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'N', ob, vb, oa * ob * va, core.gooov_ba.get(), bw_mje_b.get(),
                    bar_t.t1b(), Real(-1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_core)
          rm_gemm<double>(blas, st, 'N', 'T', ob, oa * ob * va, vb, amp.t1b(), bw_mje_b.get(),
                          bar_core->gooov_ba.get(), -1.0, 1.0);
      }
    }
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, ob * vb, oa * va, gab_menf.get(),
                  inter.bar_W_abab.get(), bt_nfjb_s[1].get(), Real(-1), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', oa * va, ob * vb, oa * va, g_menf_s[0].get(),
                  inter.bar_W_abab.get(), btab_nfa_jbb.get(), Real(0.5), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core) {
        DeviceBuf<double> bgab(dev, oa * va * ob * vb, st), bgaa(dev, oa * va * oa * va, st);
        rm_gemm<double>(blas, st, 'N', 'T', oa * va, ob * vb, ob * vb, inter.bar_W_abab.get(),
                        t_nfjb_s[1].get(), bgab.get(), -1.0, 0.0);
        rm_gemm<double>(blas, st, 'N', 'T', oa * va, oa * va, ob * vb, inter.bar_W_abab.get(),
                        tab_ia_me.get(), bgaa.get(), 0.5, 0.0);
        launch_permute4d<double, 0, 2, 1, 3>(oa, va, ob, vb, bgab.get(), bar_core->goovv_ab.get(),
                                             st, true);
        launch_permute4d<double, 0, 2, 1, 3>(oa, va, oa, va, bgaa.get(), bar_core->goovv_aa.get(),
                                             st, true);
      }
    }

    // W_baba
    rm_gemm<Real>(blas, st, 'N', 'N', q, oa * va, ob * vb, src.bov[1], inter.bar_W_baba.get(),
                  inter.bar_x[0].get(), Real(1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_b)
        rm_gemm<double>(blas, st, 'N', 'T', q, ob * vb, oa * va, c.x_s[0].get(),
                        inter.bar_W_baba.get(), bar_b->bov_b.get(), 1.0, 1.0);
      if (bar_core)
        launch_permute4d<double, 2, 0, 3, 1>(ob, vb, oa, va, inter.bar_W_baba.get(),
                                             bar_core->goovv_ab.get(), st, true);
    }
    {
      DeviceBuf<Real> bw_mje_b(dev, ob * oa * vb * va, st);
      launch_permute4d<Real, 0, 2, 1, 3>(ob, vb, oa, va, inter.bar_W_baba.get(), bw_mje_b.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'N', oa, va, ob * oa * vb, core.gooov_ab.get(), bw_mje_b.get(),
                    bar_t.t1a(), Real(-1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_core)
          rm_gemm<double>(blas, st, 'N', 'T', oa, ob * oa * vb, va, amp.t1a(), bw_mje_b.get(),
                          bar_core->gooov_ab.get(), -1.0, 1.0);
      }
    }
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, oa * va, ob * vb, gab_menf.get(),
                  inter.bar_W_baba.get(), bt_nfjb_s[0].get(), Real(-1), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, oa * va, ob * vb, g_menf_s[1].get(),
                  inter.bar_W_baba.get(), btab_nfb_jaa.get(), Real(0.5), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core) {
        DeviceBuf<double> bgab(dev, oa * va * ob * vb, st), bgbb(dev, ob * vb * ob * vb, st);
        rm_gemm<double>(blas, st, 'N', 'T', oa * va, ob * vb, oa * va, t_nfjb_s[0].get(),
                        inter.bar_W_baba.get(), bgab.get(), -1.0, 0.0);
        rm_gemm<double>(blas, st, 'N', 'T', ob * vb, ob * vb, oa * va, inter.bar_W_baba.get(),
                        tab_jb_me.get(), bgbb.get(), 0.5, 0.0);
        launch_permute4d<double, 0, 2, 1, 3>(oa, va, ob, vb, bgab.get(), bar_core->goovv_ab.get(),
                                             st, true);
        launch_permute4d<double, 0, 2, 1, 3>(ob, vb, ob, vb, bgbb.get(), bar_core->goovv_bb.get(),
                                             st, true);
      }
    }
    launch_permute4d<Real, 0, 2, 1, 3>(oa, va, ob, vb, btab_nfa_jbb.get(), bar_t.t2ab(), st, true);
    launch_permute4d<Real, 2, 0, 3, 1>(ob, vb, oa, va, btab_nfb_jaa.get(), bar_t.t2ab(), st, true);
    btab_nfb_jaa.reset();
    btab_nfa_jbb.reset();

    // W_abba
    {
      DeviceBuf<Real> bya_bvv_b(dev, oa * oa * vb * vb, st);
      wovvo_bar_to_by_bvv_kernel<Real><<<grid_1d(oa * vb * oa * vb), 256, 0, st>>>(
          oa, vb, inter.bar_W_abba.get(), bya_bvv_b.get());
      rm_gemm<Real>(blas, st, 'N', 'T', q, oa * oa, vb * vb, src.bvv[1], bya_bvv_b.get(),
                    bar_y[0].get(), Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_b)
          rm_gemm<double>(blas, st, 'N', 'N', q, vb * vb, oa * oa, c.y_s[0].get(), bya_bvv_b.get(),
                          bar_b->bvv_b.get(), 1.0, 1.0);
        if (bar_core)
          wovvo_bar_sub_govov_kernel<double><<<grid_1d(oa * vb * oa * vb), 256, 0, st>>>(
              oa, vb, inter.bar_W_abba.get(), bar_core->govov_ab.get());
      }
    }
    {
      DeviceBuf<Real> bw_mje_b(dev, oa * oa * vb * vb, st), g_n_mje(dev, ob * oa * oa * vb, st);
      launch_permute4d<Real, 0, 2, 1, 3>(oa, vb, oa, vb, inter.bar_W_abba.get(), bw_mje_b.get(), st);
      launch_permute4d<Real, 1, 0, 2, 3>(oa, ob, oa, vb, core.gooov_ab.get(), g_n_mje.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'N', ob, vb, oa * oa * vb, g_n_mje.get(), bw_mje_b.get(),
                    bar_t.t1b(), Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_core) {
          rm_gemm<double>(blas, st, 'N', 'T', ob, oa * oa * vb, vb, amp.t1b(), bw_mje_b.get(),
                          g_n_mje.get(), 1.0, 0.0);
          launch_permute4d<double, 1, 0, 2, 3>(ob, oa, oa, vb, g_n_mje.get(),
                                               bar_core->gooov_ab.get(), st, true);
        }
      }
    }
    {
      DeviceBuf<Real> gab_x(dev, oa * vb * ob * va, st), btab_x(dev, ob * va * oa * vb, st),
          bx_jf_nb(dev, oa * va * ob * vb, st);
      launch_permute4d<Real, 0, 3, 1, 2>(oa, ob, va, vb, core.goovv_ab.get(), gab_x.get(), st);
      rm_gemm<Real>(blas, st, 'T', 'N', ob * va, oa * vb, oa * vb, gab_x.get(),
                    inter.bar_W_abba.get(), btab_x.get(), Real(1), Real(0));
      scatter_half_btab_x_kernel<Real><<<grid_1d(btab_x.size()), 256, 0, st>>>(
          oa, ob, va, vb, btab_x.get(), bar_t.t2ab());
      launch_permute4d<Real, 2, 1, 0, 3>(ob, va, oa, vb, btab_x.get(), bx_jf_nb.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'N', oa * va, 1, ob * vb, bx_jf_nb.get(), amp.t1b(), bar_t.t1a(),
                    Real(1), Real(1));
      rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, 1, oa * va, bx_jf_nb.get(), amp.t1a(), bar_t.t1b(),
                    Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_core) {
          build_tab_nf_jb_x_kernel<double><<<grid_1d(btab_x.size()), 256, 0, st>>>(
              oa, ob, va, vb, amp.t2ab(), amp.t1a(), amp.t1b(), btab_x.get());
          rm_gemm<double>(blas, st, 'N', 'T', oa * vb, ob * va, oa * vb, inter.bar_W_abba.get(),
                          btab_x.get(), gab_x.get(), 1.0, 0.0);
          launch_permute4d<double, 0, 2, 3, 1>(oa, vb, ob, va, gab_x.get(),
                                               bar_core->goovv_ab.get(), st, true);
        }
      }
    }

    // W_baab
    {
      DeviceBuf<Real> byb_bvv_a(dev, ob * ob * va * va, st);
      wovvo_bar_to_by_bvv_kernel<Real><<<grid_1d(ob * va * ob * va), 256, 0, st>>>(
          ob, va, inter.bar_W_baab.get(), byb_bvv_a.get());
      rm_gemm<Real>(blas, st, 'N', 'T', q, ob * ob, va * va, src.bvv[0], byb_bvv_a.get(),
                    bar_y[1].get(), Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_b)
          rm_gemm<double>(blas, st, 'N', 'N', q, va * va, ob * ob, c.y_s[1].get(), byb_bvv_a.get(),
                          bar_b->bvv_a.get(), 1.0, 1.0);
        if (bar_core)
          wovvo_bar_sub_govov_kernel<double><<<grid_1d(ob * va * ob * va), 256, 0, st>>>(
              ob, va, inter.bar_W_baab.get(), bar_core->govov_ba.get());
      }
    }
    {
      DeviceBuf<Real> bw_mje_b(dev, ob * ob * va * va, st), g_n_mje(dev, oa * ob * ob * va, st);
      launch_permute4d<Real, 0, 2, 1, 3>(ob, va, ob, va, inter.bar_W_baab.get(), bw_mje_b.get(), st);
      launch_permute4d<Real, 1, 0, 2, 3>(ob, oa, ob, va, core.gooov_ba.get(), g_n_mje.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'N', oa, va, ob * ob * va, g_n_mje.get(), bw_mje_b.get(),
                    bar_t.t1a(), Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_core) {
          rm_gemm<double>(blas, st, 'N', 'T', oa, ob * ob * va, va, amp.t1a(), bw_mje_b.get(),
                          g_n_mje.get(), 1.0, 0.0);
          launch_permute4d<double, 1, 0, 2, 3>(oa, ob, ob, va, g_n_mje.get(),
                                               bar_core->gooov_ba.get(), st, true);
        }
      }
    }
    {
      DeviceBuf<Real> gba_x(dev, ob * va * oa * vb, st), btba_x(dev, oa * vb * ob * va, st),
          bx_nb_jf(dev, oa * va * ob * vb, st);
      launch_permute4d<Real, 1, 2, 0, 3>(oa, ob, va, vb, core.goovv_ab.get(), gba_x.get(), st);
      rm_gemm<Real>(blas, st, 'T', 'N', oa * vb, ob * va, ob * va, gba_x.get(),
                    inter.bar_W_baab.get(), btba_x.get(), Real(1), Real(0));
      scatter_half_btba_x_kernel<Real><<<grid_1d(btba_x.size()), 256, 0, st>>>(
          oa, ob, va, vb, btba_x.get(), bar_t.t2ab());
      launch_permute4d<Real, 0, 3, 2, 1>(oa, vb, ob, va, btba_x.get(), bx_nb_jf.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'N', oa * va, 1, ob * vb, bx_nb_jf.get(), amp.t1b(), bar_t.t1a(),
                    Real(1), Real(1));
      rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, 1, oa * va, bx_nb_jf.get(), amp.t1a(), bar_t.t1b(),
                    Real(1), Real(1));
      if constexpr (std::is_same_v<Real, double>) {
        if (bar_core) {
          build_tba_nf_jb_x_kernel<double><<<grid_1d(btba_x.size()), 256, 0, st>>>(
              oa, ob, va, vb, amp.t2ab(), amp.t1a(), amp.t1b(), btba_x.get());
          rm_gemm<double>(blas, st, 'N', 'T', ob * va, oa * vb, ob * va, inter.bar_W_baab.get(),
                          btba_x.get(), gba_x.get(), 1.0, 0.0);
          launch_permute4d<double, 2, 0, 1, 3>(ob, va, oa, vb, gba_x.get(),
                                               bar_core->goovv_ab.get(), st, true);
        }
      }
    }
  }
  t_nfjb_s[0].reset(); t_nfjb_s[1].reset(); tab_ia_me.reset(); tab_jb_me.reset();
  inter.bar_W_ss[0].reset(); inter.bar_W_ss[1].reset();
  inter.bar_W_abab.reset(); inter.bar_W_baba.reset();
  inter.bar_W_abba.reset(); inter.bar_W_baab.reset();

  // Scatter accumulated bt_nfjb_s -> bar_t2_s, bar_t1_s
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if (!o || !v) continue;
    scatter_half_bt_nfjb_ss_kernel<Real><<<grid_1d(bt_nfjb_s[s].size()), 256, 0, st>>>(
        o, v, bt_nfjb_s[s].get(), bar_t.t2(s));
    DeviceBuf<Real> b_jf_nb(dev, o * v * o * v, st);
    launch_permute4d<Real, 2, 1, 0, 3>(o, v, o, v, bt_nfjb_s[s].get(), b_jf_nb.get(), st);
    rm_gemm<Real>(blas, st, 'N', 'N', o * v, 1, o * v, b_jf_nb.get(), amp.t1(s), bar_t.t1(s),
                  Real(1), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', o * v, 1, o * v, b_jf_nb.get(), amp.t1(s), bar_t.t1(s),
                  Real(1), Real(1));
    bt_nfjb_s[s].reset();
  }
  cuda_check(cudaGetLastError(), "lambda pass B1");

  // B2. Pull back Fvv_d, Foo_d -> Fvv, Foo, Fov, t1
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    launch_add_scaled(inter.bar_Fvv[s].get(), inter.bar_Fvv_d[s].get(), Real(1), v * v, st);
    launch_add_scaled(inter.bar_Foo[s].get(), inter.bar_Foo_d[s].get(), Real(1), o * o, st);
    if (!o || !v) continue;
    rm_gemm<Real>(blas, st, 'N', 'T', o, v, v, c.Fov_s[s].get(), inter.bar_Fvv_d[s].get(),
                  bar_t.t1(s), Real(-0.5), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', o, v, v, amp.t1(s), inter.bar_Fvv_d[s].get(),
                  inter.bar_Fov[s].get(), Real(-0.5), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', o, v, o, inter.bar_Foo_d[s].get(), amp.t1(s),
                  inter.bar_Fov[s].get(), Real(0.5), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', o, v, o, inter.bar_Foo_d[s].get(), c.Fov_s[s].get(),
                  bar_t.t1(s), Real(0.5), Real(1));
  }

  // B3. Pull back Fov
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if (!o || !v) continue;
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_f) launch_add_scaled(bar_f->fov(s), inter.bar_Fov[s].get(), 1.0, o * v, st);
    }
    rm_gemm<Real>(blas, st, 'T', 'N', o * v, 1, o * v, g_menf_s[s].get(), inter.bar_Fov[s].get(),
                  bar_t.t1(s), Real(1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        pullback_fov_goovv_ss_kernel<double><<<grid_1d(o * o * v * v), 256, 0, st>>>(
            o, v, inter.bar_Fov[s].get(), amp.t1(s), bar_core->goovv(s));
    }
  }
  if (oa && ob && va && vb) {
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, 1, oa * va, gab_menf.get(), inter.bar_Fov[0].get(),
                  bar_t.t1b(), Real(1), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, 1, ob * vb, gab_menf.get(), inter.bar_Fov[1].get(),
                  bar_t.t1a(), Real(1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        pullback_fov_goovv_ab_kernel<double><<<grid_1d(oa * ob * va * vb), 256, 0, st>>>(
            oa, ob, va, vb, inter.bar_Fov[0].get(), inter.bar_Fov[1].get(), amp.t1a(), amp.t1b(),
            bar_core->goovv_ab.get());
    }
  }
  g_menf_s[0].reset(); g_menf_s[1].reset(); gab_menf.reset();
  cuda_check(cudaGetLastError(), "lambda pass B3");

  // B4. Pull back Foo (build tauh on demand only when bar_core != nullptr in FP64)
  DeviceBuf<double> tauh_s[2], tauh_ab;
  if constexpr (std::is_same_v<Real, double>) {
    if (bar_core) build_tau_buffers_device<double>(d, amp, 0.5, dev, st, tauh_s, tauh_ab);
  }
  DeviceBuf<Real> bar_tauh[2], bar_tauh_ab;
  for (int s = 0; s < 2; ++s) bar_tauh[s].allocate_zero(dev, src.o(s) * src.o(s) * src.v(s) * src.v(s), st);
  bar_tauh_ab.allocate_zero(dev, oa * ob * va * vb, st);

  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_f && o) launch_add_scaled(bar_f->foo(s), inter.bar_Foo[s].get(), 1.0, o * o, st);
    }
    if (!o || !v) continue;
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_f)
        rm_gemm<double>(blas, st, 'N', 'N', o, v, o, inter.bar_Foo[s].get(), amp.t1(s),
                        bar_f->fov(s), 0.5, 1.0);
    }
    rm_gemm<Real>(blas, st, 'T', 'N', o, v, o, inter.bar_Foo[s].get(), src.fov[s], bar_t.t1(s),
                  Real(0.5), Real(1));
    pullback_foo_gooov_t1_kernel<Real><<<grid_1d(o * v), 256, 0, st>>>(
        o, o, v, inter.bar_Foo[s].get(), core.gooov(s), bar_t.t1(s));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        pullback_foo_gooov_core_kernel<double><<<grid_1d(o * o * o * v), 256, 0, st>>>(
            o, o, v, inter.bar_Foo[s].get(), amp.t1(s), bar_core->gooov(s));
    }
    rm_gemm<Real>(blas, st, 'T', 'N', o, o * v * v, o, inter.bar_Foo[s].get(), core.goovv(s),
                  bar_tauh[s].get(), Real(0.5), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        rm_gemm<double>(blas, st, 'N', 'N', o, o * v * v, o, inter.bar_Foo[s].get(),
                        tauh_s[s].get(), bar_core->goovv(s), 0.5, 1.0);
    }
  }
  if (oa && ob && vb) {
    pullback_foo_gooov_t1_kernel<Real><<<grid_1d(ob * vb), 256, 0, st>>>(
        oa, ob, vb, inter.bar_Foo[0].get(), core.gooov_ab.get(), bar_t.t1b());
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        pullback_foo_gooov_core_kernel<double><<<grid_1d(oa * ob * oa * vb), 256, 0, st>>>(
            oa, ob, vb, inter.bar_Foo[0].get(), amp.t1b(), bar_core->gooov_ab.get());
    }
  }
  if (ob && oa && va) {
    pullback_foo_gooov_t1_kernel<Real><<<grid_1d(oa * va), 256, 0, st>>>(
        ob, oa, va, inter.bar_Foo[1].get(), core.gooov_ba.get(), bar_t.t1a());
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        pullback_foo_gooov_core_kernel<double><<<grid_1d(ob * oa * ob * va), 256, 0, st>>>(
            ob, oa, va, inter.bar_Foo[1].get(), amp.t1a(), bar_core->gooov_ba.get());
    }
  }
  if (oa && ob && va && vb) {
    rm_gemm<Real>(blas, st, 'T', 'N', oa, ob * va * vb, oa, inter.bar_Foo[0].get(),
                  core.goovv_ab.get(), bar_tauh_ab.get(), Real(1), Real(1));
    rm_gemm_batched<Real>(blas, st, 'T', 'N', ob, va * vb, ob, inter.bar_Foo[1].get(), 0,
                          core.goovv_ab.get(), static_cast<long long>(ob * va * vb),
                          bar_tauh_ab.get(), static_cast<long long>(ob * va * vb), oa, Real(1),
                          Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core) {
        rm_gemm<double>(blas, st, 'N', 'N', oa, ob * va * vb, oa, inter.bar_Foo[0].get(),
                        tauh_ab.get(), bar_core->goovv_ab.get(), 1.0, 1.0);
        rm_gemm_batched<double>(blas, st, 'N', 'N', ob, va * vb, ob, inter.bar_Foo[1].get(), 0,
                                tauh_ab.get(), static_cast<long long>(ob * va * vb),
                                bar_core->goovv_ab.get(), static_cast<long long>(ob * va * vb), oa,
                                1.0, 1.0);
      }
    }
  }
  cuda_check(cudaGetLastError(), "lambda pass B4");

  // B5. Pull back Fvv
  DeviceBuf<Real> bar_jq(dev, q, st);
  bar_jq.zero(st);
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if (!v) continue;
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_f) launch_add_scaled(bar_f->fvv(s), inter.bar_Fvv[s].get(), 1.0, v * v, st);
    }
    rm_gemm<Real>(blas, st, 'N', 'N', q, 1, v * v, src.bvv[s], inter.bar_Fvv[s].get(), bar_jq.get(),
                  Real(1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_b)
        rm_gemm<double>(blas, st, 'N', 'N', q, v * v, 1, c.j_q.get(), inter.bar_Fvv[s].get(),
                        bar_b->bvv(s), 1.0, 1.0);
    }
    if (!o) continue;
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_f)
        rm_gemm<double>(blas, st, 'N', 'N', o, v, v, amp.t1(s), inter.bar_Fvv[s].get(),
                        bar_f->fov(s), -0.5, 1.0);
    }
    rm_gemm<Real>(blas, st, 'N', 'T', o, v, v, src.fov[s], inter.bar_Fvv[s].get(), bar_t.t1(s),
                  Real(-0.5), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'T', q * o, v, v, src.bov[s], inter.bar_Fvv[s].get(),
                  inter.bar_x[s].get(), Real(-1), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_b)
        rm_gemm<double>(blas, st, 'N', 'N', q * o, v, v, c.x_s[s].get(), inter.bar_Fvv[s].get(),
                        bar_b->bov(s), -1.0, 1.0);
    }
    rm_gemm_batched<Real>(blas, st, 'N', 'N', v, v, v, inter.bar_Fvv[s].get(), 0, core.goovv(s),
                          static_cast<long long>(v * v), bar_tauh[s].get(),
                          static_cast<long long>(v * v), o * o, Real(-0.5), Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core)
        rm_gemm_batched<double>(blas, st, 'T', 'N', v, v, v, inter.bar_Fvv[s].get(), 0,
                                tauh_s[s].get(), static_cast<long long>(v * v), bar_core->goovv(s),
                                static_cast<long long>(v * v), o * o, -0.5, 1.0);
    }
  }
  if (oa && ob && va && vb) {
    rm_gemm_batched<Real>(blas, st, 'N', 'N', va, vb, va, inter.bar_Fvv[0].get(), 0,
                          core.goovv_ab.get(), static_cast<long long>(va * vb), bar_tauh_ab.get(),
                          static_cast<long long>(va * vb), oa * ob, Real(-1), Real(1));
    rm_gemm_batched<Real>(blas, st, 'N', 'T', va, vb, vb, core.goovv_ab.get(),
                          static_cast<long long>(va * vb), inter.bar_Fvv[1].get(), 0,
                          bar_tauh_ab.get(), static_cast<long long>(va * vb), oa * ob, Real(-1),
                          Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_core) {
        rm_gemm_batched<double>(blas, st, 'T', 'N', va, vb, va, inter.bar_Fvv[0].get(), 0,
                                tauh_ab.get(), static_cast<long long>(va * vb),
                                bar_core->goovv_ab.get(), static_cast<long long>(va * vb), oa * ob,
                                -1.0, 1.0);
        rm_gemm_batched<double>(blas, st, 'N', 'N', va, vb, vb, tauh_ab.get(),
                                static_cast<long long>(va * vb), inter.bar_Fvv[1].get(), 0,
                                bar_core->goovv_ab.get(), static_cast<long long>(va * vb), oa * ob,
                                -1.0, 1.0);
      }
    }
  }
  tauh_s[0].reset(); tauh_s[1].reset(); tauh_ab.reset();

  // B6. Pull back bt, x, y, j_q -> t1 and DF factors
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_b && v) launch_add_scaled(bar_b->bvv(s), inter.bar_bt[s].get(), 1.0, q * v * v, st);
    }
    if (!o || !v) continue;
    {
      DeviceBuf<Real> bov_p(dev, o * q * v, st), bbt_p(dev, v * q * v, st);
      launch_permute3d<Real, 1, 0, 2>(q, o, v, src.bov[s], bov_p.get(), st);
      launch_permute3d<Real, 1, 0, 2>(q, v, v, inter.bar_bt[s].get(), bbt_p.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', o, v, q * v, bov_p.get(), bbt_p.get(), bar_t.t1(s),
                    Real(-1), Real(1));
      launch_permute3d<Real, 1, 0, 2>(q, o, v, inter.bar_x[s].get(), bov_p.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'N', o, v, q * v, bov_p.get(), src.bvv[s], bar_t.t1(s), Real(1),
                    Real(1));
    }
    {
      DeviceBuf<Real> by_p(dev, o * q * o, st);
      launch_permute3d<Real, 2, 0, 1>(q, o, o, bar_y[s].get(), by_p.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'N', o, v, q * o, by_p.get(), src.bov[s], bar_t.t1(s), Real(1),
                    Real(1));
    }
    rm_gemm<Real>(blas, st, 'N', 'N', 1, o * v, q, bar_jq.get(), src.bov[s], bar_t.t1(s), Real(1),
                  Real(1));
    if constexpr (std::is_same_v<Real, double>) {
      if (bar_b) {
        rm_gemm_batched<double>(blas, st, 'N', 'N', o, v, v, amp.t1(s), 0, inter.bar_bt[s].get(),
                                static_cast<long long>(v * v), bar_b->bov(s),
                                static_cast<long long>(o * v), q, -1.0, 1.0);
        rm_gemm_batched<double>(blas, st, 'T', 'N', v, v, o, inter.bar_x[s].get(),
                                static_cast<long long>(o * v), amp.t1(s), 0, bar_b->bvv(s),
                                static_cast<long long>(v * v), q, 1.0, 1.0);
        rm_gemm_batched<double>(blas, st, 'N', 'N', o, v, o, bar_y[s].get(),
                                static_cast<long long>(o * o), amp.t1(s), 0, bar_b->bov(s),
                                static_cast<long long>(o * v), q, 1.0, 1.0);
        rm_gemm<double>(blas, st, 'N', 'N', q, o * v, 1, bar_jq.get(), amp.t1(s), bar_b->bov(s),
                        1.0, 1.0);
      }
    }
  }
  bar_y[0].reset(); bar_y[1].reset(); bar_jq.reset();

  // B7. Pull back tau (factor 1.0) and tauh (factor 0.5) -> t2, t1
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if (!o || !v) continue;
    const std::size_t n2 = o * o * v * v;
    launch_add_scaled(bar_t.t2(s), inter.bar_tau[s].get(), Real(1), n2, st);
    launch_add_scaled(bar_t.t2(s), bar_tauh[s].get(), Real(1), n2, st);
    launch_add_scaled(inter.bar_tau[s].get(), bar_tauh[s].get(), Real(0.5), n2, st);
    pullback_tau_ss_t1_kernel<Real><<<grid_1d(o * v), 256, 0, st>>>(
        o, v, inter.bar_tau[s].get(), amp.t1(s), bar_t.t1(s));
  }
  if (oa && ob && va && vb) {
    const std::size_t nab = oa * ob * va * vb;
    launch_add_scaled(bar_t.t2ab(), inter.bar_tau_ab.get(), Real(1), nab, st);
    launch_add_scaled(bar_t.t2ab(), bar_tauh_ab.get(), Real(1), nab, st);
    launch_add_scaled(inter.bar_tau_ab.get(), bar_tauh_ab.get(), Real(0.5), nab, st);
    DeviceBuf<Real> b_ia_jb(dev, nab, st);
    launch_permute4d<Real, 0, 2, 1, 3>(oa, ob, va, vb, inter.bar_tau_ab.get(), b_ia_jb.get(), st);
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, 1, ob * vb, b_ia_jb.get(), amp.t1b(), bar_t.t1a(),
                  Real(1), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, 1, oa * va, b_ia_jb.get(), amp.t1a(), bar_t.t1b(),
                  Real(1), Real(1));
  }
  launch_antisymmetrize_t2<Real>(oa, va, bar_t.t2aa(), dev, st);
  launch_antisymmetrize_t2<Real>(ob, vb, bar_t.t2bb(), dev, st);
  cuda_check(cudaGetLastError(), "lambda pass B7");
}

}  // namespace generativeqc::cc::v2_detail::cuda_detail::lambda_vjp_cuda
