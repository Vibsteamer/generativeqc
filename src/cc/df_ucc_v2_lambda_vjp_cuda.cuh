#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "cc/df_ucc_v2_cuda_internal.cuh"

namespace generativeqc::cc::v2_detail::cuda_detail::lambda_vjp_cuda {

// --- Forward-cache and VJP helper kernels (zero atomics, fully deterministic) ---

template <class Real>
__global__ void foo_gooov_contract_kernel(std::size_t o1, std::size_t o2, std::size_t v,
                                          const Real* gooov, const Real* t1, Real* Foo) {
  const std::size_t total = o1 * o1;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t m = idx / o1;
    const std::size_t i = idx % o1;
    Real acc = Real(0);
    for (std::size_t n = 0; n < o2; ++n) {
      const Real* g = gooov + ((m * o2 + n) * o1 + i) * v;
      const Real* t = t1 + n * v;
      for (std::size_t e = 0; e < v; ++e) acc += g[e] * t[e];
    }
    Foo[idx] += acc;
  }
}

template <class Real>
__global__ void build_wbare_ss_kernel(std::size_t o, std::size_t v, const Real* goooo,
                                      const Real* gooov, const Real* t1, Real* w_bare) {
  const std::size_t oo = o * o;
  const std::size_t total = oo * oo;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t mn = idx / oo;
    const std::size_t ij = idx % oo;
    const std::size_t i = ij / o;
    const std::size_t j = ij % o;
    const Real* gi = gooov + (mn * o + i) * v;
    const Real* gj = gooov + (mn * o + j) * v;
    const Real* ti = t1 + i * v;
    const Real* tj = t1 + j * v;
    Real acc = goooo[idx];
    for (std::size_t e = 0; e < v; ++e) acc += tj[e] * gi[e] - ti[e] * gj[e];
    w_bare[idx] = acc;
  }
}

template <class Real>
__global__ void build_wbare_ab_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                      std::size_t vb, const Real* goooo_ab, const Real* gooov_ab,
                                      const Real* gooov_ba, const Real* t1a, const Real* t1b,
                                      Real* w_bare) {
  const std::size_t total = oa * ob * oa * ob;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::size_t tmp = idx;
    const std::size_t j = tmp % ob; tmp /= ob;
    const std::size_t i = tmp % oa; tmp /= oa;
    const std::size_t n = tmp % ob;
    const std::size_t m = tmp / ob;
    Real acc = goooo_ab[idx];
    const Real* gab = gooov_ab + ((m * ob + n) * oa + i) * vb;
    const Real* tj = t1b + j * vb;
    for (std::size_t e = 0; e < vb; ++e) acc += tj[e] * gab[e];
    const Real* gba = gooov_ba + ((n * oa + m) * ob + j) * va;
    const Real* ti = t1a + i * va;
    for (std::size_t e = 0; e < va; ++e) acc += ti[e] * gba[e];
    w_bare[idx] = acc;
  }
}

template <class Real>
__global__ void wovvo_ss_linear_kernel(std::size_t o, std::size_t v, const Real* govov,
                                       const Real* y_bvv, const Real* gooov, const Real* t1,
                                       Real* W) {
  const std::size_t total = o * v * o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::size_t tmp = idx;
    const std::size_t b = tmp % v; tmp /= v;
    const std::size_t j = tmp % o; tmp /= o;
    const std::size_t e = tmp % v;
    const std::size_t m = tmp / v;
    Real val = -govov[((m * v + b) * o + j) * v + e] - y_bvv[((m * o + j) * v + b) * v + e];
    for (std::size_t n = 0; n < o; ++n)
      val += t1[n * v + b] * gooov[((m * o + n) * o + j) * v + e];
    W[idx] += val;
  }
}

template <class Real>
__global__ void build_t_nfjb_ss_kernel(std::size_t o, std::size_t v, const Real* t2, const Real* t1,
                                       Real* out) {
  const std::size_t total = o * v * o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::size_t tmp = idx;
    const std::size_t b = tmp % v; tmp /= v;
    const std::size_t j = tmp % o; tmp /= o;
    const std::size_t f = tmp % v;
    const std::size_t n = tmp / v;
    out[idx] = Real(0.5) * t2[((j * o + n) * v + f) * v + b] + t1[j * v + f] * t1[n * v + b];
  }
}

template <class Real>
__global__ void wovvo_abab_linear_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                         std::size_t vb, const Real* goovv_ab, const Real* gooov_ba,
                                         const Real* t1b, Real* W_abab) {
  const std::size_t total = oa * va * ob * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::size_t tmp = idx;
    const std::size_t b = tmp % vb; tmp /= vb;
    const std::size_t j = tmp % ob; tmp /= ob;
    const std::size_t e = tmp % va;
    const std::size_t m = tmp / va;
    Real val = goovv_ab[((m * ob + j) * va + e) * vb + b];
    for (std::size_t n = 0; n < ob; ++n)
      val -= t1b[n * vb + b] * gooov_ba[((n * oa + m) * ob + j) * va + e];
    W_abab[idx] += val;
  }
}

template <class Real>
__global__ void wovvo_baba_linear_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                         std::size_t vb, const Real* goovv_ab, const Real* gooov_ab,
                                         const Real* t1a, Real* W_baba) {
  const std::size_t total = ob * vb * oa * va;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::size_t tmp = idx;
    const std::size_t b = tmp % va; tmp /= va;
    const std::size_t j = tmp % oa; tmp /= oa;
    const std::size_t e = tmp % vb;
    const std::size_t m = tmp / vb;
    Real val = goovv_ab[((j * ob + m) * va + b) * vb + e];
    for (std::size_t n = 0; n < oa; ++n)
      val -= t1a[n * va + b] * gooov_ab[((n * ob + m) * oa + j) * vb + e];
    W_baba[idx] += val;
  }
}

template <class Real>
__global__ void wovvo_abba_linear_kernel(std::size_t oa, std::size_t ob, std::size_t vb,
                                         const Real* govov_ab, const Real* ya_bvv_b,
                                         const Real* gooov_ab, const Real* t1b, Real* W_abba) {
  const std::size_t total = oa * vb * oa * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::size_t tmp = idx;
    const std::size_t b = tmp % vb; tmp /= vb;
    const std::size_t j = tmp % oa; tmp /= oa;
    const std::size_t e = tmp % vb;
    const std::size_t m = tmp / vb;
    Real val = -govov_ab[((m * vb + b) * oa + j) * vb + e] -
               ya_bvv_b[((m * oa + j) * vb + b) * vb + e];
    for (std::size_t n = 0; n < ob; ++n)
      val += t1b[n * vb + b] * gooov_ab[((m * ob + n) * oa + j) * vb + e];
    W_abba[idx] = val;
  }
}

template <class Real>
__global__ void build_tab_nf_jb_x_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                         std::size_t vb, const Real* t2ab, const Real* t1a,
                                         const Real* t1b, Real* tab_x) {
  const std::size_t total = ob * va * oa * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::size_t tmp = idx;
    const std::size_t b = tmp % vb; tmp /= vb;
    const std::size_t j = tmp % oa; tmp /= oa;
    const std::size_t f = tmp % va;
    const std::size_t n = tmp / va;
    tab_x[idx] =
        Real(0.5) * t2ab[((j * ob + n) * va + f) * vb + b] + t1a[j * va + f] * t1b[n * vb + b];
  }
}

template <class Real>
__global__ void wovvo_baab_linear_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                         const Real* govov_ba, const Real* yb_bvv_a,
                                         const Real* gooov_ba, const Real* t1a, Real* W_baab) {
  const std::size_t total = ob * va * ob * va;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::size_t tmp = idx;
    const std::size_t b = tmp % va; tmp /= va;
    const std::size_t j = tmp % ob; tmp /= ob;
    const std::size_t e = tmp % va;
    const std::size_t m = tmp / va;
    Real val = -govov_ba[((m * va + b) * ob + j) * va + e] -
               yb_bvv_a[((m * ob + j) * va + b) * va + e];
    for (std::size_t n = 0; n < oa; ++n)
      val += t1a[n * va + b] * gooov_ba[((m * oa + n) * ob + j) * va + e];
    W_baab[idx] = val;
  }
}

template <class Real>
__global__ void build_tba_nf_jb_x_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                         std::size_t vb, const Real* t2ab, const Real* t1a,
                                         const Real* t1b, Real* tba_x) {
  const std::size_t total = oa * vb * ob * va;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::size_t tmp = idx;
    const std::size_t b = tmp % va; tmp /= va;
    const std::size_t j = tmp % ob; tmp /= ob;
    const std::size_t f = tmp % vb;
    const std::size_t n = tmp / vb;
    tba_x[idx] =
        Real(0.5) * t2ab[((n * ob + j) * va + b) * vb + f] + t1b[j * vb + f] * t1a[n * va + b];
  }
}

template <class Real>
__global__ void build_t1_mjb_i_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                      std::size_t vb, const Real* t1a, const Real* goovv_ab,
                                      Real* out) {
  const std::size_t total = oa * ob * vb * oa;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::size_t tmp = idx;
    const std::size_t i = tmp % oa; tmp /= oa;
    const std::size_t b = tmp % vb; tmp /= vb;
    const std::size_t j = tmp % ob;
    const std::size_t m = tmp / ob;
    Real acc = Real(0);
    for (std::size_t e = 0; e < va; ++e)
      acc += t1a[i * va + e] * goovv_ab[((m * ob + j) * va + e) * vb + b];
    out[idx] = acc;
  }
}

template <class Real>
__global__ void pack_occ_lt_kernel(std::size_t o, std::size_t v, const Real* full, Real* packed) {
  const std::size_t np = (o >= 2) ? (o * (o - 1) / 2) : 0;
  const std::size_t vv = v * v;
  const std::size_t total = np * vv;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t p = idx / vv;
    const std::size_t ab = idx % vv;
    std::size_t j = static_cast<std::size_t>((sqrt(8.0 * static_cast<double>(p) + 1.0) + 1.0) * 0.5);
    while (j * (j - 1) / 2 > p) --j;
    while ((j + 1) * j / 2 <= p) ++j;
    const std::size_t i = p - j * (j - 1) / 2;
    packed[idx] = full[(i * o + j) * vv + ab];
  }
}

template <class Real>
__global__ void unpack_occ_lt_add_kernel(std::size_t o, std::size_t v, const Real* packed,
                                         Real* full) {
  const std::size_t np = (o >= 2) ? (o * (o - 1) / 2) : 0;
  const std::size_t vv = v * v;
  const std::size_t total = np * vv;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t p = idx / vv;
    const std::size_t ab = idx % vv;
    std::size_t j = static_cast<std::size_t>((sqrt(8.0 * static_cast<double>(p) + 1.0) + 1.0) * 0.5);
    while (j * (j - 1) / 2 > p) --j;
    while ((j + 1) * j / 2 <= p) ++j;
    const std::size_t i = p - j * (j - 1) / 2;
    const Real val = packed[idx];
    full[(i * o + j) * vv + ab] += val;
    full[(j * o + i) * vv + ab] -= val;
  }
}

// --- Cached forward intermediates on device (tauh built on demand only in param pass) ---
template <class Real>
struct DeviceForwardCache {
  DeviceBuf<Real> tau_s[2], tau_ab;
  DeviceBuf<Real> j_q;
  DeviceBuf<Real> x_s[2], y_s[2], bt_s[2];
  DeviceBuf<Real> Fvv_s[2], Foo_s[2], Fov_s[2], Fvv_d_s[2], Foo_d_s[2];
  DeviceBuf<Real> z_s[2];
  DeviceBuf<Real> w_bare_s[2], w_bare_ab;
  DeviceBuf<Real> tau_oovv_s[2], g_ab;
  DeviceBuf<Real> W_ss[2], W_abab, W_baba, W_abba, W_baab;
  DeviceBuf<Real> u_tg_s[2], t1_mjb_i, u2_ab, u3_ab, u4_ab;
};

template <class Real>
inline void build_tau_buffers_device(const SpinBlockedDims& d, const DeviceUccAmplitudes<Real>& amp,
                                     Real factor, int dev, cudaStream_t st,
                                     DeviceBuf<Real> tau_s[2], DeviceBuf<Real>& tau_ab) {
  const std::size_t os[2] = {d.oa, d.ob}, vs[2] = {d.va, d.vb};
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = os[s], v = vs[s];
    tau_s[s].allocate(dev, o * o * v * v, st);
    if (o && v) {
      build_tau_ss_kernel<Real><<<grid_1d(o * o * v * v), 256, 0, st>>>(
          o, v, amp.t2(s), amp.t1(s), factor, tau_s[s].get());
    }
  }
  tau_ab.allocate(dev, d.oa * d.ob * d.va * d.vb, st);
  if (d.oa && d.ob && d.va && d.vb) {
    build_tau_ab_kernel<Real><<<grid_1d(tau_ab.size()), 256, 0, st>>>(
        d.oa, d.ob, d.va, d.vb, amp.t2ab(), amp.t1a(), amp.t1b(), factor, tau_ab.get());
  }
  cuda_check(cudaGetLastError(), "lambda build_tau_buffers");
}

template <class Real>
inline DeviceForwardCache<Real> build_forward_cache_device(const DeviceSourceView<Real>& src,
                                                           const DeviceFourIndexCore<Real>& core,
                                                           const DeviceUccAmplitudes<Real>& amp,
                                                           int dev, cublasHandle_t blas,
                                                           cudaStream_t st) {
  const auto& d = src.dims;
  const std::size_t q = d.q, oa = d.oa, ob = d.ob, va = d.va, vb = d.vb;
  DeviceForwardCache<Real> c;

  build_tau_buffers_device<Real>(d, amp, Real(1), dev, st, c.tau_s, c.tau_ab);
  DeviceBuf<Real> tauh_s[2], tauh_ab;
  build_tau_buffers_device<Real>(d, amp, Real(0.5), dev, st, tauh_s, tauh_ab);

  c.j_q.allocate_zero(dev, q, st);
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if (o && v) {
      rm_gemm<Real>(blas, st, 'N', 'N', q, 1, o * v, src.bov[s], amp.t1(s), c.j_q.get(), Real(1),
                    Real(1));
    }
  }

  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    c.x_s[s].allocate_zero(dev, q * o * v, st);
    c.y_s[s].allocate_zero(dev, q * o * o, st);
    c.bt_s[s].allocate_zero(dev, q * v * v, st);
    if (v) c.bt_s[s].copy_from_device(src.bvv[s], q * v * v, st);
    if (o && v) {
      rm_gemm_batched<Real>(blas, st, 'N', 'T', o, v, v, amp.t1(s), 0, src.bvv[s],
                            static_cast<long long>(v * v), c.x_s[s].get(),
                            static_cast<long long>(o * v), q, Real(1), Real(0));
      rm_gemm_batched<Real>(blas, st, 'N', 'T', o, o, v, src.bov[s],
                            static_cast<long long>(o * v), amp.t1(s), 0, c.y_s[s].get(),
                            static_cast<long long>(o * o), q, Real(1), Real(0));
      rm_gemm_batched<Real>(blas, st, 'T', 'N', v, v, o, amp.t1(s), 0, src.bov[s],
                            static_cast<long long>(o * v), c.bt_s[s].get(),
                            static_cast<long long>(v * v), q, Real(-1), Real(1));
    }
  }

  // Fvv
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    c.Fvv_s[s].allocate_zero(dev, v * v, st);
    if (!v) continue;
    c.Fvv_s[s].copy_from_device(src.fvv[s], v * v, st);
    rm_gemm<Real>(blas, st, 'N', 'N', 1, v * v, q, c.j_q.get(), src.bvv[s], c.Fvv_s[s].get(),
                  Real(1), Real(1));
    if (o) {
      rm_gemm<Real>(blas, st, 'T', 'N', v, v, o, amp.t1(s), src.fov[s], c.Fvv_s[s].get(),
                    Real(-0.5), Real(1));
      rm_gemm<Real>(blas, st, 'T', 'N', v, v, q * o, c.x_s[s].get(), src.bov[s], c.Fvv_s[s].get(),
                    Real(-1), Real(1));
      DeviceBuf<Real> t_tmp(dev, v * o * o * v, st), g_tmp(dev, v * o * o * v, st);
      launch_permute3d<Real, 1, 0, 2>(o * o, v, v, tauh_s[s].get(), t_tmp.get(), st);
      launch_permute3d<Real, 1, 0, 2>(o * o, v, v, core.goovv(s), g_tmp.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', v, v, o * o * v, t_tmp.get(), g_tmp.get(), c.Fvv_s[s].get(),
                    Real(-0.5), Real(1));
    }
  }
  if (oa && ob && va && vb) {
    DeviceBuf<Real> t_tmp(dev, va * oa * ob * vb, st), g_tmp(dev, va * oa * ob * vb, st);
    launch_permute3d<Real, 1, 0, 2>(oa * ob, va, vb, tauh_ab.get(), t_tmp.get(), st);
    launch_permute3d<Real, 1, 0, 2>(oa * ob, va, vb, core.goovv_ab.get(), g_tmp.get(), st);
    rm_gemm<Real>(blas, st, 'N', 'T', va, va, oa * ob * vb, t_tmp.get(), g_tmp.get(),
                  c.Fvv_s[0].get(), Real(-1), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', vb, vb, oa * ob * va, tauh_ab.get(), core.goovv_ab.get(),
                  c.Fvv_s[1].get(), Real(-1), Real(1));
  }

  // Foo
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    c.Foo_s[s].allocate_zero(dev, o * o, st);
    if (!o) continue;
    c.Foo_s[s].copy_from_device(src.foo[s], o * o, st);
    if (v) {
      rm_gemm<Real>(blas, st, 'N', 'T', o, o, v, src.fov[s], amp.t1(s), c.Foo_s[s].get(), Real(0.5),
                    Real(1));
      foo_gooov_contract_kernel<Real><<<grid_1d(o * o), 256, 0, st>>>(
          o, o, v, core.gooov(s), amp.t1(s), c.Foo_s[s].get());
      rm_gemm<Real>(blas, st, 'N', 'T', o, o, o * v * v, core.goovv(s), tauh_s[s].get(),
                    c.Foo_s[s].get(), Real(0.5), Real(1));
    }
  }
  if (oa && ob && vb) {
    foo_gooov_contract_kernel<Real><<<grid_1d(oa * oa), 256, 0, st>>>(
        oa, ob, vb, core.gooov_ab.get(), amp.t1b(), c.Foo_s[0].get());
  }
  if (ob && oa && va) {
    foo_gooov_contract_kernel<Real><<<grid_1d(ob * ob), 256, 0, st>>>(
        ob, oa, va, core.gooov_ba.get(), amp.t1a(), c.Foo_s[1].get());
  }
  if (oa && ob && va && vb) {
    rm_gemm<Real>(blas, st, 'N', 'T', oa, oa, ob * va * vb, core.goovv_ab.get(), tauh_ab.get(),
                  c.Foo_s[0].get(), Real(1), Real(1));
    DeviceBuf<Real> g_b(dev, ob * oa * va * vb, st), t_b(dev, ob * oa * va * vb, st);
    launch_permute3d<Real, 1, 0, 2>(oa, ob, va * vb, core.goovv_ab.get(), g_b.get(), st);
    launch_permute3d<Real, 1, 0, 2>(oa, ob, va * vb, tauh_ab.get(), t_b.get(), st);
    rm_gemm<Real>(blas, st, 'N', 'T', ob, ob, oa * va * vb, g_b.get(), t_b.get(), c.Foo_s[1].get(),
                  Real(1), Real(1));
  }
  cuda_check(cudaGetLastError(), "lambda cache Foo");
  tauh_s[0].reset(); tauh_s[1].reset(); tauh_ab.reset();

  // Fov, Fvv_d, Foo_d + keep g_menf_s & gab_menf for Wovvo build below
  DeviceBuf<Real> g_menf_s[2], gab_menf;
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    c.Fov_s[s].allocate_zero(dev, o * v, st);
    if (!o || !v) continue;
    c.Fov_s[s].copy_from_device(src.fov[s], o * v, st);
    g_menf_s[s].allocate(dev, o * v * o * v, st);
    launch_permute4d<Real, 0, 2, 1, 3>(o, o, v, v, core.goovv(s), g_menf_s[s].get(), st);
    rm_gemm<Real>(blas, st, 'N', 'N', o * v, 1, o * v, g_menf_s[s].get(), amp.t1(s),
                  c.Fov_s[s].get(), Real(1), Real(1));
  }
  if (oa && ob && va && vb) {
    gab_menf.allocate(dev, oa * va * ob * vb, st);
    launch_permute4d<Real, 0, 2, 1, 3>(oa, ob, va, vb, core.goovv_ab.get(), gab_menf.get(), st);
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, 1, ob * vb, gab_menf.get(), amp.t1b(),
                  c.Fov_s[0].get(), Real(1), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, 1, oa * va, gab_menf.get(), amp.t1a(),
                  c.Fov_s[1].get(), Real(1), Real(1));
  }
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    c.Fvv_d_s[s].allocate_zero(dev, v * v, st);
    if (v) c.Fvv_d_s[s].copy_from_device(c.Fvv_s[s].get(), v * v, st);
    c.Foo_d_s[s].allocate_zero(dev, o * o, st);
    if (o) c.Foo_d_s[s].copy_from_device(c.Foo_s[s].get(), o * o, st);
    if (o && v) {
      rm_gemm<Real>(blas, st, 'T', 'N', v, v, o, amp.t1(s), c.Fov_s[s].get(), c.Fvv_d_s[s].get(),
                    Real(-0.5), Real(1));
      rm_gemm<Real>(blas, st, 'N', 'T', o, o, v, c.Fov_s[s].get(), amp.t1(s), c.Foo_d_s[s].get(),
                    Real(0.5), Real(1));
    }
  }

  // tab_ia_me & tab_jb_me reused in z_s and Wovvo
  DeviceBuf<Real> tab_ia_me, tab_jb_me;
  if (oa && ob && va && vb) {
    tab_ia_me.allocate(dev, oa * va * ob * vb, st);
    tab_jb_me.allocate(dev, ob * vb * oa * va, st);
    launch_permute4d<Real, 0, 2, 1, 3>(oa, ob, va, vb, amp.t2ab(), tab_ia_me.get(), st);
    launch_permute4d<Real, 1, 3, 0, 2>(oa, ob, va, vb, amp.t2ab(), tab_jb_me.get(), st);
  }

  // DF r1 intermediate z_s[Q, i, f]
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s), o2 = src.o(1 - s), v2 = src.v(1 - s);
    c.z_s[s].allocate_zero(dev, q * o * v, st);
    if (!o || !v) continue;
    DeviceBuf<Real> t2_ifme(dev, o * v * o * v, st);
    launch_permute4d<Real, 0, 3, 1, 2>(o, o, v, v, amp.t2(s), t2_ifme.get(), st);
    rm_gemm<Real>(blas, st, 'N', 'T', q, o * v, o * v, src.bov[s], t2_ifme.get(), c.z_s[s].get(),
                  Real(-1), Real(0));
    if (o2 && v2) {
      const Real* tab = (s == 0) ? tab_ia_me.get() : tab_jb_me.get();
      rm_gemm<Real>(blas, st, 'N', 'T', q, o * v, o2 * v2, src.bov[1 - s], tab, c.z_s[s].get(),
                    Real(1), Real(1));
    }
  }

  // w_bare and tau_oovv
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if (!o || !v) continue;
    c.w_bare_s[s].allocate(dev, o * o * o * o, st);
    build_wbare_ss_kernel<Real><<<grid_1d(o * o * o * o), 256, 0, st>>>(
        o, v, core.goooo(s), core.gooov(s), amp.t1(s), c.w_bare_s[s].get());
    c.tau_oovv_s[s].allocate(dev, o * o * o * o, st);
    rm_gemm<Real>(blas, st, 'N', 'T', o * o, o * o, v * v, core.goovv(s), c.tau_s[s].get(),
                  c.tau_oovv_s[s].get(), Real(1), Real(0));
  }
  if (oa && ob && va && vb) {
    c.w_bare_ab.allocate(dev, oa * ob * oa * ob, st);
    build_wbare_ab_kernel<Real><<<grid_1d(c.w_bare_ab.size()), 256, 0, st>>>(
        oa, ob, va, vb, core.goooo_ab.get(), core.gooov_ab.get(), core.gooov_ba.get(), amp.t1a(),
        amp.t1b(), c.w_bare_ab.get());
    c.g_ab.allocate(dev, oa * ob * oa * ob, st);
    rm_gemm<Real>(blas, st, 'N', 'T', oa * ob, oa * ob, va * vb, core.goovv_ab.get(),
                  c.tau_ab.get(), c.g_ab.get(), Real(1), Real(0));
  }
  cuda_check(cudaGetLastError(), "lambda cache w_bare");

  // Wovvo (6 spin-conserving blocks)
  DeviceBuf<Real> t_nfjb_s[2];
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    c.W_ss[s].allocate_zero(dev, o * v * o * v, st);
    if (!o || !v) continue;
    rm_gemm<Real>(blas, st, 'T', 'N', o * v, o * v, q, src.bov[s], c.x_s[s].get(), c.W_ss[s].get(),
                  Real(1), Real(0));
    DeviceBuf<Real> y_bvv(dev, o * o * v * v, st);
    rm_gemm<Real>(blas, st, 'T', 'N', o * o, v * v, q, c.y_s[s].get(), src.bvv[s], y_bvv.get(),
                  Real(1), Real(0));
    wovvo_ss_linear_kernel<Real><<<grid_1d(c.W_ss[s].size()), 256, 0, st>>>(
        o, v, core.govov(s), y_bvv.get(), core.gooov(s), amp.t1(s), c.W_ss[s].get());
    t_nfjb_s[s].allocate(dev, o * v * o * v, st);
    build_t_nfjb_ss_kernel<Real><<<grid_1d(t_nfjb_s[s].size()), 256, 0, st>>>(
        o, v, amp.t2(s), amp.t1(s), t_nfjb_s[s].get());
    rm_gemm<Real>(blas, st, 'N', 'N', o * v, o * v, o * v, g_menf_s[s].get(), t_nfjb_s[s].get(),
                  c.W_ss[s].get(), Real(-1), Real(1));
  }
  if (oa && ob && va && vb) {
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, oa * va, ob * vb, gab_menf.get(), tab_jb_me.get(),
                  c.W_ss[0].get(), Real(0.5), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, ob * vb, oa * va, gab_menf.get(), tab_ia_me.get(),
                  c.W_ss[1].get(), Real(0.5), Real(1));

    c.W_abab.allocate(dev, oa * va * ob * vb, st);
    c.W_baba.allocate(dev, ob * vb * oa * va, st);
    rm_gemm<Real>(blas, st, 'T', 'N', oa * va, ob * vb, q, src.bov[0], c.x_s[1].get(),
                  c.W_abab.get(), Real(1), Real(0));
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, oa * va, q, src.bov[1], c.x_s[0].get(),
                  c.W_baba.get(), Real(1), Real(0));
    wovvo_abab_linear_kernel<Real><<<grid_1d(c.W_abab.size()), 256, 0, st>>>(
        oa, ob, va, vb, core.goovv_ab.get(), core.gooov_ba.get(), amp.t1b(), c.W_abab.get());
    wovvo_baba_linear_kernel<Real><<<grid_1d(c.W_baba.size()), 256, 0, st>>>(
        oa, ob, va, vb, core.goovv_ab.get(), core.gooov_ab.get(), amp.t1a(), c.W_baba.get());
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, ob * vb, ob * vb, gab_menf.get(), t_nfjb_s[1].get(),
                  c.W_abab.get(), Real(-1), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, ob * vb, oa * va, g_menf_s[0].get(), tab_ia_me.get(),
                  c.W_abab.get(), Real(0.5), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, oa * va, oa * va, gab_menf.get(), t_nfjb_s[0].get(),
                  c.W_baba.get(), Real(-1), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', ob * vb, oa * va, ob * vb, g_menf_s[1].get(), tab_jb_me.get(),
                  c.W_baba.get(), Real(0.5), Real(1));

    c.W_abba.allocate(dev, oa * vb * oa * vb, st);
    {
      DeviceBuf<Real> ya_bvv_b(dev, oa * oa * vb * vb, st);
      rm_gemm<Real>(blas, st, 'T', 'N', oa * oa, vb * vb, q, c.y_s[0].get(), src.bvv[1],
                    ya_bvv_b.get(), Real(1), Real(0));
      wovvo_abba_linear_kernel<Real><<<grid_1d(c.W_abba.size()), 256, 0, st>>>(
          oa, ob, vb, core.govov_ab.get(), ya_bvv_b.get(), core.gooov_ab.get(), amp.t1b(),
          c.W_abba.get());
      DeviceBuf<Real> gab_x(dev, oa * vb * ob * va, st), tab_x(dev, ob * va * oa * vb, st);
      launch_permute4d<Real, 0, 3, 1, 2>(oa, ob, va, vb, core.goovv_ab.get(), gab_x.get(), st);
      build_tab_nf_jb_x_kernel<Real><<<grid_1d(tab_x.size()), 256, 0, st>>>(
          oa, ob, va, vb, amp.t2ab(), amp.t1a(), amp.t1b(), tab_x.get());
      rm_gemm<Real>(blas, st, 'N', 'N', oa * vb, oa * vb, ob * va, gab_x.get(), tab_x.get(),
                    c.W_abba.get(), Real(1), Real(1));
    }

    c.W_baab.allocate(dev, ob * va * ob * va, st);
    {
      DeviceBuf<Real> yb_bvv_a(dev, ob * ob * va * va, st);
      rm_gemm<Real>(blas, st, 'T', 'N', ob * ob, va * va, q, c.y_s[1].get(), src.bvv[0],
                    yb_bvv_a.get(), Real(1), Real(0));
      wovvo_baab_linear_kernel<Real><<<grid_1d(c.W_baab.size()), 256, 0, st>>>(
          oa, ob, va, core.govov_ba.get(), yb_bvv_a.get(), core.gooov_ba.get(), amp.t1a(),
          c.W_baab.get());
      DeviceBuf<Real> gba_x(dev, ob * va * oa * vb, st), tba_x(dev, oa * vb * ob * va, st);
      launch_permute4d<Real, 1, 2, 0, 3>(oa, ob, va, vb, core.goovv_ab.get(), gba_x.get(), st);
      build_tba_nf_jb_x_kernel<Real><<<grid_1d(tba_x.size()), 256, 0, st>>>(
          oa, ob, va, vb, amp.t2ab(), amp.t1a(), amp.t1b(), tba_x.get());
      rm_gemm<Real>(blas, st, 'N', 'N', ob * va, ob * va, oa * vb, gba_x.get(), tba_x.get(),
                    c.W_baab.get(), Real(1), Real(1));
    }
  }
  cuda_check(cudaGetLastError(), "lambda cache Wovvo");

  // u intermediates for t1*t1*govov terms
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = src.o(s), v = src.v(s);
    if (!o || !v) continue;
    c.u_tg_s[s].allocate(dev, o * v * o * o, st);
    rm_gemm<Real>(blas, st, 'N', 'T', o * v * o, o, v, core.govov(s), amp.t1(s), c.u_tg_s[s].get(),
                  Real(1), Real(0));
  }
  if (oa && ob && va && vb) {
    c.t1_mjb_i.allocate(dev, oa * ob * vb * oa, st);
    build_t1_mjb_i_kernel<Real><<<grid_1d(c.t1_mjb_i.size()), 256, 0, st>>>(
        oa, ob, va, vb, amp.t1a(), core.goovv_ab.get(), c.t1_mjb_i.get());
    c.u2_ab.allocate(dev, oa * ob * va * ob, st);
    rm_gemm<Real>(blas, st, 'N', 'T', oa * ob * va, ob, vb, core.goovv_ab.get(), amp.t1b(),
                  c.u2_ab.get(), Real(1), Real(0));
    c.u3_ab.allocate(dev, oa * vb * oa * ob, st);
    rm_gemm<Real>(blas, st, 'N', 'T', oa * vb * oa, ob, vb, core.govov_ab.get(), amp.t1b(),
                  c.u3_ab.get(), Real(1), Real(0));
    c.u4_ab.allocate(dev, ob * va * ob * oa, st);
    rm_gemm<Real>(blas, st, 'N', 'T', ob * va * ob, oa, va, core.govov_ba.get(), amp.t1a(),
                  c.u4_ab.get(), Real(1), Real(0));
  }
  cuda_check(cudaGetLastError(), "lambda cache u_tg");
  return c;
}

// High-throughput tiled-Q DF Virtual Ladder Adjoint (ZERO vvvv materialized!):
// Forward: L_out[p, a, b] = sum_Q sum_{e,f} bt1[Q, a, e] * tau[p, e, f] * bt2[Q, b, f]
// Accumulates into bar_bt1 (scale bbt_scale), bar_bt2 (scale bbt_scale), and bar_tau (scale 1).
template <class Real>
inline void accumulate_df_ppl_vjp_tiled(int dev, cublasHandle_t blas, cudaStream_t st,
                                        std::size_t q, std::size_t P, std::size_t v1,
                                        std::size_t v2, const Real* bt1, const Real* bt2,
                                        const Real* tau_p, const Real* L2_p, Real* bar_bt1,
                                        Real* bar_bt2, Real* bar_tau_p, Real bbt_scale,
                                        std::size_t tile_budget_bytes) {
  if (!q || !P || !v1 || !v2) return;
  const auto panel = checked_mul(P, checked_mul(v1, v2));
  const auto B_cap = ucc_lambda_ppl_q_tile(q, panel, sizeof(Real), ucc_device_available(tile_budget_bytes));
  DeviceBuf<Real> tau_epf(dev, panel, st), btau_epf(dev, panel, st);
  btau_epf.zero(st);
  launch_permute3d<Real, 1, 0, 2>(P, v1, v2, tau_p, tau_epf.get(), st);
  DeviceBuf<Real> buf_a(dev, checked_mul(B_cap, panel), st), buf_b(dev, checked_mul(B_cap, panel), st);

  for (std::size_t q0 = 0; q0 < q; q0 += B_cap) {
    const std::size_t B = std::min(B_cap, q - q0);
    const Real* bt1_t = bt1 + q0 * v1 * v1;
    const Real* bt2_t = bt2 + q0 * v2 * v2;
    Real* bbt1_t = bar_bt1 + q0 * v1 * v1;
    Real* bbt2_t = bar_bt2 + q0 * v2 * v2;

    // 1. H[B*v1, P*v2] = bt1_t[B*v1, v1] @ tau_epf[v1, P*v2]
    rm_gemm<Real>(blas, st, 'N', 'N', B * v1, P * v2, v1, bt1_t, tau_epf.get(), buf_a.get(),
                  Real(1), Real(0));
    // H[B, v1, P, v2] -> Hp[B, P, v1, v2]
    launch_permute4d<Real, 0, 2, 1, 3>(B, v1, P, v2, buf_a.get(), buf_b.get(), st);
    // 2. bbt2_t[q, v2, v2] += bbt_scale * L2_p^T[v2, P*v1] @ Hp[q, P*v1, v2]
    rm_gemm_batched<Real>(blas, st, 'T', 'N', v2, v2, P * v1, L2_p, 0, buf_b.get(),
                          static_cast<long long>(P * v1 * v2), bbt2_t,
                          static_cast<long long>(v2 * v2), B, bbt_scale, Real(1));

    // 3. BHp[q, P*v1, v2] = L2_p[P*v1, v2] @ bt2_t[q, v2, v2]
    rm_gemm_batched<Real>(blas, st, 'N', 'N', P * v1, v2, v2, L2_p, 0, bt2_t,
                          static_cast<long long>(v2 * v2), buf_b.get(),
                          static_cast<long long>(P * v1 * v2), B, Real(1), Real(0));
    // BHp[B, P, v1, v2] -> BH[B, v1, P, v2]
    launch_permute4d<Real, 0, 2, 1, 3>(B, P, v1, v2, buf_b.get(), buf_a.get(), st);
    // 4. bbt1_t[q, v1, v1] += bbt_scale * BH[q, v1, P*v2] @ tau_epf^T[P*v2, v1]
    rm_gemm_batched<Real>(blas, st, 'N', 'T', v1, v1, P * v2, buf_a.get(),
                          static_cast<long long>(v1 * P * v2), tau_epf.get(), 0, bbt1_t,
                          static_cast<long long>(v1 * v1), B, bbt_scale, Real(1));

    // 5. btau_epf[v1, P*v2] += bt1_t^T[v1, B*v1] @ BH[B*v1, P*v2]
    rm_gemm<Real>(blas, st, 'T', 'N', v1, P * v2, B * v1, bt1_t, buf_a.get(), btau_epf.get(),
                  Real(1), Real(1));
  }
  launch_permute3d<Real, 1, 0, 2>(v1, P, v2, btau_epf.get(), bar_tau_p, st, Real(1), true);
}

}  // namespace generativeqc::cc::v2_detail::cuda_detail::lambda_vjp_cuda
