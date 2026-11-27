#include "cc/df_ucc_v2_cuda_internal.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace generativeqc::cc::v2_detail::cuda_detail {
namespace {

// --- Energy Kernels (always accumulate in FP64 on device) ---
template <class Real>
__global__ void energy_1b_kernel(std::size_t n, const Real* fov, const Real* t1, double* out) {
  __shared__ double smem[256];
  double local = 0.0;
  for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < n;
       k += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    local += static_cast<double>(fov[k]) * static_cast<double>(t1[k]);
  }
  smem[threadIdx.x] = local;
  __syncthreads();
  for (unsigned s = 128U; s > 0U; s >>= 1U) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, smem[0]);
}

template <class Real>
__global__ void energy_2b_ss_kernel(std::size_t o, std::size_t v, const Real* goovv,
                                    const Real* t2, const Real* t1, double* out) {
  __shared__ double smem[256];
  const std::size_t total = o * o * v * v;
  double local = 0.0;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t a = r1 % v;
    const std::size_t r2 = r1 / v;
    const std::size_t j = r2 % o;
    const std::size_t i = r2 / o;
    const double g = static_cast<double>(goovv[idx]);
    local += 0.25 * g * static_cast<double>(t2[idx]) +
             0.5 * g * static_cast<double>(t1[i * v + a]) * static_cast<double>(t1[j * v + b]);
  }
  smem[threadIdx.x] = local;
  __syncthreads();
  for (unsigned s = 128U; s > 0U; s >>= 1U) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, smem[0]);
}

template <class Real>
__global__ void energy_2b_ab_kernel(std::size_t oa, std::size_t ob, std::size_t va, std::size_t vb,
                                    const Real* goovv, const Real* t2ab, const Real* t1a,
                                    const Real* t1b, double* out) {
  __shared__ double smem[256];
  const std::size_t total = oa * ob * va * vb;
  double local = 0.0;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % vb;
    const std::size_t r1 = idx / vb;
    const std::size_t a = r1 % va;
    const std::size_t r2 = r1 / va;
    const std::size_t j = r2 % ob;
    const std::size_t i = r2 / ob;
    const double g = static_cast<double>(goovv[idx]);
    local += g * (static_cast<double>(t2ab[idx]) +
                  static_cast<double>(t1a[i * va + a]) * static_cast<double>(t1b[j * vb + b]));
  }
  smem[threadIdx.x] = local;
  __syncthreads();
  for (unsigned s = 128U; s > 0U; s >>= 1U) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, smem[0]);
}

// Optional FixedF12Coupling energy kernels
template <class Real>
__global__ void energy_f12_1b_kernel(std::size_t oa, std::size_t ob, std::size_t va, std::size_t vb,
                                     const Real* w_vo_aaaa, const Real* w_vo_bbbb,
                                     const Real* w_vo_abab, const Real* w_vo_abba, const Real* t1a,
                                     const Real* t1b, double* out) {
  __shared__ double smem[256];
  const std::size_t na = oa * va, nb = ob * vb;
  double local = 0.0;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < na + nb; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (idx < na) {
      const std::size_t i = idx / va, a = idx % va;
      double r1 = 0.0;
      for (std::size_t j = 0; j < oa; ++j)
        r1 += static_cast<double>(w_vo_aaaa[((i * oa + j) * va + a) * oa + j]);
      for (std::size_t j = 0; j < ob; ++j)
        r1 += static_cast<double>(w_vo_abab[((i * ob + j) * va + a) * ob + j]);
      local += r1 * static_cast<double>(t1a[idx]);
    } else {
      const std::size_t k = idx - na;
      const std::size_t i = k / vb, a = k % vb;
      double r1 = 0.0;
      for (std::size_t j = 0; j < ob; ++j)
        r1 += static_cast<double>(w_vo_bbbb[((i * ob + j) * vb + a) * ob + j]);
      for (std::size_t j = 0; j < oa; ++j)
        r1 -= static_cast<double>(w_vo_abba[((j * ob + i) * vb + a) * oa + j]);
      local += r1 * static_cast<double>(t1b[k]);
    }
  }
  smem[threadIdx.x] = local;
  __syncthreads();
  for (unsigned s = 128U; s > 0U; s >>= 1U) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, smem[0]);
}

template <class Real>
__global__ void energy_f12_2b_ss_kernel(std::size_t o, std::size_t v, const Real* c_f12,
                                        const Real* w_vv, const Real* t2, const Real* t1,
                                        double* out) {
  __shared__ double smem[256];
  const std::size_t total = o * o * v * v;
  double local = 0.0;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t a = r1 % v;
    const std::size_t r2 = r1 / v;
    const std::size_t j = r2 % o;
    const std::size_t i = r2 / o;
    const double cw = static_cast<double>(c_f12[idx]) + static_cast<double>(w_vv[idx]);
    local += 0.25 * cw * static_cast<double>(t2[idx]) +
             0.5 * static_cast<double>(w_vv[idx]) * static_cast<double>(t1[i * v + a]) *
                 static_cast<double>(t1[j * v + b]);
  }
  smem[threadIdx.x] = local;
  __syncthreads();
  for (unsigned s = 128U; s > 0U; s >>= 1U) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, smem[0]);
}

template <class Real>
__global__ void energy_f12_2b_ab_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                        std::size_t vb, const Real* c_f12, const Real* w_vv,
                                        const Real* t2ab, const Real* t1a, const Real* t1b,
                                        double* out) {
  __shared__ double smem[256];
  const std::size_t total = oa * ob * va * vb;
  double local = 0.0;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % vb;
    const std::size_t r1 = idx / vb;
    const std::size_t a = r1 % va;
    const std::size_t r2 = r1 / va;
    const std::size_t j = r2 % ob;
    const std::size_t i = r2 / ob;
    const double cw = static_cast<double>(c_f12[idx]) + static_cast<double>(w_vv[idx]);
    local += cw * static_cast<double>(t2ab[idx]) +
             static_cast<double>(w_vv[idx]) * static_cast<double>(t1a[i * va + a]) *
                 static_cast<double>(t1b[j * vb + b]);
  }
  smem[threadIdx.x] = local;
  __syncthreads();
  for (unsigned s = 128U; s > 0U; s >>= 1U) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, smem[0]);
}

template <class Real>
double compute_uccsd_energy_device(const DeviceSourceView<Real>& src,
                                   const DeviceFourIndexCore<Real>& core,
                                   const DeviceUccAmplitudes<Real>& amp,
                                   const DeviceFixedF12Coupling<Real>* f12, int dev,
                                   cudaStream_t st) {
  const auto& d = src.dims;
  DeviceBuf<double> d_sum(dev, 1, st);
  d_sum.zero(st);
  if (d.oa && d.va) {
    energy_1b_kernel<Real><<<grid_1d(d.oa * d.va), 256, 0, st>>>(d.oa * d.va, src.fov[0], amp.t1a(),
                                                                 d_sum.get());
    energy_2b_ss_kernel<Real><<<grid_1d(d.oa * d.oa * d.va * d.va), 256, 0, st>>>(
        d.oa, d.va, core.goovv_aa.get(), amp.t2aa(), amp.t1a(), d_sum.get());
  }
  if (d.ob && d.vb) {
    energy_1b_kernel<Real><<<grid_1d(d.ob * d.vb), 256, 0, st>>>(d.ob * d.vb, src.fov[1], amp.t1b(),
                                                                 d_sum.get());
    energy_2b_ss_kernel<Real><<<grid_1d(d.ob * d.ob * d.vb * d.vb), 256, 0, st>>>(
        d.ob, d.vb, core.goovv_bb.get(), amp.t2bb(), amp.t1b(), d_sum.get());
  }
  if (d.oa && d.ob && d.va && d.vb) {
    energy_2b_ab_kernel<Real><<<grid_1d(d.oa * d.ob * d.va * d.vb), 256, 0, st>>>(
        d.oa, d.ob, d.va, d.vb, core.goovv_ab.get(), amp.t2ab(), amp.t1a(), amp.t1b(), d_sum.get());
  }
  double e_extra = 0.0;
  if (f12) {
    e_extra = f12->fixed_geminal_energy;
    const std::size_t n1 = d.oa * d.va + d.ob * d.vb;
    if (n1) {
      energy_f12_1b_kernel<Real><<<grid_1d(n1), 256, 0, st>>>(
          d.oa, d.ob, d.va, d.vb, f12->w_vo_aaaa.get(), f12->w_vo_bbbb.get(), f12->w_vo_abab.get(),
          f12->w_vo_abba.get(), amp.t1a(), amp.t1b(), d_sum.get());
    }
    if (d.oa && d.va) {
      energy_f12_2b_ss_kernel<Real><<<grid_1d(d.oa * d.oa * d.va * d.va), 256, 0, st>>>(
          d.oa, d.va, f12->c_f12_aaaa.get(), f12->w_vv_aaaa.get(), amp.t2aa(), amp.t1a(),
          d_sum.get());
    }
    if (d.ob && d.vb) {
      energy_f12_2b_ss_kernel<Real><<<grid_1d(d.ob * d.ob * d.vb * d.vb), 256, 0, st>>>(
          d.ob, d.vb, f12->c_f12_bbbb.get(), f12->w_vv_bbbb.get(), amp.t2bb(), amp.t1b(),
          d_sum.get());
    }
    if (d.oa && d.ob && d.va && d.vb) {
      energy_f12_2b_ab_kernel<Real><<<grid_1d(d.oa * d.ob * d.va * d.vb), 256, 0, st>>>(
          d.oa, d.ob, d.va, d.vb, f12->c_f12_abab.get(), f12->w_vv_abab.get(), amp.t2ab(),
          amp.t1a(), amp.t1b(), d_sum.get());
    }
  }
  cuda_check(cudaGetLastError(), "energy kernels");
  double e = 0.0;
  d_sum.copy_to_host(&e, 1, st);
  cuda_check(cudaStreamSynchronize(st), "energy sync");
  return e + e_extra;
}

// --- Residual Kernels ---
template <class Real>
__global__ void foo_gooov_contract_kernel(std::size_t o_out, std::size_t o_sum, std::size_t v_sum,
                                          const Real* gooov, const Real* t1, Real* Foo) {
  const std::size_t total = o_out * o_out;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t m = idx / o_out, i = idx % o_out;
    Real s = Real(0);
    for (std::size_t n = 0; n < o_sum; ++n) {
      const Real* g = gooov + ((m * o_sum + n) * o_out + i) * v_sum;
      const Real* t = t1 + n * v_sum;
      for (std::size_t e = 0; e < v_sum; ++e) s += g[e] * t[e];
    }
    Foo[idx] += s;
  }
}

template <class Real>
__global__ void r1_govov_ss_kernel(std::size_t o, std::size_t v, const Real* t1, const Real* govov,
                                   Real* r1) {
  const std::size_t total = o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t i = idx / v, a = idx % v;
    Real s = Real(0);
    for (std::size_t m = 0; m < o; ++m) {
      const Real* g = govov + ((m * v + a) * o + i) * v;
      const Real* t = t1 + m * v;
      for (std::size_t e = 0; e < v; ++e) s -= t[e] * g[e];
    }
    r1[idx] += s;
  }
}

template <class Real>
__global__ void reduce_q_sum_kernel(std::size_t q, std::size_t stride, const Real* src, Real* dst) {
  for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < stride;
       k += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    Real s = Real(0);
    for (std::size_t Q = 0; Q < q; ++Q) s += src[Q * stride + k];
    dst[k] += s;
  }
}

template <class Real>
__global__ void antisym_ab_add_kernel(std::size_t o, std::size_t v, const Real* tmp, Real* r2) {
  const std::size_t total = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t a = r1 % v;
    const std::size_t ij = r1 / v;
    r2[idx] += tmp[(ij * v + a) * v + b] - tmp[(ij * v + b) * v + a];
  }
}

template <class Real>
__global__ void antisym_ij_sub_kernel(std::size_t o, std::size_t v, const Real* tmp, Real* r2) {
  const std::size_t total = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t ab = idx % (v * v);
    const std::size_t ij = idx / (v * v);
    const std::size_t i = ij / o, j = ij % o;
    r2[idx] += -tmp[(i * o + j) * v * v + ab] + tmp[(j * o + i) * v * v + ab];
  }
}

template <class Real>
__global__ void build_wbare_ss_kernel(std::size_t o, std::size_t v, const Real* goooo,
                                      const Real* gooov, const Real* t1, Real* w_bare) {
  const std::size_t total = o * o * o * o;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t j = idx % o;
    const std::size_t r1 = idx / o;
    const std::size_t i = r1 % o;
    const std::size_t mn = r1 / o;
    const Real* gi = gooov + (mn * o + i) * v;
    const Real* gj = gooov + (mn * o + j) * v;
    const Real* ti = t1 + i * v;
    const Real* tj = t1 + j * v;
    Real s = goooo[idx];
    for (std::size_t e = 0; e < v; ++e) s += tj[e] * gi[e] - ti[e] * gj[e];
    w_bare[idx] = s;
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
    const std::size_t j = idx % ob;
    const std::size_t r1 = idx / ob;
    const std::size_t i = r1 % oa;
    const std::size_t r2 = r1 / oa;
    const std::size_t n = r2 % ob;
    const std::size_t m = r2 / ob;
    Real s = goooo_ab[idx];
    const Real* gab = gooov_ab + ((m * ob + n) * oa + i) * vb;
    const Real* tj = t1b + j * vb;
    for (std::size_t e = 0; e < vb; ++e) s += tj[e] * gab[e];
    const Real* gba = gooov_ba + ((n * oa + m) * ob + j) * va;
    const Real* ti = t1a + i * va;
    for (std::size_t e = 0; e < va; ++e) s += ti[e] * gba[e];
    w_bare[idx] = s;
  }
}

// Pack upper-triangle occupied pairs (i < j) for same-spin PPL
template <class Real>
__global__ void pack_occ_lt_kernel(std::size_t o, std::size_t v, const Real* tau_full,
                                   Real* tau_lt) {
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
    tau_lt[idx] = tau_full[(i * o + j) * vv + ab];
  }
}

template <class Real>
__global__ void unpack_occ_lt_add_kernel(std::size_t o, std::size_t v, const Real* l_lt, Real* r2) {
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
    const Real val = l_lt[idx];
    r2[(i * o + j) * vv + ab] += val;
    r2[(j * o + i) * vv + ab] -= val;
  }
}

// High-throughput tiled-Q DF Virtual Ladder (ZERO vvvv materialized!):
// L[p, a, b] += sum_Q sum_{e,f} bt1[Q, a, e] * tau[p, e, f] * bt2[Q, b, f]
template <class Real>
void accumulate_df_ppl_tiled(int dev, cublasHandle_t blas, cudaStream_t st, std::size_t q,
                             std::size_t P, std::size_t v1, std::size_t v2, const Real* bt1,
                             const Real* bt2, const Real* tau_p, Real* L_out,
                             std::size_t tile_budget_bytes) {
  if (!q || !P || !v1 || !v2) return;
  const std::size_t pv1v2 = checked_mul(P, checked_mul(v1, v2));
  const std::size_t tau_bytes = checked_mul(pv1v2, sizeof(Real));
  const std::size_t per_q_elems = checked_add(checked_mul(2, pv1v2), checked_mul(v2, v2));
  const std::size_t per_q_bytes = std::max<std::size_t>(1, checked_mul(per_q_elems, sizeof(Real)));
  if (checked_add(tau_bytes, per_q_bytes) > tile_budget_bytes)
    throw std::length_error("DF-UCCSD v2 CUDA PPL minimum 1-Q tile exceeds budget");
  const std::size_t max_b = (tile_budget_bytes - tau_bytes) / per_q_bytes;
  const std::size_t B_cap = std::min<std::size_t>(q, std::min<std::size_t>(max_b, 128ULL));

  // Pre-transpose tau[p, e, f] -> tau_epf[e, p, f]
  DeviceBuf<Real> tau_epf(dev, pv1v2, st);
  launch_permute3d<Real, 1, 0, 2>(P, v1, v2, tau_p, tau_epf.get(), st);

  DeviceBuf<Real> h1(dev, B_cap * v1 * P * v2, st);
  DeviceBuf<Real> h2(dev, P * v1 * B_cap * v2, st);
  DeviceBuf<Real> bt2_T(dev, v2 * B_cap * v2, st);

  for (std::size_t q0 = 0; q0 < q; q0 += B_cap) {
    const std::size_t B = std::min(B_cap, q - q0);
    // H1[B*v1, P*v2] = bt1_tile[B*v1, v1] @ tau_epf[v1, P*v2]
    rm_gemm<Real>(blas, st, 'N', 'N', B * v1, P * v2, v1, bt1 + q0 * v1 * v1, tau_epf.get(),
                  h1.get(), Real(1), Real(0));
    // H1[B, v1, P, v2] -> H2[P, v1, B, v2]
    launch_permute4d<Real, 2, 1, 0, 3>(B, v1, P, v2, h1.get(), h2.get(), st);
    // bt2_tile[B, b, f] -> bt2_T[b, B, f]
    launch_permute3d<Real, 1, 0, 2>(B, v2, v2, bt2 + q0 * v2 * v2, bt2_T.get(), st);
    // L_out[P*v1, v2] += H2[P*v1, B*v2] @ bt2_T^T[B*v2, v2]
    rm_gemm<Real>(blas, st, 'N', 'T', P * v1, v2, B * v2, h2.get(), bt2_T.get(), L_out, Real(1),
                  Real(1));
  }
}

// Wovvo assembly kernels
template <class Real>
__global__ void wovvo_ss_linear_kernel(std::size_t o, std::size_t v, const Real* govov,
                                       const Real* y_bvv, const Real* gooov, const Real* t1,
                                       Real* W) {
  const std::size_t total = o * v * o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t j = r1 % o;
    const std::size_t r2 = r1 / o;
    const std::size_t e = r2 % v;
    const std::size_t m = r2 / v;
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
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t j = r1 % o;
    const std::size_t r2 = r1 / o;
    const std::size_t f = r2 % v;
    const std::size_t n = r2 / v;
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
    const std::size_t b = idx % vb;
    const std::size_t r1 = idx / vb;
    const std::size_t j = r1 % ob;
    const std::size_t r2 = r1 / ob;
    const std::size_t e = r2 % va;
    const std::size_t m = r2 / va;
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
    const std::size_t b = idx % va;
    const std::size_t r1 = idx / va;
    const std::size_t j = r1 % oa;
    const std::size_t r2 = r1 / oa;
    const std::size_t e = r2 % vb;
    const std::size_t m = r2 / vb;
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
    const std::size_t b = idx % vb;
    const std::size_t r1 = idx / vb;
    const std::size_t j = r1 % oa;
    const std::size_t r2 = r1 / oa;
    const std::size_t e = r2 % vb;
    const std::size_t m = r2 / vb;
    Real val = -govov_ab[((m * vb + b) * oa + j) * vb + e] -
               ya_bvv_b[((m * oa + j) * vb + b) * vb + e];
    for (std::size_t n = 0; n < ob; ++n)
      val += t1b[n * vb + b] * gooov_ab[((m * ob + n) * oa + j) * vb + e];
    W_abba[idx] = val;
  }
}

template <class Real>
__global__ void wovvo_baab_linear_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                         const Real* govov_ba, const Real* yb_bvv_a,
                                         const Real* gooov_ba, const Real* t1a, Real* W_baab) {
  const std::size_t total = ob * va * ob * va;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % va;
    const std::size_t r1 = idx / va;
    const std::size_t j = r1 % ob;
    const std::size_t r2 = r1 / ob;
    const std::size_t e = r2 % va;
    const std::size_t m = r2 / va;
    Real val = -govov_ba[((m * va + b) * ob + j) * va + e] -
               yb_bvv_a[((m * ob + j) * va + b) * va + e];
    for (std::size_t n = 0; n < oa; ++n)
      val += t1a[n * va + b] * gooov_ba[((m * oa + n) * ob + j) * va + e];
    W_baab[idx] = val;
  }
}

template <class Real>
__global__ void build_tab_nf_jb_x_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                         std::size_t vb, const Real* t2ab, const Real* t1a,
                                         const Real* t1b, Real* out) {
  // out[n_b, f_a, j_a, b_b] = 0.5 * t2ab[j, n, f, b] + t1a[j, f] * t1b[n, b]
  const std::size_t total = ob * va * oa * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % vb;
    const std::size_t r1 = idx / vb;
    const std::size_t j = r1 % oa;
    const std::size_t r2 = r1 / oa;
    const std::size_t f = r2 % va;
    const std::size_t n = r2 / va;
    out[idx] = Real(0.5) * t2ab[((j * ob + n) * va + f) * vb + b] + t1a[j * va + f] * t1b[n * vb + b];
  }
}

template <class Real>
__global__ void build_tba_nf_jb_x_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                         std::size_t vb, const Real* t2ab, const Real* t1a,
                                         const Real* t1b, Real* out) {
  // out[n_a, f_b, j_b, b_a] = 0.5 * t2ab[n, j, b, f] + t1b[j, f] * t1a[n, b]
  const std::size_t total = oa * vb * ob * va;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % va;
    const std::size_t r1 = idx / va;
    const std::size_t j = r1 % ob;
    const std::size_t r2 = r1 / ob;
    const std::size_t f = r2 % vb;
    const std::size_t n = r2 / vb;
    out[idx] = Real(0.5) * t2ab[((n * ob + j) * va + b) * vb + f] + t1b[j * vb + f] * t1a[n * va + b];
  }
}

template <class Real>
__global__ void ring_scatter_ss_kernel(std::size_t o, std::size_t v, const Real* ring, Real* r2) {
  const std::size_t total = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t a = r1 % v;
    const std::size_t r2i = r1 / v;
    const std::size_t j = r2i % o;
    const std::size_t i = r2i / o;
    r2[idx] += ring[((i * v + a) * o + j) * v + b] - ring[((j * v + a) * o + i) * v + b] -
               ring[((i * v + b) * o + j) * v + a] + ring[((j * v + b) * o + i) * v + a];
  }
}

template <class Real>
__global__ void ring_scatter_ab_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                       std::size_t vb, const Real* r12, const Real* r2,
                                       const Real* r3, const Real* r4, Real* r2ab) {
  const std::size_t total = oa * ob * va * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % vb;
    const std::size_t r1 = idx / vb;
    const std::size_t a = r1 % va;
    const std::size_t r2i = r1 / va;
    const std::size_t j = r2i % ob;
    const std::size_t i = r2i / ob;
    r2ab[idx] += r12[((i * va + a) * ob + j) * vb + b] + r2[((j * vb + b) * oa + i) * va + a] +
                 r3[((j * va + a) * oa + i) * vb + b] + r4[((i * vb + b) * ob + j) * va + a];
  }
}

template <class Real>
__global__ void r2_ss_xbtgto_kernel(std::size_t o, std::size_t v, const Real* xb, const Real* tg,
                                    const Real* to, Real* r2) {
  const std::size_t total = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t a = r1 % v;
    const std::size_t r2i = r1 / v;
    const std::size_t j = r2i % o;
    const std::size_t i = r2i / o;
    const Real xb_t = xb[((i * v + a) * o + j) * v + b] - xb[((i * v + b) * o + j) * v + a] -
                      xb[((j * v + a) * o + i) * v + b] + xb[((j * v + b) * o + i) * v + a];
    const Real tg_t = tg[((a * v + b) * o + j) * o + i] - tg[((b * v + a) * o + j) * o + i] -
                      tg[((a * v + b) * o + i) * o + j] + tg[((b * v + a) * o + i) * o + j];
    const Real to_t = -to[((i * o + j) * v + a) * v + b] + to[((i * o + j) * v + b) * v + a];
    r2[idx] += xb_t + tg_t + to_t;
  }
}

template <class Real>
__global__ void build_t1_mjb_i_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                      std::size_t vb, const Real* t1a, const Real* goovv_ab,
                                      Real* out) {
  const std::size_t total = oa * ob * vb * oa;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t i = idx % oa;
    const std::size_t r1 = idx / oa;
    const std::size_t b = r1 % vb;
    const std::size_t r2 = r1 / vb;
    const std::size_t j = r2 % ob;
    const std::size_t m = r2 / ob;
    Real s = Real(0);
    for (std::size_t e = 0; e < va; ++e)
      s += t1a[i * va + e] * goovv_ab[((m * ob + j) * va + e) * vb + b];
    out[idx] = s;
  }
}

template <class Real>
__global__ void r2_ab_xbtgto_kernel(std::size_t oa, std::size_t ob, std::size_t va, std::size_t vb,
                                    const Real* xb, const Real* to1, const Real* to2,
                                    const Real* tg1, const Real* u2, const Real* t1b,
                                    const Real* tg3, const Real* tg4, Real* r2ab) {
  const std::size_t total = oa * ob * va * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % vb;
    const std::size_t r1 = idx / vb;
    const std::size_t a = r1 % va;
    const std::size_t r2i = r1 / va;
    const std::size_t j = r2i % ob;
    const std::size_t i = r2i / ob;
    Real tg2 = Real(0);
    for (std::size_t m = 0; m < ob; ++m)
      tg2 += t1b[m * vb + b] * u2[((i * ob + m) * va + a) * ob + j];
    r2ab[idx] += xb[((i * va + a) * ob + j) * vb + b] - to1[idx] -
                 to2[((j * oa + i) * vb + b) * va + a] -
                 tg1[((a * ob + j) * vb + b) * oa + i] - tg2 -
                 tg3[((a * vb + b) * oa + i) * ob + j] - tg4[((b * va + a) * ob + j) * oa + i];
  }
}

// Optional FixedF12Coupling residual kernels
template <class Real>
__global__ void residual_f12_r1_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                       std::size_t vb, const Real* w_vo_aaaa,
                                       const Real* w_vo_bbbb, const Real* w_vo_abab,
                                       const Real* w_vo_abba, Real* r1a, Real* r1b) {
  const std::size_t na = oa * va, nb = ob * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < na + nb; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (idx < na) {
      const std::size_t i = idx / va, a = idx % va;
      Real s = Real(0);
      for (std::size_t j = 0; j < oa; ++j) s += w_vo_aaaa[((i * oa + j) * va + a) * oa + j];
      for (std::size_t j = 0; j < ob; ++j) s += w_vo_abab[((i * ob + j) * va + a) * ob + j];
      r1a[idx] += s;
    } else {
      const std::size_t k = idx - na;
      const std::size_t i = k / vb, a = k % vb;
      Real s = Real(0);
      for (std::size_t j = 0; j < ob; ++j) s += w_vo_bbbb[((i * ob + j) * vb + a) * ob + j];
      for (std::size_t j = 0; j < oa; ++j) s -= w_vo_abba[((j * ob + i) * vb + a) * oa + j];
      r1b[k] += s;
    }
  }
}

template <class Real>
__global__ void residual_f12_r2_ss_kernel(std::size_t o, std::size_t v, const Real* w_vv,
                                          const Real* c_f12, const Real* w_vo, const Real* t1,
                                          Real* r2) {
  const std::size_t total = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t a = r1 % v;
    const std::size_t ij = r1 / v;
    Real s = w_vv[idx] + c_f12[idx];
    for (std::size_t k = 0; k < o; ++k) {
      s += -w_vo[(ij * v + a) * o + k] * t1[k * v + b] + w_vo[(ij * v + b) * o + k] * t1[k * v + a];
    }
    r2[idx] += s;
  }
}

template <class Real>
__global__ void residual_f12_r2_ab_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                          std::size_t vb, const Real* w_vv_abab,
                                          const Real* c_f12_abab, const Real* w_vo_abab,
                                          const Real* w_vo_abba, const Real* t1a, const Real* t1b,
                                          Real* r2ab) {
  const std::size_t total = oa * ob * va * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % vb;
    const std::size_t r1 = idx / vb;
    const std::size_t a = r1 % va;
    const std::size_t ij = r1 / va;
    Real s = w_vv_abab[idx] + c_f12_abab[idx];
    for (std::size_t k = 0; k < ob; ++k) s -= w_vo_abab[(ij * va + a) * ob + k] * t1b[k * vb + b];
    for (std::size_t k = 0; k < oa; ++k) s += w_vo_abba[(ij * vb + b) * oa + k] * t1a[k * va + a];
    r2ab[idx] += s;
  }
}

// Preconditioner & MP2 init kernels
template <class Real>
__global__ void init_mp2_1b_kernel(std::size_t o, std::size_t v, const Real* fov,
                                   const Real* eps_o, const Real* eps_v, Real* t1) {
  const std::size_t total = o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t i = idx / v, a = idx % v;
    t1[idx] = -fov[idx] / (eps_v[a] - eps_o[i]);
  }
}

template <class Real>
__global__ void init_mp2_2b_kernel(std::size_t o1, std::size_t o2, std::size_t v1, std::size_t v2,
                                   const Real* goovv, const Real* eps_o1, const Real* eps_o2,
                                   const Real* eps_v1, const Real* eps_v2, Real* t2) {
  const std::size_t total = o1 * o2 * v1 * v2;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v2;
    const std::size_t r1 = idx / v2;
    const std::size_t a = r1 % v1;
    const std::size_t r2 = r1 / v1;
    const std::size_t j = r2 % o2;
    const std::size_t i = r2 / o2;
    t2[idx] = -goovv[idx] / (eps_v1[a] + eps_v2[b] - eps_o1[i] - eps_o2[j]);
  }
}

template <class Real>
__global__ void precond_step_1b_kernel(std::size_t o, std::size_t v, const Real* res,
                                       const Real* eps_o, const Real* eps_v, Real* step, Real* t1) {
  const std::size_t total = o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t i = idx / v, a = idx % v;
    const Real dt = -res[idx] / (eps_v[a] - eps_o[i]);
    step[idx] = dt;
    t1[idx] += dt;
  }
}

template <class Real>
__global__ void precond_step_2b_kernel(std::size_t o1, std::size_t o2, std::size_t v1,
                                       std::size_t v2, const Real* res, const Real* eps_o1,
                                       const Real* eps_o2, const Real* eps_v1, const Real* eps_v2,
                                       Real* step, Real* t2) {
  const std::size_t total = o1 * o2 * v1 * v2;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v2;
    const std::size_t r1 = idx / v2;
    const std::size_t a = r1 % v1;
    const std::size_t r2 = r1 / v1;
    const std::size_t j = r2 % o2;
    const std::size_t i = r2 / o2;
    const Real dt = -res[idx] / (eps_v1[a] + eps_v2[b] - eps_o1[i] - eps_o2[j]);
    step[idx] = dt;
    t2[idx] += dt;
  }
}

template <class Real>
void apply_jacobi_update(const DeviceSourceView<Real>& src, const DeviceUccAmplitudes<Real>& res,
                         DeviceUccAmplitudes<Real>& step, DeviceUccAmplitudes<Real>& amp,
                         cudaStream_t st) {
  const auto& d = src.dims;
  if (d.oa && d.va) {
    precond_step_1b_kernel<Real><<<grid_1d(d.oa * d.va), 256, 0, st>>>(
        d.oa, d.va, res.t1a(), src.eps_o[0], src.eps_v[0], step.t1a(), amp.t1a());
    precond_step_2b_kernel<Real><<<grid_1d(d.oa * d.oa * d.va * d.va), 256, 0, st>>>(
        d.oa, d.oa, d.va, d.va, res.t2aa(), src.eps_o[0], src.eps_o[0], src.eps_v[0], src.eps_v[0],
        step.t2aa(), amp.t2aa());
  }
  if (d.ob && d.vb) {
    precond_step_1b_kernel<Real><<<grid_1d(d.ob * d.vb), 256, 0, st>>>(
        d.ob, d.vb, res.t1b(), src.eps_o[1], src.eps_v[1], step.t1b(), amp.t1b());
    precond_step_2b_kernel<Real><<<grid_1d(d.ob * d.ob * d.vb * d.vb), 256, 0, st>>>(
        d.ob, d.ob, d.vb, d.vb, res.t2bb(), src.eps_o[1], src.eps_o[1], src.eps_v[1], src.eps_v[1],
        step.t2bb(), amp.t2bb());
  }
  if (d.oa && d.ob && d.va && d.vb) {
    precond_step_2b_kernel<Real><<<grid_1d(d.oa * d.ob * d.va * d.vb), 256, 0, st>>>(
        d.oa, d.ob, d.va, d.vb, res.t2ab(), src.eps_o[0], src.eps_o[1], src.eps_v[0], src.eps_v[1],
        step.t2ab(), amp.t2ab());
  }
  cuda_check(cudaGetLastError(), "jacobi update");
}

// --- Full 5-Block GPU DF-UCCSD Residual ---
template <class Real>
void compute_uccsd_residual_device(const DeviceSourceView<Real>& src,
                                   const DeviceFourIndexCore<Real>& core,
                                   const DeviceUccAmplitudes<Real>& amp,
                                   const DeviceFixedF12Coupling<Real>* f12,
                                   DeviceUccAmplitudes<Real>& res, int dev, cublasHandle_t blas,
                                   cudaStream_t st, std::size_t ppl_tile_budget) {
  const auto& d = src.dims;
  const std::size_t q = d.q, oa = d.oa, ob = d.ob, va = d.va, vb = d.vb;
  res.zero(st);

  // 1. tau (1.0) and tauh (0.5)
  DeviceBuf<Real> tau_aa(dev, oa * oa * va * va, st), tau_bb(dev, ob * ob * vb * vb, st),
      tau_ab(dev, oa * ob * va * vb, st);
  DeviceBuf<Real> tauh_aa(dev, oa * oa * va * va, st), tauh_bb(dev, ob * ob * vb * vb, st),
      tauh_ab(dev, oa * ob * va * vb, st);
  if (oa && va) {
    build_tau_ss_kernel<Real><<<grid_1d(tau_aa.size()), 256, 0, st>>>(
        oa, va, amp.t2aa(), amp.t1a(), Real(1), tau_aa.get());
    build_tau_ss_kernel<Real><<<grid_1d(tauh_aa.size()), 256, 0, st>>>(
        oa, va, amp.t2aa(), amp.t1a(), Real(0.5), tauh_aa.get());
  }
  if (ob && vb) {
    build_tau_ss_kernel<Real><<<grid_1d(tau_bb.size()), 256, 0, st>>>(
        ob, vb, amp.t2bb(), amp.t1b(), Real(1), tau_bb.get());
    build_tau_ss_kernel<Real><<<grid_1d(tauh_bb.size()), 256, 0, st>>>(
        ob, vb, amp.t2bb(), amp.t1b(), Real(0.5), tauh_bb.get());
  }
  if (oa && ob && va && vb) {
    build_tau_ab_kernel<Real><<<grid_1d(tau_ab.size()), 256, 0, st>>>(
        oa, ob, va, vb, amp.t2ab(), amp.t1a(), amp.t1b(), Real(1), tau_ab.get());
    build_tau_ab_kernel<Real><<<grid_1d(tauh_ab.size()), 256, 0, st>>>(
        oa, ob, va, vb, amp.t2ab(), amp.t1a(), amp.t1b(), Real(0.5), tauh_ab.get());
  }
  cuda_check(cudaGetLastError(), "build_tau");

  // 2. DF singles intermediates: j_q, x_a/b, y_a/b, bt_a/b
  DeviceBuf<Real> j_q(dev, q, st);
  j_q.zero(st);
  if (oa && va) {
    rm_gemm<Real>(blas, st, 'N', 'N', q, 1, oa * va, src.bov[0], amp.t1a(), j_q.get(), Real(1),
                  Real(1));
  }
  if (ob && vb) {
    rm_gemm<Real>(blas, st, 'N', 'N', q, 1, ob * vb, src.bov[1], amp.t1b(), j_q.get(), Real(1),
                  Real(1));
  }

  DeviceBuf<Real> x_a(dev, q * oa * va, st), y_a(dev, q * oa * oa, st), bt_a(dev, q * va * va, st);
  DeviceBuf<Real> x_b(dev, q * ob * vb, st), y_b(dev, q * ob * ob, st), bt_b(dev, q * vb * vb, st);
  if (oa && va) {
    rm_gemm_batched<Real>(blas, st, 'N', 'T', oa, va, va, amp.t1a(), 0, src.bvv[0],
                          static_cast<long long>(va * va), x_a.get(),
                          static_cast<long long>(oa * va), q, Real(1), Real(0));
    rm_gemm_batched<Real>(blas, st, 'N', 'T', oa, oa, va, src.bov[0],
                          static_cast<long long>(oa * va), amp.t1a(), 0, y_a.get(),
                          static_cast<long long>(oa * oa), q, Real(1), Real(0));
    bt_a.copy_from_device(src.bvv[0], q * va * va, st);
    rm_gemm_batched<Real>(blas, st, 'T', 'N', va, va, oa, amp.t1a(), 0, src.bov[0],
                          static_cast<long long>(oa * va), bt_a.get(),
                          static_cast<long long>(va * va), q, Real(-1), Real(1));
  }
  if (ob && vb) {
    rm_gemm_batched<Real>(blas, st, 'N', 'T', ob, vb, vb, amp.t1b(), 0, src.bvv[1],
                          static_cast<long long>(vb * vb), x_b.get(),
                          static_cast<long long>(ob * vb), q, Real(1), Real(0));
    rm_gemm_batched<Real>(blas, st, 'N', 'T', ob, ob, vb, src.bov[1],
                          static_cast<long long>(ob * vb), amp.t1b(), 0, y_b.get(),
                          static_cast<long long>(ob * ob), q, Real(1), Real(0));
    bt_b.copy_from_device(src.bvv[1], q * vb * vb, st);
    rm_gemm_batched<Real>(blas, st, 'T', 'N', vb, vb, ob, amp.t1b(), 0, src.bov[1],
                          static_cast<long long>(ob * vb), bt_b.get(),
                          static_cast<long long>(vb * vb), q, Real(-1), Real(1));
  }

  // 3. Intermediates Fvv, Foo, Fov, Fvv_d, Foo_d
  DeviceBuf<Real> Fvv_a(dev, va * va, st), Fvv_b(dev, vb * vb, st);
  Fvv_a.copy_from_device(src.fvv[0], va * va, st);
  Fvv_b.copy_from_device(src.fvv[1], vb * vb, st);
  if (oa && va) {
    rm_gemm<Real>(blas, st, 'T', 'N', va, va, oa, amp.t1a(), src.fov[0], Fvv_a.get(), Real(-0.5),
                  Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', 1, va * va, q, j_q.get(), src.bvv[0], Fvv_a.get(), Real(1),
                  Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', va, va, q * oa, x_a.get(), src.bov[0], Fvv_a.get(), Real(-1),
                  Real(1));
    DeviceBuf<Real> t_tmp(dev, va * oa * oa * va, st), g_tmp(dev, va * oa * oa * va, st);
    launch_permute3d<Real, 1, 0, 2>(oa * oa, va, va, tauh_aa.get(), t_tmp.get(), st);
    launch_permute3d<Real, 1, 0, 2>(oa * oa, va, va, core.goovv_aa.get(), g_tmp.get(), st);
    rm_gemm<Real>(blas, st, 'N', 'T', va, va, oa * oa * va, t_tmp.get(), g_tmp.get(), Fvv_a.get(),
                  Real(-0.5), Real(1));
  }
  if (ob && vb) {
    rm_gemm<Real>(blas, st, 'T', 'N', vb, vb, ob, amp.t1b(), src.fov[1], Fvv_b.get(), Real(-0.5),
                  Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', 1, vb * vb, q, j_q.get(), src.bvv[1], Fvv_b.get(), Real(1),
                  Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', vb, vb, q * ob, x_b.get(), src.bov[1], Fvv_b.get(), Real(-1),
                  Real(1));
    DeviceBuf<Real> t_tmp(dev, vb * ob * ob * vb, st), g_tmp(dev, vb * ob * ob * vb, st);
    launch_permute3d<Real, 1, 0, 2>(ob * ob, vb, vb, tauh_bb.get(), t_tmp.get(), st);
    launch_permute3d<Real, 1, 0, 2>(ob * ob, vb, vb, core.goovv_bb.get(), g_tmp.get(), st);
    rm_gemm<Real>(blas, st, 'N', 'T', vb, vb, ob * ob * vb, t_tmp.get(), g_tmp.get(), Fvv_b.get(),
                  Real(-0.5), Real(1));
  }
  if (oa && ob && va && vb) {
    DeviceBuf<Real> t_tmp(dev, va * oa * ob * vb, st), g_tmp(dev, va * oa * ob * vb, st);
    launch_permute3d<Real, 1, 0, 2>(oa * ob, va, vb, tauh_ab.get(), t_tmp.get(), st);
    launch_permute3d<Real, 1, 0, 2>(oa * ob, va, vb, core.goovv_ab.get(), g_tmp.get(), st);
    rm_gemm<Real>(blas, st, 'N', 'T', va, va, oa * ob * vb, t_tmp.get(), g_tmp.get(), Fvv_a.get(),
                  Real(-1), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', vb, vb, oa * ob * va, tauh_ab.get(), core.goovv_ab.get(),
                  Fvv_b.get(), Real(-1), Real(1));
  }

  DeviceBuf<Real> Foo_a(dev, oa * oa, st), Foo_b(dev, ob * ob, st);
  Foo_a.copy_from_device(src.foo[0], oa * oa, st);
  Foo_b.copy_from_device(src.foo[1], ob * ob, st);
  if (oa && va) {
    rm_gemm<Real>(blas, st, 'N', 'T', oa, oa, va, src.fov[0], amp.t1a(), Foo_a.get(), Real(0.5),
                  Real(1));
    foo_gooov_contract_kernel<Real><<<grid_1d(oa * oa), 256, 0, st>>>(
        oa, oa, va, core.gooov_aa.get(), amp.t1a(), Foo_a.get());
    rm_gemm<Real>(blas, st, 'N', 'T', oa, oa, oa * va * va, core.goovv_aa.get(), tauh_aa.get(),
                  Foo_a.get(), Real(0.5), Real(1));
  }
  if (ob && vb) {
    rm_gemm<Real>(blas, st, 'N', 'T', ob, ob, vb, src.fov[1], amp.t1b(), Foo_b.get(), Real(0.5),
                  Real(1));
    foo_gooov_contract_kernel<Real><<<grid_1d(ob * ob), 256, 0, st>>>(
        ob, ob, vb, core.gooov_bb.get(), amp.t1b(), Foo_b.get());
    rm_gemm<Real>(blas, st, 'N', 'T', ob, ob, ob * vb * vb, core.goovv_bb.get(), tauh_bb.get(),
                  Foo_b.get(), Real(0.5), Real(1));
  }
  if (oa && ob && vb) {
    foo_gooov_contract_kernel<Real><<<grid_1d(oa * oa), 256, 0, st>>>(
        oa, ob, vb, core.gooov_ab.get(), amp.t1b(), Foo_a.get());
  }
  if (ob && oa && va) {
    foo_gooov_contract_kernel<Real><<<grid_1d(ob * ob), 256, 0, st>>>(
        ob, oa, va, core.gooov_ba.get(), amp.t1a(), Foo_b.get());
  }
  if (oa && ob && va && vb) {
    rm_gemm<Real>(blas, st, 'N', 'T', oa, oa, ob * va * vb, core.goovv_ab.get(), tauh_ab.get(),
                  Foo_a.get(), Real(1), Real(1));
    DeviceBuf<Real> g_b(dev, ob * oa * va * vb, st), t_b(dev, ob * oa * va * vb, st);
    launch_permute3d<Real, 1, 0, 2>(oa, ob, va * vb, core.goovv_ab.get(), g_b.get(), st);
    launch_permute3d<Real, 1, 0, 2>(oa, ob, va * vb, tauh_ab.get(), t_b.get(), st);
    rm_gemm<Real>(blas, st, 'N', 'T', ob, ob, oa * va * vb, g_b.get(), t_b.get(), Foo_b.get(),
                  Real(1), Real(1));
  }
  cuda_check(cudaGetLastError(), "Foo_gooov");
  tauh_aa.reset();
  tauh_bb.reset();
  tauh_ab.reset();

  // Reusable permuted goovv views for Fov, r1, and Wovvo
  DeviceBuf<Real> g_menf_aa(dev, oa * va * oa * va, st), g_menf_bb(dev, ob * vb * ob * vb, st),
      gab_menf(dev, oa * va * ob * vb, st);
  if (oa && va)
    launch_permute4d<Real, 0, 2, 1, 3>(oa, oa, va, va, core.goovv_aa.get(), g_menf_aa.get(), st);
  if (ob && vb)
    launch_permute4d<Real, 0, 2, 1, 3>(ob, ob, vb, vb, core.goovv_bb.get(), g_menf_bb.get(), st);
  if (oa && ob && va && vb)
    launch_permute4d<Real, 0, 2, 1, 3>(oa, ob, va, vb, core.goovv_ab.get(), gab_menf.get(), st);

  DeviceBuf<Real> Fov_a(dev, oa * va, st), Fov_b(dev, ob * vb, st);
  Fov_a.copy_from_device(src.fov[0], oa * va, st);
  Fov_b.copy_from_device(src.fov[1], ob * vb, st);
  if (oa && va) {
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, 1, oa * va, g_menf_aa.get(), amp.t1a(), Fov_a.get(),
                  Real(1), Real(1));
  }
  if (ob && vb) {
    rm_gemm<Real>(blas, st, 'N', 'N', ob * vb, 1, ob * vb, g_menf_bb.get(), amp.t1b(), Fov_b.get(),
                  Real(1), Real(1));
  }
  if (oa && ob && va && vb) {
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, 1, ob * vb, gab_menf.get(), amp.t1b(), Fov_a.get(),
                  Real(1), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, 1, oa * va, gab_menf.get(), amp.t1a(), Fov_b.get(),
                  Real(1), Real(1));
  }

  DeviceBuf<Real> Fvv_d_a(dev, va * va, st), Fvv_d_b(dev, vb * vb, st);
  DeviceBuf<Real> Foo_d_a(dev, oa * oa, st), Foo_d_b(dev, ob * ob, st);
  Fvv_d_a.copy_from_device(Fvv_a.get(), va * va, st);
  Fvv_d_b.copy_from_device(Fvv_b.get(), vb * vb, st);
  Foo_d_a.copy_from_device(Foo_a.get(), oa * oa, st);
  Foo_d_b.copy_from_device(Foo_b.get(), ob * ob, st);
  if (oa && va) {
    rm_gemm<Real>(blas, st, 'T', 'N', va, va, oa, amp.t1a(), Fov_a.get(), Fvv_d_a.get(),
                  Real(-0.5), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'T', oa, oa, va, Fov_a.get(), amp.t1a(), Foo_d_a.get(), Real(0.5),
                  Real(1));
  }
  if (ob && vb) {
    rm_gemm<Real>(blas, st, 'T', 'N', vb, vb, ob, amp.t1b(), Fov_b.get(), Fvv_d_b.get(),
                  Real(-0.5), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'T', ob, ob, vb, Fov_b.get(), amp.t1b(), Foo_d_b.get(), Real(0.5),
                  Real(1));
  }

  // Reusable permuted t2 views for r1 and Ring contractions
  DeviceBuf<Real> taa_ia_me(dev, oa * va * oa * va, st), tbb_ia_me(dev, ob * vb * ob * vb, st);
  DeviceBuf<Real> tab_ia_me(dev, oa * va * ob * vb, st), tab_jb_me(dev, ob * vb * oa * va, st);
  if (oa && va)
    launch_permute4d<Real, 0, 2, 1, 3>(oa, oa, va, va, amp.t2aa(), taa_ia_me.get(), st);
  if (ob && vb)
    launch_permute4d<Real, 0, 2, 1, 3>(ob, ob, vb, vb, amp.t2bb(), tbb_ia_me.get(), st);
  if (oa && ob && va && vb) {
    launch_permute4d<Real, 0, 2, 1, 3>(oa, ob, va, vb, amp.t2ab(), tab_ia_me.get(), st);
    launch_permute4d<Real, 1, 3, 0, 2>(oa, ob, va, vb, amp.t2ab(), tab_jb_me.get(), st);
  }

  // 4. Singles residuals r1a, r1b
  if (oa && va) {
    cuda_check(cudaMemcpyAsync(res.t1a(), src.fov[0], oa * va * sizeof(Real),
                               cudaMemcpyDeviceToDevice, st),
               "r1a init");
    rm_gemm<Real>(blas, st, 'N', 'T', oa, va, va, amp.t1a(), Fvv_a.get(), res.t1a(), Real(1),
                  Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', oa, va, oa, Foo_a.get(), amp.t1a(), res.t1a(), Real(-1),
                  Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, 1, oa * va, taa_ia_me.get(), Fov_a.get(), res.t1a(),
                  Real(1), Real(1));
    r1_govov_ss_kernel<Real><<<grid_1d(oa * va), 256, 0, st>>>(oa, va, amp.t1a(),
                                                               core.govov_aa.get(), res.t1a());
    if (ob && vb) {
      rm_gemm<Real>(blas, st, 'N', 'N', oa * va, 1, ob * vb, tab_ia_me.get(), Fov_b.get(),
                    res.t1a(), Real(1), Real(1));
      rm_gemm<Real>(blas, st, 'N', 'N', oa * va, 1, ob * vb, gab_menf.get(), amp.t1b(), res.t1a(),
                    Real(1), Real(1));
    }
    // DF govvv -> r1a (ZERO ovvv)
    {
      DeviceBuf<Real> z_a(dev, q * oa * va, st), t2_ifme(dev, oa * va * oa * va, st);
      launch_permute4d<Real, 0, 3, 1, 2>(oa, oa, va, va, amp.t2aa(), t2_ifme.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', q, oa * va, oa * va, src.bov[0], t2_ifme.get(), z_a.get(),
                    Real(-1), Real(0));
      t2_ifme.reset();
      if (ob && vb) {
        rm_gemm<Real>(blas, st, 'N', 'T', q, oa * va, ob * vb, src.bov[1], tab_ia_me.get(),
                      z_a.get(), Real(1), Real(1));
      }
      DeviceBuf<Real> zb(dev, q * oa * va, st);
      rm_gemm_batched<Real>(blas, st, 'N', 'T', oa, va, va, z_a.get(),
                            static_cast<long long>(oa * va), src.bvv[0],
                            static_cast<long long>(va * va), zb.get(),
                            static_cast<long long>(oa * va), q, Real(1), Real(0));
      reduce_q_sum_kernel<Real><<<grid_1d(oa * va), 256, 0, st>>>(q, oa * va, zb.get(), res.t1a());
    }
    // -0.5 * gooov_aa * t2aa -> r1a
    {
      DeviceBuf<Real> g_p(dev, oa * oa * oa * va, st), t_p(dev, va * oa * oa * va, st);
      launch_permute3d<Real, 1, 0, 2>(oa * oa, oa, va, core.gooov_aa.get(), g_p.get(), st);
      launch_permute3d<Real, 1, 0, 2>(oa * oa, va, va, amp.t2aa(), t_p.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', oa, va, oa * oa * va, g_p.get(), t_p.get(), res.t1a(),
                    Real(-0.5), Real(1));
    }
    if (ob && vb) {
      DeviceBuf<Real> g_p(dev, oa * oa * ob * vb, st), t_p(dev, va * oa * ob * vb, st);
      launch_permute3d<Real, 1, 0, 2>(oa * ob, oa, vb, core.gooov_ab.get(), g_p.get(), st);
      launch_permute3d<Real, 1, 0, 2>(oa * ob, va, vb, amp.t2ab(), t_p.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', oa, va, oa * ob * vb, g_p.get(), t_p.get(), res.t1a(),
                    Real(-1), Real(1));
    }
  }
  if (ob && vb) {
    cuda_check(cudaMemcpyAsync(res.t1b(), src.fov[1], ob * vb * sizeof(Real),
                               cudaMemcpyDeviceToDevice, st),
               "r1b init");
    rm_gemm<Real>(blas, st, 'N', 'T', ob, vb, vb, amp.t1b(), Fvv_b.get(), res.t1b(), Real(1),
                  Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', ob, vb, ob, Foo_b.get(), amp.t1b(), res.t1b(), Real(-1),
                  Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', ob * vb, 1, ob * vb, tbb_ia_me.get(), Fov_b.get(), res.t1b(),
                  Real(1), Real(1));
    r1_govov_ss_kernel<Real><<<grid_1d(ob * vb), 256, 0, st>>>(ob, vb, amp.t1b(),
                                                               core.govov_bb.get(), res.t1b());
    if (oa && va) {
      rm_gemm<Real>(blas, st, 'N', 'N', ob * vb, 1, oa * va, tab_jb_me.get(), Fov_a.get(),
                    res.t1b(), Real(1), Real(1));
      rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, 1, oa * va, gab_menf.get(), amp.t1a(), res.t1b(),
                    Real(1), Real(1));
    }
    // DF govvv -> r1b (ZERO ovvv)
    {
      DeviceBuf<Real> z_b(dev, q * ob * vb, st), t2_ifme(dev, ob * vb * ob * vb, st);
      launch_permute4d<Real, 0, 3, 1, 2>(ob, ob, vb, vb, amp.t2bb(), t2_ifme.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', q, ob * vb, ob * vb, src.bov[1], t2_ifme.get(), z_b.get(),
                    Real(-1), Real(0));
      t2_ifme.reset();
      if (oa && va) {
        rm_gemm<Real>(blas, st, 'N', 'T', q, ob * vb, oa * va, src.bov[0], tab_jb_me.get(),
                      z_b.get(), Real(1), Real(1));
      }
      DeviceBuf<Real> zb(dev, q * ob * vb, st);
      rm_gemm_batched<Real>(blas, st, 'N', 'T', ob, vb, vb, z_b.get(),
                            static_cast<long long>(ob * vb), src.bvv[1],
                            static_cast<long long>(vb * vb), zb.get(),
                            static_cast<long long>(ob * vb), q, Real(1), Real(0));
      reduce_q_sum_kernel<Real><<<grid_1d(ob * vb), 256, 0, st>>>(q, ob * vb, zb.get(), res.t1b());
    }
    // -0.5 * gooov_bb * t2bb - gooov_ba * t2ab -> r1b
    {
      DeviceBuf<Real> g_p(dev, ob * ob * ob * vb, st), t_p(dev, vb * ob * ob * vb, st);
      launch_permute3d<Real, 1, 0, 2>(ob * ob, ob, vb, core.gooov_bb.get(), g_p.get(), st);
      launch_permute3d<Real, 1, 0, 2>(ob * ob, vb, vb, amp.t2bb(), t_p.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'T', ob, vb, ob * ob * vb, g_p.get(), t_p.get(), res.t1b(),
                    Real(-0.5), Real(1));
    }
    if (oa && va) {
      // gooov_ba[m, n, i, e] -> [i, n, m, e] to contract with t2ab[n, m, e, b]
      DeviceBuf<Real> g_p(dev, ob * oa * ob * va, st);
      launch_permute4d<Real, 2, 1, 0, 3>(ob, oa, ob, va, core.gooov_ba.get(), g_p.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'N', ob, vb, oa * ob * va, g_p.get(), amp.t2ab(), res.t1b(),
                    Real(-1), Real(1));
    }
  }
  cuda_check(cudaGetLastError(), "r1 kernels");
  Fvv_a.reset(); Fvv_b.reset(); Foo_a.reset(); Foo_b.reset(); Fov_a.reset(); Fov_b.reset();

  // 5. Doubles residuals base goovv + Fvv_d / Foo_d
  if (oa && va) {
    cuda_check(cudaMemcpyAsync(res.t2aa(), core.goovv_aa.get(),
                               oa * oa * va * va * sizeof(Real), cudaMemcpyDeviceToDevice, st),
               "r2aa init");
    DeviceBuf<Real> tmp(dev, oa * oa * va * va, st);
    rm_gemm<Real>(blas, st, 'N', 'T', oa * oa * va, va, va, amp.t2aa(), Fvv_d_a.get(), tmp.get(),
                  Real(1), Real(0));
    antisym_ab_add_kernel<Real><<<grid_1d(tmp.size()), 256, 0, st>>>(oa, va, tmp.get(), res.t2aa());
    rm_gemm_batched<Real>(blas, st, 'T', 'N', oa, va * va, oa, Foo_d_a.get(), 0, amp.t2aa(),
                          static_cast<long long>(oa * va * va), tmp.get(),
                          static_cast<long long>(oa * va * va), oa, Real(1), Real(0));
    antisym_ij_sub_kernel<Real><<<grid_1d(tmp.size()), 256, 0, st>>>(oa, va, tmp.get(), res.t2aa());
  }
  if (ob && vb) {
    cuda_check(cudaMemcpyAsync(res.t2bb(), core.goovv_bb.get(),
                               ob * ob * vb * vb * sizeof(Real), cudaMemcpyDeviceToDevice, st),
               "r2bb init");
    DeviceBuf<Real> tmp(dev, ob * ob * vb * vb, st);
    rm_gemm<Real>(blas, st, 'N', 'T', ob * ob * vb, vb, vb, amp.t2bb(), Fvv_d_b.get(), tmp.get(),
                  Real(1), Real(0));
    antisym_ab_add_kernel<Real><<<grid_1d(tmp.size()), 256, 0, st>>>(ob, vb, tmp.get(), res.t2bb());
    rm_gemm_batched<Real>(blas, st, 'T', 'N', ob, vb * vb, ob, Foo_d_b.get(), 0, amp.t2bb(),
                          static_cast<long long>(ob * vb * vb), tmp.get(),
                          static_cast<long long>(ob * vb * vb), ob, Real(1), Real(0));
    antisym_ij_sub_kernel<Real><<<grid_1d(tmp.size()), 256, 0, st>>>(ob, vb, tmp.get(), res.t2bb());
  }
  if (oa && ob && va && vb) {
    cuda_check(cudaMemcpyAsync(res.t2ab(), core.goovv_ab.get(),
                               oa * ob * va * vb * sizeof(Real), cudaMemcpyDeviceToDevice, st),
               "r2ab init");
    rm_gemm<Real>(blas, st, 'N', 'T', oa * ob * va, vb, vb, amp.t2ab(), Fvv_d_b.get(), res.t2ab(),
                  Real(1), Real(1));
    rm_gemm_batched<Real>(blas, st, 'N', 'N', va, vb, va, Fvv_d_a.get(), 0, amp.t2ab(),
                          static_cast<long long>(va * vb), res.t2ab(),
                          static_cast<long long>(va * vb), oa * ob, Real(1), Real(1));
    rm_gemm_batched<Real>(blas, st, 'T', 'N', ob, va * vb, ob, Foo_d_b.get(), 0, amp.t2ab(),
                          static_cast<long long>(ob * va * vb), res.t2ab(),
                          static_cast<long long>(ob * va * vb), oa, Real(-1), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', oa, ob * va * vb, oa, Foo_d_a.get(), amp.t2ab(), res.t2ab(),
                  Real(-1), Real(1));
  }
  cuda_check(cudaGetLastError(), "r2 Fvv_d/Foo_d");
  Fvv_d_a.reset(); Fvv_d_b.reset(); Foo_d_a.reset(); Foo_d_b.reset();

  // 6. Woooo_bare + tau_oovv * t2
  if (oa && va) {
    DeviceBuf<Real> w_bare(dev, oa * oa * oa * oa, st);
    build_wbare_ss_kernel<Real><<<grid_1d(w_bare.size()), 256, 0, st>>>(
        oa, va, core.goooo_aa.get(), core.gooov_aa.get(), amp.t1a(), w_bare.get());
    rm_gemm<Real>(blas, st, 'T', 'N', oa * oa, va * va, oa * oa, w_bare.get(), tau_aa.get(),
                  res.t2aa(), Real(0.5), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'T', oa * oa, oa * oa, va * va, core.goovv_aa.get(), tau_aa.get(),
                  w_bare.get(), Real(1), Real(0));
    rm_gemm<Real>(blas, st, 'T', 'N', oa * oa, va * va, oa * oa, w_bare.get(), amp.t2aa(),
                  res.t2aa(), Real(0.25), Real(1));
  }
  if (ob && vb) {
    DeviceBuf<Real> w_bare(dev, ob * ob * ob * ob, st);
    build_wbare_ss_kernel<Real><<<grid_1d(w_bare.size()), 256, 0, st>>>(
        ob, vb, core.goooo_bb.get(), core.gooov_bb.get(), amp.t1b(), w_bare.get());
    rm_gemm<Real>(blas, st, 'T', 'N', ob * ob, vb * vb, ob * ob, w_bare.get(), tau_bb.get(),
                  res.t2bb(), Real(0.5), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'T', ob * ob, ob * ob, vb * vb, core.goovv_bb.get(), tau_bb.get(),
                  w_bare.get(), Real(1), Real(0));
    rm_gemm<Real>(blas, st, 'T', 'N', ob * ob, vb * vb, ob * ob, w_bare.get(), amp.t2bb(),
                  res.t2bb(), Real(0.25), Real(1));
  }
  if (oa && ob && va && vb) {
    DeviceBuf<Real> w_bare(dev, oa * ob * oa * ob, st);
    build_wbare_ab_kernel<Real><<<grid_1d(w_bare.size()), 256, 0, st>>>(
        oa, ob, va, vb, core.goooo_ab.get(), core.gooov_ab.get(), core.gooov_ba.get(), amp.t1a(),
        amp.t1b(), w_bare.get());
    rm_gemm<Real>(blas, st, 'T', 'N', oa * ob, va * vb, oa * ob, w_bare.get(), tau_ab.get(),
                  res.t2ab(), Real(1), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'T', oa * ob, oa * ob, va * vb, core.goovv_ab.get(), tau_ab.get(),
                  w_bare.get(), Real(1), Real(0));
    rm_gemm<Real>(blas, st, 'T', 'N', oa * ob, va * vb, oa * ob, w_bare.get(), amp.t2ab(),
                  res.t2ab(), Real(1), Real(1));
  }
  cuda_check(cudaGetLastError(), "Woooo_bare");

  // 7. On-the-fly T1-dressed DF Virtual Ladder (ZERO vvvv, ZERO ovvv!)
  if (oa >= 2 && va) {
    const std::size_t np = oa * (oa - 1) / 2;
    DeviceBuf<Real> tau_lt(dev, np * va * va, st), l_lt(dev, np * va * va, st);
    l_lt.zero(st);
    pack_occ_lt_kernel<Real><<<grid_1d(tau_lt.size()), 256, 0, st>>>(oa, va, tau_aa.get(),
                                                                     tau_lt.get());
    accumulate_df_ppl_tiled<Real>(dev, blas, st, q, np, va, va, bt_a.get(), bt_a.get(),
                                  tau_lt.get(), l_lt.get(), ppl_tile_budget);
    unpack_occ_lt_add_kernel<Real><<<grid_1d(l_lt.size()), 256, 0, st>>>(oa, va, l_lt.get(),
                                                                        res.t2aa());
  }
  if (ob >= 2 && vb) {
    const std::size_t np = ob * (ob - 1) / 2;
    DeviceBuf<Real> tau_lt(dev, np * vb * vb, st), l_lt(dev, np * vb * vb, st);
    l_lt.zero(st);
    pack_occ_lt_kernel<Real><<<grid_1d(tau_lt.size()), 256, 0, st>>>(ob, vb, tau_bb.get(),
                                                                     tau_lt.get());
    accumulate_df_ppl_tiled<Real>(dev, blas, st, q, np, vb, vb, bt_b.get(), bt_b.get(),
                                  tau_lt.get(), l_lt.get(), ppl_tile_budget);
    unpack_occ_lt_add_kernel<Real><<<grid_1d(l_lt.size()), 256, 0, st>>>(ob, vb, l_lt.get(),
                                                                        res.t2bb());
  }
  if (oa && ob && va && vb) {
    accumulate_df_ppl_tiled<Real>(dev, blas, st, q, oa * ob, va, vb, bt_a.get(), bt_b.get(),
                                  tau_ab.get(), res.t2ab(), ppl_tile_budget);
  }
  cuda_check(cudaGetLastError(), "DF PPL");
  tau_aa.reset(); tau_bb.reset(); tau_ab.reset();
  bt_a.reset(); bt_b.reset();

  // 8. Ring intermediate Wovvo (all 6 spin-conserving blocks) & ring contractions
  DeviceBuf<Real> W_aaaa(dev, oa * va * oa * va, st), W_bbbb(dev, ob * vb * ob * vb, st);
  DeviceBuf<Real> W_abab(dev, oa * va * ob * vb, st), W_baba(dev, ob * vb * oa * va, st);
  DeviceBuf<Real> W_abba(dev, oa * vb * oa * vb, st), W_baab(dev, ob * va * ob * va, st);
  DeviceBuf<Real> ta_nfjb(dev, oa * va * oa * va, st), tb_nfjb(dev, ob * vb * ob * vb, st);

  if (oa && va) {
    rm_gemm<Real>(blas, st, 'T', 'N', oa * va, oa * va, q, src.bov[0], x_a.get(), W_aaaa.get(),
                  Real(1), Real(0));
    DeviceBuf<Real> y_bvv(dev, oa * oa * va * va, st);
    rm_gemm<Real>(blas, st, 'T', 'N', oa * oa, va * va, q, y_a.get(), src.bvv[0], y_bvv.get(),
                  Real(1), Real(0));
    wovvo_ss_linear_kernel<Real><<<grid_1d(W_aaaa.size()), 256, 0, st>>>(
        oa, va, core.govov_aa.get(), y_bvv.get(), core.gooov_aa.get(), amp.t1a(), W_aaaa.get());
    build_t_nfjb_ss_kernel<Real><<<grid_1d(ta_nfjb.size()), 256, 0, st>>>(
        oa, va, amp.t2aa(), amp.t1a(), ta_nfjb.get());
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, oa * va, oa * va, g_menf_aa.get(), ta_nfjb.get(),
                  W_aaaa.get(), Real(-1), Real(1));
    if (ob && vb) {
      rm_gemm<Real>(blas, st, 'N', 'N', oa * va, oa * va, ob * vb, gab_menf.get(), tab_jb_me.get(),
                    W_aaaa.get(), Real(0.5), Real(1));
    }
  }
  if (ob && vb) {
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, ob * vb, q, src.bov[1], x_b.get(), W_bbbb.get(),
                  Real(1), Real(0));
    DeviceBuf<Real> y_bvv(dev, ob * ob * vb * vb, st);
    rm_gemm<Real>(blas, st, 'T', 'N', ob * ob, vb * vb, q, y_b.get(), src.bvv[1], y_bvv.get(),
                  Real(1), Real(0));
    wovvo_ss_linear_kernel<Real><<<grid_1d(W_bbbb.size()), 256, 0, st>>>(
        ob, vb, core.govov_bb.get(), y_bvv.get(), core.gooov_bb.get(), amp.t1b(), W_bbbb.get());
    build_t_nfjb_ss_kernel<Real><<<grid_1d(tb_nfjb.size()), 256, 0, st>>>(
        ob, vb, amp.t2bb(), amp.t1b(), tb_nfjb.get());
    rm_gemm<Real>(blas, st, 'N', 'N', ob * vb, ob * vb, ob * vb, g_menf_bb.get(), tb_nfjb.get(),
                  W_bbbb.get(), Real(-1), Real(1));
    if (oa && va) {
      rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, ob * vb, oa * va, gab_menf.get(), tab_ia_me.get(),
                    W_bbbb.get(), Real(0.5), Real(1));
    }
  }
  if (oa && ob && va && vb) {
    // W_abab & W_baba
    rm_gemm<Real>(blas, st, 'T', 'N', oa * va, ob * vb, q, src.bov[0], x_b.get(), W_abab.get(),
                  Real(1), Real(0));
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, oa * va, q, src.bov[1], x_a.get(), W_baba.get(),
                  Real(1), Real(0));
    wovvo_abab_linear_kernel<Real><<<grid_1d(W_abab.size()), 256, 0, st>>>(
        oa, ob, va, vb, core.goovv_ab.get(), core.gooov_ba.get(), amp.t1b(), W_abab.get());
    wovvo_baba_linear_kernel<Real><<<grid_1d(W_baba.size()), 256, 0, st>>>(
        oa, ob, va, vb, core.goovv_ab.get(), core.gooov_ab.get(), amp.t1a(), W_baba.get());
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, ob * vb, ob * vb, gab_menf.get(), tb_nfjb.get(),
                  W_abab.get(), Real(-1), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, ob * vb, oa * va, g_menf_aa.get(), tab_ia_me.get(),
                  W_abab.get(), Real(0.5), Real(1));
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, oa * va, oa * va, gab_menf.get(), ta_nfjb.get(),
                  W_baba.get(), Real(-1), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', ob * vb, oa * va, ob * vb, g_menf_bb.get(), tab_jb_me.get(),
                  W_baba.get(), Real(0.5), Real(1));

    // W_abba
    {
      DeviceBuf<Real> ya_bvv_b(dev, oa * oa * vb * vb, st);
      rm_gemm<Real>(blas, st, 'T', 'N', oa * oa, vb * vb, q, y_a.get(), src.bvv[1], ya_bvv_b.get(),
                    Real(1), Real(0));
      wovvo_abba_linear_kernel<Real><<<grid_1d(W_abba.size()), 256, 0, st>>>(
          oa, ob, vb, core.govov_ab.get(), ya_bvv_b.get(), core.gooov_ab.get(), amp.t1b(),
          W_abba.get());
      DeviceBuf<Real> gab_x(dev, oa * vb * ob * va, st), tab_x(dev, ob * va * oa * vb, st);
      launch_permute4d<Real, 0, 3, 1, 2>(oa, ob, va, vb, core.goovv_ab.get(), gab_x.get(), st);
      build_tab_nf_jb_x_kernel<Real><<<grid_1d(tab_x.size()), 256, 0, st>>>(
          oa, ob, va, vb, amp.t2ab(), amp.t1a(), amp.t1b(), tab_x.get());
      rm_gemm<Real>(blas, st, 'N', 'N', oa * vb, oa * vb, ob * va, gab_x.get(), tab_x.get(),
                    W_abba.get(), Real(1), Real(1));
    }
    // W_baab
    {
      DeviceBuf<Real> yb_bvv_a(dev, ob * ob * va * va, st);
      rm_gemm<Real>(blas, st, 'T', 'N', ob * ob, va * va, q, y_b.get(), src.bvv[0], yb_bvv_a.get(),
                    Real(1), Real(0));
      wovvo_baab_linear_kernel<Real><<<grid_1d(W_baab.size()), 256, 0, st>>>(
          oa, ob, va, core.govov_ba.get(), yb_bvv_a.get(), core.gooov_ba.get(), amp.t1a(),
          W_baab.get());
      DeviceBuf<Real> gba_x(dev, ob * va * oa * vb, st), tba_x(dev, oa * vb * ob * va, st);
      launch_permute4d<Real, 1, 2, 0, 3>(oa, ob, va, vb, core.goovv_ab.get(), gba_x.get(), st);
      build_tba_nf_jb_x_kernel<Real><<<grid_1d(tba_x.size()), 256, 0, st>>>(
          oa, ob, va, vb, amp.t2ab(), amp.t1a(), amp.t1b(), tba_x.get());
      rm_gemm<Real>(blas, st, 'N', 'N', ob * va, ob * va, oa * vb, gba_x.get(), tba_x.get(),
                    W_baab.get(), Real(1), Real(1));
    }
  }
  cuda_check(cudaGetLastError(), "Wovvo build");
  y_a.reset(); y_b.reset(); ta_nfjb.reset(); tb_nfjb.reset();
  g_menf_aa.reset(); g_menf_bb.reset(); gab_menf.reset();

  // Ring contractions -> r2
  if (oa && va) {
    DeviceBuf<Real> ring(dev, oa * va * oa * va, st);
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, oa * va, oa * va, taa_ia_me.get(), W_aaaa.get(),
                  ring.get(), Real(1), Real(0));
    if (ob && vb) {
      rm_gemm<Real>(blas, st, 'N', 'N', oa * va, oa * va, ob * vb, tab_ia_me.get(), W_baba.get(),
                    ring.get(), Real(1), Real(1));
    }
    ring_scatter_ss_kernel<Real><<<grid_1d(oa * oa * va * va), 256, 0, st>>>(oa, va, ring.get(),
                                                                             res.t2aa());
  }
  if (ob && vb) {
    DeviceBuf<Real> ring(dev, ob * vb * ob * vb, st);
    rm_gemm<Real>(blas, st, 'N', 'N', ob * vb, ob * vb, ob * vb, tbb_ia_me.get(), W_bbbb.get(),
                  ring.get(), Real(1), Real(0));
    if (oa && va) {
      rm_gemm<Real>(blas, st, 'N', 'N', ob * vb, ob * vb, oa * va, tab_jb_me.get(), W_abab.get(),
                    ring.get(), Real(1), Real(1));
    }
    ring_scatter_ss_kernel<Real><<<grid_1d(ob * ob * vb * vb), 256, 0, st>>>(ob, vb, ring.get(),
                                                                             res.t2bb());
  }
  if (oa && ob && va && vb) {
    DeviceBuf<Real> r12(dev, oa * va * ob * vb, st), r2(dev, ob * vb * oa * va, st);
    DeviceBuf<Real> r3(dev, ob * va * oa * vb, st), r4(dev, oa * vb * ob * va, st);
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, ob * vb, oa * va, taa_ia_me.get(), W_abab.get(),
                  r12.get(), Real(1), Real(0));
    rm_gemm<Real>(blas, st, 'N', 'N', oa * va, ob * vb, ob * vb, tab_ia_me.get(), W_bbbb.get(),
                  r12.get(), Real(1), Real(1));
    rm_gemm<Real>(blas, st, 'N', 'N', ob * vb, oa * va, ob * vb, tbb_ia_me.get(), W_baba.get(),
                  r2.get(), Real(1), Real(0));
    rm_gemm<Real>(blas, st, 'N', 'N', ob * vb, oa * va, oa * va, tab_jb_me.get(), W_aaaa.get(),
                  r2.get(), Real(1), Real(1));
    {
      DeviceBuf<Real> tab_ja_me(dev, ob * va * oa * vb, st);
      launch_permute4d<Real, 1, 2, 0, 3>(oa, ob, va, vb, amp.t2ab(), tab_ja_me.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'N', ob * va, oa * vb, oa * vb, tab_ja_me.get(), W_abba.get(),
                    r3.get(), Real(1), Real(0));
    }
    {
      DeviceBuf<Real> tab_ib_me(dev, oa * vb * ob * va, st);
      launch_permute4d<Real, 0, 3, 1, 2>(oa, ob, va, vb, amp.t2ab(), tab_ib_me.get(), st);
      rm_gemm<Real>(blas, st, 'N', 'N', oa * vb, ob * va, ob * va, tab_ib_me.get(), W_baab.get(),
                    r4.get(), Real(1), Real(0));
    }
    ring_scatter_ab_kernel<Real><<<grid_1d(oa * ob * va * vb), 256, 0, st>>>(
        oa, ob, va, vb, r12.get(), r2.get(), r3.get(), r4.get(), res.t2ab());
  }
  cuda_check(cudaGetLastError(), "ring scatter");
  W_aaaa.reset(); W_bbbb.reset(); W_abab.reset(); W_baba.reset(); W_abba.reset(); W_baab.reset();
  taa_ia_me.reset(); tbb_ia_me.reset(); tab_ia_me.reset(); tab_jb_me.reset();

  // 9. xb, tg, to into r2
  if (oa && va) {
    DeviceBuf<Real> xb(dev, oa * va * oa * va, st), u(dev, oa * va * oa * oa, st),
        tg(dev, va * va * oa * oa, st), to(dev, oa * oa * va * va, st);
    rm_gemm<Real>(blas, st, 'T', 'N', oa * va, oa * va, q, x_a.get(), src.bov[0], xb.get(), Real(1),
                  Real(0));
    rm_gemm<Real>(blas, st, 'N', 'T', oa * va * oa, oa, va, core.govov_aa.get(), amp.t1a(), u.get(),
                  Real(1), Real(0));
    rm_gemm<Real>(blas, st, 'T', 'N', va, va * oa * oa, oa, amp.t1a(), u.get(), tg.get(), Real(1),
                  Real(0));
    rm_gemm_batched<Real>(blas, st, 'T', 'N', va, va, oa, amp.t1a(), 0, core.gooov_aa.get(),
                          static_cast<long long>(oa * va), to.get(),
                          static_cast<long long>(va * va), oa * oa, Real(1), Real(0));
    r2_ss_xbtgto_kernel<Real><<<grid_1d(oa * oa * va * va), 256, 0, st>>>(
        oa, va, xb.get(), tg.get(), to.get(), res.t2aa());
  }
  if (ob && vb) {
    DeviceBuf<Real> xb(dev, ob * vb * ob * vb, st), u(dev, ob * vb * ob * ob, st),
        tg(dev, vb * vb * ob * ob, st), to(dev, ob * ob * vb * vb, st);
    rm_gemm<Real>(blas, st, 'T', 'N', ob * vb, ob * vb, q, x_b.get(), src.bov[1], xb.get(), Real(1),
                  Real(0));
    rm_gemm<Real>(blas, st, 'N', 'T', ob * vb * ob, ob, vb, core.govov_bb.get(), amp.t1b(), u.get(),
                  Real(1), Real(0));
    rm_gemm<Real>(blas, st, 'T', 'N', vb, vb * ob * ob, ob, amp.t1b(), u.get(), tg.get(), Real(1),
                  Real(0));
    rm_gemm_batched<Real>(blas, st, 'T', 'N', vb, vb, ob, amp.t1b(), 0, core.gooov_bb.get(),
                          static_cast<long long>(ob * vb), to.get(),
                          static_cast<long long>(vb * vb), ob * ob, Real(1), Real(0));
    r2_ss_xbtgto_kernel<Real><<<grid_1d(ob * ob * vb * vb), 256, 0, st>>>(
        ob, vb, xb.get(), tg.get(), to.get(), res.t2bb());
  }
  if (oa && ob && va && vb) {
    DeviceBuf<Real> xb(dev, oa * va * ob * vb, st);
    rm_gemm<Real>(blas, st, 'T', 'N', oa * va, ob * vb, q, x_a.get(), src.bov[1], xb.get(), Real(1),
                  Real(0));
    rm_gemm<Real>(blas, st, 'T', 'N', oa * va, ob * vb, q, src.bov[0], x_b.get(), xb.get(), Real(1),
                  Real(1));
    DeviceBuf<Real> to1(dev, oa * ob * va * vb, st), to2(dev, ob * oa * vb * va, st);
    rm_gemm_batched<Real>(blas, st, 'T', 'N', va, vb, oa, amp.t1a(), 0, core.gooov_ab.get(),
                          static_cast<long long>(oa * vb), to1.get(),
                          static_cast<long long>(va * vb), oa * ob, Real(1), Real(0));
    rm_gemm_batched<Real>(blas, st, 'T', 'N', vb, va, ob, amp.t1b(), 0, core.gooov_ba.get(),
                          static_cast<long long>(ob * va), to2.get(),
                          static_cast<long long>(vb * va), ob * oa, Real(1), Real(0));
    DeviceBuf<Real> t1_mjb_i(dev, oa * ob * vb * oa, st), tg1(dev, va * ob * vb * oa, st);
    build_t1_mjb_i_kernel<Real><<<grid_1d(t1_mjb_i.size()), 256, 0, st>>>(
        oa, ob, va, vb, amp.t1a(), core.goovv_ab.get(), t1_mjb_i.get());
    rm_gemm<Real>(blas, st, 'T', 'N', va, ob * vb * oa, oa, amp.t1a(), t1_mjb_i.get(), tg1.get(),
                  Real(1), Real(0));
    t1_mjb_i.reset();

    DeviceBuf<Real> u2(dev, oa * ob * va * ob, st);
    rm_gemm<Real>(blas, st, 'N', 'T', oa * ob * va, ob, vb, core.goovv_ab.get(), amp.t1b(),
                  u2.get(), Real(1), Real(0));
    DeviceBuf<Real> u3(dev, oa * vb * oa * ob, st), tg3(dev, va * vb * oa * ob, st);
    rm_gemm<Real>(blas, st, 'N', 'T', oa * vb * oa, ob, vb, core.govov_ab.get(), amp.t1b(),
                  u3.get(), Real(1), Real(0));
    rm_gemm<Real>(blas, st, 'T', 'N', va, vb * oa * ob, oa, amp.t1a(), u3.get(), tg3.get(), Real(1),
                  Real(0));
    u3.reset();
    DeviceBuf<Real> u4(dev, ob * va * ob * oa, st), tg4(dev, vb * va * ob * oa, st);
    rm_gemm<Real>(blas, st, 'N', 'T', ob * va * ob, oa, va, core.govov_ba.get(), amp.t1a(),
                  u4.get(), Real(1), Real(0));
    rm_gemm<Real>(blas, st, 'T', 'N', vb, va * ob * oa, ob, amp.t1b(), u4.get(), tg4.get(), Real(1),
                  Real(0));
    u4.reset();

    r2_ab_xbtgto_kernel<Real><<<grid_1d(oa * ob * va * vb), 256, 0, st>>>(
        oa, ob, va, vb, xb.get(), to1.get(), to2.get(), tg1.get(), u2.get(), amp.t1b(), tg3.get(),
        tg4.get(), res.t2ab());
  }
  cuda_check(cudaGetLastError(), "xbtgto kernels");

  // 10. Optional FixedF12Coupling residual terms
  if (f12) {
    const std::size_t n1 = oa * va + ob * vb;
    if (n1) {
      residual_f12_r1_kernel<Real><<<grid_1d(n1), 256, 0, st>>>(
          oa, ob, va, vb, f12->w_vo_aaaa.get(), f12->w_vo_bbbb.get(), f12->w_vo_abab.get(),
          f12->w_vo_abba.get(), res.t1a(), res.t1b());
    }
    if (oa && va) {
      residual_f12_r2_ss_kernel<Real><<<grid_1d(oa * oa * va * va), 256, 0, st>>>(
          oa, va, f12->w_vv_aaaa.get(), f12->c_f12_aaaa.get(), f12->w_vo_aaaa.get(), amp.t1a(),
          res.t2aa());
    }
    if (ob && vb) {
      residual_f12_r2_ss_kernel<Real><<<grid_1d(ob * ob * vb * vb), 256, 0, st>>>(
          ob, vb, f12->w_vv_bbbb.get(), f12->c_f12_bbbb.get(), f12->w_vo_bbbb.get(), amp.t1b(),
          res.t2bb());
    }
    if (oa && ob && va && vb) {
      residual_f12_r2_ab_kernel<Real><<<grid_1d(oa * ob * va * vb), 256, 0, st>>>(
          oa, ob, va, vb, f12->w_vv_abab.get(), f12->c_f12_abab.get(), f12->w_vo_abab.get(),
          f12->w_vo_abba.get(), amp.t1a(), amp.t1b(), res.t2ab());
    }
    cuda_check(cudaGetLastError(), "f12 residual kernels");
  }
}

void validate_denominators_host(const CudaSpinBlockedSource& src, double thresh) {
  const auto& d = src.dims;
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t a = 0; a < d.va; ++a)
      if (std::abs(src.host_eps_va[a] - src.host_eps_oa[i]) < thresh)
        throw std::runtime_error("DF-UCCSD v2 CUDA small alpha 1-body gap");
  for (std::size_t i = 0; i < d.ob; ++i)
    for (std::size_t a = 0; a < d.vb; ++a)
      if (std::abs(src.host_eps_vb[a] - src.host_eps_ob[i]) < thresh)
        throw std::runtime_error("DF-UCCSD v2 CUDA small beta 1-body gap");
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t j = 0; j < d.oa; ++j)
      for (std::size_t a = 0; a < d.va; ++a)
        for (std::size_t b = 0; b < d.va; ++b)
          if (std::abs(src.host_eps_va[a] + src.host_eps_va[b] - src.host_eps_oa[i] -
                       src.host_eps_oa[j]) < thresh)
            throw std::runtime_error("DF-UCCSD v2 CUDA small aa gap");
  for (std::size_t i = 0; i < d.ob; ++i)
    for (std::size_t j = 0; j < d.ob; ++j)
      for (std::size_t a = 0; a < d.vb; ++a)
        for (std::size_t b = 0; b < d.vb; ++b)
          if (std::abs(src.host_eps_vb[a] + src.host_eps_vb[b] - src.host_eps_ob[i] -
                       src.host_eps_ob[j]) < thresh)
            throw std::runtime_error("DF-UCCSD v2 CUDA small bb gap");
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t j = 0; j < d.ob; ++j)
      for (std::size_t a = 0; a < d.va; ++a)
        for (std::size_t b = 0; b < d.vb; ++b)
          if (std::abs(src.host_eps_va[a] + src.host_eps_vb[b] - src.host_eps_oa[i] -
                       src.host_eps_ob[j]) < thresh)
            throw std::runtime_error("DF-UCCSD v2 CUDA small ab gap");
}

void init_mp2_amplitudes_fp64(const CudaSpinBlockedSource& src,
                              const DeviceFourIndexCore<double>& core,
                              DeviceUccAmplitudes<double>& amp) {
  const auto& d = src.dims;
  const cudaStream_t st = src.stream;
  amp.zero(st);
  if (d.oa && d.va) {
    init_mp2_1b_kernel<double><<<grid_1d(d.oa * d.va), 256, 0, st>>>(
        d.oa, d.va, src.fock.fov_a.get(), src.eps_oa.get(), src.eps_va.get(), amp.t1a());
    init_mp2_2b_kernel<double><<<grid_1d(d.oa * d.oa * d.va * d.va), 256, 0, st>>>(
        d.oa, d.oa, d.va, d.va, core.goovv_aa.get(), src.eps_oa.get(), src.eps_oa.get(),
        src.eps_va.get(), src.eps_va.get(), amp.t2aa());
  }
  if (d.ob && d.vb) {
    init_mp2_1b_kernel<double><<<grid_1d(d.ob * d.vb), 256, 0, st>>>(
        d.ob, d.vb, src.fock.fov_b.get(), src.eps_ob.get(), src.eps_vb.get(), amp.t1b());
    init_mp2_2b_kernel<double><<<grid_1d(d.ob * d.ob * d.vb * d.vb), 256, 0, st>>>(
        d.ob, d.ob, d.vb, d.vb, core.goovv_bb.get(), src.eps_ob.get(), src.eps_ob.get(),
        src.eps_vb.get(), src.eps_vb.get(), amp.t2bb());
  }
  if (d.oa && d.ob && d.va && d.vb) {
    init_mp2_2b_kernel<double><<<grid_1d(d.oa * d.ob * d.va * d.vb), 256, 0, st>>>(
        d.oa, d.ob, d.va, d.vb, core.goovv_ab.get(), src.eps_oa.get(), src.eps_ob.get(),
        src.eps_va.get(), src.eps_vb.get(), amp.t2ab());
  }
  cuda_check(cudaGetLastError(), "init_mp2");
}

}  // namespace

CudaCcsdSolveResult solve_df_uccsd_cuda_impl(const CudaSpinBlockedSource& src,
                                             const DeviceFourIndexCore<double>& core,
                                             const DFUccV2Options& options) {
  const auto& d = src.dims;
  const int dev = src.device;
  const cudaStream_t st = src.stream;
  const cublasHandle_t blas = src.blas;
  runtime::CudaDeviceScope dev_scope(dev);

  validate_denominators_host(src, options.denominator_threshold);

  // Enforce caller/device memory budget BEFORE allocating CCSD amplitudes, F12 tensors,
  // DIIS history, or PPL tiles (never apply hard floors past src.device_cap).
  const auto ov_a = checked_mul(d.oa, d.va), ov_b = checked_mul(d.ob, d.vb);
  const auto amp_elems = checked_add(checked_add(ov_a, ov_b),
      checked_add(checked_mul(ov_a, ov_a), checked_add(checked_mul(ov_a, ov_b), checked_mul(ov_b, ov_b))));
  const auto amp_bytes = checked_mul(amp_elems, sizeof(double));
  const auto f12_elems = ucc_f12_elements(options.f12);
  const auto common = checked_add(checked_add(source_resident_bytes(src), core.total_bytes()),
                                 checked_add(amp_bytes, checked_mul(f12_elems, sizeof(double))));
  const auto fp64 = ucc_iteration_budget(d, sizeof(double), amp_elems,
                                         checked_add(common, checked_mul(2, amp_bytes)), src.device_cap);
  if (!fp64)
    throw UccDeviceLimitError("DF-UCCSD v2 CUDA minimum working set exceeds device memory budget");

  CudaCcsdSolveResult out;
  out.amplitudes.allocate_zero(dev, d, st);
  init_mp2_amplitudes_fp64(src, core, out.amplitudes);

  std::unique_ptr<DeviceFixedF12Coupling<double>> f12_f64;
  if (options.f12) {
    f12_f64 = std::make_unique<DeviceFixedF12Coupling<double>>();
    f12_f64->upload(dev, *options.f12, st);
  }

  // Optional FP32 warmup stage when preconvergence == Mixed or Automatic.
  const bool want_fp32 =
      (options.preconvergence == core::DfPreconvergence::Mixed ||
       options.preconvergence == core::DfPreconvergence::Automatic) &&
      out.amplitudes.total > 0;
  int fp32_iters = 0;
  bool used_fp32_seed = false;
  if (want_fp32) {
    // The FP64 output and F12 upload remain live. Source casts also own Fock and spectra.
    const auto fock_elems = checked_add(checked_add(ov_a, ov_b),
        checked_add(checked_add(checked_mul(d.oa, d.oa), checked_mul(d.ob, d.ob)),
                    checked_add(checked_mul(d.va, d.va), checked_mul(d.vb, d.vb))));
    const auto eps_elems = checked_add(checked_add(d.oa, d.ob), checked_add(d.va, d.vb));
    const auto cast_elems = checked_add(checked_add(fock_elems, eps_elems),
        checked_add(src.factors.total_bytes(), core.total_bytes()) / sizeof(double));
    const auto fp32_base = checked_mul(sizeof(float),
        checked_add(checked_add(cast_elems, f12_elems), checked_mul(4, amp_elems)));
    const auto fp32_cap = std::min({src.device_cap - common, src.device_cap / 2,
                                    query_free_device_bytes(dev) / 3 * 2});
    const auto fp32 = ucc_iteration_budget(d, sizeof(float), amp_elems, fp32_base, fp32_cap);
    if (fp32) {
      const bool completed = try_ucc_warmup([&] {
        DeviceSourceFp32 src_f32;
        src_f32.cast_from(src);
        auto core_f32 = cast_four_index_core<float>(core, dev, d, st);
        std::unique_ptr<DeviceFixedF12Coupling<float>> f12_f32;
        if (options.f12) {
          f12_f32 = std::make_unique<DeviceFixedF12Coupling<float>>();
          f12_f32->upload(dev, *options.f12, st);
        }
        DeviceUccAmplitudes<float> amp_f32, best_f32, res_f32, step_f32;
        amp_f32.allocate_zero(dev, d, st);
        best_f32.allocate_zero(dev, d, st);
        res_f32.allocate_zero(dev, d, st);
        step_f32.allocate_zero(dev, d, st);
        launch_cast(out.amplitudes.flat.get(), amp_f32.flat.get(), out.amplitudes.total, st);
        best_f32.flat.copy_from_device(amp_f32.flat.get(), amp_f32.total, st);

        DeviceDiis<float> diis_f32(dev, blas, st, d, options.diis_space, fp32->diis_bytes);
        const auto sv_f32 = src_f32.view();
        const double switch_tol = std::max(1e-5, 100.0 * options.residual_tolerance);
        const int max_warmup = std::min(40, std::max(8, options.max_ccsd_iterations / 2));
        double best_rmax = std::numeric_limits<double>::infinity();
        int stall = 0;

        {
          CublasMathModeScope math_guard(blas, CUBLAS_PEDANTIC_MATH, true);
          for (int iter = 1; iter <= max_warmup; ++iter) {
            compute_uccsd_residual_device<float>(sv_f32, core_f32, amp_f32, f12_f32.get(), res_f32,
                                                 dev, blas, st, fp32->ppl_tile_bytes);
            const double rmax = device_max_abs<float>(res_f32.flat.get(), res_f32.total, dev, st);
            out.fp32_executed = true;
            fp32_iters = iter;
            if (!std::isfinite(rmax)) break;
            if (rmax < best_rmax) {
              best_rmax = rmax;
              best_f32.flat.copy_from_device(amp_f32.flat.get(), amp_f32.total, st);
              used_fp32_seed = true;
              stall = 0;
            } else {
              ++stall;
            }
            if (rmax <= switch_tol) break;
            if (iter >= 5 && (stall >= 4 || rmax > 1e3 * std::max(1.0, best_rmax))) break;

            apply_jacobi_update<float>(sv_f32, res_f32, step_f32, amp_f32, st);
            diis_f32.push(amp_f32, step_f32);
            (void)diis_f32.extrapolate(amp_f32);
          }
        }

        if (used_fp32_seed && std::isfinite(best_rmax)) {
          launch_cast(best_f32.flat.get(), out.amplitudes.flat.get(), out.amplitudes.total, st);
        }
      });
      used_fp32_seed = used_fp32_seed && completed;
    }
  }

  // Release the four FP32 amplitude buffers and DIIS history before the FP64 projection.
  if (used_fp32_seed) {
    launch_antisymmetrize_t2<double>(d.oa, d.va, out.amplitudes.t2aa(), dev, st);
    launch_antisymmetrize_t2<double>(d.ob, d.vb, out.amplitudes.t2bb(), dev, st);
    cuda_check(cudaStreamSynchronize(st), "fp32 promote sync");
  }

  // Strict FP64 polish (with automatic pure-FP64 restart fallback if FP32 seed diverges).
  const auto sv_f64 = src.view();
  auto run_fp64_polish = [&](int& polish_iters) -> bool {
    DeviceUccAmplitudes<double> res, step;
    res.allocate_zero(dev, d, st);
    step.allocate_zero(dev, d, st);
    DeviceDiis<double> diis(dev, blas, st, d, options.diis_space, fp64->diis_bytes);
    double prev_e =
        compute_uccsd_energy_device<double>(sv_f64, core, out.amplitudes, f12_f64.get(), dev, st);
    if (!std::isfinite(prev_e)) return false;

    for (int iter = 1; iter <= options.max_ccsd_iterations; ++iter) {
      compute_uccsd_residual_device<double>(sv_f64, core, out.amplitudes, f12_f64.get(), res, dev,
                                            blas, st, fp64->ppl_tile_bytes);
      const double rmax = device_max_abs<double>(res.flat.get(), res.total, dev, st);
      const double cur_e =
          compute_uccsd_energy_device<double>(sv_f64, core, out.amplitudes, f12_f64.get(), dev, st);
      polish_iters = iter;
      out.energy = cur_e;
      if (!std::isfinite(rmax) || !std::isfinite(cur_e)) return false;
      if (rmax < options.residual_tolerance && std::abs(cur_e - prev_e) < options.energy_tolerance)
        return true;
      prev_e = cur_e;
      apply_jacobi_update<double>(sv_f64, res, step, out.amplitudes, st);
      diis.push(out.amplitudes, step);
      (void)diis.extrapolate(out.amplitudes);
    }
    return false;
  };

  int polish_iters = 0;
  bool converged = run_fp64_polish(polish_iters);
  if (!converged && used_fp32_seed) {
    // Automatic FP64 fallback from fresh MP2 amplitudes.
    init_mp2_amplitudes_fp64(src, core, out.amplitudes);
    polish_iters = 0;
    converged = run_fp64_polish(polish_iters);
  }
  out.iterations = fp32_iters + polish_iters;
  if (!converged) throw std::runtime_error("DF-UCCSD v2 CUDA solver did not converge");
  return out;
}

}  // namespace generativeqc::cc::v2_detail::cuda_detail
