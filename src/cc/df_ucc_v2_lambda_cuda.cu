#include "cc/df_ucc_v2_lambda_pass_b_cuda.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace generativeqc::cc::v2_detail::cuda_detail {
namespace {

using namespace lambda_vjp_cuda;

// --- CCSD energy gradient kernels (used when !energy_source_only) ---

__global__ void ccsd_egrad_t1_ss_kernel(std::size_t o, std::size_t v, const double* g,
                                        const double* t1, double* bt1) {
  const std::size_t total = o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t i = idx / v, a = idx % v;
    double acc = 0.0;
    for (std::size_t j = 0; j < o; ++j) {
      for (std::size_t b = 0; b < v; ++b) {
        acc += 0.5 * (g[((i * o + j) * v + a) * v + b] + g[((j * o + i) * v + b) * v + a]) *
               t1[j * v + b];
      }
    }
    bt1[idx] += acc;
  }
}

__global__ void ccsd_egrad_t1_ab_a_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                          std::size_t vb, const double* g_ab, const double* t1b,
                                          double* bt1a) {
  const std::size_t total = oa * va;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t i = idx / va, a = idx % va;
    double acc = 0.0;
    for (std::size_t j = 0; j < ob; ++j) {
      for (std::size_t b = 0; b < vb; ++b) {
        acc += g_ab[((i * ob + j) * va + a) * vb + b] * t1b[j * vb + b];
      }
    }
    bt1a[idx] += acc;
  }
}

__global__ void ccsd_egrad_t1_ab_b_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                          std::size_t vb, const double* g_ab, const double* t1a,
                                          double* bt1b) {
  const std::size_t total = ob * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t j = idx / vb, b = idx % vb;
    double acc = 0.0;
    for (std::size_t i = 0; i < oa; ++i) {
      for (std::size_t a = 0; a < va; ++a) {
        acc += g_ab[((i * ob + j) * va + a) * vb + b] * t1a[i * va + a];
      }
    }
    bt1b[idx] += acc;
  }
}

__global__ void ccsd_egrad_goovv_ss_kernel(std::size_t o, std::size_t v, const double* t2,
                                           const double* t1, double* bc) {
  const std::size_t total = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t a = r1 % v;
    const std::size_t r2 = r1 / v;
    const std::size_t j = r2 % o, i = r2 / o;
    bc[idx] += 0.25 * t2[idx] + 0.5 * t1[i * v + a] * t1[j * v + b];
  }
}

__global__ void ccsd_egrad_goovv_ab_kernel(std::size_t oa, std::size_t ob, std::size_t va,
                                           std::size_t vb, const double* t2ab, const double* t1a,
                                           const double* t1b, double* bc) {
  const std::size_t total = oa * ob * va * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % vb;
    const std::size_t r1 = idx / vb;
    const std::size_t a = r1 % va;
    const std::size_t r2 = r1 / va;
    const std::size_t j = r2 % ob, i = r2 / ob;
    bc[idx] += t2ab[idx] + t1a[i * va + a] * t1b[j * vb + b];
  }
}

__global__ void f12_egrad_1b_kernel(std::size_t oa, std::size_t ob, std::size_t va, std::size_t vb,
                                    const double* w_vo_aaaa, const double* w_vo_bbbb,
                                    const double* w_vo_abab, const double* w_vo_abba, double* bt1a,
                                    double* bt1b) {
  const std::size_t na = oa * va, nb = ob * vb;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < na + nb; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (idx < na) {
      const std::size_t i = idx / va, a = idx % va;
      double r1 = 0.0;
      for (std::size_t j = 0; j < oa; ++j) r1 += w_vo_aaaa[((i * oa + j) * va + a) * oa + j];
      for (std::size_t j = 0; j < ob; ++j) r1 += w_vo_abab[((i * ob + j) * va + a) * ob + j];
      bt1a[idx] += r1;
    } else {
      const std::size_t k = idx - na;
      const std::size_t i = k / vb, a = k % vb;
      double r1 = 0.0;
      for (std::size_t j = 0; j < ob; ++j) r1 += w_vo_bbbb[((i * ob + j) * vb + a) * ob + j];
      for (std::size_t j = 0; j < oa; ++j) r1 -= w_vo_abba[((j * ob + i) * vb + a) * oa + j];
      bt1b[k] += r1;
    }
  }
}

void accumulate_ccsd_energy_grad_device(const DeviceSourceView<double>& src,
                                        const DeviceFourIndexCore<double>& core,
                                        const DeviceUccAmplitudes<double>& amp,
                                        const DeviceFixedF12Coupling<double>* f12,
                                        DeviceUccAmplitudes<double>* bar_t, DeviceFockGrads* bar_f,
                                        DeviceFourIndexCore<double>* bar_core,
                                        DeviceFixedF12Cotangents* bar_f12, cudaStream_t st) {
  const auto& d = src.dims;
  const std::size_t oa = d.oa, ob = d.ob, va = d.va, vb = d.vb;

  if (bar_t) {
    launch_add_scaled(bar_t->t1a(), src.fov[0], 1.0, oa * va, st);
    launch_add_scaled(bar_t->t1b(), src.fov[1], 1.0, ob * vb, st);
    for (int s = 0; s < 2; ++s) {
      const std::size_t o = src.o(s), v = src.v(s);
      if (!o || !v) continue;
      launch_add_scaled(bar_t->t2(s), core.goovv(s), 0.25, o * o * v * v, st);
      ccsd_egrad_t1_ss_kernel<<<grid_1d(o * v), 256, 0, st>>>(o, v, core.goovv(s), amp.t1(s),
                                                              bar_t->t1(s));
    }
    if (oa && ob && va && vb) {
      launch_add_scaled(bar_t->t2ab(), core.goovv_ab.get(), 1.0, oa * ob * va * vb, st);
      ccsd_egrad_t1_ab_a_kernel<<<grid_1d(oa * va), 256, 0, st>>>(
          oa, ob, va, vb, core.goovv_ab.get(), amp.t1b(), bar_t->t1a());
      ccsd_egrad_t1_ab_b_kernel<<<grid_1d(ob * vb), 256, 0, st>>>(
          oa, ob, va, vb, core.goovv_ab.get(), amp.t1a(), bar_t->t1b());
    }
  }
  if (bar_f) {
    launch_add_scaled(bar_f->fov_a.get(), amp.t1a(), 1.0, oa * va, st);
    launch_add_scaled(bar_f->fov_b.get(), amp.t1b(), 1.0, ob * vb, st);
  }
  if (bar_core) {
    for (int s = 0; s < 2; ++s) {
      const std::size_t o = src.o(s), v = src.v(s);
      if (o && v)
        ccsd_egrad_goovv_ss_kernel<<<grid_1d(o * o * v * v), 256, 0, st>>>(
            o, v, amp.t2(s), amp.t1(s), bar_core->goovv(s));
    }
    if (oa && ob && va && vb)
      ccsd_egrad_goovv_ab_kernel<<<grid_1d(oa * ob * va * vb), 256, 0, st>>>(
          oa, ob, va, vb, amp.t2ab(), amp.t1a(), amp.t1b(), bar_core->goovv_ab.get());
  }
  if (f12) {
    if (bar_f12) {
      bar_f12->bar_fixed_geminal_energy += 1.0;
      const std::size_t n_wvo = oa * oa * va + oa * ob * va + ob * ob * vb + ob * oa * vb;
      if (n_wvo) {
        f12_r1_wvo_cotangent_kernel<double><<<grid_1d(n_wvo), 256, 0, st>>>(
            oa, ob, va, vb, amp.t1a(), amp.t1b(), bar_f12->bar_w_vo_aaaa.get(),
            bar_f12->bar_w_vo_abab.get(), bar_f12->bar_w_vo_bbbb.get(),
            bar_f12->bar_w_vo_abba.get());
      }
      if (oa && va) {
        launch_add_scaled(bar_f12->bar_c_f12_aaaa.get(), amp.t2aa(), 0.25, oa * oa * va * va, st);
        ccsd_egrad_goovv_ss_kernel<<<grid_1d(oa * oa * va * va), 256, 0, st>>>(
            oa, va, amp.t2aa(), amp.t1a(), bar_f12->bar_w_vv_aaaa.get());
      }
      if (ob && vb) {
        launch_add_scaled(bar_f12->bar_c_f12_bbbb.get(), amp.t2bb(), 0.25, ob * ob * vb * vb, st);
        ccsd_egrad_goovv_ss_kernel<<<grid_1d(ob * ob * vb * vb), 256, 0, st>>>(
            ob, vb, amp.t2bb(), amp.t1b(), bar_f12->bar_w_vv_bbbb.get());
      }
      if (oa && ob && va && vb) {
        launch_add_scaled(bar_f12->bar_c_f12_abab.get(), amp.t2ab(), 1.0, oa * ob * va * vb, st);
        ccsd_egrad_goovv_ab_kernel<<<grid_1d(oa * ob * va * vb), 256, 0, st>>>(
            oa, ob, va, vb, amp.t2ab(), amp.t1a(), amp.t1b(), bar_f12->bar_w_vv_abab.get());
      }
    }
    if (bar_t) {
      if (oa * va + ob * vb) {
        f12_egrad_1b_kernel<<<grid_1d(oa * va + ob * vb), 256, 0, st>>>(
            oa, ob, va, vb, f12->w_vo_aaaa.get(), f12->w_vo_bbbb.get(), f12->w_vo_abab.get(),
            f12->w_vo_abba.get(), bar_t->t1a(), bar_t->t1b());
      }
      if (oa && va) {
        launch_add_scaled(bar_t->t2aa(), f12->c_f12_aaaa.get(), 0.25, oa * oa * va * va, st);
        launch_add_scaled(bar_t->t2aa(), f12->w_vv_aaaa.get(), 0.25, oa * oa * va * va, st);
        ccsd_egrad_t1_ss_kernel<<<grid_1d(oa * va), 256, 0, st>>>(oa, va, f12->w_vv_aaaa.get(),
                                                                  amp.t1a(), bar_t->t1a());
      }
      if (ob && vb) {
        launch_add_scaled(bar_t->t2bb(), f12->c_f12_bbbb.get(), 0.25, ob * ob * vb * vb, st);
        launch_add_scaled(bar_t->t2bb(), f12->w_vv_bbbb.get(), 0.25, ob * ob * vb * vb, st);
        ccsd_egrad_t1_ss_kernel<<<grid_1d(ob * vb), 256, 0, st>>>(ob, vb, f12->w_vv_bbbb.get(),
                                                                  amp.t1b(), bar_t->t1b());
      }
      if (oa && ob && va && vb) {
        launch_add_scaled(bar_t->t2ab(), f12->c_f12_abab.get(), 1.0, oa * ob * va * vb, st);
        launch_add_scaled(bar_t->t2ab(), f12->w_vv_abab.get(), 1.0, oa * ob * va * vb, st);
        ccsd_egrad_t1_ab_a_kernel<<<grid_1d(oa * va), 256, 0, st>>>(
            oa, ob, va, vb, f12->w_vv_abab.get(), amp.t1b(), bar_t->t1a());
        ccsd_egrad_t1_ab_b_kernel<<<grid_1d(ob * vb), 256, 0, st>>>(
            oa, ob, va, vb, f12->w_vv_abab.get(), amp.t1a(), bar_t->t1b());
      }
    }
  }
  cuda_check(cudaGetLastError(), "lambda ccsd energy grad");
}

// --- Lambda Jacobi preconditioner kernels ---

template <class Real>
__global__ void lambda_precond_1b_kernel(std::size_t o, std::size_t v, const Real* res,
                                         const Real* eps_o, const Real* eps_v, Real* step,
                                         Real* vec) {
  const std::size_t total = o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t i = idx / v, a = idx % v;
    const Real dl = -res[idx] / (eps_v[a] - eps_o[i]);
    step[idx] = dl;
    vec[idx] += dl;
  }
}

template <class Real>
__global__ void lambda_precond_2b_kernel(std::size_t o1, std::size_t o2, std::size_t v1,
                                         std::size_t v2, const Real* res, const Real* eps_o1,
                                         const Real* eps_o2, const Real* eps_v1, const Real* eps_v2,
                                         Real* step, Real* vec) {
  const std::size_t total = o1 * o2 * v1 * v2;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v2;
    const std::size_t r1 = idx / v2;
    const std::size_t a = r1 % v1;
    const std::size_t r2 = r1 / v1;
    const std::size_t j = r2 % o2, i = r2 / o2;
    const Real dl = -res[idx] / (eps_v1[a] + eps_v2[b] - eps_o1[i] - eps_o2[j]);
    step[idx] = dl;
    vec[idx] += dl;
  }
}

template <class Real>
void apply_lambda_preconditioner(const DeviceSourceView<Real>& src,
                                 const DeviceUccAmplitudes<Real>& res,
                                 DeviceUccAmplitudes<Real>& step, DeviceUccAmplitudes<Real>& vec,
                                 cudaStream_t st) {
  const auto& d = src.dims;
  if (d.oa && d.va) {
    lambda_precond_1b_kernel<Real><<<grid_1d(d.oa * d.va), 256, 0, st>>>(
        d.oa, d.va, res.t1a(), src.eps_o[0], src.eps_v[0], step.t1a(), vec.t1a());
    lambda_precond_2b_kernel<Real><<<grid_1d(d.oa * d.oa * d.va * d.va), 256, 0, st>>>(
        d.oa, d.oa, d.va, d.va, res.t2aa(), src.eps_o[0], src.eps_o[0], src.eps_v[0], src.eps_v[0],
        step.t2aa(), vec.t2aa());
  }
  if (d.ob && d.vb) {
    lambda_precond_1b_kernel<Real><<<grid_1d(d.ob * d.vb), 256, 0, st>>>(
        d.ob, d.vb, res.t1b(), src.eps_o[1], src.eps_v[1], step.t1b(), vec.t1b());
    lambda_precond_2b_kernel<Real><<<grid_1d(d.ob * d.ob * d.vb * d.vb), 256, 0, st>>>(
        d.ob, d.ob, d.vb, d.vb, res.t2bb(), src.eps_o[1], src.eps_o[1], src.eps_v[1], src.eps_v[1],
        step.t2bb(), vec.t2bb());
  }
  if (d.oa && d.ob && d.va && d.vb) {
    lambda_precond_2b_kernel<Real><<<grid_1d(d.oa * d.ob * d.va * d.vb), 256, 0, st>>>(
        d.oa, d.ob, d.va, d.vb, res.t2ab(), src.eps_o[0], src.eps_o[1], src.eps_v[0], src.eps_v[1],
        step.t2ab(), vec.t2ab());
  }
  cuda_check(cudaGetLastError(), "lambda preconditioner");
}

void validate_lambda_denominators_host(const CudaSpinBlockedSource& src, double thresh) {
  const auto& d = src.dims;
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t a = 0; a < d.va; ++a)
      if (std::abs(src.host_eps_va[a] - src.host_eps_oa[i]) < thresh)
        throw std::runtime_error("DF-ULambda v2 CUDA small alpha 1-body gap");
  for (std::size_t i = 0; i < d.ob; ++i)
    for (std::size_t a = 0; a < d.vb; ++a)
      if (std::abs(src.host_eps_vb[a] - src.host_eps_ob[i]) < thresh)
        throw std::runtime_error("DF-ULambda v2 CUDA small beta 1-body gap");
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t j = 0; j < d.oa; ++j)
      for (std::size_t a = 0; a < d.va; ++a)
        for (std::size_t b = 0; b < d.va; ++b)
          if (std::abs(src.host_eps_va[a] + src.host_eps_va[b] - src.host_eps_oa[i] -
                       src.host_eps_oa[j]) < thresh)
            throw std::runtime_error("DF-ULambda v2 CUDA small aa gap");
  for (std::size_t i = 0; i < d.ob; ++i)
    for (std::size_t j = 0; j < d.ob; ++j)
      for (std::size_t a = 0; a < d.vb; ++a)
        for (std::size_t b = 0; b < d.vb; ++b)
          if (std::abs(src.host_eps_vb[a] + src.host_eps_vb[b] - src.host_eps_ob[i] -
                       src.host_eps_ob[j]) < thresh)
            throw std::runtime_error("DF-ULambda v2 CUDA small bb gap");
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t j = 0; j < d.ob; ++j)
      for (std::size_t a = 0; a < d.va; ++a)
        for (std::size_t b = 0; b < d.vb; ++b)
          if (std::abs(src.host_eps_va[a] + src.host_eps_vb[b] - src.host_eps_oa[i] -
                       src.host_eps_ob[j]) < thresh)
            throw std::runtime_error("DF-ULambda v2 CUDA small ab gap");
}

std::size_t estimate_forward_cache_bytes(const SpinBlockedDims& d, std::size_t elem_bytes) {
  const std::size_t oa = d.oa, ob = d.ob, va = d.va, vb = d.vb, q = d.q;
  const std::size_t o2v2 = oa * oa * va * va + ob * ob * vb * vb + oa * ob * va * vb;
  const std::size_t wovvo = oa * va * oa * va + ob * vb * ob * vb + 2 * oa * va * ob * vb +
                            oa * vb * oa * vb + ob * va * ob * va;
  const std::size_t q_terms =
      q * (2 * oa * va + 2 * ob * vb + oa * oa + ob * ob + va * va + vb * vb);
  const std::size_t oooo = 2 * (oa * oa * oa * oa + ob * ob * ob * ob + oa * ob * oa * ob);
  const std::size_t u_terms = oa * va * oa * oa + ob * vb * ob * ob + oa * ob * vb * oa +
                              oa * ob * va * ob + oa * vb * oa * ob + ob * va * ob * oa;
  return (o2v2 + wovvo + q_terms + oooo + u_terms) * elem_bytes;
}

template <class Real>
void evaluate_lambda_vjp_device(
    const DeviceForwardCache<Real>& cache, const DeviceSourceView<Real>& src,
    const DeviceFourIndexCore<Real>& core, const DeviceUccAmplitudes<Real>& amp,
    const DeviceFixedF12Coupling<Real>* f12, const DeviceUccAmplitudes<Real>& lam,
    DeviceUccAmplitudes<Real>& bar_t, DeviceFockGrads* bar_f, DeviceFactorGrads* bar_b,
    DeviceFourIndexCore<double>* bar_core, DeviceFixedF12Cotangents* bar_f12, int dev,
    cublasHandle_t blas, cudaStream_t st, std::size_t ppl_tile_budget) {
  DeviceLambdaIntermediates<Real> inter;
  evaluate_lambda_pass_a_cuda<Real>(cache, src, core, amp, f12, lam, bar_t, inter, bar_f, bar_b,
                                    bar_core, bar_f12, dev, blas, st, ppl_tile_budget);
  evaluate_lambda_pass_b_cuda<Real>(cache, src, core, amp, inter, bar_t, bar_f, bar_b, bar_core,
                                    dev, blas, st);
}

void init_lambda_from_rhs_fp64(const DeviceSourceView<double>& sv,
                               const DeviceUccAmplitudes<double>& rhs,
                               DeviceUccAmplitudes<double>& lam, int dev, cudaStream_t st) {
  DeviceUccAmplitudes<double> init_step;
  init_step.allocate_zero(dev, sv.dims, st);
  lam.zero(st);
  apply_lambda_preconditioner<double>(sv, rhs, init_step, lam, st);
  launch_antisymmetrize_t2<double>(sv.dims.oa, sv.dims.va, lam.t2aa(), dev, st);
  launch_antisymmetrize_t2<double>(sv.dims.ob, sv.dims.vb, lam.t2bb(), dev, st);
}

}  // namespace

CudaLambdaSolveResult solve_df_ulambda_cuda_impl(const CudaSpinBlockedSource& src,
                                                 const DeviceFourIndexCore<double>& core,
                                                 const DeviceUccAmplitudes<double>& amp,
                                                 const CudaTriplesResult* triples,
                                                 const DFUccV2Options& options) {
  const auto& d = src.dims;
  const int dev = src.device;
  const cudaStream_t st = src.stream;
  const cublasHandle_t blas = src.blas;
  runtime::CudaDeviceScope dev_scope(dev);

  validate_lambda_denominators_host(src, options.denominator_threshold);

  CudaLambdaSolveResult out;
  out.bar_b.allocate_zero(dev, d, st);
  out.bar_f.allocate_zero(dev, d, st);

  std::unique_ptr<DeviceFixedF12Coupling<double>> f12_f64;
  std::unique_ptr<DeviceFixedF12Cotangents> d_bar_f12;
  if (options.f12) {
    f12_f64 = std::make_unique<DeviceFixedF12Coupling<double>>();
    f12_f64->upload(dev, *options.f12, st);
    d_bar_f12 = std::make_unique<DeviceFixedF12Cotangents>();
    d_bar_f12->allocate_zero(dev, *options.f12, st);
  }

  const auto sv_f64 = src.view();
  DeviceUccAmplitudes<double> rhs;
  rhs.allocate_zero(dev, d, st);
  if (triples) {
    launch_add_scaled(rhs.flat.get(), triples->bar_t.flat.get(), 1.0, rhs.total, st);
  }
  if (!options.energy_source_only) {
    accumulate_ccsd_energy_grad_device(sv_f64, core, amp, f12_f64.get(), &rhs, nullptr, nullptr,
                                       nullptr, st);
  }
  launch_antisymmetrize_t2<double>(d.oa, d.va, rhs.t2aa(), dev, st);
  launch_antisymmetrize_t2<double>(d.ob, d.vb, rhs.t2bb(), dev, st);

  const double rhs_max = device_max_abs<double>(rhs.flat.get(), rhs.total, dev, st);
  if (!std::isfinite(rhs_max))
    throw std::runtime_error("DF-ULambda v2 CUDA RHS became nonfinite");

  const auto ppl_budget = src.device_cap; // The tile builder queries live headroom at allocation.
  const auto diis_budget = ucc_device_available(src.device_cap) / 3;

  DeviceUccAmplitudes<double> lam_f64;
  lam_f64.allocate_zero(dev, d, st);
  std::unique_ptr<DeviceForwardCache<double>> cache_f64;

  if (rhs_max > 0.0) {
    init_lambda_from_rhs_fp64(sv_f64, rhs, lam_f64, dev, st);

    // Optional FP32 warmup stage when preconvergence == Mixed or Automatic.
    // All FP32 cache/DIIS buffers are released BEFORE allocating FP64 cache.
    const bool want_fp32 =
        (options.preconvergence == core::DfPreconvergence::Mixed ||
         options.preconvergence == core::DfPreconvergence::Automatic) &&
        lam_f64.total > 0;
    int fp32_iters = 0;
    bool used_fp32_seed = false;
    if (want_fp32) {
      const auto headroom = ucc_device_available(src.device_cap);
      const auto fp32_cap = std::min({headroom, src.device_cap / 2,
                                      query_free_device_bytes(dev) / 3 * 2});
      // The cache estimate is a fast optional-stage filter. Charge every allocation during
      // warmup since the subsequent transpose/adjoint workspaces depend on the active spins.
      const auto fp32_est = checked_add(checked_add(src.factors.total_bytes(), core.total_bytes()) / 2,
          checked_add(estimate_forward_cache_bytes(d, sizeof(float)),
                      checked_mul(sizeof(float), checked_add(ucc_f12_elements(options.f12),
                                                             checked_mul(8, lam_f64.total)))));
      if (fp32_est <= fp32_cap) {
        const bool completed = try_ucc_warmup([&] {
          UccDeviceCharge withheld(headroom - fp32_cap);
          DeviceSourceFp32 src_f32;
          src_f32.cast_from(src);
          auto core_f32 = cast_four_index_core<float>(core, dev, d, st);
          std::unique_ptr<DeviceFixedF12Coupling<float>> f12_f32;
          if (options.f12) {
            f12_f32 = std::make_unique<DeviceFixedF12Coupling<float>>();
            f12_f32->upload(dev, *options.f12, st);
          }
          DeviceUccAmplitudes<float> amp_f32, rhs_f32, lam_f32, best_f32, res_f32, step_f32;
          amp_f32.allocate_zero(dev, d, st);
          rhs_f32.allocate_zero(dev, d, st);
          lam_f32.allocate_zero(dev, d, st);
          best_f32.allocate_zero(dev, d, st);
          res_f32.allocate_zero(dev, d, st);
          step_f32.allocate_zero(dev, d, st);
          launch_cast(amp.flat.get(), amp_f32.flat.get(), amp.total, st);
          launch_cast(rhs.flat.get(), rhs_f32.flat.get(), rhs.total, st);
          launch_cast(lam_f64.flat.get(), lam_f32.flat.get(), lam_f64.total, st);
          best_f32.flat.copy_from_device(lam_f32.flat.get(), lam_f32.total, st);

          const auto sv_f32 = src_f32.view();
          const auto cache_f32 =
              build_forward_cache_device<float>(sv_f32, core_f32, amp_f32, dev, blas, st);
          DeviceDiis<float> diis_f32(dev, blas, st, d, options.diis_space, diis_budget / 2);

          const double switch_tol = std::max(1e-5, 100.0 * options.lambda_tolerance);
          const int max_warmup = std::min(40, std::max(8, options.max_lambda_iterations / 2));
          double best_rmax = std::numeric_limits<double>::infinity();
          int stall = 0;

          {
            CublasMathModeScope math_guard(blas, CUBLAS_PEDANTIC_MATH, true);
            for (int iter = 1; iter <= max_warmup; ++iter) {
              evaluate_lambda_vjp_device<float>(cache_f32, sv_f32, core_f32, amp_f32, f12_f32.get(),
                                                lam_f32, res_f32, nullptr, nullptr, nullptr, nullptr,
                                                dev, blas, st, ppl_budget);
              launch_add_scaled(res_f32.flat.get(), rhs_f32.flat.get(), 1.0f, res_f32.total, st);
              const double rmax = device_max_abs<float>(res_f32.flat.get(), res_f32.total, dev, st);
              out.fp32_executed = true;
              fp32_iters = iter;
              if (!std::isfinite(rmax)) break;
              if (rmax < best_rmax) {
                best_rmax = rmax;
                best_f32.flat.copy_from_device(lam_f32.flat.get(), lam_f32.total, st);
                used_fp32_seed = true;
                stall = 0;
              } else {
                ++stall;
              }
              if (rmax <= switch_tol) break;
              if (iter >= 5 && (stall >= 4 || rmax > 1e3 * std::max(1.0, best_rmax))) break;

              apply_lambda_preconditioner<float>(sv_f32, res_f32, step_f32, lam_f32, st);
              diis_f32.push(lam_f32, step_f32);
              (void)diis_f32.extrapolate(lam_f32);
            }
          }

          if (used_fp32_seed && std::isfinite(best_rmax)) {
            launch_cast(best_f32.flat.get(), lam_f64.flat.get(), lam_f64.total, st);
          }
        });
        used_fp32_seed = used_fp32_seed && completed;
      }
    }
    if (used_fp32_seed) {
      launch_antisymmetrize_t2<double>(d.oa, d.va, lam_f64.t2aa(), dev, st);
      launch_antisymmetrize_t2<double>(d.ob, d.vb, lam_f64.t2bb(), dev, st);
      cuda_check(cudaStreamSynchronize(st), "lambda fp32 promote sync");
    }

    // Strict FP64 polish (with automatic pure-FP64 restart fallback if FP32 seed diverges).
    cache_f64 = std::make_unique<DeviceForwardCache<double>>(
        build_forward_cache_device<double>(sv_f64, core, amp, dev, blas, st));

    auto run_fp64_polish = [&](int& polish_iters) -> bool {
      DeviceUccAmplitudes<double> res, step;
      res.allocate_zero(dev, d, st);
      step.allocate_zero(dev, d, st);
      DeviceDiis<double> diis(dev, blas, st, d, options.diis_space, diis_budget);
      const auto evaluate_residual = [&] {
        evaluate_lambda_vjp_device<double>(*cache_f64, sv_f64, core, amp, f12_f64.get(), lam_f64,
                                           res, nullptr, nullptr, nullptr, nullptr, dev, blas, st,
                                           ppl_budget);
        launch_add_scaled(res.flat.get(), rhs.flat.get(), 1.0, res.total, st);
        return device_max_abs<double>(res.flat.get(), res.total, dev, st);
      };
      for (int iter = 1; iter <= options.max_lambda_iterations; ++iter) {
        double rmax{};
        try {
          rmax = evaluate_residual();
        } catch (const UccDeviceLimitError&) {
          if (!diis.disable()) throw;
          // Pass A clears res before the replay. This solve has no parameter adjoint outputs.
          rmax = evaluate_residual();
        }
        polish_iters = iter;
        if (!std::isfinite(rmax)) return false;
        if (rmax < options.lambda_tolerance) return true;
        apply_lambda_preconditioner<double>(sv_f64, res, step, lam_f64, st);
        try {
          diis.push(lam_f64, step);
          (void)diis.extrapolate(lam_f64);
        } catch (const UccDeviceLimitError&) {
          diis.disable();
        }
      }
      return false;
    };

    int polish_iters = 0;
    bool converged = run_fp64_polish(polish_iters);
    if (!converged && used_fp32_seed) {
      init_lambda_from_rhs_fp64(sv_f64, rhs, lam_f64, dev, st);
      polish_iters = 0;
      converged = run_fp64_polish(polish_iters);
    }
    out.iterations = fp32_iters + polish_iters;
    if (!converged) throw std::runtime_error("DF-ULambda v2 CUDA solver did not converge");
  }
  rhs.flat.reset();

  // Single FP64 parameter-VJP pass: direct energy/triples derivatives + <L, dR/dP>.
  DeviceFourIndexCore<double> bar_core;
  bar_core.allocate_zero(dev, d, st);
  if (triples) {
    auto add_buf = [&](DeviceBuf<double>& dst, const DeviceBuf<double>& src_b) {
      if (!src_b.empty()) launch_add_scaled(dst.get(), src_b.get(), 1.0, src_b.size(), st);
    };
    add_buf(out.bar_f.foo_a, triples->bar_f.foo_a);
    add_buf(out.bar_f.fov_a, triples->bar_f.fov_a);
    add_buf(out.bar_f.fvv_a, triples->bar_f.fvv_a);
    add_buf(out.bar_f.foo_b, triples->bar_f.foo_b);
    add_buf(out.bar_f.fov_b, triples->bar_f.fov_b);
    add_buf(out.bar_f.fvv_b, triples->bar_f.fvv_b);
    add_buf(out.bar_b.boo_a, triples->bar_b.boo_a);
    add_buf(out.bar_b.bov_a, triples->bar_b.bov_a);
    add_buf(out.bar_b.bvv_a, triples->bar_b.bvv_a);
    add_buf(out.bar_b.boo_b, triples->bar_b.boo_b);
    add_buf(out.bar_b.bov_b, triples->bar_b.bov_b);
    add_buf(out.bar_b.bvv_b, triples->bar_b.bvv_b);
    add_buf(bar_core.goovv_aa, triples->bar_core.goovv_aa);
    add_buf(bar_core.goovv_ab, triples->bar_core.goovv_ab);
    add_buf(bar_core.goovv_bb, triples->bar_core.goovv_bb);
    add_buf(bar_core.goooo_aa, triples->bar_core.goooo_aa);
    add_buf(bar_core.goooo_ab, triples->bar_core.goooo_ab);
    add_buf(bar_core.goooo_bb, triples->bar_core.goooo_bb);
    add_buf(bar_core.gooov_aa, triples->bar_core.gooov_aa);
    add_buf(bar_core.gooov_bb, triples->bar_core.gooov_bb);
    add_buf(bar_core.gooov_ab, triples->bar_core.gooov_ab);
    add_buf(bar_core.gooov_ba, triples->bar_core.gooov_ba);
    add_buf(bar_core.govov_aa, triples->bar_core.govov_aa);
    add_buf(bar_core.govov_bb, triples->bar_core.govov_bb);
    add_buf(bar_core.govov_ab, triples->bar_core.govov_ab);
    add_buf(bar_core.govov_ba, triples->bar_core.govov_ba);
  }
  if (!options.energy_source_only) {
    accumulate_ccsd_energy_grad_device(sv_f64, core, amp, f12_f64.get(), nullptr, &out.bar_f,
                                       &bar_core, d_bar_f12.get(), st);
  }
  if (cache_f64) {
    DeviceUccAmplitudes<double> final_bar_t;
    final_bar_t.allocate_zero(dev, d, st);
    evaluate_lambda_vjp_device<double>(*cache_f64, sv_f64, core, amp, f12_f64.get(), lam_f64,
                                       final_bar_t, &out.bar_f, &out.bar_b, &bar_core,
                                       d_bar_f12.get(), dev, blas, st, ppl_budget);
    cache_f64.reset();
  }
  lam_f64.flat.reset();

  pullback_four_index_core_cuda_impl(src, bar_core, out.bar_b);
  bar_core = {};

  if (d_bar_f12) {
    out.f12_cotangents = d_bar_f12->download(st);
  }
  cuda_check(cudaStreamSynchronize(st), "lambda solve final sync");
  return out;
}

}  // namespace generativeqc::cc::v2_detail::cuda_detail
