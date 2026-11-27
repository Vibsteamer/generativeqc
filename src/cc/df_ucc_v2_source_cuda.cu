#include "cc/df_ucc_v2_cuda_internal.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cc/df_source_metric_capacity.hpp"
#include "cc/df_ucc_v2_source_capacity.hpp"
#include "df_mo_source_generated.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda/df_source_domain.hpp"
#include "scf/cuda/df_source_internal.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_df_nuclear_sink.hpp"
#include "scf/df_source_capacity.hpp"

namespace generativeqc::cc::v2_detail::cuda_detail {
namespace {

struct SourceDelete {
  void operator()(scf::CudaDensityFittingIntegralSource* p) const noexcept {
    scf::destroy_cuda_density_fitting_integral_source(p);
  }
};

void check_gqc_status(generativeqc_status status, const std::string& detail) {
  if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT) throw std::invalid_argument(detail);
  if (status != GENERATIVEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}

__global__ void slice_active_coeff_kernel(std::size_t n, std::size_t o_full, std::size_t frozen,
                                          std::size_t o, std::size_t v, const double* c_full,
                                          double* c_act) {
  const std::size_t n_act = o + v;
  const std::size_t total = n * n_act;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t p = idx % n_act;
    const std::size_t mu = idx / n_act;
    const std::size_t col = (p < o) ? (frozen + p) : (o_full + (p - o));
    c_act[idx] = c_full[mu * n + col];
  }
}

__global__ void slice_active_fock_kernel(std::size_t n, std::size_t o_full, std::size_t frozen,
                                         std::size_t o, std::size_t v, const double* f_mo,
                                         double* foo, double* fov, double* fvv) {
  const std::size_t noo = o * o, nov = o * v, nvv = v * v;
  const std::size_t total = noo + nov + nvv;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (idx < noo) {
      const std::size_t i = idx / o, j = idx % o;
      foo[idx] = f_mo[(frozen + i) * n + (frozen + j)];
    } else if (idx < noo + nov) {
      const std::size_t k = idx - noo;
      const std::size_t i = k / v, a = k % v;
      fov[k] = f_mo[(frozen + i) * n + (o_full + a)];
    } else {
      const std::size_t k = idx - noo - nov;
      const std::size_t a = k / v, b = k % v;
      fvv[k] = f_mo[(o_full + a) * n + (o_full + b)];
    }
  }
}

__global__ void pack_qmajor_factors_kernel(std::size_t q, std::size_t o, std::size_t v,
                                           const double* bmo, double* boo, double* bov,
                                           double* bvv, int* nonfinite) {
  const std::size_t n_act = o + v;
  const std::size_t qoo = q * o * o, qov = q * o * v, qvv = q * v * v;
  const std::size_t total = qoo + qov + qvv;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    double val = 0.0;
    if (idx < qoo) {
      const std::size_t Q = idx / (o * o);
      const std::size_t rem = idx % (o * o);
      const std::size_t i = rem / o, j = rem % o;
      val = 0.5 * (bmo[(i * n_act + j) * q + Q] + bmo[(j * n_act + i) * q + Q]);
      boo[idx] = val;
    } else if (idx < qoo + qov) {
      const std::size_t k = idx - qoo;
      const std::size_t Q = k / (o * v);
      const std::size_t rem = k % (o * v);
      const std::size_t i = rem / v, a = rem % v;
      val = 0.5 * (bmo[(i * n_act + (o + a)) * q + Q] + bmo[((o + a) * n_act + i) * q + Q]);
      bov[k] = val;
    } else {
      const std::size_t k = idx - qoo - qov;
      const std::size_t Q = k / (v * v);
      const std::size_t rem = k % (v * v);
      const std::size_t a = rem / v, b = rem % v;
      val = 0.5 * (bmo[((o + a) * n_act + (o + b)) * q + Q] +
                   bmo[((o + b) * n_act + (o + a)) * q + Q]);
      bvv[k] = val;
    }
    if (!isfinite(val)) atomicExch(nonfinite, 1);
  }
}

__global__ void embed_qmajor_bar_bmo_kernel(std::size_t q, std::size_t o, std::size_t v,
                                            const double* bar_boo, const double* bar_bov,
                                            const double* bar_bvv, double* bar_bmo) {
  const std::size_t n_act = o + v;
  const std::size_t total = n_act * n_act * q;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t Q = idx % q;
    const std::size_t pr = idx / q;
    const std::size_t p = pr / n_act;
    const std::size_t r = pr % n_act;
    double val = 0.0;
    if (p < o && r < o) {
      val = 0.5 * (bar_boo[(Q * o + p) * o + r] + bar_boo[(Q * o + r) * o + p]);
    } else if (p < o && r >= o) {
      val = 0.5 * bar_bov[(Q * o + p) * v + (r - o)];
    } else if (p >= o && r < o) {
      val = 0.5 * bar_bov[(Q * o + r) * v + (p - o)];
    } else {
      const std::size_t a = p - o, b = r - o;
      val = 0.5 * (bar_bvv[(Q * v + a) * v + b] + bar_bvv[(Q * v + b) * v + a]);
    }
    bar_bmo[idx] = val;
  }
}

__global__ void symmetrize_matrix_kernel(std::size_t n, const double* src, double* dst) {
  const std::size_t total = n * n;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t i = idx / n, j = idx % n;
    dst[idx] = 0.5 * (src[i * n + j] + src[j * n + i]);
  }
}

__device__ inline double scaled_response_dev(double seed, double a, double b, double c) {
  if (seed == 0.0) return seed;
  int es = 0, ea = 0, eb = 0, ec = 0;
  const double ms = frexp(seed, &es), ma = frexp(a, &ea);
  const double mb = frexp(b, &eb), mc = frexp(c, &ec);
  return scalbn(((ms / ma) / mb) / mc, es - ea - eb - ec);
}

// Daleckii-Krein divided-difference weighting for symmetric M^{-1/2} VJP in the eigenbasis.
__global__ void metric_inv_sqrt_dk_kernel(std::size_t q, const double* eigenvalues,
                                          double relative_threshold, double* mat, int* nonfinite) {
  const std::size_t total = q * q;
  const double cutoff = relative_threshold * eigenvalues[q - 1];
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t i = idx / q, j = idx % q;
    const double li = eigenvalues[i], lj = eigenvalues[j];
    const bool ki = li > cutoff, kj = lj > cutoff;
    double value = mat[idx];
    if (value == 0.0 || (!ki && !kj)) {
      value = 0.0;
    } else {
      double divided = 0.0;
      if (ki && kj) {
        const double si = sqrt(li), sj = sqrt(lj);
        divided = -1.0 / si / sj / (si + sj);
      } else {
        const double fi = ki ? (1.0 / sqrt(li)) : 0.0;
        const double fj = kj ? (1.0 / sqrt(lj)) : 0.0;
        divided = (fi - fj) / (li - lj);
      }
      if (isfinite(divided) && fabs(divided) >= 0x1p-1022) {
        value *= divided;
      } else if (ki && kj) {
        const double si = sqrt(li), sj = sqrt(lj);
        value = scaled_response_dev(-value, si, sj, si + sj);
      } else {
        const double kept = ki ? li : lj;
        value = scaled_response_dev(ki ? value : -value, sqrt(kept), li - lj, 1.0);
      }
    }
    if (!isfinite(value)) atomicExch(nonfinite, 1);
    mat[idx] = value;
  }
}

__global__ void audit_finite_kernel(const double* data, std::size_t count, int* nonfinite) {
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (!isfinite(data[i])) atomicExch(nonfinite, 1);
  }
}

// --- 4-Index Core Build Kernels ---
__global__ void build_goovv_ss_kernel(std::size_t o, std::size_t v, const double* ovov,
                                      double* goovv) {
  const std::size_t total = o * o * v * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t a = r1 % v;
    const std::size_t r2 = r1 / v;
    const std::size_t j = r2 % o;
    const std::size_t i = r2 / o;
    goovv[idx] = ovov[((i * v + a) * o + j) * v + b] - ovov[((i * v + b) * o + j) * v + a];
  }
}

__global__ void build_goooo_ss_kernel(std::size_t o, const double* oooo, double* goooo) {
  const std::size_t total = o * o * o * o;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t j = idx % o;
    const std::size_t r1 = idx / o;
    const std::size_t i = r1 % o;
    const std::size_t r2 = r1 / o;
    const std::size_t n = r2 % o;
    const std::size_t m = r2 / o;
    goooo[idx] = oooo[((m * o + i) * o + n) * o + j] - oooo[((m * o + j) * o + n) * o + i];
  }
}

__global__ void build_gooov_ss_kernel(std::size_t o, std::size_t v, const double* ooov,
                                      double* gooov) {
  const std::size_t total = o * o * o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t e = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t i = r1 % o;
    const std::size_t r2 = r1 / o;
    const std::size_t n = r2 % o;
    const std::size_t m = r2 / o;
    gooov[idx] = ooov[((m * o + i) * o + n) * v + e] - ooov[((n * o + i) * o + m) * v + e];
  }
}

__global__ void build_govov_ss_kernel(std::size_t o, std::size_t v, const double* oovv,
                                      const double* ovov, double* govov) {
  const std::size_t total = o * v * o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t e = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t j = r1 % o;
    const std::size_t r2 = r1 / o;
    const std::size_t b = r2 % v;
    const std::size_t m = r2 / v;
    govov[idx] = oovv[((m * o + j) * v + b) * v + e] - ovov[((m * v + e) * o + j) * v + b];
  }
}

// --- 4-Index Core Pullback Kernels ---
__global__ void pullback_goovv_ss_kernel(std::size_t o, std::size_t v, const double* bc,
                                         double* m) {
  const std::size_t total = o * v * o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t j = r1 % o;
    const std::size_t r2 = r1 / o;
    const std::size_t a = r2 % v;
    const std::size_t i = r2 / v;
    m[idx] = bc[((i * o + j) * v + a) * v + b] + bc[((j * o + i) * v + b) * v + a] -
             bc[((i * o + j) * v + b) * v + a] - bc[((j * o + i) * v + a) * v + b];
  }
}

__global__ void pullback_goooo_ss_kernel(std::size_t o, const double* bc, double* m) {
  const std::size_t total = o * o * o * o;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t j = idx % o;
    const std::size_t r1 = idx / o;
    const std::size_t r = r1 % o;
    const std::size_t r2 = r1 / o;
    const std::size_t i = r2 % o;
    const std::size_t p = r2 / o;
    m[idx] = bc[((p * o + r) * o + i) * o + j] + bc[((r * o + p) * o + j) * o + i] -
             bc[((p * o + r) * o + j) * o + i] - bc[((r * o + p) * o + i) * o + j];
  }
}

__global__ void pullback_gooov_ss_kernel(std::size_t o, std::size_t v, const double* bc,
                                         double* m) {
  const std::size_t total = o * o * o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t e = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t r = r1 % o;
    const std::size_t r2 = r1 / o;
    const std::size_t i = r2 % o;
    const std::size_t p = r2 / o;
    m[idx] = bc[((p * o + r) * o + i) * v + e] - bc[((r * o + p) * o + i) * v + e];
  }
}

__global__ void pullback_govov_ss_mx_kernel(std::size_t o, std::size_t v, const double* bc,
                                            double* mx) {
  const std::size_t total = o * v * o * v;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t b = idx % v;
    const std::size_t r1 = idx / v;
    const std::size_t j = r1 % o;
    const std::size_t r2 = r1 / o;
    const std::size_t e = r2 % v;
    const std::size_t m = r2 / v;
    mx[idx] = bc[((m * v + b) * o + j) * v + e] + bc[((j * v + e) * o + m) * v + b];
  }
}

void build_spin_factors_device(int device, scf::CudaDensityFittingJkPlan& plan, std::size_t n,
                               std::size_t q, std::size_t o, std::size_t v, const double* d_c_act,
                               double* d_boo, double* d_bov, double* d_bvv) {
  const std::size_t n_act = o + v;
  if (!n_act) return;
  const cudaStream_t stream = plan.stream;
  const std::size_t row_elems = checked_mul(n, q);
  const std::size_t t1_elems = checked_mul(n, checked_mul(n_act, q));
  const std::size_t t2_elems = checked_mul(checked_mul(n_act, n_act), q);

  DeviceBuf<double> d_row_hi(device, row_elems, stream);
  DeviceBuf<double> d_row_lo(device, row_elems, stream);
  DeviceBuf<double> d_t1(device, t1_elems, stream);
  std::string detail;
  for (std::size_t mu = 0; mu < n; ++mu) {
    check_gqc_status(scf::generate_cuda_density_fitting_raw_expansion(
                         plan.integral_source, 0, mu * n, n, 0, q, stream, d_row_hi.get(),
                         d_row_lo.get(), detail),
                     detail);
    posthf::generated::compensated::gemm('N', 'T', q, n_act, n, d_row_hi.get(), d_c_act,
                                         d_row_lo.get(), nullptr,
                                         d_t1.get() + mu * n_act * q, stream);
    cuda_check(cudaGetLastError(), "compensated source row gemm");
  }
  d_row_hi.reset();
  d_row_lo.reset();

  DeviceBuf<double> d_t2(device, t2_elems, stream);
  posthf::generated::compensated::gemm('N', 'T', n_act * q, n_act, n, d_t1.get(), d_c_act,
                                       nullptr, nullptr, d_t2.get(), stream);
  cuda_check(cudaGetLastError(), "compensated source second projection");
  d_t1.reset();

  DeviceBuf<double> d_bmo(device, t2_elems, stream);
  rm_gemm<double>(plan.blas, stream, 'N', 'N', n_act * n_act, q, q, d_t2.get(),
                  plan.inverse_square_roots, d_bmo.get(), 1.0, 0.0);
  d_t2.reset();

  DeviceBuf<int> d_bad(device, 1, stream);
  d_bad.zero(stream);
  const std::size_t pack_total = q * (o * o + o * v + v * v);
  if (pack_total) {
    pack_qmajor_factors_kernel<<<grid_1d(pack_total), 256, 0, stream>>>(
        q, o, v, d_bmo.get(), d_boo, d_bov, d_bvv, d_bad.get());
    cuda_check(cudaGetLastError(), "pack_qmajor_factors");
  }
  int bad = 0;
  d_bad.copy_to_host(&bad, 1, stream);
  cuda_check(cudaStreamSynchronize(stream), "build_spin_factors sync");
  if (bad) throw std::runtime_error("DF-UCC v2 CUDA produced nonfinite 3-index DF factors");
}

}  // namespace

CudaSpinBlockedSource build_spin_blocked_df_source_cuda_impl(
    const core::System& orbital, const core::System& auxiliary,
    const hf::UnrestrictedPhysicalReference& reference, int device, const DFUccV2Options& options) {
  if (device < 0) throw std::invalid_argument("DF-UCC v2 CUDA requires a valid device ordinal");
  CudaSpinBlockedSource src;
  src.dims = validate_and_extract_dims(orbital, auxiliary, reference, options);
  src.device = device;
  src.metric_relative_threshold = options.metric_relative_threshold;
  src.device_cap = query_device_cap_bytes(device, options.max_bytes);
  const auto& d = src.dims;

  const auto panel_peak = ucc_source_panel_peak_bytes(d);
  require_ucc_source_panel_capacity(panel_peak, 0, src.device_cap);
  const auto construction = scf::df_source_capacity::plan(
      orbital, auxiliary,
      {sizeof(scf::cuda_execution::CudaDensityFittingIntegralSourceImpl),
       sizeof(scf::cuda_execution::HostBatch), sizeof(scf::cuda_execution::DfPublicAoExpansion)});
  if (construction.device_bytes > src.device_cap ||
      (options.max_bytes && construction.numeric_bytes > options.max_bytes))
    throw std::length_error("DF-UCC v2 CUDA source construction exceeds memory budget");

  scf::cuda_execution::CudaDfSourcePolicy resolved_policy;
  std::string detail;
  if (!scf::cuda_execution::resolve_cuda_df_source_policy(resolved_policy, detail))
    throw std::invalid_argument(detail);
  if (!scf::cuda_execution::cuda_df_value_domain(orbital, auxiliary, resolved_policy, detail))
    throw std::invalid_argument(detail);

  runtime::CudaDeviceScope dev_scope(device);
  scf::CudaDensityFittingIntegralSource* raw_source = nullptr;
  std::vector<double> metric;
  std::size_t source_n = 0, source_q = 0;
  check_gqc_status(scf::create_cuda_density_fitting_integral_source(
                       device, {orbital}, {auxiliary}, &raw_source, metric, source_n, source_q,
                       detail, &resolved_policy),
                   detail);
  std::unique_ptr<scf::CudaDensityFittingIntegralSource, SourceDelete> source_guard(raw_source);
  if (source_n != d.n || source_q != d.q || metric.size() != checked_mul(d.q, d.q))
    throw std::runtime_error("DF-UCC v2 CUDA source dimension mismatch");

  const auto placement = scf::cuda_density_fitting_integral_source_diagnostic(source_guard.get());
  if ((std::strcmp(placement.value_backend, "generated_rys") != 0 &&
       std::strcmp(placement.value_backend, "generated_rys_auxiliary_g_polynomial") != 0 &&
       std::strcmp(placement.value_backend, "generated_rys_auxiliary_h_polynomial") != 0) ||
      !placement.public_transform_on_device) {
    throw std::runtime_error("DF-UCC v2 CUDA requires generated device DF integrals");
  }

  const auto source_device =
      scf::cuda_density_fitting_integral_source_device_bytes(source_guard.get());
  const auto source_host =
      scf::cuda_density_fitting_integral_source_host_peak_bytes(source_guard.get());
  const auto metric_setup_device = df_streamed_metric_setup_bound(d.n, d.q, source_device, 0);
  const auto metric_setup_total = df_streamed_metric_setup_bound(d.n, d.q, source_device, source_host);
  if (metric_setup_device > src.device_cap ||
      (options.max_bytes && metric_setup_total > options.max_bytes))
    throw std::length_error("DF-UCC v2 CUDA metric setup exceeds memory budget");

  scf::CudaDensityFittingJkPlan* raw_plan = nullptr;
  auto* transferred = source_guard.release();
  std::vector<scf::CudaDensityFittingMetricDiagnostic> diagnostics;
  const auto plan_status = scf::create_cuda_density_fitting_jk_plan_from_source(
      device, &transferred, 1, d.n, d.q, metric, options.metric_relative_threshold, d.q, d.n,
      &raw_plan, diagnostics, detail, /*retain_three_center=*/false);
  src.owned_plan.reset(raw_plan);
  check_gqc_status(plan_status, detail);
  if (!src.owned_plan || !src.owned_plan->integral_source ||
      !src.owned_plan->inverse_square_roots || diagnostics.size() != 1) {
    throw std::runtime_error("DF-UCC v2 CUDA requires streamed metric owner");
  }
  if (options.compute_forces &&
      (src.owned_plan->metric_response_valid.size() != 1 ||
       !src.owned_plan->metric_response_valid[0])) {
    throw std::runtime_error("DF-UCC v2 CUDA metric rank crossing: response branch unresolved");
  }

  const auto& plan_diagnostic = diagnostics.front();
  if (plan_diagnostic.peak_device_bytes > metric_setup_device ||
      checked_add(plan_diagnostic.peak_device_bytes, plan_diagnostic.peak_host_bytes) >
          metric_setup_total)
    throw std::logic_error("DF-UCC v2 CUDA metric setup exceeded its preflight bound");
  src.plan_resident_bytes = plan_diagnostic.device_resident_bytes;
  std::vector<double>().swap(metric);

  const auto eps_elems = checked_add(d.n_act_a(), d.n_act_b());
  const auto host_bytes =
      checked_add(plan_diagnostic.host_resident_bytes, checked_mul(eps_elems, sizeof(double)));
  if (options.max_bytes) {
    if (host_bytes >= options.max_bytes)
      throw std::length_error("DF-UCC v2 CUDA retained source exceeds caller budget");
    src.device_cap = std::min(src.device_cap, options.max_bytes - host_bytes);
  }

  require_ucc_source_panel_capacity(panel_peak, src.plan_resident_bytes, src.device_cap);

  src.stream = src.owned_plan->stream;
  src.blas = src.owned_plan->blas;
  src.blas_workspace.allocate(device, kUccBlasWorkspaceBytes, src.stream);
  cublas_check(cublasSetWorkspace(src.blas, src.blas_workspace.get(), src.blas_workspace.bytes()),
               "set bounded BLAS workspace");
  const cudaStream_t stream = src.stream;

  // Populate active MO Fock blocks and active orbital energies directly on GPU.
  src.fock.allocate_zero(device, d, stream);
  src.eps_oa.allocate(device, d.oa, stream);
  src.eps_va.allocate(device, d.va, stream);
  src.eps_ob.allocate(device, d.ob, stream);
  src.eps_vb.allocate(device, d.vb, stream);
  src.c_act_a.allocate(device, d.n * d.n_act_a(), stream);
  src.c_act_b.allocate(device, d.n * d.n_act_b(), stream);

  src.host_eps_oa.assign(reference.orbital_energies[0].begin() + static_cast<std::ptrdiff_t>(d.fa),
                         reference.orbital_energies[0].begin() +
                             static_cast<std::ptrdiff_t>(d.oa_full));
  src.host_eps_va.assign(
      reference.orbital_energies[0].begin() + static_cast<std::ptrdiff_t>(d.oa_full),
      reference.orbital_energies[0].end());
  src.host_eps_ob.assign(reference.orbital_energies[1].begin() + static_cast<std::ptrdiff_t>(d.fb),
                         reference.orbital_energies[1].begin() +
                             static_cast<std::ptrdiff_t>(d.ob_full));
  src.host_eps_vb.assign(
      reference.orbital_energies[1].begin() + static_cast<std::ptrdiff_t>(d.ob_full),
      reference.orbital_energies[1].end());
  src.eps_oa.copy_from_host(src.host_eps_oa.data(), d.oa, stream);
  src.eps_va.copy_from_host(src.host_eps_va.data(), d.va, stream);
  src.eps_ob.copy_from_host(src.host_eps_ob.data(), d.ob, stream);
  src.eps_vb.copy_from_host(src.host_eps_vb.data(), d.vb, stream);

  {
    DeviceBuf<double> d_c(device, d.n * d.n, stream);
    DeviceBuf<double> d_f(device, d.n * d.n, stream);
    DeviceBuf<double> d_tmp(device, d.n * d.n, stream);
    DeviceBuf<double> d_fmo(device, d.n * d.n, stream);
    auto prep_spin = [&](std::size_t s, std::size_t f, std::size_t o_full, std::size_t o,
                         std::size_t v, double* d_c_act, double* d_foo, double* d_fov,
                         double* d_fvv) {
      d_c.copy_from_host(reference.coefficients[s].data(), d.n * d.n, stream);
      d_f.copy_from_host(reference.fock[s].data(), d.n * d.n, stream);
      rm_gemm<double>(src.blas, stream, 'N', 'N', d.n, d.n, d.n, d_f.get(), d_c.get(), d_tmp.get(),
                      1.0, 0.0);
      rm_gemm<double>(src.blas, stream, 'T', 'N', d.n, d.n, d.n, d_c.get(), d_tmp.get(),
                      d_fmo.get(), 1.0, 0.0);
      const std::size_t n_act = o + v;
      if (n_act) {
        slice_active_coeff_kernel<<<grid_1d(d.n * n_act), 256, 0, stream>>>(
            d.n, o_full, f, o, v, d_c.get(), d_c_act);
        cuda_check(cudaGetLastError(), "slice_active_coeff");
      }
      const std::size_t ftotal = o * o + o * v + v * v;
      if (ftotal) {
        slice_active_fock_kernel<<<grid_1d(ftotal), 256, 0, stream>>>(
            d.n, o_full, f, o, v, d_fmo.get(), d_foo, d_fov, d_fvv);
        cuda_check(cudaGetLastError(), "slice_active_fock");
      }
      cuda_check(cudaStreamSynchronize(stream), "fock/coeff prep sync");
    };
    prep_spin(0, d.fa, d.oa_full, d.oa, d.va, src.c_act_a.get(), src.fock.foo_a.get(),
              src.fock.fov_a.get(), src.fock.fvv_a.get());
    prep_spin(1, d.fb, d.ob_full, d.ob, d.vb, src.c_act_b.get(), src.fock.foo_b.get(),
              src.fock.fov_b.get(), src.fock.fvv_b.get());
  }

  // Build GPU-resident spin-blocked 3-index whitened DF factors (ZERO 4-index vvvv).
  src.factors.allocate_zero(device, d, stream);
  build_spin_factors_device(device, *src.owned_plan, d.n, d.q, d.oa, d.va, src.c_act_a.get(),
                            src.factors.boo_a.get(), src.factors.bov_a.get(),
                            src.factors.bvv_a.get());
  build_spin_factors_device(device, *src.owned_plan, d.n, d.q, d.ob, d.vb, src.c_act_b.get(),
                            src.factors.boo_b.get(), src.factors.bov_b.get(),
                            src.factors.bvv_b.get());
  return src;
}

std::size_t source_resident_bytes(const CudaSpinBlockedSource& src) {
  std::size_t bytes = checked_add(src.plan_resident_bytes, src.blas_workspace.bytes());
  bytes = checked_add(bytes, src.factors.total_bytes());
  for (const auto size : {src.fock.foo_a.bytes(), src.fock.fov_a.bytes(), src.fock.fvv_a.bytes(),
                          src.fock.foo_b.bytes(), src.fock.fov_b.bytes(), src.fock.fvv_b.bytes(),
                          src.eps_oa.bytes(), src.eps_va.bytes(), src.eps_ob.bytes(), src.eps_vb.bytes(),
                          src.c_act_a.bytes(), src.c_act_b.bytes()})
    bytes = checked_add(bytes, size);
  return bytes;
}

DeviceFourIndexCore<double> build_four_index_core_cuda_impl(const CudaSpinBlockedSource& src) {
  const auto& d = src.dims;
  const int dev = src.device;
  const cudaStream_t st = src.stream;
  const cublasHandle_t blas = src.blas;
  const std::size_t q = d.q, oa = d.oa, ob = d.ob, va = d.va, vb = d.vb;

  runtime::CudaDeviceScope dev_scope(dev);
  {
    const auto src_resident_bytes = source_resident_bytes(src);
    const std::size_t oa2 = checked_mul(oa, oa), ob2 = checked_mul(ob, ob);
    const std::size_t va2 = checked_mul(va, va), vb2 = checked_mul(vb, vb);
    const std::size_t oava2 = checked_mul(oa2, va2);
    const std::size_t obvb2 = checked_mul(ob2, vb2);
    const std::size_t oabvab = checked_mul(checked_mul(oa, ob), checked_mul(va, vb));
    const std::size_t core_elems = checked_add(
        checked_add(checked_mul(2, oava2), checked_mul(2, obvb2)),
        checked_add(
            oabvab,
            checked_add(
                checked_add(checked_mul(oa2, oa2),
                            checked_add(checked_mul(oa2, ob2), checked_mul(ob2, ob2))),
                checked_add(
                    checked_add(checked_mul(oa2, checked_mul(oa, va)),
                                checked_mul(ob2, checked_mul(ob, vb))),
                    checked_add(
                        checked_add(checked_mul(oa2, checked_mul(ob, vb)),
                                    checked_mul(ob2, checked_mul(oa, va))),
                        checked_add(checked_mul(oa2, vb2), checked_mul(ob2, va2)))))));
    const std::size_t scratch_elems = std::max({
        checked_mul(2, oava2),
        checked_mul(2, obvb2),
        oabvab,
        checked_mul(oa2, oa2),
        checked_mul(ob2, ob2),
        checked_mul(oa2, ob2),
        checked_mul(oa2, checked_mul(oa, va)),
        checked_mul(ob2, checked_mul(ob, vb)),
        checked_mul(oa2, checked_mul(ob, vb)),
        checked_mul(ob2, checked_mul(oa, va)),
        checked_mul(oa2, vb2),
        checked_mul(ob2, va2),
    });
    const std::size_t peak_bytes =
        checked_add(src_resident_bytes,
                    checked_mul(checked_add(core_elems, scratch_elems), sizeof(double)));
    if (peak_bytes > src.device_cap)
      throw std::length_error("DF-UCC v2 CUDA 4-index core exceeds device memory budget");
  }
  DeviceFourIndexCore<double> c;
  c.allocate_zero(dev, d, st);

  // goovv_aa, goovv_bb, goovv_ab
  if (oa && va) {
    DeviceBuf<double> ovov(dev, oa * va * oa * va, st);
    rm_gemm<double>(blas, st, 'T', 'N', oa * va, oa * va, q, src.factors.bov_a.get(),
                    src.factors.bov_a.get(), ovov.get(), 1.0, 0.0);
    build_goovv_ss_kernel<<<grid_1d(oa * oa * va * va), 256, 0, st>>>(oa, va, ovov.get(),
                                                                      c.goovv_aa.get());
    cuda_check(cudaGetLastError(), "build_goovv_aa");
  }
  if (ob && vb) {
    DeviceBuf<double> ovov(dev, ob * vb * ob * vb, st);
    rm_gemm<double>(blas, st, 'T', 'N', ob * vb, ob * vb, q, src.factors.bov_b.get(),
                    src.factors.bov_b.get(), ovov.get(), 1.0, 0.0);
    build_goovv_ss_kernel<<<grid_1d(ob * ob * vb * vb), 256, 0, st>>>(ob, vb, ovov.get(),
                                                                      c.goovv_bb.get());
    cuda_check(cudaGetLastError(), "build_goovv_bb");
  }
  if (oa && ob && va && vb) {
    DeviceBuf<double> ovov(dev, oa * va * ob * vb, st);
    rm_gemm<double>(blas, st, 'T', 'N', oa * va, ob * vb, q, src.factors.bov_a.get(),
                    src.factors.bov_b.get(), ovov.get(), 1.0, 0.0);
    // ovov[i, a, j, b] -> goovv_ab[i, j, a, b]
    launch_permute4d<double, 0, 2, 1, 3>(oa, va, ob, vb, ovov.get(), c.goovv_ab.get(), st);
  }

  // goooo_aa, goooo_bb, goooo_ab
  if (oa) {
    DeviceBuf<double> oooo(dev, oa * oa * oa * oa, st);
    rm_gemm<double>(blas, st, 'T', 'N', oa * oa, oa * oa, q, src.factors.boo_a.get(),
                    src.factors.boo_a.get(), oooo.get(), 1.0, 0.0);
    build_goooo_ss_kernel<<<grid_1d(oa * oa * oa * oa), 256, 0, st>>>(oa, oooo.get(),
                                                                      c.goooo_aa.get());
    cuda_check(cudaGetLastError(), "build_goooo_aa");
  }
  if (ob) {
    DeviceBuf<double> oooo(dev, ob * ob * ob * ob, st);
    rm_gemm<double>(blas, st, 'T', 'N', ob * ob, ob * ob, q, src.factors.boo_b.get(),
                    src.factors.boo_b.get(), oooo.get(), 1.0, 0.0);
    build_goooo_ss_kernel<<<grid_1d(ob * ob * ob * ob), 256, 0, st>>>(ob, oooo.get(),
                                                                      c.goooo_bb.get());
    cuda_check(cudaGetLastError(), "build_goooo_bb");
  }
  if (oa && ob) {
    DeviceBuf<double> oooo(dev, oa * oa * ob * ob, st);
    rm_gemm<double>(blas, st, 'T', 'N', oa * oa, ob * ob, q, src.factors.boo_a.get(),
                    src.factors.boo_b.get(), oooo.get(), 1.0, 0.0);
    // oooo[m, i, n, j] -> goooo_ab[m, n, i, j]
    launch_permute4d<double, 0, 2, 1, 3>(oa, oa, ob, ob, oooo.get(), c.goooo_ab.get(), st);
  }

  // gooov_aa, gooov_bb, gooov_ab, gooov_ba
  if (oa && va) {
    DeviceBuf<double> ooov(dev, oa * oa * oa * va, st);
    rm_gemm<double>(blas, st, 'T', 'N', oa * oa, oa * va, q, src.factors.boo_a.get(),
                    src.factors.bov_a.get(), ooov.get(), 1.0, 0.0);
    build_gooov_ss_kernel<<<grid_1d(oa * oa * oa * va), 256, 0, st>>>(oa, va, ooov.get(),
                                                                      c.gooov_aa.get());
    cuda_check(cudaGetLastError(), "build_gooov_aa");
  }
  if (ob && vb) {
    DeviceBuf<double> ooov(dev, ob * ob * ob * vb, st);
    rm_gemm<double>(blas, st, 'T', 'N', ob * ob, ob * vb, q, src.factors.boo_b.get(),
                    src.factors.bov_b.get(), ooov.get(), 1.0, 0.0);
    build_gooov_ss_kernel<<<grid_1d(ob * ob * ob * vb), 256, 0, st>>>(ob, vb, ooov.get(),
                                                                      c.gooov_bb.get());
    cuda_check(cudaGetLastError(), "build_gooov_bb");
  }
  if (oa && ob && vb) {
    DeviceBuf<double> ooov(dev, oa * oa * ob * vb, st);
    rm_gemm<double>(blas, st, 'T', 'N', oa * oa, ob * vb, q, src.factors.boo_a.get(),
                    src.factors.bov_b.get(), ooov.get(), 1.0, 0.0);
    // ooov[m, i, n, e] -> gooov_ab[m, n, i, e]
    launch_permute4d<double, 0, 2, 1, 3>(oa, oa, ob, vb, ooov.get(), c.gooov_ab.get(), st);
  }
  if (ob && oa && va) {
    DeviceBuf<double> ooov(dev, ob * ob * oa * va, st);
    rm_gemm<double>(blas, st, 'T', 'N', ob * ob, oa * va, q, src.factors.boo_b.get(),
                    src.factors.bov_a.get(), ooov.get(), 1.0, 0.0);
    // ooov[m, i, n, e] -> gooov_ba[m, n, i, e]
    launch_permute4d<double, 0, 2, 1, 3>(ob, ob, oa, va, ooov.get(), c.gooov_ba.get(), st);
  }

  // govov_aa, govov_bb, govov_ab, govov_ba
  if (oa && va) {
    DeviceBuf<double> oovv(dev, oa * oa * va * va, st);
    DeviceBuf<double> ovov(dev, oa * va * oa * va, st);
    rm_gemm<double>(blas, st, 'T', 'N', oa * oa, va * va, q, src.factors.boo_a.get(),
                    src.factors.bvv_a.get(), oovv.get(), 1.0, 0.0);
    rm_gemm<double>(blas, st, 'T', 'N', oa * va, oa * va, q, src.factors.bov_a.get(),
                    src.factors.bov_a.get(), ovov.get(), 1.0, 0.0);
    build_govov_ss_kernel<<<grid_1d(oa * va * oa * va), 256, 0, st>>>(oa, va, oovv.get(),
                                                                      ovov.get(), c.govov_aa.get());
    cuda_check(cudaGetLastError(), "build_govov_aa");
  }
  if (ob && vb) {
    DeviceBuf<double> oovv(dev, ob * ob * vb * vb, st);
    DeviceBuf<double> ovov(dev, ob * vb * ob * vb, st);
    rm_gemm<double>(blas, st, 'T', 'N', ob * ob, vb * vb, q, src.factors.boo_b.get(),
                    src.factors.bvv_b.get(), oovv.get(), 1.0, 0.0);
    rm_gemm<double>(blas, st, 'T', 'N', ob * vb, ob * vb, q, src.factors.bov_b.get(),
                    src.factors.bov_b.get(), ovov.get(), 1.0, 0.0);
    build_govov_ss_kernel<<<grid_1d(ob * vb * ob * vb), 256, 0, st>>>(ob, vb, oovv.get(),
                                                                      ovov.get(), c.govov_bb.get());
    cuda_check(cudaGetLastError(), "build_govov_bb");
  }
  if (oa && vb) {
    DeviceBuf<double> oovv(dev, oa * oa * vb * vb, st);
    rm_gemm<double>(blas, st, 'T', 'N', oa * oa, vb * vb, q, src.factors.boo_a.get(),
                    src.factors.bvv_b.get(), oovv.get(), 1.0, 0.0);
    // oovv[m, j, b, e] -> govov_ab[m, b, j, e]
    launch_permute4d<double, 0, 2, 1, 3>(oa, oa, vb, vb, oovv.get(), c.govov_ab.get(), st);
  }
  if (ob && va) {
    DeviceBuf<double> oovv(dev, ob * ob * va * va, st);
    rm_gemm<double>(blas, st, 'T', 'N', ob * ob, va * va, q, src.factors.boo_b.get(),
                    src.factors.bvv_a.get(), oovv.get(), 1.0, 0.0);
    // oovv[m, j, b, e] -> govov_ba[m, b, j, e]
    launch_permute4d<double, 0, 2, 1, 3>(ob, ob, va, va, oovv.get(), c.govov_ba.get(), st);
  }
  return c;
}

void pullback_four_index_core_cuda_impl(const CudaSpinBlockedSource& src,
                                        const DeviceFourIndexCore<double>& bc,
                                        DeviceFactorGrads& bar_b) {
  const auto& d = src.dims;
  const int dev = src.device;
  const cudaStream_t st = src.stream;
  const cublasHandle_t blas = src.blas;
  const std::size_t q = d.q, oa = d.oa, ob = d.ob, va = d.va, vb = d.vb;

  runtime::CudaDeviceScope dev_scope(dev);

  // goovv -> bar_bov
  if (oa && va) {
    DeviceBuf<double> m(dev, oa * va * oa * va, st);
    pullback_goovv_ss_kernel<<<grid_1d(oa * va * oa * va), 256, 0, st>>>(oa, va, bc.goovv_aa.get(),
                                                                         m.get());
    cuda_check(cudaGetLastError(), "pullback_goovv_aa");
    rm_gemm<double>(blas, st, 'N', 'N', q, oa * va, oa * va, src.factors.bov_a.get(), m.get(),
                    bar_b.bov_a.get(), 1.0, 1.0);
  }
  if (ob && vb) {
    DeviceBuf<double> m(dev, ob * vb * ob * vb, st);
    pullback_goovv_ss_kernel<<<grid_1d(ob * vb * ob * vb), 256, 0, st>>>(ob, vb, bc.goovv_bb.get(),
                                                                         m.get());
    cuda_check(cudaGetLastError(), "pullback_goovv_bb");
    rm_gemm<double>(blas, st, 'N', 'N', q, ob * vb, ob * vb, src.factors.bov_b.get(), m.get(),
                    bar_b.bov_b.get(), 1.0, 1.0);
  }
  if (oa && ob && va && vb) {
    DeviceBuf<double> m(dev, oa * va * ob * vb, st);
    // bc.goovv_ab[i, j, a, b] -> m[i, a, j, b]
    launch_permute4d<double, 0, 2, 1, 3>(oa, ob, va, vb, bc.goovv_ab.get(), m.get(), st);
    rm_gemm<double>(blas, st, 'N', 'T', q, oa * va, ob * vb, src.factors.bov_b.get(), m.get(),
                    bar_b.bov_a.get(), 1.0, 1.0);
    rm_gemm<double>(blas, st, 'N', 'N', q, ob * vb, oa * va, src.factors.bov_a.get(), m.get(),
                    bar_b.bov_b.get(), 1.0, 1.0);
  }

  // goooo -> bar_boo
  if (oa) {
    DeviceBuf<double> m(dev, oa * oa * oa * oa, st);
    pullback_goooo_ss_kernel<<<grid_1d(oa * oa * oa * oa), 256, 0, st>>>(oa, bc.goooo_aa.get(),
                                                                         m.get());
    cuda_check(cudaGetLastError(), "pullback_goooo_aa");
    rm_gemm<double>(blas, st, 'N', 'N', q, oa * oa, oa * oa, src.factors.boo_a.get(), m.get(),
                    bar_b.boo_a.get(), 1.0, 1.0);
  }
  if (ob) {
    DeviceBuf<double> m(dev, ob * ob * ob * ob, st);
    pullback_goooo_ss_kernel<<<grid_1d(ob * ob * ob * ob), 256, 0, st>>>(ob, bc.goooo_bb.get(),
                                                                         m.get());
    cuda_check(cudaGetLastError(), "pullback_goooo_bb");
    rm_gemm<double>(blas, st, 'N', 'N', q, ob * ob, ob * ob, src.factors.boo_b.get(), m.get(),
                    bar_b.boo_b.get(), 1.0, 1.0);
  }
  if (oa && ob) {
    DeviceBuf<double> m(dev, oa * oa * ob * ob, st);
    // bc.goooo_ab[p, r, i, j] -> m[p, i, r, j]
    launch_permute4d<double, 0, 2, 1, 3>(oa, ob, oa, ob, bc.goooo_ab.get(), m.get(), st);
    rm_gemm<double>(blas, st, 'N', 'T', q, oa * oa, ob * ob, src.factors.boo_b.get(), m.get(),
                    bar_b.boo_a.get(), 1.0, 1.0);
    rm_gemm<double>(blas, st, 'N', 'N', q, ob * ob, oa * oa, src.factors.boo_a.get(), m.get(),
                    bar_b.boo_b.get(), 1.0, 1.0);
  }

  // gooov -> bar_boo, bar_bov
  if (oa && va) {
    DeviceBuf<double> m(dev, oa * oa * oa * va, st);
    pullback_gooov_ss_kernel<<<grid_1d(oa * oa * oa * va), 256, 0, st>>>(oa, va, bc.gooov_aa.get(),
                                                                         m.get());
    cuda_check(cudaGetLastError(), "pullback_gooov_aa");
    rm_gemm<double>(blas, st, 'N', 'T', q, oa * oa, oa * va, src.factors.bov_a.get(), m.get(),
                    bar_b.boo_a.get(), 1.0, 1.0);
    rm_gemm<double>(blas, st, 'N', 'N', q, oa * va, oa * oa, src.factors.boo_a.get(), m.get(),
                    bar_b.bov_a.get(), 1.0, 1.0);
  }
  if (ob && vb) {
    DeviceBuf<double> m(dev, ob * ob * ob * vb, st);
    pullback_gooov_ss_kernel<<<grid_1d(ob * ob * ob * vb), 256, 0, st>>>(ob, vb, bc.gooov_bb.get(),
                                                                         m.get());
    cuda_check(cudaGetLastError(), "pullback_gooov_bb");
    rm_gemm<double>(blas, st, 'N', 'T', q, ob * ob, ob * vb, src.factors.bov_b.get(), m.get(),
                    bar_b.boo_b.get(), 1.0, 1.0);
    rm_gemm<double>(blas, st, 'N', 'N', q, ob * vb, ob * ob, src.factors.boo_b.get(), m.get(),
                    bar_b.bov_b.get(), 1.0, 1.0);
  }
  if (oa && ob && vb) {
    DeviceBuf<double> m(dev, oa * oa * ob * vb, st);
    // bc.gooov_ab[p, r, i, e] -> m[p, i, r, e]
    launch_permute4d<double, 0, 2, 1, 3>(oa, ob, oa, vb, bc.gooov_ab.get(), m.get(), st);
    rm_gemm<double>(blas, st, 'N', 'T', q, oa * oa, ob * vb, src.factors.bov_b.get(), m.get(),
                    bar_b.boo_a.get(), 1.0, 1.0);
    rm_gemm<double>(blas, st, 'N', 'N', q, ob * vb, oa * oa, src.factors.boo_a.get(), m.get(),
                    bar_b.bov_b.get(), 1.0, 1.0);
  }
  if (ob && oa && va) {
    DeviceBuf<double> m(dev, ob * ob * oa * va, st);
    // bc.gooov_ba[p, r, i, e] -> m[p, i, r, e]
    launch_permute4d<double, 0, 2, 1, 3>(ob, oa, ob, va, bc.gooov_ba.get(), m.get(), st);
    rm_gemm<double>(blas, st, 'N', 'T', q, ob * ob, oa * va, src.factors.bov_a.get(), m.get(),
                    bar_b.boo_b.get(), 1.0, 1.0);
    rm_gemm<double>(blas, st, 'N', 'N', q, oa * va, ob * ob, src.factors.boo_b.get(), m.get(),
                    bar_b.bov_a.get(), 1.0, 1.0);
  }

  // govov -> bar_boo, bar_bvv, bar_bov
  if (oa && va) {
    DeviceBuf<double> mc(dev, oa * oa * va * va, st);
    DeviceBuf<double> mx(dev, oa * va * oa * va, st);
    // bc.govov_aa[m, b, j, e] -> mc[m, j, b, e]
    launch_permute4d<double, 0, 2, 1, 3>(oa, va, oa, va, bc.govov_aa.get(), mc.get(), st);
    pullback_govov_ss_mx_kernel<<<grid_1d(oa * va * oa * va), 256, 0, st>>>(
        oa, va, bc.govov_aa.get(), mx.get());
    cuda_check(cudaGetLastError(), "pullback_govov_aa_mx");
    rm_gemm<double>(blas, st, 'N', 'T', q, oa * oa, va * va, src.factors.bvv_a.get(), mc.get(),
                    bar_b.boo_a.get(), 1.0, 1.0);
    rm_gemm<double>(blas, st, 'N', 'N', q, va * va, oa * oa, src.factors.boo_a.get(), mc.get(),
                    bar_b.bvv_a.get(), 1.0, 1.0);
    rm_gemm<double>(blas, st, 'N', 'N', q, oa * va, oa * va, src.factors.bov_a.get(), mx.get(),
                    bar_b.bov_a.get(), -1.0, 1.0);
  }
  if (ob && vb) {
    DeviceBuf<double> mc(dev, ob * ob * vb * vb, st);
    DeviceBuf<double> mx(dev, ob * vb * ob * vb, st);
    launch_permute4d<double, 0, 2, 1, 3>(ob, vb, ob, vb, bc.govov_bb.get(), mc.get(), st);
    pullback_govov_ss_mx_kernel<<<grid_1d(ob * vb * ob * vb), 256, 0, st>>>(
        ob, vb, bc.govov_bb.get(), mx.get());
    cuda_check(cudaGetLastError(), "pullback_govov_bb_mx");
    rm_gemm<double>(blas, st, 'N', 'T', q, ob * ob, vb * vb, src.factors.bvv_b.get(), mc.get(),
                    bar_b.boo_b.get(), 1.0, 1.0);
    rm_gemm<double>(blas, st, 'N', 'N', q, vb * vb, ob * ob, src.factors.boo_b.get(), mc.get(),
                    bar_b.bvv_b.get(), 1.0, 1.0);
    rm_gemm<double>(blas, st, 'N', 'N', q, ob * vb, ob * vb, src.factors.bov_b.get(), mx.get(),
                    bar_b.bov_b.get(), -1.0, 1.0);
  }
  if (oa && vb) {
    DeviceBuf<double> mc(dev, oa * oa * vb * vb, st);
    launch_permute4d<double, 0, 2, 1, 3>(oa, vb, oa, vb, bc.govov_ab.get(), mc.get(), st);
    rm_gemm<double>(blas, st, 'N', 'T', q, oa * oa, vb * vb, src.factors.bvv_b.get(), mc.get(),
                    bar_b.boo_a.get(), 1.0, 1.0);
    rm_gemm<double>(blas, st, 'N', 'N', q, vb * vb, oa * oa, src.factors.boo_a.get(), mc.get(),
                    bar_b.bvv_b.get(), 1.0, 1.0);
  }
  if (ob && va) {
    DeviceBuf<double> mc(dev, ob * ob * va * va, st);
    launch_permute4d<double, 0, 2, 1, 3>(ob, va, ob, va, bc.govov_ba.get(), mc.get(), st);
    rm_gemm<double>(blas, st, 'N', 'T', q, ob * ob, va * va, src.factors.bvv_a.get(), mc.get(),
                    bar_b.boo_b.get(), 1.0, 1.0);
    rm_gemm<double>(blas, st, 'N', 'N', q, va * va, ob * ob, src.factors.boo_b.get(), mc.get(),
                    bar_b.bvv_a.get(), 1.0, 1.0);
  }
}

SpinBlockedSourcePullback pullback_spin_blocked_df_source_cuda_impl(
    const core::System& orbital, const core::System& auxiliary, const CudaSpinBlockedSource& src,
    const DeviceFactorGrads& bar_b, std::size_t budget_bytes) {
  const auto& d = src.dims;
  const int dev = src.device;
  const cudaStream_t st = src.stream;
  const cublasHandle_t blas = src.blas;
  auto& plan = *src.owned_plan;
  if (plan.metric_response_valid.size() != 1 || !plan.metric_response_valid[0]) {
    throw std::runtime_error("DF-UCC v2 CUDA metric rank crossing: response branch unresolved");
  }

  runtime::CudaDeviceScope dev_scope(dev);
  const std::size_t n = d.n, q = d.q;
  const auto cap = budget_bytes ? std::min(src.device_cap, budget_bytes) : src.device_cap;
  const auto matrix = checked_mul(n, n);
  const auto metric = checked_mul(q, q);
  const auto row = checked_mul(n, q);
  const auto cap_plan =
      ucc_source_pullback_capacity(d, source_resident_bytes(src), bar_b.total_bytes());
  if (cap_plan.serial_bytes >= cap)
    throw std::length_error("DF-UCC v2 CUDA source pullback exceeds memory budget");
  const auto sink_budget = cap - cap_plan.serial_bytes;
  scf::CudaDfNuclearSink sink(dev, orbital, auxiliary, sink_budget);
  const auto u_bytes =
      checked_mul(checked_mul(row, checked_add(d.n_act_a(), d.n_act_b())), sizeof(double));
  const bool fuse_ao_rows =
      d.n_act_a() && d.n_act_b() && u_bytes <= query_free_device_bytes(dev) / 2 &&
      cap_plan.extra_fused_bytes <= sink_budget - sink.numeric_capacity_bytes();

  SpinBlockedSourcePullback out;
  out.bar_c_alpha.assign(matrix, 0.0);
  out.bar_c_beta.assign(matrix, 0.0);
  DeviceBuf<double> d_bar_w(dev, metric, st);
  d_bar_w.zero(st);
  DeviceBuf<double> d_row(dev, row, st);

  auto compute_spin_u = [&](std::size_t o_full, std::size_t frozen, std::size_t o, std::size_t v,
                            const double* d_c_act, const double* d_bar_boo,
                            const double* d_bar_bov, const double* d_bar_bvv,
                            std::vector<double>& bar_c_full, DeviceBuf<double>& d_u_out) {
    const std::size_t n_act = o + v;
    if (!n_act) return;
    const std::size_t t1_elems = checked_mul(n, checked_mul(n_act, q));
    const std::size_t t2_elems = checked_mul(checked_mul(n_act, n_act), q);

    // 1. Regenerate T1[mu, r, P] = sum_nu C_act[nu, r] raw[mu, nu, P]
    DeviceBuf<double> d_t1(dev, t1_elems, st);
    std::string detail;
    for (std::size_t mu = 0; mu < n; ++mu) {
      check_gqc_status(scf::generate_cuda_density_fitting_raw_tile(
                           plan.integral_source, 0, mu * n, n, 0, q, -1, st, d_row.get(), detail),
                       detail);
      rm_gemm<double>(blas, st, 'T', 'N', n_act, q, n, d_c_act, d_row.get(),
                      d_t1.get() + mu * n_act * q, 1.0, 0.0);
    }

    // 2. T2[p, r, P] = C_act^T @ T1, and embed symmetric bar_bmo[p, r, Q]
    DeviceBuf<double> d_t2(dev, t2_elems, st);
    rm_gemm<double>(blas, st, 'T', 'N', n_act, n_act * q, n, d_c_act, d_t1.get(), d_t2.get(), 1.0,
                    0.0);
    DeviceBuf<double> d_bar_bmo(dev, t2_elems, st);
    embed_qmajor_bar_bmo_kernel<<<grid_1d(t2_elems), 256, 0, st>>>(
        q, o, v, d_bar_boo, d_bar_bov, d_bar_bvv, d_bar_bmo.get());
    cuda_check(cudaGetLastError(), "embed_qmajor_bar_bmo");

    // 3. Accumulate metric whitener cotangent: bar_W[P, Q] += T2^T @ bar_bmo
    rm_gemm<double>(blas, st, 'T', 'N', q, q, n_act * n_act, d_t2.get(), d_bar_bmo.get(),
                    d_bar_w.get(), 1.0, 1.0);

    // 4. bar_T2[p, r, P] = bar_bmo[p*r, Q] @ W^T[Q, P] (reuses d_t2 storage)
    rm_gemm<double>(blas, st, 'N', 'T', n_act * n_act, q, q, d_bar_bmo.get(),
                    plan.inverse_square_roots, d_t2.get(), 1.0, 0.0);
    d_bar_bmo.reset();

    // 5. bar_C_act[mu, p] = 2 * sum_{r, P} T1[mu, r, P] * bar_T2[p, r, P]
    DeviceBuf<double> d_bar_c_act(dev, n * n_act, st);
    rm_gemm<double>(blas, st, 'N', 'T', n, n_act, n_act * q, d_t1.get(), d_t2.get(),
                    d_bar_c_act.get(), 2.0, 0.0);
    std::vector<double> host_bar_c_act(n * n_act, 0.0);
    d_bar_c_act.copy_to_host(host_bar_c_act.data(), host_bar_c_act.size(), st);
    cuda_check(cudaStreamSynchronize(st), "bar_c_act sync");
    for (std::size_t mu = 0; mu < n; ++mu) {
      for (std::size_t i = 0; i < o; ++i)
        bar_c_full[mu * n + (frozen + i)] = host_bar_c_act[mu * n_act + i];
      for (std::size_t a = 0; a < v; ++a)
        bar_c_full[mu * n + (o_full + a)] = host_bar_c_act[mu * n_act + o + a];
    }
    d_bar_c_act.reset();

    // 6. U[mu, r, P] = sum_p C_act[mu, p] * bar_T2[p, r, P] (reuses d_t1 storage!)
    rm_gemm<double>(blas, st, 'N', 'N', n, n_act * q, n_act, d_c_act, d_t2.get(), d_t1.get(), 1.0,
                    0.0);
    d_t2.reset();
    d_u_out = std::move(d_t1);
  };

  if (fuse_ao_rows) {
    DeviceBuf<double> d_ua, d_ub;
    compute_spin_u(d.oa_full, d.fa, d.oa, d.va, src.c_act_a.get(), bar_b.boo_a.get(),
                   bar_b.bov_a.get(), bar_b.bvv_a.get(), out.bar_c_alpha, d_ua);
    compute_spin_u(d.ob_full, d.fb, d.ob, d.vb, src.c_act_b.get(), bar_b.boo_b.get(),
                   bar_b.bov_b.get(), bar_b.bvv_b.get(), out.bar_c_beta, d_ub);
    for (std::size_t mu = 0; mu < n; ++mu) {
      rm_gemm<double>(blas, st, 'N', 'N', n, q, d.n_act_a(), src.c_act_a.get(),
                      d_ua.get() + mu * d.n_act_a() * q, d_row.get(), 1.0, 0.0);
      rm_gemm<double>(blas, st, 'N', 'N', n, q, d.n_act_b(), src.c_act_b.get(),
                      d_ub.get() + mu * d.n_act_b() * q, d_row.get(), 1.0, 1.0);
      sink.consume(0, {mu * n * q, 1, 1, 1}, d_row.get(), n * q, st);
    }
  } else {
    auto stream_spin = [&](std::size_t o_full, std::size_t frozen, std::size_t o, std::size_t v,
                           const double* d_c_act, const double* d_bar_boo,
                           const double* d_bar_bov, const double* d_bar_bvv,
                           std::vector<double>& bar_c_full) {
      const std::size_t n_act = o + v;
      if (!n_act) return;
      DeviceBuf<double> d_u;
      compute_spin_u(o_full, frozen, o, v, d_c_act, d_bar_boo, d_bar_bov, d_bar_bvv, bar_c_full,
                     d_u);
      for (std::size_t mu = 0; mu < n; ++mu) {
        rm_gemm<double>(blas, st, 'N', 'N', n, q, n_act, d_c_act, d_u.get() + mu * n_act * q,
                        d_row.get(), 1.0, 0.0);
        sink.consume(0, {mu * n * q, 1, 1, 1}, d_row.get(), n * q, st);
      }
    };
    stream_spin(d.oa_full, d.fa, d.oa, d.va, src.c_act_a.get(), bar_b.boo_a.get(),
                bar_b.bov_a.get(), bar_b.bvv_a.get(), out.bar_c_alpha);
    stream_spin(d.ob_full, d.fb, d.ob, d.vb, src.c_act_b.get(), bar_b.boo_b.get(),
                bar_b.bov_b.get(), bar_b.bvv_b.get(), out.bar_c_beta);
  }
  d_row.reset();

  // Exact Daleckii-Krein spectral VJP for M^{-1/2}: bar_W -> bar_M.
  // plan.metric_eigenvectors stores column-major Q, which in row-major is Q^T.
  DeviceBuf<double> d_sym(dev, q * q, st);
  DeviceBuf<double> d_tmp(dev, q * q, st);
  DeviceBuf<double> d_bar_metric(dev, q * q, st);
  DeviceBuf<int> d_bad(dev, 1, st);
  d_bad.zero(st);

  symmetrize_matrix_kernel<<<grid_1d(q * q), 256, 0, st>>>(q, d_bar_w.get(), d_sym.get());
  cuda_check(cudaGetLastError(), "symmetrize bar_w");
  // tmp = Q^T @ S ; bar_w_eig = tmp @ Q
  rm_gemm<double>(blas, st, 'N', 'N', q, q, q, plan.metric_eigenvectors, d_sym.get(), d_tmp.get(),
                  1.0, 0.0);
  rm_gemm<double>(blas, st, 'N', 'T', q, q, q, d_tmp.get(), plan.metric_eigenvectors, d_sym.get(),
                  1.0, 0.0);
  metric_inv_sqrt_dk_kernel<<<grid_1d(q * q), 256, 0, st>>>(
      q, plan.metric_eigenvalues, plan.metric_relative_threshold, d_sym.get(), d_bad.get());
  cuda_check(cudaGetLastError(), "metric_inv_sqrt_dk");
  // bar_metric = Q @ d_sym @ Q^T
  rm_gemm<double>(blas, st, 'T', 'N', q, q, q, plan.metric_eigenvectors, d_sym.get(), d_tmp.get(),
                  1.0, 0.0);
  rm_gemm<double>(blas, st, 'N', 'N', q, q, q, d_tmp.get(), plan.metric_eigenvectors,
                  d_bar_metric.get(), 1.0, 0.0);
  symmetrize_matrix_kernel<<<grid_1d(q * q), 256, 0, st>>>(q, d_bar_metric.get(), d_sym.get());
  cuda_check(cudaGetLastError(), "symmetrize bar_metric");
  audit_finite_kernel<<<grid_1d(q * q), 256, 0, st>>>(d_sym.get(), q * q, d_bad.get());
  cuda_check(cudaGetLastError(), "audit bar_metric");

  int bad = 0;
  d_bad.copy_to_host(&bad, 1, st);
  cuda_check(cudaStreamSynchronize(st), "metric vjp sync");
  if (bad) throw std::runtime_error("DF-UCC v2 CUDA metric inverse-sqrt VJP produced nonfinite");

  sink.consume(1, {0, 1, 1, 1}, d_sym.get(), q * q, st);
  out.corr_gradient = sink.finish();
  return out;
}

}  // namespace generativeqc::cc::v2_detail::cuda_detail
