#include "cc/df_ucc_v2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "cc/df_ucc_v2_cuda_internal.cuh"

namespace generativeqc::cc {

DFUccV2Result run_df_ucc_v2_cuda(const core::System& orbital, const core::System& auxiliary,
                                 const hf::UnrestrictedPhysicalReference& reference, int device,
                                 const DFUccV2Options& options) {
  if (options.triples_compensated_w)
    throw std::invalid_argument("Spin-blocked CUDA triples do not support compensated W");
  if (options.compute_forces && options.energy_source_only && !options.compute_triples) {
    throw std::invalid_argument(
        "DF-UCC v2 CUDA energy_source_only forces require compute_triples=true");
  }

  using clock = std::chrono::steady_clock;
  const auto t_start = clock::now();

  const auto src = v2_detail::cuda_detail::build_spin_blocked_df_source_cuda_impl(
      orbital, auxiliary, reference, device, options);
  const v2_detail::UccDeviceCapacityScope capacity(src.device_cap,
      v2_detail::cuda_detail::source_resident_bytes(src));
  auto core = v2_detail::cuda_detail::build_four_index_core_cuda_impl(src);
  auto ccsd = v2_detail::cuda_detail::solve_df_uccsd_cuda_impl(src, core, options);
  const auto t_ccsd_end = clock::now();

  DFUccV2Result result;
  result.ccsd_correlation_energy = ccsd.energy;
  result.ccsd_iterations = ccsd.iterations;
  result.ccsd_fp32_executed = ccsd.fp32_executed;
  result.ccsd_seconds = std::chrono::duration<double>(t_ccsd_end - t_start).count();
  result.frozen_alpha = src.dims.fa;
  result.frozen_beta = src.dims.fb;

  v2_detail::cuda_detail::CudaTriplesResult triples;
  if (options.compute_triples) {
    const auto t_triples_start = clock::now();
    triples = v2_detail::cuda_detail::evaluate_df_ucc_triples_cuda_impl(
        src, core, ccsd.amplitudes, options);
    const auto t_triples_end = clock::now();
    result.triples_energy = triples.energy;
    result.triples_fp32_gemms = triples.fp32_gemms;
    result.triples_fp64_gemms = triples.fp64_gemms;
    result.triples_seconds =
        std::chrono::duration<double>(t_triples_end - t_triples_start).count();
    if (options.occupied_partition) {
      result.occupied_alpha = std::move(triples.occupied_alpha);
      result.occupied_beta = std::move(triples.occupied_beta);
    }
  } else if (options.occupied_partition) {
    result.occupied_alpha.assign(src.dims.oa, 0.0);
    result.occupied_beta.assign(src.dims.ob, 0.0);
  }

  if (options.compute_forces) {
    const auto t_lambda_start = clock::now();
    auto lambda = v2_detail::cuda_detail::solve_df_ulambda_cuda_impl(
        src, core, ccsd.amplitudes, options.compute_triples ? &triples : nullptr, options);
    const auto t_lambda_end = clock::now();
    result.lambda_seconds = std::chrono::duration<double>(t_lambda_end - t_lambda_start).count();
    result.lambda_iterations = lambda.iterations;
    result.lambda_fp32_executed = lambda.fp32_executed;
    if (options.f12) {
      result.f12_cotangents = std::move(lambda.f12_cotangents);
    }

    // Drop solved GPU intermediates before overlapping device and host gradient outputs.
    triples = {};
    ccsd.amplitudes.flat.reset();
    core = {};
    using posthf::checked_add;
    using posthf::checked_mul;
    const auto& f = lambda.bar_f;
    const auto retained_outputs = checked_add(
        checked_add(result.occupied_alpha.size(), result.occupied_beta.size()),
        v2_detail::ucc_f12_elements(options.f12));
    std::size_t host_bytes = checked_mul(retained_outputs, sizeof(double));
    for (const auto bytes : {f.foo_a.bytes(), f.fov_a.bytes(), f.fvv_a.bytes(),
                             f.foo_b.bytes(), f.fov_b.bytes(), f.fvv_b.bytes()})
      host_bytes = checked_add(host_bytes, bytes);
    const auto host_charge_bytes = options.max_bytes ? host_bytes : 0;
    const v2_detail::UccDeviceCharge retained_host_outputs(host_charge_bytes);
    const auto response_cap = src.device_cap - host_charge_bytes;

    v2_detail::SpinBlockedFockGrads host_bar_f;
    host_bar_f.allocate_zero(src.dims);
    {
      runtime::CudaDeviceScope dev_scope(src.device);
      lambda.bar_f.foo_a.copy_to_host(host_bar_f.foo_a.data(), host_bar_f.foo_a.size(), src.stream);
      lambda.bar_f.fov_a.copy_to_host(host_bar_f.fov_a.data(), host_bar_f.fov_a.size(), src.stream);
      lambda.bar_f.fvv_a.copy_to_host(host_bar_f.fvv_a.data(), host_bar_f.fvv_a.size(), src.stream);
      lambda.bar_f.foo_b.copy_to_host(host_bar_f.foo_b.data(), host_bar_f.foo_b.size(), src.stream);
      lambda.bar_f.fov_b.copy_to_host(host_bar_f.fov_b.data(), host_bar_f.fov_b.size(), src.stream);
      lambda.bar_f.fvv_b.copy_to_host(host_bar_f.fvv_b.data(), host_bar_f.fvv_b.size(), src.stream);
      v2_detail::cuda_detail::cuda_check(cudaStreamSynchronize(src.stream), "download bar_f sync");
    }
    lambda.bar_f = {};

    const auto t_resp_start = clock::now();
    auto pulled = v2_detail::cuda_detail::pullback_spin_blocked_df_source_cuda_impl(
        orbital, auxiliary, src, lambda.bar_b, response_cap);
    lambda.bar_b = {};
    // The full Fock outputs overlap the two coefficient cotangents and force vector.
    const auto output_elems = checked_add(
        checked_mul(4, checked_mul(src.dims.n, src.dims.n)), pulled.corr_gradient.size());
    if (options.max_bytes && checked_mul(output_elems, sizeof(double)) >
                                 v2_detail::ucc_device_available(response_cap))
      throw std::length_error("DF-UCC v2 host gradient outputs exceed memory budget");
    v2_detail::embed_fock_cotangents(src.dims, host_bar_f, result.bar_f_alpha, result.bar_f_beta);
    result.correlation_gradient = std::move(pulled.corr_gradient);
    result.bar_c_alpha = std::move(pulled.bar_c_alpha);
    result.bar_c_beta = std::move(pulled.bar_c_beta);
    const auto t_resp_end = clock::now();
    result.source_response_seconds =
        std::chrono::duration<double>(t_resp_end - t_resp_start).count();

    auto all_finite = [](const std::vector<double>& v) {
      return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
    };
    if (!all_finite(result.correlation_gradient) || !all_finite(result.bar_f_alpha) ||
        !all_finite(result.bar_f_beta) || !all_finite(result.bar_c_alpha) ||
        !all_finite(result.bar_c_beta)) {
      throw std::runtime_error("DF-UCC v2 CUDA produced nonfinite force source cotangents");
    }
    if (options.f12) {
      const auto& fc = result.f12_cotangents;
      if (!std::isfinite(fc.bar_fixed_geminal_energy) || !all_finite(fc.bar_w_vo_aaaa) ||
          !all_finite(fc.bar_w_vo_bbbb) || !all_finite(fc.bar_w_vo_abab) ||
          !all_finite(fc.bar_w_vo_abba) || !all_finite(fc.bar_w_vv_aaaa) ||
          !all_finite(fc.bar_w_vv_abab) || !all_finite(fc.bar_w_vv_bbbb) ||
          !all_finite(fc.bar_c_f12_aaaa) || !all_finite(fc.bar_c_f12_abab) ||
          !all_finite(fc.bar_c_f12_bbbb)) {
        throw std::runtime_error("DF-UCC v2 CUDA produced nonfinite F12 cotangents");
      }
    }
  }

  if (!std::isfinite(result.ccsd_correlation_energy) || !std::isfinite(result.triples_energy)) {
    throw std::runtime_error("DF-UCC v2 CUDA produced nonfinite energy");
  }
  return result;
}

}  // namespace generativeqc::cc
