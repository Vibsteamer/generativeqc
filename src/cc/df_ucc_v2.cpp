#include "cc/df_ucc_v2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "cc/df_ucc_v2_internal.hpp"

namespace generativeqc::cc {

DFUccV2Result run_df_ucc_v2_cpu(const core::System& orbital, const core::System& auxiliary,
                                const hf::UnrestrictedPhysicalReference& reference,
                                const DFUccV2Options& options) {
  if (options.triples_fp32_w && options.triples_compensated_w)
    throw std::invalid_argument("DF-UCC v2 requires one triples W precision policy");
  if (options.compute_forces && options.energy_source_only && !options.compute_triples) {
    throw std::invalid_argument(
        "DF-UCC v2 CPU energy_source_only forces require compute_triples=true");
  }

  using clock = std::chrono::steady_clock;
  const auto t_start = clock::now();

  const auto src =
      v2_detail::build_spin_blocked_df_source_cpu(orbital, auxiliary, reference, options);
  const auto core = v2_detail::build_four_index_core_cpu(src);
  const auto ccsd = v2_detail::solve_df_uccsd_cpu(src, core, options);
  const auto t_ccsd_end = clock::now();

  DFUccV2Result result;
  result.ccsd_correlation_energy = ccsd.energy;
  result.ccsd_iterations = ccsd.iterations;
  result.ccsd_fp32_executed = ccsd.fp32_executed;
  result.ccsd_seconds = std::chrono::duration<double>(t_ccsd_end - t_start).count();
  result.frozen_alpha = src.dims.fa;
  result.frozen_beta = src.dims.fb;

  v2_detail::SpinBlockedTriplesResult triples;
  if (options.compute_triples) {
    const auto t_triples_start = clock::now();
    triples = v2_detail::evaluate_df_ucc_triples_cpu(src, core, ccsd.amplitudes, options);
    const auto t_triples_end = clock::now();
    result.triples_energy = triples.energy;
    result.triples_fp32_gemms = triples.fp32_gemms;
    result.triples_fp64_gemms = triples.fp64_gemms;
    result.triples_fp32_reverse_gemms = triples.fp32_reverse_gemms;
    result.triples_compensated_w_executed = options.triples_compensated_w && triples.fp32_gemms > 0;
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
    auto lambda = v2_detail::solve_df_ulambda_cpu(
        src, core, ccsd.amplitudes, options.compute_triples ? &triples : nullptr, options);
    const auto t_lambda_end = clock::now();
    result.lambda_seconds = std::chrono::duration<double>(t_lambda_end - t_lambda_start).count();
    result.lambda_iterations = lambda.iterations;
    result.lambda_fp32_executed = lambda.fp32_executed;
    if (options.f12) {
      result.f12_cotangents = std::move(lambda.f12_cotangents);
    }

    const auto t_resp_start = clock::now();
    auto pulled = v2_detail::pullback_spin_blocked_df_source_cpu(orbital, auxiliary, src,
                                                                 lambda.bar_b);
    v2_detail::embed_fock_cotangents(src.dims, lambda.bar_f, result.bar_f_alpha, result.bar_f_beta);
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
      throw std::runtime_error("DF-UCC v2 CPU produced nonfinite force source cotangents");
    }
    if (options.f12) {
      const auto& fc = result.f12_cotangents;
      if (!std::isfinite(fc.bar_fixed_geminal_energy) || !all_finite(fc.bar_w_vo_aaaa) ||
          !all_finite(fc.bar_w_vo_bbbb) || !all_finite(fc.bar_w_vo_abab) ||
          !all_finite(fc.bar_w_vo_abba) || !all_finite(fc.bar_w_vv_aaaa) ||
          !all_finite(fc.bar_w_vv_abab) || !all_finite(fc.bar_w_vv_bbbb) ||
          !all_finite(fc.bar_c_f12_aaaa) || !all_finite(fc.bar_c_f12_abab) ||
          !all_finite(fc.bar_c_f12_bbbb)) {
        throw std::runtime_error("DF-UCC v2 CPU produced nonfinite F12 cotangents");
      }
    }
  }

  if (!std::isfinite(result.ccsd_correlation_energy) || !std::isfinite(result.triples_energy)) {
    throw std::runtime_error("DF-UCC v2 CPU produced nonfinite energy");
  }
  return result;
}

DFUccV2Result run_df_ucc_v2(const core::System& orbital, const core::System& auxiliary,
                            const hf::UnrestrictedPhysicalReference& reference, bool cuda,
                            int device, const DFUccV2Options& options) {
  if (cuda) {
#if GENERATIVEQC_HAS_CUDA
    return run_df_ucc_v2_cuda(orbital, auxiliary, reference, device, options);
#else
    (void)device;
    throw std::runtime_error("DF-UCC v2 CUDA backend is not enabled in this build");
#endif
  }
  return run_df_ucc_v2_cpu(orbital, auxiliary, reference, options);
}

}  // namespace generativeqc::cc
