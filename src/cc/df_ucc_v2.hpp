#pragma once

#include <cstddef>
#include <vector>

#include "core/correlation_execution.hpp"
#include "core/types.hpp"
#include "hf/reference.hpp"

namespace generativeqc::cc {

/** Optional fixed-cusp F12b spin-blocked coupling bundle.
 * When non-null, injects the exact Knizia fixed-cusp F12b residual and energy
 * contributions into the 5-block DF-UCCSD primal solver and DF-ULambda solver,
 * and exports exact stationary source cotangents when compute_forces is true.
 * When nullptr, pure conventional DF-UCCSD(T)/(T+) runs with zero overhead.
 *
 * Independent spin-conserving tensor shapes (active occupied oa/ob, virtual va/vb):
 *   w_vo_aaaa : [oa, oa, va, oa]   (antisymmetric in (i,j))
 *   w_vo_bbbb : [ob, ob, vb, ob]   (antisymmetric in (i,j))
 *   w_vo_abab : [oa, ob, va, ob]   (w_vo_baab[j,i,a,k] = -w_vo_abab[i,j,a,k])
 *   w_vo_abba : [oa, ob, vb, oa]   (w_vo_baba[j,i,a,k] = -w_vo_abba[i,j,a,k])
 *   w_vv_aaaa : [oa, oa, va, va]   (antisymmetric in (i,j) and (a,b))
 *   w_vv_abab : [oa, ob, va, vb]
 *   w_vv_bbbb : [ob, ob, vb, vb]   (antisymmetric in (i,j) and (a,b))
 *   c_f12_aaaa: [oa, oa, va, va]   (antisymmetric in (i,j) and (a,b))
 *   c_f12_abab: [oa, ob, va, vb]
 *   c_f12_bbbb: [ob, ob, vb, vb]   (antisymmetric in (i,j) and (a,b))
 */
struct FixedF12Coupling {
  double fixed_geminal_energy{};
  std::vector<double> w_vo_aaaa;
  std::vector<double> w_vo_bbbb;
  std::vector<double> w_vo_abab;
  std::vector<double> w_vo_abba;
  std::vector<double> w_vv_aaaa;
  std::vector<double> w_vv_abab;
  std::vector<double> w_vv_bbbb;
  std::vector<double> c_f12_aaaa;
  std::vector<double> c_f12_abab;
  std::vector<double> c_f12_bbbb;
};

/** Stationary cotangents of FixedF12Coupling exported by DF-ULambda when f12 != nullptr. */
struct FixedF12Cotangents {
  double bar_fixed_geminal_energy{};
  std::vector<double> bar_w_vo_aaaa;
  std::vector<double> bar_w_vo_bbbb;
  std::vector<double> bar_w_vo_abab;
  std::vector<double> bar_w_vo_abba;
  std::vector<double> bar_w_vv_aaaa;
  std::vector<double> bar_w_vv_abab;
  std::vector<double> bar_w_vv_bbbb;
  std::vector<double> bar_c_f12_aaaa;
  std::vector<double> bar_c_f12_abab;
  std::vector<double> bar_c_f12_bbbb;
};

/** Configuration for the native spin-blocked DF-UCCSD(T)/(T+) / UCCSD-F12b v2 engine. */
struct DFUccV2Options {
  bool compute_forces{false};
  bool compute_triples{true};
  bool occupied_partition{false};
  bool energy_source_only{true};
  std::size_t frozen_alpha{0};
  std::size_t frozen_beta{0};
  std::vector<double> occupied_weights_alpha;
  std::vector<double> occupied_weights_beta;
  core::DfPreconvergence preconvergence{core::DfPreconvergence::Automatic};
  bool triples_fp32_w{false};
  bool triples_compensated_w{false};  // Three-limb FP32 products and FP64 recombination.
  std::size_t max_bytes{4ULL << 30};
  int max_ccsd_iterations{120};
  int max_lambda_iterations{120};
  int diis_space{8};
  double energy_tolerance{1e-12};
  double residual_tolerance{1e-11};
  double lambda_tolerance{1e-11};
  double denominator_threshold{1e-10};
  double metric_relative_threshold{1e-10};
  const FixedF12Coupling* f12{nullptr};
};

/** Spin-blocked DF-UCCSD(T)/(T+) / UCCSD-F12b outputs, force sources, and telemetry. */
struct DFUccV2Result {
  double ccsd_correlation_energy{};
  double triples_energy{};
  std::vector<double> occupied_alpha;
  std::vector<double> occupied_beta;
  std::vector<double> correlation_gradient;
  std::vector<double> bar_f_alpha;
  std::vector<double> bar_f_beta;
  std::vector<double> bar_c_alpha;
  std::vector<double> bar_c_beta;
  FixedF12Cotangents f12_cotangents;

  bool ccsd_fp32_executed{};
  bool lambda_fp32_executed{};
  std::size_t triples_fp32_gemms{};
  std::size_t triples_fp64_gemms{};
  std::size_t triples_fp32_reverse_gemms{};
  bool triples_compensated_w_executed{};
  double ccsd_seconds{};
  double triples_seconds{};
  double lambda_seconds{};
  double source_response_seconds{};
  std::size_t frozen_alpha{};
  std::size_t frozen_beta{};
  int ccsd_iterations{};
  int lambda_iterations{};
};

/** Host OpenBLAS-backed spin-blocked DF-UCCSD(T)/(T+) / UCCSD-F12b v2 engine.
 * Shares the exact same spin-blocked DF equations (zero 4-index ovvv/vvvv,
 * independent frozen_alpha/frozen_beta, 4-channel weighted (T+), Delta-Lambda
 * DIIS, optional FixedF12Coupling, and DF nuclear gradient pullback) as CUDA.
 */
DFUccV2Result run_df_ucc_v2_cpu(const core::System& orbital, const core::System& auxiliary,
                                const hf::UnrestrictedPhysicalReference& reference,
                                const DFUccV2Options& options);

#if GENERATIVEQC_HAS_CUDA
/** GPU-resident spin-blocked DF-UCCSD(T)/(T+) / UCCSD-F12b v2 CUDA engine.
 * Keeps 3-index DF factors, 5-block UCCSD + GPU DIIS, 4-channel (T)/(T+)
 * with fused GPU pullback, optional FixedF12Coupling, and Delta-Lambda GPU DIIS
 * completely on device, supporting FP32 warmup + FP64 polish and optional FP32 forward-W.
 */
DFUccV2Result run_df_ucc_v2_cuda(const core::System& orbital, const core::System& auxiliary,
                                 const hf::UnrestrictedPhysicalReference& reference, int device,
                                 const DFUccV2Options& options);
#endif

/** Backend-dispatched spin-blocked DF-UCCSD(T)/(T+) / UCCSD-F12b v2 engine. */
DFUccV2Result run_df_ucc_v2(const core::System& orbital, const core::System& auxiliary,
                            const hf::UnrestrictedPhysicalReference& reference, bool cuda,
                            int device, const DFUccV2Options& options);

}  // namespace generativeqc::cc
