#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "runtime/execution_precision.hpp"

namespace generativeqc::cc::triples {

/** Complete standalone DF (T) evaluation, including staged inputs and BLAS allowance. */
struct DFCudaResult {
  double energy{};
  double minimum_absolute_denominator{};
  double seconds{};
  generativeqc::runtime::PrecisionDirective precision;
  std::string_view w_scientific_identity, w_semantic_identity, w_candidate_identity, w_provider,
      w_algorithm;
  int w_provider_version{}, cuda_runtime_version{};
  bool retained_incumbent{}, resource_fallback{};
  // Weighted tiles with a zero seed skip the reverse W contractions.
  bool reverse_w_executed{}, compensated_fp32_w_executed{};
  std::uint32_t w_contraction_storage_bits{64}, w_contraction_compute_bits{64};
  std::uint32_t w_contraction_accumulation_bits{64};
  std::uint32_t w_reverse_contraction_compute_bits{64};
  std::string_view w_precision_identity, w_codegen_precision_schedule_identity;
  std::size_t virtual_triples{}, occupied_tiles{};
  std::size_t workspace_bytes{}, arena_bytes{}, provider_retained_bytes{}, host_binding_bytes{};
  std::size_t panel_capacity{}, panel_gemms{}, moment_gemms{};
  std::size_t fp64_gemms{}, fp32_gemms{}, precision_cast_elements{};
  std::size_t fp32_forward_gemms{}, fp32_reverse_gemms{};
  std::size_t epilogue_kernels{}, reduction_kernels{}, epilogue_points{};
  std::size_t contraction_summands{}, h2d_bytes{}, d2h_bytes{};
  /** Optional conventional triples energies by canonical spatial occupied orbital. */
  std::vector<double> occupied_energies;

  // A bounded provider may preserve FP32 W work while changing panels or reverse arithmetic.
  [[nodiscard]] bool triples_fp32_w_forward_executed() const noexcept {
    return fp32_forward_gemms > 0;
  }

  [[nodiscard]] bool triples_fp32_w_reverse_executed() const noexcept {
    return fp32_reverse_gemms > 0;
  }

  [[nodiscard]] bool triples_fp32_w_executed() const noexcept {
    return triples_fp32_w_forward_executed() || triples_fp32_w_reverse_executed();
  }
};

/** Actual gap-schedule launches and compiler-derived logical data/work counts. */
struct DFGapReductionDiagnostic {
  bool requested{};
  bool parallel{};
  std::string_view schedule;
  std::size_t kernels{}, workspace_bytes{};
  // Cumulative logical elements, not measured device transactions or peak VRAM.
  std::size_t materialized_elements{}, value_reads{}, value_writes{}, reduction_summands{};
};

/** Compiler-owned scalar-region execution and logical work, not measured traffic. */
struct DFScalarFusionDiagnostic {
  bool requested{}, selected{}, resource_fallback{};
  std::string_view schedule;
  std::size_t tiles{}, kernels{}, avoided_kernels{}, workspace_bytes{};
  std::size_t value_reads{}, value_writes{}, arithmetic_ops{};
};

/** Complete fixed-canonical-input triples pullback; all arrays are detached.
 * Bvv is an unprojected dense Frobenius cotangent on the symmetric physical
 * factor domain. eps_o/eps_v describe canonical denominator derivatives;
 * they do not by themselves certify same-space Fock response at degeneracy.
 */
struct DFCudaResponseResult {
  DFCudaResult diagnostic;
  DFGapReductionDiagnostic gap;
  DFScalarFusionDiagnostic scalar_fusion;
  std::vector<double> bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v;
  std::size_t numeric_capacity_bytes{}, borrowed_host_bytes{};
  std::size_t reverse_gemms{}, reverse_kernels{}, audit_kernels{}, scalar_response_evaluations{};
};

/** Complete same-space Fock resolvent response, including internal degeneracy.
 * Both matrices use dense symmetric Frobenius coordinates. They replace the
 * diagonal epsilon sources; adding both would double-count denominator response.
 */
struct DFCudaFockResult {
  std::vector<double> foo, fvv;
  double seconds{}, minimum_absolute_denominator{};
  std::size_t numeric_capacity_bytes{}, borrowed_host_bytes{}, workspace_bytes{}, arena_bytes{};
  std::size_t provider_retained_bytes{}, page_capacity{}, page_count{}, panel_capacity{};
  std::size_t occupied_pairs{}, unique_vector_cubes{}, vector_cubes{}, page_builds{};
  std::size_t panel_gemms{}, w_gemms{}, fock_gemms{}, contraction_summands{};
  /** Forward-only copies into independent canonical W slots, never cotangent aliases. */
  std::size_t w_copy_calls{}, w_copy_bytes{};
  std::size_t scalar_evaluations{}, audit_kernels{}, scatter_kernels{}, h2d_bytes{}, d2h_bytes{};
};

/** Fixed-frame triples pullback plus full oo/vv Fock response with one staged
 * device-input owner. Scientific contractions, reduction order and detached
 * outputs are identical to the standalone phase APIs; only immutable input
 * staging is shared across the two sequential CUDA consumers.
 */
struct DFCudaCombinedResponseResult {
  DFCudaResponseResult pullback;
  DFCudaFockResult fock;
  std::size_t shared_input_device_bytes{}, shared_h2d_bytes{};
  /** Triangular pullback cubes whose already-built W moments also feed Fock response. */
  std::size_t shared_response_cubes{}, avoided_w_gemms{};
  double seconds{};
};

/** Standard closed-shell (T) on a supplied physical DF Hamiltonian.
 * Q-major B_ov/B_vv replace resident ovvv. Other inputs retain the canonical
 * spatial-MO layout. The owner uploads each input once, builds at most three
 * occupied integral panels and six virtual W cubes, and drains before result
 * publication. Tight budgets try one panel, then generated execution at the
 * admitted precision, then strict generated execution without changing equations.
 * max_bytes covers numeric storage, host bindings and the selected provider allowance;
 * callers composing endpoints separately charge their retained host/CC state.
 * Scientifically admitted W directives are bound by the compiler/provider layer.
 * The optional qualified variant lowers only the W reductions to FP32. W assembly,
 * V, denominators, energy epilogue and final reductions remain FP64, and the
 * result records the actual contraction precision plus the generated code's
 * compiler precision-schedule identity.
 * compensated_fp32_w selects the opt-in nine-SGEMM, three-limb policy. It requires
 * admitted FP32 W and refuses insufficient device/provider resources.
 * occupied_partition retains the contribution from each occupied orbital,
 * using Eq. (13) of Kállay et al. (10.1063/5.0057426). It adds one on-device
 * reduction and one returned vector. Standard (T) energy remains unchanged.
 */
#if GENERATIVEQC_HAS_CUDA
DFCudaResult evaluate_df_cuda(std::size_t o, std::size_t v, std::size_t q, const double* bov,
                              const double* bvv, const double* ovoo, const double* ovov,
                              const double* fov, const double* t1, const double* t2,
                              const double* eps_o, const double* eps_v,
                              double denominator_threshold, std::size_t max_bytes, int device,
                              std::size_t max_panel_buffers = 3,
                              generativeqc::runtime::PrecisionDirective admitted_w = {},
                              bool occupied_partition = false, bool compensated_fp32_w = false);
/** Differentiate the complete occupied-tile energy on CUDA.
 * Includes both the original energy and all nine input cotangents, staged once.
 * Complete admission charges borrowed host input values, detached outputs,
 * owned CUDA/provider storage and caller_bytes for all other live numeric state.
 * No CPU mathematical fallback or complete ovvv/rank-six tensor is constructed.
 * A one-panel fallback preserves the same equations when three panels do not fit.
 * The compiler's bounded FP64 reduction tree is the default gap schedule;
 * parallel_gap_reduction=false retains the original serial order for independent
 * schedule comparisons.
 * include_gap_response=false explicitly omits the epsilon cotangents (empty
 * vectors), their kernels and their workspace. All primal denominator checks
 * remain active. A molecular consumer must instead supply full same-space Fock
 * response; adding both epsilon and full-Fock sources would double count.
 * This internal fixed-frame derivative is not a complete molecular force.
 */
DFCudaResponseResult pullback_df_cuda(
    std::size_t o, std::size_t v, std::size_t q, const double* bov, const double* bvv,
    const double* ovoo, const double* ovov, const double* fov, const double* t1, const double* t2,
    const double* eps_o, const double* eps_v, double denominator_threshold, std::size_t max_bytes,
    int device, std::size_t caller_bytes = 0, std::size_t max_panel_buffers = 3,
    bool parallel_gap_reduction = true, bool include_gap_response = true);
/** Full oo/vv derivative of the separable triples Fock inverse on CUDA.
 * Fixed j>=k pages vary i and retain cubic X/Y vectors; no same-space energy
 * differences are divided. max_page_rows=0 requests all occupied rows. If they
 * do not fit, two bounded pages recompute cross-page vectors with explicit work
 * counts. max_bytes includes host inputs/outputs and caller_bytes as above.
 * This resolves fixed-frame canonicalization only, not nuclear/orbital response.
 */
DFCudaFockResult fock_response_df_cuda(std::size_t o, std::size_t v, std::size_t q,
                                       const double* bov, const double* bvv, const double* ovoo,
                                       const double* ovov, const double* fov, const double* t1,
                                       const double* t2, const double* eps_o, const double* eps_v,
                                       double denominator_threshold, std::size_t max_bytes,
                                       int device, std::size_t caller_bytes = 0,
                                       std::size_t max_page_rows = 0,
                                       std::size_t max_panel_buffers = 3);

/** Compose the two force-only triples response phases while retaining one
 * immutable device copy of B_ov/B_vv, retained blocks, amplitudes and orbital
 * energies. max_bytes remains a complete simultaneously-live numeric bound;
 * standalone APIs remain available as independent qualification or fallback.
 * fused_scalar_response opts into one compiler-derived primal/W/V seed region
 * when epsilon outputs are unrequested. Its six V seeds add five virtual cubes
 * to the admitted arena; insufficient capacity retains the original schedule
 * before reducing panel/page residency. The default remains unfused pending
 * complete-endpoint qualification. Full rank-six T3 is never retained.
 * With compensated_fp32_w=false, admitted_w optionally admits the original FP32 W
 * policy. On its fast path, both forward DF-panel/W GEMMs and reverse W-pullback bar_panel/bar_t2 GEMMs
 * execute in pedantic IEEE FP32 SGEMM with FP64 outer accumulation; denominators,
 * energy reduction, V/ovoo/pullback_panel/Fock contractions, and all outer orbital/T2
 * accumulations remain FP64. If the fast path declines for resources and the bounded
 * WExecution path executes, only forward W uses FP32 while reverse GEMMs stay FP64.
 * DFCudaResult reports forward and reverse FP32 execution separately. External energy
 * and force qualification is required. The opt-in compensated_fp32_w policy
 * recombines nine pedantic SGEMMs per K tile in FP64 for all forward DF-panel/W
 * contractions, both DF-factor adjoints and all four W adjoints. It retains FP64
 * panels between contractions. Resource refusal is explicit before publication.
 */
DFCudaCombinedResponseResult pullback_and_fock_df_cuda(
    std::size_t o, std::size_t v, std::size_t q, const double* bov, const double* bvv,
    const double* ovoo, const double* ovov, const double* fov, const double* t1, const double* t2,
    const double* eps_o, const double* eps_v, double denominator_threshold, std::size_t max_bytes,
    int device, std::size_t caller_bytes = 0, std::size_t max_page_rows = 0,
    std::size_t max_panel_buffers = 3, bool parallel_gap_reduction = true,
    bool include_gap_response = true, bool fused_scalar_response = false,
    generativeqc::runtime::PrecisionDirective admitted_w = {},
    std::span<const double> occupied_weights = {}, bool occupied_partition = false,
    bool compensated_fp32_w = false);
#endif

}  // namespace generativeqc::cc::triples
