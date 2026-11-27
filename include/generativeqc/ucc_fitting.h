#ifndef GENERATIVEQC_UCC_FITTING_H
#define GENERATIVEQC_UCC_FITTING_H

#include "generativeqc.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Prepare full-DF UCCSD or UCCSD(T) with independent fitted operators.
 * Supply the correlation fit in descriptor->density_fitting_auxiliary_basis.
 * The prepared calculation owns both bases. JK and its response use jk_fit.
 * The ordinary execute entry supplies energy and optional analytic forces.
 */
GENERATIVEQC_API generativeqc_status generativeqc_calculation_prepare_ucc_df_fits_v1(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_method_descriptor* descriptor, const generativeqc_system* jk_fit,
    generativeqc_calculation** calculation);

/** Retain the same canonical determinant that supplies the focal DF-HF arrays.
 * All matrices use the public AO order. The returned owner outlives its SCF context. */
typedef struct generativeqc_focal_reference generativeqc_focal_reference;
GENERATIVEQC_API generativeqc_status generativeqc_focal_df_reference_owned_v1(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* jk_fit, int spins, int forces, size_t budget,
    const double* initial_density, size_t initial_count, double* output, size_t count,
    generativeqc_focal_reference** owner);
GENERATIVEQC_API void generativeqc_focal_reference_destroy_v1(generativeqc_focal_reference* owner);

/** Use a fresh full-DF UCCSD(T) plan with the retained determinant.
 * Output is the (T) energy followed by optional analytic Cartesian forces.
 * The two force Lagrangians use the same converged conventional amplitudes. */
GENERATIVEQC_API generativeqc_status generativeqc_focal_ucc_triples_v1(
    generativeqc_calculation* calculation, const generativeqc_focal_reference* owner, int forces,
    double* output, size_t count);

/** Return the spatial DF-RCCSD(T) increment on the retained canonical RHF root.
 * Requires CUDA, an all-electron full-DF plan and one common JK/correlation fit.
 * This entry computes only the scalar conventional (T) energy. */
GENERATIVEQC_API generativeqc_status generativeqc_focal_rcc_triples_energy_v1(
    generativeqc_calculation* calculation, const generativeqc_focal_reference* owner,
    double* output);

/** Return the same value-only (T) energy followed by one contribution for
 * each spatial occupied orbital, in the retained RHF canonical order.
 * Count must be exactly one plus the number of occupied spatial orbitals.
 * Requires CUDA, an all-electron full-DF plan and one common JK/correlation fit. */
GENERATIVEQC_API generativeqc_status generativeqc_focal_rcc_triples_occupied_v1(
    generativeqc_calculation* calculation, const generativeqc_focal_reference* owner,
    double* output, size_t count);

/** Extended spatial DF-RCCSD(T) / (T+) focal entry on the retained canonical RHF root.
 * Supports split JK/correlation fits, scalar ECP, optional frozen_occupied orbitals,
 * optional per-active-occupied weights for exact (T+) analytic force sources,
 * preconvergence (0=Legacy/FP64, 1=FP64, 2=Mixed, 3=Automatic), and optional FP32 W
 * mixed precision (when triples_fp32_w == 1, forward DF-panel/W GEMMs and, on the fast
 * forces path, reverse W-pullback bar_panel/bar_t2 GEMMs execute in pedantic IEEE FP32
 * SGEMM while denominators, energy reduction, V/ovoo/Fock contractions, and outer
 * orbital/T2 accumulations remain FP64). Mode 0 is strict FP64. Mode 2 explicitly requests
 * three FP32 limbs per operand, nine FP32 GEMMs per inner tile, and FP64 panel storage and
 * recombination for the compensated spatial forward and reverse W contractions. Mode 2
 * refuses unsupported or insufficient resources instead of switching to another W schedule.
 * When forces != 0, writes pure triples-increment (corr_gradient, bar_f, bar_c) for focal CPHF.
 * Optional telemetry_out (count >= 8) receives [ccsd_fp32, lambda_fp32, t_fp32_gemms,
 * t_fp64_gemms, ccsd_s, triples_s, lambda_s, source_response_s], when count >= 10
 * appends [triples_fp32_w_executed, triples_fp32_w_resource_fallback], and when
 * count >= 12 appends [triples_fp32_w_forward_executed, triples_fp32_w_reverse_executed].
 * With count >= 13 slot 12 reports whether any reverse W pullback executed, including FP64.
 * A provider/resource fallback can retain FP32 W and execute the DF panels in FP64.
 * Slots 8, 10 and 11 report the work performed independently of the fallback flag.
 * Slots 13–16, when present, report observed compensated execution, physical forward and
 * reverse FP32 GEMM counts, and the workspace size in bytes, respectively. */
GENERATIVEQC_API generativeqc_status generativeqc_focal_rcc_triples_v2(
    generativeqc_calculation* calculation, const generativeqc_focal_reference* owner, int forces,
    int occupied_partition, size_t frozen_occupied, const double* occupied_weights,
    size_t weights_count, uint32_t preconvergence, int triples_fp32_w, double* energy_out,
    double* occupied_out, size_t occupied_count, double* corr_gradient_out,
    size_t corr_gradient_count, double* bar_f_out, size_t bar_f_count, double* bar_c_out,
    size_t bar_c_count, double* telemetry_out, size_t telemetry_count);

/** Returns 1 when the build supports native CUDA fixed-cusp preparation and RHS stage
 * counters, and 0 for a host-only build. This probe does not access a CUDA device. */
GENERATIVEQC_API int generativeqc_focal_rcc_f12b_cuda_stage_counters_v1(void);

/** Spatial DF-CUDA RHF CCSD-F12b primal + optional Lambda & force sources on retained RHF root.
 * Consumes active-orbital spatial opposite-spin (abab) fixed-cusp tensors:
 *   w_vo  [o_act, o_act, n_vir, o_act]
 *   w_vv  [o_act, o_act, n_vir, n_vir] (pair-symmetric)
 *   c_f12 [o_act, o_act, n_vir, n_vir] (pair-symmetric)
 *   fixed_geminal_energy (V + B - X scalar in Eh)
 * Supports split JK/MP2/F12 fits, scalar ECP, frozen_occupied, and preconvergence
 * (0=Legacy, 1=FP64, 2=Mixed, 3=Automatic).
 * Optional telemetry_out (count >= 8) receives [ccsd_fp32, lambda_fp32, ccsd_iters,
 * lambda_iters, ccsd_s, lambda_s, source_response_s, total_s]. When supplied, slots 8 and 9
 * receive completed primal CUDA fixed-cusp preparation and Lambda energy RHS evaluation counts,
 * respectively. An energy-only calculation has zero Lambda RHS evaluations. */
GENERATIVEQC_API generativeqc_status generativeqc_focal_rcc_f12b_v1(
    generativeqc_calculation* calculation, const generativeqc_focal_reference* owner, int forces,
    size_t frozen_occupied, const double* w_vo, size_t w_vo_count, const double* w_vv,
    size_t w_vv_count, const double* c_f12, size_t c_f12_count, double fixed_geminal_energy,
    uint32_t preconvergence, double* energy_out, double* t1_out, size_t t1_count, double* t2_out,
    size_t t2_count, double* lambda1_out, size_t lambda1_count, double* lambda2_out,
    size_t lambda2_count, double* corr_gradient_out, size_t corr_gradient_count, double* bar_f_out,
    size_t bar_f_count, double* bar_c_out, size_t bar_c_count, double* telemetry_out,
    size_t telemetry_count);

/** Solve spatial RHF CCSD-F12b primal (and optional Lambda + Fock cotangents) on CPU
 * from supplied canonical spatial Fock, chemists' ERI blocks, and fixed-cusp F12 tensors. */
GENERATIVEQC_API generativeqc_status generativeqc_rcc_f12b_cpu_solve_v1(
    size_t nocc, size_t nvir, const double* foo, const double* fov, const double* fvv,
    const double* ovov, const double* ovvo, const double* oovv, const double* ovvv,
    const double* ovoo, const double* oooo, const double* vvvv, const double* eps_o,
    const double* eps_v, const double* w_vo, const double* w_vv, const double* c_f12,
    double fixed_geminal_energy, int solve_lambda, size_t max_bytes, double* energy_out,
    double* t1_out, double* t2_out, double* lambda1_out, double* lambda2_out, double* bar_foo_out,
    double* bar_fov_out, double* bar_fvv_out);

/** Native spin-blocked DF-UCCSD(T) / (T+) v2 focal entry on a retained canonical RHF or UHF root.
 * Runs on CPU (OpenBLAS) or CUDA without materializing 4-index ovvv/vvvv tensors.
 * Supports split JK/correlation fits, scalar ECP, independent frozen_alpha/frozen_beta,
 * optional per-spin occupied weights for exact (T+) analytic force sources,
 * preconvergence (0=Legacy/FP64, 1=FP64, 2=Mixed, 3=Automatic), and optional FP32 forward-W.
 * Optional ccsd_energy_out receives the DF-UCCSD correlation energy; energy_out receives (T)/(T+).
 * When forces != 0, writes pure triples-increment (corr_gradient, bar_f_alpha/beta,
 * bar_c_alpha/beta) for focal CPHF relaxation.
 * Optional telemetry_out (count >= 10) receives [ccsd_fp32, lambda_fp32, t_fp32_gemms,
 * t_fp64_gemms, ccsd_s, triples_s, lambda_s, source_response_s, ccsd_iterations,
 * lambda_iterations]. Slots 10 and 11, when present, report physical reverse FP32 GEMM
 * count and observed split3 execution. Mode 2 requests the compensated host spin-blocked
 * W schedule. The spin-blocked CUDA endpoint supports modes 0 and 1. */
GENERATIVEQC_API generativeqc_status generativeqc_focal_ucc_triples_v2(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* correlation_fit, const generativeqc_focal_reference* owner,
    size_t budget_bytes, int forces, int occupied_partition, size_t frozen_alpha,
    size_t frozen_beta, const double* occupied_weights_alpha, size_t weights_alpha_count,
    const double* occupied_weights_beta, size_t weights_beta_count, uint32_t preconvergence,
    int triples_fp32_w, double* ccsd_energy_out, double* energy_out, double* occupied_out_alpha,
    size_t occupied_alpha_count, double* occupied_out_beta, size_t occupied_beta_count,
    double* corr_gradient_out, size_t corr_gradient_count, double* bar_f_alpha_out,
    size_t bar_f_alpha_count, double* bar_f_beta_out, size_t bar_f_beta_count,
    double* bar_c_alpha_out, size_t bar_c_alpha_count, double* bar_c_beta_out,
    size_t bar_c_beta_count, double* telemetry_out, size_t telemetry_count);

/** Spin-blocked fixed-cusp F12b coupling tensors in the active canonical MO frame.
 * Expected element counts for active occupied (oa, ob) and virtual (va, vb):
 *   w_vo_aaaa : oa * oa * va * oa     w_vo_bbbb : ob * ob * vb * ob
 *   w_vo_abab : oa * ob * va * ob     w_vo_abba : oa * ob * vb * oa
 *   w_vv_aaaa : oa * oa * va * va     w_vv_abab : oa * ob * va * vb
 *   w_vv_bbbb : ob * ob * vb * vb
 *   c_f12_aaaa: oa * oa * va * va     c_f12_abab: oa * ob * va * vb
 *   c_f12_bbbb: ob * ob * vb * vb */
typedef struct generativeqc_focal_ucc_f12_coupling_v1 {
  double fixed_geminal_energy;
  const double* w_vo_aaaa;
  size_t w_vo_aaaa_count;
  const double* w_vo_bbbb;
  size_t w_vo_bbbb_count;
  const double* w_vo_abab;
  size_t w_vo_abab_count;
  const double* w_vo_abba;
  size_t w_vo_abba_count;
  const double* w_vv_aaaa;
  size_t w_vv_aaaa_count;
  const double* w_vv_abab;
  size_t w_vv_abab_count;
  const double* w_vv_bbbb;
  size_t w_vv_bbbb_count;
  const double* c_f12_aaaa;
  size_t c_f12_aaaa_count;
  const double* c_f12_abab;
  size_t c_f12_abab_count;
  const double* c_f12_bbbb;
  size_t c_f12_bbbb_count;
} generativeqc_focal_ucc_f12_coupling_v1;

/** Stationary cotangents of generativeqc_focal_ucc_f12_coupling_v1 written when forces != 0. */
typedef struct generativeqc_focal_ucc_f12_cotangents_v1 {
  double* bar_fixed_geminal_energy;
  double* bar_w_vo_aaaa;
  size_t bar_w_vo_aaaa_count;
  double* bar_w_vo_bbbb;
  size_t bar_w_vo_bbbb_count;
  double* bar_w_vo_abab;
  size_t bar_w_vo_abab_count;
  double* bar_w_vo_abba;
  size_t bar_w_vo_abba_count;
  double* bar_w_vv_aaaa;
  size_t bar_w_vv_aaaa_count;
  double* bar_w_vv_abab;
  size_t bar_w_vv_abab_count;
  double* bar_w_vv_bbbb;
  size_t bar_w_vv_bbbb_count;
  double* bar_c_f12_aaaa;
  size_t bar_c_f12_aaaa_count;
  double* bar_c_f12_abab;
  size_t bar_c_f12_abab_count;
  double* bar_c_f12_bbbb;
  size_t bar_c_f12_bbbb_count;
} generativeqc_focal_ucc_f12_cotangents_v1;

/** Native spin-blocked DF-UCCSD-F12b v1 focal entry on a retained canonical UHF or RHF root.
 * Runs on CPU (OpenBLAS) or CUDA without materializing 4-index ovvv/vvvv tensors.
 * Injects the caller's fixed-cusp F12b coupling bundle into the 5-block DF-UCCSD primal
 * and DF-ULambda solvers, returning the total UCCSD-F12b correlation energy and (when
 * forces != 0) both FixedF12Cotangents and ordinary DF/MO force sources (corr_gradient,
 * bar_f_alpha/beta, bar_c_alpha/beta) for focal CPHF relaxation.
 * Optional telemetry_out (count >= 10) receives [ccsd_fp32, lambda_fp32, 0, 0,
 * ccsd_s, 0, lambda_s, source_response_s, ccsd_iterations, lambda_iterations]. */
GENERATIVEQC_API generativeqc_status generativeqc_focal_ucc_f12b_v1(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* correlation_fit, const generativeqc_focal_reference* owner,
    size_t budget_bytes, int forces, size_t frozen_alpha, size_t frozen_beta,
    uint32_t preconvergence, const generativeqc_focal_ucc_f12_coupling_v1* coupling,
    double* ccsd_f12b_energy_out, generativeqc_focal_ucc_f12_cotangents_v1* cotangents_out,
    double* corr_gradient_out, size_t corr_gradient_count, double* bar_f_alpha_out,
    size_t bar_f_alpha_count, double* bar_f_beta_out, size_t bar_f_beta_count,
    double* bar_c_alpha_out, size_t bar_c_alpha_count, double* bar_c_beta_out,
    size_t bar_c_beta_count, double* telemetry_out, size_t telemetry_count);

/** Reset the calling thread's last RCCSD Lambda diagnostic slot. */
GENERATIVEQC_API void generativeqc_cc_reset_last_lambda_telemetry_v1(void);

/** Read the calling thread's last RCCSD Lambda diagnostic (returns 1 if valid, 0 otherwise). */
GENERATIVEQC_API int generativeqc_cc_last_lambda_telemetry_v1(
    double* scalars, size_t scalar_count, uint64_t* counts, size_t count_len, char* reason,
    size_t reason_len);

/** Reset the context-scoped RCCSD Lambda diagnostic slot for \p context. */
GENERATIVEQC_API void generativeqc_cc_reset_context_lambda_telemetry_v1(
    generativeqc_context* context);

/** Copy the RCCSD Lambda diagnostic from this context's completed focal force.
 * New and reset contexts have no diagnostic. Other contexts leave this slot intact.
 * Returns 1 for a valid snapshot or 0 if unavailable. */
GENERATIVEQC_API int generativeqc_cc_context_lambda_telemetry_v1(
    const generativeqc_context* context, double* scalars, size_t scalar_count, uint64_t* counts,
    size_t count_len, char* reason, size_t reason_len);

#ifdef __cplusplus
}
#endif
#endif
