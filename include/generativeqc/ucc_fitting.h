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
