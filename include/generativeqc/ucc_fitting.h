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

/** Clone a focal determinant with its occupied columns permuted in public AO order.
 * Each integer maps a new occupied column to its old index. Virtual columns,
 * the AO density and the physical Hamiltonian retain their original identity.
 */
GENERATIVEQC_API generativeqc_status generativeqc_focal_reference_permute_occupied_v1(
    const generativeqc_focal_reference* owner, const size_t* alpha, size_t alpha_count,
    const size_t* beta, size_t beta_count, size_t budget, generativeqc_focal_reference** result);

/** Return the lowest occupied spherical HF orbitals of one neutral atom as
 * [public AO, orbitals] and their energies plus the next occupied energy.
 */
GENERATIVEQC_API generativeqc_status generativeqc_focal_atomic_core_v1(
    generativeqc_context* context, const generativeqc_system* atomic, size_t orbitals,
    size_t budget, double* coefficients, size_t coefficient_count, double* energies,
    size_t energy_count);

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

/** Configure scalar-relativistic 1-electron X2C model on a normalized system handle.
 * mode: 0=None, 1=Dlu1e (atomic-block X on AE atoms only, molecular R, ECP untouched),
 *       2=Global1e (full molecular X2C-1e, all-electron systems only),
 *       3=DluLocalR (strict local atomic R_A comparator).
 * speed_of_light is in atomic units: 0 (either sign) selects CODATA 137.03599967994 a.u.,
 * other values must be finite and positive. Negative or non-finite values return
 * INVALID_ARGUMENT and leave the system unchanged. */
GENERATIVEQC_API generativeqc_status generativeqc_system_set_x2c1e_v1(
    generativeqc_context* context, generativeqc_system* system, uint32_t mode,
    double speed_of_light);

/** Evaluate native public-AO S, T, V (ionic charges) and W = <nabla|V_nuc|nabla>.
 * Any output pointer may be NULL to skip copying that matrix. */
GENERATIVEQC_API generativeqc_status generativeqc_system_stvw_integrals_v1(
    generativeqc_context* context, const generativeqc_system* system, double* overlap_out,
    double* kinetic_out, double* attraction_out, double* small_potential_out, size_t matrix_count);

/** Evaluate scalar-relativistic X2C-1e core Hamiltonian (without V_ECP) or its delta
 * (h_X2C - T - V). speed_of_light follows generativeqc_system_set_x2c1e_v1; a rejected value
 * writes no output. */
GENERATIVEQC_API generativeqc_status generativeqc_system_x2c1e_hcore_v1(
    generativeqc_context* context, const generativeqc_system* system, uint32_t mode,
    double speed_of_light, int delta_only, double* hcore_out, size_t matrix_count);

/** Evaluate coordinate-major analytic Jacobian d(h_X2C_no_ecp)/dR or d(Delta h_SR)/dR
 * with shape (3 * natom, nbf, nbf). speed_of_light follows generativeqc_system_set_x2c1e_v1;
 * a rejected value writes no output. */
GENERATIVEQC_API generativeqc_status generativeqc_system_x2c1e_hcore_derivatives_v1(
    generativeqc_context* context, const generativeqc_system* system, uint32_t mode,
    double speed_of_light, int delta_only, double* dhcore_out, size_t count);

/** Evaluate O(1)-Sylvester/adjoint analytic nuclear gradient +d/dR Tr(W_h * h_X2C)
 * or +d/dR Tr(W_h * Delta h_SR) without materializing (3*natom, n, n) matrices.
 * speed_of_light follows generativeqc_system_set_x2c1e_v1; a rejected value writes no output. */
GENERATIVEQC_API generativeqc_status generativeqc_system_x2c1e_weighted_gradient_v1(
    generativeqc_context* context, const generativeqc_system* system, uint32_t mode,
    double speed_of_light, int delta_only, const double* hcore_weights, size_t matrix_count,
    double* gradient_out, size_t gradient_count);

/** Contract public-AO overlap and hcore weights against native CPU 1e + ECP + X2C-1e derivatives. */
GENERATIVEQC_API generativeqc_status generativeqc_system_one_electron_gradient_cpu_v1(
    generativeqc_context* context, const generativeqc_system* system, const double* overlap_weights,
    const double* hcore_weights, size_t matrix_count, int include_nuclear_repulsion,
    double* gradient_out, size_t gradient_count);

/** Evaluate native CPU two-center metric (naux x naux) and three-center (nbf x nbf x naux)
 * density-fitting integrals in public AO order. */
GENERATIVEQC_API generativeqc_status generativeqc_system_df_integrals_cpu_v1(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* auxiliary, double* metric_out, size_t metric_count,
    double* three_center_out, size_t three_center_count);

/** Contract public-AO metric (naux x naux) and three-center (nbf x nbf x naux) weights against
 * native CPU DF nuclear derivatives. */
GENERATIVEQC_API generativeqc_status generativeqc_system_df_gradient_cpu_v1(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* auxiliary, const double* metric_weights, size_t metric_count,
    const double* three_center_weights, size_t three_center_count, double* gradient_out,
    size_t gradient_count);

/** Evaluate native CPU public-AO overlap, hcore (including checked ECP and X2C-1e if enabled),
 * and nuclear repulsion energy. */
GENERATIVEQC_API generativeqc_status generativeqc_system_one_electron_integrals_cpu_v1(
    generativeqc_context* context, const generativeqc_system* system, double* overlap_out,
    double* hcore_out, size_t matrix_count, double* nuclear_repulsion_out);

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

/** Returns NULL when the experimental Kohn extended-SP (T)-F12 engine is compiled
 * into this build (-DGENERATIVEQC_ENABLE_EXPLICIT_TF12=ON), or a static detail string
 * explaining why the entry points return GENERATIVEQC_STATUS_NOT_IMPLEMENTED. */
GENERATIVEQC_API const char* generativeqc_rcc_explicit_tf12_disabled_detail_v1(void);

/** Evaluate closed-shell spatial DF Kohn extended-SP (T)-F12 energy on CPU.
 * Set q_f12=0 for exact zero-coupling degeneration to DF-RCCSD(T).
 * z2_mode: 0=Direct (R_Q g R_Q), 1=FfgCompleted (GeCCo default), 2=Zero.
 * Optional breakdown_out (count >= 28) returns the term-by-term scalar breakdown.
 * With count >= 30, slots 28 and 29 hold executed CUDA FP32 and FP64 W GEMM counts. */
GENERATIVEQC_API generativeqc_status generativeqc_rcc_explicit_tf12_df_cpu_v1(
    size_t o_full, size_t frozen, size_t obs, size_t ext, size_t q_mp2, size_t q_f12,
    const double* bov, const double* bvv, const double* ovoo, const double* ovov, const double* fov,
    const double* t1, const double* t2, const double* eps_o, const double* eps_v,
    const double* b_g_ext_obs, const double* b_g_ext_ext, const double* fit_ext_obs,
    const double* r_f_ext_obs, const double* r_fg_obs_obs, const double* r_ff_ext_obs,
    const double* r_u_obs_obs, const double* f_ext, const double* k_ext, uint32_t z2_mode,
    double geminal_scale, double* energy_out, double* breakdown_out, size_t breakdown_count);

/** Evaluate closed-shell spatial exact-4-index Kohn extended-SP (T)-F12 energy on CPU
 * for direct non-DF contract parity (e.g. GeCCo archived BH integrals). */
GENERATIVEQC_API generativeqc_status generativeqc_rcc_explicit_tf12_exact4_cpu_v1(
    size_t o_full, size_t frozen, size_t obs, size_t ext, const double* ovvv, const double* ovoo,
    const double* ovov, const double* fov, const double* t1, const double* t2, const double* eps_o,
    const double* eps_v, const double* c2_ab, const double* g_obs_obs_ext_ext,
    const double* g_chem_ext_ext_o_v, const double* g_chem_ext_o_ext_v, const double* f12,
    const double* fg, const double* ff, const double* ftf, const double* ffbar, const double* f_ext,
    const double* k_ext, uint32_t z2_mode, double geminal_scale, double* energy_out,
    double* breakdown_out, size_t breakdown_count);

/** Evaluate closed-shell spatial DF-CUDA Kohn extended-SP (T)-F12 energy on GPU.
 * Uses on-the-fly 3-index DF C-INT[2] panel augmentation, a caller-provided arena cap,
 * optional triples_fp32_w mixed precision, and GPU cuBLAS V/X/B/Bh/Z2 contractions.
 * With >10 active occupied orbitals and nonzero F12 coupling, the unevaluated
 * conventional diagnostic in breakdown slot 1 is NaN. */
GENERATIVEQC_API generativeqc_status generativeqc_rcc_explicit_tf12_df_cuda_v1(
    int device, size_t max_bytes, int triples_fp32_w, size_t o_full, size_t frozen, size_t obs,
    size_t ext, size_t q_mp2, size_t q_f12, const double* bov, const double* bvv,
    const double* ovoo, const double* ovov, const double* fov, const double* t1, const double* t2,
    const double* eps_o, const double* eps_v, const double* b_g_ext_obs, const double* b_g_ext_ext,
    const double* fit_ext_obs, const double* r_f_ext_obs, const double* r_fg_obs_obs,
    const double* r_ff_ext_obs, const double* r_u_obs_obs, const double* f_ext, const double* k_ext,
    uint32_t z2_mode, double geminal_scale, double* energy_out, double* breakdown_out,
    size_t breakdown_count);

#ifdef __cplusplus
}
#endif
#endif
