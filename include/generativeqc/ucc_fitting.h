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
 * preconvergence (0=Legacy/FP64, 1=FP64, 2=Mixed, 3=Automatic), and optional FP32 forward-W.
 * When forces != 0, writes pure triples-increment (corr_gradient, bar_f, bar_c) for focal CPHF.
 * Optional telemetry_out (count >= 8) receives [ccsd_fp32, lambda_fp32, t_fp32_gemms,
 * t_fp64_gemms, ccsd_s, triples_s, lambda_s, source_response_s]. */
GENERATIVEQC_API generativeqc_status generativeqc_focal_rcc_triples_v2(
    generativeqc_calculation* calculation, const generativeqc_focal_reference* owner, int forces,
    int occupied_partition, size_t frozen_occupied, const double* occupied_weights,
    size_t weights_count, uint32_t preconvergence, int triples_fp32_w, double* energy_out,
    double* occupied_out, size_t occupied_count, double* corr_gradient_out,
    size_t corr_gradient_count, double* bar_f_out, size_t bar_f_count, double* bar_c_out,
    size_t bar_c_count, double* telemetry_out, size_t telemetry_count);

#ifdef __cplusplus
}
#endif
#endif
