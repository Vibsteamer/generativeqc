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

#ifdef __cplusplus
}
#endif
#endif
