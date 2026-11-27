#ifndef GENERATIVEQC_UCC_FITTING_H
#define GENERATIVEQC_UCC_FITTING_H

#include "generativeqc.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Retain the same canonical determinant that supplies the focal DF-HF arrays.
 * All matrices use the public AO order. The returned owner outlives its SCF context. */
typedef struct generativeqc_focal_reference generativeqc_focal_reference;
GENERATIVEQC_API generativeqc_status generativeqc_focal_df_reference_owned_v1(
    generativeqc_context* context, const generativeqc_system* orbital,
    const generativeqc_system* jk_fit, int spins, int forces, size_t budget,
    const double* initial_density, size_t initial_count, double* output, size_t count,
    generativeqc_focal_reference** owner);
GENERATIVEQC_API void generativeqc_focal_reference_destroy_v1(generativeqc_focal_reference* owner);

#ifdef __cplusplus
}
#endif
#endif
