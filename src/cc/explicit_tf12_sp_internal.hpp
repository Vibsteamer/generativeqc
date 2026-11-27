#pragma once

#include <cstddef>
#include <vector>

#include "cc/explicit_tf12_sp.hpp"

namespace generativeqc::cc::explicit_tf12::detail {

void evaluate_a_contractions_from_blocks(
    std::size_t o, std::size_t v, const double* fov, const double* t1, const double* t2,
    const double* eps_o, const double* eps_v, const double* ovoo, const double* V7,
    const double* V8, const double* V9, const double* V10_d, const double* V10_e, const double* X2,
    const double* X4, const double* X6, const double* X7_d, const double* X7_e, const double* B4,
    const double* B7_d, const double* B7_e, const double* Bh4, double t5_z2, Breakdown& out);

double z2_ffg_11_scalar_from_obs(std::size_t o_full, std::size_t frozen, std::size_t obs,
                                 const std::vector<double>& ff_obs,
                                 const std::vector<double>& v_sp, const std::vector<double>& t2t);

double z2_ffg_11_df_cpu(std::size_t o_full, std::size_t frozen, std::size_t obs, std::size_t ext,
                        std::size_t q, const double* fit_ext_obs, const double* r_ff_ext_obs,
                        const double* b_g_ext_obs, double s2, const std::vector<double>& t2t);

}  // namespace generativeqc::cc::explicit_tf12::detail
