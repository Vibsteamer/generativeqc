#include "api/error.hpp"
#include "generativeqc/ucc_fitting.h"

#include <cmath>
#include <stdexcept>

#ifndef GENERATIVEQC_HAS_EXPLICIT_TF12
#define GENERATIVEQC_HAS_EXPLICIT_TF12 0
#endif

#if GENERATIVEQC_HAS_EXPLICIT_TF12
#include "cc/explicit_tf12_sp.hpp"
#endif

namespace {

#if !GENERATIVEQC_HAS_EXPLICIT_TF12
constexpr const char* kExplicitTf12DisabledDetail =
    "Explicit Kohn (T)-F12 engine disabled (configure with -DGENERATIVEQC_ENABLE_EXPLICIT_TF12=ON)";
#else
constexpr std::size_t min_breakdown_slots = 28;

bool valid_outputs(const double* energy, const double* breakdown, std::size_t slots) {
  return energy && (!breakdown || slots >= min_breakdown_slots);
}

void write_explicit_tf12_breakdown(const generativeqc::cc::explicit_tf12::Breakdown& b,
                                   double* out, std::size_t count) {
  if (!out) return;
  out[0] = b.e_pt_with_c2;
  out[1] = b.e_pt_conv;
  out[2] = b.t1_h3_x2;
  out[3] = b.t2_h3_x6;
  out[4] = b.t3_xh_h7;
  out[5] = b.t14_xh_h9;
  out[6] = b.t4_x2_h7;
  out[7] = b.t15_x3_h9;
  out[8] = b.t5_z2;
  out[9] = b.t6_t1_v9;
  out[10] = b.t7_t1_v7;
  out[11] = b.t8_v8;
  out[12] = b.t9_v10;
  out[13] = b.t16_bh4;
  out[14] = b.t17_fo_x7;
  out[15] = b.t18_x7_fo;
  out[16] = b.t19_fo_x4;
  out[17] = b.t20_fo_x7_ex;
  out[18] = b.t21_b4;
  out[19] = b.t22_b7;
  out[20] = b.t23_fv_x4;
  out[21] = b.t24_fv_x7;
  out[22] = b.a_sum;
  out[23] = b.total_energy;
  out[24] = b.seconds_pt;
  out[25] = b.seconds_a_vxbbh;
  out[26] = b.seconds_z2;
  out[27] = static_cast<double>(b.peak_workspace_bytes);
  if (count >= 30) {
    out[28] = static_cast<double>(b.triples_fp32_gemms);
    out[29] = static_cast<double>(b.triples_fp64_gemms);
  }
}
#endif

}  // namespace

extern "C" GENERATIVEQC_API const char* generativeqc_rcc_explicit_tf12_disabled_detail_v1(void) {
#if GENERATIVEQC_HAS_EXPLICIT_TF12
  return nullptr;
#else
  return kExplicitTf12DisabledDetail;
#endif
}

extern "C" GENERATIVEQC_API generativeqc_status generativeqc_rcc_explicit_tf12_df_cpu_v1(
    std::size_t o_full, std::size_t frozen, std::size_t obs, std::size_t ext, std::size_t q_mp2,
    std::size_t q_f12, const double* bov, const double* bvv, const double* ovoo, const double* ovov,
    const double* fov, const double* t1, const double* t2, const double* eps_o, const double* eps_v,
    const double* b_g_ext_obs, const double* b_g_ext_ext, const double* fit_ext_obs,
    const double* r_f_ext_obs, const double* r_fg_obs_obs, const double* r_ff_ext_obs,
    const double* r_u_obs_obs, const double* f_ext, const double* k_ext, std::uint32_t z2_mode,
    double geminal_scale, double* energy_out, double* breakdown_out, std::size_t breakdown_count) {
#if !GENERATIVEQC_HAS_EXPLICIT_TF12
  (void)o_full; (void)frozen; (void)obs; (void)ext; (void)q_mp2; (void)q_f12;
  (void)bov; (void)bvv; (void)ovoo; (void)ovov; (void)fov; (void)t1; (void)t2;
  (void)eps_o; (void)eps_v; (void)b_g_ext_obs; (void)b_g_ext_ext; (void)fit_ext_obs;
  (void)r_f_ext_obs; (void)r_fg_obs_obs; (void)r_ff_ext_obs; (void)r_u_obs_obs;
  (void)f_ext; (void)k_ext; (void)z2_mode; (void)geminal_scale;
  (void)energy_out; (void)breakdown_out; (void)breakdown_count;
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#else
  if (!valid_outputs(energy_out, breakdown_out, breakdown_count) || frozen >= o_full ||
      o_full >= obs || !q_mp2 || !bov || !bvv || !ovoo || !ovov || !fov || !t1 || !t2 ||
      !eps_o || !eps_v || z2_mode > 2 ||
      !std::isfinite(geminal_scale))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
    using namespace generativeqc::cc::explicit_tf12;
    const std::size_t o = o_full - frozen;
    const std::size_t v = obs - o_full;
    OwnedC2Factors owned_c2;
    if (q_f12 > 0) {
      if (ext <= obs || (z2_mode != 2 && !b_g_ext_ext) || !b_g_ext_obs || !fit_ext_obs || !r_f_ext_obs || !r_fg_obs_obs ||
          !r_ff_ext_obs || !r_u_obs_obs || !f_ext || !k_ext)
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      owned_c2 = build_c2_factors_cpu(o_full, frozen, obs, ext, q_f12, fit_ext_obs, r_f_ext_obs,
                                      f_ext, geminal_scale);
    }
    PtInputs pt{
        .o = o,
        .v = v,
        .q_mp2 = q_mp2,
        .bov = bov,
        .bvv = bvv,
        .ovoo = ovoo,
        .ovov = ovov,
        .fov = fov,
        .t1 = t1,
        .t2 = t2,
        .eps_o = eps_o,
        .eps_v = eps_v,
        .ovvv_direct = nullptr,
        .c2 = owned_c2.view(),
        .denominator_threshold = 1e-10,
    };
    DfAInputs a_in{
        .o_full = o_full,
        .frozen = frozen,
        .obs = obs,
        .ext = ext,
        .q_f12 = q_f12,
        .b_g_ext_obs = b_g_ext_obs,
        .b_g_ext_ext = b_g_ext_ext,
        .fit_ext_obs = fit_ext_obs,
        .r_f_ext_obs = r_f_ext_obs,
        .r_fg_obs_obs = r_fg_obs_obs,
        .r_ff_ext_obs = r_ff_ext_obs,
        .r_u_obs_obs = r_u_obs_obs,
        .f_ext = f_ext,
        .k_ext = k_ext,
        .ovoo = ovoo,
        .fov = fov,
        .t1 = t1,
        .t2 = t2,
        .eps_o = eps_o,
        .eps_v = eps_v,
        .z2_mode = static_cast<Z2Mode>(z2_mode),
        .geminal_scale = geminal_scale,
    };
    const auto res = evaluate_explicit_tf12_df_cpu(pt, a_in);
    if (!std::isfinite(res.total_energy)) return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    *energy_out = res.total_energy;
    write_explicit_tf12_breakdown(res, breakdown_out, breakdown_count);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::length_error&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return generativeqc::api::map_exception(nullptr);
  }
#endif
}

extern "C" GENERATIVEQC_API generativeqc_status generativeqc_rcc_explicit_tf12_exact4_cpu_v1(
    std::size_t o_full, std::size_t frozen, std::size_t obs, std::size_t ext, const double* ovvv,
    const double* ovoo, const double* ovov, const double* fov, const double* t1, const double* t2,
    const double* eps_o, const double* eps_v, const double* c2_ab,
    const double* g_obs_obs_ext_ext, const double* g_chem_ext_ext_o_v,
    const double* g_chem_ext_o_ext_v, const double* f12, const double* fg, const double* ff,
    const double* ftf, const double* ffbar, const double* f_ext, const double* k_ext,
    std::uint32_t z2_mode, double geminal_scale, double* energy_out, double* breakdown_out,
    std::size_t breakdown_count) {
#if !GENERATIVEQC_HAS_EXPLICIT_TF12
  (void)o_full; (void)frozen; (void)obs; (void)ext; (void)ovvv; (void)ovoo; (void)ovov;
  (void)fov; (void)t1; (void)t2; (void)eps_o; (void)eps_v; (void)c2_ab;
  (void)g_obs_obs_ext_ext; (void)g_chem_ext_ext_o_v; (void)g_chem_ext_o_ext_v;
  (void)f12; (void)fg; (void)ff; (void)ftf; (void)ffbar; (void)f_ext; (void)k_ext;
  (void)z2_mode; (void)geminal_scale; (void)energy_out; (void)breakdown_out; (void)breakdown_count;
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#else
  if (!valid_outputs(energy_out, breakdown_out, breakdown_count) || frozen >= o_full ||
      o_full >= obs || obs >= ext || !ovvv || !ovoo || !ovov ||
      !fov || !t1 || !t2 || !eps_o || !eps_v || !g_obs_obs_ext_ext || !f12 || !fg || !ff || !ftf ||
      !ffbar || !f_ext || !k_ext || (z2_mode != 2 && (!g_chem_ext_ext_o_v || !g_chem_ext_o_ext_v)) || z2_mode > 2 || !std::isfinite(geminal_scale))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
    using namespace generativeqc::cc::explicit_tf12;
    const std::size_t o = o_full - frozen;
    const std::size_t v = obs - o_full;
    PtInputs pt{
        .o = o,
        .v = v,
        .q_mp2 = 0,
        .bov = nullptr,
        .bvv = nullptr,
        .ovoo = ovoo,
        .ovov = ovov,
        .fov = fov,
        .t1 = t1,
        .t2 = t2,
        .eps_o = eps_o,
        .eps_v = eps_v,
        .ovvv_direct = ovvv,
        .c2 = C2Factors{.q_f12 = 0, .geminal_scale = geminal_scale, .c2_ab_direct = c2_ab},
        .denominator_threshold = 1e-10,
    };
    Exact4AInputs a_in{
        .o_full = o_full,
        .frozen = frozen,
        .obs = obs,
        .ext = ext,
        .g_obs_obs_ext_ext = g_obs_obs_ext_ext,
        .g_chem_ext_ext_o_v = g_chem_ext_ext_o_v,
        .g_chem_ext_o_ext_v = g_chem_ext_o_ext_v,
        .f12 = f12,
        .fg = fg,
        .ff = ff,
        .ftf = ftf,
        .ffbar = ffbar,
        .f_ext = f_ext,
        .k_ext = k_ext,
        .ovoo = ovoo,
        .fov = fov,
        .t1 = t1,
        .t2 = t2,
        .eps_o = eps_o,
        .eps_v = eps_v,
        .z2_mode = static_cast<Z2Mode>(z2_mode),
        .geminal_scale = geminal_scale,
    };
    Breakdown res = evaluate_explicit_tf12_a_exact4_cpu(a_in);
    double sec_pt = 0.0;
    res.e_pt_with_c2 = evaluate_explicit_tf12_pt_cpu(pt, &sec_pt);
    res.seconds_pt = sec_pt;
    PtInputs pt_conv = pt;
    pt_conv.c2 = {};
    res.e_pt_conv = evaluate_explicit_tf12_pt_cpu(pt_conv, nullptr);
    res.total_energy = res.e_pt_with_c2 + res.a_sum;
    if (!std::isfinite(res.total_energy)) return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    *energy_out = res.total_energy;
    write_explicit_tf12_breakdown(res, breakdown_out, breakdown_count);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::length_error&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return generativeqc::api::map_exception(nullptr);
  }
#endif
}

extern "C" GENERATIVEQC_API generativeqc_status generativeqc_rcc_explicit_tf12_df_cuda_v1(
    int device, std::size_t max_bytes, int triples_fp32_w, std::size_t o_full, std::size_t frozen,
    std::size_t obs, std::size_t ext, std::size_t q_mp2, std::size_t q_f12, const double* bov,
    const double* bvv, const double* ovoo, const double* ovov, const double* fov, const double* t1,
    const double* t2, const double* eps_o, const double* eps_v, const double* b_g_ext_obs,
    const double* b_g_ext_ext, const double* fit_ext_obs, const double* r_f_ext_obs,
    const double* r_fg_obs_obs, const double* r_ff_ext_obs, const double* r_u_obs_obs,
    const double* f_ext, const double* k_ext, std::uint32_t z2_mode, double geminal_scale,
    double* energy_out, double* breakdown_out, std::size_t breakdown_count) {
#if GENERATIVEQC_HAS_CUDA && GENERATIVEQC_HAS_EXPLICIT_TF12
  if (device < 0 || !max_bytes || (triples_fp32_w != 0 && triples_fp32_w != 1) ||
      !valid_outputs(energy_out, breakdown_out, breakdown_count) || frozen >= o_full ||
      o_full >= obs || !q_mp2 || !bov || !bvv || !ovoo || !ovov || !fov || !t1 || !t2 ||
      !eps_o || !eps_v ||
      z2_mode > 2 || !std::isfinite(geminal_scale))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  try {
    using namespace generativeqc;
    using namespace generativeqc::cc::explicit_tf12;
    const std::size_t o = o_full - frozen;
    const std::size_t v = obs - o_full;
    OwnedC2Factors owned_c2;
    if (q_f12 > 0) {
      if (ext <= obs || (z2_mode != 2 && !b_g_ext_ext) || !b_g_ext_obs || !fit_ext_obs || !r_f_ext_obs || !r_fg_obs_obs ||
          !r_ff_ext_obs || !r_u_obs_obs || !f_ext || !k_ext)
        return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
      owned_c2 = build_c2_factors_cpu(o_full, frozen, obs, ext, q_f12, fit_ext_obs, r_f_ext_obs,
                                      f_ext, geminal_scale);
    }
    PtInputs pt{
        .o = o,
        .v = v,
        .q_mp2 = q_mp2,
        .bov = bov,
        .bvv = bvv,
        .ovoo = ovoo,
        .ovov = ovov,
        .fov = fov,
        .t1 = t1,
        .t2 = t2,
        .eps_o = eps_o,
        .eps_v = eps_v,
        .ovvv_direct = nullptr,
        .c2 = owned_c2.view(),
        .denominator_threshold = 1e-10,
    };
    DfAInputs a_in{
        .o_full = o_full,
        .frozen = frozen,
        .obs = obs,
        .ext = ext,
        .q_f12 = q_f12,
        .b_g_ext_obs = b_g_ext_obs,
        .b_g_ext_ext = b_g_ext_ext,
        .fit_ext_obs = fit_ext_obs,
        .r_f_ext_obs = r_f_ext_obs,
        .r_fg_obs_obs = r_fg_obs_obs,
        .r_ff_ext_obs = r_ff_ext_obs,
        .r_u_obs_obs = r_u_obs_obs,
        .f_ext = f_ext,
        .k_ext = k_ext,
        .ovoo = ovoo,
        .fov = fov,
        .t1 = t1,
        .t2 = t2,
        .eps_o = eps_o,
        .eps_v = eps_v,
        .z2_mode = static_cast<Z2Mode>(z2_mode),
        .geminal_scale = geminal_scale,
    };
    const runtime::PrecisionDirective w_dir =
        triples_fp32_w ? runtime::PrecisionDirective{runtime::PrecisionDtype::Fp32,
                                                     runtime::PrecisionDtype::Fp32,
                                                     runtime::PrecisionDtype::Fp32,
                                                     "issue1764/df-triples-w-fp32-candidate-v1"}
                       : runtime::PrecisionDirective{};
    const auto res = evaluate_explicit_tf12_df_cuda(pt, a_in, max_bytes, device, 3, w_dir);
    if (!std::isfinite(res.total_energy)) return GENERATIVEQC_STATUS_NUMERICAL_FAILURE;
    *energy_out = res.total_energy;
    write_explicit_tf12_breakdown(res, breakdown_out, breakdown_count);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (const std::length_error&) {
    return GENERATIVEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    return generativeqc::api::map_exception(nullptr);
  }
#else
  (void)device; (void)max_bytes; (void)triples_fp32_w; (void)o_full; (void)frozen; (void)obs;
  (void)ext; (void)q_mp2; (void)q_f12; (void)bov; (void)bvv; (void)ovoo; (void)ovov; (void)fov;
  (void)t1; (void)t2; (void)eps_o; (void)eps_v; (void)b_g_ext_obs; (void)b_g_ext_ext;
  (void)fit_ext_obs; (void)r_f_ext_obs; (void)r_fg_obs_obs; (void)r_ff_ext_obs; (void)r_u_obs_obs;
  (void)f_ext; (void)k_ext; (void)z2_mode; (void)geminal_scale; (void)energy_out;
  (void)breakdown_out; (void)breakdown_count;
  return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
#endif
}
