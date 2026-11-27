#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "runtime/execution_precision.hpp"

namespace generativeqc::cc::explicit_tf12 {

/** Evaluation strategy for the connected Z2-INT[1] (HH/PP) intermediate. */
enum class Z2Mode : std::uint8_t {
  /** Direct CABS evaluation: R_Q^dagger g R_Q (GeCCo Z2_appr=direct). */
  Direct = 0,
  /** CABS + analytic f^2 * g completion of the Coulomb 1*1 term (GeCCo default). */
  FfgCompleted = 1,
  /** Disable Z2 contribution (ablation / zero-coupling check). */
  Zero = 2,
};

/** Exact term-by-term scalar breakdown of the 33-term Kohn extended-SP (T)-F12
 * Lagrangian L = A + l^T t + q^T (K t + g) in closed-shell spatial RHF form.
 *
 * All energies are in Hartree (Eh). Non-Brillouin Fock ov terms (t1, t2) are
 * included alongside all canonical terms so arbitrary orbital frames are exact.
 */
struct Breakdown {
  double e_pt_with_c2{};
  double e_pt_conv{};
  double t1_h3_x2{};
  double t2_h3_x6{};
  double t3_xh_h7{};
  double t14_xh_h9{};
  double t4_x2_h7{};
  double t15_x3_h9{};
  double t5_z2{};
  double t6_t1_v9{};
  double t7_t1_v7{};
  double t8_v8{};
  double t9_v10{};
  double t16_bh4{};
  double t17_fo_x7{};
  double t18_x7_fo{};
  double t19_fo_x4{};
  double t20_fo_x7_ex{};
  double t21_b4{};
  double t22_b7{};
  double t23_fv_x4{};
  double t24_fv_x7{};
  double a_sum{};
  double total_energy{};
  double seconds_pt{};
  double seconds_a_vxbbh{};
  double seconds_z2{};
  std::size_t peak_workspace_bytes{};
  std::size_t triples_fp32_gemms{};
  std::size_t triples_fp64_gemms{};
};

/** Compact 3-index DF factors for on-the-fly C-INT[2] (PP/HP) panel assembly.
 *
 * Each factor is row-major with leading auxiliary dimension q_f12:
 *   f_vo, r_vo, af_vo, ar_vo : [q_f12, o_act, v]
 *   f_vv, r_vv, af_vv, ar_vv : [q_f12, v, v]
 * where r = raw_f12 - 0.5 * M_f12 * fit, af = F_{v,X} * f_{X,.}, ar = F_{v,X} * r_{X,.}.
 * If q_f12 == 0 and c2_ab_direct == nullptr, zero C2 coupling is applied.
 */
struct C2Factors {
  std::size_t q_f12{};
  double geminal_scale{1.0};
  const double* f_vo{};
  const double* r_vo{};
  const double* f_vv{};
  const double* r_vv{};
  const double* af_vo{};
  const double* ar_vo{};
  const double* af_vv{};
  const double* ar_vv{};
  /** Optional direct 4-index C2_ab[a, b, i, d] (shape [v, v, o_act, v]) for non-DF oracles. */
  const double* c2_ab_direct{};
};

/** Physical triples + C-INT[2] input bundle (spatial active canonical RHF). */
struct PtInputs {
  std::size_t o{};
  std::size_t v{};
  std::size_t q_mp2{};
  const double* bov{};   // [q_mp2, o, v]
  const double* bvv{};   // [q_mp2, v, v] (symmetric in v x v)
  const double* ovoo{};  // [o, v, o, o]
  const double* ovov{};  // [o, v, o, v]
  const double* fov{};   // [o, v]
  const double* t1{};    // [o, v]
  const double* t2{};    // [o, o, v, v]
  const double* eps_o{}; // [o]
  const double* eps_v{}; // [v]
  /** Optional direct 4-index chemists' ovvv[i, a, b, f] = (i a | b f) for non-DF oracles. */
  const double* ovvv_direct{};
  C2Factors c2{};
  double denominator_threshold{1e-10};
};

/** Connected A-sector inputs in 3-index DF representation (F12Fit + MP2Fit).
 *
 * All 3-index arrays are row-major with auxiliary index Q outermost:
 *   b_g_ext_obs : [q_f12, ext, obs]  whitened Coulomb B^Q_{p,r} (p in ext, r in obs)
 *   b_g_ext_ext : [q_f12, ext, ext]  whitened Coulomb B^Q_{p,q} (only needed if z2_mode != Zero)
 *   fit_ext_obs : [q_f12, ext, obs]  Coulomb fit coefficients F^Q_{p,r} = (J^-1 G)^Q_{p,r}
 *   r_f_ext_obs : [q_f12, ext, obs]  half-robust geminal R^Q - 0.5 * M_f * F^Q
 *   r_fg_obs_obs: [q_f12, obs, obs]  half-robust geminal*Coulomb
 *   r_ff_ext_obs: [q_f12, ext, obs]  half-robust squared geminal (ext x obs for B and Z2 ffg)
 *   r_u_obs_obs : [q_f12, obs, obs]  half-robust double commutator U = [f,[T,f]]
 *   f_ext       : [ext, ext]         extended Fock matrix (OBS + CABS)
 *   k_ext       : [ext, ext]         positive extended exchange matrix (OBS + CABS)
 *   ovoo        : [o_act, v, o_act, o_act] MP2Fit chemists' (i a | j k) -> H7[k,l,j,c] = ovoo[l,c,k,j]
 *   fov         : [o_act, v]         active Fock ov block
 *   t1          : [o_act, v]
 *   t2          : [o_act, o_act, v, v]
 *   eps_o       : [o_act]
 *   eps_v       : [v]
 */
struct DfAInputs {
  std::size_t o_full{};
  std::size_t frozen{};
  std::size_t obs{};
  std::size_t ext{};
  std::size_t q_f12{};
  const double* b_g_ext_obs{};
  const double* b_g_ext_ext{};
  const double* fit_ext_obs{};
  const double* r_f_ext_obs{};
  const double* r_fg_obs_obs{};
  const double* r_ff_ext_obs{};
  const double* r_u_obs_obs{};
  const double* f_ext{};
  const double* k_ext{};
  const double* ovoo{};
  const double* fov{};
  const double* t1{};
  const double* t2{};
  const double* eps_o{};
  const double* eps_v{};
  Z2Mode z2_mode{Z2Mode::FfgCompleted};
  double geminal_scale{1.0};
};

/** Exact 4-index spatial inputs for non-DF contract parity (GeCCo BH oracle).
 *
 * Physicist's tensors <p q | op | r s> in row-major order:
 *   g_obs_obs_ext_ext : [obs, obs, ext, ext]  <p q | g | a b> (p,q in obs; a,b in ext)
 *   g_chem_ext_ext_o_v: [ext, ext, o_act, v]  chemists' (a b | k c) for Z2 Coulomb
 *   g_chem_ext_o_ext_v: [ext, o_act, ext, v]  chemists' (b k | a c) for Z2 Exchange
 *   f12               : [ext, ext, obs, obs]  <a b | f | r s>
 *   fg                : [obs, obs, obs, obs]  <p q | f g | r s>
 *   ff                : [obs, obs, obs, obs]  <p q | f^2 | r s>
 *   ftf               : [obs, obs, obs, obs]  <p q | [f,[T,f]] | r s>
 *   ffbar             : [obs, obs, obs, obs]  <p q | (F+K) f^2 | r s>
 *   f_ext, k_ext      : [ext, ext]
 */
struct Exact4AInputs {
  std::size_t o_full{};
  std::size_t frozen{};
  std::size_t obs{};
  std::size_t ext{};
  const double* g_obs_obs_ext_ext{};
  const double* g_chem_ext_ext_o_v{};
  const double* g_chem_ext_o_ext_v{};
  const double* f12{};
  const double* fg{};
  const double* ff{};
  const double* ftf{};
  const double* ffbar{};
  const double* f_ext{};
  const double* k_ext{};
  const double* ovoo{};
  const double* fov{};
  const double* t1{};
  const double* t2{};
  const double* eps_o{};
  const double* eps_v{};
  Z2Mode z2_mode{Z2Mode::FfgCompleted};
  double geminal_scale{1.0};
};

/** Build compact 3-index DF C2 factors on host from (fit_ext_obs, r_f_ext_obs, f_ext). */
struct OwnedC2Factors {
  std::size_t q_f12{};
  double geminal_scale{1.0};
  std::vector<double> f_vo, r_vo, f_vv, r_vv, af_vo, ar_vo, af_vv, ar_vv;

  [[nodiscard]] C2Factors view() const noexcept {
    return C2Factors{
        .q_f12 = q_f12,
        .geminal_scale = geminal_scale,
        .f_vo = f_vo.empty() ? nullptr : f_vo.data(),
        .r_vo = r_vo.empty() ? nullptr : r_vo.data(),
        .f_vv = f_vv.empty() ? nullptr : f_vv.data(),
        .r_vv = r_vv.empty() ? nullptr : r_vv.data(),
        .af_vo = af_vo.empty() ? nullptr : af_vo.data(),
        .ar_vo = ar_vo.empty() ? nullptr : ar_vo.data(),
        .af_vv = af_vv.empty() ? nullptr : af_vv.data(),
        .ar_vv = ar_vv.empty() ? nullptr : ar_vv.data(),
        .c2_ab_direct = nullptr,
    };
  }
};

OwnedC2Factors build_c2_factors_cpu(std::size_t o_full, std::size_t frozen, std::size_t obs,
                                    std::size_t ext, std::size_t q_f12, const double* fit_ext_obs,
                                    const double* r_f_ext_obs, const double* f_ext,
                                    double geminal_scale = 1.0);

/** CPU evaluation of physical triples + C-INT[2] coupling (exact DF-RCCSD(T) when C2 is empty). */
double evaluate_explicit_tf12_pt_cpu(const PtInputs& in, double* seconds_out = nullptr);

/** CPU evaluation of connected A-sector (V, X, Xh, B, Bh, Z2) from 3-index DF F12 tensors. */
Breakdown evaluate_explicit_tf12_a_cpu(const DfAInputs& in);

/** CPU evaluation of connected A-sector from exact 4-index integrals (for non-DF BH parity). */
Breakdown evaluate_explicit_tf12_a_exact4_cpu(const Exact4AInputs& in);

/** Combined CPU evaluation of full Kohn extended-SP (T)-F12 energy + breakdown. */
Breakdown evaluate_explicit_tf12_df_cpu(const PtInputs& pt_in, const DfAInputs& a_in);

#if GENERATIVEQC_HAS_CUDA
/** GPU-resident DF-CUDA physical triples + on-the-fly 3-index DF C-INT[2] panel coupling.
 * Reuses the verified generated_df occupied-tile W/V/energy pipeline and optional FP32 W. */
double evaluate_explicit_tf12_pt_cuda(const PtInputs& in, std::size_t max_bytes, int device,
                                      std::size_t max_panel_buffers = 3,
                                      generativeqc::runtime::PrecisionDirective admitted_w = {},
                                      Breakdown* telemetry_out = nullptr);

/** GPU-resident cuBLAS DF-CUDA evaluation of connected A-sector (V, X, Xh, B, Bh, Z2). */
Breakdown evaluate_explicit_tf12_a_cuda(const DfAInputs& in, std::size_t max_bytes, int device);

/** Combined GPU-resident DF-CUDA evaluation of full Kohn extended-SP (T)-F12 energy + breakdown. */
Breakdown evaluate_explicit_tf12_df_cuda(const PtInputs& pt_in, const DfAInputs& a_in,
                                         std::size_t max_bytes, int device,
                                         std::size_t max_panel_buffers = 3,
                                         generativeqc::runtime::PrecisionDirective admitted_w = {});
#endif

}  // namespace generativeqc::cc::explicit_tf12
