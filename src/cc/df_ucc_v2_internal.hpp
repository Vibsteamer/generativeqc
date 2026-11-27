#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "cc/df_ucc_v2.hpp"

namespace generativeqc::cc::v2_detail {

struct SpinBlockedDims {
  std::size_t n{};
  std::size_t q{};
  std::size_t oa_full{};
  std::size_t ob_full{};
  std::size_t fa{};
  std::size_t fb{};
  std::size_t oa{};
  std::size_t ob{};
  std::size_t va{};
  std::size_t vb{};
  std::size_t natoms{};

  std::size_t n_act_a() const noexcept { return oa + va; }
  std::size_t n_act_b() const noexcept { return ob + vb; }
};

struct SpinBlockedDFSource {
  SpinBlockedDims dims;
  // Q-major 3-index whitened DF factors:
  // boo: [q, o, o], bov: [q, o, v], bvv: [q, v, v]
  std::vector<double> boo_a, bov_a, bvv_a;
  std::vector<double> boo_b, bov_b, bvv_b;
  // Active MO Fock blocks (general dense for exact Fock cotangents)
  std::vector<double> foo_a, fov_a, fvv_a;
  std::vector<double> foo_b, fov_b, fvv_b;
  // Active canonical orbital energies
  std::vector<double> eps_oa, eps_va;
  std::vector<double> eps_ob, eps_vb;

  // Retained host DF intermediates for CPU pullback (populated only when compute_forces=true)
  std::vector<double> raw_three_center;  // [n, n, q]
  std::vector<double> raw_metric;        // [q, q]
  std::vector<double> metric_whitener;   // [q, q]
  std::vector<double> t2_alpha;          // [n_act_a, n_act_a, q]
  std::vector<double> t2_beta;           // [n_act_b, n_act_b, q]
  std::vector<double> c_act_alpha;       // [n, n_act_a]
  std::vector<double> c_act_beta;        // [n, n_act_b]
  double metric_relative_threshold{1e-10};
};

struct SpinBlockedUccAmplitudes {
  std::vector<double> t1a;   // [oa, va]
  std::vector<double> t1b;   // [ob, vb]
  std::vector<double> t2aa;  // [oa, oa, va, va] antisym in (i,j) and (a,b)
  std::vector<double> t2ab;  // [oa, ob, va, vb]
  std::vector<double> t2bb;  // [ob, ob, vb, vb] antisym in (i,j) and (a,b)

  void allocate_zero(const SpinBlockedDims& d) {
    t1a.assign(d.oa * d.va, 0.0);
    t1b.assign(d.ob * d.vb, 0.0);
    t2aa.assign(d.oa * d.oa * d.va * d.va, 0.0);
    t2ab.assign(d.oa * d.ob * d.va * d.vb, 0.0);
    t2bb.assign(d.ob * d.ob * d.vb * d.vb, 0.0);
  }
};

struct SpinBlockedFourIndexCore {
  // Physicists' antisymmetrized (same-spin) and direct (mixed-spin) small 4-index blocks.
  // ZERO ovvv and ZERO vvvv!
  std::vector<double> goovv_aa;  // [oa, oa, va, va] = <ij||ab>_aa
  std::vector<double> goovv_ab;  // [oa, ob, va, vb] = <ij|ab>_ab = (ia|jb)
  std::vector<double> goovv_bb;  // [ob, ob, vb, vb] = <ij||ab>_bb

  std::vector<double> goooo_aa;  // [oa, oa, oa, oa] = <mn||ij>_aa
  std::vector<double> goooo_ab;  // [oa, ob, oa, ob] = <mn|ij>_ab = (mi|nj)
  std::vector<double> goooo_bb;  // [ob, ob, ob, ob] = <mn||ij>_bb

  std::vector<double> gooov_aa;  // [oa, oa, oa, va] = <mn||ie>_aa
  std::vector<double> gooov_bb;  // [ob, ob, ob, vb] = <mn||ie>_bb
  std::vector<double> gooov_ab;  // [oa, ob, oa, vb] = <m_a n_b | i_a e_b> = (m_a i_a | n_b e_b)
  std::vector<double> gooov_ba;  // [ob, oa, ob, va] = <m_b n_a | i_b e_a> = (m_b i_b | n_a e_a)

  std::vector<double> govov_aa;  // [oa, va, oa, va] = <mb||je>_aa = (mj|be) - (me|bj)
  std::vector<double> govov_bb;  // [ob, vb, ob, vb] = <mb||je>_bb = (mj|be) - (me|bj)
  std::vector<double> govov_ab;  // [oa, vb, oa, vb] = <m_a b_b | j_a e_b> = (m_a j_a | b_b e_b)
  std::vector<double> govov_ba;  // [ob, va, ob, va] = <m_b b_a | j_b e_a> = (m_b j_b | b_a e_a)

  void allocate_zero(const SpinBlockedDims& d) {
    goovv_aa.assign(d.oa * d.oa * d.va * d.va, 0.0);
    goovv_ab.assign(d.oa * d.ob * d.va * d.vb, 0.0);
    goovv_bb.assign(d.ob * d.ob * d.vb * d.vb, 0.0);
    goooo_aa.assign(d.oa * d.oa * d.oa * d.oa, 0.0);
    goooo_ab.assign(d.oa * d.ob * d.oa * d.ob, 0.0);
    goooo_bb.assign(d.ob * d.ob * d.ob * d.ob, 0.0);
    gooov_aa.assign(d.oa * d.oa * d.oa * d.va, 0.0);
    gooov_bb.assign(d.ob * d.ob * d.ob * d.vb, 0.0);
    gooov_ab.assign(d.oa * d.ob * d.oa * d.vb, 0.0);
    gooov_ba.assign(d.ob * d.oa * d.ob * d.va, 0.0);
    govov_aa.assign(d.oa * d.va * d.oa * d.va, 0.0);
    govov_bb.assign(d.ob * d.vb * d.ob * d.vb, 0.0);
    govov_ab.assign(d.oa * d.vb * d.oa * d.vb, 0.0);
    govov_ba.assign(d.ob * d.va * d.ob * d.va, 0.0);
  }
};

struct SpinBlockedFactorGrads {
  std::vector<double> boo_a, bov_a, bvv_a;
  std::vector<double> boo_b, bov_b, bvv_b;

  void allocate_zero(const SpinBlockedDims& d) {
    boo_a.assign(d.q * d.oa * d.oa, 0.0);
    bov_a.assign(d.q * d.oa * d.va, 0.0);
    bvv_a.assign(d.q * d.va * d.va, 0.0);
    boo_b.assign(d.q * d.ob * d.ob, 0.0);
    bov_b.assign(d.q * d.ob * d.vb, 0.0);
    bvv_b.assign(d.q * d.vb * d.vb, 0.0);
  }
};

struct SpinBlockedFockGrads {
  std::vector<double> foo_a, fov_a, fvv_a;
  std::vector<double> foo_b, fov_b, fvv_b;

  void allocate_zero(const SpinBlockedDims& d) {
    foo_a.assign(d.oa * d.oa, 0.0);
    fov_a.assign(d.oa * d.va, 0.0);
    fvv_a.assign(d.va * d.va, 0.0);
    foo_b.assign(d.ob * d.ob, 0.0);
    fov_b.assign(d.ob * d.vb, 0.0);
    fvv_b.assign(d.vb * d.vb, 0.0);
  }
};

struct SpinBlockedSourcePullback {
  std::vector<double> corr_gradient;  // [3 * natoms]
  std::vector<double> bar_c_alpha;    // [n * n]
  std::vector<double> bar_c_beta;     // [n * n]
};

struct SpinBlockedCcsdSolveResult {
  double energy{};
  int iterations{};
  bool fp32_executed{};
  SpinBlockedUccAmplitudes amplitudes;
};

// Both triples backends return the unweighted occupied contributions, even when a
// supplied (T+) weight is zero. Every returned value must be finite before weighting.
inline void validate_ucc_triples_partition(double energy, std::span<const double> alpha,
                                           std::span<const double> beta,
                                           const DFUccV2Options& options) {
  if (!std::isfinite(energy))
    throw std::runtime_error("DF-UCCSD(T)/(T+) v2 nonfinite triples energy");
  const bool weighted =
      !options.occupied_weights_alpha.empty() || !options.occupied_weights_beta.empty();
  long double sum = 0.0L, magnitude = 0.0L;
  const auto accumulate_spin = [&](std::span<const double> occupied,
                                   std::span<const double> weights) {
    if (weighted && weights.size() != occupied.size())
      throw std::invalid_argument("DF-UCCSD(T+) v2 occupied weights length mismatch");
    for (std::size_t i = 0; i < occupied.size(); ++i) {
      if (!std::isfinite(occupied[i]))
        throw std::runtime_error("DF-UCCSD(T)/(T+) v2 nonfinite occupied triples energy");
      const long double term = weighted ? static_cast<long double>(weights[i]) * occupied[i]
                                        : static_cast<long double>(occupied[i]);
      sum += term;
      magnitude += std::abs(term);
    }
  };
  accumulate_spin(alpha, options.occupied_weights_alpha);
  accumulate_spin(beta, options.occupied_weights_beta);
  if (!std::isfinite(sum) || !std::isfinite(magnitude) ||
      std::abs(sum - static_cast<long double>(energy)) > 1e-11L * (1.0L + magnitude))
    throw std::runtime_error("DF-UCCSD(T)/(T+) v2 occupied partition failed exact scalar closure");
}

struct SpinBlockedTriplesResult {
  double energy{};
  std::vector<double> occupied_alpha;
  std::vector<double> occupied_beta;
  SpinBlockedUccAmplitudes bar_t;
  SpinBlockedFactorGrads bar_b;       // direct bar_bov, bar_bvv
  SpinBlockedFourIndexCore bar_core;  // bar_goovv, bar_gooov
  SpinBlockedFockGrads bar_f;         // bar_foo, bar_fov, bar_fvv
  std::size_t fp32_gemms{};
  std::size_t fp64_gemms{};
  std::size_t fp32_reverse_gemms{};
};

struct SpinBlockedLambdaSolveResult {
  int iterations{};
  bool fp32_executed{};
  SpinBlockedUccAmplitudes lambda;
  SpinBlockedFactorGrads bar_b;
  SpinBlockedFockGrads bar_f;
  FixedF12Cotangents f12_cotangents;
};

// --- Validation and common helpers ---
SpinBlockedDims validate_and_extract_dims(const core::System& orbital,
                                          const core::System& auxiliary,
                                          const hf::UnrestrictedPhysicalReference& reference,
                                          const DFUccV2Options& options);

void populate_active_fock_and_spectrum(const SpinBlockedDims& dims,
                                       const hf::UnrestrictedPhysicalReference& reference,
                                       SpinBlockedDFSource& source);

void embed_fock_cotangents(const SpinBlockedDims& dims, const SpinBlockedFockGrads& f,
                           std::vector<double>& bar_f_alpha, std::vector<double>& bar_f_beta);

// --- CPU DF source and pullback (df_ucc_v2_source.cpp) ---
SpinBlockedDFSource build_spin_blocked_df_source_cpu(
    const core::System& orbital, const core::System& auxiliary,
    const hf::UnrestrictedPhysicalReference& reference, const DFUccV2Options& options);

SpinBlockedFourIndexCore build_four_index_core_cpu(const SpinBlockedDFSource& src);

void pullback_four_index_core_cpu(const SpinBlockedDFSource& src,
                                  const SpinBlockedFourIndexCore& bar_core,
                                  SpinBlockedFactorGrads& bar_b);

SpinBlockedSourcePullback pullback_spin_blocked_df_source_cpu(
    const core::System& orbital, const core::System& auxiliary, const SpinBlockedDFSource& src,
    const SpinBlockedFactorGrads& bar_b);

// --- CPU 5-block DF-UCCSD solver (df_ucc_v2_ccsd.cpp) ---
double compute_uccsd_energy_cpu(const SpinBlockedDFSource& src,
                                const SpinBlockedFourIndexCore& core,
                                const SpinBlockedUccAmplitudes& amp,
                                const FixedF12Coupling* f12);

SpinBlockedUccAmplitudes compute_uccsd_residual_cpu(const SpinBlockedDFSource& src,
                                                    const SpinBlockedFourIndexCore& core,
                                                    const SpinBlockedUccAmplitudes& amp,
                                                    const FixedF12Coupling* f12);

SpinBlockedCcsdSolveResult solve_df_uccsd_cpu(const SpinBlockedDFSource& src,
                                              const SpinBlockedFourIndexCore& core,
                                              const DFUccV2Options& options);

// --- CPU 4-channel (T)/(T+) + fused pullback (df_ucc_v2_triples.cpp) ---
SpinBlockedTriplesResult evaluate_df_ucc_triples_cpu(const SpinBlockedDFSource& src,
                                                     const SpinBlockedFourIndexCore& core,
                                                     const SpinBlockedUccAmplitudes& amp,
                                                     const DFUccV2Options& options);

// --- CPU Delta-Lambda DIIS + parameter VJP (df_ucc_v2_lambda.cpp) ---
void evaluate_uccsd_residual_vjp_cpu(const SpinBlockedDFSource& src,
                                     const SpinBlockedFourIndexCore& core,
                                     const SpinBlockedUccAmplitudes& amp,
                                     const FixedF12Coupling* f12,
                                     const SpinBlockedUccAmplitudes& lambda,
                                     SpinBlockedUccAmplitudes& bar_t,
                                     SpinBlockedFockGrads* bar_f = nullptr,
                                     SpinBlockedFactorGrads* bar_b = nullptr,
                                     SpinBlockedFourIndexCore* bar_core = nullptr,
                                     FixedF12Cotangents* bar_f12 = nullptr);

SpinBlockedLambdaSolveResult solve_df_ulambda_cpu(const SpinBlockedDFSource& src,
                                                  const SpinBlockedFourIndexCore& core,
                                                  const SpinBlockedUccAmplitudes& amp,
                                                  const SpinBlockedTriplesResult* triples,
                                                  const DFUccV2Options& options);

}  // namespace generativeqc::cc::v2_detail
