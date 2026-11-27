#include "cc/df_ucc_v2_internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "integrals/density_fitting_metric.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "tensor/cpu_linalg.hpp"
#include "tensor/symmetric_matrix_function.hpp"

namespace generativeqc::cc::v2_detail {
namespace {

tensor::CpuLinalgPlan blas_plan() {
  return {tensor::CpuLinalgProvider::automatic, tensor::CpuLinalgThreadOwnership::provider_parallel,
          1};
}

void gemm(char ta, char tb, std::size_t m, std::size_t n, std::size_t k, const double* a,
          const double* b, double* c, double alpha = 1.0, double beta = 0.0) {
  if (!m || !n || !k) {
    if (beta == 0.0 && m && n) std::fill_n(c, m * n, 0.0);
    return;
  }
  tensor::cpu_gemm(ta, tb, m, n, k, a, b, c, alpha, beta, blas_plan());
}

void build_active_coefficients(std::size_t n, std::size_t o_full, std::size_t frozen,
                               const std::vector<double>& c_full, std::vector<double>& c_act) {
  const std::size_t o = o_full - frozen;
  const std::size_t v = n - o_full;
  const std::size_t n_act = o + v;
  c_act.assign(n * n_act, 0.0);
  for (std::size_t mu = 0; mu < n; ++mu) {
    for (std::size_t i = 0; i < o; ++i) c_act[mu * n_act + i] = c_full[mu * n + (frozen + i)];
    for (std::size_t a = 0; a < v; ++a) c_act[mu * n_act + o + a] = c_full[mu * n + (o_full + a)];
  }
}

void build_spin_df_factors_cpu(std::size_t n, std::size_t q, std::size_t o, std::size_t v,
                               const std::vector<double>& c_act,
                               const std::vector<double>& raw_3c,
                               const std::vector<double>& whitener, std::vector<double>& boo,
                               std::vector<double>& bov, std::vector<double>& bvv,
                               std::vector<double>* retained_t2) {
  const std::size_t n_act = o + v;
  boo.assign(q * o * o, 0.0);
  bov.assign(q * o * v, 0.0);
  bvv.assign(q * v * v, 0.0);
  if (!n_act) {
    if (retained_t2) retained_t2->clear();
    return;
  }
  // T1[p, nu, P] = C_act^T[p, mu] @ raw_3c[mu, nu*P]
  std::vector<double> t1(n_act * n * q, 0.0);
  gemm('T', 'N', n_act, n * q, n, c_act.data(), raw_3c.data(), t1.data());

  // T2[p, r, P] = sum_nu C_act[nu, r] T1[p, nu, P]
  std::vector<double> t2(n_act * n_act * q, 0.0);
  for (std::size_t p = 0; p < n_act; ++p) {
    gemm('T', 'N', n_act, q, n, c_act.data(), t1.data() + p * n * q, t2.data() + p * n_act * q);
  }
  std::vector<double>().swap(t1);

  // B_mo[p, r, Q] = T2[p*r, P] @ W[P, Q]
  std::vector<double> bmo(n_act * n_act * q, 0.0);
  gemm('N', 'N', n_act * n_act, q, q, t2.data(), whitener.data(), bmo.data());

  for (std::size_t i = 0; i < o; ++i) {
    for (std::size_t j = 0; j < o; ++j) {
      const double* src = bmo.data() + (i * n_act + j) * q;
      for (std::size_t Q = 0; Q < q; ++Q) boo[(Q * o + i) * o + j] = src[Q];
    }
    for (std::size_t a = 0; a < v; ++a) {
      const double* src = bmo.data() + (i * n_act + (o + a)) * q;
      for (std::size_t Q = 0; Q < q; ++Q) bov[(Q * o + i) * v + a] = src[Q];
    }
  }
  for (std::size_t a = 0; a < v; ++a) {
    for (std::size_t b = 0; b < v; ++b) {
      const double* src = bmo.data() + ((o + a) * n_act + (o + b)) * q;
      for (std::size_t Q = 0; Q < q; ++Q) bvv[(Q * v + a) * v + b] = src[Q];
    }
  }

  if (retained_t2) *retained_t2 = std::move(t2);
}

void pullback_spin_df_factors_cpu(std::size_t n, std::size_t q, std::size_t o_full,
                                  std::size_t frozen, std::size_t o, std::size_t v,
                                  const std::vector<double>& c_act,
                                  const std::vector<double>& raw_3c,
                                  const std::vector<double>& whitener,
                                  const std::vector<double>& t2, const std::vector<double>& bar_boo,
                                  const std::vector<double>& bar_bov,
                                  const std::vector<double>& bar_bvv, std::vector<double>& bar_w,
                                  std::vector<double>& bar_3c, std::vector<double>& bar_c_full) {
  const std::size_t n_act = o + v;
  bar_c_full.assign(n * n, 0.0);
  if (!n_act) return;

  // Embed (bar_boo, bar_bov, bar_bvv) symmetrically into bar_bmo[p, r, Q].
  std::vector<double> bar_bmo(n_act * n_act * q, 0.0);
  for (std::size_t i = 0; i < o; ++i) {
    for (std::size_t j = 0; j < o; ++j) {
      double* dst = bar_bmo.data() + (i * n_act + j) * q;
      for (std::size_t Q = 0; Q < q; ++Q) {
        dst[Q] = 0.5 * (bar_boo[(Q * o + i) * o + j] + bar_boo[(Q * o + j) * o + i]);
      }
    }
    for (std::size_t a = 0; a < v; ++a) {
      double* dst_ov = bar_bmo.data() + (i * n_act + (o + a)) * q;
      double* dst_vo = bar_bmo.data() + ((o + a) * n_act + i) * q;
      for (std::size_t Q = 0; Q < q; ++Q) {
        const double half = 0.5 * bar_bov[(Q * o + i) * v + a];
        dst_ov[Q] = half;
        dst_vo[Q] = half;
      }
    }
  }
  for (std::size_t a = 0; a < v; ++a) {
    for (std::size_t b = 0; b < v; ++b) {
      double* dst = bar_bmo.data() + ((o + a) * n_act + (o + b)) * q;
      for (std::size_t Q = 0; Q < q; ++Q) {
        dst[Q] = 0.5 * (bar_bvv[(Q * v + a) * v + b] + bar_bvv[(Q * v + b) * v + a]);
      }
    }
  }

  // bar_W[P, Q] += T2^T[P, p*r] @ bar_bmo[p*r, Q]
  gemm('T', 'N', q, q, n_act * n_act, t2.data(), bar_bmo.data(), bar_w.data(), 1.0, 1.0);

  // bar_T2[p, r, P] = bar_bmo[p*r, Q] @ W^T[Q, P] (symmetric in (p, r))
  std::vector<double> bar_t2(n_act * n_act * q, 0.0);
  gemm('N', 'T', n_act * n_act, q, q, bar_bmo.data(), whitener.data(), bar_t2.data());
  std::vector<double>().swap(bar_bmo);

  // U[p, nu, P] = sum_r C_act[nu, r] * bar_t2[p, r, P]
  std::vector<double> u(n_act * n * q, 0.0);
  for (std::size_t p = 0; p < n_act; ++p) {
    gemm('N', 'N', n, q, n_act, c_act.data(), bar_t2.data() + p * n_act * q, u.data() + p * n * q);
  }
  std::vector<double>().swap(bar_t2);

  // bar_C_act[mu, p] = 2 * sum_{nu, P} raw_3c[mu, nu, P] * U[p, nu, P]
  std::vector<double> bar_c_act(n * n_act, 0.0);
  gemm('N', 'T', n, n_act, n * q, raw_3c.data(), u.data(), bar_c_act.data(), 2.0, 0.0);
  for (std::size_t mu = 0; mu < n; ++mu) {
    for (std::size_t i = 0; i < o; ++i)
      bar_c_full[mu * n + (frozen + i)] = bar_c_act[mu * n_act + i];
    for (std::size_t a = 0; a < v; ++a)
      bar_c_full[mu * n + (o_full + a)] = bar_c_act[mu * n_act + o + a];
  }

  // bar_3c[mu, nu, P] += sum_p C_act[mu, p] * U[p, nu, P]
  gemm('N', 'N', n, n * q, n_act, c_act.data(), u.data(), bar_3c.data(), 1.0, 1.0);
}

}  // namespace

SpinBlockedDims validate_and_extract_dims(const core::System& orbital,
                                          const core::System& auxiliary,
                                          const hf::UnrestrictedPhysicalReference& reference,
                                          const DFUccV2Options& options) {
  SpinBlockedDims d;
  d.n = reference.nbf;
  d.q = molecule::ao_count(auxiliary);
  d.oa_full = reference.nocc[0];
  d.ob_full = reference.nocc[1];
  d.fa = options.frozen_alpha;
  d.fb = options.frozen_beta;
  d.natoms = orbital.atoms.size();
  if (!d.n || !d.q || molecule::ao_count(orbital) != d.n ||
      auxiliary.atoms.size() != orbital.atoms.size())
    throw std::invalid_argument("DF-UCC v2: invalid orbital/auxiliary basis dimensions");
  if (d.oa_full > d.n || d.ob_full > d.n || d.fa > d.oa_full || d.fb > d.ob_full)
    throw std::invalid_argument("DF-UCC v2: invalid occupied/frozen orbital counts");
  d.oa = d.oa_full - d.fa;
  d.ob = d.ob_full - d.fb;
  d.va = d.n - d.oa_full;
  d.vb = d.n - d.ob_full;
  for (std::size_t s = 0; s < 2; ++s) {
    if (reference.coefficients[s].size() != d.n * d.n || reference.fock[s].size() != d.n * d.n ||
        reference.orbital_energies[s].size() != d.n)
      throw std::invalid_argument("DF-UCC v2: invalid unrestricted reference matrix shapes");
  }
  if (!options.occupied_weights_alpha.empty() && options.occupied_weights_alpha.size() != d.oa)
    throw std::invalid_argument("DF-UCC v2: alpha occupied_weights length mismatch");
  if (!options.occupied_weights_beta.empty() && options.occupied_weights_beta.size() != d.ob)
    throw std::invalid_argument("DF-UCC v2: beta occupied_weights length mismatch");
  if (options.occupied_weights_alpha.empty() != options.occupied_weights_beta.empty() &&
      !(d.oa == 0 || d.ob == 0))
    throw std::invalid_argument(
        "DF-UCC v2: alpha and beta occupied_weights must both be supplied or both empty");
  if (options.f12) {
    const auto& f = *options.f12;
    auto check_block = [](const std::vector<double>& v, std::size_t expected, const char* name) {
      if (v.size() != expected)
        throw std::invalid_argument(std::string("DF-UCC v2: FixedF12Coupling size mismatch for ") +
                                    name);
      if (!std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument(std::string("DF-UCC v2: FixedF12Coupling nonfinite value in ") +
                                    name);
    };
    if (!std::isfinite(f.fixed_geminal_energy))
      throw std::invalid_argument("DF-UCC v2: nonfinite FixedF12Coupling fixed_geminal_energy");
    check_block(f.w_vo_aaaa, d.oa * d.oa * d.va * d.oa, "w_vo_aaaa");
    check_block(f.w_vo_bbbb, d.ob * d.ob * d.vb * d.ob, "w_vo_bbbb");
    check_block(f.w_vo_abab, d.oa * d.ob * d.va * d.ob, "w_vo_abab");
    check_block(f.w_vo_abba, d.oa * d.ob * d.vb * d.oa, "w_vo_abba");
    check_block(f.w_vv_aaaa, d.oa * d.oa * d.va * d.va, "w_vv_aaaa");
    check_block(f.w_vv_abab, d.oa * d.ob * d.va * d.vb, "w_vv_abab");
    check_block(f.w_vv_bbbb, d.ob * d.ob * d.vb * d.vb, "w_vv_bbbb");
    check_block(f.c_f12_aaaa, d.oa * d.oa * d.va * d.va, "c_f12_aaaa");
    check_block(f.c_f12_abab, d.oa * d.ob * d.va * d.vb, "c_f12_abab");
    check_block(f.c_f12_bbbb, d.ob * d.ob * d.vb * d.vb, "c_f12_bbbb");
  }
  return d;
}

void populate_active_fock_and_spectrum(const SpinBlockedDims& d,
                                       const hf::UnrestrictedPhysicalReference& ref,
                                       SpinBlockedDFSource& src) {
  auto fill_spin = [&](std::size_t s, std::size_t f, std::size_t o_full, std::size_t o,
                       std::size_t v, std::vector<double>& foo, std::vector<double>& fov,
                       std::vector<double>& fvv, std::vector<double>& eps_o,
                       std::vector<double>& eps_v) {
    const std::size_t n = d.n;
    // Transform AO Fock to MO basis: F_mo = C^T F_ao C
    std::vector<double> tmp(n * n, 0.0), f_mo(n * n, 0.0);
    gemm('N', 'N', n, n, n, ref.fock[s].data(), ref.coefficients[s].data(), tmp.data());
    gemm('T', 'N', n, n, n, ref.coefficients[s].data(), tmp.data(), f_mo.data());

    foo.assign(o * o, 0.0);
    fov.assign(o * v, 0.0);
    fvv.assign(v * v, 0.0);
    eps_o.assign(o, 0.0);
    eps_v.assign(v, 0.0);
    for (std::size_t i = 0; i < o; ++i) {
      eps_o[i] = ref.orbital_energies[s][f + i];
      for (std::size_t j = 0; j < o; ++j) foo[i * o + j] = f_mo[(f + i) * n + (f + j)];
      for (std::size_t a = 0; a < v; ++a) fov[i * v + a] = f_mo[(f + i) * n + (o_full + a)];
    }
    for (std::size_t a = 0; a < v; ++a) {
      eps_v[a] = ref.orbital_energies[s][o_full + a];
      for (std::size_t b = 0; b < v; ++b) fvv[a * v + b] = f_mo[(o_full + a) * n + (o_full + b)];
    }
  };
  fill_spin(0, d.fa, d.oa_full, d.oa, d.va, src.foo_a, src.fov_a, src.fvv_a, src.eps_oa,
            src.eps_va);
  fill_spin(1, d.fb, d.ob_full, d.ob, d.vb, src.foo_b, src.fov_b, src.fvv_b, src.eps_ob,
            src.eps_vb);
}

void embed_fock_cotangents(const SpinBlockedDims& d, const SpinBlockedFockGrads& f,
                           std::vector<double>& bar_f_alpha, std::vector<double>& bar_f_beta) {
  const std::size_t n = d.n;
  bar_f_alpha.assign(n * n, 0.0);
  bar_f_beta.assign(n * n, 0.0);
  for (std::size_t i = 0; i < d.oa; ++i) {
    for (std::size_t j = 0; j < d.oa; ++j)
      bar_f_alpha[(d.fa + i) * n + (d.fa + j)] = f.foo_a[i * d.oa + j];
    for (std::size_t a = 0; a < d.va; ++a)
      bar_f_alpha[(d.fa + i) * n + (d.oa_full + a)] = f.fov_a[i * d.va + a];
  }
  for (std::size_t a = 0; a < d.va; ++a)
    for (std::size_t b = 0; b < d.va; ++b)
      bar_f_alpha[(d.oa_full + a) * n + (d.oa_full + b)] = f.fvv_a[a * d.va + b];

  for (std::size_t i = 0; i < d.ob; ++i) {
    for (std::size_t j = 0; j < d.ob; ++j)
      bar_f_beta[(d.fb + i) * n + (d.fb + j)] = f.foo_b[i * d.ob + j];
    for (std::size_t a = 0; a < d.vb; ++a)
      bar_f_beta[(d.fb + i) * n + (d.ob_full + a)] = f.fov_b[i * d.vb + a];
  }
  for (std::size_t a = 0; a < d.vb; ++a)
    for (std::size_t b = 0; b < d.vb; ++b)
      bar_f_beta[(d.ob_full + a) * n + (d.ob_full + b)] = f.fvv_b[a * d.vb + b];
}

SpinBlockedDFSource build_spin_blocked_df_source_cpu(
    const core::System& orbital, const core::System& auxiliary,
    const hf::UnrestrictedPhysicalReference& reference, const DFUccV2Options& options) {
  SpinBlockedDFSource src;
  src.dims = validate_and_extract_dims(orbital, auxiliary, reference, options);
  src.metric_relative_threshold = options.metric_relative_threshold;
  populate_active_fock_and_spectrum(src.dims, reference, src);

  const auto& d = src.dims;
  auto raw = integrals::build_density_fitting_integrals(orbital, auxiliary, false);
  if (raw.nbf != d.n || raw.naux != d.q)
    throw std::runtime_error("DF-UCC v2 CPU source dimension mismatch");
  auto metric =
      integrals::factor_density_fitting_metric(raw.metric, d.q, options.metric_relative_threshold);
  if (options.compute_forces && !metric.response_valid)
    throw std::runtime_error("DF-UCC v2 CPU metric rank crossing: response branch unresolved");

  std::vector<double> c_act_a, c_act_b;
  build_active_coefficients(d.n, d.oa_full, d.fa, reference.coefficients[0], c_act_a);
  build_active_coefficients(d.n, d.ob_full, d.fb, reference.coefficients[1], c_act_b);

  build_spin_df_factors_cpu(d.n, d.q, d.oa, d.va, c_act_a, raw.three_center,
                            metric.inverse_square_root, src.boo_a, src.bov_a, src.bvv_a,
                            options.compute_forces ? &src.t2_alpha : nullptr);
  build_spin_df_factors_cpu(d.n, d.q, d.ob, d.vb, c_act_b, raw.three_center,
                            metric.inverse_square_root, src.boo_b, src.bov_b, src.bvv_b,
                            options.compute_forces ? &src.t2_beta : nullptr);

  if (options.compute_forces) {
    src.raw_three_center = std::move(raw.three_center);
    src.raw_metric = std::move(raw.metric);
    src.metric_whitener = std::move(metric.inverse_square_root);
    src.c_act_alpha = std::move(c_act_a);
    src.c_act_beta = std::move(c_act_b);
  }
  return src;
}

SpinBlockedFourIndexCore build_four_index_core_cpu(const SpinBlockedDFSource& src) {
  const auto& d = src.dims;
  SpinBlockedFourIndexCore c;
  c.allocate_zero(d);
  const std::size_t q = d.q, oa = d.oa, ob = d.ob, va = d.va, vb = d.vb;

  // goovv_aa[i, j, a, b] = (ia|jb) - (ib|ja)
  if (oa && va) {
    std::vector<double> ovov(oa * va * oa * va, 0.0);
    gemm('T', 'N', oa * va, oa * va, q, src.bov_a.data(), src.bov_a.data(), ovov.data());
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < oa; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < va; ++b)
            c.goovv_aa[((i * oa + j) * va + a) * va + b] =
                ovov[((i * va + a) * oa + j) * va + b] - ovov[((i * va + b) * oa + j) * va + a];
  }
  // goovv_bb[i, j, a, b] = (ia|jb) - (ib|ja)
  if (ob && vb) {
    std::vector<double> ovov(ob * vb * ob * vb, 0.0);
    gemm('T', 'N', ob * vb, ob * vb, q, src.bov_b.data(), src.bov_b.data(), ovov.data());
    for (std::size_t i = 0; i < ob; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < vb; ++a)
          for (std::size_t b = 0; b < vb; ++b)
            c.goovv_bb[((i * ob + j) * vb + a) * vb + b] =
                ovov[((i * vb + a) * ob + j) * vb + b] - ovov[((i * vb + b) * ob + j) * vb + a];
  }
  // goovv_ab[i, j, a, b] = (i_a a_a | j_b b_b)
  if (oa && ob && va && vb) {
    std::vector<double> ovov(oa * va * ob * vb, 0.0);
    gemm('T', 'N', oa * va, ob * vb, q, src.bov_a.data(), src.bov_b.data(), ovov.data());
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b)
            c.goovv_ab[((i * ob + j) * va + a) * vb + b] = ovov[((i * va + a) * ob + j) * vb + b];
  }

  // goooo_aa[m, n, i, j] = (mi|nj) - (mj|ni)
  if (oa) {
    std::vector<double> oooo(oa * oa * oa * oa, 0.0);
    gemm('T', 'N', oa * oa, oa * oa, q, src.boo_a.data(), src.boo_a.data(), oooo.data());
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < oa; ++n)
        for (std::size_t i = 0; i < oa; ++i)
          for (std::size_t j = 0; j < oa; ++j)
            c.goooo_aa[((m * oa + n) * oa + i) * oa + j] =
                oooo[((m * oa + i) * oa + n) * oa + j] - oooo[((m * oa + j) * oa + n) * oa + i];
  }
  if (ob) {
    std::vector<double> oooo(ob * ob * ob * ob, 0.0);
    gemm('T', 'N', ob * ob, ob * ob, q, src.boo_b.data(), src.boo_b.data(), oooo.data());
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t i = 0; i < ob; ++i)
          for (std::size_t j = 0; j < ob; ++j)
            c.goooo_bb[((m * ob + n) * ob + i) * ob + j] =
                oooo[((m * ob + i) * ob + n) * ob + j] - oooo[((m * ob + j) * ob + n) * ob + i];
  }
  if (oa && ob) {
    std::vector<double> oooo(oa * oa * ob * ob, 0.0);
    gemm('T', 'N', oa * oa, ob * ob, q, src.boo_a.data(), src.boo_b.data(), oooo.data());
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t i = 0; i < oa; ++i)
          for (std::size_t j = 0; j < ob; ++j)
            c.goooo_ab[((m * ob + n) * oa + i) * ob + j] = oooo[((m * oa + i) * ob + n) * ob + j];
  }

  // gooov_aa[m, n, i, e] = (mi|ne) - (me|ni)
  if (oa && va) {
    std::vector<double> ooov(oa * oa * oa * va, 0.0);
    gemm('T', 'N', oa * oa, oa * va, q, src.boo_a.data(), src.bov_a.data(), ooov.data());
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < oa; ++n)
        for (std::size_t i = 0; i < oa; ++i)
          for (std::size_t e = 0; e < va; ++e)
            c.gooov_aa[((m * oa + n) * oa + i) * va + e] =
                ooov[((m * oa + i) * oa + n) * va + e] - ooov[((n * oa + i) * oa + m) * va + e];
  }
  if (ob && vb) {
    std::vector<double> ooov(ob * ob * ob * vb, 0.0);
    gemm('T', 'N', ob * ob, ob * vb, q, src.boo_b.data(), src.bov_b.data(), ooov.data());
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t i = 0; i < ob; ++i)
          for (std::size_t e = 0; e < vb; ++e)
            c.gooov_bb[((m * ob + n) * ob + i) * vb + e] =
                ooov[((m * ob + i) * ob + n) * vb + e] - ooov[((n * ob + i) * ob + m) * vb + e];
  }
  if (oa && ob && vb) {
    std::vector<double> ooov(oa * oa * ob * vb, 0.0);
    gemm('T', 'N', oa * oa, ob * vb, q, src.boo_a.data(), src.bov_b.data(), ooov.data());
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t i = 0; i < oa; ++i)
          for (std::size_t e = 0; e < vb; ++e)
            c.gooov_ab[((m * ob + n) * oa + i) * vb + e] = ooov[((m * oa + i) * ob + n) * vb + e];
  }
  if (ob && oa && va) {
    std::vector<double> ooov(ob * ob * oa * va, 0.0);
    gemm('T', 'N', ob * ob, oa * va, q, src.boo_b.data(), src.bov_a.data(), ooov.data());
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t n = 0; n < oa; ++n)
        for (std::size_t i = 0; i < ob; ++i)
          for (std::size_t e = 0; e < va; ++e)
            c.gooov_ba[((m * oa + n) * ob + i) * va + e] = ooov[((m * ob + i) * oa + n) * va + e];
  }

  // govov_aa[m, b, j, e] = (mj|be) - (me|jb)
  if (oa && va) {
    std::vector<double> oovv(oa * oa * va * va, 0.0);
    std::vector<double> ovov(oa * va * oa * va, 0.0);
    gemm('T', 'N', oa * oa, va * va, q, src.boo_a.data(), src.bvv_a.data(), oovv.data());
    gemm('T', 'N', oa * va, oa * va, q, src.bov_a.data(), src.bov_a.data(), ovov.data());
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t b = 0; b < va; ++b)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t e = 0; e < va; ++e)
            c.govov_aa[((m * va + b) * oa + j) * va + e] =
                oovv[((m * oa + j) * va + b) * va + e] - ovov[((m * va + e) * oa + j) * va + b];
  }
  if (ob && vb) {
    std::vector<double> oovv(ob * ob * vb * vb, 0.0);
    std::vector<double> ovov(ob * vb * ob * vb, 0.0);
    gemm('T', 'N', ob * ob, vb * vb, q, src.boo_b.data(), src.bvv_b.data(), oovv.data());
    gemm('T', 'N', ob * vb, ob * vb, q, src.bov_b.data(), src.bov_b.data(), ovov.data());
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t b = 0; b < vb; ++b)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t e = 0; e < vb; ++e)
            c.govov_bb[((m * vb + b) * ob + j) * vb + e] =
                oovv[((m * ob + j) * vb + b) * vb + e] - ovov[((m * vb + e) * ob + j) * vb + b];
  }
  if (oa && vb) {
    std::vector<double> oovv(oa * oa * vb * vb, 0.0);
    gemm('T', 'N', oa * oa, vb * vb, q, src.boo_a.data(), src.bvv_b.data(), oovv.data());
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t b = 0; b < vb; ++b)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t e = 0; e < vb; ++e)
            c.govov_ab[((m * vb + b) * oa + j) * vb + e] = oovv[((m * oa + j) * vb + b) * vb + e];
  }
  if (ob && va) {
    std::vector<double> oovv(ob * ob * va * va, 0.0);
    gemm('T', 'N', ob * ob, va * va, q, src.boo_b.data(), src.bvv_a.data(), oovv.data());
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t b = 0; b < va; ++b)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t e = 0; e < va; ++e)
            c.govov_ba[((m * va + b) * ob + j) * va + e] = oovv[((m * ob + j) * va + b) * va + e];
  }
  return c;
}

void pullback_four_index_core_cpu(const SpinBlockedDFSource& src,
                                  const SpinBlockedFourIndexCore& bc,
                                  SpinBlockedFactorGrads& bar_b) {
  const auto& d = src.dims;
  const std::size_t q = d.q, oa = d.oa, ob = d.ob, va = d.va, vb = d.vb;

  // goovv_aa -> bar_bov_a
  if (oa && va) {
    std::vector<double> m(oa * va * oa * va, 0.0);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t a = 0; a < va; ++a)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t b = 0; b < va; ++b) {
            const double val = bc.goovv_aa[((i * oa + j) * va + a) * va + b] +
                               bc.goovv_aa[((j * oa + i) * va + b) * va + a] -
                               bc.goovv_aa[((i * oa + j) * va + b) * va + a] -
                               bc.goovv_aa[((j * oa + i) * va + a) * va + b];
            m[((i * va + a) * oa + j) * va + b] = val;
          }
    gemm('N', 'N', q, oa * va, oa * va, src.bov_a.data(), m.data(), bar_b.bov_a.data(), 1.0, 1.0);
  }
  if (ob && vb) {
    std::vector<double> m(ob * vb * ob * vb, 0.0);
    for (std::size_t i = 0; i < ob; ++i)
      for (std::size_t a = 0; a < vb; ++a)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < vb; ++b) {
            const double val = bc.goovv_bb[((i * ob + j) * vb + a) * vb + b] +
                               bc.goovv_bb[((j * ob + i) * vb + b) * vb + a] -
                               bc.goovv_bb[((i * ob + j) * vb + b) * vb + a] -
                               bc.goovv_bb[((j * ob + i) * vb + a) * vb + b];
            m[((i * vb + a) * ob + j) * vb + b] = val;
          }
    gemm('N', 'N', q, ob * vb, ob * vb, src.bov_b.data(), m.data(), bar_b.bov_b.data(), 1.0, 1.0);
  }
  if (oa && ob && va && vb) {
    std::vector<double> m(oa * va * ob * vb, 0.0);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t a = 0; a < va; ++a)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < vb; ++b)
            m[((i * va + a) * ob + j) * vb + b] = bc.goovv_ab[((i * ob + j) * va + a) * vb + b];
    gemm('N', 'T', q, oa * va, ob * vb, src.bov_b.data(), m.data(), bar_b.bov_a.data(), 1.0, 1.0);
    gemm('N', 'N', q, ob * vb, oa * va, src.bov_a.data(), m.data(), bar_b.bov_b.data(), 1.0, 1.0);
  }

  // goooo_aa -> bar_boo_a
  if (oa) {
    std::vector<double> m(oa * oa * oa * oa, 0.0);
    for (std::size_t p = 0; p < oa; ++p)
      for (std::size_t i = 0; i < oa; ++i)
        for (std::size_t r = 0; r < oa; ++r)
          for (std::size_t j = 0; j < oa; ++j) {
            const double val = bc.goooo_aa[((p * oa + r) * oa + i) * oa + j] +
                               bc.goooo_aa[((r * oa + p) * oa + j) * oa + i] -
                               bc.goooo_aa[((p * oa + r) * oa + j) * oa + i] -
                               bc.goooo_aa[((r * oa + p) * oa + i) * oa + j];
            m[((p * oa + i) * oa + r) * oa + j] = val;
          }
    gemm('N', 'N', q, oa * oa, oa * oa, src.boo_a.data(), m.data(), bar_b.boo_a.data(), 1.0, 1.0);
  }
  if (ob) {
    std::vector<double> m(ob * ob * ob * ob, 0.0);
    for (std::size_t p = 0; p < ob; ++p)
      for (std::size_t i = 0; i < ob; ++i)
        for (std::size_t r = 0; r < ob; ++r)
          for (std::size_t j = 0; j < ob; ++j) {
            const double val = bc.goooo_bb[((p * ob + r) * ob + i) * ob + j] +
                               bc.goooo_bb[((r * ob + p) * ob + j) * ob + i] -
                               bc.goooo_bb[((p * ob + r) * ob + j) * ob + i] -
                               bc.goooo_bb[((r * ob + p) * ob + i) * ob + j];
            m[((p * ob + i) * ob + r) * ob + j] = val;
          }
    gemm('N', 'N', q, ob * ob, ob * ob, src.boo_b.data(), m.data(), bar_b.boo_b.data(), 1.0, 1.0);
  }
  if (oa && ob) {
    std::vector<double> m(oa * oa * ob * ob, 0.0);
    for (std::size_t p = 0; p < oa; ++p)
      for (std::size_t i = 0; i < oa; ++i)
        for (std::size_t r = 0; r < ob; ++r)
          for (std::size_t j = 0; j < ob; ++j)
            m[((p * oa + i) * ob + r) * ob + j] = bc.goooo_ab[((p * ob + r) * oa + i) * ob + j];
    gemm('N', 'T', q, oa * oa, ob * ob, src.boo_b.data(), m.data(), bar_b.boo_a.data(), 1.0, 1.0);
    gemm('N', 'N', q, ob * ob, oa * oa, src.boo_a.data(), m.data(), bar_b.boo_b.data(), 1.0, 1.0);
  }

  // gooov_aa -> bar_boo_a, bar_bov_a
  if (oa && va) {
    std::vector<double> m(oa * oa * oa * va, 0.0);
    for (std::size_t p = 0; p < oa; ++p)
      for (std::size_t i = 0; i < oa; ++i)
        for (std::size_t r = 0; r < oa; ++r)
          for (std::size_t e = 0; e < va; ++e)
            m[((p * oa + i) * oa + r) * va + e] = bc.gooov_aa[((p * oa + r) * oa + i) * va + e] -
                                                  bc.gooov_aa[((r * oa + p) * oa + i) * va + e];
    gemm('N', 'T', q, oa * oa, oa * va, src.bov_a.data(), m.data(), bar_b.boo_a.data(), 1.0, 1.0);
    gemm('N', 'N', q, oa * va, oa * oa, src.boo_a.data(), m.data(), bar_b.bov_a.data(), 1.0, 1.0);
  }
  if (ob && vb) {
    std::vector<double> m(ob * ob * ob * vb, 0.0);
    for (std::size_t p = 0; p < ob; ++p)
      for (std::size_t i = 0; i < ob; ++i)
        for (std::size_t r = 0; r < ob; ++r)
          for (std::size_t e = 0; e < vb; ++e)
            m[((p * ob + i) * ob + r) * vb + e] = bc.gooov_bb[((p * ob + r) * ob + i) * vb + e] -
                                                  bc.gooov_bb[((r * ob + p) * ob + i) * vb + e];
    gemm('N', 'T', q, ob * ob, ob * vb, src.bov_b.data(), m.data(), bar_b.boo_b.data(), 1.0, 1.0);
    gemm('N', 'N', q, ob * vb, ob * ob, src.boo_b.data(), m.data(), bar_b.bov_b.data(), 1.0, 1.0);
  }
  if (oa && ob && vb) {
    std::vector<double> m(oa * oa * ob * vb, 0.0);
    for (std::size_t p = 0; p < oa; ++p)
      for (std::size_t i = 0; i < oa; ++i)
        for (std::size_t r = 0; r < ob; ++r)
          for (std::size_t e = 0; e < vb; ++e)
            m[((p * oa + i) * ob + r) * vb + e] = bc.gooov_ab[((p * ob + r) * oa + i) * vb + e];
    gemm('N', 'T', q, oa * oa, ob * vb, src.bov_b.data(), m.data(), bar_b.boo_a.data(), 1.0, 1.0);
    gemm('N', 'N', q, ob * vb, oa * oa, src.boo_a.data(), m.data(), bar_b.bov_b.data(), 1.0, 1.0);
  }
  if (ob && oa && va) {
    std::vector<double> m(ob * ob * oa * va, 0.0);
    for (std::size_t p = 0; p < ob; ++p)
      for (std::size_t i = 0; i < ob; ++i)
        for (std::size_t r = 0; r < oa; ++r)
          for (std::size_t e = 0; e < va; ++e)
            m[((p * ob + i) * oa + r) * va + e] = bc.gooov_ba[((p * oa + r) * ob + i) * va + e];
    gemm('N', 'T', q, ob * ob, oa * va, src.bov_a.data(), m.data(), bar_b.boo_b.data(), 1.0, 1.0);
    gemm('N', 'N', q, oa * va, ob * ob, src.boo_b.data(), m.data(), bar_b.bov_a.data(), 1.0, 1.0);
  }

  // govov_aa -> bar_boo_a, bar_bvv_a, bar_bov_a
  if (oa && va) {
    std::vector<double> mc(oa * oa * va * va, 0.0);
    std::vector<double> mx(oa * va * oa * va, 0.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t j = 0; j < oa; ++j)
        for (std::size_t b = 0; b < va; ++b)
          for (std::size_t e = 0; e < va; ++e) {
            const double val = bc.govov_aa[((m * va + b) * oa + j) * va + e];
            mc[((m * oa + j) * va + b) * va + e] = val;
            mx[((m * va + e) * oa + j) * va + b] += val;
            mx[((j * va + b) * oa + m) * va + e] += val;
          }
    gemm('N', 'T', q, oa * oa, va * va, src.bvv_a.data(), mc.data(), bar_b.boo_a.data(), 1.0, 1.0);
    gemm('N', 'N', q, va * va, oa * oa, src.boo_a.data(), mc.data(), bar_b.bvv_a.data(), 1.0, 1.0);
    gemm('N', 'N', q, oa * va, oa * va, src.bov_a.data(), mx.data(), bar_b.bov_a.data(), -1.0, 1.0);
  }
  if (ob && vb) {
    std::vector<double> mc(ob * ob * vb * vb, 0.0);
    std::vector<double> mx(ob * vb * ob * vb, 0.0);
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t b = 0; b < vb; ++b)
          for (std::size_t e = 0; e < vb; ++e) {
            const double val = bc.govov_bb[((m * vb + b) * ob + j) * vb + e];
            mc[((m * ob + j) * vb + b) * vb + e] = val;
            mx[((m * vb + e) * ob + j) * vb + b] += val;
            mx[((j * vb + b) * ob + m) * vb + e] += val;
          }
    gemm('N', 'T', q, ob * ob, vb * vb, src.bvv_b.data(), mc.data(), bar_b.boo_b.data(), 1.0, 1.0);
    gemm('N', 'N', q, vb * vb, ob * ob, src.boo_b.data(), mc.data(), bar_b.bvv_b.data(), 1.0, 1.0);
    gemm('N', 'N', q, ob * vb, ob * vb, src.bov_b.data(), mx.data(), bar_b.bov_b.data(), -1.0, 1.0);
  }
  if (oa && vb) {
    std::vector<double> mc(oa * oa * vb * vb, 0.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t j = 0; j < oa; ++j)
        for (std::size_t b = 0; b < vb; ++b)
          for (std::size_t e = 0; e < vb; ++e)
            mc[((m * oa + j) * vb + b) * vb + e] = bc.govov_ab[((m * vb + b) * oa + j) * vb + e];
    gemm('N', 'T', q, oa * oa, vb * vb, src.bvv_b.data(), mc.data(), bar_b.boo_a.data(), 1.0, 1.0);
    gemm('N', 'N', q, vb * vb, oa * oa, src.boo_a.data(), mc.data(), bar_b.bvv_b.data(), 1.0, 1.0);
  }
  if (ob && va) {
    std::vector<double> mc(ob * ob * va * va, 0.0);
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t b = 0; b < va; ++b)
          for (std::size_t e = 0; e < va; ++e)
            mc[((m * ob + j) * va + b) * va + e] = bc.govov_ba[((m * va + b) * ob + j) * va + e];
    gemm('N', 'T', q, ob * ob, va * va, src.bvv_a.data(), mc.data(), bar_b.boo_b.data(), 1.0, 1.0);
    gemm('N', 'N', q, va * va, ob * ob, src.boo_b.data(), mc.data(), bar_b.bvv_a.data(), 1.0, 1.0);
  }
}

SpinBlockedSourcePullback pullback_spin_blocked_df_source_cpu(
    const core::System& orbital, const core::System& auxiliary, const SpinBlockedDFSource& src,
    const SpinBlockedFactorGrads& bar_b) {
  const auto& d = src.dims;
  SpinBlockedSourcePullback out;
  std::vector<double> bar_w(d.q * d.q, 0.0);
  std::vector<double> bar_3c(d.n * d.n * d.q, 0.0);

  pullback_spin_df_factors_cpu(d.n, d.q, d.oa_full, d.fa, d.oa, d.va, src.c_act_alpha,
                               src.raw_three_center, src.metric_whitener, src.t2_alpha,
                               bar_b.boo_a, bar_b.bov_a, bar_b.bvv_a, bar_w, bar_3c,
                               out.bar_c_alpha);
  pullback_spin_df_factors_cpu(d.n, d.q, d.ob_full, d.fb, d.ob, d.vb, src.c_act_beta,
                               src.raw_three_center, src.metric_whitener, src.t2_beta,
                               bar_b.boo_b, bar_b.bov_b, bar_b.bvv_b, bar_w, bar_3c,
                               out.bar_c_beta);

  // Symmetrize bar_3c in (mu, nu)
  for (std::size_t mu = 0; mu < d.n; ++mu)
    for (std::size_t nu = mu + 1; nu < d.n; ++nu)
      for (std::size_t P = 0; P < d.q; ++P) {
        const double avg = 0.5 * (bar_3c[(mu * d.n + nu) * d.q + P] +
                                  bar_3c[(nu * d.n + mu) * d.q + P]);
        bar_3c[(mu * d.n + nu) * d.q + P] = avg;
        bar_3c[(nu * d.n + mu) * d.q + P] = avg;
      }

  const auto bar_metric = integrals::density_fitting_metric_response(
      src.raw_metric, src.metric_whitener, bar_w, d.q, src.metric_relative_threshold,
      tensor::SymmetricMatrixFunction::inverse_sqrt);

  out.corr_gradient = integrals::contract_weighted_density_fitting_derivative(
      orbital, auxiliary, bar_metric, bar_3c);
  return out;
}

}  // namespace generativeqc::cc::v2_detail
