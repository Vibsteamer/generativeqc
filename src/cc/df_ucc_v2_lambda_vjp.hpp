#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "cc/df_ucc_v2_internal.hpp"
#include "tensor/cpu_linalg.hpp"

namespace generativeqc::cc::v2_detail::lambda_vjp {

inline tensor::CpuLinalgPlan blas_plan() {
  return {tensor::CpuLinalgProvider::automatic, tensor::CpuLinalgThreadOwnership::provider_parallel,
          1};
}

inline void gemm(char ta, char tb, std::size_t m, std::size_t n, std::size_t k, const double* a,
                 const double* b, double* c, double alpha = 1.0, double beta = 0.0) {
  if (!m || !n || !k) {
    if (beta == 0.0 && m && n) std::fill_n(c, m * n, 0.0);
    return;
  }
  tensor::cpu_gemm(ta, tb, m, n, k, a, b, c, alpha, beta, blas_plan());
}

inline void antisymmetrize_t2(std::size_t o, std::size_t v, std::vector<double>& t2) {
  if (!o || !v) return;
  std::vector<double> out(t2.size(), 0.0);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b)
          out[((i * o + j) * v + a) * v + b] =
              0.25 * (t2[((i * o + j) * v + a) * v + b] - t2[((j * o + i) * v + a) * v + b] -
                      t2[((i * o + j) * v + b) * v + a] + t2[((j * o + i) * v + b) * v + a]);
  t2.swap(out);
}

inline void add_scaled(std::vector<double>& dst, const std::vector<double>& src, double scale = 1.0) {
  for (std::size_t i = 0; i < dst.size(); ++i) dst[i] += scale * src[i];
}

struct SpinRefs {
  std::size_t o[2]{}, v[2]{};
  const std::vector<double>* bov[2]{};
  const std::vector<double>* bvv[2]{};
  const std::vector<double>* foo[2]{};
  const std::vector<double>* fov[2]{};
  const std::vector<double>* fvv[2]{};
  const std::vector<double>* t1[2]{};
  const std::vector<double>* t2[2]{};
  const std::vector<double>* goovv[2]{};
  const std::vector<double>* goooo[2]{};
  const std::vector<double>* gooov[2]{};
  const std::vector<double>* govov[2]{};
};

inline SpinRefs make_spin_refs(const SpinBlockedDFSource& src, const SpinBlockedFourIndexCore& core,
                               const SpinBlockedUccAmplitudes& amp) {
  SpinRefs r;
  r.o[0] = src.dims.oa; r.o[1] = src.dims.ob;
  r.v[0] = src.dims.va; r.v[1] = src.dims.vb;
  r.bov[0] = &src.bov_a; r.bov[1] = &src.bov_b;
  r.bvv[0] = &src.bvv_a; r.bvv[1] = &src.bvv_b;
  r.foo[0] = &src.foo_a; r.foo[1] = &src.foo_b;
  r.fov[0] = &src.fov_a; r.fov[1] = &src.fov_b;
  r.fvv[0] = &src.fvv_a; r.fvv[1] = &src.fvv_b;
  r.t1[0] = &amp.t1a; r.t1[1] = &amp.t1b;
  r.t2[0] = &amp.t2aa; r.t2[1] = &amp.t2bb;
  r.goovv[0] = &core.goovv_aa; r.goovv[1] = &core.goovv_bb;
  r.goooo[0] = &core.goooo_aa; r.goooo[1] = &core.goooo_bb;
  r.gooov[0] = &core.gooov_aa; r.gooov[1] = &core.gooov_bb;
  r.govov[0] = &core.govov_aa; r.govov[1] = &core.govov_bb;
  return r;
}

// Forward intermediates cached once per converged amplitude set T
struct ForwardCache {
  std::vector<double> tau_s[2], tau_ab;
  std::vector<double> tauh_s[2], tauh_ab;
  std::vector<double> j_q;
  std::vector<double> x_s[2], y_s[2], bt_s[2];
  std::vector<double> Fvv_s[2], Foo_s[2], Fov_s[2], Fvv_d_s[2], Foo_d_s[2];
  std::vector<double> z_s[2];
  std::vector<double> w_bare_s[2], w_bare_ab;
  std::vector<double> tau_oovv_s[2], g_ab;
  std::vector<double> W_ss[2], W_abab, W_baba, W_abba, W_baab;
  std::vector<double> u_tg_s[2], t1_mjb_i, u2_ab, u3_ab, u4_ab;
};

inline void build_tau_block(const SpinBlockedDims& d, const SpinBlockedUccAmplitudes& amp,
                            double factor, std::vector<double> tau_s[2],
                            std::vector<double>& tau_ab) {
  tau_s[0] = amp.t2aa;
  tau_s[1] = amp.t2bb;
  tau_ab = amp.t2ab;
  const std::size_t os[2] = {d.oa, d.ob}, vs[2] = {d.va, d.vb};
  const std::vector<double>* t1s[2] = {&amp.t1a, &amp.t1b};
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = os[s], v = vs[s];
    const auto& t1 = *t1s[s];
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b)
            tau_s[s][((i * o + j) * v + a) * v + b] +=
                factor * (t1[i * v + a] * t1[j * v + b] - t1[i * v + b] * t1[j * v + a]);
  }
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t j = 0; j < d.ob; ++j)
      for (std::size_t a = 0; a < d.va; ++a)
        for (std::size_t b = 0; b < d.vb; ++b)
          tau_ab[((i * d.ob + j) * d.va + a) * d.vb + b] +=
              factor * amp.t1a[i * d.va + a] * amp.t1b[j * d.vb + b];
}

inline ForwardCache build_forward_cache(const SpinBlockedDFSource& src,
                                        const SpinBlockedFourIndexCore& core,
                                        const SpinBlockedUccAmplitudes& amp) {
  const auto& d = src.dims;
  const std::size_t q = d.q, oa = d.oa, ob = d.ob, va = d.va, vb = d.vb;
  const auto sr = make_spin_refs(src, core, amp);
  ForwardCache c;

  build_tau_block(d, amp, 1.0, c.tau_s, c.tau_ab);
  build_tau_block(d, amp, 0.5, c.tauh_s, c.tauh_ab);

  c.j_q.assign(q, 0.0);
  for (std::size_t Q = 0; Q < q; ++Q) {
    double s = 0.0;
    for (int sp = 0; sp < 2; ++sp) {
      const double* b = sr.bov[sp]->data() + Q * sr.o[sp] * sr.v[sp];
      const auto& t1 = *sr.t1[sp];
      for (std::size_t k = 0; k < t1.size(); ++k) s += b[k] * t1[k];
    }
    c.j_q[Q] = s;
  }

  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    c.x_s[s].assign(q * o * v, 0.0);
    c.y_s[s].assign(q * o * o, 0.0);
    c.bt_s[s] = *sr.bvv[s];
    for (std::size_t Q = 0; Q < q; ++Q) {
      gemm('N', 'T', o, v, v, sr.t1[s]->data(), sr.bvv[s]->data() + Q * v * v,
           c.x_s[s].data() + Q * o * v);
      gemm('N', 'T', o, o, v, sr.bov[s]->data() + Q * o * v, sr.t1[s]->data(),
           c.y_s[s].data() + Q * o * o);
      gemm('T', 'N', v, v, o, sr.t1[s]->data(), sr.bov[s]->data() + Q * o * v,
           c.bt_s[s].data() + Q * v * v, -1.0, 1.0);
    }
  }

  // Fvv, Foo, Fov, Fvv_d, Foo_d
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    c.Fvv_s[s] = *sr.fvv[s];
    gemm('T', 'N', v, v, o, sr.t1[s]->data(), sr.fov[s]->data(), c.Fvv_s[s].data(), -0.5, 1.0);
    for (std::size_t Q = 0; Q < q; ++Q) {
      const double jq = c.j_q[Q];
      const double* bv = sr.bvv[s]->data() + Q * v * v;
      for (std::size_t k = 0; k < v * v; ++k) c.Fvv_s[s][k] += jq * bv[k];
    }
    gemm('T', 'N', v, v, q * o, c.x_s[s].data(), sr.bov[s]->data(), c.Fvv_s[s].data(), -1.0, 1.0);
    for (std::size_t mn = 0; mn < o * o; ++mn)
      gemm('N', 'T', v, v, v, c.tauh_s[s].data() + mn * v * v,
           sr.goovv[s]->data() + mn * v * v, c.Fvv_s[s].data(), -0.5, 1.0);
  }
  for (std::size_t mn = 0; mn < oa * ob; ++mn) {
    gemm('N', 'T', va, va, vb, c.tauh_ab.data() + mn * va * vb,
         core.goovv_ab.data() + mn * va * vb, c.Fvv_s[0].data(), -1.0, 1.0);
    gemm('T', 'N', vb, vb, va, c.tauh_ab.data() + mn * va * vb,
         core.goovv_ab.data() + mn * va * vb, c.Fvv_s[1].data(), -1.0, 1.0);
  }

  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    c.Foo_s[s] = *sr.foo[s];
    gemm('N', 'T', o, o, v, sr.fov[s]->data(), sr.t1[s]->data(), c.Foo_s[s].data(), 0.5, 1.0);
    for (std::size_t m = 0; m < o; ++m)
      for (std::size_t n = 0; n < o; ++n)
        for (std::size_t i = 0; i < o; ++i) {
          double acc = 0.0;
          const double* g = sr.gooov[s]->data() + ((m * o + n) * o + i) * v;
          const double* t = sr.t1[s]->data() + n * v;
          for (std::size_t e = 0; e < v; ++e) acc += g[e] * t[e];
          c.Foo_s[s][m * o + i] += acc;
        }
    gemm('N', 'T', o, o, o * v * v, sr.goovv[s]->data(), c.tauh_s[s].data(), c.Foo_s[s].data(),
         0.5, 1.0);
  }
  for (std::size_t m = 0; m < oa; ++m)
    for (std::size_t n = 0; n < ob; ++n)
      for (std::size_t i = 0; i < oa; ++i) {
        double acc = 0.0;
        const double* g = core.gooov_ab.data() + ((m * ob + n) * oa + i) * vb;
        const double* t = amp.t1b.data() + n * vb;
        for (std::size_t e = 0; e < vb; ++e) acc += g[e] * t[e];
        c.Foo_s[0][m * oa + i] += acc;
      }
  for (std::size_t m = 0; m < ob; ++m)
    for (std::size_t n = 0; n < oa; ++n)
      for (std::size_t i = 0; i < ob; ++i) {
        double acc = 0.0;
        const double* g = core.gooov_ba.data() + ((m * oa + n) * ob + i) * va;
        const double* t = amp.t1a.data() + n * va;
        for (std::size_t e = 0; e < va; ++e) acc += g[e] * t[e];
        c.Foo_s[1][m * ob + i] += acc;
      }
  gemm('N', 'T', oa, oa, ob * va * vb, core.goovv_ab.data(), c.tauh_ab.data(), c.Foo_s[0].data(),
       1.0, 1.0);
  for (std::size_t n = 0; n < oa; ++n)
    gemm('N', 'T', ob, ob, va * vb, core.goovv_ab.data() + n * ob * va * vb,
         c.tauh_ab.data() + n * ob * va * vb, c.Foo_s[1].data(), 1.0, 1.0);

  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    c.Fov_s[s] = *sr.fov[s];
    for (std::size_t m = 0; m < o; ++m)
      for (std::size_t e = 0; e < v; ++e) {
        double acc = 0.0;
        for (std::size_t n = 0; n < o; ++n)
          for (std::size_t f = 0; f < v; ++f)
            acc += (*sr.goovv[s])[((m * o + n) * v + e) * v + f] * (*sr.t1[s])[n * v + f];
        c.Fov_s[s][m * v + e] += acc;
      }
  }
  for (std::size_t m = 0; m < oa; ++m)
    for (std::size_t e = 0; e < va; ++e) {
      double acc = 0.0;
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t f = 0; f < vb; ++f)
          acc += core.goovv_ab[((m * ob + n) * va + e) * vb + f] * amp.t1b[n * vb + f];
      c.Fov_s[0][m * va + e] += acc;
    }
  for (std::size_t m = 0; m < ob; ++m)
    for (std::size_t e = 0; e < vb; ++e) {
      double acc = 0.0;
      for (std::size_t n = 0; n < oa; ++n)
        for (std::size_t f = 0; f < va; ++f)
          acc += core.goovv_ab[((n * ob + m) * va + f) * vb + e] * amp.t1a[n * va + f];
      c.Fov_s[1][m * vb + e] += acc;
    }

  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    c.Fvv_d_s[s] = c.Fvv_s[s];
    gemm('T', 'N', v, v, o, sr.t1[s]->data(), c.Fov_s[s].data(), c.Fvv_d_s[s].data(), -0.5, 1.0);
    c.Foo_d_s[s] = c.Foo_s[s];
    gemm('N', 'T', o, o, v, c.Fov_s[s].data(), sr.t1[s]->data(), c.Foo_d_s[s].data(), 0.5, 1.0);
  }

  // DF r1 intermediate z_s[Q, i, f]
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s], o2 = sr.o[1 - s], v2 = sr.v[1 - s];
    c.z_s[s].assign(q * o * v, 0.0);
    if (!o || !v) continue;
    std::vector<double> t2_ifme(o * v * o * v, 0.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t m = 0; m < o; ++m)
        for (std::size_t e = 0; e < v; ++e)
          for (std::size_t f = 0; f < v; ++f)
            t2_ifme[((i * v + f) * o + m) * v + e] = (*sr.t2[s])[((i * o + m) * v + e) * v + f];
    gemm('N', 'T', q, o * v, o * v, sr.bov[s]->data(), t2_ifme.data(), c.z_s[s].data(), -1.0, 0.0);
    if (o2 && v2) {
      std::vector<double> tab(o * v * o2 * v2, 0.0);
      if (s == 0) {
        for (std::size_t i = 0; i < oa; ++i)
          for (std::size_t m = 0; m < ob; ++m)
            for (std::size_t e = 0; e < va; ++e)
              for (std::size_t f = 0; f < vb; ++f)
                tab[((i * va + e) * ob + m) * vb + f] =
                    amp.t2ab[((i * ob + m) * va + e) * vb + f];
      } else {
        for (std::size_t m = 0; m < oa; ++m)
          for (std::size_t i = 0; i < ob; ++i)
            for (std::size_t e = 0; e < va; ++e)
              for (std::size_t f = 0; f < vb; ++f)
                tab[((i * vb + f) * oa + m) * va + e] =
                    amp.t2ab[((m * ob + i) * va + e) * vb + f];
      }
      gemm('N', 'T', q, o * v, o2 * v2, sr.bov[1 - s]->data(), tab.data(), c.z_s[s].data(), 1.0,
           1.0);
    }
  }

  // w_bare and tau_oovv
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    if (!o || !v) continue;
    c.w_bare_s[s] = *sr.goooo[s];
    for (std::size_t mn = 0; mn < o * o; ++mn)
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t j = 0; j < o; ++j) {
          double acc = 0.0;
          const double* gi = sr.gooov[s]->data() + (mn * o + i) * v;
          const double* gj = sr.gooov[s]->data() + (mn * o + j) * v;
          const double* ti = sr.t1[s]->data() + i * v;
          const double* tj = sr.t1[s]->data() + j * v;
          for (std::size_t e = 0; e < v; ++e) acc += tj[e] * gi[e] - ti[e] * gj[e];
          c.w_bare_s[s][mn * o * o + i * o + j] += acc;
        }
    c.tau_oovv_s[s].assign(o * o * o * o, 0.0);
    gemm('N', 'T', o * o, o * o, v * v, sr.goovv[s]->data(), c.tau_s[s].data(),
         c.tau_oovv_s[s].data());
  }
  if (oa && ob && va && vb) {
    c.w_bare_ab = core.goooo_ab;
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t i = 0; i < oa; ++i)
          for (std::size_t j = 0; j < ob; ++j) {
            double acc = 0.0;
            const double* gab = core.gooov_ab.data() + ((m * ob + n) * oa + i) * vb;
            const double* tj = amp.t1b.data() + j * vb;
            for (std::size_t e = 0; e < vb; ++e) acc += tj[e] * gab[e];
            const double* gba = core.gooov_ba.data() + ((n * oa + m) * ob + j) * va;
            const double* ti = amp.t1a.data() + i * va;
            for (std::size_t e = 0; e < va; ++e) acc += ti[e] * gba[e];
            c.w_bare_ab[((m * ob + n) * oa + i) * ob + j] += acc;
          }
    c.g_ab.assign(oa * ob * oa * ob, 0.0);
    gemm('N', 'T', oa * ob, oa * ob, va * vb, core.goovv_ab.data(), c.tau_ab.data(), c.g_ab.data());
  }

  // Wovvo (6 spin-conserving blocks)
  std::vector<double> g_menf_s[2], t_nfjb_s[2];
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    c.W_ss[s].assign(o * v * o * v, 0.0);
    if (!o || !v) continue;
    gemm('T', 'N', o * v, o * v, q, sr.bov[s]->data(), c.x_s[s].data(), c.W_ss[s].data(), 1.0, 0.0);
    std::vector<double> y_bvv(o * o * v * v, 0.0);
    gemm('T', 'N', o * o, v * v, q, c.y_s[s].data(), sr.bvv[s]->data(), y_bvv.data());
    for (std::size_t m = 0; m < o; ++m)
      for (std::size_t e = 0; e < v; ++e)
        for (std::size_t j = 0; j < o; ++j)
          for (std::size_t b = 0; b < v; ++b) {
            double val = -(*sr.govov[s])[((m * v + b) * o + j) * v + e] -
                         y_bvv[((m * o + j) * v + b) * v + e];
            for (std::size_t n = 0; n < o; ++n)
              val += (*sr.t1[s])[n * v + b] * (*sr.gooov[s])[((m * o + n) * o + j) * v + e];
            c.W_ss[s][((m * v + e) * o + j) * v + b] += val;
          }
    g_menf_s[s].assign(o * v * o * v, 0.0);
    t_nfjb_s[s].assign(o * v * o * v, 0.0);
    for (std::size_t m = 0; m < o; ++m)
      for (std::size_t n = 0; n < o; ++n)
        for (std::size_t e = 0; e < v; ++e)
          for (std::size_t f = 0; f < v; ++f)
            g_menf_s[s][((m * v + e) * o + n) * v + f] =
                (*sr.goovv[s])[((m * o + n) * v + e) * v + f];
    for (std::size_t n = 0; n < o; ++n)
      for (std::size_t f = 0; f < v; ++f)
        for (std::size_t j = 0; j < o; ++j)
          for (std::size_t b = 0; b < v; ++b)
            t_nfjb_s[s][((n * v + f) * o + j) * v + b] =
                0.5 * (*sr.t2[s])[((j * o + n) * v + f) * v + b] +
                (*sr.t1[s])[j * v + f] * (*sr.t1[s])[n * v + b];
    gemm('N', 'N', o * v, o * v, o * v, g_menf_s[s].data(), t_nfjb_s[s].data(), c.W_ss[s].data(),
         -1.0, 1.0);
  }
  if (oa && ob && va && vb) {
    std::vector<double> gab_menf(oa * va * ob * vb, 0.0), tab_nfa_jbb(oa * va * ob * vb, 0.0),
        tab_nfb_jaa(ob * vb * oa * va, 0.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t e = 0; e < va; ++e)
          for (std::size_t f = 0; f < vb; ++f) {
            gab_menf[((m * va + e) * ob + n) * vb + f] =
                core.goovv_ab[((m * ob + n) * va + e) * vb + f];
            tab_nfa_jbb[((m * va + e) * ob + n) * vb + f] =
                amp.t2ab[((m * ob + n) * va + e) * vb + f];
            tab_nfb_jaa[((n * vb + f) * oa + m) * va + e] =
                amp.t2ab[((m * ob + n) * va + e) * vb + f];
          }
    gemm('N', 'N', oa * va, oa * va, ob * vb, gab_menf.data(), tab_nfb_jaa.data(),
         c.W_ss[0].data(), 0.5, 1.0);
    gemm('T', 'N', ob * vb, ob * vb, oa * va, gab_menf.data(), tab_nfa_jbb.data(),
         c.W_ss[1].data(), 0.5, 1.0);

    c.W_abab.assign(oa * va * ob * vb, 0.0);
    c.W_baba.assign(ob * vb * oa * va, 0.0);
    gemm('T', 'N', oa * va, ob * vb, q, src.bov_a.data(), c.x_s[1].data(), c.W_abab.data(), 1.0,
         0.0);
    gemm('T', 'N', ob * vb, oa * va, q, src.bov_b.data(), c.x_s[0].data(), c.W_baba.data(), 1.0,
         0.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t e = 0; e < va; ++e)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < vb; ++b) {
            double val = core.goovv_ab[((m * ob + j) * va + e) * vb + b];
            for (std::size_t n = 0; n < ob; ++n)
              val -= amp.t1b[n * vb + b] * core.gooov_ba[((n * oa + m) * ob + j) * va + e];
            c.W_abab[((m * va + e) * ob + j) * vb + b] += val;
          }
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t e = 0; e < vb; ++e)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t b = 0; b < va; ++b) {
            double val = core.goovv_ab[((j * ob + m) * va + b) * vb + e];
            for (std::size_t n = 0; n < oa; ++n)
              val -= amp.t1a[n * va + b] * core.gooov_ab[((n * ob + m) * oa + j) * vb + e];
            c.W_baba[((m * vb + e) * oa + j) * va + b] += val;
          }
    gemm('N', 'N', oa * va, ob * vb, ob * vb, gab_menf.data(), t_nfjb_s[1].data(), c.W_abab.data(),
         -1.0, 1.0);
    gemm('N', 'N', oa * va, ob * vb, oa * va, g_menf_s[0].data(), tab_nfa_jbb.data(),
         c.W_abab.data(), 0.5, 1.0);
    gemm('T', 'N', ob * vb, oa * va, oa * va, gab_menf.data(), t_nfjb_s[0].data(), c.W_baba.data(),
         -1.0, 1.0);
    gemm('N', 'N', ob * vb, oa * va, ob * vb, g_menf_s[1].data(), tab_nfb_jaa.data(),
         c.W_baba.data(), 0.5, 1.0);

    c.W_abba.assign(oa * vb * oa * vb, 0.0);
    std::vector<double> ya_bvv_b(oa * oa * vb * vb, 0.0);
    gemm('T', 'N', oa * oa, vb * vb, q, c.y_s[0].data(), src.bvv_b.data(), ya_bvv_b.data());
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t e = 0; e < vb; ++e)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t b = 0; b < vb; ++b) {
            double val = -core.govov_ab[((m * vb + b) * oa + j) * vb + e] -
                         ya_bvv_b[((m * oa + j) * vb + b) * vb + e];
            for (std::size_t n = 0; n < ob; ++n)
              val += amp.t1b[n * vb + b] * core.gooov_ab[((m * ob + n) * oa + j) * vb + e];
            c.W_abba[((m * vb + e) * oa + j) * vb + b] = val;
          }
    std::vector<double> gab_x(oa * vb * ob * va, 0.0), tab_x(ob * va * oa * vb, 0.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t f = 0; f < va; ++f)
          for (std::size_t e = 0; e < vb; ++e)
            gab_x[((m * vb + e) * ob + n) * va + f] = core.goovv_ab[((m * ob + n) * va + f) * vb + e];
    for (std::size_t n = 0; n < ob; ++n)
      for (std::size_t f = 0; f < va; ++f)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t b = 0; b < vb; ++b)
            tab_x[((n * va + f) * oa + j) * vb + b] =
                0.5 * amp.t2ab[((j * ob + n) * va + f) * vb + b] +
                amp.t1a[j * va + f] * amp.t1b[n * vb + b];
    gemm('N', 'N', oa * vb, oa * vb, ob * va, gab_x.data(), tab_x.data(), c.W_abba.data(), 1.0,
         1.0);

    c.W_baab.assign(ob * va * ob * va, 0.0);
    std::vector<double> yb_bvv_a(ob * ob * va * va, 0.0);
    gemm('T', 'N', ob * ob, va * va, q, c.y_s[1].data(), src.bvv_a.data(), yb_bvv_a.data());
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t e = 0; e < va; ++e)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < va; ++b) {
            double val = -core.govov_ba[((m * va + b) * ob + j) * va + e] -
                         yb_bvv_a[((m * ob + j) * va + b) * va + e];
            for (std::size_t n = 0; n < oa; ++n)
              val += amp.t1a[n * va + b] * core.gooov_ba[((m * oa + n) * ob + j) * va + e];
            c.W_baab[((m * va + e) * ob + j) * va + b] = val;
          }
    std::vector<double> gba_x(ob * va * oa * vb, 0.0), tba_x(oa * vb * ob * va, 0.0);
    for (std::size_t n = 0; n < oa; ++n)
      for (std::size_t m = 0; m < ob; ++m)
        for (std::size_t e = 0; e < va; ++e)
          for (std::size_t f = 0; f < vb; ++f)
            gba_x[((m * va + e) * oa + n) * vb + f] = core.goovv_ab[((n * ob + m) * va + e) * vb + f];
    for (std::size_t n = 0; n < oa; ++n)
      for (std::size_t f = 0; f < vb; ++f)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < va; ++b)
            tba_x[((n * vb + f) * ob + j) * va + b] =
                0.5 * amp.t2ab[((n * ob + j) * va + b) * vb + f] +
                amp.t1b[j * vb + f] * amp.t1a[n * va + b];
    gemm('N', 'N', ob * va, ob * va, oa * vb, gba_x.data(), tba_x.data(), c.W_baab.data(), 1.0,
         1.0);
  }

  // Cache u intermediates for t1*t1*govov terms
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    if (!o || !v) continue;
    c.u_tg_s[s].assign(o * v * o * o, 0.0);
    gemm('N', 'T', o * v * o, o, v, sr.govov[s]->data(), sr.t1[s]->data(), c.u_tg_s[s].data());
  }
  if (oa && ob && va && vb) {
    c.t1_mjb_i.assign(oa * ob * vb * oa, 0.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t i = 0; i < oa; ++i)
          for (std::size_t b = 0; b < vb; ++b) {
            double acc = 0.0;
            for (std::size_t e = 0; e < va; ++e)
              acc += amp.t1a[i * va + e] * core.goovv_ab[((m * ob + j) * va + e) * vb + b];
            c.t1_mjb_i[((m * ob + j) * vb + b) * oa + i] = acc;
          }
    c.u2_ab.assign(oa * ob * va * ob, 0.0);
    gemm('N', 'T', oa * ob * va, ob, vb, core.goovv_ab.data(), amp.t1b.data(), c.u2_ab.data());
    c.u3_ab.assign(oa * vb * oa * ob, 0.0);
    gemm('N', 'T', oa * vb * oa, ob, vb, core.govov_ab.data(), amp.t1b.data(), c.u3_ab.data());
    c.u4_ab.assign(ob * va * ob * oa, 0.0);
    gemm('N', 'T', ob * va * ob, oa, va, core.govov_ba.data(), amp.t1a.data(), c.u4_ab.data());
  }
  return c;
}

inline void evaluate_vjp_from_cache(const ForwardCache& c, const SpinBlockedDFSource& src,
                                    const SpinBlockedFourIndexCore& core,
                                    const SpinBlockedUccAmplitudes& amp,
                                    const FixedF12Coupling* f12,
                                    const SpinBlockedUccAmplitudes& lam,
                                    SpinBlockedUccAmplitudes& bar_t,
                                    SpinBlockedFockGrads* bar_f,
                                    SpinBlockedFactorGrads* bar_b,
                                    SpinBlockedFourIndexCore* bar_core,
                                    FixedF12Cotangents* bar_f12) {
  const auto& d = src.dims;
  const std::size_t q = d.q, oa = d.oa, ob = d.ob, va = d.va, vb = d.vb;
  const auto sr = make_spin_refs(src, core, amp);

  bar_t.allocate_zero(d);
  std::vector<double>* bt1[2] = {&bar_t.t1a, &bar_t.t1b};
  std::vector<double>* bt2[2] = {&bar_t.t2aa, &bar_t.t2bb};
  const std::vector<double>* L1[2] = {&lam.t1a, &lam.t1b};
  const std::vector<double>* L2[2] = {&lam.t2aa, &lam.t2bb};

  std::vector<double>* bf_foo[2] = {bar_f ? &bar_f->foo_a : nullptr,
                                    bar_f ? &bar_f->foo_b : nullptr};
  std::vector<double>* bf_fov[2] = {bar_f ? &bar_f->fov_a : nullptr,
                                    bar_f ? &bar_f->fov_b : nullptr};
  std::vector<double>* bf_fvv[2] = {bar_f ? &bar_f->fvv_a : nullptr,
                                    bar_f ? &bar_f->fvv_b : nullptr};
  std::vector<double>* bb_bov[2] = {bar_b ? &bar_b->bov_a : nullptr,
                                    bar_b ? &bar_b->bov_b : nullptr};
  std::vector<double>* bb_bvv[2] = {bar_b ? &bar_b->bvv_a : nullptr,
                                    bar_b ? &bar_b->bvv_b : nullptr};
  std::vector<double>* bc_oovv[2] = {bar_core ? &bar_core->goovv_aa : nullptr,
                                     bar_core ? &bar_core->goovv_bb : nullptr};
  std::vector<double>* bc_oooo[2] = {bar_core ? &bar_core->goooo_aa : nullptr,
                                     bar_core ? &bar_core->goooo_bb : nullptr};
  std::vector<double>* bc_ooov[2] = {bar_core ? &bar_core->gooov_aa : nullptr,
                                     bar_core ? &bar_core->gooov_bb : nullptr};
  std::vector<double>* bc_ovov[2] = {bar_core ? &bar_core->govov_aa : nullptr,
                                     bar_core ? &bar_core->govov_bb : nullptr};

  // Intermediate adjoint accumulators
  std::vector<double> bar_Fvv[2], bar_Foo[2], bar_Fov[2], bar_Fvv_d[2], bar_Foo_d[2];
  std::vector<double> bar_tau[2], bar_tauh[2], bar_x[2], bar_y[2], bar_bt[2], bar_W_ss[2];
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    bar_Fvv[s].assign(v * v, 0.0);
    bar_Foo[s].assign(o * o, 0.0);
    bar_Fov[s].assign(o * v, 0.0);
    bar_Fvv_d[s].assign(v * v, 0.0);
    bar_Foo_d[s].assign(o * o, 0.0);
    bar_tau[s].assign(o * o * v * v, 0.0);
    bar_tauh[s].assign(o * o * v * v, 0.0);
    bar_x[s].assign(q * o * v, 0.0);
    bar_y[s].assign(q * o * o, 0.0);
    bar_bt[s].assign(q * v * v, 0.0);
    bar_W_ss[s].assign(o * v * o * v, 0.0);
  }
  std::vector<double> bar_tau_ab(oa * ob * va * vb, 0.0), bar_tauh_ab(oa * ob * va * vb, 0.0);
  std::vector<double> bar_jq(q, 0.0);
  std::vector<double> bar_W_abab(oa * va * ob * vb, 0.0), bar_W_baba(ob * vb * oa * va, 0.0);
  std::vector<double> bar_W_abba(oa * vb * oa * vb, 0.0), bar_W_baab(ob * va * ob * va, 0.0);

  // =========================================================================
  // PASS A: Residual outputs (r1, r2, optional f12) -> intermediate adjoints
  // =========================================================================

  // A1. Optional FixedF12Coupling residual terms
  if (f12) {
    if (bar_f12) {
      for (std::size_t i = 0; i < oa; ++i)
        for (std::size_t a = 0; a < va; ++a) {
          const double l1 = lam.t1a[i * va + a];
          for (std::size_t j = 0; j < oa; ++j)
            bar_f12->bar_w_vo_aaaa[((i * oa + j) * va + a) * oa + j] += l1;
          for (std::size_t j = 0; j < ob; ++j)
            bar_f12->bar_w_vo_abab[((i * ob + j) * va + a) * ob + j] += l1;
        }
      for (std::size_t i = 0; i < ob; ++i)
        for (std::size_t a = 0; a < vb; ++a) {
          const double l1 = lam.t1b[i * vb + a];
          for (std::size_t j = 0; j < ob; ++j)
            bar_f12->bar_w_vo_bbbb[((i * ob + j) * vb + a) * ob + j] += l1;
          for (std::size_t j = 0; j < oa; ++j)
            bar_f12->bar_w_vo_abba[((j * ob + i) * vb + a) * oa + j] -= l1;
        }
      add_scaled(bar_f12->bar_w_vv_aaaa, lam.t2aa);
      add_scaled(bar_f12->bar_c_f12_aaaa, lam.t2aa);
      add_scaled(bar_f12->bar_w_vv_bbbb, lam.t2bb);
      add_scaled(bar_f12->bar_c_f12_bbbb, lam.t2bb);
      add_scaled(bar_f12->bar_w_vv_abab, lam.t2ab);
      add_scaled(bar_f12->bar_c_f12_abab, lam.t2ab);
    }
    const std::vector<double>* w_vo_ss[2] = {&f12->w_vo_aaaa, &f12->w_vo_bbbb};
    std::vector<double>* bw_vo_ss[2] = {bar_f12 ? &bar_f12->bar_w_vo_aaaa : nullptr,
                                        bar_f12 ? &bar_f12->bar_w_vo_bbbb : nullptr};
    for (int s = 0; s < 2; ++s) {
      const std::size_t o = sr.o[s], v = sr.v[s];
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t j = 0; j < o; ++j)
          for (std::size_t a = 0; a < v; ++a)
            for (std::size_t b = 0; b < v; ++b) {
              const double l2 = (*L2[s])[((i * o + j) * v + a) * v + b];
              for (std::size_t k = 0; k < o; ++k) {
                const std::size_t wa = ((i * o + j) * v + a) * o + k;
                const std::size_t wb = ((i * o + j) * v + b) * o + k;
                (*bt1[s])[k * v + b] -= l2 * (*w_vo_ss[s])[wa];
                (*bt1[s])[k * v + a] += l2 * (*w_vo_ss[s])[wb];
                if (bw_vo_ss[s]) {
                  (*bw_vo_ss[s])[wa] -= l2 * (*sr.t1[s])[k * v + b];
                  (*bw_vo_ss[s])[wb] += l2 * (*sr.t1[s])[k * v + a];
                }
              }
            }
    }
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            const double l2 = lam.t2ab[((i * ob + j) * va + a) * vb + b];
            for (std::size_t k = 0; k < ob; ++k) {
              const std::size_t widx = ((i * ob + j) * va + a) * ob + k;
              bar_t.t1b[k * vb + b] -= l2 * f12->w_vo_abab[widx];
              if (bar_f12) bar_f12->bar_w_vo_abab[widx] -= l2 * amp.t1b[k * vb + b];
            }
            for (std::size_t k = 0; k < oa; ++k) {
              const std::size_t widx = ((i * ob + j) * vb + b) * oa + k;
              bar_t.t1a[k * va + a] += l2 * f12->w_vo_abba[widx];
              if (bar_f12) bar_f12->bar_w_vo_abba[widx] += l2 * amp.t1a[k * va + a];
            }
          }
  }

  // A2. r2: bare t1*govvv (xb), t1*t1*govov (tg), -t1*gooov (to)
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    if (!o || !v) continue;
    std::vector<double> bar_xb(o * v * o * v, 0.0), bar_tg(v * v * o * o, 0.0),
        bar_to(o * o * v * v, 0.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b) {
            const double l_asym = (*L2[s])[((i * o + j) * v + a) * v + b] -
                                  (*L2[s])[((i * o + j) * v + b) * v + a] -
                                  (*L2[s])[((j * o + i) * v + a) * v + b] +
                                  (*L2[s])[((j * o + i) * v + b) * v + a];
            bar_xb[((i * v + a) * o + j) * v + b] = l_asym;
            bar_tg[((a * v + b) * o + j) * o + i] = l_asym;
            bar_to[((i * o + j) * v + a) * v + b] =
                -(*L2[s])[((i * o + j) * v + a) * v + b] + (*L2[s])[((i * o + j) * v + b) * v + a];
          }
    gemm('N', 'T', q, o * v, o * v, sr.bov[s]->data(), bar_xb.data(), bar_x[s].data(), 1.0, 1.0);
    if (bb_bov[s])
      gemm('N', 'N', q, o * v, o * v, c.x_s[s].data(), bar_xb.data(), bb_bov[s]->data(), 1.0, 1.0);

    gemm('N', 'T', o, v, v * o * o, c.u_tg_s[s].data(), bar_tg.data(), bt1[s]->data(), 1.0, 1.0);
    std::vector<double> bar_u(o * v * o * o, 0.0);
    gemm('N', 'N', o, v * o * o, v, sr.t1[s]->data(), bar_tg.data(), bar_u.data());
    gemm('T', 'N', o, v, o * v * o, bar_u.data(), sr.govov[s]->data(), bt1[s]->data(), 1.0, 1.0);
    if (bc_ovov[s])
      gemm('N', 'N', o * v * o, v, o, bar_u.data(), sr.t1[s]->data(), bc_ovov[s]->data(), 1.0, 1.0);

    for (std::size_t ij = 0; ij < o * o; ++ij) {
      gemm('N', 'T', o, v, v, sr.gooov[s]->data() + ij * o * v, bar_to.data() + ij * v * v,
           bt1[s]->data(), 1.0, 1.0);
      if (bc_ooov[s])
        gemm('N', 'N', o, v, v, sr.t1[s]->data(), bar_to.data() + ij * v * v,
             bc_ooov[s]->data() + ij * o * v, 1.0, 1.0);
    }
  }
  if (oa && ob && va && vb) {
    std::vector<double> bar_xb(oa * va * ob * vb, 0.0), bar_to1(oa * ob * va * vb, 0.0),
        bar_to2(ob * oa * vb * va, 0.0), bar_tg1(va * ob * vb * oa, 0.0),
        bar_tg3(va * vb * oa * ob, 0.0), bar_tg4(vb * va * ob * oa, 0.0);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            const double l2 = lam.t2ab[((i * ob + j) * va + a) * vb + b];
            bar_xb[((i * va + a) * ob + j) * vb + b] = l2;
            bar_to1[((i * ob + j) * va + a) * vb + b] = -l2;
            bar_to2[((j * oa + i) * vb + b) * va + a] = -l2;
            bar_tg1[((a * ob + j) * vb + b) * oa + i] = -l2;
            bar_tg3[((a * vb + b) * oa + i) * ob + j] = -l2;
            bar_tg4[((b * va + a) * ob + j) * oa + i] = -l2;
          }
    gemm('N', 'T', q, oa * va, ob * vb, src.bov_b.data(), bar_xb.data(), bar_x[0].data(), 1.0, 1.0);
    gemm('N', 'N', q, ob * vb, oa * va, src.bov_a.data(), bar_xb.data(), bar_x[1].data(), 1.0, 1.0);
    if (bar_b) {
      gemm('N', 'N', q, ob * vb, oa * va, c.x_s[0].data(), bar_xb.data(), bar_b->bov_b.data(), 1.0,
           1.0);
      gemm('N', 'T', q, oa * va, ob * vb, c.x_s[1].data(), bar_xb.data(), bar_b->bov_a.data(), 1.0,
           1.0);
    }
    for (std::size_t ij = 0; ij < oa * ob; ++ij) {
      gemm('N', 'T', oa, va, vb, core.gooov_ab.data() + ij * oa * vb,
           bar_to1.data() + ij * va * vb, bar_t.t1a.data(), 1.0, 1.0);
      if (bar_core)
        gemm('N', 'N', oa, vb, va, amp.t1a.data(), bar_to1.data() + ij * va * vb,
             bar_core->gooov_ab.data() + ij * oa * vb, 1.0, 1.0);
    }
    for (std::size_t ji = 0; ji < ob * oa; ++ji) {
      gemm('N', 'T', ob, vb, va, core.gooov_ba.data() + ji * ob * va,
           bar_to2.data() + ji * vb * va, bar_t.t1b.data(), 1.0, 1.0);
      if (bar_core)
        gemm('N', 'N', ob, va, vb, amp.t1b.data(), bar_to2.data() + ji * vb * va,
             bar_core->gooov_ba.data() + ji * ob * va, 1.0, 1.0);
    }
    // -tg1
    gemm('N', 'T', oa, va, ob * vb * oa, c.t1_mjb_i.data(), bar_tg1.data(), bar_t.t1a.data(), 1.0,
         1.0);
    std::vector<double> bar_t1_mjb_i(oa * ob * vb * oa, 0.0);
    gemm('N', 'N', oa, ob * vb * oa, va, amp.t1a.data(), bar_tg1.data(), bar_t1_mjb_i.data());
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t i = 0; i < oa; ++i)
          for (std::size_t b = 0; b < vb; ++b) {
            const double v = bar_t1_mjb_i[((m * ob + j) * vb + b) * oa + i];
            for (std::size_t e = 0; e < va; ++e) {
              bar_t.t1a[i * va + e] += v * core.goovv_ab[((m * ob + j) * va + e) * vb + b];
              if (bar_core)
                bar_core->goovv_ab[((m * ob + j) * va + e) * vb + b] += v * amp.t1a[i * va + e];
            }
          }
    // -tg2
    std::vector<double> bar_u2(oa * ob * va * ob, 0.0);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            const double neg_l = -lam.t2ab[((i * ob + j) * va + a) * vb + b];
            for (std::size_t m = 0; m < ob; ++m) {
              bar_t.t1b[m * vb + b] += neg_l * c.u2_ab[((i * ob + m) * va + a) * ob + j];
              bar_u2[((i * ob + m) * va + a) * ob + j] += neg_l * amp.t1b[m * vb + b];
            }
          }
    gemm('T', 'N', ob, vb, oa * ob * va, bar_u2.data(), core.goovv_ab.data(), bar_t.t1b.data(), 1.0,
         1.0);
    if (bar_core)
      gemm('N', 'N', oa * ob * va, vb, ob, bar_u2.data(), amp.t1b.data(), bar_core->goovv_ab.data(),
           1.0, 1.0);
    // -tg3
    gemm('N', 'T', oa, va, vb * oa * ob, c.u3_ab.data(), bar_tg3.data(), bar_t.t1a.data(), 1.0,
         1.0);
    std::vector<double> bar_u3(oa * vb * oa * ob, 0.0);
    gemm('N', 'N', oa, vb * oa * ob, va, amp.t1a.data(), bar_tg3.data(), bar_u3.data());
    gemm('T', 'N', ob, vb, oa * vb * oa, bar_u3.data(), core.govov_ab.data(), bar_t.t1b.data(), 1.0,
         1.0);
    if (bar_core)
      gemm('N', 'N', oa * vb * oa, vb, ob, bar_u3.data(), amp.t1b.data(), bar_core->govov_ab.data(),
           1.0, 1.0);
    // -tg4
    gemm('N', 'T', ob, vb, va * ob * oa, c.u4_ab.data(), bar_tg4.data(), bar_t.t1b.data(), 1.0,
         1.0);
    std::vector<double> bar_u4(ob * va * ob * oa, 0.0);
    gemm('N', 'N', ob, va * ob * oa, vb, amp.t1b.data(), bar_tg4.data(), bar_u4.data());
    gemm('T', 'N', oa, va, ob * va * ob, bar_u4.data(), core.govov_ba.data(), bar_t.t1a.data(), 1.0,
         1.0);
    if (bar_core)
      gemm('N', 'N', ob * va * ob, va, oa, bar_u4.data(), amp.t1a.data(), bar_core->govov_ba.data(),
           1.0, 1.0);
  }

  // A3. Ring contractions: t2 * Wovvo -> r2
  std::vector<double> t2_ia_me_s[2];
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    t2_ia_me_s[s].assign(o * v * o * v, 0.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t m = 0; m < o; ++m)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t e = 0; e < v; ++e)
            t2_ia_me_s[s][((i * v + a) * o + m) * v + e] =
                (*sr.t2[s])[((i * o + m) * v + a) * v + e];
  }
  std::vector<double> tab_ia_me(oa * va * ob * vb, 0.0), tab_jb_me(ob * vb * oa * va, 0.0);
  for (std::size_t i = 0; i < oa; ++i)
    for (std::size_t j = 0; j < ob; ++j)
      for (std::size_t a = 0; a < va; ++a)
        for (std::size_t b = 0; b < vb; ++b) {
          const double v = amp.t2ab[((i * ob + j) * va + a) * vb + b];
          tab_ia_me[((i * va + a) * ob + j) * vb + b] = v;
          tab_jb_me[((j * vb + b) * oa + i) * va + a] = v;
        }
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s], o2 = sr.o[1 - s], v2 = sr.v[1 - s];
    if (!o || !v) continue;
    std::vector<double> bar_ring(o * v * o * v, 0.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b)
            bar_ring[((i * v + a) * o + j) * v + b] =
                (*L2[s])[((i * o + j) * v + a) * v + b] - (*L2[s])[((j * o + i) * v + a) * v + b] -
                (*L2[s])[((i * o + j) * v + b) * v + a] + (*L2[s])[((j * o + i) * v + b) * v + a];
    std::vector<double> bar_t2_me(o * v * o * v, 0.0);
    gemm('N', 'T', o * v, o * v, o * v, bar_ring.data(), c.W_ss[s].data(), bar_t2_me.data());
    gemm('T', 'N', o * v, o * v, o * v, t2_ia_me_s[s].data(), bar_ring.data(), bar_W_ss[s].data(),
         1.0, 1.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t m = 0; m < o; ++m)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t e = 0; e < v; ++e)
            (*bt2[s])[((i * o + m) * v + a) * v + e] += bar_t2_me[((i * v + a) * o + m) * v + e];
    if (o2 && v2) {
      const auto& W_cross = (s == 0) ? c.W_baba : c.W_abab;
      auto& bar_W_cross = (s == 0) ? bar_W_baba : bar_W_abab;
      const auto& tab_view = (s == 0) ? tab_ia_me : tab_jb_me;
      std::vector<double> bar_tab(o * v * o2 * v2, 0.0);
      gemm('N', 'T', o * v, o2 * v2, o * v, bar_ring.data(), W_cross.data(), bar_tab.data());
      gemm('T', 'N', o2 * v2, o * v, o * v, tab_view.data(), bar_ring.data(), bar_W_cross.data(),
           1.0, 1.0);
      if (s == 0) {
        for (std::size_t i = 0; i < oa; ++i)
          for (std::size_t m = 0; m < ob; ++m)
            for (std::size_t a = 0; a < va; ++a)
              for (std::size_t e = 0; e < vb; ++e)
                bar_t.t2ab[((i * ob + m) * va + a) * vb + e] +=
                    bar_tab[((i * va + a) * ob + m) * vb + e];
      } else {
        for (std::size_t m = 0; m < oa; ++m)
          for (std::size_t i = 0; i < ob; ++i)
            for (std::size_t e = 0; e < va; ++e)
              for (std::size_t a = 0; a < vb; ++a)
                bar_t.t2ab[((m * ob + i) * va + e) * vb + a] +=
                    bar_tab[((i * vb + a) * oa + m) * va + e];
      }
    }
  }
  if (oa && ob && va && vb) {
    std::vector<double> br12(oa * va * ob * vb, 0.0), br2(ob * vb * oa * va, 0.0),
        br3(ob * va * oa * vb, 0.0), br4(oa * vb * ob * va, 0.0);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            const double l2 = lam.t2ab[((i * ob + j) * va + a) * vb + b];
            br12[((i * va + a) * ob + j) * vb + b] = l2;
            br2[((j * vb + b) * oa + i) * va + a] = l2;
            br3[((j * va + a) * oa + i) * vb + b] = l2;
            br4[((i * vb + b) * ob + j) * va + a] = l2;
          }
    std::vector<double> btaa(oa * va * oa * va, 0.0), btab1(oa * va * ob * vb, 0.0);
    gemm('N', 'T', oa * va, oa * va, ob * vb, br12.data(), c.W_abab.data(), btaa.data());
    gemm('T', 'N', oa * va, ob * vb, oa * va, t2_ia_me_s[0].data(), br12.data(), bar_W_abab.data(),
         1.0, 1.0);
    gemm('N', 'T', oa * va, ob * vb, ob * vb, br12.data(), c.W_ss[1].data(), btab1.data());
    gemm('T', 'N', ob * vb, ob * vb, oa * va, tab_ia_me.data(), br12.data(), bar_W_ss[1].data(),
         1.0, 1.0);

    std::vector<double> btbb(ob * vb * ob * vb, 0.0), btab2(ob * vb * oa * va, 0.0);
    gemm('N', 'T', ob * vb, ob * vb, oa * va, br2.data(), c.W_baba.data(), btbb.data());
    gemm('T', 'N', ob * vb, oa * va, ob * vb, t2_ia_me_s[1].data(), br2.data(), bar_W_baba.data(),
         1.0, 1.0);
    gemm('N', 'T', ob * vb, oa * va, oa * va, br2.data(), c.W_ss[0].data(), btab2.data());
    gemm('T', 'N', oa * va, oa * va, ob * vb, tab_jb_me.data(), br2.data(), bar_W_ss[0].data(), 1.0,
         1.0);

    std::vector<double> tab_ja_me(ob * va * oa * vb, 0.0), btab3(ob * va * oa * vb, 0.0);
    for (std::size_t j = 0; j < ob; ++j)
      for (std::size_t a = 0; a < va; ++a)
        for (std::size_t m = 0; m < oa; ++m)
          for (std::size_t e = 0; e < vb; ++e)
            tab_ja_me[((j * va + a) * oa + m) * vb + e] =
                amp.t2ab[((m * ob + j) * va + a) * vb + e];
    gemm('N', 'T', ob * va, oa * vb, oa * vb, br3.data(), c.W_abba.data(), btab3.data());
    gemm('T', 'N', oa * vb, oa * vb, ob * va, tab_ja_me.data(), br3.data(), bar_W_abba.data(), 1.0,
         1.0);

    std::vector<double> tab_ib_me(oa * vb * ob * va, 0.0), btab4(oa * vb * ob * va, 0.0);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t b = 0; b < vb; ++b)
        for (std::size_t m = 0; m < ob; ++m)
          for (std::size_t e = 0; e < va; ++e)
            tab_ib_me[((i * vb + b) * ob + m) * va + e] =
                amp.t2ab[((i * ob + m) * va + e) * vb + b];
    gemm('N', 'T', oa * vb, ob * va, ob * va, br4.data(), c.W_baab.data(), btab4.data());
    gemm('T', 'N', ob * va, ob * va, oa * vb, tab_ib_me.data(), br4.data(), bar_W_baab.data(), 1.0,
         1.0);

    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t e = 0; e < va; ++e)
            bar_t.t2aa[((i * oa + m) * va + a) * va + e] += btaa[((i * va + a) * oa + m) * va + e];
    for (std::size_t j = 0; j < ob; ++j)
      for (std::size_t m = 0; m < ob; ++m)
        for (std::size_t b = 0; b < vb; ++b)
          for (std::size_t e = 0; e < vb; ++e)
            bar_t.t2bb[((j * ob + m) * vb + b) * vb + e] += btbb[((j * vb + b) * ob + m) * vb + e];
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b)
            bar_t.t2ab[((i * ob + j) * va + a) * vb + b] +=
                btab1[((i * va + a) * ob + j) * vb + b] + btab2[((j * vb + b) * oa + i) * va + a] +
                btab3[((j * va + a) * oa + i) * vb + b] + btab4[((i * vb + b) * ob + j) * va + a];
  }

  // A4. On-the-fly T1-dressed DF Virtual Ladder adjoint
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    if (!o || !v) continue;
    std::vector<double> half(v * v), bar_half(v * v);
    for (std::size_t Q = 0; Q < q; ++Q) {
      const double* bt = c.bt_s[s].data() + Q * v * v;
      double* bbt = bar_bt[s].data() + Q * v * v;
      for (std::size_t ij = 0; ij < o * o; ++ij) {
        const double* tau_ij = c.tau_s[s].data() + ij * v * v;
        const double* l2_ij = L2[s]->data() + ij * v * v;
        gemm('N', 'N', v, v, v, bt, tau_ij, half.data());
        gemm('N', 'N', v, v, v, l2_ij, bt, bar_half.data());
        gemm('T', 'N', v, v, v, l2_ij, half.data(), bbt, 1.0, 1.0);
        gemm('N', 'T', v, v, v, bar_half.data(), tau_ij, bbt, 1.0, 1.0);
        gemm('T', 'N', v, v, v, bt, bar_half.data(), bar_tau[s].data() + ij * v * v, 1.0, 1.0);
      }
    }
  }
  if (oa && ob && va && vb) {
    std::vector<double> half(va * vb), bar_half(va * vb);
    for (std::size_t Q = 0; Q < q; ++Q) {
      const double* bta = c.bt_s[0].data() + Q * va * va;
      const double* btb = c.bt_s[1].data() + Q * vb * vb;
      double* bbta = bar_bt[0].data() + Q * va * va;
      double* bbtb = bar_bt[1].data() + Q * vb * vb;
      for (std::size_t ij = 0; ij < oa * ob; ++ij) {
        const double* tau_ij = c.tau_ab.data() + ij * va * vb;
        const double* l2_ij = lam.t2ab.data() + ij * va * vb;
        gemm('N', 'N', va, vb, va, bta, tau_ij, half.data());
        gemm('N', 'N', va, vb, vb, l2_ij, btb, bar_half.data());
        gemm('T', 'N', vb, vb, va, l2_ij, half.data(), bbtb, 1.0, 1.0);
        gemm('N', 'T', va, va, vb, bar_half.data(), tau_ij, bbta, 1.0, 1.0);
        gemm('T', 'N', va, vb, va, bta, bar_half.data(), bar_tau_ab.data() + ij * va * vb, 1.0, 1.0);
      }
    }
  }

  // A5. Woooo_bare + tau_oovv * t2 into r2
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    if (!o || !v) continue;
    std::vector<double> bw(o * o * o * o, 0.0);
    gemm('N', 'T', o * o, o * o, v * v, c.tau_s[s].data(), L2[s]->data(), bw.data(), 0.5, 0.0);
    gemm('N', 'N', o * o, v * v, o * o, c.w_bare_s[s].data(), L2[s]->data(), bar_tau[s].data(),
         0.5, 1.0);
    if (bc_oooo[s]) add_scaled(*bc_oooo[s], bw);
    for (std::size_t mn = 0; mn < o * o; ++mn)
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t j = 0; j < o; ++j) {
          const double wval = bw[mn * o * o + i * o + j];
          const double* gi = sr.gooov[s]->data() + (mn * o + i) * v;
          const double* gj = sr.gooov[s]->data() + (mn * o + j) * v;
          const double* ti = sr.t1[s]->data() + i * v;
          const double* tj = sr.t1[s]->data() + j * v;
          double* bti = bt1[s]->data() + i * v;
          double* btj = bt1[s]->data() + j * v;
          for (std::size_t e = 0; e < v; ++e) {
            btj[e] += wval * gi[e];
            bti[e] -= wval * gj[e];
          }
          if (bc_ooov[s]) {
            double* bgi = bc_ooov[s]->data() + (mn * o + i) * v;
            double* bgj = bc_ooov[s]->data() + (mn * o + j) * v;
            for (std::size_t e = 0; e < v; ++e) {
              bgi[e] += wval * tj[e];
              bgj[e] -= wval * ti[e];
            }
          }
        }
    std::vector<double> bto(o * o * o * o, 0.0);
    gemm('N', 'T', o * o, o * o, v * v, sr.t2[s]->data(), L2[s]->data(), bto.data(), 0.25, 0.0);
    gemm('N', 'N', o * o, v * v, o * o, c.tau_oovv_s[s].data(), L2[s]->data(), bt2[s]->data(), 0.25,
         1.0);
    gemm('T', 'N', o * o, v * v, o * o, bto.data(), sr.goovv[s]->data(), bar_tau[s].data(), 1.0,
         1.0);
    if (bc_oovv[s])
      gemm('N', 'N', o * o, v * v, o * o, bto.data(), c.tau_s[s].data(), bc_oovv[s]->data(), 1.0,
           1.0);
  }
  if (oa && ob && va && vb) {
    std::vector<double> bw(oa * ob * oa * ob, 0.0);
    gemm('N', 'T', oa * ob, oa * ob, va * vb, c.tau_ab.data(), lam.t2ab.data(), bw.data());
    gemm('N', 'N', oa * ob, va * vb, oa * ob, c.w_bare_ab.data(), lam.t2ab.data(),
         bar_tau_ab.data(), 1.0, 1.0);
    if (bar_core) add_scaled(bar_core->goooo_ab, bw);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t i = 0; i < oa; ++i)
          for (std::size_t j = 0; j < ob; ++j) {
            const double wval = bw[((m * ob + n) * oa + i) * ob + j];
            const double* gab = core.gooov_ab.data() + ((m * ob + n) * oa + i) * vb;
            const double* gba = core.gooov_ba.data() + ((n * oa + m) * ob + j) * va;
            for (std::size_t e = 0; e < vb; ++e) bar_t.t1b[j * vb + e] += wval * gab[e];
            for (std::size_t e = 0; e < va; ++e) bar_t.t1a[i * va + e] += wval * gba[e];
            if (bar_core) {
              double* bgab = bar_core->gooov_ab.data() + ((m * ob + n) * oa + i) * vb;
              double* bgba = bar_core->gooov_ba.data() + ((n * oa + m) * ob + j) * va;
              for (std::size_t e = 0; e < vb; ++e) bgab[e] += wval * amp.t1b[j * vb + e];
              for (std::size_t e = 0; e < va; ++e) bgba[e] += wval * amp.t1a[i * va + e];
            }
          }
    std::vector<double> bg(oa * ob * oa * ob, 0.0);
    gemm('N', 'T', oa * ob, oa * ob, va * vb, amp.t2ab.data(), lam.t2ab.data(), bg.data());
    gemm('N', 'N', oa * ob, va * vb, oa * ob, c.g_ab.data(), lam.t2ab.data(), bar_t.t2ab.data(),
         1.0, 1.0);
    gemm('T', 'N', oa * ob, va * vb, oa * ob, bg.data(), core.goovv_ab.data(), bar_tau_ab.data(),
         1.0, 1.0);
    if (bar_core)
      gemm('N', 'N', oa * ob, va * vb, oa * ob, bg.data(), c.tau_ab.data(),
           bar_core->goovv_ab.data(), 1.0, 1.0);
  }

  // A6. r2 base goovv + Fvv_d / Foo_d
  if (bar_core) {
    add_scaled(bar_core->goovv_aa, lam.t2aa);
    add_scaled(bar_core->goovv_bb, lam.t2bb);
    add_scaled(bar_core->goovv_ab, lam.t2ab);
  }
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    if (!o || !v) continue;
    std::vector<double> btmp(o * o * v * v, 0.0);
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b)
            btmp[((i * o + j) * v + a) * v + b] =
                (*L2[s])[((i * o + j) * v + a) * v + b] - (*L2[s])[((i * o + j) * v + b) * v + a];
    gemm('N', 'N', o * o * v, v, v, btmp.data(), c.Fvv_d_s[s].data(), bt2[s]->data(), 1.0, 1.0);
    gemm('T', 'N', v, v, o * o * v, btmp.data(), sr.t2[s]->data(), bar_Fvv_d[s].data(), 1.0, 1.0);

    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b)
            btmp[((i * o + j) * v + a) * v + b] =
                -(*L2[s])[((i * o + j) * v + a) * v + b] + (*L2[s])[((j * o + i) * v + a) * v + b];
    for (std::size_t i = 0; i < o; ++i) {
      gemm('N', 'T', o, o, v * v, sr.t2[s]->data() + i * o * v * v, btmp.data() + i * o * v * v,
           bar_Foo_d[s].data(), 1.0, 1.0);
      gemm('N', 'N', o, v * v, o, c.Foo_d_s[s].data(), btmp.data() + i * o * v * v,
           bt2[s]->data() + i * o * v * v, 1.0, 1.0);
    }
  }
  if (oa && ob && va && vb) {
    gemm('N', 'N', oa * ob * va, vb, vb, lam.t2ab.data(), c.Fvv_d_s[1].data(), bar_t.t2ab.data(),
         1.0, 1.0);
    gemm('T', 'N', vb, vb, oa * ob * va, lam.t2ab.data(), amp.t2ab.data(), bar_Fvv_d[1].data(),
         1.0, 1.0);
    for (std::size_t ij = 0; ij < oa * ob; ++ij) {
      gemm('N', 'T', va, va, vb, lam.t2ab.data() + ij * va * vb, amp.t2ab.data() + ij * va * vb,
           bar_Fvv_d[0].data(), 1.0, 1.0);
      gemm('T', 'N', va, vb, va, c.Fvv_d_s[0].data(), lam.t2ab.data() + ij * va * vb,
           bar_t.t2ab.data() + ij * va * vb, 1.0, 1.0);
    }
    for (std::size_t i = 0; i < oa; ++i) {
      gemm('N', 'T', ob, ob, va * vb, amp.t2ab.data() + i * ob * va * vb,
           lam.t2ab.data() + i * ob * va * vb, bar_Foo_d[1].data(), -1.0, 1.0);
      gemm('N', 'N', ob, va * vb, ob, c.Foo_d_s[1].data(), lam.t2ab.data() + i * ob * va * vb,
           bar_t.t2ab.data() + i * ob * va * vb, -1.0, 1.0);
    }
    gemm('N', 'T', oa, oa, ob * va * vb, amp.t2ab.data(), lam.t2ab.data(), bar_Foo_d[0].data(),
         -1.0, 1.0);
    gemm('N', 'N', oa, ob * va * vb, oa, c.Foo_d_s[0].data(), lam.t2ab.data(), bar_t.t2ab.data(),
         -1.0, 1.0);
  }

  // A7. Singles residuals r1a, r1b
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s], o2 = sr.o[1 - s], v2 = sr.v[1 - s];
    if (bf_fov[s]) add_scaled(*bf_fov[s], *L1[s]);
    gemm('N', 'N', o, v, v, L1[s]->data(), c.Fvv_s[s].data(), bt1[s]->data(), 1.0, 1.0);
    gemm('T', 'N', v, v, o, L1[s]->data(), sr.t1[s]->data(), bar_Fvv[s].data(), 1.0, 1.0);
    gemm('N', 'T', o, o, v, sr.t1[s]->data(), L1[s]->data(), bar_Foo[s].data(), -1.0, 1.0);
    gemm('N', 'N', o, v, o, c.Foo_s[s].data(), L1[s]->data(), bt1[s]->data(), -1.0, 1.0);

    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t a = 0; a < v; ++a) {
        const double l1 = (*L1[s])[i * v + a];
        for (std::size_t m = 0; m < o; ++m)
          for (std::size_t e = 0; e < v; ++e) {
            (*bt2[s])[((i * o + m) * v + a) * v + e] += l1 * c.Fov_s[s][m * v + e];
            bar_Fov[s][m * v + e] += l1 * (*sr.t2[s])[((i * o + m) * v + a) * v + e];
            (*bt1[s])[m * v + e] -= l1 * (*sr.govov[s])[((m * v + a) * o + i) * v + e];
            if (bc_ovov[s])
              (*bc_ovov[s])[((m * v + a) * o + i) * v + e] -= l1 * (*sr.t1[s])[m * v + e];
          }
      }

    // DF govvv -> r1
    if (o && v) {
      std::vector<double> bar_z(q * o * v, 0.0);
      for (std::size_t Q = 0; Q < q; ++Q) {
        gemm('N', 'N', o, v, v, L1[s]->data(), sr.bvv[s]->data() + Q * v * v,
             bar_z.data() + Q * o * v);
        if (bb_bvv[s])
          gemm('T', 'N', v, v, o, L1[s]->data(), c.z_s[s].data() + Q * o * v,
               bb_bvv[s]->data() + Q * v * v, 1.0, 1.0);
      }
      std::vector<double> bt2_ifme(o * v * o * v, 0.0);
      gemm('T', 'N', o * v, o * v, q, bar_z.data(), sr.bov[s]->data(), bt2_ifme.data(), -1.0, 0.0);
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t m = 0; m < o; ++m)
          for (std::size_t e = 0; e < v; ++e)
            for (std::size_t f = 0; f < v; ++f)
              (*bt2[s])[((i * o + m) * v + e) * v + f] += bt2_ifme[((i * v + f) * o + m) * v + e];
      if (bb_bov[s]) {
        std::vector<double> t2_ifme(o * v * o * v, 0.0);
        for (std::size_t i = 0; i < o; ++i)
          for (std::size_t m = 0; m < o; ++m)
            for (std::size_t e = 0; e < v; ++e)
              for (std::size_t f = 0; f < v; ++f)
                t2_ifme[((i * v + f) * o + m) * v + e] = (*sr.t2[s])[((i * o + m) * v + e) * v + f];
        gemm('N', 'N', q, o * v, o * v, bar_z.data(), t2_ifme.data(), bb_bov[s]->data(), -1.0, 1.0);
      }
      if (o2 && v2) {
        std::vector<double> btab(o * v * o2 * v2, 0.0);
        gemm('T', 'N', o * v, o2 * v2, q, bar_z.data(), sr.bov[1 - s]->data(), btab.data());
        if (s == 0) {
          for (std::size_t i = 0; i < oa; ++i)
            for (std::size_t m = 0; m < ob; ++m)
              for (std::size_t e = 0; e < va; ++e)
                for (std::size_t f = 0; f < vb; ++f)
                  bar_t.t2ab[((i * ob + m) * va + e) * vb + f] +=
                      btab[((i * va + e) * ob + m) * vb + f];
        } else {
          for (std::size_t m = 0; m < oa; ++m)
            for (std::size_t i = 0; i < ob; ++i)
              for (std::size_t e = 0; e < va; ++e)
                for (std::size_t f = 0; f < vb; ++f)
                  bar_t.t2ab[((m * ob + i) * va + e) * vb + f] +=
                      btab[((i * vb + f) * oa + m) * va + e];
        }
        if (bb_bov[1 - s]) {
          const auto& tab_view = (s == 0) ? tab_ia_me : tab_jb_me;
          gemm('N', 'N', q, o2 * v2, o * v, bar_z.data(), tab_view.data(), bb_bov[1 - s]->data(),
               1.0, 1.0);
        }
      }
    }

    // -0.5 * gooov_ss * t2_ss -> r1
    for (std::size_t mn = 0; mn < o * o; ++mn) {
      gemm('T', 'N', v, v, o, L1[s]->data(), sr.gooov[s]->data() + mn * o * v,
           bt2[s]->data() + mn * v * v, -0.5, 1.0);
      if (bc_ooov[s])
        gemm('N', 'N', o, v, v, L1[s]->data(), sr.t2[s]->data() + mn * v * v,
             bc_ooov[s]->data() + mn * o * v, -0.5, 1.0);
    }
  }
  // Cross-spin r1 terms
  for (std::size_t i = 0; i < oa; ++i)
    for (std::size_t a = 0; a < va; ++a) {
      const double l1 = lam.t1a[i * va + a];
      for (std::size_t m = 0; m < ob; ++m)
        for (std::size_t e = 0; e < vb; ++e) {
          const std::size_t idx = ((i * ob + m) * va + a) * vb + e;
          bar_t.t2ab[idx] += l1 * c.Fov_s[1][m * vb + e];
          bar_Fov[1][m * vb + e] += l1 * amp.t2ab[idx];
          bar_t.t1b[m * vb + e] += l1 * core.goovv_ab[idx];
          if (bar_core) bar_core->goovv_ab[idx] += l1 * amp.t1b[m * vb + e];
        }
    }
  for (std::size_t i = 0; i < ob; ++i)
    for (std::size_t a = 0; a < vb; ++a) {
      const double l1 = lam.t1b[i * vb + a];
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t e = 0; e < va; ++e) {
          const std::size_t idx = ((m * ob + i) * va + e) * vb + a;
          bar_t.t2ab[idx] += l1 * c.Fov_s[0][m * va + e];
          bar_Fov[0][m * va + e] += l1 * amp.t2ab[idx];
          bar_t.t1a[m * va + e] += l1 * core.goovv_ab[idx];
          if (bar_core) bar_core->goovv_ab[idx] += l1 * amp.t1a[m * va + e];
        }
    }
  for (std::size_t mn = 0; mn < oa * ob; ++mn) {
    gemm('T', 'N', va, vb, oa, lam.t1a.data(), core.gooov_ab.data() + mn * oa * vb,
         bar_t.t2ab.data() + mn * va * vb, -1.0, 1.0);
    if (bar_core)
      gemm('N', 'N', oa, vb, va, lam.t1a.data(), amp.t2ab.data() + mn * va * vb,
           bar_core->gooov_ab.data() + mn * oa * vb, -1.0, 1.0);
  }
  for (std::size_t m = 0; m < ob; ++m)
    for (std::size_t n = 0; n < oa; ++n) {
      gemm('T', 'N', va, vb, ob, core.gooov_ba.data() + (m * oa + n) * ob * va, lam.t1b.data(),
           bar_t.t2ab.data() + (n * ob + m) * va * vb, -1.0, 1.0);
      if (bar_core)
        gemm('N', 'T', ob, va, vb, lam.t1b.data(), amp.t2ab.data() + (n * ob + m) * va * vb,
             bar_core->gooov_ba.data() + (m * oa + n) * ob * va, -1.0, 1.0);
    }

  // =========================================================================
  // PASS B: Pull intermediate adjoints back to bar_t and (if requested) params
  // =========================================================================

  // B1. Pull back Wovvo (all 6 blocks)
  std::vector<double> g_menf_s[2], t_nfjb_s[2], bt_nfjb_s[2];
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    g_menf_s[s].assign(o * v * o * v, 0.0);
    t_nfjb_s[s].assign(o * v * o * v, 0.0);
    bt_nfjb_s[s].assign(o * v * o * v, 0.0);
    for (std::size_t m = 0; m < o; ++m)
      for (std::size_t n = 0; n < o; ++n)
        for (std::size_t e = 0; e < v; ++e)
          for (std::size_t f = 0; f < v; ++f)
            g_menf_s[s][((m * v + e) * o + n) * v + f] =
                (*sr.goovv[s])[((m * o + n) * v + e) * v + f];
    for (std::size_t n = 0; n < o; ++n)
      for (std::size_t f = 0; f < v; ++f)
        for (std::size_t j = 0; j < o; ++j)
          for (std::size_t b = 0; b < v; ++b)
            t_nfjb_s[s][((n * v + f) * o + j) * v + b] =
                0.5 * (*sr.t2[s])[((j * o + n) * v + f) * v + b] +
                (*sr.t1[s])[j * v + f] * (*sr.t1[s])[n * v + b];
  }
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    if (!o || !v) continue;
    gemm('N', 'N', q, o * v, o * v, sr.bov[s]->data(), bar_W_ss[s].data(), bar_x[s].data(), 1.0,
         1.0);
    if (bb_bov[s])
      gemm('N', 'T', q, o * v, o * v, c.x_s[s].data(), bar_W_ss[s].data(), bb_bov[s]->data(), 1.0,
           1.0);
    std::vector<double> by_bvv(o * o * v * v, 0.0);
    for (std::size_t m = 0; m < o; ++m)
      for (std::size_t e = 0; e < v; ++e)
        for (std::size_t j = 0; j < o; ++j)
          for (std::size_t b = 0; b < v; ++b) {
            const double bw = bar_W_ss[s][((m * v + e) * o + j) * v + b];
            by_bvv[((m * o + j) * v + b) * v + e] = -bw;
            if (bc_ovov[s]) (*bc_ovov[s])[((m * v + b) * o + j) * v + e] -= bw;
            for (std::size_t n = 0; n < o; ++n) {
              (*bt1[s])[n * v + b] += bw * (*sr.gooov[s])[((m * o + n) * o + j) * v + e];
              if (bc_ooov[s])
                (*bc_ooov[s])[((m * o + n) * o + j) * v + e] += bw * (*sr.t1[s])[n * v + b];
            }
          }
    gemm('N', 'T', q, o * o, v * v, sr.bvv[s]->data(), by_bvv.data(), bar_y[s].data(), 1.0, 1.0);
    if (bb_bvv[s])
      gemm('N', 'N', q, v * v, o * o, c.y_s[s].data(), by_bvv.data(), bb_bvv[s]->data(), 1.0, 1.0);
    gemm('T', 'N', o * v, o * v, o * v, g_menf_s[s].data(), bar_W_ss[s].data(),
         bt_nfjb_s[s].data(), -1.0, 1.0);
    if (bc_oovv[s]) {
      std::vector<double> bg(o * v * o * v, 0.0);
      gemm('N', 'T', o * v, o * v, o * v, bar_W_ss[s].data(), t_nfjb_s[s].data(), bg.data(), -1.0,
           0.0);
      for (std::size_t m = 0; m < o; ++m)
        for (std::size_t n = 0; n < o; ++n)
          for (std::size_t e = 0; e < v; ++e)
            for (std::size_t f = 0; f < v; ++f)
              (*bc_oovv[s])[((m * o + n) * v + e) * v + f] += bg[((m * v + e) * o + n) * v + f];
    }
  }
  if (oa && ob && va && vb) {
    std::vector<double> gab_menf(oa * va * ob * vb, 0.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t e = 0; e < va; ++e)
          for (std::size_t f = 0; f < vb; ++f)
            gab_menf[((m * va + e) * ob + n) * vb + f] =
                core.goovv_ab[((m * ob + n) * va + e) * vb + f];

    // Cross terms in W_aaaa and W_bbbb
    std::vector<double> btab_nfb_jaa(ob * vb * oa * va, 0.0), btab_nfa_jbb(oa * va * ob * vb, 0.0);
    gemm('T', 'N', ob * vb, oa * va, oa * va, gab_menf.data(), bar_W_ss[0].data(),
         btab_nfb_jaa.data(), 0.5, 0.0);
    gemm('N', 'N', oa * va, ob * vb, ob * vb, gab_menf.data(), bar_W_ss[1].data(),
         btab_nfa_jbb.data(), 0.5, 0.0);
    if (bar_core) {
      std::vector<double> bgab(oa * va * ob * vb, 0.0);
      gemm('N', 'T', oa * va, ob * vb, oa * va, bar_W_ss[0].data(), tab_jb_me.data(), bgab.data(),
           0.5, 0.0);
      gemm('N', 'T', oa * va, ob * vb, ob * vb, tab_ia_me.data(), bar_W_ss[1].data(), bgab.data(),
           0.5, 1.0);
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t n = 0; n < ob; ++n)
          for (std::size_t e = 0; e < va; ++e)
            for (std::size_t f = 0; f < vb; ++f)
              bar_core->goovv_ab[((m * ob + n) * va + e) * vb + f] +=
                  bgab[((m * va + e) * ob + n) * vb + f];
    }

    // W_abab
    gemm('N', 'N', q, ob * vb, oa * va, src.bov_a.data(), bar_W_abab.data(), bar_x[1].data(), 1.0,
         1.0);
    if (bar_b)
      gemm('N', 'T', q, oa * va, ob * vb, c.x_s[1].data(), bar_W_abab.data(), bar_b->bov_a.data(),
           1.0, 1.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t e = 0; e < va; ++e)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < vb; ++b) {
            const double bw = bar_W_abab[((m * va + e) * ob + j) * vb + b];
            if (bar_core) bar_core->goovv_ab[((m * ob + j) * va + e) * vb + b] += bw;
            for (std::size_t n = 0; n < ob; ++n) {
              bar_t.t1b[n * vb + b] -= bw * core.gooov_ba[((n * oa + m) * ob + j) * va + e];
              if (bar_core)
                bar_core->gooov_ba[((n * oa + m) * ob + j) * va + e] -= bw * amp.t1b[n * vb + b];
            }
          }
    gemm('T', 'N', ob * vb, ob * vb, oa * va, gab_menf.data(), bar_W_abab.data(),
         bt_nfjb_s[1].data(), -1.0, 1.0);
    gemm('T', 'N', oa * va, ob * vb, oa * va, g_menf_s[0].data(), bar_W_abab.data(),
         btab_nfa_jbb.data(), 0.5, 1.0);
    if (bar_core) {
      std::vector<double> bgab(oa * va * ob * vb, 0.0), bgaa(oa * va * oa * va, 0.0);
      gemm('N', 'T', oa * va, ob * vb, ob * vb, bar_W_abab.data(), t_nfjb_s[1].data(), bgab.data(),
           -1.0, 0.0);
      gemm('N', 'T', oa * va, oa * va, ob * vb, bar_W_abab.data(), tab_ia_me.data(), bgaa.data(),
           0.5, 0.0);
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t n = 0; n < ob; ++n)
          for (std::size_t e = 0; e < va; ++e)
            for (std::size_t f = 0; f < vb; ++f)
              bar_core->goovv_ab[((m * ob + n) * va + e) * vb + f] +=
                  bgab[((m * va + e) * ob + n) * vb + f];
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t n = 0; n < oa; ++n)
          for (std::size_t e = 0; e < va; ++e)
            for (std::size_t f = 0; f < va; ++f)
              bar_core->goovv_aa[((m * oa + n) * va + e) * va + f] +=
                  bgaa[((m * va + e) * oa + n) * va + f];
    }

    // W_baba
    gemm('N', 'N', q, oa * va, ob * vb, src.bov_b.data(), bar_W_baba.data(), bar_x[0].data(), 1.0,
         1.0);
    if (bar_b)
      gemm('N', 'T', q, ob * vb, oa * va, c.x_s[0].data(), bar_W_baba.data(), bar_b->bov_b.data(),
           1.0, 1.0);
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t e = 0; e < vb; ++e)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t b = 0; b < va; ++b) {
            const double bw = bar_W_baba[((m * vb + e) * oa + j) * va + b];
            if (bar_core) bar_core->goovv_ab[((j * ob + m) * va + b) * vb + e] += bw;
            for (std::size_t n = 0; n < oa; ++n) {
              bar_t.t1a[n * va + b] -= bw * core.gooov_ab[((n * ob + m) * oa + j) * vb + e];
              if (bar_core)
                bar_core->gooov_ab[((n * ob + m) * oa + j) * vb + e] -= bw * amp.t1a[n * va + b];
            }
          }
    gemm('N', 'N', oa * va, oa * va, ob * vb, gab_menf.data(), bar_W_baba.data(),
         bt_nfjb_s[0].data(), -1.0, 1.0);
    gemm('T', 'N', ob * vb, oa * va, ob * vb, g_menf_s[1].data(), bar_W_baba.data(),
         btab_nfb_jaa.data(), 0.5, 1.0);
    if (bar_core) {
      std::vector<double> bgab(oa * va * ob * vb, 0.0), bgbb(ob * vb * ob * vb, 0.0);
      gemm('N', 'T', oa * va, ob * vb, oa * va, t_nfjb_s[0].data(), bar_W_baba.data(), bgab.data(),
           -1.0, 0.0);
      gemm('N', 'T', ob * vb, ob * vb, oa * va, bar_W_baba.data(), tab_jb_me.data(), bgbb.data(),
           0.5, 0.0);
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t n = 0; n < ob; ++n)
          for (std::size_t e = 0; e < va; ++e)
            for (std::size_t f = 0; f < vb; ++f)
              bar_core->goovv_ab[((m * ob + n) * va + e) * vb + f] +=
                  bgab[((m * va + e) * ob + n) * vb + f];
      for (std::size_t m = 0; m < ob; ++m)
        for (std::size_t n = 0; n < ob; ++n)
          for (std::size_t e = 0; e < vb; ++e)
            for (std::size_t f = 0; f < vb; ++f)
              bar_core->goovv_bb[((m * ob + n) * vb + e) * vb + f] +=
                  bgbb[((m * vb + e) * ob + n) * vb + f];
    }

    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b)
            bar_t.t2ab[((i * ob + j) * va + a) * vb + b] +=
                btab_nfa_jbb[((i * va + a) * ob + j) * vb + b] +
                btab_nfb_jaa[((j * vb + b) * oa + i) * va + a];

    // W_abba
    std::vector<double> bya_bvv_b(oa * oa * vb * vb, 0.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t e = 0; e < vb; ++e)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t b = 0; b < vb; ++b) {
            const double bw = bar_W_abba[((m * vb + e) * oa + j) * vb + b];
            bya_bvv_b[((m * oa + j) * vb + b) * vb + e] = -bw;
            if (bar_core) bar_core->govov_ab[((m * vb + b) * oa + j) * vb + e] -= bw;
            for (std::size_t n = 0; n < ob; ++n) {
              bar_t.t1b[n * vb + b] += bw * core.gooov_ab[((m * ob + n) * oa + j) * vb + e];
              if (bar_core)
                bar_core->gooov_ab[((m * ob + n) * oa + j) * vb + e] += bw * amp.t1b[n * vb + b];
            }
          }
    gemm('N', 'T', q, oa * oa, vb * vb, src.bvv_b.data(), bya_bvv_b.data(), bar_y[0].data(), 1.0,
         1.0);
    if (bar_b)
      gemm('N', 'N', q, vb * vb, oa * oa, c.y_s[0].data(), bya_bvv_b.data(), bar_b->bvv_b.data(),
           1.0, 1.0);
    std::vector<double> gab_x(oa * vb * ob * va, 0.0), btab_x(ob * va * oa * vb, 0.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t f = 0; f < va; ++f)
          for (std::size_t e = 0; e < vb; ++e)
            gab_x[((m * vb + e) * ob + n) * va + f] =
                core.goovv_ab[((m * ob + n) * va + f) * vb + e];
    gemm('T', 'N', ob * va, oa * vb, oa * vb, gab_x.data(), bar_W_abba.data(), btab_x.data());
    for (std::size_t n = 0; n < ob; ++n)
      for (std::size_t f = 0; f < va; ++f)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t b = 0; b < vb; ++b) {
            const double v = btab_x[((n * va + f) * oa + j) * vb + b];
            bar_t.t2ab[((j * ob + n) * va + f) * vb + b] += 0.5 * v;
            bar_t.t1a[j * va + f] += v * amp.t1b[n * vb + b];
            bar_t.t1b[n * vb + b] += v * amp.t1a[j * va + f];
          }
    if (bar_core) {
      std::vector<double> tab_x(ob * va * oa * vb, 0.0), bgab_x(oa * vb * ob * va, 0.0);
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t f = 0; f < va; ++f)
          for (std::size_t j = 0; j < oa; ++j)
            for (std::size_t b = 0; b < vb; ++b)
              tab_x[((n * va + f) * oa + j) * vb + b] =
                  0.5 * amp.t2ab[((j * ob + n) * va + f) * vb + b] +
                  amp.t1a[j * va + f] * amp.t1b[n * vb + b];
      gemm('N', 'T', oa * vb, ob * va, oa * vb, bar_W_abba.data(), tab_x.data(), bgab_x.data());
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t n = 0; n < ob; ++n)
          for (std::size_t f = 0; f < va; ++f)
            for (std::size_t e = 0; e < vb; ++e)
              bar_core->goovv_ab[((m * ob + n) * va + f) * vb + e] +=
                  bgab_x[((m * vb + e) * ob + n) * va + f];
    }

    // W_baab
    std::vector<double> byb_bvv_a(ob * ob * va * va, 0.0);
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t e = 0; e < va; ++e)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < va; ++b) {
            const double bw = bar_W_baab[((m * va + e) * ob + j) * va + b];
            byb_bvv_a[((m * ob + j) * va + b) * va + e] = -bw;
            if (bar_core) bar_core->govov_ba[((m * va + b) * ob + j) * va + e] -= bw;
            for (std::size_t n = 0; n < oa; ++n) {
              bar_t.t1a[n * va + b] += bw * core.gooov_ba[((m * oa + n) * ob + j) * va + e];
              if (bar_core)
                bar_core->gooov_ba[((m * oa + n) * ob + j) * va + e] += bw * amp.t1a[n * va + b];
            }
          }
    gemm('N', 'T', q, ob * ob, va * va, src.bvv_a.data(), byb_bvv_a.data(), bar_y[1].data(), 1.0,
         1.0);
    if (bar_b)
      gemm('N', 'N', q, va * va, ob * ob, c.y_s[1].data(), byb_bvv_a.data(), bar_b->bvv_a.data(),
           1.0, 1.0);
    std::vector<double> gba_x(ob * va * oa * vb, 0.0), btba_x(oa * vb * ob * va, 0.0);
    for (std::size_t n = 0; n < oa; ++n)
      for (std::size_t m = 0; m < ob; ++m)
        for (std::size_t e = 0; e < va; ++e)
          for (std::size_t f = 0; f < vb; ++f)
            gba_x[((m * va + e) * oa + n) * vb + f] =
                core.goovv_ab[((n * ob + m) * va + e) * vb + f];
    gemm('T', 'N', oa * vb, ob * va, ob * va, gba_x.data(), bar_W_baab.data(), btba_x.data());
    for (std::size_t n = 0; n < oa; ++n)
      for (std::size_t f = 0; f < vb; ++f)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < va; ++b) {
            const double v = btba_x[((n * vb + f) * ob + j) * va + b];
            bar_t.t2ab[((n * ob + j) * va + b) * vb + f] += 0.5 * v;
            bar_t.t1b[j * vb + f] += v * amp.t1a[n * va + b];
            bar_t.t1a[n * va + b] += v * amp.t1b[j * vb + f];
          }
    if (bar_core) {
      std::vector<double> tba_x(oa * vb * ob * va, 0.0), bgba_x(ob * va * oa * vb, 0.0);
      for (std::size_t n = 0; n < oa; ++n)
        for (std::size_t f = 0; f < vb; ++f)
          for (std::size_t j = 0; j < ob; ++j)
            for (std::size_t b = 0; b < va; ++b)
              tba_x[((n * vb + f) * ob + j) * va + b] =
                  0.5 * amp.t2ab[((n * ob + j) * va + b) * vb + f] +
                  amp.t1b[j * vb + f] * amp.t1a[n * va + b];
      gemm('N', 'T', ob * va, oa * vb, ob * va, bar_W_baab.data(), tba_x.data(), bgba_x.data());
      for (std::size_t n = 0; n < oa; ++n)
        for (std::size_t m = 0; m < ob; ++m)
          for (std::size_t e = 0; e < va; ++e)
            for (std::size_t f = 0; f < vb; ++f)
              bar_core->goovv_ab[((n * ob + m) * va + e) * vb + f] +=
                  bgba_x[((m * va + e) * oa + n) * vb + f];
    }
  }
  // Scatter accumulated bt_nfjb_s -> bar_t2_s, bar_t1_s
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    for (std::size_t n = 0; n < o; ++n)
      for (std::size_t f = 0; f < v; ++f)
        for (std::size_t j = 0; j < o; ++j)
          for (std::size_t b = 0; b < v; ++b) {
            const double val = bt_nfjb_s[s][((n * v + f) * o + j) * v + b];
            (*bt2[s])[((j * o + n) * v + f) * v + b] += 0.5 * val;
            (*bt1[s])[j * v + f] += val * (*sr.t1[s])[n * v + b];
            (*bt1[s])[n * v + b] += val * (*sr.t1[s])[j * v + f];
          }
  }

  // B2. Pull back Fvv_d, Foo_d -> Fvv, Foo, Fov, t1
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    add_scaled(bar_Fvv[s], bar_Fvv_d[s]);
    gemm('N', 'T', o, v, v, c.Fov_s[s].data(), bar_Fvv_d[s].data(), bt1[s]->data(), -0.5, 1.0);
    gemm('N', 'N', o, v, v, sr.t1[s]->data(), bar_Fvv_d[s].data(), bar_Fov[s].data(), -0.5, 1.0);
    add_scaled(bar_Foo[s], bar_Foo_d[s]);
    gemm('N', 'N', o, v, o, bar_Foo_d[s].data(), sr.t1[s]->data(), bar_Fov[s].data(), 0.5, 1.0);
    gemm('T', 'N', o, v, o, bar_Foo_d[s].data(), c.Fov_s[s].data(), bt1[s]->data(), 0.5, 1.0);
  }

  // B3. Pull back Fov
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    if (bf_fov[s]) add_scaled(*bf_fov[s], bar_Fov[s]);
    for (std::size_t m = 0; m < o; ++m)
      for (std::size_t e = 0; e < v; ++e) {
        const double bf = bar_Fov[s][m * v + e];
        for (std::size_t n = 0; n < o; ++n)
          for (std::size_t f = 0; f < v; ++f) {
            const std::size_t idx = ((m * o + n) * v + e) * v + f;
            (*bt1[s])[n * v + f] += bf * (*sr.goovv[s])[idx];
            if (bc_oovv[s]) (*bc_oovv[s])[idx] += bf * (*sr.t1[s])[n * v + f];
          }
      }
  }
  for (std::size_t m = 0; m < oa; ++m)
    for (std::size_t n = 0; n < ob; ++n)
      for (std::size_t e = 0; e < va; ++e)
        for (std::size_t f = 0; f < vb; ++f) {
          const std::size_t idx = ((m * ob + n) * va + e) * vb + f;
          const double g = core.goovv_ab[idx];
          const double bfa = bar_Fov[0][m * va + e];
          const double bfb = bar_Fov[1][n * vb + f];
          bar_t.t1b[n * vb + f] += bfa * g;
          bar_t.t1a[m * va + e] += bfb * g;
          if (bar_core)
            bar_core->goovv_ab[idx] += bfa * amp.t1b[n * vb + f] + bfb * amp.t1a[m * va + e];
        }

  // B4. Pull back Foo
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    if (bf_foo[s]) add_scaled(*bf_foo[s], bar_Foo[s]);
    if (bf_fov[s])
      gemm('N', 'N', o, v, o, bar_Foo[s].data(), sr.t1[s]->data(), bf_fov[s]->data(), 0.5, 1.0);
    gemm('T', 'N', o, v, o, bar_Foo[s].data(), sr.fov[s]->data(), bt1[s]->data(), 0.5, 1.0);
    for (std::size_t m = 0; m < o; ++m)
      for (std::size_t n = 0; n < o; ++n)
        for (std::size_t i = 0; i < o; ++i) {
          const double bf = bar_Foo[s][m * o + i];
          const double* g = sr.gooov[s]->data() + ((m * o + n) * o + i) * v;
          const double* t = sr.t1[s]->data() + n * v;
          double* bt = bt1[s]->data() + n * v;
          for (std::size_t e = 0; e < v; ++e) bt[e] += bf * g[e];
          if (bc_ooov[s]) {
            double* bg = bc_ooov[s]->data() + ((m * o + n) * o + i) * v;
            for (std::size_t e = 0; e < v; ++e) bg[e] += bf * t[e];
          }
        }
    gemm('T', 'N', o, o * v * v, o, bar_Foo[s].data(), sr.goovv[s]->data(), bar_tauh[s].data(), 0.5,
         1.0);
    if (bc_oovv[s])
      gemm('N', 'N', o, o * v * v, o, bar_Foo[s].data(), c.tauh_s[s].data(), bc_oovv[s]->data(),
           0.5, 1.0);
  }
  for (std::size_t m = 0; m < oa; ++m)
    for (std::size_t n = 0; n < ob; ++n)
      for (std::size_t i = 0; i < oa; ++i) {
        const double bf = bar_Foo[0][m * oa + i];
        const double* g = core.gooov_ab.data() + ((m * ob + n) * oa + i) * vb;
        for (std::size_t e = 0; e < vb; ++e) bar_t.t1b[n * vb + e] += bf * g[e];
        if (bar_core) {
          double* bg = bar_core->gooov_ab.data() + ((m * ob + n) * oa + i) * vb;
          for (std::size_t e = 0; e < vb; ++e) bg[e] += bf * amp.t1b[n * vb + e];
        }
      }
  for (std::size_t m = 0; m < ob; ++m)
    for (std::size_t n = 0; n < oa; ++n)
      for (std::size_t i = 0; i < ob; ++i) {
        const double bf = bar_Foo[1][m * ob + i];
        const double* g = core.gooov_ba.data() + ((m * oa + n) * ob + i) * va;
        for (std::size_t e = 0; e < va; ++e) bar_t.t1a[n * va + e] += bf * g[e];
        if (bar_core) {
          double* bg = bar_core->gooov_ba.data() + ((m * oa + n) * ob + i) * va;
          for (std::size_t e = 0; e < va; ++e) bg[e] += bf * amp.t1a[n * va + e];
        }
      }
  gemm('T', 'N', oa, ob * va * vb, oa, bar_Foo[0].data(), core.goovv_ab.data(), bar_tauh_ab.data(),
       1.0, 1.0);
  if (bar_core)
    gemm('N', 'N', oa, ob * va * vb, oa, bar_Foo[0].data(), c.tauh_ab.data(),
         bar_core->goovv_ab.data(), 1.0, 1.0);
  for (std::size_t n = 0; n < oa; ++n) {
    gemm('T', 'N', ob, va * vb, ob, bar_Foo[1].data(), core.goovv_ab.data() + n * ob * va * vb,
         bar_tauh_ab.data() + n * ob * va * vb, 1.0, 1.0);
    if (bar_core)
      gemm('N', 'N', ob, va * vb, ob, bar_Foo[1].data(), c.tauh_ab.data() + n * ob * va * vb,
           bar_core->goovv_ab.data() + n * ob * va * vb, 1.0, 1.0);
  }

  // B5. Pull back Fvv
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    if (bf_fvv[s]) add_scaled(*bf_fvv[s], bar_Fvv[s]);
    if (bf_fov[s])
      gemm('N', 'N', o, v, v, sr.t1[s]->data(), bar_Fvv[s].data(), bf_fov[s]->data(), -0.5, 1.0);
    gemm('N', 'T', o, v, v, sr.fov[s]->data(), bar_Fvv[s].data(), bt1[s]->data(), -0.5, 1.0);
    for (std::size_t Q = 0; Q < q; ++Q) {
      const double* bv = sr.bvv[s]->data() + Q * v * v;
      double dot = 0.0;
      for (std::size_t k = 0; k < v * v; ++k) dot += bar_Fvv[s][k] * bv[k];
      bar_jq[Q] += dot;
      if (bb_bvv[s]) {
        double* bbv = bb_bvv[s]->data() + Q * v * v;
        const double jq = c.j_q[Q];
        for (std::size_t k = 0; k < v * v; ++k) bbv[k] += jq * bar_Fvv[s][k];
      }
    }
    gemm('N', 'T', q * o, v, v, sr.bov[s]->data(), bar_Fvv[s].data(), bar_x[s].data(), -1.0, 1.0);
    if (bb_bov[s])
      gemm('N', 'N', q * o, v, v, c.x_s[s].data(), bar_Fvv[s].data(), bb_bov[s]->data(), -1.0, 1.0);
    for (std::size_t mn = 0; mn < o * o; ++mn) {
      gemm('N', 'N', v, v, v, bar_Fvv[s].data(), sr.goovv[s]->data() + mn * v * v,
           bar_tauh[s].data() + mn * v * v, -0.5, 1.0);
      if (bc_oovv[s])
        gemm('T', 'N', v, v, v, bar_Fvv[s].data(), c.tauh_s[s].data() + mn * v * v,
             bc_oovv[s]->data() + mn * v * v, -0.5, 1.0);
    }
  }
  for (std::size_t mn = 0; mn < oa * ob; ++mn) {
    gemm('N', 'N', va, vb, va, bar_Fvv[0].data(), core.goovv_ab.data() + mn * va * vb,
         bar_tauh_ab.data() + mn * va * vb, -1.0, 1.0);
    gemm('N', 'T', va, vb, vb, core.goovv_ab.data() + mn * va * vb, bar_Fvv[1].data(),
         bar_tauh_ab.data() + mn * va * vb, -1.0, 1.0);
    if (bar_core) {
      gemm('T', 'N', va, vb, va, bar_Fvv[0].data(), c.tauh_ab.data() + mn * va * vb,
           bar_core->goovv_ab.data() + mn * va * vb, -1.0, 1.0);
      gemm('N', 'N', va, vb, vb, c.tauh_ab.data() + mn * va * vb, bar_Fvv[1].data(),
           bar_core->goovv_ab.data() + mn * va * vb, -1.0, 1.0);
    }
  }

  // B6. Pull back bt, x, y, j_q -> t1 and DF factors
  for (int s = 0; s < 2; ++s) {
    const std::size_t o = sr.o[s], v = sr.v[s];
    for (std::size_t Q = 0; Q < q; ++Q) {
      const double* bov_q = sr.bov[s]->data() + Q * o * v;
      const double* bvv_q = sr.bvv[s]->data() + Q * v * v;
      const double* bbt_q = bar_bt[s].data() + Q * v * v;
      const double* bx_q = bar_x[s].data() + Q * o * v;
      const double* by_q = bar_y[s].data() + Q * o * o;
      // bt = bvv - t1^T @ bov
      gemm('N', 'T', o, v, v, bov_q, bbt_q, bt1[s]->data(), -1.0, 1.0);
      // x = t1 @ bvv^T
      gemm('N', 'N', o, v, v, bx_q, bvv_q, bt1[s]->data(), 1.0, 1.0);
      // y = bov @ t1^T
      gemm('T', 'N', o, v, o, by_q, bov_q, bt1[s]->data(), 1.0, 1.0);
      // j_q = dot(bov, t1)
      const double bjq = bar_jq[Q];
      double* bt1_p = bt1[s]->data();
      for (std::size_t k = 0; k < o * v; ++k) bt1_p[k] += bjq * bov_q[k];
      if (bar_b) {
        double* bbvv_q = bb_bvv[s]->data() + Q * v * v;
        double* bbov_q = bb_bov[s]->data() + Q * o * v;
        for (std::size_t k = 0; k < v * v; ++k) bbvv_q[k] += bbt_q[k];
        gemm('N', 'N', o, v, v, sr.t1[s]->data(), bbt_q, bbov_q, -1.0, 1.0);
        gemm('T', 'N', v, v, o, bx_q, sr.t1[s]->data(), bbvv_q, 1.0, 1.0);
        gemm('N', 'N', o, v, o, by_q, sr.t1[s]->data(), bbov_q, 1.0, 1.0);
        const double* t1_p = sr.t1[s]->data();
        for (std::size_t k = 0; k < o * v; ++k) bbov_q[k] += bjq * t1_p[k];
      }
    }
  }

  // B7. Pull back tau (factor 1.0) and tauh (factor 0.5) -> t2, t1
  auto pullback_tau = [&](const std::vector<double> bt_s[2], const std::vector<double>& bt_ab,
                          double factor) {
    for (int s = 0; s < 2; ++s) {
      add_scaled(*bt2[s], bt_s[s]);
      const std::size_t o = sr.o[s], v = sr.v[s];
      const auto& t1 = *sr.t1[s];
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t a = 0; a < v; ++a) {
          double acc = 0.0;
          for (std::size_t j = 0; j < o; ++j)
            for (std::size_t b = 0; b < v; ++b)
              acc += (bt_s[s][((i * o + j) * v + a) * v + b] +
                      bt_s[s][((j * o + i) * v + b) * v + a] -
                      bt_s[s][((i * o + j) * v + b) * v + a] -
                      bt_s[s][((j * o + i) * v + a) * v + b]) *
                     t1[j * v + b];
          (*bt1[s])[i * v + a] += factor * acc;
        }
    }
    add_scaled(bar_t.t2ab, bt_ab);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            const double val = factor * bt_ab[((i * ob + j) * va + a) * vb + b];
            bar_t.t1a[i * va + a] += val * amp.t1b[j * vb + b];
            bar_t.t1b[j * vb + b] += val * amp.t1a[i * va + a];
          }
  };
  pullback_tau(bar_tau, bar_tau_ab, 1.0);
  pullback_tau(bar_tauh, bar_tauh_ab, 0.5);

  antisymmetrize_t2(oa, va, bar_t.t2aa);
  antisymmetrize_t2(ob, vb, bar_t.t2bb);
}

}  // namespace generativeqc::cc::v2_detail::lambda_vjp
