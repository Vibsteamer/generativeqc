#include "cc/df_ucc_v2_internal.hpp"
#include "cc/df_ucc_v2_lambda_vjp.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>

namespace generativeqc::cc::v2_detail {
namespace {

void flatten_amplitudes(const SpinBlockedUccAmplitudes& a, std::vector<double>& out) {
  out.clear();
  out.reserve(a.t1a.size() + a.t1b.size() + a.t2aa.size() + a.t2ab.size() + a.t2bb.size());
  out.insert(out.end(), a.t1a.begin(), a.t1a.end());
  out.insert(out.end(), a.t1b.begin(), a.t1b.end());
  out.insert(out.end(), a.t2aa.begin(), a.t2aa.end());
  out.insert(out.end(), a.t2ab.begin(), a.t2ab.end());
  out.insert(out.end(), a.t2bb.begin(), a.t2bb.end());
}

void unflatten_amplitudes(const std::vector<double>& flat, const SpinBlockedDims& d,
                          SpinBlockedUccAmplitudes& a) {
  a.allocate_zero(d);
  std::size_t off = 0;
  auto take = [&](std::vector<double>& dst) {
    std::copy_n(flat.data() + off, dst.size(), dst.data());
    off += dst.size();
  };
  take(a.t1a);
  take(a.t1b);
  take(a.t2aa);
  take(a.t2ab);
  take(a.t2bb);
}

bool solve_linear_system(std::vector<double> a, std::vector<double> b, std::size_t n,
                         std::vector<double>& x) {
  x.assign(n, 0.0);
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot = col;
    double best = std::abs(a[col * n + col]);
    for (std::size_t row = col + 1; row < n; ++row) {
      const double raw = a[row * n + col];
      if (!std::isfinite(raw)) return false;
      const double v = std::abs(raw);
      if (v > best) {
        best = v;
        pivot = row;
      }
    }
    if (!std::isfinite(best) || !(best > 1e-18)) return false;
    if (pivot != col) {
      for (std::size_t k = col; k < n; ++k) std::swap(a[col * n + k], a[pivot * n + k]);
      std::swap(b[col], b[pivot]);
    }
    const double diag = a[col * n + col];
    for (std::size_t row = col + 1; row < n; ++row) {
      const double factor = a[row * n + col] / diag;
      a[row * n + col] = 0.0;
      for (std::size_t k = col + 1; k < n; ++k) a[row * n + k] -= factor * a[col * n + k];
      b[row] -= factor * b[col];
    }
  }
  for (std::size_t ii = n; ii-- > 0;) {
    double sum = b[ii];
    for (std::size_t k = ii + 1; k < n; ++k) sum -= a[ii * n + k] * x[k];
    x[ii] = sum / a[ii * n + ii];
    if (!std::isfinite(x[ii])) return false;
  }
  return true;
}

class HostDiis {
 public:
  explicit HostDiis(int max_space) : max_space_(std::max(2, max_space)) {}

  void push(const std::vector<double>& vec, const std::vector<double>& err) {
    if (static_cast<int>(vecs_.size()) >= max_space_) {
      vecs_.pop_front();
      errs_.pop_front();
    }
    vecs_.push_back(vec);
    errs_.push_back(err);
  }

  bool extrapolate(std::vector<double>& out) const {
    const std::size_t m = vecs_.size();
    if (m < 2) return false;
    const std::size_t dim = m + 1;
    std::vector<double> mat(dim * dim, 0.0);
    std::vector<double> rhs(dim, 0.0);
    double scale = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j <= i; ++j) {
        double dot = 0.0;
        const auto& ei = errs_[i];
        const auto& ej = errs_[j];
        for (std::size_t k = 0; k < ei.size(); ++k) dot += ei[k] * ej[k];
        if (!std::isfinite(dot)) return false;
        mat[i * dim + j] = dot;
        mat[j * dim + i] = dot;
        if (i == j) scale = std::max(scale, std::abs(dot));
      }
      mat[i * dim + m] = -1.0;
      mat[m * dim + i] = -1.0;
    }
    rhs[m] = -1.0;
    if (!std::isfinite(scale) || !(scale > 0.0)) return false;
    const double inv_scale = 1.0 / scale;
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < m; ++j) mat[i * dim + j] *= inv_scale;
      mat[i * dim + i] += 1e-14;
    }
    std::vector<double> coeffs;
    if (!solve_linear_system(std::move(mat), std::move(rhs), dim, coeffs)) return false;
    out.assign(vecs_[0].size(), 0.0);
    for (std::size_t i = 0; i < m; ++i) {
      const double c = coeffs[i];
      const auto& vi = vecs_[i];
      for (std::size_t k = 0; k < out.size(); ++k) out[k] += c * vi[k];
    }
    return true;
  }

 private:
  int max_space_{8};
  std::deque<std::vector<double>> vecs_;
  std::deque<std::vector<double>> errs_;
};

double max_abs_amp(const SpinBlockedUccAmplitudes& r) {
  double m = 0.0;
  for (const auto* v : {&r.t1a, &r.t1b, &r.t2aa, &r.t2ab, &r.t2bb}) {
    for (double x : *v) {
      if (!std::isfinite(x)) return std::numeric_limits<double>::quiet_NaN();
      const double ax = std::abs(x);
      if (ax > m) m = ax;
    }
  }
  return m;
}

void allocate_f12_cotangents_zero(const FixedF12Coupling& f12, FixedF12Cotangents& out) {
  out.bar_fixed_geminal_energy = 0.0;
  out.bar_w_vo_aaaa.assign(f12.w_vo_aaaa.size(), 0.0);
  out.bar_w_vo_bbbb.assign(f12.w_vo_bbbb.size(), 0.0);
  out.bar_w_vo_abab.assign(f12.w_vo_abab.size(), 0.0);
  out.bar_w_vo_abba.assign(f12.w_vo_abba.size(), 0.0);
  out.bar_w_vv_aaaa.assign(f12.w_vv_aaaa.size(), 0.0);
  out.bar_w_vv_abab.assign(f12.w_vv_abab.size(), 0.0);
  out.bar_w_vv_bbbb.assign(f12.w_vv_bbbb.size(), 0.0);
  out.bar_c_f12_aaaa.assign(f12.c_f12_aaaa.size(), 0.0);
  out.bar_c_f12_abab.assign(f12.c_f12_abab.size(), 0.0);
  out.bar_c_f12_bbbb.assign(f12.c_f12_bbbb.size(), 0.0);
}

void accumulate_ccsd_energy_grad(const SpinBlockedDFSource& src,
                                 const SpinBlockedFourIndexCore& core,
                                 const SpinBlockedUccAmplitudes& amp, const FixedF12Coupling* f12,
                                 SpinBlockedUccAmplitudes* bar_t, SpinBlockedFockGrads* bar_f,
                                 SpinBlockedFourIndexCore* bar_core, FixedF12Cotangents* bar_f12) {
  const auto& d = src.dims;
  const std::size_t oa = d.oa, ob = d.ob, va = d.va, vb = d.vb;

  if (bar_t) {
    lambda_vjp::add_scaled(bar_t->t1a, src.fov_a);
    lambda_vjp::add_scaled(bar_t->t1b, src.fov_b);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < oa; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < va; ++b) {
            const std::size_t idx = ((i * oa + j) * va + a) * va + b;
            const double g = core.goovv_aa[idx];
            bar_t->t2aa[idx] += 0.25 * g;
            bar_t->t1a[i * va + a] += 0.5 * g * amp.t1a[j * va + b];
            bar_t->t1a[j * va + b] += 0.5 * g * amp.t1a[i * va + a];
          }
    for (std::size_t i = 0; i < ob; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < vb; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            const std::size_t idx = ((i * ob + j) * vb + a) * vb + b;
            const double g = core.goovv_bb[idx];
            bar_t->t2bb[idx] += 0.25 * g;
            bar_t->t1b[i * vb + a] += 0.5 * g * amp.t1b[j * vb + b];
            bar_t->t1b[j * vb + b] += 0.5 * g * amp.t1b[i * vb + a];
          }
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            const std::size_t idx = ((i * ob + j) * va + a) * vb + b;
            const double g = core.goovv_ab[idx];
            bar_t->t2ab[idx] += g;
            bar_t->t1a[i * va + a] += g * amp.t1b[j * vb + b];
            bar_t->t1b[j * vb + b] += g * amp.t1a[i * va + a];
          }
  }
  if (bar_f) {
    lambda_vjp::add_scaled(bar_f->fov_a, amp.t1a);
    lambda_vjp::add_scaled(bar_f->fov_b, amp.t1b);
  }
  if (bar_core) {
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < oa; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < va; ++b) {
            const std::size_t idx = ((i * oa + j) * va + a) * va + b;
            bar_core->goovv_aa[idx] +=
                0.25 * amp.t2aa[idx] + 0.5 * amp.t1a[i * va + a] * amp.t1a[j * va + b];
          }
    for (std::size_t i = 0; i < ob; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < vb; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            const std::size_t idx = ((i * ob + j) * vb + a) * vb + b;
            bar_core->goovv_bb[idx] +=
                0.25 * amp.t2bb[idx] + 0.5 * amp.t1b[i * vb + a] * amp.t1b[j * vb + b];
          }
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            const std::size_t idx = ((i * ob + j) * va + a) * vb + b;
            bar_core->goovv_ab[idx] += amp.t2ab[idx] + amp.t1a[i * va + a] * amp.t1b[j * vb + b];
          }
  }
  if (f12) {
    if (bar_f12) bar_f12->bar_fixed_geminal_energy += 1.0;
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t a = 0; a < va; ++a) {
        double r1 = 0.0;
        for (std::size_t j = 0; j < oa; ++j) {
          const std::size_t widx = ((i * oa + j) * va + a) * oa + j;
          r1 += f12->w_vo_aaaa[widx];
          if (bar_f12) bar_f12->bar_w_vo_aaaa[widx] += amp.t1a[i * va + a];
        }
        for (std::size_t j = 0; j < ob; ++j) {
          const std::size_t widx = ((i * ob + j) * va + a) * ob + j;
          r1 += f12->w_vo_abab[widx];
          if (bar_f12) bar_f12->bar_w_vo_abab[widx] += amp.t1a[i * va + a];
        }
        if (bar_t) bar_t->t1a[i * va + a] += r1;
      }
    for (std::size_t i = 0; i < ob; ++i)
      for (std::size_t a = 0; a < vb; ++a) {
        double r1 = 0.0;
        for (std::size_t j = 0; j < ob; ++j) {
          const std::size_t widx = ((i * ob + j) * vb + a) * ob + j;
          r1 += f12->w_vo_bbbb[widx];
          if (bar_f12) bar_f12->bar_w_vo_bbbb[widx] += amp.t1b[i * vb + a];
        }
        for (std::size_t j = 0; j < oa; ++j) {
          const std::size_t widx = ((j * ob + i) * vb + a) * oa + j;
          r1 -= f12->w_vo_abba[widx];
          if (bar_f12) bar_f12->bar_w_vo_abba[widx] -= amp.t1b[i * vb + a];
        }
        if (bar_t) bar_t->t1b[i * vb + a] += r1;
      }
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < oa; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < va; ++b) {
            const std::size_t idx = ((i * oa + j) * va + a) * va + b;
            if (bar_t) {
              bar_t->t2aa[idx] += 0.25 * (f12->c_f12_aaaa[idx] + f12->w_vv_aaaa[idx]);
              bar_t->t1a[i * va + a] += 0.5 * f12->w_vv_aaaa[idx] * amp.t1a[j * va + b];
              bar_t->t1a[j * va + b] += 0.5 * f12->w_vv_aaaa[idx] * amp.t1a[i * va + a];
            }
            if (bar_f12) {
              bar_f12->bar_c_f12_aaaa[idx] += 0.25 * amp.t2aa[idx];
              bar_f12->bar_w_vv_aaaa[idx] +=
                  0.25 * amp.t2aa[idx] + 0.5 * amp.t1a[i * va + a] * amp.t1a[j * va + b];
            }
          }
    for (std::size_t i = 0; i < ob; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < vb; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            const std::size_t idx = ((i * ob + j) * vb + a) * vb + b;
            if (bar_t) {
              bar_t->t2bb[idx] += 0.25 * (f12->c_f12_bbbb[idx] + f12->w_vv_bbbb[idx]);
              bar_t->t1b[i * vb + a] += 0.5 * f12->w_vv_bbbb[idx] * amp.t1b[j * vb + b];
              bar_t->t1b[j * vb + b] += 0.5 * f12->w_vv_bbbb[idx] * amp.t1b[i * vb + a];
            }
            if (bar_f12) {
              bar_f12->bar_c_f12_bbbb[idx] += 0.25 * amp.t2bb[idx];
              bar_f12->bar_w_vv_bbbb[idx] +=
                  0.25 * amp.t2bb[idx] + 0.5 * amp.t1b[i * vb + a] * amp.t1b[j * vb + b];
            }
          }
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            const std::size_t idx = ((i * ob + j) * va + a) * vb + b;
            if (bar_t) {
              bar_t->t2ab[idx] += f12->c_f12_abab[idx] + f12->w_vv_abab[idx];
              bar_t->t1a[i * va + a] += f12->w_vv_abab[idx] * amp.t1b[j * vb + b];
              bar_t->t1b[j * vb + b] += f12->w_vv_abab[idx] * amp.t1a[i * va + a];
            }
            if (bar_f12) {
              bar_f12->bar_c_f12_abab[idx] += amp.t2ab[idx];
              bar_f12->bar_w_vv_abab[idx] +=
                  amp.t2ab[idx] + amp.t1a[i * va + a] * amp.t1b[j * vb + b];
            }
          }
  }
}

void apply_preconditioner(const SpinBlockedDFSource& src, const SpinBlockedUccAmplitudes& res,
                          double thresh, SpinBlockedUccAmplitudes& step,
                          SpinBlockedUccAmplitudes& vec) {
  const auto& d = src.dims;
  step.allocate_zero(d);
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t a = 0; a < d.va; ++a) {
      const double den = src.eps_va[a] - src.eps_oa[i];
      if (std::abs(den) < thresh) throw std::runtime_error("DF-ULambda v2 small alpha 1-body gap");
      const std::size_t idx = i * d.va + a;
      step.t1a[idx] = -res.t1a[idx] / den;
      vec.t1a[idx] += step.t1a[idx];
    }
  for (std::size_t i = 0; i < d.ob; ++i)
    for (std::size_t a = 0; a < d.vb; ++a) {
      const double den = src.eps_vb[a] - src.eps_ob[i];
      if (std::abs(den) < thresh) throw std::runtime_error("DF-ULambda v2 small beta 1-body gap");
      const std::size_t idx = i * d.vb + a;
      step.t1b[idx] = -res.t1b[idx] / den;
      vec.t1b[idx] += step.t1b[idx];
    }
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t j = 0; j < d.oa; ++j)
      for (std::size_t a = 0; a < d.va; ++a)
        for (std::size_t b = 0; b < d.va; ++b) {
          const double den = src.eps_va[a] + src.eps_va[b] - src.eps_oa[i] - src.eps_oa[j];
          if (std::abs(den) < thresh) throw std::runtime_error("DF-ULambda v2 small aa gap");
          const std::size_t idx = ((i * d.oa + j) * d.va + a) * d.va + b;
          step.t2aa[idx] = -res.t2aa[idx] / den;
          vec.t2aa[idx] += step.t2aa[idx];
        }
  for (std::size_t i = 0; i < d.ob; ++i)
    for (std::size_t j = 0; j < d.ob; ++j)
      for (std::size_t a = 0; a < d.vb; ++a)
        for (std::size_t b = 0; b < d.vb; ++b) {
          const double den = src.eps_vb[a] + src.eps_vb[b] - src.eps_ob[i] - src.eps_ob[j];
          if (std::abs(den) < thresh) throw std::runtime_error("DF-ULambda v2 small bb gap");
          const std::size_t idx = ((i * d.ob + j) * d.vb + a) * d.vb + b;
          step.t2bb[idx] = -res.t2bb[idx] / den;
          vec.t2bb[idx] += step.t2bb[idx];
        }
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t j = 0; j < d.ob; ++j)
      for (std::size_t a = 0; a < d.va; ++a)
        for (std::size_t b = 0; b < d.vb; ++b) {
          const double den = src.eps_va[a] + src.eps_vb[b] - src.eps_oa[i] - src.eps_ob[j];
          if (std::abs(den) < thresh) throw std::runtime_error("DF-ULambda v2 small ab gap");
          const std::size_t idx = ((i * d.ob + j) * d.va + a) * d.vb + b;
          step.t2ab[idx] = -res.t2ab[idx] / den;
          vec.t2ab[idx] += step.t2ab[idx];
        }
}

}  // namespace

void evaluate_uccsd_residual_vjp_cpu(const SpinBlockedDFSource& src,
                                     const SpinBlockedFourIndexCore& core,
                                     const SpinBlockedUccAmplitudes& amp,
                                     const FixedF12Coupling* f12,
                                     const SpinBlockedUccAmplitudes& lambda,
                                     SpinBlockedUccAmplitudes& bar_t,
                                     SpinBlockedFockGrads* bar_f,
                                     SpinBlockedFactorGrads* bar_b,
                                     SpinBlockedFourIndexCore* bar_core,
                                     FixedF12Cotangents* bar_f12) {
  const auto& d = src.dims;
  if (bar_f && bar_f->foo_a.size() != d.oa * d.oa) bar_f->allocate_zero(d);
  if (bar_b && bar_b->bov_a.size() != d.q * d.oa * d.va) bar_b->allocate_zero(d);
  if (bar_core && bar_core->goovv_aa.size() != d.oa * d.oa * d.va * d.va)
    bar_core->allocate_zero(d);
  if (f12 && bar_f12 && bar_f12->bar_w_vv_aaaa.size() != f12->w_vv_aaaa.size())
    allocate_f12_cotangents_zero(*f12, *bar_f12);
  const auto cache = lambda_vjp::build_forward_cache(src, core, amp);
  lambda_vjp::evaluate_vjp_from_cache(cache, src, core, amp, f12, lambda, bar_t, bar_f, bar_b,
                                      bar_core, bar_f12);
}

SpinBlockedLambdaSolveResult solve_df_ulambda_cpu(const SpinBlockedDFSource& src,
                                                  const SpinBlockedFourIndexCore& core,
                                                  const SpinBlockedUccAmplitudes& amp,
                                                  const SpinBlockedTriplesResult* triples,
                                                  const DFUccV2Options& options) {
  const auto& d = src.dims;
  SpinBlockedLambdaSolveResult out;
  out.lambda.allocate_zero(d);
  out.bar_b.allocate_zero(d);
  out.bar_f.allocate_zero(d);
  if (options.f12) allocate_f12_cotangents_zero(*options.f12, out.f12_cotangents);

  SpinBlockedUccAmplitudes rhs;
  rhs.allocate_zero(d);
  if (triples) {
    lambda_vjp::add_scaled(rhs.t1a, triples->bar_t.t1a);
    lambda_vjp::add_scaled(rhs.t1b, triples->bar_t.t1b);
    lambda_vjp::add_scaled(rhs.t2aa, triples->bar_t.t2aa);
    lambda_vjp::add_scaled(rhs.t2ab, triples->bar_t.t2ab);
    lambda_vjp::add_scaled(rhs.t2bb, triples->bar_t.t2bb);
  }
  if (!options.energy_source_only) {
    accumulate_ccsd_energy_grad(src, core, amp, options.f12, &rhs, nullptr, nullptr, nullptr);
  }
  lambda_vjp::antisymmetrize_t2(d.oa, d.va, rhs.t2aa);
  lambda_vjp::antisymmetrize_t2(d.ob, d.vb, rhs.t2bb);

  const auto cache = lambda_vjp::build_forward_cache(src, core, amp);

  const double rhs_max = max_abs_amp(rhs);
  if (!std::isfinite(rhs_max))
    throw std::runtime_error("DF-ULambda v2 CPU RHS became nonfinite");
  if (rhs_max > 0.0) {
    SpinBlockedUccAmplitudes init_step;
    apply_preconditioner(src, rhs, options.denominator_threshold, init_step, out.lambda);
    lambda_vjp::antisymmetrize_t2(d.oa, d.va, out.lambda.t2aa);
    lambda_vjp::antisymmetrize_t2(d.ob, d.vb, out.lambda.t2bb);

    HostDiis diis(options.diis_space);
    bool converged = false;
    SpinBlockedUccAmplitudes res, step;
    for (int iter = 1; iter <= options.max_lambda_iterations; ++iter) {
      lambda_vjp::evaluate_vjp_from_cache(cache, src, core, amp, options.f12, out.lambda, res,
                                          nullptr, nullptr, nullptr, nullptr);
      lambda_vjp::add_scaled(res.t1a, rhs.t1a);
      lambda_vjp::add_scaled(res.t1b, rhs.t1b);
      lambda_vjp::add_scaled(res.t2aa, rhs.t2aa);
      lambda_vjp::add_scaled(res.t2ab, rhs.t2ab);
      lambda_vjp::add_scaled(res.t2bb, rhs.t2bb);
      const double rmax = max_abs_amp(res);
      if (!std::isfinite(rmax))
        throw std::runtime_error("DF-ULambda v2 CPU residual became nonfinite");
      out.iterations = iter;
      if (rmax < options.lambda_tolerance) {
        converged = true;
        break;
      }
      apply_preconditioner(src, res, options.denominator_threshold, step, out.lambda);
      std::vector<double> flat_l, flat_err, extrap;
      flatten_amplitudes(out.lambda, flat_l);
      flatten_amplitudes(step, flat_err);
      diis.push(flat_l, flat_err);
      if (diis.extrapolate(extrap)) {
        unflatten_amplitudes(extrap, d, out.lambda);
        lambda_vjp::antisymmetrize_t2(d.oa, d.va, out.lambda.t2aa);
        lambda_vjp::antisymmetrize_t2(d.ob, d.vb, out.lambda.t2bb);
      }
    }
    if (!converged) throw std::runtime_error("DF-ULambda v2 CPU solver did not converge");
  }

  // Final parameter VJP pass: direct energy derivatives + <L, dR/dP>
  SpinBlockedFourIndexCore bar_core;
  bar_core.allocate_zero(d);
  if (triples) {
    lambda_vjp::add_scaled(out.bar_f.foo_a, triples->bar_f.foo_a);
    lambda_vjp::add_scaled(out.bar_f.fov_a, triples->bar_f.fov_a);
    lambda_vjp::add_scaled(out.bar_f.fvv_a, triples->bar_f.fvv_a);
    lambda_vjp::add_scaled(out.bar_f.foo_b, triples->bar_f.foo_b);
    lambda_vjp::add_scaled(out.bar_f.fov_b, triples->bar_f.fov_b);
    lambda_vjp::add_scaled(out.bar_f.fvv_b, triples->bar_f.fvv_b);
    lambda_vjp::add_scaled(out.bar_b.boo_a, triples->bar_b.boo_a);
    lambda_vjp::add_scaled(out.bar_b.bov_a, triples->bar_b.bov_a);
    lambda_vjp::add_scaled(out.bar_b.bvv_a, triples->bar_b.bvv_a);
    lambda_vjp::add_scaled(out.bar_b.boo_b, triples->bar_b.boo_b);
    lambda_vjp::add_scaled(out.bar_b.bov_b, triples->bar_b.bov_b);
    lambda_vjp::add_scaled(out.bar_b.bvv_b, triples->bar_b.bvv_b);
    lambda_vjp::add_scaled(bar_core.goovv_aa, triples->bar_core.goovv_aa);
    lambda_vjp::add_scaled(bar_core.goovv_ab, triples->bar_core.goovv_ab);
    lambda_vjp::add_scaled(bar_core.goovv_bb, triples->bar_core.goovv_bb);
    lambda_vjp::add_scaled(bar_core.gooov_aa, triples->bar_core.gooov_aa);
    lambda_vjp::add_scaled(bar_core.gooov_bb, triples->bar_core.gooov_bb);
    lambda_vjp::add_scaled(bar_core.gooov_ab, triples->bar_core.gooov_ab);
    lambda_vjp::add_scaled(bar_core.gooov_ba, triples->bar_core.gooov_ba);
  }
  if (!options.energy_source_only) {
    accumulate_ccsd_energy_grad(src, core, amp, options.f12, nullptr, &out.bar_f, &bar_core,
                                options.f12 ? &out.f12_cotangents : nullptr);
  }
  SpinBlockedUccAmplitudes final_bar_t;
  lambda_vjp::evaluate_vjp_from_cache(cache, src, core, amp, options.f12, out.lambda, final_bar_t,
                                      &out.bar_f, &out.bar_b, &bar_core,
                                      options.f12 ? &out.f12_cotangents : nullptr);
  pullback_four_index_core_cpu(src, bar_core, out.bar_b);
  return out;
}

}  // namespace generativeqc::cc::v2_detail
