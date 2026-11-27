#include "cc/df_ucc_v2_internal.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>

#include "tensor/cpu_linalg.hpp"

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

void build_tau(const SpinBlockedDims& d, const SpinBlockedUccAmplitudes& amp, double factor,
               std::vector<double>& tau_aa, std::vector<double>& tau_ab,
               std::vector<double>& tau_bb) {
  tau_aa = amp.t2aa;
  tau_ab = amp.t2ab;
  tau_bb = amp.t2bb;
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t j = 0; j < d.oa; ++j)
      for (std::size_t a = 0; a < d.va; ++a)
        for (std::size_t b = 0; b < d.va; ++b)
          tau_aa[((i * d.oa + j) * d.va + a) * d.va + b] +=
              factor * (amp.t1a[i * d.va + a] * amp.t1a[j * d.va + b] -
                        amp.t1a[i * d.va + b] * amp.t1a[j * d.va + a]);
  for (std::size_t i = 0; i < d.ob; ++i)
    for (std::size_t j = 0; j < d.ob; ++j)
      for (std::size_t a = 0; a < d.vb; ++a)
        for (std::size_t b = 0; b < d.vb; ++b)
          tau_bb[((i * d.ob + j) * d.vb + a) * d.vb + b] +=
              factor * (amp.t1b[i * d.vb + a] * amp.t1b[j * d.vb + b] -
                        amp.t1b[i * d.vb + b] * amp.t1b[j * d.vb + a]);
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t j = 0; j < d.ob; ++j)
      for (std::size_t a = 0; a < d.va; ++a)
        for (std::size_t b = 0; b < d.vb; ++b)
          tau_ab[((i * d.ob + j) * d.va + a) * d.vb + b] +=
              factor * amp.t1a[i * d.va + a] * amp.t1b[j * d.vb + b];
}

double max_abs_residual(const SpinBlockedUccAmplitudes& r) {
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

}  // namespace

double compute_uccsd_energy_cpu(const SpinBlockedDFSource& src,
                                const SpinBlockedFourIndexCore& core,
                                const SpinBlockedUccAmplitudes& amp,
                                const FixedF12Coupling* f12) {
  const auto& d = src.dims;
  double e = 0.0;
  for (std::size_t k = 0; k < amp.t1a.size(); ++k) e += src.fov_a[k] * amp.t1a[k];
  for (std::size_t k = 0; k < amp.t1b.size(); ++k) e += src.fov_b[k] * amp.t1b[k];

  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t j = 0; j < d.oa; ++j)
      for (std::size_t a = 0; a < d.va; ++a)
        for (std::size_t b = 0; b < d.va; ++b) {
          const std::size_t idx = ((i * d.oa + j) * d.va + a) * d.va + b;
          const double g = core.goovv_aa[idx];
          e += 0.25 * g * amp.t2aa[idx] +
               0.5 * g * amp.t1a[i * d.va + a] * amp.t1a[j * d.va + b];
        }
  for (std::size_t i = 0; i < d.ob; ++i)
    for (std::size_t j = 0; j < d.ob; ++j)
      for (std::size_t a = 0; a < d.vb; ++a)
        for (std::size_t b = 0; b < d.vb; ++b) {
          const std::size_t idx = ((i * d.ob + j) * d.vb + a) * d.vb + b;
          const double g = core.goovv_bb[idx];
          e += 0.25 * g * amp.t2bb[idx] +
               0.5 * g * amp.t1b[i * d.vb + a] * amp.t1b[j * d.vb + b];
        }
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t j = 0; j < d.ob; ++j)
      for (std::size_t a = 0; a < d.va; ++a)
        for (std::size_t b = 0; b < d.vb; ++b) {
          const std::size_t idx = ((i * d.ob + j) * d.va + a) * d.vb + b;
          const double g = core.goovv_ab[idx];
          e += g * (amp.t2ab[idx] + amp.t1a[i * d.va + a] * amp.t1b[j * d.vb + b]);
        }

  if (f12) {
    e += f12->fixed_geminal_energy;
    for (std::size_t i = 0; i < d.oa; ++i)
      for (std::size_t a = 0; a < d.va; ++a) {
        double r1 = 0.0;
        for (std::size_t j = 0; j < d.oa; ++j)
          r1 += f12->w_vo_aaaa[((i * d.oa + j) * d.va + a) * d.oa + j];
        for (std::size_t j = 0; j < d.ob; ++j)
          r1 += f12->w_vo_abab[((i * d.ob + j) * d.va + a) * d.ob + j];
        e += r1 * amp.t1a[i * d.va + a];
      }
    for (std::size_t i = 0; i < d.ob; ++i)
      for (std::size_t a = 0; a < d.vb; ++a) {
        double r1 = 0.0;
        for (std::size_t j = 0; j < d.ob; ++j)
          r1 += f12->w_vo_bbbb[((i * d.ob + j) * d.vb + a) * d.ob + j];
        for (std::size_t j = 0; j < d.oa; ++j)
          r1 -= f12->w_vo_abba[((j * d.ob + i) * d.vb + a) * d.oa + j];
        e += r1 * amp.t1b[i * d.vb + a];
      }
    for (std::size_t i = 0; i < d.oa; ++i)
      for (std::size_t j = 0; j < d.oa; ++j)
        for (std::size_t a = 0; a < d.va; ++a)
          for (std::size_t b = 0; b < d.va; ++b) {
            const std::size_t idx = ((i * d.oa + j) * d.va + a) * d.va + b;
            e += 0.25 * (f12->c_f12_aaaa[idx] + f12->w_vv_aaaa[idx]) * amp.t2aa[idx] +
                 0.5 * f12->w_vv_aaaa[idx] * amp.t1a[i * d.va + a] * amp.t1a[j * d.va + b];
          }
    for (std::size_t i = 0; i < d.ob; ++i)
      for (std::size_t j = 0; j < d.ob; ++j)
        for (std::size_t a = 0; a < d.vb; ++a)
          for (std::size_t b = 0; b < d.vb; ++b) {
            const std::size_t idx = ((i * d.ob + j) * d.vb + a) * d.vb + b;
            e += 0.25 * (f12->c_f12_bbbb[idx] + f12->w_vv_bbbb[idx]) * amp.t2bb[idx] +
                 0.5 * f12->w_vv_bbbb[idx] * amp.t1b[i * d.vb + a] * amp.t1b[j * d.vb + b];
          }
    for (std::size_t i = 0; i < d.oa; ++i)
      for (std::size_t j = 0; j < d.ob; ++j)
        for (std::size_t a = 0; a < d.va; ++a)
          for (std::size_t b = 0; b < d.vb; ++b) {
            const std::size_t idx = ((i * d.ob + j) * d.va + a) * d.vb + b;
            e += (f12->c_f12_abab[idx] + f12->w_vv_abab[idx]) * amp.t2ab[idx] +
                 f12->w_vv_abab[idx] * amp.t1a[i * d.va + a] * amp.t1b[j * d.vb + b];
          }
  }
  return e;
}

SpinBlockedUccAmplitudes compute_uccsd_residual_cpu(const SpinBlockedDFSource& src,
                                                    const SpinBlockedFourIndexCore& core,
                                                    const SpinBlockedUccAmplitudes& amp,
                                                    const FixedF12Coupling* f12) {
  const auto& d = src.dims;
  const std::size_t q = d.q, oa = d.oa, ob = d.ob, va = d.va, vb = d.vb;
  SpinBlockedUccAmplitudes res;
  res.allocate_zero(d);

  std::vector<double> tau_aa, tau_ab, tau_bb;
  std::vector<double> tauh_aa, tauh_ab, tauh_bb;
  build_tau(d, amp, 1.0, tau_aa, tau_ab, tau_bb);
  build_tau(d, amp, 0.5, tauh_aa, tauh_ab, tauh_bb);

  // DF singles intermediates (ZERO ovvv):
  // j_q[Q] = sum_{m,f} t1a[m,f]*bov_a[Q,m,f] + sum_{m,f} t1b[m,f]*bov_b[Q,m,f]
  // x_a[Q, m, a] = sum_f t1a[m, f] * bvv_a[Q, a, f]
  // y_a[Q, m, j] = sum_f bov_a[Q, m, f] * t1a[j, f]
  std::vector<double> j_q(q, 0.0);
  for (std::size_t Q = 0; Q < q; ++Q) {
    double s = 0.0;
    const double* ba = src.bov_a.data() + Q * oa * va;
    for (std::size_t k = 0; k < oa * va; ++k) s += ba[k] * amp.t1a[k];
    const double* bb = src.bov_b.data() + Q * ob * vb;
    for (std::size_t k = 0; k < ob * vb; ++k) s += bb[k] * amp.t1b[k];
    j_q[Q] = s;
  }

  std::vector<double> x_a(q * oa * va, 0.0), y_a(q * oa * oa, 0.0);
  for (std::size_t Q = 0; Q < q; ++Q) {
    // x_a[Q, m, a] = t1a[m, f] @ bvv_a[Q, f, a] (bvv is symmetric)
    gemm('N', 'T', oa, va, va, amp.t1a.data(), src.bvv_a.data() + Q * va * va,
         x_a.data() + Q * oa * va);
    // y_a[Q, m, j] = bov_a[Q, m, f] @ t1a^T[f, j]
    gemm('N', 'T', oa, oa, va, src.bov_a.data() + Q * oa * va, amp.t1a.data(),
         y_a.data() + Q * oa * oa);
  }
  std::vector<double> x_b(q * ob * vb, 0.0), y_b(q * ob * ob, 0.0);
  for (std::size_t Q = 0; Q < q; ++Q) {
    gemm('N', 'T', ob, vb, vb, amp.t1b.data(), src.bvv_b.data() + Q * vb * vb,
         x_b.data() + Q * ob * vb);
    gemm('N', 'T', ob, ob, vb, src.bov_b.data() + Q * ob * vb, amp.t1b.data(),
         y_b.data() + Q * ob * ob);
  }

  // --- Intermediates Fvv, Foo, Fov ---
  std::vector<double> Fvv_a = src.fvv_a;
  std::vector<double> Fvv_b = src.fvv_b;
  // -0.5 fov^T @ t1  -> Fvv[a,e] -= 0.5 sum_m t1[m,a] fov[m,e]
  gemm('T', 'N', va, va, oa, amp.t1a.data(), src.fov_a.data(), Fvv_a.data(), -0.5, 1.0);
  gemm('T', 'N', vb, vb, ob, amp.t1b.data(), src.fov_b.data(), Fvv_b.data(), -0.5, 1.0);
  // + sum_Q j_q[Q] bvv[Q,a,e] - sum_{Q,m} x[Q,m,a] bov[Q,m,e]
  for (std::size_t Q = 0; Q < q; ++Q) {
    const double jq = j_q[Q];
    const double* bva = src.bvv_a.data() + Q * va * va;
    for (std::size_t k = 0; k < va * va; ++k) Fvv_a[k] += jq * bva[k];
    const double* bvb = src.bvv_b.data() + Q * vb * vb;
    for (std::size_t k = 0; k < vb * vb; ++k) Fvv_b[k] += jq * bvb[k];
  }
  gemm('T', 'N', va, va, q * oa, x_a.data(), src.bov_a.data(), Fvv_a.data(), -1.0, 1.0);
  gemm('T', 'N', vb, vb, q * ob, x_b.data(), src.bov_b.data(), Fvv_b.data(), -1.0, 1.0);
  // -0.5 tauh_aa[m,n,a,f] goovv_aa[m,n,e,f] - tauh_ab[m,n,a,f] goovv_ab[m,n,e,f]
  for (std::size_t m = 0; m < oa; ++m)
    for (std::size_t n = 0; n < oa; ++n)
      gemm('N', 'T', va, va, va, tauh_aa.data() + (m * oa + n) * va * va,
           core.goovv_aa.data() + (m * oa + n) * va * va, Fvv_a.data(), -0.5, 1.0);
  for (std::size_t m = 0; m < oa; ++m)
    for (std::size_t n = 0; n < ob; ++n)
      gemm('N', 'T', va, va, vb, tauh_ab.data() + (m * ob + n) * va * vb,
           core.goovv_ab.data() + (m * ob + n) * va * vb, Fvv_a.data(), -1.0, 1.0);
  for (std::size_t m = 0; m < ob; ++m)
    for (std::size_t n = 0; n < ob; ++n)
      gemm('N', 'T', vb, vb, vb, tauh_bb.data() + (m * ob + n) * vb * vb,
           core.goovv_bb.data() + (m * ob + n) * vb * vb, Fvv_b.data(), -0.5, 1.0);
  for (std::size_t m = 0; m < oa; ++m)
    for (std::size_t n = 0; n < ob; ++n)
      gemm('T', 'N', vb, vb, va, tauh_ab.data() + (m * ob + n) * va * vb,
           core.goovv_ab.data() + (m * ob + n) * va * vb, Fvv_b.data(), -1.0, 1.0);

  std::vector<double> Foo_a = src.foo_a;
  std::vector<double> Foo_b = src.foo_b;
  gemm('N', 'T', oa, oa, va, src.fov_a.data(), amp.t1a.data(), Foo_a.data(), 0.5, 1.0);
  gemm('N', 'T', ob, ob, vb, src.fov_b.data(), amp.t1b.data(), Foo_b.data(), 0.5, 1.0);
  // + t1[n,e] gooov[m,n,i,e]
  for (std::size_t m = 0; m < oa; ++m)
    for (std::size_t n = 0; n < oa; ++n)
      for (std::size_t i = 0; i < oa; ++i) {
        double s = 0.0;
        const double* g = core.gooov_aa.data() + ((m * oa + n) * oa + i) * va;
        const double* t = amp.t1a.data() + n * va;
        for (std::size_t e = 0; e < va; ++e) s += g[e] * t[e];
        Foo_a[m * oa + i] += s;
      }
  for (std::size_t m = 0; m < oa; ++m)
    for (std::size_t n = 0; n < ob; ++n)
      for (std::size_t i = 0; i < oa; ++i) {
        double s = 0.0;
        const double* g = core.gooov_ab.data() + ((m * ob + n) * oa + i) * vb;
        const double* t = amp.t1b.data() + n * vb;
        for (std::size_t e = 0; e < vb; ++e) s += g[e] * t[e];
        Foo_a[m * oa + i] += s;
      }
  for (std::size_t m = 0; m < ob; ++m)
    for (std::size_t n = 0; n < ob; ++n)
      for (std::size_t i = 0; i < ob; ++i) {
        double s = 0.0;
        const double* g = core.gooov_bb.data() + ((m * ob + n) * ob + i) * vb;
        const double* t = amp.t1b.data() + n * vb;
        for (std::size_t e = 0; e < vb; ++e) s += g[e] * t[e];
        Foo_b[m * ob + i] += s;
      }
  for (std::size_t m = 0; m < ob; ++m)
    for (std::size_t n = 0; n < oa; ++n)
      for (std::size_t i = 0; i < ob; ++i) {
        double s = 0.0;
        const double* g = core.gooov_ba.data() + ((m * oa + n) * ob + i) * va;
        const double* t = amp.t1a.data() + n * va;
        for (std::size_t e = 0; e < va; ++e) s += g[e] * t[e];
        Foo_b[m * ob + i] += s;
      }
  // +0.5 goovv_aa[m,n,e,f] tauh_aa[i,n,e,f] + goovv_ab[m,n,e,f] tauh_ab[i,n,e,f]
  gemm('N', 'T', oa, oa, oa * va * va, core.goovv_aa.data(), tauh_aa.data(), Foo_a.data(), 0.5,
       1.0);
  gemm('N', 'T', oa, oa, ob * va * vb, core.goovv_ab.data(), tauh_ab.data(), Foo_a.data(), 1.0,
       1.0);
  gemm('N', 'T', ob, ob, ob * vb * vb, core.goovv_bb.data(), tauh_bb.data(), Foo_b.data(), 0.5,
       1.0);
  for (std::size_t n = 0; n < oa; ++n)
    gemm('N', 'T', ob, ob, va * vb, core.goovv_ab.data() + n * ob * va * vb,
         tauh_ab.data() + n * ob * va * vb, Foo_b.data(), 1.0, 1.0);

  std::vector<double> Fov_a = src.fov_a;
  std::vector<double> Fov_b = src.fov_b;
  for (std::size_t m = 0; m < oa; ++m)
    for (std::size_t e = 0; e < va; ++e) {
      double s = 0.0;
      for (std::size_t n = 0; n < oa; ++n)
        for (std::size_t f = 0; f < va; ++f)
          s += core.goovv_aa[((m * oa + n) * va + e) * va + f] * amp.t1a[n * va + f];
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t f = 0; f < vb; ++f)
          s += core.goovv_ab[((m * ob + n) * va + e) * vb + f] * amp.t1b[n * vb + f];
      Fov_a[m * va + e] += s;
    }
  for (std::size_t m = 0; m < ob; ++m)
    for (std::size_t e = 0; e < vb; ++e) {
      double s = 0.0;
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t f = 0; f < vb; ++f)
          s += core.goovv_bb[((m * ob + n) * vb + e) * vb + f] * amp.t1b[n * vb + f];
      for (std::size_t n = 0; n < oa; ++n)
        for (std::size_t f = 0; f < va; ++f)
          s += core.goovv_ab[((n * ob + m) * va + f) * vb + e] * amp.t1a[n * va + f];
      Fov_b[m * vb + e] += s;
    }

  std::vector<double> Fvv_d_a = Fvv_a, Fvv_d_b = Fvv_b;
  gemm('T', 'N', va, va, oa, amp.t1a.data(), Fov_a.data(), Fvv_d_a.data(), -0.5, 1.0);
  gemm('T', 'N', vb, vb, ob, amp.t1b.data(), Fov_b.data(), Fvv_d_b.data(), -0.5, 1.0);

  std::vector<double> Foo_d_a = Foo_a, Foo_d_b = Foo_b;
  gemm('N', 'T', oa, oa, va, Fov_a.data(), amp.t1a.data(), Foo_d_a.data(), 0.5, 1.0);
  gemm('N', 'T', ob, ob, vb, Fov_b.data(), amp.t1b.data(), Foo_d_b.data(), 0.5, 1.0);

  // --- Singles Residuals r1a, r1b ---
  res.t1a = src.fov_a;
  res.t1b = src.fov_b;
  // + t1[i,e] Fvv[a,e] - t1[m,a] Foo[m,i]
  gemm('N', 'T', oa, va, va, amp.t1a.data(), Fvv_a.data(), res.t1a.data(), 1.0, 1.0);
  gemm('T', 'N', oa, va, oa, Foo_a.data(), amp.t1a.data(), res.t1a.data(), -1.0, 1.0);
  gemm('N', 'T', ob, vb, vb, amp.t1b.data(), Fvv_b.data(), res.t1b.data(), 1.0, 1.0);
  gemm('T', 'N', ob, vb, ob, Foo_b.data(), amp.t1b.data(), res.t1b.data(), -1.0, 1.0);

  // + t2[i,m,a,e] Fov[m,e] - t1[n,f] govov[n,a,i,f]
  for (std::size_t i = 0; i < oa; ++i)
    for (std::size_t a = 0; a < va; ++a) {
      double s = 0.0;
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t e = 0; e < va; ++e)
          s += amp.t2aa[((i * oa + m) * va + a) * va + e] * Fov_a[m * va + e] -
               amp.t1a[m * va + e] * core.govov_aa[((m * va + a) * oa + i) * va + e];
      for (std::size_t m = 0; m < ob; ++m)
        for (std::size_t e = 0; e < vb; ++e)
          s += amp.t2ab[((i * ob + m) * va + a) * vb + e] * Fov_b[m * vb + e] +
               amp.t1b[m * vb + e] * core.goovv_ab[((i * ob + m) * va + a) * vb + e];
      res.t1a[i * va + a] += s;
    }
  for (std::size_t i = 0; i < ob; ++i)
    for (std::size_t a = 0; a < vb; ++a) {
      double s = 0.0;
      for (std::size_t m = 0; m < ob; ++m)
        for (std::size_t e = 0; e < vb; ++e)
          s += amp.t2bb[((i * ob + m) * vb + a) * vb + e] * Fov_b[m * vb + e] -
               amp.t1b[m * vb + e] * core.govov_bb[((m * vb + a) * ob + i) * vb + e];
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t e = 0; e < va; ++e)
          s += amp.t2ab[((m * ob + i) * va + e) * vb + a] * Fov_a[m * va + e] +
               amp.t1a[m * va + e] * core.goovv_ab[((m * ob + i) * va + e) * vb + a];
      res.t1b[i * vb + a] += s;
    }

  // -1/2 t2[i,m,e,f] govvv[m,a,e,f] via DF bov & bvv (ZERO ovvv!):
  // r1a[i,a] += - sum_{Q,f} bvv_a[Q,a,f] (sum_{m,e} t2aa[i,m,e,f] bov_a[Q,m,e])
  //             + sum_{Q,e} bvv_a[Q,a,e] (sum_{m,f} t2ab[i,m,e,f] bov_b[Q,m,f])
  if (oa && va) {
    std::vector<double> z_a(q * oa * va, 0.0);  // Z[Q, i, f]
    // t2aa_perm[i, f, m, e] = t2aa[i, m, e, f]
    std::vector<double> t2_ifme(oa * va * oa * va, 0.0);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t e = 0; e < va; ++e)
          for (std::size_t f = 0; f < va; ++f)
            t2_ifme[((i * va + f) * oa + m) * va + e] =
                amp.t2aa[((i * oa + m) * va + e) * va + f];
    gemm('N', 'T', q, oa * va, oa * va, src.bov_a.data(), t2_ifme.data(), z_a.data(), -1.0, 0.0);
    if (ob && vb) {
      std::vector<double> t2ab_iemf(oa * va * ob * vb, 0.0);
      for (std::size_t i = 0; i < oa; ++i)
        for (std::size_t m = 0; m < ob; ++m)
          for (std::size_t e = 0; e < va; ++e)
            for (std::size_t f = 0; f < vb; ++f)
              t2ab_iemf[((i * va + e) * ob + m) * vb + f] =
                  amp.t2ab[((i * ob + m) * va + e) * vb + f];
      gemm('N', 'T', q, oa * va, ob * vb, src.bov_b.data(), t2ab_iemf.data(), z_a.data(), 1.0, 1.0);
    }
    for (std::size_t Q = 0; Q < q; ++Q)
      gemm('N', 'T', oa, va, va, z_a.data() + Q * oa * va, src.bvv_a.data() + Q * va * va,
           res.t1a.data(), 1.0, 1.0);
  }
  if (ob && vb) {
    std::vector<double> z_b(q * ob * vb, 0.0);
    std::vector<double> t2_ifme(ob * vb * ob * vb, 0.0);
    for (std::size_t i = 0; i < ob; ++i)
      for (std::size_t m = 0; m < ob; ++m)
        for (std::size_t e = 0; e < vb; ++e)
          for (std::size_t f = 0; f < vb; ++f)
            t2_ifme[((i * vb + f) * ob + m) * vb + e] =
                amp.t2bb[((i * ob + m) * vb + e) * vb + f];
    gemm('N', 'T', q, ob * vb, ob * vb, src.bov_b.data(), t2_ifme.data(), z_b.data(), -1.0, 0.0);
    if (oa && va) {
      std::vector<double> t2ab_ifme(ob * vb * oa * va, 0.0);
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t i = 0; i < ob; ++i)
          for (std::size_t e = 0; e < va; ++e)
            for (std::size_t f = 0; f < vb; ++f)
              t2ab_ifme[((i * vb + f) * oa + m) * va + e] =
                  amp.t2ab[((m * ob + i) * va + e) * vb + f];
      gemm('N', 'T', q, ob * vb, oa * va, src.bov_a.data(), t2ab_ifme.data(), z_b.data(), 1.0, 1.0);
    }
    for (std::size_t Q = 0; Q < q; ++Q)
      gemm('N', 'T', ob, vb, vb, z_b.data() + Q * ob * vb, src.bvv_b.data() + Q * vb * vb,
           res.t1b.data(), 1.0, 1.0);
  }

  // -0.5 t2[m,n,a,e] gooov[m,n,i,e]
  for (std::size_t m = 0; m < oa; ++m)
    for (std::size_t n = 0; n < oa; ++n)
      gemm('N', 'T', oa, va, va, core.gooov_aa.data() + (m * oa + n) * oa * va,
           amp.t2aa.data() + (m * oa + n) * va * va, res.t1a.data(), -0.5, 1.0);
  for (std::size_t m = 0; m < oa; ++m)
    for (std::size_t n = 0; n < ob; ++n)
      gemm('N', 'T', oa, va, vb, core.gooov_ab.data() + (m * ob + n) * oa * vb,
           amp.t2ab.data() + (m * ob + n) * va * vb, res.t1a.data(), -1.0, 1.0);
  for (std::size_t m = 0; m < ob; ++m)
    for (std::size_t n = 0; n < ob; ++n)
      gemm('N', 'T', ob, vb, vb, core.gooov_bb.data() + (m * ob + n) * ob * vb,
           amp.t2bb.data() + (m * ob + n) * vb * vb, res.t1b.data(), -0.5, 1.0);
  for (std::size_t m = 0; m < ob; ++m)
    for (std::size_t n = 0; n < oa; ++n)
      gemm('N', 'N', ob, vb, va, core.gooov_ba.data() + (m * oa + n) * ob * va,
           amp.t2ab.data() + (n * ob + m) * va * vb, res.t1b.data(), -1.0, 1.0);

  // --- Doubles Residuals r2aa, r2bb, r2ab ---
  res.t2aa = core.goovv_aa;
  res.t2bb = core.goovv_bb;
  res.t2ab = core.goovv_ab;

  // Fvv_d and Foo_d terms
  if (oa && va) {
    std::vector<double> tmp(oa * oa * va * va, 0.0);
    gemm('N', 'T', oa * oa * va, va, va, amp.t2aa.data(), Fvv_d_a.data(), tmp.data());
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < oa; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < va; ++b)
            res.t2aa[((i * oa + j) * va + a) * va + b] +=
                tmp[((i * oa + j) * va + a) * va + b] - tmp[((i * oa + j) * va + b) * va + a];

    std::fill(tmp.begin(), tmp.end(), 0.0);
    for (std::size_t i = 0; i < oa; ++i)
      gemm('T', 'N', oa, va * va, oa, Foo_d_a.data(), amp.t2aa.data() + i * oa * va * va,
           tmp.data() + i * oa * va * va);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < oa; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < va; ++b)
            res.t2aa[((i * oa + j) * va + a) * va + b] +=
                -tmp[((i * oa + j) * va + a) * va + b] + tmp[((j * oa + i) * va + a) * va + b];
  }
  if (ob && vb) {
    std::vector<double> tmp(ob * ob * vb * vb, 0.0);
    gemm('N', 'T', ob * ob * vb, vb, vb, amp.t2bb.data(), Fvv_d_b.data(), tmp.data());
    for (std::size_t i = 0; i < ob; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < vb; ++a)
          for (std::size_t b = 0; b < vb; ++b)
            res.t2bb[((i * ob + j) * vb + a) * vb + b] +=
                tmp[((i * ob + j) * vb + a) * vb + b] - tmp[((i * ob + j) * vb + b) * vb + a];

    std::fill(tmp.begin(), tmp.end(), 0.0);
    for (std::size_t i = 0; i < ob; ++i)
      gemm('T', 'N', ob, vb * vb, ob, Foo_d_b.data(), amp.t2bb.data() + i * ob * vb * vb,
           tmp.data() + i * ob * vb * vb);
    for (std::size_t i = 0; i < ob; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < vb; ++a)
          for (std::size_t b = 0; b < vb; ++b)
            res.t2bb[((i * ob + j) * vb + a) * vb + b] +=
                -tmp[((i * ob + j) * vb + a) * vb + b] + tmp[((j * ob + i) * vb + a) * vb + b];
  }
  if (oa && ob && va && vb) {
    // r2ab[i,j,a,b] += t2ab[i,j,a,e] Fvv_d_b[b,e] + Fvv_d_a[a,e] t2ab[i,j,e,b]
    gemm('N', 'T', oa * ob * va, vb, vb, amp.t2ab.data(), Fvv_d_b.data(), res.t2ab.data(), 1.0,
         1.0);
    for (std::size_t ij = 0; ij < oa * ob; ++ij)
      gemm('N', 'N', va, vb, va, Fvv_d_a.data(), amp.t2ab.data() + ij * va * vb,
           res.t2ab.data() + ij * va * vb, 1.0, 1.0);
    // r2ab[i,j,a,b] += - t2ab[i,m,a,b] Foo_d_b[m,j] - Foo_d_a[m,i] t2ab[m,j,a,b]
    for (std::size_t i = 0; i < oa; ++i)
      gemm('T', 'N', ob, va * vb, ob, Foo_d_b.data(), amp.t2ab.data() + i * ob * va * vb,
           res.t2ab.data() + i * ob * va * vb, -1.0, 1.0);
    gemm('T', 'N', oa, ob * va * vb, oa, Foo_d_a.data(), amp.t2ab.data(), res.t2ab.data(), -1.0,
         1.0);
  }

  // Woooo_bare + tau_oovv * t2 (exact cancellation with T1-dressed PPL B_tilde!)
  if (oa && va) {
    std::vector<double> w_bare = core.goooo_aa;
    for (std::size_t mn = 0; mn < oa * oa; ++mn)
      for (std::size_t i = 0; i < oa; ++i)
        for (std::size_t j = 0; j < oa; ++j) {
          double s = 0.0;
          const double* gi = core.gooov_aa.data() + (mn * oa + i) * va;
          const double* gj = core.gooov_aa.data() + (mn * oa + j) * va;
          const double* ti = amp.t1a.data() + i * va;
          const double* tj = amp.t1a.data() + j * va;
          for (std::size_t e = 0; e < va; ++e) s += tj[e] * gi[e] - ti[e] * gj[e];
          w_bare[mn * oa * oa + i * oa + j] += s;
        }
    // 0.5 * Woooo_bare[mn, ij]^T @ tau_aa[mn, ab]
    gemm('T', 'N', oa * oa, va * va, oa * oa, w_bare.data(), tau_aa.data(), res.t2aa.data(), 0.5,
         1.0);
    // tau_oovv_aa[mn, ij] = goovv_aa[mn, ef] @ tau_aa[ij, ef]^T
    std::vector<double> tau_oovv(oa * oa * oa * oa, 0.0);
    gemm('N', 'T', oa * oa, oa * oa, va * va, core.goovv_aa.data(), tau_aa.data(), tau_oovv.data());
    gemm('T', 'N', oa * oa, va * va, oa * oa, tau_oovv.data(), amp.t2aa.data(), res.t2aa.data(),
         0.25, 1.0);
  }
  if (ob && vb) {
    std::vector<double> w_bare = core.goooo_bb;
    for (std::size_t mn = 0; mn < ob * ob; ++mn)
      for (std::size_t i = 0; i < ob; ++i)
        for (std::size_t j = 0; j < ob; ++j) {
          double s = 0.0;
          const double* gi = core.gooov_bb.data() + (mn * ob + i) * vb;
          const double* gj = core.gooov_bb.data() + (mn * ob + j) * vb;
          const double* ti = amp.t1b.data() + i * vb;
          const double* tj = amp.t1b.data() + j * vb;
          for (std::size_t e = 0; e < vb; ++e) s += tj[e] * gi[e] - ti[e] * gj[e];
          w_bare[mn * ob * ob + i * ob + j] += s;
        }
    gemm('T', 'N', ob * ob, vb * vb, ob * ob, w_bare.data(), tau_bb.data(), res.t2bb.data(), 0.5,
         1.0);
    std::vector<double> tau_oovv(ob * ob * ob * ob, 0.0);
    gemm('N', 'T', ob * ob, ob * ob, vb * vb, core.goovv_bb.data(), tau_bb.data(), tau_oovv.data());
    gemm('T', 'N', ob * ob, vb * vb, ob * ob, tau_oovv.data(), amp.t2bb.data(), res.t2bb.data(),
         0.25, 1.0);
  }
  if (oa && ob && va && vb) {
    std::vector<double> w_bare = core.goooo_ab;
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t i = 0; i < oa; ++i)
          for (std::size_t j = 0; j < ob; ++j) {
            double s = 0.0;
            const double* gab = core.gooov_ab.data() + ((m * ob + n) * oa + i) * vb;
            const double* tj = amp.t1b.data() + j * vb;
            for (std::size_t e = 0; e < vb; ++e) s += tj[e] * gab[e];
            const double* gba = core.gooov_ba.data() + ((n * oa + m) * ob + j) * va;
            const double* ti = amp.t1a.data() + i * va;
            for (std::size_t e = 0; e < va; ++e) s += ti[e] * gba[e];
            w_bare[((m * ob + n) * oa + i) * ob + j] += s;
          }
    gemm('T', 'N', oa * ob, va * vb, oa * ob, w_bare.data(), tau_ab.data(), res.t2ab.data(), 1.0,
         1.0);
    std::vector<double> g_ab(oa * ob * oa * ob, 0.0);
    gemm('N', 'T', oa * ob, oa * ob, va * vb, core.goovv_ab.data(), tau_ab.data(), g_ab.data());
    gemm('T', 'N', oa * ob, va * vb, oa * ob, g_ab.data(), amp.t2ab.data(), res.t2ab.data(), 1.0,
         1.0);
  }

  // --- On-the-fly T1-dressed DF Virtual Ladder (ZERO vvvv, ZERO ovvv!) ---
  // B_tilde[Q, a, e] = bvv[Q, a, e] - sum_m t1[m, a] bov[Q, m, e]
  std::vector<double> bt_a = src.bvv_a;
  for (std::size_t Q = 0; Q < q; ++Q)
    gemm('T', 'N', va, va, oa, amp.t1a.data(), src.bov_a.data() + Q * oa * va,
         bt_a.data() + Q * va * va, -1.0, 1.0);
  std::vector<double> bt_b = src.bvv_b;
  for (std::size_t Q = 0; Q < q; ++Q)
    gemm('T', 'N', vb, vb, ob, amp.t1b.data(), src.bov_b.data() + Q * ob * vb,
         bt_b.data() + Q * vb * vb, -1.0, 1.0);

  // L_aa[i, j, a, b] = sum_Q sum_{e,f} bt_a[Q, a, e] tau_aa[i, j, e, f] bt_a[Q, b, f]
  if (oa && va) {
    std::vector<double> half(va * va, 0.0);
    for (std::size_t Q = 0; Q < q; ++Q) {
      const double* bta = bt_a.data() + Q * va * va;
      for (std::size_t ij = 0; ij < oa * oa; ++ij) {
        gemm('N', 'N', va, va, va, bta, tau_aa.data() + ij * va * va, half.data());
        gemm('N', 'T', va, va, va, half.data(), bta, res.t2aa.data() + ij * va * va, 1.0, 1.0);
      }
    }
  }
  if (ob && vb) {
    std::vector<double> half(vb * vb, 0.0);
    for (std::size_t Q = 0; Q < q; ++Q) {
      const double* btb = bt_b.data() + Q * vb * vb;
      for (std::size_t ij = 0; ij < ob * ob; ++ij) {
        gemm('N', 'N', vb, vb, vb, btb, tau_bb.data() + ij * vb * vb, half.data());
        gemm('N', 'T', vb, vb, vb, half.data(), btb, res.t2bb.data() + ij * vb * vb, 1.0, 1.0);
      }
    }
  }
  if (oa && ob && va && vb) {
    std::vector<double> half(va * vb, 0.0);
    for (std::size_t Q = 0; Q < q; ++Q) {
      const double* bta = bt_a.data() + Q * va * va;
      const double* btb = bt_b.data() + Q * vb * vb;
      for (std::size_t ij = 0; ij < oa * ob; ++ij) {
        gemm('N', 'N', va, vb, va, bta, tau_ab.data() + ij * va * vb, half.data());
        gemm('N', 'T', va, vb, vb, half.data(), btb, res.t2ab.data() + ij * va * vb, 1.0, 1.0);
      }
    }
  }

  // --- Ring intermediate Wovvo (all 6 spin-conserving blocks, stored as [m, e, j, b]) ---
  // Notice storing W as [m, e, j, b] makes every ring contraction a single BLAS GEMM!
  std::vector<double> W_aaaa(oa * va * oa * va, 0.0);
  std::vector<double> W_bbbb(ob * vb * ob * vb, 0.0);
  std::vector<double> W_abab(oa * va * ob * vb, 0.0);
  std::vector<double> W_baba(ob * vb * oa * va, 0.0);
  std::vector<double> W_abba(oa * vb * oa * vb, 0.0);  // [m_a, e_b, j_a, b_b]
  std::vector<double> W_baab(ob * va * ob * va, 0.0);  // [m_b, e_a, j_b, b_a]

  // 1) W_aaaa[m, e, j, b] = -govov_aa[m,b,j,e] + bov_a[Q,m,e] x_a[Q,j,b] - y_a[Q,m,j] bvv_a[Q,b,e]
  //    + t1a[n,b] gooov_aa[m,n,j,e] - (0.5 t2aa[j,n,f,b] + t1a[j,f] t1a[n,b]) goovv_aa[m,n,e,f]
  //    + 0.5 t2ab[j,n,b,f] goovv_ab[m,n,e,f]
  if (oa && va) {
    gemm('T', 'N', oa * va, oa * va, q, src.bov_a.data(), x_a.data(), W_aaaa.data(), 1.0, 0.0);
    std::vector<double> y_bvv(oa * oa * va * va, 0.0);
    gemm('T', 'N', oa * oa, va * va, q, y_a.data(), src.bvv_a.data(), y_bvv.data());
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t e = 0; e < va; ++e)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t b = 0; b < va; ++b) {
            double v = -core.govov_aa[((m * va + b) * oa + j) * va + e] -
                       y_bvv[((m * oa + j) * va + b) * va + e];
            for (std::size_t n = 0; n < oa; ++n)
              v += amp.t1a[n * va + b] * core.gooov_aa[((m * oa + n) * oa + j) * va + e];
            W_aaaa[((m * va + e) * oa + j) * va + b] += v;
          }
    std::vector<double> g_menf(oa * va * oa * va, 0.0), t_nfjb(oa * va * oa * va, 0.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < oa; ++n)
        for (std::size_t e = 0; e < va; ++e)
          for (std::size_t f = 0; f < va; ++f)
            g_menf[((m * va + e) * oa + n) * va + f] =
                core.goovv_aa[((m * oa + n) * va + e) * va + f];
    for (std::size_t n = 0; n < oa; ++n)
      for (std::size_t f = 0; f < va; ++f)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t b = 0; b < va; ++b)
            t_nfjb[((n * va + f) * oa + j) * va + b] =
                0.5 * amp.t2aa[((j * oa + n) * va + f) * va + b] +
                amp.t1a[j * va + f] * amp.t1a[n * va + b];
    gemm('N', 'N', oa * va, oa * va, oa * va, g_menf.data(), t_nfjb.data(), W_aaaa.data(), -1.0,
         1.0);
    if (ob && vb) {
      std::vector<double> gab_menf(oa * va * ob * vb, 0.0), tab_nfjb(ob * vb * oa * va, 0.0);
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t n = 0; n < ob; ++n)
          for (std::size_t e = 0; e < va; ++e)
            for (std::size_t f = 0; f < vb; ++f)
              gab_menf[((m * va + e) * ob + n) * vb + f] =
                  core.goovv_ab[((m * ob + n) * va + e) * vb + f];
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t f = 0; f < vb; ++f)
          for (std::size_t j = 0; j < oa; ++j)
            for (std::size_t b = 0; b < va; ++b)
              tab_nfjb[((n * vb + f) * oa + j) * va + b] =
                  amp.t2ab[((j * ob + n) * va + b) * vb + f];
      gemm('N', 'N', oa * va, oa * va, ob * vb, gab_menf.data(), tab_nfjb.data(), W_aaaa.data(),
           0.5, 1.0);
    }
  }
  // 2) W_bbbb[m, e, j, b]
  if (ob && vb) {
    gemm('T', 'N', ob * vb, ob * vb, q, src.bov_b.data(), x_b.data(), W_bbbb.data(), 1.0, 0.0);
    std::vector<double> y_bvv(ob * ob * vb * vb, 0.0);
    gemm('T', 'N', ob * ob, vb * vb, q, y_b.data(), src.bvv_b.data(), y_bvv.data());
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t e = 0; e < vb; ++e)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < vb; ++b) {
            double v = -core.govov_bb[((m * vb + b) * ob + j) * vb + e] -
                       y_bvv[((m * ob + j) * vb + b) * vb + e];
            for (std::size_t n = 0; n < ob; ++n)
              v += amp.t1b[n * vb + b] * core.gooov_bb[((m * ob + n) * ob + j) * vb + e];
            W_bbbb[((m * vb + e) * ob + j) * vb + b] += v;
          }
    std::vector<double> g_menf(ob * vb * ob * vb, 0.0), t_nfjb(ob * vb * ob * vb, 0.0);
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t e = 0; e < vb; ++e)
          for (std::size_t f = 0; f < vb; ++f)
            g_menf[((m * vb + e) * ob + n) * vb + f] =
                core.goovv_bb[((m * ob + n) * vb + e) * vb + f];
    for (std::size_t n = 0; n < ob; ++n)
      for (std::size_t f = 0; f < vb; ++f)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < vb; ++b)
            t_nfjb[((n * vb + f) * ob + j) * vb + b] =
                0.5 * amp.t2bb[((j * ob + n) * vb + f) * vb + b] +
                amp.t1b[j * vb + f] * amp.t1b[n * vb + b];
    gemm('N', 'N', ob * vb, ob * vb, ob * vb, g_menf.data(), t_nfjb.data(), W_bbbb.data(), -1.0,
         1.0);
    if (oa && va) {
      std::vector<double> gab_menf(ob * vb * oa * va, 0.0), tab_nfjb(oa * va * ob * vb, 0.0);
      for (std::size_t m = 0; m < ob; ++m)
        for (std::size_t n = 0; n < oa; ++n)
          for (std::size_t e = 0; e < vb; ++e)
            for (std::size_t f = 0; f < va; ++f)
              gab_menf[((m * vb + e) * oa + n) * va + f] =
                  core.goovv_ab[((n * ob + m) * va + f) * vb + e];
      for (std::size_t n = 0; n < oa; ++n)
        for (std::size_t f = 0; f < va; ++f)
          for (std::size_t j = 0; j < ob; ++j)
            for (std::size_t b = 0; b < vb; ++b)
              tab_nfjb[((n * va + f) * ob + j) * vb + b] =
                  amp.t2ab[((n * ob + j) * va + f) * vb + b];
      gemm('N', 'N', ob * vb, ob * vb, oa * va, gab_menf.data(), tab_nfjb.data(), W_bbbb.data(),
           0.5, 1.0);
    }
  }
  // 3) W_abab[m_a, e_a, j_b, b_b] and 4) W_baba[m_b, e_b, j_a, b_a]
  if (oa && ob && va && vb) {
    gemm('T', 'N', oa * va, ob * vb, q, src.bov_a.data(), x_b.data(), W_abab.data(), 1.0, 0.0);
    gemm('T', 'N', ob * vb, oa * va, q, src.bov_b.data(), x_a.data(), W_baba.data(), 1.0, 0.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t e = 0; e < va; ++e)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < vb; ++b) {
            double v = core.goovv_ab[((m * ob + j) * va + e) * vb + b];
            for (std::size_t n = 0; n < ob; ++n)
              v -= amp.t1b[n * vb + b] * core.gooov_ba[((n * oa + m) * ob + j) * va + e];
            W_abab[((m * va + e) * ob + j) * vb + b] += v;
          }
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t e = 0; e < vb; ++e)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t b = 0; b < va; ++b) {
            double v = core.goovv_ab[((j * ob + m) * va + b) * vb + e];
            for (std::size_t n = 0; n < oa; ++n)
              v -= amp.t1a[n * va + b] * core.gooov_ab[((n * ob + m) * oa + j) * vb + e];
            W_baba[((m * vb + e) * oa + j) * va + b] += v;
          }
    std::vector<double> gab_me_nf(oa * va * ob * vb, 0.0), gaa_me_nf(oa * va * oa * va, 0.0),
        gbb_me_nf(ob * vb * ob * vb, 0.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t e = 0; e < va; ++e)
          for (std::size_t f = 0; f < vb; ++f)
            gab_me_nf[((m * va + e) * ob + n) * vb + f] =
                core.goovv_ab[((m * ob + n) * va + e) * vb + f];
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < oa; ++n)
        for (std::size_t e = 0; e < va; ++e)
          for (std::size_t f = 0; f < va; ++f)
            gaa_me_nf[((m * va + e) * oa + n) * va + f] =
                core.goovv_aa[((m * oa + n) * va + e) * va + f];
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t e = 0; e < vb; ++e)
          for (std::size_t f = 0; f < vb; ++f)
            gbb_me_nf[((m * vb + e) * ob + n) * vb + f] =
                core.goovv_bb[((m * ob + n) * vb + e) * vb + f];

    std::vector<double> tb_nfjb(ob * vb * ob * vb, 0.0), ta_nfjb(oa * va * oa * va, 0.0);
    for (std::size_t n = 0; n < ob; ++n)
      for (std::size_t f = 0; f < vb; ++f)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < vb; ++b)
            tb_nfjb[((n * vb + f) * ob + j) * vb + b] =
                0.5 * amp.t2bb[((j * ob + n) * vb + f) * vb + b] +
                amp.t1b[j * vb + f] * amp.t1b[n * vb + b];
    for (std::size_t n = 0; n < oa; ++n)
      for (std::size_t f = 0; f < va; ++f)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t b = 0; b < va; ++b)
            ta_nfjb[((n * va + f) * oa + j) * va + b] =
                0.5 * amp.t2aa[((j * oa + n) * va + f) * va + b] +
                amp.t1a[j * va + f] * amp.t1a[n * va + b];
    std::vector<double> tab_nfa_jbb(oa * va * ob * vb, 0.0), tab_nfb_jaa(ob * vb * oa * va, 0.0);
    for (std::size_t n = 0; n < oa; ++n)
      for (std::size_t f = 0; f < va; ++f)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < vb; ++b)
            tab_nfa_jbb[((n * va + f) * ob + j) * vb + b] =
                amp.t2ab[((n * ob + j) * va + f) * vb + b];
    for (std::size_t n = 0; n < ob; ++n)
      for (std::size_t f = 0; f < vb; ++f)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t b = 0; b < va; ++b)
            tab_nfb_jaa[((n * vb + f) * oa + j) * va + b] =
                amp.t2ab[((j * ob + n) * va + b) * vb + f];

    gemm('N', 'N', oa * va, ob * vb, ob * vb, gab_me_nf.data(), tb_nfjb.data(), W_abab.data(),
         -1.0, 1.0);
    gemm('N', 'N', oa * va, ob * vb, oa * va, gaa_me_nf.data(), tab_nfa_jbb.data(), W_abab.data(),
         0.5, 1.0);
    gemm('T', 'N', ob * vb, oa * va, oa * va, gab_me_nf.data(), ta_nfjb.data(), W_baba.data(),
         -1.0, 1.0);
    gemm('N', 'N', ob * vb, oa * va, ob * vb, gbb_me_nf.data(), tab_nfb_jaa.data(), W_baba.data(),
         0.5, 1.0);

    // 5) W_abba[m_a, e_b, j_a, b_b]
    std::vector<double> ya_bvv_b(oa * oa * vb * vb, 0.0);
    gemm('T', 'N', oa * oa, vb * vb, q, y_a.data(), src.bvv_b.data(), ya_bvv_b.data());
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t e = 0; e < vb; ++e)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t b = 0; b < vb; ++b) {
            double v = -core.govov_ab[((m * vb + b) * oa + j) * vb + e] -
                       ya_bvv_b[((m * oa + j) * vb + b) * vb + e];
            for (std::size_t n = 0; n < ob; ++n)
              v += amp.t1b[n * vb + b] * core.gooov_ab[((m * ob + n) * oa + j) * vb + e];
            W_abba[((m * vb + e) * oa + j) * vb + b] = v;
          }
    std::vector<double> gab_me_nf_x(oa * vb * ob * va, 0.0), tab_nf_jb_x(ob * va * oa * vb, 0.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t n = 0; n < ob; ++n)
        for (std::size_t f = 0; f < va; ++f)
          for (std::size_t e = 0; e < vb; ++e)
            gab_me_nf_x[((m * vb + e) * ob + n) * va + f] =
                core.goovv_ab[((m * ob + n) * va + f) * vb + e];
    for (std::size_t n = 0; n < ob; ++n)
      for (std::size_t f = 0; f < va; ++f)
        for (std::size_t j = 0; j < oa; ++j)
          for (std::size_t b = 0; b < vb; ++b)
            tab_nf_jb_x[((n * va + f) * oa + j) * vb + b] =
                0.5 * amp.t2ab[((j * ob + n) * va + f) * vb + b] +
                amp.t1a[j * va + f] * amp.t1b[n * vb + b];
    gemm('N', 'N', oa * vb, oa * vb, ob * va, gab_me_nf_x.data(), tab_nf_jb_x.data(),
         W_abba.data(), 1.0, 1.0);

    // 6) W_baab[m_b, e_a, j_b, b_a]
    std::vector<double> yb_bvv_a(ob * ob * va * va, 0.0);
    gemm('T', 'N', ob * ob, va * va, q, y_b.data(), src.bvv_a.data(), yb_bvv_a.data());
    for (std::size_t m = 0; m < ob; ++m)
      for (std::size_t e = 0; e < va; ++e)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < va; ++b) {
            double v = -core.govov_ba[((m * va + b) * ob + j) * va + e] -
                       yb_bvv_a[((m * ob + j) * va + b) * va + e];
            for (std::size_t n = 0; n < oa; ++n)
              v += amp.t1a[n * va + b] * core.gooov_ba[((m * oa + n) * ob + j) * va + e];
            W_baab[((m * va + e) * ob + j) * va + b] = v;
          }
    std::vector<double> gba_me_nf_x(ob * va * oa * vb, 0.0), tba_nf_jb_x(oa * vb * ob * va, 0.0);
    for (std::size_t n = 0; n < oa; ++n)
      for (std::size_t m = 0; m < ob; ++m)
        for (std::size_t e = 0; e < va; ++e)
          for (std::size_t f = 0; f < vb; ++f)
            gba_me_nf_x[((m * va + e) * oa + n) * vb + f] =
                core.goovv_ab[((n * ob + m) * va + e) * vb + f];
    for (std::size_t n = 0; n < oa; ++n)
      for (std::size_t f = 0; f < vb; ++f)
        for (std::size_t j = 0; j < ob; ++j)
          for (std::size_t b = 0; b < va; ++b)
            tba_nf_jb_x[((n * vb + f) * ob + j) * va + b] =
                0.5 * amp.t2ab[((n * ob + j) * va + b) * vb + f] +
                amp.t1b[j * vb + f] * amp.t1a[n * va + b];
    gemm('N', 'N', ob * va, ob * va, oa * vb, gba_me_nf_x.data(), tba_nf_jb_x.data(),
         W_baab.data(), 1.0, 1.0);
  }

  // Ring contractions P(ij)P(ab) t2[i,m,a,e] Wovvo[m,e,j,b]
  if (oa && va) {
    std::vector<double> t2_ia_me(oa * va * oa * va, 0.0);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t e = 0; e < va; ++e)
            t2_ia_me[((i * va + a) * oa + m) * va + e] =
                amp.t2aa[((i * oa + m) * va + a) * va + e];
    std::vector<double> ring(oa * va * oa * va, 0.0);
    gemm('N', 'N', oa * va, oa * va, oa * va, t2_ia_me.data(), W_aaaa.data(), ring.data(), 1.0,
         0.0);
    if (ob && vb) {
      std::vector<double> tab_ia_me(oa * va * ob * vb, 0.0);
      for (std::size_t i = 0; i < oa; ++i)
        for (std::size_t m = 0; m < ob; ++m)
          for (std::size_t a = 0; a < va; ++a)
            for (std::size_t e = 0; e < vb; ++e)
              tab_ia_me[((i * va + a) * ob + m) * vb + e] =
                  amp.t2ab[((i * ob + m) * va + a) * vb + e];
      gemm('N', 'N', oa * va, oa * va, ob * vb, tab_ia_me.data(), W_baba.data(), ring.data(), 1.0,
           1.0);
    }
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < oa; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < va; ++b)
            res.t2aa[((i * oa + j) * va + a) * va + b] +=
                ring[((i * va + a) * oa + j) * va + b] - ring[((j * va + a) * oa + i) * va + b] -
                ring[((i * va + b) * oa + j) * va + a] + ring[((j * va + b) * oa + i) * va + a];
  }
  if (ob && vb) {
    std::vector<double> t2_ia_me(ob * vb * ob * vb, 0.0);
    for (std::size_t i = 0; i < ob; ++i)
      for (std::size_t m = 0; m < ob; ++m)
        for (std::size_t a = 0; a < vb; ++a)
          for (std::size_t e = 0; e < vb; ++e)
            t2_ia_me[((i * vb + a) * ob + m) * vb + e] =
                amp.t2bb[((i * ob + m) * vb + a) * vb + e];
    std::vector<double> ring(ob * vb * ob * vb, 0.0);
    gemm('N', 'N', ob * vb, ob * vb, ob * vb, t2_ia_me.data(), W_bbbb.data(), ring.data(), 1.0,
         0.0);
    if (oa && va) {
      std::vector<double> tab_ia_me(ob * vb * oa * va, 0.0);
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t i = 0; i < ob; ++i)
          for (std::size_t e = 0; e < va; ++e)
            for (std::size_t a = 0; a < vb; ++a)
              tab_ia_me[((i * vb + a) * oa + m) * va + e] =
                  amp.t2ab[((m * ob + i) * va + e) * vb + a];
      gemm('N', 'N', ob * vb, ob * vb, oa * va, tab_ia_me.data(), W_abab.data(), ring.data(), 1.0,
           1.0);
    }
    for (std::size_t i = 0; i < ob; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < vb; ++a)
          for (std::size_t b = 0; b < vb; ++b)
            res.t2bb[((i * ob + j) * vb + a) * vb + b] +=
                ring[((i * vb + a) * ob + j) * vb + b] - ring[((j * vb + a) * ob + i) * vb + b] -
                ring[((i * vb + b) * ob + j) * vb + a] + ring[((j * vb + b) * ob + i) * vb + a];
  }
  if (oa && ob && va && vb) {
    // r2ab ring terms 1 & 2: [i_a, a_a, j_b, b_b]
    std::vector<double> ring12(oa * va * ob * vb, 0.0);
    std::vector<double> taa_ia_me(oa * va * oa * va, 0.0), tab_ia_me(oa * va * ob * vb, 0.0),
        tbb_jb_me(ob * vb * ob * vb, 0.0), tab_jb_me(ob * vb * oa * va, 0.0);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t e = 0; e < va; ++e)
            taa_ia_me[((i * va + a) * oa + m) * va + e] =
                amp.t2aa[((i * oa + m) * va + a) * va + e];
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t m = 0; m < ob; ++m)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t e = 0; e < vb; ++e)
            tab_ia_me[((i * va + a) * ob + m) * vb + e] =
                amp.t2ab[((i * ob + m) * va + a) * vb + e];
    for (std::size_t j = 0; j < ob; ++j)
      for (std::size_t m = 0; m < ob; ++m)
        for (std::size_t b = 0; b < vb; ++b)
          for (std::size_t e = 0; e < vb; ++e)
            tbb_jb_me[((j * vb + b) * ob + m) * vb + e] =
                amp.t2bb[((j * ob + m) * vb + b) * vb + e];
    for (std::size_t j = 0; j < ob; ++j)
      for (std::size_t m = 0; m < oa; ++m)
        for (std::size_t b = 0; b < vb; ++b)
          for (std::size_t e = 0; e < va; ++e)
            tab_jb_me[((j * vb + b) * oa + m) * va + e] =
                amp.t2ab[((m * ob + j) * va + e) * vb + b];
    gemm('N', 'N', oa * va, ob * vb, oa * va, taa_ia_me.data(), W_abab.data(), ring12.data(), 1.0,
         0.0);
    gemm('N', 'N', oa * va, ob * vb, ob * vb, tab_ia_me.data(), W_bbbb.data(), ring12.data(), 1.0,
         1.0);
    std::vector<double> ring2(ob * vb * oa * va, 0.0);
    gemm('N', 'N', ob * vb, oa * va, ob * vb, tbb_jb_me.data(), W_baba.data(), ring2.data(), 1.0,
         0.0);
    gemm('N', 'N', ob * vb, oa * va, oa * va, tab_jb_me.data(), W_aaaa.data(), ring2.data(), 1.0,
         1.0);
    // r2ab ring terms 3 & 4: + t2ab[m, j, a, e] W_abba[m, e, i, b] + t2ab[i, m, e, b] W_baab[m, e, j, a]
    std::vector<double> tab_ja_me(ob * va * oa * vb, 0.0), ring3(ob * va * oa * vb, 0.0);
    for (std::size_t j = 0; j < ob; ++j)
      for (std::size_t a = 0; a < va; ++a)
        for (std::size_t m = 0; m < oa; ++m)
          for (std::size_t e = 0; e < vb; ++e)
            tab_ja_me[((j * va + a) * oa + m) * vb + e] =
                amp.t2ab[((m * ob + j) * va + a) * vb + e];
    gemm('N', 'N', ob * va, oa * vb, oa * vb, tab_ja_me.data(), W_abba.data(), ring3.data(), 1.0,
         0.0);
    std::vector<double> tab_ib_me(oa * vb * ob * va, 0.0), ring4(oa * vb * ob * va, 0.0);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t b = 0; b < vb; ++b)
        for (std::size_t m = 0; m < ob; ++m)
          for (std::size_t e = 0; e < va; ++e)
            tab_ib_me[((i * vb + b) * ob + m) * va + e] =
                amp.t2ab[((i * ob + m) * va + e) * vb + b];
    gemm('N', 'N', oa * vb, ob * va, ob * va, tab_ib_me.data(), W_baab.data(), ring4.data(), 1.0,
         0.0);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b)
            res.t2ab[((i * ob + j) * va + a) * vb + b] +=
                ring12[((i * va + a) * ob + j) * vb + b] + ring2[((j * vb + b) * oa + i) * va + a] +
                ring3[((j * va + a) * oa + i) * vb + b] + ring4[((i * vb + b) * ob + j) * va + a];
  }

  // t1 * t1 * govov, bare t1 * govvv (via X & bov), and -t1 * gooov into r2
  if (oa && va) {
    // bare t1*govvv: P(ij)P(ab) sum_Q x_a[Q,i,a] bov_a[Q,j,b]
    std::vector<double> xb(oa * va * oa * va, 0.0);
    gemm('T', 'N', oa * va, oa * va, q, x_a.data(), src.bov_a.data(), xb.data());
    // t1*t1*govov: P(ij)P(ab) sum_{m,e} t1a[i,e] t1a[m,a] govov_aa[m,b,j,e]
    std::vector<double> tg(oa * va * oa * va, 0.0);
    // first u[m,b,j,i] = sum_e govov_aa[m,b,j,e] t1a[i,e]
    std::vector<double> u(oa * va * oa * oa, 0.0);
    gemm('N', 'T', oa * va * oa, oa, va, core.govov_aa.data(), amp.t1a.data(), u.data());
    // then tg[a, b, j, i] = sum_m t1a[m, a] u[m, b, j, i]
    gemm('T', 'N', va, va * oa * oa, oa, amp.t1a.data(), u.data(), tg.data());
    // -t1*gooov: -P(ab) sum_m t1a[m,a] gooov_aa[i,j,m,b]
    std::vector<double> to(oa * oa * va * va, 0.0);
    for (std::size_t ij = 0; ij < oa * oa; ++ij)
      gemm('T', 'N', va, va, oa, amp.t1a.data(), core.gooov_aa.data() + ij * oa * va,
           to.data() + ij * va * va);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < oa; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < va; ++b) {
            const double xb_t = xb[((i * va + a) * oa + j) * va + b] -
                                xb[((i * va + b) * oa + j) * va + a] -
                                xb[((j * va + a) * oa + i) * va + b] +
                                xb[((j * va + b) * oa + i) * va + a];
            const double tg_t = tg[((a * va + b) * oa + j) * oa + i] -
                                tg[((b * va + a) * oa + j) * oa + i] -
                                tg[((a * va + b) * oa + i) * oa + j] +
                                tg[((b * va + a) * oa + i) * oa + j];
            const double to_t = -to[((i * oa + j) * va + a) * va + b] +
                                to[((i * oa + j) * va + b) * va + a];
            res.t2aa[((i * oa + j) * va + a) * va + b] += xb_t + tg_t + to_t;
          }
  }
  if (ob && vb) {
    std::vector<double> xb(ob * vb * ob * vb, 0.0);
    gemm('T', 'N', ob * vb, ob * vb, q, x_b.data(), src.bov_b.data(), xb.data());
    std::vector<double> u(ob * vb * ob * ob, 0.0), tg(vb * vb * ob * ob, 0.0);
    gemm('N', 'T', ob * vb * ob, ob, vb, core.govov_bb.data(), amp.t1b.data(), u.data());
    gemm('T', 'N', vb, vb * ob * ob, ob, amp.t1b.data(), u.data(), tg.data());
    std::vector<double> to(ob * ob * vb * vb, 0.0);
    for (std::size_t ij = 0; ij < ob * ob; ++ij)
      gemm('T', 'N', vb, vb, ob, amp.t1b.data(), core.gooov_bb.data() + ij * ob * vb,
           to.data() + ij * vb * vb);
    for (std::size_t i = 0; i < ob; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < vb; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            const double xb_t = xb[((i * vb + a) * ob + j) * vb + b] -
                                xb[((i * vb + b) * ob + j) * vb + a] -
                                xb[((j * vb + a) * ob + i) * vb + b] +
                                xb[((j * vb + b) * ob + i) * vb + a];
            const double tg_t = tg[((a * vb + b) * ob + j) * ob + i] -
                                tg[((b * vb + a) * ob + j) * ob + i] -
                                tg[((a * vb + b) * ob + i) * ob + j] +
                                tg[((b * vb + a) * ob + i) * ob + j];
            const double to_t = -to[((i * ob + j) * vb + a) * vb + b] +
                                to[((i * ob + j) * vb + b) * vb + a];
            res.t2bb[((i * ob + j) * vb + a) * vb + b] += xb_t + tg_t + to_t;
          }
  }
  if (oa && ob && va && vb) {
    // bare t1*govvv: + x_a[Q,i,a] bov_b[Q,j,b] + bov_a[Q,i,a] x_b[Q,j,b]
    std::vector<double> xb(oa * va * ob * vb, 0.0);
    gemm('T', 'N', oa * va, ob * vb, q, x_a.data(), src.bov_b.data(), xb.data(), 1.0, 0.0);
    gemm('T', 'N', oa * va, ob * vb, q, src.bov_a.data(), x_b.data(), xb.data(), 1.0, 1.0);
    // -t1*gooov: - t1a[m,a] gooov_ab[i,j,m,b] - t1b[m,b] gooov_ba[j,i,m,a]
    std::vector<double> to1(oa * ob * va * vb, 0.0), to2(ob * oa * vb * va, 0.0);
    for (std::size_t ij = 0; ij < oa * ob; ++ij)
      gemm('T', 'N', va, vb, oa, amp.t1a.data(), core.gooov_ab.data() + ij * oa * vb,
           to1.data() + ij * va * vb);
    for (std::size_t ji = 0; ji < ob * oa; ++ji)
      gemm('T', 'N', vb, va, ob, amp.t1b.data(), core.gooov_ba.data() + ji * ob * va,
           to2.data() + ji * vb * va);
    // t1*t1*govov into r2ab (all 4 terms negative):
    // 1: - t1a[i,e] t1a[m,a] goovv_ab[m,j,e,b]
    // 2: - t1b[j,e] t1b[m,b] goovv_ab[i,m,a,e]
    // 3: - t1b[j,e] t1a[m,a] govov_ab[m,b,i,e]
    // 4: - t1a[i,e] t1b[m,b] govov_ba[m,a,j,e]
    std::vector<double> t1_mjb_i(oa * ob * vb * oa, 0.0);
    for (std::size_t m = 0; m < oa; ++m)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t i = 0; i < oa; ++i)
          for (std::size_t b = 0; b < vb; ++b) {
            double s = 0.0;
            for (std::size_t e = 0; e < va; ++e)
              s += amp.t1a[i * va + e] * core.goovv_ab[((m * ob + j) * va + e) * vb + b];
            t1_mjb_i[((m * ob + j) * vb + b) * oa + i] = s;
          }
    std::vector<double> tg1(va * ob * vb * oa, 0.0);
    gemm('T', 'N', va, ob * vb * oa, oa, amp.t1a.data(), t1_mjb_i.data(), tg1.data());

    std::vector<double> u2(oa * ob * va * ob, 0.0);
    gemm('N', 'T', oa * ob * va, ob, vb, core.goovv_ab.data(), amp.t1b.data(), u2.data());
    std::vector<double> tg2(oa * ob * va * vb, 0.0);
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            double s = 0.0;
            for (std::size_t m = 0; m < ob; ++m)
              s += amp.t1b[m * vb + b] * u2[((i * ob + m) * va + a) * ob + j];
            tg2[((i * ob + j) * va + a) * vb + b] = s;
          }

    std::vector<double> u3(oa * vb * oa * ob, 0.0), tg3(va * vb * oa * ob, 0.0);
    gemm('N', 'T', oa * vb * oa, ob, vb, core.govov_ab.data(), amp.t1b.data(), u3.data());
    gemm('T', 'N', va, vb * oa * ob, oa, amp.t1a.data(), u3.data(), tg3.data());

    std::vector<double> u4(ob * va * ob * oa, 0.0), tg4(vb * va * ob * oa, 0.0);
    gemm('N', 'T', ob * va * ob, oa, va, core.govov_ba.data(), amp.t1a.data(), u4.data());
    gemm('T', 'N', vb, va * ob * oa, ob, amp.t1b.data(), u4.data(), tg4.data());

    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b)
            res.t2ab[((i * ob + j) * va + a) * vb + b] +=
                xb[((i * va + a) * ob + j) * vb + b] -
                to1[((i * ob + j) * va + a) * vb + b] -
                to2[((j * oa + i) * vb + b) * va + a] -
                tg1[((a * ob + j) * vb + b) * oa + i] -
                tg2[((i * ob + j) * va + a) * vb + b] -
                tg3[((a * vb + b) * oa + i) * ob + j] -
                tg4[((b * va + a) * ob + j) * oa + i];
  }

  // Optional FixedF12Coupling hook (zero overhead when nullptr)
  if (f12) {
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t a = 0; a < va; ++a) {
        double s = 0.0;
        for (std::size_t j = 0; j < oa; ++j)
          s += f12->w_vo_aaaa[((i * oa + j) * va + a) * oa + j];
        for (std::size_t j = 0; j < ob; ++j)
          s += f12->w_vo_abab[((i * ob + j) * va + a) * ob + j];
        res.t1a[i * va + a] += s;
      }
    for (std::size_t i = 0; i < ob; ++i)
      for (std::size_t a = 0; a < vb; ++a) {
        double s = 0.0;
        for (std::size_t j = 0; j < ob; ++j)
          s += f12->w_vo_bbbb[((i * ob + j) * vb + a) * ob + j];
        for (std::size_t j = 0; j < oa; ++j)
          s -= f12->w_vo_abba[((j * ob + i) * vb + a) * oa + j];
        res.t1b[i * vb + a] += s;
      }
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < oa; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < va; ++b) {
            const std::size_t idx = ((i * oa + j) * va + a) * va + b;
            double s = f12->w_vv_aaaa[idx] + f12->c_f12_aaaa[idx];
            for (std::size_t k = 0; k < oa; ++k)
              s += -f12->w_vo_aaaa[((i * oa + j) * va + a) * oa + k] * amp.t1a[k * va + b] +
                   f12->w_vo_aaaa[((i * oa + j) * va + b) * oa + k] * amp.t1a[k * va + a];
            res.t2aa[idx] += s;
          }
    for (std::size_t i = 0; i < ob; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < vb; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            const std::size_t idx = ((i * ob + j) * vb + a) * vb + b;
            double s = f12->w_vv_bbbb[idx] + f12->c_f12_bbbb[idx];
            for (std::size_t k = 0; k < ob; ++k)
              s += -f12->w_vo_bbbb[((i * ob + j) * vb + a) * ob + k] * amp.t1b[k * vb + b] +
                   f12->w_vo_bbbb[((i * ob + j) * vb + b) * ob + k] * amp.t1b[k * vb + a];
            res.t2bb[idx] += s;
          }
    for (std::size_t i = 0; i < oa; ++i)
      for (std::size_t j = 0; j < ob; ++j)
        for (std::size_t a = 0; a < va; ++a)
          for (std::size_t b = 0; b < vb; ++b) {
            const std::size_t idx = ((i * ob + j) * va + a) * vb + b;
            double s = f12->w_vv_abab[idx] + f12->c_f12_abab[idx];
            for (std::size_t k = 0; k < ob; ++k)
              s -= f12->w_vo_abab[((i * ob + j) * va + a) * ob + k] * amp.t1b[k * vb + b];
            for (std::size_t k = 0; k < oa; ++k)
              s += f12->w_vo_abba[((i * ob + j) * vb + b) * oa + k] * amp.t1a[k * va + a];
            res.t2ab[idx] += s;
          }
  }
  return res;
}

SpinBlockedCcsdSolveResult solve_df_uccsd_cpu(const SpinBlockedDFSource& src,
                                              const SpinBlockedFourIndexCore& core,
                                              const DFUccV2Options& options) {
  const auto& d = src.dims;
  SpinBlockedCcsdSolveResult out;
  out.amplitudes.allocate_zero(d);

  // Check denominators and initialize from MP2 amplitudes
  const double thresh = options.denominator_threshold;
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t a = 0; a < d.va; ++a) {
      const double den = src.eps_va[a] - src.eps_oa[i];
      if (std::abs(den) < thresh) throw std::runtime_error("DF-UCCSD v2 small alpha 1-body gap");
      out.amplitudes.t1a[i * d.va + a] = -src.fov_a[i * d.va + a] / den;
    }
  for (std::size_t i = 0; i < d.ob; ++i)
    for (std::size_t a = 0; a < d.vb; ++a) {
      const double den = src.eps_vb[a] - src.eps_ob[i];
      if (std::abs(den) < thresh) throw std::runtime_error("DF-UCCSD v2 small beta 1-body gap");
      out.amplitudes.t1b[i * d.vb + a] = -src.fov_b[i * d.vb + a] / den;
    }
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t j = 0; j < d.oa; ++j)
      for (std::size_t a = 0; a < d.va; ++a)
        for (std::size_t b = 0; b < d.va; ++b) {
          const double den = src.eps_va[a] + src.eps_va[b] - src.eps_oa[i] - src.eps_oa[j];
          if (std::abs(den) < thresh) throw std::runtime_error("DF-UCCSD v2 small aa gap");
          const std::size_t idx = ((i * d.oa + j) * d.va + a) * d.va + b;
          out.amplitudes.t2aa[idx] = -core.goovv_aa[idx] / den;
        }
  for (std::size_t i = 0; i < d.ob; ++i)
    for (std::size_t j = 0; j < d.ob; ++j)
      for (std::size_t a = 0; a < d.vb; ++a)
        for (std::size_t b = 0; b < d.vb; ++b) {
          const double den = src.eps_vb[a] + src.eps_vb[b] - src.eps_ob[i] - src.eps_ob[j];
          if (std::abs(den) < thresh) throw std::runtime_error("DF-UCCSD v2 small bb gap");
          const std::size_t idx = ((i * d.ob + j) * d.vb + a) * d.vb + b;
          out.amplitudes.t2bb[idx] = -core.goovv_bb[idx] / den;
        }
  for (std::size_t i = 0; i < d.oa; ++i)
    for (std::size_t j = 0; j < d.ob; ++j)
      for (std::size_t a = 0; a < d.va; ++a)
        for (std::size_t b = 0; b < d.vb; ++b) {
          const double den = src.eps_va[a] + src.eps_vb[b] - src.eps_oa[i] - src.eps_ob[j];
          if (std::abs(den) < thresh) throw std::runtime_error("DF-UCCSD v2 small ab gap");
          const std::size_t idx = ((i * d.ob + j) * d.va + a) * d.vb + b;
          out.amplitudes.t2ab[idx] = -core.goovv_ab[idx] / den;
        }

  HostDiis diis(options.diis_space);
  double prev_e = compute_uccsd_energy_cpu(src, core, out.amplitudes, options.f12);
  bool converged = false;

  for (int iter = 1; iter <= options.max_ccsd_iterations; ++iter) {
    const auto res = compute_uccsd_residual_cpu(src, core, out.amplitudes, options.f12);
    const double rmax = max_abs_residual(res);
    const double cur_e = compute_uccsd_energy_cpu(src, core, out.amplitudes, options.f12);
    if (!std::isfinite(rmax) || !std::isfinite(cur_e))
      throw std::runtime_error("DF-UCCSD v2 CPU residual/energy became nonfinite");
    out.iterations = iter;
    out.energy = cur_e;
    if (rmax < options.residual_tolerance && std::abs(cur_e - prev_e) < options.energy_tolerance) {
      converged = true;
      break;
    }
    prev_e = cur_e;

    SpinBlockedUccAmplitudes step;
    step.allocate_zero(d);
    for (std::size_t i = 0; i < d.oa; ++i)
      for (std::size_t a = 0; a < d.va; ++a) {
        const std::size_t idx = i * d.va + a;
        const double dt = -res.t1a[idx] / (src.eps_va[a] - src.eps_oa[i]);
        step.t1a[idx] = dt;
        out.amplitudes.t1a[idx] += dt;
      }
    for (std::size_t i = 0; i < d.ob; ++i)
      for (std::size_t a = 0; a < d.vb; ++a) {
        const std::size_t idx = i * d.vb + a;
        const double dt = -res.t1b[idx] / (src.eps_vb[a] - src.eps_ob[i]);
        step.t1b[idx] = dt;
        out.amplitudes.t1b[idx] += dt;
      }
    for (std::size_t i = 0; i < d.oa; ++i)
      for (std::size_t j = 0; j < d.oa; ++j)
        for (std::size_t a = 0; a < d.va; ++a)
          for (std::size_t b = 0; b < d.va; ++b) {
            const std::size_t idx = ((i * d.oa + j) * d.va + a) * d.va + b;
            const double dt =
                -res.t2aa[idx] / (src.eps_va[a] + src.eps_va[b] - src.eps_oa[i] - src.eps_oa[j]);
            step.t2aa[idx] = dt;
            out.amplitudes.t2aa[idx] += dt;
          }
    for (std::size_t i = 0; i < d.ob; ++i)
      for (std::size_t j = 0; j < d.ob; ++j)
        for (std::size_t a = 0; a < d.vb; ++a)
          for (std::size_t b = 0; b < d.vb; ++b) {
            const std::size_t idx = ((i * d.ob + j) * d.vb + a) * d.vb + b;
            const double dt =
                -res.t2bb[idx] / (src.eps_vb[a] + src.eps_vb[b] - src.eps_ob[i] - src.eps_ob[j]);
            step.t2bb[idx] = dt;
            out.amplitudes.t2bb[idx] += dt;
          }
    for (std::size_t i = 0; i < d.oa; ++i)
      for (std::size_t j = 0; j < d.ob; ++j)
        for (std::size_t a = 0; a < d.va; ++a)
          for (std::size_t b = 0; b < d.vb; ++b) {
            const std::size_t idx = ((i * d.ob + j) * d.va + a) * d.vb + b;
            const double dt =
                -res.t2ab[idx] / (src.eps_va[a] + src.eps_vb[b] - src.eps_oa[i] - src.eps_ob[j]);
            step.t2ab[idx] = dt;
            out.amplitudes.t2ab[idx] += dt;
          }

    std::vector<double> flat_t, flat_err, extrap;
    flatten_amplitudes(out.amplitudes, flat_t);
    flatten_amplitudes(step, flat_err);
    diis.push(flat_t, flat_err);
    if (diis.extrapolate(extrap)) unflatten_amplitudes(extrap, d, out.amplitudes);
  }
  if (!converged) throw std::runtime_error("DF-UCCSD v2 CPU solver did not converge");
  return out;
}

}  // namespace generativeqc::cc::v2_detail
