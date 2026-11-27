// Qualification and production adapter for contracted, real molecular DF and F12 shell values and adjoints.
// Cartesian ordering is CCA. Callers own shell-to-atom and pure-to-Cartesian maps.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "integrals/f12_geminal.hpp"

#ifdef GQC_SINK_COULOMB
#include "molecular-coulomb.hpp"
#ifdef GQC_F12_CABS_I
namespace poly = generativeqc::scf::generated_df_auxiliary_i_derivatives;
#else
namespace poly = generativeqc::scf::generated_df_auxiliary_h_derivatives;
#endif
#else
#include "molecular-geminal.hpp"
#ifdef GQC_F12_CABS_I
namespace poly = generativeqc::scf::generated_f12_df_auxiliary_i_derivatives;
#else
namespace poly = generativeqc::scf::generated_f12_df_auxiliary_h_derivatives;
#endif
#endif
namespace gi = generativeqc::integrals;

#if defined(GQC_SINK_HOST_REFERENCE)
#define SINK_HD
#define SINK_DEVICE
#define SINK_NOINLINE
#ifdef GQC_SINK_COULOMB
#define SINK_EXPORT gqc_f12_weighted_shells_coulomb_host_reference
#define FORWARD_EXPORT gqc_f12_cartesian_shells_coulomb_host_reference
#define SINK_EX_EXPORT gqc_f12_weighted_shells_coulomb_ex_host_reference
#define FORWARD_EX_EXPORT gqc_f12_cartesian_shells_coulomb_ex_host_reference
#define SPH_FORWARD_EXPORT gqc_f12_spherical_forward_coulomb_host_reference
#define SPH_WEIGHTED_EXPORT gqc_f12_spherical_weighted_coulomb_host_reference
#else
#define SINK_EXPORT gqc_f12_weighted_shells_geminal_host_reference
#define FORWARD_EXPORT gqc_f12_cartesian_shells_geminal_host_reference
#define SINK_EX_EXPORT gqc_f12_weighted_shells_geminal_ex_host_reference
#define FORWARD_EX_EXPORT gqc_f12_cartesian_shells_geminal_ex_host_reference
#define SPH_FORWARD_EXPORT gqc_f12_spherical_forward_geminal_host_reference
#define SPH_WEIGHTED_EXPORT gqc_f12_spherical_weighted_geminal_host_reference
#endif
#elif defined(__CUDACC__)
#define SINK_HD __host__ __device__
#define SINK_DEVICE __device__
#define SINK_NOINLINE __noinline__
#ifdef GQC_SINK_COULOMB
#define SINK_EXPORT gqc_f12_weighted_shells_coulomb
#define FORWARD_EXPORT gqc_f12_cartesian_shells_coulomb
#define SINK_EX_EXPORT gqc_f12_weighted_shells_coulomb_ex
#define FORWARD_EX_EXPORT gqc_f12_cartesian_shells_coulomb_ex
#define SPH_FORWARD_EXPORT gqc_f12_spherical_forward_coulomb
#define SPH_WEIGHTED_EXPORT gqc_f12_spherical_weighted_coulomb
#else
#define SINK_EXPORT gqc_f12_weighted_shells_geminal
#define FORWARD_EXPORT gqc_f12_cartesian_shells_geminal
#define SINK_EX_EXPORT gqc_f12_weighted_shells_geminal_ex
#define FORWARD_EX_EXPORT gqc_f12_cartesian_shells_geminal_ex
#define SPH_FORWARD_EXPORT gqc_f12_spherical_forward_geminal
#define SPH_WEIGHTED_EXPORT gqc_f12_spherical_weighted_geminal
#endif
#else
#error "Use NVCC for the CUDA sink or explicitly build the separate host reference"
#endif

namespace {
#if defined(GQC_F12_CABS_I)
constexpr int maximum_combined_l = 6;
constexpr int maximum_fitting_l = 6;
#elif defined(GQC_F12_CABS_H)
constexpr int maximum_combined_l = 5;
constexpr int maximum_fitting_l = 5;
#else
constexpr int maximum_combined_l = 4;
constexpr int maximum_fitting_l = 5;
#endif
constexpr int channels = 10;
constexpr int threads = 64;
constexpr int kPrimBatch = 8;
constexpr int kTransformTableSize = 840;  // sum_{l=0..6} (2*l+1)*(l+1)*(l+2)/2
using Index = std::int32_t;
using Size = std::uint64_t;

SINK_HD inline int cart_size(int l) { return (l + 1) * (l + 2) / 2; }
SINK_HD inline int sph_size(int l) { return 2 * l + 1; }

SINK_HD inline int transform_offset(int l) {
  constexpr int offsets[7] = {0, 1, 10, 40, 110, 245, 476};
  return offsets[l];
}

constexpr int kDefaultSlaterFitNormalizationCode = 1;

SINK_HD inline bool decode_slater_normalization(int code, gi::SlaterFitNormalization& out) {
  if (code == 0) {
    out = gi::SlaterFitNormalization::negative_exponential;
    return true;
  }
  if (code == 1) {
    out = gi::SlaterFitNormalization::negative_exponential_over_beta;
    return true;
  }
  return false;
}

struct ShellBatch {
  const Index *angular, *primitive_offsets, *block_shells, *weight_offsets;
  const double *exponents, *radial, *centers, *weights;
};

SINK_DEVICE poly::Angular component(int l, int index) {
  int degree_yz = 0;
  while (index > degree_yz) index -= ++degree_yz;
  return {static_cast<unsigned>(l - degree_yz), static_cast<unsigned>(degree_yz - index),
          static_cast<unsigned>(index)};
}

SINK_DEVICE poly::Vec3 load_center(const double* centers, Index shell) {
  const auto* xyz = centers + Size(3) * shell;
  return {xyz[0], xyz[1], xyz[2]};
}

SINK_DEVICE inline bool pair_decay_zero(double alpha, poly::Vec3 A, double beta, poly::Vec3 B) {
  const double dx = A.x - B.x;
  const double dy = A.y - B.y;
  const double dz = A.z - B.z;
  const double ab2 = dx * dx + dy * dy + dz * dz;
  // IEEE-754 binary64 exp(-x) flushes identically to 0.0 for x >= 746.0.
  return (alpha * beta / (alpha + beta)) * ab2 > 746.0;
}

struct ShellBlockMeta {
  bool metric;
  Index sa, sb, sc;
  poly::Vec3 A, B, C;
  int la, lb, lc, ncart, first_a, first_b, first_c, count_a, count_b, count_c;
  Size start;

  SINK_DEVICE void init_shells(bool is_metric, Index shell_a, Index shell_b, Index shell_c,
                               Size block_start, const Index* angular,
                               const Index* primitive_offsets, const double* centers) {
    metric = is_metric;
    sa = shell_a;
    sb = is_metric ? shell_a : shell_b;
    sc = shell_c;
    A = load_center(centers, sa);
    B = load_center(centers, sb);
    C = load_center(centers, sc);
    la = angular[sa];
    lb = is_metric ? 0 : angular[sb];
    lc = angular[sc];
    start = block_start;
    ncart = cart_size(la) * cart_size(lb) * cart_size(lc);
    first_a = primitive_offsets[sa];
    first_b = primitive_offsets[sb];
    first_c = primitive_offsets[sc];
    count_a = primitive_offsets[sa + 1] - first_a;
    count_b = is_metric ? 1 : (primitive_offsets[sb + 1] - first_b);
    count_c = primitive_offsets[sc + 1] - first_c;
  }

  SINK_DEVICE void init_from_batch(const ShellBatch& data, int ncenter, int block) {
    const auto* shells = data.block_shells + Size(block) * ncenter;
    const bool is_metric = (ncenter == 2);
    init_shells(is_metric, shells[0], is_metric ? shells[0] : shells[1], shells[ncenter - 1],
                static_cast<Size>(data.weight_offsets[block]), data.angular,
                data.primitive_offsets, data.centers);
  }
};

SINK_DEVICE SINK_NOINLINE bool prepare_shell_geometry(const ShellBlockMeta& shell, double alpha,
                                                      double beta_exp, double gamma_exp, int op,
                                                      const gi::FittedSlater& fit,
                                                      poly::Geometry& geom) {
  const unsigned total = static_cast<unsigned>(shell.la + shell.lb + shell.lc);
#ifdef GQC_SINK_COULOMB
  (void)op;
  (void)fit;
  poly::prepare_geometry(alpha, shell.A, beta_exp, shell.B, gamma_exp, shell.C, total, geom,
                         nullptr, 0.0);
  return true;
#else
  return poly::prepare_geometry(alpha, shell.A, beta_exp, shell.B, gamma_exp, shell.C, total, geom,
                                static_cast<gi::GeminalProduct>(op), fit);
#endif
}

template <bool Responses>
SINK_DEVICE SINK_NOINLINE poly::Response evaluate_from_geometry(const ShellBlockMeta& shell, int cart,
                                                                double alpha, double beta_exp,
                                                                const poly::Geometry& geom) {
  constexpr int max_slots = static_cast<int>(sizeof(poly::Geometry::f) / sizeof(double));
  const int nb = cart_size(shell.lb);
  const int nc = cart_size(shell.lc);
  const auto a = component(shell.la, cart / (nc * nb));
  const auto b = component(shell.lb, (cart / nc) % nb);
  const auto c = component(shell.lc, cart % nc);

  const double sx = geom.sx, sy = geom.sy, ip = geom.ip, iq = geom.iq;
  const double prefactor = geom.prefactor;
  const auto& f = geom.f;

  double base[3][max_slots];
  unsigned d[3];
  for (unsigned axis = 0; axis < 3; ++axis) {
    const unsigned na = poly::power(a, axis);
    const unsigned nb_a = poly::power(b, axis);
    const unsigned nc_a = poly::power(c, axis);
    d[axis] = na + nb_a + nc_a;
    poly::axis_polynomial(na, nb_a, nc_a, geom.pa[axis], geom.pb[axis], geom.dx[axis], sx, sy, ip,
                          iq, base[axis]);
  }
  const unsigned d0 = d[0], d1 = d[1], d2 = d[2];
  poly::Response result{};

  // Preserve exact poly::dot summation order for the tiny [Q, Q] metric so the
  // ill-conditioned RI metric whitener M^(-1/2) matches the pre-fusion sink bit-for-bit.
  if (shell.metric) {
    result.value = prefactor * poly::dot(d0, base[0], d1, base[1], d2, base[2], f);
    if constexpr (!Responses) return result;
    double first[3]{};
    for (unsigned axis = 0; axis < 3; ++axis) {
      const unsigned other = (axis + 1) % 3, last = (axis + 2) % 3;
      const unsigned na = poly::power(a, axis), nb_a = poly::power(b, axis),
                     nc_a = poly::power(c, axis), deg = d[axis];
      double raised[max_slots], lowered[max_slots];
      poly::axis_polynomial(na + 1, nb_a, nc_a, geom.pa[axis], geom.pb[axis], geom.dx[axis], sx,
                            sy, ip, iq, raised);
      if (na)
        poly::axis_polynomial(na - 1, nb_a, nc_a, geom.pa[axis], geom.pb[axis], geom.dx[axis], sx,
                              sy, ip, iq, lowered);
      for (unsigned i = 0; i <= deg + 1U; ++i)
        raised[i] = 2.0 * alpha * raised[i] - (na && i < deg ? na * lowered[i] : 0.0);
      first[axis] = prefactor * poly::dot(deg + 1, raised, d[other], base[other], d[last],
                                          base[last], f);
    }
    result.first = {first[0], first[1], first[2]};
    result.second = {0.0, 0.0, 0.0};
    result.third = {-first[0], -first[1], -first[2]};
    return result;
  }

  // Factorize 3D Boys/geminal contraction: contract z first, then y (and x for derivatives).
  const unsigned gz_max = Responses ? (d0 + d1 + 1U) : (d0 + d1);
  double gz[max_slots];
  for (unsigned s = 0; s <= gz_max; ++s) {
    double acc = 0.0;
    for (unsigned k = 0; k <= d2; ++k) acc += base[2][k] * f[s + k];
    gz[s] = acc;
  }

  const unsigned fx_max = Responses ? (d0 + 1U) : d0;
  double fx[max_slots];
  for (unsigned u = 0; u <= fx_max; ++u) {
    double acc = 0.0;
    for (unsigned j = 0; j <= d1; ++j) acc += base[1][j] * gz[u + j];
    fx[u] = acc;
  }

  double val = 0.0;
  for (unsigned i = 0; i <= d0; ++i) val += base[0][i] * fx[i];
  result.value = prefactor * val;
  if constexpr (!Responses) return result;

  double fy[max_slots];
  for (unsigned v = 0; v <= d1 + 1U; ++v) {
    double acc = 0.0;
    for (unsigned i = 0; i <= d0; ++i) acc += base[0][i] * gz[v + i];
    fy[v] = acc;
  }

  double gxy[max_slots]{};
  for (unsigned i = 0; i <= d0; ++i) {
    const double bi = base[0][i];
    for (unsigned j = 0; j <= d1; ++j) gxy[i + j] += bi * base[1][j];
  }

  double fz[max_slots];
  for (unsigned w = 0; w <= d2 + 1U; ++w) {
    double acc = 0.0;
    for (unsigned s = 0; s <= d0 + d1; ++s) acc += gxy[s] * f[s + w];
    fz[w] = acc;
  }

  const double* f_axis[3] = {fx, fy, fz};
  double first[3]{}, second[3]{};
  for (unsigned axis = 0; axis < 3; ++axis) {
    const unsigned na = poly::power(a, axis);
    const unsigned nb_a = poly::power(b, axis);
    const unsigned nc_a = poly::power(c, axis);
    const unsigned deg = d[axis];
    const double* fa = f_axis[axis];
    double raised[max_slots], lowered[max_slots];
    for (unsigned center = 0; center < (shell.metric ? 1U : 2U); ++center) {
      const unsigned n = (center == 0) ? na : nb_a;
      poly::axis_polynomial(na + (center == 0), nb_a + (center == 1), nc_a, geom.pa[axis],
                            geom.pb[axis], geom.dx[axis], sx, sy, ip, iq, raised);
      if (n) {
        poly::axis_polynomial(na - (center == 0), nb_a - (center == 1), nc_a, geom.pa[axis],
                              geom.pb[axis], geom.dx[axis], sx, sy, ip, iq, lowered);
      }
      const double two_exp = 2.0 * ((center == 0) ? alpha : beta_exp);
      double acc = 0.0;
      for (unsigned i = 0; i <= deg + 1U; ++i) {
        const double r_i = two_exp * raised[i] - (n && i < deg ? n * lowered[i] : 0.0);
        acc += r_i * fa[i];
      }
      const double response = prefactor * acc;
      if (center == 0)
        first[axis] = response;
      else
        second[axis] = response;
    }
  }
  result.first = {first[0], first[1], first[2]};
  result.second = {second[0], second[1], second[2]};
  result.third = {-first[0] - second[0], -first[1] - second[1], -first[2] - second[2]};
  return result;
}

SINK_DEVICE inline void compensated_add(double x, double& sum, double& correction) {
  const double adjusted = x - correction;
  const double next = sum + adjusted;
  correction = (next - sum) - adjusted;
  sum = next;
}

#ifdef GQC_SINK_HOST_REFERENCE
void cartesian_values_host(const ShellBlockMeta& shell, const double* exponents,
                           const double* radial, int op, const gi::FittedSlater& fit,
                           double* out_ptr) {
  std::fill_n(out_ptr, shell.ncart, 0.0);
  std::vector<double> corr(static_cast<std::size_t>(shell.ncart), 0.0);
  for (int di = 0; di < shell.count_a; ++di) {
    const int i = shell.first_a + di;
    const double alpha = exponents[i];
    for (int dj = 0; dj < shell.count_b; ++dj) {
      const int j = shell.first_b + dj;
      const double beta_exp = shell.metric ? 0.0 : exponents[j];
      const double ab = radial[i] * (shell.metric ? 1.0 : radial[j]);
      if (ab == 0.0) continue;
      if (!shell.metric && pair_decay_zero(alpha, shell.A, beta_exp, shell.B)) continue;
      for (int dk = 0; dk < shell.count_c; ++dk) {
        const int k = shell.first_c + dk;
        const double coeff = ab * radial[k];
        if (coeff == 0.0) continue;
        poly::Geometry geom;
        if (!prepare_shell_geometry(shell, alpha, beta_exp, exponents[k], op, fit, geom)) {
          std::fill_n(out_ptr, shell.ncart, NAN);
          return;
        }
        for (int cart = 0; cart < shell.ncart; ++cart) {
          const double value =
              evaluate_from_geometry<false>(shell, cart, alpha, beta_exp, geom).value;
          compensated_add(coeff * value, out_ptr[cart], corr[cart]);
        }
      }
    }
  }
}

void weighted_terms_host(const ShellBlockMeta& shell, const double* exponents, const double* radial,
                         const double* weights, int op, const gi::FittedSlater& fit,
                         double (&sum)[channels]) {
  bool any_nz = false;
  for (int cart = 0; cart < shell.ncart; ++cart) {
    if (weights[cart] != 0.0) {
      any_nz = true;
      break;
    }
  }
  if (!any_nz) return;
  double correction[channels]{};
  for (int di = 0; di < shell.count_a; ++di) {
    const int i = shell.first_a + di;
    const double alpha = exponents[i];
    for (int dj = 0; dj < shell.count_b; ++dj) {
      const int j = shell.first_b + dj;
      const double beta_exp = shell.metric ? 0.0 : exponents[j];
      const double ab = radial[i] * (shell.metric ? 1.0 : radial[j]);
      if (ab == 0.0) continue;
      if (!shell.metric && pair_decay_zero(alpha, shell.A, beta_exp, shell.B)) continue;
      for (int dk = 0; dk < shell.count_c; ++dk) {
        const int k = shell.first_c + dk;
        const double coeff = ab * radial[k];
        if (coeff == 0.0) continue;
        poly::Geometry geom;
        if (!prepare_shell_geometry(shell, alpha, beta_exp, exponents[k], op, fit, geom)) {
          sum[0] = NAN;
          return;
        }
        for (int cart = 0; cart < shell.ncart; ++cart) {
          const double w = weights[cart] * coeff;
          if (w == 0.0) continue;
          const auto r = evaluate_from_geometry<true>(shell, cart, alpha, beta_exp, geom);
          const double values[channels] = {r.value,    r.first.x,  r.first.y, r.first.z, r.second.x,
                                           r.second.y, r.second.z, r.third.x, r.third.y, r.third.z};
          for (int q = 0; q < channels; ++q) compensated_add(w * values[q], sum[q], correction[q]);
        }
      }
    }
  }
}
#else
// CUDA block-cooperative forward evaluation reusing hoisted primitive Geometry.
__device__ __noinline__ void cartesian_block_device(const ShellBlockMeta& shell, const double* exponents,
                                                    const double* radial, int op, const gi::FittedSlater& fit,
                                                    double* out_ptr) {
  __shared__ poly::Geometry s_geom[kPrimBatch];
  __shared__ double s_coeff[kPrimBatch];
  __shared__ double s_alpha[kPrimBatch];
  __shared__ double s_beta[kPrimBatch];

  const int lane = threadIdx.x;
  const Size nprim = Size(shell.count_a) * shell.count_b * shell.count_c;

  if (shell.ncart <= threads) {
    double acc = 0.0, corr = 0.0;
    for (Size p0 = 0; p0 < nprim; p0 += kPrimBatch) {
      const int batch_n = static_cast<int>((nprim - p0) < kPrimBatch ? (nprim - p0) : kPrimBatch);
      if (lane < batch_n) {
        const Size prim = p0 + lane;
        const int k = shell.first_c + static_cast<int>(prim % shell.count_c);
        const int j = shell.first_b + static_cast<int>((prim / shell.count_c) % shell.count_b);
        const int i = shell.first_a + static_cast<int>(prim / (Size(shell.count_c) * shell.count_b));
        const double coeff = radial[i] * (shell.metric ? 1.0 : radial[j]) * radial[k];
        const double alpha = exponents[i];
        const double beta_e = shell.metric ? 0.0 : exponents[j];
        s_coeff[lane] = coeff;
        s_alpha[lane] = alpha;
        s_beta[lane] = beta_e;
        if (coeff != 0.0) {
          if (!shell.metric && pair_decay_zero(alpha, shell.A, beta_e, shell.B)) {
            s_coeff[lane] = 0.0;
          } else if (!prepare_shell_geometry(shell, alpha, beta_e, exponents[k], op, fit,
                                             s_geom[lane])) {
            s_geom[lane].prefactor = NAN;
          }
        }
      }
      __syncthreads();
      if (lane < shell.ncart) {
        for (int b = 0; b < batch_n; ++b) {
          const double coeff = s_coeff[b];
          if (coeff == 0.0) continue;
          const double val =
              evaluate_from_geometry<false>(shell, lane, s_alpha[b], s_beta[b], s_geom[b]).value;
          compensated_add(coeff * val, acc, corr);
        }
      }
      __syncthreads();
    }
    if (lane < shell.ncart) out_ptr[lane] = acc;
  } else {
    if (nprim > 1) {
      for (int cart = lane; cart < shell.ncart; cart += threads) out_ptr[cart] = 0.0;
    }
    for (Size p0 = 0; p0 < nprim; p0 += kPrimBatch) {
      const int batch_n = static_cast<int>((nprim - p0) < kPrimBatch ? (nprim - p0) : kPrimBatch);
      if (lane < batch_n) {
        const Size prim = p0 + lane;
        const int k = shell.first_c + static_cast<int>(prim % shell.count_c);
        const int j = shell.first_b + static_cast<int>((prim / shell.count_c) % shell.count_b);
        const int i = shell.first_a + static_cast<int>(prim / (Size(shell.count_c) * shell.count_b));
        const double coeff = radial[i] * (shell.metric ? 1.0 : radial[j]) * radial[k];
        const double alpha = exponents[i];
        const double beta_e = shell.metric ? 0.0 : exponents[j];
        s_coeff[lane] = coeff;
        s_alpha[lane] = alpha;
        s_beta[lane] = beta_e;
        if (coeff != 0.0) {
          if (!shell.metric && pair_decay_zero(alpha, shell.A, beta_e, shell.B)) {
            s_coeff[lane] = 0.0;
          } else if (!prepare_shell_geometry(shell, alpha, beta_e, exponents[k], op, fit,
                                             s_geom[lane])) {
            s_geom[lane].prefactor = NAN;
          }
        }
      }
      __syncthreads();
      if (nprim == 1) {
        const double coeff = s_coeff[0];
        if (coeff == 0.0) {
          for (int cart = lane; cart < shell.ncart; cart += threads) out_ptr[cart] = 0.0;
        } else {
          for (int cart = lane; cart < shell.ncart; cart += threads) {
            out_ptr[cart] =
                coeff *
                evaluate_from_geometry<false>(shell, cart, s_alpha[0], s_beta[0], s_geom[0]).value;
          }
        }
      } else {
        for (int b = 0; b < batch_n; ++b) {
          const double coeff = s_coeff[b];
          if (coeff == 0.0) continue;
          const double alpha = s_alpha[b];
          const double beta_e = s_beta[b];
          for (int cart = lane; cart < shell.ncart; cart += threads) {
            out_ptr[cart] +=
                coeff * evaluate_from_geometry<false>(shell, cart, alpha, beta_e, s_geom[b]).value;
          }
        }
      }
      __syncthreads();
    }
  }
}

// CUDA block-cooperative weighted derivative evaluation reusing hoisted primitive Geometry.
__device__ __noinline__ bool weighted_block_device(const ShellBlockMeta& shell, const double* exponents,
                                                   const double* radial, const double* weights, int op,
                                                   const gi::FittedSlater& fit, double (&reduced)[channels]) {
  __shared__ int s_any_nz;
  __shared__ poly::Geometry s_geom[kPrimBatch];
  __shared__ double s_coeff[kPrimBatch];
  __shared__ double s_alpha[kPrimBatch];
  __shared__ double s_beta[kPrimBatch];
  __shared__ double partial[channels][threads];

  const int lane = threadIdx.x;
  if (lane == 0) s_any_nz = 0;
  __syncthreads();
  int local_nz = 0;
  for (int cart = lane; cart < shell.ncart; cart += threads) {
    if (weights[cart] != 0.0) {
      local_nz = 1;
      break;
    }
  }
  if (local_nz) s_any_nz = 1;
  __syncthreads();
  if (!s_any_nz) {
    if (lane == 0) {
      for (int q = 0; q < channels; ++q) reduced[q] = 0.0;
    }
    return false;
  }

  const Size nprim = Size(shell.count_a) * shell.count_b * shell.count_c;
  double sum[channels]{}, correction[channels]{};

  if (shell.ncart < 16) {
    for (Size prim = lane; prim < nprim; prim += threads) {
      const int k = shell.first_c + static_cast<int>(prim % shell.count_c);
      const int j = shell.first_b + static_cast<int>((prim / shell.count_c) % shell.count_b);
      const int i = shell.first_a + static_cast<int>(prim / (Size(shell.count_c) * shell.count_b));
      const double coeff = radial[i] * (shell.metric ? 1.0 : radial[j]) * radial[k];
      if (coeff == 0.0) continue;
      const double alpha = exponents[i];
      const double beta_e = shell.metric ? 0.0 : exponents[j];
      if (!shell.metric && pair_decay_zero(alpha, shell.A, beta_e, shell.B)) continue;
      poly::Geometry geom;
      if (!prepare_shell_geometry(shell, alpha, beta_e, exponents[k], op, fit, geom)) {
        sum[0] = NAN;
        continue;
      }
      for (int cart = 0; cart < shell.ncart; ++cart) {
        const double w = weights[cart] * coeff;
        if (w == 0.0) continue;
        const auto r = evaluate_from_geometry<true>(shell, cart, alpha, beta_e, geom);
        const double values[channels] = {r.value,    r.first.x,  r.first.y, r.first.z, r.second.x,
                                         r.second.y, r.second.z, r.third.x, r.third.y, r.third.z};
        for (int q = 0; q < channels; ++q) compensated_add(w * values[q], sum[q], correction[q]);
      }
    }
  } else {
    for (Size p0 = 0; p0 < nprim; p0 += kPrimBatch) {
      const int batch_n = static_cast<int>((nprim - p0) < kPrimBatch ? (nprim - p0) : kPrimBatch);
      if (lane < batch_n) {
        const Size prim = p0 + lane;
        const int k = shell.first_c + static_cast<int>(prim % shell.count_c);
        const int j = shell.first_b + static_cast<int>((prim / shell.count_c) % shell.count_b);
        const int i = shell.first_a + static_cast<int>(prim / (Size(shell.count_c) * shell.count_b));
        const double coeff = radial[i] * (shell.metric ? 1.0 : radial[j]) * radial[k];
        const double alpha = exponents[i];
        const double beta_e = shell.metric ? 0.0 : exponents[j];
        s_coeff[lane] = coeff;
        s_alpha[lane] = alpha;
        s_beta[lane] = beta_e;
        if (coeff != 0.0) {
          if (!shell.metric && pair_decay_zero(alpha, shell.A, beta_e, shell.B)) {
            s_coeff[lane] = 0.0;
          } else if (!prepare_shell_geometry(shell, alpha, beta_e, exponents[k], op, fit,
                                             s_geom[lane])) {
            s_geom[lane].prefactor = NAN;
          }
        }
      }
      __syncthreads();
      for (int b = 0; b < batch_n; ++b) {
        const double coeff = s_coeff[b];
        if (coeff == 0.0) continue;
        const double alpha = s_alpha[b];
        const double beta_e = s_beta[b];
        for (int cart = lane; cart < shell.ncart; cart += threads) {
          const double w = weights[cart] * coeff;
          if (w == 0.0) continue;
          const auto r = evaluate_from_geometry<true>(shell, cart, alpha, beta_e, s_geom[b]);
          const double values[channels] = {r.value,    r.first.x,  r.first.y, r.first.z, r.second.x,
                                           r.second.y, r.second.z, r.third.x, r.third.y, r.third.z};
          for (int q = 0; q < channels; ++q) compensated_add(w * values[q], sum[q], correction[q]);
        }
      }
      __syncthreads();
    }
  }

  for (int q = 0; q < channels; ++q) partial[q][lane] = sum[q];
  __syncthreads();
  for (int step = threads / 2; step; step /= 2) {
    if (lane < step) {
      for (int q = 0; q < channels; ++q) partial[q][lane] += partial[q][lane + step];
    }
    __syncthreads();
  }
  if (lane == 0) {
    for (int q = 0; q < channels; ++q) reduced[q] = partial[q][0];
  }
  return true;
}

__global__ void cartesian_blocks(ShellBatch data, int ncenter, int op, gi::FittedSlater fit,
                                 double* result) {
  ShellBlockMeta shell;
  shell.init_from_batch(data, ncenter, blockIdx.x);
  cartesian_block_device(shell, data.exponents, data.radial, op, fit, result + shell.start);
}

__global__ void weighted_blocks(ShellBatch data, int ncenter, int op, gi::FittedSlater fit,
                                double* result) {
  ShellBlockMeta shell;
  shell.init_from_batch(data, ncenter, blockIdx.x);
  double reduced[channels];
  weighted_block_device(shell, data.exponents, data.radial, data.weights + shell.start, op, fit,
                        reduced);
  if (threadIdx.x == 0) {
    for (int q = 0; q < channels; ++q) result[Size(blockIdx.x) * channels + q] = reduced[q];
  }
}

// =========================================================================
// GPU-Fused Spherical <-> Cartesian Transform & Slabbed Integral Kernels
// =========================================================================

// Metric (2-center) pack: spherical [naux_sph, naux_sph] -> packed pair Cartesian
__global__ void metric_pack_kernel(int n_pairs, int naux_sph, int orb_shell_count,
                                   const Index* pair_sa, const Index* pair_sc,
                                   const Index* pair_cart_offsets, const Index* angular,
                                   const Index* sph_offsets, const double* transform_table,
                                   const double* d_sph, double* d_cart, int* d_nonfinite) {
  const int p = blockIdx.x;
  if (p >= n_pairs) return;
  const int sa = pair_sa[p];
  const int sc = pair_sc[p];
  const int la = angular[sa], lc = angular[sc];
  const int ma = sph_size(la), mc = sph_size(lc);
  const int na = cart_size(la), nc = cart_size(lc);
  const int a0 = sph_offsets[sa], c0 = sph_offsets[sc];
  const double* Ta = transform_table + transform_offset(la);
  const double* Tc = transform_table + transform_offset(lc);
  double* dst = d_cart + pair_cart_offsets[p];
  const bool diff = (sa != sc);

  for (int idx = threadIdx.x; idx < na * nc; idx += blockDim.x) {
    const int a_c = idx / nc, c_c = idx % nc;
    double acc = 0.0;
    for (int a = 0; a < ma; ++a) {
      const double ta = Ta[a * na + a_c];
      double inner = 0.0;
      for (int c = 0; c < mc; ++c) {
        const double w1 = d_sph[Size(a0 + a) * naux_sph + (c0 + c)];
        const double w = diff ? (w1 + d_sph[Size(c0 + c) * naux_sph + (a0 + a)]) : w1;
        if (!isfinite(w)) atomicExch(d_nonfinite, 1);
        inner += w * Tc[c * nc + c_c];
      }
      acc += ta * inner;
    }
    dst[idx] = acc;
  }
}

// Metric (2-center) unpack: packed pair Cartesian -> spherical [naux_sph, naux_sph]
__global__ void metric_unpack_kernel(int n_pairs, int naux_sph, const Index* pair_sa,
                                     const Index* pair_sc, const Index* pair_cart_offsets,
                                     const Index* angular, const Index* sph_offsets,
                                     const double* transform_table, const double* d_cart,
                                     double* d_sph, int* d_nonfinite) {
  const int p = blockIdx.x;
  if (p >= n_pairs) return;
  const int sa = pair_sa[p];
  const int sc = pair_sc[p];
  const int la = angular[sa], lc = angular[sc];
  const int ma = sph_size(la), mc = sph_size(lc);
  const int na = cart_size(la), nc = cart_size(lc);
  const int a0 = sph_offsets[sa], c0 = sph_offsets[sc];
  const double* Ta = transform_table + transform_offset(la);
  const double* Tc = transform_table + transform_offset(lc);
  const double* src = d_cart + pair_cart_offsets[p];
  const bool diff = (sa != sc);

  for (int idx = threadIdx.x; idx < ma * mc; idx += blockDim.x) {
    const int a = idx / mc, c = idx % mc;
    double acc = 0.0;
    for (int a_c = 0; a_c < na; ++a_c) {
      const double ta = Ta[a * na + a_c];
      double inner = 0.0;
      for (int c_c = 0; c_c < nc; ++c_c) {
        inner += src[a_c * nc + c_c] * Tc[c * nc + c_c];
      }
      acc += ta * inner;
    }
    if (!isfinite(acc)) atomicExch(d_nonfinite, 1);
    d_sph[Size(a0 + a) * naux_sph + (c0 + c)] = acc;
    if (diff) d_sph[Size(c0 + c) * naux_sph + (a0 + a)] = acc;
  }
}

// 3-center Pack Step 1: fold lower-triangle symmetry + transform orbital index b -> B
__global__ void ao_pack_b_kernel(int q_sph, int norb_sph, int norb_cart, const Index* angular,
                                 const Index* sph_offsets, const Index* orb_sph_to_shell,
                                 const Index* orb_cart_to_shell, const Index* orb_cart_local,
                                 const double* transform_table, const double* d_sph, double* d_m1,
                                 int* d_nonfinite) {
  const Size total = Size(q_sph) * norb_sph * norb_cart;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int B = static_cast<int>(idx % norb_cart);
  const Size rem = idx / norb_cart;
  const int mu = static_cast<int>(rem % norb_sph);
  const int q = static_cast<int>(rem / norb_sph);

  const int sa = orb_sph_to_shell[mu];
  const int sb = orb_cart_to_shell[B];
  if (sa < sb) return;

  const int lb = angular[sb];
  const int mb = sph_size(lb);
  const int nb = cart_size(lb);
  const int b0 = sph_offsets[sb];
  const int b_c = orb_cart_local[B];
  const double* Tb = transform_table + transform_offset(lb);
  const double* w_q = d_sph + Size(q) * norb_sph * norb_sph;
  const double* row_mu = w_q + Size(mu) * norb_sph + b0;

  double acc = 0.0;
  if (sa == sb) {
    for (int b = 0; b < mb; ++b) {
      const double w = row_mu[b];
      if (!isfinite(w)) atomicExch(d_nonfinite, 1);
      acc += w * Tb[b * nb + b_c];
    }
  } else {
    for (int b = 0; b < mb; ++b) {
      const double w = row_mu[b] + w_q[Size(b0 + b) * norb_sph + mu];
      if (!isfinite(w)) atomicExch(d_nonfinite, 1);
      acc += w * Tb[b * nb + b_c];
    }
  }
  d_m1[idx] = acc;
}

// 3-center Pack Step 2: transform orbital index a -> A into packed lower-triangle pair layout
__global__ void ao_pack_a_kernel(int n_orb_pairs, int q_sph, int norb_sph, int norb_cart,
                                 Size n_pair_cart, const Index* pair_sa, const Index* pair_sb,
                                 const Index* pair_cart_offsets, const Index* angular,
                                 const Index* sph_offsets, const Index* orb_cart_offsets,
                                 const double* transform_table, const double* d_m1, double* d_m2) {
  const int pair_idx = blockIdx.x;
  const int q = blockIdx.y;
  if (pair_idx >= n_orb_pairs || q >= q_sph) return;
  const int sa = pair_sa[pair_idx];
  const int sb = pair_sb[pair_idx];
  const int la = angular[sa], lb = angular[sb];
  const int ma = sph_size(la), na = cart_size(la), nb = cart_size(lb);
  const int a0 = sph_offsets[sa];
  const int B0 = orb_cart_offsets[sb];
  const double* Ta = transform_table + transform_offset(la);
  const double* m1_q = d_m1 + Size(q) * norb_sph * norb_cart;
  double* dst = d_m2 + Size(q) * n_pair_cart + pair_cart_offsets[pair_idx];

  for (int ab = threadIdx.x; ab < na * nb; ab += blockDim.x) {
    const int a_c = ab / nb;
    const int b_c = ab % nb;
    const int B = B0 + b_c;
    double acc = 0.0;
    for (int a = 0; a < ma; ++a) {
      acc += Ta[a * na + a_c] * m1_q[Size(a0 + a) * norb_cart + B];
    }
    dst[ab] = acc;
  }
}

// 3-center Pack Step 3: transform auxiliary index p -> Q into shell-block Cartesian layout
__global__ void ao_pack_c_kernel(int p0_shell, int n_slab_aux_sh, int q_sph_start, Size n_pair_cart,
                                 const Index* angular, const Index* sph_offsets,
                                 const Index* slab_aux_cart_off, const double* transform_table,
                                 const double* d_m2, double* d_cart) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const int sc_local = blockIdx.y;
  if (idx >= n_pair_cart || sc_local >= n_slab_aux_sh) return;
  const int sc = p0_shell + sc_local;
  const int lc = angular[sc];
  const int mc = sph_size(lc);
  const int nc = cart_size(lc);
  const int q_local0 = sph_offsets[sc] - q_sph_start;
  const double* Tc = transform_table + transform_offset(lc);
  double* dst = d_cart + Size(slab_aux_cart_off[sc_local]) * n_pair_cart + idx * nc;

  for (int q_c = 0; q_c < nc; ++q_c) {
    double acc = 0.0;
    for (int p = 0; p < mc; ++p) {
      acc += Tc[p * nc + q_c] * d_m2[Size(q_local0 + p) * n_pair_cart + idx];
    }
    dst[q_c] = acc;
  }
}

// 3-center Unpack Step 1: transform auxiliary Cartesian Q -> spherical p
__global__ void ao_unpack_c_kernel(int p0_shell, int n_slab_aux_sh, int q_sph_start,
                                   Size n_pair_cart, const Index* angular, const Index* sph_offsets,
                                   const Index* slab_aux_cart_off, const double* transform_table,
                                   const double* d_cart, double* d_m2) {
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  const int sc_local = blockIdx.y;
  if (idx >= n_pair_cart || sc_local >= n_slab_aux_sh) return;
  const int sc = p0_shell + sc_local;
  const int lc = angular[sc];
  const int mc = sph_size(lc);
  const int nc = cart_size(lc);
  const int q_local0 = sph_offsets[sc] - q_sph_start;
  const double* Tc = transform_table + transform_offset(lc);
  const double* src = d_cart + Size(slab_aux_cart_off[sc_local]) * n_pair_cart + idx * nc;

  for (int p = 0; p < mc; ++p) {
    double acc = 0.0;
    for (int q_c = 0; q_c < nc; ++q_c) {
      acc += Tc[p * nc + q_c] * src[q_c];
    }
    d_m2[Size(q_local0 + p) * n_pair_cart + idx] = acc;
  }
}

// 3-center Unpack Step 2: transform orbital Cartesian A -> spherical a
__global__ void ao_unpack_a_kernel(int n_orb_pairs, int q_sph, int norb_sph, int norb_cart,
                                   Size n_pair_cart, const Index* pair_sa, const Index* pair_sb,
                                   const Index* pair_cart_offsets, const Index* angular,
                                   const Index* sph_offsets, const Index* orb_cart_offsets,
                                   const double* transform_table, const double* d_m2,
                                   double* d_m1) {
  const int pair_idx = blockIdx.x;
  const int q = blockIdx.y;
  if (pair_idx >= n_orb_pairs || q >= q_sph) return;
  const int sa = pair_sa[pair_idx];
  const int sb = pair_sb[pair_idx];
  const int la = angular[sa], lb = angular[sb];
  const int ma = sph_size(la), na = cart_size(la), nb = cart_size(lb);
  const int a0 = sph_offsets[sa];
  const int B0 = orb_cart_offsets[sb];
  const double* Ta = transform_table + transform_offset(la);
  const double* src = d_m2 + Size(q) * n_pair_cart + pair_cart_offsets[pair_idx];
  double* m1_q = d_m1 + Size(q) * norb_sph * norb_cart;

  for (int ab = threadIdx.x; ab < ma * nb; ab += blockDim.x) {
    const int a = ab / nb;
    const int b_c = ab % nb;
    double acc = 0.0;
    for (int a_c = 0; a_c < na; ++a_c) {
      acc += Ta[a * na + a_c] * src[a_c * nb + b_c];
    }
    m1_q[Size(a0 + a) * norb_cart + (B0 + b_c)] = acc;
  }
}

// 3-center Unpack Step 3: transform orbital Cartesian B -> spherical b + symmetric scatter
__global__ void ao_unpack_b_kernel(int q_sph, int norb_sph, int norb_cart, const Index* angular,
                                   const Index* orb_sph_to_shell, const Index* orb_sph_local,
                                   const Index* orb_cart_offsets, const double* transform_table,
                                   const double* d_m1, double* d_sph, int* d_nonfinite) {
  const Size total = Size(q_sph) * norb_sph * norb_sph;
  const Size idx = Size(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int nu = static_cast<int>(idx % norb_sph);
  const Size rem = idx / norb_sph;
  const int mu = static_cast<int>(rem % norb_sph);
  const int q = static_cast<int>(rem / norb_sph);

  const int sa = orb_sph_to_shell[mu];
  const int sb = orb_sph_to_shell[nu];
  if (sa < sb) return;

  const int lb = angular[sb];
  const int nb = cart_size(lb);
  const int b_local = orb_sph_local[nu];
  const int B0 = orb_cart_offsets[sb];
  const double* Tb = transform_table + transform_offset(lb);
  const double* m1_row = d_m1 + (Size(q) * norb_sph + mu) * norb_cart + B0;

  double acc = 0.0;
  for (int b_c = 0; b_c < nb; ++b_c) {
    acc += Tb[b_local * nb + b_c] * m1_row[b_c];
  }
  if (!isfinite(acc)) atomicExch(d_nonfinite, 1);
  double* out_q = d_sph + Size(q) * norb_sph * norb_sph;
  out_q[Size(mu) * norb_sph + nu] = acc;
  if (sa > sb) {
    out_q[Size(nu) * norb_sph + mu] = acc;
  }
}

// Forward slabbed 3-center kernel (no block_shells/offsets upload needed!)
__global__ void cartesian_ao_slab_blocks(int p0_shell, int n_orb_pairs, Size n_pair_cart, int op,
                                         gi::FittedSlater fit, const Index* pair_sa,
                                         const Index* pair_sb, const Index* pair_cart_offsets,
                                         const Index* slab_aux_cart_off, const Index* angular,
                                         const Index* prim_offsets, const double* exponents,
                                         const double* radial, const double* centers,
                                         double* d_cart) {
  const int pair_idx = blockIdx.x;
  const int sc_local = blockIdx.y;
  if (pair_idx >= n_orb_pairs) return;
  const Index sa = pair_sa[pair_idx];
  const Index sb = pair_sb[pair_idx];
  const Index sc = p0_shell + sc_local;
  const int nc = cart_size(angular[sc]);
  const Size start = Size(slab_aux_cart_off[sc_local]) * n_pair_cart +
                     Size(pair_cart_offsets[pair_idx]) * nc;
  ShellBlockMeta shell;
  shell.init_shells(false, sa, sb, sc, start, angular, prim_offsets, centers);
  cartesian_block_device(shell, exponents, radial, op, fit, d_cart + start);
}

// Weighted derivative slabbed 3-center kernel with direct GPU atom-gradient reduction
__global__ void weighted_ao_slab_blocks(int p0_shell, int n_orb_pairs, Size n_pair_cart, int op,
                                        gi::FittedSlater fit, const Index* pair_sa,
                                        const Index* pair_sb, const Index* pair_cart_offsets,
                                        const Index* slab_aux_cart_off, const Index* angular,
                                        const Index* atom, const Index* prim_offsets,
                                        const double* exponents, const double* radial,
                                        const double* centers, const double* d_cart,
                                        double* d_value_and_grad) {
  const int pair_idx = blockIdx.x;
  const int sc_local = blockIdx.y;
  if (pair_idx >= n_orb_pairs) return;
  const Index sa = pair_sa[pair_idx];
  const Index sb = pair_sb[pair_idx];
  const Index sc = p0_shell + sc_local;
  const int nc = cart_size(angular[sc]);
  const Size start = Size(slab_aux_cart_off[sc_local]) * n_pair_cart +
                     Size(pair_cart_offsets[pair_idx]) * nc;
  ShellBlockMeta shell;
  shell.init_shells(false, sa, sb, sc, start, angular, prim_offsets, centers);
  double reduced[channels];
  if (!weighted_block_device(shell, exponents, radial, d_cart + start, op, fit, reduced)) return;
  if (threadIdx.x == 0) {
    atomicAdd(&d_value_and_grad[0], reduced[0]);
    const int aa = atom[sa], ab = atom[sb], ac = atom[sc];
    atomicAdd(&d_value_and_grad[1 + 3 * aa + 0], reduced[1]);
    atomicAdd(&d_value_and_grad[1 + 3 * aa + 1], reduced[2]);
    atomicAdd(&d_value_and_grad[1 + 3 * aa + 2], reduced[3]);
    atomicAdd(&d_value_and_grad[1 + 3 * ab + 0], reduced[4]);
    atomicAdd(&d_value_and_grad[1 + 3 * ab + 1], reduced[5]);
    atomicAdd(&d_value_and_grad[1 + 3 * ab + 2], reduced[6]);
    atomicAdd(&d_value_and_grad[1 + 3 * ac + 0], reduced[7]);
    atomicAdd(&d_value_and_grad[1 + 3 * ac + 1], reduced[8]);
    atomicAdd(&d_value_and_grad[1 + 3 * ac + 2], reduced[9]);
  }
}

// Forward metric 2-center kernel
__global__ void cartesian_metric_blocks(int n_pairs, int op, gi::FittedSlater fit,
                                        const Index* pair_sa, const Index* pair_sc,
                                        const Index* pair_cart_offsets, const Index* angular,
                                        const Index* prim_offsets, const double* exponents,
                                        const double* radial, const double* centers,
                                        double* d_cart) {
  const int p = blockIdx.x;
  if (p >= n_pairs) return;
  const Size start = static_cast<Size>(pair_cart_offsets[p]);
  ShellBlockMeta shell;
  shell.init_shells(true, pair_sa[p], pair_sa[p], pair_sc[p], start, angular, prim_offsets,
                    centers);
  cartesian_block_device(shell, exponents, radial, op, fit, d_cart + start);
}

// Weighted metric 2-center kernel with direct GPU atom-gradient reduction
__global__ void weighted_metric_blocks(int n_pairs, int op, gi::FittedSlater fit,
                                       const Index* pair_sa, const Index* pair_sc,
                                       const Index* pair_cart_offsets, const Index* angular,
                                       const Index* atom, const Index* prim_offsets,
                                       const double* exponents, const double* radial,
                                       const double* centers, const double* d_cart,
                                       double* d_value_and_grad) {
  const int p = blockIdx.x;
  if (p >= n_pairs) return;
  const Size start = static_cast<Size>(pair_cart_offsets[p]);
  ShellBlockMeta shell;
  shell.init_shells(true, pair_sa[p], pair_sa[p], pair_sc[p], start, angular, prim_offsets,
                    centers);
  double reduced[channels];
  if (!weighted_block_device(shell, exponents, radial, d_cart + start, op, fit, reduced)) return;
  if (threadIdx.x == 0) {
    atomicAdd(&d_value_and_grad[0], reduced[0]);
    const int aa = atom[shell.sa], ac = atom[shell.sc];
    atomicAdd(&d_value_and_grad[1 + 3 * aa + 0], reduced[1]);
    atomicAdd(&d_value_and_grad[1 + 3 * aa + 1], reduced[2]);
    atomicAdd(&d_value_and_grad[1 + 3 * aa + 2], reduced[3]);
    atomicAdd(&d_value_and_grad[1 + 3 * ac + 0], reduced[7]);
    atomicAdd(&d_value_and_grad[1 + 3 * ac + 1], reduced[8]);
    atomicAdd(&d_value_and_grad[1 + 3 * ac + 2], reduced[9]);
  }
}
#endif

bool supported(ShellBatch data, int ncenter, int nshell, int nprimitive, int nblock, int nweights) {
  if (data.primitive_offsets[0] != 0 || data.primitive_offsets[nshell] != nprimitive ||
      data.weight_offsets[0] != 0 || data.weight_offsets[nblock] != nweights)
    return false;
  for (int s = 0; s < nshell; ++s) {
    if (data.angular[s] < 0 || data.angular[s] > maximum_fitting_l ||
        data.primitive_offsets[s] < 0 ||
        data.primitive_offsets[s + 1] <= data.primitive_offsets[s] ||
        data.primitive_offsets[s + 1] > nprimitive)
      return false;
  }
  for (int block = 0; block < nblock; ++block) {
    const auto* shells = data.block_shells + Size(ncenter) * block;
    int cart_count = 1;
    Size primitive_count = 1;
    for (int q = 0; q < ncenter; ++q) {
      const int s = shells[q];
      if (s < 0 || s >= nshell || (ncenter == 3 && q < 2 && data.angular[s] > maximum_combined_l))
        return false;
      cart_count *= cart_size(data.angular[s]);
      const Size length = data.primitive_offsets[s + 1] - data.primitive_offsets[s];
      if (primitive_count > UINT64_MAX / length) return false;
      primitive_count *= length;
    }
    if (data.weight_offsets[block] < 0 || data.weight_offsets[block + 1] > nweights ||
        std::int64_t(data.weight_offsets[block + 1]) - data.weight_offsets[block] != cart_count ||
        primitive_count > (UINT64_MAX - threads) / cart_count)
      return false;
  }
  for (int p = 0; p < nprimitive; ++p)
    if (!(data.exponents[p] > 0.) || !std::isfinite(data.exponents[p]) ||
        !std::isfinite(data.radial[p]))
      return false;
  const auto finite = [](double x) { return std::isfinite(x); };
  return std::all_of(data.centers, data.centers + Size(nshell) * 3, finite) &&
         (!data.weights || std::all_of(data.weights, data.weights + nweights, finite));
}

bool validate_spherical_inputs(int ncenter, int nshell, int orbital_shell_count, int nprimitive,
                               int natoms, int norb_sph, int naux_sph, const Index* angular,
                               const Index* atom, const Index* prim_offsets,
                               const double* exponents, const double* radial, const double* centers,
                               const Index* sph_offsets, const double* transform_table) {
  if ((ncenter != 2 && ncenter != 3) || nshell <= 1 || orbital_shell_count <= 0 ||
      orbital_shell_count >= nshell || nprimitive <= 0 || norb_sph <= 0 || naux_sph <= 0 ||
      !angular || !prim_offsets || !exponents || !radial || !centers || !sph_offsets ||
      !transform_table)
    return false;
  if (atom && natoms <= 0) return false;
  if (prim_offsets[0] != 0 || prim_offsets[nshell] != nprimitive) return false;

  int running_orb = 0, running_aux = 0;
  for (int s = 0; s < nshell; ++s) {
    const int l = angular[s];
    const int max_l = (s < orbital_shell_count) ? maximum_combined_l : maximum_fitting_l;
    if (l < 0 || l > max_l) return false;
    if (prim_offsets[s] < 0 || prim_offsets[s + 1] <= prim_offsets[s] ||
        prim_offsets[s + 1] > nprimitive)
      return false;
    if (atom && (atom[s] < 0 || atom[s] >= natoms)) return false;
    if (s < orbital_shell_count) {
      if (sph_offsets[s] != running_orb) return false;
      running_orb += sph_size(l);
    } else {
      if (sph_offsets[s] != running_aux) return false;
      running_aux += sph_size(l);
    }
  }
  if (running_orb != norb_sph || running_aux != naux_sph) return false;
  for (int p = 0; p < nprimitive; ++p) {
    if (!(exponents[p] > 0.0) || !std::isfinite(exponents[p]) || !std::isfinite(radial[p]))
      return false;
  }
  const auto finite = [](double x) { return std::isfinite(x); };
  return std::all_of(centers, centers + Size(nshell) * 3, finite) &&
         std::all_of(transform_table, transform_table + kTransformTableSize, finite);
}

// One exact-sized aligned global allocation.
struct Layout {
  Size length = 0;
  template <class T>
  Size take(Size count) {
    constexpr Size align = alignof(T) > 16 ? alignof(T) : 16;
    length = (length + align - 1) & ~(align - 1);
    const Size start = length;
    length += sizeof(T) * count;
    return start;
  }
};

#ifndef GQC_SINK_HOST_REFERENCE
struct DeviceMemory {
  unsigned char* ptr = nullptr;
  ~DeviceMemory() {
    if (ptr) cudaFree(ptr);
  }
};
#endif

// Status: 0 success, -1 incompatible/malformed ABI data, -2 explicit storage
// budget exceeded, -3 nonfinite scientific output, >0 CUDA runtime error.
template <bool Forward>
int run_integrals(int ncenter, int op, double beta, int slater_norm_code, int nshell,
                  int nprimitive, int nblock, int nweights, const Index* angular,
                  const Index* prim_offsets, const double* exponents, const double* radial,
                  const double* centers, const Index* block_shells, const Index* weight_offsets,
                  const double* weights, double* result, Size max_device_bytes,
                  Size* device_bytes) {
  if (!device_bytes) return -1;
  *device_bytes = 0;
  if ((ncenter != 2 && ncenter != 3) || nshell <= 0 || nprimitive <= 0 || nblock <= 0 ||
      nweights <= 0 || !angular || !prim_offsets || !exponents || !radial || !centers ||
      !block_shells || !weight_offsets || (!Forward && !weights) || !result || !max_device_bytes)
    return -1;
  gi::SlaterFitNormalization norm = gi::SlaterFitNormalization::negative_exponential_over_beta;
  if (!decode_slater_normalization(slater_norm_code, norm)) return -1;
  gi::FittedSlater fit{};
#ifdef GQC_SINK_COULOMB
  if (op != 0) return -1;
#else
  if (op < 0 || op > 3 || !gi::make_fitted_slater(beta, norm, fit)) return -1;
#endif
  const Size result_size = Forward ? Size(nweights) : Size(nblock) * channels;
  Layout layout;
  const Size exponent_pos = layout.take<double>(nprimitive),
             radial_pos = layout.take<double>(nprimitive),
             center_pos = layout.take<double>(Size(nshell) * 3),
             weight_pos = layout.take<double>(Forward ? 0 : nweights),
             result_pos = layout.take<double>(result_size),
             angular_pos = layout.take<Index>(nshell),
             primitive_pos = layout.take<Index>(Size(nshell) + 1),
             shell_pos = layout.take<Index>(Size(nblock) * ncenter),
             offset_pos = layout.take<Index>(Size(nblock) + 1);
  *device_bytes = layout.length;
  if (layout.length > max_device_bytes) return -2;
  const ShellBatch host{angular,   prim_offsets, block_shells, weight_offsets,
                        exponents, radial,       centers,      weights};
  if (!supported(host, ncenter, nshell, nprimitive, nblock, nweights)) return -1;
#ifdef GQC_SINK_HOST_REFERENCE
  for (int block = 0; block < nblock; ++block) {
    ShellBlockMeta shell;
    shell.init_from_batch(host, ncenter, block);
    if constexpr (Forward) {
      cartesian_values_host(shell, exponents, radial, op, fit, result + shell.start);
    } else {
      double sums[channels]{};
      weighted_terms_host(shell, exponents, radial, weights + shell.start, op, fit, sums);
      std::copy_n(sums, channels, result + Size(block) * channels);
    }
  }
#else
  DeviceMemory buffer;
  cudaError_t status = cudaMalloc(&buffer.ptr, layout.length);
  if (status != cudaSuccess) return status;
  auto doubles = [&](Size at) { return reinterpret_cast<double*>(buffer.ptr + at); };
  auto indices = [&](Size at) { return reinterpret_cast<Index*>(buffer.ptr + at); };
  auto upload = [&](auto* target, const auto* source, Size count) {
    if (status == cudaSuccess && count > 0)
      status = cudaMemcpy(target, source, count * sizeof(*source), cudaMemcpyHostToDevice);
  };
  upload(doubles(exponent_pos), exponents, nprimitive);
  upload(doubles(radial_pos), radial, nprimitive);
  upload(doubles(center_pos), centers, Size(nshell) * 3);
  if constexpr (!Forward) upload(doubles(weight_pos), weights, nweights);
  upload(indices(angular_pos), angular, nshell);
  upload(indices(primitive_pos), prim_offsets, Size(nshell) + 1);
  upload(indices(shell_pos), block_shells, Size(nblock) * ncenter);
  upload(indices(offset_pos), weight_offsets, Size(nblock) + 1);
  if (status != cudaSuccess) return status;
  const ShellBatch gpu{indices(angular_pos),  indices(primitive_pos),
                       indices(shell_pos),    indices(offset_pos),
                       doubles(exponent_pos), doubles(radial_pos),
                       doubles(center_pos),   Forward ? nullptr : doubles(weight_pos)};
  // clang-format off
  if constexpr (Forward)
    cartesian_blocks<<<nblock, threads>>>(gpu, ncenter, op, fit, doubles(result_pos));
  else
    weighted_blocks<<<nblock, threads>>>(gpu, ncenter, op, fit, doubles(result_pos));
  // clang-format on
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  status =
      cudaMemcpy(result, doubles(result_pos), result_size * sizeof(double), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) return status;
#endif
  const auto finite = [](double x) { return std::isfinite(x); };
  return std::all_of(result, result + result_size, finite) ? 0 : -3;
}

template <bool Forward>
int run_spherical(int ncenter, int op, double beta, int slater_norm_code, int nshell,
                  int orbital_shell_count, int nprimitive, int natoms, int norb_sph, int naux_sph,
                  const Index* angular, const Index* atom, const Index* prim_offsets,
                  const double* exponents, const double* radial, const double* centers,
                  const Index* sph_offsets, const double* transform_table, double* sph_io,
                  double* weighted_value_out, double* atom_gradient_out, Size max_device_bytes,
                  Size* device_bytes) {
  if (!device_bytes) return -1;
  *device_bytes = 0;
  if (!sph_io || !max_device_bytes) return -1;
  if constexpr (!Forward) {
    if (!atom || !weighted_value_out || !atom_gradient_out || natoms <= 0) return -1;
  }
  gi::SlaterFitNormalization norm = gi::SlaterFitNormalization::negative_exponential_over_beta;
  if (!decode_slater_normalization(slater_norm_code, norm)) return -1;
  gi::FittedSlater fit{};
#ifdef GQC_SINK_COULOMB
  if (op != 0) return -1;
#else
  if (op < 0 || op > 3 || !gi::make_fitted_slater(beta, norm, fit)) return -1;
#endif
  if (!validate_spherical_inputs(ncenter, nshell, orbital_shell_count, nprimitive, natoms, norb_sph,
                                 naux_sph, angular, Forward ? nullptr : atom, prim_offsets,
                                 exponents, radial, centers, sph_offsets, transform_table))
    return -1;

  const int n_orb_sh = orbital_shell_count;
  const int n_aux_sh = nshell - orbital_shell_count;

  // Build lower-triangle shell pair index tables.
  if (ncenter == 2) {
    const int n_pairs = n_aux_sh * (n_aux_sh + 1) / 2;
    std::vector<Index> pair_sa(n_pairs), pair_sc(n_pairs), pair_cart_off(n_pairs + 1);
    Size running_cart = 0;
    int p = 0;
    for (int r = 0; r < n_aux_sh; ++r) {
      const Index sa = n_orb_sh + r;
      const int na = cart_size(angular[sa]);
      for (int c = 0; c <= r; ++c) {
        const Index sc = n_orb_sh + c;
        const int nc = cart_size(angular[sc]);
        pair_sa[p] = sa;
        pair_sc[p] = sc;
        pair_cart_off[p] = static_cast<Index>(running_cart);
        running_cart += Size(na) * nc;
        if (running_cart >= (Size(1) << 31)) return -2;
        ++p;
      }
    }
    pair_cart_off[n_pairs] = static_cast<Index>(running_cart);

    const Size sph_count = Size(naux_sph) * naux_sph;
    Layout layout;
    const Size exp_pos = layout.take<double>(nprimitive);
    const Size rad_pos = layout.take<double>(nprimitive);
    const Size ctr_pos = layout.take<double>(Size(nshell) * 3);
    const Size tab_pos = layout.take<double>(kTransformTableSize);
    const Size sph_pos = layout.take<double>(sph_count);
    const Size cart_pos = layout.take<double>(running_cart);
    const Size vg_pos = layout.take<double>(Forward ? 0 : (1 + Size(natoms) * 3));
    const Size ang_pos = layout.take<Index>(nshell);
    const Size atm_pos = layout.take<Index>(Forward ? 0 : nshell);
    const Size prm_pos = layout.take<Index>(Size(nshell) + 1);
    const Size sof_pos = layout.take<Index>(nshell);
    const Size psa_pos = layout.take<Index>(n_pairs);
    const Size psc_pos = layout.take<Index>(n_pairs);
    const Size pco_pos = layout.take<Index>(Size(n_pairs) + 1);
    const Size nf_pos = layout.take<int>(1);
    *device_bytes = layout.length;
    if (layout.length > max_device_bytes) return -2;

#ifdef GQC_SINK_HOST_REFERENCE
    if constexpr (!Forward) {
      for (Size i = 0; i < sph_count; ++i)
        if (!std::isfinite(sph_io[i])) return -1;
      *weighted_value_out = 0.0;
      std::fill_n(atom_gradient_out, Size(natoms) * 3, 0.0);
    }
    std::vector<double> cart_buf;
    for (int idx = 0; idx < n_pairs; ++idx) {
      const Index sa = pair_sa[idx], sc = pair_sc[idx];
      const int la = angular[sa], lc = angular[sc];
      const int ma = sph_size(la), mc = sph_size(lc);
      const int na = cart_size(la), nc = cart_size(lc);
      const int a0 = sph_offsets[sa], c0 = sph_offsets[sc];
      const double* Ta = transform_table + transform_offset(la);
      const double* Tc = transform_table + transform_offset(lc);
      const bool diff = (sa != sc);
      cart_buf.resize(static_cast<std::size_t>(na * nc));
      ShellBlockMeta shell;
      shell.init_shells(true, sa, sa, sc, 0, angular, prim_offsets, centers);
      if constexpr (Forward) {
        cartesian_values_host(shell, exponents, radial, op, fit, cart_buf.data());
        for (int a = 0; a < ma; ++a) {
          for (int c = 0; c < mc; ++c) {
            double acc = 0.0;
            for (int a_c = 0; a_c < na; ++a_c) {
              double inner = 0.0;
              for (int c_c = 0; c_c < nc; ++c_c) inner += cart_buf[a_c * nc + c_c] * Tc[c * nc + c_c];
              acc += Ta[a * na + a_c] * inner;
            }
            if (!std::isfinite(acc)) return -3;
            sph_io[Size(a0 + a) * naux_sph + (c0 + c)] = acc;
            if (diff) sph_io[Size(c0 + c) * naux_sph + (a0 + a)] = acc;
          }
        }
      } else {
        for (int a_c = 0; a_c < na; ++a_c) {
          for (int c_c = 0; c_c < nc; ++c_c) {
            double acc = 0.0;
            for (int a = 0; a < ma; ++a) {
              double inner = 0.0;
              for (int c = 0; c < mc; ++c) {
                const double w1 = sph_io[Size(a0 + a) * naux_sph + (c0 + c)];
                const double w = diff ? (w1 + sph_io[Size(c0 + c) * naux_sph + (a0 + a)]) : w1;
                inner += w * Tc[c * nc + c_c];
              }
              acc += Ta[a * na + a_c] * inner;
            }
            cart_buf[a_c * nc + c_c] = acc;
          }
        }
        double sums[channels]{};
        weighted_terms_host(shell, exponents, radial, cart_buf.data(), op, fit, sums);
        *weighted_value_out += sums[0];
        const int aa = atom[sa], ac = atom[sc];
        for (int q = 0; q < 3; ++q) {
          atom_gradient_out[3 * aa + q] += sums[1 + q];
          atom_gradient_out[3 * ac + q] += sums[7 + q];
        }
      }
    }
    if constexpr (!Forward) {
      if (!std::isfinite(*weighted_value_out) ||
          !std::all_of(atom_gradient_out, atom_gradient_out + Size(natoms) * 3,
                       [](double x) { return std::isfinite(x); }))
        return -3;
    }
    return 0;
#else
    DeviceMemory buffer;
    cudaError_t status = cudaMalloc(&buffer.ptr, layout.length);
    if (status != cudaSuccess) return status;
    auto doubles = [&](Size at) { return reinterpret_cast<double*>(buffer.ptr + at); };
    auto indices = [&](Size at) { return reinterpret_cast<Index*>(buffer.ptr + at); };
    auto ints = [&](Size at) { return reinterpret_cast<int*>(buffer.ptr + at); };
    auto upload = [&](auto* target, const auto* source, Size count) {
      if (status == cudaSuccess && count > 0)
        status = cudaMemcpy(target, source, count * sizeof(*source), cudaMemcpyHostToDevice);
    };
    upload(doubles(exp_pos), exponents, nprimitive);
    upload(doubles(rad_pos), radial, nprimitive);
    upload(doubles(ctr_pos), centers, Size(nshell) * 3);
    upload(doubles(tab_pos), transform_table, kTransformTableSize);
    upload(indices(ang_pos), angular, nshell);
    if constexpr (!Forward) upload(indices(atm_pos), atom, nshell);
    upload(indices(prm_pos), prim_offsets, Size(nshell) + 1);
    upload(indices(sof_pos), sph_offsets, nshell);
    upload(indices(psa_pos), pair_sa.data(), n_pairs);
    upload(indices(psc_pos), pair_sc.data(), n_pairs);
    upload(indices(pco_pos), pair_cart_off.data(), Size(n_pairs) + 1);
    if (status == cudaSuccess) status = cudaMemset(ints(nf_pos), 0, sizeof(int));
    if constexpr (!Forward) {
      upload(doubles(sph_pos), sph_io, sph_count);
      if (status == cudaSuccess)
        status = cudaMemset(doubles(vg_pos), 0, (1 + Size(natoms) * 3) * sizeof(double));
    }
    if (status != cudaSuccess) return status;

    if constexpr (Forward) {
      cartesian_metric_blocks<<<n_pairs, threads>>>(
          n_pairs, op, fit, indices(psa_pos), indices(psc_pos), indices(pco_pos), indices(ang_pos),
          indices(prm_pos), doubles(exp_pos), doubles(rad_pos), doubles(ctr_pos),
          doubles(cart_pos));
      metric_unpack_kernel<<<n_pairs, threads>>>(
          n_pairs, naux_sph, indices(psa_pos), indices(psc_pos), indices(pco_pos), indices(ang_pos),
          indices(sof_pos), doubles(tab_pos), doubles(cart_pos), doubles(sph_pos), ints(nf_pos));
      status = cudaGetLastError();
      if (status != cudaSuccess) return status;
      int nonfinite = 0;
      status = cudaMemcpy(&nonfinite, ints(nf_pos), sizeof(int), cudaMemcpyDeviceToHost);
      if (status != cudaSuccess) return status;
      if (nonfinite) return -3;
      status = cudaMemcpy(sph_io, doubles(sph_pos), sph_count * sizeof(double),
                          cudaMemcpyDeviceToHost);
      return status == cudaSuccess ? 0 : status;
    } else {
      metric_pack_kernel<<<n_pairs, threads>>>(
          n_pairs, naux_sph, n_orb_sh, indices(psa_pos), indices(psc_pos), indices(pco_pos),
          indices(ang_pos), indices(sof_pos), doubles(tab_pos), doubles(sph_pos), doubles(cart_pos),
          ints(nf_pos));
      status = cudaGetLastError();
      if (status != cudaSuccess) return status;
      int nonfinite = 0;
      status = cudaMemcpy(&nonfinite, ints(nf_pos), sizeof(int), cudaMemcpyDeviceToHost);
      if (status != cudaSuccess) return status;
      if (nonfinite) return -1;
      weighted_metric_blocks<<<n_pairs, threads>>>(
          n_pairs, op, fit, indices(psa_pos), indices(psc_pos), indices(pco_pos), indices(ang_pos),
          indices(atm_pos), indices(prm_pos), doubles(exp_pos), doubles(rad_pos), doubles(ctr_pos),
          doubles(cart_pos), doubles(vg_pos));
      status = cudaGetLastError();
      if (status != cudaSuccess) return status;
      std::vector<double> vg(1 + Size(natoms) * 3);
      status = cudaMemcpy(vg.data(), doubles(vg_pos), vg.size() * sizeof(double),
                          cudaMemcpyDeviceToHost);
      if (status != cudaSuccess) return status;
      if (!std::all_of(vg.begin(), vg.end(), [](double x) { return std::isfinite(x); })) return -3;
      *weighted_value_out = vg[0];
      std::copy_n(vg.data() + 1, Size(natoms) * 3, atom_gradient_out);
      return 0;
    }
#endif
  }

  // =========================================================================
  // ncenter == 3 (AO 3-center): Auxiliary-slabbed GPU-fused streaming
  // =========================================================================
  std::vector<Index> orb_cart_off(n_orb_sh + 1);
  int norb_cart = 0;
  for (int s = 0; s < n_orb_sh; ++s) {
    orb_cart_off[s] = norb_cart;
    norb_cart += cart_size(angular[s]);
  }
  orb_cart_off[n_orb_sh] = norb_cart;

  std::vector<Index> orb_cart_to_sh(norb_cart), orb_cart_loc(norb_cart);
  for (int s = 0; s < n_orb_sh; ++s) {
    const int nc = cart_size(angular[s]);
    const int base = orb_cart_off[s];
    for (int c = 0; c < nc; ++c) {
      orb_cart_to_sh[base + c] = s;
      orb_cart_loc[base + c] = c;
    }
  }
  std::vector<Index> orb_sph_to_sh(norb_sph), orb_sph_loc(norb_sph);
  for (int s = 0; s < n_orb_sh; ++s) {
    const int ms = sph_size(angular[s]);
    const int base = sph_offsets[s];
    for (int a = 0; a < ms; ++a) {
      orb_sph_to_sh[base + a] = s;
      orb_sph_loc[base + a] = a;
    }
  }

  const int n_orb_pairs = n_orb_sh * (n_orb_sh + 1) / 2;
  std::vector<Index> pair_sa(n_orb_pairs), pair_sb(n_orb_pairs), pair_cart_off(n_orb_pairs + 1);
  Size n_pair_cart = 0;
  int pidx = 0;
  for (int r = 0; r < n_orb_sh; ++r) {
    const int na = cart_size(angular[r]);
    for (int c = 0; c <= r; ++c) {
      const int nb = cart_size(angular[c]);
      pair_sa[pidx] = r;
      pair_sb[pidx] = c;
      pair_cart_off[pidx] = static_cast<Index>(n_pair_cart);
      n_pair_cart += Size(na) * nb;
      if (n_pair_cart >= (Size(1) << 31)) return -2;
      ++pidx;
    }
  }
  pair_cart_off[n_orb_pairs] = static_cast<Index>(n_pair_cart);

  // Measure fixed metadata allocation first.
  Layout fixed_layout;
  const Size exp_pos = fixed_layout.take<double>(nprimitive);
  const Size rad_pos = fixed_layout.take<double>(nprimitive);
  const Size ctr_pos = fixed_layout.take<double>(Size(nshell) * 3);
  const Size tab_pos = fixed_layout.take<double>(kTransformTableSize);
  const Size vg_pos = fixed_layout.take<double>(Forward ? 0 : (1 + Size(natoms) * 3));
  const Size ang_pos = fixed_layout.take<Index>(nshell);
  const Size atm_pos = fixed_layout.take<Index>(Forward ? 0 : nshell);
  const Size prm_pos = fixed_layout.take<Index>(Size(nshell) + 1);
  const Size sof_pos = fixed_layout.take<Index>(nshell);
  const Size oco_pos = fixed_layout.take<Index>(Size(n_orb_sh) + 1);
  const Size ocs_pos = fixed_layout.take<Index>(norb_cart);
  const Size ocl_pos = fixed_layout.take<Index>(norb_cart);
  const Size oss_pos = fixed_layout.take<Index>(norb_sph);
  const Size osl_pos = fixed_layout.take<Index>(norb_sph);
  const Size psa_pos = fixed_layout.take<Index>(n_orb_pairs);
  const Size psb_pos = fixed_layout.take<Index>(n_orb_pairs);
  const Size pco_pos = fixed_layout.take<Index>(Size(n_orb_pairs) + 1);
  const Size sac_pos = fixed_layout.take<Index>(Size(n_aux_sh) + 1);
  const Size nf_pos = fixed_layout.take<int>(1);

  constexpr Size max_slab_cart_elems = Size(128) * 1000 * 1000;

  auto slab_total_bytes = [&](int q_sph, int q_cart) -> Size {
    Layout tmp = fixed_layout;
    tmp.take<double>(Size(q_sph) * norb_sph * norb_sph);
    tmp.take<double>(Size(q_sph) * norb_sph * norb_cart);
    tmp.take<double>(Size(q_sph) * n_pair_cart);
    tmp.take<double>(Size(q_cart) * n_pair_cart);
    return tmp.length;
  };

  // One workspace serves every slab, sized by the widest spherical and the widest Cartesian
  // slab, which can be different slabs. Admission checks that running envelope, seeded with
  // the widest single shell, so the minimum admissible budget does not depend on packing and
  // the workspace stays within max(min(budget, 1.5 GiB), that minimum).
  int max_q_sph = 0, max_q_cart = 0;
  for (int s = n_orb_sh; s < nshell; ++s) {
    max_q_sph = std::max(max_q_sph, sph_size(angular[s]));
    max_q_cart = std::max(max_q_cart, cart_size(angular[s]));
  }
  const Size minimum_bytes = slab_total_bytes(max_q_sph, max_q_cart);
  if (minimum_bytes > max_device_bytes) {
    *device_bytes = minimum_bytes;
    return -2;
  }
  const Size slab_cap = std::max(std::min(max_device_bytes, Size(1536) << 20), minimum_bytes);

  struct Slab {
    int p0, p1;        // relative aux shell range [p0, p1) in [0, n_aux_sh)
    int q_sph_start;   // starting spherical aux index in [0, naux_sph)
    int q_sph_count;
    int q_cart_count;
  };
  std::vector<Slab> slabs;
  int p0 = 0;
  while (p0 < n_aux_sh) {
    int p1 = p0;
    int q_sph = 0, q_cart = 0;
    while (p1 < n_aux_sh && (p1 - p0) < 65535) {
      const int lc = angular[n_orb_sh + p1];
      const int next_qs = q_sph + sph_size(lc);
      const int next_qc = q_cart + cart_size(lc);
      const Size need =
          slab_total_bytes(std::max(max_q_sph, next_qs), std::max(max_q_cart, next_qc));
      const Size cart_elems = Size(next_qc) * n_pair_cart;
      if (p1 > p0 && (need > slab_cap || cart_elems > max_slab_cart_elems || next_qs > 65535)) break;
      q_sph = next_qs;
      q_cart = next_qc;
      ++p1;
    }
    slabs.push_back({p0, p1, sph_offsets[n_orb_sh + p0], q_sph, q_cart});
    max_q_sph = std::max(max_q_sph, q_sph);
    max_q_cart = std::max(max_q_cart, q_cart);
    p0 = p1;
  }

  Layout total_layout = fixed_layout;
  const Size sph_pos = total_layout.take<double>(Size(max_q_sph) * norb_sph * norb_sph);
  const Size m1_pos = total_layout.take<double>(Size(max_q_sph) * norb_sph * norb_cart);
  const Size m2_pos = total_layout.take<double>(Size(max_q_sph) * n_pair_cart);
  const Size cart_pos = total_layout.take<double>(Size(max_q_cart) * n_pair_cart);
  *device_bytes = total_layout.length;
  if (total_layout.length > max_device_bytes) return -2;

#ifdef GQC_SINK_HOST_REFERENCE
  if constexpr (!Forward) {
    const Size full_sph = Size(naux_sph) * norb_sph * norb_sph;
    for (Size i = 0; i < full_sph; ++i)
      if (!std::isfinite(sph_io[i])) return -1;
    *weighted_value_out = 0.0;
    std::fill_n(atom_gradient_out, Size(natoms) * 3, 0.0);
  }
  std::vector<double> m1(Size(max_q_sph) * norb_sph * norb_cart);
  std::vector<double> m2(Size(max_q_sph) * n_pair_cart);
  std::vector<double> cart_slab(Size(max_q_cart) * n_pair_cart);
  std::vector<Index> slab_aux_cart_off(n_aux_sh + 1);

  for (const auto& slab : slabs) {
    const int n_slab_aux = slab.p1 - slab.p0;
    const int p0_shell = n_orb_sh + slab.p0;
    int run_c = 0;
    for (int s = 0; s < n_slab_aux; ++s) {
      slab_aux_cart_off[s] = run_c;
      run_c += cart_size(angular[p0_shell + s]);
    }
    slab_aux_cart_off[n_slab_aux] = run_c;
    double* sph_slice = sph_io + Size(slab.q_sph_start) * norb_sph * norb_sph;

    if constexpr (Forward) {
      for (int sc_l = 0; sc_l < n_slab_aux; ++sc_l) {
        const Index sc = p0_shell + sc_l;
        const int nc = cart_size(angular[sc]);
        for (int pair_i = 0; pair_i < n_orb_pairs; ++pair_i) {
          const Size start = Size(slab_aux_cart_off[sc_l]) * n_pair_cart +
                             Size(pair_cart_off[pair_i]) * nc;
          ShellBlockMeta shell;
          shell.init_shells(false, pair_sa[pair_i], pair_sb[pair_i], sc, start, angular,
                            prim_offsets, centers);
          cartesian_values_host(shell, exponents, radial, op, fit, cart_slab.data() + start);
        }
      }
      // Unpack c -> a -> b
      for (int sc_l = 0; sc_l < n_slab_aux; ++sc_l) {
        const int sc = p0_shell + sc_l;
        const int lc = angular[sc];
        const int mc = sph_size(lc), nc = cart_size(lc);
        const int q_l0 = sph_offsets[sc] - slab.q_sph_start;
        const double* Tc = transform_table + transform_offset(lc);
        const double* cbase = cart_slab.data() + Size(slab_aux_cart_off[sc_l]) * n_pair_cart;
        for (Size idx = 0; idx < n_pair_cart; ++idx) {
          const double* src = cbase + idx * nc;
          for (int p = 0; p < mc; ++p) {
            double acc = 0.0;
            for (int q_c = 0; q_c < nc; ++q_c) acc += Tc[p * nc + q_c] * src[q_c];
            m2[Size(q_l0 + p) * n_pair_cart + idx] = acc;
          }
        }
      }
      for (int q = 0; q < slab.q_sph_count; ++q) {
        double* m1_q = m1.data() + Size(q) * norb_sph * norb_cart;
        const double* m2_q = m2.data() + Size(q) * n_pair_cart;
        for (int pair_i = 0; pair_i < n_orb_pairs; ++pair_i) {
          const int sa = pair_sa[pair_i], sb = pair_sb[pair_i];
          const int la = angular[sa], lb = angular[sb];
          const int ma = sph_size(la), na = cart_size(la), nb = cart_size(lb);
          const int a0 = sph_offsets[sa], B0 = orb_cart_off[sb];
          const double* Ta = transform_table + transform_offset(la);
          const double* src = m2_q + pair_cart_off[pair_i];
          for (int a = 0; a < ma; ++a) {
            for (int b_c = 0; b_c < nb; ++b_c) {
              double acc = 0.0;
              for (int a_c = 0; a_c < na; ++a_c) acc += Ta[a * na + a_c] * src[a_c * nb + b_c];
              m1_q[Size(a0 + a) * norb_cart + (B0 + b_c)] = acc;
            }
          }
        }
        double* out_q = sph_slice + Size(q) * norb_sph * norb_sph;
        for (int mu = 0; mu < norb_sph; ++mu) {
          const int sa = orb_sph_to_sh[mu];
          for (int nu = 0; nu < norb_sph; ++nu) {
            const int sb = orb_sph_to_sh[nu];
            if (sa < sb) continue;
            const int lb = angular[sb], nb = cart_size(lb);
            const int b_loc = orb_sph_loc[nu], B0 = orb_cart_off[sb];
            const double* Tb = transform_table + transform_offset(lb);
            const double* m1_row = m1_q + Size(mu) * norb_cart + B0;
            double acc = 0.0;
            for (int b_c = 0; b_c < nb; ++b_c) acc += Tb[b_loc * nb + b_c] * m1_row[b_c];
            if (!std::isfinite(acc)) return -3;
            out_q[Size(mu) * norb_sph + nu] = acc;
            if (sa > sb) out_q[Size(nu) * norb_sph + mu] = acc;
          }
        }
      }
    } else {
      // Pack b -> a -> c
      for (int q = 0; q < slab.q_sph_count; ++q) {
        const double* w_q = sph_slice + Size(q) * norb_sph * norb_sph;
        double* m1_q = m1.data() + Size(q) * norb_sph * norb_cart;
        for (int mu = 0; mu < norb_sph; ++mu) {
          const int sa = orb_sph_to_sh[mu];
          for (int B = 0; B < norb_cart; ++B) {
            const int sb = orb_cart_to_sh[B];
            if (sa < sb) continue;
            const int lb = angular[sb], mb = sph_size(lb), nb = cart_size(lb);
            const int b0 = sph_offsets[sb], b_c = orb_cart_loc[B];
            const double* Tb = transform_table + transform_offset(lb);
            double acc = 0.0;
            if (sa == sb) {
              for (int b = 0; b < mb; ++b) acc += w_q[Size(mu) * norb_sph + (b0 + b)] * Tb[b * nb + b_c];
            } else {
              for (int b = 0; b < mb; ++b) {
                const double w = w_q[Size(mu) * norb_sph + (b0 + b)] + w_q[Size(b0 + b) * norb_sph + mu];
                acc += w * Tb[b * nb + b_c];
              }
            }
            m1_q[Size(mu) * norb_cart + B] = acc;
          }
        }
        double* m2_q = m2.data() + Size(q) * n_pair_cart;
        for (int pair_i = 0; pair_i < n_orb_pairs; ++pair_i) {
          const int sa = pair_sa[pair_i], sb = pair_sb[pair_i];
          const int la = angular[sa], lb = angular[sb];
          const int ma = sph_size(la), na = cart_size(la), nb = cart_size(lb);
          const int a0 = sph_offsets[sa], B0 = orb_cart_off[sb];
          const double* Ta = transform_table + transform_offset(la);
          double* dst = m2_q + pair_cart_off[pair_i];
          for (int a_c = 0; a_c < na; ++a_c) {
            for (int b_c = 0; b_c < nb; ++b_c) {
              double acc = 0.0;
              for (int a = 0; a < ma; ++a)
                acc += Ta[a * na + a_c] * m1_q[Size(a0 + a) * norb_cart + (B0 + b_c)];
              dst[a_c * nb + b_c] = acc;
            }
          }
        }
      }
      for (int sc_l = 0; sc_l < n_slab_aux; ++sc_l) {
        const int sc = p0_shell + sc_l;
        const int lc = angular[sc];
        const int mc = sph_size(lc), nc = cart_size(lc);
        const int q_l0 = sph_offsets[sc] - slab.q_sph_start;
        const double* Tc = transform_table + transform_offset(lc);
        double* cbase = cart_slab.data() + Size(slab_aux_cart_off[sc_l]) * n_pair_cart;
        for (Size idx = 0; idx < n_pair_cart; ++idx) {
          double* dst = cbase + idx * nc;
          for (int q_c = 0; q_c < nc; ++q_c) {
            double acc = 0.0;
            for (int p = 0; p < mc; ++p)
              acc += Tc[p * nc + q_c] * m2[Size(q_l0 + p) * n_pair_cart + idx];
            dst[q_c] = acc;
          }
        }
      }
      for (int sc_l = 0; sc_l < n_slab_aux; ++sc_l) {
        const Index sc = p0_shell + sc_l;
        const int nc = cart_size(angular[sc]);
        for (int pair_i = 0; pair_i < n_orb_pairs; ++pair_i) {
          const Size start = Size(slab_aux_cart_off[sc_l]) * n_pair_cart +
                             Size(pair_cart_off[pair_i]) * nc;
          ShellBlockMeta shell;
          shell.init_shells(false, pair_sa[pair_i], pair_sb[pair_i], sc, start, angular,
                            prim_offsets, centers);
          double sums[channels]{};
          weighted_terms_host(shell, exponents, radial, cart_slab.data() + start, op, fit, sums);
          *weighted_value_out += sums[0];
          const int aa = atom[shell.sa], ab = atom[shell.sb], ac = atom[shell.sc];
          for (int q = 0; q < 3; ++q) {
            atom_gradient_out[3 * aa + q] += sums[1 + q];
            atom_gradient_out[3 * ab + q] += sums[4 + q];
            atom_gradient_out[3 * ac + q] += sums[7 + q];
          }
        }
      }
    }
  }
  if constexpr (!Forward) {
    if (!std::isfinite(*weighted_value_out) ||
        !std::all_of(atom_gradient_out, atom_gradient_out + Size(natoms) * 3,
                     [](double x) { return std::isfinite(x); }))
      return -3;
  }
  return 0;
#else
  DeviceMemory buffer;
  cudaError_t status = cudaMalloc(&buffer.ptr, total_layout.length);
  if (status != cudaSuccess) return status;
  auto doubles = [&](Size at) { return reinterpret_cast<double*>(buffer.ptr + at); };
  auto indices = [&](Size at) { return reinterpret_cast<Index*>(buffer.ptr + at); };
  auto ints = [&](Size at) { return reinterpret_cast<int*>(buffer.ptr + at); };
  auto upload = [&](auto* target, const auto* source, Size count) {
    if (status == cudaSuccess && count > 0)
      status = cudaMemcpy(target, source, count * sizeof(*source), cudaMemcpyHostToDevice);
  };
  upload(doubles(exp_pos), exponents, nprimitive);
  upload(doubles(rad_pos), radial, nprimitive);
  upload(doubles(ctr_pos), centers, Size(nshell) * 3);
  upload(doubles(tab_pos), transform_table, kTransformTableSize);
  upload(indices(ang_pos), angular, nshell);
  if constexpr (!Forward) upload(indices(atm_pos), atom, nshell);
  upload(indices(prm_pos), prim_offsets, Size(nshell) + 1);
  upload(indices(sof_pos), sph_offsets, nshell);
  upload(indices(oco_pos), orb_cart_off.data(), Size(n_orb_sh) + 1);
  upload(indices(ocs_pos), orb_cart_to_sh.data(), norb_cart);
  upload(indices(ocl_pos), orb_cart_loc.data(), norb_cart);
  upload(indices(oss_pos), orb_sph_to_sh.data(), norb_sph);
  upload(indices(osl_pos), orb_sph_loc.data(), norb_sph);
  upload(indices(psa_pos), pair_sa.data(), n_orb_pairs);
  upload(indices(psb_pos), pair_sb.data(), n_orb_pairs);
  upload(indices(pco_pos), pair_cart_off.data(), Size(n_orb_pairs) + 1);
  if (status == cudaSuccess) status = cudaMemset(ints(nf_pos), 0, sizeof(int));
  if constexpr (!Forward) {
    if (status == cudaSuccess)
      status = cudaMemset(doubles(vg_pos), 0, (1 + Size(natoms) * 3) * sizeof(double));
  }
  if (status != cudaSuccess) return status;

  std::vector<Index> slab_aux_cart_off(n_aux_sh + 1);
  constexpr int kFlatThreads = 256;

  for (const auto& slab : slabs) {
    const int n_slab_aux = slab.p1 - slab.p0;
    const int p0_shell = n_orb_sh + slab.p0;
    int run_c = 0;
    for (int s = 0; s < n_slab_aux; ++s) {
      slab_aux_cart_off[s] = run_c;
      run_c += cart_size(angular[p0_shell + s]);
    }
    slab_aux_cart_off[n_slab_aux] = run_c;
    status = cudaMemcpy(indices(sac_pos), slab_aux_cart_off.data(),
                        (Size(n_slab_aux) + 1) * sizeof(Index), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return status;

    double* sph_slice = sph_io + Size(slab.q_sph_start) * norb_sph * norb_sph;
    const Size sph_slab_elems = Size(slab.q_sph_count) * norb_sph * norb_sph;

    if constexpr (Forward) {
      dim3 grid_int(static_cast<unsigned>(n_orb_pairs), static_cast<unsigned>(n_slab_aux));
      cartesian_ao_slab_blocks<<<grid_int, threads>>>(
          p0_shell, n_orb_pairs, n_pair_cart, op, fit, indices(psa_pos), indices(psb_pos),
          indices(pco_pos), indices(sac_pos), indices(ang_pos), indices(prm_pos), doubles(exp_pos),
          doubles(rad_pos), doubles(ctr_pos), doubles(cart_pos));

      dim3 grid_c(static_cast<unsigned>((n_pair_cart + kFlatThreads - 1) / kFlatThreads),
                  static_cast<unsigned>(n_slab_aux));
      ao_unpack_c_kernel<<<grid_c, kFlatThreads>>>(
          p0_shell, n_slab_aux, slab.q_sph_start, n_pair_cart, indices(ang_pos), indices(sof_pos),
          indices(sac_pos), doubles(tab_pos), doubles(cart_pos), doubles(m2_pos));

      dim3 grid_a(static_cast<unsigned>(n_orb_pairs), static_cast<unsigned>(slab.q_sph_count));
      ao_unpack_a_kernel<<<grid_a, threads>>>(
          n_orb_pairs, slab.q_sph_count, norb_sph, norb_cart, n_pair_cart, indices(psa_pos),
          indices(psb_pos), indices(pco_pos), indices(ang_pos), indices(sof_pos), indices(oco_pos),
          doubles(tab_pos), doubles(m2_pos), doubles(m1_pos));

      const unsigned grid_b =
          static_cast<unsigned>((sph_slab_elems + kFlatThreads - 1) / kFlatThreads);
      ao_unpack_b_kernel<<<grid_b, kFlatThreads>>>(
          slab.q_sph_count, norb_sph, norb_cart, indices(ang_pos), indices(oss_pos),
          indices(osl_pos), indices(oco_pos), doubles(tab_pos), doubles(m1_pos), doubles(sph_pos),
          ints(nf_pos));

      status = cudaGetLastError();
      if (status != cudaSuccess) return status;
      status = cudaMemcpy(sph_slice, doubles(sph_pos), sph_slab_elems * sizeof(double),
                          cudaMemcpyDeviceToHost);
      if (status != cudaSuccess) return status;
    } else {
      status = cudaMemcpy(doubles(sph_pos), sph_slice, sph_slab_elems * sizeof(double),
                          cudaMemcpyHostToDevice);
      if (status != cudaSuccess) return status;

      const Size b_elems = Size(slab.q_sph_count) * norb_sph * norb_cart;
      const unsigned grid_b = static_cast<unsigned>((b_elems + kFlatThreads - 1) / kFlatThreads);
      ao_pack_b_kernel<<<grid_b, kFlatThreads>>>(
          slab.q_sph_count, norb_sph, norb_cart, indices(ang_pos), indices(sof_pos),
          indices(oss_pos), indices(ocs_pos), indices(ocl_pos), doubles(tab_pos), doubles(sph_pos),
          doubles(m1_pos), ints(nf_pos));

      dim3 grid_a(static_cast<unsigned>(n_orb_pairs), static_cast<unsigned>(slab.q_sph_count));
      ao_pack_a_kernel<<<grid_a, threads>>>(
          n_orb_pairs, slab.q_sph_count, norb_sph, norb_cart, n_pair_cart, indices(psa_pos),
          indices(psb_pos), indices(pco_pos), indices(ang_pos), indices(sof_pos), indices(oco_pos),
          doubles(tab_pos), doubles(m1_pos), doubles(m2_pos));

      dim3 grid_c(static_cast<unsigned>((n_pair_cart + kFlatThreads - 1) / kFlatThreads),
                  static_cast<unsigned>(n_slab_aux));
      ao_pack_c_kernel<<<grid_c, kFlatThreads>>>(
          p0_shell, n_slab_aux, slab.q_sph_start, n_pair_cart, indices(ang_pos), indices(sof_pos),
          indices(sac_pos), doubles(tab_pos), doubles(m2_pos), doubles(cart_pos));

      dim3 grid_int(static_cast<unsigned>(n_orb_pairs), static_cast<unsigned>(n_slab_aux));
      weighted_ao_slab_blocks<<<grid_int, threads>>>(
          p0_shell, n_orb_pairs, n_pair_cart, op, fit, indices(psa_pos), indices(psb_pos),
          indices(pco_pos), indices(sac_pos), indices(ang_pos), indices(atm_pos), indices(prm_pos),
          doubles(exp_pos), doubles(rad_pos), doubles(ctr_pos), doubles(cart_pos), doubles(vg_pos));

      status = cudaGetLastError();
      if (status != cudaSuccess) return status;
    }
  }

  int nonfinite = 0;
  status = cudaMemcpy(&nonfinite, ints(nf_pos), sizeof(int), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) return status;
  if constexpr (Forward) {
    return nonfinite ? -3 : 0;
  } else {
    if (nonfinite) return -1;
    std::vector<double> vg(1 + Size(natoms) * 3);
    status = cudaMemcpy(vg.data(), doubles(vg_pos), vg.size() * sizeof(double),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return status;
    if (!std::all_of(vg.begin(), vg.end(), [](double x) { return std::isfinite(x); })) return -3;
    *weighted_value_out = vg[0];
    std::copy_n(vg.data() + 1, Size(natoms) * 3, atom_gradient_out);
    return 0;
  }
#endif
}
}  // namespace

extern "C" int SINK_EXPORT(int ncenter, int op, double beta, int nshell, int nprimitive, int nblock,
                           int nweights, const Index* angular, const Index* prim_offsets,
                           const double* exponents, const double* radial, const double* centers,
                           const Index* block_shells, const Index* weight_offsets,
                           const double* weights, double* result, Size max_device_bytes,
                           Size* device_bytes) {
  return run_integrals<false>(ncenter, op, beta, kDefaultSlaterFitNormalizationCode, nshell,
                              nprimitive, nblock, nweights, angular, prim_offsets, exponents,
                              radial, centers, block_shells, weight_offsets, weights, result,
                              max_device_bytes, device_bytes);
}

// Raw contracted Cartesian values use exactly the shell and element ordering
// expected by ShellProjection.spherical_values(). No AO weights are inputs.
extern "C" int FORWARD_EXPORT(int ncenter, int op, double beta, int nshell, int nprimitive,
                              int nblock, int ncart, const Index* angular,
                              const Index* prim_offsets, const double* exponents,
                              const double* radial, const double* centers,
                              const Index* block_shells, const Index* cart_offsets, double* result,
                              Size max_device_bytes, Size* device_bytes) {
  return run_integrals<true>(ncenter, op, beta, kDefaultSlaterFitNormalizationCode, nshell,
                             nprimitive, nblock, ncart, angular, prim_offsets, exponents, radial,
                             centers, block_shells, cart_offsets, nullptr, result,
                             max_device_bytes, device_bytes);
}

extern "C" int SINK_EX_EXPORT(int ncenter, int op, double beta, int slater_normalization,
                              int nshell, int nprimitive, int nblock, int nweights,
                              const Index* angular, const Index* prim_offsets,
                              const double* exponents, const double* radial, const double* centers,
                              const Index* block_shells, const Index* weight_offsets,
                              const double* weights, double* result, Size max_device_bytes,
                              Size* device_bytes) {
  return run_integrals<false>(ncenter, op, beta, slater_normalization, nshell, nprimitive, nblock,
                              nweights, angular, prim_offsets, exponents, radial, centers,
                              block_shells, weight_offsets, weights, result, max_device_bytes,
                              device_bytes);
}

extern "C" int FORWARD_EX_EXPORT(int ncenter, int op, double beta, int slater_normalization,
                                 int nshell, int nprimitive, int nblock, int ncart,
                                 const Index* angular, const Index* prim_offsets,
                                 const double* exponents, const double* radial,
                                 const double* centers, const Index* block_shells,
                                 const Index* cart_offsets, double* result, Size max_device_bytes,
                                 Size* device_bytes) {
  return run_integrals<true>(ncenter, op, beta, slater_normalization, nshell, nprimitive, nblock,
                             ncart, angular, prim_offsets, exponents, radial, centers, block_shells,
                             cart_offsets, nullptr, result, max_device_bytes, device_bytes);
}

extern "C" int SPH_FORWARD_EXPORT(int ncenter, int op, double beta, int slater_normalization,
                                  int nshell, int orbital_shell_count, int nprimitive, int norb_sph,
                                  int naux_sph, const Index* angular, const Index* prim_offsets,
                                  const double* exponents, const double* radial,
                                  const double* centers, const Index* sph_offsets,
                                  const double* transform_table, double* sph_result,
                                  Size max_device_bytes, Size* device_bytes) {
  return run_spherical<true>(ncenter, op, beta, slater_normalization, nshell, orbital_shell_count,
                             nprimitive, 0, norb_sph, naux_sph, angular, nullptr, prim_offsets,
                             exponents, radial, centers, sph_offsets, transform_table, sph_result,
                             nullptr, nullptr, max_device_bytes, device_bytes);
}

extern "C" int SPH_WEIGHTED_EXPORT(int ncenter, int op, double beta, int slater_normalization,
                                   int nshell, int orbital_shell_count, int nprimitive, int natoms,
                                   int norb_sph, int naux_sph, const Index* angular,
                                   const Index* atom, const Index* prim_offsets,
                                   const double* exponents, const double* radial,
                                   const double* centers, const Index* sph_offsets,
                                   const double* transform_table, const double* sph_weights,
                                   double* weighted_value_out, double* atom_gradient_out,
                                   Size max_device_bytes, Size* device_bytes) {
  return run_spherical<false>(ncenter, op, beta, slater_normalization, nshell, orbital_shell_count,
                              nprimitive, natoms, norb_sph, naux_sph, angular, atom, prim_offsets,
                              exponents, radial, centers, sph_offsets, transform_table,
                              const_cast<double*>(sph_weights), weighted_value_out,
                              atom_gradient_out, max_device_bytes, device_bytes);
}
