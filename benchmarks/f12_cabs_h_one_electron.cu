// Opt-in Q-CABS h-shell qualification. One Cartesian shell block covers all
// nuclei, positive matrix derivatives, and real Gaussian radial contractions.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "scf/cuda/coulomb_auxiliary.cuh"
#include "scf/cuda/hermite_recurrence.cuh"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace {
using namespace generativeqc::scf::cuda_execution;
using Size = std::size_t;
constexpr int MaxL = 5;
constexpr int MaxPair = 2 * MaxL;
struct Axis {
  Dual e[MaxPair + 1];  // Partial derivative with respect to A-B.
  Dual kinetic;
};
struct Powers {
  int v[3];
};
GENERATIVEQC_ERI_DEVICE Powers component(int l, int pos) {
  for (int x = l; x >= 0; --x)
    for (int y = l - x; y >= 0; --y)
      if (pos-- == 0) return {{x, y, l - x - y}};
  return {};
}

GENERATIVEQC_ERI_DEVICE void prepare_axis(int i, int j, double a, double b, double A, double B,
                                          Axis& result) {
  const Dual x{A, 1.0}, y{B, 0.0};
  const Dual P = (a * x + b * y) / (a + b);
  // T raises the ket by two quanta. Dual propagation gives its basis-center
  // derivative and preserves the original radial normalization.
  HermiteCoefficients<Dual, MaxL + 1, MaxL + 3, 2 * MaxL + 4> h;
  fill_hermite(i, j + 2, P, x, y, a, b, h);
  for (int t = 0; t <= i + j; ++t) result.e[t] = h.at(i, j, t);
  result.kinetic = b * (2 * j + 1) * h.at(i, j, 0) - (2 * b * b) * h.at(i, j + 2, 0);
  if (j >= 2) result.kinetic = result.kinetic - (0.5 * j * (j - 1)) * h.at(i, j - 2, 0);
}

struct Matrices {
  double* block;
  Size stride, rows;
  GENERATIVEQC_ERI_DEVICE double& entry(int op, Size row) {
    return block[(op * rows + row) * stride];
  }
  GENERATIVEQC_ERI_DEVICE void center(int op, int atom, int axis, double contribution) {
    entry(op, 1 + Size(atom) * 3 + axis) += contribution;
  }
};
struct Pair {
  int la, lb, first, second, nfirst, nsecond, natom;
  const double *alpha, *radial, *coords;
  const int* charges;
  GENERATIVEQC_ERI_DEVICE Vec3<double> xyz(int atom) const {
    return {coords[3 * atom], coords[3 * atom + 1], coords[3 * atom + 2]};
  }
};

GENERATIVEQC_ERI_DEVICE double at_axis(const Vec3<double>& v, int axis) {
  return axis == 0 ? v.x : axis == 1 ? v.y : v.z;
}
GENERATIVEQC_ERI_DEVICE void primitive(const Pair& pair, const Powers& u, const Powers& v, int ii,
                                       int jj, Matrices dst) {
  const double a = pair.alpha[ii], b = pair.alpha[jj], p = a + b;
  const double w = pair.radial[ii] * pair.radial[jj];
  const auto A = pair.xyz(pair.first), B = pair.xyz(pair.second), P = product_center(a, A, b, B);
  Axis axes[3];
  int limits[3];
  for (int k = 0; k < 3; ++k) {
    limits[k] = u.v[k] + v.v[k];
    prepare_axis(u.v[k], v.v[k], a, b, at_axis(A, k), at_axis(B, k), axes[k]);
  }
  const double overlap_scale = w * pow(kPi / p, 1.5);
  // The three one-dimensional kinetic contributions use the same overlap
  // coefficients. Only A-B enters T and S, including for repeated atoms.
  for (int term = -1; term < 3; ++term) {
    const int op = term < 0 ? 0 : 1;
    Dual factors[3];
    for (int k = 0; k < 3; ++k) factors[k] = (k == term ? axes[k].kinetic : axes[k].e[0]);
    dst.entry(op, 0) += overlap_scale * factors[0].value * factors[1].value * factors[2].value;
    for (int k = 0; k < 3; ++k) {
      const double derivative = overlap_scale * factors[k].derivative * factors[(k + 1) % 3].value *
                                factors[(k + 2) % 3].value;
      dst.center(op, pair.first, k, derivative);
      dst.center(op, pair.second, k, -derivative);
    }
  }
  for (int atom = 0; atom < pair.natom; ++atom) {
    CoulombAuxiliary<double, MaxPair + 1> R;
    fill_coulomb<MaxPair + 1>(
        p, P, pair.xyz(atom), R,
        {unsigned(limits[0]), unsigned(limits[1]), unsigned(limits[2]), true, true});
    const double prefactor = -pair.charges[atom] * w * 2 * kPi / p;
    double value = 0, derivative_ab[3]{}, derivative_pc[3]{};
    for (int x = 0; x <= limits[0]; ++x)
      for (int y = 0; y <= limits[1]; ++y)
        for (int z = 0; z <= limits[2]; ++z) {
          const Dual X = axes[0].e[x], Y = axes[1].e[y], Z = axes[2].e[z];
          const double r = R.at(0, x, y, z), prod = X.value * Y.value * Z.value;
          value += prod * r;
          derivative_ab[0] += X.derivative * Y.value * Z.value * r;
          derivative_ab[1] += X.value * Y.derivative * Z.value * r;
          derivative_ab[2] += X.value * Y.value * Z.derivative * r;
          derivative_pc[0] += prod * R.at(0, x + 1, y, z);
          derivative_pc[1] += prod * R.at(0, x, y + 1, z);
          derivative_pc[2] += prod * R.at(0, x, y, z + 1);
        }
    dst.entry(2, 0) += prefactor * value;
    // R depends on P-C. All nuclei move with their own point charges.
    for (int k = 0; k < 3; ++k) {
      dst.center(2, pair.first, k, prefactor * (derivative_ab[k] + a / p * derivative_pc[k]));
      dst.center(2, pair.second, k, prefactor * (-derivative_ab[k] + b / p * derivative_pc[k]));
      dst.center(2, atom, k, -prefactor * derivative_pc[k]);
    }
  }
}

GENERATIVEQC_ERI_DEVICE void element(const Pair& pair, int index, int stride, double* out) {
  const int nb = (pair.lb + 1) * (pair.lb + 2) / 2;
  const auto u = component(pair.la, index / nb), v = component(pair.lb, index % nb);
  Matrices dst{out + index, Size(stride), 1 + 3 * Size(pair.natom)};
  for (int i = 0; i < pair.nfirst; ++i)
    for (int j = pair.nfirst; j < pair.nfirst + pair.nsecond; ++j) primitive(pair, u, v, i, j, dst);
}
#ifdef __CUDACC__
__global__ void kernel(Pair pair, int stride, double* out) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < stride) element(pair, index, stride, out);
}
#define EXPORT gqc_f12_cabs_h_one_electron
#else
#define EXPORT gqc_f12_cabs_h_one_electron_host_reference
#endif
}  // namespace

// Output is row-major [S,T,V][value, nuclear Cartesian derivatives][cartA,cartB].
// Each primitive coefficient is radial-normalized for its ORIGINAL shell.
// A bound on staged inputs and outputs is checked before an allocation.
extern "C" int EXPORT(int la, int lb, int first, int second, int na, int nb, int natom,
                      const double* exps, const double* radial, const double* coords,
                      const int* charges, double* out, Size count, Size max_bytes, Size* bytes) {
  constexpr int imax = std::numeric_limits<int>::max();
  if (la < 0 || lb < 0 || la > MaxL || lb > MaxL || (la != MaxL && lb != MaxL) || natom <= 0 ||
      natom > imax / 3 || first < 0 || second < 0 || first >= natom || second >= natom || na <= 0 ||
      nb <= 0 || na > imax - nb || !exps || !radial || !coords || !charges || !out || !bytes)
    return 1;
  const Size nprim = Size(na) + nb;
  const int stride = (la + 1) * (la + 2) * (lb + 1) * (lb + 2) / 4;
  if (count != 3 * (1 + 3 * Size(natom)) * stride) return 1;
  const Size nd = 2 * nprim + 3 * Size(natom);
  *bytes = (nd + count) * sizeof(double) + Size(natom) * sizeof(int);
  if (*bytes > max_bytes) return 2;
  for (Size i = 0; i < nprim; ++i)
    if (!std::isfinite(exps[i]) || exps[i] <= 0 || !std::isfinite(radial[i])) return 1;
  for (int i = 0; i < natom; ++i)
    if (charges[i] <= 0 || !std::isfinite(coords[3 * i]) || !std::isfinite(coords[3 * i + 1]) ||
        !std::isfinite(coords[3 * i + 2]))
      return 1;
  Pair pair{la, lb, first, second, na, nb, natom, exps, radial, coords, charges};
#ifdef __CUDACC__
  double* buf{};
  if (cudaMalloc(&buf, *bytes) != cudaSuccess) return 3;
  double *de = buf, *dr = buf + nprim, *dx = dr + nprim, *dout = buf + nd;
  int* dz = reinterpret_cast<int*>(dout + count);
  auto upload = [](void* dst, const void* src, Size size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) == cudaSuccess;
  };
  bool ok = upload(de, exps, nprim * sizeof(double)) &&
            upload(dr, radial, nprim * sizeof(double)) &&
            upload(dx, coords, Size(natom) * 3 * sizeof(double)) &&
            upload(dz, charges, Size(natom) * sizeof(int)) &&
            cudaMemset(dout, 0, count * sizeof(double)) == cudaSuccess;
  if (ok) {
    pair.alpha = de;
    pair.radial = dr;
    pair.coords = dx;
    pair.charges = dz;
    kernel<<<(stride + 31) / 32, 32>>>(pair, stride, dout);
    ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(out, dout, count * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess;
  }
  const bool freed = cudaFree(buf) == cudaSuccess;
  return ok && freed ? 0 : 3;
#else
  std::fill_n(out, count, 0.0);
  for (int index = 0; index < stride; ++index) element(pair, index, stride, out);
  return 0;
#endif
}
