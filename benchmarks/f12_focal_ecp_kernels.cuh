// Scalar projector primitives shared by the CUDA molecular source and its host arithmetic tests.
#pragma once
#include <cmath>
#if defined(__CUDACC__)
#define GQC_ECP_HD __host__ __device__
#else
#define GQC_ECP_HD
#endif
namespace generativeqc::focal_ecp {
constexpr int channels = 16, coefficients = 28;
struct Basis {
  const int* ao;  // l, atom, primitive begin, primitive end per ordered spherical row
  const double *xyz, *alpha, *radial, *transforms;
};

GQC_ECP_HD inline void point_jet(Basis b, int a, int active, int jets, double radius,
                                 const double* direction, double* out) {
  const int* row = b.ao + 4 * a;
  const int l = row[0], atom = row[1];
  double x[3], x2 = 0, powers[3][7];
  for (int k = 0; k < 3; ++k) {
    x[k] = (b.xyz[3 * active + k] - b.xyz[3 * atom + k]) + radius * direction[k];
    x2 += x[k] * x[k];
    powers[k][0] = 1.;
    for (int j = 1; j <= l; ++j) powers[k][j] = powers[k][j - 1] * x[k];
  }
  double radial = 0., dr = 0.;
  for (int j = row[2]; j < row[3]; ++j) {
    double term = b.radial[j] * exp(-b.alpha[j] * x2);
    radial += term;
    if (jets > 1) dr -= 2 * b.alpha[j] * term;
  }
  double p = 0., dp[3] = {};
  if (radial != 0. || dr != 0.) {
    int col = 0;
    for (int ix = l; ix >= 0; --ix)
      for (int iy = l - ix; iy >= 0; --iy, ++col) {
        const int iz = l - ix - iy;
        const double c = b.transforms[coefficients * a + col];
        if (c == 0.) continue;
        p += c * powers[0][ix] * powers[1][iy] * powers[2][iz];
        if (jets == 1) continue;
        if (ix) dp[0] += c * ix * powers[0][ix - 1] * powers[1][iy] * powers[2][iz];
        if (iy) dp[1] += c * iy * powers[0][ix] * powers[1][iy - 1] * powers[2][iz];
        if (iz) dp[2] += c * iz * powers[0][ix] * powers[1][iy] * powers[2][iz - 1];
      }
  }
  out[0] = p * radial;
  if (jets > 1)
    for (int k = 0; k < 3; ++k) out[k + 1] = dp[k] * radial + p * dr * x[k];
}
GQC_ECP_HD inline double channel_weight(int active, int channel, int nterm, const int* terms,
                                        const double* parameters, double r, double dr) {
  double z = 0.;
  for (int t = 0; t < nterm; ++t)
    if (terms[3 * t] == active && terms[3 * t + 1] == channel) {
      double w = dr * parameters[2 * t + 1] * exp(-parameters[2 * t] * r * r);
      for (int p = 0; p < terms[3 * t + 2]; ++p) w *= r;
      z += w;
    }
  return z;
}
}  // namespace generativeqc::focal_ecp
#undef GQC_ECP_HD
