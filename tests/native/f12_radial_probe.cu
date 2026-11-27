#include "integrals/f12_geminal.hpp"

namespace {
namespace gi = generativeqc::integrals;
constexpr unsigned max_order = 20, inputs = 22, outputs = 35;
#ifdef __CUDACC__
#define TEST_HD __host__ __device__
#else
#define TEST_HD
#endif
struct Geometry {
  double inverse_two_p, inverse_two_q, rho, difference[3], shifts[4][3], decay[4][3],
      product_scales[4], prefactor, boys[max_order + 1];
};

template <unsigned Mask>
TEST_HD bool geometry(const double* in, Geometry& result) {
  const int op = static_cast<int>(in[0]);
  const unsigned order = static_cast<unsigned>(in[2]);
  if (op < 3)
    return gi::make_geminal_geometry<Geometry, max_order, Mask>(
        in + 6, in + 10, order, static_cast<gi::GaussianGeminalKernel>(op), in[3], result);
  gi::FittedSlater left, right;
  const auto convention = static_cast<gi::SlaterFitNormalization>(static_cast<int>(in[5]));
  if (!gi::make_fitted_slater(in[3], convention, left) ||
      !gi::make_fitted_slater(in[4], convention, right))
    return false;
  return gi::make_fitted_geminal_geometry<Geometry, max_order, Mask>(
      in + 6, in + 10, order, static_cast<gi::GeminalProduct>(op - 3), left,
      in[3] == in[4] ? left : right, result);
}

TEST_HD void evaluate(const double* in, double* out) {
  const auto op = static_cast<gi::GaussianGeminalKernel>(static_cast<int>(in[0]));
  const int mask = static_cast<int>(in[1]);
  const unsigned order = static_cast<unsigned>(in[2]);
  const double z = in[3];
  if (mask == -1) {
    out[0] = gi::gaussian_geminal_moments<max_order>(order, in[4], in[5], z, op, out + 1);
    return;
  }
  Geometry s{};
  if (mask == 0)
    out[0] = geometry<0>(in, s);
  else if (mask == 8)
    out[0] = geometry<8>(in, s);
  else if (mask == 10)
    out[0] = geometry<10>(in, s);
  else
    out[0] = 0.;
  if (out[0] == 0.) return;
  for (unsigned n = 0; n <= order; ++n) out[n + 1] = s.boys[n];
  out[22] = s.prefactor * s.boys[0];
  if (order == 0) return;
  for (unsigned slot = 0; slot < 4; ++slot)
    for (unsigned axis = 0; axis < 3; ++axis) {
      const double dt = (slot < 2 ? 2. : -2.) * s.rho * s.product_scales[slot] * s.difference[axis];
      out[23 + 3 * slot + axis] = s.prefactor * (s.decay[slot][axis] * s.boys[0] - dt * s.boys[1]);
    }
}
#undef TEST_HD

#ifdef __CUDACC__
__global__ void evaluate_rows(int count, const double* in, double* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) evaluate(in + i * inputs, out + i * outputs);
}
#endif
}  // namespace

extern "C" int evaluate_f12_rows(int count, const double* in, double* out) {
#ifdef __CUDACC__
  double *device_in = nullptr, *device_out = nullptr;
  const auto in_bytes = count * inputs * sizeof(double),
             out_bytes = count * outputs * sizeof(double);
  auto status = cudaMalloc(&device_in, in_bytes);
  if (status == cudaSuccess) status = cudaMalloc(&device_out, out_bytes);
  if (status == cudaSuccess) status = cudaMemcpy(device_in, in, in_bytes, cudaMemcpyHostToDevice);
  if (status == cudaSuccess) status = cudaMemset(device_out, 0, out_bytes);
  if (status == cudaSuccess) {
    evaluate_rows<<<(count + 63) / 64, 64>>>(count, device_in, device_out);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess)
    status = cudaMemcpy(out, device_out, out_bytes, cudaMemcpyDeviceToHost);
  if (device_out) cudaFree(device_out);
  if (device_in) cudaFree(device_in);
  return static_cast<int>(status);
#else
  for (int i = 0; i < count; ++i) evaluate(in + i * inputs, out + i * outputs);
  return 0;
#endif
}
