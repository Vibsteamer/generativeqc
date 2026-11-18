// Qualification adapter for contracted, real molecular DF and F12 shell adjoints.
// Cartesian ordering is CCA. Callers own shell-to-atom and pure-to-Cartesian maps.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "integrals/f12_geminal.hpp"

#ifdef GQC_SINK_COULOMB
#include "molecular-coulomb.hpp"
namespace poly = generativeqc::scf::generated_df_auxiliary_h_derivatives;
#else
#include "molecular-geminal.hpp"
namespace poly = generativeqc::scf::generated_f12_df_auxiliary_h_derivatives;
#endif
namespace gi = generativeqc::integrals;

#if defined(GQC_SINK_HOST_REFERENCE)
#define SINK_HD
#define SINK_DEVICE
#ifdef GQC_SINK_COULOMB
#define SINK_EXPORT gqc_f12_weighted_shells_coulomb_host_reference
#else
#define SINK_EXPORT gqc_f12_weighted_shells_geminal_host_reference
#endif
#elif defined(__CUDACC__)
#define SINK_HD __host__ __device__
#define SINK_DEVICE __device__
#ifdef GQC_SINK_COULOMB
#define SINK_EXPORT gqc_f12_weighted_shells_coulomb
#else
#define SINK_EXPORT gqc_f12_weighted_shells_geminal
#endif
#else
#error "Use NVCC for the CUDA sink or explicitly build the separate host reference"
#endif

namespace {
constexpr int channels = 10;
constexpr int threads = 64;
using Index = std::int32_t;
using Size = std::uint64_t;

struct ShellBatch {
  const Index *angular, *primitive_offsets, *block_shells, *weight_offsets;
  const double *exponents, *radial, *centers, *weights;
};

SINK_HD int cart_size(int l) { return (l + 1) * (l + 2) / 2; }

SINK_DEVICE poly::Angular component(int l, int index) {
  int degree_yz = 0;
  while (index > degree_yz) index -= ++degree_yz;
  return {static_cast<unsigned>(l - degree_yz), static_cast<unsigned>(degree_yz - index),
          static_cast<unsigned>(index)};
}

SINK_DEVICE poly::Vec3 center(const ShellBatch& data, Index shell) {
  auto xyz = data.centers + Size(3) * shell;
  return {xyz[0], xyz[1], xyz[2]};
}

// Each lane accumulates disjoint Cartesian/primitive products. Combining lane
// sums within one block requires neither inter-block atomics nor global scratch.
SINK_DEVICE void weighted_terms(ShellBatch data, int ncenter, int op, const gi::FittedSlater& fit,
                                int block, int lane, int stride, double (&sum)[channels]) {
  const auto* shells = data.block_shells + Size(block) * ncenter;
  const bool metric = ncenter == 2;
  const Index sa = shells[0], sb = metric ? sa : shells[1], sc = shells[ncenter - 1];
  const auto A = center(data, sa), B = center(data, sb), C = center(data, sc);
  const int la = data.angular[sa], lb = metric ? 0 : data.angular[sb], lc = data.angular[sc];
  const int nb = cart_size(lb), nc = cart_size(lc);
  const auto* weights = data.weights + data.weight_offsets[block];
  const int ncart = data.weight_offsets[block + 1] - data.weight_offsets[block];
  const int first_a = data.primitive_offsets[sa], first_b = data.primitive_offsets[sb],
            first_c = data.primitive_offsets[sc];
  const int count_a = data.primitive_offsets[sa + 1] - first_a,
            count_b = metric ? 1 : data.primitive_offsets[sb + 1] - first_b,
            count_c = data.primitive_offsets[sc + 1] - first_c;
  const Size terms = Size(ncart) * count_a * count_b * count_c;
  double correction[channels]{};
  for (Size term = lane; term < terms; term += stride) {
    const int cart = term % ncart;
    if (weights[cart] == 0.) continue;
    const Size prim = term / ncart;
    const int k = first_c + prim % count_c, j = first_b + (prim / count_c) % count_b,
              i = first_a + prim / (Size(count_c) * count_b);
    const double w =
        weights[cart] * data.radial[i] * (metric ? 1. : data.radial[j]) * data.radial[k];
    if (w == 0.) continue;
    const auto a = component(la, cart / (nc * nb)), b = component(lb, (cart / nc) % nb),
               c = component(lc, cart % nc);
#ifdef GQC_SINK_COULOMB
    const auto r = poly::evaluate(metric, data.exponents[i], A, a, metric ? 0. : data.exponents[j],
                                  B, b, data.exponents[k], C, c);
#else
    const auto r =
        poly::evaluate(metric, data.exponents[i], A, a, metric ? 0. : data.exponents[j], B, b,
                       data.exponents[k], C, c, static_cast<gi::GeminalProduct>(op), fit);
#endif
    const double values[channels] = {r.value,    r.first.x,  r.first.y, r.first.z, r.second.x,
                                     r.second.y, r.second.z, r.third.x, r.third.y, r.third.z};
    for (int q = 0; q < channels; ++q) {
      const double adjusted = w * values[q] - correction[q];
      const double next = sum[q] + adjusted;
      correction[q] = (next - sum[q]) - adjusted;
      sum[q] = next;
    }
  }
}

#ifndef GQC_SINK_HOST_REFERENCE
__global__ void weighted_blocks(ShellBatch data, int ncenter, int op, gi::FittedSlater fit,
                                double* result) {
  __shared__ double partial[channels][threads];
  double sum[channels]{};
  weighted_terms(data, ncenter, op, fit, blockIdx.x, threadIdx.x, threads, sum);
  for (int q = 0; q < channels; ++q) partial[q][threadIdx.x] = sum[q];
  __syncthreads();
  for (int step = threads / 2; step; step /= 2) {
    if (threadIdx.x < step) {
      for (int q = 0; q < channels; ++q) partial[q][threadIdx.x] += partial[q][threadIdx.x + step];
    }
    __syncthreads();
  }
  if (threadIdx.x < channels)
    result[Size(blockIdx.x) * channels + threadIdx.x] = partial[threadIdx.x][0];
}
#endif

bool supported(ShellBatch data, int ncenter, int nshell, int nprimitive, int nblock, int nweights) {
  if (data.primitive_offsets[0] != 0 || data.primitive_offsets[nshell] != nprimitive ||
      data.weight_offsets[0] != 0 || data.weight_offsets[nblock] != nweights)
    return false;
  for (int s = 0; s < nshell; ++s) {
    if (data.angular[s] < 0 || data.angular[s] > 5 || data.primitive_offsets[s] < 0 ||
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
      if (s < 0 || s >= nshell || (ncenter == 3 && q < 2 && data.angular[s] > 4)) return false;
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
         std::all_of(data.weights, data.weights + nweights, finite);
}

// One exact-sized global allocation. Layout is doubles, followed by int32s.
struct Layout {
  Size length = 0;
  template <class T>
  Size take(Size count) {
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
}  // namespace

// Status: 0 success, -1 incompatible/malformed ABI data, -2 explicit storage
// budget exceeded, -3 nonfinite scientific output, >0 CUDA runtime error.
// device_bytes reports required explicit device-buffer bytes even if over budget.
// CUDA runtime context, code, stack, and caches belong to the CUDA driver.
extern "C" int SINK_EXPORT(int ncenter, int op, double beta, int nshell, int nprimitive, int nblock,
                           int nweights, const Index* angular, const Index* prim_offsets,
                           const double* exponents, const double* radial, const double* centers,
                           const Index* block_shells, const Index* weight_offsets,
                           const double* weights, double* result, Size max_device_bytes,
                           Size* device_bytes) {
  if (!device_bytes) return -1;
  *device_bytes = 0;
  if ((ncenter != 2 && ncenter != 3) || nshell <= 0 || nprimitive <= 0 || nblock <= 0 ||
      nweights <= 0 || !angular || !prim_offsets || !exponents || !radial || !centers ||
      !block_shells || !weight_offsets || !weights || !result || !max_device_bytes)
    return -1;
  gi::FittedSlater fit{};
#ifdef GQC_SINK_COULOMB
  if (op != 0) return -1;
#else
  if (op < 0 || op > 3 ||
      !gi::make_fitted_slater(beta, gi::SlaterFitNormalization::negative_exponential, fit))
    return -1;
#endif
  Layout layout;
  const Size exponent_pos = layout.take<double>(nprimitive),
             radial_pos = layout.take<double>(nprimitive),
             center_pos = layout.take<double>(Size(nshell) * 3),
             weight_pos = layout.take<double>(nweights),
             result_pos = layout.take<double>(Size(nblock) * channels),
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
    double sums[channels]{};
    weighted_terms(host, ncenter, op, fit, block, 0, 1, sums);
    std::copy_n(sums, channels, result + Size(block) * channels);
  }
#else
  DeviceMemory buffer;
  cudaError_t status = cudaMalloc(&buffer.ptr, layout.length);
  if (status != cudaSuccess) return status;
  auto doubles = [&](Size at) { return reinterpret_cast<double*>(buffer.ptr + at); };
  auto indices = [&](Size at) { return reinterpret_cast<Index*>(buffer.ptr + at); };
  auto upload = [&](auto* target, const auto* source, Size count) {
    if (status == cudaSuccess)
      status = cudaMemcpy(target, source, count * sizeof(*source), cudaMemcpyHostToDevice);
  };
  upload(doubles(exponent_pos), exponents, nprimitive);
  upload(doubles(radial_pos), radial, nprimitive);
  upload(doubles(center_pos), centers, Size(nshell) * 3);
  upload(doubles(weight_pos), weights, nweights);
  upload(indices(angular_pos), angular, nshell);
  upload(indices(primitive_pos), prim_offsets, Size(nshell) + 1);
  upload(indices(shell_pos), block_shells, Size(nblock) * ncenter);
  upload(indices(offset_pos), weight_offsets, Size(nblock) + 1);
  if (status != cudaSuccess) return status;
  const ShellBatch gpu{indices(angular_pos), indices(primitive_pos), indices(shell_pos),
                       indices(offset_pos),  doubles(exponent_pos),  doubles(radial_pos),
                       doubles(center_pos),  doubles(weight_pos)};
  // clang-format off
  weighted_blocks<<<nblock, threads>>>(gpu, ncenter, op, fit, doubles(result_pos));
  // clang-format on
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  status = cudaMemcpy(result, doubles(result_pos), Size(nblock) * channels * sizeof(double),
                      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) return status;
#endif
  const auto finite = [](double x) { return std::isfinite(x); };
  return std::all_of(result, result + Size(nblock) * channels, finite) ? 0 : -3;
}
