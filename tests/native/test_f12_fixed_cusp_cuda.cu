#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cc/f12_energy_gradient_cuda.cuh"
#include "f12_fixed_cusp_oracle.hpp"
#include "cc/f12_fixed_cusp_cuda.cuh"
#include "runtime/cuda_resources.cuh"

namespace {

using f12_oracle::near;
namespace gpu = generativeqc::cc::f12_cuda;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T>
generativeqc::runtime::OwnedCudaBuffer<T> upload(const std::vector<T>& host, cudaStream_t stream) {
  generativeqc::runtime::OwnedCudaBuffer<T> device(0, host.size(), stream);
  check(cudaMemcpyAsync(device.get(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
  return device;
}

template <class T>
std::vector<T> read(const generativeqc::runtime::OwnedCudaBuffer<T>& device, cudaStream_t stream) {
  std::vector<T> values(device.size());
  check(cudaMemcpyAsync(values.data(), device.get(), values.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  return values;
}

void fixture(std::size_t o, std::size_t v) {
  const auto n1 = o * v, n2 = n1 * n1;
  const auto [raw, canonical, r1, t1, base1, base2, expected1, expected2] = f12_oracle::fixture(o, v);

  generativeqc::runtime::OwnedCudaStream stream(0);
  const auto st = stream.get();
  auto wvo = upload(raw.w_vo, st), wvv = upload(raw.w_vv, st), c = upload(raw.c_f12, st);
  auto t = upload(t1, st), src1 = upload(base1, st), src2 = upload(base2, st);
  generativeqc::runtime::OwnedCudaBuffer<double> constant(0, n1, st), rhs1(0, n1, st), rhs2(0, n2, st);
  auto error = upload(std::vector<int>{0}, st);
  gpu::launch_prepare_constants(o, v, wvo.get(), wvv.get(), c.get(), constant.get(), error.get(), st);
  gpu::launch_subtract_energy_gradient(o, v, constant.get(), wvv.get(), c.get(), t.get(),
                                      src1.get(), src2.get(), rhs1.get(), rhs2.get(), error.get(), st);
  const auto r = read(constant, st), actual1 = read(rhs1, st), actual2 = read(rhs2, st);
  if (read(error, st)[0] || read(wvv, st) != canonical.w_vv || read(c, st) != canonical.c_f12)
    throw std::runtime_error("CUDA fixed-cusp preparation failed");
  for (std::size_t k = 0; k < n1; ++k) {
    near(r[k], r1[k], 0., "CUDA singles constant");
    near(actual1[k], expected1[k], 2e-12, "CUDA singles RHS");
  }
  for (std::size_t k = 0; k < n2; ++k) near(actual2[k], expected2[k], 2e-12, "CUDA doubles RHS");

  const double bad = std::numeric_limits<double>::infinity();
  check(cudaMemcpyAsync(t.get(), &bad, sizeof(bad), cudaMemcpyHostToDevice, st));
  gpu::launch_subtract_energy_gradient(o, v, constant.get(), wvv.get(), c.get(), t.get(),
                                      src1.get(), src2.get(), rhs1.get(), rhs2.get(), error.get(), st);
  if (!read(error, st)[0]) throw std::runtime_error("nonfinite CUDA RHS escaped its error flag");
}

}  // namespace

int main() {
  try {
    int devices{};
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && !devices)) return 77;
    check(status);
    check(cudaSetDevice(0));
    fixture(1, 1);
    fixture(2, 3);
    fixture(3, 2);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
