#pragma once

// Compensated two-sided projection of the physical extended-frame DF Fock.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "posthf/cabs_compensated.cuh"
#include "posthf/cabs_cuda_support.cuh"

namespace f12_molecular_fock {
namespace gpu = generativeqc::runtime;
using generativeqc::posthf::cabs::blocks;
using generativeqc::posthf::cabs::check_flags;

using generativeqc::posthf::cabs::Extended;

__device__ Extended checked_pair(const double* source, size_t i, size_t j, int* bad) {
  const double x = source[i], y = source[j];
  if (!isfinite(x) || !isfinite(y) || fabs(x - y) > 1e-12 * fmax(1., fmax(fabs(x), fabs(y))))
    atomicExch(bad, 1);
  Extended mean{};
  mean.add(0.5 * x);
  mean.add(0.5 * y);
  return mean;
}

__global__ void trace_density(const double* fit, size_t occ, size_t mo, size_t aux, Extended* tr) {
  for (size_t a = blockIdx.x * blockDim.x + threadIdx.x; a < aux;
       a += size_t(blockDim.x) * gridDim.x) {
    Extended d{};
    for (size_t i = 0; i < occ; ++i) d.add(2. * fit[(a * occ + i) * mo + i]);
    tr[a] = d;
  }
}

__global__ void ao_operator(const double* h, const double* raw, const Extended* trace, size_t ao,
                            size_t aux, Extended* out, int* bad) {
  const auto count = ao * ao;
  for (size_t x = blockIdx.x * blockDim.x + threadIdx.x; x < count;
       x += size_t(blockDim.x) * gridDim.x) {
    const size_t y = (x % ao) * ao + x / ao;
    Extended total = checked_pair(h, x, y, bad);
    for (size_t a = 0; a < aux; ++a) {
      const auto mean = checked_pair(raw + a * count, x, y, bad);
      total.add_product(trace[a], mean.hi);
      total.add_product(trace[a], mean.lo);
    }
    out[x] = total;
  }
}

__global__ void contract_right(const Extended* a, const double* c, size_t ao, size_t mo,
                               Extended* right) {
  for (size_t p = blockIdx.x * blockDim.x + threadIdx.x; p < ao * mo;
       p += size_t(blockDim.x) * gridDim.x) {
    const size_t i = p / mo, j = p % mo;
    Extended total{};
    for (size_t k = 0; k < ao; ++k) total.add_product(a[i * ao + k], c[k * mo + j]);
    right[p] = total;
  }
}

__global__ void contract_left(const Extended* right, const double* c, const double* exchange,
                              size_t ao, size_t mo, double* f, int* bad) {
  for (size_t p = blockIdx.x * blockDim.x + threadIdx.x; p < mo * mo;
       p += size_t(blockDim.x) * gridDim.x) {
    const size_t i = p / mo, j = p % mo;
    Extended total{};
    for (size_t k = 0; k < ao; ++k) total.add_product(right[k * mo + j], c[k * mo + i]);
    total.add(-exchange[p]);
    f[p] = total.hi + total.lo;
    if (!isfinite(f[p])) atomicExch(bad, 1);
  }
}
}  // namespace f12_molecular_fock

extern "C" int gqc_f12_extended_fock(int device, int atomic, int orbitals, int occupied,
                                     int auxiliaries, const double* h, const double* g,
                                     const double* coefficients, const double* fit,
                                     const double* exchange, std::uint64_t budget, double* out,
                                     std::uint64_t* bytes, char* error, std::size_t error_size) {
  using namespace f12_molecular_fock;
  try {
    if (device < 0 || atomic < 1 || orbitals < 1 || orbitals > atomic || occupied < 1 ||
        occupied >= orbitals || auxiliaries < 1 || !h || !g || !coefficients || !fit || !exchange ||
        !out || !bytes)
      throw std::invalid_argument("invalid dimensions or arrays for the physical extended DF Fock");
    const size_t ao = atomic, mo = orbitals, occ = occupied, aux = auxiliaries;
    const auto mul = [](size_t x, size_t y) {
      return gpu::size_mul(x, y, "extended Fock byte count overflow");
    };
    const auto add = [](size_t x, size_t y) {
      return gpu::size_add(x, y, "extended Fock byte count overflow");
    };
    const size_t aa = mul(ao, ao), am = mul(ao, mo), mm = mul(mo, mo), gg = mul(aux, aa),
                 dd = mul(mul(aux, occ), mo);
    size_t doubles = add(add(aa, gg), add(am, add(dd, mul(2, mm))));
    size_t extended = add(aux, add(aa, am));
    const size_t required =
        add(add(mul(doubles, sizeof(double)), mul(extended, sizeof(Extended))), sizeof(int));
    if (required > budget)
      throw std::length_error("physical extended Fock exceeds the device byte budget");
    gpu::CudaDeviceScope scope(device);
    gpu::OwnedCudaStream stream(device);
    gpu::OwnedCudaBuffer<double> fast(device, doubles, stream.get());
    gpu::OwnedCudaBuffer<Extended> precise(device, extended, stream.get());
    gpu::OwnedCudaBuffer<int> bad(device, 1, stream.get());
    auto *dh = fast.get(), *dg = dh + aa, *dc = dg + gg, *dfit = dc + am, *dex = dfit + dd,
         *result = dex + mm;
    auto *tr = precise.get(), *a = tr + aux, *right = a + aa;
    const auto upload = [&](double* dst, const double* src, size_t n) {
      gpu::cuda_resource_check(
          cudaMemcpyAsync(dst, src, n * sizeof(double), cudaMemcpyHostToDevice, stream.get()));
    };
    upload(dh, h, aa);
    upload(dg, g, gg);
    upload(dc, coefficients, am);
    upload(dfit, fit, dd);
    upload(dex, exchange, mm);
    gpu::cuda_resource_check(cudaMemsetAsync(bad.get(), 0, sizeof(int), stream.get()));
    // clang-format off
    trace_density<<<blocks(aux),256,0,stream.get()>>>(dfit,occ,mo,aux,tr);
    ao_operator<<<blocks(aa),256,0,stream.get()>>>(dh,dg,tr,ao,aux,a,bad.get());
    contract_right<<<blocks(am),256,0,stream.get()>>>(a,dc,ao,mo,right);
    contract_left<<<blocks(mm),256,0,stream.get()>>>(right,dc,dex,ao,mo,result,bad.get());
    // clang-format on
    check_flags(stream, bad.get(), 1,
                "invalid Hermitian AO inputs or nonfinite physical extended Fock");
    std::vector<double> staged(mm);
    gpu::cuda_resource_check(cudaMemcpyAsync(staged.data(), result, mm * sizeof(double),
                                             cudaMemcpyDeviceToHost, stream.get()));
    stream.synchronize();
    std::copy(staged.begin(), staged.end(), out);
    *bytes = required;
    return 0;
  } catch (const std::exception& e) {
    generativeqc_tensor::error_text(error, error_size, e.what());
  } catch (...) {
    generativeqc_tensor::error_text(error, error_size, "unknown extended Fock error");
  }
  return 1;
}
