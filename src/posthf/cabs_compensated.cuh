#pragma once

#include <cuda_runtime.h>

namespace generativeqc::posthf::cabs {
// Sum leading values and track the lost bits using error-free sum and FMA.
struct Extended {
  double hi = 0., lo = 0.;
  __device__ void add(double value) {
    const double sum = __dadd_rn(hi, value);
    const double v = __dsub_rn(sum, hi);
    const double error = __dadd_rn(__dsub_rn(hi, __dsub_rn(sum, v)), __dsub_rn(value, v));
    const double residual = __dadd_rn(lo, error);
    const double lead = __dadd_rn(sum, residual);
    const double v2 = __dsub_rn(lead, sum);
    lo = __dadd_rn(__dsub_rn(sum, __dsub_rn(lead, v2)), __dsub_rn(residual, v2));
    hi = lead;
  }
  __device__ void add_product(double x, double y) {
    const double product = __dmul_rn(x, y);
    add(product);
    add(__fma_rn(x, y, -product));
  }
  __device__ void add_product(Extended a, double x) {
    add_product(a.hi, x);
    add_product(a.lo, x);
  }
};

}  // namespace generativeqc::posthf::cabs
