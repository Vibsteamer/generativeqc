#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cc/f12_energy_gradient.hpp"
#include "cc/f12_fixed_cusp.hpp"

namespace f12_oracle {

using generativeqc::cc::FixedCuspF12;
namespace cpu = generativeqc::cc::f12;
namespace gpu = generativeqc::cc::f12_cuda;

inline void near(double actual, double reference, double atol, const char* field) {
  if (!std::isfinite(actual) || std::abs(actual - reference) > atol)
    throw std::runtime_error(std::string(field) + " disagrees with CPU oracle");
}

inline std::vector<double> data(std::size_t size, int stride) {
  std::vector<double> values(size);
  for (std::size_t i = 0; i < size; ++i)
    values[i] = (static_cast<int>((stride * i + 3) % 31) - 15) / 64.;
  return values;
}

struct Fixture {
  FixedCuspF12 raw, canonical;
  std::vector<double> r1, t1, base1, base2, expected1, expected2;
};

inline Fixture fixture(std::size_t o, std::size_t v) {
  const auto n1 = o * v, n2 = n1 * n1;
  FixedCuspF12 raw;
  raw.fixed_geminal_energy = 0.125;
  raw.w_vo = data(o * o * v * o, 7);
  raw.w_vv = cpu::symmetrized_pair_copy(o, v, data(n2, 5));
  raw.c_f12 = cpu::symmetrized_pair_copy(o, v, data(n2, 11));
  // A small pair mismatch is admissible at input and must be normalized on the device.
  if (v > 1) raw.w_vv[1] += 0x1p-37;
  if (o > 1) raw.c_f12[o * v * v] -= 0x1p-38;
  cpu::validate(o, v, raw);
  auto canonical = raw;
  canonical.w_vv = cpu::symmetrized_pair_copy(o, v, raw.w_vv);
  canonical.c_f12 = cpu::symmetrized_pair_copy(o, v, raw.c_f12);
  const auto r1 = cpu::singles_constant(o, v, canonical.w_vo);
  auto t1 = data(n1, 17), t2 = data(n2, 19);
  const auto base1 = data(n1, 9), base2 = data(n2, 13);
  std::vector<double> expected1(n1), expected2(n2);

  const auto energy = [&] { return cpu::energy_cpu(o, v, canonical, r1, t1.data(), t2.data()); };
  // The forward energy is quadratic in T1 and linear in T2. Central differences
  // give the derivative with roundoff only, including when all four indices differ.
  const auto differentiate = [&](double& amplitude) {
    constexpr double h = 0x1p-10;
    const auto value = amplitude;
    amplitude = value + h;
    const auto plus = energy();
    amplitude = value - h;
    const auto minus = energy();
    amplitude = value;
    return (plus - minus) / (2 * h);
  };
  for (std::size_t k = 0; k < n1; ++k) {
    expected1[k] = base1[k] - differentiate(t1[k]);
    near(base1[k] - gpu::energy_gradient_one(o, v, r1.data(), canonical.w_vv.data(), t1.data(), k),
         expected1[k], 2e-12, "singles RHS");
  }
  for (std::size_t k = 0; k < n2; ++k) {
    expected2[k] = base2[k] - differentiate(t2[k]);
    near(base2[k] - gpu::energy_gradient_two(v, canonical.w_vv.data(), canonical.c_f12.data(), k),
         expected2[k], 2e-12, "doubles RHS");
  }
  return {std::move(raw), std::move(canonical), r1, std::move(t1), base1, base2,
          std::move(expected1), std::move(expected2)};
}

}  // namespace f12_oracle
