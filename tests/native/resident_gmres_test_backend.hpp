#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "response/resident_krylov.hpp"

using namespace generativeqc::response;

struct Backend : ResidentResponseBackend {
  explicit Backend(std::size_t count, const ResponsePlan& plan = {})
      : ResidentResponseBackend(plan), data(count) {}
  std::vector<std::array<double, 2>> data;
  std::size_t uploads{}, downloads{}, actions{}, norms{}, fail_norm{};
  bool nan_operator{};
  bool diagonal_operator{}, upper_operator{}, right_preconditioned{}, nan_preconditioner{};

  std::size_t dimension() const noexcept override { return 2; }
  std::size_t vector_slots() const noexcept override { return data.size(); }
  std::size_t owned_resident_bytes() const noexcept override {
    return data.size() * sizeof(data[0]);
  }
  void upload(std::size_t slot, std::span<const double> values) override {
    ++uploads;
    std::copy(values.begin(), values.end(), data.at(slot).begin());
  }
  void download(std::size_t slot, std::span<double> values) override {
    ++downloads;
    std::copy(data.at(slot).begin(), data.at(slot).end(), values.begin());
  }
  void zero(std::size_t slot) override { data.at(slot).fill(0.0); }
  void copy(std::size_t dst, std::size_t src) override { data.at(dst) = data.at(src); }
  void scale(std::size_t slot, double alpha) override {
    for (double& value : data.at(slot)) value *= alpha;
  }
  void axpy(std::size_t dst, double alpha, std::size_t src) override {
    for (std::size_t i = 0; i < 2; ++i) data.at(dst)[i] += alpha * data.at(src)[i];
  }
  double dot(std::size_t left, std::size_t right) override {
    return std::inner_product(data.at(left).begin(), data.at(left).end(),
                              data.at(right).begin(), 0.0);
  }
  double norm(std::size_t slot) override {
    if (++norms == fail_norm) throw std::overflow_error("injected norm overflow");
    return stable_norm(data.at(slot));
  }
  void apply(std::size_t dst, std::size_t src) override {
    ++actions;
    copy(dst, src);
    if (diagonal_operator) {
      data.at(dst)[0] *= 2.0;
      data.at(dst)[1] *= 4.0;
      if (upper_operator) data.at(dst)[0] += data.at(src)[1];
    }
    if (nan_operator) data.at(dst)[0] = std::numeric_limits<double>::quiet_NaN();
  }
  bool has_right_preconditioner() const noexcept override { return right_preconditioned; }
  void precondition(std::size_t dst, std::size_t src) override {
    data.at(dst)[0] = data.at(src)[0] / 2.0;
    data.at(dst)[1] = data.at(src)[1] / 4.0;
    if (nan_preconditioner) data.at(dst)[0] = std::numeric_limits<double>::quiet_NaN();
  }
};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
