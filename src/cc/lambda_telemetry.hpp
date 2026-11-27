#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace generativeqc::cc::detail {

/** An owned snapshot of the v1 C ABI fields. A disengaged slot has no result. */
struct LambdaTelemetry {
  std::array<double, 10> scalars;
  std::array<std::uint64_t, 16> counts;
  std::string handoff_reason;
};

void capture_last_lambda_telemetry(std::optional<LambdaTelemetry>& destination) noexcept;
int copy_lambda_telemetry(const std::optional<LambdaTelemetry>& slot, double* scalars,
                          std::size_t scalar_count, std::uint64_t* counts, std::size_t count_len,
                          char* reason, std::size_t reason_len) noexcept;

}  // namespace generativeqc::cc::detail
