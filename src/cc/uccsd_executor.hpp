#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace generativeqc::cc {
using UnrestrictedFeeds = std::map<std::string, std::vector<double>>;
using UnrestrictedFeedViews = std::map<std::string, std::span<const double>>;

inline UnrestrictedFeedViews borrow_unrestricted_feeds(const UnrestrictedFeeds& values) {
  UnrestrictedFeedViews result;
  for (const auto& [name, payload] : values) result.emplace(name, payload);
  return result;
}

/** Synchronous native CUDA graph boundary. Host amplitudes, packed DIIS and
 * response vectors are explicit host consumers; all graph arithmetic runs on
 * the requested device. One fixed admitted arena/stream is reused until close.
 * Publication and exceptional cleanup both drain outstanding transfers. */
class UnrestrictedCudaExecutor {
 public:
  virtual ~UnrestrictedCudaExecutor() = default;
  virtual UnrestrictedFeeds evaluate(const std::string& program,
                                     const UnrestrictedFeedViews& inputs) = 0;
  virtual std::size_t numeric_capacity_bytes() const noexcept = 0;
};

#if GENERATIVEQC_HAS_CUDA
std::unique_ptr<UnrestrictedCudaExecutor> make_unrestricted_cuda_executor(
    std::size_t nbf, std::size_t alpha, std::size_t beta, int device, std::size_t budget,
    bool response, bool mp2 = false, bool separate_reference = false);
#endif
}  // namespace generativeqc::cc
