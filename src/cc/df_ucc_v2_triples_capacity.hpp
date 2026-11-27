#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>

namespace generativeqc::cc::v2_detail {

struct UccTriplesSlabs {
  std::size_t elements_per_slab;
  std::size_t cached_count;
  std::size_t streamed_count;
};

// All non-slab operands are already live when the channel chooses a layout.
template <class RealW>
inline bool ucc_triples_cache_slabs(std::initializer_list<UccTriplesSlabs> blocks,
                                    bool forces, std::size_t available_bytes) {
  const auto fits = [&](bool cache) {
    std::size_t remaining = available_bytes / (sizeof(RealW) + (forces ? sizeof(double) : 0));
    for (const auto& block : blocks) {
      const auto count = cache ? block.cached_count : block.streamed_count;
      if (count && block.elements_per_slab > remaining / count) return false;
      remaining -= count * block.elements_per_slab;
    }
    return true;
  };
  if (fits(true)) return true;
  if (fits(false)) return false;
  throw std::length_error("DF-UCCSD(T) v2 CUDA minimum triples slabs exceed remaining memory");
}

// A complete page owns two XY arrays and one Gram. Cross-page contractions also need
// the second pair of XY arrays and a second Gram. Compare by division so an oversized
// full page can fall back to a smaller page without overflowing its byte count.
inline std::size_t ucc_triples_page_rows(std::size_t active, std::size_t vector_elements,
                                         std::size_t available_bytes) {
  if (active < 2 || !vector_elements) return 0;
  const auto fits = [&](std::size_t rows, bool single_page) {
    const std::size_t xy_copies = single_page ? 2 : 4;
    const std::size_t gram_copies = single_page ? 1 : 2;
    const std::size_t elements = available_bytes / sizeof(double);
    if (rows > elements / vector_elements / xy_copies) return false;
    const auto after_xy = elements - rows * vector_elements * xy_copies;
    return rows <= after_xy / rows / gram_copies;
  };
  if (fits(active, true)) return active;
  if (!fits(1, false))
    throw std::length_error("DF-UCCSD(T) v2 CUDA minimum XY page exceeds remaining memory");
  std::size_t low = 1, high = active - 1;
  while (low < high) {
    const auto mid = low + (high - low + 1) / 2;
    if (fits(mid, false)) low = mid;
    else high = mid - 1;
  }
  return low;
}

}  // namespace generativeqc::cc::v2_detail
