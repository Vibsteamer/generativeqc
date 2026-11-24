#pragma once

#include <cstddef>

namespace generativeqc::cc::triples::detail {

/** One occurrence count for a selected occupied orbital in a canonical tile.
 * For each fixed occupied orbital, visit all pairs `first >= second`; each
 * distinct unordered tile containing that orbital is then visited once.
 * The same overflowing-extent preflight used by visit_occupied_tiles applies.
 */
struct OccupiedTileShare {
  std::size_t index;
  std::size_t count;
};

#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr OccupiedTileShare occupied_tile_share(std::size_t orbital, std::size_t first,
                                                std::size_t second) {
  const std::size_t count = 1 + (orbital == first) + (orbital == second);
  const std::size_t high = orbital > first ? orbital : first;
  const std::size_t low = orbital < second ? orbital : second;
  const std::size_t mid = orbital >= first ? first : (orbital <= second ? second : orbital);
  return {high * (high + 1) * (high + 2) / 6 + mid * (mid + 1) / 2 + low, count};
}

/** Visit each triangular occupied tile once without materializing a tile list.
 * The serpentine schedule preserves the maximum occupied index while sharing
 * the lower-index row between adjacent tiles in the three-panel LRU. Return
 * each tile's original flat index so the final energy reduction stays ordered.
 * The caller must preflight occupied*(occupied+1)*(occupied+2) for overflow.
 * The ordinary schedule is retained for unqualified precision/storage domains.
 */
template <class Visitor>
void visit_occupied_tiles(std::size_t occupied, bool serpentine, Visitor&& visit) {
  std::size_t sequential_tile = 0;
  for (std::size_t first = 0; first < occupied; ++first) {
    if (serpentine) {
      const auto first_base = first * (first + 1) * (first + 2) / 6;
      for (std::size_t third_offset = 0; third_offset <= first; ++third_offset) {
        const auto third = first % 2 ? first - third_offset : third_offset;
        for (std::size_t second_offset = 0; second_offset <= first - third; ++second_offset) {
          const auto second = third % 2 ? first - second_offset : third + second_offset;
          visit(first, second, third, first_base + second * (second + 1) / 2 + third);
        }
      }
    } else {
      for (std::size_t second = 0; second <= first; ++second)
        for (std::size_t third = 0; third <= second; ++third)
          visit(first, second, third, sequential_tile++);
    }
  }
}

}  // namespace generativeqc::cc::triples::detail
