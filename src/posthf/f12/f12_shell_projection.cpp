// Map the molecular sink's axially normalized Cartesian polynomials to native spherical AOs.
#include <algorithm>
#include <cstddef>

#include "molecule/basis.hpp"

extern "C" int gqc_f12_raw_spherical_projection(unsigned l, double* matrix, std::size_t size) {
  namespace gm = generativeqc::molecule;
  if (l > 6 || !matrix || size != (2 * l + 1) * gm::cartesian_count(l)) return 1;
  try {
    const auto rows = gm::complementary_spherical_expansions(l);
    const double axial = gm::cartesian_component_normalization({l, 0, 0});
    std::fill_n(matrix, size, 0.);
    for (std::size_t row = 0; row < rows.size(); ++row) {
      for (const auto& term : rows[row]) {
        const auto yz = term.component[1] + term.component[2];
        const auto column = yz * (yz + 1) / 2 + term.component[2];
        matrix[row * gm::cartesian_count(l) + column] =
            term.coefficient * gm::cartesian_component_normalization(term.component) / axial;
      }
    }
    return 0;
  } catch (...) {
    return 2;
  }
}
