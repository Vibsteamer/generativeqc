#pragma once

namespace generativeqc::molecule {

struct SmallCoreHalideComposition {
  unsigned hydrogen = 0, carbon = 0, bromine = 0, iodine = 0, other = 0;

  void add(int atomic_number, int core) {
    if (atomic_number == 1 && core == 0) ++hydrogen;
    else if (atomic_number == 6 && core == 0) ++carbon;
    else if (atomic_number == 35 && core == 10) ++bromine;
    else if (atomic_number == 53 && core == 28) ++iodine;
    else ++other;
  }

  bool hydride() const { return carbon == 0 && bromine + iodine == 1 && other == 0; }
  bool carbon_halide() const { return carbon > 0 && bromine + iodine == 1 && other == 0; }

  bool carbon_halide_reference(int charge, int multiplicity, long long electrons) const {
    if (!carbon_halide() || (charge != 0 && charge != 1) ||
        (multiplicity != 1 && multiplicity != 2)) return false;
    const auto expected = 6LL * carbon + hydrogen + 25 - charge;
    return electrons == expected && electrons % 2 == multiplicity - 1;
  }

  // Zero means the fitted electronic state is outside the scalar ECP domain.
  unsigned fitted_orbital_angular_limit(int charge, int multiplicity, long long electrons) const {
    if (carbon_halide_reference(charge, multiplicity, electrons)) return 4;
    if (!hydride() || charge != 0) return 0;
    if (multiplicity == 1 && hydrogen == 1 && electrons == 26) return 4;
    if (multiplicity == 2) return hydrogen == 2 ? 4 : 3;
    return 0;
  }
};

}  // namespace generativeqc::molecule
