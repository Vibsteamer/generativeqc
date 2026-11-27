#pragma once

namespace generativeqc::molecule {

struct SmallCoreHalideComposition {
  unsigned hydrogen = 0, carbon = 0, oxygen = 0, bromine = 0, iodine = 0, other = 0;
  unsigned other_ecp = 0;
  unsigned unsupported_other = 0;
  long long neutral_valence_electrons = 0;

  void add(int atomic_number, int core) {
    if (atomic_number > core) neutral_valence_electrons += (atomic_number - core);
    if (atomic_number == 1 && core == 0) ++hydrogen;
    else if (atomic_number == 6 && core == 0) ++carbon;
    else if (atomic_number == 8 && core == 0) ++oxygen;
    else if (atomic_number == 35 && core == 10) ++bromine;
    else if (atomic_number == 53 && core == 28) ++iodine;
    else {
      ++other;
      if (core != 0) ++other_ecp;
      // Only all-electron light main-group partner atoms (H..Ar, 1 <= Z <= 18)
      // belong to the validated small-core Br/I molecular domain.
      if (core != 0 || atomic_number < 1 || atomic_number > 18) ++unsupported_other;
    }
  }

  bool hydride() const { return carbon == 0 && oxygen == 0 && bromine + iodine == 1 && other == 0; }
  bool molecular_halide() const {
    const bool ligand = (carbon > 0 && oxygen == 0) ||
                        (carbon == 0 && oxygen == 1 && hydrogen == 2);
    return ligand && bromine + iodine == 1 && other == 0;
  }

  bool molecular_halide_reference(int charge, int multiplicity, long long electrons) const {
    if (!molecular_halide() || (charge != 0 && charge != 1) ||
        (multiplicity != 1 && multiplicity != 2)) return false;
    const auto expected = 6LL * carbon + 8LL * oxygen + hydrogen + 25 - charge;
    return electrons == expected && electrons % 2 == multiplicity - 1;
  }

  // General validated small-core Br(ECP10MDF)/I(ECP28MDF) molecules with light
  // all-electron main-group partner atoms (e.g. C4H6Br2O, Br2, I2, CH2BrCl).
  bool validated_small_core_halide_molecule() const {
    return bromine + iodine >= 1 && other_ecp == 0 && unsupported_other == 0 && !hydride();
  }

  // Zero means the fitted electronic state is outside the scalar ECP domain.
  unsigned fitted_orbital_angular_limit(int charge, int multiplicity, long long electrons) const {
    if (molecular_halide_reference(charge, multiplicity, electrons)) return 4;
    if (hydride()) {
      if (charge == 0) {
        if (multiplicity == 1 && hydrogen == 1 && electrons == 26) return 4;
        if (multiplicity == 2 && (hydrogen == 0 || hydrogen == 2) &&
            electrons == 25LL + hydrogen)
          return 4;
        if (multiplicity == 2 && electrons == 25LL + hydrogen && electrons % 2 == 1) return 3;
      } else if (charge == 1) {
        if (multiplicity == 2 && hydrogen == 1 && electrons == 25) return 4;
        if (multiplicity == 1 && (hydrogen == 0 || hydrogen == 2) &&
            electrons == 24LL + hydrogen)
          return 4;
      } else if (charge == -1) {
        if (multiplicity == 1 && hydrogen == 0 && electrons == 26) return 4;
      }
      return 0;
    }
    if (validated_small_core_halide_molecule() &&
        !molecular_halide() &&
        (charge >= -1 && charge <= 1) &&
        (multiplicity == 1 || multiplicity == 2) &&
        electrons > 0 &&
        electrons == neutral_valence_electrons - charge &&
        electrons % 2 == multiplicity - 1) {
      return 4;
    }
    return 0;
  }
};

}  // namespace generativeqc::molecule
