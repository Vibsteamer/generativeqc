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
  bool methyl_halide() const {
    return carbon == 1 && hydrogen == 3 && bromine + iodine == 1 && other == 0;
  }
};

}  // namespace generativeqc::molecule
