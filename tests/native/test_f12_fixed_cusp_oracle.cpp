#include <cstdio>
#include <exception>

#include "f12_fixed_cusp_oracle.hpp"

int main() {
  try {
    f12_oracle::fixture(1, 1);
    f12_oracle::fixture(2, 3);
    f12_oracle::fixture(3, 2);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
