#pragma once

#include <cstddef>

#include "generativeqc/generativeqc.h"

// FP64 alpha then beta (vo, oo, vv, gap) for each nonempty spin channel, the electronic
// driver, one row per nuclear RHS, reference forces, and fixed-frame energy gradients.
extern "C" GENERATIVEQC_API int generativeqc_uhf_orbital_adjoint_cuda_v1(
    int device, int alpha, int beta, int orbitals, int auxiliary, int directions, int fixed_terms,
    const double* packed, std::size_t count, std::size_t maximum_bytes, double* rotations,
    double* orbital_gradient, double* forces, char* error, std::size_t error_size);
