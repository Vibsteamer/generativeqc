#pragma once

#include <cstddef>

#include "generativeqc/generativeqc.h"

// FP64 alpha then beta (vo, oo, vv, gap) for each nonempty spin channel, the electronic
// driver, one row per nuclear RHS, reference forces, and fixed-frame energy gradients.
// With zero directions and zero fixed terms, return only H-transpose-inverse times the driver.
// rho = ||H||_1 ||z||_1 / ||driver||_1 gates input sensitivity at DBL_EPSILON^-1/2.
// Extra-precise residual refinement separately checks solution error. Each correction
// must be at most half as large as its predecessor before the threshold test. Publishing z
// requires a final correction of at most 2^-40 |z|_inf and a posterior forward-error bound of
// 2^-39 |z|_inf, including for the initial LU solution.
extern "C" GENERATIVEQC_API int generativeqc_uhf_orbital_adjoint_cuda_v1(
    int device, int alpha, int beta, int orbitals, int auxiliary, int directions, int fixed_terms,
    const double* packed, std::size_t count, std::size_t maximum_bytes, double* rotations,
    double* orbital_gradient, double* forces, char* error, std::size_t error_size);
