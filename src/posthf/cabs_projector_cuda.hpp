#pragma once

#include <cstddef>

#include "generativeqc/generativeqc.h"

/** Internal FP64 boundary. The row-major AO overlap is ordered [OBS, auxiliary].
 * The returned contravariant kernel is C_cabs C_cabs^T. An optional seed
 * requests its full-Frobenius adjoint with respect to the symmetric overlap.
 * expected_rank=-1 discovers the rank, otherwise it enforces the same branch.
 * The margin measures the closest residual eigenvalue to the cutoff in units
 * of its largest eigenvalue. All outputs are published together on success. */
extern "C" GENERATIVEQC_API int generativeqc_f12_cabs_projector_cuda_v1(
    int device, int dimension, int obs, double relative_cutoff, double branch_guard,
    int expected_rank, const double* overlap, const double* projector_bar, double* projector,
    double* overlap_bar, int* rank, double* margin, char* error, std::size_t error_size);
