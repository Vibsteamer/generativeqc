#pragma once

#include <cstddef>

#include "generativeqc/generativeqc.h"

/** Internal FP64 component boundary. The supplied Fock is in the orthonormal
 * occupied + (OBS virtual and CABS) frame, in full row-major storage. Spin
 * weight 2 represents a restricted spatial channel, 1 one unrestricted channel.
 * The optional full symmetric pullback satisfies dE = sum_ij bar_Fij dFij.
 * Projector, SCF and integral response belong to the caller. Outputs are only
 * published on success. The device is the scheduler-visible CUDA ordinal. */
extern "C" GENERATIVEQC_API int generativeqc_f12_cabs_singles_cuda_v1(
    int device, int dimension, int occupied, int spin_weight, double min_gap, const double* fock,
    double* energy, double* fock_bar, char* error, std::size_t error_size);
