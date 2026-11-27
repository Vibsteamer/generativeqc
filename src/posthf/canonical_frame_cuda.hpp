#pragma once

#include <cstddef>

#include "generativeqc/generativeqc.h"

/** Private real-FP64 canonical constraint component. M=C.T barC and orbital
 * energies belong to the same canonical reference. The result is a symmetric
 * full-Frobenius Fock seed with zero diagonal/ov blocks. A small-gap block must
 * already be stationary. Numeric budget counts owned device/staging storage and
 * borrowed input/output arrays; CUDA/context and container metadata are outside
 * it. Outputs are published only on success. This is not a physical Z solve. */
extern "C" GENERATIVEQC_API int generativeqc_canonical_frame_seed_cuda_v1(
    int device, int dimension, int occupied, double gap_threshold, double stationarity_threshold,
    std::size_t maximum_bytes, const double* energies, const double* rotation_gradient,
    double* fock_seed, std::size_t* device_bytes, char* error, std::size_t error_size);

/** Full OBS canonical Fock and overlap multipliers for an unrestricted orbital
 * cotangent. Reject an unresolved occupied/virtual frontier and a nonstationary
 * degenerate block. The returned norm counts each unresolved pair once. */
extern "C" GENERATIVEQC_API int generativeqc_canonical_full_frame_seeds_cuda_v1(
    int device, int dimension, int occupied, double gap_threshold, double stationarity_threshold,
    std::size_t maximum_bytes, const double* energies, const double* rotation_gradient,
    double* fock_seed, double* overlap_seed, double* unresolved_torque, std::size_t* device_bytes,
    char* error, std::size_t error_size);
