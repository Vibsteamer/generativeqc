#include <cmath>

#include "scf/cuda/one_electron_export_kernels.hpp"
#include "scf/cuda/one_electron_native_contraction.cuh"

namespace generativeqc::scf::cuda_execution {

/** Evaluate first-coordinate one-electron response using the retained Dual path. */
__global__ void build_cuda_one_electron_derivatives_kernel(
    DeviceBatch batch, const std::int32_t* pair_first, const std::int32_t* pair_second,
    std::size_t pair_count, std::int64_t derivative_coordinate, double* overlap, double* hcore) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch.batch_size) * pair_count) {
    return;
  }
  const std::int32_t system = static_cast<std::int32_t>(element / pair_count);
  const std::size_t pair = element % pair_count;
  const std::size_t row = static_cast<std::size_t>(pair_first[pair]);
  const std::size_t column = static_cast<std::size_t>(pair_second[pair]);
  const std::int64_t system_derivative_coordinate =
      derivative_coordinate < 0 ? derivative_coordinate
                                : derivative_coordinate + batch.atom_offsets[system] * 3;
  const auto base = static_cast<std::size_t>(system) * n;
  const auto li = batch.shell_angular[batch.ao_shells[base + row]];
  const auto lj = batch.shell_angular[batch.ao_shells[base + column]];
  double ds, dh;
  if (li == 4 || lj == 4) {
    contracted_g_export_derivatives(batch, system, row, column, system_derivative_coordinate, ds,
                                    dh);
  } else {
    ds = contracted_overlap<Dual>(batch, system, row, column, system_derivative_coordinate)
             .derivative;
    dh =
        contracted_hcore<Dual>(batch, system, row, column, system_derivative_coordinate).derivative;
  }
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * n * n;
  overlap[matrix_offset + row * n + column] = ds;
  hcore[matrix_offset + row * n + column] = dh;
  if (row != column) {
    overlap[matrix_offset + column * n + row] = ds;
    hcore[matrix_offset + column * n + row] = dh;
  }
}

void launch_build_cuda_one_electron_derivatives_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const std::int32_t* pair_first, const std::int32_t* pair_second, std::size_t pair_count,
    std::int64_t derivative_coordinate, double* overlap, double* hcore) {
  build_cuda_one_electron_derivatives_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, pair_first, pair_second, pair_count, derivative_coordinate, overlap, hcore);
}

}  // namespace generativeqc::scf::cuda_execution
