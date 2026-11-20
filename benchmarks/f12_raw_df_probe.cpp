// Emit unfitted Coulomb A[mu,nu,P], M[P,Q] and one streamed raw derivative per coordinate.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/cuda_density_fitting.hpp"

namespace {
using generativeqc::core::System;
using namespace generativeqc::scf;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void checked(generativeqc_status status, const std::string& detail) {
  if (status != GENERATIVEQC_STATUS_SUCCESS)
    throw std::runtime_error("DF status " + std::to_string(status) + ": " + detail);
}

void read_basis(std::istream& input, System& system, size_t shells, unsigned max_l) {
  for (size_t i = 0; i < shells; ++i) {
    generativeqc::core::Shell shell;
    size_t primitives = 0;
    input >> shell.atom_index >> shell.angular_momentum >> primitives;
    if (!input || primitives == 0 || primitives > 256)
      throw std::runtime_error("invalid contraction");
    for (size_t j = 0; j < primitives; ++j) {
      double exponent, coefficient;
      if (!(input >> exponent >> coefficient)) throw std::runtime_error("truncated contraction");
      shell.primitives.push_back({exponent, coefficient});
    }
    system.shells.push_back(std::move(shell));
  }
  std::string detail;
  checked(generativeqc::molecule::validate_and_normalize(system, detail, max_l), detail);
}

void write(std::ofstream& file, const std::vector<double>& data, size_t n) {
  if (!std::all_of(data.begin(), data.begin() + n, [](double v) { return std::isfinite(v); }))
    throw std::runtime_error("nonfinite raw source");
  if (!file.write(reinterpret_cast<const char*>(data.data()), n * sizeof(double)))
    throw std::runtime_error("could not save raw source");
}

void run(const char* input_file, const std::filesystem::path& out) {
  std::ifstream input(input_file);
  size_t atoms = 0, orbital_shells = 0, auxiliary_shells = 0;
  System orbital;
  input >> atoms >> orbital_shells >> auxiliary_shells >> orbital.charge >> orbital.multiplicity;
  if (!input || !atoms || atoms > 256 || !orbital_shells || !auxiliary_shells ||
      orbital_shells > 4096 || auxiliary_shells > 4096)
    throw std::runtime_error("invalid shell geometry");
  for (size_t a = 0; a < atoms; ++a) {
    generativeqc::core::Atom atom;
    input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    orbital.atoms.push_back(atom);
  }
  orbital.basis_representation = GENERATIVEQC_BASIS_SPHERICAL;
  auto auxiliary = orbital;
  read_basis(input, orbital, orbital_shells, 3);
  read_basis(input, auxiliary, auxiliary_shells, 5);
  std::string detail;
  char extra;
  if (input >> extra) throw std::runtime_error("unexpected trailing shell data");
  int devices = 0;
  check(cudaGetDeviceCount(&devices));
  if (!devices) throw std::runtime_error("missing CUDA device");
  check(cudaSetDevice(0));
  CudaDensityFittingIntegralSource* raw = nullptr;
  std::vector<double> metric;
  size_t n = 0, q = 0;
  checked(create_cuda_density_fitting_integral_source(0, {orbital}, {auxiliary}, &raw, metric, n, q,
                                                      detail),
          detail);
  std::unique_ptr<CudaDensityFittingIntegralSource,
                  decltype(&destroy_cuda_density_fitting_integral_source)>
      source(raw, destroy_cuda_density_fitting_integral_source);
  if (!n || !q || n > 4096 || q > 8192 || metric.size() != q * q)
    throw std::runtime_error("invalid DF dimensions");
  constexpr size_t pairs_per_tile = 128;
  std::vector<double> tile(std::max(pairs_per_tile * q, q * q));
  double* buffer = nullptr;
  check(cudaMalloc(reinterpret_cast<void**>(&buffer), tile.size() * sizeof(double)));
  std::unique_ptr<double, decltype(&cudaFree)> device(buffer, cudaFree);
  cudaStream_t stream = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  try {
    std::ofstream a(out / "A.bin", std::ios::binary), m(out / "M.bin", std::ios::binary),
        da(out / "dA.bin", std::ios::binary), dm(out / "dM.bin", std::ios::binary);
    write(m, metric, metric.size());
    size_t a_tiles = 0, m_tiles = 0;
    auto copy = [&](std::ofstream& file, size_t count) {
      check(cudaMemcpyAsync(tile.data(), buffer, count * sizeof(double), cudaMemcpyDeviceToHost,
                            stream));
      check(cudaStreamSynchronize(stream));
      write(file, tile, count);
    };
    for (int k = -1; k < static_cast<int>(3 * atoms); ++k) {
      for (size_t i = 0; i < n * n; i += pairs_per_tile) {
        size_t count = std::min(pairs_per_tile, n * n - i);
        checked(generate_cuda_density_fitting_raw_tile(raw, 0, i, count, 0, q, k, stream, buffer,
                                                       detail),
                detail);
        copy(k < 0 ? a : da, count * q);
        ++a_tiles;
      }
      if (k >= 0) {
        checked(generate_cuda_density_fitting_metric_derivative_tile(raw, 0, 0, q, k, stream,
                                                                     buffer, detail),
                detail);
        copy(dm, q * q);
        ++m_tiles;
      }
    }
    std::cout << "{\"nao\":" << n << ",\"naux\":" << q << ",\"coordinates\":" << 3 * atoms
              << ",\"raw_three_center_tiles\":" << a_tiles
              << ",\"metric_derivative_tiles\":" << m_tiles
              << ",\"tile_device_bytes\":" << tile.size() * sizeof(double) << "}\n";
  } catch (...) {
    cudaStreamDestroy(stream);
    throw;
  }
  check(cudaStreamDestroy(stream));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("expected shell input and output directory");
    run(argv[1], argv[2]);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
