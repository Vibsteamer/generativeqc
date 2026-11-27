"""Exercise the native metadata upload contract on a deferred CUDA copy."""

import subprocess
from pathlib import Path

from conftest import NativeCxx


CUDA_API = r"""
#pragma once
#include <cstddef>

enum cudaError_t {
  cudaSuccess, cudaErrorInvalidValue, cudaErrorMemoryAllocation,
  cudaErrorInvalidDevice, cudaErrorUnknown
};
enum cudaMemcpyKind { cudaMemcpyHostToDevice };
using cudaStream_t = void*;

cudaError_t cudaGetDevice(int* device);
cudaError_t cudaMalloc(void** device, std::size_t bytes);
cudaError_t cudaFree(void* device);
cudaError_t cudaMallocAsync(void** device, std::size_t bytes, cudaStream_t stream);
cudaError_t cudaFreeAsync(void* device, cudaStream_t stream);
cudaError_t cudaMemcpy(void* device, const void* host, std::size_t bytes, cudaMemcpyKind kind);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
"""

PROBE = r"""
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "scf/cuda/metadata_upload.hpp"

namespace {
struct PendingCopy {
  void* device;
  std::vector<unsigned char> staging;
};
std::vector<PendingCopy> pending;
cudaError_t copy_error;
cudaError_t fence_error;

struct Source {
  std::size_t device_bytes{};
  std::vector<void*> allocations;
  ~Source() {
    for (void* device : allocations) (void)generativeqc::runtime::resource_cuda_free(device);
  }
};

void require(bool condition, const char* detail) {
  if (!condition) throw std::runtime_error(detail);
}

void check_upload(cudaError_t copy, cudaError_t fence, generativeqc_status expected) {
  copy_error = copy;
  fence_error = fence;
  Source source;
  constexpr unsigned char payload[] = {0x41, 0x2a, 0x77};
  void* device{};
  std::string detail;
  const auto status = generativeqc::scf::cuda_execution::source_upload(
      source, payload, sizeof(payload), &device, detail);
  require(status == expected, "metadata upload returned the wrong CUDA status");
  require(device && source.allocations == std::vector<void*>{device} &&
              source.device_bytes == sizeof(payload),
          "metadata buffer was not registered with its source for cleanup");
  if (expected == GENERATIVEQC_STATUS_SUCCESS) {
    require(std::memcmp(device, payload, sizeof(payload)) == 0,
            "metadata was still unavailable to another stream when upload returned");
  } else {
    require(!detail.empty(), "metadata upload failure lacked a diagnostic");
  }
}
}  // namespace

cudaError_t cudaGetDevice(int* device) {
  *device = 0;
  return cudaSuccess;
}

cudaError_t cudaMalloc(void** device, std::size_t bytes) {
  *device = std::calloc(1, bytes);
  return *device ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaFree(void* device) {
  std::erase_if(pending, [device](const auto& copy) { return copy.device == device; });
  std::free(device);
  return cudaSuccess;
}

cudaError_t cudaMemcpy(void* device, const void* host, std::size_t bytes, cudaMemcpyKind kind) {
  if (kind != cudaMemcpyHostToDevice) return cudaErrorInvalidValue;
  if (copy_error != cudaSuccess) return copy_error;
  // Pageable host input is staged before return, while the device write stays queued.
  const auto* begin = static_cast<const unsigned char*>(host);
  pending.push_back({device, {begin, begin + bytes}});
  return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  if (stream) return cudaErrorInvalidValue;
  if (fence_error == cudaSuccess) {
    for (const auto& copy : pending)
      std::memcpy(copy.device, copy.staging.data(), copy.staging.size());
  }
  pending.clear();
  return fence_error;
}

int main() {
  try {
    check_upload(cudaSuccess, cudaSuccess, GENERATIVEQC_STATUS_SUCCESS);
    check_upload(cudaSuccess, cudaErrorUnknown, GENERATIVEQC_STATUS_CUDA_ERROR);
    check_upload(cudaErrorInvalidValue, cudaErrorMemoryAllocation,
                 GENERATIVEQC_STATUS_CUDA_ERROR);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
"""


def test_metadata_is_ready_for_other_streams_at_return(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "cuda_runtime_api.h").write_text(CUDA_API, encoding="utf-8")
    source = tmp_path / "probe.cpp"
    source.write_text(PROBE, encoding="utf-8")
    executable = native_cxx.build_executable(
        [source],
        tmp_path / "probe",
        compile_args=(
            "-std=c++20",
            f"-I{tmp_path}",
            f"-I{root / 'src'}",
            f"-I{root / 'include'}",
        ),
        link_args=("-pthread",),
    )
    result = subprocess.run([executable], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
