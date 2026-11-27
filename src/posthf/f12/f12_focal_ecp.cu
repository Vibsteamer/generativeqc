// Generic off-center semilocal ECP. A radial tile holds the pointwise AO jets
// only until cuBLAS has projected them onto the sixteen real s/p/d/f harmonics.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "f12_focal_ecp_kernels.cuh"

namespace {
using namespace generativeqc::focal_ecp;
struct AllocationError {};
struct CudaError {};
void check(cudaError_t s) {
  if (s != cudaSuccess) throw CudaError{};
}
void check(cublasStatus_t s) {
  if (s != CUBLAS_STATUS_SUCCESS) throw CudaError{};
}
struct Arena {
  std::vector<void*> buffers;
  size_t used = 0, limit;
  cublasHandle_t blas = nullptr;
  explicit Arena(size_t budget) : limit(budget) {
    buffers.reserve(24);
    check(cublasCreate(&blas));
  }
  ~Arena() {
    for (auto* b : buffers) cudaFree(b);
    cublasDestroy(blas);
  }
  void rewind(size_t count, size_t consumed) {
    while (buffers.size() > count) {
      check(cudaFree(buffers.back()));
      buffers.pop_back();
    }
    used = consumed;
  }
  template <class T>
  T* allocate(size_t count, const T* src = nullptr) {
    if (count > (limit - used) / sizeof(T)) throw AllocationError{};
    T* p = nullptr;
    auto rc = cudaMalloc(&p, count * sizeof(T));
    if (rc == cudaErrorMemoryAllocation) throw AllocationError{};
    check(rc);
    buffers.push_back(p);
    used += count * sizeof(T);
    if (src) check(cudaMemcpy(p, src, count * sizeof(T), cudaMemcpyHostToDevice));
    return p;
  }
};
constexpr int threads = 128;
int blocks(size_t count) {
  return static_cast<int>(std::min<size_t>(65535, (count + threads - 1) / threads));
}
__global__ void point_jets(Basis b, int n, int active, int jets, int first, int nr, int nq,
                           const double* radii, const double* u, double* values) {
  size_t count = static_cast<size_t>(nr) * n * nq;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
    const int q = i % nq, a = i / nq % n, r = i / nq / n;
    double jet[4];
    point_jet(b, a, active, jets, radii[first + r], u + 3 * q, jet);
    size_t base = (static_cast<size_t>(r) * jets * n + a) * nq + q;
    for (int k = 0; k < jets; ++k) values[base + static_cast<size_t>(k) * n * nq] = jet[k];
  }
}
__global__ void channel_weights(int active, int nr, int nterm, const int* terms,
                                const double* parameters, const double* r, const double* dr,
                                double* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nr * 4; i += blockDim.x * gridDim.x) {
    const int k = i / 4;
    out[i] = channel_weight(active, i % 4, nterm, terms, parameters, r[k], dr[k]);
  }
}
__global__ void pack_projection(int n, int nr, int first, int jets, const double* proj,
                                const double* weights, double* panel, double* fitted) {
  size_t length = static_cast<size_t>(n) * nr * channels;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x) {
    const int a = i / (nr * channels), r = i / channels % nr, m = i % channels;
    const size_t src = (static_cast<size_t>(r) * jets * n + a) * channels + m;
    for (int d = 0; d < jets; ++d)
      panel[d * length + i] = proj[src + static_cast<size_t>(d) * n * channels];
    fitted[i] = proj[src] * weights[4 * (first + r) + (m < 1 ? 0 : m < 4 ? 1 : m < 9 ? 2 : 3)];
  }
}
__global__ void nuclear_scatter(int n, int active, int jets, const int* ao, const double* left,
                                double* out) {
  size_t nn = static_cast<size_t>(n) * n;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nn; i += blockDim.x * gridDim.x) {
    out[i] += left[i];
    if (jets == 1) continue;
    int a = i / n, b = i % n, at[2] = {ao[4 * a + 1], ao[4 * b + 1]};
    size_t ids[2] = {i, static_cast<size_t>(b) * n + a};
    for (int k = 0; k < 3; ++k)
      for (int j = 0; j < 2; ++j)
        if (at[j] != active) {
          const double v = left[(k + 1) * nn + ids[j]];
          out[(1 + 3 * at[j] + k) * nn + i] -= v;
          out[(1 + 3 * active + k) * nn + i] += v;
        }
  }
}
}  // namespace

extern "C" int gqc_f12_focal_ecp_projectors(int derivative, int na, int n, int nprim, int nt,
                                            int nr, int nq, const int* ao, const double* xyz,
                                            const double* alpha, const double* radial,
                                            const double* transforms, const int* terms,
                                            const double* parameters, const double* radii,
                                            const double* dr, const double* directions,
                                            const double* harmonics, double* output,
                                            size_t output_count, size_t budget, size_t* workspace) {
  if (!workspace || !output || !ao || !xyz || !alpha || !radial || !transforms || !terms ||
      !parameters || !radii || !dr || !directions || !harmonics || na < 1 ||
      na > (INT_MAX - 1) / 3 || n < 1 || n > INT_MAX / coefficients || nprim < 1 || nt < 1 ||
      nr < 16 || nr > 640 || nq < 1 || nq > 25088 || (derivative != 0 && derivative != 1))
    return 1;
  int jets = derivative ? 4 : 1;
  const size_t nn = static_cast<size_t>(n) * n;
  const size_t rows = 1ul + static_cast<size_t>(derivative) * 3 * na;
  if (nn > std::numeric_limits<size_t>::max() / sizeof(double) / rows) return 1;
  const size_t dims = nn * rows;
  if (output_count != dims) return 1;
  std::vector<int> centers;
  for (int i = 0; i < n; ++i)
    if (ao[4 * i] < 0 || ao[4 * i] > 6 || ao[4 * i + 1] < 0 || ao[4 * i + 1] >= na ||
        ao[4 * i + 2] < 0 || ao[4 * i + 2] >= ao[4 * i + 3] || ao[4 * i + 3] > nprim)
      return 1;
  for (int i = 0; i < nt; ++i) {
    int c = terms[3 * i];
    if (c < 0 || c >= na || terms[3 * i + 1] < 0 || terms[3 * i + 1] > 3 || terms[3 * i + 2] < 0 ||
        terms[3 * i + 2] > 4)
      return 1;
    if (std::find(centers.begin(), centers.end(), c) == centers.end()) centers.push_back(c);
  }
  try {
    Arena d(budget);
    // Explicit workspace also avoids the implementation's default cuBLAS pool.
    constexpr size_t blas_workspace = 4 * 1024 * 1024;
    auto scratch = d.allocate<unsigned char>(blas_workspace);
    check(cublasSetWorkspace(d.blas, scratch, blas_workspace));
    Basis basis = {d.allocate(4ul * n, ao), d.allocate(3ul * na, xyz), d.allocate(nprim, alpha),
                   d.allocate(nprim, radial),
                   d.allocate(static_cast<size_t>(coefficients) * n, transforms)};
    const auto it = d.allocate(3ul * nt, terms);
    const auto pp = d.allocate(2ul * nt, parameters);
    const auto rr = d.allocate(nr, radii), rd = d.allocate(nr, dr),
               u = d.allocate(3ul * nq, directions),
               y = d.allocate(static_cast<size_t>(channels) * nq, harmonics);
    auto weights = d.allocate<double>(4ul * nr), left = d.allocate<double>(jets * nn),
         out = d.allocate<double>(dims);
    check(cudaMemset(out, 0, dims * sizeof(double)));
    const size_t per =
        sizeof(double) * n * (static_cast<size_t>(jets) * nq + (2 * jets + 1) * channels);
    size_t free_bytes = 0, total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes));
    // The supplied numeric allowance can exceed device memory. Plan the tile
    // after allocating the persistent matrices, then retry if free memory changed
    // or the allocator cannot fit the largest individual buffer.
    int tile = static_cast<int>(std::min<size_t>(nr, std::min(budget - d.used, free_bytes) / per));
    const size_t persistent_buffers = d.buffers.size(), persistent_bytes = d.used;
    double *vals = nullptr, *proj = nullptr, *panel = nullptr, *weighted = nullptr;
    while (tile > 0) {
      try {
        const size_t layer = static_cast<size_t>(tile) * n;
        vals = d.allocate<double>(jets * layer * nq);
        proj = d.allocate<double>(jets * layer * channels);
        panel = d.allocate<double>(jets * layer * channels);
        weighted = d.allocate<double>(layer * channels);
        break;
      } catch (const AllocationError&) {
        // The persistent inputs and zeroed output survive a scratch retry.
        (void)cudaGetLastError();
        d.rewind(persistent_buffers, persistent_bytes);
        tile /= 2;
      }
    }
    if (tile == 0) return 2;
    double one = 1., zero = 0.;
    for (auto active : centers) {
      channel_weights<<<blocks(4ul * nr), threads>>>(active, nr, nt, it, pp, rr, rd, weights);
      check(cudaGetLastError());
      for (int first = 0; first < nr; first += tile) {
        int count = std::min(tile, nr - first), k = count * channels;
        point_jets<<<blocks(static_cast<size_t>(count) * n * nq), threads>>>(
            basis, n, active, jets, first, count, nq, rr, u, vals);
        check(cudaGetLastError());
        check(cublasDgemmStridedBatched(
            d.blas, CUBLAS_OP_N, CUBLAS_OP_N, channels, jets * n, nq, &one, y, channels, 0, vals,
            nq, static_cast<long long>(nq) * jets * n, &zero, proj, channels,
            static_cast<long long>(channels) * jets * n, count));
        pack_projection<<<blocks(static_cast<size_t>(count) * n * channels), threads>>>(
            n, count, first, jets, proj, weights, panel, weighted);
        check(cudaGetLastError());
        for (int j = 0; j < jets; ++j)
          check(cublasDgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, k, &one, weighted, k,
                            panel + static_cast<size_t>(j) * n * k, k, first ? &one : &zero,
                            left + j * nn, n));
      }
      nuclear_scatter<<<blocks(nn), threads>>>(n, active, jets, basis.ao, left, out);
      check(cudaGetLastError());
    }
    check(cudaMemcpy(output, out, dims * sizeof(double), cudaMemcpyDeviceToHost));
    *workspace = d.used;
    return 0;
  } catch (const AllocationError&) {
    return 2;
  } catch (...) {
    return 3;
  }
}
