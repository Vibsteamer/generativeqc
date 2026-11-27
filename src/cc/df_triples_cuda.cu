#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

#include "cc/df_triples.hpp"
#include "cc/df_triples_compensated_w.cuh"
#include "cc/df_triples_traversal.hpp"
#include "cc/phase_observation.hpp"
#include "generated_df_occupied_triples_cuda.cuh"
#include "posthf/capacity.hpp"
#include "runtime/resource_ledger.hpp"
#include "tensor/cuda_runtime.cuh"

#if defined(GENERATIVEQC_TEST_HOOKS)
namespace {
thread_local bool reject_w_library_for_test = false;
}
extern "C" void df_triples_reject_library_for_test_v1(bool reject) {
  reject_w_library_for_test = reject;
}
extern "C" bool tensor_cutensor_qualification_for_test_v1(bool enable, int reject_after) {
#if GENERATIVEQC_HAS_CUTENSOR
  // Synthetic resource limits belong to endpoint qualification, never to the
  // production profile registry or the method's scientific admission API.
  generativeqc::tensor::cutensor_reservation_for_test =
      enable ? generativeqc::tensor::ContractionProviderReservation{64ULL << 20, 256ULL << 20,
                                                                    64ULL << 20}
             : generativeqc::tensor::ContractionProviderReservation{};
  generativeqc::tensor::cutensor_preparations_before_rejection_for_test = reject_after;
  return true;
#else
  (void)enable;
  (void)reject_after;
  return false;
#endif
}
#endif

namespace generativeqc::cc::triples {
namespace {
using posthf::checked_add;
using posthf::checked_mul;
using Clock = std::chrono::steady_clock;
constexpr std::size_t provider_allowance = 96ULL << 20;
constexpr std::size_t blas_workspace = 4ULL << 20;
std::size_t bytes(std::size_t n) { return checked_mul(n, sizeof(double)); }
std::size_t align256(std::size_t n) { return n % 256 ? checked_add(n, 256 - n % 256) : n; }
std::size_t reserve(std::size_t& cursor, std::size_t n) {
  cursor = align256(cursor);
  const auto offset = cursor;
  cursor = checked_add(cursor, n);
  return offset;
}

struct Layout {
  std::array<std::size_t, 9> sizes{}, inputs{};
  std::size_t panels{}, moments{}, partials{}, energies{}, energy{}, occupied_energies{}, error{},
      library{};
  std::size_t execution_storage{}, paired_pointers{}, paired_pointer_bytes{};
  std::size_t arena{}, total{}, panel_capacity{}, v3{}, tiles{};
  unsigned blocks{};
};

Layout layout(std::size_t o, std::size_t v, std::size_t q, std::size_t panels,
              std::size_t execution_bytes = 0, std::size_t host_binding_bytes = 0,
              std::size_t execution_provider_bytes = provider_allowance,
              std::size_t library_bytes = blas_workspace, bool external_inputs = false,
              std::size_t paired_pointer_bytes = 0, bool occupied_partition = false) {
  const auto oo = checked_mul(o, o), vv = checked_mul(v, v), ov = checked_mul(o, v);
  const auto ovv = checked_mul(o, vv);
  // Check every dimension and physical leading dimension before reading inputs
  // or allocating a provider. All subsequent pointer products fit these sizes.
  if (std::max({o, v, q, oo, vv, ov, ovv}) >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("DF triples exceed BLAS indexing");
  Layout p;
  p.sizes = {checked_mul(q, ov),
             checked_mul(q, vv),
             checked_mul(ov, oo),
             checked_mul(ov, ov),
             ov,
             ov,
             checked_mul(oo, vv),
             o,
             v};
  p.v3 = checked_mul(v, vv);
  p.tiles = checked_mul(checked_mul(o, checked_add(o, 1)), checked_add(o, 2)) / 6;
  p.blocks = static_cast<unsigned>(std::min<std::size_t>(1 + (p.v3 - 1) / 256, 65535));
  p.panel_capacity = std::min(o, panels);
  // Reject unrepresentable complete work ledgers before any allocation or launch.
  (void)checked_mul(p.tiles, p.v3);
  const auto moment_terms =
      checked_mul(checked_mul(6, p.tiles), checked_add(checked_mul(p.v3, v), checked_mul(p.v3, o)));
  (void)checked_add(moment_terms, checked_mul(checked_mul(3, p.tiles), checked_mul(q, p.v3)));
  std::size_t cursor = 0;
  if (!external_inputs)
    for (std::size_t x = 0; x < p.sizes.size(); ++x)
      p.inputs[x] = reserve(cursor, bytes(p.sizes[x]));
  p.panels = reserve(cursor, bytes(checked_mul(p.panel_capacity, p.v3)));
  p.moments = reserve(cursor, bytes(checked_mul(6, p.v3)));
  p.execution_storage = reserve(cursor, execution_bytes);
  p.paired_pointer_bytes = paired_pointer_bytes;
  p.paired_pointers = reserve(cursor, paired_pointer_bytes);
  p.partials = reserve(cursor, bytes(p.blocks));
  p.energies = reserve(cursor, bytes(p.tiles));
  p.energy = reserve(cursor, sizeof(double));
  if (occupied_partition) p.occupied_energies = reserve(cursor, bytes(o));
  p.error = reserve(cursor, sizeof(int));
  p.library = reserve(cursor, library_bytes);
  p.arena = align256(cursor);
  p.total = checked_add(checked_add(p.arena, execution_provider_bytes), host_binding_bytes);
  return p;
}

double validate_inputs(std::size_t o, std::size_t v, std::size_t q, const Layout& p,
                       const std::array<const double*, 9>& host, double threshold) {
  const auto* bvv = host[1];
  const auto* eps_o = host[7];
  const auto* eps_v = host[8];
  for (std::size_t x = 0; x < host.size(); ++x) {
    if (!host[x]) throw std::invalid_argument("null DF triples input");
    if (!std::all_of(host[x], host[x] + p.sizes[x],
                     [](double value) { return std::isfinite(value); }))
      throw std::invalid_argument("nonfinite DF triples input");
  }
  for (std::size_t Q = 0; Q < q; ++Q)
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t b = 0; b < a; ++b)
        if (std::abs(bvv[(Q * v + a) * v + b] - bvv[(Q * v + b) * v + a]) > 1e-10)
          throw std::invalid_argument("DF triples require symmetric B_vv pairs");
  const auto max_occ = *std::max_element(eps_o, eps_o + o);
  const auto min_vir = *std::min_element(eps_v, eps_v + v);
  const double minimum = 3.0 * (min_vir - max_occ);
  if (!(max_occ < min_vir) || !std::isfinite(minimum) || minimum <= threshold)
    throw std::invalid_argument("noncanonical or near-zero DF triples denominator");
  return minimum;
}

std::size_t prepared_input_storage_bytes(const Layout& p) {
  std::size_t cursor = 0;
  for (const auto size : p.sizes) (void)reserve(cursor, bytes(size));
  return align256(cursor);
}

struct PreparedInputs {
  unsigned char* base{};
  std::size_t storage_bytes{}, h2d_bytes{};
  generated_df::Inputs values;

  PreparedInputs(const Layout& p, const std::array<const double*, 9>& host) {
    std::size_t cursor = 0;
    std::array<std::size_t, 9> offsets{};
    for (std::size_t x = 0; x < p.sizes.size(); ++x)
      offsets[x] = reserve(cursor, bytes(p.sizes[x]));
    storage_bytes = align256(cursor);
    generativeqc_tensor::cuda_check(cudaMalloc(reinterpret_cast<void**>(&base), storage_bytes));
    try {
      const std::array<const double**, 9> fields{&values.bov,  &values.bvv,   &values.ovoo,
                                                 &values.ovov, &values.fov,   &values.t1,
                                                 &values.t2,   &values.eps_o, &values.eps_v};
      for (std::size_t x = 0; x < host.size(); ++x) {
        auto* destination = base + offsets[x];
        *fields[x] = reinterpret_cast<const double*>(destination);
        generativeqc_tensor::cuda_check(
            cudaMemcpy(destination, host[x], bytes(p.sizes[x]), cudaMemcpyHostToDevice));
        h2d_bytes = checked_add(h2d_bytes, bytes(p.sizes[x]));
      }
    } catch (...) {
      cudaFree(base);
      base = nullptr;
      throw;
    }
  }

  PreparedInputs(const PreparedInputs&) = delete;
  PreparedInputs& operator=(const PreparedInputs&) = delete;
  ~PreparedInputs() {
    if (base) cudaFree(base);
  }
};

void validate_response_work(std::size_t o, std::size_t v, std::size_t q, const Layout& p,
                            bool include_gap_response) {
  const auto scalar_outputs = include_gap_response
                                  ? generated_df::response_scalar_outputs_with_gap
                                  : generated_df::response_scalar_outputs_without_gap;
  (void)checked_mul(scalar_outputs, checked_mul(p.tiles, p.v3));
  // At most three forward and three reverse-regenerated panels per tile, plus
  // two factor products per distinct occupied index. Charge complete work
  // before input access even for logical shapes too large to execute.
  const auto panel_work = checked_mul(checked_mul(12, p.tiles), checked_mul(q, p.v3));
  const auto w_work = checked_mul(checked_mul(18, p.tiles),
                                  checked_add(checked_mul(v, p.v3), checked_mul(o, p.v3)));
  const auto v_work = checked_mul(checked_mul(24, p.tiles), p.v3);
  (void)checked_add(panel_work, checked_add(w_work, v_work));
}

struct ResponseLayout {
  Layout value;
  std::array<std::size_t, 9> outputs{};
  std::size_t bar_w{}, bar_v{}, bar_panel{}, packed{}, gap{}, host_bytes{}, output_bytes{},
      complete{};
};
ResponseLayout response_layout(std::size_t o, std::size_t v, std::size_t q, std::size_t panels,
                               std::size_t caller_bytes, bool parallel_gap_reduction,
                               bool include_gap_response, bool external_inputs = false) {
  ResponseLayout p;
  p.value = layout(o, v, q, panels, 0, 0, provider_allowance, blas_workspace, external_inputs);
  auto cursor = p.value.arena;
  std::size_t values = 0;
  for (std::size_t x = 0; x < p.outputs.size(); ++x) {
    p.host_bytes = checked_add(p.host_bytes, bytes(p.value.sizes[x]));
    if (!include_gap_response && x >= 7) continue;
    p.outputs[x] = reserve(cursor, bytes(p.value.sizes[x]));
    values = checked_add(values, p.value.sizes[x]);
  }
  p.bar_w = reserve(cursor, bytes(checked_mul(6, p.value.v3)));
  p.bar_v = reserve(cursor, bytes(p.value.v3));
  p.bar_panel = reserve(cursor, bytes(p.value.v3));
  p.packed = reserve(cursor, bytes(checked_mul(v, v)));
  if (include_gap_response)
    p.gap = reserve(cursor,
                    bytes(generated_df::gap_response_arena_elements(o, v, parallel_gap_reduction)));
  p.value.arena = align256(cursor);
  p.value.total = checked_add(p.value.arena, provider_allowance);
  p.output_bytes = bytes(values);
  p.complete = checked_add(caller_bytes,
                           checked_add(p.value.total, checked_add(p.host_bytes, p.output_bytes)));
  validate_response_work(o, v, q, p.value, include_gap_response);
  return p;
}

struct FockLayout {
  Layout value;
  std::size_t foo{}, fvv{}, xl{}, yl{}, xr{}, yr{}, block{};
  std::size_t capacity{}, pages{}, pairs{}, cubes{}, page_builds{}, page_pairs{};
  std::size_t host_bytes{}, output_bytes{}, complete{}, contraction_summands_bound{};
};
FockLayout fock_layout(std::size_t o, std::size_t v, std::size_t q, std::size_t capacity,
                       std::size_t panels, std::size_t caller_bytes, bool external_inputs = false) {
  FockLayout r;
  r.value = layout(o, v, q, panels, 0, 0, provider_allowance, blas_workspace, external_inputs);
  auto& p = r.value;
  // Cross-page oo contracts a flattened complete virtual cube as BLAS k.
  if (p.v3 > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("DF triples Fock response exceeds BLAS indexing");
  r.capacity = capacity;
  r.pages = 1 + (o - 1) / capacity;
  r.pairs = checked_mul(o, checked_add(o, 1)) / 2;
  r.page_pairs = checked_mul(r.pages, checked_add(r.pages, 1)) / 2;
  r.page_builds = checked_mul(r.pairs, r.page_pairs);
  const auto tail = o - checked_mul(r.pages - 1, capacity);
  const auto prefix = checked_mul(checked_mul(capacity, r.pages), r.pages - 1) / 2;
  r.cubes = checked_mul(r.pairs, checked_add(prefix, checked_mul(tail, r.pages)));
  // Charge the complete replay ledger before touching input pointers, including
  // zero-padded cross-page products. Scalar evaluations are separately reported.
  (void)checked_mul(r.cubes, p.v3);
  const auto panel_work = checked_mul(checked_mul(3, r.cubes), checked_mul(q, p.v3));
  const auto w_work =
      checked_mul(checked_mul(6, r.cubes), checked_add(checked_mul(v, p.v3), checked_mul(o, p.v3)));
  const auto vv_work = checked_mul(checked_mul(2, checked_mul(r.pairs, o)), checked_mul(v, p.v3));
  const auto oo_work = checked_mul(checked_mul(2, r.page_builds),
                                   checked_mul(checked_mul(capacity, capacity), p.v3));
  r.contraction_summands_bound =
      checked_add(checked_add(panel_work, w_work), checked_add(vv_work, oo_work));
  auto cursor = p.arena;
  r.foo = reserve(cursor, bytes(checked_mul(o, o)));
  r.fvv = reserve(cursor, bytes(checked_mul(v, v)));
  const auto page_bytes = bytes(checked_mul(capacity, p.v3));
  r.xl = reserve(cursor, page_bytes);
  r.yl = reserve(cursor, page_bytes);
  // A full occupied page needs only one X/Y pair. Under a smaller budget the
  // right pair is independently owned and reused across cross-page moments.
  r.xr = r.pages == 1 ? r.xl : reserve(cursor, page_bytes);
  r.yr = r.pages == 1 ? r.yl : reserve(cursor, page_bytes);
  r.block = reserve(cursor, bytes(checked_mul(capacity, capacity)));
  p.arena = align256(cursor);
  p.total = checked_add(p.arena, provider_allowance);
  for (auto size : p.sizes) r.host_bytes = checked_add(r.host_bytes, bytes(size));
  r.output_bytes = bytes(checked_add(checked_mul(o, o), checked_mul(v, v)));
  r.complete =
      checked_add(caller_bytes, checked_add(p.total, checked_add(r.host_bytes, r.output_bytes)));
  return r;
}

struct JointResponseLayout {
  FockLayout fock;
  std::array<std::size_t, 9> response_outputs{};
  std::size_t bar_w{}, bar_v{}, bar_panel{}, packed{}, gap{};
  std::size_t unweighted_energies{}, occupied_energies{}, device_weights{};
  std::size_t response_output_bytes{}, complete{};
};

JointResponseLayout joint_response_layout(std::size_t o, std::size_t v, std::size_t q,
                                          std::size_t capacity, std::size_t panels,
                                          std::size_t caller_bytes, bool parallel_gap_reduction,
                                          bool include_gap_response,
                                          bool fused_scalar_response = false,
                                          bool has_occupied_weights = false,
                                          bool occupied_partition = false) {
  JointResponseLayout r;
  r.fock = fock_layout(o, v, q, capacity, panels, 0, true);
  auto& p = r.fock.value;
  (void)checked_mul(checked_mul(6, r.fock.cubes), bytes(p.v3));
  validate_response_work(o, v, q, p, include_gap_response);
  const auto scalar_points = checked_mul(p.tiles, p.v3);
  (void)checked_mul(scalar_points, generated_df::scalar_unfused_arithmetic_ops);
  (void)checked_mul(scalar_points, generated_df::scalar_unfused_value_reads);
  (void)checked_add(checked_mul(12, scalar_points), checked_mul(p.tiles, p.blocks));
  // A pullback can regenerate up to three evicted panels per triangular tile.
  // Those products share the Fock ledger, including its right-page replay work.
  const auto regenerated_panel_work = checked_mul(checked_mul(3, p.tiles), checked_mul(q, p.v3));
  (void)checked_add(r.fock.contraction_summands_bound, regenerated_panel_work);
  auto cursor = p.arena;
  std::size_t values = 0;
  const std::size_t requested_outputs = include_gap_response ? 9 : 7;
  for (std::size_t x = 0; x < requested_outputs; ++x) {
    r.response_outputs[x] = reserve(cursor, bytes(p.sizes[x]));
    values = checked_add(values, p.sizes[x]);
  }
  r.bar_w = reserve(cursor, bytes(checked_mul(6, p.v3)));
  r.bar_v = reserve(cursor, bytes(checked_mul(fused_scalar_response ? 6 : 1, p.v3)));
  r.bar_panel = reserve(cursor, bytes(p.v3));
  r.packed = reserve(cursor, bytes(checked_mul(v, v)));
  if (include_gap_response)
    r.gap = reserve(cursor,
                    bytes(generated_df::gap_response_arena_elements(o, v, parallel_gap_reduction)));
  if (occupied_partition) {
    if (has_occupied_weights) r.unweighted_energies = reserve(cursor, bytes(p.tiles));
    r.occupied_energies = reserve(cursor, bytes(o));
    values = checked_add(values, o);
  }
  if (has_occupied_weights) r.device_weights = reserve(cursor, bytes(o));
  p.arena = align256(cursor);
  p.total = checked_add(p.arena, provider_allowance);
  r.response_output_bytes = bytes(values);
  r.complete = checked_add(
      caller_bytes,
      checked_add(p.total, checked_add(r.fock.host_bytes,
                                       checked_add(r.fock.output_bytes, r.response_output_bytes))));
  return r;
}

// A cross-page moment is already symmetric in its two factors. Scatter each
// matrix block once and mirror only off-diagonal page pairs; discard padded rows.
__global__ void scatter_fock_block(std::size_t o, std::size_t capacity, std::size_t left,
                                   std::size_t right, const double* block, double* foo,
                                   int* error) {
  for (std::size_t flat = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       flat < capacity * capacity; flat += std::size_t(blockDim.x) * gridDim.x) {
    const auto i = left + flat / capacity, j = right + flat % capacity;
    if (i >= o || j >= o) continue;
    foo[i * o + j] = generativeqc_tensor::finite(foo[i * o + j] + block[flat], error, 10);
    if (left != right)
      foo[j * o + i] = generativeqc_tensor::finite(foo[j * o + i] + block[flat], error, 10);
  }
}

__global__ void scatter_weighted_degenerate_fock_block(std::size_t o, std::size_t capacity,
                                                       std::size_t left, std::size_t right,
                                                       std::size_t j_idx, std::size_t k_idx,
                                                       const double* eps_o,
                                                       const double* occupied_weights,
                                                       const double* block, double* foo,
                                                       int* error) {
  const double wjk = occupied_weights[j_idx] + occupied_weights[k_idx];
  for (std::size_t flat = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       flat < capacity * capacity; flat += std::size_t(blockDim.x) * gridDim.x) {
    const auto p = left + flat / capacity, r = right + flat % capacity;
    if (p >= o || r >= o || p == r) continue;
    if (fabs(eps_o[p] - eps_o[r]) >= 1e-6) continue;
    const double scaled = block[flat] * ((occupied_weights[p] + wjk) / 3.0);
    foo[p * o + r] = generativeqc_tensor::finite(foo[p * o + r] + scaled, error, 10);
    if (left != right)
      foo[r * o + p] = generativeqc_tensor::finite(foo[r * o + p] + scaled, error, 10);
  }
}

__global__ void scale_scalar(const double* src, double factor, double* dst, int* error) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    *dst = generativeqc_tensor::finite((*src) * factor, error, 3);
}

// Audit only the logical output of a strided BLAS view. Padding can contain
// unrelated valid tensor elements and is not part of this contraction's result.
__global__ void audit_matrix(const double* c, std::size_t m, std::size_t n, std::size_t ldc,
                             int* error) {
  for (std::size_t flat = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; flat < m * n;
       flat += std::size_t(blockDim.x) * gridDim.x)
    if (!isfinite(c[(flat % m) + (flat / m) * ldc])) atomicCAS(error, 0, 7);
}

/** Existing explicit-stride FP64 matrix product boundary.
 *
 * This centralizes the legacy method-local call while #1890 migrates the
 * generated triples views to the shared provider-neutral contraction binding.
 * Scientific callers still supply the compiler-derived transpose/stride data.
 */
inline void execute_strided_fp64_product(generativeqc_tensor::Context& context, char ta, char tb,
                                         std::size_t m, std::size_t n, std::size_t k, double alpha,
                                         const double* a, std::size_t lda, const double* b,
                                         std::size_t ldb, double beta, double* c, std::size_t ldc) {
  // Keep the original column-major strides and host alpha/beta pointers.
  // The shared Tensor provider owns the actual vendor submission.
  const generativeqc_tensor::ColumnGemmView request{
      .handle = context.handle,
      .transpose_left = ta != 'N',
      .transpose_right = tb != 'N',
      .rows = static_cast<int>(m),
      .columns = static_cast<int>(n),
      .reduction = static_cast<int>(k),
      .alpha = &alpha,
      .left = a,
      .left_ld = static_cast<int>(lda),
      .right = b,
      .right_ld = static_cast<int>(ldb),
      .beta = &beta,
      .output = c,
      .output_ld = static_cast<int>(ldc),
  };
  generativeqc_tensor::blas_check(generativeqc_tensor::submit_column_gemm(request));
}

// Pure view packing/scatter. The scientific V products stay compiler-owned.
template <bool Scatter>
__global__ void ovov_view(std::size_t o, std::size_t v, std::size_t i, std::size_t j,
                          const double* source, double* destination, int* error) {
  for (std::size_t x = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; x < v * v;
       x += std::size_t(blockDim.x) * gridDim.x) {
    const auto global = ((i * v + x / v) * o + j) * v + x % v;
    if constexpr (Scatter)
      destination[global] = generativeqc_tensor::finite(destination[global] + source[x], error, 8);
    else
      destination[x] = source[global];
  }
}

__global__ void scatter_gap(std::size_t v, std::size_t i, std::size_t j, std::size_t k,
                            generated_df::GapOutputs values, double* occupied, double* virtuals,
                            int* error) {
  // One thread owns all three occupied writes, including repeated indices.
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    occupied[i] = generativeqc_tensor::finite(occupied[i] + *values.bar_eps_i, error, 9);
    occupied[j] = generativeqc_tensor::finite(occupied[j] + *values.bar_eps_j, error, 9);
    occupied[k] = generativeqc_tensor::finite(occupied[k] + *values.bar_eps_k, error, 9);
  }
  for (std::size_t x = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; x < v;
       x += std::size_t(blockDim.x) * gridDim.x)
    virtuals[x] = generativeqc_tensor::finite(virtuals[x] + values.bar_eps_v[x], error, 9);
}

__device__ void store_reduction(double sum, double* output, int* error) {
  __shared__ double sums[256];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) *output = generativeqc_tensor::finite(sums[0], error, 3);
}

// Deterministic generic reduction, shared by each tile and final publication.
__global__ void reduce(const double* values, std::size_t count, double* output, int* error) {
  double sum = 0.0;
  for (std::size_t x = threadIdx.x; x < count; x += blockDim.x) sum += values[x];
  store_reduction(sum, output, error);
}

__global__ void reduce_occupied_energies(const double* tiles, std::size_t occupied,
                                         double* output, int* error) {
  const std::size_t orbital = blockIdx.x;
  double sum = 0.0;
  for (std::size_t first = threadIdx.x; first < occupied; first += blockDim.x)
    for (std::size_t second = 0; second <= first; ++second) {
      const auto share = detail::occupied_tile_share(orbital, first, second);
      sum += static_cast<double>(share.count) * tiles[share.index];
    }
  store_reduction(sum * generated_df::occupied_orbital_share, output + orbital, error);
}

std::size_t float_bytes(std::size_t n) { return checked_mul(n, sizeof(float)); }

std::size_t query_triples_device_cap(int device, std::size_t max_bytes) {
  runtime::CudaDeviceScope query_scope(device);
  std::size_t available_device_bytes = runtime::device_resource_available(device);
  std::size_t avail = 0, total = 0;
  if (cudaMemGetInfo(&avail, &total) == cudaSuccess && avail > 0) {
    available_device_bytes = std::min(available_device_bytes, avail);
  }
  return max_bytes > 0 ? std::min(max_bytes, available_device_bytes) : available_device_bytes;
}

// Both fast routines issue pedantic cuBLAS SGEMMs directly. Other selected
// providers and precisions must run through their own prepared W execution.
bool supports_fast_fp32_w(const generated_df::WPlan& plan, std::size_t max_panel_buffers) {
  return max_panel_buffers >= 3 && !plan.precision.arithmetic.is_strict_fp64() &&
         plan.algorithm() == tensor::ContractionAlgorithm::PedanticBlas;
}

__global__ void combine_fp32_w_to_fp64_kernel(const float* w1, const float* w2, double* out,
                                              std::size_t count, int* error) {
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < count;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const double a = static_cast<double>(generativeqc_tensor::finite(w1[idx], error, 22));
    const double b = static_cast<double>(generativeqc_tensor::finite(w2[idx], error, 22));
    out[idx] = generativeqc_tensor::finite(__dadd_rn(a, -b), error, 22);
  }
}

__global__ void transpose_01_cube_kernel(std::size_t v, const double* src, double* dst) {
  const std::size_t v2 = v * v, v3 = v * v2;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < v3;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t c = idx % v;
    const std::size_t ab = idx / v;
    const std::size_t b = ab % v;
    const std::size_t a = ab / v;
    dst[(b * v + a) * v + c] = src[idx];
  }
}

__global__ void add_sym_matrix_kernel(std::size_t v, double scale, const double* src, double* dst,
                                      int* error) {
  const std::size_t v2 = v * v;
  for (std::size_t idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < v2;
       idx += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t a = idx / v, d = idx % v;
    const double val = __dmul_rn(scale, __dadd_rn(src[a * v + d], src[d * v + a]));
    dst[idx] = generativeqc_tensor::finite(__dadd_rn(dst[idx], val), error, 10);
  }
}

__global__ void scatter_degenerate_list_fock_block(
    std::size_t o, std::size_t deg_count, std::size_t capacity, std::size_t left, std::size_t right,
    std::size_t j_idx, std::size_t k_idx, const std::size_t* deg_indices, const double* eps_o,
    const double* occupied_weights, const double* block, double* foo, int* error) {
  const double wjk = occupied_weights[j_idx] + occupied_weights[k_idx];
  for (std::size_t flat = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       flat < capacity * capacity; flat += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t u = flat / capacity, w = flat % capacity;
    if (left + u >= deg_count || right + w >= deg_count) continue;
    const std::size_t p = deg_indices[left + u];
    const std::size_t r = deg_indices[right + w];
    if (p == r || fabs(eps_o[p] - eps_o[r]) >= 1e-6) continue;
    const double scaled = block[flat] * ((occupied_weights[p] + wjk) / 3.0);
    foo[p * o + r] = generativeqc_tensor::finite(foo[p * o + r] + scaled, error, 10);
    if (left != right)
      foo[r * o + p] = generativeqc_tensor::finite(foo[r * o + p] + scaled, error, 10);
  }
}

/** Fast FP32 forward-W energy path for all o >= 1 with FP32 panel caching.
 * Matches try_pullback_and_fock_df_cuda_fp32_w panel and forward-W math identically
 * (including o = 1, 2, 3) while keeping W subtraction, V, denominators, epilogue,
 * and reductions strictly FP64.
 * Returns std::nullopt on resource refusal so the caller's bounded fallback runs.
 */
std::optional<DFCudaResult> try_evaluate_df_cuda_fp32_w_fast(
    std::size_t o, std::size_t v, std::size_t q, const std::array<const double*, 9>& host,
    double threshold, std::size_t max_bytes, int device,
    const generated_df::WPlan& execution_plan, bool occupied_partition,
    Clock::time_point started, bool compensated_fp32_w) {
  const auto base_l = layout(o, v, q, 1, 0, 0, provider_allowance, 0, false, 0, occupied_partition);
  const auto v2 = checked_mul(v, v);
  const auto v3 = base_l.v3;
  const auto occupied_result_bytes = occupied_partition ? bytes(o) : 0;
  const auto device_cap = query_triples_device_cap(device, max_bytes);

  struct FastEnergyLayout {
    std::array<std::size_t, 9> inputs{};
    std::size_t moments{}, partials{}, energies{}, energy{}, occupied_energies{}, error{};
    std::size_t bov_f32{}, bvv_f32{}, ovoo_f32{}, t2_f32{}, w_f32_1{}, w_f32_2{};
    std::size_t panels{}, split_workspace{}, split_workspace_bytes{};
    std::size_t panel_cap{}, arena{}, total{};
  };
  auto plan_for = [&](std::size_t cap) {
    FastEnergyLayout fl;
    fl.panel_cap = cap;
    std::size_t cursor = 0;
    for (std::size_t x = 0; x < 9; ++x) fl.inputs[x] = reserve(cursor, bytes(base_l.sizes[x]));
    fl.moments = reserve(cursor, bytes(checked_mul(6, v3)));
    fl.partials = reserve(cursor, bytes(base_l.blocks));
    fl.energies = reserve(cursor, bytes(base_l.tiles));
    fl.energy = reserve(cursor, sizeof(double));
    if (occupied_partition) fl.occupied_energies = reserve(cursor, bytes(o));
    fl.error = reserve(cursor, sizeof(int));
    const auto panel_elements = checked_mul(cap, v3);
    if (compensated_fp32_w) {
      fl.panels = reserve(cursor, bytes(panel_elements));
      fl.split_workspace_bytes = std::max({compensated_w::workspace_bytes(v, v2, q),
                                           compensated_w::workspace_bytes(v2, v, v),
                                           compensated_w::workspace_bytes(v, v2, o)});
      fl.split_workspace = reserve(cursor, fl.split_workspace_bytes);
    } else {
      fl.bov_f32 = reserve(cursor, float_bytes(base_l.sizes[0]));
      fl.bvv_f32 = reserve(cursor, float_bytes(base_l.sizes[1]));
      fl.ovoo_f32 = reserve(cursor, float_bytes(base_l.sizes[2]));
      fl.t2_f32 = reserve(cursor, float_bytes(base_l.sizes[6]));
      fl.w_f32_1 = reserve(cursor, float_bytes(v3));
      fl.w_f32_2 = reserve(cursor, float_bytes(v3));
      fl.panels = reserve(cursor, float_bytes(panel_elements));
    }
    fl.arena = align256(cursor);
    const auto prov = execution_plan.provider_bytes();
    fl.total = checked_add(checked_add(fl.arena, prov), occupied_result_bytes);
    return fl;
  };

  std::optional<FastEnergyLayout> selected;
  for (const std::size_t cap : {o, std::min<std::size_t>(o, 3), std::size_t{1}}) {
    const auto cand = plan_for(cap);
    if (cand.total <= max_bytes &&
        checked_add(cand.arena, execution_plan.provider_bytes()) <= device_cap) {
      selected = cand;
      break;
    }
  }
  if (!selected) return std::nullopt;
  const auto& fl = *selected;
  const double minimum = validate_inputs(o, v, q, base_l, host, threshold);

  DFCudaResult result;
  if (occupied_partition) result.occupied_energies.resize(o);
  int failed = 0;
  {
    runtime::CudaDeviceScope device_scope(device);
    cudaDeviceProp properties{};
    generativeqc_tensor::cuda_check(cudaGetDeviceProperties(&properties, device));
    generativeqc_tensor::Context context;
    context.prepare(device, properties.major, properties.minor, fl.arena, fl.error, 0, 0, 0, false);
    tensor::CudaContractionContext fp32_provider;
    if (!fp32_provider.prepare(context.stream)) return std::nullopt;

    generated_df::Inputs in;
    const std::array<const double**, 9> fields{&in.bov, &in.bvv, &in.ovoo,  &in.ovov, &in.fov,
                                               &in.t1,  &in.t2,  &in.eps_o, &in.eps_v};
    for (std::size_t x = 0; x < 9; ++x) {
      *fields[x] = reinterpret_cast<const double*>(context.arena + fl.inputs[x]);
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(context.arena + fl.inputs[x], host[x],
                                                      bytes(base_l.sizes[x]),
                                                      cudaMemcpyHostToDevice, context.stream));
      result.h2d_bytes = checked_add(result.h2d_bytes, bytes(base_l.sizes[x]));
    }
    generativeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));

    auto* moments = reinterpret_cast<double*>(context.arena + fl.moments);
    auto* partials = reinterpret_cast<double*>(context.arena + fl.partials);
    auto* energies = reinterpret_cast<double*>(context.arena + fl.energies);
    auto* energy = reinterpret_cast<double*>(context.arena + fl.energy);
    auto* bov_f32 = reinterpret_cast<float*>(context.arena + fl.bov_f32);
    auto* bvv_f32 = reinterpret_cast<float*>(context.arena + fl.bvv_f32);
    auto* ovoo_f32 = reinterpret_cast<float*>(context.arena + fl.ovoo_f32);
    auto* t2_f32 = reinterpret_cast<float*>(context.arena + fl.t2_f32);
    auto* w_f32_1 = reinterpret_cast<float*>(context.arena + fl.w_f32_1);
    auto* w_f32_2 = reinterpret_cast<float*>(context.arena + fl.w_f32_2);
    auto* panels_f32 = reinterpret_cast<float*>(context.arena + fl.panels);
    auto* panels_fp64 = reinterpret_cast<double*>(context.arena + fl.panels);

    if (!compensated_fp32_w) {
      generativeqc_tensor::convert_fp64_to_fp32(
          context, in.bov, bov_f32, static_cast<generativeqc_tensor::I>(base_l.sizes[0]), 20);
      generativeqc_tensor::convert_fp64_to_fp32(
          context, in.bvv, bvv_f32, static_cast<generativeqc_tensor::I>(base_l.sizes[1]), 20);
      generativeqc_tensor::convert_fp64_to_fp32(
          context, in.ovoo, ovoo_f32, static_cast<generativeqc_tensor::I>(base_l.sizes[2]), 20);
      generativeqc_tensor::convert_fp64_to_fp32(
          context, in.t2, t2_f32, static_cast<generativeqc_tensor::I>(base_l.sizes[6]), 21);
      result.precision_cast_elements = checked_add(
          result.precision_cast_elements,
          checked_add(checked_add(base_l.sizes[0], base_l.sizes[1]),
                      checked_add(base_l.sizes[2], base_l.sizes[6])));
    }

    auto split_product = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t k,
                             double alpha, const double* a, std::size_t lda, const double* b,
                             std::size_t ldb, double beta, double* c, std::size_t ldc) {
      const auto calls = compensated_w::execute(fp32_provider, context.stream, ta, tb, m, n, k,
                                                alpha, a, lda, b, ldb, beta, c, ldc,
                                                context.arena + fl.split_workspace,
                                                fl.split_workspace_bytes, context.error);
      result.fp32_gemms = checked_add(result.fp32_gemms, calls);
      result.fp32_forward_gemms = checked_add(result.fp32_forward_gemms, calls);
      result.precision_cast_elements =
          checked_add(result.precision_cast_elements, compensated_w::cast_elements(m, n, k));
      return calls;
    };
    auto build_panel = [&](std::size_t occ, std::size_t slot) {
      if (compensated_fp32_w) {
        const auto calls = split_product('T', 'N', v, v2, q, 1., in.bov + occ * v, o * v,
                                         in.bvv, v2, 0., panels_fp64 + slot * v3, v2);
        result.panel_gemms = checked_add(result.panel_gemms, calls);
      } else {
        execute_strided_fp32_product(fp32_provider, context.stream, 'T', 'N', v, v2, q, 1.0f,
                                     bov_f32 + occ * v, o * v, bvv_f32, v2, 0.0f,
                                     panels_f32 + slot * v3, v2, context.error);
        ++result.panel_gemms;
        ++result.fp32_gemms;
        ++result.fp32_forward_gemms;
      }
      result.contraction_summands = checked_add(
          result.contraction_summands, checked_mul(q, checked_mul(v3, compensated_fp32_w ? 9 : 1)));
    };

    const bool cache_all = fl.panel_cap == o;
    std::array<std::size_t, 3> lru_ids;
    lru_ids.fill(std::numeric_limits<std::size_t>::max());
    std::array<std::size_t, 3> lru_ages{};
    std::size_t epoch = 0;
    if (cache_all) {
      for (std::size_t occ = 0; occ < o; ++occ) build_panel(occ, occ);
    }
    auto panel_slot_for = [&](std::size_t occ) {
      if (cache_all) return occ;
      std::size_t slot = 0;
      for (; slot < fl.panel_cap; ++slot)
        if (lru_ids[slot] == occ) break;
      if (slot == fl.panel_cap) {
        slot = static_cast<std::size_t>(
            std::min_element(lru_ages.begin(), lru_ages.begin() + fl.panel_cap) -
            lru_ages.begin());
        build_panel(occ, slot);
        lru_ids[slot] = occ;
      }
      lru_ages[slot] = ++epoch;
      return slot;
    };

    const unsigned v3_blocks = base_l.blocks;
    std::size_t tile_count = 0;
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j <= i; ++j)
        for (std::size_t k = 0; k <= j; ++k) {
          const std::array<std::size_t, 3> occupied{i, j, k};
          const auto canonical_tile = detail::canonical_tile_index(i, j, k);
          const auto sources = generated_df::occupied_moment_sources(i, j, k);
          for (std::size_t index = 0; index < 3; ++index) {
            if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
                occupied.begin() + index)
              continue;
            const auto I = occupied[index];
            const auto panel_offset = panel_slot_for(I) * v3;
            for (std::size_t perm = 0; perm < 6; ++perm) {
              const auto* order = generated_df::permutations[perm];
              if (occupied[order[0]] != I || sources.index[perm] != perm) continue;
              const auto J = occupied[order[1]], K = occupied[order[2]];
              if (compensated_fp32_w) {
                // The two row-major views have different shapes and share the same v^3
                // contiguous offsets. Apply W2's sign while combining in FP64.
                double* w = moments + perm * v3;
                const auto w1 = split_product('N', 'T', v2, v, v, 1., panels_fp64 + panel_offset, v,
                                              in.t2 + (K * o + J) * v2, v, 0., w, v);
                const auto w2 = split_product('N', 'N', v, v2, o, -1.,
                                              in.ovoo + (I * v * o + J) * o, o * o,
                                              in.t2 + K * v2, o * v2, 1., w, v2);
                result.moment_gemms = checked_add(result.moment_gemms, checked_add(w1, w2));
              } else {
                execute_strided_fp32_product(fp32_provider, context.stream, 'N', 'T', v2, v, v, 1.0f,
                                             panels_f32 + panel_offset, v, t2_f32 + (K * o + J) * v2,
                                             v, 0.0f, w_f32_1, v, context.error);
                execute_strided_fp32_product(fp32_provider, context.stream, 'N', 'N', v, v2, o, 1.0f,
                                             ovoo_f32 + (I * v * o + J) * o, o * o,
                                             t2_f32 + K * v2, o * v2, 0.0f, w_f32_2, v2,
                                             context.error);
                combine_fp32_w_to_fp64_kernel<<<v3_blocks, 256, 0, context.stream>>>(
                    w_f32_1, w_f32_2, moments + perm * v3, v3, context.error);
                generativeqc_tensor::cuda_check(cudaGetLastError());
                result.moment_gemms += 2;
                result.fp32_gemms += 2;
                result.fp32_forward_gemms += 2;
                result.precision_cast_elements =
                    checked_add(result.precision_cast_elements, checked_mul(2, v3));
              }
              const auto summands = checked_mul(v3, checked_add(v, o));
              result.contraction_summands = checked_add(
                  result.contraction_summands, checked_mul(summands, compensated_fp32_w ? 9 : 1));
            }
          }
          const double degeneracy = i == k ? 6.0 : (i == j || j == k ? 2.0 : 1.0);
          generated_df::energy_distinct_tile(o, v, i, j, k, degeneracy, threshold, in, moments,
                                             sources, base_l.blocks, partials, context.error,
                                             context.stream);
          ++result.epilogue_kernels;
          reduce<<<1, 256, 0, context.stream>>>(partials, base_l.blocks, energies + canonical_tile,
                                                context.error);
          generativeqc_tensor::cuda_check(cudaGetLastError());
          ++result.reduction_kernels;
          ++tile_count;
        }
    if (tile_count != base_l.tiles) throw std::logic_error("DF triples occupied work mismatch");
    reduce<<<1, 256, 0, context.stream>>>(energies, base_l.tiles, energy, context.error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    ++result.reduction_kernels;
    if (occupied_partition) {
      auto* occ_out = reinterpret_cast<double*>(context.arena + fl.occupied_energies);
      reduce_occupied_energies<<<static_cast<unsigned>(o), 256, 0, context.stream>>>(
          energies, o, occ_out, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++result.reduction_kernels;
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(result.occupied_energies.data(), occ_out,
                                                      occupied_result_bytes,
                                                      cudaMemcpyDeviceToHost, context.stream));
    }
    result.d2h_bytes = checked_add(sizeof(double) + sizeof(int), occupied_result_bytes);
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&result.energy, energy, sizeof(double),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&failed, context.error, sizeof(int),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    if (failed || !std::isfinite(result.energy))
      throw std::runtime_error("nonfinite or unsafe FP32-W DF triples arithmetic");

    const auto& candidate = generated_df::w_lowering_candidates[execution_plan.selected];
    result.precision = execution_plan.precision.arithmetic;
    if (compensated_fp32_w) result.precision.qualification = compensated_w::schedule;
    result.w_contraction_storage_bits = 32;
    result.w_contraction_compute_bits = 32;
    result.w_contraction_accumulation_bits = 32;
    result.w_reverse_contraction_compute_bits = 64;
    result.w_scientific_identity = generated_df::w_lowering_request.scientific_identity;
    result.w_semantic_identity = generated_df::w_lowering_request.semantic_identity;
    result.w_candidate_identity = compensated_fp32_w ? compensated_w::schedule : candidate.identity;
    result.w_provider = candidate.provider;
    result.w_algorithm = compensated_fp32_w ? compensated_w::schedule : candidate.algorithm;
    result.retained_incumbent = !compensated_fp32_w && execution_plan.retained_incumbent;
    result.w_provider_version = fp32_provider.provider_version();
    result.cuda_runtime_version = fp32_provider.runtime_version();
    result.w_precision_identity =
        compensated_fp32_w ? compensated_w::schedule : execution_plan.precision.identity;
    result.compensated_fp32_w_executed = compensated_fp32_w && result.fp32_forward_gemms > 0;
    result.w_codegen_precision_schedule_identity =
        compensated_fp32_w ? compensated_w::schedule : execution_plan.schedule_identity;
    result.resource_fallback = false;
    result.host_binding_bytes = occupied_result_bytes;
    result.minimum_absolute_denominator = minimum;
    result.virtual_triples = checked_mul(checked_mul(v, checked_add(v, 1)), checked_add(v, 2)) / 6;
    result.occupied_tiles = base_l.tiles;
    result.epilogue_points = checked_mul(base_l.tiles, v3);
    result.workspace_bytes = fl.total;
    result.arena_bytes = fl.arena;
    result.provider_retained_bytes = fp32_provider.retained_bytes();
    result.panel_capacity = fl.panel_cap;
  }
  result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}

/** Spatial DF triples pullback and full-Fock response with bounded panel caching.
 * The original fast schedule uses FP32 for panels, W, bar_panel and the virtual
 * bar_T2 contraction. Its occupied W and DF-panel pullbacks use FP64. The
 * compensated schedule uses split FP32 for every forward and reverse panel/W
 * contraction, with FP64 recombination and cached FP64 panels. Both schedules
 * retain FP64 V terms, denominators, scalar seeds, Fock response and reductions.
 * A zero-weight tile has no W pullback. Degenerate occupied blocks retain their
 * explicit Fock response and the weighted nondegenerate traversal is canonical.
 * Resource refusal precedes publication and leaves policy choice to the caller.
 */
std::optional<DFCudaCombinedResponseResult> try_pullback_and_fock_df_cuda_fp32_w(
    std::size_t o, std::size_t v, std::size_t q, const std::array<const double*, 9>& host,
    const Layout& staging_layout, std::size_t prepared_storage, std::size_t prepared_caller,
    double minimum, double threshold, std::size_t max_bytes, std::size_t triples_device_cap,
    int device, std::size_t max_page_rows, bool parallel_gap_reduction, bool include_gap_response,
    bool caller_requested_fused, const generated_df::WPlan& requested_w_plan,
    std::span<const double> occupied_weights, bool has_weights, bool has_occupied_degeneracy,
    bool occupied_partition, Clock::time_point started, bool compensated_fp32_w) {
  constexpr std::size_t kBarT2ChunkRows = 4096;
  const auto v2 = checked_mul(v, v);
  const auto v3 = staging_layout.v3;
  const auto tiles = staging_layout.tiles;
  const auto pairs = checked_mul(o, checked_add(o, 1)) / 2;
  const std::size_t requested_outputs = include_gap_response ? 9 : 7;

  std::vector<std::size_t> deg_list;
  if (has_weights && has_occupied_degeneracy) {
    const auto* eps_o = host[7];
    for (std::size_t a = 0; a < o; ++a) {
      for (std::size_t b = 0; b < o; ++b) {
        if (a != b && std::abs(eps_o[a] - eps_o[b]) < 1e-6) {
          deg_list.push_back(a);
          break;
        }
      }
    }
  }
  const bool canonical_tplus = has_weights;
  const bool run_deg_foo = canonical_tplus && deg_list.size() >= 2;
  const std::size_t page_domain =
      canonical_tplus ? (run_deg_foo ? deg_list.size() : std::size_t{1}) : o;
  const std::size_t initial_page_cap =
      canonical_tplus && !run_deg_foo
          ? std::size_t{1}
          : (max_page_rows ? std::min(page_domain, max_page_rows) : page_domain);

  std::size_t host_bytes = 0, response_output_elems = 0;
  for (std::size_t x = 0; x < 9; ++x) host_bytes = checked_add(host_bytes, bytes(staging_layout.sizes[x]));
  for (std::size_t x = 0; x < requested_outputs; ++x)
    response_output_elems = checked_add(response_output_elems, staging_layout.sizes[x]);
  if (occupied_partition) response_output_elems = checked_add(response_output_elems, o);
  const std::size_t response_output_bytes = bytes(response_output_elems);
  const std::size_t fock_output_bytes = bytes(checked_add(checked_mul(o, o), v2));
  const std::size_t dual_provider_bytes =
      checked_add(provider_allowance, requested_w_plan.provider_bytes());

  struct FastJointLayout {
    std::array<std::size_t, 9> outputs{};
    std::size_t foo{}, fvv{}, xl{}, yl{}, xr{}, yr{}, block{};
    std::size_t moments{}, bar_w{}, bar_v{}, packed{}, gap{};
    std::size_t partials{}, energies{}, energy{}, unweighted_energies{}, occupied_energies{},
        device_weights{}, deg_indices{}, error{}, library{};
    std::size_t bov_f32{}, bvv_f32{}, ovoo_f32{}, t2_f32{}, w_f32_1{}, w_f32_2{}, panels{};
    std::size_t split_workspace{}, split_workspace_bytes{};
    std::size_t bar_panel_buf{};
    std::size_t page_cap{}, pages{}, panel_cap{};
    bool cache_all_panels{}, cache_bar_panels{}, use_fused{};
    std::size_t arena{}, total{}, device_total{}, complete{};
  };

  auto build_plan = [&](std::size_t page_cap, bool cache_all_panels, bool cache_bar_panels,
                        bool use_fused) {
    FastJointLayout p;
    p.page_cap = page_cap;
    p.pages = 1 + (page_domain - 1) / page_cap;
    p.cache_all_panels = cache_all_panels;
    p.panel_cap = cache_all_panels ? o : std::min<std::size_t>(o, 3);
    p.cache_bar_panels = cache_bar_panels;
    p.use_fused = use_fused;
    std::size_t cursor = 0;
    for (std::size_t x = 0; x < requested_outputs; ++x)
      p.outputs[x] = reserve(cursor, bytes(staging_layout.sizes[x]));
    p.foo = reserve(cursor, bytes(checked_mul(o, o)));
    p.fvv = reserve(cursor, bytes(v2));
    const auto page_bytes = bytes(checked_mul(page_cap, v3));
    p.xl = reserve(cursor, page_bytes);
    p.yl = reserve(cursor, page_bytes);
    const bool need_right = (!canonical_tplus || run_deg_foo) && p.pages > 1;
    p.xr = need_right ? reserve(cursor, page_bytes) : p.xl;
    p.yr = need_right ? reserve(cursor, page_bytes) : p.yl;
    if (!canonical_tplus || run_deg_foo)
      p.block = reserve(cursor, bytes(checked_mul(page_cap, page_cap)));
    p.moments = reserve(cursor, bytes(checked_mul(6, v3)));
    p.bar_w = reserve(cursor, bytes(checked_mul(6, v3)));
    p.bar_v = reserve(cursor, bytes(checked_mul(use_fused ? 6 : 1, v3)));
    p.packed = reserve(cursor, bytes(v2));
    if (include_gap_response)
      p.gap = reserve(cursor,
                      bytes(generated_df::gap_response_arena_elements(o, v, parallel_gap_reduction)));
    p.partials = reserve(cursor, bytes(staging_layout.blocks));
    p.energies = reserve(cursor, bytes(tiles));
    p.energy = reserve(cursor, sizeof(double));
    if (occupied_partition) {
      if (has_weights) p.unweighted_energies = reserve(cursor, bytes(tiles));
      p.occupied_energies = reserve(cursor, bytes(o));
    }
    if (has_weights) p.device_weights = reserve(cursor, bytes(o));
    if (run_deg_foo) p.deg_indices = reserve(cursor, checked_mul(deg_list.size(), sizeof(std::size_t)));
    p.error = reserve(cursor, sizeof(int));
    p.library = reserve(cursor, blas_workspace);
    if (compensated_fp32_w) {
      p.panels = reserve(cursor, bytes(checked_mul(p.panel_cap, v3)));
      p.split_workspace_bytes = std::max({compensated_w::workspace_bytes(v, v2, q),
                                          compensated_w::workspace_bytes(v2, v, v),
                                          compensated_w::workspace_bytes(v, v2, o),
                                          compensated_w::workspace_bytes(v, v, v2),
                                          compensated_w::workspace_bytes(v, o, v2),
                                          compensated_w::workspace_bytes(o, v2, v),
                                          compensated_w::workspace_bytes(q, v, v2),
                                          compensated_w::workspace_bytes(q, v2, v)});
      p.split_workspace = reserve(cursor, p.split_workspace_bytes);
    } else {
      p.bov_f32 = reserve(cursor, float_bytes(staging_layout.sizes[0]));
      p.bvv_f32 = reserve(cursor, float_bytes(staging_layout.sizes[1]));
      p.ovoo_f32 = reserve(cursor, float_bytes(staging_layout.sizes[2]));
      p.t2_f32 = reserve(cursor, float_bytes(staging_layout.sizes[6]));
      p.w_f32_1 = reserve(cursor, float_bytes(v3));
      p.w_f32_2 = reserve(cursor, float_bytes(v3));
      p.panels = reserve(cursor, float_bytes(checked_mul(p.panel_cap, v3)));
    }
    p.bar_panel_buf = reserve(cursor, bytes(checked_mul(cache_bar_panels ? o : 1, v3)));
    p.arena = align256(cursor);
    p.total = checked_add(p.arena, dual_provider_bytes);
    p.device_total = checked_add(p.total, prepared_storage);
    p.complete = checked_add(
        prepared_caller,
        checked_add(p.total,
                    checked_add(host_bytes, checked_add(fock_output_bytes, response_output_bytes))));
    return p;
  };

  const bool allow_fused = caller_requested_fused && (has_weights || !include_gap_response);
  std::optional<FastJointLayout> selected;
  for (std::size_t page_cap = initial_page_cap; page_cap >= 1 && !selected; ) {
    const std::array<std::array<bool, 3>, 6> tiers{{
        {true, true, allow_fused},
        {true, true, false},
        {true, false, allow_fused},
        {true, false, false},
        {false, true, false},
        {false, false, false},
    }};
    for (const auto& t : tiers) {
      const auto cand = build_plan(page_cap, t[0], t[1], t[2]);
      if (cand.complete <= max_bytes && cand.device_total <= triples_device_cap) {
        selected = cand;
        break;
      }
    }
    if (selected || page_cap == 1) break;
    page_cap = page_cap > 4 ? (page_cap / 2) : (page_cap - 1);
  }
  if (!selected) return std::nullopt;
  const auto& fl = *selected;

  runtime::CudaDeviceScope device_scope(device);
  PreparedInputs prepared(staging_layout, host);

  DFCudaCombinedResponseResult result;
  auto& response = result.pullback;
  auto& d = response.diagnostic;
  auto& fock = result.fock;
  const std::array<std::vector<double>*, 9> response_outputs{
      &response.bov, &response.bvv, &response.ovoo,  &response.ovov, &response.fov,
      &response.t1,  &response.t2,  &response.eps_o, &response.eps_v};
  for (std::size_t x = 0; x < requested_outputs; ++x)
    response_outputs[x]->resize(staging_layout.sizes[x]);
  if (occupied_partition) d.occupied_energies.resize(o);
  fock.foo.resize(checked_mul(o, o));
  fock.fvv.resize(v2);

  int failed = 0;
  std::size_t response_tiles = 0;
  {
    cudaDeviceProp properties{};
    generativeqc_tensor::cuda_check(cudaGetDeviceProperties(&properties, device));
    generativeqc_tensor::Context context;
    context.prepare(device, properties.major, properties.minor, fl.arena, fl.error, fl.library,
                    blas_workspace, provider_allowance, true);
    tensor::CudaContractionContext fp32_provider;
    if (!fp32_provider.prepare(context.stream)) return std::nullopt;

    auto ptr_f64 = [&](std::size_t offset) {
      return reinterpret_cast<double*>(context.arena + offset);
    };
    auto ptr_f32 = [&](std::size_t offset) {
      return reinterpret_cast<float*>(context.arena + offset);
    };

    const generated_df::Inputs in = prepared.values;
    generated_df::ResponseOutputs out;
    const std::array<double**, 9> output_fields{&out.bov, &out.bvv, &out.ovoo,  &out.ovov, &out.fov,
                                                &out.t1,  &out.t2,  &out.eps_o, &out.eps_v};
    for (std::size_t x = 0; x < requested_outputs; ++x) {
      *output_fields[x] = ptr_f64(fl.outputs[x]);
      generativeqc_tensor::cuda_check(cudaMemsetAsync(
          ptr_f64(fl.outputs[x]), 0, bytes(staging_layout.sizes[x]), context.stream));
    }
    generativeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    generativeqc_tensor::cuda_check(
        cudaMemsetAsync(ptr_f64(fl.foo), 0, bytes(checked_mul(o, o)), context.stream));
    generativeqc_tensor::cuda_check(
        cudaMemsetAsync(ptr_f64(fl.fvv), 0, bytes(v2), context.stream));
    if (fl.cache_bar_panels) {
      generativeqc_tensor::cuda_check(
          cudaMemsetAsync(ptr_f64(fl.bar_panel_buf), 0, bytes(checked_mul(o, v3)), context.stream));
    }
    if (has_weights) {
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(ptr_f64(fl.device_weights),
                                                      occupied_weights.data(), bytes(o),
                                                      cudaMemcpyHostToDevice, context.stream));
    }
    if (run_deg_foo) {
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(
          context.arena + fl.deg_indices, deg_list.data(),
          checked_mul(deg_list.size(), sizeof(std::size_t)), cudaMemcpyHostToDevice,
          context.stream));
    }

    auto* bov_f32 = ptr_f32(fl.bov_f32);
    auto* bvv_f32 = ptr_f32(fl.bvv_f32);
    auto* ovoo_f32 = ptr_f32(fl.ovoo_f32);
    auto* t2_f32 = ptr_f32(fl.t2_f32);
    auto* w_f32_1 = ptr_f32(fl.w_f32_1);
    auto* w_f32_2 = ptr_f32(fl.w_f32_2);
    auto* panels_f32 = ptr_f32(fl.panels);
    auto* panels_f64 = ptr_f64(fl.panels);
    auto* moments = ptr_f64(fl.moments);
    auto* bar_w = ptr_f64(fl.bar_w);
    auto* bar_v = ptr_f64(fl.bar_v);
    auto* packed = ptr_f64(fl.packed);
    auto* bar_panel_buf = ptr_f64(fl.bar_panel_buf);

    if (!compensated_fp32_w) {
      generativeqc_tensor::convert_fp64_to_fp32(
          context, in.bov, bov_f32, static_cast<generativeqc_tensor::I>(staging_layout.sizes[0]), 20);
      generativeqc_tensor::convert_fp64_to_fp32(
          context, in.bvv, bvv_f32, static_cast<generativeqc_tensor::I>(staging_layout.sizes[1]), 20);
      generativeqc_tensor::convert_fp64_to_fp32(
          context, in.ovoo, ovoo_f32, static_cast<generativeqc_tensor::I>(staging_layout.sizes[2]),
          20);
      generativeqc_tensor::convert_fp64_to_fp32(
          context, in.t2, t2_f32, static_cast<generativeqc_tensor::I>(staging_layout.sizes[6]), 21);
      d.precision_cast_elements = checked_add(
          d.precision_cast_elements,
          checked_add(checked_add(staging_layout.sizes[0], staging_layout.sizes[1]),
                      checked_add(staging_layout.sizes[2], staging_layout.sizes[6])));
    }

    auto forward_gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t kk,
                            double alpha, const double* a, std::size_t lda, const double* b,
                            std::size_t ldb, double beta, double* c, std::size_t ldc) {
      execute_strided_fp64_product(context, ta, tb, m, n, kk, alpha, a, lda, b, ldb, beta, c, ldc);
      const auto count = checked_mul(m, n);
      const auto blocks =
          static_cast<unsigned>(std::min<std::size_t>(1 + (count - 1) / 256, 65535));
      audit_matrix<<<blocks, 256, 0, context.stream>>>(c, m, n, ldc, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++fock.audit_kernels;
      ++d.fp64_gemms;
      fock.contraction_summands = checked_add(fock.contraction_summands, checked_mul(count, kk));
    };
    auto reverse_gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t kk,
                            double alpha, const double* a, std::size_t lda, const double* b,
                            std::size_t ldb, double beta, double* c, std::size_t ldc) {
      execute_strided_fp64_product(context, ta, tb, m, n, kk, alpha, a, lda, b, ldb, beta, c, ldc);
      const auto count = checked_mul(m, n);
      const auto blocks =
          static_cast<unsigned>(std::min<std::size_t>(1 + (count - 1) / 256, 65535));
      audit_matrix<<<blocks, 256, 0, context.stream>>>(c, m, n, ldc, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++response.audit_kernels;
      ++response.reverse_gemms;
      ++d.fp64_gemms;
      d.contraction_summands = checked_add(d.contraction_summands, checked_mul(count, kk));
    };

    auto split_product = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t k,
                             double alpha, const double* a, std::size_t lda, const double* b,
                             std::size_t ldb, double beta, double* c, std::size_t ldc,
                             bool reverse) {
      const auto calls = compensated_w::execute(fp32_provider, context.stream, ta, tb, m, n, k,
                                                alpha, a, lda, b, ldb, beta, c, ldc,
                                                context.arena + fl.split_workspace,
                                                fl.split_workspace_bytes, context.error);
      d.fp32_gemms = checked_add(d.fp32_gemms, calls);
      auto& direction_calls = reverse ? d.fp32_reverse_gemms : d.fp32_forward_gemms;
      direction_calls = checked_add(direction_calls, calls);
      d.precision_cast_elements =
          checked_add(d.precision_cast_elements, compensated_w::cast_elements(m, n, k));
      auto& summands = reverse ? d.contraction_summands : fock.contraction_summands;
      summands = checked_add(summands, checked_mul(9, checked_mul(checked_mul(m, n), k)));
      return calls;
    };

    // Generated VJPs bind column-major strided views. Computing C^T = B^T A^T
    // with swapped operands/dimensions presents those same buffers to the row-major splitter.
    auto reverse_w_gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t k,
                              double alpha, const double* a, std::size_t lda, const double* b,
                              std::size_t ldb, double beta, double* c, std::size_t ldc) {
      if (!compensated_fp32_w) {
        reverse_gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
      }
      const auto calls = split_product(tb, ta, n, m, k, alpha, b, ldb, a, lda,
                                       beta, c, ldc, true);
      response.reverse_gemms = checked_add(response.reverse_gemms, calls);
    };

    auto build_panel = [&](std::size_t occ, std::size_t slot) {
      if (compensated_fp32_w) {
        const auto calls = split_product('T', 'N', v, v2, q, 1.0, in.bov + occ * v, o * v,
                                         in.bvv, v2, 0.0, panels_f64 + slot * v3, v2, false);
        fock.panel_gemms = checked_add(fock.panel_gemms, calls);
      } else {
        execute_strided_fp32_product(fp32_provider, context.stream, 'T', 'N', v, v2, q, 1.0f,
                                     bov_f32 + occ * v, o * v, bvv_f32, v2, 0.0f,
                                     panels_f32 + slot * v3, v2, context.error);
        ++fock.panel_gemms;
        ++d.fp32_gemms;
        ++d.fp32_forward_gemms;
        fock.contraction_summands = checked_add(fock.contraction_summands, checked_mul(q, v3));
      }
    };

    std::array<std::size_t, 3> lru_ids;
    lru_ids.fill(std::numeric_limits<std::size_t>::max());
    std::array<std::size_t, 3> lru_ages{};
    std::size_t epoch = 0;
    if (fl.cache_all_panels) {
      for (std::size_t occ = 0; occ < o; ++occ) build_panel(occ, occ);
    }
    auto panel_slot_for = [&](std::size_t occ) {
      if (fl.cache_all_panels) return occ;
      std::size_t slot = 0;
      for (; slot < fl.panel_cap; ++slot)
        if (lru_ids[slot] == occ) break;
      if (slot == fl.panel_cap) {
        slot = static_cast<std::size_t>(
            std::min_element(lru_ages.begin(), lru_ages.begin() + fl.panel_cap) -
            lru_ages.begin());
        build_panel(occ, slot);
        lru_ids[slot] = occ;
      }
      lru_ages[slot] = ++epoch;
      return slot;
    };

    const unsigned v3_blocks = staging_layout.blocks;
    const auto small_blocks =
        static_cast<unsigned>(std::min<std::size_t>(1 + (v2 - 1) / 256, 65535));

    auto build_w_fp32 = [&](std::size_t I, std::size_t J, std::size_t K, std::size_t panel_slot,
                            double* moment_dst) {
      if (compensated_fp32_w) {
        const auto w1 = split_product('N', 'T', v2, v, v, 1.0, panels_f64 + panel_slot * v3, v,
                                      in.t2 + (K * o + J) * v2, v, 0.0, moment_dst, v, false);
        const auto w2 = split_product('N', 'N', v, v2, o, -1.0, in.ovoo + (I * v * o + J) * o,
                                      o * o, in.t2 + K * v2, o * v2, 1.0, moment_dst, v2, false);
        fock.w_gemms = checked_add(fock.w_gemms, checked_add(w1, w2));
        return;
      }
      const auto* panel_f32_I = panels_f32 + panel_slot * v3;
      execute_strided_fp32_product(fp32_provider, context.stream, 'N', 'T', v2, v, v, 1.0f,
                                   panel_f32_I, v, t2_f32 + (K * o + J) * v2, v, 0.0f, w_f32_1, v,
                                   context.error);
      execute_strided_fp32_product(fp32_provider, context.stream, 'N', 'N', v, v2, o, 1.0f,
                                   ovoo_f32 + (I * v * o + J) * o, o * o, t2_f32 + K * v2, o * v2,
                                   0.0f, w_f32_2, v2, context.error);
      combine_fp32_w_to_fp64_kernel<<<v3_blocks, 256, 0, context.stream>>>(
          w_f32_1, w_f32_2, moment_dst, v3, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      d.fp32_gemms += 2;
      d.fp32_forward_gemms += 2;
      fock.w_gemms += 2;
      d.precision_cast_elements = checked_add(d.precision_cast_elements, checked_mul(2, v3));
      fock.contraction_summands = checked_add(
          fock.contraction_summands, checked_add(checked_mul(v3, v), checked_mul(v3, o)));
    };

    auto build_six_w_moments = [&](std::size_t i, std::size_t j, std::size_t k, bool alias_dups) {
      const std::array<std::size_t, 3> occupied{i, j, k};
      const auto sources = alias_dups ? generated_df::occupied_moment_sources(i, j, k)
                                      : generated_df::MomentSourceMap{};
      for (std::size_t index = 0; index < 3; ++index) {
        if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
            occupied.begin() + index)
          continue;
        const auto I = occupied[index];
        const auto panel_slot = panel_slot_for(I);
        for (std::size_t perm = 0; perm < 6; ++perm) {
          const auto* order = generated_df::permutations[perm];
          if (occupied[order[0]] != I) continue;
          const auto src = sources.index[perm];
          if (src != perm) {
            generativeqc_tensor::cuda_check(cudaMemcpyAsync(
                moments + perm * v3, moments + src * v3, bytes(v3), cudaMemcpyDeviceToDevice,
                context.stream));
            ++fock.w_copy_calls;
            fock.w_copy_bytes = checked_add(fock.w_copy_bytes, bytes(v3));
            continue;
          }
          build_w_fp32(I, occupied[order[1]], occupied[order[2]], panel_slot, moments + perm * v3);
        }
      }
    };

    auto run_pullback_w_fp32 = [&](std::size_t I, std::size_t J, std::size_t K,
                                   std::size_t panel_slot, const double* bar_w_perm,
                                   double* bar_panel_dst) {
      if (compensated_fp32_w) {
        const auto panels = split_product('N', 'N', v2, v, v, 1.0, bar_w_perm, v,
                                          in.t2 + (K * o + J) * v2, v, 1.0, bar_panel_dst,
                                          v, true);
        const auto doubles = split_product('T', 'N', v, v, v2, 1.0, bar_w_perm, v,
                                           panels_f64 + panel_slot * v3, v, 1.0,
                                           out.t2 + (K * o + J) * v2, v, true);
        response.reverse_gemms = checked_add(response.reverse_gemms, checked_add(panels, doubles));
      } else {
        const auto* panel_f32_I = panels_f32 + panel_slot * v3;
        // Cast bar_w_perm -> w_f32_2 (dead scratch), then run k=v bar_panel and chunked k<=4096
        // bar_t2 via pedantic IEEE FP32 SGEMM with strict FP64 outer accumulation.
        generativeqc_tensor::convert_fp64_to_fp32(
            context, bar_w_perm, w_f32_2, static_cast<generativeqc_tensor::I>(v3), 24);
        d.precision_cast_elements = checked_add(d.precision_cast_elements, v3);

        execute_strided_fp32_product(fp32_provider, context.stream, 'N', 'N', v2, v, v, 1.0f,
                                     w_f32_2, v, t2_f32 + (K * o + J) * v2, v, 0.0f, w_f32_1, v,
                                     context.error);
        generativeqc_tensor::accumulate_fp32_into_fp64(
            context, w_f32_1, bar_panel_dst, static_cast<generativeqc_tensor::I>(v3), 1.0, 1.0, 24);
        ++d.fp32_gemms;
        ++d.fp32_reverse_gemms;
        ++response.reverse_gemms;
        d.precision_cast_elements = checked_add(d.precision_cast_elements, v3);
        d.contraction_summands = checked_add(d.contraction_summands, checked_mul(v3, v));

        for (std::size_t row0 = 0; row0 < v2; row0 += kBarT2ChunkRows) {
          const std::size_t chunk = std::min(kBarT2ChunkRows, v2 - row0);
          execute_strided_fp32_product(fp32_provider, context.stream, 'T', 'N', v, v, chunk, 1.0f,
                                       w_f32_2 + row0 * v, v, panel_f32_I + row0 * v, v, 0.0f,
                                       w_f32_1, v, context.error);
          generativeqc_tensor::accumulate_fp32_into_fp64(
              context, w_f32_1, out.t2 + (K * o + J) * v2, static_cast<generativeqc_tensor::I>(v2),
              1.0, 1.0, 25);
          ++d.fp32_gemms;
          ++d.fp32_reverse_gemms;
          ++response.reverse_gemms;
          d.precision_cast_elements = checked_add(d.precision_cast_elements, v2);
        }
        d.contraction_summands = checked_add(d.contraction_summands, checked_mul(v3, v));
      }

      reverse_w_gemm('T', 'N', o, v, v2, -1.0, in.t2 + K * v2, o * v2, bar_w_perm, v2, 1.0,
                   out.ovoo + (I * v * o + J) * o, o * o);
      reverse_w_gemm('N', 'T', v2, o, v, -1.0, bar_w_perm, v2, in.ovoo + (I * v * o + J) * o, o * o,
                   1.0, out.t2 + K * v2, o * v2);
    };

    auto run_canonical_fock_vv = [&](const double* x_ptr, const double* y_ptr, double scale) {
      // Exact identity: sum_{all perms} sum_{bc} (X Y + Y X) = (tile_w/mult) * sum_{axis=0,1,2} (M + M^T).
      // Reuses dead moments[0..2*v3] for axis-1 transpose and dead packed[v2] for M_fvv.
      forward_gemm('T', 'N', v, v, v2, 1.0, y_ptr, v2, x_ptr, v2, 0.0, packed, v);
      forward_gemm('N', 'T', v, v, v2, 1.0, y_ptr, v, x_ptr, v, 1.0, packed, v);
      transpose_01_cube_kernel<<<v3_blocks, 256, 0, context.stream>>>(v, x_ptr, moments);
      transpose_01_cube_kernel<<<v3_blocks, 256, 0, context.stream>>>(v, y_ptr, moments + v3);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      forward_gemm('T', 'N', v, v, v2, 1.0, moments + v3, v2, moments, v2, 1.0, packed, v);
      add_sym_matrix_kernel<<<small_blocks, 256, 0, context.stream>>>(
          v, scale, packed, ptr_f64(fl.fvv), context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      fock.fock_gemms += 3;
    };

    auto run_tile_pullback = [&](std::size_t i, std::size_t j, std::size_t k, bool do_canon_fvv,
                                 const double* x_ptr, const double* y_ptr) {
      const std::array<std::size_t, 3> occupied{i, j, k};
      const double multiplicity = i == k ? 6.0 : (i == j || j == k ? 2.0 : 1.0);
      const double tile_w =
          has_weights ? (occupied_weights[i] + occupied_weights[j] + occupied_weights[k]) / 3.0
                      : 1.0;
      const auto tile = detail::canonical_tile_index(i, j, k);
      if (tile >= tiles) throw std::logic_error("joint triples response tile index overflow");

      if (has_weights && std::abs(tile_w) <= 1e-300) {
        generativeqc_tensor::cuda_check(
            cudaMemsetAsync(ptr_f64(fl.energies) + tile, 0, sizeof(double), context.stream));
        if (occupied_partition) {
          generated_df::energy_tile(o, v, i, j, k, multiplicity, threshold, in, moments, v3_blocks,
                                    ptr_f64(fl.partials), context.error, context.stream);
          ++d.epilogue_kernels;
          reduce<<<1, 256, 0, context.stream>>>(ptr_f64(fl.partials), v3_blocks,
                                                ptr_f64(fl.unweighted_energies) + tile,
                                                context.error);
          generativeqc_tensor::cuda_check(cudaGetLastError());
          ++d.reduction_kernels;
        }
        ++response_tiles;
        return;
      }

      const double effective_multiplicity = has_weights ? multiplicity / tile_w : multiplicity;
      if (fl.use_fused) {
        generated_df::fused_response_tile(o, v, i, j, k, effective_multiplicity, threshold, in,
                                          moments, v3_blocks, bar_w, bar_v, ptr_f64(fl.partials),
                                          context.error, context.stream);
      } else {
        generated_df::energy_tile(o, v, i, j, k, effective_multiplicity, threshold, in, moments,
                                  v3_blocks, ptr_f64(fl.partials), context.error, context.stream);
      }
      ++d.epilogue_kernels;
      reduce<<<1, 256, 0, context.stream>>>(ptr_f64(fl.partials), v3_blocks,
                                            ptr_f64(fl.energies) + tile, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++d.reduction_kernels;
      if (occupied_partition && has_weights) {
        scale_scalar<<<1, 1, 0, context.stream>>>(ptr_f64(fl.energies) + tile, 1.0 / tile_w,
                                                  ptr_f64(fl.unweighted_energies) + tile,
                                                  context.error);
        generativeqc_tensor::cuda_check(cudaGetLastError());
      }

      if (!fl.use_fused) {
        generated_df::response_w_tile(o, v, i, j, k, effective_multiplicity, threshold, in, moments,
                                      v3_blocks, bar_w, context.error, context.stream);
        ++response.reverse_kernels;
      }
      for (unsigned perm = 0; perm < 6; ++perm) {
        const auto* order = generated_df::permutations[perm];
        const auto I = occupied[order[0]], J = occupied[order[1]], K = occupied[order[2]];
        if (!fl.use_fused)
          generated_df::response_v_tile(o, v, i, j, k, perm, effective_multiplicity, threshold, in,
                                        moments, v3_blocks, bar_v, context.error, context.stream);
        ovov_view<false><<<small_blocks, 256, 0, context.stream>>>(o, v, I, J, in.ovov, packed,
                                                                   context.error);
        generativeqc_tensor::cuda_check(cudaGetLastError());
        generated_df::pullback_v(o, v, I, J, K, in, packed, bar_v + (fl.use_fused ? perm * v3 : 0),
                                 packed, out, reverse_gemm);
        ovov_view<true><<<small_blocks, 256, 0, context.stream>>>(o, v, I, J, packed, out.ovov,
                                                                  context.error);
        generativeqc_tensor::cuda_check(cudaGetLastError());
        response.reverse_kernels += fl.use_fused ? 2 : 3;
      }

      if (include_gap_response) {
        generated_df::response_gap_tile(o, v, i, j, k, effective_multiplicity, threshold, in,
                                        moments, v3_blocks, bar_v, context.error, context.stream);
        generated_df::GapCudaState gap{o,
                                       v,
                                       bar_v,
                                       ptr_f64(fl.gap),
                                       context.error,
                                       context.stream,
                                       parallel_gap_reduction};
        const auto gaps = generated_df::gap_response_cuda(gap);
        scatter_gap<<<small_blocks, 256, 0, context.stream>>>(v, i, j, k, gaps, out.eps_o,
                                                              out.eps_v, context.error);
        generativeqc_tensor::cuda_check(cudaGetLastError());
        const auto gap_kernels =
            generated_df::gap_response_kernel_count(o, v, parallel_gap_reduction);
        response.reverse_kernels += 2 + gap_kernels;
        response.gap.kernels += gap_kernels;
      }

      // moments and packed are dead after scalar/gap/pullback_v finish.
      if (do_canon_fvv) run_canonical_fock_vv(x_ptr, y_ptr, tile_w / multiplicity);

      for (std::size_t index = 0; index < 3; ++index) {
        if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
            occupied.begin() + index)
          continue;
        const auto I = occupied[index];
        const auto panel_slot = panel_slot_for(I);
        double* bp_dst = fl.cache_bar_panels ? (bar_panel_buf + I * v3) : bar_panel_buf;
        if (!fl.cache_bar_panels) {
          generativeqc_tensor::cuda_check(
              cudaMemsetAsync(bp_dst, 0, bytes(v3), context.stream));
        }
        for (std::size_t perm = 0; perm < 6; ++perm) {
          const auto* order = generated_df::permutations[perm];
          if (occupied[order[0]] != I) continue;
          run_pullback_w_fp32(I, occupied[order[1]], occupied[order[2]], panel_slot,
                              bar_w + perm * v3, bp_dst);
          d.reverse_w_executed = true;
        }
        if (!fl.cache_bar_panels) {
          generated_df::pullback_panel(o, v, q, I, in, bp_dst, out, reverse_w_gemm);
        }
      }
      ++response_tiles;
    };

    if (canonical_tplus) {
      // Weighted (T+): visit only the o*(o+1)*(o+2)/6 canonical tiles (3x fewer W/resolvent cubes).
      double* x_one = ptr_f64(fl.xl);
      double* y_one = ptr_f64(fl.yl);
      for (std::size_t i = 0; i < o; ++i)
        for (std::size_t j = 0; j <= i; ++j)
          for (std::size_t k = 0; k <= j; ++k) {
            build_six_w_moments(i, j, k, true);
            const double tile_w = (occupied_weights[i] + occupied_weights[j] + occupied_weights[k]) / 3.0;
            if (std::abs(tile_w) > 1e-300) {
              generated_df::resolvent_tile(o, v, i, j, k, threshold, in, moments, v3_blocks, x_one,
                                           y_one, context.error, context.stream);
              ++fock.vector_cubes;
            }
            run_tile_pullback(i, j, k, std::abs(tile_w) > 1e-300, x_one, y_one);
          }
      if (fl.cache_bar_panels && (!compensated_fp32_w || d.reverse_w_executed)) {
        for (std::size_t occ = 0; occ < o; ++occ) {
          generated_df::pullback_panel(o, v, q, occ, in, bar_panel_buf + occ * v3, out,
                                       reverse_w_gemm);
        }
      }
      if (run_deg_foo) {
        const auto deg_count = deg_list.size();
        const auto deg_cap = fl.page_cap;
        const auto deg_page_bytes = bytes(checked_mul(deg_cap, v3));
        const auto deg_scatter_blocks = static_cast<unsigned>(
            std::min<std::size_t>(1 + (deg_cap * deg_cap - 1) / 256, 65535));
        const auto* d_deg_idx = reinterpret_cast<const std::size_t*>(context.arena + fl.deg_indices);
        auto build_deg_page = [&](std::size_t start, std::size_t j, std::size_t k, double* x,
                                  double* y) {
          ++fock.page_builds;
          generativeqc_tensor::cuda_check(cudaMemsetAsync(x, 0, deg_page_bytes, context.stream));
          generativeqc_tensor::cuda_check(cudaMemsetAsync(y, 0, deg_page_bytes, context.stream));
          for (std::size_t lane = 0; lane < deg_cap && start + lane < deg_count; ++lane) {
            const auto p_occ = deg_list[start + lane];
            build_six_w_moments(p_occ, j, k, true);
            generated_df::resolvent_tile(o, v, p_occ, j, k, threshold, in, moments, v3_blocks,
                                         x + lane * v3, y + lane * v3, context.error,
                                         context.stream);
            ++fock.vector_cubes;
          }
        };
        for (std::size_t j = 0; j < o; ++j)
          for (std::size_t k = 0; k <= j; ++k) {
            const double weight = j == k ? 1.0 : 2.0;
            for (std::size_t left = 0; left < deg_count; left += deg_cap) {
              build_deg_page(left, j, k, ptr_f64(fl.xl), ptr_f64(fl.yl));
              for (std::size_t right = left; right < deg_count; right += deg_cap) {
                if (left != right) build_deg_page(right, j, k, ptr_f64(fl.xr), ptr_f64(fl.yr));
                generated_df::fock_oo(v, deg_cap, ptr_f64(fl.xl),
                                      ptr_f64(left == right ? fl.xl : fl.xr), ptr_f64(fl.yl),
                                      ptr_f64(left == right ? fl.yl : fl.yr), weight,
                                      ptr_f64(fl.block), forward_gemm);
                fock.fock_gemms += 2;
                scatter_degenerate_list_fock_block<<<deg_scatter_blocks, 256, 0, context.stream>>>(
                    o, deg_count, deg_cap, left, right, j, k, d_deg_idx, in.eps_o,
                    ptr_f64(fl.device_weights), ptr_f64(fl.block), ptr_f64(fl.foo), context.error);
                generativeqc_tensor::cuda_check(cudaGetLastError());
                ++fock.scatter_kernels;
              }
            }
          }
      }
    } else {
      // Unweighted (T): full paged fock_oo + shared W moments for pullback.
      const auto page_cap = fl.page_cap;
      const auto page_bytes = bytes(checked_mul(page_cap, v3));
      const auto scatter_blocks = static_cast<unsigned>(
          std::min<std::size_t>(1 + (page_cap * page_cap - 1) / 256, 65535));
      auto run_page = [&](std::size_t start, std::size_t j, std::size_t k, double* x, double* y,
                          bool run_response) {
        ++fock.page_builds;
        generativeqc_tensor::cuda_check(cudaMemsetAsync(x, 0, page_bytes, context.stream));
        generativeqc_tensor::cuda_check(cudaMemsetAsync(y, 0, page_bytes, context.stream));
        for (std::size_t lane = 0; lane < page_cap && start + lane < o; ++lane) {
          const auto i = start + lane;
          build_six_w_moments(i, j, k, false);
          generated_df::resolvent_tile(o, v, i, j, k, threshold, in, moments, v3_blocks,
                                       x + lane * v3, y + lane * v3, context.error, context.stream);
          ++fock.vector_cubes;
          if (run_response && i >= j) run_tile_pullback(i, j, k, false, nullptr, nullptr);
        }
      };
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t k = 0; k <= j; ++k) {
          const double weight = j == k ? 1.0 : 2.0;
          for (std::size_t left = 0; left < o; left += page_cap) {
            run_page(left, j, k, ptr_f64(fl.xl), ptr_f64(fl.yl), true);
            for (std::size_t lane = 0; lane < page_cap && left + lane < o; ++lane) {
              generated_df::fock_vv(v, page_cap, ptr_f64(fl.xl) + lane * v3,
                                    ptr_f64(fl.yl) + lane * v3, weight, ptr_f64(fl.fvv),
                                    forward_gemm);
              fock.fock_gemms += 2;
            }
            for (std::size_t right = left; right < o; right += page_cap) {
              if (left != right) run_page(right, j, k, ptr_f64(fl.xr), ptr_f64(fl.yr), false);
              generated_df::fock_oo(v, page_cap, ptr_f64(fl.xl),
                                    ptr_f64(left == right ? fl.xl : fl.xr), ptr_f64(fl.yl),
                                    ptr_f64(left == right ? fl.yl : fl.yr), weight,
                                    ptr_f64(fl.block), forward_gemm);
              fock.fock_gemms += 2;
              scatter_fock_block<<<scatter_blocks, 256, 0, context.stream>>>(
                  o, page_cap, left, right, ptr_f64(fl.block), ptr_f64(fl.foo), context.error);
              generativeqc_tensor::cuda_check(cudaGetLastError());
              ++fock.scatter_kernels;
            }
          }
        }
      if (fl.cache_bar_panels && (!compensated_fp32_w || d.reverse_w_executed)) {
        for (std::size_t occ = 0; occ < o; ++occ) {
          generated_df::pullback_panel(o, v, q, occ, in, bar_panel_buf + occ * v3, out,
                                       reverse_w_gemm);
        }
      }
    }

    if (response_tiles != tiles)
      throw std::logic_error("joint FP32-W DF triples pullback work mismatch");

    reduce<<<1, 256, 0, context.stream>>>(ptr_f64(fl.energies), tiles, ptr_f64(fl.energy),
                                          context.error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    ++d.reduction_kernels;
    if (occupied_partition) {
      const auto* unweighted_tiles =
          has_weights ? ptr_f64(fl.unweighted_energies) : ptr_f64(fl.energies);
      reduce_occupied_energies<<<static_cast<unsigned>(o), 256, 0, context.stream>>>(
          unweighted_tiles, o, ptr_f64(fl.occupied_energies), context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++d.reduction_kernels;
    }
    for (std::size_t x = 0; x < requested_outputs; ++x)
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(
          response_outputs[x]->data(), ptr_f64(fl.outputs[x]), bytes(staging_layout.sizes[x]),
          cudaMemcpyDeviceToHost, context.stream));
    if (occupied_partition)
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(d.occupied_energies.data(),
                                                      ptr_f64(fl.occupied_energies), bytes(o),
                                                      cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&d.energy, ptr_f64(fl.energy), sizeof(double),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(fock.foo.data(), ptr_f64(fl.foo),
                                                    bytes(checked_mul(o, o)),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(fock.fvv.data(), ptr_f64(fl.fvv), bytes(v2),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&failed, context.error, sizeof(int),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    if (failed || !std::isfinite(d.energy))
      throw std::runtime_error("nonfinite or unsafe joint FP32-W DF triples response arithmetic");
    if (has_weights) {
      for (std::size_t idx = 0; idx < o; ++idx) fock.foo[idx * o + idx] = response.eps_o[idx];
    }
    if (occupied_partition) {
      long double weighted_sum = 0.0L, magnitude = 0.0L;
      for (std::size_t idx = 0; idx < o; ++idx) {
        const double val = d.occupied_energies[idx];
        if (!std::isfinite(val))
          throw std::runtime_error("nonfinite occupied triples partition in FP32-W response");
        const long double term =
            has_weights ? static_cast<long double>(occupied_weights[idx]) * val
                        : static_cast<long double>(val);
        weighted_sum += term;
        magnitude += std::abs(term);
      }
      if (std::abs(weighted_sum - d.energy) > 1e-11L * (1.0L + magnitude))
        throw std::runtime_error("combined FP32-W DF triples occupied partition does not close");
    }

    const auto& candidate = generated_df::w_lowering_candidates[requested_w_plan.selected];
    d.resource_fallback = false;
    d.precision = requested_w_plan.precision.arithmetic;
    if (compensated_fp32_w) d.precision.qualification = compensated_w::schedule;
    d.w_contraction_storage_bits = 32;
    d.w_contraction_compute_bits = 32;
    d.w_contraction_accumulation_bits = 32;
    d.w_reverse_contraction_compute_bits = 32;
    d.w_scientific_identity = generated_df::w_lowering_request.scientific_identity;
    d.w_semantic_identity = generated_df::w_lowering_request.semantic_identity;
    d.w_candidate_identity = compensated_fp32_w ? compensated_w::schedule : candidate.identity;
    d.w_provider = candidate.provider;
    d.w_algorithm = compensated_fp32_w ? compensated_w::schedule : candidate.algorithm;
    d.retained_incumbent = !compensated_fp32_w && requested_w_plan.retained_incumbent;
    d.w_provider_version = fp32_provider.provider_version();
    d.cuda_runtime_version = fp32_provider.runtime_version();
    d.compensated_fp32_w_executed = compensated_fp32_w && d.fp32_forward_gemms > 0;
    d.w_precision_identity =
        compensated_fp32_w ? compensated_w::schedule : requested_w_plan.precision.identity;
    d.w_codegen_precision_schedule_identity =
        compensated_fp32_w ? compensated_w::schedule : requested_w_plan.schedule_identity;
    d.host_binding_bytes = 0;
    d.minimum_absolute_denominator = minimum;
    d.virtual_triples = checked_mul(checked_mul(v, checked_add(v, 1)), checked_add(v, 2)) / 6;
    d.occupied_tiles = tiles;
    d.epilogue_points = checked_mul(tiles, v3);
    d.workspace_bytes = fl.total;
    d.arena_bytes = fl.arena;
    d.provider_retained_bytes =
        checked_add(context.metrics.provider_retained_bytes, fp32_provider.retained_bytes());
    d.panel_capacity = fl.panel_cap;
    d.h2d_bytes = prepared.h2d_bytes;
    d.d2h_bytes = checked_add(response_output_bytes, sizeof(double) + sizeof(int));

    auto& fusion = response.scalar_fusion;
    fusion.requested = caller_requested_fused;
    fusion.selected = fl.use_fused;
    fusion.resource_fallback = allow_fused && !fl.use_fused;
    fusion.schedule = fl.use_fused ? generated_df::scalar_fusion_identity : "unfused-primal-w-v";
    fusion.tiles = tiles;
    fusion.kernels = checked_mul(fl.use_fused ? 1 : 8, tiles);
    fusion.avoided_kernels = fl.use_fused ? checked_mul(7, tiles) : 0;
    fusion.workspace_bytes = bytes(checked_mul(fl.use_fused ? 12 : 7, v3));
    const auto points = checked_mul(tiles, v3);
    fusion.value_reads =
        checked_mul(points, fl.use_fused ? generated_df::scalar_fused_value_reads
                                         : generated_df::scalar_unfused_value_reads);
    fusion.arithmetic_ops =
        checked_mul(points, fl.use_fused ? generated_df::scalar_fused_arithmetic_ops
                                         : generated_df::scalar_unfused_arithmetic_ops);
    fusion.value_writes = checked_add(checked_mul(12, points), checked_mul(tiles, v3_blocks));

    response.gap.requested = include_gap_response;
    if (include_gap_response) {
      response.gap.parallel =
          generated_df::gap_response_parallel_selected(v, parallel_gap_reduction);
      response.gap.schedule = response.gap.parallel
                                  ? generated_df::gap_response_parallel_schedule_identity
                                  : "serial-source-major";
      response.gap.workspace_bytes =
          bytes(generated_df::gap_response_arena_elements(o, v, parallel_gap_reduction));
      response.gap.materialized_elements = checked_mul(
          tiles, generated_df::gap_response_materialized_elements(o, v, parallel_gap_reduction));
      response.gap.value_reads = checked_mul(
          tiles, generated_df::gap_response_value_reads(o, v, parallel_gap_reduction));
      response.gap.value_writes = checked_mul(
          tiles, generated_df::gap_response_value_writes(o, v, parallel_gap_reduction));
      response.gap.reduction_summands = checked_mul(
          tiles, generated_df::gap_response_reduction_summands(o, v, parallel_gap_reduction));
    } else {
      response.gap.schedule = generated_df::response_without_gap_identity;
    }

    response.borrowed_host_bytes = host_bytes;
    response.numeric_capacity_bytes = fl.complete;
    response.scalar_response_evaluations =
        checked_mul(include_gap_response ? generated_df::response_scalar_outputs_with_gap
                                         : generated_df::response_scalar_outputs_without_gap,
                    d.epilogue_points);

    fock.minimum_absolute_denominator = minimum;
    fock.numeric_capacity_bytes = fl.complete;
    fock.borrowed_host_bytes = host_bytes;
    fock.workspace_bytes = fl.total;
    fock.arena_bytes = fl.arena;
    fock.provider_retained_bytes = d.provider_retained_bytes;
    fock.page_capacity = fl.page_cap;
    fock.page_count = fl.pages;
    fock.panel_capacity = fl.panel_cap;
    fock.occupied_pairs = pairs;
    fock.unique_vector_cubes = checked_mul(pairs, o);
    fock.scalar_evaluations = checked_mul(fock.vector_cubes, v3);
    fock.h2d_bytes = 0;
    fock.d2h_bytes = fock_output_bytes;
  }

  result.shared_input_device_bytes = prepared.storage_bytes;
  result.shared_h2d_bytes = prepared.h2d_bytes;
  result.shared_response_cubes = response_tiles;
  result.avoided_w_gemms = checked_mul(12, response_tiles);
  result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  d.seconds = result.seconds;
  fock.seconds = 0.0;
  return result;
}
}  // namespace

DFCudaResult evaluate_df_cuda(std::size_t o, std::size_t v, std::size_t q, const double* bov,
                              const double* bvv, const double* ovoo, const double* ovov,
                              const double* fov, const double* t1, const double* t2,
                              const double* eps_o, const double* eps_v, double threshold,
                              std::size_t max_bytes, int device, std::size_t max_panel_buffers,
                              runtime::PrecisionDirective admitted_w, bool occupied_partition,
                              bool compensated_fp32_w) {
  const auto started = Clock::now();
  phase_observation::Scope observed_triples(phase_observation::Site::triples_inclusive, o, v);
  if (!o || !v || !q || !max_bytes || device < 0 || !std::isfinite(threshold) || threshold <= 0 ||
      !max_panel_buffers || max_panel_buffers > 3)
    throw std::invalid_argument("invalid DF triples dimensions, threshold or panel limit");
  bool library_available = true;
#if defined(GENERATIVEQC_TEST_HOOKS)
  library_available = !reject_w_library_for_test;
#endif
  if (compensated_fp32_w && admitted_w.is_strict_fp64())
    throw std::invalid_argument("compensated W requires admitted FP32 contractions");
  auto execution_plan = generated_df::prepare_w_plan(admitted_w, library_available, compensated_fp32_w);
  if (supports_fast_fp32_w(execution_plan, max_panel_buffers)) {
    const std::array<const double*, 9> host{bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v};
    if (auto fast = try_evaluate_df_cuda_fp32_w_fast(o, v, q, host, threshold, max_bytes, device,
                                                     execution_plan, occupied_partition, started,
                                                     compensated_fp32_w)) {
      return std::move(*fast);
    }
  }
  if (compensated_fp32_w)
    throw std::length_error("compensated triples W provider or workspace unavailable within budget");
  const auto occupied_result_bytes = occupied_partition ? bytes(o) : 0;
  const auto host_bytes = [&](const auto& plan) {
    return checked_add(generated_df::WExecution::host_bytes(plan), occupied_result_bytes);
  };
  const auto planned_layout = [&](std::size_t panels) {
    return layout(o, v, q, panels, execution_plan.storage_bytes(o, v, std::min(o, panels)),
                  host_bytes(execution_plan), execution_plan.provider_bytes(), 0, false, 0,
                  occupied_partition);
  };
  auto p = planned_layout(max_panel_buffers);
  if (p.total > max_bytes) p = planned_layout(1);
  // Reaching this owner with mixed W requested means the FP32 panel path was
  // unavailable. Its bounded replacement can still execute FP32 W contractions.
  const bool resource_fallback =
      !admitted_w.is_strict_fp64() || p.total > max_bytes || !library_available;
  while (p.total > max_bytes) {
    const auto fallback = generated_df::lower_resource_w_plan(execution_plan);
    if (!fallback) break;
    execution_plan = *fallback;
    p = planned_layout(max_panel_buffers);
    if (p.total > max_bytes) p = planned_layout(1);
  }
  if (p.total > max_bytes) throw std::length_error("DF triples exceed numeric memory budget");
  // Keep the original panel/provider choices when optional pointer storage
  // cannot fit; paired dispatch is not a reason to reduce those earlier choices.
  const auto unpaired_layout = p;
  if (p.panel_capacity == 3 && execution_plan.pair_pointer_bytes()) {
    const auto paired =
        layout(o, v, q, p.panel_capacity, execution_plan.storage_bytes(o, v, p.panel_capacity),
               host_bytes(execution_plan), execution_plan.provider_bytes(), 0, false,
               execution_plan.pair_pointer_bytes(), occupied_partition);
    if (paired.total <= max_bytes) p = paired;
  }
  const std::array<const double*, 9> host{bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v};
  const double minimum = validate_inputs(o, v, q, p, host, threshold);

  // Borrowed D2H destinations outlive Context's exception-path stream drain.
  DFCudaResult result;
  if (occupied_partition) result.occupied_energies.resize(o);
  result.resource_fallback = resource_fallback;
  int failed = 0;
  {
    runtime::CudaDeviceScope device_scope(device);
    cudaDeviceProp properties{};
    generativeqc_tensor::cuda_check(cudaGetDeviceProperties(&properties, device));
    std::optional<generativeqc_tensor::Context> context_owner;
    const auto prepare_context = [&](const Layout& requested) {
      context_owner.emplace();
      context_owner->prepare(device, properties.major, properties.minor, requested.arena,
                             requested.error, requested.library, 0, 0, false);
    };
    try {
      prepare_context(p);
    } catch (const generativeqc_tensor::DeviceAllocationError&) {
      if (!p.paired_pointer_bytes) throw;
      // A handled runtime OOM remains in this host thread's last-error slot.
      // Clear only that expected failure before retrying the smaller arena;
      // otherwise a later launch check can mistake it for execution failure.
      const auto allocation_error = cudaGetLastError();
      if (allocation_error != cudaSuccess && allocation_error != cudaErrorMemoryAllocation)
        generativeqc_tensor::cuda_check(allocation_error);
      context_owner.reset();
      p = unpaired_layout;
      prepare_context(p);
    }
    auto& context = *context_owner;
    generated_df::WExecution execution(execution_plan, o, v, q, context,
                                       context.arena + p.execution_storage, p.panel_capacity,
                                       result.fp64_gemms, result.fp32_gemms,
                                       result.contraction_summands, result.precision_cast_elements);
    const auto& selected = execution.plan();
    result.host_binding_bytes = host_bytes(selected);
    result.resource_fallback =
        result.resource_fallback || selected.selected != execution_plan.selected;
    result.precision = selected.precision.arithmetic;
    result.w_contraction_storage_bits =
        result.precision.storage_dtype == runtime::PrecisionDtype::Fp64 ? 64 : 32;
    result.w_contraction_compute_bits =
        result.precision.compute_dtype == runtime::PrecisionDtype::Fp64 ? 64 : 32;
    result.w_contraction_accumulation_bits =
        result.precision.accumulation_dtype == runtime::PrecisionDtype::Fp64 ? 64 : 32;
    const auto& candidate = generated_df::w_lowering_candidates[selected.selected];
    result.w_scientific_identity = generated_df::w_lowering_request.scientific_identity;
    result.w_semantic_identity = generated_df::w_lowering_request.semantic_identity;
    result.w_candidate_identity = candidate.identity;
    result.w_provider = candidate.provider;
    result.w_algorithm = candidate.algorithm;
    result.retained_incumbent = selected.retained_incumbent;
    result.w_provider_version = execution.provider_version();
    result.cuda_runtime_version = execution.runtime_version();
    result.w_precision_identity = selected.precision.identity;
    result.w_codegen_precision_schedule_identity = selected.schedule_identity;
    if (phase_observation::detailed_enabled())
      phase_observation::admitted_triples(
          o, v, q, p.panel_capacity, p.paired_pointer_bytes, selected.selected,
          result.w_contraction_storage_bits, result.w_contraction_compute_bits,
          result.w_contraction_accumulation_bits, max_bytes, p.total);
    generated_df::Inputs in;
    const std::array<const double**, 9> fields{&in.bov, &in.bvv, &in.ovoo,  &in.ovov, &in.fov,
                                               &in.t1,  &in.t2,  &in.eps_o, &in.eps_v};
    for (std::size_t x = 0; x < host.size(); ++x) {
      *fields[x] = reinterpret_cast<const double*>(context.arena + p.inputs[x]);
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(context.arena + p.inputs[x], host[x],
                                                      bytes(p.sizes[x]), cudaMemcpyHostToDevice,
                                                      context.stream));
      result.h2d_bytes = checked_add(result.h2d_bytes, bytes(p.sizes[x]));
    }
    generativeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    auto* panels = reinterpret_cast<double*>(context.arena + p.panels);
    auto* moments = reinterpret_cast<double*>(context.arena + p.moments);
    auto* partials = reinterpret_cast<double*>(context.arena + p.partials);
    auto* energies = reinterpret_cast<double*>(context.arena + p.energies);
    auto* energy = reinterpret_cast<double*>(context.arena + p.energy);
    execution.initialize(context, in);
    const bool distinct_moments = result.w_contraction_storage_bits == 64 &&
                                  result.w_contraction_compute_bits == 64 &&
                                  result.w_contraction_accumulation_bits == 64;
    std::array<std::size_t, 3> identities;
    identities.fill(std::numeric_limits<std::size_t>::max());
    std::array<std::size_t, 3> ages{};
    phase_observation::PanelSources panel_source_ids;
    std::size_t epoch = 0, tile = 0;
    auto panel_slot_for = [&](std::size_t occupied) {
      phase_observation::Scope host(phase_observation::Site::triples_host_preparation, occupied,
                                    p.panel_capacity);
      std::size_t slot = 0;
      for (; slot < p.panel_capacity; ++slot)
        if (identities[slot] == occupied) break;
      if (slot == p.panel_capacity) {
        slot = static_cast<std::size_t>(
            std::min_element(ages.begin(), ages.begin() + p.panel_capacity) - ages.begin());
        host.close();
        phase_observation::Scope panel(phase_observation::Site::triples_panel_submission, occupied,
                                       slot, reinterpret_cast<std::uintptr_t>(context.stream));
        execution.build_panel(context, in, occupied, panels + slot * p.v3, slot);
        panel.close();
        if (phase_observation::detailed_enabled())
          panel_source_ids.remember(slot, panel.identity());
        ++result.panel_gemms;
        identities[slot] = occupied;
      }
      ages[slot] = ++epoch;
      return slot;
    };
    auto visit_tile = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t canonical_tile) {
      phase_observation::Scope observed_tile(phase_observation::Site::triples_tile_inclusive,
                                             canonical_tile, (i * o + j) * o + k,
                                             reinterpret_cast<std::uintptr_t>(context.stream));
      phase_observation::Scope host(phase_observation::Site::triples_host_preparation,
                                    canonical_tile, 0);
      const std::array<std::size_t, 3> occupied{i, j, k};
      const bool alias_tile = distinct_moments && (i == j || j == k);
      const auto sources = alias_tile ? generated_df::occupied_moment_sources(i, j, k)
                                      : generated_df::MomentSourceMap{};
      host.close();
      // Group W seeds by their integral-panel index. A single-panel fallback
      // consumes every dependent GEMM before that storage is reused. All
      // producer/consumer work is ordered on Context's one owned stream.
      for (std::size_t index = 0; index < occupied.size(); ++index) {
        if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
            occupied.begin() + index)
          continue;
        const auto slot = panel_slot_for(occupied[index]);
        const auto* panel = panels + slot * p.v3;
        // Equal-index non-FP64 tiles still require all six original seeds.
        std::array<std::size_t, 6> seed_permutations{};
        phase_observation::Scope seed_preparation(phase_observation::Site::triples_seed_preparation,
                                                  canonical_tile, occupied[index]);
        std::size_t seed_count = 0;
        for (std::size_t permutation = 0; permutation < 6; ++permutation) {
          const auto* order = generated_df::permutations[permutation];
          if (occupied[order[0]] != occupied[index]) continue;
          // Equal physical tuples have identical W seeds; every original
          // energy contribution still loads its canonical stored seed.
          if (sources.index[permutation] != permutation) continue;
          seed_permutations[seed_count++] = permutation;
        }
        seed_preparation.close();
        if (seed_count == 2 && p.paired_pointer_bytes) {
          const auto* first_order = generated_df::permutations[seed_permutations[0]];
          const auto* second_order = generated_df::permutations[seed_permutations[1]];
          const bool swapped = occupied[second_order[0]] == occupied[first_order[0]] &&
                               occupied[second_order[1]] == occupied[first_order[2]] &&
                               occupied[second_order[2]] == occupied[first_order[1]];
          if (swapped && [&] {
                phase_observation::Scope pair(
                    phase_observation::Site::triples_pair_submission,
                    seed_permutations[0] * 6 + seed_permutations[1], 2,
                    reinterpret_cast<std::uintptr_t>(context.stream),
                    phase_observation::detailed_enabled() ? panel_source_ids.dependency(slot) : 0);
                const bool accepted = execution.build_w_pair(
                    context, in, occupied[first_order[0]], occupied[first_order[1]],
                    occupied[first_order[2]], panel, moments + seed_permutations[0] * p.v3,
                    moments + seed_permutations[1] * p.v3, context.arena + p.paired_pointers);
                pair.close(accepted ? 0 : 2);
                return accepted;
              }()) {
            result.moment_gemms += 4;
            continue;
          }
        }
        for (std::size_t seed = 0; seed < seed_count; ++seed) {
          const auto permutation = seed_permutations[seed];
          const auto* order = generated_df::permutations[permutation];
          phase_observation::Scope moment(
              phase_observation::Site::triples_moment_submission, permutation, 1,
              reinterpret_cast<std::uintptr_t>(context.stream),
              phase_observation::detailed_enabled() ? panel_source_ids.dependency(slot) : 0);
          execution.build_w(context, in, occupied[order[0]], occupied[order[1]], occupied[order[2]],
                            panel, moments + permutation * p.v3, slot);
          result.moment_gemms += 2;
        }
      }
      const double degeneracy = i == k ? 6.0 : (i == j || j == k ? 2.0 : 1.0);
      phase_observation::Scope epilogue(phase_observation::Site::triples_epilogue_submission,
                                        canonical_tile, 1,
                                        reinterpret_cast<std::uintptr_t>(context.stream));
      if (alias_tile)
        generated_df::energy_distinct_tile(o, v, i, j, k, degeneracy, threshold, in, moments,
                                           sources, p.blocks, partials, context.error,
                                           context.stream);
      else
        generated_df::energy_tile(o, v, i, j, k, degeneracy, threshold, in, moments, p.blocks,
                                  partials, context.error, context.stream);
      ++result.epilogue_kernels;
      epilogue.close();
      phase_observation::Scope reduction(phase_observation::Site::triples_reduction_submission,
                                         canonical_tile, 1,
                                         reinterpret_cast<std::uintptr_t>(context.stream));
      reduce<<<1, 256, 0, context.stream>>>(partials, p.blocks, energies + canonical_tile,
                                            context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++result.reduction_kernels;
      ++tile;
    };
    // Only the qualified three-panel FP64 domain changes its visit order;
    // canonical tile addresses preserve the original final reduction inputs.
    detail::visit_occupied_tiles(o, distinct_moments && p.panel_capacity == 3, visit_tile);
    if (tile != p.tiles) throw std::logic_error("DF triples occupied work mismatch");
    phase_observation::Scope final_reduction(phase_observation::Site::triples_reduction_submission,
                                             p.tiles, 1,
                                             reinterpret_cast<std::uintptr_t>(context.stream));
    reduce<<<1, 256, 0, context.stream>>>(energies, p.tiles, energy, context.error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    ++result.reduction_kernels;
    final_reduction.close();
    auto* occupied_energies = reinterpret_cast<double*>(context.arena + p.occupied_energies);
    if (occupied_partition) {
      phase_observation::Scope occupied_reduction(
          phase_observation::Site::triples_reduction_submission, p.tiles, o,
          reinterpret_cast<std::uintptr_t>(context.stream));
      reduce_occupied_energies<<<static_cast<unsigned>(o), 256, 0, context.stream>>>(
          energies, o, occupied_energies, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++result.reduction_kernels;
    }
    result.d2h_bytes = checked_add(sizeof(double) + sizeof(int), occupied_result_bytes);
    phase_observation::Scope download(phase_observation::Site::triples_download_submission, 0,
                                      result.d2h_bytes,
                                      reinterpret_cast<std::uintptr_t>(context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&result.energy, energy, sizeof(double),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&failed, context.error, sizeof(int),
                                                    cudaMemcpyDeviceToHost, context.stream));
    if (occupied_partition)
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(
          result.occupied_energies.data(), occupied_energies, occupied_result_bytes,
          cudaMemcpyDeviceToHost, context.stream));
    download.close();
    {
      phase_observation::Scope wait(phase_observation::Site::explicit_wait, 0, 0,
                                    reinterpret_cast<std::uintptr_t>(context.stream));
      generativeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    }
    if (failed || !std::isfinite(result.energy))
      throw std::runtime_error("nonfinite or unsafe generated DF triples arithmetic");
    result.fp32_forward_gemms = result.fp32_gemms;
    result.fp32_reverse_gemms = 0;
    result.w_reverse_contraction_compute_bits = 64;
    result.minimum_absolute_denominator = minimum;
    result.virtual_triples = checked_mul(checked_mul(v, checked_add(v, 1)), checked_add(v, 2)) / 6;
    result.occupied_tiles = p.tiles;
    result.epilogue_points = checked_mul(p.tiles, p.v3);
    result.workspace_bytes = p.total;
    result.arena_bytes = p.arena;
    result.provider_retained_bytes = execution.provider_bytes();
    result.panel_capacity = p.panel_capacity;
  }  // Drain/destroy the stream, buffers and BLAS provider inside endpoint timing.
  result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}
static DFCudaResponseResult pullback_df_cuda_impl(
    std::size_t o, std::size_t v, std::size_t q, const double* bov, const double* bvv,
    const double* ovoo, const double* ovov, const double* fov, const double* t1, const double* t2,
    const double* eps_o, const double* eps_v, double threshold, std::size_t max_bytes, int device,
    std::size_t caller_bytes, std::size_t max_panel_buffers, bool parallel_gap_reduction,
    bool include_gap_response, const generated_df::Inputs* prepared_inputs) {
  const auto started = Clock::now();
  if (!o || !v || !q || !max_bytes || device < 0 || !std::isfinite(threshold) || threshold <= 0 ||
      !max_panel_buffers || max_panel_buffers > 3)
    throw std::invalid_argument("invalid DF triples response dimensions, threshold or panel limit");
  auto r = response_layout(o, v, q, max_panel_buffers, caller_bytes, parallel_gap_reduction,
                           include_gap_response, prepared_inputs != nullptr);
  if (r.complete > max_bytes)
    r = response_layout(o, v, q, 1, caller_bytes, parallel_gap_reduction, include_gap_response,
                        prepared_inputs != nullptr);
  if (r.complete > max_bytes)
    throw std::length_error("DF triples response exceeds complete numeric budget");
  const auto& p = r.value;
  const std::array<const double*, 9> host{bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v};
  const double minimum = validate_inputs(o, v, q, p, host, threshold);
  // Detached output destinations outlive Context and its exception-path drain.
  DFCudaResponseResult result;
  auto& d = result.diagnostic;
  const std::array<std::vector<double>*, 9> outputs{&result.bov,  &result.bvv,   &result.ovoo,
                                                    &result.ovov, &result.fov,   &result.t1,
                                                    &result.t2,   &result.eps_o, &result.eps_v};
  const std::size_t requested_outputs = include_gap_response ? 9 : 7;
  for (std::size_t x = 0; x < requested_outputs; ++x) outputs[x]->resize(p.sizes[x]);
  int failed = 0;
  {
    runtime::CudaDeviceScope device_scope(device);
    generativeqc_tensor::Context context;
    cudaDeviceProp properties{};
    generativeqc_tensor::cuda_check(cudaGetDeviceProperties(&properties, device));
    context.prepare(device, properties.major, properties.minor, p.arena, p.error, p.library,
                    blas_workspace, provider_allowance, true);
    generated_df::Inputs in;
    generated_df::ResponseOutputs out;
    const std::array<const double**, 9> input_fields{&in.bov, &in.bvv, &in.ovoo,  &in.ovov, &in.fov,
                                                     &in.t1,  &in.t2,  &in.eps_o, &in.eps_v};
    const std::array<double**, 9> output_fields{&out.bov, &out.bvv, &out.ovoo,  &out.ovov, &out.fov,
                                                &out.t1,  &out.t2,  &out.eps_o, &out.eps_v};
    const std::array<const double*, 9> prepared{prepared_inputs ? prepared_inputs->bov : nullptr,
                                                prepared_inputs ? prepared_inputs->bvv : nullptr,
                                                prepared_inputs ? prepared_inputs->ovoo : nullptr,
                                                prepared_inputs ? prepared_inputs->ovov : nullptr,
                                                prepared_inputs ? prepared_inputs->fov : nullptr,
                                                prepared_inputs ? prepared_inputs->t1 : nullptr,
                                                prepared_inputs ? prepared_inputs->t2 : nullptr,
                                                prepared_inputs ? prepared_inputs->eps_o : nullptr,
                                                prepared_inputs ? prepared_inputs->eps_v : nullptr};
    auto pointer = [&](std::size_t offset) {
      return reinterpret_cast<double*>(context.arena + offset);
    };
    for (std::size_t x = 0; x < 9; ++x) {
      if (prepared_inputs) {
        *input_fields[x] = prepared[x];
      } else {
        *input_fields[x] = pointer(p.inputs[x]);
        generativeqc_tensor::cuda_check(cudaMemcpyAsync(pointer(p.inputs[x]), host[x],
                                                        bytes(p.sizes[x]), cudaMemcpyHostToDevice,
                                                        context.stream));
        d.h2d_bytes = checked_add(d.h2d_bytes, bytes(p.sizes[x]));
      }
      *output_fields[x] = x < requested_outputs ? pointer(r.outputs[x]) : nullptr;
      if (x < requested_outputs)
        generativeqc_tensor::cuda_check(
            cudaMemsetAsync(pointer(r.outputs[x]), 0, bytes(p.sizes[x]), context.stream));
    }
    generativeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    auto* panels = pointer(p.panels);
    auto* moments = pointer(p.moments);
    auto* bar_w = pointer(r.bar_w);
    auto* bar_v = pointer(r.bar_v);
    auto* bar_panel = pointer(r.bar_panel);
    auto* packed = pointer(r.packed);
    auto gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t kk, double alpha,
                    const double* a, std::size_t lda, const double* b, std::size_t ldb, double beta,
                    double* c, std::size_t ldc) {
      execute_strided_fp64_product(context, ta, tb, m, n, kk, alpha, a, lda, b, ldb, beta, c, ldc);
      const auto count = checked_mul(m, n);
      const auto blocks =
          static_cast<unsigned>(std::min<std::size_t>(1 + (count - 1) / 256, 65535));
      audit_matrix<<<blocks, 256, 0, context.stream>>>(c, m, n, ldc, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++result.audit_kernels;
      d.contraction_summands = checked_add(d.contraction_summands, checked_mul(count, kk));
    };
    auto reverse_gemm = [&](auto... args) {
      gemm(args...);
      ++result.reverse_gemms;
    };
    std::array<std::size_t, 3> identities;
    identities.fill(std::numeric_limits<std::size_t>::max());
    std::array<std::size_t, 3> ages{};
    std::size_t epoch = 0, tile = 0;
    auto panel_for = [&](std::size_t occupied) {
      std::size_t slot = 0;
      for (; slot < p.panel_capacity; ++slot)
        if (identities[slot] == occupied) break;
      if (slot == p.panel_capacity) {
        slot = static_cast<std::size_t>(
            std::min_element(ages.begin(), ages.begin() + p.panel_capacity) - ages.begin());
        generated_df::build_panel(o, v, q, occupied, in, panels + slot * p.v3, gemm);
        ++d.panel_gemms;
        identities[slot] = occupied;
      }
      ages[slot] = ++epoch;
      return panels + slot * p.v3;
    };
    const auto small_blocks =
        static_cast<unsigned>(std::min<std::size_t>(1 + (v * v - 1) / 256, 65535));
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j <= i; ++j)
        for (std::size_t k = 0; k <= j; ++k) {
          const std::array<std::size_t, 3> occupied{i, j, k};
          for (std::size_t index = 0; index < 3; ++index) {
            if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
                occupied.begin() + index)
              continue;
            const auto* panel = panel_for(occupied[index]);
            for (std::size_t perm = 0; perm < 6; ++perm) {
              const auto* order = generated_df::permutations[perm];
              if (occupied[order[0]] != occupied[index]) continue;
              generated_df::build_w(o, v, occupied[order[0]], occupied[order[1]],
                                    occupied[order[2]], in, panel, moments + perm * p.v3, gemm);
              d.moment_gemms += 2;
            }
          }
          const double multiplicity = i == k ? 6.0 : (i == j || j == k ? 2.0 : 1.0);
          generated_df::energy_tile(o, v, i, j, k, multiplicity, threshold, in, moments, p.blocks,
                                    pointer(p.partials), context.error, context.stream);
          ++d.epilogue_kernels;
          reduce<<<1, 256, 0, context.stream>>>(pointer(p.partials), p.blocks,
                                                pointer(p.energies) + tile, context.error);
          generativeqc_tensor::cuda_check(cudaGetLastError());
          ++d.reduction_kernels;
          generated_df::response_w_tile(o, v, i, j, k, multiplicity, threshold, in, moments,
                                        p.blocks, bar_w, context.error, context.stream);
          ++result.reverse_kernels;
          for (unsigned perm = 0; perm < 6; ++perm) {
            const auto* order = generated_df::permutations[perm];
            const auto I = occupied[order[0]], J = occupied[order[1]], K = occupied[order[2]];
            generated_df::response_v_tile(o, v, i, j, k, perm, multiplicity, threshold, in, moments,
                                          p.blocks, bar_v, context.error, context.stream);
            ovov_view<false><<<small_blocks, 256, 0, context.stream>>>(o, v, I, J, in.ovov, packed,
                                                                       context.error);
            generativeqc_tensor::cuda_check(cudaGetLastError());
            // Generated V reverse consumes packed ovov in its first product and
            // overwrites it only in its last product. One stream preserves this reuse.
            generated_df::pullback_v(o, v, I, J, K, in, packed, bar_v, packed, out, reverse_gemm);
            ovov_view<true><<<small_blocks, 256, 0, context.stream>>>(o, v, I, J, packed, out.ovov,
                                                                      context.error);
            generativeqc_tensor::cuda_check(cudaGetLastError());
            result.reverse_kernels += 3;
          }
          if (include_gap_response) {
            generated_df::response_gap_tile(o, v, i, j, k, multiplicity, threshold, in, moments,
                                            p.blocks, bar_v, context.error, context.stream);
            generated_df::GapCudaState gap{
                o, v, bar_v, pointer(r.gap), context.error, context.stream, parallel_gap_reduction};
            const auto gaps = generated_df::gap_response_cuda(gap);
            scatter_gap<<<small_blocks, 256, 0, context.stream>>>(v, i, j, k, gaps, out.eps_o,
                                                                  out.eps_v, context.error);
            generativeqc_tensor::cuda_check(cudaGetLastError());
            const auto gap_kernels =
                generated_df::gap_response_kernel_count(o, v, parallel_gap_reduction);
            result.reverse_kernels += 2 + gap_kernels;
            result.gap.kernels += gap_kernels;
          }
          for (std::size_t index = 0; index < 3; ++index) {
            if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
                occupied.begin() + index)
              continue;
            const auto I = occupied[index];
            const auto* panel = panel_for(I);
            generativeqc_tensor::cuda_check(
                cudaMemsetAsync(bar_panel, 0, bytes(p.v3), context.stream));
            for (std::size_t perm = 0; perm < 6; ++perm) {
              const auto* order = generated_df::permutations[perm];
              if (occupied[order[0]] != I) continue;
              generated_df::pullback_w(o, v, I, occupied[order[1]], occupied[order[2]], in, panel,
                                       bar_w + perm * p.v3, bar_panel, out, reverse_gemm);
              d.reverse_w_executed = true;
            }
            generated_df::pullback_panel(o, v, q, I, in, bar_panel, out, reverse_gemm);
          }
          ++tile;
        }
    if (tile != p.tiles) throw std::logic_error("DF triples response occupied work mismatch");
    result.gap.requested = include_gap_response;
    if (include_gap_response) {
      result.gap.parallel = generated_df::gap_response_parallel_selected(v, parallel_gap_reduction);
      result.gap.schedule = result.gap.parallel
                                ? generated_df::gap_response_parallel_schedule_identity
                                : "serial-source-major";
      result.gap.workspace_bytes =
          bytes(generated_df::gap_response_arena_elements(o, v, parallel_gap_reduction));
      result.gap.materialized_elements = checked_mul(
          p.tiles, generated_df::gap_response_materialized_elements(o, v, parallel_gap_reduction));
      result.gap.value_reads = checked_mul(
          p.tiles, generated_df::gap_response_value_reads(o, v, parallel_gap_reduction));
      result.gap.value_writes = checked_mul(
          p.tiles, generated_df::gap_response_value_writes(o, v, parallel_gap_reduction));
      result.gap.reduction_summands = checked_mul(
          p.tiles, generated_df::gap_response_reduction_summands(o, v, parallel_gap_reduction));
    } else {
      result.gap.schedule = generated_df::response_without_gap_identity;
    }
    reduce<<<1, 256, 0, context.stream>>>(pointer(p.energies), p.tiles, pointer(p.energy),
                                          context.error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    ++d.reduction_kernels;
    for (std::size_t x = 0; x < requested_outputs; ++x)
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(outputs[x]->data(), pointer(r.outputs[x]),
                                                      bytes(p.sizes[x]), cudaMemcpyDeviceToHost,
                                                      context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&d.energy, pointer(p.energy), sizeof(double),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&failed, context.error, sizeof(int),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    if (failed || !std::isfinite(d.energy))
      throw std::runtime_error("nonfinite or unsafe generated DF triples response arithmetic");
    d.minimum_absolute_denominator = minimum;
    d.virtual_triples = checked_mul(checked_mul(v, checked_add(v, 1)), checked_add(v, 2)) / 6;
    d.occupied_tiles = p.tiles;
    d.workspace_bytes = p.total;
    d.arena_bytes = p.arena;
    d.provider_retained_bytes = context.metrics.provider_retained_bytes;
    d.panel_capacity = p.panel_capacity;
    d.epilogue_points = checked_mul(p.tiles, p.v3);
    d.d2h_bytes = checked_add(r.output_bytes, sizeof(double) + sizeof(int));
    result.borrowed_host_bytes = r.host_bytes;
    result.numeric_capacity_bytes = r.complete;
    result.scalar_response_evaluations =
        checked_mul(include_gap_response ? generated_df::response_scalar_outputs_with_gap
                                         : generated_df::response_scalar_outputs_without_gap,
                    d.epilogue_points);
  }
  d.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}

DFCudaResponseResult pullback_df_cuda(std::size_t o, std::size_t v, std::size_t q,
                                      const double* bov, const double* bvv, const double* ovoo,
                                      const double* ovov, const double* fov, const double* t1,
                                      const double* t2, const double* eps_o, const double* eps_v,
                                      double threshold, std::size_t max_bytes, int device,
                                      std::size_t caller_bytes, std::size_t max_panel_buffers,
                                      bool parallel_gap_reduction, bool include_gap_response) {
  return pullback_df_cuda_impl(o, v, q, bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v, threshold,
                               max_bytes, device, caller_bytes, max_panel_buffers,
                               parallel_gap_reduction, include_gap_response, nullptr);
}

static DFCudaFockResult fock_response_df_cuda_impl(
    std::size_t o, std::size_t v, std::size_t q, const double* bov, const double* bvv,
    const double* ovoo, const double* ovov, const double* fov, const double* t1, const double* t2,
    const double* eps_o, const double* eps_v, double threshold, std::size_t max_bytes, int device,
    std::size_t caller_bytes, std::size_t max_page_rows, std::size_t max_panel_buffers,
    const generated_df::Inputs* prepared_inputs) {
  const auto started = Clock::now();
  if (!o || !v || !q || !max_bytes || device < 0 || !std::isfinite(threshold) || threshold <= 0 ||
      !max_panel_buffers || max_panel_buffers > 3)
    throw std::invalid_argument("invalid DF triples Fock dimensions, threshold or panel limit");
  auto capacity = max_page_rows ? std::min(o, max_page_rows) : o;
  auto r =
      fock_layout(o, v, q, capacity, max_panel_buffers, caller_bytes, prepared_inputs != nullptr);
  while (r.complete > max_bytes) {
    // Prefer retaining more resolvent rows over extra cached integral panels:
    // a full page eliminates every cross-occupied recomputation.
    if (r.value.panel_capacity > 1) {
      r = fock_layout(o, v, q, capacity, 1, caller_bytes, prepared_inputs != nullptr);
      if (r.complete <= max_bytes) break;
    }
    if (capacity == 1)
      throw std::length_error("DF triples Fock response exceeds complete numeric budget");
    --capacity;
    r = fock_layout(o, v, q, capacity, max_panel_buffers, caller_bytes, prepared_inputs != nullptr);
  }
  const auto& p = r.value;
  const std::array<const double*, 9> host{bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v};
  const double minimum = validate_inputs(o, v, q, p, host, threshold);
  // Detached destinations and error status outlive the device owner's drain.
  DFCudaFockResult result;
  result.foo.resize(o * o);
  result.fvv.resize(v * v);
  int failed = 0;
  {
    runtime::CudaDeviceScope device_scope(device);
    generativeqc_tensor::Context context;
    cudaDeviceProp properties{};
    generativeqc_tensor::cuda_check(cudaGetDeviceProperties(&properties, device));
    context.prepare(device, properties.major, properties.minor, p.arena, p.error, p.library,
                    blas_workspace, provider_allowance, true);
    auto pointer = [&](std::size_t offset) {
      return reinterpret_cast<double*>(context.arena + offset);
    };
    generated_df::Inputs in;
    const std::array<const double**, 9> fields{&in.bov, &in.bvv, &in.ovoo,  &in.ovov, &in.fov,
                                               &in.t1,  &in.t2,  &in.eps_o, &in.eps_v};
    const std::array<const double*, 9> prepared{prepared_inputs ? prepared_inputs->bov : nullptr,
                                                prepared_inputs ? prepared_inputs->bvv : nullptr,
                                                prepared_inputs ? prepared_inputs->ovoo : nullptr,
                                                prepared_inputs ? prepared_inputs->ovov : nullptr,
                                                prepared_inputs ? prepared_inputs->fov : nullptr,
                                                prepared_inputs ? prepared_inputs->t1 : nullptr,
                                                prepared_inputs ? prepared_inputs->t2 : nullptr,
                                                prepared_inputs ? prepared_inputs->eps_o : nullptr,
                                                prepared_inputs ? prepared_inputs->eps_v : nullptr};
    for (std::size_t x = 0; x < 9; ++x) {
      if (prepared_inputs) {
        *fields[x] = prepared[x];
      } else {
        *fields[x] = pointer(p.inputs[x]);
        generativeqc_tensor::cuda_check(cudaMemcpyAsync(pointer(p.inputs[x]), host[x],
                                                        bytes(p.sizes[x]), cudaMemcpyHostToDevice,
                                                        context.stream));
      }
    }
    generativeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    generativeqc_tensor::cuda_check(
        cudaMemsetAsync(pointer(r.foo), 0, bytes(o * o), context.stream));
    generativeqc_tensor::cuda_check(
        cudaMemsetAsync(pointer(r.fvv), 0, bytes(v * v), context.stream));
    auto gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t kk, double alpha,
                    const double* a, std::size_t lda, const double* b, std::size_t ldb, double beta,
                    double* c, std::size_t ldc) {
      execute_strided_fp64_product(context, ta, tb, m, n, kk, alpha, a, lda, b, ldb, beta, c, ldc);
      const auto count = checked_mul(m, n);
      const auto blocks =
          static_cast<unsigned>(std::min<std::size_t>(1 + (count - 1) / 256, 65535));
      audit_matrix<<<blocks, 256, 0, context.stream>>>(c, m, n, ldc, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++result.audit_kernels;
      result.contraction_summands =
          checked_add(result.contraction_summands, checked_mul(count, kk));
    };
    std::array<std::size_t, 3> identities;
    identities.fill(std::numeric_limits<std::size_t>::max());
    std::array<std::size_t, 3> ages{};
    std::size_t epoch = 0;
    auto panel_for = [&](std::size_t occupied) {
      std::size_t slot = 0;
      for (; slot < p.panel_capacity; ++slot)
        if (identities[slot] == occupied) break;
      if (slot == p.panel_capacity) {
        slot = static_cast<std::size_t>(
            std::min_element(ages.begin(), ages.begin() + p.panel_capacity) - ages.begin());
        generated_df::build_panel(o, v, q, occupied, in, pointer(p.panels) + slot * p.v3, gemm);
        ++result.panel_gemms;
        identities[slot] = occupied;
      }
      ages[slot] = ++epoch;
      return pointer(p.panels) + slot * p.v3;
    };
    auto page = [&](std::size_t start, std::size_t j, std::size_t k, double* x, double* y) {
      ++result.page_builds;
      const auto page_bytes = bytes(checked_mul(capacity, p.v3));
      // Explicit padding protects oo tail products even when the right buffer
      // was previously used for a full page in a different (j,k) pair.
      generativeqc_tensor::cuda_check(cudaMemsetAsync(x, 0, page_bytes, context.stream));
      generativeqc_tensor::cuda_check(cudaMemsetAsync(y, 0, page_bytes, context.stream));
      for (std::size_t lane = 0; lane < capacity && start + lane < o; ++lane) {
        const auto i = start + lane;
        const std::array<std::size_t, 3> occupied{i, j, k};
        for (std::size_t index = 0; index < 3; ++index) {
          if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
              occupied.begin() + index)
            continue;
          const auto* panel = panel_for(occupied[index]);
          for (std::size_t perm = 0; perm < 6; ++perm) {
            const auto* order = generated_df::permutations[perm];
            if (occupied[order[0]] != occupied[index]) continue;
            generated_df::build_w(o, v, occupied[order[0]], occupied[order[1]], occupied[order[2]],
                                  in, panel, pointer(p.moments) + perm * p.v3, gemm);
            result.w_gemms += 2;
          }
        }
        generated_df::resolvent_tile(o, v, i, j, k, threshold, in, pointer(p.moments), p.blocks,
                                     x + lane * p.v3, y + lane * p.v3, context.error,
                                     context.stream);
        ++result.vector_cubes;
      }
    };
    const auto scatter_blocks =
        static_cast<unsigned>(std::min<std::size_t>(1 + (capacity * capacity - 1) / 256, 65535));
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t k = 0; k <= j; ++k) {
        const double weight = j == k ? 1.0 : 2.0;
        for (std::size_t left = 0; left < o; left += capacity) {
          page(left, j, k, pointer(r.xl), pointer(r.yl));
          for (std::size_t lane = 0; lane < capacity && left + lane < o; ++lane) {
            generated_df::fock_vv(v, capacity, pointer(r.xl) + lane * p.v3,
                                  pointer(r.yl) + lane * p.v3, weight, pointer(r.fvv), gemm);
            result.fock_gemms += 2;
          }
          for (std::size_t right = left; right < o; right += capacity) {
            if (left != right) page(right, j, k, pointer(r.xr), pointer(r.yr));
            generated_df::fock_oo(v, capacity, pointer(r.xl), pointer(left == right ? r.xl : r.xr),
                                  pointer(r.yl), pointer(left == right ? r.yl : r.yr), weight,
                                  pointer(r.block), gemm);
            result.fock_gemms += 2;
            scatter_fock_block<<<scatter_blocks, 256, 0, context.stream>>>(
                o, capacity, left, right, pointer(r.block), pointer(r.foo), context.error);
            generativeqc_tensor::cuda_check(cudaGetLastError());
            ++result.scatter_kernels;
          }
        }
      }
    if (result.vector_cubes != r.cubes || result.page_builds != r.page_builds)
      throw std::logic_error("DF triples Fock replay work mismatch");
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(result.foo.data(), pointer(r.foo), bytes(o * o),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(result.fvv.data(), pointer(r.fvv), bytes(v * v),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&failed, context.error, sizeof(int),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    if (failed)
      throw std::runtime_error("nonfinite or unsafe generated DF triples Fock arithmetic");
    result.provider_retained_bytes = context.metrics.provider_retained_bytes;
  }
  result.minimum_absolute_denominator = minimum;
  result.numeric_capacity_bytes = r.complete;
  result.borrowed_host_bytes = r.host_bytes;
  result.workspace_bytes = p.total;
  result.arena_bytes = p.arena;
  result.page_capacity = capacity;
  result.page_count = r.pages;
  result.panel_capacity = p.panel_capacity;
  result.occupied_pairs = r.pairs;
  result.unique_vector_cubes = checked_mul(r.pairs, o);
  result.scalar_evaluations = checked_mul(r.cubes, p.v3);
  result.h2d_bytes = prepared_inputs ? 0 : r.host_bytes;
  result.d2h_bytes = checked_add(r.output_bytes, sizeof(int));
  result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}

DFCudaFockResult fock_response_df_cuda(std::size_t o, std::size_t v, std::size_t q,
                                       const double* bov, const double* bvv, const double* ovoo,
                                       const double* ovov, const double* fov, const double* t1,
                                       const double* t2, const double* eps_o, const double* eps_v,
                                       double threshold, std::size_t max_bytes, int device,
                                       std::size_t caller_bytes, std::size_t max_page_rows,
                                       std::size_t max_panel_buffers) {
  return fock_response_df_cuda_impl(o, v, q, bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v,
                                    threshold, max_bytes, device, caller_bytes, max_page_rows,
                                    max_panel_buffers, nullptr);
}

DFCudaCombinedResponseResult pullback_and_fock_df_cuda(
    std::size_t o, std::size_t v, std::size_t q, const double* bov, const double* bvv,
    const double* ovoo, const double* ovov, const double* fov, const double* t1, const double* t2,
    const double* eps_o, const double* eps_v, double threshold, std::size_t max_bytes, int device,
    std::size_t caller_bytes, std::size_t max_page_rows, std::size_t max_panel_buffers,
    bool parallel_gap_reduction, bool include_gap_response, bool fused_scalar_response,
    runtime::PrecisionDirective admitted_w, std::span<const double> occupied_weights,
    bool occupied_partition, bool compensated_fp32_w) {
  const auto started = Clock::now();
  if (!o || !v || !q || !max_bytes || device < 0 || !std::isfinite(threshold) || threshold <= 0 ||
      !max_panel_buffers || max_panel_buffers > 3)
    throw std::invalid_argument("invalid combined DF triples response dimensions or controls");
  const bool caller_requested_fused = fused_scalar_response;
  const bool has_weights = !occupied_weights.empty();
  if (has_weights) {
    if (occupied_weights.size() != o)
      throw std::invalid_argument("DF triples occupied weights must have length nocc");
    include_gap_response = true;
    fused_scalar_response = false;
  }

  const std::array<const double*, 9> host{bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v};
  const auto staging_layout = layout(o, v, q, 1);
  const auto prepared_storage = prepared_input_storage_bytes(staging_layout);
  const auto prepared_caller = checked_add(caller_bytes, prepared_storage);

  // Joint execution keeps the complete Fock pages plus the pullback outputs and
  // reverse scratch in one owner. Try the same resource order as the standalone
  // Fock owner: preserve page rows, then give up extra cached integral panels,
  // then shrink the page. Complete work/capacity is proven before input access.
  auto page_capacity = max_page_rows ? std::min(o, max_page_rows) : o;
  bool fused_selected = fused_scalar_response && !include_gap_response;
  // Validate precision before touching input pointers, even if storage later
  // refuses the candidate. Optional W ownership supplements the reverse BLAS
  // owner, so charge both providers and all cast/cache buffers simultaneously.
  bool library_available = true;
#if defined(GENERATIVEQC_TEST_HOOKS)
  library_available = !reject_w_library_for_test;
#endif
  std::optional<generated_df::WPlan> w_plan;
  if (compensated_fp32_w && admitted_w.is_strict_fp64())
    throw std::invalid_argument("compensated W requires admitted FP32 contractions");
  const auto requested_w_plan = generated_df::prepare_w_plan(admitted_w, library_available, compensated_fp32_w);
  if (!admitted_w.is_strict_fp64()) w_plan = requested_w_plan;

  // Bound triples device arena to min(caller budget, available device memory)
  // so page_capacity shrinks smoothly when memory is constrained.
  const auto triples_device_cap = query_triples_device_cap(device, max_bytes);
  // The compensated schedule owns an independent arena and, for nondegenerate (T+),
  // a single pair of X/Y cubes. The generic joint layout can be larger even at one row.
  if (compensated_fp32_w) {
    if (!supports_fast_fp32_w(requested_w_plan, max_panel_buffers))
      throw std::length_error("compensated triples W provider unavailable");
    const auto& sizes = staging_layout.sizes;
    const auto output_count = include_gap_response ? 9U : 7U;
    auto output_elems = checked_add(checked_mul(o, o), checked_mul(v, v));
    if (occupied_partition) output_elems = checked_add(output_elems, o);
    std::size_t input_elems = 0;
    for (std::size_t x = 0; x < sizes.size(); ++x) {
      input_elems = checked_add(input_elems, sizes[x]);
      if (x < output_count) output_elems = checked_add(output_elems, sizes[x]);
    }
    // Inputs and returned outputs are live on both host and device. Both providers,
    // their explicit workspace, and the forward/reverse W cubes are mandatory too.
    const auto providers = checked_add(provider_allowance, requested_w_plan.provider_bytes());
    const auto fixed_device = checked_add(
        checked_add(prepared_storage, bytes(output_elems)),
        checked_add(checked_add(providers, blas_workspace), bytes(checked_mul(12, staging_layout.v3))));
    const auto fixed_total = checked_add(
        checked_add(caller_bytes, fixed_device), bytes(checked_add(input_elems, output_elems)));
    if (fixed_device > triples_device_cap || fixed_total > max_bytes)
      throw std::length_error("compensated triples W mandatory inputs and outputs exceed budget");
  }
  if (has_weights && !std::all_of(occupied_weights.begin(), occupied_weights.end(),
                                  [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("DF triples occupied weights must be finite");

  const auto validate = [&] {
    const double minimum = validate_inputs(o, v, q, staging_layout, host, threshold);
    bool has_occupied_degeneracy = false;
    if (has_weights) {
      for (std::size_t a = 0; a < o; ++a)
        for (std::size_t b = a + 1; b < o; ++b)
          if (std::abs(eps_o[a] - eps_o[b]) < 1e-6) {
            has_occupied_degeneracy = true;
            if (std::abs(occupied_weights[a] - occupied_weights[b]) >
                1e-8 * (1.0 + std::max(std::abs(occupied_weights[a]), std::abs(occupied_weights[b]))))
              throw std::invalid_argument(
                  "weighted DF triples requires equal weights across degenerate occupied orbitals");
          }
    }
    return std::pair{minimum, has_occupied_degeneracy};
  };
  const auto try_fast = [&](double minimum, bool has_occupied_degeneracy)
      -> std::optional<DFCudaCombinedResponseResult> {
    if (!supports_fast_fp32_w(requested_w_plan, max_panel_buffers)) return std::nullopt;
    return try_pullback_and_fock_df_cuda_fp32_w(
        o, v, q, host, staging_layout, prepared_storage, prepared_caller, minimum, threshold,
        max_bytes, triples_device_cap, device, max_page_rows, parallel_gap_reduction,
        include_gap_response, caller_requested_fused, requested_w_plan, occupied_weights,
        has_weights, has_occupied_degeneracy, occupied_partition, started, compensated_fp32_w);
  };
  if (compensated_fp32_w) {
    const auto [minimum, has_occupied_degeneracy] = validate();
    if (auto fast = try_fast(minimum, has_occupied_degeneracy))
      return std::move(*fast);
    throw std::length_error("compensated triples W workspace unavailable within budget");
  }

  auto plan_exceeds = [&](const auto& cand) {
    return cand.complete > max_bytes ||
           checked_add(cand.fock.value.total, prepared_storage) > triples_device_cap;
  };
  auto admitted_layout = [&](std::size_t capacity, std::size_t panels) {
    auto candidate =
        joint_response_layout(o, v, q, capacity, panels, prepared_caller, parallel_gap_reduction,
                              include_gap_response, fused_selected, has_weights,
                              occupied_partition);
    if (fused_selected && plan_exceeds(candidate)) {
      // Never sacrifice a previously admitted page/panel policy for optional
      // seed storage. The unchanged unfused schedule remains the bounded oracle.
      fused_selected = false;
      candidate = joint_response_layout(o, v, q, capacity, panels, prepared_caller,
                                        parallel_gap_reduction, include_gap_response, false,
                                        has_weights, occupied_partition);
    }
    if (w_plan) {
      auto& value = candidate.fock.value;
      auto cursor = value.arena;
      const auto offset = reserve(cursor, w_plan->storage_bytes(o, v, value.panel_capacity));
      const auto arena_bytes = align256(cursor);
      const auto extra = checked_add(
          arena_bytes - value.arena,
          checked_add(w_plan->provider_bytes(), generated_df::WExecution::host_bytes(*w_plan)));
      const auto extra_device =
          checked_add(arena_bytes - value.arena, w_plan->provider_bytes());
      if (checked_add(candidate.complete, extra) > max_bytes ||
          checked_add(checked_add(value.total, prepared_storage), extra_device) >
              triples_device_cap) {
        // Preserve the strict page/panel admission rather than reducing
        // residency just to fit an unqualified precision experiment.
        w_plan.reset();
      } else {
        value.execution_storage = offset;
        value.arena = arena_bytes;
        value.total = checked_add(value.total, extra);
        candidate.complete = checked_add(candidate.complete, extra);
      }
    }
    return candidate;
  };
  auto plan = admitted_layout(page_capacity, max_panel_buffers);
  while (plan_exceeds(plan)) {
    if (plan.fock.value.panel_capacity > 1) {
      plan = admitted_layout(page_capacity, 1);
      if (!plan_exceeds(plan)) break;
    }
    if (page_capacity == 1) {
      throw std::length_error(
          "joint DF triples response minimum 1-row page exceeds host or safe device budget");
    }
    --page_capacity;
    plan = admitted_layout(page_capacity, max_panel_buffers);
  }
  const auto [minimum, has_occupied_degeneracy] = validate();
  if (auto fast = try_fast(minimum, has_occupied_degeneracy))
    return std::move(*fast);

  runtime::CudaDeviceScope device_scope(device);
  if (fused_selected) {
    std::size_t available = 0, total = 0;
    generativeqc_tensor::cuda_check(cudaMemGetInfo(&available, &total));
    if (plan.fock.value.total > available) {
      fused_selected = false;
      plan = admitted_layout(page_capacity, plan.fock.value.panel_capacity);
    }
  }
  PreparedInputs prepared(staging_layout, host);

  DFCudaCombinedResponseResult result;
  auto& response = result.pullback;
  auto& d = response.diagnostic;
  // The bounded response always uses FP64 reverse W, even when its forward
  // provider retains the admitted FP32 contractions.
  d.resource_fallback = !admitted_w.is_strict_fp64();
  auto& fock = result.fock;
  auto& p = plan.fock.value;
  const std::size_t requested_outputs = include_gap_response ? 9 : 7;
  const std::array<std::vector<double>*, 9> response_outputs{
      &response.bov, &response.bvv, &response.ovoo,  &response.ovov, &response.fov,
      &response.t1,  &response.t2,  &response.eps_o, &response.eps_v};
  for (std::size_t x = 0; x < requested_outputs; ++x) response_outputs[x]->resize(p.sizes[x]);
  if (occupied_partition) d.occupied_energies.resize(o);
  fock.foo.resize(checked_mul(o, o));
  fock.fvv.resize(checked_mul(v, v));

  int failed = 0;
  std::size_t response_tiles = 0;
  {
    cudaDeviceProp properties{};
    generativeqc_tensor::cuda_check(cudaGetDeviceProperties(&properties, device));
    const auto prepare_context = [&] {
      auto owner = std::make_unique<generativeqc_tensor::Context>();
      owner->prepare(device, properties.major, properties.minor, p.arena, p.error, p.library,
                     blas_workspace, provider_allowance, true);
      return owner;
    };
    std::unique_ptr<generativeqc_tensor::Context> context_owner;
    try {
      context_owner = prepare_context();
    } catch (const generativeqc_tensor::DeviceAllocationError&) {
      if (!fused_selected) throw;
      // A real failed allocation also sets the runtime's thread-local error.
      // Consume only the expected refusal; never hide a different driver error.
      const auto pending = cudaGetLastError();
      if (pending != cudaSuccess && pending != cudaErrorMemoryAllocation)
        generativeqc_tensor::cuda_check(pending);
      // Preflight cannot prevent a concurrent external allocation. Retire the
      // failed owner before one bounded retry with the original page/panel plan.
      fused_selected = false;
      plan = admitted_layout(page_capacity, p.panel_capacity);
      context_owner = prepare_context();
    }
    auto& context = *context_owner;
    auto pointer = [&](std::size_t offset) {
      return reinterpret_cast<double*>(context.arena + offset);
    };

    const generated_df::Inputs in = prepared.values;
    std::unique_ptr<generated_df::WExecution> w_execution;
    if (w_plan) {
      w_execution = std::make_unique<generated_df::WExecution>(
          *w_plan, o, v, q, context, context.arena + p.execution_storage, p.panel_capacity,
          d.fp64_gemms, d.fp32_gemms, fock.contraction_summands, d.precision_cast_elements);
      const auto& selected = w_execution->plan();
      const auto& candidate = generated_df::w_lowering_candidates[selected.selected];
      d.precision = selected.precision.arithmetic;
      d.w_contraction_storage_bits =
          d.precision.storage_dtype == runtime::PrecisionDtype::Fp64 ? 64 : 32;
      d.w_contraction_compute_bits =
          d.precision.compute_dtype == runtime::PrecisionDtype::Fp64 ? 64 : 32;
      d.w_contraction_accumulation_bits =
          d.precision.accumulation_dtype == runtime::PrecisionDtype::Fp64 ? 64 : 32;
      d.w_scientific_identity = generated_df::w_lowering_request.scientific_identity;
      d.w_semantic_identity = generated_df::w_lowering_request.semantic_identity;
      d.w_candidate_identity = candidate.identity;
      d.w_provider = candidate.provider;
      d.w_algorithm = candidate.algorithm;
      d.retained_incumbent = selected.retained_incumbent;
      d.w_provider_version = w_execution->provider_version();
      d.cuda_runtime_version = w_execution->runtime_version();
      d.w_precision_identity = selected.precision.identity;
      d.w_codegen_precision_schedule_identity = selected.schedule_identity;
      d.host_binding_bytes = generated_df::WExecution::host_bytes(selected);
      w_execution->initialize(context, in);
    }
    generated_df::ResponseOutputs out;
    const std::array<double**, 9> output_fields{&out.bov, &out.bvv, &out.ovoo,  &out.ovov, &out.fov,
                                                &out.t1,  &out.t2,  &out.eps_o, &out.eps_v};
    for (std::size_t x = 0; x < requested_outputs; ++x) {
      *output_fields[x] = pointer(plan.response_outputs[x]);
      generativeqc_tensor::cuda_check(
          cudaMemsetAsync(pointer(plan.response_outputs[x]), 0, bytes(p.sizes[x]), context.stream));
    }
    generativeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    generativeqc_tensor::cuda_check(
        cudaMemsetAsync(pointer(plan.fock.foo), 0, bytes(checked_mul(o, o)), context.stream));
    generativeqc_tensor::cuda_check(
        cudaMemsetAsync(pointer(plan.fock.fvv), 0, bytes(checked_mul(v, v)), context.stream));
    if (has_weights)
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(
          pointer(plan.device_weights), occupied_weights.data(), bytes(o),
          cudaMemcpyHostToDevice, context.stream));

    auto* panels = pointer(p.panels);
    auto* moments = pointer(p.moments);
    auto* bar_w = pointer(plan.bar_w);
    auto* bar_v = pointer(plan.bar_v);
    auto* bar_panel = pointer(plan.bar_panel);
    auto* packed = pointer(plan.packed);

    // Shared forward panel/W contractions are charged once to the Fock work
    // ledger. Pullback-only reverse contractions have a separate ledger below.
    auto forward_gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t kk,
                            double alpha, const double* a, std::size_t lda, const double* b,
                            std::size_t ldb, double beta, double* c, std::size_t ldc) {
      execute_strided_fp64_product(context, ta, tb, m, n, kk, alpha, a, lda, b, ldb, beta, c, ldc);
      const auto count = checked_mul(m, n);
      const auto blocks =
          static_cast<unsigned>(std::min<std::size_t>(1 + (count - 1) / 256, 65535));
      audit_matrix<<<blocks, 256, 0, context.stream>>>(c, m, n, ldc, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++fock.audit_kernels;
      ++d.fp64_gemms;
      fock.contraction_summands = checked_add(fock.contraction_summands, checked_mul(count, kk));
    };
    auto reverse_gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t kk,
                            double alpha, const double* a, std::size_t lda, const double* b,
                            std::size_t ldb, double beta, double* c, std::size_t ldc) {
      execute_strided_fp64_product(context, ta, tb, m, n, kk, alpha, a, lda, b, ldb, beta, c, ldc);
      const auto count = checked_mul(m, n);
      const auto blocks =
          static_cast<unsigned>(std::min<std::size_t>(1 + (count - 1) / 256, 65535));
      audit_matrix<<<blocks, 256, 0, context.stream>>>(c, m, n, ldc, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++response.audit_kernels;
      ++response.reverse_gemms;
      ++d.fp64_gemms;
      d.contraction_summands = checked_add(d.contraction_summands, checked_mul(count, kk));
    };

    std::array<std::size_t, 3> identities;
    identities.fill(std::numeric_limits<std::size_t>::max());
    std::array<std::size_t, 3> ages{};
    std::size_t epoch = 0;
    auto panel_for = [&](std::size_t occupied) {
      std::size_t slot = 0;
      for (; slot < p.panel_capacity; ++slot)
        if (identities[slot] == occupied) break;
      if (slot == p.panel_capacity) {
        slot = static_cast<std::size_t>(
            std::min_element(ages.begin(), ages.begin() + p.panel_capacity) - ages.begin());
        if (w_execution)
          w_execution->build_panel(context, in, occupied, panels + slot * p.v3, slot);
        else
          generated_df::build_panel(o, v, q, occupied, in, panels + slot * p.v3, forward_gemm);
        ++fock.panel_gemms;
        identities[slot] = occupied;
      }
      ages[slot] = ++epoch;
      return panels + slot * p.v3;
    };

    const auto small_blocks =
        static_cast<unsigned>(std::min<std::size_t>(1 + (v * v - 1) / 256, 65535));
    const auto response_index = [&](std::size_t i, std::size_t j, std::size_t k) {
      const auto prefix = checked_mul(checked_mul(i, checked_add(i, 1)), checked_add(i, 2)) / 6;
      const auto row = checked_mul(j, checked_add(j, 1)) / 2;
      return checked_add(prefix, checked_add(row, k));
    };
    const auto run_pullback = [&](std::size_t i, std::size_t j, std::size_t k) {
      const std::array<std::size_t, 3> occupied{i, j, k};
      const double multiplicity = i == k ? 6.0 : (i == j || j == k ? 2.0 : 1.0);
      const double tile_w =
          has_weights ? (occupied_weights[i] + occupied_weights[j] + occupied_weights[k]) / 3.0
                      : 1.0;
      const auto tile = response_index(i, j, k);
      if (tile >= p.tiles) throw std::logic_error("joint triples response tile index overflow");

      if (has_weights && std::abs(tile_w) <= 1e-300) {
        generativeqc_tensor::cuda_check(
            cudaMemsetAsync(pointer(p.energies) + tile, 0, sizeof(double), context.stream));
        if (occupied_partition) {
          generated_df::energy_tile(o, v, i, j, k, multiplicity, threshold, in, moments, p.blocks,
                                    pointer(p.partials), context.error, context.stream);
          ++d.epilogue_kernels;
          reduce<<<1, 256, 0, context.stream>>>(pointer(p.partials), p.blocks,
                                                pointer(plan.unweighted_energies) + tile,
                                                context.error);
          generativeqc_tensor::cuda_check(cudaGetLastError());
          ++d.reduction_kernels;
        }
        ++response_tiles;
        return;
      }

      const double effective_multiplicity = has_weights ? multiplicity / tile_w : multiplicity;
      if (fused_selected)
        generated_df::fused_response_tile(o, v, i, j, k, effective_multiplicity, threshold, in,
                                          moments, p.blocks, bar_w, bar_v, pointer(p.partials),
                                          context.error, context.stream);
      else
        generated_df::energy_tile(o, v, i, j, k, effective_multiplicity, threshold, in, moments,
                                  p.blocks, pointer(p.partials), context.error, context.stream);
      ++d.epilogue_kernels;
      reduce<<<1, 256, 0, context.stream>>>(pointer(p.partials), p.blocks,
                                            pointer(p.energies) + tile, context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++d.reduction_kernels;
      if (occupied_partition && has_weights) {
        scale_scalar<<<1, 1, 0, context.stream>>>(pointer(p.energies) + tile, 1.0 / tile_w,
                                                  pointer(plan.unweighted_energies) + tile,
                                                  context.error);
        generativeqc_tensor::cuda_check(cudaGetLastError());
      }

      if (!fused_selected) {
        generated_df::response_w_tile(o, v, i, j, k, effective_multiplicity, threshold, in,
                                      moments, p.blocks, bar_w, context.error, context.stream);
        ++response.reverse_kernels;
      }
      for (unsigned perm = 0; perm < 6; ++perm) {
        const auto* order = generated_df::permutations[perm];
        const auto I = occupied[order[0]], J = occupied[order[1]], K = occupied[order[2]];
        if (!fused_selected)
          generated_df::response_v_tile(o, v, i, j, k, perm, effective_multiplicity, threshold, in,
                                        moments, p.blocks, bar_v, context.error, context.stream);
        ovov_view<false>
            <<<small_blocks, 256, 0, context.stream>>>(o, v, I, J, in.ovov, packed, context.error);
        generativeqc_tensor::cuda_check(cudaGetLastError());
        generated_df::pullback_v(o, v, I, J, K, in, packed,
                                 bar_v + (fused_selected ? perm * p.v3 : 0), packed, out,
                                 reverse_gemm);
        ovov_view<true>
            <<<small_blocks, 256, 0, context.stream>>>(o, v, I, J, packed, out.ovov, context.error);
        generativeqc_tensor::cuda_check(cudaGetLastError());
        response.reverse_kernels += fused_selected ? 2 : 3;
      }

      if (include_gap_response) {
        generated_df::response_gap_tile(o, v, i, j, k, effective_multiplicity, threshold, in,
                                        moments, p.blocks, bar_v, context.error, context.stream);
        generated_df::GapCudaState gap{
            o, v, bar_v, pointer(plan.gap), context.error, context.stream, parallel_gap_reduction};
        const auto gaps = generated_df::gap_response_cuda(gap);
        scatter_gap<<<small_blocks, 256, 0, context.stream>>>(v, i, j, k, gaps, out.eps_o,
                                                              out.eps_v, context.error);
        generativeqc_tensor::cuda_check(cudaGetLastError());
        const auto gap_kernels =
            generated_df::gap_response_kernel_count(o, v, parallel_gap_reduction);
        response.reverse_kernels += 2 + gap_kernels;
        response.gap.kernels += gap_kernels;
      }

      for (std::size_t index = 0; index < 3; ++index) {
        if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
            occupied.begin() + index)
          continue;
        const auto I = occupied[index];
        const auto* panel = panel_for(I);
        generativeqc_tensor::cuda_check(cudaMemsetAsync(bar_panel, 0, bytes(p.v3), context.stream));
        for (std::size_t perm = 0; perm < 6; ++perm) {
          const auto* order = generated_df::permutations[perm];
          if (occupied[order[0]] != I) continue;
          generated_df::pullback_w(o, v, I, occupied[order[1]], occupied[order[2]], in, panel,
                                   bar_w + perm * p.v3, bar_panel, out, reverse_gemm);
          d.reverse_w_executed = true;
        }
        generated_df::pullback_panel(o, v, q, I, in, bar_panel, out, reverse_gemm);
      }
      ++response_tiles;
    };

    const bool distinct_moments = d.w_contraction_storage_bits == 64 &&
                                  d.w_contraction_compute_bits == 64 &&
                                  d.w_contraction_accumulation_bits == 64;
    auto page = [&](std::size_t start, std::size_t j, std::size_t k, double* x, double* y,
                    bool run_response) {
      ++fock.page_builds;
      const auto page_bytes = bytes(checked_mul(page_capacity, p.v3));
      generativeqc_tensor::cuda_check(cudaMemsetAsync(x, 0, page_bytes, context.stream));
      generativeqc_tensor::cuda_check(cudaMemsetAsync(y, 0, page_bytes, context.stream));
      for (std::size_t lane = 0; lane < page_capacity && start + lane < o; ++lane) {
        const auto i = start + lane;
        const std::array<std::size_t, 3> occupied{i, j, k};
        const auto sources = distinct_moments ? generated_df::occupied_moment_sources(i, j, k)
                                              : generated_df::MomentSourceMap{};
        for (std::size_t index = 0; index < 3; ++index) {
          if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
              occupied.begin() + index)
            continue;
          const auto* panel = panel_for(occupied[index]);
          for (std::size_t perm = 0; perm < 6; ++perm) {
            const auto* order = generated_df::permutations[perm];
            if (occupied[order[0]] != occupied[index]) continue;
            // Equal physical tuples share only forward values. Populate every
            // original slot so resolvent and all six pullbacks remain unchanged.
            const auto source = sources.index[perm];
            if (source != perm) {
              generativeqc_tensor::cuda_check(
                  cudaMemcpyAsync(moments + perm * p.v3, moments + source * p.v3, bytes(p.v3),
                                  cudaMemcpyDeviceToDevice, context.stream));
              ++fock.w_copy_calls;
              fock.w_copy_bytes = checked_add(fock.w_copy_bytes, bytes(p.v3));
              continue;
            }
            if (w_execution) {
              const auto slot = static_cast<std::size_t>((panel - panels) / p.v3);
              w_execution->build_w(context, in, occupied[order[0]], occupied[order[1]],
                                   occupied[order[2]], panel, moments + perm * p.v3, slot);
            } else {
              generated_df::build_w(o, v, occupied[order[0]], occupied[order[1]],
                                    occupied[order[2]], in, panel, moments + perm * p.v3,
                                    forward_gemm);
            }
            fock.w_gemms += 2;
          }
        }
        generated_df::resolvent_tile(o, v, i, j, k, threshold, in, moments, p.blocks,
                                     x + lane * p.v3, y + lane * p.v3, context.error,
                                     context.stream);
        ++fock.vector_cubes;
        // k<=j is fixed by the outer traversal. i>=j selects every unordered
        // occupied triple exactly once, so its W moments can serve pullback too.
        if (run_response && i >= j) run_pullback(i, j, k);
      }
    };

    const auto scatter_blocks = static_cast<unsigned>(
        std::min<std::size_t>(1 + (page_capacity * page_capacity - 1) / 256, 65535));
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t k = 0; k <= j; ++k) {
        const double weight = j == k ? 1.0 : 2.0;
        for (std::size_t left = 0; left < o; left += page_capacity) {
          // Every page becomes a left page exactly once. Run pullback only here;
          // right-page replay below is purely a bounded Fock-storage fallback.
          page(left, j, k, pointer(plan.fock.xl), pointer(plan.fock.yl), true);
          for (std::size_t lane = 0; lane < page_capacity && left + lane < o; ++lane) {
            const double lane_w =
                has_weights ? (occupied_weights[left + lane] + occupied_weights[j] +
                               occupied_weights[k]) / 3.0
                            : 1.0;
            if (has_weights && lane_w == 0.0) continue;
            generated_df::fock_vv(v, page_capacity, pointer(plan.fock.xl) + lane * p.v3,
                                  pointer(plan.fock.yl) + lane * p.v3, weight * lane_w,
                                  pointer(plan.fock.fvv), forward_gemm);
            fock.fock_gemms += 2;
          }
          if (!has_weights || has_occupied_degeneracy) {
            for (std::size_t right = left; right < o; right += page_capacity) {
              if (left != right)
                page(right, j, k, pointer(plan.fock.xr), pointer(plan.fock.yr), false);
              generated_df::fock_oo(v, page_capacity, pointer(plan.fock.xl),
                                    pointer(left == right ? plan.fock.xl : plan.fock.xr),
                                    pointer(plan.fock.yl),
                                    pointer(left == right ? plan.fock.yl : plan.fock.yr), weight,
                                    pointer(plan.fock.block), forward_gemm);
              fock.fock_gemms += 2;
              if (has_weights) {
                scatter_weighted_degenerate_fock_block<<<scatter_blocks, 256, 0, context.stream>>>(
                    o, page_capacity, left, right, j, k, in.eps_o, pointer(plan.device_weights),
                    pointer(plan.fock.block), pointer(plan.fock.foo), context.error);
              } else {
                scatter_fock_block<<<scatter_blocks, 256, 0, context.stream>>>(
                    o, page_capacity, left, right, pointer(plan.fock.block),
                    pointer(plan.fock.foo), context.error);
              }
              generativeqc_tensor::cuda_check(cudaGetLastError());
              ++fock.scatter_kernels;
            }
          }
        }
      }

    if (response_tiles != p.tiles)
      throw std::logic_error("joint DF triples pullback work mismatch");
    auto& fusion = response.scalar_fusion;
    fusion.requested = fused_scalar_response;
    fusion.selected = fused_selected;
    fusion.resource_fallback = fused_scalar_response && !include_gap_response && !fused_selected;
    fusion.schedule = fused_selected ? generated_df::scalar_fusion_identity : "unfused-primal-w-v";
    fusion.tiles = p.tiles;
    fusion.kernels = checked_mul(fused_selected ? 1 : 8, p.tiles);
    fusion.avoided_kernels = fused_selected ? checked_mul(7, p.tiles) : 0;
    fusion.workspace_bytes = bytes(checked_mul(fused_selected ? 12 : 7, p.v3));
    const auto points = checked_mul(p.tiles, p.v3);
    fusion.value_reads =
        checked_mul(points, fused_selected ? generated_df::scalar_fused_value_reads
                                           : generated_df::scalar_unfused_value_reads);
    fusion.arithmetic_ops =
        checked_mul(points, fused_selected ? generated_df::scalar_fused_arithmetic_ops
                                           : generated_df::scalar_unfused_arithmetic_ops);
    fusion.value_writes = checked_add(checked_mul(12, points), checked_mul(p.tiles, p.blocks));
    if ((!has_weights || has_occupied_degeneracy) &&
        (fock.vector_cubes != plan.fock.cubes || fock.page_builds != plan.fock.page_builds))
      throw std::logic_error("joint DF triples Fock replay work mismatch");

    response.gap.requested = include_gap_response;
    if (include_gap_response) {
      response.gap.parallel =
          generated_df::gap_response_parallel_selected(v, parallel_gap_reduction);
      response.gap.schedule = response.gap.parallel
                                  ? generated_df::gap_response_parallel_schedule_identity
                                  : "serial-source-major";
      response.gap.workspace_bytes =
          bytes(generated_df::gap_response_arena_elements(o, v, parallel_gap_reduction));
      response.gap.materialized_elements = checked_mul(
          p.tiles, generated_df::gap_response_materialized_elements(o, v, parallel_gap_reduction));
      response.gap.value_reads = checked_mul(
          p.tiles, generated_df::gap_response_value_reads(o, v, parallel_gap_reduction));
      response.gap.value_writes = checked_mul(
          p.tiles, generated_df::gap_response_value_writes(o, v, parallel_gap_reduction));
      response.gap.reduction_summands = checked_mul(
          p.tiles, generated_df::gap_response_reduction_summands(o, v, parallel_gap_reduction));
    } else {
      response.gap.schedule = generated_df::response_without_gap_identity;
    }

    reduce<<<1, 256, 0, context.stream>>>(pointer(p.energies), p.tiles, pointer(p.energy),
                                          context.error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    ++d.reduction_kernels;
    if (occupied_partition) {
      const auto* unweighted_tiles =
          has_weights ? pointer(plan.unweighted_energies) : pointer(p.energies);
      reduce_occupied_energies<<<static_cast<unsigned>(o), 256, 0, context.stream>>>(
          unweighted_tiles, o, pointer(plan.occupied_energies), context.error);
      generativeqc_tensor::cuda_check(cudaGetLastError());
      ++d.reduction_kernels;
    }
    for (std::size_t x = 0; x < requested_outputs; ++x)
      generativeqc_tensor::cuda_check(
          cudaMemcpyAsync(response_outputs[x]->data(), pointer(plan.response_outputs[x]),
                          bytes(p.sizes[x]), cudaMemcpyDeviceToHost, context.stream));
    if (occupied_partition)
      generativeqc_tensor::cuda_check(
          cudaMemcpyAsync(d.occupied_energies.data(), pointer(plan.occupied_energies), bytes(o),
                          cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&d.energy, pointer(p.energy), sizeof(double),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(fock.foo.data(), pointer(plan.fock.foo),
                                                    bytes(checked_mul(o, o)),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(fock.fvv.data(), pointer(plan.fock.fvv),
                                                    bytes(checked_mul(v, v)),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&failed, context.error, sizeof(int),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    if (failed || !std::isfinite(d.energy))
      throw std::runtime_error("nonfinite or unsafe joint DF triples response arithmetic");
    if (has_weights) {
      // Combine exact weighted diagonal gap derivatives with degenerate off-diagonals.
      for (std::size_t idx = 0; idx < o; ++idx) fock.foo[idx * o + idx] = response.eps_o[idx];
    }
    if (occupied_partition) {
      long double weighted_sum = 0.0L, magnitude = 0.0L;
      for (std::size_t idx = 0; idx < o; ++idx) {
        const double val = d.occupied_energies[idx];
        if (!std::isfinite(val))
          throw std::runtime_error("nonfinite occupied triples partition in combined response");
        const long double term =
            has_weights ? static_cast<long double>(occupied_weights[idx]) * val
                        : static_cast<long double>(val);
        weighted_sum += term;
        magnitude += std::abs(term);
      }
      if (std::abs(weighted_sum - d.energy) > 1e-11L * (1.0L + magnitude))
        throw std::runtime_error("combined DF triples occupied partition does not close to scalar");
    }

    d.fp32_forward_gemms = d.fp32_gemms;
    d.fp32_reverse_gemms = 0;
    d.w_reverse_contraction_compute_bits = 64;
    d.minimum_absolute_denominator = minimum;
    d.virtual_triples = checked_mul(checked_mul(v, checked_add(v, 1)), checked_add(v, 2)) / 6;
    d.occupied_tiles = p.tiles;
    d.epilogue_points = checked_mul(p.tiles, p.v3);
    d.workspace_bytes = p.total;
    d.arena_bytes = p.arena;
    d.provider_retained_bytes = checked_add(context.metrics.provider_retained_bytes,
                                            w_execution ? w_execution->provider_bytes() : 0);
    d.panel_capacity = p.panel_capacity;
    d.h2d_bytes = prepared.h2d_bytes;
    d.d2h_bytes = checked_add(plan.response_output_bytes, sizeof(double) + sizeof(int));
    response.borrowed_host_bytes = plan.fock.host_bytes;
    response.numeric_capacity_bytes = plan.complete;
    response.scalar_response_evaluations =
        checked_mul(include_gap_response ? generated_df::response_scalar_outputs_with_gap
                                         : generated_df::response_scalar_outputs_without_gap,
                    d.epilogue_points);

    fock.minimum_absolute_denominator = minimum;
    fock.numeric_capacity_bytes = plan.complete;
    fock.borrowed_host_bytes = plan.fock.host_bytes;
    fock.workspace_bytes = p.total;
    fock.arena_bytes = p.arena;
    fock.provider_retained_bytes = d.provider_retained_bytes;
    fock.page_capacity = page_capacity;
    fock.page_count = plan.fock.pages;
    fock.panel_capacity = p.panel_capacity;
    fock.occupied_pairs = plan.fock.pairs;
    fock.unique_vector_cubes = checked_mul(plan.fock.pairs, o);
    fock.scalar_evaluations = checked_mul(plan.fock.cubes, p.v3);
    fock.h2d_bytes = 0;
    // The shared sticky status is attributed to pullback, so the Fock D2H count
    // contains only its detached matrices.
    fock.d2h_bytes = plan.fock.output_bytes;
  }

  result.shared_input_device_bytes = prepared.storage_bytes;
  result.shared_h2d_bytes = prepared.h2d_bytes;
  result.shared_response_cubes = response_tiles;
  result.avoided_w_gemms = checked_mul(12, response_tiles);
  result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  d.seconds = result.seconds;
  fock.seconds = 0.0;
  return result;
}

}  // namespace generativeqc::cc::triples
