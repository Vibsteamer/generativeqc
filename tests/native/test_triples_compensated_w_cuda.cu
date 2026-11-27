#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cc/df_triples_compensated_w.cuh"

namespace {

namespace cw = generativeqc::cc::triples::compensated_w;
using generativeqc_tensor::cuda_check;

struct Case {
  char ta, tb;
  std::size_t m, n, k;
  double alpha, beta;
};

struct Device {
  double* a{};
  double* b{};
  double* c{};
  void* scratch{};
  int* error{};
  ~Device() {
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(scratch);
    cudaFree(error);
  }
};

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

double element(const std::vector<double>& x, char trans, std::size_t ld, std::size_t row,
               std::size_t col) {
  return trans == 'N' ? x[row * ld + col] : x[col * ld + row];
}

// Vector scales span many binades and their products exceed the FP32 exponent
// range, so the per-vector grids and the exponent bookkeeping are exercised.
// Along K a vector holds zeros and falls by up to 2^-36, so some components
// first become significant in a later limb, yet every segment stays inside
// the window that the low limb can represent. `depth_major` stores element
// (vector v, depth t) at t * vectors + v, otherwise at v * depth + t.
std::vector<double> operand(std::size_t vectors, std::size_t depth, bool depth_major,
                            std::mt19937_64& rng) {
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  std::vector<double> x(vectors * depth);
  for (std::size_t v = 0; v < vectors; ++v) {
    const double scale = std::ldexp(1.0, static_cast<int>(v % 7) * 23 - 70);
    for (std::size_t t = 0; t < depth; ++t) {
      const auto i = v * depth + t;
      const double fall = std::ldexp(1.0, -4 * static_cast<int>(t % 10));
      x[depth_major ? t * vectors + v : i] = (i % 11 == 3) ? 0.0 : unit(rng) * scale * fall;
    }
  }
  return x;
}

struct Reference {
  std::vector<double> value, tolerance;
};

// Long-double products with an a priori bound on the schedule error. High and
// middle partials are exact, so FP32 rounding enters only through residual
// limbs. Their grid is at most 2^-16 of a vector's top grid, which is below
// twice the segment maximum, and each residual digit rounds once, so a
// segment of width W <= 64 errs by at most
// (W^2 2^-37 + W 2^-38) max|x| max|y| <= 2^-30 W max|x| max|y|.
// The seed and the nine FP64 terms per segment each round by at most 2^-53
// of a running sum below |beta c| + 1.02 |alpha| times that grid sum, and
// 2^-48 of the absolute sum covers the long-double oracle itself.
Reference oracle(const Case& t, const std::vector<double>& a, const std::vector<double>& b,
                 const std::vector<double>& c0) {
  const auto lda = t.ta == 'N' ? t.k : t.m, ldb = t.tb == 'N' ? t.n : t.k;
  const auto segments = t.k / cw::tile + (t.k % cw::tile != 0);
  const auto roundings = static_cast<long double>(cw::limbs * cw::limbs * segments + 2);
  Reference ref{std::vector<double>(t.m * t.n), std::vector<double>(t.m * t.n)};
  for (std::size_t i = 0; i < t.m; ++i)
    for (std::size_t j = 0; j < t.n; ++j) {
      long double sum = 0, abs = 0, grid = 0;
      for (std::size_t first = 0; first < t.k; first += cw::tile) {
        const auto last = std::min(t.k, first + cw::tile);
        long double mx = 0, my = 0;
        for (std::size_t p = first; p < last; ++p) {
          const long double x = element(a, t.ta, lda, i, p), y = element(b, t.tb, ldb, p, j);
          sum += x * y;
          abs += std::fabs(x * y);
          mx = std::max(mx, std::fabs(x));
          my = std::max(my, std::fabs(y));
        }
        grid += static_cast<long double>(last - first) * mx * my;
      }
      const long double seed =
          t.beta == 0 ? 0.0L : static_cast<long double>(t.beta) * c0[i * t.n + j];
      ref.value[i * t.n + j] = static_cast<double>(t.alpha * sum + seed);
      const long double scale = std::fabs(static_cast<long double>(t.alpha));
      ref.tolerance[i * t.n + j] =
          static_cast<double>(scale * std::ldexp(grid, -30) +
                              std::ldexp(roundings * (std::fabs(seed) + scale * grid), -52) +
                              std::ldexp(scale * abs + std::fabs(seed), -48));
    }
  return ref;
}

struct Run {
  std::vector<double> c;
  std::size_t calls{};
  int error{};
};

Run run(const generativeqc::tensor::CudaContractionContext& provider, cudaStream_t stream,
        const Case& t, const std::vector<double>& a, const std::vector<double>& b,
        const std::vector<double>& c0, std::size_t limit) {
  Device d;
  cuda_check(cudaMalloc(&d.a, a.size() * sizeof(double)));
  cuda_check(cudaMalloc(&d.b, b.size() * sizeof(double)));
  cuda_check(cudaMalloc(&d.c, c0.size() * sizeof(double)));
  cuda_check(cudaMalloc(&d.scratch, limit));
  cuda_check(cudaMalloc(&d.error, sizeof(int)));
  cuda_check(cudaMemcpy(d.a, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d.b, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d.c, c0.data(), c0.size() * sizeof(double), cudaMemcpyHostToDevice));
  cuda_check(cudaMemset(d.error, 0, sizeof(int)));
  const auto lda = t.ta == 'N' ? t.k : t.m, ldb = t.tb == 'N' ? t.n : t.k;
  Run out;
  out.calls = cw::execute(provider, stream, t.ta, t.tb, t.m, t.n, t.k, t.alpha, d.a, lda, d.b, ldb,
                          t.beta, d.c, t.n, d.scratch, limit, d.error);
  cuda_check(cudaStreamSynchronize(stream));
  out.c.resize(c0.size());
  cuda_check(cudaMemcpy(out.c.data(), d.c, c0.size() * sizeof(double), cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out.error, d.error, sizeof(int), cudaMemcpyDeviceToHost));
  return out;
}

struct Limit {
  const char* name;
  std::size_t bytes;
  cw::Mode mode;
  std::size_t segments;
};

std::vector<Limit> limits(const Case& t) {
  const auto cap = std::min(t.k, cw::tile), full = t.k / cap;
  std::vector<Limit> out{
      {"pairs", cw::workspace_bytes(t.m, t.n, t.k), cw::Mode::pairs, 1},
      {"rows", cw::stacked_bytes(t.m, t.n, cap, 1, 1), cw::Mode::rows, 1},
      {"batched-1", cw::stacked_bytes(t.m, t.n, cap, 1, cw::limbs), cw::Mode::batched, 1}};
  if (full > 2)
    out.push_back(
        {"batched-2", cw::stacked_bytes(t.m, t.n, cap, 2, cw::limbs) + 255, cw::Mode::batched, 2});
  if (full > 1)
    out.push_back({"batched-all", cw::stacked_bytes(t.m, t.n, cap, full, cw::limbs),
                   cw::Mode::batched, full});
  return out;
}

void check_case(const generativeqc::tensor::CudaContractionContext& provider, cudaStream_t stream,
                const Case& t, std::mt19937_64& rng) {
  const auto a = operand(t.m, t.k, t.ta == 'T', rng);
  const auto b = operand(t.n, t.k, t.tb == 'N', rng);
  auto c0 = operand(t.m, t.n, false, rng);
  if (t.beta == 0) c0.assign(c0.size(), std::nan(""));
  const auto ref = oracle(t, a, b, c0);
  const auto segments = t.k / cw::tile + (t.k % cw::tile != 0);
  const std::string label = std::string(1, t.ta) + t.tb + " " + std::to_string(t.m) + "x" +
                            std::to_string(t.n) + "x" + std::to_string(t.k);
  std::vector<Run> runs;
  for (const auto& limit : limits(t)) {
    const auto selected = cw::plan(t.m, t.n, t.k, limit.bytes);
    require(selected.mode == limit.mode && selected.segments_per_batch == limit.segments,
            label + ": limit " + limit.name + " selected another schedule");
    auto result = run(provider, stream, t, a, b, c0, limit.bytes);
    require(result.error == 0,
            label + " " + limit.name + ": device error " + std::to_string(result.error));
    require(result.calls == cw::limbs * cw::limbs * segments,
            label + " " + limit.name + ": nine limb-pair products per K segment");
    for (std::size_t i = 0; i < ref.value.size(); ++i) {
      const double tolerance = ref.tolerance[i] + 1e-300;
      require(std::isfinite(result.c[i]) && std::fabs(result.c[i] - ref.value[i]) <= tolerance,
              label + " " + limit.name + ": element " + std::to_string(i) + " differs from the " +
                  "long-double oracle");
      if (!runs.empty())
        require(std::fabs(result.c[i] - runs.front().c[i]) <= 2 * tolerance,
                label + " " + limit.name + ": schedules disagree at element " + std::to_string(i));
    }
    const auto again = run(provider, stream, t, a, b, c0, limit.bytes);
    require(std::memcmp(again.c.data(), result.c.data(), result.c.size() * sizeof(double)) == 0,
            label + " " + limit.name + ": repeated execution is not bitwise reproducible");
    runs.push_back(std::move(result));
  }
}

void check_failures(const generativeqc::tensor::CudaContractionContext& provider,
                    cudaStream_t stream, std::mt19937_64& rng) {
  const Case t{'N', 'N', 6, 5, 150, 1.0, 0.0};
  auto a = operand(t.m, t.k, false, rng);
  const auto b = operand(t.n, t.k, true, rng);
  const std::vector<double> c0(t.m * t.n, 0.0);
  const auto probe = 2 * t.k + 131;
  a[probe] = std::nan("");
  for (const auto& limit : limits(t)) {
    const auto result = run(provider, stream, t, a, b, c0, limit.bytes);
    require(result.error == 26, std::string("nonfinite input not reported by ") + limit.name);
  }
  // A component far below the low-limb window of its segment would be lost.
  a[probe] = 0;
  double peak = 0;
  for (std::size_t p = 2 * cw::tile; p < t.k; ++p) peak = std::max(peak, std::fabs(a[2 * t.k + p]));
  a[probe] = std::ldexp(peak, -200);
  for (const auto& limit : limits(t)) {
    const auto result = run(provider, stream, t, a, b, c0, limit.bytes);
    require(result.error == 26,
            std::string("component below the segment window not reported by ") + limit.name);
  }
  bool rejected = false;
  try {
    (void)run(provider, stream, t, a, b, c0, cw::workspace_bytes(t.m, t.n, t.k) - 1);
  } catch (const std::length_error&) {
    rejected = true;
  }
  require(rejected, "scratch one byte below the per-pair requirement was admitted");
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::puts("SKIP: no CUDA device");
    return 77;
  }
  try {
    cudaStream_t stream{};
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    generativeqc::tensor::CudaContractionContext provider;
    require(provider.prepare(stream), "pedantic FP32 provider unavailable");
    std::mt19937_64 rng(20261127);
    const Case cases[] = {{'N', 'N', 5, 7, 3, 1.0, 0.0},      {'T', 'N', 9, 4, 64, -1.0, 1.0},
                          {'N', 'T', 6, 5, 200, 0.75, 0.5},   {'T', 'T', 3, 11, 517, 1.0, 1.0},
                          {'N', 'N', 33, 2, 1000, -3.0, 0.0}, {'T', 'N', 17, 19, 4099, 1.0, 1.0}};
    for (const auto& t : cases) check_case(provider, stream, t, rng);
    check_failures(provider, stream, rng);
    provider.reset();
    cuda_check(cudaStreamDestroy(stream));
    std::puts("compensated triples W schedules agree with the long-double oracle");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
