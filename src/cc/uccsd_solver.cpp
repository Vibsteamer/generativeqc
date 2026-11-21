#include "cc/uccsd_solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "cc/iteration_driver.hpp"
#include "cc/uccsd_executor.hpp"
#include "core/work_budget.hpp"
#include "generated_uccsd_cpu.hpp"
#include "generated_uccsd_response.hpp"
#include "posthf/capacity.hpp"
#include "solver/diis.hpp"

namespace generativeqc::cc {
namespace {

namespace graph = ucc_generated;
using Shape = std::array<std::size_t, 4>;
using Clock = std::chrono::steady_clock;
constexpr std::array<const char*, 5> kAmplitudes{"t1a", "t1b", "t2aa", "t2ab", "t2bb"};

double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

Shape populations(const hf::UnrestrictedPhysicalReference& reference) {
  return {reference.nocc[0], reference.nocc[1], reference.nbf - reference.nocc[0],
          reference.nbf - reference.nocc[1]};
}

Shape feed_shape(const graph::FeedSpec& feed, const Shape& sizes) {
  Shape shape{1, 1, 1, 1};
  for (unsigned axis = 0; axis < feed.rank; ++axis) shape[axis] = sizes[feed.axes[axis]];
  return shape;
}

std::size_t elements(const Shape& shape) {
  return graph::checked_product({shape[0], shape[1], shape[2], shape[3]});
}

std::size_t offset(const Shape& indices, const Shape& shape, unsigned rank) {
  std::size_t result = 0;
  for (unsigned axis = 0; axis < rank; ++axis) result = result * shape[axis] + indices[axis];
  return result;
}

Shape coordinates(std::size_t flat, const Shape& shape, unsigned rank) {
  Shape result{};
  for (unsigned axis = rank; axis > 0; --axis) {
    result[axis - 1] = flat % shape[axis - 1];
    flat /= shape[axis - 1];
  }
  return result;
}

std::size_t falling(std::size_t count, unsigned selected) {
  std::size_t result = 1;
  for (unsigned index = 0; index < selected; ++index) {
    if (count <= index) return 0;
    result = posthf::checked_mul(result, count - index);
  }
  return result;
}

std::size_t factorial(unsigned count) {
  return count < 2 ? 1 : count == 2 ? 2 : 6;
}

std::size_t triple_count(const Shape& sizes, unsigned alpha_count) {
  const auto beta_count = 3 - alpha_count;
  return graph::checked_product({falling(sizes[0], alpha_count) / factorial(alpha_count),
                                 falling(sizes[1], beta_count) / factorial(beta_count),
                                 falling(sizes[2], alpha_count) / factorial(alpha_count),
                                 falling(sizes[3], beta_count) / factorial(beta_count)});
}

std::size_t triples_required(const Shape& sizes) {
  return std::max({graph::required_aaa(sizes[0], sizes[1], sizes[2], sizes[3], 1),
                   graph::required_aab(sizes[0], sizes[1], sizes[2], sizes[3], 1),
                   graph::required_abb(sizes[0], sizes[1], sizes[2], sizes[3], 1),
                   graph::required_bbb(sizes[0], sizes[1], sizes[2], sizes[3], 1)});
}

struct Feed {
  Shape shape{};
  unsigned rank{};
  std::vector<double> values;
};
using Feeds = std::map<std::string, Feed>;

/** Transform real antisymmetrized physicists' blocks from the native ordered
 * chemists' provider. The spin Kronecker deltas are independent: mixed-spin
 * exchange is neither invented nor discarded by restricted symmetry. */
void integral_feed(const graph::FeedSpec& spec, Feed& feed, const Shape& sizes,
                   posthf::NativeBlockProvider& provider, posthf::ProviderWork& work, bool cuda,
                   int device) {
  if (feed.values.empty()) return;
  Shape spins{};
  posthf::MOSlots slots;
  for (unsigned axis = 0; axis < 4; ++axis) {
    spins[axis] = spec.axes[axis] % 2;
    const auto begin = spec.axes[axis] >= 2 ? sizes[spins[axis]] : 0;
    slots[axis].resize(feed.shape[axis]);
    std::iota(slots[axis].begin(), slots[axis].end(), begin);
  }
  const auto accumulate = [&](const std::array<unsigned, 4>& axes, double sign) {
    posthf::SpinMOSlots request;
    Shape request_shape{};
    for (unsigned axis = 0; axis < 4; ++axis) {
      request.slots[axis] = slots[axes[axis]];
      request.spins[axis] = static_cast<unsigned>(spins[axes[axis]]);
      request_shape[axis] = feed.shape[axes[axis]];
    }
    const auto transformed = provider.get_spin(request, &work, cuda, device);
    if (transformed.size() != feed.values.size())
      throw std::runtime_error("UCCSD MO provider returned an invalid block");
    for (std::size_t flat = 0; flat < feed.values.size(); ++flat) {
      const auto indices = coordinates(flat, feed.shape, 4);
      Shape ordered{};
      for (unsigned axis = 0; axis < 4; ++axis) ordered[axis] = indices[axes[axis]];
      feed.values[flat] += sign * transformed[offset(ordered, request_shape, 4)];
    }
  };
  if (spins[0] == spins[2] && spins[1] == spins[3]) accumulate({0, 2, 1, 3}, 1.0);
  if (spins[0] == spins[3] && spins[1] == spins[2]) accumulate({0, 3, 1, 2}, -1.0);
}

struct Excitation {
  unsigned block{};
  Shape indices{};
  double denominator{};
};

std::size_t packed_excitation_count(const Shape& sizes) {
  const auto [oa, ob, va, vb] = sizes;
  const auto pairs = [](std::size_t n) {
    return n > 1 ? posthf::checked_mul(n, n - 1) / 2 : 0;
  };
  std::size_t count = posthf::checked_add(posthf::checked_mul(oa, va), posthf::checked_mul(ob, vb));
  for (const auto block : {posthf::checked_mul(pairs(oa), pairs(va)),
                           posthf::checked_mul(pairs(ob), pairs(vb)),
                           graph::checked_product({oa, ob, va, vb})})
    count = posthf::checked_add(count, block);
  return count;
}

std::vector<Excitation> excitations(const hf::UnrestrictedPhysicalReference& reference,
                                    const Feeds& feeds, const SolverOptions& options,
                                    double& minimum) {
  std::vector<Excitation> result;
  result.reserve(packed_excitation_count(populations(reference)));
  minimum = std::numeric_limits<double>::infinity();
  for (unsigned block = 0; block < kAmplitudes.size(); ++block) {
    const auto& feed = feeds.at(kAmplitudes[block]);
    const unsigned left = block == 1 || block == 4 ? 1 : 0;
    const unsigned right = block == 3 || block == 4 ? 1 : 0;
    for (std::size_t flat = 0; flat < feed.values.size(); ++flat) {
      const auto indices = coordinates(flat, feed.shape, feed.rank);
      if ((block == 2 || block == 4) && (indices[0] >= indices[1] || indices[2] >= indices[3]))
        continue;
      double denominator = reference.orbital_energies[left][indices[0]];
      if (block < 2) {
        denominator -= reference.orbital_energies[left][reference.nocc[left] + indices[1]];
      } else {
        denominator += reference.orbital_energies[right][indices[1]];
        denominator -= reference.orbital_energies[left][reference.nocc[left] + indices[2]];
        denominator -= reference.orbital_energies[right][reference.nocc[right] + indices[3]];
      }
      if (!std::isfinite(denominator) || denominator >= -options.denominator_threshold)
        throw std::invalid_argument("UCCSD requires separated unshifted occupied/virtual energies");
      minimum = std::min(minimum, -denominator);
      denominator -= (block < 2 ? 1 : 2) * options.level_shift;
      if (!std::isfinite(denominator))
        throw std::invalid_argument("nonfinite UCCSD shifted denominator");
      result.push_back({block, indices, denominator});
    }
  }
  return result;
}

void unpack(const std::vector<double>& packed, const std::vector<Excitation>& entries,
            Feeds& feeds) {
  for (const auto* name : kAmplitudes) {
    auto& values = feeds.at(name).values;
    std::fill(values.begin(), values.end(), 0.0);
  }
  for (std::size_t slot = 0; slot < entries.size(); ++slot) {
    const auto& entry = entries[slot];
    auto& feed = feeds.at(kAmplitudes[entry.block]);
    feed.values[offset(entry.indices, feed.shape, feed.rank)] = packed[slot];
    if (entry.block != 2 && entry.block != 4) continue;
    auto indices = entry.indices;
    std::swap(indices[0], indices[1]);
    feed.values[offset(indices, feed.shape, 4)] = -packed[slot];
    std::swap(indices[2], indices[3]);
    feed.values[offset(indices, feed.shape, 4)] = packed[slot];
    std::swap(indices[0], indices[1]);
    feed.values[offset(indices, feed.shape, 4)] = -packed[slot];
  }
}

std::array<const double*, 5> residuals(const graph::Outputs& output) {
  return {output.r1a, output.r1b, output.r2aa, output.r2ab, output.r2bb};
}

IterationMetrics metrics(const graph::Outputs& output, const Feeds& feeds) {
  IterationMetrics result{*output.correlation_energy, 0.0, 0.0};
  const auto blocks = residuals(output);
  if (!std::isfinite(result.energy)) throw std::runtime_error("nonfinite UCCSD energy");
  for (unsigned block = 0; block < blocks.size(); ++block) {
    const auto size = feeds.at(kAmplitudes[block]).values.size();
    auto& maximum = block < 2 ? result.r1 : result.r2;
    for (std::size_t index = 0; index < size; ++index) {
      if (!std::isfinite(blocks[block][index]))
        throw std::runtime_error("nonfinite UCCSD residual");
      maximum = std::max(maximum, std::abs(blocks[block][index]));
    }
  }
  return result;
}

std::vector<double> packed_residual(const graph::Outputs& output,
                                    const std::vector<Excitation>& entries, const Feeds& feeds) {
  const auto blocks = residuals(output);
  std::vector<double> result(entries.size());
  for (std::size_t slot = 0; slot < entries.size(); ++slot) {
    const auto& entry = entries[slot];
    const auto& feed = feeds.at(kAmplitudes[entry.block]);
    result[slot] = blocks[entry.block][offset(entry.indices, feed.shape, feed.rank)];
  }
  return result;
}

/** A singleton tile never materializes a six-dimensional triples tensor.
 * Contracted m/e axes retain their full own-spin population; external slots
 * are gathered through compiler-owned bindings, including mixed-spin signs. */
template <class Inputs, class Bindings, class Evaluate>
double triples_channel(const std::array<unsigned, 3>& spins, const Bindings& bindings,
                       Evaluate evaluate, const hf::UnrestrictedPhysicalReference& reference,
                       const Feeds& feeds, const Shape& sizes, const SolverOptions& options,
                       std::vector<double>& arena, UnrestrictedSolve& result,
                       UnrestrictedCudaExecutor* executor, const char* channel) {
  const unsigned alpha_count = static_cast<unsigned>(std::count(spins.begin(), spins.end(), 0));
  if (!triple_count(sizes, alpha_count)) return 0.0;
  std::array<std::size_t, 6> shape{}, selected{};
  for (unsigned axis = 0; axis < shape.size(); ++axis)
    shape[axis] = sizes[spins[axis % 3] + (axis < 3 ? 0 : 2)];
  std::vector<std::vector<double>> buffers(bindings.size());
  constexpr std::string_view labels = "ijkabc";
  const auto lanes = executor ? graph::kTriplesMomentPanelLanes : 1;
  std::size_t loaded = 0;
  Inputs inputs;
  double energy = 0.0;
  const auto flush = [&] {
    if (!loaded) return;
    if (executor) {
      UnrestrictedFeedViews source;
      for (std::size_t index = 0; index < bindings.size(); ++index) {
        const auto& binding = bindings[index];
        auto& buffer = buffers[index];
        // A masked final lane has zero numerators and the finite denominators
        // already qualified for the first physical lane in this batch.
        const auto width = buffer.size() / lanes;
        const auto padding = std::string_view(binding.source).starts_with("eps_")
                                 ? buffer[0]
                                 : 0.0;
        std::fill(buffer.begin() + loaded * width, buffer.end(), padding);
        source.emplace(binding.name, buffer);
      }
      const auto tile = executor->evaluate(std::string("energy_panel_") + channel, source);
      const auto& values = tile.at("triples_energy");
      for (std::size_t lane = 0; lane < loaded; ++lane) energy += values[lane];
    } else {
      const auto tile =
          evaluate(sizes[0], sizes[1], sizes[2], sizes[3], 1, inputs, arena.data(), arena.size());
      energy += *tile.triples_energy;
    }
    if (!std::isfinite(energy)) throw std::runtime_error("nonfinite unrestricted triples energy");
    loaded = 0;
  };
  const auto accumulate = [&] {
    double denominator = 0.0;
    for (unsigned axis = 0; axis < 3; ++axis) {
      const auto spin = spins[axis];
      denominator += reference.orbital_energies[spin][selected[axis]];
      denominator -= reference.orbital_energies[spin][reference.nocc[spin] + selected[axis + 3]];
    }
    if (!std::isfinite(denominator) || denominator >= -options.denominator_threshold)
      throw std::invalid_argument("near-zero or nonphysical unrestricted triples denominator");
    result.minimum_denominator = std::min(result.minimum_denominator, -denominator);
    for (std::size_t binding_index = 0; binding_index < bindings.size(); ++binding_index) {
      const auto& binding = bindings[binding_index];
      auto& buffer = buffers[binding_index];
      if (std::string_view(binding.source).starts_with("eps_")) {
        const auto position = labels.find(binding.labels[0]);
        const auto spin = spins[position % 3];
        buffer.resize(lanes);
        buffer[loaded] = reference.orbital_energies[spin][selected[position] +
                                                          (position < 3 ? 0 : reference.nocc[spin])];
      } else {
        const auto& feed = feeds.at(binding.source);
        Shape tile_shape{1, 1, 1, 1};
        for (unsigned axis = 0; axis < binding.rank; ++axis) {
          const char label = binding.labels[axis];
          if (label == 'm' || label == 'e') tile_shape[axis] = feed.shape[binding.axes[axis]];
        }
        const auto width = elements(tile_shape);
        buffer.resize(posthf::checked_mul(width, lanes));
        for (std::size_t tile_flat = 0; tile_flat < width; ++tile_flat) {
          const auto tile_indices = coordinates(tile_flat, tile_shape, binding.rank);
          Shape source_indices{};
          for (unsigned axis = 0; axis < binding.rank; ++axis) {
            const auto external = labels.find(binding.labels[axis]);
            source_indices[binding.axes[axis]] =
                external == std::string_view::npos ? tile_indices[axis] : selected[external];
          }
          buffer[loaded * width + tile_flat] =
              feed.values[offset(source_indices, feed.shape, feed.rank)];
        }
      }
      inputs.*(binding.field) = buffer.data();
    }
    if (++loaded == lanes) flush();
    ++result.triples_tiles;
  };
  // The generated scalar includes all spin-position multiplicities. Within
  // one spin channel an equal-spin exchange changes the sign of both
  // numerators and leaves their product and denominator invariant. Enumerate
  // just increasing occupied and virtual tuples within each spin group.
  const auto visit = [&](auto& self, unsigned axis) -> void {
    if (axis == shape.size()) {
      accumulate();
      return;
    }
    const unsigned position = axis % 3;
    const auto first = position && spins[position] == spins[position - 1] ? selected[axis - 1] + 1 : 0;
    for (selected[axis] = first; selected[axis] < shape[axis]; ++selected[axis])
      self(self, axis + 1);
  };
  visit(visit, 0);
  flush();
  const auto permutations = factorial(alpha_count) * factorial(3 - alpha_count);
  const auto full_energy = energy * static_cast<double>(permutations * permutations);
  if (!std::isfinite(full_energy)) throw std::runtime_error("nonfinite unrestricted triples energy");
  return full_energy;
}

}  // namespace

std::size_t unrestricted_capacity(std::size_t nbf, std::size_t alpha, std::size_t beta,
                                  const SolverOptions& options, bool triples,
                                  std::size_t reference_capacity, bool cuda,
                                  std::optional<std::size_t> provider_capacity) {
  if (!nbf || alpha > nbf || beta > nbf)
    throw std::invalid_argument("native unrestricted CC requires valid spin populations");
  const Shape sizes{alpha, beta, nbf - alpha, nbf - beta};
  std::size_t feed_elements = 0, amplitude_elements = 0;
  for (const auto& feed : graph::kFeeds) {
    const auto count = elements(feed_shape(feed, sizes));
    feed_elements = posthf::checked_add(feed_elements, count);
    if (feed.name[0] == 't')
      amplitude_elements = posthf::checked_add(amplitude_elements, count);
  }
  const auto arena = cuda ? 0 : graph::required(sizes[0], sizes[1], sizes[2], sizes[3]);
  const auto triples_arena = triples && !cuda ? triples_required(sizes) : 0;
  const auto n4 = graph::checked_product({nbf, nbf, nbf, nbf});
  const auto packed = packed_excitation_count(sizes);
  // Pulay histories and temporary updates pack only excitation coordinates.
  // Integral feeds are owned once. A synchronous device replay borrows them.
  const auto state = graph::checked_product({packed, posthf::checked_add(
      12, posthf::checked_mul(2, options.diis_size))});
  auto retained = posthf::checked_add(feed_elements, std::max(arena, triples_arena));
  retained = posthf::checked_add(retained, state);
  if (cuda)
    // The previous detached residual may coexist with the next graph result.
    retained = posthf::checked_add(retained, posthf::checked_add(amplitude_elements, 1));
  if (triples)
    retained = posthf::checked_add(retained, graph::checked_product(
        {256, nbf, cuda ? graph::kTriplesMomentPanelLanes : 1}));
  auto capacity =
      posthf::checked_add(reference_capacity, posthf::checked_mul(sizeof(double), retained));
  capacity = posthf::checked_add(capacity, posthf::checked_mul(packed, sizeof(Excitation)));
  // The shape-only caller has no concrete transform owner yet. Actual
  // execution passes the checked peak of its exact provider requests.
  const auto transform_capacity =
      provider_capacity ? *provider_capacity : graph::checked_product({8, n4, sizeof(double)});
  capacity = posthf::checked_add(capacity, transform_capacity);
  capacity = posthf::checked_add(
      capacity, posthf::checked_mul(
                    8 * sizeof(double),
                    graph::checked_product({options.diis_size + 1ULL, options.diis_size + 1ULL})));
  if (capacity > options.max_bytes)
    throw std::length_error("complete native UCCSD endpoint exceeds numeric memory budget");
  auto work =
      posthf::checked_mul(graph::work(sizes[0], sizes[1], sizes[2], sizes[3]),
                          posthf::checked_add(posthf::checked_mul(2, options.max_iterations), 3));
  work = posthf::checked_add(work, graph::checked_product({8, n4, nbf, graph::kFeeds.size()}));
  if (triples) {
    const std::array<std::size_t, 4> tile_work{
        graph::work_bbb(sizes[0], sizes[1], sizes[2], sizes[3], 1),
        graph::work_abb(sizes[0], sizes[1], sizes[2], sizes[3], 1),
        graph::work_aab(sizes[0], sizes[1], sizes[2], sizes[3], 1),
        graph::work_aaa(sizes[0], sizes[1], sizes[2], sizes[3], 1)};
    const std::array<std::string, 4> programs{
        "energy_panel_bbb", "energy_panel_abb", "energy_panel_aab", "energy_panel_aaa"};
    for (unsigned alpha_count = 0; alpha_count <= 3; ++alpha_count) {
      auto panels = triple_count(sizes, alpha_count);
      auto panel_work = tile_work[alpha_count];
      if (cuda && panels) {
        const auto lanes = graph::kTriplesMomentPanelLanes;
        panels = (panels - 1) / lanes + 1;
        panel_work = graph::response_work(programs[alpha_count], sizes[0], sizes[1], sizes[2],
                                          sizes[3], nbf, nbf, 1, lanes);
      }
      work = posthf::checked_add(work, posthf::checked_mul(panels, panel_work));
    }
  }
  core::admit_correlation_work("uccsd_primal", work, capacity, nbf, alpha, beta,
                               options.max_iterations);
  return capacity;
}

std::size_t unrestricted_solver_provider_capacity(
    const hf::UnrestrictedPhysicalReference& reference,
    const posthf::NativeBlockProvider& provider, bool cuda) {
  const auto sizes = populations(reference);
  std::size_t peak = 0;
  for (const auto& spec : graph::kFeeds) {
    if (!std::string(spec.name).starts_with('g')) continue;
    const auto shape = feed_shape(spec, sizes);
    if (!elements(shape)) continue;
    Shape spins{};
    for (unsigned axis = 0; axis < 4; ++axis) spins[axis] = spec.axes[axis] % 2;
    const auto admit = [&](const std::array<unsigned, 4>& axes) {
      Shape request{};
      for (unsigned axis = 0; axis < 4; ++axis) request[axis] = shape[axes[axis]];
      peak = std::max(peak, provider.batch_bytes(request, 1, cuda));
    };
    if (spins[0] == spins[2] && spins[1] == spins[3]) admit({0, 2, 1, 3});
    if (spins[0] == spins[3] && spins[1] == spins[2]) admit({0, 3, 1, 2});
  }
  return peak;
}

UnrestrictedSolve solve_unrestricted(const hf::UnrestrictedPhysicalReference& reference,
                                     const integrals::ElectronInteractionSource& source,
                                     const SolverOptions& options, bool triples,
                                     std::size_t reference_capacity, bool retain_amplitudes,
                                     bool cuda, int device) {
  UnrestrictedSolve result;
  const auto sizes = populations(reference);
  auto started = Clock::now();
  posthf::NativeBlockProvider provider(
      source, reference, options.max_bytes, cuda ? 8 : 2,
      cuda ? posthf::AOTileDomain::Basis : posthf::AOTileDomain::Shell);
  const auto provider_peak = unrestricted_solver_provider_capacity(reference, provider, cuda);
  result.numeric_capacity_bytes = unrestricted_capacity(
      reference.nbf, reference.nocc[0], reference.nocc[1], options, triples, reference_capacity,
      cuda, provider_peak);
  std::unique_ptr<UnrestrictedCudaExecutor> executor;
#if !GENERATIVEQC_HAS_CUDA
  if (cuda) throw std::invalid_argument("unrestricted CUDA CC requires a CUDA build");
#endif
  Feeds feeds;
  graph::Inputs inputs;
  for (const auto& spec : graph::kFeeds) {
    const std::string name = spec.name;
    auto& feed = feeds[name];
    feed.shape = feed_shape(spec, sizes);
    feed.rank = spec.rank;
    feed.values.resize(elements(feed.shape));
    if (name.starts_with('g'))
      integral_feed(spec, feed, sizes, provider, result.provider_work, cuda, device);
    if (name.starts_with('f')) {
      const auto spin = spec.axes[0] % 2;
      const auto& coefficients = reference.coefficients[spin];
      const auto& fock = reference.fock[spin];
      for (std::size_t row = 0; row < feed.shape[0]; ++row)
        for (std::size_t column = 0; column < feed.shape[1]; ++column) {
          const auto orbital_row = row + (spec.axes[0] >= 2 ? sizes[spin] : 0);
          const auto orbital_column = column + (spec.axes[1] >= 2 ? sizes[spin] : 0);
          double value = 0.0;
          for (std::size_t mu = 0; mu < reference.nbf; ++mu)
            for (std::size_t nu = 0; nu < reference.nbf; ++nu)
              value += coefficients[mu * reference.nbf + orbital_row] *
                       fock[mu * reference.nbf + nu] *
                       coefficients[nu * reference.nbf + orbital_column];
          feed.values[row * feed.shape[1] + column] = value;
        }
    }
    inputs.*(spec.field) = feed.values.data();
  }
#if GENERATIVEQC_HAS_CUDA
  if (cuda) {
    // Integral blocks are staged synchronously. Their temporary device buffers
    // are gone before reserving the graph arena. Retain the conservative
    // combined numeric bound for the public correlation memory budget.
    executor =
        make_unrestricted_cuda_executor(reference.nbf, reference.nocc[0], reference.nocc[1], device,
                                        options.max_bytes - result.numeric_capacity_bytes, false);
    result.numeric_capacity_bytes =
        posthf::checked_add(result.numeric_capacity_bytes, executor->numeric_capacity_bytes());
  }
#endif
  const auto entries = excitations(reference, feeds, options, result.minimum_denominator);
  if (entries.empty()) result.minimum_denominator = 0.0;
  std::vector<double> current(entries.size());
  std::vector<double> arena(executor ? 0 : graph::required(sizes[0], sizes[1], sizes[2], sizes[3]));
  solver::Diis diis(options.diis_size, current.size());
  result.problem_seconds = seconds(started);
  started = Clock::now();
  UnrestrictedFeeds device_outputs;
  const auto evaluate = [&](bool replay = false) {
    const auto phase_started = Clock::now();
    unpack(current, entries, feeds);
    ++result.graph_calls;
    graph::Outputs output;
    if (executor) {
      UnrestrictedFeedViews values;
      for (const auto& [name, feed] : feeds) values.emplace(name, feed.values);
      device_outputs = executor->evaluate("cc", values);
      output = {device_outputs.at("correlation_energy").data(),
                device_outputs.at("r1a").data(),
                device_outputs.at("r1b").data(),
                device_outputs.at("r2aa").data(),
                device_outputs.at("r2ab").data(),
                device_outputs.at("r2bb").data()};
    } else {
      output = graph::evaluate(sizes[0], sizes[1], sizes[2], sizes[3], inputs, arena.data(),
                               arena.size());
    }
    if (replay)
      result.replay_seconds += seconds(phase_started);
    else
      result.iteration_seconds += seconds(phase_started);
    return output;
  };
  run_cc_iterations<graph::Outputs>(
      options,
      [&](const auto&) {
        const auto output = evaluate();
        return std::pair{output, metrics(output, feeds)};
      },
      [&](unsigned iterations, const IterationMetrics& status, double delta) {
        result.iterations = iterations;
        result.correlation_energy = status.energy;
        result.energy_change = delta;
        result.singles_residual = status.r1;
        result.doubles_residual = status.r2;
      },
      [&] {
        const auto fresh = metrics(evaluate(true), feeds);
        ++result.replay_calls;
        result.replay_singles_residual = fresh.r1;
        result.replay_doubles_residual = fresh.r2;
        return fresh;
      },
      [&](const graph::Outputs& output) -> std::optional<graph::Outputs> {
        ++result.update_calls;
        auto phase_started = Clock::now();
        auto error = packed_residual(output, entries, feeds);
        for (std::size_t slot = 0; slot < current.size(); ++slot) {
          current[slot] += (1 - options.damping) * error[slot] / entries[slot].denominator;
          if (!std::isfinite(current[slot]))
            throw std::runtime_error("nonfinite UCCSD amplitude update");
        }
        result.update_seconds += seconds(phase_started);
        const auto trial = evaluate();
        phase_started = Clock::now();
        error = packed_residual(trial, entries, feeds);
        current = diis.update_with_status(current, std::move(error)).vector;
        result.diis_seconds += seconds(phase_started);
        return std::nullopt;
      },
      [&] { result.converged = true; },
      [](const std::runtime_error& error) { throw std::runtime_error(error.what()); });
  result.solver_seconds = seconds(started);
  const auto retain = [&] {
    if (retain_amplitudes && result.converged)
      for (unsigned block = 0; block < kAmplitudes.size(); ++block)
        result.amplitudes[block] = std::move(feeds.at(kAmplitudes[block]).values);
  };
  if (!result.converged || !triples) {
    retain();
    return result;
  }
  // The converged replay leaves feeds bound to exactly the admitted parent
  // excitation state. Retire DIIS and CC arena before allocating triples work.
  diis.clear();
  std::vector<double>().swap(current);
  std::vector<double>().swap(arena);
  if (!executor) arena.resize(triples_required(sizes));
  result.triples_workspace_bytes = posthf::checked_mul(
      sizeof(double), posthf::checked_add(
          arena.size(), graph::checked_product(
              {256, reference.nbf, executor ? graph::kTriplesMomentPanelLanes : 1})));
  started = Clock::now();
  result.triples_energy += triples_channel<graph::Inputs_aaa>(
      {0, 0, 0}, graph::kBindings_aaa, graph::evaluate_aaa, reference, feeds, sizes,
      options, arena, result, executor.get(), "aaa");
  result.triples_energy += triples_channel<graph::Inputs_aab>(
      {0, 0, 1}, graph::kBindings_aab, graph::evaluate_aab, reference, feeds, sizes,
      options, arena, result, executor.get(), "aab");
  result.triples_energy += triples_channel<graph::Inputs_abb>(
      {0, 1, 1}, graph::kBindings_abb, graph::evaluate_abb, reference, feeds, sizes,
      options, arena, result, executor.get(), "abb");
  result.triples_energy += triples_channel<graph::Inputs_bbb>(
      {1, 1, 1}, graph::kBindings_bbb, graph::evaluate_bbb, reference, feeds, sizes,
      options, arena, result, executor.get(), "bbb");
  result.triples_seconds = seconds(started);
  retain();
  return result;
}

const char* unrestricted_equation_hash() noexcept { return graph::kEquationHash; }
const char* unrestricted_triples_hash() noexcept { return graph::kTriplesHash; }

}  // namespace generativeqc::cc
