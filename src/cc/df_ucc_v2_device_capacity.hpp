#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include "cc/df_ucc_v2_residual_capacity.hpp"

namespace generativeqc::cc::v2_detail {

class UccDeviceLimitError : public std::length_error {
 public:
  explicit UccDeviceLimitError(const char* message) : std::length_error(message) {}
};

// DeviceBuf owners are created synchronously on the endpoint thread. Its per-call cap
// applies in addition to any outer prepared-resource ledger attached to the CUDA allocator.
inline thread_local std::shared_ptr<std::size_t> ucc_device_headroom;

class UccDeviceCapacityScope {
 public:
  UccDeviceCapacityScope(std::size_t cap, std::size_t resident) {
    if (resident > cap) throw UccDeviceLimitError("DF-UCC v2 CUDA source exceeds device budget");
    if (ucc_device_headroom) throw std::logic_error("nested DF-UCC v2 CUDA capacity scope");
    ucc_device_headroom = std::make_shared<std::size_t>(cap - resident);
  }
  UccDeviceCapacityScope(const UccDeviceCapacityScope&) = delete;
  UccDeviceCapacityScope& operator=(const UccDeviceCapacityScope&) = delete;
  ~UccDeviceCapacityScope() { ucc_device_headroom.reset(); }
};

inline std::size_t ucc_device_available(std::size_t cap) {
  return ucc_device_headroom ? std::min(cap, *ucc_device_headroom) : cap;
}

// A failed CUDA allocation unwinds the reservation. Moving a buffer transfers its charge.
class UccDeviceCharge {
 public:
  UccDeviceCharge() = default;
  explicit UccDeviceCharge(std::size_t bytes) : owner_(ucc_device_headroom), bytes_(bytes) {
    if (!owner_) return;
    if (bytes > *owner_) throw UccDeviceLimitError("DF-UCC v2 CUDA allocation exceeds device budget");
    *owner_ -= bytes;
  }
  UccDeviceCharge(const UccDeviceCharge&) = delete;
  UccDeviceCharge& operator=(const UccDeviceCharge&) = delete;
  UccDeviceCharge(UccDeviceCharge&&) noexcept = default;
  UccDeviceCharge& operator=(UccDeviceCharge&& other) noexcept {
    if (this != &other) {
      reset();
      owner_ = std::move(other.owner_);
      bytes_ = other.bytes_;
    }
    return *this;
  }
  ~UccDeviceCharge() { reset(); }
  void reset() noexcept {
    if (owner_) *owner_ += bytes_;
    owner_.reset();
  }

 private:
  std::shared_ptr<std::size_t> owner_;
  std::size_t bytes_{};
};

// A rejected optional warmup leaves the original FP64 seed available. All of its
// buffers have unwound before the mandatory solve starts.
template <class Warmup>
inline bool try_ucc_warmup(Warmup&& warmup) {
  try {
    warmup();
    return true;
  } catch (const UccDeviceLimitError&) {
    return false;
  } catch (const std::bad_alloc&) {
    return false;
  }
}

struct UccIterationBudget {
  std::size_t ppl_tile_bytes{}, diis_bytes{};
};

// The PPL tile only overlaps the ladder-resident intermediates. DIIS history can
// remain live in every phase, including max-abs reductions and Jacobi projection.
inline std::optional<UccIterationBudget> ucc_iteration_budget(
    const SpinBlockedDims& d, std::size_t elem_size, std::size_t amp_elems,
    std::size_t base, std::size_t cap) {
  using posthf::checked_add;
  using posthf::checked_mul;
  const auto residual = ucc_residual_capacity(d, elem_size);
  const auto blocks = amp_elems ? std::min<std::size_t>(1 + (amp_elems - 1) / 256, 1024) : 0;
  const auto reduction = blocks ? (blocks + 1) * sizeof(double) + sizeof(int) : sizeof(double);
  const auto max_ov = std::max(checked_mul(d.oa, d.va), checked_mul(d.ob, d.vb));
  const auto projection = checked_mul(checked_mul(max_ov, max_ov), elem_size);
  const auto min_scratch = std::max({residual.non_ppl_peak_bytes, reduction, projection,
      checked_add(residual.ppl_resident_bytes, residual.min_ppl_tile_bytes)});
  if (base > cap || min_scratch > cap - base) return std::nullopt;
  const auto room = cap - base;
  const auto spare = room - min_scratch;
  const auto diis = residual.min_ppl_tile_bytes ? spare - spare / 3 : spare;
  const auto tile = residual.min_ppl_tile_bytes ? room - diis - residual.ppl_resident_bytes : 0;
  return UccIterationBudget{tile, diis};
}

inline int ucc_diis_space(int requested, std::size_t packed_bytes, std::size_t budget) {
  if (!packed_bytes) return 0;
  // One packed linear combination lives alongside at most two packed vectors per history slot.
  const auto vectors = budget / packed_bytes;
  const auto slots = vectors ? (vectors - 1) / 2 : 0;
  return slots < 2 ? 0 : static_cast<int>(
      std::min<std::size_t>(slots, static_cast<std::size_t>(std::max(2, requested))));
}

inline std::size_t ucc_lambda_ppl_q_tile(std::size_t q, std::size_t panel, std::size_t elem_size,
                                         std::size_t available) {
  if (!q || !panel) return 0;
  const auto pair_bytes = posthf::checked_mul(posthf::checked_mul(2, panel), elem_size);
  // Two transposed tau panels coexist with two panels per Q in the adjoint tile.
  if (available / pair_bytes < 2)
    throw UccDeviceLimitError("DF-ULambda v2 CUDA minimum PPL tile exceeds device memory budget");
  return std::min({q, std::size_t{128}, available / pair_bytes - 1});
}

inline std::size_t ucc_f12_elements(const FixedF12Coupling* f12) {
  if (!f12) return 0;
  std::size_t total = 0;
  for (auto n : {f12->w_vo_aaaa.size(), f12->w_vo_bbbb.size(), f12->w_vo_abab.size(),
                 f12->w_vo_abba.size(), f12->w_vv_aaaa.size(), f12->w_vv_abab.size(),
                 f12->w_vv_bbbb.size(), f12->c_f12_aaaa.size(), f12->c_f12_abab.size(),
                 f12->c_f12_bbbb.size()}) total = posthf::checked_add(total, n);
  return total;
}

}  // namespace generativeqc::cc::v2_detail
