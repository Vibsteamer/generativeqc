#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace generativeqc::integrals::detail {

// Deterministic static partition of [0, total) across nthreads workers.
// Thread tid owns contiguous ascending indices [begin, end).
inline std::pair<std::size_t, std::size_t> static_radial_partition(
    std::size_t total, unsigned nthreads, unsigned tid) noexcept {
  if (nthreads == 0 || tid >= nthreads) return {total, total};
  const std::size_t base = total / nthreads;
  const std::size_t rem = total % nthreads;
  const std::size_t begin = static_cast<std::size_t>(tid) * base +
                            std::min<std::size_t>(tid, rem);
  const std::size_t end = begin + base + (tid < rem ? 1U : 0U);
  return {begin, end};
}

// Execute `fn(tid, cancelled)` across `nthreads` workers, capturing the first
// thrown exception, signalling peers to stop early via `cancelled`, joining all
// started threads cleanly, and rethrowing on the caller thread.
template <class Fn>
void run_ecp_workers(unsigned nthreads, Fn&& fn) {
  if (nthreads <= 1U) {
    std::atomic<bool> cancelled{false};
    fn(0U, cancelled);
    return;
  }
  std::atomic<bool> cancelled{false};
  std::mutex error_mutex;
  std::exception_ptr first_error;
  auto record_error = [&]() noexcept {
    cancelled.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(error_mutex);
    if (!first_error) first_error = std::current_exception();
  };
  std::vector<std::thread> pool;
  pool.reserve(nthreads);
  try {
    for (unsigned t = 0; t < nthreads; ++t) {
      pool.emplace_back([&, t]() noexcept {
        try {
          fn(t, cancelled);
        } catch (...) {
          record_error();
        }
      });
    }
  } catch (...) {
    record_error();
  }
  for (auto& th : pool) {
    if (th.joinable()) th.join();
  }
  if (first_error) std::rethrow_exception(first_error);
}

}  // namespace generativeqc::integrals::detail
