#pragma once

/// @file
/// @brief getrf: tiny, cta, blocked, vendor. evidence: docs/perf/lu.md @ingroup selection_ops

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::getrf {

struct Tiny : select::NoFields<"tiny"> {};        ///< getrf_tiny_dispatch: register-resident, n <= 32 (cdouble 16)
struct Cta : select::NoFields<"cta"> {};          ///< getrf_cta_dispatch: SLM-resident panel, n <= the SLM ceiling
struct Blocked : select::NoFields<"blocked"> {};  ///< getrf_blocked_dispatch + the public gemm and trsm
struct Vendor : select::NoFields<"vendor"> {};    ///< backend::getrf_vendor (cuBLAS / rocSOLVER / LAPACKE)

using GetrfChoice = std::variant<Tiny, Cta, Blocked, Vendor>;  ///< natives: GPU, sub-group 32, square, not NETLIB

template <class T>  /// Every compiled choice, once, in tie-break order (§6.3).
constexpr auto candidates() { return select::all_of<GetrfChoice>(); }

inline constexpr select::OpSpec spec{Op::getrf, select::Lib::factorization};  ///< §5.5: blocked, then vendor (CPU)

inline constexpr std::array<std::string_view, 2> key_names{"n:log:3", "batch:log"};  ///< work ~ n^3 batch

/// Both sides of every old threshold (4|5, 7|8|9, 16|17, 24|25, 32|33, 255|256, 511|512; batch
/// 255|256), a coarse log grid elsewhere. The transcriber (tuned/README.md) spells the same grid.
inline constexpr std::array<int, 28> grid_n{1,  2,  3,  4,   5,   6,   7,   8,   9,   12,  16,  17,  24,  25,
                                            32, 33, 48, 64,  96,  128, 192, 255, 256, 384, 511, 512, 768, 1024};
inline constexpr std::array<int, 10> grid_batch{1, 8, 64, 128, 255, 256, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::getrf
