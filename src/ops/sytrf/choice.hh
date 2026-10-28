#pragma once

/// @file
/// @brief sytrf: cta, wg, blocked, vendor. `evidence: docs/perf/sytrf.md` @ingroup api_selection_ops

#include <src/select/select.hh>

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::sytrf {

struct Cta : select::NoFields<"cta"> {};        ///< sytrf_cta_dispatch: one sub-group per matrix, n <= 32, local memory
struct Wg : select::NoFields<"wg"> {};          ///< sytrf_wg_dispatch: one work-group per matrix, unblocked, any n
struct Blocked : select::NoFields<"blocked"> {};  ///< sytrf_blocked_dispatch: ?lasyf panels + the public gemm, any n
struct Vendor : select::NoFields<"vendor"> {};  ///< backend::sytrf_vendor: cuSOLVER ?sytrf per item, LAPACKE ?sytrf

using SytrfChoice = std::variant<Cta, Wg, Blocked, Vendor>;  ///< natives: GPU, sub-group 32, homogeneous, not NETLIB

template <class T>  /// Every compiled choice, once, in tie-break order (§6.3).
constexpr auto candidates() { return select::all_of<SytrfChoice>(); }

inline constexpr std::array<std::string_view, 2> kLastResort{"wg", "vendor"};

inline constexpr select::OpSpec spec{Op::sytrf, select::Lib::solver, select::Rules{kLastResort}};

inline constexpr std::array<std::string_view, 2> key_names{"n:log:3", "batch:log"};  ///< work ~ n^3 batch / 3

/// The transcriber's grid: both sides of the CTA ceiling (32|33), log-spaced elsewhere.
inline constexpr std::array<int, 16> grid_n{1, 2, 4, 8, 12, 16, 24, 32, 33, 48, 64, 128, 256, 512, 1024, 2048};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::sytrf
