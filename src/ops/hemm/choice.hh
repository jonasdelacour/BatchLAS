// SPDX-License-Identifier: MIT
#pragma once

/// @file
/// @brief hemm: expand, vendor. evidence: docs/perf/level3.md @ingroup api_selection_ops

#include <src/select/select.hh>

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::hemm {

struct Expand : select::NoFields<"expand"> {};  ///< expand_mirrored<Conjugate> into scratch + the public gemm, chunked
struct Vendor : select::NoFields<"vendor"> {};  ///< backend::hemm_vendor (cublas?hemm / cblas_?hemm loop)

using HemmChoice = std::variant<Expand, Vendor>;  ///< fieldless: expansion ld, chunk and gemm tile derived

template <class T>  /// Every compiled choice, in tie-break order: the expansion, then the vendor loop.
constexpr auto candidates() { return select::all_of<HemmChoice>(); }

inline constexpr std::array<std::string_view, 2> last_resort{"expand", "vendor"};  ///< nothing in the row runs
inline constexpr select::OpSpec spec{Op::hemm, select::Lib::level3, {last_resort}};  ///< op, vendor library, rules

/// Keys: m, n = C's extents; batch (side is no key: the old rule read max(m, n)).
inline constexpr std::array<std::string_view, 3> key_names{"m:log", "n:log", "batch:log"};
inline constexpr std::array<int, 15> grid_mn{1, 2, 4, 8, 16, 32, 64, 128, 255, 256, 512, 1024, 2048, 4096, 8192};  ///< 255|256 straddles the rule
inline constexpr std::array<int, 9> grid_batch{1, 2, 3, 4, 8, 128, 1024, 8192, 32768};  ///< batch axis

inline constexpr int kExpandWg = 256;  ///< expand_mirrored's work-group: kMirrorGroupCols x kMirrorTile

}  // namespace batchlas::ops::hemm
