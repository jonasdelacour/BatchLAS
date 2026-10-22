#pragma once

/// @file
/// @brief potri: fused, blocked.
/// `evidence: docs/perf/potri.md#potri-selection` @ingroup api_selection_ops

#include <src/select/select.hh>

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::potri {

struct Fused : select::NoFields<"fused"> {};      ///< identity in the workspace + the one-launch potrs kernel; n <= 8
struct Blocked : select::NoFields<"blocked"> {};  ///< identity in the workspace + two public trsm; any homogeneous batch

using PotriChoice = std::variant<Fused, Blocked>;  ///< fieldless: the solve derives its own launch shape

template <class T>  /// Tie-break order (§6.3): native first, the general family last. No vendor family exists.
constexpr auto candidates() { return select::all_of<PotriChoice>(); }

inline constexpr select::OpSpec spec{Op::potri, select::Lib::none};  ///< no library; the trsm children report errors

inline constexpr std::array<std::string_view, 2> key_names{"n:log:3", "batch:log"};  ///< work ~ n^3 batch

inline constexpr std::array<int, 18> grid_n{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::potri
