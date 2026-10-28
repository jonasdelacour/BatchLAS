// SPDX-License-Identifier: MIT
#pragma once

/// @file
/// @brief potrs: fused, trsm.
/// `evidence: docs/perf/potrs.md#potrs-selection` @ingroup api_selection_ops

#include <src/select/select.hh>

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::potrs {

struct Fused : select::NoFields<"fused"> {};  ///< potrs_fused_dispatch: both solves in one launch; B fits in SLM
struct Trsm : select::NoFields<"trsm"> {};    ///< two public trsm calls, each choosing its own kernel; any homogeneous batch

using PotrsChoice = std::variant<Fused, Trsm>;  ///< fieldless: the fused solve derives its own launch shape

template <class T>  /// Tie-break order (§6.3): native first, the general family last. No vendor family exists.
constexpr auto candidates() { return select::all_of<PotrsChoice>(); }

inline constexpr std::array<std::string_view, 1> last_resort{"trsm"};  ///< runs every homogeneous call
inline constexpr select::OpSpec spec{Op::potrs, select::Lib::none, {last_resort}};  ///< the trsm children report errors

inline constexpr std::array<std::string_view, 4> key_names{  ///< work ~ n^2 nrhs batch
    "uplo:exact", "n:log:2", "nrhs:log", "batch:log"};

/// The fused accumulator buckets (1, 2, 4, 8) and their midpoints; above 8 only `trsm` runs.
inline constexpr std::array<int, 17> grid_n{1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024};
inline constexpr std::array<int, 6> grid_nrhs{1, 2, 3, 4, 6, 8};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::potrs
