#pragma once

/// @file
/// @brief posv: tiny, cta, blocked.
/// `evidence: docs/perf/potrf.md#posv-selection-since-flat-kernel-selection-phase-3` @ingroup api_selection_ops

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::posv {

struct Tiny : select::NoFields<"tiny"> {};        ///< posv_tiny_dispatch: fused factor + solve, small n and nrhs
struct Cta : select::NoFields<"cta"> {};          ///< public potrf + potrs_fused_dispatch; B fits in SLM
struct Blocked : select::NoFields<"blocked"> {};  ///< public potrf + two public trsm; any homogeneous batch

using PosvChoice = std::variant<Tiny, Cta, Blocked>;  ///< fieldless: buckets, nb and wg are derived in the driver

template <class T>  /// Tie-break order (§6.3). No vendor family: a `vendor` pin warns and runs Auto.
constexpr auto candidates() { return select::all_of<PosvChoice>(); }

inline constexpr select::OpSpec spec{Op::posv, select::Lib::none};  ///< no library; blocked's children report errors

inline constexpr std::array<std::string_view, 4> key_names{  ///< work: n^3/3 + 2 n^2 nrhs
    "uplo:exact", "n:log:3", "nrhs:log", "batch:log"};

/// potrf's grid_n within [1, 1024]; the transcriber (tuned/README.md) spells the same grid.
inline constexpr std::array<int, 33> grid_n{1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56,
    64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 640, 768, 1024};
inline constexpr std::array<int, 6> grid_nrhs{1, 2, 4, 8, 16, 64};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::posv
