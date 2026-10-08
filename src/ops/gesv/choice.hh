#pragma once

/// @file
/// @brief gesv: tiny, blocked. `evidence: docs/perf/lu.md#the-fused-gesv-tier` @ingroup api_selection_ops

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::gesv {

struct Tiny : select::NoFields<"tiny"> {};        ///< gesv_tiny_dispatch: fused LU factor + solve (buckets derived)
struct Blocked : select::NoFields<"blocked"> {};  ///< public getrf + public getrs; any homogeneous batch

using GesvChoice = std::variant<Tiny, Blocked>;  ///< no vendor family: a `vendor` pin warns and runs Auto

template <class T>  /// Every compiled choice, once, in tie-break order (§6.3).
constexpr auto candidates() { return select::all_of<GesvChoice>(); }

inline constexpr select::OpSpec spec{Op::gesv};  ///< no vendor library; last resort (§5.5): blocked

inline constexpr std::array<std::string_view, 2> key_names{"n:log:3", "nrhs:log"};  ///< work: 2n^3/3 + 2 n^2 nrhs

/// Both sides of every old threshold (16|17, 32|33, nrhs 4|5), log-spaced elsewhere; the transcriber spells it too.
inline constexpr std::array<int, 29> grid_n{1, 2, 3, 4, 6, 8, 12, 16, 17, 20, 24, 28, 32, 33, 40, 48, 64,
                                            96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096};
inline constexpr std::array<int, 8> grid_nrhs{1, 2, 4, 5, 8, 16, 64, 256};

}  // namespace batchlas::ops::gesv
