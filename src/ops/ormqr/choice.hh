#pragma once

/// @file
/// @brief ormqr: blocked, vendor. evidence: docs/perf/qr.md @ingroup selection_ops

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::ormqr {

struct Blocked : select::NoFields<"blocked"> {};  ///< ormqr_blocked (larft + level-3 WY updates); GPU, no complex Trans
struct Vendor : select::NoFields<"vendor"> {};    ///< backend::ormqr_vendor; no complex Trans either

using OrmqrChoice = std::variant<Blocked, Vendor>;  ///< fieldless: WY width and gemm/trmm spelling are derived

template <class T>  /// Every compiled choice, in tie-break order (§6.3).
constexpr auto candidates() { return select::all_of<OrmqrChoice>(); }

inline constexpr select::OpSpec spec{Op::ormqr, select::Lib::factorization};  ///< last resort (§5.5): blocked, vendor

inline constexpr std::array<std::string_view, 6> key_names{  ///< m: Q's order, k: reflectors, q: C's other extent
    "side:exact", "trans:exact", "m:log", "k:log", "q:log", "batch:log"};  // T and C stay apart; work ~ m k q batch

/// The transcription grid; the transcriber (tuned/README.md) spells the same one (k <= m).
inline constexpr std::array<int, 4> grid_m{1, 8, 64, 512};
inline constexpr std::array<int, 3> grid_q{1, 32, 1024};
inline constexpr std::array<int, 3> grid_batch{128, 2048, 32768};

}  // namespace batchlas::ops::ormqr
