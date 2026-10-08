#pragma once

/// @file
/// @brief trsm: cta, sg_left, blocked, vendor. evidence: docs/perf/trsm.md @ingroup api_selection_ops

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::trsm {

struct Cta : select::NoFields<"cta"> {};          ///< trsm_native_v1_dispatch, order <= 32
struct SgLeft : select::NoFields<"sg_left"> {};   ///< trsm_native_sg_left_dispatch: Left, order <= 32, sg 32
struct Blocked : select::NoFields<"blocked"> {};  ///< trsm_native_blocked + the public gemm, any order
struct Vendor : select::NoFields<"vendor"> {};    ///< backend::trsm_vendor; needs the level-3 library

using TrsmChoice = std::variant<Cta, SgLeft, Blocked, Vendor>;  ///< fieldless: buckets, wg ladder, outer block derived

template <class T>  /// Every compiled choice, once, in tie-break order (§6.3): native first, vendor last.
constexpr auto candidates() { return select::all_of<TrsmChoice>(); }

inline constexpr select::OpSpec spec{Op::trsm, select::Lib::level3};  ///< last resort (§5.5): blocked, vendor (CPU)

inline constexpr std::array<std::string_view, 5> key_names{  ///< q: B.cols (Left) or B.rows (Right)
    "side:exact", "trans:exact", "order:log:2", "q:log", "batch:log"};  // ConjTrans is T; no uplo/diag key; work ~ order^2 q batch

/// The tuner's grid; the transcriber (tuned/README.md) spells the same grid.
/// evidence: docs/design/flat-kernel-selection-phase3-plan.md (§3)
inline constexpr std::array<int, 18> grid_order{1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
                                                768, 1024};
inline constexpr std::array<int, 12> grid_q{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::trsm
