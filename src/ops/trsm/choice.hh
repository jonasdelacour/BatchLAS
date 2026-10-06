#pragma once

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::trsm {

// All fieldless: buckets, work-group ladder and outer block are derived in the drivers (phase 4).
struct Cta : select::NoFields<"cta"> {};          // trsm_native_v1_dispatch, order <= 32
struct SgLeft : select::NoFields<"sg_left"> {};   // trsm_native_sg_left_dispatch, Side::Left, order <= 32
struct Blocked : select::NoFields<"blocked"> {};  // trsm_native_blocked + the public gemm
struct Vendor : select::NoFields<"vendor"> {};    // backend::trsm_vendor

using TrsmChoice = std::variant<Cta, SgLeft, Blocked, Vendor>;

// Every compiled choice, once, in tie-break order (§6.3): native first, vendor last.
template <class T>
constexpr auto candidates() { return select::all_of<TrsmChoice>(); }

inline constexpr select::OpSpec spec{Op::trsm, select::Lib::level3};  // §5.5: Blocked runs every GPU shape, Vendor everything else (CPU).

// q: B.cols (Left) or B.rows (Right); ConjTrans folds to T; no uplo/diag key. Work ~ order^2 q batch.
inline constexpr std::array<std::string_view, 5> key_names{
    "side:exact", "trans:exact", "order:log:2", "q:log", "batch:log"};

// The tuner's grid (plan §3); the transcriber (tuned/README.md) spells the same grid.
inline constexpr std::array<int, 18> grid_order{1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
                                                768, 1024};
inline constexpr std::array<int, 12> grid_q{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::trsm
