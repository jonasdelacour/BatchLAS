#pragma once

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::ormqr {

// Fieldless: Blocked's WY width and gemm/trmm spelling are derived, never chosen by the router.
struct Blocked : select::NoFields<"blocked"> {};  // ormqr_blocked (larft + level-3 WY updates)
struct Vendor : select::NoFields<"vendor"> {};    // backend::ormqr_vendor

using OrmqrChoice = std::variant<Blocked, Vendor>;

template <class T>
constexpr auto candidates() { return select::all_of<OrmqrChoice>(); }

inline constexpr select::OpSpec spec{Op::ormqr, select::Lib::factorization};  // §5.5: Blocked runs every GPU shape but complex Trans, Vendor the rest.

// m = order of Q, k = reflectors, q = C's other extent (work ~ m k q batch); T and C stay apart.
inline constexpr std::array<std::string_view, 6> key_names{
    "side:exact", "trans:exact", "m:log", "k:log", "q:log", "batch:log"};

// The transcription grid; the transcriber (tuned/README.md) spells the same one (k <= m).
inline constexpr std::array<int, 4> grid_m{1, 8, 64, 512};
inline constexpr std::array<int, 3> grid_q{1, 32, 1024};
inline constexpr std::array<int, 3> grid_batch{128, 2048, 32768};

}  // namespace batchlas::ops::ormqr
