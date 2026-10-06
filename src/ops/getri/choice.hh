#pragma once

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::getri {

struct Blocked : select::NoFields<"blocked"> {};  // getri_blocked_dispatch: P into C + two public trsm
struct Vendor : select::NoFields<"vendor"> {};    // backend::getri_vendor

using GetriChoice = std::variant<Blocked, Vendor>;  // both fieldless: the driver derives wg

template <class T>  // tie-break order (§6.3): native first
constexpr auto candidates() { return select::all_of<GetriChoice>(); }

// Generality order (§5.5): Vendor also runs CPU, NETLIB and heterogeneous batches.
inline constexpr std::array<std::string_view, 2> last_resort{"vendor", "blocked"};
inline constexpr select::OpSpec spec{Op::getri, select::Lib::factorization, {last_resort}};

// Work ~ n^3 batch; the old router read only n (batch >= 1 was a correctness term).
inline constexpr std::array<std::string_view, 2> key_names{"n:log:3", "batch:log"};

// Log grid + both sides of the old edges (float 128, cfloat 256); the transcriber (tuned/README.md) spells it.
inline constexpr std::array<int, 24> grid_n{1,   2,   3,   4,   6,   8,   12,  16,  24,  32,   48,   64,
                                            96,  127, 128, 192, 255, 256, 384, 512, 768, 1024, 2048, 4096};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::getri
