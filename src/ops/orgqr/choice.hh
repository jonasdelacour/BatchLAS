#pragma once

/// @file
/// @brief orgqr: blocked, vendor. `evidence: docs/perf/qr.md#the-shipped-orgqr-ceiling` @ingroup selection_ops

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::orgqr {

struct Blocked : select::NoFields<"blocked"> {};  ///< orgqr_blocked_dispatch: identity + the public ormqr; GPU, n <= m
struct Vendor : select::NoFields<"vendor"> {};    ///< backend::orgqr_vendor; any shape (n > m, CPU, heterogeneous)

using OrgqrChoice = std::variant<Blocked, Vendor>;  ///< fieldless: orgqr_nb is derived from type and min(m, n)

template <class T>  /// Every compiled choice, in tie-break order (§6.3).
constexpr auto candidates() { return select::all_of<OrgqrChoice>(); }

inline constexpr std::array<std::string_view, 2> last_resort{"vendor", "blocked"};  ///< §5.5: vendor is the general one
inline constexpr select::OpSpec spec{Op::orgqr, select::Lib::factorization, {last_resort}};  ///< op, library, rules

inline constexpr std::array<std::string_view, 2> key_names{"m:log", "n:log:2"};  ///< work ~ m n^2; no batch key

/// Both sides of the old 512 ceiling, a coarse log grid elsewhere, n <= m (transcriber: tuned/README.md).
inline constexpr std::array<int, 17> grid{1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 513, 768, 1024, 2048, 4096,
                                          8192};

}  // namespace batchlas::ops::orgqr
