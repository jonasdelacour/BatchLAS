#pragma once

/// @file
/// @brief gesvd: jacobi, cta, blocked, vendor. evidence: docs/perf/gesvd.md @ingroup api_selection_ops

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::gesvd {

struct Jacobi : select::NoFields<"jacobi"> {};    ///< gesvdj_cta: one-sided Jacobi, max(m, n) <= 64, not Hermitian
struct Cta : select::NoFields<"cta"> {};          ///< gesvd_cta: normal equations + syev_cta, max(m, n) <= 32, not thin
struct Blocked : select::NoFields<"blocked"> {};  ///< gesvd_blocked: bidiagonalisation + bdsdc, or syev_blocked (Lower)
struct Vendor : select::NoFields<"vendor"> {};    ///< backend::gesvd_vendor; needs the solver library

using GesvdChoice = std::variant<Jacobi, Cta, Blocked, Vendor>;  ///< natives need a GPU; cta, blocked: complex only if Hermitian

template <class T>  /// Every compiled choice, in tie-break order (§6.3): native first, vendor last.
constexpr auto candidates() { return select::all_of<GesvdChoice>(); }

inline constexpr select::OpSpec spec{Op::gesvd, select::Lib::solver};  ///< last resort (§5.5): blocked, vendor

/// Keys: herm N|L|U and vec none|all|thin (from the CANONICAL jobs) exact; m, n log weight 1.5.
inline constexpr std::array<std::string_view, 4> key_names{"herm:exact", "vec:exact", "m:log:1.5", "n:log:1.5"};

/// Both sides of every driver ceiling (32 cta, 64 jacobi) plus a coarse log grid, as tuned/README.md's transcriber.
inline constexpr std::array<int, 15> grid_mn{1, 2, 4, 8, 16, 24, 32, 33, 48, 64, 65, 128, 256, 512, 1024};
inline constexpr std::array<std::string_view, 3> grid_herm{"N", "L", "U"};
inline constexpr std::array<std::string_view, 3> grid_vec{"none", "all", "thin"};

}  // namespace batchlas::ops::gesvd
