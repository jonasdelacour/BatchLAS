#pragma once

// gesvd's selection vocabulary (flat-kernel-selection.md §4.2, phase 5), header-only.

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::gesvd {

struct Jacobi : select::NoFields<"jacobi"> {};    // gesvdj_cta: one-sided Jacobi, max(m, n) <= 64
struct Cta : select::NoFields<"cta"> {};          // gesvd_cta: normal equations + syev_cta, max(m, n) <= 32
struct Blocked : select::NoFields<"blocked"> {};  // gesvd_blocked: bidiagonalisation + bdsqr, or syev_blocked
struct Vendor : select::NoFields<"vendor"> {};    // backend::gesvd_vendor

using GesvdChoice = std::variant<Jacobi, Cta, Blocked, Vendor>;

template <class T>
constexpr auto candidates() {  // tie-break order (§6.3): native first, vendor last
    return std::array<GesvdChoice, 4>{Jacobi{}, Cta{}, Blocked{}, Vendor{}};
}

inline constexpr std::array<std::string_view, 2> last_resort{"blocked", "vendor"};
inline constexpr select::Rules rules{last_resort};  // §5.5: Blocked serves the most GPU shapes, Vendor the rest

// herm N|L|U; vec none|all|thin from the CANONICAL jobs (the only job facts a gate reads).
inline constexpr std::array<std::string_view, 4> key_names{"herm:exact", "vec:exact", "m:log:1.5", "n:log:1.5"};

// Both sides of every driver ceiling (32 for cta, 64 for jacobi) plus a coarse log grid;
// tools/transcribe/gesvd_transcribe.cc spells the same grid.
inline constexpr std::array<int, 15> grid_mn{1, 2, 4, 8, 16, 24, 32, 33, 48, 64, 65, 128, 256, 512, 1024};
inline constexpr std::array<std::string_view, 3> grid_herm{"N", "L", "U"};
inline constexpr std::array<std::string_view, 3> grid_vec{"none", "all", "thin"};

}  // namespace batchlas::ops::gesvd
