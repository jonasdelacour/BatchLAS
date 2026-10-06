#pragma once

/// @file
/// @brief geqrf: tiny, cta, blocked, vendor. evidence: docs/perf/qr.md @ingroup selection_ops

#include "../../select/select.hh"

#include <array>
#include <cstdint>
#include <string_view>
#include <variant>

namespace batchlas::ops::geqrf {

struct Tiny : select::NoFields<"tiny"> {};        ///< geqrf_tiny_dispatch: square, registers, n <= the tiny ceiling
struct Cta : select::NoFields<"cta"> {};          ///< geqrf_cta_dispatch: the whole panel in SLM (geqrf_cta_fits)
struct Blocked : select::NoFields<"blocked"> {};  ///< geqrf_blocked_dispatch + the public gemm, any m >= n
struct Vendor : select::NoFields<"vendor"> {};    ///< backend::geqrf_vendor; needs the factorization library

using GeqrfChoice = std::variant<Tiny, Cta, Blocked, Vendor>;  ///< fieldless: buckets, packing, nb, leaf derived

/// Every compiled choice, once, in tie-break order (§6.3): narrowest first, vendor last.
template <class T>
constexpr auto candidates() { return select::all_of<GeqrfChoice>(); }

inline constexpr select::OpSpec spec{Op::geqrf, select::Lib::factorization};  ///< last resort (§5.5): blocked, vendor

/// Keys: form sq | tall (m > n) | wide (m < n), exact; n log weight 3; aspect log. Work ~ n^3 * aspect.
inline constexpr std::array<std::string_view, 3> key_names{"form:exact", "n:log:3", "aspect:log"};

template <class Int>  /// The aspect key: max(m, n) / min(m, n), 1 for an empty view.
constexpr std::int64_t aspect_of(Int m, Int n) {
    const std::int64_t lo = m < n ? m : n, hi = m < n ? n : m;
    return lo < 1 ? 1 : hi / lo;
}

/// Rows: sq x grid_n, tall x grid_n x grid_aspect, wide x grid_wide_n x grid_wide_aspect; both
/// sides of every old threshold (the transcriber, tuned/README.md, spells the same grid).
inline constexpr std::array<int, 52> grid_n{1,  2,  3,  4,  5,  6,  8,   9,   10,  11,  12,  14,  16,  17,  20,  21,  22,
                                            23, 24, 28, 31, 32, 33, 40,  47,  48,  49,  56,  63,  64,  75,  76,  80,  96,
                                            97, 112, 128, 160, 192, 224, 255, 256, 288, 384, 512, 768, 1024, 1536, 2048,
                                            3072, 4096, 8192};
inline constexpr std::array<int, 11> grid_aspect{1, 2, 3, 4, 5, 7, 8, 12, 16, 64, 256};
inline constexpr std::array<int, 4> grid_wide_n{2, 64, 1024, 8192};
inline constexpr std::array<int, 3> grid_wide_aspect{1, 4, 64};

}  // namespace batchlas::ops::geqrf
