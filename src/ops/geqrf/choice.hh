#pragma once

// geqrf's selection vocabulary (docs/design/flat-kernel-selection.md#phase-5-geqrf), header-only.

#include "../../select/select.hh"

#include <array>
#include <cstdint>
#include <string_view>
#include <variant>

namespace batchlas::ops::geqrf {

// All fieldless: buckets, packing, nb and panel leaf are derived in the drivers.
struct Tiny : select::NoFields<"tiny"> {};        // geqrf_tiny_dispatch: square, registers
struct Cta : select::NoFields<"cta"> {};          // geqrf_cta_dispatch: the whole panel in SLM
struct Blocked : select::NoFields<"blocked"> {};  // geqrf_blocked_dispatch + the public gemm
struct Vendor : select::NoFields<"vendor"> {};    // backend::geqrf_vendor

using GeqrfChoice = std::variant<Tiny, Cta, Blocked, Vendor>;

// Every compiled choice, once, in tie-break order (§6.3): narrowest first, vendor last.
template <class T>
constexpr auto candidates() {
    return std::array<GeqrfChoice, 4>{Tiny{}, Cta{}, Blocked{}, Vendor{}};
}

// Generality order (§5.5): Blocked runs every m >= n shape on a GPU, Vendor everything else.
inline constexpr std::array<std::string_view, 2> last_resort{"blocked", "vendor"};
inline constexpr select::Rules rules{last_resort};

// form: sq | tall (m > n) | wide (m < n); aspect = max(m,n) / min(m,n). Work ~ n^3 * aspect.
inline constexpr std::array<std::string_view, 3> key_names{"form:exact", "n:log:3", "aspect:log"};

template <class Int>
constexpr std::int64_t aspect_of(Int m, Int n) {
    const std::int64_t lo = m < n ? m : n, hi = m < n ? n : m;
    return lo < 1 ? 1 : hi / lo;
}

// Rows: sq x grid_n, tall x grid_n x grid_aspect, wide x grid_wide_n x grid_wide_aspect; both
// sides of every old threshold (the transcriber, tuned/README.md, spells the same grid).
inline constexpr std::array<int, 52> grid_n{1,  2,  3,  4,  5,  6,  8,   9,   10,  11,  12,  14,  16,  17,  20,  21,  22,
                                            23, 24, 28, 31, 32, 33, 40,  47,  48,  49,  56,  63,  64,  75,  76,  80,  96,
                                            97, 112, 128, 160, 192, 224, 255, 256, 288, 384, 512, 768, 1024, 1536, 2048,
                                            3072, 4096, 8192};
inline constexpr std::array<int, 11> grid_aspect{1, 2, 3, 4, 5, 7, 8, 12, 16, 64, 256};
inline constexpr std::array<int, 4> grid_wide_n{2, 64, 1024, 8192};
inline constexpr std::array<int, 3> grid_wide_aspect{1, 4, 64};

}  // namespace batchlas::ops::geqrf
