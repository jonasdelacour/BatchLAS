#pragma once

// syrk's selection vocabulary (docs/design/flat-kernel-selection.md §12), header-only.

#include "../../select/select.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <variant>

namespace batchlas::ops::syrk {

// Fieldless: gram's tile width and triangular's aligned leg are derived in the kernels.
struct Gram : select::NoFields<"gram"> {};              // syrk_gram_tiles: one tile covers C
struct Triangular : select::NoFields<"triangular"> {};  // syrk_triangular_tiles<float>
struct Vendor : select::NoFields<"vendor"> {};          // backend::syrk_vendor

using SyrkChoice = std::variant<Gram, Triangular, Vendor>;

// Tie-break order (§6.3). Triangular stages 128-bit float packets: a float candidate only (R7).
template <class T>
constexpr auto candidates() {
    if constexpr (std::is_same_v<T, float>) return std::array<SyrkChoice, 3>{Gram{}, Triangular{}, Vendor{}};
    else return std::array<SyrkChoice, 2>{Gram{}, Vendor{}};
}

inline constexpr std::array<std::string_view, 3> last_resort{"triangular", "gram", "vendor"};
inline constexpr select::Rules rules{last_resort};

// k: op(A)'s inner extent; form: the old squareish ratio as an axis; trans N|T|C. Work ~ n^2 k batch.
inline constexpr std::array<std::string_view, 5> key_names{"form:exact", "trans:exact", "n:log:2", "k:log",
                                                           "batch:log"};

inline std::string_view form_of(std::int64_t a, std::int64_t b) {
    return 2 * std::min(a, b) >= std::max(a, b) ? "sq" : (a > 2 * b ? "tall" : "wide");
}

// Both sides of every old threshold (n 128|129 and each 128-tile band edge,
// k 7|8, every band's batch floor); the transcriber spells the same grid.
inline constexpr std::array<int, 30> grid_n{1,   2,   4,   8,   16,  32,   64,   128,  129,  256,
                                            257, 384, 385, 512, 513, 640,  641,  768,  769,  896,
                                            897, 1024, 1025, 1152, 1153, 1536, 1537, 2176, 2177, 4096};
// N and T take the full product; C only the n axis at k = batch = 1 (vendor-first everywhere).
inline constexpr std::array<std::string_view, 3> grid_trans{"N", "T", "C"};
inline constexpr std::array<int, 6> grid_k{1, 7, 8, 64, 512, 4096};
inline constexpr std::array<int, 17> grid_batch{1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 15, 16, 26, 27, 128, 1024, 32768};

// Both tile kernels put the batch in a grid dimension, and triangular its T(T+1)/2 tiles
// (T = ceil(n/128)) in another; each is capped at 65535 work-groups, past which the launch throws.
inline constexpr std::int64_t kMaxGridBatch = 65535;
inline constexpr std::int64_t kMaxGridTiles = 65535;
inline constexpr int kTriangularWg = 256;  // syrk_triangular_tiles' 16 x 16 work-group
inline constexpr std::int64_t triangular_groups(std::int64_t n, std::int64_t tile) {
    const std::int64_t t = (n + tile - 1) / tile;
    return t * (t + 1) / 2;
}

// gram's launch for real T (syrk_gram_tiles.hh): 4x4 thread tiles meeting the triangle, KC = 32.
inline constexpr int gram_tile(std::int64_t n) { return n <= 32 ? 32 : (n <= 64 ? 64 : 128); }
inline constexpr int gram_threads(std::int64_t n) {
    const int lanes = gram_tile(n) / 4;
    return (lanes * (lanes + 1) / 2 + 31) / 32 * 32;
}
template <class T>
constexpr std::int64_t gram_slm_bytes(std::int64_t n) {
    return std::int64_t(32) * gram_tile(n) * std::int64_t(sizeof(T));
}

}  // namespace batchlas::ops::syrk
