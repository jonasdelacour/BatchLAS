#pragma once

// herk's selection vocabulary (docs/design/flat-kernel-selection.md §12), header-only.

#include "../../select/select.hh"

#include <array>
#include <cstdint>
#include <string_view>
#include <variant>

namespace batchlas::ops::herk {

// Fieldless: the fold's ld and gram's tile and thread tile are derived from n.
struct Fold : select::NoFields<"fold"> {};      // public gemm into scratch + accumulate_hermitian<false>
struct Gram : select::NoFields<"gram"> {};      // syrk_gram_tiles<T, conj>: one tile covers C (n <= 128)
struct Vendor : select::NoFields<"vendor"> {};  // backend::herk_vendor (cublas?herk / cblas_?herk loop)

using HerkChoice = std::variant<Fold, Gram, Vendor>;

// Complex only; declaration order. Fold before gram: a transcribed row lists the natives in this order, and
// the fold won every gram shape measured (docs/perf/level3.md#herk-on-the-gram-tile-kernel).
template <class T>
constexpr auto candidates() { return select::all_of<HerkChoice>(); }

inline constexpr std::array<std::string_view, 3> last_resort{"fold", "gram", "vendor"};
inline constexpr select::OpSpec spec{Op::herk, select::Lib::level3, {last_resort}};

// n = C's order, k = op(A)'s inner extent. Work ~ n^2 k batch.
inline constexpr std::array<std::string_view, 3> key_names{"n:log:2", "k:log", "batch:log"};

// The transcriber's grid: n 768|769 and batch 3|4 straddle the old rule, n 128|129 gram's tile.
inline constexpr std::array<int, 18> grid_n{1,   2,   4,   8,   16,  32,   64,   127,  128,
                                            129, 256, 512, 767, 768, 769, 1024, 2048, 4096};
inline constexpr std::array<int, 5> grid_k{1, 8, 64, 512, 4096};
inline constexpr std::array<int, 10> grid_batch{1, 2, 3, 4, 5, 8, 128, 1024, 8192, 32768};

inline constexpr std::int64_t kMaxGridBatch = 65535;  // the fold's grid z, gram's grid y
inline constexpr int kFoldWg = 256;  // accumulate_hermitian's group (expand_group_shape)

// gram's launch for complex T (syrk_gram_tiles.hh): a 128 tile takes an 8-wide thread tile.
inline constexpr int gram_tile(std::int64_t n) { return n <= 32 ? 32 : (n <= 64 ? 64 : 128); }
inline constexpr int gram_threads(std::int64_t n) {
    const int lanes = gram_tile(n) / (gram_tile(n) == 128 ? 8 : 4);
    return (lanes * (lanes + 1) / 2 + 31) / 32 * 32;
}
template <class T>
constexpr std::int64_t gram_slm_bytes(std::int64_t n) {
    return std::int64_t(32) * gram_tile(n) * std::int64_t(sizeof(T));
}

}  // namespace batchlas::ops::herk
