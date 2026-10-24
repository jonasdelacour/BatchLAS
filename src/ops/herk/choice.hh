// SPDX-License-Identifier: MIT
#pragma once

/// @file
/// @brief herk: gram, triangular, fold, vendor. evidence: docs/perf/level3.md @ingroup api_selection_ops

#include <src/select/select.hh>

#include <array>
#include <complex>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <variant>

namespace batchlas::ops::herk {

struct Gram : select::NoFields<"gram"> {};      ///< syrk_gram_tiles<T, Conjugate=true>: one tile covers C; n <= 128
struct Triangular : select::NoFields<"triangular"> {};  ///< syrk_triangular_tiles<T, Conjugate=true>; cfloat only
struct Fold : select::NoFields<"fold"> {};      ///< public gemm of the whole n x n product into scratch + fold, chunked
struct Vendor : select::NoFields<"vendor"> {};  ///< backend::herk_vendor (per-item ?herk loop)

using HerkChoice = std::variant<Gram, Triangular, Fold, Vendor>;  ///< fieldless: tile, chunk, gemm kernel derived

/// Every compiled choice, in tie-break order. No complex<double> triangular: its 64 accumulators
/// alone fill the 255-register file.
template <class T>
constexpr auto candidates() {
    if constexpr (std::is_same_v<T, std::complex<float>>) return select::all_of<HerkChoice>();
    else return std::array<HerkChoice, 3>{Gram{}, Fold{}, Vendor{}};
}

inline constexpr std::array<std::string_view, 4> last_resort{"fold", "triangular", "gram", "vendor"};  ///< nothing in the row runs
inline constexpr select::OpSpec spec{Op::herk, select::Lib::level3, {last_resort}};  ///< op, vendor library, rules

/// Keys: trans N|C exact; n = C's order, weight 2 (work ~ n^2 k batch); k = op(A)'s inner extent.
inline constexpr std::array<std::string_view, 4> key_names{"trans:exact", "n:log:2", "k:log", "batch:log"};

// The transcription grid: 768|769 and 3|4 straddle the old rule (fold iff batch >= 4 and n <= 768).
inline constexpr std::array<std::string_view, 2> grid_trans{"N", "C"};  ///< trans axis
inline constexpr std::array<int, 15> grid_n{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 769, 1024, 2048, 4096};
inline constexpr std::array<int, 5> grid_k{1, 8, 64, 512, 4096};  ///< k axis
inline constexpr std::array<int, 9> grid_batch{1, 2, 3, 4, 8, 128, 1024, 8192, 32768};  ///< batch axis

// gram's launch for complex T (syrk_gram_tiles.hh): 4x4 thread tiles, 8x8 at the 128 tile; KC = 32.
inline constexpr int gram_tile(std::int64_t n) { return n <= 32 ? 32 : (n <= 64 ? 64 : 128); }  ///< tile width
inline constexpr int gram_threads(std::int64_t n) {  ///< lanes meeting the triangle, warp-rounded
    const int lanes = gram_tile(n) / (gram_tile(n) == 128 ? 8 : 4);
    return (lanes * (lanes + 1) / 2 + 31) / 32 * 32;
}
template <class T>
constexpr std::int64_t gram_slm_bytes(std::int64_t n) {  ///< the staged [KC][tile] operand
    return std::int64_t(32) * gram_tile(n) * std::int64_t(sizeof(T));
}

inline constexpr int kFoldWg = 256;  ///< fold_hermitian's work-group (expand_group_shape)
inline constexpr int kTriangularWg = 256;  ///< syrk_triangular_tiles' 16 x 16 work-group
template <class T>
constexpr std::int64_t triangular_slm_bytes() {  ///< two staged 8 x 128 operand tiles
    return std::int64_t(2) * 8 * 128 * std::int64_t(sizeof(T));
}

}  // namespace batchlas::ops::herk
