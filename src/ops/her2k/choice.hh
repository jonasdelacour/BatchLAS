#pragma once

/// @file
/// @brief her2k: triangular, fold, vendor. evidence: docs/perf/level3.md @ingroup api_selection_ops

#include <src/select/select.hh>

#include <array>
#include <complex>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <variant>

namespace batchlas::ops::her2k {

struct Triangular : select::NoFields<"triangular"> {};  ///< syr2k_triangular_tiles<T, Conjugate=true>; cfloat only
struct Fold : select::NoFields<"fold"> {};      ///< one gemm of alpha op(A) op(B)^H into scratch + two-sided fold, chunked
struct Vendor : select::NoFields<"vendor"> {};  ///< backend::her2k_vendor (per-item ?her2k loop)

using Her2kChoice = std::variant<Triangular, Fold, Vendor>;  ///< fieldless: tile, chunk, gemm kernel derived

/// Every compiled choice, in tie-break order. No complex<double> triangular: its accumulators alone
/// fill the register file, and its four staged tiles exceed 48 KB of local memory.
template <class T>
constexpr auto candidates() {
    if constexpr (std::is_same_v<T, std::complex<float>>) return select::all_of<Her2kChoice>();
    else return std::array<Her2kChoice, 2>{Fold{}, Vendor{}};
}

inline constexpr std::array<std::string_view, 3> last_resort{"fold", "triangular", "vendor"};  ///< nothing in the row runs
inline constexpr select::OpSpec spec{Op::her2k, select::Lib::level3, {last_resort}};  ///< op, vendor library, rules

/// Keys: n = C's order, weight 2 (work ~ n^2 k batch); k = op(A)'s inner extent; batch.
inline constexpr std::array<std::string_view, 3> key_names{"n:log:2", "k:log", "batch:log"};

// The transcription grid: 127|128 and 1|2 straddle the old rule (fold iff batch >= 2 or n >= 128).
inline constexpr std::array<int, 14> grid_n{1, 2, 4, 8, 16, 32, 64, 127, 128, 256, 512, 1024, 2048, 4096};
inline constexpr std::array<int, 5> grid_k{1, 8, 64, 512, 4096};  ///< k axis
inline constexpr std::array<int, 7> grid_batch{1, 2, 3, 128, 1024, 8192, 32768};  ///< batch axis

inline constexpr int kFoldWg = 256;  ///< fold_hermitian's work-group (expand_group_shape)
inline constexpr int kTriangularWg = 256;  ///< syr2k_triangular_tiles' 16 x 16 work-group
template <class T>
constexpr std::int64_t triangular_slm_bytes() {  ///< four staged 8 x 128 operand tiles
    return std::int64_t(4) * 8 * 128 * std::int64_t(sizeof(T));
}

}  // namespace batchlas::ops::her2k
