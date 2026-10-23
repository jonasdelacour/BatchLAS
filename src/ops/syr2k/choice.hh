#pragma once

/// @file
/// @brief syr2k: triangular, vendor. evidence: docs/perf/level3.md @ingroup api_selection_ops

#include <src/select/select.hh>

#include <array>
#include <complex>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <variant>

namespace batchlas::ops::syr2k {

struct Triangular : select::NoFields<"triangular"> {};  ///< syr2k_triangular_tiles<T>, GPU; real ConjTrans runs as Trans
struct Vendor : select::NoFields<"vendor"> {};          ///< backend::syr2k_vendor (per-item ?syr2k loop)

using Syr2kChoice = std::variant<Triangular, Vendor>;  ///< fieldless: the aligned/predicated leg is derived

/// Every compiled choice, in tie-break order; complex<double> has no triangular (registers, 64 KB SLM).
template <class T>
constexpr auto candidates() {
    if constexpr (std::is_same_v<T, std::complex<double>>) return std::array<Syr2kChoice, 1>{Vendor{}};
    else return select::all_of<Syr2kChoice>();
}

/// Is the per-item vendor loop linked for (B, T)? rocblas.cc wraps the real ?syr2k only.
template <Backend B, class T>
inline constexpr bool vendor_wired = select::has_library<B>(select::Lib::level3) &&
                                     !(B == Backend::ROCM && !std::is_floating_point_v<T>);

template <class T>
constexpr std::int64_t triangular_slm_bytes() {  ///< four staged 8 x 128 operand tiles
    return std::int64_t(4) * 8 * 128 * std::int64_t(sizeof(T));
}

inline constexpr std::array<std::string_view, 2> last_resort{"triangular", "vendor"};  ///< triangular: any GPU shape
inline constexpr select::OpSpec spec{Op::syr2k, select::Lib::level3, {last_resort}};  ///< op, vendor library, rules

/// Keys: n = C's order, weight 2 (work ~ n^2 k batch); k = op(A)'s inner extent; batch.
inline constexpr std::array<std::string_view, 3> key_names{"n:log:2", "k:log", "batch:log"};

// The transcribed grid; the transcribed rule turned on batch 1|2 alone (tuned/README.md).
inline constexpr std::array<int, 13> grid_n{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};  ///< n axis
inline constexpr std::array<int, 5> grid_k{1, 8, 64, 512, 4096};  ///< k axis of the grid
inline constexpr std::array<int, 7> grid_batch{1, 2, 3, 128, 1024, 8192, 32768};  ///< batch axis of the grid

inline constexpr std::int64_t kMaxGridBatch = 65535;  ///< batch ceiling of the tile kernel
inline constexpr std::int64_t kMaxGridTiles = 65535;  ///< tile-count ceiling of the tile kernel

}  // namespace batchlas::ops::syr2k
