#pragma once

/// @file
/// @brief syr2k: triangular, vendor. evidence: docs/perf/level3.md @ingroup selection_ops

#include "../../select/select.hh"

#include <array>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <variant>

namespace batchlas::ops::syr2k {

struct Triangular : select::NoFields<"triangular"> {};  ///< syr2k_triangular_tiles<float>, CUDA GPU; not ConjTrans
struct Vendor : select::NoFields<"vendor"> {};          ///< backend::syr2k_vendor (per-item ?syr2k loop)

using Syr2kChoice = std::variant<Triangular, Vendor>;  ///< fieldless: the aligned/predicated leg is derived

template <class T>  /// Every compiled choice, in tie-break order: triangular (float only, R7), vendor.
constexpr auto candidates() {
    if constexpr (std::is_same_v<T, float>)
        return std::array<Syr2kChoice, 2>{Triangular{}, Vendor{}};
    else
        return std::array<Syr2kChoice, 1>{Vendor{}};  // the tile kernel is float-only (R7)
}

inline constexpr std::array<std::string_view, 2> last_resort{"triangular", "vendor"};  ///< triangular: any GPU shape
inline constexpr select::OpSpec spec{Op::syr2k, select::Lib::level3, {last_resort}};  ///< op, vendor library, rules

/// Keys: n = C's order, weight 2 (work ~ n^2 k batch); k = op(A)'s inner extent; batch.
inline constexpr std::array<std::string_view, 3> key_names{"n:log:2", "k:log", "batch:log"};

// The transcribed grid; the old rule turned on batch 1|2 alone (tuned/README.md).
inline constexpr std::array<int, 13> grid_n{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};  ///< n axis
inline constexpr std::array<int, 5> grid_k{1, 8, 64, 512, 4096};  ///< k axis of the grid
inline constexpr std::array<int, 7> grid_batch{1, 2, 3, 128, 1024, 8192, 32768};  ///< batch axis of the grid

// The tile kernel puts the batch in grid z (SYCL dim 0) and its tile list in grid y: 65535 each.
inline constexpr std::int64_t kMaxGridBatch = 65535;  ///< batch ceiling of the tile kernel
inline constexpr std::int64_t kMaxGridTiles = 65535;  ///< tile-count ceiling of the tile kernel

}  // namespace batchlas::ops::syr2k
