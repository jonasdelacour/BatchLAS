#pragma once

/// @file
/// @brief symm: expand, vendor. evidence: docs/perf/level3.md @ingroup api_selection_ops

#include "../../select/select.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <variant>

namespace batchlas::ops::symm {

struct Expand : select::NoFields<"expand"> {};  ///< expand_mirrored into scratch + the public gemm; scratch must fit
struct Vendor : select::NoFields<"vendor"> {};  ///< backend::symm_vendor (cublas?symm / cblas_?symm loop)

using SymmChoice = std::variant<Expand, Vendor>;  ///< fieldless: expansion ld and gemm tile derived; native: CUDA GPU

template <class T>  /// Every compiled choice, in tie-break order (§6.3): the expansion, then the vendor loop.
constexpr auto candidates() { return select::all_of<SymmChoice>(); }

inline constexpr std::array<std::string_view, 2> last_resort{"expand", "vendor"};  ///< nothing in the row runs
inline constexpr select::OpSpec spec{Op::symm, select::Lib::level3, {last_resort}};  ///< op, vendor library, rules

/// Keys: form exact (the old squareish test as an axis); m, n: C's extents; batch. Side is no key (never read).
inline constexpr std::array<std::string_view, 4> key_names{"form:exact", "m:log", "n:log", "batch:log"};

/// sq if 2 min >= max, else tall (a > 2b) or wide: the old squareish ratio as an exact key.
inline std::string_view form_of(std::int64_t a, std::int64_t b) {
    return 2 * std::min(a, b) >= std::max(a, b) ? "sq" : (a > 2 * b ? "tall" : "wide");
}

// The transcriber's grid: 255|256 and 3|4 straddle the old thresholds (flat-kernel-selection.md §12).
inline constexpr std::array<std::string_view, 3> grid_form{"sq", "tall", "wide"};  ///< form axis of the grid
inline constexpr std::array<int, 14> grid_mn{1, 2, 4, 8, 16, 32, 64, 128, 255, 256, 512, 1024, 2048, 4096};  ///< m, n
inline constexpr std::array<int, 9> grid_batch{1, 2, 3, 4, 8, 128, 1024, 8192, 32768};  ///< batch axis of the grid

inline constexpr std::int64_t kMaxGridBatch = 65535;  ///< expand_mirrored puts the batch in SYCL dim 0 = CUDA grid z
inline constexpr int kExpandWg = 256;  ///< expand_mirrored's work-group: kMirrorGroupCols x kMirrorTile

}  // namespace batchlas::ops::symm
