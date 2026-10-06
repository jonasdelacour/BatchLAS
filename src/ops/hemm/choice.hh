#pragma once

// hemm's selection vocabulary (docs/design/flat-kernel-selection.md §12), header-only.

#include "../../select/select.hh"

#include <array>
#include <cstdint>
#include <string_view>
#include <variant>

namespace batchlas::ops::hemm {

struct Expand : select::NoFields<"expand"> {};  // expand_mirrored<conj> into scratch + the public gemm
struct Vendor : select::NoFields<"vendor"> {};  // backend::hemm_vendor (cublas?hemm / cblas_?hemm loop)

using HemmChoice = std::variant<Expand, Vendor>;

// Complex only (BLAS has no real ?hemm); fieldless: the expansion's ld and tile are derived.
template <class T>
constexpr auto candidates() {
    return std::array<HemmChoice, 2>{Expand{}, Vendor{}};
}

inline constexpr std::array<std::string_view, 2> last_resort{"expand", "vendor"};
inline constexpr select::Rules rules{last_resort};

// order = A's order (m on the left, n on the right); q = C's other extent. Work ~ order^2 q batch.
inline constexpr std::array<std::string_view, 3> key_names{"order:log:2", "q:log", "batch:log"};

// The transcriber's grid: 255|256 (max extent) and 3|4 (batch) straddle the old rule.
inline constexpr std::array<int, 14> grid_extent{1, 2, 4, 8, 16, 32, 64, 128, 255, 256, 512, 1024, 2048, 4096};
inline constexpr std::array<int, 10> grid_batch{1, 2, 3, 4, 5, 8, 128, 1024, 8192, 32768};

inline constexpr std::int64_t kMaxGridBatch = 65535;  // expand_mirrored: batch in grid z
inline constexpr int kExpandWg = 256;  // expand_mirrored's work-group: kMirrorGroupCols x kMirrorTile

}  // namespace batchlas::ops::hemm
