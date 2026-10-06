#pragma once

// symm's selection vocabulary (docs/design/flat-kernel-selection.md §12), header-only.

#include "../../select/select.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <variant>

namespace batchlas::ops::symm {

// Both fieldless: the old router chose no knob (the expansion's ld and tile are derived).
struct Expand : select::NoFields<"expand"> {};  // expand_mirrored into scratch + the public gemm
struct Vendor : select::NoFields<"vendor"> {};  // backend::symm_vendor (cublas?symm / cblas_?symm loop)

using SymmChoice = std::variant<Expand, Vendor>;

template <class T>
constexpr auto candidates() {
    return std::array<SymmChoice, 2>{Expand{}, Vendor{}};
}

inline constexpr std::array<std::string_view, 2> last_resort{"expand", "vendor"};
inline constexpr select::Rules rules{last_resort};

// C's m x n and batch (side was never read); form lines the old squareish test up with an axis.
inline constexpr std::array<std::string_view, 4> key_names{"form:exact", "m:log", "n:log", "batch:log"};

inline std::string_view form_of(std::int64_t a, std::int64_t b) {
    return 2 * std::min(a, b) >= std::max(a, b) ? "sq" : (a > 2 * b ? "tall" : "wide");
}

// The transcriber's grid: 255|256 and 3|4 straddle the old thresholds (flat-kernel-selection.md §12).
inline constexpr std::array<std::string_view, 3> grid_form{"sq", "tall", "wide"};
inline constexpr std::array<int, 14> grid_mn{1, 2, 4, 8, 16, 32, 64, 128, 255, 256, 512, 1024, 2048, 4096};
inline constexpr std::array<int, 9> grid_batch{1, 2, 3, 4, 8, 128, 1024, 8192, 32768};

// expand_mirrored puts the batch in SYCL dim 0 = CUDA grid z.
inline constexpr std::int64_t kMaxGridBatch = 65535;
inline constexpr int kExpandWg = 256;  // expand_mirrored's work-group: kMirrorGroupCols x kMirrorTile

}  // namespace batchlas::ops::symm
