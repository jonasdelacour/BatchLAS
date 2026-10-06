#pragma once

// syr2k's selection vocabulary (docs/design/flat-kernel-selection.md §12), header-only.

#include "../../select/select.hh"

#include <array>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <variant>

namespace batchlas::ops::syr2k {

// Fieldless: the tile kernel derives its aligned/predicated leg from the operands.
struct Triangular : select::NoFields<"triangular"> {};  // detail::syr2k_triangular_tiles<float>, CUDA
struct Vendor : select::NoFields<"vendor"> {};          // backend::syr2k_vendor

using Syr2kChoice = std::variant<Triangular, Vendor>;

template <class T>
constexpr auto candidates() {
    if constexpr (std::is_same_v<T, float>)
        return std::array<Syr2kChoice, 2>{Triangular{}, Vendor{}};
    else
        return std::array<Syr2kChoice, 1>{Vendor{}};  // the tile kernel is float-only (R7)
}

inline constexpr std::array<std::string_view, 2> last_resort{"triangular", "vendor"};
inline constexpr select::Rules rules{last_resort};  // §5.5: triangular runs every homogeneous GPU shape.

// n = C's order, k = op(A)'s inner extent. Work ~ n^2 k batch.
inline constexpr std::array<std::string_view, 3> key_names{"n:log:2", "k:log", "batch:log"};

// The transcribed grid; the old rule turned on batch 1|2 alone (tuned/README.md).
inline constexpr std::array<int, 13> grid_n{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
inline constexpr std::array<int, 5> grid_k{1, 8, 64, 512, 4096};
inline constexpr std::array<int, 7> grid_batch{1, 2, 3, 128, 1024, 8192, 32768};

// The tile kernel puts the batch in grid z (SYCL dim 0) and its tile list in grid y: 65535 each.
inline constexpr std::int64_t kMaxGridBatch = 65535;
inline constexpr std::int64_t kMaxGridTiles = 65535;

}  // namespace batchlas::ops::syr2k
