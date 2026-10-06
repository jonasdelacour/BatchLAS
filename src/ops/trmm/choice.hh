#pragma once

// trmm's selection vocabulary (docs/design/flat-kernel-selection.md §4.2), header-only.

#include "../../select/select.hh"

#include <array>
#include <cstdint>
#include <string_view>
#include <variant>

namespace batchlas::ops::trmm {

// Fieldless: the row tile (trmm_row_tile, BATCHLAS_TRMM_TILE_M) stays derived; the old router chose no knob.
struct Triangular : select::NoFields<"triangular"> {};  // detail::trmm_triangular_tiles, Side::Left
struct Expand : select::NoFields<"expand"> {};          // expand_triangular + the public gemm
struct Vendor : select::NoFields<"vendor"> {};          // backend::trmm_vendor (cublas?trmm loop)

using TrmmChoice = std::variant<Triangular, Expand, Vendor>;

template <class T>
constexpr auto candidates() {
    return std::array<TrmmChoice, 3>{Triangular{}, Expand{}, Vendor{}};
}

inline constexpr std::array<std::string_view, 3> last_resort{"expand", "triangular", "vendor"};
inline constexpr select::Rules rules{last_resort};  // §5.5: Expand serves both sides, Triangular Left only.

inline constexpr std::int64_t kMaxGridBatch = 65535;  // both natives: batch in grid z; 65536 throws
inline constexpr std::int64_t kMaxGridTiles = 65535;  // triangular: row x column tiles in grid y
inline constexpr int kNativeWg = 256;  // widest native work-group: the 128-row tile, the expansion

// order = A.rows(); q = B.cols() (Left) or B.rows() (Right). Work ~ order^2 q batch.
inline constexpr std::array<std::string_view, 4> key_names{"side:exact", "order:log:2", "q:log", "batch:log"};

// The transcription grid (tuned/README.md); the old rule read only the side.
inline constexpr std::array<int, 9> grid_order{1, 16, 32, 64, 128, 256, 512, 1024, 2048};
inline constexpr std::array<int, 5> grid_q{1, 16, 128, 1024, 4096};
inline constexpr std::array<int, 4> grid_batch{1, 128, 1024, 32768};

}  // namespace batchlas::ops::trmm
