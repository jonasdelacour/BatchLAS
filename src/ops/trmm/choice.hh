#pragma once

/// @file
/// @brief trmm: triangular, expand, vendor. evidence: docs/perf/level3.md @ingroup selection_ops

#include "../../select/select.hh"

#include <array>
#include <cstdint>
#include <string_view>
#include <variant>

namespace batchlas::ops::trmm {

struct Triangular : select::NoFields<"triangular"> {};  ///< trmm_triangular_tiles: Side::Left; tile grid <= 65535
struct Expand : select::NoFields<"expand"> {};          ///< expand_triangular + the public gemm; scratch must fit
struct Vendor : select::NoFields<"vendor"> {};          ///< backend::trmm_vendor (cublas?trmm loop)

using TrmmChoice = std::variant<Triangular, Expand, Vendor>;  ///< row tile (trmm_row_tile) derived; natives: CUDA GPU

template <class T>  /// Every compiled choice, in tie-break order (§6.3): triangular, expand, vendor.
constexpr auto candidates() { return select::all_of<TrmmChoice>(); }

inline constexpr std::array<std::string_view, 3> last_resort{"expand", "triangular", "vendor"};  ///< nothing runs
// §5.5: Expand serves both sides, Triangular Left only.
inline constexpr select::OpSpec spec{Op::trmm, select::Lib::level3, {last_resort}};  ///< op, vendor library, rules

inline constexpr std::int64_t kMaxGridBatch = 65535;  ///< both natives: batch in grid z; 65536 throws
inline constexpr std::int64_t kMaxGridTiles = 65535;  ///< triangular: row x column tiles in grid y
inline constexpr int kNativeWg = 256;  ///< widest native work-group: the 128-row tile, the expansion

/// Keys: side exact; order = A.rows(), weight 2 (work ~ order^2 q batch); q = B.cols() (Left) or B.rows() (Right).
inline constexpr std::array<std::string_view, 4> key_names{"side:exact", "order:log:2", "q:log", "batch:log"};

// The transcription grid (tuned/README.md); the old rule read only the side.
inline constexpr std::array<int, 9> grid_order{1, 16, 32, 64, 128, 256, 512, 1024, 2048};  ///< order axis
inline constexpr std::array<int, 5> grid_q{1, 16, 128, 1024, 4096};  ///< q axis of the grid
inline constexpr std::array<int, 4> grid_batch{1, 128, 1024, 32768};  ///< batch axis of the grid

}  // namespace batchlas::ops::trmm
