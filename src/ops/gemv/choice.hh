#pragma once

/// @file
/// @brief gemv: cta, direct, vendor. evidence: docs/perf/gemv.md @ingroup api_selection_ops

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::gemv {

struct Cta : select::NoFields<"cta"> {};        ///< sycl_gemv::gemv_native_cta, sub-group per output; Trans/ConjTrans
struct Direct : select::NoFields<"direct"> {};  ///< sycl_gemv::gemv_native_direct, item per output; every device
struct Vendor : select::NoFields<"vendor"> {};  ///< backend::gemv_vendor; any shape, needs the library

using GemvChoice = std::variant<Cta, Direct, Vendor>;  ///< fieldless: body and segment width W derived in the drivers

template <class T>  /// Every compiled choice, in tie-break order: the old ladder.
constexpr auto candidates() { return select::all_of<GemvChoice>(); }

/// can_run's device terms (host-callable for tests). Cta body 3 is reqd_sub_group_size(32).
inline bool device_allows(const GemvChoice& c, const select::Device& d, bool transposed) {
    if (std::holds_alternative<Cta>(c)) return d.is_gpu && d.has_sg32 && transposed;
    if (std::holds_alternative<Vendor>(c)) return d.has_vendor;
    return true;
}

inline constexpr std::array<std::string_view, 2> last_resort{"vendor", "direct"};  ///< CPU: vendor, else direct (no GPU gate)
inline constexpr select::OpSpec spec{Op::gemv, select::Lib::level3, {last_resort}};  ///< op, vendor library, rules

/// Keys: trans exact (ConjTrans folds to T); out, red: y and x lengths (swap with trans). Work ~ out red batch.
inline constexpr std::array<std::string_view, 4> key_names{"trans:exact", "out:log", "red:log", "batch:log"};

/// Both sides of every old threshold plus a coarse log grid; the transcriber spells the same.
inline constexpr std::array<int, 8> grid_out{1, 8, 64, 255, 256, 1024, 4096, 32768};
inline constexpr std::array<int, 11> grid_red{1, 8, 32, 63, 64, 128, 352, 353, 1024, 4096, 32768};
inline constexpr std::array<int, 8> grid_batch{1, 16, 128, 319, 320, 1024, 8192, 32768};

}  // namespace batchlas::ops::gemv
