#pragma once

// gemv's selection vocabulary (flat-kernel-selection.md §4.2; docs/design/flat-select-p5/gemv.md),
// header-only.

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::gemv {

// All fieldless: Direct's body 1/2/4 split, Cta's body 3/5 split and the segment width W are
// derived in the drivers from the shape, the device and BATCHLAS_GEMV_SEGT.
struct Cta : select::NoFields<"cta"> {};        // sycl_gemv::gemv_native_cta, Trans/ConjTrans only
struct Direct : select::NoFields<"direct"> {};  // sycl_gemv::gemv_native_direct, every device
struct Vendor : select::NoFields<"vendor"> {};  // backend::gemv_vendor

using GemvChoice = std::variant<Cta, Direct, Vendor>;

// Every compiled choice, once, in tie-break order (§6.3): the old capability ladder.
template <class T>
constexpr auto candidates() {
    return std::array<GemvChoice, 3>{Cta{}, Direct{}, Vendor{}};
}

inline constexpr std::array<select::Alias, 2> aliases{{  // legacy spellings until phase 5
    {"native:cta", "cta"},
    {"native:direct", "direct"},
}};
// A CPU device has no table, so this order decides it: the vendor (netlib) where compiled, as
// before, else Direct, the one native tier without a GPU gate.
inline constexpr std::array<std::string_view, 2> last_resort{"vendor", "direct"};
inline constexpr select::Rules rules{aliases, last_resort};

// out/red: y's and x's lengths (they swap with trans); ConjTrans folds to T. Work ~ out red batch.
inline constexpr std::array<std::string_view, 4> key_names{"trans:exact", "out:log", "red:log", "batch:log"};

// Both sides of every old threshold (red 63/64, 352/353; out 255/256; batch 319/320) plus a
// coarse log grid; tools/transcribe/gemv_transcribe.cc spells the same grid.
inline constexpr std::array<int, 8> grid_out{1, 8, 64, 255, 256, 1024, 4096, 32768};
inline constexpr std::array<int, 11> grid_red{1, 8, 32, 63, 64, 128, 352, 353, 1024, 4096, 32768};
inline constexpr std::array<int, 8> grid_batch{1, 16, 128, 319, 320, 1024, 8192, 32768};

}  // namespace batchlas::ops::gemv
