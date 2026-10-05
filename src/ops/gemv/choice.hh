#pragma once

// gemv's selection vocabulary (docs/design/flat-select-p5/gemv.md), header-only.

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::gemv {

// All fieldless: the body split and segment width W are derived in the drivers.
struct Cta : select::NoFields<"cta"> {};        // sycl_gemv::gemv_native_cta, Trans/ConjTrans only
struct Direct : select::NoFields<"direct"> {};  // sycl_gemv::gemv_native_direct, every device
struct Vendor : select::NoFields<"vendor"> {};  // backend::gemv_vendor

using GemvChoice = std::variant<Cta, Direct, Vendor>;

template <class T>
constexpr auto candidates() {
    return std::array<GemvChoice, 3>{Cta{}, Direct{}, Vendor{}};  // the old ladder
}

// can_run's device terms (host-callable for tests). Cta body 3 is reqd_sub_group_size(32).
inline bool device_allows(const GemvChoice& c, const select::Device& d, bool transposed) {
    if (std::holds_alternative<Cta>(c)) return d.is_gpu && d.has_sg32 && transposed;
    if (std::holds_alternative<Vendor>(c)) return d.has_vendor_blas;
    return true;
}

inline constexpr std::array<std::string_view, 2> last_resort{"vendor", "direct"};
inline constexpr select::Rules rules{last_resort};  // CPU: vendor, else Direct (no GPU gate)

// out/red: y's and x's lengths (they swap with trans); ConjTrans folds to T. Work ~ out red batch.
inline constexpr std::array<std::string_view, 4> key_names{"trans:exact", "out:log", "red:log", "batch:log"};

// Both sides of every old threshold plus a coarse log grid; the transcriber spells the same.
inline constexpr std::array<int, 8> grid_out{1, 8, 64, 255, 256, 1024, 4096, 32768};
inline constexpr std::array<int, 11> grid_red{1, 8, 32, 63, 64, 128, 352, 353, 1024, 4096, 32768};
inline constexpr std::array<int, 8> grid_batch{1, 16, 128, 319, 320, 1024, 8192, 32768};

}  // namespace batchlas::ops::gemv
