#pragma once

// posv's selection vocabulary (flat-kernel-selection-phase3-plan.md §1.1), header-only.

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::posv {

// All fieldless: Tiny's N/NC/NR buckets and Cta's nb/wg are derived from the shape in the driver.
struct Tiny : select::NoFields<"tiny"> {};        // posv_tiny_dispatch: fused factor + solve
struct Cta : select::NoFields<"cta"> {};          // public potrf + potrs_fused_dispatch
struct Blocked : select::NoFields<"blocked"> {};  // public potrf + two public trsm

using PosvChoice = std::variant<Tiny, Cta, Blocked>;

// Tie-break order (§6.3). No vendor family: a `vendor` pin warns and falls back to Auto.
template <class T>
constexpr auto candidates() {
    return std::array<PosvChoice, 3>{Tiny{}, Cta{}, Blocked{}};
}

inline constexpr std::array<select::Alias, 3> aliases{{  // legacy spellings until phase 5
    {"native:tiny", "tiny"},
    {"native:cta", "cta"},
    {"native:blocked", "blocked"},
}};
// Blocked runs every shape the validator accepts; a failing child reports its own error.
inline constexpr std::array<std::string_view, 1> last_resort{"blocked"};
inline constexpr select::Rules rules{aliases, last_resort};

inline constexpr std::array<std::string_view, 4> key_names{  // work: n^3/3 + 2 n^2 nrhs
    "uplo:exact", "n:log:3", "nrhs:log", "batch:log"};

// potrf's grid_n within [1, 1024]; tools/transcribe/posv_transcribe.cc spells the same grid.
inline constexpr std::array<int, 33> grid_n{1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56,
    64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 640, 768, 1024};
inline constexpr std::array<int, 6> grid_nrhs{1, 2, 4, 8, 16, 64};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::posv
