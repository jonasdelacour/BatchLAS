#pragma once

// getrs's selection vocabulary (docs/design/flat-kernel-selection.md §4.2), header-only.

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::getrs {

// All fieldless: nb, accumulator width, work-group and permutation spelling are derived in the drivers.
struct Cta : select::NoFields<"cta"> {};          // getrs_fused_dispatch: permute + both solves, one kernel
struct Blocked : select::NoFields<"blocked"> {};  // getrs_blocked_dispatch: laswp + two public trsm
struct Vendor : select::NoFields<"vendor"> {};    // backend::getrs_vendor

using GetrsChoice = std::variant<Cta, Blocked, Vendor>;

template <class T>
constexpr auto candidates() {
    return std::array<GetrsChoice, 3>{Cta{}, Blocked{}, Vendor{}};
}

inline constexpr std::array<select::Alias, 2> aliases{{
    {"native:cta", "cta"},
    {"native:blocked", "blocked"},
}};
inline constexpr std::array<std::string_view, 2> last_resort{"blocked", "vendor"};
inline constexpr select::Rules rules{aliases, last_resort};

// The keys the old predicates read (not transA). Work ~ n^2 nrhs batch.
inline constexpr std::array<std::string_view, 3> key_names{"n:log:2", "nrhs:log", "batch:log"};

// A log grid plus both sides of every old threshold (n 31/32; nrhs 2/3, 4/5, 63/64, 127/128;
// batch 127/128); tools/transcribe/getrs_transcribe.cc spells the same grid.
inline constexpr std::array<int, 18> grid_n{1, 2, 4, 8, 16, 24, 31, 32, 48, 64, 96, 128, 192, 256, 384, 512,
                                            768, 1024};
inline constexpr std::array<int, 15> grid_nrhs{1, 2, 3, 4, 5, 8, 16, 32, 63, 64, 127, 128, 256, 512, 1024};
inline constexpr std::array<int, 8> grid_batch{1, 16, 127, 128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::getrs
