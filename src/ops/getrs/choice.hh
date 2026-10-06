#pragma once

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
constexpr auto candidates() { return select::all_of<GetrsChoice>(); }

inline constexpr select::OpSpec spec{Op::getrs, select::Lib::factorization};

inline constexpr std::array<std::string_view, 3> key_names{"n:log:2", "nrhs:log", "batch:log"};  // work ~ n^2 nrhs batch

// A log grid plus both sides of every old threshold (n 31/32; nrhs 2/3, 4/5, 63/64, 127/128;
// batch 127/128); the transcriber (tuned/README.md) spells the same grid.
inline constexpr std::array<int, 18> grid_n{1, 2, 4, 8, 16, 24, 31, 32, 48, 64, 96, 128, 192, 256, 384, 512,
                                            768, 1024};
inline constexpr std::array<int, 15> grid_nrhs{1, 2, 3, 4, 5, 8, 16, 32, 63, 64, 127, 128, 256, 512, 1024};
inline constexpr std::array<int, 8> grid_batch{1, 16, 127, 128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::getrs
