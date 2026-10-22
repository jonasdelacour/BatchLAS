#pragma once

/// @file
/// @brief sysv: cta, compose. `evidence: docs/perf/sytrf.md` @ingroup api_selection_ops

#include <src/select/select.hh>

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::sysv {

struct Cta : select::NoFields<"cta"> {};          ///< sysv_cta_dispatch: fused factor + solve, n <= 32
struct Compose : select::NoFields<"compose"> {};  ///< public sytrf + sytrs; any homogeneous batch

using SysvChoice = std::variant<Cta, Compose>;  ///< no vendor family: a `vendor` pin warns and runs Auto

template <class T>  /// Every compiled choice, once, in tie-break order (§6.3).
constexpr auto candidates() { return select::all_of<SysvChoice>(); }

inline constexpr std::array<std::string_view, 1> kLastResort{"compose"};

inline constexpr select::OpSpec spec{Op::sysv, select::Lib::none, select::Rules{kLastResort}};

inline constexpr std::array<std::string_view, 3> key_names{"n:log:3", "nrhs:log", "batch:log"};  ///< n^3/3 + 2 n^2 nrhs

/// The transcriber's grid: both sides of the CTA ceiling (32|33), log-spaced elsewhere.
inline constexpr std::array<int, 16> grid_n{1, 2, 4, 8, 12, 16, 24, 32, 33, 48, 64, 128, 256, 512, 1024, 2048};
inline constexpr std::array<int, 4> grid_nrhs{1, 4, 16, 64};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::sysv
