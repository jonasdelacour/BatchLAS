#pragma once

/// @file
/// @brief sytrs: cta, sg, blocked, wg, vendor. `evidence: docs/perf/sytrf.md` @ingroup api_selection_ops

#include <src/select/select.hh>

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::sytrs {

struct Cta : select::NoFields<"cta"> {};          ///< sytrs_cta_dispatch: converted factor in local memory, a sub-group per rhs
struct Sg : select::NoFields<"sg"> {};            ///< sytrs_sg_dispatch: a sub-group per (item, rhs slot), factor read as stored
struct Blocked : select::NoFields<"blocked"> {};  ///< sytrs_blocked_dispatch: ?sytrs2, the public trsm twice, any n
struct Wg : select::NoFields<"wg"> {};            ///< sytrs_native_dispatch: one work-group per right-hand side
struct Vendor : select::NoFields<"vendor"> {};    ///< backend::sytrs_vendor: LAPACKE ?sytrs per item (NETLIB only)

using SytrsChoice = std::variant<Cta, Sg, Blocked, Wg, Vendor>;  ///< natives: GPU, sub-group 32, homogeneous, not NETLIB

template <class T>  /// Every compiled choice, once, in tie-break order (§6.3).
constexpr auto candidates() { return select::all_of<SytrsChoice>(); }

inline constexpr std::array<std::string_view, 2> kLastResort{"blocked", "vendor"};

inline constexpr select::OpSpec spec{Op::sytrs, select::Lib::solver, select::Rules{kLastResort}};

inline constexpr std::array<std::string_view, 3> key_names{"n:log:2", "nrhs:log", "batch:log"};  ///< ~ n^2 nrhs batch

/// Log-spaced, plus both sides of the register-tile steps (32|33, 64|65) and the ceilings (128|129, 256|257).
inline constexpr std::array<int, 18> grid_n{1, 2, 4, 8, 16, 24, 32, 33, 48, 64, 65, 96, 128, 129, 256, 257, 512, 1024};
inline constexpr std::array<int, 7> grid_nrhs{1, 2, 4, 8, 16, 64, 256};
inline constexpr std::array<int, 4> grid_batch{128, 1024, 8192, 32768};

}  // namespace batchlas::ops::sytrs
