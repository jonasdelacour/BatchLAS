#pragma once

/// @file
/// @brief syev: cta, cta_fused, jacobi, blocked, two_stage, vendor. evidence: docs/perf/syev.md @ingroup selection_ops

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::syev {

struct Cta : select::NoFields<"cta"> {};               ///< syev_cta: sytrd + steqr_cta, n <= 32
struct CtaFused : select::NoFields<"cta_fused"> {};    ///< syev_cta_fused: one kernel, n <= 32
struct Jacobi : select::NoFields<"jacobi"> {};         ///< syev_jacobi_cta, n <= 32
struct Blocked : select::NoFields<"blocked"> {};       ///< syev_blocked: sytrd_blocked + stedc, any n
struct TwoStage : select::NoFields<"two_stage"> {};    ///< syev_two_stage: sy2sb + sb2st + stedc, any n
struct Vendor : select::NoFields<"vendor"> {};         ///< backend::syev_vendor; needs the solver library

using SyevChoice = std::variant<Cta, CtaFused, Jacobi, Blocked, TwoStage, Vendor>;  ///< GPU natives: square, no NETLIB

template <class T>  /// Every compiled choice, once, in tie-break order (§6.3): native first, vendor last.
constexpr auto candidates() { return select::all_of<SyevChoice>(); }

inline constexpr select::OpSpec spec{Op::syev, select::Lib::solver};  ///< last resort (§5.5): blocked, vendor (CPU)

/// Keys: jobz N|V exact; uplo is no key (Upper mirrors into Lower); n log weight 3; batch log.
inline constexpr std::array<std::string_view, 3> key_names{"jobz:exact", "n:log:3", "batch:log"};

/// Both sides of every old threshold plus a log grid; the transcriber (tuned/README.md) spells the same grid.
inline constexpr std::array<int, 37> grid_n{1,   2,   3,   4,   6,   8,   9,   12,  16,  20,  24,   25,   28,
                                            32,  33,  40,  48,  64,  96,  128, 192, 256, 257, 320, 321,  384,
                                            448, 449, 512, 513, 640, 768, 1024, 1025, 1536, 2048, 4096};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::syev
