#pragma once

/// @file
/// @brief potrf: tiny, cta, lpanel:panel, blocked, vendor. evidence: docs/perf/potrf.md @ingroup selection_ops

#include "../../select/select.hh"

#include <array>
#include <complex>
#include <string_view>
#include <type_traits>
#include <variant>

namespace batchlas::ops::potrf {

struct Tiny : select::NoFields<"tiny"> {};  ///< potrf_tiny_dispatch: registers; n <= potrf_tiny_max_n, max_wg >= 64
struct Cta : select::NoFields<"cta"> {};    ///< potrf_cta_dispatch: whole matrix in SLM; n <= the CTA SLM ceiling
/// potrf_lpanel_dispatch: SLM-resident panels; Lower only, n <= the LPanel SLM ceiling for `panel`.
struct Lpanel {
    int panel = 8;  ///< potrf_lpanel_dispatch's nb_hint; 16 is instantiated for float only
    static constexpr std::string_view name = "lpanel";
    static constexpr std::array<std::string_view, 1> fields{"panel"};
    std::array<int, 1> values() const { return {panel}; }
    static Lpanel from(std::array<int, 1> v) { return {v[0]}; }
    bool operator==(const Lpanel&) const = default;
};
struct Blocked : select::NoFields<"blocked"> {};  ///< potrf_blocked_dispatch + gemm/trsm, Lower; nb/W: PotrfBlockedConst + env
struct Vendor : select::NoFields<"vendor"> {};    ///< backend::potrf_vendor; needs the solver library

using PotrfChoice = std::variant<Tiny, Cta, Lpanel, Blocked, Vendor>;  ///< natives need a GPU, sub-group 32, n >= 1

/// Every compiled choice, once, in tie-break order (§6.3): simpler first, vendor last.
template <class T>
constexpr auto candidates() {
    if constexpr (std::is_same_v<T, float>)
        return std::array<PotrfChoice, 6>{Tiny{}, Cta{}, Lpanel{8}, Lpanel{16}, Blocked{}, Vendor{}};
    else
        return std::array<PotrfChoice, 5>{Tiny{}, Cta{}, Lpanel{8}, Blocked{}, Vendor{}};
}

inline constexpr select::OpSpec spec{Op::potrf, select::Lib::solver};  ///< default last resort: blocked, vendor

/// Table keys: uplo exact (L|U); n log-distance, weight 3 (work ~ n^3); batch log, weight 1.
/// evidence: docs/design/flat-kernel-selection.md (§5.4)
inline constexpr std::array<std::string_view, 3> key_names{"uplo:exact", "n:log:3", "batch:log"};

/// The tuner's coarse grid (§6.2): today's sweep grid, so converted and tuned tables line up.
inline constexpr std::array<int, 34> grid_n{1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48,
    56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 640, 768, 1024, 1280};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};  ///< batch axis of the grid

}  // namespace batchlas::ops::potrf
