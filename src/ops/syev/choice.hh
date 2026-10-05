#pragma once

// syev's selection vocabulary (docs/design/flat-select-p5/syev.md), header-only.

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::syev {

// All fieldless: wg multipliers, STEDC/STEQR parameters and the P bucket are derived.
struct Cta : select::NoFields<"cta"> {};               // syev_cta: sytrd + steqr_cta, n <= 32
struct CtaFused : select::NoFields<"cta_fused"> {};    // syev_cta_fused: one kernel, n <= 32
struct Jacobi : select::NoFields<"jacobi"> {};         // syev_jacobi_cta, n <= 32
struct Blocked : select::NoFields<"blocked"> {};       // syev_blocked: sytrd_blocked + stedc
struct TwoStage : select::NoFields<"two_stage"> {};    // syev_two_stage: sy2sb + sb2st + stedc
struct Vendor : select::NoFields<"vendor"> {};         // backend::syev_vendor

using SyevChoice = std::variant<Cta, CtaFused, Jacobi, Blocked, TwoStage, Vendor>;

// Every compiled choice, once, in tie-break order (§6.3): native first, vendor last.
template <class T>
constexpr auto candidates() {
    return std::array<SyevChoice, 6>{Cta{}, CtaFused{}, Jacobi{}, Blocked{}, TwoStage{}, Vendor{}};
}

inline constexpr std::array<std::string_view, 2> last_resort{"blocked", "vendor"};
inline constexpr select::Rules rules{last_resort};  // §5.5: Blocked runs every square GPU shape, Vendor everything else (CPU).

// jobz N|V; uplo is no key (Upper mirrors into Lower). Work ~ n^3 batch.
inline constexpr std::array<std::string_view, 3> key_names{"jobz:exact", "n:log:3", "batch:log"};

// Both sides of every old threshold plus a log grid; the transcriber (tuned/README.md) spells the same grid.
inline constexpr std::array<int, 37> grid_n{1,   2,   3,   4,   6,   8,   9,   12,  16,  20,  24,   25,   28,
                                            32,  33,  40,  48,  64,  96,  128, 192, 256, 257, 320, 321,  384,
                                            448, 449, 512, 513, 640, 768, 1024, 1025, 1536, 2048, 4096};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::syev
