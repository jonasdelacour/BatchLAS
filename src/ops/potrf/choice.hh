#pragma once

// potrf's selection vocabulary (docs/design/flat-kernel-selection.md §4.2). Header-only, so
// tests and benchmarks name the same choices the library runs.

#include "../../select/select.hh"

#include <array>
#include <complex>
#include <string_view>
#include <type_traits>
#include <variant>

namespace batchlas::ops::potrf {

// One family per kernel driver; int fields are chosen knobs, anything derived stays in the driver.
struct Tiny : select::NoFields<"tiny"> {};
struct Cta : select::NoFields<"cta"> {};
struct Lpanel {
    int panel = 8;  // potrf_lpanel_dispatch's nb_hint; 16 is instantiated for float only
    static constexpr std::string_view name = "lpanel";
    static constexpr std::array<std::string_view, 1> fields{"panel"};
    std::array<int, 1> values() const { return {panel}; }
    static Lpanel from(std::array<int, 1> v) { return {v[0]}; }
    bool operator==(const Lpanel&) const = default;
};
struct Blocked : select::NoFields<"blocked"> {};  // nb/W stay PotrfBlockedConst + env (§11)
struct Vendor : select::NoFields<"vendor"> {};

using PotrfChoice = std::variant<Tiny, Cta, Lpanel, Blocked, Vendor>;

// Every compiled choice, once, in tie-break order (§6.3): simpler first, vendor last.
template <class T>
constexpr auto candidates() {
    if constexpr (std::is_same_v<T, float>)
        return std::array<PotrfChoice, 6>{Tiny{}, Cta{}, Lpanel{8}, Lpanel{16}, Blocked{}, Vendor{}};
    else
        return std::array<PotrfChoice, 5>{Tiny{}, Cta{}, Lpanel{8}, Blocked{}, Vendor{}};
}

// Legacy BATCHLAS_POTRF_ROUTE spellings, accepted until phase 5 (§5.3); bare `lpanel` meant native.
inline constexpr std::array<select::Alias, 5> aliases{{
    {"native:tiny", "tiny"},
    {"native:cta", "cta"},
    {"native:lpanel", "lpanel:panel=8"},
    {"native:blocked", "blocked"},
    {"lpanel", "lpanel:panel=8"},
}};
// Generality order (§5.5): Blocked runs every Lower shape on a GPU, Vendor everything else.
inline constexpr std::array<std::string_view, 2> last_resort{"blocked", "vendor"};
inline constexpr select::Rules rules{aliases, last_resort};

// Table keys: the exact-match key first, then the log-distance keys (§5.4). n weighs 3: work
// grows as n^3 and linearly in batch, so the distance approximates log-cost.
inline constexpr std::array<std::string_view, 3> key_names{"uplo:exact", "n:log:3", "batch:log"};

// The tuner's coarse grid (§6.2): today's sweep grid, so converted and tuned tables line up.
inline constexpr std::array<int, 34> grid_n{1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48,
    56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 640, 768, 1024, 1280};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::potrf
