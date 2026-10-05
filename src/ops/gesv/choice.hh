#pragma once

// gesv's selection vocabulary (docs/design/flat-kernel-selection.md#phase-5-gesv), header-only.

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::gesv {

struct Tiny : select::NoFields<"tiny"> {};        // gesv_tiny_dispatch: fused LU factor + solve (buckets derived)
struct Blocked : select::NoFields<"blocked"> {};  // public getrf + public getrs

using GesvChoice = std::variant<Tiny, Blocked>;

// Tie-break order (§6.3). No vendor family: a `vendor` pin warns and falls back to Auto.
template <class T>
constexpr auto candidates() {
    return std::array<GesvChoice, 2>{Tiny{}, Blocked{}};
}

inline constexpr std::array<std::string_view, 1> last_resort{"blocked"};
inline constexpr select::Rules rules{last_resort};

// What the old predicates read: the order and nrhs (batch only as >= 1). Work: 2n^3/3 + 2 n^2 nrhs.
inline constexpr std::array<std::string_view, 2> key_names{"n:log:3", "nrhs:log"};

// Both sides of every old threshold (16|17, 32|33, nrhs 4|5), log-spaced elsewhere; the transcriber spells it too.
inline constexpr std::array<int, 29> grid_n{1, 2, 3, 4, 6, 8, 12, 16, 17, 20, 24, 28, 32, 33, 40, 48, 64,
                                            96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096};
inline constexpr std::array<int, 8> grid_nrhs{1, 2, 4, 5, 8, 16, 64, 256};

}  // namespace batchlas::ops::gesv
