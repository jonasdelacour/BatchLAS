#pragma once

// orgqr's selection vocabulary (flat-kernel-selection.md#phase-5-orgqr), header-only.

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::orgqr {

// Fieldless (Blocked's orgqr_nb is derived from type and min(m, n)); tie-break order (§6.3).
struct Blocked : select::NoFields<"blocked"> {};  // orgqr_blocked_dispatch: identity + the public ormqr
struct Vendor : select::NoFields<"vendor"> {};    // backend::orgqr_vendor

using OrgqrChoice = std::variant<Blocked, Vendor>;

template <class T>
constexpr auto candidates() {
    return std::array<OrgqrChoice, 2>{Blocked{}, Vendor{}};
}

inline constexpr std::array<std::string_view, 2> last_resort{"vendor", "blocked"};
inline constexpr select::Rules rules{last_resort};  // §5.5: Vendor runs every shape it is given (n > m, CPU, heterogeneous).

// The old predicates read m and n only (no batch, no arch). Work ~ m n^2.
inline constexpr std::array<std::string_view, 2> key_names{"m:log", "n:log:2"};

// Both sides of the old 512 ceiling, a coarse log grid elsewhere, n <= m (transcriber: tuned/README.md).
inline constexpr std::array<int, 17> grid{1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 513, 768, 1024, 2048, 4096,
                                          8192};

}  // namespace batchlas::ops::orgqr
