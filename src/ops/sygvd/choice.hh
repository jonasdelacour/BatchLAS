#pragma once

/// @file
/// @brief sygvd: compose, vendor. `evidence: docs/perf/sygvd.md#sygvd-routes` @ingroup api_selection_ops

#include <src/select/select.hh>

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::sygvd {

struct Compose : select::NoFields<"compose"> {};  ///< potrf + sygst + syev + trsm/trmm, each child routed
struct Vendor : select::NoFields<"vendor"> {};    ///< backend::sygvd_vendor: cuSOLVER ?sygvd per item, LAPACKE

using SygvdChoice = std::variant<Compose, Vendor>;  ///< fieldless

template <class T>  /// Tie-break order (§6.3): native first, vendor last.
constexpr auto candidates() { return select::all_of<SygvdChoice>(); }

inline constexpr std::array<std::string_view, 2> kLastResort{"vendor", "compose"};

inline constexpr select::OpSpec spec{Op::sygvd, select::Lib::solver, select::Rules{kLastResort}};

/// Keys as syev's: jobz exact, n log weight 3, batch log; itype and uplo are no keys.
inline constexpr std::array<std::string_view, 3> key_names{"jobz:exact", "n:log:3", "batch:log"};

inline constexpr std::array<int, 14> grid_n{1, 2, 4, 8, 16, 32, 64, 128, 240, 256, 480, 512, 1024, 1200};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::sygvd
