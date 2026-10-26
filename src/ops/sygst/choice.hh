#pragma once

/// @file
/// @brief sygst: twosided, vendor. `evidence: docs/perf/sygvd.md#sygvd-routes` @ingroup api_selection_ops

#include <src/select/select.hh>

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::sygst {

struct TwoSided : select::NoFields<"twosided"> {};  ///< hermitize + two public trsm (itype 1) or trmm (2, 3)
struct Vendor : select::NoFields<"vendor"> {};      ///< backend::sygst_vendor: LAPACKE ?sygst/?hegst per item

using SygstChoice = std::variant<TwoSided, Vendor>;  ///< fieldless: the children select their own kernels

template <class T>  /// Tie-break order (§6.3): native first, vendor last.
constexpr auto candidates() { return select::all_of<SygstChoice>(); }

/// The CPU (no table, never borrows) takes the library first; a GPU table ranks twosided first.
inline constexpr std::array<std::string_view, 2> kLastResort{"vendor", "twosided"};

inline constexpr select::OpSpec spec{Op::sygst, select::Lib::solver, select::Rules{kLastResort}};

inline constexpr std::array<std::string_view, 2> key_names{"n:log:3", "batch:log"};  ///< work ~ n^3 batch

inline constexpr std::array<int, 14> grid_n{1, 2, 4, 8, 16, 32, 64, 128, 240, 256, 480, 512, 1024, 1200};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::sygst
