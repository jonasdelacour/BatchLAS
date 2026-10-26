#pragma once

/// @file
/// @brief sygst: blocked:nb, twosided, vendor. `evidence: docs/perf/sygvd.md#sygvd-routes` @ingroup api_selection_ops

#include <src/select/select.hh>

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::sygst {

/// LAPACK's blocked ?sygst/?hegst on the uplo triangles only: per nb block the SLM diagonal kernel,
/// then trsm (itype 1) or trmm (2, 3), gemm and her2k/syr2k on the panel and trailing parts.
struct Blocked {
    int nb = 32;  ///< block order; the diagonal kernel takes at most 32, a larger block recurses
    static constexpr std::string_view name = "blocked";
    static constexpr std::array<std::string_view, 1> fields{"nb"};
    std::array<int, 1> values() const { return {nb}; }
    static Blocked from(std::array<int, 1> v) { return {v[0]}; }
    bool operator==(const Blocked&) const = default;
};
struct TwoSided : select::NoFields<"twosided"> {};  ///< hermitize + two public trsm (itype 1) or trmm (2, 3)
struct Vendor : select::NoFields<"vendor"> {};      ///< backend::sygst_vendor: LAPACKE ?sygst/?hegst per item

using SygstChoice = std::variant<Blocked, TwoSided, Vendor>;  ///< the children select their own kernels

template <class T>  /// Tie-break order (§6.3): native first, vendor last.
constexpr auto candidates() {
    return std::array<SygstChoice, 6>{Blocked{32}, Blocked{64}, Blocked{128}, Blocked{256}, TwoSided{}, Vendor{}};
}

/// The CPU (no table, never borrows) takes the library first; blocked needs a GPU.
inline constexpr std::array<std::string_view, 3> kLastResort{"vendor", "blocked", "twosided"};

inline constexpr select::OpSpec spec{Op::sygst, select::Lib::solver, select::Rules{kLastResort}};

inline constexpr std::array<std::string_view, 2> key_names{"n:log:3", "batch:log"};  ///< work ~ n^3 batch

inline constexpr std::array<int, 15> grid_n{1, 2, 4, 8, 16, 32, 33, 64, 128, 240, 256, 480, 512, 1024, 1200};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::sygst
