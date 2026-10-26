#pragma once

/// @file
/// @brief potri: cta, blocked:nb.
/// `evidence: docs/perf/potri.md#potri-selection` @ingroup api_selection_ops

#include <src/select/select.hh>

#include <array>
#include <string_view>
#include <type_traits>
#include <variant>

namespace batchlas::ops::potri {

struct Cta : select::NoFields<"cta"> {};  ///< sycl_potri::potri_cta: trtri + lauum in local memory, one launch
/// LAPACK's blocked trtri then lauum: the CTA kernel on the nb x nb diagonal blocks, public trmm, gemm and herk/syrk.
struct Blocked {
    int nb = 32;  ///< diagonal block order; the CTA kernel must hold one block
    static constexpr std::string_view name = "blocked";
    static constexpr std::array<std::string_view, 1> fields{"nb"};
    std::array<int, 1> values() const { return {nb}; }
    static Blocked from(std::array<int, 1> v) { return {v[0]}; }
    bool operator==(const Blocked&) const = default;
};
struct Fused : select::NoFields<"fused"> {};        ///< A/B baseline only: identity + the one-launch potrs kernel
struct Identity : select::NoFields<"identity"> {};  ///< A/B baseline only: identity + two public trsm

using PotriChoice = std::variant<Cta, Blocked, Fused, Identity>;

template <class T>  /// Tie-break order (§6.3): the one-launch kernel first. No vendor family exists.
constexpr auto candidates() {
    if constexpr (std::is_same_v<T, float>)
        return std::array<PotriChoice, 6>{Cta{}, Blocked{32}, Blocked{64}, Blocked{128}, Fused{}, Identity{}};
    else
        return std::array<PotriChoice, 5>{Cta{}, Blocked{32}, Blocked{64}, Fused{}, Identity{}};
}

inline constexpr std::array<std::string_view, 2> last_resort{"blocked", "cta"};  ///< generality order (§5.5)
inline constexpr select::OpSpec spec{Op::potri, select::Lib::none, {last_resort}};  ///< no library

inline constexpr std::array<std::string_view, 2> key_names{"n:log:3", "batch:log"};  ///< work ~ n^3 batch

inline constexpr std::array<int, 18> grid_n{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};
inline constexpr std::array<int, 5> grid_batch{128, 512, 2048, 8192, 32768};

}  // namespace batchlas::ops::potri
