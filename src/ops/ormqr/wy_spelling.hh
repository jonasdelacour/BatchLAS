// SPDX-License-Identifier: MIT
#pragma once

/// @file
/// @brief ormqr_blocked's W2 = op(T) W1 spelling (gemm or trmm).

#include <batchlas/settings.hh>

#include <src/math-helpers.hh>
#include <src/select/vendor.hh>

#include <string>

namespace batchlas::ops::ormqr {

// T is upper triangular, so W2 = op(T) W1 may be a trmm; BATCHLAS_ORMQR_WY=gemm|trmm pins it.
enum class WyPin { Measured, Gemm, Trmm };

inline WyPin wy_pin() {
    static const WyPin result = []() {
        const char* v = batchlas::settings().selection.ormqr_wy.get();
        if (!v) return WyPin::Measured;
        const std::string s(v);
        if (s == "gemm") return WyPin::Gemm;
        if (s == "trmm") return WyPin::Trmm;
        return WyPin::Measured;
    }();
    return result;
}

// The trmm only where its tile kernel is enabled and ahead: real types, ib <= 64.
// evidence: docs/perf/qr.md#ormqr-wy-trmm-gate
template <Backend B, typename T>
inline bool wy_trmm_applicable(int ib) {
    const WyPin pin = wy_pin();
    if (pin != WyPin::Measured) {
        return pin == WyPin::Trmm;
    }
    // Whether the substitution is enabled for (B, T): a vendor-free build is still Backend::CUDA.
    constexpr bool tile_kernel_enabled = select::level3_tile_substitution_enabled<B, T>;
    constexpr bool type_beats_gemm_at_this_m = !batchlas::detail::is_complex<T>::value;
    return tile_kernel_enabled && type_beats_gemm_at_this_m && ib <= 64;
}

}  // namespace batchlas::ops::ormqr
