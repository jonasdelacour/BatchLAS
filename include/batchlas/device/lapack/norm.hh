// SPDX-License-Identifier: MIT
#pragma once
// Public device norm (v22 design C.2): core lange under the host op's name, one body.

#include <batchlas/blas/enums.hh>
#include <batchlas/device/lapack_aux.hh>

namespace batchlas::device::inline v2 {

/// @brief \f$\lVert A \rVert\f$ of one m x n matrix per executor, the host norm() inside a kernel.
/// @tparam NT   Max, One, Inf or Frobenius; compile-time, so one kernel holds one reduction
/// @tparam Alg  Frobenius only: alg::Unscaled (one reduction) or alg::Scaled (two-pass lassq)
/// @tparam S    the reduction scope, as core lange's
/// @param[in] ex  the executor; every item calls with the same arguments (collective)
/// @param[in] a   the matrix operand, any m and n
/// @return the norm, the same value on every item
template <NormType NT, class Alg = alg::Unscaled, Scope S = Scope::Auto, class Ex, class MA>
BATCHLAS_DLA_INLINE auto norm(const Ex& ex, const MA& a) {
    static_assert(NT != NormType::Spectral, "norm: NormType::Spectral needs a device syev, not a device op yet");
    return lange<NT, Alg, S>(ex, a);
}

}  // namespace batchlas::device
