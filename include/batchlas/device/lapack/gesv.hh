// SPDX-License-Identifier: MIT
#pragma once
// ?GESV's tiny-tier step: the rank-1 row update by the broadcast pivot row (two roundings, a
// source lane, a start column), in detail until core ger_row takes them.
// The public op gesv<Tier>(ex, A, ipiv, B, ws): getrf<Tier> then getrs<NoTrans, Tier> on one
// executor and one workspace (the host blocked route's composition); tier::Blocked, any n.

#include <cstdint>

#include <batchlas/device.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/lapack/getrf.hh>
#include <batchlas/device/lapack/getrs.hh>

namespace batchlas::device::detail {

// x[k] -= m * x[k]@src, k0 <= k < min(N, kend), where act; collective: every lane reaches every k.
template <int N, class Px, class Scalar>
BATCHLAS_DEVICE_INLINE void ger_frag_bcast(const Px& px, Scalar m, Scalar (&x)[N], std::uint32_t src, int k0, int kend, bool act) {
    using Ar = arith_t<arith::Fma, Scalar>;
BATCHLAS_UNROLL_FULL
    for (int k = k0; k < N; ++k) {
        if (k >= kend) continue;
        const Scalar u = px.bcast(x[k], src);
        if (act) x[k] = Ar::sub(x[k], Ar::mul(m, u));
    }
}

}  // namespace batchlas::device::detail

// ==== the public op: gesv<Tier>(ex, A, ipiv, B, ws [, live]) =====================================
namespace batchlas::device {

namespace routine {
struct gesv {
    static constexpr const char* name = "gesv";
};
}  // namespace routine

/// gesv's Blocked<NB>: getrf's Blocked executor and local plan; getrs reuses the same bytes.
/// buffer_size(n[, nrhs]): nrhs does not enter it.
template <int NB, class Alg, class Scalar>
struct tier_info<routine::gesv, tier::Blocked<NB, Alg>, Scalar> : tier_info<routine::getrf, tier::Blocked<NB, Alg>, Scalar> {
    using tier_info<routine::getrf, tier::Blocked<NB, Alg>, Scalar>::buffer_size;
    static constexpr WorkspaceSize buffer_size(int n, int) { return buffer_size(n); }
};
template <class Scalar> struct tiers<routine::gesv, Scalar> {
    using type = tier::list<tier::Blocked<32>>;
};

/// @brief ?GESV in a kernel: \f$A = PLU\f$ in place, then \f$X = A^{-1} B\f$ over B.
///
/// Collective as getrf. As the host blocked route, the solve runs whatever info says: for
/// info > 0, X is not meaningful (Inf / NaN from the zero pivot).
/// - `tier::Blocked<NB>`: `ex` a WorkGroup<tier_info::wg_size, 32>; A n x n and B n x nrhs of
///   any size in global memory; ipiv in memory every item reads; `ws.local` >=
///   buffer_size<routine::gesv, Tier, T>(n) bytes.
/// @param[in]     ex    the executor
/// @param[in,out] a     this matrix; overwritten with L and U
/// @param[out]    ipiv  this matrix's n pivots, 1-based (a Vec<int>)
/// @param[in,out] b     this matrix's right-hand sides; overwritten with X
/// @param[in]     ws    this matrix's workspace slice
/// @param[in]     live  false: a dead matrix of the launch (executor-uniform)
/// @return getrf's info, uniform across `ex`
template <class Tier, class Ex, class MA, class PV, class MB>
BATCHLAS_DEVICE_INLINE int gesv(const Ex& ex, const MA& a, const PV& ipiv, const MB& b, const Workspace& ws = {}, bool live = true) {
    using D = detail::lu_value_t<MA>;
    using Tr = resolve_tier_t<routine::gesv, D, Tier>;
    static_assert(HasTier<routine::gesv, Tr, D>, "gesv: this tier is not offered for this scalar type");
    const int info = getrf<Tr>(ex, a, ipiv, ws, live);
    (void)getrs<Transpose::NoTrans, Tr>(ex, a, ipiv, b, ws, live);
    return info;
}

}  // namespace batchlas::device
