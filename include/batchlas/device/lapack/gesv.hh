// SPDX-License-Identifier: MIT
#pragma once
// ?GESV's tiny-tier step: the rank-1 row update by the broadcast pivot row (two roundings, a
// source lane, a start column), in gesv_detail until core ger_row takes them.
// The public op gesv<Tier>(ex, A, B, ipiv, ws): getrf<Tier> then getrs<NoTrans, Tier> on one
// executor and one workspace (the host blocked route's composition); tier::Blocked, any n.

#include <cstdint>

#include <batchlas/device.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/lapack/getrf.hh>
#include <batchlas/device/lapack/getrs.hh>

namespace batchlas::device::inline v2::gesv_detail {

// x[k] -= m * x[k]@src, k0 <= k < min(N, kend), where act; collective: every lane reaches every k.
template <int N, class Px, class T>
BATCHLAS_DLA_INLINE void ger_frag_bcast(const Px& px, T m, T (&x)[N], std::uint32_t src, int k0, int kend, bool act) {
    using Ar = arith_t<arith::Fma, T>;
BATCHLAS_UNROLL_FULL
    for (int k = k0; k < N; ++k) {
        if (k >= kend) continue;
        const T u = px.bcast(x[k], src);
        if (act) x[k] = Ar::sub(x[k], Ar::mul(m, u));
    }
}

}  // namespace batchlas::device::gesv_detail

// ==== the public op: gesv<Tier>(ex, A, B, ipiv, ws [, live]) (v22 design C.2-C.7) ================
namespace batchlas::device::inline v2 {

namespace routine {
struct gesv {
    static constexpr const char* name = "gesv";
};
}  // namespace routine

/// gesv's Blocked<NB>: getrf's Blocked executor and local plan; getrs reuses the same bytes.
template <int NB, class Alg, class T>
struct tier_info<routine::gesv, tier::Blocked<NB, Alg>, T> : tier_info<routine::getrf, tier::Blocked<NB, Alg>, T> {};
template <class T> struct tiers<routine::gesv, T> {
    using type = tier::list<tier::Blocked<32>>;
};

/// The workspace of one n x n system with nrhs right-hand sides for gesv<Tier> (constexpr).
template <class Tier, class T> constexpr WorkspaceSize gesv_buffer_size(int n, int nrhs = 1) {
    (void)nrhs;
    return tier_info<routine::gesv, resolve_tier_t<routine::gesv, T, Tier>, T>::buffer_size(n);
}

/// @brief ?GESV in a kernel: \f$A = PLU\f$ in place, then \f$X = A^{-1} B\f$ over B.
///
/// Collective as getrf. As the host blocked route, the solve runs whatever info says: for
/// info > 0, X is not meaningful (Inf / NaN from the zero pivot).
/// - `tier::Blocked<NB>`: `ex` a WorkGroup<tier_info::wg_size, 32>; A n x n and B n x nrhs of
///   any size in global memory; ipiv in memory every item reads; `ws.local` >=
///   gesv_buffer_size<Tier, T>(n) bytes.
/// @param[in]     ex    the executor
/// @param[in,out] a     this matrix; overwritten with L and U
/// @param[in,out] b     this matrix's right-hand sides; overwritten with X
/// @param[out]    ipiv  this matrix's n pivots, 1-based int32
/// @param[in]     ws    this matrix's workspace slice
/// @param[in]     live  false: a dead matrix of the launch (executor-uniform)
/// @return getrf's info, uniform across `ex`
template <class Tier, class Ex, class MA, class MB>
BATCHLAS_DLA_INLINE std::int32_t gesv(const Ex& ex, const MA& a, const MB& b, std::int32_t* ipiv, const Workspace& ws = {},
                                      bool live = true) {
    using D = getrf_detail::lu_value_t<MA>;
    using Tr = resolve_tier_t<routine::gesv, D, Tier>;
    static_assert(HasTier<routine::gesv, Tr, D>, "gesv: this tier is not offered for this scalar type");
    const std::int32_t info = getrf<Tr>(ex, a, ipiv, ws, live);
    (void)getrs<Transpose::NoTrans, Tr>(ex, a, b, ipiv, ws, live);
    return info;
}

}  // namespace batchlas::device
