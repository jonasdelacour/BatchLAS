// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief `ritz_values`: the Rayleigh quotients \f$w_j = \mathrm{Re}(v_j^H A v_j) / (v_j^H v_j)\f$
/// of the columns of V, inside a kernel, as the host ritz_values() computes them.
///
/// Two forms, any n and k: `ritz_values(ex, A, V, w)` needs no workspace and forms \f$A v_j\f$ on
/// read; `ritz_values<tier::Blocked<NB>, Cfg>(ex, A, V, w, ws)` is the host route's algorithm on
/// the in-group gemm, per NB columns \f$A V(:, \mathrm{blk})\f$ into this matrix's global
/// workspace and then the quotient per column. The Blocked form runs on a
/// `WorkGroup<Cfg::threads, 32>` at sub-group size 32. The per-column `ritz_values_quotient` is
/// in @ref api_device_lowlevel.
/// @ingroup api_device_lapack_eig
// The host route forms A V with gemm / spmm and calls ritz_values_quotient per (item, column).

#include <cstddef>
#include <type_traits>

#include <batchlas/device/level1.hh>
#include <batchlas/device/launch.hh>
#include <batchlas/device/level3/group.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/view.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

namespace detail {
// The one workspace plan of the Blocked form (buffer_size and the carve): the Level 3 stages
// (local) and A V(:, blk), n x min(k, NB) at ld n (global).
template <class Scalar> struct RitzValuesBlockedPlan {
    WsSlot<Scalar> smem;
    WsSlot<Scalar> av;
    WorkspaceSize size;
    template <class Tier, class Cfg> static constexpr RitzValuesBlockedPlan make(int n, int k) {
        static_assert(Tier::n_max < 0, "ritz_values: the workspace form takes tier::Blocked<NB> (the tier-free form needs no workspace)");
        static_assert(l3::Config<Cfg>, "ritz_values: tier::Blocked runs on the in-group gemm, Cfg is an l3::cfg preset");
        if constexpr (l3::Config<Cfg>) static_assert(std::is_same_v<typename Cfg::value_type, Scalar>, "ritz_values: Cfg::value_type is A's T");
        ::batchlas::device::Plan p;
        const auto s = p.take<Scalar>(WsSpace::Local, static_cast<std::size_t>(l3::smem_elems<Cfg>()));
        const int nb = k < Tier::nb ? k : Tier::nb;
        const auto av = p.take<Scalar>(WsSpace::Global, static_cast<std::size_t>(n) * static_cast<std::size_t>(nb));
        return {s, av, p.size()};
    }
};
}  // namespace detail

/// @brief One Ritz value from a given product: \f$\mathrm{Re}(v^H a_v) / (v^H v)\f$.
/// @param[in] ex  the executor (collective)
/// @param[in] v   the trial vector
/// @param[in] av  \f$A v\f$, same length as @p v
/// @return the quotient (\f$v^H v\f$ as `sumsq<alg::Fused>`), the same on every item
/// @ingroup api_device_lowlevel
template <class Ex, class V, class AV> BATCHLAS_DEVICE_INLINE auto ritz_values_quotient(const Ex& ex, const V& v, const AV& av) {
    using T = elem_t<V>;
    const T num = dotc(ex, v, av);
    const real_t<T> den = sumsq<alg::Fused>(ex, v);
    return Arith<T>::re(num) / den;
}

/// @brief The host ritz_values() inside a kernel: \f$w_j = \mathrm{Re}(v_j^H A v_j) / (v_j^H v_j)\f$
/// for the k columns of V, one dense n x n A per executor, any n and k, no workspace.
/// @param[in]  ex  the executor (collective); A and V are read only
/// @param[in]  a   A, n x n
/// @param[in]  v   V, n x k, a Mat (col())
/// @param[out] w   k reals, written by the leader (other items read them after a barrier)
/// @return 0 (the host op has no info)
/// @ingroup api_device_lapack_eig
template <class Ex, class MA, class MV, class W> BATCHLAS_DEVICE_INLINE int ritz_values(const Ex& ex, const MA& a, const MV& v, const W& w) {
    const int n = a.rows(), k = v.cols();
    for (int j = 0; j < k; ++j) {
        const auto vj = v.col(j);
        const auto av = fn_vec(n, [a, vj, n](int r) BATCHLAS_DEVICE_BODY {
            return dot(Thread{}, fn_vec(n, [a, r](int c) BATCHLAS_DEVICE_BODY { return a(r, c); }), vj);
        });
        const auto q = ritz_values_quotient(ex, vj, av);
        if (ex.leader()) w(j) = q;
    }
    return 0;
}

namespace routine {
/// @brief The routine tag of the workspace form of `ritz_values` for `launch` and `buffer_size`.
/// @tparam Cfg  the gemm's `l3::cfg` preset, which sizes the stages, so the tag carries it
/// @ingroup api_device_lapack_eig
template <class Cfg> struct ritz_values {
    /// The name `with_tier` puts in its error message.
    static constexpr const char* name = "ritz_values";
};
}  // namespace routine

/// @brief The `tier::Blocked<NB>` tier of `ritz_values`: one matrix per
/// `WorkGroup<Cfg::threads, 32>` (`sg_size` 32), any n.
///
/// `buffer_size(n, k)` is one matrix's local bytes (the gemm's stages) and global bytes
/// (n x min(k, NB) values of the scalar type).
/// @ingroup api_device_lapack_eig
template <class Cfg, int NB, class A, class Scalar> struct tier_info<routine::ritz_values<Cfg>, tier::Blocked<NB, A>, Scalar> {
    static constexpr int wg_size = Cfg::threads, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int n, int k) {
        return detail::RitzValuesBlockedPlan<Scalar>::template make<tier::Blocked<NB, A>, Cfg>(n, k).size;
    }
};

/// @brief The host ritz_values() inside a kernel on the in-group gemm, the host route's
/// algorithm: per block of NB columns, P := A V(:, blk) (gemm), then w_j = ritz_values_quotient(v_j, p_j).
/// @tparam Tier  `tier::Blocked<NB>`
/// @tparam Cfg   the `l3::cfg` preset of the gemm
/// @param[in]  ex  a `WorkGroup<Cfg::threads, 32>` (collective, launched at sub-group size 32);
///                 A and V read only, visible on entry
/// @param[in]  a   A, n x n, a Mat over global memory
/// @param[in]  v   V, n x k, a Mat over global memory (col(), sub())
/// @param[out] w   k reals, written by the leader; visible to every item on return
/// @param[in]  ws  this matrix's slice of `buffer_size<routine::ritz_values<Cfg>, Tier, T>(n, k)`
/// @return 0 (the host op has no info)
/// @ingroup api_device_lapack_eig
template <class Tier, class Cfg, class Ex, class MA, class MV, class W>
BATCHLAS_DEVICE_INLINE int ritz_values(const Ex& ex, const MA& a, const MV& v, const W& w, const Workspace& ws) {
    using T = typename MA::value_type;
    constexpr int NB = Tier::nb;
    const int n = a.rows(), k = v.cols();
    const auto pl = detail::RitzValuesBlockedPlan<T>::template make<Tier, Cfg>(n, k);
    const Workspace sm = sub_workspace(ws, pl.smem);
    T* const p = carve_global(ws, pl.av);
    for (int c0 = 0; c0 < k; c0 += NB) {
        const int nb = k - c0 < NB ? k - c0 : NB;
        gemm<Transpose::NoTrans, Transpose::NoTrans, Cfg>(ex, One{}, a, v.sub(0, c0, n, nb), Zero{}, mat(p, n, nb, n), sm);
        for (int j = 0; j < nb; ++j) {
            const auto q = ritz_values_quotient(ex, v.col(c0 + j), vec(p + static_cast<std::ptrdiff_t>(j) * n, n));
            if (ex.leader()) w(c0 + j) = q;
        }
        ex.barrier(Mem::Global);  // P is read by every item before the next block's gemm writes it; w is published
    }
    return 0;
}

}  // namespace batchlas::device
