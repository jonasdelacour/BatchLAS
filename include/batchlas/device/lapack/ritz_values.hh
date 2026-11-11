// SPDX-License-Identifier: MIT
#pragma once
// Public device ritz_values (v22 design C.2, C.7, B8). The host route forms A V with gemm / spmm
// and calls ritz_values_quotient per (item, column). Device forms, any n and k:
//   ritz_values(ex, A, V, w)               no workspace: forms A v_j on read
//   ritz_values<Blocked<NB>, Cfg>(..., ws)  the host route's algorithm on the in-group gemm: per
//                                          NB columns, A V(:, blk) into this matrix's global
//                                          workspace, then the quotient per column

#include <cstddef>
#include <type_traits>

#include <batchlas/device/level1.hh>
#include <batchlas/device/level3/group.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/view.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

namespace ritz_values_detail {
// The one workspace plan of the Blocked form (buffer_size and the carve): the Level 3 stages
// (local) and A V(:, blk), n x min(k, NB) at ld n (global).
template <class T> struct BlockedPlan {
    ws::Slot<T> smem;
    ws::Slot<T> av;
    WorkspaceSize size;
    template <class Tier, class Cfg> static constexpr BlockedPlan make(int n, int k) {
        static_assert(Tier::n_max < 0, "ritz_values: the workspace form takes tier::Blocked<NB> (the tier-free form needs no workspace)");
        static_assert(l3::Config<Cfg>, "ritz_values: tier::Blocked runs on the in-group gemm, Cfg is an l3::cfg preset");
        if constexpr (l3::Config<Cfg>) static_assert(std::is_same_v<typename Cfg::value_type, T>, "ritz_values: Cfg::value_type is A's T");
        ::batchlas::device::Plan p;
        const auto s = p.take<T>(ws::Space::Local, static_cast<std::size_t>(l3::smem_elems<Cfg>()));
        const int nb = k < Tier::nb ? k : Tier::nb;
        const auto av = p.take<T>(ws::Space::Global, static_cast<std::size_t>(n) * static_cast<std::size_t>(nb));
        return {s, av, p.size()};
    }
};
}  // namespace ritz_values_detail

/// @brief One Ritz value from a given product: \f$\mathrm{Re}(v^H a_v) / (v^H v)\f$.
/// @param[in] ex  the executor (collective)
/// @param[in] v   the trial vector
/// @param[in] av  \f$A v\f$, same length as @p v
/// @return the quotient (\f$v^H v\f$ as sumsq<alg::Fused>), the same on every item
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

/// @brief Workspace of ONE matrix for the tier::Blocked ritz_values() (constexpr, host and device).
/// @tparam Tier  tier::Blocked<NB>: NB columns of V per gemm
/// @tparam Cfg   the l3::cfg preset of the gemm (Cfg::value_type == T)
/// @param[in] n  the order of A
/// @param[in] k  the number of columns of V
/// @return local bytes (the gemm's stages) and global bytes (n x min(k, NB) values of T)
template <class Tier, class T, class Cfg> constexpr WorkspaceSize ritz_values_buffer_size(int n, int k) {
    return ritz_values_detail::BlockedPlan<T>::template make<Tier, Cfg>(n, k).size;
}

/// @brief The host ritz_values() inside a kernel on the in-group gemm, the host route's
/// algorithm: per block of NB columns, P := A V(:, blk) (gemm), then w_j = ritz_values_quotient(v_j, p_j).
/// @tparam Tier  tier::Blocked<NB>
/// @tparam Cfg   the l3::cfg preset of the gemm
/// @param[in]  ex  a WorkGroup<Cfg::threads, SG> (collective); A and V read only, visible on entry
/// @param[in]  a   A, n x n, a Mat over global memory
/// @param[in]  v   V, n x k, a Mat over global memory (col(), sub())
/// @param[out] w   k reals, written by the leader; visible to every item on return
/// @param[in]  ws  this matrix's slice of ritz_values_buffer_size<Tier, T, Cfg>(n, k)
/// @return 0 (the host op has no info)
template <class Tier, class Cfg, class Ex, class MA, class MV, class W>
BATCHLAS_DEVICE_INLINE int ritz_values(const Ex& ex, const MA& a, const MV& v, const W& w, const Workspace& ws) {
    using T = typename MA::value_type;
    constexpr int NB = Tier::nb;
    const int n = a.rows(), k = v.cols();
    const auto pl = ritz_values_detail::BlockedPlan<T>::template make<Tier, Cfg>(n, k);
    const l3::Smem<Cfg> sm{carve_local(ws, pl.smem).get_raw()};
    T* const p = carve_global(ws, pl.av);
    for (int c0 = 0; c0 < k; c0 += NB) {
        const int nb = k - c0 < NB ? k - c0 : NB;
        gemm<Transpose::NoTrans, Transpose::NoTrans, Cfg>(ex, a, v.sub(0, c0, n, nb), mat(p, n, nb, n), One{}, Zero{}, sm);
        for (int j = 0; j < nb; ++j) {
            const auto q = ritz_values_quotient(ex, v.col(c0 + j), vec(p + static_cast<std::ptrdiff_t>(j) * n, n));
            if (ex.leader()) w(c0 + j) = q;
        }
        ex.barrier(Mem::Global);  // P is read by every item before the next block's gemm writes it; w is published
    }
    return 0;
}

}  // namespace batchlas::device
