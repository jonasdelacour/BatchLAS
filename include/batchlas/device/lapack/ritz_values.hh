// SPDX-License-Identifier: MIT
#pragma once
// Public device ritz_values (v22 design C.2, C.7). The host route forms A V with gemm / spmm
// and calls ritz_values_quotient per (item, column); ritz_values forms A v_j on read.

#include <batchlas/device/level1.hh>
#include <batchlas/device/view.hh>

namespace batchlas::device::inline v2 {

/// @brief One Ritz value from a given product: \f$\mathrm{Re}(v^H a_v) / (v^H v)\f$.
/// @param[in] ex  the executor (collective)
/// @param[in] v   the trial vector
/// @param[in] av  \f$A v\f$, same length as @p v
/// @return the quotient (\f$v^H v\f$ as sumsq<alg::Fused>), the same on every item
template <class Ex, class V, class AV> BATCHLAS_DLA_INLINE auto ritz_values_quotient(const Ex& ex, const V& v, const AV& av) {
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
template <class Ex, class MA, class MV, class W> BATCHLAS_DLA_INLINE int ritz_values(const Ex& ex, const MA& a, const MV& v, const W& w) {
    const int n = a.rows(), k = v.cols();
    for (int j = 0; j < k; ++j) {
        const auto vj = v.col(j);
        const auto av = fn_vec(n, [a, vj, n](int r) BATCHLAS_DLA_BODY {
            return dot(Thread{}, fn_vec(n, [a, r](int c) BATCHLAS_DLA_BODY { return a(r, c); }), vj);
        });
        const auto q = ritz_values_quotient(ex, vj, av);
        if (ex.leader()) w(j) = q;
    }
    return 0;
}

}  // namespace batchlas::device
