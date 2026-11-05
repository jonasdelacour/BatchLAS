// SPDX-License-Identifier: MIT
#pragma once
// Public device ortho (v22 design C.2, C.4, C.5, C.7). OrthoAlgorithm::CGS2 is the device
// algorithm today, the host CGS2 route's arithmetic (its normalize kernel calls
// ortho_detail::normalize). Not yet device ops: the rows arm, CholeskyQR (potrf + trsm), SVQB
// (syev) and Householder (geqrf + orgqr).

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/level2.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device::inline v2 {

namespace ortho_detail {
template <class Tier> inline constexpr bool is_cta_v = false;
template <int N, class A> inline constexpr bool is_cta_v<tier::Cta<N, A>> = true;
template <class Tier> inline constexpr bool is_blocked_v = false;
template <int NB, class A> inline constexpr bool is_blocked_v<tier::Blocked<NB, A>> = true;

template <Transpose TR, OrthoAlgorithm Algo, class Tier> constexpr void check() {
    static_assert(TR == Transpose::NoTrans, "ortho: the rows arm (TR != NoTrans) is not a device op yet");
    static_assert(Algo == OrthoAlgorithm::CGS2, "ortho: OrthoAlgorithm::CGS2 is the only device algorithm today");
    static_assert(is_cta_v<Tier> || is_blocked_v<Tier>, "ortho: tier::Cta<N> or tier::Blocked<NB>");
}

// The one workspace plan (buffer_size and the carve): h, k values of T, in the tier's space.
template <class T> struct Layout {
    ws::Slot<T> h;
    WorkspaceSize size;
};
template <class Tier, class T> constexpr Layout<T> layout(int k) {
    ::batchlas::device::Plan p;
    const auto h = p.take<T>(is_cta_v<Tier> ? ws::Space::Local : ws::Space::Global, static_cast<std::size_t>(k));
    return {h, p.size()};
}

// v := v / sqrt(sum abs2(v_i)), a true division. Collective.
template <class Ex, class V> BATCHLAS_DLA_INLINE void normalize(const Ex& ex, const V& v) {
    using T = elem_t<V>;
    using R = real_t<T>;
    const R nrm = std::sqrt(sumsq<alg::Unfused>(ex, v));
    map(ex, v, [=](T x) { return Arith<T>::div_real(x, nrm); }, v);
}

// Per column i: twice h := A(:, :i)^H a_i, a_i -= A(:, :i) h; then normalize. The RowOwner
// gemv and normalize own a_i cyclically by the same item, so only cross-owner reads publish.
template <class Ex, class MA, class H> BATCHLAS_DLA_INLINE void cgs2(const Ex& ex, const MA& a, const H& h) {
    const int m = a.rows(), k = a.cols();
    for (int i = 0; i < k; ++i) {
        const auto ai = a.col(i);
        if (i > 0) {
            const auto prev = a.sub(0, 0, m, i);
            const auto hi = h.head(i);
            for (int pass = 0; pass < 2; ++pass) {
                gemv<Transpose::ConjTrans>(ex, One{}, prev, ex.publish(ai), Zero{}, hi);
                gemv<Transpose::NoTrans>(ex, NegOne{}, prev, ex.publish(hi), One{}, ai);
            }
        }
        normalize(ex, ai);
    }
    ex.barrier(Mem::Global);
}
}  // namespace ortho_detail

/// @brief Workspace of ONE matrix for ortho() on k columns (constexpr, host and device).
/// @tparam Tier  tier::Cta<N> (local bytes) or tier::Blocked<NB> (global bytes)
/// @param[in] k  the number of columns
/// @return local and global bytes; the caller's global buffer holds batch times global_bytes
template <Transpose TR, OrthoAlgorithm Algo, class Tier, class T> constexpr WorkspaceSize ortho_buffer_size(int k) {
    ortho_detail::check<TR, Algo, Tier>();
    return ortho_detail::layout<Tier, T>(k).size;
}

/// @brief The host ortho() inside a kernel: the k columns of the m x k A (k <= m) become an
/// orthonormal basis of their span, one matrix per work-group.
/// @tparam TR    Transpose::NoTrans (the columns)
/// @tparam Algo  OrthoAlgorithm::CGS2
/// @tparam Tier  tier::Cta<N> (k <= N, the coefficients in local memory) or tier::Blocked<NB>
///               (any k, the coefficients in global memory; NB is unused by CGS2)
/// @param[in]     ex  a WorkGroup or RtGroup (collective); A visible to every item on entry
/// @param[in,out] a   A, a Mat (col(), sub()); visible to every item on return
/// @param[in]     ws  this matrix's slice of ortho_buffer_size<TR, Algo, Tier, T>(k)
/// @return 0, or -2 when k exceeds a tier::Cta<N> bound (A untouched); uniform
template <Transpose TR, OrthoAlgorithm Algo, class Tier, class Ex, class MA>
BATCHLAS_DLA_INLINE int ortho(const Ex& ex, const MA& a, const Workspace& ws) {
    using T = typename MA::value_type;
    ortho_detail::check<TR, Algo, Tier>();
    const int k = a.cols();
    const auto lay = ortho_detail::layout<Tier, T>(k);
    if constexpr (ortho_detail::is_cta_v<Tier>) {
        if (k > Tier::n_max) return -2;
        ortho_detail::cgs2(ex, a, mat_local(carve_local(ws, lay.h), k, 1, k).col(0));
    } else {
        ortho_detail::cgs2(ex, a, vec(carve_global(ws, lay.h), k));
    }
    return 0;
}

}  // namespace batchlas::device
