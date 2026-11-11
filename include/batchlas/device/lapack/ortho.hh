// SPDX-License-Identifier: MIT
#pragma once
// Public device ortho (v22 design C.2, C.4, C.5, C.7, B8), the column arm (TR = NoTrans):
//   CGS2          tier::Cta<N>       k <= N; per column twice h := A(:, :i)^H a_i, a_i -= A(:, :i) h,
//                                    then normalize (the host CGS2 route's normalize kernel calls it)
//                 tier::Blocked<NB>  any k: block CGS2 over NB-column blocks on the in-group gemm
//   Cholesky, Chol2, ShiftChol3
//                 tier::Cta<N>       k <= N, any m: G := A^H A (herk), potrf<Lower, Cta<N>> on G,
//                                    A := A L^-H (trsm); the host's passes and ShiftChol3 shift
// The Level 3 forms take an l3::cfg preset Cfg (Cfg::threads == ex.size, Cfg::value_type == T).
// Not yet device ops: the rows arm, the Cholesky algorithms on tier::Blocked (potrf tier::Blocked,
// v22 B1), SVQB (a device syev) and Householder (geqrf + orgqr).

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/lapack/potrf.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/level2.hh>
#include <batchlas/device/level3/group.hh>
#include <batchlas/device/scalar.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

namespace detail {
template <OrthoAlgorithm A>
inline constexpr bool is_chol_v = A == OrthoAlgorithm::Cholesky || A == OrthoAlgorithm::Chol2 || A == OrthoAlgorithm::ShiftChol3;
template <OrthoAlgorithm A, class Tier> inline constexpr bool on_l3_v = is_chol_v<A> || tier::is_blocked_v<Tier>;

template <Transpose TR, OrthoAlgorithm Algo, class Tier, class Cfg, class T> constexpr void check() {
    static_assert(TR == Transpose::NoTrans, "ortho: the rows arm (TR != NoTrans) is not a device op yet");
    static_assert(Algo == OrthoAlgorithm::CGS2 || is_chol_v<Algo>,
                  "ortho: CGS2, Cholesky, Chol2 and ShiftChol3 are the device algorithms (SVQB needs a device syev, "
                  "Householder a device geqrf + orgqr)");
    static_assert(tier::is_cta_v<Tier> || tier::is_blocked_v<Tier>, "ortho: tier::Cta<N> or tier::Blocked<NB>");
    static_assert(!is_chol_v<Algo> || tier::is_cta_v<Tier>,
                  "ortho: the Cholesky algorithms take tier::Cta<N>; tier::Blocked waits for potrf tier::Blocked (v22 B1)");
    if constexpr (on_l3_v<Algo, Tier>) {
        static_assert(l3::Config<Cfg>, "ortho: this tier runs on the in-group Level 3, Cfg is an l3::cfg preset");
        if constexpr (l3::Config<Cfg>) static_assert(std::is_same_v<typename Cfg::value_type, T>, "ortho: Cfg::value_type is A's T");
    } else {
        static_assert(std::is_void_v<Cfg>, "ortho: CGS2 tier::Cta takes no Cfg");
    }
}

// The one workspace plan (buffer_size and the carve). Local: the Level 3 stages, potrf's tile or
// the Cta coefficients h (k); global: Blocked's H (k x NB, ld k) and h (NB), or the Gram k x k.
template <class T> struct OrthoLayout {
    WsSlot<T> smem, h, hb, gram;
    WsSlot<std::byte> chol;
    WorkspaceSize size;
};
template <OrthoAlgorithm Algo, class Tier, class Cfg, class T> constexpr OrthoLayout<T> ortho_layout(int k) {
    const std::size_t uk = static_cast<std::size_t>(k);
    ::batchlas::device::Plan p;
    OrthoLayout<T> l{};
    if constexpr (on_l3_v<Algo, Tier>) l.smem = p.take<T>(WsSpace::Local, static_cast<std::size_t>(l3::smem_elems<Cfg>()));
    if constexpr (is_chol_v<Algo>) {
        l.chol = p.take<std::byte>(WsSpace::Local, potrf_buffer_size<Uplo::Lower, tier::Cta<Tier::n_max>, T>(k).local_bytes);
        l.gram = p.take<T>(WsSpace::Global, uk * uk);
    } else if constexpr (tier::is_cta_v<Tier>) {
        l.h = p.take<T>(WsSpace::Local, uk);
    } else {
        l.hb = p.take<T>(WsSpace::Global, uk * static_cast<std::size_t>(Tier::nb));
        l.h = p.take<T>(WsSpace::Global, static_cast<std::size_t>(Tier::nb));
    }
    l.size = p.size();
    return l;
}

// v := v / sqrt(sum abs2(v_i)), a true division. Collective.
template <class Ex, class V> BATCHLAS_DEVICE_INLINE void normalize(const Ex& ex, const V& v) {
    using T = elem_t<V>;
    using R = real_t<T>;
    const R nrm = std::sqrt(sumsq<alg::Unfused>(ex, v));
    map(ex, v, [=](T x) { return Arith<T>::div_real(x, nrm); }, v);
}

// Per column i: twice h := A(:, :i)^H a_i, a_i -= A(:, :i) h; then normalize. The RowOwner
// gemv and normalize own a_i cyclically by the same item, so only cross-owner reads publish.
template <class Ex, class MA, class H> BATCHLAS_DEVICE_INLINE void cgs2(const Ex& ex, const MA& a, const H& h) {
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

// Block CGS2 (Barlow and Smoktunowicz): per NB-column block A_j, twice { H := Q^H A_j;
// A_j -= Q H; cgs2(A_j) } with Q the columns left of the block; the first block runs cgs2 once.
// The two projections are in-group gemms (H: c0 x nb at ld k); every step ends on a global barrier.
template <int NB, class Cfg, class Ex, class MA, class T>
BATCHLAS_DEVICE_INLINE void bcgs2(const Ex& ex, const MA& a, T* hb, T* h, l3::Smem<Cfg> sm) {
    const int m = a.rows(), k = a.cols();
    const auto g = l3::global_of(a);
    for (int c0 = 0; c0 < k; c0 += NB) {
        const int nb = k - c0 < NB ? k - c0 : NB;
        const auto blk = a.sub(0, c0, m, nb);
        const int passes = c0 > 0 ? 2 : 1;
        for (int pass = 0; pass < passes; ++pass) {
            if (c0 > 0) {
                const l3::Global<T> q{g.p, g.ld, m, c0};
                const l3::Global<T> aj{g.p + static_cast<std::ptrdiff_t>(c0) * g.ld, g.ld, m, nb};
                const l3::Global<T> hj{hb, k, c0, nb};
                gemm<Transpose::ConjTrans, Transpose::NoTrans, Cfg>(ex, q, aj, hj, One{}, Zero{}, sm);
                gemm<Transpose::NoTrans, Transpose::NoTrans, Cfg>(ex, q, hj, aj, NegOne{}, One{}, sm);
            }
            cgs2(ex, blk, vec(h, nb));
        }
    }
}

// One CholeskyQR pass: G := A^H A (its lower triangle), for ShiftChol3's first pass G += s I with
// the host's s, then G = L L^H and A := A L^-H. Returns potrf's info (k > 0: A not changed).
template <int N, bool Shift, class Cfg, class Ex, class MA, class T>
BATCHLAS_DEVICE_INLINE int cholqr_pass(const Ex& ex, const MA& a, T* gp, const Workspace& pws, l3::Smem<Cfg> sm) {
    using R = real_t<T>;
    const int m = a.rows(), k = a.cols();
    const auto gram = mat(gp, k, k, k);
    herk<Uplo::Lower, Transpose::ConjTrans, Cfg>(ex, a, gram, One{}, Zero{}, sm);
    if constexpr (Shift) {
        // trace(A^H A) = ||A||_F^2 bounds ||A||_2^2, so the shift scales with the Gram matrix.
        const R fro2 = reduce(ex, k, [&](int i) { return Arith<T>::re(gram(i, i)); });
        const R shift = R(11) * (R(m * k) + R(k + 1) * R(k)) * lamch<'P', R>() * fro2;
        for_each_owned(ex, k, [&](int i) { gram(i, i) = Arith<T>::add(gram(i, i), T(shift)); });
        ex.barrier(Mem::Global);
    }
    const int info = potrf<Uplo::Lower, tier::Cta<N>>(ex, gram, pws);
    if (info != 0) return info;
    trsm<Side::Right, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit, Cfg>(ex, gram, a, One{}, sm);
    return 0;
}
}  // namespace detail

/// @brief Workspace of ONE matrix for ortho() on k columns (constexpr, host and device).
/// @tparam Algo  CGS2, Cholesky, Chol2 or ShiftChol3
/// @tparam Tier  tier::Cta<N> or tier::Blocked<NB> (see ortho())
/// @tparam Cfg   the l3::cfg preset of the Level 3 forms; void for CGS2 tier::Cta
/// @param[in] k  the number of columns
/// @return local and global bytes; the caller's global buffer holds batch times global_bytes
template <Transpose TR, OrthoAlgorithm Algo, class Tier, class T, class Cfg = void> constexpr WorkspaceSize ortho_buffer_size(int k) {
    detail::check<TR, Algo, Tier, Cfg, T>();
    return detail::ortho_layout<Algo, Tier, Cfg, T>(k).size;
}

/// @brief The host ortho() inside a kernel: the k columns of the m x k A (k <= m) become an
/// orthonormal basis of their span, one matrix per work-group.
/// @tparam TR    Transpose::NoTrans (the columns)
/// @tparam Algo  CGS2: tier::Cta<N> (k <= N, the coefficients in local memory) or tier::Blocked<NB>
///               (any k, block CGS2 over NB columns on the in-group gemm); Cholesky, Chol2,
///               ShiftChol3: tier::Cta<N> (k <= N, any m; herk + potrf<Lower, Cta<N>> + trsm)
/// @tparam Cfg   the l3::cfg preset of every form except CGS2 tier::Cta (void there)
/// @param[in]     ex  a WorkGroup (collective; WorkGroup<Cfg::threads, 32> on the Level 3 forms, an
///                    RtGroup also runs CGS2 tier::Cta); A visible to every item on entry
/// @param[in,out] a   A, a Mat over global memory (col(), sub()); visible to every item on return
/// @param[in]     ws  this matrix's slice of ortho_buffer_size<TR, Algo, Tier, T, Cfg>(k)
/// @return 0; -2 when k exceeds a tier::Cta<N> bound (A untouched); for the Cholesky algorithms
///         potrf's info k > 0 when a pass's Gram matrix is not positive definite (that pass leaves
///         A as the previous pass left it); uniform
template <Transpose TR, OrthoAlgorithm Algo, class Tier, class Cfg = void, class Ex, class MA>
BATCHLAS_DEVICE_INLINE int ortho(const Ex& ex, const MA& a, const Workspace& ws) {
    using T = typename MA::value_type;
    detail::check<TR, Algo, Tier, Cfg, T>();
    const int k = a.cols();
    const auto lay = detail::ortho_layout<Algo, Tier, Cfg, T>(k);
    if constexpr (detail::is_chol_v<Algo>) {
        constexpr int N = Tier::n_max;
        if (k > N) return -2;
        if (k == 0) return 0;
        const l3::Smem<Cfg> sm{carve_local(ws, lay.smem).get_raw()};
        const Workspace pws{carve_local(ws, lay.chol), nullptr};
        T* const g = carve_global(ws, lay.gram);
        int info = detail::cholqr_pass<N, Algo == OrthoAlgorithm::ShiftChol3>(ex, a, g, pws, sm);
        if constexpr (Algo != OrthoAlgorithm::Cholesky) {
            if (info == 0) info = detail::cholqr_pass<N, false>(ex, a, g, pws, sm);
        }
        if constexpr (Algo == OrthoAlgorithm::ShiftChol3) {
            if (info == 0) info = detail::cholqr_pass<N, false>(ex, a, g, pws, sm);
        }
        return info;
    } else if constexpr (tier::is_cta_v<Tier>) {
        if (k > Tier::n_max) return -2;
        detail::cgs2(ex, a, mat_local(carve_local(ws, lay.h), k, 1, k).col(0));
    } else {
        const l3::Smem<Cfg> sm{carve_local(ws, lay.smem).get_raw()};
        detail::bcgs2<Tier::nb>(ex, a, carve_global(ws, lay.hb), carve_global(ws, lay.h), sm);
    }
    return 0;
}

}  // namespace batchlas::device
