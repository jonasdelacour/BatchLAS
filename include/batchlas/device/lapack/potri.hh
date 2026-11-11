// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xPOTRI, xTRTRI and xLAUUM inside a kernel (v22 design C.1, C.2, C.4-C.7). Public entry:
//   int potri<UL, Tier>(ex, A, ws)        A's UL triangle: potrf's factor -> that of A^-1
//   int trtri<UL, DG, Tier>(ex, A, ws)    the UL triangle T -> T^-1 (DG = Unit: diagonal implied)
//   int lauum<UL, Tier>(ex, A, ws)        L -> L^H L (Lower), U -> U U^H (Upper), in its triangle
//   WorkspaceSize potri_buffer_size<UL, Tier, T>(n), trtri_buffer_size<UL, DG, Tier, T>(n),
//                 lauum_buffer_size<UL, Tier, T>(n)
//   tier::Blocked<NB>  ex = WorkGroup<256, 32>, any n: LAPACK's blocked trtri and lauum over
//                      NB-wide blocks on the in-group L3 (trsm, trmm, gemm, herk); ws.local = the
//                      L3 stages, ws.global = an n x NB panel of T per matrix (trmm is out of place)
// The CTA tier (the host potri cta body) lands with WP7. Every op reads and writes only the UL
// triangle; the diagonal of a potri / lauum result is real. Collective; inputs visible on entry,
// results visible on return. info (potri, trtri NonUnit) = i > 0: the factor's diagonal entry i is
// exactly zero and A is unchanged; -2: A not square; -3: ws.local or ws.global missing or misaligned.

#include <cstddef>
#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/lapack/potrf.hh>
#include <batchlas/device/lapack_aux.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/level3/group.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

namespace potri_detail {

// One plan for every op here: the L3 stage slot (local) and the n x NB panel W (global). Each
// use of W takes its own leading dimension, inside the n * NB elements.
template <class T> struct BlockedPanelPlan {
    ws::Slot<T> smem;
    ws::Slot<T> w;
    WorkspaceSize size;
    static constexpr BlockedPanelPlan make(int n, int nb) {
        Plan p;
        const auto s = p.take<T>(ws::Space::Local, static_cast<std::size_t>(potrf_detail::BlockedCfg<T>::smem_elems));
        const auto g = p.take<T>(ws::Space::Global, static_cast<std::size_t>(n) * static_cast<std::size_t>(nb));
        return {s, g, p.size()};
    }
};

// The UL triangle T of the n x n A := T^-1, A in global memory, LAPACK xTRTRI's block order:
// Lower right to left, A21 := -A22^-1 A21 A11^-1 (A22 already inverted); Upper left to right,
// A12 := -A11^-1 A12 A22^-1. The trsm applies the not yet inverted diagonal block in place, the
// trmm writes W, W is copied back. The diagonal block is inverted as A11^-1 I (W := I, trsm),
// then its triangle copied into A. Ends on a global barrier.
template <Uplo UL, Diag DG, int NB, class Cfg, class Ex, class MA, class T>
BATCHLAS_DEVICE_INLINE void trtri_blocked_body(const Ex& ex, const MA& A, T* w, l3::Smem<Cfg> sm) {
    constexpr bool unit = DG == Diag::Unit;
    constexpr Part kTri = UL == Uplo::Lower ? (unit ? Part::StrictLower : Part::Lower) : (unit ? Part::StrictUpper : Part::Upper);
    const int n = A.rows();
    auto invert_diag = [&](int j, int jb) BATCHLAS_DEVICE_BODY {
        const auto W = mat(w, jb, jb, jb);
        const auto A11 = A.sub(j, j, jb, jb);
        laset(ex, Zero{}, One{}, W);
        ex.barrier(Mem::Global);
        trsm<Side::Left, UL, Transpose::NoTrans, DG, Cfg, NB>(ex, A11, W, One{}, sm);
        lacpy<kTri>(ex, W, A11);
        ex.barrier(Mem::Global);
    };
    if constexpr (UL == Uplo::Lower) {
        for (int j = (n - 1) / NB * NB; j >= 0; j -= NB) {
            const int jb = n - j < NB ? n - j : NB;
            const int rest = n - j - jb;
            if (rest > 0) {
                const auto A21 = A.sub(j + jb, j, rest, jb);
                const auto W = mat(w, rest, jb, rest);
                trsm<Side::Right, Uplo::Lower, Transpose::NoTrans, DG, Cfg, NB>(ex, A.sub(j, j, jb, jb), A21, NegOne{}, sm);
                trmm<Side::Left, Uplo::Lower, Transpose::NoTrans, DG, Cfg>(ex, A.sub(j + jb, j + jb, rest, rest), A21, W, One{}, sm);
                lacpy(ex, W, A21);
                ex.barrier(Mem::Global);
            }
            invert_diag(j, jb);
        }
    } else {
        for (int j = 0; j < n; j += NB) {
            const int jb = n - j < NB ? n - j : NB;
            if (j > 0) {
                const auto A12 = A.sub(0, j, j, jb);
                const auto W = mat(w, j, jb, j);
                trsm<Side::Right, Uplo::Upper, Transpose::NoTrans, DG, Cfg, NB>(ex, A.sub(j, j, jb, jb), A12, NegOne{}, sm);
                trmm<Side::Left, Uplo::Upper, Transpose::NoTrans, DG, Cfg>(ex, A.sub(0, 0, j, j), A12, W, One{}, sm);
                lacpy(ex, W, A12);
                ex.barrier(Mem::Global);
            }
            invert_diag(j, jb);
        }
    }
}

// L := L^H L (Lower) / U := U U^H (Upper) in place, LAPACK xLAUUM's block order (left to right):
// the off-diagonal block row (column) := its trmm by the diagonal block plus the gemm with the
// blocks below (right), through W; the diagonal block := its own herk (a zero-padded copy in W,
// beta 0: xLAUU2) plus the herk of the blocks below (right). Ends on a global barrier.
template <Uplo UL, int NB, class Cfg, class Ex, class MA, class T>
BATCHLAS_DEVICE_INLINE void lauum_blocked_body(const Ex& ex, const MA& A, T* w, l3::Smem<Cfg> sm) {
    constexpr Transpose kCT = Transpose::ConjTrans, kN = Transpose::NoTrans;
    const int n = A.rows();
    for (int i = 0; i < n; i += NB) {
        const int ib = n - i < NB ? n - i : NB;
        const int rest = n - i - ib;
        const auto A11 = A.sub(i, i, ib, ib);
        const auto W2 = mat(w, ib, ib, ib);
        if constexpr (UL == Uplo::Lower) {
            if (i > 0) {
                const auto Ai0 = A.sub(i, 0, ib, i);
                const auto W = mat(w, ib, i, ib);
                trmm<Side::Left, Uplo::Lower, kCT, Diag::NonUnit, Cfg>(ex, A11, Ai0, W, One{}, sm);
                if (rest > 0) gemm<kCT, kN, Cfg>(ex, A.sub(i + ib, i, rest, ib), A.sub(i + ib, 0, rest, i), W, One{}, One{}, sm);
                lacpy(ex, W, Ai0);
                ex.barrier(Mem::Global);
            }
            lacpy<Part::Lower, kN, alg::Walk, edge::ZeroPad>(ex, A11, W2);
            ex.barrier(Mem::Global);
            herk<Uplo::Lower, kCT, Cfg>(ex, W2, A11, One{}, Zero{}, sm);
            if (rest > 0) herk<Uplo::Lower, kCT, Cfg>(ex, A.sub(i + ib, i, rest, ib), A11, One{}, One{}, sm);
        } else {
            if (i > 0) {
                const auto A0i = A.sub(0, i, i, ib);
                const auto W = mat(w, i, ib, i);
                trmm<Side::Right, Uplo::Upper, kCT, Diag::NonUnit, Cfg>(ex, A11, A0i, W, One{}, sm);
                if (rest > 0) gemm<kN, kCT, Cfg>(ex, A.sub(0, i + ib, i, rest), A.sub(i, i + ib, ib, rest), W, One{}, One{}, sm);
                lacpy(ex, W, A0i);
                ex.barrier(Mem::Global);
            }
            lacpy<Part::Upper, kN, alg::Walk, edge::ZeroPad>(ex, A11, W2);
            ex.barrier(Mem::Global);
            herk<Uplo::Upper, kN, Cfg>(ex, W2, A11, One{}, Zero{}, sm);
            if (rest > 0) herk<Uplo::Upper, kN, Cfg>(ex, A.sub(i, i + ib, ib, rest), A11, One{}, One{}, sm);
        }
    }
}

// The shared prologue: -2 / -3 argument checks, then (zero_check) the first exactly zero
// diagonal entry as info. 0 means: go on with the carved pointers.
template <class Ex, class MA> BATCHLAS_DEVICE_INLINE int potri_blocked_prologue(const Ex& ex, const MA& a, const Workspace& ws, bool zero_check) {
    using T = typename MA::value_type;
    const int n = a.rows();
    if (n < 0 || a.cols() != n) return -2;
    if (n == 0) return 0;
    if (!potrf_detail::blocked_ws_ok(ws, true)) return -3;
    if (!zero_check) return 0;
    const int z = find_first(ex, n, [&](int i) BATCHLAS_DEVICE_BODY { return Arith<T>::is_zero(a(i, i)); });
    return z + 1;
}

// Which of the two bodies an entry runs.
enum class BlockedChain { Trtri, Lauum, Potri };

template <BlockedChain C, Uplo UL, Diag DG, int NB, class Alg, int WG, int SG, class MA>
BATCHLAS_DEVICE_INLINE int potri_blocked_op(const WorkGroup<WG, SG>& ex, const MA& a, const Workspace& ws) {
    using T = typename MA::value_type;
    using Cfg = potrf_detail::BlockedCfg<T>;
    static_assert(std::is_same_v<Alg, alg::Default>, "potri / trtri / lauum: tier::Blocked has one algorithm");
    static_assert(NB >= 1, "potri / trtri / lauum: tier::Blocked<NB> needs NB >= 1");
    static_assert(WG == Cfg::threads && SG == 32, "potri / trtri / lauum: tier::Blocked runs on WorkGroup<256, 32>");
    const int n = a.rows();
    const int pre = potri_blocked_prologue(ex, a, ws, C != BlockedChain::Lauum && DG == Diag::NonUnit);
    if (pre != 0 || n == 0) return pre;
    const auto pl = BlockedPanelPlan<T>::make(n, NB);
    const l3::Smem<Cfg> sm{carve_local(ws, pl.smem).get_raw()};
    T* const w = carve_global(ws, pl.w);
    const auto A = mat(&a(0, 0), n, n, static_cast<int>(a.ld));
    if constexpr (C != BlockedChain::Lauum) trtri_blocked_body<UL, DG, NB, Cfg>(ex, A, w, sm);
    if constexpr (C != BlockedChain::Trtri) lauum_blocked_body<UL, NB, Cfg>(ex, A, w, sm);
    return 0;
}

}  // namespace potri_detail

namespace routine {
template <Uplo UL> struct potri {
    static constexpr const char* name = "potri";
    static constexpr Uplo uplo = UL;
};
template <Uplo UL, Diag DG> struct trtri {
    static constexpr const char* name = "trtri";
    static constexpr Uplo uplo = UL;
    static constexpr Diag diag = DG;
};
template <Uplo UL> struct lauum {
    static constexpr const char* name = "lauum";
    static constexpr Uplo uplo = UL;
};
}  // namespace routine

template <Uplo UL, int NB, class Alg, class T> struct tier_info<routine::potri<UL>, tier::Blocked<NB, Alg>, T> {
    static constexpr int wg_size = potrf_detail::kBlockedWg;
    static constexpr int sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int n) { return potri_detail::BlockedPanelPlan<T>::make(n, NB).size; }
};
template <Uplo UL, Diag DG, int NB, class Alg, class T> struct tier_info<routine::trtri<UL, DG>, tier::Blocked<NB, Alg>, T> {
    static constexpr int wg_size = potrf_detail::kBlockedWg;
    static constexpr int sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int n) { return potri_detail::BlockedPanelPlan<T>::make(n, NB).size; }
};
template <Uplo UL, int NB, class Alg, class T> struct tier_info<routine::lauum<UL>, tier::Blocked<NB, Alg>, T> {
    static constexpr int wg_size = potrf_detail::kBlockedWg;
    static constexpr int sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int n) { return potri_detail::BlockedPanelPlan<T>::make(n, NB).size; }
};
template <Uplo UL, class T> struct tiers<routine::potri<UL>, T> {
    using type = tier::list<tier::Blocked<32>>;
};
template <Uplo UL, Diag DG, class T> struct tiers<routine::trtri<UL, DG>, T> {
    using type = tier::list<tier::Blocked<32>>;
};
template <Uplo UL, class T> struct tiers<routine::lauum<UL>, T> {
    using type = tier::list<tier::Blocked<32>>;
};

// The workspace of one matrix (host and device). Tier may be tier::upto<NMax>.
template <Uplo UL, class Tier, class T> constexpr WorkspaceSize potri_buffer_size(int n) {
    using Tr = resolve_tier_t<routine::potri<UL>, T, Tier>;
    return tier_info<routine::potri<UL>, Tr, T>::buffer_size(n);
}
template <Uplo UL, Diag DG, class Tier, class T> constexpr WorkspaceSize trtri_buffer_size(int n) {
    using Tr = resolve_tier_t<routine::trtri<UL, DG>, T, Tier>;
    return tier_info<routine::trtri<UL, DG>, Tr, T>::buffer_size(n);
}
template <Uplo UL, class Tier, class T> constexpr WorkspaceSize lauum_buffer_size(int n) {
    using Tr = resolve_tier_t<routine::lauum<UL>, T, Tier>;
    return tier_info<routine::lauum<UL>, Tr, T>::buffer_size(n);
}

// A's UL triangle, potrf's factor of the Hermitian positive definite A, := that of A^-1 (the other
// triangle neither read nor written); returns info (see the file head).
template <Uplo UL, class Tier, class Ex, class MA> BATCHLAS_DEVICE_INLINE int potri(const Ex& ex, const MA& A, const Workspace& ws) {
    using T = typename MA::value_type;
    using Tr = resolve_tier_t<routine::potri<UL>, T, Tier>;
    static_assert(potrf_detail::is_blocked_v<Tr>, "potri: tier::Blocked<NB> (the CTA tier lands with WP7)");
    return potri_detail::potri_blocked_op<potri_detail::BlockedChain::Potri, UL, Diag::NonUnit, Tr::nb, typename Tr::alg_type>(ex, A, ws);
}

// A's UL triangle := its inverse; DG = Unit never reads or writes the diagonal.
template <Uplo UL, Diag DG, class Tier, class Ex, class MA> BATCHLAS_DEVICE_INLINE int trtri(const Ex& ex, const MA& A, const Workspace& ws) {
    using T = typename MA::value_type;
    using Tr = resolve_tier_t<routine::trtri<UL, DG>, T, Tier>;
    static_assert(potrf_detail::is_blocked_v<Tr>, "trtri: tier::Blocked<NB>");
    return potri_detail::potri_blocked_op<potri_detail::BlockedChain::Trtri, UL, DG, Tr::nb, typename Tr::alg_type>(ex, A, ws);
}

// A's UL triangle := L^H L (Lower) or U U^H (Upper) of that triangle; returns 0 or -i.
template <Uplo UL, class Tier, class Ex, class MA> BATCHLAS_DEVICE_INLINE int lauum(const Ex& ex, const MA& A, const Workspace& ws) {
    using T = typename MA::value_type;
    using Tr = resolve_tier_t<routine::lauum<UL>, T, Tier>;
    static_assert(potrf_detail::is_blocked_v<Tr>, "lauum: tier::Blocked<NB>");
    return potri_detail::potri_blocked_op<potri_detail::BlockedChain::Lauum, UL, Diag::NonUnit, Tr::nb, typename Tr::alg_type>(ex, A, ws);
}

}  // namespace batchlas::device
