// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @ingroup api_device_lapack_chol
/// @brief ?POTRI, ?TRTRI and ?LAUUM in a kernel: the inverse of a Hermitian positive definite
/// matrix from potrf()'s factor, and the two steps it is made of.
///
/// The routines are potri(), trtri() (the inverse of a triangular matrix) and lauum() (the
/// product of a triangle with its conjugate transpose); their routine tags are routine::potri,
/// routine::trtri and routine::lauum. Each has one tier, `tier::Blocked<32>`: LAPACK's blocked
/// trtri and lauum over NB-wide blocks on the in-group level 3 (trsm, trmm, gemm, herk). Every
/// routine here reads and writes only the `UL` triangle.
// The CTA tier (the host potri cta body) lands with WP7.

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

namespace detail {

// One plan for every op here: the L3 stage slot (local) and the n x NB panel W (global). Each
// use of W takes its own leading dimension, inside the n * NB elements.
template <class Scalar> struct BlockedPanelPlan {
    WsSlot<Scalar> smem;
    WsSlot<Scalar> w;
    WorkspaceSize size;
    static constexpr BlockedPanelPlan make(int n, int nb) {
        Plan p;
        const auto s = p.take<Scalar>(WsSpace::Local,
                                      static_cast<std::size_t>(detail::PotrfBlockedCfg<Scalar>::smem_elems));
        const auto g = p.take<Scalar>(WsSpace::Global, static_cast<std::size_t>(n) * static_cast<std::size_t>(nb));
        return {s, g, p.size()};
    }
};

// The UL triangle T of the n x n A := T^-1, A in global memory, LAPACK xTRTRI's block order:
// Lower right to left, A21 := -A22^-1 A21 A11^-1 (A22 already inverted); Upper left to right,
// A12 := -A11^-1 A12 A22^-1. The trsm applies the not yet inverted diagonal block in place, the
// trmm writes W, W is copied back. The diagonal block is inverted as A11^-1 I (W := I, trsm),
// then its triangle copied into A. Ends on a global barrier.
template <Uplo UL, Diag DG, int NB, class Cfg, class Ex, class MA, class Scalar>
BATCHLAS_DEVICE_INLINE void trtri_blocked_body(const Ex& ex, const MA& A, Scalar* w, const Workspace& sm) {
    constexpr bool unit = DG == Diag::Unit;
    constexpr Part kTri = UL == Uplo::Lower ? (unit ? Part::StrictLower : Part::Lower) : (unit ? Part::StrictUpper : Part::Upper);
    const int n = A.rows();
    auto invert_diag = [&](int j, int jb) BATCHLAS_DEVICE_BODY {
        const auto W = mat(w, jb, jb, jb);
        const auto A11 = A.sub(j, j, jb, jb);
        laset(ex, Zero{}, One{}, W);
        ex.barrier(Mem::Global);
        trsm<Side::Left, UL, Transpose::NoTrans, DG, Cfg, NB>(ex, One{}, A11, W, sm);
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
                trsm<Side::Right, Uplo::Lower, Transpose::NoTrans, DG, Cfg, NB>(ex, NegOne{}, A.sub(j, j, jb, jb), A21, sm);
                trmm_out<Side::Left, Uplo::Lower, Transpose::NoTrans, DG, Cfg>(ex, One{}, A.sub(j + jb, j + jb, rest, rest), A21, W, sm);
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
                trsm<Side::Right, Uplo::Upper, Transpose::NoTrans, DG, Cfg, NB>(ex, NegOne{}, A.sub(j, j, jb, jb), A12, sm);
                trmm_out<Side::Left, Uplo::Upper, Transpose::NoTrans, DG, Cfg>(ex, One{}, A.sub(0, 0, j, j), A12, W, sm);
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
template <Uplo UL, int NB, class Cfg, class Ex, class MA, class Scalar>
BATCHLAS_DEVICE_INLINE void lauum_blocked_body(const Ex& ex, const MA& A, Scalar* w, const Workspace& sm) {
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
                trmm_out<Side::Left, Uplo::Lower, kCT, Diag::NonUnit, Cfg>(ex, One{}, A11, Ai0, W, sm);
                if (rest > 0) gemm<kCT, kN, Cfg>(ex, One{}, A.sub(i + ib, i, rest, ib), A.sub(i + ib, 0, rest, i), One{}, W, sm);
                lacpy(ex, W, Ai0);
                ex.barrier(Mem::Global);
            }
            lacpy<Part::Lower, kN, alg::Walk, alg::EdgeZeroPad>(ex, A11, W2);
            ex.barrier(Mem::Global);
            herk<Uplo::Lower, kCT, Cfg>(ex, One{}, W2, Zero{}, A11, sm);
            if (rest > 0) herk<Uplo::Lower, kCT, Cfg>(ex, One{}, A.sub(i + ib, i, rest, ib), One{}, A11, sm);
        } else {
            if (i > 0) {
                const auto A0i = A.sub(0, i, i, ib);
                const auto W = mat(w, i, ib, i);
                trmm_out<Side::Right, Uplo::Upper, kCT, Diag::NonUnit, Cfg>(ex, One{}, A11, A0i, W, sm);
                if (rest > 0) gemm<kN, kCT, Cfg>(ex, One{}, A.sub(0, i + ib, i, rest), A.sub(i, i + ib, ib, rest), One{}, W, sm);
                lacpy(ex, W, A0i);
                ex.barrier(Mem::Global);
            }
            lacpy<Part::Upper, kN, alg::Walk, alg::EdgeZeroPad>(ex, A11, W2);
            ex.barrier(Mem::Global);
            herk<Uplo::Upper, kN, Cfg>(ex, One{}, W2, Zero{}, A11, sm);
            if (rest > 0) herk<Uplo::Upper, kN, Cfg>(ex, One{}, A.sub(i, i + ib, ib, rest), One{}, A11, sm);
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
    if (!detail::blocked_ws_ok(ws, true)) return -3;
    if (!zero_check) return 0;
    const int z = find_first(ex, fn_vec(n, [&](int i) BATCHLAS_DEVICE_BODY { return a(i, i); }),
                             [](const T& d) BATCHLAS_DEVICE_BODY { return Arith<T>::is_zero(d); });
    return z + 1;
}

// Which of the two bodies an entry runs.
enum class BlockedChain { Trtri, Lauum, Potri };

template <BlockedChain C, Uplo UL, Diag DG, int NB, class Alg, int WG, int SG, class MA>
BATCHLAS_DEVICE_INLINE int potri_blocked_op(const WorkGroup<WG, SG>& ex, const MA& a, const Workspace& ws) {
    using T = typename MA::value_type;
    using Cfg = detail::PotrfBlockedCfg<T>;
    static_assert(std::is_same_v<Alg, alg::Default>, "potri / trtri / lauum: tier::Blocked has one algorithm");
    static_assert(NB >= 1, "potri / trtri / lauum: tier::Blocked<NB> needs NB >= 1");
    static_assert(WG == Cfg::threads && SG == 32, "potri / trtri / lauum: tier::Blocked runs on WorkGroup<256, 32>");
    const int n = a.rows();
    const int pre = potri_blocked_prologue(ex, a, ws, C != BlockedChain::Lauum && DG == Diag::NonUnit);
    if (pre != 0 || n == 0) return pre;
    const auto pl = BlockedPanelPlan<T>::make(n, NB);
    const Workspace sm = sub_workspace(ws, pl.smem);
    T* const w = carve_global(ws, pl.w);
    const auto A = mat(&a(0, 0), n, n, static_cast<int>(a.ld));
    if constexpr (C != BlockedChain::Lauum) trtri_blocked_body<UL, DG, NB, Cfg>(ex, A, w, sm);
    if constexpr (C != BlockedChain::Trtri) lauum_blocked_body<UL, NB, Cfg>(ex, A, w, sm);
    return 0;
}

}  // namespace detail

namespace routine {
/// @ingroup api_device_lapack_chol
/// @brief The routine tag of potri() for the triangle `UL`: names the routine to `with_tier`,
/// `launch`, `tier_info`, `tiers_t` and `buffer_size`.
template <Uplo UL> struct potri {
    static constexpr const char* name = "potri";  ///< The routine's name, "potri".
    static constexpr Uplo uplo = UL;              ///< The triangle that holds the factor.
};
/// @ingroup api_device_lapack_chol
/// @brief The routine tag of trtri() for the triangle `UL` and diagonal `DG`.
template <Uplo UL, Diag DG> struct trtri {
    static constexpr const char* name = "trtri";  ///< The routine's name, "trtri".
    static constexpr Uplo uplo = UL;              ///< The triangle that is inverted.
    static constexpr Diag diag = DG;              ///< `Diag::Unit`: the diagonal is implied.
};
/// @ingroup api_device_lapack_chol
/// @brief The routine tag of lauum() for the triangle `UL`.
template <Uplo UL> struct lauum {
    static constexpr const char* name = "lauum";  ///< The routine's name, "lauum".
    static constexpr Uplo uplo = UL;              ///< The triangle that is read and written.
};
}  // namespace routine

/// @ingroup api_device_lapack_chol
/// @brief potri()'s blocked tier `tier::Blocked<NB>`: any n, one matrix per 256-item work-group,
/// `A` in global memory.
///
/// The executor is `WorkGroup<256, 32>`. `admits(n)` is n >= 0. `buffer_size(n)` has a local
/// part (the level 3 stages, independent of n) and a global part per matrix (an n x NB panel:
/// the level 3 trmm works out of place).
template <Uplo UL, int NB, class Alg, class Scalar> struct tier_info<routine::potri<UL>, tier::Blocked<NB, Alg>, Scalar> {
    static constexpr int wg_size = detail::kBlockedWg;
    static constexpr int sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int n) { return detail::BlockedPanelPlan<Scalar>::make(n, NB).size; }
};
/// @ingroup api_device_lapack_chol
/// @brief trtri()'s blocked tier `tier::Blocked<NB>`: as potri()'s, the same executor and the
/// same local and global workspace.
template <Uplo UL, Diag DG, int NB, class Alg, class Scalar> struct tier_info<routine::trtri<UL, DG>, tier::Blocked<NB, Alg>, Scalar> {
    static constexpr int wg_size = detail::kBlockedWg;
    static constexpr int sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int n) { return detail::BlockedPanelPlan<Scalar>::make(n, NB).size; }
};
/// @ingroup api_device_lapack_chol
/// @brief lauum()'s blocked tier `tier::Blocked<NB>`: as potri()'s, the same executor and the
/// same local and global workspace.
template <Uplo UL, int NB, class Alg, class Scalar> struct tier_info<routine::lauum<UL>, tier::Blocked<NB, Alg>, Scalar> {
    static constexpr int wg_size = detail::kBlockedWg;
    static constexpr int sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int n) { return detail::BlockedPanelPlan<Scalar>::make(n, NB).size; }
};
/// @ingroup api_device_lapack_chol
/// @brief potri()'s tiers for every `Scalar`: `Blocked<32>`.
template <Uplo UL, class Scalar> struct tiers<routine::potri<UL>, Scalar> {
    using type = tier::list<tier::Blocked<32>>;
};
/// @ingroup api_device_lapack_chol
/// @brief trtri()'s tiers for every `Scalar`: `Blocked<32>`.
template <Uplo UL, Diag DG, class Scalar> struct tiers<routine::trtri<UL, DG>, Scalar> {
    using type = tier::list<tier::Blocked<32>>;
};
/// @ingroup api_device_lapack_chol
/// @brief lauum()'s tiers for every `Scalar`: `Blocked<32>`.
template <Uplo UL, class Scalar> struct tiers<routine::lauum<UL>, Scalar> {
    using type = tier::list<tier::Blocked<32>>;
};

/// @ingroup api_device_lapack_chol
/// @brief ?POTRI in a kernel: the `UL` triangle of `A`, potrf()'s factor of a Hermitian positive
/// definite matrix, := the same triangle of \f$A^{-1}\f$.
///
/// The other triangle is neither read nor written, and the result's diagonal is real. potri() is
/// trtri() of the factor followed by lauum() of the inverted factor. When info > 0, `A` is
/// unchanged.
/// - `tier::Blocked<NB>`: `ex` a `WorkGroup<256, 32>`; any n; `A` in global memory. The local-memory
///   tier is not offered yet.
/// - `tier::upto<N>`: the first tier of `tiers_t<routine::potri<UL>, T>` that admits N.
///
/// Workspace: `ws` is this matrix's slice of `buffer_size<routine::potri<UL>, Tier, T>(n)`: a
/// local part and a global part (an n x NB panel), both 16-byte aligned.
///
/// @pre Collective: every item of `ex` calls potri() with the same arguments; `A` is visible to
/// every item on entry and on return.
/// @tparam UL    `Uplo::Lower` or `Uplo::Upper`: the triangle that holds the factor
/// @tparam Tier  `tier::Blocked<NB>` or `tier::upto<N>`
/// @param[in]     ex  the executor that owns this matrix
/// @param[in,out] A   the n x n factor; its `UL` triangle is overwritten with that of the inverse
/// @param[in]     ws  this matrix's workspace slice
/// @return LAPACK's info, the same on every item of `ex`:
/// @retval 0   success
/// @retval i   (i > 0) the factor's diagonal entry i is exactly zero, so A is singular; `A` is
///             unchanged
/// @retval -2  `A` is not square
/// @retval -3  `ws.local` or `ws.global` is null or not 16-byte aligned
template <Uplo UL, class Tier, class Ex, class MA> BATCHLAS_DEVICE_INLINE int potri(const Ex& ex, const MA& A, const Workspace& ws) {
    using T = typename MA::value_type;
    using Tr = resolve_tier_t<routine::potri<UL>, T, Tier>;
    static_assert(tier::is_blocked_v<Tr>, "potri: tier::Blocked<NB> (the CTA tier lands with WP7)");
    return detail::potri_blocked_op<detail::BlockedChain::Potri, UL, Diag::NonUnit, Tr::nb, typename Tr::alg_type>(ex, A, ws);
}

/// @ingroup api_device_lapack_chol
/// @brief ?TRTRI in a kernel: the `UL` triangle T of `A` := \f$T^{-1}\f$, for one matrix.
///
/// The other triangle is neither read nor written; with `DG` = `Diag::Unit` the diagonal is
/// taken as one and is neither read nor written either. With `Diag::NonUnit`, an exactly zero
/// diagonal entry is reported in info and `A` is left unchanged.
/// - `tier::Blocked<NB>`: `ex` a `WorkGroup<256, 32>`; any n; `A` in global memory.
/// - `tier::upto<N>`: the first tier of `tiers_t<routine::trtri<UL, DG>, T>` that admits N.
///
/// Workspace: `ws` is this matrix's slice of `buffer_size<routine::trtri<UL, DG>, Tier, T>(n)`: a
/// local part and a global part (an n x NB panel), both 16-byte aligned.
///
/// @pre Collective: every item of `ex` calls trtri() with the same arguments; `A` is visible to
/// every item on entry and on return.
/// @tparam UL    `Uplo::Lower` or `Uplo::Upper`: the triangle to invert
/// @tparam DG    `Diag::NonUnit`, or `Diag::Unit` for a unit triangle (diagonal not stored)
/// @tparam Tier  `tier::Blocked<NB>` or `tier::upto<N>`
/// @param[in]     ex  the executor that owns this matrix
/// @param[in,out] A   the n x n matrix; its `UL` triangle is overwritten with the inverse
/// @param[in]     ws  this matrix's workspace slice
/// @return LAPACK's info, the same on every item of `ex`:
/// @retval 0   success
/// @retval i   (i > 0, `Diag::NonUnit` only) the diagonal entry i is exactly zero; `A` is
///             unchanged
/// @retval -2  `A` is not square
/// @retval -3  `ws.local` or `ws.global` is null or not 16-byte aligned
template <Uplo UL, Diag DG, class Tier, class Ex, class MA> BATCHLAS_DEVICE_INLINE int trtri(const Ex& ex, const MA& A, const Workspace& ws) {
    using T = typename MA::value_type;
    using Tr = resolve_tier_t<routine::trtri<UL, DG>, T, Tier>;
    static_assert(tier::is_blocked_v<Tr>, "trtri: tier::Blocked<NB>");
    return detail::potri_blocked_op<detail::BlockedChain::Trtri, UL, DG, Tr::nb, typename Tr::alg_type>(ex, A, ws);
}

/// @ingroup api_device_lapack_chol
/// @brief ?LAUUM in a kernel: the `UL` triangle of `A` := \f$L^H L\f$ (`Uplo::Lower`, L that
/// triangle) or \f$U U^H\f$ (`Uplo::Upper`), for one matrix.
///
/// The other triangle is neither read nor written, and the result's diagonal is real.
/// - `tier::Blocked<NB>`: `ex` a `WorkGroup<256, 32>`; any n; `A` in global memory.
/// - `tier::upto<N>`: the first tier of `tiers_t<routine::lauum<UL>, T>` that admits N.
///
/// Workspace: `ws` is this matrix's slice of `buffer_size<routine::lauum<UL>, Tier, T>(n)`: a
/// local part and a global part (an n x NB panel), both 16-byte aligned.
///
/// @pre Collective: every item of `ex` calls lauum() with the same arguments; `A` is visible to
/// every item on entry and on return.
/// @tparam UL    `Uplo::Lower` or `Uplo::Upper`: the triangle to read and overwrite
/// @tparam Tier  `tier::Blocked<NB>` or `tier::upto<N>`
/// @param[in]     ex  the executor that owns this matrix
/// @param[in,out] A   the n x n matrix; its `UL` triangle is overwritten with the product
/// @param[in]     ws  this matrix's workspace slice
/// @return 0 or a negative argument number, the same on every item of `ex`:
/// @retval 0   success
/// @retval -2  `A` is not square
/// @retval -3  `ws.local` or `ws.global` is null or not 16-byte aligned
template <Uplo UL, class Tier, class Ex, class MA> BATCHLAS_DEVICE_INLINE int lauum(const Ex& ex, const MA& A, const Workspace& ws) {
    using T = typename MA::value_type;
    using Tr = resolve_tier_t<routine::lauum<UL>, T, Tier>;
    static_assert(tier::is_blocked_v<Tr>, "lauum: tier::Blocked<NB>");
    return detail::potri_blocked_op<detail::BlockedChain::Lauum, UL, Diag::NonUnit, Tr::nb, typename Tr::alg_type>(ex, A, ws);
}

}  // namespace batchlas::device
