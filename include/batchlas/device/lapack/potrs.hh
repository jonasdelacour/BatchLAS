// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xPOTRS inside a kernel (v22 design C.2, C.4-C.7): A X = B with A = L L^H (Lower) or
// U^H U (Upper) from potrf, X overwriting B. Public entry:
//   int potrs<UL, Tier>(ex, A, B, ws)          0, or -i for argument i (uniform over ex)
//   WorkspaceSize potrs_buffer_size<UL, Tier, T>(n)
//   tier::Blocked<NB>  ex = WorkGroup<256, 32>, any n and nrhs: the two triangular solves are the
//                      in-group L3 trsm (level3/group.hh) over NB-wide diagonal blocks;
//                      ws.local = the L3 stages (the same slot potrf's Blocked plan starts with)
// The small tiers (the host posv tiny and potrs fused bodies) land with WP7. Collective: every
// item of ex calls with the same arguments; inputs visible on entry; X visible on return.
// -2: A not square; -3: B's rows differ from n; -4: ws.local missing or misaligned.

#include <cstddef>
#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/lapack/potrf.hh>
#include <batchlas/device/level3/group.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

namespace potrs_detail {

// The Blocked plan: the L3 stage slot alone, at potrf's offset.
template <class T> struct BlockedSolvePlan {
    ws::Slot<T> smem;
    WorkspaceSize size;
    static constexpr BlockedSolvePlan make() {
        Plan p;
        const auto s = p.take<T>(ws::Space::Local, static_cast<std::size_t>(potrf_detail::BlockedCfg<T>::smem_elems));
        return {s, p.size()};
    }
};
static_assert(BlockedSolvePlan<float>::make().smem.offset == potrf_detail::BlockedPlan<float, 32>::make().smem.offset);

// B := A^-1 B from the factor: Lower L (L^H X = Y after L Y = B), Upper U^H then U. Arguments
// already checked; n, nrhs >= 1.
template <Uplo UL, int NB, int WG, int SG, class MA, class MB>
BATCHLAS_DEVICE_INLINE void potrs_blocked_solve(const WorkGroup<WG, SG>& ex, const MA& a, const MB& b, const Workspace& ws) {
    using T = typename MA::value_type;
    using Cfg = potrf_detail::BlockedCfg<T>;
    constexpr BlockedSolvePlan<T> pl = BlockedSolvePlan<T>::make();
    const l3::Smem<Cfg> sm{carve_local(ws, pl.smem).get_raw()};
    const int n = a.rows(), nrhs = b.cols();
    const auto A = mat(&a(0, 0), n, n, static_cast<int>(a.ld));
    const auto B = mat(&b(0, 0), n, nrhs, static_cast<int>(b.ld));
    if constexpr (UL == Uplo::Lower) {
        trsm<Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, Cfg, NB>(ex, A, B, One{}, sm);
        trsm<Side::Left, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit, Cfg, NB>(ex, A, B, One{}, sm);
    } else {
        trsm<Side::Left, Uplo::Upper, Transpose::ConjTrans, Diag::NonUnit, Cfg, NB>(ex, A, B, One{}, sm);
        trsm<Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, Cfg, NB>(ex, A, B, One{}, sm);
    }
}

// The argument checks potrs and posv share (posv's ws is also argument 4).
template <class MA, class MB> BATCHLAS_DEVICE_INLINE int potrs_blocked_args(const MA& a, const MB& b, const Workspace& ws) {
    static_assert(std::is_same_v<typename MA::value_type, typename MB::value_type>, "potrs: A and B hold the same scalar type");
    const int n = a.rows();
    if (n < 0 || a.cols() != n) return -2;
    if (b.rows() != n || b.cols() < 0) return -3;
    if (n > 0 && b.cols() > 0 && !potrf_detail::blocked_ws_ok(ws, false)) return -4;
    return 0;
}

template <Uplo UL, int NB, class Alg, int WG, int SG, class MA, class MB>
BATCHLAS_DEVICE_INLINE int potrs_blocked_op(const WorkGroup<WG, SG>& ex, const MA& a, const MB& b, const Workspace& ws) {
    using T = typename MA::value_type;
    static_assert(std::is_same_v<Alg, alg::Default>, "potrs: tier::Blocked has one algorithm");
    static_assert(NB >= 1, "potrs: tier::Blocked<NB> needs NB >= 1");
    static_assert(WG == potrf_detail::BlockedCfg<T>::threads && SG == 32, "potrs: tier::Blocked runs on WorkGroup<256, 32>");
    const int bad = potrs_blocked_args(a, b, ws);
    if (bad != 0) return bad;
    if (a.rows() == 0 || b.cols() == 0) return 0;
    potrs_blocked_solve<UL, NB>(ex, a, b, ws);
    return 0;
}
}  // namespace potrs_detail

namespace routine {
template <Uplo UL> struct potrs {
    static constexpr const char* name = "potrs";
    static constexpr Uplo uplo = UL;
};
}  // namespace routine

template <Uplo UL, int NB, class Alg, class T> struct tier_info<routine::potrs<UL>, tier::Blocked<NB, Alg>, T> {
    static constexpr int wg_size = potrf_detail::kBlockedWg;
    static constexpr int sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int) { return potrs_detail::BlockedSolvePlan<T>::make().size; }
};
template <Uplo UL, class T> struct tiers<routine::potrs<UL>, T> {
    using type = tier::list<tier::Blocked<32>>;
};

// The workspace of one matrix (host and device); nrhs does not enter it. Tier may be tier::upto<NMax>.
template <Uplo UL, class Tier, class T> constexpr WorkspaceSize potrs_buffer_size(int n) {
    using Tr = resolve_tier_t<routine::potrs<UL>, T, Tier>;
    return tier_info<routine::potrs<UL>, Tr, T>::buffer_size(n);
}

// B := A^-1 B for the n x nrhs B, A's UL triangle holding potrf's factor (the other triangle not
// read); returns 0 or -i (see the file head).
template <Uplo UL, class Tier, class Ex, class MA, class MB>
BATCHLAS_DEVICE_INLINE int potrs(const Ex& ex, const MA& A, const MB& B, const Workspace& ws) {
    using T = typename MA::value_type;
    using Tr = resolve_tier_t<routine::potrs<UL>, T, Tier>;
    static_assert(potrf_detail::is_blocked_v<Tr>, "potrs: tier::Blocked<NB> (the small tiers land with WP7)");
    return potrs_detail::potrs_blocked_op<UL, Tr::nb, typename Tr::alg_type>(ex, A, B, ws);
}

}  // namespace batchlas::device
