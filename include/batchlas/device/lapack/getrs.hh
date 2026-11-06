// SPDX-License-Identifier: MIT
#pragma once
// ?GETRS's device steps (the fused and composed host tiers run them), in getrs_detail until core
// Level 3 / lapack_aux has SASS-identical forms of them. Everything else comes from core
// (RtGroup, for_each_owned, lacpy, laswp, conj_if, op_view, shift_reduce, select, arith::Fma):
//   trsm_lane          op(A)^{-1} Y, one sub-group, lane i owns row i, shuffles only
//   gemm_nr            C -= op(A) X with X.cols() <= NR accumulators
//   lapmr_copy         B(i, :) := A(perm(i), :), out of place, 0-based perm, Alg walk
// Arithmetic is arith::Fma: sycl_device::dev_mul / fma_acc / dev_div bit for bit.
// The public op getrs<TR, Tier>(ex, A, B, ipiv, ws): tier::Blocked (any n, nrhs) is LAPACK's
// ?GETRS on the in-group trsm, with the LU family's executor and local plan (lapack/getrf.hh).

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/util/group-collectives.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/lapack/getrf.hh>
#include <batchlas/device/level3/group.hh>

namespace batchlas::device::inline v2::getrs_detail {

// Y(:, c) := op(A)^{-1} Y(:, c) for c = c0, c0 + cs, ... < Y.cols(), lane i owning row i (rows
// >= A.rows() idle). Lower runs forward, Upper backward; Unit skips the last pivot, which
// updates nothing. Collective: every lane calls it; the lane guards are inside.
template <Uplo UL, Diag DG, int SG, class MA, class MY>
BATCHLAS_DLA_INLINE void trsm_lane(const SubGroup<SG>& ex, const MA& a, const MY& y, int c0, int cs) {
    using T = typename MY::value_type;
    using Ar = arith_t<arith::Fma, T>;
    const int lane = ex.lid();
    const int jb = a.rows();
    for (int c = c0; c < y.cols(); c += cs) {
        T v = (lane < jb) ? y(lane, c) : Ar::zero();
        auto pivot = [&](int s) {
            if constexpr (DG == Diag::NonUnit) {
                if (lane == s) v = Ar::div(v, a(s, s));
            }
            return batchlas::portable::group_broadcast(ex.sg, v, s);
        };
        if constexpr (UL == Uplo::Lower) {
            const int end = DG == Diag::Unit ? jb - 1 : jb;
            for (int s = 0; s < end; ++s) {
                const T vs = pivot(s);
                if (lane > s && lane < jb) v = Ar::sub(v, Ar::mul(a(lane, s), vs));
            }
        } else {
            const int stop = DG == Diag::Unit ? 1 : 0;
            for (int s = jb - 1; s >= stop; --s) {
                const T vs = pivot(s);
                if (lane < s) v = Ar::sub(v, Ar::mul(a(lane, s), vs));
            }
        }
        if (lane < jb) y(lane, c) = v;
    }
}

// C := C - op(A) X; X.cols() <= NR is read at run time, accumulators past it are never stored.
// NoTrans: item per row of C, k serial (coalesced down A's columns), the NR columns expanded at
// compile time by static_for. Columns past nrhs re-read column nrhs - 1 into an accumulator that
// is never stored: no branch per column, which a complex mac kept (GetrsFusedN<complex<float>,
// 4>: 1.02-1.03x slower than the hand body), and LLVM still runtime-unrolls k for a real T. Trans:
// sub-group per row of C, lanes stride k, a leader-only shift_reduce, lane 0 writes. ConjTrans
// is Trans of conj_if(A, true).
template <Transpose TA, int NR, class Ex, class MA, class MX, class MC>
BATCHLAS_DLA_INLINE void gemm_nr(const Ex& ex, NegOne, const MA& a, const MX& x, One, const MC& c) {
    using T = typename MC::value_type;
    using Ar = arith_t<arith::Fma, T>;
    const int nrhs = x.cols();
    if constexpr (TA == Transpose::NoTrans) {
        const int k = a.cols();
        for_each_owned(ex, c.rows(), [&](int i) BATCHLAS_DLA_BODY {
            T acc[NR];
            static_for<NR>([&](auto r) BATCHLAS_DLA_BODY { acc[r] = Ar::zero(); });
            for (int kk = 0; kk < k; ++kk) {
                const T av = a(i, kk);
                static_for<NR>([&](auto r) BATCHLAS_DLA_BODY {
                    Ar::mac(acc[r], av, x(kk, r < nrhs ? static_cast<int>(r) : nrhs - 1));
                });
            }
            static_for<NR>([&](auto r) BATCHLAS_DLA_BODY {
                if (r < nrhs) c(i, r) = Ar::sub(c(i, r), acc[r]);
            });
        });
    } else {
        static_assert(TA == Transpose::Trans, "getrs: gemm_nr ConjTrans is Trans of conj_if(A, true)");
        constexpr int SG = Ex::sg_size;
        const int m = c.rows(), k = a.rows();
        const int lane = ex.lane();
        const auto add = [](T u, T v) { return Ar::add(u, v); };
        for (int t = ex.team(); t < m; t += detail::nteams_of(ex)) {
            T acc[NR];
BATCHLAS_UNROLL_FULL
            for (int r = 0; r < NR; ++r) acc[r] = Ar::zero();
            for (int i = lane; i < k; i += SG) {
                const T av = a(i, t);
BATCHLAS_UNROLL_FULL
                for (int r = 0; r < NR; ++r)
                    if (r < nrhs) Ar::mac(acc[r], av, x(i, r));
            }
BATCHLAS_UNROLL_FULL
            for (int r = 0; r < NR; ++r)
                if (r < nrhs) {
                    const T s = shift_reduce<SG>(ex.sub(), acc[r], add);
                    if (lane == 0) c(t, r) = Ar::sub(c(t, r), s);
                }
        }
    }
}

// B(i, :) := A(perm(i), :) (LAPACK lapmr, forward, out of place); perm(i) is 0-based.
template <class Alg = alg::Stepped, class Ex, class MA, class P, class MB>
BATCHLAS_DLA_INLINE void lapmr_copy(const Ex& ex, const MA& a, const P& perm, const MB& b) {
    for_each_owned_2d<Alg>(ex, b.rows(), b.cols(), [&](int i, int c) BATCHLAS_DLA_BODY { b(i, c) = a(perm(i), c); });
}

}  // namespace batchlas::device::getrs_detail

// ==== the public op: getrs<TR, Tier>(ex, A, B, ipiv, ws [, live]) (v22 design C.2-C.7) ===========
namespace batchlas::device::inline v2 {

namespace routine {
struct getrs {
    static constexpr const char* name = "getrs";
};
}  // namespace routine

namespace getrs_detail {

// getrf's pivots, read only (1-based int32).
struct PivIn {
    const std::int32_t* p;
    BATCHLAS_DLA_INLINE std::int32_t operator()(int i) const { return p[i]; }
};

// B := U^-1 L^-1 B (NoTrans), or op(L)^-1 op(U)^-1 B (Trans, ConjTrans): the two triangles of
// getrf's factor on the in-group trsm (NB-wide diagonal blocks); each solve ends on a global barrier.
template <Transpose TR, int NB, class Cfg, class Ex, class GA, class GB>
BATCHLAS_DLA_INLINE void lu_trsm_pair(const Ex& ex, const GA& a, const GB& b, l3::Smem<Cfg> sm) {
    if constexpr (TR == Transpose::NoTrans) {
        trsm<Side::Left, Uplo::Lower, TR, Diag::Unit, Cfg, NB>(ex, a, b, One{}, sm);
        trsm<Side::Left, Uplo::Upper, TR, Diag::NonUnit, Cfg, NB>(ex, a, b, One{}, sm);
    } else {
        trsm<Side::Left, Uplo::Upper, TR, Diag::NonUnit, Cfg, NB>(ex, a, b, One{}, sm);
        trsm<Side::Left, Uplo::Lower, TR, Diag::Unit, Cfg, NB>(ex, a, b, One{}, sm);
    }
}

// The Blocked tier: ?GETRS as LAPACK spells it. NoTrans: laswp forward on B, then L and U;
// Trans / ConjTrans: op(U), op(L), then laswp backward.
template <Transpose TR, int NB, class Cfg, class Ex, class MA, class MB>
BATCHLAS_DLA_INLINE void solve_blocked(const Ex& ex, const MA& a_in, const MB& b_in, const std::int32_t* ipiv, const Workspace& w) {
    using D = typename Cfg::value_type;
    const auto a = l3::global_of(a_in);
    const auto b = l3::global_of(b_in);
    const auto bw = mat_wide(b.p, b.m, b.n, b.ld);
    constexpr auto pl = getrf_detail::lu_blocked_plan<D, Cfg>();
    const l3::Smem<Cfg> sm{carve_local(w, pl.smem).get_raw()};
    const PivIn piv{ipiv};
    if constexpr (TR == Transpose::NoTrans) {
        laswp(ex, bw, piv, 0, a.n);
        ex.barrier(Mem::Global);
    }
    lu_trsm_pair<TR, NB>(ex, a, b, sm);
    if constexpr (TR != Transpose::NoTrans) {
        laswp<dir::Backward>(ex, bw, piv, 0, a.n);
        ex.barrier(Mem::Global);
    }
}

}  // namespace getrs_detail

/// getrs's Blocked<NB>: any n and nrhs; getrf's Blocked executor and local plan (no global workspace).
template <int NB, class Alg, class T>
struct tier_info<routine::getrs, tier::Blocked<NB, Alg>, T> : tier_info<routine::getrf, tier::Blocked<NB, Alg>, T> {};
template <class T> struct tiers<routine::getrs, T> {
    using type = tier::list<tier::Blocked<32>>;
};

/// The workspace of one n x nrhs solve for getrs<TR, Tier> (host and device; constexpr).
template <Transpose TR, class Tier, class T> constexpr WorkspaceSize getrs_buffer_size(int n, int nrhs = 1) {
    (void)nrhs;
    return tier_info<routine::getrs, resolve_tier_t<routine::getrs, T, Tier>, T>::buffer_size(n);
}

/// @brief ?GETRS in a kernel: \f$\mathrm{op}(A) X = B\f$ from getrf's factor, B overwritten by X.
///
/// Collective as getrf: every item calls with the same arguments; A, B and ipiv visible to all
/// items on entry; B visible to all on return. A zero U(k, k) gives Inf / NaN, as LAPACK does
/// (getrf's info reports it).
/// - `tier::Blocked<NB>`: `ex` a WorkGroup<tier_info::wg_size, 32>; A n x n and B n x nrhs of
///   any size in global memory; `ws.local` >= getrs_buffer_size<TR, Tier, T>(n) bytes.
/// @tparam TR  op(A): NoTrans, Trans or ConjTrans (the host's run-time transA)
/// @param[in]     ex    the executor
/// @param[in]     a     getrf's L and U of this matrix
/// @param[in,out] b     this matrix's right-hand sides; overwritten with X
/// @param[in]     ipiv  getrf's n pivots, 1-based int32
/// @param[in]     ws    this matrix's workspace slice
/// @param[in]     live  false: a dead matrix of the launch (executor-uniform)
/// @return 0 (the host getrs has no info)
template <Transpose TR, class Tier, class Ex, class MA, class MB>
BATCHLAS_DLA_INLINE std::int32_t getrs(const Ex& ex, const MA& a, const MB& b, const std::int32_t* ipiv, const Workspace& ws = {},
                                       bool live = true) {
    using D = getrf_detail::lu_value_t<MA>;
    using Tr = resolve_tier_t<routine::getrs, D, Tier>;
    static_assert(HasTier<routine::getrs, Tr, D>, "getrs: this tier is not offered for this scalar type");
    static_assert(std::is_same_v<D, getrf_detail::lu_value_t<MB>>, "getrs: A and B hold one scalar type");
    using Cfg = getrf_detail::lu_blocked_cfg_t<D, Tr>;
    static_assert(Ex::size == Cfg::threads, "getrs: tier::Blocked runs on a WorkGroup of tier_info::wg_size items");
    if (live) getrs_detail::solve_blocked<TR, Tr::nb, Cfg>(ex, a, b, ipiv, ws);
    return 0;
}

}  // namespace batchlas::device
