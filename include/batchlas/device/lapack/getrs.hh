// SPDX-License-Identifier: MIT
#pragma once
// ?GETRS's device steps (the fused and composed host tiers run them), in getrs_detail until core
// Level 3 / lapack_aux has SASS-identical forms of them. Everything else comes from core
// (RtGroup, for_each_owned, lacpy, laswp, conj_if, op_view, shift_reduce, select, arith::Fma):
//   trsm_lane          op(A)^{-1} Y, one sub-group, lane i owns row i, shuffles only
//   gemm_nr            C -= op(A) X with X.cols() <= NR accumulators
//   lapmr_copy         B(i, :) := A(perm(i), :), out of place, 0-based perm, Alg walk
// Arithmetic is arith::Fma: sycl_device::dev_mul / fma_acc / dev_div bit for bit.

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/util/group-collectives.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

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
