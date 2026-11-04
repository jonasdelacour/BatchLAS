// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xPOTF2, Lower: the unblocked Cholesky recurrence A = L L^H as ONE column loop whose data
// placement is a distribution policy (v22 design A.3). Per column k the loop asks the
// distribution, in this order:
//   skip(k)               (only when Dist::skips) column k is past the run-time order: no work,
//                         no collective (legal only when the order is uniform over the executor)
//   pivot(k) -> akk       Re A(k, k) as every item sees it (collective for register forms)
//   check(k, akk) -> act  LAPACK's test `!(akk > 0)` (NaN fails), sticky: the first failure records
//                         info and every later column stops changing; act = column k factors
//   scale(k, akk, act)    L(k, k) := sqrt(akk), L(k+1:, k) *= 1 / L(k, k)
//   update(k, act)        A(k+1:, k+1:) -= L(k+1:, k) L(k+1:, k)^H on the lower triangle
// and returns d.result(). Dist::static_cols > 0 is a compile-time column count (fully unrolled,
// register-resident forms), 0 a run-time d.cols() (not unrolled). Distributions here:
//   potf2_sg_slm      SubGroup, lane r holds row r in a Frag, column k published to local memory
//   potf2_panel_rows  any executor, one item per row of a local n x kb panel, caller's barrier
// The Partition RowFrag form (lane r holds row r, columns by shuffle or SLM broadcast) is the
// tiny tiers' distribution (src/kernels/common/dla_v2_potf2.hh) until core has its scalar forms.
// Rsqrt: 1/sqrt(a) given sqrt(a), `static R inv_sqrt(R a, R sqrt_a)` (core inv_sqrt pending).

#include <batchlas/blas/enums.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/operand.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

#include <sycl/sycl.hpp>

#include <cstdint>
#include <type_traits>

namespace batchlas::device::inline v2 {

// LAPACK's pivot test: a pivot that is not > 0 fails, NaN included.
template <class R> BATCHLAS_DLA_INLINE bool potf2_bad_pivot(R akk) { return !(akk > R(0)); }

namespace potrf_detail {
template <class Dist> BATCHLAS_DLA_INLINE void potf2_step(Dist& d, int k) {
    if constexpr (Dist::skips) {
        if (d.skip(k)) return;
    }
    const auto akk = d.pivot(k);
    const bool act = d.check(k, akk);
    d.scale(k, akk, act);
    d.update(k, act);
}
}  // namespace potrf_detail

template <Uplo UL, class Dist> BATCHLAS_DLA_INLINE auto potf2(Dist& d) {
    static_assert(UL == Uplo::Lower, "potf2: only Uplo::Lower (an Upper caller factors the conjugate transpose)");
    if constexpr (Dist::static_cols > 0) {
BATCHLAS_UNROLL_FULL
        for (int k = 0; k < Dist::static_cols; ++k) potrf_detail::potf2_step(d, k);
    } else {
        for (int k = 0; k < d.cols(); ++k) potrf_detail::potf2_step(d, k);
    }
    return d.result();
}

namespace potrf_detail {
// The ib x ib block `a` (ib <= NB <= SG) on one sub-group: lane r holds row r in `d`; column k is
// written to `a` (local) and read back after one sub-group barrier; diag(k) := L(k, k). The first
// bad pivot sets *info := info_base + k + 1. The NB barriers are unconditional.
template <int NB, class Rsqrt, int SG, class MA, class DV> struct SgSlm {
    using D = typename MA::value_type;
    using A = arith_t<arith::Fma, D>;
    using R = real_t<D>;
    static constexpr int static_cols = NB;
    static constexpr bool skips = false;
    const SubGroup<SG>& ex;
    const MA& a;
    int ib;
    const DV& diag;
    int* info;
    int info_base;
    int lane;
    Frag<D, NB>& d;
    bool alive;

    BATCHLAS_DLA_INLINE R pivot(int k) const {
        return sycl::select_from_group(ex.sg, A::re(d[k]), static_cast<std::uint32_t>(k));
    }
    BATCHLAS_DLA_INLINE bool check(int k, R akk) {
        bool active = alive && (k < ib);
        if (active && potf2_bad_pivot(akk)) {
            if (lane == 0) *info = info_base + k + 1;
            active = false;
            alive = false;
        }
        return active;
    }
    BATCHLAS_DLA_INLINE void scale(int k, R akk, bool active) {
        const R dkk = active ? sycl::sqrt(akk) : R(1);
        const R r = Rsqrt::inv_sqrt(active ? akk : R(1), dkk);
        if (active) {
            if (lane == k) d[k] = A::make(dkk, R(0));
            else if (lane > k && lane < ib) d[k] = A::scale_real(r, d[k]);
            if (lane < ib && lane >= k) a(lane, k) = d[k];
            if (lane == 0) diag(k) = dkk;
        }
        sycl::group_barrier(ex.sg);
    }
    BATCHLAS_DLA_INLINE void update(int k, bool active) {
        if (active) {
            BATCHLAS_UNROLL_FULL
            for (int c = k + 1; c < NB; ++c) {
                if (c < ib && lane < ib && lane >= c) d[c] = A::sub(d[c], A::mul(d[k], A::conj(a(c, k))));
            }
        }
    }
    BATCHLAS_DLA_INLINE void result() const {}
};

template <int NB, class Rsqrt, int SG, class MA, class DV>
BATCHLAS_DLA_HELPER void potf2_sg_slm_body(SubGroup<SG> ex, MA a, int ib, DV diag, int* info, int info_base) {
    using D = typename MA::value_type;
    static_assert(NB <= SG, "potf2_sg_slm: one lane per row");
    const int lane = ex.lid();
    Frag<D, NB> d;
    BATCHLAS_UNROLL_FULL
    for (int c = 0; c < NB; ++c) d[c] = (lane < ib && c < ib && c <= lane) ? a(lane, c) : D{};
    SgSlm<NB, Rsqrt, SG, MA, DV> dist{ex, a, ib, diag, info, info_base, lane, d, true};
    potf2<Uplo::Lower>(dist);
}

// The n x kb panel P whose diagonal block starts at row c0, one item per row (row = ex.lid();
// rows outside (c0, n) idle). `bar` is the caller's barrier over every item reading P, three per
// column: pivot read / its overwrite, scale, rank-1 update. `ok` is sticky; the first bad pivot
// sets *info := c0 + i + 1 (leader).
template <class Rsqrt, class Ex, class PM, class Bar> struct PanelRows {
    using D = typename PM::value_type;
    using A = arith_t<arith::Fma, D>;
    using R = real_t<D>;
    static constexpr int static_cols = 0;
    static constexpr bool skips = false;
    const Ex& ex;
    const PM& p;
    int c0;
    int kb;
    int n;
    bool ok;
    int* info;
    Bar& bar;
    int row;

    BATCHLAS_DLA_INLINE int cols() const { return kb; }
    BATCHLAS_DLA_INLINE R pivot(int i) const { return A::re(p(c0 + i, i)); }
    BATCHLAS_DLA_INLINE bool check(int i, R d) {
        const bool bad = ok && potf2_bad_pivot(d);
        if (bad) {
            ok = false;
            if (ex.leader()) *info = c0 + i + 1;
        }
        bar();  // WAR: every item read the pivot cell above, item c0 + i overwrites it below
        return ok;
    }
    BATCHLAS_DLA_INLINE void scale(int i, R d, bool act) {
        if (act) {
            const R dkk = sycl::sqrt(d);
            const R rs = Rsqrt::inv_sqrt(d, dkk);  // act: d > 0, not NaN
            if (row == c0 + i) p(row, i) = A::make(dkk, R(0));
            else if (row > c0 + i && row < n) p(row, i) = A::scale_real(rs, p(row, i));
        }
        bar();
    }
    BATCHLAS_DLA_INLINE void update(int i, bool act) {
        if (act && row > c0 + i && row < n) {
            const D lri = p(row, i);
            for (int c = i + 1; c < kb; ++c) {
                const D ljc = p(c0 + c, i);
                p(row, c) = A::sub(p(row, c), A::mul(lri, A::conj(ljc)));
            }
        }
        bar();
    }
    BATCHLAS_DLA_INLINE bool result() const { return ok; }
};
}  // namespace potrf_detail

// potf2 of the ib x ib local block `a` on one sub-group (SgSlm above). Real T: the body is a
// by-value phase op the inliner takes after simplifying it alone (its column-0..3 reciprocal
// shapes depend on it). Complex T: inlined at the call; out of line the by-value operands spill.
template <int NB, class Rsqrt, int SG, class MA, class DV>
BATCHLAS_DLA_INLINE void potf2_sg_slm(const SubGroup<SG>& ex, const MA& a, int ib, const DV& diag, int* info,
                                      int info_base) {
    if constexpr (arith_t<arith::Fma, typename MA::value_type>::is_complex) {
        [[clang::always_inline]] potrf_detail::potf2_sg_slm_body<NB, Rsqrt>(ex, a, ib, diag, info, info_base);
    } else {
        potrf_detail::potf2_sg_slm_body<NB, Rsqrt>(ex, a, ib, diag, info, info_base);
    }
}

// potf2 of the panel P, one item per row (PanelRows above). Returns the sticky `ok`.
template <class Rsqrt, class Ex, class PM, class Bar>
BATCHLAS_DLA_INLINE bool potf2_panel_rows(const Ex& ex, const PM& p, int c0, int kb, int n, bool ok, int* info,
                                          Bar&& bar) {
    using B = std::remove_reference_t<Bar>;
    potrf_detail::PanelRows<Rsqrt, Ex, PM, B> dist{ex, p, c0, kb, n, ok, info, bar, ex.lid()};
    return potf2<Uplo::Lower>(dist);
}

}  // namespace batchlas::device
