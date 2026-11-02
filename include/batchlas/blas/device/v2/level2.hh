// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 level 2. Argument order is BLAS's: (ex, alpha, A, x, beta, y) and
// (ex, alpha, x, y, A). The algorithm is a template parameter whose default is a type
// computed from (executor, transpose); no kernel ever compiles the path it does not take.
//
//   gemv NoTrans      alg::RowOwner  item per output row; y in Cyclic layout
//   gemv Trans/Conj   alg::SgPerCol  team per output column, Scope::Sub; y written by lane 0
//                                    of team c mod nteams (publish y before Cyclic reads)
//   gemv on Thread    alg::Serial
//   symv/hemv Lower   alg::RowOwnerSplit (the latrd split loop); y in Cyclic layout
//                     hemv alg::RowOwnerSplitStoredDiag: A(r,r) read as stored (Im assumed 0)
//   ger/geru/gerc     alg::Cyclic    column loop outside, owned rows inside

#include <type_traits>

#include <batchlas/blas/device/v2/level1.hh>

namespace batchlas::device::v2 {

namespace detail {
template <class Ex, Transpose TR> constexpr auto default_gemv_alg() {
    if constexpr (is_thread_v<Ex>) return alg::Serial{};
    else if constexpr (TR == Transpose::NoTrans) return alg::RowOwner{};
    else return alg::SgPerCol{};
}
template <class Alg, class Default> using pick_t = std::conditional_t<std::is_void_v<Alg>, Default, Alg>;
}  // namespace detail

// y := alpha op(A) x + beta y; beta == Zero never reads y.
template <Transpose TR, class Alg = void, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DLA_INLINE void gemv(const Ex& ex, AS alpha_, const MA& a, const X& xp, BS beta_, const Y& y) {
    using Al = detail::pick_t<Alg, decltype(detail::default_gemv_alg<Ex, TR>())>;
    using T = typename MA::value_type;
    using Ar = Arith<T>;
    constexpr bool conj = TR == Transpose::ConjTrans;
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    const int m = a.rows(), n = a.cols();
    if constexpr (std::is_same_v<Al, alg::RowOwner>) {
        static_assert(TR == Transpose::NoTrans, "v2::gemv: alg::RowOwner is the NoTrans algorithm");
        const int step = detail::size_of(ex);
        for (int r = ex.lid(); r < m; r += step) {
            T acc = Ar::zero();
#pragma unroll 4
            for (int c = 0; c < n; ++c) Ar::mac(acc, a(r, c), x(c));
            store(alpha, acc, beta, y(r));
        }
    } else if constexpr (std::is_same_v<Al, alg::SgPerCol>) {
        static_assert(TR != Transpose::NoTrans, "v2::gemv: alg::SgPerCol is the (Conj)Trans algorithm");
        const int lane = ex.lane();
        const int nteams = detail::nteams_of(ex), tsize = detail::team_size_of(ex);
        for (int c = ex.team(); c < n; c += nteams) {
            T acc = Ar::zero();
            for (int r = lane; r < m; r += tsize) {
                if constexpr (conj) Ar::mac_conj(acc, a(r, c), x(r));
                else Ar::mac(acc, a(r, c), x(r));
            }
            acc = ex.team_reduce(acc);
            if (lane == 0) store(alpha, acc, beta, y(c));
        }
    } else if constexpr (std::is_same_v<Al, alg::Serial>) {
        static_assert(is_thread_v<Ex>, "v2::gemv: alg::Serial needs a one-item executor (Thread)");
        if constexpr (TR == Transpose::NoTrans) {
            for (int r = 0; r < m; ++r) {
                T acc = Ar::zero();
                for (int c = 0; c < n; ++c) Ar::mac(acc, a(r, c), x(c));
                store(alpha, acc, beta, y(r));
            }
        } else {
            for (int c = 0; c < n; ++c) {
                T acc = Ar::zero();
                for (int r = 0; r < m; ++r) {
                    if constexpr (conj) Ar::mac_conj(acc, a(r, c), x(r));
                    else Ar::mac(acc, a(r, c), x(r));
                }
                store(alpha, acc, beta, y(c));
            }
        }
    } else if constexpr (std::is_same_v<Al, alg::GroupReducePerOutput>) {
        static_assert(TR != Transpose::NoTrans, "v2::gemv: alg::GroupReducePerOutput is a (Conj)Trans algorithm");
        for (int c = 0; c < n; ++c) {
            T acc = Ar::zero();
            for (int r = ex.lid(); r < m; r += detail::size_of(ex)) {
                if constexpr (conj) Ar::mac_conj(acc, a(r, c), x(r));
                else Ar::mac(acc, a(r, c), x(r));
            }
            acc = ex.group_reduce(acc);
            if (ex.leader()) store(alpha, acc, beta, y(c));
        }
    } else {
        static_assert(sizeof(Al) == 0, "v2::gemv: unknown algorithm tag");
    }
}

namespace detail {
template <bool Herm, Uplo UL, class Alg, class Ops, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DLA_INLINE void symv_impl(const Ex& ex, AS alpha_, const MA& a, const X& xp, BS beta_, const Y& y) {
    using Al = pick_t<Alg, alg::RowOwnerSplit>;
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    static_assert(UL == Uplo::Lower, "v2::symv/hemv: only Uplo::Lower is implemented");
    static_assert(std::is_same_v<Al, alg::RowOwnerSplit> || (Herm && std::is_same_v<Al, alg::RowOwnerSplitStoredDiag>),
                  "v2::symv/hemv: alg::RowOwnerSplit (or, hemv only, alg::RowOwnerSplitStoredDiag)");
    constexpr bool masked_diag = Herm && Ar::is_complex && std::is_same_v<Al, alg::RowOwnerSplit>;
    require_published<Ex, X>();
    const auto& x = unpub(xp);
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    if constexpr (is_trailing_v<MA>) {
        // Parent coordinates: rows/columns [k, n), the hand kernels' loop shape.
        static_assert(is_trailing_v<std::remove_cvref_t<decltype(x)>> && is_trailing_v<Y>,
                      "v2::symv/hemv: a Trailing A takes Trailing x and y");
        const int k = a.first(), n = a.end();
        for (int r = k + ex.lid(); r < n; r += size_of(ex)) {
            const int c_split = sycl::min(r, n - 1);
            T acc = Ar::zero();
            if constexpr (masked_diag) {
#pragma unroll 4
                for (int c = k; c < c_split; ++c) Ar::mac(acc, a(r, c), x(c));
                Ar::mac(acc, Ar::make(Ar::re(a(r, r)), real_t<T>(0)), x(r));
            } else {
#pragma unroll 4
                for (int c = k; c <= c_split; ++c) Ar::mac(acc, a(r, c), x(c));
            }
#pragma unroll 4
            for (int c = c_split + 1; c < n; ++c) {
                if constexpr (Herm) Ar::mac_conj(acc, a(c, r), x(c));
                else Ar::mac(acc, a(c, r), x(c));
            }
            store(alpha, acc, beta, y(r));
        }
    } else {
    const int n = a.rows();
    // Row r of the lower triangle (c <= r) then column r below the diagonal (c > r): ascending
    // c crosses c == r once, so the split keeps the accumulation order and removes the branch.
    for (int r = ex.lid(); r < n; r += size_of(ex)) {
        T acc = Ar::zero();
        if constexpr (masked_diag) {
#pragma unroll 4
            for (int c = 0; c < r; ++c) Ar::mac(acc, a(r, c), x(c));
            Ar::mac(acc, Ar::make(Ar::re(a(r, r)), real_t<T>(0)), x(r));  // Im(A(r,r)) is not read
#pragma unroll 4
            for (int c = r + 1; c < n; ++c) Ar::mac_conj(acc, a(c, r), x(c));
        } else if constexpr (Herm) {  // StoredDiag: the diagonal joins the row loop as stored
#pragma unroll 4
            for (int c = 0; c <= r; ++c) Ar::mac(acc, a(r, c), x(c));
#pragma unroll 4
            for (int c = r + 1; c < n; ++c) Ar::mac_conj(acc, a(c, r), x(c));
        } else {
#pragma unroll 4
            for (int c = 0; c <= r; ++c) Ar::mac(acc, a(r, c), x(c));
#pragma unroll 4
            for (int c = r + 1; c < n; ++c) Ar::mac(acc, a(c, r), x(c));
        }
        store(alpha, acc, beta, y(r));
    }
    }
}

template <bool Conj, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DLA_INLINE void ger_impl(const Ex& ex, AS alpha_, const X& x, const Y& yp, const MA& a) {
    using T = typename MA::value_type;
    using Ar = Arith<T>;
    require_published<Ex, Y>();
    const auto& y = unpub(yp);
    const auto alpha = as_tag(alpha_);
    const int m = a.rows(), n = a.cols();
    for (int c = 0; c < n; ++c) {
        const T yc = tag_mul(alpha, Conj ? Ar::conj(static_cast<T>(y(c))) : static_cast<T>(y(c)));
        for (int r = ex.lid(); r < m; r += size_of(ex)) Ar::mac(a(r, c), x(r), yc);
    }
}
}  // namespace detail

// y := alpha A x + beta y, A symmetric with the UL triangle stored.
template <Uplo UL, class Alg = void, class Ops = arith::Explicit, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DLA_INLINE void symv(const Ex& ex, AS alpha, const MA& a, const X& x, BS beta, const Y& y) {
    detail::symv_impl<false, UL, Alg, Ops>(ex, alpha, a, x, beta, y);
}
// y := alpha A x + beta y, A Hermitian with the UL triangle stored (Im of the diagonal ignored).
template <Uplo UL, class Alg = void, class Ops = arith::Explicit, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DLA_INLINE void hemv(const Ex& ex, AS alpha, const MA& a, const X& x, BS beta, const Y& y) {
    detail::symv_impl<true, UL, Alg, Ops>(ex, alpha, a, x, beta, y);
}

// A := A + alpha x y^T (ger, geru) or A + alpha x y^H (gerc). y is read by every item.
template <class Alg = alg::Cyclic, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DLA_INLINE void geru(const Ex& ex, AS alpha, const X& x, const Y& y, const MA& a) {
    static_assert(std::is_same_v<Alg, alg::Cyclic>, "v2::ger: alg::Cyclic is the only algorithm");
    detail::ger_impl<false>(ex, alpha, x, y, a);
}
template <class Alg = alg::Cyclic, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DLA_INLINE void gerc(const Ex& ex, AS alpha, const X& x, const Y& y, const MA& a) {
    static_assert(std::is_same_v<Alg, alg::Cyclic>, "v2::ger: alg::Cyclic is the only algorithm");
    detail::ger_impl<true>(ex, alpha, x, y, a);
}
template <class Alg = alg::Cyclic, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DLA_INLINE void ger(const Ex& ex, AS alpha, const X& x, const Y& y, const MA& a) {
    geru<Alg>(ex, alpha, x, y, a);
}

// ---- v2.1: subtract-from and multi-term gemv, triangular solve / multiply -------------------

// y := b - A x (NoTrans, RowOwner): the running value starts at b(r) and every term is
// subtracted from it (Ops::msub), the order of the hand kernels' column updates. That rounding
// is not alpha * (A x) + b, so a strict port of such a loop needs this form (latrd P3, sytrf).
// y may alias b.
template <class Ops = arith::Explicit, class Ex, class MA, class X, class B, class Y>
BATCHLAS_DLA_INLINE void gemv_sub(const Ex& ex, const MA& a, const X& xp, const B& b, const Y& y) {
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    const int m = a.rows(), n = a.cols();
    for_each_owned(ex, m, [&](int r) {
        T acc = b(r);
        for (int c = 0; c < n; ++c) Ar::msub(acc, static_cast<T>(a(r, c)), static_cast<T>(x(c)));
        y(r) = acc;
    });
}

// A term A_t x_t of gemv_acc.
template <class MA, class X> struct GemvTerm {
    MA a;
    X x;
};
template <class MA, class X> BATCHLAS_DLA_INLINE GemvTerm<MA, X> term(const MA& a, const X& x) { return {a, x}; }

// y := y0 + alpha * sum_t A_t x_t (NoTrans, RowOwner) in ONE accumulator per row: the terms'
// products are summed in argument order, then scaled once (gebrd's gemv_pair, sytrd's Fit
// gemv). Every x_t is read by every item (Published, or assume_published). y may alias y0.
template <class Ops = arith::Explicit, class Ex, class AS, class Y0, class Y, class... Terms>
BATCHLAS_DLA_INLINE void gemv_acc(const Ex& ex, AS alpha_, const Y0& y0, const Y& y, const Terms&... terms) {
    using T = elem_t<Y>;
    using Ar = arith_t<Ops, T>;
    (detail::require_published<Ex, decltype(terms.x)>(), ...);
    const auto alpha = as_tag(alpha_);
    for_each_owned(ex, y.size(), [&](int r) {
        T acc = Ar::zero();
        auto one = [&](const auto& t) {
            const auto& x = detail::unpub(t.x);
            for (int c = 0; c < t.a.cols(); ++c) Ar::mac(acc, static_cast<T>(t.a(r, c)), static_cast<T>(x(c)));
        };
        (one(terms), ...);
        y(r) = Arith<T>::add(static_cast<T>(y0(r)), tag_mul(alpha, acc));
    });
}

// x := op(A)^{-1} x (trsv) or op(A) x (trmv) in place, A n x n triangular (UL stored), on one
// item: x is a register Frag<T, N> (n <= N live entries, every index a compile-time constant:
// sygst trsv_frag, potrf trsm_rowfrag) or a memory vector (runtime n). Entries past n are not
// touched. Diag::Unit never reads the diagonal; the opposite triangle is never read.
namespace detail {
template <class X> constexpr int tri_cap_of(const X*) { return 0; }
template <class T, int N, class L> constexpr int tri_cap_of(const Frag<T, N, L>*) { return N; }

// Element (i, j) of op(A) for a stored UL triangle; called only inside op(A)'s triangle.
template <Transpose TR, class T, class MA> BATCHLAS_DLA_INLINE T op_at(const MA& a, int i, int j) {
    if constexpr (TR == Transpose::NoTrans) return a(i, j);
    else if constexpr (TR == Transpose::Trans) return a(j, i);
    else return Arith<T>::conj(static_cast<T>(a(j, i)));
}
}  // namespace detail

template <Uplo UL, Transpose TR, Diag DG, class Ops = arith::Explicit, class MA, class X>
BATCHLAS_DLA_INLINE void trsv(const Thread&, const MA& a, X& x, int n) {
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    constexpr int N = detail::tri_cap_of(static_cast<const X*>(nullptr));
    // op(A) lower triangular <=> forward substitution.
    constexpr bool lower = (UL == Uplo::Lower) == (TR == Transpose::NoTrans);
    if constexpr (N > 0) {
        if constexpr (lower) {
#pragma unroll
            for (int i = 0; i < N; ++i) {
                if (i >= n) break;
                T xi = x[i];
#pragma unroll
                for (int j = 0; j < i; ++j) Ar::msub(xi, detail::op_at<TR, T>(a, i, j), x[j]);
                if constexpr (DG == Diag::NonUnit) xi = Ar::div(xi, detail::op_at<TR, T>(a, i, i));
                x[i] = xi;
            }
        } else {
#pragma unroll
            for (int i = N - 1; i >= 0; --i) {
                if (i >= n) continue;
                T xi = x[i];
#pragma unroll
                for (int j = i + 1; j < N; ++j)
                    if (j < n) Ar::msub(xi, detail::op_at<TR, T>(a, i, j), x[j]);
                if constexpr (DG == Diag::NonUnit) xi = Ar::div(xi, detail::op_at<TR, T>(a, i, i));
                x[i] = xi;
            }
        }
    } else {
        if constexpr (lower) {
            for (int i = 0; i < n; ++i) {
                T xi = x(i);
                for (int j = 0; j < i; ++j) Ar::msub(xi, detail::op_at<TR, T>(a, i, j), static_cast<T>(x(j)));
                if constexpr (DG == Diag::NonUnit) xi = Ar::div(xi, detail::op_at<TR, T>(a, i, i));
                x(i) = xi;
            }
        } else {
            for (int i = n - 1; i >= 0; --i) {
                T xi = x(i);
                for (int j = i + 1; j < n; ++j) Ar::msub(xi, detail::op_at<TR, T>(a, i, j), static_cast<T>(x(j)));
                if constexpr (DG == Diag::NonUnit) xi = Ar::div(xi, detail::op_at<TR, T>(a, i, i));
                x(i) = xi;
            }
        }
    }
}

template <Uplo UL, Transpose TR, Diag DG, class Ops = arith::Explicit, class MA, class X>
BATCHLAS_DLA_INLINE void trmv(const Thread&, const MA& a, X& x, int n) {
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    constexpr int N = detail::tri_cap_of(static_cast<const X*>(nullptr));
    constexpr bool lower = (UL == Uplo::Lower) == (TR == Transpose::NoTrans);
    // Lower: x(i) depends on x(0..i), so go from the bottom up; upper from the top down.
    auto row = [&](int i, auto&& xs) {
        T acc = DG == Diag::Unit ? T(xs(i)) : Ar::mul(detail::op_at<TR, T>(a, i, i), T(xs(i)));
        if constexpr (lower)
            for (int j = 0; j < i; ++j) Ar::mac(acc, detail::op_at<TR, T>(a, i, j), T(xs(j)));
        else
            for (int j = i + 1; j < n; ++j) Ar::mac(acc, detail::op_at<TR, T>(a, i, j), T(xs(j)));
        return acc;
    };
    if constexpr (N > 0) {
        auto xs = [&](int j) -> T { return x[j]; };
        if constexpr (lower) {
#pragma unroll
            for (int i = N - 1; i >= 0; --i)
                if (i < n) x[i] = row(i, xs);
        } else {
#pragma unroll
            for (int i = 0; i < N; ++i)
                if (i < n) x[i] = row(i, xs);
        }
    } else {
        auto xs = [&](int j) -> T { return x(j); };
        if constexpr (lower)
            for (int i = n - 1; i >= 0; --i) x(i) = row(i, xs);
        else
            for (int i = 0; i < n; ++i) x(i) = row(i, xs);
    }
}
// Memory-vector forms take x by const view (the view, not the data, is const).
template <Uplo UL, Transpose TR, Diag DG, class Ops = arith::Explicit, class MA, class X>
    requires(requires(const X& v) { v.size(); })
BATCHLAS_DLA_INLINE void trsv(const Thread& th, const MA& a, const X& x) {
    X xv = x;
    trsv<UL, TR, DG, Ops>(th, a, xv, x.size());
}
template <Uplo UL, Transpose TR, Diag DG, class Ops = arith::Explicit, class MA, class X>
    requires(requires(const X& v) { v.size(); })
BATCHLAS_DLA_INLINE void trmv(const Thread& th, const MA& a, const X& x) {
    X xv = x;
    trmv<UL, TR, DG, Ops>(th, a, xv, x.size());
}

}  // namespace batchlas::device::v2
