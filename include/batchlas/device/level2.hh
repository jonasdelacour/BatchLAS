// SPDX-License-Identifier: MIT
#pragma once
// Device LA level 2. Argument order is BLAS's: (ex, alpha, A, x, beta, y) and
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

#include <batchlas/device/level1.hh>

namespace batchlas::device {

namespace detail {
template <class Ex, Transpose TR> constexpr auto default_gemv_alg() {
    if constexpr (is_thread_v<Ex>) return alg::Serial{};
    else if constexpr (TR == Transpose::NoTrans) return alg::RowOwner{};
    else return alg::SgPerCol{};
}
template <class Alg, class Default> using pick_t = std::conditional_t<std::is_void_v<Alg>, Default, Alg>;
// gemv's output: store<Shape, Ops>(alpha, acc, beta, y); alpha_side::Right with a Scalar alpha
// forms acc * alpha first (then beta's term as a One-alpha store).
template <class S> inline constexpr bool is_scalar_value_v = false;
template <class T> inline constexpr bool is_scalar_value_v<Scalar<T>> = true;
template <class Shape, class Ops, class Side, class AS, class T, class BS, class Ref>
BATCHLAS_DEVICE_INLINE void gemv_put(AS alpha, T acc, BS beta, Ref&& y) {
    if constexpr (std::is_same_v<Side, alpha_side::Right> && is_scalar_value_v<AS>) {
        store<Shape, Ops>(One{}, arith_t<Ops, T>::mul(acc, static_cast<T>(alpha.v)), beta, y);
    } else {
        static_assert(std::is_same_v<Side, alpha_side::Left> || std::is_same_v<Side, alpha_side::Right>, "gemv: alpha_side::Left or Right");
        store<Shape, Ops>(alpha, acc, beta, y);
    }
}
template <class Ex> BATCHLAS_DEVICE_INLINE auto lane_group(const Ex& ex) {
    if constexpr (requires { ex.part; }) return ex.part;
    else return ex.sub();
}
}  // namespace detail

// y := alpha op(A) x + beta y; beta == Zero never reads y. Ops: the arithmetic of the dot;
// Shape: the epilogue (store<Shape, Ops>, arith.hh); AlphaSide: alpha_side::Right forms acc * alpha.
//   alg::TeamPerRow  NoTrans: team t owns rows t mod nteams, lanes stride the columns, lane 0 stores
//   alg::ColDot<W>   (Conj)Trans: W consecutive lanes per column (W | the sub-group), a W-lane
//                    butterfly; the W-block's first lane stores
template <Transpose TR, class Alg = void, class Ops = arith::Explicit, class Shape = epi::Sum, class AlphaSide = alpha_side::Left,
          class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DEVICE_INLINE void gemv(const Ex& ex, AS alpha_, const MA& a, const X& xp, BS beta_, const Y& y) {
    using Al = detail::pick_t<Alg, decltype(detail::default_gemv_alg<Ex, TR>())>;
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    constexpr bool conj = TR == Transpose::ConjTrans;
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    const int m = a.rows(), n = a.cols();
    if constexpr (std::is_same_v<Al, alg::RowOwner>) {
        static_assert(TR == Transpose::NoTrans, "gemv: alg::RowOwner is the NoTrans algorithm");
        const int step = detail::size_of(ex);
        for (int r = ex.lid(); r < m; r += step) {
            T acc = Ar::zero();
#pragma unroll 4
            for (int c = 0; c < n; ++c) Ar::mac(acc, a(r, c), x(c));
            detail::gemv_put<Shape, Ops, AlphaSide>(alpha, acc, beta, y(r));
        }
    } else if constexpr (std::is_same_v<Al, alg::SgPerCol>) {
        static_assert(TR != Transpose::NoTrans, "gemv: alg::SgPerCol is the (Conj)Trans algorithm");
        const int lane = ex.lane();
        const int nteams = detail::nteams_of(ex), tsize = detail::team_size_of(ex);
        for (int c = ex.team(); c < n; c += nteams) {
            T acc = Ar::zero();
            for (int r = lane; r < m; r += tsize) {
                if constexpr (conj) Ar::mac_conj(acc, a(r, c), x(r));
                else Ar::mac(acc, a(r, c), x(r));
            }
            acc = ex.team_reduce(acc);
            if (lane == 0) detail::gemv_put<Shape, Ops, AlphaSide>(alpha, acc, beta, y(c));
        }
    } else if constexpr (std::is_same_v<Al, alg::Serial>) {
        static_assert(is_thread_v<Ex>, "gemv: alg::Serial needs a one-item executor (Thread)");
        if constexpr (TR == Transpose::NoTrans) {
            for (int r = 0; r < m; ++r) {
                T acc = Ar::zero();
                for (int c = 0; c < n; ++c) Ar::mac(acc, a(r, c), x(c));
                detail::gemv_put<Shape, Ops, AlphaSide>(alpha, acc, beta, y(r));
            }
        } else {
            for (int c = 0; c < n; ++c) {
                T acc = Ar::zero();
                for (int r = 0; r < m; ++r) {
                    if constexpr (conj) Ar::mac_conj(acc, a(r, c), x(r));
                    else Ar::mac(acc, a(r, c), x(r));
                }
                detail::gemv_put<Shape, Ops, AlphaSide>(alpha, acc, beta, y(c));
            }
        }
    } else if constexpr (std::is_same_v<Al, alg::GroupReducePerOutput>) {
        static_assert(TR != Transpose::NoTrans, "gemv: alg::GroupReducePerOutput is a (Conj)Trans algorithm");
        for (int c = 0; c < n; ++c) {
            T acc = Ar::zero();
            for (int r = ex.lid(); r < m; r += detail::size_of(ex)) {
                if constexpr (conj) Ar::mac_conj(acc, a(r, c), x(r));
                else Ar::mac(acc, a(r, c), x(r));
            }
            acc = ex.group_reduce(acc);
            if (ex.leader()) detail::gemv_put<Shape, Ops, AlphaSide>(alpha, acc, beta, y(c));
        }
    } else if constexpr (std::is_same_v<Al, alg::TeamPerRow>) {
        static_assert(TR == Transpose::NoTrans, "gemv: alg::TeamPerRow is a NoTrans algorithm");
        const int lane = ex.lane();
        const int nteams = detail::nteams_of(ex), tsize = detail::team_size_of(ex);
        for (int r = ex.team(); r < m; r += nteams) {
            T acc = Ar::zero();
            for (int c = lane; c < n; c += tsize) Ar::mac(acc, a(r, c), x(c));
            acc = ex.team_reduce(acc);
            if (lane == 0) detail::gemv_put<Shape, Ops, AlphaSide>(alpha, acc, beta, y(r));
        }
    } else if constexpr (requires { Al::width; }) {
        static_assert(TR != Transpose::NoTrans, "gemv: alg::ColDot<W> is a (Conj)Trans algorithm");
        constexpr int W = Al::width;
        const int w = ex.lid() & (W - 1), g = ex.lid() / W, ng = detail::size_of(ex) / W;
        const auto grp = detail::lane_group(ex);
        for (int c = g; c < n; c += ng) {
            T acc = Ar::zero();
            for (int r = w; r < m; r += W) {
                if constexpr (conj) Ar::mac_conj(acc, a(r, c), x(r));
                else Ar::mac(acc, a(r, c), x(r));
            }
            acc = butterfly<1, W>(grp, acc, [](const T& p, const T& q) { return Ar::add(p, q); });
            if (w == 0) detail::gemv_put<Shape, Ops, AlphaSide>(alpha, acc, beta, y(c));
        }
    } else {
        static_assert(sizeof(Al) == 0, "gemv: unknown algorithm tag");
    }
}

namespace detail {
template <bool Herm, Uplo UL, class Alg, class Ops, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DEVICE_INLINE void symv_impl(const Ex& ex, AS alpha_, const MA& a, const X& xp, BS beta_, const Y& y) {
    using Al = pick_t<Alg, alg::RowOwnerSplit>;
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    static_assert(UL == Uplo::Lower, "symv/hemv: only Uplo::Lower is implemented");
    static_assert(std::is_same_v<Al, alg::RowOwnerSplit> || (Herm && std::is_same_v<Al, alg::RowOwnerSplitStoredDiag>),
                  "symv/hemv: alg::RowOwnerSplit (or, hemv only, alg::RowOwnerSplitStoredDiag)");
    constexpr bool masked_diag = Herm && Ar::is_complex && std::is_same_v<Al, alg::RowOwnerSplit>;
    require_published<Ex, X>();
    const auto& x = unpub(xp);
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    if constexpr (is_trailing_v<MA>) {
        // Parent coordinates: rows/columns [k, n), the hand kernels' loop shape.
        static_assert(is_trailing_v<std::remove_cvref_t<decltype(x)>> && is_trailing_v<Y>,
                      "symv/hemv: a Trailing A takes Trailing x and y");
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

template <bool Conj, class Alg = alg::Cyclic, class Ops = arith::Explicit, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DEVICE_INLINE void ger_impl(const Ex& ex, AS alpha_, const X& x, const Y& yp, const MA& a) {
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    require_published<Ex, Y>();
    const auto& y = unpub(yp);
    const auto alpha = as_tag(alpha_);
    const int m = a.rows(), n = a.cols();
    if constexpr (std::is_same_v<Alg, alg::Cyclic>) {
        for (int c = 0; c < n; ++c) {
            const T yc = tag_mul(alpha, Conj ? Ar::conj(static_cast<T>(y(c))) : static_cast<T>(y(c)));
            for (int r = ex.lid(); r < m; r += size_of(ex)) Ar::mac(a(r, c), x(r), yc);
        }
    } else {
        static_assert(std::is_same_v<Alg, alg::Flat>, "ger: alg::Cyclic or alg::Flat");
        require_published<Ex, X>();
        const auto& xv = unpub(x);
        for_each_owned_2d<alg::Walk>(ex, m, n, [&](int r, int c) {
            const T yc = tag_mul(alpha, Conj ? Ar::conj(static_cast<T>(y(c))) : static_cast<T>(y(c)));
            Ar::mac(a(r, c), static_cast<T>(xv(r)), yc);
        });
    }
}
}  // namespace detail

// y := alpha A x + beta y, A symmetric with the UL triangle stored.
template <Uplo UL, class Alg = void, class Ops = arith::Explicit, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DEVICE_INLINE void symv(const Ex& ex, AS alpha, const MA& a, const X& x, BS beta, const Y& y) {
    detail::symv_impl<false, UL, Alg, Ops>(ex, alpha, a, x, beta, y);
}
// y := alpha A x + beta y, A Hermitian with the UL triangle stored (Im of the diagonal ignored).
template <Uplo UL, class Alg = void, class Ops = arith::Explicit, class Ex, class AS, class MA, class X, class BS, class Y>
BATCHLAS_DEVICE_INLINE void hemv(const Ex& ex, AS alpha, const MA& a, const X& x, BS beta, const Y& y) {
    detail::symv_impl<true, UL, Alg, Ops>(ex, alpha, a, x, beta, y);
}

// A := A + alpha x y^T (ger, geru) or A + alpha x y^H (gerc). y is read by every item.
// alg::Cyclic: column loop outside, owned rows inside (x Cyclic); alg::Flat: the 2-D flat
// ownership of for_each_owned_2d (x read across owners: Published). Ops: the mac's arithmetic.
// RowFrag form: the item holds row r of A in a register Frag and x(r) as xr: a[j] += xr * alpha y(j).
template <class Alg = alg::Cyclic, class Ops = arith::Explicit, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DEVICE_INLINE void geru(const Ex& ex, AS alpha, const X& x, const Y& y, const MA& a) {
    detail::ger_impl<false, Alg, Ops>(ex, alpha, x, y, a);
}
template <class Alg = alg::Cyclic, class Ops = arith::Explicit, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DEVICE_INLINE void gerc(const Ex& ex, AS alpha, const X& x, const Y& y, const MA& a) {
    detail::ger_impl<true, Alg, Ops>(ex, alpha, x, y, a);
}
template <class Alg = alg::Cyclic, class Ops = arith::Explicit, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DEVICE_INLINE void ger(const Ex& ex, AS alpha, const X& x, const Y& y, const MA& a) {
    geru<Alg, Ops>(ex, alpha, x, y, a);
}
template <bool Conj = false, class Ops = arith::Explicit, class AS, class T, class Y, int N>
BATCHLAS_DEVICE_INLINE void ger_row(AS alpha_, const T& xr, const Y& y, Frag<T, N, RowFrag>& arow, int live = N) {
    using Ar = arith_t<Ops, T>;
    const auto alpha = as_tag(alpha_);
#pragma unroll
    for (int j = 0; j < N; ++j) {
        if (j >= live) continue;
        const T yj = tag_mul(alpha, Conj ? Ar::conj(static_cast<T>(y(j))) : static_cast<T>(y(j)));
        Ar::mac(arow[j], xr, yj);
    }
}

// ---- subtract-from and multi-term gemv, triangular solve / multiply -------------------------

// y := b - A x (NoTrans, RowOwner): the running value starts at b(r) and every term is
// subtracted from it (Ops::msub), the order of the hand kernels' column updates. That rounding
// is not alpha * (A x) + b, so a strict port of such a loop needs this form (latrd P3, sytrf).
// y may alias b.
template <class Ops = arith::Explicit, class Ex, class MA, class X, class B, class Y>
BATCHLAS_DEVICE_INLINE void gemv_sub(const Ex& ex, const MA& a, const X& xp, const B& b, const Y& y) {
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
template <class MA, class X> BATCHLAS_DEVICE_INLINE GemvTerm<MA, X> term(const MA& a, const X& x) { return {a, x}; }

// y := y0 + alpha * sum_t A_t x_t (NoTrans, RowOwner) in ONE accumulator per row: the terms'
// products are summed in argument order, then scaled once (gebrd's gemv_pair, sytrd's Fit
// gemv). Every x_t is read by every item (Published, or assume_published). y may alias y0.
template <class Ops = arith::Explicit, class Ex, class AS, class Y0, class Y, class... Terms>
BATCHLAS_DEVICE_INLINE void gemv_acc(const Ex& ex, AS alpha_, const Y0& y0, const Y& y, const Terms&... terms) {
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
template <Transpose TR, class T, class MA> BATCHLAS_DEVICE_INLINE T op_at(const MA& a, int i, int j) {
    if constexpr (TR == Transpose::NoTrans) return a(i, j);
    else if constexpr (TR == Transpose::Trans) return a(j, i);
    else return Arith<T>::conj(static_cast<T>(a(j, i)));
}
}  // namespace detail

template <Uplo UL, Transpose TR, Diag DG, class Ops = arith::Explicit, class MA, class X>
BATCHLAS_DEVICE_INLINE void trsv(const Thread&, const MA& a, X& x, int n) {
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
BATCHLAS_DEVICE_INLINE void trmv(const Thread&, const MA& a, X& x, int n) {
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
BATCHLAS_DEVICE_INLINE void trsv(const Thread& th, const MA& a, const X& x) {
    X xv = x;
    trsv<UL, TR, DG, Ops>(th, a, xv, x.size());
}
template <Uplo UL, Transpose TR, Diag DG, class Ops = arith::Explicit, class MA, class X>
    requires(requires(const X& v) { v.size(); })
BATCHLAS_DEVICE_INLINE void trmv(const Thread& th, const MA& a, const X& x) {
    X xv = x;
    trmv<UL, TR, DG, Ops>(th, a, xv, x.size());
}

// ---- level 2 (design A.2) -----------------------------------------------------------------------

// A := alpha x y^H + conj(alpha) y x^H + A on part P of A (real T: alpha (x y^T + y x^T)). Each
// element adds tag_mul(alpha, x(r) conj(y(c))) and then tag_mul(conj(alpha), y(r) conj(x(c))), in
// that order; on a triangle (P != Full) the diagonal's imaginary part is set to zero (LAPACK
// zher2). x and y are read across owners (Published).
//   alg::RowOwner    rows Cyclic over the items, each item serial over its row's columns
//   alg::TeamPerRow  team t owns rows t mod nteams, its lanes stride the columns
template <Part P, class Alg = alg::RowOwner, class Ops = arith::Explicit, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DEVICE_INLINE void her2(const Ex& ex, AS alpha_, const X& xp, const Y& yp, const MA& a) {
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    detail::require_published<Ex, X>();
    detail::require_published<Ex, Y>();
    const auto& x = detail::unpub(xp);
    const auto& y = detail::unpub(yp);
    const auto alpha = as_tag(alpha_);
    const auto alpha_c = [&] {
        if constexpr (detail::is_scalar_value_v<std::remove_cv_t<decltype(alpha)>>) return Scalar<T>{Ar::conj(static_cast<T>(alpha.v))};
        else return alpha;
    }();
    using S = std::remove_cv_t<decltype(alpha)>;
    constexpr bool native_tag = std::is_same_v<Ops, arith::Native> && (std::is_same_v<S, One> || std::is_same_v<S, NegOne>);
    const int m = a.rows(), n = a.cols();
    // Columns of row r that part P can touch: the triangle bound keeps a Lower trip count at r + 1.
    auto col_end = [&](int r) BATCHLAS_DEVICE_BODY {
        if constexpr (P == Part::Lower) return sycl::min(n, r + 1);
        else if constexpr (P == Part::StrictLower) return sycl::min(n, r);
        else return n;
    };
    auto upd = [&](int r, int c) BATCHLAS_DEVICE_BODY {
        if (!detail::in_part<P>(r, c)) return;
        T v;
        // Native with a One / NegOne alpha: one expression, so -ffp-contract=on fuses each product
        // into the sum as T's own operators do (the strict-port rounding, D6).
        if constexpr (native_tag && std::is_same_v<S, NegOne>)
            v = static_cast<T>(a(r, c)) - static_cast<T>(x(r)) * Ar::conj(static_cast<T>(y(c))) -
                static_cast<T>(y(r)) * Ar::conj(static_cast<T>(x(c)));
        else if constexpr (native_tag)
            v = static_cast<T>(a(r, c)) + static_cast<T>(x(r)) * Ar::conj(static_cast<T>(y(c))) +
                static_cast<T>(y(r)) * Ar::conj(static_cast<T>(x(c)));
        else {
            const T t1 = Ar::mul(static_cast<T>(x(r)), Ar::conj(static_cast<T>(y(c))));
            const T t2 = Ar::mul(static_cast<T>(y(r)), Ar::conj(static_cast<T>(x(c))));
            v = Ar::add(Ar::add(static_cast<T>(a(r, c)), tag_mul(alpha, t1)), tag_mul(alpha_c, t2));
        }
        if constexpr (Ar::is_complex && P != Part::Full)
            if (r == c) v = Ar::make(Ar::re(v), real_t<T>(0));
        a(r, c) = v;
    };
    if constexpr (std::is_same_v<Alg, alg::RowOwner>) {
        for_each_owned(ex, m, [&](int r) {
            const int ce = col_end(r);
            for (int c = 0; c < ce; ++c) upd(r, c);
        });
    } else {
        static_assert(std::is_same_v<Alg, alg::TeamPerRow>, "her2: alg::RowOwner or alg::TeamPerRow");
        const int nt = detail::nteams_of(ex), ts = detail::team_size_of(ex);
        for (int r = ex.team(); r < m; r += nt) {
            const int ce = col_end(r);
            for (int c = ex.lane(); c < ce; c += ts) upd(r, c);
        }
    }
}

// x := op(A)^{-1} x for A n x n triangular (UL stored), diagonal policy DG (tri_diag::, core.hh):
//   alg::Serial      one item (Thread); x a register Frag (n <= N) or a memory vector
//   alg::LanePerRow  lane i holds x(i) in the register x (n <= the executor's size); step j
//                    finalises x(j) on lane j and broadcasts it, the later lanes update. COLLECTIVE.
namespace detail {
template <class DG, class Ar, class T> BATCHLAS_DEVICE_INLINE T tri_diag_apply(T xi, const T& aii) {
    if constexpr (std::is_same_v<DG, tri_diag::Unit>) return xi;
    else if constexpr (std::is_same_v<DG, tri_diag::Div>) return Ar::div(xi, aii);
    else if constexpr (std::is_same_v<DG, tri_diag::Recip>) return Ar::mul(xi, Ar::recip(aii));
    else {
        static_assert(std::is_same_v<DG, tri_diag::Real>, "trsv: tri_diag::Unit, Div, Recip or Real");
        return Ar::div_real(xi, Ar::re(aii));
    }
}
}  // namespace detail
template <Uplo UL, Transpose TR, class DG, class Alg = alg::Serial, class Ops = arith::Explicit, class Ex, class MA, class X>
BATCHLAS_DEVICE_INLINE void trsv(const Ex& ex, const MA& a, X& x, int n) {
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    constexpr bool lower = (UL == Uplo::Lower) == (TR == Transpose::NoTrans);
    auto diag = [&](int i) -> T {
        if constexpr (std::is_same_v<DG, tri_diag::Unit>) return Ar::one();
        else return detail::op_at<TR, T>(a, i, i);
    };
    if constexpr (std::is_same_v<Alg, alg::Serial>) {
        static_assert(is_thread_v<Ex>, "trsv: alg::Serial is a Thread algorithm");
        constexpr int N = detail::tri_cap_of(static_cast<const X*>(nullptr));
        if constexpr (N > 0) {
            if constexpr (lower) {
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if (i >= n) break;
                    T xi = x[i];
#pragma unroll
                    for (int j = 0; j < i; ++j) Ar::msub(xi, detail::op_at<TR, T>(a, i, j), x[j]);
                    x[i] = detail::tri_diag_apply<DG, Ar>(xi, diag(i));
                }
            } else {
#pragma unroll
                for (int i = N - 1; i >= 0; --i) {
                    if (i >= n) continue;
                    T xi = x[i];
#pragma unroll
                    for (int j = i + 1; j < N; ++j)
                        if (j < n) Ar::msub(xi, detail::op_at<TR, T>(a, i, j), x[j]);
                    x[i] = detail::tri_diag_apply<DG, Ar>(xi, diag(i));
                }
            }
        } else if constexpr (lower) {
            for (int i = 0; i < n; ++i) {
                T xi = x(i);
                for (int j = 0; j < i; ++j) Ar::msub(xi, detail::op_at<TR, T>(a, i, j), static_cast<T>(x(j)));
                x(i) = detail::tri_diag_apply<DG, Ar>(xi, diag(i));
            }
        } else {
            for (int i = n - 1; i >= 0; --i) {
                T xi = x(i);
                for (int j = i + 1; j < n; ++j) Ar::msub(xi, detail::op_at<TR, T>(a, i, j), static_cast<T>(x(j)));
                x(i) = detail::tri_diag_apply<DG, Ar>(xi, diag(i));
            }
        }
    } else {
        static_assert(std::is_same_v<Alg, alg::LanePerRow>, "trsv: alg::Serial or alg::LanePerRow");
        static_assert(std::is_same_v<X, T>, "trsv<LanePerRow>: x is the lane's register element");
        const int l = ex.lid();
        for (int s = 0; s < n; ++s) {
            const int j = lower ? s : n - 1 - s;
            if (l == j) x = detail::tri_diag_apply<DG, Ar>(x, diag(j));
            const T xj = ex.bcast(x, j);
            if (lower ? (l > j && l < n) : (l < j)) Ar::msub(x, detail::op_at<TR, T>(a, l, j), xj);
        }
    }
}

// A := diag(d) A (Side::Left) or A diag(d) (Side::Right) in place, or C := the same of B (four
// operands; C may alias B); 2-D flat ownership. A real d on a complex matrix scales both parts
// (no cross terms); otherwise each element is mul(b, d).
template <Side SD, class Ops = arith::Explicit, class Ex, class D, class MA>
BATCHLAS_DEVICE_INLINE void dgmm(const Ex& ex, const D& d, const MA& a) {
    using T = typename MA::value_type;
    using A = arith_t<Ops, T>;
    using S = std::remove_cvref_t<decltype(d(0))>;
    for_each_owned_2d(ex, a.rows(), a.cols(), [&](int i, int j) {
        const S s = d(SD == Side::Left ? i : j);
        const T aij = a(i, j);
        if constexpr (A::is_complex && !Arith<S>::is_complex) a(i, j) = A::make(A::re(aij) * s, A::im(aij) * s);
        else a(i, j) = A::mul(aij, static_cast<T>(s));
    });
}
template <Side SD, class Ops = arith::Explicit, class Ex, class D, class MB, class MC>
BATCHLAS_DEVICE_INLINE void dgmm(const Ex& ex, const D& d, const MB& b, const MC& c) {
    using T = typename MC::value_type;
    using A = arith_t<Ops, T>;
    using S = std::remove_cvref_t<decltype(d(0))>;
    for_each_owned_2d(ex, c.rows(), c.cols(), [&](int i, int j) {
        const S s = d(SD == Side::Left ? i : j);
        const T bij = b(i, j);
        if constexpr (A::is_complex && !Arith<S>::is_complex) c(i, j) = A::make(A::re(bij) * s, A::im(bij) * s);
        else c(i, j) = A::mul(bij, static_cast<T>(s));
    });
}

// y := alpha A x + beta y for A m x n general band with KL sub- and KU super-diagonals in LAPACK
// band storage, A(i, j) = ab(KU + i - j, j). Row owner, ascending j; x read across owners
// (Published); beta == Zero never reads y.
template <int KL, int KU, class Ops = arith::Explicit, class Ex, class AS, class MB, class X, class BS, class Y>
BATCHLAS_DEVICE_INLINE void gbmv(const Ex& ex, AS alpha_, const MB& ab, const X& xp, BS beta_, const Y& y) {
    static_assert(KL >= 0 && KU >= 0, "gbmv: KL, KU >= 0");
    using T = elem_t<Y>;
    using Ar = arith_t<Ops, T>;
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    const int n = x.size();
    for_each_owned(ex, y.size(), [&](int i) {
        T acc = Ar::zero();
        const int j0 = i - KL > 0 ? i - KL : 0, j1 = i + KU < n - 1 ? i + KU : n - 1;
        for (int j = j0; j <= j1; ++j) Ar::mac(acc, static_cast<T>(ab(KU + i - j, j)), static_cast<T>(x(j)));
        store<epi::Sum, Ops>(alpha, acc, beta, y(i));
    });
}

}  // namespace batchlas::device
