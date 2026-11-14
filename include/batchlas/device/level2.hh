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

#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/view.hh>

namespace batchlas::device {

namespace detail {
template <class Ex, Transpose TR> constexpr auto default_gemv_alg() {
    if constexpr (is_thread_v<Ex>) return alg::Serial{};
    else if constexpr (TR == Transpose::NoTrans) return alg::RowOwner{};
    else return alg::SgPerCol{};
}
template <class Alg, class Default> using pick_t = std::conditional_t<std::is_void_v<Alg>, Default, Alg>;
// gemv's output: store_epi<Shape, Ops>(alpha, acc, beta, y); Side::Right with a Scalar alpha
// forms acc * alpha first (then beta's term as a One-alpha store).
template <class S> inline constexpr bool is_scalar_value_v = false;
template <class T> inline constexpr bool is_scalar_value_v<Scalar<T>> = true;
template <class Shape, class Ops, Side AlphaSide, class AS, class T, class BS, class Ref>
BATCHLAS_DEVICE_INLINE void gemv_put(AS alpha, T acc, BS beta, Ref&& y) {
    if constexpr (AlphaSide == Side::Right && is_scalar_value_v<AS>) {
        store_epi<Shape, Ops>(One{}, arith_t<Ops, T>::mul(acc, static_cast<T>(alpha.v)), beta, y);
    } else {
        store_epi<Shape, Ops>(alpha, acc, beta, y);
    }
}
template <class Ex> BATCHLAS_DEVICE_INLINE auto lane_group(const Ex& ex) {
    if constexpr (requires { ex.part; }) return ex.part;
    else return ex.sub();
}
}  // namespace detail

// y := alpha op(A) x + beta y; beta == Zero never reads y. Ops: the arithmetic of the dot;
// Shape: the epilogue (store_epi<Shape, Ops>, arith.hh); AlphaSide: Side::Right forms acc * alpha.
//   alg::TeamPerRow  NoTrans: team t owns rows t mod nteams, lanes stride the columns, lane 0 stores
//   alg::ColDot<W>   (Conj)Trans: W consecutive lanes per column (W | the sub-group), a W-lane
//                    butterfly; the W-block's first lane stores
template <Transpose TR, class Alg = void, class Ops = arith::Explicit, class Shape = epi::Sum, Side AlphaSide = Side::Left,
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
            store_epi(alpha, acc, beta, y(r));
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
        store_epi(alpha, acc, beta, y(r));
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

// Register-tile form on the Thread executor: A := x y^T + A, one Ops::mac per entry, row by row
// (x, y register Frags: the memory-operand forms above read y per column).
template <class Ops = arith::Explicit, class T, int R, int C, class L1, class L2>
BATCHLAS_DEVICE_INLINE void ger(const Thread&, const Frag<T, R, L1>& x, const Frag<T, C, L2>& y, Tile<T, R, C>& a) {
    using Ar = arith_t<Ops, T>;
BATCHLAS_UNROLL_FULL
    for (int i = 0; i < R; ++i) {
BATCHLAS_UNROLL_FULL
        for (int j = 0; j < C; ++j) Ar::mac(a(i, j), x[i], y[j]);
    }
}

// ---- the guarded diagonal step of trsv's RowFrag block form ----------------------------------
// The diagonal of row lid(): its reciprocal, divide = any reciprocal in the partition is
// non-finite (a bit_or team_reduce, shuffles only), unit = the diagonal is not applied.
template <class T> struct GuardedRecip {
    T d, rinv;
    bool divide, unit;
};
template <class Ar, int P, class T> BATCHLAS_DEVICE_INLINE GuardedRecip<T> guarded_recip(const Partition<P>& ex, T d, bool unit) {
    const T rinv = Ar::recip(d);
    int bad = Ar::isfinite(rinv) ? 0 : 1;
    bad = ex.team_reduce(bad, sycl::bit_or<int>());
    return {d, rinv, bad != 0, unit};
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

// x := op(A) x (trmv) in place, A n x n triangular (UL stored), on one item: x is a register
// Frag<T, N> (n <= N live entries, every index a compile-time constant) or a memory vector
// (runtime n). Entries past n are not touched. Diag::Unit never reads the diagonal; the opposite
// triangle is never read. trsv (below) solves with the same operands. Alg: alg::Serial, or
// alg::SplitDot<W> (as trsv's: x_i = re(a_ii) x_i + the split column dot).
namespace detail {
template <class X> constexpr int tri_cap_of(const X*) { return 0; }
template <class T, int N, class L> constexpr int tri_cap_of(const Frag<T, N, L>*) { return N; }

// Element (i, j) of op(A) for a stored UL triangle; called only inside op(A)'s triangle.
template <Transpose TR, class T, class MA> BATCHLAS_DEVICE_INLINE T op_at(const MA& a, int i, int j) {
    if constexpr (TR == Transpose::NoTrans) return a(i, j);
    else if constexpr (TR == Transpose::Trans) return a(j, i);
    else return Arith<T>::conj(static_cast<T>(a(j, i)));
}
template <class A> inline constexpr bool is_split_dot_v = false;
template <int W> inline constexpr bool is_split_dot_v<alg::SplitDot<W>> = true;

// One aligned 16-byte load of consecutive elements (one element when T is 16 bytes).
template <class T> struct alignas(16) Pack16 {
    T e[16 / sizeof(T) > 0 ? 16 / sizeof(T) : 1];
};
template <class T> inline constexpr int pack16_elems = static_cast<int>(sizeof(Pack16<T>::e) / sizeof(T));

// alg::SplitDot<W>, trsv (Solve) and trmv: x_i combines column i of A with the other entries of
// x (Below: p < i, else p > i); Solve: x_i = (x_i - sum) / a_ii, else x_i = a_ii x_i + sum. In
// place, so i is visited before every entry whose sum still needs it.
template <bool Below, bool Solve, bool Conj, int W, class Ops, class T, int N, class L, class MA>
BATCHLAS_DEVICE_INLINE void split_dot_tri(const MA& a, Frag<T, N, L>& x, int n) {
    static_assert(W > 0 && (W & (W - 1)) == 0, "trsv / trmv alg::SplitDot<W>: W is a power of two");
    using A = Arith<T>;
    using Ar = arith_t<Ops, T>;
    constexpr int V = pack16_elems<T>;
    constexpr int K = static_extent_v<std::remove_cv_t<decltype(a.n)>>;  // -1: a run-time extent
    constexpr int Trip = K > 0 ? K : N;
    static_assert(Trip <= N, "trsv / trmv alg::SplitDot<W>: the extent exceeds the fragment");
    constexpr bool ascending = Below == Solve;
    auto step = [&](T& acc, const T& aij, const T& xj) BATCHLAS_DEVICE_BODY {
        if constexpr (Solve) Ar::msub(acc, Conj ? A::conj(aij) : aij, xj);  // acc - op(a) x
        else if constexpr (Conj) Ar::mac_conj(acc, aij, xj);
        else Ar::mac(acc, aij, xj);
    };
    BATCHLAS_UNROLL_FULL
    for (int s = 0; s < Trip; ++s) {
        const int i = ascending ? s : Trip - 1 - s;
        if (i < n) {
            T acc[W];
            BATCHLAS_UNROLL_FULL
            for (int k = 0; k < W; ++k) acc[k] = A::zero();
            const T* const line = &a(0, i);  // column i of A
            if constexpr (Below) {
                BATCHLAS_UNROLL_FULL
                for (int p0 = 0; p0 < i; p0 += V) {
                    const Pack16<T> pk = *reinterpret_cast<const Pack16<T>*>(line + p0);
                    BATCHLAS_UNROLL_FULL
                    for (int u = 0; u < V; ++u)
                        if (p0 + u < i) step(acc[(p0 + u) % W], pk.e[u], x[p0 + u]);
                }
            } else {
                BATCHLAS_UNROLL_FULL
                for (int p0 = (i + 1) / V * V; p0 < Trip; p0 += V) {
                    if (p0 >= n) continue;
                    const Pack16<T> pk = *reinterpret_cast<const Pack16<T>*>(line + p0);
                    BATCHLAS_UNROLL_FULL
                    for (int u = 0; u < V; ++u)
                        if (p0 + u > i && p0 + u < n) step(acc[(p0 + u) % W], pk.e[u], x[p0 + u]);
                }
            }
            BATCHLAS_UNROLL_FULL
            for (int w = W / 2; w >= 1; w /= 2) {
                BATCHLAS_UNROLL_FULL
                for (int k = 0; k < w; ++k) acc[k] = A::add(acc[2 * k], acc[2 * k + 1]);
            }
            const auto d = A::re(line[i]);
            if constexpr (Solve) x[i] = div_real(A::add(x[i], acc[0]), d);
            else x[i] = A::add(scale_real(d, x[i]), acc[0]);
        }
    }
}

}  // namespace detail

template <Uplo UL, Transpose TR, Diag DG, class Ops = arith::Explicit, class Alg = alg::Serial, class MA, class X>
BATCHLAS_DEVICE_INLINE void trmv(const Thread&, const MA& a, X& x, int n) {
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    constexpr int N = detail::tri_cap_of(static_cast<const X*>(nullptr));
    constexpr bool lower = (UL == Uplo::Lower) == (TR == Transpose::NoTrans);
    if constexpr (detail::is_split_dot_v<Alg>) {
        static_assert(TR != Transpose::NoTrans && DG == Diag::NonUnit, "trmv: alg::SplitDot<W> is the (Conj)Trans form with a real diagonal");
        detail::split_dot_tri<UL == Uplo::Upper, false, TR == Transpose::ConjTrans, Alg::width, Ops>(a, x, n);
        return;
    } else {
        static_assert(std::is_same_v<Alg, alg::Serial>, "trmv: alg::Serial or alg::SplitDot<W>");
    }
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
// The memory-vector form takes x by const view (the view, not the data, is const).
template <Uplo UL, Transpose TR, Diag DG, class Ops = arith::Explicit, class MA, class X>
    requires(requires(const X& v) { v.size(); })
BATCHLAS_DEVICE_INLINE void trmv(const Thread& th, const MA& a, const X& x) {
    X xv = x;
    trmv<UL, TR, DG, Ops>(th, a, xv, x.size());
}

// ---- level 2 (design A.2) -----------------------------------------------------------------------

// A := alpha x y^H + conj(alpha) y x^H + A on the UL triangle of A (real T: alpha (x y^T + y x^T));
// the other triangle is neither read nor written. Each element adds tag_mul(alpha, x(r) conj(y(c)))
// and then tag_mul(conj(alpha), y(r) conj(x(c))), in that order; the diagonal's imaginary part is
// set to zero (LAPACK zher2). x and y are read across owners (Published).
//   alg::RowOwner    rows Cyclic over the items, each item serial over its row's columns
//   alg::TeamPerRow  team t owns rows t mod nteams, its lanes stride the columns
template <Uplo UL, class Alg = alg::RowOwner, class Ops = arith::Explicit, class Ex, class AS, class X, class Y, class MA>
BATCHLAS_DEVICE_INLINE void her2(const Ex& ex, AS alpha_, const X& xp, const Y& yp, const MA& a) {
    constexpr Part P = UL == Uplo::Lower ? Part::Lower : Part::Upper;
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
    // Columns of row r that the triangle can touch: the bound keeps a Lower trip count at r + 1.
    auto col_end = [&](int r) BATCHLAS_DEVICE_BODY {
        if constexpr (P == Part::Lower) return sycl::min(n, r + 1);
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
        if constexpr (Ar::is_complex)
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

// x := op(A)^{-1} x in place for A n x n triangular (UL stored; the opposite triangle is never
// read), entries past n untouched. Diag::Unit never reads the diagonal (nor takes the last
// pivot, which updates nothing); Diag::NonUnit divides by it under Div: alg::Div, alg::Recip or
// alg::RealDiv (core.hh) on a_ii, or alg::DiagApply, x_i := d.apply<Ar>(i, x_i) for a diagonal
// step object d (TpDiag, GuardedRecip: a guarded reciprocal, unit at run time). The diagonal is
// op(A)'s own, or d(i) when the diagonal operand d is a vector (potrf's real diagonal). Upd:
//   alg::Fused         Ops::msub
//   alg::Unfused       sub of mul, a * x_j: two roundings (getrs / sytrs)
//   alg::UnfusedRight  sub of mul, x_j * a: the X op(A) = B operand order (potrf; a complex fma
//                      product is not bit-commutative)
//   alg::DotThenSub    the row's dot from zero, then x_i := b(i) - dot (alg::SerialRows)
//   alg::Axpy          x += a x_j on every row with a the caller's NEGATED multipliers, zero on and
//                      above the diagonal (no row predicate; alg::LanePerRow on a RowFrag)
// Alg:
//   alg::Serial        one item (Thread); x a register Frag<T, N> (n <= N live entries, every index
//                      a compile-time constant) or a memory vector (runtime n)
//   alg::SerialRows<R> Lower / NoTrans alg::Serial on rows [r0, r0 + R) only (rows below r0 solved), the
//                      right-hand side b(i) read after the row's dot (the tiled forward solve)
//   alg::SplitDot<W>   Thread, (Conj)Trans, Diag::NonUnit with alg::RealDiv: x_i combines column i
//                      of A with the solved entries in W accumulators (term p in p mod W) summed as
//                      a tree, 16-byte column loads (every column start of A 16-byte aligned)
//   alg::LanePerRow    row t on lane t % size: x the lane's register (n <= size) or a lane-cyclic
//                      Frag<T, R, Cyclic> (row lane + size r in x[r]); step j finalises row j and
//                      broadcasts it, the rows it reaches update. No barrier. COLLECTIVE. On a
//                      RowFrag x (row lane of an n x QC block X, a trsm over QC columns; Lower,
//                      NoTrans, alg::Axpy, alg::DiagApply with a GuardedRecip, A the lane's row of
//                      multipliers as a Frag): the division sits behind a partition-uniform branch.
//   alg::LanePerRowRolled  the RowFrag form with the step loop rolled (A's row indexed at run
//                      time: local memory, fewer registers)
namespace detail {
template <Diag DG, class Div, class Ar, class T, class D> BATCHLAS_DEVICE_INLINE T trsv_diag_apply(T xi, const D& aii) {
    if constexpr (DG == Diag::Unit) return xi;
    else if constexpr (std::is_same_v<Div, alg::Div>) return Ar::div(xi, aii);
    else if constexpr (std::is_same_v<Div, alg::Recip>) return Ar::mul(xi, Ar::recip(aii));
    else {
        static_assert(std::is_same_v<Div, alg::RealDiv>, "trsv: alg::Div, alg::Recip, alg::RealDiv or alg::DiagApply");
        if constexpr (std::is_same_v<D, real_t<T>>) return Ar::div_real(xi, aii);
        else return Ar::div_real(xi, Ar::re(aii));
    }
}
template <class Upd, class Ar, class T> BATCHLAS_DEVICE_INLINE void trsv_update(T& x, const T& a, const T& xj) {
    if constexpr (std::is_same_v<Upd, alg::Fused>) Ar::msub(x, a, xj);
    else if constexpr (std::is_same_v<Upd, alg::UnfusedRight>) x = Ar::sub(x, Ar::mul(xj, a));
    else {
        static_assert(std::is_same_v<Upd, alg::Unfused>, "trsv: alg::Fused, Unfused or UnfusedRight here");
        x = Ar::sub(x, Ar::mul(a, xj));
    }
}
template <class A> inline constexpr int serial_rows_v = 0;
template <int R> inline constexpr int serial_rows_v<alg::SerialRows<R>> = R;
template <class X> inline constexpr bool is_row_frag_v = false;
template <class T, int N> inline constexpr bool is_row_frag_v<Frag<T, N, RowFrag>> = true;
// alg::DiagApply's step: x_i := d.apply<Ar>(i, x_i).
template <class Ar, class D, class T> BATCHLAS_DEVICE_INLINE T trsv_step(const D& d, int i, T v) { return d.template apply<Ar>(i, v); }
// trsv's element type: A's (an OpView has no value_type), or a row Frag's (the RowFrag form).
template <class MA> struct trsv_value {
    using type = std::remove_cvref_t<decltype(std::declval<const MA&>()(0, 0))>;
};
template <class T, int N, class L> struct trsv_value<Frag<T, N, L>> {
    using type = T;
};

// The one trsv body: d the diagonal operand (None: op(A)'s own), b the right-hand side of
// alg::SerialRows (None: x in place) and r0 its first row.
template <Uplo UL, Transpose TR, Diag DG, class Div, class Alg, class Ops, class Upd, class Ex, class MA, class DV, class X, class B>
BATCHLAS_DEVICE_INLINE void trsv_run(const Ex& ex, const MA& a, const DV& d, X& x, int n, int r0, const B& b) {
    using T = typename trsv_value<MA>::type;
    using Ar = arith_t<Ops, T>;
    constexpr bool lower = (UL == Uplo::Lower) == (TR == Transpose::NoTrans);  // forward substitution
    constexpr bool step_obj = std::is_same_v<Div, alg::DiagApply>;
    static_assert(!step_obj || (DG == Diag::NonUnit && !is_none_v<DV>), "trsv: alg::DiagApply takes the step object d (Diag::NonUnit)");
    auto diag = [&](auto i) -> T {
        if constexpr (DG == Diag::Unit) return Ar::one();
        else if constexpr (!is_none_v<DV>) return static_cast<T>(d(i));
        else return detail::op_at<TR, T>(a, i, i);
    };
    // x_i after its updates: the diagonal step.
    auto finish = [&](T xi, auto i) -> T {
        if constexpr (step_obj) return trsv_step<Ar>(d, i, xi);
        else if constexpr (DG == Diag::NonUnit && !is_none_v<DV> && std::is_same_v<Div, alg::RealDiv>)
            return detail::trsv_diag_apply<DG, Div, Ar>(xi, d(i));
        else return detail::trsv_diag_apply<DG, Div, Ar>(xi, diag(i));
    };
    auto upd = [&](T& xi, int i, auto j, const T& xj) { detail::trsv_update<Upd, Ar>(xi, detail::op_at<TR, T>(a, i, j), xj); };
    if constexpr (is_split_dot_v<Alg>) {
        static_assert(is_thread_v<Ex> && TR != Transpose::NoTrans && DG == Diag::NonUnit && std::is_same_v<Div, alg::RealDiv> &&
                          is_none_v<DV>,
                      "trsv: alg::SplitDot<W> is the Thread (Conj)Trans form with a real diagonal (alg::RealDiv)");
        split_dot_tri<UL == Uplo::Upper, true, TR == Transpose::ConjTrans, Alg::width, Ops>(a, x, n);
    } else if constexpr (serial_rows_v<Alg> > 0) {
        static_assert(is_thread_v<Ex> && UL == Uplo::Lower && TR == Transpose::NoTrans && std::is_same_v<Upd, alg::DotThenSub> && !is_none_v<B>,
                      "trsv: alg::SerialRows<R> is the Thread Lower / NoTrans solve with alg::DotThenSub and a right-hand side b");
        constexpr int R = serial_rows_v<Alg>;
        constexpr int N = detail::tri_cap_of(static_cast<const X*>(nullptr));
        static_assert(N > 0, "trsv: alg::SerialRows<R> solves a register Frag");
        BATCHLAS_UNROLL_FULL
        for (int k = 0; k < R; ++k) {
            const int s = r0 + k;
            if (s >= N || s >= n) continue;
            T acc = Ar::zero();
            BATCHLAS_UNROLL_FULL
            for (int t = 0; t < N; ++t) {
                if (t < s) Ar::mac(acc, a(s, t), x[t]);  // by reference: a complex element is read part by part
            }
            const T rhs = b(s);
            x[s] = finish(Ar::sub(rhs, acc), s);
        }
    } else if constexpr (std::is_same_v<Alg, alg::Serial>) {
        static_assert(is_thread_v<Ex>, "trsv: alg::Serial is a Thread algorithm");
        constexpr int N = detail::tri_cap_of(static_cast<const X*>(nullptr));
        if constexpr (N > 0) {
            if constexpr (lower) {
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if (i >= n) break;
                    T xi = x[i];
#pragma unroll
                    for (int j = 0; j < i; ++j) upd(xi, i, j, x[j]);
                    x[i] = finish(xi, i);
                }
            } else {
#pragma unroll
                for (int i = N - 1; i >= 0; --i) {
                    if (i >= n) continue;
                    T xi = x[i];
#pragma unroll
                    for (int j = i + 1; j < N; ++j)
                        if (j < n) upd(xi, i, j, x[j]);
                    x[i] = finish(xi, i);
                }
            }
        } else if constexpr (lower) {
            for (int i = 0; i < n; ++i) {
                T xi = x(i);
                for (int j = 0; j < i; ++j) upd(xi, i, j, static_cast<T>(x(j)));
                x(i) = finish(xi, i);
            }
        } else {
            for (int i = n - 1; i >= 0; --i) {
                T xi = x(i);
                for (int j = i + 1; j < n; ++j) upd(xi, i, j, static_cast<T>(x(j)));
                x(i) = finish(xi, i);
            }
        }
    } else if constexpr (is_row_frag_v<X>) {
        static_assert(std::is_same_v<Alg, alg::LanePerRow> || std::is_same_v<Alg, alg::LanePerRowRolled>,
                      "trsv: a RowFrag x is the alg::LanePerRow(Rolled) block form");
        static_assert(lower && std::is_same_v<Upd, alg::Axpy> && step_obj,
                      "trsv: the RowFrag block form is Lower / NoTrans with alg::Axpy and alg::DiagApply");
        constexpr int P = Ex::size;
        constexpr int QC = detail::tri_cap_of(static_cast<const X*>(nullptr));
        constexpr int Unroll = std::is_same_v<Alg, alg::LanePerRowRolled> ? 1 : P;
        // lane % P, not ex.lid() (lane & (P-1)): the AND form shifts ptxas' register allocation in
        // the unrolled P=16 kernels by 3-4% of the instructions.
        const int r = static_cast<int>(ex.part.get_local_linear_id()) % P;
        const bool any_divide = sycl::any_of_group(ex.part, d.divide);
#pragma unroll Unroll
        for (int s = 0; s < P; ++s) {
            if (s >= n) {
                if constexpr (Unroll == 1) break;
                else continue;
            }
            if (r == s && !d.unit) {
                if (any_divide) [[unlikely]] {
                    BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < QC; ++j) x[j] = d.divide ? Ar::div(x[j], d.d) : Ar::mul(x[j], d.rinv);
                } else {
                    BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < QC; ++j) x[j] = Ar::mul(x[j], d.rinv);
                }
            }
            const T l = a[s];
            BATCHLAS_UNROLL_FULL
            for (int j = 0; j < QC; ++j) {
                const T xs = ex.bcast(x[j], static_cast<std::uint32_t>(s));
                Ar::mac(x[j], l, xs);
            }
        }
    } else {
        static_assert(std::is_same_v<Alg, alg::LanePerRow>, "trsv: alg::Serial, SerialRows, SplitDot, LanePerRow or LanePerRowRolled");
        const int l = ex.lid();
        auto step = [&](int j) BATCHLAS_DEVICE_BODY {
            if constexpr (detail::tri_cap_of(static_cast<const X*>(nullptr)) == 0) {  // the lane's register
                if constexpr (DG == Diag::NonUnit) {
                    if (l == j) x = finish(x, j);
                }
                const T xj = ex.bcast(x, j);
                if (lower ? (l > j && l < n) : (l < j)) upd(x, l, j, xj);
            } else {
                constexpr int R = detail::tri_cap_of(static_cast<const X*>(nullptr));
                static_assert(R > 0, "trsv<LanePerRow>: x is the lane's register or a lane-cyclic Frag");
                constexpr int P = Ex::size;
                if constexpr (DG == Diag::NonUnit) {
                    BATCHLAS_UNROLL_FULL
                    for (int r = 0; r < R; ++r)
                        if (l + P * r == j) x[r] = finish(x[r], j);
                }
                const T xj = frag_bcast(ex, x, j);
                BATCHLAS_UNROLL_FULL
                for (int r = 0; r < R; ++r) {
                    const int t = l + P * r;
                    if (lower ? (t > j && t < n) : (t < j)) upd(x[r], t, j, xj);
                }
            }
        };
        const int last = DG == Diag::Unit ? 1 : 0;
        if constexpr (lower) {
            for (int j = 0; j < n - last; ++j) step(j);
        } else {
            for (int j = n - 1; j >= last; --j) step(j);
        }
    }
}
}  // namespace detail
template <Uplo UL, Transpose TR, Diag DG, class Div = alg::Div, class Alg = alg::Serial, class Ops = arith::Explicit,
          class Upd = alg::Fused, class Ex, class MA, class X>
BATCHLAS_DEVICE_INLINE void trsv(const Ex& ex, const MA& a, X& x, int n) {
    detail::trsv_run<UL, TR, DG, Div, Alg, Ops, Upd>(ex, a, None{}, x, n, 0, None{});
}
// With the diagonal operand d: a vector (d(i) is a_ii) or, under alg::DiagApply, a step object.
template <Uplo UL, Transpose TR, Diag DG, class Div = alg::Div, class Alg = alg::Serial, class Ops = arith::Explicit,
          class Upd = alg::Fused, class Ex, class MA, class DV, class X>
BATCHLAS_DEVICE_INLINE void trsv(const Ex& ex, const MA& a, const DV& d, X& x, int n) {
    detail::trsv_run<UL, TR, DG, Div, Alg, Ops, Upd>(ex, a, d, x, n, 0, None{});
}
// alg::SerialRows<R>: rows [r0, r0 + R) of the n-row solve, b(i) their right-hand side.
template <Uplo UL, Transpose TR, Diag DG, class Div, class Alg, class Ops, class Upd, class Ex, class MA, class DV, class X, class B>
BATCHLAS_DEVICE_INLINE void trsv(const Ex& ex, const MA& a, const DV& d, X& x, int n, int r0, const B& b) {
    detail::trsv_run<UL, TR, DG, Div, Alg, Ops, Upd>(ex, a, d, x, n, r0, b);
}
// The memory-vector form takes x by const view (the view, not the data, is const).
template <Uplo UL, Transpose TR, Diag DG, class Div = alg::Div, class Ops = arith::Explicit, class MA, class X>
    requires(requires(const X& v) { v.size(); })
BATCHLAS_DEVICE_INLINE void trsv(const Thread& th, const MA& a, const X& x) {
    X xv = x;
    trsv<UL, TR, DG, Div, alg::Serial, Ops>(th, a, xv, x.size());
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
        store_epi<epi::Sum, Ops>(alpha, acc, beta, y(i));
    });
}

}  // namespace batchlas::device
