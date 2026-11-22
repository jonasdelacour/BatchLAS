// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief In-kernel BLAS level 1: vector operations, reductions and the ownership loops they are
/// built from (@ref api_device_blas1).
///
/// Every operation distributes over the executor's cyclic layout: element `i` is owned by the
/// item with `lid == i mod size`. Two operations on the same executor therefore touch an element
/// from the same item and compose with no barrier. No operation here contains a work-group
/// barrier except the packed `Scratch` exchange, which names the scratch in its signature.
/// Reductions return a uniform result: every item of the reduction scope gets the same value.
///
/// Included by `<batchlas/device.hh>`.
/// @ingroup api_device_blas1

#include <array>
#include <climits>
#include <limits>
#include <type_traits>
#include <utility>

#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/operand.hh>
#include <batchlas/device/reduce.hh>
#include <batchlas/device/scalar.hh>

namespace batchlas::device {

/// @brief The element type of a vector operand `V`: what `v(0)` returns, without cv and reference.
/// @ingroup api_device_blas1
/// @tparam V a vector operand (anything callable as `v(i)`).
template <class V> using elem_t = std::remove_cvref_t<decltype(std::declval<const V&>()(0))>;

// The primitive: f(i) for every i this item owns, of [0, n) or of a Range [lo, hi) in parent
// coordinates (i owned by lid == (i - lo) mod size, as an op on x.tail(lo): the two compose with
// no barrier). A Range keeps the hand loop's bounds: a 0-based loop over a tail view unrolls
// differently (latrd P3, gebrd, stein). U is the unroll policy of this loop (core.hh unroll::),
// emitted as loop metadata; whether ptxas honours a PTX nounroll needs a gate check per use.
namespace detail {
template <class U, class F> BATCHLAS_DEVICE_INLINE void strided_loop(int lo, int hi, int step, F& f) {
    if constexpr (std::is_same_v<U, alg::UnrollDefault>) {
        for (int i = lo; i < hi; i += step) f(i);
    } else if constexpr (std::is_same_v<U, alg::UnrollNone>) {
#pragma clang loop unroll(disable)
        for (int i = lo; i < hi; i += step) f(i);
    } else if constexpr (std::is_same_v<U, alg::UnrollFull>) {
#pragma clang loop unroll(full)
        for (int i = lo; i < hi; i += step) f(i);
    } else {
        constexpr int N = U::count;
        static_assert(N >= 1, "for_each_owned: alg::UnrollCount<N> needs N >= 1");
#pragma clang loop unroll_count(N)
        for (int i = lo; i < hi; i += step) f(i);
    }
}
}  // namespace detail

/// @brief Calls `f(i)` for every `i` in `[0, n)` that this item owns.
///
/// The ownership primitive every level-1 operation is built on: item `lid` visits
/// `i = lid, lid + size, ...` in ascending order. Operations that use the same executor and
/// `n` therefore touch each element from the same item, and compose with no barrier.
/// On a one-item executor (`Thread`) the loop visits every `i`.
/// @ingroup api_device_blas1
/// @tparam U the unroll policy of the loop: `alg::UnrollDefault` (the compiler decides),
///   `alg::UnrollNone`, `alg::UnrollFull` or `alg::UnrollCount<N>` (`N >= 1`), emitted as loop
///   metadata.
/// @param ex the executor.
/// @param n the extent; `n <= 0` visits nothing.
/// @param f called as `f(int i)`.
template <class U = alg::UnrollDefault, class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_owned(const Ex& ex, int n, F&& f) {
    if constexpr (std::is_same_v<U, alg::UnrollDefault> && !is_runtime_v<Ex>) {
        for (int i = ex.lid(); i < n; i += Ex::size) f(i);  // the first spelling, SASS-pinned by the lab
    } else {
        detail::strided_loop<U>(ex.lid(), n, detail::size_of(ex), f);
    }
}
/// @brief Calls `f(i)` for every `i` in `[r.lo, r.hi)` that this item owns, in parent coordinates.
///
/// `i` is owned by the item with `lid == (i - r.lo) mod size`, the same ownership as an operation
/// on the view `x.tail(r.lo)`, so the two compose with no barrier.
/// @ingroup api_device_blas1
/// @tparam U the unroll policy, as for the `[0, n)` form.
/// @param ex the executor.
/// @param r the index range; an empty range visits nothing.
/// @param f called as `f(int i)` with `i` in `[r.lo, r.hi)`.
template <class U = alg::UnrollDefault, class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_owned(const Ex& ex, Range r, F&& f) {
    if constexpr (std::is_same_v<U, alg::UnrollDefault> && !is_runtime_v<Ex>) {
        for (int i = r.lo + ex.lid(); i < r.hi; i += Ex::size) f(i);
    } else {
        detail::strided_loop<U>(r.lo + ex.lid(), r.hi, detail::size_of(ex), f);
    }
}
/// @brief `for_each_owned(ex, Range{first, end}, f)`: calls `f(i)` for the owned `i` in `[first, end)`.
/// @ingroup api_device_blas1
/// @tparam U the unroll policy, as for the `[0, n)` form.
/// @param ex the executor.
/// @param first the first index of the range.
/// @param end one past the last index.
/// @param f called as `f(int i)`.
template <class U = alg::UnrollDefault, class Ex, class F>
BATCHLAS_DEVICE_INLINE void for_each_owned(const Ex& ex, int first, int end, F&& f) {
    for_each_owned<U>(ex, Range{first, end}, f);
}
/// @brief Calls `f(ex.lid())` once, with no loop and no bound test.
/// @ingroup api_device_lowlevel
/// @pre `n` equals the executor's size, so each item owns exactly `i = lid`.
/// @param ex the executor.
/// @param n the extent; not read.
/// @param f called once as `f(int i)`.
template <class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_owned_exact(const Ex& ex, [[maybe_unused]] int n, F&& f) {
    f(ex.lid());
}
/// @brief On a `Fit` executor: calls `f(lid)` once if `lid < n`.
/// @ingroup api_device_blas1
/// @pre `n` is at most the executor's size (a `Fit` executor owns one index per item).
/// @tparam U ignored; present so that the call spells the same as the looping form.
/// @param ex the executor.
/// @param n the extent.
/// @param f called as `f(int i)`.
template <class U = alg::UnrollDefault, class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_owned(const Fit<Ex>& ex, int n, F&& f) {
    const int i = ex.lid();
    if (i >= 0 && i < n) f(i);
}
/// @brief On a `Fit` executor: calls `f(r.lo + lid)` once if it is below `r.hi`.
/// @ingroup api_device_blas1
/// @pre `r.hi - r.lo` is at most the executor's size.
/// @tparam U ignored, as for the `[0, n)` form.
/// @param ex the executor.
/// @param r the index range, in parent coordinates.
/// @param f called as `f(int i)`.
template <class U = alg::UnrollDefault, class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_owned(const Fit<Ex>& ex, Range r, F&& f) {
    const int i = r.lo + ex.lid();
    if (i < r.hi) f(i);
}
/// @brief On a `Grid1D` executor: calls `f(g.i)` for the launch's one element, if it is below `n`.
/// @ingroup api_device_blas1
/// @tparam U ignored, as for the `[0, n)` form.
/// @tparam E `true` (an exact grid) drops the `i < n` test.
/// @param g the one-dimensional grid executor.
/// @param n the extent.
/// @param f called as `f(int i)`.
template <class U = alg::UnrollDefault, bool E, class F> BATCHLAS_DEVICE_INLINE void for_each_owned(const Grid1D<E>& g, int n, F&& f) {
    if constexpr (E) f(g.i);
    else if (g.i < n) f(g.i);
}

/// @brief The team loop: calls `f(o, live)` for the outputs `o` that this item's team owns.
///
/// Team `t` of `nteams` owns `o = t, t + nteams, ...`; every lane of the team makes the same calls.
/// A one-item executor is one team. Use it when each output is a team collective (a team
/// reduction per output column, for example).
/// @ingroup api_device_blas1
/// @tparam Loop `alg::Lockstep` (default): every team runs `ceil(nout / nteams)` trips with
///   `live = o < nout`, so a team collective inside `f` is reached by every lane even when teams
///   share a sub-group. `alg::Diverge`: only the trips with `o < nout`, `live` always `true`.
/// @param ex the executor.
/// @param nout the number of outputs.
/// @param f called as `f(int o, bool live)`; when `live` is `false`, `f` must still reach its
///   team collectives but must not touch output `o`.
template <class Loop = alg::Lockstep, class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_team_owned(const Ex& ex, int nout, F&& f) {
    const int nt = detail::nteams_of(ex);
    const int t = is_thread_v<Ex> ? 0 : ex.team();
    if constexpr (std::is_same_v<Loop, alg::Lockstep>) {
        const int nb = (nout + nt - 1) / nt;
        for (int ob = 0; ob < nb; ++ob) {
            const int o = ob * nt + t;
            f(o, o < nout);
        }
    } else {
        static_assert(std::is_same_v<Loop, alg::Diverge>, "for_each_team_owned: alg::Lockstep or alg::Diverge");
        for (int o = t; o < nout; o += nt) f(o, true);
    }
}

/// @brief Calls `f(i0, j0, live_r, live_c)` for every `TR` x `TC` tile of an m x n grid that this
/// item owns.
///
/// Tiles are numbered in column-major tile order and tile `t` is owned by the item with
/// `lid == t mod size`. `(i0, j0)` is the tile's first element; `live_r = min(TR, m - i0)` and
/// `live_c = min(TC, n - j0)` clip the edge tiles.
/// @ingroup api_device_blas1
/// @tparam TR tile rows, `>= 1`.
/// @tparam TC tile columns, `>= 1`.
/// @param ex the executor.
/// @param m grid rows.
/// @param n grid columns.
/// @param f called as `f(int i0, int j0, int live_r, int live_c)`.
template <int TR, int TC, class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_owned_tiles(const Ex& ex, int m, int n, F&& f) {
    static_assert(TR >= 1 && TC >= 1, "for_each_owned_tiles: TR, TC >= 1");
    const int tr = (m + TR - 1) / TR, tc = (n + TC - 1) / TC;
    for_each_owned(ex, tr * tc, [&](int t) {
        const int bj = t / tr, bi = t - bj * tr;
        const int i0 = bi * TR, j0 = bj * TC;
        f(i0, j0, sycl::min(TR, m - i0), sycl::min(TC, n - j0));
    });
}

/// @brief Calls `f(r, c)` for every element of an m x n column-major grid that this item owns.
///
/// Ownership is flat: element `e = r + c * m` is owned by the item with `lid == e mod size`, so
/// consecutive items take consecutive rows and their loads coalesce. Every 2-D operation that uses
/// the same executor and extents composes with this one with no barrier. On `Thread` every
/// algorithm is the column-major double loop; on a `Grid2D` executor, the item's one element.
/// @ingroup api_device_blas1
/// @tparam Alg how the item computes `(r, c)` from `e`; every form visits the same elements:
///   - `alg::Walk` (default): one division per item, then `(r, c) += (size % m, size / m)` with
///     one carry per step.
///   - `alg::Stepped`: one division per item, then `r += size` and subtract `m` while `r >= m`.
///   - `alg::DivMod`: `(e % m, e / m)` per step.
///   - `alg::Flat`: one division per step with `e` of type `Idx`.
///   - `alg::RowOwner`: row ownership instead (row `i` by `lid == i mod size`), each row walked
///     serially over its columns.
/// @tparam Idx the index type of `alg::Flat` and `alg::DivMod` (`std::ptrdiff_t` for 64-bit index
///   arithmetic); the other forms ignore it.
/// @param ex the executor.
/// @param m grid rows.
/// @param n grid columns.
/// @param f called as `f(int r, int c)`.
template <class Alg = alg::Walk, class Idx = int, class Ex, class F>
BATCHLAS_DEVICE_INLINE void for_each_owned_2d(const Ex& ex, int m, int n, F&& f) {
    if constexpr (requires { ex.row(); ex.col(); }) {  // Grid2D (grid.hh)
        const int i = ex.row(), j = ex.col();
        if (i < m && j < n) f(i, j);
    } else if constexpr (is_thread_v<Ex>) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) f(i, j);
    } else if constexpr (std::is_same_v<Alg, alg::RowOwner>) {
        for_each_owned(ex, m, [&](int i) {
            for (int j = 0; j < n; ++j) f(i, j);
        });
    } else if constexpr (std::is_same_v<Alg, alg::Flat>) {
        const Idx total = static_cast<Idx>(m) * static_cast<Idx>(n);
        const Idx w = static_cast<Idx>(detail::size_of(ex));
        for (Idx e = static_cast<Idx>(ex.lid()); e < total; e += w) {
            const Idx q = e / static_cast<Idx>(m);
            f(static_cast<int>(e - q * static_cast<Idx>(m)), static_cast<int>(q));
        }
    } else if constexpr (std::is_same_v<Alg, alg::DivMod>) {
        const Idx total = static_cast<Idx>(m) * static_cast<Idx>(n);
        const Idx w = static_cast<Idx>(detail::size_of(ex));
        for (Idx e = static_cast<Idx>(ex.lid()); e < total; e += w)
            f(static_cast<int>(e % static_cast<Idx>(m)), static_cast<int>(e / static_cast<Idx>(m)));
    } else if constexpr (std::is_same_v<Alg, alg::Stepped>) {
        if (m <= 0) return;
        const int w = detail::size_of(ex);
        int c = ex.lid() / m;
        int r = ex.lid() - c * m;
        while (c < n) {
            f(r, c);
            r += w;
            while (r >= m) {
                r -= m;
                ++c;
            }
        }
    } else {
        static_assert(std::is_same_v<Alg, alg::Walk>, "for_each_owned_2d: alg::Flat, DivMod, Stepped, Walk or RowOwner");
        if (m <= 0) return;
        const int w = detail::size_of(ex);
        const int dc = w / m, dr = w - dc * m;
        int c = ex.lid() / m;
        int r = ex.lid() - c * m;
        while (c < n) {
            f(r, c);
            r += dr;
            c += dc;
            if (r >= m) {
                r -= m;
                ++c;
            }
        }
    }
}

namespace detail {
// (e % m, e / m) for e = start, start + step, ... with no division per step.
struct Walk2d {
    int r, c, dr, dc, m;
    BATCHLAS_DEVICE_INLINE Walk2d(int start, int step, int m_) : r(start % m_), c(start / m_), dr(step % m_), dc(step / m_), m(m_) {}
    BATCHLAS_DEVICE_INLINE void next() {
        r += dr;
        c += dc;
        if (r >= m) {
            r -= m;
            ++c;
        }
    }
};
}  // namespace detail
/// @brief `for_each_owned_2d`'s `alg::Walk` ownership, `U` positions per round, loads before uses.
///
/// Each round calls `load(live, r, c)` for `U` positions (so their loads overlap), then `use(e)` for
/// each of the `U` results in order. A slot past the grid still gets `load(false, r, c)` and
/// `use(e)`; `load` must return a harmless value for it.
/// @ingroup api_device_lowlevel
/// @tparam U positions per round.
/// @param ex the executor (needs `width()`).
/// @param rows grid rows.
/// @param cols grid columns.
/// @param load called as `load(bool live, int r, int c)`; returns the item passed to `use`.
/// @param use called as `use(item)`.
template <int U, class Ex, class Load, class Use>
BATCHLAS_DEVICE_INLINE void for_each_owned_2d_batched(const Ex& ex, int rows, int cols, Load&& load, Use&& use) {
    using Item = decltype(load(true, 0, 0));
    for (detail::Walk2d w(ex.lid(), ex.width(), rows); w.c < cols;) {
        Item e[U];
BATCHLAS_UNROLL_FULL
        for (int u = 0; u < U; ++u) {
            const bool live = w.c < cols;
            e[u] = load(live, w.r, w.c);
            if (live) w.next();
        }
BATCHLAS_UNROLL_FULL
        for (int u = 0; u < U; ++u) use(e[u]);
    }
}

/// @brief On an `ElemAt` executor: calls `f(ex.r, ex.c)` for the item's one element, if it is inside m x n.
/// @ingroup api_device_blas1
/// @tparam Alg ignored; present so that the call spells the same as the looping form.
/// @tparam Idx ignored.
/// @tparam E `true` (an exact grid) drops the extent test.
/// @param ex the one-element executor.
/// @param m grid rows.
/// @param n grid columns.
/// @param f called as `f(int r, int c)`.
template <class Alg = alg::Walk, class Idx = int, bool E, class F>
BATCHLAS_DEVICE_INLINE void for_each_owned_2d(const ElemAt<E>& ex, int m, int n, F&& f) {
    if constexpr (E) f(ex.r, ex.c);
    else if (ex.r < m && ex.c < n) f(ex.r, ex.c);
}
/// @brief On an `ElemIn` executor: calls `f(r - r0, c - c0)` for the item's one element, if it lies
/// inside the m x n block at `(r0, c0)`.
/// @ingroup api_device_blas1
/// @tparam Alg ignored.
/// @tparam Idx ignored.
/// @param ex the one-element executor with its block origin.
/// @param m block rows.
/// @param n block columns.
/// @param f called as `f(int r, int c)` in block coordinates.
template <class Alg = alg::Walk, class Idx = int, class F>
BATCHLAS_DEVICE_INLINE void for_each_owned_2d(const ElemIn& ex, int m, int n, F&& f) {
    if (ex.c >= ex.c0) {
        if (ex.r >= ex.r0 && ex.r < ex.r0 + m && ex.c < ex.c0 + n) f(ex.r - ex.r0, ex.c - ex.c0);
    }
}

/// @brief Elementwise 2-D map: `out(i, j) = f(ins(i, j)...)` over `out`'s extents.
///
/// Ownership is `for_each_owned_2d`'s, so `out` may alias an input.
/// @ingroup api_device_blas1
/// @param ex the executor (any cyclic executor or a `Grid2D`).
/// @param out the output matrix operand (`rows()`, `cols()`, assignable `out(i, j)`).
/// @param f called as `f(ins(i, j)...)`.
/// @param ins the input matrix operands, at least `out`'s extents.
template <class Ex, class Out, class F, class... In>
BATCHLAS_DEVICE_INLINE void map2d(const Ex& ex, const Out& out, F&& f, const In&... ins) {
    for_each_owned_2d(ex, out.rows(), out.cols(), [&](int i, int j) { out(i, j) = f(ins(i, j)...); });
}
/// @brief Elementwise map: `out(i) = f(ins(i)...)` for `i < out.size()`.
///
/// Ownership is `for_each_owned`'s, so `out` may alias an input.
/// @ingroup api_device_blas1
/// @param ex the executor.
/// @param out the output vector operand.
/// @param f called as `f(ins(i)...)`.
/// @param ins the input vector operands, at least `out.size()` long.
template <class Ex, class Out, class F, class... In>
BATCHLAS_DEVICE_INLINE void map(const Ex& ex, const Out& out, F&& f, const In&... ins) {
    for_each_owned(ex, out.size(), [&](int i) { out(i) = f(ins(i)...); });
}

/// @brief `y := a x + b y` (BLAS axpby).
///
/// A `Zero` tag for `b` never reads `y`; with any other `b`, `y` is read where it is written.
/// Elementwise over `y.size()`, no barrier.
/// @ingroup api_device_blas1
/// @param ex the executor.
/// @param a_ the scalar `a`: a value or a tag (`One`, `Zero`, `NegOne`, `Scalar<T>`).
/// @param x the input vector, at least `y.size()` long.
/// @param b_ the scalar `b`: a value or a tag.
/// @param[in,out] y the output vector.
template <class Ex, class A, class X, class B, class Y>
BATCHLAS_DEVICE_INLINE void axpby(const Ex& ex, A a_, const X& x, B b_, const Y& y) {
    using T = elem_t<Y>;
    const auto a = as_tag(a_);
    const auto b = as_tag(b_);
    for_each_owned(ex, y.size(), [&](int i) {
        if constexpr (std::is_same_v<tag_t<B>, Zero>) y(i) = tag_mul(a, static_cast<T>(x(i)));
        else y(i) = Arith<T>::add(tag_mul(a, static_cast<T>(x(i))), tag_mul(b, static_cast<T>(y(i))));
    });
}

/// @brief `y := a x + y` (BLAS axpy).
///
/// A `Zero` tag returns at once; `One` and `NegOne` add or subtract without a multiply.
/// Elementwise over `y.size()`, no barrier.
/// @ingroup api_device_blas1
/// @tparam Ops the arithmetic policy (`arith::Explicit` default, `arith::Native`, `arith::Fma`).
/// @param ex the executor.
/// @param a_ the scalar `a`: a value or a tag.
/// @param x the input vector, at least `y.size()` long.
/// @param[in,out] y the output vector.
template <class Ops = arith::Explicit, class Ex, class A, class X, class Y>
BATCHLAS_DEVICE_INLINE void axpy(const Ex& ex, A a_, const X& x, const Y& y) {
    using T = elem_t<Y>;
    using Ar = arith_t<Ops, T>;
    using S = tag_t<A>;
    const S a = as_tag(a_);
    if constexpr (std::is_same_v<S, Zero>) return;
    for_each_owned(ex, y.size(), [&](int i) {
        if constexpr (std::is_same_v<S, One>) y(i) = Ar::add(y(i), x(i));
        else if constexpr (std::is_same_v<S, NegOne>) y(i) = Ar::sub(y(i), x(i));
        else if constexpr (!std::is_same_v<S, Zero>) Ar::mac(y(i), static_cast<T>(a.v), x(i));
    });
}

/// @brief `x := a x` (BLAS scal).
///
/// Elementwise over `x.size()`, no barrier. With `arith::Native` and a value `a`, each element is
/// `x(i) *= a` in the element type's own operator.
/// @ingroup api_device_blas1
/// @tparam Ops the arithmetic policy (`arith::Explicit` default, `arith::Native`).
/// @param ex the executor.
/// @param a_ the scalar `a`: a value or a tag.
/// @param[in,out] x the vector.
template <class Ops = arith::Explicit, class Ex, class A, class X> BATCHLAS_DEVICE_INLINE void scal(const Ex& ex, A a_, const X& x) {
    using T = elem_t<X>;
    using S = tag_t<A>;
    const S a = as_tag(a_);
    constexpr bool native_value = std::is_same_v<Ops, arith::Native> && !std::is_same_v<S, One> &&
                                  !std::is_same_v<S, Zero> && !std::is_same_v<S, NegOne>;
    for_each_owned(ex, x.size(), [&](int i) {
        if constexpr (native_value) x(i) *= static_cast<T>(a.v);
        else x(i) = tag_mul(a, static_cast<T>(x(i)));
    });
}
/// @brief `x := a x` on a lane vector: only the items in the window `[x.lo, x.hi)` scale their element.
/// @ingroup api_device_blas1
/// @tparam Ops the arithmetic policy, as for the memory form.
/// @param ex the executor.
/// @param a_ the scalar `a`: a value or a tag.
/// @param[in,out] x the lane vector (one element per item).
template <class Ops = arith::Explicit, class Ex, class A, class Scalar> BATCHLAS_DEVICE_INLINE void scal(const Ex& ex, A a_, const LaneVec<Scalar>& x) {
    using V = std::remove_cv_t<Scalar>;
    using S = tag_t<A>;
    const S a = as_tag(a_);
    constexpr bool native_value = std::is_same_v<Ops, arith::Native> && !std::is_same_v<S, One> &&
                                  !std::is_same_v<S, Zero> && !std::is_same_v<S, NegOne>;
    if constexpr (native_value) {
        if (x.owns(ex)) x.v *= a.v;
    } else {
        if (x.owns(ex)) x.v = tag_mul(a, static_cast<V>(x.v));
    }
}

/// @brief `y := x` (BLAS copy) over `y.size()` elements; elementwise, no barrier.
/// @ingroup api_device_blas1
/// @param ex the executor.
/// @param x the source vector, at least `y.size()` long.
/// @param[out] y the destination vector.
template <class Ex, class X, class Y> BATCHLAS_DEVICE_INLINE void copy(const Ex& ex, const X& x, const Y& y) {
    for_each_owned(ex, y.size(), [&](int i) { y(i) = x(i); });
}

/// @brief Exchanges `x` and `y` (BLAS swap) over `x.size()` elements; elementwise, no barrier.
/// @ingroup api_device_blas1
/// @param ex the executor.
/// @param[in,out] x the first vector.
/// @param[in,out] y the second vector, at least `x.size()` long.
template <class Ex, class X, class Y> BATCHLAS_DEVICE_INLINE void swap(const Ex& ex, const X& x, const Y& y) {
    for_each_owned(ex, x.size(), [&](int i) {
        const auto t = x(i);
        x(i) = y(i);
        y(i) = t;
    });
}

// ---- reductions ----------------------------------------------------------------------------
// Scope::Sub: shuffles inside one team (sub-group / partition), 0 BAR, 0 SLM.
// Scope::Group: the whole executor; on a WorkGroup without Scratch this is
// portable::reduce_over_group (the tree the legacy kernels use, so ports stay bit-identical).
namespace detail {
template <Scope S, class Ex> constexpr Scope resolve_scope() { return S == Scope::Own ? Ex::natural_scope : S; }

template <Scope S, class Ex, class T, class Op = sycl::plus<T>>
BATCHLAS_DEVICE_INLINE T scope_reduce(const Ex& ex, T v, Op op = {}) {
    if constexpr (resolve_scope<S, Ex>() == Scope::Sub) {
        static_assert(Ex::sub_whole, "reduce: Scope::Sub on a WorkGroup leaves one partial per sub-group; use Scope::Group");
        return ex.team_reduce(v, op);
    } else {
        return ex.group_reduce(v, op);
    }
}

// One scope-S reduction per element of v, by pack expansion: never a `#pragma unroll` loop
// around a collective (it leaks into the collective's own loops: stedc allreduce 32x vs 8x
// unroll, +40%; bdsqr LDS 22 -> 52).
template <Scope S, class Ex, class T, std::size_t K, class Op, std::size_t... I>
BATCHLAS_DEVICE_INLINE void reduce_each(const Ex& ex, std::array<T, K>& v, Op op, std::index_sequence<I...>) {
    ((v[I] = scope_reduce<S>(ex, v[I], op)), ...);
}
template <Scope S, class Ex, class T, std::size_t K, class Op = sycl::plus<T>>
BATCHLAS_DEVICE_INLINE void reduce_each(const Ex& ex, std::array<T, K>& v, Op op = {}) {
    reduce_each<S>(ex, v, op, std::make_index_sequence<K>{});
}
template <class Ex, class T, std::size_t K, class Op, std::size_t... I>
BATCHLAS_DEVICE_INLINE void team_reduce_each(const Ex& ex, std::array<T, K>& v, Op op, std::index_sequence<I...>) {
    ((v[I] = ex.team_reduce(v[I], op)), ...);
}

// Sub-group-size contract of the packed exchange: it assumes nteams = WG / SG hardware
// sub-groups of SG lanes (team = lid / SG, leader = lane 0). NVPTX always runs SG = 32, so the
// assumption is a compile-time fact there. Elsewhere (the CPU device, AdaptiveCpp, which may
// ignore reqd_sub_group_size) the kernel checks the launch's uniform maximum sub-group size
// and, on a mismatch, reduces each value over the work-group instead (portable, no Scratch).
template <class Ex> BATCHLAS_DEVICE_INLINE bool packed_contract_holds(const Ex& ex) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    if constexpr (Ex::sg_size == 32) return true;
#endif
    const auto sg = ex.sub();
    return static_cast<int>(sg.get_max_local_range()[0]) == Ex::sg_size &&
           static_cast<int>(sg.get_group_linear_range()) == Ex::nteams;
}

// K reductions (op; sycl::plus<T> sums with Arith<T>::add) from ONE scratch exchange: per-team
// butterfly, lane 0 writes K values, one barrier, every item reads nteams*K in a fixed order
// (uniform result), one trailing barrier so the scratch may be reused (trailing_barrier = false
// skips it: the next write to the scratch is behind another barrier anyway).
template <int K, class Ex, class T, int Slots, class Op = sycl::plus<T>>
BATCHLAS_DEVICE_INLINE std::array<T, K> packed_reduce(const Ex& ex, Scratch<T, Slots> scr, std::array<T, K> v, Op op = {},
                                                      bool trailing_barrier = true) {
    if constexpr (is_runtime_v<Ex>) {
        reduce_each<Scope::Group>(ex, v, op);
        return v;
    } else if constexpr (!is_work_group_v<Ex> || Ex::nteams == 1) {
        team_reduce_each(ex, v, op, std::make_index_sequence<K>{});
        return v;
    } else {
        static_assert(Slots >= Ex::nteams * K, "allreduce: Scratch smaller than scratch_elements<reduce_op::ReduceN<K>, Ex, T>()");
        if (!packed_contract_holds(ex)) {
            reduce_each<Scope::Group>(ex, v, op);
            return v;
        }
        team_reduce_each(ex, v, op, std::make_index_sequence<K>{});
        if (ex.lane() == 0) {
#pragma unroll
            for (int k = 0; k < K; ++k) scr.p[ex.team() * K + k] = v[k];
        }
        ex.barrier(Mem::Local);
        std::array<T, K> out;
#pragma unroll
        for (int k = 0; k < K; ++k) out[k] = scr.p[k];
#pragma unroll 4  // a full unroll hoists nteams*K loads into registers (WG 1024: > 64 regs)
        for (int s = 1; s < Ex::nteams; ++s) {
#pragma unroll
            for (int k = 0; k < K; ++k) {
                if constexpr (std::is_same_v<Op, sycl::plus<T>>) out[k] = Arith<T>::add(out[k], scr.p[s * K + k]);
                else out[k] = op(out[k], scr.p[s * K + k]);
            }
        }
        if (trailing_barrier) ex.barrier(Mem::Local);
        return out;
    }
}
}  // namespace detail

/// @brief The sum of `f(i)` over `i < n`, reduced over scope `S`; the result is uniform.
///
/// Each item sums its owned terms in ascending order, then one reduction combines the partials.
/// COLLECTIVE: every item of the scope must call it.
/// @ingroup api_device_blas1
/// @tparam S the reduction scope: `Scope::Own` (default; the executor's natural scope),
///   `Scope::Sub` (one team, shuffles only, no barrier; not on a `WorkGroup`, which has one partial
///   per sub-group) or `Scope::Group` (the whole executor).
/// @param ex the executor.
/// @param n the number of terms.
/// @param f called as `f(int i)`; its return type is the result type.
/// @return the sum, the same on every item.
template <Scope S = Scope::Own, class Ex, class F> BATCHLAS_DEVICE_INLINE auto reduce(const Ex& ex, int n, F&& f) {
    using T = std::remove_cvref_t<decltype(f(0))>;
    T acc = Arith<T>::zero();
    for_each_owned(ex, n, [&](int i) { acc = Arith<T>::add(acc, f(i)); });
    return detail::scope_reduce<S>(ex, acc);
}
/// @brief The sum of `f(i)` over `i < n` over the whole executor, through one packed `Scratch` exchange.
///
/// On a `WorkGroup` with several sub-groups: one team reduction, lane 0 of each team writes its
/// partial to the scratch, a local barrier, every item sums the partials in team order, and a
/// trailing barrier so the scratch can be reused. Other executors reduce in their team and ignore
/// the scratch. COLLECTIVE.
/// @ingroup api_device_blas1
/// @param ex the executor.
/// @param scr local scratch of at least `scratch_elements<reduce_op::ReduceN<1>, Ex, T>()` slots.
/// @param n the number of terms.
/// @param f called as `f(int i)`, returning `T`.
/// @return the sum, the same on every item.
template <class Ex, class T, int Slots, class F>
BATCHLAS_DEVICE_INLINE T reduce(const Ex& ex, Scratch<T, Slots> scr, int n, F&& f) {
    T acc = Arith<T>::zero();
    for_each_owned(ex, n, [&](int i) { acc = Arith<T>::add(acc, f(i)); });
    return detail::packed_reduce<1>(ex, scr, std::array<T, 1>{acc})[0];
}

namespace detail {
// The item's partial sum (TR: Trans x^T y, ConjTrans x^H y). On Fit<Ex> the item's one product
// (0 past x.size()): no accumulator, so no +0 seed to add (the sytd2 form).
template <Transpose TR, class Ops = arith::Explicit, class Ex, class X, class Y>
BATCHLAS_DEVICE_INLINE auto dot_partial(const Ex& ex, const X& x, const Y& y) {
    static_assert(TR != Transpose::NoTrans, "dot_partial: TR is Trans or ConjTrans");
    constexpr bool conj = TR == Transpose::ConjTrans;
    using T = elem_t<X>;
    using Ar = arith_t<Ops, T>;
    if constexpr (is_fit_v<Ex>) {
        const int i = ex.lid();
        const bool in = i >= 0 && i < x.size();
        T p;
        if constexpr (conj) p = in ? Ar::mul(Ar::conj(x(i)), y(i)) : Ar::zero();
        else p = in ? Ar::mul(x(i), y(i)) : Ar::zero();
        return p;
    } else {
        T acc = Ar::zero();
        for_each_owned(ex, x.size(), [&](int i) {
            if constexpr (conj) Ar::mac_conj(acc, x(i), y(i));
            else Ar::mac(acc, x(i), y(i));
        });
        return acc;
    }
}
}  // namespace detail

/// @brief `x^T y` (BLAS dotu: no conjugation) over `x.size()` elements, reduced over scope `S`.
///
/// COLLECTIVE; the result is uniform over the scope.
/// @ingroup api_device_blas1
/// @tparam S the reduction scope (`Scope::Own` default, `Scope::Sub`, `Scope::Group`), as for `reduce`.
/// @tparam Ops the arithmetic policy of the products.
/// @param ex the executor.
/// @param x the first vector.
/// @param y the second vector, at least `x.size()` long.
/// @return the dot product, the same on every item.
template <Scope S = Scope::Own, class Ops = arith::Explicit, class Ex, class X, class Y>
BATCHLAS_DEVICE_INLINE auto dot(const Ex& ex, const X& x, const Y& y) {
    return detail::scope_reduce<S>(ex, detail::dot_partial<Transpose::Trans, Ops>(ex, x, y));
}
/// @brief `x^H y` (BLAS dotc: `x` conjugated) over `x.size()` elements, reduced over scope `S`.
///
/// For a real element type this is `dot`. COLLECTIVE; the result is uniform over the scope.
/// @ingroup api_device_blas1
/// @tparam S the reduction scope, as for `reduce`.
/// @tparam Ops the arithmetic policy of the products.
/// @param ex the executor.
/// @param x the conjugated vector.
/// @param y the second vector, at least `x.size()` long.
/// @return the dot product, the same on every item.
template <Scope S = Scope::Own, class Ops = arith::Explicit, class Ex, class X, class Y>
BATCHLAS_DEVICE_INLINE auto dotc(const Ex& ex, const X& x, const Y& y) {
    return detail::scope_reduce<S>(ex, detail::dot_partial<Transpose::ConjTrans, Ops>(ex, x, y));
}

namespace detail {
template <int K, Transpose TR, class Ex, class Xs, class Y> BATCHLAS_DEVICE_INLINE auto dotn_partial(const Ex& ex, const Xs& xs, const Y& y) {
    static_assert(TR != Transpose::NoTrans, "dotN: TR is Trans or ConjTrans");
    using T = elem_t<Y>;
    std::array<T, K> acc;
#pragma unroll
    for (int k = 0; k < K; ++k) acc[k] = Arith<T>::zero();
    for_each_owned(ex, y.size(), [&](int i) {
        const T yi = y(i);
#pragma unroll
        for (int k = 0; k < K; ++k) {
            if constexpr (TR == Transpose::ConjTrans) Arith<T>::mac_conj(acc[k], xs[k](i), yi);
            else Arith<T>::mac(acc[k], xs[k](i), yi);
        }
    });
    return acc;
}
}  // namespace detail

/// @brief `K` dot products `op(xs[k]) y` for `k < K` in one loop over `y` and one packed `Scratch` exchange.
///
/// On a `WorkGroup` with several sub-groups the `K` sums share one scratch exchange (two barriers);
/// other executors reduce in their team and ignore the scratch. COLLECTIVE; uniform results.
/// @ingroup api_device_blas1
/// @tparam K the number of dot products.
/// @tparam TR `Transpose::Trans` (default, `xs[k]^T y`) or `Transpose::ConjTrans` (`xs[k]^H y`).
/// @param ex the executor.
/// @param scr local scratch of at least `scratch_elements<reduce_op::ReduceN<K>, Ex, Scalar>()` slots.
/// @param xs `K` vectors of one type, each at least `y.size()` long.
/// @param y the shared vector.
/// @return the `K` dot products, the same on every item.
template <int K, Transpose TR = Transpose::Trans, class Ex, class Scalar, int Slots, class X, class Y>
BATCHLAS_DEVICE_INLINE std::array<Scalar, K> dotN(const Ex& ex, Scratch<Scalar, Slots> scr, const std::array<X, K>& xs, const Y& y) {
    return detail::packed_reduce<K>(ex, scr, detail::dotn_partial<K, TR>(ex, xs, y));
}
namespace detail {
template <Transpose TR, class Ops, class Ex, class Y, class... Xs> BATCHLAS_DEVICE_INLINE auto dot_many_partial(const Ex& ex, const Y& y, const Xs&... xs) {
    constexpr bool conj = TR == Transpose::ConjTrans;
    using T = elem_t<Y>;
    using Ar = arith_t<Ops, T>;
    constexpr int K = sizeof...(Xs);
    std::array<T, K> acc;
#pragma unroll
    for (int k = 0; k < K; ++k) acc[k] = Ar::zero();
    for_each_owned(ex, y.size(), [&](int i) {
        const T yi = y(i);
        int k = 0;
        ((conj ? Ar::mac_conj(acc[k++], xs(i), yi) : Ar::mac(acc[k++], xs(i), yi)), ...);
    });
    return acc;
}
}  // namespace detail

/// @brief One dot product `op(x) y` per operand in `xs`, which may have different types, in one
/// loop over `y` and one packed `Scratch` exchange.
///
/// For example a column and its `unit_head` view. COLLECTIVE; uniform results.
/// @ingroup api_device_blas1
/// @tparam TR `Transpose::Trans` (default, `x^T y`) or `Transpose::ConjTrans` (`x^H y`).
/// @tparam Ops the arithmetic policy of the products.
/// @param ex the executor.
/// @param scr local scratch of at least `scratch_elements<reduce_op::ReduceN<K>, Ex, Scalar>()`
///   slots, `K = sizeof...(Xs)`.
/// @param y the shared vector.
/// @param xs the vectors, each at least `y.size()` long.
/// @return the dot products in argument order, the same on every item.
template <Transpose TR = Transpose::Trans, class Ops = arith::Explicit, class Ex, class Scalar, int Slots, class Y, class... Xs>
BATCHLAS_DEVICE_INLINE std::array<Scalar, sizeof...(Xs)> dot_many(const Ex& ex, Scratch<Scalar, Slots> scr, const Y& y, const Xs&... xs) {
    static_assert(TR != Transpose::NoTrans, "dot_many: TR is Trans or ConjTrans");
    return detail::packed_reduce<sizeof...(Xs)>(ex, scr, detail::dot_many_partial<TR, Ops>(ex, y, xs...));
}
// On a WorkGroup each reduction is portable::reduce_over_group: the legacy kernels' trees, bit-identical.
/// @brief One dot product `op(x) y` per operand in `xs`, in one loop over `y`, then one scope-`S`
/// reduction per result in argument order (no scratch).
///
/// COLLECTIVE; uniform results.
/// @ingroup api_device_blas1
/// @tparam TR `Transpose::Trans` (default) or `Transpose::ConjTrans`.
/// @tparam S the reduction scope, as for `reduce`.
/// @tparam Ops the arithmetic policy of the products.
/// @param ex the executor.
/// @param y the shared vector.
/// @param xs the vectors, each at least `y.size()` long.
/// @return a `std::array` of the dot products in argument order.
template <Transpose TR = Transpose::Trans, Scope S = Scope::Own, class Ops = arith::Explicit, class Ex, class Y, class... Xs>
BATCHLAS_DEVICE_INLINE auto dot_many(const Ex& ex, const Y& y, const Xs&... xs) {
    static_assert(TR != Transpose::NoTrans, "dot_many: TR is Trans or ConjTrans");
    auto acc = detail::dot_many_partial<TR, Ops>(ex, y, xs...);
    detail::reduce_each<S>(ex, acc);
    return acc;
}

/// @brief `K` dot products `op(xs[k]) y` in one loop over `y`, then `K` independent scope-`S` reductions (no scratch).
///
/// On a sub-group or a partition the reductions are shuffles only. COLLECTIVE; uniform results.
/// @ingroup api_device_blas1
/// @tparam K the number of dot products.
/// @tparam TR `Transpose::Trans` (default) or `Transpose::ConjTrans`.
/// @tparam S the reduction scope, as for `reduce`.
/// @param ex the executor.
/// @param xs `K` vectors of one type, each at least `y.size()` long.
/// @param y the shared vector.
/// @return a `std::array` of the `K` dot products.
template <int K, Transpose TR = Transpose::Trans, Scope S = Scope::Own, class Ex, class X, class Y>
BATCHLAS_DEVICE_INLINE auto dotN(const Ex& ex, const std::array<X, K>& xs, const Y& y) {
    auto acc = detail::dotn_partial<K, TR>(ex, xs, y);
    detail::reduce_each<S>(ex, acc);
    return acc;
}

/// @brief The sum of `|re(x_i)| + |im(x_i)|` (BLAS asum) over `x.size()` elements, reduced over scope `S`.
///
/// COLLECTIVE; the result is real and uniform.
/// @ingroup api_device_blas1
/// @tparam S the reduction scope, as for `reduce`.
/// @param ex the executor.
/// @param x the vector.
/// @return the sum, of the real type of `x`'s elements.
template <Scope S = Scope::Own, class Ex, class X> BATCHLAS_DEVICE_INLINE auto asum(const Ex& ex, const X& x) {
    using R = real_t<elem_t<X>>;
    R acc = R(0);
    for_each_owned(ex, x.size(), [&](int i) { acc += abs1(x(i)); });
    return detail::scope_reduce<S>(ex, acc);
}

/// @brief The maximum of `|re(x_i)| + |im(x_i)|` over `x.size()` elements, reduced over scope `S`.
///
/// Any NaN element makes the result NaN. An empty `x` gives 0. COLLECTIVE; uniform result. For
/// another key or NaN policy see `amax_by`; for the index see `iamax`.
/// @ingroup api_device_blas1
/// @tparam S the reduction scope, as for `reduce`.
/// @param ex the executor.
/// @param x the vector.
/// @return the maximum, of the real type of `x`'s elements.
template <Scope S = Scope::Own, class Ex, class X> BATCHLAS_DEVICE_INLINE auto amax(const Ex& ex, const X& x) {
    using R = real_t<elem_t<X>>;
    R m = R(0);
    int nan = 0;
    for_each_owned(ex, x.size(), [&](int i) {
        const R a = abs1(x(i));
        if (sycl::isnan(a)) nan = 1;
        else m = a > m ? a : m;
    });
    m = detail::scope_reduce<S>(ex, m, sycl::maximum<R>());
    nan = detail::scope_reduce<S>(ex, nan, sycl::maximum<int>());
    return nan ? std::numeric_limits<R>::quiet_NaN() : m;
}

/// @brief LAPACK lassq without the incoming pair: `scale` and `ssq` with `scale^2 * ssq = sum |x_i|^2`.
///
/// `scale` is the largest real or imaginary component in absolute value (NaN if any is NaN). Two
/// passes, a maximum then the sum of `(x_i / scale)^2`, so no square overflows or underflows to
/// zero. Two reductions over scope `S`. COLLECTIVE; uniform result.
/// @ingroup api_device_blas1
/// @tparam S the reduction scope, as for `reduce`.
/// @param ex the executor.
/// @param x the vector.
/// @return a `ScaledSsq` `{scale, ssq}`; an all-zero `x` gives `{0, 0}`.
template <Scope S = Scope::Own, class Ex, class X> BATCHLAS_DEVICE_INLINE auto lassq(const Ex& ex, const X& x) {
    using T = elem_t<X>;
    using A = Arith<T>;
    using R = real_t<T>;
    R sc = R(0);
    auto upd = [&](R a) { sc = (a > sc || sycl::isnan(a)) ? a : sc; };
    for_each_owned(ex, x.size(), [&](int i) {
        const T v = x(i);
        upd(sycl::fabs(A::re(v)));
        if constexpr (A::is_complex) upd(sycl::fabs(A::im(v)));
    });
    const R scale = detail::scope_reduce<S>(ex, sc, sycl::maximum<R>());
    const bool finite_pos = scale > R(0) && !sycl::isinf(scale);
    const R inv = finite_pos ? R(1) / scale : R(1);  // one divide; |x_i| * inv <= 1 (+1 ulp)
    R ssq = R(0);
    auto acc = [&](R a) {
        const R t = a * inv;
        ssq += t * t;
    };
    for_each_owned(ex, x.size(), [&](int i) {
        const T v = x(i);
        acc(A::re(v));
        if constexpr (A::is_complex) acc(A::im(v));
    });
    return ScaledSsq<R>{scale, detail::scope_reduce<S>(ex, ssq)};
}

// ||x||_2 by Blue's three accumulators and constants (dnrm2.f90; a complex x accumulates |re|
// and |im| separately, as dznrm2): overflow- and underflow-safe in one pass. The three sums
// leave in one exchange: a sycl::vec<R, 3> group reduction on a work-group (stedc's
// detail::nrm2), three butterflies on a team executor.
namespace detail {
template <class Real> struct BlueConst {
    static constexpr int radix = std::numeric_limits<Real>::radix, p = std::numeric_limits<Real>::digits;
    static constexpr int emin = std::numeric_limits<Real>::min_exponent, emax = std::numeric_limits<Real>::max_exponent;
    static constexpr int fdiv(int a, int b) { return a / b - (((a ^ b) < 0) && (a % b) ? 1 : 0); }
    static constexpr int cdiv(int a, int b) { return a / b + (((a ^ b) >= 0) && (a % b) ? 1 : 0); }
    static constexpr Real pw(int e) {
        Real r = Real(1);
        for (int i = 0; i < (e < 0 ? -e : e); ++i) r *= Real(radix);
        return e < 0 ? Real(1) / r : r;
    }
    static constexpr Real tsml = pw(cdiv(emin - 1, 2)), tbig = pw(fdiv(emax - p + 1, 2));
    static constexpr Real ssml = pw(-fdiv(emin - p, 2)), sbig = pw(-cdiv(emax + p - 1, 2));
};
template <class Real> BATCHLAS_DEVICE_INLINE void blue_acc(Real ax, Real& abig, Real& amed, Real& asml) {
    using C = BlueConst<Real>;
    if (ax > C::tbig) {
        const Real t = ax * C::sbig;
        abig += t * t;
    } else if (ax < C::tsml) {
        const Real t = ax * C::ssml;
        asml += t * t;
    } else {
        amed += ax * ax;
    }
}
template <class Real> BATCHLAS_DEVICE_INLINE Real blue_finish(Real abig, Real amed, Real asml) {
    using C = BlueConst<Real>;
    const Real maxN = std::numeric_limits<Real>::max();
    Real scl = Real(1), ss = Real(0);
    if (abig > Real(0)) {
        if ((amed > Real(0)) || (amed > maxN) || (amed != amed)) abig += (amed * C::sbig) * C::sbig;
        scl = Real(1) / C::sbig;
        ss = abig;
    } else if (asml > Real(0)) {
        if ((amed > Real(0)) || (amed > maxN) || (amed != amed)) {
            const Real am = sycl::sqrt(amed), as = sycl::sqrt(asml) / C::ssml;
            const Real ymin = sycl::fmin(am, as), ymax = sycl::fmax(am, as);
            ss = ymax * ymax * (Real(1) + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = Real(1) / C::ssml;
            ss = asml;
        }
    } else {
        ss = amed;
    }
    return scl * sycl::sqrt(ss);
}
}  // namespace detail

/// @brief The Euclidean norm `||x||_2` (BLAS nrm2), with the algorithm named by the caller.
///
/// There is no default algorithm: `alg::Blue` is overflow-safe in one pass and one exchange but
/// costs more than `alg::Unscaled`. A complex `x` accumulates `|re|` and `|im|` separately.
/// COLLECTIVE; uniform result.
/// @ingroup api_device_blas1
/// @tparam Alg one of
///   - `alg::Unscaled`: `sqrt(re(x^H x))`, one reduction; overflows past `sqrt(max)` and
///     underflows below `sqrt(min)`.
///   - `alg::Blue`: Blue's three accumulators (reference `dnrm2.f90`), one exchange; no overflow
///     or underflow.
///   - `alg::Scaled`: the two-pass `lassq` (maximum, then the scaled sum), two exchanges.
/// @tparam S the reduction scope, as for `reduce`.
/// @param ex the executor.
/// @param x the vector.
/// @return the norm, of the real type of `x`'s elements.
template <class Alg, Scope S = Scope::Own, class Ex, class X> BATCHLAS_DEVICE_INLINE auto nrm2(const Ex& ex, const X& x) {
    using T = elem_t<X>;
    using R = real_t<T>;
    if constexpr (std::is_same_v<Alg, alg::Unscaled>) {
        return static_cast<R>(sycl::sqrt(Arith<T>::re(dotc<S>(ex, x, x))));
    } else if constexpr (std::is_same_v<Alg, alg::Blue>) {
        using A = Arith<T>;
        R abig = R(0), amed = R(0), asml = R(0);
        for_each_owned(ex, x.size(), [&](int i) {
            const T v = x(i);
            detail::blue_acc(sycl::fabs(A::re(v)), abig, amed, asml);
            if constexpr (A::is_complex) detail::blue_acc(sycl::fabs(A::im(v)), abig, amed, asml);
        });
        if constexpr (detail::resolve_scope<S, Ex>() == Scope::Group && !Ex::sub_whole) {
            const sycl::vec<R, 3> t = batchlas::portable::reduce_over_group(ex.it.get_group(), sycl::vec<R, 3>{abig, amed, asml},
                                                                             sycl::plus<sycl::vec<R, 3>>());
            return detail::blue_finish<R>(t[0], t[1], t[2]);
        } else {
            std::array<R, 3> t{abig, amed, asml};
            detail::reduce_each<S>(ex, t);
            return detail::blue_finish<R>(t[0], t[1], t[2]);
        }
    } else {
        static_assert(std::is_same_v<Alg, alg::Scaled>, "nrm2: Alg is alg::Unscaled, alg::Blue or alg::Scaled");
        const auto q = lassq<S>(ex, x);
        if (sycl::isinf(q.scale)) return q.scale;
        return q.scale > R(0) ? q.scale * sycl::sqrt(q.ssq) : sycl::sqrt(q.ssq);
    }
}

// ---- reductions ------------------------------------------------------------------------------

/// @brief Each item's `v`, reduced with `op` over scope `S`; the result is uniform.
///
/// COLLECTIVE: every item of the scope must call it with the same `op`.
/// @ingroup api_device_blas1
/// @tparam S the reduction scope, as for `reduce`.
/// @param ex the executor.
/// @param v this item's value.
/// @param op an associative binary operation (default: `sycl::plus<T>`, a sum).
/// @return the reduction, the same on every item of the scope.
template <Scope S = Scope::Own, class Ex, class T, class Op = sycl::plus<T>>
BATCHLAS_DEVICE_INLINE T allreduce(const Ex& ex, T v, Op op = {}) {
    return detail::scope_reduce<S>(ex, v, op);
}
/// @brief Each element of `v` reduced with `op` over scope `S`, one reduction per element.
///
/// COLLECTIVE; uniform results.
/// @ingroup api_device_blas1
/// @tparam S the reduction scope, as for `reduce`.
/// @param ex the executor.
/// @param v this item's `K` values.
/// @param op an associative binary operation (default: a sum).
/// @return the `K` reductions.
template <Scope S = Scope::Own, class Ex, class T, std::size_t K, class Op = sycl::plus<T>>
BATCHLAS_DEVICE_INLINE std::array<T, K> allreduce(const Ex& ex, std::array<T, K> v, Op op = {}) {
    detail::reduce_each<S>(ex, v, op);
    return v;
}
/// @brief `K` values reduced with `op` over every item of the executor in ONE packed `Scratch` exchange.
///
/// On a `WorkGroup` with several sub-groups: a team reduction per value, lane 0 of each team
/// writes its `K` partials, one local barrier, every item combines the partials in team order (a
/// uniform result), then a trailing barrier so the scratch can be reused at once. Sub-scope
/// executors (`SubGroup`, `Partition`, `Teams`) and `Thread` reduce in their team and ignore the
/// scratch. COLLECTIVE.
/// @ingroup api_device_blas1
/// @param ex the executor.
/// @param scr local scratch of at least `scratch_elements<reduce_op::ReduceN<K>, Ex, T>()` slots.
/// @param v this item's `K` values.
/// @param op an associative binary operation (default: a sum).
/// @param trailing_barrier `false` skips the trailing barrier, when the next write to the scratch
///   is behind another barrier anyway.
/// @return the `K` reductions, the same on every item.
template <class Ex, class T, int Slots, std::size_t K, class Op = sycl::plus<T>>
BATCHLAS_DEVICE_INLINE std::array<T, K> allreduce(const Ex& ex, Scratch<T, Slots> scr, std::array<T, K> v, Op op = {},
                                                  bool trailing_barrier = true) {
    return detail::packed_reduce<int(K)>(ex, scr, v, op, trailing_barrier);
}
/// @brief One value reduced with `op` over every item of the executor through a `Scratch` exchange.
///
/// The `K = 1` case of the array form. COLLECTIVE.
/// @ingroup api_device_blas1
/// @param ex the executor.
/// @param scr local scratch of at least `scratch_elements<reduce_op::ReduceN<1>, Ex, T>()` slots.
/// @param v this item's value.
/// @param op an associative binary operation (default: a sum).
/// @param trailing_barrier `false` skips the trailing barrier.
/// @return the reduction, the same on every item.
template <class Ex, class T, int Slots, class Op = sycl::plus<T>>
BATCHLAS_DEVICE_INLINE T allreduce(const Ex& ex, Scratch<T, Slots> scr, T v, Op op = {}, bool trailing_barrier = true) {
    return detail::packed_reduce<1>(ex, scr, std::array<T, 1>{v}, op, trailing_barrier)[0];
}

// sum |x_i|^2 for one operand (-> R) or K operands of equal size in one loop (-> std::array<R, K>,
// one reduction each). Alg: alg::Fused accumulates re^2 then im^2 into the running sum (each a
// contractible a*a + acc: 2 FMA per complex element); alg::Unfused adds abs2(x) (its own
// rounding: FMUL + FFMA + FADD), the spelling of the hand kernels that call abs2.
namespace detail {
template <class Alg, class Real, class Scalar> BATCHLAS_DEVICE_INLINE void acc_abs2(Real& acc, const Scalar& x) {
    using A = Arith<std::remove_cv_t<Scalar>>;
    if constexpr (std::is_same_v<Alg, alg::Unfused>) {
        acc += abs2(x);
    } else {
        static_assert(std::is_same_v<Alg, alg::Fused>, "sumsq: alg::Fused or alg::Unfused");
        const Real re = A::re(x);
        acc = acc + re * re;
        if constexpr (A::is_complex) {
            const Real im = A::im(x);
            acc = acc + im * im;
        }
    }
}
}  // namespace detail
/// @brief `sum |x_i|^2` of one vector, or of several vectors of equal size in one loop.
///
/// No scaling: the squares can overflow or underflow (see `lassq` and `nrm2`). One reduction
/// over scope `S` per vector. COLLECTIVE; uniform results.
/// @ingroup api_device_blas1
/// @tparam Alg `alg::Fused` (default): `re^2` then `im^2` are added to the running sum, each a
///   contractible `a * a + acc`. `alg::Unfused`: adds `abs2(x_i)`, its own rounding.
/// @tparam S the reduction scope, as for `reduce`.
/// @param ex the executor.
/// @param x the first vector; its size is the trip count.
/// @param xs further vectors, each at least `x.size()` long.
/// @return the real sum for one vector, or a `std::array` of `1 + sizeof...(Xs)` sums.
template <class Alg = alg::Fused, Scope S = Scope::Own, class Ex, class X, class... Xs>
BATCHLAS_DEVICE_INLINE auto sumsq(const Ex& ex, const X& x, const Xs&... xs) {
    using R = real_t<elem_t<X>>;
    constexpr std::size_t K = 1 + sizeof...(Xs);
    std::array<R, K> acc{};
    for_each_owned(ex, x.size(), [&](int i) {
        std::size_t k = 0;
        detail::acc_abs2<Alg>(acc[k++], x(i));
        (detail::acc_abs2<Alg>(acc[k++], xs(i)), ...);
    });
    detail::reduce_each<S>(ex, acc);
    if constexpr (K == 1) return acc[0];
    else return acc;
}

/// @brief The first `i` in `[0, x.size())` with `pred(x(i))`, or `none`.
///
/// Each item stops at its first owned hit (owned indices ascend), then one minimum reduction over
/// scope `S`. To search on an index, pass `fn_vec(n, f)` as `x`. COLLECTIVE; uniform result.
/// @ingroup api_device_blas1
/// @tparam S the reduction scope, as for `reduce`.
/// @param ex the executor.
/// @param x the vector searched.
/// @param pred called as `pred(x(i))`, returning `bool`.
/// @param none the value returned when no element satisfies `pred` (default -1).
/// @return the smallest matching index, or `none`.
template <Scope S = Scope::Own, class Ex, class X, class P>
    requires(requires(const X& x) { x.size(); x(0); })
BATCHLAS_DEVICE_INLINE int find_first(const Ex& ex, const X& x, P&& pred, int none = -1) {
    const int n = x.size();
    int first = INT_MAX;
    const int step = detail::size_of(ex);
    for (int i = ex.lid(); i < n; i += step)
        if (pred(x(i))) {
            first = i;
            break;
        }
    first = detail::scope_reduce<S>(ex, first, sycl::minimum<int>());
    return first == INT_MAX ? none : first;
}

namespace alg {
/// @brief `iamax` and `amax_by` key: `|re| + |im|`, LAPACK's `cabs1` (the BLAS `i?amax` key).
/// @ingroup api_device_blas1
struct KeyAbs1 {};
/// @brief `iamax` and `amax_by` key: the modulus `|x|` (a hypot for complex).
/// @ingroup api_device_blas1
struct KeyAbs {};
/// @brief `iamax` key: the real part, signed (`amax_by`: its absolute value).
/// @ingroup api_device_blas1
struct KeyReal {};
}  // namespace alg
/// @brief `{index, key}` of the largest `key(x(i))` over `i` in `r`, optionally leaving out one index
/// (BLAS `i?amax`; the pivot search of LAPACK getf2 and sytf2).
///
/// Indices are 0-based. When no index is a candidate the result is `{none, seed}` with `seed = -1`
/// (`alg::KeyReal`: `-inf`). COLLECTIVE; the result is uniform.
/// @ingroup api_device_blas1
/// @tparam NaN `alg::NanAsInf` (default; a NaN key ranks as `+inf`, so it is found) or
///   `alg::NanSkip` (a NaN never beats a number).
/// @tparam Key `alg::KeyAbs1` (default), `alg::KeyAbs` or `alg::KeyReal`.
/// @param ex the executor.
/// @param x the vector searched.
/// @param r the index range searched, in `x`'s coordinates.
/// @param tie which index wins a tie: `alg::TieLowest` (default), `alg::TieHighest` or a run-time
///   `alg::TieRt{low}`.
/// @param slots the work-group exchange: `None` (default; two group collectives) or an
///   `ArgMaxSlots` in local memory (its own barriers).
/// @param skip an index left out of the search; `None` (default) leaves none out.
/// @param none the index reported when there is no candidate (default -1).
/// @param lanes the lane-exchange policy passed to `argmax_allreduce`; `None` by default.
/// @return an `ArgMax` with the winning index and its key.
template <class NaN = alg::NanAsInf, class Key = alg::KeyAbs1, class Ex, class X, class Tie = alg::TieLowest, class S = None,
          class K = None, class W = None>
BATCHLAS_DEVICE_INLINE ArgMax<real_t<elem_t<X>>> iamax(const Ex& ex, const X& x, Range r, const Tie& tie = {}, S slots = {},
                                                    K skip = {}, int none = -1, W lanes = {}) {
    using T = elem_t<X>;
    using R = real_t<T>;
    static_assert(std::is_same_v<NaN, alg::NanAsInf> || std::is_same_v<NaN, alg::NanSkip>, "iamax: alg::NanAsInf or alg::NanSkip");
    const R seed = std::is_same_v<Key, alg::KeyReal> ? -std::numeric_limits<R>::infinity() : R(-1);
    ArgMax<R> best{none, seed};
    for_each_owned(ex, r, [&](int i) BATCHLAS_DEVICE_BODY {
        if constexpr (!std::is_same_v<K, None>) {
            if (i == skip) return;
        }
        const T xi = x(i);
        R v;
        if constexpr (std::is_same_v<Key, alg::KeyAbs1>) v = abs1<T>(xi);
        else if constexpr (std::is_same_v<Key, alg::KeyAbs>) v = modulus(xi);
        else v = Arith<T>::re(xi);
        // NanSkip needs no map here: every compare with a NaN is false, so it is never taken.
        if constexpr (std::is_same_v<NaN, alg::NanAsInf>) v = sycl::isnan(v) ? std::numeric_limits<R>::infinity() : v;
        // An item's indices ascend: a static tie needs no index compare.
        if constexpr (std::is_same_v<Tie, alg::TieLowest>) {
            if (v > best.value) best.value = v, best.index = i;
        } else if constexpr (std::is_same_v<Tie, alg::TieHighest>) {
            if (v >= best.value) best.value = v, best.index = i;
        } else {
            detail::argmax_take(best, v, i, tie);
        }
    });
    return argmax_allreduce(ex, best, tie, slots, none, seed, lanes);
}
/// @brief `{index, key}` of the largest `key(x(i))` over all of `x` (BLAS `i?amax`, 0-based).
///
/// The range form over `Range{0, x.size()}` with no skip and the default exchange. COLLECTIVE;
/// uniform result.
/// @ingroup api_device_blas1
/// @tparam NaN `alg::NanAsInf` (default) or `alg::NanSkip`, as for the range form.
/// @tparam Key `alg::KeyAbs1` (default), `alg::KeyAbs` or `alg::KeyReal`.
/// @param ex the executor.
/// @param x the vector searched.
/// @param tie which index wins a tie (default `alg::TieLowest`).
/// @return an `ArgMax` with the winning index (-1 for an empty `x`) and its key.
template <class NaN = alg::NanAsInf, class Key = alg::KeyAbs1, class Ex, class X, class Tie = alg::TieLowest>
    requires(requires(const X& v) { v.size(); })
BATCHLAS_DEVICE_INLINE ArgMax<real_t<elem_t<X>>> iamax(const Ex& ex, const X& x, const Tie& tie = {}) {
    return iamax<NaN, Key>(ex, x, Range{0, x.size()}, tie);
}

/// @brief The largest key over `x`, with the NaN policy named.
///
/// An empty `x` gives 0. COLLECTIVE; uniform result.
/// @ingroup api_device_blas1
/// @tparam NaN `alg::NanPropagate` (default; any NaN gives NaN, exactly `amax`; only with
///   `alg::KeyAbs1`) or `alg::NanSkip` (`fmax`: a NaN element is ignored, the rule of bdsqr and
///   the tridiagonal solvers).
/// @tparam Key `alg::KeyAbs1` (default), `alg::KeyAbs`, or `alg::KeyReal` (here `|re|`).
/// @tparam S the reduction scope, as for `reduce`.
/// @param ex the executor.
/// @param x the vector.
/// @return the maximum key, of the real type of `x`'s elements.
template <class NaN = alg::NanPropagate, class Key = alg::KeyAbs1, Scope S = Scope::Own, class Ex, class X>
BATCHLAS_DEVICE_INLINE auto amax_by(const Ex& ex, const X& x) {
    using T = elem_t<X>;
    using R = real_t<T>;
    if constexpr (std::is_same_v<NaN, alg::NanPropagate> && std::is_same_v<Key, alg::KeyAbs1>) {
        return amax<S>(ex, x);
    } else {
        static_assert(std::is_same_v<NaN, alg::NanSkip>, "amax_by: alg::NanPropagate (alg::KeyAbs1) or alg::NanSkip");
        R m = R(0);
        for_each_owned(ex, x.size(), [&](int i) {
            const T v = x(i);
            R a;
            if constexpr (std::is_same_v<Key, alg::KeyAbs1>) a = abs1(v);
            else if constexpr (std::is_same_v<Key, alg::KeyAbs>) a = modulus(v);
            else a = sycl::fabs(Arith<T>::re(v));
            m = sycl::fmax(m, a);
        });
        return detail::scope_reduce<S>(ex, m, sycl::maximum<R>());
    }
}

// ---- level 1 (design A.2) -----------------------------------------------------------------------

/// @brief Calls `fs(i)...` in argument order for every owned `i < n`: several outputs in one pass.
/// @ingroup api_device_blas1
/// @param ex the executor.
/// @param n the extent.
/// @param fs the callables, each called as `f(int i)`.
template <class Ex, class... Fs> BATCHLAS_DEVICE_INLINE void map_many(const Ex& ex, int n, const Fs&... fs) {
    for_each_owned(ex, n, [&](int i) { (fs(i), ...); });
}

/// @brief `x(i) := alpha` for `i < x.size()`; elementwise, no barrier.
/// @ingroup api_device_blas1
/// @param ex the executor.
/// @param alpha the value: a value or a tag (`Zero`, `One`, ...).
/// @param[out] x the vector.
template <class Ex, class AS, class X> BATCHLAS_DEVICE_INLINE void fill(const Ex& ex, AS alpha, const X& x) {
    using T = elem_t<X>;
    const T v = tag_mul(as_tag(alpha), Arith<T>::one());
    for_each_owned(ex, x.size(), [&](int i) { x(i) = v; });
}

/// @brief The perfect shuffle: `y(2k) := a(k)` and `y(2k + 1) := b(k)`, one owned loop over `k < a.size()`.
/// @ingroup api_device_lowlevel
/// @pre `b.size()` is `a.size()` or `a.size() - 1`, and `y` holds `a.size() + b.size()` elements.
/// @param ex the executor.
/// @param a the even-position source.
/// @param b the odd-position source.
/// @param[out] y the destination.
template <class Ex, class A, class B, class Y> BATCHLAS_DEVICE_INLINE void interleave(const Ex& ex, const A& a, const B& b, const Y& y) {
    const int nb = b.size();
    for_each_owned(ex, a.size(), [&](int k) {
        y(2 * k) = a(k);
        if (k < nb) y(2 * k + 1) = b(k);
    });
}

/// @brief Reverses `x` in place: the item owning `i < n / 2` swaps `x(i)` and `x(n - 1 - i)`.
///
/// On `Thread` this is the serial two-pointer loop. Each element is written by the item that owns
/// its lower index, so a later operation that reads it under the usual ownership needs a barrier.
/// @ingroup api_device_lowlevel
/// @param ex the executor.
/// @param[in,out] x the vector, `n = x.size()`.
template <class Ex, class X> BATCHLAS_DEVICE_INLINE void reverse(const Ex& ex, const X& x) {
    const int n = x.size();
    if constexpr (is_thread_v<Ex>) {
        for (int a = 0, b = n - 1; a < b; ++a, --b) {
            const auto t = x(a);
            x(a) = x(b);
            x(b) = t;
        }
    } else {
        for_each_owned(ex, n / 2, [&](int i) {
            const auto t = x(i);
            x(i) = x(n - 1 - i);
            x(n - 1 - i) = t;
        });
    }
}
/// @brief Reverses the window `[x.lo, x.hi)` of a lane vector by one shuffle, when `on`.
///
/// Items outside the window keep their value. COLLECTIVE: every lane calls, with a uniform `on`.
/// @ingroup api_device_lowlevel
/// @param ex the executor (a sub-group or partition).
/// @param[in,out] x the lane vector (one element per item).
/// @param on `false` leaves every value in place.
template <class Ex, class T> BATCHLAS_DEVICE_INLINE void reverse(const Ex& ex, const LaneVec<T>& x, bool on) {
    const int l = ex.lid();
    const int src = (on && l >= x.lo && l < x.hi) ? (x.lo + x.hi - 1 - l) : l;
    x.v = ex.shuffle(static_cast<std::remove_cv_t<T>>(x.v), src);
}

/// @brief The number of `x(i) <= v` in an ascending `x` (the offset `std::upper_bound` returns), by
/// bisection on one item.
/// @ingroup api_device_lowlevel
/// @pre `x` is sorted ascending.
/// @param x the sorted vector.
/// @param v the value searched for.
/// @return an index in `[0, x.size()]`.
template <class X, class Real> BATCHLAS_DEVICE_INLINE int upper_bound(const Thread&, const X& x, Real v) {
    int lo = 0, hi = x.size();
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (!(v < x(mid))) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

/// @brief `x := x / d` (LAPACK rscl), elementwise, no barrier.
///
/// One reciprocal and a multiply per element when `1 / d` is finite and nonzero, else a division
/// per element (`SafeRecip`); `d == 0` divides, giving Inf or NaN as LAPACK does.
/// @ingroup api_device_blas1
/// @tparam Ops the arithmetic policy (default `arith::Fma`).
/// @param ex the executor.
/// @param d the divisor, converted to `x`'s element type.
/// @param[in,out] x the vector.
template <class Ops = arith::Fma, class Ex, class Scalar, class X> BATCHLAS_DEVICE_INLINE void rscl(const Ex& ex, const Scalar& d, const X& x) {
    using E = elem_t<X>;
    const auto s = SafeRecip<E, Ops>::make(static_cast<E>(d));
    for_each_owned(ex, x.size(), [&](int i) {
        const E v = x(i);
        x(i) = s.use_mul ? arith_t<Ops, E>::mul(v, s.r) : arith_t<Ops, E>::div(v, s.d);
    });
}

/// @brief This item's partial `op(x) y` over its owned elements, with no reduction.
///
/// The caller combines the partials, for example in one `allreduce` through a `Scratch` exchange
/// shared with other sums. On a `Fit` executor, the item's one product (0 past `x.size()`).
/// @ingroup api_device_blas1
/// @tparam TR `Transpose::Trans` (default, `x^T y`) or `Transpose::ConjTrans` (`x^H y`).
/// @tparam Ops the arithmetic policy of the products.
/// @param ex the executor.
/// @param x the first vector; its size is the trip count.
/// @param y the second vector.
/// @return this item's partial sum (not uniform).
template <Transpose TR = Transpose::Trans, class Ops = arith::Explicit, class Ex, class X, class Y>
BATCHLAS_DEVICE_INLINE auto dot_partial(const Ex& ex, const X& x, const Y& y) {
    return detail::dot_partial<TR, Ops>(ex, x, y);
}

/// @brief `acc += x(t) y(t)` for `t < N` with `kk0 + t < k`, in index order: one item's dot over a staged slab.
/// @ingroup api_device_lowlevel
/// @tparam N the slab length.
/// @tparam Ops the arithmetic policy (default `arith::Native`).
/// @param[in,out] acc the accumulator.
/// @param x the slab of the first operand, indexed from 0.
/// @param y the slab of the second operand.
/// @param kk0 the slab's offset in the full dot.
/// @param k the full dot's length.
template <int N, class Ops = arith::Native, class Scalar, class VX, class VY>
BATCHLAS_DEVICE_INLINE void dot_slab(Scalar& acc, const VX& x, const VY& y, int kk0, int k) {
    using A = arith_t<Ops, Scalar>;
    for (int t = 0; t < N && kk0 + t < k; ++t) A::mac(acc, x(t), y(t));
}

/// @brief A product as `mant * 2^expo`, the result of `prod_scaled`.
/// @ingroup api_device_lowlevel
template <class Real> struct ScaledProd {
    /// The mantissa, in `[0.5, 1)` in magnitude unless the product is 0, Inf or NaN.
    Real mant;
    /// The binary exponent.
    std::int32_t expo;
};
/// @brief The product of `f(i)` over `i < n` as `mant * 2^expo`, free of overflow and underflow.
///
/// Each item renormalises its running product with `frexp`, then the items' mantissas combine 32
/// at a time with renormalisation (a product of 32 mantissas lies in `[2^-32, 1]`). COLLECTIVE;
/// uniform result.
/// @ingroup api_device_lowlevel
/// @param ex the executor.
/// @param n the number of factors.
/// @param f called as `f(int i)`, returning a real factor.
/// @return a `ScaledProd`.
template <class Ex, class F> BATCHLAS_DEVICE_INLINE auto prod_scaled(const Ex& ex, int n, F&& f) {
    using R = std::remove_cvref_t<decltype(f(0))>;
    R mant = R(1);
    std::int32_t expo = 0;
    for_each_owned(ex, n, [&](int i) {
        mant *= f(i);
        if (mant != R(0) && sycl::isfinite(mant)) {
            int e = 0;
            mant = sycl::frexp(mant, &e);
            expo += e;
        }
    });
    R mant_all = R(1);
    std::int32_t expo_all = allreduce(ex, expo);
    const int width = detail::size_of(ex);
    const int lane = ex.lid();
    for (std::int32_t r0 = 0; r0 < width; r0 += 32) {
        const bool mine = lane >= r0 && lane < r0 + 32;
        mant_all *= allreduce(ex, mine ? mant : R(1), sycl::multiplies<R>());
        if (mant_all != R(0) && sycl::isfinite(mant_all)) {
            int e = 0;
            mant_all = sycl::frexp(mant_all, &e);
            expo_all += e;
        }
    }
    return ScaledProd<R>{mant_all, expo_all};
}
/// @brief `sqrt(|q.mant| 2^q.expo)` without forming the power.
/// @ingroup api_device_lowlevel
/// @param q a scaled product.
/// @return the square root of its magnitude.
template <class Real> BATCHLAS_DEVICE_INLINE Real sqrt_scaled(ScaledProd<Real> q) {
    const std::int32_t half = q.expo >> 1;  // floor(expo / 2)
    const std::int32_t rem = q.expo - 2 * half;
    const Real mag = sycl::sqrt(sycl::fabs(q.mant) * (rem ? Real(2) : Real(1)));
    return sycl::ldexp(mag, half);
}

/// @brief On a lane vector: the smallest `lid` in `[x.lo, x.hi)` whose element satisfies `pred`, or `none`.
///
/// One minimum butterfly over the executor's lanes. COLLECTIVE; uniform result.
/// @ingroup api_device_blas1
/// @param ex the executor (a sub-group or partition).
/// @param x the lane vector (one element per item).
/// @param pred called as `pred(value)`, returning `bool`.
/// @param none the value returned when no element satisfies `pred` (default -1).
/// @return the smallest matching lane, or `none`.
template <class Ex, class T, class P> BATCHLAS_DEVICE_INLINE int find_first(const Ex& ex, const LaneVec<T>& x, P&& pred, int none = -1) {
    const int l = ex.lid();
    const int cand = (l >= x.lo && l < x.hi && pred(static_cast<std::remove_cv_t<T>>(x.v))) ? l : none;
    return detail::butterfly_inline<1, Ex::size, alg::BflyAsc>(ex, cand, sycl::minimum<int>());
}

// ---- Cyclic rows of a static extent N (N % size == 0): item l owns rows l + k*size ----------
/// @brief Calls `f(k, row)` for the item's `N / size` cyclic rows `row = lid + k * size`, `k` the
/// register-fragment slot; no bound check.
/// @ingroup api_device_lowlevel
/// @tparam N the static extent, a multiple of the executor's size.
/// @tparam U `alg::UnrollFull` (default, a full unroll) or `alg::UnrollDefault`.
/// @param ex the executor (a compile-time size).
/// @param f called as `f(int k, int row)`.
template <int N, class U = alg::UnrollFull, class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_owned_frag(const Ex& ex, F&& f) {
    static_assert(N % Ex::size == 0, "for_each_owned_frag: N must be a multiple of the executor size");
    constexpr int K = N / Ex::size;
    const int l = ex.lid();
    if constexpr (std::is_same_v<U, alg::UnrollFull>) {
#pragma unroll
        for (int k = 0; k < K; ++k) f(k, l + k * Ex::size);
    } else {
        static_assert(std::is_same_v<U, alg::UnrollDefault>, "for_each_owned_frag: alg::UnrollFull or alg::UnrollDefault");
        for (int k = 0; k < K; ++k) f(k, l + k * Ex::size);
    }
}

/// @brief `x[k] := v(lid + k * size)` for rows below `n`, else 0: a masked load into a cyclic fragment.
/// @ingroup api_device_lowlevel
/// @param ex the executor.
/// @param v the vector loaded.
/// @param n the number of valid rows.
/// @param[out] x the item's fragment of `K` slots.
template <class Ex, class V, class T, int K> BATCHLAS_DEVICE_INLINE void load_owned_masked(const Ex& ex, const V& v, int n, Frag<T, K>& x) {
    for_each_owned_frag<K * Ex::size>(ex, [&](int k, int r) { x[k] = (r < n) ? static_cast<T>(v(r)) : T(0); });
}
/// @brief `x := e_j`, the unit vector, restricted to the item's cyclic rows.
/// @ingroup api_device_lowlevel
/// @param ex the executor.
/// @param j the row of the 1.
/// @param[out] x the item's fragment.
template <class Ex, class T, int K> BATCHLAS_DEVICE_INLINE void set_unit_owned(const Ex& ex, int j, Frag<T, K>& x) {
    for_each_owned_frag<K * Ex::size>(ex, [&](int k, int r) { x[k] = (r == j) ? T(1) : T(0); });
}
/// @brief `sum_k conj(x[k]) y[k]` over the item's fragment: a partial dotc to reduce with `allreduce`.
/// @ingroup api_device_lowlevel
/// @tparam Ops the arithmetic policy (default `arith::Native`).
/// @param x the conjugated fragment.
/// @param y the second fragment.
/// @return this item's partial sum.
template <class Ops = arith::Native, class Scalar, int K> BATCHLAS_DEVICE_INLINE Scalar dotc_owned(const Frag<Scalar, K>& x, const Frag<Scalar, K>& y) {
    Scalar acc = Scalar(0);
#pragma unroll
    for (int k = 0; k < K; ++k) {
        if constexpr (detail::is_native_v<Ops>) acc = acc + conj(x[k]) * y[k];
        else arith_t<Ops, Scalar>::mac_conj(acc, x[k], y[k]);
    }
    return acc;
}
// Two loads plus dotc_owned gave complex T extra BRA.
/// @brief Loads `x[k] := u(lid + k * size)` and `y[k] := v(lid + k * size)` and returns the partial
/// dotc `sum_k conj(x[k]) y[k]`, in one pass.
/// @ingroup api_device_lowlevel
/// @tparam K the fragment slots.
/// @param ex the executor.
/// @param u the first vector loaded.
/// @param v the second vector loaded.
/// @param[out] x the first fragment (a `Frag` or `T[K]`).
/// @param[out] y the second fragment.
/// @return this item's partial sum.
template <int K, class Ex, class U, class V, class X>
BATCHLAS_DEVICE_INLINE auto load_dotc_owned(const Ex& ex, const U& u, const V& v, X& x, X& y) {
    using T = std::remove_cvref_t<decltype(x[0])>;
    T acc = T(0);
    const int l = ex.lid();
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int r = l + k * Ex::size;
        x[k] = u(r);
        y[k] = v(r);
        acc = acc + conj(x[k]) * y[k];
    }
    return acc;
}
/// @brief `y[k] := y[k] - x[k] a` over a fragment: one classical Gram-Schmidt projection step.
/// @ingroup api_device_lowlevel
/// @tparam Ops the arithmetic policy (default `arith::Native`).
/// @param[in,out] y the fragment updated.
/// @param x the fragment subtracted.
/// @param a the coefficient.
template <class Ops = arith::Native, class Scalar, int K> BATCHLAS_DEVICE_INLINE void sub_mul(Frag<Scalar, K>& y, const Frag<Scalar, K>& x, Scalar a) {
#pragma unroll
    for (int k = 0; k < K; ++k) {
        if constexpr (detail::is_native_v<Ops>) y[k] = y[k] - x[k] * a;
        else arith_t<Ops, Scalar>::msub(y[k], x[k], a);
    }
}
/// @brief This item's `sum |x[k]|^2` over its cyclic rows below `n` (each `abs2` its own rounding), not reduced.
/// @ingroup api_device_lowlevel
/// @param ex the executor.
/// @param x the item's fragment.
/// @param n the number of valid rows.
/// @return this item's partial sum.
template <class Ex, class Scalar, int K> BATCHLAS_DEVICE_INLINE real_t<Scalar> sumsq_owned(const Ex& ex, const Frag<Scalar, K>& x, int n) {
    real_t<Scalar> acc = real_t<Scalar>(0);
    for_each_owned_frag<K * Ex::size>(ex, [&](int k, int r) {
        if (r < n) acc += abs2(x[k]);
    });
    return acc;
}
/// @brief `sum |x_i|^2` of a cyclic register vector, reduced over scope `S`.
///
/// `x[k]` is row `lid + k * size`; rows at or past `n` are skipped. COLLECTIVE; uniform result.
/// @ingroup api_device_blas1
/// @tparam Alg `alg::Fused` (default) or `alg::Unfused`, as for the memory form.
/// @tparam S the reduction scope, as for `reduce`.
/// @param ex the executor.
/// @param x the item's share of the vector.
/// @param n the vector's length.
/// @return the sum, the same on every item.
template <class Alg = alg::Fused, Scope S = Scope::Own, class Ex, class Scalar, int N>
BATCHLAS_DEVICE_INLINE real_t<Scalar> sumsq(const Ex& ex, const Frag<Scalar, N, Cyclic>& x, int n) {
    real_t<Scalar> acc = real_t<Scalar>(0);
    for_each_owned_frag<N * Ex::size>(ex, [&](int k, int r) {
        if (r < n) detail::acc_abs2<Alg>(acc, x[k]);
    });
    return allreduce<S>(ex, acc);
}

/// @brief Lane `l` gets `||a(:, l)||^2` for the `P = Ex::size` columns of `a`.
///
/// `a`'s static row count (an `Ext`) is cyclic over the executor: per-lane partial sums, then one
/// `reduce_scatter<P>` (`log2 P` exchanges rather than `P` all-reductions). COLLECTIVE.
/// @ingroup api_device_lowlevel
/// @param ex the executor (a sub-group or partition of `P` lanes).
/// @param a a matrix with `P` columns and a static row count.
/// @return this lane's column's squared norm.
template <class Ex, class M> BATCHLAS_DEVICE_INLINE auto col_sumsq(const Ex& ex, const M& a) {
    using T = typename M::value_type;
    using R = real_t<T>;
    constexpr int P = Ex::size;
    constexpr int Rows = decltype(M::m)::value;
    Frag<R, P> x;
#pragma unroll
    for (int c = 0; c < P; ++c) {
        R acc = R(0);
        for_each_owned_frag<Rows>(ex, [&](int, int r) { acc += abs2(static_cast<T>(a(r, c))); });
        x[c] = acc;
    }
    return reduce_scatter<P>(ex, x);
}

}  // namespace batchlas::device
