// SPDX-License-Identifier: MIT
#pragma once
// Device LA level 1. Every op distributes over the executor's Cyclic layout (i owned by
// lid == i mod size), so two ops on the same executor touch an element from the same item and
// compose with NO barrier. No op here contains a work-group barrier except the packed
// Scratch exchange, which names it in its signature.

#include <array>
#include <climits>
#include <limits>
#include <type_traits>
#include <utility>

#include <batchlas/device/operand.hh>
#include <batchlas/device/reduce.hh>
#include <batchlas/device/scalar.hh>

namespace batchlas::device {

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

template <class U = alg::UnrollDefault, class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_owned(const Ex& ex, int n, F&& f) {
    if constexpr (std::is_same_v<U, alg::UnrollDefault> && !is_runtime_v<Ex>) {
        for (int i = ex.lid(); i < n; i += Ex::size) f(i);  // the first spelling, SASS-pinned by the lab
    } else {
        detail::strided_loop<U>(ex.lid(), n, detail::size_of(ex), f);
    }
}
template <class U = alg::UnrollDefault, class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_owned(const Ex& ex, Range r, F&& f) {
    if constexpr (std::is_same_v<U, alg::UnrollDefault> && !is_runtime_v<Ex>) {
        for (int i = r.lo + ex.lid(); i < r.hi; i += Ex::size) f(i);
    } else {
        detail::strided_loop<U>(r.lo + ex.lid(), r.hi, detail::size_of(ex), f);
    }
}
// for_each_owned(ex, Range{first, end}, f), the P3 spelling.
template <class U = alg::UnrollDefault, class Ex, class F>
BATCHLAS_DEVICE_INLINE void for_each_owned(const Ex& ex, int first, int end, F&& f) {
    for_each_owned<U>(ex, Range{first, end}, f);
}
// n == the executor's size: each item owns exactly i = lid, no loop and no guard (the caller
// guarantees n == size; krylov's ItemGroup).
template <class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_owned_exact(const Ex& ex, [[maybe_unused]] int n, F&& f) {
    f(ex.lid());
}
// Fit<Ex> (exec.hh): index lid only, one guarded call (n <= size); Range: lo + lid.
template <class U = alg::UnrollDefault, class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_owned(const Fit<Ex>& ex, int n, F&& f) {
    const int i = ex.lid();
    if (i >= 0 && i < n) f(i);
}
template <class U = alg::UnrollDefault, class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_owned(const Fit<Ex>& ex, Range r, F&& f) {
    const int i = r.lo + ex.lid();
    if (i < r.hi) f(i);
}
// Grid1D (exec.hh): the launch's element i; Exact drops the i < n test.
template <class U = alg::UnrollDefault, bool E, class F> BATCHLAS_DEVICE_INLINE void for_each_owned(const Grid1D<E>& g, int n, F&& f) {
    if constexpr (E) f(g.i);
    else if (g.i < n) f(g.i);
}

// The team loop: f(o, live) for the outputs o this item's team owns (o = team + k * nteams).
// alg::Lockstep: every team runs ceil(nout / nteams) trips with live = o < nout, so a team
// collective inside f is reached by every lane; alg::Diverge: o < nout only (live is true).
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

// f(i0, j0, live_r, live_c) for every TR x TC tile of an m x n grid this item owns: tiles in
// column-major tile order, tile t owned by lid == t mod size; live_* clip the edge tiles.
template <int TR, int TC, class Ex, class F> BATCHLAS_DEVICE_INLINE void for_each_owned_tiles(const Ex& ex, int m, int n, F&& f) {
    static_assert(TR >= 1 && TC >= 1, "for_each_owned_tiles: TR, TC >= 1");
    const int tr = (m + TR - 1) / TR, tc = (n + TC - 1) / TC;
    for_each_owned(ex, tr * tc, [&](int t) {
        const int bj = t / tr, bi = t - bj * tr;
        const int i0 = bi * TR, j0 = bj * TC;
        f(i0, j0, sycl::min(TR, m - i0), sycl::min(TC, n - j0));
    });
}

// f(r, c) for every element of an m x n column-major grid this item owns: flat e = lid + s *
// size, so consecutive items take consecutive rows (loads coalesce) and per-item partial sums
// keep the legacy grid-stride order. Alg (core.hh alg::) is the hand kernel's index form:
//   Flat     q = e / m, (e - q m, q) per step, e in Idx (ptrdiff_t: the legacy 64-bit index math)
//   DivMod   (e % m, e / m) per step
//   Stepped  one division per item; r += size, then subtract m while r >= m
//   Walk     one division per item; (r, c) += (size % m, size / m) with one carry per step
//   RowOwner rows owned as for_each_owned (row i by lid == i mod size), each walked serially
//            over its columns (the hand kernels' row-copy form; on Fit, one guarded row)
// On Thread every Alg is the column-major double loop; on a Grid2D, the item's one element.
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

// ElemAt (exec.hh): the item's one element; Exact drops the extent test. ElemIn: the
// element relative to its block origin, column test first.
template <class Alg = alg::Walk, class Idx = int, bool E, class F>
BATCHLAS_DEVICE_INLINE void for_each_owned_2d(const ElemAt<E>& ex, int m, int n, F&& f) {
    if constexpr (E) f(ex.r, ex.c);
    else if (ex.r < m && ex.c < n) f(ex.r, ex.c);
}
template <class Alg = alg::Walk, class Idx = int, class F>
BATCHLAS_DEVICE_INLINE void for_each_owned_2d(const ElemIn& ex, int m, int n, F&& f) {
    if (ex.c >= ex.c0) {
        if (ex.r >= ex.r0 && ex.r < ex.r0 + m && ex.c < ex.c0 + n) f(ex.r - ex.r0, ex.c - ex.c0);
    }
}

// out(i) = f(ins(i)...)
template <class Ex, class Out, class F, class... In>
BATCHLAS_DEVICE_INLINE void map(const Ex& ex, const Out& out, F&& f, const In&... ins) {
    for_each_owned(ex, out.size(), [&](int i) { out(i) = f(ins(i)...); });
}

// y := a x + b y; b == Zero never reads y.
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

// y := a x + y
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

// x := a x (arith::Native with a value: x *= a, T's own in-place multiply)
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

template <class Ex, class X, class Y> BATCHLAS_DEVICE_INLINE void copy(const Ex& ex, const X& x, const Y& y) {
    for_each_owned(ex, y.size(), [&](int i) { y(i) = x(i); });
}

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
template <class Ex, class T, std::size_t K, std::size_t... I>
BATCHLAS_DEVICE_INLINE void team_reduce_each(const Ex& ex, std::array<T, K>& v, std::index_sequence<I...>) {
    ((v[I] = ex.team_reduce(v[I])), ...);
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

// K sums from ONE scratch exchange: per-team butterfly, lane 0 writes K values, one barrier,
// every item reads nteams*K in a fixed order (uniform result), one trailing barrier so the
// scratch may be reused.
template <int K, class Ex, class T, int Slots>
BATCHLAS_DEVICE_INLINE std::array<T, K> packed_sum(const Ex& ex, Scratch<T, Slots> scr, std::array<T, K> v) {
    if constexpr (is_runtime_v<Ex>) {
        reduce_each<Scope::Group>(ex, v);
        return v;
    } else if constexpr (!is_work_group_v<Ex> || Ex::nteams == 1) {
        team_reduce_each(ex, v, std::make_index_sequence<K>{});
        return v;
    } else {
        static_assert(Slots >= Ex::nteams * K, "packed_sum: Scratch smaller than scratch_elements<reduce_op::ReduceN<K>, Ex, T>()");
        if (!packed_contract_holds(ex)) {
            reduce_each<Scope::Group>(ex, v);
            return v;
        }
        team_reduce_each(ex, v, std::make_index_sequence<K>{});
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
            for (int k = 0; k < K; ++k) out[k] = Arith<T>::add(out[k], scr.p[s * K + k]);
        }
        ex.barrier(Mem::Local);
        return out;
    }
}
}  // namespace detail

// sum over i < n of f(i)
template <Scope S = Scope::Own, class Ex, class F> BATCHLAS_DEVICE_INLINE auto reduce(const Ex& ex, int n, F&& f) {
    using T = std::remove_cvref_t<decltype(f(0))>;
    T acc = Arith<T>::zero();
    for_each_owned(ex, n, [&](int i) { acc = Arith<T>::add(acc, f(i)); });
    return detail::scope_reduce<S>(ex, acc);
}
template <class Ex, class T, int Slots, class F>
BATCHLAS_DEVICE_INLINE T reduce(const Ex& ex, Scratch<T, Slots> scr, int n, F&& f) {
    T acc = Arith<T>::zero();
    for_each_owned(ex, n, [&](int i) { acc = Arith<T>::add(acc, f(i)); });
    return detail::packed_sum<1>(ex, scr, std::array<T, 1>{acc})[0];
}

namespace detail {
template <bool Conj, class Ops = arith::Explicit, class Ex, class X, class Y>
BATCHLAS_DEVICE_INLINE auto dot_partial(const Ex& ex, const X& x, const Y& y) {
    using T = elem_t<X>;
    using Ar = arith_t<Ops, T>;
    T acc = Ar::zero();
    for_each_owned(ex, x.size(), [&](int i) {
        if constexpr (Conj) Ar::mac_conj(acc, x(i), y(i));
        else Ar::mac(acc, x(i), y(i));
    });
    return acc;
}
}  // namespace detail

// x^T y (no conjugation)
template <Scope S = Scope::Own, class Ops = arith::Explicit, class Ex, class X, class Y>
BATCHLAS_DEVICE_INLINE auto dot(const Ex& ex, const X& x, const Y& y) {
    return detail::scope_reduce<S>(ex, detail::dot_partial<false, Ops>(ex, x, y));
}
// x^H y
template <Scope S = Scope::Own, class Ops = arith::Explicit, class Ex, class X, class Y>
BATCHLAS_DEVICE_INLINE auto dotc(const Ex& ex, const X& x, const Y& y) {
    return detail::scope_reduce<S>(ex, detail::dot_partial<true, Ops>(ex, x, y));
}

namespace detail {
template <int K, bool Conj, class Ex, class Xs, class Y> BATCHLAS_DEVICE_INLINE auto dotn_partial(const Ex& ex, const Xs& xs, const Y& y) {
    using T = elem_t<Y>;
    std::array<T, K> acc;
#pragma unroll
    for (int k = 0; k < K; ++k) acc[k] = Arith<T>::zero();
    for_each_owned(ex, y.size(), [&](int i) {
        const T yi = y(i);
#pragma unroll
        for (int k = 0; k < K; ++k) {
            if constexpr (Conj) Arith<T>::mac_conj(acc[k], xs[k](i), yi);
            else Arith<T>::mac(acc[k], xs[k](i), yi);
        }
    });
    return acc;
}
}  // namespace detail

// {xs[k]^T y} (Conj: xs[k]^H y) for k < K. With Scratch on a WorkGroup: one packed exchange.
template <int K, bool Conj = false, class Ex, class T, int Slots, class X, class Y>
BATCHLAS_DEVICE_INLINE std::array<T, K> dotN(const Ex& ex, Scratch<T, Slots> scr, const std::array<X, K>& xs, const Y& y) {
    return detail::packed_sum<K>(ex, scr, detail::dotn_partial<K, Conj>(ex, xs, y));
}
namespace detail {
template <bool Conj, class Ops, class Ex, class Y, class... Xs> BATCHLAS_DEVICE_INLINE auto dot_many_partial(const Ex& ex, const Y& y, const Xs&... xs) {
    using T = elem_t<Y>;
    using Ar = arith_t<Ops, T>;
    constexpr int K = sizeof...(Xs);
    std::array<T, K> acc;
#pragma unroll
    for (int k = 0; k < K; ++k) acc[k] = Ar::zero();
    for_each_owned(ex, y.size(), [&](int i) {
        const T yi = y(i);
        int k = 0;
        ((Conj ? Ar::mac_conj(acc[k++], xs(i), yi) : Ar::mac(acc[k++], xs(i), yi)), ...);
    });
    return acc;
}
}  // namespace detail

// {xs^T y...} (Conj: xs^H y) for operands of different types (e.g. a column and its
// unit_head), one loop over y and one packed exchange: latrd's gamma/delta pair.
template <bool Conj = false, class Ops = arith::Explicit, class Ex, class T, int Slots, class Y, class... Xs>
BATCHLAS_DEVICE_INLINE std::array<T, sizeof...(Xs)> dot_many(const Ex& ex, Scratch<T, Slots> scr, const Y& y, const Xs&... xs) {
    return detail::packed_sum<sizeof...(Xs)>(ex, scr, detail::dot_many_partial<Conj, Ops>(ex, y, xs...));
}
// Without Scratch: one loop over y, then one scope-S reduction per x in argument order (on a
// WorkGroup, portable::reduce_over_group each: the legacy kernels' trees, bit-identical).
template <bool Conj = false, Scope S = Scope::Own, class Ops = arith::Explicit, class Ex, class Y, class... Xs>
BATCHLAS_DEVICE_INLINE auto dot_many(const Ex& ex, const Y& y, const Xs&... xs) {
    auto acc = detail::dot_many_partial<Conj, Ops>(ex, y, xs...);
    detail::reduce_each<S>(ex, acc);
    return acc;
}

// Without Scratch: K independent reductions of scope S (on a sub-group or partition these are
// shuffles only, so nothing is exchanged).
template <int K, bool Conj = false, Scope S = Scope::Own, class Ex, class X, class Y>
BATCHLAS_DEVICE_INLINE auto dotN(const Ex& ex, const std::array<X, K>& xs, const Y& y) {
    auto acc = detail::dotn_partial<K, Conj>(ex, xs, y);
    detail::reduce_each<S>(ex, acc);
    return acc;
}

// sum |re| + |im|
template <Scope S = Scope::Own, class Ex, class X> BATCHLAS_DEVICE_INLINE auto asum(const Ex& ex, const X& x) {
    using R = real_t<elem_t<X>>;
    R acc = R(0);
    for_each_owned(ex, x.size(), [&](int i) { acc += abs1(x(i)); });
    return detail::scope_reduce<S>(ex, acc);
}

// max |re| + |im|. NaN policy (NaN::Propagate): any NaN element makes the result NaN.
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

// 0-based index of max |re| + |im|; ties -> the lowest index; NaN policy (NaN::First): the
// lowest index holding a NaN wins over every number; n <= 0 -> -1. Three collectives.
template <Scope S = Scope::Own, class Ex, class X> BATCHLAS_DEVICE_INLINE int iamax(const Ex& ex, const X& x) {
    using R = real_t<elem_t<X>>;
    R best = R(-1);
    int bi = INT_MAX, nani = INT_MAX;
    for_each_owned(ex, x.size(), [&](int i) {
        const R a = abs1(x(i));
        if (sycl::isnan(a)) nani = i < nani ? i : nani;
        else if (a > best) { best = a; bi = i; }
    });
    nani = detail::scope_reduce<S>(ex, nani, sycl::minimum<int>());
    if (nani != INT_MAX) return nani;
    const R m = detail::scope_reduce<S>(ex, best, sycl::maximum<R>());
    const int idx = detail::scope_reduce<S>(ex, best == m ? bi : INT_MAX, sycl::minimum<int>());
    return idx == INT_MAX ? -1 : idx;
}

// LAPACK lassq without the incoming pair: scale = max |component|, scale^2 * ssq = sum |x_i|^2,
// computed in two passes so no square overflows or underflows to zero.
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
template <class R> struct BlueConst {
    static constexpr int radix = std::numeric_limits<R>::radix, p = std::numeric_limits<R>::digits;
    static constexpr int emin = std::numeric_limits<R>::min_exponent, emax = std::numeric_limits<R>::max_exponent;
    static constexpr int fdiv(int a, int b) { return a / b - (((a ^ b) < 0) && (a % b) ? 1 : 0); }
    static constexpr int cdiv(int a, int b) { return a / b + (((a ^ b) >= 0) && (a % b) ? 1 : 0); }
    static constexpr R pw(int e) {
        R r = R(1);
        for (int i = 0; i < (e < 0 ? -e : e); ++i) r *= R(radix);
        return e < 0 ? R(1) / r : r;
    }
    static constexpr R tsml = pw(cdiv(emin - 1, 2)), tbig = pw(fdiv(emax - p + 1, 2));
    static constexpr R ssml = pw(-fdiv(emin - p, 2)), sbig = pw(-cdiv(emax + p - 1, 2));
};
template <class R> BATCHLAS_DEVICE_INLINE void blue_acc(R ax, R& abig, R& amed, R& asml) {
    using C = BlueConst<R>;
    if (ax > C::tbig) {
        const R t = ax * C::sbig;
        abig += t * t;
    } else if (ax < C::tsml) {
        const R t = ax * C::ssml;
        asml += t * t;
    } else {
        amed += ax * ax;
    }
}
template <class R> BATCHLAS_DEVICE_INLINE R blue_finish(R abig, R amed, R asml) {
    using C = BlueConst<R>;
    const R maxN = std::numeric_limits<R>::max();
    R scl = R(1), ss = R(0);
    if (abig > R(0)) {
        if ((amed > R(0)) || (amed > maxN) || (amed != amed)) abig += (amed * C::sbig) * C::sbig;
        scl = R(1) / C::sbig;
        ss = abig;
    } else if (asml > R(0)) {
        if ((amed > R(0)) || (amed > maxN) || (amed != amed)) {
            const R am = sycl::sqrt(amed), as = sycl::sqrt(asml) / C::ssml;
            const R ymin = sycl::fmin(am, as), ymax = sycl::fmax(am, as);
            ss = ymax * ymax * (R(1) + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = R(1) / C::ssml;
            ss = asml;
        }
    } else {
        ss = amed;
    }
    return scl * sycl::sqrt(ss);
}
}  // namespace detail

// ||x||_2 with the algorithm named (no default: alg::Blue is overflow-safe in one pass and one
// exchange but costs more than Unscaled; see the design report):
//   alg::Unscaled  sqrt(re(x^H x)), one reduction; overflows past sqrt(max), underflows below
//                  sqrt(min). Inf beats NaN only by accident of the sum.
//   alg::Blue      dnrm2.f90's three accumulators, one exchange.
//   alg::Scaled    two-pass lassq (max, then the scaled sum): two exchanges.
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

// v reduced over scope S (T, or std::array<T, K>: one reduction per value, pack-expanded).
template <Scope S = Scope::Own, class Ex, class T, class Op = sycl::plus<T>>
BATCHLAS_DEVICE_INLINE T allreduce(const Ex& ex, T v, Op op = {}) {
    return detail::scope_reduce<S>(ex, v, op);
}
template <Scope S = Scope::Own, class Ex, class T, std::size_t K, class Op = sycl::plus<T>>
BATCHLAS_DEVICE_INLINE std::array<T, K> allreduce(const Ex& ex, std::array<T, K> v, Op op = {}) {
    detail::reduce_each<S>(ex, v, op);
    return v;
}
// K sums in one packed exchange through Scratch (scratch_elements<reduce_op::ReduceN<K>, Ex, T>()).
template <class Ex, class T, int Slots, std::size_t K>
BATCHLAS_DEVICE_INLINE std::array<T, K> allreduce(const Ex& ex, Scratch<T, Slots> scr, std::array<T, K> v) {
    return detail::packed_sum<int(K)>(ex, scr, v);
}

// sum |x_i|^2 for one operand (-> R) or K operands of equal size in one loop (-> std::array<R, K>,
// one reduction each). Alg: alg::Fused accumulates re^2 then im^2 into the running sum (each a
// contractible a*a + acc: 2 FMA per complex element); alg::Unfused adds abs2(x) (its own
// rounding: FMUL + FFMA + FADD), the spelling of the hand kernels that call abs2.
namespace detail {
template <class Alg, class R, class T> BATCHLAS_DEVICE_INLINE void acc_abs2(R& acc, const T& x) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (std::is_same_v<Alg, alg::Unfused>) {
        acc += abs2(x);
    } else {
        static_assert(std::is_same_v<Alg, alg::Fused>, "sumsq: alg::Fused or alg::Unfused");
        const R re = A::re(x);
        acc = acc + re * re;
        if constexpr (A::is_complex) {
            const R im = A::im(x);
            acc = acc + im * im;
        }
    }
}
}  // namespace detail
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

// The first i in [0, n) with pred(i), or -1 (uniform). Each item stops at its first owned hit
// (owned indices ascend), then one min-reduction. The x overload tests pred(x(i)).
template <Scope S = Scope::Own, class Ex, class P> BATCHLAS_DEVICE_INLINE int find_first(const Ex& ex, int n, P&& pred) {
    int first = INT_MAX;
    const int step = detail::size_of(ex);
    for (int i = ex.lid(); i < n; i += step)
        if (pred(i)) {
            first = i;
            break;
        }
    first = detail::scope_reduce<S>(ex, first, sycl::minimum<int>());
    return first == INT_MAX ? -1 : first;
}
template <Scope S = Scope::Own, class Ex, class X, class P>
    requires(requires(const X& x) { x.size(); x(0); })
BATCHLAS_DEVICE_INLINE int find_first(const Ex& ex, const X& x, P&& pred) {
    return find_first<S>(ex, x.size(), [&](int i) { return pred(x(i)); });
}

// argmax over i of key(x(i)): {index, key} (ArgMax, tie::, nan::: reduce.hh). Tie: alg::TieLowest | alg::TieHighest index among equal
// keys. NaN: alg::NanFirst (the lowest index holding a NaN key wins, as iamax), alg::NanSkip (NaN keys
// never win; all NaN -> {-1, -inf-like}). Key: alg::KeyAbs1 (|re| + |im|, BLAS), alg::KeyAbs (hypot),
// alg::KeyReal (re). Uniform result; n <= 0 -> index -1.
namespace alg {
struct KeyAbs1 {};
struct KeyAbs {};
struct KeyReal {};
}  // namespace alg
template <class Tie = alg::TieLowest, class NaN = alg::NanFirst, class Key = alg::KeyAbs1, Scope S = Scope::Own, class Ex, class X>
BATCHLAS_DEVICE_INLINE ArgMax<real_t<elem_t<X>>> argmax(const Ex& ex, const X& x) {
    using T = elem_t<X>;
    using R = real_t<T>;
    constexpr bool low = std::is_same_v<Tie, alg::TieLowest>;
    static_assert(low || std::is_same_v<Tie, alg::TieHighest>, "argmax: alg::TieLowest or alg::TieHighest");
    static_assert(std::is_same_v<NaN, alg::NanFirst> || std::is_same_v<NaN, alg::NanSkip>, "argmax: alg::NanFirst or alg::NanSkip");
    auto keyf = [](const T& v) -> R {
        if constexpr (std::is_same_v<Key, alg::KeyAbs1>) return abs1(v);
        else if constexpr (std::is_same_v<Key, alg::KeyAbs>) return modulus(v);
        else return Arith<T>::re(v);
    };
    const R lowest = -std::numeric_limits<R>::infinity();
    R best = lowest;
    int bi = low ? INT_MAX : -1, nani = INT_MAX;
    for_each_owned(ex, x.size(), [&](int i) {
        const R a = keyf(x(i));
        if (sycl::isnan(a)) {
            if constexpr (std::is_same_v<NaN, alg::NanFirst>) nani = i < nani ? i : nani;
        } else if (a > best || (a == best && !low)) {
            best = a;
            bi = i;
        } else if (low && a == best && bi == INT_MAX) {
            bi = i;  // the first -inf key
        }
    });
    if constexpr (std::is_same_v<NaN, alg::NanFirst>) {
        nani = detail::scope_reduce<S>(ex, nani, sycl::minimum<int>());
        if (nani != INT_MAX) return {nani, std::numeric_limits<R>::quiet_NaN()};
    }
    const R m = detail::scope_reduce<S>(ex, best, sycl::maximum<R>());
    int idx;
    if constexpr (low) {
        idx = detail::scope_reduce<S>(ex, best == m ? bi : INT_MAX, sycl::minimum<int>());
        idx = idx == INT_MAX ? -1 : idx;
    } else {
        idx = detail::scope_reduce<S>(ex, best == m ? bi : -1, sycl::maximum<int>());
    }
    return {idx, m};
}

// max key over x with a NaN policy: alg::NanPropagate (any NaN -> NaN, as amax) or alg::NanSkip
// (fmax: NaN ignored, the bdsqr / tridiag rule).
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

// fs(i)... in argument order for every owned i < n: several outputs in ONE owned pass.
template <class Ex, class... Fs> BATCHLAS_DEVICE_INLINE void map_many(const Ex& ex, int n, const Fs&... fs) {
    for_each_owned(ex, n, [&](int i) { (fs(i), ...); });
}

// x(i) := alpha (a tag or a value).
template <class Ex, class AS, class X> BATCHLAS_DEVICE_INLINE void fill(const Ex& ex, AS alpha, const X& x) {
    using T = elem_t<X>;
    const T v = tag_mul(as_tag(alpha), Arith<T>::one());
    for_each_owned(ex, x.size(), [&](int i) { x(i) = v; });
}

// y(2k) := a(k) for k < a.size(), y(2k + 1) := b(k) for k < b.size() (the perfect shuffle;
// b.size() is a.size() or a.size() - 1), one owned loop over k.
template <class Ex, class A, class B, class Y> BATCHLAS_DEVICE_INLINE void interleave(const Ex& ex, const A& a, const B& b, const Y& y) {
    const int nb = b.size();
    for_each_owned(ex, a.size(), [&](int k) {
        y(2 * k) = a(k);
        if (k < nb) y(2 * k + 1) = b(k);
    });
}

// x := x reversed, in place: item owning i < n / 2 swaps x(i) and x(n - 1 - i) (Thread: the
// serial two-pointer loop). LaneVec (operand.hh) form: each window [lo, hi) reversed by one
// shuffle when on (uniform); items outside it keep their value.
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
template <class Ex, class T> BATCHLAS_DEVICE_INLINE void reverse(const Ex& ex, const LaneVec<T>& x, bool on) {
    const int l = ex.lid();
    const int src = (on && l >= x.lo && l < x.hi) ? (x.lo + x.hi - 1 - l) : l;
    x.v = ex.shuffle(static_cast<std::remove_cv_t<T>>(x.v), src);
}

// The number of x(i) <= v on an ascending x (C++ upper_bound's offset), by bisection on one item.
template <class X, class R> BATCHLAS_DEVICE_INLINE int upper_bound(const Thread&, const X& x, R v) {
    int lo = 0, hi = x.size();
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (!(v < x(mid))) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// x := x / d (LAPACK ?RSCL): one reciprocal and a multiply when 1/d is finite and nonzero, else a
// division per element (SafeRecip, scalar.hh); d == 0 divides (Inf / NaN, as LAPACK).
template <class Ops = arith::Fma, class Ex, class T, class X> BATCHLAS_DEVICE_INLINE void rscl(const Ex& ex, const T& d, const X& x) {
    using E = elem_t<X>;
    const auto s = SafeRecip<E, Ops>::make(static_cast<E>(d));
    for_each_owned(ex, x.size(), [&](int i) {
        const E v = x(i);
        x(i) = s.use_mul ? arith_t<Ops, E>::mul(v, s.r) : arith_t<Ops, E>::div(v, s.d);
    });
}

// This item's partial x^T y (Conj: x^H y) over its owned elements; no reduction (the caller
// combines partials, e.g. through a Scratch exchange shared with other sums).
template <bool Conj = false, class Ops = arith::Explicit, class Ex, class X, class Y>
BATCHLAS_DEVICE_INLINE auto dot_partial(const Ex& ex, const X& x, const Y& y) {
    return detail::dot_partial<Conj, Ops>(ex, x, y);
}

// acc += x(t) y(t) for t < N with kk0 + t < k, in index order: one item's dot over a staged slab.
template <int N, class Ops = arith::Native, class T, class VX, class VY>
BATCHLAS_DEVICE_INLINE void dot_slab(T& acc, const VX& x, const VY& y, int kk0, int k) {
    using A = arith_t<Ops, T>;
    for (int t = 0; t < N && kk0 + t < k; ++t) A::mac(acc, x(t), y(t));
}

// prod_i f(i) as mant * 2^expo (mant in [0.5, 1) unless the product is 0, Inf or NaN), free of
// over/underflow: each item renormalises its running product by frexp, then the mantissas
// combine 32 items at a time (2^-32 <= a product of 32 mantissas <= 1) with renormalisation.
template <class R> struct ScaledProd {
    R mant;
    std::int32_t expo;
};
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
// sqrt(|mant| 2^expo) without forming the power.
template <class R> BATCHLAS_DEVICE_INLINE R sqrt_scaled(ScaledProd<R> q) {
    const std::int32_t half = q.expo >> 1;  // floor(expo / 2)
    const std::int32_t rem = q.expo - 2 * half;
    const R mag = sycl::sqrt(sycl::fabs(q.mant) * (rem ? R(2) : R(1)));
    return sycl::ldexp(mag, half);
}

// find_first with the miss value named (none instead of -1); LaneVec form: the smallest lid in
// [x.lo, x.hi) whose element satisfies pred, else none (a min butterfly, masks ascending).
template <Scope S = Scope::Own, class Ex, class P> BATCHLAS_DEVICE_INLINE int find_first(const Ex& ex, int n, P&& pred, int none) {
    const int f = find_first<S>(ex, n, pred);
    return f < 0 ? none : f;
}
template <class Ex, class T, class P> BATCHLAS_DEVICE_INLINE int find_first(const Ex& ex, const LaneVec<T>& x, int none, P&& pred) {
    const int l = ex.lid();
    const int cand = (l >= x.lo && l < x.hi && pred(static_cast<std::remove_cv_t<T>>(x.v))) ? l : none;
    return detail::butterfly_inline<1, Ex::size, alg::BflyAsc>(ex, cand, sycl::minimum<int>());
}

// argmax over x(i), i in [r.lo, r.hi) except i == skip, through a Group-scope exchange: per-item
// candidates merged by argmax_merge<Tie, NaN> in ONE allreduce (reduce.hh) over Scratch<ArgMax<R>>
// (scratch_elements<reduce_op::Reduce, Ex, ArgMax<R>>() slots). Uniform; no candidate -> index -1.
template <class Tie = alg::TieLowest, class NaN = alg::NanFirst, class Key = alg::KeyAbs1, class Ex, class R, int Slots, class X>
BATCHLAS_DEVICE_INLINE ArgMax<R> argmax(const Ex& ex, Scratch<ArgMax<R>, Slots> scr, const X& x, Range r, int skip = -1) {
    using T = elem_t<X>;
    auto keyf = [](const T& v) -> R {
        if constexpr (std::is_same_v<Key, alg::KeyAbs1>) return abs1(v);
        else if constexpr (std::is_same_v<Key, alg::KeyAbs>) return modulus(v);
        else return Arith<T>::re(v);
    };
    ArgMax<R> best{-1, R(0)};
    for_each_owned(ex, r, [&](int i) {
        if (i != skip) best = argmax_merge<Tie, NaN>(best, ArgMax<R>{i, keyf(x(i))});
    });
    return allreduce(ex, scr, best, ArgMaxMerge<Tie, NaN>{});
}

// ---- Cyclic rows of a static extent N (N % size == 0): item l owns rows l + k*size ----------
// f(k, row) for the item's N / size rows, k a compile-time Frag slot; no bound check.
// U = alg::UnrollFull is a full unroll pragma, alg::UnrollDefault none.
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

// x[k] = v(l + k*size) for rows < n, else 0.
template <class Ex, class V, class T, int K> BATCHLAS_DEVICE_INLINE void load_owned_masked(const Ex& ex, const V& v, int n, Frag<T, K>& x) {
    for_each_owned_frag<K * Ex::size>(ex, [&](int k, int r) { x[k] = (r < n) ? static_cast<T>(v(r)) : T(0); });
}
// x = e_j restricted to the item's rows.
template <class Ex, class T, int K> BATCHLAS_DEVICE_INLINE void set_unit_owned(const Ex& ex, int j, Frag<T, K>& x) {
    for_each_owned_frag<K * Ex::size>(ex, [&](int k, int r) { x[k] = (r == j) ? T(1) : T(0); });
}
// sum_k conj(x[k]) y[k] over the item's fragment (a partial dotc; reduce it with allreduce).
template <class Ops = arith::Native, class T, int K> BATCHLAS_DEVICE_INLINE T dotc_owned(const Frag<T, K>& x, const Frag<T, K>& y) {
    T acc = T(0);
#pragma unroll
    for (int k = 0; k < K; ++k) {
        if constexpr (detail::is_native_v<Ops>) acc = acc + conj(x[k]) * y[k];
        else arith_t<Ops, T>::mac_conj(acc, x[k], y[k]);
    }
    return acc;
}
// x[k] = u(l + k*size), y[k] = v(l + k*size) and the partial dotc sum_k conj(x[k]) y[k] in one
// pass (x, y: Frag or T[K]). Two loads plus dotc_owned gave complex T extra BRA.
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
// y[k] := y[k] - x[k] a (one classical Gram-Schmidt projection step).
template <class Ops = arith::Native, class T, int K> BATCHLAS_DEVICE_INLINE void sub_mul(Frag<T, K>& y, const Frag<T, K>& x, T a) {
#pragma unroll
    for (int k = 0; k < K; ++k) {
        if constexpr (detail::is_native_v<Ops>) y[k] = y[k] - x[k] * a;
        else arith_t<Ops, T>::msub(y[k], x[k], a);
    }
}
// sum of |x[k]|^2 over the item's rows < n (each abs2 its own rounding, as the hand loop).
template <class Ex, class T, int K> BATCHLAS_DEVICE_INLINE real_t<T> sumsq_owned(const Ex& ex, const Frag<T, K>& x, int n) {
    real_t<T> acc = real_t<T>(0);
    for_each_owned_frag<K * Ex::size>(ex, [&](int k, int r) {
        if (r < n) acc += abs2(x[k]);
    });
    return acc;
}
// sumsq of a Cyclic register vector: x[k] is row l + k*size of the item's share, rows >= n
// skipped (read nothing); Alg as sumsq (alg::Unfused: each abs2 its own rounding), reduced
// over scope S.
template <class Alg = alg::Fused, Scope S = Scope::Own, class Ex, class T, int N>
BATCHLAS_DEVICE_INLINE real_t<T> sumsq(const Ex& ex, const Frag<T, N, Cyclic>& x, int n) {
    real_t<T> acc = real_t<T>(0);
    for_each_owned_frag<N * Ex::size>(ex, [&](int k, int r) {
        if (r < n) detail::acc_abs2<Alg>(acc, x[k]);
    });
    return allreduce<S>(ex, acc);
}

// Lane l returns ||a(:, l)||^2 for the P = Ex::size columns of a, whose static row count (a.m,
// an Ext) is Cyclic over the executor: per-lane partial sums, then reduce_scatter<P> (log2 P
// exchanges; core sumsq over P columns would be P all-reduces).
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
