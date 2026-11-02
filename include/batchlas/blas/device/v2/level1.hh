// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 level 1. Every op distributes over the executor's Cyclic layout (i owned by
// lid == i mod size), so two ops on the same executor touch an element from the same item and
// compose with NO barrier. No op here contains a work-group barrier except the packed
// Scratch exchange, which names it in its signature.

#include <array>
#include <climits>
#include <limits>
#include <type_traits>
#include <utility>

#include <batchlas/blas/device/v2/operand.hh>

namespace batchlas::device::v2 {

template <class V> using elem_t = std::remove_cvref_t<decltype(std::declval<const V&>()(0))>;

// The primitive: f(i) for every i this item owns, of [0, n) or of a Range [lo, hi) in parent
// coordinates (i owned by lid == (i - lo) mod size, as an op on x.tail(lo): the two compose with
// no barrier). A Range keeps the hand loop's bounds: a 0-based loop over a tail view unrolls
// differently (latrd P3, gebrd, stein). U is the unroll policy of this loop (core.hh unroll::),
// emitted as loop metadata; whether ptxas honours a PTX nounroll needs a gate check per use.
namespace detail {
template <class U, class F> BATCHLAS_DLA_INLINE void strided_loop(int lo, int hi, int step, F& f) {
    if constexpr (std::is_same_v<U, unroll::Default>) {
        for (int i = lo; i < hi; i += step) f(i);
    } else if constexpr (std::is_same_v<U, unroll::None>) {
#pragma clang loop unroll(disable)
        for (int i = lo; i < hi; i += step) f(i);
    } else if constexpr (std::is_same_v<U, unroll::Full>) {
#pragma clang loop unroll(full)
        for (int i = lo; i < hi; i += step) f(i);
    } else {
        constexpr int N = U::count;
        static_assert(N >= 1, "v2::for_each_owned: unroll::Count<N> needs N >= 1");
#pragma clang loop unroll_count(N)
        for (int i = lo; i < hi; i += step) f(i);
    }
}
}  // namespace detail

template <class U = unroll::Default, class Ex, class F> BATCHLAS_DLA_INLINE void for_each_owned(const Ex& ex, int n, F&& f) {
    if constexpr (std::is_same_v<U, unroll::Default> && !is_runtime_v<Ex>) {
        for (int i = ex.lid(); i < n; i += Ex::size) f(i);  // the v2.0 spelling, SASS-pinned by the lab
    } else {
        detail::strided_loop<U>(ex.lid(), n, detail::size_of(ex), f);
    }
}
template <class U = unroll::Default, class Ex, class F> BATCHLAS_DLA_INLINE void for_each_owned(const Ex& ex, Range r, F&& f) {
    if constexpr (std::is_same_v<U, unroll::Default> && !is_runtime_v<Ex>) {
        for (int i = r.lo + ex.lid(); i < r.hi; i += Ex::size) f(i);
    } else {
        detail::strided_loop<U>(r.lo + ex.lid(), r.hi, detail::size_of(ex), f);
    }
}
// for_each_owned(ex, Range{first, end}, f), the P3 spelling.
template <class U = unroll::Default, class Ex, class F>
BATCHLAS_DLA_INLINE void for_each_owned(const Ex& ex, int first, int end, F&& f) {
    for_each_owned<U>(ex, Range{first, end}, f);
}
// n == the executor's size: each item owns exactly i = lid, no loop and no guard (the caller
// guarantees n == size; krylov's ItemGroup).
template <class Ex, class F> BATCHLAS_DLA_INLINE void for_each_owned_exact(const Ex& ex, [[maybe_unused]] int n, F&& f) {
    f(ex.lid());
}

// f(r, c) for every element of an m x n column-major grid this item owns: flat e = lid + s *
// size, so consecutive items take consecutive rows (loads coalesce) and per-item partial sums
// keep the legacy grid-stride order. Alg (core.hh alg::) is the hand kernel's index form:
//   Flat     q = e / m, (e - q m, q) per step, e in Idx (ptrdiff_t: v1's 64-bit index math)
//   DivMod   (e % m, e / m) per step
//   Stepped  one division per item; r += size, then subtract m while r >= m
//   Walk     one division per item; (r, c) += (size % m, size / m) with one carry per step
// On Thread every Alg is the column-major double loop; on a Grid2D, the item's one element.
template <class Alg = alg::Walk, class Idx = int, class Ex, class F>
BATCHLAS_DLA_INLINE void for_each_owned_2d(const Ex& ex, int m, int n, F&& f) {
    if constexpr (requires { ex.row(); ex.col(); }) {  // Grid2D (grid.hh)
        const int i = ex.row(), j = ex.col();
        if (i < m && j < n) f(i, j);
    } else if constexpr (is_thread_v<Ex>) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) f(i, j);
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
        static_assert(std::is_same_v<Alg, alg::Walk>, "v2::for_each_owned_2d: alg::Flat, DivMod, Stepped or Walk");
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

// out(i) = f(ins(i)...)
template <class Ex, class Out, class F, class... In>
BATCHLAS_DLA_INLINE void map(const Ex& ex, const Out& out, F&& f, const In&... ins) {
    for_each_owned(ex, out.size(), [&](int i) { out(i) = f(ins(i)...); });
}

// y := a x + b y; b == Zero never reads y.
template <class Ex, class A, class X, class B, class Y>
BATCHLAS_DLA_INLINE void axpby(const Ex& ex, A a_, const X& x, B b_, const Y& y) {
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
BATCHLAS_DLA_INLINE void axpy(const Ex& ex, A a_, const X& x, const Y& y) {
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
template <class Ops = arith::Explicit, class Ex, class A, class X> BATCHLAS_DLA_INLINE void scal(const Ex& ex, A a_, const X& x) {
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

template <class Ex, class X, class Y> BATCHLAS_DLA_INLINE void copy(const Ex& ex, const X& x, const Y& y) {
    for_each_owned(ex, y.size(), [&](int i) { y(i) = x(i); });
}

template <class Ex, class X, class Y> BATCHLAS_DLA_INLINE void swap(const Ex& ex, const X& x, const Y& y) {
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
template <Scope S, class Ex> constexpr Scope resolve_scope() { return S == Scope::Auto ? Ex::natural_scope : S; }

template <Scope S, class Ex, class T, class Op = sycl::plus<T>>
BATCHLAS_DLA_INLINE T scope_reduce(const Ex& ex, T v, Op op = {}) {
    if constexpr (resolve_scope<S, Ex>() == Scope::Sub) {
        static_assert(Ex::sub_whole, "v2: Scope::Sub on a WorkGroup leaves one partial per sub-group; use Scope::Group");
        return ex.team_reduce(v, op);
    } else {
        return ex.group_reduce(v, op);
    }
}

// K sums from ONE scratch exchange: per-team butterfly, lane 0 writes K values, one barrier,
// every item reads nteams*K in a fixed order (uniform result), one trailing barrier so the
// scratch may be reused.
template <int K, class Ex, class T, int Slots>
BATCHLAS_DLA_INLINE std::array<T, K> packed_sum(const Ex& ex, Scratch<T, Slots> scr, std::array<T, K> v) {
    if constexpr (is_runtime_v<Ex>) {
#pragma unroll
        for (int k = 0; k < K; ++k) v[k] = ex.group_reduce(v[k]);
        return v;
    } else if constexpr (!is_work_group_v<Ex> || Ex::nteams == 1) {
#pragma unroll
        for (int k = 0; k < K; ++k) v[k] = ex.team_reduce(v[k]);
        return v;
    } else {
        static_assert(Slots >= Ex::nteams * K, "v2: Scratch smaller than scratch_elements<op::ReduceN<K>, Ex, T>()");
#pragma unroll
        for (int k = 0; k < K; ++k) v[k] = ex.team_reduce(v[k]);
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
template <Scope S = Scope::Auto, class Ex, class F> BATCHLAS_DLA_INLINE auto reduce(const Ex& ex, int n, F&& f) {
    using T = std::remove_cvref_t<decltype(f(0))>;
    T acc = Arith<T>::zero();
    for_each_owned(ex, n, [&](int i) { acc = Arith<T>::add(acc, f(i)); });
    return detail::scope_reduce<S>(ex, acc);
}
template <class Ex, class T, int Slots, class F>
BATCHLAS_DLA_INLINE T reduce(const Ex& ex, Scratch<T, Slots> scr, int n, F&& f) {
    T acc = Arith<T>::zero();
    for_each_owned(ex, n, [&](int i) { acc = Arith<T>::add(acc, f(i)); });
    return detail::packed_sum<1>(ex, scr, std::array<T, 1>{acc})[0];
}

namespace detail {
template <bool Conj, class Ops = arith::Explicit, class Ex, class X, class Y>
BATCHLAS_DLA_INLINE auto dot_partial(const Ex& ex, const X& x, const Y& y) {
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
template <Scope S = Scope::Auto, class Ops = arith::Explicit, class Ex, class X, class Y>
BATCHLAS_DLA_INLINE auto dot(const Ex& ex, const X& x, const Y& y) {
    return detail::scope_reduce<S>(ex, detail::dot_partial<false, Ops>(ex, x, y));
}
// x^H y
template <Scope S = Scope::Auto, class Ops = arith::Explicit, class Ex, class X, class Y>
BATCHLAS_DLA_INLINE auto dotc(const Ex& ex, const X& x, const Y& y) {
    return detail::scope_reduce<S>(ex, detail::dot_partial<true, Ops>(ex, x, y));
}

namespace detail {
template <int K, bool Conj, class Ex, class Xs, class Y> BATCHLAS_DLA_INLINE auto dotn_partial(const Ex& ex, const Xs& xs, const Y& y) {
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
BATCHLAS_DLA_INLINE std::array<T, K> dotN(const Ex& ex, Scratch<T, Slots> scr, const std::array<X, K>& xs, const Y& y) {
    return detail::packed_sum<K>(ex, scr, detail::dotn_partial<K, Conj>(ex, xs, y));
}
namespace detail {
template <bool Conj, class Ops, class Ex, class Y, class... Xs> BATCHLAS_DLA_INLINE auto dot_many_partial(const Ex& ex, const Y& y, const Xs&... xs) {
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
BATCHLAS_DLA_INLINE std::array<T, sizeof...(Xs)> dot_many(const Ex& ex, Scratch<T, Slots> scr, const Y& y, const Xs&... xs) {
    return detail::packed_sum<sizeof...(Xs)>(ex, scr, detail::dot_many_partial<Conj, Ops>(ex, y, xs...));
}
// Without Scratch: one loop over y, then one scope-S reduction per x in argument order (on a
// WorkGroup, portable::reduce_over_group each: the legacy kernels' trees, bit-identical).
template <bool Conj = false, Scope S = Scope::Auto, class Ops = arith::Explicit, class Ex, class Y, class... Xs>
BATCHLAS_DLA_INLINE auto dot_many(const Ex& ex, const Y& y, const Xs&... xs) {
    auto acc = detail::dot_many_partial<Conj, Ops>(ex, y, xs...);
#pragma unroll
    for (int k = 0; k < int(sizeof...(Xs)); ++k) acc[k] = detail::scope_reduce<S>(ex, acc[k]);
    return acc;
}

// Without Scratch: K independent reductions of scope S (on a sub-group or partition these are
// shuffles only, so nothing is exchanged).
template <int K, bool Conj = false, Scope S = Scope::Auto, class Ex, class X, class Y>
BATCHLAS_DLA_INLINE auto dotN(const Ex& ex, const std::array<X, K>& xs, const Y& y) {
    auto acc = detail::dotn_partial<K, Conj>(ex, xs, y);
#pragma unroll
    for (int k = 0; k < K; ++k) acc[k] = detail::scope_reduce<S>(ex, acc[k]);
    return acc;
}

// sum |re| + |im|
template <Scope S = Scope::Auto, class Ex, class X> BATCHLAS_DLA_INLINE auto asum(const Ex& ex, const X& x) {
    using R = real_t<elem_t<X>>;
    R acc = R(0);
    for_each_owned(ex, x.size(), [&](int i) { acc += abs1(x(i)); });
    return detail::scope_reduce<S>(ex, acc);
}

// max |re| + |im|. NaN policy (NaN::Propagate): any NaN element makes the result NaN.
template <Scope S = Scope::Auto, class Ex, class X> BATCHLAS_DLA_INLINE auto amax(const Ex& ex, const X& x) {
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
template <Scope S = Scope::Auto, class Ex, class X> BATCHLAS_DLA_INLINE int iamax(const Ex& ex, const X& x) {
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
template <class R> struct ScaledSsq {
    R scale;
    R ssq;
};
template <Scope S = Scope::Auto, class Ex, class X> BATCHLAS_DLA_INLINE auto lassq(const Ex& ex, const X& x) {
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

// ||x||_2. Default alg::Scaled (lassq; Inf beats NaN, else NaN propagates); alg::Unscaled is
// sqrt(re(x^H x)), one reduction, overflow past sqrt(max).
template <class Alg = alg::Scaled, Scope S = Scope::Auto, class Ex, class X>
BATCHLAS_DLA_INLINE auto nrm2(const Ex& ex, const X& x) {
    using T = elem_t<X>;
    using R = real_t<T>;
    if constexpr (std::is_same_v<Alg, alg::Unscaled>) {
        return static_cast<R>(sycl::sqrt(Arith<T>::re(dotc<S>(ex, x, x))));
    } else {
        static_assert(std::is_same_v<Alg, alg::Scaled>, "v2::nrm2: Alg is alg::Scaled or alg::Unscaled");
        const auto q = lassq<S>(ex, x);
        if (sycl::isinf(q.scale)) return q.scale;
        return q.scale > R(0) ? q.scale * sycl::sqrt(q.ssq) : sycl::sqrt(q.ssq);
    }
}

}  // namespace batchlas::device::v2
