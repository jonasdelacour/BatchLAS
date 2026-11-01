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

// The primitive: f(i) for every i < n this item owns.
template <class Ex, class F> BATCHLAS_DLA_INLINE void for_each_owned(const Ex& ex, int n, F&& f) {
    for (int i = ex.lid(); i < n; i += Ex::size) f(i);
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
template <class Ex, class A, class X, class Y> BATCHLAS_DLA_INLINE void axpy(const Ex& ex, A a_, const X& x, const Y& y) {
    using T = elem_t<Y>;
    using S = tag_t<A>;
    const S a = as_tag(a_);
    if constexpr (std::is_same_v<S, Zero>) return;
    for_each_owned(ex, y.size(), [&](int i) {
        if constexpr (std::is_same_v<S, One>) y(i) = Arith<T>::add(y(i), x(i));
        else if constexpr (std::is_same_v<S, NegOne>) y(i) = Arith<T>::sub(y(i), x(i));
        else if constexpr (!std::is_same_v<S, Zero>) Arith<T>::mac(y(i), static_cast<T>(a.v), x(i));
    });
}

// x := a x
template <class Ex, class A, class X> BATCHLAS_DLA_INLINE void scal(const Ex& ex, A a_, const X& x) {
    using T = elem_t<X>;
    const auto a = as_tag(a_);
    for_each_owned(ex, x.size(), [&](int i) { x(i) = tag_mul(a, static_cast<T>(x(i))); });
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
    if constexpr (!is_work_group_v<Ex> || Ex::nteams == 1) {
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
template <bool Conj, class Ex, class X, class Y> BATCHLAS_DLA_INLINE auto dot_partial(const Ex& ex, const X& x, const Y& y) {
    using T = elem_t<X>;
    T acc = Arith<T>::zero();
    for_each_owned(ex, x.size(), [&](int i) {
        if constexpr (Conj) Arith<T>::mac_conj(acc, x(i), y(i));
        else Arith<T>::mac(acc, x(i), y(i));
    });
    return acc;
}
}  // namespace detail

// x^T y (no conjugation)
template <Scope S = Scope::Auto, class Ex, class X, class Y> BATCHLAS_DLA_INLINE auto dot(const Ex& ex, const X& x, const Y& y) {
    return detail::scope_reduce<S>(ex, detail::dot_partial<false>(ex, x, y));
}
// x^H y
template <Scope S = Scope::Auto, class Ex, class X, class Y> BATCHLAS_DLA_INLINE auto dotc(const Ex& ex, const X& x, const Y& y) {
    return detail::scope_reduce<S>(ex, detail::dot_partial<true>(ex, x, y));
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
// {xs^T y...} (Conj: xs^H y) for operands of different types (e.g. a column and its
// unit_head), one packed exchange: latrd's gamma/delta pair.
template <bool Conj = false, class Ex, class T, int Slots, class Y, class... Xs>
BATCHLAS_DLA_INLINE std::array<T, sizeof...(Xs)> dot_many(const Ex& ex, Scratch<T, Slots> scr, const Y& y, const Xs&... xs) {
    constexpr int K = sizeof...(Xs);
    std::array<T, K> acc;
#pragma unroll
    for (int k = 0; k < K; ++k) acc[k] = Arith<T>::zero();
    for_each_owned(ex, y.size(), [&](int i) {
        const T yi = y(i);
        int k = 0;
        ((Conj ? Arith<T>::mac_conj(acc[k++], xs(i), yi) : Arith<T>::mac(acc[k++], xs(i), yi)), ...);
    });
    return detail::packed_sum<K>(ex, scr, acc);
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
