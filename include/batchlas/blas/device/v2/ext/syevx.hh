// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, syevx-family extension (general ops): RtGroup, a run-time-width executor;
// group_sync, allreduce; lazy / flipped / cols_flipped operands; abs2, scale_re, sub_scaled; sumsq,
// sumsq_mat; 2-D lacpy, laset, swap_mat; count_le. A 2-D op hands l = r + c*m to owner
// l mod width, so ops on one executor and shape compose with no barrier.

#include <type_traits>

#include <batchlas/blas/device/v2/dla.hh>

namespace batchlas::device::v2 {

// Width = the launch's local range (S = Group) or the sub-group's (S = Sub). No static `size`:
// gemv/ger/symv do not compile on it; Cyclic level-1 ops run through for_each_owned below.
template <Scope S = Scope::Group> struct RtGroup {
    static_assert(S == Scope::Sub || S == Scope::Group, "v2::RtGroup: Scope::Sub or Scope::Group");
    static constexpr bool sub_whole = (S == Scope::Sub);
    static constexpr Scope natural_scope = S;
    sycl::nd_item<1> it;

    BATCHLAS_DLA_INLINE int lane() const { return static_cast<int>(it.get_sub_group().get_local_linear_id()); }
    BATCHLAS_DLA_INLINE int team() const { return static_cast<int>(it.get_sub_group().get_group_linear_id()); }
    BATCHLAS_DLA_INLINE int nteams() const { return static_cast<int>(it.get_sub_group().get_group_linear_range()); }
    BATCHLAS_DLA_INLINE int lid() const {
        if constexpr (S == Scope::Sub) return lane();
        else return static_cast<int>(it.get_local_linear_id());
    }
    BATCHLAS_DLA_INLINE int width() const {
        if constexpr (S == Scope::Sub) return static_cast<int>(it.get_sub_group().get_local_linear_range());
        else return static_cast<int>(it.get_local_range(0));
    }
    BATCHLAS_DLA_INLINE bool leader() const { return lid() == 0; }
    BATCHLAS_DLA_INLINE RtGroup<Scope::Sub> sub() const { return {it}; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(it.get_sub_group(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        if constexpr (S == Scope::Sub) return team_reduce(v, op);
        else return detail::sycl_reduce(it.get_group(), v, op);
    }
};

// Found by ADL from every core level-1 op, and more specialised than the core template.
template <Scope S, class F> BATCHLAS_DLA_INLINE void for_each_owned(const RtGroup<S>& ex, int n, F&& f) {
    const int w = ex.width();
    for (int i = ex.lid(); i < n; i += w) f(i);
}

// sycl::group_barrier over the executor's group: the barrier (and fence) the legacy kernels
// issue, so a port keeps their BAR/MEMBAR count. WorkGroup::barrier fences local memory only.
template <class Ex> BATCHLAS_DLA_INLINE void group_sync(const Ex& ex) {
    if constexpr (Ex::sub_whole && !is_work_group_v<Ex>) sycl::group_barrier(ex.it.get_sub_group());
    else sycl::group_barrier(ex.it.get_group());
}

// v (this item's partial) reduced over the executor's scope with op; uniform result.
template <Scope S = Scope::Auto, class Ex, class T, class Op> BATCHLAS_DLA_INLINE T allreduce(const Ex& ex, T v, Op op) {
    return detail::scope_reduce<S>(ex, v, op);
}

template <class F> struct Lazy {
    using value_type = std::remove_cvref_t<decltype(std::declval<const F&>()(0))>;
    F f;
    int n;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const { return f(i); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
// x(i) = f(i) for i < n, evaluated on read (a fused elementwise expression, never stored).
template <class F> BATCHLAS_DLA_INLINE Lazy<F> lazy(int n, F f) { return {f, n}; }

// x(rev ? n-1-i : i): the order is a run-time choice, so one kernel serves both directions.
template <class V> struct Flipped {
    V v;
    bool rev;
    using value_type = elem_t<V>;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return v(rev ? v.size() - 1 - i : i); }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
};
template <class V> BATCHLAS_DLA_INLINE Flipped<V> flipped(V v, bool rev) { return {v, rev}; }

// A(i, rev ? n-1-j : j): the column order reversed at run time (LAPACK lapmt with a reversal).
template <class M> struct ColsFlipped {
    M a;
    bool rev;
    using value_type = typename M::value_type;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return a(i, rev ? a.cols() - 1 - j : j); }
    BATCHLAS_DLA_INLINE int rows() const { return a.rows(); }
    BATCHLAS_DLA_INLINE int cols() const { return a.cols(); }
};
template <class M> BATCHLAS_DLA_INLINE ColsFlipped<M> cols_flipped(M a, bool rev) { return {a, rev}; }

// re^2 + im^2 in real arithmetic (never the Annex-G operator*); norm_squared up to contraction.
#ifndef BATCHLAS_DLA_V2_INTEG_ABS2  // integ/dla-v2-all: defined by several ext headers
#define BATCHLAS_DLA_V2_INTEG_ABS2
template <class T> BATCHLAS_DLA_INLINE real_t<T> abs2(T x) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return A::re(x) * A::re(x) + A::im(x) * A::im(x);
    else return x * x;
}
#endif  // BATCHLAS_DLA_V2_INTEG_ABS2
// s x for a real s: two real products for complex x, no cross terms.
template <class T> BATCHLAS_DLA_INLINE T scale_re(real_t<T> s, T x) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::make(s * A::re(x), s * A::im(x));
    else return s * x;
}
// a - s x for a real s, one expression per component so it contracts to FMA (as `a - T(s) * x`
// does for real T); complex: no cross terms.
template <class T> BATCHLAS_DLA_INLINE T sub_scaled(T a, real_t<T> s, T x) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::make(A::re(a) - s * A::re(x), A::im(a) - s * A::im(x));
    else return a - s * x;
}

namespace detail {
// acc += |x|^2 as ONE expression: fp-contract=on fuses only within an expression, so a
// separate abs2() return would cost the FMA a legacy `acc += d * d` gets.
template <class R, class T> BATCHLAS_DLA_INLINE void acc_abs2(R& acc, T x) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) acc += A::re(x) * A::re(x) + A::im(x) * A::im(x);
    else acc += x * x;
}
}  // namespace detail

// sum |x_i|^2 over the executor's scope (unscaled lassq: overflows past sqrt(max)).
template <Scope S = Scope::Auto, class Ex, class X> BATCHLAS_DLA_INLINE auto sumsq(const Ex& ex, const X& x) {
    using R = real_t<elem_t<X>>;
    R acc = R(0);
    if constexpr (is_thread_v<Ex>) {
        // Serial and rolled in PTX. ptxas still unrolls it x4 (FilterResidual f32 LDG 8 -> 18,
        // total +8%); without the pragma LLVM unrolls too (LDG 82, +98%), as does an int64 counter.
#pragma nounroll
        for (int i = 0; i < x.size(); ++i) detail::acc_abs2(acc, x(i));
    } else {
        for_each_owned(ex, x.size(), [&](int i) { detail::acc_abs2(acc, x(i)); });
    }
    return detail::scope_reduce<S>(ex, acc);
}

// The m x n index space, column-major, flat over the executor: f(r, c) for every owned element.
template <class Ex, class F> BATCHLAS_DLA_INLINE void for_each_owned_2d(const Ex& ex, int m, int n, F&& f) {
    for_each_owned(ex, m * n, [&](int l) {
        const int c = l / m;
        f(l - c * m, c);
    });
}

// sum |A(i, j)|^2 (the squared Frobenius norm) over the executor's scope.
template <Scope S = Scope::Auto, class Ex, class MA> BATCHLAS_DLA_INLINE auto sumsq_mat(const Ex& ex, const MA& a) {
    using R = real_t<typename MA::value_type>;
    R acc = R(0);
    for_each_owned_2d(ex, a.rows(), a.cols(), [&](int r, int c) { detail::acc_abs2(acc, a(r, c)); });
    return detail::scope_reduce<S>(ex, acc);
}

// B := A over B's shape (dlacpy 'A'); A is any (i, j) view, e.g. cols_flipped(a.sub(...), rev).
template <class Ex, class MA, class MB> BATCHLAS_DLA_INLINE void lacpy(const Ex& ex, const MA& a, const MB& b) {
    for_each_owned_2d(ex, b.rows(), b.cols(), [&](int r, int c) { b(r, c) = a(r, c); });
}

namespace detail {
#ifndef BATCHLAS_DLA_V2_INTEG_DETAIL_TAG_VALUE  // integ/dla-v2-all: defined by several ext headers
#define BATCHLAS_DLA_V2_INTEG_DETAIL_TAG_VALUE
template <class T, class S> BATCHLAS_DLA_INLINE T tag_value(S s) {
    if constexpr (std::is_same_v<S, Zero>) return Arith<T>::zero();
    else if constexpr (std::is_same_v<S, One>) return Arith<T>::one();
    else if constexpr (std::is_same_v<S, NegOne>) return Arith<T>::neg(Arith<T>::one());
    else return static_cast<T>(s.v);
}
#endif  // BATCHLAS_DLA_V2_INTEG_DETAIL_TAG_VALUE
template <class S> inline constexpr bool is_fixed_tag_v =
    std::is_same_v<S, Zero> || std::is_same_v<S, One> || std::is_same_v<S, NegOne>;
}  // namespace detail

// A(i, j) := alpha off the diagonal, beta on it (dlaset 'A'). Equal fixed tags (Zero, Zero) are
// a plain fill with no diagonal test.
template <class Ex, class AS, class BS, class MA> BATCHLAS_DLA_INLINE void laset(const Ex& ex, AS alpha_, BS beta_, const MA& a) {
    using T = typename MA::value_type;
    const T off = detail::tag_value<T>(as_tag(alpha_));
    if constexpr (std::is_same_v<tag_t<AS>, tag_t<BS>> && detail::is_fixed_tag_v<tag_t<AS>>) {
        for_each_owned_2d(ex, a.rows(), a.cols(), [&](int r, int c) { a(r, c) = off; });
    } else {
        const T dia = detail::tag_value<T>(as_tag(beta_));
        for_each_owned_2d(ex, a.rows(), a.cols(), [&](int r, int c) { a(r, c) = r == c ? dia : off; });
    }
}

// A <-> B elementwise over A's shape (the 2-D dswap); B may be a re-indexed view of A's own
// storage as long as the two index sets are disjoint (a column reversal swaps halves).
template <class Ex, class MA, class MB> BATCHLAS_DLA_INLINE void swap_mat(const Ex& ex, const MA& a, const MB& b) {
    for_each_owned_2d(ex, a.rows(), a.cols(), [&](int r, int c) {
        const auto t = a(r, c);
        a(r, c) = b(r, c);
        b(r, c) = t;
    });
}

// #{i < n : x(i) <= v} for an ascending x (upper_bound, binary search, serial). `<=` is the
// Sturm-count predicate of stebz; NaN v counts nothing.
template <class X, class R> BATCHLAS_DLA_INLINE int count_le(const X& x, R v) {
    int lo = 0, hi = x.size();
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (x(mid) <= v) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

}  // namespace batchlas::device::v2
