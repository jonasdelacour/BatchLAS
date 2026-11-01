// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 extension for lanczos.cc, ortho.cc, ritz_values.cc: executors whose width is the
// launch's local range, and the level-1 / elementwise ops those kernels are sequences of.
// -ffp-contract=on fuses a*b+c only inside one expression, so the rounding is a contract:
//   alg::Fused    acc += x*y, one FMA per term (the core dot/dotc form).
//   alg::Unfused  products rounded, then added: the legacy SLM-staged joint_reduce, minus the SLM.

#include <cassert>
#include <type_traits>

#include <batchlas/blas/device/v2/dla.hh>

namespace batchlas::device::v2 {

namespace alg {
struct Fused {};
struct Unfused {};
}  // namespace alg

// ---- executors with a run-time width --------------------------------------------------------
// Local range min(n, device max). No static `size`: gemv/ger/symv/packed sums do not compile on
// them; Cyclic level-1 ops and reductions run through the for_each_owned overloads below.
namespace detail {
struct RuntimeGroup {
    static constexpr bool sub_whole = false;
    static constexpr Scope natural_scope = Scope::Group;
    sycl::nd_item<1> it;

    BATCHLAS_DLA_INLINE explicit RuntimeGroup(sycl::nd_item<1> i) : it(i) {}
    BATCHLAS_DLA_INLINE int lid() const { return static_cast<int>(it.get_local_linear_id()); }
    BATCHLAS_DLA_INLINE int width() const { return static_cast<int>(it.get_local_range(0)); }
    BATCHLAS_DLA_INLINE int lane() const { return static_cast<int>(it.get_sub_group().get_local_linear_id()); }
    BATCHLAS_DLA_INLINE int team() const { return static_cast<int>(it.get_sub_group().get_group_linear_id()); }
    BATCHLAS_DLA_INLINE bool leader() const { return lid() == 0; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return sycl_reduce(it.get_sub_group(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return sycl_reduce(it.get_group(), v, op);
    }
    // sycl::group_barrier: the barrier and fence the legacy kernels issue.
    BATCHLAS_DLA_INLINE void barrier() const { sycl::group_barrier(it.get_group()); }
    template <class V> BATCHLAS_DLA_INLINE Published<V> publish(V v) const {
        barrier();
        return {v};
    }
};
}  // namespace detail

// The whole work-group, Cyclic: index i is owned by lid == i mod width.
struct DynGroup : detail::RuntimeGroup {
    using detail::RuntimeGroup::RuntimeGroup;
};
// One index per item (lid == i): the launch covers n, as in the WG = k kernels; no stride loop.
struct ItemGroup : detail::RuntimeGroup {
    using detail::RuntimeGroup::RuntimeGroup;
};

// Found by ADL from every core level-1 op, and more specialised than the core template.
template <class F> BATCHLAS_DLA_INLINE void for_each_owned(const DynGroup& ex, int n, F&& f) {
    const int w = ex.width();
    for (int i = ex.lid(); i < n; i += w) f(i);
}
template <class F> BATCHLAS_DLA_INLINE void for_each_owned(const ItemGroup& ex, int n, F&& f) {
    assert(n <= ex.width());
    const int i = ex.lid();
    if (i < n) f(i);
}

// ---- operands ------------------------------------------------------------------------------
// The main diagonal of a column-major Mat (stride ld + 1).
template <class T, class R, class C, class Ld, Mem M, class L>
BATCHLAS_DLA_INLINE Vec<T, Dyn, Dyn, M> diag(const Mat<T, R, C, Ld, M, L>& a) {
    const int d = a.rows() < a.cols() ? a.rows() : a.cols();
    return {a.p, Dyn{d}, Dyn{int(a.ld) + 1}};
}

// x(i) = f(i), computed on read (never stored, so any item may read any index unpublished).
template <class F> struct Generated {
    using value_type = std::remove_cvref_t<decltype(std::declval<const F&>()(0))>;
    F f;
    int n;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const { return f(i); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class F> BATCHLAS_DLA_INLINE Generated<F> generated(int n, F f) { return {f, n}; }

template <class V> inline constexpr bool is_generated_v = false;
template <class F> inline constexpr bool is_generated_v<Generated<F>> = true;

// ---- level 1 / elementwise -----------------------------------------------------------------
// y(i) := f(i)
template <class Ex, class Y, class F> BATCHLAS_DLA_INLINE void generate(const Ex& ex, const Y& y, F&& f) {
    for_each_owned(ex, y.size(), [&](int i) { y(i) = f(i); });
}

// y := y + (a x + b w), a x fused into the pair's sum: the Lanczos update's rounding.
template <class Ex, class T, class X, class W, class Y>
BATCHLAS_DLA_INLINE void axpy2(const Ex& ex, T a, const X& x, T b, const W& w, const Y& y) {
    using Ar = Arith<elem_t<Y>>;
    for_each_owned(ex, y.size(), [&](int i) {
        if constexpr (Ar::is_complex) y(i) = Ar::add(y(i), Ar::add(Ar::mul(a, x(i)), Ar::mul(b, w(i))));
        else y(i) = y(i) + (a * x(i) + b * w(i));
    });
}

// y(i) := x(i) / d, a true division per element (scal(1/d) rounds twice); d is real; y may be x.
template <class Ex, class R, class X, class Y> BATCHLAS_DLA_INLINE void scal_div(const Ex& ex, R d, const X& x, const Y& y) {
    using Ar = Arith<elem_t<Y>>;
    static_assert(std::is_same_v<R, real_t<elem_t<Y>>>, "v2::scal_div: the divisor is real");
    for_each_owned(ex, y.size(), [&](int i) {
        const auto v = static_cast<elem_t<Y>>(x(i));
        if constexpr (Ar::is_complex) y(i) = Ar::make(Ar::re(v) / d, Ar::im(v) / d);
        else y(i) = v / d;
    });
}

// x(i) := x(i) + s; on diag(A) this is A + s I.
template <class Ex, class S, class X> BATCHLAS_DLA_INLINE void add_scalar(const Ex& ex, S s, const X& x) {
    using T = elem_t<X>;
    for_each_owned(ex, x.size(), [&](int i) { x(i) = Arith<T>::add(x(i), static_cast<T>(s)); });
}

// sum |x_i|^2, real (Alg: see the top of this file).
template <class Alg = alg::Fused, Scope S = Scope::Auto, class Ex, class X> BATCHLAS_DLA_INLINE auto sumsq(const Ex& ex, const X& x) {
    using Ar = Arith<elem_t<X>>;
    using R = real_t<elem_t<X>>;
    static_assert(std::is_same_v<Alg, alg::Fused> || std::is_same_v<Alg, alg::Unfused>, "v2::sumsq: alg::Fused or alg::Unfused");
    R acc = R(0);
    for_each_owned(ex, x.size(), [&](int i) {
        const auto v = static_cast<elem_t<X>>(x(i));
        if constexpr (std::is_same_v<Alg, alg::Fused>) {
            Arith<R>::mac(acc, Ar::re(v), Ar::re(v));
            if constexpr (Ar::is_complex) Arith<R>::mac(acc, Ar::im(v), Ar::im(v));
        } else if constexpr (Ar::is_complex) {
            const R re2 = Arith<R>::mul(Ar::re(v), Ar::re(v));
            const R im2 = Arith<R>::mul(Ar::im(v), Ar::im(v));
            acc = Arith<R>::add(acc, Arith<R>::add(re2, im2));
        } else {
            acc = Arith<R>::add(acc, Arith<R>::mul(v, v));
        }
    });
    return detail::scope_reduce<S>(ex, acc);
}

// x^T y with every product rounded before the sum (alg::Unfused); core dot is the fused form.
template <Scope S = Scope::Auto, class Ex, class X, class Y> BATCHLAS_DLA_INLINE auto dot_unfused(const Ex& ex, const X& x, const Y& y) {
    using T = elem_t<X>;
    return reduce<S>(ex, x.size(), [&](int i) { return Arith<T>::mul(x(i), y(i)); });
}

// A := diag(r) A diag(c) (LAPACK laqge): a(i,j) * (r(i) c(j)), rows owned, r and c real; c is
// read by every item, so it is Published or Generated.
template <class Ex, class Rv, class Cv, class MA> BATCHLAS_DLA_INLINE void laqge(const Ex& ex, const Rv& r, const Cv& cp, const MA& a) {
    using Ar = Arith<typename MA::value_type>;
    static_assert(is_published_v<Cv> || is_generated_v<Cv>, "v2::laqge: c is read across owners; publish it");
    const auto& c = detail::unpub(cp);
    const int n = a.cols();
    for_each_owned(ex, a.rows(), [&](int i) {
        const auto ri = r(i);
        for (int j = 0; j < n; ++j) {
            const auto s = ri * c(j);
            if constexpr (Ar::is_complex) a(i, j) = Ar::make(Ar::re(a(i, j)) * s, Ar::im(a(i, j)) * s);
            else a(i, j) = a(i, j) * s;
        }
    });
}

// B := op(A), A m x n, distributed in A's column-major order (i = lin mod m, j = lin / m).
template <Transpose TR, class Ex, class MA, class MB> BATCHLAS_DLA_INLINE void lacpy(const Ex& ex, const MA& a, const MB& b) {
    using T = typename MA::value_type;
    const int m = a.rows();
    for_each_owned(ex, m * a.cols(), [&](int lin) {
        const int i = lin % m, j = lin / m;
        const T v = a(i, j);
        if constexpr (TR == Transpose::NoTrans) b(i, j) = v;
        else if constexpr (TR == Transpose::Trans) b(j, i) = v;
        else b(j, i) = Arith<T>::conj(v);
    });
}

}  // namespace batchlas::device::v2
