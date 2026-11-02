// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 extension: general ops the divide-and-conquer eigensolver (stedc) needs and the
// core lacks. Nothing here is stedc-specific; every op is host-tested in
// tests/device_blas_v2_stedc_tests.cc.
//
//   DynWorkGroup<SG>         a work-group whose size is a launch parameter (tuned at run time)
//   exec_width, for_each_owned_in, uniform, sync_group
//   allreduce<S>             K independent reductions in a fixed order (no packed exchange)
//   reduce_op<S>             op-reduction of f(i) from a caller-given init (product, max, ...)
//   fold                     sycl::joint_reduce over a contiguous vector
//   nrm2_blue<Alg>           ||x||_2 by Blue's three accumulators (dnrm2.f90)
//   scal_inv                 x := x / a, a true division (not x * (1/a))
//   rot                      BLAS rot with real c, s
//   fill_cols<Alg>           A(:, j) := x for every column
//   prod_scaled, sqrt_scaled product of n factors as (mantissa, exponent), and its square root

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/util/group-collectives.hh>

namespace batchlas::device::v2 {

namespace alg {
struct Blue {};        // nrm2_blue: three scalar reductions (shuffles on a team executor)
struct BluePacked {};  // nrm2_blue: one sycl::vec<R, 3> work-group reduction (detail::nrm2)
struct Flat {};        // fill_cols: one owned loop over m * n (i = k % m, j = k / m)
}  // namespace alg

// ---- a work-group of run-time size --------------------------------------------------------
// For kernels whose local range is a tuned launch parameter. Cyclic ops stride by width();
// Scope::Group is portable::reduce_over_group. There is no static size, nteams or usable
// team_reduce, so level-2 ops and packed Scratch exchanges do not compile on it.
template <int SG = 32> struct DynWorkGroup {
    static constexpr int team_size = SG, sg_size = SG;
    static constexpr bool sub_whole = false;
    static constexpr Scope natural_scope = Scope::Group;
    sycl::nd_item<1> it;
    int n;

    BATCHLAS_DLA_INLINE explicit DynWorkGroup(sycl::nd_item<1> i) : it(i), n(static_cast<int>(i.get_local_range(0))) {}
    BATCHLAS_DLA_INLINE int width() const { return n; }
    BATCHLAS_DLA_INLINE int lid() const { return static_cast<int>(it.get_local_linear_id()); }
    BATCHLAS_DLA_INLINE int lane() const { return static_cast<int>(it.get_sub_group().get_local_linear_id()); }
    BATCHLAS_DLA_INLINE bool leader() const { return lid() == 0; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(it.get_group(), v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op = {}) const {
        static_assert(sizeof(T) == 0, "v2::DynWorkGroup: no team (Scope::Sub) or packed Scratch reduction on a run-time-size group");
        return v;
    }
    // sycl::group_barrier: the barrier the run-time-size kernels have always used.
    BATCHLAS_DLA_INLINE void barrier() const { sycl::group_barrier(it.get_group()); }
};

template <class Ex> inline constexpr bool is_dyn_work_group_v = false;
template <int SG> inline constexpr bool is_dyn_work_group_v<DynWorkGroup<SG>> = true;
template <class Ex> inline constexpr bool is_group_exec_v = is_work_group_v<Ex> || is_dyn_work_group_v<Ex>;
template <class Ex> inline constexpr bool is_partition_v = false;
template <int P, bool M, class Impl> inline constexpr bool is_partition_v<Partition<P, M, Impl>> = true;
template <class Ex> inline constexpr bool is_sub_group_exec_v = false;
template <int SG> inline constexpr bool is_sub_group_exec_v<SubGroup<SG>> = true;
template <int WG, int SG> inline constexpr bool is_sub_group_exec_v<Teams<WG, SG>> = true;

// Found by ADL from every level-1 op (they call for_each_owned unqualified), so map, copy,
// axpy, dot, reduce, lassq ... run on a DynWorkGroup unchanged.
template <int SG, class F> BATCHLAS_DLA_INLINE void for_each_owned(const DynWorkGroup<SG>& ex, int n, F&& f) {
    const int w = ex.width();
    for (int i = ex.lid(); i < n; i += w) f(i);
}

// Items sharing a Cyclic operand: the static size, or the launch's width.
template <class Ex> BATCHLAS_DLA_INLINE int exec_width(const Ex& ex) {
    if constexpr (requires { Ex::size; }) return Ex::size;
    else return ex.width();
}

// f(i) for every i in [lo, hi) this item owns: i = lo + lid + k * width.
template <class Ex, class F> BATCHLAS_DLA_INLINE void for_each_owned_in(const Ex& ex, int lo, int hi, F&& f) {
    const int w = exec_width(ex);
    for (int i = lo + ex.lid(); i < hi; i += w) f(i);
}

// The leader's v in every item. A reduced value is mathematically uniform but the compiler
// cannot prove it: a loop exit on it goes through uniform(), or a partition may diverge and
// deadlock in its next shuffle.
template <class Ex, class T> BATCHLAS_DLA_INLINE T uniform(const Ex& ex, T v) {
    if constexpr (is_thread_v<Ex>) {
        return v;
    } else if constexpr (is_group_exec_v<Ex>) {
        return batchlas::portable::group_broadcast(ex.it.get_group(), v, 0);
    } else if constexpr (is_partition_v<Ex>) {
        using Impl = std::remove_cvref_t<decltype(ex.part)>;
        if constexpr (sycl::is_group_v<Impl>) {
            const auto first = static_cast<std::uint32_t>(ex.part.get_local_linear_id()) & ~static_cast<std::uint32_t>(Ex::size - 1);
            return sycl::select_from_group(ex.part, v, first);
        } else {
            return select_from_group(ex.part, v, 0u);  // ADL: SubGroupPartition (src/kernels/common)
        }
    } else if constexpr (is_sub_group_exec_v<Ex>) {
        if constexpr (requires { ex.sg; }) return batchlas::portable::group_broadcast(ex.sg, v, 0);
        else return batchlas::portable::group_broadcast(ex.it.get_sub_group(), v, 0);
    } else {
        static_assert(sizeof(T) == 0, "v2::uniform: unknown executor");
        return v;
    }
}

// A barrier on a work-group executor; nothing on a team-wide one, whose items run one trip
// count together on data no other team touches.
template <class Ex> BATCHLAS_DLA_INLINE void sync_group(const Ex& ex) {
    if constexpr (is_dyn_work_group_v<Ex>) ex.barrier();
    else if constexpr (is_work_group_v<Ex>) ex.barrier(Mem::Local);
}

// ---- reductions ------------------------------------------------------------------------------

// K independent scope-S reductions in index order, each its own collective.
// A pack expansion (a braced list runs left to right), not a `#pragma unroll` loop: the pragma
// also turned each inlined work-group reduction's scratch loop from 8x to 32x unrolled.
namespace detail {
template <Scope S, class Ex, class T, std::size_t K, std::size_t... I>
BATCHLAS_DLA_INLINE std::array<T, K> allreduce_each(const Ex& ex, const std::array<T, K>& v, std::index_sequence<I...>) {
    return std::array<T, K>{detail::scope_reduce<S>(ex, v[I])...};
}
}  // namespace detail

template <Scope S = Scope::Auto, class Ex, class T, std::size_t K>
BATCHLAS_DLA_INLINE std::array<T, K> allreduce(const Ex& ex, std::array<T, K> v) {
    return detail::allreduce_each<S>(ex, v, std::make_index_sequence<K>{});
}

// acc = op(acc, f(i)) from init over the owned i < n, then a scope-S reduction with op. The
// product of n factors is reduce_op(ex, n, sycl::multiplies<T>(), T(1), f).
template <Scope S = Scope::Auto, class Ex, class Op, class T, class F>
BATCHLAS_DLA_INLINE T reduce_op(const Ex& ex, int n, Op op, T init, F&& f) {
    T acc = init;
    for_each_owned(ex, n, [&](int i) { acc = op(acc, static_cast<T>(f(i))); });
    return detail::scope_reduce<S>(ex, acc, op);
}

// sycl::joint_reduce over a contiguous x: item lid folds x(lid), x(lid + width), ... from op's
// identity (the Cyclic owner of each element, so an owned write before it needs no barrier),
// an empty x gives T{}, and the result is op(identity, reduction): portable::joint_reduce.
template <class Ex, class X, class Op> BATCHLAS_DLA_INLINE auto fold(const Ex& ex, const X& x, Op op) {
    using T = elem_t<X>;
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(x.inc)>, Unit>, "v2::fold: x must be contiguous (Unit inc)");
    static_assert(sycl::has_known_identity_v<Op, T>, "v2::fold: op needs a known identity");
    if constexpr (is_group_exec_v<Ex>) {
        return batchlas::portable::joint_reduce(ex.it.get_group(), x.p, x.p + x.size(), op);
    } else {
        if (x.size() == 0) return T{};
        const T id = sycl::known_identity_v<Op, T>;
        T part = id;
        for_each_owned(ex, x.size(), [&](int i) { part = op(part, static_cast<T>(x(i))); });
        return op(id, detail::scope_reduce<Scope::Auto>(ex, part, op));
    }
}

// ||x||_2 with Blue's three accumulators and constants (dnrm2.f90; a complex x accumulates |re|
// and |im| separately, as dznrm2). Default alg::BluePacked on a work-group executor (one
// sycl::vec<R, 3> reduction, src/math-helpers.hh detail::nrm2), alg::Blue elsewhere.
template <class Alg = void, Scope S = Scope::Auto, class Ex, class X> BATCHLAS_DLA_INLINE auto nrm2_blue(const Ex& ex, const X& x) {
    using T = elem_t<X>;
    using A = Arith<T>;
    using R = real_t<T>;
    using Al = std::conditional_t<std::is_void_v<Alg>, std::conditional_t<is_group_exec_v<Ex>, alg::BluePacked, alg::Blue>, Alg>;

    const int radix = std::numeric_limits<R>::radix;
    const int p = std::numeric_limits<R>::digits;
    const int emin = std::numeric_limits<R>::min_exponent;
    const int emax = std::numeric_limits<R>::max_exponent;
    auto floor_div = [](int a, int b) {
        int q = a / b;
        if (((a ^ b) < 0) && (a % b)) --q;
        return q;
    };
    auto ceil_div = [](int a, int b) {
        int q = a / b;
        if (((a ^ b) >= 0) && (a % b)) ++q;
        return q;
    };
    const R tsml = sycl::pown(static_cast<R>(radix), ceil_div(emin - 1, 2));
    const R tbig = sycl::pown(static_cast<R>(radix), floor_div(emax - p + 1, 2));
    const R ssml = sycl::pown(static_cast<R>(radix), -floor_div(emin - p, 2));
    const R sbig = sycl::pown(static_cast<R>(radix), -ceil_div(emax + p - 1, 2));

    R abig = R(0), amed = R(0), asml = R(0);
    auto acc = [&](R ax) {
        if (ax > tbig) {
            const R t = ax * sbig;
            abig += t * t;
        } else if (ax < tsml) {
            const R t = ax * ssml;
            asml += t * t;
        } else {
            amed += ax * ax;
        }
    };
    for_each_owned(ex, x.size(), [&](int i) {
        const T v = x(i);
        if constexpr (A::is_complex) {
            acc(sycl::fabs(A::re(v)));
            acc(sycl::fabs(A::im(v)));
        } else {
            acc(sycl::fabs(v));
        }
    });

    if constexpr (std::is_same_v<Al, alg::BluePacked>) {
        static_assert(is_group_exec_v<Ex>, "v2::nrm2_blue: alg::BluePacked is a work-group reduction");
        const sycl::vec<R, 3> t = batchlas::portable::reduce_over_group(ex.it.get_group(), sycl::vec<R, 3>{abig, amed, asml},
                                                                         sycl::plus<sycl::vec<R, 3>>());
        abig = t[0];
        amed = t[1];
        asml = t[2];
    } else {
        static_assert(std::is_same_v<Al, alg::Blue>, "v2::nrm2_blue: Alg is alg::Blue or alg::BluePacked");
        const auto t = allreduce<S>(ex, std::array<R, 3>{abig, amed, asml});
        abig = t[0];
        amed = t[1];
        asml = t[2];
    }

    const R zero = R(0), one = R(1), maxN = std::numeric_limits<R>::max();
    R scl = one, sumsq = zero;
    if (abig > zero) {
        if ((amed > zero) || (amed > maxN) || (amed != amed)) abig += (amed * sbig) * sbig;
        scl = one / sbig;
        sumsq = abig;
    } else if (asml > zero) {
        if ((amed > zero) || (amed > maxN) || (amed != amed)) {
            const R am = sycl::sqrt(amed);
            const R as = sycl::sqrt(asml) / ssml;
            const R ymin = sycl::fmin(am, as);
            const R ymax = sycl::fmax(am, as);
            scl = one;
            sumsq = ymax * ymax * (one + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = one / ssml;
            sumsq = asml;
        }
    } else {
        scl = one;
        sumsq = amed;
    }
    return scl * sycl::sqrt(sumsq);
}

// ---- level 1 -----------------------------------------------------------------------------------

// x := x / a for real a: one division per element (bit-different from scal(1 / a)).
template <class Ex, class R, class X> BATCHLAS_DLA_INLINE void scal_inv(const Ex& ex, R a, const X& x) {
    using T = elem_t<X>;
    using A = Arith<T>;
    static_assert(std::is_same_v<R, real_t<T>>, "v2::scal_inv: a is the real type of x");
    for_each_owned(ex, x.size(), [&](int i) {
        if constexpr (A::is_complex) {
            const T v = x(i);
            x(i) = A::make(A::re(v) / a, A::im(v) / a);
        } else {
            x(i) = x(i) / a;
        }
    });
}

// BLAS rot with real c, s: x := c x + s y, y := c y - s x.
template <class Ex, class X, class Y, class R> BATCHLAS_DLA_INLINE void rot(const Ex& ex, const X& x, const Y& y, R c, R s) {
    using T = elem_t<X>;
    using A = Arith<T>;
    static_assert(std::is_same_v<R, real_t<T>>, "v2::rot: c and s are the real type of x");
    for_each_owned(ex, x.size(), [&](int i) {
        const T xi = x(i), yi = y(i);
        if constexpr (A::is_complex) {
            x(i) = A::make(c * A::re(xi) + s * A::re(yi), c * A::im(xi) + s * A::im(yi));
            y(i) = A::make(c * A::re(yi) - s * A::re(xi), c * A::im(yi) - s * A::im(xi));
        } else {
            x(i) = c * xi + s * yi;
            y(i) = c * yi - s * xi;
        }
    });
}

// A(:, j) := x for every column j < A.cols(); x has A.rows() elements. alg::Flat: one owned
// loop over the m * n elements (one trip count, a % and a / per element); alg::Cyclic: the
// column loop outside, owned rows inside.
template <class Alg = alg::Cyclic, class Ex, class X, class MA> BATCHLAS_DLA_INLINE void fill_cols(const Ex& ex, const X& x, const MA& a) {
    const int m = a.rows(), n = a.cols();
    if constexpr (std::is_same_v<Alg, alg::Flat>) {
        for_each_owned(ex, m * n, [&](int k) {
            const int i = k % m;
            const int j = k / m;
            a(i, j) = x(i);
        });
    } else {
        static_assert(std::is_same_v<Alg, alg::Cyclic>, "v2::fill_cols: Alg is alg::Flat or alg::Cyclic");
        for (int j = 0; j < n; ++j) for_each_owned(ex, m, [&](int i) { a(i, j) = x(i); });
    }
}

// ---- scaled product ----------------------------------------------------------------------------

// mant * 2^expo; |mant| in [0.5, 1) unless the product is 0, Inf or NaN.
template <class R> struct ScaledProd {
    R mant;
    std::int32_t expo;
};

// prod over i < n of f(i) without leaving the floating range: each item renormalises its
// running product by frexp (exact, a power-of-two rescale), the exponents are summed, and the
// mantissas multiplied in rounds of 32 items with a renormalisation between rounds, so a
// partial stays >= 2^-33 on any width. In-range results equal the naive product bit for bit.
template <class Ex, class F> BATCHLAS_DLA_INLINE auto prod_scaled(const Ex& ex, int n, F&& f) {
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
    std::int32_t expo_all = detail::scope_reduce<Scope::Auto>(ex, expo);
    const int width = exec_width(ex);
    const int lane = ex.lid();
    for (std::int32_t r0 = 0; r0 < width; r0 += 32) {
        const bool mine = lane >= r0 && lane < r0 + 32;
        mant_all *= detail::scope_reduce<Scope::Auto>(ex, mine ? mant : R(1), sycl::multiplies<R>());
        if (mant_all != R(0) && sycl::isfinite(mant_all)) {
            int e = 0;
            mant_all = sycl::frexp(mant_all, &e);
            expo_all += e;
        }
    }
    return ScaledProd<R>{mant_all, expo_all};
}

// sqrt(|mant| 2^expo), split so neither the square root's argument nor its result leaves range.
template <class R> BATCHLAS_DLA_INLINE R sqrt_scaled(ScaledProd<R> q) {
    const std::int32_t half = q.expo >> 1;       // floor(expo / 2)
    const std::int32_t rem = q.expo - 2 * half;  // 0 or 1
    const R mag = sycl::sqrt(sycl::fabs(q.mant) * (rem ? R(2) : R(1)));
    return sycl::ldexp(mag, half);
}

}  // namespace batchlas::device::v2
