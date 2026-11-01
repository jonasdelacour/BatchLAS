// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, tridiagonal-eigensolver extension (the steqr family):
//   Givens, rot, lasr, lasr_rows  plane rotations (LAPACK dlasr, SIDE = R, PIVOT = V, DIRECT = F)
//   lanst_max                     max |entry| of a symmetric tridiagonal (dlanst 'M')
//   tridiag_split                 dsteqr's relative split test: zero every negligible e(i)
//   find_first                    first index of a window whose element passes a predicate
//   reverse                       x(lo..hi-1) := x(hi-1..lo), so a QR sweep runs as QL on J T J
//   tridiag_rot                   one bulge-chase similarity G^T T G on rows/cols (i, i+1)
// Memory operands (Vec, VectorView, any x(i) with x.size()) run on Thread or a Cyclic executor.
// LaneVec is the register layout of the partition solvers: element i is held by the item with
// lid == i (n <= Ex::size), and an op reads or writes only the items of the operand's window.
//
// Rotation convention (the one steqr stores): G = Givens{c, s} maps the pair (x, y) to
// (c x - s y, s x + c y). c and s are real; x and y may be complex.

#include <array>
#include <cstdint>
#include <type_traits>

#include <batchlas/blas/device/v2/level1.hh>
#include <batchlas/util/group-collectives.hh>

namespace batchlas::device::v2 {

namespace alg {
struct Dense {};         // lasr: apply every rotation
struct SkipIdentity {};  // lasr: skip a stored (1, 0), the replay of a padded rotation table
}  // namespace alg

struct OnePerItem {};  // layout of LaneVec

template <class R> struct Givens {
    R c;
    R s;
};
template <class R> BATCHLAS_DLA_INLINE Givens<R> givens(Givens<R> g) { return g; }
template <class R> BATCHLAS_DLA_INLINE Givens<R> givens(const std::array<R, 2>& cs) { return {cs[0], cs[1]}; }

// dsteqr's split tolerances: e(i) is negligible when |e(i)|^2 <= eps2 |d(i)| |d(i+1)| + safmin.
template <class R> struct SplitTol {
    R eps2;
    R safmin;
};

// One element per item, held in a register: element i belongs to the item with lid == i.
template <class T> struct LaneVec {
    using value_type = std::remove_cv_t<T>;
    using layout = OnePerItem;
    T& v;
    int lo;
    int hi;

    BATCHLAS_DLA_INLINE T& operator()(int) const { return v; }
    BATCHLAS_DLA_INLINE int size() const { return hi; }
    template <class Ex> BATCHLAS_DLA_INLINE bool owns(const Ex& ex) const {
        const int l = ex.lid();
        return l >= lo && l < hi;
    }
};
// The window [lo, hi) of the lane-register vector whose own element is v.
template <class T> BATCHLAS_DLA_INLINE LaneVec<T> lane_vec(T& v, int lo, int hi) { return {v, lo, hi}; }

namespace detail {

struct FmaxOp {
    template <class R> BATCHLAS_DLA_INLINE R operator()(R a, R b) const { return sycl::fmax(a, b); }
};

template <class Ex> inline constexpr bool is_partition_v = false;
template <int P, bool M, class I> inline constexpr bool is_partition_v<Partition<P, M, I>> = true;

// fmax where the executor's reduction takes any functor (Thread, the Partition butterfly);
// the sycl group reductions take sycl::maximum, equal to fmax on NaN-free |x|.
template <Scope S, class Ex, class R> BATCHLAS_DLA_INLINE R reduce_fmax(const Ex& ex, R v) {
    if constexpr (is_thread_v<Ex> || is_partition_v<Ex>) return scope_reduce<S>(ex, v, FmaxOp{});
    else return scope_reduce<S>(ex, v, sycl::maximum<R>());
}

// The value of item lid + 1 of the executor's team (unspecified on its last item).
template <class Ex, class T> BATCHLAS_DLA_INLINE T lane_next(const Ex& ex, T v) {
    if constexpr (requires { ex.part; }) {
        if constexpr (sycl::is_group_v<std::remove_cvref_t<decltype(ex.part)>>) {
            return batchlas::portable::shift_group_left(ex.part, v, 1u);
        } else {
            return shift_group_left(ex.part, v, 1u);  // ADL: SubGroupPartition (src/kernels/common)
        }
    } else {
        static_assert(requires { ex.sg; }, "v2: lane_next needs a SubGroup or Partition executor");
        return batchlas::portable::shift_group_left(ex.sg, v, 1u);
    }
}

// The value of team item `src` (a lid, 0 <= src < Ex::size).
template <class Ex, class T> BATCHLAS_DLA_INLINE T lane_select(const Ex& ex, T v, int src) {
    if constexpr (requires { ex.part; }) {
        if constexpr (sycl::is_group_v<std::remove_cvref_t<decltype(ex.part)>>) {
            // A lockstep partition shuffles over the whole sub-group: add the chunk base.
            const int base = static_cast<int>(ex.part.get_local_linear_id()) & ~(Ex::size - 1);
            return batchlas::portable::select_from_group(ex.part, v, static_cast<std::uint32_t>(base + src));
        } else {
            return select_from_group(ex.part, v, static_cast<std::uint32_t>(src));  // ADL
        }
    } else {
        static_assert(requires { ex.sg; }, "v2: lane_select needs a SubGroup or Partition executor");
        return batchlas::portable::select_from_group(ex.sg, v, static_cast<std::uint32_t>(src));
    }
}

}  // namespace detail

// ---- plane rotations -------------------------------------------------------------------------

// (x, y) := (c x - s y, s x + c y) on two scalars the caller owns.
template <class R, class T> BATCHLAS_DLA_INLINE void rot(Givens<R> g, T& x, T& y) {
    using A = Arith<T>;
    const T a = x;
    const T b = y;
    if constexpr (A::is_complex) {
        x = A::make(g.c * A::re(a) - g.s * A::re(b), g.c * A::im(a) - g.s * A::im(b));
        y = A::make(g.s * A::re(a) + g.c * A::re(b), g.s * A::im(a) + g.c * A::im(b));
    } else {
        x = g.c * a - g.s * b;
        y = g.s * a + g.c * b;
    }
}

// (x(i), y(i)) := G (x(i), y(i)) for every owned i; elementwise, no barrier.
template <class Ex, class R, class X, class Y>
BATCHLAS_DLA_INLINE void rot(const Ex& ex, Givens<R> g, const X& x, const Y& y) {
    for_each_owned(ex, x.size(), [&](int i) { rot(g, x(i), y(i)); });
}

// x := x G(0) G(1) ... G(n-2): rotation j = givens(rots(j)) acts on (x(j), x(j+1)), j ascending.
// One vector is one dependence chain, so this is serial (lasr_rows distributes a matrix).
template <class Alg = alg::Dense, class Ex, class Rots, class X>
BATCHLAS_DLA_INLINE void lasr(const Ex&, int n, const Rots& rots, const X& x) {
    static_assert(Ex::size == 1, "v2::lasr on one vector is serial: pass Thread, or use lasr_rows");
    for (int j = 0; j < n - 1; ++j) {
        const auto g = givens(rots(j));
        if constexpr (std::is_same_v<Alg, alg::SkipIdentity>) {
            using R = decltype(g.c);
            if (g.c == R(1) && g.s == R(0)) continue;
        }
        rot(g, x(j), x(j + 1));
    }
}

// A := A G(0) ... G(n-2) (dlasr SIDE = R): each owned row runs the serial sequence.
template <class Alg = alg::Dense, class Ex, class Rots, class M>
BATCHLAS_DLA_INLINE void lasr_rows(const Ex& ex, const Rots& rots, const M& a) {
    for_each_owned(ex, a.rows(), [&](int r) {
        lasr<Alg>(Thread{}, a.cols(), rots, [&](int j) -> auto& { return a(r, j); });
    });
}

// ---- symmetric tridiagonal (d, e), n = d.size() >= 1, e(i) couples d(i) and d(i+1) -------------

// max(|d(i)|, |e(i)|), dlanst 'M'. fmax ignores a NaN entry (the steqr rule); uniform result.
template <Scope S = Scope::Auto, class Ex, class D, class E>
BATCHLAS_DLA_INLINE auto lanst_max(const Ex& ex, const D& d, const E& e) {
    using R = elem_t<D>;
    static_assert(!Arith<R>::is_complex, "v2::lanst_max: d and e are real");
    const int n = d.size();
    R acc = R(0);
    if constexpr (Ex::size == 1) acc = sycl::fabs(d(n - 1));
    else if (ex.lid() == (n - 1) % Ex::size) acc = sycl::fabs(d(n - 1));
    for_each_owned(ex, n - 1, [&](int i) { acc = sycl::fmax(acc, sycl::fmax(sycl::fabs(d(i)), sycl::fabs(e(i)))); });
    return detail::reduce_fmax<S>(ex, acc);
}

// Lane-register form over the windows of d and e.
template <Scope S = Scope::Auto, class Ex, class T>
BATCHLAS_DLA_INLINE T lanst_max(const Ex& ex, const LaneVec<T>& d, const LaneVec<T>& e) {
    T acc = T(0);
    if (d.owns(ex)) acc = sycl::fabs(d.v);
    if (e.owns(ex)) acc = sycl::fmax(acc, sycl::fabs(e.v));
    return detail::reduce_fmax<S>(ex, acc);
}

// x := a x on the window.
template <class Ex, class A, class T> BATCHLAS_DLA_INLINE void scal(const Ex& ex, A a_, const LaneVec<T>& x) {
    if (x.owns(ex)) x.v = tag_mul(as_tag(a_), static_cast<std::remove_cv_t<T>>(x.v));
}

// e(j) := 0 for every owned j < e.size() that is negligible; true when this item zeroed one
// (an e(j) already zero counts). d.size() > e.size().
template <class Ex, class D, class E, class R>
BATCHLAS_DLA_INLINE bool tridiag_split(const Ex& ex, const D& d, const E& e, SplitTol<R> tol) {
    bool any = false;
    for_each_owned(ex, e.size(), [&](int j) {
        if (sycl::fabs(e(j)) * sycl::fabs(e(j)) <= tol.eps2 * sycl::fabs(d(j)) * sycl::fabs(d(j + 1)) + tol.safmin) {
            e(j) = R(0);
            any = true;
        }
    });
    return any;
}

// Lane-register form: d is this item's d(lid); e's window must end before the last valid d
// (d(lid + 1) is read from the next item). An e already zero is left untouched.
template <class Ex, class T>
BATCHLAS_DLA_INLINE void tridiag_split(const Ex& ex, const T& d, const LaneVec<T>& e, SplitTol<T> tol) {
    const T d_next = detail::lane_next(ex, d);  // every item shuffles; the last one's value is unused
    if (e.owns(ex)) {
        if (e.v != T(0)) {
            const T rhs = tol.eps2 * sycl::fabs(d) * sycl::fabs(d_next) + tol.safmin;
            if (sycl::fabs(e.v) * sycl::fabs(e.v) <= rhs) e.v = T(0);
        }
    }
}

// The smallest i in x's window with pred(x(i)), else `none`; uniform over the team.
template <Scope S = Scope::Auto, class Ex, class T, class Pred>
BATCHLAS_DLA_INLINE int find_first(const Ex& ex, const LaneVec<T>& x, int none, Pred&& pred) {
    const int l = ex.lid();
    const int cand = (l >= x.lo && l < x.hi && pred(static_cast<std::remove_cv_t<T>>(x.v))) ? l : none;
    return detail::scope_reduce<S>(ex, cand, sycl::minimum<int>());
}

// x(lo..hi-1) := x(hi-1..lo) when `on` (one shuffle on every item, so `on` may differ per team).
template <class Ex, class T> BATCHLAS_DLA_INLINE void reverse(const Ex& ex, const LaneVec<T>& x, bool on) {
    const int l = ex.lid();
    const int src = (on && l >= x.lo && l < x.hi) ? (x.lo + x.hi - 1 - l) : l;
    x.v = detail::lane_select(ex, static_cast<std::remove_cv_t<T>>(x.v), src);
}

// Memory form, serial: x(lo..hi-1) := x(hi-1..lo).
template <class X> BATCHLAS_DLA_INLINE void reverse(const Thread&, int lo, int hi, const X& x) {
    for (int a = lo, b = hi - 1; a < b; ++a, --b) {
        const auto t = x(a);
        x(a) = x(b);
        x(b) = t;
    }
}

// One step of the implicit-shift bulge chase on (d, e): T := G^T T G on rows/cols (i, j = i+1),
// prev_bulge the fill at (i-1, j); returns the new fill at (i, j+1). forward == false reads T
// bottom-up (index k is k' = n-1-k), so one chase implements both QR and QL. I is the caller's
// index type (kept so a port reproduces its address arithmetic).
template <class D, class E, class T, class I>
BATCHLAS_DLA_INLINE T tridiag_rot(const Thread&, const D& d, const E& e, const T& prev_bulge, I i, I j,
                                  Givens<T> g, bool forward) {
    const T c = g.c;
    const T s = g.s;
    const I n = d.size();
    const I ne = e.size();
    auto d_get = [&](I k) -> T { return forward ? d(k) : d(n - 1 - k); };
    auto d_set = [&](I k, T val) {
        if (forward) d(k) = val;
        else d(n - 1 - k) = val;
    };
    auto e_get = [&](I k) -> T { return forward ? e(k) : e(ne - 1 - k); };
    auto e_set = [&](I k, T val) {
        if (forward) e(k) = val;
        else e(ne - 1 - k) = val;
    };

    const T di = d_get(i);
    const T dj = d_get(j);
    const T ei = e_get(i);
    const T ej = (j < ne) ? e_get(j) : T(0);

    d_set(i, c * (c * di - ei * s) - s * (ei * c - s * dj));
    d_set(j, c * (c * dj + ei * s) + s * (ei * c + s * di));
    if (i > 0) e_set(i - 1, e_get(i - 1) * c - prev_bulge * s);
    e_set(i, c * (c * ei + s * di) - s * (c * dj + s * ei));
    if (j < ne) e_set(j, c * ej);
    return -ej * s;
}

}  // namespace batchlas::device::v2
