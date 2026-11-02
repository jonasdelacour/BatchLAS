// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 extension ops first needed by gebrd (labrd panel, gebd2, the CTA tile). They are
// general: nothing here knows about bidiagonalisation.
//
//   abs2, lacgv, conj_view, transposed, masked_view      elementwise ops and read-only views
//   bcast, invoke_bcast, select_lane                     leader / lane broadcasts
//   gemv_term<TR>, interleave, gemv_acc                  fused RowOwner gemv sequences
//   gemv_pair<TR>                                        two gemv^T sharing x, one sweep
//   larfg_bcast (items share x), larfg_lanes (x one element per lane)
//   larf<Side, Alg>                                      apply I - tau v v^H, v(0) = 1
//   lacpy<RowPerLane>                                    2-D copy into a lane-per-row tile
//
// The bodies keep the operand order and statement shape of the kernels they replaced: under
// -ffp-contract=on only `acc += a * b` written as one expression becomes an FMA, so a product
// returned from a function and added later stays two roundings (docs/guide/build.md).

#include <array>
#include <cstdint>
#include <type_traits>

#include <batchlas/blas/device/v2/level2.hh>
#include <batchlas/util/group-invoke.hh>

namespace batchlas::device::v2 {

// Layout: lane l owns row l of a tile whose rows fit the executor (rows <= size).
struct RowPerLane {};

// ---- elementwise and views -----------------------------------------------------------------

// |x|^2 = re^2 + im^2, by value: a caller's `acc += abs2(x)` is an add, not an FMA.
#ifndef BATCHLAS_DLA_V2_INTEG_ABS2  // integ/dla-v2-all: defined by several ext headers
#define BATCHLAS_DLA_V2_INTEG_ABS2
template <class T> BATCHLAS_DLA_INLINE real_t<T> abs2(T x) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) {
        const real_t<T> r = A::re(x), i = A::im(x);
        return r * r + i * i;
    } else {
        return x * x;
    }
}
#endif  // BATCHLAS_DLA_V2_INTEG_ABS2

// x := conj(x) (LAPACK lacgv); the identity for real T.
template <class Ex, class X> BATCHLAS_DLA_INLINE void lacgv(const Ex& ex, const X& x) {
    using T = elem_t<X>;
    if constexpr (Arith<T>::is_complex) for_each_owned(ex, x.size(), [&](int i) { x(i) = Arith<T>::conj(x(i)); });
}

// conj(v(i)), read only.
template <class V> struct ConjView {
    V v;
    using value_type = typename V::value_type;
    static constexpr Mem mem = V::mem;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const { return Arith<value_type>::conj(v(i)); }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
    BATCHLAS_DLA_INLINE ConjView<decltype(std::declval<V>().tail(0))> tail(int k) const { return {v.tail(k)}; }
    BATCHLAS_DLA_INLINE ConjView<decltype(std::declval<V>().head(0))> head(int k) const { return {v.head(k)}; }
};
template <class V> BATCHLAS_DLA_INLINE ConjView<V> conj_view(V v) { return {v}; }
template <class V> BATCHLAS_DLA_INLINE Published<ConjView<V>> conj_view(Published<V> v) { return {{v.v}}; }

// A^T as a matrix operand: (i, j) reads a(j, i).
template <class M> struct TransView {
    M a;
    using value_type = typename M::value_type;
    static constexpr Mem mem = M::mem;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return a(j, i); }
    BATCHLAS_DLA_INLINE int rows() const { return a.cols(); }
    BATCHLAS_DLA_INLINE int cols() const { return a.rows(); }
};
template <class M> BATCHLAS_DLA_INLINE TransView<M> transposed(M a) { return {a}; }

// a(i, j) when live, else 0 (a masked or padding batch item reads zeros, never a).
template <class M> struct MaskedView {
    M a;
    bool live;
    using value_type = typename M::value_type;
    BATCHLAS_DLA_INLINE value_type operator()(int i, int j) const { return live ? a(i, j) : value_type(0); }
};
template <class M> BATCHLAS_DLA_INLINE MaskedView<M> masked_view(M a, bool live) { return {a, live}; }

// ---- broadcasts ----------------------------------------------------------------------------

// The leader's v on every item of ex. Collective.
template <class Ex, class T> BATCHLAS_DLA_INLINE T bcast(const Ex& ex, T v) {
    if constexpr (is_thread_v<Ex>) {
        return v;
    } else if constexpr (is_work_group_v<Ex>) {
        return batchlas::portable::group_broadcast(ex.it.get_group(), v);
    } else if constexpr (requires { ex.sg; }) {
        return batchlas::portable::group_broadcast(ex.sg, v);
    } else if constexpr (requires { ex.part; }) {
        if constexpr (std::is_same_v<std::remove_cv_t<decltype(ex.part)>, sycl::sub_group>) {
            const auto base = static_cast<std::uint32_t>(ex.part.get_local_linear_id()) & ~static_cast<std::uint32_t>(Ex::size - 1);
            return batchlas::portable::select_from_group(ex.part, v, base);
        } else {
            return batchlas::broadcast_from_leader(ex.part, v);  // sg_leader_broadcast by ADL
        }
    } else {
        return batchlas::portable::group_broadcast(ex.it.get_sub_group(), v);  // Teams
    }
}

// fn() evaluated on the leader only, returned on every item (batchlas::invoke_one_broadcast).
template <class Ex, class F> BATCHLAS_DLA_INLINE auto invoke_bcast(const Ex& ex, F&& fn) {
    std::remove_cvref_t<decltype(fn())> value{};
    if (ex.leader()) value = fn();
    return bcast(ex, value);
}

// Lane `lane`'s v on every lane of a Partition. Collective.
template <class Ex, class T> BATCHLAS_DLA_INLINE T select_lane(const Ex& ex, T v, int lane) {
    if constexpr (std::is_same_v<std::remove_cv_t<decltype(ex.part)>, sycl::sub_group>) {
        const auto base = static_cast<std::uint32_t>(ex.part.get_local_linear_id()) & ~static_cast<std::uint32_t>(Ex::size - 1);
        return batchlas::portable::select_from_group(ex.part, v, base + static_cast<std::uint32_t>(lane));
    } else {
        return select_from_group(ex.part, v, static_cast<std::uint32_t>(lane));  // ADL: SubGroupPartition
    }
}

// ---- fused RowOwner gemv sequences ---------------------------------------------------------

namespace detail {
template <class S, class T> BATCHLAS_DLA_INLINE T signed_elem(T e) {
    static_assert(std::is_same_v<S, One> || std::is_same_v<S, NegOne>, "v2::gemv_term: the sign is One or NegOne");
    if constexpr (std::is_same_v<S, One>) return e;
    else return Arith<T>::neg(e);
}
}  // namespace detail

// One term of gemv_acc: sum over k (ascending) of s op(A)(r, k) x(k), each product added to
// (s = One) or subtracted from (s = NegOne) the running sum as it is formed.
template <Transpose TR, class S, class MA, class X> struct GemvTerm {
    MA a;
    X x;
    BATCHLAS_DLA_INLINE int len() const { return TR == Transpose::NoTrans ? a.cols() : a.rows(); }
    template <class T> BATCHLAS_DLA_INLINE void step(T& acc, int r, int k) const {
        if constexpr (TR == Transpose::NoTrans) {
            Arith<T>::mac(acc, detail::signed_elem<S, T>(a(r, k)), x(k));
        } else if constexpr (TR == Transpose::Trans) {
            Arith<T>::mac(acc, detail::signed_elem<S, T>(a(k, r)), x(k));
        } else {
            Arith<T>::mac_conj(acc, detail::signed_elem<S, T>(a(k, r)), x(k));
        }
    }
    template <class T> BATCHLAS_DLA_INLINE void run(T& acc, int r) const {
        const int n = len();
        for (int k = 0; k < n; ++k) step(acc, r, k);
    }
};
// x is read by every item: it must be Published.
template <Transpose TR, class S, class MA, class V>
BATCHLAS_DLA_INLINE GemvTerm<TR, S, MA, Published<V>> gemv_term(S, const MA& a, const Published<V>& x) {
    return {a, x};
}

// Two terms of the same length whose k-th products alternate: t1(k), t2(k), t1(k+1), ...
template <class T1, class T2> struct Interleaved {
    T1 t1;
    T2 t2;
    template <class T> BATCHLAS_DLA_INLINE void run(T& acc, int r) const {
        const int n = t1.len();
        for (int k = 0; k < n; ++k) {
            t1.step(acc, r, k);
            t2.step(acc, r, k);
        }
    }
};
template <class T1, class T2> BATCHLAS_DLA_INLINE Interleaved<T1, T2> interleave(const T1& t1, const T2& t2) { return {t1, t2}; }

// y(r) := alpha (y0(r) + terms(r)...) for every owned r < y.size() (Cyclic, one item per output
// row, the sum in one register across every term: LAPACK labrd's chained gemv updates in one
// sweep). y0 is Zero or an operand read at r, possibly y itself. No barrier.
template <class Ex, class AS, class Y0, class Y, class... Terms>
BATCHLAS_DLA_INLINE void gemv_acc(const Ex& ex, AS alpha_, const Y0& y0, const Y& y, const Terms&... terms) {
    using T = elem_t<Y>;
    const auto alpha = as_tag(alpha_);
    const int n = y.size();
    for (int r = ex.lid(); r < n; r += Ex::size) {
        T acc;
        if constexpr (std::is_same_v<Y0, Zero>) acc = Arith<T>::zero();
        else acc = y0(r);
        (terms.run(acc, r), ...);
        y(r) = tag_mul(alpha, acc);
    }
}

// y1 := alpha op(A1) x + beta y1 and y2 := alpha op(A2) x + beta y2, TR = Trans or ConjTrans,
// A1 and A2 with the same rows: one sweep reads x(r) once for both columns; per output column
// two group reductions (alg::GroupReducePerOutput), then the leader stores both. y1, y2 uniform.
template <Transpose TR, class Alg = alg::GroupReducePerOutput, class Ex, class AS, class M1, class M2, class X, class BS,
          class Y1, class Y2>
BATCHLAS_DLA_INLINE void gemv_pair(const Ex& ex, AS alpha_, const M1& a1, const M2& a2, const X& xp, BS beta_, const Y1& y1,
                                   const Y2& y2) {
    static_assert(TR != Transpose::NoTrans, "v2::gemv_pair: a (Conj)Trans op");
    static_assert(std::is_same_v<Alg, alg::GroupReducePerOutput>, "v2::gemv_pair: alg::GroupReducePerOutput is the only algorithm");
    using T = typename M1::value_type;
    using Ar = Arith<T>;
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    const int m = a1.rows(), n = a1.cols();
    for (int c = 0; c < n; ++c) {
        T acc1 = Ar::zero(), acc2 = Ar::zero();
        for (int r = ex.lid(); r < m; r += Ex::size) {
            const T xr = x(r);
            if constexpr (TR == Transpose::ConjTrans) {
                Ar::mac_conj(acc1, a1(r, c), xr);
                Ar::mac_conj(acc2, a2(r, c), xr);
            } else {
                Ar::mac(acc1, a1(r, c), xr);
                Ar::mac(acc2, a2(r, c), xr);
            }
        }
        acc1 = ex.group_reduce(acc1);
        acc2 = ex.group_reduce(acc2);
        if (ex.leader()) {
            store(alpha, acc1, beta, y1(c));
            store(alpha, acc2, beta, y2(c));
        }
    }
}

// ---- Householder reflectors ----------------------------------------------------------------

template <class T> struct Reflector {
    T beta;
    T tau;
    T scale;
};

// Householder scalars of [alpha; x] when the executor's items share x. sigma = ||x||^2 is the
// caller's reduction (uniform). The leader reads alpha_src and calls gen(alpha, sqrt(sigma), len)
// (anything with .beta .tau .scale, e.g. detail::larfg_result); alpha and the three scalars are
// broadcast. x is not touched: the caller scales it (larfg with tau == 0 is not uniform across
// LAPACK variants).
template <class Ex, class T, class R, class Gen>
BATCHLAS_DLA_INLINE Reflector<T> larfg_bcast(const Ex& ex, const T& alpha_src, R sigma, int len, Gen&& gen) {
    using Ar = Arith<T>;
    T alpha = Ar::zero();
    if (ex.leader()) alpha = alpha_src;
    alpha = bcast(ex, alpha);
    T tau = Ar::zero();
    T beta = alpha;
    T scale = Ar::zero();
    if (ex.leader()) {
        const auto s = gen(alpha, sycl::sqrt(sigma), len);
        beta = s.beta;
        tau = s.tau;
        scale = s.scale;
    }
    tau = bcast(ex, tau);
    beta = bcast(ex, beta);
    scale = bcast(ex, scale);
    return {beta, tau, scale};
}

// larfg with [alpha; x] one element per lane of a Partition: lane `head` holds alpha, lanes with
// x_active hold x. gen(alpha, xnorm, len) -> std::array<T, 3>{beta, tau, scale} runs on the
// leader. On return lane `head` holds beta, the x lanes hold v (x * scale when tau != 0); the
// result is tau on every lane. Shuffles only, no barrier.
template <class Ex, class T, class Gen>
BATCHLAS_DLA_INLINE T larfg_lanes(const Ex& ex, int len, int head, T& alpha, T& x, bool x_active, Gen&& gen) {
    using R = real_t<T>;
    const R xsq = x_active ? abs2(x) : R(0);
    const R sumsq = ex.team_reduce(xsq);
    const R xnorm = invoke_bcast(ex, [&]() { return sycl::sqrt(sumsq); });
    const T alpha_h = select_lane(ex, alpha, head);
    const std::array<T, 3> s = invoke_bcast(ex, [&]() { return gen(alpha_h, xnorm, len); });
    if (ex.lid() == head) {
        alpha = s[0];
    } else if (x_active && s[1] != Arith<T>::zero()) {
        x = Arith<T>::mul(x, s[2]);
    }
    return s[1];
}

namespace detail {
template <class V> inline constexpr bool is_unit_head_v = false;
template <class V> inline constexpr bool is_unit_head_v<UnitHead<V>> = true;

// v(r) for alg::Serial: the select `r == 0 ? 1 : v(r)` (gebd2) for a UnitHead with head 0, a
// plain read when the caller stored the 1.
template <class T, class V> BATCHLAS_DLA_INLINE T serial_v(const V& v, int r) {
    if constexpr (is_unit_head_v<V>) return r == 0 ? Arith<T>::one() : static_cast<T>(v.v(r));
    else return static_cast<T>(v(r));
}
template <class Ex, Side SD> constexpr auto default_larf_alg() {
    if constexpr (is_thread_v<Ex>) return alg::Serial{};
    else if constexpr (SD == Side::Left) return ColPerLane{};
    else return RowPerLane{};
}
}  // namespace detail

// C := H C (Left) or C H (Right), H = I - tau v v^H. tau == 0 is a no-op; no barrier.
//   alg::Serial (Thread): column (Left) / row (Right) loop outside. v is a UnitHead with head 0
//     (v(0) = 1 read as a select) or a plain operand whose v(0) the caller stored as 1.
//   ColPerLane (Left) / RowPerLane (Right), one lane per column / row of a tile: lane l updates
//     column / row l when lo <= l < C.cols() / C.rows(). v is a UnitHead with head 0, peeled:
//     the dot starts from C's head row / column and the implied 1 is never multiplied.
template <Side SD, class Alg = void, class Ex, class T, class V, class MC>
BATCHLAS_DLA_INLINE void larf(const Ex& ex, T tau, const V& v, const MC& c, int lo = 0) {
    using Al = detail::pick_t<Alg, decltype(detail::default_larf_alg<Ex, SD>())>;
    using Ar = Arith<T>;
    static_assert(std::is_same_v<T, typename MC::value_type>, "v2::larf: tau has C's scalar type");
    if (tau == Ar::zero()) return;
    if constexpr (std::is_same_v<Al, alg::Serial>) {
        static_assert(Ex::size == 1, "v2::larf: alg::Serial needs a one-item executor (Thread)");
        const int m = c.rows(), n = c.cols();
        if constexpr (SD == Side::Left) {
            for (int j = 0; j < n; ++j) {
                T dot = Ar::zero();
                for (int r = 0; r < m; ++r) Ar::mac_conj(dot, detail::serial_v<T>(v, r), c(r, j));
                dot = Ar::mul(dot, tau);
                for (int r = 0; r < m; ++r) Ar::mac(c(r, j), Ar::neg(detail::serial_v<T>(v, r)), dot);
            }
        } else {
            for (int r = 0; r < m; ++r) {
                T dot = Ar::zero();
                for (int j = 0; j < n; ++j) Ar::mac(dot, c(r, j), detail::serial_v<T>(v, j));
                dot = Ar::mul(dot, tau);
                for (int j = 0; j < n; ++j) Ar::mac(c(r, j), Ar::neg(dot), Ar::conj(detail::serial_v<T>(v, j)));
            }
        }
    } else if constexpr (std::is_same_v<Al, ColPerLane>) {
        static_assert(SD == Side::Left && detail::is_unit_head_v<V>, "v2::larf: ColPerLane applies a UnitHead from the left");
        const int j = ex.lid();
        if (j >= lo && j < c.cols()) {
            const int m = c.rows();
            T dot = c(0, j);
            for (int r = 1; r < m; ++r) Ar::mac_conj(dot, static_cast<T>(v.v(r)), c(r, j));
            const T gamma = Ar::mul(tau, dot);
            c(0, j) = Ar::sub(c(0, j), gamma);
            for (int r = 1; r < m; ++r) Ar::mac(c(r, j), Ar::neg(static_cast<T>(v.v(r))), gamma);
        }
    } else if constexpr (std::is_same_v<Al, RowPerLane>) {
        static_assert(SD == Side::Right && detail::is_unit_head_v<V>, "v2::larf: RowPerLane applies a UnitHead from the right");
        const int r = ex.lid();
        if (r >= lo && r < c.rows()) {
            const int n = c.cols();
            T dot = c(r, 0);
            for (int j = 1; j < n; ++j) Ar::mac(dot, c(r, j), static_cast<T>(v.v(j)));
            const T gamma = Ar::mul(tau, dot);
            c(r, 0) = Ar::sub(c(r, 0), gamma);
            for (int j = 1; j < n; ++j) Ar::mac(c(r, j), Ar::neg(gamma), Ar::conj(static_cast<T>(v.v(j))));
        }
    } else {
        static_assert(sizeof(Al) == 0, "v2::larf: unknown algorithm tag");
    }
}

// dst(r, c) := src(r, c) (src: any (r, c) operand, e.g. a Mat or masked_view), lane l copying
// row l when active and l < dst.rows(); column loop outside. No barrier.
template <class Layout, class Ex, class Src, class Dst>
BATCHLAS_DLA_INLINE void lacpy(const Ex& ex, const Src& src, const Dst& dst, bool active = true) {
    static_assert(std::is_same_v<Layout, RowPerLane>, "v2::lacpy: RowPerLane is the only layout");
    const int r = ex.lid();
    const int m = dst.rows(), n = dst.cols();
    for (int c = 0; c < n; ++c) {
        if (active && r < m) dst(r, c) = src(r, c);
    }
}

}  // namespace batchlas::device::v2
