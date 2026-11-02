// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 extension (native gemv family): the pieces a batched gemv body is made of when it
// keeps its hand-written rounding, index arithmetic and shuffle trees bit for bit.
//
//   FmaArith<T>        mac as the fma chain of src/sycl/device_scalar.hh fma_acc (re: fma(ar, br,
//                      acc), fma(-ai, bi, .); im: fma(ar, bi, acc), fma(ai, br, .)); core
//                      Arith::mac rounds the complex product before the add
//   RtScalar<T>        a launch-time alpha/beta with a host-decided zero flag. Not a v2 scalar tag:
//                      core ops reject it at compile time, so the flag is never silently ignored
//   store_fma          y := alpha acc + beta y as fma(alpha, acc, 0) then fma(beta, y, .);
//                      beta zero (Zero, or RtScalar::zero) never reads y
//   WideMat / WideVec  global operands with 64-bit index arithmetic (core Vec/Mat are int32, D7):
//                      A(i, j) = p[i + int64(j) * ld], x(i) = p[int64(i) * inc]
//   conj_rt(v, on)     an operand conjugated when a launch-time flag is set
//   Member<N>          member `id` of an N-strided slice whose mapping the caller computed (e.g.
//                      lane / seg); for_each_owned strides by N; no collectives
//   dot_owned_fma      this item's partial x^T y over the indices it owns; no reduction
//   fold_down<Count>   shift-down sum of lanes {l + stride*t : t < Count} onto lane l; only head
//                      lanes (l < stride) hold a defined sum. Count shuffles per component, no
//                      broadcast (Scope::Sub's xor butterfly makes every lane uniform)
//   leader_sum(team,v) fold_down over a SubGroup, Teams or lockstep Partition<P>; leader() has it

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/device/v2/dla.hh>

namespace batchlas::device::v2 {

// ---- arithmetic -------------------------------------------------------------------------------
template <class T> struct FmaArith {
    using A = Arith<T>;
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) {
        if constexpr (A::is_complex) {
            using R = typename A::real_t;
            R re = A::re(acc), im = A::im(acc);
            re = sycl::fma(A::re(a), A::re(b), re);
            re = sycl::fma(-A::im(a), A::im(b), re);
            im = sycl::fma(A::re(a), A::im(b), im);
            im = sycl::fma(A::im(a), A::re(b), im);
            acc = A::make(re, im);
        } else {
            acc = sycl::fma(a, b, acc);
        }
    }
};

// ---- launch-time scalars ----------------------------------------------------------------------
// `zero` is decided on the host from the BLAS value (alpha == 0, beta == 0). store_fma honours
// beta's flag; alpha's flag is the caller's cue to skip the reduction that makes acc.
template <class T> struct RtScalar {
    T v;
    bool zero;
};
template <class T> BATCHLAS_DLA_INLINE constexpr RtScalar<T> rt_scalar(T v, bool zero) { return {v, zero}; }

template <class S> inline constexpr bool is_rt_scalar_v = false;
template <class T> inline constexpr bool is_rt_scalar_v<RtScalar<T>> = true;

namespace detail {
// out += s * x with fused rounding; Zero adds nothing, One / NegOne add exactly.
template <class T, class S> BATCHLAS_DLA_INLINE void fma_term(T& out, S s, T x) {
    if constexpr (std::is_same_v<S, Zero>) return;
    else if constexpr (std::is_same_v<S, One>) out = Arith<T>::add(out, x);
    else if constexpr (std::is_same_v<S, NegOne>) out = Arith<T>::sub(out, x);
    else FmaArith<T>::mac(out, static_cast<T>(s.v), x);
}
template <class S> BATCHLAS_DLA_INLINE constexpr auto rt_or_tag(S s) {
    if constexpr (is_rt_scalar_v<S>) return s;
    else return as_tag(s);
}
}  // namespace detail

// y := alpha acc + beta y, rounded as fma(alpha, acc, 0) then fma(beta, y, .).
template <class AS, class T, class BS, class Ref> BATCHLAS_DLA_INLINE void store_fma(AS alpha_, T acc, BS beta_, Ref&& y) {
    const auto alpha = detail::rt_or_tag(alpha_);
    const auto beta = detail::rt_or_tag(beta_);
    using B = std::remove_cv_t<decltype(beta)>;
    T out = Arith<T>::zero();
    detail::fma_term(out, alpha, acc);
    if constexpr (is_rt_scalar_v<B>) {
        if (!beta.zero) detail::fma_term(out, beta, static_cast<T>(y));
    } else if constexpr (!std::is_same_v<B, Zero>) {
        detail::fma_term(out, beta, static_cast<T>(y));
    }
    y = out;
}

// ---- 64-bit index operands (Inc = Unit: contiguous, p[i]) -------------------------------------
template <class T, class Inc = std::int64_t> struct WideVec {
    using value_type = std::remove_cv_t<T>;
    using layout = Cyclic;
    static constexpr Mem mem = Mem::Global;
    T* p;
    int n;
    Inc inc{};
    BATCHLAS_DLA_INLINE T& operator()(int i) const {
        if constexpr (std::is_same_v<Inc, Unit>) return p[i];
        else return p[static_cast<std::int64_t>(i) * inc];
    }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};

// Row r of a column-major matrix, addressed from the matrix base: p[r + int64(j) * ld].
template <class T> struct WideRow {
    using value_type = std::remove_cv_t<T>;
    using layout = Cyclic;
    static constexpr Mem mem = Mem::Global;
    T* p;
    int r;
    int n;
    std::int64_t ld;
    BATCHLAS_DLA_INLINE T& operator()(int j) const { return p[r + static_cast<std::int64_t>(j) * ld]; }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};

template <class T> struct WideMat {
    using value_type = std::remove_cv_t<T>;
    using layout = Cyclic;
    static constexpr Mem mem = Mem::Global;
    T* p;
    int m;
    int n;
    std::int64_t ld;
    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const { return p[i + static_cast<std::int64_t>(j) * ld]; }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
    BATCHLAS_DLA_INLINE WideVec<T, Unit> col(int j) const { return {p + static_cast<std::int64_t>(j) * ld, m}; }
    BATCHLAS_DLA_INLINE WideRow<T> row(int i) const { return {p, i, n, ld}; }
};

template <class T> BATCHLAS_DLA_INLINE WideMat<T> wide_mat(T* p, int m, int n, std::int64_t ld) { return {p, m, n, ld}; }
template <class T> BATCHLAS_DLA_INLINE WideVec<T> wide_vec(T* p, int n, std::int64_t inc) { return {p, n, inc}; }
template <class T> BATCHLAS_DLA_INLINE WideVec<T, Unit> wide_vec(T* p, int n) { return {p, n}; }

// ---- operand modifier -------------------------------------------------------------------------
template <class V> struct ConjRt {
    using value_type = typename V::value_type;
    V v;
    bool on;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const {
        value_type a = v(i);
        if constexpr (Arith<value_type>::is_complex) {
            if (on) a = Arith<value_type>::conj(a);
        }
        return a;
    }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
};
template <class V> BATCHLAS_DLA_INLINE ConjRt<V> conj_rt(V v, bool on) { return {v, on}; }

// ---- executor ---------------------------------------------------------------------------------
template <int N> struct Member {
    static_assert(N >= 1, "v2::Member: N >= 1");
    static constexpr int size = N, team_size = N, nteams = 1;
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;
    int id;
    BATCHLAS_DLA_INLINE int lid() const { return id; }
    BATCHLAS_DLA_INLINE int lane() const { return id; }
    BATCHLAS_DLA_INLINE int team() const { return 0; }
    BATCHLAS_DLA_INLINE bool leader() const { return id == 0; }
};

// ---- partial dot and leader reductions --------------------------------------------------------
template <class Ex, class X, class Y> BATCHLAS_DLA_INLINE auto dot_owned_fma(const Ex& ex, const X& x, const Y& y) {
    using T = elem_t<X>;
    T acc = Arith<T>::zero();
    for_each_owned(ex, x.size(), [&](int i) { FmaArith<T>::mac(acc, x(i), y(i)); });
    return acc;
}

// Every lane of g must call (lockstep). Closed when Count * stride <= the sub-group size; lanes
// at or above that shift in values no head lane reads.
template <int Count, class G, class T, class Stride = Unit>
BATCHLAS_DLA_INLINE T fold_down(const G& g, T v, Stride stride = {}) {
    static_assert(detail::is_pow2(Count), "v2::fold_down: Count must be a power of two");
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        using R = typename A::real_t;
        R re = A::re(v), im = A::im(v);
#pragma unroll
        for (int w = Count / 2; w >= 1; w >>= 1) {
            const int off = int(stride) * w;
            re += sycl::shift_group_left(g, re, off);
            im += sycl::shift_group_left(g, im, off);
        }
        return A::make(re, im);
    } else {
#pragma unroll
        for (int w = Count / 2; w >= 1; w >>= 1) v += sycl::shift_group_left(g, v, int(stride) * w);
        return v;
    }
}

template <int SG, class T> BATCHLAS_DLA_INLINE T leader_sum(const SubGroup<SG>& ex, T v) { return fold_down<SG>(ex.sg, v); }
template <int WG, int SG, class T> BATCHLAS_DLA_INLINE T leader_sum(const Teams<WG, SG>& ex, T v) {
    return fold_down<SG>(ex.it.get_sub_group(), v);
}
template <int P, class T> BATCHLAS_DLA_INLINE T leader_sum(const Partition<P, false, sycl::sub_group>& ex, T v) {
    return fold_down<P>(ex.part, v);
}
template <class T> BATCHLAS_DLA_INLINE T leader_sum(const Thread&, T v) { return v; }

}  // namespace batchlas::device::v2
