// SPDX-License-Identifier: MIT
#pragma once
// Native gemv's private device LA v2 ops: what the five bodies need and v2.1 core lacks.
//
//   RtScalar<T>        a launch-time alpha/beta with a host-decided zero flag; not a v2 scalar
//                      tag, so core ops reject it at compile time and never ignore the flag
//   store_fma          y := alpha acc + beta y as fma(alpha, acc, 0) then fma(beta, y, .) (core
//                      store() rounds the products); beta zero never reads y
//   conj_if(v, on)     v conjugated when a launch-time flag is set (core has no adaptor yet)
//   Member<N>          member `id` of an N-strided slice whose mapping the caller computed
//                      (lane / seg); Cyclic ops stride by N; no collectives
//   fold_down<Count>   shift-down sum of lanes {l + stride*t : t < Count} onto lane l; only head
//                      lanes hold the sum. Core reductions are xor butterflies with a uniform
//                      result, and cannot fold at a runtime, non-power-of-two stride
//   leader_sum(team,v) fold_down over a SubGroup or lockstep Partition<P>; leader() has it

#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/arith.hh>

namespace batchlas::sycl_gemv::dla {

namespace v2 = batchlas::device;

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
    if constexpr (std::is_same_v<S, v2::Zero>) return;
    else if constexpr (std::is_same_v<S, v2::One>) out = v2::Arith<T>::add(out, x);
    else if constexpr (std::is_same_v<S, v2::NegOne>) out = v2::Arith<T>::sub(out, x);
    else v2::arith_t<v2::arith::Fma, T>::mac(out, static_cast<T>(s.v), x);
}
template <class S> BATCHLAS_DLA_INLINE constexpr auto rt_or_tag(S s) {
    if constexpr (is_rt_scalar_v<S>) return s;
    else return v2::as_tag(s);
}
}  // namespace detail

template <class AS, class T, class BS, class Ref> BATCHLAS_DLA_INLINE void store_fma(AS alpha_, T acc, BS beta_, Ref&& y) {
    const auto alpha = detail::rt_or_tag(alpha_);
    const auto beta = detail::rt_or_tag(beta_);
    using B = std::remove_cv_t<decltype(beta)>;
    T out = v2::Arith<T>::zero();
    detail::fma_term(out, alpha, acc);
    if constexpr (is_rt_scalar_v<B>) {
        if (!beta.zero) detail::fma_term(out, beta, static_cast<T>(y));
    } else if constexpr (!std::is_same_v<B, v2::Zero>) {
        detail::fma_term(out, beta, static_cast<T>(y));
    }
    y = out;
}

template <class V> struct ConjIf {
    using value_type = typename V::value_type;
    V v;
    bool on;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const {
        value_type a = v(i);
        if constexpr (v2::Arith<value_type>::is_complex) {
            if (on) a = v2::Arith<value_type>::conj(a);
        }
        return a;
    }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
};
template <class V> BATCHLAS_DLA_INLINE ConjIf<V> conj_if(V v, bool on) { return {v, on}; }

template <int N> struct Member {
    static_assert(N >= 1, "gemv Member: N >= 1");
    static constexpr int size = N, team_size = N, nteams = 1;
    static constexpr bool sub_whole = true;
    static constexpr v2::Scope natural_scope = v2::Scope::Sub;
    int id;
    BATCHLAS_DLA_INLINE int lid() const { return id; }
    BATCHLAS_DLA_INLINE int lane() const { return id; }
    BATCHLAS_DLA_INLINE int team() const { return 0; }
    BATCHLAS_DLA_INLINE bool leader() const { return id == 0; }
};

// Every lane of g must call (lockstep). Closed when Count * stride <= the sub-group size; lanes
// at or above that shift in values no head lane reads.
template <int Count, class G, class T, class Stride = v2::Unit>
BATCHLAS_DLA_INLINE T fold_down(const G& g, T v, Stride stride = {}) {
    static_assert(Count > 0 && (Count & (Count - 1)) == 0, "gemv fold_down: Count must be a power of two");
    using A = v2::Arith<T>;
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

template <int SG, class T> BATCHLAS_DLA_INLINE T leader_sum(const v2::SubGroup<SG>& ex, T v) { return fold_down<SG>(ex.sg, v); }
template <int P, class T> BATCHLAS_DLA_INLINE T leader_sum(const v2::Partition<P, false, sycl::sub_group>& ex, T v) {
    return fold_down<P>(ex.part, v);
}

}  // namespace batchlas::sycl_gemv::dla
