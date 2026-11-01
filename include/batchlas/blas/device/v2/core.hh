// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, core vocabulary: extents, memory tags, scalar tags, the arithmetic trait and
// the algorithm tags. Every choice a kernel's SASS depends on is a type here, never a runtime
// value.

#include <complex>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/enums.hh>

#ifndef BATCHLAS_DLA_INLINE
#define BATCHLAS_DLA_INLINE __attribute__((always_inline)) inline
#endif

namespace batchlas::device::v2 {

// ---- extents ---------------------------------------------------------------------------
template <int N> struct Ext {
    static constexpr int value = N;
    BATCHLAS_DLA_INLINE constexpr operator int() const { return N; }
};
struct Dyn {
    int value;
    BATCHLAS_DLA_INLINE constexpr operator int() const { return value; }
};
using Unit = Ext<1>;

enum class Mem { Global, Local, Reg };
enum class Scope { Auto, Sub, Group };

// ---- layouts: who owns which element -------------------------------------------------
struct Cyclic {};      // element / row i owned by lid == i mod size (r = lid + k*size)
struct RowFrag {};     // lane l holds matrix row l in registers
struct ColPerLane {};  // lane l owns column l
struct SgPerCol {};    // team t owns columns c = t mod nteams; its lanes stride the rows
struct Uniform {};     // replicated value (every reduction result)

// ---- scalar tags: alpha/beta values the compiler can see --------------------------------
struct One {};
struct Zero {};
struct NegOne {};
template <class T> struct Scalar {
    T v;
};
template <class T> Scalar(T) -> Scalar<T>;

template <class S> inline constexpr bool is_scalar_tag_v = false;
template <> inline constexpr bool is_scalar_tag_v<One> = true;
template <> inline constexpr bool is_scalar_tag_v<Zero> = true;
template <> inline constexpr bool is_scalar_tag_v<NegOne> = true;
template <class T> inline constexpr bool is_scalar_tag_v<Scalar<T>> = true;

// A plain value becomes Scalar<T>; a tag passes through.
template <class S> BATCHLAS_DLA_INLINE constexpr auto as_tag(S s) {
    if constexpr (is_scalar_tag_v<S>) return s;
    else return Scalar<S>{s};
}
template <class S> using tag_t = decltype(as_tag(std::declval<S>()));

// ---- arithmetic trait ---------------------------------------------------------------------
// Inner loops multiply through Arith only: for complex it is explicit real arithmetic, never
// the Annex-G operator* (isnan branches + __mulsc3). Specialise it for another scalar type
// (src/kernels/common/dla_v2_scalar.hh does so for sycl_device::Cx).
template <class T> struct Arith {
    using real_t = T;
    static constexpr bool is_complex = false;
    BATCHLAS_DLA_INLINE static T mul(T a, T b) { return a * b; }
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) { acc += a * b; }
    BATCHLAS_DLA_INLINE static void mac_conj(T& acc, T a, T b) { acc += a * b; }
    BATCHLAS_DLA_INLINE static T conj(T a) { return a; }
    BATCHLAS_DLA_INLINE static T re(T a) { return a; }
    BATCHLAS_DLA_INLINE static T im(T) { return T(0); }
    BATCHLAS_DLA_INLINE static T make(T r, T) { return r; }
    BATCHLAS_DLA_INLINE static T add(T a, T b) { return a + b; }
    BATCHLAS_DLA_INLINE static T sub(T a, T b) { return a - b; }
    BATCHLAS_DLA_INLINE static T neg(T a) { return -a; }
    BATCHLAS_DLA_INLINE static T zero() { return T(0); }
    BATCHLAS_DLA_INLINE static T one() { return T(1); }
};

template <class R> struct Arith<std::complex<R>> {
    using T = std::complex<R>;
    using real_t = R;
    static constexpr bool is_complex = true;
    BATCHLAS_DLA_INLINE static T mul(T a, T b) {
        return T(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
    }
    // The latrd split-loop form: bit-identical to the legacy kernel's mac().
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) {
        const R pr = a.real() * b.real() - a.imag() * b.imag();
        const R pi = a.real() * b.imag() + a.imag() * b.real();
        acc = T(acc.real() + pr, acc.imag() + pi);
    }
    BATCHLAS_DLA_INLINE static void mac_conj(T& acc, T a, T b) {
        const R pr = a.real() * b.real() + a.imag() * b.imag();
        const R pi = a.real() * b.imag() - a.imag() * b.real();
        acc = T(acc.real() + pr, acc.imag() + pi);
    }
    BATCHLAS_DLA_INLINE static T conj(T a) { return T(a.real(), -a.imag()); }
    BATCHLAS_DLA_INLINE static R re(T a) { return a.real(); }
    BATCHLAS_DLA_INLINE static R im(T a) { return a.imag(); }
    BATCHLAS_DLA_INLINE static T make(R r, R i) { return T(r, i); }
    BATCHLAS_DLA_INLINE static T add(T a, T b) { return T(a.real() + b.real(), a.imag() + b.imag()); }
    BATCHLAS_DLA_INLINE static T sub(T a, T b) { return T(a.real() - b.real(), a.imag() - b.imag()); }
    BATCHLAS_DLA_INLINE static T neg(T a) { return T(-a.real(), -a.imag()); }
    BATCHLAS_DLA_INLINE static T zero() { return T(R(0), R(0)); }
    BATCHLAS_DLA_INLINE static T one() { return T(R(1), R(0)); }
};

template <class T> using real_t = typename Arith<std::remove_cv_t<T>>::real_t;

// BLAS |re| + |im| (dcabs1): the key of asum, amax and iamax.
template <class T> BATCHLAS_DLA_INLINE real_t<T> abs1(T a) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return sycl::fabs(A::re(a)) + sycl::fabs(A::im(a));
    else return sycl::fabs(a);
}

// s * x for a tag s; Zero never reads x.
template <class T, class S> BATCHLAS_DLA_INLINE T tag_mul(S s, T x) {
    if constexpr (std::is_same_v<S, One>) return x;
    else if constexpr (std::is_same_v<S, Zero>) return Arith<T>::zero();
    else if constexpr (std::is_same_v<S, NegOne>) return Arith<T>::neg(x);
    else return Arith<T>::mul(static_cast<T>(s.v), x);
}

// y := alpha * acc + beta * y; beta == Zero is a store that never loads y.
template <class A, class B, class T, class Ref> BATCHLAS_DLA_INLINE void store(A alpha, T acc, B beta, Ref&& y) {
    if constexpr (std::is_same_v<B, Zero>) {
        y = tag_mul(alpha, acc);
    } else if constexpr (std::is_same_v<B, One>) {
        y = Arith<T>::add(tag_mul(alpha, acc), static_cast<T>(y));
    } else {
        y = Arith<T>::add(tag_mul(alpha, acc), tag_mul(beta, static_cast<T>(y)));
    }
}

// ---- algorithm tags (every default is a type, chosen from executor and operation) -------
namespace alg {
struct RowOwner {};             // gemv N: one work-item per output row, serial dot, no reduction
struct SgPerCol {};             // gemv T/C: one team (sub-group) per output column, shuffles only
struct Serial {};               // Thread executor
struct GroupReducePerOutput {}; // v1's generic algorithm: one group reduction per output; opt-in only
struct RowOwnerSplit {};        // symv/hemv lower: the latrd split loop (row part, then column part)
struct Cyclic {};               // ger: column loop outside, owned rows inside; no % or /
struct Scaled {};               // nrm2: two-pass lassq, overflow-safe
struct Unscaled {};             // nrm2: sqrt(dotc(x, x)); overflows past sqrt(max)
}  // namespace alg

// Op tags for scratch_elements.
namespace op {
template <int K> struct ReduceN {};
using Reduce = ReduceN<1>;
}  // namespace op

}  // namespace batchlas::device::v2
