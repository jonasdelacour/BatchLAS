// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, sparse extension (Sparse BLAS naming: us*). Operands for a CSR row segment, for
// batch-interleaved (slot-major) storage and for a dense op(B) whose transposition is a kernel
// argument, plus the products the CSR kernels are made of: the row residual of a triangular
// solve, a gathered axpy, the CSR-row times dense-block product, its transposed rank-1 scatter
// and the max-modulus norm. Every element expression matches the hand-written kernels it
// replaced (int64 offsets where those used them), so the ports keep their SASS.
//
// Two arithmetic policies, beside Arith<T> (the default for new code):
//   FmaChainArith<T>  the explicit fma chain of sycl_device::fma_acc / dev_mul (GEMM order);
//                     identical under every -ffp-contract setting.
//   NativeArith<T>    the scalar's own operators, each update one expression (acc -= a * b):
//                     -ffp-contract=on fuses within one expression only, and std::complex keeps
//                     its own multiply and divide. For strict ports of kernels written that way.
// Sibling ext work packages define the same two policies under other names (FmaArith /
// OpArith); folding them into core is an integration step, so these names differ (one ODR
// definition per name across ext headers linked into one device image).

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>
#include <batchlas/blas/device/v2/core.hh>
#include <batchlas/blas/device/v2/exec.hh>
#include <batchlas/blas/device/v2/level1.hh>
#include <batchlas/blas/device/v2/operand.hh>

#if BATCHLAS_SYCL_IMPL_ACPP
#define BATCHLAS_DLA_SP_UNROLL _Pragma("clang loop unroll(full)")  // SSCP JIT: bare unroll is partial
#else
#define BATCHLAS_DLA_SP_UNROLL _Pragma("unroll")
#endif

namespace batchlas::device::v2 {

// ---- arithmetic policies -----------------------------------------------------------------
template <class T> struct FmaChainArith {
    using A = Arith<T>;
    BATCHLAS_DLA_INLINE static T mul(T a, T b) {
        if constexpr (A::is_complex) {
            return A::make(sycl::fma(A::re(a), A::re(b), -A::im(a) * A::im(b)),
                           sycl::fma(A::re(a), A::im(b), A::im(a) * A::re(b)));
        } else {
            return a * b;
        }
    }
    // re = fma(-ai, bi, fma(ar, br, re)); im = fma(ai, br, fma(ar, bi, im)).
    BATCHLAS_DLA_INLINE static void mac(T& acc, T a, T b) {
        if constexpr (A::is_complex) {
            auto re = sycl::fma(A::re(a), A::re(b), A::re(acc));
            re = sycl::fma(-A::im(a), A::im(b), re);
            auto im = sycl::fma(A::re(a), A::im(b), A::im(acc));
            im = sycl::fma(A::im(a), A::re(b), im);
            acc = A::make(re, im);
        } else {
            acc = sycl::fma(a, b, acc);
        }
    }
};

template <class T> struct NativeArith {
    BATCHLAS_DLA_INLINE static void msub(T& acc, const T& a, const T& b) { acc -= a * b; }
    BATCHLAS_DLA_INLINE static T div(const T& a, const T& b) { return a / b; }
};

// The default msub for new code: acc + (-a) b through Arith (real: the same fma as NativeArith).
template <class T> struct ArithMsub {
    BATCHLAS_DLA_INLINE static void msub(T& acc, const T& a, const T& b) { Arith<T>::mac(acc, Arith<T>::neg(a), b); }
};

// |x| with the true modulus (LAPACK ABS, not BLAS abs1): hypot for complex.
template <class T> BATCHLAS_DLA_INLINE real_t<T> abs_modulus(const T& x) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return sycl::hypot(A::re(x), A::im(x));
    else return static_cast<real_t<T>>(std::abs(x));
}

// ---- element access: one scalar access, or a complex as one 2-wide vector access ------------
// Pair needs 2 * sizeof(real) alignment; a real T ignores it.
template <bool Pair> struct PairAccess {
    template <class T> BATCHLAS_DLA_INLINE static std::remove_cv_t<T> load(T* p) {
        using U = std::remove_cv_t<T>;
        using A = Arith<U>;
        if constexpr (Pair && A::is_complex) {
            using R = typename A::real_t;
            const sycl::vec<R, 2> v = *reinterpret_cast<const sycl::vec<R, 2>*>(p);
            return A::make(v[0], v[1]);
        } else {
            return *p;
        }
    }
    template <class T> BATCHLAS_DLA_INLINE static void store(T* p, T x) {
        using A = Arith<T>;
        if constexpr (Pair && A::is_complex) {
            using R = typename A::real_t;
            *reinterpret_cast<sycl::vec<R, 2>*>(p) = sycl::vec<R, 2>(A::re(x), A::im(x));
        } else {
            *p = x;
        }
    }
};

// Relaxed device-scope atomic y += v. A complex is two independent real atomics, NOT an atomic
// complex add: correct only while nothing reads y between the first and the last update.
template <class T> BATCHLAS_DLA_INLINE void atomic_accumulate(T* p, T v) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        using R = typename A::real_t;
        atomic_accumulate(reinterpret_cast<R*>(p), A::re(v));
        atomic_accumulate(reinterpret_cast<R*>(p) + 1, A::im(v));
    } else {
        sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(*p);
        ref.fetch_add(v);
    }
}

// ---- operands ------------------------------------------------------------------------------
// Element i at p[base + i], the offset in Off arithmetic (int64_t when base = b * stride).
template <class T, class Off = int> struct BasedVec {
    using value_type = std::remove_cv_t<T>;
    T* p;
    Off base;
    BATCHLAS_DLA_INLINE T& operator()(int i) const { return p[base + i]; }
};

// Read-only element i = PairAccess<Pair>::load(p + i).
template <class T, bool Pair = false> struct LoadVec {
    using value_type = std::remove_cv_t<T>;
    T* p;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const { return PairAccess<Pair>::load(p + i); }
};

// Batch-interleaved (slot-major) storage: slot s of item `off` at p[s * stride + off], in
// size_t arithmetic (slots * batch passes 2^31).
template <class T> struct InterleavedVec {
    using value_type = std::remove_cv_t<T>;
    T* p;
    std::size_t stride;
    std::size_t off;
    BATCHLAS_DLA_INLINE T& operator()(int s) const { return p[static_cast<std::size_t>(s) * stride + off]; }
};

// Elements [first, first + n) of v as a sized vector (the shape v2 ops iterate).
template <class V> struct SpRange {
    using value_type = typename V::value_type;
    V v;
    int first;
    int n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int k) const { return v(first + k); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class V> BATCHLAS_DLA_INLINE SpRange<V> sp_range(V v, int first, int n) { return {v, first, n}; }

// Sparse BLAS gather: element k = v(idx[k]), k < n.
template <class V> struct SpGather {
    using value_type = typename V::value_type;
    V v;
    const int* idx;
    int n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int k) const { return v(idx[k]); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class V> BATCHLAS_DLA_INLINE SpGather<V> us_gather(V v, const int* idx, int n) { return {v, idx, n}; }

// One CSR row segment: entries p in [begin, end) are (val(p), idx(p)), positions absolute.
template <class VV, class IV> struct SpRow {
    using value_type = typename VV::value_type;
    VV vals;
    IV cols;
    int begin;
    int end;
    BATCHLAS_DLA_INLINE value_type val(int p) const { return vals(p); }
    BATCHLAS_DLA_INLINE int idx(int p) const { return cols(p); }
};
template <class VV, class IV> BATCHLAS_DLA_INLINE SpRow<VV, IV> sp_row(VV vals, IV cols, int begin, int end) {
    return {vals, cols, begin, end};
}

// op(B)(r, c) of a column-major B (int64 ld) whose transposition is a kernel argument:
// NoTrans p[c * ld + r], else p[r * ld + c], conjugated when conj. A compile-time Transpose
// would be the v2 form; this keeps the base kernel's single instantiation.
template <class T, bool Pair = false> struct RtOpMat {
    using value_type = std::remove_cv_t<T>;
    T* p;
    std::int64_t ld;
    bool notrans;
    bool conj;
    BATCHLAS_DLA_INLINE value_type operator()(int r, int c) const {
        value_type v = PairAccess<Pair>::load(notrans ? p + static_cast<std::int64_t>(c) * ld + r
                                                      : p + static_cast<std::int64_t>(r) * ld + c);
        if constexpr (Arith<value_type>::is_complex) {
            if (conj) v = Arith<value_type>::conj(v);
        }
        return v;
    }
};

// Column-major C (int64 ld): element (r, c) at p + (c * ld + r).
template <class T, bool Pair = false> struct ColMat64 {
    using value_type = std::remove_cv_t<T>;
    T* p;
    std::int64_t ld;
    BATCHLAS_DLA_INLINE T* ptr(int r, int c) const { return p + (static_cast<std::int64_t>(c) * ld + r); }
    BATCHLAS_DLA_INLINE value_type load(int r, int c) const { return PairAccess<Pair>::load(ptr(r, c)); }
    BATCHLAS_DLA_INLINE void store(int r, int c, value_type x) const { PairAccess<Pair>::store(ptr(r, c), x); }
};

template <class T, int N> BATCHLAS_DLA_INLINE Frag<T, N> frag_fill(T v) {
    Frag<T, N> f;
    BATCHLAS_DLA_SP_UNROLL
    for (int t = 0; t < N; ++t) f[t] = v;
    return f;
}

// ---- level 1 -------------------------------------------------------------------------------
// y := y - a x, each update one Ar::msub (NativeArith: y -= a * x, the ILU elimination update).
template <class Ar, class Ex, class T, class X, class Y> BATCHLAS_DLA_INLINE void axpy_sub(const Ex& ex, const T& a, const X& x, const Y& y) {
    for_each_owned(ex, y.size(), [&](int k) { Ar::msub(y(k), a, x(k)); });
}

// max |x_i| with the true modulus (LAPACK 'M' norm). NaN policy (unlike amax's Propagate): `m >
// s` skips a NaN element, the ILU kernels' rule.
template <Scope S = Scope::Auto, class Ex, class X> BATCHLAS_DLA_INLINE auto norm_max(const Ex& ex, const X& x) {
    using R = real_t<elem_t<X>>;
    R s = R(0);
    for_each_owned(ex, x.size(), [&](int k) {
        const R m = abs_modulus(x(k));
        if (m > s) s = m;
    });
    return detail::scope_reduce<S>(ex, s, sycl::maximum<R>());
}

// acc := acc - sum_p a.val(p) y(a.idx(p)), left to right: the row residual of a triangular solve.
template <class Ar, class Row, class Y, class T> BATCHLAS_DLA_INLINE T usdot_sub(const Thread&, T acc, const Row& a, const Y& y) {
    for (int p = a.begin; p < a.end; ++p) Ar::msub(acc, a.val(p), y(a.idx(p)));
    return acc;
}

// As usdot_sub over the entries with idx < bound only: the row is sorted, so the first idx >=
// bound ends it (the strict lower part of a CSR row); no value at or past it is read.
template <class Ar, class Row, class Y, class T>
BATCHLAS_DLA_INLINE T usdot_sub_below(const Thread&, T acc, const Row& a, const Y& y, int bound) {
    for (int p = a.begin; p < a.end; ++p) {
        const int j = a.idx(p);
        if (j >= bound) break;
        Ar::msub(acc, a.val(p), y(j));
    }
    return acc;
}

// ---- CSR row x dense block (Sparse BLAS usmm, one output row) -------------------------------
// acc[t] += sum_p a.val(p) b(a.idx(p), c0 + t) for every t < NC with c0 + t < width.
template <class Ar, class Row, class B, class T, int NC, class L>
BATCHLAS_DLA_INLINE void usmm_row(const Thread&, const Row& a, const B& b, int c0, int width, Frag<T, NC, L>& acc) {
    for (int p = a.begin; p < a.end; ++p) {
        const T av = a.val(p);
        const int j = a.idx(p);
        BATCHLAS_DLA_SP_UNROLL
        for (int t = 0; t < NC; ++t) {
            const int c = c0 + t;
            if (c < width) Ar::mac(acc[t], av, b(j, c));
        }
    }
}

// C(r, c0 + t) := alpha acc[t] + beta C(r, c0 + t) as fma(beta, C, fma(alpha, acc, 0)) for
// c0 + t < width; beta_zero (a kernel argument) never reads C.
template <class Ar, class T, int NC, class L, class CM>
BATCHLAS_DLA_INLINE void store_cols(const Thread&, T alpha, const Frag<T, NC, L>& acc, T beta, bool beta_zero, const CM& C, int r, int c0,
                                    int width) {
    BATCHLAS_DLA_SP_UNROLL
    for (int t = 0; t < NC; ++t) {
        const int c = c0 + t;
        if (c < width) {
            T out = Arith<T>::zero();
            Ar::mac(out, alpha, acc[t]);
            if (!beta_zero) Ar::mac(out, beta, C.load(r, c));
            C.store(r, c, out);
        }
    }
}

// w[t] := alpha b(r, c0 + t) for c0 + t < width, else 0: a scaled row block of op(B).
template <class Ar, int NC, class B, class T> BATCHLAS_DLA_INLINE Frag<T, NC> load_cols_scaled(const Thread&, T alpha, const B& b, int r, int c0, int width) {
    Frag<T, NC> w;
    BATCHLAS_DLA_SP_UNROLL
    for (int t = 0; t < NC; ++t) {
        const int c = c0 + t;
        w[t] = Arith<T>::zero();
        if (c < width) w[t] = Ar::mul(alpha, b(r, c));
    }
    return w;
}

// The transposed product's scatter: C(a.idx(p), c0 + t) += conj?(a.val(p)) w[t] by atomics, for
// c0 + t < width; an index outside [0, rows) is skipped (unsigned compare), never written.
template <class Ar, class Row, class T, int NC, class L>
BATCHLAS_DLA_INLINE void usger_atomic(const Thread&, const Row& a, bool conj_a, const Frag<T, NC, L>& w, T* C, std::int64_t ldc, int rows,
                                      int c0, int width) {
    for (int p = a.begin; p < a.end; ++p) {
        T av = a.val(p);
        if constexpr (Arith<T>::is_complex) {
            if (conj_a) av = Arith<T>::conj(av);
        }
        const int j = a.idx(p);
        if (static_cast<unsigned>(j) >= static_cast<unsigned>(rows)) continue;
        BATCHLAS_DLA_SP_UNROLL
        for (int t = 0; t < NC; ++t) {
            const int c = c0 + t;
            if (c < width) atomic_accumulate(C + static_cast<std::int64_t>(c) * ldc + j, Ar::mul(av, w[t]));
        }
    }
}

// y := beta_zero ? 0 : beta y; beta_zero (a kernel argument) never reads y.
template <class Ar, class T> BATCHLAS_DLA_INLINE void scal_or_zero(T beta, bool beta_zero, T& y) {
    y = beta_zero ? Arith<T>::zero() : Ar::mul(beta, y);
}

}  // namespace batchlas::device::v2
