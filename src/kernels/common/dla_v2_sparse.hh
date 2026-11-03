// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, sparse operands and CSR row products (Sparse BLAS naming: us*), private to the
// ILU(k) and native SpMM kernels. Every element expression matches the hand kernel it replaced
// (int64 offsets where that one used them), so the ports keep their SASS. Arithmetic comes from
// core: arith_t<arith::Fma, T> for SpMM (sycl_device::fma_acc / dev_mul), arith::Native for ILU.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/device/v2/core.hh>
#include <batchlas/blas/device/v2/exec.hh>
#include <batchlas/blas/device/v2/operand.hh>
#include <src/sycl/kernel_attrs.hh>

namespace batchlas::device::v2 {

// One scalar access, or a complex as one 2-wide vector access (needs 2 * sizeof(real) alignment).
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

// Relaxed device-scope atomic *p += v. A complex is two independent real atomics, NOT an atomic
// complex add: correct only while nothing reads *p between the first and the last update.
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
// Element i at p[base + i], the sum in Off (int64_t where base = b * stride passes 2^31; core
// Vec's int32 off cannot hold it).
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

// Batch-interleaved (slot-major) storage: slot s of item `off` at p[size_t(s) * stride + off].
// stride and off stay int and widen at each use, as the hand index did: held as size_t they
// cost the ILU eliminate kernel +3..5 REG. Core Vec's index is int32 or signed ptrdiff_t.
template <class T> struct InterleavedVec {
    using value_type = std::remove_cv_t<T>;
    T* p;
    int stride;
    int off;
    BATCHLAS_DLA_INLINE T& operator()(int s) const { return p[static_cast<std::size_t>(s) * stride + off]; }
};

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

// op(B)(r, c) of a column-major B (int64 ld) whose transposition is a kernel argument: NoTrans
// p[c * ld + r], else p[r * ld + c], conjugated when conj (one instantiation, as the hand kernel).
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

// ---- CSR row products ----------------------------------------------------------------------
// acc - sum a.val(p) y(a.idx(p)) over the entries with idx < bound, left to right: the row is
// sorted, so the first idx >= bound ends it (the strict lower part of a CSR row); nothing at or
// past it is read.
template <class Ar, class Row, class Y, class T>
BATCHLAS_DLA_INLINE T usdot_sub_below(const Thread&, T acc, const Row& a, const Y& y, int bound) {
    for (int p = a.begin; p < a.end; ++p) {
        const int j = a.idx(p);
        if (j >= bound) break;
        Ar::msub(acc, a.val(p), y(j));
    }
    return acc;
}

// usmm, one output row: acc[t] += sum_p a.val(p) b(a.idx(p), c0 + t) for c0 + t < width.
template <class Ar, class Row, class B, class T, int NC, class L>
BATCHLAS_DLA_INLINE void usmm_row(const Thread&, const Row& a, const B& b, int c0, int width, Frag<T, NC, L>& acc) {
    for (int p = a.begin; p < a.end; ++p) {
        const T av = a.val(p);
        const int j = a.idx(p);
        BATCHLAS_UNROLL_FULL
        for (int t = 0; t < NC; ++t) {
            const int c = c0 + t;
            if (c < width) Ar::mac(acc[t], av, b(j, c));
        }
    }
}

// C(r, c0 + t) := fma(beta, C, fma(alpha, acc[t], 0)) for c0 + t < width; beta_zero (a kernel
// argument) never reads C.
template <class Ar, class T, int NC, class L, class CM>
BATCHLAS_DLA_INLINE void store_cols(const Thread&, T alpha, const Frag<T, NC, L>& acc, T beta, bool beta_zero, const CM& C, int r,
                                    int c0, int width) {
    BATCHLAS_UNROLL_FULL
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
template <class Ar, class B, class T, int NC, class L>
BATCHLAS_DLA_INLINE void load_cols_scaled(const Thread&, T alpha, const B& b, int r, int c0, int width, Frag<T, NC, L>& w) {
    BATCHLAS_UNROLL_FULL
    for (int t = 0; t < NC; ++t) {
        const int c = c0 + t;
        w[t] = Arith<T>::zero();
        if (c < width) w[t] = Ar::mul(alpha, b(r, c));
    }
}

}  // namespace batchlas::device::v2
