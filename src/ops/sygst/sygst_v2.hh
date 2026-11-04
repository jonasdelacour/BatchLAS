// SPDX-License-Identifier: MIT
#pragma once
// sygst's private device LA v2 ops (core: include/batchlas/device), each waiting on a core form:
//   - trsv_packed / trmv_packed: op(A)^-1 x and op(A) x, op = A^T or A^H, on one work-item with x
//     in a register Frag; A's columns are read as aligned 16-byte packs into W split
//     accumulators and its diagonal as real (a Cholesky factor). Core trsv / trmv are the row
//     form with one accumulator and a complex divide, another rounding order;
//   - herm(A, uplo): the Hermitian matrix one triangle stores with uplo at run time (core
//     herm_view takes it at compile time), real diagonal, one load; .col(j) feeds v2::load;
//   - for_each_tile_col: the group's coalesced sweep over m x m tiles, an item per tile row;
//   - Pack16: the 16-byte-aligned SLM element (core Packet carries no alignment).
// Arithmetic is arith::Fma, bit-equal to the hand kernel's four-fma helpers.

#include <type_traits>

#include <batchlas/device.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

namespace batchlas::sycl_sygst::dla {

namespace v2 = batchlas::device;

// One aligned 16-byte load of consecutive elements (one element when T is 16 bytes).
template <class T> struct alignas(16) Pack16 {
    T e[16 / sizeof(T) > 0 ? 16 / sizeof(T) : 1];
};
template <class T> inline constexpr int pack16_elems = static_cast<int>(sizeof(Pack16<T>::e) / sizeof(T));

BATCHLAS_DLA_INLINE bool in_triangle(Uplo uplo, int i, int j) { return uplo == Uplo::Lower ? i >= j : i <= j; }

template <class H> struct HermCol {
    H h;
    int j;
    using value_type = typename H::value_type;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const { return h(i, j); }
    BATCHLAS_DLA_INLINE int size() const { return h.rows(); }
};

// The Hermitian (or symmetric) matrix whose uplo triangle M stores: the diagonal read as real,
// the other triangle as the conjugate of its mirror.
template <class M> struct HermView {
    M a;
    Uplo uplo;
    using value_type = typename M::value_type;
    BATCHLAS_DLA_INLINE value_type operator()(int i, int j) const {
        const bool own = in_triangle(uplo, i, j);  // one load, its address selected
        const value_type x = a(own ? i : j, own ? j : i);
        return i == j ? v2::force_real(x) : own ? x : v2::Arith<value_type>::conj(x);
    }
    BATCHLAS_DLA_INLINE int rows() const { return a.rows(); }
    BATCHLAS_DLA_INLINE int cols() const { return a.cols(); }
    BATCHLAS_DLA_INLINE HermCol<HermView> col(int j) const { return {*this, j}; }
};
template <class M> BATCHLAS_DLA_INLINE HermView<M> herm(M a, Uplo uplo) { return {a, uplo}; }

// A group of `size` items over `tiles` m x m tiles, column by column: item lid owns row lid % m
// of tile columns lid / m, lid / m + size / m, ...; f(tile, row, col). When m does not divide
// size, the last items revisit columns another item covers: f must be idempotent.
template <class F> BATCHLAS_DLA_INLINE void for_each_tile_col(int lid, int size, int m, int tiles, F&& f) {
    const int row = lid % m, step = size / m;
    for (int idx = lid / m; idx < tiles * m; idx += step) {
        const int g = idx / m, c = idx - g * m;
        f(g, row, c);
    }
}

namespace detail {
// x_i combines column i of A with the other entries of x (Below: p < i, else p > i); term p
// goes to accumulator p mod W, summed as a tree. Solve: x_i = (x_i - sum) / a_ii, else
// x_i = a_ii x_i + sum. In place, so i is visited before every entry whose sum still needs it.
template <bool Below, bool Solve, bool Conj, int W, class T, int N, class L, class MA>
BATCHLAS_DLA_INLINE void coldot_tri(const MA& a, v2::Frag<T, N, L>& x) {
    static_assert(W > 0 && (W & (W - 1)) == 0, "sygst trsv_packed/trmv_packed: W is a power of two");
    using A = v2::Arith<T>;
    using Ar = v2::arith_t<v2::arith::Fma, T>;
    constexpr int V = pack16_elems<T>;
    constexpr int K = v2::static_extent_v<std::remove_cv_t<decltype(a.n)>>;  // -1: a run-time extent
    constexpr int Trip = K > 0 ? K : N;
    static_assert(Trip <= N, "sygst trsv_packed/trmv_packed: the extent exceeds the fragment");
    constexpr bool ascending = Below == Solve;
    const int n = a.cols();
    auto step = [&](T& acc, const T& aij, const T& xj) BATCHLAS_DLA_BODY {
        if constexpr (Solve) Ar::msub(acc, Conj ? A::conj(aij) : aij, xj);  // acc - op(a) x
        else if constexpr (Conj) Ar::mac_conj(acc, aij, xj);
        else Ar::mac(acc, aij, xj);
    };
    BATCHLAS_UNROLL_FULL
    for (int s = 0; s < Trip; ++s) {
        const int i = ascending ? s : Trip - 1 - s;
        if (i < n) {
            T acc[W];
            BATCHLAS_UNROLL_FULL
            for (int k = 0; k < W; ++k) acc[k] = A::zero();
            const T* const line = &a(0, i);  // column i of A
            if constexpr (Below) {
                BATCHLAS_UNROLL_FULL
                for (int p0 = 0; p0 < i; p0 += V) {
                    const Pack16<T> pk = *reinterpret_cast<const Pack16<T>*>(line + p0);
                    BATCHLAS_UNROLL_FULL
                    for (int u = 0; u < V; ++u)
                        if (p0 + u < i) step(acc[(p0 + u) % W], pk.e[u], x[p0 + u]);
                }
            } else {
                BATCHLAS_UNROLL_FULL
                for (int p0 = (i + 1) / V * V; p0 < Trip; p0 += V) {
                    if (p0 >= n) continue;
                    const Pack16<T> pk = *reinterpret_cast<const Pack16<T>*>(line + p0);
                    BATCHLAS_UNROLL_FULL
                    for (int u = 0; u < V; ++u)
                        if (p0 + u > i && p0 + u < n) step(acc[(p0 + u) % W], pk.e[u], x[p0 + u]);
                }
            }
            BATCHLAS_UNROLL_FULL
            for (int w = W / 2; w >= 1; w /= 2) {
                BATCHLAS_UNROLL_FULL
                for (int k = 0; k < w; ++k) acc[k] = A::add(acc[2 * k], acc[2 * k + 1]);
            }
            const auto d = A::re(line[i]);
            if constexpr (Solve) x[i] = v2::div_real(A::add(x[i], acc[0]), d);
            else x[i] = A::add(v2::scale_real(d, x[i]), acc[0]);
        }
    }
}
}  // namespace detail

// x := op(A)^-1 x in place, op = Trans or ConjTrans, A n x n triangular (UL stored) with a real
// diagonal; n is A's extent (compile-time for Ext<K>). Every column start of A is 16-byte
// aligned. x is the caller's Frag: a by-value copy hoists every load of A (REG 37 -> 70).
template <Uplo UL, Transpose TR, int W = 4, class T, int N, class L, class MA>
BATCHLAS_DLA_INLINE void trsv_packed(const v2::Thread&, const MA& a, v2::Frag<T, N, L>& x) {
    static_assert(TR != Transpose::NoTrans, "sygst trsv_packed: the column form is op(A) = A^T or A^H");
    detail::coldot_tri<UL == Uplo::Upper, true, TR == Transpose::ConjTrans, W>(a, x);
}

// x := op(A) x in place, op = Trans or ConjTrans, A as for trsv_packed.
template <Uplo UL, Transpose TR, int W = 4, class T, int N, class L, class MA>
BATCHLAS_DLA_INLINE void trmv_packed(const v2::Thread&, const MA& a, v2::Frag<T, N, L>& x) {
    static_assert(TR != Transpose::NoTrans, "sygst trmv_packed: the column form is op(A) = A^T or A^H");
    detail::coldot_tri<UL == Uplo::Upper, false, TR == Transpose::ConjTrans, W>(a, x);
}

}  // namespace batchlas::sycl_sygst::dla
