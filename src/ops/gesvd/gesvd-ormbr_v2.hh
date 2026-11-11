// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, the gesvd / ormbr / bdsdc operand adaptors over v2::item() operands, waiting for
// core (item(v, b, n) in interop.hh; row / col / sub members on ItemMat). The gesvd driver's
// bidiagonal steps are public (batchlas/device/lapack/gesvd.hh, gesvd_detail).

#include <type_traits>

#include <batchlas/device.hh>
#include <batchlas/device/lapack/gesvd.hh>

namespace batchlas::device::gesvd_glue {

// ---- operand adaptors ---------------------------------------------------------------------------
// v2::item(VectorView, b) with n elements (the view's own size may be larger).
template <class T> BATCHLAS_DLA_INLINE ItemVec<T> item_n(const VectorView<T>& v, int b, int n) {
    ItemVec<T> x = item(v, b);
    x.n = n;
    return x;
}
// Column j, row i and the m x n block at (r0, c0) of a matrix operand: A(i, j), A(i, j),
// A(r0 + i, c0 + j), indexed through the operand's own operator() (item()'s v1 address).
template <class M> struct ColOf {
    using value_type = typename M::value_type;
    M a;
    int j;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return a(i, j); }
    BATCHLAS_DLA_INLINE int size() const { return a.rows(); }
};
template <class M> struct RowOf {
    using value_type = typename M::value_type;
    M a;
    int i;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int j) const { return a(i, j); }
    BATCHLAS_DLA_INLINE int size() const { return a.cols(); }
};
template <class M> struct SubOf {
    using value_type = typename M::value_type;
    M a;
    int r0, c0, m, n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return a(r0 + i, c0 + j); }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
};
template <class M> BATCHLAS_DLA_INLINE ColOf<M> col_of(const M& a, int j) { return {a, j}; }
template <class M> BATCHLAS_DLA_INLINE RowOf<M> row_of(const M& a, int i) { return {a, i}; }
template <class M> BATCHLAS_DLA_INLINE SubOf<M> sub_of(const M& a, int r0, int c0, int m, int n) {
    return {a, r0, c0, m, n};
}

}  // namespace batchlas::device::gesvd_glue
