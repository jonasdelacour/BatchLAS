// SPDX-License-Identifier: MIT
#pragma once
// Device LA, the gesvd / ormbr / bdsdc operand adaptors over device::item() operands, waiting for
// core (item(v, b, n) in interop.hh; row / col / sub members on ItemMat). The gesvd driver's
// bidiagonal steps are public (batchlas/device/lapack/gesvd.hh, detail).

#include <type_traits>

#include <batchlas/device.hh>
#include <batchlas/device/lapack/gesvd.hh>

namespace batchlas::device::gesvd_glue {

// ---- operand adaptors ---------------------------------------------------------------------------
// device::item(VectorView, b) with n elements (the view's own size may be larger).
template <class T> BATCHLAS_DEVICE_INLINE ItemVec<T> item_n(const VectorView<T>& v, int b, int n) {
    ItemVec<T> x = item(v, b);
    x.n = n;
    return x;
}
// Column j, row i and the m x n block at (r0, c0) of a matrix operand: A(i, j), A(i, j),
// A(r0 + i, c0 + j), indexed through the operand's own operator() (item()'s host-view address).
template <class M> struct ColOf {
    using value_type = typename M::value_type;
    M a;
    int j;
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i) const { return a(i, j); }
    BATCHLAS_DEVICE_INLINE int size() const { return a.rows(); }
};
template <class M> struct RowOf {
    using value_type = typename M::value_type;
    M a;
    int i;
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int j) const { return a(i, j); }
    BATCHLAS_DEVICE_INLINE int size() const { return a.cols(); }
};
template <class M> struct SubOf {
    using value_type = typename M::value_type;
    M a;
    int r0, c0, m, n;
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i, int j) const { return a(r0 + i, c0 + j); }
    BATCHLAS_DEVICE_INLINE int rows() const { return m; }
    BATCHLAS_DEVICE_INLINE int cols() const { return n; }
};
template <class M> BATCHLAS_DEVICE_INLINE ColOf<M> col_of(const M& a, int j) { return {a, j}; }
template <class M> BATCHLAS_DEVICE_INLINE RowOf<M> row_of(const M& a, int i) { return {a, i}; }
template <class M> BATCHLAS_DEVICE_INLINE SubOf<M> sub_of(const M& a, int r0, int c0, int m, int n) {
    return {a, r0, c0, m, n};
}

}  // namespace batchlas::device::gesvd_glue
