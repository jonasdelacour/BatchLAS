// SPDX-License-Identifier: MIT
#pragma once
// host view -> device operand. Call after batch_item(b): the view's item-0 pointer is the item's.
// The views are free in SASS (lab F1), so these only re-type: int32 extents, Dyn strides.

#include <cassert>
#include <cstdint>

#include <batchlas/device/operand.hh>
#include <batchlas/blas/matrix.hh>

namespace batchlas::device {

template <Mem M = Mem::Global, class T> BATCHLAS_DEVICE_INLINE Vec<T, Dyn, Dyn, M> vec(const VectorView<T>& v) {
    return {v.data_ptr(), Dyn{v.size()}, Dyn{v.inc()}};
}
// Unit-stride form: the caller asserts inc() == 1 (checked in debug builds).
template <Mem M = Mem::Global, class T> BATCHLAS_DEVICE_INLINE Vec<T, Dyn, Unit, M> vec_unit(const VectorView<T>& v) {
    assert(v.inc() == 1);
    return {v.data_ptr(), Dyn{v.size()}};
}
template <Mem M = Mem::Global, class T>
BATCHLAS_DEVICE_INLINE Mat<T, Dyn, Dyn, Dyn, M> mat(const KernelMatrixView<T, MatrixFormat::Dense>& a) {
    return {a.data(), Dyn{a.rows()}, Dyn{a.cols()}, Dyn{a.ld()}};
}

namespace detail {
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
    BATCHLAS_DEVICE_INLINE ColOf<SubOf> col(int j) const { return {*this, j}; }
    BATCHLAS_DEVICE_INLINE RowOf<SubOf> row(int i) const { return {*this, i}; }
};
}  // namespace detail

// item(view, b): batch item b of a host view addressed exactly as its operator()(i, j, b):
// data_[int64(b) * stride + j * ld + i] (vector: data_[int64(i) * inc + int64(b) * stride]),
// one index expression per element with the batch term formed lazily (no pointer is moved).
// With the extents given, the active_rows/active_cols load of a heterogeneous view is skipped
// (stedc: Q.batch_item costs one LDG).
template <class T> struct ItemMat {
    using value_type = std::remove_cv_t<T>;
    using layout = Cyclic;
    static constexpr Mem mem = Mem::Global;
    T* p;
    int b, stride, ld, m, n;
    BATCHLAS_DEVICE_INLINE T& operator()(int i, int j) const { return p[static_cast<int64_t>(b) * stride + j * ld + i]; }
    BATCHLAS_DEVICE_INLINE int rows() const { return m; }
    BATCHLAS_DEVICE_INLINE int cols() const { return n; }
    // Column j, row i, the m x n block at (r0, c0): each element through operator() above.
    BATCHLAS_DEVICE_INLINE detail::ColOf<ItemMat> col(int j) const { return {*this, j}; }
    BATCHLAS_DEVICE_INLINE detail::RowOf<ItemMat> row(int i) const { return {*this, i}; }
    BATCHLAS_DEVICE_INLINE detail::SubOf<ItemMat> sub(int r0, int c0, int rows_, int cols_) const { return {*this, r0, c0, rows_, cols_}; }
};
template <class T> struct ItemVec {
    using value_type = std::remove_cv_t<T>;
    using layout = Cyclic;
    static constexpr Mem mem = Mem::Global;
    T* p;
    int b, stride, inc, n;
    BATCHLAS_DEVICE_INLINE T& operator()(int i) const {
        return p[static_cast<int64_t>(i) * inc + static_cast<int64_t>(b) * stride];
    }
    BATCHLAS_DEVICE_INLINE int size() const { return n; }
};
template <class T> BATCHLAS_DEVICE_INLINE ItemMat<T> item(const KernelMatrixView<T, MatrixFormat::Dense>& a, int b) {
    return {a.data(), b, a.stride(), a.ld(), a.rows(b), a.cols(b)};
}
template <class T> BATCHLAS_DEVICE_INLINE ItemMat<T> item(const KernelMatrixView<T, MatrixFormat::Dense>& a, int b, int m, int n) {
    return {a.data(), b, a.stride(), a.ld(), m, n};
}
template <class T> BATCHLAS_DEVICE_INLINE ItemVec<T> item(const VectorView<T>& v, int b) {
    return {v.data_ptr(), b, v.stride(), v.inc(), v.size()};
}
// With n elements (the view's own size may be larger).
template <class T> BATCHLAS_DEVICE_INLINE ItemVec<T> item(const VectorView<T>& v, int b, int n) {
    return {v.data_ptr(), b, v.stride(), v.inc(), n};
}

}  // namespace batchlas::device
