// SPDX-License-Identifier: MIT
#pragma once
// v1 view -> v2 operand. Call after batch_item(b): the view's item-0 pointer is the item's.
// The views are free in SASS (lab F1), so these only re-type: int32 extents, Dyn strides.

#include <cassert>
#include <cstdint>

#include <batchlas/blas/device/v2/operand.hh>
#include <batchlas/blas/matrix.hh>

namespace batchlas::device::v2 {

template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE Vec<T, Dyn, Dyn, M> vec(const VectorView<T>& v) {
    return {v.data_ptr(), Dyn{v.size()}, Dyn{v.inc()}};
}
// Unit-stride form: the caller asserts inc() == 1 (checked in debug builds).
template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE Vec<T, Dyn, Unit, M> vec_unit(const VectorView<T>& v) {
    assert(v.inc() == 1);
    return {v.data_ptr(), Dyn{v.size()}};
}
template <Mem M = Mem::Global, class T>
BATCHLAS_DLA_INLINE Mat<T, Dyn, Dyn, Dyn, M> mat(const KernelMatrixView<T, MatrixFormat::Dense>& a) {
    return {a.data(), Dyn{a.rows()}, Dyn{a.cols()}, Dyn{a.ld()}};
}

// item(view, b): batch item b of a v1 view addressed exactly as v1's operator()(i, j, b):
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
    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const { return p[static_cast<int64_t>(b) * stride + j * ld + i]; }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
};
template <class T> struct ItemVec {
    using value_type = std::remove_cv_t<T>;
    using layout = Cyclic;
    static constexpr Mem mem = Mem::Global;
    T* p;
    int b, stride, inc, n;
    BATCHLAS_DLA_INLINE T& operator()(int i) const {
        return p[static_cast<int64_t>(i) * inc + static_cast<int64_t>(b) * stride];
    }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class T> BATCHLAS_DLA_INLINE ItemMat<T> item(const KernelMatrixView<T, MatrixFormat::Dense>& a, int b) {
    return {a.data(), b, a.stride(), a.ld(), a.rows(b), a.cols(b)};
}
template <class T> BATCHLAS_DLA_INLINE ItemMat<T> item(const KernelMatrixView<T, MatrixFormat::Dense>& a, int b, int m, int n) {
    return {a.data(), b, a.stride(), a.ld(), m, n};
}
template <class T> BATCHLAS_DLA_INLINE ItemVec<T> item(const VectorView<T>& v, int b) {
    return {v.data_ptr(), b, v.stride(), v.inc(), v.size()};
}

}  // namespace batchlas::device::v2
