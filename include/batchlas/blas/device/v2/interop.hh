// SPDX-License-Identifier: MIT
#pragma once
// v1 view -> v2 operand. Call after batch_item(b): the view's item-0 pointer is the item's.
// The views are free in SASS (lab F1), so these only re-type: int32 extents, Dyn strides.

#include <cassert>

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

}  // namespace batchlas::device::v2
