// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Host views to device operands: vec and mat from a VectorView or KernelMatrixView, and
/// item(view, b), one batch item addressed as the view addresses it.
/// @ingroup api_device_operands
///
/// Call vec and mat after `batch_item(b)`: the view's item-0 pointer is then the item's. They
/// only re-type, to int32 extents and Dyn strides.

#include <cassert>
#include <cstdint>

#include <batchlas/device/operand.hh>
#include <batchlas/blas/matrix.hh>

namespace batchlas::device {

/// @brief The device vector of a host VectorView (after batch_item): n elements, stride inc().
/// @ingroup api_device_operands
/// @tparam M the memory space
/// @param v the host view of one batch item
/// @return the operand
template <Mem M = Mem::Global, class T> BATCHLAS_DEVICE_INLINE Vec<T, Dyn, Dyn, M> vec(const VectorView<T>& v) {
    return {v.data_ptr(), Dyn{v.size()}, Dyn{v.inc()}};
}
/// @brief The unit-stride device vector of a host VectorView; the caller asserts `inc() == 1`.
/// @ingroup api_device_operands
///
/// A debug build checks the stride.
/// @tparam M the memory space
/// @param v the host view of one batch item, with inc() == 1
/// @return the operand
template <Mem M = Mem::Global, class T> BATCHLAS_DEVICE_INLINE Vec<T, Dyn, Unit, M> vec_unit(const VectorView<T>& v) {
    assert(v.inc() == 1);
    return {v.data_ptr(), Dyn{v.size()}};
}
/// @brief The device matrix of a dense host KernelMatrixView (after batch_item).
/// @ingroup api_device_operands
/// @tparam M the memory space
/// @param a the host view of one batch item
/// @return the operand: rows(), cols() and ld() of the view
template <Mem M = Mem::Global, class Scalar>
BATCHLAS_DEVICE_INLINE Mat<Scalar, Dyn, Dyn, Dyn, M> mat(const KernelMatrixView<Scalar, MatrixFormat::Dense>& a) {
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

/// @brief Batch item b of a dense host matrix view, addressed exactly as its `operator()(i, j, b)`.
/// @ingroup api_device_operands
///
/// Element (i, j) is `p[int64(b) * stride + j * ld + i]`: one index expression per element, with
/// the batch term formed lazily, so no pointer is moved. item() builds it.
/// @tparam Scalar the element type
// With the extents given, item() skips the active_rows / active_cols load of a heterogeneous view
// (stedc: Q.batch_item costs one LDG).
template <class Scalar> struct ItemMat {
    using value_type = std::remove_cv_t<Scalar>;  ///< The element type without cv.
    using layout = Cyclic;  ///< Cyclic.
    static constexpr Mem mem = Mem::Global;  ///< Mem::Global.
    Scalar* p;  ///< The view's data pointer (item 0).
    ///@{
    /// @brief The batch item, the batch stride, the leading dimension, the rows and the columns.
    int b, stride, ld, m, n;
    ///@}
    /// @brief Element (i, j) of batch item b.
    /// @param i the row
    /// @param j the column
    /// @return a reference to the element
    BATCHLAS_DEVICE_INLINE Scalar& operator()(int i, int j) const { return p[static_cast<int64_t>(b) * stride + j * ld + i]; }
    /// @brief The number of rows.
    BATCHLAS_DEVICE_INLINE int rows() const { return m; }
    /// @brief The number of columns.
    BATCHLAS_DEVICE_INLINE int cols() const { return n; }
    /// @brief Column j as a vector, each element through operator().
    /// @param j the column
    /// @return the column view
    BATCHLAS_DEVICE_INLINE detail::ColOf<ItemMat> col(int j) const { return {*this, j}; }
    /// @brief Row i as a vector, each element through operator().
    /// @param i the row
    /// @return the row view
    BATCHLAS_DEVICE_INLINE detail::RowOf<ItemMat> row(int i) const { return {*this, i}; }
    /// @brief The rows_ x cols_ block at (r0, c0), each element through operator().
    /// @param r0 the block's first row
    /// @param c0 the block's first column
    /// @param rows_ the block's rows
    /// @param cols_ the block's columns
    /// @return the block view
    BATCHLAS_DEVICE_INLINE detail::SubOf<ItemMat> sub(int r0, int c0, int rows_, int cols_) const { return {*this, r0, c0, rows_, cols_}; }
};
/// @brief Batch item b of a host vector view: element i is `p[int64(i) * inc + int64(b) * stride]`.
/// @ingroup api_device_operands
/// @tparam T the element type
template <class T> struct ItemVec {
    using value_type = std::remove_cv_t<T>;  ///< The element type without cv.
    using layout = Cyclic;  ///< Cyclic.
    static constexpr Mem mem = Mem::Global;  ///< Mem::Global.
    T* p;  ///< The view's data pointer (item 0).
    ///@{
    /// @brief The batch item, the batch stride, the element stride and the size.
    int b, stride, inc, n;
    ///@}
    /// @brief Element i of batch item b.
    /// @param i the index
    /// @return a reference to the element
    BATCHLAS_DEVICE_INLINE T& operator()(int i) const {
        return p[static_cast<int64_t>(i) * inc + static_cast<int64_t>(b) * stride];
    }
    /// @brief The number of elements.
    BATCHLAS_DEVICE_INLINE int size() const { return n; }
};
/// @brief Batch item b of a dense host matrix view, with the view's extents for item b.
/// @ingroup api_device_operands
/// @param a the host view
/// @param b the batch item
/// @return the ItemMat
template <class Scalar> BATCHLAS_DEVICE_INLINE ItemMat<Scalar> item(const KernelMatrixView<Scalar,
                                                                    MatrixFormat::Dense>& a, int b) {
    return {a.data(), b, a.stride(), a.ld(), a.rows(b), a.cols(b)};
}
/// @brief Batch item b of a dense host matrix view, with the given extents.
/// @ingroup api_device_operands
///
/// The extents skip the active_rows / active_cols load of a heterogeneous view.
/// @param a the host view
/// @param b the batch item
/// @param m the rows
/// @param n the columns
/// @return the ItemMat
template <class Scalar> BATCHLAS_DEVICE_INLINE ItemMat<Scalar> item(const KernelMatrixView<Scalar, MatrixFormat::Dense>& a, int b, int m, int n) {
    return {a.data(), b, a.stride(), a.ld(), m, n};
}
/// @brief Batch item b of a host vector view, with the view's size.
/// @ingroup api_device_operands
/// @param v the host view
/// @param b the batch item
/// @return the ItemVec
template <class T> BATCHLAS_DEVICE_INLINE ItemVec<T> item(const VectorView<T>& v, int b) {
    return {v.data_ptr(), b, v.stride(), v.inc(), v.size()};
}
/// @brief Batch item b of a host vector view, with n elements (the view's own size may be larger).
/// @ingroup api_device_operands
/// @param v the host view
/// @param b the batch item
/// @param n the number of elements
/// @return the ItemVec
template <class T> BATCHLAS_DEVICE_INLINE ItemVec<T> item(const VectorView<T>& v, int b, int n) {
    return {v.data_ptr(), b, v.stride(), v.inc(), n};
}

}  // namespace batchlas::device
