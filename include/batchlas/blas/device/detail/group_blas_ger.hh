// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/blas/device/detail/group_blas_common.hh>

namespace batchlas::device {

namespace detail::generic {

/// @brief Generic rank-1 update: each work-item owns a strided set of elements of \f$A\f$.
/// @param[in]     group      executor; every work-item must call
/// @param[in]     x          left vector \f$x\f$ of length \f$m\f$ = `operand.a.rows()`
/// @param[in,out] operand    \f$y\f$ (read), the \f$m \times n\f$ matrix \f$A\f$ (updated in place) and \f$\alpha\f$ (read)
/// @param[in]     transform  whether \f$x\f$ and \f$y\f$ are conjugated
template <typename Group, typename T>
inline constexpr void ger(const Group& group,
                          const VectorView<T>& x,
                          Rank1UpdateOperand<T> operand,
                          OuterProductTransform transform) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int row_extent = operand.a.rows();
    const int col_extent = operand.a.cols();

    for (int linear_index = local_id; linear_index < row_extent * col_extent; linear_index += local_size) {
        const int row = linear_index % row_extent;
        const int col = linear_index / row_extent;
        const T lhs = detail::maybe_conjugate(x(row), transform.conjugate_x);
        const T rhs = detail::maybe_conjugate(operand.y(col), transform.conjugate_y);
        detail::accumulate_rank1_output(operand, row, col, lhs * rhs);
    }
}

} // namespace detail::generic

namespace detail {

/// @brief Validates the operands and runs the generic ger with the conjugation flags of `Tag`.
/// @param[in]     group    executor; every work-item must call
/// @param[in]     x        left vector \f$x\f$ of length \f$m\f$ = `operand.a.rows()`
/// @param[in,out] operand  \f$y\f$ (read), the \f$m \times n\f$ matrix \f$A\f$ (updated in place) and \f$\alpha\f$ (read)
template <typename Tag, typename Group, typename T>
inline constexpr void dispatch_ger(const Group& group,
                                   const VectorView<T>& x,
                                   Rank1UpdateOperand<T> operand) {
    validate_rank1_operand(x, operand);
    generic::ger(group,
                 x,
                 operand,
                 OuterProductTransform{.conjugate_x = Tag::conjugate_x, .conjugate_y = Tag::conjugate_y});
}

} // namespace detail

/// @addtogroup api_device_blas
/// @{

/// @brief Rank-1 update \f$ A := A + \alpha\,\tilde{x}\,\tilde{y}^{T} \f$, with \f$\tilde{x}\f$, \f$\tilde{y}\f$ optionally conjugated.
///
/// geru() and gerc() are the BLAS-named spellings (no conjugation, and
/// conjugated `y`). Each element of `A` is owned by one work-item, so no
/// reduction and no barrier is involved.
/// @tparam ConjugateXV  use \f$\bar{x}\f$
/// @tparam ConjugateYV  use \f$\bar{y}\f$
/// @tparam Group        `sycl::group`, `sycl::sub_group` or an `nd_item`; every work-item must call
/// @param[in] group       executor whose work-items share the elements of \f$A\f$; every work-item must call
/// @param[in] x           left vector \f$x\f$ of length \f$m\f$ = `operand.a.rows()`; one problem, not a batch
/// @param[in,out] operand  \f$y\f$ (read, length \f$n\f$ = `operand.a.cols()`), the \f$m \times n\f$ matrix \f$A\f$ (updated in place) and \f$\alpha\f$ (read)
/// @pre single-problem views; sizes as above. Checked by `assert` only.
template <bool ConjugateXV = false, bool ConjugateYV = false, typename Group, typename T>
inline constexpr void ger(const Group& group,
                          const VectorView<T>& x,
                          Rank1UpdateOperand<T> operand) {
    detail::dispatch_ger<OuterProductTransformTag<ConjugateXV, ConjugateYV>>(group, x, operand);
}

/// @brief As ger(), with an explicit DeviceBlasPolicy (ignored: one implementation).
/// @tparam Policy       DeviceBlasPolicy; accepted for API symmetry and ignored (one implementation)
/// @tparam ConjugateXV  use \f$\bar{x}\f$
/// @tparam ConjugateYV  use \f$\bar{y}\f$
/// @param[in] group       executor whose work-items share the elements of \f$A\f$; every work-item must call
/// @param[in] x           left vector \f$x\f$ of length \f$m\f$ = `operand.a.rows()`; one problem, not a batch
/// @param[in,out] operand  \f$y\f$ (read, length \f$n\f$ = `operand.a.cols()`), the \f$m \times n\f$ matrix \f$A\f$ (updated in place) and \f$\alpha\f$ (read)
template <DeviceBlasPolicy Policy,
          bool ConjugateXV = false,
          bool ConjugateYV = false,
          typename Group,
          typename T>
inline constexpr void ger(const Group& group,
                          const VectorView<T>& x,
                          Rank1UpdateOperand<T> operand) {
    (void)Policy;
    ger<ConjugateXV, ConjugateYV>(group, x, operand);
}

/// @brief As ger(), with the operands passed separately instead of in an operand struct.
/// @param[in] group       executor whose work-items share the elements of \f$A\f$; every work-item must call
/// @param[in] x           left vector \f$x\f$ of length \f$m\f$ = `a.rows()`; one problem, not a batch
/// @param[in] y           right vector \f$y\f$ of length \f$n\f$ = `a.cols()`; one problem, not a batch
/// @param[in,out] a       dense \f$m \times n\f$ matrix \f$A\f$; overwritten with the updated matrix
/// @param[in] alpha       scalar \f$\alpha\f$ scaling the outer product (default 1)
template <bool ConjugateXV = false, bool ConjugateYV = false, typename Group, typename T>
inline constexpr void ger(const Group& group,
                          const VectorView<T>& x,
                          const VectorView<T>& y,
                          const KernelMatrixView<T, MatrixFormat::Dense>& a,
                          T alpha = T(1)) {
    ger<ConjugateXV, ConjugateYV>(group, x, make_rank1_update_operand(y, a, alpha));
}

/// @brief As ger(), with the operands passed separately and an explicit DeviceBlasPolicy (ignored: one implementation).
/// @tparam Policy       DeviceBlasPolicy; accepted for API symmetry and ignored (one implementation)
/// @tparam ConjugateXV  use \f$\bar{x}\f$
/// @tparam ConjugateYV  use \f$\bar{y}\f$
/// @param[in] group       executor whose work-items share the elements of \f$A\f$; every work-item must call
/// @param[in] x           left vector \f$x\f$ of length \f$m\f$ = `a.rows()`; one problem, not a batch
/// @param[in] y           right vector \f$y\f$ of length \f$n\f$ = `a.cols()`; one problem, not a batch
/// @param[in,out] a       dense \f$m \times n\f$ matrix \f$A\f$; overwritten with the updated matrix
/// @param[in] alpha       scalar \f$\alpha\f$ scaling the outer product (default 1)
template <DeviceBlasPolicy Policy,
          bool ConjugateXV = false,
          bool ConjugateYV = false,
          typename Group,
          typename T>
inline constexpr void ger(const Group& group,
                          const VectorView<T>& x,
                          const VectorView<T>& y,
                          const KernelMatrixView<T, MatrixFormat::Dense>& a,
                          T alpha = T(1)) {
    (void)Policy;
    ger<ConjugateXV, ConjugateYV>(group, x, y, a, alpha);
}

/// @brief Unconjugated rank-1 update \f$ A := A + \alpha\,x\,y^{T} \f$; ger() without conjugation.
/// @param[in] group       executor whose work-items share the elements of \f$A\f$; every work-item must call
/// @param[in] x           left vector \f$x\f$ of length \f$m\f$ = `operand.a.rows()`; one problem, not a batch
/// @param[in,out] operand  \f$y\f$ (read, length \f$n\f$ = `operand.a.cols()`), the \f$m \times n\f$ matrix \f$A\f$ (updated in place) and \f$\alpha\f$ (read)
template <typename Group, typename T>
inline constexpr void geru(const Group& group,
                           const VectorView<T>& x,
                           Rank1UpdateOperand<T> operand) {
    ger<false, false>(group, x, operand);
}

/// @brief As geru(), with an explicit DeviceBlasPolicy (ignored: one implementation).
/// @tparam Policy       DeviceBlasPolicy; accepted for API symmetry and ignored (one implementation)
/// @param[in] group       executor whose work-items share the elements of \f$A\f$; every work-item must call
/// @param[in] x           left vector \f$x\f$ of length \f$m\f$ = `operand.a.rows()`; one problem, not a batch
/// @param[in,out] operand  \f$y\f$ (read, length \f$n\f$ = `operand.a.cols()`), the \f$m \times n\f$ matrix \f$A\f$ (updated in place) and \f$\alpha\f$ (read)
template <DeviceBlasPolicy Policy, typename Group, typename T>
inline constexpr void geru(const Group& group,
                           const VectorView<T>& x,
                           Rank1UpdateOperand<T> operand) {
    (void)Policy;
    geru(group, x, operand);
}

/// @brief As geru(), with the operands passed separately instead of in an operand struct.
/// @param[in] group       executor whose work-items share the elements of \f$A\f$; every work-item must call
/// @param[in] x           left vector \f$x\f$ of length \f$m\f$ = `a.rows()`; one problem, not a batch
/// @param[in] y           right vector \f$y\f$ of length \f$n\f$ = `a.cols()`; one problem, not a batch
/// @param[in,out] a       dense \f$m \times n\f$ matrix \f$A\f$; overwritten with the updated matrix
/// @param[in] alpha       scalar \f$\alpha\f$ scaling the outer product (default 1)
template <typename Group, typename T>
inline constexpr void geru(const Group& group,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           T alpha = T(1)) {
    ger<false, false>(group, x, y, a, alpha);
}

/// @brief As geru(), with the operands passed separately and an explicit DeviceBlasPolicy (ignored: one implementation).
/// @tparam Policy       DeviceBlasPolicy; accepted for API symmetry and ignored (one implementation)
/// @param[in] group       executor whose work-items share the elements of \f$A\f$; every work-item must call
/// @param[in] x           left vector \f$x\f$ of length \f$m\f$ = `a.rows()`; one problem, not a batch
/// @param[in] y           right vector \f$y\f$ of length \f$n\f$ = `a.cols()`; one problem, not a batch
/// @param[in,out] a       dense \f$m \times n\f$ matrix \f$A\f$; overwritten with the updated matrix
/// @param[in] alpha       scalar \f$\alpha\f$ scaling the outer product (default 1)
template <DeviceBlasPolicy Policy, typename Group, typename T>
inline constexpr void geru(const Group& group,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           T alpha = T(1)) {
    (void)Policy;
    geru(group, x, y, a, alpha);
}

/// @brief Conjugated rank-1 update \f$ A := A + \alpha\,x\,y^{H} \f$; ger() with conjugated `y`.
/// @param[in] group       executor whose work-items share the elements of \f$A\f$; every work-item must call
/// @param[in] x           left vector \f$x\f$ of length \f$m\f$ = `operand.a.rows()`; one problem, not a batch
/// @param[in,out] operand  \f$y\f$ (read, length \f$n\f$ = `operand.a.cols()`), the \f$m \times n\f$ matrix \f$A\f$ (updated in place) and \f$\alpha\f$ (read)
template <typename Group, typename T>
inline constexpr void gerc(const Group& group,
                           const VectorView<T>& x,
                           Rank1UpdateOperand<T> operand) {
    ger<false, true>(group, x, operand);
}

/// @brief As gerc(), with an explicit DeviceBlasPolicy (ignored: one implementation).
/// @tparam Policy       DeviceBlasPolicy; accepted for API symmetry and ignored (one implementation)
/// @param[in] group       executor whose work-items share the elements of \f$A\f$; every work-item must call
/// @param[in] x           left vector \f$x\f$ of length \f$m\f$ = `operand.a.rows()`; one problem, not a batch
/// @param[in,out] operand  \f$y\f$ (read, length \f$n\f$ = `operand.a.cols()`), the \f$m \times n\f$ matrix \f$A\f$ (updated in place) and \f$\alpha\f$ (read)
template <DeviceBlasPolicy Policy, typename Group, typename T>
inline constexpr void gerc(const Group& group,
                           const VectorView<T>& x,
                           Rank1UpdateOperand<T> operand) {
    (void)Policy;
    gerc(group, x, operand);
}

/// @brief As gerc(), with the operands passed separately instead of in an operand struct.
/// @param[in] group       executor whose work-items share the elements of \f$A\f$; every work-item must call
/// @param[in] x           left vector \f$x\f$ of length \f$m\f$ = `a.rows()`; one problem, not a batch
/// @param[in] y           right vector \f$y\f$ of length \f$n\f$ = `a.cols()`; one problem, not a batch
/// @param[in,out] a       dense \f$m \times n\f$ matrix \f$A\f$; overwritten with the updated matrix
/// @param[in] alpha       scalar \f$\alpha\f$ scaling the outer product (default 1)
template <typename Group, typename T>
inline constexpr void gerc(const Group& group,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           T alpha = T(1)) {
    ger<false, true>(group, x, y, a, alpha);
}

/// @brief As gerc(), with the operands passed separately and an explicit DeviceBlasPolicy (ignored: one implementation).
/// @tparam Policy       DeviceBlasPolicy; accepted for API symmetry and ignored (one implementation)
/// @param[in] group       executor whose work-items share the elements of \f$A\f$; every work-item must call
/// @param[in] x           left vector \f$x\f$ of length \f$m\f$ = `a.rows()`; one problem, not a batch
/// @param[in] y           right vector \f$y\f$ of length \f$n\f$ = `a.cols()`; one problem, not a batch
/// @param[in,out] a       dense \f$m \times n\f$ matrix \f$A\f$; overwritten with the updated matrix
/// @param[in] alpha       scalar \f$\alpha\f$ scaling the outer product (default 1)
template <DeviceBlasPolicy Policy, typename Group, typename T>
inline constexpr void gerc(const Group& group,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           T alpha = T(1)) {
    (void)Policy;
    gerc(group, x, y, a, alpha);
}

/// @}

} // namespace batchlas::device
