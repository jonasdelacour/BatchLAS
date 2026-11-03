// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/blas/device/detail/group_blas_common.hh>

namespace batchlas::device {

namespace detail::generic {

/// @brief Generic triangular matrix-vector product \f$ y := \alpha\,\mathrm{op}(A)\,x + \beta\,y \f$; no operand checks.
/// @param[in]     group    executor whose work-items share each row reduction; every work-item must call
/// @param[in]     a        single \f$n \times n\f$ triangular matrix \f$A\f$; only the `Tag::uplo` triangle is read (not its diagonal when `Tag::diag` is `Unit`)
/// @param[in,out] operand  \f$x\f$ (read) and \f$y\f$ (read scaled by \f$\beta\f$, overwritten with the
///                         result by the group leader), and the scalars \f$\alpha\f$, \f$\beta\f$
template <typename Tag, typename Group, typename T>
inline constexpr void trmv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixVectorOperand<T> operand) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int extent = a.rows();
    constexpr bool effective_lower = Tag::trans == Transpose::NoTrans ? Tag::uplo == Uplo::Lower : Tag::uplo == Uplo::Upper;

    for (int output_step = 0; output_step < extent; ++output_step) {
        const int output_index = effective_lower ? (extent - 1 - output_step) : output_step;
        T partial{};
        for (int input_index = local_id; input_index < extent; input_index += local_size) {
            partial += detail::triangular_matrix_entry<Tag>(a, output_index, input_index) * operand.x(input_index);
        }
        partial = detail::reduce_sum_group(group, partial);
        if (detail::group_is_leader(group)) {
            operand.y(output_index) = operand.alpha * partial + operand.beta * operand.y(output_index);
        }
    }
}

} // namespace detail::generic

namespace detail {

/// @brief Validates the operands of a triangular matrix-vector product, then runs generic::trmv().
/// @param[in]     group    executor whose work-items share each row reduction; every work-item must call
/// @param[in]     a        single \f$n \times n\f$ triangular matrix \f$A\f$; only the `Tag::uplo` triangle is read (not its diagonal when `Tag::diag` is `Unit`)
/// @param[in,out] operand  \f$x\f$ (read) and \f$y\f$ (read scaled by \f$\beta\f$, overwritten with the
///                         result), and the scalars \f$\alpha\f$, \f$\beta\f$
template <typename Tag, typename Group, typename T>
inline constexpr void dispatch_trmv(const Group& group,
                                    const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                    MatrixVectorOperand<T> operand) {
    validate_triangular_operand(a,
                                operand,
                                TriangularTransform{.side = Tag::side, .uplo = Tag::uplo, .trans = Tag::trans, .diag = Tag::diag});
    generic::trmv<Tag>(group, a, operand);
}

} // namespace detail

/// @addtogroup api_device_blas
/// @{

/// @brief Triangular matrix-vector product \f$ y := \alpha\,\mathrm{op}(A)\,x + \beta\,y \f$, `A` triangular.
///
/// Out of place, unlike BLAS `trmv`: `A` is read only on its `UploV` triangle
/// (and not on the diagonal when `DiagV` is `Unit`). `x` and `y` may be the same
/// vector: rows are visited in the order that never overwrites an entry a later
/// row still reads.
/// @tparam UploV   stored triangle of `A`
/// @tparam TransV  \f$\mathrm{op}()\f$ applied to `A`
/// @tparam DiagV   `Unit`: the diagonal is taken as 1
/// @tparam Group   `sycl::group`, `sycl::sub_group` or an `nd_item`; every work-item must call
/// @param[in]     group    executor whose work-items share the rows of \f$A\f$; every work-item must call
/// @param[in]     a        single \f$n \times n\f$ triangular matrix \f$A\f$; only the `UploV` triangle is read (not its diagonal when `DiagV` is `Unit`)
/// @param[in,out] operand  \f$x\f$ (read) and \f$y\f$ (read scaled by \f$\beta\f$, overwritten with the
///                         result), each of length at least \f$n\f$, and the scalars \f$\alpha\f$, \f$\beta\f$
/// @pre `y` holds finite values even when `beta == 0`.
/// @note Each `y` element is written by the group leader; barrier before other work-items read it.
template <Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          Diag DiagV = Diag::NonUnit,
          typename Group,
          typename T>
inline constexpr void trmv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixVectorOperand<T> operand) {
    detail::dispatch_trmv<detail::TriangularTransformTag<Side::Left, UploV, TransV, DiagV>>(group, a, operand);
}

/// @brief As trmv(), with an explicit DeviceBlasPolicy (ignored: one implementation).
/// @param[in]     group    executor whose work-items share the rows of \f$A\f$; every work-item must call
/// @param[in]     a        single \f$n \times n\f$ triangular matrix \f$A\f$; only the `UploV` triangle is read (not its diagonal when `DiagV` is `Unit`)
/// @param[in,out] operand  \f$x\f$ (read) and \f$y\f$ (read scaled by \f$\beta\f$, overwritten with the
///                         result), each of length at least \f$n\f$, and the scalars \f$\alpha\f$, \f$\beta\f$
template <DeviceBlasPolicy Policy,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          Diag DiagV = Diag::NonUnit,
          typename Group,
          typename T>
inline constexpr void trmv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixVectorOperand<T> operand) {
    (void)Policy;
    trmv<UploV, TransV, DiagV>(group, a, operand);
}

/// @brief As trmv(), with the operands passed separately instead of in an operand struct.
/// @param[in]     group    executor whose work-items share the rows of \f$A\f$; every work-item must call
/// @param[in]     a        single \f$n \times n\f$ triangular matrix \f$A\f$; only the `UploV` triangle is read (not its diagonal when `DiagV` is `Unit`)
/// @param[in]     x        vector \f$x\f$ of length at least \f$n\f$; may alias `y`
/// @param[in,out] y        vector \f$y\f$ of length at least \f$n\f$; read scaled by \f$\beta\f$, overwritten
///                         with the result
/// @param[in]     alpha    scalar \f$\alpha\f$ multiplying \f$\mathrm{op}(A)\,x\f$ (default 1)
/// @param[in]     beta     scalar \f$\beta\f$ multiplying the input \f$y\f$ (default 0)
template <Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          Diag DiagV = Diag::NonUnit,
          typename Group,
          typename T>
inline constexpr void trmv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0)) {
    trmv<UploV, TransV, DiagV>(group, a, make_matvec_operand(x, y, alpha, beta));
}

/// @brief As trmv(), with an explicit DeviceBlasPolicy and the operands passed separately (ignored: one implementation).
/// @param[in]     group    executor whose work-items share the rows of \f$A\f$; every work-item must call
/// @param[in]     a        single \f$n \times n\f$ triangular matrix \f$A\f$; only the `UploV` triangle is read (not its diagonal when `DiagV` is `Unit`)
/// @param[in]     x        vector \f$x\f$ of length at least \f$n\f$; may alias `y`
/// @param[in,out] y        vector \f$y\f$ of length at least \f$n\f$; read scaled by \f$\beta\f$, overwritten
///                         with the result
/// @param[in]     alpha    scalar \f$\alpha\f$ multiplying \f$\mathrm{op}(A)\,x\f$ (default 1)
/// @param[in]     beta     scalar \f$\beta\f$ multiplying the input \f$y\f$ (default 0)
template <DeviceBlasPolicy Policy,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          Diag DiagV = Diag::NonUnit,
          typename Group,
          typename T>
inline constexpr void trmv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0)) {
    (void)Policy;
    trmv<UploV, TransV, DiagV>(group, a, x, y, alpha, beta);
}

/// @}

} // namespace batchlas::device
