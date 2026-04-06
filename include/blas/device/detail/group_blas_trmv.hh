#pragma once

#include <blas/device/detail/group_blas_common.hh>

namespace batchlas::device {

namespace detail::generic {

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

} // namespace batchlas::device
