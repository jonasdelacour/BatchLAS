#pragma once

#include <blas/device/detail/group_blas_common.hh>

namespace batchlas::device {

namespace detail::generic {

template <typename Tag, typename Group, typename T>
inline constexpr void gemv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixVectorOperand<T> operand) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int inner_extent = detail::input_size<Tag::trans>(a);
    const int outer_extent = detail::output_size<Tag::trans>(a);

    if (inner_extent < local_size) {
        for (int output_index = local_id; output_index < outer_extent; output_index += local_size) {
            T partial{};
            for (int input_index = 0; input_index < inner_extent; ++input_index) {
                partial += detail::matrix_entry<Tag::trans>(a, output_index, input_index) * operand.x(input_index);
            }
            operand.y(output_index) = operand.alpha * partial + operand.beta * operand.y(output_index);
        }
        return;
    }

    for (int output_index = 0; output_index < outer_extent; ++output_index) {
        T partial{};
        for (int input_index = local_id; input_index < inner_extent; input_index += local_size) {
            partial += detail::matrix_entry<Tag::trans>(a, output_index, input_index) * operand.x(input_index);
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
inline constexpr void dispatch_gemv(const Group& group,
                                    const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                    MatrixVectorOperand<T> operand) {
    validate_operand(a, operand, Tag::trans);
    generic::gemv<Tag>(group, a, operand);
}

} // namespace detail

template <Transpose TransV = Transpose::NoTrans, typename Group, typename T>
inline constexpr void gemv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixVectorOperand<T> operand) {
    detail::dispatch_gemv<MatrixVectorTransformTag<TransV>>(group, a, operand);
}

template <DeviceBlasPolicy Policy,
          Transpose TransV = Transpose::NoTrans,
          typename Group,
          typename T>
inline constexpr void gemv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixVectorOperand<T> operand) {
    (void)Policy;
    gemv<TransV>(group, a, operand);
}

template <Transpose TransV = Transpose::NoTrans, typename Group, typename T>
inline constexpr void gemv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0)) {
    gemv<TransV>(group, a, make_matvec_operand(x, y, alpha, beta));
}

template <DeviceBlasPolicy Policy,
          Transpose TransV = Transpose::NoTrans,
          typename Group,
          typename T>
inline constexpr void gemv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0)) {
    (void)Policy;
    gemv<TransV>(group, a, x, y, alpha, beta);
}

} // namespace batchlas::device
