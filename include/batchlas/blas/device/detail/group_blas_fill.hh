#pragma once

#include <batchlas/blas/device/detail/group_blas_common.hh>

namespace batchlas::device {

template <typename Group, typename T>
inline constexpr void fill(const Group& group,
                           const VectorView<T>& x,
                           const T& value) {
    detail::validate_vector_operand(x, "fill");
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);

    for (int index = local_id; index < x.size(); index += local_size) {
        x(index) = value;
    }
}

template <typename Group, typename T>
inline constexpr void fill(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const T& value) {
    detail::validate_single_problem(a, Transpose::NoTrans);
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int extent = a.rows() * a.cols();

    for (int linear_index = local_id; linear_index < extent; linear_index += local_size) {
        const int row = linear_index % a.rows();
        const int col = linear_index / a.rows();
        a(row, col) = value;
    }
}

} // namespace batchlas::device