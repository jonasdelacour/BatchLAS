#pragma once

#include <batchlas/blas/device/detail/group_blas_common.hh>

namespace batchlas::device {

namespace detail::generic {

template <typename Group, typename T, typename Op, typename... Inputs>
inline constexpr void hadamard(const Group& group,
                               const VectorView<T>& z,
                               Op op,
                               const Inputs&... inputs) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);

    for (int index = local_id; index < z.size(); index += local_size) {
        z(index) = op(inputs(index)...);
    }
}

} // namespace detail::generic

template <typename Group, typename T>
inline constexpr void copy(const Group& group,
                           const VectorView<T>& x,
                           const VectorView<T>& y) {
    detail::validate_vector_operands(x, y, "copy");
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);

    for (int index = local_id; index < x.size(); index += local_size) {
        y(index) = x(index);
    }
}

template <typename Group, typename T>
inline constexpr void copyc(const Group& group,
                            const VectorView<T>& x,
                            const VectorView<T>& y) {
    detail::validate_vector_operands(x, y, "copyc");
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);

    for (int index = local_id; index < x.size(); index += local_size) {
        y(index) = detail::conj(x(index));
    }
}

template <typename Group, typename T>
inline constexpr void scal(const Group& group,
                           const VectorView<T>& x,
                           T alpha) {
    detail::validate_vector_operand(x, "scal");
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);

    for (int index = local_id; index < x.size(); index += local_size) {
        x(index) *= alpha;
    }
}

template <typename Group, typename T>
inline constexpr void axpy(const Group& group,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1)) {
    detail::validate_vector_operands(x, y, "axpy");
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);

    for (int index = local_id; index < x.size(); index += local_size) {
        y(index) += alpha * x(index);
    }
}

template <typename Group, typename T>
inline constexpr void hadamard(const Group& group,
                               const VectorView<T>& x,
                               const VectorView<T>& y,
                               const VectorView<T>& z) {
    hadamard(group, z, [](const T& lhs, const T& rhs) { return lhs * rhs; }, x, y);
}

template <typename Group, typename T, typename Op, typename... Inputs>
    requires(sizeof...(Inputs) > 0 && (detail::VectorOperandFor<T, Inputs> && ...))
inline constexpr void hadamard(const Group& group,
                               const VectorView<T>& z,
                               Op op,
                               const Inputs&... inputs) {
    detail::validate_hadamard_operands(z, "hadamard", inputs...);
    detail::generic::hadamard(group, z, op, inputs...);
}

template <typename Group, typename T>
inline constexpr T dotu(const Group& group,
                        const VectorView<T>& x,
                        const VectorView<T>& y) {
    detail::validate_vector_operands(x, y, "dotu");
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    T partial = T(0);

    for (int index = local_id; index < x.size(); index += local_size) {
        partial += x(index) * y(index);
    }

    return detail::reduce_sum_group(group, partial);
}

template <typename Group, typename T>
inline constexpr T dotc(const Group& group,
                        const VectorView<T>& x,
                        const VectorView<T>& y) {
    detail::validate_vector_operands(x, y, "dotc");
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    T partial = T(0);

    for (int index = local_id; index < x.size(); index += local_size) {
        partial += detail::conj(x(index)) * y(index);
    }

    return detail::reduce_sum_group(group, partial);
}

} // namespace batchlas::device