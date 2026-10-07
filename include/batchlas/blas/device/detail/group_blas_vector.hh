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

/// @addtogroup device
/// @{

/// @brief \f$ y := x \f$, cooperatively across `group`.
/// @tparam Group  `sycl::group`, `sycl::sub_group` or an `nd_item`; every work-item must call
/// @param group  executor whose work-items share the elements
/// @param x      input vector
/// @param y      output vector
/// @pre `x.size() == y.size()`; both single vectors (`batch_size() == 1`). Checked by `assert` only.
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

/// @brief \f$ y := \bar{x} \f$ (a plain copy for real `T`).
/// @param group  executor whose work-items share the elements
/// @param x      input vector
/// @param y      output vector
/// @pre `x.size() == y.size()`.
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

/// @brief \f$ x := \alpha\,x \f$.
/// @param group  executor whose work-items share the elements
/// @param x      vector scaled in place
/// @param alpha  scale
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

/// @brief \f$ y := \alpha\,x + y \f$.
/// @param group  executor whose work-items share the elements
/// @param x      input vector
/// @param y      vector updated in place
/// @param alpha  scale of `x`
/// @pre `x.size() == y.size()`.
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

/// @brief Element-wise product \f$ z_i := x_i\,y_i \f$.
/// @param group  executor whose work-items share the elements
/// @param x      first factor
/// @param y      second factor
/// @param z      output; may alias `x` or `y`
/// @pre all three vectors have the same size.
template <typename Group, typename T>
inline constexpr void hadamard(const Group& group,
                               const VectorView<T>& x,
                               const VectorView<T>& y,
                               const VectorView<T>& z) {
    hadamard(group, z, [](const T& lhs, const T& rhs) { return lhs * rhs; }, x, y);
}

/// @brief Element-wise n-ary map \f$ z_i := \mathrm{op}(u_i, v_i, \ldots) \f$.
/// @tparam Op      callable taking one `T` per input and returning `T`; must be device-callable
/// @tparam Inputs  one or more `VectorView<T>`
/// @param group   executor whose work-items share the elements
/// @param z       output vector
/// @param op      the element-wise function
/// @param inputs  input vectors, each of `z.size()`
template <typename Group, typename T, typename Op, typename... Inputs>
    requires(sizeof...(Inputs) > 0 && (detail::VectorOperandFor<T, Inputs> && ...))
inline constexpr void hadamard(const Group& group,
                               const VectorView<T>& z,
                               Op op,
                               const Inputs&... inputs) {
    detail::validate_hadamard_operands(z, "hadamard", inputs...);
    detail::generic::hadamard(group, z, op, inputs...);
}

/// @brief Unconjugated dot product \f$ \sum_i x_i\,y_i \f$.
/// @param group  executor whose work-items share the elements
/// @param x      first vector
/// @param y      second vector
/// @return the group-wide sum, on every work-item of `group`
/// @pre `x.size() == y.size()`.
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

/// @brief Conjugated dot product \f$ \sum_i \bar{x}_i\,y_i \f$ (dotu for real `T`).
/// @param group  executor whose work-items share the elements
/// @param x      conjugated vector
/// @param y      second vector
/// @return the group-wide sum, on every work-item of `group`
/// @pre `x.size() == y.size()`.
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

/// @}

} // namespace batchlas::device