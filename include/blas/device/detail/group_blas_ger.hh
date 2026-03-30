#pragma once

#include <blas/device/detail/group_blas_common.hh>

namespace batchlas::device {

namespace detail::generic {

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

template <bool ConjugateXV = false, bool ConjugateYV = false, typename Group, typename T>
inline constexpr void ger(const Group& group,
                          const VectorView<T>& x,
                          Rank1UpdateOperand<T> operand) {
    detail::dispatch_ger<OuterProductTransformTag<ConjugateXV, ConjugateYV>>(group, x, operand);
}

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

template <bool ConjugateXV = false, bool ConjugateYV = false, typename Group, typename T>
inline constexpr void ger(const Group& group,
                          const VectorView<T>& x,
                          const VectorView<T>& y,
                          const KernelMatrixView<T, MatrixFormat::Dense>& a,
                          T alpha = T(1)) {
    ger<ConjugateXV, ConjugateYV>(group, x, make_rank1_update_operand(y, a, alpha));
}

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

template <typename Group, typename T>
inline constexpr void geru(const Group& group,
                           const VectorView<T>& x,
                           Rank1UpdateOperand<T> operand) {
    ger<false, false>(group, x, operand);
}

template <DeviceBlasPolicy Policy, typename Group, typename T>
inline constexpr void geru(const Group& group,
                           const VectorView<T>& x,
                           Rank1UpdateOperand<T> operand) {
    (void)Policy;
    geru(group, x, operand);
}

template <typename Group, typename T>
inline constexpr void geru(const Group& group,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           T alpha = T(1)) {
    ger<false, false>(group, x, y, a, alpha);
}

template <DeviceBlasPolicy Policy, typename Group, typename T>
inline constexpr void geru(const Group& group,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           T alpha = T(1)) {
    (void)Policy;
    geru(group, x, y, a, alpha);
}

template <typename Group, typename T>
inline constexpr void gerc(const Group& group,
                           const VectorView<T>& x,
                           Rank1UpdateOperand<T> operand) {
    ger<false, true>(group, x, operand);
}

template <DeviceBlasPolicy Policy, typename Group, typename T>
inline constexpr void gerc(const Group& group,
                           const VectorView<T>& x,
                           Rank1UpdateOperand<T> operand) {
    (void)Policy;
    gerc(group, x, operand);
}

template <typename Group, typename T>
inline constexpr void gerc(const Group& group,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           T alpha = T(1)) {
    ger<false, true>(group, x, y, a, alpha);
}

template <DeviceBlasPolicy Policy, typename Group, typename T>
inline constexpr void gerc(const Group& group,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           T alpha = T(1)) {
    (void)Policy;
    gerc(group, x, y, a, alpha);
}

} // namespace batchlas::device
