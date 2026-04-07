#pragma once

#include <blas/device/detail/group_blas_common.hh>

namespace batchlas::device {

namespace detail::generic {

template <typename Group>
inline constexpr auto gemv_work_group(const Group& group) {
    if constexpr (detail::NdItemLike<Group>) {
        return group.get_group();
    } else {
        return group;
    }
}

template <typename T>
struct GemvTransposeWorkspace {
    static constexpr int kMaxTile = 32;
    T matrix[kMaxTile * (kMaxTile + 1)];
    T x[kMaxTile];
};

template <typename T>
inline constexpr int gemv_transpose_tile_index(int row, int col) {
    return row * (GemvTransposeWorkspace<T>::kMaxTile + 1) + col;
}

template <typename Group>
inline constexpr int gemv_tiled_tile_size(const Group& group, DeviceBlasPolicy policy) {
    const int local_size = detail::group_local_linear_range(group);
    if (policy == DeviceBlasPolicy::Generic) {
        return 0;
    }
    if (policy == DeviceBlasPolicy::Subgroup32) {
        return local_size >= 32 ? 32 : 0;
    }
    if (policy == DeviceBlasPolicy::Subgroup16) {
        return local_size >= 16 ? 16 : 0;
    }
    if (local_size >= 32) {
        return 32;
    }
    if (local_size >= 16) {
        return 16;
    }
    return 0;
}

template <typename Tag, typename Group, typename T>
inline constexpr bool can_use_tiled_gemv(const Group& group,
                                         const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                         DeviceBlasPolicy policy) {
    if constexpr (Tag::trans != Transpose::NoTrans) {
        (void)group;
        (void)a;
        (void)policy;
        return false;
    } else {
        const int tile = gemv_tiled_tile_size(group, policy);
        return tile > 0 && a.rows() >= tile && a.cols() >= tile;
    }
}

template <typename Tag, typename Group>
inline constexpr bool can_use_tiled_gemv(const Group& group,
                                         int rows,
                                         int cols,
                                         DeviceBlasPolicy policy) {
    if constexpr (Tag::trans != Transpose::NoTrans) {
        (void)group;
        (void)rows;
        (void)cols;
        (void)policy;
        return false;
    } else {
        const int tile = gemv_tiled_tile_size(group, policy);
        return tile > 0 && rows >= tile && cols >= tile;
    }
}

template <typename Group, typename T>
inline void gemv_tiled(const Group& group,
                       const KernelMatrixView<T, MatrixFormat::Dense>& a,
                       MatrixVectorOperand<T> operand,
                       int tile,
                       T* workspace_ptr) {
    auto work_group = gemv_work_group(group);
    auto* workspace = detail::workspace_ptr_cast<GemvTransposeWorkspace<T>>(workspace_ptr);
    T* matrix_tile = workspace->matrix;
    T* x_tile = workspace->x;
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int row_extent = a.rows();
    const int col_extent = a.cols();

    for (int row_base = 0; row_base < row_extent; row_base += tile) {
        const int row_tile_extent = std::min(tile, row_extent - row_base);

        for (int row = local_id; row < row_tile_extent; row += local_size) {
            operand.y(row_base + row) *= operand.beta;
        }

        for (int col_base = 0; col_base < col_extent; col_base += tile) {
            const int col_tile_extent = std::min(tile, col_extent - col_base);

            for (int col = local_id; col < col_tile_extent; col += local_size) {
                x_tile[col] = operand.x(col_base + col);
            }
            for (int linear_index = local_id; linear_index < row_tile_extent * col_tile_extent; linear_index += local_size) {
                const int row = linear_index % row_tile_extent;
                const int col = linear_index / row_tile_extent;
                matrix_tile[gemv_transpose_tile_index<T>(row, col)] = a(row_base + row, col_base + col);
            }
            sycl::group_barrier(work_group);

            for (int row = local_id; row < row_tile_extent; row += local_size) {
                T partial{};
                for (int col = 0; col < col_tile_extent; ++col) {
                    partial += matrix_tile[gemv_transpose_tile_index<T>(row, col)] * x_tile[col];
                }
                operand.y(row_base + row) += operand.alpha * partial;
            }
            sycl::group_barrier(work_group);
        }
    }
}

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

template <typename Tag, DeviceBlasPolicy Policy, typename T>
inline constexpr std::size_t gemv_workspace_elements(const DeviceBlasLaunchInfo& launch,
                                                     int rows,
                                                     int cols) {
    assert(rows >= 0 && cols >= 0 && "device::gemv workspace query expects non-negative matrix extents");
    if (generic::can_use_tiled_gemv<Tag>(launch, rows, cols, Policy)) {
        return detail::workspace_elements_v<T, generic::GemvTransposeWorkspace<T>>;
    }
    return 0;
}

template <typename Tag, DeviceBlasPolicy Policy, typename Group, typename T>
inline void dispatch_gemv(const Group& group,
                          const KernelMatrixView<T, MatrixFormat::Dense>& a,
                          MatrixVectorOperand<T> operand,
                          T* workspace = nullptr) {
    validate_operand(a, operand, Tag::trans);
    if (workspace != nullptr && generic::can_use_tiled_gemv<Tag>(group, a, Policy)) {
        generic::gemv_tiled(group, a, operand, generic::gemv_tiled_tile_size(group, Policy), workspace);
        return;
    }
    generic::gemv<Tag>(group, a, operand);
}

} // namespace detail

template <Transpose TransV = Transpose::NoTrans, typename Group, typename T>
inline constexpr void gemv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixVectorOperand<T> operand,
                           T* workspace = nullptr) {
    detail::dispatch_gemv<MatrixVectorTransformTag<TransV>, DeviceBlasPolicy::Auto>(group, a, operand, workspace);
}

template <DeviceBlasPolicy Policy,
          Transpose TransV = Transpose::NoTrans,
          typename Group,
          typename T>
inline constexpr void gemv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixVectorOperand<T> operand,
                           T* workspace = nullptr) {
    detail::dispatch_gemv<MatrixVectorTransformTag<TransV>, Policy>(group, a, operand, workspace);
}

template <typename T, Transpose TransV = Transpose::NoTrans>
inline constexpr std::size_t gemv_workspace_elements(const DeviceBlasLaunchInfo& launch,
                                                     int rows,
                                                     int cols) {
    return detail::gemv_workspace_elements<MatrixVectorTransformTag<TransV>, DeviceBlasPolicy::Auto, T>(launch, rows, cols);
}

template <typename T, DeviceBlasPolicy Policy, Transpose TransV = Transpose::NoTrans>
inline constexpr std::size_t gemv_workspace_elements(const DeviceBlasLaunchInfo& launch,
                                                     int rows,
                                                     int cols) {
    return detail::gemv_workspace_elements<MatrixVectorTransformTag<TransV>, Policy, T>(launch, rows, cols);
}

template <Transpose TransV = Transpose::NoTrans, typename Group, typename T>
inline constexpr void gemv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0),
                           T* workspace = nullptr) {
    gemv<TransV>(group, a, make_matvec_operand(x, y, alpha, beta), workspace);
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
                           T beta = T(0),
                           T* workspace = nullptr) {
    gemv<Policy, TransV>(group, a, make_matvec_operand(x, y, alpha, beta), workspace);
}

} // namespace batchlas::device
