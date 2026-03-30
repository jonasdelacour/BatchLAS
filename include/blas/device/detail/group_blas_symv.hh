#pragma once

#include <blas/device/detail/group_blas_common.hh>
#include <sycl/ext/oneapi/group_local_memory.hpp>

namespace batchlas::device {

namespace detail::generic {

template <typename Tag, typename T>
inline constexpr T symmetric_mirror(const T& value) {
    if constexpr (Tag::hermitian) {
        return detail::conj(value);
    }
    return value;
}

template <typename T>
struct SymvTransposeWorkspace {
    static constexpr int kTile = 16;
    T matrix[kTile * (kTile + 1)];
    T x_col[kTile];
    T x_row[kTile];
};

template <typename T>
inline constexpr int symv_transpose_tile_index(int row, int col) {
    return row * (SymvTransposeWorkspace<T>::kTile + 1) + col;
}

template <typename Group>
inline constexpr bool symv_tiled_policy_matches(const Group& group, DeviceBlasPolicy policy) {
    const int local_size = detail::group_local_linear_range(group);
    if (policy == DeviceBlasPolicy::Generic) {
        return false;
    }
    if (policy == DeviceBlasPolicy::Subgroup16) {
        return local_size >= 16;
    }
    if (policy == DeviceBlasPolicy::Subgroup32) {
        return local_size >= 32;
    }
    return local_size >= 16;
}

template <typename Tag, typename Group, typename T>
inline constexpr bool can_use_tiled_symv(const Group& group,
                                         const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                         DeviceBlasPolicy policy) {
    return symv_tiled_policy_matches(group, policy) &&
        a.rows() >= SymvTransposeWorkspace<T>::kTile &&
        a.cols() >= SymvTransposeWorkspace<T>::kTile;
}

template <typename Tag, typename Group, typename T>
inline void symv_tiled(const Group& group,
                       const KernelMatrixView<T, MatrixFormat::Dense>& a,
                       MatrixVectorOperand<T> operand) {
    constexpr int Tile = SymvTransposeWorkspace<T>::kTile;
    auto* workspace = sycl::ext::oneapi::group_local_memory_for_overwrite<SymvTransposeWorkspace<T>>(group).get();
    T* matrix_tile = workspace->matrix;
    T* x_col_tile = workspace->x_col;
    T* x_row_tile = workspace->x_row;
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int extent = a.rows();

    for (int output_index = local_id; output_index < extent; output_index += local_size) {
        operand.y(output_index) *= operand.beta;
    }
    sycl::group_barrier(group);

    for (int col_base = 0; col_base < extent; col_base += Tile) {
        const int col_extent = std::min(Tile, extent - col_base);

        for (int col = local_id; col < col_extent; col += local_size) {
            x_col_tile[col] = operand.x(col_base + col);
        }
        sycl::group_barrier(group);

        for (int linear_index = local_id; linear_index < col_extent * col_extent; linear_index += local_size) {
            const int row = linear_index % col_extent;
            const int col = linear_index / col_extent;
            T value{};
            if constexpr (Tag::uplo == Uplo::Lower) {
                if (row >= col) {
                    value = a(col_base + row, col_base + col);
                } else {
                    value = symmetric_mirror<Tag>(a(col_base + col, col_base + row));
                }
            } else {
                if (row <= col) {
                    value = a(col_base + row, col_base + col);
                } else {
                    value = symmetric_mirror<Tag>(a(col_base + col, col_base + row));
                }
            }
            matrix_tile[symv_transpose_tile_index<T>(row, col)] = value;
        }
        sycl::group_barrier(group);

        for (int row = local_id; row < col_extent; row += local_size) {
            T sum{};
            for (int col = 0; col < col_extent; ++col) {
                sum += matrix_tile[symv_transpose_tile_index<T>(row, col)] * x_col_tile[col];
            }
            operand.y(col_base + row) += operand.alpha * sum;
        }
        sycl::group_barrier(group);

        if constexpr (Tag::uplo == Uplo::Lower) {
            for (int row_base = col_base + Tile; row_base < extent; row_base += Tile) {
                const int row_extent = std::min(Tile, extent - row_base);
                for (int row = local_id; row < row_extent; row += local_size) {
                    x_row_tile[row] = operand.x(row_base + row);
                }
                for (int linear_index = local_id; linear_index < row_extent * col_extent; linear_index += local_size) {
                    const int row = linear_index % row_extent;
                    const int col = linear_index / row_extent;
                    matrix_tile[symv_transpose_tile_index<T>(row, col)] = a(row_base + row, col_base + col);
                }
                sycl::group_barrier(group);

                for (int row = local_id; row < row_extent; row += local_size) {
                    T sum{};
                    for (int col = 0; col < col_extent; ++col) {
                        sum += matrix_tile[symv_transpose_tile_index<T>(row, col)] * x_col_tile[col];
                    }
                    operand.y(row_base + row) += operand.alpha * sum;
                }
                for (int col = local_id; col < col_extent; col += local_size) {
                    T sum{};
                    for (int row = 0; row < row_extent; ++row) {
                        sum += symmetric_mirror<Tag>(matrix_tile[symv_transpose_tile_index<T>(row, col)]) * x_row_tile[row];
                    }
                    operand.y(col_base + col) += operand.alpha * sum;
                }
                sycl::group_barrier(group);
            }
        } else {
            for (int row_base = 0; row_base < col_base; row_base += Tile) {
                const int row_extent = std::min(Tile, col_base - row_base);
                for (int row = local_id; row < row_extent; row += local_size) {
                    x_row_tile[row] = operand.x(row_base + row);
                }
                for (int linear_index = local_id; linear_index < row_extent * col_extent; linear_index += local_size) {
                    const int row = linear_index % row_extent;
                    const int col = linear_index / row_extent;
                    matrix_tile[symv_transpose_tile_index<T>(row, col)] = a(row_base + row, col_base + col);
                }
                sycl::group_barrier(group);

                for (int row = local_id; row < row_extent; row += local_size) {
                    T sum{};
                    for (int col = 0; col < col_extent; ++col) {
                        sum += matrix_tile[symv_transpose_tile_index<T>(row, col)] * x_col_tile[col];
                    }
                    operand.y(row_base + row) += operand.alpha * sum;
                }
                for (int col = local_id; col < col_extent; col += local_size) {
                    T sum{};
                    for (int row = 0; row < row_extent; ++row) {
                        sum += symmetric_mirror<Tag>(matrix_tile[symv_transpose_tile_index<T>(row, col)]) * x_row_tile[row];
                    }
                    operand.y(col_base + col) += operand.alpha * sum;
                }
                sycl::group_barrier(group);
            }
        }
    }
}

template <typename Tag, typename Group, typename T>
inline constexpr void symv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixVectorOperand<T> operand) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int extent = a.rows();

    for (int output_index = local_id; output_index < extent; output_index += local_size) {
        operand.y(output_index) *= operand.beta;
    }
    sycl::group_barrier(group);

    for (int col = 0; col < extent; ++col) {
        const T x_col = operand.x(col);
        T mirrored_partial{};

        if constexpr (Tag::uplo == Uplo::Lower) {
            for (int row = col + 1 + local_id; row < extent; row += local_size) {
                const T a_row_col = a(row, col);
                operand.y(row) += operand.alpha * a_row_col * x_col;
                mirrored_partial += symmetric_mirror<Tag>(a_row_col) * operand.x(row);
            }
        } else {
            for (int row = local_id; row < col; row += local_size) {
                const T a_row_col = a(row, col);
                operand.y(row) += operand.alpha * a_row_col * x_col;
                mirrored_partial += symmetric_mirror<Tag>(a_row_col) * operand.x(row);
            }
        }

        mirrored_partial = detail::reduce_sum_group(group, mirrored_partial);
        if (detail::group_is_leader(group)) {
            operand.y(col) += operand.alpha * (a(col, col) * x_col + mirrored_partial);
        }
        sycl::group_barrier(group);
    }
}

} // namespace detail::generic

namespace detail {

template <typename Tag, DeviceBlasPolicy Policy, typename Group, typename T>
inline void dispatch_symv(const Group& group,
                          const KernelMatrixView<T, MatrixFormat::Dense>& a,
                          MatrixVectorOperand<T> operand) {
    validate_symmetric_operand(a,
                               operand,
                               SymmetricTransform{.side = Tag::side, .uplo = Tag::uplo, .hermitian = Tag::hermitian});
    if (generic::can_use_tiled_symv<Tag>(group, a, Policy)) {
        generic::symv_tiled<Tag>(group, a, operand);
        return;
    }
    generic::symv<Tag>(group, a, operand);
}

} // namespace detail

template <Uplo UploV = Uplo::Upper, typename Group, typename T>
inline constexpr void symv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixVectorOperand<T> operand) {
    detail::dispatch_symv<detail::SymmetricTransformTag<Side::Left, UploV, false>, DeviceBlasPolicy::Auto>(group, a, operand);
}

template <DeviceBlasPolicy Policy, Uplo UploV = Uplo::Upper, typename Group, typename T>
inline constexpr void symv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixVectorOperand<T> operand) {
    detail::dispatch_symv<detail::SymmetricTransformTag<Side::Left, UploV, false>, Policy>(group, a, operand);
}

template <Uplo UploV = Uplo::Upper, typename Group, typename T>
inline constexpr void symv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0)) {
    symv<UploV>(group, a, make_matvec_operand(x, y, alpha, beta));
}

template <DeviceBlasPolicy Policy, Uplo UploV = Uplo::Upper, typename Group, typename T>
inline constexpr void symv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0)) {
    (void)Policy;
    symv<UploV>(group, a, x, y, alpha, beta);
}

template <Uplo UploV = Uplo::Upper, typename Group, typename T>
inline constexpr void hemv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixVectorOperand<T> operand) {
    detail::dispatch_symv<detail::SymmetricTransformTag<Side::Left, UploV, ComplexScalar<T>>, DeviceBlasPolicy::Auto>(group, a, operand);
}

template <DeviceBlasPolicy Policy, Uplo UploV = Uplo::Upper, typename Group, typename T>
inline constexpr void hemv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixVectorOperand<T> operand) {
    detail::dispatch_symv<detail::SymmetricTransformTag<Side::Left, UploV, ComplexScalar<T>>, Policy>(group, a, operand);
}

template <Uplo UploV = Uplo::Upper, typename Group, typename T>
inline constexpr void hemv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0)) {
    hemv<UploV>(group, a, make_matvec_operand(x, y, alpha, beta));
}

template <DeviceBlasPolicy Policy, Uplo UploV = Uplo::Upper, typename Group, typename T>
inline constexpr void hemv(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha = T(1),
                           T beta = T(0)) {
    (void)Policy;
    hemv<UploV>(group, a, x, y, alpha, beta);
}

} // namespace batchlas::device
