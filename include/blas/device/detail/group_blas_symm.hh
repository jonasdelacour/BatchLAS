#pragma once

#include <blas/device/detail/group_blas_common.hh>
#include <blas/device/detail/group_blas_subgroup_common.hh>

namespace batchlas::device {

namespace detail::generic {

template <typename Tag, typename Group, typename T>
inline constexpr void symm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int row_extent = operand.c.rows();
    const int col_extent = operand.c.cols();
    const int contract_extent = Tag::side == Side::Left ? operand.b.rows() : operand.b.cols();

    for (int col = 0; col < col_extent; ++col) {
        for (int row = 0; row < row_extent; ++row) {
            T partial{};
            for (int k = local_id; k < contract_extent; k += local_size) {
                if constexpr (Tag::side == Side::Left) {
                    partial += detail::symmetric_matrix_entry<Tag>(a, row, k) * operand.b(k, col);
                } else {
                    partial += operand.b(row, k) * detail::symmetric_matrix_entry<Tag>(a, k, col);
                }
            }
            partial = detail::reduce_sum_group(group, partial);
            if (detail::group_is_leader(group)) {
                detail::write_matrix_output(operand, row, col, partial);
            }
        }
    }
}

} // namespace detail::generic

namespace detail::subgroup {

template <typename Item, typename T>
inline constexpr void symm(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           SymmetricTransform transform) {
    const auto sg = item.get_sub_group();
    const int sg_size = subgroup_size(item);
    const int lane = subgroup_local_id(item);
    const int sg_id = subgroup_group_id(item);
    const int total_sg = subgroup_count(item);
    const int row_extent = operand.c.rows();
    const int col_extent = operand.c.cols();
    const int contract_extent = transform.side == Side::Left ? operand.b.rows() : operand.b.cols();
    const int tile_k = matrix_tile_k(contract_extent, sg_size, transform.side);
    const int rows_per_sg = matrix_rows_per_subgroup(row_extent, total_sg, sg_size, contract_extent, transform.side);
    const int col_tiles = (col_extent + sg_size - 1) / sg_size;
    const int row_tiles = (row_extent + rows_per_sg - 1) / rows_per_sg;

    for (int linear_tile = sg_id; linear_tile < row_tiles * col_tiles; linear_tile += total_sg) {
        const int tile_row = linear_tile / col_tiles;
        const int tile_col = linear_tile % col_tiles;
        const int base_row = tile_row * rows_per_sg;
        const int base_col = tile_col * sg_size;
        const int col = base_col + lane;
        std::array<T, detail::subgroup::kMaxMatrixRowsPerSubgroup> partials{};

        for (int k_base = 0; k_base < contract_extent; k_base += tile_k) {
            const int tile_extent = std::min(tile_k, contract_extent - k_base);

            if (transform.side == Side::Left) {
                for (int tile_offset = 0; tile_offset < tile_extent; ++tile_offset) {
                    const int k = k_base + tile_offset;
                    const T b_value = col < col_extent ? operand.b(k, col) : T(0);
                    for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
                        const int row = base_row + row_offset;
                        if (row >= row_extent || col >= col_extent) {
                            continue;
                        }
                        const T a_lane = lane == row_offset ? detail::symmetric_matrix_entry(a, row, k, transform) : T(0);
                        const T a_value = sycl::select_from_group(sg, a_lane, static_cast<uint32_t>(row_offset));
                        partials[static_cast<std::size_t>(row_offset)] += a_value * b_value;
                    }
                }
            } else {
                for (int tile_offset = 0; tile_offset < tile_extent; ++tile_offset) {
                    const int k = k_base + tile_offset;
                    const T a_value = col < col_extent ? detail::symmetric_matrix_entry(a, k, col, transform) : T(0);
                    for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
                        const int row = base_row + row_offset;
                        const T b_lane = row < row_extent && lane == row_offset ? operand.b(row, k) : T(0);
                        const T b_value = sycl::select_from_group(sg, b_lane, static_cast<uint32_t>(row_offset));
                        if (row >= row_extent || col >= col_extent) {
                            continue;
                        }
                        partials[static_cast<std::size_t>(row_offset)] += b_value * a_value;
                    }
                }
            }
        }

        if (col < col_extent) {
            for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
                const int row = base_row + row_offset;
                if (row >= row_extent) {
                    continue;
                }
                detail::write_matrix_output(operand, row, col, partials[static_cast<std::size_t>(row_offset)]);
            }
        }
    }
}

template <typename Item, typename T>
inline constexpr void symm_register_tiled(const Item& item,
                                          const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                          MatrixMatrixOperand<T> operand,
                                          SymmetricTransform transform,
                                          RegisterMatrixWorkspace<T>* workspace) {
    const int linear_tid = detail::item_local_linear_id(item);
    const int subgroup_id = subgroup_group_id(item);
    const int local_row = linear_tid / kRegisterMatrixLocalCols;
    const int local_col = linear_tid % kRegisterMatrixLocalCols;
    const int row_extent = operand.c.rows();
    const int col_extent = operand.c.cols();
    const int contract_extent = transform.side == Side::Left ? operand.b.rows() : operand.b.cols();
    const int row_tiles = (row_extent + kRegisterMatrixTileM - 1) / kRegisterMatrixTileM;
    const int col_tiles = (col_extent + kRegisterMatrixTileN - 1) / kRegisterMatrixTileN;
    const int tile_row_start = matrix_tile_group_row(item);
    const int tile_col_start = matrix_tile_group_col(item);
    const int tile_row_stride = matrix_tile_group_row_stride(item);
    const int tile_col_stride = matrix_tile_group_col_stride(item);
    for (int tile_row = tile_row_start; tile_row < row_tiles; tile_row += tile_row_stride) {
        for (int tile_col = tile_col_start; tile_col < col_tiles; tile_col += tile_col_stride) {
            const int row_base = tile_row * kRegisterMatrixTileM + local_row * kRegisterMatrixThreadTileRows;
            const int col_base = tile_col * kRegisterMatrixTileN + local_col * kRegisterMatrixThreadTileCols;
            RegisterMatrixAccumTile<T> accum;

            const int tile_count = (contract_extent + kRegisterMatrixTileK - 1) / kRegisterMatrixTileK;
            if (tile_count > 0) {
                auto lhs_loader = [&](int tile_r, int global_k) {
                    const int global_row = tile_row * kRegisterMatrixTileM + tile_r;
                    if (global_row >= row_extent) {
                        return T(0);
                    }
                    return transform.side == Side::Left ?
                        detail::symmetric_matrix_entry(a, global_row, global_k, transform) :
                        operand.b(global_row, global_k);
                };
                auto rhs_loader = [&](int tile_c, int global_k) {
                    const int global_col = tile_col * kRegisterMatrixTileN + tile_c;
                    if (global_col >= col_extent) {
                        return T(0);
                    }
                    return transform.side == Side::Left ?
                        operand.b(global_k, global_col) :
                        detail::symmetric_matrix_entry(a, global_k, global_col, transform);
                };

                load_register_matrix_stage(item,
                                           workspace,
                                           linear_tid,
                                           subgroup_id,
                                           0,
                                           0,
                                           true,
                                           true,
                                           0,
                                           std::min(kRegisterMatrixTileK, contract_extent),
                                           lhs_loader,
                                           rhs_loader);
                sycl::group_barrier(item.get_group());

                for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
                    const int current_k_base = tile_idx * kRegisterMatrixTileK;
                    const int current_tile_extent = std::min(kRegisterMatrixTileK, contract_extent - current_k_base);
                    const int next_tile_idx = tile_idx + 1;
                    const int current_lhs_stage = register_matrix_lhs_stage(tile_idx);
                    const int current_rhs_stage = register_matrix_rhs_stage(tile_idx);

                    if (next_tile_idx < tile_count) {
                        const int next_k_base = next_tile_idx * kRegisterMatrixTileK;
                        const int next_tile_extent = std::min(kRegisterMatrixTileK, contract_extent - next_k_base);
                        if constexpr (kRegisterMatrixRhsStages > 1) {
                            load_register_matrix_stage(item,
                                                       workspace,
                                                       linear_tid,
                                                       subgroup_id,
                                                       current_lhs_stage,
                                                       register_matrix_rhs_stage(next_tile_idx),
                                                       false,
                                                       kRegisterMatrixRhsStages > 1,
                                                       next_k_base,
                                                       next_tile_extent,
                                                       lhs_loader,
                                                       rhs_loader);
                        }
                    }

                    accumulate_register_matrix_stage(item,
                                                     workspace,
                                                     subgroup_id,
                                                     current_lhs_stage,
                                                     current_rhs_stage,
                                                     local_row,
                                                     local_col,
                                                     current_tile_extent,
                                                     accum);

                    if (next_tile_idx < tile_count) {
                        const int next_k_base = next_tile_idx * kRegisterMatrixTileK;
                        const int next_tile_extent = std::min(kRegisterMatrixTileK, contract_extent - next_k_base);
                        if constexpr (kRegisterMatrixRhsStages == 1) {
                            sycl::group_barrier(item.get_group());
                        } else {
                            sycl::group_barrier(item.get_sub_group());
                        }
                        load_register_matrix_stage(item,
                                                   workspace,
                                                   linear_tid,
                                                   subgroup_id,
                                                   0,
                                                   kRegisterMatrixRhsStages == 1 ? 0 : register_matrix_rhs_stage(next_tile_idx),
                                                   true,
                                                   kRegisterMatrixRhsStages == 1,
                                                   next_k_base,
                                                   next_tile_extent,
                                                   lhs_loader,
                                                   rhs_loader);
                        sycl::group_barrier(item.get_group());
                    }
                }
            }

            write_register_matrix_tile(operand, row_base, col_base, row_extent, col_extent, accum);
        }
    }
}

} // namespace detail::subgroup

namespace detail {

template <typename Tag, DeviceBlasPolicy Policy, typename T>
inline constexpr std::size_t symm_workspace_elements(const DeviceBlasLaunchInfo& launch,
                                                     int row_extent,
                                                     int col_extent) {
    assert(row_extent >= 0 && col_extent >= 0 && "device::symm workspace query expects non-negative matrix extents");
    if (!detail::subgroup::is_nd_item_3d_launch(launch)) {
        return 0;
    }

    const int contract_extent = Tag::side == Side::Left ? row_extent : col_extent;
    if (detail::subgroup::can_use_matrix_register_fast_path<T>(launch, row_extent, col_extent, contract_extent, Policy)) {
        return detail::workspace_elements_v<T, detail::subgroup::RegisterMatrixWorkspace<T>>;
    }
    return 0;
}

template <typename Tag, DeviceBlasPolicy Policy, typename Exec, typename T>
inline constexpr void dispatch_symm(const Exec& exec,
                                    const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                    MatrixMatrixOperand<T> operand,
                                    T* workspace = nullptr) {
    const SymmetricTransform transform{.side = Tag::side, .uplo = Tag::uplo, .hermitian = Tag::hermitian};
    validate_symmetric_operand(a, operand, transform);

    if constexpr (detail::NdItemLike<Exec>) {
        const int row_extent = operand.c.rows();
        const int col_extent = operand.c.cols();
        const int contract_extent = transform.side == Side::Left ? operand.b.rows() : operand.b.cols();
        auto* register_workspace = workspace == nullptr ? nullptr : detail::workspace_ptr_cast<detail::subgroup::RegisterMatrixWorkspace<T>>(workspace);
        if (register_workspace != nullptr &&
            detail::subgroup::can_use_matrix_register_fast_path<T>(exec, row_extent, col_extent, contract_extent, Policy) &&
            std::same_as<std::remove_cvref_t<Exec>, sycl::nd_item<3>>) {
            detail::subgroup::symm_register_tiled(exec, a, operand, transform, register_workspace);
            return;
        }
        if (detail::subgroup::can_use_matrix_fast_path<T>(exec, row_extent, col_extent, contract_extent, Policy)) {
            detail::subgroup::symm(exec, a, operand, transform);
            return;
        }
    }

    generic::symm<Tag>(exec, a, operand);
}

} // namespace detail

template <Side SideV = Side::Left, Uplo UploV = Uplo::Upper, typename Group, typename T>
inline constexpr void symm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           T* workspace = nullptr) {
    detail::dispatch_symm<detail::SymmetricTransformTag<SideV, UploV, false>, DeviceBlasPolicy::Auto>(group, a, operand, workspace);
}

template <DeviceBlasPolicy Policy,
          Side SideV = Side::Left,
          Uplo UploV = Uplo::Upper,
          typename Group,
          typename T>
inline constexpr void symm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           T* workspace = nullptr) {
    detail::dispatch_symm<detail::SymmetricTransformTag<SideV, UploV, false>, Policy>(group, a, operand, workspace);
}

template <typename T, Side SideV = Side::Left, Uplo UploV = Uplo::Upper>
inline constexpr std::size_t symm_workspace_elements(const DeviceBlasLaunchInfo& launch,
                                                     int row_extent,
                                                     int col_extent) {
    return detail::symm_workspace_elements<detail::SymmetricTransformTag<SideV, UploV, false>, DeviceBlasPolicy::Auto, T>(launch, row_extent, col_extent);
}

template <typename T, DeviceBlasPolicy Policy, Side SideV = Side::Left, Uplo UploV = Uplo::Upper>
inline constexpr std::size_t symm_workspace_elements(const DeviceBlasLaunchInfo& launch,
                                                     int row_extent,
                                                     int col_extent) {
    return detail::symm_workspace_elements<detail::SymmetricTransformTag<SideV, UploV, false>, Policy, T>(launch, row_extent, col_extent);
}

template <Side SideV = Side::Left, Uplo UploV = Uplo::Upper, typename Group, typename T>
inline constexpr void symm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0),
                           T* workspace = nullptr) {
    symm<SideV, UploV>(group, a, make_matmat_operand(b, c, alpha, beta), workspace);
}

template <DeviceBlasPolicy Policy,
          Side SideV = Side::Left,
          Uplo UploV = Uplo::Upper,
          typename Group,
          typename T>
inline constexpr void symm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0),
                           T* workspace = nullptr) {
    symm<Policy, SideV, UploV>(group, a, make_matmat_operand(b, c, alpha, beta), workspace);
}

} // namespace batchlas::device
