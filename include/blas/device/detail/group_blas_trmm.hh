#pragma once

#include <blas/device/detail/group_blas_common.hh>
#include <blas/device/detail/group_blas_subgroup_common.hh>

namespace batchlas::device {

namespace detail::generic {

template <typename Tag, typename Group, typename T>
inline constexpr void trmm(const Group& group,
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
                    partial += detail::triangular_matrix_entry<Tag>(a, row, k) * operand.b(k, col);
                } else {
                    partial += operand.b(row, k) * detail::triangular_matrix_entry<Tag>(a, k, col);
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
inline constexpr void trmm(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           TriangularTransform transform) {
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
        std::array<int, detail::subgroup::kMaxMatrixRowsPerSubgroup> begins{};
        std::array<int, detail::subgroup::kMaxMatrixRowsPerSubgroup> ends{};
        const int col_begin = transform.side == Side::Right && col < col_extent ?
            triangular_begin(col, contract_extent, transform, Side::Right) :
            0;
        const int col_end = transform.side == Side::Right && col < col_extent ?
            triangular_end(col, contract_extent, transform, Side::Right) :
            0;

        if (transform.side == Side::Left) {
            for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
                const int row = base_row + row_offset;
                if (row >= row_extent) {
                    continue;
                }
                begins[static_cast<std::size_t>(row_offset)] = triangular_begin(row, contract_extent, transform, Side::Left);
                ends[static_cast<std::size_t>(row_offset)] = triangular_end(row, contract_extent, transform, Side::Left);
            }
        }

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
                        if (k < begins[static_cast<std::size_t>(row_offset)] || k >= ends[static_cast<std::size_t>(row_offset)]) {
                            continue;
                        }
                        const T a_lane = lane == row_offset ? detail::triangular_matrix_entry(a, row, k, transform) : T(0);
                        const T a_value = sycl::select_from_group(sg, a_lane, static_cast<uint32_t>(row_offset));
                        partials[static_cast<std::size_t>(row_offset)] += a_value * b_value;
                    }
                }
            } else {
                for (int tile_offset = 0; tile_offset < tile_extent; ++tile_offset) {
                    const int k = k_base + tile_offset;
                    if (k < col_begin || k >= col_end) {
                        continue;
                    }
                    const T a_value = col < col_extent ? detail::triangular_matrix_entry(a, k, col, transform) : T(0);
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
inline constexpr void trmm_register_tiled(const Item& item,
                                          const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                          MatrixMatrixOperand<T> operand,
                                          TriangularTransform transform) {
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
    auto* workspace = sycl::ext::oneapi::group_local_memory_for_overwrite<RegisterMatrixWorkspace<T>>(item.get_group()).get();

    for (int tile_row = tile_row_start; tile_row < row_tiles; tile_row += tile_row_stride) {
        for (int tile_col = tile_col_start; tile_col < col_tiles; tile_col += tile_col_stride) {
            const int row_base = tile_row * kRegisterMatrixTileM + local_row * kRegisterMatrixThreadTileRows;
            const int col_base = tile_col * kRegisterMatrixTileN + local_col * kRegisterMatrixThreadTileCols;
            RegisterMatrixAccumTile<T> accum;
            const int tile_contract_begin = triangular_tile_begin(tile_row * kRegisterMatrixTileM,
                                                                  row_extent,
                                                                  tile_col * kRegisterMatrixTileN,
                                                                  col_extent,
                                                                  contract_extent,
                                                                  transform);
            const int tile_contract_end = triangular_tile_end(tile_row * kRegisterMatrixTileM,
                                                              row_extent,
                                                              tile_col * kRegisterMatrixTileN,
                                                              col_extent,
                                                              contract_extent,
                                                              transform);

            const int tile_count = std::max(0, (tile_contract_end - tile_contract_begin + kRegisterMatrixTileK - 1) / kRegisterMatrixTileK);
            if (tile_count > 0) {
                auto lhs_loader = [&](int tile_r, int global_k) {
                    const int global_row = tile_row * kRegisterMatrixTileM + tile_r;
                    if (global_row >= row_extent) {
                        return T(0);
                    }
                    return transform.side == Side::Left ?
                        detail::triangular_matrix_entry(a, global_row, global_k, transform) :
                        operand.b(global_row, global_k);
                };
                auto rhs_loader = [&](int tile_c, int global_k) {
                    const int global_col = tile_col * kRegisterMatrixTileN + tile_c;
                    if (global_col >= col_extent) {
                        return T(0);
                    }
                    return transform.side == Side::Left ?
                        operand.b(global_k, global_col) :
                        detail::triangular_matrix_entry(a, global_k, global_col, transform);
                };

                load_register_matrix_stage(item,
                                           workspace,
                                           linear_tid,
                                           subgroup_id,
                                           0,
                                           0,
                                           true,
                                           true,
                                           tile_contract_begin,
                                           std::min(kRegisterMatrixTileK, tile_contract_end - tile_contract_begin),
                                           lhs_loader,
                                           rhs_loader);
                sycl::group_barrier(item.get_group());

                for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
                    const int next_tile_idx = tile_idx + 1;
                    const int current_k_base = tile_contract_begin + tile_idx * kRegisterMatrixTileK;
                    const int current_tile_extent = std::min(kRegisterMatrixTileK, tile_contract_end - current_k_base);
                    const int current_lhs_stage = register_matrix_lhs_stage(tile_idx);
                    const int current_rhs_stage = register_matrix_rhs_stage(tile_idx);

                    if (next_tile_idx < tile_count) {
                        const int next_k_base = tile_contract_begin + next_tile_idx * kRegisterMatrixTileK;
                        const int next_tile_extent = std::min(kRegisterMatrixTileK, tile_contract_end - next_k_base);
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
                        const int next_k_base = tile_contract_begin + next_tile_idx * kRegisterMatrixTileK;
                        const int next_tile_extent = std::min(kRegisterMatrixTileK, tile_contract_end - next_k_base);
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

template <typename Tag, DeviceBlasPolicy Policy, typename Exec, typename T>
inline constexpr void dispatch_trmm(const Exec& exec,
                                    const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                    MatrixMatrixOperand<T> operand) {
    const TriangularTransform transform{.side = Tag::side, .uplo = Tag::uplo, .trans = Tag::trans, .diag = Tag::diag};
    validate_triangular_operand(a, operand, transform);

    if constexpr (detail::NdItemLike<Exec>) {
        const int row_extent = operand.c.rows();
        const int col_extent = operand.c.cols();
        const int contract_extent = transform.side == Side::Left ? operand.b.rows() : operand.b.cols();
        if (detail::subgroup::can_use_matrix_register_fast_path<T>(exec, row_extent, col_extent, contract_extent, Policy) &&
            std::same_as<std::remove_cvref_t<Exec>, sycl::nd_item<3>>) {
            detail::subgroup::trmm_register_tiled(exec, a, operand, transform);
            return;
        }
        if (detail::subgroup::can_use_matrix_fast_path<T>(exec, row_extent, col_extent, contract_extent, Policy)) {
            detail::subgroup::trmm(exec, a, operand, transform);
            return;
        }
    }

    generic::trmm<Tag>(exec, a, operand);
}

} // namespace detail

template <Side SideV = Side::Left,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          Diag DiagV = Diag::NonUnit,
          typename Group,
          typename T>
inline constexpr void trmm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand) {
    detail::dispatch_trmm<detail::TriangularTransformTag<SideV, UploV, TransV, DiagV>, DeviceBlasPolicy::Auto>(group, a, operand);
}

template <DeviceBlasPolicy Policy,
          Side SideV = Side::Left,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          Diag DiagV = Diag::NonUnit,
          typename Group,
          typename T>
inline constexpr void trmm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand) {
    detail::dispatch_trmm<detail::TriangularTransformTag<SideV, UploV, TransV, DiagV>, Policy>(group, a, operand);
}

template <Side SideV = Side::Left,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          Diag DiagV = Diag::NonUnit,
          typename Group,
          typename T>
inline constexpr void trmm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0)) {
    trmm<SideV, UploV, TransV, DiagV>(group, a, make_matmat_operand(b, c, alpha, beta));
}

template <DeviceBlasPolicy Policy,
          Side SideV = Side::Left,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          Diag DiagV = Diag::NonUnit,
          typename Group,
          typename T>
inline constexpr void trmm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0)) {
    (void)Policy;
    trmm<SideV, UploV, TransV, DiagV>(group, a, b, c, alpha, beta);
}

} // namespace batchlas::device
