#pragma once

#include <blas/device/detail/group_blas_common.hh>
#include <blas/device/detail/group_blas_subgroup_common.hh>

namespace batchlas::device {

namespace detail::generic {

template <typename Tag, typename Group, typename T>
inline constexpr void gemm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int row_extent = detail::output_size<Tag::trans_a>(a);
    const int col_extent = detail::input_size<Tag::trans_b>(operand.b);
    const int contract_extent = detail::input_size<Tag::trans_a>(a);

    for (int col = 0; col < col_extent; ++col) {
        for (int row = 0; row < row_extent; ++row) {
            T partial{};
            for (int k = local_id; k < contract_extent; k += local_size) {
                partial += detail::matrix_entry<Tag::trans_a>(a, row, k) *
                    detail::matrix_entry<Tag::trans_b>(operand.b, k, col);
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
inline constexpr void gemm(const Item& item,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           GeneralMatrixTransform transform) {
    const auto sg = item.get_sub_group();
    const int sg_size = subgroup_size(item);
    const int lane = subgroup_local_id(item);
    const int sg_id = subgroup_group_id(item);
    const int total_sg = subgroup_count(item);
    const int row_extent = detail::output_size(a, transform.trans_a);
    const int col_extent = detail::input_size(operand.b, transform.trans_b);
    const int contract_extent = detail::input_size(a, transform.trans_a);
    const int tile_k = matrix_tile_k(contract_extent, sg_size, Side::Left);
    const int rows_per_sg = matrix_rows_per_subgroup(row_extent, total_sg, sg_size, contract_extent, Side::Left);
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

            for (int tile_offset = 0; tile_offset < tile_extent; ++tile_offset) {
                const int k = k_base + tile_offset;
                const T b_value = col < col_extent ? detail::matrix_entry(operand.b, k, col, transform.trans_b) : T(0);
                for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
                    const int row = base_row + row_offset;
                    if (row >= row_extent || col >= col_extent) {
                        continue;
                    }
                    const T a_lane = lane == row_offset ? detail::matrix_entry(a, row, k, transform.trans_a) : T(0);
                    const T a_value = sycl::select_from_group(sg, a_lane, static_cast<uint32_t>(row_offset));
                    partials[static_cast<std::size_t>(row_offset)] += a_value * b_value;
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
inline constexpr void gemm_register_tiled(const Item& item,
                                          const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                          MatrixMatrixOperand<T> operand,
                                          GeneralMatrixTransform transform,
                                          detail::subgroup::GemmWorkspace<T>* gemm_workspace) {
    const int linear_tid = detail::item_local_linear_id(item);
    const int subgroup_id = detail::subgroup::subgroup_group_id(item);
    const int local_row = linear_tid / detail::subgroup::kRegisterMatrixLocalCols;
    const int local_col = linear_tid % detail::subgroup::kRegisterMatrixLocalCols;
    const int row_extent = detail::output_size(a, transform.trans_a);
    const int col_extent = detail::input_size(operand.b, transform.trans_b);
    const int contract_extent = detail::input_size(a, transform.trans_a);
    const int row_tiles = (row_extent + detail::subgroup::kRegisterMatrixTileM - 1) / detail::subgroup::kRegisterMatrixTileM;
    const int col_tiles = (col_extent + detail::subgroup::kRegisterMatrixTileN - 1) / detail::subgroup::kRegisterMatrixTileN;
    const int tile_row_start = detail::subgroup::matrix_tile_group_row(item);
    const int tile_col_start = detail::subgroup::matrix_tile_group_col(item);
    const int tile_row_stride = detail::subgroup::matrix_tile_group_row_stride(item);
    const int tile_col_stride = detail::subgroup::matrix_tile_group_col_stride(item);
    auto* workspace = &gemm_workspace->register_workspace;

    for (int tile_row = tile_row_start; tile_row < row_tiles; tile_row += tile_row_stride) {
        for (int tile_col = tile_col_start; tile_col < col_tiles; tile_col += tile_col_stride) {
            const int row_base = tile_row * detail::subgroup::kRegisterMatrixTileM + local_row * detail::subgroup::kRegisterMatrixThreadTileRows;
            const int col_base = tile_col * detail::subgroup::kRegisterMatrixTileN + local_col * detail::subgroup::kRegisterMatrixThreadTileCols;
            detail::subgroup::RegisterMatrixAccumTile<T> accum;
            const int tile_count = (contract_extent + detail::subgroup::kRegisterMatrixTileK - 1) / detail::subgroup::kRegisterMatrixTileK;

            if (tile_count > 0) {
                auto lhs_loader = [&](int tile_r, int global_k) {
                    const int global_row = tile_row * detail::subgroup::kRegisterMatrixTileM + tile_r;
                    if (global_row >= row_extent) {
                        return T(0);
                    }
                    return detail::matrix_entry(a, global_row, global_k, transform.trans_a);
                };
                auto rhs_loader = [&](int tile_c, int global_k) {
                    const int global_col = tile_col * detail::subgroup::kRegisterMatrixTileN + tile_c;
                    if (global_col >= col_extent) {
                        return T(0);
                    }
                    return detail::matrix_entry(operand.b, global_k, global_col, transform.trans_b);
                };

                detail::subgroup::load_register_matrix_stage(item,
                                                             workspace,
                                                             linear_tid,
                                                             subgroup_id,
                                                             0,
                                                             0,
                                                             true,
                                                             true,
                                                             0,
                                                             std::min(detail::subgroup::kRegisterMatrixTileK, contract_extent),
                                                             lhs_loader,
                                                             rhs_loader);
                sycl::group_barrier(item.get_group());

                for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
                    const int current_tile_extent = std::min(detail::subgroup::kRegisterMatrixTileK,
                                                             contract_extent - tile_idx * detail::subgroup::kRegisterMatrixTileK);
                    const int next_tile_idx = tile_idx + 1;
                    const int current_lhs_stage = detail::subgroup::register_matrix_lhs_stage(tile_idx);
                    const int current_rhs_stage = detail::subgroup::register_matrix_rhs_stage(tile_idx);

                    if (next_tile_idx < tile_count) {
                        const int next_k_base = next_tile_idx * detail::subgroup::kRegisterMatrixTileK;
                        const int next_tile_extent = std::min(detail::subgroup::kRegisterMatrixTileK, contract_extent - next_k_base);
                        if constexpr (detail::subgroup::kRegisterMatrixRhsStages > 1) {
                            detail::subgroup::load_register_matrix_stage(item,
                                                                         workspace,
                                                                         linear_tid,
                                                                         subgroup_id,
                                                                         current_lhs_stage,
                                                                         detail::subgroup::register_matrix_rhs_stage(next_tile_idx),
                                                                         false,
                                                                         detail::subgroup::kRegisterMatrixRhsStages > 1,
                                                                         next_k_base,
                                                                         next_tile_extent,
                                                                         lhs_loader,
                                                                         rhs_loader);
                        }
                    }

                    detail::subgroup::accumulate_register_matrix_stage(item,
                                                                       workspace,
                                                                       subgroup_id,
                                                                       current_lhs_stage,
                                                                       current_rhs_stage,
                                                                       local_row,
                                                                       local_col,
                                                                       current_tile_extent,
                                                                       accum);

                    if (next_tile_idx < tile_count) {
                        const int next_k_base = next_tile_idx * detail::subgroup::kRegisterMatrixTileK;
                        const int next_tile_extent = std::min(detail::subgroup::kRegisterMatrixTileK, contract_extent - next_k_base);
                        if constexpr (detail::subgroup::kRegisterMatrixRhsStages == 1) {
                            sycl::group_barrier(item.get_group());
                        } else {
                            sycl::group_barrier(item.get_sub_group());
                        }
                        detail::subgroup::load_register_matrix_stage(item,
                                                                     workspace,
                                                                     linear_tid,
                                                                     subgroup_id,
                                                                     0,
                                                                     detail::subgroup::kRegisterMatrixRhsStages == 1 ? 0 : detail::subgroup::register_matrix_rhs_stage(next_tile_idx),
                                                                     true,
                                                                     detail::subgroup::kRegisterMatrixRhsStages == 1,
                                                                     next_k_base,
                                                                     next_tile_extent,
                                                                     lhs_loader,
                                                                     rhs_loader);
                        sycl::group_barrier(item.get_group());
                    }
                }
            }

            detail::subgroup::write_register_matrix_tile(operand, row_base, col_base, row_extent, col_extent, accum);
        }
    }
}

template <typename Item, typename T>
inline constexpr void gemm_aligned_nn_large(const Item& item,
                                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                            MatrixMatrixOperand<T> operand,
                                            detail::subgroup::GemmWorkspace<T>* gemm_workspace) {
    const int linear_tid = detail::item_local_linear_id(item);
    const int local_row = linear_tid / detail::subgroup::kOptimizedGemmLocalCols;
    const int local_col = linear_tid % detail::subgroup::kOptimizedGemmLocalCols;
    const int row_extent = a.rows();
    const int col_extent = operand.b.cols();
    const int contract_extent = a.cols();
    const int row_tiles = row_extent / detail::subgroup::kOptimizedGemmTileM;
    const int col_tiles = col_extent / detail::subgroup::kRegisterMatrixTileN;
    const int tile_row_start = detail::subgroup::matrix_tile_group_row(item);
    const int tile_col_start = detail::subgroup::matrix_tile_group_col(item);
    const int tile_row_stride = detail::subgroup::matrix_tile_group_row_stride(item);
    const int tile_col_stride = detail::subgroup::matrix_tile_group_col_stride(item);
    auto* workspace = &gemm_workspace->optimized_workspace;

    for (int tile_row = tile_row_start; tile_row < row_tiles; tile_row += tile_row_stride) {
        for (int tile_col = tile_col_start; tile_col < col_tiles; tile_col += tile_col_stride) {
            const int row_base = tile_row * detail::subgroup::kOptimizedGemmTileM + local_row * detail::subgroup::kOptimizedGemmThreadTileRows;

            for (int subtile_col = 0; subtile_col < detail::subgroup::kRegisterMatrixTileN / detail::subgroup::kOptimizedGemmTileN; ++subtile_col) {
                const int col_tile_base = tile_col * detail::subgroup::kRegisterMatrixTileN + subtile_col * detail::subgroup::kOptimizedGemmTileN;
                const int col_base = col_tile_base + local_col * detail::subgroup::kOptimizedGemmThreadTileCols;
                T accum[detail::subgroup::kOptimizedGemmThreadTileRows][detail::subgroup::kOptimizedGemmThreadTileCols]{};

                auto load_stage = [&](int k_base, int stage) {
                    T* lhs_stage = workspace->lhs + stage * detail::subgroup::kOptimizedGemmStageASize;
                    T* rhs_stage = workspace->rhs + stage * detail::subgroup::kOptimizedGemmStageBSize;

                    constexpr int lhs_packets_per_col = detail::subgroup::kOptimizedGemmTileM / detail::subgroup::kOptimizedGemmVecA;
                    constexpr int lhs_packet_count = lhs_packets_per_col * detail::subgroup::kOptimizedGemmTileK;
                    constexpr int lhs_packet_iterations = lhs_packet_count / detail::subgroup::kOptimizedGemmThreadsPerGroup;
                    detail::subgroup::static_for<lhs_packet_iterations>([&](auto iter) {
                        constexpr int iteration = iter;
                        const int packet = linear_tid + iteration * detail::subgroup::kOptimizedGemmThreadsPerGroup;
                        const int lhs_row = (packet % lhs_packets_per_col) * detail::subgroup::kOptimizedGemmVecA;
                        const int lhs_col = packet / lhs_packets_per_col;
                        const int base = (k_base + lhs_col) * a.ld() + tile_row * detail::subgroup::kOptimizedGemmTileM + lhs_row;
                        const auto packet_values = detail::subgroup::packet_load_aligned<T, detail::subgroup::kOptimizedGemmVecA>(a.data(), base);
                        detail::subgroup::static_for<detail::subgroup::kOptimizedGemmVecA>([&](auto lane_idx) {
                            constexpr int lane = lane_idx;
                            lhs_stage[lhs_col * detail::subgroup::kOptimizedGemmTileAStride + lhs_row + lane] = packet_values[lane];
                        });
                    });

                    constexpr int rhs_packets_per_col = detail::subgroup::kOptimizedGemmTileK / detail::subgroup::kOptimizedGemmVecB;
                    constexpr int rhs_packet_count = detail::subgroup::kOptimizedGemmTileN * rhs_packets_per_col;
                    constexpr int rhs_packet_iterations = rhs_packet_count / detail::subgroup::kOptimizedGemmThreadsPerGroup;
                    detail::subgroup::static_for<rhs_packet_iterations>([&](auto iter) {
                        constexpr int iteration = iter;
                        const int packet = linear_tid + iteration * detail::subgroup::kOptimizedGemmThreadsPerGroup;
                        const int rhs_row = (packet % rhs_packets_per_col) * detail::subgroup::kOptimizedGemmVecB;
                        const int rhs_col = packet / rhs_packets_per_col;
                        const int base = (col_tile_base + rhs_col) * operand.b.ld() + k_base + rhs_row;
                        const auto packet_values = detail::subgroup::packet_load_aligned<T, detail::subgroup::kOptimizedGemmVecB>(operand.b.data(), base);
                        detail::subgroup::static_for<detail::subgroup::kOptimizedGemmVecB>([&](auto lane_idx) {
                            constexpr int lane = lane_idx;
                            rhs_stage[rhs_col * detail::subgroup::kOptimizedGemmTileBStride + rhs_row + lane] = packet_values[lane];
                        });
                    });
                };

                auto accumulate_stage = [&](const T* lhs_stage, const T* rhs_stage) {
                    for (int t0 = 0; t0 < detail::subgroup::kOptimizedGemmTileK; t0 += detail::subgroup::kOptimizedGemmUnrollK) {
                        for (int unroll = 0; unroll < detail::subgroup::kOptimizedGemmUnrollK; ++unroll) {
                            const int t = t0 + unroll;
                            T lhs_frag[detail::subgroup::kOptimizedGemmThreadTileRows];
                            T rhs_frag[detail::subgroup::kOptimizedGemmThreadTileCols];
                            for (int i = 0; i < detail::subgroup::kOptimizedGemmThreadTileRows; ++i) {
                                lhs_frag[i] = lhs_stage[t * detail::subgroup::kOptimizedGemmTileAStride +
                                                         local_row * detail::subgroup::kOptimizedGemmThreadTileRows + i];
                            }
                            for (int j = 0; j < detail::subgroup::kOptimizedGemmThreadTileCols; ++j) {
                                rhs_frag[j] = rhs_stage[(local_col * detail::subgroup::kOptimizedGemmThreadTileCols + j) *
                                                        detail::subgroup::kOptimizedGemmTileBStride + t];
                            }
                            for (int i = 0; i < detail::subgroup::kOptimizedGemmThreadTileRows; ++i) {
                                for (int j = 0; j < detail::subgroup::kOptimizedGemmThreadTileCols; ++j) {
                                    accum[i][j] += lhs_frag[i] * rhs_frag[j];
                                }
                            }
                        }
                    }
                };

                const int tile_count = contract_extent / detail::subgroup::kOptimizedGemmTileK;
                if (tile_count > 0) {
                    load_stage(0, 0);
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int tile_idx = 0; tile_idx + 1 < tile_count; ++tile_idx) {
                        const int current_stage = tile_idx & 1;
                        const int next_stage = current_stage ^ 1;
                        load_stage((tile_idx + 1) * detail::subgroup::kOptimizedGemmTileK, next_stage);
                        accumulate_stage(workspace->lhs + current_stage * detail::subgroup::kOptimizedGemmStageASize,
                                         workspace->rhs + current_stage * detail::subgroup::kOptimizedGemmStageBSize);
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    const int final_stage = (tile_count - 1) & 1;
                    accumulate_stage(workspace->lhs + final_stage * detail::subgroup::kOptimizedGemmStageASize,
                                     workspace->rhs + final_stage * detail::subgroup::kOptimizedGemmStageBSize);
                    item.barrier(sycl::access::fence_space::local_space);
                }

                for (int i = 0; i < detail::subgroup::kOptimizedGemmThreadTileRows; ++i) {
                    const int row = row_base + i;
                    for (int j = 0; j < detail::subgroup::kOptimizedGemmThreadTileCols; ++j) {
                        const int col = col_base + j;
                        detail::write_matrix_output(operand, row, col, accum[i][j]);
                    }
                }
            }
        }
    }
}

} // namespace detail::subgroup

namespace detail {

template <typename Tag, DeviceBlasPolicy Policy, typename T>
inline constexpr std::size_t gemm_workspace_elements(const DeviceBlasLaunchInfo& launch,
                                                     int row_extent,
                                                     int col_extent,
                                                     int contract_extent,
                                                     bool aligned_a = false,
                                                     bool aligned_b = false) {
    assert(row_extent >= 0 && col_extent >= 0 && contract_extent >= 0 &&
           "device::gemm workspace query expects non-negative matrix extents");

    if (aligned_a && aligned_b &&
        detail::subgroup::can_use_matrix_aligned_nn_large_fast_path<T>(launch,
                                                                       row_extent,
                                                                       col_extent,
                                                                       contract_extent,
                                                                       Tag::trans_a,
                                                                       Tag::trans_b,
                                                                       Policy) &&
        detail::subgroup::gemm_workspace_supported_v<T>) {
        return detail::workspace_elements_v<T, detail::subgroup::GemmWorkspace<T>>;
    }

    if (detail::subgroup::can_use_matrix_register_fast_path<T>(launch, row_extent, col_extent, contract_extent, Policy)) {
        // can_use_matrix_register_fast_path already checks register_matrix_workspace_supported_v<T>.
        // Allocate full GemmWorkspace if it fits (to allow aligned-nn-large fallback at runtime),
        // otherwise allocate only the smaller RegisterMatrixWorkspace.
        if constexpr (detail::subgroup::gemm_workspace_supported_v<T>) {
            return detail::workspace_elements_v<T, detail::subgroup::GemmWorkspace<T>>;
        } else {
            return detail::workspace_elements_v<T, detail::subgroup::RegisterMatrixWorkspace<T>>;
        }
    }
    return 0;
}

template <typename Tag, DeviceBlasPolicy Policy, typename Exec, typename T>
inline constexpr void dispatch_gemm(const Exec& exec,
                                    const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                    MatrixMatrixOperand<T> operand,
                                    T* workspace = nullptr) {
    const GeneralMatrixTransform transform{.trans_a = Tag::trans_a, .trans_b = Tag::trans_b};
    validate_gemm_operand(a, operand, transform);

    if constexpr (detail::NdItemLike<Exec>) {
        auto* gemm_workspace = workspace == nullptr ? nullptr : detail::workspace_ptr_cast<detail::subgroup::GemmWorkspace<T>>(workspace);

        if (gemm_workspace != nullptr && detail::subgroup::can_use_matrix_aligned_nn_large_fast_path<T>(exec, a, operand, transform, Policy) &&
            detail::subgroup::optimized_gemm_workspace_supported_v<T>) {
            detail::subgroup::gemm_aligned_nn_large(exec, a, operand, gemm_workspace);
            return;
        }

        const int row_extent = detail::output_size(a, transform.trans_a);
        const int col_extent = detail::input_size(operand.b, transform.trans_b);
        const int contract_extent = detail::input_size(a, transform.trans_a);
        if (gemm_workspace != nullptr && detail::subgroup::can_use_matrix_register_fast_path<T>(exec, row_extent, col_extent, contract_extent, Policy) &&
            detail::subgroup::register_matrix_workspace_supported_v<T>) {
            detail::subgroup::gemm_register_tiled(exec, a, operand, transform, gemm_workspace);
            return;
        }
        if (detail::subgroup::can_use_matrix_fast_path<T>(exec, row_extent, col_extent, contract_extent, Policy)) {
            detail::subgroup::gemm(exec, a, operand, transform);
            return;
        }
        // For 3D nd_item launches without a fast path, multiple work-groups would
        // independently iterate over all output cells in the generic fallback, causing
        // data races. Restrict the generic fallback to the primary work-group only.
        if constexpr (std::is_same_v<std::remove_cvref_t<Exec>, sycl::nd_item<3>>) {
            if (detail::subgroup::matrix_tile_group_row(exec) == 0 &&
                detail::subgroup::matrix_tile_group_col(exec) == 0) {
                generic::gemm<Tag>(exec, a, operand);
            }
            return;
        }
    }

    generic::gemm<Tag>(exec, a, operand);
}

} // namespace detail

template <Transpose TransAV = Transpose::NoTrans, Transpose TransBV = Transpose::NoTrans, typename Group, typename T>
inline constexpr void gemm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           T* workspace = nullptr) {
    detail::dispatch_gemm<GeneralMatrixTransformTag<TransAV, TransBV>, DeviceBlasPolicy::Auto>(group, a, operand, workspace);
}

template <DeviceBlasPolicy Policy,
          Transpose TransAV = Transpose::NoTrans,
          Transpose TransBV = Transpose::NoTrans,
          typename Group,
          typename T>
inline constexpr void gemm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           T* workspace = nullptr) {
    detail::dispatch_gemm<GeneralMatrixTransformTag<TransAV, TransBV>, Policy>(group, a, operand, workspace);
}

template <typename T, Transpose TransAV = Transpose::NoTrans, Transpose TransBV = Transpose::NoTrans>
inline constexpr std::size_t gemm_workspace_elements(const DeviceBlasLaunchInfo& launch,
                                                     int row_extent,
                                                     int col_extent,
                                                     int contract_extent,
                                                     bool aligned_a = false,
                                                     bool aligned_b = false) {
    return detail::gemm_workspace_elements<GeneralMatrixTransformTag<TransAV, TransBV>, DeviceBlasPolicy::Auto, T>(
        launch, row_extent, col_extent, contract_extent, aligned_a, aligned_b);
}

template <typename T,
          DeviceBlasPolicy Policy,
          Transpose TransAV = Transpose::NoTrans,
          Transpose TransBV = Transpose::NoTrans>
inline constexpr std::size_t gemm_workspace_elements(const DeviceBlasLaunchInfo& launch,
                                                     int row_extent,
                                                     int col_extent,
                                                     int contract_extent,
                                                     bool aligned_a = false,
                                                     bool aligned_b = false) {
    return detail::gemm_workspace_elements<GeneralMatrixTransformTag<TransAV, TransBV>, Policy, T>(
        launch, row_extent, col_extent, contract_extent, aligned_a, aligned_b);
}

template <Transpose TransAV = Transpose::NoTrans, Transpose TransBV = Transpose::NoTrans, typename Group, typename T>
inline constexpr void gemm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0),
                           T* workspace = nullptr) {
    gemm<TransAV, TransBV>(group, a, make_matmat_operand(b, c, alpha, beta), workspace);
}

template <DeviceBlasPolicy Policy,
          Transpose TransAV = Transpose::NoTrans,
          Transpose TransBV = Transpose::NoTrans,
          typename Group,
          typename T>
inline constexpr void gemm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& b,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0),
                           T* workspace = nullptr) {
    gemm<Policy, TransAV, TransBV>(group, a, make_matmat_operand(b, c, alpha, beta), workspace);
}

} // namespace batchlas::device
