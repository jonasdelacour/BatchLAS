// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/blas/device/detail/group_blas_common.hh>
#include <batchlas/blas/device/detail/group_blas_subgroup_common.hh>

namespace batchlas::device {

namespace detail::generic {

/// @brief Generic triangular matrix-matrix product, one group-wide reduction per element of \f$C\f$; no operand checks.
///
/// Elements are visited in an order that never overwrites an entry of `B` a later element still reads,
/// so `B` and `C` may be the same view.
/// @param[in]     group    executor whose work-items share each reduction; every work-item must call
/// @param[in]     a        single triangular matrix \f$A\f$; only the `Tag::uplo` triangle is read (not its diagonal when `Tag::diag` is `Unit`)
/// @param[in,out] operand  `B` (read), `C` (read when scaled by \f$\beta\f$, overwritten with the result
///                         by the group leader), \f$\alpha\f$, \f$\beta\f$
template <typename Tag, typename Group, typename T>
inline constexpr void trmm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int row_extent = operand.c.rows();
    const int col_extent = operand.c.cols();
    const int contract_extent = Tag::side == Side::Left ? operand.b.rows() : operand.b.cols();
    constexpr bool effective_lower = Tag::trans == Transpose::NoTrans ? Tag::uplo == Uplo::Lower : Tag::uplo == Uplo::Upper;

    if constexpr (Tag::side == Side::Left) {
        for (int row_step = 0; row_step < row_extent; ++row_step) {
            const int row = effective_lower ? (row_extent - 1 - row_step) : row_step;
            for (int col = 0; col < col_extent; ++col) {
                T partial{};
                for (int k = local_id; k < contract_extent; k += local_size) {
                    partial += detail::triangular_matrix_entry<Tag>(a, row, k) * operand.b(k, col);
                }
                partial = detail::reduce_sum_group(group, partial);
                if (detail::group_is_leader(group)) {
                    detail::write_matrix_output(operand, row, col, partial);
                }
            }
        }
        return;
    }

    for (int col_step = 0; col_step < col_extent; ++col_step) {
        const int col = effective_lower ? col_step : (col_extent - 1 - col_step);
        for (int row = 0; row < row_extent; ++row) {
            T partial{};
            for (int k = local_id; k < contract_extent; k += local_size) {
                partial += operand.b(row, k) * detail::triangular_matrix_entry<Tag>(a, k, col);
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

/// @brief Sub-group tiled triangular matrix-matrix product: sub-groups stride over tiles of \f$C\f$ (`rows_per_sg` rows by sub-group-size columns); each lane owns one column of a tile.
/// @param[in]     item       `nd_item` of the calling work-item; every work-item of the work-group must call
/// @param[in]     a          single triangular matrix \f$A\f$; only the `transform.uplo` triangle is read (not its diagonal when `transform.diag` is `Unit`)
/// @param[in,out] operand    `B` (read; must not overlap `C`), `C` (read when scaled by \f$\beta\f$,
///                           overwritten with the result), \f$\alpha\f$, \f$\beta\f$
/// @param[in]     transform  side, stored triangle, \f$\mathrm{op}(\cdot)\f$ and diagonal kind of \f$A\f$
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
                        const T a_value = batchlas::portable::select_from_group(sg, a_lane, static_cast<uint32_t>(row_offset));
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
                        const T b_value = batchlas::portable::select_from_group(sg, b_lane, static_cast<uint32_t>(row_offset));
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

/// @brief Register-tiled triangular matrix-matrix product staging \f$K\f$-tiles of both factors through local memory.
/// @param[in]     item       `sycl::nd_item<3>` of the calling work-item; every work-item of the work-group must call
/// @param[in]     a          single triangular matrix \f$A\f$; only the `transform.uplo` triangle is read (not its diagonal when `transform.diag` is `Unit`)
/// @param[in,out] operand    `B` (read; must not overlap `C`), `C` (read when scaled by \f$\beta\f$,
///                           overwritten with the result), \f$\alpha\f$, \f$\beta\f$
/// @param[in]     transform  side, stored triangle, \f$\mathrm{op}(\cdot)\f$ and diagonal kind of \f$A\f$
/// @param[out]    workspace  local-memory staging buffers; contents undefined on return
template <typename Item, typename T>
inline constexpr void trmm_register_tiled(const Item& item,
                                          const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                          MatrixMatrixOperand<T> operand,
                                          TriangularTransform transform,
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

/// @brief Local memory, in elements of `T`, that dispatch_trmm() can use for `Tag` and `Policy`.
/// @param[in] launch      description of the launch the call runs in
/// @param[in] row_extent  \f$m\f$, rows of `C`; \f$m \ge 0\f$
/// @param[in] col_extent  \f$n\f$, columns of `C`; \f$n \ge 0\f$
/// @param[in] aliased     `B` and `C` overlap; no staged path then runs, so this returns 0
/// @return element count of the register-tiled staging buffers, or 0 when that path cannot run
template <typename Tag, DeviceBlasPolicy Policy, typename T>
inline constexpr std::size_t trmm_workspace_elements(const DeviceBlasLaunchInfo& launch,
                                                     int row_extent,
                                                     int col_extent,
                                                     bool aliased = false) {
    assert(row_extent >= 0 && col_extent >= 0 && "device::trmm workspace query expects non-negative matrix extents");
    if (aliased || !detail::subgroup::is_nd_item_3d_launch(launch)) {
        return 0;
    }

    const int contract_extent = Tag::side == Side::Left ? row_extent : col_extent;
    if (detail::subgroup::can_use_matrix_register_fast_path<T>(launch, row_extent, col_extent, contract_extent, Policy)) {
        return detail::workspace_elements_v<T, detail::subgroup::RegisterMatrixWorkspace<T>>;
    }
    return 0;
}

/// @brief Validates the operands of a triangular matrix-matrix product, then picks the register-tiled, sub-group or generic path.
///
/// Overlapping `B` and `C` always take the generic path.
/// @param[in]     exec       executor (`sycl::group`, `sycl::sub_group` or an `nd_item`); every work-item must call
/// @param[in]     a          single triangular matrix \f$A\f$; only the `Tag::uplo` triangle is read (not its diagonal when `Tag::diag` is `Unit`)
/// @param[in,out] operand    `B` (read; may be the same view as `C`), `C` (read when scaled by \f$\beta\f$,
///                           overwritten with the result), \f$\alpha\f$, \f$\beta\f$
/// @param[out]    workspace  local-memory scratch of trmm_workspace_elements() elements, or `nullptr`;
///                           contents undefined on return
template <typename Tag, DeviceBlasPolicy Policy, typename Exec, typename T>
inline constexpr void dispatch_trmm(const Exec& exec,
                                    const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                    MatrixMatrixOperand<T> operand,
                                    T* workspace = nullptr) {
    const TriangularTransform transform{.side = Tag::side, .uplo = Tag::uplo, .trans = Tag::trans, .diag = Tag::diag};
    validate_triangular_operand(a, operand, transform);
    const bool aliased = detail::views_overlap(operand.b, operand.c);

    if constexpr (detail::NdItemLike<Exec>) {
        const int row_extent = operand.c.rows();
        const int col_extent = operand.c.cols();
        const int contract_extent = transform.side == Side::Left ? operand.b.rows() : operand.b.cols();
        auto* register_workspace = workspace == nullptr ? nullptr : detail::workspace_ptr_cast<detail::subgroup::RegisterMatrixWorkspace<T>>(workspace);
        if (!aliased && register_workspace != nullptr &&
            detail::subgroup::can_use_matrix_register_fast_path<T>(exec, row_extent, col_extent, contract_extent, Policy) &&
            std::same_as<std::remove_cvref_t<Exec>, sycl::nd_item<3>>) {
            detail::subgroup::trmm_register_tiled(exec, a, operand, transform, register_workspace);
            return;
        }
        if (!aliased && detail::subgroup::can_use_matrix_fast_path<T>(exec, row_extent, col_extent, contract_extent, Policy)) {
            detail::subgroup::trmm(exec, a, operand, transform);
            return;
        }
    }

    generic::trmm<Tag>(exec, a, operand);
}

} // namespace detail

/// @addtogroup api_device_blas
/// @{

/// @brief Triangular matrix-matrix product \f$ C := \alpha\,\mathrm{op}(A)\,B + \beta\,C \f$ (Left) or \f$ C := \alpha\,B\,\mathrm{op}(A) + \beta\,C \f$ (Right).
///
/// Unlike BLAS `trmm` it has a separate output and a `beta`. `B` and `C` may be
/// the same view (in place): the generic path orders its traversal so no entry is
/// overwritten before it is read, and overlap disables both fast paths. Partial
/// overlap is not supported. The fast paths
/// (`float`, an `nd_item` executor; the register-tiled one also an
/// `nd_item<3>`, 256 work-items and a workspace) are otherwise as for gemm().
/// @tparam SideV   side `A` multiplies from
/// @tparam UploV   stored triangle of `A`
/// @tparam TransV  \f$\mathrm{op}(\cdot)\f$ applied to `A`
/// @tparam DiagV   `Unit`: the diagonal is taken as 1
/// @tparam Group   `sycl::group`, `sycl::sub_group` or an `nd_item`; every work-item must call
/// @param[in] group          executor. With `nd_item<3>` and no register-tiled path, every work-group computes all of `C` (no tile-group guard, unlike gemm()), so launch a single group in dimensions 1 and 2.
/// @param[in] a              single triangular matrix, \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right) for an \f$m \times n\f$ `B`; read only
/// @param[in,out] operand    `B` (\f$m \times n\f$, read; may be the same view as `C` for an in-place product, no other overlap), `C` (same shape as `B`, read and overwritten with the result), `alpha`, `beta`
/// @param[out] workspace     local-memory scratch of trmm_workspace_elements() elements (contents undefined on return), or `nullptr`
/// @pre `C` holds finite values even when \f$\beta = 0\f$.
/// @note On the generic path each `C` element is written by the group leader; barrier before other work-items read `C`.
template <Side SideV = Side::Left,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          Diag DiagV = Diag::NonUnit,
          typename Group,
          typename T>
inline constexpr void trmm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           T* workspace = nullptr) {
    detail::dispatch_trmm<detail::TriangularTransformTag<SideV, UploV, TransV, DiagV>, DeviceBlasPolicy::Auto>(group, a, operand, workspace);
}

/// @brief As trmm(), with an explicit DeviceBlasPolicy.
/// @tparam Policy  `Auto` (any eligible path), `Generic` (only the group-reduction path) or `Subgroup16`/`Subgroup32` (the sub-group kernel only at that sub-group size; the tiled kernels ignore the size)
/// @param[in] group          executor; every work-item must call
/// @param[in] a              single triangular matrix, \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right) for an \f$m \times n\f$ `B`; read only
/// @param[in,out] operand    `B` (\f$m \times n\f$, read; may be the same view as `C` for an in-place product, no other overlap), `C` (same shape as `B`, read and overwritten with the result), `alpha`, `beta`
/// @param[out] workspace     local-memory scratch of trmm_workspace_elements() elements (contents undefined on return), or `nullptr`
template <DeviceBlasPolicy Policy,
          Side SideV = Side::Left,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          Diag DiagV = Diag::NonUnit,
          typename Group,
          typename T>
inline constexpr void trmm(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           MatrixMatrixOperand<T> operand,
                           T* workspace = nullptr) {
    detail::dispatch_trmm<detail::TriangularTransformTag<SideV, UploV, TransV, DiagV>, Policy>(group, a, operand, workspace);
}

/// @brief Local memory, in elements of `T`, that trmm() can use for this launch and shape.
///
/// Nonzero only for an `nd_item<3>` launch description whose shape admits the
/// register-tiled path. When the result is 0 pass `nullptr`.
/// @trap `T` is the LAST template parameter here, after the four mode parameters,
///       unlike every other `*_workspace_elements` and unlike the policy overload.
/// @param[in] launch      description of the launch the call runs in
/// @param[in] row_extent  \f$m\f$, rows of `C`; \f$m \ge 0\f$
/// @param[in] col_extent  \f$n\f$, columns of `C`; \f$n \ge 0\f$
/// @param[in] aliased     `B` and `C` overlap; the call then takes no staged path, so this returns 0
/// @return element count, or 0 when no staged path applies
/// @ingroup api_device_blas_lowlevel
template <Side SideV = Side::Left,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          Diag DiagV = Diag::NonUnit,
          typename T>
inline constexpr std::size_t trmm_workspace_elements(const DeviceBlasLaunchInfo& launch,
                                                     int row_extent,
                                                     int col_extent,
                                                     bool aliased = false) {
    return detail::trmm_workspace_elements<detail::TriangularTransformTag<SideV, UploV, TransV, DiagV>, DeviceBlasPolicy::Auto, T>(launch, row_extent, col_extent, aliased);
}

/// @brief Local memory, in elements of `T`, that trmm() can use under `Policy`; 0 means pass `nullptr`.
/// @param[in] launch      description of the launch the call runs in
/// @param[in] row_extent  \f$m\f$, rows of `C`; \f$m \ge 0\f$
/// @param[in] col_extent  \f$n\f$, columns of `C`; \f$n \ge 0\f$
/// @param[in] aliased     `B` and `C` overlap; the call then takes no staged path, so this returns 0
/// @return element count, or 0 when no staged path applies
/// @ingroup api_device_blas_lowlevel
template <typename T,
          DeviceBlasPolicy Policy,
          Side SideV = Side::Left,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          Diag DiagV = Diag::NonUnit>
inline constexpr std::size_t trmm_workspace_elements(const DeviceBlasLaunchInfo& launch,
                                                     int row_extent,
                                                     int col_extent,
                                                     bool aliased = false) {
    return detail::trmm_workspace_elements<detail::TriangularTransformTag<SideV, UploV, TransV, DiagV>, Policy, T>(launch, row_extent, col_extent, aliased);
}

/// @brief As trmm(), with the operands passed separately instead of in an operand struct.
/// @param[in] group          executor; every work-item must call
/// @param[in] a              single triangular matrix, \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right); read only
/// @param[in] b              \f$m \times n\f$ matrix, read; it may be the same view as `c` (in-place, overwritten), but no other overlap is supported
/// @param[in,out] c          \f$m \times n\f$ matrix, read (scaled by \f$\beta\f$) and overwritten with the result
/// @param[in] alpha          scalar \f$\alpha\f$ applied to the product
/// @param[in] beta           scalar \f$\beta\f$ applied to the input `C`; 0 still reads `C`
/// @param[out] workspace     local-memory scratch of trmm_workspace_elements() elements (contents undefined on return), or `nullptr`
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
                           T beta = T(0),
                           T* workspace = nullptr) {
    trmm<SideV, UploV, TransV, DiagV>(group, a, make_matmat_operand(b, c, alpha, beta), workspace);
}

/// @brief As trmm(), with an explicit DeviceBlasPolicy and the operands passed separately.
/// @tparam Policy  `Auto` (any eligible path), `Generic` (only the group-reduction path) or `Subgroup16`/`Subgroup32` (the sub-group kernel only at that sub-group size; the tiled kernels ignore the size)
/// @param[in] group          executor; every work-item must call
/// @param[in] a              single triangular matrix, \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right); read only
/// @param[in] b              \f$m \times n\f$ matrix, read; it may be the same view as `c` (in-place, overwritten), but no other overlap is supported
/// @param[in,out] c          \f$m \times n\f$ matrix, read (scaled by \f$\beta\f$) and overwritten with the result
/// @param[in] alpha          scalar \f$\alpha\f$ applied to the product
/// @param[in] beta           scalar \f$\beta\f$ applied to the input `C`; 0 still reads `C`
/// @param[out] workspace     local-memory scratch of trmm_workspace_elements() elements (contents undefined on return), or `nullptr`
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
                           T beta = T(0),
                           T* workspace = nullptr) {
    trmm<Policy, SideV, UploV, TransV, DiagV>(group, a, make_matmat_operand(b, c, alpha, beta), workspace);
}

/// @}

} // namespace batchlas::device
