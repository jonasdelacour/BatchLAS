#pragma once

#include <blas/device/detail/group_blas_common.hh>
#include <blas/device/detail/group_blas_subgroup_common.hh>

namespace batchlas::device {

namespace detail::generic {

template <typename Tag, typename Group, typename T>
inline constexpr void rank2k(const Group& group,
                             const KernelMatrixView<T, MatrixFormat::Dense>& a,
                             MatrixMatrixOperand<T> operand) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int extent = detail::output_size(a, Tag::trans);
    const int contract_extent = detail::input_size(a, Tag::trans);
    constexpr Transpose rhs_transform = detail::rank2k_rhs_transform<Tag>();
    const T alpha2 = detail::secondary_rank2k_alpha<Tag>(operand.alpha);

    for (int linear_index = local_id; linear_index < extent * extent; linear_index += local_size) {
        const int row = linear_index % extent;
        const int col = linear_index / extent;
        if (!detail::triangular_storage_contains<Tag>(row, col)) {
            continue;
        }

        T partial{};
        for (int k = 0; k < contract_extent; ++k) {
            const T lhs1 = detail::matrix_entry(a, row, k, Tag::trans);
            const T rhs1 = detail::matrix_entry(operand.b, k, col, rhs_transform);
            const T lhs2 = detail::matrix_entry(operand.b, row, k, Tag::trans);
            const T rhs2 = detail::matrix_entry(a, k, col, rhs_transform);
            partial += operand.alpha * lhs1 * rhs1 + alpha2 * lhs2 * rhs2;
        }

        T value = operand.beta * operand.c(row, col) + partial;
        if constexpr (ComplexScalar<T>) {
            if constexpr (Tag::hermitian) {
                if (row == col) {
                    value = T(value.real(), typename T::value_type(0));
                }
            }
        }
        operand.c(row, col) = value;
    }
}

template <typename Tag, typename Group, typename T>
inline constexpr void rankk(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            RankKOperand<T> operand) {
    const int local_id = detail::group_local_linear_id(group);
    const int local_size = detail::group_local_linear_range(group);
    const int extent = detail::output_size(a, Tag::trans);
    const int contract_extent = detail::input_size(a, Tag::trans);
    constexpr Transpose rhs_transform = detail::rankk_rhs_transform<Tag>();

    for (int linear_index = local_id; linear_index < extent * extent; linear_index += local_size) {
        const int row = linear_index % extent;
        const int col = linear_index / extent;
        if (!detail::triangular_storage_contains<Tag>(row, col)) {
            continue;
        }

        T partial{};
        for (int k = 0; k < contract_extent; ++k) {
            partial += operand.alpha * detail::matrix_entry(a, row, k, Tag::trans) *
                detail::matrix_entry(a, k, col, rhs_transform);
        }

        T value = operand.beta * operand.c(row, col) + partial;
        if constexpr (ComplexScalar<T>) {
            if constexpr (Tag::hermitian) {
                if (row == col) {
                    value = T(value.real(), typename T::value_type(0));
                }
            }
        }
        operand.c(row, col) = value;
    }
}

} // namespace detail::generic

namespace detail::subgroup {

template <typename Item, typename T>
inline constexpr void rank2k_complex_tiled(const Item& item,
                                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                           MatrixMatrixOperand<T> operand,
                                           SymmetricRank2kTransform transform,
                                           ComplexRank2kWorkspace<T>* workspace) {
    const int linear_tid = detail::item_local_linear_id(item);
    const int subgroup_id = subgroup_group_id(item);
    const int local_row = linear_tid / kComplexRank2kLocalCols;
    const int local_col = linear_tid % kComplexRank2kLocalCols;
    const int extent = detail::output_size(a, transform.trans);
    const int contract_extent = detail::input_size(a, transform.trans);
    const int row_tiles = (extent + kComplexRank2kTileM - 1) / kComplexRank2kTileM;
    const int col_tiles = (extent + kComplexRank2kTileN - 1) / kComplexRank2kTileN;
    const int tile_row_start = matrix_tile_group_row(item);
    const int tile_col_start = matrix_tile_group_col(item);
    const int tile_row_stride = matrix_tile_group_row_stride(item);
    const int tile_col_stride = matrix_tile_group_col_stride(item);
    const Transpose rhs_transform = detail::rank2k_rhs_transform(transform);
    const T alpha2 = detail::secondary_rank2k_alpha(operand.alpha, transform.hermitian);

    for (int tile_row = tile_row_start; tile_row < row_tiles; tile_row += tile_row_stride) {
        for (int tile_col = tile_col_start; tile_col < col_tiles; tile_col += tile_col_stride) {
            const int tile_row_base = tile_row * kComplexRank2kTileM;
            const int tile_col_base = tile_col * kComplexRank2kTileN;
            const int row_base = tile_row_base + local_row * kComplexRank2kThreadTileRows;
            const int col_base = tile_col_base + local_col * kComplexRank2kThreadTileCols;

            if (transform.uplo == Uplo::Lower && tile_row_base + kComplexRank2kTileM <= tile_col_base) {
                continue;
            }
            if (transform.uplo == Uplo::Upper && tile_col_base + kComplexRank2kTileN <= tile_row_base) {
                continue;
            }

            ComplexRank2kAccumTile<T> accum;

            auto lhs1_loader = [&](int tile_r, int global_k) {
                const int global_row = tile_row_base + tile_r;
                if (global_row >= extent) {
                    return T(0);
                }
                return operand.alpha * detail::matrix_entry(a, global_row, global_k, transform.trans);
            };
            auto rhs1_loader = [&](int tile_c, int global_k) {
                const int global_col = tile_col_base + tile_c;
                if (global_col >= extent) {
                    return T(0);
                }
                return detail::matrix_entry(operand.b, global_k, global_col, rhs_transform);
            };
            auto lhs2_loader = [&](int tile_r, int global_k) {
                const int global_row = tile_row_base + tile_r;
                if (global_row >= extent) {
                    return T(0);
                }
                return alpha2 * detail::matrix_entry(operand.b, global_row, global_k, transform.trans);
            };
            auto rhs2_loader = [&](int tile_c, int global_k) {
                const int global_col = tile_col_base + tile_c;
                if (global_col >= extent) {
                    return T(0);
                }
                return detail::matrix_entry(a, global_k, global_col, rhs_transform);
            };

            accumulate_complex_rank2k_tiled_pass_impl<kComplexRank2kTileK, kComplexRank2kTileAStride>(
                item, workspace, linear_tid, subgroup_id, local_row, local_col, contract_extent, lhs1_loader, rhs1_loader, accum);
            accumulate_complex_rank2k_tiled_pass_impl<kComplexRank2kTileK, kComplexRank2kTileAStride>(
                item, workspace, linear_tid, subgroup_id, local_row, local_col, contract_extent, lhs2_loader, rhs2_loader, accum);

            write_complex_rank2k_tile(operand, transform, row_base, col_base, extent, accum);
        }
    }
}

template <typename Item, typename T>
inline constexpr void rankk_complex_tiled(const Item& item,
                                          const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                          RankKOperand<T> operand,
                                          SymmetricRankKTransform transform,
                                          ComplexRank2kWorkspace<T>* workspace) {
    const int linear_tid = detail::item_local_linear_id(item);
    const int subgroup_id = subgroup_group_id(item);
    const int local_row = linear_tid / kComplexRank2kLocalCols;
    const int local_col = linear_tid % kComplexRank2kLocalCols;
    const int extent = detail::output_size(a, transform.trans);
    const int contract_extent = detail::input_size(a, transform.trans);
    const int row_tiles = (extent + kComplexRank2kTileM - 1) / kComplexRank2kTileM;
    const int col_tiles = (extent + kComplexRank2kTileN - 1) / kComplexRank2kTileN;
    const int tile_row_start = matrix_tile_group_row(item);
    const int tile_col_start = matrix_tile_group_col(item);
    const int tile_row_stride = matrix_tile_group_row_stride(item);
    const int tile_col_stride = matrix_tile_group_col_stride(item);
    const Transpose rhs_transform = detail::rankk_rhs_transform(transform);
    MatrixMatrixOperand<T> rank_operand{a, operand.c, operand.alpha, operand.beta};

    for (int tile_row = tile_row_start; tile_row < row_tiles; tile_row += tile_row_stride) {
        for (int tile_col = tile_col_start; tile_col < col_tiles; tile_col += tile_col_stride) {
            const int tile_row_base = tile_row * kComplexRank2kTileM;
            const int tile_col_base = tile_col * kComplexRank2kTileN;
            const int row_base = tile_row_base + local_row * kComplexRank2kThreadTileRows;
            const int col_base = tile_col_base + local_col * kComplexRank2kThreadTileCols;

            if (transform.uplo == Uplo::Lower && tile_row_base + kComplexRank2kTileM <= tile_col_base) {
                continue;
            }
            if (transform.uplo == Uplo::Upper && tile_col_base + kComplexRank2kTileN <= tile_row_base) {
                continue;
            }

            ComplexRank2kAccumTile<T> accum;

            auto lhs_loader = [&](int tile_r, int global_k) {
                const int global_row = tile_row_base + tile_r;
                if (global_row >= extent) {
                    return T(0);
                }
                return operand.alpha * detail::matrix_entry(a, global_row, global_k, transform.trans);
            };
            auto rhs_loader = [&](int tile_c, int global_k) {
                const int global_col = tile_col_base + tile_c;
                if (global_col >= extent) {
                    return T(0);
                }
                return detail::matrix_entry(a, global_k, global_col, rhs_transform);
            };

            accumulate_complex_rank2k_tiled_pass_impl<kComplexRank2kTileK, kComplexRank2kTileAStride>(
                item, workspace, linear_tid, subgroup_id, local_row, local_col, contract_extent, lhs_loader, rhs_loader, accum);

            write_complex_rank2k_tile(rank_operand, transform, row_base, col_base, extent, accum);
        }
    }
}

template <typename Item, typename T>
inline constexpr void rank2k_register_tiled(const Item& item,
                                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                            MatrixMatrixOperand<T> operand,
                                            SymmetricRank2kTransform transform,
                                            RegisterMatrixWorkspace<T>* workspace) {
    const int linear_tid = detail::item_local_linear_id(item);
    const int subgroup_id = subgroup_group_id(item);
    const int local_row = linear_tid / kRegisterMatrixLocalCols;
    const int local_col = linear_tid % kRegisterMatrixLocalCols;
    const int extent = detail::output_size(a, transform.trans);
    const int contract_extent = detail::input_size(a, transform.trans);
    const int row_tiles = (extent + kRegisterMatrixTileM - 1) / kRegisterMatrixTileM;
    const int col_tiles = (extent + kRegisterMatrixTileN - 1) / kRegisterMatrixTileN;
    const int tile_row_start = matrix_tile_group_row(item);
    const int tile_col_start = matrix_tile_group_col(item);
    const int tile_row_stride = matrix_tile_group_row_stride(item);
    const int tile_col_stride = matrix_tile_group_col_stride(item);
    const Transpose rhs_transform = detail::rank2k_rhs_transform(transform);
    const T alpha2 = detail::secondary_rank2k_alpha(operand.alpha, transform.hermitian);

    for (int tile_row = tile_row_start; tile_row < row_tiles; tile_row += tile_row_stride) {
        for (int tile_col = tile_col_start; tile_col < col_tiles; tile_col += tile_col_stride) {
            const int tile_row_base = tile_row * kRegisterMatrixTileM;
            const int tile_col_base = tile_col * kRegisterMatrixTileN;
            const int row_base = tile_row * kRegisterMatrixTileM + local_row * kRegisterMatrixThreadTileRows;
            const int col_base = tile_col * kRegisterMatrixTileN + local_col * kRegisterMatrixThreadTileCols;

            if (transform.uplo == Uplo::Lower && tile_row_base + kRegisterMatrixTileM <= tile_col_base) {
                continue;
            }
            if (transform.uplo == Uplo::Upper && tile_col_base + kRegisterMatrixTileN <= tile_row_base) {
                continue;
            }

            RegisterMatrixAccumTile<T> accum;

            auto lhs1_loader = [&](int tile_r, int global_k) {
                const int global_row = tile_row * kRegisterMatrixTileM + tile_r;
                if (global_row >= extent) {
                    return T(0);
                }
                return operand.alpha * detail::matrix_entry(a, global_row, global_k, transform.trans);
            };
            auto rhs1_loader = [&](int tile_c, int global_k) {
                const int global_col = tile_col * kRegisterMatrixTileN + tile_c;
                if (global_col >= extent) {
                    return T(0);
                }
                return detail::matrix_entry(operand.b, global_k, global_col, rhs_transform);
            };
            auto lhs2_loader = [&](int tile_r, int global_k) {
                const int global_row = tile_row * kRegisterMatrixTileM + tile_r;
                if (global_row >= extent) {
                    return T(0);
                }
                return alpha2 * detail::matrix_entry(operand.b, global_row, global_k, transform.trans);
            };
            auto rhs2_loader = [&](int tile_c, int global_k) {
                const int global_col = tile_col * kRegisterMatrixTileN + tile_c;
                if (global_col >= extent) {
                    return T(0);
                }
                return detail::matrix_entry(a, global_k, global_col, rhs_transform);
            };

            accumulate_rank2k_register_tiled_pass(item, workspace, linear_tid, subgroup_id, contract_extent, lhs1_loader, rhs1_loader, accum);
            accumulate_rank2k_register_tiled_pass(item, workspace, linear_tid, subgroup_id, contract_extent, lhs2_loader, rhs2_loader, accum);

            write_rank2k_register_tile(operand, transform, row_base, col_base, extent, accum);
        }
    }
}

template <typename Item, typename T>
inline constexpr void rankk_register_tiled(const Item& item,
                                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                           RankKOperand<T> operand,
                                           SymmetricRankKTransform transform,
                                           RegisterMatrixWorkspace<T>* workspace) {
    const int linear_tid = detail::item_local_linear_id(item);
    const int subgroup_id = subgroup_group_id(item);
    const int local_row = linear_tid / kRegisterMatrixLocalCols;
    const int local_col = linear_tid % kRegisterMatrixLocalCols;
    const int extent = detail::output_size(a, transform.trans);
    const int contract_extent = detail::input_size(a, transform.trans);
    const int row_tiles = (extent + kRegisterMatrixTileM - 1) / kRegisterMatrixTileM;
    const int col_tiles = (extent + kRegisterMatrixTileN - 1) / kRegisterMatrixTileN;
    const int tile_row_start = matrix_tile_group_row(item);
    const int tile_col_start = matrix_tile_group_col(item);
    const int tile_row_stride = matrix_tile_group_row_stride(item);
    const int tile_col_stride = matrix_tile_group_col_stride(item);
    const Transpose rhs_transform = detail::rankk_rhs_transform(transform);
    MatrixMatrixOperand<T> rank_operand{a, operand.c, operand.alpha, operand.beta};

    for (int tile_row = tile_row_start; tile_row < row_tiles; tile_row += tile_row_stride) {
        for (int tile_col = tile_col_start; tile_col < col_tiles; tile_col += tile_col_stride) {
            const int tile_row_base = tile_row * kRegisterMatrixTileM;
            const int tile_col_base = tile_col * kRegisterMatrixTileN;
            const int row_base = tile_row * kRegisterMatrixTileM + local_row * kRegisterMatrixThreadTileRows;
            const int col_base = tile_col * kRegisterMatrixTileN + local_col * kRegisterMatrixThreadTileCols;

            if (transform.uplo == Uplo::Lower && tile_row_base + kRegisterMatrixTileM <= tile_col_base) {
                continue;
            }
            if (transform.uplo == Uplo::Upper && tile_col_base + kRegisterMatrixTileN <= tile_row_base) {
                continue;
            }

            RegisterMatrixAccumTile<T> accum;

            auto lhs_loader = [&](int tile_r, int global_k) {
                const int global_row = tile_row * kRegisterMatrixTileM + tile_r;
                if (global_row >= extent) {
                    return T(0);
                }
                return operand.alpha * detail::matrix_entry(a, global_row, global_k, transform.trans);
            };
            auto rhs_loader = [&](int tile_c, int global_k) {
                const int global_col = tile_col * kRegisterMatrixTileN + tile_c;
                if (global_col >= extent) {
                    return T(0);
                }
                return detail::matrix_entry(a, global_k, global_col, rhs_transform);
            };

            accumulate_rank2k_register_tiled_pass(item, workspace, linear_tid, subgroup_id, contract_extent, lhs_loader, rhs_loader, accum);

            write_rank2k_register_tile(rank_operand, transform, row_base, col_base, extent, accum);
        }
    }
}

} // namespace detail::subgroup

namespace detail {

template <typename Tag, DeviceBlasPolicy Policy, typename Exec, typename T>
inline constexpr void dispatch_rank2k(const Exec& exec,
                                      const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                      MatrixMatrixOperand<T> operand) {
    const SymmetricRank2kTransform transform{.uplo = Tag::uplo, .trans = Tag::trans, .hermitian = Tag::hermitian};
    validate_rank2k_operand(a, operand, transform);

    if constexpr (detail::NdItemLike<Exec>) {
        if (detail::subgroup::can_use_complex_rank2k_tiled_fast_path<T>(exec, a, operand, transform, Policy) &&
            detail::subgroup::complex_rank2k_workspace_supported_v<T>) {
            auto* workspace = sycl::ext::oneapi::group_local_memory_for_overwrite<detail::subgroup::ComplexRank2kWorkspace<T>>(exec.get_group()).get();
            detail::subgroup::rank2k_complex_tiled(exec, a, operand, transform, workspace);
            return;
        }
        if (detail::subgroup::can_use_rank2k_register_fast_path<T>(exec, a, operand, transform, Policy) &&
            detail::subgroup::register_matrix_workspace_supported_v<T>) {
            auto* workspace = sycl::ext::oneapi::group_local_memory_for_overwrite<detail::subgroup::RegisterMatrixWorkspace<T>>(exec.get_group()).get();
            detail::subgroup::rank2k_register_tiled(exec, a, operand, transform, workspace);
            return;
        }
    }

    generic::rank2k<Tag>(exec, a, operand);
}

template <typename Tag, DeviceBlasPolicy Policy, typename Exec, typename T>
inline constexpr void dispatch_rankk(const Exec& exec,
                                     const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                     RankKOperand<T> operand) {
    const SymmetricRankKTransform transform{.uplo = Tag::uplo, .trans = Tag::trans, .hermitian = Tag::hermitian};
    validate_rankk_operand(a, operand, transform);

    if constexpr (detail::NdItemLike<Exec>) {
        if (detail::subgroup::can_use_complex_rankk_tiled_fast_path<T>(exec, a, operand, transform, Policy) &&
            detail::subgroup::complex_rank2k_workspace_supported_v<T>) {
            auto* workspace = sycl::ext::oneapi::group_local_memory_for_overwrite<detail::subgroup::ComplexRank2kWorkspace<T>>(exec.get_group()).get();
            detail::subgroup::rankk_complex_tiled(exec, a, operand, transform, workspace);
            return;
        }
        if (detail::subgroup::can_use_rankk_register_fast_path<T>(exec, a, operand, transform, Policy) &&
            detail::subgroup::register_matrix_workspace_supported_v<T>) {
            auto* workspace = sycl::ext::oneapi::group_local_memory_for_overwrite<detail::subgroup::RegisterMatrixWorkspace<T>>(exec.get_group()).get();
            detail::subgroup::rankk_register_tiled(exec, a, operand, transform, workspace);
            return;
        }
    }

    generic::rankk<Tag>(exec, a, operand);
}

} // namespace detail

template <Uplo UploV = Uplo::Upper, Transpose TransV = Transpose::NoTrans, typename Group, typename T>
inline constexpr void syrk(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           RankKOperand<T> operand) {
    detail::dispatch_rankk<detail::SymmetricRankTransformTag<UploV, TransV, false>, DeviceBlasPolicy::Auto>(group, a, operand);
}

template <DeviceBlasPolicy Policy,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          typename Group,
          typename T>
inline constexpr void syrk(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           RankKOperand<T> operand) {
    detail::dispatch_rankk<detail::SymmetricRankTransformTag<UploV, TransV, false>, Policy>(group, a, operand);
}

template <Uplo UploV = Uplo::Upper, Transpose TransV = Transpose::NoTrans, typename Group, typename T>
inline constexpr void syrk(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0)) {
    syrk<UploV, TransV>(group, a, make_rankk_operand(c, alpha, beta));
}

template <DeviceBlasPolicy Policy,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          typename Group,
          typename T>
inline constexpr void syrk(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0)) {
    (void)Policy;
    syrk<UploV, TransV>(group, a, c, alpha, beta);
}

template <Uplo UploV = Uplo::Upper, Transpose TransV = Transpose::NoTrans, typename Group, typename T>
inline constexpr void herk(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           RankKOperand<T> operand) {
    detail::dispatch_rankk<detail::SymmetricRankTransformTag<UploV, TransV, ComplexScalar<T>>, DeviceBlasPolicy::Auto>(group, a, operand);
}

template <DeviceBlasPolicy Policy,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          typename Group,
          typename T>
inline constexpr void herk(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           RankKOperand<T> operand) {
    detail::dispatch_rankk<detail::SymmetricRankTransformTag<UploV, TransV, ComplexScalar<T>>, Policy>(group, a, operand);
}

template <Uplo UploV = Uplo::Upper, Transpose TransV = Transpose::NoTrans, typename Group, typename T>
inline constexpr void herk(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0)) {
    herk<UploV, TransV>(group, a, make_rankk_operand(c, alpha, beta));
}

template <DeviceBlasPolicy Policy,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          typename Group,
          typename T>
inline constexpr void herk(const Group& group,
                           const KernelMatrixView<T, MatrixFormat::Dense>& a,
                           const KernelMatrixView<T, MatrixFormat::Dense>& c,
                           T alpha = T(1),
                           T beta = T(0)) {
    (void)Policy;
    herk<UploV, TransV>(group, a, c, alpha, beta);
}

template <Uplo UploV = Uplo::Upper, Transpose TransV = Transpose::NoTrans, typename Group, typename T>
inline constexpr void syr2k(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            MatrixMatrixOperand<T> operand) {
    detail::dispatch_rank2k<detail::SymmetricRankTransformTag<UploV, TransV, false>, DeviceBlasPolicy::Auto>(group, a, operand);
}

template <DeviceBlasPolicy Policy,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          typename Group,
          typename T>
inline constexpr void syr2k(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            MatrixMatrixOperand<T> operand) {
    detail::dispatch_rank2k<detail::SymmetricRankTransformTag<UploV, TransV, false>, Policy>(group, a, operand);
}

template <Uplo UploV = Uplo::Upper, Transpose TransV = Transpose::NoTrans, typename Group, typename T>
inline constexpr void syr2k(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            const KernelMatrixView<T, MatrixFormat::Dense>& b,
                            const KernelMatrixView<T, MatrixFormat::Dense>& c,
                            T alpha = T(1),
                            T beta = T(0)) {
    syr2k<UploV, TransV>(group, a, make_matmat_operand(b, c, alpha, beta));
}

template <DeviceBlasPolicy Policy,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          typename Group,
          typename T>
inline constexpr void syr2k(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            const KernelMatrixView<T, MatrixFormat::Dense>& b,
                            const KernelMatrixView<T, MatrixFormat::Dense>& c,
                            T alpha = T(1),
                            T beta = T(0)) {
    (void)Policy;
    syr2k<UploV, TransV>(group, a, b, c, alpha, beta);
}

template <Uplo UploV = Uplo::Upper, Transpose TransV = Transpose::NoTrans, typename Group, typename T>
inline constexpr void her2k(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            MatrixMatrixOperand<T> operand) {
    detail::dispatch_rank2k<detail::SymmetricRankTransformTag<UploV, TransV, ComplexScalar<T>>, DeviceBlasPolicy::Auto>(group, a, operand);
}

template <DeviceBlasPolicy Policy,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          typename Group,
          typename T>
inline constexpr void her2k(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            MatrixMatrixOperand<T> operand) {
    detail::dispatch_rank2k<detail::SymmetricRankTransformTag<UploV, TransV, ComplexScalar<T>>, Policy>(group, a, operand);
}

template <Uplo UploV = Uplo::Upper, Transpose TransV = Transpose::NoTrans, typename Group, typename T>
inline constexpr void her2k(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            const KernelMatrixView<T, MatrixFormat::Dense>& b,
                            const KernelMatrixView<T, MatrixFormat::Dense>& c,
                            T alpha = T(1),
                            T beta = T(0)) {
    her2k<UploV, TransV>(group, a, make_matmat_operand(b, c, alpha, beta));
}

template <DeviceBlasPolicy Policy,
          Uplo UploV = Uplo::Upper,
          Transpose TransV = Transpose::NoTrans,
          typename Group,
          typename T>
inline constexpr void her2k(const Group& group,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            const KernelMatrixView<T, MatrixFormat::Dense>& b,
                            const KernelMatrixView<T, MatrixFormat::Dense>& c,
                            T alpha = T(1),
                            T beta = T(0)) {
    (void)Policy;
    her2k<UploV, TransV>(group, a, b, c, alpha, beta);
}

} // namespace batchlas::device
