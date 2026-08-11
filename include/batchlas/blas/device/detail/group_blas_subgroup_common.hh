#pragma once

#include <cstdint>

#include <batchlas/blas/device/detail/group_blas_common.hh>

namespace batchlas::device {

namespace detail::subgroup {

inline constexpr int kMaxSupportedSubgroupSize = 32;
inline constexpr int kMaxSubgroupsPerWorkGroup = 8;
inline constexpr int kVectorTileK = 64;
inline constexpr int kMaxVectorRowsPerSubgroup = 4;
inline constexpr int kMaxMatrixRowsPerSubgroup = 8;
inline constexpr int kRegisterMatrixTileM = 128;
inline constexpr int kRegisterMatrixTileN = 64;
inline constexpr int kRegisterMatrixTileK = 32;
inline constexpr int kRegisterMatrixThreadTileRows = 4;
inline constexpr int kRegisterMatrixThreadTileCols = 8;
inline constexpr int kRegisterMatrixLocalRows = kRegisterMatrixTileM / kRegisterMatrixThreadTileRows;
inline constexpr int kRegisterMatrixLocalCols = kRegisterMatrixTileN / kRegisterMatrixThreadTileCols;
inline constexpr int kRegisterMatrixThreadsPerGroup = kRegisterMatrixLocalRows * kRegisterMatrixLocalCols;
inline constexpr int kRegisterMatrixSubgroupRows = kMaxSupportedSubgroupSize / kRegisterMatrixLocalCols;
inline constexpr int kRegisterMatrixSubgroupTileM = kRegisterMatrixSubgroupRows * kRegisterMatrixThreadTileRows;
inline constexpr int kRegisterMatrixTileAStride = kRegisterMatrixTileK + 1;
inline constexpr int kRegisterMatrixTileBStride = kRegisterMatrixTileN + 1;
inline constexpr int kRegisterMatrixLhsStages = 1;
inline constexpr int kRegisterMatrixRhsStages = 1;
inline constexpr int kComplexRank2kTileM = 64;
inline constexpr int kComplexRank2kTileN = 64;
inline constexpr int kComplexRank2kTileK = 32;
inline constexpr int kComplexRank2kInKernelTileK = 24;
inline constexpr int kComplexRank2kThreadTileRows = 4;
inline constexpr int kComplexRank2kThreadTileCols = 4;
inline constexpr int kComplexRank2kLocalRows = kComplexRank2kTileM / kComplexRank2kThreadTileRows;
inline constexpr int kComplexRank2kLocalCols = kComplexRank2kTileN / kComplexRank2kThreadTileCols;
inline constexpr int kComplexRank2kThreadsPerGroup = kComplexRank2kLocalRows * kComplexRank2kLocalCols;
inline constexpr int kComplexRank2kSubgroupRows = kMaxSupportedSubgroupSize / kComplexRank2kLocalCols;
inline constexpr int kComplexRank2kSubgroupTileM = kComplexRank2kSubgroupRows * kComplexRank2kThreadTileRows;
inline constexpr int kComplexRank2kTileAStride = kComplexRank2kTileK + 1;
inline constexpr int kComplexRank2kInKernelTileAStride = kComplexRank2kInKernelTileK + 1;
inline constexpr int kComplexRank2kTileBStride = kComplexRank2kTileN + 1;
inline constexpr int kOptimizedGemmTileM = 128;
inline constexpr int kOptimizedGemmTileN = 32;
inline constexpr int kOptimizedGemmTileK = 32;
inline constexpr int kOptimizedGemmThreadTileRows = 4;
inline constexpr int kOptimizedGemmThreadTileCols = 4;
inline constexpr int kOptimizedGemmVecA = 4;
inline constexpr int kOptimizedGemmVecB = 4;
inline constexpr int kOptimizedGemmUnrollK = 2;
inline constexpr int kOptimizedGemmStages = 2;
inline constexpr int kOptimizedGemmLocalRows = kOptimizedGemmTileM / kOptimizedGemmThreadTileRows;
inline constexpr int kOptimizedGemmLocalCols = kOptimizedGemmTileN / kOptimizedGemmThreadTileCols;
inline constexpr int kOptimizedGemmThreadsPerGroup = kOptimizedGemmLocalRows * kOptimizedGemmLocalCols;
inline constexpr int kOptimizedGemmTileAStride = kOptimizedGemmTileM + 1;
inline constexpr int kOptimizedGemmTileBStride = kOptimizedGemmTileK + 1;
inline constexpr int kOptimizedGemmStageASize = kOptimizedGemmTileAStride * kOptimizedGemmTileK;
inline constexpr int kOptimizedGemmStageBSize = kOptimizedGemmTileBStride * kOptimizedGemmTileN;
inline constexpr std::size_t kSubgroupWorkspaceBudgetBytes =
    static_cast<std::size_t>(::batchlas::device_limits::subgroup_workspace_budget_bytes());

template <std::size_t FixedElements, std::size_t ElementsPerSubgroup, typename T>
inline constexpr int subgroup_limit_for_workspace_v = []() {
    constexpr std::size_t budget_elements = kSubgroupWorkspaceBudgetBytes / sizeof(T);
    if constexpr (budget_elements <= FixedElements) {
        return 0;
    } else {
        constexpr std::size_t remaining_elements = budget_elements - FixedElements;
        constexpr std::size_t subgroup_limit = ElementsPerSubgroup == 0
            ? static_cast<std::size_t>(kMaxSubgroupsPerWorkGroup)
            : (remaining_elements / ElementsPerSubgroup);
        return static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(kMaxSubgroupsPerWorkGroup), subgroup_limit));
    }
}();

template <typename T>
inline constexpr int kVectorWorkspaceMaxSubgroupsPerWorkGroup = subgroup_limit_for_workspace_v<0, 2 * kVectorTileK, T>;

template <typename T>
inline constexpr int kColumnSweepWorkspaceMaxSubgroupsPerWorkGroup = subgroup_limit_for_workspace_v<kVectorTileK, kVectorTileK, T>;

template <typename T>
inline constexpr int kRegisterMatrixWorkspaceMaxSubgroupsPerWorkGroup = subgroup_limit_for_workspace_v<
    kRegisterMatrixRhsStages * kRegisterMatrixTileK * kRegisterMatrixTileBStride,
    kRegisterMatrixLhsStages * kRegisterMatrixSubgroupTileM * kRegisterMatrixTileAStride,
    T>;

template <typename T>
inline constexpr int kComplexRank2kWorkspaceMaxSubgroupsPerWorkGroup = subgroup_limit_for_workspace_v<
    kComplexRank2kTileK * kComplexRank2kTileBStride,
    kComplexRank2kSubgroupTileM * kComplexRank2kTileAStride,
    T>;

template <typename T>
inline constexpr int kComplexRank2kInKernelWorkspaceMaxSubgroupsPerWorkGroup = subgroup_limit_for_workspace_v<
    kComplexRank2kInKernelTileK * kComplexRank2kTileBStride,
    kComplexRank2kSubgroupTileM * kComplexRank2kInKernelTileAStride,
    T>;

template <int Count, typename Fn, std::size_t... I>
inline constexpr void static_for_impl(Fn&& fn, std::index_sequence<I...>) {
    (fn(std::integral_constant<int, static_cast<int>(I)>{}), ...);
}

template <int Count, typename Fn>
inline constexpr void static_for(Fn&& fn) {
    static_for_impl<Count>(std::forward<Fn>(fn), std::make_index_sequence<Count>{});
}

template <typename T>
struct VectorWorkspace {
    T operand_tiles[2][kVectorWorkspaceMaxSubgroupsPerWorkGroup<T> * kVectorTileK];
};

template <typename T>
struct DenseGemvWorkspace {
    T x_tile[kVectorTileK];
};

template <typename T>
struct ColumnSweepVectorWorkspace {
    T x_tile[kVectorTileK];
    T accum[kColumnSweepWorkspaceMaxSubgroupsPerWorkGroup<T> * kVectorTileK];
};

template <typename T>
struct RegisterMatrixAccumTile {
    std::array<T, kRegisterMatrixThreadTileRows * kRegisterMatrixThreadTileCols> values{};

    template <int Row, int Col>
    inline constexpr T& get() {
        return std::get<Row * kRegisterMatrixThreadTileCols + Col>(values);
    }

    template <int Row, int Col>
    inline constexpr const T& get() const {
        return std::get<Row * kRegisterMatrixThreadTileCols + Col>(values);
    }
};

template <typename T>
struct ComplexRank2kAccumTile {
    std::array<T, kComplexRank2kThreadTileRows * kComplexRank2kThreadTileCols> values{};

    template <int Row, int Col>
    inline constexpr T& get() {
        return std::get<Row * kComplexRank2kThreadTileCols + Col>(values);
    }

    template <int Row, int Col>
    inline constexpr const T& get() const {
        return std::get<Row * kComplexRank2kThreadTileCols + Col>(values);
    }
};

template <typename T>
struct RegisterMatrixWorkspace {
    // clang-format off
    // Array dimensions use max(1,...) to avoid zero-length arrays on AMD SYCL targets.
    // If MaxSubgroups==0 the kernel must not be launched for this type (insufficient local memory).
    static constexpr int kMaxSubgroups = kRegisterMatrixWorkspaceMaxSubgroupsPerWorkGroup<T> > 0 ? kRegisterMatrixWorkspaceMaxSubgroupsPerWorkGroup<T> : 1;
    T lhs[kMaxSubgroups][kRegisterMatrixLhsStages][kRegisterMatrixSubgroupTileM * kRegisterMatrixTileAStride];
    T rhs[kRegisterMatrixRhsStages][kRegisterMatrixTileK * kRegisterMatrixTileBStride];
    // clang-format on
};

template <typename T>
struct ComplexRank2kWorkspace {
    static constexpr int kMaxSubgroups = kComplexRank2kWorkspaceMaxSubgroupsPerWorkGroup<T> > 0 ? kComplexRank2kWorkspaceMaxSubgroupsPerWorkGroup<T> : 1;
    T lhs[kMaxSubgroups][kComplexRank2kSubgroupTileM * kComplexRank2kTileAStride];
    T rhs[kComplexRank2kTileK * kComplexRank2kTileBStride];
};

template <typename T>
struct ComplexRank2kInKernelWorkspace {
    static constexpr int kMaxSubgroups = kComplexRank2kInKernelWorkspaceMaxSubgroupsPerWorkGroup<T> > 0 ? kComplexRank2kInKernelWorkspaceMaxSubgroupsPerWorkGroup<T> : 1;
    T lhs[kMaxSubgroups][kComplexRank2kSubgroupTileM * kComplexRank2kInKernelTileAStride];
    T rhs[kComplexRank2kInKernelTileK * kComplexRank2kTileBStride];
};

template <typename T>
struct OptimizedGemmWorkspace {
    T lhs[kOptimizedGemmStages * kOptimizedGemmStageASize];
    T rhs[kOptimizedGemmStages * kOptimizedGemmStageBSize];
};

template <typename T>
struct GemmWorkspace {
    union {
        RegisterMatrixWorkspace<T> register_workspace;
        OptimizedGemmWorkspace<T> optimized_workspace;
    };
};

template <typename T>
inline constexpr bool vector_workspace_supported_v = sizeof(VectorWorkspace<T>) <= kSubgroupWorkspaceBudgetBytes;

template <typename T>
inline constexpr bool dense_gemv_workspace_supported_v = sizeof(DenseGemvWorkspace<T>) <= kSubgroupWorkspaceBudgetBytes;

template <typename T>
inline constexpr bool column_sweep_workspace_supported_v = sizeof(ColumnSweepVectorWorkspace<T>) <= kSubgroupWorkspaceBudgetBytes;

template <typename T>
inline constexpr bool register_matrix_workspace_supported_v = sizeof(RegisterMatrixWorkspace<T>) <= kSubgroupWorkspaceBudgetBytes;

template <typename T>
inline constexpr bool complex_rank2k_workspace_supported_v = sizeof(ComplexRank2kWorkspace<T>) <= kSubgroupWorkspaceBudgetBytes;

template <typename T>
inline constexpr bool complex_rank2k_in_kernel_workspace_supported_v =
    sizeof(ComplexRank2kInKernelWorkspace<T>) <= kSubgroupWorkspaceBudgetBytes;

template <typename T>
inline constexpr bool optimized_gemm_workspace_supported_v = sizeof(OptimizedGemmWorkspace<T>) <= kSubgroupWorkspaceBudgetBytes;

template <typename T>
inline constexpr bool gemm_workspace_supported_v = sizeof(GemmWorkspace<T>) <= kSubgroupWorkspaceBudgetBytes;

template <typename T>
inline constexpr bool supports_packet4_v = std::is_same_v<T, float>;

template <typename T, int Width>
inline constexpr bool supports_packet_v = false;

template <typename T>
inline constexpr bool supports_packet_v<T, 4> = supports_packet4_v<T>;

template <typename T, int Width>
inline constexpr bool supports_aligned_packet_loads(const T* ptr, int ld, int stride) {
    if constexpr (!supports_packet_v<T, Width>) {
        static_cast<void>(ptr);
        static_cast<void>(ld);
        static_cast<void>(stride);
        return false;
    } else {
        const auto address = reinterpret_cast<std::uintptr_t>(ptr);
        return (address % alignof(sycl::vec<T, Width>) == 0) && (ld % Width == 0) && (stride % Width == 0);
    }
}

template <typename T, int Width>
inline constexpr sycl::vec<T, Width> packet_load_aligned(const T* ptr, int offset) {
    return *reinterpret_cast<const sycl::vec<T, Width>*>(ptr + offset);
}

inline constexpr int register_matrix_lhs_stage(int tile_idx) {
    return tile_idx % kRegisterMatrixLhsStages;
}

inline constexpr int register_matrix_rhs_stage(int tile_idx) {
    return tile_idx % kRegisterMatrixRhsStages;
}

template <typename Item>
inline constexpr int matrix_tile_group_row(const Item&) {
    return 0;
}

template <typename Item>
inline constexpr int matrix_tile_group_col(const Item&) {
    return 0;
}

template <typename Item>
inline constexpr int matrix_tile_group_row_stride(const Item&) {
    return 1;
}

template <typename Item>
inline constexpr int matrix_tile_group_col_stride(const Item&) {
    return 1;
}

inline constexpr int matrix_tile_group_row(const sycl::nd_item<3>& item) {
    return static_cast<int>(item.get_group(1));
}

inline constexpr int matrix_tile_group_col(const sycl::nd_item<3>& item) {
    return static_cast<int>(item.get_group(2));
}

inline constexpr int matrix_tile_group_row_stride(const sycl::nd_item<3>& item) {
    return std::max(1, static_cast<int>(item.get_group_range(1)));
}

inline constexpr int matrix_tile_group_col_stride(const sycl::nd_item<3>& item) {
    return std::max(1, static_cast<int>(item.get_group_range(2)));
}

template <typename Item>
inline constexpr int subgroup_size(const Item& item) {
    return static_cast<int>(item.get_sub_group().get_local_range().size());
}

inline constexpr int subgroup_size(const DeviceBlasLaunchInfo& launch) {
    return std::max(1, launch.subgroup_size);
}

template <typename Item>
inline constexpr int subgroup_local_id(const Item& item) {
    return static_cast<int>(item.get_sub_group().get_local_linear_id());
}

template <typename Item>
inline constexpr int subgroup_group_id(const Item& item) {
    return detail::item_local_linear_id(item) / subgroup_size(item);
}

template <typename Item>
inline constexpr int subgroup_count(const Item& item) {
    const int sg_size = subgroup_size(item);
    return std::max(1, detail::group_local_linear_range(item) / std::max(1, sg_size));
}

inline constexpr int subgroup_count(const DeviceBlasLaunchInfo& launch) {
    const int sg_size = subgroup_size(launch);
    return std::max(1, detail::group_local_linear_range(launch) / std::max(1, sg_size));
}

template <typename Exec>
inline constexpr bool is_nd_item_1d_launch(const Exec&) {
    return std::is_same_v<std::remove_cvref_t<Exec>, sycl::nd_item<1>>;
}

inline constexpr bool is_nd_item_1d_launch(const DeviceBlasLaunchInfo& launch) {
    return launch.kind == DeviceBlasLaunchKind::NdItem1D;
}

template <typename Exec>
inline constexpr bool is_nd_item_3d_launch(const Exec&) {
    return std::is_same_v<std::remove_cvref_t<Exec>, sycl::nd_item<3>>;
}

inline constexpr bool is_nd_item_3d_launch(const DeviceBlasLaunchInfo& launch) {
    return launch.kind == DeviceBlasLaunchKind::NdItem3D;
}

template <typename Item, typename Fn>
inline constexpr void for_each_subgroup_vector_index(const Item& item, int extent, Fn&& fn) {
    const int sg_size = subgroup_size(item);
    const int lane = subgroup_local_id(item);
    const int sg_id = subgroup_group_id(item);
    const int total_sg = subgroup_count(item);
    const int block = 2 * sg_size;

    for (int base = sg_id * block; base < extent; base += total_sg * block) {
        const int index0 = base + lane;
        if (index0 < extent) {
            fn(index0);
        }
        const int index1 = index0 + sg_size;
        if (index1 < extent) {
            fn(index1);
        }
    }
}

inline constexpr int rows_per_subgroup(int extent, int total_sg, int max_rows) {
    if (extent >= total_sg * max_rows) {
        return max_rows;
    }
    if (max_rows >= 4 && extent >= total_sg * 2) {
        return 2;
    }
    return 1;
}

inline constexpr int matrix_tile_k(int contract_extent, int sg_size, Side side) {
    if (contract_extent >= 256 && sg_size == 32) {
        return side == Side::Left ? 32 : 16;
    }
    if (contract_extent >= 128) {
        return 16;
    }
    return 8;
}

inline constexpr int matrix_rows_per_subgroup(int row_extent,
                                              int total_sg,
                                              int sg_size,
                                              int contract_extent,
                                              Side side) {
    if (side == Side::Left && sg_size == 32 && contract_extent >= 128 && row_extent >= total_sg * 8) {
        return 8;
    }
    return rows_per_subgroup(row_extent, total_sg, 4);
}

inline constexpr bool subgroup_policy_matches(DeviceBlasPolicy policy, int actual_size) {
    switch (policy) {
    case DeviceBlasPolicy::Generic:
        return false;
    case DeviceBlasPolicy::Subgroup16:
        return actual_size == 16;
    case DeviceBlasPolicy::Subgroup32:
        return actual_size == 32;
    case DeviceBlasPolicy::Auto:
        return actual_size == 16 || actual_size == 32;
    }
    return false;
}

template <typename T, typename... Ops, typename Item>
inline constexpr bool can_use_vector_fast_path(const Item& item,
                                               const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                               MatrixVectorTransform transform,
                                               DeviceBlasPolicy policy) {
    if constexpr (!vector_workspace_supported_v<T>) {
        return false;
    }
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }
    if constexpr (sizeof...(Ops) == 0 || sizeof...(Ops) > 2) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    if (!subgroup_policy_matches(policy, sg_size)) {
        return false;
    }
    if (sg_size > kMaxSupportedSubgroupSize || subgroup_count(item) > kVectorWorkspaceMaxSubgroupsPerWorkGroup<T>) {
        return false;
    }

    const int inner_extent = detail::input_size(a, transform.trans);
    const int outer_extent = detail::output_size(a, transform.trans);
    return inner_extent >= sg_size && outer_extent >= 2;
}

template <typename T, typename Item>
inline constexpr bool can_use_matrix_fast_path(const Item& item,
                                               int row_extent,
                                               int col_extent,
                                               int contract_extent,
                                               DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    if (!subgroup_policy_matches(policy, sg_size)) {
        return false;
    }
    if (sg_size > kMaxSupportedSubgroupSize) {
        return false;
    }

    return row_extent >= 8 && col_extent >= sg_size && contract_extent >= sg_size;
}

template <typename T, typename Item>
inline constexpr bool can_use_matrix_register_fast_path(const Item& item,
                                                        int row_extent,
                                                        int col_extent,
                                                        int contract_extent,
                                                        DeviceBlasPolicy policy) {
    if constexpr (!register_matrix_workspace_supported_v<T>) {
        return false;
    }
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    if (policy == DeviceBlasPolicy::Generic) {
        return false;
    }
    if (detail::group_local_linear_range(item) != kRegisterMatrixThreadsPerGroup) {
        return false;
    }

    return row_extent >= kRegisterMatrixThreadTileRows &&
        col_extent >= kRegisterMatrixThreadTileCols &&
        contract_extent >= kRegisterMatrixTileK;
}

template <typename T, typename Item>
inline constexpr bool can_use_rankk_fast_path(const Item& item,
                                              const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                              const RankKOperand<T>& operand,
                                              SymmetricRankKTransform transform,
                                              DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    if (policy == DeviceBlasPolicy::Generic) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    if (policy == DeviceBlasPolicy::Auto) {
        if (transform.trans != Transpose::NoTrans || sg_size != 16) {
            return false;
        }
    } else if (!subgroup_policy_matches(policy, sg_size)) {
        return false;
    }
    const int sg_count = subgroup_count(item);
    if (sg_size > kMaxSupportedSubgroupSize || sg_count > kMaxSubgroupsPerWorkGroup) {
        return false;
    }

    return detail::output_size(a, transform.trans) > 0 &&
        detail::input_size(a, transform.trans) > 0 &&
        operand.c.rows() > 0 &&
        detail::output_size(a, transform.trans) <= 256 &&
        detail::input_size(a, transform.trans) <= 32;
}

template <typename T, typename Item>
inline constexpr bool can_use_matrix_aligned_nn_large_fast_path(const Item& item,
                                                                int row_extent,
                                                                int col_extent,
                                                                int contract_extent,
                                                                Transpose trans_a,
                                                                Transpose trans_b,
                                                                DeviceBlasPolicy policy);

template <typename T, typename Item>
inline constexpr bool can_use_matrix_aligned_nn_large_fast_path(const Item& item,
                                                                const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                                const MatrixMatrixOperand<T>& operand,
                                                                GeneralMatrixTransform transform,
                                                                DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    if (policy == DeviceBlasPolicy::Generic) {
        return false;
    }
    if (transform.trans_a != Transpose::NoTrans || transform.trans_b != Transpose::NoTrans) {
        return false;
    }
    if (detail::group_local_linear_range(item) != kOptimizedGemmThreadsPerGroup) {
        return false;
    }

    const int row_extent = detail::output_size(a, transform.trans_a);
    const int col_extent = detail::input_size(operand.b, transform.trans_b);
    const int contract_extent = detail::input_size(a, transform.trans_a);
    if (row_extent < kOptimizedGemmTileM || col_extent < kOptimizedGemmTileN || contract_extent < kOptimizedGemmTileK) {
        return false;
    }
    if ((row_extent % kOptimizedGemmTileM) != 0 ||
        (col_extent % kRegisterMatrixTileN) != 0 ||
        (contract_extent % kOptimizedGemmTileK) != 0) {
        return false;
    }

    return supports_aligned_packet_loads<T, kOptimizedGemmVecA>(a.data(), a.ld(), a.stride()) &&
        supports_aligned_packet_loads<T, kOptimizedGemmVecB>(operand.b.data(), operand.b.ld(), operand.b.stride());
}

template <typename T, typename Item>
inline constexpr bool can_use_matrix_aligned_nn_large_fast_path(const Item& item,
                                                                int row_extent,
                                                                int col_extent,
                                                                int contract_extent,
                                                                Transpose trans_a,
                                                                Transpose trans_b,
                                                                DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    if (policy == DeviceBlasPolicy::Generic) {
        return false;
    }
    if (trans_a != Transpose::NoTrans || trans_b != Transpose::NoTrans) {
        return false;
    }
    if (detail::group_local_linear_range(item) != kOptimizedGemmThreadsPerGroup) {
        return false;
    }

    if (row_extent < kOptimizedGemmTileM || col_extent < kOptimizedGemmTileN || contract_extent < kOptimizedGemmTileK) {
        return false;
    }
    if ((row_extent % kOptimizedGemmTileM) != 0 ||
        (col_extent % kRegisterMatrixTileN) != 0 ||
        (contract_extent % kOptimizedGemmTileK) != 0) {
        return false;
    }

    return true;
}

template <typename T, typename Item>
inline constexpr bool can_use_complex_rankk_in_kernel_fast_path(const Item& item,
                                                                const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                                const RankKOperand<T>& operand,
                                                                SymmetricRankKTransform transform,
                                                                DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, std::complex<float>>) {
        return false;
    }

    if (!is_nd_item_1d_launch(item)) {
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::group_local_linear_range(item) != kComplexRank2kThreadsPerGroup) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    const int sg_count = subgroup_count(item);
    if (sg_size != kMaxSupportedSubgroupSize || sg_count > kMaxSubgroupsPerWorkGroup) {
        return false;
    }

    const int extent = detail::output_size(a, transform.trans);
    const int contract_extent = detail::input_size(a, transform.trans);
    return extent >= kComplexRank2kTileN &&
        contract_extent >= 16 &&
        operand.c.rows() == extent &&
        operand.c.cols() == extent;
}

template <typename T, typename Item>
inline constexpr bool can_use_complex_rank2k_tiled_fast_path(const Item& item,
                                                             int extent,
                                                             int contract_extent,
                                                             DeviceBlasPolicy policy);

template <typename T, typename Item>
inline constexpr bool can_use_complex_rank2k_tiled_fast_path(const Item& item,
                                                             const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                             const MatrixMatrixOperand<T>& operand,
                                                             SymmetricRank2kTransform transform,
                                                             DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, std::complex<float>>) {
        return false;
    }

    if (!is_nd_item_3d_launch(item)) {
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::group_local_linear_range(item) != kComplexRank2kThreadsPerGroup) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    const int sg_count = subgroup_count(item);
    if (sg_size != kMaxSupportedSubgroupSize || sg_count > kComplexRank2kWorkspaceMaxSubgroupsPerWorkGroup<T>) {
        return false;
    }

    const int extent = detail::output_size(a, transform.trans);
    const int contract_extent = detail::input_size(a, transform.trans);
    return extent >= kComplexRank2kTileN &&
        contract_extent >= 16 &&
        operand.c.rows() == extent &&
        operand.c.cols() == extent;
}

template <typename T, typename Item>
inline constexpr bool can_use_complex_rank2k_tiled_fast_path(const Item& item,
                                                             int extent,
                                                             int contract_extent,
                                                             DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, std::complex<float>>) {
        return false;
    }

    if (!is_nd_item_3d_launch(item)) {
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::group_local_linear_range(item) != kComplexRank2kThreadsPerGroup) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    const int sg_count = subgroup_count(item);
    if (sg_size != kMaxSupportedSubgroupSize || sg_count > kComplexRank2kWorkspaceMaxSubgroupsPerWorkGroup<T>) {
        return false;
    }

    return extent >= kComplexRank2kTileN &&
        contract_extent >= 16 &&
        extent > 0;
}

template <typename T, typename Item>
inline constexpr bool can_use_complex_rankk_tiled_fast_path(const Item& item,
                                                            int extent,
                                                            int contract_extent,
                                                            DeviceBlasPolicy policy);

template <typename T, typename Item>
inline constexpr bool can_use_complex_rankk_tiled_fast_path(const Item& item,
                                                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                            const RankKOperand<T>& operand,
                                                            SymmetricRankKTransform transform,
                                                            DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, std::complex<float>>) {
        return false;
    }

    if (!is_nd_item_3d_launch(item)) {
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::group_local_linear_range(item) != kComplexRank2kThreadsPerGroup) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    const int sg_count = subgroup_count(item);
    if (sg_size != kMaxSupportedSubgroupSize || sg_count > kComplexRank2kWorkspaceMaxSubgroupsPerWorkGroup<T>) {
        return false;
    }

    const int extent = detail::output_size(a, transform.trans);
    const int contract_extent = detail::input_size(a, transform.trans);
    return extent >= kComplexRank2kTileN &&
        contract_extent >= 16 &&
        operand.c.rows() == extent &&
        operand.c.cols() == extent;
}

template <typename T, typename Item>
inline constexpr bool can_use_complex_rankk_tiled_fast_path(const Item& item,
                                                            int extent,
                                                            int contract_extent,
                                                            DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, std::complex<float>>) {
        return false;
    }

    if (!is_nd_item_3d_launch(item)) {
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::group_local_linear_range(item) != kComplexRank2kThreadsPerGroup) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    const int sg_count = subgroup_count(item);
    if (sg_size != kMaxSupportedSubgroupSize || sg_count > kComplexRank2kWorkspaceMaxSubgroupsPerWorkGroup<T>) {
        return false;
    }

    return extent >= kComplexRank2kTileN &&
        contract_extent >= 16 &&
        extent > 0;
}

template <typename T, typename Item>
inline constexpr bool can_use_rank2k_register_fast_path(const Item& item,
                                                        int extent,
                                                        int contract_extent,
                                                        DeviceBlasPolicy policy);

template <typename T, typename Item>
inline constexpr bool can_use_rank2k_register_fast_path(const Item& item,
                                                        const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                        const MatrixMatrixOperand<T>& operand,
                                                        SymmetricRank2kTransform transform,
                                                        DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    if (!is_nd_item_3d_launch(item)) {
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::group_local_linear_range(item) != kRegisterMatrixThreadsPerGroup) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    const int sg_count = subgroup_count(item);
    if (sg_size != kMaxSupportedSubgroupSize || sg_count > kMaxSubgroupsPerWorkGroup) {
        return false;
    }

    const int extent = detail::output_size(a, transform.trans);
    const int contract_extent = detail::input_size(a, transform.trans);
    return extent >= kRegisterMatrixTileN &&
        contract_extent >= 16 &&
        operand.c.rows() == extent &&
        operand.c.cols() == extent;
}

template <typename T, typename Item>
inline constexpr bool can_use_rank2k_register_fast_path(const Item& item,
                                                        int extent,
                                                        int contract_extent,
                                                        DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    if (!is_nd_item_3d_launch(item)) {
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::group_local_linear_range(item) != kRegisterMatrixThreadsPerGroup) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    const int sg_count = subgroup_count(item);
    if (sg_size != kMaxSupportedSubgroupSize || sg_count > kMaxSubgroupsPerWorkGroup) {
        return false;
    }

    return extent >= kRegisterMatrixTileN &&
        contract_extent >= 16 &&
        extent > 0;
}

template <typename T, typename Item>
inline constexpr bool can_use_rankk_register_fast_path(const Item& item,
                                                       int extent,
                                                       int contract_extent,
                                                       DeviceBlasPolicy policy);

template <typename T, typename Item>
inline constexpr bool can_use_rankk_register_fast_path(const Item& item,
                                                       const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                       const RankKOperand<T>& operand,
                                                       SymmetricRankKTransform transform,
                                                       DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    if (!is_nd_item_3d_launch(item)) {
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::group_local_linear_range(item) != kRegisterMatrixThreadsPerGroup) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    const int sg_count = subgroup_count(item);
    if (sg_size != kMaxSupportedSubgroupSize || sg_count > kMaxSubgroupsPerWorkGroup) {
        return false;
    }

    const int extent = detail::output_size(a, transform.trans);
    const int contract_extent = detail::input_size(a, transform.trans);
    return extent >= kRegisterMatrixTileN &&
        contract_extent >= 16 &&
        operand.c.rows() == extent &&
        operand.c.cols() == extent;
}

template <typename T, typename Item>
inline constexpr bool can_use_rankk_register_fast_path(const Item& item,
                                                       int extent,
                                                       int contract_extent,
                                                       DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    if (!is_nd_item_3d_launch(item)) {
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::group_local_linear_range(item) != kRegisterMatrixThreadsPerGroup) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    const int sg_count = subgroup_count(item);
    if (sg_size != kMaxSupportedSubgroupSize || sg_count > kMaxSubgroupsPerWorkGroup) {
        return false;
    }

    return extent >= kRegisterMatrixTileN &&
        contract_extent >= 16 &&
        extent > 0;
}

inline constexpr int triangular_begin(int index, int extent, TriangularTransform transform, Side side) {
    if (side == Side::Left) {
        if (transform.trans == Transpose::NoTrans) {
            return transform.uplo == Uplo::Lower ? 0 : index;
        }
        return transform.uplo == Uplo::Lower ? index : 0;
    }

    if (transform.trans == Transpose::NoTrans) {
        return transform.uplo == Uplo::Lower ? index : 0;
    }
    return transform.uplo == Uplo::Lower ? 0 : index;
}

inline constexpr int triangular_end(int index, int extent, TriangularTransform transform, Side side) {
    if (side == Side::Left) {
        if (transform.trans == Transpose::NoTrans) {
            return transform.uplo == Uplo::Lower ? index + 1 : extent;
        }
        return transform.uplo == Uplo::Lower ? extent : index + 1;
    }

    if (transform.trans == Transpose::NoTrans) {
        return transform.uplo == Uplo::Lower ? extent : index + 1;
    }
    return transform.uplo == Uplo::Lower ? index + 1 : extent;
}

inline constexpr int triangular_tile_begin(int row_base,
                                           int row_extent,
                                           int col_base,
                                           int col_extent,
                                           int contract_extent,
                                           TriangularTransform transform) {
    if (transform.side == Side::Left) {
        const int first_row = std::min(row_extent - 1, row_base);
        const int last_row = std::min(row_extent - 1, row_base + kRegisterMatrixTileM - 1);
        if (first_row < 0 || last_row < 0) {
            return 0;
        }
        if (transform.trans == Transpose::NoTrans) {
            return transform.uplo == Uplo::Lower ? 0 : first_row;
        }
        return transform.uplo == Uplo::Lower ? first_row : 0;
    }

    const int first_col = std::min(col_extent - 1, col_base);
    const int last_col = std::min(col_extent - 1, col_base + kRegisterMatrixTileN - 1);
    if (first_col < 0 || last_col < 0) {
        return 0;
    }
    if (transform.trans == Transpose::NoTrans) {
        return transform.uplo == Uplo::Lower ? first_col : 0;
    }
    return transform.uplo == Uplo::Lower ? 0 : first_col;
}

inline constexpr int triangular_tile_end(int row_base,
                                         int row_extent,
                                         int col_base,
                                         int col_extent,
                                         int contract_extent,
                                         TriangularTransform transform) {
    if (transform.side == Side::Left) {
        const int last_row = std::min(row_extent - 1, row_base + kRegisterMatrixTileM - 1);
        if (last_row < 0) {
            return 0;
        }
        if (transform.trans == Transpose::NoTrans) {
            return transform.uplo == Uplo::Lower ? std::min(contract_extent, last_row + 1) : contract_extent;
        }
        return transform.uplo == Uplo::Lower ? contract_extent : std::min(contract_extent, last_row + 1);
    }

    const int last_col = std::min(col_extent - 1, col_base + kRegisterMatrixTileN - 1);
    if (last_col < 0) {
        return 0;
    }
    if (transform.trans == Transpose::NoTrans) {
        return transform.uplo == Uplo::Lower ? contract_extent : std::min(contract_extent, last_col + 1);
    }
    return transform.uplo == Uplo::Lower ? std::min(contract_extent, last_col + 1) : contract_extent;
}

template <typename Item, typename T, typename LhsLoader, typename RhsLoader>
inline constexpr void load_register_matrix_stage(const Item& item,
                                                 RegisterMatrixWorkspace<T>* workspace,
                                                 int linear_tid,
                                                 int subgroup_id,
                                                 int lhs_stage,
                                                 int rhs_stage,
                                                 bool load_lhs,
                                                 bool load_rhs,
                                                 int k_base,
                                                 int tile_extent,
                                                 LhsLoader&& lhs_loader,
                                                 RhsLoader&& rhs_loader) {
    const int subgroup_lane = subgroup_local_id(item);
    T* lhs_stage_ptr = workspace->lhs[subgroup_id][lhs_stage];
    T* rhs_stage_ptr = workspace->rhs[rhs_stage];

    if (load_lhs) {
        for (int index = subgroup_lane; index < kRegisterMatrixSubgroupTileM * kRegisterMatrixTileK; index += kMaxSupportedSubgroupSize) {
            const int subgroup_tile_r = index / kRegisterMatrixTileK;
            const int tile_k = index % kRegisterMatrixTileK;
            const int tile_r = subgroup_id * kRegisterMatrixSubgroupTileM + subgroup_tile_r;
            lhs_stage_ptr[subgroup_tile_r * kRegisterMatrixTileAStride + tile_k] =
                tile_k < tile_extent ? lhs_loader(tile_r, k_base + tile_k) : T(0);
        }
    }
    if (load_rhs) {
        for (int index = linear_tid; index < kRegisterMatrixTileK * kRegisterMatrixTileN; index += kRegisterMatrixThreadsPerGroup) {
            const int tile_k = index / kRegisterMatrixTileN;
            const int tile_c = index % kRegisterMatrixTileN;
            rhs_stage_ptr[tile_k * kRegisterMatrixTileBStride + tile_c] =
                tile_k < tile_extent ? rhs_loader(tile_c, k_base + tile_k) : T(0);
        }
    }
}

template <typename T>
struct RegisterMatrixThreadTileValues {
    std::array<T, kRegisterMatrixThreadTileRows> lhs{};
    std::array<T, kRegisterMatrixThreadTileCols> rhs{};

    template <int Row>
    inline constexpr T& lhs_value() {
        return std::get<Row>(lhs);
    }

    template <int Row>
    inline constexpr const T& lhs_value() const {
        return std::get<Row>(lhs);
    }

    template <int Col>
    inline constexpr T& rhs_value() {
        return std::get<Col>(rhs);
    }

    template <int Col>
    inline constexpr const T& rhs_value() const {
        return std::get<Col>(rhs);
    }
};

template <typename Item, typename T>
inline constexpr void accumulate_register_matrix_stage(const Item& item,
                                                       const RegisterMatrixWorkspace<T>* workspace,
                                                       int subgroup_id,
                                                       int lhs_stage,
                                                       int rhs_stage,
                                                       int local_row,
                                                       int local_col,
                                                       int tile_extent,
                                                       RegisterMatrixAccumTile<T>& accum) {
    const auto sg = item.get_sub_group();
    const int sg_lane = subgroup_local_id(item);
    const int sg_row = sg_lane / kRegisterMatrixLocalCols;
    const int sg_col = sg_lane % kRegisterMatrixLocalCols;
    const int lhs_row_base = (local_row % kRegisterMatrixSubgroupRows) * kRegisterMatrixThreadTileRows;
    const int lhs_source_lane = sg_row * kRegisterMatrixLocalCols;
    const int rhs_source_lane_base = sg_col;
    const T* lhs_stage_ptr = workspace->lhs[subgroup_id][lhs_stage];
    const T* rhs_stage_ptr = workspace->rhs[rhs_stage];
    const int rhs_col_base = local_col * kRegisterMatrixThreadTileCols;

    for (int kk = 0; kk < tile_extent; ++kk) {
        RegisterMatrixThreadTileValues<T> fragments;
        static_for<kRegisterMatrixThreadTileRows>([&](auto row_idx) {
            constexpr int i = row_idx;
            const T lhs_lane = sg_col == 0 ? lhs_stage_ptr[(lhs_row_base + i) * kRegisterMatrixTileAStride + kk] : T(0);
            fragments.template lhs_value<i>() = sycl::select_from_group(sg, lhs_lane, static_cast<uint32_t>(lhs_source_lane));
        });

        static_for<kRegisterMatrixThreadTileCols>([&](auto col_idx) {
            constexpr int j = col_idx;
            const T rhs_lane = sg_row == 0 ? rhs_stage_ptr[kk * kRegisterMatrixTileBStride + rhs_col_base + j] : T(0);
            fragments.template rhs_value<j>() =
                sycl::select_from_group(sg, rhs_lane, static_cast<uint32_t>(rhs_source_lane_base));
        });

        static_for<kRegisterMatrixThreadTileRows>([&](auto row_idx) {
            constexpr int i = row_idx;
            const T lhs_value = fragments.template lhs_value<i>();
            static_for<kRegisterMatrixThreadTileCols>([&](auto col_idx) {
                constexpr int j = col_idx;
                accum.template get<i, j>() += lhs_value * fragments.template rhs_value<j>();
            });
        });
    }
}

template <typename T>
inline constexpr void write_register_matrix_tile(MatrixMatrixOperand<T> operand,
                                                 int row_base,
                                                 int col_base,
                                                 int row_extent,
                                                 int col_extent,
                                                 const RegisterMatrixAccumTile<T>& accum) {
    static_for<kRegisterMatrixThreadTileRows>([&](auto row_idx) {
        constexpr int i = row_idx;
        const int row = row_base + i;
        if (row >= row_extent) {
            return;
        }
        static_for<kRegisterMatrixThreadTileCols>([&](auto col_idx) {
            constexpr int j = col_idx;
            const int col = col_base + j;
            if (col >= col_extent) {
                return;
            }
            detail::write_matrix_output(operand, row, col, accum.template get<i, j>());
        });
    });
}

template <typename T, typename Transform>
inline constexpr void write_rank2k_register_tile(MatrixMatrixOperand<T> operand,
                                                 Transform transform,
                                                 int row_base,
                                                 int col_base,
                                                 int extent,
                                                 const RegisterMatrixAccumTile<T>& accum) {
    static_for<kRegisterMatrixThreadTileRows>([&](auto row_idx) {
        constexpr int i = row_idx;
        const int row = row_base + i;
        if (row >= extent) {
            return;
        }
        static_for<kRegisterMatrixThreadTileCols>([&](auto col_idx) {
            constexpr int j = col_idx;
            const int col = col_base + j;
            if (col >= extent || !detail::triangular_storage_contains(transform.uplo, row, col)) {
                return;
            }
            T value = operand.beta * operand.c(row, col) + accum.template get<i, j>();
            if constexpr (ComplexScalar<T>) {
                if (transform.hermitian && row == col) {
                    value = T(value.real(), typename T::value_type(0));
                }
            }
            operand.c(row, col) = value;
        });
    });
}

template <typename T, typename Transform>
inline constexpr void write_complex_rank2k_tile(MatrixMatrixOperand<T> operand,
                                                Transform transform,
                                                int row_base,
                                                int col_base,
                                                int extent,
                                                const ComplexRank2kAccumTile<T>& accum) {
    static_for<kComplexRank2kThreadTileRows>([&](auto row_idx) {
        constexpr int i = row_idx;
        const int row = row_base + i;
        if (row >= extent) {
            return;
        }
        static_for<kComplexRank2kThreadTileCols>([&](auto col_idx) {
            constexpr int j = col_idx;
            const int col = col_base + j;
            if (col >= extent || !detail::triangular_storage_contains(transform.uplo, row, col)) {
                return;
            }
            T value = operand.beta * operand.c(row, col) + accum.template get<i, j>();
            if constexpr (ComplexScalar<T>) {
                if (transform.hermitian && row == col) {
                    value = T(value.real(), typename T::value_type(0));
                }
            }
            operand.c(row, col) = value;
        });
    });
}

template <typename T, typename... Ops>
inline constexpr void accumulate_cached_operands(std::array<T, sizeof...(Ops)>& partials,
                                                 const std::tuple<Ops...>& operands,
                                                 int input_index,
                                                 const T& a_ij,
                                                 const T& x0,
                                                 const T& x1) {
    static_cast<void>(input_index);
    partials[0] += a_ij * x0;
    if constexpr (sizeof...(Ops) == 2) {
        partials[1] += a_ij * x1;
    }
}

template <typename Item, typename T, typename LhsLoader, typename RhsLoader>
inline constexpr void accumulate_rank2k_register_tiled_pass(const Item& item,
                                                            RegisterMatrixWorkspace<T>* workspace,
                                                            int linear_tid,
                                                            int subgroup_id,
                                                            int contract_extent,
                                                            LhsLoader&& lhs_loader,
                                                            RhsLoader&& rhs_loader,
                                                            RegisterMatrixAccumTile<T>& accum) {
    const int tile_count = (contract_extent + kRegisterMatrixTileK - 1) / kRegisterMatrixTileK;
    if (tile_count <= 0) {
        return;
    }

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
                                         linear_tid / kRegisterMatrixLocalCols,
                                         linear_tid % kRegisterMatrixLocalCols,
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

    sycl::group_barrier(item.get_group());
}

template <int TileK, int TileAStride, typename Workspace, typename Item, typename T, typename LhsLoader, typename RhsLoader>
inline constexpr void accumulate_complex_rank2k_tiled_pass_impl(const Item& item,
                                                                Workspace* workspace,
                                                                int linear_tid,
                                                                int subgroup_id,
                                                                int local_row,
                                                                int local_col,
                                                                int contract_extent,
                                                                LhsLoader&& lhs_loader,
                                                                RhsLoader&& rhs_loader,
                                                                ComplexRank2kAccumTile<T>& accum) {
    const auto sg = item.get_sub_group();
    const int sg_lane = subgroup_local_id(item);
    const int sg_row = sg_lane / kComplexRank2kLocalCols;
    const int sg_col = sg_lane % kComplexRank2kLocalCols;
    const int lhs_row_base = (local_row % kComplexRank2kSubgroupRows) * kComplexRank2kThreadTileRows;
    const int lhs_source_lane = sg_row * kComplexRank2kLocalCols;
    const int rhs_source_lane_base = sg_col;
    const int rhs_col_base = local_col * kComplexRank2kThreadTileCols;
    T* lhs_tile = workspace->lhs[subgroup_id];
    T* rhs_tile = workspace->rhs;

    for (int k_base = 0; k_base < contract_extent; k_base += TileK) {
        const int tile_extent = std::min(TileK, contract_extent - k_base);

        for (int index = sg_lane; index < kComplexRank2kSubgroupTileM * TileK; index += kMaxSupportedSubgroupSize) {
            const int subgroup_tile_r = index / TileK;
            const int tile_k = index % TileK;
            const int tile_r = subgroup_id * kComplexRank2kSubgroupTileM + subgroup_tile_r;
            lhs_tile[subgroup_tile_r * TileAStride + tile_k] =
                tile_k < tile_extent ? lhs_loader(tile_r, k_base + tile_k) : T(0);
        }
        for (int index = linear_tid; index < TileK * kComplexRank2kTileN; index += kComplexRank2kThreadsPerGroup) {
            const int tile_k = index / kComplexRank2kTileN;
            const int tile_c = index % kComplexRank2kTileN;
            rhs_tile[tile_k * kComplexRank2kTileBStride + tile_c] =
                tile_k < tile_extent ? rhs_loader(tile_c, k_base + tile_k) : T(0);
        }
        sycl::group_barrier(item.get_group());

        for (int kk = 0; kk < tile_extent; ++kk) {
            std::array<T, kComplexRank2kThreadTileRows> lhs_frag{};
            std::array<T, kComplexRank2kThreadTileCols> rhs_frag{};

            static_for<kComplexRank2kThreadTileRows>([&](auto row_idx) {
                constexpr int i = row_idx;
                const T lhs_lane = sg_col == 0 ? lhs_tile[(lhs_row_base + i) * TileAStride + kk] : T(0);
                lhs_frag[static_cast<std::size_t>(i)] =
                    sycl::select_from_group(sg, lhs_lane, static_cast<uint32_t>(lhs_source_lane));
            });

            static_for<kComplexRank2kThreadTileCols>([&](auto col_idx) {
                constexpr int j = col_idx;
                const T rhs_lane = sg_row == 0 ? rhs_tile[kk * kComplexRank2kTileBStride + rhs_col_base + j] : T(0);
                rhs_frag[static_cast<std::size_t>(j)] =
                    sycl::select_from_group(sg, rhs_lane, static_cast<uint32_t>(rhs_source_lane_base));
            });

            static_for<kComplexRank2kThreadTileRows>([&](auto row_idx) {
                constexpr int i = row_idx;
                const T lhs_value = lhs_frag[static_cast<std::size_t>(i)];
                static_for<kComplexRank2kThreadTileCols>([&](auto col_idx) {
                    constexpr int j = col_idx;
                    accum.template get<i, j>() += lhs_value * rhs_frag[static_cast<std::size_t>(j)];
                });
            });
        }

        sycl::group_barrier(item.get_group());
    }
}

} // namespace detail::subgroup

} // namespace batchlas::device