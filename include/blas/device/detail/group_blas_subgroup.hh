#include <cstdint>

namespace detail::subgroup {

inline constexpr int kMaxSupportedSubgroupSize = 32;
inline constexpr int kMaxSubgroupsPerWorkGroup = 8;
inline constexpr int kVectorTileK = 64;
inline constexpr int kMaxVectorRowsPerSubgroup = 4;
inline constexpr int kMaxMatrixTileK = 32;
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
    T operand_tiles[2][kMaxSubgroupsPerWorkGroup * kVectorTileK];
};

template <typename T>
struct DenseGemvWorkspace {
    T x_tile[kVectorTileK];
};

inline constexpr int kColumnSweepVectorMaxExtent = kMaxSubgroupsPerWorkGroup * kVectorTileK;
inline constexpr int kColumnSweepMaxRowsPerThread = 32;

template <typename T>
struct ColumnSweepVectorWorkspace {
    T x_tile[kVectorTileK];
    T accum[kColumnSweepVectorMaxExtent];
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
    T lhs[kMaxSubgroupsPerWorkGroup][kRegisterMatrixLhsStages][kRegisterMatrixSubgroupTileM * kRegisterMatrixTileAStride];
    T rhs[kRegisterMatrixRhsStages][kRegisterMatrixTileK * kRegisterMatrixTileBStride];
};

template <typename T>
struct ComplexRank2kWorkspace {
    T lhs[kMaxSubgroupsPerWorkGroup][kComplexRank2kSubgroupTileM * kComplexRank2kTileAStride];
    T rhs[kComplexRank2kTileK * kComplexRank2kTileBStride];
};

template <typename T>
struct ComplexRank2kInKernelWorkspace {
    T lhs[kMaxSubgroupsPerWorkGroup][kComplexRank2kSubgroupTileM * kComplexRank2kInKernelTileAStride];
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

// GroupBlasWorkspace: a single union type that holds the local-memory workspace
// needed by any group-blas kernel function, so that all functions in a kernel
// share a single local-memory allocation rather than accumulating separate ones.
//
// The primary template covers types that do not have large fast-path kernels
// (e.g. double).  Specialisations for float and complex<float> add the larger
// workspace variants that are actually used for those element types.
template <typename T>
struct GroupBlasWorkspace {
    union {
        VectorWorkspace<T> vector_workspace;
        DenseGemvWorkspace<T> dense_gemv_workspace;
        ColumnSweepVectorWorkspace<T> column_sweep_workspace;
    };
};

template <>
struct GroupBlasWorkspace<float> {
    union {
        VectorWorkspace<float> vector_workspace;
        DenseGemvWorkspace<float> dense_gemv_workspace;
        ColumnSweepVectorWorkspace<float> column_sweep_workspace;
        GemmWorkspace<float> gemm_workspace;
    };
};

template <>
struct GroupBlasWorkspace<std::complex<float>> {
    union {
        VectorWorkspace<std::complex<float>> vector_workspace;
        DenseGemvWorkspace<std::complex<float>> dense_gemv_workspace;
        ColumnSweepVectorWorkspace<std::complex<float>> column_sweep_workspace;
        ComplexRank2kWorkspace<std::complex<float>> complex_rank2k_workspace;
        ComplexRank2kInKernelWorkspace<std::complex<float>> complex_rank2k_in_kernel_workspace;
    };
};

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
    return std::max(1, detail::item_local_linear_range(item) / std::max(1, sg_size));
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
    if (sg_size > kMaxSupportedSubgroupSize || subgroup_count(item) > kMaxSubgroupsPerWorkGroup) {
        return false;
    }

    const int inner_extent = detail::input_size(a, transform.trans);
    const int outer_extent = detail::output_size(a, transform.trans);
    return inner_extent >= sg_size && outer_extent >= 2;
}

template <typename T, typename Item>
inline constexpr bool can_use_dense_gemv_no_trans_fast_path(const Item& item,
                                                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                            DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    if (!subgroup_policy_matches(policy, sg_size)) {
        return false;
    }
    if (sg_size > kMaxSupportedSubgroupSize || subgroup_count(item) > kMaxSubgroupsPerWorkGroup) {
        return false;
    }

    return a.rows() >= sg_size && a.cols() >= sg_size;
}

template <typename T, typename Item>
inline constexpr bool can_use_trmv_no_trans_column_sweep_fast_path(const Item& item,
                                                                   const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                                   TriangularTransform transform,
                                                                   DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    if (!subgroup_policy_matches(policy, sg_size)) {
        return false;
    }
    if (transform.trans != Transpose::NoTrans) {
        return false;
    }
    if (sg_size > kMaxSupportedSubgroupSize || subgroup_count(item) > kMaxSubgroupsPerWorkGroup) {
        return false;
    }

    return a.rows() == a.cols() && a.rows() <= kColumnSweepVectorMaxExtent && a.rows() >= sg_size;
}

template <typename T, typename Item>
inline constexpr bool can_use_trmv_transpose_subgroup_dot_fast_path(const Item& item,
                                                                    const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                                    TriangularTransform transform,
                                                                    DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    if (!subgroup_policy_matches(policy, sg_size)) {
        return false;
    }
    if (transform.trans == Transpose::NoTrans) {
        return false;
    }
    if (sg_size > kMaxSupportedSubgroupSize || subgroup_count(item) > kMaxSubgroupsPerWorkGroup) {
        return false;
    }

    return a.rows() == a.cols() && a.rows() >= sg_size;
}

template <typename T, typename Item>
inline constexpr bool can_use_symv_no_trans_column_sweep_fast_path(const Item& item,
                                                                   const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                                   DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    if (!subgroup_policy_matches(policy, sg_size)) {
        return false;
    }
    if (sg_size > kMaxSupportedSubgroupSize || subgroup_count(item) > kMaxSubgroupsPerWorkGroup) {
        return false;
    }

    return a.rows() == a.cols() && a.rows() <= kColumnSweepVectorMaxExtent && a.rows() >= sg_size;
}

template <typename T, typename Item>
inline constexpr bool can_use_symv_no_trans_row_sweep_fast_path(const Item& item,
                                                                const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                                DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }

    const int local_size = detail::item_local_linear_range(item);
    const int extent = a.rows();
    return a.rows() == a.cols() && extent > 0 && extent <= 128 && local_size >= 32;
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
inline constexpr bool can_use_rank1_update_fast_path(const Item& item,
                                                     const VectorView<T>& x,
                                                     const Rank1UpdateOperand<T>& operand,
                                                     DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    if (policy == DeviceBlasPolicy::Auto || policy == DeviceBlasPolicy::Generic) {
        return false;
    }

    const int sg_size = subgroup_size(item);
    if (!subgroup_policy_matches(policy, sg_size)) {
        return false;
    }
    const int sg_count = subgroup_count(item);
    if (sg_size > kMaxSupportedSubgroupSize || sg_count > kMaxSubgroupsPerWorkGroup) {
        return false;
    }

    return x.size() > 0 && operand.y.size() > 0 && x.size() <= 64 && operand.y.size() <= 64;
}

template <typename T, typename Item>
inline constexpr bool can_use_rank2k_fast_path(const Item& item,
                                               const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                               const MatrixMatrixOperand<T>& operand,
                                               SymmetricRank2kTransform transform,
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
inline constexpr bool can_use_matrix_register_fast_path(const Item& item,
                                                        int row_extent,
                                                        int col_extent,
                                                        int contract_extent,
                                                        DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    if (policy == DeviceBlasPolicy::Generic) {
        return false;
    }
    if (detail::item_local_linear_range(item) != kRegisterMatrixThreadsPerGroup) {
        return false;
    }

    return row_extent >= kRegisterMatrixThreadTileRows &&
        col_extent >= kRegisterMatrixThreadTileCols &&
        contract_extent >= kRegisterMatrixTileK;
}

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
    if (detail::item_local_linear_range(item) != kOptimizedGemmThreadsPerGroup) {
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

template <typename T, typename Item>
inline constexpr bool can_use_complex_rank2k_in_kernel_fast_path(const Item& item,
                                                                 const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                                 const MatrixMatrixOperand<T>& operand,
                                                                 SymmetricRank2kTransform transform,
                                                                 DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, std::complex<float>>) {
        return false;
    }

    if constexpr (!std::is_same_v<std::remove_cvref_t<Item>, sycl::nd_item<1>>) {
        static_cast<void>(item);
        static_cast<void>(a);
        static_cast<void>(operand);
        static_cast<void>(transform);
        static_cast<void>(policy);
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::item_local_linear_range(item) != kComplexRank2kThreadsPerGroup) {
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
inline constexpr bool can_use_complex_rankk_in_kernel_fast_path(const Item& item,
                                                                const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                                const RankKOperand<T>& operand,
                                                                SymmetricRankKTransform transform,
                                                                DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, std::complex<float>>) {
        return false;
    }

    if constexpr (!std::is_same_v<std::remove_cvref_t<Item>, sycl::nd_item<1>>) {
        static_cast<void>(item);
        static_cast<void>(a);
        static_cast<void>(operand);
        static_cast<void>(transform);
        static_cast<void>(policy);
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::item_local_linear_range(item) != kComplexRank2kThreadsPerGroup) {
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
                                                             const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                             const MatrixMatrixOperand<T>& operand,
                                                             SymmetricRank2kTransform transform,
                                                             DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, std::complex<float>>) {
        return false;
    }

    if constexpr (!std::is_same_v<std::remove_cvref_t<Item>, sycl::nd_item<3>>) {
        static_cast<void>(item);
        static_cast<void>(a);
        static_cast<void>(operand);
        static_cast<void>(transform);
        static_cast<void>(policy);
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::item_local_linear_range(item) != kComplexRank2kThreadsPerGroup) {
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
inline constexpr bool can_use_complex_rankk_tiled_fast_path(const Item& item,
                                                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                            const RankKOperand<T>& operand,
                                                            SymmetricRankKTransform transform,
                                                            DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, std::complex<float>>) {
        return false;
    }

    if constexpr (!std::is_same_v<std::remove_cvref_t<Item>, sycl::nd_item<3>>) {
        static_cast<void>(item);
        static_cast<void>(a);
        static_cast<void>(operand);
        static_cast<void>(transform);
        static_cast<void>(policy);
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::item_local_linear_range(item) != kComplexRank2kThreadsPerGroup) {
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
inline constexpr bool can_use_rank2k_register_fast_path(const Item& item,
                                                        const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                        const MatrixMatrixOperand<T>& operand,
                                                        SymmetricRank2kTransform transform,
                                                        DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    if constexpr (!std::is_same_v<std::remove_cvref_t<Item>, sycl::nd_item<3>>) {
        static_cast<void>(item);
        static_cast<void>(a);
        static_cast<void>(operand);
        static_cast<void>(transform);
        static_cast<void>(policy);
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::item_local_linear_range(item) != kRegisterMatrixThreadsPerGroup) {
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
                                                       const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                       const RankKOperand<T>& operand,
                                                       SymmetricRankKTransform transform,
                                                       DeviceBlasPolicy policy) {
    if constexpr (!std::is_same_v<T, float>) {
        return false;
    }

    if constexpr (!std::is_same_v<std::remove_cvref_t<Item>, sycl::nd_item<3>>) {
        static_cast<void>(item);
        static_cast<void>(a);
        static_cast<void>(operand);
        static_cast<void>(transform);
        static_cast<void>(policy);
        return false;
    }

    if (policy != DeviceBlasPolicy::Auto) {
        return false;
    }
    if (detail::item_local_linear_range(item) != kRegisterMatrixThreadsPerGroup) {
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

template <typename T, typename... Ops>
inline constexpr void accumulate_cached_operands(std::array<T, sizeof...(Ops)>& partials,
                                                 const std::tuple<Ops...>& operands,
                                                 int input_index,
                                                 const T& a_ij,
                                                 const T& x0,
                                                 const T& x1) {
    if constexpr (sizeof...(Ops) == 1) {
        (void)operands;
        (void)input_index;
        (void)x1;
        partials[0] += a_ij * x0;
    } else {
        partials[0] += a_ij * x0;
        partials[1] += a_ij * x1;
        (void)operands;
        (void)input_index;
    }
}

template <typename Item, typename T>
inline constexpr void ger(const Item& item,
                          const VectorView<T>& x,
                          Rank1UpdateOperand<T> operand,
                          OuterProductTransform transform) {
    const auto sg = item.get_sub_group();
    const int sg_size = subgroup_size(item);
    const int lane = subgroup_local_id(item);
    const int sg_id = subgroup_group_id(item);
    const int total_sg = subgroup_count(item);
    const int row_extent = operand.a.rows();
    const int col_extent = operand.a.cols();
    const int rows_per_sg = rows_per_subgroup(row_extent, total_sg, kMaxMatrixRowsPerSubgroup);
    const int col_tiles = (col_extent + sg_size - 1) / sg_size;
    const int row_tiles = (row_extent + rows_per_sg - 1) / rows_per_sg;

    for (int linear_tile = sg_id; linear_tile < row_tiles * col_tiles; linear_tile += total_sg) {
        const int tile_row = linear_tile / col_tiles;
        const int tile_col = linear_tile % col_tiles;
        const int base_row = tile_row * rows_per_sg;
        const int col = tile_col * sg_size + lane;
        const T y_value = col < col_extent ? detail::maybe_conjugate(operand.y(col), transform.conjugate_y) : T(0);
        std::array<T, kMaxMatrixRowsPerSubgroup> x_values{};

        for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
            const int row = base_row + row_offset;
            const T x_lane = (lane == row_offset && row < row_extent)
                ? detail::maybe_conjugate(x(row), transform.conjugate_x)
                : T(0);
            x_values[static_cast<std::size_t>(row_offset)] =
                sycl::select_from_group(sg, x_lane, static_cast<uint32_t>(row_offset));
        }

        if (col < col_extent) {
            for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
                const int row = base_row + row_offset;
                if (row >= row_extent) {
                    continue;
                }
                detail::accumulate_rank1_output(operand, row, col, x_values[static_cast<std::size_t>(row_offset)] * y_value);
            }
        }
    }
}

template <typename Item, typename T>
inline constexpr void copy(const Item& item,
                           const VectorView<T>& x,
                           const VectorView<T>& y) {
    for_each_subgroup_vector_index(item, x.size(), [&](int index) {
        y(index) = x(index);
    });
}

template <typename Item, typename T>
inline constexpr void copyc(const Item& item,
                            const VectorView<T>& x,
                            const VectorView<T>& y) {
    for_each_subgroup_vector_index(item, x.size(), [&](int index) {
        y(index) = detail::conjugate_if_needed(x(index));
    });
}

template <typename Item, typename T>
inline constexpr void scal(const Item& item,
                           const VectorView<T>& x,
                           T alpha) {
    for_each_subgroup_vector_index(item, x.size(), [&](int index) {
        x(index) *= alpha;
    });
}

template <typename Item, typename T>
inline constexpr void axpy(const Item& item,
                           const VectorView<T>& x,
                           const VectorView<T>& y,
                           T alpha) {
    for_each_subgroup_vector_index(item, x.size(), [&](int index) {
        y(index) += alpha * x(index);
    });
}

template <typename Item, typename T>
inline constexpr void hadamard(const Item& item,
                               const VectorView<T>& x,
                               const VectorView<T>& y,
                               const VectorView<T>& z) {
    for_each_subgroup_vector_index(item, x.size(), [&](int index) {
        z(index) = x(index) * y(index);
    });
}

template <typename Item, typename T, typename... Ops>
inline constexpr void gemxv(const Item& item,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            MatrixVectorTransform transform,
                            const std::tuple<Ops...>& operands,
                            VectorWorkspace<T>* workspace) {
    const auto sg = item.get_sub_group();
    const int sg_size = subgroup_size(item);
    const int lane = subgroup_local_id(item);
    const int sg_id = subgroup_group_id(item);
    const int total_sg = subgroup_count(item);
    const int inner_extent = detail::input_size(a, transform.trans);
    const int outer_extent = detail::output_size(a, transform.trans);
    const int rows_per_sg = rows_per_subgroup(outer_extent, total_sg, kMaxVectorRowsPerSubgroup);
    T* x0_tile = workspace->operand_tiles[0] + sg_id * kVectorTileK;
    T* x1_tile = workspace->operand_tiles[1] + sg_id * kVectorTileK;

    for (int base_output = sg_id * rows_per_sg; base_output < outer_extent; base_output += total_sg * rows_per_sg) {
        std::array<std::array<T, sizeof...(Ops)>, kMaxVectorRowsPerSubgroup> partials{};

        for (int tile_begin = 0; tile_begin < inner_extent; tile_begin += kVectorTileK) {
            const int tile_extent = std::min(kVectorTileK, inner_extent - tile_begin);

            for (int tile_offset = lane; tile_offset < tile_extent; tile_offset += sg_size) {
                const int input_index = tile_begin + tile_offset;
                x0_tile[tile_offset] = std::get<0>(operands).x(input_index);
                if constexpr (sizeof...(Ops) == 2) {
                    x1_tile[tile_offset] = std::get<1>(operands).x(input_index);
                }
            }
            sycl::group_barrier(sg);

            for (int tile_offset = lane; tile_offset < tile_extent; tile_offset += sg_size) {
                const int input_index = tile_begin + tile_offset;
                const T x0 = x0_tile[tile_offset];
                const T x1 = sizeof...(Ops) == 2 ? x1_tile[tile_offset] : T(0);
                for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
                    const int output_index = base_output + row_offset;
                    if (output_index >= outer_extent) {
                        continue;
                    }
                    const T a_ij = detail::matrix_entry(a, output_index, input_index, transform.trans);
                    accumulate_cached_operands(partials[static_cast<std::size_t>(row_offset)], operands, input_index, a_ij, x0, x1);
                }
            }
            sycl::group_barrier(sg);
        }

        for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
            detail::reduce_partials(sg, partials[static_cast<std::size_t>(row_offset)], operands);
        }
        if (lane < rows_per_sg) {
            const int output_index = base_output + lane;
            if (output_index < outer_extent) {
                detail::write_outputs(operands, output_index, partials[static_cast<std::size_t>(lane)]);
            }
        }
    }
}

template <typename Item, typename T>
inline constexpr void gemv_dense_no_trans(const Item& item,
                                          const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                          const MatrixVectorOperand<T>& operand,
                                          DenseGemvWorkspace<T>* workspace) {
    constexpr int rows_per_lane = 2;
    const int linear_tid = detail::item_local_linear_id(item);
    const int local_size = detail::item_local_linear_range(item);
    const int sg_size = subgroup_size(item);
    const int sg_id = subgroup_group_id(item);
    const int total_sg = subgroup_count(item);
    const int row_tile = sg_size * rows_per_lane;
    const int row_stride = total_sg * row_tile;
    const int row_extent = a.rows();
    const int col_extent = a.cols();
    T* x_tile = workspace->x_tile;

    for (int row_base = sg_id * row_tile; row_base < row_extent; row_base += row_stride) {
        const int row0 = row_base + subgroup_local_id(item);
        const int row1 = row0 + sg_size;
        T accum0 = T(0);
        T accum1 = T(0);

        for (int col_base = 0; col_base < col_extent; col_base += kVectorTileK) {
            const int tile_extent = std::min(kVectorTileK, col_extent - col_base);

            for (int tile_offset = linear_tid; tile_offset < tile_extent; tile_offset += local_size) {
                x_tile[tile_offset] = operand.x(col_base + tile_offset);
            }
            sycl::group_barrier(item.get_group());

            if (row0 < row_extent) {
                for (int tile_offset = 0; tile_offset < tile_extent; ++tile_offset) {
                    accum0 += a(row0, col_base + tile_offset) * x_tile[tile_offset];
                }
            }
            if (row1 < row_extent) {
                for (int tile_offset = 0; tile_offset < tile_extent; ++tile_offset) {
                    accum1 += a(row1, col_base + tile_offset) * x_tile[tile_offset];
                }
            }
            sycl::group_barrier(item.get_group());
        }

        if (row0 < row_extent) {
            operand.y(row0) = operand.alpha * accum0 + operand.beta * operand.y(row0);
        }
        if (row1 < row_extent) {
            operand.y(row1) = operand.alpha * accum1 + operand.beta * operand.y(row1);
        }
    }
}

template <typename Item, typename T>
inline constexpr void trmv_no_trans_column_sweep(const Item& item,
                                                 const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                 const MatrixVectorOperand<T>& operand,
                                                 TriangularTransform transform,
                                                 ColumnSweepVectorWorkspace<T>* workspace) {
    const int linear_tid = detail::item_local_linear_id(item);
    const int local_size = detail::item_local_linear_range(item);
    const int extent = a.rows();
    T* x_tile = workspace->x_tile;
    if (extent <= 2 * local_size) {
        const int row0 = linear_tid;
        const int row1 = linear_tid + local_size;
        const bool has_row0 = row0 < extent;
        const bool has_row1 = row1 < extent;
        T accum0 = has_row0 ? operand.beta * operand.y(row0) : T(0);
        T accum1 = has_row1 ? operand.beta * operand.y(row1) : T(0);

        for (int col_base = 0; col_base < extent; col_base += kVectorTileK) {
            const int tile_extent = std::min(kVectorTileK, extent - col_base);

            for (int tile_offset = linear_tid; tile_offset < tile_extent; tile_offset += local_size) {
                x_tile[tile_offset] = operand.x(col_base + tile_offset);
            }
            sycl::group_barrier(item.get_group());

            for (int tile_offset = 0; tile_offset < tile_extent; ++tile_offset) {
                const int col = col_base + tile_offset;
                const T x_col = x_tile[tile_offset];
                if (has_row0 && ((transform.uplo == Uplo::Lower && row0 >= col) ||
                                 (transform.uplo == Uplo::Upper && row0 <= col))) {
                    const T value = (row0 == col && transform.diag == Diag::Unit) ? T(1) : a(row0, col);
                    accum0 += operand.alpha * value * x_col;
                }
                if (has_row1 && ((transform.uplo == Uplo::Lower && row1 >= col) ||
                                 (transform.uplo == Uplo::Upper && row1 <= col))) {
                    const T value = (row1 == col && transform.diag == Diag::Unit) ? T(1) : a(row1, col);
                    accum1 += operand.alpha * value * x_col;
                }
            }

            sycl::group_barrier(item.get_group());
        }

        if (has_row0) {
            operand.y(row0) = accum0;
        }
        if (has_row1) {
            operand.y(row1) = accum1;
        }
        return;
    }

    T* accum = workspace->accum;
    for (int row = linear_tid; row < extent; row += local_size) {
        accum[row] = operand.beta * operand.y(row);
    }
    sycl::group_barrier(item.get_group());

    for (int col_base = 0; col_base < extent; col_base += kVectorTileK) {
        const int tile_extent = std::min(kVectorTileK, extent - col_base);

        for (int tile_offset = linear_tid; tile_offset < tile_extent; tile_offset += local_size) {
            x_tile[tile_offset] = operand.x(col_base + tile_offset);
        }
        sycl::group_barrier(item.get_group());

        for (int tile_offset = 0; tile_offset < tile_extent; ++tile_offset) {
            const int col = col_base + tile_offset;
            const T x_col = x_tile[tile_offset];
            const int row_begin = transform.uplo == Uplo::Lower ? col : 0;
            const int row_end = transform.uplo == Uplo::Lower ? extent : col + 1;
            for (int row = row_begin + linear_tid; row < row_end; row += local_size) {
                const T value = (row == col && transform.diag == Diag::Unit) ? T(1) : a(row, col);
                accum[row] += operand.alpha * value * x_col;
            }
            sycl::group_barrier(item.get_group());
        }
    }

    for (int row = linear_tid; row < extent; row += local_size) {
        operand.y(row) = accum[row];
    }
}

template <typename Item, typename T>
inline constexpr void trmv_transpose_subgroup_dots(const Item& item,
                                                   const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                   const MatrixVectorOperand<T>& operand,
                                                   TriangularTransform transform) {
    const auto sg = item.get_sub_group();
    const int lane = subgroup_local_id(item);
    const int sg_id = subgroup_group_id(item);
    const int total_sg = subgroup_count(item);
    const int extent = a.rows();

    for (int output_index = sg_id; output_index < extent; output_index += total_sg) {
        const int begin = triangular_begin(output_index, extent, transform, Side::Left);
        const int end = triangular_end(output_index, extent, transform, Side::Left);
        T partial = T(0);

        for (int input_index = begin + lane; input_index < end; input_index += subgroup_size(item)) {
            partial += detail::triangular_matrix_entry(a, output_index, input_index, transform) * operand.x(input_index);
        }

        partial = sycl::reduce_over_group(sg, partial, sycl::plus<T>());
        if (lane == 0) {
            operand.y(output_index) = operand.alpha * partial + operand.beta * operand.y(output_index);
        }
    }
}

template <typename Item, typename T>
inline constexpr void symv_no_trans_row_sweep(const Item& item,
                                              const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                              const MatrixVectorOperand<T>& operand,
                                              SymmetricTransform transform) {
    const int linear_tid = detail::item_local_linear_id(item);
    const int local_size = detail::item_local_linear_range(item);
    const int extent = a.rows();

    for (int row = linear_tid; row < extent; row += local_size) {
        T accum = T(0);
        for (int col = 0; col < extent; ++col) {
            accum += detail::symmetric_matrix_entry(a, row, col, transform) * operand.x(col);
        }
        operand.y(row) = operand.alpha * accum + operand.beta * operand.y(row);
    }
}

template <typename Item, typename T>
inline constexpr void symv_no_trans_column_sweep(const Item& item,
                                                 const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                 const MatrixVectorOperand<T>& operand,
                                                 SymmetricTransform transform,
                                                 ColumnSweepVectorWorkspace<T>* workspace) {
    const int linear_tid = detail::item_local_linear_id(item);
    const int local_size = detail::item_local_linear_range(item);
    const int extent = a.rows();
    T* x_tile = workspace->x_tile;
    T* accum = workspace->accum;

    for (int row = linear_tid; row < extent; row += local_size) {
        accum[row] = operand.beta * operand.y(row);
    }
    sycl::group_barrier(item.get_group());

    for (int col_base = 0; col_base < extent; col_base += kVectorTileK) {
        const int tile_extent = std::min(kVectorTileK, extent - col_base);

        for (int tile_offset = linear_tid; tile_offset < tile_extent; tile_offset += local_size) {
            x_tile[tile_offset] = operand.x(col_base + tile_offset);
        }
        sycl::group_barrier(item.get_group());

        for (int tile_offset = 0; tile_offset < tile_extent; ++tile_offset) {
            const int col = col_base + tile_offset;
            const T x_col = x_tile[tile_offset];
            const int row_begin = transform.uplo == Uplo::Lower ? col : 0;
            const int row_end = transform.uplo == Uplo::Lower ? extent : col + 1;
            T col_partial = T(0);

            for (int row = row_begin + linear_tid; row < row_end; row += local_size) {
                const T value = a(row, col);
                if (row == col) {
                    col_partial += value * x_col;
                } else {
                    accum[row] += operand.alpha * value * x_col;
                    col_partial += value * operand.x(row);
                }
            }

            col_partial = sycl::reduce_over_group(item.get_group(), col_partial, sycl::plus<T>());
            if (linear_tid == 0) {
                accum[col] += operand.alpha * col_partial;
            }
            sycl::group_barrier(item.get_group());
        }
    }

    for (int row = linear_tid; row < extent; row += local_size) {
        operand.y(row) = accum[row];
    }
}

template <typename Item, typename T, typename... Ops>
inline constexpr void trmxv(const Item& item,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            TriangularTransform transform,
                            const std::tuple<Ops...>& operands,
                            VectorWorkspace<T>* workspace) {
    const auto sg = item.get_sub_group();
    const int sg_size = subgroup_size(item);
    const int lane = subgroup_local_id(item);
    const int sg_id = subgroup_group_id(item);
    const int total_sg = subgroup_count(item);
    const int extent = a.rows();
    const int rows_per_sg = rows_per_subgroup(extent, total_sg, kMaxVectorRowsPerSubgroup);
    T* x0_tile = workspace->operand_tiles[0] + sg_id * kVectorTileK;
    T* x1_tile = workspace->operand_tiles[1] + sg_id * kVectorTileK;

    for (int base_output = sg_id * rows_per_sg; base_output < extent; base_output += total_sg * rows_per_sg) {
        std::array<std::array<T, sizeof...(Ops)>, kMaxVectorRowsPerSubgroup> partials{};
        std::array<int, kMaxVectorRowsPerSubgroup> begins{};
        std::array<int, kMaxVectorRowsPerSubgroup> ends{};

        for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
            const int output_index = base_output + row_offset;
            if (output_index >= extent) {
                continue;
            }
            begins[static_cast<std::size_t>(row_offset)] = triangular_begin(output_index, extent, transform, Side::Left);
            ends[static_cast<std::size_t>(row_offset)] = triangular_end(output_index, extent, transform, Side::Left);
        }

        for (int tile_begin = 0; tile_begin < extent; tile_begin += kVectorTileK) {
            const int tile_extent = std::min(kVectorTileK, extent - tile_begin);

            for (int tile_offset = lane; tile_offset < tile_extent; tile_offset += sg_size) {
                const int input_index = tile_begin + tile_offset;
                x0_tile[tile_offset] = std::get<0>(operands).x(input_index);
                if constexpr (sizeof...(Ops) == 2) {
                    x1_tile[tile_offset] = std::get<1>(operands).x(input_index);
                }
            }
            sycl::group_barrier(sg);

            for (int tile_offset = lane; tile_offset < tile_extent; tile_offset += sg_size) {
                const int input_index = tile_begin + tile_offset;
                const T x0 = x0_tile[tile_offset];
                const T x1 = sizeof...(Ops) == 2 ? x1_tile[tile_offset] : T(0);
                for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
                    const int output_index = base_output + row_offset;
                    if (output_index >= extent) {
                        continue;
                    }
                    if (input_index < begins[static_cast<std::size_t>(row_offset)] ||
                        input_index >= ends[static_cast<std::size_t>(row_offset)]) {
                        continue;
                    }
                    const T a_ij = detail::triangular_matrix_entry(a, output_index, input_index, transform);
                    accumulate_cached_operands(partials[static_cast<std::size_t>(row_offset)], operands, input_index, a_ij, x0, x1);
                }
            }
            sycl::group_barrier(sg);
        }

        for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
            detail::reduce_partials(sg, partials[static_cast<std::size_t>(row_offset)], operands);
        }
        if (lane < rows_per_sg) {
            const int output_index = base_output + lane;
            if (output_index < extent) {
                detail::write_outputs(operands, output_index, partials[static_cast<std::size_t>(lane)]);
            }
        }
    }
}

template <typename Item, typename T, typename... Ops>
inline constexpr void symxv(const Item& item,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            SymmetricTransform transform,
                            const std::tuple<Ops...>& operands,
                            VectorWorkspace<T>* workspace) {
    const auto sg = item.get_sub_group();
    const int sg_size = subgroup_size(item);
    const int lane = subgroup_local_id(item);
    const int sg_id = subgroup_group_id(item);
    const int total_sg = subgroup_count(item);
    const int extent = a.rows();
    const int rows_per_sg = rows_per_subgroup(extent, total_sg, kMaxVectorRowsPerSubgroup);
    T* x0_tile = workspace->operand_tiles[0] + sg_id * kVectorTileK;
    T* x1_tile = workspace->operand_tiles[1] + sg_id * kVectorTileK;

    for (int base_output = sg_id * rows_per_sg; base_output < extent; base_output += total_sg * rows_per_sg) {
        std::array<std::array<T, sizeof...(Ops)>, kMaxVectorRowsPerSubgroup> partials{};

        for (int tile_begin = 0; tile_begin < extent; tile_begin += kVectorTileK) {
            const int tile_extent = std::min(kVectorTileK, extent - tile_begin);

            for (int tile_offset = lane; tile_offset < tile_extent; tile_offset += sg_size) {
                const int input_index = tile_begin + tile_offset;
                x0_tile[tile_offset] = std::get<0>(operands).x(input_index);
                if constexpr (sizeof...(Ops) == 2) {
                    x1_tile[tile_offset] = std::get<1>(operands).x(input_index);
                }
            }
            sycl::group_barrier(sg);

            for (int tile_offset = lane; tile_offset < tile_extent; tile_offset += sg_size) {
                const int input_index = tile_begin + tile_offset;
                const T x0 = x0_tile[tile_offset];
                const T x1 = sizeof...(Ops) == 2 ? x1_tile[tile_offset] : T(0);
                for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
                    const int output_index = base_output + row_offset;
                    if (output_index >= extent) {
                        continue;
                    }
                    const T a_ij = detail::symmetric_matrix_entry(a, output_index, input_index, transform);
                    accumulate_cached_operands(partials[static_cast<std::size_t>(row_offset)], operands, input_index, a_ij, x0, x1);
                }
            }
            sycl::group_barrier(sg);
        }

        for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
            detail::reduce_partials(sg, partials[static_cast<std::size_t>(row_offset)], operands);
        }
        if (lane < rows_per_sg) {
            const int output_index = base_output + lane;
            if (output_index < extent) {
                detail::write_outputs(operands, output_index, partials[static_cast<std::size_t>(lane)]);
            }
        }
    }
}

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
        std::array<T, kMaxMatrixRowsPerSubgroup> partials{};

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
inline constexpr void rank2k(const Item& item,
                             const KernelMatrixView<T, MatrixFormat::Dense>& a,
                             MatrixMatrixOperand<T> operand,
                             SymmetricRank2kTransform transform) {
    const auto sg = item.get_sub_group();
    const int sg_size = subgroup_size(item);
    const int lane = subgroup_local_id(item);
    const int sg_id = subgroup_group_id(item);
    const int total_sg = subgroup_count(item);
    const int extent = detail::output_size(a, transform.trans);
    const int contract_extent = detail::input_size(a, transform.trans);
    const Transpose rhs_transform = detail::rank2k_rhs_transform(transform);
    const T alpha2 = detail::secondary_rank2k_alpha(operand.alpha, transform.hermitian);
    const int rows_per_sg = rows_per_subgroup(extent, total_sg, kMaxMatrixRowsPerSubgroup);
    const int col_tiles = (extent + sg_size - 1) / sg_size;
    const int row_tiles = (extent + rows_per_sg - 1) / rows_per_sg;

    for (int linear_tile = sg_id; linear_tile < row_tiles * col_tiles; linear_tile += total_sg) {
        const int tile_row = linear_tile / col_tiles;
        const int tile_col = linear_tile % col_tiles;
        const int base_row = tile_row * rows_per_sg;
        const int col = tile_col * sg_size + lane;
        std::array<T, kMaxMatrixRowsPerSubgroup> partials{};

        for (int k = 0; k < contract_extent; ++k) {
            const T rhs1 = col < extent ? detail::matrix_entry(operand.b, k, col, rhs_transform) : T(0);
            const T rhs2 = col < extent ? detail::matrix_entry(a, k, col, rhs_transform) : T(0);

            for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
                const int row = base_row + row_offset;
                const T a_lane = (lane == row_offset && row < extent)
                    ? detail::matrix_entry(a, row, k, transform.trans)
                    : T(0);
                const T b_lane = (lane == row_offset && row < extent)
                    ? detail::matrix_entry(operand.b, row, k, transform.trans)
                    : T(0);
                const T lhs1 = sycl::select_from_group(sg, a_lane, static_cast<uint32_t>(row_offset));
                const T lhs2 = sycl::select_from_group(sg, b_lane, static_cast<uint32_t>(row_offset));

                if (col < extent && row < extent && detail::triangular_storage_contains(transform.uplo, row, col)) {
                    partials[static_cast<std::size_t>(row_offset)] += operand.alpha * lhs1 * rhs1 + alpha2 * lhs2 * rhs2;
                }
            }
        }

        if (col < extent) {
            for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
                const int row = base_row + row_offset;
                if (row >= extent || !detail::triangular_storage_contains(transform.uplo, row, col)) {
                    continue;
                }
                T value = operand.beta * operand.c(row, col) + partials[static_cast<std::size_t>(row_offset)];
                if constexpr (ComplexScalar<T>) {
                    if (transform.hermitian && row == col) {
                        value = T(value.real(), typename T::value_type(0));
                    }
                }
                operand.c(row, col) = value;
            }
        }
    }
}

template <typename Item, typename T>
inline constexpr void rankk(const Item& item,
                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                            RankKOperand<T> operand,
                            SymmetricRankKTransform transform) {
    const auto sg = item.get_sub_group();
    const int sg_size = subgroup_size(item);
    const int lane = subgroup_local_id(item);
    const int sg_id = subgroup_group_id(item);
    const int total_sg = subgroup_count(item);
    const int extent = detail::output_size(a, transform.trans);
    const int contract_extent = detail::input_size(a, transform.trans);
    const Transpose rhs_transform = detail::rankk_rhs_transform(transform);
    const int rows_per_sg = rows_per_subgroup(extent, total_sg, kMaxMatrixRowsPerSubgroup);
    const int col_tiles = (extent + sg_size - 1) / sg_size;
    const int row_tiles = (extent + rows_per_sg - 1) / rows_per_sg;

    for (int linear_tile = sg_id; linear_tile < row_tiles * col_tiles; linear_tile += total_sg) {
        const int tile_row = linear_tile / col_tiles;
        const int tile_col = linear_tile % col_tiles;
        const int base_row = tile_row * rows_per_sg;
        const int col = tile_col * sg_size + lane;
        std::array<T, kMaxMatrixRowsPerSubgroup> partials{};

        for (int k = 0; k < contract_extent; ++k) {
            const T rhs = col < extent ? detail::matrix_entry(a, k, col, rhs_transform) : T(0);

            for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
                const int row = base_row + row_offset;
                const T a_lane = (lane == row_offset && row < extent)
                    ? detail::matrix_entry(a, row, k, transform.trans)
                    : T(0);
                const T lhs = sycl::select_from_group(sg, a_lane, static_cast<uint32_t>(row_offset));

                if (col < extent && row < extent && detail::triangular_storage_contains(transform.uplo, row, col)) {
                    partials[static_cast<std::size_t>(row_offset)] += operand.alpha * lhs * rhs;
                }
            }
        }

        if (col < extent) {
            for (int row_offset = 0; row_offset < rows_per_sg; ++row_offset) {
                const int row = base_row + row_offset;
                if (row >= extent || !detail::triangular_storage_contains(transform.uplo, row, col)) {
                    continue;
                }
                T value = operand.beta * operand.c(row, col) + partials[static_cast<std::size_t>(row_offset)];
                if constexpr (ComplexScalar<T>) {
                    if (transform.hermitian && row == col) {
                        value = T(value.real(), typename T::value_type(0));
                    }
                }
                operand.c(row, col) = value;
            }
        }
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
inline constexpr void rank2k_complex_in_kernel_tiled(const Item& item,
                                                     const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                     MatrixMatrixOperand<T> operand,
                                                     SymmetricRank2kTransform transform,
                                                     ComplexRank2kInKernelWorkspace<T>* workspace) {
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

            accumulate_complex_rank2k_tiled_pass_impl<kComplexRank2kInKernelTileK, kComplexRank2kInKernelTileAStride>(
                item, workspace, linear_tid, subgroup_id, local_row, local_col, contract_extent, lhs1_loader, rhs1_loader, accum);
            accumulate_complex_rank2k_tiled_pass_impl<kComplexRank2kInKernelTileK, kComplexRank2kInKernelTileAStride>(
                item, workspace, linear_tid, subgroup_id, local_row, local_col, contract_extent, lhs2_loader, rhs2_loader, accum);

            write_complex_rank2k_tile(operand, transform, row_base, col_base, extent, accum);
        }
    }
}

template <typename Item, typename T>
inline constexpr void rankk_complex_in_kernel_tiled(const Item& item,
                                                    const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                                    RankKOperand<T> operand,
                                                    SymmetricRankKTransform transform,
                                                    ComplexRank2kInKernelWorkspace<T>* workspace) {
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

            accumulate_complex_rank2k_tiled_pass_impl<kComplexRank2kInKernelTileK, kComplexRank2kInKernelTileAStride>(
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

            accumulate_rank2k_register_tiled_pass(
                item, workspace, linear_tid, subgroup_id, contract_extent, lhs1_loader, rhs1_loader, accum);
            accumulate_rank2k_register_tiled_pass(
                item, workspace, linear_tid, subgroup_id, contract_extent, lhs2_loader, rhs2_loader, accum);

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

            accumulate_rank2k_register_tiled_pass(
                item, workspace, linear_tid, subgroup_id, contract_extent, lhs_loader, rhs_loader, accum);

            write_rank2k_register_tile(rank_operand, transform, row_base, col_base, extent, accum);
        }
    }
}

template <typename Item, typename T>
inline constexpr void gemm_register_tiled(const Item& item,
                                          const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                          MatrixMatrixOperand<T> operand,
                                          GeneralMatrixTransform transform,
                                          GemmWorkspace<T>* gemm_workspace) {
    const int linear_tid = detail::item_local_linear_id(item);
    const int subgroup_id = subgroup_group_id(item);
    const int local_row = linear_tid / kRegisterMatrixLocalCols;
    const int local_col = linear_tid % kRegisterMatrixLocalCols;
    const int row_extent = detail::output_size(a, transform.trans_a);
    const int col_extent = detail::input_size(operand.b, transform.trans_b);
    const int contract_extent = detail::input_size(a, transform.trans_a);
    const int row_tiles = (row_extent + kRegisterMatrixTileM - 1) / kRegisterMatrixTileM;
    const int col_tiles = (col_extent + kRegisterMatrixTileN - 1) / kRegisterMatrixTileN;
    const int tile_row_start = matrix_tile_group_row(item);
    const int tile_col_start = matrix_tile_group_col(item);
    const int tile_row_stride = matrix_tile_group_row_stride(item);
    const int tile_col_stride = matrix_tile_group_col_stride(item);
    auto* workspace = &gemm_workspace->register_workspace;

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
                    return detail::matrix_entry(a, global_row, global_k, transform.trans_a);
                };
                auto rhs_loader = [&](int tile_c, int global_k) {
                    const int global_col = tile_col * kRegisterMatrixTileN + tile_c;
                    if (global_col >= col_extent) {
                        return T(0);
                    }
                    return detail::matrix_entry(operand.b, global_k, global_col, transform.trans_b);
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

template <typename Item, typename T>
inline constexpr void gemm_aligned_nn_large(const Item& item,
                                            const KernelMatrixView<T, MatrixFormat::Dense>& a,
                                            MatrixMatrixOperand<T> operand,
                                            GemmWorkspace<T>* gemm_workspace) {
    const int linear_tid = detail::item_local_linear_id(item);
    const int local_row = linear_tid / kOptimizedGemmLocalCols;
    const int local_col = linear_tid % kOptimizedGemmLocalCols;
    const int row_extent = a.rows();
    const int col_extent = operand.b.cols();
    const int contract_extent = a.cols();
    const int row_tiles = row_extent / kOptimizedGemmTileM;
    const int col_tiles = col_extent / kRegisterMatrixTileN;
    const int tile_row_start = matrix_tile_group_row(item);
    const int tile_col_start = matrix_tile_group_col(item);
    const int tile_row_stride = matrix_tile_group_row_stride(item);
    const int tile_col_stride = matrix_tile_group_col_stride(item);
    auto* workspace = &gemm_workspace->optimized_workspace;

    for (int tile_row = tile_row_start; tile_row < row_tiles; tile_row += tile_row_stride) {
        for (int tile_col = tile_col_start; tile_col < col_tiles; tile_col += tile_col_stride) {
            const int row_base = tile_row * kOptimizedGemmTileM + local_row * kOptimizedGemmThreadTileRows;

            for (int subtile_col = 0; subtile_col < kRegisterMatrixTileN / kOptimizedGemmTileN; ++subtile_col) {
                const int col_tile_base = tile_col * kRegisterMatrixTileN + subtile_col * kOptimizedGemmTileN;
                const int col_base = col_tile_base + local_col * kOptimizedGemmThreadTileCols;
                T accum[kOptimizedGemmThreadTileRows][kOptimizedGemmThreadTileCols]{};

                auto load_stage = [&](int k_base, int stage) {
                    T* lhs_stage = workspace->lhs + stage * kOptimizedGemmStageASize;
                    T* rhs_stage = workspace->rhs + stage * kOptimizedGemmStageBSize;

                    constexpr int lhs_packets_per_col = kOptimizedGemmTileM / kOptimizedGemmVecA;
                    constexpr int lhs_packet_count = lhs_packets_per_col * kOptimizedGemmTileK;
                    constexpr int lhs_packet_iterations = lhs_packet_count / kOptimizedGemmThreadsPerGroup;
                    static_for<lhs_packet_iterations>([&](auto iter) {
                        constexpr int iteration = iter;
                        const int packet = linear_tid + iteration * kOptimizedGemmThreadsPerGroup;
                        const int lhs_row = (packet % lhs_packets_per_col) * kOptimizedGemmVecA;
                        const int lhs_col = packet / lhs_packets_per_col;
                        const int base = (k_base + lhs_col) * a.ld() + tile_row * kOptimizedGemmTileM + lhs_row;
                        const auto packet_values = packet_load_aligned<T, kOptimizedGemmVecA>(a.data(), base);
                        static_for<kOptimizedGemmVecA>([&](auto lane_idx) {
                            constexpr int lane = lane_idx;
                            lhs_stage[lhs_col * kOptimizedGemmTileAStride + lhs_row + lane] = packet_values[lane];
                        });
                    });

                    constexpr int rhs_packets_per_col = kOptimizedGemmTileK / kOptimizedGemmVecB;
                    constexpr int rhs_packet_count = kOptimizedGemmTileN * rhs_packets_per_col;
                    constexpr int rhs_packet_iterations = rhs_packet_count / kOptimizedGemmThreadsPerGroup;
                    static_for<rhs_packet_iterations>([&](auto iter) {
                        constexpr int iteration = iter;
                        const int packet = linear_tid + iteration * kOptimizedGemmThreadsPerGroup;
                        const int rhs_row = (packet % rhs_packets_per_col) * kOptimizedGemmVecB;
                        const int rhs_col = packet / rhs_packets_per_col;
                        const int base = (col_tile_base + rhs_col) * operand.b.ld() + k_base + rhs_row;
                        const auto packet_values = packet_load_aligned<T, kOptimizedGemmVecB>(operand.b.data(), base);
                        static_for<kOptimizedGemmVecB>([&](auto lane_idx) {
                            constexpr int lane = lane_idx;
                            rhs_stage[rhs_col * kOptimizedGemmTileBStride + rhs_row + lane] = packet_values[lane];
                        });
                    });
                };

                auto accumulate_stage = [&](const T* lhs_stage, const T* rhs_stage) {
                    for (int t0 = 0; t0 < kOptimizedGemmTileK; t0 += kOptimizedGemmUnrollK) {
                        for (int unroll = 0; unroll < kOptimizedGemmUnrollK; ++unroll) {
                            const int t = t0 + unroll;
                            T lhs_frag[kOptimizedGemmThreadTileRows];
                            T rhs_frag[kOptimizedGemmThreadTileCols];
                            for (int i = 0; i < kOptimizedGemmThreadTileRows; ++i) {
                                lhs_frag[i] = lhs_stage[t * kOptimizedGemmTileAStride + local_row * kOptimizedGemmThreadTileRows + i];
                            }
                            for (int j = 0; j < kOptimizedGemmThreadTileCols; ++j) {
                                rhs_frag[j] = rhs_stage[(local_col * kOptimizedGemmThreadTileCols + j) * kOptimizedGemmTileBStride + t];
                            }
                            for (int i = 0; i < kOptimizedGemmThreadTileRows; ++i) {
                                for (int j = 0; j < kOptimizedGemmThreadTileCols; ++j) {
                                    accum[i][j] += lhs_frag[i] * rhs_frag[j];
                                }
                            }
                        }
                    }
                };

                const int tile_count = contract_extent / kOptimizedGemmTileK;
                if (tile_count > 0) {
                    load_stage(0, 0);
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int tile_idx = 0; tile_idx + 1 < tile_count; ++tile_idx) {
                        const int current_stage = tile_idx & 1;
                        const int next_stage = current_stage ^ 1;
                        load_stage((tile_idx + 1) * kOptimizedGemmTileK, next_stage);
                        accumulate_stage(workspace->lhs + current_stage * kOptimizedGemmStageASize,
                                         workspace->rhs + current_stage * kOptimizedGemmStageBSize);
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    const int final_stage = (tile_count - 1) & 1;
                    accumulate_stage(workspace->lhs + final_stage * kOptimizedGemmStageASize,
                                     workspace->rhs + final_stage * kOptimizedGemmStageBSize);
                    item.barrier(sycl::access::fence_space::local_space);
                }

                for (int i = 0; i < kOptimizedGemmThreadTileRows; ++i) {
                    const int row = row_base + i;
                    for (int j = 0; j < kOptimizedGemmThreadTileCols; ++j) {
                        const int col = col_base + j;
                        detail::write_matrix_output(operand, row, col, accum[i][j]);
                    }
                }
            }
        }
    }
}

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
        std::array<T, kMaxMatrixRowsPerSubgroup> partials{};
        std::array<int, kMaxMatrixRowsPerSubgroup> begins{};
        std::array<int, kMaxMatrixRowsPerSubgroup> ends{};
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
                begins[static_cast<std::size_t>(row_offset)] =
                    triangular_begin(row, contract_extent, transform, Side::Left);
                ends[static_cast<std::size_t>(row_offset)] =
                    triangular_end(row, contract_extent, transform, Side::Left);
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
                        if (k < begins[static_cast<std::size_t>(row_offset)] ||
                            k >= ends[static_cast<std::size_t>(row_offset)]) {
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
        std::array<T, kMaxMatrixRowsPerSubgroup> partials{};

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
