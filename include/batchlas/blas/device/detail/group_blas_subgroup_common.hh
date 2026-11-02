// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

#include <batchlas/blas/device/detail/group_blas_common.hh>

namespace batchlas::device {

namespace detail::subgroup {

inline constexpr int kMaxSupportedSubgroupSize = 32;
inline constexpr int kMaxSubgroupsPerWorkGroup = 8;
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

/// @brief Expansion step of static_for(): calls `fn` once per index in the pack, in order.
/// @param[in] fn  callable invoked as `fn(std::integral_constant<int, I>{})` for each `I`
template <int Count, typename Fn, std::size_t... I>
inline constexpr void static_for_impl(Fn&& fn, std::index_sequence<I...>) {
    (fn(std::integral_constant<int, static_cast<int>(I)>{}), ...);
}

/// @brief Compile-time loop: calls `fn` with `std::integral_constant<int, i>` for \f$i = 0, \dots, \mathrm{Count}-1\f$.
/// @param[in] fn  callable taking one `std::integral_constant<int, i>` argument
template <int Count, typename Fn>
inline constexpr void static_for(Fn&& fn) {
    static_for_impl<Count>(std::forward<Fn>(fn), std::make_index_sequence<Count>{});
}

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
    // Extents clamp to 1 (no zero-length arrays); a budget of 0 sub-groups means never launch for T.
    // evidence: docs/design/device-group-blas.md#device-group-blas-register-and-local-memory-caveats
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

/// @brief Whether a strided batch of matrices can be read with aligned `Width`-wide packet loads.
/// @param[in] ptr     base pointer of the first matrix
/// @param[in] ld      leading dimension of each matrix
/// @param[in] stride  element distance between consecutive matrices of the batch
/// @return true when `T` has a `Width`-wide packet type, `ptr` is aligned to it, and `ld` and
///         `stride` are multiples of `Width`; always false for types without a packet type
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

// Non-packet types (complex) only instantiate dead fast paths; sycl::vec<std::complex> is not a
// valid type outside DPC++, so they get a plain array.
template <typename T, int Width>
struct alignas(Width * sizeof(T)) PacketArray {
    T v[Width];
    /// @brief Element access, mirroring `sycl::vec::operator[]`.
    /// @param[in] i  element index, \f$0 \le i < \mathrm{Width}\f$
    /// @return a reference to element `i`
    constexpr const T& operator[](int i) const { return v[i]; }
};

template <typename T, int Width>
using Packet = std::conditional_t<supports_packet_v<T, Width>, sycl::vec<T, Width>, PacketArray<T, Width>>;

/// @brief Loads `Width` consecutive elements as one packet.
/// @param[in] ptr     base pointer
/// @param[in] offset  element offset from `ptr`; `ptr + offset` must be aligned to the packet type
///                    (see supports_aligned_packet_loads())
/// @return the packet holding `ptr[offset]`, ..., `ptr[offset + Width - 1]`
template <typename T, int Width>
inline constexpr Packet<T, Width> packet_load_aligned(const T* ptr, int offset) {
    return *reinterpret_cast<const Packet<T, Width>*>(ptr + offset);
}

/// @brief Local-memory stage that holds the left-hand-side panel of a contraction tile.
/// @param[in] tile_idx  index of the `kRegisterMatrixTileK`-wide contraction tile
/// @return `tile_idx` modulo `kRegisterMatrixLhsStages`
inline constexpr int register_matrix_lhs_stage(int tile_idx) {
    return tile_idx % kRegisterMatrixLhsStages;
}

/// @brief Local-memory stage that holds the right-hand-side panel of a contraction tile.
/// @param[in] tile_idx  index of the `kRegisterMatrixTileK`-wide contraction tile
/// @return `tile_idx` modulo `kRegisterMatrixRhsStages`
inline constexpr int register_matrix_rhs_stage(int tile_idx) {
    return tile_idx % kRegisterMatrixRhsStages;
}

/// @brief Output tile row a non-tiled executor starts at; always 0 (one group covers every tile).
/// @return 0
template <typename Item>
inline constexpr int matrix_tile_group_row(const Item&) {
    return 0;
}

/// @brief Output tile column a non-tiled executor starts at; always 0 (one group covers every tile).
/// @return 0
template <typename Item>
inline constexpr int matrix_tile_group_col(const Item&) {
    return 0;
}

/// @brief Output tile row step of a non-tiled executor; always 1.
/// @return 1
template <typename Item>
inline constexpr int matrix_tile_group_row_stride(const Item&) {
    return 1;
}

/// @brief Output tile column step of a non-tiled executor; always 1.
/// @return 1
template <typename Item>
inline constexpr int matrix_tile_group_col_stride(const Item&) {
    return 1;
}

/// @brief Output tile row a `sycl::nd_item<3>` work-group starts at.
/// @param[in] item  the work-item; dimension 1 of its group id indexes tile rows
/// @return `item.get_group(1)`
inline constexpr int matrix_tile_group_row(const sycl::nd_item<3>& item) {
    return static_cast<int>(item.get_group(1));
}

/// @brief Output tile column a `sycl::nd_item<3>` work-group starts at.
/// @param[in] item  the work-item; dimension 2 of its group id indexes tile columns
/// @return `item.get_group(2)`
inline constexpr int matrix_tile_group_col(const sycl::nd_item<3>& item) {
    return static_cast<int>(item.get_group(2));
}

/// @brief Output tile row step between iterations of a `sycl::nd_item<3>` work-group.
/// @param[in] item  the work-item
/// @return the group range in dimension 1, at least 1
inline constexpr int matrix_tile_group_row_stride(const sycl::nd_item<3>& item) {
    return std::max(1, static_cast<int>(item.get_group_range(1)));
}

/// @brief Output tile column step between iterations of a `sycl::nd_item<3>` work-group.
/// @param[in] item  the work-item
/// @return the group range in dimension 2, at least 1
inline constexpr int matrix_tile_group_col_stride(const sycl::nd_item<3>& item) {
    return std::max(1, static_cast<int>(item.get_group_range(2)));
}

/// @brief Sub-group size of the executing work-item.
/// @param[in] item  the work-item (an `nd_item`)
/// @return the local range of `item.get_sub_group()`
template <typename Item>
inline constexpr int subgroup_size(const Item& item) {
    return static_cast<int>(item.get_sub_group().get_local_range().size());
}

/// @brief Sub-group size a host-side launch description promises.
/// @param[in] launch  the launch description
/// @return `launch.subgroup_size`, at least 1
inline constexpr int subgroup_size(const DeviceBlasLaunchInfo& launch) {
    return std::max(1, launch.subgroup_size);
}

/// @brief Lane of the executing work-item within its sub-group.
/// @param[in] item  the work-item (an `nd_item`)
/// @return the sub-group local linear id, in \f$[0, \mathrm{sg\_size})\f$
template <typename Item>
inline constexpr int subgroup_local_id(const Item& item) {
    return static_cast<int>(item.get_sub_group().get_local_linear_id());
}

/// @brief Index of the executing work-item's sub-group within its work-group.
/// @param[in] item  the work-item (an `nd_item`)
/// @return the work-group local linear id divided by the sub-group size
template <typename Item>
inline constexpr int subgroup_group_id(const Item& item) {
    return detail::item_local_linear_id(item) / subgroup_size(item);
}

/// @brief Number of sub-groups in the executing work-group.
/// @param[in] item  the work-item (an `nd_item`)
/// @return the work-group size divided by the sub-group size, at least 1
template <typename Item>
inline constexpr int subgroup_count(const Item& item) {
    const int sg_size = subgroup_size(item);
    return std::max(1, detail::group_local_linear_range(item) / std::max(1, sg_size));
}

/// @brief Number of sub-groups per work-group a host-side launch description promises.
/// @param[in] launch  the launch description
/// @return `launch.local_size` divided by the sub-group size, at least 1
inline constexpr int subgroup_count(const DeviceBlasLaunchInfo& launch) {
    const int sg_size = subgroup_size(launch);
    return std::max(1, detail::group_local_linear_range(launch) / std::max(1, sg_size));
}

/// @brief Whether an executor is a `sycl::nd_item<3>` (tiled output launch).
/// @return true exactly when `Exec` is `sycl::nd_item<3>` after removing cv and reference qualifiers
template <typename Exec>
inline constexpr bool is_nd_item_3d_launch(const Exec&) {
    return std::is_same_v<std::remove_cvref_t<Exec>, sycl::nd_item<3>>;
}

/// @brief Whether a host-side launch description is a `sycl::nd_item<3>` (tiled output) launch.
/// @param[in] launch  the launch description
/// @return true when `launch.kind` is DeviceBlasLaunchKind::NdItem3D
inline constexpr bool is_nd_item_3d_launch(const DeviceBlasLaunchInfo& launch) {
    return launch.kind == DeviceBlasLaunchKind::NdItem3D;
}

/// @brief Output rows each sub-group handles per pass of a row-distributed kernel.
/// @param[in] extent    number of output rows
/// @param[in] total_sg  number of sub-groups in the work-group
/// @param[in] max_rows  largest rows-per-sub-group the kernel supports
/// @return `max_rows` when \f$\mathrm{extent} \ge \mathrm{total\_sg} \cdot \mathrm{max\_rows}\f$, else 2 when
///         `max_rows` is at least 4 and \f$\mathrm{extent} \ge 2\,\mathrm{total\_sg}\f$, else 1
inline constexpr int rows_per_subgroup(int extent, int total_sg, int max_rows) {
    if (extent >= total_sg * max_rows) {
        return max_rows;
    }
    if (max_rows >= 4 && extent >= total_sg * 2) {
        return 2;
    }
    return 1;
}

/// @brief Contraction tile width of the sub-group matrix-matrix fast path.
/// @param[in] contract_extent  length \f$k\f$ of the contraction dimension
/// @param[in] sg_size          sub-group size
/// @param[in] side             which operand is the structured one (Side::Left for the general case)
/// @return 32 (Side::Left) or 16 when \f$k \ge 256\f$ and the sub-group size is 32, else 16 when
///         \f$k \ge 128\f$, else 8
inline constexpr int matrix_tile_k(int contract_extent, int sg_size, Side side) {
    if (contract_extent >= 256 && sg_size == 32) {
        return side == Side::Left ? 32 : 16;
    }
    if (contract_extent >= 128) {
        return 16;
    }
    return 8;
}

/// @brief Output rows each sub-group handles per pass of the sub-group matrix-matrix fast path.
/// @param[in] row_extent       number of output rows
/// @param[in] total_sg         number of sub-groups in the work-group
/// @param[in] sg_size          sub-group size
/// @param[in] contract_extent  length \f$k\f$ of the contraction dimension
/// @param[in] side             which operand is the structured one (Side::Left for the general case)
/// @return 8 for Side::Left with sub-group size 32, \f$k \ge 128\f$ and at least 8 rows per sub-group,
///         otherwise rows_per_subgroup() with a cap of 4
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

/// @brief Whether a policy admits the sub-group fast paths at the actual sub-group size.
/// @param[in] policy       the requested implementation family
/// @param[in] actual_size  sub-group size the kernel runs with
/// @return false for DeviceBlasPolicy::Generic; for Subgroup16 / Subgroup32 true only at that size;
///         for Auto true at size 16 or 32
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

/// @brief Eligibility of the sub-group matrix-matrix fast path (`float` only).
/// @param[in] item             the work-item (or a DeviceBlasLaunchInfo for workspace queries)
/// @param[in] row_extent       number of output rows
/// @param[in] col_extent       number of output columns
/// @param[in] contract_extent  length \f$k\f$ of the contraction dimension
/// @param[in] policy           the requested implementation family
/// @return true when `T` is `float`, `policy` matches the sub-group size (at most
///         `kMaxSupportedSubgroupSize`), there are at least 8 rows, and both the column and
///         contraction extents are at least the sub-group size
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

/// @brief Eligibility of the register-tiled matrix-matrix path (`float` only).
/// @param[in] item             the work-item (or a DeviceBlasLaunchInfo for workspace queries)
/// @param[in] row_extent       number of output rows
/// @param[in] col_extent       number of output columns
/// @param[in] contract_extent  length \f$k\f$ of the contraction dimension
/// @param[in] policy           the requested implementation family; DeviceBlasPolicy::Generic rejects
/// @return true when RegisterMatrixWorkspace<T> fits the local-memory budget, `T` is `float`, the
///         work-group has `kRegisterMatrixThreadsPerGroup` work-items, and the extents cover one
///         thread tile and one `kRegisterMatrixTileK` contraction tile
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
inline constexpr bool can_use_matrix_aligned_nn_large_fast_path(const Item& item,
                                                                int row_extent,
                                                                int col_extent,
                                                                int contract_extent,
                                                                Transpose trans_a,
                                                                Transpose trans_b,
                                                                DeviceBlasPolicy policy);

/// @brief Eligibility of the packet-load NN GEMM path for large, tile-aligned `float` problems.
/// @param[in] item       the work-item (or a DeviceBlasLaunchInfo for workspace queries)
/// @param[in] a          the left operand \f$A\f$
/// @param[in] operand    the remaining GEMM operands; only `operand.b` is inspected
/// @param[in] transform  transposition of \f$A\f$ and \f$B\f$; both must be Transpose::NoTrans
/// @param[in] policy     the requested implementation family; DeviceBlasPolicy::Generic rejects
/// @return true when `T` is `float`, the work-group has `kOptimizedGemmThreadsPerGroup` work-items,
///         \f$m\f$ is a positive multiple of `kOptimizedGemmTileM`, \f$n\f$ is a multiple of
///         `kRegisterMatrixTileN` (not `kOptimizedGemmTileN`), \f$k\f$ is a positive multiple of
///         `kOptimizedGemmTileK`, and \f$A\f$ and \f$B\f$ admit aligned packet loads
///         (supports_aligned_packet_loads())
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

/// @brief Shape-only eligibility of the packet-load NN GEMM path (no pointer alignment check).
/// @param[in] item             the work-item (or a DeviceBlasLaunchInfo for workspace queries)
/// @param[in] row_extent       number of output rows \f$m\f$
/// @param[in] col_extent       number of output columns \f$n\f$
/// @param[in] contract_extent  length \f$k\f$ of the contraction dimension
/// @param[in] trans_a          transposition of \f$A\f$; must be Transpose::NoTrans
/// @param[in] trans_b          transposition of \f$B\f$; must be Transpose::NoTrans
/// @param[in] policy           the requested implementation family; DeviceBlasPolicy::Generic rejects
/// @return true when `T` is `float`, the work-group has `kOptimizedGemmThreadsPerGroup` work-items,
///         \f$m\f$ is a positive multiple of `kOptimizedGemmTileM`, \f$n\f$ is a multiple of
///         `kRegisterMatrixTileN`, and \f$k\f$ is a positive multiple of `kOptimizedGemmTileK`
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
inline constexpr bool can_use_complex_rank2k_tiled_fast_path(const Item& item,
                                                             int extent,
                                                             int contract_extent,
                                                             DeviceBlasPolicy policy);

/// @brief Eligibility of the tiled `std::complex<float>` rank-2k path.
/// @param[in] item       the work-item (or a DeviceBlasLaunchInfo for workspace queries); must be a
///                       `sycl::nd_item<3>` launch
/// @param[in] a          the operand \f$A\f$
/// @param[in] operand    the remaining rank-2k operands; `operand.c` must be \f$n \times n\f$
/// @param[in] transform  transposition and triangle of the update
/// @param[in] policy     the requested implementation family; only DeviceBlasPolicy::Auto admits
/// @return true when `T` is `std::complex<float>`, the work-group has `kComplexRank2kThreadsPerGroup`
///         work-items in sub-groups of 32 that fit ComplexRank2kWorkspace, \f$n \ge\f$
///         `kComplexRank2kTileN` and \f$k \ge 16\f$
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

/// @brief Shape-only eligibility of the tiled `std::complex<float>` rank-2k path.
/// @param[in] item             the work-item (or a DeviceBlasLaunchInfo for workspace queries); must
///                             be a `sycl::nd_item<3>` launch
/// @param[in] extent           order \f$n\f$ of the output matrix
/// @param[in] contract_extent  length \f$k\f$ of the contraction dimension
/// @param[in] policy           the requested implementation family; only DeviceBlasPolicy::Auto admits
/// @return true when `T` is `std::complex<float>`, the work-group has `kComplexRank2kThreadsPerGroup`
///         work-items in sub-groups of 32 that fit ComplexRank2kWorkspace, \f$n \ge\f$
///         `kComplexRank2kTileN` and \f$k \ge 16\f$
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

/// @brief Eligibility of the tiled `std::complex<float>` rank-k path.
/// @param[in] item       the work-item (or a DeviceBlasLaunchInfo for workspace queries); must be a
///                       `sycl::nd_item<3>` launch
/// @param[in] a          the operand \f$A\f$
/// @param[in] operand    the remaining rank-k operands; `operand.c` must be \f$n \times n\f$
/// @param[in] transform  transposition and triangle of the update
/// @param[in] policy     the requested implementation family; only DeviceBlasPolicy::Auto admits
/// @return true when `T` is `std::complex<float>`, the work-group has `kComplexRank2kThreadsPerGroup`
///         work-items in sub-groups of 32 that fit ComplexRank2kWorkspace, \f$n \ge\f$
///         `kComplexRank2kTileN` and \f$k \ge 16\f$
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

/// @brief Shape-only eligibility of the tiled `std::complex<float>` rank-k path.
/// @param[in] item             the work-item (or a DeviceBlasLaunchInfo for workspace queries); must
///                             be a `sycl::nd_item<3>` launch
/// @param[in] extent           order \f$n\f$ of the output matrix
/// @param[in] contract_extent  length \f$k\f$ of the contraction dimension
/// @param[in] policy           the requested implementation family; only DeviceBlasPolicy::Auto admits
/// @return true when `T` is `std::complex<float>`, the work-group has `kComplexRank2kThreadsPerGroup`
///         work-items in sub-groups of 32 that fit ComplexRank2kWorkspace, \f$n \ge\f$
///         `kComplexRank2kTileN` and \f$k \ge 16\f$
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

/// @brief Eligibility of the register-tiled `float` rank-2k path.
/// @param[in] item       the work-item (or a DeviceBlasLaunchInfo for workspace queries); must be a
///                       `sycl::nd_item<3>` launch
/// @param[in] a          the operand \f$A\f$
/// @param[in] operand    the remaining rank-2k operands; `operand.c` must be \f$n \times n\f$
/// @param[in] transform  transposition and triangle of the update
/// @param[in] policy     the requested implementation family; only DeviceBlasPolicy::Auto admits
/// @return true when `T` is `float`, the work-group has `kRegisterMatrixThreadsPerGroup` work-items
///         in at most `kMaxSubgroupsPerWorkGroup` sub-groups of 32, \f$n \ge\f$
///         `kRegisterMatrixTileN` and \f$k \ge 16\f$
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

/// @brief Shape-only eligibility of the register-tiled `float` rank-2k path.
/// @param[in] item             the work-item (or a DeviceBlasLaunchInfo for workspace queries); must
///                             be a `sycl::nd_item<3>` launch
/// @param[in] extent           order \f$n\f$ of the output matrix
/// @param[in] contract_extent  length \f$k\f$ of the contraction dimension
/// @param[in] policy           the requested implementation family; only DeviceBlasPolicy::Auto admits
/// @return true when `T` is `float`, the work-group has `kRegisterMatrixThreadsPerGroup` work-items
///         in at most `kMaxSubgroupsPerWorkGroup` sub-groups of 32, \f$n \ge\f$
///         `kRegisterMatrixTileN` and \f$k \ge 16\f$
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

/// @brief Eligibility of the register-tiled `float` rank-k path.
/// @param[in] item       the work-item (or a DeviceBlasLaunchInfo for workspace queries); must be a
///                       `sycl::nd_item<3>` launch
/// @param[in] a          the operand \f$A\f$
/// @param[in] operand    the remaining rank-k operands; `operand.c` must be \f$n \times n\f$
/// @param[in] transform  transposition and triangle of the update
/// @param[in] policy     the requested implementation family; only DeviceBlasPolicy::Auto admits
/// @return true when `T` is `float`, the work-group has `kRegisterMatrixThreadsPerGroup` work-items
///         in at most `kMaxSubgroupsPerWorkGroup` sub-groups of 32, \f$n \ge\f$
///         `kRegisterMatrixTileN` and \f$k \ge 16\f$
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

/// @brief Shape-only eligibility of the register-tiled `float` rank-k path.
/// @param[in] item             the work-item (or a DeviceBlasLaunchInfo for workspace queries); must
///                             be a `sycl::nd_item<3>` launch
/// @param[in] extent           order \f$n\f$ of the output matrix
/// @param[in] contract_extent  length \f$k\f$ of the contraction dimension
/// @param[in] policy           the requested implementation family; only DeviceBlasPolicy::Auto admits
/// @return true when `T` is `float`, the work-group has `kRegisterMatrixThreadsPerGroup` work-items
///         in at most `kMaxSubgroupsPerWorkGroup` sub-groups of 32, \f$n \ge\f$
///         `kRegisterMatrixTileN` and \f$k \ge 16\f$
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

/// @brief First contraction index where \f$\mathrm{op}(A)\f$ of a triangular operand is stored-nonzero.
/// @param[in] index      output row (Side::Left) or output column (Side::Right) being computed
/// @param[in] extent     order of the triangular matrix (unused; kept symmetric with triangular_end())
/// @param[in] transform  triangle and transposition of \f$A\f$
/// @param[in] side       whether \f$\mathrm{op}(A)\f$ multiplies from the left or the right
/// @return the inclusive lower bound of the contraction range, 0 or `index`
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

/// @brief One past the last contraction index where \f$\mathrm{op}(A)\f$ of a triangular operand is
///        stored-nonzero.
/// @param[in] index      output row (Side::Left) or output column (Side::Right) being computed
/// @param[in] extent     order of the triangular matrix
/// @param[in] transform  triangle and transposition of \f$A\f$
/// @param[in] side       whether \f$\mathrm{op}(A)\f$ multiplies from the left or the right
/// @return the exclusive upper bound of the contraction range, `index + 1` or `extent`
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

/// @brief First contraction index a register tile of a triangular product needs.
///
/// The tile spans `kRegisterMatrixTileM` output rows from `row_base` and `kRegisterMatrixTileN`
/// output columns from `col_base`; the bound is the union of triangular_begin() over the tile.
/// @param[in] row_base         first output row of the tile
/// @param[in] row_extent       number of output rows \f$m\f$
/// @param[in] col_base         first output column of the tile
/// @param[in] col_extent       number of output columns \f$n\f$
/// @param[in] contract_extent  length of the contraction dimension (unused; kept symmetric with
///                             triangular_tile_end())
/// @param[in] transform        side, triangle and transposition of \f$A\f$
/// @return the inclusive lower bound of the contraction range; 0 when `row_extent` (Side::Left) or
///         `col_extent` (Side::Right) is not positive. Tile bases past the extent are clamped to
///         the last row/column.
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

/// @brief One past the last contraction index a register tile of a triangular product needs.
///
/// The tile spans `kRegisterMatrixTileM` output rows from `row_base` and `kRegisterMatrixTileN`
/// output columns from `col_base`; the bound is the union of triangular_end() over the tile.
/// @param[in] row_base         first output row of the tile
/// @param[in] row_extent       number of output rows \f$m\f$
/// @param[in] col_base         first output column of the tile
/// @param[in] col_extent       number of output columns \f$n\f$
/// @param[in] contract_extent  length of the contraction dimension; the result never exceeds it
/// @param[in] transform        side, triangle and transposition of \f$A\f$
/// @return the exclusive upper bound of the contraction range, at most `contract_extent`; 0 when
///         `row_extent` (Side::Left) or `col_extent` (Side::Right) is not positive
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

/// @brief Stages one contraction tile of both operands into local memory for the register-tiled path.
///
/// Each sub-group loads its own `kRegisterMatrixSubgroupTileM` left-hand-side rows; the whole
/// work-group loads the shared `kRegisterMatrixTileN`-column right-hand-side panel. Entries past
/// `tile_extent` are zero-filled. No barrier is issued.
/// @param[in]  item         the work-item
/// @param[out] workspace    local-memory staging area; the selected stages are overwritten
/// @param[in]  linear_tid   work-group local linear id of the work-item
/// @param[in]  subgroup_id  sub-group index within the work-group
/// @param[in]  lhs_stage    left-hand-side stage to fill
/// @param[in]  rhs_stage    right-hand-side stage to fill
/// @param[in]  load_lhs     whether to fill the left-hand-side stage
/// @param[in]  load_rhs     whether to fill the right-hand-side stage
/// @param[in]  k_base       first contraction index of the tile
/// @param[in]  tile_extent  valid contraction indices in the tile, at most `kRegisterMatrixTileK`
/// @param[in]  lhs_loader   callable `(tile_row, k) -> T` returning the left-hand-side entry
/// @param[in]  rhs_loader   callable `(tile_col, k) -> T` returning the right-hand-side entry
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

/// @brief Accumulates one staged contraction tile into a work-item's register tile.
///
/// Operand fragments are broadcast across the sub-group with `select_from_group`, so every lane
/// of the sub-group must call this together.
/// @param[in]     item         the work-item
/// @param[in]     workspace    local-memory staging area filled by load_register_matrix_stage()
/// @param[in]     subgroup_id  sub-group index within the work-group
/// @param[in]     lhs_stage    left-hand-side stage to read
/// @param[in]     rhs_stage    right-hand-side stage to read
/// @param[in]     local_row    thread-tile row of the work-item in the work-group tile
/// @param[in]     local_col    thread-tile column of the work-item in the work-group tile
/// @param[in]     tile_extent  valid contraction indices in the stage, at most `kRegisterMatrixTileK`
/// @param[in,out] accum        `kRegisterMatrixThreadTileRows` \f$\times\f$ `kRegisterMatrixThreadTileCols`
///                             accumulator; the tile's partial product is added to it
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
            fragments.template lhs_value<i>() = batchlas::portable::select_from_group(sg, lhs_lane, static_cast<uint32_t>(lhs_source_lane));
        });

        static_for<kRegisterMatrixThreadTileCols>([&](auto col_idx) {
            constexpr int j = col_idx;
            const T rhs_lane = sg_row == 0 ? rhs_stage_ptr[kk * kRegisterMatrixTileBStride + rhs_col_base + j] : T(0);
            fragments.template rhs_value<j>() =
                batchlas::portable::select_from_group(sg, rhs_lane, static_cast<uint32_t>(rhs_source_lane_base));
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

/// @brief Writes a register tile to the output as \f$C := \alpha\,\mathrm{acc} + \beta\,C\f$.
/// @param[in,out] operand     matrix-matrix operands; `operand.c` is read and overwritten through its view
/// @param[in]     row_base    output row of the tile's first row
/// @param[in]     col_base    output column of the tile's first column
/// @param[in]     row_extent  number of output rows; rows at or past it are skipped
/// @param[in]     col_extent  number of output columns; columns at or past it are skipped
/// @param[in]     accum       the accumulated product tile
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

/// @brief Writes a register tile of a rank-k / rank-2k update to the stored triangle of \f$C\f$.
///
/// Computes \f$C := \beta\,C + \mathrm{acc}\f$ (the loaders fold \f$\alpha\f$ into `acc`) for
/// entries inside the `transform.uplo` triangle; a Hermitian update zeroes the imaginary part of
/// the diagonal.
/// @param[in,out] operand    rank-k / rank-2k operands; only `operand.c` and `operand.beta` are
///                           used, and `operand.c` is read and overwritten through its view
/// @param[in]     transform  triangle and Hermitian flag of the update
/// @param[in]     row_base   output row of the tile's first row
/// @param[in]     col_base   output column of the tile's first column
/// @param[in]     extent     order \f$n\f$ of \f$C\f$; entries at or past it are skipped
/// @param[in]     accum      the accumulated product tile
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

/// @brief Writes a complex rank-k / rank-2k accumulator tile to the stored triangle of \f$C\f$.
///
/// Computes \f$C := \beta\,C + \mathrm{acc}\f$ (the loaders fold \f$\alpha\f$ into `acc`) for
/// entries inside the `transform.uplo` triangle; a Hermitian update zeroes the imaginary part of
/// the diagonal.
/// @param[in,out] operand    rank-k / rank-2k operands; only `operand.c` and `operand.beta` are
///                           used, and `operand.c` is read and overwritten through its view
/// @param[in]     transform  triangle and Hermitian flag of the update
/// @param[in]     row_base   output row of the tile's first row
/// @param[in]     col_base   output column of the tile's first column
/// @param[in]     extent     order \f$n\f$ of \f$C\f$; entries at or past it are skipped
/// @param[in]     accum      the accumulated product tile
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

/// @brief Runs a full contraction through the register-tiled pipeline into one accumulator tile.
///
/// Loops over `kRegisterMatrixTileK`-wide contraction tiles, staging each with
/// load_register_matrix_stage() and accumulating it with accumulate_register_matrix_stage().
/// Contains work-group barriers, so every work-item of the group must call it.
/// @param[in]     item             the work-item
/// @param[out]    workspace        local-memory staging area; contents undefined on return
/// @param[in]     linear_tid       work-group local linear id of the work-item
/// @param[in]     subgroup_id      sub-group index within the work-group
/// @param[in]     contract_extent  length \f$k\f$ of the contraction dimension
/// @param[in]     lhs_loader       callable `(tile_row, k) -> T` returning the left-hand-side entry
/// @param[in]     rhs_loader       callable `(tile_col, k) -> T` returning the right-hand-side entry
/// @param[in,out] accum            accumulator; the full product over \f$k\f$ is added to it
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

/// @brief Runs a full contraction through the complex rank-2k tiled pipeline into one accumulator tile.
///
/// Loops over `TileK`-wide contraction tiles: each sub-group stages its left-hand-side rows, the
/// work-group stages the shared right-hand-side panel, and fragments are broadcast across the
/// sub-group. Contains work-group barriers, so every work-item of the group must call it.
/// @param[in]     item             the work-item
/// @param[out]    workspace        local-memory staging area (ComplexRank2kWorkspace or
///                                 ComplexRank2kInKernelWorkspace); contents undefined on return
/// @param[in]     linear_tid       work-group local linear id of the work-item
/// @param[in]     subgroup_id      sub-group index within the work-group
/// @param[in]     local_row        thread-tile row of the work-item in the work-group tile
/// @param[in]     local_col        thread-tile column of the work-item in the work-group tile
/// @param[in]     contract_extent  length \f$k\f$ of the contraction dimension
/// @param[in]     lhs_loader       callable `(tile_row, k) -> T` returning the left-hand-side entry
/// @param[in]     rhs_loader       callable `(tile_col, k) -> T` returning the right-hand-side entry
/// @param[in,out] accum            accumulator; the full product over \f$k\f$ is added to it
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
                    batchlas::portable::select_from_group(sg, lhs_lane, static_cast<uint32_t>(lhs_source_lane));
            });

            static_for<kComplexRank2kThreadTileCols>([&](auto col_idx) {
                constexpr int j = col_idx;
                const T rhs_lane = sg_row == 0 ? rhs_tile[kk * kComplexRank2kTileBStride + rhs_col_base + j] : T(0);
                rhs_frag[static_cast<std::size_t>(j)] =
                    batchlas::portable::select_from_group(sg, rhs_lane, static_cast<uint32_t>(rhs_source_lane_base));
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