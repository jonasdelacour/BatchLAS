// SPDX-License-Identifier: MIT
#pragma once
#include <cstddef>

#include <src/ops/gemm/kernels/accessors.hh>
#include <src/ops/gemm/kernels/epilogue_linear.hh>
#include <src/ops/gemm/kernels/load_policies.hh>

#include <src/ops/gemm/gemm_kernels.hh>

#include <src/linalg-impl.hh>

#include <batchlas/blas/device/v2/ext/gemm-components.hh>
#include <sycl/sycl.hpp>
#include <utility>

namespace batchlas::sycl_gemm {

namespace v2 = batchlas::device::v2;

template <int Count, typename Fn, std::size_t... I>
inline void static_for_impl(Fn&& fn, std::index_sequence<I...>) {
    (fn(std::integral_constant<int, static_cast<int>(I)>{}), ...);
}

template <int Count, typename Fn>
inline void static_for(Fn&& fn) {
    static_for_impl<Count>(std::forward<Fn>(fn), std::make_index_sequence<Count>{});
}

template <typename T,
          int TileM,
          int TileN,
          int TileK,
          int ThreadTileRows,
          int ThreadTileCols,
          int VecA,
          int VecB,
          int UnrollK,
          int Stages,
          bool AlignedFastPath,
          Transpose OpA,
          Transpose OpB>
class GemmRegisterTiledKernel;

template <typename T>
inline int register_ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

template <int TileM_,
          int TileN_,
          int TileK_,
          int ThreadTileRows_,
          int ThreadTileCols_ = ThreadTileRows_,
          int VecA_ = 4,
          int VecB_ = 4,
          int UnrollK_ = 1,
          int Stages_ = 1>
struct RegisterTilePolicy {
    static constexpr int TileM = TileM_;
    static constexpr int TileN = TileN_;
    static constexpr int TileK = TileK_;
    static constexpr int ThreadTileRows = ThreadTileRows_;
    static constexpr int ThreadTileCols = ThreadTileCols_;
    static constexpr int VecA = VecA_;
    static constexpr int VecB = VecB_;
    static constexpr int UnrollK = UnrollK_;
    static constexpr int Stages = Stages_;
    static constexpr int LocalRows = TileM / ThreadTileRows;
    static constexpr int LocalCols = TileN / ThreadTileCols;
    static constexpr int ThreadsPerGroup = LocalRows * LocalCols;
    static constexpr int TileAStride = TileM + 1;
    static constexpr int TileBStride = TileK + 1;
    static constexpr int StageASize = TileAStride * TileK;
    static constexpr int StageBSize = TileBStride * TileN;
    // B slot of (k row, n col). With ThreadTileCols 4 and an odd stride, local_col and
    // local_col + 8 read the same bank at every t (64-wide tiles: 2-way; 32x128: 4-way), and
    // no stride pad can fix that (4 * stride * 8 == 0 mod 32). XOR-ing the row with the col's
    // 32-column group stays inside the row's aligned 4-block, so it needs no extra memory.
    // evidence: docs/perf/gemm.md#gemm-wp-g-transpose-aware-scalar-staging
    static constexpr int b_slot(int col, int row) {
        if constexpr (TileN <= 32) return col * TileBStride + row;  // one group: the compiler cannot see q == 0
        else return col * TileBStride + (row ^ ((col >> 5) & 3));
    }
};

template <typename T>
inline bool is_contiguous_dense_matrix(const MatrixView<T, MatrixFormat::Dense>& M) {
    return M.ld() == M.rows() && M.stride() == M.ld() * M.cols();
}

template <typename T, int TileM, int TileN, int TileK, int VecA, int VecB>
inline bool can_use_aligned_nn_fast_path(const MatrixView<T, MatrixFormat::Dense>& A,
                                         const MatrixView<T, MatrixFormat::Dense>& B,
                                         const MatrixView<T, MatrixFormat::Dense>& C) {
    if constexpr (!supports_packet_v<T, VecA> || !supports_packet_v<T, VecB>) {
        static_cast<void>(A);
        static_cast<void>(B);
        static_cast<void>(C);
        return false;
    } else {
        return (A.rows() % TileM) == 0 &&
            (B.cols() % TileN) == 0 &&
            (A.cols() % TileK) == 0 &&
            is_contiguous_dense_matrix(A) &&
            is_contiguous_dense_matrix(B) &&
            is_contiguous_dense_matrix(C) &&
            supports_aligned_packet_loads<T, VecA>(A.data_ptr(), A.ld(), A.stride()) &&
            supports_aligned_packet_loads<T, VecB>(B.data_ptr(), B.ld(), B.stride());
    }
}

template <typename T,
          int TileM,
          int TileN,
          int TileK,
          int ThreadTileRows,
          int ThreadTileCols = ThreadTileRows,
          int VecA = 4,
          int VecB = 4,
          int UnrollK = 1,
          int Stages = 1,
          bool AlignedFastPath = false,
          Transpose OpA = Transpose::NoTrans,
          Transpose OpB = Transpose::NoTrans>
Event launch_register_tiled(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& A,
                            const MatrixView<T, MatrixFormat::Dense>& B,
                            const MatrixView<T, MatrixFormat::Dense>& C,
                            T alpha,
                            T beta,
                            const char* kernel_trace_name) {
    using Policy = RegisterTilePolicy<TileM, TileN, TileK, ThreadTileRows, ThreadTileCols, VecA, VecB, UnrollK, Stages>;

    BATCHLAS_KERNEL_TRACE_SCOPE(kernel_trace_name);

    static_assert(TileM % ThreadTileRows == 0, "TileM must divide evenly by ThreadTileRows");
    static_assert(TileN % ThreadTileCols == 0, "TileN must divide evenly by ThreadTileCols");
    static_assert(TileM * TileK >= Policy::ThreadsPerGroup, "TileA footprint must cover all threads");
    static_assert(TileN * TileK >= Policy::ThreadsPerGroup, "TileB footprint must cover all threads");
    static_assert(TileM % VecA == 0 && TileK % VecA == 0, "VecA must divide TileM and TileK");
    static_assert(TileN % VecB == 0 && TileK % VecB == 0, "VecB must divide TileN and TileK");
    static_assert(TileK % 4 == 0 && 32 % ThreadTileCols == 0, "b_slot's swizzle needs TileK % 4 == 0 and one group per thread");
    static_assert(UnrollK >= 1, "UnrollK must be positive");
    static_assert(Stages >= 1, "Stages must be positive");
    static_assert(Stages <= 2, "Register-tiled GEMM only supports one-stage and two-stage pipelines");
    static_assert(!AlignedFastPath || (OpA == Transpose::NoTrans && OpB == Transpose::NoTrans),
                  "Aligned fast path is only supported for NN kernels");

    const auto [m, k] = get_effective_dims(A, OpA);
    const auto [_, n] = get_effective_dims(B, OpB);
    static_cast<void>(_);
    const bool use_packet_a = AlignedFastPath ? true : supports_aligned_packet_loads<T, VecA>(A.data_ptr(), A.ld(), A.stride());
    const bool use_packet_b = AlignedFastPath ? true : supports_aligned_packet_loads<T, VecB>(B.data_ptr(), B.ld(), B.stride());

    const sycl::range<3> local(1, Policy::LocalRows, Policy::LocalCols);
    const int group_rows = AlignedFastPath ? (m / TileM) : register_ceil_div<T>(m, TileM);
    const int group_cols = AlignedFastPath ? (n / TileN) : register_ceil_div<T>(n, TileN);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()),
                                static_cast<size_t>(group_rows * Policy::LocalRows),
                                static_cast<size_t>(group_cols * Policy::LocalCols));

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> tile_a(sycl::range<1>(Policy::Stages * Policy::StageASize), h);
        sycl::local_accessor<T, 1> tile_b(sycl::range<1>(Policy::Stages * Policy::StageBSize), h);

        const T* a_ptr = A.data_ptr();
        const T* b_ptr = B.data_ptr();
        T* c_ptr = C.data_ptr();
        const int lda = A.ld();
        const int ldb = B.ld();
        const int ldc = C.ld();
        const int stride_a = A.stride();
        const int stride_b = B.stride();
        const int stride_c = C.stride();
        const int batch = A.batch_size();
        const std::uint8_t* mask = item_mask(A, B, C);
        const bool packet_a = use_packet_a;
        const bool packet_b = use_packet_b;

        batchlas::impl::parallel_for<
            GemmRegisterTiledKernel<T, TileM, TileN, TileK, ThreadTileRows, ThreadTileCols, VecA, VecB, UnrollK, Stages, AlignedFastPath, OpA, OpB>>(h,
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));
                const int local_row = static_cast<int>(item.get_local_id(1));
                const int local_col = static_cast<int>(item.get_local_id(2));
                if (bid >= batch || item_off(mask, bid)) {
                    return;
                }

                const int linear_tid = local_row * Policy::LocalCols + local_col;

                // Rebased per item in 64 bits (bid * stride passes 2^31); offsets within an item stay int.
                const T* const a_item = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
                const T* const b_item = b_ptr + static_cast<std::ptrdiff_t>(bid) * stride_b;
                T* const c_item = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;
                const int group_row = static_cast<int>(item.get_group(1));
                const int group_col = static_cast<int>(item.get_group(2));
                const v2::WorkGroupNd<Policy::ThreadsPerGroup> ex{item};
                T accum[ThreadTileRows][ThreadTileCols];
                v2::frag_fill<v2::Unroll::Compiler>(accum, T(0));

                // Stage op(A) (m x k; slot (r, c) at c * TileAStride + r) and op(B) (k x n; the
                // swizzled b_slot) of k slab kk0 into local stage `stage`. Scalar staging (odd ld or
                // base) walks the operand's contiguous index fastest; the shared layout is the same.
                auto load_stage = [&](int kk0, int stage) __attribute__((always_inline)) {
                    T* tile_a_stage = &tile_a[stage * Policy::StageASize];
                    T* tile_b_stage = &tile_b[stage * Policy::StageBSize];
                    auto dst_a = [&](int r, int c) -> T& { return tile_a_stage[c * Policy::TileAStride + r]; };
                    auto dst_b = [&](int r, int c) -> T& { return tile_b_stage[Policy::b_slot(c, r)]; };
                    const int m0 = group_row * TileM, n0 = group_col * TileN;
                    constexpr int Thr = Policy::ThreadsPerGroup;
                    if constexpr (AlignedFastPath && OpA == Transpose::NoTrans && OpB == Transpose::NoTrans
                                  && supports_packet_v<T, VecA> && supports_packet_v<T, VecB>) {
                        v2::stage_block_packet_aligned<TileM, TileK, VecA, Thr>(linear_tid, a_item, lda, m0, kk0, dst_a);
                        v2::stage_block_packet_aligned<TileK, TileN, VecB, Thr>(linear_tid, b_item, ldb, kk0, n0, dst_b);
                        return;
                    }
                    if constexpr (supports_packet_v<T, VecA> || supports_packet_v<T, VecB>) {
                        if (packet_a) v2::stage_block_packet<OpA, TileM, TileK, VecA, Thr>(linear_tid, a_item, lda, m0, kk0, m, k, dst_a);
                        else v2::stage_block<OpA, TileM, TileK, Thr>(linear_tid, a_item, lda, m0, kk0, m, k, dst_a);
                        if (packet_b) v2::stage_block_packet<OpB, TileK, TileN, VecB, Thr>(linear_tid, b_item, ldb, kk0, n0, k, n, dst_b);
                        else v2::stage_block<OpB, TileK, TileN, Thr>(linear_tid, b_item, ldb, kk0, n0, k, n, dst_b);
                    } else {
                        v2::stage_block<OpA, TileM, TileK, Thr>(linear_tid, a_item, lda, m0, kk0, m, k, dst_a);
                        v2::stage_block<OpB, TileK, TileN, Thr>(linear_tid, b_item, ldb, kk0, n0, k, n, dst_b);
                    }
                };

                // accum += A(:, t) B(t, :) over the slab's k steps (all TileK when aligned).
                auto accumulate_stage = [&](int stage, int kk0) {
                    const T* tile_a_stage = &tile_a[stage * Policy::StageASize];
                    const T* tile_b_stage = &tile_b[stage * Policy::StageBSize];
                    v2::mma_slab<TileK, UnrollK, AlignedFastPath>(
                        accum, kk0, k,
                        [&](int t, int i) { return tile_a_stage[t * Policy::TileAStride + local_row * ThreadTileRows + i]; },
                        [&](int t, int j) { return tile_b_stage[Policy::b_slot(local_col * ThreadTileCols + j, t)]; });
                };

                const int tile_count = AlignedFastPath ? (k / TileK) : register_ceil_div<T>(k, TileK);
                v2::kloop<Policy::Stages, AlignedFastPath, TileK>(ex, tile_count, load_stage, accumulate_stage);

                // C goes out through tile_a/tile_b, lane = consecutive row: lanes run along n here,
                // so a direct store puts adjacent lanes ldc apart. kloop ends on a barrier after its
                // last read of both tiles. The odd column stride spreads the TC-strided writes over the banks.
                // evidence: docs/perf/gemm.md#gemm-wp-b-coalesced-c-epilogue-on-the-predicated-legs
                v2::store_tile_staged<TileM, TileN, ThreadTileRows, ThreadTileCols, Policy::ThreadsPerGroup,
                                      Policy::Stages * Policy::StageASize, Policy::Stages * Policy::StageBSize>(
                    ex, linear_tid, local_row, local_col, accum, &tile_a[0], &tile_b[0], c_item, ldc,
                    group_row * TileM, group_col * TileN, m, n, alpha, beta);
            });
    });

    return ctx.get_event();
}

} // namespace batchlas::sycl_gemm