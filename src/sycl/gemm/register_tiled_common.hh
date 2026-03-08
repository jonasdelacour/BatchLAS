#pragma once

#include "accessors.hh"
#include "epilogue_linear.hh"
#include "load_policies.hh"

#include "../gemm_kernels.hh"

#include "../../linalg-impl.hh"

#include <sycl/sycl.hpp>

namespace batchlas::sycl_gemm {

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

template <int TileM,
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
constexpr KernelVariant register_kernel_variant() {
    if constexpr (TileM == 128 && TileN == 64 && TileK == 32 && ThreadTileRows == 8 && ThreadTileCols == 4) {
        if constexpr (UnrollK == 2 && Stages == 2) {
            return KernelVariant::Tiled128x64RegisterK32LargeU2;
        }
        return KernelVariant::Tiled128x64RegisterK32Large;
    }
    if constexpr (TileM == 128 && TileN == 64 && TileK == 32 && ThreadTileRows == 4 && ThreadTileCols == 8) {
        if constexpr (UnrollK == 2 && Stages == 2) {
            return KernelVariant::Tiled128x64RegisterK32LargeTT4x8U2;
        }
        return KernelVariant::Tiled128x64RegisterK32LargeTT4x8;
    }
    if constexpr (TileM == 128 && TileN == 32 && TileK == 32 && OpA == Transpose::NoTrans && OpB == Transpose::NoTrans) {
        if constexpr (Stages == 1 && UnrollK == 1 && ThreadTileRows == 4 && ThreadTileCols == 4 && VecA == 4 && VecB == 4) {
            return KernelVariant::Tiled128x32RegisterK32S1U1;
        }
        if constexpr (Stages == 2 && UnrollK == 1 && ThreadTileRows == 4 && ThreadTileCols == 4 && VecA == 4 && VecB == 4) {
            if constexpr (AlignedFastPath) {
                return KernelVariant::Tiled128x32RegisterK32S2U1Aligned;
            }
            return KernelVariant::Tiled128x32RegisterK32S2U1Generic;
        }
        if constexpr (Stages == 2 && UnrollK == 2 && ThreadTileRows == 4 && ThreadTileCols == 4 && VecA == 4 && VecB == 4) {
            return KernelVariant::Tiled128x32RegisterK32S2U2;
        }
        if constexpr (Stages == 2 && UnrollK == 2 && ThreadTileRows == 8 && ThreadTileCols == 4 && VecA == 4 && VecB == 4) {
            return KernelVariant::Tiled128x32RegisterK32S2U2TT8x4;
        }
        if constexpr (Stages == 2 && UnrollK == 2 && ThreadTileRows == 4 && ThreadTileCols == 8 && VecA == 4 && VecB == 4) {
            return KernelVariant::Tiled128x32RegisterK32S2U2TT4x8;
        }
        if constexpr (Stages == 1 && UnrollK == 4) {
            return KernelVariant::Tiled128x32RegisterK32S1U4;
        }
    }
    if constexpr (TileM == 64 && TileN == 64 && TileK == 16 && OpA == Transpose::Trans && OpB == Transpose::NoTrans) {
        return KernelVariant::Tiled64x64RegisterK16TN;
    }
    if constexpr (TileM == 64 && TileN == 64 && TileK == 16 && OpA == Transpose::NoTrans && OpB == Transpose::Trans) {
        return KernelVariant::Tiled64x64RegisterK16NT;
    }
    if constexpr (TileM == 64 && TileN == 64 && TileK == 16 && OpA == Transpose::Trans && OpB == Transpose::Trans) {
        return KernelVariant::Tiled64x64RegisterK16TT;
    }
    if constexpr (TileM == 128 && TileN == 64 && TileK == 16 && OpA == Transpose::Trans && OpB == Transpose::NoTrans) {
        return KernelVariant::Tiled128x64RegisterK16TN;
    }
    if constexpr (TileM == 128 && TileN == 64 && TileK == 16 && OpA == Transpose::NoTrans && OpB == Transpose::Trans) {
        return KernelVariant::Tiled128x64RegisterK16NT;
    }
    if constexpr (TileM == 128 && TileN == 64 && TileK == 16 && OpA == Transpose::Trans && OpB == Transpose::Trans) {
        return KernelVariant::Tiled128x64RegisterK16TT;
    }
    if constexpr (TileM == 128 && TileN == 32 && TileK == 32 && OpA == Transpose::Trans && OpB == Transpose::NoTrans) {
        return KernelVariant::Tiled128x32RegisterK32TN;
    }
    if constexpr (TileM == 128 && TileN == 32 && TileK == 32 && OpA == Transpose::NoTrans && OpB == Transpose::Trans) {
        return KernelVariant::Tiled128x32RegisterK32NT;
    }
    if constexpr (TileM == 128 && TileN == 32 && TileK == 32 && OpA == Transpose::Trans && OpB == Transpose::Trans) {
        return KernelVariant::Tiled128x32RegisterK32TT;
    }
    if constexpr (TileM == 128 && TileN == 32 && TileK == 16 && OpA == Transpose::Trans && OpB == Transpose::NoTrans) {
        return KernelVariant::Tiled128x32RegisterK16TN;
    }
    if constexpr (TileM == 32 && TileN == 128 && TileK == 16 && OpA == Transpose::Trans && OpB == Transpose::NoTrans) {
        return KernelVariant::Tiled32x128RegisterK16TN;
    }
    if constexpr (TileM == 128 && TileN == 32 && TileK == 16 && OpA == Transpose::NoTrans && OpB == Transpose::Trans) {
        return KernelVariant::Tiled128x32RegisterK16NT;
    }
    if constexpr (TileM == 128 && TileN == 32 && TileK == 16 && OpA == Transpose::Trans && OpB == Transpose::Trans) {
        return KernelVariant::Tiled128x32RegisterK16TT;
    }
    if constexpr (TileM == 32 && TileN == 128 && TileK == 16 && OpA == Transpose::Trans && OpB == Transpose::Trans) {
        return KernelVariant::Tiled32x128RegisterK16TT;
    }
    if constexpr (TileM == 32 && TileN == 32 && TileK == 8) {
        return KernelVariant::Tiled32x32Register;
    }
    if constexpr (TileM == 64 && TileN == 64 && TileK == 8) {
        return KernelVariant::Tiled64x64Register;
    }
    if constexpr (TileM == 64 && TileN == 64 && TileK == 16) {
        return KernelVariant::Tiled64x64RegisterK16;
    }
    if constexpr (TileM == 128 && TileN == 32 && TileK == 16) {
        return KernelVariant::Tiled128x32RegisterK16;
    }
    if constexpr (TileM == 128 && TileN == 32 && TileK == 32) {
        return KernelVariant::Tiled128x32RegisterK32;
    }
    if constexpr (TileM == 32 && TileN == 128 && TileK == 16) {
        return KernelVariant::Tiled32x128RegisterK16;
    }

    return KernelVariant::Tiled32x32Register;
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
                            const char* (*kernel_trace_name)(KernelVariant)) {
    using Policy = RegisterTilePolicy<TileM, TileN, TileK, ThreadTileRows, ThreadTileCols, VecA, VecB, UnrollK, Stages>;

    BATCHLAS_KERNEL_TRACE_SCOPE(kernel_trace_name(
        register_kernel_variant<TileM, TileN, TileK, ThreadTileRows, ThreadTileCols, VecA, VecB, UnrollK, Stages, AlignedFastPath, OpA, OpB>()));

    static_assert(TileM % ThreadTileRows == 0, "TileM must divide evenly by ThreadTileRows");
    static_assert(TileN % ThreadTileCols == 0, "TileN must divide evenly by ThreadTileCols");
    static_assert(TileM * TileK >= Policy::ThreadsPerGroup, "TileA footprint must cover all threads");
    static_assert(TileN * TileK >= Policy::ThreadsPerGroup, "TileB footprint must cover all threads");
    static_assert(TileM % VecA == 0 && TileK % VecA == 0, "VecA must divide TileM and TileK");
    static_assert(TileN % VecB == 0 && TileK % VecB == 0, "VecB must divide TileN and TileK");
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
        const bool packet_a = use_packet_a;
        const bool packet_b = use_packet_b;

        h.parallel_for<
            GemmRegisterTiledKernel<T, TileM, TileN, TileK, ThreadTileRows, ThreadTileCols, VecA, VecB, UnrollK, Stages, AlignedFastPath, OpA, OpB>>(
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));
                const int local_row = static_cast<int>(item.get_local_id(1));
                const int local_col = static_cast<int>(item.get_local_id(2));
                if (bid >= batch) {
                    return;
                }

                const int row_base = static_cast<int>(item.get_group(1)) * TileM + local_row * ThreadTileRows;
                const int col_base = static_cast<int>(item.get_group(2)) * TileN + local_col * ThreadTileCols;
                const int linear_tid = local_row * Policy::LocalCols + local_col;

                const int batch_a = bid * stride_a;
                const int batch_b = bid * stride_b;
                const int batch_c = bid * stride_c;
                T accum[ThreadTileRows][ThreadTileCols];
                for (int i = 0; i < ThreadTileRows; ++i) {
                    for (int j = 0; j < ThreadTileCols; ++j) {
                        accum[i][j] = T(0);
                    }
                }

                auto load_stage = [&](int kk0, int stage) {
                    T* tile_a_stage = &tile_a[stage * Policy::StageASize];
                    T* tile_b_stage = &tile_b[stage * Policy::StageBSize];

                    if constexpr (AlignedFastPath && OpA == Transpose::NoTrans && OpB == Transpose::NoTrans) {
                        constexpr int APacketsPerCol = TileM / VecA;
                        constexpr int APacketCount = APacketsPerCol * TileK;
                        for (int packet = linear_tid; packet < APacketCount; packet += Policy::ThreadsPerGroup) {
                            const int a_row = (packet % APacketsPerCol) * VecA;
                            const int a_col = packet / APacketsPerCol;
                            const int logical_a_row = static_cast<int>(item.get_group(1)) * TileM + a_row;
                            const int logical_a_col = kk0 + a_col;
                            const int base = batch_a + logical_a_col * lda + logical_a_row;
                            const auto packet_values = packet_load_aligned<T, VecA>(a_ptr, base);
                            for (int lane = 0; lane < VecA; ++lane) {
                                tile_a_stage[a_col * Policy::TileAStride + a_row + lane] = packet_values[lane];
                            }
                        }

                        constexpr int BPacketsPerCol = TileK / VecB;
                        constexpr int BPacketCount = TileN * BPacketsPerCol;
                        for (int packet = linear_tid; packet < BPacketCount; packet += Policy::ThreadsPerGroup) {
                            const int b_row = (packet % BPacketsPerCol) * VecB;
                            const int b_col = packet / BPacketsPerCol;
                            const int logical_b_row = kk0 + b_row;
                            const int logical_b_col = static_cast<int>(item.get_group(2)) * TileN + b_col;
                            const int base = batch_b + logical_b_col * ldb + logical_b_row;
                            const auto packet_values = packet_load_aligned<T, VecB>(b_ptr, base);
                            for (int lane = 0; lane < VecB; ++lane) {
                                tile_b_stage[b_col * Policy::TileBStride + b_row + lane] = packet_values[lane];
                            }
                        }
                        return;
                    }

                    if constexpr (supports_packet_v<T, VecA> || supports_packet_v<T, VecB>) {
                        if (packet_a) {
                            if constexpr (OpA == Transpose::NoTrans) {
                                constexpr int PacketsPerCol = TileM / VecA;
                                constexpr int PacketCount = PacketsPerCol * TileK;
                                for (int packet = linear_tid; packet < PacketCount; packet += Policy::ThreadsPerGroup) {
                                    const int a_row = (packet % PacketsPerCol) * VecA;
                                    const int a_col = packet / PacketsPerCol;
                                    const int logical_a_row = static_cast<int>(item.get_group(1)) * TileM + a_row;
                                    const int logical_a_col = kk0 + a_col;
                                    if (logical_a_row + (VecA - 1) < m && logical_a_col < k) {
                                        const int base = batch_a + logical_a_col * lda + logical_a_row;
                                        const auto packet_values = packet_load_aligned<T, VecA>(a_ptr, base);
                                        for (int lane = 0; lane < VecA; ++lane) {
                                            tile_a_stage[a_col * Policy::TileAStride + a_row + lane] = packet_values[lane];
                                        }
                                    } else {
                                        for (int lane = 0; lane < VecA; ++lane) {
                                            const int row = logical_a_row + lane;
                                            tile_a_stage[a_col * Policy::TileAStride + a_row + lane] = (row < m && logical_a_col < k)
                                                ? OperandAccessor<T, OpA>::load(a_ptr, lda, batch_a, row, logical_a_col)
                                                : T(0);
                                        }
                                    }
                                }
                            } else {
                                constexpr int PacketsPerRow = TileK / VecA;
                                constexpr int PacketCount = TileM * PacketsPerRow;
                                for (int packet = linear_tid; packet < PacketCount; packet += Policy::ThreadsPerGroup) {
                                    const int a_row = packet / PacketsPerRow;
                                    const int a_col = (packet % PacketsPerRow) * VecA;
                                    const int logical_a_row = static_cast<int>(item.get_group(1)) * TileM + a_row;
                                    const int logical_a_col = kk0 + a_col;
                                    if (logical_a_row < m && logical_a_col + (VecA - 1) < k) {
                                        const int base = batch_a + logical_a_row * lda + logical_a_col;
                                        const auto packet_values = packet_load_aligned<T, VecA>(a_ptr, base);
                                        for (int lane = 0; lane < VecA; ++lane) {
                                            tile_a_stage[(a_col + lane) * Policy::TileAStride + a_row] = packet_values[lane];
                                        }
                                    } else {
                                        for (int lane = 0; lane < VecA; ++lane) {
                                            const int col = logical_a_col + lane;
                                            tile_a_stage[(a_col + lane) * Policy::TileAStride + a_row] = (logical_a_row < m && col < k)
                                                ? OperandAccessor<T, OpA>::load(a_ptr, lda, batch_a, logical_a_row, col)
                                                : T(0);
                                        }
                                    }
                                }
                            }
                        } else {
                            for (int index = linear_tid; index < TileM * TileK; index += Policy::ThreadsPerGroup) {
                                const int a_row = index % TileM;
                                const int a_col = index / TileM;
                                const int logical_a_row = static_cast<int>(item.get_group(1)) * TileM + a_row;
                                const int logical_a_col = kk0 + a_col;
                                tile_a_stage[a_col * Policy::TileAStride + a_row] = (logical_a_row < m && logical_a_col < k)
                                    ? OperandAccessor<T, OpA>::load(a_ptr, lda, batch_a, logical_a_row, logical_a_col)
                                    : T(0);
                            }
                        }

                        if (packet_b) {
                            if constexpr (OpB == Transpose::NoTrans) {
                                constexpr int PacketsPerCol = TileK / VecB;
                                constexpr int PacketCount = TileN * PacketsPerCol;
                                for (int packet = linear_tid; packet < PacketCount; packet += Policy::ThreadsPerGroup) {
                                    const int b_row = (packet % PacketsPerCol) * VecB;
                                    const int b_col = packet / PacketsPerCol;
                                    const int logical_b_row = kk0 + b_row;
                                    const int logical_b_col = static_cast<int>(item.get_group(2)) * TileN + b_col;
                                    if (logical_b_row + (VecB - 1) < k && logical_b_col < n) {
                                        const int base = batch_b + logical_b_col * ldb + logical_b_row;
                                        const auto packet_values = packet_load_aligned<T, VecB>(b_ptr, base);
                                        for (int lane = 0; lane < VecB; ++lane) {
                                            tile_b_stage[b_col * Policy::TileBStride + b_row + lane] = packet_values[lane];
                                        }
                                    } else {
                                        for (int lane = 0; lane < VecB; ++lane) {
                                            const int row = logical_b_row + lane;
                                            tile_b_stage[b_col * Policy::TileBStride + b_row + lane] = (row < k && logical_b_col < n)
                                                ? OperandAccessor<T, OpB>::load(b_ptr, ldb, batch_b, row, logical_b_col)
                                                : T(0);
                                        }
                                    }
                                }
                            } else {
                                constexpr int PacketsPerRow = TileN / VecB;
                                constexpr int PacketCount = TileK * PacketsPerRow;
                                for (int packet = linear_tid; packet < PacketCount; packet += Policy::ThreadsPerGroup) {
                                    const int b_row = packet / PacketsPerRow;
                                    const int b_col = (packet % PacketsPerRow) * VecB;
                                    const int logical_b_row = kk0 + b_row;
                                    const int logical_b_col = static_cast<int>(item.get_group(2)) * TileN + b_col;
                                    if (logical_b_row < k && logical_b_col + (VecB - 1) < n) {
                                        const int base = batch_b + logical_b_row * ldb + logical_b_col;
                                        const auto packet_values = packet_load_aligned<T, VecB>(b_ptr, base);
                                        for (int lane = 0; lane < VecB; ++lane) {
                                            tile_b_stage[(b_col + lane) * Policy::TileBStride + b_row] = packet_values[lane];
                                        }
                                    } else {
                                        for (int lane = 0; lane < VecB; ++lane) {
                                            const int col = logical_b_col + lane;
                                            tile_b_stage[(b_col + lane) * Policy::TileBStride + b_row] = (logical_b_row < k && col < n)
                                                ? OperandAccessor<T, OpB>::load(b_ptr, ldb, batch_b, logical_b_row, col)
                                                : T(0);
                                        }
                                    }
                                }
                            }
                        } else {
                            for (int index = linear_tid; index < TileN * TileK; index += Policy::ThreadsPerGroup) {
                                const int b_row = index % TileK;
                                const int b_col = index / TileK;
                                const int logical_b_row = kk0 + b_row;
                                const int logical_b_col = static_cast<int>(item.get_group(2)) * TileN + b_col;
                                tile_b_stage[b_col * Policy::TileBStride + b_row] = (logical_b_row < k && logical_b_col < n)
                                    ? OperandAccessor<T, OpB>::load(b_ptr, ldb, batch_b, logical_b_row, logical_b_col)
                                    : T(0);
                            }
                        }
                    } else {
                        for (int index = linear_tid; index < TileM * TileK; index += Policy::ThreadsPerGroup) {
                            const int a_row = index % TileM;
                            const int a_col = index / TileM;
                            const int logical_a_row = static_cast<int>(item.get_group(1)) * TileM + a_row;
                            const int logical_a_col = kk0 + a_col;
                            tile_a_stage[a_col * Policy::TileAStride + a_row] = (logical_a_row < m && logical_a_col < k)
                                ? OperandAccessor<T, OpA>::load(a_ptr, lda, batch_a, logical_a_row, logical_a_col)
                                : T(0);
                        }

                        for (int index = linear_tid; index < TileN * TileK; index += Policy::ThreadsPerGroup) {
                            const int b_row = index % TileK;
                            const int b_col = index / TileK;
                            const int logical_b_row = kk0 + b_row;
                            const int logical_b_col = static_cast<int>(item.get_group(2)) * TileN + b_col;
                            tile_b_stage[b_col * Policy::TileBStride + b_row] = (logical_b_row < k && logical_b_col < n)
                                ? OperandAccessor<T, OpB>::load(b_ptr, ldb, batch_b, logical_b_row, logical_b_col)
                                : T(0);
                        }
                    }
                };

                auto accumulate_stage = [&](const T* tile_a_stage, const T* tile_b_stage, int kk0) {
                    if constexpr (AlignedFastPath) {
                        for (int t0 = 0; t0 < TileK; t0 += UnrollK) {
                            for (int unroll = 0; unroll < UnrollK; ++unroll) {
                                const int t = t0 + unroll;
                                T a_frag[ThreadTileRows];
                                T b_frag[ThreadTileCols];
                                for (int i = 0; i < ThreadTileRows; ++i) {
                                    a_frag[i] = tile_a_stage[t * Policy::TileAStride + local_row * ThreadTileRows + i];
                                }
                                for (int j = 0; j < ThreadTileCols; ++j) {
                                    b_frag[j] = tile_b_stage[(local_col * ThreadTileCols + j) * Policy::TileBStride + t];
                                }
                                for (int i = 0; i < ThreadTileRows; ++i) {
                                    for (int j = 0; j < ThreadTileCols; ++j) {
                                        accum[i][j] += a_frag[i] * b_frag[j];
                                    }
                                }
                            }
                        }
                    } else {
                        for (int t0 = 0; t0 < TileK && kk0 + t0 < k; t0 += UnrollK) {
                            for (int unroll = 0; unroll < UnrollK && t0 + unroll < TileK && kk0 + t0 + unroll < k; ++unroll) {
                                const int t = t0 + unroll;
                                T a_frag[ThreadTileRows];
                                T b_frag[ThreadTileCols];
                                for (int i = 0; i < ThreadTileRows; ++i) {
                                    a_frag[i] = tile_a_stage[t * Policy::TileAStride + local_row * ThreadTileRows + i];
                                }
                                for (int j = 0; j < ThreadTileCols; ++j) {
                                    b_frag[j] = tile_b_stage[(local_col * ThreadTileCols + j) * Policy::TileBStride + t];
                                }
                                for (int i = 0; i < ThreadTileRows; ++i) {
                                    for (int j = 0; j < ThreadTileCols; ++j) {
                                        accum[i][j] += a_frag[i] * b_frag[j];
                                    }
                                }
                            }
                        }
                    }
                };

                const int tile_count = AlignedFastPath ? (k / TileK) : register_ceil_div<T>(k, TileK);
                if constexpr (Policy::Stages == 1) {
                    for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
                        const int kk0 = tile_idx * TileK;
                        load_stage(kk0, 0);
                        item.barrier(sycl::access::fence_space::local_space);
                        accumulate_stage(&tile_a[0], &tile_b[0], kk0);
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                } else {
                    load_stage(0, 0);
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
                        const int kk0 = tile_idx * TileK;
                        const int current_stage = tile_idx & 1;
                        const int next_tile_idx = tile_idx + 1;

                        if (next_tile_idx < tile_count) {
                            load_stage(next_tile_idx * TileK, current_stage ^ 1);
                        }

                        accumulate_stage(&tile_a[current_stage * Policy::StageASize],
                                         &tile_b[current_stage * Policy::StageBSize],
                                         kk0);
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                }

                if constexpr (AlignedFastPath) {
                    for (int i = 0; i < ThreadTileRows; ++i) {
                        const int row = row_base + i;
                        for (int j = 0; j < ThreadTileCols; ++j) {
                            const int col = col_base + j;
                            const T prior = c_ptr[batch_c + col * ldc + row];
                            c_ptr[batch_c + col * ldc + row] = LinearEpilogue<T>::apply(alpha, beta, accum[i][j], prior);
                        }
                    }
                } else {
                    for (int i = 0; i < ThreadTileRows; ++i) {
                        const int row = row_base + i;
                        if (row >= m) {
                            continue;
                        }
                        for (int j = 0; j < ThreadTileCols; ++j) {
                            const int col = col_base + j;
                            if (col < n) {
                                const T prior = c_ptr[batch_c + col * ldc + row];
                                c_ptr[batch_c + col * ldc + row] = LinearEpilogue<T>::apply(alpha, beta, accum[i][j], prior);
                            }
                        }
                    }
                }
            });
    });

    return ctx.get_event();
}

} // namespace batchlas::sycl_gemm