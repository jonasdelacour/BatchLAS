#pragma once

#include "register_tiled_common.hh"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include <util/sycl-vector.hh>

namespace batchlas::sycl_gemm {

namespace {

template <typename T>
struct GemmRegisterTiledPersistentKernel;

constexpr int kExperimentalPersistentGroupsPerComputeUnit = 4;

template <typename T>
inline bool can_use_persistent_128x32x32_experimental(const MatrixView<T, MatrixFormat::Dense>& A,
                                                      const MatrixView<T, MatrixFormat::Dense>& B,
                                                      const MatrixView<T, MatrixFormat::Dense>& C,
                                                      Transpose transA,
                                                      Transpose transB) {
    if constexpr (!std::is_same_v<T, float>) {
        static_cast<void>(A);
        static_cast<void>(B);
        static_cast<void>(C);
        static_cast<void>(transA);
        static_cast<void>(transB);
        return false;
    } else {
        if (transA != Transpose::NoTrans || transB != Transpose::NoTrans) {
            return false;
        }

        const int m = A.rows();
        const int k = A.cols();
        const int n = B.cols();
        if (m < 256 || n < 256 || k < 256) {
            return false;
        }

        return can_use_aligned_nn_fast_path<T, 128, 32, 32, 4, 4>(A, B, C);
    }
}

template <typename T>
Event launch_register_128x32_k32_persistent(Queue& ctx,
                                            const MatrixView<T, MatrixFormat::Dense>& A,
                                            const MatrixView<T, MatrixFormat::Dense>& B,
                                            const MatrixView<T, MatrixFormat::Dense>& C,
                                            T alpha,
                                            T beta,
                                            Transpose transA,
                                            Transpose transB,
                                            const char* (*kernel_trace_name)(KernelVariant)) {
    if constexpr (!std::is_same_v<T, float>) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(B);
        static_cast<void>(C);
        static_cast<void>(alpha);
        static_cast<void>(beta);
        static_cast<void>(transA);
        static_cast<void>(transB);
        static_cast<void>(kernel_trace_name);
        throw std::runtime_error("Experimental persistent GEMM is currently only implemented for float");
    } else {
        if (!can_use_persistent_128x32x32_experimental(A, B, C, transA, transB)) {
            throw std::runtime_error("Experimental persistent GEMM requires large aligned NN float inputs");
        }

        using Policy = RegisterTilePolicy<128, 32, 32, 4, 4, 4, 4, 2, 2>;
        constexpr int TileM = 128;
        constexpr int TileN = 32;
        constexpr int TileK = 32;
        constexpr int ThreadTileRows = 4;
        constexpr int ThreadTileCols = 4;
        constexpr int VecA = 4;
        constexpr int VecB = 4;
        constexpr int UnrollK = 2;

        BATCHLAS_KERNEL_TRACE_SCOPE(kernel_trace_name(KernelVariant::Tiled128x32RegisterK32Persistent));

        const int m = A.rows();
        const int k = A.cols();
        const int n = B.cols();
        const int batch = A.batch_size();
        const int tile_rows = m / TileM;
        const int tile_cols = n / TileN;
        const int tiles_per_batch = tile_rows * tile_cols;
        const int total_tiles = batch * tiles_per_batch;
        if (total_tiles == 0) {
            return ctx.get_event();
        }

        const int compute_units = std::max<int>(1, static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_COMPUTE_UNITS)));
        const int persistent_groups = std::min(total_tiles, compute_units * kExperimentalPersistentGroupsPerComputeUnit);

        auto next_tile = std::make_shared<UnifiedVector<int>>(1, 0);
        const sycl::range<2> local(Policy::LocalRows, Policy::LocalCols);
        const sycl::range<2> global(static_cast<size_t>(persistent_groups * Policy::LocalRows),
                                    static_cast<size_t>(Policy::LocalCols));

        ctx->submit([&](sycl::handler& h) {
            sycl::local_accessor<T, 1> tile_a(sycl::range<1>(Policy::StageASize * Policy::Stages), h);
            sycl::local_accessor<T, 1> tile_b(sycl::range<1>(Policy::StageBSize * Policy::Stages), h);
            sycl::local_accessor<int, 1> claimed_tile(sycl::range<1>(1), h);

            const T* a_ptr = A.data_ptr();
            const T* b_ptr = B.data_ptr();
            T* c_ptr = C.data_ptr();
            const int lda = A.ld();
            const int ldb = B.ld();
            const int ldc = C.ld();
            const int stride_a = A.stride();
            const int stride_b = B.stride();
            const int stride_c = C.stride();
            int* next_tile_ptr = next_tile->data();

            h.parallel_for<GemmRegisterTiledPersistentKernel<T>>(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                const int local_row = static_cast<int>(item.get_local_id(0));
                const int local_col = static_cast<int>(item.get_local_id(1));
                const int linear_tid = local_row * Policy::LocalCols + local_col;

                while (true) {
                    if (linear_tid == 0) {
                        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            next_tile_ref(next_tile_ptr[0]);
                        claimed_tile[0] = next_tile_ref.fetch_add(1);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    const int flat_tile = claimed_tile[0];
                    if (flat_tile >= total_tiles) {
                        break;
                    }

                    const int bid = flat_tile / tiles_per_batch;
                    const int tile_index = flat_tile % tiles_per_batch;
                    const int tile_row = tile_index / tile_cols;
                    const int tile_col = tile_index % tile_cols;
                    const int row_base = tile_row * TileM + local_row * ThreadTileRows;
                    const int col_base = tile_col * TileN + local_col * ThreadTileCols;
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

                        constexpr int APacketsPerCol = TileM / VecA;
                        constexpr int APacketCount = APacketsPerCol * TileK;
                        for (int packet = linear_tid; packet < APacketCount; packet += Policy::ThreadsPerGroup) {
                            const int a_row = (packet % APacketsPerCol) * VecA;
                            const int a_col = packet / APacketsPerCol;
                            const int logical_a_row = tile_row * TileM + a_row;
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
                            const int logical_b_col = tile_col * TileN + b_col;
                            const int base = batch_b + logical_b_col * ldb + logical_b_row;
                            const auto packet_values = packet_load_aligned<T, VecB>(b_ptr, base);
                            for (int lane = 0; lane < VecB; ++lane) {
                                tile_b_stage[b_col * Policy::TileBStride + b_row + lane] = packet_values[lane];
                            }
                        }
                    };

                    auto accumulate_stage = [&](const T* tile_a_stage, const T* tile_b_stage) {
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
                    };

                    const int tile_count = k / TileK;
                    load_stage(0, 0);
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int tile_k = 0; tile_k < tile_count; ++tile_k) {
                        const int kk0 = tile_k * TileK;
                        const int current_stage = tile_k & 1;
                        const int next_k = tile_k + 1;
                        if (next_k < tile_count) {
                            load_stage(next_k * TileK, current_stage ^ 1);
                        }

                        accumulate_stage(&tile_a[current_stage * Policy::StageASize],
                                         &tile_b[current_stage * Policy::StageBSize]);
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    for (int i = 0; i < ThreadTileRows; ++i) {
                        const int row = row_base + i;
                        for (int j = 0; j < ThreadTileCols; ++j) {
                            const int col = col_base + j;
                            const T prior = c_ptr[batch_c + col * ldc + row];
                            c_ptr[batch_c + col * ldc + row] = LinearEpilogue<T>::apply(alpha, beta, accum[i][j], prior);
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }
            });
        });

        ctx->submit([next_tile](sycl::handler& h) {
            h.host_task([next_tile]() {});
        });

        return ctx.get_event();
    }
}

} // namespace

} // namespace batchlas::sycl_gemm