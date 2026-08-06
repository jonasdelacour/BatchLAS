#pragma once

// A 128x128x8 register-tiled SYRK that visits only the half of C the caller
// asked for.
//
// Routing SYRK at a batched GEMM is correct but does twice the arithmetic BLAS
// specifies: C = alpha*A*A^T is symmetric, the caller names one triangle, and
// every 128x128 output tile strictly outside that triangle is computed and then
// discarded. Here the grid is indexed over the triangular tile set instead, so
// a tile outside the triangle is never launched. Tiles on the diagonal are the
// only ones computed in full, and their epilogue drops the elements that fall
// in the unreferenced half -- BLAS forbids writing them, and with beta != 0 it
// forbids reading them too.
//
// The tile decode packs row-major over the lower triangle: tile t maps to
// (bi, bj) with bj <= bi via the inverse of bi*(bi+1)/2. sqrt is only a seed
// here; the two correction loops make the result exact regardless of what the
// device's sqrt rounds to. Uplo::Upper is the same set with the pair swapped.
//
// The inner loop and shared-memory layout are those of
// src/sycl/gemm/register_128x128.hh, and for the same reasons: an aligned
// shared stride so the fragment loads become LDS.128, both operands staged
// [k][row] so a thread's 8 values are contiguous, and the 8x8 thread tile split
// into two 4-wide bands so an LDS.128 is bank-conflict free.
//
// What differs is staging. SYRK's two operands are the same matrix read at two
// different row offsets, so there is no separate B, and the access pattern
// depends only on the transpose mode:
//
//   NoTrans  A is n x k, so a column of A is contiguous in the output row
//            index and both tiles stage with one vector load per thread.
//   Trans    A is k x n, so the contiguous direction is k and both tiles stage
//            transposed, four consecutive k per thread scattered into shared.

#include "../queue.hh"
#include "../util/kernel-trace.hh"

#include <blas/enums.hh>
#include <blas/matrix.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::backend::detail {

template <typename T, bool TransOperand, bool AlignedFastPath>
class SyrkTriangularTilesKernel;

// A vector of four T with the alignment the 128-bit load/store forms need.
template <typename T>
struct alignas(4 * sizeof(T)) SyrkPacket4 {
    T v[4];
};

template <typename T>
inline const SyrkPacket4<T>& syrk_packet4(const T* p) {
    return *reinterpret_cast<const SyrkPacket4<T>*>(p);
}

template <typename T>
inline SyrkPacket4<T>& syrk_packet4(T* p) {
    return *reinterpret_cast<SyrkPacket4<T>*>(p);
}

inline constexpr int kSyrkTriangularTile = 128;
inline constexpr int kSyrkTriangularTileK = 8;

inline int syrk_tiles_per_side(int n) {
    return (n + kSyrkTriangularTile - 1) / kSyrkTriangularTile;
}

inline int syrk_triangular_tile_count(int n) {
    const int t = syrk_tiles_per_side(n);
    return t * (t + 1) / 2;
}

// Does this problem satisfy everything the unpredicated path assumes? A tile
// that is not a whole 128x128 of C, or a k the 8-deep staging cannot fill
// exactly, has to take the predicated path, as does any operand whose base or
// leading dimension breaks the 4-element alignment the packet forms need.
template <typename T>
bool syrk_triangular_fast_path(const MatrixView<T, MatrixFormat::Dense>& A,
                               const MatrixView<T, MatrixFormat::Dense>& C,
                               int n,
                               int k) {
    if ((n % kSyrkTriangularTile) != 0 || (k % kSyrkTriangularTileK) != 0) {
        return false;
    }
    auto aligned = [](const T* p, int ld, int stride) {
        return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) % (4 * sizeof(T))) == 0 &&
            (ld % 4) == 0 && (stride % 4) == 0;
    };
    return aligned(A.data_ptr(), A.ld(), A.stride()) &&
        aligned(C.data_ptr(), C.ld(), C.stride());
}

template <typename T, bool TransOperand, bool AlignedFastPath>
Event launch_syrk_triangular_tiles(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                   const MatrixView<T, MatrixFormat::Dense>& C,
                                   T alpha,
                                   T beta,
                                   Uplo uplo) {
    BATCHLAS_KERNEL_TRACE_SCOPE("syrk_cuda_custom.triangular_tiles");

    constexpr int TileM = kSyrkTriangularTile;
    constexpr int TileN = kSyrkTriangularTile;
    constexpr int TileK = kSyrkTriangularTileK;
    constexpr int ThreadTile = 8;                  // rows and cols per thread
    constexpr int Band = ThreadTile / 2;           // 4: the vectorized band width
    constexpr int LocalRows = TileM / ThreadTile;  // 16
    constexpr int LocalCols = TileN / ThreadTile;  // 16
    constexpr int Threads = LocalRows * LocalCols; // 256
    constexpr int AStride = TileM;
    constexpr int BStride = TileN;

    static_assert(TileM * TileK == 4 * Threads, "staging assumes one packet per thread");

    const int n = static_cast<int>(C.rows());
    const int k = TransOperand ? static_cast<int>(A.rows()) : static_cast<int>(A.cols());
    const int tile_count = syrk_triangular_tile_count(n);

    const sycl::range<3> local(1, LocalRows, LocalCols);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()),
                                static_cast<size_t>(tile_count * LocalRows),
                                static_cast<size_t>(LocalCols));

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> tile_a(sycl::range<1>(TileK * AStride), h);
        sycl::local_accessor<T, 1> tile_b(sycl::range<1>(TileK * BStride), h);

        const T* a_ptr = A.data_ptr();
        T* c_ptr = C.data_ptr();
        const int lda = A.ld();
        const int ldc = C.ld();
        const int stride_a = A.stride();
        const int stride_c = C.stride();
        const int batch = A.batch_size();
        const bool lower = uplo == Uplo::Lower;

        h.parallel_for<SyrkTriangularTilesKernel<T, TransOperand, AlignedFastPath>>(
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));
                if (bid >= batch) {
                    return;
                }
                // The m index is the fastest-varying one, so lanes that differ
                // in m touch consecutive addresses of a column-major C.
                const int ty = static_cast<int>(item.get_local_id(2));  // 0..15, m
                const int tx = static_cast<int>(item.get_local_id(1));  // 0..15, n
                const int tid = tx * LocalRows + ty;

                const int tile = static_cast<int>(item.get_group(1));
                int bi = static_cast<int>((sycl::sqrt(8.0 * tile + 1.0) - 1.0) * 0.5);
                while (bi > 0 && bi * (bi + 1) / 2 > tile) {
                    --bi;
                }
                while ((bi + 1) * (bi + 2) / 2 <= tile) {
                    ++bi;
                }
                const int bj = tile - bi * (bi + 1) / 2;
                const bool on_diagonal = bi == bj;

                const int m0 = (lower ? bi : bj) * TileM;
                const int n0 = (lower ? bj : bi) * TileN;

                const T* Ab = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
                T* Cb = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;

                T* sa = tile_a.template get_multi_ptr<sycl::access::decorated::no>().get();
                T* sb = tile_b.template get_multi_ptr<sycl::access::decorated::no>().get();

                T accum[ThreadTile][ThreadTile];
#pragma unroll
                for (int i = 0; i < ThreadTile; ++i) {
#pragma unroll
                    for (int j = 0; j < ThreadTile; ++j) {
                        accum[i][j] = T(0);
                    }
                }

                // Staging coordinates. NoTrans walks the output row index,
                // which is the contiguous direction of a column-major n x k A;
                // Trans walks k, which is the contiguous direction of a k x n A
                // and has to be transposed on the way into shared.
                const int s_row = TransOperand ? tid / 2 : (tid % 32) * 4;
                const int s_l = TransOperand ? (tid % 2) * 4 : tid / 32;

                for (int k0 = 0; k0 < k; k0 += TileK) {
                    if constexpr (AlignedFastPath) {
                        if constexpr (TransOperand) {
                            const SyrkPacket4<T> va =
                                syrk_packet4(Ab + (k0 + s_l) +
                                             static_cast<std::ptrdiff_t>(m0 + s_row) * lda);
                            const SyrkPacket4<T> vb =
                                syrk_packet4(Ab + (k0 + s_l) +
                                             static_cast<std::ptrdiff_t>(n0 + s_row) * lda);
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                sa[(s_l + i) * AStride + s_row] = va.v[i];
                                sb[(s_l + i) * BStride + s_row] = vb.v[i];
                            }
                        } else {
                            syrk_packet4(&sa[s_l * AStride + s_row]) =
                                syrk_packet4(Ab + (m0 + s_row) +
                                             static_cast<std::ptrdiff_t>(k0 + s_l) * lda);
                            syrk_packet4(&sb[s_l * BStride + s_row]) =
                                syrk_packet4(Ab + (n0 + s_row) +
                                             static_cast<std::ptrdiff_t>(k0 + s_l) * lda);
                        }
                    } else {
                        // The shared tiles are always filled to their full
                        // 128x8, with zeros outside the matrix, so the inner
                        // loop below needs no bounds checks at all.
                        if constexpr (TransOperand) {
                            const int gm = m0 + s_row;
                            const int gn = n0 + s_row;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int gk = k0 + s_l + i;
                                sa[(s_l + i) * AStride + s_row] =
                                    (gk < k && gm < n)
                                    ? Ab[gk + static_cast<std::ptrdiff_t>(gm) * lda]
                                    : T(0);
                                sb[(s_l + i) * BStride + s_row] =
                                    (gk < k && gn < n)
                                    ? Ab[gk + static_cast<std::ptrdiff_t>(gn) * lda]
                                    : T(0);
                            }
                        } else {
                            const int gk = k0 + s_l;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int gm = m0 + s_row + i;
                                const int gn = n0 + s_row + i;
                                sa[s_l * AStride + s_row + i] =
                                    (gm < n && gk < k)
                                    ? Ab[gm + static_cast<std::ptrdiff_t>(gk) * lda]
                                    : T(0);
                                sb[s_l * BStride + s_row + i] =
                                    (gn < n && gk < k)
                                    ? Ab[gn + static_cast<std::ptrdiff_t>(gk) * lda]
                                    : T(0);
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
                    for (int kk = 0; kk < TileK; ++kk) {
                        const SyrkPacket4<T> a0 = syrk_packet4(&sa[kk * AStride + ty * Band]);
                        const SyrkPacket4<T> a1 = syrk_packet4(&sa[kk * AStride + 64 + ty * Band]);
                        const SyrkPacket4<T> b0 = syrk_packet4(&sb[kk * BStride + tx * Band]);
                        const SyrkPacket4<T> b1 = syrk_packet4(&sb[kk * BStride + 64 + tx * Band]);
                        const T af[ThreadTile] = {a0.v[0], a0.v[1], a0.v[2], a0.v[3],
                                                  a1.v[0], a1.v[1], a1.v[2], a1.v[3]};
                        const T bf[ThreadTile] = {b0.v[0], b0.v[1], b0.v[2], b0.v[3],
                                                  b1.v[0], b1.v[1], b1.v[2], b1.v[3]};
#pragma unroll
                        for (int i = 0; i < ThreadTile; ++i) {
#pragma unroll
                            for (int j = 0; j < ThreadTile; ++j) {
                                accum[i][j] += af[i] * bf[j];
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Epilogue. Off the diagonal every element of the tile is in
                // the requested triangle, so a whole 4-row band is one 128-bit
                // access; on the diagonal the mask makes the band partial and
                // the stores go one at a time.
                auto store_element = [&](int row, int col, T value) {
                    if (row >= n || col >= n) {
                        return;
                    }
                    if (on_diagonal && (lower ? row < col : row > col)) {
                        return;
                    }
                    T* p = &Cb[row + static_cast<std::ptrdiff_t>(col) * ldc];
                    *p = beta == T(0) ? alpha * value : alpha * value + beta * *p;
                };

#pragma unroll
                for (int band = 0; band < 2; ++band) {
                    const int gm = m0 + band * 64 + ty * Band;
#pragma unroll
                    for (int j = 0; j < ThreadTile; ++j) {
                        const int gn = n0 + (j < Band ? tx * Band + j : 64 + tx * Band + j - Band);
                        if constexpr (AlignedFastPath) {
                            if (!on_diagonal) {
                                T* p = &Cb[gm + static_cast<std::ptrdiff_t>(gn) * ldc];
                                SyrkPacket4<T> out;
                                if (beta == T(0)) {
#pragma unroll
                                    for (int i = 0; i < 4; ++i) {
                                        out.v[i] = alpha * accum[band * Band + i][j];
                                    }
                                } else {
                                    const SyrkPacket4<T> prior = syrk_packet4(const_cast<const T*>(p));
#pragma unroll
                                    for (int i = 0; i < 4; ++i) {
                                        out.v[i] = alpha * accum[band * Band + i][j] +
                                            beta * prior.v[i];
                                    }
                                }
                                syrk_packet4(p) = out;
                                continue;
                            }
                        }
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            store_element(gm + i, gn, accum[band * Band + i][j]);
                        }
                    }
                }
            });
    });

    return ctx.get_event();
}

template <typename T>
Event syrk_triangular_tiles(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& A,
                            const MatrixView<T, MatrixFormat::Dense>& C,
                            T alpha,
                            T beta,
                            Uplo uplo,
                            Transpose transA) {
    const int n = static_cast<int>(C.rows());
    const int k = transA == Transpose::NoTrans ? static_cast<int>(A.cols())
                                               : static_cast<int>(A.rows());
    const bool fast = syrk_triangular_fast_path(A, C, n, k);

    if (transA == Transpose::NoTrans) {
        return fast ? launch_syrk_triangular_tiles<T, false, true>(ctx, A, C, alpha, beta, uplo)
                    : launch_syrk_triangular_tiles<T, false, false>(ctx, A, C, alpha, beta, uplo);
    }
    return fast ? launch_syrk_triangular_tiles<T, true, true>(ctx, A, C, alpha, beta, uplo)
                : launch_syrk_triangular_tiles<T, true, false>(ctx, A, C, alpha, beta, uplo);
}

} // namespace batchlas::backend::detail
