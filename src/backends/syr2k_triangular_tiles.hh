#pragma once

// A 128x128x8 register-tiled SYR2K that visits only the half of C the caller
// asked for, and fuses both of its rank-k products into one pass.
//
// C = alpha*A*B^T + alpha*B*A^T + beta*C is symmetric and the caller names one
// triangle, so decomposing it into two batched GEMMs aimed at C is wrong twice
// over: each GEMM writes the whole n x n, clobbering the half that belongs to
// the caller, and the pair streams C through memory three times -- written with
// beta, read back, written again -- for an output that is touched once here.
//
// The grid is indexed over the triangular tile set, so a tile lying outside the
// requested triangle is never launched. A tile on the diagonal is the only one
// computed in full, and its epilogue drops the elements in the unreferenced
// half: BLAS forbids writing them, and with beta != 0 it forbids reading them
// too, so the diagonal tile also gives up the 128-bit store form and goes one
// element at a time.
//
// Both products land in the same accumulators:
//
//     accum += A[bi] * B[bj]^T   and   accum += B[bi] * A[bj]^T
//
// which needs four staged tiles per k step instead of two, and pays for them
// with twice the arithmetic -- 8 x LDS.128 against 128 FFMA, the same 16:1
// ratio a plain GEMM tile reaches. The four 128x8 tiles are 16 KB of shared
// memory, well inside what lets two blocks share an SM.
//
// The two products are issued one after the other rather than interleaved, so
// that only one pair of 8-wide fragments is live at a time on top of the 64
// accumulators. Holding both pairs at once fits in a thread, but not inside the
// register budget that leaves room for a second block on the SM, and the
// occupancy that costs is worth 1.53x: 5.11 ms against 3.34 at n = 512 batch
// 512.
//
// The inner loop and shared-memory layout are those of
// src/sycl/gemm/register_128x128.hh, and for the same reasons: an aligned
// shared stride so the fragment loads become LDS.128, operands staged [k][row]
// so a thread's 8 values are contiguous, and the 8x8 thread tile split into two
// 4-wide bands so an LDS.128 is bank-conflict free. Staging follows the
// transpose mode:
//
//   NoTrans  A and B are n x k, so a column is contiguous in the output row
//            index and every tile stages with one vector load per thread.
//   Trans    A and B are k x n, so the contiguous direction is k and the tiles
//            stage transposed, four consecutive k per thread scattered into
//            shared.
//
// On the diagonal the two row offsets coincide, so the bj-side fragments are
// the bi-side ones and the kernel aliases the pointers rather than staging the
// same rows twice. on_diagonal is uniform across the block, so this costs no
// divergence.

#include "triangular_tiles.hh"

#include "../queue.hh"
#include "../util/kernel-trace.hh"

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::backend::detail {

template <typename T, bool TransOperand, bool AlignedFastPath>
class Syr2kTriangularTilesKernel;

// Does this problem satisfy everything the unpredicated path assumes? A tile
// that is not a whole 128x128 of C, or a k the 8-deep staging cannot fill
// exactly, has to take the predicated path, as does any operand whose base or
// leading dimension breaks the 4-element alignment the vector forms need.
template <typename T>
bool syr2k_triangular_fast_path(const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& B,
                                const MatrixView<T, MatrixFormat::Dense>& C,
                                int n,
                                int k) {
    if ((n % kTriangularTile) != 0 || (k % kTriangularTileK) != 0) {
        return false;
    }
    auto aligned = [](const T* p, int ld, int stride) {
        return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) % (4 * sizeof(T))) == 0 &&
            (ld % 4) == 0 && (stride % 4) == 0;
    };
    return aligned(A.data_ptr(), A.ld(), A.stride()) &&
        aligned(B.data_ptr(), B.ld(), B.stride()) &&
        aligned(C.data_ptr(), C.ld(), C.stride());
}

template <typename T, bool TransOperand, bool AlignedFastPath>
Event launch_syr2k_triangular_tiles(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    T alpha,
                                    T beta,
                                    Uplo uplo) {
    BATCHLAS_KERNEL_TRACE_SCOPE("syr2k_cuda_custom.triangular_tiles");

    constexpr int TileM = kTriangularTile;
    constexpr int TileN = kTriangularTile;
    constexpr int TileK = kTriangularTileK;
    constexpr int ThreadTile = 8;                  // rows and cols per thread
    constexpr int Band = ThreadTile / 2;           // 4: the vectorized band width
    constexpr int LocalRows = TileM / ThreadTile;  // 16
    constexpr int LocalCols = TileN / ThreadTile;  // 16
    constexpr int Threads = LocalRows * LocalCols; // 256
    constexpr int Stride = TileM;

    static_assert(TileM * TileK == 4 * Threads, "staging assumes one vector per thread");

    const int n = static_cast<int>(C.rows());
    const int k = TransOperand ? static_cast<int>(A.rows()) : static_cast<int>(A.cols());
    const int tile_count = triangular_tile_count(n);

    const sycl::range<3> local(1, LocalRows, LocalCols);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()),
                                static_cast<size_t>(tile_count * LocalRows),
                                static_cast<size_t>(LocalCols));

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> tiles(sycl::range<1>(4 * TileK * Stride), h);

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
        const bool lower = uplo == Uplo::Lower;

        h.parallel_for<Syr2kTriangularTilesKernel<T, TransOperand, AlignedFastPath>>(
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

                int bi = 0;
                int bj = 0;
                triangular_tile_decode(static_cast<int>(item.get_group(1)), bi, bj);
                const bool on_diagonal = bi == bj;

                const int m0 = (lower ? bi : bj) * TileM;
                const int n0 = (lower ? bj : bi) * TileN;

                const T* Ab = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
                const T* Bb = b_ptr + static_cast<std::ptrdiff_t>(bid) * stride_b;
                T* Cb = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;

                T* shared = tiles.template get_multi_ptr<sycl::access::decorated::no>().get();
                T* sam = shared;
                T* sbm = shared + TileK * Stride;
                T* san = on_diagonal ? sam : shared + 2 * TileK * Stride;
                T* sbn = on_diagonal ? sbm : shared + 3 * TileK * Stride;

                T accum[ThreadTile][ThreadTile];
#pragma unroll
                for (int i = 0; i < ThreadTile; ++i) {
#pragma unroll
                    for (int j = 0; j < ThreadTile; ++j) {
                        accum[i][j] = T(0);
                    }
                }

                // Staging coordinates. NoTrans walks the output row index,
                // which is the contiguous direction of a column-major n x k
                // operand; Trans walks k, which is the contiguous direction of
                // a k x n one and has to be transposed on the way into shared.
                const int s_row = TransOperand ? tid / 2 : (tid % 32) * 4;
                const int s_l = TransOperand ? (tid % 2) * 4 : tid / 32;

                for (int k0 = 0; k0 < k; k0 += TileK) {
                    // A shared tile is always filled to its full 128x8, with
                    // zeros outside the matrix, so the inner loop below needs
                    // no bounds checks at all.
                    auto stage = [&](const T* src, int ld, T* dst, int row0) {
                        if constexpr (TransOperand) {
                            const int row = row0 + s_row;
                            if constexpr (AlignedFastPath) {
                                const TileVec4<T> v =
                                    tile_vec4(src + (k0 + s_l) +
                                              static_cast<std::ptrdiff_t>(row) * ld);
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    dst[(s_l + i) * Stride + s_row] = v.v[i];
                                }
                            } else {
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    const int gk = k0 + s_l + i;
                                    dst[(s_l + i) * Stride + s_row] =
                                        (gk < k && row < n)
                                        ? src[gk + static_cast<std::ptrdiff_t>(row) * ld]
                                        : T(0);
                                }
                            }
                        } else {
                            const int gk = k0 + s_l;
                            if constexpr (AlignedFastPath) {
                                tile_vec4(&dst[s_l * Stride + s_row]) =
                                    tile_vec4(src + (row0 + s_row) +
                                              static_cast<std::ptrdiff_t>(gk) * ld);
                            } else {
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    const int row = row0 + s_row + i;
                                    dst[s_l * Stride + s_row + i] =
                                        (row < n && gk < k)
                                        ? src[row + static_cast<std::ptrdiff_t>(gk) * ld]
                                        : T(0);
                                }
                            }
                        }
                    };

                    stage(Ab, lda, sam, m0);
                    stage(Bb, ldb, sbm, m0);
                    if (!on_diagonal) {
                        stage(Ab, lda, san, n0);
                        stage(Bb, ldb, sbn, n0);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
                    for (int kk = 0; kk < TileK; ++kk) {
#pragma unroll
                        for (int product = 0; product < 2; ++product) {
                            const T* srow = product == 0 ? sam : sbm;
                            const T* scol = product == 0 ? sbn : san;
                            const TileVec4<T> r0 = tile_vec4(&srow[kk * Stride + ty * Band]);
                            const TileVec4<T> r1 = tile_vec4(&srow[kk * Stride + 64 + ty * Band]);
                            const TileVec4<T> c0 = tile_vec4(&scol[kk * Stride + tx * Band]);
                            const TileVec4<T> c1 = tile_vec4(&scol[kk * Stride + 64 + tx * Band]);
                            const T rf[ThreadTile] = {r0.v[0], r0.v[1], r0.v[2], r0.v[3],
                                                      r1.v[0], r1.v[1], r1.v[2], r1.v[3]};
                            const T cf[ThreadTile] = {c0.v[0], c0.v[1], c0.v[2], c0.v[3],
                                                      c1.v[0], c1.v[1], c1.v[2], c1.v[3]};
#pragma unroll
                            for (int i = 0; i < ThreadTile; ++i) {
#pragma unroll
                                for (int j = 0; j < ThreadTile; ++j) {
                                    accum[i][j] += rf[i] * cf[j];
                                }
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
                                TileVec4<T> out;
                                if (beta == T(0)) {
#pragma unroll
                                    for (int i = 0; i < 4; ++i) {
                                        out.v[i] = alpha * accum[band * Band + i][j];
                                    }
                                } else {
                                    const TileVec4<T> prior = tile_vec4(const_cast<const T*>(p));
#pragma unroll
                                    for (int i = 0; i < 4; ++i) {
                                        out.v[i] = alpha * accum[band * Band + i][j] +
                                            beta * prior.v[i];
                                    }
                                }
                                tile_vec4(p) = out;
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
Event syr2k_triangular_tiles(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             T alpha,
                             T beta,
                             Uplo uplo,
                             Transpose transA) {
    const int n = static_cast<int>(C.rows());
    const int k = transA == Transpose::NoTrans ? static_cast<int>(A.cols())
                                               : static_cast<int>(A.rows());
    const bool fast = syr2k_triangular_fast_path(A, B, C, n, k);

    if (transA == Transpose::NoTrans) {
        return fast
            ? launch_syr2k_triangular_tiles<T, false, true>(ctx, A, B, C, alpha, beta, uplo)
            : launch_syr2k_triangular_tiles<T, false, false>(ctx, A, B, C, alpha, beta, uplo);
    }
    return fast ? launch_syr2k_triangular_tiles<T, true, true>(ctx, A, B, C, alpha, beta, uplo)
                : launch_syr2k_triangular_tiles<T, true, false>(ctx, A, B, C, alpha, beta, uplo);
}

} // namespace batchlas::backend::detail
