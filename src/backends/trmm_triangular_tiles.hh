#pragma once

// A batched TRMM that actually skips the zero half of A.
//
// Until now nothing in this tree did. `src/extensions/trmm.cc` recurses only
// until the block is 256 wide and then calls the very GEMM it is meant to
// replace, so for any ib in {16,32,64,128,256} -- which is every block size
// ormqr and ormbr use -- the triangular structure was never exploited at all.
// On CUDA the operation does not even reach that recursion: `trmm_vendor_impl`
// expands the triangle into a k x k x batch scratch buffer and hands the result
// to a full GEMM, paying an extra write and an extra read of the expansion for
// the privilege of doing twice the arithmetic. PR #61 measured the consequence
// and correctly refused to use trmm anywhere.
//
// The structure worth exploiting is in the k loop, not in a mask. For
// C = alpha * op(A) * B with op(A) upper triangular, output row i only ever
// touches op(A)_{i,p} for p >= i, so an output tile starting at row m0 can
// start its reduction at p = m0 and skip everything before it -- a loop bound
// rather than a predicate, so the arithmetic is not done and then discarded.
// Only the one k-tile straddling the diagonal needs masking, and that mask
// lives in the A staging where the tile is small and read once.
//
// How much that saves depends entirely on how many row tiles m covers, and it
// is worth being precise because it is not the textbook 2x: with R row tiles
// the reduction shrinks to (R+1)/2R of the square, so 1.0x at R = 1, 1.33x at
// R = 2, and only 1.78x by R = 8. `trmm_prefer_triangular_tiles` carries the
// measured consequence -- this kernel is ahead of the vendor at m <= 64 (on
// fusion, not arithmetic) and at m >= 512, and behind it in between.
//
// Which end of the reduction is skipped comes from uplo and trans together:
// transposing an upper triangle gives a lower one, so the kernel keys off
// `lower_eff = (uplo == Lower) != transposed` and nothing else.
//
// The tile is sized to m rather than fixed at 128, for the same reason the Gram
// SYRK kernel next door sizes its tile to n: ormqr's W2 = T^H W1 has m = ib in
// the tens, and a 128-row tile would spend four times the arithmetic on it.
//
// Shared layout, staging and the packet swizzle are those of
// syrk_gram_tiles.hh -- an aligned stride so the fragment loads issue as
// LDS.128, and a packet rotation by reduction-row group so the staging writes
// that have to transpose do not all land in the same bank.

#include "triangular_tiles.hh"

#include "../queue.hh"
#include "../util/kernel-trace.hh"

#include <blas/enums.hh>
#include <blas/matrix.hh>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::backend::detail {

inline constexpr int kTrmmTileN = 128;
inline constexpr int kTrmmTileK = 16;

// Threads per side of the tile, and hence the per-thread tile. A 128-wide side
// gets 16 lanes of 8; anything narrower keeps the 4-wide band the vectorized
// fragment load is built on and drops lanes instead.
inline constexpr int trmm_lanes(int tile) { return tile >= 64 ? 16 : 8; }

template <typename T, int TileM>
class TrmmTriangularTilesKernel;

template <typename T, int TileM>
Event launch_trmm_triangular_tiles(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                   const MatrixView<T, MatrixFormat::Dense>& B,
                                   const MatrixView<T, MatrixFormat::Dense>& C,
                                   T alpha,
                                   Uplo uplo,
                                   Transpose transA,
                                   Diag diag) {
    BATCHLAS_KERNEL_TRACE_SCOPE("trmm_cuda_custom.triangular_tiles");

    constexpr int TileN = kTrmmTileN;
    constexpr int TileK = kTrmmTileK;
    constexpr int LocalRows = trmm_lanes(TileM);
    constexpr int LocalCols = trmm_lanes(TileN);
    constexpr int ThreadRows = TileM / LocalRows;
    constexpr int ThreadCols = TileN / LocalCols;
    constexpr int Threads = LocalRows * LocalCols;
    constexpr int SA = TileM;                    // aligned strides
    constexpr int SB = TileN;
    constexpr int PacksA = TileM / 4;
    constexpr int PacksB = TileN / 4;
    constexpr int BandsR = ThreadRows / 4;
    constexpr int BandsC = ThreadCols / 4;
    constexpr int BandSpanR = TileM / BandsR;
    constexpr int BandSpanC = TileN / BandsC;
    constexpr int PacketsA = TileK * PacksA;
    constexpr int PacketsB = TileK * PacksB;

    static_assert(ThreadRows % 4 == 0 && ThreadCols % 4 == 0, "fragments are 4-wide bands");
    static_assert(TileM % TileK == 0, "the diagonal tile has to fall on a k boundary");
    static_assert((PacksA & (PacksA - 1)) == 0 && (PacksB & (PacksB - 1)) == 0,
                  "the swizzle needs a power-of-two packet count");

    const int m = static_cast<int>(C.rows());
    const int n = static_cast<int>(C.cols());
    const int batch = A.batch_size();
    const int tiles_m = (m + TileM - 1) / TileM;
    const int tiles_n = (n + TileN - 1) / TileN;

    const bool transposed = transA != Transpose::NoTrans;
    // Transposing an upper triangle gives a lower one, so this single flag is
    // everything the kernel needs from uplo and trans.
    const bool lower_eff = (uplo == Uplo::Lower) != transposed;
    const bool unit = diag == Diag::Unit;

    const sycl::range<3> local(1, 1, static_cast<size_t>(Threads));
    const sycl::range<3> global(static_cast<size_t>(batch),
                                static_cast<size_t>(tiles_m * tiles_n),
                                static_cast<size_t>(Threads));

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> tile_a(sycl::range<1>(TileK * SA), h);
        sycl::local_accessor<T, 1> tile_b(sycl::range<1>(TileK * SB), h);

        const T* a_ptr = A.data_ptr();
        const T* b_ptr = B.data_ptr();
        T* c_ptr = C.data_ptr();
        const int lda = A.ld();
        const int ldb = B.ld();
        const int ldc = C.ld();
        const int stride_a = A.stride();
        const int stride_b = B.stride();
        const int stride_c = C.stride();

        h.parallel_for<TrmmTriangularTilesKernel<T, TileM>>(
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));
                if (bid >= batch) {
                    return;
                }
                const int gid = static_cast<int>(item.get_group(1));
                const int tile_row = gid / tiles_n;
                const int tile_col = gid % tiles_n;
                const int m0 = tile_row * TileM;
                const int n0 = tile_col * TileN;

                const int tid = static_cast<int>(item.get_local_id(2));
                const int tr = tid % LocalRows;
                const int tc = tid / LocalRows;

                const T* Ab = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
                const T* Bb = b_ptr + static_cast<std::ptrdiff_t>(bid) * stride_b;
                T* Cb = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;
                T* sa = tile_a.template get_multi_ptr<sycl::access::decorated::no>().get();
                T* sb = tile_b.template get_multi_ptr<sycl::access::decorated::no>().get();

                auto swizzle_a = [](int packet, int pp) {
                    return packet ^ ((pp >> 2) & (PacksA - 1));
                };
                auto swizzle_b = [](int packet, int pp) {
                    return packet ^ ((pp >> 2) & (PacksB - 1));
                };

                T accum[ThreadRows][ThreadCols];
#pragma unroll
                for (int i = 0; i < ThreadRows; ++i) {
#pragma unroll
                    for (int j = 0; j < ThreadCols; ++j) {
                        accum[i][j] = T(0);
                    }
                }

                // The whole point: an output tile rooted at row m0 reads only
                // the part of the reduction its triangle can reach.
                const int p_begin = lower_eff ? 0 : m0;
                const int p_end = lower_eff ? sycl::min(m, m0 + TileM) : m;

                for (int p0 = p_begin; p0 < p_end; p0 += TileK) {
                    // op(A): rows [m0, m0+TileM) against reduction [p0, p0+TileK),
                    // with the triangle, the unit diagonal and the edges of the
                    // matrix all resolved here, so the inner loop sees a dense
                    // tile and the zero half is never multiplied.
                    // op(A) is read down i when it is not transposed and down p
                    // when it is, and the staging assignment has to follow -- a
                    // warp walking i over a transposed A strides by lda and each
                    // lane pulls its own 32-byte sector, which measured as the
                    // dominant cost before the split.
                    if (transposed) {
                        for (int flat = tid; flat < PacketsA; flat += Threads) {
                            const int i = m0 + flat / (TileK / 4);
                            const int pp0 = (flat % (TileK / 4)) * 4;
                            const int ii = i - m0;
                            const int phys = (swizzle_a(ii >> 2, pp0) << 2) | (ii & 3);
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                const int p = p0 + pp0 + e;
                                T value = T(0);
                                if (i < m && p < m) {
                                    if (i == p) {
                                        value = unit
                                            ? T(1)
                                            : Ab[i + static_cast<std::ptrdiff_t>(i) * lda];
                                    } else if (lower_eff ? (p < i) : (p > i)) {
                                        value = Ab[p + static_cast<std::ptrdiff_t>(i) * lda];
                                    }
                                }
                                sa[(pp0 + e) * SA + phys] = value;
                            }
                        }
                    } else {
                    for (int flat = tid; flat < PacketsA; flat += Threads) {
                        const int pp = flat / PacksA;
                        const int ii0 = (flat % PacksA) * 4;
                        const int p = p0 + pp;
                        const int phys = swizzle_a(ii0 >> 2, pp) << 2;
                        // The four rows a thread stages are adjacent in shared,
                        // so they go back as one 128-bit store. Writing them
                        // singly would put every lane of the warp on one of only
                        // eight banks -- the row stride is a multiple of 32, so
                        // it drops out, and a single-element store cannot spread
                        // wider than the packet index does.
                        TileVec4<T> packet;
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const int i = m0 + ii0 + e;
                            T value = T(0);
                            if (i < m && p < m) {
                                if (i == p) {
                                    value = unit
                                        ? T(1)
                                        : Ab[i + static_cast<std::ptrdiff_t>(i) * lda];
                                } else if (lower_eff ? (p < i) : (p > i)) {
                                    value = Ab[i + static_cast<std::ptrdiff_t>(p) * lda];
                                }
                            }
                            packet.v[e] = value;
                        }
                        tile_vec4(&sa[pp * SA + phys]) = packet;
                    }
                    }

                    // B: reduction rows [p0, p0+TileK) against columns
                    // [n0, n0+TileN). B is contiguous down its row index, so a
                    // thread takes four adjacent reduction rows of one column.
                    for (int flat = tid; flat < PacketsB; flat += Threads) {
                        const int col = flat / (TileK / 4);
                        const int pp0 = (flat % (TileK / 4)) * 4;
                        const int j = n0 + col;
                        const int phys = (swizzle_b(col >> 2, pp0) << 2) | (col & 3);
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const int p = p0 + pp0 + e;
                            sb[(pp0 + e) * SB + phys] =
                                (j < n && p < m)
                                ? Bb[p + static_cast<std::ptrdiff_t>(j) * ldb]
                                : T(0);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

#pragma unroll 4
                    for (int p = 0; p < TileK; ++p) {
                        const T* rowa = &sa[p * SA];
                        const T* rowb = &sb[p * SB];
                        T af[ThreadRows];
                        T bf[ThreadCols];
#pragma unroll
                        for (int b = 0; b < BandsR; ++b) {
                            const TileVec4<T> v =
                                tile_vec4(rowa + (swizzle_a(b * (BandSpanR / 4) + tr, p) << 2));
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                af[b * 4 + e] = v.v[e];
                            }
                        }
#pragma unroll
                        for (int b = 0; b < BandsC; ++b) {
                            const TileVec4<T> v =
                                tile_vec4(rowb + (swizzle_b(b * (BandSpanC / 4) + tc, p) << 2));
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                bf[b * 4 + e] = v.v[e];
                            }
                        }
#pragma unroll
                        for (int i = 0; i < ThreadRows; ++i) {
#pragma unroll
                            for (int j = 0; j < ThreadCols; ++j) {
                                accum[i][j] += af[i] * bf[j];
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // TRMM overwrites C; there is no beta, so nothing here reads it.
#pragma unroll
                for (int j = 0; j < ThreadCols; ++j) {
                    const int col = n0 + (j / 4) * BandSpanC + tc * 4 + (j % 4);
                    if (col >= n) {
                        continue;
                    }
#pragma unroll
                    for (int i = 0; i < ThreadRows; ++i) {
                        const int row = m0 + (i / 4) * BandSpanR + tr * 4 + (i % 4);
                        if (row >= m) {
                            continue;
                        }
                        Cb[row + static_cast<std::ptrdiff_t>(col) * ldc] = alpha * accum[i][j];
                    }
                }
            });
    });

    return ctx.get_event();
}

// The row tile is sized to m when m is small -- ormqr hands this ib in the tens
// and a 128-row tile would quadruple its arithmetic -- and capped at 128, above
// which the grid tiles m and the k-range restriction does the saving.
template <typename T>
Event trmm_triangular_tiles(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& A,
                            const MatrixView<T, MatrixFormat::Dense>& B,
                            const MatrixView<T, MatrixFormat::Dense>& C,
                            T alpha,
                            Uplo uplo,
                            Transpose transA,
                            Diag diag) {
    const int m = static_cast<int>(C.rows());
    if (m <= 32) {
        return launch_trmm_triangular_tiles<T, 32>(ctx, A, B, C, alpha, uplo, transA, diag);
    }
    if (m <= 64) {
        return launch_trmm_triangular_tiles<T, 64>(ctx, A, B, C, alpha, uplo, transA, diag);
    }
    return launch_trmm_triangular_tiles<T, 128>(ctx, A, B, C, alpha, uplo, transA, diag);
}

} // namespace batchlas::backend::detail
