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
// R = 2, and only 1.78x by R = 8. Getting R above 1 is the single thing that
// decides whether this beats a GEMM -- see `trmm_row_tile`, which is where the
// scalar type enters.
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

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

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
// fragment load is built on and drops lanes instead. That is what fixes the
// 16-row tile at 4 lanes: ThreadRows must stay a multiple of 4, and 16/8 = 2 is
// not one, so the lanes go rather than the band.
inline constexpr int trmm_lanes(int tile) {
    if (tile >= 64) return 16;
    return tile >= 32 ? 8 : 4;
}

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
    // A complex scalar halves how many elements the shared path delivers per
    // clock while leaving the FMA rate per element alone, so the fragment-to-
    // FFMA ratio that is comfortable in float becomes the binding constraint:
    // a 4x8 thread tile reads 12 complex and issues 32 complex MACs, which
    // needs 2.67 MAC per load against a capability of 2. Widening the thread
    // tile to 4x16 takes it to 3.2 and puts the kernel back on the FMA pipe.
    // Halving the lane count is what pays for it, and the row tiling -- where
    // the triangle saving lives -- is left untouched.
    //
    // Only at the 64-row tile, though. At the 32-row tile the block is already
    // down to 64 threads and halving the lanes again costs more in outstanding
    // loads than the ratio buys -- that end is bandwidth bound, not FMA bound,
    // and measured 0.374 -> 0.424 ms at m = 32.
    constexpr int LocalCols = (sycl::detail::is_complex<T>::value && TileM >= 64)
                                  ? trmm_lanes(TileN) / 2
                                  : trmm_lanes(TileN);
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
    const bool conjugated = transA == Transpose::ConjTrans;

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
                                    // ConjTrans means op(A) = A^H, so the
                                    // element is conjugated on the way in. The
                                    // unit diagonal is a literal 1 and has
                                    // nothing to conjugate.
                                    if (i == p) {
                                        value = unit
                                            ? T(1)
                                            : conjugated
                                                ? conj_if(Ab[i + static_cast<std::ptrdiff_t>(i) * lda])
                                                : Ab[i + static_cast<std::ptrdiff_t>(i) * lda];
                                    } else if (lower_eff ? (p < i) : (p > i)) {
                                        const T raw = Ab[p + static_cast<std::ptrdiff_t>(i) * lda];
                                        value = conjugated ? conj_if(raw) : raw;
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
                        tile_store4(&sa[pp * SA + phys], packet);
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
                                tile_load4(rowa + (swizzle_a(b * (BandSpanR / 4) + tr, p) << 2));
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                af[b * 4 + e] = v.v[e];
                            }
                        }
#pragma unroll
                        for (int b = 0; b < BandsC; ++b) {
                            const TileVec4<T> v =
                                tile_load4(rowb + (swizzle_b(b * (BandSpanC / 4) + tc, p) << 2));
#pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                bf[b * 4 + e] = v.v[e];
                            }
                        }
#pragma unroll
                        for (int i = 0; i < ThreadRows; ++i) {
#pragma unroll
                            for (int j = 0; j < ThreadCols; ++j) {
                                accum[i][j] = accumulate(accum[i][j], af[i], bf[j]);
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

// Everything the kernel needs from the problem, independent of scalar type and
// of the queue. Left side only: the right-side product C = alpha * B * op(A)
// puts the triangle on the column index, which is a different k-loop bound and
// a different staging assignment, not a transpose of this one.
template <typename T>
bool trmm_tiles_supported(const MatrixView<T, MatrixFormat::Dense>& A,
                          const MatrixView<T, MatrixFormat::Dense>& B,
                          const MatrixView<T, MatrixFormat::Dense>& C,
                          Side side) {
    if (side != Side::Left) {
        return false;
    }
    if (A.rows() != A.cols() || A.rows() != C.rows()) {
        return false;
    }
    if (A.batch_size() != B.batch_size() || B.batch_size() != C.batch_size()) {
        return false;
    }
    if (B.rows() != C.rows() || B.cols() != C.cols()) {
        return false;
    }
    if (A.is_heterogeneous() || B.is_heterogeneous() || C.is_heterogeneous()) {
        return false;
    }
    return C.rows() > 0 && C.cols() > 0;
}

// How wide a row tile to cut m into, which is the kernel's one real trade-off.
//
// R = m/TileM row tiles shrink the reduction to (R+1)/2R of the square, so a
// smaller tile does strictly less arithmetic: 1.0x at R = 1, 0.75x at R = 2,
// 0.5625x at R = 8. It also re-reads B, because each row tile stages its own
// copy of the reduction range it needs -- (R+1)/2 times over in the worst case,
// though in practice much of that is L2 hits.
//
// Which of the two dominates is decided by the scalar type, not by the shape:
//
//   float   is bandwidth bound at these sizes -- intensity is m/4 flop per byte
//           against a ridge near 40 -- so paying B twice to save a quarter of
//           the arithmetic is a bad trade, and the widest tile that fits wins.
//   double  runs at 1/64 rate on this part, which puts the ridge near 1.4 flop
//           per byte and the problem far on the compute side of it. There the
//           arithmetic is the whole cost and B's re-read is close to free, so
//           the narrowest tile wins. Complex is the same argument: a complex
//           multiply is four real ones.
//
// This is why one threshold cannot serve both, and why the first version of
// this kernel -- tuned on float alone, one tile for m <= 128 -- could not beat
// a GEMM at m = 128 in any type: R = 1 saves nothing at all.
//
// The 32-row tile used to be the floor, which left ormqr's WY update -- m = ib,
// in the tens -- with R = 1 and no saving at all, and at m = 16 with half the
// tile masked off at the epilogue after the arithmetic had already been issued.
// A 16-row tile fixes both, and measured through ormqr_blocked_benchmark
// (n 256/512, batch 128-256, ib 16/32/64) it goes exactly where the argument
// above says it should -- tile16 against tile32:
//
//   double           1.007x - 1.022x   wins at every ib, including 64
//   complex<double>  0.995x - 1.040x   wins to ib 32, a wash at 64
//   complex<float>   0.993x - 1.026x   wins to ib 32, a wash at 64
//   float            0.966x - 0.997x   loses everywhere, as predicted
//
// float losing is the B re-read, which is the same reason 32 loses to 64 for it
// further up. The thresholds below are that table.
template <typename T>
inline int trmm_row_tile(int m) {
    constexpr bool complex_t = sycl::detail::is_complex<T>::value;
    constexpr bool wide = sizeof(typename base_type<T>::type) > 4 || complex_t;
    if constexpr (wide) {
        // Complex stops at 32 because its 64-row cell measured a wash either
        // way (0.993x in complex<float>), and the wider tile keeps more of the
        // block: at TileM 16 the launch is down to 64 threads.
        if (m <= (complex_t ? 32 : 64)) {
            return 16;
        }
        // Never 128. Beyond the argument above, a 128x128 tile is an 8x8 thread
        // tile, which in complex<double> is 256 registers of accumulator alone
        // -- 256 threads x 256 is the entire 65536 a work-group gets, and the
        // runtime rejects the launch rather than spilling.
        return m <= 64 ? 32 : 64;
    } else {
        if (m <= 32) {
            return 32;
        }
        // Measured, float, against the GEMM (ms): at m = 128 nC = 512 batch
        // 1024, TileM 32/64/128 gives 0.663 / 0.658 / 0.699 against a GEMM's
        // 0.674 -- so 128 loses and 64 wins. 32 is worse than 64 everywhere,
        // which is the B re-read showing up. Only at m = 1024 does 128 come
        // back ahead (1.146 vs 1.180).
        return m <= 512 ? 64 : 128;
    }
}
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
    // BATCHLAS_TRMM_TILE_M pins the row tile so the trade-off below can be
    // swept from one binary.
    const int forced = [] {
        const char* raw = std::getenv("BATCHLAS_TRMM_TILE_M");
        return raw ? std::atoi(raw) : 0;
    }();

    const int tile_m = forced ? forced : trmm_row_tile<T>(m);
    if (tile_m <= 16) {
        return launch_trmm_triangular_tiles<T, 16>(ctx, A, B, C, alpha, uplo, transA, diag);
    }
    if (tile_m <= 32) {
        return launch_trmm_triangular_tiles<T, 32>(ctx, A, B, C, alpha, uplo, transA, diag);
    }
    if (tile_m <= 64) {
        return launch_trmm_triangular_tiles<T, 64>(ctx, A, B, C, alpha, uplo, transA, diag);
    }
    return launch_trmm_triangular_tiles<T, 128>(ctx, A, B, C, alpha, uplo, transA, diag);
}

} // namespace batchlas::backend::detail
