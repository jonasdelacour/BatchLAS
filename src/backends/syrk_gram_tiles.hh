#pragma once

// A single-tile batched SYRK for the shape ortho actually hands it: a tall,
// skinny A and a small square C.
//
// The 128x128 triangular-tile kernel next door is built for the other end of
// the range -- n in the hundreds, where the tile grid is many tiles wide and
// skipping the ones outside the triangle is the whole point. It cannot serve
// n <= 128 at all: one tile is the entire matrix, so the grid has nothing to
// skip, and a 128-wide tile spends 128*128*k arithmetic on an n*n/2 answer. At
// n = 32 that is a 32x overcharge, and the router (correctly) refused it, which
// is why every Gram matrix in ortho fell through to a host loop over
// cublasSsyrk and lost by up to two orders of magnitude at large batch.
//
// This kernel sizes the tile to n instead, and takes the triangle one level
// further down -- at thread-tile granularity rather than block granularity:
//
//   * One tile covers the whole of C, so the two operands of A^T A are the
//     same columns of A. There is one shared tile, not two, and A crosses the
//     bus exactly once. That is the thing worth optimising at the skinny end,
//     where arithmetic intensity is n/4 flop per byte -- 8 at n = 32 against
//     the 4090's ridge of ~40, so the kernel is bandwidth bound by 5x and the
//     measured 933 GB/s is the whole story.
//   * Only the Lanes*(Lanes+1)/2 thread tiles that meet the requested triangle
//     are carried -- 136 of 256 at n = 128, so 160 threads instead of 256.
//     Merely masking the epilogue would leave the block doing exactly a GEMM's
//     arithmetic, which is why the first cut of this kernel could match a GEMM
//     at n = 128 and never beat one. The tail of the last warp stages and
//     reaches every barrier but computes nothing.
//   * The shared tile is stored [k][n] with the stride exactly n, so the
//     fragment loads keep 16-byte alignment and issue as LDS.128. That matters
//     more here than anywhere: an 8x8 thread tile reads 16 floats per k-step
//     and issues 64 FFMAs, and the SM's 32-floats-per-clock shared path against
//     its 128-FFMA-per-clock math path puts that ratio exactly on the balance
//     point. There is no headroom to give away to a padded stride.
//
// Staging then has to reach that aligned layout without either giving up
// coalescing on the global side or colliding on the shared side, and the two
// transpose modes need opposite assignments to do it:
//
//   Trans    A is k x n, contiguous down k. One thread takes four adjacent
//            reduction rows of one column, so eight lanes span a 128-byte
//            run of it.
//   NoTrans  A is n x k, contiguous across n. One thread takes four adjacent
//            columns of one reduction row: a float4 either side.

#include "triangular_tiles.hh"

#include "../queue.hh"
#include "../util/kernel-trace.hh"

#include <blas/enums.hh>
#include <blas/matrix.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::backend::detail {

// The output tile is the whole matrix, so this is also the largest n the
// kernel can serve.
inline constexpr int kGramMaxTile = 128;

// Conjugation is a no-op on a real scalar, which is what lets one kernel serve
// both syrk and herk rather than two near-copies.
template <typename T>
inline T conj_if(const T& value) {
    if constexpr (sycl::detail::is_complex<T>::value) {
        return std::conj(value);
    } else {
        return value;
    }
}

// accum += a * b, written out rather than delegated to std::complex.
//
// `std::complex<float>::operator*` lowers to the __mulsc3 libcall, which
// implements C99 Annex G -- a branch on Inf and NaN around every single
// multiply. In the innermost loop of a GEMM that is ruinous and it is invisible
// in the source: the first complex build of this kernel ran at 1.2 TFLOP/s
// against float's 13.8, and at n = 128 took 38 ms where a cuBLAS GEMM took 1.5.
// Four real multiplies and two adds is the whole operation; there is no
// exceptional case here worth a branch, because a NaN in the input is already
// a NaN in the answer.
template <typename T>
inline void accumulate(T& accum, const T& a, const T& b) {
    if constexpr (sycl::detail::is_complex<T>::value) {
        using Real = typename T::value_type;
        const Real ar = a.real();
        const Real ai = a.imag();
        const Real br = b.real();
        const Real bi = b.imag();
        accum = T(accum.real() + ar * br - ai * bi,
                  accum.imag() + ar * bi + ai * br);
    } else {
        accum += a * b;
    }
}

// Drops the imaginary part BLAS guarantees is zero on a HERK diagonal. It is
// zero only up to rounding, and leaving the residue there would make a matrix
// that is not quite Hermitian, which the eigensolvers downstream do notice.
template <typename T>
inline T real_part_of(const T& value) {
    if constexpr (sycl::detail::is_complex<T>::value) {
        return T(value.real(), typename T::value_type(0));
    } else {
        return value;
    }
}

template <typename T, int NTile, int ThreadTile, int KC, bool TransOperand, bool Conjugate>
class SyrkGramTilesKernel;

template <typename T, int NTile, int ThreadTile, int KC, bool TransOperand, bool Conjugate>
Event launch_syrk_gram_tiles(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             T alpha,
                             T beta,
                             Uplo uplo) {
    BATCHLAS_KERNEL_TRACE_SCOPE("syrk_cuda_custom.gram_tiles");

    static_assert(NTile % ThreadTile == 0, "the thread grid has to tile the output exactly");
    static_assert(ThreadTile % 4 == 0, "fragments are loaded as 4-wide bands");
    static_assert(KC % 4 == 0, "staging moves whole packets");

    constexpr int Lanes = NTile / ThreadTile;          // thread tiles per side
    constexpr int TriTiles = Lanes * (Lanes + 1) / 2;  // ... that meet the triangle
    constexpr int Threads = ((TriTiles + 31) / 32) * 32;
    constexpr int SPad = NTile;                        // aligned: LDS.128 fragments
    constexpr int Bands = ThreadTile / 4;              // vectorized bands per thread
    constexpr int Packs = NTile / 4;                   // 4-wide packets per row
    constexpr int SwizzleMask = Packs - 1;
    constexpr int Packets = KC * Packs;
    static_assert((Packs & SwizzleMask) == 0, "the swizzle needs a power-of-two packet count");

    const int n = static_cast<int>(C.rows());
    const int k = TransOperand ? static_cast<int>(A.rows()) : static_cast<int>(A.cols());
    const int batch = A.batch_size();

    // Taking a staging thread's four elements as one 128-bit load was tried and
    // measured slower everywhere (n = 64 batch 1024: 0.333 -> 0.384 ms). The
    // four predicated scalar loads land in one cache line, so the merge saves
    // no traffic, and the branch it needs costs more than the instructions it
    // removes. Left as four loads deliberately.

    const sycl::range<2> local(1, static_cast<size_t>(Threads));
    const sycl::range<2> global(static_cast<size_t>(batch), static_cast<size_t>(Threads));

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> tile(sycl::range<1>(KC * SPad), h);

        const T* a_ptr = A.data_ptr();
        T* c_ptr = C.data_ptr();
        const int lda = A.ld();
        const int ldc = C.ld();
        const int stride_a = A.stride();
        const int stride_c = C.stride();
        const bool lower = uplo == Uplo::Lower;

        h.parallel_for<SyrkGramTilesKernel<T, NTile, ThreadTile, KC, TransOperand, Conjugate>>(
            sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                const int bid = static_cast<int>(item.get_group(0));
                if (bid >= batch) {
                    return;
                }
                const int tid = static_cast<int>(item.get_local_id(1));
                const bool active = tid < TriTiles;

                int bi = 0;
                int bj = 0;
                triangular_tile_decode(active ? tid : 0, bi, bj);
                const int tr = lower ? bi : bj;   // row tile
                const int tc = lower ? bj : bi;   // column tile

                const T* Ab = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
                T* Cb = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;
                T* sh = tile.template get_multi_ptr<sycl::access::decorated::no>().get();

                // Packet q of reduction row kk lives at q ^ (kk / 4) instead of
                // q. An aligned stride alone would put every lane of a staging
                // write in the same bank -- the rows a lane writes are 4 apart
                // and the stride is a multiple of 32, so the row index drops out
                // and only the column is left to separate them. Rotating the
                // packet by the row group puts it back, and because the rotation
                // is by whole 4-wide packets the fragment loads stay 16-byte
                // aligned and stay LDS.128. Both sides come out conflict-free.
                auto swizzle = [](int packet, int kk) {
                    return packet ^ ((kk >> 2) & SwizzleMask);
                };

                T accum[ThreadTile][ThreadTile];
#pragma unroll
                for (int i = 0; i < ThreadTile; ++i) {
#pragma unroll
                    for (int j = 0; j < ThreadTile; ++j) {
                        accum[i][j] = T(0);
                    }
                }

                for (int k0 = 0; k0 < k; k0 += KC) {
                    // The tile is always filled to its full KC x NTile, with
                    // zeros for anything off the end of A, so the inner loop
                    // below carries no bounds checks and a short n or k simply
                    // contributes nothing.
                    if constexpr (TransOperand) {
                        // Four adjacent reduction rows of one column. Eight
                        // lanes cover a column's whole 128-byte run, so the
                        // global side stays coalesced; the shared side pays a
                        // bank conflict for it, which is the cheaper half --
                        // the tile is written once and read KC times over.
                        for (int flat = tid; flat < Packets; flat += Threads) {
                            const int col = flat / (KC / 4);
                            const int kk0 = (flat % (KC / 4)) * 4;
                            const bool col_ok = col < n;
                            const int phys = (swizzle(col >> 2, kk0) << 2) | (col & 3);
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int gk = k0 + kk0 + i;
                                sh[(kk0 + i) * SPad + phys] =
                                    (col_ok && gk < k)
                                    ? Ab[gk + static_cast<std::ptrdiff_t>(col) * lda]
                                    : T(0);
                            }
                        }
                    } else {
                        for (int flat = tid; flat < Packets; flat += Threads) {
                            const int kk = flat / Packs;
                            const int col0 = (flat % Packs) * 4;
                            const int gk = k0 + kk;
                            const bool k_ok = gk < k;
                            const int phys = swizzle(col0 >> 2, kk) << 2;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int col = col0 + i;
                                sh[kk * SPad + phys + i] =
                                    (k_ok && col < n)
                                    ? Ab[col + static_cast<std::ptrdiff_t>(gk) * lda]
                                    : T(0);
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (active) {
#pragma unroll 4
                        for (int p = 0; p < KC; ++p) {
                            const T* row = &sh[p * SPad];
                            T af[ThreadTile];
                            T bf[ThreadTile];
#pragma unroll
                            for (int b = 0; b < Bands; ++b) {
                                // A thread's rows and columns are contiguous,
                                // and have to be: the triangular decode below
                                // decides a whole thread tile is inside the
                                // requested half from its tile indices alone.
                                // Splitting the 8 into two bands 64 apart --
                                // which is what the square 128x128 kernels do
                                // to spread shared-memory banks -- makes that
                                // false, and it fails quietly: thread (0,1)
                                // then owns element (64,4), which is in the
                                // lower triangle while its tile is not, so
                                // nothing writes it.
                                const int qa = swizzle(tr * Bands + b, p);
                                const int qb = swizzle(tc * Bands + b, p);
                                T va[4];
                                T vb[4];
                                tile_load4(row + qa * 4, va);
                                tile_load4(row + qb * 4, vb);
#pragma unroll
                                for (int e = 0; e < 4; ++e) {
                                    // HERK conjugates whichever operand carries
                                    // the ^H. With transA = ConjTrans that is
                                    // the row index, with NoTrans the column;
                                    // conjugating the wrong one still yields a
                                    // Hermitian matrix, so this cannot be
                                    // caught by inspecting the result's shape.
                                    if constexpr (Conjugate) {
                                        af[b * 4 + e] = TransOperand ? conj_if(va[e]) : va[e];
                                        bf[b * 4 + e] = TransOperand ? vb[e] : conj_if(vb[e]);
                                    } else {
                                        af[b * 4 + e] = va[e];
                                        bf[b * 4 + e] = vb[e];
                                    }
                                }
                            }
#pragma unroll
                            for (int i = 0; i < ThreadTile; ++i) {
#pragma unroll
                                for (int j = 0; j < ThreadTile; ++j) {
                                    accumulate(accum[i][j], af[i], bf[j]);
                                }
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (!active) {
                    return;
                }

                // Only the requested triangle is written. BLAS forbids touching
                // the other half, and with beta != 0 it forbids reading it too.
                // Off-diagonal thread tiles lie wholly inside it; the ones on
                // the diagonal are the only ones the element mask trims.
                const bool on_diagonal = bi == bj;
#pragma unroll
                for (int j = 0; j < ThreadTile; ++j) {
                    const int col = tc * ThreadTile + j;
                    if (col >= n) {
                        continue;
                    }
#pragma unroll
                    for (int i = 0; i < ThreadTile; ++i) {
                        const int row = tr * ThreadTile + i;
                        if (row >= n) {
                            continue;
                        }
                        if (on_diagonal && (lower ? row < col : row > col)) {
                            continue;
                        }
                        T* p = &Cb[row + static_cast<std::ptrdiff_t>(col) * ldc];
                        T value = beta == T(0) ? alpha * accum[i][j]
                                               : alpha * accum[i][j] + beta * *p;
                        // HERK's diagonal is real by construction and BLAS says
                        // so; rounding leaves a residue that would make C only
                        // almost Hermitian.
                        if constexpr (Conjugate) {
                            if (row == col) {
                                value = real_part_of(value);
                            }
                        }
                        *p = value;
                    }
                }
            });
    });

    return ctx.get_event();
}

// Everything the kernel needs from the problem, independent of scalar type and
// of the queue. Only the tile-to-n range is served: above kGramMaxTile there is
// no single tile to put C in, and for anything but float that is the end of the
// road, since syrk_triangular_tiles' staging and fragment loads are written
// around a 128-bit packet that only float has.
template <typename T>
bool syrk_gram_supported(const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& C,
                         Transpose transA,
                         bool conjugated) {
    if (C.rows() != C.cols() || C.rows() <= 0 || C.rows() > kGramMaxTile) {
        return false;
    }
    if (A.batch_size() != C.batch_size()) {
        return false;
    }
    if (A.is_heterogeneous() || C.is_heterogeneous()) {
        return false;
    }
    // SYRK spells A*A^T and must not be handed a ConjTrans; HERK spells A*A^H
    // and must not be handed a plain Trans, which would be complex-symmetric.
    if (conjugated ? (transA != Transpose::NoTrans && transA != Transpose::ConjTrans)
                   : (transA == Transpose::ConjTrans)) {
        return false;
    }
    const int n = C.rows();
    const int k = transA == Transpose::NoTrans ? A.cols() : A.rows();
    const int expected_n = transA == Transpose::NoTrans ? A.rows() : A.cols();
    return expected_n == n && k > 0;
}

// The tile has to cover n, and wants to be no wider than that: every column of
// slack is arithmetic thrown away. ThreadTile then follows from keeping the
// thread tile at the 4-wide band the vectorized fragment loads are built on.
template <typename T, bool Conjugate = false>
Event syrk_gram_tiles(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      T alpha,
                      T beta,
                      Uplo uplo,
                      Transpose transA) {
    const int n = static_cast<int>(C.rows());
    const bool trans = transA != Transpose::NoTrans;

    auto dispatch = [&](auto tile_tag) {
        constexpr int NTile = decltype(tile_tag)::value;
        // A 4-wide thread tile puts 528 tiles -- 544 threads -- on the 128-wide
        // case, which is what float wants and measured fastest there. A complex
        // scalar cannot afford the block: two components per accumulator took it
        // to 205 registers per work-item, and 544 x 205 is past the 65536 a
        // work-group gets, which the runtime rejects outright rather than
        // spilling. Doubling the thread tile quarters the thread count to 160
        // and the same accumulators fit.
        constexpr int ThreadTile =
            (NTile == 128 && sycl::detail::is_complex<T>::value) ? 8 : 4;
        constexpr int KC = 32;
        return trans
            ? launch_syrk_gram_tiles<T, NTile, ThreadTile, KC, true, Conjugate>(
                  ctx, A, C, alpha, beta, uplo)
            : launch_syrk_gram_tiles<T, NTile, ThreadTile, KC, false, Conjugate>(
                  ctx, A, C, alpha, beta, uplo);
    };

    if (n <= 32) {
        return dispatch(std::integral_constant<int, 32>{});
    }
    if (n <= 64) {
        return dispatch(std::integral_constant<int, 64>{});
    }
    return dispatch(std::integral_constant<int, 128>{});
}

} // namespace batchlas::backend::detail
