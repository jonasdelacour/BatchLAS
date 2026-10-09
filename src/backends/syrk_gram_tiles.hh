#pragma once

// A single-tile batched SYRK/HERK for n <= kGramMaxTile (ortho's tall, skinny
// A). The tile is sized to n, one shared tile serves both operands, and only the
// thread tiles meeting the requested triangle are carried. The shared stride is
// exactly n (unpadded) so fragments stay LDS.128; the swizzle below keeps that
// conflict-free. Staging assignment depends on the transpose mode.
// evidence: docs/perf/level3.md#syrk-gram-tiles-kernel-design

#include "triangular_tiles.hh"

#include "../queue.hh"
#include "../util/kernel-trace.hh"
#include "../sycl/kernel_attrs.hh"

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::backend::detail {

// The output tile is the whole matrix, so this is also the largest n the
// kernel can serve.
inline constexpr int kGramMaxTile = 128;

// Drops the imaginary part BLAS guarantees is zero on a HERK diagonal. It is
// zero only up to rounding, and leaving the residue there would make a matrix
// that is not quite Hermitian, which the eigensolvers downstream do notice.
template <typename T>
inline T real_part_of(const T& value) {
    if constexpr (is_std_complex_v<T>) {
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

    // Four scalar staging loads, not one 128-bit load, deliberately (measured slower).
    // evidence: docs/perf/level3.md#syrk-gram-tiles-kernel-design

    const sycl::range<2> local(1, static_cast<size_t>(Threads));
    const sycl::range<2> global(static_cast<size_t>(batch), static_cast<size_t>(Threads));

    impl::check_group_count(*ctx, sycl::nd_range<2>(global, local));
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

                // Packet q of row kk lives at q ^ (kk / 4): the aligned stride
                // alone puts every staging lane in one bank. Whole-packet
                // rotation keeps fragment loads 16-byte aligned (LDS.128).
                auto swizzle = [](int packet, int kk) {
                    return packet ^ ((kk >> 2) & SwizzleMask);
                };

                T accum[ThreadTile][ThreadTile];
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < ThreadTile; ++i) {
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < ThreadTile; ++j) {
                        accum[i][j] = T(0);
                    }
                }

                for (int k0 = 0; k0 < k; k0 += KC) {
                    // Zero-filled to the full KC x NTile, so the inner loop has no bounds checks.
                    if constexpr (TransOperand) {
                        // Coalesced global side; the shared side pays a conflict (written once).
                        for (int flat = tid; flat < Packets; flat += Threads) {
                            const int col = flat / (KC / 4);
                            const int kk0 = (flat % (KC / 4)) * 4;
                            const bool col_ok = col < n;
                            const int phys = (swizzle(col >> 2, kk0) << 2) | (col & 3);
BATCHLAS_UNROLL_FULL
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
BATCHLAS_UNROLL_FULL
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
BATCHLAS_UNROLL_FULL
                            for (int b = 0; b < Bands; ++b) {
                                // Invariant: a thread's rows/columns are
                                // contiguous. Two bands 64 apart (as in the
                                // 128x128 kernels) silently leave elements unwritten.
                                // evidence: docs/perf/level3.md#the-band-split-syrk-bug
                                const int qa = swizzle(tr * Bands + b, p);
                                const int qb = swizzle(tc * Bands + b, p);
                                const TileVec4<T> va = tile_load4(row + qa * 4);
                                const TileVec4<T> vb = tile_load4(row + qb * 4);
BATCHLAS_UNROLL_FULL
                                for (int e = 0; e < 4; ++e) {
                                    // Conjugate the operand carrying the ^H (row
                                    // for ConjTrans, column for NoTrans). The
                                    // wrong one is still Hermitian: test vs GEMM.
                                    if constexpr (Conjugate) {
                                        af[b * 4 + e] = TransOperand ? conj_if(va.v[e]) : va.v[e];
                                        bf[b * 4 + e] = TransOperand ? vb.v[e] : conj_if(vb.v[e]);
                                    } else {
                                        af[b * 4 + e] = va.v[e];
                                        bf[b * 4 + e] = vb.v[e];
                                    }
                                }
                            }
BATCHLAS_UNROLL_FULL
                            for (int i = 0; i < ThreadTile; ++i) {
BATCHLAS_UNROLL_FULL
                                for (int j = 0; j < ThreadTile; ++j) {
                                    accum[i][j] = accumulate(accum[i][j], af[i], bf[j]);
                                }
                            }
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (!active) {
                    return;
                }

                // Only the requested triangle is read or written; only
                // diagonal thread tiles need the element mask.
                const bool on_diagonal = bi == bj;
BATCHLAS_UNROLL_FULL
                for (int j = 0; j < ThreadTile; ++j) {
                    const int col = tc * ThreadTile + j;
                    if (col >= n) {
                        continue;
                    }
BATCHLAS_UNROLL_FULL
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
                        // HERK's diagonal is real; drop the rounding residue.
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

// Problem-side support, independent of scalar type and queue. Only n <=
// kGramMaxTile; above it non-float has no native route (the triangular kernel is float-only).
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

// The tile covers n with as little slack as possible: slack is wasted arithmetic.
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
        // Complex at NTile 128 needs the 8-wide thread tile: the 4-wide one
        // exceeds the work-group register file and the launch is rejected.
        // evidence: docs/perf/level3.md#syrk-gram-tiles-kernel-design
        constexpr int ThreadTile =
            (NTile == 128 && is_std_complex_v<T>) ? 8 : 4;
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
