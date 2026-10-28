// SPDX-License-Identifier: MIT
#pragma once

// A 128x128x8 register-tiled SYR2K over the triangular tile set, fusing both
// rank-k products into the same accumulators:
//     accum += A[bi] * B[bj]^T   and   accum += B[bi] * A[bj]^T
// Products are issued sequentially, not interleaved, to keep one fragment pair
// live (occupancy). Diagonal tiles alias the bj-side tiles to the bi-side ones
// (block-uniform, no divergence) and mask their epilogue.
// evidence: docs/perf/level3.md#syrk-and-syr2k-triangular-tiles-kernel-design

#include <src/kernels/common/triangular_tiles.hh>

#include <src/queue.hh>
#include <src/util/batch_mask.hh>
#include <src/util/kernel-trace.hh>
#include <src/sycl/kernel_attrs.hh>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::backend::detail {

template <typename T, bool TransOperand, bool AlignedFastPath, bool Conjugate>
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

template <typename T, bool TransOperand, bool AlignedFastPath, bool Conjugate>
Event launch_syr2k_triangular_tiles(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    T alpha,
                                    T beta,
                                    Uplo uplo,
                                    T a_scale) {
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

    impl::check_group_count(*ctx, sycl::nd_range<3>(global, local));
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
        // The batch mask (null unmasked); one item per work-group, so its return is uniform (I1).
        const std::uint8_t* mask = batchlas::detail::mask_ptr(batchlas::detail::call_mask("syr2k", A, B, C));

        batchlas::impl::parallel_for<Syr2kTriangularTilesKernel<T, TransOperand, AlignedFastPath, Conjugate>>(h,
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));
                if (bid >= batch || (mask != nullptr && mask[bid] == 0)) {
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
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < ThreadTile; ++i) {
BATCHLAS_UNROLL_FULL
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
                    // her2k stages A scaled (Conjugate only), so both products share one alpha.
                    auto stage = [&](const T* src, int ld, T* dst, int row0, bool scaled) {
                        auto sc = [&](T v) {
                            if constexpr (Conjugate) return scaled ? a_scale * v : v;
                            else return v;
                        };
                        if constexpr (TransOperand) {
                            const int row = row0 + s_row;
                            if constexpr (AlignedFastPath) {
                                const TileVec4<T> v =
                                    tile_load4(src + (k0 + s_l) +
                                               static_cast<std::ptrdiff_t>(row) * ld);
BATCHLAS_UNROLL_FULL
                                for (int i = 0; i < 4; ++i) {
                                    dst[(s_l + i) * Stride + s_row] = sc(v.v[i]);
                                }
                            } else {
BATCHLAS_UNROLL_FULL
                                for (int i = 0; i < 4; ++i) {
                                    const int gk = k0 + s_l + i;
                                    dst[(s_l + i) * Stride + s_row] =
                                        (gk < k && row < n)
                                        ? sc(src[gk + static_cast<std::ptrdiff_t>(row) * ld])
                                        : T(0);
                                }
                            }
                        } else {
                            const int gk = k0 + s_l;
                            if constexpr (AlignedFastPath) {
                                TileVec4<T> v = tile_load4(src + (row0 + s_row) +
                                                           static_cast<std::ptrdiff_t>(gk) * ld);
                                if constexpr (Conjugate) {
BATCHLAS_UNROLL_FULL
                                    for (int i = 0; i < 4; ++i) v.v[i] = sc(v.v[i]);
                                }
                                tile_store4(&dst[s_l * Stride + s_row], v);
                            } else {
BATCHLAS_UNROLL_FULL
                                for (int i = 0; i < 4; ++i) {
                                    const int row = row0 + s_row + i;
                                    dst[s_l * Stride + s_row + i] =
                                        (row < n && gk < k)
                                        ? sc(src[row + static_cast<std::ptrdiff_t>(gk) * ld])
                                        : T(0);
                                }
                            }
                        }
                    };

                    stage(Ab, lda, sam, m0, true);
                    stage(Bb, ldb, sbm, m0, false);
                    if (!on_diagonal) {
                        stage(Ab, lda, san, n0, true);
                        stage(Bb, ldb, sbn, n0, false);
                    }
                    item.barrier(sycl::access::fence_space::local_space);

BATCHLAS_UNROLL_FULL
                    for (int kk = 0; kk < TileK; ++kk) {
BATCHLAS_UNROLL_FULL
                        for (int product = 0; product < 2; ++product) {
                            const T* srow = product == 0 ? sam : sbm;
                            const T* scol = product == 0 ? sbn : san;
                            const TileVec4<T> r0 = tile_load4(&srow[kk * Stride + ty * Band]);
                            const TileVec4<T> r1 = tile_load4(&srow[kk * Stride + 64 + ty * Band]);
                            const TileVec4<T> c0 = tile_load4(&scol[kk * Stride + tx * Band]);
                            const TileVec4<T> c1 = tile_load4(&scol[kk * Stride + 64 + tx * Band]);
                            // her2k: op(X) op(Y)^H conjugates the column factor (NoTrans) or the row one.
                            constexpr bool kConjR = Conjugate && TransOperand;
                            constexpr bool kConjC = Conjugate && !TransOperand;
                            const T rf[ThreadTile] = {conj_if<kConjR>(r0.v[0]), conj_if<kConjR>(r0.v[1]),
                                                      conj_if<kConjR>(r0.v[2]), conj_if<kConjR>(r0.v[3]),
                                                      conj_if<kConjR>(r1.v[0]), conj_if<kConjR>(r1.v[1]),
                                                      conj_if<kConjR>(r1.v[2]), conj_if<kConjR>(r1.v[3])};
                            const T cf[ThreadTile] = {conj_if<kConjC>(c0.v[0]), conj_if<kConjC>(c0.v[1]),
                                                      conj_if<kConjC>(c0.v[2]), conj_if<kConjC>(c0.v[3]),
                                                      conj_if<kConjC>(c1.v[0]), conj_if<kConjC>(c1.v[1]),
                                                      conj_if<kConjC>(c1.v[2]), conj_if<kConjC>(c1.v[3])};
BATCHLAS_UNROLL_FULL
                            for (int i = 0; i < ThreadTile; ++i) {
BATCHLAS_UNROLL_FULL
                                for (int j = 0; j < ThreadTile; ++j) {
                                    accum[i][j] = accumulate(accum[i][j], rf[i], cf[j]);
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
                    T out = beta == T(0) ? alpha * value : alpha * value + beta * *p;
                    if constexpr (Conjugate) {
                        if (row == col) out = T(out.real(), typename T::value_type(0));  // Hermitian diagonal
                    }
                    *p = out;
                };

BATCHLAS_UNROLL_FULL
                for (int band = 0; band < 2; ++band) {
                    const int gm = m0 + band * 64 + ty * Band;
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < ThreadTile; ++j) {
                        const int gn = n0 + (j < Band ? tx * Band + j : 64 + tx * Band + j - Band);
                        if constexpr (AlignedFastPath) {
                            if (!on_diagonal) {
                                T* p = &Cb[gm + static_cast<std::ptrdiff_t>(gn) * ldc];
                                TileVec4<T> out;
                                if (beta == T(0)) {
BATCHLAS_UNROLL_FULL
                                    for (int i = 0; i < 4; ++i) {
                                        out.v[i] = alpha * accum[band * Band + i][j];
                                    }
                                } else {
                                    const TileVec4<T> prior = tile_load4(const_cast<const T*>(p));
BATCHLAS_UNROLL_FULL
                                    for (int i = 0; i < 4; ++i) {
                                        out.v[i] = alpha * accum[band * Band + i][j] +
                                            beta * prior.v[i];
                                    }
                                }
                                tile_store4(p, out);
                                continue;
                            }
                        }
BATCHLAS_UNROLL_FULL
                        for (int i = 0; i < 4; ++i) {
                            store_element(gm + i, gn, accum[band * Band + i][j]);
                        }
                    }
                }
            });
    });

    return ctx.get_event();
}

// Conjugate: her2k's alpha op(A) op(B)^H + conj(alpha) op(B) op(A)^H, with a real diagonal and
// beta real. A is staged times alpha (NoTrans) or conj(alpha) (ConjTrans) and the epilogue scale
// is 1, which is what gives the second product the conjugate scalar.
template <typename T, bool Conjugate = false>
Event syr2k_triangular_tiles(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             T alpha,
                             T beta,
                             Uplo uplo,
                             Transpose transA) {
    T a_scale = T(1);
    if constexpr (Conjugate) {
        a_scale = transA == Transpose::NoTrans ? alpha : T(alpha.real(), -alpha.imag());
        alpha = T(1);
    }
    const int n = static_cast<int>(C.rows());
    const int k = transA == Transpose::NoTrans ? static_cast<int>(A.cols())
                                               : static_cast<int>(A.rows());
    const bool fast = syr2k_triangular_fast_path(A, B, C, n, k);

    if (transA == Transpose::NoTrans) {
        return fast
            ? launch_syr2k_triangular_tiles<T, false, true, Conjugate>(ctx, A, B, C, alpha, beta, uplo, a_scale)
            : launch_syr2k_triangular_tiles<T, false, false, Conjugate>(ctx, A, B, C, alpha, beta, uplo, a_scale);
    }
    return fast ? launch_syr2k_triangular_tiles<T, true, true, Conjugate>(ctx, A, B, C, alpha, beta, uplo, a_scale)
                : launch_syr2k_triangular_tiles<T, true, false, Conjugate>(ctx, A, B, C, alpha, beta, uplo, a_scale);
}

} // namespace batchlas::backend::detail
