// SPDX-License-Identifier: MIT
#pragma once

// A 128x128x8 register-tiled SYR2K over the triangular tile set, fusing both
// rank-k products into the same accumulators:
//     accum += A[bi] * B[bj]^T   and   accum += B[bi] * A[bj]^T
// Products are issued sequentially, not interleaved, to keep one fragment pair
// live (occupancy). Diagonal tiles alias the bj-side tiles to the bi-side ones
// (block-uniform, no divergence) and mask their epilogue.
// Components: v2 ext/l3-triangular-tiles.hh (PacketStager, frag_ld4 + ger_acc, epilogue_tri_packets).
// evidence: docs/perf/level3.md#syrk-and-syr2k-triangular-tiles-kernel-design

#include <src/kernels/common/triangular_tiles.hh>
#include <batchlas/blas/device/v2/ext/l3-triangular-tiles.hh>

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

namespace dla = batchlas::device::v2;

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
    constexpr int LocalRows = TileM / ThreadTile;  // 16
    constexpr int LocalCols = TileN / ThreadTile;  // 16
    constexpr int Threads = LocalRows * LocalCols; // 256
    constexpr int Stride = TileM;
    // A thread's 8 rows (and 8 columns): two 4-wide bands, 64 apart.
    using Lanes = dla::Bands4<ThreadTile / 4, TileM / 2, 4>;
    // Staging walks the output index for NoTrans (contiguous in an n x k operand); Trans walks
    // k, contiguous in a k x n one, and transposes on the way into shared.
    using Stager = dla::PacketStager<TileM, TileK, Threads, TransOperand, AlignedFastPath>;

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
                dla::tri_tile_decode(static_cast<int>(item.get_group(1)), bi, bj);
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

                dla::RegTile<T, ThreadTile, ThreadTile> acc;
                dla::fill_tile(acc, T(0));

                const Stager stage(tid);
                // her2k stages A scaled (Conjugate only), so both products share one alpha.
                const dla::StageScale<T, Conjugate> scale_a{a_scale};
                // her2k: op(X) op(Y)^H conjugates the column factor (NoTrans) or the row one.
                constexpr bool kConjR = Conjugate && TransOperand;
                constexpr bool kConjC = Conjugate && !TransOperand;

                for (int k0 = 0; k0 < k; k0 += TileK) {
                    stage.run(k0, n, k, dla::Panel<T, dla::StageScale<T, Conjugate>>{Ab, lda, sam, m0, scale_a});
                    stage.run(k0, n, k, dla::Panel<T>{Bb, ldb, sbm, m0});
                    if (!on_diagonal) {
                        stage.run(k0, n, k, dla::Panel<T, dla::StageScale<T, Conjugate>>{Ab, lda, san, n0, scale_a});
                        stage.run(k0, n, k, dla::Panel<T>{Bb, ldb, sbn, n0});
                    }
                    item.barrier(sycl::access::fence_space::local_space);

BATCHLAS_UNROLL_FULL
                    for (int kk = 0; kk < TileK; ++kk) {
BATCHLAS_UNROLL_FULL
                        for (int product = 0; product < 2; ++product) {
                            const T* srow = product == 0 ? sam : sbm;
                            const T* scol = product == 0 ? sbn : san;
                            dla::Frag<T, ThreadTile> rf;
                            dla::Frag<T, ThreadTile> cf;
                            dla::frag_ld4<Lanes, kConjR>(rf, srow + kk * Stride, ty);
                            dla::frag_ld4<Lanes, kConjC>(cf, scol + kk * Stride, tx);
                            dla::ger_acc(acc, rf, cf);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Off the diagonal a 4-row band is one 128-bit access; on it the
                // triangle mask makes the band partial and the stores go one at a time.
                dla::epilogue_tri_packets<AlignedFastPath, Lanes, Lanes>(
                    dla::TriMask<Conjugate>{on_diagonal, lower}, acc, Cb, ldc, m0, n0, ty, tx, n, alpha, beta);
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
