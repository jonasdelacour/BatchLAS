// SPDX-License-Identifier: MIT
#pragma once

// A 128x128x8 register-tiled SYRK over the triangular tile set: tiles outside
// the requested triangle are never launched, and diagonal tiles mask their
// epilogue (the other half is neither read nor written). Inner loop as in
// src/ops/gemm/kernels/register_128x128.hh; both operands are A at two row offsets. Any scalar:
// packets are 128-bit only for float (v2 ld4), and complex MACs are written out (v2 mac_value).
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
class SyrkTriangularTilesKernel;

// Does this problem satisfy everything the unpredicated path assumes? A tile
// that is not a whole 128x128 of C, or a k the 8-deep staging cannot fill
// exactly, has to take the predicated path, as does any operand whose base or
// leading dimension breaks the 4-element alignment the packet forms need.
template <typename T>
bool syrk_triangular_fast_path(const MatrixView<T, MatrixFormat::Dense>& A,
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
        aligned(C.data_ptr(), C.ld(), C.stride());
}

template <typename T, bool TransOperand, bool AlignedFastPath, bool Conjugate>
Event launch_syrk_triangular_tiles(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                   const MatrixView<T, MatrixFormat::Dense>& C,
                                   T alpha,
                                   T beta,
                                   Uplo uplo) {
    BATCHLAS_KERNEL_TRACE_SCOPE("syrk_cuda_custom.triangular_tiles");

    constexpr int TileM = kTriangularTile;
    constexpr int TileN = kTriangularTile;
    constexpr int TileK = kTriangularTileK;
    constexpr int ThreadTile = 8;                  // rows and cols per thread
    constexpr int LocalRows = TileM / ThreadTile;  // 16
    constexpr int LocalCols = TileN / ThreadTile;  // 16
    constexpr int Threads = LocalRows * LocalCols; // 256
    // A thread's 8 rows (and 8 columns): two 4-wide bands, 64 apart.
    using Lanes = dla::Bands4<ThreadTile / 4, TileM / 2, 4>;
    // Staging walks the output index for NoTrans (contiguous in an n x k A); Trans walks k,
    // contiguous in a k x n A, and transposes on the way into shared.
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
        sycl::local_accessor<T, 1> tile_a(sycl::range<1>(TileK * TileM), h);
        sycl::local_accessor<T, 1> tile_b(sycl::range<1>(TileK * TileN), h);

        const T* a_ptr = A.data_ptr();
        T* c_ptr = C.data_ptr();
        const int lda = A.ld();
        const int ldc = C.ld();
        const int stride_a = A.stride();
        const int stride_c = C.stride();
        const int batch = A.batch_size();
        const bool lower = uplo == Uplo::Lower;
        // The batch mask (null unmasked); one item per work-group, so its return is uniform (I1).
        const std::uint8_t* mask = batchlas::detail::mask_ptr(batchlas::detail::call_mask("syrk", A, C));

        batchlas::impl::parallel_for<SyrkTriangularTilesKernel<T, TransOperand, AlignedFastPath, Conjugate>>(h,
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
                T* Cb = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;

                T* sa = tile_a.template get_multi_ptr<sycl::access::decorated::no>().get();
                T* sb = tile_b.template get_multi_ptr<sycl::access::decorated::no>().get();

                dla::RegTile<T, ThreadTile, ThreadTile> acc;
                dla::fill_tile(acc, T(0));

                const Stager stage(tid);
                const dla::Panel<T> pa{Ab, lda, sa, m0};
                const dla::Panel<T> pb{Ab, lda, sb, n0};
                // herk: A A^H conjugates the column factor, A^H A the row factor.
                constexpr bool kConjA = Conjugate && TransOperand;
                constexpr bool kConjB = Conjugate && !TransOperand;

                for (int k0 = 0; k0 < k; k0 += TileK) {
                    stage(k0, n, k, pa, pb);
                    item.barrier(sycl::access::fence_space::local_space);
BATCHLAS_UNROLL_FULL
                    for (int kk = 0; kk < TileK; ++kk) {
                        dla::Frag<T, ThreadTile> af;
                        dla::Frag<T, ThreadTile> bf;
                        dla::frag_ld4<Lanes, kConjA>(af, sa + kk * TileM, ty);
                        dla::frag_ld4<Lanes, kConjB>(bf, sb + kk * TileN, tx);
                        dla::ger_acc(acc, af, bf);
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

// Conjugate: herk's A A^H (NoTrans) or A^H A (ConjTrans), with a real diagonal; alpha and beta
// then carry herk's real scalars.
template <typename T, bool Conjugate = false>
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
        return fast ? launch_syrk_triangular_tiles<T, false, true, Conjugate>(ctx, A, C, alpha, beta, uplo)
                    : launch_syrk_triangular_tiles<T, false, false, Conjugate>(ctx, A, C, alpha, beta, uplo);
    }
    return fast ? launch_syrk_triangular_tiles<T, true, true, Conjugate>(ctx, A, C, alpha, beta, uplo)
                : launch_syrk_triangular_tiles<T, true, false, Conjugate>(ctx, A, C, alpha, beta, uplo);
}

} // namespace batchlas::backend::detail
