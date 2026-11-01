// SPDX-License-Identifier: MIT
#pragma once

// A single-tile batched SYRK/HERK for n <= kGramMaxTile (ortho's tall, skinny
// A). The tile is sized to n, one shared tile serves both operands, and only the
// thread tiles meeting the requested triangle are carried. The shared stride is
// exactly n (unpadded) so fragments stay LDS.128; the swizzle below keeps that
// conflict-free. Staging assignment depends on the transpose mode.
// Components: v2 ext/l3-triangular-tiles.hh (stage_swz_kc/cc, frag_ld4 + ger_acc, epilogue_cm).
// evidence: docs/perf/level3.md#syrk-gram-tiles-kernel-design

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
#include <type_traits>

namespace batchlas::backend::detail {

namespace dla = batchlas::device::v2;

// The output tile is the whole matrix, so this is also the largest n the
// kernel can serve.
inline constexpr int kGramMaxTile = 128;

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
    // Invariant: a thread's rows/columns are contiguous. Two bands 64 apart (as in the
    // 128x128 kernels) silently leave elements unwritten.
    // evidence: docs/perf/level3.md#the-band-split-syrk-bug
    using Band = dla::Bands4<ThreadTile / 4, 4, ThreadTile>;
    using Sw = dla::Swizzle<NTile / 4>;

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
        // The batch mask (null unmasked); one item per work-group, so its return is uniform (I1).
        const std::uint8_t* mask = batchlas::detail::mask_ptr(batchlas::detail::call_mask("syrk", A, C));

        batchlas::impl::parallel_for<SyrkGramTilesKernel<T, NTile, ThreadTile, KC, TransOperand, Conjugate>>(h,
            sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                const int bid = static_cast<int>(item.get_group(0));
                if (bid >= batch || (mask != nullptr && mask[bid] == 0)) {
                    return;
                }
                const int tid = static_cast<int>(item.get_local_id(1));
                const bool active = tid < TriTiles;

                int bi = 0;
                int bj = 0;
                dla::tri_tile_decode(active ? tid : 0, bi, bj);
                const int tr = lower ? bi : bj;   // row tile
                const int tc = lower ? bj : bi;   // column tile

                const T* Ab = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
                T* Cb = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;
                T* sh = tile.template get_multi_ptr<sycl::access::decorated::no>().get();

                dla::RegTile<T, ThreadTile, ThreadTile> acc;
                dla::fill_tile(acc, T(0));
                // Conjugate the operand carrying the ^H (row for ConjTrans, column for
                // NoTrans). The wrong one is still Hermitian: test vs GEMM.
                constexpr bool kConjR = Conjugate && TransOperand;
                constexpr bool kConjC = Conjugate && !TransOperand;

                for (int k0 = 0; k0 < k; k0 += KC) {
                    // Zero-filled to the full KC x NTile, so the inner loop has no bounds checks.
                    if constexpr (TransOperand) {
                        // Coalesced global side; the shared side pays a conflict (written once).
                        dla::stage_swz_kc<NTile, KC, Threads>(sh, tid, 0, k0, [&](int col, int gk) {
                            return (col < n && gk < k) ? Ab[gk + static_cast<std::ptrdiff_t>(col) * lda] : T(0);
                        });
                    } else {
                        dla::stage_swz_cc<NTile, KC, Threads, false>(sh, tid, 0, k0, [&](int col, int gk) {
                            return (gk < k && col < n) ? Ab[col + static_cast<std::ptrdiff_t>(gk) * lda] : T(0);
                        });
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    if (active) {
#pragma unroll 4
                        for (int p = 0; p < KC; ++p) {
                            const T* row = &sh[p * SPad];
                            dla::Frag<T, ThreadTile> af;
                            dla::Frag<T, ThreadTile> bf;
                            dla::frag_ld4<Band, kConjR, Sw>(af, row, tr, p);
                            dla::frag_ld4<Band, kConjC, Sw>(bf, row, tc, p);
                            dla::ger_acc(acc, af, bf);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (!active) {
                    return;
                }

                // Only the requested triangle is read or written; only diagonal thread tiles
                // need the element mask, and HERK drops its diagonal's rounding residue.
                dla::epilogue_cm<Band, Band>(dla::TriMask<Conjugate>{bi == bj, lower}, acc, Cb, ldc, 0, 0, tr, tc, n, n,
                                             alpha, beta);
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
