// SPDX-License-Identifier: MIT
#pragma once

// Batched Side::Left TRMM that skips the zero half of A with a loop bound, not a mask: an output
// tile rooted at row m0 reduces only over the p its triangle reaches, and only the k-tile on the
// diagonal is masked (in the A staging). The saving is (R+1)/2R for R = m/TileM row tiles, not 2x,
// so the row tile is sized to m (trmm_row_tile). Layout and swizzle follow syrk_gram_tiles.hh.
// Components: v2 ext/l3-triangular-tiles.hh (stage_swz_kc/cc + trmat_op_at, frag_ld4 + ger_acc,
// epilogue_cm).
// evidence: docs/perf/level3.md#trmm-triangular-tiles-kernel-design

#include <src/kernels/common/triangular_tiles.hh>
#include <batchlas/blas/device/v2/ext/l3-triangular-tiles.hh>

#include <src/queue.hh>
#include <src/util/batch_mask.hh>
#include <src/util/kernel-trace.hh>
#include <src/sycl/kernel_attrs.hh>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <batchlas/settings.hh>

namespace batchlas::backend::detail {

namespace dla = batchlas::device::v2;

inline constexpr int kTrmmTileN = 128;
inline constexpr int kTrmmTileK = 16;

// Lanes per tile side: narrower sides keep the 4-wide fragment band and drop lanes, so
// ThreadRows stays a multiple of 4 (the 16-row tile gets 4 lanes).
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
    // Complex at TileM >= 64 halves the column lanes (4x16 thread tile) to get off the shared-load
    // limit; not below 64, where the block is already small and bandwidth bound.
    // evidence: docs/perf/level3.md#trmm-triangular-tiles-kernel-design
    constexpr int LocalCols = (is_std_complex_v<T> && TileM >= 64)
                                  ? trmm_lanes(TileN) / 2
                                  : trmm_lanes(TileN);
    constexpr int ThreadRows = TileM / LocalRows;
    constexpr int ThreadCols = TileN / LocalCols;
    constexpr int Threads = LocalRows * LocalCols;
    constexpr int SA = TileM;                    // aligned strides
    constexpr int SB = TileN;
    // A lane's rows (columns): ThreadRows / 4 bands of 4, one band per TileM / bands rows.
    using RowBands = dla::Bands4<ThreadRows / 4, TileM / (ThreadRows / 4), 4>;
    using ColBands = dla::Bands4<ThreadCols / 4, TileN / (ThreadCols / 4), 4>;
    using SwA = dla::Swizzle<TileM / 4>;
    using SwB = dla::Swizzle<TileN / 4>;

    static_assert(ThreadRows % 4 == 0 && ThreadCols % 4 == 0, "fragments are 4-wide bands");
    static_assert(TileM % TileK == 0, "the diagonal tile has to fall on a k boundary");

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

    impl::check_group_count(*ctx, sycl::nd_range<3>(global, local));
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
        // The batch mask (null unmasked); one item per work-group, so its return is uniform (I1).
        const std::uint8_t* mask = batchlas::detail::mask_ptr(batchlas::detail::call_mask("trmm", A, B, C));

        batchlas::impl::parallel_for<TrmmTriangularTilesKernel<T, TileM>>(h,
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));
                if (bid >= batch || (mask != nullptr && mask[bid] == 0)) {
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

                dla::RegTile<T, ThreadRows, ThreadCols> acc;
                dla::fill_tile(acc, T(0));

                // The whole point: an output tile rooted at row m0 reads only
                // the part of the reduction its triangle can reach.
                const int p_begin = lower_eff ? 0 : m0;
                const int p_end = lower_eff ? sycl::min(m, m0 + TileM) : m;

                for (int p0 = p_begin; p0 < p_end; p0 += TileK) {
                    // op(A) tile with triangle, unit diagonal and edges resolved here, so the
                    // inner loop sees a dense tile. Staging follows op(A)'s contiguous direction;
                    // NoTrans stores one 128-bit packet of four adjacent rows (single stores
                    // would hit 8 banks).
                    if (transposed) {
                        dla::stage_swz_kc<TileM, TileK, Threads>(sa, tid, m0, p0, [&](int i, int p) {
                            return dla::trmat_op_at<true>(Ab, lda, m, i, p, lower_eff, unit, conjugated);
                        });
                    } else {
                        dla::stage_swz_cc<TileM, TileK, Threads, true>(sa, tid, m0, p0, [&](int i, int p) {
                            return dla::trmat_op_at<false>(Ab, lda, m, i, p, lower_eff, unit, conjugated);
                        });
                    }
                    // B: reduction rows [p0, p0+TileK) against columns [n0, n0+TileN); B is
                    // contiguous down its row index, so a thread takes four adjacent reduction
                    // rows of one column.
                    dla::stage_swz_kc<TileN, TileK, Threads>(sb, tid, n0, p0, [&](int j, int p) {
                        return (j < n && p < m) ? Bb[p + static_cast<std::ptrdiff_t>(j) * ldb] : T(0);
                    });
                    item.barrier(sycl::access::fence_space::local_space);

#pragma unroll 4
                    for (int p = 0; p < TileK; ++p) {
                        dla::Frag<T, ThreadRows> af;
                        dla::Frag<T, ThreadCols> bf;
                        dla::frag_ld4<RowBands, false, SwA>(af, &sa[p * SA], tr, p);
                        dla::frag_ld4<ColBands, false, SwB>(bf, &sb[p * SB], tc, p);
                        dla::ger_acc(acc, af, bf);
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // TRMM overwrites C; there is no beta, so nothing here reads it.
                dla::epilogue_cm<RowBands, ColBands>(dla::FullMask{}, acc, Cb, ldc, m0, n0, tr, tc, m, n, alpha,
                                                     dla::Zero{});
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

// The row tile: a smaller tile does less arithmetic but re-reads B. The scalar type decides which
// wins (float: bandwidth bound, widest tile; double and complex: compute bound, narrowest), so the
// thresholds are per type. evidence: docs/perf/level3.md#trmm-choosing-the-row-tile-by-scalar-type
template <typename T>
inline int trmm_row_tile(int m) {
    constexpr bool complex_t = is_std_complex_v<T>;
    constexpr bool wide = sizeof(typename base_type<T>::type) > 4 || complex_t;
    if constexpr (wide) {
        // Complex stops at 32: its 64-row cell was a wash and the wider tile keeps the block larger.
        if (m <= (complex_t ? 32 : 64)) {
            return 16;
        }
        // Never 128: an 8x8 complex<double> thread tile fills the register file and the launch is rejected.
        return m <= 64 ? 32 : 64;
    } else {
        if (m <= 32) {
            return 32;
        }
        return m <= 512 ? 64 : 128;
    }
}
// The row tile the launch below instantiates: BATCHLAS_TRMM_TILE_M (0 = unset, not latched) or
// trmm_row_tile, rounded up to 16/32/64/128. can_run's grid term reads the same function.
template <typename T>
inline int trmm_launch_tile_m(int m) {
    const int forced = batchlas::settings().geometry.trmm_tile_m;
    const int tile_m = forced ? forced : trmm_row_tile<T>(m);
    return tile_m <= 16 ? 16 : (tile_m <= 32 ? 32 : (tile_m <= 64 ? 64 : 128));
}

// Work-groups in SYCL dim 1 (CUDA grid y, capped at 65535): one per (row tile, column tile).
template <typename T>
inline std::int64_t trmm_tile_groups(std::int64_t m, std::int64_t n) {
    const std::int64_t tile_m = trmm_launch_tile_m<T>(static_cast<int>(m));
    return ((m + tile_m - 1) / tile_m) * ((n + kTrmmTileN - 1) / kTrmmTileN);
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
    // BATCHLAS_TRMM_TILE_M pins the row tile so the trade-off above can be swept from one binary.
    const int tile_m = trmm_launch_tile_m<T>(static_cast<int>(C.rows()));
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
