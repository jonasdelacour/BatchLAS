// SPDX-License-Identifier: MIT
#pragma once

#include <src/expansion_budget.hh>
#include <src/queue.hh>
#include <src/math-helpers.hh>
#include <batchlas/device.hh>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <cstddef>
#include <limits>

// Scratch expansions that turn a one-triangle operand (SYMM, HEMM, TRMM) into a
// dense one a batched GEMM can read; the caller's unreferenced storage is never
// read. Scratch is a workspace lease, never a fresh Matrix (freed while enqueued).
// Sizing and the fit ceiling live in ../expansion_budget.hh.
// evidence: docs/perf/level3.md#level-3-scratch-expansions-and-their-ceilings
namespace batchlas::backend::detail {

// Work-group shape for the elementwise expansions: lanes walk a column (both
// sides coalesce), and no more rows than the matrix has (tiny matrices).
struct ExpandGroupShape {
    int rows;
    int cols;
};

inline ExpandGroupShape expand_group_shape(int n) {
    constexpr int kItemsPerGroup = 256;
    constexpr int kMaxGroupRows = 32;
    int rows = 1;
    while (rows < kMaxGroupRows && rows < n) {
        rows *= 2;
    }
    return {rows, kItemsPerGroup / rows};
}

// Dense op of a triangular A: zeros opposite, ones on a unit diagonal. That
// storage is never read, so it may hold anything.
template <typename T>
Event expand_triangular(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& out,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        Uplo uplo,
                        Diag diag) {
    const int n = A.rows();
    const int batch = A.batch_size();
    const bool lower = uplo == Uplo::Lower;
    const bool unit = diag == Diag::Unit;

    const T* src = A.data_ptr();
    T* dst = out.data_ptr();
    const int lda = A.ld();
    const int ldo = out.ld();
    const std::size_t stride_a = static_cast<std::size_t>(A.stride());
    const std::size_t stride_o = static_cast<std::size_t>(out.stride());

    const auto shape = expand_group_shape(n);
    const sycl::range<3> global(static_cast<std::size_t>(batch),
                                static_cast<std::size_t>(::batchlas::detail::ceil_div(n, shape.cols) * shape.cols),
                                static_cast<std::size_t>(::batchlas::detail::ceil_div(n, shape.rows) * shape.rows));
    const sycl::range<3> local(1,
                               static_cast<std::size_t>(shape.cols),
                               static_cast<std::size_t>(shape.rows));

    ctx->parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
        const device::Grid2D<0, 0, 1> ex{item};
        const int b = ex.batch();
        device::tri_expand(ex, lower ? Uplo::Lower : Uplo::Upper, unit ? Diag::Unit : Diag::NonUnit,
                             device::mat_wide(src + static_cast<std::size_t>(b) * stride_a, n, n, lda),
                             device::mat_wide(dst + static_cast<std::size_t>(b) * stride_o, n, n, ldo));
    });

    return ctx.get_event();
}

// Tile edge of the mirrored expansion below, and the number of columns a work
// group covers per pass over it.
constexpr int kMirrorTile = 32;
constexpr int kMirrorGroupCols = 8;

// Full symmetric (Conjugate = false) or Hermitian (true) matrix from A's
// referenced triangle. One tile pair per work-group through local memory, so the
// read and both writes coalesce. evidence: docs/perf/level3.md#level-3-scratch-expansions-and-their-ceilings
template <typename T, bool Conjugate>
Event expand_mirrored(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& out,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      Uplo uplo) {
    const int n = A.rows();
    const int batch = A.batch_size();
    const int tiles = ::batchlas::detail::ceil_div(n, kMirrorTile);
    const bool lower = uplo == Uplo::Lower;

    const T* src = A.data_ptr();
    T* dst = out.data_ptr();
    const int lda = A.ld();
    const int ldo = out.ld();
    const std::size_t stride_a = static_cast<std::size_t>(A.stride());
    const std::size_t stride_o = static_cast<std::size_t>(out.stride());

    // Groups on the unreferenced side exit before their first barrier (cheaper
    // than a triangular-index sqrt per work-item).
    const sycl::range<3> global(static_cast<std::size_t>(batch),
                                static_cast<std::size_t>(tiles) * kMirrorGroupCols,
                                static_cast<std::size_t>(tiles) * kMirrorTile);
    const sycl::range<3> local(1, kMirrorGroupCols, kMirrorTile);

    impl::check_group_count(*ctx, sycl::nd_range<3>(global, local));
    ctx->submit([&](sycl::handler& cgh) {
        // Padded by one column so the transposed read strides across all banks.
        auto tile = sycl::local_accessor<T, 1>(
            sycl::range<1>(kMirrorTile * (kMirrorTile + 1)), cgh);

        batchlas::impl::parallel_for(cgh, sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
            const device::Grid2D<kMirrorTile, kMirrorGroupCols, 1> ex{item};
            const int ti = static_cast<int>(item.get_group(1));
            const int tj = static_cast<int>(item.get_group(2));
            if (ti < tj) {
                return;
            }
            const int b = ex.batch();
            // Row/column origin of the tile inside the referenced triangle.
            const int src_row0 = (lower ? ti : tj) * kMirrorTile;
            const int src_col0 = (lower ? tj : ti) * kMirrorTile;
            const auto a_b = device::mat_wide(src + static_cast<std::size_t>(b) * stride_a, n, n, lda);
            const auto o_b = device::mat_wide(dst + static_cast<std::size_t>(b) * stride_o, n, n, ldo);
            const auto t = device::mat<device::Mem::Local>(::batchlas::util::get_raw_ptr(tile), kMirrorTile, kMirrorTile, kMirrorTile + 1);

            device::tile_load<kMirrorTile>(ex, a_b, src_row0, src_col0, t);
            sycl::group_barrier(item.get_group());
            // A diagonal tile writes the referenced member of each pair (the two writes would collide).
            if (ti == tj) {
                device::tile_store_diag<Conjugate, kMirrorTile>(ex, lower ? Uplo::Lower : Uplo::Upper, t, o_b, src_row0);
                return;
            }
            device::tile_store_pair<Conjugate, kMirrorTile>(ex, t, o_b, src_row0, src_col0);
        });
    });

    return ctx.get_event();
}

} // namespace batchlas::backend::detail
