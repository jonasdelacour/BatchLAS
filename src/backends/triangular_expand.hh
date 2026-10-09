#pragma once

#include "../expansion_budget.hh"
#include "../queue.hh"
#include "route_common.hh"

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <batchlas/settings.hh>

// Scratch expansions that turn a one-triangle operand (SYMM, HEMM, TRMM) into a
// dense one a batched GEMM can read; the caller's unreferenced storage is never
// read. Scratch is a workspace lease, never a fresh Matrix (freed while enqueued).
// Sizing and the fit ceiling live in ../expansion_budget.hh.
// evidence: docs/perf/level3.md#level-3-scratch-expansions-and-their-ceilings
namespace batchlas::backend::detail {

// hemm's expansion vs its per-item vendor loop (symm and trmm choose from tables now). The
// constants are deliberately more conservative than the measured loss region.
// evidence: docs/perf/level3.md#symm-and-hemm-expansion-crossover
constexpr int kExpandMinBatch = 4;
constexpr int kExpandMinDim = 256;

// BATCHLAS_EXPAND_ROUTE pins "expand" or "loop"; it only ever narrows expansion_fits.
inline bool expansion_preferred(int max_dim, int batch) {
    // Same Settings field as expansion_route_pin(), so the two cannot disagree.
    if (const char* route = batchlas::settings().selection.expand_route.get()) {
        if (std::string_view(route) == "expand") {
            return true;
        }
        if (std::string_view(route) == "loop") {
            return false;
        }
    }
    return batch >= kExpandMinBatch || max_dim >= kExpandMinDim;
}

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
                                static_cast<std::size_t>(ceil_div(n, shape.cols) * shape.cols),
                                static_cast<std::size_t>(ceil_div(n, shape.rows) * shape.rows));
    const sycl::range<3> local(1,
                               static_cast<std::size_t>(shape.cols),
                               static_cast<std::size_t>(shape.rows));

    ctx->parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
        const int i = static_cast<int>(item.get_global_id(2));
        const int j = static_cast<int>(item.get_global_id(1));
        if (i >= n || j >= n) {
            return;
        }
        const int b = static_cast<int>(item.get_group(0));

        T value;
        if (i == j) {
            value = unit ? T(1)
                         : src[static_cast<std::size_t>(b) * stride_a +
                               static_cast<std::size_t>(j) * lda + i];
        } else if (lower ? (i > j) : (i < j)) {
            value = src[static_cast<std::size_t>(b) * stride_a +
                        static_cast<std::size_t>(j) * lda + i];
        } else {
            value = T(0);
        }

        dst[static_cast<std::size_t>(b) * stride_o + static_cast<std::size_t>(j) * ldo + i] = value;
    });

    return ctx.get_event();
}

// Tile edge of the mirrored expansion below, and the number of columns a work
// group covers per pass over it.
constexpr int kMirrorTile = 32;
constexpr int kMirrorGroupCols = 8;

// The mirrored half of a Hermitian matrix is the conjugate of the referenced
// one; of a symmetric matrix it is the element itself.
template <bool Conjugate, typename T>
inline T mirror_of(T value) {
    if constexpr (Conjugate) {
        return T(value.real(), -value.imag());
    } else {
        return value;
    }
}

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
    const int tiles = ceil_div(n, kMirrorTile);
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

        cgh.parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
            const int ti = static_cast<int>(item.get_group(1));
            const int tj = static_cast<int>(item.get_group(2));
            if (ti < tj) {
                return;
            }

            const int b = static_cast<int>(item.get_group(0));
            const int r = static_cast<int>(item.get_local_id(2));
            const int c0 = static_cast<int>(item.get_local_id(1));

            // Row/column origin of the tile inside the referenced triangle.
            const int src_row0 = (lower ? ti : tj) * kMirrorTile;
            const int src_col0 = (lower ? tj : ti) * kMirrorTile;

            const T* src_batch = src + static_cast<std::size_t>(b) * stride_a;
            T* dst_batch = dst + static_cast<std::size_t>(b) * stride_o;

            for (int c = c0; c < kMirrorTile; c += kMirrorGroupCols) {
                const int i = src_row0 + r;
                const int j = src_col0 + c;
                tile[c * (kMirrorTile + 1) + r] =
                    (i < n && j < n) ? src_batch[static_cast<std::size_t>(j) * lda + i] : T(0);
            }

            sycl::group_barrier(item.get_group());

            if (ti == tj) {
                // Diagonal tile: pick the referenced member of each pair (the
                // two writes would collide). A Hermitian diagonal is forced real.
                for (int c = c0; c < kMirrorTile; c += kMirrorGroupCols) {
                    const int i = src_row0 + r;
                    const int j = src_col0 + c;
                    if (i >= n || j >= n) {
                        continue;
                    }
                    const bool referenced = lower ? (r >= c) : (r <= c);
                    T value = referenced ? tile[c * (kMirrorTile + 1) + r]
                                         : mirror_of<Conjugate>(tile[r * (kMirrorTile + 1) + c]);
                    if constexpr (Conjugate) {
                        if (i == j) {
                            value = T(value.real(), 0);
                        }
                    }
                    dst_batch[static_cast<std::size_t>(j) * ldo + i] = value;
                }
                return;
            }

            for (int c = c0; c < kMirrorTile; c += kMirrorGroupCols) {
                if (src_row0 + r < n && src_col0 + c < n) {
                    dst_batch[static_cast<std::size_t>(src_col0 + c) * ldo + (src_row0 + r)] =
                        tile[c * (kMirrorTile + 1) + r];
                }
                if (src_col0 + r < n && src_row0 + c < n) {
                    dst_batch[static_cast<std::size_t>(src_row0 + c) * ldo + (src_col0 + r)] =
                        mirror_of<Conjugate>(tile[r * (kMirrorTile + 1) + c]);
                }
            }
        });
    });

    return ctx.get_event();
}

} // namespace batchlas::backend::detail
