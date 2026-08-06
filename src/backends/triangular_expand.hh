#pragma once

#include "../queue.hh"
#include "cublasdx_dispatch_common.hh"

#include <blas/enums.hh>
#include <blas/matrix.hh>
#include <util/mempool.hh>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>

// Scratch expansions that turn a matrix whose meaning lives in one triangle
// into an ordinary dense operand a batched GEMM can read.
//
// SYMM, HEMM and TRMM all need this: BLAS forbids any of them from touching the
// unreferenced triangle, so pointing a GEMM at the caller's A is wrong even
// when the caller happens to have zeroed it. The expansion is written into a
// workspace lease rather than a fresh Matrix -- a Matrix is a managed
// allocation whose pages migrate on first touch, which at n=512 batch=512 costs
// several times the GEMM it feeds, and it would be freed on return while the
// kernels reading it have only been enqueued.
namespace batchlas::backend::detail {

// Leading dimension of an expanded copy. The caller's own ld is irrelevant --
// the expansion writes every element -- so pack the columns and pad only to
// 16 bytes, which is the alignment the vendor and cuBLASDx GEMM kernels want
// before they will use packet loads.
template <typename T>
int expanded_ld(int n) {
    constexpr int elements_per_packet = std::max<int>(1, 16 / sizeof(T));
    return ceil_div(n, elements_per_packet) * elements_per_packet;
}

template <typename T>
std::size_t expanded_workspace_bytes(Queue& ctx, int n, int batch) {
    auto sizer = BumpAllocator::measuring();
    sizer.allocate<T>(ctx, static_cast<std::size_t>(expanded_ld<T>(n)) *
                               static_cast<std::size_t>(n) *
                               static_cast<std::size_t>(batch));
    return sizer.required_bytes();
}

// Whether an n x n x batch expansion can be built at all. Two ceilings, both
// hard rather than tuned:
//
//   - SYCL linearises the global id, and the runtime rejects a range whose
//     product does not fit in an int. The grid is one work item per element, so
//     it hits that at 2^31 elements -- measured, as a thrown sycl::exception at
//     n = 2048 batch = 512.
//   - The scratch shares the device with A, B and C, which for a square problem
//     are together about three times its size. A quarter of global memory
//     leaves room for them; at n = 2048 batch = 256 that is 4.3 GB of scratch
//     inside 17 GB of live operands, which runs.
//
// A caller that exceeds either has to fall back to whatever route needs no
// scratch.
//
// BATCHLAS_EXPAND_MAX_BYTES lowers the memory ceiling, for sharing a device
// with something else -- and for reaching the no-scratch fallback from a test
// without allocating gigabytes to get there.
inline bool expansion_fits(const Queue& ctx, int n, int batch, std::size_t bytes) {
    const std::size_t elements = static_cast<std::size_t>(n) *
                                 static_cast<std::size_t>(n) *
                                 static_cast<std::size_t>(batch);
    if (elements > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    std::size_t budget = ctx.device().get_property(DeviceProperty::GLOBAL_MEM_SIZE) / 4;
    if (const char* capped = std::getenv("BATCHLAS_EXPAND_MAX_BYTES")) {
        budget = std::min(budget, static_cast<std::size_t>(std::strtoull(capped, nullptr, 10)));
    }
    return bytes <= budget;
}

// Work-group shape for the elementwise expansions below: rows first, so that a
// group's lanes walk a column and both the load and the store coalesce, and
// only as many rows as the matrix actually has, so that a batch of tiny
// matrices does not retire mostly-idle groups.
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

// Materialise the dense matrix that A's referenced triangle stands for: zeros
// opposite it, and ones on the diagonal when the caller declared it unit --
// storage that TRMM is not allowed to read, and that therefore may hold
// anything at all.
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

// Materialise the full symmetric (Conjugate = false) or Hermitian
// (Conjugate = true) matrix that A's referenced triangle stands for, so a plain
// batched GEMM can read it as an ordinary dense operand.
//
// One tile pair per work group, staged through local memory. The mirrored half
// is the reason: the mirror of a coalesced column read is a row write, one
// cache line per element, and at these sizes the expansion is pure bandwidth.
// Going through a tile keeps the read and both writes coalesced and moves
// 1.5 n^2 of traffic, against the 3 n^2 of a copy followed by an in-place
// symmetrize.
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

    // The tile grid covers both triangles and the groups on the unreferenced
    // side exit before their first barrier. Half the groups retire empty, which
    // is cheaper than putting the integer square root of an unranked triangular
    // index in front of every work item.
    const sycl::range<3> global(static_cast<std::size_t>(batch),
                                static_cast<std::size_t>(tiles) * kMirrorGroupCols,
                                static_cast<std::size_t>(tiles) * kMirrorTile);
    const sycl::range<3> local(1, kMirrorGroupCols, kMirrorTile);

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
                // The two writes below would collide on a diagonal tile, so pick
                // the referenced member of each mirrored pair instead. The
                // diagonal itself is the one element a Hermitian matrix pins
                // rather than mirrors: A = A^H forces its imaginary part to
                // zero, so whatever the caller stored there is not part of the
                // operand.
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
