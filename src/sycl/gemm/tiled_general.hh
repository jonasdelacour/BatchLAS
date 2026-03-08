#pragma once

#include "accessors.hh"
#include "epilogue_linear.hh"

#include <sycl/sycl.hpp>

namespace batchlas::sycl_gemm {

template <typename T, int Tile, Transpose OpA, Transpose OpB>
class GemmTiledGeneralKernel;

template <typename T, int Tile, Transpose OpA, Transpose OpB>
Event launch_tiled_general(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           const MatrixView<T, MatrixFormat::Dense>& C,
                           T alpha,
                           T beta,
                           const char* (*kernel_trace_name)(KernelVariant)) {
    BATCHLAS_KERNEL_TRACE_SCOPE(kernel_trace_name(KernelVariant::Tiled16));

    const auto [m, k] = get_effective_dims(A, OpA);
    const auto [_, n] = get_effective_dims(B, OpB);
    static_cast<void>(_);

    const sycl::range<3> local(1, Tile, Tile);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()),
                                static_cast<size_t>(((m + Tile - 1) / Tile) * Tile),
                                static_cast<size_t>(((n + Tile - 1) / Tile) * Tile));

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> tile_a(sycl::range<1>(Tile * Tile), h);
        sycl::local_accessor<T, 1> tile_b(sycl::range<1>(Tile * Tile), h);

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

        h.parallel_for<GemmTiledGeneralKernel<T, Tile, OpA, OpB>>(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
            const int bid = static_cast<int>(item.get_group(0));
            const int local_row = static_cast<int>(item.get_local_id(1));
            const int local_col = static_cast<int>(item.get_local_id(2));
            const int row = static_cast<int>(item.get_group(1) * Tile + local_row);
            const int col = static_cast<int>(item.get_group(2) * Tile + local_col);
            if (bid >= batch) {
                return;
            }

            const int batch_a = bid * stride_a;
            const int batch_b = bid * stride_b;
            const int batch_c = bid * stride_c;
            T sum = T(0);

            for (int kk0 = 0; kk0 < k; kk0 += Tile) {
                const int tile_col = kk0 + local_col;
                const int tile_row = kk0 + local_row;
                tile_a[local_row * Tile + local_col] = (row < m && tile_col < k)
                    ? OperandAccessor<T, OpA>::load(a_ptr, lda, batch_a, row, tile_col)
                    : T(0);
                tile_b[local_row * Tile + local_col] = (col < n && tile_row < k)
                    ? OperandAccessor<T, OpB>::load(b_ptr, ldb, batch_b, tile_row, col)
                    : T(0);

                item.barrier(sycl::access::fence_space::local_space);
                for (int t = 0; t < Tile && kk0 + t < k; ++t) {
                    sum += tile_a[local_row * Tile + t] * tile_b[t * Tile + local_col];
                }
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (row < m && col < n) {
                const T prior = c_ptr[batch_c + col * ldc + row];
                c_ptr[batch_c + col * ldc + row] = LinearEpilogue<T>::apply(alpha, beta, sum, prior);
            }
        });
    });

    return ctx.get_event();
}

} // namespace batchlas::sycl_gemm