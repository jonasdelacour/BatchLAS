// SPDX-License-Identifier: MIT
#pragma once
#include <cstddef>

#include <src/ops/gemm/gemm-components_v2.hh>
#include <src/ops/gemm/gemm_kernels.hh>

#include <sycl/sycl.hpp>

namespace batchlas::sycl_gemm {

namespace v2 = batchlas::device::v2;
namespace gc = batchlas::sycl_gemm::components;

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
        const std::uint8_t* mask = item_mask(A, B, C);

        batchlas::impl::parallel_for<GemmTiledGeneralKernel<T, Tile, OpA, OpB>>(h, sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
            const int bid = static_cast<int>(item.get_group(0));
            const int local_row = static_cast<int>(item.get_local_id(1));
            const int local_col = static_cast<int>(item.get_local_id(2));
            const int row = static_cast<int>(item.get_group(1) * Tile + local_row);
            const int col = static_cast<int>(item.get_group(2) * Tile + local_col);
            if (bid >= batch || item_off(mask, bid)) {
                return;
            }

            // Rebased per item in 64 bits (bid * stride passes 2^31); offsets within an item stay int.
            const T* const a_item = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
            const T* const b_item = b_ptr + static_cast<std::ptrdiff_t>(bid) * stride_b;
            T* const c_item = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;
            const v2::Grid2D<> ex{item};
            // This item's row of the A tile and column of the B tile, read through the accessors
            // (shared address space; a generic-pointer Vec cost 4 REG on complex<float> N/T x C).
            auto a_row = [&](int t) -> const T& { return tile_a[local_row * Tile + t]; };
            auto b_col = [&](int t) -> const T& { return tile_b[t * Tile + local_col]; };
            T sum = T(0);

            for (int kk0 = 0; kk0 < k; kk0 += Tile) {
                const int tile_col = kk0 + local_col;
                const int tile_row = kk0 + local_row;
                tile_a[local_row * Tile + local_col] =
                    (row < m && tile_col < k) ? OperandAccessor<T, OpA>::load(a_item, lda, 0, row, tile_col) : T(0);
                tile_b[local_row * Tile + local_col] =
                    (col < n && tile_row < k) ? OperandAccessor<T, OpB>::load(b_item, ldb, 0, tile_row, col) : T(0);
                ex.barrier();
                gc::dot_slab<Tile, v2::arith::Native>(sum, a_row, b_col, kk0, k);  // operator*: Annex G for complex, as tuned
                ex.barrier();
            }

            if (row < m && col < n) LinearEpilogue<T>::store(c_item + col * ldc + row, alpha, beta, sum);
        });
    });

    return ctx.get_event();
}

} // namespace batchlas::sycl_gemm