// SPDX-License-Identifier: MIT
#include <batchlas/blas/extra.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/kernels/common/elementwise_glue_v2.hh>
#include <src/queue.hh>
#include <src/util/batch_mask.hh>

namespace batchlas {

    template <typename T, MatrixFormat MF>
    struct TransposeKernel;

    template <typename T, MatrixFormat MF>
    Event transpose_impl(Queue &ctx,
                         const MatrixView<T, MF> &A,
                         const MatrixView<T, MF> &B) {
        detail::require_device_fp64<T>(ctx, "transpose");
        constexpr int TILE = 16;
        const std::uint8_t* mask = detail::mask_ptr(detail::call_mask("transpose", A, B));
        ctx->submit([&](sycl::handler &cgh) {
            auto src = A.data_ptr();
            auto dst = B.data_ptr();
            auto rows = A.rows();
            auto cols = A.cols();
            auto ld_src = A.ld();
            auto ld_dst = B.ld();
            auto stride_src = A.stride();
            auto stride_dst = B.stride();
            auto batch_size = A.batch_size();
            sycl::local_accessor<T, 2> tile(sycl::range<2>(TILE, TILE + 1), cgh);
            size_t grid_x = (cols + TILE - 1) / TILE;
            size_t grid_y = (rows + TILE - 1) / TILE;
            batchlas::impl::parallel_for<TransposeKernel<T, MF>>(cgh,
                sycl::nd_range<3>(
                    sycl::range<3>(batch_size, grid_x * TILE, grid_y * TILE),
                    sycl::range<3>(1, TILE, TILE)),
                [=](sycl::nd_item<3> item) {
                    const device::Grid2D<TILE, TILE, 1> ex{item};
                    const int b = ex.batch();
                    if (mask && !mask[b]) return;  // uniform per work-group: before the barrier
                    // B = A^T through a padded SLM tile that holds the A tile transposed.
                    const auto t = device::mat<device::Mem::Local>(util::get_raw_ptr(tile), TILE, TILE, TILE + 1);
                    const int row0 = static_cast<int>(item.get_group(2)) * TILE, col0 = static_cast<int>(item.get_group(1)) * TILE;
                    device::glue::tile_load_trans<TILE>(ex, device::mat(src + static_cast<size_t>(b) * stride_src, rows, cols, ld_src),
                                                    row0, col0, t);
                    ex.barrier();
                    device::glue::tile_store<TILE>(ex, t, device::mat(dst + static_cast<size_t>(b) * stride_dst, cols, rows, ld_dst),
                                               col0, row0);
                });
        });
        return ctx.get_event();
    }

    template <typename T, MatrixFormat MF>
    Event transpose(Queue &ctx,
                    const MatrixView<T, MF> &A,
                    const MatrixView<T, MF> &B) {
        return transpose_impl(ctx, A, B);
    }

    template <typename T, MatrixFormat MF>
    Matrix<T, MF> transpose(Queue &ctx,
                            const MatrixView<T, MF> &A) {
        Matrix<T, MF> result(A.cols(), A.rows(), A.batch_size());
        // The result keeps A's mask (copied on ctx, after its producer); its inactive items stay uninitialised.
        if (A.has_batch_mask()) ctx->memcpy(result.enable_batch_mask().data(), A.batch_mask().data(), A.batch_mask().size());
        // (void) on an Event: deliberate. This Queue is in-order, so the next submission
        // is already ordered after this one and the Event carries nothing the caller needs.
        (void)transpose_impl(ctx, A, result);
        return result;
    }

#define TRANSPOSE_INSTANTIATE(fp, fmt)                           \
    template Event transpose<fp, fmt>(Queue &,                  \
                                      const MatrixView<fp, fmt> &, \
                                      const MatrixView<fp, fmt> &); \
    template Matrix<fp, fmt> transpose<fp, fmt>(Queue &,         \
                                                const MatrixView<fp, fmt> &);

    TRANSPOSE_INSTANTIATE(float, MatrixFormat::Dense)
    TRANSPOSE_INSTANTIATE(double, MatrixFormat::Dense)
    //TRANSPOSE_INSTANTIATE(std::complex<float>, MatrixFormat::Dense)
    //TRANSPOSE_INSTANTIATE(std::complex<double>, MatrixFormat::Dense)

} // namespace batchlas

