#include <blas/extra.hh>
#include "../queue.hh"

namespace batchlas {

    template <typename T, MatrixFormat MF>
    struct TransposeKernel;

    template <typename T, MatrixFormat MF>
    Event transpose_impl(Queue &ctx,
                         const MatrixView<T, MF> &A,
                         const MatrixView<T, MF> &B) {
        constexpr int TILE = 16;
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
            cgh.parallel_for<TransposeKernel<T, MF>>(
                sycl::nd_range<3>(
                    sycl::range<3>(batch_size, grid_x * TILE, grid_y * TILE),
                    sycl::range<3>(1, TILE, TILE)),
                [=](sycl::nd_item<3> item) {
                    size_t b = item.get_group(0);
                    size_t tile_x = item.get_group(1) * TILE;
                    size_t tile_y = item.get_group(2) * TILE;
                    size_t local_x = item.get_local_id(1);
                    size_t local_y = item.get_local_id(2);

                    size_t row = tile_y + local_y;
                    size_t col = tile_x + local_x;

                    T val = 0;
                    if (row < rows && col < cols) {
                        val = src[b * stride_src + col * ld_src + row];
                    }
                    tile[local_y][local_x] = val;
                    item.barrier(sycl::access::fence_space::local_space);

                    row = tile_x + local_y;
                    col = tile_y + local_x;
                    if (row < cols && col < rows) {
                        dst[b * stride_dst + col * ld_dst + row] = tile[local_x][local_y];
                    }
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
        transpose_impl(ctx, A, result.view());
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

