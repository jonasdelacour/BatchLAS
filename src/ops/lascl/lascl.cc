// SPDX-License-Identifier: MIT
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/queue-dispatch.hh>
#include <batchlas/device.hh>
#include <src/queue.hh>
#include <src/util/batch_mask.hh>

namespace batchlas {

template <typename T>
struct LasclKernel;

// One batched 2-D launch; each element applies core lascl's safe steps (every item computes the
// same steps, so no step is a separate kernel). Inactive batch items are not touched.
template <MatrixFormat MType, typename T>
Event lascl(Queue& ctx, const MatrixView<T, MType>& mat, T cfrom, T cto) {
    static_assert(MType == MatrixFormat::Dense, "lascl: dense format");
    const std::uint8_t* mk = detail::mask_ptr(detail::call_mask("lascl", mat));
    detail::require_device_fp64<T>(ctx, "lascl");
    const int m = mat.rows(), n = mat.cols(), batch = mat.batch_size();
    if (m == 0 || n == 0 || batch == 0) return ctx.get_event();
    const bool masked = mk != nullptr;  // a captured null pointer can read as non-null on AdaptiveCpp Metal
    const auto av = mat.kernel_view();
    ctx->parallel_for<LasclKernel<T>>(device::batched2d_range(batch, m, n, device::batched2d_shape(m, n)), [=](sycl::nd_item<3> item) {
        const device::Grid2D<> ex{item};
        ex.for_each_batch(batch, [&](int b) {
            if (masked && !mk[b]) return;
            device::lascl(ex, cfrom, cto, device::item(av, b, m, n));
        });
    });
    return ctx.get_event();
}

template Event lascl<MatrixFormat::Dense, float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&, float, float);
template Event lascl<MatrixFormat::Dense, double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&, double, double);

}
