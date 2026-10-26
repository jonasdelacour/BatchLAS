// Masked batches: the gather/scatter between a batch and the packed copy of its active items
// (batch_mask.hh). One work-item per element; no shared or sub-group state.

#include <src/kernels/common/batch_mask.hh>

#include <src/queue.hh>

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>

namespace batchlas::batch_mask {

namespace {
template <typename T> class BatchMaskMoveKernel;
}  // namespace

template <typename T>
Event move_items(Queue& ctx, T* full, std::int64_t ld, std::int64_t stride, T* packed, std::int64_t pld,
                 std::int64_t pstride, int rows, int cols, Span<const std::int32_t> idx, bool to_packed) {
    const std::int32_t* const ix = idx.data();
    const std::size_t k = idx.size();
    const std::size_t elems = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (k == 0 || elems == 0) return ctx.create_event_after_external_work();
    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<BatchMaskMoveKernel<T>>(sycl::range<2>(k, elems), [=](sycl::item<2> it) {
            const std::size_t g = it.get_id(0), e = it.get_id(1);
            const std::int64_t r = static_cast<std::int64_t>(e % static_cast<std::size_t>(rows));
            const std::int64_t c = static_cast<std::int64_t>(e / static_cast<std::size_t>(rows));
            T* const f = full + static_cast<std::int64_t>(ix[g]) * stride + c * ld + r;
            T* const p = packed + static_cast<std::int64_t>(g) * pstride + c * pld + r;
            if (to_packed) *p = *f;
            else *f = *p;
        });
    });
    return ctx.get_event();
}

void upload(Queue& q, Span<std::int32_t> dst, const std::vector<std::int32_t>& idx) {
    if (!idx.empty()) q->memcpy(dst.data(), idx.data(), idx.size() * sizeof(std::int32_t)).wait();
}

#define BATCHLAS_BATCH_MASK_INSTANTIATE(T)                                                                     \
    template Event move_items<T>(Queue&, T*, std::int64_t, std::int64_t, T*, std::int64_t, std::int64_t, int, int, \
                                 Span<const std::int32_t>, bool);

BATCHLAS_BATCH_MASK_INSTANTIATE(float)
BATCHLAS_BATCH_MASK_INSTANTIATE(double)
BATCHLAS_BATCH_MASK_INSTANTIATE(std::complex<float>)
BATCHLAS_BATCH_MASK_INSTANTIATE(std::complex<double>)
BATCHLAS_BATCH_MASK_INSTANTIATE(std::int32_t)

#undef BATCHLAS_BATCH_MASK_INSTANTIATE

}  // namespace batchlas::batch_mask
