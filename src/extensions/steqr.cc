#include <algorithm>
#include <batchlas/blas/extensions.hh>
#include "steqr_internal.hh"
#include "info_span.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

namespace batchlas {

namespace {

inline int32_t device_max_sub_group_size(const Queue& ctx) {
    const auto dev = ctx->get_device();
    const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    int32_t max_sg = 1;
    for (const auto s : sg_sizes) {
        max_sg = std::max(max_sg, static_cast<int32_t>(s));
    }
    return max_sg;
}

template <Backend B, typename T>
inline bool should_use_cta(const Queue& ctx, int64_t n) {
    // steqr_cta uses chunked sub-group operations that produce incorrect
    // eigenvalues on AMD gfx1200. Disable the CTA path only for ROCm.
    if constexpr (B == Backend::ROCM) {
        (void)ctx; (void)n;
        return false;
    }
    return n > 0 && n <= static_cast<int64_t>(device_max_sub_group_size(ctx));
}

} // namespace

template <Backend B, typename T>
Event steqr_dispatch(Queue& ctx,
                     const VectorView<T>& d_in,
                     const VectorView<T>& e_in,
                     const VectorView<T>& eigenvalues,
                     const Span<std::byte>& ws,
                     JobType jobz,
                     SteqrParams<T> params,
                     const MatrixView<T, MatrixFormat::Dense>& eigvects,
                     Span<int32_t> info) {
    const int64_t n = d_in.size();
    if (should_use_cta<B, T>(ctx, n)) {
        return steqr_cta<B, T>(ctx, d_in, e_in, eigenvalues, ws, jobz, params, eigvects, info);
    }
    return steqr_wg<B, T>(ctx, d_in, e_in, eigenvalues, ws, jobz, params, eigvects, info);
}

template <Backend B, typename T>
Event steqr(Queue& ctx,
            const VectorView<T>& d_in,
            const VectorView<T>& e_in,
            const VectorView<T>& eigenvalues,
            const Span<std::byte>& ws,
            JobType jobz,
            SteqrParams<T> params,
            const MatrixView<T, MatrixFormat::Dense>& eigvects,
            Span<int32_t> info) {
    // The one clear, here rather than in the tiers: neither steqr_cta nor steqr_wg
    // touches an item that converged, so without this the caller would read
    // whatever was in its span for the healthy items. Everything below only ever
    // raises a status -- see the accumulator rule in src/extensions/info_span.hh.
    detail::info_clear(ctx, info, d_in.batch_size());
    return steqr_dispatch<B, T>(ctx, d_in, e_in, eigenvalues, ws, jobz, params, eigvects, info);
}

template <typename T>
size_t steqr_buffer_size(Queue& ctx,
                         const VectorView<T>& d,
                         const VectorView<T>& e,
                         const VectorView<T>& eigenvalues,
                         JobType jobz,
                         SteqrParams<T> params) {
    const int64_t n = d.size();
    // Return the maximum of both paths so the buffer is sufficient regardless
    // of which backend's dispatch is used at runtime.
    const size_t cta_sz = n > 0 ? steqr_cta_buffer_size<T>(ctx, d, e, eigenvalues, jobz, params) : 0;
    const size_t wg_sz  = steqr_wg_buffer_size<T>(ctx, d, e, eigenvalues, jobz, params);
    return std::max(cta_sz, wg_sz);
}

#define STEQR_INSTANTIATE(back, fp) \
template Event steqr_dispatch<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<std::byte>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, Span<int32_t>); \
template Event steqr<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<std::byte>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, Span<int32_t>);

BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(STEQR_INSTANTIATE)

#define STEQR_BUFFER_SIZE_INSTANTIATE(fp) \
template size_t steqr_buffer_size<BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>);

BATCHLAS_FOR_EACH_REAL_TYPE(STEQR_BUFFER_SIZE_INSTANTIATE)

#undef STEQR_BUFFER_SIZE_INSTANTIATE
#undef STEQR_INSTANTIATE

} // namespace batchlas
