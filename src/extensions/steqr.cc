#include <algorithm>
#include <blas/extensions.hh>
#include "steqr_internal.hh"
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

template <typename T>
inline bool should_use_cta(const Queue& ctx, int64_t n) {
    return n > 0 && n <= static_cast<int64_t>(device_max_sub_group_size(ctx));
}

} // namespace

template <Backend B, typename T>
Event steqr(Queue& ctx,
            const VectorView<T>& d_in,
            const VectorView<T>& e_in,
            const VectorView<T>& eigenvalues,
            const Span<std::byte>& ws,
            JobType jobz,
            SteqrParams<T> params,
            const MatrixView<T, MatrixFormat::Dense>& eigvects) {
    const int64_t n = d_in.size();
    if (should_use_cta<T>(ctx, n)) {
        return steqr_cta<B, T>(ctx, d_in, e_in, eigenvalues, ws, jobz, params, eigvects);
    }
    return steqr_wg<B, T>(ctx, d_in, e_in, eigenvalues, ws, jobz, params, eigvects);
}

template <typename T>
size_t steqr_buffer_size(Queue& ctx,
                         const VectorView<T>& d,
                         const VectorView<T>& e,
                         const VectorView<T>& eigenvalues,
                         JobType jobz,
                         SteqrParams<T> params) {
    const int64_t n = d.size();
    if (should_use_cta<T>(ctx, n)) {
        return steqr_cta_buffer_size<T>(ctx, d, e, eigenvalues, jobz, params);
    }
    return steqr_wg_buffer_size<T>(ctx, d, e, eigenvalues, jobz, params);
}

#define STEQR_INSTANTIATE(back, fp) \
template Event steqr<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<std::byte>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&);

#define STEQR_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(STEQR_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
STEQR_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
STEQR_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#define STEQR_BUFFER_SIZE_INSTANTIATE(fp) \
template size_t steqr_buffer_size<BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>);

BATCHLAS_FOR_EACH_REAL_TYPE(STEQR_BUFFER_SIZE_INSTANTIATE)

#undef STEQR_BUFFER_SIZE_INSTANTIATE
#undef STEQR_INSTANTIATE_FOR_BACKEND
#undef STEQR_INSTANTIATE

} // namespace batchlas
