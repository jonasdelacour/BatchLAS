#include <algorithm>
#include <blas/extensions.hh>
#include "steqr_internal.hh"
#include "../queue.hh"

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

#if BATCHLAS_HAS_CUDA_BACKEND
template Event steqr<Backend::CUDA, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, const Span<std::byte>&, JobType, SteqrParams<float>, const MatrixView<float, MatrixFormat::Dense>&);
template Event steqr<Backend::CUDA, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, const Span<std::byte>&, JobType, SteqrParams<double>, const MatrixView<double, MatrixFormat::Dense>&);
#endif

#if BATCHLAS_HAS_HOST_BACKEND
template Event steqr<Backend::NETLIB, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, const Span<std::byte>&, JobType, SteqrParams<float>, const MatrixView<float, MatrixFormat::Dense>&);
template Event steqr<Backend::NETLIB, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, const Span<std::byte>&, JobType, SteqrParams<double>, const MatrixView<double, MatrixFormat::Dense>&);
#endif

template size_t steqr_buffer_size<float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, JobType, SteqrParams<float>);
template size_t steqr_buffer_size<double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, JobType, SteqrParams<double>);

} // namespace batchlas
