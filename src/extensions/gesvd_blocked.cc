#include <blas/extensions.hh>
#include <batchlas/backend_config.h>
#include <util/mempool.hh>

#include "../math-helpers.hh"
#include "../util/template-instantiations.hh"

#include <algorithm>
#include <stdexcept>

namespace batchlas {

namespace {

template <typename T>
void validate_gesvd_blocked_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                 Span<typename base_type<T>::type> singular_values,
                                 const MatrixView<T, MatrixFormat::Dense>& u,
                                 const MatrixView<T, MatrixFormat::Dense>& vh,
                                 SvdVectors jobu,
                                 SvdVectors jobvh) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("gesvd_blocked: initial implementation supports square matrices only");
    }
    if (a.batch_size() < 1 || a.rows() < 1) {
        throw std::invalid_argument("gesvd_blocked: invalid matrix dimensions or batch size");
    }
    const int64_t n = a.rows();
    const int64_t batch = a.batch_size();
    const std::size_t need_s = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
    if (singular_values.size() < need_s) {
        throw std::invalid_argument("gesvd_blocked: singular_values span too small");
    }

    if (jobu == SvdVectors::All && (u.rows() != n || u.cols() != n || u.batch_size() != batch)) {
        throw std::invalid_argument("gesvd_blocked: U must be (n x n) with matching batch size");
    }
    if (jobvh == SvdVectors::All && (vh.rows() != n || vh.cols() != n || vh.batch_size() != batch)) {
        throw std::invalid_argument("gesvd_blocked: Vh must be (n x n) with matching batch size");
    }
}

} // namespace

template <Backend B, typename T>
Event gesvd_blocked(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a_in,
                    Span<typename base_type<T>::type> singular_values,
                    const MatrixView<T, MatrixFormat::Dense>& u_out,
                    const MatrixView<T, MatrixFormat::Dense>& vh_out,
                    SvdVectors jobu,
                    SvdVectors jobvh,
                    const Span<std::byte>& ws) {
    static_cast<void>(u_out);
    static_cast<void>(vh_out);
    validate_gesvd_blocked_dims(a_in, singular_values, u_out, vh_out, jobu, jobvh);

    if (!ctx.in_order()) {
        throw std::runtime_error("gesvd_blocked: requires an in-order Queue");
    }

    const int32_t n = static_cast<int32_t>(a_in.rows());
    const int32_t batch = static_cast<int32_t>(a_in.batch_size());

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
    BumpAllocator pool(ws_mut);

    auto d_span = pool.allocate<typename base_type<T>::type>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch));
    auto e_span = pool.allocate<typename base_type<T>::type>(ctx, static_cast<size_t>(n > 0 ? n - 1 : 0) * static_cast<size_t>(batch));
    auto tauq_span = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch));
    auto taup_span = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch));

    VectorView<typename base_type<T>::type> d_view(d_span, n, batch, 1, n);
    VectorView<typename base_type<T>::type> e_view(e_span, std::max(0, n - 1), batch, 1, std::max(1, n - 1));
    VectorView<T> tauq_view(tauq_span, n, batch, 1, n);
    VectorView<T> taup_view(taup_span, n, batch, 1, n);
    auto bdsqr_ws = pool.allocate<std::byte>(ctx, bdsqr_buffer_size<typename base_type<T>::type>(ctx,
                                                                                                  d_view,
                                                                                                  e_view,
                                                                                                  singular_values));
    if constexpr (internal::is_complex<T>::value) {
        throw std::runtime_error("gesvd_blocked: complex types are not implemented yet in the native path");
    } else {
        gebrd_unblocked<B, T>(ctx, a, d_view, e_view, tauq_view, taup_view);
        bdsqr<B, typename base_type<T>::type>(ctx,
                                              d_view,
                                              e_view,
                                              singular_values,
                                              bdsqr_ws,
                                              /*sort_desc=*/true);
    }

    if (jobu == SvdVectors::All) {
        u_out.fill_identity(ctx);

        const std::size_t ormbr_bytes = ormbr_buffer_size<B, T>(ctx,
                                                                 a,
                                                                 tauq_view,
                                                                 u_out,
                                                                 'Q',
                                                                 Side::Left,
                                                                 Transpose::NoTrans,
                                                                 /*block_size=*/32);
        auto ormbr_ws = pool.allocate<std::byte>(ctx, ormbr_bytes);
        ormbr<B, T>(ctx,
                    a,
                    tauq_view,
                    u_out,
                    'Q',
                    Side::Left,
                    Transpose::NoTrans,
                    ormbr_ws,
                    /*block_size=*/32);
    }

    if (jobvh == SvdVectors::All) {
        vh_out.fill_identity(ctx);

        const std::size_t ormbr_bytes = ormbr_buffer_size<B, T>(ctx,
                                                                 a,
                                                                 taup_view,
                                                                 vh_out,
                                                                 'P',
                                                                 Side::Right,
                                                                 Transpose::ConjTrans,
                                                                 /*block_size=*/32);
        auto ormbr_ws = pool.allocate<std::byte>(ctx, ormbr_bytes);
        ormbr<B, T>(ctx,
                    a,
                    taup_view,
                    vh_out,
                    'P',
                    Side::Right,
                    Transpose::ConjTrans,
                    ormbr_ws,
                    /*block_size=*/32);
    }

    return ctx.get_event();
}

template <Backend B, typename T>
size_t gesvd_blocked_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 Span<typename base_type<T>::type> singular_values,
                                 const MatrixView<T, MatrixFormat::Dense>& u_out,
                                 const MatrixView<T, MatrixFormat::Dense>& vh_out,
                                 SvdVectors jobu,
                                 SvdVectors jobvh) {
    validate_gesvd_blocked_dims(a, singular_values, u_out, vh_out, jobu, jobvh);

    const size_t n = static_cast<size_t>(a.rows());
    const size_t batch = static_cast<size_t>(a.batch_size());

    size_t bytes = 0;
    bytes += BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, n * batch);                 // d
    bytes += BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, (n > 0 ? n - 1 : 0) * batch); // e
    bytes += BumpAllocator::allocation_size<T>(ctx, n * batch);                                            // tauq
    bytes += BumpAllocator::allocation_size<T>(ctx, n * batch);                                            // taup

    VectorView<typename base_type<T>::type> d_dummy(nullptr,
                                                    static_cast<int32_t>(n),
                                                    static_cast<int32_t>(batch),
                                                    1,
                                                    static_cast<int32_t>(n));
    VectorView<typename base_type<T>::type> e_dummy(nullptr,
                                                    static_cast<int32_t>(n > 0 ? n - 1 : 0),
                                                    static_cast<int32_t>(batch),
                                                    1,
                                                    static_cast<int32_t>(n > 1 ? n - 1 : 1));
    VectorView<T> tauq_dummy(nullptr,
                             static_cast<int32_t>(n),
                             static_cast<int32_t>(batch),
                             1,
                             static_cast<int32_t>(n));
    VectorView<T> taup_dummy(nullptr,
                             static_cast<int32_t>(n),
                             static_cast<int32_t>(batch),
                             1,
                             static_cast<int32_t>(n));
    bytes += BumpAllocator::allocation_size<std::byte>(ctx,
        bdsqr_buffer_size<typename base_type<T>::type>(ctx, d_dummy, e_dummy, singular_values));

    if (jobu == SvdVectors::All) {
        bytes += BumpAllocator::allocation_size<std::byte>(ctx,
            ormbr_buffer_size<B, T>(ctx,
                                    a,
                                    tauq_dummy,
                                    u_out,
                                    'Q',
                                    Side::Left,
                                    Transpose::NoTrans,
                                    /*block_size=*/32));
    }

    if (jobvh == SvdVectors::All) {
        bytes += BumpAllocator::allocation_size<std::byte>(ctx,
            ormbr_buffer_size<B, T>(ctx,
                                    a,
                                    taup_dummy,
                                    vh_out,
                                    'P',
                                    Side::Right,
                                    Transpose::ConjTrans,
                                    /*block_size=*/32));
    }

    return bytes;
}

#define GESVD_BLOCKED_INSTANTIATE(back, fp) \
    template Event gesvd_blocked<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, \
        SvdVectors, \
        const Span<std::byte>&); \
    template size_t gesvd_blocked_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        SvdVectors, \
        SvdVectors);

#define GESVD_BLOCKED_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(GESVD_BLOCKED_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
GESVD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
GESVD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
GESVD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef GESVD_BLOCKED_INSTANTIATE_FOR_BACKEND
#undef GESVD_BLOCKED_INSTANTIATE

} // namespace batchlas
