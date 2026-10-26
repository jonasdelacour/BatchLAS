// cuSOLVER sytrf, the vendor arm of the sytrf op. cuSOLVER 13.2 offers the legacy cusolverDn<t>sytrf
// (int32 ipiv, LAPACK's convention, one matrix per call) and the 64-bit cusolverDnXsytrs (int64 ipiv);
// there is no Xsytrf and no batched sytrf, so this is a per-item loop on one stream with one shared
// work buffer. evidence: docs/perf/sytrf.md#sytrf-vendor

#include <src/linalg-impl.hh>
#include <src/queue.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/blas/functions/sytrf.hh>
#include <src/util/template-instantiations.hh>

#include <complex>

namespace batchlas::backend {

namespace {

template <Backend B, typename T>
int sytrf_lwork(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    static LinalgHandle<B> handle;
    handle.setStream(ctx);
    int lwork = 0;
    call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSsytrf_bufferSize, cusolverDnDsytrf_bufferSize,
                                                 cusolverDnCsytrf_bufferSize, cusolverDnZsytrf_bufferSize, handle,
                                                 A.rows(), A.data_ptr(), A.ld(), &lwork);
    return lwork < 1 ? 1 : lwork;
}

template <typename T>
struct SytrfVendorLayout {
    Span<T> work;
    Span<int32_t> info;
};

template <Backend B, typename T>
SytrfVendorLayout<T> sytrf_vendor_layout(Queue& ctx, BumpAllocator& pool, const MatrixView<T, MatrixFormat::Dense>& A,
                                         Span<int32_t> info_out) {
    SytrfVendorLayout<T> l;
    l.work = pool.allocate<T>(ctx, static_cast<size_t>(sytrf_lwork<B, T>(ctx, A)));
    const auto batch = static_cast<size_t>(A.batch_size());
    l.info = info_out.size() >= batch ? info_out : pool.allocate<int32_t>(ctx, batch);
    return l;
}

}  // namespace

template <Backend B, typename T>
size_t sytrf_vendor_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo /*uplo*/) {
    return workspace_bytes([&](BumpAllocator& p) { return sytrf_vendor_layout<B, T>(ctx, p, A, Span<int32_t>{}); });
}

template <Backend B, typename T>
Event sytrf_vendor(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, Span<int64_t> pivots,
                   Span<std::byte> workspace, Span<int32_t> info_out) {
    static LinalgHandle<B> handle;
    BumpAllocator pool(workspace);
    const auto l = sytrf_vendor_layout<B, T>(ctx, pool, A, info_out);
    const int lwork = static_cast<int>(l.work.size());
    int* const piv = pivots.as_span<int>().data();
    const auto view = A;
    impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
        handle.setStream(stream);
        const int n = view.rows();
        for (int b = 0; b < view.batch_size(); ++b) {
            call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSsytrf, cusolverDnDsytrf, cusolverDnCsytrf,
                                                         cusolverDnZsytrf, handle, uplo, n,
                                                         view.data_ptr() + static_cast<size_t>(b) * view.stride(),
                                                         view.ld(), piv + static_cast<size_t>(b) * n, l.work.data(),
                                                         lwork, l.info.data() + b);
        }
    });
    return ctx.create_event_after_external_work();
}

// ONLY `backend::*_vendor` rows. evidence: docs/design/runtime-internals.md#runtime-internals-vendor-tus-instantiate-only-vendor-symbols
#define CUSOLVER_SYTRF_OPS(B, fp)                                  \
    BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, sytrf_vendor)           \
    BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, sytrf_vendor_buffer_size)
BATCHLAS_FOR_EACH_SCALAR_TYPE_1(CUSOLVER_SYTRF_OPS, Backend::CUDA)
#undef CUSOLVER_SYTRF_OPS

}  // namespace batchlas::backend
