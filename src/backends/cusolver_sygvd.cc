// cuSOLVER sygvd, the vendor arm of the sygvd op. cuSOLVER 13.2 has the legacy cusolverDn<t>sygvd (and
// the 64-bit Xsygvd) for one matrix per call and no batched generalized eigensolver, so this is a
// per-item loop on one stream with one shared work buffer. evidence: docs/perf/sygvd.md#sygvd-routes

#include <src/linalg-impl.hh>
#include <src/queue.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/blas/functions/sygvd.hh>
#include <src/util/template-instantiations.hh>

#include <complex>

namespace batchlas::backend {

namespace {

template <Backend B, typename T>
int sygvd_lwork(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& Bm,
                Span<typename base_type<T>::type> w, int itype, JobType jobz, Uplo uplo) {
    static LinalgHandle<B> handle;
    handle.setStream(ctx);
    int lwork = 0;
    call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSsygvd_bufferSize, cusolverDnDsygvd_bufferSize,
                                                 cusolverDnChegvd_bufferSize, cusolverDnZhegvd_bufferSize, handle,
                                                 static_cast<cusolverEigType_t>(itype), jobz, uplo, A.rows(),
                                                 A.data_ptr(), A.ld(), Bm.data_ptr(), Bm.ld(), w.data(), &lwork);
    return lwork < 1 ? 1 : lwork;
}

template <typename T>
struct SygvdVendorLayout {
    Span<T> work;
    Span<int32_t> info;
};

template <Backend B, typename T>
SygvdVendorLayout<T> sygvd_vendor_layout(Queue& ctx, BumpAllocator& pool, const MatrixView<T, MatrixFormat::Dense>& A,
                                         const MatrixView<T, MatrixFormat::Dense>& Bm,
                                         Span<typename base_type<T>::type> w, int itype, JobType jobz, Uplo uplo,
                                         Span<int32_t> info_out) {
    SygvdVendorLayout<T> l;
    l.work = pool.allocate<T>(ctx, static_cast<size_t>(sygvd_lwork<B, T>(ctx, A, Bm, w, itype, jobz, uplo)));
    // Unconditional when no caller span is given, so the size never depends on whether info is asked for.
    const auto batch = static_cast<size_t>(A.batch_size());
    l.info = info_out.size() >= batch ? info_out : pool.allocate<int32_t>(ctx, batch);
    return l;
}

}  // namespace

template <Backend B, typename T>
size_t sygvd_vendor_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& Bm, Span<typename base_type<T>::type> w,
                                int itype, JobType jobz, Uplo uplo) {
    return workspace_bytes([&](BumpAllocator& p) {
        return sygvd_vendor_layout<B, T>(ctx, p, A, Bm, w, itype, jobz, uplo, Span<int32_t>{});
    });
}

template <Backend B, typename T>
Event sygvd_vendor(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& Bm,
                   Span<typename base_type<T>::type> w, int itype, JobType jobz, Uplo uplo, Span<std::byte> workspace,
                   Span<int32_t> info_out) {
    static LinalgHandle<B> handle;
    BumpAllocator pool(workspace);
    const auto l = sygvd_vendor_layout<B, T>(ctx, pool, A, Bm, w, itype, jobz, uplo, info_out);
    const int lwork = static_cast<int>(l.work.size());
    const auto a = A;
    const auto b = Bm;
    const auto et = static_cast<cusolverEigType_t>(itype);
    impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
        handle.setStream(stream);
        const int n = a.rows();
        for (int i = 0; i < a.batch_size(); ++i) {
            call_backend<T, BackendLibrary::CUSOLVER, B>(
                cusolverDnSsygvd, cusolverDnDsygvd, cusolverDnChegvd, cusolverDnZhegvd, handle, et, jobz, uplo, n,
                a.data_ptr() + static_cast<size_t>(i) * a.stride(), a.ld(),
                b.data_ptr() + static_cast<size_t>(i) * b.stride(), b.ld(), w.data() + static_cast<size_t>(i) * n,
                l.work.data(), lwork, l.info.data() + i);
        }
    });
    return ctx.create_event_after_external_work();
}

// ONLY `backend::*_vendor` rows. evidence: docs/design/runtime-internals.md#runtime-internals-vendor-tus-instantiate-only-vendor-symbols
#define CUSOLVER_SYGVD_OPS(B, fp)                                  \
    BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, sygvd_vendor)           \
    BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, sygvd_vendor_buffer_size)
BATCHLAS_FOR_EACH_SCALAR_TYPE_1(CUSOLVER_SYGVD_OPS, Backend::CUDA)
#undef CUSOLVER_SYGVD_OPS

}  // namespace batchlas::backend
