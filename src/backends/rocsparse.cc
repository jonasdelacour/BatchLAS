#include "../linalg-impl.hh"
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <batchlas/blas/linalg.hh>
#include "../util/template-instantiations.hh"
#include <complex>
#include "backend_handle_impl.hh"

namespace batchlas {

    namespace backend {

    template <Backend Back, typename T, MatrixFormat MFormat>
    Event spmm_vendor(Queue& ctx,
               const MatrixView<T, MFormat>& A,
               const MatrixView<T, MatrixFormat::Dense>& B,
               const MatrixView<T, MatrixFormat::Dense>& C,
               T alpha,
               T beta,
               Transpose transA,
               Transpose transB,
               Span<std::byte> workspace) {
        static LinalgHandle<Back> handle;
        handle.setStream(ctx);
        BumpAllocator pool(workspace);
        auto buffer_size = spmm_buffer_size<Back>(ctx, A, B, C, alpha, beta, transA, transB);
        auto buffer = pool.allocate<std::byte>(ctx, buffer_size);
        rocsparse_spmm(handle,
                       enum_convert<BackendLibrary::ROCSPARSE>(transA),
                       enum_convert<BackendLibrary::ROCSPARSE>(transB),
                       &alpha,
                       *A,
                       *B,
                       &beta,
                       *C,
                       BackendScalar<T,BackendLibrary::ROCSPARSE>::type,
                       rocsparse_spmm_alg_default,
                       rocsparse_spmm_stage_compute,
                       &buffer_size,
                       buffer.data());
        return ctx.create_event_after_external_work();
    }

    } // namespace backend

    namespace backend {

    template <Backend Back, typename T, MatrixFormat MFormat>
    size_t spmm_vendor_buffer_size(Queue& ctx,
                          const MatrixView<T, MFormat>& A,
                          const MatrixView<T, MatrixFormat::Dense>& B,
                          const MatrixView<T, MatrixFormat::Dense>& C,
                          T alpha,
                          T beta,
                          Transpose transA,
                          Transpose transB) {
        static LinalgHandle<Back> handle;
        handle.setStream(ctx);
        size_t size = 0;
        rocsparse_spmm(handle,
                       enum_convert<BackendLibrary::ROCSPARSE>(transA),
                       enum_convert<BackendLibrary::ROCSPARSE>(transB),
                                 &alpha,
                                 *A,
                                 *B,
                                 &beta,
                                 *C,
                       BackendScalar<T,BackendLibrary::ROCSPARSE>::type,
                       rocsparse_spmm_alg_default,
                       rocsparse_spmm_stage_buffer_size,
                       &size,
                       nullptr);
        return size;
    }

    } // namespace backend

    // ONLY `backend::spmm_vendor` (CSR only), via raw BATCHLAS_INSTANTIATE: there is no _BACKEND_FORMAT_OP.
    // evidence: docs/design/runtime-internals.md#runtime-internals-vendor-tus-instantiate-only-vendor-symbols
    #define ROCSPARSE_OPS(B, fp) \
        BATCHLAS_INSTANTIATE(sig::spmm_vendor<BATCHLAS_UNPAREN fp BATCHLAS_COMMA MatrixFormat::CSR>, \
                             backend::spmm_vendor, B, BATCHLAS_UNPAREN fp, MatrixFormat::CSR) \
        BATCHLAS_INSTANTIATE(sig::spmm_vendor_buffer_size<BATCHLAS_UNPAREN fp BATCHLAS_COMMA MatrixFormat::CSR>, \
                             backend::spmm_vendor_buffer_size, B, BATCHLAS_UNPAREN fp, MatrixFormat::CSR)

    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(ROCSPARSE_OPS, Backend::ROCM)

    #undef ROCSPARSE_OPS
}
