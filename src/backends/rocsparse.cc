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

    template <Backend Back, typename T, MatrixFormat MFormat>
    Event spmm(Queue& ctx,
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

    template <Backend Back, typename T, MatrixFormat MFormat>
    size_t spmm_buffer_size(Queue& ctx,
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

    // Explicit instantiations. Signatures live in the `sig` namespace beside each
    // public declaration (include/batchlas/blas/functions/*.hh), so changing one is a single
    // header edit rather than one edit per backend TU. CSR is the only sparse
    // format rocSPARSE is wired up for here.
    #define ROCSPARSE_OPS(B, fp) \
        BATCHLAS_INSTANTIATE_FORMAT_OP(B, fp, MatrixFormat::CSR, spmm) \
        BATCHLAS_INSTANTIATE_FORMAT_OP(B, fp, MatrixFormat::CSR, spmm_buffer_size)

    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(ROCSPARSE_OPS, Backend::ROCM)

    #undef ROCSPARSE_OPS
}
