#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <blas/linalg.hh>
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

    #define SPMM_INSTANTIATE(fp, F) \
    template Event spmm<Backend::ROCM, fp, F>(Queue&, const MatrixView<fp, F>&, const MatrixView<fp, MatrixFormat::Dense>&, const MatrixView<fp, MatrixFormat::Dense>&, fp, fp, Transpose, Transpose, Span<std::byte>);
    #define SPMM_BUFFER_SIZE_INSTANTIATE(fp, F) \
    template size_t spmm_buffer_size<Backend::ROCM, fp, F>(Queue&, const MatrixView<fp, F>&, const MatrixView<fp, MatrixFormat::Dense>&, const MatrixView<fp, MatrixFormat::Dense>&, fp, fp, Transpose, Transpose);

    #define ROCSPARSE_INSTANTIATE(fp, F) \
        SPMM_INSTANTIATE(fp, F) \
        SPMM_BUFFER_SIZE_INSTANTIATE(fp, F)

    #define ROCSPARSE_INSTANTIATE_FOR_FP(fp) \
        ROCSPARSE_INSTANTIATE(fp, MatrixFormat::CSR)

    ROCSPARSE_INSTANTIATE_FOR_FP(float)
    ROCSPARSE_INSTANTIATE_FOR_FP(double)
    ROCSPARSE_INSTANTIATE_FOR_FP(std::complex<float>)
    ROCSPARSE_INSTANTIATE_FOR_FP(std::complex<double>)

    #undef SPMM_INSTANTIATE
    #undef SPMM_BUFFER_SIZE_INSTANTIATE
    #undef ROCSPARSE_INSTANTIATE
    #undef ROCSPARSE_INSTANTIATE_FOR_FP
}
