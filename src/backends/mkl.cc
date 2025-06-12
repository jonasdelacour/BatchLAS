#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include "../queue.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/lapack.hpp>

namespace batchlas {

    template <Backend Back, typename T>
    Event gemm(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const MatrixView<T, MatrixFormat::Dense>& B,
               const MatrixView<T, MatrixFormat::Dense>& C,
               T alpha,
               T beta,
               Transpose transA,
               Transpose transB,
               ComputePrecision precision) {
        static_cast<void>(precision);
        auto [m, k] = get_effective_dims(A, transA);
        auto [kB, n] = get_effective_dims(B, transB);
        if (A.batch_size() == 1) {
            oneapi::mkl::blas::column_major::gemm(
                *ctx,
                enum_convert<BackendLibrary::MKL>(transA),
                enum_convert<BackendLibrary::MKL>(transB),
                m, n, k,
                alpha,
                A.data_ptr(), A.ld(),
                B.data_ptr(), B.ld(),
                beta,
                C.data_ptr(), C.ld());
        } else {
            oneapi::mkl::blas::column_major::gemm_batch(
                *ctx,
                enum_convert<BackendLibrary::MKL>(transA),
                enum_convert<BackendLibrary::MKL>(transB),
                m, n, k,
                alpha,
                A.data_ptr(), A.ld(), A.stride(),
                B.data_ptr(), B.ld(), B.stride(),
                beta,
                C.data_ptr(), C.ld(), C.stride(),
                A.batch_size());
        }
        return ctx.get_event();
    }

    template <Backend Back, typename T>
    Event geqrf(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<T> tau,
                Span<std::byte> workspace) {
        auto m = A.rows();
        auto n = A.cols();
        auto stride_a = A.stride();
        auto stride_tau = std::min(m, n);
        size_t scratch = geqrf_buffer_size<Back>(ctx, A, tau);
        if (workspace.size() < scratch) {
            throw std::runtime_error("Insufficient workspace for MKL geqrf");
        }
        auto* scratch_ptr = reinterpret_cast<T*>(workspace.data());
        oneapi::mkl::lapack::geqrf_batch(
            *ctx, m, n,
            A.data_ptr(), A.ld(), stride_a,
            tau.data(), stride_tau,
            A.batch_size(),
            scratch_ptr, scratch / sizeof(T));
        return ctx.get_event();
    }

    template <Backend Back, typename T>
    size_t geqrf_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau) {
        auto m = A.rows();
        auto n = A.cols();
        auto stride_a = A.stride();
        auto stride_tau = std::min(m, n);
        return oneapi::mkl::lapack::geqrf_batch_scratchpad_size<T>(
            *ctx, m, n, A.ld(), stride_a, stride_tau, A.batch_size()) * sizeof(T);
    }

#define GEMM_INSTANTIATE(fp) \
    template Event gemm<Backend::MKL, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, const MatrixView<fp, MatrixFormat::Dense>&, const MatrixView<fp, MatrixFormat::Dense>&, fp, fp, Transpose, Transpose, ComputePrecision);

#define GEQRF_INSTANTIATE(fp) \
    template Event geqrf<Backend::MKL, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Span<fp>, Span<std::byte>);

#define GEQRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t geqrf_buffer_size<Backend::MKL, fp>(Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Span<fp>);

GEMM_INSTANTIATE(float)
GEMM_INSTANTIATE(double)
GEMM_INSTANTIATE(std::complex<float>)
GEMM_INSTANTIATE(std::complex<double>)

GEQRF_INSTANTIATE(float)
GEQRF_INSTANTIATE(double)
GEQRF_INSTANTIATE(std::complex<float>)
GEQRF_INSTANTIATE(std::complex<double>)

GEQRF_BUFFER_SIZE_INSTANTIATE(float)
GEQRF_BUFFER_SIZE_INSTANTIATE(double)
GEQRF_BUFFER_SIZE_INSTANTIATE(std::complex<float>)
GEQRF_BUFFER_SIZE_INSTANTIATE(std::complex<double>)

#undef GEMM_INSTANTIATE
#undef GEQRF_INSTANTIATE
#undef GEQRF_BUFFER_SIZE_INSTANTIATE

} // namespace batchlas
