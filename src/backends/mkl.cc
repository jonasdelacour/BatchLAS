#include <batchlas/blas/linalg.hh>
#include "../util/template-instantiations.hh"
#include "../linalg-impl.hh"
#include "../queue.hh"
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/lapack.hpp>

#include "gemm_variant.hh"
#include "../sycl/gemm_kernels.hh"

namespace batchlas {

    namespace backend {

    template <Backend Back, typename T>
    Event gemm_vendor(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const MatrixView<T, MatrixFormat::Dense>& B,
               const MatrixView<T, MatrixFormat::Dense>& C,
               T alpha,
               T beta,
               Transpose transA,
               Transpose transB,
               ComputePrecision precision) {
        if (!gemm_batch_dimensions_compatible(A, B, C, transA, transB)) {
            throw std::invalid_argument("GEMM: incompatible matrix dimensions");
        }

        if (gemm_has_heterogeneous_batch(A, B, C)) {
            Event last_event;
            bool launched = false;
            for (int batch_index = 0; batch_index < A.batch_size(); ++batch_index) {
                const auto [m, k] = get_effective_dims(A, transA, batch_index);
                const auto [k_b, n] = get_effective_dims(B, transB, batch_index);
                static_cast<void>(k_b);
                if (m == 0 || n == 0) {
                    continue;
                }
                if (k == 0) {
                    last_event = scale(ctx, beta, C.batch_item(batch_index));
                    launched = true;
                    continue;
                }
                last_event = gemm_vendor<Back, T>(ctx,
                                                  A.batch_item(batch_index),
                                                  B.batch_item(batch_index),
                                                  C.batch_item(batch_index),
                                                  alpha,
                                                  beta,
                                                  transA,
                                                  transB,
                                                  precision);
                launched = true;
            }
            if (launched) {
                return std::move(last_event);
            }
            return ctx.get_event();
        }

        if (gemm_use_sycl_custom(ctx, A, B, C, transA, transB, precision)) {
            return sycl_gemm::gemm_custom(ctx, A, B, C, alpha, beta, transA, transB, precision);
        }

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

    } // namespace backend

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
        return backend::gemm_vendor<Back, T>(ctx, A, B, C, alpha, beta, transA, transB, precision);
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
            throw std::invalid_argument("Insufficient workspace for MKL geqrf");
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

// Explicit instantiations. Signatures live in the `sig` namespace beside each
// public declaration (include/batchlas/blas/functions/*.hh), so changing one is a single
// header edit rather than one edit per backend TU.
#define MKL_OPS(B, fp) \
    BATCHLAS_INSTANTIATE_OP(B, fp, gemm) \
    BATCHLAS_INSTANTIATE_OP(B, fp, geqrf) \
    BATCHLAS_INSTANTIATE_OP(B, fp, geqrf_buffer_size)

BATCHLAS_FOR_EACH_SCALAR_TYPE_1(MKL_OPS, Backend::MKL)

#undef MKL_OPS

} // namespace batchlas
