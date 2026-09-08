#include "../linalg-impl.hh"
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <complex>
#include <algorithm>
#include <batchlas/blas/linalg.hh>
#include "../util/template-instantiations.hh"

#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/blas/dispatch/op.hh>

namespace batchlas {

    namespace backend {

    template <Backend B, typename T>
    size_t potrf_vendor_buffer_size(Queue& ctx,
                            const MatrixView<T,MatrixFormat::Dense>& A,
                            Uplo uplo) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        size_t size = 0;
        if (A.batch_size() == 1) {
            size = BumpAllocator::allocation_size<int>(ctx,1);
        } else {
            size = BumpAllocator::allocation_size<int>(ctx,A.batch_size());
        }
        return size;
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    Event potrf_vendor(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Uplo uplo,
                Span<std::byte> workspace,
                Span<int32_t> info_out) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        BumpAllocator pool(workspace);
        // rocSOLVER's info is a per-item device array with LAPACK semantics; it
        // used to be pool scratch nothing read (issue #73). A caller span, when
        // supplied, replaces the scratch -- so the workspace size is unchanged.
        auto info = detail::info_target(ctx, pool, info_out, static_cast<size_t>(A.batch_size()));
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_spotrf, rocsolver_dpotrf, rocsolver_cpotrf, rocsolver_zpotrf,
                handle, enum_convert<BackendLibrary::ROCSOLVER>(uplo), A.rows(), A.data_ptr(), A.ld(), info.data());
        } else {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_spotrf_strided_batched, rocsolver_dpotrf_strided_batched,
                rocsolver_cpotrf_strided_batched, rocsolver_zpotrf_strided_batched,
                handle, enum_convert<BackendLibrary::ROCSOLVER>(uplo), A.rows(), A.data_ptr(), A.ld(), A.stride(), info.data(), A.batch_size());
        }
        return ctx.create_event_after_external_work();
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    Event geqrf_vendor(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<T> tau,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sgeqrf, rocsolver_dgeqrf,
                                                         rocsolver_cgeqrf, rocsolver_zgeqrf,
                                                         handle, A.rows(), A.cols(),
                                                         A.data_ptr(), A.ld(), tau.data());
        } else {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sgeqrf_strided_batched,
                                                         rocsolver_dgeqrf_strided_batched,
                                                         rocsolver_cgeqrf_strided_batched,
                                                         rocsolver_zgeqrf_strided_batched,
                                                         handle, A.rows(), A.cols(),
                                                         A.data_ptr(), A.ld(), A.stride(),
                                                         tau.data(), std::min(A.rows(), A.cols()),
                                                         A.batch_size());
        }
        return ctx.create_event_after_external_work();
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    size_t geqrf_vendor_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(tau);
        return 0;
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    Event ormqr_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      Side side,
                      Transpose trans,
                      Span<T> tau,
                      Span<std::byte> workspace) {
        return op_external("rocsolver.ormqr_vendor", [&] {
            static_cast<void>(workspace);
            static LinalgHandle<B> handle;
            handle.setStream(ctx);
            auto m = C.rows();
            auto n = C.cols();
            auto k = std::min(A.rows(), A.cols());
            if (A.batch_size() == 1) {
                call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sormqr, rocsolver_dormqr,
                                                             rocsolver_cunmqr, rocsolver_zunmqr,
                                                             handle,
                                                             enum_convert<BackendLibrary::ROCSOLVER>(side),
                                                             enum_convert<BackendLibrary::ROCSOLVER>(trans),
                                                             m, n, k,
                                                             A.data_ptr(), A.ld(),
                                                             tau.data(),
                                                             C.data_ptr(), C.ld());
            } else {
                Queue sub_queue(ctx.device(), false);
                for (int i = 0; i < A.batch_size(); ++i) {
                    ormqr_vendor<B, T>(sub_queue,
                                       A.batch_item(i),
                                       C.batch_item(i),
                                       side,
                                       trans,
                                       tau.subspan(i * k, k),
                                       {});
                }
                sub_queue.wait();
            }
            return ctx.create_event_after_external_work();
        });
    }

    template <Backend B, typename T>
    size_t ormqr_vendor_buffer_size(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    Side side,
                                    Transpose trans,
                                    Span<T> tau) {
        return op_external("rocsolver.ormqr_vendor_buffer_size", [&] {
            static_cast<void>(ctx);
            static_cast<void>(A);
            static_cast<void>(C);
            static_cast<void>(side);
            static_cast<void>(trans);
            static_cast<void>(tau);
            return static_cast<size_t>(0);
        });
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    Event orgqr_vendor(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<T> tau,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = A.rows();
        auto n = A.cols();
        auto k = std::min(m, n);
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sorgqr, rocsolver_dorgqr,
                                                         rocsolver_cungqr, rocsolver_zungqr,
                                                         handle, m, n, k, A.data_ptr(), A.ld(),
                                                         tau.data());
        } else {
            Queue sub_queue(ctx.device(), false);
            for (int i = 0; i < A.batch_size(); ++i) {
                orgqr<B>(sub_queue, A.batch_item(i), tau.subspan(i * k, k), {});
            }
            sub_queue.wait();
        }
        return ctx.create_event_after_external_work();
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    size_t orgqr_vendor_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(tau);
        return 0;
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    Event getrf_vendor(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<int64_t> pivots,
                Span<std::byte> workspace,
                Span<int32_t> info_out) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        BumpAllocator pool(workspace);
        // See potrf above (issue #73).
        auto info = detail::info_target(ctx, pool, info_out, static_cast<size_t>(A.batch_size()));
        auto ipiv = pivots.as_span<int>();
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sgetrf, rocsolver_dgetrf,
                                                         rocsolver_cgetrf, rocsolver_zgetrf,
                                                         handle, A.rows(), A.cols(),
                                                         A.data_ptr(), A.ld(), ipiv.data(),
                                                         info.data());
        } else {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sgetrf_strided_batched,
                                                         rocsolver_dgetrf_strided_batched,
                                                         rocsolver_cgetrf_strided_batched,
                                                         rocsolver_zgetrf_strided_batched,
                                                         handle, A.rows(), A.cols(), A.data_ptr(),
                                                         A.ld(), A.stride(), ipiv.data(),
                                                         std::min(A.rows(), A.cols()), info.data(),
                                                         A.batch_size());
        }
        return ctx.create_event_after_external_work();
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    size_t getrf_vendor_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A) {
        return BumpAllocator::allocation_size<int>(ctx, A.batch_size());
    }

    } // namespace backend

    namespace backend {

    template <Backend Back, typename T>
    Event getrs_vendor(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& B,
                Transpose transA,
                Span<int64_t> pivots,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        static LinalgHandle<Back> handle;
        handle.setStream(ctx);
        auto ipiv = pivots.as_span<int>();
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, Back>(rocsolver_sgetrs, rocsolver_dgetrs,
                                                         rocsolver_cgetrs, rocsolver_zgetrs,
                                                         handle,
                                                         enum_convert<BackendLibrary::ROCSOLVER>(transA),
                                                         A.rows(), B.cols(),
                                                         A.data_ptr(), A.ld(), ipiv.data(),
                                                         B.data_ptr(), B.ld());
        } else {
            call_backend<T, BackendLibrary::ROCSOLVER, Back>(rocsolver_sgetrs_strided_batched,
                                                         rocsolver_dgetrs_strided_batched,
                                                         rocsolver_cgetrs_strided_batched,
                                                         rocsolver_zgetrs_strided_batched,
                                                         handle,
                                                         enum_convert<BackendLibrary::ROCSOLVER>(transA),
                                                         A.rows(), B.cols(),
                                                         A.data_ptr(), A.ld(), A.stride(),
                                                         ipiv.data(), std::min(A.rows(), A.cols()),
                                                         B.data_ptr(), B.ld(), B.stride(),
                                                         A.batch_size());
        }
        return ctx.create_event_after_external_work();
    }

    } // namespace backend

    namespace backend {

    template <Backend Back, typename T>
    size_t getrs_vendor_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             Transpose transA) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(B);
        static_cast<void>(transA);
        return 0;
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    Event getri_vendor(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& C,
                Span<int64_t> pivots,
                Span<std::byte> workspace,
                Span<int32_t> info_out) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        BumpAllocator pool(workspace);
        // See potrf above (issue #73).
        auto info = detail::info_target(ctx, pool, info_out, static_cast<size_t>(A.batch_size()));
        auto ipiv = pivots.as_span<int>();
        if (A.data_ptr() != C.data_ptr()) {
            ctx->memcpy(C.data_ptr(), A.data_ptr(), sizeof(T) * static_cast<size_t>(A.stride()) * A.batch_size());
        }
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sgetri, rocsolver_dgetri,
                                                         rocsolver_cgetri, rocsolver_zgetri,
                                                         handle, A.rows(), C.data_ptr(), C.ld(),
                                                         ipiv.data(), info.data());
        } else {
            call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_sgetri_strided_batched,
                                                         rocsolver_dgetri_strided_batched,
                                                         rocsolver_cgetri_strided_batched,
                                                         rocsolver_zgetri_strided_batched,
                                                         handle, A.rows(), C.data_ptr(), C.ld(), C.stride(),
                                                         ipiv.data(), std::min(A.rows(), A.cols()),
                                                         info.data(), A.batch_size());
        }
        return ctx.create_event_after_external_work();
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    size_t getri_vendor_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A) {
        return BumpAllocator::allocation_size<int>(ctx, A.batch_size());
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    Event syev_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      Span<typename base_type<T>::type> eigenvalues,
                      JobType jobtype,
                      Uplo uplo,
                      Span<std::byte> workspace) {
        return op_external("rocsolver.syev_vendor", [&] {
            static LinalgHandle<B> handle;
            handle.setStream(ctx);
            BumpAllocator pool(workspace);
            auto info = pool.allocate<int>(ctx, A.batch_size());
            auto ws = pool.allocate<typename base_type<T>::type>(ctx, A.rows() * A.batch_size());
            if (A.batch_size() == 1) {
                call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_ssyev, rocsolver_dsyev, rocsolver_cheev, rocsolver_zheev,
                    handle, jobtype, uplo,
                    A.rows(), A.data_ptr(), A.ld(), eigenvalues.data(), ws.data(),
                    info.data());
            } else {
                call_backend<T, BackendLibrary::ROCSOLVER, B>(rocsolver_ssyev_strided_batched, rocsolver_dsyevd_strided_batched,
                    rocsolver_cheevd_strided_batched, rocsolver_zheevd_strided_batched,
                    handle, jobtype, uplo,
                    A.rows(), A.data_ptr(), A.ld(), A.stride(), eigenvalues.data(), A.rows(), ws.data(), A.rows(), info.data(), A.batch_size());
            }
            return ctx.create_event_after_external_work();
        });
    }

    template <Backend B, typename T>
    size_t syev_vendor_buffer_size(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                   Span<typename base_type<T>::type> /*eigenvalues*/,
                                   JobType /*jobtype*/,
                                   Uplo /*uplo*/) {
        return op_external("rocsolver.syev_vendor_buffer_size", [&] {
            return BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, A.rows() * A.batch_size()) +
                   BumpAllocator::allocation_size<int>(ctx, A.batch_size());
        });
    }

    // gesvd has no rocSOLVER binding yet. This stub exists because
    // include/batchlas/blas/functions/gesvd.hh is now declaration-only (it used to define a
    // generic throwing template, which is what blocked a cuSOLVER implementation).
    // Without a definition here a ROCM build fails to link rather than failing at
    // the call, so the throw is preserved deliberately.
    template <Backend B, typename T>
    Event gesvd_vendor(Queue& /*ctx*/,
                       const MatrixView<T, MatrixFormat::Dense>& /*A*/,
                       Span<typename base_type<T>::type> /*singular_values*/,
                       const MatrixView<T, MatrixFormat::Dense>& /*U*/,
                       const MatrixView<T, MatrixFormat::Dense>& /*Vh*/,
                       SvdVectors /*jobu*/,
                       SvdVectors /*jobvh*/,
                       Span<std::byte> /*workspace*/) {
        throw std::runtime_error("gesvd_vendor (ROCSOLVER): not implemented");
    }

    template <Backend B, typename T>
    size_t gesvd_vendor_buffer_size(Queue& /*ctx*/,
                                    const MatrixView<T, MatrixFormat::Dense>& /*A*/,
                                    Span<typename base_type<T>::type> /*singular_values*/,
                                    const MatrixView<T, MatrixFormat::Dense>& /*U*/,
                                    const MatrixView<T, MatrixFormat::Dense>& /*Vh*/,
                                    SvdVectors /*jobu*/,
                                    SvdVectors /*jobvh*/) {
        throw std::runtime_error("gesvd_vendor_buffer_size (ROCSOLVER): not implemented");
    }

    } // namespace backend

    // Explicit instantiations. Signatures live in the `sig` namespace beside each
    // public declaration (include/batchlas/blas/functions/*.hh), so changing one is a single
    // header edit rather than one edit per backend TU.
    //
    // Every row names a `backend::`-qualified `_vendor` symbol, and that is the
    // WP0b invariant rather than an oversight: the public potrf/syev/geqrf/
    // getrf/getrs/getri/ormqr/orgqr definitions moved out of the vendor TUs into
    // src/dispatch/entry_points/{factorization,eigen}.cc, which instantiate them
    // keyed on the device family instead of on any vendor library. A
    // BATCHLAS_INSTANTIATE_OP row for a public op here would collide with those.
    // _BACKEND_OP still looks the alias up as `sig::OP` -- only the FUNCTION is
    // backend-qualified.
    #define ROCSOLVER_OPS(B, fp) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, potrf_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, potrf_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, syev_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, syev_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, gesvd_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, gesvd_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, geqrf_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, geqrf_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getrf_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getrf_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getrs_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getrs_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getri_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getri_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, ormqr_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, ormqr_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, orgqr_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, orgqr_vendor_buffer_size)

    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(ROCSOLVER_OPS, Backend::ROCM)

    #undef ROCSOLVER_OPS
}
