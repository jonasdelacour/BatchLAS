//#include <batchlas/blas/linalg.hh>
#include "../linalg-impl.hh"
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include "../queue.hh"
#include <batchlas/util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <batchlas/blas/linalg.hh>

#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/dispatch/op.hh>

// This file contains cuSOLVER primitives implementation
namespace batchlas {

    #if defined(CUDART_VERSION) && CUDART_VERSION >= 12060
        #define USE_CUSOLVER_X_API 1
    #else
        #define USE_CUSOLVER_X_API 0
        #pragma message("cuSOLVER X API is not available, using legacy API be wary batches of matrices larger than 128x128")
    #endif

    template <Backend B, typename T>
    size_t potrf_buffer_size(Queue& ctx,
                            const MatrixView<T,MatrixFormat::Dense>& A,
                            Uplo uplo) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        int size = 0;
        if (A.batch_size() == 1) {
            call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSpotrf_bufferSize, cusolverDnDpotrf_bufferSize, cusolverDnCpotrf_bufferSize, cusolverDnZpotrf_bufferSize,
                handle, uplo, A.rows(), A.data_ptr(), A.ld(), &size);
            size = BumpAllocator::allocation_size<std::byte>(ctx, size) + BumpAllocator::allocation_size<int>(ctx, 1);
        } else {
            size =  BumpAllocator::allocation_size<int>(ctx, A.batch_size());
        }
        return size;
    }

    template <Backend B, typename T>
    Event potrf(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& descrA,
                    Uplo uplo,
                    Span<std::byte> workspace) {        
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        BumpAllocator pool(workspace);
        auto Lwork = potrf_buffer_size<B>(ctx, descrA, uplo) - BumpAllocator::allocation_size<int>(ctx, 1);
        if (descrA.batch_size() == 1) {
            auto potrf_span = pool.allocate<std::byte>(ctx, Lwork);
            auto info = pool.allocate<int>(ctx, 1);
            auto status = call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSpotrf, cusolverDnDpotrf, cusolverDnCpotrf, cusolverDnZpotrf,
                handle, uplo, descrA.rows(), descrA.data_ptr(), descrA.ld(), reinterpret_cast<T*>(potrf_span.data()), Lwork, info.data());
        } else {
            auto info = pool.allocate<int>(ctx, descrA.batch_size());
            call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSpotrfBatched, cusolverDnDpotrfBatched, cusolverDnCpotrfBatched, cusolverDnZpotrfBatched,
                handle, uplo, descrA.rows(), descrA.data_ptrs(ctx).data(), descrA.ld(), info.data(), descrA.batch_size());
        }
        return ctx.create_event_after_external_work();
    }

    namespace backend {
        template <Backend B, typename T>
        Event syev_vendor(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& descrA,
                          Span<typename base_type<T>::type> eigenvalues,
                          JobType jobtype,
                          Uplo uplo,
                          Span<std::byte> workspace) {
            return op_external("cusolver.syev_vendor", [&] {
                static LinalgHandle<B> handle;
                handle.setStream(ctx);
                BumpAllocator pool(workspace);
                size_t l_work_device_bytes = 0;
                size_t l_work_host_bytes = 0;
                int l_work_device_elems = 0; // legacy API returns lwork in elements of T
                cusolverDnParams_t params;
                check_status(cusolverDnCreateParams(&params));
                const auto eig_mode = enum_convert<BackendLibrary::CUSOLVER>(jobtype);
                const auto fill_mode = enum_convert<BackendLibrary::CUSOLVER>(uplo);
                // cuSOLVER's batched SYEV APIs assume the batch is tightly packed with
                // per-matrix stride == lda * n (no extra padding between matrices).
                // MatrixView can represent subviews/slices with arbitrary stride, so we
                // dispatch to a per-batch loop when the batch isn't tightly packed.
                const bool tightly_packed =
                    (descrA.batch_size() > 1) &&
                    (descrA.stride() == descrA.ld() * descrA.cols());

                if (descrA.batch_size() == 1 || !tightly_packed) {
                    // Per-batch loop (also used for batch_size==1).
                    check_status(cusolverDnXsyevd_bufferSize(handle, params, eig_mode, fill_mode, descrA.rows(),
                                                            BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                            descrA.data_ptr(), descrA.ld(),
                                                            BackendScalar<float_t<T>, BackendLibrary::CUSOLVER>::type,
                                                            eigenvalues.data(),
                                                            BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                            &l_work_device_bytes, &l_work_host_bytes));

                    auto host_workspace = pool.allocate<std::byte>(ctx, l_work_host_bytes);
                    auto device_workspace_bytes = pool.allocate<std::byte>(ctx, l_work_device_bytes);

                    auto info = pool.allocate<int>(ctx, descrA.batch_size());
                    for (int i = 0; i < descrA.batch_size(); ++i) {
                        check_status(cusolverDnXsyevd(handle,
                                                     params,
                                                     eig_mode,
                                                     fill_mode,
                                                     descrA.rows(),
                                                     BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                     descrA.data_ptr() + i * descrA.stride(),
                                                     descrA.ld(),
                                                     BackendScalar<float_t<T>, BackendLibrary::CUSOLVER>::type,
                                                     eigenvalues.data() + i * descrA.rows(),
                                                     BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                     device_workspace_bytes.data(),
                                                     l_work_device_bytes,
                                                     host_workspace.data(),
                                                     l_work_host_bytes,
                                                     info.data() + i));
                    }
                } else {
                    // Tightly packed batch: safe to use cuSOLVER batched API.
                    #if USE_CUSOLVER_X_API
                        check_status(cusolverDnXsyevBatched_bufferSize(handle, params, eig_mode, fill_mode, descrA.rows(),
                                                                      BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                                      descrA.data_ptr(), descrA.ld(),
                                                                      BackendScalar<float_t<T>, BackendLibrary::CUSOLVER>::type,
                                                                      eigenvalues.data(),
                                                                      BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                                      &l_work_device_bytes, &l_work_host_bytes, descrA.batch_size()));

                        auto host_workspace = pool.allocate<std::byte>(ctx, l_work_host_bytes);
                        auto device_workspace_bytes = pool.allocate<std::byte>(ctx, l_work_device_bytes);
                        auto info = pool.allocate<int>(ctx, descrA.batch_size());
                        check_status(cusolverDnXsyevBatched(handle,
                                                           params,
                                                           eig_mode,
                                                           fill_mode,
                                                           descrA.rows(),
                                                           BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                           descrA.data_ptr(),
                                                           descrA.ld(),
                                                           BackendScalar<float_t<T>, BackendLibrary::CUSOLVER>::type,
                                                           eigenvalues.data(),
                                                           BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                           device_workspace_bytes.data(),
                                                           l_work_device_bytes,
                                                           host_workspace.data(),
                                                           l_work_host_bytes,
                                                           info.data(),
                                                           descrA.batch_size()));
                    #else
                        syevjInfo_t syevj_info;
                        check_status(cusolverDnCreateSyevjInfo(&syevj_info));
                        call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSsyevjBatched_bufferSize, cusolverDnDsyevjBatched_bufferSize, cusolverDnCheevjBatched_bufferSize, cusolverDnZheevjBatched_bufferSize,
                            handle, eig_mode, fill_mode, descrA.rows(), descrA.data_ptr(), descrA.ld(), base_float_ptr_convert(eigenvalues.data()), &l_work_device_elems, syevj_info, descrA.batch_size());

                        auto device_workspace_elems = pool.allocate<T>(ctx, static_cast<size_t>(l_work_device_elems));
                        auto info = pool.allocate<int>(ctx, descrA.batch_size());
                        call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSsyevjBatched, cusolverDnDsyevjBatched, cusolverDnCheevjBatched, cusolverDnZheevjBatched,
                            handle, eig_mode, fill_mode, descrA.rows(), descrA.data_ptr(), descrA.ld(), base_float_ptr_convert(eigenvalues.data()), device_workspace_elems.data(), l_work_device_elems, info.data(), syevj_info, descrA.batch_size());
                        check_status(cusolverDnDestroySyevjInfo(syevj_info));
                    #endif
                }
                check_status(cusolverDnDestroyParams(params));
                return ctx.create_event_after_external_work();
            });
        }

        template <Backend B, typename T>
        size_t syev_vendor_buffer_size(Queue& ctx,
                                       const MatrixView<T,MatrixFormat::Dense>& descrA,
                                       Span<typename base_type<T>::type> eigenvalues,
                                       JobType jobtype,
                                       Uplo uplo) {
            return op_external("cusolver.syev_vendor_buffer_size", [&] {
                static LinalgHandle<B> handle;
                handle.setStream(ctx);
                size_t l_work_device_bytes = 0;
                size_t l_work_host_bytes = 0;
                int l_work_device_elems = 0;
                cusolverDnParams_t params;
                check_status(cusolverDnCreateParams(&params));
                const auto eig_mode = enum_convert<BackendLibrary::CUSOLVER>(jobtype);
                const auto fill_mode = enum_convert<BackendLibrary::CUSOLVER>(uplo);
                const bool tightly_packed =
                    (descrA.batch_size() > 1) &&
                    (descrA.stride() == descrA.ld() * descrA.cols());

                if (descrA.batch_size() == 1 || !tightly_packed) {
                    // Per-batch loop uses Xsyevd workspace for a single matrix.
                    check_status(cusolverDnXsyevd_bufferSize(handle, params, eig_mode, fill_mode, descrA.rows(),
                                                            BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                            descrA.data_ptr(), descrA.ld(),
                                                            BackendScalar<float_t<T>, BackendLibrary::CUSOLVER>::type,
                                                            eigenvalues.data(),
                                                            BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                            &l_work_device_bytes, &l_work_host_bytes));
                } else {
                    #if USE_CUSOLVER_X_API
                        check_status(cusolverDnXsyevBatched_bufferSize(handle, params, eig_mode, fill_mode, descrA.rows(),
                                                                      BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                                      descrA.data_ptr(), descrA.ld(),
                                                                      BackendScalar<float_t<T>, BackendLibrary::CUSOLVER>::type,
                                                                      eigenvalues.data(),
                                                                      BackendScalar<T, BackendLibrary::CUSOLVER>::type,
                                                                      &l_work_device_bytes, &l_work_host_bytes, descrA.batch_size()));
                    #else
                        syevjInfo_t syevj_info;
                        check_status(cusolverDnCreateSyevjInfo(&syevj_info));
                        call_backend<T, BackendLibrary::CUSOLVER, B>(cusolverDnSsyevjBatched_bufferSize, cusolverDnDsyevjBatched_bufferSize, cusolverDnCheevjBatched_bufferSize, cusolverDnZheevjBatched_bufferSize,
                            handle, eig_mode, fill_mode, descrA.rows(), descrA.data_ptr(), descrA.ld(), base_float_ptr_convert(eigenvalues.data()), &l_work_device_elems, syevj_info, descrA.batch_size());
                        check_status(cusolverDnDestroySyevjInfo(syevj_info));
                    #endif
                }

                check_status(cusolverDnDestroyParams(params));

                return BumpAllocator::allocation_size<std::byte>(ctx, l_work_host_bytes)
                     + BumpAllocator::allocation_size<std::byte>(ctx, l_work_device_bytes)
                     + BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(l_work_device_elems))
                     + BumpAllocator::allocation_size<int>(ctx, descrA.batch_size());
            });
        }

        namespace gesvd_detail {

            template <typename T> struct GesvdVhKernel;

            // Local trait: batchlas::internal::is_complex is not visible in this TU,
            // and pulling in the header that defines it just for a conjugate is not
            // worth the include coupling in a backend wrapper.
            template <typename U> struct is_cplx : std::false_type {};
            template <typename U> struct is_cplx<std::complex<U>> : std::true_type {};

            template <typename U>
            inline U conj_v(const U& x) {
                if constexpr (is_cplx<U>::value) {
                    return U(x.real(), -x.imag());
                } else {
                    return x;
                }
            }

            // cuSOLVER's gesvdj family returns V; the BatchLAS gesvd contract is V^H.
            // Out-of-place: an in-place transpose would race between work-items.
            // src is n x n column-major with ld == n and stride == n*n (cuSOLVER's layout).
            template <typename T>
            void write_vh_from_v(Queue& ctx,
                                 const T* v_src,
                                 int64_t n,
                                 int64_t batch,
                                 const MatrixView<T, MatrixFormat::Dense>& vh_out) {
                ctx->submit([&](sycl::handler& cgh) {
                    auto Vh = vh_out.kernel_view();
                    const int64_t nn = n;
                    const T* src = v_src;
                    cgh.parallel_for<GesvdVhKernel<T>>(
                        sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(n * n)),
                        [=](sycl::id<2> id) {
                            const int64_t b = static_cast<int64_t>(id[0]);
                            const int64_t lin = static_cast<int64_t>(id[1]);
                            const int64_t r = lin / nn;
                            const int64_t c = lin - r * nn;
                            // Vh(r,c) = conj(V(c,r)); V(i,j) == src[b*nn*nn + j*nn + i]
                            Vh(r, c, b) = conj_v(src[b * nn * nn + r * nn + c]);
                        });
                });
            }

            // gesvdjBatched carries no stride arguments: A, U and V are each assumed
            // tightly packed with stride == ld * cols. MatrixView can carry arbitrary
            // strides (subviews), so every view handed to it must be checked.
            template <typename T>
            inline bool packed(const MatrixView<T, MatrixFormat::Dense>& m) {
                return m.batch_size() == 1 || m.stride() == m.ld() * m.cols();
            }

            // Documented limit of cusolverDn<t>gesvdjBatched.
            constexpr int64_t kGesvdjBatchedMaxDim = 32;

            template <typename T>
            inline bool batched_route_ok(const MatrixView<T, MatrixFormat::Dense>& A) {
                return A.rows() <= kGesvdjBatchedMaxDim &&
                       A.cols() <= kGesvdjBatchedMaxDim &&
                       packed(A);
            }

        } // namespace gesvd_detail

        template <Backend B, typename T>
        size_t gesvd_vendor_buffer_size(Queue& ctx,
                                        const MatrixView<T, MatrixFormat::Dense>& A,
                                        Span<typename base_type<T>::type> singular_values,
                                        const MatrixView<T, MatrixFormat::Dense>& U,
                                        const MatrixView<T, MatrixFormat::Dense>& Vh,
                                        SvdVectors jobu,
                                        SvdVectors jobvh) {
            static_cast<void>(U);
            static_cast<void>(Vh);
            return op_external("cusolver.gesvd_vendor_buffer_size", [&] () -> size_t {
                if (!gesvd_detail::batched_route_ok(A)) {
                    // Deliberately not silently falling back to a looped gesvdj: that is a
                    // different algorithm with a different cost, and reporting it under the
                    // same name would corrupt the very comparison this path exists to make.
                    // See GESVD_PLAN.md Tier 0.
                    throw std::runtime_error(
                        "gesvd_vendor (CUSOLVER): only the gesvdjBatched route is implemented "
                        "(requires m <= 32, n <= 32 and a tightly packed batch)");
                }

                static LinalgHandle<B> handle;
                handle.setStream(ctx);

                const int m = static_cast<int>(A.rows());
                const int n = static_cast<int>(A.cols());
                const int batch = static_cast<int>(A.batch_size());
                // cusolverDnXgesvdjBatched has no `econ` flag -- econ belongs
                // to the non-batched cusolverDnXgesvdj, and gesvdaStridedBatched
                // is a different, rank-truncated algorithm. Refuse rather than
                // silently mis-serve: want_u below is `== All`, so a Thin
                // request would quietly mean "no vectors" and the shape checks
                // would pass with U never written. Costs nothing in practice --
                // this route caps at 32x32, where canonicalisation has already
                // rewritten Thin to All for every square case.
                if (jobu == SvdVectors::Thin || jobvh == SvdVectors::Thin) {
                    throw std::runtime_error(
                        "gesvd_vendor (CUSOLVER): thin singular vectors are not supported by the "
                        "gesvdjBatched route");
                }
                const bool want_u = (jobu == SvdVectors::All);
                const bool want_vh = (jobvh == SvdVectors::All);
                const bool vectors = want_u || want_vh;

                gesvdjInfo_t params;
                check_status(cusolverDnCreateGesvdjInfo(&params));
                const cusolverEigMode_t jobz =
                    vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

                int lwork = 0;
                call_backend<T, BackendLibrary::CUSOLVER, B>(
                    cusolverDnSgesvdjBatched_bufferSize, cusolverDnDgesvdjBatched_bufferSize,
                    cusolverDnCgesvdjBatched_bufferSize, cusolverDnZgesvdjBatched_bufferSize,
                    handle, jobz, m, n,
                    A.data_ptr(), A.ld(),
                    base_float_ptr_convert(singular_values.data()),
                    A.data_ptr(), m,
                    A.data_ptr(), n,
                    &lwork, params, batch);
                check_status(cusolverDnDestroyGesvdjInfo(params));

                // One allocation_size per allocation the call side makes. Rounding a
                // summed byte total instead under-provisions: each request is padded up
                // to the pool's alignment independently.
                size_t bytes = BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(lwork));
                bytes += BumpAllocator::allocation_size<int>(ctx, static_cast<size_t>(batch));
                if (vectors) {
                    // V scratch is unconditional when vectors are computed: cuSOLVER needs
                    // somewhere to put V even when only U was asked for, and when Vh IS
                    // wanted we still cannot transpose in place.
                    bytes += BumpAllocator::allocation_size<T>(
                        ctx, static_cast<size_t>(n) * static_cast<size_t>(n) * static_cast<size_t>(batch));
                    if (!want_u) {
                        bytes += BumpAllocator::allocation_size<T>(
                            ctx, static_cast<size_t>(m) * static_cast<size_t>(m) * static_cast<size_t>(batch));
                    }
                }
                return bytes;
            });
        }

        template <Backend B, typename T>
        Event gesvd_vendor(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           Span<typename base_type<T>::type> singular_values,
                           const MatrixView<T, MatrixFormat::Dense>& U,
                           const MatrixView<T, MatrixFormat::Dense>& Vh,
                           SvdVectors jobu,
                           SvdVectors jobvh,
                           Span<std::byte> workspace) {
            return op_external("cusolver.gesvd_vendor", [&] {
                if (!gesvd_detail::batched_route_ok(A)) {
                    throw std::runtime_error(
                        "gesvd_vendor (CUSOLVER): only the gesvdjBatched route is implemented "
                        "(requires m <= 32, n <= 32 and a tightly packed batch)");
                }

                const int m = static_cast<int>(A.rows());
                const int n = static_cast<int>(A.cols());
                const int k = std::min(m, n);
                const int batch = static_cast<int>(A.batch_size());
                // cusolverDnXgesvdjBatched has no `econ` flag -- econ belongs
                // to the non-batched cusolverDnXgesvdj, and gesvdaStridedBatched
                // is a different, rank-truncated algorithm. Refuse rather than
                // silently mis-serve: want_u below is `== All`, so a Thin
                // request would quietly mean "no vectors" and the shape checks
                // would pass with U never written. Costs nothing in practice --
                // this route caps at 32x32, where canonicalisation has already
                // rewritten Thin to All for every square case.
                if (jobu == SvdVectors::Thin || jobvh == SvdVectors::Thin) {
                    throw std::runtime_error(
                        "gesvd_vendor (CUSOLVER): thin singular vectors are not supported by the "
                        "gesvdjBatched route");
                }
                const bool want_u = (jobu == SvdVectors::All);
                const bool want_vh = (jobvh == SvdVectors::All);
                const bool vectors = want_u || want_vh;

                if (singular_values.size() < static_cast<size_t>(k) * static_cast<size_t>(batch)) {
                    throw std::invalid_argument("gesvd_vendor (CUSOLVER): singular_values span too small");
                }
                if (want_u && (U.rows() != m || U.cols() != m || U.batch_size() != batch)) {
                    throw std::invalid_argument("gesvd_vendor (CUSOLVER): U must be (m x m) with matching batch");
                }
                if (want_vh && (Vh.rows() != n || Vh.cols() != n || Vh.batch_size() != batch)) {
                    throw std::invalid_argument("gesvd_vendor (CUSOLVER): Vh must be (n x n) with matching batch");
                }
                if (want_u && !gesvd_detail::packed(U)) {
                    throw std::runtime_error("gesvd_vendor (CUSOLVER): U must be a tightly packed batch");
                }

                static LinalgHandle<B> handle;
                handle.setStream(ctx);
                BumpAllocator pool(workspace);

                gesvdjInfo_t params;
                check_status(cusolverDnCreateGesvdjInfo(&params));
                // Match the BatchLAS contract: singular values descending. cuSOLVER's sort
                // flag does exactly that, so no post-pass is needed.
                check_status(cusolverDnXgesvdjSetSortEig(params, 1));

                const cusolverEigMode_t jobz =
                    vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

                int lwork = 0;
                call_backend<T, BackendLibrary::CUSOLVER, B>(
                    cusolverDnSgesvdjBatched_bufferSize, cusolverDnDgesvdjBatched_bufferSize,
                    cusolverDnCgesvdjBatched_bufferSize, cusolverDnZgesvdjBatched_bufferSize,
                    handle, jobz, m, n,
                    A.data_ptr(), A.ld(),
                    base_float_ptr_convert(singular_values.data()),
                    A.data_ptr(), m,
                    A.data_ptr(), n,
                    &lwork, params, batch);

                // Allocation order must mirror gesvd_vendor_buffer_size exactly.
                auto work = pool.allocate<T>(ctx, static_cast<size_t>(lwork));
                auto info = pool.allocate<int>(ctx, static_cast<size_t>(batch));

                T* v_ptr = nullptr;
                T* u_ptr = nullptr;
                if (vectors) {
                    auto v_scratch = pool.allocate<T>(
                        ctx, static_cast<size_t>(n) * static_cast<size_t>(n) * static_cast<size_t>(batch));
                    v_ptr = v_scratch.data();
                    if (want_u) {
                        u_ptr = U.data_ptr();
                    } else {
                        auto u_scratch = pool.allocate<T>(
                            ctx, static_cast<size_t>(m) * static_cast<size_t>(m) * static_cast<size_t>(batch));
                        u_ptr = u_scratch.data();
                    }
                }

                call_backend<T, BackendLibrary::CUSOLVER, B>(
                    cusolverDnSgesvdjBatched, cusolverDnDgesvdjBatched,
                    cusolverDnCgesvdjBatched, cusolverDnZgesvdjBatched,
                    handle, jobz, m, n,
                    A.data_ptr(), A.ld(),
                    base_float_ptr_convert(singular_values.data()),
                    u_ptr, want_u ? static_cast<int>(U.ld()) : m,
                    v_ptr, n,
                    work.data(), lwork, info.data(), params, batch);

                check_status(cusolverDnDestroyGesvdjInfo(params));

                Event e = ctx.create_event_after_external_work();
                if (want_vh) {
                    // cuSOLVER hands back V; the BatchLAS contract is V^H.
                    gesvd_detail::write_vh_from_v<T>(ctx, v_ptr, n, batch, Vh);
                    e = ctx.get_event();
                }
                return e;
            });
        }

    } // namespace backend

    #define POTRF_INSTANTIATE(fp) \
    template Event potrf<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Uplo, \
        Span<std::byte>);
    
    #define POTRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t potrf_buffer_size<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Uplo);

    #define SYEV_INSTANTIATE(fp) \
    template Event syev<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        JobType, \
        Uplo, \
        Span<std::byte>);

    #define SYEV_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t syev_buffer_size<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        JobType, \
        Uplo);

    #define SYEV_VENDOR_INSTANTIATE(fp) \
    template Event backend::syev_vendor<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        JobType, \
        Uplo, \
        Span<std::byte>);

    #define SYEV_VENDOR_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t backend::syev_vendor_buffer_size<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        JobType, \
        Uplo);

    #define GESVD_VENDOR_INSTANTIATE(fp) \
    template Event backend::gesvd_vendor<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        SvdVectors, \
        SvdVectors, \
        Span<std::byte>);

    #define GESVD_VENDOR_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t backend::gesvd_vendor_buffer_size<Backend::CUDA, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        SvdVectors, \
        SvdVectors);

    #define CUSOLVER_INSTANTIATE(fp) \
        POTRF_INSTANTIATE(fp) \
        POTRF_BUFFER_SIZE_INSTANTIATE(fp) \
        SYEV_INSTANTIATE(fp) \
        SYEV_BUFFER_SIZE_INSTANTIATE(fp) \
        SYEV_VENDOR_INSTANTIATE(fp) \
        SYEV_VENDOR_BUFFER_SIZE_INSTANTIATE(fp) \
        GESVD_VENDOR_INSTANTIATE(fp) \
        GESVD_VENDOR_BUFFER_SIZE_INSTANTIATE(fp)

    // Instantiate for the floating-point types of interest
    CUSOLVER_INSTANTIATE(float)
    CUSOLVER_INSTANTIATE(double)
    CUSOLVER_INSTANTIATE(std::complex<float>)
    CUSOLVER_INSTANTIATE(std::complex<double>)

    #undef POTRF_INSTANTIATE
    #undef POTRF_BUFFER_SIZE_INSTANTIATE
    #undef SYEV_INSTANTIATE
    #undef SYEV_BUFFER_SIZE_INSTANTIATE
    #undef SYEV_VENDOR_INSTANTIATE
    #undef SYEV_VENDOR_BUFFER_SIZE_INSTANTIATE
    #undef GESVD_VENDOR_INSTANTIATE
    #undef GESVD_VENDOR_BUFFER_SIZE_INSTANTIATE
    #undef CUSOLVER_INSTANTIATE
    #undef CUSOLVER_INSTANTIATE_FOR_FP
}