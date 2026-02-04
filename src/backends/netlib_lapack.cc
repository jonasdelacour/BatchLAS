#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <complex>
#include <utility>
#include <algorithm>
#include <lapack.h>
#include <blas/linalg.hh>
#include <util/mempool.hh>

#include <blas/functions/ormqr.hh>
#include <blas/functions/syev.hh>
#include <blas/dispatch/op.hh>

namespace batchlas{

    namespace detail {
    template <typename F>
    Event submit_host_task(Queue& ctx, const char* /*label*/, F&& f) {
        ctx.wait();
        f();
        try {
            sycl::event e = ctx->ext_oneapi_submit_barrier();
            return Event(EventImpl(std::move(e)));
        } catch (const sycl::exception&) {
            EventImpl ev = ctx->submit([&](sycl::handler& h) {
                h.single_task([]() {});
            });
            return Event(std::move(ev));
        }
    }
    } // namespace detail

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
        static_cast<void>(workspace); // no workspace needed for CPU implementation
        auto A_view = A;
        auto B_view = B;
        auto C_view = C;
        return detail::submit_host_task(ctx, "netlib.spmm", [=] {
            if constexpr (MFormat == MatrixFormat::CSR) {
                int batch = A_view.batch_size();
                for (int b = 0; b < batch; ++b) {
                    auto A_b = A_view[b];
                    auto B_b = B_view[b];
                    auto C_b = C_view[b];

                    int m = A_b.rows();
                    int k = A_b.cols();
                    int n = B_b.cols();

                    // Only handle no transpose cases for now
                    if (transA != Transpose::NoTrans || transB != Transpose::NoTrans) {
                        throw std::runtime_error("NETLIB spmm only supports NoTrans for now");
                    }

                    for (int row = 0; row < m; ++row) {
                        for (int col = 0; col < n; ++col) {
                            T sum = beta * C_b.at(row, col);
                            for (int idx = A_b.row_offsets()[row]; idx < A_b.row_offsets()[row + 1]; ++idx) {
                                int a_col = A_b.col_indices()[idx];
                                sum += alpha * A_b.data()[idx] * B_b.at(a_col, col);
                            }
                            C_b.at(row, col) = sum;
                        }
                    }
                }
            } else {
                throw std::runtime_error("Unsupported sparse format for NETLIB spmm");
            }
        });
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
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(B);
        static_cast<void>(C);
        static_cast<void>(alpha);
        static_cast<void>(beta);
        static_cast<void>(transA);
        static_cast<void>(transB);
        return 0;
    }
    
    template <Backend B, typename T>
    Event gemm(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& descrA,
                   const MatrixView<T, MatrixFormat::Dense>& descrB,
                   const MatrixView<T, MatrixFormat::Dense>& descrC,
                   T alpha,
                   T beta,
                   Transpose transA,
                   Transpose transB,
                   ComputePrecision precision) {
        static_cast<void>(precision);
        auto A_view = descrA;
        auto B_view = descrB;
        auto C_view = descrC;
        return detail::submit_host_task(ctx, "netlib.gemm", [=] {
            auto [m, k] = get_effective_dims(A_view, transA);
            auto [kB, n] = get_effective_dims(B_view, transB);
            static_cast<void>(kB);

            if (A_view.batch_size() == 1) {
                call_backend_nh<T, BackendLibrary::CBLAS>(
                    cblas_sgemm, cblas_dgemm, cblas_cgemm, cblas_zgemm,
                    Layout::ColMajor, transA, transB,
                    m, n, k,
                    alpha,
                    A_view.data_ptr(), A_view.ld(),
                    B_view.data_ptr(), B_view.ld(),
                    beta,
                    C_view.data_ptr(), C_view.ld());
            } else {
                for (int i = 0; i < A_view.batch_size(); ++i) {
                    call_backend_nh<T, BackendLibrary::CBLAS>(
                        cblas_sgemm, cblas_dgemm, cblas_cgemm, cblas_zgemm,
                        Layout::ColMajor, transA, transB,
                        m, n, k,
                        alpha,
                        A_view[i].data_ptr(), A_view[i].ld(),
                        B_view[i].data_ptr(), B_view[i].ld(),
                        beta,
                        C_view[i].data_ptr(), C_view[i].ld());
                }
            }
        });
    }

    template <Backend B, typename T>
    Event gemv(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const VectorView<T>& X,
               const VectorView<T>& Y,
               T alpha,
               T beta,
               Transpose transA) {
        auto A_view = A;
        auto X_view = X;
        auto Y_view = Y;
        return detail::submit_host_task(ctx, "netlib.gemv", [=] {
            const int m = A_view.rows();
            const int n = A_view.cols();
            if (A_view.batch_size() > 1) {
                for (int i = 0; i < A_view.batch_size(); ++i) {
                    auto Xi = X_view.batch_item(i);
                    auto Yi = Y_view.batch_item(i);
                    call_backend_nh<T, BackendLibrary::CBLAS>(
                        cblas_sgemv, cblas_dgemv, cblas_cgemv, cblas_zgemv,
                        Layout::ColMajor,
                        transA,
                        m,
                        n,
                        alpha,
                        A_view[i].data_ptr(),
                        A_view[i].ld(),
                        Xi.data_ptr(),
                        Xi.inc(),
                        beta,
                        Yi.data_ptr(),
                        Yi.inc());
                }
            } else {
                call_backend_nh<T, BackendLibrary::CBLAS>(
                    cblas_sgemv, cblas_dgemv, cblas_cgemv, cblas_zgemv,
                    Layout::ColMajor,
                    transA,
                    m,
                    n,
                    alpha,
                    A_view.data_ptr(),
                    A_view.ld(),
                    X_view.data_ptr(),
                    X_view.inc(),
                    beta,
                    Y_view.data_ptr(),
                    Y_view.inc());
            }
        });
    }

    template <Backend B, typename T>
    Event trsm(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& descrA,
        const MatrixView<T, MatrixFormat::Dense>& descrB,
        Side side,
        Uplo uplo,
        Transpose transA,
        Diag diag,
        T alpha) {
        auto A_view = descrA;
        auto B_view = descrB;
        return detail::submit_host_task(ctx, "netlib.trsm", [=] {
            const int m = B_view.rows();
            const int n = B_view.cols();
            constexpr bool is_complex = std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;
            const bool do_conj = (transA == Transpose::ConjTrans) && is_complex;
            const bool do_trans = (transA != Transpose::NoTrans);
            const bool op_is_lower = (uplo == Uplo::Lower) ? !do_trans : do_trans;
            const bool unit_diag = (diag == Diag::Unit);

            auto conj_if = [=](T v) {
                if constexpr (is_complex) {
                    if (!do_conj) return v;
                    using std::conj;
                    return conj(v);
                } else {
                    return v;
                }
            };

            const int batch = A_view.batch_size();
            for (int b = 0; b < batch; ++b) {
                auto Ab = A_view.batch_item(b);
                auto Bb = B_view.batch_item(b);
                const int rows = Bb.rows();
                const int cols = Bb.cols();

                auto opA = [&](int r, int c) {
                    if (transA == Transpose::NoTrans) {
                        return Ab.at(r, c, 0);
                    }
                    return conj_if(Ab.at(c, r, 0));
                };

                if (side == Side::Left) {
                    const int dim = rows;
                    for (int j = 0; j < cols; ++j) {
                        if (op_is_lower) {
                            for (int i = 0; i < dim; ++i) {
                                T sum = T(0);
                                for (int k = 0; k < i; ++k) {
                                    sum += opA(i, k) * Bb.at(k, j, 0);
                                }
                                T x = alpha * Bb.at(i, j, 0) - sum;
                                if (!unit_diag) {
                                    x /= opA(i, i);
                                }
                                Bb.at(i, j, 0) = x;
                            }
                        } else {
                            for (int i = dim - 1; i >= 0; --i) {
                                T sum = T(0);
                                for (int k = i + 1; k < dim; ++k) {
                                    sum += opA(i, k) * Bb.at(k, j, 0);
                                }
                                T x = alpha * Bb.at(i, j, 0) - sum;
                                if (!unit_diag) {
                                    x /= opA(i, i);
                                }
                                Bb.at(i, j, 0) = x;
                            }
                        }
                    }
                } else {
                    const int dim = cols;
                    for (int i = 0; i < rows; ++i) {
                        if (op_is_lower) {
                            for (int j = dim - 1; j >= 0; --j) {
                                T sum = T(0);
                                for (int k = j + 1; k < dim; ++k) {
                                    sum += Bb.at(i, k, 0) * opA(k, j);
                                }
                                T x = alpha * Bb.at(i, j, 0) - sum;
                                if (!unit_diag) {
                                    x /= opA(j, j);
                                }
                                Bb.at(i, j, 0) = x;
                            }
                        } else {
                            for (int j = 0; j < dim; ++j) {
                                T sum = T(0);
                                for (int k = 0; k < j; ++k) {
                                    sum += Bb.at(i, k, 0) * opA(k, j);
                                }
                                T x = alpha * Bb.at(i, j, 0) - sum;
                                if (!unit_diag) {
                                    x /= opA(j, j);
                                }
                                Bb.at(i, j, 0) = x;
                            }
                        }
                    }
                }
            }
        });
    }

    template <Backend B, typename T>
    Event potrf(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& descrA,
                    Uplo uplo,
                    Span<std::byte> workspace) {
        static_cast<void>(workspace);
        auto A_view = descrA;
        return detail::submit_host_task(ctx, "netlib.potrf", [=] {
            if (A_view.batch_size() == 1) {
                call_backend_nh<T, BackendLibrary::LAPACKE>(
                    LAPACKE_spotrf, LAPACKE_dpotrf, LAPACKE_cpotrf, LAPACKE_zpotrf,
                    Layout::ColMajor, uplo,
                    A_view.rows(), A_view.data_ptr(), A_view.ld());
            } else {
                for (int i = 0; i < A_view.batch_size(); ++i) {
                    call_backend_nh<T, BackendLibrary::LAPACKE>(
                        LAPACKE_spotrf, LAPACKE_dpotrf, LAPACKE_cpotrf, LAPACKE_zpotrf,
                        Layout::ColMajor, uplo,
                        A_view[i].rows(), A_view[i].data_ptr(), A_view[i].ld());
                }
            }
        });
    }

    namespace backend {

    template <Backend B, typename T>
    Event syev_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& descrA,
                      Span<typename base_type<T>::type> eigenvalues,
                      JobType jobtype,
                      Uplo uplo,
                      Span<std::byte> /*workspace*/) {
        auto A_view = descrA;
        auto eig = eigenvalues;
        return op_external("lapacke.syev", [&, A_view, eig, jobtype, uplo] {
            return detail::submit_host_task(ctx, "lapacke.syev", [=] {
                if (A_view.batch_size() == 1) {
                    call_backend_nh<T, BackendLibrary::LAPACKE>(
                        LAPACKE_ssyev, LAPACKE_dsyev, LAPACKE_cheev, LAPACKE_zheev,
                        Layout::ColMajor, jobtype, uplo,
                        A_view.rows(), A_view.data_ptr(), A_view.ld(),
                        base_float_ptr_convert(eig.data()));
                } else {
                    for (int i = 0; i < A_view.batch_size(); ++i) {
                        call_backend_nh<T, BackendLibrary::LAPACKE>(
                            LAPACKE_ssyev, LAPACKE_dsyev, LAPACKE_cheev, LAPACKE_zheev,
                            Layout::ColMajor, jobtype, uplo,
                            A_view[i].rows(),
                            A_view[i].data_ptr(),
                            A_view[i].ld(),
                            base_float_ptr_convert(eig.subspan(i * A_view.rows()).data()));
                    }
                }
            });
        });
    }

    template <Backend B, typename T>
    size_t syev_vendor_buffer_size(Queue& /*ctx*/, 
                                   const MatrixView<T, MatrixFormat::Dense>& /*descrA*/,
                                   Span<typename base_type<T>::type> /*eigenvalues*/,
                                   JobType /*jobtype*/,
                                   Uplo /*uplo*/) {
        // LAPACKE path uses no user-provided workspace.
        return op_external("lapacke.syev_buffer_size", [&] { return static_cast<size_t>(0); });
    }

    } // namespace backend

    template <Backend Back, typename T>
    Event getrs(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& B,
                Transpose transA,
                Span<int64_t> pivots,
                Span<std::byte> workspace) {
        BumpAllocator pool(workspace);
        auto A_view = A;
        auto B_view = B;
        auto piv = pivots;
        const int n = A_view.rows();
        const int batch = A_view.batch_size();
        auto piv_i32 = pool.allocate<int>(ctx, n * batch);

        EventImpl conv_impl = ctx->submit([&](sycl::handler& h) {
            auto piv_in = piv.as_span<int64_t>();
            auto piv_out = piv_i32;
            h.parallel_for(sycl::range<1>(static_cast<size_t>(n * batch)), [=](sycl::id<1> idx) {
                piv_out[static_cast<int>(idx[0])] = static_cast<int>(piv_in[static_cast<int>(idx[0])]);
            });
        });
        Event conv_event(std::move(conv_impl));
        ctx.enqueue(conv_event);
        return detail::submit_host_task(ctx, "netlib.getrs", [=] {
            int nrhs = B_view.cols();
            for (int i = 0; i < A_view.batch_size(); ++i) {
                call_backend_nh<T, BackendLibrary::LAPACKE>(
                    LAPACKE_sgetrs, LAPACKE_dgetrs, LAPACKE_cgetrs, LAPACKE_zgetrs,
                    Layout::ColMajor,
                    transA,
                    n,
                    nrhs,
                    A_view[i].data_ptr(),
                    A_view.ld(),
                    piv_i32.data() + i * n,
                    B_view[i].data_ptr(),
                    B_view.ld());
            }
        });
    }

    template <Backend Back, typename T>
    size_t getrs_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             Transpose transA) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(B);
        static_cast<void>(transA);
        return BumpAllocator::allocation_size<int>(ctx, A.rows() * A.batch_size());
    }

    template <Backend B, typename T>
    Event getrf(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<int64_t> pivots,
                Span<std::byte> workspace) {
        BumpAllocator pool(workspace);
        auto A_view = A;
        auto piv = pivots;
        const int n = A_view.rows();
        const int batch = A_view.batch_size();
        auto piv_i32 = pool.allocate<int>(ctx, n * batch);

        Event getrf_event = detail::submit_host_task(ctx, "netlib.getrf", [=] {
            for (int i = 0; i < batch; ++i) {
                call_backend_nh<T, BackendLibrary::LAPACKE>(
                    LAPACKE_sgetrf, LAPACKE_dgetrf, LAPACKE_cgetrf, LAPACKE_zgetrf,
                    Layout::ColMajor,
                    n,
                    n,
                    A_view[i].data_ptr(),
                    A_view.ld(),
                    piv_i32.data() + i * n);
            }
        });
        ctx.enqueue(getrf_event);

        EventImpl piv_impl = ctx->submit([&](sycl::handler& h) {
            auto piv_out = piv.as_span<int64_t>();
            auto piv_in = piv_i32;
            h.depends_on(static_cast<sycl::event>(*ctx.get_event()));
            h.parallel_for(sycl::range<1>(static_cast<size_t>(n * batch)), [=](sycl::id<1> idx) {
                piv_out[static_cast<int>(idx[0])] = static_cast<int64_t>(piv_in[static_cast<int>(idx[0])]);
            });
        });
        Event piv_event(std::move(piv_impl));
        ctx.enqueue(piv_event);
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t getrf_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        return BumpAllocator::allocation_size<int>(ctx, A.rows() * A.batch_size());
    }

    template <Backend B, typename T>
    Event getri(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& C,
                Span<int64_t> pivots,
                Span<std::byte> workspace) {
        BumpAllocator pool(workspace);
        auto A_view = A;
        auto C_view = C;
        auto piv = pivots;
        const int n = A_view.rows();
        const int batch = A_view.batch_size();
        auto piv_i32 = pool.allocate<int>(ctx, n * batch);
        return detail::submit_host_task(ctx, "netlib.getri", [=] {
            auto piv_in = piv.as_span<int64_t>();
            for (int b = 0; b < batch; ++b) {
                auto Ab = A_view[b];
                auto Cb = C_view[b];
                std::copy(Ab.data_ptr(), Ab.data_ptr() + n * n, Cb.data_ptr());
                for (int i = 0; i < n; ++i) {
                    piv_i32[b * n + i] = static_cast<int>(piv_in[b * n + i]);
                }

                call_backend_nh<T, BackendLibrary::LAPACKE>(
                    LAPACKE_sgetri, LAPACKE_dgetri, LAPACKE_cgetri, LAPACKE_zgetri,
                    Layout::ColMajor,
                    n,
                    Cb.data_ptr(),
                    Cb.ld(),
                    piv_i32.data() + b * n);
            }
        });
    }

    template <Backend B, typename T>
    size_t getri_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        return BumpAllocator::allocation_size<int>(ctx, A.rows() * A.batch_size());
    }

    template <Backend B, typename T>
    Event geqrf(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<T> tau,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        auto A_view = A;
        auto tau_view = tau;
        return detail::submit_host_task(ctx, "netlib.geqrf", [=] {
            int m = A_view.rows();
            int n = A_view.cols();
            for (int i = 0; i < A_view.batch_size(); ++i) {
                call_backend_nh<T, BackendLibrary::LAPACKE>(
                    LAPACKE_sgeqrf, LAPACKE_dgeqrf, LAPACKE_cgeqrf, LAPACKE_zgeqrf,
                    Layout::ColMajor,
                    m,
                    n,
                    A_view[i].data_ptr(),
                    A_view.ld(),
                    tau_view.data() + i * std::min(m, n));
            }
        });
    }

    template <Backend B, typename T>
    size_t geqrf_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(tau);
        return 0;
    }

    template <Backend B, typename T>
    Event orgqr(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<T> tau,
                Span<std::byte> workspace) {
        static_cast<void>(workspace);
        auto A_view = A;
        auto tau_view = tau;
        return detail::submit_host_task(ctx, "netlib.orgqr", [=] {
            int m = A_view.rows();
            int n = A_view.cols();
            int k = std::min(m, n);
            for (int i = 0; i < A_view.batch_size(); ++i) {
                call_backend_nh<T, BackendLibrary::LAPACKE>(
                    LAPACKE_sorgqr, LAPACKE_dorgqr, LAPACKE_cungqr, LAPACKE_zungqr,
                    Layout::ColMajor,
                    m,
                    n,
                    k,
                    A_view[i].data_ptr(),
                    A_view.ld(),
                    tau_view.data() + i * k);
            }
        });
    }

    template <Backend B, typename T>
    size_t orgqr_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(tau);
        return 0;
    }

    namespace backend {

    template <Backend B, typename T>
    Event ormqr_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      Side side,
                      Transpose trans,
                      Span<T> tau,
                      Span<std::byte> workspace) {
        auto A_view = A;
        auto C_view = C;
        auto tau_view = tau;
        return op_external("lapacke.ormqr_vendor", [&, A_view, C_view, tau_view, side, trans] {
            return detail::submit_host_task(ctx, "lapacke.ormqr_vendor", [=] {
                static_cast<void>(workspace);
                int m = C_view.rows();
                int n = C_view.cols();
                int k = std::min(A_view.rows(), A_view.cols());
                for (int i = 0; i < A_view.batch_size(); ++i) {
                    call_backend_nh<T, BackendLibrary::LAPACKE>(
                        LAPACKE_sormqr, LAPACKE_dormqr, LAPACKE_cunmqr, LAPACKE_zunmqr,
                        Layout::ColMajor,
                        side,
                        trans,
                        m,
                        n,
                        k,
                        A_view[i].data_ptr(),
                        A_view.ld(),
                        tau_view.data() + i * k,
                        C_view[i].data_ptr(),
                        C_view.ld());
                }
            });
        });
    }

    template <Backend B, typename T>
    size_t ormqr_vendor_buffer_size(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    Side side,
                                    Transpose trans,
                                    Span<T> tau) {
        return op_external("lapacke.ormqr_vendor_buffer_size", [&] {
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

    template <Backend B, typename T>
    size_t potrf_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& descrA,
                             Uplo uplo) {
        static_cast<void>(ctx);
        static_cast<void>(descrA);
        static_cast<void>(uplo);
        return 0;
    }


    #define SPMM_INSTANTIATE(fp, F) \
    template Event spmm<Backend::NETLIB, fp, F>( \
        Queue&, \
        const MatrixView<fp, F>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, fp, Transpose, Transpose, Span<std::byte>);

    #define SPMM_BUFFER_SIZE_INSTANTIATE(fp, F) \
    template size_t spmm_buffer_size<Backend::NETLIB, fp, F>( \
        Queue&, \
        const MatrixView<fp, F>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, fp, Transpose, Transpose);

    #define GEMM_INSTANTIATE(fp) \
    template Event gemm<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, fp, Transpose, Transpose, ComputePrecision);

    #define GEMV_INSTANTIATE(fp) \
    template Event gemv<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<fp>&, \
        const VectorView<fp>&, \
        fp, fp, Transpose);

    #define TRSM_INSTANTIATE(fp) \
    template Event trsm<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Side, Uplo, Transpose, Diag, fp);

    #define GEQRF_INSTANTIATE(fp) \
    template Event geqrf<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>, \
        Span<std::byte>);

    #define GEQRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t geqrf_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>);

    #define ORGQR_INSTANTIATE(fp) \
    template Event orgqr<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>, \
        Span<std::byte>);

    #define ORGQR_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t orgqr_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<fp>);

    #define GETRS_INSTANTIATE(fp) \
    template Event getrs<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Transpose, \
        Span<int64_t>, \
        Span<std::byte>);

    #define GETRS_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t getrs_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Transpose);

    #define GETRF_INSTANTIATE(fp) \
    template Event getrf<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<int64_t>, \
        Span<std::byte>);

    #define GETRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t getrf_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&);

    #define GETRI_INSTANTIATE(fp) \
    template Event getri<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<int64_t>, \
        Span<std::byte>);

    #define GETRI_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t getri_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&);

    #define ORMQR_INSTANTIATE(fp) \
    template Event ormqr<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Side, Transpose, \
        Span<fp>, \
        Span<std::byte>);

    #define ORMQR_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t ormqr_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Side, Transpose, \
        Span<fp>);

    #define ORMQR_VENDOR_INSTANTIATE(fp) \
    template Event backend::ormqr_vendor<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Side, Transpose, \
        Span<fp>, \
        Span<std::byte>);

    #define ORMQR_VENDOR_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t backend::ormqr_vendor_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Side, Transpose, \
        Span<fp>);

    #define POTRF_INSTANTIATE(fp) \
    template Event potrf<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Uplo, Span<std::byte>);

    #define POTRF_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t potrf_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Uplo);

    #define SYEV_INSTANTIATE(fp) \
    template Event syev<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        JobType, Uplo, \
        Span<std::byte>);

    #define SYEV_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t syev_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        JobType, Uplo);

    #define SYEV_VENDOR_INSTANTIATE(fp) \
    template Event backend::syev_vendor<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        JobType, Uplo, \
        Span<std::byte>);

    #define SYEV_VENDOR_BUFFER_SIZE_INSTANTIATE(fp) \
    template size_t backend::syev_vendor_buffer_size<Backend::NETLIB, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        JobType, Uplo);

    #define BLAS_LEVEL3_INSTANTIATE(fp) \
        SPMM_INSTANTIATE(fp, MatrixFormat::CSR) \
        SPMM_BUFFER_SIZE_INSTANTIATE(fp, MatrixFormat::CSR) \
        GEMM_INSTANTIATE(fp) \
        GEMV_INSTANTIATE(fp) \
        TRSM_INSTANTIATE(fp) \
        GEQRF_INSTANTIATE(fp) \
        GEQRF_BUFFER_SIZE_INSTANTIATE(fp) \
        GETRS_INSTANTIATE(fp) \
        GETRS_BUFFER_SIZE_INSTANTIATE(fp) \
        GETRF_INSTANTIATE(fp) \
        GETRF_BUFFER_SIZE_INSTANTIATE(fp) \
        GETRI_INSTANTIATE(fp) \
        GETRI_BUFFER_SIZE_INSTANTIATE(fp) \
        ORMQR_INSTANTIATE(fp) \
        ORMQR_BUFFER_SIZE_INSTANTIATE(fp) \
        ORMQR_VENDOR_INSTANTIATE(fp) \
        ORMQR_VENDOR_BUFFER_SIZE_INSTANTIATE(fp) \
        ORGQR_INSTANTIATE(fp) \
        ORGQR_BUFFER_SIZE_INSTANTIATE(fp) \
        POTRF_INSTANTIATE(fp) \
        POTRF_BUFFER_SIZE_INSTANTIATE(fp) \
        SYEV_INSTANTIATE(fp) \
        SYEV_BUFFER_SIZE_INSTANTIATE(fp) \
        SYEV_VENDOR_INSTANTIATE(fp) \
        SYEV_VENDOR_BUFFER_SIZE_INSTANTIATE(fp)

    // Instantiate for the floating-point types of interest.
    BLAS_LEVEL3_INSTANTIATE(float)
    BLAS_LEVEL3_INSTANTIATE(double)
    BLAS_LEVEL3_INSTANTIATE(std::complex<float>)
    BLAS_LEVEL3_INSTANTIATE(std::complex<double>)

    #undef SPMM_INSTANTIATE
    #undef SPMM_BUFFER_SIZE_INSTANTIATE
    #undef GEMM_INSTANTIATE
    #undef GEMV_INSTANTIATE
    #undef TRSM_INSTANTIATE
    #undef GEQRF_INSTANTIATE
    #undef GEQRF_BUFFER_SIZE_INSTANTIATE
    #undef GETRS_INSTANTIATE
    #undef GETRS_BUFFER_SIZE_INSTANTIATE
    #undef GETRF_INSTANTIATE
    #undef GETRF_BUFFER_SIZE_INSTANTIATE
    #undef GETRI_INSTANTIATE
    #undef GETRI_BUFFER_SIZE_INSTANTIATE
    #undef ORMQR_INSTANTIATE
    #undef ORMQR_BUFFER_SIZE_INSTANTIATE
    #undef ORMQR_VENDOR_INSTANTIATE
    #undef ORMQR_VENDOR_BUFFER_SIZE_INSTANTIATE
    #undef ORGQR_INSTANTIATE
    #undef ORGQR_BUFFER_SIZE_INSTANTIATE
    #undef POTRF_INSTANTIATE
    #undef POTRF_BUFFER_SIZE_INSTANTIATE
    #undef SYEV_INSTANTIATE
    #undef SYEV_BUFFER_SIZE_INSTANTIATE
    #undef SYEV_VENDOR_INSTANTIATE
    #undef SYEV_VENDOR_BUFFER_SIZE_INSTANTIATE
    #undef BLAS_LEVEL3_INSTANTIATE
}