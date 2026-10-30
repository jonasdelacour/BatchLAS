// SPDX-License-Identifier: MIT
#include <batchlas/blas/linalg.hh>
#include <src/linalg-impl.hh>
#include <src/backends/level3_shape.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <src/queue.hh>
#include <sycl/sycl.hpp>
#include <batchlas/backend_config.h>
#include <complex>
#include <utility>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <lapack.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/mempool.hh>

#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/functions/sytrf.hh>
#include <batchlas/blas/functions/sygvd.hh>

#include <src/backends/gemm_heterogeneous.hh>
#include <src/backends/host_blas_health.hh>
#include <src/util/template-instantiations.hh>
#include <src/util/batch_mask.hh>
#include <batchlas/settings.hh>

namespace batchlas{

    namespace detail {

    // Host BLAS dgemm health check, re-run at first double use (the configure-time answer is stale
    // on another machine). Deliberately NO setenv(): OpenBLAS already read OPENBLAS_CORETYPE.
    // evidence: docs/design/runtime-internals.md#runtime-internals-the-host-blas-dgemm-health-check

    // off | warn (default) | error; `off` is refused by the unsafe gate, `error` is let through.
    inline batchlas::UnsafeSettings::BlasHealth host_blas_health_mode() {
        return batchlas::settings().unsafe.blas_health;
    }

    inline HostBlasDoubleHealth probe_host_dgemm() {
        return probe_dgemm([](int n, double* a, double* b, double* c) {
            call_backend_nh<double, BackendLibrary::CBLAS>(
                cblas_sgemm, cblas_dgemm, cblas_cgemm, cblas_zgemm,
                Layout::ColMajor, Transpose::NoTrans, Transpose::NoTrans,
                n, n, n, 1.0, a, n, b, n, 0.0, c, n);
        });
    }

    inline const HostBlasDoubleHealth& host_blas_double_health() {
        static const HostBlasDoubleHealth health = [] {
            if (host_blas_health_mode() == batchlas::UnsafeSettings::BlasHealth::Off) {
                return HostBlasDoubleHealth{};
            }
            return probe_host_dgemm();
        }();
        return health;
    }

    inline void check_host_blas_double_health() {
        const HostBlasDoubleHealth& health = host_blas_double_health();
        if (health.ok) return;

        const std::string msg = host_blas_double_health_message(health);
        if (host_blas_health_mode() == batchlas::UnsafeSettings::BlasHealth::Error) {
            throw batchlas::device_error(msg);
        }
        // Warn once, loudly.
        static const bool warned = [&] {
            std::fputs("\n============================================================\n", stderr);
            std::fputs(msg.c_str(), stderr);
            std::fputs("============================================================\n\n", stderr);
            std::fflush(stderr);
            return true;
        }();
        static_cast<void>(warned);
    }

    // Double precision only: a broken dgemm does not make single precision wrong.
    template <typename T>
    inline void host_blas_double_guard() {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) {
            check_host_blas_double_health();
        }
    }

    // T is the element type of the operation, used only for the double-precision
    // health check above; it defaults to void (check skipped) so untyped callers
    // keep compiling.
    template <typename T = void, typename F>
    Event submit_host_task(Queue& ctx, const char* /*label*/, F&& f) {
        if constexpr (!std::is_void_v<T>) {
            host_blas_double_guard<T>();
        }
        ctx.wait();
        f();
        try {
            sycl::event e = impl::submit_barrier(*ctx);
            return Event(EventImpl(std::move(e)));
        } catch (const sycl::exception&) {
        }
        EventImpl ev = ctx->submit([&](sycl::handler& h) {
            h.single_task([]() {});
        });
        return Event(std::move(ev));
    }
    } // namespace detail

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
        ::batchlas::detail::refuse_batch_mask<T>(Op::spmm, "spmm_vendor", Back, A, B, C);
        static_cast<void>(workspace); // no workspace needed for CPU implementation
        auto A_view = A;
        auto B_view = B;
        auto C_view = C;
        // No double-precision guard here: the CSR path below is hand-written and
        // never calls into the host BLAS, so a broken dgemm cannot affect it.
        return ::batchlas::detail::submit_host_task(ctx, "netlib.spmm", [=] {
            if constexpr (MFormat == MatrixFormat::CSR) {
                int batch = A_view.batch_size();
                // alpha == 0 means A is NOT READ -- not its values, not its
                // column indices, and not its row offsets. Callers hand spmm an
                // A-adjacent BumpAllocator allocation that is not zeroed
                // (mempool.hh), and 0 * NaN is NaN, so multiplying an unread A
                // by a zero alpha poisons the result instead of dropping it.
                // The native bodies in src/ops/spmm/kernels/spmm_native.cc make the same
                // guarantee (gather skips the nonzero loop, scatter skips the
                // launch outright); this is the host arm agreeing. Note this is
                // a skip of the alpha TERM only: C is still written below, so
                // alpha == 0 && beta == 0 zeroes C rather than leaving it alone.
                const bool alpha_zero = (alpha == T(0));
                for (int b = 0; b < batch; ++b) {
                    auto A_b = A_view[b];
                    auto B_b = B_view[b];
                    auto C_b = C_view[b];

                    int m = A_b.rows();
                    int k = A_b.cols();
                    int n = B_b.cols();

                    // Only handle no transpose cases for now
                    if (transA != Transpose::NoTrans || transB != Transpose::NoTrans) {
                        throw batchlas::unsupported("NETLIB spmm only supports NoTrans for now");
                    }

                    for (int row = 0; row < m; ++row) {
                        for (int col = 0; col < n; ++col) {
                            // beta == 0 means C is NOT READ. Every in-library
                            // caller passes beta = 0 into a BumpAllocator
                            // allocation, which is not zeroed (mempool.hh), so
                            // reading C here propagates whatever was in that
                            // memory -- NaN included -- into the result. The
                            // native bodies in src/ops/spmm/kernels/spmm_native.cc make the
                            // same guarantee; this is the host arm agreeing.
                            T sum = (beta == T(0)) ? T(0)
                                                   : beta * C_b.at(row, col);
                            if (!alpha_zero) {
                                for (int idx = A_b.row_offsets()[row]; idx < A_b.row_offsets()[row + 1]; ++idx) {
                                    int a_col = A_b.col_indices()[idx];
                                    // B is only ever touched from inside this
                                    // loop, so guarding the loop is also what
                                    // keeps B unread at alpha == 0.
                                    sum += alpha * A_b.data()[idx] * B_b.at(a_col, col);
                                }
                            }
                            C_b.at(row, col) = sum;
                        }
                    }
                }
            } else {
                throw batchlas::unsupported("Unsupported sparse format for NETLIB spmm");
            }
        });
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

    } // namespace backend

    // The netlib gemm IS the vendor implementation (the CBLAS call), hence gemm_vendor here.
    namespace backend {

    template <Backend B, typename T>
    Event gemm_vendor(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& descrA,
                   const MatrixView<T, MatrixFormat::Dense>& descrB,
                   const MatrixView<T, MatrixFormat::Dense>& descrC,
                   T alpha,
                   T beta,
                   Transpose transA,
                   Transpose transB,
                   ComputePrecision precision) {
        ::batchlas::detail::refuse_batch_mask<T>(Op::gemm, "gemm_vendor", B, descrA, descrB, descrC);
        static_cast<void>(precision);
        auto A_view = descrA;
        auto B_view = descrB;
        auto C_view = descrC;
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.gemm", [=] {
            if (!backend::gemm_batch_dimensions_compatible(A_view, B_view, C_view, transA, transB)) {
                throw batchlas::invalid_argument("gemm: incompatible matrix dimensions");
            }

            if (A_view.batch_size() == 1) {
                auto [m, k] = get_effective_dims(A_view, transA);
                auto [kB, n] = get_effective_dims(B_view, transB);
                static_cast<void>(kB);
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
                    auto A_i = A_view[i];
                    auto B_i = B_view[i];
                    auto C_i = C_view[i];
                    auto [m, k] = get_effective_dims(A_i, transA);
                    auto [kB, n] = get_effective_dims(B_i, transB);
                    static_cast<void>(kB);
                    if (m == 0 || n == 0) {
                        continue;
                    }
                    if (k == 0) {
                        for (int col = 0; col < n; ++col) {
                            for (int row = 0; row < m; ++row) {
                                C_i.at(row, col) = beta == T(0) ? T(0) : C_i.at(row, col) * beta;
                            }
                        }
                        continue;
                    }
                    call_backend_nh<T, BackendLibrary::CBLAS>(
                        cblas_sgemm, cblas_dgemm, cblas_cgemm, cblas_zgemm,
                        Layout::ColMajor, transA, transB,
                        m, n, k,
                        alpha,
                        A_i.data_ptr(), A_i.ld(),
                        B_i.data_ptr(), B_i.ld(),
                        beta,
                        C_i.data_ptr(), C_i.ld());
                }
            }
        });
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    Event gemv_vendor(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const VectorView<T>& X,
               const VectorView<T>& Y,
               T alpha,
               T beta,
               Transpose transA) {
        ::batchlas::detail::refuse_batch_mask<T>(Op::gemv, "gemv_vendor", B, A, X, Y);
        auto A_view = A;
        auto X_view = X;
        auto Y_view = Y;
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.gemv", [=] {
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

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    Event trsm_vendor(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& descrA,
        const MatrixView<T, MatrixFormat::Dense>& descrB,
        Side side,
        Uplo uplo,
        Transpose transA,
        Diag diag,
        T alpha) {
        ::batchlas::detail::refuse_batch_mask<T>(Op::trsm, "trsm_vendor", B, descrA, descrB);
        // Parameter order matches backend::trsm_vendor as cuBLAS defines it:
        // alpha LAST, unlike the public trsm, which takes it third. The two
        // orders coexisted for as long as each TU declared its own public trsm;
        // now that one declaration serves every backend, they have to agree.

        auto A_view = descrA;
        auto B_view = descrB;
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.trsm", [=] {
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

                // BLAS: alpha == 0 sets B := 0 without reading A or B.
                if (alpha == T(0)) {
                    for (int j = 0; j < cols; ++j) {
                        for (int i = 0; i < rows; ++i) {
                            Bb.at(i, j, 0) = T(0);
                        }
                    }
                    continue;
                }

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

    } // namespace backend

    namespace backend {

    template <Backend B, FloatingOrComplexScalar T>
    Event symm_vendor(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const MatrixView<T, MatrixFormat::Dense>& Bmat,
               const MatrixView<T, MatrixFormat::Dense>& Cmat,
               T alpha,
               T beta,
               Side side,
               Uplo uplo) {
        ::batchlas::detail::refuse_batch_mask<T>(Op::symm, "symm_vendor", B, A, Bmat, Cmat);
        auto A_view = A;
        auto B_view = Bmat;
        auto C_view = Cmat;
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.symm", [=] {
            const auto [m, n, k] = backend::shape::validate_product<batchlas::internal_error>("symm", A_view, B_view, C_view, side);
            static_cast<void>(k);  // cblas_?symm takes A's order from side, not as an argument

            // The complex slots are the complex-symmetric ?symm (hemm is the Hermitian one).
            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& B_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                call_backend_nh<T, BackendLibrary::CBLAS>(
                    cblas_ssymm, cblas_dsymm, cblas_csymm, cblas_zsymm,
                    Layout::ColMajor,
                    side,
                    uplo,
                    m,
                    n,
                    alpha,
                    A_i.data_ptr(),
                    A_i.ld(),
                    B_i.data_ptr(),
                    B_i.ld(),
                    beta,
                    C_i.data_ptr(),
                    C_i.ld());
            };

            for (int batch = 0; batch < A_view.batch_size(); ++batch) {
                launch_single(A_view[batch], B_view[batch], C_view[batch]);
            }
        });
    }

    } // namespace backend

    namespace backend {

    template <Backend B, ComplexScalar T>
    Event hemm_vendor(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const MatrixView<T, MatrixFormat::Dense>& Bmat,
               const MatrixView<T, MatrixFormat::Dense>& Cmat,
               T alpha,
               T beta,
               Side side,
               Uplo uplo) {
        ::batchlas::detail::refuse_batch_mask<T>(Op::hemm, "hemm_vendor", B, A, Bmat, Cmat);
        auto A_view = A;
        auto B_view = Bmat;
        auto C_view = Cmat;
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.hemm", [=] {
            const auto [m, n, k] = backend::shape::validate_product<batchlas::internal_error>("hemm", A_view, B_view, C_view, side);
            static_cast<void>(k);  // cblas_?hemm takes A's order from side, not as an argument

            // The two real slots have no callee because BLAS has no real ?hemm;
            // T is constrained to complex, so they are never selected.
            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& B_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                call_backend_nh<T, BackendLibrary::CBLAS>(
                    nullptr, nullptr, cblas_chemm, cblas_zhemm,
                    Layout::ColMajor,
                    side,
                    uplo,
                    m,
                    n,
                    alpha,
                    A_i.data_ptr(),
                    A_i.ld(),
                    B_i.data_ptr(),
                    B_i.ld(),
                    beta,
                    C_i.data_ptr(),
                    C_i.ld());
            };

            for (int batch = 0; batch < A_view.batch_size(); ++batch) {
                launch_single(A_view[batch], B_view[batch], C_view[batch]);
            }
        });
    }

    } // namespace backend


    namespace backend {

    template <Backend B, ComplexScalar T>
    Event herk_vendor(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const MatrixView<T, MatrixFormat::Dense>& Cmat,
               float_t<T> alpha,
               float_t<T> beta,
               Uplo uplo,
               Transpose transA) {
        ::batchlas::detail::refuse_batch_mask<T>(Op::herk, "herk_vendor", B, A, Cmat);
        auto A_view = A;
        auto C_view = Cmat;
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.herk", [=] {
            const auto [n, k] = backend::shape::validate_rank_k<batchlas::internal_error>("herk", A_view, C_view, transA, /*hermitian=*/true);

            // The two real slots have no callee because BLAS has no real
            // ?herk -- that is syrk; T is constrained to complex, so they are
            // never selected.
            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                call_backend_nh<T, BackendLibrary::CBLAS>(
                    nullptr, nullptr, cblas_cherk, cblas_zherk,
                    Layout::ColMajor,
                    uplo,
                    transA,
                    n,
                    k,
                    alpha,
                    A_i.data_ptr(),
                    A_i.ld(),
                    beta,
                    C_i.data_ptr(),
                    C_i.ld());
            };

            for (int batch = 0; batch < A_view.batch_size(); ++batch) {
                launch_single(A_view[batch], C_view[batch]);
            }
        });
    }

    } // namespace backend

    namespace backend {

    template <Backend B, ComplexScalar T>
    Event her2k_vendor(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& Bmat,
                const MatrixView<T, MatrixFormat::Dense>& Cmat,
                T alpha,
                float_t<T> beta,
                Uplo uplo,
                Transpose transA) {
        ::batchlas::detail::refuse_batch_mask<T>(Op::her2k, "her2k_vendor", B, A, Bmat, Cmat);
        auto A_view = A;
        auto B_view = Bmat;
        auto C_view = Cmat;
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.her2k", [=] {
            const auto [n, k] = backend::shape::validate_rank_2k<batchlas::internal_error>("her2k", A_view, B_view, C_view, transA, /*hermitian=*/true);

            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& B_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                call_backend_nh<T, BackendLibrary::CBLAS>(
                    nullptr, nullptr, cblas_cher2k, cblas_zher2k,
                    Layout::ColMajor,
                    uplo,
                    transA,
                    n,
                    k,
                    alpha,
                    A_i.data_ptr(),
                    A_i.ld(),
                    B_i.data_ptr(),
                    B_i.ld(),
                    beta,
                    C_i.data_ptr(),
                    C_i.ld());
            };

            for (int batch = 0; batch < A_view.batch_size(); ++batch) {
                launch_single(A_view[batch], B_view[batch], C_view[batch]);
            }
        });
    }

    } // namespace backend

    namespace backend {

    template <Backend B, FloatingOrComplexScalar T>
    Event syrk_vendor(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const MatrixView<T, MatrixFormat::Dense>& Cmat,
               T alpha,
               T beta,
               Uplo uplo,
               Transpose transA) {
        ::batchlas::detail::refuse_batch_mask<T>(Op::syrk, "syrk_vendor", B, A, Cmat);
        auto A_view = A;
        auto C_view = Cmat;
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.syrk", [=] {
            const auto [n, k] = backend::shape::validate_rank_k<batchlas::internal_error>("syrk", A_view, C_view, transA, /*hermitian=*/false);

            // The complex slots are the complex-symmetric ?syrk (herk is the Hermitian one).
            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                call_backend_nh<T, BackendLibrary::CBLAS>(
                    cblas_ssyrk, cblas_dsyrk, cblas_csyrk, cblas_zsyrk,
                    Layout::ColMajor,
                    uplo,
                    transA,
                    n,
                    k,
                    alpha,
                    A_i.data_ptr(),
                    A_i.ld(),
                    beta,
                    C_i.data_ptr(),
                    C_i.ld());
            };

            for (int batch = 0; batch < A_view.batch_size(); ++batch) {
                launch_single(A_view[batch], C_view[batch]);
            }
        });
    }

    } // namespace backend

    namespace backend {

    template <Backend B, FloatingOrComplexScalar T>
    Event syr2k_vendor(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& Bmat,
                const MatrixView<T, MatrixFormat::Dense>& Cmat,
                T alpha,
                T beta,
                Uplo uplo,
                Transpose transA) {
        ::batchlas::detail::refuse_batch_mask<T>(Op::syr2k, "syr2k_vendor", B, A, Bmat, Cmat);
        auto A_view = A;
        auto B_view = Bmat;
        auto C_view = Cmat;
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.syr2k", [=] {
            const auto [n, k] = backend::shape::validate_rank_2k<batchlas::internal_error>("syr2k", A_view, B_view, C_view, transA, /*hermitian=*/false);

            // The complex slots are the complex-symmetric ?syr2k (her2k is the Hermitian one).
            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& B_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                call_backend_nh<T, BackendLibrary::CBLAS>(
                    cblas_ssyr2k, cblas_dsyr2k, cblas_csyr2k, cblas_zsyr2k,
                    Layout::ColMajor,
                    uplo,
                    transA,
                    n,
                    k,
                    alpha,
                    A_i.data_ptr(),
                    A_i.ld(),
                    B_i.data_ptr(),
                    B_i.ld(),
                    beta,
                    C_i.data_ptr(),
                    C_i.ld());
            };

            for (int batch = 0; batch < A_view.batch_size(); ++batch) {
                launch_single(A_view[batch], B_view[batch], C_view[batch]);
            }
        });
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    Event trmm_vendor(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const MatrixView<T, MatrixFormat::Dense>& Bmat,
               const MatrixView<T, MatrixFormat::Dense>& Cmat,
               T alpha,
               Side side,
               Uplo uplo,
               Transpose transA,
               Diag diag) {
        ::batchlas::detail::refuse_batch_mask<T>(Op::trmm, "trmm_vendor", B, A, Bmat, Cmat);
        auto A_view = A;
        auto B_view = Bmat;
        auto C_view = Cmat;
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.trmm", [=] {
            const auto [m, n, k] = backend::shape::validate_product<batchlas::internal_error>("trmm", A_view, B_view, C_view, side);
            static_cast<void>(k);  // cblas_?trmm takes A's order from side, not as an argument

            // CBLAS ?trmm works in place, so the operand has to arrive in C.
            // Going through cblas_?gemm instead would be a plain dense product
            // against A's whole storage, which reads the triangle TRMM is
            // forbidden to touch and ignores Diag::Unit outright.
            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& B_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                for (int col = 0; col < n; ++col) {
                    std::copy_n(B_i.data_ptr() + static_cast<std::size_t>(col) * B_i.ld(),
                                m,
                                C_i.data_ptr() + static_cast<std::size_t>(col) * C_i.ld());
                }

                call_backend_nh<T, BackendLibrary::CBLAS>(
                    cblas_strmm, cblas_dtrmm, cblas_ctrmm, cblas_ztrmm,
                    Layout::ColMajor,
                    side,
                    uplo,
                    transA,
                    diag,
                    m,
                    n,
                    alpha,
                    A_i.data_ptr(),
                    A_i.ld(),
                    C_i.data_ptr(),
                    C_i.ld());
            };

            for (int batch = 0; batch < A_view.batch_size(); ++batch) {
                launch_single(A_view[batch], B_view[batch], C_view[batch]);
            }
        });
    }

    } // namespace backend

    namespace backend {

    // LAPACKE ?sytrf / ?sytrs per item. The pivots are written as int32 straight into the packed
    // first half of the int64 span (lapack_int is int32 in this build; a mismatch fails to compile),
    // the format the native kernels read. A batch mask is read inside the host task (idiom I4): an
    // inactive item's A, ipiv and info are never touched. evidence: docs/perf/sytrf.md
    template <Backend B, typename T>
    Event sytrf_vendor(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo, Span<int64_t> pivots,
                       Span<std::byte> /*workspace*/, Span<int32_t> info_out) {
        const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("sytrf_vendor", A));
        auto A_view = A;
        auto piv = pivots.as_span<int>();
        auto info = info_out;
        const bool want_info = info.size() >= static_cast<size_t>(A.batch_size());
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.sytrf", [=] {
            const int n = A_view.rows();
            for (int i = 0; i < A_view.batch_size(); ++i) {
                if (mask != nullptr && mask[i] == 0) continue;
                auto st = call_backend_nh_r<T, BackendLibrary::LAPACKE>(
                    LAPACKE_ssytrf, LAPACKE_dsytrf, LAPACKE_csytrf, LAPACKE_zsytrf, Layout::ColMajor, uplo, n,
                    A_view[i].data_ptr(), A_view.ld(), piv.data() + static_cast<size_t>(i) * n);
                if (want_info) info[i] = static_cast<int32_t>(st);
            }
        });
    }

    template <Backend B, typename T>
    size_t sytrf_vendor_buffer_size(Queue& /*ctx*/, const MatrixView<T, MatrixFormat::Dense>& /*A*/, Uplo /*uplo*/) {
        return 0;
    }

    template <Backend B, typename T>
    Event sytrs_vendor(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                       const MatrixView<T, MatrixFormat::Dense>& Bm, Uplo uplo, Span<int64_t> pivots,
                       const int32_t* skip_info) {
        const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("sytrs_vendor", A, Bm));
        auto A_view = A;
        auto B_view = Bm;
        auto piv = pivots.as_span<int>();
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.sytrs", [=] {
            const int n = A_view.rows();
            for (int i = 0; i < A_view.batch_size(); ++i) {
                if (mask != nullptr && mask[i] == 0) continue;
                if (skip_info != nullptr && skip_info[i] != 0) continue;
                (void)call_backend_nh_r<T, BackendLibrary::LAPACKE>(
                    LAPACKE_ssytrs, LAPACKE_dsytrs, LAPACKE_csytrs, LAPACKE_zsytrs, Layout::ColMajor, uplo, n,
                    B_view.cols(), A_view[i].data_ptr(), A_view.ld(), piv.data() + static_cast<size_t>(i) * n,
                    B_view[i].data_ptr(), B_view.ld());
            }
        });
    }

    #define NETLIB_SYTRF_OPS(B, fp) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, sytrf_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, sytrf_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, sytrs_vendor)
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(NETLIB_SYTRF_OPS, Backend::NETLIB)
    #undef NETLIB_SYTRF_OPS

    // LAPACKE ?sygst/?hegst and ?sygvd/?hegvd per item: sygst's and sygvd's vendor families on the host.
    // evidence: docs/perf/sygvd.md#sygvd-routes
    template <Backend B, typename T>
    Event sygst_vendor(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                       const MatrixView<T, MatrixFormat::Dense>& Bm, int itype, Uplo uplo,
                       Span<std::byte> /*workspace*/) {
        // I4: the mask is read inside the host task, after its producer on the in-order queue.
        const std::uint8_t* mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("sygst_vendor", A, Bm));
        auto a = A;
        auto b = Bm;
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.sygst", [=] {
            for (int i = 0; i < a.batch_size(); ++i) {
                if (mask != nullptr && mask[i] == 0) continue;
                (void)call_backend_nh_r<T, BackendLibrary::LAPACKE>(
                    LAPACKE_ssygst, LAPACKE_dsygst, LAPACKE_chegst, LAPACKE_zhegst, Layout::ColMajor, itype, uplo,
                    a.rows(), a[i].data_ptr(), a.ld(), b[i].data_ptr(), b.ld());
            }
        });
    }

    template <Backend B, typename T>
    Event sygvd_vendor(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                       const MatrixView<T, MatrixFormat::Dense>& Bm, Span<typename base_type<T>::type> w, int itype,
                       JobType jobz, Uplo uplo, Span<std::byte> /*workspace*/, Span<int32_t> info_out) {
        const std::uint8_t* mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("sygvd_vendor", A, Bm));
        auto a = A;
        auto b = Bm;
        auto eig = w;
        auto info = info_out;
        const bool want_info = info.size() >= static_cast<size_t>(A.batch_size());
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.sygvd", [=] {
            const int n = a.rows();
            for (int i = 0; i < a.batch_size(); ++i) {
                if (mask != nullptr && mask[i] == 0) continue;  // I4
                auto st = call_backend_nh_r<T, BackendLibrary::LAPACKE>(
                    LAPACKE_ssygvd, LAPACKE_dsygvd, LAPACKE_chegvd, LAPACKE_zhegvd, Layout::ColMajor, itype, jobz,
                    uplo, n, a[i].data_ptr(), a.ld(), b[i].data_ptr(), b.ld(),
                    base_float_ptr_convert(eig.subspan(static_cast<size_t>(i) * n).data()));
                if (want_info) info[i] = static_cast<int32_t>(st);
            }
        });
    }

    template <Backend B, typename T>
    size_t sygvd_vendor_buffer_size(Queue& /*ctx*/, const MatrixView<T, MatrixFormat::Dense>& /*A*/,
                                    const MatrixView<T, MatrixFormat::Dense>& /*Bm*/,
                                    Span<typename base_type<T>::type> /*w*/, int /*itype*/, JobType /*jobz*/,
                                    Uplo /*uplo*/) {
        return 0;
    }

    #define NETLIB_SYGVD_OPS(B, fp) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, sygst_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, sygvd_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, sygvd_vendor_buffer_size)
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(NETLIB_SYGVD_OPS, Backend::NETLIB)
    #undef NETLIB_SYGVD_OPS

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    Event potrf_vendor(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& descrA,
                    Uplo uplo,
                    Span<std::byte> workspace,
                    Span<int32_t> info_out) {
        ::batchlas::detail::refuse_batch_mask<T>(Op::potrf, "potrf_vendor", B, descrA);
        static_cast<void>(workspace);
        auto A_view = descrA;
        // LAPACKE reports failure only through its return value, and this loop
        // used to drop it on the floor (issue #73): an indefinite item came back
        // looking exactly like a factorised one. There is no pool fallback to
        // pick here -- unlike the vendor backends netlib needs no status array of
        // its own -- so an empty span simply means "do not record".
        auto info = info_out;
        const bool want_info = info.size() >= static_cast<size_t>(descrA.batch_size());
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.potrf", [=] {
            if (A_view.batch_size() == 1) {
                auto st = call_backend_nh_r<T, BackendLibrary::LAPACKE>(
                    LAPACKE_spotrf, LAPACKE_dpotrf, LAPACKE_cpotrf, LAPACKE_zpotrf,
                    Layout::ColMajor, uplo,
                    A_view.rows(), A_view.data_ptr(), A_view.ld());
                if (want_info) info[0] = static_cast<int32_t>(st);
            } else {
                for (int i = 0; i < A_view.batch_size(); ++i) {
                    auto st = call_backend_nh_r<T, BackendLibrary::LAPACKE>(
                        LAPACKE_spotrf, LAPACKE_dpotrf, LAPACKE_cpotrf, LAPACKE_zpotrf,
                        Layout::ColMajor, uplo,
                        A_view[i].rows(), A_view[i].data_ptr(), A_view[i].ld());
                    if (want_info) info[i] = static_cast<int32_t>(st);
                }
            }
        });
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    Event syev_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& descrA,
                      Span<typename base_type<T>::type> eigenvalues,
                      JobType jobtype,
                      Uplo uplo,
                      Span<std::byte> /*workspace*/,
                      Span<int32_t> info_out) {
        // I4: the mask is read inside the host task, after its producer on the in-order queue.
        const std::uint8_t* mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("syev_vendor", descrA));
        auto A_view = descrA;
        auto eig = eigenvalues;
        // call_backend_nh_r, not call_backend_nh: `_nh` DISCARDS the return value,
        // and LAPACKE_?syev's return value IS the LAPACK info -- > 0 means that many
        // off-diagonal elements failed to converge, which is exactly the contract
        // `info` documents. It was one macro away from potrf's working pattern in
        // this same file (see potrf_vendor above) and was being dropped for free.
        //
        // No pool fallback and no workspace: this arm takes none, so
        // syev_vendor_buffer_size still returns 0 whether or not status is asked for.
        auto info = info_out;
        const bool want_info = info.size() >= static_cast<size_t>(descrA.batch_size());
        return ::batchlas::detail::submit_host_task<T>(ctx, "lapacke.syev", [=] {
            if (A_view.batch_size() == 1) {
                if (mask != nullptr && mask[0] == 0) return;
                auto st = call_backend_nh_r<T, BackendLibrary::LAPACKE>(
                    LAPACKE_ssyev, LAPACKE_dsyev, LAPACKE_cheev, LAPACKE_zheev,
                    Layout::ColMajor, jobtype, uplo,
                    A_view.rows(), A_view.data_ptr(), A_view.ld(),
                    base_float_ptr_convert(eig.data()));
                if (want_info) info[0] = static_cast<int32_t>(st);
            } else {
                for (int i = 0; i < A_view.batch_size(); ++i) {
                    if (mask != nullptr && mask[i] == 0) continue;
                    auto st = call_backend_nh_r<T, BackendLibrary::LAPACKE>(
                        LAPACKE_ssyev, LAPACKE_dsyev, LAPACKE_cheev, LAPACKE_zheev,
                        Layout::ColMajor, jobtype, uplo,
                        A_view[i].rows(),
                        A_view[i].data_ptr(),
                        A_view[i].ld(),
                        base_float_ptr_convert(eig.subspan(i * A_view.rows()).data()));
                    if (want_info) info[i] = static_cast<int32_t>(st);
                }
            }
        });
    }

    template <Backend B, typename T>
    size_t syev_vendor_buffer_size(Queue& /*ctx*/,
                                   const MatrixView<T, MatrixFormat::Dense>& /*descrA*/,
                                   Span<typename base_type<T>::type> /*eigenvalues*/,
                                   JobType /*jobtype*/,
                                   Uplo /*uplo*/) {
        // LAPACKE path uses no user-provided workspace.
        return static_cast<size_t>(0);
    }

    // Synchronous ctx.wait() is deliberate: LAPACKE ?gesvd needs A on the host; this is the reference path.
    template <Backend B, typename T>
    Event gesvd_vendor(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       Span<typename base_type<T>::type> singular_values,
                       const MatrixView<T, MatrixFormat::Dense>& U,
                       const MatrixView<T, MatrixFormat::Dense>& Vh,
                       SvdVectors jobu,
                       SvdVectors jobvh,
                       Span<std::byte> workspace,
                       Span<int32_t> info_out) {
        // I4: the host loop below runs after ctx.wait(), so it reads the (possibly device-only,
        // host-addressable here) mask itself; an inactive item's A, s, U, Vh and info are untouched.
        const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("gesvd_vendor", A, U, Vh));
        static_cast<void>(workspace);

        // This path calls LAPACKE directly rather than going through
        // submit_host_task, so it has to run the double-precision health check
        // itself; otherwise a broken host dgemm is silent here.
        ::batchlas::detail::host_blas_double_guard<T>();

        if (A.batch_size() < 1 || A.rows() < 1 || A.cols() < 1) {
            throw batchlas::invalid_argument("gesvd_vendor (NETLIB): invalid matrix shape or batch size");
        }

        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        const int k = std::min(m, n);
        const int batch = static_cast<int>(A.batch_size());
        const std::size_t need_s = static_cast<std::size_t>(k) * static_cast<std::size_t>(batch);
        if (singular_values.size() < need_s) {
            throw batchlas::invalid_argument("gesvd_vendor (NETLIB): singular_values span too small");
        }

        // NETLIB implements Thin rather than refusing it. On this backend can_run
        // (src/ops/gesvd/gesvd.cc) admits only Vendor, so refusing would leave the
        // whole CPU backend unable to serve Thin -- and this is the reference
        // the GPU thin results are checked against.
        jobu = canonical_jobu(jobu, m, k);
        jobvh = canonical_jobvh(jobvh, n, k);

        const auto lapack_job = [](SvdVectors j) -> char {
            switch (j) {
                case SvdVectors::All:  return 'A';
                case SvdVectors::Thin: return 'S';
                default:               return 'N';
            }
        };
        const char lapack_jobu = lapack_job(jobu);
        const char lapack_jobvt = lapack_job(jobvh);

        if (jobu != SvdVectors::None) {
            const int want_cols = static_cast<int>(svd_u_cols(jobu, m, k));
            if (U.rows() != m || U.cols() != want_cols || U.batch_size() != batch) {
                throw batchlas::invalid_argument("gesvd_vendor (NETLIB): U must be (m x " +
                                            std::to_string(want_cols) + ") with matching batch");
            }
        }
        if (jobvh != SvdVectors::None) {
            const int want_rows = static_cast<int>(svd_vh_rows(jobvh, n, k));
            if (Vh.rows() != want_rows || Vh.cols() != n || Vh.batch_size() != batch) {
                throw batchlas::invalid_argument("gesvd_vendor (NETLIB): Vh must be (" +
                                            std::to_string(want_rows) + " x n) with matching batch");
            }
        }

        ctx.wait();

        std::vector<typename base_type<T>::type> superb(static_cast<std::size_t>(std::max(0, k - 1)));
        auto& A_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(A);
        for (int b = 0; b < batch; ++b) {
            if (mask != nullptr && mask[b] == 0) continue;
            auto Ab = A_mut.batch_item(b);
            // A None side may be an empty view with batch size 1: never index it.
            using DenseView = MatrixView<T, MatrixFormat::Dense>;
            const DenseView Ub = jobu != SvdVectors::None ? U.batch_item(b) : DenseView();
            const DenseView Vhb = jobvh != SvdVectors::None ? Vh.batch_item(b) : DenseView();
            typename base_type<T>::type* sb = singular_values.data() + static_cast<std::size_t>(b) * static_cast<std::size_t>(k);

            lapack_int info = 0;
            if constexpr (std::is_same_v<T, float>) {
                info = LAPACKE_sgesvd(LAPACK_COL_MAJOR, lapack_jobu, lapack_jobvt, m, n,
                                      Ab.data_ptr(), Ab.ld(), sb,
                                      (jobu != SvdVectors::None) ? Ub.data_ptr() : nullptr,
                                      (jobu != SvdVectors::None) ? Ub.ld() : 1,
                                      (jobvh != SvdVectors::None) ? Vhb.data_ptr() : nullptr,
                                      (jobvh != SvdVectors::None) ? Vhb.ld() : 1,
                                      superb.data());
            } else if constexpr (std::is_same_v<T, double>) {
                info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, lapack_jobu, lapack_jobvt, m, n,
                                      Ab.data_ptr(), Ab.ld(), sb,
                                      (jobu != SvdVectors::None) ? Ub.data_ptr() : nullptr,
                                      (jobu != SvdVectors::None) ? Ub.ld() : 1,
                                      (jobvh != SvdVectors::None) ? Vhb.data_ptr() : nullptr,
                                      (jobvh != SvdVectors::None) ? Vhb.ld() : 1,
                                      superb.data());
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                info = LAPACKE_cgesvd(LAPACK_COL_MAJOR, lapack_jobu, lapack_jobvt, m, n,
                                      reinterpret_cast<lapack_complex_float*>(Ab.data_ptr()), Ab.ld(), sb,
                                      (jobu != SvdVectors::None) ? reinterpret_cast<lapack_complex_float*>(Ub.data_ptr()) : nullptr,
                                      (jobu != SvdVectors::None) ? Ub.ld() : 1,
                                      (jobvh != SvdVectors::None) ? reinterpret_cast<lapack_complex_float*>(Vhb.data_ptr()) : nullptr,
                                      (jobvh != SvdVectors::None) ? Vhb.ld() : 1,
                                      superb.data());
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                info = LAPACKE_zgesvd(LAPACK_COL_MAJOR, lapack_jobu, lapack_jobvt, m, n,
                                      reinterpret_cast<lapack_complex_double*>(Ab.data_ptr()), Ab.ld(), sb,
                                      (jobu != SvdVectors::None) ? reinterpret_cast<lapack_complex_double*>(Ub.data_ptr()) : nullptr,
                                      (jobu != SvdVectors::None) ? Ub.ld() : 1,
                                      (jobvh != SvdVectors::None) ? reinterpret_cast<lapack_complex_double*>(Vhb.data_ptr()) : nullptr,
                                      (jobvh != SvdVectors::None) ? Vhb.ld() : 1,
                                      superb.data());
            } else {
                throw batchlas::unsupported("gesvd_vendor (NETLIB): unsupported scalar type");
            }

            // The per-item status existed inside this loop and the throw destroyed
            // it: a batch-wide exception says nothing about WHICH item failed.
            // Record it first, and when the caller asked for status let the call
            // return normally so it can read every item rather than just the first
            // failure.
            if (info_out.size() >= static_cast<size_t>(batch)) {
                info_out[static_cast<size_t>(b)] = static_cast<int32_t>(info);
            } else if (info != 0) {
                throw batchlas::convergence_error("gesvd_vendor (NETLIB): LAPACKE gesvd failed");
            }
        }

        return ctx.create_event_after_external_work();
    }

    template <Backend B, typename T>
    size_t gesvd_vendor_buffer_size(Queue& /*ctx*/,
                                    const MatrixView<T, MatrixFormat::Dense>& /*A*/,
                                    Span<typename base_type<T>::type> /*singular_values*/,
                                    const MatrixView<T, MatrixFormat::Dense>& /*U*/,
                                    const MatrixView<T, MatrixFormat::Dense>& /*Vh*/,
                                    SvdVectors /*jobu*/,
                                    SvdVectors /*jobvh*/) {
        // LAPACKE path uses no user-provided workspace.
        return static_cast<size_t>(0);
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
        // I4: the host loop reads the (possibly device-only, host-addressable here) mask itself.
        const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("getrs_vendor", A, B));
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
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.getrs", [=] {
            int nrhs = B_view.cols();
            for (int i = 0; i < A_view.batch_size(); ++i) {
                if (mask != nullptr && mask[i] == 0) continue;
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
        return BumpAllocator::allocation_size<int>(ctx, A.rows() * A.batch_size());
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    Event getrf_vendor(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<int64_t> pivots,
                Span<std::byte> workspace,
                Span<int32_t> info_out) {
        // I4: the host loop and the pivot copy skip an inactive item (matrix, pivots, info untouched).
        const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("getrf_vendor", A));
        BumpAllocator pool(workspace);
        auto A_view = A;
        auto piv = pivots;
        // An m x n item (B6): LAPACKE factors the whole matrix, min(m, n) pivots per item, packed.
        const int m = A_view.rows(), n = A_view.cols(), mn = std::min(m, n);
        const int batch = A_view.batch_size();
        auto piv_i32 = pool.allocate<int>(ctx, mn * batch);
        // See potrf above: the LAPACKE return was discarded, so a singular item
        // was indistinguishable from a factorised one (issue #73).
        auto info = info_out;
        const bool want_info = info.size() >= static_cast<size_t>(batch);

        Event getrf_event = ::batchlas::detail::submit_host_task<T>(ctx, "netlib.getrf", [=] {
            for (int i = 0; i < batch; ++i) {
                if (mask != nullptr && mask[i] == 0) continue;
                auto st = call_backend_nh_r<T, BackendLibrary::LAPACKE>(
                    LAPACKE_sgetrf, LAPACKE_dgetrf, LAPACKE_cgetrf, LAPACKE_zgetrf,
                    Layout::ColMajor,
                    m,
                    n,
                    A_view[i].data_ptr(),
                    A_view.ld(),
                    piv_i32.data() + i * mn);
                if (want_info) info[i] = static_cast<int32_t>(st);
            }
        });
        ctx.enqueue(getrf_event);

        EventImpl piv_impl = ctx->submit([&](sycl::handler& h) {
            auto piv_out = piv.as_span<int64_t>();
            auto piv_in = piv_i32;
            h.depends_on(static_cast<sycl::event>(*ctx.get_event()));
            h.parallel_for(sycl::range<1>(static_cast<size_t>(mn * batch)), [=](sycl::id<1> idx) {
                if (mask != nullptr && mask[static_cast<int>(idx[0]) / mn] == 0) return;
                piv_out[static_cast<int>(idx[0])] = static_cast<int64_t>(piv_in[static_cast<int>(idx[0])]);
            });
        });
        Event piv_event(std::move(piv_impl));
        ctx.enqueue(piv_event);
        return ctx.get_event();
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    size_t getrf_vendor_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        return BumpAllocator::allocation_size<int>(ctx, std::min(A.rows(), A.cols()) * A.batch_size());
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
        // I4: the copy and the host loop skip an inactive item, whose C and info stay untouched.
        const auto mask_span = ::batchlas::detail::call_mask("getri_vendor", A, C);
        const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(mask_span);
        BumpAllocator pool(workspace);
        auto A_view = A.with_batch_mask(mask_span);
        auto C_view = C.with_batch_mask(mask_span);
        auto piv = pivots;
        const int n = A_view.rows();
        const int batch = A_view.batch_size();
        auto piv_i32 = pool.allocate<int>(ctx, n * batch);
        // See potrf above: the LAPACKE return was discarded, so an item with no
        // inverse was indistinguishable from one that inverted (issue #73).
        auto info = info_out;
        const bool want_info = info.size() >= static_cast<size_t>(batch);
        // ld- and stride-aware; submit_host_task waits for it before the LAPACKE calls.
        (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, C_view, A_view);
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.getri", [=] {
            auto piv_in = piv.as_span<int64_t>();
            for (int b = 0; b < batch; ++b) {
                if (mask != nullptr && mask[b] == 0) continue;
                auto Cb = C_view[b];
                for (int i = 0; i < n; ++i) {
                    piv_i32[b * n + i] = static_cast<int>(piv_in[b * n + i]);
                }

                auto st = call_backend_nh_r<T, BackendLibrary::LAPACKE>(
                    LAPACKE_sgetri, LAPACKE_dgetri, LAPACKE_cgetri, LAPACKE_zgetri,
                    Layout::ColMajor,
                    n,
                    Cb.data_ptr(),
                    Cb.ld(),
                    piv_i32.data() + b * n);
                if (want_info) info[b] = static_cast<int32_t>(st);
            }
        });
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    size_t getri_vendor_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        return BumpAllocator::allocation_size<int>(ctx, A.rows() * A.batch_size());
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    Event geqrf_vendor(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<T> tau,
                Span<std::byte> workspace) {
        // I4: the host loop reads the (possibly device-only, host-addressable here) mask itself;
        // an inactive item's A and tau are untouched.
        const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("geqrf_vendor", A));
        static_cast<void>(workspace);
        auto A_view = A;
        auto tau_view = tau;
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.geqrf", [=] {
            int m = A_view.rows();
            int n = A_view.cols();
            for (int i = 0; i < A_view.batch_size(); ++i) {
                if (mask != nullptr && mask[i] == 0) continue;
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
    Event orgqr_vendor(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<T> tau,
                Span<std::byte> workspace) {
        // I4, as geqrf_vendor: an inactive item's A is untouched.
        const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("orgqr_vendor", A));
        static_cast<void>(workspace);
        auto A_view = A;
        auto tau_view = tau;
        return ::batchlas::detail::submit_host_task<T>(ctx, "netlib.orgqr", [=] {
            int m = A_view.rows();
            int n = A_view.cols();
            int k = std::min(m, n);
            for (int i = 0; i < A_view.batch_size(); ++i) {
                if (mask != nullptr && mask[i] == 0) continue;
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
    Event ormqr_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      Side side,
                      Transpose trans,
                      Span<T> tau,
                      Span<std::byte> workspace) {
        // I4, as geqrf_vendor: an inactive item's C is untouched.
        const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("ormqr_vendor", A, C));
        auto A_view = A;
        auto C_view = C;
        auto tau_view = tau;
        return ::batchlas::detail::submit_host_task<T>(ctx, "lapacke.ormqr_vendor", [=] {
            static_cast<void>(workspace);
            int m = C_view.rows();
            int n = C_view.cols();
            int k = std::min(A_view.rows(), A_view.cols());
            for (int i = 0; i < A_view.batch_size(); ++i) {
                if (mask != nullptr && mask[i] == 0) continue;
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
    }

    template <Backend B, typename T>
    size_t ormqr_vendor_buffer_size(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    Side side,
                                    Transpose trans,
                                    Span<T> tau) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        static_cast<void>(C);
        static_cast<void>(side);
        static_cast<void>(trans);
        static_cast<void>(tau);
        return static_cast<size_t>(0);
    }

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    size_t potrf_vendor_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& descrA,
                             Uplo uplo) {
        static_cast<void>(ctx);
        static_cast<void>(descrA);
        static_cast<void>(uplo);
        return 0;
    }

    } // namespace backend


    // ONLY `backend::*_vendor` rows: a public-op row would duplicate src/ops/.
    // evidence: docs/design/runtime-internals.md#runtime-internals-vendor-tus-instantiate-only-vendor-symbols
    #define B_ Backend::NETLIB

    // Local shim: there is no _BACKEND_FORMAT_OP for backend::spmm_vendor.
    #define NETLIB_FORMAT_BACKEND_OP(B, fp, F, OP) \
        BATCHLAS_INSTANTIATE(sig::OP<BATCHLAS_UNPAREN fp BATCHLAS_COMMA F>, backend::OP, B, BATCHLAS_UNPAREN fp, F)

    #define NETLIB_OPS(B, fp) \
        NETLIB_FORMAT_BACKEND_OP(B, fp, MatrixFormat::CSR, spmm_vendor) \
        NETLIB_FORMAT_BACKEND_OP(B, fp, MatrixFormat::CSR, spmm_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, gemm_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, gemv_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, trsm_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, trmm_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, geqrf_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, geqrf_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getrs_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getrs_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getrf_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getrf_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getri_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getri_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, ormqr_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, ormqr_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, orgqr_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, orgqr_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, potrf_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, potrf_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, syev_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, syev_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, gesvd_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, gesvd_vendor_buffer_size)

    // hemm/herk/her2k are complex-only, so they get their own table.
    #define NETLIB_SYMMETRIC_OPS(B, fp) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, symm_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, syrk_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, syr2k_vendor)

    #define NETLIB_COMPLEX_OPS(B, fp) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, hemm_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, herk_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, her2k_vendor)

    // Instantiate for the floating-point types of interest.
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(NETLIB_OPS, B_)
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(NETLIB_SYMMETRIC_OPS, B_)
    BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(NETLIB_COMPLEX_OPS, B_)

    #undef NETLIB_OPS
    #undef NETLIB_SYMMETRIC_OPS
    #undef NETLIB_COMPLEX_OPS
    #undef NETLIB_FORMAT_BACKEND_OP
    #undef B_
}