#include <batchlas/blas/linalg.hh>
#include "../linalg-impl.hh"
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include "../queue.hh"
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
#include <batchlas/blas/dispatch/op.hh>

#include "gemm_variant.hh"
#include "../util/template-instantiations.hh"

namespace batchlas{

    namespace detail {

    // ---------------------------------------------------------------------
    // Host BLAS double-precision health check.
    //
    // Some OpenBLAS builds ship a CPU-dispatch kernel that computes dgemm
    // wrongly on the machine auto-detection picks it for; the known case is
    // OpenBLAS 0.3.20's "Cooperlake" kernel on recent Intel parts, off by
    // O(1)-O(100) at some sizes while sgemm is fine. Everything layered on top
    // silently inherits the garbage.
    //
    // cmake/BatchLASBlasHealthCheck.cmake detects this at configure time and
    // records the OPENBLAS_CORETYPE value that repairs it, but a configure-time
    // answer is stale by construction: an install tree is routinely consumed on
    // a machine other than the one that built it. So the recorded value is
    // compiled in as BATCHLAS_REQUIRED_OPENBLAS_CORETYPE and we re-run a cheap
    // version of the same probe here, once, on first double-precision use.
    //
    // Deliberately no setenv(): OpenBLAS reads OPENBLAS_CORETYPE in its library
    // constructor, which has already run by the time any BatchLAS code
    // executes, so setting it from here would look like it worked and change
    // nothing. Only the environment of the process before it starts can fix it.
    // ---------------------------------------------------------------------

#ifndef BATCHLAS_REQUIRED_OPENBLAS_CORETYPE
#define BATCHLAS_REQUIRED_OPENBLAS_CORETYPE ""
#endif

    struct HostBlasDoubleHealth {
        bool ok = true;
        int bad_size = 0;
        double worst_error = 0.0;
    };

    // off | warn (default) | error
    inline const char* host_blas_health_mode() {
        const char* mode = std::getenv("BATCHLAS_BLAS_HEALTH");
        return mode ? mode : "warn";
    }

    // Port of the configure-time probe in cmake/BatchLASBlasHealthCheck.cmake.
    // The naive reference is evaluated on a strided sample of C (at most 32x32
    // entries per size) so the whole thing costs a few milliseconds: a broken
    // kernel is wrong by O(1)+ across the result, not in one isolated entry.
    // Sizes are the ones that expose the known defect (n=64 and n=256 happen to
    // be correct there, so a single small size proves nothing).
    inline HostBlasDoubleHealth probe_host_dgemm() {
        HostBlasDoubleHealth health;
        static const int sizes[] = {128, 200, 512};
        for (int n : sizes) {
            const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
            std::vector<double> a(nn), b(nn), c(nn, 0.0);
            for (int col = 0; col < n; ++col) {
                for (int row = 0; row < n; ++row) {
                    a[row + static_cast<size_t>(col) * n] = std::sin(0.5 * (row + 1) * (col + 1));
                    b[row + static_cast<size_t>(col) * n] = std::cos(0.25 * (row + 1) * (col + 2));
                }
            }

            call_backend_nh<double, BackendLibrary::CBLAS>(
                cblas_sgemm, cblas_dgemm, cblas_cgemm, cblas_zgemm,
                Layout::ColMajor, Transpose::NoTrans, Transpose::NoTrans,
                n, n, n,
                1.0,
                a.data(), n,
                b.data(), n,
                0.0,
                c.data(), n);

            const int step = std::max(1, n / 32);
            double worst = 0.0;
            for (int col = 0; col < n; col += step) {
                for (int row = 0; row < n; row += step) {
                    double want = 0.0;
                    for (int i = 0; i < n; ++i) {
                        want += a[row + static_cast<size_t>(i) * n] * b[i + static_cast<size_t>(col) * n];
                    }
                    const double diff = std::abs(want - c[row + static_cast<size_t>(col) * n]);
                    if (diff > worst) worst = diff;
                }
            }
            // Rounding differences are ~1e-13 here; a broken kernel is off by O(1)+.
            if (worst > 1e-6) {
                health.ok = false;
                health.bad_size = n;
                health.worst_error = worst;
                return health;
            }
        }
        return health;
    }

    inline const HostBlasDoubleHealth& host_blas_double_health() {
        static const HostBlasDoubleHealth health = [] {
            if (std::strcmp(host_blas_health_mode(), "off") == 0) {
                return HostBlasDoubleHealth{};
            }
            return probe_host_dgemm();
        }();
        return health;
    }

    inline std::string host_blas_double_health_message(const HostBlasDoubleHealth& health) {
        const std::string required = BATCHLAS_REQUIRED_OPENBLAS_CORETYPE;
        const char* current_env = std::getenv("OPENBLAS_CORETYPE");
        const std::string current = current_env ? current_env : "";

        std::string msg =
            "BatchLAS: the host BLAS computes dgemm INCORRECTLY on this machine.\n"
            "  probe: max abs error " + std::to_string(health.worst_error) +
            " at n=" + std::to_string(health.bad_size) + " (rounding alone is ~1e-13)\n"
            "  OPENBLAS_CORETYPE is currently " +
            (current.empty() ? std::string("unset") : ("\"" + current + "\"")) + "\n";

        if (!required.empty()) {
            msg += "  this build detected that OPENBLAS_CORETYPE=" + required + " repairs it\n";
        }
        msg += "Double-precision results from the host/NETLIB backend cannot be trusted.\n"
               "OpenBLAS reads OPENBLAS_CORETYPE in its library constructor, before main(),\n"
               "so it must be set in the environment before the process starts - BatchLAS\n"
               "cannot set it for you.\n";

        if (!required.empty() && required != current) {
            msg += "    export OPENBLAS_CORETYPE=" + required + "\n";
        } else if (!required.empty()) {
            msg += "OPENBLAS_CORETYPE=" + required + " is already set and dgemm is still wrong: "
                   "this host BLAS is unusable for double precision, upgrade or replace it.\n";
        } else {
            msg += "No working value was recorded at build time. Try, in order:\n"
                   "    export OPENBLAS_CORETYPE=SKYLAKEX   (then HASWELL, SANDYBRIDGE, NEHALEM, PRESCOTT)\n";
        }
        msg += "Set BATCHLAS_BLAS_HEALTH=error to turn this into an exception, "
               "or BATCHLAS_BLAS_HEALTH=off to skip the check.\n";
        return msg;
    }

    inline void check_host_blas_double_health() {
        const HostBlasDoubleHealth& health = host_blas_double_health();
        if (health.ok) return;

        const std::string msg = host_blas_double_health_message(health);
        if (std::strcmp(host_blas_health_mode(), "error") == 0) {
            throw std::runtime_error(msg);
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

    // No-op for anything that is not double precision: a broken dgemm kernel
    // does not make single precision wrong, and a float-only user should not be
    // told to change their environment.
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
        static_cast<void>(workspace); // no workspace needed for CPU implementation
        auto A_view = A;
        auto B_view = B;
        auto C_view = C;
        // No double-precision guard here: the CSR path below is hand-written and
        // never calls into the host BLAS, so a broken dgemm cannot affect it.
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
    
    // The netlib gemm is the vendor implementation, so it moves into
    // `backend` under its vendor name rather than being deleted: unlike
    // cublas.cc and rocblas.cc, this TU had no separate gemm_vendor to forward
    // to -- its public `gemm` WAS the CBLAS call.
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
        static_cast<void>(precision);
        auto A_view = descrA;
        auto B_view = descrB;
        auto C_view = descrC;
        return detail::submit_host_task<T>(ctx, "netlib.gemm", [=] {
            if (!backend::gemm_batch_dimensions_compatible(A_view, B_view, C_view, transA, transB)) {
                throw std::runtime_error("GEMM: incompatible matrix dimensions");
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
                                C_i.at(row, col) *= beta;
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
        auto A_view = A;
        auto X_view = X;
        auto Y_view = Y;
        return detail::submit_host_task<T>(ctx, "netlib.gemv", [=] {
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
        // Parameter order matches backend::trsm_vendor as cuBLAS defines it:
        // alpha LAST, unlike the public trsm, which takes it third. The two
        // orders coexisted for as long as each TU declared its own public trsm;
        // now that one declaration serves every backend, they have to agree.

        auto A_view = descrA;
        auto B_view = descrB;
        return detail::submit_host_task<T>(ctx, "netlib.trsm", [=] {
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

    } // namespace backend

    namespace backend {

    template <Backend B, RealScalar T>
    Event symm_vendor(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const MatrixView<T, MatrixFormat::Dense>& Bmat,
               const MatrixView<T, MatrixFormat::Dense>& Cmat,
               T alpha,
               T beta,
               Side side,
               Uplo uplo) {
        auto A_view = A;
        auto B_view = Bmat;
        auto C_view = Cmat;
        return detail::submit_host_task<T>(ctx, "netlib.symm", [=] {
            if (A_view.rows() != A_view.cols()) {
                throw std::runtime_error("SYMM: A must be square");
            }
            if (A_view.batch_size() != B_view.batch_size() || A_view.batch_size() != C_view.batch_size()) {
                throw std::runtime_error("SYMM: batch size mismatch");
            }

            const int m = C_view.rows();
            const int n = C_view.cols();
            const int expected_a = side == Side::Left ? B_view.rows() : B_view.cols();
            if (A_view.rows() != expected_a || B_view.rows() != m || B_view.cols() != n) {
                throw std::runtime_error("SYMM: incompatible matrix dimensions");
            }

            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& B_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                if constexpr (std::is_same_v<T, float>) {
                    cblas_ssymm(CblasColMajor,
                                enum_convert<BackendLibrary::CBLAS>(side),
                                enum_convert<BackendLibrary::CBLAS>(uplo),
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
                } else if constexpr (std::is_same_v<T, double>) {
                    cblas_dsymm(CblasColMajor,
                                enum_convert<BackendLibrary::CBLAS>(side),
                                enum_convert<BackendLibrary::CBLAS>(uplo),
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
                }
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
        auto A_view = A;
        auto B_view = Bmat;
        auto C_view = Cmat;
        return detail::submit_host_task<T>(ctx, "netlib.hemm", [=] {
            if (A_view.rows() != A_view.cols()) {
                throw std::runtime_error("HEMM: A must be square");
            }
            if (A_view.batch_size() != B_view.batch_size() || A_view.batch_size() != C_view.batch_size()) {
                throw std::runtime_error("HEMM: batch size mismatch");
            }

            const int m = C_view.rows();
            const int n = C_view.cols();
            const int expected_a = side == Side::Left ? B_view.rows() : B_view.cols();
            if (A_view.rows() != expected_a || B_view.rows() != m || B_view.cols() != n) {
                throw std::runtime_error("HEMM: incompatible matrix dimensions");
            }

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
        auto A_view = A;
        auto C_view = Cmat;
        return detail::submit_host_task<T>(ctx, "netlib.herk", [=] {
            if (C_view.rows() != C_view.cols()) {
                throw std::runtime_error("HERK: C must be square");
            }
            if (A_view.batch_size() != C_view.batch_size()) {
                throw std::runtime_error("HERK: batch size mismatch");
            }
            // Transpose::Trans would ask for A * A^T, which is
            // complex-symmetric rather than Hermitian; that operation is
            // syrk's, and BLAS does not spell it here.
            if (transA != Transpose::NoTrans && transA != Transpose::ConjTrans) {
                throw std::runtime_error("HERK: transA must be NoTrans or ConjTrans");
            }

            const int n = C_view.rows();
            const int k = transA == Transpose::NoTrans ? A_view.cols() : A_view.rows();
            const int expected_n = transA == Transpose::NoTrans ? A_view.rows() : A_view.cols();
            if (expected_n != n || k <= 0) {
                throw std::runtime_error("HERK: incompatible matrix dimensions");
            }

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
        auto A_view = A;
        auto B_view = Bmat;
        auto C_view = Cmat;
        return detail::submit_host_task<T>(ctx, "netlib.her2k", [=] {
            if (C_view.rows() != C_view.cols()) {
                throw std::runtime_error("HER2K: C must be square");
            }
            if (A_view.batch_size() != B_view.batch_size() ||
                A_view.batch_size() != C_view.batch_size()) {
                throw std::runtime_error("HER2K: batch size mismatch");
            }
            if (transA != Transpose::NoTrans && transA != Transpose::ConjTrans) {
                throw std::runtime_error("HER2K: transA must be NoTrans or ConjTrans");
            }

            const int n = C_view.rows();
            const bool no_trans = transA == Transpose::NoTrans;
            const int k = no_trans ? A_view.cols() : A_view.rows();
            const int expected_n = no_trans ? A_view.rows() : A_view.cols();
            const int b_n = no_trans ? B_view.rows() : B_view.cols();
            const int b_k = no_trans ? B_view.cols() : B_view.rows();
            if (expected_n != n || b_n != n || b_k != k || k <= 0) {
                throw std::runtime_error("HER2K: incompatible matrix dimensions");
            }

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

    template <Backend B, RealScalar T>
    Event syrk_vendor(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& A,
               const MatrixView<T, MatrixFormat::Dense>& Cmat,
               T alpha,
               T beta,
               Uplo uplo,
               Transpose transA) {
        auto A_view = A;
        auto C_view = Cmat;
        return detail::submit_host_task<T>(ctx, "netlib.syrk", [=] {
            if (C_view.rows() != C_view.cols()) {
                throw std::runtime_error("SYRK: C must be square");
            }
            if (A_view.batch_size() != C_view.batch_size()) {
                throw std::runtime_error("SYRK: batch size mismatch");
            }

            const int n = C_view.rows();
            const int k = transA == Transpose::NoTrans ? A_view.cols() : A_view.rows();
            const int expected_n = transA == Transpose::NoTrans ? A_view.rows() : A_view.cols();
            if (expected_n != n || k <= 0) {
                throw std::runtime_error("SYRK: incompatible matrix dimensions");
            }

            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                if constexpr (std::is_same_v<T, float>) {
                    cblas_ssyrk(CblasColMajor,
                                enum_convert<BackendLibrary::CBLAS>(uplo),
                                enum_convert<BackendLibrary::CBLAS>(transA),
                                n,
                                k,
                                alpha,
                                A_i.data_ptr(),
                                A_i.ld(),
                                beta,
                                C_i.data_ptr(),
                                C_i.ld());
                } else if constexpr (std::is_same_v<T, double>) {
                    cblas_dsyrk(CblasColMajor,
                                enum_convert<BackendLibrary::CBLAS>(uplo),
                                enum_convert<BackendLibrary::CBLAS>(transA),
                                n,
                                k,
                                alpha,
                                A_i.data_ptr(),
                                A_i.ld(),
                                beta,
                                C_i.data_ptr(),
                                C_i.ld());
                }
            };

            for (int batch = 0; batch < A_view.batch_size(); ++batch) {
                launch_single(A_view[batch], C_view[batch]);
            }
        });
    }

    } // namespace backend

    namespace backend {

    template <Backend B, RealScalar T>
    Event syr2k_vendor(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& Bmat,
                const MatrixView<T, MatrixFormat::Dense>& Cmat,
                T alpha,
                T beta,
                Uplo uplo,
                Transpose transA) {
        auto A_view = A;
        auto B_view = Bmat;
        auto C_view = Cmat;
        return detail::submit_host_task<T>(ctx, "netlib.syr2k", [=] {
            if (C_view.rows() != C_view.cols()) {
                throw std::runtime_error("SYR2K: C must be square");
            }
            if (A_view.batch_size() != B_view.batch_size() || A_view.batch_size() != C_view.batch_size()) {
                throw std::runtime_error("SYR2K: batch size mismatch");
            }

            const int n = C_view.rows();
            const int expected_n = transA == Transpose::NoTrans ? A_view.rows() : A_view.cols();
            const int expected_b_n = transA == Transpose::NoTrans ? B_view.rows() : B_view.cols();
            const int k = transA == Transpose::NoTrans ? A_view.cols() : A_view.rows();
            const int b_k = transA == Transpose::NoTrans ? B_view.cols() : B_view.rows();
            if (expected_n != n || expected_b_n != n || b_k != k || k <= 0) {
                throw std::runtime_error("SYR2K: incompatible matrix dimensions");
            }

            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& B_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                if constexpr (std::is_same_v<T, float>) {
                    cblas_ssyr2k(CblasColMajor,
                                 enum_convert<BackendLibrary::CBLAS>(uplo),
                                 enum_convert<BackendLibrary::CBLAS>(transA),
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
                } else if constexpr (std::is_same_v<T, double>) {
                    cblas_dsyr2k(CblasColMajor,
                                 enum_convert<BackendLibrary::CBLAS>(uplo),
                                 enum_convert<BackendLibrary::CBLAS>(transA),
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
                }
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
        auto A_view = A;
        auto B_view = Bmat;
        auto C_view = Cmat;
        return detail::submit_host_task<T>(ctx, "netlib.trmm", [=] {
            if (A_view.rows() != A_view.cols()) {
                throw std::runtime_error("TRMM: A must be square");
            }
            if (A_view.batch_size() != B_view.batch_size() || A_view.batch_size() != C_view.batch_size()) {
                throw std::runtime_error("TRMM: batch size mismatch");
            }

            const int m = C_view.rows();
            const int n = C_view.cols();
            const int expected_dim = side == Side::Left ? m : n;
            if (A_view.rows() != expected_dim || B_view.rows() != m || B_view.cols() != n) {
                throw std::runtime_error("TRMM: incompatible matrix dimensions");
            }

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

    template <Backend B, typename T>
    Event potrf_vendor(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& descrA,
                    Uplo uplo,
                    Span<std::byte> workspace,
                    Span<int32_t> info_out) {
        static_cast<void>(workspace);
        auto A_view = descrA;
        // LAPACKE reports failure only through its return value, and this loop
        // used to drop it on the floor (issue #73): an indefinite item came back
        // looking exactly like a factorised one. There is no pool fallback to
        // pick here -- unlike the vendor backends netlib needs no status array of
        // its own -- so an empty span simply means "do not record".
        auto info = info_out;
        const bool want_info = info.size() >= static_cast<size_t>(descrA.batch_size());
        return detail::submit_host_task<T>(ctx, "netlib.potrf", [=] {
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
                      Span<std::byte> /*workspace*/) {
        auto A_view = descrA;
        auto eig = eigenvalues;
        return op_external("lapacke.syev", [&, A_view, eig, jobtype, uplo] {
            return detail::submit_host_task<T>(ctx, "lapacke.syev", [=] {
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

    // Moved verbatim from include/batchlas/blas/functions/gesvd.hh, which used to *define*
    // the primary template (and therefore made a cuSOLVER definition a
    // redefinition error). Semantics are unchanged, including the synchronous
    // ctx.wait() -- LAPACKE ?gesvd needs A on the host and this path is the
    // reference implementation, not a fast one.
    template <Backend B, typename T>
    Event gesvd_vendor(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       Span<typename base_type<T>::type> singular_values,
                       const MatrixView<T, MatrixFormat::Dense>& U,
                       const MatrixView<T, MatrixFormat::Dense>& Vh,
                       SvdVectors jobu,
                       SvdVectors jobvh,
                       Span<std::byte> workspace) {
        static_cast<void>(workspace);

        // This path calls LAPACKE directly rather than going through
        // submit_host_task, so it has to run the double-precision health check
        // itself; otherwise a broken host dgemm is silent here.
        detail::host_blas_double_guard<T>();

        if (A.batch_size() < 1 || A.rows() < 1 || A.cols() < 1) {
            throw std::invalid_argument("gesvd_vendor (NETLIB): invalid matrix shape or batch size");
        }

        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        const int k = std::min(m, n);
        const int batch = static_cast<int>(A.batch_size());
        const std::size_t need_s = static_cast<std::size_t>(k) * static_cast<std::size_t>(batch);
        if (singular_values.size() < need_s) {
            throw std::invalid_argument("gesvd_vendor (NETLIB): singular_values span too small");
        }

        // NETLIB implements Thin rather than refusing it. gesvd_dispatch pins
        // this backend to Vendor unconditionally, so refusing would leave the
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
                throw std::invalid_argument("gesvd_vendor (NETLIB): U must be (m x " +
                                            std::to_string(want_cols) + ") with matching batch");
            }
        }
        if (jobvh != SvdVectors::None) {
            const int want_rows = static_cast<int>(svd_vh_rows(jobvh, n, k));
            if (Vh.rows() != want_rows || Vh.cols() != n || Vh.batch_size() != batch) {
                throw std::invalid_argument("gesvd_vendor (NETLIB): Vh must be (" +
                                            std::to_string(want_rows) + " x n) with matching batch");
            }
        }

        ctx.wait();

        std::vector<typename base_type<T>::type> superb(static_cast<std::size_t>(std::max(0, k - 1)));
        auto& A_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(A);
        for (int b = 0; b < batch; ++b) {
            auto Ab = A_mut.batch_item(b);
            auto Ub = U.batch_item(b);
            auto Vhb = Vh.batch_item(b);
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
                throw std::runtime_error("gesvd_vendor (NETLIB): unsupported scalar type");
            }

            if (info != 0) {
                throw std::runtime_error("gesvd_vendor (NETLIB): LAPACKE gesvd failed");
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
        return op_external("lapacke.gesvd_buffer_size", [&] { return static_cast<size_t>(0); });
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
        return detail::submit_host_task<T>(ctx, "netlib.getrs", [=] {
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
        BumpAllocator pool(workspace);
        auto A_view = A;
        auto piv = pivots;
        const int n = A_view.rows();
        const int batch = A_view.batch_size();
        auto piv_i32 = pool.allocate<int>(ctx, n * batch);
        // See potrf above: the LAPACKE return was discarded, so a singular item
        // was indistinguishable from a factorised one (issue #73).
        auto info = info_out;
        const bool want_info = info.size() >= static_cast<size_t>(batch);

        Event getrf_event = detail::submit_host_task<T>(ctx, "netlib.getrf", [=] {
            for (int i = 0; i < batch; ++i) {
                auto st = call_backend_nh_r<T, BackendLibrary::LAPACKE>(
                    LAPACKE_sgetrf, LAPACKE_dgetrf, LAPACKE_cgetrf, LAPACKE_zgetrf,
                    Layout::ColMajor,
                    n,
                    n,
                    A_view[i].data_ptr(),
                    A_view.ld(),
                    piv_i32.data() + i * n);
                if (want_info) info[i] = static_cast<int32_t>(st);
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

    } // namespace backend

    namespace backend {

    template <Backend B, typename T>
    size_t getrf_vendor_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A) {
        static_cast<void>(ctx);
        static_cast<void>(A);
        return BumpAllocator::allocation_size<int>(ctx, A.rows() * A.batch_size());
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
        BumpAllocator pool(workspace);
        auto A_view = A;
        auto C_view = C;
        auto piv = pivots;
        const int n = A_view.rows();
        const int batch = A_view.batch_size();
        auto piv_i32 = pool.allocate<int>(ctx, n * batch);
        // See potrf above: the LAPACKE return was discarded, so an item with no
        // inverse was indistinguishable from one that inverted (issue #73).
        auto info = info_out;
        const bool want_info = info.size() >= static_cast<size_t>(batch);
        return detail::submit_host_task<T>(ctx, "netlib.getri", [=] {
            auto piv_in = piv.as_span<int64_t>();
            for (int b = 0; b < batch; ++b) {
                auto Ab = A_view[b];
                auto Cb = C_view[b];
                std::copy(Ab.data_ptr(), Ab.data_ptr() + n * n, Cb.data_ptr());
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
        static_cast<void>(workspace);
        auto A_view = A;
        auto tau_view = tau;
        return detail::submit_host_task<T>(ctx, "netlib.geqrf", [=] {
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
        static_cast<void>(workspace);
        auto A_view = A;
        auto tau_view = tau;
        return detail::submit_host_task<T>(ctx, "netlib.orgqr", [=] {
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
        auto A_view = A;
        auto C_view = C;
        auto tau_view = tau;
        return op_external("lapacke.ormqr_vendor", [&, A_view, C_view, tau_view, side, trans] {
            return detail::submit_host_task<T>(ctx, "lapacke.ormqr_vendor", [=] {
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


    // Explicit instantiations. Signatures live in the `sig` namespace beside each
    // public declaration (include/batchlas/blas/functions/*.hh), so changing one is a single
    // header edit rather than one edit per backend TU.
    #define B_ Backend::NETLIB

    #define SPMM_INSTANTIATE(fp, F)             BATCHLAS_INSTANTIATE(sig::spmm_vendor<fp BATCHLAS_COMMA F>, backend::spmm_vendor, B_, fp, F)
    #define SPMM_BUFFER_SIZE_INSTANTIATE(fp, F) BATCHLAS_INSTANTIATE(sig::spmm_vendor_buffer_size<fp BATCHLAS_COMMA F>, backend::spmm_vendor_buffer_size, B_, fp, F)
    #define GEMM_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::gemm_vendor<fp>, backend::gemm_vendor, B_, fp)
    #define GEMV_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::gemv_vendor<fp>, backend::gemv_vendor, B_, fp)
    #define TRSM_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::trsm_vendor<fp>, backend::trsm_vendor, B_, fp)
    #define SYMM_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::symm_vendor<fp>, backend::symm_vendor, B_, fp)
    #define HEMM_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::hemm_vendor<fp>, backend::hemm_vendor, B_, fp)
    #define SYRK_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::syrk_vendor<fp>, backend::syrk_vendor, B_, fp)
    #define HERK_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::herk_vendor<fp>, backend::herk_vendor, B_, fp)
    #define HER2K_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::her2k_vendor<fp>, backend::her2k_vendor, B_, fp)
    #define SYR2K_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::syr2k_vendor<fp>, backend::syr2k_vendor, B_, fp)
    #define TRMM_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::trmm_vendor<fp>, backend::trmm_vendor, B_, fp)
    #define GEQRF_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::geqrf_vendor<fp>, backend::geqrf_vendor, B_, fp)
    #define GEQRF_BUFFER_SIZE_INSTANTIATE(fp)   BATCHLAS_INSTANTIATE(sig::geqrf_vendor_buffer_size<fp>, backend::geqrf_vendor_buffer_size, B_, fp)
    #define ORGQR_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::orgqr_vendor<fp>, backend::orgqr_vendor, B_, fp)
    #define ORGQR_BUFFER_SIZE_INSTANTIATE(fp)   BATCHLAS_INSTANTIATE(sig::orgqr_vendor_buffer_size<fp>, backend::orgqr_vendor_buffer_size, B_, fp)
    #define GETRS_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::getrs_vendor<fp>, backend::getrs_vendor, B_, fp)
    #define GETRS_BUFFER_SIZE_INSTANTIATE(fp)   BATCHLAS_INSTANTIATE(sig::getrs_vendor_buffer_size<fp>, backend::getrs_vendor_buffer_size, B_, fp)
    #define GETRF_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::getrf_vendor<fp>, backend::getrf_vendor, B_, fp)
    #define GETRF_BUFFER_SIZE_INSTANTIATE(fp)   BATCHLAS_INSTANTIATE(sig::getrf_vendor_buffer_size<fp>, backend::getrf_vendor_buffer_size, B_, fp)
    #define GETRI_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::getri_vendor<fp>, backend::getri_vendor, B_, fp)
    #define GETRI_BUFFER_SIZE_INSTANTIATE(fp)   BATCHLAS_INSTANTIATE(sig::getri_vendor_buffer_size<fp>, backend::getri_vendor_buffer_size, B_, fp)
    #define ORMQR_VENDOR_INSTANTIATE(fp)        BATCHLAS_INSTANTIATE(sig::ormqr_vendor<fp>, backend::ormqr_vendor, B_, fp)
    #define ORMQR_VENDOR_BUFFER_SIZE_INSTANTIATE(fp) BATCHLAS_INSTANTIATE(sig::ormqr_vendor_buffer_size<fp>, backend::ormqr_vendor_buffer_size, B_, fp)
    #define POTRF_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::potrf_vendor<fp>, backend::potrf_vendor, B_, fp)
    #define POTRF_BUFFER_SIZE_INSTANTIATE(fp)   BATCHLAS_INSTANTIATE(sig::potrf_vendor_buffer_size<fp>, backend::potrf_vendor_buffer_size, B_, fp)
    #define SYEV_VENDOR_INSTANTIATE(fp)         BATCHLAS_INSTANTIATE(sig::syev_vendor<fp>, backend::syev_vendor, B_, fp)
    #define SYEV_VENDOR_BUFFER_SIZE_INSTANTIATE(fp) BATCHLAS_INSTANTIATE(sig::syev_vendor_buffer_size<fp>, backend::syev_vendor_buffer_size, B_, fp)
    #define GESVD_VENDOR_INSTANTIATE(fp)        BATCHLAS_INSTANTIATE(sig::gesvd_vendor<fp>, backend::gesvd_vendor, B_, fp)
    #define GESVD_VENDOR_BUFFER_SIZE_INSTANTIATE(fp) BATCHLAS_INSTANTIATE(sig::gesvd_vendor_buffer_size<fp>, backend::gesvd_vendor_buffer_size, B_, fp)

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
        ORMQR_VENDOR_INSTANTIATE(fp) \
        ORMQR_VENDOR_BUFFER_SIZE_INSTANTIATE(fp) \
        ORGQR_INSTANTIATE(fp) \
        ORGQR_BUFFER_SIZE_INSTANTIATE(fp) \
        POTRF_INSTANTIATE(fp) \
        POTRF_BUFFER_SIZE_INSTANTIATE(fp) \
        SYEV_VENDOR_INSTANTIATE(fp) \
        SYEV_VENDOR_BUFFER_SIZE_INSTANTIATE(fp) \
        GESVD_VENDOR_INSTANTIATE(fp) \
        GESVD_VENDOR_BUFFER_SIZE_INSTANTIATE(fp)

    // Instantiate for the floating-point types of interest.
    BLAS_LEVEL3_INSTANTIATE(float)
    BLAS_LEVEL3_INSTANTIATE(double)
    BLAS_LEVEL3_INSTANTIATE(std::complex<float>)
    BLAS_LEVEL3_INSTANTIATE(std::complex<double>)
    TRMM_INSTANTIATE(float)
    TRMM_INSTANTIATE(double)
    TRMM_INSTANTIATE(std::complex<float>)
    TRMM_INSTANTIATE(std::complex<double>)
    SYMM_INSTANTIATE(float)
    SYMM_INSTANTIATE(double)
    HEMM_INSTANTIATE(std::complex<float>)
    HEMM_INSTANTIATE(std::complex<double>)
    HERK_INSTANTIATE(std::complex<float>)
    HERK_INSTANTIATE(std::complex<double>)
    HER2K_INSTANTIATE(std::complex<float>)
    HER2K_INSTANTIATE(std::complex<double>)
    SYRK_INSTANTIATE(float)
    SYRK_INSTANTIATE(double)
    SYR2K_INSTANTIATE(float)
    SYR2K_INSTANTIATE(double)

    #undef SPMM_INSTANTIATE
    #undef SPMM_BUFFER_SIZE_INSTANTIATE
    #undef GEMM_INSTANTIATE
    #undef GEMV_INSTANTIATE
    #undef SYMM_INSTANTIATE
    #undef HEMM_INSTANTIATE
    #undef HERK_INSTANTIATE
    #undef HER2K_INSTANTIATE
    #undef SYRK_INSTANTIATE
    #undef SYR2K_INSTANTIATE
    #undef TRSM_INSTANTIATE
    #undef TRMM_INSTANTIATE
    #undef GEQRF_INSTANTIATE
    #undef GEQRF_BUFFER_SIZE_INSTANTIATE
    #undef GETRS_INSTANTIATE
    #undef GETRS_BUFFER_SIZE_INSTANTIATE
    #undef GETRF_INSTANTIATE
    #undef GETRF_BUFFER_SIZE_INSTANTIATE
    #undef GETRI_INSTANTIATE
    #undef GETRI_BUFFER_SIZE_INSTANTIATE
    #undef ORMQR_VENDOR_INSTANTIATE
    #undef ORMQR_VENDOR_BUFFER_SIZE_INSTANTIATE
    #undef ORGQR_INSTANTIATE
    #undef ORGQR_BUFFER_SIZE_INSTANTIATE
    #undef POTRF_INSTANTIATE
    #undef POTRF_BUFFER_SIZE_INSTANTIATE
    #undef SYEV_VENDOR_INSTANTIATE
    #undef SYEV_VENDOR_BUFFER_SIZE_INSTANTIATE
    #undef GESVD_VENDOR_INSTANTIATE
    #undef GESVD_VENDOR_BUFFER_SIZE_INSTANTIATE
    #undef BLAS_LEVEL3_INSTANTIATE
    #undef B_
}