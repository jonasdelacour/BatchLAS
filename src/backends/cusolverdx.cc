#include "cusolverdx.hh"

#include "../linalg-impl.hh"
#include "../queue.hh"
#include <blas/dispatch/op.hh>
#include <batchlas/backend_config.h>
#include <complex>
#include <string>
#include <stdexcept>
#include <type_traits>

namespace batchlas::backend::cusolverdx {

// NOTE:
// This wrapper dispatches to cuSolverDx CUDA kernels when available/supported
// and falls back to existing cuSOLVER vendor paths for unsupported shapes,
// types, or toolchains.

namespace {

template <typename T>
inline void validate_htev_args(const VectorView<T>& d,
                               const VectorView<T>& e,
                               const VectorView<T>& eigenvalues,
                               JobType jobz,
                               const MatrixView<T, MatrixFormat::Dense>& eigvects) {
    if (d.size() < 1) {
        throw std::invalid_argument("cusolverdx::htev: d must have positive length");
    }
    if (e.size() != d.size() - 1) {
        throw std::invalid_argument("cusolverdx::htev: e size must be n-1");
    }
    if (eigenvalues.size() != d.size()) {
        throw std::invalid_argument("cusolverdx::htev: eigenvalues size must match d size");
    }
    if (e.batch_size() != d.batch_size() || eigenvalues.batch_size() != d.batch_size()) {
        throw std::invalid_argument("cusolverdx::htev: d/e/eigenvalues batch sizes must match");
    }
    if (jobz == JobType::EigenVectors) {
        if (eigvects.rows() != d.size() || eigvects.cols() != d.size()) {
            throw std::invalid_argument("cusolverdx::htev: eigvects must be n x n when eigenvectors are requested");
        }
        if (eigvects.batch_size() != d.batch_size()) {
            throw std::invalid_argument("cusolverdx::htev: eigvects batch size must match d batch size");
        }
    }
}

template <typename T>
inline MatrixView<T, MatrixFormat::Dense> dense_from_tridiagonal_workspace(Queue& ctx,
                                                                            BumpAllocator& pool,
                                                                            const VectorView<T>& d,
                                                                            const VectorView<T>& e,
                                                                            Uplo uplo) {
    const int n = d.size();
    const int batch = d.batch_size();
    auto dense_data = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(n) * static_cast<size_t>(batch));
    MatrixView<T, MatrixFormat::Dense> dense(dense_data.data(), n, n, n, n * n, batch);
    auto dense_kv = dense.kernel_view();

    ctx->parallel_for(sycl::range<3>(static_cast<size_t>(batch), static_cast<size_t>(n), static_cast<size_t>(n)),
                      [=](sycl::id<3> id) {
                          const int b = static_cast<int>(id[0]);
                          const int r = static_cast<int>(id[1]);
                          const int c = static_cast<int>(id[2]);
                          T v = T(0);
                          if (r == c) {
                              v = d(r, b);
                          } else if (r == c + 1) {
                              v = e(c, b);
                          } else if (c == r + 1) {
                              if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                                  v = std::conj(e(r, b));
                              } else {
                                  v = e(r, b);
                              }
                          }

                          if (uplo == Uplo::Lower) {
                              if (r >= c) dense_kv(r, c, b) = v;
                          } else {
                              if (r <= c) dense_kv(r, c, b) = v;
                          }
                      });
    return dense;
}

#if BATCHLAS_HAS_CUDA_BACKEND
inline cudaStream_t cuda_stream_from_queue(const Queue& ctx) {
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(*ctx);
}

inline void throw_on_cuda_error(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
inline bool try_heev_dx(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        Span<typename base_type<T>::type> eigenvalues,
                        JobType jobz,
                        Uplo uplo,
                        Span<std::byte> workspace) {
    if (!cuda_kernels::available()) return false;
    if (!cuda_kernels::heev_supported_n(A.rows())) return false;
    if (A.rows() != A.cols()) return false;
    if (A.batch_size() < 1) return false;
    if (workspace.size() < BumpAllocator::allocation_size<int>(ctx, A.batch_size())) return false;

    BumpAllocator pool(workspace);
    auto info = pool.allocate<int>(ctx, A.batch_size());
    auto stream = cuda_stream_from_queue(ctx);
    const bool compute_vectors = (jobz == JobType::EigenVectors);
    const bool lower = (uplo == Uplo::Lower);

    if constexpr (std::is_same_v<T, float>) {
        const auto err = cuda_kernels::heev_launch_float(A.data_ptr(),
                                                         A.rows(),
                                                         A.ld(),
                                                         eigenvalues.data(),
                                                         info.data(),
                                                         A.batch_size(),
                                                         compute_vectors,
                                                         lower,
                                                         stream);
        if (err == cudaErrorNotSupported) return false;
        throw_on_cuda_error(err, "cusolverdx::heev(float)");
        return true;
    } else if constexpr (std::is_same_v<T, double>) {
        const auto err = cuda_kernels::heev_launch_double(A.data_ptr(),
                                                          A.rows(),
                                                          A.ld(),
                                                          eigenvalues.data(),
                                                          info.data(),
                                                          A.batch_size(),
                                                          compute_vectors,
                                                          lower,
                                                          stream);
        if (err == cudaErrorNotSupported) return false;
        throw_on_cuda_error(err, "cusolverdx::heev(double)");
        return true;
    }
    return false;
}

template <typename T>
inline bool try_htev_dx(Queue& ctx,
                        const VectorView<T>& d,
                        const VectorView<T>& e,
                        const VectorView<T>& eigenvalues,
                        JobType jobz,
                        const MatrixView<T, MatrixFormat::Dense>& eigvects,
                        Span<std::byte> workspace) {
    if (!cuda_kernels::available()) return false;
    if (!cuda_kernels::htev_supported_n(d.size())) return false;
    if (d.batch_size() < 1) return false;
    if (workspace.size() < BumpAllocator::allocation_size<int>(ctx, d.batch_size())) return false;

    BumpAllocator pool(workspace);
    auto info = pool.allocate<int>(ctx, d.batch_size());
    auto stream = cuda_stream_from_queue(ctx);
    const bool compute_vectors = (jobz == JobType::EigenVectors);

    T* v_ptr = nullptr;
    int ldv = d.size();
    if (compute_vectors) {
        v_ptr = eigvects.data_ptr();
        ldv = eigvects.ld();
    }

    if constexpr (std::is_same_v<T, float>) {
        const auto err = cuda_kernels::htev_launch_float(d.data_ptr(),
                                                         d.size(),
                                                         e.data_ptr(),
                                                         v_ptr,
                                                         ldv,
                                                         info.data(),
                                                         d.batch_size(),
                                                         compute_vectors,
                                                         stream);
        if (err == cudaErrorNotSupported) return false;
        throw_on_cuda_error(err, "cusolverdx::htev(float)");
        VectorView<T>::copy(ctx, eigenvalues, d);
        return true;
    } else if constexpr (std::is_same_v<T, double>) {
        const auto err = cuda_kernels::htev_launch_double(d.data_ptr(),
                                                          d.size(),
                                                          e.data_ptr(),
                                                          v_ptr,
                                                          ldv,
                                                          info.data(),
                                                          d.batch_size(),
                                                          compute_vectors,
                                                          stream);
        if (err == cudaErrorNotSupported) return false;
        throw_on_cuda_error(err, "cusolverdx::htev(double)");
        VectorView<T>::copy(ctx, eigenvalues, d);
        return true;
    }
    return false;
}
#endif

} // namespace

template <typename T>
Event heev(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           Span<typename base_type<T>::type> eigenvalues,
           JobType jobz,
           Uplo uplo,
           Span<std::byte> workspace) {
#if BATCHLAS_HAS_CUDA_BACKEND
    return op_external("cusolverdx.heev", [&] {
        if (try_heev_dx<T>(ctx, A, eigenvalues, jobz, uplo, workspace)) {
            return ctx.create_event_after_external_work();
        }
        return backend::syev_vendor<Backend::CUDA, T>(ctx, A, eigenvalues, jobz, uplo, workspace);
    });
#else
    static_cast<void>(ctx);
    static_cast<void>(A);
    static_cast<void>(eigenvalues);
    static_cast<void>(jobz);
    static_cast<void>(uplo);
    static_cast<void>(workspace);
    throw std::runtime_error("cusolverdx::heev requires CUDA backend support");
#endif
}

template <typename T>
size_t heev_buffer_size(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        Span<typename base_type<T>::type> eigenvalues,
                        JobType jobz,
                        Uplo uplo) {
#if BATCHLAS_HAS_CUDA_BACKEND
    return op_external("cusolverdx.heev_buffer_size", [&] {
        return backend::syev_vendor_buffer_size<Backend::CUDA, T>(ctx, A, eigenvalues, jobz, uplo);
    });
#else
    static_cast<void>(ctx);
    static_cast<void>(A);
    static_cast<void>(eigenvalues);
    static_cast<void>(jobz);
    static_cast<void>(uplo);
    throw std::runtime_error("cusolverdx::heev_buffer_size requires CUDA backend support");
#endif
}

template <typename T>
Event htev(Queue& ctx,
           const VectorView<T>& d,
           const VectorView<T>& e,
           const VectorView<T>& eigenvalues,
           JobType jobz,
           const MatrixView<T, MatrixFormat::Dense>& eigvects,
           Span<std::byte> workspace,
           Uplo uplo) {
#if BATCHLAS_HAS_CUDA_BACKEND
    return op_external("cusolverdx.htev", [&] {
        validate_htev_args(d, e, eigenvalues, jobz, eigvects);

        if (try_htev_dx<T>(ctx, d, e, eigenvalues, jobz, eigvects, workspace)) {
            return ctx.get_event();
        }

        BumpAllocator pool(workspace);

        auto dense = dense_from_tridiagonal_workspace(ctx, pool, d, e, uplo);
        auto lambda = Span<T>(eigenvalues.data(), static_cast<size_t>(eigenvalues.size()) * static_cast<size_t>(eigenvalues.batch_size()));

        const size_t syev_ws = backend::syev_vendor_buffer_size<Backend::CUDA, T>(ctx, dense, lambda, jobz, uplo);
        auto syev_workspace = pool.allocate<std::byte>(ctx, syev_ws);

        backend::syev_vendor<Backend::CUDA, T>(ctx, dense, lambda, jobz, uplo, syev_workspace);

        if (jobz == JobType::EigenVectors) {
            MatrixView<T, MatrixFormat::Dense>::copy(ctx, eigvects, dense);
        }
        return ctx.get_event();
    });
#else
    static_cast<void>(ctx);
    static_cast<void>(d);
    static_cast<void>(e);
    static_cast<void>(eigenvalues);
    static_cast<void>(jobz);
    static_cast<void>(eigvects);
    static_cast<void>(workspace);
    static_cast<void>(uplo);
    throw std::runtime_error("cusolverdx::htev requires CUDA backend support");
#endif
}

template <typename T>
size_t htev_buffer_size(Queue& ctx,
                        const VectorView<T>& d,
                        const VectorView<T>& e,
                        JobType jobz,
                        Uplo uplo) {
#if BATCHLAS_HAS_CUDA_BACKEND
    return op_external("cusolverdx.htev_buffer_size", [&] {
        if (d.size() < 1 || e.size() != d.size() - 1 || e.batch_size() != d.batch_size()) {
            throw std::invalid_argument("cusolverdx::htev_buffer_size: invalid d/e sizes");
        }
        const int n = d.size();
        const int batch = d.batch_size();
        const size_t dense_bytes = BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(n) * static_cast<size_t>(batch));

        auto dense_tmp = Matrix<T, MatrixFormat::Dense>::Zeros(n, n, batch);
        auto eigvals_tmp = Vector<T>::zeros(n, batch);
        Span<T> eigvals_tmp_span(eigvals_tmp.data_ptr(), static_cast<size_t>(n) * static_cast<size_t>(batch));
        const size_t syev_ws = backend::syev_vendor_buffer_size<Backend::CUDA, T>(ctx,
                                                                                    dense_tmp.view(),
                                                eigvals_tmp_span,
                                                                                    jobz,
                                                                                    uplo);
        return dense_bytes + BumpAllocator::allocation_size<std::byte>(ctx, syev_ws);
    });
#else
    static_cast<void>(ctx);
    static_cast<void>(d);
    static_cast<void>(e);
    static_cast<void>(jobz);
    static_cast<void>(uplo);
    throw std::runtime_error("cusolverdx::htev_buffer_size requires CUDA backend support");
#endif
}

#if BATCHLAS_HAS_CUDA_BACKEND
template Event heev<float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&, Span<float>, JobType, Uplo, Span<std::byte>);
template Event heev<double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&, Span<double>, JobType, Uplo, Span<std::byte>);
template Event heev<std::complex<float>>(Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&, Span<float>, JobType, Uplo, Span<std::byte>);
template Event heev<std::complex<double>>(Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&, Span<double>, JobType, Uplo, Span<std::byte>);

template size_t heev_buffer_size<float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&, Span<float>, JobType, Uplo);
template size_t heev_buffer_size<double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&, Span<double>, JobType, Uplo);
template size_t heev_buffer_size<std::complex<float>>(Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&, Span<float>, JobType, Uplo);
template size_t heev_buffer_size<std::complex<double>>(Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&, Span<double>, JobType, Uplo);

template Event htev<float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, JobType, const MatrixView<float, MatrixFormat::Dense>&, Span<std::byte>, Uplo);
template Event htev<double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, JobType, const MatrixView<double, MatrixFormat::Dense>&, Span<std::byte>, Uplo);
template size_t htev_buffer_size<float>(Queue&, const VectorView<float>&, const VectorView<float>&, JobType, Uplo);
template size_t htev_buffer_size<double>(Queue&, const VectorView<double>&, const VectorView<double>&, JobType, Uplo);
#endif

} // namespace batchlas::backend::cusolverdx
