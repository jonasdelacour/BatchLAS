#include <blas/extra.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <util/sycl-vector.hh>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <type_traits>

namespace batchlas {

namespace detail {
    template <typename T>
    constexpr bool is_std_complex_v =
        std::is_same_v<T, std::complex<float>> ||
        std::is_same_v<T, std::complex<double>>;
}

// Create a batch of random dense matrices with a specified log10 condition number.
template <Backend B, typename T>
Matrix<T, MatrixFormat::Dense> random_with_log10_cond(Queue &ctx,
                                                      int n,
                                                      float_t<T> log10_cond,
                                                      int batch_size,
                                                      unsigned int seed,
                                                      OrthoAlgorithm algo) {
    using real_t = float_t<T>;

    if (n <= 0 || batch_size <= 0) {
        throw std::runtime_error("random_with_log10_cond: n and batch_size must be positive");
    }
    if (log10_cond < real_t(0)) {
        throw std::runtime_error("random_with_log10_cond: log10_cond must be non-negative");
    }

    // Random matrices for orthogonal factors
    Matrix<T, MatrixFormat::Dense> U = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch_size, seed);
    Matrix<T, MatrixFormat::Dense> V = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch_size, seed + 1);

    // Orthonormalize columns of U and V
    const size_t ortho_ws = ortho_buffer_size<B>(ctx, U.view(), Transpose::NoTrans, algo);
    UnifiedVector<std::byte> workspace(ortho_ws);
    ortho<B>(ctx, U.view(), Transpose::NoTrans, workspace.to_span(), algo).wait();
    ortho<B>(ctx, V.view(), Transpose::NoTrans, workspace.to_span(), algo).wait();

    // Build singular values with the requested condition number.
    UnifiedVector<T> diag_vals(n);
    if (n == 1) {
        diag_vals[0] = T(1);
    } else {
        const real_t step = log10_cond / real_t(n - 1);
        for (int i = 0; i < n; ++i) {
            const real_t s = std::pow(real_t(10), step * real_t(i));
            if constexpr (detail::is_std_complex_v<T>) {
                diag_vals[i] = T(s, real_t(0));
            } else {
                diag_vals[i] = T(s);
            }
        }
    }

    Matrix<T, MatrixFormat::Dense> S = Matrix<T, MatrixFormat::Dense>::Diagonal(diag_vals.to_span(), batch_size);
    Matrix<T, MatrixFormat::Dense> tmp(n, n, batch_size);
    Matrix<T, MatrixFormat::Dense> A(n, n, batch_size);

    // A = U * S * V^H
    gemm<B>(ctx, U.view(), S.view(), tmp.view(), T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
    const Transpose v_trans = detail::is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;
    gemm<B>(ctx, tmp.view(), V.view(), A.view(), T(1), T(0), Transpose::NoTrans, v_trans);
    ctx.wait();

    return A;
}

#define RANDOM_LOGCOND_INSTANTIATE(back, fp) \
    template Matrix<fp, MatrixFormat::Dense> random_with_log10_cond<back, fp>( \
        Queue&, int, float_t<fp>, int, unsigned int, OrthoAlgorithm);

#if BATCHLAS_HAS_CUDA_BACKEND
    RANDOM_LOGCOND_INSTANTIATE(Backend::CUDA, float)
    RANDOM_LOGCOND_INSTANTIATE(Backend::CUDA, double)
    RANDOM_LOGCOND_INSTANTIATE(Backend::CUDA, std::complex<float>)
    RANDOM_LOGCOND_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    RANDOM_LOGCOND_INSTANTIATE(Backend::ROCM, float)
    RANDOM_LOGCOND_INSTANTIATE(Backend::ROCM, double)
    RANDOM_LOGCOND_INSTANTIATE(Backend::ROCM, std::complex<float>)
    RANDOM_LOGCOND_INSTANTIATE(Backend::ROCM, std::complex<double>)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
    RANDOM_LOGCOND_INSTANTIATE(Backend::NETLIB, float)
    RANDOM_LOGCOND_INSTANTIATE(Backend::NETLIB, double)
    RANDOM_LOGCOND_INSTANTIATE(Backend::NETLIB, std::complex<float>)
    RANDOM_LOGCOND_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

} // namespace batchlas
