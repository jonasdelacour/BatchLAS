#include <blas/extra.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <util/sycl-vector.hh>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace batchlas {

namespace detail {
    template <typename T>
    constexpr bool is_std_complex_v =
        std::is_same_v<T, std::complex<float>> ||
        std::is_same_v<T, std::complex<double>>;

    template <typename T>
    using real_t = float_t<T>;

    template <typename T>
    inline T real_to_T(real_t<T> v) {
        if constexpr (is_std_complex_v<T>) {
            return T(v, real_t<T>(0));
        } else {
            return T(v);
        }
    }

    template <typename T>
    UnifiedVector<T> build_log_spectrum(int n, real_t<T> log10_cond) {
        UnifiedVector<T> diag_vals(n);
        if (n == 1) {
            diag_vals[0] = real_to_T<T>(real_t<T>(1));
            return diag_vals;
        }

        const real_t<T> step = log10_cond / real_t<T>(n - 1);
        for (int i = 0; i < n; ++i) {
            const real_t<T> s = std::pow(real_t<T>(10), step * real_t<T>(i));
            diag_vals[i] = real_to_T<T>(s);
        }
        return diag_vals;
    }

    template <typename T>
    void apply_givens_rows(Matrix<T, MatrixFormat::Dense>& Q,
                           int row1,
                           int row2,
                           int batch,
                           real_t<T> c,
                           real_t<T> s) {
        const int n = Q.cols();
        const T tc = real_to_T<T>(c);
        const T ts = real_to_T<T>(s);
        for (int col = 0; col < n; ++col) {
            const T q1 = Q(row1, col, batch);
            const T q2 = Q(row2, col, batch);
            Q(row1, col, batch) = tc * q1 + ts * q2;
            Q(row2, col, batch) = real_to_T<T>(-s) * q1 + tc * q2;
        }
    }

    template <typename T>
    Matrix<T, MatrixFormat::Dense> banded_orthogonal(int n,
                                                     int bandwidth,
                                                     int batch_size,
                                                     unsigned int seed) {
        Matrix<T, MatrixFormat::Dense> Q = Matrix<T, MatrixFormat::Dense>::Identity(n, batch_size);
        if (bandwidth <= 0) return Q;

        const real_t<T> two_pi = real_t<T>(2) * std::acos(real_t<T>(-1));
        for (int b = 0; b < batch_size; ++b) {
            std::mt19937 rng(seed + 1315423911u * static_cast<unsigned int>(b));
            std::uniform_real_distribution<real_t<T>> angle_dist(real_t<T>(0), two_pi);

            const int offset = static_cast<int>(rng() % static_cast<unsigned int>(bandwidth + 1));
            for (int i = offset; i + bandwidth < n; i += (bandwidth + 1)) {
                const real_t<T> theta = angle_dist(rng);
                const real_t<T> c = std::cos(theta);
                const real_t<T> s = std::sin(theta);
                apply_givens_rows(Q, i, i + bandwidth, b, c, s);
            }
        }
        return Q;
    }
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
    UnifiedVector<T> diag_vals = detail::build_log_spectrum<T>(n, log10_cond);

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

// Create a batch of random symmetric/Hermitian matrices with a specified log10 condition number.
template <Backend B, typename T>
Matrix<T, MatrixFormat::Dense> random_hermitian_with_log10_cond(Queue &ctx,
                                                                int n,
                                                                float_t<T> log10_cond,
                                                                int batch_size,
                                                                unsigned int seed,
                                                                OrthoAlgorithm algo) {
    using real_t = float_t<T>;
    if (n <= 0 || batch_size <= 0) {
        throw std::runtime_error("random_hermitian_with_log10_cond: n and batch_size must be positive");
    }
    if (log10_cond < real_t(0)) {
        throw std::runtime_error("random_hermitian_with_log10_cond: log10_cond must be non-negative");
    }

    Matrix<T, MatrixFormat::Dense> Q = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch_size, seed);
    const size_t ortho_ws = ortho_buffer_size<B>(ctx, Q.view(), Transpose::NoTrans, algo);
    UnifiedVector<std::byte> workspace(ortho_ws);
    ortho<B>(ctx, Q.view(), Transpose::NoTrans, workspace.to_span(), algo).wait();

    UnifiedVector<T> diag_vals = detail::build_log_spectrum<T>(n, log10_cond);
    Matrix<T, MatrixFormat::Dense> D = Matrix<T, MatrixFormat::Dense>::Diagonal(diag_vals.to_span(), batch_size);

    Matrix<T, MatrixFormat::Dense> tmp(n, n, batch_size);
    Matrix<T, MatrixFormat::Dense> A(n, n, batch_size);

    gemm<B>(ctx, Q.view(), D.view(), tmp.view(), T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
    const Transpose q_trans = detail::is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;
    gemm<B>(ctx, tmp.view(), Q.view(), A.view(), T(1), T(0), Transpose::NoTrans, q_trans);
    ctx.wait();

    return A;
}

// Create a batch of random dense banded matrices with a specified log10 condition number.
template <Backend B, typename T>
Matrix<T, MatrixFormat::Dense> random_banded_with_log10_cond(Queue &ctx,
                                                             int n,
                                                             int kd,
                                                             float_t<T> log10_cond,
                                                             int batch_size,
                                                             unsigned int seed) {
    using real_t = float_t<T>;
    if (n <= 0 || batch_size <= 0) {
        throw std::runtime_error("random_banded_with_log10_cond: n and batch_size must be positive");
    }
    if (kd < 0) {
        throw std::runtime_error("random_banded_with_log10_cond: kd must be non-negative");
    }
    if (log10_cond < real_t(0)) {
        throw std::runtime_error("random_banded_with_log10_cond: log10_cond must be non-negative");
    }
    if (kd >= n - 1) {
        return random_with_log10_cond<B, T>(ctx, n, log10_cond, batch_size, seed);
    }

    const int band = kd / 2;
    Matrix<T, MatrixFormat::Dense> Q = detail::banded_orthogonal<T>(n, band, batch_size, seed);
    Matrix<T, MatrixFormat::Dense> R = detail::banded_orthogonal<T>(n, band, batch_size, seed + 1);

    UnifiedVector<T> diag_vals = detail::build_log_spectrum<T>(n, log10_cond);
    Matrix<T, MatrixFormat::Dense> D = Matrix<T, MatrixFormat::Dense>::Diagonal(diag_vals.to_span(), batch_size);
    Matrix<T, MatrixFormat::Dense> tmp(n, n, batch_size);
    Matrix<T, MatrixFormat::Dense> A(n, n, batch_size);

    gemm<B>(ctx, Q.view(), D.view(), tmp.view(), T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
    const Transpose r_trans = detail::is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;
    gemm<B>(ctx, tmp.view(), R.view(), A.view(), T(1), T(0), Transpose::NoTrans, r_trans);
    ctx.wait();

    return A;
}

// Create a batch of random symmetric/Hermitian banded matrices with a specified log10 condition number.
template <Backend B, typename T>
Matrix<T, MatrixFormat::Dense> random_hermitian_banded_with_log10_cond(Queue &ctx,
                                                                       int n,
                                                                       int kd,
                                                                       float_t<T> log10_cond,
                                                                       int batch_size,
                                                                       unsigned int seed) {
    using real_t = float_t<T>;
    if (n <= 0 || batch_size <= 0) {
        throw std::runtime_error("random_hermitian_banded_with_log10_cond: n and batch_size must be positive");
    }
    if (kd < 0) {
        throw std::runtime_error("random_hermitian_banded_with_log10_cond: kd must be non-negative");
    }
    if (log10_cond < real_t(0)) {
        throw std::runtime_error("random_hermitian_banded_with_log10_cond: log10_cond must be non-negative");
    }
    if (kd >= n - 1) {
        return random_hermitian_with_log10_cond<B, T>(ctx, n, log10_cond, batch_size, seed);
    }

    const int band = kd / 2;
    Matrix<T, MatrixFormat::Dense> Q = detail::banded_orthogonal<T>(n, band, batch_size, seed);

    UnifiedVector<T> diag_vals = detail::build_log_spectrum<T>(n, log10_cond);
    Matrix<T, MatrixFormat::Dense> D = Matrix<T, MatrixFormat::Dense>::Diagonal(diag_vals.to_span(), batch_size);
    Matrix<T, MatrixFormat::Dense> tmp(n, n, batch_size);
    Matrix<T, MatrixFormat::Dense> A(n, n, batch_size);

    gemm<B>(ctx, Q.view(), D.view(), tmp.view(), T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
    const Transpose q_trans = detail::is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;
    gemm<B>(ctx, tmp.view(), Q.view(), A.view(), T(1), T(0), Transpose::NoTrans, q_trans);
    ctx.wait();

    return A;
}

// Create a batch of random tridiagonal matrices with a specified log10 condition number.
template <Backend B, typename T>
Matrix<T, MatrixFormat::Dense> random_tridiagonal_with_log10_cond(Queue &ctx,
                                                                  int n,
                                                                  float_t<T> log10_cond,
                                                                  int batch_size,
                                                                  unsigned int /*seed*/) {
    using real_t = float_t<T>;
    if (n <= 0 || batch_size <= 0) {
        throw std::runtime_error("random_tridiagonal_with_log10_cond: n and batch_size must be positive");
    }
    if (log10_cond < real_t(0)) {
        throw std::runtime_error("random_tridiagonal_with_log10_cond: log10_cond must be non-negative");
    }

    UnifiedVector<T> diag_vals = detail::build_log_spectrum<T>(n, log10_cond);
    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Zeros(n, n, batch_size);
    A.view().template fill_diagonal<MatrixFormat::Dense>(ctx, diag_vals.to_span()).wait();
    return A;
}

// Create a batch of random symmetric/Hermitian tridiagonal matrices with a specified log10 condition number.
template <Backend B, typename T>
Matrix<T, MatrixFormat::Dense> random_hermitian_tridiagonal_with_log10_cond(Queue &ctx,
                                                                            int n,
                                                                            float_t<T> log10_cond,
                                                                            int batch_size,
                                                                            unsigned int /*seed*/) {
    using real_t = float_t<T>;
    if (n <= 0 || batch_size <= 0) {
        throw std::runtime_error("random_hermitian_tridiagonal_with_log10_cond: n and batch_size must be positive");
    }
    if (log10_cond < real_t(0)) {
        throw std::runtime_error("random_hermitian_tridiagonal_with_log10_cond: log10_cond must be non-negative");
    }

    UnifiedVector<T> diag_vals = detail::build_log_spectrum<T>(n, log10_cond);
    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Zeros(n, n, batch_size);
    A.view().template fill_diagonal<MatrixFormat::Dense>(ctx, diag_vals.to_span()).wait();
    return A;
}

#define RANDOM_LOGCOND_INSTANTIATE(back, fp) \
    template Matrix<fp, MatrixFormat::Dense> random_with_log10_cond<back, fp>( \
        Queue&, int, float_t<fp>, int, unsigned int, OrthoAlgorithm);

#define RANDOM_HERMITIAN_LOGCOND_INSTANTIATE(back, fp) \
    template Matrix<fp, MatrixFormat::Dense> random_hermitian_with_log10_cond<back, fp>( \
        Queue&, int, float_t<fp>, int, unsigned int, OrthoAlgorithm);

#define RANDOM_BANDED_LOGCOND_INSTANTIATE(back, fp) \
    template Matrix<fp, MatrixFormat::Dense> random_banded_with_log10_cond<back, fp>( \
        Queue&, int, int, float_t<fp>, int, unsigned int);

#define RANDOM_HERMITIAN_BANDED_LOGCOND_INSTANTIATE(back, fp) \
    template Matrix<fp, MatrixFormat::Dense> random_hermitian_banded_with_log10_cond<back, fp>( \
        Queue&, int, int, float_t<fp>, int, unsigned int);

#define RANDOM_TRIDIAG_LOGCOND_INSTANTIATE(back, fp) \
    template Matrix<fp, MatrixFormat::Dense> random_tridiagonal_with_log10_cond<back, fp>( \
        Queue&, int, float_t<fp>, int, unsigned int);

#define RANDOM_HERMITIAN_TRIDIAG_LOGCOND_INSTANTIATE(back, fp) \
    template Matrix<fp, MatrixFormat::Dense> random_hermitian_tridiagonal_with_log10_cond<back, fp>( \
        Queue&, int, float_t<fp>, int, unsigned int);

#if BATCHLAS_HAS_CUDA_BACKEND
    RANDOM_LOGCOND_INSTANTIATE(Backend::CUDA, float)
    RANDOM_LOGCOND_INSTANTIATE(Backend::CUDA, double)
    RANDOM_LOGCOND_INSTANTIATE(Backend::CUDA, std::complex<float>)
    RANDOM_LOGCOND_INSTANTIATE(Backend::CUDA, std::complex<double>)
    RANDOM_HERMITIAN_LOGCOND_INSTANTIATE(Backend::CUDA, float)
    RANDOM_HERMITIAN_LOGCOND_INSTANTIATE(Backend::CUDA, double)
    RANDOM_HERMITIAN_LOGCOND_INSTANTIATE(Backend::CUDA, std::complex<float>)
    RANDOM_HERMITIAN_LOGCOND_INSTANTIATE(Backend::CUDA, std::complex<double>)
    RANDOM_BANDED_LOGCOND_INSTANTIATE(Backend::CUDA, float)
    RANDOM_BANDED_LOGCOND_INSTANTIATE(Backend::CUDA, double)
    RANDOM_BANDED_LOGCOND_INSTANTIATE(Backend::CUDA, std::complex<float>)
    RANDOM_BANDED_LOGCOND_INSTANTIATE(Backend::CUDA, std::complex<double>)
    RANDOM_HERMITIAN_BANDED_LOGCOND_INSTANTIATE(Backend::CUDA, float)
    RANDOM_HERMITIAN_BANDED_LOGCOND_INSTANTIATE(Backend::CUDA, double)
    RANDOM_HERMITIAN_BANDED_LOGCOND_INSTANTIATE(Backend::CUDA, std::complex<float>)
    RANDOM_HERMITIAN_BANDED_LOGCOND_INSTANTIATE(Backend::CUDA, std::complex<double>)
    RANDOM_TRIDIAG_LOGCOND_INSTANTIATE(Backend::CUDA, float)
    RANDOM_TRIDIAG_LOGCOND_INSTANTIATE(Backend::CUDA, double)
    RANDOM_TRIDIAG_LOGCOND_INSTANTIATE(Backend::CUDA, std::complex<float>)
    RANDOM_TRIDIAG_LOGCOND_INSTANTIATE(Backend::CUDA, std::complex<double>)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_INSTANTIATE(Backend::CUDA, float)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_INSTANTIATE(Backend::CUDA, double)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_INSTANTIATE(Backend::CUDA, std::complex<float>)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    RANDOM_LOGCOND_INSTANTIATE(Backend::ROCM, float)
    RANDOM_LOGCOND_INSTANTIATE(Backend::ROCM, double)
    RANDOM_LOGCOND_INSTANTIATE(Backend::ROCM, std::complex<float>)
    RANDOM_LOGCOND_INSTANTIATE(Backend::ROCM, std::complex<double>)
    RANDOM_HERMITIAN_LOGCOND_INSTANTIATE(Backend::ROCM, float)
    RANDOM_HERMITIAN_LOGCOND_INSTANTIATE(Backend::ROCM, double)
    RANDOM_HERMITIAN_LOGCOND_INSTANTIATE(Backend::ROCM, std::complex<float>)
    RANDOM_HERMITIAN_LOGCOND_INSTANTIATE(Backend::ROCM, std::complex<double>)
    RANDOM_BANDED_LOGCOND_INSTANTIATE(Backend::ROCM, float)
    RANDOM_BANDED_LOGCOND_INSTANTIATE(Backend::ROCM, double)
    RANDOM_BANDED_LOGCOND_INSTANTIATE(Backend::ROCM, std::complex<float>)
    RANDOM_BANDED_LOGCOND_INSTANTIATE(Backend::ROCM, std::complex<double>)
    RANDOM_HERMITIAN_BANDED_LOGCOND_INSTANTIATE(Backend::ROCM, float)
    RANDOM_HERMITIAN_BANDED_LOGCOND_INSTANTIATE(Backend::ROCM, double)
    RANDOM_HERMITIAN_BANDED_LOGCOND_INSTANTIATE(Backend::ROCM, std::complex<float>)
    RANDOM_HERMITIAN_BANDED_LOGCOND_INSTANTIATE(Backend::ROCM, std::complex<double>)
    RANDOM_TRIDIAG_LOGCOND_INSTANTIATE(Backend::ROCM, float)
    RANDOM_TRIDIAG_LOGCOND_INSTANTIATE(Backend::ROCM, double)
    RANDOM_TRIDIAG_LOGCOND_INSTANTIATE(Backend::ROCM, std::complex<float>)
    RANDOM_TRIDIAG_LOGCOND_INSTANTIATE(Backend::ROCM, std::complex<double>)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_INSTANTIATE(Backend::ROCM, float)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_INSTANTIATE(Backend::ROCM, double)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_INSTANTIATE(Backend::ROCM, std::complex<float>)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_INSTANTIATE(Backend::ROCM, std::complex<double>)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
    RANDOM_LOGCOND_INSTANTIATE(Backend::NETLIB, float)
    RANDOM_LOGCOND_INSTANTIATE(Backend::NETLIB, double)
    RANDOM_LOGCOND_INSTANTIATE(Backend::NETLIB, std::complex<float>)
    RANDOM_LOGCOND_INSTANTIATE(Backend::NETLIB, std::complex<double>)
    RANDOM_HERMITIAN_LOGCOND_INSTANTIATE(Backend::NETLIB, float)
    RANDOM_HERMITIAN_LOGCOND_INSTANTIATE(Backend::NETLIB, double)
    RANDOM_HERMITIAN_LOGCOND_INSTANTIATE(Backend::NETLIB, std::complex<float>)
    RANDOM_HERMITIAN_LOGCOND_INSTANTIATE(Backend::NETLIB, std::complex<double>)
    RANDOM_BANDED_LOGCOND_INSTANTIATE(Backend::NETLIB, float)
    RANDOM_BANDED_LOGCOND_INSTANTIATE(Backend::NETLIB, double)
    RANDOM_BANDED_LOGCOND_INSTANTIATE(Backend::NETLIB, std::complex<float>)
    RANDOM_BANDED_LOGCOND_INSTANTIATE(Backend::NETLIB, std::complex<double>)
    RANDOM_HERMITIAN_BANDED_LOGCOND_INSTANTIATE(Backend::NETLIB, float)
    RANDOM_HERMITIAN_BANDED_LOGCOND_INSTANTIATE(Backend::NETLIB, double)
    RANDOM_HERMITIAN_BANDED_LOGCOND_INSTANTIATE(Backend::NETLIB, std::complex<float>)
    RANDOM_HERMITIAN_BANDED_LOGCOND_INSTANTIATE(Backend::NETLIB, std::complex<double>)
    RANDOM_TRIDIAG_LOGCOND_INSTANTIATE(Backend::NETLIB, float)
    RANDOM_TRIDIAG_LOGCOND_INSTANTIATE(Backend::NETLIB, double)
    RANDOM_TRIDIAG_LOGCOND_INSTANTIATE(Backend::NETLIB, std::complex<float>)
    RANDOM_TRIDIAG_LOGCOND_INSTANTIATE(Backend::NETLIB, std::complex<double>)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_INSTANTIATE(Backend::NETLIB, float)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_INSTANTIATE(Backend::NETLIB, double)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_INSTANTIATE(Backend::NETLIB, std::complex<float>)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

} // namespace batchlas
