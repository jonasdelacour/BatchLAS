#include <blas/extra.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <util/sycl-vector.hh>
#include <cmath>
#include <complex>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <limits>
#include <algorithm>
#include <string>

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
    UnifiedVector<T> build_spectrum_kappa2(int n, real_t<T> log10_kappa2) {
        UnifiedVector<T> diag_vals(n);
        if (n == 1) {
            diag_vals[0] = real_to_T<T>(real_t<T>(1));
            return diag_vals;
        }

        const long double log_kappa2 = std::log(10.0L) * static_cast<long double>(log10_kappa2);
        const long double t = log_kappa2 / static_cast<long double>(n - 1);
        const long double m = static_cast<long double>(n - 1) * 0.5L;
        for (int i = 0; i < n; ++i) {
            const long double exp_arg = (static_cast<long double>(i) - m) * t;
            const long double s = std::exp(exp_arg);
            diag_vals[i] = real_to_T<T>(static_cast<real_t<T>>(s));
        }
        return diag_vals;
    }

    inline long double log_kappaF_geometric(int n, long double t) {
        const long double m = static_cast<long double>(n - 1) * 0.5L;
        long double max1 = -std::numeric_limits<long double>::infinity();
        long double max2 = -std::numeric_limits<long double>::infinity();

        for (int i = 0; i < n; ++i) {
            const long double x = (static_cast<long double>(i) - m) * t;
            const long double v = 2.0L * x;
            const long double w = -2.0L * x;
            if (v > max1) max1 = v;
            if (w > max2) max2 = w;
        }

        long double sum1 = 0.0L;
        long double sum2 = 0.0L;
        for (int i = 0; i < n; ++i) {
            const long double x = (static_cast<long double>(i) - m) * t;
            sum1 += std::exp(2.0L * x - max1);
            sum2 += std::exp(-2.0L * x - max2);
        }

        const long double log_sum1 = max1 + std::log(sum1);
        const long double log_sum2 = max2 + std::log(sum2);
        return 0.5L * (log_sum1 + log_sum2);
    }

    template <typename T>
    UnifiedVector<T> build_spectrum_kappaF(int n, real_t<T> log10_kappaF) {
        UnifiedVector<T> diag_vals(n);
        if (n == 1) {
            diag_vals[0] = real_to_T<T>(real_t<T>(1));
            return diag_vals;
        }

        const long double log10_n = std::log10(static_cast<long double>(n));
        if (static_cast<long double>(log10_kappaF) < log10_n) {
            throw std::runtime_error("build_spectrum_kappaF: log10_kappaF must be >= log10(n)");
        }

        const long double target = std::log(10.0L) * static_cast<long double>(log10_kappaF);
        const long double log_kappa0 = std::log(static_cast<long double>(n));
        if (target <= log_kappa0) {
            for (int i = 0; i < n; ++i) {
                diag_vals[i] = real_to_T<T>(real_t<T>(1));
            }
            return diag_vals;
        }

        long double t_lo = 0.0L;
        long double t_hi = 1.0L;
        while (log_kappaF_geometric(n, t_hi) < target) {
            t_hi *= 2.0L;
        }

        for (int iter = 0; iter < 80; ++iter) {
            const long double t_mid = 0.5L * (t_lo + t_hi);
            if (log_kappaF_geometric(n, t_mid) < target) {
                t_lo = t_mid;
            } else {
                t_hi = t_mid;
            }
        }

        const long double t = 0.5L * (t_lo + t_hi);
        const long double m = static_cast<long double>(n - 1) * 0.5L;
        for (int i = 0; i < n; ++i) {
            const long double exp_arg = (static_cast<long double>(i) - m) * t;
            const long double s = std::exp(exp_arg);
            diag_vals[i] = real_to_T<T>(static_cast<real_t<T>>(s));
        }
        return diag_vals;
    }

    template <typename T>
    UnifiedVector<T> build_spectrum_for_metric(int n,
                                               real_t<T> log10_kappa,
                                               NormType metric,
                                               const char* func_name) {
        switch (metric) {
            case NormType::Spectral:
                return build_spectrum_kappa2<T>(n, log10_kappa);
            case NormType::Frobenius:
                return build_spectrum_kappaF<T>(n, log10_kappa);
            default:
                throw std::runtime_error(std::string(func_name) + ": metric must be NormType::Spectral or NormType::Frobenius");
        }
    }

    template <Backend B, typename T>
    Matrix<T, MatrixFormat::Dense> reduce_hermitian_to_tridiagonal(Queue& ctx,
                                                                   Matrix<T, MatrixFormat::Dense>& A,
                                                                   int block_size) {
        const int n = A.rows();
        const int batch_size = A.batch_size();
        Vector<T> d(n, batch_size);
        Vector<T> e(std::max(0, n - 1), batch_size);
        Vector<T> tau(std::max(0, n - 1), batch_size);

        auto dv = static_cast<VectorView<T>>(d);
        auto ev = static_cast<VectorView<T>>(e);
        auto tauv = static_cast<VectorView<T>>(tau);

        const size_t ws_bytes = sytrd_blocked_buffer_size<B, T>(ctx, A.view(), dv, ev, tauv, Uplo::Lower, block_size);
        UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});
        sytrd_blocked<B, T>(ctx, A.view(), dv, ev, tauv, Uplo::Lower, ws.to_span(), block_size).wait();

        Matrix<T, MatrixFormat::Dense> Tmat = Matrix<T, MatrixFormat::Dense>::Zeros(n, n, batch_size);
        auto Tv = Tmat.view();
        Tv.template fill_diagonal<MatrixFormat::Dense>(ctx, dv).wait();

        if (n > 1) {
            Vector<T> e_conj(n - 1, batch_size);
            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < n - 1; ++i) {
                    if constexpr (is_std_complex_v<T>) {
                        e_conj(i, b) = std::conj(e(i, b));
                    } else {
                        e_conj(i, b) = e(i, b);
                    }
                }
            }

            auto ev_conj = static_cast<VectorView<T>>(e_conj);
            Tv.template fill_diagonal<MatrixFormat::Dense>(ctx, ev, -1).wait();
            Tv.template fill_diagonal<MatrixFormat::Dense>(ctx, ev_conj, 1).wait();
        }

        return Tmat;
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

    template <Backend B>
    constexpr OrthoAlgorithm effective_ortho_algo(OrthoAlgorithm algo) {
        if constexpr (B == Backend::NETLIB) {
            if (algo == OrthoAlgorithm::Cholesky ||
                algo == OrthoAlgorithm::Chol2 ||
                algo == OrthoAlgorithm::ShiftChol3) {
                return OrthoAlgorithm::Householder;
            }
        }
        return algo;
    }
}

// Create a batch of random dense matrices with a specified log10 conditioning metric.
template <Backend B, typename T>
Matrix<T, MatrixFormat::Dense> random_with_log10_cond_metric(Queue &ctx,
                                                             int n,
                                                             float_t<T> log10_kappa,
                                                             NormType metric,
                                                             int batch_size,
                                                             unsigned int seed,
                                                             OrthoAlgorithm algo) {
    using real_t = float_t<T>;

    if (n <= 0 || batch_size <= 0) {
        throw std::runtime_error("random_with_log10_cond_metric: n and batch_size must be positive");
    }
    if (log10_kappa < real_t(0)) {
        throw std::runtime_error("random_with_log10_cond_metric: log10_kappa must be non-negative");
    }

    if constexpr (B == Backend::NETLIB) {
        UnifiedVector<T> diag_vals = detail::build_spectrum_for_metric<T>(
            n, log10_kappa, metric, "random_with_log10_cond_metric");
        return Matrix<T>::Diagonal(diag_vals.to_span(), batch_size);
    }

    const OrthoAlgorithm ortho_algo = detail::effective_ortho_algo<B>(algo);

    // Random matrices for orthogonal factors
    Matrix<T> U = Matrix<T>::Random(n, n, false, batch_size, seed);
    Matrix<T> V = Matrix<T>::Random(n, n, false, batch_size, seed + 1);

    // Orthonormalize columns of U and V
    const size_t ortho_ws = ortho_buffer_size<B>(ctx, U.view(), Transpose::NoTrans, ortho_algo);
    UnifiedVector<std::byte> workspace(ortho_ws);
    ortho<B>(ctx, U.view(), Transpose::NoTrans, workspace.to_span(), ortho_algo).wait();
    ortho<B>(ctx, V.view(), Transpose::NoTrans, workspace.to_span(), ortho_algo).wait();

    // Build singular values with the requested condition number.
    UnifiedVector<T> diag_vals = detail::build_spectrum_for_metric<T>(
        n, log10_kappa, metric, "random_with_log10_cond_metric");

    Matrix<T> S = Matrix<T>::Diagonal(diag_vals.to_span(), batch_size);
    Matrix<T> tmp(n, n, batch_size);
    Matrix<T> A(n, n, batch_size);

    // A = U * S * V^H
    gemm<B>(ctx, U.view(), S.view(), tmp.view(), T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
    const Transpose v_trans = detail::is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;
    gemm<B>(ctx, tmp.view(), V.view(), A.view(), T(1), T(0), Transpose::NoTrans, v_trans);
    ctx.wait();

    return A;
}

// Create a batch of random symmetric/Hermitian matrices with a specified conditioning metric.
template <Backend B, typename T>
Matrix<T, MatrixFormat::Dense> random_hermitian_with_log10_cond_metric(Queue &ctx,
                                                                       int n,
                                                                       float_t<T> log10_kappa,
                                                                       NormType metric,
                                                                       int batch_size,
                                                                       unsigned int seed,
                                                                       OrthoAlgorithm algo) {
    using real_t = float_t<T>;
    if (n <= 0 || batch_size <= 0) {
        throw std::runtime_error("random_hermitian_with_log10_cond_metric: n and batch_size must be positive");
    }
    if (log10_kappa < real_t(0)) {
        throw std::runtime_error("random_hermitian_with_log10_cond_metric: log10_kappa must be non-negative");
    }

    if constexpr (B == Backend::NETLIB) {
        UnifiedVector<T> diag_vals = detail::build_spectrum_for_metric<T>(
            n, log10_kappa, metric, "random_hermitian_with_log10_cond_metric");
        return Matrix<T>::Diagonal(diag_vals.to_span(), batch_size);
    }

    const OrthoAlgorithm ortho_algo = detail::effective_ortho_algo<B>(algo);

    Matrix<T> Q = Matrix<T>::Random(n, n, false, batch_size, seed);
    const size_t ortho_ws = ortho_buffer_size<B>(ctx, Q.view(), Transpose::NoTrans, ortho_algo);
    UnifiedVector<std::byte> workspace(ortho_ws);
    ortho<B>(ctx, Q.view(), Transpose::NoTrans, workspace.to_span(), ortho_algo).wait();

    UnifiedVector<T> diag_vals = detail::build_spectrum_for_metric<T>(
        n, log10_kappa, metric, "random_hermitian_with_log10_cond_metric");

    Matrix<T> D = Matrix<T>::Diagonal(diag_vals.to_span(), batch_size);
    Matrix<T> tmp(n, n, batch_size);
    Matrix<T> A(n, n, batch_size);

    gemm<B>(ctx, Q.view(), D.view(), tmp.view(), T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
    const Transpose q_trans = detail::is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;
    gemm<B>(ctx, tmp.view(), Q.view(), A.view(), T(1), T(0), Transpose::NoTrans, q_trans);
    ctx.wait();

    return A;
}

// Create a batch of random dense banded matrices with a specified log10 conditioning metric.
template <Backend B, typename T>
Matrix<T, MatrixFormat::Dense> random_banded_with_log10_cond_metric(Queue &ctx,
                                                                    int n,
                                                                    int kd,
                                                                    float_t<T> log10_kappa,
                                                                    NormType metric,
                                                                    int batch_size,
                                                                    unsigned int seed) {
    using real_t = float_t<T>;
    if (n <= 0 || batch_size <= 0) {
        throw std::runtime_error("random_banded_with_log10_cond_metric: n and batch_size must be positive");
    }
    if (kd < 0) {
        throw std::runtime_error("random_banded_with_log10_cond_metric: kd must be non-negative");
    }
    if (log10_kappa < real_t(0)) {
        throw std::runtime_error("random_banded_with_log10_cond_metric: log10_kappa must be non-negative");
    }
    if (kd >= n - 1) {
        return random_with_log10_cond_metric<B, T>(ctx, n, log10_kappa, metric, batch_size, seed);
    }

    if constexpr (B == Backend::NETLIB) {
        UnifiedVector<T> diag_vals = detail::build_spectrum_for_metric<T>(
            n, log10_kappa, metric, "random_banded_with_log10_cond_metric");
        return Matrix<T>::Diagonal(diag_vals.to_span(), batch_size);
    }

    const int band = kd / 2;
    Matrix<T> Q = detail::banded_orthogonal<T>(n, band, batch_size, seed);
    Matrix<T> R = detail::banded_orthogonal<T>(n, band, batch_size, seed + 1);

    UnifiedVector<T> diag_vals = detail::build_spectrum_for_metric<T>(
        n, log10_kappa, metric, "random_banded_with_log10_cond_metric");
    Matrix<T> D = Matrix<T>::Diagonal(diag_vals.to_span(), batch_size);
    Matrix<T> tmp(n, n, batch_size);
    Matrix<T> A(n, n, batch_size);

    gemm<B>(ctx, Q.view(), D.view(), tmp.view(), T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
    const Transpose r_trans = detail::is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;
    gemm<B>(ctx, tmp.view(), R.view(), A.view(), T(1), T(0), Transpose::NoTrans, r_trans);
    ctx.wait();

    return A;
}

// Create a batch of random symmetric/Hermitian banded matrices with a specified conditioning metric.
template <Backend B, typename T>
Matrix<T, MatrixFormat::Dense> random_hermitian_banded_with_log10_cond_metric(Queue &ctx,
                                                                              int n,
                                                                              int kd,
                                                                              float_t<T> log10_kappa,
                                                                              NormType metric,
                                                                              int batch_size,
                                                                              unsigned int seed) {
    using real_t = float_t<T>;
    if (n <= 0 || batch_size <= 0) {
        throw std::runtime_error("random_hermitian_banded_with_log10_cond_metric: n and batch_size must be positive");
    }
    if (kd < 0) {
        throw std::runtime_error("random_hermitian_banded_with_log10_cond_metric: kd must be non-negative");
    }
    if (log10_kappa < real_t(0)) {
        throw std::runtime_error("random_hermitian_banded_with_log10_cond_metric: log10_kappa must be non-negative");
    }
    if (kd >= n - 1) {
        return random_hermitian_with_log10_cond_metric<B, T>(ctx, n, log10_kappa, metric, batch_size, seed);
    }

    if constexpr (B == Backend::NETLIB) {
        UnifiedVector<T> diag_vals = detail::build_spectrum_for_metric<T>(
            n, log10_kappa, metric, "random_hermitian_banded_with_log10_cond_metric");
        return Matrix<T>::Diagonal(diag_vals.to_span(), batch_size);
    }

    const int band = kd / 2;
    Matrix<T> Q = detail::banded_orthogonal<T>(n, band, batch_size, seed);

    UnifiedVector<T> diag_vals = detail::build_spectrum_for_metric<T>(
        n, log10_kappa, metric, "random_hermitian_banded_with_log10_cond_metric");

    Matrix<T> D = Matrix<T>::Diagonal(diag_vals.to_span(), batch_size);
    Matrix<T> tmp(n, n, batch_size);
    Matrix<T> A(n, n, batch_size);

    gemm<B>(ctx, Q.view(), D.view(), tmp.view(), T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
    const Transpose q_trans = detail::is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;
    gemm<B>(ctx, tmp.view(), Q.view(), A.view(), T(1), T(0), Transpose::NoTrans, q_trans);
    ctx.wait();

    return A;
}

// Create a batch of random tridiagonal matrices with a specified log10 conditioning metric.
template <Backend B, typename T>
Matrix<T, MatrixFormat::Dense> random_tridiagonal_with_log10_cond_metric(Queue &ctx,
                                                                         int n,
                                                                         float_t<T> log10_kappa,
                                                                         NormType metric,
                                                                         int batch_size,
                                                                         unsigned int /*seed*/) {
    using real_t = float_t<T>;
    if (n <= 0 || batch_size <= 0) {
        throw std::runtime_error("random_tridiagonal_with_log10_cond_metric: n and batch_size must be positive");
    }
    if (log10_kappa < real_t(0)) {
        throw std::runtime_error("random_tridiagonal_with_log10_cond_metric: log10_kappa must be non-negative");
    }

    UnifiedVector<T> diag_vals = detail::build_spectrum_for_metric<T>(
        n, log10_kappa, metric, "random_tridiagonal_with_log10_cond_metric");
    Matrix<T> A = Matrix<T>::Zeros(n, n, batch_size);
    A.view().template fill_diagonal<MatrixFormat::Dense>(ctx, diag_vals.to_span()).wait();
    return A;
}

// Create a batch of random symmetric/Hermitian tridiagonal matrices with a specified conditioning metric.
template <Backend B, typename T>
Matrix<T, MatrixFormat::Dense> random_hermitian_tridiagonal_with_log10_cond_metric(Queue &ctx,
                                                                                   int n,
                                                                                   float_t<T> log10_kappa,
                                                                                   NormType metric,
                                                                                   int batch_size,
                                                                                   unsigned int seed) {
    using real_t = float_t<T>;
    if constexpr (std::is_same_v<T, float>) {
        if (metric == NormType::Spectral) {
            auto mat_d = random_hermitian_tridiagonal_with_log10_cond_metric<B, double>(
                ctx, n, static_cast<double>(log10_kappa), metric, batch_size, seed);
            return mat_d.template astype<float>();
        }
    }
    if (n <= 0 || batch_size <= 0) {
        throw std::runtime_error("random_hermitian_tridiagonal_with_log10_cond_metric: n and batch_size must be positive");
    }
    if (log10_kappa < real_t(0)) {
        throw std::runtime_error("random_hermitian_tridiagonal_with_log10_cond_metric: log10_kappa must be non-negative");
    }

    if constexpr (B == Backend::NETLIB) {
        UnifiedVector<T> diag_vals = detail::build_spectrum_for_metric<T>(
            n, log10_kappa, metric, "random_hermitian_tridiagonal_with_log10_cond_metric");
        Matrix<T> A = Matrix<T>::Zeros(n, n, batch_size);
        A.view().template fill_diagonal<MatrixFormat::Dense>(ctx, diag_vals.to_span()).wait();
        return A;
    }

    const OrthoAlgorithm ortho_algo = OrthoAlgorithm::SVQB2;

    Matrix<T> Q = Matrix<T>::Random(n, n, false, batch_size, seed);
    const size_t ortho_ws = ortho_buffer_size<B>(ctx, Q.view(), Transpose::NoTrans, ortho_algo);
    UnifiedVector<std::byte> workspace(ortho_ws);
    ortho<B>(ctx, Q.view(), Transpose::NoTrans, workspace.to_span(), ortho_algo).wait();

    UnifiedVector<T> diag_vals = detail::build_spectrum_for_metric<T>(
        n, log10_kappa, metric, "random_hermitian_tridiagonal_with_log10_cond_metric");

    Matrix<T> D = Matrix<T>::Diagonal(diag_vals.to_span(), batch_size);
    Matrix<T> tmp(n, n, batch_size);
    Matrix<T> A(n, n, batch_size);

    gemm<B>(ctx, Q.view(), D.view(), tmp.view(), T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
    const Transpose q_trans = detail::is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;
    gemm<B>(ctx, tmp.view(), Q.view(), A.view(), T(1), T(0), Transpose::NoTrans, q_trans);
    ctx.wait();

    constexpr int block_size = 32;
    return detail::reduce_hermitian_to_tridiagonal<B, T>(ctx, A, block_size);
}

#define RANDOM_LOGCOND_METRIC_INSTANTIATE(back, fp) \
    template Matrix<fp, MatrixFormat::Dense> random_with_log10_cond_metric<back, fp>( \
        Queue&, int, float_t<fp>, NormType, int, unsigned int, OrthoAlgorithm);

#define RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE(back, fp) \
    template Matrix<fp, MatrixFormat::Dense> random_hermitian_with_log10_cond_metric<back, fp>( \
        Queue&, int, float_t<fp>, NormType, int, unsigned int, OrthoAlgorithm);

#define RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE(back, fp) \
    template Matrix<fp, MatrixFormat::Dense> random_banded_with_log10_cond_metric<back, fp>( \
        Queue&, int, int, float_t<fp>, NormType, int, unsigned int);

#define RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE(back, fp) \
    template Matrix<fp, MatrixFormat::Dense> random_hermitian_banded_with_log10_cond_metric<back, fp>( \
        Queue&, int, int, float_t<fp>, NormType, int, unsigned int);

#define RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(back, fp) \
    template Matrix<fp, MatrixFormat::Dense> random_tridiagonal_with_log10_cond_metric<back, fp>( \
        Queue&, int, float_t<fp>, NormType, int, unsigned int);

#define RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(back, fp) \
    template Matrix<fp, MatrixFormat::Dense> random_hermitian_tridiagonal_with_log10_cond_metric<back, fp>( \
        Queue&, int, float_t<fp>, NormType, int, unsigned int);

#if BATCHLAS_HAS_CUDA_BACKEND
    RANDOM_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, float)
    RANDOM_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, double)
    RANDOM_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, std::complex<float>)
    RANDOM_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, std::complex<double>)
    RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, float)
    RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, double)
    RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, std::complex<float>)
    RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, std::complex<double>)
    RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, float)
    RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, double)
    RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, std::complex<float>)
    RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, std::complex<double>)
    RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, float)
    RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, double)
    RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, std::complex<float>)
    RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, std::complex<double>)
    RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, float)
    RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, double)
    RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, std::complex<float>)
    RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, std::complex<double>)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, float)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, double)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, std::complex<float>)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    RANDOM_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, float)
    RANDOM_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, double)
    RANDOM_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, std::complex<float>)
    RANDOM_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, std::complex<double>)
    RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, float)
    RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, double)
    RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, std::complex<float>)
    RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, std::complex<double>)
    RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, float)
    RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, double)
    RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, std::complex<float>)
    RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, std::complex<double>)
    RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, float)
    RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, double)
    RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, std::complex<float>)
    RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, std::complex<double>)
    RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, float)
    RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, double)
    RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, std::complex<float>)
    RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, std::complex<double>)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, float)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, double)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, std::complex<float>)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::ROCM, std::complex<double>)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
    RANDOM_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, float)
    RANDOM_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, double)
    RANDOM_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, std::complex<float>)
    RANDOM_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, std::complex<double>)
    RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, float)
    RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, double)
    RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, std::complex<float>)
    RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, std::complex<double>)
    RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, float)
    RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, double)
    RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, std::complex<float>)
    RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, std::complex<double>)
    RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, float)
    RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, double)
    RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, std::complex<float>)
    RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, std::complex<double>)
    RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, float)
    RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, double)
    RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, std::complex<float>)
    RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, std::complex<double>)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, float)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, double)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, std::complex<float>)
    RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

} // namespace batchlas
