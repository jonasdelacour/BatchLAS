// SPDX-License-Identifier: MIT
#include <batchlas/blas/extra.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/util/template-instantiations.hh>
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
template <typename Scalar>
using real_t = float_t<Scalar>;

template <typename Scalar>
inline Scalar real_to_T(real_t<Scalar> v) {
    if constexpr (is_std_complex_v<Scalar>) {
        return Scalar(v, real_t<Scalar>(0));
    } else {
        return Scalar(v);
    }
}

template <typename Scalar>
UnifiedVector<Scalar> build_spectrum_kappa2(int n, real_t<Scalar> log10_kappa2) {
    UnifiedVector<Scalar> diag_vals(n);
    if (n == 1) {
        diag_vals[0] = real_to_T<Scalar>(real_t<Scalar>(1));
        return diag_vals;
    }

    const long double log_kappa2 = std::log(10.0L) * static_cast<long double>(log10_kappa2);
    const long double t = log_kappa2 / static_cast<long double>(n - 1);
    const long double m = static_cast<long double>(n - 1) * 0.5L;
    for (int i = 0; i < n; ++i) {
        const long double exp_arg = (static_cast<long double>(i) - m) * t;
        const long double s = std::exp(exp_arg);
        diag_vals[i] = real_to_T<Scalar>(static_cast<real_t<Scalar>>(s));
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

template <typename Scalar>
UnifiedVector<Scalar> build_spectrum_kappaF(int n, real_t<Scalar> log10_kappaF) {
    UnifiedVector<Scalar> diag_vals(n);
    if (n == 1) {
        diag_vals[0] = real_to_T<Scalar>(real_t<Scalar>(1));
        return diag_vals;
    }

    const long double log10_n = std::log10(static_cast<long double>(n));
    if (static_cast<long double>(log10_kappaF) < log10_n) {
        throw batchlas::invalid_argument("build_spectrum_kappaF: log10_kappaF must be >= log10(n)");
    }

    const long double target = std::log(10.0L) * static_cast<long double>(log10_kappaF);
    const long double log_kappa0 = std::log(static_cast<long double>(n));
    if (target <= log_kappa0) {
        for (int i = 0; i < n; ++i) {
            diag_vals[i] = real_to_T<Scalar>(real_t<Scalar>(1));
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
        diag_vals[i] = real_to_T<Scalar>(static_cast<real_t<Scalar>>(s));
    }
    return diag_vals;
}

template <typename Scalar>
UnifiedVector<Scalar> build_spectrum_for_metric(int n,
                                                real_t<Scalar> log10_kappa,
                                                NormType metric,
                                                const char* func_name) {
    switch (metric) {
        case NormType::Spectral:
            return build_spectrum_kappa2<Scalar>(n, log10_kappa);
        case NormType::Frobenius:
            return build_spectrum_kappaF<Scalar>(n, log10_kappa);
        default:
            throw batchlas::invalid_argument(std::string(func_name) + ": metric must be NormType::Spectral or NormType::Frobenius");
    }
}

template <Backend B, typename Scalar>
Matrix<Scalar, MatrixFormat::Dense> reduce_hermitian_to_tridiagonal(Queue& ctx,
                                                                    Matrix<Scalar, MatrixFormat::Dense>& A,
                                                                    int block_size) {
    const int n = A.rows();
    const int batch_size = A.batch_size();
    Vector<Scalar> d(n, batch_size);
    Vector<Scalar> e(std::max(0, n - 1), batch_size);
    Vector<Scalar> tau(std::max(0, n - 1), batch_size);

    auto dv = static_cast<VectorView<Scalar>>(d);
    auto ev = static_cast<VectorView<Scalar>>(e);
    auto tauv = static_cast<VectorView<Scalar>>(tau);

    const size_t ws_bytes = sytrd_blocked_buffer_size<B, Scalar>(ctx, A, dv, ev, tauv, Uplo::Lower, block_size);
    auto ws = ctx.workspace(ws_bytes);
    sytrd_blocked<B, Scalar>(ctx, A, dv, ev, tauv, Uplo::Lower, ws.span(), block_size).wait();

    Matrix<Scalar, MatrixFormat::Dense> Tmat = Matrix<Scalar, MatrixFormat::Dense>::Zeros(n, n, batch_size);
    auto Tv = Tmat.view();
    Tv.template fill_diagonal<MatrixFormat::Dense>(ctx, dv).wait();

    if (n > 1) {
        Vector<Scalar> e_conj(n - 1, batch_size);
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < n - 1; ++i) {
                if constexpr (is_std_complex_v<Scalar>) {
                    e_conj(i, b) = std::conj(e(i, b));
                } else {
                    e_conj(i, b) = e(i, b);
                }
            }
        }

        auto ev_conj = static_cast<VectorView<Scalar>>(e_conj);
        Tv.template fill_diagonal<MatrixFormat::Dense>(ctx, ev, -1).wait();
        Tv.template fill_diagonal<MatrixFormat::Dense>(ctx, ev_conj, 1).wait();
    }

    return Tmat;
}

template <typename Scalar>
void apply_givens_rows(Matrix<Scalar, MatrixFormat::Dense>& Q,
                       int row1,
                       int row2,
                       int batch,
                       real_t<Scalar> c,
                       real_t<Scalar> s) {
    const int n = Q.cols();
    const Scalar tc = real_to_T<Scalar>(c);
    const Scalar ts = real_to_T<Scalar>(s);
    for (int col = 0; col < n; ++col) {
        const Scalar q1 = Q(row1, col, batch);
        const Scalar q2 = Q(row2, col, batch);
        Q(row1, col, batch) = tc * q1 + ts * q2;
        Q(row2, col, batch) = real_to_T<Scalar>(-s) * q1 + tc * q2;
    }
}

template <typename Scalar>
Matrix<Scalar, MatrixFormat::Dense> banded_orthogonal(int n,
                                                      int bandwidth,
                                                      int batch_size,
                                                      unsigned seed) {
    Matrix<Scalar, MatrixFormat::Dense> Q = Matrix<Scalar, MatrixFormat::Dense>::Identity(n, batch_size);
    if (bandwidth <= 0) return Q;

    const real_t<Scalar> two_pi = real_t<Scalar>(2) * std::acos(real_t<Scalar>(-1));
    for (int b = 0; b < batch_size; ++b) {
        std::mt19937 rng(seed + 1315423911u * static_cast<unsigned>(b));
        std::uniform_real_distribution<real_t<Scalar>> angle_dist(real_t<Scalar>(0), two_pi);

        const int offset = static_cast<int>(rng() % static_cast<unsigned>(bandwidth + 1));
        for (int i = offset; i + bandwidth < n; i += (bandwidth + 1)) {
            const real_t<Scalar> theta = angle_dist(rng);
            const real_t<Scalar> c = std::cos(theta);
            const real_t<Scalar> s = std::sin(theta);
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
}  // namespace detail

// Create a batch of random dense matrices with a specified log10 conditioning metric.
template <Backend B, typename Scalar>
Matrix<Scalar, MatrixFormat::Dense> random_with_log10_cond_metric(Queue& ctx,
                                                                  int n,
                                                                  float_t<Scalar> log10_kappa,
                                                                  NormType metric,
                                                                  int batch_size,
                                                                  unsigned seed,
                                                                  OrthoAlgorithm algo) {
    using real_t = float_t<Scalar>;

    if (n <= 0 || batch_size <= 0) {
        throw batchlas::invalid_argument("random_with_log10_cond_metric: n and batch_size must be positive");
    }
    if (log10_kappa < real_t(0)) {
        throw batchlas::invalid_argument("random_with_log10_cond_metric: log10_kappa must be non-negative");
    }

    if constexpr (B == Backend::NETLIB) {
        UnifiedVector<Scalar> diag_vals = detail::build_spectrum_for_metric<Scalar>(
            n, log10_kappa, metric, "random_with_log10_cond_metric");
        return Matrix<Scalar>::Diagonal(diag_vals.to_span(), batch_size);
    }

    const OrthoAlgorithm ortho_algo = detail::effective_ortho_algo<B>(algo);

    // Random matrices for orthogonal factors
    Matrix<Scalar> U = Matrix<Scalar>::Random(n, n, false, batch_size, seed);
    Matrix<Scalar> V = Matrix<Scalar>::Random(n, n, false, batch_size, seed + 1);

    // Orthonormalize columns of U and V
    const size_t ortho_ws = ortho_buffer_size<B>(ctx, U, Transpose::NoTrans, ortho_algo);
    auto workspace = ctx.workspace(ortho_ws);
    ortho<B>(ctx, U, Transpose::NoTrans, workspace.span(), ortho_algo).wait();
    ortho<B>(ctx, V, Transpose::NoTrans, workspace.span(), ortho_algo).wait();

    // Build singular values with the requested condition number.
    UnifiedVector<Scalar> diag_vals = detail::build_spectrum_for_metric<Scalar>(
        n, log10_kappa, metric, "random_with_log10_cond_metric");

    Matrix<Scalar> S = Matrix<Scalar>::Diagonal(diag_vals.to_span(), batch_size);
    Matrix<Scalar> tmp(n, n, batch_size);
    Matrix<Scalar> A(n, n, batch_size);

    // A = U * S * V^H
    // (void) on an Event: deliberate. This Queue is in-order, so the next submission
    // is already ordered after this one and the Event carries nothing the caller needs.
    (void)gemm<B>(ctx, U, S, tmp, GemmOptions<Scalar>{});
    const Transpose v_trans = batchlas::is_std_complex_v<Scalar> ? Transpose::ConjTrans : Transpose::Trans;
    (void)gemm<B>(ctx, tmp, V, A, {.transB = v_trans});
    ctx.wait();

    return A;
}

// Create a batch of random symmetric/Hermitian matrices with a specified conditioning metric.
template <Backend B, typename Scalar>
Matrix<Scalar, MatrixFormat::Dense> random_hermitian_with_log10_cond_metric(Queue& ctx,
                                                                            int n,
                                                                            float_t<Scalar> log10_kappa,
                                                                            NormType metric,
                                                                            int batch_size,
                                                                            unsigned seed,
                                                                            OrthoAlgorithm algo) {
    using real_t = float_t<Scalar>;
    if (n <= 0 || batch_size <= 0) {
        throw batchlas::invalid_argument("random_hermitian_with_log10_cond_metric: n and batch_size must be positive");
    }
    if (log10_kappa < real_t(0)) {
        throw batchlas::invalid_argument("random_hermitian_with_log10_cond_metric: log10_kappa must be non-negative");
    }

    if constexpr (B == Backend::NETLIB) {
        UnifiedVector<Scalar> diag_vals = detail::build_spectrum_for_metric<Scalar>(
            n, log10_kappa, metric, "random_hermitian_with_log10_cond_metric");
        return Matrix<Scalar>::Diagonal(diag_vals.to_span(), batch_size);
    }

    const OrthoAlgorithm ortho_algo = detail::effective_ortho_algo<B>(algo);

    Matrix<Scalar> Q = Matrix<Scalar>::Random(n, n, false, batch_size, seed);
    const size_t ortho_ws = ortho_buffer_size<B>(ctx, Q, Transpose::NoTrans, ortho_algo);
    auto workspace = ctx.workspace(ortho_ws);
    ortho<B>(ctx, Q, Transpose::NoTrans, workspace.span(), ortho_algo).wait();

    UnifiedVector<Scalar> diag_vals = detail::build_spectrum_for_metric<Scalar>(
        n, log10_kappa, metric, "random_hermitian_with_log10_cond_metric");

    Matrix<Scalar> D = Matrix<Scalar>::Diagonal(diag_vals.to_span(), batch_size);
    Matrix<Scalar> tmp(n, n, batch_size);
    Matrix<Scalar> A(n, n, batch_size);

    (void)gemm<B>(ctx, Q, D, tmp, GemmOptions<Scalar>{});
    const Transpose q_trans = batchlas::is_std_complex_v<Scalar> ? Transpose::ConjTrans : Transpose::Trans;
    (void)gemm<B>(ctx, tmp, Q, A, {.transB = q_trans});
    ctx.wait();

    return A;
}

// Create a batch of random dense banded matrices with a specified log10 conditioning metric.
template <Backend B, typename Scalar>
Matrix<Scalar, MatrixFormat::Dense> random_banded_with_log10_cond_metric(Queue& ctx,
                                                                         int n,
                                                                         int kd,
                                                                         float_t<Scalar> log10_kappa,
                                                                         NormType metric,
                                                                         int batch_size,
                                                                         unsigned seed) {
    using real_t = float_t<Scalar>;
    if (n <= 0 || batch_size <= 0) {
        throw batchlas::invalid_argument("random_banded_with_log10_cond_metric: n and batch_size must be positive");
    }
    if (kd < 0) {
        throw batchlas::invalid_argument("random_banded_with_log10_cond_metric: kd must be non-negative");
    }
    if (log10_kappa < real_t(0)) {
        throw batchlas::invalid_argument("random_banded_with_log10_cond_metric: log10_kappa must be non-negative");
    }
    if (kd >= n - 1) {
        return random_with_log10_cond_metric<B, Scalar>(ctx, n, log10_kappa, metric, batch_size, seed);
    }

    if constexpr (B == Backend::NETLIB) {
        UnifiedVector<Scalar> diag_vals = detail::build_spectrum_for_metric<Scalar>(
            n, log10_kappa, metric, "random_banded_with_log10_cond_metric");
        return Matrix<Scalar>::Diagonal(diag_vals.to_span(), batch_size);
    }

    const int band = kd / 2;
    Matrix<Scalar> Q = detail::banded_orthogonal<Scalar>(n, band, batch_size, seed);
    Matrix<Scalar> R = detail::banded_orthogonal<Scalar>(n, band, batch_size, seed + 1);

    UnifiedVector<Scalar> diag_vals = detail::build_spectrum_for_metric<Scalar>(
        n, log10_kappa, metric, "random_banded_with_log10_cond_metric");
    Matrix<Scalar> D = Matrix<Scalar>::Diagonal(diag_vals.to_span(), batch_size);
    Matrix<Scalar> tmp(n, n, batch_size);
    Matrix<Scalar> A(n, n, batch_size);

    (void)gemm<B>(ctx, Q, D, tmp, GemmOptions<Scalar>{});
    const Transpose r_trans = batchlas::is_std_complex_v<Scalar> ? Transpose::ConjTrans : Transpose::Trans;
    (void)gemm<B>(ctx, tmp, R, A, {.transB = r_trans});
    ctx.wait();

    return A;
}

// Create a batch of random symmetric/Hermitian banded matrices with a specified conditioning metric.
template <Backend B, typename Scalar>
Matrix<Scalar, MatrixFormat::Dense> random_hermitian_banded_with_log10_cond_metric(Queue& ctx,
                                                                                   int n,
                                                                                   int kd,
                                                                                   float_t<Scalar> log10_kappa,
                                                                                   NormType metric,
                                                                                   int batch_size,
                                                                                   unsigned seed) {
    using real_t = float_t<Scalar>;
    if (n <= 0 || batch_size <= 0) {
        throw batchlas::invalid_argument("random_hermitian_banded_with_log10_cond_metric: n and batch_size must be positive");
    }
    if (kd < 0) {
        throw batchlas::invalid_argument("random_hermitian_banded_with_log10_cond_metric: kd must be non-negative");
    }
    if (log10_kappa < real_t(0)) {
        throw batchlas::invalid_argument("random_hermitian_banded_with_log10_cond_metric: log10_kappa must be non-negative");
    }
    if (kd >= n - 1) {
        return random_hermitian_with_log10_cond_metric<B, Scalar>(ctx, n, log10_kappa, metric, batch_size, seed);
    }

    if constexpr (B == Backend::NETLIB) {
        UnifiedVector<Scalar> diag_vals = detail::build_spectrum_for_metric<Scalar>(
            n, log10_kappa, metric, "random_hermitian_banded_with_log10_cond_metric");
        return Matrix<Scalar>::Diagonal(diag_vals.to_span(), batch_size);
    }

    const int band = kd / 2;
    Matrix<Scalar> Q = detail::banded_orthogonal<Scalar>(n, band, batch_size, seed);

    UnifiedVector<Scalar> diag_vals = detail::build_spectrum_for_metric<Scalar>(
        n, log10_kappa, metric, "random_hermitian_banded_with_log10_cond_metric");

    Matrix<Scalar> D = Matrix<Scalar>::Diagonal(diag_vals.to_span(), batch_size);
    Matrix<Scalar> tmp(n, n, batch_size);
    Matrix<Scalar> A(n, n, batch_size);

    (void)gemm<B>(ctx, Q, D, tmp, GemmOptions<Scalar>{});
    const Transpose q_trans = batchlas::is_std_complex_v<Scalar> ? Transpose::ConjTrans : Transpose::Trans;
    (void)gemm<B>(ctx, tmp, Q, A, {.transB = q_trans});
    ctx.wait();

    return A;
}

// Create a batch of random tridiagonal matrices with a specified log10 conditioning metric.
template <Backend B, typename Scalar>
Matrix<Scalar, MatrixFormat::Dense> random_tridiagonal_with_log10_cond_metric(Queue& ctx,
                                                                              int n,
                                                                              float_t<Scalar> log10_kappa,
                                                                              NormType metric,
                                                                              int batch_size,
                                                                              unsigned /*seed*/) {
    using real_t = float_t<Scalar>;
    if (n <= 0 || batch_size <= 0) {
        throw batchlas::invalid_argument("random_tridiagonal_with_log10_cond_metric: n and batch_size must be positive");
    }
    if (log10_kappa < real_t(0)) {
        throw batchlas::invalid_argument("random_tridiagonal_with_log10_cond_metric: log10_kappa must be non-negative");
    }

    UnifiedVector<Scalar> diag_vals = detail::build_spectrum_for_metric<Scalar>(
        n, log10_kappa, metric, "random_tridiagonal_with_log10_cond_metric");
    Matrix<Scalar> A = Matrix<Scalar>::Zeros(n, n, batch_size);
    A.template fill_diagonal<MatrixFormat::Dense>(ctx, diag_vals.to_span()).wait();
    return A;
}

// Create a batch of random symmetric/Hermitian tridiagonal matrices with a specified conditioning metric.
template <Backend B, typename Scalar>
Matrix<Scalar, MatrixFormat::Dense> random_hermitian_tridiagonal_with_log10_cond_metric(Queue& ctx,
                                                                                        int n,
                                                                                        float_t<Scalar> log10_kappa,
                                                                                        NormType metric,
                                                                                        int batch_size,
                                                                                        unsigned seed) {
    using real_t = float_t<Scalar>;
    if constexpr (std::is_same_v<Scalar, float>) {
        if (metric == NormType::Spectral) {
            if constexpr (B != Backend::SYCL || BATCHLAS_HAS_DEVICE_FP64) {
                auto mat_d = random_hermitian_tridiagonal_with_log10_cond_metric<B, double>(
                    ctx, n, static_cast<double>(log10_kappa), metric, batch_size, seed);
                return mat_d.template astype<float>();
            } else if constexpr (BATCHLAS_HAS_HOST_BACKEND) {
                // No double kernels on this device: widen on the host backend instead.
                ctx.wait();
                static Queue* host = new Queue(Device("cpu"), Backend::NETLIB);  // leaked on purpose
                auto mat_d = random_hermitian_tridiagonal_with_log10_cond_metric<Backend::NETLIB, double>(
                    *host, n, static_cast<double>(log10_kappa), metric, batch_size, seed);
                return mat_d.template astype<float>();
            }
        }
    }
    if (n <= 0 || batch_size <= 0) {
        throw batchlas::invalid_argument("random_hermitian_tridiagonal_with_log10_cond_metric: n and batch_size must be positive");
    }
    if (log10_kappa < real_t(0)) {
        throw batchlas::invalid_argument("random_hermitian_tridiagonal_with_log10_cond_metric: log10_kappa must be non-negative");
    }

    if constexpr (B == Backend::NETLIB) {
        UnifiedVector<Scalar> diag_vals = detail::build_spectrum_for_metric<Scalar>(
            n, log10_kappa, metric, "random_hermitian_tridiagonal_with_log10_cond_metric");
        Matrix<Scalar> A = Matrix<Scalar>::Zeros(n, n, batch_size);
        A.template fill_diagonal<MatrixFormat::Dense>(ctx, diag_vals.to_span()).wait();
        return A;
    }

    const OrthoAlgorithm ortho_algo = OrthoAlgorithm::SVQB2;

    Matrix<Scalar> Q = Matrix<Scalar>::Random(n, n, false, batch_size, seed);
    const size_t ortho_ws = ortho_buffer_size<B>(ctx, Q, Transpose::NoTrans, ortho_algo);
    auto workspace = ctx.workspace(ortho_ws);
    ortho<B>(ctx, Q, Transpose::NoTrans, workspace.span(), ortho_algo).wait();

    UnifiedVector<Scalar> diag_vals = detail::build_spectrum_for_metric<Scalar>(
        n, log10_kappa, metric, "random_hermitian_tridiagonal_with_log10_cond_metric");

    Matrix<Scalar> D = Matrix<Scalar>::Diagonal(diag_vals.to_span(), batch_size);
    Matrix<Scalar> tmp(n, n, batch_size);
    Matrix<Scalar> A(n, n, batch_size);

    (void)gemm<B>(ctx, Q, D, tmp, GemmOptions<Scalar>{});
    const Transpose q_trans = batchlas::is_std_complex_v<Scalar> ? Transpose::ConjTrans : Transpose::Trans;
    (void)gemm<B>(ctx, tmp, Q, A, {.transB = q_trans});
    ctx.wait();

    constexpr int block_size = 32;
    return detail::reduce_hermitian_to_tridiagonal<B, Scalar>(ctx, A, block_size);
}

#define RANDOM_LOGCOND_METRIC_INSTANTIATE(back, fp) \
    template Matrix<BATCHLAS_UNPAREN fp, MatrixFormat::Dense> random_with_log10_cond_metric<back, BATCHLAS_UNPAREN fp>( \
        Queue&, int, float_t<BATCHLAS_UNPAREN fp>, NormType, int, unsigned, OrthoAlgorithm);

#define RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE(back, fp) \
    template Matrix<BATCHLAS_UNPAREN fp, MatrixFormat::Dense> random_hermitian_with_log10_cond_metric<back, BATCHLAS_UNPAREN fp>( \
        Queue&, int, float_t<BATCHLAS_UNPAREN fp>, NormType, int, unsigned, OrthoAlgorithm);

#define RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE(back, fp) \
    template Matrix<BATCHLAS_UNPAREN fp, MatrixFormat::Dense> random_banded_with_log10_cond_metric<back, BATCHLAS_UNPAREN fp>( \
        Queue&, int, int, float_t<BATCHLAS_UNPAREN fp>, NormType, int, unsigned);

#define RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE(back, fp) \
    template Matrix<BATCHLAS_UNPAREN fp, MatrixFormat::Dense> random_hermitian_banded_with_log10_cond_metric<back, BATCHLAS_UNPAREN fp>( \
        Queue&, int, int, float_t<BATCHLAS_UNPAREN fp>, NormType, int, unsigned);

#define RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(back, fp) \
    template Matrix<BATCHLAS_UNPAREN fp, MatrixFormat::Dense> random_tridiagonal_with_log10_cond_metric<back, BATCHLAS_UNPAREN fp>( \
        Queue&, int, float_t<BATCHLAS_UNPAREN fp>, NormType, int, unsigned);

#define RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE(back, fp) \
    template Matrix<BATCHLAS_UNPAREN fp, MatrixFormat::Dense> random_hermitian_tridiagonal_with_log10_cond_metric<back, BATCHLAS_UNPAREN fp>( \
        Queue&, int, float_t<BATCHLAS_UNPAREN fp>, NormType, int, unsigned);

BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(RANDOM_LOGCOND_METRIC_INSTANTIATE)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(RANDOM_HERMITIAN_LOGCOND_METRIC_INSTANTIATE)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(RANDOM_BANDED_LOGCOND_METRIC_INSTANTIATE)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(RANDOM_HERMITIAN_BANDED_LOGCOND_METRIC_INSTANTIATE)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(RANDOM_TRIDIAG_LOGCOND_METRIC_INSTANTIATE)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(RANDOM_HERMITIAN_TRIDIAG_LOGCOND_METRIC_INSTANTIATE)

}  // namespace batchlas
