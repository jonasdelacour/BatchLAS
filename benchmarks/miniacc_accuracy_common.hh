// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/miniacc.hh>
#include <batchlas/backend_config.h>
#include <src/queue.hh>
#include <benchmarks/accuracy_utils.hh>

#include <batchlas/verify/reference.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace miniacc_acc {

using namespace batchlas;

template <typename Real>
inline const char* dtype_name() {
    if constexpr (std::is_same_v<Real, float>) return "float";
    if constexpr (std::is_same_v<Real, double>) return "double";
    if constexpr (std::is_same_v<Real, std::complex<float>>) return "complex64";
    return "complex128";
}

template <Backend B>
inline const char* backend_name() {
    if constexpr (B == Backend::CUDA) return "CUDA";
    if constexpr (B == Backend::ROCM) return "ROCM";
    if constexpr (B == Backend::MKL) return "MKL";
    return "NETLIB";
}

inline int chunk_batch_from_samples(size_t samples, int cap = 128) {
    if (samples == 0) return 1;
    const size_t capped = std::min(samples, static_cast<size_t>(std::max(1, cap)));
    return static_cast<int>(std::max<size_t>(1, capped));
}

inline void record_failed_samples(miniacc::State& state,
                                  int n,
                                  int neigs,
                                  int count,
                                  double log10_cond,
                                  const std::string& reason) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    (void)neigs;
    for (int i = 0; i < count; ++i) {
        state.RecordSample(
            {
                {"log10_cond", log10_cond},
                {"res_num", nan},
                {"res_denom", nan},
                {"ortho_num", nan},
                {"ortho_denom", static_cast<double>(n)},
                {"R", nan},
                {"O", nan},
                {"max_relerr", nan}
            },
            false,
            reason);
    }
}

using batchlas::accuracy::extract_tridiagonal;
using batchlas::accuracy::orthogonality_residuals;

template <Backend B, typename Real>
inline UnifiedVector<typename base_type<Real>::type> residual_residuals(
    Queue& q,
    const Matrix<Real, MatrixFormat::Dense>& A,
    const Matrix<Real, MatrixFormat::Dense>& Z,
    const VectorView<Real>& evals) {
    const int n = A.rows();
    const int m = Z.cols();
    const int batch = A.batch_size();
    auto R = Matrix<Real>::Zeros(n, m, batch);

    (void)gemm<B, Real>(q,
                  A,
                  Z,
                  R,
                  Real(1),
                  Real(0),
                  Transpose::NoTrans,
                  Transpose::NoTrans);

    auto r_view = R.kernel_view();
    auto z_view = Z.kernel_view();
    auto lambda_ptr = evals.data_ptr();
    const int lambda_inc = evals.inc();
    const int lambda_stride = evals.stride();

    q->parallel_for(sycl::range<3>(
                        static_cast<size_t>(batch),
                        static_cast<size_t>(n),
                        static_cast<size_t>(m)),
                    [=](sycl::id<3> idx) {
                        const int b = static_cast<int>(idx[0]);
                        const int i = static_cast<int>(idx[1]);
                        const int j = static_cast<int>(idx[2]);
                        const Real lambda = lambda_ptr[b * lambda_stride + j * lambda_inc];
                        r_view(i, j, b) -= z_view(i, j, b) * lambda;
                    });
    q.wait();

    return norm(q, R, NormType::Frobenius);
}

inline double max_relative_eig_error(const std::vector<double>& ref_sorted,
                                     const std::vector<double>& est_unsorted) {
    if (ref_sorted.empty() || ref_sorted.size() != est_unsorted.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::vector<double> est_sorted = est_unsorted;
    std::sort(est_sorted.begin(), est_sorted.end());

    double max_rel = 0.0;
    const double tiny = std::numeric_limits<double>::min();
    for (size_t i = 0; i < ref_sorted.size(); ++i) {
        const double ref = ref_sorted[i];
        const double est = est_sorted[i];
        if (!std::isfinite(ref) || !std::isfinite(est)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double denom = std::max(std::abs(ref), tiny);
        const double rel = std::abs(est - ref) / denom;
        if (!std::isfinite(rel)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        max_rel = std::max(max_rel, rel);
    }

    return max_rel;
}

template <typename Real>
inline void make_tridiag_reference(const VectorView<Real>& d,
                                   const VectorView<Real>& e,
                                   std::vector<std::vector<double>>& ref_eigs_sorted,
                                   std::vector<char>& ref_ok) {
    const int n = d.size();
    const int batch = d.batch_size();
    ref_eigs_sorted.assign(static_cast<size_t>(batch), std::vector<double>(static_cast<size_t>(n), std::numeric_limits<double>::quiet_NaN()));
    ref_ok.assign(static_cast<size_t>(batch), 0);

    for (int b = 0; b < batch; ++b) {
        std::vector<double> d_work(static_cast<size_t>(n));
        std::vector<double> e_work(static_cast<size_t>(std::max(0, n - 1)));
        for (int i = 0; i < n; ++i) {
            d_work[static_cast<size_t>(i)] = static_cast<double>(d(i, b));
            if (i < n - 1) e_work[static_cast<size_t>(i)] = static_cast<double>(e(i, b));
        }
        if (batchlas::verify::tridiagonal_eigenvalues(d_work, e_work)) {
            std::sort(d_work.begin(), d_work.end());
            ref_eigs_sorted[static_cast<size_t>(b)] = std::move(d_work);
            ref_ok[static_cast<size_t>(b)] = 1;
        }
    }
}

template <typename Real>
inline void make_dense_reference(const Matrix<Real, MatrixFormat::Dense>& A,
                                 std::vector<std::vector<double>>& ref_eigs_sorted,
                                 std::vector<char>& ref_ok) {
    const int n = A.rows();
    const int batch = A.batch_size();
    const auto Av = A.view();

    ref_eigs_sorted.assign(static_cast<size_t>(batch), std::vector<double>(static_cast<size_t>(n), std::numeric_limits<double>::quiet_NaN()));
    ref_ok.assign(static_cast<size_t>(batch), 0);

    for (int b = 0; b < batch; ++b) {
        std::vector<double> a_work(static_cast<size_t>(n * n));
        std::vector<double> w_work;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                a_work[static_cast<size_t>(i + j * n)] = static_cast<double>(Av(i, j, b));
            }
        }
        if (batchlas::verify::eigenvalues(n, a_work, w_work)) {
            std::sort(w_work.begin(), w_work.end());
            ref_eigs_sorted[static_cast<size_t>(b)] = std::move(w_work);
            ref_ok[static_cast<size_t>(b)] = 1;
        }
    }
}

template <Backend B, typename Real>
inline void record_eigensolver_metrics(miniacc::State& state,
                                       Queue& q,
                                       int neigs,
                                       bool compare_largest,
                                       const Matrix<Real, MatrixFormat::Dense>& A,
                                       const VectorView<Real>& evals,
                                       const Matrix<Real, MatrixFormat::Dense>& Z,
                                       const UnifiedVector<Real>& conds,
                                       const std::vector<std::vector<double>>& ref_eigs_sorted,
                                       const std::vector<char>& ref_ok) {
    const int n = A.rows();
    const int m = evals.size();
    const int batch = A.batch_size();
    (void)neigs;

    const auto residual_num = residual_residuals<B, Real>(q, A, Z, evals);
    q.wait();
    const auto ortho_num = orthogonality_residuals<B, Real>(q, Z);

    const double n_scale = static_cast<double>(n);

    for (int b = 0; b < batch; ++b) {
        const double cond = static_cast<double>(conds[static_cast<size_t>(b)]);
        const double log10_cond = std::log10(std::max(cond, 1e-300));
        const double r_den = n_scale;
        const double r_num = static_cast<double>(residual_num[static_cast<size_t>(b)]);
        const double o_num = static_cast<double>(ortho_num[static_cast<size_t>(b)]);

        double R = std::numeric_limits<double>::quiet_NaN();
        if (r_den > 0.0 && std::isfinite(r_num)) R = r_num / r_den;

        double O = std::numeric_limits<double>::quiet_NaN();
        if (n_scale > 0.0 && std::isfinite(o_num)) O = o_num / n_scale;

        std::vector<double> est_vals(static_cast<size_t>(m));
        for (int i = 0; i < m; ++i) est_vals[static_cast<size_t>(i)] = static_cast<double>(evals(i, b));

        double max_rel = std::numeric_limits<double>::quiet_NaN();
        if (ref_ok[static_cast<size_t>(b)]) {
            std::vector<double> ref_subset;
            const auto& ref_full = ref_eigs_sorted[static_cast<size_t>(b)];
            if (m == n) {
                ref_subset = ref_full;
            } else if (m > 0 && m <= n) {
                ref_subset.resize(static_cast<size_t>(m));
                if (compare_largest) {
                    for (int i = 0; i < m; ++i) ref_subset[static_cast<size_t>(i)] = ref_full[static_cast<size_t>(n - m + i)];
                } else {
                    for (int i = 0; i < m; ++i) ref_subset[static_cast<size_t>(i)] = ref_full[static_cast<size_t>(i)];
                }
            }
            max_rel = max_relative_eig_error(ref_subset, est_vals);
        }

        const bool ok = ref_ok[static_cast<size_t>(b)] && std::isfinite(R) && std::isfinite(O) && std::isfinite(max_rel);

        state.RecordSample(
            {
                {"log10_cond", log10_cond},
                {"res_num", r_num},
                {"res_denom", r_den},
                {"ortho_num", o_num},
                {"ortho_denom", n_scale},
                {"R", R},
                {"O", O},
                {"max_relerr", max_rel}
            },
            ok,
            ok ? "" : "non_finite_metric_or_reference_failed");
    }
}

template <typename R, typename T>
inline double gesvd_max_abs_singular_error(const R* ref_desc,
                                           const T* est_desc,
                                           int n) {
    double max_abs = 0.0;
    for (int i = 0; i < n; ++i) {
        const double ref = static_cast<double>(ref_desc[i]);
        const double est = static_cast<double>(est_desc[i]);
        const double abs_err = std::abs(est - ref);
        if (!std::isfinite(abs_err)) return std::numeric_limits<double>::quiet_NaN();
        max_abs = std::max(max_abs, abs_err);
    }
    return max_abs;
}

template <typename T>
struct gesvd_is_complex : std::false_type {};

template <typename T>
struct gesvd_is_complex<std::complex<T>> : std::true_type {};

template <typename T>
inline T gesvd_conj_value(const T& value) {
    if constexpr (gesvd_is_complex<T>::value) {
        return std::conj(value);
    } else {
        return value;
    }
}

template <typename T>
inline typename base_type<T>::type gesvd_abs_squared_value(const T& value) {
    using Real = typename base_type<T>::type;
    if constexpr (gesvd_is_complex<T>::value) {
        return static_cast<Real>(std::norm(value));
    } else {
        return value * value;
    }
}

template <typename Real>
inline double gesvd_sv_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return 5e-2;
    }
    return 1e-10;
}

template <typename Real>
inline double gesvd_ortho_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return 2e-1;
    }
    return 5e-8;
}

template <typename Real>
inline double gesvd_recon_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return 3e-1;
    }
    return 1e-8;
}

inline std::string gesvd_failure_reason(double u_ortho,
                                        double vh_ortho,
                                        double recon_rel,
                                        double sv_max_abs_err,
                                        double ortho_tol,
                                        double recon_tol,
                                        double sv_tol) {
    if (!std::isfinite(u_ortho) || !std::isfinite(vh_ortho) ||
        !std::isfinite(recon_rel) || !std::isfinite(sv_max_abs_err)) {
        return "non_finite_metric_or_reference_failed";
    }
    if (u_ortho > ortho_tol) return "u_ortho_exceeds_tol";
    if (vh_ortho > ortho_tol) return "vh_ortho_exceeds_tol";
    if (recon_rel > recon_tol) return "recon_rel_exceeds_tol";
    if (sv_max_abs_err > sv_tol) return "sv_max_abs_err_exceeds_tol";
    return {};
}

// Shared square-n gesvd accuracy run. buffer_size and solve forward their arguments to one
// driver (gesvd_cta, gesvd_blocked, ...); complex scalars get the trailing Uplo::Lower.
template <typename Scalar, Backend B, typename BufferSize, typename Solve>
void run_gesvd_acc(miniacc::State& state, const char* impl, BufferSize&& buffer_size, Solve&& solve) {
    using Real = typename base_type<Scalar>::type;
    const int n = std::max(2, state.arg_int(0));
    const int chunk_batch = chunk_batch_from_samples(state.samples());

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    state.SetTag("impl", impl);
    state.SetTag("backend", backend_name<B>());
    state.SetTag("dtype", dtype_name<Scalar>());
    state.SetTag("mode", gesvd_is_complex<Scalar>::value ? "hermitian" : "general");

    size_t produced = 0;
    while (produced < state.samples()) {
        const int cur_batch = static_cast<int>(std::min<size_t>(static_cast<size_t>(chunk_batch), state.samples() - produced));
        const unsigned seed = state.seed() + static_cast<unsigned>(produced);

        auto A = Matrix<Scalar>::Random(n, n, /*hermitian=*/gesvd_is_complex<Scalar>::value, cur_batch, seed);
        Matrix<Scalar> A_ref(n, n, cur_batch);
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*q, A_ref, A).wait();

        Matrix<Scalar> A_work(n, n, cur_batch);
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*q, A_work, A).wait();

        Matrix<Scalar> U(n, n, cur_batch);
        Matrix<Scalar> Vh(n, n, cur_batch);
        UnifiedVector<Real> s(static_cast<size_t>(n) * static_cast<size_t>(cur_batch));

        try {
            const size_t ws_bytes = [&]() {
                if constexpr (gesvd_is_complex<Scalar>::value) {
                    return buffer_size(*q, A_work, s.to_span(), U, Vh,
                                       SvdVectors::All, SvdVectors::All, Uplo::Lower);
                } else {
                    return buffer_size(*q, A_work, s.to_span(), U, Vh,
                                       SvdVectors::All, SvdVectors::All);
                }
            }();
            UnifiedVector<std::byte> ws(ws_bytes);
            if constexpr (gesvd_is_complex<Scalar>::value) {
                (void)solve(*q, A_work, s.to_span(), U, Vh,
                            SvdVectors::All, SvdVectors::All, Uplo::Lower, ws.to_span());
            } else {
                (void)solve(*q, A_work, s.to_span(), U, Vh,
                            SvdVectors::All, SvdVectors::All, ws.to_span());
            }
            q->wait();
        } catch (const std::exception& ex) {
            for (int b = 0; b < cur_batch; ++b) {
                state.RecordSample(
                    {
                        {"n", static_cast<double>(n)},
                        {"u_ortho", std::numeric_limits<double>::quiet_NaN()},
                        {"vh_ortho", std::numeric_limits<double>::quiet_NaN()},
                        {"recon_rel", std::numeric_limits<double>::quiet_NaN()},
                        {"sv_max_abs_err", std::numeric_limits<double>::quiet_NaN()}
                    },
                    false,
                    std::string("solver_exception:") + ex.what());
            }
            produced += static_cast<size_t>(cur_batch);
            continue;
        }

        for (int b = 0; b < cur_batch; ++b) {
            auto Ub = U.batch_item(b);
            auto Vhb = Vh.batch_item(b);
            auto Ab_ref = A_ref.batch_item(b);

            double u_ortho_num = 0.0;
            double vh_ortho_num = 0.0;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    Scalar dot_u = Scalar(0);
                    Scalar dot_vh = Scalar(0);
                    for (int k = 0; k < n; ++k) {
                        dot_u += gesvd_conj_value(Ub(k, i, 0)) * Ub(k, j, 0);
                        dot_vh += Vhb(i, k, 0) * gesvd_conj_value(Vhb(j, k, 0));
                    }
                    const Scalar target = (i == j) ? Scalar(1) : Scalar(0);
                    const Scalar du = dot_u - target;
                    const Scalar dv = dot_vh - target;
                    u_ortho_num += static_cast<double>(gesvd_abs_squared_value(du));
                    vh_ortho_num += static_cast<double>(gesvd_abs_squared_value(dv));
                }
            }

            double err2 = 0.0;
            double ref2 = 0.0;
            const Real* sb = s.data() + static_cast<size_t>(b) * static_cast<size_t>(n);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    Scalar recon = Scalar(0);
                    for (int k = 0; k < n; ++k) {
                        recon += Ub(i, k, 0) * static_cast<Scalar>(sb[k]) * Vhb(k, j, 0);
                    }
                    const Scalar ref = Ab_ref(i, j, 0);
                    const Scalar diff = recon - ref;
                    err2 += static_cast<double>(gesvd_abs_squared_value(diff));
                    ref2 += static_cast<double>(gesvd_abs_squared_value(ref));
                }
            }
            const double recon_rel = std::sqrt(err2 / std::max(ref2, 1e-30));

            // Reference: double ?gesdd since 2026-10-09 (was working-precision ?gesvd), so older CSVs'
            // sv_max_abs_err is not like for like. evidence: docs/perf/gesvd.md#gesvd-defect-c-an-accuracy-harness-that-could-not-see-relative-error
            double sv_max_abs_err = std::numeric_limits<double>::quiet_NaN();
            {
                std::vector<double> s_ref;
                if (batchlas::verify::singular_values(Ab_ref, 0, s_ref)) {
                    sv_max_abs_err = gesvd_max_abs_singular_error(s_ref.data(), sb, n);
                }
            }

            const double u_ortho = std::sqrt(u_ortho_num) / static_cast<double>(n);
            const double vh_ortho = std::sqrt(vh_ortho_num) / static_cast<double>(n);
            const std::string failure = gesvd_failure_reason(u_ortho,
                                                             vh_ortho,
                                                             recon_rel,
                                                             sv_max_abs_err,
                                                             gesvd_ortho_tol<Real>(),
                                                             gesvd_recon_tol<Real>(),
                                                             gesvd_sv_tol<Real>());
            const bool ok = failure.empty();
            state.RecordSample(
                {
                    {"n", static_cast<double>(n)},
                    {"u_ortho", u_ortho},
                    {"vh_ortho", vh_ortho},
                    {"recon_rel", recon_rel},
                    {"sv_max_abs_err", sv_max_abs_err}
                },
                ok,
                ok ? "" : failure);
        }

        produced += static_cast<size_t>(cur_batch);
    }
}

} // namespace miniacc_acc
