#pragma once

#include <blas/functions.hh>
#include <blas/extra.hh>
#include <blas/linalg.hh>
#include <util/miniacc.hh>
#include <batchlas/backend_config.h>
#include "../src/queue.hh"

#include <lapacke.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace miniacc_acc {

using namespace batchlas;

template <typename Real>
inline const char* dtype_name() {
    if constexpr (std::is_same_v<Real, float>) return "float";
    return "double";
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

template <typename Real>
inline void extract_tridiagonal(Queue& q,
                                const MatrixView<Real, MatrixFormat::Dense>& dense,
                                Vector<Real>& d,
                                Vector<Real>& e) {
    const int n = dense.rows();
    const int batch = dense.batch_size();
    auto a_view = dense.kernel_view();
    auto d_ptr = d.data_ptr();
    auto e_ptr = e.data_ptr();
    const int d_inc = d.inc();
    const int e_inc = e.inc();
    const int d_stride = d.stride();
    const int e_stride = e.stride();

    q->parallel_for(sycl::range<1>(static_cast<size_t>(batch * n)), [=](sycl::id<1> idx) {
        const int linear = static_cast<int>(idx[0]);
        const int b = linear / n;
        const int i = linear - b * n;
        d_ptr[b * d_stride + i * d_inc] = a_view(i, i, b);
        if (i < n - 1) {
            e_ptr[b * e_stride + i * e_inc] = a_view(i + 1, i, b);
        }
    });
    q.wait();
}

template <Backend B, typename Real>
inline UnifiedVector<typename base_type<Real>::type> orthogonality_residuals(
    Queue& q,
    const Matrix<Real, MatrixFormat::Dense>& Z) {
    const int m = Z.cols();
    const int batch = Z.batch_size();
    auto ztz_minus_i = Matrix<Real>::Identity(m, batch);
    gemm<B, Real>(q,
                  Z.view(),
                  Z.view(),
                  ztz_minus_i.view(),
                  Real(1),
                  Real(-1),
                  Transpose::Trans,
                  Transpose::NoTrans);
    q.wait();
    return norm(q, ztz_minus_i.view(), NormType::Frobenius);
}

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

    gemm<B, Real>(q,
                  A.view(),
                  Z.view(),
                  R.view(),
                  Real(1),
                  Real(0),
                  Transpose::NoTrans,
                  Transpose::NoTrans);

    auto r_view = R.kernel_view();
    auto z_view = Z.view().kernel_view();
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

    return norm(q, R.view(), NormType::Frobenius);
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
        const int info = LAPACKE_dsterf(static_cast<lapack_int>(n), d_work.data(), e_work.data());
        if (info == 0) {
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
        std::vector<double> w_work(static_cast<size_t>(n));
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                a_work[static_cast<size_t>(i + j * n)] = static_cast<double>(Av(i, j, b));
            }
        }
        const int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'L', static_cast<lapack_int>(n), a_work.data(), static_cast<lapack_int>(n), w_work.data());
        if (info == 0) {
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

} // namespace miniacc_acc
