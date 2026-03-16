#include <blas/extensions.hh>
#include <util/miniacc.hh>

#include "acc_utils.hh"
#include "miniacc_accuracy_common.hh"

#if BATCHLAS_HAS_HOST_BACKEND
#include <lapacke.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <complex>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

using namespace batchlas;

namespace {

template <typename Benchmark>
void GesvdAccSizes(Benchmark* b) {
    for (double n : {16.0, 32.0, 64.0, 128.0}) b->Args({n});
}

template <typename T>
inline double max_abs_singular_error(const T* ref_desc,
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
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
inline T conj_value(const T& value) {
    if constexpr (is_complex<T>::value) {
        return std::conj(value);
    } else {
        return value;
    }
}

template <typename T>
inline typename base_type<T>::type abs_squared_value(const T& value) {
    using Real = typename base_type<T>::type;
    if constexpr (is_complex<T>::value) {
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

template <typename Scalar>
inline int lapacke_gesvd_values_only(int n,
                                     Scalar* a_col_major,
                                     typename base_type<Scalar>::type* s_out,
                                     typename base_type<Scalar>::type* superb) {
    if constexpr (std::is_same_v<Scalar, float>) {
        return LAPACKE_sgesvd(LAPACK_COL_MAJOR,
                              'N',
                              'N',
                              n,
                              n,
                              a_col_major,
                              n,
                              s_out,
                              nullptr,
                              1,
                              nullptr,
                              1,
                              superb);
    } else if constexpr (std::is_same_v<Scalar, double>) {
        return LAPACKE_dgesvd(LAPACK_COL_MAJOR,
                              'N',
                              'N',
                              n,
                              n,
                              a_col_major,
                              n,
                              s_out,
                              nullptr,
                              1,
                              nullptr,
                              1,
                              superb);
    } else if constexpr (std::is_same_v<Scalar, std::complex<float>>) {
        return LAPACKE_cgesvd(LAPACK_COL_MAJOR,
                              'N',
                              'N',
                              n,
                              n,
                              reinterpret_cast<lapack_complex_float*>(a_col_major),
                              n,
                              s_out,
                              nullptr,
                              1,
                              nullptr,
                              1,
                              superb);
    } else {
        static_assert(std::is_same_v<Scalar, std::complex<double>>);
        return LAPACKE_zgesvd(LAPACK_COL_MAJOR,
                              'N',
                              'N',
                              n,
                              n,
                              reinterpret_cast<lapack_complex_double*>(a_col_major),
                              n,
                              s_out,
                              nullptr,
                              1,
                              nullptr,
                              1,
                              superb);
    }
}

template <typename Scalar, Backend B>
void run_gesvd_blocked_acc(miniacc::State& state) {
    using Real = typename base_type<Scalar>::type;
    const int n = std::max(2, state.arg_int(0));
    const int chunk_batch = miniacc_acc::chunk_batch_from_samples(state.samples());

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    state.SetTag("impl", "gesvd_blocked");
    state.SetTag("backend", miniacc_acc::backend_name<B>());
    state.SetTag("dtype", miniacc_acc::dtype_name<Scalar>());
    state.SetTag("mode", is_complex<Scalar>::value ? "hermitian" : "general");

    size_t produced = 0;
    while (produced < state.samples()) {
        const int cur_batch = static_cast<int>(std::min<size_t>(static_cast<size_t>(chunk_batch), state.samples() - produced));
        const unsigned seed = state.seed() + static_cast<unsigned>(produced);

        auto A = Matrix<Scalar>::Random(n, n, /*hermitian=*/is_complex<Scalar>::value, cur_batch, seed);
        Matrix<Scalar> A_ref(n, n, cur_batch);
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*q, A_ref.view(), A.view()).wait();

        Matrix<Scalar> A_work(n, n, cur_batch);
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*q, A_work.view(), A.view()).wait();

        Matrix<Scalar> U(n, n, cur_batch);
        Matrix<Scalar> Vh(n, n, cur_batch);
        UnifiedVector<Real> s(static_cast<size_t>(n) * static_cast<size_t>(cur_batch));

        try {
            const size_t ws_bytes = [&]() {
                if constexpr (is_complex<Scalar>::value) {
                    return gesvd_blocked_buffer_size<B, Scalar>(*q,
                                                                A_work.view(),
                                                                s.to_span(),
                                                                U.view(),
                                                                Vh.view(),
                                                                SvdVectors::All,
                                                                SvdVectors::All,
                                                                Uplo::Lower);
                } else {
                    return gesvd_blocked_buffer_size<B, Scalar>(*q,
                                                                A_work.view(),
                                                                s.to_span(),
                                                                U.view(),
                                                                Vh.view(),
                                                                SvdVectors::All,
                                                                SvdVectors::All);
                }
            }();
            UnifiedVector<std::byte> ws(ws_bytes);
            if constexpr (is_complex<Scalar>::value) {
                gesvd_blocked<B, Scalar>(*q,
                                         A_work.view(),
                                         s.to_span(),
                                         U.view(),
                                         Vh.view(),
                                         SvdVectors::All,
                                         SvdVectors::All,
                                         Uplo::Lower,
                                         ws.to_span());
            } else {
                gesvd_blocked<B, Scalar>(*q,
                                         A_work.view(),
                                         s.to_span(),
                                         U.view(),
                                         Vh.view(),
                                         SvdVectors::All,
                                         SvdVectors::All,
                                         ws.to_span());
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
            auto Ub = U.view().batch_item(b);
            auto Vhb = Vh.view().batch_item(b);
            auto Ab_ref = A_ref.view().batch_item(b);

            double u_ortho_num = 0.0;
            double vh_ortho_num = 0.0;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    Scalar dot_u = Scalar(0);
                    Scalar dot_vh = Scalar(0);
                    for (int k = 0; k < n; ++k) {
                        dot_u += conj_value(Ub(k, i, 0)) * Ub(k, j, 0);
                        dot_vh += Vhb(i, k, 0) * conj_value(Vhb(j, k, 0));
                    }
                    const Scalar target = (i == j) ? Scalar(1) : Scalar(0);
                    const Scalar du = dot_u - target;
                    const Scalar dv = dot_vh - target;
                    u_ortho_num += static_cast<double>(abs_squared_value(du));
                    vh_ortho_num += static_cast<double>(abs_squared_value(dv));
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
                    err2 += static_cast<double>(abs_squared_value(diff));
                    ref2 += static_cast<double>(abs_squared_value(ref));
                }
            }
            const double recon_rel = std::sqrt(err2 / std::max(ref2, 1e-30));

            double sv_max_abs_err = std::numeric_limits<double>::quiet_NaN();
#if BATCHLAS_HAS_HOST_BACKEND
            {
                std::vector<Scalar> a_host(static_cast<size_t>(n) * static_cast<size_t>(n));
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < n; ++i) {
                        a_host[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)] = Ab_ref(i, j, 0);
                    }
                }
                std::vector<Real> s_ref(static_cast<size_t>(n));
                std::vector<Real> superb(static_cast<size_t>(std::max(0, n - 1)));
                const int info = lapacke_gesvd_values_only<Scalar>(n, a_host.data(), s_ref.data(), superb.data());
                if (info == 0) {
                    sv_max_abs_err = max_abs_singular_error<Real>(s_ref.data(), sb, n);
                }
            }
#endif

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

} // namespace

template <typename Real, Backend B>
static void ACC_GESVD_BLOCKED(miniacc::State& state) {
    run_gesvd_blocked_acc<Real, B>(state);
}

template <typename Benchmark>
void GesvdAccSizesNetlib(Benchmark* b) {
    GesvdAccSizes(b);
}

BATCHLAS_REGISTER_ACCURACY_ALL_TYPES(ACC_GESVD_BLOCKED, GesvdAccSizes)

MINI_ACC_MAIN()
