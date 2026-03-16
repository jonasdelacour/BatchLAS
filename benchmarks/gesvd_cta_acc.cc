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
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

template <typename Benchmark>
void GesvdCtaAccSizes(Benchmark* b) {
    for (double n : {8.0, 16.0, 32.0}) b->Args({n});
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

template <typename Real>
inline int lapacke_gesvd_values_only(int n,
                                     Real* a_col_major,
                                     Real* s_out,
                                     Real* superb) {
    if constexpr (std::is_same_v<Real, float>) {
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
    } else {
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
    }
}

template <typename Real, Backend B>
void run_gesvd_cta_acc(miniacc::State& state) {
    const int n = std::max(2, state.arg_int(0));
    const int chunk_batch = miniacc_acc::chunk_batch_from_samples(state.samples());

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    state.SetTag("impl", "gesvd_cta");
    state.SetTag("backend", miniacc_acc::backend_name<B>());
    state.SetTag("dtype", miniacc_acc::dtype_name<Real>());

    size_t produced = 0;
    while (produced < state.samples()) {
        const int cur_batch = static_cast<int>(std::min<size_t>(static_cast<size_t>(chunk_batch), state.samples() - produced));
        const unsigned seed = state.seed() + static_cast<unsigned>(produced);

        auto A = Matrix<Real>::Random(n, n, /*hermitian=*/false, cur_batch, seed);
        Matrix<Real> A_ref(n, n, cur_batch);
        MatrixView<Real, MatrixFormat::Dense>::copy(*q, A_ref.view(), A.view()).wait();

        Matrix<Real> A_work(n, n, cur_batch);
        MatrixView<Real, MatrixFormat::Dense>::copy(*q, A_work.view(), A.view()).wait();

        Matrix<Real> U(n, n, cur_batch);
        Matrix<Real> Vh(n, n, cur_batch);
        UnifiedVector<Real> s(static_cast<size_t>(n) * static_cast<size_t>(cur_batch));

        try {
            UnifiedVector<std::byte> ws(
                gesvd_cta_buffer_size<B, Real>(*q,
                                               A_work.view(),
                                               s.to_span(),
                                               U.view(),
                                               Vh.view(),
                                               SvdVectors::All,
                                               SvdVectors::All));
            gesvd_cta<B, Real>(*q,
                               A_work.view(),
                               s.to_span(),
                               U.view(),
                               Vh.view(),
                               SvdVectors::All,
                               SvdVectors::All,
                               ws.to_span());
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
                    double dot_u = 0.0;
                    double dot_vh = 0.0;
                    for (int k = 0; k < n; ++k) {
                        dot_u += static_cast<double>(Ub(k, i, 0)) * static_cast<double>(Ub(k, j, 0));
                        dot_vh += static_cast<double>(Vhb(i, k, 0)) * static_cast<double>(Vhb(j, k, 0));
                    }
                    const double target = (i == j) ? 1.0 : 0.0;
                    const double du = dot_u - target;
                    const double dv = dot_vh - target;
                    u_ortho_num += du * du;
                    vh_ortho_num += dv * dv;
                }
            }

            double err2 = 0.0;
            double ref2 = 0.0;
            const Real* sb = s.data() + static_cast<size_t>(b) * static_cast<size_t>(n);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    double recon = 0.0;
                    for (int k = 0; k < n; ++k) {
                        recon += static_cast<double>(Ub(i, k, 0)) *
                                 static_cast<double>(sb[k]) *
                                 static_cast<double>(Vhb(k, j, 0));
                    }
                    const double ref = static_cast<double>(Ab_ref(i, j, 0));
                    const double diff = recon - ref;
                    err2 += diff * diff;
                    ref2 += ref * ref;
                }
            }
            const double recon_rel = std::sqrt(err2 / std::max(ref2, 1e-30));

            double sv_max_abs_err = std::numeric_limits<double>::quiet_NaN();
#if BATCHLAS_HAS_HOST_BACKEND
            {
                std::vector<Real> a_host(static_cast<size_t>(n) * static_cast<size_t>(n));
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < n; ++i) {
                        a_host[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)] = Ab_ref(i, j, 0);
                    }
                }
                std::vector<Real> s_ref(static_cast<size_t>(n));
                std::vector<Real> superb(static_cast<size_t>(std::max(0, n - 1)));
                const int info = lapacke_gesvd_values_only<Real>(n, a_host.data(), s_ref.data(), superb.data());
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
static void ACC_GESVD_CTA(miniacc::State& state) {
    run_gesvd_cta_acc<Real, B>(state);
}

BATCHLAS_ACC_CUDA(ACC_GESVD_CTA, GesvdCtaAccSizes)

MINI_ACC_MAIN()
