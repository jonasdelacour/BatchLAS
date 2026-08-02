// 6. Symmetric and Hermitian eigensolvers
//
// The whole syev family: the dispatching driver and each variant it can pick,
// the parameter structs that tune them, complex Hermitian input, and what
// `uplo` actually promises.
//
// Every solver here has the same contract: A is overwritten with the
// eigenvectors (when asked for), and the eigenvalues come back ascending in a
// span of REAL scalars — `float_t<T>`, not T — of length n per batch item.

#include <algorithm>
#include <complex>
#include <vector>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_common.hh"
#include "example_linalg.hh"
#include "example_runner.hh"

using namespace batchlas;
using namespace examples;

namespace {

constexpr int kBatch = 3;

// The spectrum every solver in this example is asked to find.
std::vector<double> target_spectrum(int n) {
    std::vector<double> w(n);
    for (int i = 0; i < n; ++i) w[i] = -2.0 + 0.75 * i;
    return w;
}

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        support_section(ctx);
        syev_section(ctx);
        values_only_section(ctx);
        small_variants_section(ctx);
        large_variants_section(ctx);
        tuning_section(ctx);
        hermitian_section(ctx);
        uplo_section(ctx);
    }

    // Solve with a callable, then check eigenvalues and (optionally) the
    // residual |A V - V diag(w)| against the matrix we started from.
    template <typename Solve>
    static void check_solver(const char* name, Queue& ctx, const HostMatrix<double>& original, int n, bool vectors,
                             double tol, Solve&& solve) {
        auto A = broadcast(original, kBatch);
        UnifiedVector<double> w(static_cast<size_t>(n) * kBatch);
        solve(A, w);
        ctx.wait();

        const auto want = target_spectrum(n);
        double val_err = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            std::vector<double> got(w.begin() + b * n, w.begin() + (b + 1) * n);
            val_err = std::max(val_err, max_abs_diff(got, want));
        }
        report_error(std::string(name) + ": eigenvalues", val_err, tol);

        if (vectors) {
            double res = 0.0;
            for (int b = 0; b < kBatch; ++b) {
                std::vector<double> got(w.begin() + b * n, w.begin() + (b + 1) * n);
                res = std::max(res, eigen_residual(original, to_host(A, b), got));
            }
            report_error(std::string(name) + ": |A V - V diag(w)|", res, tol);
        }
    }

    // -----------------------------------------------------------------------
    // Which variants does this device support?
    //
    // There is no runtime query in the C++ API (the Python facade's
    // `syev_variant_support` has no direct counterpart). Ask the device
    // instead: the CTA variants map one work-group onto one matrix and need a
    // sub-group width of 32, so they are GPU-only and limited to n <= 32.
    // -----------------------------------------------------------------------
    static void support_section(Queue& ctx) {
        section("Which variants does this device support?");

        report("device type", on_gpu(ctx) ? "gpu" : "cpu");
        report("max sub-group size", ctx.device().get_property(DeviceProperty::MAX_SUB_GROUP_SIZE));
        report("CTA variants usable here", supports_cta(ctx));
        report("blocked / two-stage", std::string(on_gpu(ctx) ? "usable (GPU, Uplo::Lower only)" : "GPU only"));
    }

    // -----------------------------------------------------------------------
    // syev — the dispatching driver.
    //
    // Picks a provider for you based on the device and the problem: a CTA
    // variant for tiny matrices on a GPU, the blocked or two-stage path for
    // larger ones, the vendor library otherwise. The BATCHLAS_SYEV_*
    // environment variables override the choice.
    // -----------------------------------------------------------------------
    static void syev_section(Queue& ctx) {
        section("syev - eigenvalues and eigenvectors");

        const int n = 24;
        const auto original = symmetric_with_eigenvalues<double>(target_spectrum(n), 11);

        check_solver("syev", ctx, original, n, /*vectors=*/true, 1e-9, [&](Matrix<double>& A, UnifiedVector<double>& w) {
            UnifiedVector<std::byte> ws(
                syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower));
            syev<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
        });
    }

    // -----------------------------------------------------------------------
    // Eigenvalues only
    //
    // JobType::NoEigenVectors skips the back-transform, which is most of the
    // work for a large matrix. Whether A survives depends on the variant —
    // syev_cta_fused and syev_jacobi_cta leave it alone, the others do not —
    // so do not rely on A afterwards.
    // -----------------------------------------------------------------------
    static void values_only_section(Queue& ctx) {
        section("Eigenvalues only");

        const int n = 24;
        const auto original = symmetric_with_eigenvalues<double>(target_spectrum(n), 11);

        check_solver("syev(NoEigenVectors)", ctx, original, n, /*vectors=*/false, 1e-9,
                     [&](Matrix<double>& A, UnifiedVector<double>& w) {
                         UnifiedVector<std::byte> ws(syev_buffer_size<B>(ctx, A.view(), w.to_span(),
                                                                          JobType::NoEigenVectors, Uplo::Lower));
                         syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws.to_span());
                     });
    }

    // -----------------------------------------------------------------------
    // The small-matrix variants
    //
    // Three ways to solve an n <= 32 problem with one work-group per matrix:
    //
    //   syev_cta        tridiagonalise, QL/QR on the tridiagonal, back-transform
    //   syev_cta_fused  the same three stages fused into one kernel launch
    //   syev_jacobi_cta two-sided Jacobi; slower, but see example 10
    //
    // None of them needs a global workspace worth speaking of, and the fused
    // and Jacobi ones ignore the workspace argument entirely.
    // -----------------------------------------------------------------------
    static void small_variants_section(Queue& ctx) {
        section("The small-matrix variants");

        if constexpr (!has_cta_variants<B>) {
            report_skip("CTA variants", "not instantiated for this backend");
            return;
        } else {
            if (!supports_cta(ctx)) {
                report_skip("CTA variants", "needs a GPU with sub-group width 32");
                return;
            }

            const int n = 16;
            const auto original = symmetric_with_eigenvalues<double>(target_spectrum(n), 21);

            check_solver("syev_cta", ctx, original, n, true, 1e-9, [&](Matrix<double>& A, UnifiedVector<double>& w) {
                UnifiedVector<std::byte> ws(syev_cta_buffer_size<B>(ctx, A.view(), JobType::EigenVectors));
                syev_cta<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
            });

            check_solver("syev_cta_fused", ctx, original, n, true, 1e-9,
                         [&](Matrix<double>& A, UnifiedVector<double>& w) {
                             syev_cta_fused<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower);
                         });

            check_solver("syev_jacobi_cta", ctx, original, n, true, 1e-9,
                         [&](Matrix<double>& A, UnifiedVector<double>& w) {
                             syev_jacobi_cta<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower);
                         });
        }
    }

    // -----------------------------------------------------------------------
    // The medium and large variants
    //
    //   syev_blocked    sytrd_blocked -> stedc -> ormqr_blocked
    //   syev_two_stage  dense -> band -> tridiagonal -> stedc
    //
    // Both are GPU-only and both accept Uplo::Lower only. Example 07 takes the
    // reduction stages apart; example 12 measures which one to reach for.
    // -----------------------------------------------------------------------
    static void large_variants_section(Queue& ctx) {
        section("The medium and large variants");

        if (!on_gpu(ctx)) {
            report_skip("syev_blocked / syev_two_stage", "GPU only");
            return;
        }

        const int n = 48;
        const auto original = symmetric_with_eigenvalues<double>(target_spectrum(n), 31);

        check_solver("syev_blocked", ctx, original, n, true, 1e-8, [&](Matrix<double>& A, UnifiedVector<double>& w) {
            UnifiedVector<std::byte> ws(
                syev_blocked_buffer_size<B>(ctx, A.view(), JobType::EigenVectors, Uplo::Lower));
            syev_blocked<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
        });

        check_solver("syev_two_stage", ctx, original, n, true, 1e-8, [&](Matrix<double>& A, UnifiedVector<double>& w) {
            UnifiedVector<std::byte> ws(
                syev_two_stage_buffer_size<B>(ctx, A.view(), JobType::EigenVectors, Uplo::Lower));
            syev_two_stage<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
        });

        // Known defect: with JobType::NoEigenVectors, syev_two_stage returns
        // wrong eigenvalues for n >= 32 — silently, no error. Measured here so
        // it stays visible; see the known issues in the README. Ask for
        // vectors and discard them, as elsewhere in this library.
        {
            auto A = broadcast(original, 1);
            UnifiedVector<double> w(n);
            UnifiedVector<std::byte> ws(
                syev_two_stage_buffer_size<B>(ctx, A.view(), JobType::NoEigenVectors, Uplo::Lower));
            syev_two_stage<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws.to_span());
            ctx.wait();
            std::vector<double> got(w.begin(), w.begin() + n);
            report_magnitude("syev_two_stage(NoEigenVectors) error at n=" + std::to_string(n) + " (known bad)",
                             max_abs_diff(got, target_spectrum(n)));
        }
    }

    // -----------------------------------------------------------------------
    // Tuning a variant through its parameter struct
    //
    // Where the Python facade takes dataclasses, C++ takes plain structs with
    // defaulted members: SteqrParams for the QR iteration, StedcParams for
    // divide and conquer, JacobiParams for the Jacobi sweep. Pass one by value
    // and it applies to that call only. Note that the *buffer size* can depend
    // on the parameters, so pass the same struct to both calls.
    // -----------------------------------------------------------------------
    static void tuning_section(Queue& ctx) {
        section("Tuning a variant through its parameter struct");

        const int n = 16;
        const auto original = symmetric_with_eigenvalues<double>(target_spectrum(n), 41);

        // Descending order instead of the default ascending.
        if constexpr (has_cta_variants<B>) {
            if (supports_cta(ctx)) {
                SteqrParams<double> params;
                params.sort_order = SortOrder::Descending;
                params.max_sweeps = 100;

                auto A = broadcast(original, kBatch);
                UnifiedVector<double> w(static_cast<size_t>(n) * kBatch);
                UnifiedVector<std::byte> ws(
                    syev_cta_buffer_size<B>(ctx, A.view(), JobType::EigenVectors, params));
                syev_cta<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span(), params);
                ctx.wait();

                auto want = target_spectrum(n);
                std::sort(want.begin(), want.end(), std::greater<double>());
                std::vector<double> got(w.begin(), w.begin() + n);
                report_error("SteqrParams::sort_order = Descending", max_abs_diff(got, want), 1e-9);

                // JacobiParams tunes the sweep instead.
                JacobiParams<double> jp;
                jp.sort_order = SortOrder::Descending;
                jp.max_sweeps = 40;
                auto A2 = broadcast(original, kBatch);
                UnifiedVector<double> w2(static_cast<size_t>(n) * kBatch);
                syev_jacobi_cta<B>(ctx, A2.view(), w2.to_span(), JobType::EigenVectors, Uplo::Lower,
                                   Span<std::byte>(), jp);
                ctx.wait();
                std::vector<double> got2(w2.begin(), w2.begin() + n);
                report_error("JacobiParams::sort_order = Descending", max_abs_diff(got2, want), 1e-9);
            } else {
                report_skip("SteqrParams / JacobiParams", "needs a GPU with sub-group width 32");
            }
        } else {
            report_skip("SteqrParams / JacobiParams", "CTA variants not instantiated for this backend");
        }

        // StedcParams tunes the divide-and-conquer tridiagonal solve that
        // syev_blocked runs underneath.
        if (on_gpu(ctx)) {
            StedcParams<double> sp;
            sp.recursion_threshold = 8;  // switch to the leaf solver sooner

            const int n2 = 32;
            const auto orig2 = symmetric_with_eigenvalues<double>(target_spectrum(n2), 51);
            auto A = broadcast(orig2, kBatch);
            UnifiedVector<double> w(static_cast<size_t>(n2) * kBatch);
            UnifiedVector<std::byte> ws(
                syev_blocked_buffer_size<B>(ctx, A.view(), JobType::EigenVectors, Uplo::Lower, sp));
            syev_blocked<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span(), sp);
            ctx.wait();

            std::vector<double> got(w.begin(), w.begin() + n2);
            report_error("StedcParams::recursion_threshold = 8", max_abs_diff(got, target_spectrum(n2)), 1e-8);
        } else {
            report_skip("StedcParams", "syev_blocked is GPU only");
        }
    }

    // -----------------------------------------------------------------------
    // Hermitian (complex) input
    //
    // Same calls, T = std::complex<double>. The eigenvalues stay real, so the
    // span is `double` while the matrix is complex — that asymmetry is what
    // `Span<typename base_type<T>::type>` in the signatures means.
    // -----------------------------------------------------------------------
    static void hermitian_section(Queue& ctx) {
        section("Hermitian (complex) input");

        using C = std::complex<double>;
        const int n = 16;
        const auto want = target_spectrum(n);
        const auto original = symmetric_with_eigenvalues<C>(want, 61);

        auto A = broadcast(original, kBatch);
        UnifiedVector<double> w(static_cast<size_t>(n) * kBatch);  // real, even though A is complex
        UnifiedVector<std::byte> ws(
            syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower));
        syev<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
        ctx.wait();

        double val_err = 0.0, res = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            std::vector<double> got(w.begin() + b * n, w.begin() + (b + 1) * n);
            val_err = std::max(val_err, max_abs_diff(got, want));
            res = std::max(res, eigen_residual(original, to_host(A, b), got));
        }
        report_error("Hermitian eigenvalues", val_err, 1e-9);
        report_error("Hermitian |A V - V diag(w)|", res, 1e-9);
    }

    // -----------------------------------------------------------------------
    // uplo — which triangle holds your data
    //
    // `uplo` says which triangle the solver may read; the other one is ignored
    // and may hold anything. Passing a full symmetric matrix is therefore
    // always safe, and is what to do if you are unsure.
    //
    // Filling only one triangle is where backends differ: on CUDA, Uplo::Upper
    // with only the upper triangle populated gives the wrong answer, while
    // Uplo::Lower with only the lower triangle is fine.
    // -----------------------------------------------------------------------
    static void uplo_section(Queue& ctx) {
        section("uplo - which triangle holds your data");

        const int n = 16;
        const auto want = target_spectrum(n);
        const auto full = symmetric_with_eigenvalues<double>(want, 71);

        auto solve = [&](const HostMatrix<double>& input, Uplo uplo) {
            auto A = broadcast(input, 1);
            UnifiedVector<double> w(n);
            UnifiedVector<std::byte> ws(
                syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, uplo));
            syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, uplo, ws.to_span());
            ctx.wait();
            return std::vector<double>(w.begin(), w.begin() + n);
        };

        report_error("full matrix, Uplo::Lower", max_abs_diff(solve(full, Uplo::Lower), want), 1e-9);
        report_error("full matrix, Uplo::Upper", max_abs_diff(solve(full, Uplo::Upper), want), 1e-9);
        report_error("lower triangle only, Uplo::Lower",
                     max_abs_diff(solve(keep_triangle(full, Uplo::Lower), Uplo::Lower), want), 1e-9);

        const double upper_only = max_abs_diff(solve(keep_triangle(full, Uplo::Upper), Uplo::Upper), want);
        if (on_gpu(ctx)) {
            // Known defect; see the README. Reported, not checked.
            report_magnitude("upper triangle only, Uplo::Upper (known bad on CUDA)", upper_only);
        } else {
            report_error("upper triangle only, Uplo::Upper", upper_only, 1e-9);
        }
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("6. Symmetric and Hermitian eigensolvers")
