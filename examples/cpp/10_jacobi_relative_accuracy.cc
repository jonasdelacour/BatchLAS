// 10. Relative accuracy: why syev_jacobi_cta exists
//
// syev_cta and syev_jacobi_cta solve the same problem and agree to machine
// precision on a well-scaled matrix. On a *graded* one they do not — and
// absolute error hides the difference completely.
//
// The matrix class that matters is A = D S D, with D a diagonal scaling whose
// entries span many orders of magnitude and S well conditioned. Every entry of
// such a matrix still carries full relative precision, so its tiny eigenvalues
// are well determined by the data — but only an algorithm that never forms
// differences of wildly different magnitudes can recover them.
// Tridiagonalization does form them; two-sided Jacobi with a relative stopping
// criterion does not. (Demmel & Veselic, SIMAX 13(4), 1992.)
//
// Note the qualifier: a matrix with a wide spectrum is NOT automatically in
// this class. Take a random orthogonal Q and form Q diag(w) Q^T, and the small
// eigenvalues are genuinely lost in the rounding of the entries themselves —
// no algorithm recovers them, and Jacobi has no advantage. The last section
// shows exactly that.
//
// This example is GPU-only: both solvers are CTA variants.

#include <algorithm>
#include <cmath>
#include <vector>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_common.hh"
#include "example_linalg.hh"
#include "example_runner.hh"

using namespace batchlas;
using namespace examples;

namespace {

constexpr int kN = 16;

// A = D S D: S is symmetric positive definite with a unit diagonal and modest
// off-diagonal entries; D grades the rows and columns from 1 down to
// 10^-grading. The result is positive definite with a condition number around
// 10^(2*grading), and every entry is exact to full relative precision.
HostMatrix<double> graded_matrix(int n, double grading, unsigned seed = 42) {
    auto R = random_host<double>(n, n, seed);
    HostMatrix<double> S(n, n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) S(i, j) = 0.25 * (R(i, j) + R(j, i)) * 0.5;
    }
    for (int i = 0; i < n; ++i) S(i, i) = 1.0;

    std::vector<double> d(n);
    for (int i = 0; i < n; ++i) d[i] = std::pow(10.0, -grading * i / (n - 1));

    HostMatrix<double> A(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) A(i, j) = d[i] * S(i, j) * d[j];
    return A;
}

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        if constexpr (!has_cta_variants<B>) {
            report_skip("the whole example", "the CTA variants are not instantiated for this backend");
            return;
        } else {
            if (!supports_cta(ctx)) {
                report_skip("the whole example", "needs a GPU with sub-group width 32");
                return;
            }
            well_scaled_section(ctx);
            graded_section(ctx);
            absolute_section(ctx);
            relative_section(ctx);
            sweep_section(ctx);
            tuning_section(ctx);
            vectors_section(ctx);
            not_automatic_section(ctx);
        }
    }

    static std::vector<double> solve_cta(Queue& ctx, const HostMatrix<double>& A) {
        auto M = broadcast(A, 1);
        UnifiedVector<double> w(kN);
        UnifiedVector<std::byte> ws(syev_cta_buffer_size<B>(ctx, M.view(), JobType::NoEigenVectors));
        syev_cta<B>(ctx, M.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws.to_span());
        ctx.wait();
        return sorted(std::vector<double>(w.begin(), w.begin() + kN));
    }

    static std::vector<double> solve_jacobi(Queue& ctx, const HostMatrix<double>& A,
                                            JacobiParams<double> params = JacobiParams<double>()) {
        auto M = broadcast(A, 1);
        UnifiedVector<double> w(kN);
        syev_jacobi_cta<B>(ctx, M.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, Span<std::byte>(),
                           params);
        ctx.wait();
        return sorted(std::vector<double>(w.begin(), w.begin() + kN));
    }

    // The reference. examples::jacobi_eigenvalues is itself a two-sided Jacobi
    // sweep, run on the host in double — which is precisely the algorithm that
    // is relatively accurate on this matrix class, so it is a fair yardstick
    // for the small eigenvalues.
    static std::vector<double> reference(const HostMatrix<double>& A) { return jacobi_eigenvalues(A); }

    // -----------------------------------------------------------------------
    // On a well-scaled matrix, both are fine
    // -----------------------------------------------------------------------
    static void well_scaled_section(Queue& ctx) {
        section("On a well-scaled matrix, both are fine");

        const auto A = graded_matrix(kN, 0.0);  // D = I
        const auto want = reference(A);

        report_error("syev_cta relative error", max_rel_diff(solve_cta(ctx, A), want), 1e-12);
        report_error("syev_jacobi_cta relative error", max_rel_diff(solve_jacobi(ctx, A), want), 1e-12);
    }

    // -----------------------------------------------------------------------
    // A graded matrix
    // -----------------------------------------------------------------------
    static void graded_section(Queue& ctx) {
        section("A graded matrix");

        const auto A = graded_matrix(kN, 7.0);
        const auto want = reference(A);
        report_magnitude("largest eigenvalue", want.back());
        report_magnitude("smallest eigenvalue", want.front());
        report_magnitude("condition number", want.back() / want.front());
    }

    // -----------------------------------------------------------------------
    // Absolute error hides the problem entirely
    //
    // Both solvers are accurate to ~eps times the norm of A. Since the norm is
    // set by the largest eigenvalue, that number says nothing at all about the
    // small ones.
    // -----------------------------------------------------------------------
    static void absolute_section(Queue& ctx) {
        section("Absolute error hides the problem entirely");

        const auto A = graded_matrix(kN, 7.0);
        const auto want = reference(A);

        report_error("syev_cta absolute error", max_abs_diff(solve_cta(ctx, A), want), 1e-13);
        report_error("syev_jacobi_cta absolute error", max_abs_diff(solve_jacobi(ctx, A), want), 1e-13);
        report_skip("conclusion from absolute error", "both look perfect - which is the trap");
    }

    // -----------------------------------------------------------------------
    // Relative error tells the real story
    // -----------------------------------------------------------------------
    static void relative_section(Queue& ctx) {
        section("Relative error tells the real story");

        const auto A = graded_matrix(kN, 7.0);
        const auto want = reference(A);

        const double cta_rel = max_rel_diff(solve_cta(ctx, A), want);
        const double jac_rel = max_rel_diff(solve_jacobi(ctx, A), want);

        report_magnitude("syev_cta relative error", cta_rel);
        report_magnitude("syev_jacobi_cta relative error", jac_rel);
        report_check("Jacobi is more accurate in the relative sense", jac_rel < cta_rel);
    }

    // -----------------------------------------------------------------------
    // Sweeping the grading
    //
    // The gap opens as the scaling widens: identical at grading 0, orders of
    // magnitude apart once the diagonal spans 1e-6 or more.
    // -----------------------------------------------------------------------
    static void sweep_section(Queue& ctx) {
        section("Sweeping the grading");

        double best_ratio = 0.0;
        for (double grading : {0.0, 2.0, 4.0, 6.0, 8.0}) {
            const auto A = graded_matrix(kN, grading);
            const auto want = reference(A);
            const double cta_rel = max_rel_diff(solve_cta(ctx, A), want);
            const double jac_rel = max_rel_diff(solve_jacobi(ctx, A), want);
            const std::string tag = "grading 1e-" + std::to_string(static_cast<int>(grading));
            report_magnitude(tag + ": syev_cta", cta_rel);
            report_magnitude(tag + ": jacobi", jac_rel);
            if (grading >= 6.0 && jac_rel > 0.0) best_ratio = std::max(best_ratio, cta_rel / jac_rel);
        }
        report_check("Jacobi wins by orders of magnitude at the graded end", best_ratio > 10.0);
    }

    // -----------------------------------------------------------------------
    // Tuning the Jacobi sweep
    //
    // JacobiParams::tol_multiplier scales the relative threshold a rotation
    // has to exceed to be applied, and max_sweeps caps the cyclic sweeps.
    // Convergence normally takes well under 10 sweeps; cutting it short is
    // what actually costs accuracy.
    // -----------------------------------------------------------------------
    static void tuning_section(Queue& ctx) {
        section("Tuning the Jacobi sweep");

        const auto A = graded_matrix(kN, 7.0);
        const auto want = reference(A);

        for (double mult : {1.0, 1e4, 1e8}) {
            JacobiParams<double> params;
            params.tol_multiplier = mult;
            params.max_sweeps = 30;
            report_magnitude("tol_multiplier = 1e" + std::to_string(static_cast<int>(std::log10(mult))),
                             max_rel_diff(solve_jacobi(ctx, A, params), want));
        }

        JacobiParams<double> few;
        few.max_sweeps = 1;
        const double stunted = max_rel_diff(solve_jacobi(ctx, A, few), want);
        JacobiParams<double> enough;
        enough.max_sweeps = 30;
        const double converged = max_rel_diff(solve_jacobi(ctx, A, enough), want);
        report_magnitude("max_sweeps = 1", stunted);
        report_magnitude("max_sweeps = 30", converged);
        report_check("stopping after one sweep costs accuracy", stunted > converged);
    }

    // -----------------------------------------------------------------------
    // Eigenvectors too
    // -----------------------------------------------------------------------
    static void vectors_section(Queue& ctx) {
        section("Eigenvectors too");

        const auto A = graded_matrix(kN, 7.0);
        auto M = broadcast(A, 1);
        UnifiedVector<double> w(kN);
        syev_jacobi_cta<B>(ctx, M.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower);
        ctx.wait();

        std::vector<double> as_returned(w.begin(), w.begin() + kN);
        auto V = to_host(M, 0);
        report_error("|V^T V - I|", orthogonality_error(V), 1e-12);
        // The residual is measured against the norm of A, so like the absolute
        // error it is dominated by the largest eigenvalue.
        report_error("|A V - V diag(w)|", eigen_residual(A, V, as_returned), 1e-12);
    }

    // -----------------------------------------------------------------------
    // A wide spectrum is not enough
    //
    // Build Q diag(w) Q^T with a random orthogonal Q and the same eigenvalue
    // range. The matrix is no longer of the form D S D: forming it already
    // rounded the small eigenvalues away, so both solvers lose them and Jacobi
    // has nothing left to recover. Reach for syev_jacobi_cta because your data
    // is *graded*, not merely because its spectrum is wide.
    // -----------------------------------------------------------------------
    static void not_automatic_section(Queue& ctx) {
        section("A wide spectrum is not enough");

        std::vector<double> spectrum(kN);
        for (int i = 0; i < kN; ++i) spectrum[i] = std::pow(10.0, -14.0 * i / (kN - 1));
        std::sort(spectrum.begin(), spectrum.end());

        const auto A = symmetric_with_eigenvalues<double>(spectrum, 91);
        const double cta_rel = max_rel_diff(solve_cta(ctx, A), spectrum);
        const double jac_rel = max_rel_diff(solve_jacobi(ctx, A), spectrum);

        report_magnitude("Q diag(w) Q^T: syev_cta relative error", cta_rel);
        report_magnitude("Q diag(w) Q^T: jacobi relative error", jac_rel);
        report_check("neither solver recovers the small eigenvalues here", cta_rel > 1e-6 && jac_rel > 1e-6);
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("10. Relative accuracy: why syev_jacobi_cta exists")
