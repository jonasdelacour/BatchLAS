// 8. Tridiagonal eigensolvers
//
// Once a matrix is tridiagonal (example 07), these solve the eigenproblem:
//
//   steqr        implicit QR iteration
//   steqr_cta    the same, one work-group per matrix (n <= 32, GPU)
//   stedc        divide and conquer
//   stedc_flat   divide and conquer, non-recursive driver
//   tridiagonal_solver  a convenience wrapper taking plain spans
//
// The input is always (d, e): the diagonal, length n, and the off-diagonal,
// length n-1.

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

constexpr int kBatch = 2;
constexpr double kPi = 3.14159265358979323846;

// A tridiagonal Toeplitz matrix — diagonal `a`, off-diagonal `b` — has the
// closed-form spectrum a + 2b cos(k pi / (n+1)), k = 1..n. That gives an exact
// reference to check against, with no reference solver in the loop.
std::vector<double> toeplitz_spectrum(int n, double a, double b) {
    std::vector<double> w(n);
    for (int k = 1; k <= n; ++k) w[k - 1] = a + 2.0 * b * std::cos(k * kPi / (n + 1));
    std::sort(w.begin(), w.end());
    return w;
}

template <Backend B>
struct Example {
    static constexpr int n = 32;
    static constexpr double kDiag = 2.0;
    static constexpr double kOff = -1.0;

    static void run(Queue& ctx) {
        all_solvers_section(ctx);
        values_only_section(ctx);
        sort_order_section(ctx);
        steqr_tuning_section(ctx);
        stedc_tuning_section(ctx);
        convenience_section(ctx);
        generator_section(ctx);
    }

    // Fill (d, e) with the tridiagonal Toeplitz matrix.
    static void fill_toeplitz(Vector<double>& d, Vector<double>& e) {
        for (int b = 0; b < kBatch; ++b) {
            for (int i = 0; i < n; ++i) d(i, b) = kDiag;
            for (int i = 0; i + 1 < n; ++i) e(i, b) = kOff;
        }
    }

    static std::vector<double> host_values(Vector<double>& w, int batch_item = 0) {
        std::vector<double> out(n);
        for (int i = 0; i < n; ++i) out[i] = w(i, batch_item);
        return out;
    }

    // The dense form of the tridiagonal matrix these solvers are given.
    static HostMatrix<double> dense_toeplitz() {
        HostMatrix<double> T(n, n);
        for (int i = 0; i < n; ++i) T(i, i) = kDiag;
        for (int i = 0; i + 1 < n; ++i) {
            T(i + 1, i) = kOff;
            T(i, i + 1) = kOff;
        }
        return T;
    }

    // -----------------------------------------------------------------------
    // All four solvers on the same problem
    //
    // Each takes (d, e), a `eigenvalues` vector to fill, and a workspace.
    // JobType::EigenVectors additionally fills an n x n matrix you pass as the
    // last argument. d and e are consumed — treat them as destroyed.
    // -----------------------------------------------------------------------
    static void all_solvers_section(Queue& ctx) {
        section("All four solvers on the same problem");

        const auto want = toeplitz_spectrum(n, kDiag, kOff);

        // steqr
        {
            Vector<double> d(n, kBatch), e(n - 1, kBatch), w(n, kBatch);
            fill_toeplitz(d, e);
            auto V = Matrix<double>::Zeros(n, n, kBatch);
            UnifiedVector<std::byte> ws(steqr_buffer_size<double>(ctx, VectorView<double>(d), VectorView<double>(e),
                                                                   VectorView<double>(w), JobType::EigenVectors));
            steqr<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                     JobType::EigenVectors, SteqrParams<double>(), V.view());
            ctx.wait();
            report_error("steqr eigenvalues", max_abs_diff(host_values(w), want), 1e-10);
            report_error("steqr |T V - V diag(w)|", eigen_residual(dense_toeplitz(), to_host(V, 0), host_values(w)),
                         1e-9);
        }

        // steqr_cta
        if constexpr (has_cta_variants<B>) {
            if (supports_cta(ctx)) {
                Vector<double> d(n, kBatch), e(n - 1, kBatch), w(n, kBatch);
                fill_toeplitz(d, e);
                auto V = Matrix<double>::Zeros(n, n, kBatch);
                UnifiedVector<std::byte> ws(steqr_cta_buffer_size<double>(
                    ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), JobType::EigenVectors));
                steqr_cta<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                             JobType::EigenVectors, SteqrParams<double>(), V.view());
                ctx.wait();
                report_error("steqr_cta eigenvalues", max_abs_diff(host_values(w), want), 1e-10);
                report_error("steqr_cta |T V - V diag(w)|",
                             eigen_residual(dense_toeplitz(), to_host(V, 0), host_values(w)), 1e-9);
            } else {
                report_skip("steqr_cta", "needs a GPU with sub-group width 32");
            }
        } else {
            report_skip("steqr_cta", "not instantiated for this backend");
        }

        // stedc
        {
            Vector<double> d(n, kBatch), e(n - 1, kBatch), w(n, kBatch);
            fill_toeplitz(d, e);
            auto V = Matrix<double>::Zeros(n, n, kBatch);
            StedcParams<double> params;
            UnifiedVector<std::byte> ws(stedc_workspace_size<B, double>(ctx, n, kBatch, JobType::EigenVectors, params));
            stedc<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                     JobType::EigenVectors, params, V.view());
            ctx.wait();
            report_error("stedc eigenvalues", max_abs_diff(host_values(w), want), 1e-10);
            report_error("stedc |T V - V diag(w)|", eigen_residual(dense_toeplitz(), to_host(V, 0), host_values(w)),
                         1e-9);
        }

        // stedc_flat
        {
            Vector<double> d(n, kBatch), e(n - 1, kBatch), w(n, kBatch);
            fill_toeplitz(d, e);
            auto V = Matrix<double>::Zeros(n, n, kBatch);
            StedcParams<double> params;
            UnifiedVector<std::byte> ws(
                stedc_flat_workspace_size<B, double>(ctx, n, kBatch, JobType::EigenVectors, params));
            stedc_flat<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                          JobType::EigenVectors, params, V.view());
            ctx.wait();
            report_error("stedc_flat eigenvalues", max_abs_diff(host_values(w), want), 1e-10);
            // Known defect: stedc_flat's eigenvalues are right but its
            // eigenvectors do not satisfy T V = V diag(w). Its columns are
            // orthonormal, so only the residual exposes it — reported without
            // a tolerance so it stays visible. See the README.
            report_magnitude("stedc_flat |V^T V - I|", orthogonality_error(to_host(V, 0)));
            report_magnitude("stedc_flat |T V - V diag(w)| (known bad)",
                             eigen_residual(dense_toeplitz(), to_host(V, 0), host_values(w)));
        }
    }

    // -----------------------------------------------------------------------
    // Eigenvalues only
    //
    // JobType::NoEigenVectors and no eigenvector matrix. Note the known defect
    // for stedc here: it returns wrong eigenvalues in this mode, which is why
    // the Python bindings always request vectors internally and throw them
    // away. steqr is unaffected.
    // -----------------------------------------------------------------------
    static void values_only_section(Queue& ctx) {
        section("Eigenvalues only");

        const auto want = toeplitz_spectrum(n, kDiag, kOff);

        // steqr's values-only path uses a kernel the SYCL native-CPU device
        // does not support, so it is attempted rather than assumed. See the
        // known issues in the README.
        try {
            Vector<double> d(n, kBatch), e(n - 1, kBatch), w(n, kBatch);
            fill_toeplitz(d, e);
            UnifiedVector<std::byte> ws(steqr_buffer_size<double>(ctx, VectorView<double>(d), VectorView<double>(e),
                                                                   VectorView<double>(w), JobType::NoEigenVectors));
            steqr<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                     JobType::NoEigenVectors);
            ctx.wait();
            report_error("steqr(NoEigenVectors)", max_abs_diff(host_values(w), want), 1e-10);
        } catch (const std::exception& ex) {
            report_skip("steqr(NoEigenVectors)", std::string("unsupported on this device: ") + ex.what());
        }

        // stedc has a known defect in this mode: it slices an eigenvector
        // output it was told not to produce, and throws
        // "Invalid slice dimensions". The workaround the Python bindings use —
        // and what syev_blocked does internally — is to ask for vectors and
        // discard them.
        Vector<double> d2(n, kBatch), e2(n - 1, kBatch), w2(n, kBatch);
        fill_toeplitz(d2, e2);
        auto discarded = Matrix<double>::Zeros(n, n, kBatch);
        StedcParams<double> params;
        UnifiedVector<std::byte> ws2(stedc_workspace_size<B, double>(ctx, n, kBatch, JobType::EigenVectors, params));
        stedc<B>(ctx, VectorView<double>(d2), VectorView<double>(e2), VectorView<double>(w2), ws2.to_span(),
                 JobType::EigenVectors, params, discarded.view());
        ctx.wait();
        report_error("stedc, asking for vectors and discarding them", max_abs_diff(host_values(w2), want), 1e-10);
        report_skip("stedc(NoEigenVectors)", "known defect: throws \"Invalid slice dimensions\"; use the workaround above");
    }

    // -----------------------------------------------------------------------
    // Sort order
    //
    // SteqrParams::sort_order flips ascending to descending, and permutes the
    // eigenvectors to match. sort = false leaves whatever order the iteration
    // produced.
    // -----------------------------------------------------------------------
    static void sort_order_section(Queue& ctx) {
        section("Sort order");

        auto want = toeplitz_spectrum(n, kDiag, kOff);
        std::sort(want.begin(), want.end(), std::greater<double>());

        Vector<double> d(n, kBatch), e(n - 1, kBatch), w(n, kBatch);
        fill_toeplitz(d, e);
        auto V = Matrix<double>::Zeros(n, n, kBatch);
        SteqrParams<double> params;
        params.sort_order = SortOrder::Descending;
        UnifiedVector<std::byte> ws(steqr_buffer_size<double>(ctx, VectorView<double>(d), VectorView<double>(e),
                                                               VectorView<double>(w), JobType::EigenVectors, params));
        steqr<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                 JobType::EigenVectors, params, V.view());
        ctx.wait();
        report_error("SortOrder::Descending", max_abs_diff(host_values(w), want), 1e-10);
    }

    // -----------------------------------------------------------------------
    // Tuning the QR iteration
    //
    // SteqrParams::block_size controls how many Givens rotations are applied
    // together — larger blocks do redundant flops but reuse memory better.
    // max_sweeps caps the iteration, and zero_threshold decides when an
    // off-diagonal entry counts as zero (which is what splits the problem).
    // -----------------------------------------------------------------------
    static void steqr_tuning_section(Queue& ctx) {
        section("Tuning the QR iteration");

        const auto want = toeplitz_spectrum(n, kDiag, kOff);

        for (size_t block_size : {size_t{1}, size_t{8}, size_t{32}}) {
            // On the NETLIB path, steqr_buffer_size under-reports for the
            // larger blocked settings and the routine then overruns its own
            // workspace. Attempted rather than assumed; see the README.
            try {
                Vector<double> d(n, kBatch), e(n - 1, kBatch), w(n, kBatch);
                fill_toeplitz(d, e);
                auto V = Matrix<double>::Zeros(n, n, kBatch);
                SteqrParams<double> params;
                params.block_size = block_size;
                params.block_rotations = block_size > 1;
                UnifiedVector<std::byte> ws(steqr_buffer_size<double>(
                    ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), JobType::EigenVectors,
                    params));
                steqr<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                         JobType::EigenVectors, params, V.view());
                ctx.wait();
                report_error("block_size = " + std::to_string(block_size), max_abs_diff(host_values(w), want), 1e-10);
            } catch (const std::exception& ex) {
                report_skip("block_size = " + std::to_string(block_size),
                            std::string("workspace sizing is wrong here: ") + ex.what());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Tuning divide and conquer
    //
    // StedcParams::recursion_threshold is the size below which stedc stops
    // dividing and calls the leaf QR solver; 0 means "use the tuning tables".
    // merge_variant selects how the merge step is dispatched.
    // -----------------------------------------------------------------------
    static void stedc_tuning_section(Queue& ctx) {
        section("Tuning divide and conquer");

        const auto want = toeplitz_spectrum(n, kDiag, kOff);

        for (int64_t threshold : {int64_t{0}, int64_t{8}, int64_t{16}}) {
            Vector<double> d(n, kBatch), e(n - 1, kBatch), w(n, kBatch);
            fill_toeplitz(d, e);
            auto V = Matrix<double>::Zeros(n, n, kBatch);
            StedcParams<double> params;
            params.recursion_threshold = threshold;
            UnifiedVector<std::byte> ws(
                stedc_workspace_size<B, double>(ctx, n, kBatch, JobType::EigenVectors, params));
            stedc<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                     JobType::EigenVectors, params, V.view());
            ctx.wait();
            report_error("recursion_threshold = " + std::to_string(threshold), max_abs_diff(host_values(w), want),
                         1e-10);
        }

        // The merge variant is a performance knob, not a numerical one.
        Vector<double> d(n, kBatch), e(n - 1, kBatch), w(n, kBatch);
        fill_toeplitz(d, e);
        auto V = Matrix<double>::Zeros(n, n, kBatch);
        StedcParams<double> params;
        params.merge_variant = StedcMergeVariant::Baseline;
        UnifiedVector<std::byte> ws(stedc_workspace_size<B, double>(ctx, n, kBatch, JobType::EigenVectors, params));
        stedc<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                 JobType::EigenVectors, params, V.view());
        ctx.wait();
        report_error("merge_variant = Baseline", max_abs_diff(host_values(w), want), 1e-10);
    }

    // -----------------------------------------------------------------------
    // tridiagonal_solver — the convenience driver
    //
    // Takes plain `Span`s and the dimensions rather than Vector views, which
    // suits code that already has the coefficients in flat arrays — the
    // Lanczos recurrence in example 09, for instance.
    //
    // Its QR iteration does not converge reliably; accuracy varies with n and
    // with the data. Prefer steqr or stedc.
    // -----------------------------------------------------------------------
    static void convenience_section(Queue& ctx) {
        section("tridiagonal_solver - the convenience driver");

        const auto want = toeplitz_spectrum(n, kDiag, kOff);

        UnifiedVector<double> alphas(static_cast<size_t>(n) * kBatch);
        UnifiedVector<double> betas(static_cast<size_t>(n) * kBatch);
        UnifiedVector<double> w(static_cast<size_t>(n) * kBatch);
        for (int b = 0; b < kBatch; ++b) {
            for (int i = 0; i < n; ++i) alphas[b * n + i] = kDiag;
            for (int i = 0; i < n; ++i) betas[b * n + i] = (i + 1 < n) ? kOff : 0.0;
        }

        auto Q = Matrix<double>::Zeros(n, n, kBatch);
        UnifiedVector<std::byte> ws(tridiagonal_solver_buffer_size<B, double>(ctx, n, kBatch, JobType::EigenVectors));
        tridiagonal_solver<B>(ctx, alphas.to_span(), betas.to_span(), w.to_span(), ws.to_span(), JobType::EigenVectors,
                              Q.view(), n, kBatch);
        ctx.wait();

        std::vector<double> got(w.begin(), w.begin() + n);
        std::sort(got.begin(), got.end());
        // Reported without a tolerance: this solver is known not to converge
        // reliably. See the known issues in the README.
        report_magnitude("tridiagonal_solver error vs the exact spectrum", max_abs_diff(got, want));
    }

    // -----------------------------------------------------------------------
    // A generator with a known closed-form spectrum
    //
    // Matrix::TriDiagToeplitz builds the same matrix as a dense Matrix, which
    // is handy for checking a *dense* solver against the closed form. Here it
    // confirms syev finds the eigenvalues we just computed by hand.
    // -----------------------------------------------------------------------
    static void generator_section(Queue& ctx) {
        section("A generator with a known closed-form spectrum");

        const int m = 24;
        auto A = Matrix<double>::TriDiagToeplitz(m, kDiag, kOff, kOff, kBatch);
        UnifiedVector<double> w(static_cast<size_t>(m) * kBatch);
        UnifiedVector<std::byte> ws(
            syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower));
        syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws.to_span());
        ctx.wait();

        std::vector<double> got(w.begin(), w.begin() + m);
        std::sort(got.begin(), got.end());
        report_error("syev on TriDiagToeplitz vs the closed form",
                     max_abs_diff(got, toeplitz_spectrum(m, kDiag, kOff)), 1e-9);
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("8. Tridiagonal eigensolvers")
