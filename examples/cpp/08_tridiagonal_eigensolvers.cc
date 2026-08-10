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
// length n-1. Both are consumed — treat them as destroyed by the call.

#include <cmath>
#include <cstddef>
#include <iostream>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_utils.hh"

using namespace batchlas;
using namespace examples;

namespace {

constexpr int kBatch = 2;
constexpr int kN = 16;
constexpr double kDiag = 2.0;
constexpr double kOff = -1.0;
constexpr double kPi = 3.14159265358979323846;

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        exact_section(ctx);
        all_solvers_section(ctx);
        sort_order_section(ctx);
        steqr_tuning_section(ctx);
        stedc_tuning_section(ctx);
        convenience_section(ctx);
        generator_section(ctx);
    }

    // The tridiagonal Toeplitz matrix used throughout: diagonal 2,
    // off-diagonal -1.
    static void fill(Vector<double>& d, Vector<double>& e) {
        for (int b = 0; b < kBatch; ++b) {
            for (int i = 0; i < kN; ++i) d(i, b) = kDiag;
            for (int i = 0; i + 1 < kN; ++i) e(i, b) = kOff;
        }
    }

    // That matrix has a closed-form spectrum, so there is something to compare
    // against without running another solver.
    static void exact_section(Queue& ctx) {
        section("The problem, and its exact answer");

        std::cout << "tridiagonal Toeplitz, diag " << kDiag << ", off-diag " << kOff << ", n = " << kN << "\n";
        std::cout << "exact eigenvalues a + 2b cos(k pi / (n+1)), ascending:";
        for (int k = 1; k <= 6; ++k) std::cout << " " << kDiag + 2.0 * kOff * std::cos(k * kPi / (kN + 1));
        std::cout << " ...\n";
    }

    // All four solvers on the same problem
    //
    // Each takes (d, e), a vector to fill with eigenvalues, and a workspace.
    // JobType::EigenVectors additionally fills an n x n matrix passed as the
    // last argument.
    static void all_solvers_section(Queue& ctx) {
        section("All four solvers on the same problem");

        // steqr
        {
            Vector<double> d(kN, kBatch), e(kN - 1, kBatch), w(kN, kBatch);
            fill(d, e);
            auto V = Matrix<double>::Zeros(kN, kN, kBatch);
            UnifiedVector<std::byte> ws(steqr_buffer_size<double>(ctx, VectorView<double>(d), VectorView<double>(e),
                                                                   VectorView<double>(w), JobType::EigenVectors));
            steqr<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                     JobType::EigenVectors, SteqrParams<double>(), V.view());
            ctx.wait();
            std::cout << "steqr:       ";
            w.batch_item(0).print(std::cout, 6);
        }

        // steqr_cta
        if constexpr (has_cta_variants<B>) {
            if (supports_cta(ctx)) {
                Vector<double> d(kN, kBatch), e(kN - 1, kBatch), w(kN, kBatch);
                fill(d, e);
                auto V = Matrix<double>::Zeros(kN, kN, kBatch);
                UnifiedVector<std::byte> ws(steqr_cta_buffer_size<double>(
                    ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), JobType::EigenVectors));
                steqr_cta<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                             JobType::EigenVectors, SteqrParams<double>(), V.view());
                ctx.wait();
                std::cout << "steqr_cta:   ";
                w.batch_item(0).print(std::cout, 6);
            } else {
                skip("steqr_cta", "needs a GPU with sub-group width 32");
            }
        } else {
            skip("steqr_cta", "not instantiated for this backend");
        }

        // stedc
        {
            Vector<double> d(kN, kBatch), e(kN - 1, kBatch), w(kN, kBatch);
            fill(d, e);
            auto V = Matrix<double>::Zeros(kN, kN, kBatch);
            StedcParams<double> params;
            UnifiedVector<std::byte> ws(
                stedc_workspace_size<B, double>(ctx, kN, kBatch, JobType::EigenVectors, params));
            stedc<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                     JobType::EigenVectors, params, V.view());
            ctx.wait();
            std::cout << "stedc:       ";
            w.batch_item(0).print(std::cout, 6);
        }

        // stedc_flat. Its eigenvalues are right; its eigenvectors do not
        // satisfy T V = V diag(w) on CUDA. See the known issues in the README.
        {
            Vector<double> d(kN, kBatch), e(kN - 1, kBatch), w(kN, kBatch);
            fill(d, e);
            auto V = Matrix<double>::Zeros(kN, kN, kBatch);
            StedcParams<double> params;
            UnifiedVector<std::byte> ws(
                stedc_flat_workspace_size<B, double>(ctx, kN, kBatch, JobType::EigenVectors, params));
            stedc_flat<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                          JobType::EigenVectors, params, V.view());
            ctx.wait();
            std::cout << "stedc_flat:  ";
            w.batch_item(0).print(std::cout, 6);
        }

        // On eigenvalues-only mode: stedc and stedc_flat throw
        // "Invalid slice dimensions" with JobType::NoEigenVectors, and steqr's
        // values-only path is unsupported on the SYCL native-CPU device. Ask
        // for vectors and discard them, as syev_blocked does internally.
    }

    // Sort order
    //
    // SteqrParams::sort_order flips ascending to descending and permutes the
    // eigenvectors to match. `sort = false` leaves whatever order the iteration
    // produced.
    static void sort_order_section(Queue& ctx) {
        section("Sort order");

        Vector<double> d(kN, kBatch), e(kN - 1, kBatch), w(kN, kBatch);
        fill(d, e);
        auto V = Matrix<double>::Zeros(kN, kN, kBatch);
        SteqrParams<double> params;
        params.sort_order = SortOrder::Descending;

        UnifiedVector<std::byte> ws(steqr_buffer_size<double>(ctx, VectorView<double>(d), VectorView<double>(e),
                                                               VectorView<double>(w), JobType::EigenVectors, params));
        steqr<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                 JobType::EigenVectors, params, V.view());
        ctx.wait();
        std::cout << "descending:  ";
        w.batch_item(0).print(std::cout, 6);
    }

    // Tuning the QR iteration
    //
    // SteqrParams::block_size controls how many Givens rotations are applied
    // together — larger blocks do redundant flops but reuse memory better.
    // max_sweeps caps the iteration and zero_threshold decides when an
    // off-diagonal entry counts as zero, which is what splits the problem.
    static void steqr_tuning_section(Queue& ctx) {
        section("Tuning the QR iteration");

        for (size_t block_size : {size_t{1}, size_t{8}, size_t{32}}) {
            // On NETLIB, steqr_buffer_size under-reports for the larger blocked
            // settings and the routine then overruns its workspace. See the
            // known issues in the README.
            try {
                Vector<double> d(kN, kBatch), e(kN - 1, kBatch), w(kN, kBatch);
                fill(d, e);
                auto V = Matrix<double>::Zeros(kN, kN, kBatch);
                SteqrParams<double> params;
                params.block_size = block_size;
                params.block_rotations = block_size > 1;

                UnifiedVector<std::byte> ws(steqr_buffer_size<double>(ctx, VectorView<double>(d),
                                                                       VectorView<double>(e), VectorView<double>(w),
                                                                       JobType::EigenVectors, params));
                steqr<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                         JobType::EigenVectors, params, V.view());
                ctx.wait();
                std::cout << "block_size = " << block_size << ": ";
                w.batch_item(0).print(std::cout, 4);
            } catch (const std::exception& ex) {
                skip("block_size = " + std::to_string(block_size), ex.what());
            }
        }
    }

    // Tuning divide and conquer
    //
    // StedcParams::recursion_threshold is the size below which stedc stops
    // dividing and calls the leaf QR solver; 0 means "use the tuning tables".
    // merge_variant selects how the merge step is dispatched — a performance
    // knob, not a numerical one.
    static void stedc_tuning_section(Queue& ctx) {
        section("Tuning divide and conquer");

        auto solve = [&](StedcParams<double> params, const char* label) {
            Vector<double> d(kN, kBatch), e(kN - 1, kBatch), w(kN, kBatch);
            fill(d, e);
            auto V = Matrix<double>::Zeros(kN, kN, kBatch);
            UnifiedVector<std::byte> ws(
                stedc_workspace_size<B, double>(ctx, kN, kBatch, JobType::EigenVectors, params));
            stedc<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                     JobType::EigenVectors, params, V.view());
            ctx.wait();
            std::cout << label << ": ";
            w.batch_item(0).print(std::cout, 4);
        };

        for (int64_t threshold : {int64_t{0}, int64_t{8}}) {
            StedcParams<double> params;
            params.recursion_threshold = threshold;
            solve(params, threshold == 0 ? "recursion_threshold = 0 (tuned)" : "recursion_threshold = 8");
        }

        StedcParams<double> baseline;
        baseline.merge_variant = StedcMergeVariant::Baseline;
        solve(baseline, "merge_variant = Baseline");
    }

    // tridiagonal_solver — the convenience driver
    //
    // Takes plain `Span`s and the dimensions rather than Vector views, which
    // suits code that already has the coefficients in flat arrays — the Lanczos
    // recurrence in example 09, for instance.
    //
    // Its QR iteration does not converge reliably; accuracy varies with n and
    // with the data. Prefer steqr or stedc.
    static void convenience_section(Queue& ctx) {
        section("tridiagonal_solver - the convenience driver");

        UnifiedVector<double> alphas(static_cast<size_t>(kN) * kBatch);
        UnifiedVector<double> betas(static_cast<size_t>(kN) * kBatch);
        UnifiedVector<double> w(static_cast<size_t>(kN) * kBatch);
        for (int b = 0; b < kBatch; ++b) {
            for (int i = 0; i < kN; ++i) alphas[b * kN + i] = kDiag;
            for (int i = 0; i < kN; ++i) betas[b * kN + i] = (i + 1 < kN) ? kOff : 0.0;
        }

        auto Q = Matrix<double>::Zeros(kN, kN, kBatch);
        UnifiedVector<std::byte> ws(
            tridiagonal_solver_buffer_size<B, double>(ctx, kN, kBatch, JobType::EigenVectors));
        tridiagonal_solver<B>(ctx, alphas.to_span(), betas.to_span(), w.to_span(), ws.to_span(),
                              JobType::EigenVectors, Q.view(), kN, kBatch);
        ctx.wait();
        print_values("tridiagonal_solver (compare with the exact values above)", w.to_span(), 6);
    }

    // A generator with a known closed-form spectrum
    //
    // Matrix::TriDiagToeplitz builds the same matrix densely, which is handy
    // for driving a *dense* solver with a problem whose answer you know.
    static void generator_section(Queue& ctx) {
        section("A generator with a known closed-form spectrum");

        auto A = Matrix<double>::TriDiagToeplitz(kN, kDiag, kOff, kOff, kBatch);
        UnifiedVector<double> w(static_cast<size_t>(kN) * kBatch);
        UnifiedVector<std::byte> ws(
            syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower));
        syev<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
        ctx.wait();
        print_values("syev on TriDiagToeplitz", w.to_span(), 6);
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("8. Tridiagonal eigensolvers")
