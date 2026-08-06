// 10. Relative accuracy: why syev_jacobi_cta exists
//
// syev_cta and syev_jacobi_cta solve the same problem and agree to machine
// precision on a well-scaled matrix. On a *graded* one they do not — and
// absolute error hides the difference completely.
//
// The matrix class that matters is A = D S D, with D a diagonal scaling
// spanning many orders of magnitude and S well conditioned. Every entry still
// carries full relative precision, so the tiny eigenvalues are well determined
// by the data — but only an algorithm that never differences numbers of wildly
// different magnitude can recover them. Tridiagonalization does difference
// them; two-sided Jacobi with a relative stopping criterion does not.
// (Demmel & Veselic, SIMAX 13(4), 1992.)
//
// Both solvers are CTA variants, so this example is GPU-only.

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_utils.hh"

using namespace batchlas;
using namespace examples;

namespace {

constexpr int kN = 16;

// A = D S D. S has a unit diagonal and modest off-diagonal entries; D grades
// the rows and columns from 1 down to 10^-grading.
Matrix<double> graded_matrix(Queue& ctx, double grading, unsigned seed = 42) {
    auto A = Matrix<double>::Random(kN, kN, /*hermitian=*/true, 1, seed);
    ctx.wait();
    for (int i = 0; i < kN; ++i) A(i, i, 0) = 1.0;
    for (int j = 0; j < kN; ++j)
        for (int i = 0; i < kN; ++i)
            if (i != j) A(i, j, 0) *= 0.25;

    for (int j = 0; j < kN; ++j) {
        const double dj = std::pow(10.0, -grading * j / (kN - 1));
        for (int i = 0; i < kN; ++i) {
            const double di = std::pow(10.0, -grading * i / (kN - 1));
            A(i, j, 0) *= di * dj;
        }
    }
    return A;
}

// Print two spectra side by side, smallest first, with the relative gap.
void compare(const UnifiedVector<double>& cta, const UnifiedVector<double>& jacobi, int count = 6) {
    std::cout << "      syev_cta            syev_jacobi_cta      relative difference\n";
    for (int i = 0; i < count; ++i) {
        const double a = cta[i], b = jacobi[i];
        const double rel = (b != 0.0) ? std::abs(a - b) / std::abs(b) : 0.0;
        std::cout << "  " << std::scientific << std::setprecision(6) << std::setw(14) << a << "     " << std::setw(14)
                  << b << "     " << std::setprecision(2) << rel << "\n";
    }
    std::cout << std::defaultfloat;
}

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        if constexpr (!has_cta_variants<B>) {
            skip("this example", "the CTA variants are not instantiated for this backend");
            return;
        } else {
            if (!supports_cta(ctx)) {
                skip("this example", "needs a GPU with sub-group width 32");
                return;
            }
            well_scaled_section(ctx);
            graded_section(ctx);
            sweep_section(ctx);
            tuning_section(ctx);
            vectors_section(ctx);
            not_automatic_section(ctx);
        }
    }

    static UnifiedVector<double> solve_cta(Queue& ctx, const Matrix<double>& A) {
        auto M = A.clone();
        UnifiedVector<double> w(kN);
        UnifiedVector<std::byte> ws(syev_cta_buffer_size<B>(ctx, M.view(), JobType::NoEigenVectors));
        syev_cta<B>(ctx, M.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws.to_span());
        ctx.wait();
        return w;
    }

    static UnifiedVector<double> solve_jacobi(Queue& ctx, const Matrix<double>& A,
                                              JacobiParams<double> params = JacobiParams<double>()) {
        auto M = A.clone();
        UnifiedVector<double> w(kN);
        syev_jacobi_cta<B>(ctx, M.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, Span<std::byte>(),
                           params);
        ctx.wait();
        return w;
    }

    // On a well-scaled matrix, both are fine
    static void well_scaled_section(Queue& ctx) {
        section("On a well-scaled matrix, both agree");

        auto A = graded_matrix(ctx, 0.0);  // D = I
        compare(solve_cta(ctx, A), solve_jacobi(ctx, A), 4);
    }

    // A graded matrix — where they part company
    //
    // The largest eigenvalue sets the norm, so an absolute error of eps*|A| is
    // enormous *relative* to the smallest. Both solvers hit that bound; only
    // one does better.
    static void graded_section(Queue& ctx) {
        section("A graded matrix: D S D with D spanning 1e-7");

        auto A = graded_matrix(ctx, 7.0);
        const auto cta = solve_cta(ctx, A);
        const auto jac = solve_jacobi(ctx, A);

        std::cout << "smallest eigenvalues — the two solvers agree to only a few digits:\n";
        compare(cta, jac, 4);
        std::cout << "largest eigenvalues — here the two agree:\n";
        std::cout << "  syev_cta " << cta[kN - 1] << "   jacobi " << jac[kN - 1] << "\n";
        std::cout << "\nBoth are correct to ~1e-16 times the norm of A, so an absolute-error\n"
                     "check calls them identical. The disagreement is entirely in the digits\n"
                     "an absolute measure cannot see — and it is the tridiagonal path that\n"
                     "loses them, since Jacobi's stopping criterion is relative.\n";
    }

    // Sweeping the grading
    //
    // The gap opens as the scaling widens: identical at grading 0, then orders
    // of magnitude apart.
    static void sweep_section(Queue& ctx) {
        section("Sweeping the grading");

        std::cout << "  grading   smallest from syev_cta   smallest from jacobi\n";
        for (double grading : {0.0, 2.0, 4.0, 6.0, 8.0}) {
            auto A = graded_matrix(ctx, grading);
            const auto cta = solve_cta(ctx, A);
            const auto jac = solve_jacobi(ctx, A);
            std::cout << "  1e-" << std::setw(2) << static_cast<int>(grading) << "     " << std::scientific
                      << std::setprecision(6) << std::setw(14) << cta[0] << "           " << std::setw(14) << jac[0]
                      << std::defaultfloat << "\n";
        }
    }

    // Tuning the Jacobi sweep
    //
    // JacobiParams::tol_multiplier scales the relative threshold a rotation has
    // to exceed to be applied, and max_sweeps caps the cyclic sweeps.
    // Convergence normally takes well under 10 sweeps; cutting it short is what
    // actually costs accuracy.
    static void tuning_section(Queue& ctx) {
        section("Tuning the Jacobi sweep");

        auto A = graded_matrix(ctx, 7.0);

        for (double mult : {1.0, 1e4, 1e8}) {
            JacobiParams<double> params;
            params.tol_multiplier = mult;
            params.max_sweeps = 30;
            const auto w = solve_jacobi(ctx, A, params);
            std::cout << "tol_multiplier = 1e" << static_cast<int>(std::log10(mult)) << ": smallest " << w[0] << "\n";
        }

        for (size_t sweeps : {size_t{1}, size_t{30}}) {
            JacobiParams<double> params;
            params.max_sweeps = sweeps;
            const auto w = solve_jacobi(ctx, A, params);
            std::cout << "max_sweeps = " << sweeps << ": smallest " << w[0] << "\n";
        }
    }

    // Eigenvectors too
    static void vectors_section(Queue& ctx) {
        section("Eigenvectors too");

        auto A = graded_matrix(ctx, 7.0);
        auto M = A.clone();
        UnifiedVector<double> w(kN);
        syev_jacobi_cta<B>(ctx, M.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower);
        ctx.wait();

        // V^T V is the identity when the vectors are orthonormal.
        auto G = Matrix<double>::Zeros(kN, kN, 1);
        gemm<B>(ctx, M, M, G, 1.0, 0.0, Transpose::Trans, Transpose::NoTrans);
        ctx.wait();
        std::cout << "V^T V, top-left corner:\n";
        G.view()[0].print(std::cout, 4, 4);
    }

    // A wide spectrum is not enough
    //
    // Matrix::Random with hermitian = true has a spectrum spread around zero,
    // but it is not of the form D S D. Scale it by a diagonal *after* the fact
    // and the grading is genuine; take an arbitrary matrix with a wide spectrum
    // and it is not. Reach for syev_jacobi_cta because your data is graded, not
    // merely because its eigenvalues span a wide range.
    static void not_automatic_section(Queue& ctx) {
        section("A wide spectrum is not enough");

        // Same eigenvalue range as the graded case, but the smallness is spread
        // over the whole matrix rather than living in a diagonal scaling.
        auto A = Matrix<double>::Random(kN, kN, true, 1, 91);
        ctx.wait();
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kN; ++i) A(i, j, 0) *= 1e-7;
        for (int i = 0; i < kN / 2; ++i) A(i, i, 0) += 1.0;

        compare(solve_cta(ctx, A), solve_jacobi(ctx, A), 4);
        std::cout << "Here the two track each other: there is no graded structure for\n"
                     "Jacobi's relative criterion to exploit.\n";
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("10. Relative accuracy: why syev_jacobi_cta exists")
