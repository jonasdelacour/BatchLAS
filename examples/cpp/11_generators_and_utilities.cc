// 11. Generators and utilities
//
// The pieces around the solvers: structured constructors, random generators
// with a requested condition number, norms, condition numbers, transpose,
// lascl, type and layout conversion, and dense-to-sparse conversion.

#include <algorithm>
#include <cmath>
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

const char* norm_name(NormType t) {
    switch (t) {
        case NormType::Frobenius: return "Frobenius";
        case NormType::One: return "One";
        case NormType::Inf: return "Inf";
        case NormType::Max: return "Max";
        case NormType::Spectral: return "Spectral";
        default: return "?";
    }
}

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        constructors_section(ctx);
        batched_constructors_section(ctx);
        random_section(ctx);
        conditioned_section(ctx);
        norm_section(ctx);
        cond_section(ctx);
        transpose_lascl_section(ctx);
        conversion_section(ctx);
    }

    // -----------------------------------------------------------------------
    // Structured constructors
    //
    // Static factories on Matrix, all of them column-major and all of them
    // taking a batch size. They allocate and fill on the device.
    // -----------------------------------------------------------------------
    static void constructors_section(Queue& ctx) {
        section("Structured constructors");

        const int n = 5;

        auto I = Matrix<double>::Identity(n);
        ctx.wait();
        report_error("Identity", max_abs_diff(to_host(I, 0), identity<double>(n)), 0.0);

        auto Z = Matrix<double>::Zeros(n, n);
        auto O = Matrix<double>::Ones(n, n);
        ctx.wait();
        report_error("Zeros", max_abs(to_host(Z, 0)), 0.0);
        {
            HostMatrix<double> ones(n, n);
            std::fill(ones.data.begin(), ones.data.end(), 1.0);
            report_error("Ones", max_abs_diff(to_host(O, 0), ones), 0.0);
        }

        UnifiedVector<double> diag(n);
        for (int i = 0; i < n; ++i) diag[i] = 1.0 + i;
        auto D = Matrix<double>::Diagonal(diag.to_span());
        ctx.wait();
        {
            HostMatrix<double> want(n, n);
            for (int i = 0; i < n; ++i) want(i, i) = 1.0 + i;
            report_error("Diagonal", max_abs_diff(to_host(D, 0), want), 0.0);
        }

        // Triangular(n, uplo, diagonal_value, ...) fills one triangle.
        auto L = Matrix<double>::Triangular(n, Uplo::Lower, /*diagonal_value=*/2.0);
        ctx.wait();
        {
            auto got = to_host(L, 0);
            double above = 0.0;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < j; ++i) above = std::max(above, std::abs(got(i, j)));
            report_error("Triangular(Lower): upper triangle is zero", above, 0.0);
            report_check("Triangular(Lower): diagonal value", got(0, 0) == 2.0);
        }

        // TriDiagToeplitz(n, diag, sub, super) — example 08 uses its closed-form
        // spectrum.
        auto T = Matrix<double>::TriDiagToeplitz(n, 2.0, -1.0, -1.0);
        ctx.wait();
        {
            auto got = to_host(T, 0);
            double outside = 0.0;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    if (std::abs(i - j) > 1) outside = std::max(outside, std::abs(got(i, j)));
            report_error("TriDiagToeplitz: nothing off the three diagonals", outside, 0.0);
            report_check("TriDiagToeplitz: values", got(0, 0) == 2.0 && got(1, 0) == -1.0);
        }

        // RandomTriangular respects Diag::Unit.
        auto RT = Matrix<double>::RandomTriangular(n, Uplo::Upper, Diag::Unit);
        ctx.wait();
        {
            auto got = to_host(RT, 0);
            bool unit = true;
            for (int i = 0; i < n; ++i) unit = unit && got(i, i) == 1.0;
            report_check("RandomTriangular(Diag::Unit) has a unit diagonal", unit);
        }
    }

    // -----------------------------------------------------------------------
    // Batched constructors
    //
    // Every factory takes a batch size, and every item gets the same content —
    // except Random, which varies per item.
    // -----------------------------------------------------------------------
    static void batched_constructors_section(Queue& ctx) {
        section("Batched constructors");

        const int n = 4;
        auto I = Matrix<double>::Identity(n, kBatch);
        ctx.wait();
        report_check("batch size", I.batch_size() == kBatch);

        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b) worst = std::max(worst, max_abs_diff(to_host(I, b), identity<double>(n)));
        report_error("every item is the identity", worst, 0.0);
    }

    // -----------------------------------------------------------------------
    // Random — reproducible pseudo-random matrices
    //
    // Random(rows, cols, hermitian, batch_size, seed). The same seed gives the
    // same matrix, and `hermitian = true` makes each item symmetric.
    // -----------------------------------------------------------------------
    static void random_section(Queue& ctx) {
        section("Random - reproducible pseudo-random matrices");

        const int n = 6;
        auto A = Matrix<double>::Random(n, n, /*hermitian=*/false, kBatch, /*seed=*/1234);
        auto A2 = Matrix<double>::Random(n, n, false, kBatch, 1234);
        auto A3 = Matrix<double>::Random(n, n, false, kBatch, 4321);
        ctx.wait();

        report_error("the same seed reproduces the matrix", max_abs_diff_batched(A.view(), A2.view()), 0.0);
        report_check("a different seed does not", max_abs_diff_batched(A.view(), A3.view()) > 0.0);

        auto H = Matrix<double>::Random(n, n, /*hermitian=*/true, 1, 7);
        ctx.wait();
        auto got = to_host(H, 0);
        report_error("hermitian = true gives a symmetric matrix", max_abs_diff(got, transposed(got)), 1e-14);

        // Batch items differ from one another.
        report_check("batch items are not identical",
                     max_abs_diff(to_host(A, 0), to_host(A, 1)) > 0.0);
    }

    // -----------------------------------------------------------------------
    // Generators with a requested condition number
    //
    // These take log10(kappa) and a metric — Spectral or Frobenius — and build
    // a matrix with (approximately) that conditioning. Useful for testing how
    // an algorithm degrades, as example 04 does by hand.
    //
    // They run on the device, so they take the queue as their first argument,
    // unlike the Matrix factories above.
    // -----------------------------------------------------------------------
    static void conditioned_section(Queue& ctx) {
        section("Generators with a requested condition number");

        const int n = 16;
        const double log10_kappa = 6.0;

        auto A = random_with_log10_cond_metric<B, double>(ctx, n, log10_kappa, NormType::Spectral, kBatch, 11);
        ctx.wait();
        {
            auto svals = singular_values_host(to_host(A, 0));
            report_magnitude("random_with_log10_cond_metric: log10(kappa)",
                             std::log10(svals.back() / svals.front()));
            report_error("close to the requested conditioning",
                         std::abs(std::log10(svals.back() / svals.front()) - log10_kappa), 0.5);
        }

        auto H = random_hermitian_with_log10_cond_metric<B, double>(ctx, n, log10_kappa, NormType::Spectral, kBatch,
                                                                     21);
        ctx.wait();
        {
            auto got = to_host(H, 0);
            report_error("random_hermitian_...: symmetric", max_abs_diff(got, transposed(got)), 1e-12);
        }

        // Banded and tridiagonal variants, same idea.
        auto Band = random_hermitian_banded_with_log10_cond_metric<B, double>(ctx, n, /*kd=*/3, log10_kappa,
                                                                              NormType::Spectral, 1, 31);
        ctx.wait();
        {
            auto got = to_host(Band, 0);
            double outside = 0.0;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    if (std::abs(i - j) > 3) outside = std::max(outside, std::abs(got(i, j)));
            report_error("banded generator respects kd = 3", outside, 1e-14);
        }

        auto Tri = random_hermitian_tridiagonal_with_log10_cond_metric<B, double>(ctx, n, log10_kappa,
                                                                                   NormType::Spectral, 1, 41);
        ctx.wait();
        {
            auto got = to_host(Tri, 0);
            double outside = 0.0;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    if (std::abs(i - j) > 1) outside = std::max(outside, std::abs(got(i, j)));
            report_error("tridiagonal generator is tridiagonal", outside, 1e-14);
        }
    }

    // -----------------------------------------------------------------------
    // norm — matrix norms
    //
    // One value per batch item. The allocating overload returns a
    // UnifiedVector; there is also one that fills a span you own.
    // -----------------------------------------------------------------------
    static void norm_section(Queue& ctx) {
        section("norm - matrix norms");

        const int n = 6;
        auto A = batch_of<double>(n, n, kBatch, random_host<double>, 51);
        ctx.wait();

        for (auto type : {NormType::Frobenius, NormType::One, NormType::Inf, NormType::Max}) {
            auto got = norm<double, MatrixFormat::Dense>(ctx, A.view(), type);
            ctx.wait();

            // Same norm, computed on the host.
            auto Ah = to_host(A, 0);
            double want = 0.0;
            if (type == NormType::Frobenius) {
                for (const auto& v : Ah.data) want += v * v;
                want = std::sqrt(want);
            } else if (type == NormType::Max) {
                want = max_abs(Ah);
            } else if (type == NormType::One) {  // max absolute column sum
                for (int j = 0; j < n; ++j) {
                    double s = 0.0;
                    for (int i = 0; i < n; ++i) s += std::abs(Ah(i, j));
                    want = std::max(want, s);
                }
            } else {  // Inf: max absolute row sum
                for (int i = 0; i < n; ++i) {
                    double s = 0.0;
                    for (int j = 0; j < n; ++j) s += std::abs(Ah(i, j));
                    want = std::max(want, s);
                }
            }
            report_error(std::string(norm_name(type)) + " norm", std::abs(got[0] - want), 1e-12);
        }

        // One value per batch item.
        auto all = norm<double, MatrixFormat::Dense>(ctx, A.view(), NormType::Frobenius);
        ctx.wait();
        report_check("one norm per batch item", all.size() == static_cast<size_t>(kBatch));
    }

    // -----------------------------------------------------------------------
    // cond — condition numbers
    //
    // Takes a backend template parameter because it factorizes internally.
    // The allocating overload is the easy one; the other takes a workspace
    // sized by cond_buffer_size.
    // -----------------------------------------------------------------------
    static void cond_section(Queue& ctx) {
        section("cond - condition numbers");

        const int n = 8;

        // A matrix whose condition number we know exactly.
        std::vector<double> svals(n);
        for (int i = 0; i < n; ++i) svals[i] = std::pow(10.0, -3.0 * i / (n - 1));
        auto A = broadcast(with_singular_values<double>(n, svals, 61), kBatch);
        ctx.wait();

        auto kappa = cond<B, double, MatrixFormat::Dense>(ctx, A.view(), NormType::Frobenius);
        ctx.wait();
        report_magnitude("Frobenius condition number", kappa[0]);
        report_check("one value per batch item", kappa.size() == static_cast<size_t>(kBatch));

        // `cond` also has a workspace-taking overload, but there is currently
        // no way to size that workspace from the public headers: the
        // `cond_buffer_size` that blas/extra.hh declares is a non-template
        // taking a MatrixView<float> and has no definition (a link error),
        // while the real one is a template that the header never declares.
        // Use the allocating overload above. See the known issues in the
        // README.
        report_skip("cond with an explicit workspace", "cond_buffer_size is not usable from the public headers");
    }

    // -----------------------------------------------------------------------
    // transpose and lascl
    //
    // `transpose` has an allocating overload and one writing into a matrix you
    // provide. `lascl` rescales in place by cto/cfrom, done carefully so it
    // does not overflow — the way LAPACK's xLASCL does.
    // -----------------------------------------------------------------------
    static void transpose_lascl_section(Queue& ctx) {
        section("transpose and lascl");

        const int m = 5, n = 3;
        auto A = batch_of<double>(m, n, kBatch, random_host<double>, 71);
        ctx.wait();

        auto At = transpose<double, MatrixFormat::Dense>(ctx, A.view());
        ctx.wait();
        report_check("transpose swaps the shape", At.rows() == n && At.cols() == m);

        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b) worst = std::max(worst, max_abs_diff(to_host(At, b), transposed(to_host(A, b))));
        report_error("transpose", worst, 0.0);

        // Into a matrix you already own.
        auto Out = Matrix<double>::Zeros(n, m, kBatch);
        transpose<double, MatrixFormat::Dense>(ctx, A.view(), Out.view());
        ctx.wait();
        report_error("transpose into a preallocated matrix", max_abs_diff_batched(Out.view(), At.view()), 0.0);

        // lascl: multiply by cto/cfrom.
        auto S = A.clone();
        lascl<MatrixFormat::Dense, double>(ctx, S.view(), /*cfrom=*/1.0, /*cto=*/8.0);
        ctx.wait();
        worst = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            auto before = to_host(A, b);
            auto after = to_host(S, b);
            for (size_t i = 0; i < before.data.size(); ++i)
                worst = std::max(worst, std::abs(after.data[i] - 8.0 * before.data[i]));
        }
        report_error("lascl(cfrom=1, cto=8)", worst, 1e-13);
    }

    // -----------------------------------------------------------------------
    // Type and format conversion
    //
    // `astype` changes the scalar type, `to_row_major`/`to_column_major`
    // change the layout, and `convert_to` changes the storage format —
    // dense to CSR, dropping entries below a threshold.
    // -----------------------------------------------------------------------
    static void conversion_section(Queue& ctx) {
        section("Type and format conversion");

        const int n = 6;
        auto A = batch_of<double>(n, n, kBatch, random_host<double>, 81);
        ctx.wait();

        auto Af = A.astype<float>();
        ctx.wait();
        double worst = 0.0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                worst = std::max(worst, std::abs(static_cast<double>(Af(i, j, 0)) - A(i, j, 0)));
        report_error("astype<float> round-trips within float precision", worst, 1e-6);

        auto Ac = A.astype<std::complex<double>>();
        ctx.wait();
        report_check("astype to complex keeps the real part", Ac(0, 0, 0).real() == A(0, 0, 0));

        // Row-major and back.
        auto R = A.to_row_major();
        auto C = R.to_column_major();
        ctx.wait();
        report_error("to_row_major then to_column_major is the identity", max_abs_diff_batched(C.view(), A.view()),
                     0.0);

        // Dense to CSR. Entries below the threshold are dropped, so a matrix
        // with a few large entries becomes genuinely sparse.
        auto Sparse = Matrix<double>::Zeros(n, n, 1);
        for (int i = 0; i < n; ++i) Sparse(i, i, 0) = 1.0 + i;
        Sparse(0, n - 1, 0) = 0.5;
        ctx.wait();

        auto Csr = Sparse.convert_to<MatrixFormat::CSR>(1e-12);
        ctx.wait();
        report_check("convert_to<CSR> keeps only the non-zeros", Csr.nnz() == n + 1);
    }

    // Singular values of a small host matrix, ascending, via the eigenvalues
    // of A^T A. Good enough to report a condition number.
    static std::vector<double> singular_values_host(const HostMatrix<double>& A) {
        auto vals = jacobi_eigenvalues(matmul(A, A, Transpose::Trans, Transpose::NoTrans));
        for (auto& v : vals) v = std::sqrt(std::max(0.0, v));
        std::sort(vals.begin(), vals.end());
        return vals;
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("11. Generators and utilities")
