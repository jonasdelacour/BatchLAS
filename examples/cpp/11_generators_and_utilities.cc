// 11. Generators and utilities
//
// The pieces around the solvers: structured constructors, random generators
// with a requested condition number, norms, condition numbers, transpose,
// lascl, and type/layout/format conversion.

#include <complex>
#include <cstddef>
#include <iostream>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_utils.hh"

using namespace batchlas;
using namespace examples;

namespace {

constexpr int kBatch = 3;

const char* norm_name(NormType t) {
    switch (t) {
        case NormType::Frobenius: return "Frobenius";
        case NormType::One: return "One (max column sum)";
        case NormType::Inf: return "Inf (max row sum)";
        case NormType::Max: return "Max (largest entry)";
        case NormType::Spectral: return "Spectral";
        default: return "?";
    }
}

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        constructors_section(ctx);
        random_section(ctx);
        conditioned_section(ctx);
        norm_section(ctx);
        cond_section(ctx);
        transpose_lascl_section(ctx);
        conversion_section(ctx);
    }

    // Structured constructors
    //
    // Static factories on Matrix, all column-major and all taking a batch size.
    // They allocate and fill on the device.
    static void constructors_section(Queue& ctx) {
        section("Structured constructors");

        const int n = 4;

        auto I = Matrix<double>::Identity(n, kBatch);
        ctx.wait();
        std::cout << "Identity(4, batch=3), item 0:\n";
        I.view()[0].print();

        UnifiedVector<double> diag(n);
        for (int i = 0; i < n; ++i) diag[i] = 1.0 + i;
        auto D = Matrix<double>::Diagonal(diag.to_span());
        ctx.wait();
        std::cout << "Diagonal([1,2,3,4]):\n";
        D.print();

        // Triangular(n, uplo, diagonal_value)
        auto L = Matrix<double>::Triangular(n, Uplo::Lower, /*diagonal_value=*/2.0);
        ctx.wait();
        std::cout << "Triangular(Lower, diagonal 2):\n";
        L.print();

        // TriDiagToeplitz(n, diag, sub, super) — example 08 uses its
        // closed-form spectrum.
        auto T = Matrix<double>::TriDiagToeplitz(n, 2.0, -1.0, -1.0);
        ctx.wait();
        std::cout << "TriDiagToeplitz(2, -1, -1):\n";
        T.print();

        // Zeros / Ones round out the set, and RandomTriangular respects
        // Diag::Unit.
        auto RT = Matrix<double>::RandomTriangular(n, Uplo::Upper, Diag::Unit);
        ctx.wait();
        std::cout << "RandomTriangular(Upper, Diag::Unit) — note the unit diagonal:\n";
        RT.print();
    }

    // Random — reproducible pseudo-random matrices
    //
    // Random(rows, cols, hermitian, batch_size, seed). The same seed gives the
    // same matrix, and `hermitian = true` makes each item symmetric.
    static void random_section(Queue& ctx) {
        section("Random - reproducible pseudo-random matrices");

        const int n = 4;
        auto A = Matrix<double>::Random(n, n, false, kBatch, /*seed=*/1234);
        auto A2 = Matrix<double>::Random(n, n, false, kBatch, 1234);
        auto A3 = Matrix<double>::Random(n, n, false, kBatch, 4321);
        ctx.wait();

        print("same seed reproduces (0,0)", A(0, 0, 0) == A2(0, 0, 0));
        print("a different seed does not", A(0, 0, 0) != A3(0, 0, 0));
        print("batch items differ from one another", A(0, 0, 0) != A(0, 0, 1));

        auto H = Matrix<double>::Random(n, n, /*hermitian=*/true, 1, 7);
        ctx.wait();
        std::cout << "Random(hermitian = true):\n";
        H.print();
    }

    // Generators with a requested condition number
    //
    // These take log10(kappa) and a metric — Spectral or Frobenius — and build
    // a matrix with approximately that conditioning, which is useful for
    // testing how an algorithm degrades. They run on the device, so unlike the
    // Matrix factories above they take the queue as their first argument.
    static void conditioned_section(Queue& ctx) {
        section("Generators with a requested condition number");

        const int n = 16;
        const double log10_kappa = 6.0;

        auto A = random_with_log10_cond_metric<B, double>(ctx, n, log10_kappa, NormType::Spectral, kBatch, 11);
        ctx.wait();
        auto kappa = cond<B, double, MatrixFormat::Dense>(ctx, A.view(), NormType::Frobenius);
        ctx.wait();
        print("requested log10(kappa)", log10_kappa);
        print("Frobenius condition number of the result", kappa[0]);

        auto H = random_hermitian_with_log10_cond_metric<B, double>(ctx, n, log10_kappa, NormType::Spectral, kBatch,
                                                                    21);
        ctx.wait();
        print("hermitian variant is symmetric", H(0, 1, 0) == H(1, 0, 0));

        // Banded and tridiagonal variants, same idea.
        auto Band = random_hermitian_banded_with_log10_cond_metric<B, double>(ctx, 8, /*kd=*/2, log10_kappa,
                                                                              NormType::Spectral, 1, 31);
        ctx.wait();
        std::cout << "random_hermitian_banded_..., kd = 2:\n";
        Band.print();

        auto Tri = random_hermitian_tridiagonal_with_log10_cond_metric<B, double>(ctx, 6, log10_kappa,
                                                                                  NormType::Spectral, 1, 41);
        ctx.wait();
        std::cout << "random_hermitian_tridiagonal_...:\n";
        Tri.print();
    }

    // norm — matrix norms
    //
    // One value per batch item. The allocating overload returns a
    // UnifiedVector; there is also one that fills a span you own.
    static void norm_section(Queue& ctx) {
        section("norm - matrix norms");

        const int n = 5;
        auto A = Matrix<double>::Random(n, n, false, kBatch, 51);
        ctx.wait();

        for (auto type : {NormType::Frobenius, NormType::One, NormType::Inf, NormType::Max}) {
            auto values = norm<double, MatrixFormat::Dense>(ctx, A.view(), type);
            ctx.wait();
            print(norm_name(type), values[0]);
        }

        auto all = norm<double, MatrixFormat::Dense>(ctx, A.view(), NormType::Frobenius);
        ctx.wait();
        print("one value per batch item", all.size() == static_cast<size_t>(kBatch));

        // The span-filling form, when you already own the output buffer.
        UnifiedVector<double> out(kBatch);
        norm<double, MatrixFormat::Dense>(ctx, A.view(), NormType::Frobenius, out.to_span());
        ctx.wait();
        print_values("filled into a span you own", out.to_span(), kBatch);
    }

    // cond — condition numbers
    //
    // Takes a backend template parameter because it factorizes internally.
    // Use the allocating overload: `cond` also has a workspace-taking form, but
    // there is currently no way to size that workspace from the public headers.
    // See the known issues in the README.
    static void cond_section(Queue& ctx) {
        section("cond - condition numbers");

        const int n = 8;
        auto well = Matrix<double>::Identity(n, kBatch);
        auto ill = random_with_log10_cond_metric<B, double>(ctx, n, 8.0, NormType::Spectral, kBatch, 61);
        ctx.wait();

        auto k1 = cond<B, double, MatrixFormat::Dense>(ctx, well.view(), NormType::Frobenius);
        auto k2 = cond<B, double, MatrixFormat::Dense>(ctx, ill.view(), NormType::Frobenius);
        ctx.wait();
        print("cond(I)", k1[0]);
        print("cond(a matrix built for log10(kappa) = 8)", k2[0]);
    }

    // transpose and lascl
    //
    // `transpose` has an allocating overload and one writing into a matrix you
    // provide. `lascl` rescales in place by cto/cfrom, done carefully so it
    // does not overflow — the way LAPACK's xLASCL does.
    static void transpose_lascl_section(Queue& ctx) {
        section("transpose and lascl");

        auto A = Matrix<double>::Random(4, 2, false, kBatch, 71);
        ctx.wait();
        std::cout << "A (4x2), item 0:\n";
        A.view()[0].print();

        auto At = transpose<double, MatrixFormat::Dense>(ctx, A.view());
        ctx.wait();
        std::cout << "transpose(A) (2x4), allocated for you:\n";
        At.view()[0].print();

        auto Out = Matrix<double>::Zeros(2, 4, kBatch);
        transpose<double, MatrixFormat::Dense>(ctx, A.view(), Out.view());
        ctx.wait();
        print("the two-argument form writes into a matrix you own", Out(0, 1, 0) == At(0, 1, 0));

        auto S = A.clone();
        lascl<MatrixFormat::Dense, double>(ctx, S.view(), /*cfrom=*/1.0, /*cto=*/8.0);
        ctx.wait();
        std::cout << "lascl(cfrom = 1, cto = 8) scales in place:\n";
        S.view()[0].print();
    }

    // Type, layout and format conversion
    //
    // `astype` changes the scalar type, `to_row_major`/`to_column_major` change
    // the layout, and `convert_to` changes the storage format — dense to CSR,
    // dropping entries below a threshold.
    static void conversion_section(Queue& ctx) {
        section("Type, layout and format conversion");

        const int n = 4;
        auto A = Matrix<double>::Random(n, n, false, kBatch, 81);
        ctx.wait();

        auto Af = A.astype<float>();
        auto Ac = A.astype<std::complex<double>>();
        ctx.wait();
        print("astype<float> keeps the value", static_cast<double>(Af(0, 0, 0)));
        print("astype<complex> puts it in the real part", Ac(0, 0, 0).real() == A(0, 0, 0));

        auto R = A.to_row_major();
        auto C = R.to_column_major();
        ctx.wait();
        print("to_row_major then back is the identity", C(1, 2, 0) == A(1, 2, 0));

        // Dense to CSR: entries below the threshold are dropped, so a mostly
        // empty dense matrix becomes genuinely sparse.
        auto Sparse = Matrix<double>::Zeros(n, n, 1);
        for (int i = 0; i < n; ++i) Sparse(i, i, 0) = 1.0 + i;
        Sparse(0, n - 1, 0) = 0.5;
        ctx.wait();

        auto Csr = Sparse.convert_to<MatrixFormat::CSR>(/*zero_threshold=*/1e-12);
        ctx.wait();
        print("dense entries", n * n);
        print("stored after convert_to<CSR>", Csr.nnz());
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("11. Generators and utilities")
