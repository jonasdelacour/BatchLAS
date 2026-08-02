// 4. QR factorization and orthogonalization
//
// geqrf (QR in packed reflector form), orgqr (materialise Q), ormqr (apply Q
// without forming it), and the `ortho` family for orthonormalising a block of
// vectors — including against an existing basis.

#include <cstddef>
#include <iostream>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_utils.hh"

using namespace batchlas;
using namespace examples;

namespace {

constexpr int kBatch = 3;

const char* algo_name(OrthoAlgorithm a) {
    switch (a) {
        case OrthoAlgorithm::Chol2: return "Chol2";
        case OrthoAlgorithm::Cholesky: return "Cholesky";
        case OrthoAlgorithm::ShiftChol3: return "ShiftChol3";
        case OrthoAlgorithm::Householder: return "Householder";
        case OrthoAlgorithm::CGS2: return "CGS2";
        case OrthoAlgorithm::SVQB: return "SVQB";
        case OrthoAlgorithm::SVQB2: return "SVQB2";
        default: return "?";
    }
}

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        geqrf_section(ctx);
        orgqr_section(ctx);
        ormqr_section(ctx);
        ortho_section(ctx);
        ortho_metric_section(ctx);
    }

    // Q^T Q for the first batch item, so you can see how close to I it is.
    static void show_gram(Queue& ctx, const Matrix<double>& Q, const char* label) {
        auto G = Matrix<double>::Zeros(Q.cols(), Q.cols(), Q.batch_size());
        gemm<B>(ctx, Q, Q, G, 1.0, 0.0, Transpose::Trans, Transpose::NoTrans);
        ctx.wait();
        std::cout << label << ":\n";
        G.view()[0].print();
    }

    // geqrf — QR factorization in packed form.
    //
    // A is overwritten: the upper triangle becomes R and the columns below the
    // diagonal hold the Householder vectors. Together with `tau` — one scalar
    // per reflector, min(m, n) per batch item — they represent Q implicitly.
    // You allocate tau; it is a `Span<T>`, not part of A.
    static void geqrf_section(Queue& ctx) {
        section("geqrf - QR in packed reflector form");

        const int m = 6, n = 4;
        auto A = Matrix<double>::Random(m, n, false, kBatch, 11);

        UnifiedVector<double> tau(static_cast<size_t>(std::min(m, n)) * kBatch);
        UnifiedVector<std::byte> ws(geqrf_buffer_size<B>(ctx, A, tau.to_span()));
        geqrf<B>(ctx, A, tau.to_span(), ws.to_span());
        ctx.wait();

        print("tau length (min(m,n) per item)", tau.size());
        std::cout << "packed factors: R above the diagonal, reflectors below, item 0:\n";
        A.view()[0].print();
    }

    // orgqr — turn the packed form into an explicit Q.
    //
    // Overwrites the packed A in place with the first n columns of Q. Pass the
    // same tau that geqrf produced.
    static void orgqr_section(Queue& ctx) {
        section("orgqr - materialise Q");

        const int m = 6, n = 4;
        auto A = Matrix<double>::Random(m, n, false, kBatch, 21);

        UnifiedVector<double> tau(static_cast<size_t>(n) * kBatch);
        UnifiedVector<std::byte> ws(geqrf_buffer_size<B>(ctx, A, tau.to_span()));
        geqrf<B>(ctx, A, tau.to_span(), ws.to_span());

        UnifiedVector<std::byte> ws2(orgqr_buffer_size<B>(ctx, A, tau.to_span()));
        orgqr<B>(ctx, A, tau.to_span(), ws2.to_span());
        ctx.wait();

        std::cout << "Q, item 0:\n";
        A.view()[0].print();
        show_gram(ctx, A, "Q^T Q");
    }

    // ormqr — apply Q to another matrix without forming it.
    //
    // C <- op(Q)*C (Side::Left) or C*op(Q) (Side::Right). Cheaper than
    // orgqr + gemm, and the usual way to use a QR factorization in a
    // least-squares solve: Q^T b is the projection step.
    static void ormqr_section(Queue& ctx) {
        section("ormqr - apply Q without forming it");

        const int m = 6, n = 4, nrhs = 2;
        auto A = Matrix<double>::Random(m, n, false, kBatch, 31);

        UnifiedVector<double> tau(static_cast<size_t>(n) * kBatch);
        UnifiedVector<std::byte> ws(geqrf_buffer_size<B>(ctx, A, tau.to_span()));
        geqrf<B>(ctx, A, tau.to_span(), ws.to_span());
        ctx.wait();

        auto C = Matrix<double>::Random(m, nrhs, false, kBatch, 32);
        std::cout << "C before, item 0:\n";
        C.view()[0].print();

        UnifiedVector<std::byte> ws2(
            ormqr_buffer_size<B>(ctx, A, C, Side::Left, Transpose::Trans, tau.to_span()));
        ormqr<B>(ctx, A, C, Side::Left, Transpose::Trans, tau.to_span(), ws2.to_span());
        ctx.wait();

        std::cout << "Q^T C, in place:\n";
        C.view()[0].print();

        // Applying Q afterwards undoes it.
        ormqr<B>(ctx, A, C, Side::Left, Transpose::NoTrans, tau.to_span(), ws2.to_span());
        ctx.wait();
        std::cout << "Q (Q^T C) is C again:\n";
        C.view()[0].print();
    }

    // ortho — orthonormalise a block of vectors in place.
    //
    // Unlike geqrf/orgqr this gives you no R: it just replaces A with a matrix
    // whose columns (Transpose::NoTrans) or rows (Transpose::Trans) span the
    // same space and are orthonormal. Several algorithms trade accuracy for
    // speed; Chol2 is the default.
    static void ortho_section(Queue& ctx) {
        section("ortho - orthonormalise a block of vectors");

        const int m = 10, k = 4;
        for (auto algo : {OrthoAlgorithm::Chol2, OrthoAlgorithm::ShiftChol3, OrthoAlgorithm::CGS2,
                          OrthoAlgorithm::SVQB2}) {
            auto A = Matrix<double>::Random(m, k, false, kBatch, 41);
            UnifiedVector<std::byte> ws(ortho_buffer_size<B>(ctx, A, Transpose::NoTrans, algo));
            ortho<B>(ctx, A, Transpose::NoTrans, ws.to_span(), algo);
            ctx.wait();

            // The diagonal of Q^T Q is 1 and the rest ~0 when it worked.
            auto G = Matrix<double>::Zeros(k, k, kBatch);
            gemm<B>(ctx, A, A, G, 1.0, 0.0, Transpose::Trans, Transpose::NoTrans);
            ctx.wait();
            print(std::string("Q^T Q with ") + algo_name(algo) + ": (0,0) and (0,1)",
                  std::to_string(G(0, 0, 0)) + ", " + std::to_string(G(0, 1, 0)));
        }

        // The cheapest algorithms form A^T A, which squares the condition
        // number — on a badly conditioned block that is where single-pass
        // Cholesky gives up and Chol2 or CGS2 keep working. Example 10 makes
        // the same point about eigenvalues.
        //
        // Note also the known defect in `ortho(algorithm = Householder)` on
        // CUDA; see the README.

        // Transpose::Trans orthonormalises the rows instead of the columns.
        auto R = Matrix<double>::Random(k, k, false, 1, 42);
        UnifiedVector<std::byte> ws(ortho_buffer_size<B>(ctx, R, Transpose::Trans));
        ortho<B>(ctx, R, Transpose::Trans, ws.to_span());
        ctx.wait();
        std::cout << "row-orthonormalised (square input; see the README for wide input on NETLIB):\n";
        R.print();
    }

    // ortho against an existing basis
    //
    // The overload taking a second matrix M orthogonalises A against M as well
    // as internally, so the result spans the part of A's range orthogonal to M.
    // This is the block Gram-Schmidt step inside an iterative eigensolver,
    // which is what syevx and lanczos use it for. `iterations` repeats the
    // projection for stability (2 by default).
    static void ortho_metric_section(Queue& ctx) {
        section("ortho against an existing basis");

        const int m = 10, kM = 3, kA = 2;

        auto M = Matrix<double>::Random(m, kM, false, kBatch, 51);
        UnifiedVector<std::byte> wsm(ortho_buffer_size<B>(ctx, M, Transpose::NoTrans));
        ortho<B>(ctx, M, Transpose::NoTrans, wsm.to_span());

        auto A = Matrix<double>::Random(m, kA, false, kBatch, 52);
        UnifiedVector<std::byte> ws(ortho_buffer_size<B>(ctx, A, M, Transpose::NoTrans, Transpose::NoTrans));
        ortho<B>(ctx, A, M, Transpose::NoTrans, Transpose::NoTrans, ws.to_span());
        ctx.wait();

        // M^T A is zero when A really is orthogonal to the basis.
        auto P = Matrix<double>::Zeros(kM, kA, kBatch);
        gemm<B>(ctx, M, A, P, 1.0, 0.0, Transpose::Trans, Transpose::NoTrans);
        ctx.wait();
        std::cout << "M^T A after the projection (all ~0), item 0:\n";
        P.view()[0].print();
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("4. QR factorization and orthogonalization")
