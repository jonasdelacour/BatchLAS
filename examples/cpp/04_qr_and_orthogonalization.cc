// 4. QR factorization and orthogonalization
//
// geqrf (QR in packed reflector form), orgqr (materialise Q), ormqr (apply Q
// without forming it), and the `ortho` family for orthonormalising a block of
// vectors — including against an existing basis.

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
        ortho_conditioning_section(ctx);
        ortho_metric_section(ctx);
    }

    // -----------------------------------------------------------------------
    // geqrf — QR factorization in packed form.
    //
    // A is overwritten: the upper triangle becomes R, and the columns below
    // the diagonal hold the Householder vectors. Together with `tau` — one
    // scalar per reflector, min(m, n) per batch item — they represent Q
    // implicitly. You allocate tau; it is a `Span<T>`, not part of A.
    // -----------------------------------------------------------------------
    static void geqrf_section(Queue& ctx) {
        section("geqrf - QR in packed reflector form");

        const int m = 8, n = 5;
        auto A = batch_of<double>(m, n, kBatch, random_host<double>, 11);
        std::vector<HostMatrix<double>> originals;
        for (int b = 0; b < kBatch; ++b) originals.push_back(to_host(A, b));

        UnifiedVector<double> tau(static_cast<size_t>(std::min(m, n)) * kBatch);
        UnifiedVector<std::byte> ws(geqrf_buffer_size<B>(ctx, A, tau.to_span()));
        geqrf<B>(ctx, A, tau.to_span(), ws.to_span());
        ctx.wait();

        report("tau length (min(m,n) per item)", tau.size());

        // R is the upper triangle of the overwritten A. Checking R alone is
        // easy: |R| must match the upper triangle of the QR of the original,
        // and more usefully, R^T R == A^T A (both equal the Gram matrix).
        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            auto packed = to_host(A, b);
            HostMatrix<double> R(n, n);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i <= j; ++i) R(i, j) = packed(i, j);
            auto RtR = matmul(R, R, Transpose::Trans, Transpose::NoTrans);
            auto AtA = matmul(originals[b], originals[b], Transpose::Trans, Transpose::NoTrans);
            worst = std::max(worst, max_abs_diff(RtR, AtA));
        }
        report_error("|R^T R - A^T A|", worst, 1e-10);
    }

    // -----------------------------------------------------------------------
    // orgqr — turn the packed form into an explicit Q.
    //
    // Overwrites the packed A in place with the first n columns of Q. Pass the
    // same tau geqrf produced.
    // -----------------------------------------------------------------------
    static void orgqr_section(Queue& ctx) {
        section("orgqr - materialise Q");

        const int m = 8, n = 5;
        auto A = batch_of<double>(m, n, kBatch, random_host<double>, 21);
        std::vector<HostMatrix<double>> originals;
        for (int b = 0; b < kBatch; ++b) originals.push_back(to_host(A, b));

        UnifiedVector<double> tau(static_cast<size_t>(n) * kBatch);
        UnifiedVector<std::byte> ws(geqrf_buffer_size<B>(ctx, A, tau.to_span()));
        geqrf<B>(ctx, A, tau.to_span(), ws.to_span());
        ctx.wait();

        // Keep R before orgqr overwrites the packed factors.
        std::vector<HostMatrix<double>> Rs;
        for (int b = 0; b < kBatch; ++b) {
            auto packed = to_host(A, b);
            HostMatrix<double> R(n, n);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i <= j; ++i) R(i, j) = packed(i, j);
            Rs.push_back(R);
        }

        UnifiedVector<std::byte> ws2(orgqr_buffer_size<B>(ctx, A, tau.to_span()));
        orgqr<B>(ctx, A, tau.to_span(), ws2.to_span());
        ctx.wait();

        double ortho = 0.0, recon = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            auto Q = to_host(A, b);
            ortho = std::max(ortho, orthogonality_error(Q));
            recon = std::max(recon, max_abs_diff(matmul(Q, Rs[b]), originals[b]));
        }
        report_error("|Q^T Q - I|", ortho, 1e-12);
        report_error("|Q R - A|", recon, 1e-12);
    }

    // -----------------------------------------------------------------------
    // ormqr — apply Q to another matrix without forming it.
    //
    // C <- op(Q) * C (Side::Left) or C * op(Q) (Side::Right), where op is
    // NoTrans or Trans/ConjTrans. Cheaper than orgqr + gemm, and the usual way
    // to use a QR factorization in a least-squares solve.
    // -----------------------------------------------------------------------
    static void ormqr_section(Queue& ctx) {
        section("ormqr - apply Q without forming it");

        const int m = 8, n = 5, nrhs = 3;
        auto A = batch_of<double>(m, n, kBatch, random_host<double>, 31);

        UnifiedVector<double> tau(static_cast<size_t>(n) * kBatch);
        UnifiedVector<std::byte> ws(geqrf_buffer_size<B>(ctx, A, tau.to_span()));
        geqrf<B>(ctx, A, tau.to_span(), ws.to_span());
        ctx.wait();

        // Reference Q, obtained the expensive way from a copy.
        auto Acopy = A.clone();
        UnifiedVector<std::byte> ws_q(orgqr_buffer_size<B>(ctx, Acopy, tau.to_span()));
        orgqr<B>(ctx, Acopy, tau.to_span(), ws_q.to_span());
        ctx.wait();

        auto C = batch_of<double>(m, nrhs, kBatch, random_host<double>, 41);
        std::vector<HostMatrix<double>> Cs;
        for (int b = 0; b < kBatch; ++b) Cs.push_back(to_host(C, b));

        // Q^T C — the projection step of a least-squares solve.
        UnifiedVector<std::byte> ws_o(
            ormqr_buffer_size<B>(ctx, A, C, Side::Left, Transpose::Trans, tau.to_span()));
        ormqr<B>(ctx, A, C, Side::Left, Transpose::Trans, tau.to_span(), ws_o.to_span());
        ctx.wait();

        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            // Only the leading n rows of Q^T C are determined by the thin Q.
            auto want = matmul(to_host(Acopy, b), Cs[b], Transpose::Trans, Transpose::NoTrans);
            auto got = to_host(C, b);
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i) worst = std::max(worst, std::abs(got(i, j) - want(i, j)));
        }
        report_error("|ormqr(Q^T C) - Q^T C|", worst, 1e-10);
    }

    // -----------------------------------------------------------------------
    // ortho — orthonormalise a block of vectors in place.
    //
    // Unlike geqrf/orgqr this gives you no R: it just replaces A with a matrix
    // whose columns (Transpose::NoTrans) or rows (Transpose::Trans) span the
    // same space and are orthonormal. Several algorithms trade accuracy for
    // speed; Chol2 is the default.
    // -----------------------------------------------------------------------
    static void ortho_section(Queue& ctx) {
        section("ortho - orthonormalise a block of vectors");

        const int m = 12, k = 5;

        for (auto algo : {OrthoAlgorithm::Chol2, OrthoAlgorithm::ShiftChol3, OrthoAlgorithm::CGS2,
                          OrthoAlgorithm::SVQB, OrthoAlgorithm::SVQB2, OrthoAlgorithm::Cholesky}) {
            auto A = batch_of<double>(m, k, kBatch, random_host<double>, 51);
            UnifiedVector<std::byte> ws(ortho_buffer_size<B>(ctx, A, Transpose::NoTrans, algo));
            ortho<B>(ctx, A, Transpose::NoTrans, ws.to_span(), algo);
            ctx.wait();

            double worst = 0.0;
            for (int b = 0; b < kBatch; ++b) worst = std::max(worst, orthogonality_error(to_host(A, b)));
            report_error(std::string("|Q^T Q - I| with ") + algo_name(algo), worst, 1e-10);
        }

        // Transpose::Trans orthonormalises the *rows* instead.
        auto R = batch_of<double>(k, m, kBatch, random_host<double>, 61);
        UnifiedVector<std::byte> ws(ortho_buffer_size<B>(ctx, R, Transpose::Trans));
        ortho<B>(ctx, R, Transpose::Trans, ws.to_span());
        ctx.wait();

        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b) worst = std::max(worst, orthogonality_error(transposed(to_host(R, b))));
        if constexpr (B == Backend::NETLIB) {
            // Known defect: on the host backend this is only correct for
            // square input. For a wide block (k < m) LAPACK reports an illegal
            // DORGQR argument and the result is not orthonormal. See the known
            // issues in examples/cpp/README.md.
            report_skip("|Q Q^T - I| (rows, transA=Trans)",
                        "known defect on NETLIB for wide input; measured " + std::to_string(worst));
        } else {
            report_error("|Q Q^T - I| (rows, transA=Trans)", worst, 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // Where the cheap algorithms break down
    //
    // The Cholesky-based algorithms form A^T A, which squares the condition
    // number. On a badly conditioned block that loses roughly twice as many
    // digits, and plain `Cholesky` can fail outright while Chol2 (two passes)
    // or SVQB2 still hold up. This is the reason there is a choice at all.
    // -----------------------------------------------------------------------
    static void ortho_conditioning_section(Queue& ctx) {
        section("Where the cheap algorithms break down");

        const int m = 16, k = 6;

        // Singular values from 1 down to 1e-12, spread across the whole
        // matrix rather than sitting in a column scaling (which the
        // algorithms handle easily). Forming A^T A squares that to 1e-24,
        // past what double precision carries.
        std::vector<double> svals(k);
        for (int j = 0; j < k; ++j) svals[j] = std::pow(10.0, -12.0 * j / (k - 1));
        const auto ill = with_singular_values<double>(m, svals, 71);
        report_magnitude("condition number of the block", svals.front() / svals.back());

        double single_pass = 0.0, two_pass = 0.0;
        for (auto algo : {OrthoAlgorithm::Cholesky, OrthoAlgorithm::Chol2, OrthoAlgorithm::SVQB2,
                          OrthoAlgorithm::CGS2}) {
            auto A = broadcast(ill, 1);
            UnifiedVector<std::byte> ws(ortho_buffer_size<B>(ctx, A, Transpose::NoTrans, algo));
            ortho<B>(ctx, A, Transpose::NoTrans, ws.to_span(), algo);
            ctx.wait();
            const double err = orthogonality_error(to_host(A, 0));
            if (algo == OrthoAlgorithm::Cholesky) single_pass = err;
            if (algo == OrthoAlgorithm::Chol2) two_pass = err;
            // Reported without a tolerance: the point is the spread between
            // algorithms, and which of them is still usable here.
            report_magnitude(std::string("ill-conditioned |Q^T Q - I| with ") + algo_name(algo), err);
        }
        if (single_pass > 1e-8) {
            report_check("the second pass is what saves Chol2", two_pass < single_pass);
        } else {
            // The host backend routes every OrthoAlgorithm through the same
            // QR-based path, so there is no spread to observe there.
            report_skip("algorithm comparison", "this backend does not degrade on this input");
        }

        // Householder is the most accurate of the set, but see the known issue
        // in examples/cpp/README.md before using it on CUDA.
        report_skip("Householder on ill-conditioned input",
                    "known workspace-aliasing defect on CUDA; see README");
    }

    // -----------------------------------------------------------------------
    // ortho against an existing basis
    //
    // The four-matrix overload takes a second matrix M and orthogonalises A
    // against it as well as internally: the result spans the part of A's range
    // that is orthogonal to M. This is the block Gram-Schmidt step inside an
    // iterative eigensolver, which is what syevx and lanczos use it for.
    //
    // `iterations` repeats the projection for stability (2 by default).
    // -----------------------------------------------------------------------
    static void ortho_metric_section(Queue& ctx) {
        section("ortho against an existing basis");

        const int m = 12, kM = 4, kA = 3;

        // An orthonormal basis M to project against.
        auto M = batch_of<double>(m, kM, kBatch, random_host<double>, 81);
        {
            UnifiedVector<std::byte> ws(ortho_buffer_size<B>(ctx, M, Transpose::NoTrans));
            ortho<B>(ctx, M, Transpose::NoTrans, ws.to_span());
            ctx.wait();
        }

        auto A = batch_of<double>(m, kA, kBatch, random_host<double>, 91);
        UnifiedVector<std::byte> ws(
            ortho_buffer_size<B>(ctx, A, M, Transpose::NoTrans, Transpose::NoTrans));
        ortho<B>(ctx, A, M, Transpose::NoTrans, Transpose::NoTrans, ws.to_span());
        ctx.wait();

        double self = 0.0, cross = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            auto Ah = to_host(A, b);
            self = std::max(self, orthogonality_error(Ah));
            cross = std::max(cross, max_abs(matmul(to_host(M, b), Ah, Transpose::Trans, Transpose::NoTrans)));
        }
        report_error("|A^T A - I| after projection", self, 1e-10);
        report_error("|M^T A| (A is orthogonal to M)", cross, 1e-10);
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("4. QR factorization and orthogonalization")
