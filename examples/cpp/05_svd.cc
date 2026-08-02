// 5. Singular value decomposition
//
// The gesvd driver and its native variants, plus the pieces they are built
// from: gebrd (reduction to bidiagonal form), bdsqr (bidiagonal QR iteration)
// and ormbr (applying the reduction's reflectors).
//
// A = U * diag(s) * V^H, with s in descending order.

#include <algorithm>
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

// The singular values these examples ask for, descending.
std::vector<double> target_svals(int k) {
    std::vector<double> s(k);
    for (int i = 0; i < k; ++i) s[i] = 1.0 + 0.5 * (k - 1 - i);
    return s;
}

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        gesvd_section(ctx);
        values_only_section(ctx);
        variants_section(ctx);
        symmetric_section(ctx);
        gebrd_section(ctx);
        bdsqr_section(ctx);
        ormbr_section(ctx);
    }

    // Largest |A - U diag(s) V^H| over the batch.
    static double reconstruction_error(const std::vector<HostMatrix<double>>& originals,
                                       const Matrix<double>& U, const UnifiedVector<double>& s,
                                       const Matrix<double>& Vh, int k) {
        double worst = 0.0;
        for (size_t b = 0; b < originals.size(); ++b) {
            auto Uh = to_host(U, static_cast<int>(b));
            auto Vhh = to_host(Vh, static_cast<int>(b));
            HostMatrix<double> US(Uh.rows, k);
            for (int j = 0; j < k; ++j)
                for (int i = 0; i < Uh.rows; ++i) US(i, j) = Uh(i, j) * s[b * k + j];
            HostMatrix<double> Vk(k, Vhh.cols);
            for (int j = 0; j < Vhh.cols; ++j)
                for (int i = 0; i < k; ++i) Vk(i, j) = Vhh(i, j);
            worst = std::max(worst, max_abs_diff(matmul(US, Vk), originals[b]));
        }
        return worst;
    }

    // -----------------------------------------------------------------------
    // gesvd — the general driver.
    //
    // A is overwritten. You supply U (m x m), V^H (n x n) and a real
    // `singular_values` span of length min(m, n) per batch item — note that
    // the values are real even when A is complex, hence `float_t<T>`.
    // SvdVectors::All asks for the vectors, SvdVectors::None skips them.
    // -----------------------------------------------------------------------
    static void gesvd_section(Queue& ctx) {
        section("gesvd - the general driver");

        const int m = 8, n = 5, k = std::min(m, n);
        auto A = Matrix<double>(m, n, kBatch);
        std::vector<HostMatrix<double>> originals;
        for (int b = 0; b < kBatch; ++b) {
            originals.push_back(with_singular_values<double>(m, target_svals(n), 11 + b));
            from_host(originals.back(), A, b);
        }

        auto U = Matrix<double>::Zeros(m, m, kBatch);
        auto Vh = Matrix<double>::Zeros(n, n, kBatch);
        UnifiedVector<double> s(static_cast<size_t>(k) * kBatch);

        const size_t bytes =
            gesvd_buffer_size<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(), SvdVectors::All, SvdVectors::All);
        UnifiedVector<std::byte> ws(bytes);
        gesvd<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(), SvdVectors::All, SvdVectors::All, ws.to_span());
        ctx.wait();

        // The singular values are the ones we built in.
        const auto want = target_svals(n);
        double sval_err = 0.0;
        for (int b = 0; b < kBatch; ++b)
            for (int i = 0; i < k; ++i) sval_err = std::max(sval_err, std::abs(s[b * k + i] - want[i]));
        report_error("singular values vs the requested spectrum", sval_err, 1e-10);

        report_error("|A - U diag(s) V^H|", reconstruction_error(originals, U, s, Vh, k), 1e-10);

        double uo = 0.0, vo = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            uo = std::max(uo, orthogonality_error(to_host(U, b)));
            vo = std::max(vo, orthogonality_error(transposed(to_host(Vh, b))));
        }
        report_error("|U^T U - I|", uo, 1e-10);
        report_error("|V^T V - I|", vo, 1e-10);
    }

    // -----------------------------------------------------------------------
    // Values only
    //
    // SvdVectors::None skips the back-transforms. U and V^H still have to be
    // passed — the signature has no overload without them — but they are not
    // written, so a 1x1 placeholder is enough.
    // -----------------------------------------------------------------------
    static void values_only_section(Queue& ctx) {
        section("Values only");

        const int m = 8, n = 5, k = std::min(m, n);
        auto A = Matrix<double>(m, n, kBatch);
        for (int b = 0; b < kBatch; ++b) from_host(with_singular_values<double>(m, target_svals(n), 11 + b), A, b);

        auto U = Matrix<double>::Zeros(m, m, kBatch);
        auto Vh = Matrix<double>::Zeros(n, n, kBatch);
        UnifiedVector<double> s(static_cast<size_t>(k) * kBatch);

        UnifiedVector<std::byte> ws(gesvd_buffer_size<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(),
                                                         SvdVectors::None, SvdVectors::None));
        gesvd<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(), SvdVectors::None, SvdVectors::None, ws.to_span());
        ctx.wait();

        const auto want = target_svals(n);
        double err = 0.0;
        for (int b = 0; b < kBatch; ++b)
            for (int i = 0; i < k; ++i) err = std::max(err, std::abs(s[b * k + i] - want[i]));
        report_error("values-only singular values", err, 1e-10);
    }

    // -----------------------------------------------------------------------
    // The native variants
    //
    // gesvd dispatches to a vendor library. `gesvd_blocked` and `gesvd_cta`
    // are BatchLAS's own implementations, callable directly when you want a
    // specific one: blocked for medium sizes, CTA for very small square
    // matrices (n <= 32, one work-group per matrix, so GPU only).
    // -----------------------------------------------------------------------
    static void variants_section(Queue& ctx) {
        section("The blocked and CTA drivers");

        const int n = 6;
        auto make = [&](Matrix<double>& A, std::vector<HostMatrix<double>>& originals) {
            for (int b = 0; b < kBatch; ++b) {
                originals.push_back(with_singular_values<double>(n, target_svals(n), 31 + b));
                from_host(originals.back(), A, b);
            }
        };

        {
            auto A = Matrix<double>(n, n, kBatch);
            std::vector<HostMatrix<double>> originals;
            make(A, originals);
            auto U = Matrix<double>::Zeros(n, n, kBatch);
            auto Vh = Matrix<double>::Zeros(n, n, kBatch);
            UnifiedVector<double> s(static_cast<size_t>(n) * kBatch);
            UnifiedVector<std::byte> ws(gesvd_blocked_buffer_size<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(),
                                                                     SvdVectors::All, SvdVectors::All));
            gesvd_blocked<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(), SvdVectors::All, SvdVectors::All,
                             ws.to_span());
            ctx.wait();
            report_error("gesvd_blocked |A - U S V^H|", reconstruction_error(originals, U, s, Vh, n), 1e-9);
        }

        if constexpr (has_cta_variants<B>) {
        if (supports_cta(ctx)) {
            auto A = Matrix<double>(n, n, kBatch);
            std::vector<HostMatrix<double>> originals;
            make(A, originals);
            auto U = Matrix<double>::Zeros(n, n, kBatch);
            auto Vh = Matrix<double>::Zeros(n, n, kBatch);
            UnifiedVector<double> s(static_cast<size_t>(n) * kBatch);
            UnifiedVector<std::byte> ws(gesvd_cta_buffer_size<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(),
                                                                 SvdVectors::All, SvdVectors::All));
            gesvd_cta<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(), SvdVectors::All, SvdVectors::All,
                         ws.to_span());
            ctx.wait();
            report_error("gesvd_cta |A - U S V^H|", reconstruction_error(originals, U, s, Vh, n), 1e-9);
        } else {
            report_skip("gesvd_cta", "needs a GPU with sub-group width 32");
        }
        } else {
            report_skip("gesvd_cta", "not instantiated for this backend");
        }
    }

    // -----------------------------------------------------------------------
    // Symmetric input
    //
    // The overload taking a `Uplo` says "this matrix is symmetric/Hermitian,
    // read only that triangle". For a symmetric positive definite matrix the
    // singular values are the eigenvalues, which gives a check with a known
    // answer.
    // -----------------------------------------------------------------------
    static void symmetric_section(Queue& ctx) {
        section("Symmetric input");

        const int n = 6;
        std::vector<double> eigs(n);
        for (int i = 0; i < n; ++i) eigs[i] = 1.0 + i;  // all positive, so |lambda| == sigma

        const auto full = symmetric_with_eigenvalues<double>(eigs, 41);

        // Passing the *full* symmetric matrix is always safe.
        auto solve = [&](const HostMatrix<double>& input) {
            auto A = broadcast(input, kBatch);
            auto U = Matrix<double>::Zeros(n, n, kBatch);
            auto Vh = Matrix<double>::Zeros(n, n, kBatch);
            UnifiedVector<double> s(static_cast<size_t>(n) * kBatch);
            UnifiedVector<std::byte> ws(gesvd_buffer_size<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(),
                                                             SvdVectors::All, SvdVectors::All, Uplo::Lower));
            gesvd<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(), SvdVectors::All, SvdVectors::All, Uplo::Lower,
                     ws.to_span());
            ctx.wait();
            std::vector<double> got(s.begin(), s.begin() + n);
            std::sort(got.begin(), got.end());
            return got;
        };

        report_error("singular values of a symmetric matrix", max_abs_diff(solve(full), eigs), 1e-9);

        // Filling only the nominated triangle is the point of the overload —
        // but the host backend reads the whole matrix regardless, so this is
        // only checked where it holds. See the known issues in the README.
        if constexpr (B == Backend::NETLIB) {
            report_skip("half-filled symmetric input", "known defect: NETLIB ignores the uplo hint here");
        } else {
            report_error("half-filled symmetric input", max_abs_diff(solve(keep_triangle(full, Uplo::Lower)), eigs),
                         1e-9);
        }
    }

    // -----------------------------------------------------------------------
    // gebrd — reduction to bidiagonal form.
    //
    // The first half of any SVD: A = Q * Bd * P^H with Bd bidiagonal, returned
    // as its diagonal `d` and superdiagonal `e`. Q and P stay in packed
    // reflector form in A plus the scalars tauq/taup, exactly like geqrf.
    //
    // Three implementations, same contract: unblocked, cta (small, GPU), and
    // blocked (the only one taking a workspace).
    // -----------------------------------------------------------------------
    static void gebrd_section(Queue& ctx) {
        section("gebrd - reduction to bidiagonal form");

        const int n = 8;
        auto original = with_singular_values<double>(n, target_svals(n), 51);

        // The singular values of the bidiagonal factor equal those of A, which
        // is what makes the reduction useful — and gives us a check that needs
        // no reference implementation of the reduction itself.
        auto check_bidiagonal = [&](const char* name, Vector<double>& d, Vector<double>& e) {
            HostMatrix<double> Bd(n, n);
            for (int i = 0; i < n; ++i) Bd(i, i) = d(i, 0);
            for (int i = 0; i + 1 < n; ++i) Bd(i, i + 1) = e(i, 0);

            // Singular values of Bd via the eigenvalues of Bd^T Bd, computed
            // on the host with a plain Jacobi sweep.
            auto BtB = matmul(Bd, Bd, Transpose::Trans, Transpose::NoTrans);
            auto vals = jacobi_eigenvalues(BtB);
            for (auto& v : vals) v = std::sqrt(std::max(0.0, v));
            std::sort(vals.begin(), vals.end());
            auto want = target_svals(n);
            std::sort(want.begin(), want.end());
            report_error(std::string(name) + ": singular values preserved", max_abs_diff(vals, want), 1e-8);
        };

        {
            auto A = broadcast(original, kBatch);
            Vector<double> d(n, kBatch), e(n - 1, kBatch), tauq(n, kBatch), taup(n, kBatch);
            gebrd_unblocked<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e),
                               VectorView<double>(tauq), VectorView<double>(taup));
            ctx.wait();
            check_bidiagonal("gebrd_unblocked", d, e);
        }

        {
            auto A = broadcast(original, kBatch);
            Vector<double> d(n, kBatch), e(n - 1, kBatch), tauq(n, kBatch), taup(n, kBatch);
            UnifiedVector<std::byte> ws(gebrd_blocked_buffer_size<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e),
                                                                     VectorView<double>(tauq), VectorView<double>(taup), 16));
            gebrd_blocked<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e),
                             VectorView<double>(tauq), VectorView<double>(taup), ws.to_span(), 16);
            ctx.wait();
            check_bidiagonal("gebrd_blocked", d, e);
        }

        if constexpr (has_cta_variants<B>) {
            if (supports_cta(ctx)) {
                auto A = broadcast(original, kBatch);
                Vector<double> d(n, kBatch), e(n - 1, kBatch), tauq(n, kBatch), taup(n, kBatch);
                gebrd_cta<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e),
                             VectorView<double>(tauq), VectorView<double>(taup));
                ctx.wait();
                check_bidiagonal("gebrd_cta", d, e);
            } else {
                report_skip("gebrd_cta", "needs a GPU with sub-group width 32");
            }
        } else {
            report_skip("gebrd_cta", "not instantiated for this backend");
        }
    }

    // -----------------------------------------------------------------------
    // bdsqr — bidiagonal QR iteration.
    //
    // The second half: singular values from (d, e). The values-only overload
    // just writes them out; the other one also accumulates the rotations into
    // matrices you supply, which is how the full SVD gets its vectors.
    // -----------------------------------------------------------------------
    static void bdsqr_section(Queue& ctx) {
        section("bdsqr - bidiagonal QR iteration");

        const int n = 8;
        auto original = with_singular_values<double>(n, target_svals(n), 61);
        auto A = broadcast(original, kBatch);
        Vector<double> d(n, kBatch), e(n - 1, kBatch), tauq(n, kBatch), taup(n, kBatch);
        gebrd_unblocked<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e),
                               VectorView<double>(tauq), VectorView<double>(taup));
        ctx.wait();

        UnifiedVector<double> s(static_cast<size_t>(n) * kBatch);
        UnifiedVector<std::byte> ws(bdsqr_buffer_size<double>(ctx, d, e, s.to_span()));
        bdsqr<B>(ctx, VectorView<double>(d), VectorView<double>(e), s.to_span(), ws.to_span(), /*sort_desc=*/true);
        ctx.wait();

        auto want = target_svals(n);  // already descending
        double err = 0.0;
        for (int b = 0; b < kBatch; ++b)
            for (int i = 0; i < n; ++i) err = std::max(err, std::abs(s[b * n + i] - want[i]));
        report_error("gebrd + bdsqr singular values", err, 1e-9);
        report_check("sort_desc gives descending order", s[0] >= s[1] && s[1] >= s[2]);
    }

    // -----------------------------------------------------------------------
    // ormbr — apply the Q or P factor from gebrd.
    //
    // `vect` selects which: 'Q' uses tauq (the left reflectors), 'P' uses taup
    // (the right ones). This is the back-transform that turns the bidiagonal
    // problem's vectors into A's. Applying Q and then Q^T must give back what
    // you started with, which is the check here.
    // -----------------------------------------------------------------------
    static void ormbr_section(Queue& ctx) {
        section("ormbr - apply the Q or P factor from gebrd");

        const int n = 8;
        auto A = broadcast(with_singular_values<double>(n, target_svals(n), 71), kBatch);
        Vector<double> d(n, kBatch), e(n - 1, kBatch), tauq(n, kBatch), taup(n, kBatch);
        gebrd_unblocked<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e),
                               VectorView<double>(tauq), VectorView<double>(taup));
        ctx.wait();

        auto C = batch_of<double>(n, n, kBatch, random_host<double>, 81);
        std::vector<HostMatrix<double>> originals;
        for (int b = 0; b < kBatch; ++b) originals.push_back(to_host(C, b));

        UnifiedVector<std::byte> ws(
            ormbr_buffer_size<B>(ctx, A.view(), VectorView<double>(tauq), C.view(), 'Q', Side::Left,
                                 Transpose::NoTrans));
        ormbr<B>(ctx, A.view(), VectorView<double>(tauq), C.view(), 'Q', Side::Left, Transpose::NoTrans, ws.to_span());
        ctx.wait();

        // Q is orthogonal, so applying it must preserve column norms.
        double norm_err = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            auto got = to_host(C, b);
            for (int j = 0; j < n; ++j) {
                double a = 0.0, c = 0.0;
                for (int i = 0; i < n; ++i) {
                    a += originals[b](i, j) * originals[b](i, j);
                    c += got(i, j) * got(i, j);
                }
                norm_err = std::max(norm_err, std::abs(std::sqrt(a) - std::sqrt(c)));
            }
        }
        report_error("ormbr('Q') preserves column norms", norm_err, 1e-10);

        // Apply Q^T to undo it.
        ormbr<B>(ctx, A.view(), VectorView<double>(tauq), C.view(), 'Q', Side::Left, Transpose::Trans, ws.to_span());
        ctx.wait();

        double round_trip = 0.0;
        for (int b = 0; b < kBatch; ++b) round_trip = std::max(round_trip, max_abs_diff(to_host(C, b), originals[b]));
        report_error("ormbr('Q') then ormbr('Q^T') is the identity", round_trip, 1e-10);
    }

};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("5. Singular value decomposition")
