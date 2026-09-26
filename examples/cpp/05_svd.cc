// 5. Singular value decomposition
//
// The gesvd driver and its native variants, plus the pieces they are built
// from: gebrd (reduction to bidiagonal form), bdsqr (bidiagonal QR iteration)
// and ormbr (applying the reduction's reflectors).
//
// A = U * diag(s) * V^H, with s descending.

#include <cstddef>
#include <iostream>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_utils.hh"

using namespace batchlas;
using namespace examples;

namespace {

constexpr int kBatch = 2;

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

    // gesvd — the general driver.
    //
    // A is overwritten. You supply U (m x m), V^H (n x n) and a
    // `singular_values` span of length min(m, n) per batch item — REAL, even
    // when A is complex, hence `Span<float_t<T>>` in the signature.
    // SvdVectors::All asks for the vectors, SvdVectors::None skips them.
    static void gesvd_section(Queue& ctx) {
        section("gesvd - the general driver");

        const int m = 6, n = 4, k = std::min(m, n);
        auto A = Matrix<double>::Random(m, n, false, kBatch, 11);
        auto U = Matrix<double>::Zeros(m, m, kBatch);
        auto Vh = Matrix<double>::Zeros(n, n, kBatch);
        UnifiedVector<double> s(static_cast<size_t>(k) * kBatch);

        UnifiedVector<std::byte> ws(gesvd_buffer_size<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(),
                                                          SvdVectors::All, SvdVectors::All));
        gesvd<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(), SvdVectors::All, SvdVectors::All, ws.to_span());
        ctx.wait();

        print_values("singular values (descending)", s.to_span(), k);
        print("U shape", std::to_string(U.rows()) + "x" + std::to_string(U.cols()));
        print("V^H shape", std::to_string(Vh.rows()) + "x" + std::to_string(Vh.cols()));
        std::cout << "U, item 0:\n";
        U.view()[0].print(std::cout, 6, 6);
    }

    // Values only
    //
    // SvdVectors::None skips the back-transforms. U and V^H still have to be
    // passed — there is no overload without them — but they are not written.
    static void values_only_section(Queue& ctx) {
        section("Values only");

        const int m = 6, n = 4, k = std::min(m, n);
        auto A = Matrix<double>::Random(m, n, false, kBatch, 11);
        auto U = Matrix<double>::Zeros(m, m, kBatch);
        auto Vh = Matrix<double>::Zeros(n, n, kBatch);
        UnifiedVector<double> s(static_cast<size_t>(k) * kBatch);

        UnifiedVector<std::byte> ws(gesvd_buffer_size<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(),
                                                          SvdVectors::None, SvdVectors::None));
        gesvd<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(), SvdVectors::None, SvdVectors::None, ws.to_span());
        ctx.wait();
        print_values("same values, no vectors computed", s.to_span(), k);
    }

    // The native variants
    //
    // gesvd dispatches to a vendor library. `gesvd_blocked` and `gesvd_cta` are
    // BatchLAS's own implementations, callable directly when you want a
    // specific one: blocked for medium sizes, CTA for very small square
    // matrices (n <= 32, one work-group per matrix, so GPU only).
    static void variants_section(Queue& ctx) {
        section("The blocked and CTA drivers");

        const int n = 6;
        auto solve = [&](auto&& call, const char* label) {
            auto A = Matrix<double>::Random(n, n, false, kBatch, 21);
            auto U = Matrix<double>::Zeros(n, n, kBatch);
            auto Vh = Matrix<double>::Zeros(n, n, kBatch);
            UnifiedVector<double> s(static_cast<size_t>(n) * kBatch);
            call(A, s, U, Vh);
            ctx.wait();
            print_values(label, s.to_span(), n);
        };

        solve(
            [&](auto& A, auto& s, auto& U, auto& Vh) {
                UnifiedVector<std::byte> ws(gesvd_blocked_buffer_size<B>(ctx, A.view(), s.to_span(), U.view(),
                                                                         Vh.view(), SvdVectors::All, SvdVectors::All));
                gesvd_blocked<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(), SvdVectors::All, SvdVectors::All,
                                 ws.to_span());
            },
            "gesvd_blocked");

        if constexpr (has_cta_variants<B>) {
            if (supports_cta(ctx)) {
                solve(
                    [&](auto& A, auto& s, auto& U, auto& Vh) {
                        UnifiedVector<std::byte> ws(gesvd_cta_buffer_size<B>(
                            ctx, A.view(), s.to_span(), U.view(), Vh.view(), SvdVectors::All, SvdVectors::All));
                        gesvd_cta<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(), SvdVectors::All,
                                     SvdVectors::All, ws.to_span());
                    },
                    "gesvd_cta");
            } else {
                skip("gesvd_cta", "needs a GPU with sub-group width 32");
            }
        } else {
            skip("gesvd_cta", "not instantiated for this backend");
        }
    }

    // Symmetric input
    //
    // The overload taking a `Uplo` says "this matrix is symmetric/Hermitian,
    // read only that triangle". Note the known defect on NETLIB, where the hint
    // is ignored and the whole matrix is read — passing the full symmetric
    // matrix, as here, is safe everywhere.
    static void symmetric_section(Queue& ctx) {
        section("Symmetric input");

        const int n = 5;
        auto A = Matrix<double>::Random(n, n, /*hermitian=*/true, kBatch, 31);
        auto U = Matrix<double>::Zeros(n, n, kBatch);
        auto Vh = Matrix<double>::Zeros(n, n, kBatch);
        UnifiedVector<double> s(static_cast<size_t>(n) * kBatch);

        UnifiedVector<std::byte> ws(gesvd_buffer_size<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(),
                                                          SvdVectors::All, SvdVectors::All, Uplo::Lower));
        gesvd<B>(ctx, A.view(), s.to_span(), U.view(), Vh.view(), SvdVectors::All, SvdVectors::All, Uplo::Lower,
                 ws.to_span());
        ctx.wait();
        print_values("singular values (= |eigenvalues| for a symmetric matrix)", s.to_span(), n);
    }

    // gebrd — reduction to bidiagonal form.
    //
    // The first half of any SVD: A = Q * Bd * P^H with Bd bidiagonal, returned
    // as its diagonal `d` and superdiagonal `e`. Q and P stay in packed
    // reflector form in A plus the scalars tauq/taup, exactly like geqrf.
    //
    // Three implementations with the same contract: unblocked, cta (small,
    // GPU) and blocked (the only one taking a workspace). Note `e` must have
    // length n-1; the blocked path validates this and throws otherwise.
    static void gebrd_section(Queue& ctx) {
        section("gebrd - reduction to bidiagonal form");

        const int n = 8;

        {
            auto A = Matrix<double>::Random(n, n, false, kBatch, 41);
            Vector<double> d(n, kBatch), e(n - 1, kBatch), tauq(n, kBatch), taup(n, kBatch);
            gebrd_unblocked<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tauq),
                               VectorView<double>(taup));
            ctx.wait();
            std::cout << "gebrd_unblocked diagonal:      ";
            d.batch_item(0).print();
            std::cout << "gebrd_unblocked superdiagonal: ";
            e.batch_item(0).print();
        }

        {
            auto A = Matrix<double>::Random(n, n, false, kBatch, 41);
            Vector<double> d(n, kBatch), e(n - 1, kBatch), tauq(n, kBatch), taup(n, kBatch);
            UnifiedVector<std::byte> ws(gebrd_blocked_buffer_size<B>(ctx, A.view(), VectorView<double>(d),
                                                                      VectorView<double>(e), VectorView<double>(tauq),
                                                                      VectorView<double>(taup), 16));
            gebrd_blocked<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tauq),
                             VectorView<double>(taup), ws.to_span(), /*block_size=*/16);
            ctx.wait();
            std::cout << "gebrd_blocked diagonal:        ";
            d.batch_item(0).print();
        }

        if constexpr (has_cta_variants<B>) {
            if (supports_cta(ctx)) {
                auto A = Matrix<double>::Random(n, n, false, kBatch, 41);
                Vector<double> d(n, kBatch), e(n - 1, kBatch), tauq(n, kBatch), taup(n, kBatch);
                gebrd_cta<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tauq),
                             VectorView<double>(taup));
                ctx.wait();
                std::cout << "gebrd_cta diagonal:            ";
                d.batch_item(0).print();
            } else {
                skip("gebrd_cta", "needs a GPU with sub-group width 32");
            }
        } else {
            skip("gebrd_cta", "not instantiated for this backend");
        }
    }

    // bdsqr — bidiagonal QR iteration.
    //
    // The second half: singular values from (d, e). The values-only overload
    // just writes them out; the other also accumulates the rotations into
    // matrices you supply, which is how the full SVD gets its vectors.
    static void bdsqr_section(Queue& ctx) {
        section("bdsqr - bidiagonal QR iteration");

        const int n = 8;
        auto A = Matrix<double>::Random(n, n, false, kBatch, 51);
        Vector<double> d(n, kBatch), e(n - 1, kBatch), tauq(n, kBatch), taup(n, kBatch);
        gebrd_unblocked<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tauq),
                           VectorView<double>(taup));
        ctx.wait();

        UnifiedVector<double> s(static_cast<size_t>(n) * kBatch);
        UnifiedVector<std::byte> ws(bdsqr_buffer_size<double>(ctx, d, e, s.to_span()));
        bdsqr<B>(ctx, VectorView<double>(d), VectorView<double>(e), s.to_span(), ws.to_span(), /*sort_desc=*/true);
        ctx.wait();

        print_values("gebrd then bdsqr: singular values", s.to_span(), n);
    }

    // ormbr — apply the Q or P factor from gebrd.
    //
    // `vect` picks which: 'Q' uses tauq (the left reflectors), 'P' uses taup
    // (the right ones). This is the back-transform that turns the bidiagonal
    // problem's vectors into A's.
    static void ormbr_section(Queue& ctx) {
        section("ormbr - apply the Q or P factor from gebrd");

        const int n = 6;
        auto A = Matrix<double>::Random(n, n, false, kBatch, 61);
        Vector<double> d(n, kBatch), e(n - 1, kBatch), tauq(n, kBatch), taup(n, kBatch);
        gebrd_unblocked<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tauq),
                           VectorView<double>(taup));
        ctx.wait();

        auto C = Matrix<double>::Identity(n, kBatch);
        UnifiedVector<std::byte> ws(ormbr_buffer_size<B>(ctx, A.view(), VectorView<double>(tauq), C.view(), 'Q',
                                                          Side::Left, Transpose::NoTrans));
        ormbr<B>(ctx, A.view(), VectorView<double>(tauq), C.view(), 'Q', Side::Left, Transpose::NoTrans,
                 ws.to_span());
        ctx.wait();

        std::cout << "Q applied to the identity, i.e. Q itself, item 0:\n";
        C.view()[0].print();

        ormbr<B>(ctx, A.view(), VectorView<double>(tauq), C.view(), 'Q', Side::Left, Transpose::Trans, ws.to_span());
        ctx.wait();
        std::cout << "Q^T Q is the identity again:\n";
        C.view()[0].print();
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("5. Singular value decomposition")
