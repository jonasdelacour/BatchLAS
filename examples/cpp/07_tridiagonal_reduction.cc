// 7. Reduction to tridiagonal form
//
// Every dense symmetric eigensolver starts by reducing A to a symmetric
// tridiagonal matrix with the same eigenvalues. BatchLAS exposes each way of
// doing that separately, so you can build your own pipeline or benchmark the
// stages: sytrd_cta, sytrd_blocked, the two-stage pair sytrd_sy2sb +
// sytrd_sb2st, and the BANDR1-style sytrd_band_reduction.
//
// Most of this is GPU-only. Each section prints the (d, e) it produced and, to
// show the reduction really is a similarity transform, feeds them to stedc and
// compares with syev on the original dense matrix.

#include <complex>
#include <cstddef>
#include <iostream>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_utils.hh"

using namespace batchlas;
using namespace examples;

namespace {

constexpr int kBatch = 2;
constexpr int kN = 32;

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        reference_section(ctx);
        cta_section(ctx);
        blocked_section(ctx);
        sy2sb_section(ctx);
        sb2st_section(ctx);
        band_reduction_section(ctx);
        complex_section(ctx);
    }

    // The matrix every section reduces.
    static Matrix<double> input(unsigned seed = 11) {
        return Matrix<double>::Random(kN, kN, /*hermitian=*/true, kBatch, seed);
    }

    // Eigenvalues of the tridiagonal matrix (d, e), via stedc. If the reduction
    // was a similarity transform these match the dense matrix's spectrum.
    static void show_spectrum(Queue& ctx, Vector<double>& d, Vector<double>& e, const char* label) {
        Vector<double> w(kN, kBatch);
        auto V = Matrix<double>::Zeros(kN, kN, kBatch);
        StedcParams<double> params;
        UnifiedVector<std::byte> ws(stedc_workspace_size<B, double>(ctx, kN, kBatch, JobType::EigenVectors, params));
        // Ask for vectors and ignore them: stedc's values-only mode is broken.
        // See the known issues in the README.
        stedc<B>(ctx, VectorView<double>(d), VectorView<double>(e), VectorView<double>(w), ws.to_span(),
                 JobType::EigenVectors, params, V.view());
        ctx.wait();
        std::cout << label << ": ";
        w.batch_item(0).print(std::cout, 6);
    }

    // The spectrum to match, straight from the dense solver.
    static void reference_section(Queue& ctx) {
        section("The spectrum every reduction has to preserve");

        auto A = input();
        UnifiedVector<double> w(static_cast<size_t>(kN) * kBatch);
        UnifiedVector<std::byte> ws(
            syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower));
        syev<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
        ctx.wait();
        print_values("syev on the dense matrix", w.to_span(), 6);
    }

    // sytrd_cta — one work-group per matrix.
    //
    // For n <= 32 on a GPU. Writes the diagonal into d, the sub-diagonal into e
    // (length n-1) and the Householder scalars into tau (length n-1); A is
    // overwritten with the reflectors, in the packed layout geqrf uses.
    static void cta_section(Queue& ctx) {
        section("sytrd_cta - one work-group per matrix");

        if constexpr (!has_cta_variants<B>) {
            skip("sytrd_cta", "not instantiated for this backend");
            return;
        } else {
            if (!supports_cta(ctx)) {
                skip("sytrd_cta", "needs a GPU with sub-group width 32");
                return;
            }
            auto A = input();
            Vector<double> d(kN, kBatch), e(kN - 1, kBatch), tau(kN - 1, kBatch);
            sytrd_cta<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tau),
                         Uplo::Lower, Span<std::byte>());
            ctx.wait();
            std::cout << "diagonal:     ";
            d.batch_item(0).print(std::cout, 6);
            std::cout << "off-diagonal: ";
            e.batch_item(0).print(std::cout, 6);
            show_spectrum(ctx, d, e, "eigenvalues of the tridiagonal factor");
        }
    }

    // sytrd_blocked — blocked panel plus a BLAS-3 trailing update.
    //
    // The n > 32 counterpart. `block_size` is the panel width; larger panels
    // mean more BLAS-3 work and fewer passes. Needs an in-order queue,
    // Uplo::Lower, and a GPU.
    static void blocked_section(Queue& ctx) {
        section("sytrd_blocked - blocked panel plus BLAS-3 update");

        if (!on_gpu(ctx)) {
            skip("sytrd_blocked", "GPU only");
            return;
        }

        auto A = input();
        Vector<double> d(kN, kBatch), e(kN - 1, kBatch), tau(kN - 1, kBatch);
        UnifiedVector<std::byte> ws(sytrd_blocked_buffer_size<B>(ctx, A.view(), VectorView<double>(d),
                                                                  VectorView<double>(e), VectorView<double>(tau),
                                                                  Uplo::Lower, /*block_size=*/16));
        sytrd_blocked<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tau),
                         Uplo::Lower, ws.to_span(), /*block_size=*/16);
        ctx.wait();
        show_spectrum(ctx, d, e, "eigenvalues of the tridiagonal factor");
    }

    // Two-stage reduction, step 1 — sytrd_sy2sb (dense -> band).
    //
    // Reduces A to a band matrix of semibandwidth kd, written into AB in LAPACK
    // band storage: AB is (kd+1) x n and, for Uplo::Lower, AB(i-j, j) = A(i, j).
    // A itself is overwritten with the reflectors and tau has length n-kd.
    //
    // Going dense -> band -> tridiagonal instead of straight to tridiagonal
    // moves most of the work into BLAS-3, which is the whole point.
    static void sy2sb_section(Queue& ctx) {
        section("Two-stage step 1 - sytrd_sy2sb (dense to band)");

        if (!on_gpu(ctx)) {
            skip("sytrd_sy2sb", "GPU only");
            return;
        }

        const int kd = 4;
        auto A = input();
        auto AB = Matrix<double>::Zeros(kd + 1, kN, kBatch);
        Vector<double> tau(kN - kd, kBatch);

        UnifiedVector<std::byte> ws(
            sytrd_sy2sb_buffer_size<B>(ctx, A.view(), AB.view(), VectorView<double>(tau), Uplo::Lower, kd));
        sytrd_sy2sb<B>(ctx, A.view(), AB.view(), VectorView<double>(tau), Uplo::Lower, kd, ws.to_span());
        ctx.wait();

        print("band storage shape", std::to_string(AB.rows()) + " x " + std::to_string(AB.cols()) + " (kd+1 by n)");
        print("tau length (n - kd)", tau.size());
        std::cout << "first columns of the band, row r holds the r-th sub-diagonal:\n";
        AB.view()[0].print(std::cout, kd + 1, 6);
    }

    // Two-stage reduction, step 2 — sytrd_sb2st (band -> tridiagonal).
    //
    // Bulge chasing on the band storage stage 1 produced. Note the output type:
    // d and e are REAL even for complex input, which is why the signature says
    // `VectorView<float_t<T>>`.
    static void sb2st_section(Queue& ctx) {
        section("Two-stage step 2 - sytrd_sb2st (band to tridiagonal)");

        if (!on_gpu(ctx)) {
            skip("sytrd_sb2st", "GPU only");
            return;
        }

        const int kd = 4;
        auto AB = to_band(ctx, kd);

        Vector<double> d(kN, kBatch), e(kN - 1, kBatch), tau(kN - 1, kBatch);
        UnifiedVector<std::byte> ws(sytrd_sb2st_buffer_size<B>(ctx, AB.view(), VectorView<double>(d),
                                                                VectorView<double>(e), VectorView<double>(tau),
                                                                Uplo::Lower, kd, /*block_size=*/32));
        sytrd_sb2st<B>(ctx, AB.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tau),
                       Uplo::Lower, kd, ws.to_span(), /*block_size=*/32);
        ctx.wait();
        show_spectrum(ctx, d, e, "dense -> band -> tridiagonal, eigenvalues");
    }

    // sytrd_band_reduction — the BANDR1 blocked schedule.
    //
    // A different band -> tridiagonal algorithm with the same contract as
    // sb2st. `SytrdBandReductionParams` exposes the schedule: diagonals to
    // eliminate per sweep (d_seq), block size per sweep (block_size_seq), and a
    // sweep cap. The two sequences must be the same length; 0 means "use the
    // implementation default".
    static void band_reduction_section(Queue& ctx) {
        section("sytrd_band_reduction - the BANDR1 blocked schedule");

        if (!on_gpu(ctx)) {
            skip("sytrd_band_reduction", "GPU only");
            return;
        }

        const int kd = 4;
        {
            auto AB = to_band(ctx, kd);
            Vector<double> d(kN, kBatch), e(kN - 1, kBatch), tau(kN - 1, kBatch);
            UnifiedVector<std::byte> ws(sytrd_band_reduction_buffer_size<B>(
                ctx, AB.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tau), Uplo::Lower,
                kd, /*block_size=*/32));
            sytrd_band_reduction<B>(ctx, AB.view(), VectorView<double>(d), VectorView<double>(e),
                                    VectorView<double>(tau), Uplo::Lower, kd, ws.to_span(), /*block_size=*/32);
            ctx.wait();
            show_spectrum(ctx, d, e, "default schedule, eigenvalues");
        }

        {
            SytrdBandReductionParams params;
            params.d_seq = {2, 1};
            params.block_size_seq = {32, 32};

            auto AB = to_band(ctx, kd);
            Vector<double> d(kN, kBatch), e(kN - 1, kBatch), tau(kN - 1, kBatch);
            UnifiedVector<std::byte> ws(sytrd_band_reduction_buffer_size<B>(
                ctx, AB.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tau), Uplo::Lower,
                kd, params));
            sytrd_band_reduction<B>(ctx, AB.view(), VectorView<double>(d), VectorView<double>(e),
                                    VectorView<double>(tau), Uplo::Lower, kd, ws.to_span(), params);
            ctx.wait();
            show_spectrum(ctx, d, e, "d_seq = {2,1}, eigenvalues");
        }
    }

    // Complex Hermitian input
    //
    // hetrd_hb2st is the Hermitian name for sytrd_sb2st — literally an alias,
    // so code can read the way LAPACK does. The tridiagonal output is real
    // either way.
    static void complex_section(Queue& ctx) {
        section("Complex Hermitian input");

        if (!on_gpu(ctx)) {
            skip("hetrd_hb2st", "GPU only");
            return;
        }

        using C64 = std::complex<double>;
        const int kd = 4;
        auto A = Matrix<C64>::Random(kN, kN, /*hermitian=*/true, kBatch, 71);
        auto AB = Matrix<C64>::Zeros(kd + 1, kN, kBatch);
        Vector<C64> tau1(kN - kd, kBatch);

        UnifiedVector<std::byte> ws1(
            sytrd_sy2sb_buffer_size<B>(ctx, A.view(), AB.view(), VectorView<C64>(tau1), Uplo::Lower, kd));
        sytrd_sy2sb<B>(ctx, A.view(), AB.view(), VectorView<C64>(tau1), Uplo::Lower, kd, ws1.to_span());

        // d and e are real even though the matrix is complex.
        Vector<double> d(kN, kBatch), e(kN - 1, kBatch);
        Vector<C64> tau2(kN - 1, kBatch);
        UnifiedVector<std::byte> ws2(hetrd_hb2st_buffer_size<B, C64>(ctx, AB.view(), VectorView<double>(d),
                                                                      VectorView<double>(e), VectorView<C64>(tau2),
                                                                      Uplo::Lower, kd, 32));
        hetrd_hb2st<B, C64>(ctx, AB.view(), VectorView<double>(d), VectorView<double>(e), VectorView<C64>(tau2),
                            Uplo::Lower, kd, ws2.to_span(), 32);
        ctx.wait();

        std::cout << "real diagonal from a Hermitian matrix: ";
        d.batch_item(0).print(std::cout, 6);
        show_spectrum(ctx, d, e, "eigenvalues");
    }

    // Stage 1 on its own, to give the band-to-tridiagonal sections an input.
    static Matrix<double> to_band(Queue& ctx, int kd) {
        auto A = input();
        auto AB = Matrix<double>::Zeros(kd + 1, kN, kBatch);
        Vector<double> tau(kN - kd, kBatch);
        UnifiedVector<std::byte> ws(
            sytrd_sy2sb_buffer_size<B>(ctx, A.view(), AB.view(), VectorView<double>(tau), Uplo::Lower, kd));
        sytrd_sy2sb<B>(ctx, A.view(), AB.view(), VectorView<double>(tau), Uplo::Lower, kd, ws.to_span());
        ctx.wait();
        return AB;
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("7. Reduction to tridiagonal form")
