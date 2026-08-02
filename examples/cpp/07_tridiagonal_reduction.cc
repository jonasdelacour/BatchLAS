// 7. Reduction to tridiagonal form
//
// Every dense symmetric eigensolver starts by reducing A to a symmetric
// tridiagonal matrix with the same eigenvalues. BatchLAS exposes each way of
// doing that separately, so you can build your own pipeline or benchmark the
// stages: sytrd_cta, sytrd_blocked, the two-stage pair sytrd_sy2sb +
// sytrd_sb2st, and the BANDR1-style sytrd_band_reduction.
//
// The check throughout is the one that matters: a similarity transform must
// preserve the spectrum. We start from a matrix with a known spectrum and
// confirm the tridiagonal factor still has it.

#include <algorithm>
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

constexpr int kBatch = 2;

std::vector<double> target_spectrum(int n) {
    std::vector<double> w(n);
    for (int i = 0; i < n; ++i) w[i] = -3.0 + 0.5 * i;
    return w;
}

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        cta_section(ctx);
        blocked_section(ctx);
        sy2sb_section(ctx);
        sb2st_section(ctx);
        band_reduction_section(ctx);
        pipeline_section(ctx);
        complex_section(ctx);
    }

    // Pull (d, e) for batch item b onto the host and compare the tridiagonal
    // matrix's spectrum with what we built into A.
    static void check_spectrum(const char* name, Vector<double>& d, Vector<double>& e, int n, double tol,
                               int batch = kBatch) {
        double worst = 0.0;
        for (int b = 0; b < batch; ++b) {
            std::vector<double> dh(n), eh(std::max(0, n - 1));
            for (int i = 0; i < n; ++i) dh[i] = d(i, b);
            for (int i = 0; i + 1 < n; ++i) eh[i] = e(i, b);
            worst = std::max(worst, max_abs_diff(tridiagonal_eigenvalues(dh, eh), target_spectrum(n)));
        }
        report_error(std::string(name) + ": spectrum preserved", worst, tol);
    }

    // -----------------------------------------------------------------------
    // sytrd_cta — one work-group per matrix.
    //
    // For n <= 32 on a GPU. Writes the diagonal into d, the sub-diagonal into
    // e (length n-1) and the Householder scalars into tau (length n-1); A is
    // overwritten with the reflectors, in the same packed layout geqrf uses.
    // -----------------------------------------------------------------------
    static void cta_section(Queue& ctx) {
        section("sytrd_cta - one work-group per matrix");

        if constexpr (!has_cta_variants<B>) {
            report_skip("sytrd_cta", "not instantiated for this backend");
            return;
        } else {
            if (!supports_cta(ctx)) {
                report_skip("sytrd_cta", "needs a GPU with sub-group width 32");
                return;
            }
            const int n = 24;
            auto A = broadcast(symmetric_with_eigenvalues<double>(target_spectrum(n), 11), kBatch);
            Vector<double> d(n, kBatch), e(n - 1, kBatch), tau(n - 1, kBatch);
            sytrd_cta<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tau),
                         Uplo::Lower, Span<std::byte>());
            ctx.wait();
            check_spectrum("sytrd_cta", d, e, n, 1e-9);
        }
    }

    // -----------------------------------------------------------------------
    // sytrd_blocked — blocked panel plus a BLAS-3 trailing update.
    //
    // The n > 32 counterpart. `block_size` is the panel width; larger panels
    // mean more BLAS-3 work and fewer passes. Requires an in-order queue,
    // Uplo::Lower, and a GPU.
    // -----------------------------------------------------------------------
    static void blocked_section(Queue& ctx) {
        section("sytrd_blocked - blocked panel plus BLAS-3 update");

        if (!on_gpu(ctx)) {
            report_skip("sytrd_blocked", "GPU only");
            return;
        }

        const int n = 64;
        for (int32_t block_size : {16, 32}) {
            auto A = broadcast(symmetric_with_eigenvalues<double>(target_spectrum(n), 21), kBatch);
            Vector<double> d(n, kBatch), e(n - 1, kBatch), tau(n - 1, kBatch);
            UnifiedVector<std::byte> ws(sytrd_blocked_buffer_size<B>(
                ctx, A.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tau), Uplo::Lower,
                block_size));
            sytrd_blocked<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tau),
                             Uplo::Lower, ws.to_span(), block_size);
            ctx.wait();
            check_spectrum(("sytrd_blocked(block_size=" + std::to_string(block_size) + ")").c_str(), d, e, n, 1e-8);
        }
    }

    // -----------------------------------------------------------------------
    // Two-stage reduction, step 1 — sytrd_sy2sb (dense -> band).
    //
    // Reduces A to a band matrix of semibandwidth kd, written into AB in
    // LAPACK band storage: AB is (kd+1) x n and, for Uplo::Lower,
    // AB(i-j, j) = A(i, j). A itself is overwritten with the reflectors and
    // tau has length n-kd.
    //
    // Going dense -> band -> tridiagonal instead of straight to tridiagonal
    // moves most of the work into BLAS-3, which is the whole point.
    // -----------------------------------------------------------------------
    static void sy2sb_section(Queue& ctx) {
        section("Two-stage step 1 - sytrd_sy2sb (dense to band)");

        if (!on_gpu(ctx)) {
            report_skip("sytrd_sy2sb", "GPU only");
            return;
        }

        const int n = 64, kd = 8;
        auto A = broadcast(symmetric_with_eigenvalues<double>(target_spectrum(n), 31), kBatch);
        auto AB = Matrix<double>::Zeros(kd + 1, n, kBatch);
        Vector<double> tau(n - kd, kBatch);

        UnifiedVector<std::byte> ws(
            sytrd_sy2sb_buffer_size<B>(ctx, A.view(), AB.view(), VectorView<double>(tau), Uplo::Lower, kd));
        sytrd_sy2sb<B>(ctx, A.view(), AB.view(), VectorView<double>(tau), Uplo::Lower, kd, ws.to_span());
        ctx.wait();

        report("band storage shape", std::to_string(AB.rows()) + " x " + std::to_string(AB.cols()) + " (kd+1 by n)");

        // The band matrix is similar to A, so its spectrum is unchanged.
        double worst = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            worst = std::max(worst, max_abs_diff(jacobi_eigenvalues(band_to_dense(to_host(AB, b), kd, n)),
                                                 target_spectrum(n)));
        }
        report_error("sytrd_sy2sb: spectrum preserved", worst, 1e-8);

        // Everything outside the band really is zero.
        double outside = 0.0;
        for (int b = 0; b < kBatch; ++b) {
            auto dense = band_to_dense(to_host(AB, b), kd, n);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    if (std::abs(i - j) > kd) outside = std::max(outside, std::abs(dense(i, j)));
        }
        report_error("nothing outside the band", outside, 0.0);
    }

    // -----------------------------------------------------------------------
    // Two-stage reduction, step 2 — sytrd_sb2st (band -> tridiagonal).
    //
    // Bulge chasing. Takes the same band storage sy2sb produced and gives back
    // (d, e). Note the output type: d and e are REAL even for complex input,
    // which is why the signature says `VectorView<float_t<T>>`.
    // -----------------------------------------------------------------------
    static void sb2st_section(Queue& ctx) {
        section("Two-stage step 2 - sytrd_sb2st (band to tridiagonal)");

        if (!on_gpu(ctx)) {
            report_skip("sytrd_sb2st", "GPU only");
            return;
        }

        const int n = 64, kd = 8;
        auto AB = band_from_known_spectrum(ctx, n, kd, 41);

        Vector<double> d(n, kBatch), e(n - 1, kBatch), tau(n - 1, kBatch);
        const int32_t block_size = 32;
        UnifiedVector<std::byte> ws(sytrd_sb2st_buffer_size<B>(ctx, AB.view(), VectorView<double>(d),
                                                                VectorView<double>(e), VectorView<double>(tau),
                                                                Uplo::Lower, kd, block_size));
        sytrd_sb2st<B>(ctx, AB.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tau),
                       Uplo::Lower, kd, ws.to_span(), block_size);
        ctx.wait();

        check_spectrum("sytrd_sb2st", d, e, n, 1e-8);
    }

    // -----------------------------------------------------------------------
    // sytrd_band_reduction — the BANDR1 blocked schedule.
    //
    // A different band -> tridiagonal algorithm with the same contract as
    // sb2st. `SytrdBandReductionParams` exposes the schedule: how many
    // diagonals to eliminate per sweep (d_seq), the block size per sweep
    // (block_size_seq), and a cap on sweeps. Zeros mean "use the default".
    // -----------------------------------------------------------------------
    static void band_reduction_section(Queue& ctx) {
        section("sytrd_band_reduction - the BANDR1 blocked schedule");

        if (!on_gpu(ctx)) {
            report_skip("sytrd_band_reduction", "GPU only");
            return;
        }

        const int n = 64, kd = 8;

        {
            auto AB = band_from_known_spectrum(ctx, n, kd, 51);
            Vector<double> d(n, kBatch), e(n - 1, kBatch), tau(n - 1, kBatch);
            UnifiedVector<std::byte> ws(sytrd_band_reduction_buffer_size<B>(
                ctx, AB.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tau), Uplo::Lower, kd,
                /*block_size=*/32));
            sytrd_band_reduction<B>(ctx, AB.view(), VectorView<double>(d), VectorView<double>(e),
                                    VectorView<double>(tau), Uplo::Lower, kd, ws.to_span(), /*block_size=*/32);
            ctx.wait();
            check_spectrum("sytrd_band_reduction", d, e, n, 1e-8);
        }

        {
            // The same reduction with an explicit schedule.
            // One entry per sweep, and the two sequences must be the same
            // length — the buffer-size query rejects a mismatch.
            SytrdBandReductionParams params;
            params.d_seq = {2, 1};
            params.block_size_seq = {32, 32};
            params.max_sweeps = -1;  // implementation default

            auto AB = band_from_known_spectrum(ctx, n, kd, 51);
            Vector<double> d(n, kBatch), e(n - 1, kBatch), tau(n - 1, kBatch);
            UnifiedVector<std::byte> ws(sytrd_band_reduction_buffer_size<B>(
                ctx, AB.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tau), Uplo::Lower, kd,
                params));
            sytrd_band_reduction<B>(ctx, AB.view(), VectorView<double>(d), VectorView<double>(e),
                                    VectorView<double>(tau), Uplo::Lower, kd, ws.to_span(), params);
            ctx.wait();
            check_spectrum("sytrd_band_reduction(d_seq={2,1})", d, e, n, 1e-8);
        }
    }

    // -----------------------------------------------------------------------
    // Putting it together
    //
    // sy2sb then sb2st is what syev_two_stage does internally. Running the two
    // by hand gives the same tridiagonal factor — and the same eigenvalues as
    // the one-stage sytrd_blocked.
    // -----------------------------------------------------------------------
    static void pipeline_section(Queue& ctx) {
        section("Putting it together");

        if (!on_gpu(ctx)) {
            report_skip("two-stage pipeline", "GPU only");
            return;
        }

        const int n = 64, kd = 8;
        const auto original = symmetric_with_eigenvalues<double>(target_spectrum(n), 61);

        // Stage 1 then stage 2, by hand.
        std::vector<double> two_stage;
        {
            auto A = broadcast(original, kBatch);
            auto AB = Matrix<double>::Zeros(kd + 1, n, kBatch);
            Vector<double> tau1(n - kd, kBatch);
            UnifiedVector<std::byte> ws1(
                sytrd_sy2sb_buffer_size<B>(ctx, A.view(), AB.view(), VectorView<double>(tau1), Uplo::Lower, kd));
            sytrd_sy2sb<B>(ctx, A.view(), AB.view(), VectorView<double>(tau1), Uplo::Lower, kd, ws1.to_span());

            Vector<double> d(n, kBatch), e(n - 1, kBatch), tau2(n - 1, kBatch);
            UnifiedVector<std::byte> ws2(sytrd_sb2st_buffer_size<B>(ctx, AB.view(), VectorView<double>(d),
                                                                     VectorView<double>(e), VectorView<double>(tau2),
                                                                     Uplo::Lower, kd, 32));
            sytrd_sb2st<B>(ctx, AB.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tau2),
                           Uplo::Lower, kd, ws2.to_span(), 32);
            ctx.wait();

            std::vector<double> dh(n), eh(n - 1);
            for (int i = 0; i < n; ++i) dh[i] = d(i, 0);
            for (int i = 0; i + 1 < n; ++i) eh[i] = e(i, 0);
            two_stage = tridiagonal_eigenvalues(dh, eh);
        }

        // The one-stage reduction, for comparison.
        std::vector<double> one_stage;
        {
            auto A = broadcast(original, kBatch);
            Vector<double> d(n, kBatch), e(n - 1, kBatch), tau(n - 1, kBatch);
            UnifiedVector<std::byte> ws(sytrd_blocked_buffer_size<B>(ctx, A.view(), VectorView<double>(d),
                                                                      VectorView<double>(e), VectorView<double>(tau),
                                                                      Uplo::Lower, 32));
            sytrd_blocked<B>(ctx, A.view(), VectorView<double>(d), VectorView<double>(e), VectorView<double>(tau),
                             Uplo::Lower, ws.to_span(), 32);
            ctx.wait();

            std::vector<double> dh(n), eh(n - 1);
            for (int i = 0; i < n; ++i) dh[i] = d(i, 0);
            for (int i = 0; i + 1 < n; ++i) eh[i] = e(i, 0);
            one_stage = tridiagonal_eigenvalues(dh, eh);
        }

        report_error("one-stage vs two-stage eigenvalues", max_abs_diff(one_stage, two_stage), 1e-8);
        report_error("two-stage vs the known spectrum", max_abs_diff(two_stage, target_spectrum(n)), 1e-8);
    }

    // -----------------------------------------------------------------------
    // Complex Hermitian input
    //
    // hetrd_hb2st is the Hermitian name for sytrd_sb2st — literally an alias,
    // provided so code reads the way LAPACK does. The tridiagonal output is
    // real either way.
    // -----------------------------------------------------------------------
    static void complex_section(Queue& ctx) {
        section("Complex Hermitian input");

        if (!on_gpu(ctx)) {
            report_skip("hetrd_hb2st", "GPU only");
            return;
        }

        using C = std::complex<double>;
        const int n = 48, kd = 8;
        const auto want = target_spectrum(n);
        auto A = broadcast(symmetric_with_eigenvalues<C>(want, 71), kBatch);
        auto AB = Matrix<C>::Zeros(kd + 1, n, kBatch);
        Vector<C> tau1(n - kd, kBatch);

        UnifiedVector<std::byte> ws1(
            sytrd_sy2sb_buffer_size<B>(ctx, A.view(), AB.view(), VectorView<C>(tau1), Uplo::Lower, kd));
        sytrd_sy2sb<B>(ctx, A.view(), AB.view(), VectorView<C>(tau1), Uplo::Lower, kd, ws1.to_span());

        // d and e are real even though the matrix is complex.
        Vector<double> d(n, kBatch), e(n - 1, kBatch);
        Vector<C> tau2(n - 1, kBatch);
        UnifiedVector<std::byte> ws2(hetrd_hb2st_buffer_size<B, C>(ctx, AB.view(), VectorView<double>(d),
                                                                    VectorView<double>(e), VectorView<C>(tau2),
                                                                    Uplo::Lower, kd, 32));
        hetrd_hb2st<B, C>(ctx, AB.view(), VectorView<double>(d), VectorView<double>(e), VectorView<C>(tau2),
                          Uplo::Lower, kd, ws2.to_span(), 32);
        ctx.wait();

        check_spectrum("hetrd_hb2st (Hermitian)", d, e, n, 1e-8);
    }

    // A band matrix whose spectrum is target_spectrum(n), obtained by running
    // stage 1 on a dense matrix we constructed.
    static Matrix<double> band_from_known_spectrum(Queue& ctx, int n, int kd, unsigned seed) {
        auto A = broadcast(symmetric_with_eigenvalues<double>(target_spectrum(n), seed), kBatch);
        auto AB = Matrix<double>::Zeros(kd + 1, n, kBatch);
        Vector<double> tau(n - kd, kBatch);
        UnifiedVector<std::byte> ws(
            sytrd_sy2sb_buffer_size<B>(ctx, A.view(), AB.view(), VectorView<double>(tau), Uplo::Lower, kd));
        sytrd_sy2sb<B>(ctx, A.view(), AB.view(), VectorView<double>(tau), Uplo::Lower, kd, ws.to_span());
        ctx.wait();
        return AB;
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("7. Reduction to tridiagonal form")
