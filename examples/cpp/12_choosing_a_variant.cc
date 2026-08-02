// 12. Choosing a variant
//
// The measurement example. What batching actually buys, how throughput scales
// with batch size, which syev variant wins at which size, and what the same
// work costs on the CPU.
//
// Timings are wall-clock around a queue that is waited on, so they include
// launch overhead — which is the point: for small matrices that overhead *is*
// the cost, and amortising it across a batch is the whole reason this library
// exists.
//
// The numbers below come from whatever machine you run on. Nothing here is a
// pass/fail check except the handful of correctness assertions; the rest is
// reported for you to read.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_common.hh"
#include "example_linalg.hh"
#include "example_runner.hh"

using namespace batchlas;
using namespace examples;

namespace {

// Median wall-clock milliseconds of `reps` runs, after a warm-up.
template <typename F>
double time_ms(Queue& ctx, F&& f, int reps = 5) {
    f();
    ctx.wait();
    std::vector<double> samples;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        ctx.wait();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

std::string fixed(double v, int places = 2) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(places) << v;
    return os.str();
}

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        batching_section(ctx);
        scaling_section(ctx);
        variant_section(ctx);
        device_section(ctx);
        savings_section(ctx);
    }

    // Solve `batch` symmetric eigenproblems of size n in one call.
    static double time_syev(Queue& ctx, int n, int batch) {
        auto A = Matrix<double>::Random(n, n, /*hermitian=*/true, batch, 42);
        UnifiedVector<double> w(static_cast<size_t>(n) * batch);
        UnifiedVector<std::byte> ws(
            syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower));
        ctx.wait();
        return time_ms(ctx, [&] {
            syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws.to_span());
        });
    }

    // -----------------------------------------------------------------------
    // One launch beats a loop
    //
    // The same work, submitted as one batched call versus one call per matrix.
    // For small matrices the per-launch overhead dominates completely.
    // -----------------------------------------------------------------------
    static void batching_section(Queue& ctx) {
        section("One launch beats a loop");

        const int n = 16, batch = 256;

        auto A = Matrix<double>::Random(n, n, true, batch, 42);
        UnifiedVector<double> w(static_cast<size_t>(n) * batch);
        UnifiedVector<std::byte> ws(
            syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower));
        ctx.wait();

        const double batched = time_ms(ctx, [&] {
            syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws.to_span());
        });

        // The same batch, one matrix at a time.
        auto One = Matrix<double>::Random(n, n, true, 1, 42);
        UnifiedVector<double> w1(n);
        UnifiedVector<std::byte> ws1(
            syev_buffer_size<B>(ctx, One.view(), w1.to_span(), JobType::NoEigenVectors, Uplo::Lower));
        ctx.wait();

        const double looped = time_ms(
            ctx,
            [&] {
                for (int i = 0; i < batch; ++i) {
                    syev<B>(ctx, One.view(), w1.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws1.to_span());
                }
            },
            3);

        report(std::to_string(batch) + " matrices of size " + std::to_string(n) + ", batched",
               fixed(batched) + " ms");
        report(std::to_string(batch) + " matrices of size " + std::to_string(n) + ", one at a time",
               fixed(looped) + " ms");
        report("speed-up", fixed(looped / std::max(batched, 1e-9), 1) + "x");
        // Reported, not asserted: on a CPU device the per-call overhead is
        // small enough that a loop can match or beat one batched submission.
        // The gain is a GPU story.
    }

    // -----------------------------------------------------------------------
    // How throughput scales with batch size
    //
    // Time per matrix falls as the batch grows, until the device is saturated
    // and it flattens out. Where that happens is what you want to know when
    // sizing work.
    // -----------------------------------------------------------------------
    static void scaling_section(Queue& ctx) {
        section("How throughput scales with batch size");

        const int n = 32;
        double best_per_matrix = 1e30, worst_per_matrix = 0.0;

        for (int batch : {1, 8, 64, 512}) {
            const double ms = time_syev(ctx, n, batch);
            const double per = ms / batch;
            best_per_matrix = std::min(best_per_matrix, per);
            worst_per_matrix = std::max(worst_per_matrix, per);
            report("batch " + std::to_string(batch),
                   fixed(ms) + " ms total, " + fixed(per * 1000.0, 1) + " us per matrix");
        }
        report("throughput gain from batch 1 to 512",
               fixed(worst_per_matrix / std::max(best_per_matrix, 1e-12), 1) + "x");
    }

    // -----------------------------------------------------------------------
    // Which syev variant wins at which size?
    //
    // syev picks for you. These are the variants it picks between, timed
    // directly so you can see the crossovers — and confirm they all agree.
    //
    // Run with JobType::EigenVectors: syev_two_stage returns wrong eigenvalues
    // in values-only mode for n >= 32 (see the known issues in the README), so
    // a values-only comparison would be measuring a broken path.
    // -----------------------------------------------------------------------
    static void variant_section(Queue& ctx) {
        section("Which syev variant wins at which size?");

        if (!on_gpu(ctx)) {
            report_skip("variant comparison", "the alternatives to the vendor path are GPU only");
            return;
        }

        const int batch = 64;
        for (int n : {16, 32, 64, 128}) {
            // A known, well-separated spectrum, so "do they agree" is measured
            // against the right answer rather than against each other. Random
            // symmetric matrices have eigenvalues near zero, where a relative
            // comparison between solvers means nothing.
            std::vector<double> want(n);
            for (int i = 0; i < n; ++i) want[i] = 1.0 + i;
            const auto original = symmetric_with_eigenvalues<double>(want, 7);

            auto run_one = [&](const char* name, auto&& call) {
                // Correctness first, on a fresh copy and a single solve. These
                // routines overwrite A, so a repeated-call timing loop would
                // be re-solving its own output.
                {
                    auto A = broadcast(original, batch);
                    UnifiedVector<double> w(static_cast<size_t>(n) * batch);
                    ctx.wait();
                    call(A, w);
                    ctx.wait();
                    std::vector<double> vals(w.begin(), w.begin() + n);
                    std::sort(vals.begin(), vals.end());
                    report_error(std::string("n=") + std::to_string(n) + " " + name + ": eigenvalues",
                                 max_abs_diff(vals, want), 1e-8 * want.back());
                }

                // Then the timing. Later repetitions operate on the previous
                // result rather than the original matrix — the same amount of
                // work on different data, which is what we are measuring.
                auto A = broadcast(original, batch);
                UnifiedVector<double> w(static_cast<size_t>(n) * batch);
                ctx.wait();
                const double ms = time_ms(ctx, [&] { call(A, w); }, 3);
                report("n=" + std::to_string(n) + " " + name, fixed(ms) + " ms");
            };

            run_one("syev (dispatched)", [&](Matrix<double>& A, UnifiedVector<double>& w) {
                UnifiedVector<std::byte> ws(
                    syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower));
                syev<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
            });

            if constexpr (has_cta_variants<B>) {
                if (n <= 32 && supports_cta(ctx)) {
                    run_one("syev_cta", [&](Matrix<double>& A, UnifiedVector<double>& w) {
                        UnifiedVector<std::byte> ws(syev_cta_buffer_size<B>(ctx, A.view(), JobType::EigenVectors));
                        syev_cta<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
                    });
                }
            }

            run_one("syev_blocked", [&](Matrix<double>& A, UnifiedVector<double>& w) {
                UnifiedVector<std::byte> ws(
                    syev_blocked_buffer_size<B>(ctx, A.view(), JobType::EigenVectors, Uplo::Lower));
                syev_blocked<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
            });

            run_one("syev_two_stage", [&](Matrix<double>& A, UnifiedVector<double>& w) {
                UnifiedVector<std::byte> ws(
                    syev_two_stage_buffer_size<B>(ctx, A.view(), JobType::EigenVectors, Uplo::Lower));
                syev_two_stage<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
            });
        }
    }

    // -----------------------------------------------------------------------
    // CPU versus GPU
    //
    // The backend is fixed at compile time, so this cannot switch mid-run.
    // What it can do is report which side of the choice you are on and what
    // this device manages — run the example again with `cpu` for the other
    // half of the comparison.
    // -----------------------------------------------------------------------
    static void device_section(Queue& ctx) {
        section("CPU versus GPU");

        report("device", ctx.device().get_name());
        report("compute units", ctx.device().get_property(DeviceProperty::MAX_COMPUTE_UNITS));
        report("max work-group size", ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));

        const int n = 32, batch = 256;
        const double ms = time_syev(ctx, n, batch);
        report(std::to_string(batch) + " eigenproblems of size " + std::to_string(n),
               fixed(ms) + " ms (" + fixed(ms * 1000.0 / batch, 1) + " us each)");
        report_skip("the other device", "run this example again with the `cpu` argument to compare");
    }

    // -----------------------------------------------------------------------
    // Two easy savings
    //
    // Ask for eigenvalues only when you do not need vectors, and reuse a
    // workspace across calls instead of sizing and allocating each time.
    // -----------------------------------------------------------------------
    static void savings_section(Queue& ctx) {
        section("Two easy savings");

        const int n = 64, batch = 64;

        // 1. Skip the back-transform.
        auto A = Matrix<double>::Random(n, n, true, batch, 42);
        UnifiedVector<double> w(static_cast<size_t>(n) * batch);
        UnifiedVector<std::byte> ws_v(
            syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower));
        UnifiedVector<std::byte> ws_n(
            syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower));
        ctx.wait();

        const double with_vectors = time_ms(ctx, [&] {
            syev<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws_v.to_span());
        }, 3);
        const double without = time_ms(ctx, [&] {
            syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws_n.to_span());
        }, 3);

        report("with eigenvectors", fixed(with_vectors) + " ms");
        report("eigenvalues only", fixed(without) + " ms");
        report("saved", fixed(100.0 * (1.0 - without / std::max(with_vectors, 1e-12)), 0) + "%");
        report("workspace with vectors", ws_v.size());
        report("workspace without", ws_n.size());

        // 2. Reuse the workspace. Sizing and allocating inside a loop pays for
        // a device allocation every iteration.
        const int reps = 20;
        const double reused = time_ms(ctx, [&] {
            for (int i = 0; i < reps; ++i)
                syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws_n.to_span());
        }, 3);
        const double reallocated = time_ms(ctx, [&] {
            for (int i = 0; i < reps; ++i) {
                UnifiedVector<std::byte> tmp(
                    syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower));
                syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, tmp.to_span());
            }
        }, 3);

        report(std::to_string(reps) + " calls, workspace reused", fixed(reused) + " ms");
        report(std::to_string(reps) + " calls, workspace reallocated each time", fixed(reallocated) + " ms");
        report_check("reusing the workspace is not slower", reused <= reallocated * 1.05);
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("12. Choosing a variant")
