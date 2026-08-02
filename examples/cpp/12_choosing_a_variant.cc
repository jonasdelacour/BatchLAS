// 12. Choosing a variant
//
// The measurement example. What batching actually buys, how throughput scales
// with batch size, which syev variant wins at which size, and two easy savings.
//
// Timings are wall-clock around a queue that is waited on, so they include
// launch overhead — which is the point: for small matrices that overhead *is*
// the cost, and amortising it across a batch is the whole reason this library
// exists. The numbers come from whatever machine you run on.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <blas/linalg.hh>
#include <util/sycl-vector.hh>

#include "example_utils.hh"

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
        samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

std::string ms(double v, int places = 2) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(places) << v << " ms";
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

    // One launch beats a loop
    //
    // The same work, submitted as one batched call versus one call per matrix.
    // For small matrices the per-launch overhead dominates completely — on a
    // CPU device, where that overhead is small, expect a modest gap.
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

        auto One = Matrix<double>::Random(n, n, true, 1, 42);
        UnifiedVector<double> w1(n);
        UnifiedVector<std::byte> ws1(
            syev_buffer_size<B>(ctx, One.view(), w1.to_span(), JobType::NoEigenVectors, Uplo::Lower));
        ctx.wait();
        const double looped = time_ms(
            ctx,
            [&] {
                for (int i = 0; i < batch; ++i)
                    syev<B>(ctx, One.view(), w1.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws1.to_span());
            },
            3);

        print(std::to_string(batch) + " matrices of size " + std::to_string(n) + ", batched", ms(batched));
        print(std::to_string(batch) + " matrices of size " + std::to_string(n) + ", one at a time", ms(looped));
        std::ostringstream os;
        os << std::fixed << std::setprecision(1) << looped / std::max(batched, 1e-9) << "x";
        print("speed-up", os.str());
    }

    // How throughput scales with batch size
    //
    // Time per matrix falls as the batch grows, until the device is saturated
    // and it flattens out. Where that happens is what you want to know when
    // sizing work.
    static void scaling_section(Queue& ctx) {
        section("How throughput scales with batch size");

        const int n = 32;
        for (int batch : {1, 8, 64, 512}) {
            auto A = Matrix<double>::Random(n, n, true, batch, 42);
            UnifiedVector<double> w(static_cast<size_t>(n) * batch);
            UnifiedVector<std::byte> ws(
                syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower));
            ctx.wait();
            const double t = time_ms(ctx, [&] {
                syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws.to_span());
            });
            std::ostringstream os;
            os << ms(t) << " total, " << std::fixed << std::setprecision(1) << t * 1000.0 / batch
               << " us per matrix";
            print("batch " + std::to_string(batch), os.str());
        }
    }

    // Which syev variant wins at which size?
    //
    // syev picks for you. These are the variants it picks between, timed
    // directly so you can see the crossovers.
    //
    // Run with JobType::EigenVectors: syev_two_stage returns wrong eigenvalues
    // in values-only mode for n >= 32 (see the README), so a values-only
    // comparison would be timing a broken path.
    //
    // These routines overwrite A, so repeated timing runs operate on the
    // previous result — the same amount of work on different data, which is
    // what we are measuring.
    static void variant_section(Queue& ctx) {
        section("Which syev variant wins at which size?");

        if (!on_gpu(ctx)) {
            skip("variant comparison", "the alternatives to the vendor path are GPU only");
            return;
        }

        const int batch = 64;
        for (int n : {16, 32, 64, 128}) {
            auto run_one = [&](const char* name, auto&& call) {
                auto A = Matrix<double>::Random(n, n, true, batch, 7);
                UnifiedVector<double> w(static_cast<size_t>(n) * batch);
                ctx.wait();
                const double t = time_ms(ctx, [&] { call(A, w); }, 3);
                print("n=" + std::to_string(n) + " " + name, ms(t));
            };

            run_one("syev (dispatched)", [&](Matrix<double>& A, UnifiedVector<double>& w) {
                UnifiedVector<std::byte> ws(
                    syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower));
                syev<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
            });

            if constexpr (has_cta_variants<B>) {
                if (n <= 32 && supports_cta(ctx)) {
                    run_one("syev_cta         ", [&](Matrix<double>& A, UnifiedVector<double>& w) {
                        UnifiedVector<std::byte> ws(syev_cta_buffer_size<B>(ctx, A.view(), JobType::EigenVectors));
                        syev_cta<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
                    });
                }
            }

            run_one("syev_blocked     ", [&](Matrix<double>& A, UnifiedVector<double>& w) {
                UnifiedVector<std::byte> ws(
                    syev_blocked_buffer_size<B>(ctx, A.view(), JobType::EigenVectors, Uplo::Lower));
                syev_blocked<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
            });

            run_one("syev_two_stage   ", [&](Matrix<double>& A, UnifiedVector<double>& w) {
                UnifiedVector<std::byte> ws(
                    syev_two_stage_buffer_size<B>(ctx, A.view(), JobType::EigenVectors, Uplo::Lower));
                syev_two_stage<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
            });
        }
    }

    // CPU versus GPU
    //
    // The backend is fixed at compile time, so this cannot switch mid-run. What
    // it can do is report what this device manages — run the example again with
    // `cpu` for the other half of the comparison.
    static void device_section(Queue& ctx) {
        section("CPU versus GPU");

        print("device", ctx.device().get_name());
        print("compute units", ctx.device().get_property(DeviceProperty::MAX_COMPUTE_UNITS));
        print("max work-group size", ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));

        const int n = 32, batch = 256;
        auto A = Matrix<double>::Random(n, n, true, batch, 42);
        UnifiedVector<double> w(static_cast<size_t>(n) * batch);
        UnifiedVector<std::byte> ws(
            syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower));
        ctx.wait();
        const double t = time_ms(ctx, [&] {
            syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws.to_span());
        });

        std::ostringstream os;
        os << ms(t) << " (" << std::fixed << std::setprecision(1) << t * 1000.0 / batch << " us each)";
        print(std::to_string(batch) + " eigenproblems of size " + std::to_string(n), os.str());
        std::cout << "run this example again with the `cpu` argument to compare\n";
    }

    // Two easy savings
    //
    // Ask for eigenvalues only when you do not need vectors, and reuse a
    // workspace across calls instead of sizing and allocating each time.
    static void savings_section(Queue& ctx) {
        section("Two easy savings");

        const int n = 64, batch = 64;
        auto A = Matrix<double>::Random(n, n, true, batch, 42);
        UnifiedVector<double> w(static_cast<size_t>(n) * batch);
        UnifiedVector<std::byte> ws_v(
            syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower));
        UnifiedVector<std::byte> ws_n(
            syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower));
        ctx.wait();

        const double with_vectors = time_ms(
            ctx,
            [&] { syev<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws_v.to_span()); }, 3);
        const double without = time_ms(
            ctx,
            [&] { syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws_n.to_span()); }, 3);

        print("with eigenvectors", ms(with_vectors));
        print("eigenvalues only", ms(without));
        print("workspace with vectors", ws_v.size());
        print("workspace without", ws_n.size());

        // Sizing and allocating inside a loop pays for a device allocation
        // every iteration.
        const int reps = 20;
        const double reused = time_ms(
            ctx,
            [&] {
                for (int i = 0; i < reps; ++i)
                    syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws_n.to_span());
            },
            3);
        const double reallocated = time_ms(
            ctx,
            [&] {
                for (int i = 0; i < reps; ++i) {
                    UnifiedVector<std::byte> tmp(
                        syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower));
                    syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, tmp.to_span());
                }
            },
            3);

        print(std::to_string(reps) + " calls, workspace reused", ms(reused));
        print(std::to_string(reps) + " calls, workspace reallocated each time", ms(reallocated));
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("12. Choosing a variant")
