// 6. Symmetric and Hermitian eigensolvers
//
// The whole syev family: the dispatching driver, each variant it can pick, the
// parameter structs that tune them, complex Hermitian input, and what `uplo`
// actually promises.
//
// They all share a contract: A is overwritten with the eigenvectors (when
// asked for), and the eigenvalues come back ascending in a span of REAL
// scalars — `float_t<T>`, not T — of length n per batch item.

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

template <Backend B>
struct Example {
    static void run(Queue& ctx) {
        support_section(ctx);
        syev_section(ctx);
        values_only_section(ctx);
        small_variants_section(ctx);
        large_variants_section(ctx);
        tuning_section(ctx);
        hermitian_section(ctx);
        uplo_section(ctx);
    }

    // Which variants does this device support?
    //
    // There is no runtime query in the C++ API (the Python facade's
    // `syev_variant_support` has no counterpart). Ask the device instead: the
    // CTA variants put one work-group on one matrix and need a sub-group width
    // of 32, so they are GPU-only and capped at n <= 32.
    static void support_section(Queue& ctx) {
        section("Which variants does this device support?");

        print("device type", on_gpu(ctx) ? "gpu" : "cpu");
        print("max sub-group size", ctx.device().get_property(DeviceProperty::MAX_SUB_GROUP_SIZE));
        print("CTA variants usable here", supports_cta(ctx));
        print("blocked / two-stage", std::string(on_gpu(ctx) ? "usable (GPU, Uplo::Lower only)" : "GPU only"));
    }

    // syev — the dispatching driver.
    //
    // Picks a provider based on the device and the problem: a CTA variant for
    // tiny matrices on a GPU, the blocked or two-stage path for larger ones,
    // the vendor library otherwise. The BATCHLAS_SYEV_* environment variables
    // override the choice.
    static void syev_section(Queue& ctx) {
        section("syev - eigenvalues and eigenvectors");

        const int n = 8;
        auto A = Matrix<double>::Random(n, n, /*hermitian=*/true, kBatch, 11);
        UnifiedVector<double> w(static_cast<size_t>(n) * kBatch);

        UnifiedVector<std::byte> ws(
            syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower));
        syev<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
        ctx.wait();

        print_values("eigenvalues (ascending)", w.to_span(), n);
        std::cout << "eigenvectors, now in A, item 0:\n";
        A.view()[0].print(std::cout, 8, 8);
    }

    // Eigenvalues only
    //
    // JobType::NoEigenVectors skips the back-transform, most of the work for a
    // large matrix. Whether A survives depends on the variant — syev_cta_fused
    // and syev_jacobi_cta leave it alone, the others do not — so do not rely on
    // A afterwards.
    static void values_only_section(Queue& ctx) {
        section("Eigenvalues only");

        const int n = 8;
        auto A = Matrix<double>::Random(n, n, true, kBatch, 11);
        UnifiedVector<double> w(static_cast<size_t>(n) * kBatch);
        UnifiedVector<std::byte> ws(
            syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower));
        syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws.to_span());
        ctx.wait();

        print_values("same eigenvalues, no vectors", w.to_span(), n);
    }

    // The small-matrix variants
    //
    // Three ways to solve an n <= 32 problem with one work-group per matrix:
    //
    //   syev_cta        tridiagonalise, QL/QR, back-transform
    //   syev_cta_fused  the same three stages in one kernel launch
    //   syev_jacobi_cta two-sided Jacobi; slower, but see example 10
    //
    // The fused and Jacobi ones need no global workspace and ignore the
    // argument entirely.
    static void small_variants_section(Queue& ctx) {
        section("The small-matrix variants");

        if constexpr (!has_cta_variants<B>) {
            skip("CTA variants", "not instantiated for this backend");
            return;
        } else {
            if (!supports_cta(ctx)) {
                skip("CTA variants", "needs a GPU with sub-group width 32");
                return;
            }

            const int n = 8;
            auto solve = [&](auto&& call, const char* label) {
                auto A = Matrix<double>::Random(n, n, true, kBatch, 21);
                UnifiedVector<double> w(static_cast<size_t>(n) * kBatch);
                call(A, w);
                ctx.wait();
                print_values(label, w.to_span(), n);
            };

            solve(
                [&](auto& A, auto& w) {
                    UnifiedVector<std::byte> ws(syev_cta_buffer_size<B>(ctx, A.view(), JobType::EigenVectors));
                    syev_cta<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
                },
                "syev_cta");

            solve([&](auto& A,
                      auto& w) { syev_cta_fused<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower); },
                  "syev_cta_fused");

            solve(
                [&](auto& A, auto& w) {
                    syev_jacobi_cta<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower);
                },
                "syev_jacobi_cta");
        }
    }

    // The medium and large variants
    //
    //   syev_blocked    sytrd_blocked -> stedc -> ormqr_blocked
    //   syev_two_stage  dense -> band -> tridiagonal -> stedc
    //
    // Both are GPU-only and accept Uplo::Lower only. Example 07 takes the
    // reduction stages apart; example 12 measures which one to reach for.
    static void large_variants_section(Queue& ctx) {
        section("The medium and large variants");

        if (!on_gpu(ctx)) {
            skip("syev_blocked / syev_two_stage", "GPU only");
            return;
        }

        const int n = 48;
        auto solve = [&](auto&& call, const char* label) {
            auto A = Matrix<double>::Random(n, n, true, kBatch, 31);
            UnifiedVector<double> w(static_cast<size_t>(n) * kBatch);
            call(A, w);
            ctx.wait();
            print_values(label, w.to_span(), 6);
        };

        solve(
            [&](auto& A, auto& w) {
                UnifiedVector<std::byte> ws(
                    syev_blocked_buffer_size<B>(ctx, A.view(), JobType::EigenVectors, Uplo::Lower));
                syev_blocked<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
            },
            "syev_blocked, smallest 6");

        // Ask for vectors here even if you only want values: syev_two_stage
        // returns wrong eigenvalues in values-only mode for n >= 32. See the
        // known issues in the README.
        solve(
            [&](auto& A, auto& w) {
                UnifiedVector<std::byte> ws(
                    syev_two_stage_buffer_size<B>(ctx, A.view(), JobType::EigenVectors, Uplo::Lower));
                syev_two_stage<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
            },
            "syev_two_stage, smallest 6");
    }

    // Tuning a variant through its parameter struct
    //
    // Where the Python facade takes dataclasses, C++ takes plain structs with
    // defaulted members: SteqrParams for the QR iteration, StedcParams for
    // divide and conquer, JacobiParams for the Jacobi sweep. Pass one by value
    // and it applies to that call only. The *buffer size* can depend on the
    // parameters, so pass the same struct to both calls.
    static void tuning_section(Queue& ctx) {
        section("Tuning a variant through its parameter struct");

        const int n = 8;

        if constexpr (has_cta_variants<B>) {
            if (supports_cta(ctx)) {
                SteqrParams<double> params;
                params.sort_order = SortOrder::Descending;
                params.max_sweeps = 100;

                auto A = Matrix<double>::Random(n, n, true, kBatch, 41);
                UnifiedVector<double> w(static_cast<size_t>(n) * kBatch);
                UnifiedVector<std::byte> ws(syev_cta_buffer_size<B>(ctx, A.view(), JobType::EigenVectors, params));
                syev_cta<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span(), params);
                ctx.wait();
                print_values("SteqrParams::sort_order = Descending", w.to_span(), n);

                JacobiParams<double> jp;
                jp.sort_order = SortOrder::Descending;
                jp.max_sweeps = 40;
                auto A2 = Matrix<double>::Random(n, n, true, kBatch, 41);
                UnifiedVector<double> w2(static_cast<size_t>(n) * kBatch);
                syev_jacobi_cta<B>(ctx, A2.view(), w2.to_span(), JobType::EigenVectors, Uplo::Lower,
                                   Span<std::byte>(), jp);
                ctx.wait();
                print_values("JacobiParams::sort_order = Descending", w2.to_span(), n);
            } else {
                skip("SteqrParams / JacobiParams", "needs a GPU with sub-group width 32");
            }
        } else {
            skip("SteqrParams / JacobiParams", "CTA variants not instantiated for this backend");
        }

        // StedcParams tunes the divide-and-conquer tridiagonal solve that
        // syev_blocked runs underneath.
        if (on_gpu(ctx)) {
            StedcParams<double> sp;
            sp.recursion_threshold = 8;  // switch to the leaf solver sooner

            const int n2 = 32;
            auto A = Matrix<double>::Random(n2, n2, true, kBatch, 51);
            UnifiedVector<double> w(static_cast<size_t>(n2) * kBatch);
            UnifiedVector<std::byte> ws(
                syev_blocked_buffer_size<B>(ctx, A.view(), JobType::EigenVectors, Uplo::Lower, sp));
            syev_blocked<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span(), sp);
            ctx.wait();
            print_values("StedcParams::recursion_threshold = 8", w.to_span(), 6);
        } else {
            skip("StedcParams", "syev_blocked is GPU only");
        }
    }

    // Hermitian (complex) input
    //
    // Same calls, T = std::complex<double>. The eigenvalues stay real, so the
    // span is `double` while the matrix is complex — that asymmetry is what
    // `Span<typename base_type<T>::type>` means in the signatures.
    static void hermitian_section(Queue& ctx) {
        section("Hermitian (complex) input");

        using C64 = std::complex<double>;
        const int n = 6;
        auto A = Matrix<C64>::Random(n, n, /*hermitian=*/true, kBatch, 61);
        UnifiedVector<double> w(static_cast<size_t>(n) * kBatch);  // real

        UnifiedVector<std::byte> ws(
            syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower));
        syev<B>(ctx, A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
        ctx.wait();

        print_values("Hermitian eigenvalues (real)", w.to_span(), n);
        std::cout << "eigenvectors are complex, item 0:\n";
        A.view()[0].print(std::cout, 3, 3);
    }

    // uplo — which triangle holds your data
    //
    // `uplo` says which triangle the solver may read; the other is ignored and
    // may hold anything. Passing a full symmetric matrix is therefore always
    // safe, and is what to do if you are unsure.
    //
    // Filling only one triangle is where backends differ: on CUDA, Uplo::Upper
    // with only the upper triangle populated gives the wrong answer, while
    // Uplo::Lower with only the lower triangle is fine. See the README.
    static void uplo_section(Queue& ctx) {
        section("uplo - which triangle holds your data");

        const int n = 6;
        auto solve = [&](Uplo uplo, bool zero_other_triangle, const char* label) {
            auto A = Matrix<double>::Random(n, n, true, 1, 71);
            if (zero_other_triangle) {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i) {
                        const bool keep = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                        if (!keep) A(i, j, 0) = 0.0;
                    }
            }
            UnifiedVector<double> w(n);
            UnifiedVector<std::byte> ws(
                syev_buffer_size<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, uplo));
            syev<B>(ctx, A.view(), w.to_span(), JobType::NoEigenVectors, uplo, ws.to_span());
            ctx.wait();
            print_values(label, w.to_span(), n);
        };

        solve(Uplo::Lower, false, "full matrix, Uplo::Lower");
        solve(Uplo::Upper, false, "full matrix, Uplo::Upper (same answer)");
        solve(Uplo::Lower, true, "lower triangle only, Uplo::Lower (same answer)");
        solve(Uplo::Upper, true, "upper triangle only, Uplo::Upper (wrong on CUDA)");
    }
};

}  // namespace

BATCHLAS_EXAMPLE_MAIN("6. Symmetric and Hermitian eigensolvers")
