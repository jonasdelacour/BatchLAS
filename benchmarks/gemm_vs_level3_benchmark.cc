// Head-to-head: the GEMM spellings that src/extensions actually uses, against
// the specialised level-3 op that expresses the same thing.
//
// Every shape here is lifted from a real call site, not invented:
//
//   Gram      C = A^H A, k x k, from ortho.cc's chol/shift-chol/svqb paths.
//             gemm(A, A, C, transA=inv_trans) vs syrk/herk(A, C, uplo=Lower).
//             potrf and syev both default to Uplo::Lower, so a one-triangle
//             syrk is what the consumer reads.
//
//   Trailing  A22 -= V W^H + W V^H, from sytrd_blocked.cc's blocked update.
//             two gemms vs one syr2k. The syr2k writes one triangle, so the
//             honest comparison charges it the symmetrize the non-legacy path
//             needs afterwards; both variants are measured.
//
//   TW        W2 = T^H W1 with T upper triangular, from ormqr_blocked.cc and
//             ormbr.cc's WY update. gemm reads T's zero half; trmm does not.
//
// Run one family at a time with --name Gram / Trailing / TW.

#include <util/minibench.hh>
#include <util/bench_structured.hh>

#include <blas/linalg.hh>

#include "bench_utils.hh"

#include <batchlas/backend_config.h>

#include <memory>

using namespace batchlas;

namespace {

// ---------------------------------------------------------------- Gram sizes
// (m, k, batch). k <= m always; ortho asserts it.
inline void GramSizes(minibench::Benchmark* b) {
    // LOBPCG/syevx shape: tall and very skinny (k = a few block vectors).
    b->Args({256, 32, 2048});
    b->Args({512, 32, 2048});
    b->Args({512, 64, 1024});
    b->Args({1024, 64, 1024});
    b->Args({1024, 128, 512});
    b->Args({2048, 128, 256});
    // Squarer shapes, from ortho_benchmark's own grid.
    b->Args({256, 256, 512});
    b->Args({512, 512, 128});
    b->Args({1024, 1024, 32});
}
inline void GramSizesNetlib(minibench::Benchmark* b) { GramSizes(b); }

// ------------------------------------------------------------ Trailing sizes
// (n2, ib, batch). A22 is n2 x n2; V and W are n2 x ib.
inline void TrailingSizes(minibench::Benchmark* b) {
    b->Args({256, 32, 1024});
    b->Args({256, 64, 1024});
    b->Args({512, 32, 512});
    b->Args({512, 64, 512});
    b->Args({1024, 32, 128});
    b->Args({1024, 64, 128});
    b->Args({2048, 64, 32});
}
inline void TrailingSizesNetlib(minibench::Benchmark* b) { TrailingSizes(b); }

// ------------------------------------------------------------------ TW sizes
// (ib, nC, batch). T is ib x ib upper triangular; W1 is ib x nC.
inline void TWSizes(minibench::Benchmark* b) {
    b->Args({32, 256, 2048});
    b->Args({64, 256, 2048});
    b->Args({64, 512, 1024});
    b->Args({128, 512, 1024});
    b->Args({128, 1024, 512});
    b->Args({256, 1024, 256});
}
inline void TWSizesNetlib(minibench::Benchmark* b) { TWSizes(b); }

// ------------------------------------------------------------- Square sizes
// (m, nC, batch), m the triangular side. The TW shapes above are the regime
// where trmm cannot win and the roofline says so: m is in the tens, so B and C
// dominate the traffic, the GEMM being replaced is already bandwidth bound at
// ~800 GB/s, and halving its arithmetic buys nothing it can spend. Halving the
// arithmetic only pays once the GEMM is compute bound, which needs m past
// ~160 -- intensity is m/4 flop per byte against a ridge near 40. These are
// those shapes, at batches that still saturate.
inline void SquareSizes(minibench::Benchmark* b) {
    b->Args({256, 256, 512});
    b->Args({512, 512, 128});
    b->Args({512, 1024, 64});
    b->Args({1024, 1024, 32});
}
inline void SquareSizesNetlib(minibench::Benchmark* b) { SquareSizes(b); }

template <typename T>
constexpr Transpose conj_trans_for() {
    return sycl::detail::is_complex<T>::value ? Transpose::ConjTrans : Transpose::Trans;
}

// Wires a prepared kernel into the state with event timing, the way
// gemm_steady_benchmark does.
template <typename F>
void install(minibench::State& state,
             std::shared_ptr<Queue> q,
             bench::ManagedInputs& managed,
             F kernel_once,
             double flops,
             size_t batch) {
    state.SetPrepare(managed.make_prepare_once());
    state.SetBeforeEachRun(managed.make_before_each_run());
    state.SetKernel(std::function<void()>(kernel_once));
    state.SetTimedKernelMs(bench::make_event_timed_kernel_ms(q, std::move(kernel_once)));
    state.SetBatchEndWait(q);
    state.SetMetric("GFLOPS", static_cast<double>(batch) * 1e-9 * flops, minibench::Rate);
    state.SetMetric("Time (µs) / matrix",
                    (1.0 / static_cast<double>(batch)) * 1e6,
                    minibench::Reciprocal);
}

// ============================================================== Gram: C = A^H A

template <typename T, Backend B, bool UseSpecialised>
void configure_gram(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t k = state.range(1);
    const size_t batch = state.range(2);

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = std::make_shared<Matrix<T>>(Matrix<T>::Random(m, k, false, batch));
    auto C = std::make_shared<Matrix<T>>(Matrix<T>::Random(k, k, false, batch));

    bench::ManagedInputs managed(q);
    managed.prepare(*A);
    managed.pristine(C);

    // Both spellings charge the same k*k*m multiply-adds; syrk is allowed to do
    // half of them, which is exactly what we are here to find out.
    const double flops = 2.0 * static_cast<double>(k) * static_cast<double>(k) * static_cast<double>(m);

    if constexpr (UseSpecialised) {
        auto kernel = [q, A, C]() mutable {
            if constexpr (sycl::detail::is_complex<T>::value) {
                using Real = typename T::value_type;
                herk<B, T>(*q, A->view(), C->view(), Real(1), Real(0),
                           Uplo::Lower, Transpose::ConjTrans);
            } else {
                syrk<B, T>(*q, A->view(), C->view(), T(1), T(0),
                           Uplo::Lower, Transpose::Trans);
            }
        };
        install(state, q, managed, std::move(kernel), flops, batch);
    } else {
        auto kernel = [q, A, C]() mutable {
            gemm<B>(*q, A->view(), A->view(), C->view(),
                    {.alpha = T(1), .beta = T(0),
                     .transA = conj_trans_for<T>(), .transB = Transpose::NoTrans});
        };
        install(state, q, managed, std::move(kernel), flops, batch);
    }
}

template <typename T, Backend B>
static void BM_Gram_gemm(minibench::State& state) { configure_gram<T, B, false>(state); }
template <typename T, Backend B>
static void BM_Gram_syrk(minibench::State& state) { configure_gram<T, B, true>(state); }

// ================================== Trailing: A22 -= V W^H + W V^H  (sytrd)

// 0 = two gemms, 1 = syr2k only, 2 = syr2k + symmetrize.
template <typename T, Backend B, int Variant>
void configure_trailing(minibench::State& state) {
    const size_t n2 = state.range(0);
    const size_t ib = state.range(1);
    const size_t batch = state.range(2);

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto V = std::make_shared<Matrix<T>>(Matrix<T>::Random(n2, ib, false, batch));
    auto W = std::make_shared<Matrix<T>>(Matrix<T>::Random(n2, ib, false, batch));
    auto A22 = std::make_shared<Matrix<T>>(Matrix<T>::Random(n2, n2, false, batch));

    bench::ManagedInputs managed(q);
    managed.prepare(*V);
    managed.prepare(*W);
    managed.pristine(A22);

    // Two rank-ib updates of an n2 x n2 block.
    const double flops = 2.0 * 2.0 * static_cast<double>(n2) * static_cast<double>(n2)
                         * static_cast<double>(ib);

    if constexpr (Variant == 0) {
        auto kernel = [q, V, W, A22]() mutable {
            gemm<B>(*q, V->view(), W->view(), A22->view(),
                    {.alpha = T(-1), .beta = T(1), .transB = Transpose::ConjTrans});
            gemm<B>(*q, W->view(), V->view(), A22->view(),
                    {.alpha = T(-1), .beta = T(1), .transB = Transpose::ConjTrans});
        };
        install(state, q, managed, std::move(kernel), flops, batch);
    } else {
        auto kernel = [q, V, W, A22]() mutable {
            syr2k<B, T>(*q, V->view(), W->view(), A22->view(), T(-1), T(1),
                        Uplo::Lower, Transpose::NoTrans);
            if constexpr (Variant == 2) {
                A22->view().symmetrize(*q, Uplo::Lower);
            }
        };
        install(state, q, managed, std::move(kernel), flops, batch);
    }
}

template <typename T, Backend B>
static void BM_Trailing_gemm2(minibench::State& state) { configure_trailing<T, B, 0>(state); }
template <typename T, Backend B>
static void BM_Trailing_syr2k(minibench::State& state) { configure_trailing<T, B, 1>(state); }
template <typename T, Backend B>
static void BM_Trailing_syr2k_sym(minibench::State& state) { configure_trailing<T, B, 2>(state); }

// ============================ TW: W2 = T^H W1, T upper triangular (ormqr/ormbr)

template <typename T, Backend B, bool UseTrmm>
void configure_tw(minibench::State& state) {
    const size_t ib = state.range(0);
    const size_t nC = state.range(1);
    const size_t batch = state.range(2);

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto Tm = std::make_shared<Matrix<T>>(Matrix<T>::Random(ib, ib, false, batch));
    auto W1 = std::make_shared<Matrix<T>>(Matrix<T>::Random(ib, nC, false, batch));
    auto W2 = std::make_shared<Matrix<T>>(Matrix<T>::Random(ib, nC, false, batch));

    bench::ManagedInputs managed(q);
    managed.prepare(*Tm);
    managed.prepare(*W1);
    managed.pristine(W2);

    const double flops = 2.0 * static_cast<double>(ib) * static_cast<double>(ib)
                         * static_cast<double>(nC);

    if constexpr (UseTrmm) {
        // Left side, upper triangle, non-unit diagonal: larft's T exactly.
        //
        // The CUDA and ROCm paths overwrite C (beta = 0); only the MKL fallback
        // in src/extensions accumulates into it. PR #61 charged trmm a zeroing
        // pass here to be safe against that, which on a GPU is pure overhead --
        // at ib = 32, nC = 256, batch 2048 the fill alone is ~70 us of a 189 us
        // measurement, and it inflated every trmm row in that report. The GPU
        // spelling is what these shapes are here to measure, so it is gone.
        auto kernel = [q, Tm, W1, W2]() mutable {
            trmm<B, T>(*q, Tm->view(), W1->view(), W2->view(), T(1),
                       Side::Left, Uplo::Upper, conj_trans_for<T>(), Diag::NonUnit);
        };
        install(state, q, managed, std::move(kernel), flops, batch);
    } else {
        auto kernel = [q, Tm, W1, W2]() mutable {
            gemm<B>(*q, Tm->view(), W1->view(), W2->view(),
                    {.alpha = T(1), .beta = T(0), .transA = conj_trans_for<T>()});
        };
        install(state, q, managed, std::move(kernel), flops, batch);
    }
}

template <typename T, Backend B>
static void BM_TW_gemm(minibench::State& state) { configure_tw<T, B, false>(state); }
template <typename T, Backend B>
static void BM_TW_trmm(minibench::State& state) { configure_tw<T, B, true>(state); }

// Same operation, same code, shapes chosen from the other side of the ridge.
template <typename T, Backend B>
static void BM_Square_gemm(minibench::State& state) { configure_tw<T, B, false>(state); }
template <typename T, Backend B>
static void BM_Square_trmm(minibench::State& state) { configure_tw<T, B, true>(state); }

}  // namespace

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_Gram_gemm, GramSizes)
BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_Gram_syrk, GramSizes)

BATCHLAS_REGISTER_BENCHMARK(BM_Trailing_gemm2, TrailingSizes)
BATCHLAS_REGISTER_BENCHMARK(BM_Trailing_syr2k, TrailingSizes)
BATCHLAS_REGISTER_BENCHMARK(BM_Trailing_syr2k_sym, TrailingSizes)

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_TW_gemm, TWSizes)
BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_TW_trmm, TWSizes)

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_Square_gemm, SquareSizes)
BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_Square_trmm, SquareSizes)

MINI_BENCHMARK_MAIN();
