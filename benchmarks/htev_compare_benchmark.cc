// htev_compare_benchmark.cc
//
// Head-to-head benchmark: BatchLAS STEQR and STEDC vs cuSolverDx HTEV
// for batched tridiagonal eigenvalue decomposition.
//
// All three solvers use the same SteqrBenchSizes grid so their output columns
// are directly comparable when plotted together.

#include <batchlas/util/minibench.hh>
#include <batchlas/blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

#if BATCHLAS_HAS_CUDA_BACKEND
#include "../src/backends/cusolverdx.hh"
#endif

using namespace batchlas;

namespace {

template <typename Benchmark>
inline void HtevDxCompareBenchSizes(Benchmark* b) {
    // Include intermediate matrix sizes, not only powers of two.
    for (int n = 8; n <= 64; ++n) {
        for (int bs : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
            b->Args({n, bs});
        }
    }
}

template <typename Benchmark>
inline void HtevDxCompareBenchSizesNetlib(Benchmark* b) {
    HtevDxCompareBenchSizes(b);
}

} // namespace

// ── STEQR ────────────────────────────────────────────────────────────────────

template <typename T, Backend B>
static void BM_HTEV_STEQR(minibench::State& state) {
    const size_t n     = state.range(0);
    const size_t batch = state.range(1);

    JobType jobz = JobType::EigenVectors;

    SteqrParams<T> params;
    params.cta_update_scheme = SteqrUpdateScheme::EXP;
    params.back_transform    = false;

    auto q         = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto diags     = Vector<T>::random(static_cast<int>(n), static_cast<int>(batch));
    auto off_diags = Vector<T>::random(static_cast<int>(n - 1), static_cast<int>(batch));
    auto eigvals   = Vector<T>::zeros(static_cast<int>(n), static_cast<int>(batch));
    auto eigvects  = Matrix<T>::Identity(static_cast<int>(n), static_cast<int>(batch));

    const size_t ws_size = steqr_buffer_size<T>(*q, diags, off_diags, eigvals, jobz, params);
    UnifiedVector<std::byte> ws(ws_size);

    state.SetKernel(q,
                    bench::pristine(diags),
                    bench::pristine(off_diags),
                    std::move(eigvals),
                    std::move(ws),
                    jobz,
                    params,
                    bench::pristine(eigvects),
                    [](Queue& q_ref, auto&&... xs) {
                        steqr(q_ref, std::forward<decltype(xs)>(xs)...);
                    });

    state.SetMetric("Time (µs) / matrix", (1.0 / static_cast<double>(batch)) * 1e6, minibench::Reciprocal);
}

// ── STEDC ────────────────────────────────────────────────────────────────────

template <typename T, Backend B>
static void BM_HTEV_STEDC(minibench::State& state) {
    const size_t n     = state.range(0);
    const size_t batch = state.range(1);

    JobType jobz = JobType::EigenVectors;

    StedcParams<T> params;

    auto q         = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto diags     = Vector<T>::random(static_cast<int>(n), static_cast<int>(batch));
    auto off_diags = Vector<T>::random(static_cast<int>(n - 1), static_cast<int>(batch));
    auto eigvals   = Vector<T>::zeros(static_cast<int>(n), static_cast<int>(batch));
    auto eigvects  = Matrix<T>::Identity(static_cast<int>(n), static_cast<int>(batch));

    UnifiedVector<std::byte> ws(stedc_workspace_size(*q, n, batch, jobz, params));

    // stedc has a non-standard calling convention; use a capturing lambda and
    // omit the leading Queue& parameter (the adapter will call k(xs...) form).
    auto kernel = [q](auto& d_arg, auto& e_arg, auto& w_arg, auto& ws_arg,
                      JobType jz, StedcParams<T> p, auto& z_arg) {
        auto d = static_cast<VectorView<T>>(d_arg);
        auto e = static_cast<VectorView<T>>(e_arg);
        auto w = static_cast<VectorView<T>>(w_arg);
        auto Z = z_arg.view();
        stedc(*q, d, e, w, ws_arg.to_span(), jz, p, Z);
    };

    state.SetKernel(q,
                    bench::pristine(diags),
                    bench::pristine(off_diags),
                    std::move(eigvals),
                    std::move(ws),
                    jobz,
                    params,
                    bench::pristine(eigvects),
                    kernel);

    state.SetMetric("Time (µs) / matrix", (1.0 / static_cast<double>(batch)) * 1e6, minibench::Reciprocal);
}

// ── cuSolverDx HTEV ──────────────────────────────────────────────────────────

template <typename T, Backend B>
static void BM_HTEV_DX(minibench::State& state) {
#if BATCHLAS_HAS_CUDA_BACKEND
    const size_t n     = state.range(0);
    const size_t batch = state.range(1);

    auto q         = std::make_shared<Queue>(Device("gpu"), B);
    auto diags     = Vector<T>::random(static_cast<int>(n), static_cast<int>(batch));
    auto off_diags = Vector<T>::random(static_cast<int>(n - 1), static_cast<int>(batch));
    auto eigvals   = Vector<T>::zeros(static_cast<int>(n), static_cast<int>(batch));
    auto eigvects  = Matrix<T>::Identity(static_cast<int>(n), static_cast<int>(batch));

    const size_t ws_size = backend::cusolverdx::htev_buffer_size<T>(
        *q,
        static_cast<VectorView<T>>(diags),
        static_cast<VectorView<T>>(off_diags),
        JobType::EigenVectors,
        Uplo::Lower);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(diags),
                    bench::pristine(off_diags),
                    std::move(eigvals),
                    JobType::EigenVectors,
                    bench::pristine(eigvects),
                    std::move(workspace),
                    Uplo::Lower,
                    [](Queue& q_ref, auto&&... xs) {
                        backend::cusolverdx::htev<T>(q_ref, std::forward<decltype(xs)>(xs)...);
                    });

    state.SetMetric("Time (µs) / matrix", (1.0 / static_cast<double>(batch)) * 1e6, minibench::Reciprocal);
#else
    static_cast<void>(state);
    static_cast<void>(B);
#endif
}

// ── Registration ─────────────────────────────────────────────────────────────

#if BATCHLAS_HAS_CUDA_BACKEND
BATCHLAS_BENCH_CUDA(BM_HTEV_STEQR, HtevDxCompareBenchSizes)
BATCHLAS_BENCH_CUDA(BM_HTEV_STEDC, HtevDxCompareBenchSizes)
MINI_BENCHMARK_REGISTER_SIZES((BM_HTEV_DX<float,  batchlas::Backend::CUDA>), HtevDxCompareBenchSizes);
MINI_BENCHMARK_REGISTER_SIZES((BM_HTEV_DX<double, batchlas::Backend::CUDA>), HtevDxCompareBenchSizes);
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
BATCHLAS_BENCH_ROCM(BM_HTEV_STEQR, HtevDxCompareBenchSizes)
BATCHLAS_BENCH_ROCM(BM_HTEV_STEDC, HtevDxCompareBenchSizes)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
BATCHLAS_BENCH_NETLIB(BM_HTEV_STEQR, HtevDxCompareBenchSizes)
BATCHLAS_BENCH_NETLIB(BM_HTEV_STEDC, HtevDxCompareBenchSizes)
#endif

MINI_BENCHMARK_MAIN();
