#include <util/minibench.hh>
#include <blas/extensions.hh>

#include "bench_utils.hh"

#include <batchlas/backend_config.h>

using namespace batchlas;

namespace {

template <typename Benchmark>
inline void LatrdLowerPanelBenchSizes(Benchmark* b) {
    for (int n : {64, 128, 256, 512, 1024}) {
        for (int bs : {1, 8, 32, 64}) {
            for (int ib : {8, 16, 32, 64}) {
                // j0: benchmark the first panel by default.
                b->Args({n, bs, ib, 0});
            }
        }
    }
}

} // namespace

template <typename T, Backend B>
static void BM_LATRD_LOWER_PANEL(minibench::State& state) {
#if BATCHLAS_HAS_CUDA_BACKEND
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const int ib = static_cast<int>(state.range(2));
    const int j0 = static_cast<int>(state.range(3));

    // Approximate: panel work dominated by matvecs over the trailing matrix.
    // Use a coarse model O(n^2 * ib).
    const double approx_flops = 2.0 * double(n) * double(n) * double(ib) * double(batch);

    auto A0 = Matrix<T>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/2026);

    auto e0 = Vector<T>::zeros(n - 1, batch);
    auto tau0 = Vector<T>::zeros(n - 1, batch);
    auto W0 = Matrix<T>::Zeros(n, std::max<int>(1, ib), batch);

    auto q = std::make_shared<Queue>("gpu", /*in_order=*/true);

    state.SetKernel(
        q,
        bench::pristine(A0),
        bench::pristine(e0),
        bench::pristine(tau0),
        bench::pristine(W0),
        j0,
        ib,
        [](Queue& q,
           MatrixView<T, MatrixFormat::Dense> A,
           VectorView<T> e,
           VectorView<T> tau,
           MatrixView<T, MatrixFormat::Dense> W,
           int j0,
           int ib) {
            auto A_panel = A({j0, SliceEnd()}, {j0, SliceEnd()});
            auto e_panel = e(Slice(j0, j0 + ib));
            auto tau_panel = tau(Slice(j0, j0 + ib));
            auto W_panel = W({j0, SliceEnd()}, {0, ib});
            latrd_lower_panel<B, T>(q, A_panel, e_panel, tau_panel, W_panel);
        });

    state.SetMetric("GFLOPS", approx_flops * 1e-9, minibench::Rate);
    state.SetMetric("T(µs)/matrix", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
#else
    (void)state;
#endif
}

BATCHLAS_BENCH_CUDA(BM_LATRD_LOWER_PANEL, LatrdLowerPanelBenchSizes);

MINI_BENCHMARK_MAIN();
