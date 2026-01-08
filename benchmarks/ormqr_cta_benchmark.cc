#include <util/minibench.hh>
#include <blas/linalg.hh>
#include <blas/extensions.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

namespace {

template <typename Benchmark>
inline void OrmqrCtaBenchSizes(Benchmark* b) {
    for (int n : {8, 16, 32}) {
        for (int bs : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
            for (int wg_mult : {1, 2, 4, 8}) {
                for (int side : {0, 1}) {
                    for (int trans : {0, 1}) {
                        b->Args({n, bs, wg_mult, side, trans});
                    }
                }
            }
        }
    }
}

} // namespace

// Batched ORMQR-CTA benchmark
template <typename T, Backend B>
static void BM_ORMQR_CTA(minibench::State& state) {
#if BATCHLAS_HAS_CUDA_BACKEND
    const int n = static_cast<int>(state.range(0));
    const int batch = static_cast<int>(state.range(1));
    const size_t wg_mult = state.range(2) > 0 ? state.range(2) : 1;
    const Side side = (state.range(3) == 0) ? Side::Left : Side::Right;
    const Transpose trans = (state.range(4) != 0) ? Transpose::Trans : Transpose::NoTrans;

    // Approx flop model: applying k=n Householder reflectors to an n×n C.
    // Each reflector costs ~2n^2 flops; total ~2n^3 (order-of-magnitude model).
    const double total_flops = 2.0 * double(n) * double(n) * double(n) * double(batch);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    // Build reflectors via QR once (outside timed region) using the same backend.
    // This avoids an unconditional dependency on the NETLIB/CPU backend.
    auto A = Matrix<T>::Random(n, n, /*hermitian=*/false, batch, /*seed=*/2025);
    // NOTE: minibench executes the SetKernel callback after this function returns.
    // So we must capture *owning* storage (not just a VectorView) into the callback.
    UnifiedVector<T> tau_storage(static_cast<size_t>(n) * static_cast<size_t>(batch));
    size_t geqrf_ws = geqrf_buffer_size<B>(*q, A.view(), tau_storage.to_span());
    UnifiedVector<std::byte> ws_geqrf(geqrf_ws);

    geqrf<B>(*q, A.view(), tau_storage.to_span(), ws_geqrf.to_span()).wait();
    q->wait();

    auto C = Matrix<T>::Random(n, n, /*hermitian=*/false, batch, /*seed=*/1337);
    UnifiedVector<std::byte> ws_dummy(1, std::byte{0});

    state.SetKernel(q,
                    std::move(A),
                    bench::pristine(std::move(C)),
                    std::move(tau_storage),
                    std::move(ws_dummy),
                    Uplo::Upper,
                    side,
                    trans,
                    n,
                    batch,
                    wg_mult,
                    [](Queue& q,
                       auto&& A,
                       auto&& C,
                       auto&& tau_storage,
                       auto&& ws_dummy,
                       auto factorization,
                       auto side,
                       auto trans,
                       auto n,
                       auto batch,
                       auto wg_mult) {
                        VectorView<T> tau_view_local(tau_storage,
                                                     static_cast<int>(n),
                                                     static_cast<int>(batch));
                        ormqx_cta<B>(q,
                                    A,
                                    tau_view_local,
                                    C,
                                    factorization,
                                    side,
                                    trans,
                                    /*k=*/static_cast<int32_t>(n),
                                    ws_dummy,
                                    wg_mult);
                    });
    state.SetMetric("GFLOPS", total_flops * 1e-9, minibench::Rate);
    state.SetMetric("T(µs)/Batch", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
#else
    (void)state;
#endif
}

// Register size/batch combinations at static-init time
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_ORMQR_CTA, OrmqrCtaBenchSizes);

MINI_BENCHMARK_MAIN();
