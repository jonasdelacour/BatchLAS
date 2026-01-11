#include <util/minibench.hh>
#include <blas/linalg.hh>
#include <internal/ormqr_blocked.hh>

#include "bench_utils.hh"
#include <batchlas/backend_config.h>

#include <algorithm>
#include <cstdint>
#include <memory>

using namespace batchlas;

namespace {

inline Side parse_side(int v) {
    return (v == 0) ? Side::Left : Side::Right;
}

inline Transpose parse_transpose(int v) {
    // 0 -> NoTrans, otherwise -> ConjTrans (for real this behaves like Trans).
    return (v == 0) ? Transpose::NoTrans : Transpose::ConjTrans;
}

template <typename Benchmark>
inline void OrmqrBlockedBenchSizes(Benchmark* b) {
    // Args: n, batch, side, trans, block_size
    for (int n : {64, 128, 256, 512, 1024}) {
        for (int bs : {1, 2, 4, 8, 16, 32, 64, 128}) {
            for (int side : {0, 1}) {
                for (int trans : {0, 1}) {
                    for (int nb : {16, 32, 64}) {
                        b->Args({n, bs, side, trans, nb});
                    }
                }
            }
        }
    }
}

template <typename Benchmark>
inline void OrmqrBlockedBenchSizesNetlib(Benchmark* b) {
    // Reduced set for CPU runs.
    for (int n : {32, 64, 128, 256}) {
        for (int bs : {1, 10, 100}) {
            for (int side : {0, 1}) {
                for (int trans : {0, 1}) {
                    for (int nb : {16, 32, 64}) {
                        b->Args({n, bs, side, trans, nb});
                    }
                }
            }
        }
    }
}

} // namespace

// Batched blocked ORMQR (LARFT+GEMM WY) benchmark.
//
// This benchmarks the blocked implementation directly (ormqr_blocked) while still
// supporting batched inputs by looping over batch items (matching backend behavior).
template <typename T, Backend B>
static void BM_ORMQR_BLOCKED(minibench::State& state) {
    const int n = state.range(0);
    const int batch = state.range(1);
    const int block_size = std::max(1, state.range(2));
    const Side side = parse_side(state.range(3));
    const Transpose trans = parse_transpose(state.range(4));

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    auto A = Matrix<T>::Random(n, n, /*hermitian=*/false, batch, /*seed=*/2026);
    UnifiedVector<T> tau_storage(static_cast<size_t>(n) * static_cast<size_t>(batch));

    // Build reflectors via QR once (outside timed region).
    size_t geqrf_ws = geqrf_buffer_size<B>(*q, A.view(), tau_storage.to_span());
    UnifiedVector<std::byte> ws_geqrf(geqrf_ws);
    geqrf<B>(*q, A.view(), tau_storage.to_span(), ws_geqrf.to_span()).wait();
    q->wait();

    // Apply to identity to avoid numerical blow-up across iterations.
    auto C = Matrix<T>::Identity(n, batch);

    size_t ws_bytes = ormqr_blocked_buffer_size<B, T>(*q,
                                                     A.view(),
                                                     C.view(),
                                                     side,
                                                     trans,
                                                     tau_storage.to_span(),
                                                     block_size);
    UnifiedVector<std::byte> ws(ws_bytes);

    // Approx flop model from LAPACK lawn18 appendix C (same as ormqr_benchmark.cc).
    const double m = double(n);
    const double nn = double(n);
    const double gflops_nominal = double(batch) * (1e-9 * (4.0 * m * nn * nn - 2.0 * nn * nn * nn + 3.0 * nn * nn));

    auto kernel = [q](auto& A,
                      auto& C,
                      auto& tau_storage,
                      auto& ws,
                      Side side,
                      Transpose trans,
                      int block_size) {
        return ormqr_blocked<B, T>(*q,
                                  A.view(),
                                  C.view(),
                                  side,
                                  trans,
                                  tau_storage.to_span(),
                                  ws.to_span(),
                                  block_size);
    };

    state.SetKernel(q,
                    std::move(A),
                    bench::pristine(std::move(C)),
                    std::move(tau_storage),
                    std::move(ws),
                    side,
                    trans,
                    block_size,
                    kernel);
    state.SetMetric("GFLOPS", gflops_nominal, minibench::Rate);
    state.SetMetric("Time (\u00b5s) / Batch", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_ORMQR_BLOCKED, OrmqrBlockedBenchSizes);

MINI_BENCHMARK_MAIN();
