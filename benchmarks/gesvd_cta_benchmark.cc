#include <batchlas/util/minibench.hh>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/extensions.hh>

#include "bench_utils.hh"

#include <cstddef>
#include <cstdint>
#include <memory>

using namespace batchlas;

namespace {

template <typename Benchmark>
inline void GesvdCtaBenchSizes(Benchmark* b) {
    // Args: n, batch, jobu_all, jobvh_all
    for (int n : {8, 16, 32}) {
        for (int bs : {1, 2, 4, 8, 16, 32, 64}) {
            for (int jobu : {0, 1}) {
                for (int jobvh : {0, 1}) {
                    b->Args({n, bs, jobu, jobvh});
                }
            }
        }
    }
}

inline SvdVectors parse_job(int v) {
    return (v == 0) ? SvdVectors::None : SvdVectors::All;
}

} // namespace

template <typename T, Backend B>
static void BM_GESVD_CTA(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const SvdVectors jobu = parse_job(static_cast<int>(state.range(2)));
    const SvdVectors jobvh = parse_job(static_cast<int>(state.range(3)));

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);

    auto A = Matrix<T>::Random(n, n, /*hermitian=*/false, batch);
    Matrix<T> U(n, n, batch);
    Matrix<T> Vh(n, n, batch);
    UnifiedVector<typename base_type<T>::type> s(n * batch);

    const size_t ws_size = gesvd_cta_buffer_size(*q,
                                                       A.view(),
                                                       s.to_span(),
                                                       U.view(),
                                                       Vh.view(),
                                                       jobu,
                                                       jobvh);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(s),
                    std::move(U),
                    std::move(Vh),
                    jobu,
                    jobvh,
                    std::move(workspace),
                    [](Queue& q, auto&&... xs) {
                        gesvd_cta(q, std::forward<decltype(xs)>(xs)...);
                    });

    state.SetMetric("Matrices/s", static_cast<double>(batch), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / static_cast<double>(batch)) * 1e6, minibench::Reciprocal);
}

BATCHLAS_BENCH_CUDA(BM_GESVD_CTA, GesvdCtaBenchSizes);

MINI_BENCHMARK_MAIN();
