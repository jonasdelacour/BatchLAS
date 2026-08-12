#include <batchlas/util/minibench.hh>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/extensions.hh>

#include "bench_utils.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

using namespace batchlas;

namespace {

// Args: m, n, batch, jobu, jobvh.
//
// m is separate from n so tall-skinny shapes are expressible at all. The body
// used to read a single size and build Random(n, n, ...), which made the whole
// m >> n regime -- the one the tall-skinny route exists for -- unmeasurable.
// The registered set stays square; positional CLI args override it wholesale
// (minibench.hh), so a sweep is `<m-list> <n> <batch> <jobu> <jobvh>`.
template <typename Benchmark>
inline void GesvdBlockedBenchSizes(Benchmark* b) {
    for (int n : {8, 16, 32, 64, 128, 256}) {
        for (int bs : {1, 2, 4, 8, 16, 32, 64}) {
            for (int jobu : {0, 1}) {
                for (int jobvh : {0, 1}) {
                    b->Args({n, n, bs, jobu, jobvh});
                }
            }
        }
    }
}

template <typename Benchmark>
inline void GesvdBlockedBenchSizesNetlib(Benchmark* b) {
    for (int n : {16, 32, 64, 128}) {
        for (int bs : {1, 2, 4, 8, 16}) {
            for (int jobu : {0, 1}) {
                for (int jobvh : {0, 1}) {
                    b->Args({n, n, bs, jobu, jobvh});
                }
            }
        }
    }
}

// 0 -> None, 1 -> All, 2 -> Thin. Thin is what makes a tall-skinny U
// expressible: at m >> n an All U is m x m and cannot be allocated.
inline SvdVectors parse_job(int v) {
    switch (v) {
        case 0:  return SvdVectors::None;
        case 2:  return SvdVectors::Thin;
        default: return SvdVectors::All;
    }
}

// When a factor is not requested its view is never written, so allocate the
// THIN extent rather than the full one. Sizing U as m x m under jobu=None costs
// 137 GB at m=16384/batch=128 and kills the process before the first timing --
// which is how the large-m arm of the tall-skinny sweep first failed.
inline int64_t u_cols_for(SvdVectors job, int64_t m, int64_t k) {
    return job == SvdVectors::None ? k : svd_u_cols(job, m, k);
}

inline int64_t vh_rows_for(SvdVectors job, int64_t n, int64_t k) {
    return job == SvdVectors::None ? k : svd_vh_rows(job, n, k);
}

} // namespace

template <typename T, Backend B>
static void BM_GESVD_BLOCKED(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t batch = state.range(2);
    const SvdVectors jobu = parse_job(static_cast<int>(state.range(3)));
    const SvdVectors jobvh = parse_job(static_cast<int>(state.range(4)));
    const size_t k = std::min(m, n);

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);

    auto A = Matrix<T>::Random(m, n, /*hermitian=*/false, batch);
    Matrix<T> U(m, u_cols_for(jobu, m, k), batch);
    Matrix<T> Vh(vh_rows_for(jobvh, n, k), n, batch);
    UnifiedVector<typename base_type<T>::type> s(k * batch);

    const size_t ws_size = gesvd_blocked_buffer_size(*q,
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
                        gesvd_blocked(q, std::forward<decltype(xs)>(xs)...);
                    });

    state.SetMetric("Matrices/s", static_cast<double>(batch), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / static_cast<double>(batch)) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK(BM_GESVD_BLOCKED, GesvdBlockedBenchSizes);

MINI_BENCHMARK_MAIN();
