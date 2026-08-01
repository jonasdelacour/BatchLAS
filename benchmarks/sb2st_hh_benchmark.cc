// Splits the Householder stage-2 cost: the chase (sytrd_sb2st_hh) versus the
// Q2 back-transform (unmqr_hb2st), so it is clear which one to optimise.

#include <util/minibench.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include "bench_utils.hh"

#include "../src/extensions/sytrd_sb2st_hh.hh"

#include <algorithm>
#include <vector>

using namespace batchlas;

namespace {

inline void Sizes(minibench::Benchmark* b) {
    b->Args({256, 1024, 16});
    b->Args({512, 512, 32});
    b->Args({1024, 128, 32});
    b->Args({1024, 256, 32});
}

} // namespace

template <typename T, Backend B>
static void BM_SB2ST_HH_CHASE(minibench::State& state) {
    using Real = typename base_type<T>::type;
    const int n = static_cast<int>(state.range(0));
    const int batch = static_cast<int>(state.range(1));
    const int kd = static_cast<int>(state.range(2));

    auto q = std::make_shared<Queue>("gpu");
    const int nrefl = std::max(1, internal::sb2st_hh_num_reflectors(n, kd));

    Matrix<T> ab = Matrix<T>::Random(kd + 1, n, false, batch);
    Matrix<T> ab_tri(2, n, batch);
    Matrix<T> vmat(kd, nrefl, batch);
    Vector<T> tau(nrefl, batch);
    Vector<Real> d(n, batch);
    Vector<Real> e(std::max(1, n - 1), batch);
    UnifiedVector<std::byte> ws(
        internal::sytrd_sb2st_hh_buffer_size<B, T>(*q, n, kd, batch));

    state.SetKernel(q, bench::pristine(ab), std::move(ab_tri), std::move(d),
                    std::move(e), std::move(vmat), std::move(tau), Uplo::Lower, kd,
                    std::move(ws),
                    [](Queue& qq, auto&&... xs) {
                        internal::sytrd_sb2st_hh<B, T>(qq, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("Time (ms)", 1.0, minibench::Reciprocal);
}

template <typename T, Backend B>
static void BM_SB2ST_HH_BACK(minibench::State& state) {
    const int n = static_cast<int>(state.range(0));
    const int batch = static_cast<int>(state.range(1));
    const int kd = static_cast<int>(state.range(2));

    auto q = std::make_shared<Queue>("gpu");
    const auto sched = internal::build_sb2st_hh_schedule(n, kd);
    const int nrefl = std::max<int>(1, static_cast<int>(sched.size()));

    UnifiedVector<int32_t> starts(sched.size());
    UnifiedVector<int32_t> lens(sched.size());
    for (size_t k = 0; k < sched.size(); ++k) {
        starts[k] = sched[k].start;
        lens[k] = sched[k].len;
    }

    Matrix<T> vmat = Matrix<T>::Random(kd, nrefl, false, batch);
    Vector<T> tau(nrefl, T(0.5), batch);
    Matrix<T> Z = Matrix<T>::Random(n, n, false, batch);

    // starts/lens are captured rather than passed as managed inputs: the bench
    // harness would try to instantiate Span<const int32_t> device helpers.
    const int32_t* sp = starts.data();
    const int32_t* lp = lens.data();
    const size_t nr = sched.size();
    state.SetKernel(q, std::move(vmat), std::move(tau), bench::pristine(Z), n, kd,
                    [sp, lp, nr](Queue& qq, auto&&... xs) {
                        internal::unmqr_hb2st<B, T>(qq, std::forward<decltype(xs)>(xs)...,
                                                    Span<const int32_t>(sp, nr),
                                                    Span<const int32_t>(lp, nr));
                    });
    state.SetMetric("Time (ms)", 1.0, minibench::Reciprocal);
}

#if BATCHLAS_HAS_CUDA_BACKEND
MINI_BENCHMARK_REGISTER_SIZES((BM_SB2ST_HH_CHASE<float, batchlas::Backend::CUDA>), Sizes);
MINI_BENCHMARK_REGISTER_SIZES((BM_SB2ST_HH_BACK<float, batchlas::Backend::CUDA>), Sizes);
#endif

MINI_BENCHMARK_MAIN();
